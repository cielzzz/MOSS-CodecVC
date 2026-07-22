#!/usr/bin/env python3
"""Batch-53 Fix-J step-100 functional gate.

This evaluator measures the actual decoder output, with the 64-frame prompt-zq
path enabled.  It reports:

* raw t=0 semantic and all-speaker-off advantages;
* raw conditional Euler ODE cosine;
* the contribution of each new speaker path (broadcast, prompt, FiLM);
* decoder velocity-delta speaker specificity for 8 trusted groups x 2 prompts.

The script intentionally does not use ``speaker_expand`` cosine as a proxy.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moss_codecvc.audio.zq_normalization import load_zq_channel_stats, sha256_file
from moss_codecvc.moss_codec import MossCodec
from scripts.ver3_1.evaluate_batch52_identifiability import (
    DEFAULT_ADAPTER,
    DEFAULT_CAMP_CHECKPOINT,
    DEFAULT_CAMP_REPO,
    DEFAULT_CODEC,
    DEFAULT_MOSS_ROOT,
    DEFAULT_STATS,
    DEFAULT_VALIDATION,
    campplus_embedding,
    decode_one,
    iter_jsonl,
    load_campplus,
    masked_square_sum,
    resolve_quantizer,
    select_rows,
)
from scripts.ver3_1.evaluate_ddlfm_validation import load_adapter
from scripts.ver3_1.infer_ddlfm_cfm import load_checkpoint
from scripts.ver3_1.train_content_adapter import load_feature
from scripts.ver3_1.train_ddlfm_cfm import load_embedding


DEFAULT_CHECKPOINT = (
    ROOT / "outputs/ver3_1_batch53_fix_j_sanity100_20260722/step-000100.infer.pt"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/ver3_1_batch53_fix_j_sanity100_20260722/functional_gate"
)
DEFAULT_TRAIN_INDEX = ROOT / "prepared/ddlfm_v2_cleaned_batch50/index.jsonl"
DEFAULT_CLUSTERS = ROOT / "prepared/cam_pp_trusted_index_v2_cleaned/clusters.jsonl"


@dataclass
class Condition:
    sample_id: str
    language: str
    target: torch.Tensor
    semantic: torch.Tensor
    speaker: torch.Tensor
    prompt: torch.Tensor
    prompt_mask: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--validation-jsonl", default=str(DEFAULT_VALIDATION))
    parser.add_argument("--train-index", default=str(DEFAULT_TRAIN_INDEX))
    parser.add_argument("--trusted-clusters", default=str(DEFAULT_CLUSTERS))
    parser.add_argument("--adapter-checkpoint", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--zq-channel-stats", default=str(DEFAULT_STATS))
    parser.add_argument("--codec-path", default=str(DEFAULT_CODEC))
    parser.add_argument("--moss-root", default=str(DEFAULT_MOSS_ROOT))
    parser.add_argument("--campplus-repo", default=str(DEFAULT_CAMP_REPO))
    parser.add_argument("--campplus-checkpoint", default=str(DEFAULT_CAMP_CHECKPOINT))
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--prompt-frames", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ode-steps", type=int, default=20)
    parser.add_argument("--specificity-speakers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("float32", "bf16"), default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def normalize_zq(value: torch.Tensor, stats: dict[str, Any]) -> torch.Tensor:
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32).view(1, -1)
    std = torch.as_tensor(stats["std"], dtype=torch.float32).view(1, -1)
    return ((value.float() - mean) / std).contiguous()


def crop_pad_prompt(
    value: torch.Tensor,
    frames: int,
    *,
    crop_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if value.ndim != 2 or int(value.shape[1]) != 768:
        raise ValueError(f"prompt zq must be [T,768], got {tuple(value.shape)}")
    length = int(value.shape[0])
    if length > frames:
        span = length - frames + 1
        offset = int(stable_rank(0, crop_key)[:12], 16) % span
        value = value[offset : offset + frames]
        valid = frames
    else:
        valid = length
        if length < frames:
            value = F.pad(value, (0, 0, 0, frames - length))
    mask = torch.zeros(frames, dtype=torch.bool)
    mask[:valid] = True
    return value.contiguous(), mask


@torch.inference_mode()
def prepare_conditions(
    rows: Sequence[dict[str, Any]],
    *,
    codec: MossCodec,
    adapter: torch.nn.Module,
    campplus: torch.nn.Module,
    stats: dict[str, Any],
    device: torch.device,
    max_frames: int,
    prompt_frames: int,
) -> list[Condition]:
    quantizer = resolve_quantizer(codec.model)
    conditions: list[Condition] = []
    for index, row in enumerate(rows):
        references = row.get("reference_audio_codes") or []
        if len(references) < 2 or not references[1]:
            raise ValueError(f"missing timbre reference codes: {row['sample_id']}")
        target = normalize_zq(decode_one(quantizer, row["audio_codes"], device), stats)
        prompt = normalize_zq(decode_one(quantizer, references[1], device), stats)
        prompt, prompt_mask = crop_pad_prompt(
            prompt, prompt_frames, crop_key=str(row["sample_id"])
        )
        feature = load_feature(str(row["source_wavlm_bnf_features_path"]))
        feature_mask = torch.ones((1, feature.shape[0]), dtype=torch.bool, device=device)
        state = adapter(feature.unsqueeze(0).to(device), feature_mask)
        semantic = state.semantic[0, state.semantic_mask[0]].detach().float().cpu()
        length = min(int(target.shape[0]), int(semantic.shape[0]), int(max_frames))
        if length <= 0:
            raise ValueError(f"empty aligned condition: {row['sample_id']}")
        meta = row.get("moss_codecvc_meta") or {}
        speaker = campplus_embedding(campplus, str(meta["timbre_ref_audio"]), device)
        conditions.append(
            Condition(
                sample_id=str(row["sample_id"]),
                language=str(row["language"]),
                target=target[:length].contiguous(),
                semantic=semantic[:length].contiguous(),
                speaker=speaker,
                prompt=prompt,
                prompt_mask=prompt_mask,
            )
        )
        if (index + 1) % 8 == 0 or index + 1 == len(rows):
            print(f"[batch53-gate] prepared={index + 1}/{len(rows)}", flush=True)
    return conditions


def pad_conditions(
    conditions: Sequence[Condition],
    shuffled: Sequence[Condition],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch = len(conditions)
    max_t = max(int(item.target.shape[0]) for item in conditions)
    max_s = max(
        max(int(a.semantic.shape[0]), int(b.semantic.shape[0]))
        for a, b in zip(conditions, shuffled)
    )
    prompt_frames = int(conditions[0].prompt.shape[0])
    target = torch.zeros(batch, max_t, 768, device=device)
    target_mask = torch.zeros(batch, max_t, dtype=torch.bool, device=device)
    semantic = torch.zeros(batch, max_s, 512, device=device)
    semantic_mask = torch.zeros(batch, max_s, dtype=torch.bool, device=device)
    shuffled_semantic = torch.zeros_like(semantic)
    shuffled_mask = torch.zeros_like(semantic_mask)
    prompt = torch.zeros(batch, prompt_frames, 768, device=device)
    prompt_mask = torch.zeros(batch, prompt_frames, dtype=torch.bool, device=device)
    speaker = torch.stack([item.speaker for item in conditions]).to(device)
    for index, (item, other) in enumerate(zip(conditions, shuffled)):
        t_len = int(item.target.shape[0])
        s_len = int(item.semantic.shape[0])
        o_len = int(other.semantic.shape[0])
        target[index, :t_len] = item.target.to(device)
        target_mask[index, :t_len] = True
        semantic[index, :s_len] = item.semantic.to(device)
        semantic_mask[index, :s_len] = True
        shuffled_semantic[index, :o_len] = other.semantic.to(device)
        shuffled_mask[index, :o_len] = True
        prompt[index] = item.prompt.to(device)
        prompt_mask[index] = item.prompt_mask.to(device)
    return {
        "target": target,
        "target_mask": target_mask,
        "semantic": semantic,
        "semantic_mask": semantic_mask,
        "shuffled_semantic": shuffled_semantic,
        "shuffled_mask": shuffled_mask,
        "speaker": speaker,
        "prompt": prompt,
        "prompt_mask": prompt_mask,
    }


def forward_variant(
    module: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    batch: dict[str, torch.Tensor],
    variant: str,
    *,
    shuffled_semantic: bool = False,
) -> torch.Tensor:
    semantic_key = "shuffled_semantic" if shuffled_semantic else "semantic"
    mask_key = "shuffled_mask" if shuffled_semantic else "semantic_mask"
    speaker = batch["speaker"]
    prompt = batch["prompt"]
    prompt_mask = batch["prompt_mask"]
    switches = {
        "enable_speaker_broadcast": variant != "no_broadcast",
        "enable_prompt_prefix": variant != "no_prompt",
        "enable_speaker_film": variant != "no_film",
    }
    if variant == "all_off":
        speaker = torch.zeros_like(speaker)
        prompt = torch.zeros_like(prompt)
        prompt_mask = torch.zeros_like(prompt_mask)
        switches = {
            "enable_speaker_broadcast": False,
            "enable_prompt_prefix": False,
            "enable_speaker_film": False,
        }
    modality = torch.zeros(len(speaker), dtype=torch.long, device=x.device)
    return module.decoder(
        x,
        t,
        batch[semantic_key],
        speaker,
        prompt_zq=prompt,
        prompt_mask=prompt_mask,
        target_mask=batch["target_mask"],
        semantic_mask=batch[mask_key],
        semantic_modality=modality,
        **switches,
    ).velocity


@torch.inference_mode()
def evaluate_identifiability(
    module: torch.nn.Module,
    conditions: Sequence[Condition],
    *,
    batch_size: int,
    ode_steps: int,
    seed: int,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    variants = ("full", "no_broadcast", "no_prompt", "no_film", "all_off")
    totals = {
        "target_energy": 0.0,
        "elements": 0,
        "shuffled_error": 0.0,
        "ode_cosine_sum": 0.0,
        "samples": 0,
        **{f"{name}_error": 0.0 for name in variants},
    }
    shuffled_all = list(conditions[1:]) + list(conditions[:1])
    amp_dtype = torch.bfloat16
    for start in range(0, len(conditions), batch_size):
        group = conditions[start : start + batch_size]
        shuffled = shuffled_all[start : start + len(group)]
        batch = pad_conditions(group, shuffled, device)
        generator = torch.Generator(device=device).manual_seed(seed + start)
        noise = torch.randn(batch["target"].shape, generator=generator, device=device)
        velocity_target = batch["target"] - noise
        t0 = torch.zeros(len(group), device=device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda" and precision == "bf16",
        ):
            predictions = {
                name: forward_variant(module, noise, t0, batch, name)
                for name in variants
            }
            shuffled_velocity = forward_variant(
                module, noise, t0, batch, "full", shuffled_semantic=True
            )
        for name, prediction in predictions.items():
            amount, _ = masked_square_sum(
                prediction.float() - velocity_target, batch["target_mask"]
            )
            totals[f"{name}_error"] += amount
        shuffled_amount, _ = masked_square_sum(
            shuffled_velocity.float() - velocity_target, batch["target_mask"]
        )
        totals["shuffled_error"] += shuffled_amount
        energy, elements = masked_square_sum(velocity_target, batch["target_mask"])
        totals["target_energy"] += energy
        totals["elements"] += elements

        x = noise.clone()
        for ode_index in range(ode_steps):
            t = torch.full(
                (len(group),), float(ode_index) / float(ode_steps), device=device
            )
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda" and precision == "bf16",
            ):
                velocity = forward_variant(module, x, t, batch, "full")
            x = (x + velocity.float() / float(ode_steps)).masked_fill(
                ~batch["target_mask"].unsqueeze(-1), 0.0
            )
        for index, item in enumerate(group):
            length = int(item.target.shape[0])
            cosine = F.cosine_similarity(
                x[index, :length].reshape(1, -1),
                batch["target"][index, :length].reshape(1, -1),
                dim=1,
            )
            totals["ode_cosine_sum"] += float(cosine.item())
            totals["samples"] += 1
        print(
            f"[batch53-gate] identifiability="
            f"{min(start + len(group), len(conditions))}/{len(conditions)}",
            flush=True,
        )
    denominator = max(float(totals["target_energy"]), 1.0e-12)
    full_error = float(totals["full_error"])
    path_advantages = {
        name: (float(totals[f"no_{name}_error"]) - full_error) / denominator
        for name in ("broadcast", "prompt", "film")
    }
    speaker_advantage = (float(totals["all_off_error"]) - full_error) / denominator
    independent = {
        name: value >= 0.005
        and speaker_advantage > 0.0
        and value / speaker_advantage >= 0.10
        for name, value in path_advantages.items()
    }
    elements = max(int(totals["elements"]), 1)
    return {
        "semantic_advantage_raw": (
            float(totals["shuffled_error"]) - full_error
        )
        / denominator,
        "speaker_advantage_raw": speaker_advantage,
        "free_ode_cosine": float(totals["ode_cosine_sum"])
        / max(int(totals["samples"]), 1),
        "path_ablation_advantage_raw": path_advantages,
        "path_independent_contribution": independent,
        "independent_path_count": sum(bool(value) for value in independent.values()),
        "mse": {
            "full": full_error / elements,
            "shuffled_semantic": float(totals["shuffled_error"]) / elements,
            "all_speaker_off": float(totals["all_off_error"]) / elements,
            **{
                f"no_{name}": float(totals[f"no_{name}_error"]) / elements
                for name in ("broadcast", "prompt", "film")
            },
            "target_velocity": denominator / elements,
        },
    }


def load_training_index(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["utterance_id"]): row
        for row in iter_jsonl(path)
        if row.get("utterance_id")
        and row.get("zq_path")
        and row.get("speaker_embedding_path")
    }


def select_specificity_groups(
    clusters_path: Path,
    index: dict[str, dict[str, Any]],
    count: int,
    seed: int,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for cluster in iter_jsonl(clusters_path):
        rows = [index[uid] for uid in cluster.get("utterance_ids", []) if uid in index]
        if len(rows) < 2:
            continue
        rows = sorted(
            rows,
            key=lambda row: stable_rank(seed + 1, str(row["utterance_id"])),
        )
        candidates.append((str(cluster["trusted_group_id"]), rows[0], rows[1]))
    candidates.sort(key=lambda value: stable_rank(seed, value[0]))
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} eligible trusted groups")
    return candidates[:count]


def load_prompt_from_index(
    row: dict[str, Any],
    stats: dict[str, Any],
    prompt_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    value = torch.from_numpy(
        np.load(str(row["zq_path"])).astype("float32", copy=False)
    ).transpose(0, 1)
    value = normalize_zq(value, stats)
    return crop_pad_prompt(
        value, prompt_frames, crop_key=str(row["utterance_id"])
    )


@torch.inference_mode()
def evaluate_specificity(
    module: torch.nn.Module,
    base: Condition,
    groups: Sequence[tuple[str, dict[str, Any], dict[str, Any]]],
    *,
    stats: dict[str, Any],
    prompt_frames: int,
    seed: int,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for group_id, first, second in groups:
        speaker = load_embedding(
            str(first["speaker_embedding_path"]), 192
        ).float()
        for prompt_index, prompt_row in enumerate((first, second), 1):
            prompt, prompt_mask = load_prompt_from_index(
                prompt_row, stats, prompt_frames
            )
            records.append(
                {
                    "group_id": group_id,
                    "prompt_index": prompt_index,
                    "utterance_id": str(prompt_row["utterance_id"]),
                    "speaker": speaker,
                    "prompt": prompt,
                    "prompt_mask": prompt_mask,
                }
            )
    batch_size = len(records)
    length = int(base.target.shape[0])
    semantic = base.semantic.unsqueeze(0).expand(batch_size, -1, -1).to(device)
    semantic_mask = torch.ones(
        batch_size, int(base.semantic.shape[0]), dtype=torch.bool, device=device
    )
    target_mask = torch.ones(batch_size, length, dtype=torch.bool, device=device)
    speakers = torch.stack([record["speaker"] for record in records]).to(device)
    prompts = torch.stack([record["prompt"] for record in records]).to(device)
    prompt_masks = torch.stack([record["prompt_mask"] for record in records]).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise_one = torch.randn((1, length, 768), generator=generator, device=device)
    noise = noise_one.expand(batch_size, -1, -1).contiguous()
    batch = {
        "target_mask": target_mask,
        "semantic": semantic,
        "semantic_mask": semantic_mask,
        "speaker": speakers,
        "prompt": prompts,
        "prompt_mask": prompt_masks,
    }
    t0 = torch.zeros(batch_size, device=device)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and precision == "bf16",
    ):
        full = forward_variant(module, noise, t0, batch, "full")
        all_off = forward_variant(module, noise, t0, batch, "all_off")
    delta = (full.float() - all_off.float()).reshape(batch_size, -1)
    delta = F.normalize(delta, dim=1)
    same_values: list[float] = []
    first_indices: list[int] = []
    for group_index in range(len(groups)):
        first_index = 2 * group_index
        first_indices.append(first_index)
        same_values.append(
            float((delta[first_index] * delta[first_index + 1]).sum().item())
        )
    cross_values = [
        float((delta[left] * delta[right]).sum().item())
        for left, right in itertools.combinations(first_indices, 2)
    ]
    same_mean = sum(same_values) / len(same_values)
    cross_mean = sum(cross_values) / len(cross_values)
    return {
        "representation": "flatten(decoder_velocity_full-decoder_velocity_all_speaker_off)",
        "fixed_noise_semantic_t": True,
        "speaker_groups": len(groups),
        "prompts": len(records),
        "same_speaker_cosines": same_values,
        "same_speaker_mean": same_mean,
        "same_speaker_min": min(same_values),
        "cross_speaker_cosines": cross_values,
        "cross_speaker_mean": cross_mean,
        "cross_speaker_max": max(cross_values),
        "same_cross_gap": same_mean - cross_mean,
        "records": [
            {
                "group_id": record["group_id"],
                "prompt_index": record["prompt_index"],
                "utterance_id": record["utterance_id"],
            }
            for record in records
        ],
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    result_path = output_dir / "functional_gate.json"
    report_path = output_dir / "functional_gate.md"
    if (result_path.exists() or report_path.exists()) and not args.overwrite:
        raise FileExistsError("functional gate output exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    validation = Path(args.validation_jsonl).expanduser().resolve()
    stats_path = Path(args.zq_channel_stats).expanduser().resolve()
    stats = load_zq_channel_stats(stats_path)
    selected = select_rows(validation, int(args.samples), int(args.seed))
    selected_manifest = output_dir / "heldout_selection.jsonl"
    selected_manifest.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": row["sample_id"],
                    "language": row["language"],
                    "validation_set": row["validation_set"],
                },
                ensure_ascii=False,
            )
            + "\n"
            for row in selected
        ),
        encoding="utf-8",
    )
    started = time.time()
    codec = MossCodec(
        args.codec_path,
        moss_root=args.moss_root,
        device=str(device),
        dtype="float32",
    )
    adapter, _ = load_adapter(
        Path(args.adapter_checkpoint).expanduser().resolve(), device
    )
    campplus = load_campplus(
        Path(args.campplus_repo).expanduser().resolve(),
        Path(args.campplus_checkpoint).expanduser().resolve(),
        device,
    )
    conditions = prepare_conditions(
        selected,
        codec=codec,
        adapter=adapter,
        campplus=campplus,
        stats=stats,
        device=device,
        max_frames=int(args.max_frames),
        prompt_frames=int(args.prompt_frames),
    )
    del codec, adapter, campplus
    torch.cuda.empty_cache()
    module, cfg = load_checkpoint(checkpoint, device, use_ema=True)
    identifiability = evaluate_identifiability(
        module,
        conditions,
        batch_size=int(args.batch_size),
        ode_steps=int(args.ode_steps),
        seed=int(args.seed) + 100,
        device=device,
        precision=str(args.precision),
    )
    train_index_path = Path(args.train_index).expanduser().resolve()
    clusters_path = Path(args.trusted_clusters).expanduser().resolve()
    train_index = load_training_index(train_index_path)
    groups = select_specificity_groups(
        clusters_path,
        train_index,
        int(args.specificity_speakers),
        int(args.seed),
    )
    specificity = evaluate_specificity(
        module,
        conditions[0],
        groups,
        stats=stats,
        prompt_frames=int(args.prompt_frames),
        seed=int(args.seed) + 200,
        device=device,
        precision=str(args.precision),
    )
    loss_finite = True
    train_log = checkpoint.parent / "train_log.jsonl"
    if train_log.exists():
        log_rows = list(iter_jsonl(train_log))
        loss_finite = bool(log_rows) and all(
            math.isfinite(float(row["loss"])) for row in log_rows
        )
    checks = {
        "speaker_advantage_gt_0_05": identifiability["speaker_advantage_raw"] > 0.05,
        "same_speaker_mean_gt_0_7": specificity["same_speaker_mean"] > 0.7,
        "cross_speaker_mean_lt_0_5": specificity["cross_speaker_mean"] < 0.5,
        "specificity_gap_gt_0_2": specificity["same_cross_gap"] > 0.2,
        "at_least_two_independent_paths": identifiability["independent_path_count"] >= 2,
        "loss_finite": loss_finite,
    }
    payload = {
        "schema": "ver3_1_batch53_fix_j_functional_gate_v1",
        "status": "completed",
        "created_at_utc": utc_now(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "weights": "ema",
        "checkpoint_config": {
            "prompt_zq_frames": cfg.get("prompt_zq_frames"),
            "zq_channel_stats_sha256": cfg.get("zq_channel_stats_sha256"),
        },
        "protocol": {
            "identifiability_samples": len(conditions),
            "languages": {
                language: sum(item.language == language for item in conditions)
                for language in ("zh", "en")
            },
            "raw_no_cfg": True,
            "ode_steps": int(args.ode_steps),
            "prompt_frames": int(args.prompt_frames),
            "path_independence": (
                "no-path advantage >=0.5 percentage point and >=10% of "
                "full all-speaker-off advantage"
            ),
        },
        "identifiability": identifiability,
        "specificity": specificity,
        "checks": checks,
        "pass": all(checks.values()),
        "elapsed_sec": round(time.time() - started, 3),
        "inputs": {
            "validation_jsonl": str(validation),
            "validation_sha256": sha256_file(validation),
            "selected_manifest": str(selected_manifest),
            "selected_manifest_sha256": sha256_file(selected_manifest),
            "zq_channel_stats": str(stats_path),
            "zq_channel_stats_sha256": sha256_file(stats_path),
            "train_index": str(train_index_path),
            "trusted_clusters": str(clusters_path),
        },
    }
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ablation = identifiability["path_ablation_advantage_raw"]
    independent = identifiability["path_independent_contribution"]
    lines = [
        "# Batch-53 Fix J functional gate",
        "",
        f"- Verdict: **{'PASS' if payload['pass'] else 'FAIL'}**",
        f"- Semantic advantage: {identifiability['semantic_advantage_raw']:.2%}",
        f"- Speaker cond-vs-zero: {identifiability['speaker_advantage_raw']:.2%}",
        f"- Free-ODE cosine: {identifiability['free_ode_cosine']:.4f}",
        f"- Same-speaker cosine mean/min: {specificity['same_speaker_mean']:.4f} / {specificity['same_speaker_min']:.4f}",
        f"- Cross-speaker cosine mean/max: {specificity['cross_speaker_mean']:.4f} / {specificity['cross_speaker_max']:.4f}",
        f"- Same-cross gap: {specificity['same_cross_gap']:.4f}",
        "",
        "## Three-path ablation",
        "",
        "| Path disabled | Raw advantage loss | Independent contribution |",
        "|---|---:|---|",
    ]
    for name in ("broadcast", "prompt", "film"):
        lines.append(
            f"| {name} | {ablation[name]:.2%} | "
            f"{'yes' if independent[name] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Independent paths: {identifiability['independent_path_count']}/3.",
            "",
            "## Gate checks",
            "",
            "```json",
            json.dumps(checks, ensure_ascii=False, indent=2),
            "```",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

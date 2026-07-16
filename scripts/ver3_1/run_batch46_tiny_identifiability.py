#!/usr/bin/env python3
"""Local real-data identifiability gate for the Batch-46 DDLFM retry.

This is deliberately a tiny memorization test, not a quality benchmark.  It
selects two no_text rows that share the same speaker sidecar but have different
semantic/target sequences, trains the small DDLFM smoke model, and checks that
semantic matching helps at the true inference endpoint ``t=0`` and during a
free Euler ODE rollout.  A different-speaker sidecar is used only for speaker
sensitivity diagnostics.

The gate must run locally before any H200 submission.  It never touches QZ.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moss_codecvc.audio.zq_normalization import load_zq_channel_stats, sha256_file
from scripts.ver3_1.train_ddlfm_cfm import (
    DDLFMTrainModule,
    ExponentialMovingAverage,
    apply_speaker_dropout,
    cfm_loss_weights,
    load_embedding,
    masked_mse,
    sample_cfm_time,
)
from scripts.ver3_1.infer_ddlfm_cfm import combine_cfg_velocity

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=str(ROOT / "prepared/ddlfm_v1_index.jsonl"))
    parser.add_argument(
        "--zq-channel-stats",
        default=str(ROOT / "prepared/zq_targets_v1/channel_stats.pt"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "testset/outputs/ver3_1_batch46_tiny_identifiability_20260716"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--gate-warmup-steps", type=int, default=100)
    parser.add_argument("--ode-steps", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                yield line_no, json.loads(line)


def select_rows(index: Path, frames: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_speaker: dict[str, dict[str, Any]] = {}
    different_speaker: dict[str, Any] | None = None
    selected: list[dict[str, Any]] | None = None
    selected_speaker = ""
    for line_no, row in iter_jsonl(index):
        if str(row.get("moss_codecvc_mode")) != "no_text":
            continue
        if int(row.get("zq_frames", 0)) < frames or int(row.get("semantic_frames", 0)) < frames:
            continue
        speaker_path = str(row.get("speaker_embedding_path") or "")
        if not speaker_path:
            continue
        row = dict(row)
        row["_index_line_no"] = line_no
        previous = by_speaker.get(speaker_path)
        if previous is None:
            by_speaker[speaker_path] = row
        elif str(previous.get("semantic_path")) != str(row.get("semantic_path")):
            selected = [previous, row]
            selected_speaker = speaker_path
            break
    if selected is None:
        raise RuntimeError("could not find two no_text rows sharing one speaker sidecar")
    for _, row in iter_jsonl(index):
        if str(row.get("moss_codecvc_mode")) != "no_text":
            continue
        speaker_path = str(row.get("speaker_embedding_path") or "")
        if speaker_path and speaker_path != selected_speaker:
            different_speaker = row
            break
    if different_speaker is None:
        raise RuntimeError("could not find a different-speaker diagnostic row")
    return selected, different_speaker


def load_pair(
    rows: list[dict[str, Any]],
    alternate_speaker_row: dict[str, Any],
    frames: int,
    stats: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    targets: list[torch.Tensor] = []
    semantics: list[torch.Tensor] = []
    speakers: list[torch.Tensor] = []
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32).view(1, -1)
    std = torch.as_tensor(stats["std"], dtype=torch.float32).view(1, -1)
    for row in rows:
        zq = torch.from_numpy(np.load(row["zq_path"]).astype("float32", copy=False)).transpose(0, 1)
        semantic = torch.from_numpy(np.load(row["semantic_path"]).astype("float32", copy=False))
        if zq.ndim != 2 or zq.shape[1] != 768:
            raise ValueError(f"invalid zq shape: {tuple(zq.shape)}")
        if semantic.ndim != 2 or semantic.shape[1] != 512:
            raise ValueError(f"invalid semantic shape: {tuple(semantic.shape)}")
        targets.append(((zq[:frames] - mean) / std).contiguous())
        semantics.append(semantic[:frames].contiguous())
        speakers.append(load_embedding(str(row["speaker_embedding_path"]), 192))
    alternate = load_embedding(str(alternate_speaker_row["speaker_embedding_path"]), 192)
    return (
        torch.stack(targets).to(device),
        torch.stack(semantics).to(device),
        torch.stack(speakers).to(device),
        alternate.unsqueeze(0).expand(2, -1).contiguous().to(device),
    )


def model_args() -> SimpleNamespace:
    return SimpleNamespace(
        latent_dim=768,
        semantic_dim=512,
        speaker_dim=192,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        ffn_size=3072,
        text_vocab_size=8001,
        text_padding_id=0,
        smoke_small_model=True,
        cross_gate_init=0.05,
    )


@torch.inference_mode()
def evaluate_identifiability(
    module: DDLFMTrainModule,
    target: torch.Tensor,
    semantic: torch.Tensor,
    speaker: torch.Tensor,
    alternate_speaker: torch.Tensor,
    *,
    ode_steps: int,
    cfg_scale: float,
    seed: int,
) -> dict[str, float]:
    module.eval()
    batch, frames, _ = target.shape
    generator = torch.Generator(device=target.device).manual_seed(int(seed))
    noise = torch.randn(target.shape, generator=generator, device=target.device, dtype=target.dtype)
    target_mask = torch.ones((batch, frames), dtype=torch.bool, device=target.device)
    semantic_mask = torch.ones((batch, frames), dtype=torch.bool, device=target.device)
    modality = torch.zeros(batch, dtype=torch.long, device=target.device)
    t0 = torch.zeros(batch, device=target.device)
    v_target = target - noise

    def conditional_velocity(
        x: torch.Tensor,
        t: torch.Tensor,
        sem: torch.Tensor,
        spk: torch.Tensor,
    ) -> torch.Tensor:
        return module.decoder(
            x,
            t,
            sem,
            spk,
            target_mask=target_mask,
            semantic_mask=semantic_mask,
            semantic_modality=modality,
        ).velocity

    def guided_velocity(
        x: torch.Tensor,
        t: torch.Tensor,
        sem: torch.Tensor,
        spk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditioned = conditional_velocity(x, t, sem, spk)
        unconditioned = conditional_velocity(x, t, sem, torch.zeros_like(spk))
        return combine_cfg_velocity(conditioned, unconditioned, float(cfg_scale)), unconditioned

    matched_velocity, matched_unconditioned = guided_velocity(noise, t0, semantic, speaker)
    shuffled_velocity, _ = guided_velocity(noise, t0, semantic.flip(0), speaker)
    permuted_velocity, _ = guided_velocity(noise, t0, semantic.flip(1), speaker)
    alternate_velocity, _ = guided_velocity(noise, t0, semantic, alternate_speaker)
    zero_velocity = matched_unconditioned

    matched_t0_mse = float(torch.mean((matched_velocity - v_target) ** 2).item())
    shuffled_t0_mse = float(torch.mean((shuffled_velocity - v_target) ** 2).item())

    target_velocity_rms = torch.mean(v_target**2).sqrt().clamp_min(1.0e-6)

    def delta_metrics(other: torch.Tensor) -> tuple[float, float]:
        numerator = torch.mean((other - matched_velocity) ** 2).sqrt()
        return float(numerator.item()), float((numerator / target_velocity_rms).item())

    def integrate(sem: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        x = noise.clone()
        for step in range(int(ode_steps)):
            t = torch.full((batch,), float(step) / float(ode_steps), device=target.device)
            velocity, _ = guided_velocity(x, t, sem, spk)
            x = x + velocity / float(ode_steps)
        return x

    matched_ode = integrate(semantic, speaker)
    shuffled_ode = integrate(semantic.flip(0), speaker)
    matched_ode_mse = float(torch.mean((matched_ode - target) ** 2).item())
    shuffled_ode_mse = float(torch.mean((shuffled_ode - target) ** 2).item())
    matched_cosine = float(
        torch.nn.functional.cosine_similarity(matched_ode.flatten(1), target.flatten(1), dim=1).mean().item()
    )
    shuffled_cosine = float(
        torch.nn.functional.cosine_similarity(shuffled_ode.flatten(1), target.flatten(1), dim=1).mean().item()
    )
    permutation_abs, permutation_rel = delta_metrics(permuted_velocity)
    alternate_abs, alternate_rel = delta_metrics(alternate_velocity)
    zero_abs, zero_rel = delta_metrics(zero_velocity)
    return {
        "cfg_scale": float(cfg_scale),
        "matched_t0_mse": matched_t0_mse,
        "shuffled_t0_mse": shuffled_t0_mse,
        "matched_t0_advantage": (shuffled_t0_mse - matched_t0_mse) / max(shuffled_t0_mse, 1.0e-8),
        "semantic_permutation_absolute_rms": permutation_abs,
        "semantic_permutation_relative_to_target_rms": permutation_rel,
        "alternate_speaker_absolute_rms": alternate_abs,
        "alternate_speaker_relative_to_target_rms": alternate_rel,
        "zero_speaker_absolute_rms": zero_abs,
        "zero_speaker_relative_to_target_rms": zero_rel,
        "matched_ode_mse": matched_ode_mse,
        "shuffled_ode_mse": shuffled_ode_mse,
        "matched_ode_advantage": (shuffled_ode_mse - matched_ode_mse) / max(shuffled_ode_mse, 1.0e-8),
        "matched_ode_cosine": matched_cosine,
        "shuffled_ode_cosine": shuffled_cosine,
        "ode_cosine_advantage": matched_cosine - shuffled_cosine,
    }


def main() -> int:
    args = parse_args()
    if args.steps < 100 or args.frames <= 0 or args.ode_steps <= 0:
        raise ValueError("steps must be >=100; frames and ode-steps must be positive")
    if not math.isfinite(float(args.cfg_scale)) or float(args.cfg_scale) < 0.0:
        raise ValueError("cfg-scale must be finite and non-negative")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing non-empty output dir without --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    stats_path = Path(args.zq_channel_stats).expanduser().resolve()
    stats = load_zq_channel_stats(stats_path)
    if stats.get("status") != "completed" or bool(stats.get("partial", False)):
        raise ValueError(f"refusing incomplete channel stats: {stats_path}")
    selected, alternate_row = select_rows(Path(args.index).expanduser().resolve(), int(args.frames))
    target, semantic, speaker, alternate_speaker = load_pair(
        selected,
        alternate_row,
        int(args.frames),
        stats,
        device,
    )

    module = DDLFMTrainModule(model_args()).to(device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=float(args.lr), betas=(0.9, 0.95), weight_decay=0.01)
    ema = ExponentialMovingAverage(module, 0.9999, warmup=True)
    target_mask = torch.ones(target.shape[:2], dtype=torch.bool, device=device)
    semantic_mask = torch.ones(semantic.shape[:2], dtype=torch.bool, device=device)
    modality = torch.zeros(target.shape[0], dtype=torch.long, device=device)
    losses: list[float] = []
    started = time.time()
    module.train()
    for step in range(1, int(args.steps) + 1):
        lr_scale = min(1.0, float(step) / max(1, int(args.warmup_steps)))
        for group in optimizer.param_groups:
            group["lr"] = float(args.lr) * lr_scale
        gate_progress = min(1.0, float(step) / max(1, int(args.gate_warmup_steps)))
        gate_scale = 0.05 + 0.95 * gate_progress
        t = sample_cfm_time(target.shape[0], device=device, schedule="logit_normal")
        noise = torch.randn_like(target)
        x_t = (1.0 - t[:, None, None]) * noise + t[:, None, None] * target
        v_target = target - noise
        speaker_input, _ = apply_speaker_dropout(speaker, 0.10)
        weights = cfm_loss_weights(t, mode="low_t", eps=0.05, cap=5.0)
        prediction = module(
            x_t,
            t,
            semantic,
            speaker_input,
            target_mask=target_mask,
            semantic_mask=semantic_mask,
            semantic_modality=modality,
            condition_gate_scale=gate_scale,
        ).velocity
        loss = masked_mse(prediction, v_target, target_mask, sample_weight=weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite tiny-gate loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        optimizer.step()
        ema.update(module)
        losses.append(float(loss.detach().cpu().item()))
        if step == 1 or step % 100 == 0:
            print(json.dumps({"step": step, "loss": losses[-1], "ema_decay": ema.last_decay}), flush=True)

    raw_metrics = evaluate_identifiability(
        module,
        target,
        semantic,
        speaker,
        alternate_speaker,
        ode_steps=int(args.ode_steps),
        cfg_scale=float(args.cfg_scale),
        seed=int(args.seed) + 1,
    )
    ema_module = DDLFMTrainModule(model_args()).to(device)
    ema_module.load_state_dict(ema.state_dict_cpu(), strict=True)
    ema_metrics = evaluate_identifiability(
        ema_module,
        target,
        semantic,
        speaker,
        alternate_speaker,
        ode_steps=int(args.ode_steps),
        cfg_scale=float(args.cfg_scale),
        seed=int(args.seed) + 1,
    )
    first_window = float(np.mean(losses[: min(50, len(losses))]))
    last_window = float(np.mean(losses[-min(50, len(losses)) :]))
    gates = {
        "loss_finite_and_decreasing": math.isfinite(last_window) and last_window < 0.70 * first_window,
        "raw_t0_semantic_advantage_ge_10pct": raw_metrics["matched_t0_advantage"] >= 0.10,
        "raw_semantic_permutation_sensitivity_ge_1pct": raw_metrics["semantic_permutation_relative_to_target_rms"] >= 0.01,
        "raw_cond_vs_zero_speaker_sensitivity_ge_0_5pct": raw_metrics["zero_speaker_relative_to_target_rms"] >= 0.005,
        "raw_free_ode_semantic_advantage_positive": raw_metrics["matched_ode_advantage"] > 0.0,
        "ema_t0_semantic_advantage_ge_5pct": ema_metrics["matched_t0_advantage"] >= 0.05,
        "ema_semantic_permutation_sensitivity_ge_0_5pct": ema_metrics["semantic_permutation_relative_to_target_rms"] >= 0.005,
        "ema_cond_vs_zero_speaker_sensitivity_positive": ema_metrics["zero_speaker_relative_to_target_rms"] > 0.0,
        "ema_free_ode_semantic_advantage_positive": ema_metrics["matched_ode_advantage"] > 0.0,
    }
    report = {
        "schema": "ver3_1_batch46_tiny_identifiability_v1",
        "status": "passed" if all(gates.values()) else "failed",
        "purpose": "local structural identifiability gate; not a quality benchmark",
        "device": str(device),
        "steps": int(args.steps),
        "frames": int(args.frames),
        "ode_steps": int(args.ode_steps),
        "cfg_scale": float(args.cfg_scale),
        "zq_channel_stats": str(stats_path),
        "zq_channel_stats_sha256": sha256_file(stats_path),
        "elapsed_sec": round(time.time() - started, 3),
        "selected_rows": [
            {
                "index_line_no": row.get("_index_line_no"),
                "utterance_id": row.get("utterance_id"),
                "speaker_embedding_path": row.get("speaker_embedding_path"),
                "semantic_path": row.get("semantic_path"),
                "zq_path": row.get("zq_path"),
            }
            for row in selected
        ],
        "loss": {
            "first_window_mean": first_window,
            "last_window_mean": last_window,
            "ratio": last_window / max(first_window, 1.0e-8),
        },
        "raw": raw_metrics,
        "ema": ema_metrics,
        "ema_metadata": {
            "target_decay": ema.decay,
            "effective_decay": ema.last_decay,
            "num_updates": ema.num_updates,
        },
        "gates": gates,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "model": module.state_dict(),
            "ema_model": ema.state_dict_cpu(),
            "report": report,
        },
        output_dir / "tiny_model.pt",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not all(gates.values()):
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

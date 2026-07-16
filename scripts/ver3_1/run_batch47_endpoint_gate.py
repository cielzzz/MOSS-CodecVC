#!/usr/bin/env python3
"""Local real-data endpoint identifiability gate for Batch-47.

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
    apply_semantic_dropout,
    apply_speaker_dropout,
    cfm_loss_weights,
    estimate_cfm_weight_reference,
    load_embedding,
    masked_mse,
    sample_cfm_time,
)
from scripts.ver3_1.infer_ddlfm_cfm import (
    combine_cfg_velocity,
    combine_dual_cfg_velocity,
    load_checkpoint,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=str(ROOT / "prepared/ddlfm_v1_index.jsonl"))
    parser.add_argument(
        "--zq-channel-stats",
        default=str(ROOT / "prepared/zq_targets_v1/channel_stats.pt"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "testset/outputs/ver3_1_batch47_endpoint_gate_20260716"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--gate-warmup-steps", type=int, default=100)
    parser.add_argument("--ode-steps", type=int, default=20)
    parser.add_argument("--speaker-cfg-scale", type=float, default=2.5)
    parser.add_argument("--semantic-cfg-scale", type=float, default=2.0)
    parser.add_argument("--speaker-dropout", type=float, default=0.25)
    parser.add_argument("--semantic-dropout", type=float, default=0.15)
    parser.add_argument("--aux-loss-weight", type=float, default=1.0)
    parser.add_argument("--aux-warmup-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--checkpoint", default="", help="Evaluate an existing inference checkpoint instead of training")
    parser.add_argument("--weights", choices=("raw", "ema"), default="ema")
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
        num_speaker_prompt_tokens=4,
        speaker_condition_scale=4.0,
        speaker_input_scale=1.0,
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
    speaker_cfg_scale: float,
    semantic_cfg_scale: float,
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
        sem_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return module.decoder(
            x,
            t,
            sem,
            spk,
            target_mask=target_mask,
            semantic_mask=semantic_mask if sem_mask is None else sem_mask,
            semantic_modality=modality,
        ).velocity

    zero_semantic = torch.zeros_like(semantic)
    zero_semantic_mask = torch.zeros_like(semantic_mask)

    def raw_velocity(
        x: torch.Tensor,
        t: torch.Tensor,
        sem: torch.Tensor,
        spk: torch.Tensor,
        sem_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Unconditional-free conditional forward, with no CFG mixing."""

        return conditional_velocity(x, t, sem, spk, sem_mask)

    def guided_velocity(
        x: torch.Tensor,
        t: torch.Tensor,
        sem: torch.Tensor,
        spk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Batch-48 uses the standard four-state additive dual-CFG contract:
        # v00 + lambda_s (v10-v00) + lambda_c (v01-v00).  Keep v00 as the
        # returned second value so endpoint diagnostics do not mistake a
        # semantic-only state for the unconditional baseline.
        unconditional = conditional_velocity(
            x, t, zero_semantic, torch.zeros_like(spk), zero_semantic_mask
        )
        speaker_only = conditional_velocity(x, t, zero_semantic, spk, zero_semantic_mask)
        semantic_only = conditional_velocity(x, t, sem, torch.zeros_like(spk))
        guided = combine_dual_cfg_velocity(
            unconditional,
            speaker_only,
            semantic_only,
            float(speaker_cfg_scale),
            float(semantic_cfg_scale),
        )
        return guided, unconditional

    # Batch-48 gates are deliberately raw: no CFG scale is allowed to make a
    # weak speaker/semantic path look stronger.  The guided values below are
    # retained only as a diagnostic reference.
    raw_matched_velocity = raw_velocity(noise, t0, semantic, speaker)
    raw_shuffled_velocity = raw_velocity(noise, t0, semantic.flip(0), speaker)
    raw_permuted_velocity = raw_velocity(noise, t0, semantic.flip(1), speaker)
    raw_alternate_velocity = raw_velocity(noise, t0, semantic, alternate_speaker)
    raw_zero_speaker = raw_velocity(noise, t0, semantic, torch.zeros_like(speaker))
    cfg_matched_velocity, cfg_unconditioned = guided_velocity(noise, t0, semantic, speaker)
    cfg_shuffled_velocity, _ = guided_velocity(noise, t0, semantic.flip(0), speaker)
    cfg_permuted_velocity, _ = guided_velocity(noise, t0, semantic.flip(1), speaker)
    cfg_alternate_velocity, _ = guided_velocity(noise, t0, semantic, alternate_speaker)

    target_velocity_rms = torch.mean(v_target**2).sqrt().clamp_min(1.0e-6)

    def delta_metrics(other: torch.Tensor, matched: torch.Tensor) -> tuple[float, float]:
        numerator = torch.mean((other - matched) ** 2).sqrt()
        return float(numerator.item()), float((numerator / target_velocity_rms).item())

    def integrate_raw(sem: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        x = noise.clone()
        for step in range(int(ode_steps)):
            t = torch.full((batch,), float(step) / float(ode_steps), device=target.device)
            velocity = raw_velocity(x, t, sem, spk)
            x = x + velocity / float(ode_steps)
        return x

    raw_matched_ode = integrate_raw(semantic, speaker)
    raw_shuffled_ode = integrate_raw(semantic.flip(0), speaker)
    raw_matched_ode_mse = float(torch.mean((raw_matched_ode - target) ** 2).item())
    raw_shuffled_ode_mse = float(torch.mean((raw_shuffled_ode - target) ** 2).item())
    raw_matched_cosine = float(
        torch.nn.functional.cosine_similarity(raw_matched_ode.flatten(1), target.flatten(1), dim=1).mean().item()
    )
    raw_shuffled_cosine = float(
        torch.nn.functional.cosine_similarity(raw_shuffled_ode.flatten(1), target.flatten(1), dim=1).mean().item()
    )
    raw_matched_t0_mse = float(torch.mean((raw_matched_velocity - v_target) ** 2).item())
    raw_shuffled_t0_mse = float(torch.mean((raw_shuffled_velocity - v_target) ** 2).item())
    raw_permutation_abs, raw_permutation_rel = delta_metrics(raw_permuted_velocity, raw_matched_velocity)
    raw_alternate_abs, raw_alternate_rel = delta_metrics(raw_alternate_velocity, raw_matched_velocity)
    raw_zero_abs, raw_zero_rel = delta_metrics(raw_zero_speaker, raw_matched_velocity)
    speaker_raw_abs = raw_zero_abs
    speaker_raw_rel = raw_zero_rel
    cfg_permutation_abs, cfg_permutation_rel = delta_metrics(cfg_permuted_velocity, cfg_matched_velocity)
    cfg_alternate_abs, cfg_alternate_rel = delta_metrics(cfg_alternate_velocity, cfg_matched_velocity)
    cfg_zero_abs, cfg_zero_rel = delta_metrics(cfg_unconditioned, cfg_matched_velocity)
    cfg_matched_t0_mse = float(torch.mean((cfg_matched_velocity - v_target) ** 2).item())
    cfg_shuffled_t0_mse = float(torch.mean((cfg_shuffled_velocity - v_target) ** 2).item())
    return {
        "speaker_cfg_scale": float(speaker_cfg_scale),
        "semantic_cfg_scale": float(semantic_cfg_scale),
        "cfg_formula": "four_state_v00_v10_v01",
        "raw_matched_t0_mse": raw_matched_t0_mse,
        "raw_shuffled_t0_mse": raw_shuffled_t0_mse,
        "raw_matched_t0_advantage": (raw_shuffled_t0_mse - raw_matched_t0_mse) / max(raw_shuffled_t0_mse, 1.0e-8),
        "raw_semantic_permutation_absolute_rms": raw_permutation_abs,
        "raw_semantic_permutation_relative_to_target_rms": raw_permutation_rel,
        "raw_alternate_speaker_absolute_rms": raw_alternate_abs,
        "raw_alternate_speaker_relative_to_target_rms": raw_alternate_rel,
        "raw_zero_speaker_absolute_rms": raw_zero_abs,
        "raw_zero_speaker_relative_to_target_rms": raw_zero_rel,
        "raw_speaker_cond_vs_zero_relative_to_target_rms": speaker_raw_rel,
        "raw_matched_ode_mse": raw_matched_ode_mse,
        "raw_shuffled_ode_mse": raw_shuffled_ode_mse,
        "raw_matched_ode_advantage": (raw_shuffled_ode_mse - raw_matched_ode_mse) / max(raw_shuffled_ode_mse, 1.0e-8),
        "raw_matched_ode_cosine": raw_matched_cosine,
        "raw_shuffled_ode_cosine": raw_shuffled_cosine,
        "raw_ode_cosine_advantage": raw_matched_cosine - raw_shuffled_cosine,
        # Backward-compatible aliases now intentionally mean raw metrics.
        "matched_t0_mse": raw_matched_t0_mse,
        "shuffled_t0_mse": raw_shuffled_t0_mse,
        "matched_t0_advantage": (raw_shuffled_t0_mse - raw_matched_t0_mse) / max(raw_shuffled_t0_mse, 1.0e-8),
        "semantic_permutation_absolute_rms": raw_permutation_abs,
        "semantic_permutation_relative_to_target_rms": raw_permutation_rel,
        "alternate_speaker_absolute_rms": raw_alternate_abs,
        "alternate_speaker_relative_to_target_rms": raw_alternate_rel,
        "zero_speaker_absolute_rms": raw_zero_abs,
        "zero_speaker_relative_to_target_rms": raw_zero_rel,
        "speaker_advantage_relative_to_target_rms": speaker_raw_rel,
        "speaker_raw_cond_vs_zero_relative_to_target_rms": speaker_raw_rel,
        "speaker_cfg_advantage_relative_to_target_rms": cfg_zero_rel,
        "cfg_matched_t0_mse": cfg_matched_t0_mse,
        "cfg_shuffled_t0_mse": cfg_shuffled_t0_mse,
        "cfg_matched_t0_advantage": (cfg_shuffled_t0_mse - cfg_matched_t0_mse) / max(cfg_shuffled_t0_mse, 1.0e-8),
        "cfg_semantic_permutation_relative_to_target_rms": cfg_permutation_rel,
        "cfg_alternate_speaker_relative_to_target_rms": cfg_alternate_rel,
        "cfg_zero_speaker_relative_to_target_rms": cfg_zero_rel,
        "matched_ode_mse": raw_matched_ode_mse,
        "shuffled_ode_mse": raw_shuffled_ode_mse,
        "matched_ode_advantage": (raw_shuffled_ode_mse - raw_matched_ode_mse) / max(raw_shuffled_ode_mse, 1.0e-8),
        "matched_ode_cosine": raw_matched_cosine,
        "shuffled_ode_cosine": raw_shuffled_cosine,
        "ode_cosine_advantage": raw_matched_cosine - raw_shuffled_cosine,
    }


def main() -> int:
    args = parse_args()
    if args.steps < 100 or args.frames <= 0 or args.ode_steps <= 0:
        raise ValueError("steps must be >=100; frames and ode-steps must be positive")
    if (
        not math.isfinite(float(args.speaker_cfg_scale))
        or float(args.speaker_cfg_scale) < 0.0
        or not math.isfinite(float(args.semantic_cfg_scale))
        or float(args.semantic_cfg_scale) < 0.0
    ):
        raise ValueError("CFG scales must be finite and non-negative")
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

    if str(args.checkpoint):
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
        module, checkpoint_cfg = load_checkpoint(
            checkpoint_path,
            device,
            use_ema=str(args.weights) == "ema",
        )
        metrics = evaluate_identifiability(
            module,
            target,
            semantic,
            speaker,
            alternate_speaker,
            ode_steps=int(args.ode_steps),
            speaker_cfg_scale=float(args.speaker_cfg_scale),
            semantic_cfg_scale=float(args.semantic_cfg_scale),
            seed=int(args.seed) + 1,
        )
        report = {
            "schema": "ver3_1_batch48_identifiability_v1",
            "status": "completed",
            "purpose": "fixed real-data endpoint identifiability diagnostic; no training",
            "cfg_formula": "four_state_additive_v1",
            "checkpoint": str(checkpoint_path),
            "weights": str(args.weights),
            "device": str(device),
            "steps": int(args.ode_steps),
            "speaker_cfg_scale": float(args.speaker_cfg_scale),
            "semantic_cfg_scale": float(args.semantic_cfg_scale),
            "metrics": metrics,
            "checkpoint_config": {
                "speaker_condition_scale": checkpoint_cfg.get("speaker_condition_scale"),
                "speaker_input_scale": checkpoint_cfg.get("speaker_input_scale"),
                "num_speaker_prompt_tokens": checkpoint_cfg.get("num_speaker_prompt_tokens"),
            },
        }
        (output_dir / "checkpoint_identifiability.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0

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
        t = sample_cfm_time(
            target.shape[0],
            device=device,
            schedule="shift_low",
            shift_power=4.0,
        )
        noise = torch.randn_like(target)
        x_t = (1.0 - t[:, None, None]) * noise + t[:, None, None] * target
        v_target = target - noise
        semantic_input, semantic_mask_input, _ = apply_semantic_dropout(
            semantic,
            semantic_mask,
            float(args.semantic_dropout),
        )
        speaker_input, _ = apply_speaker_dropout(speaker, float(args.speaker_dropout))
        weights = cfm_loss_weights(
            t,
            mode="low_t",
            eps=0.02,
            cap=25.0,
            normalize=False,
        )
        weights = weights / estimate_cfm_weight_reference(
            schedule="shift_low", eps=0.02, cap=25.0, shift_power=4.0
        )
        prediction = module(
            x_t,
            t,
            semantic_input,
            speaker_input,
            target_mask=target_mask,
            semantic_mask=semantic_mask_input,
            semantic_modality=modality,
            condition_gate_scale=gate_scale,
        ).velocity
        cfm = masked_mse(prediction, v_target, target_mask, sample_weight=weights, normalize_sample_weight=False)
        aux = module(
            noise,
            torch.zeros_like(t),
            semantic,
            speaker,
            target_mask=target_mask,
            semantic_mask=semantic_mask,
            semantic_modality=modality,
            condition_gate_scale=gate_scale,
        ).velocity
        aux_loss = masked_mse(aux, v_target, target_mask)
        aux_weight = float(args.aux_loss_weight) * min(1.0, float(step) / max(1, int(args.aux_warmup_steps)))
        loss = cfm + aux_weight * aux_loss
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
        speaker_cfg_scale=float(args.speaker_cfg_scale),
        semantic_cfg_scale=float(args.semantic_cfg_scale),
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
        speaker_cfg_scale=float(args.speaker_cfg_scale),
        semantic_cfg_scale=float(args.semantic_cfg_scale),
        seed=int(args.seed) + 1,
    )
    first_window = float(np.mean(losses[: min(50, len(losses))]))
    last_window = float(np.mean(losses[-min(50, len(losses)) :]))
    gates = {
        "loss_finite_and_bounded": math.isfinite(last_window) and last_window < 1.50 * first_window,
        "raw_t0_semantic_advantage_ge_10pct": raw_metrics["raw_matched_t0_advantage"] >= 0.10,
        "raw_semantic_permutation_sensitivity_ge_1pct": raw_metrics["raw_semantic_permutation_relative_to_target_rms"] >= 0.01,
        "raw_cond_vs_zero_speaker_sensitivity_ge_15pct": raw_metrics["raw_speaker_cond_vs_zero_relative_to_target_rms"] >= 0.15,
        "raw_free_ode_semantic_advantage_positive": raw_metrics["raw_matched_ode_advantage"] > 0.0,
        "ema_t0_semantic_advantage_ge_5pct": ema_metrics["raw_matched_t0_advantage"] >= 0.05,
        "ema_semantic_permutation_sensitivity_ge_0_5pct": ema_metrics["raw_semantic_permutation_relative_to_target_rms"] >= 0.005,
        "ema_cond_vs_zero_speaker_sensitivity_ge_15pct": ema_metrics["raw_speaker_cond_vs_zero_relative_to_target_rms"] >= 0.15,
        "ema_free_ode_semantic_advantage_positive": ema_metrics["raw_matched_ode_advantage"] > 0.0,
    }
    report = {
        "schema": "ver3_1_batch48_identifiability_v1",
        "status": "passed" if all(gates.values()) else "failed",
        "purpose": "local structural identifiability gate; not a quality benchmark",
        "cfg_formula": "four_state_additive_v1",
        "device": str(device),
        "steps": int(args.steps),
        "frames": int(args.frames),
        "ode_steps": int(args.ode_steps),
        "speaker_cfg_scale": float(args.speaker_cfg_scale),
        "semantic_cfg_scale": float(args.semantic_cfg_scale),
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

#!/usr/bin/env python
"""Euler inference for a ver3.1 DDLFM checkpoint.

This is a small, auditable single-example runner used by the later quick20
and full320 wrappers.  It predicts zq in ``[B,T,768]`` and calls the frozen
MOSS decoder through ``decode_latents``; no codebook quantization is performed
at inference time.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moss_codecvc.audio import (
    decode_latents,
    denormalize_zq,
    load_zq_channel_stats,
    sha256_file,
)
from moss_codecvc.moss_codec import MossCodec
from moss_codecvc.models.source_semantic_memory import SourceTokenMemoryEncoder
from scripts.ver3_1.train_ddlfm_cfm import DDLFMTrainModule, load_embedding


def combine_cfg_velocity(
    velocity_cond: torch.Tensor,
    velocity_uncond: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Combine conditional and zero-speaker velocities for speaker CFG."""

    if velocity_cond.shape != velocity_uncond.shape:
        raise ValueError("conditional and unconditional velocity shapes must match")
    if not math.isfinite(float(scale)) or float(scale) < 0.0:
        raise ValueError("cfg_scale must be finite and non-negative")
    return velocity_uncond + float(scale) * (velocity_cond - velocity_uncond)


def combine_dual_cfg_velocity(
    velocity_cond: torch.Tensor,
    velocity_speaker: torch.Tensor,
    velocity_semantic: torch.Tensor,
    speaker_scale: float,
    semantic_scale: float,
) -> torch.Tensor:
    """Independent CFG anchored at the fully conditioned velocity.

    ``velocity_cond`` is v11 (speaker + semantic), ``velocity_speaker`` is
    v10 (speaker only) and ``velocity_semantic`` is v01 (semantic only).  The
    anchored form is identity-preserving at scale (1,1): it returns v11 rather
    than the additive ``v00 + ...`` approximation, which would silently drop
    the learned interaction term.
    """

    if not (velocity_cond.shape == velocity_speaker.shape == velocity_semantic.shape):
        raise ValueError("dual CFG velocity tensors must have identical shapes")
    for name, value in (("speaker_scale", speaker_scale), ("semantic_scale", semantic_scale)):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    return (
        velocity_cond
        + (float(speaker_scale) - 1.0) * (velocity_cond - velocity_semantic)
        + (float(semantic_scale) - 1.0) * (velocity_cond - velocity_speaker)
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-wav", required=True)
    ap.add_argument("--mode", choices=("no_text", "text"), required=True)
    ap.add_argument("--semantic-path", default="")
    ap.add_argument("--text", default="")
    ap.add_argument(
        "--tokenizer-model",
        default=str(ROOT / "trainset/shared_content_tokenizers/spm_multilingual_byte_fallback_v1.model"),
    )
    ap.add_argument("--speaker-embedding", required=True)
    ap.add_argument("--target-frames", type=int, default=0)
    ap.add_argument("--sampling-steps", type=int, default=20)
    ap.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="Default: 1.5 for CFG-trained checkpoints, otherwise 1.0 for legacy checkpoints.",
    )
    ap.add_argument(
        "--speaker-cfg-scale",
        type=float,
        default=None,
        help="Batch-47 speaker CFG; defaults to checkpoint speaker_cfg_scale/cfg_scale.",
    )
    ap.add_argument(
        "--semantic-cfg-scale",
        type=float,
        default=None,
        help="Batch-47 semantic CFG; zero preserves legacy speaker-only CFG.",
    )
    ap.add_argument(
        "--zq-channel-stats",
        default="",
        help="Override channel_stats.pt; default comes from the checkpoint config.",
    )
    ap.add_argument(
        "--no-ema",
        action="store_true",
        help="Use raw training weights instead of the Batch-46 EMA weights.",
    )
    ap.add_argument("--seed", type=int, default=20260715)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--codec-path", default="")
    ap.add_argument("--moss-root", default="")
    ap.add_argument("--codec-dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).replace("\t", " ").replace("\n", " ")).strip()


def encode_text(text: str, tokenizer_model: str) -> torch.Tensor:
    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    ids = processor.encode(normalize_text(text), out_type=int)
    if not ids:
        raise ValueError("text tokenizer produced no tokens")
    # v1 manifest content_token_ids use the CTC offset +1; zero is padding.
    return torch.as_tensor([int(x) + 1 for x in ids], dtype=torch.long)


def load_checkpoint(
    path: Path,
    device: torch.device,
    *,
    use_ema: bool = True,
) -> tuple[DDLFMTrainModule, dict[str, Any]]:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = dict(payload.get("config") or {})
    state_probe = payload.get("ema_model") or payload.get("model") or {}
    prompt_value = state_probe.get("decoder.speaker_prompt") if isinstance(state_probe, dict) else None
    inferred_prompt_tokens = int(prompt_value.shape[1]) if torch.is_tensor(prompt_value) and prompt_value.ndim == 3 else 4
    args = SimpleNamespace(
        latent_dim=int(cfg.get("latent_dim", 768)),
        semantic_dim=int(cfg.get("semantic_dim", 512)),
        speaker_dim=int(cfg.get("speaker_dim", 192)),
        hidden_size=int(cfg.get("hidden_size", 768)),
        num_layers=int(cfg.get("num_layers", 12)),
        num_heads=int(cfg.get("num_heads", 12)),
        ffn_size=int(cfg.get("ffn_size", 3072)),
        cross_gate_init=float(cfg.get("cross_gate_init", 0.0)),
        num_speaker_prompt_tokens=int(cfg.get("num_speaker_prompt_tokens", inferred_prompt_tokens)),
        speaker_condition_scale=float(cfg.get("speaker_condition_scale", 4.0)),
        speaker_input_scale=float(cfg.get("speaker_input_scale", 1.0)),
        text_vocab_size=int(cfg.get("text_vocab_size", 8001)),
        text_padding_id=int(cfg.get("text_padding_id", 0)),
        smoke_small_model=bool(cfg.get("smoke_small_model", False)),
    )
    module = DDLFMTrainModule(args).to(device).eval()
    state = payload.get("ema_model") if use_ema else None
    using_ema = state is not None
    if state is None:
        state = payload["model"]
    # Batch-47 adds AdaLN branches and multi-token prompts.  Loading an older
    # Batch-45/46 checkpoint remains useful for diagnostics: preserve all
    # matching legacy weights and leave only the new branches initialized.
    module.load_state_dict(state, strict=False)
    cfg["_checkpoint_has_ema"] = payload.get("ema_model") is not None
    cfg["_checkpoint_using_ema"] = bool(using_ema)
    cfg["_checkpoint_ema_metadata"] = dict(payload.get("ema") or {})
    return module, cfg


@torch.inference_mode()
def sample_velocity(
    module: DDLFMTrainModule,
    *,
    target_frames: int,
    semantic: torch.Tensor,
    semantic_mask: torch.Tensor,
    speaker: torch.Tensor,
    modality: int,
    steps: int,
    cfg_scale: float,
    semantic_cfg_scale: float = 0.0,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    if int(target_frames) <= 0:
        raise ValueError("target_frames must be positive")
    if int(steps) <= 0:
        raise ValueError("sampling steps must be positive")
    generator = torch.Generator(device=device).manual_seed(int(seed))
    x = torch.randn((1, int(target_frames), 768), generator=generator, device=device)
    target_mask = torch.ones((1, int(target_frames)), dtype=torch.bool, device=device)
    semantic = semantic.to(device=device)
    semantic_mask = semantic_mask.to(device=device)
    speaker = speaker.to(device=device)
    if not math.isfinite(float(cfg_scale)) or float(cfg_scale) < 0.0:
        raise ValueError("cfg_scale must be finite and non-negative")
    zero_speaker = torch.zeros_like(speaker)
    zero_semantic = torch.zeros_like(semantic)
    zero_semantic_mask = torch.zeros_like(semantic_mask, dtype=torch.bool)
    for index in range(int(steps)):
        t_value = float(index) / float(steps)
        t = torch.full((1,), t_value, device=device)
        modality_tensor = torch.tensor([int(modality)], device=device)
        if float(semantic_cfg_scale) > 0.0:
            velocity_cond = module.decoder(
                x, t, semantic, speaker,
                target_mask=target_mask,
                semantic_mask=semantic_mask,
                semantic_modality=modality_tensor,
            ).velocity
            velocity_speaker = module.decoder(
                x, t, zero_semantic, speaker,
                target_mask=target_mask,
                semantic_mask=zero_semantic_mask,
                semantic_modality=modality_tensor,
            ).velocity
            velocity_semantic = module.decoder(
                x, t, semantic, zero_speaker,
                target_mask=target_mask,
                semantic_mask=semantic_mask,
                semantic_modality=modality_tensor,
            ).velocity
            velocity = combine_dual_cfg_velocity(
                velocity_cond,
                velocity_speaker,
                velocity_semantic,
                float(cfg_scale),
                float(semantic_cfg_scale),
            )
        else:
            velocity_cond = module.decoder(
                x,
                t,
                semantic,
                speaker,
                target_mask=target_mask,
                semantic_mask=semantic_mask,
                semantic_modality=modality_tensor,
            ).velocity
            if float(cfg_scale) == 1.0:
                velocity = velocity_cond
            else:
                velocity_uncond = module.decoder(
                    x,
                    t,
                    semantic,
                    zero_speaker,
                    target_mask=target_mask,
                    semantic_mask=semantic_mask,
                    semantic_modality=modality_tensor,
                ).velocity
                velocity = combine_cfg_velocity(velocity_cond, velocity_uncond, float(cfg_scale))
        x = x + velocity / float(steps)
    return x


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    module, cfg = load_checkpoint(
        Path(args.checkpoint).expanduser().resolve(),
        device,
        use_ema=not bool(args.no_ema),
    )
    cfg_scale = (
        float(args.cfg_scale)
        if args.cfg_scale is not None
        else (1.5 if float(cfg.get("speaker_dropout", 0.0)) > 0.0 else 1.0)
    )
    speaker_cfg_scale = (
        float(args.speaker_cfg_scale)
        if args.speaker_cfg_scale is not None
        else float(cfg.get("speaker_cfg_scale", cfg_scale))
    )
    semantic_cfg_scale = (
        float(args.semantic_cfg_scale)
        if args.semantic_cfg_scale is not None
        else float(cfg.get("semantic_cfg_scale", 0.0))
    )
    using_ema = bool(cfg.get("_checkpoint_using_ema", False))
    normalization_enabled = bool(cfg.get("zq_normalization_enabled", False))
    if args.mode == "no_text":
        if not args.semantic_path:
            raise ValueError("--semantic-path is required for no_text")
        arr = np.load(args.semantic_path).astype("float32", copy=False)
        if arr.ndim != 2 or arr.shape[1] != int(cfg.get("semantic_dim", 512)):
            raise ValueError(f"semantic must be [T,512], got {arr.shape}")
        semantic = torch.from_numpy(arr).unsqueeze(0)
        semantic_mask = torch.ones((1, arr.shape[0]), dtype=torch.bool)
        target_frames = int(args.target_frames or arr.shape[0])
    else:
        if not args.text:
            raise ValueError("--text is required for text mode")
        ids = encode_text(args.text, args.tokenizer_model).unsqueeze(0)
        state = module.text_encoder(ids.to(device), torch.ones_like(ids, dtype=torch.bool, device=device))
        semantic = state.memory.detach().cpu()
        semantic_mask = state.mask.detach().cpu()
        target_frames = int(args.target_frames)
        if target_frames <= 0:
            raise ValueError("--target-frames is required for text mode")
    speaker = load_embedding(args.speaker_embedding, int(cfg.get("speaker_dim", 192))).unsqueeze(0)
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run",
            "mode": args.mode,
            "target_frames": target_frames,
            "semantic_shape": list(semantic.shape),
            "speaker_shape": list(speaker.shape),
            "parameters": module.decoder.parameter_count(),
            "cfg_scale": cfg_scale,
            "speaker_cfg_scale": speaker_cfg_scale,
            "semantic_cfg_scale": semantic_cfg_scale,
            "using_ema": using_ema,
            "zq_normalization_enabled": normalization_enabled,
        }, ensure_ascii=False))
        return 0
    z_pred = sample_velocity(
        module,
        target_frames=target_frames,
        semantic=semantic,
        semantic_mask=semantic_mask,
        speaker=speaker,
        modality=0 if args.mode == "no_text" else 1,
        steps=int(args.sampling_steps),
        cfg_scale=speaker_cfg_scale,
        semantic_cfg_scale=semantic_cfg_scale,
        device=device,
        seed=int(args.seed),
    )
    stats_path: Path | None = None
    actual_stats_sha = ""
    if normalization_enabled:
        stats_value = args.zq_channel_stats or str(cfg.get("zq_channel_stats") or "")
        if not stats_value:
            raise ValueError("normalized DDLFM inference requires zq channel stats")
        stats_path = Path(stats_value).expanduser().resolve()
        expected_stats_sha = str(cfg.get("zq_channel_stats_sha256") or "")
        actual_stats_sha = sha256_file(stats_path)
        if expected_stats_sha and actual_stats_sha != expected_stats_sha:
            raise ValueError(
                f"zq channel stats SHA256 mismatch: {actual_stats_sha} != {expected_stats_sha}"
            )
        stats = load_zq_channel_stats(stats_path)
        if str(stats.get("status")) != "completed" or bool(stats.get("partial", False)):
            raise ValueError(f"refusing incomplete zq channel stats: {stats_path}")
        z_pred = denormalize_zq(z_pred, stats, channel_dim=-1)
    if not args.codec_path:
        raise ValueError("--codec-path is required unless --dry-run is used")
    codec = MossCodec(
        args.codec_path,
        moss_root=args.moss_root or None,
        device=str(device),
        dtype=args.codec_dtype,
    )
    latents = z_pred.transpose(1, 2).contiguous()
    lengths = torch.tensor([target_frames], dtype=torch.long, device=device)
    waveform, waveform_lengths = decode_latents(codec.model, latents, lengths)
    wav = waveform[0, 0, : int(waveform_lengths[0].item())].detach().float().cpu().numpy()
    import soundfile as sf

    output = Path(args.output_wav).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), wav, codec.sample_rate)
    print(json.dumps({
        "status": "completed",
        "output_wav": str(output),
        "samples": int(wav.shape[0]),
        "sample_rate": int(codec.sample_rate),
        "target_frames": target_frames,
        "sampling_steps": int(args.sampling_steps),
        "cfg_scale": cfg_scale,
        "speaker_cfg_scale": speaker_cfg_scale,
        "semantic_cfg_scale": semantic_cfg_scale,
        "using_ema": using_ema,
        "zq_normalization_enabled": normalization_enabled,
        "zq_channel_stats": str(stats_path) if stats_path is not None else None,
        "zq_channel_stats_sha256": actual_stats_sha or None,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

from moss_codecvc.audio import decode_latents
from moss_codecvc.moss_codec import MossCodec
from moss_codecvc.models.source_semantic_memory import SourceTokenMemoryEncoder
from scripts.ver3_1.train_ddlfm_cfm import DDLFMTrainModule, load_embedding


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


def load_checkpoint(path: Path, device: torch.device) -> tuple[DDLFMTrainModule, dict[str, Any]]:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = dict(payload.get("config") or {})
    args = SimpleNamespace(
        latent_dim=int(cfg.get("latent_dim", 768)),
        semantic_dim=int(cfg.get("semantic_dim", 512)),
        speaker_dim=int(cfg.get("speaker_dim", 192)),
        hidden_size=int(cfg.get("hidden_size", 768)),
        num_layers=int(cfg.get("num_layers", 12)),
        num_heads=int(cfg.get("num_heads", 12)),
        ffn_size=int(cfg.get("ffn_size", 3072)),
        text_vocab_size=int(cfg.get("text_vocab_size", 8001)),
        text_padding_id=int(cfg.get("text_padding_id", 0)),
        smoke_small_model=bool(cfg.get("smoke_small_model", False)),
    )
    module = DDLFMTrainModule(args).to(device).eval()
    module.load_state_dict(payload["model"], strict=True)
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
    for index in range(int(steps)):
        t_value = float(index) / float(steps)
        t = torch.full((1,), t_value, device=device)
        velocity = module.decoder(
            x,
            t,
            semantic,
            speaker,
            target_mask=target_mask,
            semantic_mask=semantic_mask,
            semantic_modality=torch.tensor([int(modality)], device=device),
        ).velocity
        x = x + velocity / float(steps)
    return x


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    module, cfg = load_checkpoint(Path(args.checkpoint).expanduser().resolve(), device)
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
        device=device,
        seed=int(args.seed),
    )
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
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

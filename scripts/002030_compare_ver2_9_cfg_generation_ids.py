#!/usr/bin/env python3
"""Compare saved ver2.9 CFG generation ids at codec-frame level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg1", type=Path, required=True, help="Saved generation ids .pt for cfg=1.0.")
    ap.add_argument("--cfg13", type=Path, required=True, help="Saved generation ids .pt for cfg=1.3.")
    ap.add_argument("--audio-pad-code", type=int, default=1024)
    ap.add_argument("--output-json", type=Path, required=True)
    return ap.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def first_codec_sequence(payload: dict[str, Any], *, audio_pad_code: int) -> torch.Tensor:
    output = payload.get("output") or []
    if not output:
        raise ValueError("saved generation payload has no output entries")
    _, generation_ids = output[0]
    if not torch.is_tensor(generation_ids):
        generation_ids = torch.as_tensor(generation_ids)
    if generation_ids.dim() != 2 or generation_ids.shape[-1] < 2:
        raise ValueError(f"generation ids must be [T, 1+n_vq], got {tuple(generation_ids.shape)}")
    codec = generation_ids[:, 1:].detach().float()
    valid = (codec >= 0).all(dim=-1) & (codec != float(audio_pad_code)).any(dim=-1)
    return codec[valid]


def main() -> int:
    args = parse_args()
    cfg1_payload = load_payload(args.cfg1)
    cfg13_payload = load_payload(args.cfg13)
    codec1 = first_codec_sequence(cfg1_payload, audio_pad_code=int(args.audio_pad_code))
    codec13 = first_codec_sequence(cfg13_payload, audio_pad_code=int(args.audio_pad_code))
    aligned = min(int(codec1.shape[0]), int(codec13.shape[0]))
    if aligned <= 0:
        raise ValueError("no aligned codec frames to compare")
    frame_cos = F.cosine_similarity(codec1[:aligned], codec13[:aligned], dim=-1)
    exact_match = (codec1[:aligned].long() == codec13[:aligned].long()).all(dim=-1).float()
    payload = {
        "cfg1_path": str(args.cfg1),
        "cfg13_path": str(args.cfg13),
        "case_id_cfg1": cfg1_payload.get("case_id"),
        "case_id_cfg13": cfg13_payload.get("case_id"),
        "cfg1_scale": cfg1_payload.get("timbre_cfg_scale"),
        "cfg13_scale": cfg13_payload.get("timbre_cfg_scale"),
        "frames_cfg1": int(codec1.shape[0]),
        "frames_cfg13": int(codec13.shape[0]),
        "aligned_frames": int(aligned),
        "frame_cosine_mean": float(frame_cos.mean().item()),
        "frame_cosine_min": float(frame_cos.min().item()),
        "frame_cosine_max": float(frame_cos.max().item()),
        "exact_frame_match_rate": float(exact_match.mean().item()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

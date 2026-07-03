#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.config import deep_get, load_config
from moss_codecvc.io_utils import iter_jsonl, pick_first, stable_id, write_jsonl


def infer_language(text: str | None) -> str:
    if text and any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return "zh"
    return "en"


def build_from_pairs(rows: list[dict[str, Any]], cfg: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    source_audio_keys = deep_get(cfg, "data.source_audio_keys", [])
    source_text_keys = deep_get(cfg, "data.source_text_keys", [])
    timbre_audio_keys = deep_get(cfg, "data.timbre_audio_keys", [])
    timbre_text_keys = deep_get(cfg, "data.timbre_text_keys", [])
    target_audio_keys = deep_get(cfg, "data.target_audio_keys", [])
    target_text_keys = deep_get(cfg, "data.target_text_keys", [])
    default_instruction = deep_get(cfg, "instruction.prosody_no_timbre") or deep_get(cfg, "instruction.default")

    out = []
    for idx, row in enumerate(rows):
        source_audio = pick_first(row, source_audio_keys)
        source_text = pick_first(row, source_text_keys)
        timbre_ref_audio = pick_first(row, timbre_audio_keys)
        timbre_ref_text = pick_first(row, timbre_text_keys)
        target_audio = pick_first(row, target_audio_keys)
        target_text = pick_first(row, target_text_keys, source_text)
        if args.identity_target and not target_audio:
            target_audio = source_audio
        if not source_audio or not timbre_ref_audio:
            continue
        if args.require_target and not target_audio:
            continue
        sample_id = row.get("pair_id") or f"{args.run_name}:{idx:06d}:{stable_id(source_audio, timbre_ref_audio, target_audio)}"
        language = row.get("language") or infer_language(source_text or target_text)
        out.append(
            {
                "sample_id": sample_id,
                "pair_type": row.get("pair_type") or args.pair_type,
                "language": language,
                "source_audio": source_audio,
                "source_text": source_text,
                "timbre_ref_audio": timbre_ref_audio,
                "timbre_ref_text": timbre_ref_text,
                "target_audio": target_audio,
                "target_text": target_text,
                "instruction": row.get("instruction") or default_instruction,
                "meta": {
                    "source_row_index": idx,
                    "source_pair_id": row.get("pair_id"),
                    "builder_mode": "from_pairs",
                },
            }
        )
    return out


def build_from_sources(rows: list[dict[str, Any]], cfg: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    source_audio_keys = deep_get(cfg, "data.source_audio_keys", [])
    source_text_keys = deep_get(cfg, "data.source_text_keys", [])
    default_instruction = deep_get(cfg, "instruction.prosody_no_timbre") or deep_get(cfg, "instruction.default")
    cleaned = []
    for idx, row in enumerate(rows):
        audio = pick_first(row, source_audio_keys)
        text = pick_first(row, source_text_keys)
        if audio:
            cleaned.append((idx, row, audio, text))

    out = []
    n = len(cleaned)
    if n == 0:
        return out
    for local_idx, (src_idx, row, audio, text) in enumerate(cleaned):
        ref_idx = (local_idx + args.timbre_offset) % n
        if ref_idx == local_idx and n > 1:
            ref_idx = (ref_idx + 1) % n
        _, ref_row, ref_audio, ref_text = cleaned[ref_idx]
        target_audio = audio if args.identity_target else None
        if args.require_target and not target_audio:
            continue
        sample_id = row.get("pair_id") or f"{args.run_name}:{local_idx:06d}:{stable_id(audio, ref_audio)}"
        out.append(
            {
                "sample_id": sample_id,
                "pair_type": args.pair_type,
                "language": row.get("language") or infer_language(text),
                "source_audio": audio,
                "source_text": text,
                "timbre_ref_audio": ref_audio,
                "timbre_ref_text": ref_text,
                "target_audio": target_audio,
                "target_text": text,
                "instruction": default_instruction,
                "meta": {
                    "source_row_index": src_idx,
                    "timbre_ref_row_index": cleaned[ref_idx][0],
                    "builder_mode": "from_sources",
                    "identity_target": bool(args.identity_target),
                },
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--mode", choices=["from_pairs", "from_sources"], default="from_pairs")
    ap.add_argument("--run-name", default="moss_codecvc")
    ap.add_argument("--pair-type", default="MOSSCodecVC")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timbre-offset", type=int, default=1)
    ap.add_argument("--identity-target", action="store_true")
    ap.add_argument("--require-target", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    rows = list(iter_jsonl(args.input_jsonl))
    if args.limit > 0:
        rows = rows[: args.limit]
    if args.mode == "from_pairs":
        out = build_from_pairs(rows, cfg, args)
    else:
        out = build_from_sources(rows, cfg, args)
    n = write_jsonl(args.output_jsonl, out)
    print(f"wrote {n} VC manifest rows -> {Path(args.output_jsonl).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

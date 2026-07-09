#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc


def nested_get(row: dict[str, Any], key: str) -> Any | None:
    value = row.get(key)
    if value not in (None, ""):
        return value
    meta = row.get("moss_codecvc_meta")
    if isinstance(meta, dict):
        value = meta.get(key)
        if value not in (None, ""):
            return value
    return None


def pick_text(row: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, str]:
    for key in keys:
        value = nested_get(row, key)
        if value in (None, ""):
            continue
        text = str(value).strip()
        if text and text not in {"<NO_TEXT>", "None", "none", "null"}:
            return text, key
    return "", ""


def normalize_for_distance(text: str) -> str:
    out: list[str] = []
    for ch in str(text or "").lower():
        code = ord(ch)
        if ch.isalnum() or 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF:
            out.append(ch)
    return "".join(out)


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]


def safe_ratio(num: float, den: float) -> float | None:
    if not math.isfinite(num) or not math.isfinite(den) or den <= 0:
        return None
    return float(num) / float(den)


def main() -> int:
    ap = argparse.ArgumentParser(description="Mark v2 real-target no-text rows with known source/target content metadata.")
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--progress-every", type=int, default=10000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input_jsonl).expanduser()
    output_path = Path(args.output_jsonl).expanduser()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists, pass --overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")

    rows = 0
    kept = 0
    missing = 0
    with tmp_path.open("w", encoding="utf-8") as out:
        for _line_no, row in iter_jsonl(input_path):
            rows += 1
            cur = dict(row)
            source_text, source_key = pick_text(cur, ("source_text", "content_ref_text", "target_text", "asr_src_text"))
            target_text, target_key = pick_text(cur, ("target_text", "source_text", "content_ref_text", "asr_tgt_text"))
            content_text = source_text or target_text
            content_key = source_key or target_key
            if content_text:
                kept += 1
                content_keep = True
                reason = "known_text_v2_real_no_text"
            else:
                missing += 1
                content_keep = False
                reason = "missing_known_text"

            norm_src = normalize_for_distance(source_text)
            norm_tgt = normalize_for_distance(target_text)
            dist = edit_distance(norm_tgt, norm_src) if norm_src or norm_tgt else 0
            cer = safe_ratio(dist, max(1, len(norm_tgt)))

            meta = dict(cur.get("moss_codecvc_meta") or {})
            source_frames = meta.get("source_codec_frames") or cur.get("source_codec_frames")
            target_frames = meta.get("target_codec_frames") or cur.get("tokens") or cur.get("target_codec_frames")
            duration_ratio = None
            try:
                duration_ratio = safe_ratio(float(target_frames), float(source_frames))
            except Exception:
                duration_ratio = None

            cur.update(
                {
                    "asr_src_text": source_text,
                    "asr_tgt_text": target_text,
                    "content_ref_text": content_text,
                    "content_ref_text_key": content_key,
                    "content_keep": content_keep,
                    "content_filter_reason": "keep" if content_keep else reason,
                    "cer_tgt": 0.0 if cer is None else cer,
                    "wer_tgt": 0.0 if cer is None else cer,
                    "repeat_score": 0.0,
                    "duration_ratio_tgt_src": duration_ratio,
                    "v2_real_no_text_known_content": {
                        "source_text_key": source_key,
                        "target_text_key": target_key,
                        "content_text_key": content_key,
                        "normalized_source_len": len(norm_src),
                        "normalized_target_len": len(norm_tgt),
                        "source_target_edit_distance": dist,
                    },
                }
            )
            out.write(json.dumps(cur, ensure_ascii=False) + "\n")
            if rows % max(1, int(args.progress_every)) == 0:
                print(f"[known-content] rows={rows} kept={kept} missing={missing}", flush=True)

    tmp_path.replace(output_path)
    summary = {
        "status": "complete",
        "input_jsonl": str(input_path.resolve(strict=False)),
        "output_jsonl": str(output_path.resolve(strict=False)),
        "rows": rows,
        "content_keep": kept,
        "missing_known_text": missing,
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_path.with_name(output_path.name + ".done.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[known-content] wrote rows={rows} kept={kept} output={output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

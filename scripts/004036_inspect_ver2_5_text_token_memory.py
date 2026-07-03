#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPM = ROOT / "trainset/shared_content_tokenizers/spm_multilingual_byte_fallback_v1.model"
TEXT_KEYS = ("source_text", "content_ref_text", "asr_src_text", "source_asr_text", "text")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Inspect the transcript/token input used by Ver2.5 P0-A text-token memory."
    )
    ap.add_argument("--text", default="", help="Direct transcript text to tokenize.")
    ap.add_argument("--jsonl", default="", help="Optional JSONL with validation/train rows.")
    ap.add_argument("--case-id", default="", help="Case id to select from --jsonl.")
    ap.add_argument("--row-index", type=int, default=0, help="Fallback row index if --case-id is empty.")
    ap.add_argument("--spm-model", default=str(DEFAULT_SPM))
    ap.add_argument("--max-pieces", type=int, default=80)
    return ap.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc


def pick_row(path: Path, case_id: str, row_index: int) -> dict[str, Any]:
    rows_seen = 0
    for row in iter_jsonl(path):
        if case_id:
            if str(row.get("case_id") or row.get("sample_id") or "") == case_id:
                return row
        elif rows_seen == row_index:
            return row
        rows_seen += 1
    if case_id:
        raise ValueError(f"case_id not found: {case_id}")
    raise ValueError(f"row_index out of range: {row_index}")


def pick_text(row: dict[str, Any]) -> tuple[str, str]:
    for key in TEXT_KEYS:
        value = row.get(key)
        if value in (None, ""):
            continue
        text = str(value).strip()
        if text and text != "<NO_TEXT>":
            return text, key
    return "", ""


def main() -> int:
    args = parse_args()
    row: dict[str, Any] = {}
    text = str(args.text or "").strip()
    text_key = "direct_text" if text else ""
    if args.jsonl:
        row = pick_row(Path(args.jsonl).expanduser(), args.case_id, int(args.row_index))
        if not text:
            text, text_key = pick_text(row)
    if not text:
        raise ValueError("missing transcript text; pass --text or a JSONL row with source/content text")

    import sentencepiece as spm

    model_path = Path(args.spm_model).expanduser()
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    ids = list(processor.encode(text, out_type=int))
    pieces = list(processor.encode(text, out_type=str))
    max_pieces = max(0, int(args.max_pieces))
    payload = {
        "case_id": row.get("case_id") or row.get("sample_id"),
        "text_key": text_key,
        "text": text,
        "text_char_length": len(text),
        "spm_model": str(model_path),
        "token_count": len(ids),
        "token_ids": ids[:max_pieces],
        "token_pieces": pieces[:max_pieces],
        "truncated": len(ids) > max_pieces,
        "content_ref_text_source": row.get("content_ref_text_source") or row.get("eval_text_source"),
        "content_asr_backend": row.get("content_asr_backend"),
        "content_asr_model": row.get("content_asr_model"),
        "source_asr_backend": row.get("source_asr_backend"),
        "source_asr_model": row.get("source_asr_model"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

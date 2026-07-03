#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moss_codecvc.io_utils import iter_jsonl
from moss_codecvc.modes import VC_NO_TEXT_PLACEHOLDER


DEFAULT_TEXT_KEYS = (
    "content_ref_text",
    "text",
    "target_text",
    "normalized_text",
    "asr_src_text",
    "source_asr_text",
    "source_text",
    "source_transcript",
    "transcript",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Attach shared multilingual SentencePiece TextCTC token IDs to a MOSS-CodecVC JSONL."
    )
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--spm-model", required=True)
    ap.add_argument(
        "--tokenizer-meta",
        default="",
        help="Optional tokenizer .json metadata produced by 001045. Defaults to <spm_model without .model>.json.",
    )
    ap.add_argument("--tokenizer-id", default="", help="Override tokenizer id stored in each row.")
    ap.add_argument("--text-keys", default=",".join(DEFAULT_TEXT_KEYS))
    ap.add_argument("--lowercase-latin", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--strip-extra-whitespace", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--require-content-keep", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--blank-id", type=int, default=0)
    ap.add_argument("--token-offset", type=int, default=1)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=10000)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def split_keys(spec: str) -> list[str]:
    return [item.strip() for item in str(spec or "").split(",") if item.strip()]


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


def parse_bool_value(value: Any | None, *, default: bool = True) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "keep"}:
        return True
    if text in {"0", "false", "no", "n", "drop", "filtered"}:
        return False
    return bool(default)


def content_allowed(row: dict[str, Any], *, require_content_keep: bool) -> bool:
    if not require_content_keep:
        return True
    return parse_bool_value(nested_get(row, "content_keep"), default=True) and parse_bool_value(
        nested_get(row, "content_token_keep"),
        default=True,
    )


def pick_text(row: dict[str, Any], keys: list[str]) -> tuple[str, str]:
    for key in keys:
        value = nested_get(row, key)
        if value in (None, ""):
            continue
        text = str(value).strip()
        if text and text not in {VC_NO_TEXT_PLACEHOLDER, "None", "none", "null"}:
            return text, key
    return "", ""


def normalize_text(text: str, *, lowercase_latin: bool, strip_extra_whitespace: bool) -> str:
    text = str(text or "").replace("\t", " ").replace("\n", " ").strip()
    if lowercase_latin:
        text = re.sub(r"[A-Z]+", lambda m: m.group(0).lower(), text)
    if strip_extra_whitespace:
        text = re.sub(r"\s+", " ", text)
    return text.strip()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_tokenizer_meta(spm_model: Path, explicit_meta: str) -> dict[str, Any]:
    meta_path = Path(explicit_meta).expanduser() if explicit_meta else Path(str(spm_model).removesuffix(".model") + ".json")
    if meta_path.exists():
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        payload["metadata_path"] = str(meta_path.resolve(strict=False))
        return payload
    return {
        "tokenizer_id": spm_model.stem,
        "tokenizer": "sentencepiece",
        "metadata_path": "",
    }


def main() -> int:
    args = parse_args()
    try:
        import sentencepiece as spm
    except Exception as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError("sentencepiece is required. Install it in the training/data env first.") from exc

    input_path = Path(args.input_jsonl).expanduser()
    output_path = Path(args.output_jsonl).expanduser()
    spm_model = Path(args.spm_model).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not spm_model.exists():
        raise FileNotFoundError(spm_model)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists, pass --overwrite: {output_path}")

    processor = spm.SentencePieceProcessor(model_file=str(spm_model))
    tokenizer_meta = load_tokenizer_meta(spm_model, args.tokenizer_meta)
    tokenizer_id = str(args.tokenizer_id or tokenizer_meta.get("tokenizer_id") or spm_model.stem)
    vocab_size_with_blank = int(processor.get_piece_size()) + int(args.token_offset)
    text_keys = split_keys(args.text_keys)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")

    stats: Counter[str] = Counter()
    key_counts: Counter[str] = Counter()
    progress_every = max(1, int(args.progress_every))
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(input_path):
            if args.max_rows > 0 and stats["rows"] >= args.max_rows:
                break
            out = dict(row)
            allowed = content_allowed(out, require_content_keep=bool(args.require_content_keep))
            text, text_key = pick_text(out, text_keys)
            ids: list[int] = []
            normalized = ""
            if allowed and text:
                normalized = normalize_text(
                    text,
                    lowercase_latin=bool(args.lowercase_latin),
                    strip_extra_whitespace=bool(args.strip_extra_whitespace),
                )
                if normalized:
                    ids = [int(item) + int(args.token_offset) for item in processor.encode(normalized, out_type=int)]
            out["content_ref_text"] = text
            out["content_ref_text_key"] = text_key
            out["content_token_ids"] = ids
            out["content_token_length"] = len(ids)
            out["content_ctc_vocab_size"] = vocab_size_with_blank
            out["content_ctc_blank_id"] = int(args.blank_id)
            out["content_ctc_token_offset"] = int(args.token_offset)
            out["content_tokenizer"] = "sentencepiece"
            out["content_tokenizer_id"] = tokenizer_id
            out["content_vocab_path"] = str(spm_model.resolve(strict=False))
            out["content_tokenizer_meta_path"] = str(tokenizer_meta.get("metadata_path") or "")
            out["content_token_keep"] = bool(ids) and allowed
            if not allowed:
                out["content_token_filter_reason"] = "content_keep_false"
                stats["content_keep_false"] += 1
            elif not text:
                out["content_token_filter_reason"] = "missing_content_ref_text"
                stats["missing_text"] += 1
            elif not normalized or not ids:
                out["content_token_filter_reason"] = "empty_content_tokens"
                stats["empty_tokens"] += 1
            else:
                out["content_token_filter_reason"] = "keep"
                stats["tokenized"] += 1
                key_counts[text_key] += 1
            stats["rows"] += 1
            stats["tokens_total"] += len(ids)
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            if stats["rows"] % progress_every == 0:
                print(
                    f"[content-tokens-spm] rows={stats['rows']} tokenized={stats['tokenized']} "
                    f"missing={stats['missing_text']} filtered={stats['content_keep_false']}",
                    flush=True,
                )

    tmp_path.replace(output_path)
    summary = {
        "status": "complete",
        "input_jsonl": str(input_path.resolve(strict=False)),
        "output_jsonl": str(output_path.resolve(strict=False)),
        "spm_model": str(spm_model.resolve(strict=False)),
        "tokenizer_id": tokenizer_id,
        "content_ctc_vocab_size": vocab_size_with_blank,
        "blank_id": int(args.blank_id),
        "token_offset": int(args.token_offset),
        "stats": dict(stats),
        "text_key_counts": dict(key_counts),
        "avg_tokens": float(stats["tokens_total"]) / max(1, int(stats["tokenized"])),
    }
    write_json_atomic(output_path.with_suffix(output_path.suffix + ".summary.json"), summary)
    write_json_atomic(output_path.with_name(output_path.name + ".done.json"), summary)
    print(
        f"[content-tokens-spm] wrote rows={stats['rows']} tokenized={stats['tokenized']} "
        f"vocab_size={vocab_size_with_blank} output={output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

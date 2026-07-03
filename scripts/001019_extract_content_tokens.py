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
    "asr_src_text",
    "source_asr_text",
    "source_text",
    "source_transcript",
    "transcript",
    "text",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Attach Ver2.1 CTC content token IDs to a MOSS-CodecVC JSONL manifest."
    )
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--vocab-json", required=True)
    ap.add_argument("--text-keys", default=",".join(DEFAULT_TEXT_KEYS))
    ap.add_argument("--tokenizer", choices=("char", "byte"), default="char")
    ap.add_argument("--lowercase-latin", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--strip-whitespace", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--drop-punctuation", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--require-content-keep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only emit CTC token IDs for rows whose content_keep/content_token_keep is not explicitly false.",
    )
    ap.add_argument("--min-token-count", type=int, default=1)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=10000)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--store",
        choices=("inline",),
        default="inline",
        help="Currently stores token IDs inline as content_token_ids.",
    )
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


def content_ctc_allowed(row: dict[str, Any], *, require_content_keep: bool) -> bool:
    if not require_content_keep:
        return True
    return parse_bool_value(nested_get(row, "content_keep"), default=True) and parse_bool_value(
        nested_get(row, "content_token_keep"),
        default=True,
    )


def pick_text(row: dict[str, Any], keys: list[str]) -> str:
    text, _key = pick_text_with_key(row, keys)
    return text


def pick_text_with_key(row: dict[str, Any], keys: list[str]) -> tuple[str, str]:
    for key in keys:
        value = nested_get(row, key)
        if value in (None, ""):
            continue
        text = str(value).strip()
        if text and text not in {VC_NO_TEXT_PLACEHOLDER, "None", "none", "null"}:
            return text, key
    return "", ""


def normalize_text(text: str, *, lowercase_latin: bool, strip_whitespace: bool, drop_punctuation: bool) -> str:
    text = str(text or "").replace("\t", " ").replace("\n", " ").strip()
    if lowercase_latin:
        text = re.sub(r"[A-Z]+", lambda m: m.group(0).lower(), text)
    if drop_punctuation:
        text = re.sub(r"[^\w\u4e00-\u9fff\s]+", "", text)
    if strip_whitespace:
        text = re.sub(r"\s+", "", text)
    else:
        text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_to_symbols(text: str, tokenizer: str) -> list[str]:
    if tokenizer == "byte":
        return [f"b:{value}" for value in text.encode("utf-8", errors="ignore")]
    return [f"c:{ch}" for ch in text]


def load_vocab(path: Path) -> dict[str, int] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    symbols = payload.get("symbols")
    if not isinstance(symbols, dict):
        raise ValueError(f"invalid vocab JSON, missing symbols: {path}")
    return {str(key): int(value) for key, value in symbols.items()}


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def build_vocab(args: argparse.Namespace, text_keys: list[str], vocab_path: Path) -> dict[str, int]:
    existing = load_vocab(vocab_path)
    if existing is not None and not args.overwrite:
        return existing

    counts: Counter[str] = Counter()
    rows = 0
    missing_text = 0
    for row in iter_jsonl(args.input_jsonl):
        if args.max_rows > 0 and rows >= args.max_rows:
            break
        rows += 1
        if not content_ctc_allowed(row, require_content_keep=bool(args.require_content_keep)):
            continue
        text = pick_text(row, text_keys)
        if not text:
            missing_text += 1
            continue
        normalized = normalize_text(
            text,
            lowercase_latin=bool(args.lowercase_latin),
            strip_whitespace=bool(args.strip_whitespace),
            drop_punctuation=bool(args.drop_punctuation),
        )
        counts.update(text_to_symbols(normalized, args.tokenizer))

    symbols = [
        symbol
        for symbol, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if int(count) >= int(args.min_token_count)
    ]
    vocab = {symbol: idx + 1 for idx, symbol in enumerate(symbols)}
    payload = {
        "blank_id": 0,
        "token_offset": 1,
        "tokenizer": args.tokenizer,
        "lowercase_latin": bool(args.lowercase_latin),
        "strip_whitespace": bool(args.strip_whitespace),
        "drop_punctuation": bool(args.drop_punctuation),
        "min_token_count": int(args.min_token_count),
        "symbols": vocab,
        "vocab_size_with_blank": len(vocab) + 1,
        "rows_scanned": rows,
        "missing_text": missing_text,
    }
    write_json_atomic(vocab_path, payload)
    print(
        f"[content-tokens] built vocab symbols={len(vocab)} vocab_size_with_blank={len(vocab) + 1} "
        f"rows={rows} missing_text={missing_text} path={vocab_path}",
        flush=True,
    )
    return vocab


def encode_text(text: str, vocab: dict[str, int], args: argparse.Namespace) -> list[int]:
    normalized = normalize_text(
        text,
        lowercase_latin=bool(args.lowercase_latin),
        strip_whitespace=bool(args.strip_whitespace),
        drop_punctuation=bool(args.drop_punctuation),
    )
    ids = [vocab[symbol] for symbol in text_to_symbols(normalized, args.tokenizer) if symbol in vocab]
    return ids


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_jsonl).expanduser()
    output_path = Path(args.output_jsonl).expanduser()
    vocab_path = Path(args.vocab_json).expanduser()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists, pass --overwrite: {output_path}")
    text_keys = split_keys(args.text_keys)
    vocab = build_vocab(args, text_keys, vocab_path)
    vocab_size = len(vocab) + 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    stats = Counter()
    progress_every = max(1, int(args.progress_every))
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(input_path):
            if args.max_rows > 0 and stats["rows"] >= args.max_rows:
                break
            out = dict(row)
            text, text_key = pick_text_with_key(out, text_keys)
            ctc_allowed = content_ctc_allowed(out, require_content_keep=bool(args.require_content_keep))
            ids = encode_text(text, vocab, args) if text and ctc_allowed else []
            out["content_ref_text"] = text
            out["content_ref_text_key"] = text_key
            out["content_token_ids"] = ids
            out["content_ctc_vocab_size"] = vocab_size
            out["content_tokenizer"] = args.tokenizer
            out["content_vocab_path"] = str(vocab_path.resolve(strict=False))
            out["content_token_keep"] = bool(ids) and ctc_allowed
            if not ctc_allowed:
                out["content_token_filter_reason"] = "content_keep_false"
                stats["content_keep_false"] += 1
            elif not text:
                out["content_token_filter_reason"] = "missing_content_ref_text"
                stats["missing_text"] += 1
            elif not ids:
                out["content_token_filter_reason"] = "empty_content_tokens"
                stats["empty_tokens"] += 1
            else:
                out["content_token_filter_reason"] = "keep"
                stats["tokenized"] += 1
            stats["rows"] += 1
            stats["tokens_total"] += len(ids)
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            if stats["rows"] % progress_every == 0:
                print(
                    f"[content-tokens] rows={stats['rows']} tokenized={stats['tokenized']} "
                    f"missing_text={stats['missing_text']} empty_tokens={stats['empty_tokens']}",
                    flush=True,
                )

    tmp_path.replace(output_path)
    summary = {
        "status": "complete",
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "vocab_json": str(vocab_path),
        "vocab_size_with_blank": vocab_size,
        "stats": dict(stats),
        "avg_tokens": float(stats["tokens_total"]) / max(1, int(stats["tokenized"])),
    }
    write_json_atomic(output_path.with_suffix(output_path.suffix + ".summary.json"), summary)
    write_json_atomic(output_path.with_name(output_path.name + ".done.json"), summary)
    print(f"[content-tokens] wrote rows={stats['rows']} output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
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
        description="Train a shared multilingual SentencePiece tokenizer for Ver2.3 TextCTC."
    )
    ap.add_argument(
        "--input-jsonl",
        action="append",
        default=[],
        help="Input JSONL. Can be passed multiple times. Use no_text and text manifests together.",
    )
    ap.add_argument(
        "--input-jsonl-spec",
        default="",
        help="Comma/newline separated PATH[::max_rows=N] list. Convenience alternative to repeated --input-jsonl.",
    )
    ap.add_argument("--output-prefix", required=True, help="Output prefix without .model/.vocab suffix.")
    ap.add_argument("--tokenizer-id", default="", help="Stable id stored in metadata. Defaults to output prefix stem.")
    ap.add_argument("--text-keys", default=",".join(DEFAULT_TEXT_KEYS))
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--model-type", choices=("unigram", "bpe"), default="unigram")
    ap.add_argument("--character-coverage", type=float, default=0.9995)
    ap.add_argument("--byte-fallback", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lowercase-latin", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--strip-extra-whitespace", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--require-content-keep", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-rows-per-source", type=int, default=0)
    ap.add_argument("--min-chars", type=int, default=1)
    ap.add_argument("--progress-every", type=int, default=100000)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def split_keys(spec: str) -> list[str]:
    return [item.strip() for item in str(spec or "").split(",") if item.strip()]


def parse_input_spec(spec: str) -> list[tuple[Path, int]]:
    items: list[tuple[Path, int]] = []
    for line in str(spec or "").splitlines():
        for chunk in line.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = chunk.split("::")
            path = Path(parts[0]).expanduser()
            max_rows = 0
            for part in parts[1:]:
                if not part:
                    continue
                if "=" not in part:
                    raise ValueError(f"Invalid input spec part {part!r} in {chunk!r}")
                key, value = part.split("=", 1)
                key = key.strip().lower().replace("-", "_")
                if key == "max_rows":
                    max_rows = int(value)
                else:
                    raise ValueError(f"Unknown input spec key {key!r} in {chunk!r}")
            items.append((path, max_rows))
    return items


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


def iter_sources(args: argparse.Namespace) -> list[tuple[Path, int]]:
    sources = [(Path(item).expanduser(), int(args.max_rows_per_source)) for item in args.input_jsonl]
    sources.extend(parse_input_spec(args.input_jsonl_spec))
    if not sources:
        raise ValueError("Pass at least one --input-jsonl or --input-jsonl-spec.")
    return sources


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{Path(sys.argv[0]).stem}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    try:
        import sentencepiece as spm
    except Exception as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError("sentencepiece is required. Install it in the training/data env first.") from exc

    output_prefix = Path(args.output_prefix).expanduser()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    model_path = Path(str(output_prefix) + ".model")
    vocab_path = Path(str(output_prefix) + ".vocab")
    meta_path = Path(str(output_prefix) + ".json")
    if (model_path.exists() or vocab_path.exists() or meta_path.exists()) and not args.overwrite:
        raise FileExistsError(f"tokenizer outputs exist, pass --overwrite: {output_prefix}.*")

    text_keys = split_keys(args.text_keys)
    stats: Counter[str] = Counter()
    key_counts: Counter[str] = Counter()
    source_summaries: list[dict[str, Any]] = []
    progress_every = max(1, int(args.progress_every))
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as corpus:
        corpus_path = Path(corpus.name)
        for source_idx, (path, max_rows) in enumerate(iter_sources(args)):
            if not path.exists():
                raise FileNotFoundError(path)
            source_stats: Counter[str] = Counter()
            for row in iter_jsonl(path):
                if max_rows > 0 and source_stats["rows"] >= max_rows:
                    break
                source_stats["rows"] += 1
                stats["rows"] += 1
                if not content_allowed(row, require_content_keep=bool(args.require_content_keep)):
                    source_stats["content_keep_false"] += 1
                    stats["content_keep_false"] += 1
                    continue
                text, key = pick_text(row, text_keys)
                if not text:
                    source_stats["missing_text"] += 1
                    stats["missing_text"] += 1
                    continue
                text = normalize_text(
                    text,
                    lowercase_latin=bool(args.lowercase_latin),
                    strip_extra_whitespace=bool(args.strip_extra_whitespace),
                )
                if len(text) < int(args.min_chars):
                    source_stats["too_short"] += 1
                    stats["too_short"] += 1
                    continue
                corpus.write(text + "\n")
                source_stats["written"] += 1
                stats["written"] += 1
                key_counts[key] += 1
                if stats["rows"] % progress_every == 0:
                    print(
                        f"[content-tokenizer] rows={stats['rows']} written={stats['written']} "
                        f"missing={stats['missing_text']} filtered={stats['content_keep_false']}",
                        flush=True,
                    )
            source_summaries.append(
                {
                    "source_index": source_idx,
                    "path": str(path.resolve(strict=False)),
                    "max_rows": int(max_rows),
                    "stats": dict(source_stats),
                }
            )

    if stats["written"] <= 0:
        corpus_path.unlink(missing_ok=True)
        raise ValueError("No text was written to tokenizer corpus.")

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(output_prefix),
        model_type=str(args.model_type),
        vocab_size=int(args.vocab_size),
        character_coverage=float(args.character_coverage),
        byte_fallback=bool(args.byte_fallback),
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
        hard_vocab_limit=False,
        train_extremely_large_corpus=True,
    )
    corpus_path.unlink(missing_ok=True)

    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    tokenizer_id = str(args.tokenizer_id or output_prefix.name)
    meta = {
        "tokenizer_id": tokenizer_id,
        "tokenizer": "sentencepiece",
        "model_path": str(model_path.resolve(strict=False)),
        "vocab_path": str(vocab_path.resolve(strict=False)),
        "model_type": str(args.model_type),
        "vocab_size": int(processor.get_piece_size()),
        "vocab_size_with_blank": int(processor.get_piece_size()) + 1,
        "blank_id": 0,
        "token_offset": 1,
        "byte_fallback": bool(args.byte_fallback),
        "character_coverage": float(args.character_coverage),
        "lowercase_latin": bool(args.lowercase_latin),
        "strip_extra_whitespace": bool(args.strip_extra_whitespace),
        "require_content_keep": bool(args.require_content_keep),
        "text_keys": text_keys,
        "stats": dict(stats),
        "text_key_counts": dict(key_counts),
        "sources": source_summaries,
    }
    write_json_atomic(meta_path, meta)
    print(
        f"[content-tokenizer] wrote model={model_path} vocab={vocab_path} meta={meta_path} "
        f"vocab_size_with_blank={meta['vocab_size_with_blank']} written={stats['written']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

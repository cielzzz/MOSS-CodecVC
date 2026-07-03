#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.io_utils import iter_jsonl, safe_stem, stable_id
from moss_codecvc.modes import VC_NO_TEXT_PLACEHOLDER


DEFAULT_SOURCE_TEXT_KEYS = (
    "source_text",
    "source_transcript",
    "transcript",
    "asr_text",
    "source_asr_text",
)
DEFAULT_TARGET_TEXT_KEYS = (
    "target_text",
    "target_transcript",
    "transcript",
    "asr_text",
    "target_asr_text",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Attach offline content embeddings for Ver2 content auxiliary loss.")
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--embedding-dim", type=int, default=256)
    ap.add_argument(
        "--method",
        choices=("text_hash",),
        default="text_hash",
        help="text_hash is deterministic and dependency-free; feed it ASR/transcript text.",
    )
    ap.add_argument("--source-text-keys", default=",".join(DEFAULT_SOURCE_TEXT_KEYS))
    ap.add_argument("--target-text-keys", default=",".join(DEFAULT_TARGET_TEXT_KEYS))
    ap.add_argument("--include-target", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=1000)
    return ap.parse_args()


def split_keys(spec: str) -> list[str]:
    return [item.strip() for item in str(spec or "").split(",") if item.strip()]


def get_nested(row: dict[str, Any], key: str) -> Any | None:
    value = row.get(key)
    if value not in (None, ""):
        return value
    meta = row.get("moss_codecvc_meta")
    if isinstance(meta, dict):
        value = meta.get(key)
        if value not in (None, ""):
            return value
    return None


def pick_text(row: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = get_nested(row, key)
        if value in (None, ""):
            continue
        text = str(value).strip()
        if not text or text in {VC_NO_TEXT_PLACEHOLDER, "None", "none", "null"}:
            continue
        return text
    return None


def normalize_text(text: str) -> str:
    return " ".join(str(text).replace("\t", " ").replace("\n", " ").split()).strip().lower()


def _hash_to_index(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="little", signed=False)
    index = value % int(dim)
    sign = 1.0 if ((value >> 63) & 1) == 0 else -1.0
    return index, sign


def text_hash_embedding(text: str, dim: int) -> torch.Tensor:
    text = normalize_text(text)
    vec = torch.zeros(int(dim), dtype=torch.float32)
    if not text:
        return vec

    chars = [ch for ch in text if not ch.isspace()]
    tokens: list[tuple[str, float]] = []
    for ch in chars:
        tokens.append((f"c:{ch}", 1.0))
    for ngram in (2, 3):
        if len(chars) >= ngram:
            weight = 1.5 if ngram == 2 else 2.0
            for idx in range(len(chars) - ngram + 1):
                tokens.append((f"g{ngram}:{''.join(chars[idx:idx + ngram])}", weight))
    for word in text.split():
        tokens.append((f"w:{word}", 2.0))

    for token, weight in tokens:
        index, sign = _hash_to_index(token, int(dim))
        vec[index] += float(weight) * sign
    return F.normalize(vec, dim=0) if bool((vec != 0).any().item()) else vec


def feature_path(root: Path, split: str, sample_id: str, text: str, dim: int) -> Path:
    name = f"{safe_stem(sample_id)}_{stable_id(split, text, dim, length=12)}.pt"
    return root / split / name


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def save_embedding(path: Path, *, text: str, embedding: torch.Tensor, method: str, split: str) -> None:
    payload = {
        "content_embedding": embedding.float().cpu(),
        "content_text": text,
        "method": method,
        "split": split,
        "embedding_dim": int(embedding.numel()),
    }
    torch.save(payload, path)


def main() -> int:
    args = parse_args()
    if int(args.embedding_dim) <= 0:
        raise ValueError("--embedding-dim must be positive")
    feature_root = Path(args.feature_root).expanduser()
    output = Path(args.output_jsonl).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(output.name + ".tmp")
    done_output = output.with_name(output.name + ".done.json")
    if tmp_output.exists():
        tmp_output.unlink()

    source_keys = split_keys(args.source_text_keys)
    target_keys = split_keys(args.target_text_keys)
    progress_every = max(1, int(args.progress_every))
    stats = {
        "rows": 0,
        "source_embeddings": 0,
        "target_embeddings": 0,
        "missing_source_text": 0,
        "missing_target_text": 0,
    }

    with tmp_output.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(args.input_jsonl):
            if args.max_rows > 0 and stats["rows"] >= args.max_rows:
                break
            row = dict(row)
            sample_id = str(row.get("sample_id") or stats["rows"])
            for split, keys, out_key in (
                ("source", source_keys, "source_content_path"),
                ("target", target_keys, "target_content_path"),
            ):
                if split == "target" and not args.include_target:
                    continue
                text = pick_text(row, keys)
                if text is None:
                    stats[f"missing_{split}_text"] += 1
                    continue
                out_path = feature_path(feature_root, split, sample_id, text, int(args.embedding_dim))
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if args.overwrite or not out_path.exists():
                    if args.method == "text_hash":
                        embedding = text_hash_embedding(text, int(args.embedding_dim))
                    else:  # pragma: no cover - argparse choices prevent this
                        raise ValueError(f"Unsupported method: {args.method}")
                    save_embedding(out_path, text=text, embedding=embedding, method=args.method, split=split)
                row[out_key] = str(out_path.resolve(strict=False))
                stats[f"{split}_embeddings"] += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["rows"] += 1
            if stats["rows"] % progress_every == 0:
                handle.flush()
                print(
                    "processed "
                    f"rows={stats['rows']} source_embeddings={stats['source_embeddings']} "
                    f"target_embeddings={stats['target_embeddings']} "
                    f"missing_source_text={stats['missing_source_text']} "
                    f"missing_target_text={stats['missing_target_text']}",
                    flush=True,
                )
    tmp_output.replace(output)
    summary = {
        "status": "complete",
        "input_jsonl": str(Path(args.input_jsonl).expanduser()),
        "output_jsonl": str(output),
        "feature_root": str(feature_root),
        "method": args.method,
        "embedding_dim": int(args.embedding_dim),
        **stats,
    }
    write_json_atomic(output.with_suffix(".summary.json"), summary)
    write_json_atomic(done_output, summary)
    print(f"wrote rows={stats['rows']} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

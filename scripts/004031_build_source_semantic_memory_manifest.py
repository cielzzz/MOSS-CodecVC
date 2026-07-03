#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SOURCE_ALIASES = (
    "source_semantic_feature_path",
    "source_semantic_features_path",
    "source_hubert_feature_path",
    "source_hubert_features_path",
)
TARGET_ALIASES = (
    "teacher_target_semantic_feature_path",
    "teacher_target_semantic_features_path",
    "target_semantic_feature_path",
    "target_semantic_features_path",
    "target_hubert_feature_path",
    "target_hubert_features_path",
)


def get_value(record: dict[str, Any], aliases: tuple[str, ...]) -> Any | None:
    meta = record.get("moss_codecvc_meta")
    for key in aliases:
        value = record.get(key)
        if value not in (None, ""):
            return value
        if isinstance(meta, dict):
            value = meta.get(key)
            if value not in (None, ""):
                return value
    return None


def exists(path: Any | None) -> bool:
    return bool(path) and Path(str(path)).expanduser().exists()


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalize manifest fields for Ver2.5 SourceSemanticMemory training.")
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--feature-type", default="hubert_continuous")
    ap.add_argument("--feature-dim", type=int, default=768)
    ap.add_argument("--require-source", action="store_true")
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    input_path = Path(args.input_jsonl).expanduser()
    output_path = Path(args.output_jsonl).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    kept = 0
    missing = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if args.max_rows > 0 and rows >= args.max_rows:
                break
            if not line.strip():
                continue
            rows += 1
            record = json.loads(line)
            source_path = get_value(record, SOURCE_ALIASES)
            target_path = get_value(record, TARGET_ALIASES)
            source_ok = exists(source_path)
            if args.require_source and not source_ok:
                missing += 1
                continue
            if source_ok:
                kept += 1
            else:
                missing += 1
            if source_path:
                record["source_semantic_feature_path"] = str(source_path)
            if target_path:
                record["teacher_target_semantic_feature_path"] = str(target_path)
            record["source_semantic_feature_type"] = args.feature_type
            record["source_semantic_feature_dim"] = int(args.feature_dim)
            record["semantic_memory_available"] = bool(source_ok)
            dst.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "input_jsonl": str(input_path),
                "output_jsonl": str(output_path),
                "rows_read": rows,
                "rows_written": rows - missing if args.require_source else rows,
                "semantic_memory_available": kept,
                "missing_source_feature": missing,
                "require_source": bool(args.require_source),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

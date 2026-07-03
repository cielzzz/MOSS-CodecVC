#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moss_codecvc.io_utils import iter_jsonl


TRUE_VALUES = {"1", "true", "yes", "y", "keep"}
FALSE_VALUES = {"0", "false", "no", "n", "drop", "filter", "filtered"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Filter a JSONL manifest to rows whose content_keep flag is true."
    )
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--summary-json", default="")
    ap.add_argument("--keep-key", default="content_keep")
    ap.add_argument(
        "--missing-as",
        choices=("drop", "keep"),
        default="drop",
        help="How to handle rows without the keep key.",
    )
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=100000)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def nested_get(row: dict[str, Any], key: str) -> Any | None:
    if key in row:
        return row.get(key)
    meta = row.get("moss_codecvc_meta")
    if isinstance(meta, dict) and key in meta:
        return meta.get(key)
    return None


def is_keep(value: Any, *, missing_as_keep: bool) -> bool:
    if value is None:
        return bool(missing_as_keep)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return bool(missing_as_keep)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json) if args.summary_json else output_path.with_suffix(output_path.suffix + ".summary.json")

    if not input_path.is_file():
        raise FileNotFoundError(f"input JSONL does not exist: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists, pass --overwrite: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")

    total = 0
    kept = 0
    dropped = 0
    missing = 0
    reasons: Counter[str] = Counter()
    missing_as_keep = args.missing_as == "keep"

    with tmp_path.open("w", encoding="utf-8") as f:
        for row in iter_jsonl(str(input_path)):
            if args.max_rows > 0 and total >= args.max_rows:
                break
            total += 1
            value = nested_get(row, args.keep_key)
            if value is None:
                missing += 1
            keep = is_keep(value, missing_as_keep=missing_as_keep)
            if keep:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
            else:
                dropped += 1
                reason = nested_get(row, "content_filter_reason") or "content_keep_false"
                for item in str(reason).split(","):
                    item = item.strip() or "unknown"
                    reasons[item] += 1
            if args.progress_every > 0 and total % args.progress_every == 0:
                print(f"[content-keep-filter] rows={total} kept={kept} dropped={dropped}", flush=True)

    tmp_path.replace(output_path)
    summary = {
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "keep_key": str(args.keep_key),
        "missing_as": str(args.missing_as),
        "total": int(total),
        "kept": int(kept),
        "dropped": int(dropped),
        "missing_keep_key": int(missing),
        "drop_reason_counts": dict(reasons.most_common()),
    }
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

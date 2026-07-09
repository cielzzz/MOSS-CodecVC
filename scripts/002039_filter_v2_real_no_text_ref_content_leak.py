#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            yield line_no, line, json.loads(line)


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


def normalize_content_text(text: Any) -> str:
    out: list[str] = []
    for ch in str(text or "").lower():
        code = ord(ch)
        if ch.isalnum() or 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF:
            out.append(ch)
    return "".join(out)


def is_ref_content_leak(row: dict[str, Any]) -> bool:
    ref_text = normalize_content_text(nested_get(row, "timbre_ref_text"))
    target_text = normalize_content_text(nested_get(row, "target_text"))
    return bool(ref_text and target_text and ref_text == target_text)


def main() -> int:
    ap = argparse.ArgumentParser(description="Filter v2 no-text rows where timbre_ref_text equals target_text.")
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--dropped-jsonl", default="")
    ap.add_argument("--summary-json", default="")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input_jsonl).expanduser()
    output_path = Path(args.output_jsonl).expanduser()
    dropped_path = Path(args.dropped_jsonl).expanduser() if args.dropped_jsonl else None
    summary_path = Path(args.summary_json).expanduser() if args.summary_json else output_path.with_suffix(output_path.suffix + ".summary.json")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists, pass --overwrite: {output_path}")
    if dropped_path and dropped_path.exists() and not args.overwrite:
        raise FileExistsError(f"dropped output exists, pass --overwrite: {dropped_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if dropped_path:
        dropped_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = output_path.with_name(output_path.name + ".tmp")
    tmp_drop = dropped_path.with_name(dropped_path.name + ".tmp") if dropped_path else None

    stats: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    with tmp_out.open("w", encoding="utf-8") as out:
        drop_handle = tmp_drop.open("w", encoding="utf-8") if tmp_drop else None
        try:
            for line_no, line, row in iter_jsonl(input_path):
                stats["rows"] += 1
                if is_ref_content_leak(row):
                    stats["dropped_ref_content_leak"] += 1
                    if len(examples) < 20:
                        examples.append(
                            {
                                "line_no": line_no,
                                "sample_id": row.get("sample_id"),
                                "timbre_ref_text": nested_get(row, "timbre_ref_text"),
                                "target_text": nested_get(row, "target_text"),
                            }
                        )
                    if drop_handle is not None:
                        drop_handle.write(line if line.endswith("\n") else line + "\n")
                    continue
                stats["kept"] += 1
                out.write(line if line.endswith("\n") else line + "\n")
        finally:
            if drop_handle is not None:
                drop_handle.close()

    tmp_out.replace(output_path)
    if dropped_path and tmp_drop:
        tmp_drop.replace(dropped_path)
    summary = {
        "status": "complete",
        "input_jsonl": str(input_path.resolve(strict=False)),
        "output_jsonl": str(output_path.resolve(strict=False)),
        "dropped_jsonl": "" if dropped_path is None else str(dropped_path.resolve(strict=False)),
        "rows": int(stats["rows"]),
        "kept": int(stats["kept"]),
        "dropped_ref_content_leak": int(stats["dropped_ref_content_leak"]),
        "drop_rate": float(stats["dropped_ref_content_leak"]) / float(max(1, stats["rows"])),
        "examples": examples,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_path.with_name(output_path.name + ".done.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

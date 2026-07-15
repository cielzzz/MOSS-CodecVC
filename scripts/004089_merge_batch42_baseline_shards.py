#!/usr/bin/env python3
"""Audit and merge sharded Batch-42 baseline inference manifests.

The merged all-status manifest is the inference ledger.  A second JSONL keeps
only successful, existing generated WAVs and is the canonical input for
``004082_run_unified_vc_eval.py``.  Failed cases remain visible in the ledger
instead of silently disappearing from paper-facing denominators.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


SUCCESS_STATUSES = {"ok", "skipped_existing"}


def read_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: row must be an object")
                row = dict(row)
                row["_merge_source"] = str(path.resolve())
                row["_merge_source_line"] = line_number
                rows.append(row)
    return rows


def output_wav_ready(row: dict[str, Any]) -> bool:
    path = Path(str(row.get("generated_audio") or ""))
    return path.is_file() and path.stat().st_size >= 44


def ordered(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("input_index", 10**18)),
            str(row.get("case_uid") or ""),
        ),
    )


def audit_rows(
    rows: list[dict[str, Any]],
    *,
    expected_shards: int,
    expected_cases: int | None,
    system_id: str | None,
    test_set_id: str | None,
) -> dict[str, Any]:
    if expected_shards < 1:
        raise ValueError("expected_shards must be >= 1")
    if expected_cases is not None and len(rows) != expected_cases:
        raise ValueError(f"expected {expected_cases} rows, got {len(rows)}")
    case_uids = [str(row.get("case_uid") or "") for row in rows]
    case_ids = [str(row.get("case_id") or "") for row in rows]
    if any(not value for value in case_uids):
        raise ValueError("one or more rows are missing case_uid")
    if any(not value for value in case_ids):
        raise ValueError("one or more rows are missing case_id")
    if len(set(case_uids)) != len(case_uids):
        duplicates = [key for key, count in Counter(case_uids).items() if count > 1]
        raise ValueError(f"duplicate case_uid values: {duplicates[:10]}")
    if len(set(case_ids)) != len(case_ids):
        duplicates = [key for key, count in Counter(case_ids).items() if count > 1]
        raise ValueError(f"duplicate case_id values: {duplicates[:10]}")

    observed_shards: set[int] = set()
    for row in rows:
        provenance = row.get("provenance") or {}
        num_shards = int(provenance.get("num_shards", -1))
        shard_index = int(provenance.get("shard_index", -1))
        input_index = int(row.get("input_index", -1))
        if num_shards != expected_shards:
            raise ValueError(
                f"{row['case_id']}: provenance num_shards={num_shards}, expected {expected_shards}"
            )
        if not 0 <= shard_index < expected_shards:
            raise ValueError(f"{row['case_id']}: invalid shard_index={shard_index}")
        if input_index < 0 or input_index % expected_shards != shard_index:
            raise ValueError(
                f"{row['case_id']}: input_index={input_index} is not assigned to shard {shard_index}"
            )
        observed_shards.add(shard_index)
        if system_id is not None and row.get("system_id") != system_id:
            raise ValueError(
                f"{row['case_id']}: system_id={row.get('system_id')!r}, expected {system_id!r}"
            )
        if test_set_id is not None and row.get("test_set_id") != test_set_id:
            raise ValueError(
                f"{row['case_id']}: test_set_id={row.get('test_set_id')!r}, expected {test_set_id!r}"
            )
    expected_indices = set(range(expected_shards))
    if observed_shards != expected_indices:
        raise ValueError(
            f"incomplete shard coverage: observed={sorted(observed_shards)} "
            f"expected={sorted(expected_indices)}"
        )

    statuses = Counter(str(row.get("status") or "missing") for row in rows)
    success = [
        row
        for row in rows
        if str(row.get("status") or "") in SUCCESS_STATUSES and output_wav_ready(row)
    ]
    success_status_but_missing_wav = [
        row
        for row in rows
        if str(row.get("status") or "") in SUCCESS_STATUSES and not output_wav_ready(row)
    ]
    if success_status_but_missing_wav:
        raise ValueError(
            "successful manifest row(s) have missing/empty generated WAV: "
            + ", ".join(str(row.get("case_id")) for row in success_status_but_missing_wav[:10])
        )
    return {
        "rows": len(rows),
        "unique_case_ids": len(set(case_ids)),
        "unique_case_uids": len(set(case_uids)),
        "expected_shards": expected_shards,
        "observed_shards": sorted(observed_shards),
        "status_counts": dict(sorted(statuses.items())),
        "successful_rows": len(success),
        "failed_or_input_error_rows": len(rows) - len(success),
        "all_ok": len(success) == len(rows),
    }


def without_merge_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_merge_")}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--merged-manifest", type=Path, required=True)
    parser.add_argument("--successful-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--expected-cases", type=int)
    parser.add_argument("--system-id")
    parser.add_argument("--test-set-id")
    parser.add_argument("--require-all-ok", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = sorted(path.expanduser().resolve() for path in args.input)
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate --input manifest path")
    rows = ordered(read_rows(paths))
    audit = audit_rows(
        rows,
        expected_shards=args.expected_shards,
        expected_cases=args.expected_cases,
        system_id=args.system_id,
        test_set_id=args.test_set_id,
    )
    clean_rows = [without_merge_metadata(row) for row in rows]
    successful = [
        row
        for row in clean_rows
        if str(row.get("status") or "") in SUCCESS_STATUSES and output_wav_ready(row)
    ]
    summary = {
        "schema_version": "moss_codecvc.baseline_vc_infer_merge.v1",
        "inputs": [str(path) for path in paths],
        "merged_manifest": str(args.merged_manifest.resolve()),
        "successful_jsonl": str(args.successful_jsonl.resolve()),
        "system_id": args.system_id,
        "test_set_id": args.test_set_id,
        **audit,
    }
    atomic_jsonl(args.merged_manifest, clean_rows)
    atomic_jsonl(args.successful_jsonl, successful)
    atomic_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if args.require_all_ok and not audit["all_ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

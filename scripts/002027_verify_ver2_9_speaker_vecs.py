#!/usr/bin/env python3
"""Verify ver2.9 prepared manifests with precomputed speaker vectors."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SPLITS = (
    "no_text.train.jsonl",
    "text.train.jsonl",
    "no_text.valid.jsonl",
    "text.valid.jsonl",
    "no_text.seen_valid.jsonl",
    "text.seen_valid.jsonl",
    "no_text.unseen_valid.jsonl",
    "text.unseen_valid.jsonl",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared-dir", type=Path, required=True)
    ap.add_argument("--input-prepared-dir", type=Path, default=None, help="Optional source dir for expected row counts.")
    ap.add_argument("--splits", nargs="*", default=list(DEFAULT_SPLITS))
    ap.add_argument("--expected-dim", type=int, default=512)
    ap.add_argument("--sample-per-split", type=int, default=200, help="0 checks all rows.")
    ap.add_argument("--output-json", type=Path, default=None)
    return ap.parse_args()


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            if line.strip():
                yield idx, json.loads(line)


def should_check(row_idx: int, total_rows: int, sample_per_split: int) -> bool:
    if sample_per_split <= 0 or total_rows <= sample_per_split:
        return True
    if row_idx <= 3 or row_idx == total_rows:
        return True
    stride = max(1, total_rows // max(1, sample_per_split))
    return row_idx % stride == 0


def verify_split(
    prepared_dir: Path,
    split: str,
    *,
    input_prepared_dir: Path | None,
    expected_dim: int,
    sample_per_split: int,
) -> dict[str, Any]:
    path = prepared_dir / split
    out: dict[str, Any] = {"split": split, "path": str(path), "exists": path.exists()}
    if not path.exists():
        out["ok"] = False
        out["error"] = "missing manifest"
        return out
    total_rows = count_rows(path)
    out["rows"] = total_rows
    if input_prepared_dir is not None:
        input_path = input_prepared_dir / split
        out["expected_rows"] = count_rows(input_path) if input_path.exists() else None
        out["row_count_matches_input"] = out["expected_rows"] == total_rows
    missing_field = 0
    missing_file = 0
    bad_shape = 0
    bad_norm = 0
    checked = 0
    norm_min = math.inf
    norm_max = -math.inf
    examples: list[dict[str, Any]] = []
    for row_idx, row in iter_jsonl(path):
        speaker_vec_path = row.get("speaker_vec_path")
        if not speaker_vec_path:
            missing_field += 1
            if len(examples) < 5:
                examples.append({"row": row_idx, "error": "missing speaker_vec_path"})
            continue
        vec_path = Path(str(speaker_vec_path))
        if not vec_path.exists():
            missing_file += 1
            if len(examples) < 5:
                examples.append({"row": row_idx, "error": "missing speaker vec file", "path": str(vec_path)})
            continue
        if not should_check(row_idx, total_rows, sample_per_split):
            continue
        checked += 1
        try:
            vec = np.load(vec_path)
        except Exception as exc:  # noqa: BLE001
            bad_shape += 1
            if len(examples) < 5:
                examples.append({"row": row_idx, "error": f"load failed: {exc}", "path": str(vec_path)})
            continue
        if tuple(vec.shape) != (int(expected_dim),):
            bad_shape += 1
            if len(examples) < 5:
                examples.append({"row": row_idx, "error": "bad shape", "shape": list(vec.shape), "path": str(vec_path)})
            continue
        norm = float(np.linalg.norm(vec.astype(np.float32)))
        norm_min = min(norm_min, norm)
        norm_max = max(norm_max, norm)
        if not math.isfinite(norm) or abs(norm - 1.0) > 1.0e-3:
            bad_norm += 1
            if len(examples) < 5:
                examples.append({"row": row_idx, "error": "bad norm", "norm": norm, "path": str(vec_path)})
    out.update(
        {
            "checked_vectors": checked,
            "missing_speaker_vec_path": missing_field,
            "missing_speaker_vec_file": missing_file,
            "bad_shape": bad_shape,
            "bad_norm": bad_norm,
            "norm_min": None if norm_min == math.inf else norm_min,
            "norm_max": None if norm_max == -math.inf else norm_max,
            "examples": examples,
        }
    )
    out["ok"] = (
        missing_field == 0
        and missing_file == 0
        and bad_shape == 0
        and bad_norm == 0
        and (input_prepared_dir is None or bool(out.get("row_count_matches_input", False)))
    )
    return out


def main() -> int:
    args = parse_args()
    prepared_dir = args.prepared_dir.expanduser().resolve()
    input_prepared_dir = args.input_prepared_dir.expanduser().resolve() if args.input_prepared_dir else None
    splits = [
        verify_split(
            prepared_dir,
            split,
            input_prepared_dir=input_prepared_dir,
            expected_dim=args.expected_dim,
            sample_per_split=args.sample_per_split,
        )
        for split in args.splits
    ]
    manifest_paths: list[str] = []
    for split in args.splits:
        path = prepared_dir / split
        if not path.exists():
            continue
        for _row_idx, row in iter_jsonl(path):
            speaker_vec_path = row.get("speaker_vec_path")
            if speaker_vec_path:
                manifest_paths.append(str(Path(str(speaker_vec_path)).resolve()))
    path_counts = Counter(manifest_paths)
    unique_manifest_paths = set(path_counts)
    npy_files = {str(path.resolve()) for path in (prepared_dir / "speaker_vecs").glob("*.npy")}
    duplicate_paths = sum(1 for count in path_counts.values() if count > 1)
    duplicate_path_rows = sum(count - 1 for count in path_counts.values() if count > 1)
    missing_unique_files = sorted(unique_manifest_paths - npy_files)[:20]
    orphan_npy_files = sorted(npy_files - unique_manifest_paths)[:20]
    global_vector_count_ok = (
        len(manifest_paths) == len(unique_manifest_paths) == len(npy_files)
        and not missing_unique_files
        and not orphan_npy_files
    )
    summary = {
        "prepared_dir": str(prepared_dir),
        "input_prepared_dir": str(input_prepared_dir) if input_prepared_dir else None,
        "expected_dim": int(args.expected_dim),
        "sample_per_split": int(args.sample_per_split),
        "splits": splits,
        "total_rows": sum(int(item.get("rows", 0) or 0) for item in splits),
        "manifest_speaker_vec_path_rows": len(manifest_paths),
        "unique_manifest_speaker_vec_paths": len(unique_manifest_paths),
        "npy_file_count": len(npy_files),
        "manifest_rows_match_npy_count": len(manifest_paths) == len(npy_files),
        "unique_manifest_paths_match_npy_count": len(unique_manifest_paths) == len(npy_files),
        "global_vector_count_ok": global_vector_count_ok,
        "duplicate_speaker_vec_paths": duplicate_paths,
        "duplicate_speaker_vec_path_extra_rows": duplicate_path_rows,
        "missing_unique_speaker_vec_files": missing_unique_files,
        "orphan_npy_files": orphan_npy_files,
        "ok": all(bool(item.get("ok", False)) for item in splits) and global_vector_count_ok,
    }
    output_json = args.output_json or (prepared_dir / "speaker_vec_verify_summary.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {output_json}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

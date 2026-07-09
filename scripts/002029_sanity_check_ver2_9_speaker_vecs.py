#!/usr/bin/env python3
"""Sanity check WavLM-SV speaker vectors by target speaker grouping."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import random
from typing import Any

import numpy as np


SPEAKER_KEYS = (
    "target_speaker_id",
    "target_pseudo_speaker_id",
    "target_speaker_pseudo_id",
    "timbre_ref_speaker_id",
    "timbre_ref_pseudo_speaker_id",
    "timbre_ref_speaker_pseudo_id",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared-dir", type=Path, required=True)
    ap.add_argument("--split", default="no_text.train.jsonl")
    ap.add_argument("--sample-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--min-delta", type=float, default=0.15)
    ap.add_argument(
        "--speaker-balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample paired rows per speaker so same-speaker cosine is measurable.",
    )
    ap.add_argument("--rows-per-speaker", type=int, default=2)
    ap.add_argument("--output-json", type=Path, default=None)
    return ap.parse_args()


def record_value(record: dict[str, Any], key: str) -> Any | None:
    if record.get(key) not in (None, ""):
        return record[key]
    meta = record.get("moss_codecvc_meta")
    if isinstance(meta, dict) and meta.get(key) not in (None, ""):
        return meta[key]
    return None


def speaker_id(record: dict[str, Any]) -> str | None:
    for key in SPEAKER_KEYS:
        value = record_value(record, key)
        if value not in (None, ""):
            return str(value)
    return None


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            yield line_no, json.loads(line)


def reservoir_sample(path: Path, sample_size: int, seed: int) -> tuple[int, list[dict[str, Any]]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    usable = 0
    for line_no, row in iter_jsonl(path):
        spk = speaker_id(row)
        vec_path = row.get("speaker_vec_path")
        if not spk or not vec_path:
            continue
        usable += 1
        item = {
            "line_no": line_no,
            "sample_id": row.get("sample_id") or row.get("pair_id") or row.get("id"),
            "target_speaker_id": spk,
            "speaker_vec_path": str(vec_path),
        }
        if len(rows) < sample_size:
            rows.append(item)
            continue
        idx = rng.randrange(usable)
        if idx < sample_size:
            rows[idx] = item
    return usable, rows


def speaker_balanced_sample(
    path: Path,
    sample_size: int,
    seed: int,
    rows_per_speaker: int,
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    usable = 0
    for line_no, row in iter_jsonl(path):
        spk = speaker_id(row)
        vec_path = row.get("speaker_vec_path")
        if not spk or not vec_path:
            continue
        usable += 1
        groups[spk].append(
            {
                "line_no": line_no,
                "sample_id": row.get("sample_id") or row.get("pair_id") or row.get("id"),
                "target_speaker_id": spk,
                "speaker_vec_path": str(vec_path),
            }
        )

    rows_per_speaker = max(2, int(rows_per_speaker))
    eligible = [spk for spk, rows in groups.items() if len(rows) >= rows_per_speaker]
    rng.shuffle(eligible)
    target_speakers = min(len(eligible), max(1, sample_size // rows_per_speaker))
    sample_rows: list[dict[str, Any]] = []
    selected_lines: set[int] = set()
    for spk in eligible[:target_speakers]:
        chosen = rng.sample(groups[spk], rows_per_speaker)
        sample_rows.extend(chosen)
        selected_lines.update(int(row["line_no"]) for row in chosen)

    if len(sample_rows) < sample_size:
        pool = [
            row
            for rows in groups.values()
            for row in rows
            if int(row["line_no"]) not in selected_lines
        ]
        fill = min(sample_size - len(sample_rows), len(pool))
        if fill > 0:
            sample_rows.extend(rng.sample(pool, fill))

    rng.shuffle(sample_rows)
    meta = {
        "sampling_strategy": "speaker_balanced",
        "full_speaker_count": len(groups),
        "eligible_speaker_count": len(eligible),
        "rows_per_speaker": rows_per_speaker,
        "target_speakers": target_speakers,
    }
    return usable, sample_rows, meta


def load_unit_vec(path: Path) -> np.ndarray:
    vec = np.load(path).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"bad vector norm for {path}: {norm}")
    return vec / norm


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def main() -> int:
    args = parse_args()
    prepared_dir = args.prepared_dir.expanduser().resolve()
    manifest = prepared_dir / args.split
    if not manifest.exists():
        raise SystemExit(f"missing split manifest: {manifest}")
    output_json = args.output_json or (prepared_dir / "speaker_vec_sanity_no_text_train_100.json")

    if bool(args.speaker_balanced):
        usable_rows, sample_rows, sampling_meta = speaker_balanced_sample(
            manifest,
            int(args.sample_size),
            int(args.seed),
            int(args.rows_per_speaker),
        )
    else:
        usable_rows, sample_rows = reservoir_sample(manifest, int(args.sample_size), int(args.seed))
        sampling_meta = {"sampling_strategy": "reservoir"}
    vecs: list[np.ndarray] = []
    kept_rows: list[dict[str, Any]] = []
    load_errors: list[dict[str, Any]] = []
    for row in sample_rows:
        path = Path(str(row["speaker_vec_path"]))
        try:
            vecs.append(load_unit_vec(path))
            kept_rows.append(row)
        except Exception as exc:  # noqa: BLE001
            load_errors.append({"sample_id": row.get("sample_id"), "path": str(path), "error": str(exc)})

    same_values: list[float] = []
    cross_values: list[float] = []
    for left_idx, right_idx in itertools.combinations(range(len(kept_rows)), 2):
        sim = float(np.dot(vecs[left_idx], vecs[right_idx]))
        if kept_rows[left_idx]["target_speaker_id"] == kept_rows[right_idx]["target_speaker_id"]:
            same_values.append(sim)
        else:
            cross_values.append(sim)

    same_mean = mean_or_none(same_values)
    cross_mean = mean_or_none(cross_values)
    delta = None if same_mean is None or cross_mean is None else float(same_mean - cross_mean)
    speaker_counts = Counter(str(row["target_speaker_id"]) for row in kept_rows)
    grouped_examples: dict[str, list[str | None]] = defaultdict(list)
    for row in kept_rows:
        spk = str(row["target_speaker_id"])
        if len(grouped_examples[spk]) < 5:
            grouped_examples[spk].append(row.get("sample_id"))

    summary = {
        "prepared_dir": str(prepared_dir),
        "split": args.split,
        "manifest": str(manifest),
        "seed": int(args.seed),
        "sample_size_requested": int(args.sample_size),
        "usable_manifest_rows": usable_rows,
        "sample_rows": len(sample_rows),
        **sampling_meta,
        "loaded_rows": len(kept_rows),
        "speaker_count": len(speaker_counts),
        "speaker_group_size_min": min(speaker_counts.values()) if speaker_counts else None,
        "speaker_group_size_max": max(speaker_counts.values()) if speaker_counts else None,
        "speaker_group_size_mean": float(np.mean(list(speaker_counts.values()))) if speaker_counts else None,
        "same_pair_count": len(same_values),
        "cross_pair_count": len(cross_values),
        "same_speaker_cos_mean": same_mean,
        "cross_speaker_cos_mean": cross_mean,
        "delta_same_minus_cross": delta,
        "min_delta": float(args.min_delta),
        "pass": bool(delta is not None and delta >= float(args.min_delta)),
        "load_errors": load_errors[:20],
        "speaker_group_counts_top20": speaker_counts.most_common(20),
        "speaker_group_examples_top20": {spk: grouped_examples[spk] for spk, _count in speaker_counts.most_common(20)},
        "sampled_rows": kept_rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {output_json}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

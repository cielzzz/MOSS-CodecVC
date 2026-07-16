#!/usr/bin/env python
"""Compute per-channel statistics for the verified ver3.1 zq targets.

The input manifest contains one record per saved ``[D,T]`` ``.npy`` target.
This command reads each target exactly once, accumulates float64 sums and
squared sums per channel, and atomically publishes:

* ``channel_stats.pt``: tensor payload consumed by training/inference;
* ``channel_stats.json``: human/audit-friendly companion;
* ``channel_stats.progress.pt``: restart checkpoint (removed after success).

The command is intentionally CPU-only.  It does not instantiate the MOSS
codec, copy the 100+ GiB target dataset, or alter the DDLFM training loop.
Use ``--max-rows`` only for a clearly marked partial smoke run; a partial
payload is never labelled ``completed``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moss_codecvc.audio.zq_normalization import (
    CHANNEL_STATS_SCHEMA,
    DEFAULT_STD_FLOOR,
    ZQChannelStatsAccumulator,
    atomic_json_save,
    atomic_torch_save,
    sha256_file,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "prepared/zq_targets_v1/manifest.jsonl"),
        help="zq target manifest with one output_path per line",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "prepared/zq_targets_v1/channel_stats.pt"),
        help="canonical torch payload path",
    )
    parser.add_argument(
        "--audit-output",
        default="",
        help="JSON audit path; default is <output stem>.json",
    )
    parser.add_argument(
        "--progress-path",
        default="",
        help="restart checkpoint; default is <output stem>.progress.pt",
    )
    parser.add_argument("--latent-dim", type=int, default=768)
    parser.add_argument("--expected-dtype", default="float32")
    parser.add_argument("--expected-rate-hz", type=float, default=12.5)
    parser.add_argument("--rate-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--std-floor", type=float, default=DEFAULT_STD_FLOOR)
    parser.add_argument("--chunk-frames", type=int, default=8192)
    parser.add_argument(
        "--checkpoint-every-rows",
        type=int,
        default=1000,
        help="write progress checkpoint after this many completed rows",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="partial smoke cap; 0 means all rows")
    parser.add_argument("--resume", action="store_true", help="resume from progress checkpoint")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing final payload")
    return parser.parse_args(argv)


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _default_sibling(path: Path, suffix: str) -> Path:
    # Path.with_suffix() turns channel_stats.pt into channel_stats.json, while
    # preserving a useful name for custom output paths.
    return path.with_suffix(suffix)


def _iter_manifest_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"manifest line {line_no} is not an object")
            yield line_no, row


def _validate_row(
    row: dict[str, Any],
    *,
    line_no: int,
    latent_dim: int,
    expected_rate_hz: float,
    rate_tolerance: float,
) -> Path:
    output_value = row.get("output_path")
    if not output_value:
        raise ValueError(f"manifest line {line_no} has no output_path")
    path = _path(str(output_value))
    if not path.is_file():
        raise FileNotFoundError(f"manifest line {line_no} target does not exist: {path}")
    row_dim = row.get("latent_dim")
    if row_dim not in (None, "") and int(row_dim) != int(latent_dim):
        raise ValueError(
            f"manifest line {line_no} latent_dim={row_dim} does not match {latent_dim}"
        )
    row_rate = row.get("frame_rate_hz")
    if row_rate not in (None, "") and abs(float(row_rate) - float(expected_rate_hz)) > float(rate_tolerance):
        raise ValueError(
            f"manifest line {line_no} frame_rate_hz={row_rate} does not match {expected_rate_hz}"
        )
    return path


def _load_target(
    path: Path,
    *,
    line_no: int,
    latent_dim: int,
    expected_frames: int | None,
    expected_dtype: str,
) -> np.ndarray:
    # mmap_mode avoids a second resident copy for large targets.  The
    # accumulator itself consumes bounded frame chunks.
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.ndim != 2 or int(value.shape[0]) != int(latent_dim):
        raise ValueError(
            f"manifest line {line_no} target {path} must be [{latent_dim},T], got {value.shape}"
        )
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"manifest line {line_no} target {path} is not floating point: {value.dtype}")
    if str(value.dtype) != str(expected_dtype):
        raise TypeError(
            f"manifest line {line_no} target {path} has dtype={value.dtype}, expected {expected_dtype}"
        )
    if int(value.shape[1]) <= 0:
        raise ValueError(f"manifest line {line_no} target {path} has no frames")
    if expected_frames not in (None, "") and int(expected_frames) != int(value.shape[1]):
        raise ValueError(
            f"manifest line {line_no} num_frames={expected_frames} disagrees with {value.shape[1]} for {path}"
        )
    return value


def _write_progress(
    path: Path,
    *,
    manifest: Path,
    manifest_sha256: str,
    manifest_size_bytes: int,
    latent_dim: int,
    std_floor: float,
    chunk_frames: int,
    expected_dtype: str,
    next_line_no: int,
    accumulator: ZQChannelStatsAccumulator,
) -> None:
    payload = {
        "schema": f"{CHANNEL_STATS_SCHEMA}.progress",
        "created_at_unix": time.time(),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "manifest_size_bytes": int(manifest_size_bytes),
        "latent_dim": int(latent_dim),
        "std_floor": float(std_floor),
        "chunk_frames": int(chunk_frames),
        "expected_dtype": str(expected_dtype),
        "next_line_no": int(next_line_no),
        "accumulator": accumulator.state_dict(),
    }
    atomic_torch_save(payload, path)


def _load_progress(
    path: Path,
    *,
    manifest: Path,
    manifest_sha256: str,
    manifest_size_bytes: int,
    latent_dim: int,
    std_floor: float,
    chunk_frames: int,
    expected_dtype: str,
) -> tuple[int, ZQChannelStatsAccumulator]:
    import torch

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not str(payload.get("schema", "")).endswith(".progress"):
        raise ValueError(f"invalid stats progress checkpoint: {path}")
    checks = {
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "manifest_size_bytes": int(manifest_size_bytes),
        "latent_dim": int(latent_dim),
        "std_floor": float(std_floor),
        "chunk_frames": int(chunk_frames),
        "expected_dtype": str(expected_dtype),
    }
    for key, expected in checks.items():
        actual = payload.get(key)
        if isinstance(expected, float):
            if actual is None or abs(float(actual) - expected) > 1.0e-15:
                raise ValueError(f"progress {key} mismatch: {actual!r} != {expected!r}")
        elif actual != expected:
            raise ValueError(f"progress {key} mismatch: {actual!r} != {expected!r}")
    next_line_no = int(payload.get("next_line_no", 1))
    if next_line_no < 1:
        raise ValueError(f"invalid next_line_no={next_line_no}")
    accumulator = ZQChannelStatsAccumulator.from_state_dict(payload["accumulator"])
    if int(accumulator.latent_dim) != int(latent_dim):
        raise ValueError("progress accumulator latent_dim mismatch")
    return next_line_no, accumulator


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _path(args.manifest)
    output = _path(args.output)
    audit_output = _path(args.audit_output) if args.audit_output else _default_sibling(output, ".json")
    progress = _path(args.progress_path) if args.progress_path else _default_sibling(output, ".progress.pt")
    latent_dim = int(args.latent_dim)
    expected_rate_hz = float(args.expected_rate_hz)
    rate_tolerance = float(args.rate_tolerance)
    std_floor = float(args.std_floor)
    expected_dtype = str(args.expected_dtype)
    chunk_frames = int(args.chunk_frames)
    checkpoint_every = int(args.checkpoint_every_rows)
    max_rows = int(args.max_rows)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if latent_dim <= 0 or chunk_frames <= 0 or checkpoint_every <= 0:
        raise ValueError("latent_dim, chunk_frames and checkpoint-every-rows must be positive")
    if not np.isfinite(std_floor) or std_floor <= 0:
        raise ValueError("std-floor must be finite and positive")
    if max_rows < 0:
        raise ValueError("max-rows cannot be negative")
    try:
        expected_dtype = str(np.dtype(expected_dtype))
    except TypeError as exc:
        raise ValueError(f"invalid expected-dtype={expected_dtype!r}") from exc
    if output.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(f"output exists; pass --overwrite or --resume: {output}")
    if progress.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(
            f"progress checkpoint exists; pass --resume or --overwrite: {progress}"
        )
    if args.overwrite and not args.resume:
        progress.unlink(missing_ok=True)

    manifest_stat = manifest.stat()
    manifest_sha256 = sha256_file(manifest)
    completion_path = manifest.parent / "COMPLETED.json"
    completion: dict[str, Any] | None = None
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if not isinstance(completion, dict) or completion.get("status") != "completed":
            raise ValueError(f"invalid zq dataset completion marker: {completion_path}")
        if int(completion.get("latent_dim", latent_dim)) != latent_dim:
            raise ValueError(f"completion latent_dim disagrees with --latent-dim: {completion_path}")
        if str(completion.get("dtype", expected_dtype)) != expected_dtype:
            raise ValueError(f"completion dtype disagrees with --expected-dtype: {completion_path}")
    if args.resume:
        if not progress.is_file():
            raise FileNotFoundError(f"--resume requested but progress checkpoint is missing: {progress}")
        next_line_no, accumulator = _load_progress(
            progress,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            manifest_size_bytes=manifest_stat.st_size,
            latent_dim=latent_dim,
            std_floor=std_floor,
            chunk_frames=chunk_frames,
            expected_dtype=expected_dtype,
        )
    else:
        next_line_no = 1
        accumulator = ZQChannelStatsAccumulator(latent_dim)

    started = time.time()
    rows_seen = int(accumulator.row_count)
    last_line_no = next_line_no - 1
    for line_no, row in _iter_manifest_rows(manifest):
        if line_no < next_line_no:
            continue
        target = _validate_row(
            row,
            line_no=line_no,
            latent_dim=latent_dim,
            expected_rate_hz=expected_rate_hz,
            rate_tolerance=rate_tolerance,
        )
        value = _load_target(
            target,
            line_no=line_no,
            latent_dim=latent_dim,
            expected_frames=row.get("num_frames"),
            expected_dtype=expected_dtype,
        )
        accumulator.update(value, chunk_frames=chunk_frames)
        rows_seen += 1
        last_line_no = line_no
        next_line_no = line_no + 1
        if rows_seen % checkpoint_every == 0:
            _write_progress(
                progress,
                manifest=manifest,
                manifest_sha256=manifest_sha256,
                manifest_size_bytes=manifest_stat.st_size,
                latent_dim=latent_dim,
                std_floor=std_floor,
                chunk_frames=chunk_frames,
                expected_dtype=expected_dtype,
                next_line_no=next_line_no,
                accumulator=accumulator,
            )
        if max_rows and rows_seen >= max_rows:
            break

    # A capped run is deliberately marked partial.  For a full run, reaching
    # EOF is required; max_rows only limits the smoke path above.
    reached_eof = not max_rows or rows_seen < max_rows
    partial = bool(max_rows and rows_seen >= max_rows)
    if max_rows and rows_seen < max_rows:
        # The manifest ended before the requested cap: this is still complete.
        partial = False
    if not partial and completion is not None:
        expected_rows = int(completion.get("total_utterances", 0))
        expected_frames = int(completion.get("total_frames", 0))
        if accumulator.row_count != expected_rows or accumulator.frame_count != expected_frames:
            raise ValueError(
                "full channel-stat pass disagrees with zq COMPLETED.json: "
                f"rows={accumulator.row_count}/{expected_rows}, "
                f"frames={accumulator.frame_count}/{expected_frames}"
            )
    metadata = {
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "manifest_size_bytes": int(manifest_stat.st_size),
        "manifest_last_line_no": int(last_line_no),
        "expected_rate_hz": expected_rate_hz,
        "frame_rate_hz": expected_rate_hz,
        "chunk_frames": chunk_frames,
        "source_dtype": expected_dtype,
        "stats_accumulator": "float64_sum_and_sum_squares",
        "dataset_completion": str(completion_path) if completion is not None else None,
        "dataset_completion_sha256": sha256_file(completion_path) if completion is not None else None,
        "rows_requested": int(max_rows),
        "elapsed_sec": float(time.time() - started),
        "status_note": "partial smoke run; do not use for training" if partial else "full manifest pass",
    }
    payload = accumulator.finalize(std_floor=std_floor, metadata=metadata, partial=partial)
    atomic_torch_save(payload, output)
    output_sha256 = sha256_file(output)
    atomic_json_save(
        {
            **payload,
            "channel_stats_path": str(output),
            "channel_stats_sha256": output_sha256,
        },
        audit_output,
    )
    if not partial:
        progress.unlink(missing_ok=True)
    else:
        _write_progress(
            progress,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            manifest_size_bytes=manifest_stat.st_size,
            latent_dim=latent_dim,
            std_floor=std_floor,
            chunk_frames=chunk_frames,
            expected_dtype=expected_dtype,
            next_line_no=next_line_no,
            accumulator=accumulator,
        )
    summary = {
        "status": payload["status"],
        "output": str(output),
        "audit_output": str(audit_output),
        "progress": str(progress) if progress.exists() else None,
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "output_sha256": output_sha256,
        "rows": int(accumulator.row_count),
        "frames": int(accumulator.frame_count),
        "latent_dim": latent_dim,
        "std_floor": std_floor,
        "mean_min": float(payload["mean"].min().item()),
        "mean_max": float(payload["mean"].max().item()),
        "raw_std_min": float(payload["raw_std"].min().item()),
        "raw_std_max": float(payload["raw_std"].max().item()),
        "std_floor_count": int((payload["raw_std"] < float(std_floor)).sum().item()),
        "reached_eof": bool(reached_eof),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

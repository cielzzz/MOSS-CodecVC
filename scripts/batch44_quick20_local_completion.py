#!/usr/bin/env python3
"""Capture and finalize fail-closed local Batch-44 quick20 provenance."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import socket
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


COMPLETION_SCHEMA = "moss_codecvc.batch44_v1_quick20_completion.v2"
MARKER_SCHEMA = "moss_codecvc.batch44_v1_quick20_complete_marker.v2"
RUNTIME_SCHEMA = "moss_codecvc.batch44_v1_quick20_local_runtime.v1"
GPU_MODEL = "NVIDIA GeForce RTX 4090"
HOST_PREFIX = "xyzhang-dev--"
TRAINING_JOBS = {
    "r3": "job-2b91d332-d500-4279-84f9-0a6a81a376aa",
    "r5": "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c",
}
LABELS = {"r3": "ver2_9_5_final_r3", "r5": "ver2_9_5_final_r5"}
CHECKPOINT_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "README.md",
    "timbre_memory_adapter.pt",
    "timbre_memory_config.json",
)


def sha256_file(path: Path) -> str:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing/empty provenance input: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def parse_csv_line(line: str, expected_fields: int) -> list[str]:
    fields = [field.strip() for field in next(csv.reader([line]))]
    if len(fields) != expected_fields:
        raise ValueError(f"unexpected nvidia-smi row: {line!r}")
    return fields


def capture_runtime(
    *,
    output: Path,
    runner: Path,
    common_library: Path,
    completion_helper: Path,
    step: int,
    r3_checkpoint: Path,
    r5_checkpoint: Path,
    max_initial_memory_mib: int,
    allow_any_host: bool = False,
) -> dict[str, Any]:
    hostname = socket.gethostname()
    if not allow_any_host and not hostname.startswith(HOST_PREFIX):
        raise ValueError(
            f"local quick20 is restricted to {HOST_PREFIX}*; got hostname={hostname!r}"
        )
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if query.returncode != 0:
        raise ValueError(f"nvidia-smi GPU query failed: {query.stderr.strip()}")
    rows = [line for line in query.stdout.splitlines() if line.strip()]
    if len(rows) != 2:
        raise ValueError(f"local quick20 requires exactly two GPUs, got {len(rows)}")
    gpus: list[dict[str, Any]] = []
    for line in rows:
        index, uuid, name, total, used, driver = parse_csv_line(line, 6)
        gpu = {
            "index": int(index),
            "uuid": uuid,
            "name": name,
            "memory_total_mib": int(total),
            "memory_used_mib_at_start": int(used),
            "driver_version": driver,
        }
        gpus.append(gpu)
    gpus.sort(key=lambda row: row["index"])
    if [row["index"] for row in gpus] != [0, 1]:
        raise ValueError(f"local quick20 requires GPU indices [0, 1], got {gpus}")
    if any(row["name"] != GPU_MODEL for row in gpus):
        raise ValueError(f"local quick20 requires two {GPU_MODEL} GPUs, got {gpus}")
    if any(row["memory_total_mib"] < 48_000 for row in gpus):
        raise ValueError(f"local quick20 GPU memory contract failed: {gpus}")
    busy = [
        row
        for row in gpus
        if row["memory_used_mib_at_start"] > max_initial_memory_mib
    ]
    if busy:
        raise ValueError(
            "local quick20 refuses busy GPUs; "
            f"limit={max_initial_memory_mib} MiB rows={busy}"
        )
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    checkpoints: dict[str, Any] = {}
    for arm, checkpoint in (("r3", r3_checkpoint), ("r5", r5_checkpoint)):
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_dir() or checkpoint.name != f"step-{step}":
            raise ValueError(f"invalid {arm} checkpoint binding: {checkpoint}")
        checkpoints[arm] = {
            "path": str(checkpoint),
            "step": step,
            "training_job_id": TRAINING_JOBS[arm],
            "files": {
                name: artifact(checkpoint / name) for name in CHECKPOINT_FILES
            },
        }
    payload = {
        "schema": RUNTIME_SCHEMA,
        "backend": "local",
        "status": "started",
        "started_utc": now,
        "hostname": hostname,
        "pid": os.getppid(),
        "gpu_count": 2,
        "gpu_indices": [0, 1],
        "gpu_model": GPU_MODEL,
        "gpus": gpus,
        "max_initial_gpu_memory_mib": max_initial_memory_mib,
        "scheduling": "four lanes sequential; each lane uses GPUs 0,1 with two shards",
        "runner": artifact(runner),
        "common_library": artifact(common_library),
        "completion_helper": artifact(completion_helper),
        "checkpoints": checkpoints,
    }
    atomic_json(output, payload)
    return payload


def require_artifact_matches(spec: Mapping[str, Any], path: Path, label: str) -> None:
    actual = artifact(path)
    for key in ("path", "size", "sha256"):
        if spec.get(key) != actual[key]:
            raise ValueError(
                f"{label} changed after runtime capture: {key} "
                f"captured={spec.get(key)!r} actual={actual[key]!r}"
            )


def validate_metrics(
    *, record_root: Path, eval_root: Path, step: int
) -> dict[str, dict[str, Any]]:
    paths = {
        "json": record_root / "metrics.json",
        "tsv": record_root / "metrics.tsv",
        "md": record_root / "metrics.md",
    }
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 4:
        raise ValueError("metrics.json must contain exactly four rows")
    with paths["tsv"].open(encoding="utf-8", newline="") as handle:
        tsv_rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(tsv_rows) != 4:
        raise ValueError("metrics.tsv must contain exactly four rows")
    expected = {(arm, mode) for arm in ("r3", "r5") for mode in ("no_text", "text")}
    seen: set[tuple[str, str]] = set()
    tsv_seen: set[tuple[str, str]] = set()
    json_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    tsv_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("metrics.json rows must be objects")
        key = (str(row.get("arm") or ""), str(row.get("mode") or ""))
        if key not in expected or key in seen:
            raise ValueError(f"metrics.json invalid/duplicate identity: {key}")
        arm, mode = key
        run_id = f"{LABELS[arm]}_step-{step}_{mode}_quick20_d2d3_seed1234"
        wanted = {
            "step": step,
            "train_job_id": TRAINING_JOBS[arm],
            "n": 20,
            "run_id": run_id,
            "output_dir": str((eval_root / run_id).resolve()),
        }
        bad = {
            name: {"expected": value, "actual": row.get(name)}
            for name, value in wanted.items()
            if row.get(name) != value
        }
        if bad:
            raise ValueError(f"metrics.json {key} provenance drift: {bad}")
        numeric = {}
        for field in (
            "fail", "cer", "sim_ref", "sim_src", "margin", "ref_bound",
            "ref_content_f1",
        ):
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"metrics.json {key}.{field} is not numeric") from exc
            if not math.isfinite(value):
                raise ValueError(f"metrics.json {key}.{field} is not finite")
            numeric[field] = value
        keep = row.get("keep")
        if isinstance(keep, bool) or not isinstance(keep, int) or not 0 <= keep <= 20:
            raise ValueError(f"metrics.json {key}.keep is invalid: {keep!r}")
        if not math.isclose(numeric["fail"], (20 - keep) / 20, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"metrics.json {key} fail/keep mismatch")
        if not math.isclose(
            numeric["margin"], numeric["sim_ref"] - numeric["sim_src"],
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError(f"metrics.json {key} margin mismatch")
        seen.add(key)
        json_by_key[key] = row
    for row in tsv_rows:
        key = (str(row.get("arm") or ""), str(row.get("mode") or ""))
        if key not in expected or key in tsv_seen:
            raise ValueError(f"metrics.tsv invalid/duplicate identity: {key}")
        tsv_seen.add(key)
        tsv_by_key[key] = row
    if seen != expected or tsv_seen != expected:
        raise ValueError("quick20 metric identity set is incomplete")

    identity_fields = (
        "step", "arm", "train_job_id", "mode", "n", "keep", "run_id",
        "output_dir", "text_en_src_scope",
    )
    numeric_fields = (
        "fail", "cer", "sim_ref", "sim_src", "margin", "ref_bound_count",
        "ref_bound", "ref_content_f1", "text_en_src_quick_n",
        "text_en_src_quick_fail",
    )
    for key in sorted(expected):
        json_row = json_by_key[key]
        tsv_row = tsv_by_key[key]
        for field in identity_fields:
            if str(json_row.get(field, "")) != str(tsv_row.get(field, "")):
                raise ValueError(f"metrics JSON/TSV disagree for {key}.{field}")
        for field in numeric_fields:
            json_value = json_row.get(field, "")
            tsv_value = tsv_row.get(field, "")
            if json_value in {"", None} or tsv_value in {"", None}:
                if json_value not in {"", None} or tsv_value not in {"", None}:
                    raise ValueError(f"metrics JSON/TSV disagree for {key}.{field}")
                continue
            try:
                left, right = float(json_value), float(tsv_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"metrics JSON/TSV non-numeric {key}.{field}") from exc
            if not math.isfinite(left) or not math.isfinite(right) or not math.isclose(
                left, right, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"metrics JSON/TSV disagree for {key}.{field}")
    return {name: artifact(path) for name, path in paths.items()}


def finalize_completion(
    *,
    record_root: Path,
    eval_root: Path,
    project_root: Path,
    code_root: Path,
    step: int,
    r3_checkpoint: Path,
    r5_checkpoint: Path,
    no_text20: Path,
    no_text20_sha256: str,
    text_source: Path,
    text_source_sha256: str,
    text20: Path,
    text20_sha256: str,
    runner: Path,
    common_library: Path,
    completion_helper: Path,
    runtime_manifest: Path,
) -> dict[str, Any]:
    record_root = record_root.resolve()
    eval_root = eval_root.resolve()
    project_root = project_root.resolve()
    code_root = code_root.resolve()
    completion_path = record_root / "COMPLETED.json"
    marker_path = record_root / "complete.marker"
    if os.path.lexists(record_root / "submitted_jobs.tsv"):
        raise ValueError("local completion forbids submitted_jobs.tsv")
    if completion_path.exists() or marker_path.exists():
        raise ValueError("completion evidence already exists; refusing overwrite")
    runtime = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    expected_runtime = {
        "schema": RUNTIME_SCHEMA,
        "backend": "local",
        "status": "started",
        "gpu_count": 2,
        "gpu_indices": [0, 1],
        "gpu_model": GPU_MODEL,
    }
    bad_runtime = {
        key: {"expected": wanted, "actual": runtime.get(key)}
        for key, wanted in expected_runtime.items()
        if runtime.get(key) != wanted
    }
    if bad_runtime:
        raise ValueError(f"local runtime manifest drift: {bad_runtime}")
    hostname = str(runtime.get("hostname") or "")
    if not hostname.startswith(HOST_PREFIX):
        raise ValueError(f"invalid local runtime hostname: {hostname!r}")
    gpus = runtime.get("gpus")
    if not isinstance(gpus, list) or len(gpus) != 2:
        raise ValueError("local runtime must bind two GPU objects")
    for index, gpu in enumerate(gpus):
        if not isinstance(gpu, dict):
            raise ValueError("local runtime GPU row must be an object")
        if gpu.get("index") != index or gpu.get("name") != GPU_MODEL:
            raise ValueError(f"local runtime GPU identity drift: {gpu}")
        if not str(gpu.get("uuid") or "").startswith("GPU-"):
            raise ValueError(f"local runtime GPU UUID drift: {gpu}")
        if int(gpu.get("memory_total_mib") or 0) < 48_000:
            raise ValueError(f"local runtime GPU memory drift: {gpu}")
    require_artifact_matches(runtime["runner"], runner, "runner")
    require_artifact_matches(runtime["common_library"], common_library, "common library")
    require_artifact_matches(
        runtime["completion_helper"], completion_helper, "completion helper"
    )

    fixed_inputs: dict[str, dict[str, Any]] = {}
    for name, path, wanted_sha in (
        ("no_text20", no_text20, no_text20_sha256),
        ("text_source", text_source, text_source_sha256),
        ("text20", text20, text20_sha256),
    ):
        got = sha256_file(path)
        if got != wanted_sha:
            raise ValueError(f"fixed input SHA drift: {name}={got}, expected {wanted_sha}")
        fixed_inputs[name] = artifact(path)

    checkpoints = runtime.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != {"r3", "r5"}:
        raise ValueError("local runtime checkpoint identity set is invalid")
    for arm, checkpoint in (("r3", r3_checkpoint), ("r5", r5_checkpoint)):
        checkpoint = checkpoint.resolve()
        if checkpoint.name != f"step-{step}" or not checkpoint.is_dir():
            raise ValueError(f"invalid {arm} checkpoint binding: {checkpoint}")
        captured = checkpoints[arm]
        if not isinstance(captured, dict):
            raise ValueError(f"local runtime {arm} checkpoint must be an object")
        expected_identity = {
            "path": str(checkpoint),
            "step": step,
            "training_job_id": TRAINING_JOBS[arm],
        }
        bad_identity = {
            key: {"expected": wanted, "actual": captured.get(key)}
            for key, wanted in expected_identity.items()
            if captured.get(key) != wanted
        }
        if bad_identity:
            raise ValueError(f"local runtime {arm} checkpoint drift: {bad_identity}")
        captured_files = captured.get("files")
        if not isinstance(captured_files, dict) or set(captured_files) != set(CHECKPOINT_FILES):
            raise ValueError(f"local runtime {arm} checkpoint artifact set is invalid")
        for name in CHECKPOINT_FILES:
            require_artifact_matches(
                captured_files[name], checkpoint / name, f"{arm} checkpoint {name}"
            )

    identity_root = (
        project_root
        / "trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713"
    )
    training_provenance = {
        "pair_ledger": artifact(identity_root / "submitted_pair.tsv"),
        "train_args": {
            arm: artifact(identity_root / arm / "train_args_dry_run_core.json")
            for arm in ("r3", "r5")
        },
    }
    metrics = validate_metrics(record_root=record_root, eval_root=eval_root, step=step)

    runs: list[dict[str, Any]] = []
    for arm in ("r3", "r5"):
        for mode in ("no_text", "text"):
            run_id = f"{LABELS[arm]}_step-{step}_{mode}_quick20_d2d3_seed1234"
            output_dir = (eval_root / run_id).resolve()
            output_artifacts = {
                "summary": artifact(output_dir / f"{run_id}.summary.json"),
                "asr": artifact(output_dir / f"{run_id}.asr_eval.jsonl"),
                "speaker": artifact(output_dir / f"{run_id}.speaker_sim.csv"),
                "ref_content": artifact(
                    output_dir / f"{run_id}.ref_content_similarity_summary.json"
                ),
            }
            runs.append(
                {
                    "arm": arm,
                    "mode": mode,
                    "run_id": run_id,
                    "training_job_id": TRAINING_JOBS[arm],
                    "checkpoint": str((r3_checkpoint if arm == "r3" else r5_checkpoint).resolve()),
                    "output_dir": str(output_dir),
                    "artifacts": output_artifacts,
                }
            )

    completed_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    runner_artifact = artifact(runner)
    runtime_artifact = artifact(runtime_manifest)
    payload = {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "backend": "local",
        "step": step,
        "completed_utc": completed_utc,
        "record_root": str(record_root),
        "eval_root": str(eval_root),
        "code_root": str(code_root),
        "training_jobs": TRAINING_JOBS,
        "training_provenance": training_provenance,
        "execution": {
            "hostname": hostname,
            "gpu_count": 2,
            "gpu_indices": [0, 1],
            "gpu_model": GPU_MODEL,
            "gpus": gpus,
            "scheduling": runtime.get("scheduling"),
            "runtime_manifest": runtime_artifact,
        },
        "runner": runner_artifact,
        "common_library": artifact(common_library),
        "completion_helper": artifact(completion_helper),
        "fixed_inputs": fixed_inputs,
        "checkpoints": checkpoints,
        "metrics": metrics,
        "runs": runs,
    }
    atomic_json(completion_path, payload)
    completion_sha = sha256_file(completion_path)
    marker = {
        "schema": MARKER_SCHEMA,
        "status": "complete",
        "backend": "local",
        "step": step,
        "completed_utc": completed_utc,
        "completed_json_sha256": completion_sha,
    }
    atomic_json(marker_path, marker)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture-runtime")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--runner", type=Path, required=True)
    capture.add_argument("--common-library", type=Path, required=True)
    capture.add_argument("--completion-helper", type=Path, required=True)
    capture.add_argument("--step", type=int, required=True)
    capture.add_argument("--r3-checkpoint", type=Path, required=True)
    capture.add_argument("--r5-checkpoint", type=Path, required=True)
    capture.add_argument("--max-initial-memory-mib", type=int, default=2048)
    capture.add_argument("--allow-any-host", action="store_true")

    finalize = subparsers.add_parser("finalize")
    for name in (
        "record-root",
        "eval-root",
        "project-root",
        "code-root",
        "r3-checkpoint",
        "r5-checkpoint",
        "no-text20",
        "text-source",
        "text20",
        "runner",
        "common-library",
        "completion-helper",
        "runtime-manifest",
    ):
        finalize.add_argument(f"--{name}", type=Path, required=True)
    finalize.add_argument("--step", type=int, required=True)
    finalize.add_argument("--no-text20-sha256", required=True)
    finalize.add_argument("--text-source-sha256", required=True)
    finalize.add_argument("--text20-sha256", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "capture-runtime":
        payload = capture_runtime(
            output=args.output,
            runner=args.runner,
            common_library=args.common_library,
            completion_helper=args.completion_helper,
            step=args.step,
            r3_checkpoint=args.r3_checkpoint,
            r5_checkpoint=args.r5_checkpoint,
            max_initial_memory_mib=args.max_initial_memory_mib,
            allow_any_host=args.allow_any_host,
        )
        print(
            "[batch44-local-runtime] PASS "
            f"host={payload['hostname']} gpu_count={payload['gpu_count']} "
            f"model={payload['gpu_model']}"
        )
        return 0
    payload = finalize_completion(
        record_root=args.record_root,
        eval_root=args.eval_root,
        project_root=args.project_root,
        code_root=args.code_root,
        step=args.step,
        r3_checkpoint=args.r3_checkpoint,
        r5_checkpoint=args.r5_checkpoint,
        no_text20=args.no_text20,
        no_text20_sha256=args.no_text20_sha256,
        text_source=args.text_source,
        text_source_sha256=args.text_source_sha256,
        text20=args.text20,
        text20_sha256=args.text20_sha256,
        runner=args.runner,
        common_library=args.common_library,
        completion_helper=args.completion_helper,
        runtime_manifest=args.runtime_manifest,
    )
    print(
        "[batch44-local-completion] PASS "
        f"step={payload['step']} backend={payload['backend']} runs={len(payload['runs'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

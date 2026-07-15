#!/usr/bin/env python3
"""Fail-closed provenance helpers for Batch-44 r3 warm-start quick20.

The continuation is a weights-only warm start from effective step 10,000.  Its
physical checkpoint names restart at ``step-2000`` while scientific reporting
continues at effective step 12,000.  This module keeps both identities in every
metric and completion artifact; it never talks to QZ or starts inference.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE_EFFECTIVE_STEP = 10_000
EFFECTIVE_STEPS = tuple(range(12_000, 30_001, 2_000))
COMPLETION_SCHEMA = "moss_codecvc.batch44_r3_warmstart_quick20_completion.v1"
MARKER_SCHEMA = "moss_codecvc.batch44_r3_warmstart_quick20_complete_marker.v1"
RUNTIME_SCHEMA = "moss_codecvc.batch44_r3_warmstart_quick20_local_runtime.v1"
WARM_START_SCHEMA = "batch44_r3_weights_only_warm_start_v1"
GPU_MODEL = "NVIDIA GeForce RTX 4090"
HOST_PREFIX = "xyzhang-dev--"
RUN_LABEL = "ver2_9_5_final_r3"
ALLOWED_COMPUTE_GROUP = "lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"
CHECKPOINT_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "README.md",
    "timbre_memory_adapter.pt",
    "timbre_memory_config.json",
)
JOB_RE = re.compile(
    r"^job-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
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


def checkpoint_manifest_sha256(files: Mapping[str, Mapping[str, Any]]) -> str:
    payload = {
        name: {
            "path": str(spec["path"]),
            "size": int(spec["size"]),
            "sha256": str(spec["sha256"]),
        }
        for name, spec in sorted(files.items())
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def validate_step_mapping(effective_step: int, continuation_local_step: int) -> None:
    if effective_step not in EFFECTIVE_STEPS:
        raise ValueError(
            f"effective step must be one of {EFFECTIVE_STEPS}, got {effective_step}"
        )
    if continuation_local_step <= 0 or continuation_local_step % 2_000:
        raise ValueError(
            "continuation local step must be a positive multiple of 2000; "
            f"got {continuation_local_step}"
        )
    if BASE_EFFECTIVE_STEP + continuation_local_step != effective_step:
        raise ValueError(
            "effective/local mapping drift: "
            f"{BASE_EFFECTIVE_STEP}+{continuation_local_step}!={effective_step}"
        )


def run_id(effective_step: int, mode: str) -> str:
    if mode not in {"no_text", "text"}:
        raise ValueError(f"unsupported quick20 mode: {mode}")
    return (
        f"{RUN_LABEL}_step-{effective_step}_{mode}_quick20_d2d3_seed1234"
    )


def require_artifact_matches(
    spec: Mapping[str, Any], path: Path, label: str
) -> Path:
    if not isinstance(spec, Mapping):
        raise ValueError(f"{label} artifact must be an object")
    actual = artifact(path)
    for key in ("path", "size", "sha256"):
        if spec.get(key) != actual[key]:
            raise ValueError(
                f"{label} artifact drift for {key}: "
                f"captured={spec.get(key)!r} actual={actual[key]!r}"
            )
    return path.resolve()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _checkpoint_files(checkpoint: Path, *, test_mode: bool) -> dict[str, dict[str, Any]]:
    minimum_large = 1 if test_mode else 1_000_000
    minimum = {
        "adapter_model.safetensors": minimum_large,
        "adapter_config.json": 1,
        "README.md": 1,
        "timbre_memory_adapter.pt": minimum_large,
        "timbre_memory_config.json": 1,
    }
    result: dict[str, dict[str, Any]] = {}
    for name in CHECKPOINT_FILES:
        path = checkpoint / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size < minimum[name]:
            raise ValueError(f"missing/small/symlink continuation checkpoint file: {path}")
        result[name] = artifact(path)
    _load_json(checkpoint / "adapter_config.json", "adapter config")
    config = _load_json(checkpoint / "timbre_memory_config.json", "timbre config")
    expected: dict[str, Any] = {
        "content_cross_attn_enabled": True,
        "content_cross_attn_layers": "all",
        "content_cross_attn_feature_dim": 768,
        "content_cross_attn_gate_init": -0.5,
        "content_cross_attn_output_scale": 0.3,
        "content_encoder_layers": 2,
        "guided_attn_loss_weight": 0.05,
        # The effective run is already beyond the original 1k loss warmup.
        "guided_attn_warmup_steps": 0,
        "phoneme_classifier_loss_weight": 0.02,
        "content_ctc_weight": 0.0,
        "progress_loss_weight": 0.1,
        "stop_loss_weight": 0.2,
        "target_front_ce_weight": 4.0,
        "target_front_ce_seconds": 0.75,
        "use_role_routing": True,
        "num_memory_tokens": 0,
        "timbre_side_only": False,
        "source_semantic_memory_enabled": False,
        "speaker_side_pathway_enabled": False,
        "speaker_cross_attn_enabled": False,
    }
    mismatches = {
        key: {"expected": wanted, "actual": config.get(key)}
        for key, wanted in expected.items()
        if config.get(key) != wanted
    }
    if mismatches:
        raise ValueError(f"continuation Path-X checkpoint config drift: {mismatches}")
    return result


def audit_binding(
    *,
    project_root: Path,
    effective_step: int,
    continuation_local_step: int,
    checkpoint: Path,
    train_job_id: str,
    warm_start_contract: Path,
    min_checkpoint_age_sec: int = 90,
    test_mode: bool = False,
) -> dict[str, Any]:
    validate_step_mapping(effective_step, continuation_local_step)
    project_root = project_root.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    warm_start_contract = warm_start_contract.expanduser().resolve()
    if not JOB_RE.fullmatch(train_job_id):
        raise ValueError(f"invalid continuation training job id: {train_job_id!r}")
    if warm_start_contract.is_symlink() or not warm_start_contract.is_file():
        raise ValueError(f"warm-start contract missing or symlink: {warm_start_contract}")
    contract = _load_json(warm_start_contract, "warm-start contract")
    expected_contract = {
        "schema": WARM_START_SCHEMA,
        "status": "submitted",
        "job_id": train_job_id,
        "source_effective_step": BASE_EFFECTIVE_STEP,
        "effective_step_offset": BASE_EFFECTIVE_STEP,
        "continuation_local_target_step": 20_000,
        "effective_target_step": 30_000,
        "resume_semantics": "weights_only_warm_start_not_exact_resume",
    }
    contract_drift = {
        key: {"expected": wanted, "actual": contract.get(key)}
        for key, wanted in expected_contract.items()
        if contract.get(key) != wanted
    }
    if contract_drift:
        raise ValueError(f"warm-start contract identity drift: {contract_drift}")
    output_dir = Path(str(contract.get("output_dir") or "")).expanduser().resolve()
    expected_checkpoint = output_dir / f"step-{continuation_local_step}"
    if checkpoint != expected_checkpoint:
        raise ValueError(
            f"continuation checkpoint={checkpoint}, expected {expected_checkpoint}"
        )
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise ValueError(f"invalid continuation checkpoint directory: {checkpoint}")
    files = _checkpoint_files(checkpoint, test_mode=test_mode)
    newest = max(Path(spec["path"]).stat().st_mtime for spec in files.values())
    age = time.time() - newest
    if age < min_checkpoint_age_sec:
        raise ValueError(
            f"continuation checkpoint is still settling: age={age:.1f}s "
            f"< {min_checkpoint_age_sec}s"
        )

    # The base checkpoint hashes are part of the immutable warm-start contract.
    source_files = contract.get("source_checkpoint_files")
    if not isinstance(source_files, dict) or set(source_files) != set(CHECKPOINT_FILES):
        raise ValueError("warm-start contract base checkpoint artifact set drift")
    for name in CHECKPOINT_FILES:
        spec = source_files[name]
        if not isinstance(spec, dict):
            raise ValueError(f"invalid base checkpoint artifact spec: {name}")
        source_path = Path(str(spec.get("path") or "")).expanduser().resolve()
        wanted_size = spec.get("bytes")
        wanted_sha = spec.get("sha256")
        if (
            not source_path.is_file()
            or source_path.stat().st_size != wanted_size
            or sha256_file(source_path) != wanted_sha
        ):
            raise ValueError(f"base checkpoint artifact drift: {source_path}")

    record_root = warm_start_contract.parent
    ledger = record_root / "submitted_jobs.tsv"
    with ledger.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 1:
        raise ValueError(f"continuation submitted_jobs.tsv must contain one row: {ledger}")
    row = rows[0]
    expected_ledger = {
        "job_name": str(contract.get("job_name") or ""),
        "job_id": train_job_id,
        "compute_group": ALLOWED_COMPUTE_GROUP,
        "out_dir": str(output_dir),
    }
    ledger_drift = {
        key: {"expected": wanted, "actual": row.get(key)}
        for key, wanted in expected_ledger.items()
        if row.get(key) != wanted
    }
    if ledger_drift:
        raise ValueError(f"continuation submission ledger drift: {ledger_drift}")

    core = record_root / "train_args_dry_run_core.json"
    core_payload = _load_json(core, "continuation train args")
    expected_core = {
        "OUT_DIR": str(output_dir),
        "TEXT_REPEAT": "3",
        "MAX_TRAIN_STEPS": "20000",
        "SAVE_STEPS": "2000",
        "EVAL_STEPS": "2000",
        "LEARNING_RATE": "1e-5",
        "LR_SCHEDULER_TYPE": "constant_with_warmup",
        "WARMUP_RATIO": "0.0",
        "GUIDED_ATTN_WARMUP_STEPS": "0",
        "GUIDED_ATTN_LOSS_WEIGHT": "0.05",
        "PHONEME_CLASSIFIER_LOSS_WEIGHT": "0.02",
        "CONTENT_CTC_WEIGHT": "0.0",
        "ENABLE_CONTENT_CROSS_ATTN": "1",
        "CONTENT_CROSS_ATTN_LAYERS": "all",
    }
    core_drift = {
        key: {"expected": wanted, "actual": core_payload.get(key)}
        for key, wanted in expected_core.items()
        if core_payload.get(key) != wanted
    }
    if core_drift:
        raise ValueError(f"continuation train args drift: {core_drift}")
    training_runner = record_root / "run_train_entrypoint.sh"
    runner_text = training_runner.read_text(encoding="utf-8")
    for needle in (
        str(contract.get("source_checkpoint") or ""),
        str(output_dir),
        "--resume-adapter-path $RESUME_ADAPTER_PATH",
    ):
        if needle not in runner_text:
            raise ValueError(f"continuation training runner missing binding: {needle!r}")

    training_artifacts = {
        "submitted_jobs": artifact(ledger),
        "train_args": artifact(core),
        "train_runner": artifact(training_runner),
    }
    for key, name in (
        ("generated_config_audit", "generated_config_audit.json"),
        ("qz_payload", "qz_payload.json"),
    ):
        path = record_root / name
        if path.is_file() and path.stat().st_size > 0:
            training_artifacts[key] = artifact(path)

    return {
        "base_effective_step": BASE_EFFECTIVE_STEP,
        "effective_step": effective_step,
        "continuation_local_step": continuation_local_step,
        "checkpoint": str(checkpoint),
        "checkpoint_files": files,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256(files),
        "checkpoint_age_seconds": age,
        "train_job_id": train_job_id,
        "warm_start_contract": artifact(warm_start_contract),
        "warm_start_contract_payload": contract,
        "training_provenance": training_artifacts,
    }


def _parse_gpu_row(line: str) -> list[str]:
    row = [piece.strip() for piece in next(csv.reader([line]))]
    if len(row) != 6:
        raise ValueError(f"unexpected nvidia-smi row: {line!r}")
    return row


def capture_runtime(
    *,
    output: Path,
    runner: Path,
    common_library: Path,
    completion_helper: Path,
    validator: Path,
    effective_step: int,
    continuation_local_step: int,
    checkpoint: Path,
    train_job_id: str,
    warm_start_contract: Path,
    max_initial_memory_mib: int,
    allow_any_host: bool = False,
) -> dict[str, Any]:
    validate_step_mapping(effective_step, continuation_local_step)
    hostname = socket.gethostname()
    if not allow_any_host and not hostname.startswith(HOST_PREFIX):
        raise ValueError(
            f"local quick20 is restricted to {HOST_PREFIX}*; got {hostname!r}"
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
    rows = [_parse_gpu_row(line) for line in query.stdout.splitlines() if line.strip()]
    if len(rows) != 2:
        raise ValueError(f"local quick20 requires exactly two GPUs, got {len(rows)}")
    gpus = [
        {
            "index": int(index),
            "uuid": uuid,
            "name": name,
            "memory_total_mib": int(total),
            "memory_used_mib_at_start": int(used),
            "driver_version": driver,
        }
        for index, uuid, name, total, used, driver in rows
    ]
    gpus.sort(key=lambda item: item["index"])
    if [item["index"] for item in gpus] != [0, 1]:
        raise ValueError(f"local quick20 requires GPU indices [0,1], got {gpus}")
    if any(item["name"] != GPU_MODEL for item in gpus):
        raise ValueError(f"local quick20 requires two {GPU_MODEL} GPUs, got {gpus}")
    if any(item["memory_total_mib"] < 48_000 for item in gpus):
        raise ValueError(f"local quick20 GPU memory contract failed: {gpus}")
    busy = [
        item for item in gpus
        if item["memory_used_mib_at_start"] > max_initial_memory_mib
    ]
    if busy:
        raise ValueError(
            f"local quick20 refuses busy GPUs above {max_initial_memory_mib} MiB: {busy}"
        )
    files = _checkpoint_files(checkpoint.resolve(), test_mode=False)
    payload = {
        "schema": RUNTIME_SCHEMA,
        "status": "started",
        "backend": "local",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": hostname,
        "pid": os.getppid(),
        "base_effective_step": BASE_EFFECTIVE_STEP,
        "effective_step": effective_step,
        "continuation_local_step": continuation_local_step,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_files": files,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256(files),
        "train_job_id": train_job_id,
        "warm_start_contract": artifact(warm_start_contract),
        "gpu_count": 2,
        "gpu_indices": [0, 1],
        "gpu_model": GPU_MODEL,
        "gpus": gpus,
        "max_initial_gpu_memory_mib": max_initial_memory_mib,
        "scheduling": "two lanes sequential; each lane uses GPUs 0,1 with two shards",
        "runner": artifact(runner),
        "common_library": artifact(common_library),
        "completion_helper": artifact(completion_helper),
        "validator": artifact(validator),
    }
    atomic_json(output, payload)
    return payload


def collect_metrics(
    *,
    record_root: Path,
    output_root: Path,
    effective_step: int,
    continuation_local_step: int,
    checkpoint: Path,
    train_job_id: str,
    warm_start_contract: Path,
) -> list[dict[str, Any]]:
    validate_step_mapping(effective_step, continuation_local_step)
    record_root = record_root.resolve()
    output_root = output_root.resolve()
    checkpoint = checkpoint.resolve()
    contract_sha = sha256_file(warm_start_contract)
    checkpoint_files = _checkpoint_files(checkpoint, test_mode=False)
    manifest_sha = checkpoint_manifest_sha256(checkpoint_files)
    rows_out: list[dict[str, Any]] = []
    for mode in ("no_text", "text"):
        identity = run_id(effective_step, mode)
        output_dir = output_root / identity
        summary_path = output_dir / f"{identity}.summary.json"
        speaker_path = output_dir / f"{identity}.speaker_sim.csv"
        ref_content_path = output_dir / f"{identity}.ref_content_similarity_summary.json"
        asr_path = output_dir / f"{identity}.asr_eval.jsonl"
        summary = _load_json(summary_path, f"{mode} summary").get("overall")
        if not isinstance(summary, dict):
            raise ValueError(f"missing overall summary: {summary_path}")
        with speaker_path.open(encoding="utf-8", newline="") as handle:
            speaker_rows = list(csv.DictReader(handle))
        valid_speaker = [
            row for row in speaker_rows
            if row.get("status") in {"ok", "ok_after_rerun", "skipped_exists"}
        ]
        asr_rows = [
            json.loads(line)
            for line in asr_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        n, keep = int(summary["n"]), int(summary["keep"])
        if n != 20 or len(valid_speaker) != 20 or len(asr_rows) != 20:
            raise ValueError(
                f"incomplete {mode} quick20: summary={n} "
                f"speaker={len(valid_speaker)} asr={len(asr_rows)}"
            )
        asr_ids = [str(row.get("case_id") or row.get("sample_id") or "") for row in asr_rows]
        speaker_ids = [str(row.get("case_id") or "") for row in valid_speaker]
        if (
            any(not item for item in asr_ids + speaker_ids)
            or len(set(asr_ids)) != 20
            or len(set(speaker_ids)) != 20
            or set(asr_ids) != set(speaker_ids)
        ):
            raise ValueError(f"ASR/speaker case identity mismatch: {identity}")
        if any(row.get("mode") != mode for row in asr_rows):
            raise ValueError(f"wrong ASR mode in {identity}")
        sim_ref = sum(float(row["sim_gen_ref"]) for row in valid_speaker) / 20
        sim_src = sum(float(row["sim_gen_source"]) for row in valid_speaker) / 20
        ref_bound_count = sum(
            float(row["sim_gen_ref"]) - float(row["sim_gen_source"]) > 0.05
            for row in valid_speaker
        )
        ref_content = _load_json(ref_content_path, f"{mode} ref-content").get("overall")
        if not isinstance(ref_content, dict):
            raise ValueError(f"missing ref-content overall: {ref_content_path}")
        en_src_rows = (
            [row for row in asr_rows if str(row.get("cell") or "").startswith("en_src_")]
            if mode == "text" else []
        )
        if mode == "text" and len(en_src_rows) != 12:
            raise ValueError(f"text en_src quick proxy must contain 12 rows, got {len(en_src_rows)}")
        en_src_fail = (
            sum(row.get("content_keep") is not True for row in en_src_rows) / len(en_src_rows)
            if en_src_rows else None
        )
        rows_out.append(
            {
                "step": effective_step,
                "effective_step": effective_step,
                "base_effective_step": BASE_EFFECTIVE_STEP,
                "continuation_local_step": continuation_local_step,
                "arm": "r3",
                "train_job_id": train_job_id,
                "mode": mode,
                "n": n,
                "keep": keep,
                "fail": (n - keep) / n,
                "cer": float(summary["cer"]),
                "sim_ref": sim_ref,
                "sim_src": sim_src,
                "margin": sim_ref - sim_src,
                "ref_bound_count": ref_bound_count,
                "ref_bound": ref_bound_count / 20,
                "ref_content_f1": float(ref_content["ref_content_lcs_f1_mean"]),
                "text_en_src_quick_n": len(en_src_rows) if en_src_rows else "",
                "text_en_src_quick_fail": en_src_fail if en_src_fail is not None else "",
                "text_en_src_scope": (
                    "quick20 proxy n=12; not the full text en_src n=80 gate"
                    if en_src_rows else ""
                ),
                "checkpoint": str(checkpoint),
                "checkpoint_manifest_sha256": manifest_sha,
                "warm_start_contract": str(warm_start_contract.resolve()),
                "warm_start_contract_sha256": contract_sha,
                "run_id": identity,
                "output_dir": str(output_dir),
            }
        )

    atomic_json(record_root / "metrics.json", rows_out)
    fields = list(rows_out[0])
    tsv_tmp = record_root / f".metrics.tsv.tmp-{os.getpid()}"
    with tsv_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows_out)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tsv_tmp, record_root / "metrics.tsv")
    lines = [
        f"# Batch-44 r3 warm-start quick20 effective step-{effective_step}",
        "",
        f"Continuation local step: `{continuation_local_step}`; base effective step: `{BASE_EFFECTIVE_STEP}`.",
        "",
        "| Mode | fail | CER | sim(ref) | sim(src) | margin | ref-bound | F1(ref-content) | text en_src quick fail |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows_out:
        en_src = (
            "—" if row["text_en_src_quick_fail"] == ""
            else f"{row['text_en_src_quick_fail']:.1%} (n=12)"
        )
        lines.append(
            f"| {row['mode']} | {row['fail']:.1%} | {row['cer']:.4f} | "
            f"{row['sim_ref']:.4f} | {row['sim_src']:.4f} | {row['margin']:.4f} | "
            f"{row['ref_bound']:.1%} | {row['ref_content_f1']:.4f} | {en_src} |"
        )
    atomic_text(record_root / "metrics.md", "\n".join(lines) + "\n")
    return rows_out


def _validate_metrics(
    *,
    record_root: Path,
    output_root: Path,
    effective_step: int,
    continuation_local_step: int,
    checkpoint: Path,
    train_job_id: str,
    warm_start_contract: Path,
) -> dict[str, dict[str, Any]]:
    paths = {
        "json": record_root / "metrics.json",
        "tsv": record_root / "metrics.tsv",
        "md": record_root / "metrics.md",
    }
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 2:
        raise ValueError("r3 warm-start metrics.json must contain exactly two rows")
    with paths["tsv"].open(encoding="utf-8", newline="") as handle:
        tsv_rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(tsv_rows) != 2:
        raise ValueError("r3 warm-start metrics.tsv must contain exactly two rows")
    expected_modes = {"no_text", "text"}
    seen: set[str] = set()
    contract_sha = sha256_file(warm_start_contract)
    checkpoint_files = _checkpoint_files(checkpoint, test_mode=False)
    manifest_sha = checkpoint_manifest_sha256(checkpoint_files)
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("metrics rows must be JSON objects")
        mode = str(row.get("mode") or "")
        if mode not in expected_modes or mode in seen:
            raise ValueError(f"invalid/duplicate r3 warm-start metric mode: {mode!r}")
        identity = run_id(effective_step, mode)
        wanted = {
            "step": effective_step,
            "effective_step": effective_step,
            "base_effective_step": BASE_EFFECTIVE_STEP,
            "continuation_local_step": continuation_local_step,
            "arm": "r3",
            "train_job_id": train_job_id,
            "n": 20,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_manifest_sha256": manifest_sha,
            "warm_start_contract": str(warm_start_contract.resolve()),
            "warm_start_contract_sha256": contract_sha,
            "run_id": identity,
            "output_dir": str((output_root / identity).resolve()),
        }
        drift = {
            key: {"expected": value, "actual": row.get(key)}
            for key, value in wanted.items()
            if row.get(key) != value
        }
        if drift:
            raise ValueError(f"metrics {mode} provenance drift: {drift}")
        numeric: dict[str, float] = {}
        for field in ("fail", "cer", "sim_ref", "sim_src", "margin", "ref_bound", "ref_content_f1"):
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"metrics {mode}.{field} is not numeric") from exc
            if not math.isfinite(value):
                raise ValueError(f"metrics {mode}.{field} is not finite")
            numeric[field] = value
        keep = row.get("keep")
        if isinstance(keep, bool) or not isinstance(keep, int) or not 0 <= keep <= 20:
            raise ValueError(f"metrics {mode}.keep is invalid: {keep!r}")
        if not math.isclose(numeric["fail"], (20 - keep) / 20, abs_tol=1e-12):
            raise ValueError(f"metrics {mode} fail/keep mismatch")
        if not math.isclose(
            numeric["margin"], numeric["sim_ref"] - numeric["sim_src"], abs_tol=1e-12
        ):
            raise ValueError(f"metrics {mode} margin mismatch")
        seen.add(mode)
    if seen != expected_modes:
        raise ValueError(f"metrics mode set drift: {seen}")
    json_modes = {str(row["mode"]): row for row in payload}
    for row in tsv_rows:
        mode = str(row.get("mode") or "")
        if mode not in json_modes:
            raise ValueError(f"unexpected metrics.tsv mode: {mode}")
        for field in (
            "step", "effective_step", "base_effective_step", "continuation_local_step",
            "arm", "train_job_id", "mode", "n", "keep", "run_id", "output_dir",
            "checkpoint", "checkpoint_manifest_sha256", "warm_start_contract",
            "warm_start_contract_sha256",
        ):
            if str(row.get(field, "")) != str(json_modes[mode].get(field, "")):
                raise ValueError(f"metrics JSON/TSV disagree for {mode}.{field}")
    return {name: artifact(path) for name, path in paths.items()}


def finalize_completion(
    *,
    record_root: Path,
    output_root: Path,
    project_root: Path,
    code_root: Path,
    effective_step: int,
    continuation_local_step: int,
    checkpoint: Path,
    train_job_id: str,
    warm_start_contract: Path,
    no_text20: Path,
    no_text20_sha256: str,
    text_source: Path,
    text_source_sha256: str,
    text20: Path,
    text20_sha256: str,
    runner: Path,
    common_library: Path,
    completion_helper: Path,
    validator: Path,
    runtime_manifest: Path,
) -> dict[str, Any]:
    validate_step_mapping(effective_step, continuation_local_step)
    record_root = record_root.resolve()
    output_root = output_root.resolve()
    project_root = project_root.resolve()
    code_root = code_root.resolve()
    checkpoint = checkpoint.resolve()
    warm_start_contract = warm_start_contract.resolve()
    completion_path = record_root / "COMPLETED.json"
    marker_path = record_root / "complete.marker"
    if completion_path.exists() or marker_path.exists():
        raise ValueError("completion evidence already exists; refusing overwrite")
    if os.path.lexists(record_root / "submitted_jobs.tsv"):
        raise ValueError("local evaluation record may not contain submitted_jobs.tsv")

    binding = audit_binding(
        project_root=project_root,
        effective_step=effective_step,
        continuation_local_step=continuation_local_step,
        checkpoint=checkpoint,
        train_job_id=train_job_id,
        warm_start_contract=warm_start_contract,
        min_checkpoint_age_sec=0,
    )
    runtime = _load_json(runtime_manifest, "local runtime")
    expected_runtime = {
        "schema": RUNTIME_SCHEMA,
        "status": "started",
        "backend": "local",
        "base_effective_step": BASE_EFFECTIVE_STEP,
        "effective_step": effective_step,
        "continuation_local_step": continuation_local_step,
        "checkpoint": str(checkpoint),
        "train_job_id": train_job_id,
        "gpu_count": 2,
        "gpu_indices": [0, 1],
        "gpu_model": GPU_MODEL,
    }
    runtime_drift = {
        key: {"expected": wanted, "actual": runtime.get(key)}
        for key, wanted in expected_runtime.items()
        if runtime.get(key) != wanted
    }
    if runtime_drift:
        raise ValueError(f"local runtime manifest drift: {runtime_drift}")
    hostname = str(runtime.get("hostname") or "")
    if not hostname.startswith(HOST_PREFIX):
        raise ValueError(f"invalid local runtime hostname: {hostname!r}")
    for key, path, label in (
        ("runner", runner, "runner"),
        ("common_library", common_library, "common library"),
        ("completion_helper", completion_helper, "completion helper"),
        ("validator", validator, "validator"),
        ("warm_start_contract", warm_start_contract, "warm-start contract"),
    ):
        require_artifact_matches(runtime[key], path, label)
    runtime_files = runtime.get("checkpoint_files")
    if not isinstance(runtime_files, dict) or set(runtime_files) != set(CHECKPOINT_FILES):
        raise ValueError("runtime checkpoint artifact set drift")
    for name in CHECKPOINT_FILES:
        require_artifact_matches(runtime_files[name], checkpoint / name, f"checkpoint {name}")
    if runtime.get("checkpoint_manifest_sha256") != binding["checkpoint_manifest_sha256"]:
        raise ValueError("runtime checkpoint manifest SHA drift")

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

    metrics = _validate_metrics(
        record_root=record_root,
        output_root=output_root,
        effective_step=effective_step,
        continuation_local_step=continuation_local_step,
        checkpoint=checkpoint,
        train_job_id=train_job_id,
        warm_start_contract=warm_start_contract,
    )
    runs: list[dict[str, Any]] = []
    for mode in ("no_text", "text"):
        identity = run_id(effective_step, mode)
        output_dir = (output_root / identity).resolve()
        run_artifacts = {
            "summary": artifact(output_dir / f"{identity}.summary.json"),
            "asr": artifact(output_dir / f"{identity}.asr_eval.jsonl"),
            "speaker": artifact(output_dir / f"{identity}.speaker_sim.csv"),
            "ref_content": artifact(
                output_dir / f"{identity}.ref_content_similarity_summary.json"
            ),
        }
        runs.append(
            {
                "arm": "r3",
                "mode": mode,
                "run_id": identity,
                "effective_step": effective_step,
                "continuation_local_step": continuation_local_step,
                "checkpoint": str(checkpoint),
                "train_job_id": train_job_id,
                "output_dir": str(output_dir),
                "artifacts": run_artifacts,
            }
        )

    completed_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    contract_sha = sha256_file(warm_start_contract)
    payload = {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "backend": "local",
        "step": effective_step,
        "effective_step": effective_step,
        "base_effective_step": BASE_EFFECTIVE_STEP,
        "continuation_local_step": continuation_local_step,
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_sha256": binding["checkpoint_manifest_sha256"],
        "train_job_id": train_job_id,
        "warm_start_contract": str(warm_start_contract),
        "warm_start_contract_sha256": contract_sha,
        "completed_utc": completed_utc,
        "record_root": str(record_root),
        "output_root": str(output_root),
        "project_root": str(project_root),
        "code_root": str(code_root),
        "warm_start": {
            "contract": binding["warm_start_contract"],
            "resume_semantics": "weights_only_warm_start_not_exact_resume",
            "source_effective_step": BASE_EFFECTIVE_STEP,
            "effective_step_offset": BASE_EFFECTIVE_STEP,
        },
        "training_provenance": binding["training_provenance"],
        "checkpoint_files": binding["checkpoint_files"],
        "execution": {
            "hostname": hostname,
            "gpu_count": 2,
            "gpu_indices": [0, 1],
            "gpu_model": GPU_MODEL,
            "gpus": runtime.get("gpus"),
            "scheduling": runtime.get("scheduling"),
            "runtime_manifest": artifact(runtime_manifest),
        },
        "runner": artifact(runner),
        "common_library": artifact(common_library),
        "completion_helper": artifact(completion_helper),
        "validator": artifact(validator),
        "fixed_inputs": fixed_inputs,
        "metrics": metrics,
        "runs": runs,
    }
    atomic_json(completion_path, payload)
    completion_sha = sha256_file(completion_path)
    marker = {
        "schema": MARKER_SCHEMA,
        "status": "complete",
        "backend": "local",
        "step": effective_step,
        "effective_step": effective_step,
        "base_effective_step": BASE_EFFECTIVE_STEP,
        "continuation_local_step": continuation_local_step,
        "completed_utc": completed_utc,
        "completed_json_sha256": completion_sha,
    }
    # Marker is deliberately the last producer artifact.
    atomic_json(marker_path, marker)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit-binding")
    audit.add_argument("--project-root", type=Path, required=True)
    audit.add_argument("--effective-step", type=int, required=True)
    audit.add_argument("--continuation-local-step", type=int, required=True)
    audit.add_argument("--checkpoint", type=Path, required=True)
    audit.add_argument("--train-job-id", required=True)
    audit.add_argument("--warm-start-contract", type=Path, required=True)
    audit.add_argument("--min-checkpoint-age-sec", type=int, default=90)
    audit.add_argument("--test-mode", action="store_true")

    capture = commands.add_parser("capture-runtime")
    for name in ("output", "runner", "common-library", "completion-helper", "validator", "checkpoint", "warm-start-contract"):
        capture.add_argument(f"--{name}", type=Path, required=True)
    capture.add_argument("--effective-step", type=int, required=True)
    capture.add_argument("--continuation-local-step", type=int, required=True)
    capture.add_argument("--train-job-id", required=True)
    capture.add_argument("--max-initial-memory-mib", type=int, default=2048)
    capture.add_argument("--allow-any-host", action="store_true")

    collect = commands.add_parser("collect-metrics")
    for name in ("record-root", "output-root", "checkpoint", "warm-start-contract"):
        collect.add_argument(f"--{name}", type=Path, required=True)
    collect.add_argument("--effective-step", type=int, required=True)
    collect.add_argument("--continuation-local-step", type=int, required=True)
    collect.add_argument("--train-job-id", required=True)

    finalize = commands.add_parser("finalize")
    for name in (
        "record-root", "output-root", "project-root", "code-root", "checkpoint",
        "warm-start-contract", "no-text20", "text-source", "text20", "runner",
        "common-library", "completion-helper", "validator", "runtime-manifest",
    ):
        finalize.add_argument(f"--{name}", type=Path, required=True)
    finalize.add_argument("--effective-step", type=int, required=True)
    finalize.add_argument("--continuation-local-step", type=int, required=True)
    finalize.add_argument("--train-job-id", required=True)
    finalize.add_argument("--no-text20-sha256", required=True)
    finalize.add_argument("--text-source-sha256", required=True)
    finalize.add_argument("--text20-sha256", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "audit-binding":
        payload = audit_binding(
            project_root=args.project_root,
            effective_step=args.effective_step,
            continuation_local_step=args.continuation_local_step,
            checkpoint=args.checkpoint,
            train_job_id=args.train_job_id,
            warm_start_contract=args.warm_start_contract,
            min_checkpoint_age_sec=args.min_checkpoint_age_sec,
            test_mode=args.test_mode,
        )
        print(
            "[batch44-r3-warmstart-binding] PASS "
            f"effective={payload['effective_step']} local={payload['continuation_local_step']} "
            f"checkpoint_sha={payload['checkpoint_manifest_sha256']}"
        )
        return 0
    if args.command == "capture-runtime":
        payload = capture_runtime(
            output=args.output,
            runner=args.runner,
            common_library=args.common_library,
            completion_helper=args.completion_helper,
            validator=args.validator,
            effective_step=args.effective_step,
            continuation_local_step=args.continuation_local_step,
            checkpoint=args.checkpoint,
            train_job_id=args.train_job_id,
            warm_start_contract=args.warm_start_contract,
            max_initial_memory_mib=args.max_initial_memory_mib,
            allow_any_host=args.allow_any_host,
        )
        print(
            "[batch44-r3-warmstart-runtime] PASS "
            f"host={payload['hostname']} effective={payload['effective_step']}"
        )
        return 0
    if args.command == "collect-metrics":
        rows = collect_metrics(
            record_root=args.record_root,
            output_root=args.output_root,
            effective_step=args.effective_step,
            continuation_local_step=args.continuation_local_step,
            checkpoint=args.checkpoint,
            train_job_id=args.train_job_id,
            warm_start_contract=args.warm_start_contract,
        )
        print(
            "[batch44-r3-warmstart-metrics] PASS "
            f"effective={args.effective_step} rows={len(rows)}"
        )
        return 0
    payload = finalize_completion(
        record_root=args.record_root,
        output_root=args.output_root,
        project_root=args.project_root,
        code_root=args.code_root,
        effective_step=args.effective_step,
        continuation_local_step=args.continuation_local_step,
        checkpoint=args.checkpoint,
        train_job_id=args.train_job_id,
        warm_start_contract=args.warm_start_contract,
        no_text20=args.no_text20,
        no_text20_sha256=args.no_text20_sha256,
        text_source=args.text_source,
        text_source_sha256=args.text_source_sha256,
        text20=args.text20,
        text20_sha256=args.text20_sha256,
        runner=args.runner,
        common_library=args.common_library,
        completion_helper=args.completion_helper,
        validator=args.validator,
        runtime_manifest=args.runtime_manifest,
    )
    print(
        "[batch44-r3-warmstart-completion] PASS "
        f"effective={payload['effective_step']} local={payload['continuation_local_step']} "
        f"runs={len(payload['runs'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

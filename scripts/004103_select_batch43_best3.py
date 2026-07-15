#!/usr/bin/env python3
"""Select the Batch-44 v1 Best3 candidates from registered quick20 evidence.

The six candidates are the Cartesian product of arms ``r3``/``r5`` and
checkpoints 26k/28k/30k.  Selection happens *before* the expensive full-320
pass and uses the pooled no_text20 + text20 WavLM SIM(ref) mean.  Since both
modes contain exactly 20 cases, this is an unweighted mean of the two mode
means.  Ties are broken by no_text SIM(ref), pooled SIM(ref)-SIM(src), lower
pooled CER, then the stable candidate id.

The output is a provenance-bearing plan, not a final model declaration.  A
candidate can become ``path_x_final`` only after full-320 evidence and a
completed blind20 review are recorded by the later finalization scripts.

Default inputs are the per-step ``metrics.json`` files written by 004110.
When the six registered artifacts are not yet ready, the script writes a
``pending`` plan (unless ``--no-pending-output`` is used) and exits with code
3.  It never submits a QZ task.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "moss_codecvc.batch44_v1_best3_selection.v1"
EXPERIMENT_ID = "batch44_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTERED_STEPS = (26000, 28000, 30000)
REGISTERED_ARMS = ("r3", "r5")
EXPECTED_N_PER_MODE = 20
EXPECTED_TRAIN_JOBS = {
    "r3": "job-2b91d332-d500-4279-84f9-0a6a81a376aa",
    "r5": "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c",
}
RUN_DIRS = {
    "r3": "ver2_9_5_final_r3_v1_30k",
    "r5": "ver2_9_5_final_r5_v1_30k",
}
REJECTED_BATCH43_V2_TRAIN_JOBS = {
    "job-a34d84d4-59cc-4824-b197-0829bfe79004",
    "job-aef79753-7fcd-444e-b94d-3e21eedb2394",
}
TEXT_REPEATS = {"r3": 3, "r5": 5}
ALLOWED_COMPUTE_GROUP = "lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"
ALLOWED_SPEC = "67b10bc6-78b0-41a3-aaf4-358eeeb99009"
EXPECTED_EVAL_CODE_ROOT = (
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
    "MOSS-CodecVC_snapshots/ver23_batch37_eval_20260711_1092820"
)
QZ_QUICK20_COMPLETION_SCHEMA = "moss_codecvc.batch44_v1_quick20_completion.v1"
QZ_QUICK20_MARKER_SCHEMA = "moss_codecvc.batch44_v1_quick20_complete_marker.v1"
LOCAL_QUICK20_COMPLETION_SCHEMA = "moss_codecvc.batch44_v1_quick20_completion.v2"
LOCAL_QUICK20_MARKER_SCHEMA = "moss_codecvc.batch44_v1_quick20_complete_marker.v2"
LOCAL_GPU_MODEL = "NVIDIA GeForce RTX 4090"
LOCAL_HOST_RE = re.compile(r"^xyzhang-dev--[A-Za-z0-9-]+$")
GPU_UUID_RE = re.compile(r"^GPU-[0-9a-fA-F-]{36}$")
JOB_ID_RE = re.compile(
    r"^job-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
REQUIRED_CHECKPOINT_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "README.md",
    "timbre_memory_adapter.pt",
    "timbre_memory_config.json",
)
LOCAL_FIXED_INPUTS = {
    "no_text20": (
        "testset/validation/ver2_8_timbre_quick20_seedtts_no_text_20260704.jsonl",
        "f28de52e87b8c422380fe22052039ce48d59a1662a8bf7b137ce67405c35fba0",
    ),
    "text_source": (
        "testset/validation/seedtts_vc_ver2_3_validation.jsonl",
        "725ee9d58a7e6066d2a7b79c858cb6ff4dd7292cc167c45dc6b6ebbeaff2fe14",
    ),
}
LOCAL_TEXT20_SHA256 = "0952c4162e7ff7a9c2850f1f76f572f2f710e205b222c874016b05564f21bea8"
TRAINING_PROVENANCE_SHA256 = {
    "pair_ledger": "4fc395492617147d24459e02997ff30afec4c119e100d57accef42f34646cb7c",
    "r3_args": "3637b90fe14bbda7c4915362d5bd726b072833f07853df4fc555fde0d2c32d7b",
    "r5_args": "161f05b766454ce7d3e38af1772a1126ec612eaeb0682678bd6a46ca1d62daff",
}


class PendingEvidence(RuntimeError):
    """Raised when an expected future artifact is not complete yet."""


def finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_artifact(
    value: Any,
    *,
    label: str,
    expected_path: Path | None = None,
    required_parent: Path | None = None,
) -> Path:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an artifact object")
    path = Path(str(value.get("path") or "")).expanduser().resolve()
    if expected_path is not None and path != expected_path.expanduser().resolve():
        raise ValueError(f"{label} path={path}, expected {expected_path.resolve()}")
    if required_parent is not None and path.parent != required_parent.expanduser().resolve():
        raise ValueError(f"{label} must be stored directly under {required_parent.resolve()}")
    if not path.is_file() or path.stat().st_size <= 0:
        raise PendingEvidence(f"missing/empty {label}: {path}")
    if value.get("size") != path.stat().st_size:
        raise ValueError(f"{label} size drift")
    expected_sha = value.get("sha256")
    if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ValueError(f"{label} has invalid SHA256")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(f"{label} SHA256={actual_sha}, expected {expected_sha}")
    return path


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def checkpoint_path(project_root: Path, arm: str, step: int) -> Path:
    return (
        project_root / "outputs/lora_runs" / RUN_DIRS[arm] / f"step-{step}"
    ).resolve()


def quick20_record_paths(
    project_root: Path, step: int, stamp: str
) -> tuple[Path, Path]:
    """Return the legacy-QZ and local-workstation record candidates.

    Batch-44 changed evaluation placement after step 6000.  The already
    running step-8000 local evaluation retained the old ``qz_jobs`` record
    spelling, while step 10000 onward is written under ``local_jobs``.  Keep
    both candidates explicit so consumers can accept the migration record but
    never silently choose between two competing artifacts for one step.
    """

    project_root = project_root.expanduser().resolve()
    name = f"ver23_batch44_quick20_step{step}_{stamp}"
    return (
        project_root / "trainset/qz_jobs" / name,
        project_root / "trainset/local_jobs" / name,
    )


def quick20_record_path(project_root: Path, step: int, stamp: str) -> Path:
    qz_record, local_record = quick20_record_paths(project_root, step, stamp)
    existing = [path for path in (qz_record, local_record) if os.path.lexists(path)]
    if len(existing) > 1:
        raise ValueError(
            f"quick20 step-{step} has conflicting QZ/local record roots: "
            f"{qz_record} and {local_record}"
        )
    if existing:
        return existing[0].resolve()
    # Backward-compatible pending-path spelling.  The local watcher explicitly
    # creates local_jobs for future records before this selector consumes them.
    return qz_record.resolve()


def metrics_path(project_root: Path, step: int, stamp: str) -> Path:
    return quick20_record_path(project_root, step, stamp) / "metrics.json"


def audit_checkpoint(path: Path, *, arm: str, step: int) -> dict[str, Any]:
    if not path.is_dir():
        raise PendingEvidence(f"missing checkpoint directory: {path}")
    if path.name != f"step-{step}":
        raise ValueError(f"checkpoint name drift: {path.name!r}, expected step-{step}")
    files: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_CHECKPOINT_FILES:
        item = path / name
        if not item.is_file() or item.stat().st_size <= 0:
            raise PendingEvidence(f"missing/empty checkpoint file: {item}")
        files[name] = {"path": str(item.resolve()), "size": item.stat().st_size}
    for name in ("adapter_config.json", "timbre_memory_config.json"):
        try:
            json.loads((path / name).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid checkpoint JSON {path / name}: {exc}") from exc
    config = json.loads((path / "timbre_memory_config.json").read_text(encoding="utf-8"))
    expected = {
        "content_cross_attn_enabled": True,
        "content_cross_attn_layers": "all",
        "content_cross_attn_feature_dim": 768,
        "content_cross_attn_gate_init": -0.5,
        "content_cross_attn_output_scale": 0.3,
        "content_encoder_layers": 2,
        "guided_attn_loss_weight": 0.05,
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
        "speaker_condition_dropout": 0.0,
    }
    mismatches = {
        key: {"expected": wanted, "actual": config.get(key)}
        for key, wanted in expected.items()
        if config.get(key) != wanted
    }
    if mismatches:
        raise ValueError(f"{arm} step-{step} Path-X config drift: {mismatches}")
    return {
        "path": str(path),
        "arm": arm,
        "step": step,
        "text_repeat": TEXT_REPEATS[arm],
        "files": files,
    }


def audit_quick20_provenance(
    path: Path, *, project_root: Path, step: int
) -> dict[str, Any]:
    project_root = project_root.expanduser().resolve()
    record = path.parent.resolve()
    prefix = f"ver23_batch44_quick20_step{step}_"
    if not record.name.startswith(prefix):
        raise ValueError(f"unexpected quick20 record directory: {record}")
    stamp = record.name[len(prefix):]
    expected_path = metrics_path(project_root, step, stamp)
    if path.resolve() != expected_path:
        raise ValueError(f"quick20 metrics path={path.resolve()}, expected {expected_path}")
    marker = record / "complete.marker"
    completion_path = record / "COMPLETED.json"
    ledger = record / "submitted_jobs.tsv"
    if not completion_path.is_file() or completion_path.stat().st_size <= 0:
        raise PendingEvidence(f"missing quick20 completion manifest: {completion_path}")
    if not marker.is_file() or marker.stat().st_size <= 0:
        raise PendingEvidence(f"missing quick20 completion marker: {marker}")
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid quick20 completion JSON: {exc}") from exc
    if not isinstance(completion, dict) or not isinstance(marker_payload, dict):
        raise ValueError("quick20 completion and marker must be JSON objects")

    schema = completion.get("schema")
    declared_backend = completion.get("backend")
    if schema == QZ_QUICK20_COMPLETION_SCHEMA and declared_backend in {None, "qz"}:
        backend = "qz"
    elif schema == LOCAL_QUICK20_COMPLETION_SCHEMA and declared_backend == "local":
        backend = "local"
    else:
        raise ValueError(
            f"{completion_path}: unsupported schema/backend "
            f"{schema!r}/{declared_backend!r}"
        )
    qz_record, local_record = quick20_record_paths(project_root, step, stamp)
    qz_record = qz_record.resolve()
    local_record = local_record.resolve()
    if backend == "qz" and record != qz_record:
        raise ValueError(f"{completion_path}: QZ completion must live under qz_jobs")
    if backend == "local":
        if step == 8000:
            allowed_local_records = {qz_record, local_record}
        elif step >= 10000:
            allowed_local_records = {local_record}
        else:
            allowed_local_records = set()
        if record not in allowed_local_records:
            raise ValueError(
                f"{completion_path}: local completion has invalid migration path "
                f"for step-{step}"
            )
    if completion.get("status") != "complete" or completion.get("step") != step:
        raise ValueError(f"{completion_path}: incomplete/wrong step")
    expected_eval_root = (
        project_root / f"testset/outputs/ver23_batch44_quick20_{stamp}"
    ).resolve()
    if Path(str(completion.get("record_root") or "")).resolve() != record:
        raise ValueError(f"{completion_path}: record_root drift")
    if Path(str(completion.get("eval_root") or "")).resolve() != expected_eval_root:
        raise ValueError(f"{completion_path}: eval_root drift")
    if completion.get("training_jobs") != EXPECTED_TRAIN_JOBS:
        raise ValueError(f"{completion_path}: training job provenance drift")
    if backend == "local" and Path(str(completion.get("code_root") or "")).resolve() != Path(
        EXPECTED_EVAL_CODE_ROOT
    ).resolve():
        raise ValueError(f"{completion_path}: local evaluation code_root drift")

    metrics = completion.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != {"json", "tsv", "md"}:
        raise ValueError(f"{completion_path}: metrics artifact set drift")
    for name in ("json", "tsv", "md"):
        require_artifact(
            metrics[name],
            label=f"quick20 {name} metrics",
            expected_path=record / f"metrics.{name}",
        )

    completion_sha = sha256_file(completion_path)
    expected_marker_schema = (
        QZ_QUICK20_MARKER_SCHEMA if backend == "qz" else LOCAL_QUICK20_MARKER_SCHEMA
    )
    expected_marker = {
        "schema": expected_marker_schema,
        "status": "complete",
        "step": step,
        "completed_json_sha256": completion_sha,
    }
    bad_marker = {
        key: marker_payload.get(key)
        for key, wanted in expected_marker.items()
        if marker_payload.get(key) != wanted
    }
    if backend == "local" and marker_payload.get("backend") != "local":
        bad_marker["backend"] = marker_payload.get("backend")
    if bad_marker:
        raise ValueError(f"{marker}: completion marker drift: {bad_marker}")

    if backend == "qz":
        if not ledger.is_file():
            raise PendingEvidence(f"missing quick20 QZ submission ledger: {ledger}")
        with ledger.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        if len(rows) != 1:
            raise ValueError(f"{ledger}: expected one QZ submission row")
        row = rows[0]
        expected = {
            "step": str(step),
            "compute_group": ALLOWED_COMPUTE_GROUP,
            "spec": ALLOWED_SPEC,
        }
        bad = {
            key: row.get(key)
            for key, wanted in expected.items()
            if row.get(key) != wanted
        }
        if bad:
            raise ValueError(f"{ledger}: quick20 MTTS/provenance drift: {bad}")
        expected_paths = {
            "record_root": record,
            "eval_root": expected_eval_root,
            "code_root": Path(EXPECTED_EVAL_CODE_ROOT),
        }
        for key, wanted in expected_paths.items():
            if Path(str(row.get(key) or "")).resolve() != wanted.resolve():
                raise ValueError(f"{ledger}: quick20 path provenance drift: {key}")
        job_id = str(row.get("job_id") or "")
        if not JOB_ID_RE.fullmatch(job_id):
            raise ValueError(f"{ledger}: invalid QZ job id {job_id!r}")
        evaluation_job = completion.get("evaluation_job")
        if not isinstance(evaluation_job, dict):
            raise ValueError(f"{completion_path}: missing QZ evaluation_job")
        if (
            evaluation_job.get("job_id") != job_id
            or evaluation_job.get("job_name") != row.get("job_name")
        ):
            raise ValueError(f"{completion_path}: QZ job identity drift")
        require_artifact(
            evaluation_job.get("submission_ledger"),
            label="quick20 QZ submission ledger",
            expected_path=ledger,
        )
        resource = completion.get("resource_contract")
        expected_resource = {
            "compute_group": "MTTS-3-2-0715",
            "compute_group_id": ALLOWED_COMPUTE_GROUP,
            "spec": ALLOWED_SPEC,
            "instances": 1,
            "gpus": 8,
            "gpu_type": "NVIDIA_H200_SXM_141G",
        }
        if resource != expected_resource:
            raise ValueError(f"{completion_path}: QZ resource contract drift")
        runner = require_artifact(
            completion.get("frozen_runner"),
            label="quick20 QZ frozen runner",
            required_parent=record,
        )
        if runner.name != "004110_submit_batch44_v1_quick20_qz.frozen.sh":
            raise ValueError(f"{completion_path}: unexpected QZ frozen runner {runner.name}")
        evaluation_id = job_id
    else:
        if os.path.lexists(ledger):
            raise ValueError(
                f"{record}: local quick20 must not contain a QZ submission ledger"
            )
        if completion.get("evaluation_job") is not None or completion.get("resource_contract") is not None:
            raise ValueError(f"{completion_path}: local completion contains QZ-only fields")
        execution = completion.get("execution")
        if not isinstance(execution, dict):
            raise ValueError(f"{completion_path}: missing local execution provenance")
        hostname = str(execution.get("hostname") or "")
        if not LOCAL_HOST_RE.fullmatch(hostname):
            raise ValueError(f"{completion_path}: invalid local development hostname {hostname!r}")
        if (
            execution.get("gpu_count") != 2
            or execution.get("gpu_indices") != [0, 1]
            or execution.get("gpu_model") != LOCAL_GPU_MODEL
        ):
            raise ValueError(f"{completion_path}: local RTX 4090 resource contract drift")
        gpus = execution.get("gpus")
        if not isinstance(gpus, list) or len(gpus) != 2:
            raise ValueError(f"{completion_path}: expected two local GPU records")
        for index, gpu in enumerate(gpus):
            if not isinstance(gpu, dict):
                raise ValueError(f"{completion_path}: local GPU record must be an object")
            memory = gpu.get("memory_total_mib")
            if (
                gpu.get("index") != index
                or gpu.get("name") != LOCAL_GPU_MODEL
                or not GPU_UUID_RE.fullmatch(str(gpu.get("uuid") or ""))
                or isinstance(memory, bool)
                or not isinstance(memory, (int, float))
                or float(memory) < 48000.0
            ):
                raise ValueError(f"{completion_path}: invalid local GPU-{index} provenance")
        require_artifact(
            execution.get("runtime_manifest"),
            label="quick20 local runtime manifest",
            required_parent=record,
        )
        runner = require_artifact(
            completion.get("runner"),
            label="quick20 local frozen runner",
            required_parent=record,
        )
        if "004117" not in runner.name or "frozen" not in runner.name:
            raise ValueError(f"{completion_path}: unexpected local frozen runner {runner.name}")
        common_library = require_artifact(
            completion.get("common_library"),
            label="quick20 local frozen common library",
            expected_path=record / "004110_batch44_quick20_common.frozen.sh",
        )
        completion_helper = require_artifact(
            completion.get("completion_helper"),
            label="quick20 local frozen completion helper",
            expected_path=record / "batch44_quick20_local_completion.frozen.py",
        )
        runtime_path = require_artifact(
            execution.get("runtime_manifest"),
            label="quick20 local runtime manifest",
            expected_path=record / "LOCAL_RUNTIME.json",
        )
        try:
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid local runtime manifest {runtime_path}: {exc}") from exc
        if not isinstance(runtime, dict):
            raise ValueError(f"{runtime_path}: local runtime must be a JSON object")
        expected_runtime = {
            "schema": "moss_codecvc.batch44_v1_quick20_local_runtime.v1",
            "backend": "local",
            "status": "started",
            "hostname": hostname,
            "gpu_count": 2,
            "gpu_indices": [0, 1],
            "gpu_model": LOCAL_GPU_MODEL,
        }
        runtime_drift = {
            key: {"expected": wanted, "actual": runtime.get(key)}
            for key, wanted in expected_runtime.items()
            if runtime.get(key) != wanted
        }
        if runtime_drift:
            raise ValueError(f"{runtime_path}: local runtime identity drift: {runtime_drift}")
        if runtime.get("gpus") != gpus:
            raise ValueError(f"{runtime_path}: runtime/completion GPU inventory drift")
        if runtime.get("scheduling") != execution.get("scheduling"):
            raise ValueError(f"{runtime_path}: runtime/completion scheduling drift")
        for key, wanted_path in (
            ("runner", runner),
            ("common_library", common_library),
            ("completion_helper", completion_helper),
        ):
            bound = require_artifact(
                runtime.get(key),
                label=f"quick20 runtime {key}",
                expected_path=wanted_path,
            )
            if bound != wanted_path:
                raise AssertionError("unreachable runtime artifact mismatch")

        fixed_inputs = completion.get("fixed_inputs")
        if not isinstance(fixed_inputs, dict) or set(fixed_inputs) != {
            "no_text20", "text_source", "text20"
        }:
            raise ValueError(f"{completion_path}: local fixed-input set drift")
        fixed_paths = {
            name: require_artifact(
                fixed_inputs[name],
                label=f"quick20 local fixed input {name}",
                expected_path=(
                    project_root / relative
                    if name != "text20"
                    else record / "ver23_batch44_text_quick20_8cell_20260713.jsonl"
                ),
            )
            for name, (relative, _sha) in {
                **LOCAL_FIXED_INPUTS,
                "text20": ("", LOCAL_TEXT20_SHA256),
            }.items()
        }
        if project_root == PROJECT_ROOT.resolve():
            expected_fixed_sha = {
                name: sha for name, (_relative, sha) in LOCAL_FIXED_INPUTS.items()
            }
            expected_fixed_sha["text20"] = LOCAL_TEXT20_SHA256
            for name, fixed_path in fixed_paths.items():
                actual_sha = sha256_file(fixed_path)
                if actual_sha != expected_fixed_sha[name]:
                    raise ValueError(
                        f"{fixed_path}: fixed input SHA256={actual_sha}, "
                        f"expected {expected_fixed_sha[name]}"
                    )

        checkpoints = completion.get("checkpoints")
        if not isinstance(checkpoints, dict) or set(checkpoints) != set(REGISTERED_ARMS):
            raise ValueError(f"{completion_path}: local checkpoint set drift")
        checkpoint_roots: dict[str, Path] = {}
        for arm in REGISTERED_ARMS:
            captured = checkpoints[arm]
            if not isinstance(captured, dict):
                raise ValueError(f"{completion_path}: {arm} checkpoint must be an object")
            checkpoint = checkpoint_path(project_root, arm, step)
            expected_identity = {
                "path": str(checkpoint),
                "step": step,
                "training_job_id": EXPECTED_TRAIN_JOBS[arm],
            }
            drift = {
                key: {"expected": wanted, "actual": captured.get(key)}
                for key, wanted in expected_identity.items()
                if captured.get(key) != wanted
            }
            if drift:
                raise ValueError(f"{completion_path}: {arm} checkpoint drift: {drift}")
            files = captured.get("files")
            if not isinstance(files, dict) or set(files) != set(REQUIRED_CHECKPOINT_FILES):
                raise ValueError(f"{completion_path}: {arm} checkpoint artifact set drift")
            for name in REQUIRED_CHECKPOINT_FILES:
                require_artifact(
                    files[name],
                    label=f"quick20 {arm} checkpoint {name}",
                    expected_path=checkpoint / name,
                )
            checkpoint_roots[arm] = checkpoint

        identity_root = (
            project_root
            / "trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713"
        )
        training = completion.get("training_provenance")
        if not isinstance(training, dict) or set(training) != {"pair_ledger", "train_args"}:
            raise ValueError(f"{completion_path}: local training provenance set drift")
        pair_ledger = require_artifact(
            training["pair_ledger"],
            label="quick20 training pair ledger",
            expected_path=identity_root / "submitted_pair.tsv",
        )
        train_args = training.get("train_args")
        if not isinstance(train_args, dict) or set(train_args) != set(REGISTERED_ARMS):
            raise ValueError(f"{completion_path}: local train-args provenance set drift")
        args_paths = {
            arm: require_artifact(
                train_args[arm],
                label=f"quick20 {arm} train args",
                expected_path=identity_root / arm / "train_args_dry_run_core.json",
            )
            for arm in REGISTERED_ARMS
        }
        if project_root == PROJECT_ROOT.resolve():
            expected_training_sha = {
                pair_ledger: TRAINING_PROVENANCE_SHA256["pair_ledger"],
                args_paths["r3"]: TRAINING_PROVENANCE_SHA256["r3_args"],
                args_paths["r5"]: TRAINING_PROVENANCE_SHA256["r5_args"],
            }
            for provenance_path, wanted_sha in expected_training_sha.items():
                actual_sha = sha256_file(provenance_path)
                if actual_sha != wanted_sha:
                    raise ValueError(
                        f"{provenance_path}: training provenance SHA256={actual_sha}, "
                        f"expected {wanted_sha}"
                    )

        runs = completion.get("runs")
        if not isinstance(runs, list) or len(runs) != 4:
            raise ValueError(f"{completion_path}: local run provenance must contain four rows")
        run_keys: set[tuple[str, str]] = set()
        for run in runs:
            if not isinstance(run, dict):
                raise ValueError(f"{completion_path}: local run row must be an object")
            arm, mode = str(run.get("arm") or ""), str(run.get("mode") or "")
            key = (arm, mode)
            if arm not in REGISTERED_ARMS or mode not in {"no_text", "text"} or key in run_keys:
                raise ValueError(f"{completion_path}: invalid/duplicate local run identity {key}")
            run_keys.add(key)
            run_id = f"ver2_9_5_final_{arm}_step-{step}_{mode}_quick20_d2d3_seed1234"
            output_dir = expected_eval_root / run_id
            expected_run = {
                "run_id": run_id,
                "training_job_id": EXPECTED_TRAIN_JOBS[arm],
                "checkpoint": str(checkpoint_roots[arm]),
                "output_dir": str(output_dir),
            }
            drift = {
                key_name: {"expected": wanted, "actual": run.get(key_name)}
                for key_name, wanted in expected_run.items()
                if run.get(key_name) != wanted
            }
            if drift:
                raise ValueError(f"{completion_path}: local run {key} drift: {drift}")
            artifacts = run.get("artifacts")
            expected_artifacts = {
                "summary": output_dir / f"{run_id}.summary.json",
                "asr": output_dir / f"{run_id}.asr_eval.jsonl",
                "speaker": output_dir / f"{run_id}.speaker_sim.csv",
                "ref_content": output_dir / f"{run_id}.ref_content_similarity_summary.json",
            }
            if not isinstance(artifacts, dict) or set(artifacts) != set(expected_artifacts):
                raise ValueError(f"{completion_path}: local run {key} artifact set drift")
            for name, expected_artifact in expected_artifacts.items():
                require_artifact(
                    artifacts[name],
                    label=f"quick20 local {arm}/{mode} {name}",
                    expected_path=expected_artifact,
                )
        evaluation_id = f"local:{hostname}"

    return {
        "backend": backend,
        "evaluation_id": evaluation_id,
        "completion_json": str(completion_path),
        "completion_sha256": completion_sha,
        "marker": str(marker),
        "marker_sha256": sha256_file(marker),
        "runner": str(runner),
        "runner_sha256": sha256_file(runner),
    }


def load_metrics(
    path: Path, *, project_root: Path, step: int
) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.is_file():
        raise PendingEvidence(f"missing quick20 metrics: {path}")
    audit_quick20_provenance(path, project_root=project_root, step=step)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid quick20 metrics JSON {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"quick20 metrics root must be a list: {path}")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {index} must be an object")
        arm = str(row.get("arm") or "")
        mode = str(row.get("mode") or "")
        key = (arm, mode)
        if arm not in REGISTERED_ARMS or mode not in {"no_text", "text"}:
            raise ValueError(f"{path}: unsupported quick20 identity {key}")
        if key in indexed:
            raise ValueError(f"{path}: duplicate quick20 identity {key}")
        if row.get("step") != step:
            raise ValueError(f"{path}: {key} step={row.get('step')!r}, expected {step}")
        if row.get("n") != EXPECTED_N_PER_MODE:
            raise ValueError(
                f"{path}: {key} n={row.get('n')!r}, expected {EXPECTED_N_PER_MODE}"
            )
        expected_job = EXPECTED_TRAIN_JOBS[arm]
        if row.get("train_job_id") in REJECTED_BATCH43_V2_TRAIN_JOBS:
            raise ValueError(
                f"{path}: {key} references stopped Batch-43 v2 training"
            )
        if row.get("train_job_id") != expected_job:
            raise ValueError(
                f"{path}: {key} train_job_id={row.get('train_job_id')!r}, "
                f"expected {expected_job!r}"
            )
        for metric in ("fail", "cer", "sim_ref", "sim_src", "margin", "ref_bound", "ref_content_f1"):
            finite(row.get(metric), label=f"{path}:{key}.{metric}")
        keep = row.get("keep")
        if isinstance(keep, bool) or not isinstance(keep, int) or not 0 <= keep <= EXPECTED_N_PER_MODE:
            raise ValueError(f"{path}: {key} invalid keep={keep!r}")
        expected_fail = (EXPECTED_N_PER_MODE - keep) / EXPECTED_N_PER_MODE
        if not math.isclose(float(row["fail"]), expected_fail, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{path}: {key} fail/keep mismatch")
        for metric in ("fail", "ref_bound", "ref_content_f1"):
            if not 0.0 <= float(row[metric]) <= 1.0:
                raise ValueError(f"{path}: {key} {metric} outside [0,1]")
        if float(row["cer"]) < 0.0:
            raise ValueError(f"{path}: {key} negative CER")
        if not math.isclose(
            float(row["margin"]),
            float(row["sim_ref"]) - float(row["sim_src"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{path}: {key} margin mismatch")
        expected_run_id = (
            f"ver2_9_5_final_{arm}_step-{step}_{mode}_quick20_d2d3_seed1234"
        )
        if row.get("run_id") != expected_run_id:
            raise ValueError(f"{path}: {key} run_id drift")
        if mode == "text":
            if row.get("text_en_src_quick_n") != 12:
                raise ValueError(f"{path}: {key} text en_src proxy n drift")
            en_fail = finite(
                row.get("text_en_src_quick_fail"), label=f"{path}:{key}.text_en_src_quick_fail"
            )
            if not 0.0 <= en_fail <= 1.0:
                raise ValueError(f"{path}: {key} text en_src proxy outside [0,1]")
        elif any(row.get(field) not in {"", None} for field in (
            "text_en_src_quick_n", "text_en_src_quick_fail", "text_en_src_scope"
        )):
            raise ValueError(f"{path}: {key} no_text contains text en_src proxy")
        indexed[key] = dict(row)
    expected_keys = {
        (arm, mode) for arm in REGISTERED_ARMS for mode in ("no_text", "text")
    }
    if set(indexed) != expected_keys:
        raise ValueError(
            f"{path}: quick20 identities={sorted(indexed)}, expected={sorted(expected_keys)}"
        )
    return indexed


def candidate_from_rows(
    *,
    arm: str,
    step: int,
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    metrics_json: Path,
    completion_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    no_text = rows[(arm, "no_text")]
    text = rows[(arm, "text")]
    pooled_sim_ref = (float(no_text["sim_ref"]) + float(text["sim_ref"])) / 2.0
    pooled_sim_src = (float(no_text["sim_src"]) + float(text["sim_src"])) / 2.0
    pooled_cer = (float(no_text["cer"]) + float(text["cer"])) / 2.0
    candidate_id = f"{arm}_step-{step}"
    return {
        "candidate_id": candidate_id,
        "arm": arm,
        "text_repeat": TEXT_REPEATS[arm],
        "step": step,
        "train_job_id": EXPECTED_TRAIN_JOBS[arm],
        "checkpoint": checkpoint,
        "quick20": {
            "metrics_json": str(metrics_json),
            "metrics_sha256": sha256_file(metrics_json),
            "completion_provenance": dict(completion_provenance),
            "no_text": dict(no_text),
            "text": dict(text),
            "pooled_n": EXPECTED_N_PER_MODE * 2,
            "pooled_wavlm_sim_ref": pooled_sim_ref,
            "pooled_wavlm_sim_src": pooled_sim_src,
            "pooled_margin": pooled_sim_ref - pooled_sim_src,
            "pooled_cer": pooled_cer,
        },
    }


def ranking_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    quick = candidate["quick20"]
    return (
        -float(quick["pooled_wavlm_sim_ref"]),
        -float(quick["no_text"]["sim_ref"]),
        -float(quick["pooled_margin"]),
        float(quick["pooled_cer"]),
        str(candidate["candidate_id"]),
    )


def build_plan(project_root: Path, *, stamp: str) -> dict[str, Any]:
    project_root = project_root.expanduser().resolve()
    candidates: list[dict[str, Any]] = []
    metrics_by_step: dict[int, dict[tuple[str, str], dict[str, Any]]] = {}
    for step in REGISTERED_STEPS:
        metric_path = metrics_path(project_root, step, stamp)
        rows = load_metrics(metric_path, project_root=project_root, step=step)
        completion_provenance = audit_quick20_provenance(
            metric_path, project_root=project_root, step=step
        )
        metrics_by_step[step] = rows
        for arm in REGISTERED_ARMS:
            checkpoint = audit_checkpoint(
                checkpoint_path(project_root, arm, step), arm=arm, step=step
            )
            candidates.append(
                candidate_from_rows(
                    arm=arm,
                    step=step,
                    rows=rows,
                    checkpoint=checkpoint,
                    metrics_json=metric_path,
                    completion_provenance=completion_provenance,
                )
            )
    if len(candidates) != 6:
        raise AssertionError(f"expected exactly six candidates, got {len(candidates)}")
    ranked = sorted(candidates, key=ranking_key)
    selected_ids = {item["candidate_id"] for item in ranked[:3]}
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
        candidate["selected_for_full320"] = candidate["candidate_id"] in selected_ids
    selected = ranked[:3]
    selected_steps = sorted({int(item["step"]) for item in selected})
    paired_extra = sorted(
        f"{arm}_step-{step}"
        for step in selected_steps
        for arm in REGISTERED_ARMS
        if f"{arm}_step-{step}" not in selected_ids
    )
    commands = [
        f"STEP={step} bash scripts/004118_run_batch44_v1_paired_full320_local.sh"
        for step in selected_steps
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "data_version": "v1_20260709",
        "status": "selected",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "registered_candidate_space": {
            "arms": list(REGISTERED_ARMS),
            "steps": list(REGISTERED_STEPS),
            "candidate_count": 6,
        },
        "ranking": {
            "primary": "mean(no_text20_wavlm_sim_ref, text20_wavlm_sim_ref)",
            "primary_scope": "fixed pooled quick40 proxy; full320 not yet used",
            "tie_breaks": [
                "no_text20_wavlm_sim_ref descending",
                "pooled_wavlm_margin descending",
                "pooled_CER ascending",
                "candidate_id ascending",
            ],
            "warning": (
                "Best3 is an evaluation shortlist, not the final model. Full320 and "
                "blind20 evidence are still mandatory."
            ),
        },
        "candidates": ranked,
        "selected_candidate_ids": [item["candidate_id"] for item in selected],
        "paired_full320_plan": {
            "selected_steps": selected_steps,
            "commands_default_dry_run": commands,
            "best3_wrapper": "scripts/004118_run_batch44_v1_paired_full320_local.sh",
            "execution_default": "plan_only; ACTION=run and CONFIRM_LOCAL_FULL320=1 are explicit",
            "selected_candidates": [item["candidate_id"] for item in selected],
            "extra_counterparts_evaluated_by_paired_wrapper": paired_extra,
            "note": (
                "004118 preserves same-step r3/r5 pairing on the local RTX 4090 host. "
                "It can therefore evaluate "
                "an unselected counterpart at a selected step; only the registered "
                "Best3 advance to blind20."
            ),
        },
    }


def pending_plan(project_root: Path, *, stamp: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "data_version": "v1_20260709",
        "status": "pending",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "registered_candidate_space": {
            "arms": list(REGISTERED_ARMS),
            "steps": list(REGISTERED_STEPS),
            "candidate_count": 6,
        },
        "reason": reason,
        "required_quick20_metrics": [
            str(metrics_path(project_root, step, stamp)) for step in REGISTERED_STEPS
        ],
        "required_checkpoints": [
            str(checkpoint_path(project_root, arm, step))
            for step in REGISTERED_STEPS
            for arm in REGISTERED_ARMS
        ],
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = ["# Batch-44 v1 Best3 selection", ""]
    if payload.get("status") != "selected":
        lines.extend([f"- Status: **pending**", f"- Reason: `{payload.get('reason')}`", ""])
        return "\n".join(lines)
    lines.extend(
        [
            "- Ranking evidence: pooled fixed quick20 no_text + text WavLM SIM(ref).",
            "- This is only the full-320 shortlist; final selection still requires blind20.",
            "",
            "| Rank | Candidate | repeat | pooled SIM(ref) | no_text SIM(ref) | text SIM(ref) | pooled margin | pooled CER | Best3 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["candidates"]:
        quick = row["quick20"]
        lines.append(
            f"| {row['rank']} | {row['candidate_id']} | {row['text_repeat']} | "
            f"{quick['pooled_wavlm_sim_ref']:.4f} | {quick['no_text']['sim_ref']:.4f} | "
            f"{quick['text']['sim_ref']:.4f} | {quick['pooled_margin']:.4f} | "
            f"{quick['pooled_cer']:.4f} | {'yes' if row['selected_for_full320'] else 'no'} |"
        )
    lines.extend(["", "## Default dry-run commands", ""])
    lines.extend(
        f"- `{command}`"
        for command in payload["paired_full320_plan"]["commands_default_dry_run"]
    )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--quick20-stamp", default="20260713")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT
        / "testset/outputs/batch44_best3_20260713/best3_selection.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=PROJECT_ROOT
        / "testset/outputs/batch44_best3_20260713/best3_selection.md",
    )
    parser.add_argument(
        "--no-pending-output",
        action="store_true",
        help="do not write the pending plan when future evidence is absent",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    try:
        payload = build_plan(project_root, stamp=args.quick20_stamp)
        status = 0
    except PendingEvidence as exc:
        payload = pending_plan(project_root, stamp=args.quick20_stamp, reason=str(exc))
        status = 3
    if status == 0 or not args.no_pending_output:
        atomic_json(args.output_json.expanduser().resolve(), payload)
        md_path = args.output_md.expanduser().resolve()
        md_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = md_path.with_name(f".{md_path.name}.tmp-{os.getpid()}")
        temporary.write_text(render_markdown(payload) + "\n", encoding="utf-8")
        os.replace(temporary, md_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return status


if __name__ == "__main__":
    sys.exit(main())

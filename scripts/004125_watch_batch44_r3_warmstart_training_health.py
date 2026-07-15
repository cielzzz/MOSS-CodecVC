#!/usr/bin/env python3
"""Read-only health watcher for the Batch-44 r3 10k->30k warm start.

The continuation is a weights-only warm start.  Its local optimizer step runs
from 0 to 20,000 and maps to effective step 10,000 to 30,000.  This watcher
hard-binds the submitted job, warm-start contract, output directory and the
MTTS one-node 8xH200 resource contract.  It only calls::

    qzcli status JOB_ID --json

It never logs in, submits, stops, restarts, or mutates the remote task.  Local
state files under the dedicated watcher root are observational evidence only.

One-shot audit::

    python scripts/004125_watch_batch44_r3_warmstart_training_health.py --mode once

The monitor mode is intentionally not started by this file::

    python scripts/004125_watch_batch44_r3_warmstart_training_health.py --mode monitor
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


CANONICAL_PROJECT_ROOT = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
    "MOSS-CodecVC"
)
CANONICAL_QZCLI = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
    "pair_construction/scripts/qzcli_with_deps.sh"
)
CANONICAL_QZCLI_HOME = Path(
    "/inspire/ssd/project/embodied-multimodality/public/xyzhang/.codex/qzcli_home"
)

RECORD_RELATIVE = Path(
    "trainset/qz_jobs/ver23_batch44_r3_v1_warmstart10k_to30k_20260713"
)
STATE_RELATIVE = Path(
    "trainset/qz_jobs/batch44_r3_warmstart_training_health_20260713"
)
RUN_RELATIVE = Path(
    "outputs/lora_runs/ver2_9_5_final_r3_v1_warmstart10k_to30k"
)
SOURCE_CHECKPOINT_RELATIVE = Path(
    "outputs/lora_runs/ver2_9_5_final_r3_v1_30k/step-10000"
)

JOB_ID = "job-165f3b1d-8c7d-47d8-8882-bdb86ef642ab"
JOB_NAME = "ver2_9_5_final_r3_v1_warmstart10k_to30k"
WARM_START_CONTRACT_SHA256 = (
    "2d686e5e57b70fcaa3db8c8eb2b306003a38599b2c9ac37023979d80b6d9fc34"
)

ALLOWED_WORKSPACE_ID = "ws-8207e9e2-e733-4eec-a475-cfa1c36480ba"
ALLOWED_PROJECT_ID = "project-c67c548f-f02c-453b-ba5b-8745db6886e7"
ALLOWED_COMPUTE_GROUP_ID = "lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"
ALLOWED_COMPUTE_GROUP_NAME = "MTTS-3-2-0715"
ALLOWED_SPEC_ID = "67b10bc6-78b0-41a3-aaf4-358eeeb99009"
ALLOWED_GPU_TYPE = "NVIDIA_H200_SXM_141G"
ALLOWED_INSTANCES = 1
ALLOWED_GPUS = 8

SOURCE_EFFECTIVE_STEP = 10_000
LOCAL_TARGET_STEP = 20_000
EFFECTIVE_TARGET_STEP = 30_000
SAVE_STEPS = 2_000
POLL_SECONDS = 300
QZ_TIMEOUT_SECONDS = 90
LOG_STALL_SECONDS = 15 * 60
CHECKPOINT_GRACE_SECONDS = 15 * 60
TMP_STALE_WARN_SECONDS = 30 * 60
TMP_STALE_CRITICAL_SECONDS = 90 * 60

EXPECTED_MODEL_FILES = {
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "timbre_memory_config.json",
    "timbre_memory_adapter.pt",
}
STRICT_RESUME_STATE_FILES = {
    "optimizer.bin",
    "optimizer.pt",
    "scheduler.bin",
    "scheduler.pt",
    "random_states_0.pkl",
    "trainer_state.json",
}
TERMINAL_QZ_STATUSES = {"job_succeeded", "job_failed", "job_stopped"}
ACCEPTED_QZ_STATUSES = {
    "job_pending",
    "job_queued",
    "job_running",
    *TERMINAL_QZ_STATUSES,
}
JOB_RE = re.compile(
    r"^job-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
PROGRESS_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+"
    r"step=(?P<step>\d+)/(?P<target>\d+)\b(?P<body>.*)$"
)
FINISHED_RE = re.compile(r"\bfinished global_step=(?P<step>\d+)\b")
NONFINITE_RE = re.compile(
    r"(?:^|[=:\s,\[(])(?:nan|[+-]?inf(?:inity)?)(?=$|[\s,;\])])",
    re.IGNORECASE,
)
FATAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("traceback", re.compile(r"traceback \(most recent call last\)", re.I)),
    ("oom", re.compile(r"(?:cuda\s+)?out of memory|cudnn_status_alloc_failed", re.I)),
    ("nccl", re.compile(r"nccl[^\n]*(?:error|failed|failure|timeout|abort)", re.I)),
    ("runtime_error", re.compile(r"\bruntimeerror\b", re.I)),
    ("fatal", re.compile(r"\bfatal\b", re.I)),
    ("process_killed", re.compile(r"\b(?:sigkill|segmentation fault|exitcode\s*[:=]\s*-9)\b", re.I)),
)
AUTH_ERROR_RE = re.compile(
    r"(?:unauthori[sz]ed|authentication required|please\s+(?:re-)?login|"
    r"not logged in|cookie[^\n]*(?:expired|invalid)|未登录|登录[^\n]*(?:过期|失效))",
    re.I,
)


class ContractError(RuntimeError):
    """A hard-bound watcher or job contract was violated."""


@dataclass(frozen=True)
class Config:
    project_root: Path
    record_root: Path
    state_root: Path
    run_dir: Path
    source_checkpoint: Path
    qzcli: Path
    qzcli_home: Path
    expected_contract_sha256: str
    mode: str
    poll_seconds: int
    max_scans: int
    test_mode: bool
    qz_timeout_seconds: int
    log_stall_seconds: int
    checkpoint_grace_seconds: int
    tmp_stale_warn_seconds: int
    tmp_stale_critical_seconds: int


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def format_utc(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> dt.datetime:
    return dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=dt.timezone.utc
    )


def lexical_absolute(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(value))))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON object required: {path}")
    return value


def env_int(name: str, default: int, *, test_mode: bool) -> int:
    if name not in os.environ:
        return default
    if not test_mode:
        raise ContractError(f"production {name} is hard-locked")
    try:
        value = int(os.environ[name])
    except ValueError as exc:
        raise ContractError(f"{name} must be an integer") from exc
    if value < 0:
        raise ContractError(f"{name} must be non-negative")
    return value


def build_config(args: argparse.Namespace) -> Config:
    test_flag = os.environ.get("BATCH44_WARMSTART_HEALTH_TEST_MODE", "0")
    if test_flag not in {"0", "1"}:
        raise ContractError("BATCH44_WARMSTART_HEALTH_TEST_MODE must be 0 or 1")
    test_mode = test_flag == "1"
    canonical = CANONICAL_PROJECT_ROOT.resolve()
    project_root = Path(os.environ.get("PROJECT_ROOT", str(canonical))).resolve()
    record_root = lexical_absolute(
        os.environ.get("RECORD_ROOT", str(project_root / RECORD_RELATIVE))
    )
    state_root = lexical_absolute(
        os.environ.get("STATE_ROOT", str(project_root / STATE_RELATIVE))
    )
    run_dir = lexical_absolute(
        os.environ.get("RUN_DIR", str(project_root / RUN_RELATIVE))
    )
    source_checkpoint = lexical_absolute(
        os.environ.get(
            "SOURCE_CHECKPOINT", str(project_root / SOURCE_CHECKPOINT_RELATIVE)
        )
    )
    qzcli = Path(os.environ.get("QZCLI", str(CANONICAL_QZCLI))).resolve()
    qzcli_home = Path(
        os.environ.get("QZCLI_HOME", str(CANONICAL_QZCLI_HOME))
    ).resolve()
    expected_sha = os.environ.get(
        "WARM_START_CONTRACT_SHA256", WARM_START_CONTRACT_SHA256
    )
    if not test_mode:
        if project_root != canonical:
            raise ContractError("production PROJECT_ROOT is hard-locked")
        expected_paths = {
            "RECORD_ROOT": lexical_absolute(canonical / RECORD_RELATIVE),
            "STATE_ROOT": lexical_absolute(canonical / STATE_RELATIVE),
            "RUN_DIR": lexical_absolute(canonical / RUN_RELATIVE),
            "SOURCE_CHECKPOINT": lexical_absolute(canonical / SOURCE_CHECKPOINT_RELATIVE),
        }
        actual_paths = {
            "RECORD_ROOT": record_root,
            "STATE_ROOT": state_root,
            "RUN_DIR": run_dir,
            "SOURCE_CHECKPOINT": source_checkpoint,
        }
        if actual_paths != expected_paths:
            raise ContractError(f"production path contract drift: {actual_paths}")
        if qzcli != CANONICAL_QZCLI.resolve() or qzcli_home != CANONICAL_QZCLI_HOME.resolve():
            raise ContractError("production qzcli wrapper/HOME are hard-locked")
        if expected_sha != WARM_START_CONTRACT_SHA256:
            raise ContractError("production warm-start contract SHA is hard-locked")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ContractError("WARM_START_CONTRACT_SHA256 must be lowercase SHA256")
    if args.mode not in {"once", "monitor"}:
        raise ContractError("--mode must be once or monitor")
    poll_seconds = args.poll_seconds if args.poll_seconds is not None else POLL_SECONDS
    if poll_seconds <= 0:
        raise ContractError("--poll-seconds must be positive")
    if not test_mode and poll_seconds != POLL_SECONDS:
        raise ContractError("production polling is hard-locked to 300 seconds")
    if args.max_scans < 0:
        raise ContractError("--max-scans must be non-negative")
    return Config(
        project_root=project_root,
        record_root=record_root,
        state_root=state_root,
        run_dir=run_dir,
        source_checkpoint=source_checkpoint,
        qzcli=qzcli,
        qzcli_home=qzcli_home,
        expected_contract_sha256=expected_sha,
        mode=args.mode,
        poll_seconds=poll_seconds,
        max_scans=args.max_scans,
        test_mode=test_mode,
        qz_timeout_seconds=env_int(
            "BATCH44_WARMSTART_HEALTH_QZ_TIMEOUT_SECONDS",
            QZ_TIMEOUT_SECONDS,
            test_mode=test_mode,
        ),
        log_stall_seconds=env_int(
            "BATCH44_WARMSTART_HEALTH_LOG_STALL_SECONDS",
            LOG_STALL_SECONDS,
            test_mode=test_mode,
        ),
        checkpoint_grace_seconds=env_int(
            "BATCH44_WARMSTART_HEALTH_CHECKPOINT_GRACE_SECONDS",
            CHECKPOINT_GRACE_SECONDS,
            test_mode=test_mode,
        ),
        tmp_stale_warn_seconds=env_int(
            "BATCH44_WARMSTART_HEALTH_TMP_WARN_SECONDS",
            TMP_STALE_WARN_SECONDS,
            test_mode=test_mode,
        ),
        tmp_stale_critical_seconds=env_int(
            "BATCH44_WARMSTART_HEALTH_TMP_CRITICAL_SECONDS",
            TMP_STALE_CRITICAL_SECONDS,
            test_mode=test_mode,
        ),
    )


def validate_resource_payload(payload: Mapping[str, Any], runner: Path) -> None:
    expected = {
        "name": JOB_NAME,
        "workspace_id": ALLOWED_WORKSPACE_ID,
        "project_id": ALLOWED_PROJECT_ID,
        "logic_compute_group_id": ALLOWED_COMPUTE_GROUP_ID,
    }
    drift = {key: payload.get(key) for key, value in expected.items() if payload.get(key) != value}
    if drift:
        raise ContractError(f"QZ identity/resource payload drift: {drift}")
    framework = payload.get("framework_config")
    if not isinstance(framework, list) or len(framework) != 1 or not isinstance(framework[0], dict):
        raise ContractError("QZ payload must contain exactly one framework_config entry")
    item = framework[0]
    resource = item.get("resource_spec_price") or item.get("instance_spec_price_info") or {}
    gpu = resource.get("gpu_info") or {}
    gpu_type = gpu.get("gpu_type") or resource.get("gpu_type")
    if (
        item.get("instance_count") != ALLOWED_INSTANCES
        or item.get("gpu_count") != ALLOWED_GPUS
        or resource.get("quota_id") != ALLOWED_SPEC_ID
        or gpu_type != ALLOWED_GPU_TYPE
    ):
        raise ContractError("QZ payload is not MTTS registered spec / one 8xH200 node")
    if str(payload.get("command") or "").strip() != f"sh {runner}":
        raise ContractError(f"QZ command is not bound to {runner}")


def validate_static_contract(config: Config) -> dict[str, Any]:
    if not config.qzcli.is_file() or not os.access(config.qzcli, os.X_OK):
        raise ContractError(f"qzcli-local wrapper is not executable: {config.qzcli}")
    if not config.qzcli_home.is_dir():
        raise ContractError(f"qzcli-local HOME is missing: {config.qzcli_home}")
    if config.tmp_stale_critical_seconds < config.tmp_stale_warn_seconds:
        raise ContractError("hidden-tmp critical threshold must be >= warning threshold")
    if not JOB_RE.fullmatch(JOB_ID):
        raise ContractError("hard-bound QZ job ID is invalid")

    contract_path = config.record_root / "warm_start_contract.json"
    try:
        actual_contract_sha = file_sha256(contract_path)
    except OSError as exc:
        raise ContractError(f"missing warm-start contract: {contract_path}") from exc
    if actual_contract_sha != config.expected_contract_sha256:
        raise ContractError(
            "warm_start_contract SHA drift: "
            f"{actual_contract_sha} != {config.expected_contract_sha256}"
        )
    contract = load_json(contract_path)
    expected_contract = {
        "schema": "batch44_r3_weights_only_warm_start_v1",
        "status": "submitted",
        "job_id": JOB_ID,
        "job_name": JOB_NAME,
        "output_dir": str(config.run_dir),
        "source_checkpoint": str(config.source_checkpoint),
        "source_effective_step": SOURCE_EFFECTIVE_STEP,
        "continuation_local_target_step": LOCAL_TARGET_STEP,
        "effective_step_offset": SOURCE_EFFECTIVE_STEP,
        "effective_target_step": EFFECTIVE_TARGET_STEP,
        "resume_semantics": "weights_only_warm_start_not_exact_resume",
        "full_data_sha256_verified": True,
    }
    bad = {
        key: contract.get(key)
        for key, value in expected_contract.items()
        if contract.get(key) != value
    }
    if bad:
        raise ContractError(f"warm-start contract field drift: {bad}")
    overrides = contract.get("mechanical_recovery_overrides") or {}
    if overrides.get("warmup_ratio") != 0.0 or overrides.get("guided_attn_warmup_steps") != 0:
        raise ContractError("warm-start contract lost the two zero-warmup overrides")

    runner = config.record_root / "run_train_entrypoint.sh"
    if not runner.is_file():
        raise ContractError(f"missing frozen continuation runner: {runner}")
    runner_text = runner.read_text(encoding="utf-8")
    required_runner_fragments = (
        f'OUT_DIR="{config.run_dir}"',
        f'RESUME_ADAPTER_PATH="{config.source_checkpoint}"',
        "text.train.jsonl::repeat=3",
        'lr_scheduler_type=constant_with_warmup warmup_ratio=0.0',
        'guided_weight=0.05 guided_warmup=0',
        '--max-train-steps "20000"',
        '--save-steps "2000"',
        'RESUME_ARGS="--resume-adapter-path $RESUME_ADAPTER_PATH"',
    )
    missing = [item for item in required_runner_fragments if item not in runner_text]
    if missing:
        raise ContractError(f"continuation runner binding drift: {missing}")

    ledger_path = config.record_root / "submitted_jobs.tsv"
    try:
        with ledger_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    except OSError as exc:
        raise ContractError(f"missing continuation ledger: {ledger_path}") from exc
    if len(rows) != 1:
        raise ContractError(f"continuation ledger must contain one row, got {len(rows)}")
    expected_row = {
        "job_name": JOB_NAME,
        "job_id": JOB_ID,
        "compute_group": ALLOWED_COMPUTE_GROUP_ID,
        "runner": str(runner),
        "out_dir": str(config.run_dir),
    }
    if rows[0] != expected_row:
        raise ContractError(f"continuation ledger drift: {rows[0]}")

    payload = load_json(config.record_root / "qz_payload.json")
    validate_resource_payload(payload, runner)
    return {
        "warm_start_contract_path": str(contract_path),
        "warm_start_contract_sha256": actual_contract_sha,
        "runner": str(runner),
        "source_checkpoint": str(config.source_checkpoint),
        "resume_semantics": contract["resume_semantics"],
    }


def add_alert(
    alerts: list[dict[str, Any]],
    *,
    alert_id: str,
    severity: str,
    summary: str,
    evidence: Any,
    recommendation: str,
) -> None:
    alerts.append(
        {
            "id": alert_id,
            "severity": severity,
            "arm": "r3-warmstart",
            "summary": summary,
            "evidence": evidence,
            "recommendation": recommendation,
        }
    )


def extract_qz_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    matches: dict[str, dict[str, Any]] = {}
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("job_id") == JOB_ID:
            matches[json.dumps(payload, ensure_ascii=False, sort_keys=True)] = payload
    if len(matches) != 1:
        raise ContractError(
            f"qzcli status output must contain one API JSON for {JOB_ID}; got {len(matches)}"
        )
    return next(iter(matches.values()))


def audit_live_qz_payload(payload: Mapping[str, Any], config: Config) -> str:
    expected = {
        "job_id": JOB_ID,
        "name": JOB_NAME,
        "workspace_id": ALLOWED_WORKSPACE_ID,
        "project_id": ALLOWED_PROJECT_ID,
        "logic_compute_group_id": ALLOWED_COMPUTE_GROUP_ID,
        "logic_compute_group_name": ALLOWED_COMPUTE_GROUP_NAME,
    }
    drift = {key: payload.get(key) for key, value in expected.items() if payload.get(key) != value}
    if drift:
        raise ContractError(f"live QZ identity/workspace/compute-group drift: {drift}")
    validate_resource_payload(payload, config.record_root / "run_train_entrypoint.sh")
    status = str(payload.get("status") or "").strip().lower()
    if status not in ACCEPTED_QZ_STATUSES:
        raise ContractError(f"unrecognized QZ status: {status!r}")
    return status


def query_qz_status(config: Config) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    alerts: list[dict[str, Any]] = []
    environment = os.environ.copy()
    for key in (
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        environment.pop(key, None)
    environment["HOME"] = str(config.qzcli_home)
    command = [str(config.qzcli), "status", JOB_ID, "--json"]
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            env=environment,
            timeout=config.qz_timeout_seconds,
        )
        raw = result.stdout + ("\n" + result.stderr if result.stderr else "")
        returncode: int | None = result.returncode
    except subprocess.TimeoutExpired as exc:
        raw = f"qzcli status timed out after {config.qz_timeout_seconds}s: {exc}"
        returncode = None
    except OSError as exc:
        raw = f"qzcli status could not execute: {exc}"
        returncode = None
    atomic_write_text(config.state_root / "qz_status_latest.txt", raw[-200_000:])
    observed: dict[str, Any] = {
        "queried": True,
        "query_command": ["qzcli-local", "status", JOB_ID, "--json"],
        "query_returncode": returncode,
        "status": "query_error",
        "authentication_error": False,
    }
    if returncode != 0:
        is_auth = bool(AUTH_ERROR_RE.search(raw))
        observed["authentication_error"] = is_auth
        observed["query_error_excerpt"] = raw[-2000:]
        add_alert(
            alerts,
            alert_id=(
                "warmstart:qz_authentication_required"
                if is_auth
                else "warmstart:qz_query_unavailable"
            ),
            severity="warning",
            summary=(
                "qzcli-local 明确返回认证失效；watcher 未尝试自动登录"
                if is_auth
                else "QZ 状态查询不可用；watcher 未重试或修改任务"
            ),
            evidence={"returncode": returncode, "excerpt": raw[-1000:]},
            recommendation="人工核验 qzcli/network；不要据此自动 stop/restart。",
        )
        return observed, alerts
    try:
        payload = extract_qz_json(raw)
        status = audit_live_qz_payload(payload, config)
    except ContractError as exc:
        observed["query_error_excerpt"] = str(exc)
        add_alert(
            alerts,
            alert_id="warmstart:qz_resource_or_identity_drift",
            severity="critical",
            summary="QZ 实时身份或 MTTS/1x8 H200 资源合同漂移",
            evidence=str(exc),
            recommendation="暂停使用 watcher 结果并人工核验平台；watcher 不会修改任务。",
        )
        return observed, alerts
    observed.update(
        {
            "status": status,
            "job_id": JOB_ID,
            "job_name": JOB_NAME,
            "workspace_id": ALLOWED_WORKSPACE_ID,
            "project_id": ALLOWED_PROJECT_ID,
            "compute_group_id": ALLOWED_COMPUTE_GROUP_ID,
            "compute_group_name": ALLOWED_COMPUTE_GROUP_NAME,
            "spec_id": ALLOWED_SPEC_ID,
            "instances": ALLOWED_INSTANCES,
            "gpus": ALLOWED_GPUS,
            "gpu_type": ALLOWED_GPU_TYPE,
        }
    )
    return observed, alerts


def read_previous_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def parse_train_log(path: Path, now: dt.datetime) -> dict[str, Any]:
    empty = {
        "exists": False,
        "latest_local_step": 0,
        "latest_effective_step": SOURCE_EFFECTIVE_STEP,
        "target_local_step": LOCAL_TARGET_STEP,
        "target_effective_step": EFFECTIVE_TARGET_STEP,
        "observed_targets": [],
        "progress_records": 0,
        "fatal_matches": [],
        "nonfinite_matches": [],
        "finished_local_step": None,
    }
    if not path.is_file():
        return empty
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {**empty, "exists": True, "read_error": str(exc)}
    records: list[tuple[dt.datetime, int, int]] = []
    fatal_matches: list[dict[str, Any]] = []
    nonfinite_matches: list[dict[str, Any]] = []
    finished_step: int | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = PROGRESS_RE.match(line)
        if match:
            records.append(
                (
                    parse_utc(match.group("timestamp")),
                    int(match.group("step")),
                    int(match.group("target")),
                )
            )
            if NONFINITE_RE.search(match.group("body")):
                nonfinite_matches.append({"line": line_number, "text": line[-1000:]})
        finished = FINISHED_RE.search(line)
        if finished:
            finished_step = int(finished.group("step"))
        for label, pattern in FATAL_PATTERNS:
            if pattern.search(line):
                fatal_matches.append(
                    {"kind": label, "line": line_number, "text": line[-1000:]}
                )
    latest = records[-1] if records else None
    intervals: list[float] = []
    rate_window = records[-61:]
    for previous, current in zip(rate_window, rate_window[1:]):
        delta_step = current[1] - previous[1]
        delta_seconds = (current[0] - previous[0]).total_seconds()
        if delta_step > 0 and delta_seconds > 0:
            intervals.append(delta_seconds / delta_step)
    seconds_per_step = statistics.median(intervals) if intervals else None
    latest_time = latest[0] if latest else None
    latest_step = latest[1] if latest else 0
    mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
    activity_time = max(item for item in (latest_time, mtime) if item is not None)
    eta = None
    if latest_time and seconds_per_step and latest_step < LOCAL_TARGET_STEP:
        eta = latest_time + dt.timedelta(
            seconds=(LOCAL_TARGET_STEP - latest_step) * seconds_per_step
        )
    return {
        "exists": True,
        "path": str(path),
        "bytes": path.stat().st_size,
        "mtime_utc": format_utc(mtime),
        "latest_local_step": latest_step,
        "latest_effective_step": SOURCE_EFFECTIVE_STEP + latest_step,
        "target_local_step": LOCAL_TARGET_STEP,
        "target_effective_step": EFFECTIVE_TARGET_STEP,
        "observed_targets": sorted({item[2] for item in records}),
        "progress_records": len(records),
        "latest_progress_utc": format_utc(latest_time),
        "activity_age_seconds": max(0.0, (now - activity_time).total_seconds()),
        "seconds_per_step": seconds_per_step,
        "eta_utc": format_utc(eta),
        "finished_local_step": finished_step,
        "fatal_matches": fatal_matches[-20:],
        "nonfinite_matches": nonfinite_matches[-20:],
    }


def checkpoint_inventory(config: Config, log: Mapping[str, Any], now: dt.datetime) -> dict[str, Any]:
    complete: list[int] = []
    incomplete: list[dict[str, Any]] = []
    strict_resume: list[int | str] = []
    save_errors: list[dict[str, Any]] = []
    for local_step in range(SAVE_STEPS, LOCAL_TARGET_STEP + 1, SAVE_STEPS):
        directory = config.run_dir / f"step-{local_step}"
        if not directory.is_dir():
            continue
        files = {item.name for item in directory.iterdir() if item.is_file()}
        missing = sorted(EXPECTED_MODEL_FILES - files)
        if missing:
            incomplete.append(
                {
                    "local_step": local_step,
                    "effective_step": SOURCE_EFFECTIVE_STEP + local_step,
                    "path": str(directory),
                    "missing": missing,
                }
            )
        else:
            complete.append(local_step)
        if files & STRICT_RESUME_STATE_FILES:
            strict_resume.append(local_step)
        error = directory / "checkpoint_save_error.txt"
        if error.is_file():
            save_errors.append(
                {"path": str(error), "excerpt": error.read_text(encoding="utf-8", errors="replace")[-2000:]}
            )
    final_dir = config.run_dir / "final"
    final_complete = False
    if final_dir.is_dir():
        final_files = {item.name for item in final_dir.iterdir() if item.is_file()}
        final_complete = not (EXPECTED_MODEL_FILES - final_files)
        if final_files & STRICT_RESUME_STATE_FILES:
            strict_resume.append("final")
        error = final_dir / "checkpoint_save_error.txt"
        if error.is_file():
            save_errors.append(
                {"path": str(error), "excerpt": error.read_text(encoding="utf-8", errors="replace")[-2000:]}
            )
    hidden_tmp: list[dict[str, Any]] = []
    if config.run_dir.is_dir():
        for path in sorted(config.run_dir.glob(".step-*.tmp-*")):
            if not path.is_dir():
                continue
            age = max(0.0, (now.timestamp() - path.stat().st_mtime))
            severity = "active"
            if age >= config.tmp_stale_critical_seconds:
                severity = "critical"
            elif age >= config.tmp_stale_warn_seconds:
                severity = "warning"
            hidden_tmp.append({"path": str(path), "age_seconds": age, "severity": severity})
    latest_step = int(log.get("latest_local_step") or 0)
    rate = log.get("seconds_per_step")
    overdue: list[int] = []
    for local_step in range(SAVE_STEPS, min(latest_step, LOCAL_TARGET_STEP) + 1, SAVE_STEPS):
        if local_step in complete:
            continue
        updates_past = max(0, latest_step - local_step)
        estimated_seconds_past = (
            updates_past * float(rate) if isinstance(rate, (int, float)) else None
        )
        if estimated_seconds_past is not None and estimated_seconds_past >= config.checkpoint_grace_seconds:
            overdue.append(local_step)
    return {
        "save_steps": SAVE_STEPS,
        "complete_local_steps": complete,
        "complete_effective_steps": [SOURCE_EFFECTIVE_STEP + item for item in complete],
        "incomplete": incomplete,
        "overdue_local_steps": overdue,
        "overdue_effective_steps": [SOURCE_EFFECTIVE_STEP + item for item in overdue],
        "save_errors": save_errors,
        "hidden_tmp": hidden_tmp,
        "final_exists": final_dir.is_dir(),
        "final_complete": final_complete,
        "strict_resume_state_locations": strict_resume,
        "strict_resume_available": bool(strict_resume),
        "resume_contract": "weights_only continuation checkpoints; exact optimizer/scheduler/RNG resume is not guaranteed",
    }


def analyze(
    config: Config,
    qz: Mapping[str, Any],
    previous: Mapping[str, Any],
    now: dt.datetime,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    alerts: list[dict[str, Any]] = []
    log = parse_train_log(config.run_dir / "train.log", now)
    latest = int(log.get("latest_local_step") or 0)
    previous_local = previous.get("latest_local_step")
    delta = latest - int(previous_local) if isinstance(previous_local, int) else None
    log["previous_observed_local_step"] = previous_local
    log["local_step_delta_since_previous_scan"] = delta
    log["progressing_since_previous_scan"] = delta is not None and delta > 0
    checkpoints = checkpoint_inventory(config, log, now)
    qz_status = str(qz.get("status") or "query_error")

    if not log.get("exists"):
        add_alert(
            alerts,
            alert_id="warmstart:train_log_missing",
            severity="critical" if qz_status == "job_running" else "warning",
            summary="continuation train.log 缺失",
            evidence=str(config.run_dir / "train.log"),
            recommendation="人工核验共享挂载和远端日志；不要自动 restart。",
        )
    if log.get("read_error"):
        add_alert(
            alerts,
            alert_id="warmstart:train_log_unreadable",
            severity="warning",
            summary="continuation train.log 暂时不可读",
            evidence=log["read_error"],
            recommendation="检查 QB I/O，watcher 下轮继续只读观察。",
        )
    observed_targets = log.get("observed_targets") or []
    if observed_targets and observed_targets != [LOCAL_TARGET_STEP]:
        add_alert(
            alerts,
            alert_id="warmstart:local_target_drift",
            severity="critical",
            summary="train.log 的 local target 不是硬绑定 20k",
            evidence=observed_targets,
            recommendation="立即核验 runner/job 身份；watcher 不会修改任务。",
        )
    if latest > LOCAL_TARGET_STEP:
        add_alert(
            alerts,
            alert_id="warmstart:local_step_exceeds_target",
            severity="critical",
            summary="continuation local step 超过硬绑定 20k",
            evidence={"latest_local_step": latest},
            recommendation="核验是否错误重用输出目录或修改 max_train_steps。",
        )
    if log.get("fatal_matches"):
        add_alert(
            alerts,
            alert_id="warmstart:fatal_log_signature",
            severity="critical",
            summary="train.log 出现 OOM/NCCL/RuntimeError/fatal 类错误",
            evidence=log["fatal_matches"],
            recommendation="保留现场并人工决定处理；watcher 不会 stop。",
        )
    if log.get("nonfinite_matches"):
        add_alert(
            alerts,
            alert_id="warmstart:nonfinite_training_metric",
            severity="critical",
            summary="train.log 训练指标出现 NaN/Inf",
            evidence=log["nonfinite_matches"],
            recommendation="人工检查数值发散；watcher 不会 stop。",
        )
    if delta is not None and delta < 0:
        add_alert(
            alerts,
            alert_id="warmstart:step_regression",
            severity="critical",
            summary="continuation local step 相对上次扫描回退",
            evidence={"previous": previous_local, "current": latest},
            recommendation="核验日志截断、输出身份或二次 warm start。",
        )
    age = log.get("activity_age_seconds")
    if (
        qz_status == "job_running"
        and latest < LOCAL_TARGET_STEP
        and isinstance(age, (int, float))
        and age >= config.log_stall_seconds
    ):
        add_alert(
            alerts,
            alert_id="warmstart:training_stalled",
            severity="critical" if age >= 2 * config.log_stall_seconds else "warning",
            summary="QZ 显示 running，但 train.log 过久没有 local step 活动",
            evidence={"latest_local_step": latest, "activity_age_seconds": age},
            recommendation="人工查看远端日志/GPU；watcher 不会 restart。",
        )
    if checkpoints["save_errors"]:
        add_alert(
            alerts,
            alert_id="warmstart:checkpoint_save_error",
            severity="critical",
            summary="发现 checkpoint_save_error.txt",
            evidence=checkpoints["save_errors"],
            recommendation="保留错误文件并检查 I/O；禁止使用该 checkpoint。",
        )
    if checkpoints["incomplete"]:
        add_alert(
            alerts,
            alert_id="warmstart:checkpoint_incomplete",
            severity="critical",
            summary="公开的 continuation checkpoint 缺少必要权重",
            evidence=checkpoints["incomplete"],
            recommendation="禁止评测这些 checkpoint，人工核验保存状态。",
        )
    if checkpoints["overdue_local_steps"]:
        add_alert(
            alerts,
            alert_id="warmstart:checkpoint_overdue",
            severity="critical",
            summary="训练越过 local 保存点且 checkpoint 超时未完整落盘",
            evidence={
                "latest_local_step": latest,
                "overdue_local_steps": checkpoints["overdue_local_steps"],
                "overdue_effective_steps": checkpoints["overdue_effective_steps"],
            },
            recommendation="暂停依赖该 step 的评测并人工检查 I/O；watcher 不会 stop。",
        )
    stale = [item for item in checkpoints["hidden_tmp"] if item["severity"] != "active"]
    if stale:
        add_alert(
            alerts,
            alert_id="warmstart:stale_hidden_checkpoint_tmp",
            severity=("critical" if any(item["severity"] == "critical" for item in stale) else "warning"),
            summary="隐藏 checkpoint 临时目录存在过久",
            evidence=stale,
            recommendation="人工检查 mtime/大小是否仍增长，不要删除活跃 tmp。",
        )

    if qz_status in TERMINAL_QZ_STATUSES and latest < LOCAL_TARGET_STEP:
        add_alert(
            alerts,
            alert_id="warmstart:terminal_before_local_20000",
            severity="critical",
            summary=f"QZ 任务在 local 20k / effective 30k 前进入终态 {qz_status}",
            evidence={
                "qz_status": qz_status,
                "latest_local_step": latest,
                "latest_effective_step": SOURCE_EFFECTIVE_STEP + latest,
            },
            recommendation="封存最后完整 checkpoint 并人工评估；watcher 不会自动 stop/restart。",
        )
    elif qz_status == "job_failed":
        add_alert(
            alerts,
            alert_id="warmstart:failed_at_or_after_target",
            severity="critical",
            summary="QZ 到达 local 20k 附近后仍以 failed 结束",
            evidence={"latest_local_step": latest},
            recommendation="核验 local20k/final 完整性后再决定科学可用性。",
        )
    elif qz_status == "job_stopped" and latest >= LOCAL_TARGET_STEP:
        add_alert(
            alerts,
            alert_id="warmstart:stopped_at_or_after_target",
            severity="warning",
            summary="QZ 到达目标后以 stopped 而非 succeeded 结束",
            evidence={"latest_local_step": latest},
            recommendation="核验 local20k/final 权重，不要把 stopped 等同成功。",
        )
    if qz_status == "job_succeeded":
        step_ok = LOCAL_TARGET_STEP in checkpoints["complete_local_steps"]
        final_ok = bool(checkpoints["final_complete"])
        if latest < LOCAL_TARGET_STEP or not step_ok or not final_ok:
            add_alert(
                alerts,
                alert_id="warmstart:success_artifact_mismatch",
                severity="critical",
                summary="QZ succeeded，但 local20k/final 证据不完整",
                evidence={
                    "latest_local_step": latest,
                    "local_20000_complete": step_ok,
                    "final_complete": final_ok,
                },
                recommendation="不要宣布 effective30k 完成，先人工核验共享落盘。",
            )
    return (
        {
            "job_id": JOB_ID,
            "job_name": JOB_NAME,
            "run_dir": str(config.run_dir),
            "latest_local_step": latest,
            "latest_effective_step": SOURCE_EFFECTIVE_STEP + latest,
            "target_local_step": LOCAL_TARGET_STEP,
            "target_effective_step": EFFECTIVE_TARGET_STEP,
            "qz": dict(qz),
            "train_log": log,
            "checkpoints": checkpoints,
        },
        alerts,
    )


def overall_status(alerts: Sequence[Mapping[str, Any]], run: Mapping[str, Any]) -> str:
    if any(item.get("severity") == "critical" for item in alerts):
        return "critical"
    if alerts:
        return "warning"
    if (
        run.get("qz", {}).get("status") == "job_succeeded"
        and run.get("latest_local_step", 0) >= LOCAL_TARGET_STEP
    ):
        return "complete"
    return "healthy"


def render_status_markdown(status: Mapping[str, Any]) -> str:
    run = status["run"]
    log = run["train_log"]
    checkpoints = run["checkpoints"]
    lines = [
        "# Batch-44 r3 warm-start training health",
        "",
        f"- Observed: `{status['observed_at_utc']}`",
        f"- Overall: **{status['status']}**",
        f"- Job: `{JOB_ID}`",
        f"- QZ: `{run['qz'].get('status', 'query_error')}` on `{ALLOWED_COMPUTE_GROUP_NAME}`, 1×8 H200",
        f"- Progress: local `{run['latest_local_step']}/{LOCAL_TARGET_STEP}` -> effective `{run['latest_effective_step']}/{EFFECTIVE_TARGET_STEP}`",
        f"- ETA: `{log.get('eta_utc') or '—'}`; sec/step: `{log.get('seconds_per_step') or '—'}`",
        f"- Complete local checkpoints: `{checkpoints['complete_local_steps'] or 'none'}`",
        f"- Complete effective checkpoints: `{checkpoints['complete_effective_steps'] or 'none'}`",
        f"- Warm-start contract SHA256: `{status['contract']['warm_start_contract_sha256']}`",
        "",
    ]
    if status["alerts"]:
        lines.extend(["## Current alerts", ""])
        for item in status["alerts"]:
            lines.append(f"- **{item['severity']}** `{item['id']}`: {item['summary']}")
    else:
        lines.append("No current alert.")
    lines.append("")
    return "\n".join(lines)


def render_recommendation(status: Mapping[str, Any]) -> str:
    lines = [
        "# STOP OR RECOVERY RECOMMENDATION (manual decision only)",
        "",
        f"Observed: `{status['observed_at_utc']}`",
        "",
        "This watcher has **not** stopped, restarted, logged in to, or submitted any QZ job.",
        "",
    ]
    for item in status["alerts"]:
        lines.extend(
            [
                f"## {item['severity'].upper()} {item['id']}",
                "",
                item["summary"],
                "",
                f"Recommendation: {item['recommendation']}",
                "",
            ]
        )
    lines.extend(
        [
            "Reminder: this is a weights-only warm start. Local step N maps to effective step 10000+N; optimizer/scheduler/RNG continuity is unavailable.",
            "",
        ]
    )
    return "\n".join(lines)


def scan_once(config: Config) -> dict[str, Any]:
    contract = validate_static_contract(config)
    config.state_root.mkdir(parents=True, exist_ok=True)
    now = utc_now()
    previous = read_previous_status(config.state_root / "STATUS.json")
    previous_run = previous.get("run") if isinstance(previous.get("run"), dict) else {}
    qz_lock = config.state_root / "qz_status_query.lock"
    with qz_lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        qz, alerts = query_qz_status(config)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    run_payload, run_alerts = analyze(config, qz, previous_run, now)
    alerts.extend(run_alerts)
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda item: (severity_order.get(str(item.get("severity")), 9), str(item["id"])))
    status = {
        "schema_version": "moss_codecvc.batch44_r3_warmstart_training_health.v1",
        "observed_at_utc": format_utc(now),
        "status": overall_status(alerts, run_payload),
        "poll_seconds": config.poll_seconds,
        "contract": contract,
        "step_mapping": {
            "source_effective_step": SOURCE_EFFECTIVE_STEP,
            "local_target_step": LOCAL_TARGET_STEP,
            "effective_target_step": EFFECTIVE_TARGET_STEP,
            "formula": "effective_step = 10000 + local_step",
        },
        "read_only_contract": {
            "qzcli_subcommand": "status",
            "job_id": JOB_ID,
            "shared_qzcli_home": str(config.qzcli_home),
            "proxy_variables_removed": [
                "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"
            ],
            "automatic_login": False,
            "automatic_submit": False,
            "automatic_stop": False,
            "automatic_restart": False,
        },
        "resource_contract": {
            "workspace_id": ALLOWED_WORKSPACE_ID,
            "project_id": ALLOWED_PROJECT_ID,
            "compute_group_id": ALLOWED_COMPUTE_GROUP_ID,
            "compute_group_name": ALLOWED_COMPUTE_GROUP_NAME,
            "spec_id": ALLOWED_SPEC_ID,
            "instances": ALLOWED_INSTANCES,
            "gpus": ALLOWED_GPUS,
            "gpu_type": ALLOWED_GPU_TYPE,
        },
        "run": run_payload,
        "alerts": alerts,
    }
    atomic_write_json(config.state_root / "STATUS.json", status)
    atomic_write_text(config.state_root / "STATUS.md", render_status_markdown(status))
    append_jsonl(
        config.state_root / "history.jsonl",
        {
            "observed_at_utc": status["observed_at_utc"],
            "status": status["status"],
            "qz_status": run_payload["qz"].get("status"),
            "latest_local_step": run_payload["latest_local_step"],
            "latest_effective_step": run_payload["latest_effective_step"],
            "complete_local_steps": run_payload["checkpoints"]["complete_local_steps"],
            "alert_ids": [item["id"] for item in alerts],
        },
    )
    alert_path = config.state_root / "ALERT.json"
    recommendation_path = config.state_root / "STOP_OR_RECOVERY_RECOMMENDATION.md"
    if alerts:
        alert_payload = {
            "schema_version": "moss_codecvc.batch44_r3_warmstart_training_health_alert.v1",
            "observed_at_utc": status["observed_at_utc"],
            "status": status["status"],
            "alerts": alerts,
            "automatic_action_taken": False,
        }
        atomic_write_json(alert_path, alert_payload)
        atomic_write_text(recommendation_path, render_recommendation(status))
        append_jsonl(config.state_root / "alert_history.jsonl", alert_payload)
    else:
        for path in (alert_path, recommendation_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    return status


def write_contract_alert(config: Config, error: ContractError) -> None:
    observed = format_utc(utc_now())
    alert = {
        "id": "watcher:static_contract_error",
        "severity": "critical",
        "arm": "r3-warmstart",
        "summary": "warm-start watcher 的硬绑定 contract/job/path/resource 校验失败",
        "evidence": str(error),
        "recommendation": "不要使用该 watcher 结果；人工核验固定 job 与 contract。watcher 未修改 QZ。",
    }
    payload = {
        "schema_version": "moss_codecvc.batch44_r3_warmstart_training_health_alert.v1",
        "observed_at_utc": observed,
        "status": "critical",
        "alerts": [alert],
        "automatic_action_taken": False,
    }
    atomic_write_json(config.state_root / "ALERT.json", payload)
    atomic_write_text(
        config.state_root / "STOP_OR_RECOVERY_RECOMMENDATION.md",
        "# STOP OR RECOVERY RECOMMENDATION (manual decision only)\n\n"
        f"Observed: `{observed}`\n\n"
        f"Static contract failure: `{error}`\n\n"
        "The watcher did not call any QZ mutation command.\n",
    )
    append_jsonl(config.state_root / "alert_history.jsonl", payload)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("once", "monitor"), default="once")
    result.add_argument(
        "--poll-seconds",
        type=int,
        default=None,
        help="Production is hard-locked to 300; shorter values are test-only.",
    )
    result.add_argument(
        "--max-scans",
        type=int,
        default=0,
        help="0 means unlimited in monitor mode; once mode scans once.",
    )
    return result


def run(config: Config) -> int:
    config.state_root.mkdir(parents=True, exist_ok=True)
    monitor_lock = config.state_root / "monitor.lock"
    with monitor_lock.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError(f"another warm-start health watcher holds {monitor_lock}") from exc
        scans = 0
        while True:
            started = time.monotonic()
            try:
                status = scan_once(config)
            except ContractError as exc:
                write_contract_alert(config, exc)
                raise
            scans += 1
            run_payload = status["run"]
            print(
                f"[batch44-r3-warmstart-health] scan={scans} status={status['status']} "
                f"qz={run_payload['qz'].get('status')} "
                f"local={run_payload['latest_local_step']}/{LOCAL_TARGET_STEP} "
                f"effective={run_payload['latest_effective_step']}/{EFFECTIVE_TARGET_STEP}",
                flush=True,
            )
            if config.mode == "once":
                return 0 if status["status"] in {"healthy", "complete"} else 1
            if config.max_scans and scans >= config.max_scans:
                return 0 if status["status"] in {"healthy", "complete"} else 1
            if run_payload["qz"].get("status") in TERMINAL_QZ_STATUSES:
                return 0 if status["status"] == "complete" else 1
            elapsed = time.monotonic() - started
            time.sleep(max(0.0, config.poll_seconds - elapsed))


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        return run(build_config(args))
    except ContractError as exc:
        print(f"[batch44-r3-warmstart-health] CONTRACT ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[batch44-r3-warmstart-health] interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

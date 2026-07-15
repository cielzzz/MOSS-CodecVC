#!/usr/bin/env python3
"""Read-only health watcher for the two live Batch-44 v1 30k jobs.

The current training recipe writes inference-only checkpoints.  They are useful
for evaluation, but do not contain optimizer/scheduler/RNG state and therefore
cannot provide a strict training resume after a remote failure.  This watcher
reduces that operational risk by checking, every five minutes:

* the two hard-bound QZ jobs, sequentially and with the qzcli-local HOME;
* train.log progress, non-finite values, OOM/NCCL/fatal signatures and ETA;
* checkpoint_save_error.txt, overdue checkpoints and stale hidden save dirs;
* available bytes and inodes on the QB filesystem; and
* failed/stopped/succeeded terminal states before step 30,000.

It is observation-only.  The only qzcli subprocess is ``status JOB --json``.
It never logs in, submits, stops, restarts, or otherwise mutates a QZ job.  An
anomaly only writes ``ALERT.json`` and
``STOP_OR_RECOVERY_RECOMMENDATION.md`` under the dedicated watcher state root.

Safe one-shot audit::

    python scripts/004116_watch_batch44_training_health.py --mode once

Five-minute watcher (prepare only unless explicitly started by an operator)::

    nohup python scripts/004116_watch_batch44_training_health.py \
      --mode monitor > trainset/qz_jobs/batch44_training_health_20260713/watcher.log 2>&1 &
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
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

PAIR_RECORD_RELATIVE = Path(
    "trainset/qz_jobs/"
    "ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713"
)
DEFAULT_STATE_RELATIVE = Path("trainset/qz_jobs/batch44_training_health_20260713")

ALLOWED_WORKSPACE_ID = "ws-8207e9e2-e733-4eec-a475-cfa1c36480ba"
ALLOWED_PROJECT_ID = "project-c67c548f-f02c-453b-ba5b-8745db6886e7"
ALLOWED_COMPUTE_GROUP_ID = "lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"
ALLOWED_COMPUTE_GROUP_NAME = "MTTS-3-2-0715"
ALLOWED_SPEC_ID = "67b10bc6-78b0-41a3-aaf4-358eeeb99009"
ALLOWED_GPU_TYPE = "NVIDIA_H200_SXM_141G"
ALLOWED_INSTANCES = 1
ALLOWED_GPUS = 8
TARGET_STEP = 30_000
SAVE_STEPS = 2_000
POLL_SECONDS = 300

# The QB volume is very large, so absolute free-space thresholds are more
# meaningful than the rounded percentage shown by ``df``.
SPACE_WARN_BYTES = 2 * 1024**4  # 2 TiB
SPACE_CRITICAL_BYTES = 512 * 1024**3  # 512 GiB
INODE_WARN_COUNT = 10_000_000
INODE_CRITICAL_COUNT = 1_000_000
LOG_STALL_SECONDS = 15 * 60
CHECKPOINT_GRACE_SECONDS = 15 * 60
TMP_STALE_WARN_SECONDS = 30 * 60
TMP_STALE_CRITICAL_SECONDS = 90 * 60
QZ_TIMEOUT_SECONDS = 90

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
    ("traceback", re.compile(r"traceback \(most recent call last\)", re.IGNORECASE)),
    ("oom", re.compile(r"(?:cuda\s+)?out of memory|cudnn_status_alloc_failed", re.IGNORECASE)),
    (
        "nccl",
        re.compile(r"nccl[^\n]*(?:error|failed|failure|timeout|abort)", re.IGNORECASE),
    ),
    ("runtime_error", re.compile(r"\bruntimeerror\b", re.IGNORECASE)),
    ("fatal", re.compile(r"\bfatal\b", re.IGNORECASE)),
    (
        "process_killed",
        re.compile(r"\b(?:sigkill|segmentation fault|killed process|exitcode\s*[:=]\s*-9)\b", re.IGNORECASE),
    ),
)
AUTH_ERROR_RE = re.compile(
    r"(?:\b401\b[^\n]*(?:unauthori[sz]ed|auth)|"
    r"unauthori[sz]ed|authentication required|please\s+(?:re-)?login|"
    r"not logged in|cookie[^\n]*(?:expired|invalid)|"
    r"未登录|登录[^\n]*(?:过期|失效)|认证[^\n]*(?:过期|失败))",
    re.IGNORECASE,
)


class ContractError(RuntimeError):
    """A hard-bound watcher or job-identity contract was violated."""


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    job_name: str
    job_id: str
    run_name: str
    repeat: int


ARMS: tuple[ArmSpec, ...] = (
    ArmSpec(
        arm="r3",
        job_name="ver2_9_5_final_r3_v1_30k",
        job_id="job-2b91d332-d500-4279-84f9-0a6a81a376aa",
        run_name="ver2_9_5_final_r3_v1_30k",
        repeat=3,
    ),
    ArmSpec(
        arm="r5",
        job_name="ver2_9_5_final_r5_v1_30k",
        job_id="job-b8eb2f1f-a3eb-483b-a289-b4cce281525c",
        run_name="ver2_9_5_final_r5_v1_30k",
        repeat=5,
    ),
)


@dataclass(frozen=True)
class Config:
    project_root: Path
    pair_record_root: Path
    state_root: Path
    qzcli: Path
    qzcli_home: Path
    mode: str
    poll_seconds: int
    max_scans: int
    test_mode: bool
    qz_timeout_seconds: int
    log_stall_seconds: int
    checkpoint_grace_seconds: int
    tmp_stale_warn_seconds: int
    tmp_stale_critical_seconds: int
    space_warn_bytes: int
    space_critical_bytes: int
    inode_warn_count: int
    inode_critical_count: int

    def run_dir(self, arm: ArmSpec) -> Path:
        return self.project_root / "outputs/lora_runs" / arm.run_name


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


def lexical_absolute(value: str | os.PathLike[str]) -> Path:
    """Return an absolute normalized path without resolving mounted symlinks.

    The project's ``trainset`` is a symlink, while the immutable QZ ledgers and
    remote command intentionally record the user-facing ``.../trainset/...``
    path.  ``Path.resolve()`` would silently rewrite that identity.
    """

    return Path(os.path.abspath(os.path.expanduser(os.fspath(value))))


def build_config(args: argparse.Namespace) -> Config:
    test_mode = os.environ.get("BATCH44_HEALTH_TEST_MODE", "0") == "1"
    if os.environ.get("BATCH44_HEALTH_TEST_MODE", "0") not in {"0", "1"}:
        raise ContractError("BATCH44_HEALTH_TEST_MODE must be 0 or 1")
    canonical = CANONICAL_PROJECT_ROOT.resolve()
    project_root = Path(os.environ.get("PROJECT_ROOT", str(canonical))).resolve()
    if project_root != canonical and not test_mode:
        raise ContractError("production PROJECT_ROOT is hard-locked")
    pair_record = lexical_absolute(
        os.environ.get("PAIR_RECORD_ROOT", str(project_root / PAIR_RECORD_RELATIVE))
    )
    state_root = lexical_absolute(
        os.environ.get("STATE_ROOT", str(project_root / DEFAULT_STATE_RELATIVE))
    )
    qzcli = Path(os.environ.get("QZCLI", str(CANONICAL_QZCLI))).resolve()
    qzcli_home = Path(
        os.environ.get("QZCLI_HOME", str(CANONICAL_QZCLI_HOME))
    ).resolve()
    if not test_mode:
        if pair_record != lexical_absolute(canonical / PAIR_RECORD_RELATIVE):
            raise ContractError("production PAIR_RECORD_ROOT is hard-locked")
        if state_root != lexical_absolute(canonical / DEFAULT_STATE_RELATIVE):
            raise ContractError("production STATE_ROOT is hard-locked")
        if qzcli != CANONICAL_QZCLI.resolve() or qzcli_home != CANONICAL_QZCLI_HOME.resolve():
            raise ContractError("production qzcli wrapper/HOME are hard-locked")
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
        pair_record_root=pair_record,
        state_root=state_root,
        qzcli=qzcli,
        qzcli_home=qzcli_home,
        mode=args.mode,
        poll_seconds=poll_seconds,
        max_scans=args.max_scans,
        test_mode=test_mode,
        qz_timeout_seconds=env_int(
            "BATCH44_HEALTH_QZ_TIMEOUT_SECONDS", QZ_TIMEOUT_SECONDS, test_mode=test_mode
        ),
        log_stall_seconds=env_int(
            "BATCH44_HEALTH_LOG_STALL_SECONDS", LOG_STALL_SECONDS, test_mode=test_mode
        ),
        checkpoint_grace_seconds=env_int(
            "BATCH44_HEALTH_CHECKPOINT_GRACE_SECONDS",
            CHECKPOINT_GRACE_SECONDS,
            test_mode=test_mode,
        ),
        tmp_stale_warn_seconds=env_int(
            "BATCH44_HEALTH_TMP_WARN_SECONDS", TMP_STALE_WARN_SECONDS, test_mode=test_mode
        ),
        tmp_stale_critical_seconds=env_int(
            "BATCH44_HEALTH_TMP_CRITICAL_SECONDS",
            TMP_STALE_CRITICAL_SECONDS,
            test_mode=test_mode,
        ),
        space_warn_bytes=env_int(
            "BATCH44_HEALTH_SPACE_WARN_BYTES", SPACE_WARN_BYTES, test_mode=test_mode
        ),
        space_critical_bytes=env_int(
            "BATCH44_HEALTH_SPACE_CRITICAL_BYTES",
            SPACE_CRITICAL_BYTES,
            test_mode=test_mode,
        ),
        inode_warn_count=env_int(
            "BATCH44_HEALTH_INODE_WARN_COUNT", INODE_WARN_COUNT, test_mode=test_mode
        ),
        inode_critical_count=env_int(
            "BATCH44_HEALTH_INODE_CRITICAL_COUNT",
            INODE_CRITICAL_COUNT,
            test_mode=test_mode,
        ),
    )


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"JSON object required: {path}")
    return payload


def validate_qz_resource_payload(
    payload: Mapping[str, Any], arm: ArmSpec, pair_record_root: Path
) -> None:
    expected = {
        "name": arm.job_name,
        "workspace_id": ALLOWED_WORKSPACE_ID,
        "project_id": ALLOWED_PROJECT_ID,
        "logic_compute_group_id": ALLOWED_COMPUTE_GROUP_ID,
    }
    bad = {key: payload.get(key) for key, value in expected.items() if payload.get(key) != value}
    if bad:
        raise ContractError(f"{arm.arm} QZ identity drift: {bad}")
    framework = payload.get("framework_config")
    if not isinstance(framework, list) or len(framework) != 1 or not isinstance(framework[0], dict):
        raise ContractError(f"{arm.arm} must have exactly one QZ framework instance group")
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
        raise ContractError(f"{arm.arm} is not spec={ALLOWED_SPEC_ID}, one 8xH200 node")
    runner = pair_record_root / arm.arm / "run_train_entrypoint.sh"
    if str(payload.get("command") or "").strip() != f"sh {runner}":
        raise ContractError(f"{arm.arm} QZ command is not bound to {runner}")


def validate_static_contract(config: Config) -> None:
    if not config.qzcli.is_file() or not os.access(config.qzcli, os.X_OK):
        raise ContractError(f"qzcli-local wrapper is not executable: {config.qzcli}")
    if not config.qzcli_home.is_dir():
        raise ContractError(f"shared qzcli-local HOME is missing: {config.qzcli_home}")
    if config.tmp_stale_critical_seconds < config.tmp_stale_warn_seconds:
        raise ContractError("hidden-tmp critical threshold must be >= warning threshold")
    if config.space_warn_bytes < config.space_critical_bytes:
        raise ContractError("space warning threshold must be >= critical threshold")
    if config.inode_warn_count < config.inode_critical_count:
        raise ContractError("inode warning threshold must be >= critical threshold")

    ledger = config.pair_record_root / "submitted_pair.tsv"
    try:
        lines = ledger.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractError(f"missing Batch-44 submitted-pair ledger: {ledger}") from exc
    expected_header = "arm\tjob_name\tjob_id\tcompute_group\trunner\tout_dir"
    if not lines or lines[0] != expected_header or len(lines) != 3:
        raise ContractError("Batch-44 submitted_pair.tsv schema/cardinality drift")
    rows: dict[str, list[str]] = {}
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != 6 or fields[0] in rows:
            raise ContractError("Batch-44 submitted_pair.tsv row drift")
        rows[fields[0]] = fields
    for arm in ARMS:
        run_dir = lexical_absolute(config.run_dir(arm))
        runner = lexical_absolute(
            config.pair_record_root / arm.arm / "run_train_entrypoint.sh"
        )
        wanted = [
            arm.arm,
            arm.job_name,
            arm.job_id,
            ALLOWED_COMPUTE_GROUP_ID,
            str(runner),
            str(run_dir),
        ]
        if rows.get(arm.arm) != wanted:
            raise ContractError(f"{arm.arm} submitted-pair identity/path/resource drift")
        if not JOB_RE.fullmatch(arm.job_id):
            raise ContractError(f"invalid hard-bound QZ job id: {arm.job_id}")
        if not runner.is_file():
            raise ContractError(f"missing frozen {arm.arm} training runner: {runner}")
        runner_text = runner.read_text(encoding="utf-8")
        required_runner_lines = (
            f'OUT_DIR="{run_dir}"',
            f"text.train.jsonl::repeat={arm.repeat}",
            'EVAL_STEPS="2000"',
            '--max-train-steps "30000"',
            '--save-steps "2000"',
        )
        missing_runner_lines = [
            value for value in required_runner_lines if value not in runner_text
        ]
        if missing_runner_lines:
            raise ContractError(
                f"{arm.arm} frozen runner lost 30k/repeat/save/output binding: "
                f"{missing_runner_lines}"
            )
        payload_path = config.pair_record_root / arm.arm / "qz_payload.json"
        payload = load_json(payload_path)
        validate_qz_resource_payload(payload, arm, config.pair_record_root)


def add_alert(
    alerts: list[dict[str, Any]],
    *,
    alert_id: str,
    severity: str,
    arm: str | None,
    summary: str,
    evidence: Any,
    recommendation: str,
) -> None:
    alerts.append(
        {
            "id": alert_id,
            "severity": severity,
            "arm": arm,
            "summary": summary,
            "evidence": evidence,
            "recommendation": recommendation,
        }
    )


def extract_qz_json(text: str, expected_job: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    matches: dict[str, dict[str, Any]] = {}
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("job_id") == expected_job:
            matches[json.dumps(payload, ensure_ascii=False, sort_keys=True)] = payload
    if len(matches) != 1:
        raise ContractError(
            f"qzcli status output must contain one API JSON for {expected_job}; got {len(matches)}"
        )
    return next(iter(matches.values()))


def audit_live_qz_payload(payload: Mapping[str, Any], arm: ArmSpec, config: Config) -> str:
    expected = {
        "job_id": arm.job_id,
        "name": arm.job_name,
        "workspace_id": ALLOWED_WORKSPACE_ID,
        "project_id": ALLOWED_PROJECT_ID,
        "logic_compute_group_id": ALLOWED_COMPUTE_GROUP_ID,
        "logic_compute_group_name": ALLOWED_COMPUTE_GROUP_NAME,
    }
    bad = {key: payload.get(key) for key, value in expected.items() if payload.get(key) != value}
    if bad:
        raise ContractError(f"live QZ identity/workspace/compute-group drift: {bad}")
    framework = payload.get("framework_config")
    if not isinstance(framework, list) or len(framework) != 1 or not isinstance(framework[0], dict):
        raise ContractError("live QZ payload must have exactly one framework_config entry")
    item = framework[0]
    resource = item.get("instance_spec_price_info") or item.get("resource_spec_price") or {}
    gpu = resource.get("gpu_info") or {}
    gpu_type = gpu.get("gpu_type") or resource.get("gpu_type")
    if (
        item.get("instance_count") != ALLOWED_INSTANCES
        or item.get("gpu_count") != ALLOWED_GPUS
        or resource.get("quota_id") != ALLOWED_SPEC_ID
        or gpu_type != ALLOWED_GPU_TYPE
    ):
        raise ContractError("live QZ job is not registered spec / one 8xH200 instance")
    runner = config.pair_record_root / arm.arm / "run_train_entrypoint.sh"
    if str(payload.get("command") or "").strip() != f"sh {runner}":
        raise ContractError("live QZ command is not bound to frozen Batch-44 runner")
    status = str(payload.get("status") or "").strip().lower()
    if status not in ACCEPTED_QZ_STATUSES:
        raise ContractError(f"unrecognized QZ status: {status!r}")
    return status


def query_qz_status(config: Config, arm: ArmSpec) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run exactly one status-only qzcli call; never retry or login."""

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
    command = [str(config.qzcli), "status", arm.job_id, "--json"]
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            env=environment,
            timeout=config.qz_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raw = f"qzcli status timed out after {config.qz_timeout_seconds}s: {exc}"
        result_returncode = None
    except OSError as exc:
        raw = f"qzcli status could not execute: {exc}"
        result_returncode = None
    else:
        raw = result.stdout + ("\n" + result.stderr if result.stderr else "")
        result_returncode = result.returncode
    atomic_write_text(config.state_root / f"qz_status_{arm.arm}_latest.txt", raw[-200_000:])
    observed: dict[str, Any] = {
        "queried": True,
        "query_command": ["qzcli-local", "status", arm.job_id, "--json"],
        "query_returncode": result_returncode,
        "status": "query_error",
        "authentication_error": False,
    }
    if result_returncode != 0:
        is_auth = bool(AUTH_ERROR_RE.search(raw))
        observed["authentication_error"] = is_auth
        observed["query_error_excerpt"] = raw[-2000:]
        if is_auth:
            add_alert(
                alerts,
                alert_id=f"{arm.arm}:qz_authentication_required",
                severity="warning",
                arm=arm.arm,
                summary="qzcli-local 明确返回认证失效；watcher 未尝试自动登录",
                evidence={"returncode": result_returncode, "excerpt": raw[-1000:]},
                recommendation=(
                    "先确认没有其他 qzcli login 正在使用共享 HOME，再由人工串行执行一次登录；"
                    "登录前后继续以本地 train.log/checkpoint 证据判断训练，不要据此自动 stop/restart。"
                ),
            )
        else:
            add_alert(
                alerts,
                alert_id=f"{arm.arm}:qz_query_unavailable",
                severity="warning",
                arm=arm.arm,
                summary="QZ 状态查询不可用；没有足够证据把它归因为认证问题",
                evidence={"returncode": result_returncode, "excerpt": raw[-1000:]},
                recommendation=(
                    "人工检查 qzcli/network；watcher 不重试登录，也不把查询失败解释成训练失败。"
                ),
            )
        return observed, alerts
    try:
        payload = extract_qz_json(raw, arm.job_id)
        status = audit_live_qz_payload(payload, arm, config)
    except ContractError as exc:
        observed["query_error_excerpt"] = str(exc)
        add_alert(
            alerts,
            alert_id=f"{arm.arm}:qz_resource_or_identity_drift",
            severity="critical",
            arm=arm.arm,
            summary="QZ 实时任务身份或 MTTS/1x8 H200 资源合同漂移",
            evidence=str(exc),
            recommendation="暂停后续科学结论；人工核验平台任务详情。watcher 不会修改任务。",
        )
        return observed, alerts
    observed.update(
        {
            "status": status,
            "job_id": payload.get("job_id"),
            "job_name": payload.get("name"),
            "workspace_id": payload.get("workspace_id"),
            "project_id": payload.get("project_id"),
            "compute_group_id": payload.get("logic_compute_group_id"),
            "compute_group_name": payload.get("logic_compute_group_name"),
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
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def parse_train_log(path: Path, now: dt.datetime) -> dict[str, Any]:
    if not path.is_file():
        return {
            "exists": False,
            "latest_step": 0,
            "target_step": TARGET_STEP,
            "progress_records": 0,
            "fatal_matches": [],
            "nonfinite_matches": [],
            "finished_step": None,
        }
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {
            "exists": True,
            "read_error": str(exc),
            "latest_step": 0,
            "target_step": TARGET_STEP,
            "progress_records": 0,
            "fatal_matches": [],
            "nonfinite_matches": [],
            "finished_step": None,
        }
    records: list[tuple[dt.datetime, int, int, str]] = []
    fatal_matches: list[dict[str, Any]] = []
    nonfinite_matches: list[dict[str, Any]] = []
    finished_step: int | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        progress = PROGRESS_RE.match(line)
        if progress:
            timestamp = parse_utc(progress.group("timestamp"))
            records.append(
                (
                    timestamp,
                    int(progress.group("step")),
                    int(progress.group("target")),
                    line,
                )
            )
            if NONFINITE_RE.search(progress.group("body")):
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
    seconds_per_step: float | None = None
    intervals: list[float] = []
    rate_window = records[-61:]
    for previous, current in zip(rate_window, rate_window[1:]):
        delta_step = current[1] - previous[1]
        delta_time = (current[0] - previous[0]).total_seconds()
        if delta_step > 0 and delta_time > 0:
            intervals.append(delta_time / delta_step)
    if intervals:
        seconds_per_step = statistics.median(intervals)
    latest_time = latest[0] if latest else None
    latest_step = latest[1] if latest else 0
    target_step = latest[2] if latest else TARGET_STEP
    eta = None
    if latest_time and seconds_per_step and latest_step < target_step:
        eta = latest_time + dt.timedelta(seconds=(target_step - latest_step) * seconds_per_step)
    stat = path.stat()
    mtime = dt.datetime.fromtimestamp(stat.st_mtime, tz=dt.timezone.utc)
    activity_time = max(item for item in (latest_time, mtime) if item is not None)
    return {
        "exists": True,
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_utc": format_utc(mtime),
        "latest_step": latest_step,
        "target_step": target_step,
        "progress_percent": round(100.0 * latest_step / max(target_step, 1), 4),
        "progress_records": len(records),
        "latest_progress_utc": format_utc(latest_time),
        "activity_age_seconds": max(0.0, (now - activity_time).total_seconds()),
        "seconds_per_step": seconds_per_step,
        "eta_utc": format_utc(eta),
        "finished_step": finished_step,
        "fatal_matches": fatal_matches[-20:],
        "nonfinite_matches": nonfinite_matches[-20:],
    }


def checkpoint_inventory(
    config: Config,
    run_dir: Path,
    latest_step: int,
    log_activity_age: float | None,
    now: dt.datetime,
) -> dict[str, Any]:
    complete: list[int] = []
    incomplete: list[dict[str, Any]] = []
    save_errors: list[dict[str, Any]] = []
    strict_resume_checkpoints: list[int | str] = []
    for step in range(SAVE_STEPS, TARGET_STEP + 1, SAVE_STEPS):
        directory = run_dir / f"step-{step}"
        if not directory.is_dir():
            continue
        files = {item.name for item in directory.iterdir() if item.is_file()}
        missing = sorted(EXPECTED_MODEL_FILES - files)
        if missing:
            incomplete.append({"step": step, "path": str(directory), "missing": missing})
        else:
            complete.append(step)
        if files & STRICT_RESUME_STATE_FILES:
            strict_resume_checkpoints.append(step)
    final_dir = run_dir / "final"
    if final_dir.is_dir():
        final_files = {item.name for item in final_dir.iterdir() if item.is_file()}
        if final_files & STRICT_RESUME_STATE_FILES:
            strict_resume_checkpoints.append("final")
    for error_path in sorted(run_dir.glob("step-*/checkpoint_save_error.txt")) + sorted(
        run_dir.glob("final/checkpoint_save_error.txt")
    ):
        try:
            excerpt = error_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        except OSError as exc:
            excerpt = f"cannot read: {exc}"
        save_errors.append({"path": str(error_path), "excerpt": excerpt})
    hidden_tmp: list[dict[str, Any]] = []
    if run_dir.is_dir():
        for path in sorted(run_dir.glob(".step-*.tmp-*")) + sorted(
            run_dir.glob(".final.tmp-*")
        ):
            try:
                newest_mtime = path.stat().st_mtime
                if path.is_dir():
                    for child in path.rglob("*"):
                        try:
                            newest_mtime = max(newest_mtime, child.stat().st_mtime)
                        except OSError:
                            continue
                age = max(0.0, now.timestamp() - newest_mtime)
            except OSError:
                continue
            hidden_tmp.append(
                {
                    "path": str(path),
                    "age_seconds": age,
                    "severity": (
                        "critical"
                        if age >= config.tmp_stale_critical_seconds
                        else "warning"
                        if age >= config.tmp_stale_warn_seconds
                        else "active"
                    ),
                }
            )
    expected_through = (min(latest_step, TARGET_STEP) // SAVE_STEPS) * SAVE_STEPS
    overdue: list[int] = []
    for step in range(SAVE_STEPS, expected_through + 1, SAVE_STEPS):
        if step in complete:
            continue
        # Once training advances beyond a save boundary, that save must have
        # completed because the loop is synchronous.  At the exact boundary,
        # allow the configured grace period for the atomic save.
        if latest_step > step or (
            latest_step == step
            and log_activity_age is not None
            and log_activity_age >= config.checkpoint_grace_seconds
        ):
            overdue.append(step)
    return {
        "save_interval": SAVE_STEPS,
        "expected_model_files": sorted(EXPECTED_MODEL_FILES),
        "complete_steps": complete,
        "incomplete": incomplete,
        "overdue_steps": overdue,
        "save_errors": save_errors,
        "hidden_tmp": hidden_tmp,
        "final_exists": final_dir.is_dir(),
        "strict_resume_available": bool(strict_resume_checkpoints),
        "strict_resume_state_locations": strict_resume_checkpoints,
        "resume_contract": (
            "inference_only: optimizer/scheduler/RNG state is not saved; "
            "strict training resume is unavailable"
        ),
    }


def analyze_storage(config: Config) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    alerts: list[dict[str, Any]] = []
    stats = os.statvfs(config.project_root)
    available_bytes = stats.f_bavail * stats.f_frsize
    total_bytes = stats.f_blocks * stats.f_frsize
    inode_available = stats.f_favail
    inode_total = stats.f_files
    payload = {
        "path": str(config.project_root),
        "available_bytes": available_bytes,
        "available_gib": available_bytes / 1024**3,
        "total_bytes": total_bytes,
        "inode_available": inode_available,
        "inode_total": inode_total,
        "thresholds": {
            "space_warning_bytes": config.space_warn_bytes,
            "space_critical_bytes": config.space_critical_bytes,
            "inode_warning": config.inode_warn_count,
            "inode_critical": config.inode_critical_count,
        },
    }
    if available_bytes < config.space_critical_bytes:
        severity = "critical"
    elif available_bytes < config.space_warn_bytes:
        severity = "warning"
    else:
        severity = None
    if severity:
        add_alert(
            alerts,
            alert_id="storage:available_bytes_low",
            severity=severity,
            arm=None,
            summary="QB 可用空间低于 Batch-44 watcher 阈值",
            evidence={"available_bytes": available_bytes, "available_gib": payload["available_gib"]},
            recommendation=(
                "人工清理与本实验无关且已确认可删除的数据，优先保护现有 step checkpoint；"
                "watcher 不会删除文件或停止任务。"
            ),
        )
    if inode_available < config.inode_critical_count:
        inode_severity = "critical"
    elif inode_available < config.inode_warn_count:
        inode_severity = "warning"
    else:
        inode_severity = None
    if inode_severity:
        add_alert(
            alerts,
            alert_id="storage:available_inodes_low",
            severity=inode_severity,
            arm=None,
            summary="QB 可用 inode 低于 Batch-44 watcher 阈值",
            evidence={"inode_available": inode_available, "inode_total": inode_total},
            recommendation="人工审计小文件占用；watcher 不会删除任何文件。",
        )
    return payload, alerts


def analyze_arm(
    config: Config,
    arm: ArmSpec,
    qz: Mapping[str, Any],
    previous: Mapping[str, Any],
    now: dt.datetime,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    alerts: list[dict[str, Any]] = []
    run_dir = config.run_dir(arm)
    log = parse_train_log(run_dir / "train.log", now)
    latest_step = int(log.get("latest_step") or 0)
    previous_step = int(previous.get("latest_step") or 0)
    step_delta = latest_step - previous_step if previous else None
    log["previous_observed_step"] = previous_step if previous else None
    log["step_delta_since_previous_scan"] = step_delta
    log["progressing_since_previous_scan"] = step_delta is not None and step_delta > 0
    checkpoint = checkpoint_inventory(
        config,
        run_dir,
        latest_step,
        float(log["activity_age_seconds"]) if "activity_age_seconds" in log else None,
        now,
    )
    if not log.get("exists"):
        add_alert(
            alerts,
            alert_id=f"{arm.arm}:train_log_missing",
            severity="critical" if qz.get("status") == "job_running" else "warning",
            arm=arm.arm,
            summary="训练 train.log 缺失",
            evidence=str(run_dir / "train.log"),
            recommendation="人工核验共享输出挂载与远端任务日志；不要盲目 restart。",
        )
    if log.get("read_error"):
        add_alert(
            alerts,
            alert_id=f"{arm.arm}:train_log_unreadable",
            severity="warning",
            arm=arm.arm,
            summary="训练 train.log 暂时不可读",
            evidence=log["read_error"],
            recommendation="检查 QB I/O；watcher 下轮继续只读观察。",
        )
    if log.get("fatal_matches"):
        add_alert(
            alerts,
            alert_id=f"{arm.arm}:fatal_log_signature",
            severity="critical",
            arm=arm.arm,
            summary="train.log 出现 OOM/NCCL/RuntimeError/fatal 类错误",
            evidence=log["fatal_matches"],
            recommendation="保留现场并人工判断 stop/recovery；当前 checkpoint 不支持严格训练 resume。",
        )
    if log.get("nonfinite_matches"):
        add_alert(
            alerts,
            alert_id=f"{arm.arm}:nonfinite_training_metric",
            severity="critical",
            arm=arm.arm,
            summary="train.log 训练指标出现 NaN/Inf",
            evidence=log["nonfinite_matches"],
            recommendation="人工检查数值发散并考虑停止；watcher 不会调用 stop。",
        )
    if step_delta is not None and step_delta < 0:
        add_alert(
            alerts,
            alert_id=f"{arm.arm}:step_regression",
            severity="critical",
            arm=arm.arm,
            summary="train.log 最新 step 相对上次扫描回退",
            evidence={"previous": previous_step, "current": latest_step},
            recommendation="可能发生日志截断或非严格重启；立即人工核验任务/输出身份。",
        )
    qz_status = str(qz.get("status") or "query_error")
    activity_age = log.get("activity_age_seconds")
    if (
        qz_status == "job_running"
        and latest_step < TARGET_STEP
        and isinstance(activity_age, (int, float))
        and activity_age >= config.log_stall_seconds
    ):
        add_alert(
            alerts,
            alert_id=f"{arm.arm}:training_stalled",
            severity="critical" if activity_age >= 2 * config.log_stall_seconds else "warning",
            arm=arm.arm,
            summary="QZ 显示 running，但 train.log 过久没有 step 活动",
            evidence={"latest_step": latest_step, "activity_age_seconds": activity_age},
            recommendation="人工查看远端日志/GPU 与 checkpoint 保存状态；不要自动 restart。",
        )
    if checkpoint["save_errors"]:
        add_alert(
            alerts,
            alert_id=f"{arm.arm}:checkpoint_save_error",
            severity="critical",
            arm=arm.arm,
            summary="发现 checkpoint_save_error.txt",
            evidence=checkpoint["save_errors"],
            recommendation="保留错误文件并核查 QB 空间/I/O；不要假定该 step checkpoint 可用。",
        )
    if checkpoint["incomplete"]:
        add_alert(
            alerts,
            alert_id=f"{arm.arm}:checkpoint_incomplete",
            severity="critical",
            arm=arm.arm,
            summary="已公开的 step checkpoint 缺少必要推理权重文件",
            evidence=checkpoint["incomplete"],
            recommendation="禁止评测这些 checkpoint；人工核验保存失败与磁盘状态。",
        )
    if checkpoint["overdue_steps"]:
        add_alert(
            alerts,
            alert_id=f"{arm.arm}:checkpoint_overdue",
            severity="critical",
            arm=arm.arm,
            summary="训练已越过保存点，但对应 checkpoint 未完整落盘",
            evidence={"latest_step": latest_step, "overdue_steps": checkpoint["overdue_steps"]},
            recommendation="暂停依赖这些 step 的评测并人工检查保存/I/O；watcher 不执行 stop。",
        )
    stale = [item for item in checkpoint["hidden_tmp"] if item["severity"] != "active"]
    if stale:
        severity = "critical" if any(item["severity"] == "critical" for item in stale) else "warning"
        add_alert(
            alerts,
            alert_id=f"{arm.arm}:stale_hidden_checkpoint_tmp",
            severity=severity,
            arm=arm.arm,
            summary="隐藏 checkpoint 临时目录存在过久，可能卡在保存阶段",
            evidence=stale,
            recommendation="人工检查文件大小/mtime 是否仍增长及 QB I/O；不要删除活跃 tmp。",
        )
    if qz_status in TERMINAL_QZ_STATUSES and latest_step < TARGET_STEP:
        add_alert(
            alerts,
            alert_id=f"{arm.arm}:terminal_before_30000",
            severity="critical",
            arm=arm.arm,
            summary=f"QZ 任务在 30k 前进入终态 {qz_status}",
            evidence={"qz_status": qz_status, "latest_step": latest_step},
            recommendation=(
                "先封存最后完整 inference checkpoint 并评估科学可用性；当前训练没有"
                " optimizer/scheduler/RNG 状态，不能宣称严格 resume。"
            ),
        )
    elif qz_status == "job_failed":
        add_alert(
            alerts,
            alert_id=f"{arm.arm}:failed_at_or_after_target",
            severity="critical",
            arm=arm.arm,
            summary="QZ 任务在到达目标附近后仍以 failed 结束",
            evidence={"qz_status": qz_status, "latest_step": latest_step},
            recommendation="核验 step-30000/final 完整性后再决定是否仅用于推理评测。",
        )
    elif qz_status == "job_stopped" and latest_step >= TARGET_STEP:
        add_alert(
            alerts,
            alert_id=f"{arm.arm}:stopped_at_or_after_target",
            severity="warning",
            arm=arm.arm,
            summary="QZ 任务到达 30k 后以 stopped 而非 succeeded 结束",
            evidence={"qz_status": qz_status, "latest_step": latest_step},
            recommendation="核验 step-30000 推理权重；不要把 stopped 等同完整成功。",
        )
    if qz_status == "job_succeeded":
        step_final_ok = TARGET_STEP in checkpoint["complete_steps"]
        final_ok = checkpoint["final_exists"]
        if latest_step < TARGET_STEP or not step_final_ok or not final_ok:
            add_alert(
                alerts,
                alert_id=f"{arm.arm}:success_artifact_mismatch",
                severity="critical",
                arm=arm.arm,
                summary="QZ 显示 succeeded，但 30k 日志/step-30000/final 证据不完整",
                evidence={
                    "latest_step": latest_step,
                    "step_30000_complete": step_final_ok,
                    "final_exists": final_ok,
                },
                recommendation="不要宣布训练完成；人工核验远端退出和共享文件落盘。",
            )
    payload = {
        "arm": arm.arm,
        "job_name": arm.job_name,
        "job_id": arm.job_id,
        "text_repeat": arm.repeat,
        "run_dir": str(run_dir),
        "latest_step": latest_step,
        "target_step": TARGET_STEP,
        "qz": dict(qz),
        "train_log": log,
        "checkpoints": checkpoint,
    }
    return payload, alerts


def overall_status(alerts: Sequence[Mapping[str, Any]], arms: Mapping[str, Any]) -> str:
    if any(item.get("severity") == "critical" for item in alerts):
        return "critical"
    if alerts:
        return "warning"
    if all(
        arm.get("qz", {}).get("status") == "job_succeeded"
        and arm.get("latest_step", 0) >= TARGET_STEP
        for arm in arms.values()
    ):
        return "complete"
    return "healthy"


def render_status_markdown(status: Mapping[str, Any]) -> str:
    lines = [
        "# Batch-44 training health",
        "",
        f"- Observed: `{status['observed_at_utc']}`",
        f"- Overall: **{status['status']}**",
        f"- QZ contract: `{ALLOWED_COMPUTE_GROUP_NAME}`, 1×8 H200, status-only queries",
        "- Resume: inference-only checkpoints; strict optimizer/scheduler/RNG resume is unavailable",
        "",
        "| Arm | QZ | Step | Δ since scan | sec/step | ETA (UTC) | Complete checkpoints |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for arm_name in ("r3", "r5"):
        arm = status["arms"][arm_name]
        log = arm["train_log"]
        delta = log.get("step_delta_since_previous_scan")
        rate = log.get("seconds_per_step")
        lines.append(
            "| {arm} | {qz} | {step}/{target} | {delta} | {rate} | {eta} | {ckpts} |".format(
                arm=arm_name,
                qz=arm["qz"].get("status", "query_error"),
                step=arm["latest_step"],
                target=arm["target_step"],
                delta="—" if delta is None else delta,
                rate="—" if rate is None else f"{rate:.3f}",
                eta=log.get("eta_utc") or "—",
                ckpts=", ".join(map(str, arm["checkpoints"]["complete_steps"])) or "none",
            )
        )
    storage = status["storage"]
    lines.extend(
        [
            "",
            f"QB free: `{storage['available_gib']:.1f} GiB`; "
            f"free inodes: `{storage['inode_available']}`.",
            "",
        ]
    )
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
        arm = f"[{item['arm']}] " if item.get("arm") else ""
        lines.extend(
            [
                f"## {item['severity'].upper()} {arm}{item['id']}",
                "",
                item["summary"],
                "",
                f"Recommendation: {item['recommendation']}",
                "",
            ]
        )
    lines.extend(
        [
            "Reminder: saved model directories contain inference weights only. A remote interruption "
            "cannot be strictly resumed with optimizer/scheduler/RNG continuity.",
            "",
        ]
    )
    return "\n".join(lines)


def scan_once(config: Config) -> dict[str, Any]:
    validate_static_contract(config)
    config.state_root.mkdir(parents=True, exist_ok=True)
    now = utc_now()
    previous_status = read_previous_status(config.state_root / "STATUS.json")
    previous_arms = previous_status.get("arms") if isinstance(previous_status.get("arms"), dict) else {}
    alerts: list[dict[str, Any]] = []
    qz_by_arm: dict[str, dict[str, Any]] = {}

    # One lock and a plain for-loop make the two status calls sequential.  The
    # watcher never invokes qzcli login, so it cannot race a login with itself.
    qz_lock_path = config.state_root / "qz_status_query.lock"
    with qz_lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        for arm in ARMS:
            qz, qz_alerts = query_qz_status(config, arm)
            qz_by_arm[arm.arm] = qz
            alerts.extend(qz_alerts)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    arm_payloads: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        prior = previous_arms.get(arm.arm, {}) if isinstance(previous_arms, dict) else {}
        if not isinstance(prior, dict):
            prior = {}
        payload, arm_alerts = analyze_arm(
            config, arm, qz_by_arm[arm.arm], prior, now
        )
        arm_payloads[arm.arm] = payload
        alerts.extend(arm_alerts)
    storage, storage_alerts = analyze_storage(config)
    alerts.extend(storage_alerts)
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda item: (severity_order.get(str(item.get("severity")), 9), str(item["id"])))
    status = {
        "schema_version": "moss_codecvc.batch44_training_health.v1",
        "observed_at_utc": format_utc(now),
        "status": overall_status(alerts, arm_payloads),
        "poll_seconds": config.poll_seconds,
        "read_only_contract": {
            "qzcli_subcommand": "status",
            "sequential_job_order": [arm.job_id for arm in ARMS],
            "shared_qzcli_home": str(config.qzcli_home),
            "proxy_variables_removed": [
                "HTTPS_PROXY",
                "https_proxy",
                "HTTP_PROXY",
                "http_proxy",
                "ALL_PROXY",
                "all_proxy",
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
        "resume_risk": {
            "strict_resume_available": False,
            "reason": "training checkpoints save inference weights only, without optimizer/scheduler/RNG state",
        },
        "arms": arm_payloads,
        "storage": storage,
        "alerts": alerts,
    }
    atomic_write_json(config.state_root / "STATUS.json", status)
    atomic_write_text(config.state_root / "STATUS.md", render_status_markdown(status))
    append_jsonl(
        config.state_root / "history.jsonl",
        {
            "observed_at_utc": status["observed_at_utc"],
            "status": status["status"],
            "arms": {
                name: {
                    "qz_status": item["qz"].get("status"),
                    "latest_step": item["latest_step"],
                    "eta_utc": item["train_log"].get("eta_utc"),
                    "complete_checkpoints": item["checkpoints"]["complete_steps"],
                }
                for name, item in arm_payloads.items()
            },
            "available_bytes": storage["available_bytes"],
            "inode_available": storage["inode_available"],
            "alert_ids": [item["id"] for item in alerts],
        },
    )
    alert_path = config.state_root / "ALERT.json"
    recommendation_path = config.state_root / "STOP_OR_RECOVERY_RECOMMENDATION.md"
    if alerts:
        alert_payload = {
            "schema_version": "moss_codecvc.batch44_training_health_alert.v1",
            "observed_at_utc": status["observed_at_utc"],
            "status": status["status"],
            "alerts": alerts,
            "automatic_action_taken": False,
        }
        atomic_write_json(alert_path, alert_payload)
        atomic_write_text(recommendation_path, render_recommendation(status))
        append_jsonl(config.state_root / "alert_history.jsonl", alert_payload)
    else:
        # These are current-signal files. Historical alerts remain in
        # alert_history.jsonl and history.jsonl.
        for path in (alert_path, recommendation_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    return status


def write_contract_alert(config: Config, error: ContractError) -> None:
    """Persist a fail-closed identity/configuration error without touching QZ."""

    observed = format_utc(utc_now())
    alert = {
        "id": "watcher:static_contract_error",
        "severity": "critical",
        "arm": None,
        "summary": "Batch-44 health watcher 的硬绑定身份/资源/路径合同失败",
        "evidence": str(error),
        "recommendation": (
            "不要用该 watcher 状态支持训练结论；人工核验两个固定 job、run、ledger 与 "
            "MTTS-3-2-0715 资源。watcher 未调用任何 QZ 修改命令。"
        ),
    }
    payload = {
        "schema_version": "moss_codecvc.batch44_training_health_alert.v1",
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
        "The watcher static contract failed before a trustworthy health scan.\n\n"
        f"Evidence: `{error}`\n\n"
        f"Recommendation: {alert['recommendation']}\n",
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
        help="0 means unlimited in monitor mode; once mode always scans once.",
    )
    return result


def run(config: Config) -> int:
    config.state_root.mkdir(parents=True, exist_ok=True)
    monitor_lock = config.state_root / "monitor.lock"
    with monitor_lock.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError(f"another Batch-44 health watcher holds {monitor_lock}") from exc
        scans = 0
        while True:
            started = time.monotonic()
            try:
                status = scan_once(config)
            except ContractError as exc:
                write_contract_alert(config, exc)
                raise
            scans += 1
            print(
                f"[batch44-health] scan={scans} status={status['status']} "
                + " ".join(
                    f"{name}={item['qz'].get('status')}:{item['latest_step']}/{TARGET_STEP}"
                    for name, item in status["arms"].items()
                ),
                flush=True,
            )
            if config.mode == "once":
                return 0 if status["status"] in {"healthy", "complete"} else 1
            if config.max_scans and scans >= config.max_scans:
                return 0 if status["status"] in {"healthy", "complete"} else 1
            if all(
                item["qz"].get("status") in TERMINAL_QZ_STATUSES
                for item in status["arms"].values()
            ):
                return 0 if status["status"] == "complete" else 1
            elapsed = time.monotonic() - started
            time.sleep(max(0.0, config.poll_seconds - elapsed))


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        config = build_config(args)
        return run(config)
    except ContractError as exc:
        print(f"[batch44-health] CONTRACT ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[batch44-health] interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

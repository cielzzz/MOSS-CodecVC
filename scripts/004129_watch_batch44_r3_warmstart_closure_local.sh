#!/usr/bin/env bash
# Local-only Batch-44 r3 continuation closure orchestrator.
#
# Safe default: print the registered plan and exit without creating state.
#
# Live one-shot scan:
#   ACTION=run MODE=once \
#   CONFIRM_BATCH44_CLOSURE_WATCHER=1 \
#   CONFIRM_BATCH44_LOCAL_EVALUATIONS=1 \
#     bash scripts/004129_watch_batch44_r3_warmstart_closure_local.sh
#
# Persistent monitoring additionally requires:
#   MODE=monitor CONFIRM_BATCH44_MONITOR_LOOP=1

set -euo pipefail

CANONICAL_PROJECT_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
PROJECT_ROOT="${PROJECT_ROOT:-$CANONICAL_PROJECT_ROOT}"
TEST_MODE="${BATCH44_R3_WARMSTART_CLOSURE_WATCHER_TEST_MODE:-0}"
ACTION="${ACTION:-plan}"
MODE="${MODE:-once}"
CONFIRM_WATCHER="${CONFIRM_BATCH44_CLOSURE_WATCHER:-0}"
CONFIRM_LOCAL_EVAL="${CONFIRM_BATCH44_LOCAL_EVALUATIONS:-0}"
CONFIRM_MONITOR="${CONFIRM_BATCH44_MONITOR_LOOP:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_SCANS="${MAX_SCANS:-0}"
MIN_CHECKPOINT_AGE_SEC="${MIN_CHECKPOINT_AGE_SEC:-90}"
MAX_INITIAL_GPU_MEMORY_MIB="${MAX_INITIAL_GPU_MEMORY_MIB:-2048}"
STOP_WHEN_COMPLETE="${STOP_WHEN_COMPLETE:-1}"

STAMP="20260713"
EARLY_FULL320_STEP="20000"
BEST2_CANDIDATE_STEPS="24000 26000 28000 30000"
TRAIN_JOB_ID="job-165f3b1d-8c7d-47d8-8882-bdb86ef642ab"
CONTRACT_SHA256="2d686e5e57b70fcaa3db8c8eb2b306003a38599b2c9ac37023979d80b6d9fc34"
CONTINUATION_RUN_DIR="${CONTINUATION_RUN_DIR:-$PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r3_v1_warmstart10k_to30k}"

STATE_ROOT="${STATE_ROOT:-$PROJECT_ROOT/trainset/local_jobs/ver23_batch44_r3_warmstart_closure_scheduler_${STAMP}}"
CLOSURE_ROOT="${CLOSURE_ROOT:-$PROJECT_ROOT/testset/outputs/batch44_closure_${STAMP}}"
FULL320_RUNNER="${FULL320_RUNNER:-$PROJECT_ROOT/scripts/004127_run_batch44_r3_warmstart_full320_local.sh}"
BEST2_SELECTOR="${BEST2_SELECTOR:-$PROJECT_ROOT/scripts/004126_select_batch44_r3_warmstart_best2.py}"
REPORT_BUILDER="${REPORT_BUILDER:-$PROJECT_ROOT/scripts/004128_build_batch44_completion_reports.py}"
QUICK20_VALIDATOR="${QUICK20_VALIDATOR:-$PROJECT_ROOT/scripts/batch44_r3_warmstart_quick20_validator.py}"
PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
NVIDIA_SMI="${NVIDIA_SMI:-nvidia-smi}"

WATCH_LOCK="$STATE_ROOT/.closure_watcher.lock"
SCAN_JSON="$STATE_ROOT/scan_latest.json"
SCAN_LOG="$STATE_ROOT/watcher.log"
BEST2_JSON="$CLOSURE_ROOT/best2_r3_selection.json"
BEST2_MD="$CLOSURE_ROOT/best2_r3_summary.md"
DISABLED_FINAL_SELECTION="$STATE_ROOT/FINAL_SELECTION_DISABLED_DO_NOT_CREATE.json"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

case "$TEST_MODE:$CONFIRM_WATCHER:$CONFIRM_LOCAL_EVAL:$CONFIRM_MONITOR:$STOP_WHEN_COMPLETE" in
  [01]:[01]:[01]:[01]:[01]) ;;
  *) die "boolean flags must be 0 or 1" ;;
esac
case "$ACTION" in plan|run) ;; *) die "ACTION must be plan or run" ;; esac
case "$MODE" in once|monitor) ;; *) die "MODE must be once or monitor" ;; esac
for value in "$POLL_SECONDS" "$MAX_SCANS" "$MIN_CHECKPOINT_AGE_SEC" "$MAX_INITIAL_GPU_MEMORY_MIB"; do
  case "$value" in ''|*[!0-9]*) die "poll/scan/settle/GPU limits must be non-negative integers" ;; esac
done
[ "$POLL_SECONDS" -gt 0 ] || die "POLL_SECONDS must be positive"
[ "$MAX_INITIAL_GPU_MEMORY_MIB" -gt 0 ] || die "MAX_INITIAL_GPU_MEMORY_MIB must be positive"

echo "=========================================="
echo "Batch-44 r3 warm-start local closure watcher"
echo "  ACTION=$ACTION MODE=$MODE"
echo "  STAGE_1=effective-20000 strict full320"
echo "  STAGE_2=wait strict quick20 for {24000,26000,28000,30000}"
echo "  STAGE_3=004126 Best2 shortlist only"
echo "  STAGE_4=serial 004127 strict full320 for both winners"
echo "  REPORT_REFRESH=004128 after every accepted full320"
echo "  FINAL_SELECTION=disabled"
echo "  REMOTE_JOB_MUTATION=disabled"
echo "  STATE_ROOT=$STATE_ROOT"
echo "  CLOSURE_ROOT=$CLOSURE_ROOT"
echo "=========================================="

if [ "$ACTION" = "plan" ]; then
  echo "[batch44-r3-closure-watch] plan complete; no state, GPU work, or monitor started"
  exit 0
fi

[ "$CONFIRM_WATCHER" = "1" ] \
  || die "ACTION=run requires CONFIRM_BATCH44_CLOSURE_WATCHER=1"
[ "$CONFIRM_LOCAL_EVAL" = "1" ] \
  || die "ACTION=run requires CONFIRM_BATCH44_LOCAL_EVALUATIONS=1"
if [ "$MODE" = "monitor" ]; then
  [ "$CONFIRM_MONITOR" = "1" ] \
    || die "MODE=monitor requires CONFIRM_BATCH44_MONITOR_LOOP=1"
fi
[ -x "$PYTHON" ] || die "Python interpreter is not executable: $PYTHON"
for path in "$FULL320_RUNNER" "$BEST2_SELECTOR" "$REPORT_BUILDER" "$QUICK20_VALIDATOR"; do
  [ -s "$path" ] || die "missing closure dependency: $path"
done
bash -n "$FULL320_RUNNER"
"$PYTHON" -m py_compile "$BEST2_SELECTOR" "$REPORT_BUILDER" "$QUICK20_VALIDATOR"

if [ "$PROJECT_ROOT" != "$CANONICAL_PROJECT_ROOT" ] && [ "$TEST_MODE" != "1" ]; then
  die "non-canonical PROJECT_ROOT is test-only"
fi
if [ "$TEST_MODE" = "0" ]; then
  [ "$FULL320_RUNNER" = "$PROJECT_ROOT/scripts/004127_run_batch44_r3_warmstart_full320_local.sh" ] \
    || die "production full320 runner is hard-locked to 004127"
  [ "$BEST2_SELECTOR" = "$PROJECT_ROOT/scripts/004126_select_batch44_r3_warmstart_best2.py" ] \
    || die "production Best2 selector is hard-locked to 004126"
  [ "$REPORT_BUILDER" = "$PROJECT_ROOT/scripts/004128_build_batch44_completion_reports.py" ] \
    || die "production report builder is hard-locked to 004128"
  [ "$QUICK20_VALIDATOR" = "$PROJECT_ROOT/scripts/batch44_r3_warmstart_quick20_validator.py" ] \
    || die "production quick20 validator path drift"
  [ "$STATE_ROOT" = "$PROJECT_ROOT/trainset/local_jobs/ver23_batch44_r3_warmstart_closure_scheduler_${STAMP}" ] \
    || die "production state root drift"
  [ "$CLOSURE_ROOT" = "$PROJECT_ROOT/testset/outputs/batch44_closure_${STAMP}" ] \
    || die "production closure root drift"
  [ "$MIN_CHECKPOINT_AGE_SEC" -ge 90 ] || die "production checkpoint settle time must be >=90s"
fi
[ ! -L "$STATE_ROOT" ] || die "state root may not be a symlink"
[ ! -L "$CLOSURE_ROOT" ] || die "closure root may not be a symlink"

# Reject any evaluation runner with an executable remote create/stop/delete
# command.  The selector and reporter are Python read/aggregate tools only.
if sed '/^[[:space:]]*#/d' "$FULL320_RUNNER" | rg -n '(qzcli|create-job|stop-job|delete-job)' >/dev/null; then
  die "004127 violates the local-only contract"
fi
if rg -n '(qzcli|create-job|stop-job|delete-job)' "$BEST2_SELECTOR" "$REPORT_BUILDER" >/dev/null; then
  die "closure selector/reporter unexpectedly contains remote mutation tooling"
fi

mkdir -p "$STATE_ROOT" "$CLOSURE_ROOT"
if [ -e "$DISABLED_FINAL_SELECTION" ] || [ -L "$DISABLED_FINAL_SELECTION" ]; then
  die "reserved FINAL_SELECTION-disabled path must remain absent"
fi
mkdir "$WATCH_LOCK" || die "another closure watcher or stale lock exists: $WATCH_LOCK"
"$PYTHON" - "$WATCH_LOCK/owner.json" <<'PY'
import datetime as dt
import json
import os
import socket
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "pid": os.getppid(),
    "hostname": socket.gethostname(),
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "policy": "local closure watcher; never mutates training or remote jobs",
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
cleanup() {
  rm -rf "$WATCH_LOCK"
}
trap cleanup EXIT INT TERM

write_halt() {
  local message="$1"
  "$PYTHON" - "$STATE_ROOT/HALTED.json" "$message" <<'PY'
import datetime as dt
import json
import os
import sys
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "schema": "moss_codecvc.batch44_r3_warmstart_closure_halt.v1",
    "status": "halted",
    "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "reason": sys.argv[2],
    "automatic_retry": False,
    "training_mutation": False,
    "remote_job_mutation": False,
}
tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, path)
PY
}

scan_state() {
  "$PYTHON" - "$PROJECT_ROOT" "$STATE_ROOT" "$CLOSURE_ROOT" "$QUICK20_VALIDATOR" \
    "$BEST2_SELECTOR" "$TEST_MODE" "$TRAIN_JOB_ID" "$CONTRACT_SHA256" <<'PY'
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

project = Path(sys.argv[1]).resolve()
state_root = Path(sys.argv[2]).resolve()
closure_root = Path(sys.argv[3]).resolve()
quick_validator_path = Path(sys.argv[4]).resolve()
selector_path = Path(sys.argv[5]).resolve()
test_mode = sys.argv[6] == "1"
train_job_id = sys.argv[7]
contract_sha = sys.argv[8]
stamp = "20260713"
candidate_steps = (24000, 26000, 28000, 30000)
full_steps = (20000, 24000, 26000, 28000, 30000)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import closure dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


quick = load_module("batch44_closure_quick_validator", quick_validator_path)
selector = load_module("batch44_closure_best2_selector", selector_path)


def digest(path: Path) -> str:
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing/empty artifact: {path}")
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def lexists(path: Path) -> bool:
    return os.path.lexists(path)


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be an object: {path}")
    return value


def verify_artifact(spec: Any, label: str) -> Path:
    if not isinstance(spec, dict):
        raise SystemExit(f"{label} artifact spec missing")
    path = Path(str(spec.get("path") or "")).resolve()
    if not path.is_file():
        raise SystemExit(f"{label} artifact missing: {path}")
    if path.stat().st_size != int(spec.get("size", -1)):
        raise SystemExit(f"{label} artifact size drift: {path}")
    if digest(path) != spec.get("sha256"):
        raise SystemExit(f"{label} artifact SHA drift: {path}")
    return path


def classify_quick(effective: int) -> dict[str, Any]:
    local = effective - 10000
    record = project / f"trainset/local_jobs/ver23_batch44_r3_warmstart_quick20_step{effective}_{stamp}"
    if record.is_symlink():
        raise SystemExit(f"effective-{effective} quick20 record is a symlink")
    completion = record / "COMPLETED.json"
    marker = record / "complete.marker"
    lock = record / ".local_quick20.lock"
    if lock.is_dir():
        return {"status": "running", "record": str(record)}
    pair = (completion.is_file(), marker.is_file())
    if pair == (False, False):
        if lexists(record) and (not record.is_dir() or any(record.iterdir())):
            raise SystemExit(f"effective-{effective} unbound/partial quick20 evidence: {record}")
        return {"status": "pending", "record": str(record)}
    if pair != (True, True):
        raise SystemExit(f"effective-{effective} partial quick20 completion evidence")
    try:
        payload = quick.validate_completion(
            record,
            expected_effective_step=effective,
            expected_continuation_local_step=local,
            expected_train_job_id=train_job_id,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"effective-{effective} strict quick20 validation failed: {exc}") from exc
    if payload.get("warm_start_contract_sha256") != contract_sha:
        raise SystemExit(f"effective-{effective} quick20 contract SHA drift")
    return {
        "status": "complete",
        "record": str(record),
        "completion_sha256": digest(completion),
    }


def full_record(effective: int) -> tuple[Path, Path]:
    record = project / f"trainset/local_jobs/ver23_batch44_r3_warmstart_full320_step{effective}_{stamp}"
    eval_root = project / f"testset/outputs/ver23_batch44_r3_warmstart_full320_{stamp}/step-{effective}"
    return record, eval_root


def validate_full(effective: int, record: Path, eval_root: Path) -> dict[str, Any]:
    local = effective - 10000
    completion_path = record / "COMPLETED.json"
    marker_path = record / "complete.marker"
    completion = load_object(completion_path, "full320 completion")
    marker = load_object(marker_path, "full320 marker")
    expected = {
        "schema": "moss_codecvc.batch44_r3_warmstart_full320_local.v1",
        "status": "complete",
        "backend": "local",
        "base_effective_step": 10000,
        "effective_step": effective,
        "continuation_local_step": local,
        "arm": "r3",
        "text_repeat": 3,
        "train_job_id": train_job_id,
        "record_root": str(record.resolve()),
        "eval_root": str(eval_root.resolve()),
        "expected_warm_start_contract_sha256": contract_sha,
    }
    drift = {
        key: {"expected": wanted, "actual": completion.get(key)}
        for key, wanted in expected.items()
        if completion.get(key) != wanted
    }
    if drift:
        raise SystemExit(f"effective-{effective} full320 completion drift: {drift}")
    marker_expected = {
        "schema": "moss_codecvc.batch44_r3_warmstart_full320_marker.v1",
        "status": "complete",
        "backend": "local",
        "effective_step": effective,
        "continuation_local_step": local,
        "completion_json": str(completion_path.resolve()),
        "completion_sha256": digest(completion_path),
    }
    marker_drift = {
        key: {"expected": wanted, "actual": marker.get(key)}
        for key, wanted in marker_expected.items()
        if marker.get(key) != wanted
    }
    if marker_drift:
        raise SystemExit(f"effective-{effective} full320 marker drift: {marker_drift}")
    checkpoint = completion.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"effective-{effective} checkpoint binding missing")
    wanted_checkpoint = project / (
        "outputs/lora_runs/ver2_9_5_final_r3_v1_warmstart10k_to30k/"
        f"step-{local}"
    )
    if Path(str(checkpoint.get("path") or "")).resolve() != wanted_checkpoint.resolve():
        raise SystemExit(f"effective-{effective} checkpoint path drift")
    files = checkpoint.get("files")
    wanted_files = {
        "adapter_model.safetensors", "adapter_config.json", "README.md",
        "timbre_memory_adapter.pt", "timbre_memory_config.json",
    }
    if not isinstance(files, dict) or set(files) != wanted_files:
        raise SystemExit(f"effective-{effective} checkpoint five-file set drift")
    for name in wanted_files:
        path = verify_artifact(files[name], f"effective-{effective} checkpoint {name}")
        if path != (wanted_checkpoint / name).resolve():
            raise SystemExit(f"effective-{effective} checkpoint artifact escaped: {name}")
    run = completion.get("run")
    if not isinstance(run, dict):
        raise SystemExit(f"effective-{effective} run evidence missing")
    wanted_run = {
        "validation_rows": 320,
        "inference_rows": 320,
        "qwen_asr_rows": 320,
        "audio_rows": 320,
    }
    for key, wanted in wanted_run.items():
        if run.get(key) != wanted:
            raise SystemExit(f"effective-{effective} run {key} drift")
    bnf = run.get("bnf_audit") or {}
    if bnf.get("run_case_counts") != {"no_text": 160, "text": 160}:
        raise SystemExit(f"effective-{effective} inference mode count drift")
    if bnf.get("bnf_extraction_counts") != {"no_text": 160, "text": 0}:
        raise SystemExit(f"effective-{effective} BNF bypass drift")
    metrics = completion.get("metrics")
    if not isinstance(metrics, list) or len(metrics) != 3:
        raise SystemExit(f"effective-{effective} full320 metrics must have 3 scopes")
    indexed = {str(row.get("scope")): row for row in metrics if isinstance(row, dict)}
    if set(indexed) != {"no_text", "text", "all"}:
        raise SystemExit(f"effective-{effective} metric scopes drift")
    for scope, n in (("no_text", 160), ("text", 160), ("all", 320)):
        row = indexed[scope]
        if row.get("effective_step") != effective or row.get("continuation_local_step") != local:
            raise SystemExit(f"effective-{effective} {scope} metric step drift")
        if row.get("n") != n:
            raise SystemExit(f"effective-{effective} {scope} metric n drift")
        for field in (
            "fail_rate", "qwen_primary_error", "qwen_cer", "qwen_wer",
            "wavlm_sim_ref", "wavlm_sim_src", "wavlm_margin",
            "speechbrain_sim_ref", "speechbrain_sim_src", "speechbrain_margin",
        ):
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(f"effective-{effective} {scope}.{field} invalid") from exc
            if not math.isfinite(value):
                raise SystemExit(f"effective-{effective} {scope}.{field} non-finite")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, dict):
        raise SystemExit(f"effective-{effective} result artifact set missing")
    required = {
        "validation", "binding", "runtime", "summary", "qwen_asr",
        "dual_cases", "dual_summary", "diagnostics", "metrics_json",
        "metrics_tsv", "metrics_md", "fail_reasons_json", "fail_reasons_md",
        "unified_eval_input", "unified_eval_input_summary", "manifest_shard0",
        "manifest_shard1", "infer_log_shard0", "infer_log_shard1",
    }
    if not required.issubset(artifacts):
        raise SystemExit(
            f"effective-{effective} result artifacts missing: {sorted(required-set(artifacts))}"
        )
    for name in required:
        verify_artifact(artifacts[name], f"effective-{effective} {name}")
    unified = Path(str(artifacts["unified_eval_input"]["path"]))
    rows = [json.loads(line) for line in unified.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 320 or len({row.get("case_id") for row in rows}) != 320:
        raise SystemExit(f"effective-{effective} unified evaluator input is not unique 320")
    if Counter((row.get("metadata") or {}).get("mode") for row in rows) != Counter({"no_text": 160, "text": 160}):
        raise SystemExit(f"effective-{effective} unified evaluator mode drift")
    if any(row.get("status") != "ok" for row in rows):
        raise SystemExit(f"effective-{effective} unified evaluator has non-ok rows")
    return {
        "status": "complete",
        "record": str(record),
        "eval_root": str(eval_root),
        "completion_sha256": digest(completion_path),
        "metrics": indexed,
    }


def classify_full(effective: int) -> dict[str, Any]:
    record, eval_root = full_record(effective)
    if record.is_symlink() or eval_root.is_symlink():
        raise SystemExit(f"effective-{effective} full320 root is a symlink")
    lock = record / ".local_full320.lock"
    if lock.is_dir():
        return {"status": "running", "record": str(record), "eval_root": str(eval_root)}
    required = [
        record / "COMPLETED.json",
        record / "complete.marker",
        eval_root / "aggregate/metrics.json",
        eval_root / "aggregate/unified_eval_input.jsonl",
    ]
    present = [lexists(path) for path in required]
    if not any(present):
        for root in (record, eval_root):
            if lexists(root) and (not root.is_dir() or any(root.iterdir())):
                raise SystemExit(f"effective-{effective} unbound/partial full320 evidence: {root}")
        return {"status": "pending", "record": str(record), "eval_root": str(eval_root)}
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        raise SystemExit(f"effective-{effective} partial full320 completion evidence")
    return validate_full(effective, record, eval_root)


try:
    selector.audit_contract(project)
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"warm-start contract audit failed: {exc}") from exc

quick_status = {str(step): classify_quick(step) for step in candidate_steps}
full_status = {str(step): classify_full(step) for step in full_steps}
all_quick = all(item["status"] == "complete" for item in quick_status.values())

selection_path = closure_root / "best2_r3_selection.json"
selection_md = closure_root / "best2_r3_summary.md"
selection_exists = selection_path.is_file()
selection_md_exists = selection_md.is_file()
if selection_exists != selection_md_exists:
    raise SystemExit("partial Best2 selection evidence")
selection_status = "missing"
selected_steps: list[int] = []
expected_selection = None
if all_quick:
    try:
        expected_selection = selector.build_selection(project)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Best2 replay failed after all quick20 completed: {exc}") from exc
if selection_exists:
    selection = load_object(selection_path, "Best2 selection")
    selection_status = str(selection.get("status") or "")
    if selection_status == "selected":
        if not all_quick or not isinstance(expected_selection, dict):
            raise SystemExit("Best2 was selected before all registered quick20 evidence completed")
        replay_keys = (
            "schema_version", "experiment_id", "status", "project_root",
            "read_only_contract", "registered_candidate_space", "warm_start",
            "strict_validator", "ranking", "selected_candidate_ids", "best2",
            "candidates",
        )
        replay_drift = {
            key: {"expected": expected_selection.get(key), "actual": selection.get(key)}
            for key in replay_keys
            if selection.get(key) != expected_selection.get(key)
        }
        if replay_drift:
            raise SystemExit(f"Best2 selection drift from strict replay: {replay_drift}")
        expected_summary = selector.render_summary(expected_selection)
        if selection_md.read_text(encoding="utf-8") != expected_summary:
            raise SystemExit("Best2 Markdown summary drift from strict replay")
        selected_steps = [int(row["effective_step"]) for row in selection.get("best2", [])]
        expected_steps = [int(row["effective_step"]) for row in expected_selection.get("best2", [])]
        if selected_steps != expected_steps or len(selected_steps) != 2:
            raise SystemExit("Best2 selected checkpoint mapping drift")
    elif selection_status != "pending":
        raise SystemExit(f"invalid Best2 status: {selection_status!r}")

early = full_status["20000"]["status"]
if early != "complete":
    next_action = "wait_full320" if early == "running" else "run_full320:20000"
elif not all_quick:
    next_action = "wait_quick20_candidates"
elif selection_status != "selected":
    next_action = "select_best2"
else:
    next_action = "complete"
    for step in selected_steps:
        status = full_status[str(step)]["status"]
        if status != "complete":
            next_action = "wait_full320" if status == "running" else f"run_full320:{step}"
            break

payload = {
    "schema": "moss_codecvc.batch44_r3_warmstart_closure_scan.v1",
    "status": "complete" if next_action == "complete" else "waiting",
    "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "backend": "local",
    "training_mutation": False,
    "remote_job_mutation": False,
    "final_selection": "disabled",
    "quick20": quick_status,
    "full320": full_status,
    "all_best2_quick20_complete": all_quick,
    "selection_status": selection_status,
    "selected_steps": selected_steps,
    "next_action": next_action,
}
output = state_root / "scan_latest.json"
temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
print(json.dumps({
    "status": payload["status"],
    "next_action": next_action,
    "selected_steps": selected_steps,
}, sort_keys=True))
PY
}

scan_value() {
  local field="$1"
  "$PYTHON" - "$SCAN_JSON" "$field" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
value = payload[sys.argv[2]]
if isinstance(value, list):
    print(" ".join(str(item) for item in value))
else:
    print(value)
PY
}

checkpoint_probe() {
  local effective="$1" local_step
  local_step=$((effective - 10000))
  "$PYTHON" - "$CONTINUATION_RUN_DIR" "$local_step" "$MIN_CHECKPOINT_AGE_SEC" "$TEST_MODE" <<'PY'
import json
import sys
import time
from pathlib import Path
run_dir = Path(sys.argv[1])
local_step = int(sys.argv[2])
minimum_age = int(sys.argv[3])
test_mode = sys.argv[4] == "1"
checkpoint = run_dir / f"step-{local_step}"
if not checkpoint.is_dir() or checkpoint.is_symlink():
    print("waiting")
    raise SystemExit(0)
minimum_large = 1 if test_mode else 1_000_000
required = {
    "adapter_model.safetensors": minimum_large,
    "adapter_config.json": 1,
    "README.md": 1,
    "timbre_memory_adapter.pt": minimum_large,
    "timbre_memory_config.json": 1,
}
newest = 0.0
for name, minimum in required.items():
    path = checkpoint / name
    if not path.is_file() or path.is_symlink() or path.stat().st_size < minimum:
        print("waiting")
        raise SystemExit(0)
    newest = max(newest, path.stat().st_mtime)
if time.time() - newest < minimum_age:
    print("waiting")
    raise SystemExit(0)
try:
    json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    json.loads((checkpoint / "timbre_memory_config.json").read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"invalid settled checkpoint JSON: {exc}")
print("ready")
PY
}

gpu_ready() {
  command -v "$NVIDIA_SMI" >/dev/null 2>&1 || die "nvidia-smi unavailable"
  local rows names indices
  rows=$("$NVIDIA_SMI" --query-gpu=index --format=csv,noheader,nounits | wc -l | tr -d ' ')
  [ "$rows" = "2" ] || die "local closure requires exactly two GPUs"
  indices=$("$NVIDIA_SMI" --query-gpu=index --format=csv,noheader,nounits | paste -sd, -)
  [ "$indices" = "0,1" ] || die "local closure requires GPU indices 0,1"
  names=$("$NVIDIA_SMI" --query-gpu=name --format=csv,noheader | sort -u)
  [ "$names" = "NVIDIA GeForce RTX 4090" ] || die "local closure requires two RTX 4090 GPUs"
  while IFS= read -r used; do
    if [ "$used" -gt "$MAX_INITIAL_GPU_MEMORY_MIB" ]; then
      return 1
    fi
  done < <("$NVIDIA_SMI" --query-gpu=memory.used --format=csv,noheader,nounits)
  return 0
}

full_completion_sha() {
  local step="$1"
  sha256sum "$PROJECT_ROOT/trainset/local_jobs/ver23_batch44_r3_warmstart_full320_step${step}_${STAMP}/COMPLETED.json" | awk '{print $1}'
}

refresh_report() {
  local step="$1"
  local marker="$STATE_ROOT/report_refreshed_effective-${step}.json"
  local completion_sha
  completion_sha=$(full_completion_sha "$step")
  if [ -f "$marker" ]; then
    "$PYTHON" - "$marker" "$completion_sha" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("completion_sha256") != sys.argv[2]:
    raise SystemExit("accepted full320 completion changed after report refresh")
PY
    return 0
  fi
  "$PYTHON" "$REPORT_BUILDER" \
    --project-root "$PROJECT_ROOT" \
    --output-dir "$CLOSURE_ROOT" \
    --best2-selection "$BEST2_JSON" \
    --final-selection "$DISABLED_FINAL_SELECTION" \
    >>"$SCAN_LOG" 2>&1
  [ -s "$CLOSURE_ROOT/closure_manifest.json" ] || die "004128 did not produce closure_manifest.json"
  "$PYTHON" - "$marker" "$step" "$completion_sha" "$CLOSURE_ROOT/closure_manifest.json" <<'PY'
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
marker = Path(sys.argv[1])
manifest = Path(sys.argv[4])
with manifest.open("rb") as handle:
    manifest_sha = hashlib.file_digest(handle, "sha256").hexdigest()
payload = {
    "schema": "moss_codecvc.batch44_r3_warmstart_report_refresh.v1",
    "status": "complete",
    "effective_step": int(sys.argv[2]),
    "completion_sha256": sys.argv[3],
    "closure_manifest": str(manifest.resolve()),
    "closure_manifest_sha256_at_refresh": manifest_sha,
    "refreshed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
}
temporary = marker.with_name(f".{marker.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, marker)
PY
  echo "[batch44-r3-closure-watch] report refreshed after effective-$step" >>"$SCAN_LOG"
}

refresh_accepted_reports() {
  local selected step status
  status=$("$PYTHON" - "$SCAN_JSON" <<'PY'
import json, sys
p=json.load(open(sys.argv[1], encoding="utf-8"))
print(p["full320"]["20000"]["status"])
PY
)
  [ "$status" != "complete" ] || refresh_report 20000
  selected=$(scan_value selected_steps)
  for step in $selected; do
    status=$("$PYTHON" - "$SCAN_JSON" "$step" <<'PY'
import json, sys
p=json.load(open(sys.argv[1], encoding="utf-8"))
print(p["full320"][sys.argv[2]]["status"])
PY
)
    [ "$status" != "complete" ] || refresh_report "$step"
  done
}

run_full320() {
  local step="$1"
  local dispatch_log="$STATE_ROOT/full320_effective-${step}.dispatch.log"
  if ! ACTION=preflight EFFECTIVE_STEP="$step" \
    MIN_CHECKPOINT_AGE_SEC="$MIN_CHECKPOINT_AGE_SEC" \
    MAX_INITIAL_GPU_MEMORY_MIB="$MAX_INITIAL_GPU_MEMORY_MIB" \
    bash "$FULL320_RUNNER" >"$dispatch_log" 2>&1; then
    if rg -n '(GPU memory is in use|GPUs are not idle|local GPUs are not idle)' "$dispatch_log" >/dev/null; then
      return 75
    fi
    cat "$dispatch_log" >>"$SCAN_LOG"
    return 1
  fi
  cat "$dispatch_log" >>"$SCAN_LOG"
  if ! ACTION=run EFFECTIVE_STEP="$step" \
    CONFIRM_LOCAL_R3_FULL320=1 \
    CONFIRM_EFFECTIVE_STEP="$step" \
    CONFIRM_LOCAL_ONLY=RTX4090x2 \
    MIN_CHECKPOINT_AGE_SEC="$MIN_CHECKPOINT_AGE_SEC" \
    MAX_INITIAL_GPU_MEMORY_MIB="$MAX_INITIAL_GPU_MEMORY_MIB" \
    bash "$FULL320_RUNNER" >"$dispatch_log" 2>&1; then
    if rg -n '(GPU memory is in use|GPUs are not idle|local GPUs are not idle)' "$dispatch_log" >/dev/null \
      && [ ! -e "$PROJECT_ROOT/trainset/local_jobs/ver23_batch44_r3_warmstart_full320_step${step}_${STAMP}" ] \
      && [ ! -e "$PROJECT_ROOT/testset/outputs/ver23_batch44_r3_warmstart_full320_${STAMP}/step-${step}" ]; then
      return 75
    fi
    cat "$dispatch_log" >>"$SCAN_LOG"
    return 1
  fi
  cat "$dispatch_log" >>"$SCAN_LOG"
}

scan_count=0
while true; do
  scan_count=$((scan_count + 1))
  if ! scan_output=$(scan_state 2>&1); then
    write_halt "$scan_output"
    echo "$scan_output" >&2
    exit 1
  fi
  echo "[batch44-r3-closure-watch] scan=$scan_count $scan_output" | tee -a "$SCAN_LOG"
  refresh_accepted_reports
  next_action=$(scan_value next_action)
  case "$next_action" in
    complete)
      echo "[batch44-r3-closure-watch] closure objective evidence complete; FINAL_SELECTION remains disabled" | tee -a "$SCAN_LOG"
      [ "$STOP_WHEN_COMPLETE" = "1" ] && exit 0
      ;;
    wait_full320|wait_quick20_candidates)
      ;;
    select_best2)
      if ! "$PYTHON" "$BEST2_SELECTOR" \
        --project-root "$PROJECT_ROOT" --output-dir "$CLOSURE_ROOT" >>"$SCAN_LOG" 2>&1; then
        write_halt "004126 Best2 selection failed after strict quick20 completion"
        exit 1
      fi
      ;;
    run_full320:*)
      step=${next_action#run_full320:}
      checkpoint_state=$(checkpoint_probe "$step")
      if [ "$checkpoint_state" = "ready" ]; then
        if gpu_ready; then
          rm -f "$STATE_ROOT/WAITING_GPU.json"
          if run_full320 "$step"; then
            :
          else
            run_code=$?
            if [ "$run_code" = "75" ]; then
              "$PYTHON" - "$STATE_ROOT/WAITING_GPU.json" "$step" <<'PY'
import datetime as dt
import json
import os
import sys
from pathlib import Path
path=Path(sys.argv[1])
payload={
    "status":"waiting",
    "reason":"local RTX4090 GPUs became busy during 004127 preflight",
    "effective_step":int(sys.argv[2]),
    "checked_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
    "automatic_retry":True,
}
tmp=path.with_name(f".{path.name}.tmp-{os.getpid()}")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n", encoding="utf-8")
os.replace(tmp,path)
PY
            else
              write_halt "004127 failed for effective-$step; inspect persistent record lock and logs"
              exit 1
            fi
          fi
        else
          "$PYTHON" - "$STATE_ROOT/WAITING_GPU.json" "$step" <<'PY'
import datetime as dt
import json
import os
import sys
from pathlib import Path
path=Path(sys.argv[1])
payload={
    "status":"waiting",
    "reason":"local RTX4090 GPUs busy",
    "effective_step":int(sys.argv[2]),
    "checked_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
    "automatic_retry":True,
}
tmp=path.with_name(f".{path.name}.tmp-{os.getpid()}")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n", encoding="utf-8")
os.replace(tmp,path)
PY
        fi
      fi
      ;;
    *)
      write_halt "unknown closure next action: $next_action"
      exit 1
      ;;
  esac

  if [ "$MODE" = "once" ]; then
    exit 0
  fi
  if [ "$MAX_SCANS" -gt 0 ] && [ "$scan_count" -ge "$MAX_SCANS" ]; then
    exit 0
  fi
  sleep "$POLL_SECONDS"
done

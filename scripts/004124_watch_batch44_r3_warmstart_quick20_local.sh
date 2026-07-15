#!/usr/bin/env bash
# Watch the Batch-44 r3 weights-only continuation and run r3-only quick20
# evaluations on the local dual-RTX4090 host.
#
# The continuation resets its local step counter after loading effective step
# 10000.  This watcher therefore uses the immutable mapping:
#
#   effective_step = 10000 + continuation_local_step
#
# Safe default (one scan, plan only, never starts inference):
#   bash scripts/004124_watch_batch44_r3_warmstart_quick20_local.sh
#
# Live monitoring requires two independent confirmations.  The runner itself
# has a third, per-invocation confirmation (`CONFIRM_RUN=1`) supplied only by
# this watcher after checkpoint and GPU preflight pass:
#   MODE=monitor ACTION=run \
#   CONFIRM_R3_WARMSTART_QUICK20_WATCHER=1 \
#   CONFIRM_R3_WARMSTART_QUICK20_RUN=1 \
#     bash scripts/004124_watch_batch44_r3_warmstart_quick20_local.sh

set -euo pipefail

CANONICAL_PROJECT_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
PROJECT_ROOT="${PROJECT_ROOT:-$CANONICAL_PROJECT_ROOT}"
TEST_MODE="${BATCH44_R3_WARMSTART_QUICK20_WATCHER_TEST_MODE:-0}"
MODE="${MODE:-once}"
ACTION="${ACTION:-plan}"
CONFIRM_WATCHER="${CONFIRM_R3_WARMSTART_QUICK20_WATCHER:-0}"
CONFIRM_RUN="${CONFIRM_R3_WARMSTART_QUICK20_RUN:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_SCANS="${MAX_SCANS:-0}"
MIN_CHECKPOINT_AGE_SEC="${MIN_CHECKPOINT_AGE_SEC:-90}"
MAX_INITIAL_GPU_MEMORY_MIB="${MAX_INITIAL_GPU_MEMORY_MIB:-2048}"
STOP_WHEN_COMPLETE="${STOP_WHEN_COMPLETE:-1}"

STAMP="20260713"
BASE_EFFECTIVE_STEP="10000"
EFFECTIVE_STEPS="12000 14000 16000 18000 20000 22000 24000 26000 28000 30000"
CONTINUATION_RUN_DIR="${CONTINUATION_RUN_DIR:-$PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r3_v1_warmstart10k_to30k}"
TRAIN_JOB_ID="${TRAIN_JOB_ID:-job-165f3b1d-8c7d-47d8-8882-bdb86ef642ab}"
WARM_START_RECORD_ROOT="${WARM_START_RECORD_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_r3_v1_warmstart10k_to30k_20260713}"
WARM_START_CONTRACT="${WARM_START_CONTRACT:-$WARM_START_RECORD_ROOT/warm_start_contract.json}"
TRAIN_LEDGER="${TRAIN_LEDGER:-$WARM_START_RECORD_ROOT/submitted_jobs.tsv}"
EXPECTED_CONTRACT_SHA256="${EXPECTED_CONTRACT_SHA256:-2d686e5e57b70fcaa3db8c8eb2b306003a38599b2c9ac37023979d80b6d9fc34}"
EXPECTED_LEDGER_SHA256="${EXPECTED_LEDGER_SHA256:-f2ae5a3f5eced6fdb358e70ef43fa56aea163b89b5dcef547f7036823d0d973f}"
EXPECTED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"
STATE_ROOT="${STATE_ROOT:-$PROJECT_ROOT/trainset/local_jobs/ver23_batch44_r3_warmstart_quick20_scheduler_${STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/testset/outputs/ver23_batch44_r3_warmstart_quick20_${STAMP}}"
LOCAL_RUNNER="${LOCAL_RUNNER:-$PROJECT_ROOT/scripts/004123_run_batch44_r3_warmstart_quick20_local.sh}"
STRICT_VALIDATOR="${STRICT_VALIDATOR:-$PROJECT_ROOT/scripts/batch44_r3_warmstart_quick20_validator.py}"
PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
NVIDIA_SMI="${NVIDIA_SMI:-nvidia-smi}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

case "$TEST_MODE:$CONFIRM_WATCHER:$CONFIRM_RUN:$STOP_WHEN_COMPLETE" in
  [01]:[01]:[01]:[01]) ;;
  *) die "boolean flags must be 0 or 1" ;;
esac
case "$MODE" in once|monitor) ;; *) die "MODE must be once or monitor" ;; esac
case "$ACTION" in plan|run) ;; *) die "ACTION must be plan or run" ;; esac
for value in "$POLL_SECONDS" "$MAX_SCANS" "$MIN_CHECKPOINT_AGE_SEC" "$MAX_INITIAL_GPU_MEMORY_MIB"; do
  case "$value" in
    ''|*[!0-9]*) die "poll/scan/settle/GPU-memory values must be non-negative integers" ;;
  esac
done
[ "$POLL_SECONDS" -gt 0 ] || die "POLL_SECONDS must be positive"
[ "$MAX_INITIAL_GPU_MEMORY_MIB" -gt 0 ] || die "MAX_INITIAL_GPU_MEMORY_MIB must be positive"
[ -x "$PYTHON" ] || die "Python interpreter is not executable: $PYTHON"
[ -s "$LOCAL_RUNNER" ] || die "missing r3 warm-start local runner: $LOCAL_RUNNER"
[ -s "$STRICT_VALIDATOR" ] || die "missing r3 warm-start strict validator: $STRICT_VALIDATOR"
bash -n "$LOCAL_RUNNER"

if [ "$PROJECT_ROOT" != "$CANONICAL_PROJECT_ROOT" ] && [ "$TEST_MODE" != "1" ]; then
  die "non-canonical PROJECT_ROOT is allowed only in watcher test mode"
fi
if [ "$TEST_MODE" = "0" ]; then
  [ "$CONTINUATION_RUN_DIR" = "$PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r3_v1_warmstart10k_to30k" ] || \
    die "production continuation output is hard-locked"
  [ "$TRAIN_JOB_ID" = "job-165f3b1d-8c7d-47d8-8882-bdb86ef642ab" ] || \
    die "production continuation job ID is hard-locked"
  [ "$WARM_START_CONTRACT" = "$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_r3_v1_warmstart10k_to30k_20260713/warm_start_contract.json" ] || \
    die "production warm-start contract path is hard-locked"
  [ "$LOCAL_RUNNER" = "$PROJECT_ROOT/scripts/004123_run_batch44_r3_warmstart_quick20_local.sh" ] || \
    die "production runner is hard-locked to 004123"
  [ "$STRICT_VALIDATOR" = "$PROJECT_ROOT/scripts/batch44_r3_warmstart_quick20_validator.py" ] || \
    die "production completion validator is hard-locked"
  [ "$MIN_CHECKPOINT_AGE_SEC" -ge 90 ] || \
    die "production checkpoint settle time must be at least 90 seconds"
fi
if [ "$ACTION" = "run" ]; then
  [ "$CONFIRM_WATCHER" = "1" ] || \
    die "ACTION=run requires CONFIRM_R3_WARMSTART_QUICK20_WATCHER=1"
  [ "$CONFIRM_RUN" = "1" ] || \
    die "ACTION=run requires CONFIRM_R3_WARMSTART_QUICK20_RUN=1"
fi

# This is deliberately local-only.  Refuse a runner that can submit remote
# jobs; evaluation must never create a QZ task or mutate training state.
if sed '/^[[:space:]]*#/d' "$LOCAL_RUNNER" | rg -n 'qzcli|create-job|\$\{?QZCLI' >/dev/null; then
  die "004123 violates the local-only/no-QZ contract"
fi

mkdir -p "$STATE_ROOT" "$OUTPUT_ROOT"

audit_warm_start_contract() {
  "$PYTHON" - "$WARM_START_CONTRACT" "$EXPECTED_CONTRACT_SHA256" \
    "$TRAIN_LEDGER" "$EXPECTED_LEDGER_SHA256" "$CONTINUATION_RUN_DIR" \
    "$TRAIN_JOB_ID" "$EXPECTED_COMPUTE_GROUP" "$TEST_MODE" \
    "$STATE_ROOT/contract_audit.json" <<'PY'
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

contract_path = Path(sys.argv[1])
wanted_contract_sha = sys.argv[2]
ledger_path = Path(sys.argv[3])
wanted_ledger_sha = sys.argv[4]
run_dir = Path(sys.argv[5]).resolve()
job_id = sys.argv[6]
compute_group = sys.argv[7]
test_mode = sys.argv[8] == "1"
output = Path(sys.argv[9])


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


errors: list[str] = []
if not contract_path.is_file():
    errors.append(f"missing warm-start contract: {contract_path}")
if not ledger_path.is_file():
    errors.append(f"missing continuation submission ledger: {ledger_path}")
if errors:
    raise SystemExit("Warm-start provenance audit failed:\n- " + "\n- ".join(errors))

contract_sha = digest(contract_path)
ledger_sha = digest(ledger_path)
if contract_sha != wanted_contract_sha:
    errors.append(f"contract SHA256={contract_sha}, expected {wanted_contract_sha}")
if ledger_sha != wanted_ledger_sha:
    errors.append(f"ledger SHA256={ledger_sha}, expected {wanted_ledger_sha}")
payload = json.loads(contract_path.read_text(encoding="utf-8"))
expected = {
    "schema": "batch44_r3_weights_only_warm_start_v1",
    "status": "submitted",
    "job_id": job_id,
    "output_dir": str(run_dir),
    "source_effective_step": 10000,
    "effective_step_offset": 10000,
    "continuation_local_target_step": 20000,
    "effective_target_step": 30000,
    "resume_semantics": "weights_only_warm_start_not_exact_resume",
    "step_mapping": "effective_step = 10000 + continuation_local_step",
}
for key, wanted in expected.items():
    if payload.get(key) != wanted:
        errors.append(f"contract {key}={payload.get(key)!r}, expected {wanted!r}")
overrides = payload.get("mechanical_recovery_overrides") or {}
if overrides.get("warmup_ratio") != 0 or overrides.get("guided_attn_warmup_steps") != 0:
    errors.append(f"second-warmup prevention drifted: {overrides}")
data = payload.get("data") or {}
if (data.get("no_text") or {}).get("repeat") != 1 or (data.get("text") or {}).get("repeat") != 3:
    errors.append("continuation data mix is not old-v1 no_text::1 + text::3")
if not payload.get("full_data_sha256_verified"):
    errors.append("continuation contract lacks full data SHA verification")
if payload.get("state_resets") != ["optimizer", "scheduler", "rng", "global_step", "data_iterator"]:
    errors.append(f"weights-only reset provenance drifted: {payload.get('state_resets')}")

with ledger_path.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if len(rows) != 1:
    errors.append(f"submission ledger row count={len(rows)}, expected 1")
else:
    row = rows[0]
    if row.get("job_id") != job_id or Path(row.get("out_dir", "")).resolve() != run_dir:
        errors.append(f"submission ledger identity mismatch: {row}")
    if row.get("compute_group") != compute_group:
        errors.append(f"continuation compute group drifted: {row.get('compute_group')}")
if not re.fullmatch(r"job-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", job_id, re.I):
    errors.append(f"invalid training job ID: {job_id}")
if errors:
    raise SystemExit("Warm-start provenance audit failed:\n- " + "\n- ".join(errors))

result = {
    "schema": "moss_codecvc.batch44_r3_warmstart_quick20_contract_audit.v1",
    "status": "pass",
    "contract": str(contract_path.resolve()),
    "warm_start_contract_sha256": contract_sha,
    "ledger": str(ledger_path.resolve()),
    "ledger_sha256": ledger_sha,
    "train_job_id": job_id,
    "continuation_run_dir": str(run_dir),
    "base_effective_step": 10000,
    "effective_step_offset": 10000,
    "resume_semantics": payload["resume_semantics"],
    "test_mode": test_mode,
}
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[r3-warmstart-quick20-watch] contract PASS sha256={contract_sha}")
PY
}

audit_warm_start_contract

record_root_for() {
  local effective="$1"
  printf '%s/trainset/local_jobs/ver23_batch44_r3_warmstart_quick20_step%s_%s\n' \
    "$PROJECT_ROOT" "$effective" "$STAMP"
}

checkpoint_probe() {
  local effective="$1" local_step
  local_step=$((effective - BASE_EFFECTIVE_STEP))
  "$PYTHON" - "$CONTINUATION_RUN_DIR" "$effective" "$local_step" \
    "$MIN_CHECKPOINT_AGE_SEC" "$TEST_MODE" <<'PY'
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

run_dir = Path(sys.argv[1])
effective = int(sys.argv[2])
local_step = int(sys.argv[3])
min_age = int(sys.argv[4])
test_mode = sys.argv[5] == "1"
if effective != 10000 + local_step or local_step not in range(2000, 20001, 2000):
    raise SystemExit("invalid continuation/effective step mapping")
checkpoint = run_dir / f"step-{local_step}"
if not checkpoint.is_dir():
    print("waiting")
    print(f"missing:{checkpoint}")
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
    if not path.is_file():
        print("waiting")
        print(f"missing:{name}")
        raise SystemExit(0)
    stat = path.stat()
    newest = max(newest, stat.st_mtime)
    if stat.st_size < minimum:
        print("waiting")
        print(f"small:{name}:{stat.st_size}")
        raise SystemExit(0)
age = time.time() - newest
if age < min_age:
    print("waiting")
    print(f"settling:{age:.0f}s<{min_age}s")
    raise SystemExit(0)
try:
    json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    cfg = json.loads((checkpoint / "timbre_memory_config.json").read_text(encoding="utf-8"))
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"invalid checkpoint JSON: {exc}")


def equal(got: object, wanted: object) -> bool:
    if isinstance(wanted, float):
        try:
            return math.isclose(float(got), wanted, rel_tol=0.0, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return got == wanted


expected_cfg = {
    "content_cross_attn_enabled": True,
    "content_cross_attn_layers": "all",
    "content_cross_attn_feature_dim": 768,
    "content_cross_attn_gate_init": -0.5,
    "content_cross_attn_output_scale": 0.3,
    "content_encoder_layers": 2,
    "guided_attn_loss_weight": 0.05,
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
bad = [key for key, wanted in expected_cfg.items() if not equal(cfg.get(key), wanted)]
if bad:
    raise SystemExit("continuation checkpoint config drift: " + ",".join(bad))
print("ready")
print(f"checkpoint={checkpoint.resolve()} age={age:.0f}s")
PY
}

scan_completions() {
  "$PYTHON" - "$PROJECT_ROOT" "$STATE_ROOT" "$OUTPUT_ROOT" "$EFFECTIVE_STEPS" \
    "$BASE_EFFECTIVE_STEP" "$TRAIN_JOB_ID" "$EXPECTED_CONTRACT_SHA256" \
    "$CONTINUATION_RUN_DIR" "$STAMP" "$STRICT_VALIDATOR" "$TEST_MODE" <<'PY'
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

project = Path(sys.argv[1]).resolve()
state_root = Path(sys.argv[2]).resolve()
output_root = Path(sys.argv[3]).resolve()
steps = [int(value) for value in sys.argv[4].split()]
base = int(sys.argv[5])
job_id = sys.argv[6]
contract_sha = sys.argv[7]
run_dir = Path(sys.argv[8]).resolve()
stamp = sys.argv[9]
strict_validator_path = Path(sys.argv[10]).resolve()
test_mode = sys.argv[11] == "1"

strict_validator = None
if not test_mode:
    spec = importlib.util.spec_from_file_location(
        "batch44_r3_warmstart_watcher_strict_validator", strict_validator_path
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import strict validator: {strict_validator_path}")
    strict_validator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = strict_validator
    spec.loader.exec_module(strict_validator)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def record_for(effective: int) -> Path:
    return project / (
        "trainset/local_jobs/"
        f"ver23_batch44_r3_warmstart_quick20_step{effective}_{stamp}"
    )


def validate_artifact(item: object, label: str) -> None:
    if not isinstance(item, dict):
        raise SystemExit(f"{label} artifact is not an object")
    path = Path(str(item.get("path") or ""))
    if not path.is_file():
        raise SystemExit(f"{label} artifact missing: {path}")
    if item.get("size") is not None and path.stat().st_size != int(item["size"]):
        raise SystemExit(f"{label} artifact size mismatch: {path}")
    if item.get("sha256") and digest(path) != item["sha256"]:
        raise SystemExit(f"{label} artifact SHA mismatch: {path}")


def validate_complete(record: Path, effective: int) -> list[dict[str, object]]:
    local_step = effective - base
    completion_path = record / "COMPLETED.json"
    marker_path = record / "complete.marker"
    metrics_path = record / "metrics.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected_top = {
        "schema": "moss_codecvc.batch44_r3_warmstart_quick20_completion.v1",
        "status": "complete",
        "backend": "local",
        "step": effective,
        "effective_step": effective,
        "base_effective_step": base,
        "continuation_local_step": local_step,
        "checkpoint": str((run_dir / f"step-{local_step}").resolve()),
        "train_job_id": job_id,
        "warm_start_contract_sha256": contract_sha,
    }
    for key, wanted in expected_top.items():
        got = completion.get(key)
        if key == "checkpoint":
            got = str(Path(str(got or "")).resolve())
        if got != wanted:
            raise SystemExit(f"effective-{effective} completion {key}={got!r}, expected {wanted!r}")
    expected_marker = {
        "schema": "moss_codecvc.batch44_r3_warmstart_quick20_complete_marker.v1",
        "status": "complete",
        "step": effective,
        "effective_step": effective,
        "base_effective_step": base,
        "continuation_local_step": local_step,
    }
    for key, wanted in expected_marker.items():
        if marker.get(key) != wanted:
            raise SystemExit(f"effective-{effective} marker {key} drifted")
    if marker.get("completed_json_sha256") != digest(completion_path):
        raise SystemExit(f"effective-{effective} completion marker SHA mismatch")
    if isinstance(completion.get("metrics"), dict):
        validate_artifact(completion["metrics"].get("json"), f"effective-{effective} metrics.json")
    rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 2:
        raise SystemExit(f"effective-{effective} metrics rows must be exactly 2")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("arm")), str(row.get("mode")))
        seen.add(key)
        expected_row = {
            "step": effective,
            "base_effective_step": base,
            "continuation_local_step": local_step,
            "effective_step": effective,
            "checkpoint": str((run_dir / f"step-{local_step}").resolve()),
            "warm_start_contract_sha256": contract_sha,
            "train_job_id": job_id,
        }
        for field, wanted in expected_row.items():
            got = row.get(field)
            if field == "checkpoint":
                got = str(Path(str(got or "")).resolve())
            if got != wanted:
                raise SystemExit(f"effective-{effective} {key} {field}={got!r}, expected {wanted!r}")
        for field in ("fail", "cer", "sim_ref", "sim_src", "margin", "ref_bound", "ref_content_f1"):
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(f"effective-{effective} {key} invalid {field}") from exc
            if not math.isfinite(value):
                raise SystemExit(f"effective-{effective} {key} non-finite {field}")
        if not math.isclose(float(row["margin"]), float(row["sim_ref"]) - float(row["sim_src"]), rel_tol=0.0, abs_tol=1e-6):
            raise SystemExit(f"effective-{effective} {key} margin arithmetic mismatch")
        if int(row.get("n", 0)) != 20:
            raise SystemExit(f"effective-{effective} {key} n must be 20")
    if seen != {("r3", "no_text"), ("r3", "text")}:
        raise SystemExit(f"effective-{effective} metrics identity mismatch: {sorted(seen)}")
    runs = completion.get("runs")
    if not isinstance(runs, list) or {(r.get("arm"), r.get("mode")) for r in runs} != seen:
        raise SystemExit(f"effective-{effective} completion runs identity mismatch")
    return rows


statuses: list[dict[str, object]] = []
all_metrics: list[dict[str, object]] = []
alerts: list[dict[str, object]] = []
active_records: list[str] = []
for effective in steps:
    local_step = effective - base
    if local_step not in range(2000, 20001, 2000):
        raise SystemExit(f"invalid scheduled mapping effective={effective} local={local_step}")
    record = record_for(effective)
    parts = [record / name for name in ("COMPLETED.json", "complete.marker", "metrics.json", "metrics.tsv", "metrics.md")]
    any_part = any(os.path.lexists(path) for path in parts)
    all_parts = all(path.is_file() and path.stat().st_size > 0 for path in parts)
    lock = record / ".local_quick20.lock"
    if all_parts:
        if strict_validator is not None:
            strict_validator.validate_completion(
                record,
                expected_effective_step=effective,
                expected_continuation_local_step=local_step,
                expected_train_job_id=job_id,
            )
        rows = validate_complete(record, effective)
        all_metrics.extend(rows)
        for row in rows:
            if row["mode"] == "no_text" and float(row["margin"]) < 0.0:
                alerts.append({
                    "effective_step": effective,
                    "continuation_local_step": local_step,
                    "margin": float(row["margin"]),
                    "sim_ref": float(row["sim_ref"]),
                    "sim_src": float(row["sim_src"]),
                    "cer": float(row["cer"]),
                    "record_root": str(record.resolve()),
                    "train_job_id": job_id,
                })
        status, detail = "complete", "strict local completion"
    elif lock.is_dir():
        runtime = record / "LOCAL_RUNTIME.json"
        frozen_runner = record / "004123_run_batch44_r3_warmstart_quick20_local.frozen.sh"
        if not runtime.is_file() or not frozen_runner.is_file():
            raise SystemExit(f"effective-{effective} lock lacks runtime/frozen runner provenance")
        active_records.append(str(record.resolve()))
        status, detail = "running_local", "004123 local lock active"
    elif any_part:
        raise SystemExit(f"effective-{effective} has partial completion evidence: {record}")
    elif os.path.lexists(record):
        if not record.is_dir() or any(record.iterdir()):
            raise SystemExit(f"effective-{effective} has unbound partial record: {record}")
        status, detail = "waiting_checkpoint", "empty record root"
    else:
        status, detail = "waiting_checkpoint", "record not created"
    statuses.append({
        "effective_step": effective,
        "continuation_local_step": local_step,
        "status": status,
        "detail": detail,
        "record_root": str(record),
    })
if len(active_records) > 1:
    raise SystemExit(f"multiple active local evaluations violate serialization: {active_records}")
first_noncomplete_index = next(
    (index for index, row in enumerate(statuses) if row["status"] != "complete"),
    len(statuses),
)
for index, row in enumerate(statuses):
    if index > first_noncomplete_index and row["status"] in {"complete", "running_local"}:
        raise SystemExit(
            "out-of-order quick20 state violates strict serialization: "
            f"first_noncomplete={statuses[first_noncomplete_index]} later={row}"
        )

atomic_json(state_root / "scan_latest.json", statuses)
with (state_root / "scan_latest.tsv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(statuses[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(statuses)
if all_metrics:
    atomic_json(output_root / "metrics_all.json", all_metrics)
    with (output_root / "metrics_all.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_metrics[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(all_metrics)
if alerts:
    alert = {
        "schema": "moss_codecvc.batch44_r3_warmstart_negative_margin_alert.v1",
        "status": "alert",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "trigger": "strictly validated r3 no_text quick20 WavLM margin < 0",
        "scheduler_action": "stop scheduling later local quick20 evaluations",
        "training_action": "alert/report only; watcher never stops, kills, or mutates the training job",
        "train_job_id": job_id,
        "alerts": alerts,
    }
    for root in (state_root, output_root):
        atomic_json(root / "ALERT_NEGATIVE_NO_TEXT_MARGIN.json", alert)
summary = {
    "complete": sum(row["status"] == "complete" for row in statuses),
    "total": len(statuses),
    "alert": bool(alerts),
    "active": len(active_records),
    "first_incomplete": next((row for row in statuses if row["status"] != "complete"), None),
}
atomic_json(state_root / "scan_summary.json", summary)
print(f"[r3-warmstart-quick20-audit] complete={summary['complete']}/{summary['total']} alert={summary['alert']} active={summary['active']}")
PY
}

summary_value() {
  local field="$1"
  "$PYTHON" - "$STATE_ROOT/scan_summary.json" "$field" <<'PY'
import json
import sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))[sys.argv[2]]
print("1" if value is True else "0" if value is False else value)
PY
}

first_incomplete() {
  "$PYTHON" - "$STATE_ROOT/scan_summary.json" <<'PY'
import json
import sys
from pathlib import Path
row = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("first_incomplete")
if isinstance(row, dict):
    print(row["effective_step"])
    print(row["continuation_local_step"])
    print(row["status"])
PY
}

gpu_preflight() {
  local output
  output="$($NVIDIA_SMI --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits)" || \
    die "nvidia-smi GPU preflight failed"
  "$PYTHON" - "$MAX_INITIAL_GPU_MEMORY_MIB" "$TEST_MODE" "$output" <<'PY'
import sys

limit = int(sys.argv[1])
test_mode = sys.argv[2] == "1"
lines = [line.strip() for line in sys.argv[3].splitlines() if line.strip()]
rows = []
for line in lines:
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 4:
        raise SystemExit(f"malformed nvidia-smi row: {line}")
    rows.append((int(fields[0]), fields[1], int(fields[2]), int(fields[3])))
if [row[0] for row in rows] != [0, 1]:
    raise SystemExit(f"local quick20 requires exactly GPU indices 0,1; got {rows}")
if not test_mode and any(row[1] != "NVIDIA GeForce RTX 4090" for row in rows):
    raise SystemExit(f"local quick20 requires two RTX 4090 GPUs; got {rows}")
busy = [row for row in rows if row[2] > limit]
if busy:
    raise SystemExit(f"GPU memory preflight exceeds {limit} MiB: {busy}")
print(f"[r3-warmstart-quick20-watch] GPU preflight PASS rows={rows}")
PY
}

run_scan() {
  local scan="$1" first effective local_step status probe ready detail record
  scan_completions
  cp "$STATE_ROOT/scan_latest.tsv" "$STATE_ROOT/scan_${scan}.tsv"
  echo "[r3-warmstart-quick20-watch] scan=$scan action=$ACTION utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  awk -F '\t' 'NR == 1 || $3 != "waiting_checkpoint" {print}' "$STATE_ROOT/scan_latest.tsv"
  if [ "$(summary_value alert)" = "1" ]; then
    echo "[r3-warmstart-quick20-watch] ALERT: no_text margin<0; later evaluation scheduling paused; training untouched"
    return 0
  fi
  if [ "$(summary_value active)" != "0" ]; then
    echo "[r3-warmstart-quick20-watch] one local evaluation is active; no second task scheduled"
    return 0
  fi
  first="$(first_incomplete)"
  effective="$(printf '%s\n' "$first" | sed -n '1p')"
  local_step="$(printf '%s\n' "$first" | sed -n '2p')"
  status="$(printf '%s\n' "$first" | sed -n '3p')"
  if [ -z "$effective" ]; then
    echo "[r3-warmstart-quick20-watch] all 10 continuation checkpoints complete"
    return 0
  fi
  if [ "$status" = "running_local" ]; then
    echo "[r3-warmstart-quick20-watch] effective=$effective local=$local_step evaluation still running; serialization holds"
    return 0
  fi
  [ "$status" = "waiting_checkpoint" ] || die "unexpected status=$status effective=$effective"
  probe="$(checkpoint_probe "$effective")"
  ready="$(printf '%s\n' "$probe" | sed -n '1p')"
  detail="$(printf '%s\n' "$probe" | sed -n '2p')"
  if [ "$ready" != "ready" ]; then
    echo "[r3-warmstart-quick20-watch] waiting effective=$effective local=$local_step $detail"
    return 0
  fi
  record="$(record_root_for "$effective")"
  if [ "$ACTION" = "plan" ]; then
    echo "[r3-warmstart-quick20-watch] next_ready effective=$effective local=$local_step record=$record (plan only)"
    return 0
  fi
  gpu_preflight
  echo "[r3-warmstart-quick20-watch] RUN effective=$effective local=$local_step record=$record"
  EFFECTIVE_STEP="$effective" CONTINUATION_LOCAL_STEP="$local_step" \
    CHECKPOINT="$CONTINUATION_RUN_DIR/step-$local_step" TRAIN_JOB_ID="$TRAIN_JOB_ID" \
    WARM_START_CONTRACT="$WARM_START_CONTRACT" RECORD_ROOT="$record" \
    OUTPUT_ROOT="$OUTPUT_ROOT" DRY_RUN=0 CONFIRM_RUN=1 \
    bash "$LOCAL_RUNNER"
  if [ "$TEST_MODE" = "0" ]; then
    scan_completions
    post="$($PYTHON - "$STATE_ROOT/scan_latest.json" "$effective" <<'PY'
import json
import sys
from pathlib import Path
rows = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
effective = int(sys.argv[2])
matches = [row for row in rows if int(row["effective_step"]) == effective]
if len(matches) != 1:
    raise SystemExit(f"post-run identity drift: {matches}")
print(matches[0]["status"])
PY
)"
    [ "$post" = "complete" ] || \
      die "004123 returned zero without a strictly validated completion for effective-$effective"
  fi
  echo "[r3-warmstart-quick20-watch] runner returned successfully effective=$effective local=$local_step"
}

LOCK_DIR="$STATE_ROOT/.watch.lock"
PID_FILE="$STATE_ROOT/monitor.pid"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  die "another r3 warm-start quick20 watcher appears active: $LOCK_DIR"
fi
cleanup() {
  if [ -s "$PID_FILE" ] && [ "$(cat "$PID_FILE" 2>/dev/null || true)" = "$$" ]; then
    rm -f "$PID_FILE"
  fi
  rm -f "$LOCK_DIR/owner.txt"
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
printf 'pid=%s host=%s mode=%s action=%s started=%s\n' \
  "$$" "$(hostname)" "$MODE" "$ACTION" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCK_DIR/owner.txt"
printf '%s\n' "$$" > "$PID_FILE"

scan=0
while :; do
  scan=$((scan + 1))
  run_scan "$scan"
  if [ "$(summary_value alert)" = "1" ]; then
    break
  fi
  if [ "$STOP_WHEN_COMPLETE" = "1" ] && [ "$(summary_value complete)" = "10" ]; then
    break
  fi
  if [ "$MODE" = "once" ]; then
    break
  fi
  if [ "$MAX_SCANS" -gt 0 ] && [ "$scan" -ge "$MAX_SCANS" ]; then
    echo "[r3-warmstart-quick20-watch] reached MAX_SCANS=$MAX_SCANS"
    break
  fi
  sleep "$POLL_SECONDS"
done

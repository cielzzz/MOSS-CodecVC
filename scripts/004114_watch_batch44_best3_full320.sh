#!/usr/bin/env bash
# Wait for the complete Batch-44 v1 quick20 schedule, select the registered
# 26k/28k/30k Best3, and optionally run their full320 evaluations on the
# local dual-RTX-4090 development host.
#
# In live mode it then waits for every selected-step full320 artifact, validates
# each one with 004107's strict provenance validator, and builds 004104's three
# anonymous blind20 pages.  It deliberately stops at BLIND20_READY.json: it
# never chooses a subjective winner, never creates FINAL_SELECTION.json, and
# never launches Batch-42 final inference/scoring.  Those stages require human
# listening and completed review JSONs.
#
# Safe one-shot readiness check (no qzcli call):
#   MODE=once ACTION=plan bash scripts/004114_watch_batch44_best3_full320.sh
#
# Repeated local preflight after all evidence is ready (no inference):
#   MODE=monitor ACTION=preflight bash scripts/004114_watch_batch44_best3_full320.sh
#
# Local recovery mode after selected full320 jobs were submitted elsewhere:
#   MODE=monitor ACTION=blind-only bash scripts/004114_watch_batch44_best3_full320.sh
#
# Local full320 execution is intentionally double-gated here and remains gated
# in 004118.  It has no QZ submission path:
#   MODE=monitor ACTION=run CONFIRM_LOCAL_FULL320_ORCHESTRATOR=1 \
#     bash scripts/004114_watch_batch44_best3_full320.sh

set -euo pipefail

CANONICAL_PROJECT_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
PROJECT_ROOT="${PROJECT_ROOT:-$CANONICAL_PROJECT_ROOT}"
TEST_MODE="${BATCH44_BEST3_TEST_MODE:-0}"
STAMP="20260713"
MODE="${MODE:-once}"
ACTION="${ACTION:-plan}"
CONFIRM_LOCAL_FULL320_ORCHESTRATOR="${CONFIRM_LOCAL_FULL320_ORCHESTRATOR:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_SCANS="${MAX_SCANS:-0}"
STOP_WHEN_BLIND_READY="${STOP_WHEN_BLIND_READY:-1}"

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
SELECTOR="${SELECTOR:-$PROJECT_ROOT/scripts/004103_select_batch43_best3.py}"
QUICK_PROVENANCE_VALIDATOR="${QUICK_PROVENANCE_VALIDATOR:-$CANONICAL_PROJECT_ROOT/scripts/004103_select_batch43_best3.py}"
LOCAL_FULL_WRAPPER="${LOCAL_FULL_WRAPPER:-$PROJECT_ROOT/scripts/004118_run_batch44_v1_paired_full320_local.sh}"
BLIND_BUILDER="${BLIND_BUILDER:-$PROJECT_ROOT/scripts/004104_build_batch43_best3_blind20.py}"
FULL_VALIDATOR="${FULL_VALIDATOR:-$PROJECT_ROOT/scripts/004107_finalize_batch43_pathx_final.py}"

BEST3_ROOT="${BEST3_ROOT:-$PROJECT_ROOT/testset/outputs/batch44_best3_${STAMP}}"
SELECTION_JSON="${SELECTION_JSON:-$BEST3_ROOT/best3_selection.json}"
SELECTION_MD="${SELECTION_MD:-$BEST3_ROOT/best3_selection.md}"
STATE_ROOT="${STATE_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_best3_orchestrator_${STAMP}}"
PLAN_ROOT="${PLAN_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_best3_full320_plan_${STAMP}}"
BLIND_OUTPUT_ROOT="${BLIND_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/listening_frontend/seedtts_valid_benchmark/batch44_best3_${STAMP}}"
BLIND_PRIVATE_ROOT="${BLIND_PRIVATE_ROOT:-$BEST3_ROOT/private_blind20}"
BLIND_READY="$BLIND_PRIVATE_ROOT/BLIND20_READY.json"
QUICK_STATE_ROOT="$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_quick20_scheduler_${STAMP}"
QUICK_EVAL_ROOT="$PROJECT_ROOT/testset/outputs/ver23_batch44_quick20_${STAMP}"
FULL_STATE_ROOT="$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_paired_full320_scheduler_${STAMP}"
FULL_EVAL_ROOT="$PROJECT_ROOT/testset/outputs/ver23_batch44_paired_full320_${STAMP}"
QUICK_STEPS="2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 22000 24000 26000 28000 30000"

R3_TRAIN_JOB_ID="job-2b91d332-d500-4279-84f9-0a6a81a376aa"
R5_TRAIN_JOB_ID="job-b8eb2f1f-a3eb-483b-a289-b4cce281525c"
ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122" # MTTS-3-2-0715
ALLOWED_SPEC="67b10bc6-78b0-41a3-aaf4-358eeeb99009"              # 1x8 H200
EXPECTED_EVAL_CODE_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch37_eval_20260711_1092820"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

case "$TEST_MODE:$CONFIRM_LOCAL_FULL320_ORCHESTRATOR:$STOP_WHEN_BLIND_READY" in
  [01]:[01]:[01]) ;;
  *) die "boolean flags must be 0 or 1" ;;
esac
case "$MODE" in once|monitor) ;; *) die "MODE must be once or monitor" ;; esac
case "$ACTION" in plan|preflight|blind-only|run) ;; *) die "ACTION must be plan, preflight, blind-only, or run" ;; esac
for value in "$POLL_SECONDS" "$MAX_SCANS"; do
  case "$value" in ''|*[!0-9]*) die "POLL_SECONDS and MAX_SCANS must be non-negative integers" ;; esac
done
[ "$POLL_SECONDS" -gt 0 ] || die "POLL_SECONDS must be positive"
[ -x "$PYTHON" ] || die "Python interpreter is not executable: $PYTHON"

if [ "$PROJECT_ROOT" != "$CANONICAL_PROJECT_ROOT" ] && [ "$TEST_MODE" != "1" ]; then
  die "non-canonical PROJECT_ROOT is allowed only with BATCH44_BEST3_TEST_MODE=1"
fi
if [ "$ACTION" = "run" ]; then
  [ "$CONFIRM_LOCAL_FULL320_ORCHESTRATOR" = "1" ] || \
    die "ACTION=run requires CONFIRM_LOCAL_FULL320_ORCHESTRATOR=1"
  [ "$TEST_MODE" = "0" ] || die "test mode may not execute the local-run branch"
  [ "$PROJECT_ROOT" = "$CANONICAL_PROJECT_ROOT" ] || die "local run requires canonical PROJECT_ROOT"
fi

[ -s "$SELECTOR" ] || die "missing Best3 selector: $SELECTOR"
[ -s "$QUICK_PROVENANCE_VALIDATOR" ] || \
  die "missing quick20 provenance validator: $QUICK_PROVENANCE_VALIDATOR"
[ -s "$BLIND_BUILDER" ] || die "missing Best3 blind20 builder: $BLIND_BUILDER"
[ -s "$FULL_VALIDATOR" ] || die "missing strict full320/final validator: $FULL_VALIDATOR"
if [ "$ACTION" = "preflight" ] || [ "$ACTION" = "run" ]; then
  [ -s "$LOCAL_FULL_WRAPPER" ] || die "missing local Best3 full320 wrapper: $LOCAL_FULL_WRAPPER"
  bash -n "$LOCAL_FULL_WRAPPER"
fi

if [ "$TEST_MODE" = "0" ]; then
  [ "$SELECTOR" = "$PROJECT_ROOT/scripts/004103_select_batch43_best3.py" ] || \
    die "production selector is hard-locked to canonical 004103"
  [ "$QUICK_PROVENANCE_VALIDATOR" = "$PROJECT_ROOT/scripts/004103_select_batch43_best3.py" ] || \
    die "production quick20 provenance validation is hard-locked to canonical 004103"
  [ "$LOCAL_FULL_WRAPPER" = "$PROJECT_ROOT/scripts/004118_run_batch44_v1_paired_full320_local.sh" ] || \
    die "production full320 runner is hard-locked to local 004118"
  [ "$BLIND_BUILDER" = "$PROJECT_ROOT/scripts/004104_build_batch43_best3_blind20.py" ] || \
    die "production blind builder is hard-locked to canonical 004104"
  [ "$FULL_VALIDATOR" = "$PROJECT_ROOT/scripts/004107_finalize_batch43_pathx_final.py" ] || \
    die "production full320 validator is hard-locked to canonical 004107"
  grep -Fq 'ACTION=run CONFIRM_LOCAL_FULL320=1' "$LOCAL_FULL_WRAPPER" || \
    die "004118 lost its explicit local-run confirmation gate"
  grep -Fq 'GPU_REQUIREMENT=2x NVIDIA GeForce RTX 4090' "$LOCAL_FULL_WRAPPER" || \
    die "004118 lost the local dual-RTX-4090 resource contract"
  grep -Fq 'local record must not contain submitted_jobs.tsv' "$LOCAL_FULL_WRAPPER" || \
    die "004118 lost the no-QZ-ledger invariant"
fi

mkdir -p "$STATE_ROOT" "$BEST3_ROOT"

write_state() {
  local state="$1"
  local detail="$2"
  "$PYTHON" - "$STATE_ROOT/scan_latest.json" "$state" "$detail" "$ACTION" "$MODE" <<'PY'
import datetime as dt
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": "moss_codecvc.batch44_v1_best3_orchestrator_state.v1",
    "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "state": sys.argv[2],
    "detail": sys.argv[3],
    "action": sys.argv[4],
    "mode": sys.argv[5],
}
tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, path)
PY
}

alert_path() {
  for path in \
    "$QUICK_STATE_ROOT/ALERT_NEGATIVE_NO_TEXT_MARGIN.json" \
    "$QUICK_EVAL_ROOT/ALERT_NEGATIVE_NO_TEXT_MARGIN.json" \
    "$FULL_STATE_ROOT/ALERT_FULL320_RED_FLAGS.json" \
    "$FULL_EVAL_ROOT/ALERT_FULL320_RED_FLAGS.json" \
    "$FULL_STATE_ROOT/ALERT_NEGATIVE_NO_TEXT_MARGIN.json" \
    "$FULL_EVAL_ROOT/ALERT_NEGATIVE_NO_TEXT_MARGIN.json"; do
    if [ -s "$path" ]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

audit_complete_quick20() {
  "$PYTHON" - "$PROJECT_ROOT" "$STATE_ROOT/quick20_audit.json" "$STAMP" \
    "$QUICK_STEPS" "$QUICK_PROVENANCE_VALIDATOR" <<'PY'
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

project = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
stamp = sys.argv[3]
steps = [int(item) for item in sys.argv[4].split()]
validator_path = Path(sys.argv[5]).resolve()
spec = importlib.util.spec_from_file_location(
    "batch44_best3_quick20_provenance", validator_path
)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot import quick20 provenance validator: {validator_path}")
validator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = validator
spec.loader.exec_module(validator)
missing: list[str] = []
audited = []

for step in steps:
    try:
        metrics_json = validator.metrics_path(project, step, stamp)
        record = metrics_json.parent
        rows = validator.load_metrics(metrics_json, project_root=project, step=step)
        provenance = validator.audit_quick20_provenance(
            metrics_json, project_root=project, step=step
        )
    except validator.PendingEvidence as exc:
        missing.append(str(exc))
        continue
    for arm in ("r3", "r5"):
        margin = float(rows[(arm, "no_text")]["margin"])
        if margin < 0.0:
            raise SystemExit(f"{metrics_json}: {arm} no_text margin is negative ({margin})")
    audited.append({
        "step": step,
        "record_root": str(record.resolve()),
        **provenance,
    })

if missing:
    print("pending: incomplete quick20 schedule\n- " + "\n- ".join(missing[:12]), file=sys.stderr)
    raise SystemExit(3)
if len(audited) != 15:
    raise SystemExit(f"expected 15 completed quick20 steps, got {len(audited)}")
result = {
    "schema_version": "moss_codecvc.batch44_v1_complete_quick20_audit.v1",
    "status": "complete",
    "audited_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "steps": steps,
    "completed_steps": len(audited),
    "training_jobs": validator.EXPECTED_TRAIN_JOBS,
    "backend_counts": dict(sorted(Counter(row["backend"] for row in audited).items())),
    "accepted_backend_contracts": {
        "qz": "legacy completion.v1 with strict MTTS-3-2-0715 / one 8xH200 ledger",
        "local": "completion.v2 with strict xyzhang-dev host / two RTX 4090 and no QZ ledger",
    },
    "records": audited,
}
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
print(f"complete quick20={len(audited)}/15")
PY
}

select_best3() {
  local rc=0
  "$PYTHON" "$SELECTOR" \
    --project-root "$PROJECT_ROOT" \
    --quick20-stamp "$STAMP" \
    --output-json "$SELECTION_JSON" \
    --output-md "$SELECTION_MD" \
    --no-pending-output > "$STATE_ROOT/selector_latest.log" 2>&1 || rc=$?
  if [ "$rc" = "3" ]; then
    return 3
  fi
  if [ "$rc" != "0" ]; then
    sed -n '1,160p' "$STATE_ROOT/selector_latest.log" >&2
    die "Best3 selector failed with rc=$rc"
  fi
}

audit_selection() {
  "$PYTHON" - "$SELECTION_JSON" "$PROJECT_ROOT" "$STATE_ROOT/selection_audit.json" <<'PY'
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

selection_path = Path(sys.argv[1]).resolve()
project = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3])
selection = json.loads(selection_path.read_text(encoding="utf-8"))
if selection.get("schema_version") != "moss_codecvc.batch44_v1_best3_selection.v1":
    raise SystemExit("wrong Best3 schema")
if selection.get("status") != "selected" or selection.get("experiment_id") != "batch44_v1" or selection.get("data_version") != "v1_20260709":
    raise SystemExit("Best3 is not the registered Batch-44 v1 selection")
if selection.get("registered_candidate_space") != {
    "arms": ["r3", "r5"], "steps": [26000, 28000, 30000], "candidate_count": 6,
}:
    raise SystemExit("Best3 candidate space drift")
selected = selection.get("selected_candidate_ids")
if not isinstance(selected, list) or len(selected) != 3 or len(set(selected)) != 3:
    raise SystemExit("Best3 must contain exactly three unique candidates")
jobs = {
    "r3": "job-2b91d332-d500-4279-84f9-0a6a81a376aa",
    "r5": "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c",
}
runs = {"r3": "ver2_9_5_final_r3_v1_30k", "r5": "ver2_9_5_final_r5_v1_30k"}
repeats = {"r3": 3, "r5": 5}
rows = selection.get("candidates")
if not isinstance(rows, list) or len(rows) != 6:
    raise SystemExit("Best3 must retain all six ranked candidates")
indexed = {}
for row in rows:
    arm, step = row.get("arm"), row.get("step")
    if arm not in runs or step not in {26000, 28000, 30000}:
        raise SystemExit("candidate outside registered Batch-44 v1 space")
    candidate_id = f"{arm}_step-{step}"
    expected_checkpoint = (project / "outputs/lora_runs" / runs[arm] / f"step-{step}").resolve()
    if (
        row.get("candidate_id") != candidate_id
        or row.get("train_job_id") != jobs[arm]
        or row.get("text_repeat") != repeats[arm]
        or Path(str((row.get("checkpoint") or {}).get("path") or "")).resolve() != expected_checkpoint
    ):
        raise SystemExit(f"candidate provenance drift: {candidate_id}")
    if "_v2_30k" in json.dumps(row, sort_keys=True):
        raise SystemExit(f"candidate retains forbidden Batch-43 v2 identity: {candidate_id}")
    indexed[candidate_id] = row
expected_ids = {f"{arm}_step-{step}" for arm in runs for step in (26000, 28000, 30000)}
if set(indexed) != expected_ids:
    raise SystemExit("Best3 six-candidate identity set drift")
if set(selected) != {cid for cid, row in indexed.items() if row.get("selected_for_full320") is True}:
    raise SystemExit("Best3 selected flags disagree")
steps = sorted({int(indexed[candidate_id]["step"]) for candidate_id in selected})
result = {
    "schema_version": "moss_codecvc.batch44_v1_best3_selection_audit.v1",
    "status": "ready",
    "audited_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "selection_json": str(selection_path),
    "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
    "selected_candidate_ids": selected,
    "selected_steps": steps,
    "note": "paired full320 may also evaluate the unselected counterpart at each selected step",
}
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
print(" ".join(str(step) for step in steps))
PY
}

strict_replay_best3_selection() {
  "$PYTHON" - "$FULL_VALIDATOR" "$SELECTION_JSON" "$PROJECT_ROOT" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

validator_path = Path(sys.argv[1])
selection_path = Path(sys.argv[2]).resolve()
project = Path(sys.argv[3]).resolve()
spec = importlib.util.spec_from_file_location(
    "batch44_best3_orchestrator_selection_replay", validator_path
)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot import strict Best3 validator: {validator_path}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
_payload, selected, _candidates = module.load_best3(
    selection_path,
    project_root=project,
)
print(json.dumps({"status": "strict_replay_pass", "selected": selected}, sort_keys=True))
PY
}

audit_selected_full320_state() {
  "$PYTHON" - "$STATE_ROOT/selection_audit.json" "$PROJECT_ROOT" "$STAMP" \
    "$STATE_ROOT/full320_state.json" "$ALLOWED_COMPUTE_GROUP" "$ALLOWED_SPEC" \
    "$R3_TRAIN_JOB_ID" "$R5_TRAIN_JOB_ID" "$EXPECTED_EVAL_CODE_ROOT" \
    "$FULL_VALIDATOR" <<'PY'
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

audit_path = Path(sys.argv[1])
project = Path(sys.argv[2]).resolve()
stamp = sys.argv[3]
output = Path(sys.argv[4])
compute_group, spec, r3_job, r5_job = sys.argv[5:9]
code_root = Path(sys.argv[9]).resolve()
validator_path = Path(sys.argv[10]).resolve()
audit = json.loads(audit_path.read_text(encoding="utf-8"))
states = []
job_re = re.compile(
    r"^job-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
sha_re = re.compile(r"^[0-9a-f]{64}$")
gpu_uuid_re = re.compile(r"^GPU-[0-9a-fA-F-]{36}$")
validator_spec = importlib.util.spec_from_file_location(
    "batch44_best3_full320_state_validator", validator_path
)
if validator_spec is None or validator_spec.loader is None:
    raise SystemExit(f"cannot import strict full320 validator: {validator_path}")
validator = importlib.util.module_from_spec(validator_spec)
sys.modules[validator_spec.name] = validator
validator_spec.loader.exec_module(validator)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_artifact(value, *, label: str, expected: Path | None = None, parent: Path | None = None) -> Path:
    if not isinstance(value, dict):
        raise SystemExit(f"{label}: expected artifact object")
    path = Path(str(value.get("path") or "")).expanduser().resolve()
    if expected is not None and path != expected.resolve():
        raise SystemExit(f"{label}: path={path}, expected={expected.resolve()}")
    if parent is not None and path.parent != parent.resolve():
        raise SystemExit(f"{label}: artifact is outside {parent.resolve()}")
    expected_sha = value.get("sha256")
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"{label}: missing/empty artifact {path}")
    if not isinstance(expected_sha, str) or not sha_re.fullmatch(expected_sha):
        raise SystemExit(f"{label}: invalid SHA256")
    if sha256(path) != expected_sha:
        raise SystemExit(f"{label}: SHA256 drift")
    size = value.get("size")
    if size is not None and size != path.stat().st_size:
        raise SystemExit(f"{label}: size drift")
    return path


def read_qz_ledger(path: Path, *, step: int, record: Path, step_root: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 1:
        raise SystemExit(f"{path}: expected exactly one QZ submission")
    row = rows[0]
    expected = {
        "step": str(step),
        "compute_group": compute_group,
        "spec": spec,
        "record_root": str(record.resolve()),
        "step_root": str(step_root.resolve()),
        "code_root": str(code_root),
        "r3_train_job_id": r3_job,
        "r5_train_job_id": r5_job,
    }
    bad = {key: row.get(key) for key, wanted in expected.items() if row.get(key) != wanted}
    if bad:
        raise SystemExit(f"{path}: selected full320 QZ provenance drift: {bad}")
    job_id = str(row.get("job_id") or "")
    if not job_re.fullmatch(job_id):
        raise SystemExit(f"{path}: invalid QZ job id")
    return row, job_id


def validate_local_completion(*, completion: Path, marker: Path, record: Path, step: int, step_root: Path):
    payload = json.loads(completion.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "batch44_v1_paired_full320_v1"
        or payload.get("backend") != "local"
        or payload.get("status") != "complete"
        or payload.get("step") != step
    ):
        raise SystemExit(f"{completion}: invalid local full320 schema/backend/identity")
    marker_text = marker.read_text(encoding="utf-8").strip()
    expected_marker = f"COMPLETED.json sha256\t{sha256(completion)}"
    if marker_text != expected_marker:
        raise SystemExit(f"{marker}: local completion SHA marker drift")
    if os.path.lexists(record / "submitted_jobs.tsv"):
        raise SystemExit(f"{record}: local full320 must not contain a QZ submission ledger")
    execution = payload.get("execution")
    if not isinstance(execution, dict):
        raise SystemExit(f"{completion}: missing local execution provenance")
    hostname = str(execution.get("hostname") or "")
    models = execution.get("gpu_models")
    memories = execution.get("gpu_memory_total_mib")
    uuids = execution.get("gpu_uuids")
    if (
        not re.fullmatch(r"xyzhang-dev--[A-Za-z0-9-]+", hostname)
        or execution.get("gpu_count") != 2
        or execution.get("gpu_indices") != [0, 1]
        or models != ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090"]
        or not isinstance(memories, list)
        or len(memories) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 48000
            for value in memories
        )
        or not isinstance(uuids, list)
        or len(uuids) != 2
        or any(not gpu_uuid_re.fullmatch(str(item or "")) for item in uuids)
        or not sha_re.fullmatch(str(execution.get("gpu_inventory_sha256") or ""))
    ):
        raise SystemExit(f"{completion}: local RTX 4090 execution provenance drift")
    inventory = Path(str(execution.get("gpu_inventory") or "")).expanduser().resolve()
    if (
        inventory != (record / "runtime_gpu_inventory.json").resolve()
        or not inventory.is_file()
        or sha256(inventory) != execution["gpu_inventory_sha256"]
    ):
        raise SystemExit(f"{completion}: local GPU inventory artifact drift")
    runner = validate_artifact(
        payload.get("runner"), label="local full320 frozen runner", parent=record
    )
    if runner.name != "004118_run_batch44_v1_paired_full320_local.frozen.sh":
        raise SystemExit(f"{completion}: unexpected local full320 runner {runner.name}")
    artifacts = payload.get("artifacts")
    required = {
        "completeness_json": step_root / "aggregate/completeness.json",
        "dual_encoder_cases_csv": step_root / "aggregate/dual_encoder_cases.csv",
        "paired_metrics_json": step_root / "aggregate/paired_metrics.json",
        "paired_metrics_tsv": step_root / "aggregate/paired_metrics.tsv",
        "paired_metrics_md": step_root / "aggregate/paired_metrics.md",
    }
    if not isinstance(artifacts, dict) or not set(required).issubset(artifacts):
        raise SystemExit(f"{completion}: local full320 artifact set drift")
    for name, expected in required.items():
        validate_artifact(artifacts[name], label=f"local full320 {name}", expected=expected)
    validator.validate_full320_step(
        step=step,
        completion_path=completion,
        metrics_path=step_root / "aggregate/paired_metrics.json",
        project_root=project,
    )
    return payload, f"local:{hostname}"


for step in audit["selected_steps"]:
    qz_record = project / f"trainset/qz_jobs/ver23_batch44_paired_full320_step{step}_{stamp}"
    local_record = project / f"trainset/local_jobs/ver23_batch44_paired_full320_step{step}_{stamp}"
    step_root = project / f"testset/outputs/ver23_batch44_paired_full320_{stamp}/step-{step}"
    metrics = step_root / "aggregate/paired_metrics.json"
    qz_completion, qz_marker = qz_record / "COMPLETED.json", qz_record / "complete.marker"
    local_completion, local_marker = local_record / "COMPLETED.json", local_record / "complete.marker"
    qz_ledger = qz_record / "submitted_jobs.tsv"
    local_ledger = local_record / "submitted_jobs.tsv"
    qz_core = [qz_completion.is_file(), qz_marker.is_file()]
    local_core = [local_completion.is_file(), local_marker.is_file()]
    if os.path.lexists(local_ledger):
        state = "inconsistent_partial_completion"
        job_id = ""
        backend = "local"
        completion_path = ""
        record = local_record
    elif all(qz_core) and all(local_core):
        state = "conflicting_backend_completions"
        job_id = ""
        backend = "conflict"
        completion_path = ""
        record = local_record
    elif any(qz_core) and not all(qz_core) or any(local_core) and not all(local_core):
        state = "inconsistent_partial_completion"
        job_id = ""
        backend = "unknown"
        completion_path = ""
        record = local_record if any(local_core) else qz_record
    elif all(local_core):
        if not metrics.is_file():
            state = "inconsistent_partial_completion"
            job_id = ""
        elif qz_ledger.is_file():
            state = "conflicting_backend_completions"
            job_id = ""
        else:
            _payload, job_id = validate_local_completion(
                completion=local_completion,
                marker=local_marker,
                record=local_record,
                step=step,
                step_root=step_root,
            )
            state = "complete_existing"
        backend = "local"
        completion_path = str(local_completion.resolve())
        record = local_record
    elif all(qz_core):
        if not metrics.is_file() or not qz_ledger.is_file():
            state = "inconsistent_partial_completion"
            job_id = ""
        else:
            _row, job_id = read_qz_ledger(
                qz_ledger, step=step, record=qz_record, step_root=step_root
            )
            payload = json.loads(qz_completion.read_text(encoding="utf-8"))
            if payload.get("backend") not in {None, "qz"}:
                raise SystemExit(f"{qz_completion}: invalid QZ backend")
            validator.validate_full320_step(
                step=step,
                completion_path=qz_completion,
                metrics_path=metrics,
                project_root=project,
            )
            state = "complete_existing"
        backend = "qz"
        completion_path = str(qz_completion.resolve())
        record = qz_record
    elif qz_ledger.is_file():
        _row, job_id = read_qz_ledger(
            qz_ledger, step=step, record=qz_record, step_root=step_root
        )
        state = "submitted_waiting"
        backend = "qz"
        completion_path = ""
        record = qz_record
    elif (qz_record / ".live_submit.lock").is_dir() or (local_record / ".local_run.lock").is_dir():
        state = "locked_manual_audit"
        job_id = ""
        backend = "local" if (local_record / ".local_run.lock").is_dir() else "qz"
        completion_path = ""
        record = local_record if backend == "local" else qz_record
    else:
        state = "not_dispatched"
        job_id = ""
        backend = "none"
        completion_path = ""
        record = local_record
    states.append({
        "step": step,
        "state": state,
        "backend": backend,
        "job_id": job_id,
        "completion_path": completion_path,
        "record_root": str(record),
        "step_root": str(step_root),
    })
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"status": "audited", "steps": states}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if any(row["state"] in {"inconsistent_partial_completion", "conflicting_backend_completions", "locked_manual_audit"} for row in states):
    print("manual_audit")
elif any(row["state"] == "submitted_waiting" for row in states):
    print("waiting_existing")
elif states and all(row["state"] == "complete_existing" for row in states):
    print("all_complete")
else:
    print("dispatchable")
PY
}

validate_full320_and_blind_ready() {
  local require_blind="$1"
  "$PYTHON" - "$FULL_VALIDATOR" "$SELECTION_JSON" "$PROJECT_ROOT" "$STAMP" \
    "$BLIND_READY" "$require_blind" "$STATE_ROOT/full320_state.json" <<'PY'
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

validator_path = Path(sys.argv[1])
selection_path = Path(sys.argv[2]).resolve()
project = Path(sys.argv[3]).resolve()
stamp = sys.argv[4]
blind_ready = Path(sys.argv[5]).resolve()
require_blind = sys.argv[6] == "1"
full_state_path = Path(sys.argv[7]).resolve()
spec = importlib.util.spec_from_file_location("batch44_best3_orchestrator_validator", validator_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot import strict validator: {validator_path}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

_payload, selected, candidates = module.load_best3(
    selection_path,
    project_root=project,
)
full_state = json.loads(full_state_path.read_text(encoding="utf-8"))
state_by_step = {
    int(row["step"]): row
    for row in full_state.get("steps", [])
    if isinstance(row, dict) and row.get("state") == "complete_existing"
}
for step in sorted({int(candidates[candidate_id]["step"]) for candidate_id in selected}):
    state = state_by_step.get(step)
    if state is None or state.get("backend") not in {"qz", "local"}:
        raise SystemExit(f"step-{step}: missing validated QZ/local full320 state")
    completion = Path(str(state.get("completion_path") or "")).resolve()
    metrics = (
        project
        / f"testset/outputs/ver23_batch44_paired_full320_{stamp}/step-{step}/aggregate/paired_metrics.json"
    )
    module.validate_full320_step(
        step=step,
        completion_path=completion,
        metrics_path=metrics,
        project_root=project,
    )
if require_blind:
    module.load_blind_ready(
        blind_ready,
        selected,
        best3_path=selection_path,
        project_root=project,
    )
print(json.dumps({
    "status": "validated",
    "selected": selected,
    "blind_ready": str(blind_ready) if require_blind else None,
}, sort_keys=True))
PY
}

build_blind20() {
  local bindings="$STATE_ROOT/blind20_bindings.tsv"
  "$PYTHON" - "$SELECTION_JSON" "$PROJECT_ROOT" "$STAMP" "$bindings" \
    "$STATE_ROOT/full320_state.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

selection_path, project = map(Path, sys.argv[1:3])
stamp = sys.argv[3]
output = Path(sys.argv[4])
full_state = json.loads(Path(sys.argv[5]).read_text(encoding="utf-8"))
completion_by_step = {
    int(row["step"]): Path(str(row.get("completion_path") or "")).resolve()
    for row in full_state.get("steps", [])
    if isinstance(row, dict) and row.get("state") == "complete_existing"
}
selection = json.loads(selection_path.read_text(encoding="utf-8"))
selected = selection["selected_candidate_ids"]
candidates = {row["candidate_id"]: row for row in selection["candidates"]}
rows = []
for candidate_id in selected:
    row = candidates[candidate_id]
    arm, step = row["arm"], int(row["step"])
    diagnostics = (
        project
        / f"testset/outputs/ver23_batch44_paired_full320_{stamp}/step-{step}/aggregate/dual_encoder_cases.csv"
    ).resolve()
    run_id = f"ver2_9_5_final_{arm}_step-{step}_no_text_seedtts160_d2d3_seed1234"
    completion = completion_by_step.get(step)
    if completion is None or not completion.is_file():
        raise SystemExit(f"step-{step}: missing validated full320 completion binding")
    rows.append({
        "candidate_id": candidate_id,
        "diagnostics": str(diagnostics),
        "run_id": run_id,
        "completion": str(completion),
    })
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=("candidate_id", "diagnostics", "run_id", "completion"),
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(rows)
PY

  local args=(
    "$PYTHON" "$BLIND_BUILDER"
    --selection "$SELECTION_JSON"
    --output-root "$BLIND_OUTPUT_ROOT"
    --private-root "$BLIND_PRIVATE_ROOT"
  )
  local candidate_id diagnostics run_id completion
  while IFS=$'\t' read -r candidate_id diagnostics run_id completion; do
    [ "$candidate_id" = "candidate_id" ] && continue
    args+=(--candidate-diagnostics "$candidate_id=$diagnostics")
    args+=(--candidate-run "$candidate_id=$run_id")
    args+=(--candidate-completion "$candidate_id=$completion")
  done < "$bindings"
  "${args[@]}"
}

finish_blind_ready() {
  validate_full320_and_blind_ready 0 > "$STATE_ROOT/full320_strict_validation.json"
  if [ ! -s "$BLIND_READY" ]; then
    build_blind20 > "$STATE_ROOT/blind20_builder.log" 2>&1
  fi
  validate_full320_and_blind_ready 1 > "$STATE_ROOT/blind20_strict_validation.json"
  write_terminal_manifest "$STATE_ROOT/BLIND20_READY.json" "blind20_ready"
  write_state "blind20_ready" "$BLIND_READY"
  echo "[batch44-best3-watch] strict full320 validation passed; blind20 ready=$BLIND_READY"
}

write_terminal_manifest() {
  local output="$1"
  local status="$2"
  "$PYTHON" - "$output" "$status" "$STATE_ROOT/selection_audit.json" \
    "$STATE_ROOT/full320_state.json" "$ACTION" <<'PY'
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
status = sys.argv[2]
selection_audit = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
full_state = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
payload = {
    "schema_version": "moss_codecvc.batch44_v1_best3_orchestrator_terminal.v1",
    "status": status,
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "action": sys.argv[5],
    "selection": selection_audit,
    "selected_full320_state_before_action": full_state,
    "scope_boundary": (
        "Best3 selection/full320/blind-page orchestration only; human reviews, winner "
        "selection, final inference, scoring and Batch-42 table publication are never automated here"
    ),
}
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY
}

run_scan() {
  local scan="$1"
  local found_alert="" quick_output="" quick_rc=0 selection_steps="" full_state=""
  echo "[batch44-best3-watch] scan=$scan action=$ACTION utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  found_alert="$(alert_path || true)"
  if [ -n "$found_alert" ]; then
    write_state "blocked_by_registered_alert" "$found_alert"
    echo "[batch44-best3-watch] BLOCKED registered alert=$found_alert; no selector/QZ action"
    return 20
  fi

  quick_output="$(audit_complete_quick20 2>&1)" || quick_rc=$?
  if [ "$quick_rc" = "3" ]; then
    write_state "waiting_quick20" "15-step quick20 schedule is incomplete"
    echo "[batch44-best3-watch] waiting for complete 2k..30k quick20 evidence"
    printf '%s\n' "$quick_output" | sed -n '1,20p'
    return 0
  fi
  if [ "$quick_rc" != "0" ]; then
    printf '%s\n' "$quick_output" >&2
    die "complete quick20 audit failed with rc=$quick_rc"
  fi
  echo "[batch44-best3-watch] $quick_output"

  local selector_rc=0
  select_best3 || selector_rc=$?
  if [ "$selector_rc" = "3" ]; then
    write_state "waiting_selector_evidence" "selector still reports pending checkpoint evidence"
    echo "[batch44-best3-watch] quick20 is complete but selector checkpoint audit is pending"
    return 0
  fi
  [ "$selector_rc" = "0" ] || return "$selector_rc"
  selection_steps="$(audit_selection)"
  strict_replay_best3_selection > "$STATE_ROOT/selection_strict_replay.json"
  echo "[batch44-best3-watch] Best3 selected steps=$selection_steps selection=$SELECTION_JSON"

  found_alert="$(alert_path || true)"
  if [ -n "$found_alert" ]; then
    write_state "blocked_by_registered_alert" "$found_alert"
    echo "[batch44-best3-watch] BLOCKED alert appeared after selection=$found_alert"
    return 20
  fi

  full_state="$(audit_selected_full320_state)"
  case "$full_state" in
    manual_audit)
      write_state "manual_full320_audit_required" "$STATE_ROOT/full320_state.json"
      echo "[batch44-best3-watch] selected full320 state is inconsistent/locked; no evaluation action"
      return 21
      ;;
    waiting_existing)
      write_state "waiting_existing_full320" "$STATE_ROOT/full320_state.json"
      echo "[batch44-best3-watch] a historical QZ selected-step full320 is still running; no duplicate local run"
      return 0
      ;;
    all_complete)
      if [ "$ACTION" = "run" ] || [ "$ACTION" = "blind-only" ]; then
        finish_blind_ready
        return 10
      fi
      ;;
    dispatchable) ;;
    *) die "unexpected selected full320 state: $full_state" ;;
  esac

  if [ "$ACTION" = "blind-only" ]; then
    write_state "waiting_selected_full320" "$STATE_ROOT/full320_state.json"
    echo "[batch44-best3-watch] blind-only mode: selected full320 is not complete; no evaluation action"
    return 0
  fi

  case "$ACTION" in
    plan)
      write_terminal_manifest "$STATE_ROOT/PLAN_READY.json" "plan_ready"
      write_state "plan_ready" "$SELECTION_JSON"
      echo "[batch44-best3-watch] plan ready; no GPU work or remote submission was started"
      return 10
      ;;
    preflight)
      local wrapper_rc=0 step="" pending_steps=""
      pending_steps="$("$PYTHON" - "$STATE_ROOT/full320_state.json" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(" ".join(str(row["step"]) for row in payload["steps"] if row["state"] == "not_dispatched"))
PY
)"
      for step in $pending_steps; do
        STEP="$step" ACTION=preflight PROJECT_ROOT="$PROJECT_ROOT" PYTHON="$PYTHON" \
          bash "$LOCAL_FULL_WRAPPER" || wrapper_rc=$?
        if [ "$wrapper_rc" -ne 0 ]; then
          write_state "full320_preflight_failed" \
            "004118 local preflight step-$step rc=$wrapper_rc; no inference was started"
          echo "ERROR: Batch-44 Best3 local full320 preflight step-$step failed rc=$wrapper_rc" >&2
          return 2
        fi
      done
      write_terminal_manifest "$STATE_ROOT/PREFLIGHT_COMPLETE.json" "local_preflight_complete"
      write_state "preflight_complete" "$STATE_ROOT/PREFLIGHT_COMPLETE.json"
      echo "[batch44-best3-watch] all missing selected-step local preflights passed; no inference was started"
      return 10
      ;;
    blind-only)
      die "internal state error: blind-only reached the dispatch branch"
      ;;
    run)
      local wrapper_rc=0 step="" pending_steps="" state_after=""
      pending_steps="$("$PYTHON" - "$STATE_ROOT/full320_state.json" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(" ".join(str(row["step"]) for row in payload["steps"] if row["state"] == "not_dispatched"))
PY
)"
      for step in $pending_steps; do
        STEP="$step" ACTION=run CONFIRM_LOCAL_FULL320=1 PROJECT_ROOT="$PROJECT_ROOT" \
          PYTHON="$PYTHON" bash "$LOCAL_FULL_WRAPPER" || wrapper_rc=$?
        if [ "$wrapper_rc" -ne 0 ]; then
          audit_selected_full320_state > "$STATE_ROOT/full320_state_after_failed_local_run.txt" 2>&1 || true
          write_state "full320_local_run_failed" \
            "004118 local run step-$step rc=$wrapper_rc; inspect local lock/process/artifacts before recovery"
          echo "ERROR: Batch-44 Best3 local full320 step-$step failed rc=$wrapper_rc" >&2
          return 2
        fi
      done
      state_after="$(audit_selected_full320_state)"
      if [ "$state_after" != "all_complete" ]; then
        write_state "manual_full320_audit_required" \
          "local runners returned success but strict selected-step state is $state_after"
        echo "ERROR: local full320 runners returned without complete strict evidence: $state_after" >&2
        return 21
      fi
      finish_blind_ready
      return 10
      ;;
  esac
}

LOCK_DIR="$STATE_ROOT/.watch.lock"
PID_FILE="$STATE_ROOT/monitor.pid"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  die "another Batch-44 Best3 watcher or stale lock exists: $LOCK_DIR"
fi
cleanup() {
  if [ -s "$PID_FILE" ] && [ "$(cat "$PID_FILE" 2>/dev/null || true)" = "$$" ]; then
    rm -f "$PID_FILE"
  fi
  rm -f "$LOCK_DIR/owner.txt"
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
printf '%s\n' "pid=$$ host=$(hostname) mode=$MODE action=$ACTION started=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCK_DIR/owner.txt"
printf '%s\n' "$$" > "$PID_FILE"

scan=0
while :; do
  scan=$((scan + 1))
  set +e
  run_scan "$scan"
  rc=$?
  set -e
  if [ "$rc" = "10" ] && [ "$STOP_WHEN_BLIND_READY" = "1" ]; then
    exit 0
  fi
  if [ "$rc" = "20" ] || [ "$rc" = "21" ]; then
    exit "$rc"
  fi
  if [ "$rc" != "0" ] && [ "$rc" != "10" ]; then
    exit "$rc"
  fi
  if [ "$MODE" = "once" ]; then
    break
  fi
  if [ "$MAX_SCANS" -gt 0 ] && [ "$scan" -ge "$MAX_SCANS" ]; then
    echo "[batch44-best3-watch] MAX_SCANS=$MAX_SCANS reached"
    break
  fi
  sleep "$POLL_SECONDS"
done
exit 0

#!/usr/bin/env bash
# Watch Batch-44 r3/r5 checkpoints and run paired quick20 locally.
#
# Historical migration contract:
#   * step 2000/4000/6000: legacy QZ completion.v1 under trainset/qz_jobs
#   * step 8000: local completion.v2 under the same legacy qz_jobs spelling
#   * step 10000..30000: local completion.v2 under trainset/local_jobs
#
# Safe one-shot plan (default; never starts inference):
#   bash scripts/004119_watch_batch44_v1_quick20_local.sh
#
# Live monitoring is explicitly gated and runs at most one paired checkpoint
# at a time.  004117 blocks until all four lanes and completion provenance are
# finished, so later steps cannot overlap it:
#   MODE=monitor ACTION=run CONFIRM_LOCAL_QUICK20_WATCHER=1 \
#     bash scripts/004119_watch_batch44_v1_quick20_local.sh

set -euo pipefail

CANONICAL_PROJECT_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
PROJECT_ROOT="${PROJECT_ROOT:-$CANONICAL_PROJECT_ROOT}"
TEST_MODE="${BATCH44_LOCAL_QUICK20_WATCHER_TEST_MODE:-0}"
STAMP="20260713"
MODE="${MODE:-once}"
ACTION="${ACTION:-plan}"
CONFIRM_LOCAL_QUICK20_WATCHER="${CONFIRM_LOCAL_QUICK20_WATCHER:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_SCANS="${MAX_SCANS:-0}"
MIN_CHECKPOINT_AGE_SEC="${MIN_CHECKPOINT_AGE_SEC:-90}"
STOP_WHEN_COMPLETE="${STOP_WHEN_COMPLETE:-1}"

R3_RUN_DIR="${R3_RUN_DIR:-$PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r3_v1_30k}"
R5_RUN_DIR="${R5_RUN_DIR:-$PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r5_v1_30k}"
R3_TRAIN_JOB_ID="job-2b91d332-d500-4279-84f9-0a6a81a376aa"
R5_TRAIN_JOB_ID="job-b8eb2f1f-a3eb-483b-a289-b4cce281525c"
TRAIN_IDENTITY_ROOT="$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713"
TRAIN_PAIR_LEDGER="$TRAIN_IDENTITY_ROOT/submitted_pair.tsv"
STATE_ROOT="${STATE_ROOT:-$PROJECT_ROOT/trainset/local_jobs/ver23_batch44_quick20_scheduler_${STAMP}}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/testset/outputs/ver23_batch44_quick20_${STAMP}}"
LOCAL_RUNNER="${LOCAL_RUNNER:-$PROJECT_ROOT/scripts/004117_run_batch44_v1_quick20_local.sh}"
PROVENANCE_VALIDATOR="${PROVENANCE_VALIDATOR:-$PROJECT_ROOT/scripts/004103_select_batch43_best3.py}"
PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
STEPS="2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 22000 24000 26000 28000 30000"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

case "$TEST_MODE:$CONFIRM_LOCAL_QUICK20_WATCHER:$STOP_WHEN_COMPLETE" in
  [01]:[01]:[01]) ;;
  *) die "boolean flags must be 0 or 1" ;;
esac
case "$MODE" in once|monitor) ;; *) die "MODE must be once or monitor" ;; esac
case "$ACTION" in plan|run) ;; *) die "ACTION must be plan or run" ;; esac
for value in "$POLL_SECONDS" "$MAX_SCANS" "$MIN_CHECKPOINT_AGE_SEC"; do
  case "$value" in
    ''|*[!0-9]*) die "POLL_SECONDS, MAX_SCANS and MIN_CHECKPOINT_AGE_SEC must be non-negative integers" ;;
  esac
done
[ "$POLL_SECONDS" -gt 0 ] || die "POLL_SECONDS must be positive"
[ -x "$PYTHON" ] || die "Python interpreter is not executable: $PYTHON"
[ -s "$PROVENANCE_VALIDATOR" ] || die "missing strict quick20 provenance validator: $PROVENANCE_VALIDATOR"
[ -s "$LOCAL_RUNNER" ] || die "missing local quick20 runner: $LOCAL_RUNNER"
bash -n "$LOCAL_RUNNER"

if [ "$PROJECT_ROOT" != "$CANONICAL_PROJECT_ROOT" ] && [ "$TEST_MODE" != "1" ]; then
  die "non-canonical PROJECT_ROOT is allowed only in watcher test mode"
fi
if [ "$ACTION" = "run" ]; then
  [ "$CONFIRM_LOCAL_QUICK20_WATCHER" = "1" ] || \
    die "ACTION=run requires CONFIRM_LOCAL_QUICK20_WATCHER=1"
  if [ "$TEST_MODE" = "0" ]; then
    [ "$PROJECT_ROOT" = "$CANONICAL_PROJECT_ROOT" ] || die "live run requires canonical PROJECT_ROOT"
    [ "$LOCAL_RUNNER" = "$PROJECT_ROOT/scripts/004117_run_batch44_v1_quick20_local.sh" ] || \
      die "production local runner is hard-locked to canonical 004117"
  fi
fi

# Static no-remote invariant.  The local runner may source 004110 only in its
# inert library mode; neither this watcher nor 004117 may own a remote submit
# ledger or remote-submit lock.
grep -Fq 'BATCH44_QUICK20_LIBRARY_MODE=1' "$LOCAL_RUNNER" || \
  [ "$TEST_MODE" = "1" ] || die "004117 lost its inert common-library gate"
if rg -n 'qzcli|create-job|submitted_jobs\.tsv|\.live_submit\.lock' "$LOCAL_RUNNER" >/dev/null; then
  die "004117 violates the local-only/no-QZ-ledger contract"
fi

mkdir -p "$STATE_ROOT" "$EVAL_ROOT"

audit_training_pair() {
  if [ "$TEST_MODE" = "1" ]; then
    return 0
  fi
  "$PYTHON" - "$TRAIN_PAIR_LEDGER" "$R3_RUN_DIR" "$R5_RUN_DIR" \
    "$R3_TRAIN_JOB_ID" "$R5_TRAIN_JOB_ID" <<'PY'
import csv
import sys
from pathlib import Path

ledger = Path(sys.argv[1])
r3_dir, r5_dir, r3_job, r5_job = sys.argv[2:6]
if not ledger.is_file():
    raise SystemExit(f"missing Batch-44 training pair ledger: {ledger}")
with ledger.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
expected = {
    "ver2_9_5_final_r3_v1_30k": ("r3", r3_job, r3_dir),
    "ver2_9_5_final_r5_v1_30k": ("r5", r5_job, r5_dir),
}
errors = []
if len(rows) != 2:
    errors.append(f"expected two training rows, got {len(rows)}")
seen = set()
for row in rows:
    name = row.get("job_name", "")
    seen.add(name)
    if name not in expected:
        errors.append(f"unexpected training job_name={name!r}")
        continue
    arm, job, out_dir = expected[name]
    if row.get("arm") != arm or row.get("job_id") != job or row.get("out_dir") != out_dir:
        errors.append(f"training provenance drift for {name}: {row}")
if seen != set(expected):
    errors.append(f"training names={sorted(seen)}, expected={sorted(expected)}")
if errors:
    raise SystemExit("Batch-44 local quick20 watcher training audit failed:\n- " + "\n- ".join(errors))
print(f"[batch44-local-quick20-watch] training provenance r3={r3_job} r5={r5_job}")
PY
}

audit_training_pair

checkpoint_probe() {
  local step="$1"
  "$PYTHON" - "$R3_RUN_DIR" "$R5_RUN_DIR" "$PROJECT_ROOT" "$step" \
    "$MIN_CHECKPOINT_AGE_SEC" "$TEST_MODE" <<'PY'
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

r3, r5, project = map(Path, sys.argv[1:4])
step = int(sys.argv[4])
min_age = int(sys.argv[5])
test_mode = sys.argv[6] == "1"
minimum_large = 1 if test_mode else 1_000_000
required = {
    "adapter_model.safetensors": minimum_large,
    "adapter_config.json": 1,
    "README.md": 1,
    "timbre_memory_adapter.pt": minimum_large,
    "timbre_memory_config.json": 1,
}


def equal(got: object, wanted: object) -> bool:
    if isinstance(wanted, float):
        try:
            return math.isclose(float(got), wanted, rel_tol=0.0, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return got == wanted


def probe(arm: str, run_dir: Path, repeat: int) -> tuple[bool, str]:
    checkpoint = run_dir / f"step-{step}"
    if not checkpoint.is_dir():
        return False, f"{arm}:missing"
    newest = 0.0
    for name, minimum in required.items():
        path = checkpoint / name
        if not path.is_file():
            return False, f"{arm}:missing:{name}"
        stat = path.stat()
        newest = max(newest, stat.st_mtime)
        if stat.st_size < minimum:
            return False, f"{arm}:small:{name}:{stat.st_size}"
    age = time.time() - newest
    if age < min_age:
        return False, f"{arm}:settling:{age:.0f}s<{min_age}s"
    try:
        json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
        cfg = json.loads((checkpoint / "timbre_memory_config.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, f"{arm}:invalid_json:{exc}"
    expected_cfg = {
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
    }
    bad_cfg = [key for key, wanted in expected_cfg.items() if not equal(cfg.get(key), wanted)]
    if bad_cfg:
        return False, f"{arm}:config:" + ",".join(bad_cfg)
    args_path = (
        project
        / "trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713"
        / arm
        / "train_args_dry_run_core.json"
    )
    if not args_path.is_file():
        return False, f"{arm}:missing:train_args_dry_run_core.json"
    if not test_mode:
        wanted_sha = {
            "r3": "3637b90fe14bbda7c4915362d5bd726b072833f07853df4fc555fde0d2c32d7b",
            "r5": "161f05b766454ce7d3e38af1772a1126ec612eaeb0682678bd6a46ca1d62daff",
        }[arm]
        with args_path.open("rb") as handle:
            got_sha = hashlib.file_digest(handle, "sha256").hexdigest()
        if got_sha != wanted_sha:
            return False, f"{arm}:identity_sha256_mismatch:{got_sha}"
    try:
        args = json.loads(args_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, f"{arm}:invalid_args:{exc}"
    no_text = project / "trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/no_text.train.jsonl"
    text = project / "trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/text.train.jsonl"
    expected_args = {
        "OUT_DIR": str(run_dir),
        "TRAIN_JSONL_SPEC": f"{no_text}::repeat=1,{text}::repeat={repeat}",
        "TEXT_REPEAT": str(repeat),
        "MAX_TRAIN_STEPS": "30000",
        "SAVE_STEPS": "2000",
        "EVAL_STEPS": "2000",
        "LEARNING_RATE": "1e-5",
        "LR_SCHEDULER_TYPE": "constant_with_warmup",
        "WARMUP_RATIO": "0.03",
    }
    bad_args = [key for key, wanted in expected_args.items() if args.get(key) != wanted]
    if bad_args:
        return False, f"{arm}:identity:" + ",".join(bad_args)
    return True, f"{arm}:ready:age={age:.0f}s"


r3_ok, r3_reason = probe("r3", r3, 3)
r5_ok, r5_reason = probe("r5", r5, 5)
print("ready" if r3_ok and r5_ok else "waiting")
print(r3_reason)
print(r5_reason)
PY
}

scan_completions() {
  "$PYTHON" - "$PROJECT_ROOT" "$STATE_ROOT" "$EVAL_ROOT" "$STAMP" \
    "$STEPS" "$PROVENANCE_VALIDATOR" "$R3_TRAIN_JOB_ID" "$R5_TRAIN_JOB_ID" <<'PY'
from __future__ import annotations

import csv
import datetime as dt
import importlib.util
import json
import os
import sys
from pathlib import Path

project = Path(sys.argv[1]).resolve()
state_root = Path(sys.argv[2]).resolve()
eval_root = Path(sys.argv[3]).resolve()
stamp = sys.argv[4]
steps = [int(value) for value in sys.argv[5].split()]
validator_path = Path(sys.argv[6]).resolve()
r3_job, r5_job = sys.argv[7:9]

spec = importlib.util.spec_from_file_location("batch44_local_watcher_validator", validator_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot import validator: {validator_path}")
validator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = validator
spec.loader.exec_module(validator)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


status_rows: list[dict[str, object]] = []
all_metrics: list[dict[str, object]] = []
negative: list[dict[str, object]] = []
for step in steps:
    qz_record, local_record = validator.quick20_record_paths(project, step, stamp)
    qz_exists = os.path.lexists(qz_record)
    local_exists = os.path.lexists(local_record)
    if qz_exists and local_exists:
        raise SystemExit(
            f"step-{step} has conflicting QZ/local record roots: {qz_record} and {local_record}"
        )
    if step >= 10000 and qz_exists:
        raise SystemExit(
            f"step-{step} evaluation must be local_jobs; unexpected legacy/QZ record: {qz_record}"
        )
    record = qz_record if qz_exists else local_record if local_exists else (
        qz_record if step <= 8000 else local_record
    )
    complete_parts = [record / "COMPLETED.json", record / "complete.marker", record / "metrics.json", record / "metrics.tsv", record / "metrics.md"]
    any_complete_part = any(os.path.lexists(path) for path in complete_parts)
    all_complete_parts = all(path.is_file() and path.stat().st_size > 0 for path in complete_parts)
    local_lock = record / ".local_quick20.lock"
    forbidden_local = [record / "submitted_jobs.tsv", record / ".live_submit.lock"]

    if all_complete_parts:
        metrics_json = record / "metrics.json"
        rows = validator.load_metrics(metrics_json, project_root=project, step=step)
        provenance = validator.audit_quick20_provenance(
            metrics_json, project_root=project, step=step
        )
        backend = provenance["backend"]
        if step <= 6000 and backend != "qz":
            raise SystemExit(f"step-{step} must retain legacy QZ completion provenance")
        if step >= 8000 and backend != "local":
            raise SystemExit(f"step-{step} must use local completion provenance")
        if step == 8000 and record.resolve() != qz_record.resolve():
            raise SystemExit("step-8000 migration completion must retain its legacy qz_jobs record root")
        if step >= 10000 and record.resolve() != local_record.resolve():
            raise SystemExit(f"step-{step} local completion is outside local_jobs")
        status = "complete"
        detail = f"backend={backend}"
        for arm in ("r3", "r5"):
            for mode in ("no_text", "text"):
                row = dict(rows[(arm, mode)])
                all_metrics.append(row)
                if mode == "no_text" and float(row["margin"]) < 0.0:
                    negative.append({
                        "step": step,
                        "arm": arm,
                        "margin": float(row["margin"]),
                        "sim_ref": float(row["sim_ref"]),
                        "sim_src": float(row["sim_src"]),
                        "cer": float(row["cer"]),
                        "record_root": str(record.resolve()),
                        "training_job_id": r3_job if arm == "r3" else r5_job,
                    })
    elif local_lock.is_dir():
        if any(os.path.lexists(path) for path in forbidden_local):
            raise SystemExit(f"local step-{step} contains a QZ ledger/lock: {record}")
        runtime = record / "LOCAL_RUNTIME.json"
        runner = record / "004117_run_batch44_v1_quick20_local.frozen.sh"
        if not runtime.is_file() or not runner.is_file():
            raise SystemExit(f"local step-{step} lock exists without runtime/runner provenance")
        status = "running_local"
        detail = "004117 local lock active"
        backend = "local"
    elif any_complete_part:
        raise SystemExit(f"step-{step} has partial completion evidence requiring manual audit: {record}")
    elif os.path.lexists(record):
        entries = list(record.iterdir()) if record.is_dir() else []
        if entries:
            raise SystemExit(f"step-{step} has unbound partial record requiring manual audit: {record}")
        status = "waiting_checkpoint"
        detail = "empty record root"
        backend = "pending"
    else:
        status = "waiting_checkpoint"
        detail = "record not created"
        backend = "pending"

    status_rows.append({
        "step": step,
        "status": status,
        "backend": backend,
        "detail": detail,
        "record_root": str(record),
    })

atomic_json(state_root / "scan_latest.json", status_rows)
with (state_root / "scan_latest.tsv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(status_rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(status_rows)

if all_metrics:
    fields = list(all_metrics[0])
    with (eval_root / "metrics_all.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_metrics)
    atomic_json(eval_root / "metrics_all.json", all_metrics)

if negative:
    generated = dt.datetime.now(dt.timezone.utc).isoformat()
    alert = {
        "schema": "batch44_v1_quick20_negative_margin_alert_v2",
        "status": "alert",
        "generated_utc": generated,
        "trigger": "any strictly validated no_text quick20 WavLM margin < 0",
        "scheduler_action": "stop scheduling further local quick20 evaluations",
        "training_action": "report only; watcher never stops or mutates training jobs",
        "training_jobs": {"r3": r3_job, "r5": r5_job},
        "alerts": negative,
    }
    recommendation = (
        "# Batch-44 stop recommendation\n\n"
        "A strictly validated no_text quick20 has WavLM `sim(ref)-sim(src) < 0`.\n"
        "The local quick20 watcher stopped future evaluation scheduling.\n"
        "It did not stop or modify either training job.\n"
    )
    for root in (state_root, eval_root):
        atomic_json(root / "ALERT_NEGATIVE_NO_TEXT_MARGIN.json", alert)
        (root / "STOP_RECOMMENDATION.md").write_text(recommendation, encoding="utf-8")

summary = {
    "complete": sum(row["status"] == "complete" for row in status_rows),
    "total": len(status_rows),
    "alert": bool(negative),
    "first_incomplete": next((row for row in status_rows if row["status"] != "complete"), None),
}
atomic_json(state_root / "scan_summary.json", summary)
print(
    f"[batch44-local-quick20-audit] complete={summary['complete']}/{summary['total']} "
    f"alert={summary['alert']}"
)
PY
}

first_incomplete_step() {
  "$PYTHON" - "$STATE_ROOT/scan_summary.json" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
row = payload.get("first_incomplete")
if isinstance(row, dict):
    print(row["step"])
    print(row["status"])
PY
}

summary_flag() {
  local field="$1"
  "$PYTHON" - "$STATE_ROOT/scan_summary.json" "$field" <<'PY'
import json
import sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))[sys.argv[2]]
print("1" if value is True else "0" if value is False else value)
PY
}

run_scan() {
  local scan="$1" first step status probe ready detail_r3 detail_r5 record_root
  scan_completions
  cp "$STATE_ROOT/scan_latest.tsv" "$STATE_ROOT/scan_${scan}.tsv"
  echo "[batch44-local-quick20-watch] scan=$scan action=$ACTION utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  awk -F '\t' 'NR == 1 || $2 != "waiting_checkpoint" {print}' "$STATE_ROOT/scan_latest.tsv"

  if [ "$(summary_flag alert)" = "1" ]; then
    echo "[batch44-local-quick20-watch] ALERT: margin<0; no later evaluation scheduled; training untouched"
    return 0
  fi
  first="$(first_incomplete_step)"
  step="$(printf '%s\n' "$first" | sed -n '1p')"
  status="$(printf '%s\n' "$first" | sed -n '2p')"
  if [ -z "$step" ]; then
    echo "[batch44-local-quick20-watch] all 15 checkpoints complete"
    return 0
  fi
  if [ "$status" = "running_local" ]; then
    echo "[batch44-local-quick20-watch] step-$step local evaluation is still running; strict serialization holds"
    return 0
  fi
  [ "$status" = "waiting_checkpoint" ] || die "unexpected first-incomplete status=$status step=$step"
  probe="$(checkpoint_probe "$step")"
  ready="$(printf '%s\n' "$probe" | sed -n '1p')"
  detail_r3="$(printf '%s\n' "$probe" | sed -n '2p')"
  detail_r5="$(printf '%s\n' "$probe" | sed -n '3p')"
  if [ "$ready" != "ready" ]; then
    echo "[batch44-local-quick20-watch] waiting step=$step $detail_r3 $detail_r5"
    return 0
  fi
  record_root="$PROJECT_ROOT/trainset/local_jobs/ver23_batch44_quick20_step${step}_${STAMP}"
  if [ "$ACTION" = "plan" ]; then
    echo "[batch44-local-quick20-watch] next_ready_step=$step backend=local record=$record_root (plan only)"
    return 0
  fi
  [ "$step" -ge 10000 ] || die "live local watcher may only create step-10000+ records"
  echo "[batch44-local-quick20-watch] RUN step=$step backend=local record=$record_root"
  STEP="$step" DRY_RUN=0 CONFIRM_LOCAL_QUICK20=1 \
    PROJECT_ROOT="$PROJECT_ROOT" R3_RUN_DIR="$R3_RUN_DIR" R5_RUN_DIR="$R5_RUN_DIR" \
    RECORD_ROOT="$record_root" EVAL_ROOT="$EVAL_ROOT" \
    bash "$LOCAL_RUNNER"
  if [ "$TEST_MODE" = "0" ]; then
    # A zero exit is not sufficient evidence.  Re-run the dual-backend audit
    # and require this exact step to be a fully SHA-bound local completion
    # before a future monitor scan may advance.
    scan_completions
    post_status="$($PYTHON - "$STATE_ROOT/scan_latest.json" "$step" <<'PY'
import json
import sys
from pathlib import Path
rows = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
step = int(sys.argv[2])
matches = [row for row in rows if int(row["step"]) == step]
if len(matches) != 1:
    raise SystemExit(f"post-run status identity drift for step-{step}: {matches}")
print(matches[0]["status"])
PY
)"
    [ "$post_status" = "complete" ] || \
      die "004117 returned zero without a strictly validated completion at step-$step"
  fi
  echo "[batch44-local-quick20-watch] local runner returned successfully step=$step"
}

LOCK_DIR="$STATE_ROOT/.watch.lock"
PID_FILE="$STATE_ROOT/monitor.pid"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  die "another local quick20 watcher appears active: $LOCK_DIR"
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
  if [ "$(summary_flag alert)" = "1" ]; then
    break
  fi
  if [ "$STOP_WHEN_COMPLETE" = "1" ] && [ "$(summary_flag complete)" = "15" ]; then
    break
  fi
  if [ "$MODE" = "once" ]; then
    break
  fi
  if [ "$MAX_SCANS" -gt 0 ] && [ "$scan" -ge "$MAX_SCANS" ]; then
    echo "[batch44-local-quick20-watch] reached MAX_SCANS=$MAX_SCANS"
    break
  fi
  sleep "$POLL_SECONDS"
done

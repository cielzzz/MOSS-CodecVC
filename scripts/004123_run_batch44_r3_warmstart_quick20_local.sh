#!/usr/bin/env bash
# Run one r3 warm-start quick20 on the local two-RTX-4090 host.
#
# Scientific steps remain 12k..30k, while the continuation checkpoint names
# restart at local step 2k..20k.  This runner binds both identities and never
# invokes remote submission tooling or mutates the training job.

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
CANONICAL_PROJECT_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
PROJECT_ROOT="${PROJECT_ROOT:-$CANONICAL_PROJECT_ROOT}"
CODE_ROOT="${CODE_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch37_eval_20260711_1092820}"
STAMP="20260713"

EFFECTIVE_STEP="${EFFECTIVE_STEP:-12000}"
CONTINUATION_LOCAL_STEP="${CONTINUATION_LOCAL_STEP:-2000}"
CONTINUATION_RUN_DIR="${CONTINUATION_RUN_DIR:-$PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r3_v1_warmstart10k_to30k}"
CHECKPOINT="${CHECKPOINT:-$CONTINUATION_RUN_DIR/step-$CONTINUATION_LOCAL_STEP}"
TRAIN_JOB_ID="${TRAIN_JOB_ID:-job-165f3b1d-8c7d-47d8-8882-bdb86ef642ab}"
WARM_START_CONTRACT="${WARM_START_CONTRACT:-$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_r3_v1_warmstart10k_to30k_${STAMP}/warm_start_contract.json}"
RECORD_ROOT="${RECORD_ROOT:-$PROJECT_ROOT/trainset/local_jobs/ver23_batch44_r3_warmstart_quick20_step${EFFECTIVE_STEP}_${STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/testset/outputs/ver23_batch44_r3_warmstart_quick20_${STAMP}}"

DRY_RUN="${DRY_RUN:-1}"
CONFIRM_RUN="${CONFIRM_RUN:-0}"
TEST_MODE="${BATCH44_R3_WARMSTART_QUICK20_TEST_MODE:-0}"
MIN_CHECKPOINT_AGE_SEC="${MIN_CHECKPOINT_AGE_SEC:-90}"
MAX_INITIAL_GPU_MEMORY_MIB="${MAX_INITIAL_GPU_MEMORY_MIB:-2048}"

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
ASR_PYTHON="${ASR_PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
RUNNER_SOURCE="$SCRIPT_DIR/004123_run_batch44_r3_warmstart_quick20_local.sh"
COMMON_SOURCE="$SCRIPT_DIR/004110_submit_batch44_v1_quick20_qz.sh"
HELPER_SOURCE="$SCRIPT_DIR/batch44_r3_warmstart_quick20_completion.py"
VALIDATOR_SOURCE="$SCRIPT_DIR/batch44_r3_warmstart_quick20_validator.py"

LOCAL_LOCK="$RECORD_ROOT/.local_quick20.lock"
RUNTIME_MANIFEST="$RECORD_ROOT/LOCAL_RUNTIME.json"
FROZEN_RUNNER="$RECORD_ROOT/004123_run_batch44_r3_warmstart_quick20_local.frozen.sh"
FROZEN_COMMON="$RECORD_ROOT/004110_batch44_quick20_common.frozen.sh"
FROZEN_HELPER="$RECORD_ROOT/batch44_r3_warmstart_quick20_completion.frozen.py"
FROZEN_VALIDATOR="$RECORD_ROOT/batch44_r3_warmstart_quick20_validator.frozen.py"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

case "$DRY_RUN:$CONFIRM_RUN:$TEST_MODE" in
  [01]:[01]:[01]) ;;
  *) die "DRY_RUN, CONFIRM_RUN and BATCH44_R3_WARMSTART_QUICK20_TEST_MODE must be 0 or 1" ;;
esac
case "$EFFECTIVE_STEP:$CONTINUATION_LOCAL_STEP" in
  12000:2000|14000:4000|16000:6000|18000:8000|20000:10000|22000:12000|24000:14000|26000:16000|28000:18000|30000:20000) ;;
  *) die "invalid effective/local mapping: $EFFECTIVE_STEP/$CONTINUATION_LOCAL_STEP" ;;
esac
for value in "$MIN_CHECKPOINT_AGE_SEC" "$MAX_INITIAL_GPU_MEMORY_MIB"; do
  case "$value" in ''|*[!0-9]*) die "age/memory limits must be non-negative integers" ;; esac
done
[ "$MAX_INITIAL_GPU_MEMORY_MIB" -gt 0 ] || die "MAX_INITIAL_GPU_MEMORY_MIB must be positive"
[ -x "$PYTHON" ] || die "Python is not executable: $PYTHON"
[ -x "$ASR_PYTHON" ] || die "ASR Python is not executable: $ASR_PYTHON"
for path in "$RUNNER_SOURCE" "$COMMON_SOURCE" "$HELPER_SOURCE" "$VALIDATOR_SOURCE"; do
  [ -s "$path" ] || die "missing runner dependency: $path"
done
bash -n "$RUNNER_SOURCE"
"$PYTHON" -m py_compile "$HELPER_SOURCE" "$VALIDATOR_SOURCE"

if [ "$PROJECT_ROOT" != "$CANONICAL_PROJECT_ROOT" ] && [ "$TEST_MODE" != "1" ]; then
  die "non-canonical PROJECT_ROOT is test-only"
fi
if [ "$DRY_RUN" = "0" ]; then
  [ "$CONFIRM_RUN" = "1" ] || die "DRY_RUN=0 requires CONFIRM_RUN=1"
  [ "$TEST_MODE" = "0" ] || die "test mode may not start inference"
  [ "$PROJECT_ROOT" = "$CANONICAL_PROJECT_ROOT" ] || die "live run requires canonical PROJECT_ROOT"
  [ "$RUNNER_SOURCE" = "$PROJECT_ROOT/scripts/004123_run_batch44_r3_warmstart_quick20_local.sh" ] || die "live runner path drift"
  [ "$CHECKPOINT" = "$CONTINUATION_RUN_DIR/step-$CONTINUATION_LOCAL_STEP" ] || die "live checkpoint path drift"
  [ "$RECORD_ROOT" = "$PROJECT_ROOT/trainset/local_jobs/ver23_batch44_r3_warmstart_quick20_step${EFFECTIVE_STEP}_${STAMP}" ] || die "live record path drift"
  [ "$OUTPUT_ROOT" = "$PROJECT_ROOT/testset/outputs/ver23_batch44_r3_warmstart_quick20_${STAMP}" ] || die "live output path drift"
fi

# Source only the audited fixed-input/evaluation functions.  Library mode
# returns before every platform submission branch.
STEP="$EFFECTIVE_STEP"
EVAL_ROOT="$OUTPUT_ROOT"
BATCH44_QUICK20_TEST_MODE="$TEST_MODE"
original_dry_run="$DRY_RUN"
DRY_RUN=1
BATCH44_QUICK20_LIBRARY_MODE=1
# shellcheck source=004110_submit_batch44_v1_quick20_qz.sh
source "$COMMON_SOURCE"
unset BATCH44_QUICK20_LIBRARY_MODE
DRY_RUN="$original_dry_run"

# Replace paired checkpoint routing with the single continuation checkpoint.
arm_run_dir() {
  [ "$1" = "r3" ] || die "r3-only quick20 does not support arm=$1"
  printf '%s\n' "$CONTINUATION_RUN_DIR"
}
arm_checkpoint() {
  [ "$1" = "r3" ] || die "r3-only quick20 does not support arm=$1"
  printf '%s\n' "$CHECKPOINT"
}
arm_label() {
  [ "$1" = "r3" ] || die "r3-only quick20 does not support arm=$1"
  printf '%s\n' "ver2_9_5_final_r3"
}

audit_args=(
  audit-binding
  --project-root "$PROJECT_ROOT"
  --effective-step "$EFFECTIVE_STEP"
  --continuation-local-step "$CONTINUATION_LOCAL_STEP"
  --checkpoint "$CHECKPOINT"
  --train-job-id "$TRAIN_JOB_ID"
  --warm-start-contract "$WARM_START_CONTRACT"
  --min-checkpoint-age-sec "$MIN_CHECKPOINT_AGE_SEC"
)
[ "$TEST_MODE" = "1" ] && audit_args+=(--test-mode)
"$PYTHON" "$HELPER_SOURCE" "${audit_args[@]}"
audit_code_root

echo "=========================================="
echo "Batch-44 r3 warm-start local quick20"
echo "  DRY_RUN=$DRY_RUN"
echo "  EFFECTIVE_STEP=$EFFECTIVE_STEP"
echo "  CONTINUATION_LOCAL_STEP=$CONTINUATION_LOCAL_STEP"
echo "  CHECKPOINT=$CHECKPOINT"
echo "  TRAIN_JOB_ID=$TRAIN_JOB_ID"
echo "  WARM_START_CONTRACT=$WARM_START_CONTRACT"
echo "  RECORD_ROOT=$RECORD_ROOT"
echo "  OUTPUT_ROOT=$OUTPUT_ROOT"
echo "  LANES=r3/no_text20 -> r3/text20 (sequential on GPUs 0,1)"
echo "=========================================="

if [ "$DRY_RUN" = "1" ]; then
  # Verify the exact fixed sets without leaving a record behind.
  temporary_root=$(mktemp -d)
  original_record_root="$RECORD_ROOT"
  original_text20="$TEXT20_JSONL"
  RECORD_ROOT="$temporary_root"
  TEXT20_JSONL="$temporary_root/ver23_batch44_text_quick20_8cell_20260713.jsonl"
  prepare_and_validate_quick_sets
  RECORD_ROOT="$original_record_root"
  TEXT20_JSONL="$original_text20"
  rm -rf "$temporary_root"
  echo "[batch44-r3-warmstart-local] dry-run PASS; no GPU work started"
  exit 0
fi

if [ -e "$RECORD_ROOT" ] || [ -L "$RECORD_ROOT" ]; then
  die "record root already exists; manual audit required: $RECORD_ROOT"
fi
for mode in no_text text; do
  identity="$(run_id_for r3 "$mode")"
  if [ -e "$OUTPUT_ROOT/$identity" ] || [ -L "$OUTPUT_ROOT/$identity" ]; then
    die "evaluation output already exists; manual audit required: $OUTPUT_ROOT/$identity"
  fi
done

mkdir -p "$RECORD_ROOT" "$OUTPUT_ROOT"
mkdir "$LOCAL_LOCK" || die "failed to acquire local quick20 lock: $LOCAL_LOCK"
"$PYTHON" - "$LOCAL_LOCK/owner.json" "$EFFECTIVE_STEP" "$CONTINUATION_LOCAL_STEP" <<'PY'
import datetime as dt
import json
import os
import socket
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "pid": os.getppid(),
    "hostname": socket.gethostname(),
    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "effective_step": int(sys.argv[2]),
    "continuation_local_step": int(sys.argv[3]),
    "policy": "persistent lock; inspect artifacts/processes before recovery",
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

install -m 0555 "$RUNNER_SOURCE" "$FROZEN_RUNNER"
install -m 0444 "$COMMON_SOURCE" "$FROZEN_COMMON"
install -m 0444 "$HELPER_SOURCE" "$FROZEN_HELPER"
install -m 0444 "$VALIDATOR_SOURCE" "$FROZEN_VALIDATOR"
prepare_and_validate_quick_sets

"$PYTHON" "$FROZEN_HELPER" capture-runtime \
  --output "$RUNTIME_MANIFEST" \
  --runner "$FROZEN_RUNNER" \
  --common-library "$FROZEN_COMMON" \
  --completion-helper "$FROZEN_HELPER" \
  --validator "$FROZEN_VALIDATOR" \
  --effective-step "$EFFECTIVE_STEP" \
  --continuation-local-step "$CONTINUATION_LOCAL_STEP" \
  --checkpoint "$CHECKPOINT" \
  --train-job-id "$TRAIN_JOB_ID" \
  --warm-start-contract "$WARM_START_CONTRACT" \
  --max-initial-memory-mib "$MAX_INITIAL_GPU_MEMORY_MIB"

exec >> "$RECORD_ROOT/run.local.log" 2>&1
echo "[batch44-r3-warmstart-local] start=$(date -u +%Y-%m-%dT%H:%M:%SZ) effective=$EFFECTIVE_STEP local=$CONTINUATION_LOCAL_STEP"
nvidia-smi

echo "[batch44-r3-warmstart-local] lane=1/2 mode=no_text start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_eval r3 no_text 0,1
echo "[batch44-r3-warmstart-local] lane=1/2 mode=no_text complete=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[batch44-r3-warmstart-local] lane=2/2 mode=text start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_eval r3 text 0,1
echo "[batch44-r3-warmstart-local] lane=2/2 mode=text complete=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"$PYTHON" "$FROZEN_HELPER" collect-metrics \
  --record-root "$RECORD_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --effective-step "$EFFECTIVE_STEP" \
  --continuation-local-step "$CONTINUATION_LOCAL_STEP" \
  --checkpoint "$CHECKPOINT" \
  --train-job-id "$TRAIN_JOB_ID" \
  --warm-start-contract "$WARM_START_CONTRACT"

"$PYTHON" "$FROZEN_HELPER" finalize \
  --record-root "$RECORD_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --project-root "$PROJECT_ROOT" \
  --code-root "$CODE_ROOT" \
  --effective-step "$EFFECTIVE_STEP" \
  --continuation-local-step "$CONTINUATION_LOCAL_STEP" \
  --checkpoint "$CHECKPOINT" \
  --train-job-id "$TRAIN_JOB_ID" \
  --warm-start-contract "$WARM_START_CONTRACT" \
  --no-text20 "$NO_TEXT20_JSONL" --no-text20-sha256 "$NO_TEXT20_SHA256" \
  --text-source "$TEXT_SOURCE_JSONL" --text-source-sha256 "$TEXT_SOURCE_SHA256" \
  --text20 "$TEXT20_JSONL" --text20-sha256 "$TEXT20_SHA256" \
  --runner "$FROZEN_RUNNER" \
  --common-library "$FROZEN_COMMON" \
  --completion-helper "$FROZEN_HELPER" \
  --validator "$FROZEN_VALIDATOR" \
  --runtime-manifest "$RUNTIME_MANIFEST"

PYTHONPATH="$RECORD_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON" "$FROZEN_VALIDATOR" \
  --record-root "$RECORD_ROOT" \
  --expected-effective-step "$EFFECTIVE_STEP" \
  --expected-continuation-local-step "$CONTINUATION_LOCAL_STEP" \
  --expected-train-job-id "$TRAIN_JOB_ID"

rm -rf "$LOCAL_LOCK"
echo "[batch44-r3-warmstart-local] complete effective=$EFFECTIVE_STEP metrics=$RECORD_ROOT/metrics.tsv"

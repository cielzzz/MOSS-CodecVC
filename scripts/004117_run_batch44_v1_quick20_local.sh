#!/usr/bin/env bash
# Run one Batch-44 r3/r5 quick20 checkpoint on the local dual RTX 4090 host.
#
# The scientific protocol is inherited from 004110/004039: the same fixed
# no_text20/text20 cases, decoding parameters, ASR, WavLM speaker scorer,
# ref-content scorer and metric collector are used.  With only two GPUs, the
# four lanes run strictly sequentially; each lane uses both GPUs as two shards.
#
# Safe audit (no inference):
#   STEP=8000 DRY_RUN=1 bash scripts/004117_run_batch44_v1_quick20_local.sh
# Live local run:
#   STEP=8000 DRY_RUN=0 CONFIRM_LOCAL_QUICK20=1 \
#     bash scripts/004117_run_batch44_v1_quick20_local.sh

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
LOCAL_RUNNER_SOURCE="$SCRIPT_DIR/004117_run_batch44_v1_quick20_local.sh"
COMMON_LIBRARY_SOURCE="$SCRIPT_DIR/004110_submit_batch44_v1_quick20_qz.sh"
COMPLETION_HELPER="$SCRIPT_DIR/batch44_quick20_local_completion.py"
[ -s "$LOCAL_RUNNER_SOURCE" ] || { echo "ERROR: missing local runner: $LOCAL_RUNNER_SOURCE" >&2; exit 1; }
[ -s "$COMMON_LIBRARY_SOURCE" ] || { echo "ERROR: missing common protocol library: $COMMON_LIBRARY_SOURCE" >&2; exit 1; }
LOCAL_RUNNER_SHA_AT_START="$(sha256sum "$LOCAL_RUNNER_SOURCE" | awk '{print $1}')"
COMMON_LIBRARY_SHA_AT_SOURCE="$(sha256sum "$COMMON_LIBRARY_SOURCE" | awk '{print $1}')"

# Source only the audited protocol functions.  Force the sourced wrapper into
# its inert library branch; this local runner owns all execution behavior.
LOCAL_DRY_RUN="${DRY_RUN:-1}"
DRY_RUN=1
BATCH44_QUICK20_LIBRARY_MODE=1
# shellcheck source=004110_submit_batch44_v1_quick20_qz.sh
source "$COMMON_LIBRARY_SOURCE"
unset BATCH44_QUICK20_LIBRARY_MODE
DRY_RUN="$LOCAL_DRY_RUN"

CONFIRM_LOCAL_QUICK20="${CONFIRM_LOCAL_QUICK20:-0}"
MAX_INITIAL_GPU_MEMORY_MIB="${MAX_INITIAL_GPU_MEMORY_MIB:-2048}"
LOCAL_TEST_MODE="${BATCH44_LOCAL_QUICK20_TEST_MODE:-0}"
LOCAL_RUNTIME_MANIFEST="$RECORD_ROOT/LOCAL_RUNTIME.json"
LOCAL_LOCK="$RECORD_ROOT/.local_quick20.lock"
FROZEN_LOCAL_RUNNER="$RECORD_ROOT/004117_run_batch44_v1_quick20_local.frozen.sh"
FROZEN_COMMON_LIBRARY="$RECORD_ROOT/004110_batch44_quick20_common.frozen.sh"
FROZEN_COMPLETION_HELPER="$RECORD_ROOT/batch44_quick20_local_completion.frozen.py"

case "$DRY_RUN:$CONFIRM_LOCAL_QUICK20:$LOCAL_TEST_MODE" in
  [01]:[01]:[01]) ;;
  *) die "DRY_RUN, CONFIRM_LOCAL_QUICK20 and BATCH44_LOCAL_QUICK20_TEST_MODE must be 0 or 1" ;;
esac
if [ "$DRY_RUN" = "0" ] && [ "$CONFIRM_LOCAL_QUICK20" != "1" ]; then
  die "live local execution requires CONFIRM_LOCAL_QUICK20=1"
fi
if [ "$LOCAL_TEST_MODE" = "1" ] && [ "$DRY_RUN" = "0" ]; then
  die "test mode may not start a live local evaluation"
fi
case "$MAX_INITIAL_GPU_MEMORY_MIB" in
  ''|*[!0-9]*) die "MAX_INITIAL_GPU_MEMORY_MIB must be a positive integer" ;;
esac
if [ "$MAX_INITIAL_GPU_MEMORY_MIB" -le 0 ]; then
  die "MAX_INITIAL_GPU_MEMORY_MIB must be a positive integer"
fi

ensure_local_evaluation_is_new() {
  local path name arm mode run_id output_dir
  if [ -L "$RECORD_ROOT" ]; then
    die "local record root may not be a symlink: $RECORD_ROOT"
  fi
  if [ -d "$RECORD_ROOT" ]; then
    while IFS= read -r -d '' path; do
      name="$(basename "$path")"
      case "$name" in
        ver23_batch44_text_quick20_8cell_20260713.jsonl) ;;
        *) die "existing/partial local quick20 artifact requires manual audit: $path" ;;
      esac
    done < <(find "$RECORD_ROOT" -mindepth 1 -maxdepth 1 -print0)
  fi
  for arm in r3 r5; do
    for mode in no_text text; do
      run_id="$(run_id_for "$arm" "$mode")"
      output_dir="$EVAL_ROOT/$run_id"
      if [ -e "$output_dir" ] || [ -L "$output_dir" ]; then
        die "existing/partial local eval output requires manual audit: $output_dir"
      fi
    done
  done
}

print_plan() {
  echo "=========================================="
  echo "Local Batch-44 paired quick20"
  echo "  BACKEND=local"
  echo "  STEP=$STEP"
  echo "  HOST=$(hostname)"
  echo "  GPU_CONTRACT=2x NVIDIA GeForce RTX 4090, indices 0,1"
  echo "  SCHEDULING=sequential lanes, each lane uses GPUs 0,1"
  echo "  LANE_1=r3 no_text"
  echo "  LANE_2=r3 text"
  echo "  LANE_3=r5 no_text"
  echo "  LANE_4=r5 text"
  echo "  R3_CHECKPOINT=$(arm_checkpoint r3)"
  echo "  R5_CHECKPOINT=$(arm_checkpoint r5)"
  echo "  RECORD_ROOT=$RECORD_ROOT"
  echo "  EVAL_ROOT=$EVAL_ROOT"
  echo "  DRY_RUN=$DRY_RUN"
  echo "=========================================="
}

run_lane() {
  local lane="$1" arm="$2" mode="$3"
  echo "[batch44-local] lane=$lane/4 arm=$arm mode=$mode start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  run_eval "$arm" "$mode" 0,1
  echo "[batch44-local] lane=$lane/4 arm=$arm mode=$mode complete=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

[ -x "$PYTHON" ] || die "Python interpreter is not executable: $PYTHON"
[ -x "$ASR_PYTHON" ] || die "ASR Python interpreter is not executable: $ASR_PYTHON"
[ -s "$COMPLETION_HELPER" ] || die "missing local completion helper: $COMPLETION_HELPER"
[ -s "$LOCAL_RUNNER_SOURCE" ] || die "missing local runner: $LOCAL_RUNNER_SOURCE"
if [ "$(sha256sum "$LOCAL_RUNNER_SOURCE" | awk '{print $1}')" != "$LOCAL_RUNNER_SHA_AT_START" ]; then
  die "local runner changed during preflight"
fi
if [ "$(sha256sum "$COMMON_LIBRARY_SOURCE" | awk '{print $1}')" != "$COMMON_LIBRARY_SHA_AT_SOURCE" ]; then
  die "sourced quick20 protocol changed during preflight"
fi

audit_code_root
audit_training_pair
prepare_and_validate_quick_sets
validate_checkpoint r3
validate_checkpoint r5
ensure_local_evaluation_is_new
print_plan

if [ "$DRY_RUN" = "1" ]; then
  # Runtime identity is still checked in a dry run, but no execution record,
  # lock, metric, output directory or completion marker is created.
  runtime_tmp="$(mktemp --tmpdir batch44-local-runtime.XXXXXX.json)"
  "$PYTHON" "$COMPLETION_HELPER" capture-runtime \
    --output "$runtime_tmp" \
    --runner "$LOCAL_RUNNER_SOURCE" \
    --common-library "$COMMON_LIBRARY_SOURCE" \
    --completion-helper "$COMPLETION_HELPER" \
    --step "$STEP" \
    --r3-checkpoint "$(arm_checkpoint r3)" \
    --r5-checkpoint "$(arm_checkpoint r5)" \
    --max-initial-memory-mib "$MAX_INITIAL_GPU_MEMORY_MIB"
  rm -f "$runtime_tmp"
  echo "[batch44-local] dry-run passed; no inference started"
  exit 0
fi

mkdir -p "$RECORD_ROOT" "$EVAL_ROOT"
if ! mkdir "$LOCAL_LOCK" 2>/dev/null; then
  die "local quick20 lock already exists; inspect partial state: $LOCAL_LOCK"
fi
install -m 0555 "$LOCAL_RUNNER_SOURCE" "$FROZEN_LOCAL_RUNNER"
install -m 0444 "$COMMON_LIBRARY_SOURCE" "$FROZEN_COMMON_LIBRARY"
install -m 0444 "$COMPLETION_HELPER" "$FROZEN_COMPLETION_HELPER"
[ "$(sha256sum "$FROZEN_LOCAL_RUNNER" | awk '{print $1}')" = "$LOCAL_RUNNER_SHA_AT_START" ] || \
  die "frozen local runner SHA mismatch"
[ "$(sha256sum "$FROZEN_COMMON_LIBRARY" | awk '{print $1}')" = "$COMMON_LIBRARY_SHA_AT_SOURCE" ] || \
  die "frozen common protocol SHA mismatch"

"$PYTHON" "$COMPLETION_HELPER" capture-runtime \
  --output "$LOCAL_RUNTIME_MANIFEST" \
  --runner "$FROZEN_LOCAL_RUNNER" \
  --common-library "$FROZEN_COMMON_LIBRARY" \
  --completion-helper "$FROZEN_COMPLETION_HELPER" \
  --step "$STEP" \
  --r3-checkpoint "$(arm_checkpoint r3)" \
  --r5-checkpoint "$(arm_checkpoint r5)" \
  --max-initial-memory-mib "$MAX_INITIAL_GPU_MEMORY_MIB"

# Do not use a long-lived process-substitution tee here. The lane evaluator
# already owns its own tee process, and nesting the outer tee caused the parent
# shell to exit between sequential lanes after the first scorer completed on
# the local development host. The detached launcher keeps a separate log;
# this file is the canonical in-record execution log.
exec >> "$RECORD_ROOT/run.local.log" 2>&1
echo "[batch44-local] start step=$STEP backend=local host=$(hostname) date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi

# Strictly sequential: at most one evaluation process owns the two GPUs.
run_lane 1 r3 no_text
run_lane 2 r3 text
run_lane 3 r5 no_text
run_lane 4 r5 text

collect_metrics
"$PYTHON" "$FROZEN_COMPLETION_HELPER" finalize \
  --record-root "$RECORD_ROOT" \
  --eval-root "$EVAL_ROOT" \
  --project-root "$PROJECT_ROOT" \
  --code-root "$CODE_ROOT" \
  --step "$STEP" \
  --r3-checkpoint "$(arm_checkpoint r3)" \
  --r5-checkpoint "$(arm_checkpoint r5)" \
  --no-text20 "$NO_TEXT20_JSONL" \
  --no-text20-sha256 "$NO_TEXT20_SHA256" \
  --text-source "$TEXT_SOURCE_JSONL" \
  --text-source-sha256 "$TEXT_SOURCE_SHA256" \
  --text20 "$TEXT20_JSONL" \
  --text20-sha256 "$TEXT20_SHA256" \
  --runner "$FROZEN_LOCAL_RUNNER" \
  --common-library "$FROZEN_COMMON_LIBRARY" \
  --completion-helper "$FROZEN_COMPLETION_HELPER" \
  --runtime-manifest "$LOCAL_RUNTIME_MANIFEST"

rm -rf "$LOCAL_LOCK"
echo "[batch44-local] complete step=$STEP backend=local metrics=$RECORD_ROOT/metrics.tsv"

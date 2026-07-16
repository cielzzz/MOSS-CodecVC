#!/usr/bin/env bash
# Batch-46 fixed no_text quick20 orchestration on the local RTX4090 host.
#
# Default is ACTION=plan: print the complete immutable-marker-driven plan and
# start no GPU work.  To process one ready checkpoint:
#
#   ACTION=once ALLOW_CODECVC_BATCH46_LOCAL_EVAL=1 sh scripts/ver3_1/run_batch46_local_quick20.sh
#
# To wait safely and process step 500..3000 as markers arrive:
#
#   ACTION=watch ALLOW_CODECVC_BATCH46_LOCAL_EVAL=1 sh scripts/ver3_1/run_batch46_local_quick20.sh
#
# The watcher never invokes qzcli and never signals/kills the training job.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -Eeuo pipefail

ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
PY="/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python"
RUNNER="$ROOT/scripts/ver3_1/run_batch46_local_quick20.py"

TASK_NAME="codecVC-ver3-1-batch46-local-quick20-20260716"
CHECKPOINT_ROOT="$ROOT/outputs/ver3_1_batch46_ddlfm_no_text_probe_20260716"
OUTPUT_ROOT="$ROOT/testset/outputs/codecVC-ver3-1-batch46-local-quick20-20260716"
RECORD_ROOT="$ROOT/trainset/local_jobs/codecVC-ver3-1-batch46-local-quick20-20260716"

ACTION="${ACTION:-plan}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_SCANS="${MAX_SCANS:-0}"
ALLOW_RUN="${ALLOW_CODECVC_BATCH46_LOCAL_EVAL:-0}"

case "$ACTION" in plan|once|watch) ;; *) echo "ERROR: ACTION must be plan, once, or watch" >&2; exit 2 ;; esac
[[ "$TASK_NAME" == codecVC-* ]] || { echo "ERROR: task name must start with codecVC-" >&2; exit 2; }
[ -x "$PY" ] || { echo "ERROR: missing Python: $PY" >&2; exit 2; }
[ -f "$RUNNER" ] || { echo "ERROR: missing runner: $RUNNER" >&2; exit 2; }

args=(
  --action "$ACTION"
  --task-name "$TASK_NAME"
  --checkpoint-root "$CHECKPOINT_ROOT"
  --output-root "$OUTPUT_ROOT"
  --record-root "$RECORD_ROOT"
  --poll-seconds "$POLL_SECONDS"
  --max-scans "$MAX_SCANS"
)

if [ "$ACTION" != "plan" ]; then
  [ "$ALLOW_RUN" = "1" ] || {
    echo "ERROR: local GPU evaluation is guarded; set ALLOW_CODECVC_BATCH46_LOCAL_EVAL=1" >&2
    exit 2
  }
  args+=(--allow-run)
fi

cd "$ROOT"
exec "$PY" "$RUNNER" "${args[@]}"

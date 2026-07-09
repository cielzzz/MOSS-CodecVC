#!/usr/bin/env bash
# Run Batch-24 V1-second + v2-data pure architecture quick20 eval on Case A/B/C.

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
ROOT=$(CDPATH= cd "${SCRIPT_DIR}/.." && pwd)

RUN_DIR="${RUN_DIR:-$ROOT/outputs/lora_runs/ver2_9_v1sec_v2_pure_steps3000}" \
RUN_LABEL="${RUN_LABEL:-v1sec_v2_pure}" \
EVAL_ROOT="${EVAL_ROOT:-$ROOT/testset/outputs/ver2_9_v1sec_v2_pure_step_quick_eval}" \
bash "$ROOT/scripts/004066_run_ver2_9_v2full_abc_quick_eval.sh" "$@"

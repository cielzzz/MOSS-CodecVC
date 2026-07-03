#!/usr/bin/env sh
set -eu

# No-arg wrapper for the selected Ver2.5 P0-A text-token-memory run.
# Default usage:
#   sh scripts/004038_run_ver2_5_selected_p0a_eval.sh
#
# To evaluate another checkpoint, edit STEP below or override it:
#   STEP=2000 sh scripts/004038_run_ver2_5_selected_p0a_eval.sh

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
ROOT=$(CDPATH= cd "${SCRIPT_DIR}/.." && pwd)

STEP="${STEP:-1000}"
RUN_NAME="${RUN_NAME:-ver2_5_selected_p0a_text_token_memory_5k}"
BASE_DIR="${BASE_DIR:-$ROOT/outputs/lora_runs/ver2_5_debug_5k/$RUN_NAME}"

MODEL_PATH="${MODEL_PATH:-$BASE_DIR/step-$STEP}"
MAX_CASES="${MAX_CASES:-48}"
PER_CELL="${PER_CELL:-6}"
RUN_ID="${RUN_ID:-ver2_5_selected_p0a_step${STEP}_notext_core${MAX_CASES}}"
DEVICE="${DEVICE:-cuda:1}"
ASR_DEVICE="${ASR_DEVICE:-cuda:0}"
OVERWRITE_INFER="${OVERWRITE_INFER:-0}"

export MODEL_PATH MAX_CASES PER_CELL RUN_ID DEVICE ASR_DEVICE OVERWRITE_INFER

exec sh "$ROOT/scripts/004034_run_ver2_5_no_text_validation_eval.sh"

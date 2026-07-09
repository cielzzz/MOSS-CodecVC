#!/usr/bin/env bash
# Submit Batch-19-r1 V1'''-lite smoke on v2 pilot no-text with corrected mix ratio.

set -euo pipefail

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
PREPARED_DIR="${PREPARED_DIR:-$ROOT/trainset/ver2_9_prepared_v2_pilot_10k_20260708}"
NO_TEXT_REPEAT="${NO_TEXT_REPEAT:-3}"
TEXT_REPEAT="${TEXT_REPEAT:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1500}"
SAVE_STEPS="${SAVE_STEPS:-500}"
EVAL_STEPS="${EVAL_STEPS:-500}"
DRY_RUN="${DRY_RUN:-1}"

NO_TEXT_JSONL="$PREPARED_DIR/no_text.train.jsonl"
TEXT_JSONL="$PREPARED_DIR/text.train.jsonl"

if [ ! -f "$NO_TEXT_JSONL" ]; then
  echo "ERROR: missing pilot no-text train split: $NO_TEXT_JSONL" >&2
  echo "Run scripts/002041_submit_v2_pilot10k_ver2_9_data_qz.sh and wait for data prep first." >&2
  exit 1
fi
if [ ! -f "$TEXT_JSONL" ]; then
  echo "ERROR: missing old text train symlink: $TEXT_JSONL" >&2
  exit 1
fi

export ARM="${ARM:-v1_v2pilot_r1}"
export PREPARED_DIR
export NO_TEXT_TRAIN_JSONL="$NO_TEXT_JSONL"
export TEXT_TRAIN_JSONL="$TEXT_JSONL"
export TEXT_REPEAT
export TRAIN_JSONL_SPEC="${TRAIN_JSONL_SPEC:-$NO_TEXT_JSONL::repeat=$NO_TEXT_REPEAT,$TEXT_JSONL::repeat=$TEXT_REPEAT}"
export MAX_TRAIN_STEPS
export SAVE_STEPS
export EVAL_STEPS
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export POST_TRAIN_QUICK_EVAL="${POST_TRAIN_QUICK_EVAL:-0}"
export POST_TRAIN_RUN_T11=0
export POST_TRAIN_EVAL_LABEL="${POST_TRAIN_EVAL_LABEL:-v1_v2pilot_r1}"
export JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codecvc-ver2-9-v1-v2pilot-r1-lite}"
export OUT_DIR="${OUT_DIR:-$ROOT/outputs/lora_runs/ver2_9_v1_v2pilot_r1_cross_attn_lite_steps${MAX_TRAIN_STEPS}}"
export BATCH_ID="${BATCH_ID:-v1-v2pilot-r1-$(date -u +%Y%m%d-%H%M%S)}"
export DRY_RUN

echo "[v2pilot-r1-submit] train_jsonl_spec=$TRAIN_JSONL_SPEC"
echo "[v2pilot-r1-submit] no_text_repeat=$NO_TEXT_REPEAT text_repeat=$TEXT_REPEAT max_train_steps=$MAX_TRAIN_STEPS"

bash "$ROOT/scripts/002024_submit_ver2_9_speaker_side_pathway_qz.sh"

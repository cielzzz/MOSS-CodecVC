#!/usr/bin/env bash
# Submit Batch-18 V1'''-lite smoke on prepared v2 pilot-10k no-text + old text data.

set -euo pipefail

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
PREPARED_DIR="${PREPARED_DIR:-$ROOT/trainset/ver2_9_prepared_v2_pilot_10k_20260708}"
TEXT_REPEAT="${TEXT_REPEAT:-10}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
EVAL_STEPS="${EVAL_STEPS:-500}"
DRY_RUN="${DRY_RUN:-1}"

if [ ! -f "$PREPARED_DIR/no_text.train.jsonl" ]; then
  echo "ERROR: missing pilot no-text train split: $PREPARED_DIR/no_text.train.jsonl" >&2
  echo "Run scripts/002041_submit_v2_pilot10k_ver2_9_data_qz.sh and wait for data prep first." >&2
  exit 1
fi
if [ ! -f "$PREPARED_DIR/text.train.jsonl" ]; then
  echo "ERROR: missing old text train symlink: $PREPARED_DIR/text.train.jsonl" >&2
  exit 1
fi

export ARM="${ARM:-v1_v2pilot_cross_attn_lite}"
export PREPARED_DIR
export TEXT_REPEAT
export MAX_TRAIN_STEPS
export SAVE_STEPS
export EVAL_STEPS
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export POST_TRAIN_QUICK_EVAL="${POST_TRAIN_QUICK_EVAL:-1}"
export POST_TRAIN_RUN_T11="${POST_TRAIN_RUN_T11:-0}"
export POST_TRAIN_EVAL_LABEL="${POST_TRAIN_EVAL_LABEL:-v1_v2pilot_cross_attn_lite}"
export JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codecvc-ver2-9-v1-v2pilot10k-lite}"
export OUT_DIR="${OUT_DIR:-$ROOT/outputs/lora_runs/ver2_9_v1_v2pilot10k_cross_attn_lite_steps${MAX_TRAIN_STEPS}}"
export BATCH_ID="${BATCH_ID:-v1-v2pilot10k-lite-$(date -u +%Y%m%d-%H%M%S)}"
export DRY_RUN

bash "$ROOT/scripts/002024_submit_ver2_9_speaker_side_pathway_qz.sh"

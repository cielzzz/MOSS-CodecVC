#!/usr/bin/env bash
# Submit Track-2 v2-data pilot: V1'''-lite architecture on 5-10k no-text v2 rows.

set -euo pipefail

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
FULL_PREPARED_DIR="${FULL_PREPARED_DIR:-$ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_v2_data_20260708}"
PILOT_PREPARED_DIR="${PILOT_PREPARED_DIR:-$ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_v2_data_pilot_20260708}"
TEXT_PREPARED_DIR="${TEXT_PREPARED_DIR:-$ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_20260707}"
PILOT_ROWS="${PILOT_ROWS:-10000}"
PILOT_SEED="${PILOT_SEED:-20260708}"
TEXT_REPEAT="${TEXT_REPEAT:-10}"
PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
DRY_RUN="${DRY_RUN:-1}"

if [ ! -f "$FULL_PREPARED_DIR/no_text.v2.train.filtered.jsonl" ]; then
  echo "ERROR: full filtered v2 prepared split is missing: $FULL_PREPARED_DIR/no_text.v2.train.filtered.jsonl" >&2
  echo "Wait for the v2 real no-text data-prep job to finish first." >&2
  exit 1
fi

"$PY" "$ROOT/scripts/002037_make_v2_real_no_text_pilot.py" \
  --source-prepared-dir "$FULL_PREPARED_DIR" \
  --output-prepared-dir "$PILOT_PREPARED_DIR" \
  --text-prepared-dir "$TEXT_PREPARED_DIR" \
  --sample-size "$PILOT_ROWS" \
  --seed "$PILOT_SEED" \
  --text-repeat "$TEXT_REPEAT"

export ARM="${ARM:-v1_tprime_cross_attn_lite}"
export PREPARED_DIR="$PILOT_PREPARED_DIR"
export TEXT_REPEAT
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-500}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export EVAL_STEPS="${EVAL_STEPS:-500}"
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export POST_TRAIN_QUICK_EVAL="${POST_TRAIN_QUICK_EVAL:-1}"
export POST_TRAIN_RUN_T11=0
export POST_TRAIN_EVAL_LABEL="${POST_TRAIN_EVAL_LABEL:-v1_pilot_v2data}"
export JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codecvc-ver2-9-v1-pilot-v2data-lite}"
export OUT_DIR="${OUT_DIR:-$ROOT/outputs/lora_runs/ver2_9_v1_pilot_v2data_lite_rows${PILOT_ROWS}_steps${MAX_TRAIN_STEPS}}"
export BATCH_ID="${BATCH_ID:-v1-pilot-v2data-$(date -u +%Y%m%d-%H%M%S)}"
export DRY_RUN

sh "$ROOT/scripts/002024_submit_ver2_9_speaker_side_pathway_qz.sh"

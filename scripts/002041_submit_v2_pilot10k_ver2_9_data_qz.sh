#!/usr/bin/env bash
# Build a 10k v2 real-target no-text pilot manifest and submit ver2.9 data prep for it.

set -euo pipefail

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"

SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-/inspire/qb-ilm/project/embodied-multimodality/public/xyzhang/data/VC_train/v2_real_target_no_text_300k_zh_en_balanced_20260707_seedvc_triples}"
RAW_TRAIN_JSONL="${RAW_TRAIN_JSONL:-$SOURCE_DATA_DIR/no_text.train.refdecorr.train_minus_valid.manifest.jsonl}"
PILOT_ROOT="${PILOT_ROOT:-$ROOT/trainset/v2_real_target_no_text_refdecorr_pilot_10k_20260708}"
PILOT_MANIFEST="${PILOT_MANIFEST:-$PILOT_ROOT/manifests/no_text.v2.pilot_10k.manifest.jsonl}"
PILOT_ROWS="${PILOT_ROWS:-10000}"
PILOT_SEED="${PILOT_SEED:-20260708}"

WORK_ROOT="${WORK_ROOT:-$PILOT_ROOT/work}"
PREPARED_DIR="${PREPARED_DIR:-$ROOT/trainset/ver2_9_prepared_v2_pilot_10k_20260708}"
TEXT_PREPARED_DIR="${TEXT_PREPARED_DIR:-$ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_20260707}"
TEXT_REPEAT="${TEXT_REPEAT:-10}"
BATCH_ID="${BATCH_ID:-v2-pilot10k-prep-$(date -u +%Y%m%d-%H%M%S)}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codecvc-ver2-9-v2pilot10k-prep}"
DRY_RUN="${DRY_RUN:-1}"

cd "$ROOT"

"$PY" scripts/002040_make_v2_real_no_text_pilot_manifest.py \
  --input-jsonl "$RAW_TRAIN_JSONL" \
  --output-jsonl "$PILOT_MANIFEST" \
  --sample-size "$PILOT_ROWS" \
  --seed "$PILOT_SEED" \
  --overwrite

export TRAIN_INPUT_JSONL="$PILOT_MANIFEST"
export WORK_ROOT
export PREPARED_DIR
export TEXT_PREPARED_DIR
export TEXT_REPEAT
export BATCH_ID
export JOB_NAME_PREFIX
export DRY_RUN

bash scripts/002035_submit_v2_real_no_text_ver2_9_data_qz.sh

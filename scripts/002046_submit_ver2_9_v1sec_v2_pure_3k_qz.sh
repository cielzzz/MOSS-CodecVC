#!/usr/bin/env bash
# Submit Batch-24: V1-second architecture on full v2 no-text real-target data.
#
# This intentionally disables the K-token / sequence cross-attention path.
# Architecture: [text?, C_src] + full 32-layer AdaLN + K/V bias speaker side
# pathway, with old timbre/source-semantic paths closed.

set -euo pipefail

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
PREPARED_DIR="${PREPARED_DIR:-$ROOT/trainset/ver2_9_prepared_v2_real_no_text_refdecorr_wavlm_sv_20260708}"
NO_TEXT_REPEAT="${NO_TEXT_REPEAT:-1}"
TEXT_REPEAT="${TEXT_REPEAT:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
EVAL_STEPS="${EVAL_STEPS:-500}"
DRY_RUN="${DRY_RUN:-1}"

NO_TEXT_JSONL="$PREPARED_DIR/no_text.v2.train.jsonl"
TEXT_JSONL="$PREPARED_DIR/text.train.jsonl"

if [ ! -f "$NO_TEXT_JSONL" ]; then
  echo "ERROR: missing full v2 no-text train split: $NO_TEXT_JSONL" >&2
  exit 1
fi
if [ ! -f "$TEXT_JSONL" ]; then
  echo "ERROR: missing text train split: $TEXT_JSONL" >&2
  exit 1
fi
if ! head -n 1 "$NO_TEXT_JSONL" | grep -q '"speaker_vec_path"'; then
  echo "ERROR: v2 no-text manifest lacks speaker_vec_path: $NO_TEXT_JSONL" >&2
  exit 1
fi
if ! head -n 1 "$TEXT_JSONL" | grep -q '"speaker_vec_path"'; then
  echo "ERROR: text manifest lacks speaker_vec_path: $TEXT_JSONL" >&2
  exit 1
fi

export ARM="${ARM:-v1}"
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
export POST_TRAIN_EVAL_LABEL="${POST_TRAIN_EVAL_LABEL:-v1sec_v2_pure}"
export JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codecvc-ver2-9-v1sec-v2-pure}"
export OUT_DIR="${OUT_DIR:-$ROOT/outputs/lora_runs/ver2_9_v1sec_v2_pure_steps${MAX_TRAIN_STEPS}}"
export BATCH_ID="${BATCH_ID:-v1sec-v2-pure-$(date -u +%Y%m%d-%H%M%S)}"
export DRY_RUN

# Batch-24 invariant: pure V1-second architecture, no speaker cross-attention.
export ENABLE_SPEAKER_CROSS_ATTN=0
export SPEAKER_CROSS_ATTN_LAYERS=""
export SPEAKER_CROSS_ATTN_TOKENS=0
export SPEAKER_CROSS_ATTN_SOURCE=vector
export SPEAKER_CROSS_ATTN_GATE_INIT=0.0
export SPEAKER_CROSS_ATTN_DROPOUT=0.0
export SPEAKER_CROSS_ATTN_OUTPUT_SCALE=0.0
export SPEAKER_CROSS_ATTN_TOKEN_INIT_STD=""
export SPEAKER_CROSS_ATTN_ALPHA_WARMUP_STEPS=0

echo "[v1sec-v2-pure-submit] train_jsonl_spec=$TRAIN_JSONL_SPEC"
echo "[v1sec-v2-pure-submit] no_text_repeat=$NO_TEXT_REPEAT text_repeat=$TEXT_REPEAT max_train_steps=$MAX_TRAIN_STEPS"
echo "[v1sec-v2-pure-submit] prepared_dir=$PREPARED_DIR out_dir=$OUT_DIR"
echo "[v1sec-v2-pure-submit] speaker_cross_attn=$ENABLE_SPEAKER_CROSS_ATTN"

bash "$ROOT/scripts/002024_submit_ver2_9_speaker_side_pathway_qz.sh"

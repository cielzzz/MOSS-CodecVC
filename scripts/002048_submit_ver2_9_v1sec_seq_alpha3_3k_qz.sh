#!/usr/bin/env bash
# Submit Batch-26 fallback: V1-second + old data + speaker sequence alpha3.

set -euo pipefail

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
PREPARED_DIR="${PREPARED_DIR:-$ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709}"
NO_TEXT_REPEAT="${NO_TEXT_REPEAT:-1}"
TEXT_REPEAT="${TEXT_REPEAT:-10}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
EVAL_STEPS="${EVAL_STEPS:-500}"
DRY_RUN="${DRY_RUN:-1}"

NO_TEXT_JSONL="$PREPARED_DIR/no_text.train.jsonl"
TEXT_JSONL="$PREPARED_DIR/text.train.jsonl"

if [ ! -f "$NO_TEXT_JSONL" ]; then
  echo "ERROR: missing old no-text train split: $NO_TEXT_JSONL" >&2
  exit 1
fi
if [ ! -f "$TEXT_JSONL" ]; then
  echo "ERROR: missing old text train split: $TEXT_JSONL" >&2
  exit 1
fi
for jsonl in "$NO_TEXT_JSONL" "$TEXT_JSONL"; do
  if ! head -n 1 "$jsonl" | grep -q '"speaker_vec_path"'; then
    echo "ERROR: manifest lacks speaker_vec_path: $jsonl" >&2
    exit 1
  fi
  if ! head -n 1 "$jsonl" | grep -q '"speaker_seq_path"'; then
    echo "ERROR: manifest lacks speaker_seq_path: $jsonl" >&2
    exit 1
  fi
done

export ARM="${ARM:-v1sec_seq_alpha3}"
export PREPARED_DIR
export NO_TEXT_TRAIN_JSONL="$NO_TEXT_JSONL"
export TEXT_TRAIN_JSONL="$TEXT_JSONL"
export NO_TEXT_REPEAT
export TEXT_REPEAT
export TRAIN_JSONL_SPEC="${TRAIN_JSONL_SPEC:-$NO_TEXT_JSONL::repeat=$NO_TEXT_REPEAT,$TEXT_JSONL::repeat=$TEXT_REPEAT}"
export MAX_TRAIN_STEPS
export SAVE_STEPS
export EVAL_STEPS
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export POST_TRAIN_QUICK_EVAL="${POST_TRAIN_QUICK_EVAL:-0}"
export POST_TRAIN_RUN_T11=0
export POST_TRAIN_EVAL_LABEL="${POST_TRAIN_EVAL_LABEL:-v1sec_seq_alpha3}"
export JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codecvc-ver2-9-v1sec-seq-alpha3-olddata}"
export OUT_DIR="${OUT_DIR:-$ROOT/outputs/lora_runs/ver2_9_v1sec_seq_alpha3_olddata_steps${MAX_TRAIN_STEPS}}"
export BATCH_ID="${BATCH_ID:-v1sec-seq-alpha3-olddata-$(date -u +%Y%m%d-%H%M%S)}"
export DRY_RUN

echo "[v1sec-seq-alpha3-submit] train_jsonl_spec=$TRAIN_JSONL_SPEC"
echo "[v1sec-seq-alpha3-submit] no_text_repeat=$NO_TEXT_REPEAT text_repeat=$TEXT_REPEAT max_train_steps=$MAX_TRAIN_STEPS"
echo "[v1sec-seq-alpha3-submit] prepared_dir=$PREPARED_DIR out_dir=$OUT_DIR"
echo "[v1sec-seq-alpha3-submit] cross_attn_output_scale=0.6 alpha_warmup_steps=1000"

bash "$ROOT/scripts/002024_submit_ver2_9_speaker_side_pathway_qz.sh"

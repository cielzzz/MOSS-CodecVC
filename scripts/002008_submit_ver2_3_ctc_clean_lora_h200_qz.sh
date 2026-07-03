#!/usr/bin/env sh
set -eu

# Ver2.3 CTC-clean training submission wrapper.
# This version drops every row without reliable CTC supervision:
#   content_keep=true AND nonempty content_token_ids/content_token_length>0

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"

NO_TEXT_DATASET_NAME="${NO_TEXT_DATASET_NAME:-zh45w_en22w_no_text}"
TEXT_DATASET_NAME="${TEXT_DATASET_NAME:-zh3w_en3w_text_prosody_independent_timbre}"

export NO_TEXT_TRAIN_JSONL="${NO_TEXT_TRAIN_JSONL:-$ROOT/trainset/$NO_TEXT_DATASET_NAME/sft/moss_codecvc_sft.$NO_TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_asr_filter.with_hubert.with_spm_content_tokens.ctc_clean.jsonl}"
export TEXT_TRAIN_JSONL="${TEXT_TRAIN_JSONL:-$ROOT/trainset/$TEXT_DATASET_NAME/sft/moss_codecvc_sft.$TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_target_asr.with_content_tokens.with_target_hubert.with_spm_content_tokens.ctc_clean.jsonl}"

# Clean rows: no_text=166158, text=33235. repeat=5 makes the effective mix close to 1:1.
export TEXT_REPEAT="${TEXT_REPEAT:-5}"
export TRAIN_JSONL_SPEC="${TRAIN_JSONL_SPEC:-$NO_TEXT_TRAIN_JSONL::repeat=1,$TEXT_TRAIN_JSONL::repeat=$TEXT_REPEAT}"
export OUT_DIR="${OUT_DIR:-$ROOT/outputs/lora_runs/ver2_3_ctc_clean_textrep${TEXT_REPEAT}_spm_lora_r16_a32_gbs64}"
export JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codecvc-ver2-3-ctc-clean-textrep${TEXT_REPEAT}-train-lora}"
export ENABLE_EVAL="${ENABLE_EVAL:-1}"
if [ "$ENABLE_EVAL" = "1" ]; then
  export EVAL_JSONL="${EVAL_JSONL:-$ROOT/testset/validation/ver2_3_debug/moss_codecvc_ver2_3_loss_valid_160.jsonl}"
else
  unset EVAL_JSONL
  unset EVAL_JSONL_SPEC
fi

# First clean run: ~332k effective rows at global batch 64 => ~5193 steps/epoch.
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"

export CONTENT_CTC_WEIGHT="${CONTENT_CTC_WEIGHT:-0.10}"
export SEMANTIC_LOSS_WEIGHT="${SEMANTIC_LOSS_WEIGHT:-0.05}"
export PROGRESS_LOSS_WEIGHT="${PROGRESS_LOSS_WEIGHT:-0.02}"
export STOP_LOSS_WEIGHT="${STOP_LOSS_WEIGHT:-0.05}"
export PROGRESS_NUM_BINS="${PROGRESS_NUM_BINS:-32}"
export LAMBDA_PROSODY="${LAMBDA_PROSODY:-0.05}"
export TARGET_SPK_WEIGHT="${TARGET_SPK_WEIGHT:-0.05}"
export SOURCE_SUPPRESS_WEIGHT="${SOURCE_SUPPRESS_WEIGHT:-0.05}"
export ROUTING_GATE_LR_MULTIPLIER="${ROUTING_GATE_LR_MULTIPLIER:-10.0}"

echo "[ver2.3-ctc-clean-submit] no_text=$NO_TEXT_TRAIN_JSONL"
echo "[ver2.3-ctc-clean-submit] text=$TEXT_TRAIN_JSONL"
echo "[ver2.3-ctc-clean-submit] train_jsonl_spec=$TRAIN_JSONL_SPEC"
echo "[ver2.3-ctc-clean-submit] out_dir=$OUT_DIR"
echo "[ver2.3-ctc-clean-submit] eval_jsonl=${EVAL_JSONL:-}"
echo "[ver2.3-ctc-clean-submit] max_train_steps=$MAX_TRAIN_STEPS text_repeat=$TEXT_REPEAT"
echo "[ver2.3-ctc-clean-submit] content_ctc=$CONTENT_CTC_WEIGHT semantic=$SEMANTIC_LOSS_WEIGHT progress=$PROGRESS_LOSS_WEIGHT stop=$STOP_LOSS_WEIGHT"

test -f "$NO_TEXT_TRAIN_JSONL"
test -f "$TEXT_TRAIN_JSONL"

exec sh "$ROOT/scripts/002004_submit_ver2_lora_68w_h200_qz.sh" "$@"

#!/usr/bin/env sh
set -eu

# Ver2.7 repair mainline:
# - use the Ver2.3 CTC-clean structure that preserved text-mode timbre better;
# - disable Ver2.6 SourceSemanticMemory/content-memory by default;
# - train by epochs instead of a fixed step budget.

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
SUBMIT_SCRIPT="$ROOT/scripts/002004_submit_ver2_lora_68w_h200_qz.sh"

NO_TEXT_DATASET_NAME="${NO_TEXT_DATASET_NAME:-zh45w_en22w_no_text}"
TEXT_DATASET_NAME="${TEXT_DATASET_NAME:-zh3w_en3w_text_prosody_independent_timbre}"

NO_TEXT_CTC_CLEAN="$ROOT/trainset/$NO_TEXT_DATASET_NAME/sft/moss_codecvc_sft.$NO_TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_asr_filter.with_hubert.with_spm_content_tokens.ctc_clean.jsonl"
TEXT_CTC_CLEAN="$ROOT/trainset/$TEXT_DATASET_NAME/sft/moss_codecvc_sft.$TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_target_asr.with_content_tokens.with_target_hubert.with_spm_content_tokens.ctc_clean.jsonl"
VER2_6_VALID_JSONL="$ROOT/testset/validation/ver2_6/ver2_6_full_valid_seed20260701.jsonl"

for required_path in "$SUBMIT_SCRIPT" "$NO_TEXT_CTC_CLEAN" "$TEXT_CTC_CLEAN" "$VER2_6_VALID_JSONL"; do
  if [ ! -e "$required_path" ]; then
    echo "ERROR: missing required path: $required_path" >&2
    exit 1
  fi
done

TEXT_REPEAT="${TEXT_REPEAT:-5}"
BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/lora_runs/ver2_7_ver23_ctc_clean_epoch_textrep${TEXT_REPEAT}_$BATCH_ID}"
QZ_RECORD_ROOT="${QZ_RECORD_ROOT:-$ROOT/trainset/qz_jobs/ver2_7_ver23_ctc_epoch_$BATCH_ID}"

echo "=========================================="
echo "[ver2.7-submit] strategy=ver2.3-ctc-clean-epoch"
echo "[ver2.7-submit] no_text=$NO_TEXT_CTC_CLEAN"
echo "[ver2.7-submit] text=$TEXT_CTC_CLEAN"
echo "[ver2.7-submit] text_repeat=$TEXT_REPEAT"
echo "[ver2.7-submit] out_dir=$OUT_DIR"
echo "[ver2.7-submit] qz_record_root=$QZ_RECORD_ROOT"
echo "[ver2.7-submit] valid_jsonl=$VER2_6_VALID_JSONL"
echo "=========================================="

env \
  ROOT="$ROOT" \
  NO_TEXT_TRAIN_JSONL="$NO_TEXT_CTC_CLEAN" \
  TEXT_TRAIN_JSONL="$TEXT_CTC_CLEAN" \
  TEXT_REPEAT="$TEXT_REPEAT" \
  TRAIN_JSONL_SPEC="$NO_TEXT_CTC_CLEAN::repeat=1,$TEXT_CTC_CLEAN::repeat=$TEXT_REPEAT" \
  OUT_DIR="$OUT_DIR" \
  JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codecvc-ver2-7-ver23-ctc-clean-epoch-textrep${TEXT_REPEAT}}" \
  BATCH_ID="$BATCH_ID" \
  QZ_RECORD_ROOT="$QZ_RECORD_ROOT" \
  NUM_EPOCHS="${NUM_EPOCHS:-6}" \
  MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}" \
  SAVE_STEPS="${SAVE_STEPS:-1000}" \
  EVAL_JSONL="${EVAL_JSONL:-$VER2_6_VALID_JSONL}" \
  EVAL_STEPS="${EVAL_STEPS:-1000}" \
  EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-100}" \
  EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}" \
  COMPUTE_GROUP="${COMPUTE_GROUP:-lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122}" \
  PRIORITY="${PRIORITY:-10}" \
  PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}" \
  GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}" \
  LEARNING_RATE="${LEARNING_RATE:-1e-5}" \
  WARMUP_RATIO="${WARMUP_RATIO:-0.03}" \
  LORA_R="${LORA_R:-16}" \
  LORA_ALPHA="${LORA_ALPHA:-32}" \
  LORA_DROPOUT="${LORA_DROPOUT:-0.05}" \
  NUM_WORKERS="${NUM_WORKERS:-4}" \
  ENABLE_SOURCE_SEMANTIC_MEMORY=0 \
  CONTENT_CTC_WEIGHT="${CONTENT_CTC_WEIGHT:-0.10}" \
  SEMANTIC_LOSS_WEIGHT="${SEMANTIC_LOSS_WEIGHT:-0.05}" \
  SEMANTIC_MODE="${SEMANTIC_MODE:-continuous}" \
  SEMANTIC_SOURCE="${SEMANTIC_SOURCE:-mode_aware}" \
  PROGRESS_LOSS_WEIGHT="${PROGRESS_LOSS_WEIGHT:-0.02}" \
  STOP_LOSS_WEIGHT="${STOP_LOSS_WEIGHT:-0.05}" \
  PROGRESS_NUM_BINS="${PROGRESS_NUM_BINS:-32}" \
  LAMBDA_PROSODY="${LAMBDA_PROSODY:-0.05}" \
  PROSODY_F0_WEIGHT="${PROSODY_F0_WEIGHT:-0.0}" \
  PROSODY_VOICED_WEIGHT="${PROSODY_VOICED_WEIGHT:-0.0}" \
  PROSODY_ENERGY_WEIGHT="${PROSODY_ENERGY_WEIGHT:-0.5}" \
  PROSODY_PAUSE_WEIGHT="${PROSODY_PAUSE_WEIGHT:-1.0}" \
  PROSODY_DURATION_WEIGHT="${PROSODY_DURATION_WEIGHT:-0.5}" \
  TARGET_SPK_WEIGHT="${TARGET_SPK_WEIGHT:-0.05}" \
  SOURCE_SUPPRESS_WEIGHT="${SOURCE_SUPPRESS_WEIGHT:-0.05}" \
  SPEAKER_LOSS_WARMUP_STEPS="${SPEAKER_LOSS_WARMUP_STEPS:-1000}" \
  SPEAKER_LOSS_WARMUP_WEIGHT="${SPEAKER_LOSS_WARMUP_WEIGHT:-0.02}" \
  SPEAKER_LOSS_MARGIN="${SPEAKER_LOSS_MARGIN:-0.10}" \
  ROUTING_GATE_LR_MULTIPLIER="${ROUTING_GATE_LR_MULTIPLIER:-10.0}" \
  CONTENT_CTC_HEAD_LR_MULTIPLIER="${CONTENT_CTC_HEAD_LR_MULTIPLIER:-1.0}" \
  TIMBRE_ADAPTER_INIT_GATE="${TIMBRE_ADAPTER_INIT_GATE:--4.0}" \
  TIMBRE_ADAPTER_GATE_LR_MULTIPLIER="${TIMBRE_ADAPTER_GATE_LR_MULTIPLIER:-1.0}" \
  LAMBDA_CONTENT=0 \
  CONTENT_TOKEN_WEIGHT=0 \
  CONTENT_SOURCE_CODEC_WEIGHT=0 \
  sh "$SUBMIT_SCRIPT" "$@"

#!/bin/sh
# Submit the five Ver2.6 full-data LoRA runs.
#
# Default:
#   sh scripts/002010_submit_ver2_6_full_lora_h200_qz.sh
#
# Check generated runners without submitting:
#   sh scripts/002010_submit_ver2_6_full_lora_h200_qz.sh --dry-run

set -eu

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
SUBMIT_SCRIPT="$ROOT/scripts/002004_submit_ver2_lora_68w_h200_qz.sh"
DRY_ARG=""
if [ "${1:-}" = "--dry-run" ]; then
  DRY_ARG="--dry-run"
  shift
fi
if [ "$#" -ne 0 ]; then
  echo "Usage: sh scripts/002010_submit_ver2_6_full_lora_h200_qz.sh [--dry-run]" >&2
  exit 2
fi

NO_TEXT_TRAIN_JSONL="$ROOT/trainset/zh45w_en22w_no_text/sft/moss_codecvc_sft.zh45w_en22w_no_text.with_light_ecapa_spk.with_prosody.with_asr_filter.with_hubert.with_spm_content_tokens.train_seed20260701.jsonl"
TEXT_OLD_TRAIN_JSONL="$ROOT/trainset/zh3w_en3w_text_prosody_independent_timbre/sft/moss_codecvc_sft.zh3w_en3w_text_prosody_independent_timbre.with_light_ecapa_spk.with_prosody.with_target_asr.with_content_tokens.with_target_hubert.with_spm_content_tokens.train_seed20260701.jsonl"
TEXT_NEW_TRAIN_JSONL="$ROOT/trainset/zh11w_en11w_0005_0015_vcdata_first_text_prosody/sft/moss_codecvc_sft.zh11w_en11w_0005_0015_vcdata_first_text_prosody.with_light_ecapa_spk.with_prosody.with_target_asr.with_content_tokens.with_target_hubert.with_spm_content_tokens.train_seed20260701.jsonl"
VER2_6_VALID_JSONL="$ROOT/testset/validation/ver2_6/ver2_6_full_valid_seed20260701.jsonl"

TRAIN_JSONL_SPEC_VER2_6="$NO_TEXT_TRAIN_JSONL::repeat=1,$TEXT_OLD_TRAIN_JSONL::repeat=3,$TEXT_NEW_TRAIN_JSONL::repeat=2"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/lora_runs/ver2_6_full}"
BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%d-%H%M%S)}"
MASTER_RECORD_ROOT="${MASTER_RECORD_ROOT:-$ROOT/trainset/qz_jobs/ver2_6_full_$BATCH_ID}"
MAIN_COMPUTE_GROUP="${MAIN_COMPUTE_GROUP:-lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122}"  # MTTS-3-2-0715
MAIN_PRIORITY="${MAIN_PRIORITY:-10}"
INFRA_DEBUG_COMPUTE_GROUP="${INFRA_DEBUG_COMPUTE_GROUP:-lcg-4202c8f7-8308-412c-92b9-77daccab3c7f}"
INFRA_DEBUG_PRIORITY="${INFRA_DEBUG_PRIORITY:-3}"
DEFAULT_SAVE_STEPS="${SAVE_STEPS:-1000}"

for required_path in "$SUBMIT_SCRIPT" "$NO_TEXT_TRAIN_JSONL" "$TEXT_OLD_TRAIN_JSONL" "$TEXT_NEW_TRAIN_JSONL" "$VER2_6_VALID_JSONL"; do
  if [ ! -e "$required_path" ]; then
    echo "ERROR: missing required path: $required_path" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT" "$MASTER_RECORD_ROOT"

submit_run() {
  run_id="$1"
  run_name="$2"
  compute_group="$3"
  priority="$4"
  shift 4

  out_dir="$OUT_ROOT/${run_name}_$BATCH_ID"
  qz_record_root="$MASTER_RECORD_ROOT/$run_id"
  echo "=========================================="
  echo "[ver2.6-submit] run_id=$run_id run_name=$run_name"
  echo "[ver2.6-submit] compute_group=$compute_group priority=$priority"
  echo "[ver2.6-submit] out_dir=$out_dir"
  echo "[ver2.6-submit] qz_record_root=$qz_record_root"
  echo "=========================================="

  env \
    ROOT="$ROOT" \
    NO_TEXT_TRAIN_JSONL="$NO_TEXT_TRAIN_JSONL" \
    TEXT_TRAIN_JSONL="$TEXT_OLD_TRAIN_JSONL" \
    TRAIN_JSONL_SPEC="$TRAIN_JSONL_SPEC_VER2_6" \
    OUT_DIR="$out_dir" \
    JOB_NAME_PREFIX="codecvc-$run_name" \
    BATCH_ID="$BATCH_ID-$run_id" \
    QZ_RECORD_ROOT="$qz_record_root" \
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20000}" \
    SAVE_STEPS="$DEFAULT_SAVE_STEPS" \
    EVAL_JSONL="$VER2_6_VALID_JSONL" \
    EVAL_STEPS="${EVAL_STEPS:-2000}" \
    EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-100}" \
    EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}" \
    COMPUTE_GROUP="$compute_group" \
    PRIORITY="$priority" \
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}" \
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}" \
    LEARNING_RATE="${LEARNING_RATE:-1e-5}" \
    WARMUP_RATIO="${WARMUP_RATIO:-0.03}" \
    LORA_R="${LORA_R:-16}" \
    LORA_ALPHA="${LORA_ALPHA:-32}" \
    LORA_DROPOUT="${LORA_DROPOUT:-0.05}" \
    NUM_WORKERS="${NUM_WORKERS:-4}" \
    ENABLE_SOURCE_SEMANTIC_MEMORY=1 \
    SOURCE_CONTENT_MEMORY_TYPE=text_tokens \
    SOURCE_CONTENT_VOCAB_SIZE=8001 \
    SOURCE_CONTENT_PADDING_ID=0 \
    SOURCE_SEMANTIC_INIT_GATE=-1.0 \
    SOURCE_SEMANTIC_POSITION_SCALE=0.10 \
    SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH=2.0 \
    SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH=0.25 \
    SOURCE_SEMANTIC_NO_TEXT_GATE=1.0 \
    SOURCE_SEMANTIC_TEXT_GATE=0.0 \
    SOURCE_SEMANTIC_LR_MULTIPLIER=1.0 \
    SOURCE_SEMANTIC_GATE_LR_MULTIPLIER=10.0 \
    CONTENT_CTC_WEIGHT=0 \
    SEMANTIC_LOSS_WEIGHT=0 \
    LAMBDA_CONTENT=0 \
    LAMBDA_PROSODY=0 \
    PROGRESS_LOSS_WEIGHT=0 \
    STOP_LOSS_WEIGHT=0 \
    TARGET_SPK_WEIGHT=0.03 \
    SOURCE_SUPPRESS_WEIGHT=0.03 \
    SPEAKER_LOSS_WARMUP_STEPS=2000 \
    SPEAKER_LOSS_WARMUP_WEIGHT=0.01 \
    SPEAKER_LOSS_MARGIN=0.10 \
    "$@" \
    sh "$SUBMIT_SCRIPT" $DRY_ARG
}

submit_run \
  "ver2_6_1" \
  "ver2-6-1-p0a-from0-spk-full" \
  "$MAIN_COMPUTE_GROUP" \
  "$MAIN_PRIORITY"

submit_run \
  "ver2_6_2" \
  "ver2-6-2-p0a-from0-spk-progress-stop-full" \
  "$MAIN_COMPUTE_GROUP" \
  "$MAIN_PRIORITY" \
  PROGRESS_LOSS_WEIGHT=0.01 \
  STOP_LOSS_WEIGHT=0.02

submit_run \
  "ver2_6_3" \
  "ver2-6-3-p0a-from0-spk-prosody-full" \
  "$MAIN_COMPUTE_GROUP" \
  "$MAIN_PRIORITY" \
  LAMBDA_PROSODY=0.02 \
  PROSODY_F0_WEIGHT=0.0 \
  PROSODY_VOICED_WEIGHT=0.0 \
  PROSODY_ENERGY_WEIGHT=0.3 \
  PROSODY_PAUSE_WEIGHT=0.5 \
  PROSODY_DURATION_WEIGHT=0.3

submit_run \
  "ver2_6_4" \
  "ver2-6-4-p0a-from0-spk-gate0-full" \
  "$MAIN_COMPUTE_GROUP" \
  "$MAIN_PRIORITY" \
  SOURCE_SEMANTIC_INIT_GATE=0.0 \
  SOURCE_SEMANTIC_LR_MULTIPLIER=5.0 \
  SOURCE_SEMANTIC_GATE_LR_MULTIPLIER=20.0

submit_run \
  "ver2_6_5" \
  "ver2-6-5-p0c-from0-spk-full" \
  "$INFRA_DEBUG_COMPUTE_GROUP" \
  "$INFRA_DEBUG_PRIORITY" \
  SOURCE_CONTENT_MEMORY_TYPE=codec_bottleneck \
  SOURCE_CONTENT_VOCAB_SIZE=0 \
  SOURCE_CONTENT_CODEC_CODEBOOKS=first_4 \
  SOURCE_CONTENT_CODEC_BOTTLENECK_DIM=256

echo "=========================================="
echo "[ver2.6-submit] done"
echo "  batch_id=$BATCH_ID"
echo "  records=$MASTER_RECORD_ROOT"
echo "  save_steps=$DEFAULT_SAVE_STEPS"
echo "  main_compute_group=$MAIN_COMPUTE_GROUP priority=$MAIN_PRIORITY"
echo "  infra_debug_compute_group=$INFRA_DEBUG_COMPUTE_GROUP priority=$INFRA_DEBUG_PRIORITY"
echo "  train_jsonl_spec=$TRAIN_JSONL_SPEC_VER2_6"
echo "  eval_jsonl=$VER2_6_VALID_JSONL"
echo "=========================================="

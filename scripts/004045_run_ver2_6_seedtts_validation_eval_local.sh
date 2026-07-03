#!/usr/bin/env bash
set -euo pipefail

# Local fallback for the five Ver2.6 SeedTTS-320 evaluations.
# Uses persistent inference so each GPU shard loads a checkpoint once, then
# loops over its validation subset.

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
ROOT=$(CDPATH= cd "${SCRIPT_DIR}/.." && pwd)

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
ASR_PYTHON="${ASR_PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
EVAL_STEP="${EVAL_STEP:-10000}"
DECODING_PROFILE="${DECODING_PROFILE:-topk1}"
BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%d-%H%M%S)-local4090}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/testset/outputs/ver2_6_seedtts320_eval/step${EVAL_STEP}_${DECODING_PROFILE}_${BATCH_ID}}"

if [ -z "${GPU_COUNT:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l | tr -d ' ')
  else
    GPU_COUNT=1
  fi
fi
if [ "$GPU_COUNT" -lt 1 ]; then
  GPU_COUNT=1
fi
NUM_SHARDS="${NUM_SHARDS:-$GPU_COUNT}"
ASR_NUM_SHARDS="${ASR_NUM_SHARDS:-$NUM_SHARDS}"
RUN_ASR="${RUN_ASR:-1}"
BUILD_PAGE="${BUILD_PAGE:-1}"

RUN_BASE="$ROOT/outputs/lora_runs/ver2_6_full"
RUN1="$RUN_BASE/ver2-6-1-p0a-from0-spk-full_20260701-163106-save1000"
RUN2="$RUN_BASE/ver2-6-2-p0a-from0-spk-progress-stop-full_20260701-163106-save1000"
RUN3="$RUN_BASE/ver2-6-3-p0a-from0-spk-prosody-full_20260701-163106-save1000"
RUN4="$RUN_BASE/ver2-6-4-p0a-from0-spk-gate0-full_20260701-163106-save1000"
RUN5="$RUN_BASE/ver2-6-5-p0c-from0-spk-full_20260701-163106-save1000"

for run_dir in "$RUN1" "$RUN2" "$RUN3" "$RUN4" "$RUN5"; do
  if [ ! -d "$run_dir/step-$EVAL_STEP" ]; then
    echo "ERROR: missing checkpoint: $run_dir/step-$EVAL_STEP" >&2
    exit 1
  fi
done

export PYTHON
export ASR_PYTHON
export DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download}"
export HF_HOME="${HF_HOME:-$DOWNLOAD_ROOT/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$DOWNLOAD_ROOT/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$DOWNLOAD_ROOT/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$DOWNLOAD_ROOT/huggingface/datasets}"
export TORCH_HOME="${TORCH_HOME:-$DOWNLOAD_ROOT/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DOWNLOAD_ROOT/cache}"
export TOKENIZERS_PARALLELISM=false

cd "$ROOT"
mkdir -p "$EVAL_ROOT"

echo "[local-ver2.6-seedtts320] eval_root=$EVAL_ROOT"
echo "[local-ver2.6-seedtts320] gpu_count=$GPU_COUNT shards=$NUM_SHARDS asr_shards=$ASR_NUM_SHARDS run_asr=$RUN_ASR"

"$PYTHON" "$ROOT/scripts/004041_summarize_ver2_6_loss_trends.py" \
  --output-dir "$EVAL_ROOT/loss_trends" \
  --eval-step "$EVAL_STEP"

run_eval() {
  local run_id="$1"
  local label="$2"
  local model_path="$3"
  local output_dir="$EVAL_ROOT/$run_id"
  RUN_ID="$run_id" \
  RUN_LABEL="$label" \
  MODEL_PATH="$model_path" \
  OUTPUT_DIR="$output_dir" \
  NUM_SHARDS="$NUM_SHARDS" \
  ASR_NUM_SHARDS="$ASR_NUM_SHARDS" \
  GPU_COUNT="$GPU_COUNT" \
  MODE=all \
  MAX_CASES=0 \
  PER_MODE=0 \
  PER_CELL=0 \
  PERSISTENT_INFER=1 \
  OVERWRITE_INFER=0 \
  RESET_MANIFESTS=1 \
  DECODING_PROFILE="$DECODING_PROFILE" \
  CONTENT_REFERENCE_MODE=text \
  RUN_ASR="$RUN_ASR" \
  BUILD_PAGE="$BUILD_PAGE" \
  PAGE_DIR="$EVAL_ROOT/listening_page" \
  bash "$ROOT/scripts/004039_run_seedtts_validation_eval.sh"
}

run_eval "ver2_6_1_step${EVAL_STEP}_${DECODING_PROFILE}" "ver2.6.1 P0-A speaker step-${EVAL_STEP}" "$RUN1/step-$EVAL_STEP"
run_eval "ver2_6_2_step${EVAL_STEP}_${DECODING_PROFILE}" "ver2.6.2 progress/stop step-${EVAL_STEP}" "$RUN2/step-$EVAL_STEP"
run_eval "ver2_6_3_step${EVAL_STEP}_${DECODING_PROFILE}" "ver2.6.3 prosody step-${EVAL_STEP}" "$RUN3/step-$EVAL_STEP"
run_eval "ver2_6_4_step${EVAL_STEP}_${DECODING_PROFILE}" "ver2.6.4 stronger source memory step-${EVAL_STEP}" "$RUN4/step-$EVAL_STEP"
run_eval "ver2_6_5_step${EVAL_STEP}_${DECODING_PROFILE}" "ver2.6.5 codec bottleneck step-${EVAL_STEP}" "$RUN5/step-$EVAL_STEP"

"$PYTHON" "$ROOT/scripts/004043_compare_seedtts_validation_runs.py" \
  --eval-root "$EVAL_ROOT" \
  --output-md "$EVAL_ROOT/COMPARE.md" \
  --output-csv "$EVAL_ROOT/compare_summary.csv"

echo "[local-ver2.6-seedtts320] done: $EVAL_ROOT"

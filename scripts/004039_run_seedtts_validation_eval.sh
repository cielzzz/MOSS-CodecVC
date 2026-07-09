#!/usr/bin/env bash
set -euo pipefail

# Run the fixed 320-case SeedTTS validation set for one checkpoint.
# The script parallelizes inference and ASR by modulo shards, then writes
# manifest/eval-input/asr/metrics/SUMMARY files into OUTPUT_DIR.

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
ROOT=$(CDPATH= cd "${SCRIPT_DIR}/.." && pwd)

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
ASR_PYTHON="${ASR_PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
VALIDATION_JSONL="${VALIDATION_JSONL:-$ROOT/testset/validation/seedtts_vc_ver2_3_validation.jsonl}"
MODEL_PATH="${MODEL_PATH:-}"
RUN_ID="${RUN_ID:-}"
RUN_LABEL="${RUN_LABEL:-$RUN_ID}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

VALID_INFER_SCRIPT="${VALID_INFER_SCRIPT:-$ROOT/scripts/004013_run_seedtts_validation_infer.py}"
PERSISTENT_INFER_SCRIPT="${PERSISTENT_INFER_SCRIPT:-$ROOT/scripts/004044_run_seedtts_validation_infer_persistent.py}"
RUN_SCRIPT="${RUN_SCRIPT:-$ROOT/scripts/003003_run_moss_codecvc_infer.sh}"
BUILD_EVAL_SCRIPT="${BUILD_EVAL_SCRIPT:-$ROOT/scripts/004017_build_seedtts_generated_eval_jsonl.py}"
ASR_SCRIPT="${ASR_SCRIPT:-$ROOT/scripts/001017_asr_content_filter.py}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-$ROOT/scripts/004042_summarize_seedtts_validation_eval.py}"
PAGE_SCRIPT="${PAGE_SCRIPT:-$ROOT/scripts/004014_build_seedtts_validation_benchmark_page.py}"

MODE="${MODE:-all}"
MAX_CASES="${MAX_CASES:-0}"
PER_MODE="${PER_MODE:-0}"
PER_CELL="${PER_CELL:-0}"
OVERWRITE_INFER="${OVERWRITE_INFER:-0}"
RESET_MANIFESTS="${RESET_MANIFESTS:-1}"
RUN_INFER="${RUN_INFER:-1}"
RUN_ASR="${RUN_ASR:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
BUILD_PAGE="${BUILD_PAGE:-0}"
PAGE_DIR="${PAGE_DIR:-$ROOT/outputs/listening_frontend/ver2_6_seedtts320}"
PERSISTENT_INFER="${PERSISTENT_INFER:-0}"
INFER_SHARD_START_DELAY_SEC="${INFER_SHARD_START_DELAY_SEC:-0}"
SEED="${SEED:-}"
REF_PROMPT_CODEC_PERMUTATION="${REF_PROMPT_CODEC_PERMUTATION:-}"
REF_PROMPT_CODEC_PERMUTATION_MIN_SECONDS="${REF_PROMPT_CODEC_PERMUTATION_MIN_SECONDS:-}"
REF_PROMPT_CODEC_PERMUTATION_MAX_SECONDS="${REF_PROMPT_CODEC_PERMUTATION_MAX_SECONDS:-}"
REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE="${REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE:-}"
REF_PROMPT_CODEC_PERMUTATION_SEED="${REF_PROMPT_CODEC_PERMUTATION_SEED:-1234}"
REF_PROMPT_CODEC_PERMUTATION_MODE="${REF_PROMPT_CODEC_PERMUTATION_MODE:-}"
REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS="${REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS:-}"
REF_PROMPT_CODEC_PERMUTATION_BOOTSTRAP="${REF_PROMPT_CODEC_PERMUTATION_BOOTSTRAP:-}"
TIMBRE_CFG_SCALE="${TIMBRE_CFG_SCALE:-1.0}"

GPU_COUNT="${GPU_COUNT:-}"
if [ -z "$GPU_COUNT" ]; then
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

DECODING_PROFILE="${DECODING_PROFILE:-topk1}"
case "$DECODING_PROFILE" in
  topk1|near_greedy|greedy)
    AUDIO_TEMPERATURE_VALUE="${AUDIO_TEMPERATURE:-1.0}"
    AUDIO_TOP_P_VALUE="${AUDIO_TOP_P:-1.0}"
    AUDIO_TOP_K_VALUE="${AUDIO_TOP_K:-1}"
    AUDIO_REPETITION_PENALTY_VALUE="${AUDIO_REPETITION_PENALTY:-1.05}"
    ;;
  lowrand|topk20)
    AUDIO_TEMPERATURE_VALUE="${AUDIO_TEMPERATURE:-1.20}"
    AUDIO_TOP_P_VALUE="${AUDIO_TOP_P:-0.70}"
    AUDIO_TOP_K_VALUE="${AUDIO_TOP_K:-20}"
    AUDIO_REPETITION_PENALTY_VALUE="${AUDIO_REPETITION_PENALTY:-1.10}"
    ;;
  default)
    AUDIO_TEMPERATURE_VALUE="${AUDIO_TEMPERATURE:-}"
    AUDIO_TOP_P_VALUE="${AUDIO_TOP_P:-}"
    AUDIO_TOP_K_VALUE="${AUDIO_TOP_K:-}"
    AUDIO_REPETITION_PENALTY_VALUE="${AUDIO_REPETITION_PENALTY:-}"
    ;;
  *)
    echo "ERROR: unsupported DECODING_PROFILE=$DECODING_PROFILE" >&2
    exit 2
    ;;
esac
NO_TEXT_AUDIO_TEMPERATURE_VALUE="${NO_TEXT_AUDIO_TEMPERATURE:-$AUDIO_TEMPERATURE_VALUE}"
NO_TEXT_AUDIO_TOP_P_VALUE="${NO_TEXT_AUDIO_TOP_P:-$AUDIO_TOP_P_VALUE}"
NO_TEXT_AUDIO_TOP_K_VALUE="${NO_TEXT_AUDIO_TOP_K:-$AUDIO_TOP_K_VALUE}"
NO_TEXT_AUDIO_REPETITION_PENALTY_VALUE="${NO_TEXT_AUDIO_REPETITION_PENALTY:-$AUDIO_REPETITION_PENALTY_VALUE}"

CONTENT_REFERENCE_MODE="${CONTENT_REFERENCE_MODE:-text}"
ASR_BACKEND="${ASR_BACKEND:-qwen_asr}"
ASR_DEVICE_PREFIX="${ASR_DEVICE_PREFIX:-cuda}"
QWEN_ASR_MODEL="${QWEN_ASR_MODEL:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/checkpoint/qwen-asr-1_7b}"
QWEN_ASR_DTYPE="${QWEN_ASR_DTYPE:-bfloat16}"
QWEN_ASR_MAX_BATCH_SIZE="${QWEN_ASR_MAX_BATCH_SIZE:-1}"
QWEN_ASR_MAX_NEW_TOKENS="${QWEN_ASR_MAX_NEW_TOKENS:-256}"
NO_TEXT_ZH_CER_THRESHOLD="${NO_TEXT_ZH_CER_THRESHOLD:-0.35}"
NO_TEXT_EN_WER_THRESHOLD="${NO_TEXT_EN_WER_THRESHOLD:-0.30}"
ZH_CER_THRESHOLD="${ZH_CER_THRESHOLD:-0.20}"
EN_WER_THRESHOLD="${EN_WER_THRESHOLD:-0.25}"
MAX_REPEAT_SCORE="${MAX_REPEAT_SCORE:-0.30}"

if [ -z "$MODEL_PATH" ]; then
  echo "ERROR: MODEL_PATH is required" >&2
  exit 2
fi
if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
  exit 1
fi
if [ -z "$RUN_ID" ]; then
  parent=$(basename "$(dirname "$MODEL_PATH")")
  leaf=$(basename "$MODEL_PATH")
  RUN_ID="${parent}_${leaf}_${DECODING_PROFILE}"
fi
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$ROOT/testset/outputs/seedtts_validation_eval/$RUN_ID"
fi

mkdir -p "$OUTPUT_DIR/logs"
EVAL_INPUT_JSONL="$OUTPUT_DIR/${RUN_ID}.generated_eval_input.jsonl"
ASR_JSONL="$OUTPUT_DIR/${RUN_ID}.asr_eval.jsonl"
METRICS_CSV="$OUTPUT_DIR/${RUN_ID}.metrics.csv"
SUMMARY_MD="$OUTPUT_DIR/SUMMARY.md"
SUMMARY_JSON="$OUTPUT_DIR/${RUN_ID}.summary.json"

echo "[seedtts-eval] run_id=$RUN_ID"
echo "[seedtts-eval] model=$MODEL_PATH"
echo "[seedtts-eval] validation=$VALIDATION_JSONL"
echo "[seedtts-eval] output_dir=$OUTPUT_DIR"
echo "[seedtts-eval] mode=$MODE max_cases=$MAX_CASES per_mode=$PER_MODE per_cell=$PER_CELL"
echo "[seedtts-eval] shards=$NUM_SHARDS gpu_count=$GPU_COUNT asr_shards=$ASR_NUM_SHARDS"
echo "[seedtts-eval] decoding=$DECODING_PROFILE audio_temperature=${AUDIO_TEMPERATURE_VALUE:-default} audio_top_p=${AUDIO_TOP_P_VALUE:-default} audio_top_k=${AUDIO_TOP_K_VALUE:-default} audio_repetition_penalty=${AUDIO_REPETITION_PENALTY_VALUE:-default}"
echo "[seedtts-eval] persistent_infer=$PERSISTENT_INFER"
echo "[seedtts-eval] infer_shard_start_delay_sec=$INFER_SHARD_START_DELAY_SEC"
echo "[seedtts-eval] seed=${SEED:-unset}"
echo "[seedtts-eval] ref_prompt_codec_permutation=${REF_PROMPT_CODEC_PERMUTATION:-checkpoint} seconds=${REF_PROMPT_CODEC_PERMUTATION_MIN_SECONDS:-checkpoint}-${REF_PROMPT_CODEC_PERMUTATION_MAX_SECONDS:-checkpoint} frame_rate=${REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE:-checkpoint} mode=${REF_PROMPT_CODEC_PERMUTATION_MODE:-checkpoint} block_seconds=${REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS:-checkpoint} bootstrap=${REF_PROMPT_CODEC_PERMUTATION_BOOTSTRAP:-off} seed=$REF_PROMPT_CODEC_PERMUTATION_SEED"
echo "[seedtts-eval] timbre_cfg_scale=$TIMBRE_CFG_SCALE"

manifest_args=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  manifest="$OUTPUT_DIR/manifest.shard${shard}.jsonl"
  manifest_args+=(--manifest-jsonl "$manifest")
  if [ "$RESET_MANIFESTS" = "1" ] && [ "$RUN_INFER" = "1" ]; then
    rm -f "$manifest"
  fi
done

infer_one_shard() {
  local shard="$1"
  local gpu=$((shard % GPU_COUNT))
  local device="cuda:${gpu}"
  local manifest="$OUTPUT_DIR/manifest.shard${shard}.jsonl"
  local log="$OUTPUT_DIR/logs/infer.shard${shard}.log"
  local overwrite_args=()
  if [ "$OVERWRITE_INFER" = "1" ]; then
    overwrite_args+=(--overwrite)
  fi
  (
    if [ "${INFER_SHARD_START_DELAY_SEC}" != "0" ]; then
      delay=$(awk -v shard="$shard" -v step="$INFER_SHARD_START_DELAY_SEC" 'BEGIN { printf "%.3f", shard * step }')
      echo "[seedtts-eval] shard=$shard start_delay=${delay}s"
      sleep "$delay"
    fi
    export AUDIO_TEMPERATURE="$AUDIO_TEMPERATURE_VALUE"
    export AUDIO_TOP_P="$AUDIO_TOP_P_VALUE"
    export AUDIO_TOP_K="$AUDIO_TOP_K_VALUE"
    export AUDIO_REPETITION_PENALTY="$AUDIO_REPETITION_PENALTY_VALUE"
    export NO_TEXT_AUDIO_TEMPERATURE="$NO_TEXT_AUDIO_TEMPERATURE_VALUE"
    export NO_TEXT_AUDIO_TOP_P="$NO_TEXT_AUDIO_TOP_P_VALUE"
    export NO_TEXT_AUDIO_TOP_K="$NO_TEXT_AUDIO_TOP_K_VALUE"
    export NO_TEXT_AUDIO_REPETITION_PENALTY="$NO_TEXT_AUDIO_REPETITION_PENALTY_VALUE"
    export REF_PROMPT_CODEC_PERMUTATION
    export REF_PROMPT_CODEC_PERMUTATION_MIN_SECONDS
    export REF_PROMPT_CODEC_PERMUTATION_MAX_SECONDS
    export REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE
    export REF_PROMPT_CODEC_PERMUTATION_SEED
    export REF_PROMPT_CODEC_PERMUTATION_MODE
    export REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS
    export REF_PROMPT_CODEC_PERMUTATION_BOOTSTRAP
    export TIMBRE_CFG_SCALE
    if [ "$PERSISTENT_INFER" = "1" ]; then
      "$PYTHON" "$PERSISTENT_INFER_SCRIPT" \
        --validation-jsonl "$VALIDATION_JSONL" \
        --model-path "$MODEL_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --manifest-jsonl "$manifest" \
        --mode "$MODE" \
        --per-mode "$PER_MODE" \
        --per-cell "$PER_CELL" \
        --max-cases "$MAX_CASES" \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$shard" \
        --device "$device" \
        "${overwrite_args[@]}"
    else
      "$PYTHON" "$VALID_INFER_SCRIPT" \
        --validation-jsonl "$VALIDATION_JSONL" \
        --model-path "$MODEL_PATH" \
        --run-script "$RUN_SCRIPT" \
        --output-dir "$OUTPUT_DIR" \
        --manifest-jsonl "$manifest" \
        --mode "$MODE" \
        --per-mode "$PER_MODE" \
        --per-cell "$PER_CELL" \
        --max-cases "$MAX_CASES" \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$shard" \
        --device "$device" \
        --python "$PYTHON" \
        "${overwrite_args[@]}"
    fi
  ) >"$log" 2>&1
}

if [ "$RUN_INFER" = "1" ]; then
  pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    infer_one_shard "$shard" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ]; then
    echo "ERROR: one or more inference shards failed; see $OUTPUT_DIR/logs" >&2
    exit 1
  fi
else
  echo "[seedtts-eval] RUN_INFER=0, reuse existing manifests/audio."
fi

"$PYTHON" "$BUILD_EVAL_SCRIPT" \
  --validation-jsonl "$VALIDATION_JSONL" \
  --output-dir "$OUTPUT_DIR" \
  "${manifest_args[@]}" \
  --run-id "$RUN_ID" \
  --output-jsonl "$EVAL_INPUT_JSONL" \
  --status "ok,ok_after_rerun,skipped_exists"

run_asr_shard() {
  local shard="$1"
  local gpu=$((shard % GPU_COUNT))
  local device="${ASR_DEVICE_PREFIX}:${gpu}"
  local out="$OUTPUT_DIR/${RUN_ID}.asr_eval.shard${shard}.jsonl"
  local log="$OUTPUT_DIR/logs/asr.shard${shard}.log"
  "$ASR_PYTHON" "$ASR_SCRIPT" \
    --input-jsonl "$EVAL_INPUT_JSONL" \
    --output-jsonl "$out" \
    --asr-backend "$ASR_BACKEND" \
    --qwen-asr-model "$QWEN_ASR_MODEL" \
    --qwen-asr-dtype "$QWEN_ASR_DTYPE" \
    --qwen-asr-max-batch-size "$QWEN_ASR_MAX_BATCH_SIZE" \
    --qwen-asr-max-new-tokens "$QWEN_ASR_MAX_NEW_TOKENS" \
    --device "$device" \
    --content-reference-mode "$CONTENT_REFERENCE_MODE" \
    --skip-source-asr \
    --zh-cer-threshold "$ZH_CER_THRESHOLD" \
    --en-wer-threshold "$EN_WER_THRESHOLD" \
    --no-text-zh-cer-threshold "$NO_TEXT_ZH_CER_THRESHOLD" \
    --no-text-en-wer-threshold "$NO_TEXT_EN_WER_THRESHOLD" \
    --max-repeat-score "$MAX_REPEAT_SCORE" \
    --num-shards "$ASR_NUM_SHARDS" \
    --shard-index "$shard" \
    --progress-every 20 \
    --overwrite >"$log" 2>&1
}

if [ "$RUN_ASR" = "1" ]; then
  pids=()
  for shard in $(seq 0 $((ASR_NUM_SHARDS - 1))); do
    run_asr_shard "$shard" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ]; then
    echo "ERROR: one or more ASR shards failed; see $OUTPUT_DIR/logs" >&2
    exit 1
  fi
  "$PYTHON" - "$EVAL_INPUT_JSONL" "$ASR_JSONL" "$OUTPUT_DIR"/"$RUN_ID".asr_eval.shard*.jsonl <<'PY'
import json
import sys
from pathlib import Path

eval_input = Path(sys.argv[1])
out_path = Path(sys.argv[2])
shards = [Path(p) for p in sys.argv[3:]]
order = {}
for idx, line in enumerate(eval_input.read_text(encoding="utf-8").splitlines()):
    if not line.strip():
        continue
    row = json.loads(line)
    order[str(row.get("case_id") or row.get("sample_id") or idx)] = idx
rows = []
for shard in shards:
    if not shard.exists():
        continue
    for line in shard.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
rows.sort(key=lambda row: order.get(str(row.get("case_id") or row.get("sample_id") or ""), 10**12))
with out_path.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"[merge-asr] rows={len(rows)} output={out_path}")
PY
else
  echo "[seedtts-eval] RUN_ASR=0, skip ASR."
fi

if [ "$RUN_SUMMARY" = "1" ] && [ -f "$ASR_JSONL" ]; then
  "$PYTHON" "$SUMMARY_SCRIPT" \
    --asr-jsonl "$ASR_JSONL" \
    --metrics-csv "$METRICS_CSV" \
    --summary-md "$SUMMARY_MD" \
    --summary-json "$SUMMARY_JSON" \
    --run-id "$RUN_ID" \
    --run-label "$RUN_LABEL" \
    --model-path "$MODEL_PATH"
fi

if [ "$BUILD_PAGE" = "1" ]; then
  "$PYTHON" "$PAGE_SCRIPT" \
    --validation-jsonl "$VALIDATION_JSONL" \
    --output-dir "$OUTPUT_DIR" \
    --page-dir "$PAGE_DIR" \
    "${manifest_args[@]}" \
    --run-id "$RUN_ID" \
    --run-label "$RUN_LABEL" \
    --model-path "$MODEL_PATH" \
    --append
fi

echo "[seedtts-eval] done: $OUTPUT_DIR"

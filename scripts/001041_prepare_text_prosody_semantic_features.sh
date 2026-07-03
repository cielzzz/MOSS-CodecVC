#!/usr/bin/env bash
set -euo pipefail

# Text_prosody semantic-enhanced preprocessing.
# Input: train-ready text_prosody SFT JSONL with codec tokens, ECAPA paths and prosody paths.
# Output: train-ready JSONL with CTC content token ids and target HuBERT features.
# Text-mode target audio must match input text. By default we run target ASR
# against the provided text and keep only rows whose target ASR passes
# content_keep. This avoids training TextCTC on wrong teacher targets.

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
cd "$PROJECT_ROOT"

PYTHON_MAIN="${PYTHON_MAIN:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
PYTHON_ASR="${PYTHON_ASR:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download}"

DATASET_NAME="${DATASET_NAME:-zh3w_en3w_text_prosody_independent_timbre}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/trainset/$DATASET_NAME}"
SFT_DIR="${SFT_DIR:-$DATASET_ROOT/sft}"
INPUT_JSONL="${INPUT_JSONL:-$SFT_DIR/moss_codecvc_sft.$DATASET_NAME.with_light_ecapa_spk.with_prosody.jsonl}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$SFT_DIR/moss_codecvc_sft.$DATASET_NAME.with_light_ecapa_spk.with_prosody}"

RUN_P0="${RUN_P0:-1}"
RUN_P1="${RUN_P1:-1}"
RUN_P2="${RUN_P2:-1}"
RUN_PARALLEL_SEMANTIC_BRANCHES="${RUN_PARALLEL_SEMANTIC_BRANCHES:-1}"
FILTER_CONTENT_KEEP="${FILTER_CONTENT_KEEP:-1}"
CONTENT_REQUIRE_CONTENT_KEEP="${CONTENT_REQUIRE_CONTENT_KEEP:-1}"
OVERWRITE="${OVERWRITE:-0}"
FORCE="${FORCE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RESUME_SHARDS="${RESUME_SHARDS:-1}"
MAX_ROWS="${MAX_ROWS:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1000}"
WRITE_TRAIN_COMMAND="${WRITE_TRAIN_COMMAND:-1}"
DRY_RUN=0

ASR_JSONL="${ASR_JSONL:-$OUTPUT_PREFIX.with_asr_filter.jsonl}"
ASR_KEPT_JSONL="${ASR_KEPT_JSONL:-$OUTPUT_PREFIX.with_asr_filter.keep.jsonl}"
CONTENT_JSONL="${CONTENT_JSONL:-$OUTPUT_PREFIX.with_content_tokens.jsonl}"
HUBERT_JSONL="${HUBERT_JSONL:-$OUTPUT_PREFIX.with_target_hubert.jsonl}"
FINAL_JSONL="${FINAL_JSONL:-$OUTPUT_PREFIX.with_content_tokens.with_target_hubert.jsonl}"
FINAL_DONE_JSON="${FINAL_DONE_JSON:-$FINAL_JSONL.done.json}"
CONTENT_VOCAB_JSON="${CONTENT_VOCAB_JSON:-$SFT_DIR/content_ctc_char_vocab.json}"
HUBERT_FEATURE_ROOT="${HUBERT_FEATURE_ROOT:-$DATASET_ROOT/semantic_features/hubert_target}"
SEMANTIC_SHARD_ROOT="${SEMANTIC_SHARD_ROOT:-$DATASET_ROOT/shards_semantic_text_prosody}"
TRAIN_COMMAND_SH="${TRAIN_COMMAND_SH:-$SFT_DIR/train_ver2_1_semantic.$DATASET_NAME.sh}"

ASR_BACKEND="${ASR_BACKEND:-qwen_asr}"
QWEN_ASR_MODEL="${QWEN_ASR_MODEL:-$DOWNLOAD_ROOT/checkpoint/qwen-asr-1_7b}"
QWEN_ASR_DTYPE="${QWEN_ASR_DTYPE:-bfloat16}"
QWEN_ASR_MAX_BATCH_SIZE="${QWEN_ASR_MAX_BATCH_SIZE:-16}"
QWEN_ASR_MAX_NEW_TOKENS="${QWEN_ASR_MAX_NEW_TOKENS:-256}"
ASR_DEVICE="${ASR_DEVICE:-cuda:0}"
ASR_NUM_SHARDS="${ASR_NUM_SHARDS:-8}"
ASR_DEVICES="${ASR_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7}"
ASR_SHARD_DIR="${ASR_SHARD_DIR:-$ASR_JSONL.shards}"
ASR_MAP_JSONL="${ASR_MAP_JSONL:-}"
FASTER_WHISPER_MODEL="${FASTER_WHISPER_MODEL:-}"
WHISPER_MODEL="${WHISPER_MODEL:-small}"
LANGUAGE="${LANGUAGE:-}"
CONTENT_REFERENCE_MODE="${CONTENT_REFERENCE_MODE:-text}"
ASR_SKIP_SOURCE="${ASR_SKIP_SOURCE:-1}"
ASR_DISABLE_DURATION_RATIO_CHECK="${ASR_DISABLE_DURATION_RATIO_CHECK:-1}"
ZH_CER_THRESHOLD="${ZH_CER_THRESHOLD:-0.20}"
EN_WER_THRESHOLD="${EN_WER_THRESHOLD:-0.25}"
NO_TEXT_ZH_CER_THRESHOLD="${NO_TEXT_ZH_CER_THRESHOLD:-0.25}"
NO_TEXT_EN_WER_THRESHOLD="${NO_TEXT_EN_WER_THRESHOLD:-0.30}"
MAX_REPEAT_SCORE="${MAX_REPEAT_SCORE:-0.30}"
MIN_ASR_CHARS="${MIN_ASR_CHARS:-2}"
MIN_DURATION_RATIO="${MIN_DURATION_RATIO:-0.50}"
MAX_DURATION_RATIO="${MAX_DURATION_RATIO:-1.80}"
CONTENT_KEEP_MISSING_AS="${CONTENT_KEEP_MISSING_AS:-drop}"

CONTENT_TEXT_KEYS="${CONTENT_TEXT_KEYS:-content_ref_text,text,target_text}"
CONTENT_TOKENIZER="${CONTENT_TOKENIZER:-char}"
CONTENT_LOWERCASE_LATIN="${CONTENT_LOWERCASE_LATIN:-1}"
CONTENT_STRIP_WHITESPACE="${CONTENT_STRIP_WHITESPACE:-1}"
CONTENT_DROP_PUNCTUATION="${CONTENT_DROP_PUNCTUATION:-0}"
CONTENT_MIN_TOKEN_COUNT="${CONTENT_MIN_TOKEN_COUNT:-1}"

HUBERT_CACHE_DIR="${HUBERT_CACHE_DIR:-$DOWNLOAD_ROOT/huggingface}"
DEFAULT_HUBERT_MODEL="$HUBERT_CACHE_DIR/models--facebook--hubert-base-ls960/snapshots/dba3bb02fda4248b6e082697eee756de8fe8aa8a"
if [[ ! -f "$DEFAULT_HUBERT_MODEL/config.json" ]]; then
  DEFAULT_HUBERT_MODEL="facebook/hubert-base-ls960"
fi
HUBERT_MODEL="${HUBERT_MODEL:-$DEFAULT_HUBERT_MODEL}"
HUBERT_INPUT_JSONL="${HUBERT_INPUT_JSONL:-}"
HUBERT_DEVICE="${HUBERT_DEVICE:-cuda:0}"
HUBERT_DTYPE="${HUBERT_DTYPE:-auto}"
HUBERT_SAVE_DTYPE="${HUBERT_SAVE_DTYPE:-float16}"
HUBERT_LAYER="${HUBERT_LAYER:-9}"
HUBERT_DOWNSAMPLE_STRIDE="${HUBERT_DOWNSAMPLE_STRIDE:-1}"
HUBERT_USE_SAFETENSORS="${HUBERT_USE_SAFETENSORS:-false}"
HUBERT_LOCAL_FILES_ONLY="${HUBERT_LOCAL_FILES_ONLY:-1}"
HUBERT_SOURCE="${HUBERT_SOURCE:-target}"
HUBERT_NUM_SHARDS="${HUBERT_NUM_SHARDS:-8}"
HUBERT_DEVICES="${HUBERT_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7}"
HUBERT_SHARD_DIR="${HUBERT_SHARD_DIR:-$HUBERT_JSONL.shards}"
HUBERT_PRE_SPLIT="${HUBERT_PRE_SPLIT:-1}"
P2_DRY_RUN="${P2_DRY_RUN:-0}"

SEMANTIC_GPU_KEEPALIVE="${SEMANTIC_GPU_KEEPALIVE:-0}"
SEMANTIC_GPU_KEEPALIVE_GPU_IDS="${SEMANTIC_GPU_KEEPALIVE_GPU_IDS:-0,1,2,3,4,5,6,7}"
SEMANTIC_GPU_KEEPALIVE_LOG_DIR="${SEMANTIC_GPU_KEEPALIVE_LOG_DIR:-$SEMANTIC_SHARD_ROOT/gpu_keepalive}"
SEMANTIC_GPU_KEEPALIVE_MATMUL_SIZE="${SEMANTIC_GPU_KEEPALIVE_MATMUL_SIZE:-2048}"
SEMANTIC_GPU_KEEPALIVE_SLEEP_SEC="${SEMANTIC_GPU_KEEPALIVE_SLEEP_SEC:-0.05}"
SEMANTIC_GPU_KEEPALIVE_DTYPE="${SEMANTIC_GPU_KEEPALIVE_DTYPE:-float16}"
SEMANTIC_GPU_KEEPALIVE_PIDS=()

TRAIN_VERSION="${TRAIN_VERSION:-ver2}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-outputs/lora_runs/ver2_1_${DATASET_NAME}_text_prosody_semantic}"
TRAIN_TARGET_SPK_WEIGHT="${TRAIN_TARGET_SPK_WEIGHT:-0.05}"
TRAIN_SOURCE_SUPPRESS_WEIGHT="${TRAIN_SOURCE_SUPPRESS_WEIGHT:-0.05}"
TRAIN_SPEAKER_MARGIN="${TRAIN_SPEAKER_MARGIN:-0.1}"
TRAIN_LAMBDA_ROUTE="${TRAIN_LAMBDA_ROUTE:-0.01}"
TRAIN_LAMBDA_PROSODY="${TRAIN_LAMBDA_PROSODY:-0.05}"
TRAIN_PROSODY_F0_WEIGHT="${TRAIN_PROSODY_F0_WEIGHT:-0.0}"
TRAIN_PROSODY_VOICED_WEIGHT="${TRAIN_PROSODY_VOICED_WEIGHT:-0.0}"
TRAIN_PROSODY_ENERGY_WEIGHT="${TRAIN_PROSODY_ENERGY_WEIGHT:-0.5}"
TRAIN_PROSODY_PAUSE_WEIGHT="${TRAIN_PROSODY_PAUSE_WEIGHT:-1.0}"
TRAIN_PROSODY_DURATION_WEIGHT="${TRAIN_PROSODY_DURATION_WEIGHT:-0.5}"
TRAIN_CONTENT_CTC_WEIGHT="${TRAIN_CONTENT_CTC_WEIGHT:-0.05}"
TRAIN_CONTENT_CTC_VOCAB_SIZE="${TRAIN_CONTENT_CTC_VOCAB_SIZE:-0}"
TRAIN_CONTENT_CTC_BLANK_ID="${TRAIN_CONTENT_CTC_BLANK_ID:-0}"
TRAIN_CONTENT_CTC_TOKEN_OFFSET="${TRAIN_CONTENT_CTC_TOKEN_OFFSET:-0}"
TRAIN_SEMANTIC_LOSS_WEIGHT="${TRAIN_SEMANTIC_LOSS_WEIGHT:-0.05}"
TRAIN_SEMANTIC_MODE="${TRAIN_SEMANTIC_MODE:-continuous}"
TRAIN_SEMANTIC_SOURCE="${TRAIN_SEMANTIC_SOURCE:-target}"
TRAIN_SEMANTIC_FEATURE_DIM="${TRAIN_SEMANTIC_FEATURE_DIM:-0}"
TRAIN_SEMANTIC_FEATURE_LOSS_TYPE="${TRAIN_SEMANTIC_FEATURE_LOSS_TYPE:-cosine}"

for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [ "$DRY_RUN" -eq 0 ]; then
    "$@"
  fi
}

should_reuse() {
  local path="$1"
  if truthy "$SKIP_EXISTING" && [ "$FORCE" != "1" ] && [ -s "$path" ]; then
    return 0
  fi
  return 1
}

pick_csv_item() {
  local csv="$1"
  local index="$2"
  IFS=',' read -r -a items <<< "$csv"
  if [ "${#items[@]}" -eq 0 ]; then
    echo ""
    return 1
  fi
  echo "${items[$((index % ${#items[@]}))]}"
}

wait_for_pids() {
  local failed=0
  local pid
  for pid in "$@"; do
    if [ "$pid" -eq 0 ]; then
      continue
    fi
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ]; then
    echo "At least one background process failed." >&2
    exit 1
  fi
}

start_semantic_gpu_keepalive() {
  if ! truthy "$SEMANTIC_GPU_KEEPALIVE"; then
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[semantic-gpu-keepalive] dry-run"
    return 0
  fi
  if [ "${#SEMANTIC_GPU_KEEPALIVE_PIDS[@]}" -gt 0 ]; then
    return 0
  fi
  mkdir -p "$SEMANTIC_GPU_KEEPALIVE_LOG_DIR"
  IFS=',' read -r -a keepalive_gpu_ids <<< "$SEMANTIC_GPU_KEEPALIVE_GPU_IDS"
  local raw_gpu_id gpu_id worker_index log_file
  worker_index=0
  for raw_gpu_id in "${keepalive_gpu_ids[@]}"; do
    gpu_id="${raw_gpu_id//[[:space:]]/}"
    if [ -z "$gpu_id" ]; then
      continue
    fi
    log_file=$(printf "%s/text_prosody_semantic_gpu%s_w%02d.log" "$SEMANTIC_GPU_KEEPALIVE_LOG_DIR" "$gpu_id" "$worker_index")
    "$PYTHON_MAIN" - "$gpu_id" "$SEMANTIC_GPU_KEEPALIVE_MATMUL_SIZE" "$SEMANTIC_GPU_KEEPALIVE_SLEEP_SEC" "$SEMANTIC_GPU_KEEPALIVE_DTYPE" >"$log_file" 2>&1 <<'PY' &
from __future__ import annotations

import sys
import time

gpu_id = int(sys.argv[1])
size = int(sys.argv[2])
sleep_sec = float(sys.argv[3])
dtype_name = sys.argv[4]
try:
    import torch
except Exception as exc:
    print(f"[semantic-gpu-keepalive] torch import failed: {type(exc).__name__}: {exc}", flush=True)
    while True:
        time.sleep(60)
if not torch.cuda.is_available():
    print("[semantic-gpu-keepalive] cuda unavailable", flush=True)
    while True:
        time.sleep(60)
dtype = torch.float16 if dtype_name in {"float16", "fp16", "half"} else torch.float32
device = torch.device(f"cuda:{gpu_id}")
torch.cuda.set_device(device)
actual_size = size
while actual_size >= 512:
    try:
        a = torch.randn((actual_size, actual_size), device=device, dtype=dtype)
        b = torch.randn((actual_size, actual_size), device=device, dtype=dtype)
        out = torch.empty((actual_size, actual_size), device=device, dtype=dtype)
        break
    except RuntimeError as exc:
        print(f"[semantic-gpu-keepalive] allocation failed gpu={gpu_id} size={actual_size}: {type(exc).__name__}: {exc}", flush=True)
        torch.cuda.empty_cache()
        actual_size //= 2
else:
    print(f"[semantic-gpu-keepalive] no usable allocation gpu={gpu_id}; sleeping only", flush=True)
    while True:
        time.sleep(60)
print(f"[semantic-gpu-keepalive] running gpu={gpu_id} size={actual_size} dtype={dtype} sleep={sleep_sec}", flush=True)
step = 0
while True:
    torch.matmul(a, b, out=out)
    a.add_(out.mean() * 0.0)
    torch.cuda.synchronize(device)
    step += 1
    if step % 100 == 0:
        print(f"[semantic-gpu-keepalive] gpu={gpu_id} step={step}", flush=True)
    if sleep_sec > 0:
        time.sleep(sleep_sec)
PY
    local pid="$!"
    SEMANTIC_GPU_KEEPALIVE_PIDS+=("$pid")
    echo "[semantic-gpu-keepalive] started gpu=$gpu_id pid=$pid log=$log_file"
    worker_index=$((worker_index + 1))
  done
}

stop_semantic_gpu_keepalive() {
  local pid
  if [ "${#SEMANTIC_GPU_KEEPALIVE_PIDS[@]}" -eq 0 ]; then
    return 0
  fi
  for pid in "${SEMANTIC_GPU_KEEPALIVE_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  SEMANTIC_GPU_KEEPALIVE_PIDS=()
}

trap 'stop_semantic_gpu_keepalive' EXIT
trap 'stop_semantic_gpu_keepalive; exit 130' INT TERM

merge_jsonl_shards() {
  local shard_dir="$1"
  local output_jsonl="$2"
  local num_shards="$3"
  local tmp_path="$output_jsonl.tmp"
  : > "$tmp_path"
  local i
  for i in $(seq 0 $((num_shards - 1))); do
    local shard_file
    shard_file=$(printf "%s/shard_%05d.jsonl" "$shard_dir" "$i")
    if [[ ! -f "$shard_file" ]]; then
      echo "ERROR: missing shard output: $shard_file" >&2
      return 1
    fi
    cat "$shard_file" >> "$tmp_path"
  done
  mv "$tmp_path" "$output_jsonl"
}

merge_summary_shards() {
  local shard_dir="$1"
  local output_jsonl="$2"
  local num_shards="$3"
  "$PYTHON_MAIN" - "$shard_dir" "$output_jsonl" "$num_shards" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

shard_dir = Path(sys.argv[1])
output_jsonl = Path(sys.argv[2])
num_shards = int(sys.argv[3])
stats = Counter()
rows = 0
summaries = []
for i in range(num_shards):
    shard_jsonl = shard_dir / f"shard_{i:05d}.jsonl"
    shard_summary = shard_jsonl.with_suffix(shard_jsonl.suffix + ".summary.json")
    payload = {}
    if shard_summary.exists():
        payload = json.loads(shard_summary.read_text(encoding="utf-8"))
    shard_rows = int(payload.get("rows", payload.get("stats", {}).get("rows", 0)))
    rows += shard_rows
    stats.update(payload.get("stats", {}))
    summaries.append({"shard": i, "jsonl": str(shard_jsonl), "summary": str(shard_summary), "rows": shard_rows})
out = {"rows": rows, "num_shards": num_shards, "shards": summaries, "stats": dict(stats)}
summary_path = output_jsonl.with_suffix(output_jsonl.suffix + ".summary.json")
summary_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
done_path = output_jsonl.with_name(output_jsonl.name + ".done.json")
done_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[shard-merge] output={output_jsonl} rows={rows} summary={summary_path}", flush=True)
PY
}

split_hubert_jsonl_shards() {
  local input_jsonl="$1"
  local out_dir="$2"
  local shard_count="$3"
  local max_rows="$4"
  local marker="$out_dir/.split_${shard_count}_${max_rows}.done"
  if truthy "$SKIP_EXISTING" && [ "$FORCE" != "1" ] && [ -s "$marker" ]; then
    echo "[text-prosody-semantic] reuse P2 pre-split inputs: $out_dir"
    return 0
  fi
  echo "[text-prosody-semantic] pre-split P2 input=$input_jsonl -> $out_dir shards=$shard_count max_rows=$max_rows"
  if [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  rm -rf "$out_dir"
  mkdir -p "$out_dir"
  "$PYTHON_MAIN" - "$input_jsonl" "$out_dir" "$shard_count" "$max_rows" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

input_jsonl = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
shard_count = int(sys.argv[3])
max_rows = int(sys.argv[4])
handles = []
try:
    for idx in range(shard_count):
        handles.append((out_dir / f"input_shard_{idx:05d}.jsonl").open("w", encoding="utf-8"))
    with input_jsonl.open("r", encoding="utf-8") as src:
        for line_idx, line in enumerate(src):
            if max_rows > 0 and line_idx >= max_rows:
                break
            handles[line_idx % shard_count].write(line)
finally:
    for handle in handles:
        handle.close()
PY
  {
    printf 'input_jsonl=%s\n' "$input_jsonl"
    printf 'shard_count=%s\n' "$shard_count"
    printf 'max_rows=%s\n' "$max_rows"
  } > "$marker"
}

overwrite_flag=()
if [[ "$OVERWRITE" == "1" || "$FORCE" == "1" ]]; then
  overwrite_flag+=(--overwrite)
fi

resume_flag=()
if [[ "$RESUME_SHARDS" == "1" ]]; then
  resume_flag+=(--resume)
fi

limit_flag=()
max_rows_flag=()
if [[ "$MAX_ROWS" != "0" ]]; then
  limit_flag+=(--limit "$MAX_ROWS")
  max_rows_flag+=(--max-rows "$MAX_ROWS")
fi

language_flag=()
if [[ -n "$LANGUAGE" ]]; then
  language_flag+=(--language "$LANGUAGE")
fi

content_require_keep_flag=(--require-content-keep)
if [[ "$CONTENT_REQUIRE_CONTENT_KEEP" == "0" ]]; then
  content_require_keep_flag=(--no-require-content-keep)
fi

content_case_flag=(--lowercase-latin)
if [[ "$CONTENT_LOWERCASE_LATIN" == "0" ]]; then
  content_case_flag=(--no-lowercase-latin)
fi

content_whitespace_flag=(--strip-whitespace)
if [[ "$CONTENT_STRIP_WHITESPACE" == "0" ]]; then
  content_whitespace_flag=(--no-strip-whitespace)
fi

content_punct_flag=(--no-drop-punctuation)
if [[ "$CONTENT_DROP_PUNCTUATION" == "1" ]]; then
  content_punct_flag=(--drop-punctuation)
fi

run_asr_filter() {
  if [[ "$RUN_P0" != "1" ]]; then
    echo "[text-prosody-semantic] skip P0 ASR; ASR_JSONL=$ASR_JSONL"
    return 0
  fi
  if should_reuse "$ASR_JSONL"; then
    echo "[text-prosody-semantic] reuse P0 ASR output: $ASR_JSONL"
    return 0
  fi
  echo "[text-prosody-semantic] P0 target ASR vs target_text backend=$ASR_BACKEND shards=$ASR_NUM_SHARDS"
  asr_args=(
    scripts/001012_filter_teacher_content_quality.py
    --input-jsonl "$INPUT_JSONL"
    --asr-backend "$ASR_BACKEND"
    --content-reference-mode "$CONTENT_REFERENCE_MODE"
    --zh-cer-threshold "$ZH_CER_THRESHOLD"
    --en-wer-threshold "$EN_WER_THRESHOLD"
    --no-text-zh-cer-threshold "$NO_TEXT_ZH_CER_THRESHOLD"
    --no-text-en-wer-threshold "$NO_TEXT_EN_WER_THRESHOLD"
    --max-repeat-score "$MAX_REPEAT_SCORE"
    --min-asr-chars "$MIN_ASR_CHARS"
    --min-duration-ratio "$MIN_DURATION_RATIO"
    --max-duration-ratio "$MAX_DURATION_RATIO"
    --progress-every "$PROGRESS_EVERY"
    "${limit_flag[@]}"
    "${language_flag[@]}"
    "${overwrite_flag[@]}"
    "${resume_flag[@]}"
  )
  if [[ "$ASR_SKIP_SOURCE" == "1" ]]; then
    asr_args+=(--skip-source-asr)
  fi
  if [[ "$ASR_DISABLE_DURATION_RATIO_CHECK" == "1" ]]; then
    asr_args+=(--disable-duration-ratio-check)
  fi
  case "$ASR_BACKEND" in
    qwen_asr)
      asr_args+=(
        --qwen-asr-model "$QWEN_ASR_MODEL"
        --qwen-asr-dtype "$QWEN_ASR_DTYPE"
        --qwen-asr-max-batch-size "$QWEN_ASR_MAX_BATCH_SIZE"
        --qwen-asr-max-new-tokens "$QWEN_ASR_MAX_NEW_TOKENS"
      )
      ;;
    jsonl_map)
      asr_args+=(--asr-map-jsonl "$ASR_MAP_JSONL")
      ;;
    faster_whisper)
      asr_args+=(--faster-whisper-model "$FASTER_WHISPER_MODEL")
      ;;
    whisper)
      asr_args+=(--whisper-model "$WHISPER_MODEL")
      ;;
  esac

  if [[ "$ASR_NUM_SHARDS" -le 1 ]]; then
    run_cmd "$PYTHON_ASR" "${asr_args[@]}" \
      --output-jsonl "$ASR_JSONL" \
      --device "$ASR_DEVICE"
  else
    if [[ "$OVERWRITE" == "1" || "$FORCE" == "1" ]]; then
      rm -rf "$ASR_SHARD_DIR"
      rm -f "$ASR_JSONL" "$ASR_JSONL.tmp" "$ASR_JSONL.summary.json" "$ASR_JSONL.done.json"
    fi
    mkdir -p "$ASR_SHARD_DIR"
    pids=()
    for shard in $(seq 0 $((ASR_NUM_SHARDS - 1))); do
      shard_out=$(printf "%s/shard_%05d.jsonl" "$ASR_SHARD_DIR" "$shard")
      shard_log=$(printf "%s/shard_%05d.log" "$ASR_SHARD_DIR" "$shard")
      shard_device="$(pick_csv_item "$ASR_DEVICES" "$shard")"
      echo "[text-prosody-semantic] launch P0 shard=$shard/$ASR_NUM_SHARDS device=$shard_device out=$shard_out"
      if [ "$DRY_RUN" -eq 0 ]; then
        (
          "$PYTHON_ASR" "${asr_args[@]}" \
            --output-jsonl "$shard_out" \
            --device "$shard_device" \
            --num-shards "$ASR_NUM_SHARDS" \
            --shard-index "$shard"
        ) >"$shard_log" 2>&1 &
        pids+=("$!")
      else
        pids+=(0)
      fi
    done
    wait_for_pids "${pids[@]}"
    if [ "$DRY_RUN" -eq 0 ]; then
      merge_jsonl_shards "$ASR_SHARD_DIR" "$ASR_JSONL" "$ASR_NUM_SHARDS"
      merge_summary_shards "$ASR_SHARD_DIR" "$ASR_JSONL" "$ASR_NUM_SHARDS"
    fi
  fi
}

content_source_jsonl() {
  if [[ "$RUN_P0" == "1" || -s "$ASR_JSONL" ]]; then
    echo "$ASR_JSONL"
  else
    echo "$INPUT_JSONL"
  fi
}

run_content_tokens() {
  local p1_source
  p1_source="$(content_source_jsonl)"
  if [[ "$RUN_P1" != "1" ]]; then
    echo "[text-prosody-semantic] skip P1 CTC tokens; CONTENT_JSONL=$CONTENT_JSONL"
    return 0
  fi
  if should_reuse "$CONTENT_JSONL"; then
    echo "[text-prosody-semantic] reuse P1 content token output: $CONTENT_JSONL"
    return 0
  fi
  if [[ "$FILTER_CONTENT_KEEP" == "1" ]]; then
    if should_reuse "$ASR_KEPT_JSONL"; then
      echo "[text-prosody-semantic] reuse content_keep filtered output: $ASR_KEPT_JSONL"
    else
      echo "[text-prosody-semantic] filter content_keep=true rows"
      run_cmd "$PYTHON_MAIN" scripts/001021_filter_content_keep_jsonl.py \
        --input-jsonl "$p1_source" \
        --output-jsonl "$ASR_KEPT_JSONL" \
        --missing-as "$CONTENT_KEEP_MISSING_AS" \
        --progress-every "$PROGRESS_EVERY" \
        "${max_rows_flag[@]}" \
        "${overwrite_flag[@]}"
    fi
    p1_source="$ASR_KEPT_JSONL"
  fi
  echo "[text-prosody-semantic] P1 content_ref_text -> content_token_ids input=$p1_source"
  run_cmd "$PYTHON_MAIN" scripts/001019_extract_content_tokens.py \
    --input-jsonl "$p1_source" \
    --output-jsonl "$CONTENT_JSONL" \
    --vocab-json "$CONTENT_VOCAB_JSON" \
    --text-keys "$CONTENT_TEXT_KEYS" \
    --tokenizer "$CONTENT_TOKENIZER" \
    --min-token-count "$CONTENT_MIN_TOKEN_COUNT" \
    --progress-every "$PROGRESS_EVERY" \
    "${content_require_keep_flag[@]}" \
    "${content_case_flag[@]}" \
    "${content_whitespace_flag[@]}" \
    "${content_punct_flag[@]}" \
    "${max_rows_flag[@]}" \
    "${overwrite_flag[@]}"
}

pick_hubert_input_jsonl() {
  if [[ -n "$HUBERT_INPUT_JSONL" ]]; then
    echo "$HUBERT_INPUT_JSONL"
  elif truthy "$RUN_PARALLEL_SEMANTIC_BRANCHES"; then
    echo "$INPUT_JSONL"
  elif [[ -s "$CONTENT_JSONL" ]]; then
    echo "$CONTENT_JSONL"
  else
    echo "$INPUT_JSONL"
  fi
}

run_hubert_features() {
  if [[ "$RUN_P2" != "1" ]]; then
    echo "[text-prosody-semantic] skip P2 HuBERT; HUBERT_JSONL=$HUBERT_JSONL"
    return 0
  fi
  if should_reuse "$HUBERT_JSONL"; then
    echo "[text-prosody-semantic] reuse P2 HuBERT output: $HUBERT_JSONL"
    return 0
  fi
  local p2_source
  p2_source="$(pick_hubert_input_jsonl)"
  echo "[text-prosody-semantic] P2 $HUBERT_SOURCE wav -> HuBERT features input=$p2_source shards=$HUBERT_NUM_SHARDS"
  p2_args=(
    scripts/001020_extract_hubert_semantic_features.py
    --feature-root "$HUBERT_FEATURE_ROOT"
    --extractor hubert
    --model-name-or-path "$HUBERT_MODEL"
    --cache-dir "$HUBERT_CACHE_DIR"
    --source "$HUBERT_SOURCE"
    --layer "$HUBERT_LAYER"
    --dtype "$HUBERT_DTYPE"
    --save-dtype "$HUBERT_SAVE_DTYPE"
    --downsample-stride "$HUBERT_DOWNSAMPLE_STRIDE"
    --use-safetensors "$HUBERT_USE_SAFETENSORS"
    --progress-every "$PROGRESS_EVERY"
    "${overwrite_flag[@]}"
  )
  if [[ "$HUBERT_NUM_SHARDS" -le 1 || "$HUBERT_PRE_SPLIT" != "1" ]]; then
    p2_args+=("${max_rows_flag[@]}")
  fi
  if [[ "$HUBERT_LOCAL_FILES_ONLY" == "1" ]]; then
    p2_args+=(--local-files-only)
  fi
  if [[ "$P2_DRY_RUN" == "1" ]]; then
    p2_args+=(--dry-run)
  fi

  if [[ "$HUBERT_NUM_SHARDS" -le 1 ]]; then
    run_cmd "$PYTHON_MAIN" "${p2_args[@]}" \
      --input-jsonl "$p2_source" \
      --output-jsonl "$HUBERT_JSONL" \
      --device "$HUBERT_DEVICE"
  else
    if [[ "$OVERWRITE" == "1" || "$FORCE" == "1" ]]; then
      rm -rf "$HUBERT_SHARD_DIR"
      rm -f "$HUBERT_JSONL" "$HUBERT_JSONL.tmp" "$HUBERT_JSONL.summary.json" "$HUBERT_JSONL.done.json"
    fi
    mkdir -p "$HUBERT_SHARD_DIR"
    local hubert_split_dir="$HUBERT_SHARD_DIR/inputs"
    if [[ "$HUBERT_PRE_SPLIT" == "1" ]]; then
      split_hubert_jsonl_shards "$p2_source" "$hubert_split_dir" "$HUBERT_NUM_SHARDS" "$MAX_ROWS"
    fi
    pids=()
    for shard in $(seq 0 $((HUBERT_NUM_SHARDS - 1))); do
      shard_out=$(printf "%s/shard_%05d.jsonl" "$HUBERT_SHARD_DIR" "$shard")
      shard_log=$(printf "%s/shard_%05d.log" "$HUBERT_SHARD_DIR" "$shard")
      shard_device="$(pick_csv_item "$HUBERT_DEVICES" "$shard")"
      shard_input="$p2_source"
      shard_num_shards="$HUBERT_NUM_SHARDS"
      shard_index="$shard"
      if [[ "$HUBERT_PRE_SPLIT" == "1" ]]; then
        shard_input=$(printf "%s/input_shard_%05d.jsonl" "$hubert_split_dir" "$shard")
        shard_num_shards=1
        shard_index=0
      fi
      echo "[text-prosody-semantic] launch P2 shard=$shard/$HUBERT_NUM_SHARDS device=$shard_device input=$shard_input out=$shard_out"
      if [ "$DRY_RUN" -eq 0 ]; then
        (
          "$PYTHON_MAIN" "${p2_args[@]}" \
            --input-jsonl "$shard_input" \
            --output-jsonl "$shard_out" \
            --device "$shard_device" \
            --num-shards "$shard_num_shards" \
            --shard-index "$shard_index"
        ) >"$shard_log" 2>&1 &
        pids+=("$!")
      else
        pids+=(0)
      fi
    done
    wait_for_pids "${pids[@]}"
    if [ "$DRY_RUN" -eq 0 ]; then
      merge_jsonl_shards "$HUBERT_SHARD_DIR" "$HUBERT_JSONL" "$HUBERT_NUM_SHARDS"
      merge_summary_shards "$HUBERT_SHARD_DIR" "$HUBERT_JSONL" "$HUBERT_NUM_SHARDS"
    fi
  fi
}

merge_final_jsonl() {
  if should_reuse "$FINAL_JSONL"; then
    echo "[text-prosody-semantic] reuse final output: $FINAL_JSONL"
    return 0
  fi
  local base_jsonl="$INPUT_JSONL"
  if [[ -s "$CONTENT_JSONL" ]]; then
    base_jsonl="$CONTENT_JSONL"
  elif [[ -s "$ASR_JSONL" ]]; then
    base_jsonl="$ASR_JSONL"
  fi
  local hubert_jsonl=""
  if [[ "$RUN_P2" == "1" && -s "$HUBERT_JSONL" ]]; then
    hubert_jsonl="$HUBERT_JSONL"
  fi
  echo "[text-prosody-semantic] merge final base=$base_jsonl hubert=${hubert_jsonl:-none} -> $FINAL_JSONL"
  run_cmd "$PYTHON_MAIN" - "$base_jsonl" "$hubert_jsonl" "$FINAL_JSONL" "$FINAL_DONE_JSON" "$DATASET_NAME" "$HUBERT_SOURCE" <<'PY'
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

base_path = Path(sys.argv[1])
hubert_arg = sys.argv[2]
final_path = Path(sys.argv[3])
done_path = Path(sys.argv[4])
dataset_name = sys.argv[5]
hubert_source = sys.argv[6]
hubert_path = Path(hubert_arg) if hubert_arg else None

semantic_keys = (
    "source_semantic_features_path",
    "source_hubert_features_path",
    "source_semantic_feature_dim",
    "target_semantic_features_path",
    "target_hubert_features_path",
    "target_semantic_feature_dim",
)

hubert_by_sample: dict[str, dict[str, object]] = {}
if hubert_path and hubert_path.exists():
    with hubert_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or "")
            if not sample_id:
                continue
            item = {key: row[key] for key in semantic_keys if key in row and row[key] not in (None, "")}
            if item:
                hubert_by_sample[sample_id] = item

final_path.parent.mkdir(parents=True, exist_ok=True)
stats: Counter[str] = Counter()
with base_path.open("r", encoding="utf-8") as src, final_path.open("w", encoding="utf-8") as dst:
    for line in src:
        row = json.loads(line)
        sample_id = str(row.get("sample_id") or "")
        item = hubert_by_sample.get(sample_id)
        if item is None and sample_id.endswith(":text"):
            item = hubert_by_sample.get(sample_id[: -len(":text")])
        if item:
            row.update(item)
            stats["semantic_attached"] += 1
        elif hubert_by_sample:
            stats["missing_semantic"] += 1
        language = str(row.get("language") or row.get("moss_codecvc_meta", {}).get("language") or "")
        if language:
            stats[f"language:{language}"] += 1
        stats["rows"] += 1
        dst.write(json.dumps(row, ensure_ascii=False) + "\n")

payload = {
    "status": "complete",
    "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "dataset_name": dataset_name,
    "base_jsonl": str(base_path.resolve()),
    "hubert_jsonl": str(hubert_path.resolve()) if hubert_path else "",
    "output_jsonl": str(final_path.resolve()),
    "hubert_source": hubert_source,
    "stats": dict(stats),
}
done_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
final_path.with_suffix(final_path.suffix + ".summary.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
if hubert_by_sample and stats.get("missing_semantic", 0):
    raise SystemExit(f"missing semantic features for {stats['missing_semantic']} rows")
PY
}

write_train_command() {
  if [[ "$WRITE_TRAIN_COMMAND" != "1" || "$DRY_RUN" -ne 0 ]]; then
    return 0
  fi
  mkdir -p "$(dirname "$TRAIN_COMMAND_SH")"
  cat > "$TRAIN_COMMAND_SH" <<EOF
#!/usr/bin/env bash
set -euo pipefail

PY="\${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
cd "$PROJECT_ROOT"

"\$PY" scripts/002002_train_moss_codecvc_lora.py \\
  --config configs/remote_full.yaml \\
  --train-jsonl "$FINAL_JSONL" \\
  --output-dir "$TRAIN_OUTPUT_DIR" \\
  --version "$TRAIN_VERSION" \\
  --use-timbre-memory \\
  --speaker-encoder-type embedding_loader \\
  --speaker-embedding-dim 192 \\
  --target-speaker-similarity-weight "$TRAIN_TARGET_SPK_WEIGHT" \\
  --source-speaker-suppression-weight "$TRAIN_SOURCE_SUPPRESS_WEIGHT" \\
  --speaker-loss-margin "$TRAIN_SPEAKER_MARGIN" \\
  --lambda-route "$TRAIN_LAMBDA_ROUTE" \\
  --lambda-prosody "$TRAIN_LAMBDA_PROSODY" \\
  --prosody-f0-weight "$TRAIN_PROSODY_F0_WEIGHT" \\
  --prosody-voiced-weight "$TRAIN_PROSODY_VOICED_WEIGHT" \\
  --prosody-energy-weight "$TRAIN_PROSODY_ENERGY_WEIGHT" \\
  --prosody-pause-weight "$TRAIN_PROSODY_PAUSE_WEIGHT" \\
  --prosody-duration-weight "$TRAIN_PROSODY_DURATION_WEIGHT" \\
  --content-ctc-weight "$TRAIN_CONTENT_CTC_WEIGHT" \\
  --content-ctc-vocab-size "$TRAIN_CONTENT_CTC_VOCAB_SIZE" \\
  --content-ctc-blank-id "$TRAIN_CONTENT_CTC_BLANK_ID" \\
  --content-ctc-token-offset "$TRAIN_CONTENT_CTC_TOKEN_OFFSET" \\
  --semantic-loss-weight "$TRAIN_SEMANTIC_LOSS_WEIGHT" \\
  --semantic-mode "$TRAIN_SEMANTIC_MODE" \\
  --semantic-source "$TRAIN_SEMANTIC_SOURCE" \\
  --semantic-feature-dim "$TRAIN_SEMANTIC_FEATURE_DIM" \\
  --semantic-feature-loss-type "$TRAIN_SEMANTIC_FEATURE_LOSS_TYPE" \\
  --learning-rate 1e-5 \\
  --per-device-batch-size 1 \\
  --gradient-accumulation-steps 8 \\
  --mixed-precision bf16
EOF
  chmod +x "$TRAIN_COMMAND_SH"
  echo "[text-prosody-semantic] train command: $TRAIN_COMMAND_SH"
}

mkdir -p \
  "$(dirname "$ASR_JSONL")" \
  "$(dirname "$CONTENT_JSONL")" \
  "$(dirname "$HUBERT_JSONL")" \
  "$(dirname "$FINAL_JSONL")" \
  "$HUBERT_FEATURE_ROOT" \
  "$SEMANTIC_SHARD_ROOT"

echo "=========================================="
echo "Text_prosody semantic feature preprocessing"
echo "  PROJECT_ROOT=$PROJECT_ROOT"
echo "  DATASET_NAME=$DATASET_NAME"
echo "  INPUT_JSONL=$INPUT_JSONL"
echo "  ASR_JSONL=$ASR_JSONL"
echo "  CONTENT_JSONL=$CONTENT_JSONL"
echo "  HUBERT_JSONL=$HUBERT_JSONL"
echo "  FINAL_JSONL=$FINAL_JSONL"
echo "  RUN_P0=$RUN_P0 RUN_P1=$RUN_P1 RUN_P2=$RUN_P2 RUN_PARALLEL_SEMANTIC_BRANCHES=$RUN_PARALLEL_SEMANTIC_BRANCHES"
echo "  ASR_BACKEND=$ASR_BACKEND ASR_NUM_SHARDS=$ASR_NUM_SHARDS ASR_DEVICES=$ASR_DEVICES"
echo "  CONTENT_REFERENCE_MODE=$CONTENT_REFERENCE_MODE ASR_SKIP_SOURCE=$ASR_SKIP_SOURCE ASR_DISABLE_DURATION_RATIO_CHECK=$ASR_DISABLE_DURATION_RATIO_CHECK"
echo "  HUBERT_SOURCE=$HUBERT_SOURCE HUBERT_NUM_SHARDS=$HUBERT_NUM_SHARDS HUBERT_DEVICES=$HUBERT_DEVICES HUBERT_PRE_SPLIT=$HUBERT_PRE_SPLIT"
echo "  FILTER_CONTENT_KEEP=$FILTER_CONTENT_KEEP CONTENT_REQUIRE_CONTENT_KEEP=$CONTENT_REQUIRE_CONTENT_KEEP"
echo "  SEMANTIC_GPU_KEEPALIVE=$SEMANTIC_GPU_KEEPALIVE SEMANTIC_GPU_KEEPALIVE_GPU_IDS=$SEMANTIC_GPU_KEEPALIVE_GPU_IDS"
echo "  FORCE=$FORCE SKIP_EXISTING=$SKIP_EXISTING OVERWRITE=$OVERWRITE DRY_RUN=$DRY_RUN"
echo "=========================================="

if [ "$DRY_RUN" -eq 0 ] && [ ! -s "$INPUT_JSONL" ]; then
  echo "Input JSONL is missing or empty: $INPUT_JSONL" >&2
  exit 2
fi

run_cmd "$PYTHON_MAIN" -m py_compile \
  scripts/001012_filter_teacher_content_quality.py \
  scripts/001017_asr_content_filter.py \
  scripts/001019_extract_content_tokens.py \
  scripts/001020_extract_hubert_semantic_features.py

start_semantic_gpu_keepalive

if truthy "$RUN_PARALLEL_SEMANTIC_BRANCHES" && [[ "$RUN_P2" == "1" ]] && [[ "$RUN_P0" == "1" || "$RUN_P1" == "1" ]]; then
  mkdir -p "$SEMANTIC_SHARD_ROOT/logs"
  content_log="$SEMANTIC_SHARD_ROOT/logs/content_branch.log"
  hubert_log="$SEMANTIC_SHARD_ROOT/logs/hubert_branch.log"
  echo "[text-prosody-semantic] launch content and HuBERT branches in parallel"
  if [ "$DRY_RUN" -eq 0 ]; then
    ( run_asr_filter; run_content_tokens ) >"$content_log" 2>&1 &
    content_pid=$!
    ( run_hubert_features ) >"$hubert_log" 2>&1 &
    hubert_pid=$!
    wait_for_pids "$content_pid" "$hubert_pid"
  else
    run_asr_filter
    run_content_tokens
    run_hubert_features
  fi
else
  run_asr_filter
  run_content_tokens
  run_hubert_features
fi

merge_final_jsonl
write_train_command

echo "=========================================="
echo "Text_prosody semantic preprocessing finished"
echo "  train_jsonl=$FINAL_JSONL"
echo "  done_json=$FINAL_DONE_JSON"
echo "  train_command=$TRAIN_COMMAND_SH"
echo "=========================================="

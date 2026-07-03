#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
cd "$PROJECT_ROOT"

PYTHON_MAIN="${PYTHON_MAIN:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
PYTHON_ASR="${PYTHON_ASR:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download}"

TRAINSET_DIR="${TRAINSET_DIR:-$PROJECT_ROOT/trainset/zh45w_en22w_no_text}"
SFT_DIR="${SFT_DIR:-$TRAINSET_DIR/sft}"
INPUT_JSONL="${INPUT_JSONL:-$SFT_DIR/moss_codecvc_sft.zh45w_en22w_no_text.with_light_ecapa_spk.with_prosody.jsonl}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$SFT_DIR/moss_codecvc_sft.zh45w_en22w_no_text.with_light_ecapa_spk.with_prosody}"

ASR_JSONL="${ASR_JSONL:-$OUTPUT_PREFIX.with_asr_filter.jsonl}"
ASR_KEPT_JSONL="${ASR_KEPT_JSONL:-$OUTPUT_PREFIX.with_asr_filter.keep.jsonl}"
CONTENT_JSONL="${CONTENT_JSONL:-$OUTPUT_PREFIX.with_asr_filter.with_content_tokens.jsonl}"
FINAL_JSONL="${FINAL_JSONL:-$OUTPUT_PREFIX.with_asr_filter.with_content_tokens.with_hubert.jsonl}"
CONTENT_VOCAB_JSON="${CONTENT_VOCAB_JSON:-$SFT_DIR/content_ctc_char_vocab.json}"
HUBERT_FEATURE_ROOT="${HUBERT_FEATURE_ROOT:-$TRAINSET_DIR/semantic_features/hubert}"

RUN_P0="${RUN_P0:-1}"
FILTER_CONTENT_KEEP="${FILTER_CONTENT_KEEP:-1}"
RUN_P1="${RUN_P1:-1}"
RUN_P2="${RUN_P2:-1}"
OVERWRITE="${OVERWRITE:-0}"
RESUME_SHARDS="${RESUME_SHARDS:-0}"
MAX_ROWS="${MAX_ROWS:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1000}"

ASR_BACKEND="${ASR_BACKEND:-qwen_asr}"
QWEN_ASR_MODEL="${QWEN_ASR_MODEL:-$DOWNLOAD_ROOT/checkpoint/qwen-asr-1_7b}"
QWEN_ASR_DTYPE="${QWEN_ASR_DTYPE:-bfloat16}"
QWEN_ASR_MAX_BATCH_SIZE="${QWEN_ASR_MAX_BATCH_SIZE:-16}"
QWEN_ASR_MAX_NEW_TOKENS="${QWEN_ASR_MAX_NEW_TOKENS:-256}"
ASR_DEVICE="${ASR_DEVICE:-cuda:0}"
ASR_NUM_SHARDS="${ASR_NUM_SHARDS:-1}"
ASR_DEVICES="${ASR_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7}"
ASR_SHARD_PARALLELISM="${ASR_SHARD_PARALLELISM:-$ASR_NUM_SHARDS}"
ASR_SHARD_DIR="${ASR_SHARD_DIR:-$ASR_JSONL.shards}"
ASR_MAP_JSONL="${ASR_MAP_JSONL:-}"
FASTER_WHISPER_MODEL="${FASTER_WHISPER_MODEL:-}"
WHISPER_MODEL="${WHISPER_MODEL:-small}"
LANGUAGE="${LANGUAGE:-}"
ZH_CER_THRESHOLD="${ZH_CER_THRESHOLD:-0.20}"
EN_WER_THRESHOLD="${EN_WER_THRESHOLD:-0.25}"
NO_TEXT_ZH_CER_THRESHOLD="${NO_TEXT_ZH_CER_THRESHOLD:-0.25}"
NO_TEXT_EN_WER_THRESHOLD="${NO_TEXT_EN_WER_THRESHOLD:-0.30}"
MAX_REPEAT_SCORE="${MAX_REPEAT_SCORE:-0.30}"
MIN_ASR_CHARS="${MIN_ASR_CHARS:-2}"
MIN_DURATION_RATIO="${MIN_DURATION_RATIO:-0.50}"
MAX_DURATION_RATIO="${MAX_DURATION_RATIO:-1.80}"
CONTENT_KEEP_MISSING_AS="${CONTENT_KEEP_MISSING_AS:-drop}"

HUBERT_CACHE_DIR="${HUBERT_CACHE_DIR:-$DOWNLOAD_ROOT/huggingface}"
DEFAULT_HUBERT_MODEL="$HUBERT_CACHE_DIR/models--facebook--hubert-base-ls960/snapshots/dba3bb02fda4248b6e082697eee756de8fe8aa8a"
if [[ ! -f "$DEFAULT_HUBERT_MODEL/config.json" ]]; then
  DEFAULT_HUBERT_MODEL="facebook/hubert-base-ls960"
fi
HUBERT_MODEL="${HUBERT_MODEL:-$DEFAULT_HUBERT_MODEL}"
HUBERT_DEVICE="${HUBERT_DEVICE:-cuda}"
HUBERT_DTYPE="${HUBERT_DTYPE:-auto}"
HUBERT_SAVE_DTYPE="${HUBERT_SAVE_DTYPE:-float16}"
HUBERT_LAYER="${HUBERT_LAYER:-9}"
HUBERT_DOWNSAMPLE_STRIDE="${HUBERT_DOWNSAMPLE_STRIDE:-1}"
HUBERT_USE_SAFETENSORS="${HUBERT_USE_SAFETENSORS:-false}"
HUBERT_LOCAL_FILES_ONLY="${HUBERT_LOCAL_FILES_ONLY:-1}"
HUBERT_SOURCE="${HUBERT_SOURCE:-both}"
HUBERT_NUM_SHARDS="${HUBERT_NUM_SHARDS:-1}"
HUBERT_DEVICES="${HUBERT_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7}"
HUBERT_SHARD_DIR="${HUBERT_SHARD_DIR:-$FINAL_JSONL.shards}"
P2_DRY_RUN="${P2_DRY_RUN:-0}"

SEMANTIC_GPU_KEEPALIVE="${SEMANTIC_GPU_KEEPALIVE:-0}"
SEMANTIC_GPU_KEEPALIVE_GPU_IDS="${SEMANTIC_GPU_KEEPALIVE_GPU_IDS:-0,1,2,3,4,5,6,7}"
SEMANTIC_GPU_KEEPALIVE_LOG_DIR="${SEMANTIC_GPU_KEEPALIVE_LOG_DIR:-$TRAINSET_DIR/shards_semantic_no_text/gpu_keepalive}"
SEMANTIC_GPU_KEEPALIVE_MATMUL_SIZE="${SEMANTIC_GPU_KEEPALIVE_MATMUL_SIZE:-2048}"
SEMANTIC_GPU_KEEPALIVE_SLEEP_SEC="${SEMANTIC_GPU_KEEPALIVE_SLEEP_SEC:-0.05}"
SEMANTIC_GPU_KEEPALIVE_DTYPE="${SEMANTIC_GPU_KEEPALIVE_DTYPE:-float16}"
SEMANTIC_GPU_KEEPALIVE_PIDS=()

pick_csv_item() {
  local csv="$1"
  local index="$2"
  IFS=',' read -r -a items <<< "$csv"
  if [[ "${#items[@]}" -eq 0 ]]; then
    echo ""
    return 1
  fi
  echo "${items[$((index % ${#items[@]}))]}"
}

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

start_semantic_gpu_keepalive() {
  if ! truthy "$SEMANTIC_GPU_KEEPALIVE"; then
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
    log_file=$(printf "%s/no_text_semantic_gpu%s_w%02d.log" "$SEMANTIC_GPU_KEEPALIVE_LOG_DIR" "$gpu_id" "$worker_index")
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
out = {
    "rows": rows,
    "num_shards": num_shards,
    "shards": summaries,
    "stats": dict(stats),
}
summary_path = output_jsonl.with_suffix(output_jsonl.suffix + ".summary.json")
summary_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
done_path = output_jsonl.with_name(output_jsonl.name + ".done.json")
done_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[shard-merge] output={output_jsonl} rows={rows} summary={summary_path}", flush=True)
PY
}

overwrite_flag=()
if [[ "$OVERWRITE" == "1" ]]; then
  overwrite_flag+=(--overwrite)
fi

resume_flag=()
if [[ "$RESUME_SHARDS" == "1" ]]; then
  resume_flag+=(--resume)
fi

limit_flag=()
if [[ "$MAX_ROWS" != "0" ]]; then
  limit_flag+=(--limit "$MAX_ROWS")
fi

max_rows_flag=()
if [[ "$MAX_ROWS" != "0" ]]; then
  max_rows_flag+=(--max-rows "$MAX_ROWS")
fi

language_flag=()
if [[ -n "$LANGUAGE" ]]; then
  language_flag+=(--language "$LANGUAGE")
fi

echo "[ver2.1-data] project=$PROJECT_ROOT"
echo "[ver2.1-data] input=$INPUT_JSONL"
echo "[ver2.1-data] asr=$ASR_JSONL"
echo "[ver2.1-data] asr_kept=$ASR_KEPT_JSONL filter_content_keep=$FILTER_CONTENT_KEEP"
echo "[ver2.1-data] content=$CONTENT_JSONL"
echo "[ver2.1-data] final=$FINAL_JSONL"
echo "[ver2.1-data] max_rows=$MAX_ROWS overwrite=$OVERWRITE resume_shards=$RESUME_SHARDS"
echo "[ver2.1-data] asr_shard_parallelism=$ASR_SHARD_PARALLELISM"
echo "[ver2.1-data] semantic_gpu_keepalive=$SEMANTIC_GPU_KEEPALIVE ids=$SEMANTIC_GPU_KEEPALIVE_GPU_IDS"

start_semantic_gpu_keepalive

if [[ "$RUN_P0" == "1" ]]; then
  echo "[ver2.1-data] P0 ASR/content filter backend=$ASR_BACKEND python=$PYTHON_ASR shards=$ASR_NUM_SHARDS"
  asr_args=(
    scripts/001012_filter_teacher_content_quality.py
    --input-jsonl "$INPUT_JSONL"
    --asr-backend "$ASR_BACKEND"
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
    "$PYTHON_ASR" "${asr_args[@]}" \
      --output-jsonl "$ASR_JSONL" \
      --device "$ASR_DEVICE"
  else
    if [[ -e "$ASR_JSONL" && "$OVERWRITE" != "1" ]]; then
      if [[ "$RESUME_SHARDS" == "1" ]]; then
        echo "[ver2.1-data] ASR_JSONL already exists; reuse completed P0 output: $ASR_JSONL"
      else
        echo "ERROR: ASR_JSONL exists, set OVERWRITE=1 for a fresh sharded run: $ASR_JSONL" >&2
        exit 1
      fi
    else
      if [[ "$OVERWRITE" == "1" ]]; then
        rm -rf "$ASR_SHARD_DIR"
        rm -f "$ASR_JSONL" "$ASR_JSONL.tmp" "$ASR_JSONL.summary.json" "$ASR_JSONL.done.json"
      fi
      mkdir -p "$ASR_SHARD_DIR"
      if [[ "$ASR_SHARD_PARALLELISM" -lt 1 ]]; then
        echo "ERROR: ASR_SHARD_PARALLELISM must be >= 1, got $ASR_SHARD_PARALLELISM" >&2
        exit 2
      fi
      pids=()
      failed=0
      for shard in $(seq 0 $((ASR_NUM_SHARDS - 1))); do
        shard_out=$(printf "%s/shard_%05d.jsonl" "$ASR_SHARD_DIR" "$shard")
        shard_log=$(printf "%s/shard_%05d.log" "$ASR_SHARD_DIR" "$shard")
        shard_device="$(pick_csv_item "$ASR_DEVICES" "$shard")"
        echo "[ver2.1-data] launch P0 shard=$shard/$ASR_NUM_SHARDS device=$shard_device out=$shard_out log=$shard_log"
        (
          "$PYTHON_ASR" "${asr_args[@]}" \
            --output-jsonl "$shard_out" \
            --device "$shard_device" \
            --num-shards "$ASR_NUM_SHARDS" \
            --shard-index "$shard"
        ) >"$shard_log" 2>&1 &
        pids+=("$!")
        if [[ "${#pids[@]}" -ge "$ASR_SHARD_PARALLELISM" ]]; then
          for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
              failed=1
            fi
          done
          pids=()
          if [[ "$failed" -ne 0 ]]; then
            break
          fi
        fi
      done
      for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
          failed=1
        fi
      done
      if [[ "$failed" != "0" ]]; then
        echo "ERROR: one or more P0 ASR shards failed. Recent shard logs:" >&2
        tail -n 80 "$ASR_SHARD_DIR"/shard_*.log >&2 || true
        exit 1
      fi
      merge_jsonl_shards "$ASR_SHARD_DIR" "$ASR_JSONL" "$ASR_NUM_SHARDS"
      merge_summary_shards "$ASR_SHARD_DIR" "$ASR_JSONL" "$ASR_NUM_SHARDS"
    fi
  fi
else
  echo "[ver2.1-data] skip P0, using ASR_JSONL=$ASR_JSONL"
fi

P1_INPUT_JSONL="$ASR_JSONL"
if [[ "$RUN_P1" == "1" && "$FILTER_CONTENT_KEEP" == "1" ]]; then
  echo "[ver2.1-data] filter content_keep=true rows python=$PYTHON_MAIN"
  "$PYTHON_MAIN" scripts/001021_filter_content_keep_jsonl.py \
    --input-jsonl "$ASR_JSONL" \
    --output-jsonl "$ASR_KEPT_JSONL" \
    --missing-as "$CONTENT_KEEP_MISSING_AS" \
    --progress-every "$PROGRESS_EVERY" \
    "${max_rows_flag[@]}" \
    "${overwrite_flag[@]}"
  P1_INPUT_JSONL="$ASR_KEPT_JSONL"
else
  echo "[ver2.1-data] skip content_keep filtering; P1 input remains $ASR_JSONL"
fi

if [[ "$RUN_P1" == "1" ]]; then
  echo "[ver2.1-data] P1 content_ref_text -> content_token_ids python=$PYTHON_MAIN"
  "$PYTHON_MAIN" scripts/001019_extract_content_tokens.py \
    --input-jsonl "$P1_INPUT_JSONL" \
    --output-jsonl "$CONTENT_JSONL" \
    --vocab-json "$CONTENT_VOCAB_JSON" \
    --progress-every "$PROGRESS_EVERY" \
    "${max_rows_flag[@]}" \
    "${overwrite_flag[@]}"
else
  echo "[ver2.1-data] skip P1, using CONTENT_JSONL=$CONTENT_JSONL"
fi

if [[ "$RUN_P2" == "1" ]]; then
  echo "[ver2.1-data] P2 $HUBERT_SOURCE wav -> HuBERT semantic features python=$PYTHON_MAIN shards=$HUBERT_NUM_SHARDS"
  p2_args=(
    scripts/001020_extract_hubert_semantic_features.py
    --input-jsonl "$CONTENT_JSONL"
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
    "${max_rows_flag[@]}"
    "${overwrite_flag[@]}"
  )
  if [[ "$HUBERT_LOCAL_FILES_ONLY" == "1" ]]; then
    p2_args+=(--local-files-only)
  fi
  if [[ "$P2_DRY_RUN" == "1" ]]; then
    p2_args+=(--dry-run)
  fi
  if [[ "$HUBERT_NUM_SHARDS" -le 1 ]]; then
    "$PYTHON_MAIN" "${p2_args[@]}" \
      --output-jsonl "$FINAL_JSONL" \
      --device "$HUBERT_DEVICE"
  else
    if [[ -e "$FINAL_JSONL" && "$OVERWRITE" != "1" ]]; then
      echo "ERROR: FINAL_JSONL exists, set OVERWRITE=1 for a fresh sharded run: $FINAL_JSONL" >&2
      exit 1
    fi
    if [[ "$OVERWRITE" == "1" ]]; then
      rm -rf "$HUBERT_SHARD_DIR"
      rm -f "$FINAL_JSONL" "$FINAL_JSONL.tmp" "$FINAL_JSONL.summary.json" "$FINAL_JSONL.done.json"
    fi
    mkdir -p "$HUBERT_SHARD_DIR"
    pids=()
    for shard in $(seq 0 $((HUBERT_NUM_SHARDS - 1))); do
      shard_out=$(printf "%s/shard_%05d.jsonl" "$HUBERT_SHARD_DIR" "$shard")
      shard_log=$(printf "%s/shard_%05d.log" "$HUBERT_SHARD_DIR" "$shard")
      shard_device="$(pick_csv_item "$HUBERT_DEVICES" "$shard")"
      echo "[ver2.1-data] launch P2 shard=$shard/$HUBERT_NUM_SHARDS device=$shard_device out=$shard_out log=$shard_log"
      (
        "$PYTHON_MAIN" "${p2_args[@]}" \
          --output-jsonl "$shard_out" \
          --device "$shard_device" \
          --num-shards "$HUBERT_NUM_SHARDS" \
          --shard-index "$shard"
      ) >"$shard_log" 2>&1 &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        failed=1
      fi
    done
    if [[ "$failed" != "0" ]]; then
      echo "ERROR: one or more P2 HuBERT shards failed. Recent shard logs:" >&2
      tail -n 80 "$HUBERT_SHARD_DIR"/shard_*.log >&2 || true
      exit 1
    fi
    merge_jsonl_shards "$HUBERT_SHARD_DIR" "$FINAL_JSONL" "$HUBERT_NUM_SHARDS"
    merge_summary_shards "$HUBERT_SHARD_DIR" "$FINAL_JSONL" "$HUBERT_NUM_SHARDS"
  fi
else
  echo "[ver2.1-data] skip P2, final remains CONTENT_JSONL=$CONTENT_JSONL"
fi

TRAIN_JSONL_OUT="$FINAL_JSONL"
if [[ "$RUN_P2" != "1" ]]; then
  TRAIN_JSONL_OUT="$CONTENT_JSONL"
fi

echo "[ver2.1-data] complete"
echo "[ver2.1-data] train_jsonl=$TRAIN_JSONL_OUT"

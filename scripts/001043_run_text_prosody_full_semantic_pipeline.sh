#!/usr/bin/env bash
set -euo pipefail

# End-to-end reusable text_prosody pipeline:
# MOSS-TTS vcdata JSONL(s) -> independent-timbre text_prosody wav triples -> train-ready SFT -> ASR/CTC/HuBERT semantic JSONL.

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
PY_MOSS="${PY_MOSS:-$PY}"
PY_SEEDVC="${PY_SEEDVC:-/inspire/ssd/project/embodied-multimodality/public/yqzhang/miniconda3/envs/contts-train/bin/python}"
PYTHON_MAIN="${PYTHON_MAIN:-$PY}"
PYTHON_ASR="${PYTHON_ASR:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download}"

DATASET_NAME="${DATASET_NAME:-zh3w_en3w_text_prosody_independent_timbre_semantic}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/trainset/$DATASET_NAME}"
RUN_NAME="${RUN_NAME:-$DATASET_NAME}"

MANIFEST_JSONL="${MANIFEST_JSONL:-$DATASET_ROOT/manifests/vc_manifest.$DATASET_NAME.jsonl}"
MANIFEST_SUMMARY_JSON="${MANIFEST_SUMMARY_JSON:-$DATASET_ROOT/manifests/vc_manifest.$DATASET_NAME.summary.json}"
TEXT_SEEDVC_WORK_ROOT="${TEXT_SEEDVC_WORK_ROOT:-$DATASET_ROOT/intermediate/text_seedvc}"
TEXT_SEEDVC_JOBS_JSONL="${TEXT_SEEDVC_JOBS_JSONL:-$TEXT_SEEDVC_WORK_ROOT/text_seedvc_jobs.jsonl}"
TEXT_SEEDVC_RESULTS_JSONL="${TEXT_SEEDVC_RESULTS_JSONL:-$TEXT_SEEDVC_WORK_ROOT/text_seedvc_results.jsonl}"
TEXT_SEEDVC_TARGET_AUDIO_ROOT="${TEXT_SEEDVC_TARGET_AUDIO_ROOT:-$DATASET_ROOT/seedvc_targets}"

TRAIN_READY_JSONL="${TRAIN_READY_JSONL:-$DATASET_ROOT/sft/moss_codecvc_sft.$DATASET_NAME.with_light_ecapa_spk.with_prosody.jsonl}"
SEMANTIC_OUTPUT_PREFIX="${SEMANTIC_OUTPUT_PREFIX:-$DATASET_ROOT/sft/moss_codecvc_sft.$DATASET_NAME.with_light_ecapa_spk.with_prosody}"
SEMANTIC_FINAL_JSONL="${SEMANTIC_FINAL_JSONL:-$SEMANTIC_OUTPUT_PREFIX.with_content_tokens.with_target_hubert.jsonl}"
PIPELINE_DONE_JSON="${PIPELINE_DONE_JSON:-$DATASET_ROOT/text_prosody_full_semantic_pipeline.done.json}"

RUN_TEXT_TRIPLE_STAGE="${RUN_TEXT_TRIPLE_STAGE:-1}"
RUN_TRAIN_READY_STAGE="${RUN_TRAIN_READY_STAGE:-1}"
RUN_SEMANTIC_STAGE="${RUN_SEMANTIC_STAGE:-1}"
FORCE="${FORCE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN=0

RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_SEEDVC="${RUN_SEEDVC:-1}"
RUN_COLLECT="${RUN_COLLECT:-1}"
PREPARE_REQUIRE_EXISTING_AUDIO="${PREPARE_REQUIRE_EXISTING_AUDIO:-1}"
PREPARE_PROGRESS_EVERY="${PREPARE_PROGRESS_EVERY:-1000}"
LANGUAGES="${LANGUAGES:-zh,en}"
TIMBRE_REF_POLICY="${TIMBRE_REF_POLICY:-random_different_text}"
TIMBRE_REF_SEED="${TIMBRE_REF_SEED:-20260627}"
MAX_JOBS="${MAX_JOBS:-0}"
MAX_JOBS_PER_LANGUAGE="${MAX_JOBS_PER_LANGUAGE:-0}"
MAX_ROWS_PER_INPUT="${MAX_ROWS_PER_INPUT:-0}"
MIN_BEST_SIMILARITY="${MIN_BEST_SIMILARITY:-0.0}"
MIN_DNSMOS="${MIN_DNSMOS:-0.0}"
SKIP_FLAGS="${SKIP_FLAGS:-}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
SEEDVC_GPU_IDS="${SEEDVC_GPU_IDS:-$GPU_IDS}"
SEEDVC_SHARD_COUNT="${SEEDVC_SHARD_COUNT:-8}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-25}"
LENGTH_ADJUST="${LENGTH_ADJUST:-1.0}"
INFERENCE_CFG_RATE="${INFERENCE_CFG_RATE:-0.7}"
FP16="${FP16:-true}"
SEEDVC_SKIP_EXISTING="${SEEDVC_SKIP_EXISTING:-1}"
FAIL_FAST="${FAIL_FAST:-0}"
SHOW_MODEL_OUTPUT="${SHOW_MODEL_OUTPUT:-0}"
MIN_TARGET_AUDIO_BYTES="${MIN_TARGET_AUDIO_BYTES:-4096}"

TRAIN_READY_MAX_ROWS="${TRAIN_READY_MAX_ROWS:-0}"
N_VQ="${N_VQ:-32}"
CODEC_GPU_IDS="${CODEC_GPU_IDS:-$GPU_IDS}"
SPEAKER_GPU_IDS="${SPEAKER_GPU_IDS:-$GPU_IDS}"
CODEC_SHARD_COUNT="${CODEC_SHARD_COUNT:-8}"
SPEAKER_SHARD_COUNT="${SPEAKER_SHARD_COUNT:-8}"
PROSODY_SHARD_COUNT="${PROSODY_SHARD_COUNT:-16}"
TRAIN_READY_GPU_KEEPALIVE="${TRAIN_READY_GPU_KEEPALIVE:-0}"
TRAIN_READY_WRITE_TRAIN_COMMAND="${TRAIN_READY_WRITE_TRAIN_COMMAND:-0}"

SEMANTIC_MAX_ROWS="${SEMANTIC_MAX_ROWS:-0}"
SEMANTIC_RUN_PARALLEL_BRANCHES="${SEMANTIC_RUN_PARALLEL_BRANCHES:-1}"
SEMANTIC_FILTER_CONTENT_KEEP="${SEMANTIC_FILTER_CONTENT_KEEP:-1}"
SEMANTIC_CONTENT_REQUIRE_CONTENT_KEEP="${SEMANTIC_CONTENT_REQUIRE_CONTENT_KEEP:-1}"
SEMANTIC_GPU_KEEPALIVE="${SEMANTIC_GPU_KEEPALIVE:-0}"
SEMANTIC_GPU_KEEPALIVE_GPU_IDS="${SEMANTIC_GPU_KEEPALIVE_GPU_IDS:-$GPU_IDS}"
SEMANTIC_GPU_KEEPALIVE_MATMUL_SIZE="${SEMANTIC_GPU_KEEPALIVE_MATMUL_SIZE:-2048}"
SEMANTIC_GPU_KEEPALIVE_SLEEP_SEC="${SEMANTIC_GPU_KEEPALIVE_SLEEP_SEC:-0.05}"
SEMANTIC_GPU_KEEPALIVE_DTYPE="${SEMANTIC_GPU_KEEPALIVE_DTYPE:-float16}"
ASR_BACKEND="${ASR_BACKEND:-qwen_asr}"
ASR_NUM_SHARDS="${ASR_NUM_SHARDS:-8}"
ASR_DEVICES="${ASR_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7}"
HUBERT_NUM_SHARDS="${HUBERT_NUM_SHARDS:-8}"
HUBERT_DEVICES="${HUBERT_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7}"
HUBERT_PRE_SPLIT="${HUBERT_PRE_SPLIT:-1}"
HUBERT_SOURCE="${HUBERT_SOURCE:-target}"
SEMANTIC_WRITE_TRAIN_COMMAND="${SEMANTIC_WRITE_TRAIN_COMMAND:-1}"

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

mkdir -p "$DATASET_ROOT" "$(dirname "$MANIFEST_JSONL")" "$TEXT_SEEDVC_WORK_ROOT" "$TEXT_SEEDVC_TARGET_AUDIO_ROOT"

echo "=========================================="
echo "Text_prosody full semantic pipeline"
echo "  ROOT=$ROOT"
echo "  DATASET_NAME=$DATASET_NAME"
echo "  DATASET_ROOT=$DATASET_ROOT"
echo "  RUN_TEXT_TRIPLE_STAGE=$RUN_TEXT_TRIPLE_STAGE"
echo "  RUN_TRAIN_READY_STAGE=$RUN_TRAIN_READY_STAGE"
echo "  RUN_SEMANTIC_STAGE=$RUN_SEMANTIC_STAGE"
echo "  MANIFEST_JSONL=$MANIFEST_JSONL"
echo "  TRAIN_READY_JSONL=$TRAIN_READY_JSONL"
echo "  SEMANTIC_FINAL_JSONL=$SEMANTIC_FINAL_JSONL"
echo "  VCDATA_JSONLS=${VCDATA_JSONLS:-<default 001035 list>}"
echo "  TIMBRE_REF_POLICY=$TIMBRE_REF_POLICY"
echo "  SEEDVC_SHARD_COUNT=$SEEDVC_SHARD_COUNT CODEC_SHARD_COUNT=$CODEC_SHARD_COUNT SPEAKER_SHARD_COUNT=$SPEAKER_SHARD_COUNT"
echo "  ASR_NUM_SHARDS=$ASR_NUM_SHARDS HUBERT_NUM_SHARDS=$HUBERT_NUM_SHARDS"
echo "  FORCE=$FORCE SKIP_EXISTING=$SKIP_EXISTING DRY_RUN=$DRY_RUN"
echo "=========================================="

run_cmd "$PY" -m py_compile \
  scripts/001034_build_text_prosody_from_mosstts_vcdata.py \
  scripts/001002_encode_codec_tokens.py \
  scripts/001003_build_moss_sft_jsonl.py \
  scripts/001017_asr_content_filter.py \
  scripts/001019_extract_content_tokens.py \
  scripts/001020_extract_hubert_semantic_features.py

if truthy "$RUN_TEXT_TRIPLE_STAGE"; then
  if should_reuse "$MANIFEST_JSONL"; then
    echo "[skip] text_prosody manifest exists: $MANIFEST_JSONL"
  else
    echo "=========================================="
    echo "Stage 1/3: vcdata -> independent-timbre text_prosody wav triples"
    echo "=========================================="
    run_cmd env \
      PY_MOSS="$PY_MOSS" \
      PY_SEEDVC="$PY_SEEDVC" \
      RUN_NAME="$RUN_NAME" \
      WORK_ROOT="$TEXT_SEEDVC_WORK_ROOT" \
      JOBS_JSONL="$TEXT_SEEDVC_JOBS_JSONL" \
      RESULTS_JSONL="$TEXT_SEEDVC_RESULTS_JSONL" \
      MANIFEST_JSONL="$MANIFEST_JSONL" \
      TARGET_AUDIO_ROOT="$TEXT_SEEDVC_TARGET_AUDIO_ROOT" \
      RUN_PREPARE="$RUN_PREPARE" \
      RUN_SEEDVC="$RUN_SEEDVC" \
      RUN_COLLECT="$RUN_COLLECT" \
      PREPARE_REQUIRE_EXISTING_AUDIO="$PREPARE_REQUIRE_EXISTING_AUDIO" \
      PREPARE_PROGRESS_EVERY="$PREPARE_PROGRESS_EVERY" \
      LANGUAGES="$LANGUAGES" \
      TIMBRE_REF_POLICY="$TIMBRE_REF_POLICY" \
      TIMBRE_REF_SEED="$TIMBRE_REF_SEED" \
      MAX_JOBS="$MAX_JOBS" \
      MAX_JOBS_PER_LANGUAGE="$MAX_JOBS_PER_LANGUAGE" \
      MAX_ROWS_PER_INPUT="$MAX_ROWS_PER_INPUT" \
      MIN_BEST_SIMILARITY="$MIN_BEST_SIMILARITY" \
      MIN_DNSMOS="$MIN_DNSMOS" \
      SKIP_FLAGS="$SKIP_FLAGS" \
      SEEDVC_GPU_IDS="$SEEDVC_GPU_IDS" \
      SEEDVC_SHARD_COUNT="$SEEDVC_SHARD_COUNT" \
      DIFFUSION_STEPS="$DIFFUSION_STEPS" \
      LENGTH_ADJUST="$LENGTH_ADJUST" \
      INFERENCE_CFG_RATE="$INFERENCE_CFG_RATE" \
      FP16="$FP16" \
      SKIP_EXISTING="$SEEDVC_SKIP_EXISTING" \
      FAIL_FAST="$FAIL_FAST" \
      SHOW_MODEL_OUTPUT="$SHOW_MODEL_OUTPUT" \
      MIN_TARGET_AUDIO_BYTES="$MIN_TARGET_AUDIO_BYTES" \
      VCDATA_JSONLS="${VCDATA_JSONLS:-}" \
      bash scripts/001035_run_text_prosody_mosstts_seedvc_pipeline.sh
    if [ "$DRY_RUN" -eq 0 ] && [ -s "$TEXT_SEEDVC_WORK_ROOT/vc_manifest.text_prosody.summary.json" ]; then
      cp "$TEXT_SEEDVC_WORK_ROOT/vc_manifest.text_prosody.summary.json" "$MANIFEST_SUMMARY_JSON"
    fi
  fi
fi

if [ "$DRY_RUN" -eq 0 ] && [ ! -s "$MANIFEST_JSONL" ]; then
  echo "Missing text_prosody manifest: $MANIFEST_JSONL" >&2
  exit 2
fi

if truthy "$RUN_TRAIN_READY_STAGE"; then
  echo "=========================================="
  echo "Stage 2/3: wav triples -> codec/SFT/ECAPA/prosody train-ready JSONL"
  echo "=========================================="
  train_ready_args=()
  if [ "$DRY_RUN" -eq 1 ]; then
    train_ready_args=(--dry-run)
  fi
  run_cmd env \
    PY="$PY" \
    DATASET_NAME="$DATASET_NAME" \
    DATASET_ROOT="$DATASET_ROOT" \
    INPUT_JSONL="$MANIFEST_JSONL" \
    EMIT_MODES="text" \
    N_VQ="$N_VQ" \
    GPU_IDS="$GPU_IDS" \
    CODEC_GPU_IDS="$CODEC_GPU_IDS" \
    SPEAKER_GPU_IDS="$SPEAKER_GPU_IDS" \
    CODEC_SHARD_COUNT="$CODEC_SHARD_COUNT" \
    SPEAKER_SHARD_COUNT="$SPEAKER_SHARD_COUNT" \
    PROSODY_SHARD_COUNT="$PROSODY_SHARD_COUNT" \
    MAX_ROWS="$TRAIN_READY_MAX_ROWS" \
    SKIP_EXISTING="$SKIP_EXISTING" \
    FORCE="$FORCE" \
    GPU_KEEPALIVE="$TRAIN_READY_GPU_KEEPALIVE" \
    WRITE_TRAIN_COMMAND="$TRAIN_READY_WRITE_TRAIN_COMMAND" \
    bash scripts/001039_run_train_ready_text_prosody_parallel.sh "${train_ready_args[@]}"
fi

if [ "$DRY_RUN" -eq 0 ] && [ ! -s "$TRAIN_READY_JSONL" ]; then
  echo "Missing train-ready JSONL: $TRAIN_READY_JSONL" >&2
  exit 2
fi

if truthy "$RUN_SEMANTIC_STAGE"; then
  echo "=========================================="
  echo "Stage 3/3: train-ready JSONL -> ASR/CTC/HuBERT semantic JSONL"
  echo "=========================================="
  semantic_args=()
  if [ "$DRY_RUN" -eq 1 ]; then
    semantic_args=(--dry-run)
  fi
  run_cmd env \
    PROJECT_ROOT="$ROOT" \
    PYTHON_MAIN="$PYTHON_MAIN" \
    PYTHON_ASR="$PYTHON_ASR" \
    DOWNLOAD_ROOT="$DOWNLOAD_ROOT" \
    DATASET_NAME="$DATASET_NAME" \
    DATASET_ROOT="$DATASET_ROOT" \
    INPUT_JSONL="$TRAIN_READY_JSONL" \
    OUTPUT_PREFIX="$SEMANTIC_OUTPUT_PREFIX" \
    FINAL_JSONL="$SEMANTIC_FINAL_JSONL" \
    RUN_PARALLEL_SEMANTIC_BRANCHES="$SEMANTIC_RUN_PARALLEL_BRANCHES" \
    FILTER_CONTENT_KEEP="$SEMANTIC_FILTER_CONTENT_KEEP" \
    CONTENT_REQUIRE_CONTENT_KEEP="$SEMANTIC_CONTENT_REQUIRE_CONTENT_KEEP" \
    ASR_BACKEND="$ASR_BACKEND" \
    ASR_NUM_SHARDS="$ASR_NUM_SHARDS" \
    ASR_DEVICES="$ASR_DEVICES" \
    HUBERT_SOURCE="$HUBERT_SOURCE" \
    HUBERT_NUM_SHARDS="$HUBERT_NUM_SHARDS" \
    HUBERT_DEVICES="$HUBERT_DEVICES" \
    HUBERT_PRE_SPLIT="$HUBERT_PRE_SPLIT" \
    SEMANTIC_GPU_KEEPALIVE="$SEMANTIC_GPU_KEEPALIVE" \
    SEMANTIC_GPU_KEEPALIVE_GPU_IDS="$SEMANTIC_GPU_KEEPALIVE_GPU_IDS" \
    SEMANTIC_GPU_KEEPALIVE_MATMUL_SIZE="$SEMANTIC_GPU_KEEPALIVE_MATMUL_SIZE" \
    SEMANTIC_GPU_KEEPALIVE_SLEEP_SEC="$SEMANTIC_GPU_KEEPALIVE_SLEEP_SEC" \
    SEMANTIC_GPU_KEEPALIVE_DTYPE="$SEMANTIC_GPU_KEEPALIVE_DTYPE" \
    MAX_ROWS="$SEMANTIC_MAX_ROWS" \
    SKIP_EXISTING="$SKIP_EXISTING" \
    FORCE="$FORCE" \
    WRITE_TRAIN_COMMAND="$SEMANTIC_WRITE_TRAIN_COMMAND" \
    bash scripts/001041_prepare_text_prosody_semantic_features.sh "${semantic_args[@]}"
fi

if [ "$DRY_RUN" -eq 0 ]; then
  "$PY" - "$PIPELINE_DONE_JSON" "$DATASET_NAME" "$DATASET_ROOT" "$MANIFEST_JSONL" "$TRAIN_READY_JSONL" "$SEMANTIC_FINAL_JSONL" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

done_path = Path(sys.argv[1])
dataset_name = sys.argv[2]
dataset_root = Path(sys.argv[3])
manifest = Path(sys.argv[4])
train_ready = Path(sys.argv[5])
semantic_final = Path(sys.argv[6])
payload = {
    "status": "complete" if semantic_final.exists() else "partial",
    "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "dataset_name": dataset_name,
    "dataset_root": str(dataset_root.resolve()),
    "manifest_jsonl": str(manifest.resolve()),
    "train_ready_jsonl": str(train_ready.resolve()),
    "semantic_final_jsonl": str(semantic_final.resolve()),
    "manifest_exists": manifest.exists(),
    "train_ready_exists": train_ready.exists(),
    "semantic_final_exists": semantic_final.exists(),
    "semantic_final_rows": sum(1 for _ in semantic_final.open("r", encoding="utf-8")) if semantic_final.exists() else 0,
}
done_path.parent.mkdir(parents=True, exist_ok=True)
done_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
PY
fi

echo "=========================================="
echo "Text_prosody full semantic pipeline finished"
echo "  manifest_jsonl=$MANIFEST_JSONL"
echo "  train_ready_jsonl=$TRAIN_READY_JSONL"
echo "  semantic_train_jsonl=$SEMANTIC_FINAL_JSONL"
echo "  pipeline_done=$PIPELINE_DONE_JSON"
echo "=========================================="

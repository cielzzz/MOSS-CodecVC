#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
if [ ! -x "$PY" ]; then
  PY=python
fi

DATASET_NAME="${DATASET_NAME:-zh3w_en3w_text_prosody_independent_timbre}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/trainset/$DATASET_NAME}"
RUN_NAME="${RUN_NAME:-text_prosody_independent_timbre_zh_en_0001_0004}"

MANIFEST_JSONL="${MANIFEST_JSONL:-$DATASET_ROOT/manifests/vc_manifest.$DATASET_NAME.jsonl}"
MANIFEST_SUMMARY_JSON="${MANIFEST_SUMMARY_JSON:-$DATASET_ROOT/manifests/vc_manifest.$DATASET_NAME.summary.json}"
TEXT_SEEDVC_WORK_ROOT="${TEXT_SEEDVC_WORK_ROOT:-$DATASET_ROOT/intermediate/text_seedvc}"
TEXT_SEEDVC_JOBS_JSONL="${TEXT_SEEDVC_JOBS_JSONL:-$TEXT_SEEDVC_WORK_ROOT/text_seedvc_jobs.jsonl}"
TEXT_SEEDVC_RESULTS_JSONL="${TEXT_SEEDVC_RESULTS_JSONL:-$TEXT_SEEDVC_WORK_ROOT/text_seedvc_results.jsonl}"
TEXT_SEEDVC_TARGET_AUDIO_ROOT="${TEXT_SEEDVC_TARGET_AUDIO_ROOT:-$DATASET_ROOT/seedvc_targets}"

RUN_TEXT_TRIPLE_STAGE="${RUN_TEXT_TRIPLE_STAGE:-1}"
RUN_TRAIN_READY_STAGE="${RUN_TRAIN_READY_STAGE:-1}"
RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_SEEDVC="${RUN_SEEDVC:-1}"
RUN_COLLECT="${RUN_COLLECT:-1}"
PREPARE_REQUIRE_EXISTING_AUDIO="${PREPARE_REQUIRE_EXISTING_AUDIO:-1}"
PREPARE_PROGRESS_EVERY="${PREPARE_PROGRESS_EVERY:-1000}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE="${FORCE:-0}"
DRY_RUN=0

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

N_VQ="${N_VQ:-32}"
CODEC_SHARD_COUNT="${CODEC_SHARD_COUNT:-8}"
SPEAKER_SHARD_COUNT="${SPEAKER_SHARD_COUNT:-8}"
PROSODY_SHARD_COUNT="${PROSODY_SHARD_COUNT:-16}"
SPEAKER_DEVICE="${SPEAKER_DEVICE:-cuda:0}"
RUN_PROSODY_FEATURES="${RUN_PROSODY_FEATURES:-1}"
ATTACH_REQUIRE_EMBEDDING_EXISTS="${ATTACH_REQUIRE_EMBEDDING_EXISTS:-1}"
GPU_KEEPALIVE="${GPU_KEEPALIVE:-0}"
GPU_KEEPALIVE_STAGES="${GPU_KEEPALIVE_STAGES:-speaker_extract,attach,prosody_extract}"
GPU_KEEPALIVE_GPU_IDS="${GPU_KEEPALIVE_GPU_IDS:-$GPU_IDS}"
WAIT_HEARTBEAT_SECS="${WAIT_HEARTBEAT_SECS:-60}"
WRITE_TRAIN_COMMAND="${WRITE_TRAIN_COMMAND:-1}"
PIPELINE_DONE_JSON="${PIPELINE_DONE_JSON:-$DATASET_ROOT/pipeline.done.json}"

RUN_CODEC="${RUN_CODEC:-1}"
RUN_SFT="${RUN_SFT:-1}"
RUN_SPEAKER_PLAN="${RUN_SPEAKER_PLAN:-1}"
RUN_SPEAKER_EXTRACT="${RUN_SPEAKER_EXTRACT:-1}"
RUN_ATTACH="${RUN_ATTACH:-1}"

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

mkdir -p \
  "$(dirname "$MANIFEST_JSONL")" \
  "$TEXT_SEEDVC_WORK_ROOT" \
  "$TEXT_SEEDVC_TARGET_AUDIO_ROOT" \
  "$DATASET_ROOT"

echo "=========================================="
echo "MOSS-CodecVC text_prosody independent-timbre full pipeline"
echo "  ROOT=$ROOT"
echo "  PY=$PY"
echo "  DATASET_NAME=$DATASET_NAME"
echo "  DATASET_ROOT=$DATASET_ROOT"
echo "  RUN_TEXT_TRIPLE_STAGE=$RUN_TEXT_TRIPLE_STAGE"
echo "  RUN_TRAIN_READY_STAGE=$RUN_TRAIN_READY_STAGE"
echo "  MANIFEST_JSONL=$MANIFEST_JSONL"
echo "  TEXT_SEEDVC_JOBS_JSONL=$TEXT_SEEDVC_JOBS_JSONL"
echo "  TEXT_SEEDVC_RESULTS_JSONL=$TEXT_SEEDVC_RESULTS_JSONL"
echo "  TEXT_SEEDVC_TARGET_AUDIO_ROOT=$TEXT_SEEDVC_TARGET_AUDIO_ROOT"
  echo "  TIMBRE_REF_POLICY=$TIMBRE_REF_POLICY"
  echo "  TIMBRE_REF_SEED=$TIMBRE_REF_SEED"
  echo "  PREPARE_REQUIRE_EXISTING_AUDIO=$PREPARE_REQUIRE_EXISTING_AUDIO"
  echo "  PREPARE_PROGRESS_EVERY=$PREPARE_PROGRESS_EVERY"
  echo "  SKIP_FLAGS=$SKIP_FLAGS"
echo "  MAX_JOBS=$MAX_JOBS"
echo "  MAX_JOBS_PER_LANGUAGE=$MAX_JOBS_PER_LANGUAGE"
echo "  SEEDVC_GPU_IDS=$SEEDVC_GPU_IDS"
echo "  SEEDVC_SHARD_COUNT=$SEEDVC_SHARD_COUNT"
echo "  CODEC_SHARD_COUNT=$CODEC_SHARD_COUNT"
echo "  SPEAKER_SHARD_COUNT=$SPEAKER_SHARD_COUNT"
echo "  PROSODY_SHARD_COUNT=$PROSODY_SHARD_COUNT"
echo "  RUN_PROSODY_FEATURES=$RUN_PROSODY_FEATURES"
echo "  FORCE=$FORCE"
echo "  SKIP_EXISTING=$SKIP_EXISTING"
echo "  DRY_RUN=$DRY_RUN"
echo "=========================================="

if truthy "$RUN_TEXT_TRIPLE_STAGE"; then
  if truthy "$SKIP_EXISTING" && [ "$FORCE" != "1" ] && [ -s "$MANIFEST_JSONL" ] && [ -s "$MANIFEST_SUMMARY_JSON" ]; then
    echo "[skip] text_prosody wav triple manifest already exists: $MANIFEST_JSONL"
  else
    echo "=========================================="
    echo "Pipeline part 1/2: MOSS-TTS vcdata -> Seed-VC text_prosody wav triple manifest"
    echo "=========================================="
    run_cmd env \
      PY_MOSS="$PY" \
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
  echo "Pipeline part 2/2: text_prosody wav triples -> train-ready SFT"
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
    CODEC_SHARD_COUNT="$CODEC_SHARD_COUNT" \
    SPEAKER_SHARD_COUNT="$SPEAKER_SHARD_COUNT" \
    PROSODY_SHARD_COUNT="$PROSODY_SHARD_COUNT" \
    RUN_CODEC="$RUN_CODEC" \
    RUN_SFT="$RUN_SFT" \
    RUN_SPEAKER_PLAN="$RUN_SPEAKER_PLAN" \
    RUN_SPEAKER_EXTRACT="$RUN_SPEAKER_EXTRACT" \
    RUN_ATTACH="$RUN_ATTACH" \
    RUN_PROSODY_FEATURES="$RUN_PROSODY_FEATURES" \
    SPEAKER_DEVICE="$SPEAKER_DEVICE" \
    ATTACH_REQUIRE_EMBEDDING_EXISTS="$ATTACH_REQUIRE_EMBEDDING_EXISTS" \
    WAIT_HEARTBEAT_SECS="$WAIT_HEARTBEAT_SECS" \
    GPU_KEEPALIVE="$GPU_KEEPALIVE" \
    GPU_KEEPALIVE_STAGES="$GPU_KEEPALIVE_STAGES" \
    GPU_KEEPALIVE_GPU_IDS="$GPU_KEEPALIVE_GPU_IDS" \
    MAX_ROWS="$MAX_JOBS" \
    SKIP_EXISTING="$SKIP_EXISTING" \
    FORCE="$FORCE" \
    WRITE_TRAIN_COMMAND="$WRITE_TRAIN_COMMAND" \
    bash scripts/001025_run_train_ready_no_text_ver2.sh "${train_ready_args[@]}"
fi

if [ "$DRY_RUN" -eq 0 ]; then
  "$PY" - "$PIPELINE_DONE_JSON" "$DATASET_NAME" "$DATASET_ROOT" "$MANIFEST_JSONL" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

done_path = Path(sys.argv[1])
dataset_name = sys.argv[2]
dataset_root = Path(sys.argv[3])
manifest = Path(sys.argv[4])
prosody_jsonl = dataset_root / "sft" / f"moss_codecvc_sft.{dataset_name}.with_light_ecapa_spk.with_prosody.jsonl"
attached_jsonl = dataset_root / "sft" / f"moss_codecvc_sft.{dataset_name}.with_light_ecapa_spk.jsonl"
active = prosody_jsonl if prosody_jsonl.exists() else attached_jsonl
payload = {
    "status": "complete" if active.exists() else "partial",
    "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "dataset_name": dataset_name,
    "dataset_root": str(dataset_root.resolve()),
    "manifest_jsonl": str(manifest.resolve()),
    "manifest_exists": manifest.exists(),
    "train_jsonl": str(active.resolve()),
    "train_jsonl_exists": active.exists(),
}
done_path.parent.mkdir(parents=True, exist_ok=True)
done_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
fi

echo "=========================================="
echo "Text_prosody independent-timbre full pipeline finished"
echo "  dataset_root=$DATASET_ROOT"
echo "  manifest_jsonl=$MANIFEST_JSONL"
echo "  pipeline_done=$PIPELINE_DONE_JSON"
echo "=========================================="

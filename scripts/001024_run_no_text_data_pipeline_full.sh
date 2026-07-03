#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
if [ ! -x "$PY" ]; then
  PY=python
fi

DATASET_NAME="${DATASET_NAME:-zh45w_en22w_no_text}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/trainset/$DATASET_NAME}"
RAW_INPUT_ROOT="${RAW_INPUT_ROOT:-${INPUT_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vc_data_temp/mtd_pass_nonmulti_primary_le_0p3_split_10k}}"

TRIPLE_JSONL="${TRIPLE_JSONL:-$DATASET_ROOT/manifests/vc_manifest.$DATASET_NAME.jsonl}"
TRIPLE_SUMMARY_JSON="${TRIPLE_SUMMARY_JSON:-$DATASET_ROOT/manifests/vc_manifest.$DATASET_NAME.summary.json}"
TRIPLE_PLAN_JSONL="${TRIPLE_PLAN_JSONL:-$DATASET_ROOT/intermediate/vc_manifest.large_mixed_mode.seedvc_plan.jsonl}"
TRIPLE_PLAN_SUMMARY_JSON="${TRIPLE_PLAN_SUMMARY_JSON:-$DATASET_ROOT/intermediate/vc_manifest.large_mixed_mode.seedvc_plan.summary.json}"
TRIPLE_SEEDVC_JOBS_JSONL="${TRIPLE_SEEDVC_JOBS_JSONL:-$DATASET_ROOT/seedvc_jobs/vc_manifest.large_mixed_mode.seedvc_jobs.jsonl}"
TRIPLE_SEEDVC_RESULTS_JSONL="${TRIPLE_SEEDVC_RESULTS_JSONL:-$DATASET_ROOT/seedvc_results/vc_manifest.large_mixed_mode.seedvc_results.jsonl}"
TRIPLE_SEEDVC_OUTPUT_ROOT="${TRIPLE_SEEDVC_OUTPUT_ROOT:-$DATASET_ROOT/seedvc_targets/large_mixed_mode_full}"
RUN_NAME="${RUN_NAME:-large_mixed_mode_seedvc_final}"

RUN_TRIPLE_STAGE="${RUN_TRIPLE_STAGE:-1}"
RUN_TRAIN_READY_STAGE="${RUN_TRAIN_READY_STAGE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE="${FORCE:-0}"
DRY_RUN=0

LANGUAGES="${LANGUAGES:-zh,en}"
EMIT_PAIR_TYPES="${EMIT_PAIR_TYPES:-no_text}"
MAX_ROWS="${MAX_ROWS:-0}"
MAX_PAIRS="${MAX_PAIRS:-0}"
MIN_DURATION_SEC="${MIN_DURATION_SEC:-1.0}"
MAX_DURATION_SEC="${MAX_DURATION_SEC:-30.0}"
TEXT_SOURCE_POLICY="${TEXT_SOURCE_POLICY:-different_speaker}"
INFER_SPEAKER_FROM="${INFER_SPEAKER_FROM:-auto}"
ALLOW_MISSING_AUDIO="${ALLOW_MISSING_AUDIO:-false}"
ALLOW_MISSING_SPEAKER="${ALLOW_MISSING_SPEAKER:-false}"
AUGMENT_SOURCE_JSONL="${AUGMENT_SOURCE_JSONL:-true}"
REQUIRE_TARGET_AUDIO="${REQUIRE_TARGET_AUDIO:-true}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100000}"
SEED="${SEED:-42}"
RUN_SEEDVC="${RUN_SEEDVC:-true}"
SEEDVC_SHARD_COUNT="${SEEDVC_SHARD_COUNT:-8}"
SEEDVC_GPU_IDS="${SEEDVC_GPU_IDS:-${GPU_IDS:-0,1,2,3,4,5,6,7}}"
SEEDVC_SKIP_EXISTING="${SEEDVC_SKIP_EXISTING:-1}"
SEEDVC_MAX_JOBS="${SEEDVC_MAX_JOBS:-0}"
SEEDVC_FAIL_FAST="${SEEDVC_FAIL_FAST:-0}"
SEEDVC_SHOW_MODEL_OUTPUT="${SEEDVC_SHOW_MODEL_OUTPUT:-0}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-25}"
LENGTH_ADJUST="${LENGTH_ADJUST:-1.0}"
INFERENCE_CFG_RATE="${INFERENCE_CFG_RATE:-0.7}"
SEEDVC_FP16="${SEEDVC_FP16:-true}"
RESUME_EXISTING="${RESUME_EXISTING:-true}"

N_VQ="${N_VQ:-32}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
CODEC_SHARD_COUNT="${CODEC_SHARD_COUNT:-8}"
SPEAKER_SHARD_COUNT="${SPEAKER_SHARD_COUNT:-8}"
PROSODY_SHARD_COUNT="${PROSODY_SHARD_COUNT:-16}"
RUN_PROSODY_FEATURES="${RUN_PROSODY_FEATURES:-1}"
SPEAKER_DEVICE="${SPEAKER_DEVICE:-cuda:0}"
ATTACH_REQUIRE_EMBEDDING_EXISTS="${ATTACH_REQUIRE_EMBEDDING_EXISTS:-0}"
WAIT_HEARTBEAT_SECS="${WAIT_HEARTBEAT_SECS:-60}"
GPU_KEEPALIVE="${GPU_KEEPALIVE:-0}"
GPU_KEEPALIVE_STAGES="${GPU_KEEPALIVE_STAGES:-speaker_extract,attach,prosody_extract}"
GPU_KEEPALIVE_GPU_IDS="${GPU_KEEPALIVE_GPU_IDS:-$GPU_IDS}"
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
  "$(dirname "$TRIPLE_JSONL")" \
  "$(dirname "$TRIPLE_SUMMARY_JSON")" \
  "$(dirname "$TRIPLE_PLAN_JSONL")" \
  "$(dirname "$TRIPLE_SEEDVC_JOBS_JSONL")" \
  "$(dirname "$TRIPLE_SEEDVC_RESULTS_JSONL")" \
  "$TRIPLE_SEEDVC_OUTPUT_ROOT" \
  "$DATASET_ROOT"

echo "=========================================="
echo "MOSS-CodecVC no-text full data pipeline"
echo "  ROOT=$ROOT"
echo "  PY=$PY"
echo "  DATASET_NAME=$DATASET_NAME"
echo "  DATASET_ROOT=$DATASET_ROOT"
echo "  RAW_INPUT_ROOT=$RAW_INPUT_ROOT"
echo "  RUN_TRIPLE_STAGE=$RUN_TRIPLE_STAGE"
echo "  RUN_TRAIN_READY_STAGE=$RUN_TRAIN_READY_STAGE"
echo "  TRIPLE_JSONL=$TRIPLE_JSONL"
echo "  TRIPLE_SUMMARY_JSON=$TRIPLE_SUMMARY_JSON"
echo "  LANGUAGES=$LANGUAGES"
echo "  EMIT_PAIR_TYPES=$EMIT_PAIR_TYPES"
echo "  MAX_ROWS=$MAX_ROWS"
echo "  MAX_PAIRS=$MAX_PAIRS"
echo "  RUN_SEEDVC=$RUN_SEEDVC"
echo "  SEEDVC_SHARD_COUNT=$SEEDVC_SHARD_COUNT"
echo "  SEEDVC_GPU_IDS=$SEEDVC_GPU_IDS"
echo "  SEEDVC_MAX_JOBS=$SEEDVC_MAX_JOBS"
echo "  CODEC_SHARD_COUNT=$CODEC_SHARD_COUNT"
echo "  SPEAKER_SHARD_COUNT=$SPEAKER_SHARD_COUNT"
echo "  PROSODY_SHARD_COUNT=$PROSODY_SHARD_COUNT"
echo "  SPEAKER_DEVICE=$SPEAKER_DEVICE"
echo "  ATTACH_REQUIRE_EMBEDDING_EXISTS=$ATTACH_REQUIRE_EMBEDDING_EXISTS"
echo "  GPU_KEEPALIVE=$GPU_KEEPALIVE"
echo "  GPU_KEEPALIVE_STAGES=$GPU_KEEPALIVE_STAGES"
echo "  SKIP_EXISTING=$SKIP_EXISTING"
echo "  FORCE=$FORCE"
echo "  DRY_RUN=$DRY_RUN"
echo "=========================================="

if truthy "$RUN_TRIPLE_STAGE"; then
  if truthy "$SKIP_EXISTING" && [ "$FORCE" != "1" ] && [ -s "$TRIPLE_JSONL" ] && [ -s "$TRIPLE_SUMMARY_JSON" ]; then
    echo "[skip] wav triple manifest already exists: $TRIPLE_JSONL"
  else
    echo "=========================================="
    echo "Pipeline part 1/2: raw JSONL -> Seed-VC wav triple manifest"
    echo "=========================================="
    run_cmd env \
      PY="$PY" \
      INPUT_ROOT="$RAW_INPUT_ROOT" \
      OUTPUT_JSONL="$TRIPLE_JSONL" \
      SUMMARY_JSON="$TRIPLE_SUMMARY_JSON" \
      PLAN_JSONL="$TRIPLE_PLAN_JSONL" \
      PLAN_SUMMARY_JSON="$TRIPLE_PLAN_SUMMARY_JSON" \
      SEEDVC_JOBS_JSONL="$TRIPLE_SEEDVC_JOBS_JSONL" \
      SEEDVC_RESULTS_JSONL="$TRIPLE_SEEDVC_RESULTS_JSONL" \
      SEEDVC_OUTPUT_ROOT="$TRIPLE_SEEDVC_OUTPUT_ROOT" \
      RUN_NAME="$RUN_NAME" \
      LANGUAGES="$LANGUAGES" \
      EMIT_PAIR_TYPES="$EMIT_PAIR_TYPES" \
      MAX_ROWS="$MAX_ROWS" \
      MAX_PAIRS="$MAX_PAIRS" \
      MIN_DURATION_SEC="$MIN_DURATION_SEC" \
      MAX_DURATION_SEC="$MAX_DURATION_SEC" \
      TEXT_SOURCE_POLICY="$TEXT_SOURCE_POLICY" \
      INFER_SPEAKER_FROM="$INFER_SPEAKER_FROM" \
      ALLOW_MISSING_AUDIO="$ALLOW_MISSING_AUDIO" \
      ALLOW_MISSING_SPEAKER="$ALLOW_MISSING_SPEAKER" \
      AUGMENT_SOURCE_JSONL="$AUGMENT_SOURCE_JSONL" \
      REQUIRE_TARGET_AUDIO="$REQUIRE_TARGET_AUDIO" \
      PROGRESS_EVERY="$PROGRESS_EVERY" \
      SEED="$SEED" \
      RUN_SEEDVC="$RUN_SEEDVC" \
      SEEDVC_SHARD_COUNT="$SEEDVC_SHARD_COUNT" \
      SEEDVC_GPU_IDS="$SEEDVC_GPU_IDS" \
      SEEDVC_SKIP_EXISTING="$SEEDVC_SKIP_EXISTING" \
      SEEDVC_MAX_JOBS="$SEEDVC_MAX_JOBS" \
      SEEDVC_FAIL_FAST="$SEEDVC_FAIL_FAST" \
      SEEDVC_SHOW_MODEL_OUTPUT="$SEEDVC_SHOW_MODEL_OUTPUT" \
      DIFFUSION_STEPS="$DIFFUSION_STEPS" \
      LENGTH_ADJUST="$LENGTH_ADJUST" \
      INFERENCE_CFG_RATE="$INFERENCE_CFG_RATE" \
      SEEDVC_FP16="$SEEDVC_FP16" \
      RESUME_EXISTING="$RESUME_EXISTING" \
      FORCE="$FORCE" \
      bash scripts/001026_run_large_mixed_mode_manifest_full.sh
  fi
fi

if [ ! -s "$TRIPLE_JSONL" ] && [ "$DRY_RUN" -eq 0 ]; then
  echo "Missing triple manifest after part 1: $TRIPLE_JSONL" >&2
  exit 2
fi

if truthy "$RUN_TRAIN_READY_STAGE"; then
  echo "=========================================="
  echo "Pipeline part 2/2: wav triple manifest -> train-ready SFT"
  echo "=========================================="
  train_ready_args=()
  if [ "$DRY_RUN" -eq 1 ]; then
    train_ready_args=(--dry-run)
  fi
  run_cmd env \
    PY="$PY" \
    DATASET_NAME="$DATASET_NAME" \
    DATASET_ROOT="$DATASET_ROOT" \
    INPUT_JSONL="$TRIPLE_JSONL" \
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
    MAX_ROWS="$MAX_ROWS" \
    SKIP_EXISTING="$SKIP_EXISTING" \
    FORCE="$FORCE" \
    WRITE_TRAIN_COMMAND="$WRITE_TRAIN_COMMAND" \
    bash scripts/001025_run_train_ready_no_text_ver2.sh "${train_ready_args[@]}"
fi

if [ "$DRY_RUN" -eq 0 ]; then
  "$PY" - "$PIPELINE_DONE_JSON" "$DATASET_NAME" "$DATASET_ROOT" "$RUN_TRIPLE_STAGE" "$RUN_TRAIN_READY_STAGE" "$TRIPLE_JSONL" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

done_path = Path(sys.argv[1])
dataset_name = sys.argv[2]
dataset_root = Path(sys.argv[3])
run_triple_stage = sys.argv[4]
run_train_ready_stage = sys.argv[5]
triple_jsonl = Path(sys.argv[6])
prosody_jsonl = dataset_root / "sft" / f"moss_codecvc_sft.{dataset_name}.with_light_ecapa_spk.with_prosody.jsonl"
attached_jsonl = dataset_root / "sft" / f"moss_codecvc_sft.{dataset_name}.with_light_ecapa_spk.jsonl"
active = prosody_jsonl if prosody_jsonl.exists() else attached_jsonl
train_done = active.exists() and active.with_name(active.name + ".done.json").exists()
status = "complete" if (run_train_ready_stage.lower() in {"0", "false", "no", "off"} or train_done) else "partial"
payload = {
    "status": status,
    "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "dataset_name": dataset_name,
    "dataset_root": str(dataset_root.resolve()),
    "run_triple_stage": run_triple_stage,
    "run_train_ready_stage": run_train_ready_stage,
    "triple_jsonl": str(triple_jsonl.resolve()),
    "triple_jsonl_exists": triple_jsonl.exists(),
    "train_jsonl": str(active.resolve()),
    "train_jsonl_exists": active.exists(),
    "train_jsonl_done_exists": train_done,
}
done_path.parent.mkdir(parents=True, exist_ok=True)
done_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
fi

echo "=========================================="
echo "Full no-text data pipeline finished"
echo "  dataset_root=$DATASET_ROOT"
echo "  triple_manifest=$TRIPLE_JSONL"
echo "  pipeline_done=$PIPELINE_DONE_JSON"
echo "=========================================="

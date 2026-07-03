#!/usr/bin/env bash
set -euo pipefail

# Stage 2 reusable entrypoint:
#   vcdata merged JSONLs -> text_prosody and/or no_text train-ready semantic data.
#
# Stage 1 is scripts/001048_submit_vcdata_clone_qz.sh.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$ROOT"

PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
PY_MOSS="${PY_MOSS:-$PY}"
PY_SEEDVC="${PY_SEEDVC:-/inspire/ssd/project/embodied-multimodality/public/yqzhang/miniconda3/envs/contts-train/bin/python}"
PYTHON_MAIN="${PYTHON_MAIN:-$PY}"
PYTHON_ASR="${PYTHON_ASR:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download}"

PAIR_CONSTRUCTION_ROOT="${PAIR_CONSTRUCTION_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction}"
SEED_VC_DIR="${SEED_VC_DIR:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction_prosody_routes/third_party/seed-vc}"
SEEDVC_DEPS_DIR="${SEEDVC_DEPS_DIR:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction_prosody_routes/.deps/seedvc}"

DATASET_NAME="${DATASET_NAME:-zh11w_en11w_0005_0015_vcdata_first}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/trainset/$DATASET_NAME}"
VCDATA_ROOT="${VCDATA_ROOT:-$DATASET_ROOT/vcdata}"
VCDATA_JSONLS_FILE="${VCDATA_JSONLS_FILE:-$DATASET_ROOT/vcdata_jsonls.txt}"
VCDATA_JSONLS="${VCDATA_JSONLS:-}"

RUN_TEXT_BRANCH="${RUN_TEXT_BRANCH:-1}"
RUN_NO_TEXT_BRANCH="${RUN_NO_TEXT_BRANCH:-1}"
RUN_NO_TEXT_TRIPLE_STAGE="${RUN_NO_TEXT_TRIPLE_STAGE:-1}"
RUN_NO_TEXT_TRAIN_READY_STAGE="${RUN_NO_TEXT_TRAIN_READY_STAGE:-1}"
RUN_NO_TEXT_SEMANTIC_STAGE="${RUN_NO_TEXT_SEMANTIC_STAGE:-1}"
FORCE="${FORCE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN=0

LANGUAGES="${LANGUAGES:-zh,en}"
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
CODEC_GPU_IDS="${CODEC_GPU_IDS:-$GPU_IDS}"
SPEAKER_GPU_IDS="${SPEAKER_GPU_IDS:-$GPU_IDS}"
CODEC_SHARD_COUNT="${CODEC_SHARD_COUNT:-8}"
SPEAKER_SHARD_COUNT="${SPEAKER_SHARD_COUNT:-8}"
PROSODY_SHARD_COUNT="${PROSODY_SHARD_COUNT:-16}"
TRAIN_READY_GPU_KEEPALIVE="${TRAIN_READY_GPU_KEEPALIVE:-0}"
NO_TEXT_TRAIN_READY_GPU_KEEPALIVE="${NO_TEXT_TRAIN_READY_GPU_KEEPALIVE:-$TRAIN_READY_GPU_KEEPALIVE}"
TEXT_TRAIN_READY_GPU_KEEPALIVE="${TEXT_TRAIN_READY_GPU_KEEPALIVE:-$TRAIN_READY_GPU_KEEPALIVE}"

TEXT_DATASET_NAME="${TEXT_DATASET_NAME:-${DATASET_NAME}_text_prosody}"
TEXT_DATASET_ROOT="${TEXT_DATASET_ROOT:-$ROOT/trainset/$TEXT_DATASET_NAME}"
TEXT_RUN_NAME="${TEXT_RUN_NAME:-$TEXT_DATASET_NAME}"
TEXT_TIMBRE_REF_POLICY="${TEXT_TIMBRE_REF_POLICY:-random_different_text}"
TEXT_TIMBRE_REF_SEED="${TEXT_TIMBRE_REF_SEED:-20260627}"
TEXT_MAX_JOBS="${TEXT_MAX_JOBS:-0}"
TEXT_MAX_JOBS_PER_LANGUAGE="${TEXT_MAX_JOBS_PER_LANGUAGE:-0}"
TEXT_MAX_ROWS_PER_INPUT="${TEXT_MAX_ROWS_PER_INPUT:-0}"
TEXT_MIN_BEST_SIMILARITY="${TEXT_MIN_BEST_SIMILARITY:-0.0}"
TEXT_SKIP_FLAGS="${TEXT_SKIP_FLAGS:-}"
TEXT_SEMANTIC_RUN_P0="${TEXT_SEMANTIC_RUN_P0:-1}"
TEXT_ASR_NUM_SHARDS="${TEXT_ASR_NUM_SHARDS:-8}"
TEXT_HUBERT_NUM_SHARDS="${TEXT_HUBERT_NUM_SHARDS:-16}"
TEXT_HUBERT_DEVICES="${TEXT_HUBERT_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7,cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7}"
TEXT_SEMANTIC_GPU_KEEPALIVE="${TEXT_SEMANTIC_GPU_KEEPALIVE:-0}"

NO_TEXT_DATASET_NAME="${NO_TEXT_DATASET_NAME:-${DATASET_NAME}_no_text}"
NO_TEXT_DATASET_ROOT="${NO_TEXT_DATASET_ROOT:-$ROOT/trainset/$NO_TEXT_DATASET_NAME}"
NO_TEXT_RUN_NAME="${NO_TEXT_RUN_NAME:-$NO_TEXT_DATASET_NAME}"
NO_TEXT_MANIFEST_JSONL="${NO_TEXT_MANIFEST_JSONL:-$NO_TEXT_DATASET_ROOT/manifests/vc_manifest.$NO_TEXT_DATASET_NAME.jsonl}"
NO_TEXT_MANIFEST_SUMMARY_JSON="${NO_TEXT_MANIFEST_SUMMARY_JSON:-$NO_TEXT_DATASET_ROOT/manifests/vc_manifest.$NO_TEXT_DATASET_NAME.summary.json}"
NO_TEXT_SEEDVC_WORK_ROOT="${NO_TEXT_SEEDVC_WORK_ROOT:-$NO_TEXT_DATASET_ROOT/intermediate/no_text_seedvc}"
NO_TEXT_SEEDVC_JOBS_JSONL="${NO_TEXT_SEEDVC_JOBS_JSONL:-$NO_TEXT_SEEDVC_WORK_ROOT/no_text_seedvc_jobs.jsonl}"
NO_TEXT_SEEDVC_RESULTS_JSONL="${NO_TEXT_SEEDVC_RESULTS_JSONL:-$NO_TEXT_SEEDVC_WORK_ROOT/no_text_seedvc_results.jsonl}"
NO_TEXT_SEEDVC_TARGET_AUDIO_ROOT="${NO_TEXT_SEEDVC_TARGET_AUDIO_ROOT:-$NO_TEXT_DATASET_ROOT/seedvc_targets}"
NO_TEXT_TIMBRE_REF_POLICY="${NO_TEXT_TIMBRE_REF_POLICY:-random_original_different_row}"
NO_TEXT_TIMBRE_REF_SEED="${NO_TEXT_TIMBRE_REF_SEED:-20260629}"
NO_TEXT_REQUIRE_DIFFERENT_TIMBRE_TEXT="${NO_TEXT_REQUIRE_DIFFERENT_TIMBRE_TEXT:-1}"
NO_TEXT_MAX_JOBS="${NO_TEXT_MAX_JOBS:-0}"
NO_TEXT_MAX_JOBS_PER_LANGUAGE="${NO_TEXT_MAX_JOBS_PER_LANGUAGE:-0}"
NO_TEXT_MAX_ROWS_PER_INPUT="${NO_TEXT_MAX_ROWS_PER_INPUT:-0}"
NO_TEXT_MIN_BEST_SIMILARITY="${NO_TEXT_MIN_BEST_SIMILARITY:-0.0}"
NO_TEXT_SKIP_FLAGS="${NO_TEXT_SKIP_FLAGS:-}"
NO_TEXT_SEMANTIC_ASR_NUM_SHARDS="${NO_TEXT_SEMANTIC_ASR_NUM_SHARDS:-8}"
NO_TEXT_SEMANTIC_HUBERT_NUM_SHARDS="${NO_TEXT_SEMANTIC_HUBERT_NUM_SHARDS:-16}"
NO_TEXT_SEMANTIC_HUBERT_SOURCE="${NO_TEXT_SEMANTIC_HUBERT_SOURCE:-both}"
NO_TEXT_SEMANTIC_RESUME_SHARDS="${NO_TEXT_SEMANTIC_RESUME_SHARDS:-1}"
NO_TEXT_SEMANTIC_GPU_KEEPALIVE="${NO_TEXT_SEMANTIC_GPU_KEEPALIVE:-0}"

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

resolve_vcdata_jsonls() {
  if [ -n "$VCDATA_JSONLS" ]; then
    printf '%s\n' "$VCDATA_JSONLS"
    return 0
  fi
  if [ -s "$VCDATA_JSONLS_FILE" ]; then
    paste -sd, "$VCDATA_JSONLS_FILE"
    return 0
  fi
  find "$VCDATA_ROOT" -mindepth 2 -maxdepth 2 -name merged.stepaudio_input.all.jsonl | sort | paste -sd,
}

vcdata_csv="$(resolve_vcdata_jsonls)"
if [ -z "$vcdata_csv" ] && { truthy "$RUN_TEXT_BRANCH" || truthy "$RUN_NO_TEXT_BRANCH"; }; then
  echo "ERROR: no vcdata JSONLs found. Expected $VCDATA_JSONLS_FILE or $VCDATA_ROOT/*/merged.stepaudio_input.all.jsonl" >&2
  exit 2
fi
vcdata_count=$(printf '%s' "$vcdata_csv" | tr ',' '\n' | awk 'NF { c += 1 } END { print c + 0 }')

mkdir -p "$DATASET_ROOT" "$TEXT_DATASET_ROOT" "$NO_TEXT_DATASET_ROOT" "$NO_TEXT_SEEDVC_WORK_ROOT"

echo "=========================================="
echo "VC data branch pipeline from vcdata"
echo "  ROOT=$ROOT"
echo "  DATASET_NAME=$DATASET_NAME"
echo "  DATASET_ROOT=$DATASET_ROOT"
echo "  VCDATA_ROOT=$VCDATA_ROOT"
echo "  vcdata_count=$vcdata_count"
echo "  RUN_TEXT_BRANCH=$RUN_TEXT_BRANCH TEXT_DATASET_NAME=$TEXT_DATASET_NAME"
echo "  RUN_NO_TEXT_BRANCH=$RUN_NO_TEXT_BRANCH NO_TEXT_DATASET_NAME=$NO_TEXT_DATASET_NAME"
echo "  NO_TEXT_TIMBRE_REF_POLICY=$NO_TEXT_TIMBRE_REF_POLICY"
echo "  SEEDVC_SHARD_COUNT=$SEEDVC_SHARD_COUNT GPU_IDS=$GPU_IDS"
echo "  CODEC_SHARD_COUNT=$CODEC_SHARD_COUNT SPEAKER_SHARD_COUNT=$SPEAKER_SHARD_COUNT PROSODY_SHARD_COUNT=$PROSODY_SHARD_COUNT"
echo "  SKIP_EXISTING=$SKIP_EXISTING FORCE=$FORCE DRY_RUN=$DRY_RUN"
echo "=========================================="

run_cmd "$PY" -m py_compile \
  scripts/001034_build_text_prosody_from_mosstts_vcdata.py \
  scripts/001049_build_no_text_from_mosstts_vcdata.py \
  scripts/001002_encode_codec_tokens.py \
  scripts/001003_build_moss_sft_jsonl.py \
  scripts/001007_build_speaker_embedding_plan.py \
  scripts/001010_attach_speaker_embeddings.py \
  scripts/001011_extract_speaker_embeddings.py \
  scripts/001015_extract_prosody_content_features.py \
  scripts/001019_extract_content_tokens.py \
  scripts/001020_extract_hubert_semantic_features.py

if truthy "$RUN_TEXT_BRANCH"; then
  echo "=========================================="
  echo "Branch A: vcdata -> text_prosody full semantic data"
  echo "=========================================="
  text_args=()
  if [ "$DRY_RUN" -eq 1 ]; then
    text_args=(--dry-run)
  fi
  export RUN_P0="$TEXT_SEMANTIC_RUN_P0"
  run_cmd env \
    PY="$PY" \
    PY_MOSS="$PY_MOSS" \
    PY_SEEDVC="$PY_SEEDVC" \
    PYTHON_MAIN="$PYTHON_MAIN" \
    PYTHON_ASR="$PYTHON_ASR" \
    DOWNLOAD_ROOT="$DOWNLOAD_ROOT" \
    DATASET_NAME="$TEXT_DATASET_NAME" \
    DATASET_ROOT="$TEXT_DATASET_ROOT" \
    RUN_NAME="$TEXT_RUN_NAME" \
    VCDATA_JSONLS="$vcdata_csv" \
    LANGUAGES="$LANGUAGES" \
    TIMBRE_REF_POLICY="$TEXT_TIMBRE_REF_POLICY" \
    TIMBRE_REF_SEED="$TEXT_TIMBRE_REF_SEED" \
    MAX_JOBS="$TEXT_MAX_JOBS" \
    MAX_JOBS_PER_LANGUAGE="$TEXT_MAX_JOBS_PER_LANGUAGE" \
    MAX_ROWS_PER_INPUT="$TEXT_MAX_ROWS_PER_INPUT" \
    MIN_BEST_SIMILARITY="$TEXT_MIN_BEST_SIMILARITY" \
    SKIP_FLAGS="$TEXT_SKIP_FLAGS" \
    GPU_IDS="$GPU_IDS" \
    SEEDVC_GPU_IDS="$SEEDVC_GPU_IDS" \
    SEEDVC_SHARD_COUNT="$SEEDVC_SHARD_COUNT" \
    DIFFUSION_STEPS="$DIFFUSION_STEPS" \
    LENGTH_ADJUST="$LENGTH_ADJUST" \
    INFERENCE_CFG_RATE="$INFERENCE_CFG_RATE" \
    FP16="$FP16" \
    SEEDVC_SKIP_EXISTING="$SEEDVC_SKIP_EXISTING" \
    FAIL_FAST="$FAIL_FAST" \
    SHOW_MODEL_OUTPUT="$SHOW_MODEL_OUTPUT" \
    MIN_TARGET_AUDIO_BYTES="$MIN_TARGET_AUDIO_BYTES" \
    N_VQ="$N_VQ" \
    CODEC_GPU_IDS="$CODEC_GPU_IDS" \
    SPEAKER_GPU_IDS="$SPEAKER_GPU_IDS" \
    CODEC_SHARD_COUNT="$CODEC_SHARD_COUNT" \
    SPEAKER_SHARD_COUNT="$SPEAKER_SHARD_COUNT" \
    PROSODY_SHARD_COUNT="$PROSODY_SHARD_COUNT" \
    TRAIN_READY_GPU_KEEPALIVE="$TEXT_TRAIN_READY_GPU_KEEPALIVE" \
    ASR_NUM_SHARDS="$TEXT_ASR_NUM_SHARDS" \
    HUBERT_NUM_SHARDS="$TEXT_HUBERT_NUM_SHARDS" \
    HUBERT_DEVICES="$TEXT_HUBERT_DEVICES" \
    HUBERT_PRE_SPLIT=1 \
    HUBERT_SOURCE=target \
    SEMANTIC_GPU_KEEPALIVE="$TEXT_SEMANTIC_GPU_KEEPALIVE" \
    SKIP_EXISTING="$SKIP_EXISTING" \
    FORCE="$FORCE" \
    bash scripts/001043_run_text_prosody_full_semantic_pipeline.sh "${text_args[@]}"
fi

if truthy "$RUN_NO_TEXT_BRANCH"; then
  echo "=========================================="
  echo "Branch B: vcdata -> no_text full semantic data"
  echo "=========================================="

  if truthy "$RUN_NO_TEXT_TRIPLE_STAGE"; then
    if should_reuse "$NO_TEXT_MANIFEST_JSONL"; then
      echo "[skip] no_text manifest exists: $NO_TEXT_MANIFEST_JSONL"
    else
      different_timbre_text_arg=(--require-different-timbre-text)
      if ! truthy "$NO_TEXT_REQUIRE_DIFFERENT_TIMBRE_TEXT"; then
        different_timbre_text_arg=(--no-require-different-timbre-text)
      fi
      run_cmd "$PY_MOSS" scripts/001049_build_no_text_from_mosstts_vcdata.py \
        prepare \
        --vcdata-jsonl "$vcdata_csv" \
        --jobs-jsonl "$NO_TEXT_SEEDVC_JOBS_JSONL" \
        --target-audio-root "$NO_TEXT_SEEDVC_TARGET_AUDIO_ROOT" \
        --run-name "$NO_TEXT_RUN_NAME" \
        --languages "$LANGUAGES" \
        --max-rows-per-input "$NO_TEXT_MAX_ROWS_PER_INPUT" \
        --max-jobs "$NO_TEXT_MAX_JOBS" \
        --max-jobs-per-language "$NO_TEXT_MAX_JOBS_PER_LANGUAGE" \
        --timbre-ref-policy "$NO_TEXT_TIMBRE_REF_POLICY" \
        --timbre-ref-seed "$NO_TEXT_TIMBRE_REF_SEED" \
        --min-best-similarity "$NO_TEXT_MIN_BEST_SIMILARITY" \
        --skip-flags "$NO_TEXT_SKIP_FLAGS" \
        --summary-json "$NO_TEXT_SEEDVC_WORK_ROOT/no_text_seedvc_jobs.summary.json" \
        "${different_timbre_text_arg[@]}" \
        --overwrite

      export PYTHONPATH="${SEEDVC_DEPS_DIR}:${PAIR_CONSTRUCTION_ROOT}/scripts:${SEED_VC_DIR}:${PYTHONPATH:-}"
      export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
      # Seed-VC keeps Whisper/BigVGAN in its own cache. Do not let a generic
      # outer HF_HOME override this, otherwise offline Qizhi jobs cannot load.
      export HF_HOME="$SEED_VC_DIR/checkpoints/hf_cache"
      export HUGGINGFACE_HUB_CACHE="$SEED_VC_DIR/checkpoints/hf_cache"
      export TRANSFORMERS_CACHE="$SEED_VC_DIR/checkpoints/hf_cache"
      export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
      export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

      run_cmd env \
        PY="$PY_SEEDVC" \
        JOBS_JSONL="$NO_TEXT_SEEDVC_JOBS_JSONL" \
        RESULTS_JSONL="$NO_TEXT_SEEDVC_RESULTS_JSONL" \
        SEED_VC_DIR="$SEED_VC_DIR" \
        SEEDVC_GPU_IDS="$SEEDVC_GPU_IDS" \
        SEEDVC_SHARD_COUNT="$SEEDVC_SHARD_COUNT" \
        DIFFUSION_STEPS="$DIFFUSION_STEPS" \
        LENGTH_ADJUST="$LENGTH_ADJUST" \
        INFERENCE_CFG_RATE="$INFERENCE_CFG_RATE" \
        FP16="$FP16" \
        SKIP_EXISTING="$SEEDVC_SKIP_EXISTING" \
        FAIL_FAST="$FAIL_FAST" \
        SHOW_MODEL_OUTPUT="$SHOW_MODEL_OUTPUT" \
        bash "$PAIR_CONSTRUCTION_ROOT/scripts/run_seedvc_jobs_sharded.sh"

      run_cmd "$PY_MOSS" scripts/001049_build_no_text_from_mosstts_vcdata.py \
        collect \
        --jobs-jsonl "$NO_TEXT_SEEDVC_JOBS_JSONL" \
        --results-jsonl "$NO_TEXT_SEEDVC_RESULTS_JSONL" \
        --output-jsonl "$NO_TEXT_MANIFEST_JSONL" \
        --run-name "$NO_TEXT_RUN_NAME" \
        --summary-json "$NO_TEXT_MANIFEST_SUMMARY_JSON" \
        --min-target-audio-bytes "$MIN_TARGET_AUDIO_BYTES" \
        --overwrite
    fi
  fi

  if [ "$DRY_RUN" -eq 0 ] && [ ! -s "$NO_TEXT_MANIFEST_JSONL" ]; then
    echo "ERROR: missing no_text manifest: $NO_TEXT_MANIFEST_JSONL" >&2
    exit 2
  fi

  if truthy "$RUN_NO_TEXT_TRAIN_READY_STAGE"; then
    train_ready_args=()
    if [ "$DRY_RUN" -eq 1 ]; then
      train_ready_args=(--dry-run)
    fi
    run_cmd env \
      PY="$PY" \
      DATASET_NAME="$NO_TEXT_DATASET_NAME" \
      DATASET_ROOT="$NO_TEXT_DATASET_ROOT" \
      INPUT_JSONL="$NO_TEXT_MANIFEST_JSONL" \
      EMIT_MODES=no_text \
      N_VQ="$N_VQ" \
      GPU_IDS="$GPU_IDS" \
      CODEC_SHARD_COUNT="$CODEC_SHARD_COUNT" \
      SPEAKER_SHARD_COUNT="$SPEAKER_SHARD_COUNT" \
      PROSODY_SHARD_COUNT="$PROSODY_SHARD_COUNT" \
      RUN_PROSODY_FEATURES=1 \
      GPU_KEEPALIVE="$NO_TEXT_TRAIN_READY_GPU_KEEPALIVE" \
      SKIP_EXISTING="$SKIP_EXISTING" \
      FORCE="$FORCE" \
      bash scripts/001025_run_train_ready_no_text_ver2.sh "${train_ready_args[@]}"
  fi

  no_text_train_ready_jsonl="$NO_TEXT_DATASET_ROOT/sft/moss_codecvc_sft.$NO_TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.jsonl"
  if truthy "$RUN_NO_TEXT_SEMANTIC_STAGE"; then
    if [ "$DRY_RUN" -eq 0 ] && [ ! -s "$no_text_train_ready_jsonl" ]; then
      echo "ERROR: missing no_text train-ready JSONL: $no_text_train_ready_jsonl" >&2
      exit 2
    fi
    semantic_args=()
    if [ "$DRY_RUN" -eq 1 ]; then
      semantic_args=()
    fi
    run_cmd env \
      PROJECT_ROOT="$ROOT" \
      PYTHON_MAIN="$PYTHON_MAIN" \
      PYTHON_ASR="$PYTHON_ASR" \
      DOWNLOAD_ROOT="$DOWNLOAD_ROOT" \
      TRAINSET_DIR="$NO_TEXT_DATASET_ROOT" \
      SFT_DIR="$NO_TEXT_DATASET_ROOT/sft" \
      INPUT_JSONL="$no_text_train_ready_jsonl" \
      OUTPUT_PREFIX="$NO_TEXT_DATASET_ROOT/sft/moss_codecvc_sft.$NO_TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody" \
      ASR_NUM_SHARDS="$NO_TEXT_SEMANTIC_ASR_NUM_SHARDS" \
      ASR_DEVICES="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7" \
      HUBERT_NUM_SHARDS="$NO_TEXT_SEMANTIC_HUBERT_NUM_SHARDS" \
      HUBERT_DEVICES="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7,cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7" \
      HUBERT_SOURCE="$NO_TEXT_SEMANTIC_HUBERT_SOURCE" \
      RESUME_SHARDS="$NO_TEXT_SEMANTIC_RESUME_SHARDS" \
      SEMANTIC_GPU_KEEPALIVE="$NO_TEXT_SEMANTIC_GPU_KEEPALIVE" \
      SEMANTIC_GPU_KEEPALIVE_GPU_IDS="$GPU_IDS" \
      FILTER_CONTENT_KEEP=1 \
      RUN_P0=1 \
      RUN_P1=1 \
      RUN_P2=1 \
      bash scripts/001018_prepare_ver2_1_content_semantic_68w.sh "${semantic_args[@]}"
  fi
fi

echo "=========================================="
echo "VC branch pipeline finished"
echo "  text_dataset_root=$TEXT_DATASET_ROOT"
echo "  no_text_dataset_root=$NO_TEXT_DATASET_ROOT"
echo "=========================================="

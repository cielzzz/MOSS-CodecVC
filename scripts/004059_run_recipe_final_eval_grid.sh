#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
ROOT=$(CDPATH= cd "${SCRIPT_DIR}/.." && pwd)

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
ASR_PYTHON="${ASR_PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"

RUN_DIR="${RUN_DIR:-$ROOT/outputs/lora_runs/ver2_8_timbre_repair_recipe_final_varlen_block_permuted_cref_prompt_a4_refsup_cosramp_infonce_dropout_steps30000}"
RUN_LABEL="${RUN_LABEL:-codecvc-ver2-8-recipe-final-varlen-block}"
CHECKPOINT="${CHECKPOINT:-$RUN_DIR/step-30000}"
STEP_LABEL="$(basename "$CHECKPOINT")"

OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/testset/outputs/ver2_8_recipe_final_eval_grid_20260705}"
DOCS_MD="${DOCS_MD:-$ROOT/docs/ver2_8_recipe_final_eval_grid_20260705.md}"
ASSET_PREFIX="${ASSET_PREFIX:-$ROOT/docs/assets/ver2_8_recipe_final_eval_grid_20260705}"

FULL_VALIDATION_JSONL="${FULL_VALIDATION_JSONL:-$ROOT/testset/validation/seedtts_vc_ver2_3_validation.jsonl}"
FULL_MODE="${FULL_MODE:-all}"
BADCASE_VALIDATION_JSONL="${BADCASE_VALIDATION_JSONL:-$ROOT/testset/validation/ver2_8_p0_e1_no_text_badcase24_seedtts.jsonl}"

WINDOWS="${WINDOWS:-4,8,10}"
CFG_SCALES="${CFG_SCALES:-1.0,1.5}"
SEED="${SEED:-1234}"
BADCASE_WINDOWS="${BADCASE_WINDOWS:-8}"
BADCASE_CFG_SCALES="${BADCASE_CFG_SCALES:-1.0}"
BADCASE_SEEDS="${BADCASE_SEEDS:-1234,2025,3407}"

RUN_FULL="${RUN_FULL:-1}"
RUN_BADCASE="${RUN_BADCASE:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
DRY_RUN="${DRY_RUN:-0}"

GPU_COUNT="${GPU_COUNT:-}"
NUM_SHARDS="${NUM_SHARDS:-${GPU_COUNT:-}}"
ASR_NUM_SHARDS="${ASR_NUM_SHARDS:-${NUM_SHARDS:-}}"
BUILD_PAGE="${BUILD_PAGE:-0}"
OVERWRITE_INFER="${OVERWRITE_INFER:-1}"
RESET_MANIFESTS="${RESET_MANIFESTS:-1}"
PERSISTENT_INFER="${PERSISTENT_INFER:-1}"
RUN_ASR="${RUN_ASR:-1}"
EXTRA_SPEAKER_ENCODER="${EXTRA_SPEAKER_ENCODER:-speechbrain_ecapa}"
EXTRA_SPEAKER_DEVICE="${EXTRA_SPEAKER_DEVICE:-${SPEAKER_DEVICE:-cuda:0}}"
SPEECHBRAIN_ECAPA_MODEL_SOURCE="${SPEECHBRAIN_ECAPA_MODEL_SOURCE:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/models/speechbrain/spkrec-ecapa-voxceleb}"

TIMBRE_SIDE_ONLY="${TIMBRE_SIDE_ONLY:-0}"
REF_PROMPT_CODEC_PERMUTATION_MODE="${REF_PROMPT_CODEC_PERMUTATION_MODE:-block_shuffle}"
REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE="${REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE:-12.5}"
REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS="${REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS:-0.4}"
REF_PROMPT_CODEC_PERMUTATION_BOOTSTRAP="${REF_PROMPT_CODEC_PERMUTATION_BOOTSTRAP:-block}"
REF_PROMPT_CODEC_PERMUTATION_SEED="${REF_PROMPT_CODEC_PERMUTATION_SEED:-1234}"

if [ ! -d "$CHECKPOINT" ] && [ "$DRY_RUN" != "1" ]; then
  echo "ERROR: checkpoint does not exist: $CHECKPOINT" >&2
  exit 1
fi
if [ ! -d "$CHECKPOINT" ] && [ "$DRY_RUN" = "1" ]; then
  echo "[recipe-final-grid] DRY_RUN=1 with missing checkpoint: $CHECKPOINT"
fi
if [ ! -f "$FULL_VALIDATION_JSONL" ]; then
  echo "ERROR: full validation JSONL not found: $FULL_VALIDATION_JSONL" >&2
  exit 1
fi
if [ ! -f "$BADCASE_VALIDATION_JSONL" ]; then
  echo "ERROR: badcase validation JSONL not found: $BADCASE_VALIDATION_JSONL" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$(dirname "$DOCS_MD")" "$(dirname "$ASSET_PREFIX")"

split_csv() {
  local value="$1"
  local old_ifs="$IFS"
  IFS=','
  for item in $value; do
    if [ -n "$item" ]; then
      printf '%s\n' "$item"
    fi
  done
  IFS="$old_ifs"
}

tag_value() {
  printf '%s' "$1" | sed 's/\./p/g'
}

run_specs=()

run_one_eval() {
  local validation_jsonl="$1"
  local mode="$2"
  local eval_kind="$3"
  local window_s="$4"
  local cfg_scale="$5"
  local seed="$6"

  local window_tag
  local cfg_tag
  window_tag=$(tag_value "$window_s")
  cfg_tag=$(tag_value "$cfg_scale")
  local run_id="${RUN_LABEL}_${STEP_LABEL}_${eval_kind}_w${window_tag}s_cfg${cfg_tag}_seed${seed}"
  local output_dir="$OUTPUT_ROOT/$run_id"
  run_specs+=("${run_id}=${output_dir}")

  echo "[recipe-final-grid] run_id=$run_id validation=$validation_jsonl mode=$mode window=${window_s}s cfg=$cfg_scale seed=$seed"
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi

  env \
    PYTHON="$PYTHON" \
    ASR_PYTHON="$ASR_PYTHON" \
    VALIDATION_JSONL="$validation_jsonl" \
    MODEL_PATH="$CHECKPOINT" \
    RUN_ID="$run_id" \
    RUN_LABEL="$run_id" \
    OUTPUT_DIR="$output_dir" \
    MODE="$mode" \
    MAX_CASES=0 \
    DECODING_PROFILE=default \
    PERSISTENT_INFER="$PERSISTENT_INFER" \
    OVERWRITE_INFER="$OVERWRITE_INFER" \
    RESET_MANIFESTS="$RESET_MANIFESTS" \
    RUN_ASR="$RUN_ASR" \
    RUN_SUMMARY=1 \
    BUILD_PAGE="$BUILD_PAGE" \
    PAGE_DIR="$output_dir/listening_page" \
    GPU_COUNT="$GPU_COUNT" \
    NUM_SHARDS="$NUM_SHARDS" \
    ASR_NUM_SHARDS="$ASR_NUM_SHARDS" \
    SEED="$seed" \
    SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH=0.0 \
    TEMPERATURE=0.7 \
    NO_TEXT_AUDIO_TEMPERATURE=1.1 \
    NO_TEXT_AUDIO_TOP_P=0.7 \
    NO_TEXT_AUDIO_TOP_K=20 \
    AUDIO_TEMPERATURE=1.1 \
    AUDIO_TOP_P=0.7 \
    AUDIO_TOP_K=20 \
    TIMBRE_SIDE_ONLY="$TIMBRE_SIDE_ONLY" \
    REF_PROMPT_CODEC_PERMUTATION=1 \
    REF_PROMPT_CODEC_PERMUTATION_MODE="$REF_PROMPT_CODEC_PERMUTATION_MODE" \
    REF_PROMPT_CODEC_PERMUTATION_MIN_SECONDS="$window_s" \
    REF_PROMPT_CODEC_PERMUTATION_MAX_SECONDS="$window_s" \
    REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE="$REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE" \
    REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS="$REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS" \
    REF_PROMPT_CODEC_PERMUTATION_BOOTSTRAP="$REF_PROMPT_CODEC_PERMUTATION_BOOTSTRAP" \
    REF_PROMPT_CODEC_PERMUTATION_SEED="$REF_PROMPT_CODEC_PERMUTATION_SEED" \
    TIMBRE_CFG_SCALE="$cfg_scale" \
    bash "$ROOT/scripts/004039_run_seedtts_validation_eval.sh"
}

if [ "$RUN_FULL" = "1" ]; then
  while IFS= read -r window_s; do
    while IFS= read -r cfg_scale; do
      run_one_eval "$FULL_VALIDATION_JSONL" "$FULL_MODE" "full320" "$window_s" "$cfg_scale" "$SEED"
    done < <(split_csv "$CFG_SCALES")
  done < <(split_csv "$WINDOWS")
fi

if [ "$RUN_BADCASE" = "1" ]; then
  while IFS= read -r window_s; do
    while IFS= read -r cfg_scale; do
      while IFS= read -r bad_seed; do
        run_one_eval "$BADCASE_VALIDATION_JSONL" "no_text" "badcase24" "$window_s" "$cfg_scale" "$bad_seed"
      done < <(split_csv "$BADCASE_SEEDS")
    done < <(split_csv "$BADCASE_CFG_SCALES")
  done < <(split_csv "$BADCASE_WINDOWS")
fi

if [ "$RUN_SUMMARY" = "1" ]; then
  if [ "$DRY_RUN" = "1" ]; then
    echo "[recipe-final-grid] DRY_RUN=1, skip summary generation"
  else
    summary_args=()
    for spec in "${run_specs[@]}"; do
      summary_args+=(--run "$spec")
    done
    "$PYTHON" "$ROOT/scripts/004048_summarize_seedtts_ablation_metrics.py" \
      --validation-jsonl "$FULL_VALIDATION_JSONL" \
      "${summary_args[@]}" \
      --output-csv "${ASSET_PREFIX}.cases.csv" \
      --summary-json "${ASSET_PREFIX}.ablation_summary.json" \
      --summary-md "${ASSET_PREFIX}.ablation_summary.md" \
      --speaker-device "${SPEAKER_DEVICE:-cuda:0}" \
      --extra-speaker-encoder "$EXTRA_SPEAKER_ENCODER" \
      --extra-speaker-device "$EXTRA_SPEAKER_DEVICE" \
      --speechbrain-ecapa-model-source "$SPEECHBRAIN_ECAPA_MODEL_SOURCE"
    "$PYTHON" "$ROOT/scripts/004060_summarize_recipe_final_eval_grid.py" \
      --ablation-summary-json "${ASSET_PREFIX}.ablation_summary.json" \
      "${summary_args[@]}" \
      --output-json "${ASSET_PREFIX}.summary.json" \
      --output-csv "${ASSET_PREFIX}.summary.csv" \
      --output-md "$DOCS_MD"
  fi
fi

echo "[recipe-final-grid] done output_root=$OUTPUT_ROOT docs=$DOCS_MD"

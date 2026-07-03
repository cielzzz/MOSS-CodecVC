#!/usr/bin/env sh
set -eu

# Edit this block for daily no_text inference.
PROJECT_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/003003_run_moss_codecvc_infer.sh"

# Leave MODEL_PATH empty to use PREFERRED_STEP under DEFAULT_VER2_RUN_DIR.
# Set PREFERRED_STEP=latest to use the latest step-* checkpoint.
# Override MODEL_PATH/SOURCE_AUDIO/TIMBRE_REF_AUDIO/OUTPUT_DIR/OUTPUT_WAV at invocation time for one-off runs.
DEFAULT_VER2_RUN_DIR="${DEFAULT_VER2_RUN_DIR:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/outputs/lora_runs/ver2_8_sideonly_wavlmbnf_codecres_textrep10_mergednotext_modeawareprosody_lora_r16_a32_gbs64}"
MODEL_PATH="${MODEL_PATH:-}"
PREFERRED_STEP="${PREFERRED_STEP:-latest}"

SOURCE_AUDIO="${SOURCE_AUDIO:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/testset/source/zh_a_000003_source.flac}"  # 你在胡扯什么
TIMBRE_REF_AUDIO="${TIMBRE_REF_AUDIO:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/testset/timbre_ref/zh_a_000001_timbre.wav}"

DEVICE="${DEVICE:-auto}"
DEBUG_GENERATION_STRUCTURE="${DEBUG_GENERATION_STRUCTURE:-1}"
TIMBRE_SIDE_ONLY="${TIMBRE_SIDE_ONLY:-1}"
NO_TEXT_MAX_TOKEN_MARGIN="${NO_TEXT_MAX_TOKEN_MARGIN:-0}"
NO_TEXT_MIN_AUDIO_TOKENS="${NO_TEXT_MIN_AUDIO_TOKENS:-}"

# Optional no_text A/B knobs. Empty means use the model/config defaults.
NO_TEXT_SOURCE_GATE_FLOOR="${NO_TEXT_SOURCE_GATE_FLOOR:-}"
NO_TEXT_AUDIO_TEMPERATURE="${NO_TEXT_AUDIO_TEMPERATURE:-1.20}"
NO_TEXT_AUDIO_TOP_P="${NO_TEXT_AUDIO_TOP_P:-0.70}"
NO_TEXT_AUDIO_TOP_K="${NO_TEXT_AUDIO_TOP_K:-20}"
NO_TEXT_AUDIO_REPETITION_PENALTY="${NO_TEXT_AUDIO_REPETITION_PENALTY:-1.10}"
SOURCE_CONTENT_TOKEN_IDS="${SOURCE_CONTENT_TOKEN_IDS:-}"
SOURCE_CONTENT_TOKEN_IDS_PATH="${SOURCE_CONTENT_TOKEN_IDS_PATH:-}"
# Used by Ver2.5 text-token memory. Keep this aligned with SOURCE_AUDIO.
SOURCE_CONTENT_TEXT="${SOURCE_CONTENT_TEXT:-你在胡扯什么？我待在公司里真的只是加班工作，为什么你不能相信我呢？}"
SOURCE_CONTENT_SPM_MODEL="${SOURCE_CONTENT_SPM_MODEL:-${PROJECT_ROOT}/trainset/shared_content_tokenizers/spm_multilingual_byte_fallback_v1.model}"

resolve_latest_model_path() {
  if [ -n "${MODEL_PATH}" ]; then
    return 0
  fi
  if [ "${PREFERRED_STEP}" != "latest" ]; then
    preferred_dir="${DEFAULT_VER2_RUN_DIR}/${PREFERRED_STEP}"
    if [ -d "${preferred_dir}" ] && [ -f "${preferred_dir}/adapter_model.safetensors" ] && [ -f "${preferred_dir}/timbre_memory_adapter.pt" ]; then
      MODEL_PATH="${preferred_dir}"
      return 0
    fi
    echo "Preferred checkpoint not found or incomplete: ${preferred_dir}" >&2
    echo "Set PREFERRED_STEP=latest to use the latest available checkpoint, or set MODEL_PATH explicitly." >&2
    exit 1
  fi
  best_step=-1
  best_dir=""
  for step_dir in "${DEFAULT_VER2_RUN_DIR}"/step-*; do
    [ -d "${step_dir}" ] || continue
    [ -f "${step_dir}/adapter_model.safetensors" ] || continue
    [ -f "${step_dir}/timbre_memory_adapter.pt" ] || continue
    step_name=${step_dir##*/step-}
    case "${step_name}" in
      ""|*[!0-9]*) continue ;;
    esac
    if [ "${step_name}" -gt "${best_step}" ]; then
      best_step=${step_name}
      best_dir=${step_dir}
    fi
  done
  if [ -z "${best_dir}" ]; then
    echo "No valid step-* checkpoint found under ${DEFAULT_VER2_RUN_DIR}" >&2
    exit 1
  fi
  MODEL_PATH="${best_dir}"
}

resolve_latest_model_path
run_name=$(basename "$(dirname "${MODEL_PATH}")")
step_name=$(basename "${MODEL_PATH}")
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/testset/outputs/${run_name}_${step_name}_no_text}"

export DEFAULT_VER2_RUN_DIR
export MODEL_PATH
export PREFERRED_STEP
export SOURCE_AUDIO
export TIMBRE_REF_AUDIO
export OUTPUT_DIR
export DEVICE
export DEBUG_GENERATION_STRUCTURE
export TIMBRE_SIDE_ONLY
export NO_TEXT_MAX_TOKEN_MARGIN
export NO_TEXT_MIN_AUDIO_TOKENS
export NO_TEXT_SOURCE_GATE_FLOOR
export NO_TEXT_AUDIO_TEMPERATURE
export NO_TEXT_AUDIO_TOP_P
export NO_TEXT_AUDIO_TOP_K
export NO_TEXT_AUDIO_REPETITION_PENALTY
export SOURCE_CONTENT_TOKEN_IDS
export SOURCE_CONTENT_TOKEN_IDS_PATH
export SOURCE_CONTENT_TEXT
export SOURCE_CONTENT_SPM_MODEL
export SAVE_CODEC_INTERMEDIATES="${SAVE_CODEC_INTERMEDIATES:-0}"
export OUTPUT_GENERATED_CODEC="${OUTPUT_GENERATED_CODEC:-}"
export OUTPUT_CODEC_JSONL="${OUTPUT_CODEC_JSONL:-}"
export MODE="no_text"
export TEXT=""

exec sh "${RUN_SCRIPT}"

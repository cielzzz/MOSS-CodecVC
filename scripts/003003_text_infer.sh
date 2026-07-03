#!/usr/bin/env sh
set -eu

# Edit this block for daily text/text_prosody inference.
PROJECT_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/003003_run_moss_codecvc_infer.sh"

# Leave MODEL_PATH empty to use PREFERRED_STEP under DEFAULT_VER2_RUN_DIR.
# Set PREFERRED_STEP=latest to use the latest step-* checkpoint.
# Override MODEL_PATH/SOURCE_AUDIO/TIMBRE_REF_AUDIO/TEXT/OUTPUT_DIR/OUTPUT_WAV at invocation time for one-off runs.
DEFAULT_VER2_RUN_DIR="${DEFAULT_VER2_RUN_DIR:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/outputs/lora_runs/ver2_8_sideonly_wavlmbnf_codecres_textrep10_mergednotext_modeawareprosody_lora_r16_a32_gbs64}"
MODEL_PATH="${MODEL_PATH:-}"
PREFERRED_STEP="${PREFERRED_STEP:-latest}"

# In text mode, SOURCE_AUDIO is the prosody/style carrier; TEXT is the target lexical content.
# SOURCE_AUDIO="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/testset/source/media.wav"
SOURCE_AUDIO="${SOURCE_AUDIO:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/testset/source/zh_a_000003_source.flac}"  # 你在胡扯什么

TIMBRE_REF_AUDIO="${TIMBRE_REF_AUDIO:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/testset/timbre_ref/zh_a_000001_timbre.wav}"
TEXT="${TEXT:-你就一直没动过心，那倒也不是，我又不是圣人。所以说啊，你也是活生生的有血有肉的人啊。不是什么特殊材料制成的，很好理解也很简单。}"

DEVICE="${DEVICE:-auto}"
DEBUG_GENERATION_STRUCTURE="${DEBUG_GENERATION_STRUCTURE:-1}"
TIMBRE_SIDE_ONLY="${TIMBRE_SIDE_ONLY:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
TEXT_AUTO_MAX_NEW_TOKENS="${TEXT_AUTO_MAX_NEW_TOKENS:-1}"
TEXT_CJK_CHARS_PER_SECOND="${TEXT_CJK_CHARS_PER_SECOND:-5.2}"
TEXT_LATIN_WORDS_PER_SECOND="${TEXT_LATIN_WORDS_PER_SECOND:-2.8}"
TEXT_DURATION_MARGIN="${TEXT_DURATION_MARGIN:-1.15}"
TEXT_EXTRA_NEW_TOKENS="${TEXT_EXTRA_NEW_TOKENS:-48}"
TEXT_MIN_NEW_TOKENS_FLOOR="${TEXT_MIN_NEW_TOKENS_FLOOR:-96}"
TEMPERATURE="${TEMPERATURE:-}"
TOP_P="${TOP_P:-}"
TOP_K="${TOP_K:-}"
AUDIO_TEMPERATURE="${AUDIO_TEMPERATURE:-1.20}"
AUDIO_TOP_P="${AUDIO_TOP_P:-0.70}"
AUDIO_TOP_K="${AUDIO_TOP_K:-20}"
AUDIO_REPETITION_PENALTY="${AUDIO_REPETITION_PENALTY:-1.10}"
AUDIO_SEGMENT_POLICY="${AUDIO_SEGMENT_POLICY:-first}"
SOURCE_CONTENT_TOKEN_IDS="${SOURCE_CONTENT_TOKEN_IDS:-}"
SOURCE_CONTENT_TOKEN_IDS_PATH="${SOURCE_CONTENT_TOKEN_IDS_PATH:-}"
SOURCE_CONTENT_TEXT="${SOURCE_CONTENT_TEXT:-}"
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
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/testset/outputs/${run_name}_${step_name}_text}"

export DEFAULT_VER2_RUN_DIR
export MODEL_PATH
export PREFERRED_STEP
export SOURCE_AUDIO
export TIMBRE_REF_AUDIO
export TEXT
export OUTPUT_DIR
export DEVICE
export DEBUG_GENERATION_STRUCTURE
export TIMBRE_SIDE_ONLY
export MAX_NEW_TOKENS
export TEXT_AUTO_MAX_NEW_TOKENS
export TEXT_CJK_CHARS_PER_SECOND
export TEXT_LATIN_WORDS_PER_SECOND
export TEXT_DURATION_MARGIN
export TEXT_EXTRA_NEW_TOKENS
export TEXT_MIN_NEW_TOKENS_FLOOR
export TEMPERATURE
export TOP_P
export TOP_K
export AUDIO_TEMPERATURE
export AUDIO_TOP_P
export AUDIO_TOP_K
export AUDIO_REPETITION_PENALTY
export AUDIO_SEGMENT_POLICY
export SOURCE_CONTENT_TOKEN_IDS
export SOURCE_CONTENT_TOKEN_IDS_PATH
export SOURCE_CONTENT_TEXT
export SOURCE_CONTENT_SPM_MODEL
export SAVE_CODEC_INTERMEDIATES="${SAVE_CODEC_INTERMEDIATES:-0}"
export OUTPUT_GENERATED_CODEC="${OUTPUT_GENERATED_CODEC:-}"
export OUTPUT_CODEC_JSONL="${OUTPUT_CODEC_JSONL:-}"
export MODE="text"

exec sh "${RUN_SCRIPT}"

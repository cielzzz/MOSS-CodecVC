#!/usr/bin/env sh
set -eu

# 推荐用法 1：text/text_prosody，TEXT 是目标内容，SOURCE_AUDIO 只作为 prosody/style reference
#   MODE=text \
#   SOURCE_AUDIO=/abs/path/source.wav \
#   TIMBRE_REF_AUDIO=/abs/path/ref.wav \
#   TEXT="你好，这是一条测试文本。" \
#   sh /abs/path/003003_run_moss_codecvc_infer.sh
#
# 推荐用法 2：no-text VC
#   MODE=no_text \
#   SOURCE_AUDIO=/abs/path/source.wav \
#   TIMBRE_REF_AUDIO=/abs/path/ref.wav \
#   sh /abs/path/003003_run_moss_codecvc_infer.sh
#
# 兼容用法 3：如果只想传 CASE_ID，也可以从 metadata.tsv 里自动补齐路径和默认文本
#   CASE_ID=zh_a_000001 sh /abs/path/003003_run_moss_codecvc_infer.sh

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd "${SCRIPT_DIR}/.." && pwd)

# 默认都写成完整路径，便于直接复制和覆盖
PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
CONFIG="${CONFIG:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/configs/remote_full.yaml}"
DEFAULT_VER2_RUN_DIR="${DEFAULT_VER2_RUN_DIR:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/outputs/lora_runs/ver2_1_68w_textrep3_lora_r16_a32_gbs64_syncfix}"
LEGACY_MODEL_PATH="${LEGACY_MODEL_PATH:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/outputs/lora_runs/formal_lora_zh1000_en1000_mixed_mode_lr5e5/final}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction/MOSS-TTS}"
METADATA="${METADATA:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/testset/metadata.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/testset/outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
AUDIO_SEGMENT_POLICY="${AUDIO_SEGMENT_POLICY:-all}"
SAVE_CODEC_INTERMEDIATES="${SAVE_CODEC_INTERMEDIATES:-0}"
OUTPUT_GENERATED_CODEC="${OUTPUT_GENERATED_CODEC:-}"
OUTPUT_CODEC_JSONL="${OUTPUT_CODEC_JSONL:-}"

# DEVICE 支持两种模式：
# 1. 显式指定，例如 DEVICE=cuda:1
# 2. 自动选择空闲卡，例如 DEVICE=auto
DEVICE="${DEVICE:-auto}"
MODE="${MODE:-no_text}"
SEED="${SEED:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
TEXT_AUTO_MAX_NEW_TOKENS="${TEXT_AUTO_MAX_NEW_TOKENS:-1}"
TEXT_CJK_CHARS_PER_SECOND="${TEXT_CJK_CHARS_PER_SECOND:-5.2}"
TEXT_LATIN_WORDS_PER_SECOND="${TEXT_LATIN_WORDS_PER_SECOND:-2.8}"
TEXT_DURATION_MARGIN="${TEXT_DURATION_MARGIN:-1.15}"
TEXT_EXTRA_NEW_TOKENS="${TEXT_EXTRA_NEW_TOKENS:-48}"
TEXT_MIN_NEW_TOKENS_FLOOR="${TEXT_MIN_NEW_TOKENS_FLOOR:-96}"
# Keep no_text codec generation tightly aligned to the source.
NO_TEXT_MAX_TOKEN_MARGIN="${NO_TEXT_MAX_TOKEN_MARGIN:-0}"
NO_TEXT_DURATION_BUDGET_RATIO="${NO_TEXT_DURATION_BUDGET_RATIO:-1.0}"
NO_TEXT_SOFT_DURATION_BUDGET="${NO_TEXT_SOFT_DURATION_BUDGET:-0}"
NO_TEXT_SOFT_MIN_AUDIO_RATIO="${NO_TEXT_SOFT_MIN_AUDIO_RATIO:-0.5}"
NO_TEXT_SOFT_EXTRA_TOKEN_MARGIN="${NO_TEXT_SOFT_EXTRA_TOKEN_MARGIN:-}"
DISABLE_TIMBRE_MEMORY="${DISABLE_TIMBRE_MEMORY:-0}"
TIMBRE_SIDE_ONLY="${TIMBRE_SIDE_ONLY:-auto}"
SPEAKER_ENCODER_TYPE="${SPEAKER_ENCODER_TYPE:-speechbrain_ecapa}"
SPEAKER_ENCODER_PATH="${SPEAKER_ENCODER_PATH:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/models/speechbrain/spkrec-ecapa-voxceleb}"
SPEAKER_EMBEDDING_DIM="${SPEAKER_EMBEDDING_DIM:-192}"
TIMBRE_REF_SPEAKER_EMBEDDING_PATH="${TIMBRE_REF_SPEAKER_EMBEDDING_PATH:-}"
TEMPERATURE="${TEMPERATURE:-}"
TOP_P="${TOP_P:-}"
TOP_K="${TOP_K:-}"
AUDIO_TEMPERATURE="${AUDIO_TEMPERATURE:-}"
AUDIO_TOP_P="${AUDIO_TOP_P:-}"
AUDIO_TOP_K="${AUDIO_TOP_K:-}"
AUDIO_REPETITION_PENALTY="${AUDIO_REPETITION_PENALTY:-}"
# Optional no_text A/B knobs. Empty means use the config/model defaults.
NO_TEXT_AUDIO_TEMPERATURE="${NO_TEXT_AUDIO_TEMPERATURE:-}"
NO_TEXT_AUDIO_TOP_P="${NO_TEXT_AUDIO_TOP_P:-}"
NO_TEXT_AUDIO_TOP_K="${NO_TEXT_AUDIO_TOP_K:-}"
NO_TEXT_AUDIO_REPETITION_PENALTY="${NO_TEXT_AUDIO_REPETITION_PENALTY:-}"
NO_TEXT_SOURCE_GATE_FLOOR="${NO_TEXT_SOURCE_GATE_FLOOR:-}"
NO_TEXT_MIN_AUDIO_TOKENS="${NO_TEXT_MIN_AUDIO_TOKENS:-}"
DEBUG_GENERATION_STRUCTURE="${DEBUG_GENERATION_STRUCTURE:-0}"
DISABLE_SOURCE_SEMANTIC_MEMORY="${DISABLE_SOURCE_SEMANTIC_MEMORY:-0}"
SOURCE_SEMANTIC_FEATURE_PATH="${SOURCE_SEMANTIC_FEATURE_PATH:-}"
SOURCE_SEMANTIC_MODEL_NAME_OR_PATH="${SOURCE_SEMANTIC_MODEL_NAME_OR_PATH:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/huggingface/models--facebook--hubert-base-ls960/snapshots/dba3bb02fda4248b6e082697eee756de8fe8aa8a}"
SOURCE_SEMANTIC_CACHE_DIR="${SOURCE_SEMANTIC_CACHE_DIR:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/huggingface}"
SOURCE_SEMANTIC_LOCAL_FILES_ONLY="${SOURCE_SEMANTIC_LOCAL_FILES_ONLY:-1}"
SOURCE_SEMANTIC_LAYER="${SOURCE_SEMANTIC_LAYER:-9}"
SOURCE_SEMANTIC_DEVICE="${SOURCE_SEMANTIC_DEVICE:-same}"
SOURCE_SEMANTIC_DTYPE="${SOURCE_SEMANTIC_DTYPE:-auto}"
SOURCE_SEMANTIC_DOWNSAMPLE_STRIDE="${SOURCE_SEMANTIC_DOWNSAMPLE_STRIDE:-1}"
SOURCE_SEMANTIC_ATTENTION_DEBUG_DIR="${SOURCE_SEMANTIC_ATTENTION_DEBUG_DIR:-}"
SOURCE_SEMANTIC_ATTENTION_DEBUG_MAX_TOKENS="${SOURCE_SEMANTIC_ATTENTION_DEBUG_MAX_TOKENS:-2048}"
SOURCE_SEMANTIC_POSITION_SCALE="${SOURCE_SEMANTIC_POSITION_SCALE:-}"
SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH="${SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH:-}"
SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH="${SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH:-}"
DISABLE_SOURCE_SEMANTIC_MONOTONIC_BIAS="${DISABLE_SOURCE_SEMANTIC_MONOTONIC_BIAS:-0}"
SOURCE_SEMANTIC_PROGRESS_CLOCK="${SOURCE_SEMANTIC_PROGRESS_CLOCK:-decode_step}"
SOURCE_SEMANTIC_RELEASE_AFTER_PROGRESS="${SOURCE_SEMANTIC_RELEASE_AFTER_PROGRESS:-0}"
SOURCE_SEMANTIC_RELEASE_START="${SOURCE_SEMANTIC_RELEASE_START:-1.0}"
SOURCE_CONTENT_TOKEN_IDS="${SOURCE_CONTENT_TOKEN_IDS:-}"
SOURCE_CONTENT_TOKEN_IDS_PATH="${SOURCE_CONTENT_TOKEN_IDS_PATH:-}"
SOURCE_CONTENT_TEXT="${SOURCE_CONTENT_TEXT:-}"
SOURCE_CONTENT_SPM_MODEL="${SOURCE_CONTENT_SPM_MODEL:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/trainset/shared_content_tokenizers/spm_multilingual_byte_fallback_v1.model}"

# 默认测试对
# SOURCE_AUDIO="${SOURCE_AUDIO:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/testset/source/zh_a_000003_source.flac}"
SOURCE_AUDIO="${SOURCE_AUDIO:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/testset/source/media.wav}"

# # 低沉男
# TIMBRE_REF_AUDIO="${TIMBRE_REF_AUDIO:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/testset/timbre_ref/zh_a_000000_timbre.wav}"
# 女
TIMBRE_REF_AUDIO="${TIMBRE_REF_AUDIO:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/testset/timbre_ref/zh_a_000001_timbre.wav}"

# 如果明确给了 CASE_ID，脚本会优先尝试从 metadata 读取默认文本；
# 但 SOURCE_AUDIO / TIMBRE_REF_AUDIO 如果手动传了，就不会被覆盖。
CASE_ID="${CASE_ID:-}"
TEXT="${TEXT:-}"
OUTPUT_WAV="${OUTPUT_WAV:-}"

resolve_default_model_path() {
  if [ -n "${MODEL_PATH:-}" ]; then
    return 0
  fi

  best_step=-1
  best_dir=""
  if [ -d "${DEFAULT_VER2_RUN_DIR}" ]; then
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
  fi

  if [ -n "${best_dir}" ]; then
    MODEL_PATH="${best_dir}"
  else
    MODEL_PATH="${LEGACY_MODEL_PATH}"
  fi
}

resolve_output_dir() {
  if [ -n "${OUTPUT_DIR}" ]; then
    return 0
  fi

  model_path_dir=$(dirname "${MODEL_PATH}")
  model_path_base=$(basename "${MODEL_PATH}")
  model_path_label=${model_path_base%.*}
  run_name=$(basename "${model_path_dir}")

  if [ -z "${run_name}" ] || [ "${run_name}" = "." ]; then
    OUTPUT_DIR="${OUTPUT_ROOT}/${model_path_label}"
    return 0
  fi

  case "${model_path_label}" in
    step-*)
      OUTPUT_DIR="${OUTPUT_ROOT}/${run_name}_${model_path_label}"
      ;;
    final)
      OUTPUT_DIR="${OUTPUT_ROOT}/${run_name}_final"
      ;;
    *)
      OUTPUT_DIR="${OUTPUT_ROOT}/${run_name}_${model_path_label}"
      ;;
  esac
}

resolve_from_metadata() {
  if [ -z "${CASE_ID}" ] || [ ! -f "${METADATA}" ]; then
    return 0
  fi

  row=$(awk -F '\t' -v case_id="${CASE_ID}" 'NR > 1 && $1 == case_id {print; exit}' "${METADATA}")
  if [ -z "${row}" ]; then
    echo "未在 metadata 中找到 CASE_ID=${CASE_ID}: ${METADATA}" >&2
    exit 1
  fi

  case_id=$(printf '%s\n' "${row}" | cut -f1)
  meta_source=$(printf '%s\n' "${row}" | cut -f2)
  meta_timbre=$(printf '%s\n' "${row}" | cut -f3)
  meta_default_text=$(printf '%s\n' "${row}" | cut -f4)
  meta_source_text=$(printf '%s\n' "${row}" | cut -f5)

  # 只有在用户没有手动覆盖时，才从 metadata 填充
  if [ -z "${SOURCE_AUDIO_SET_BY_USER:-}" ]; then
    SOURCE_AUDIO="${PROJECT_ROOT}/${meta_source}"
  fi
  if [ -z "${TIMBRE_REF_AUDIO_SET_BY_USER:-}" ]; then
    TIMBRE_REF_AUDIO="${PROJECT_ROOT}/${meta_timbre}"
  fi
  if [ -z "${TEXT}" ]; then
    if [ "${MODE}" = "text" ] && [ -n "${meta_source_text}" ]; then
      TEXT="${meta_source_text}"
    else
      TEXT="${meta_default_text}"
    fi
  fi
}

resolve_device() {
  if [ "${DEVICE}" = "cuda" ]; then
    DEVICE="cuda:0"
    return 0
  fi

  if [ "${DEVICE}" != "auto" ]; then
    return 0
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    DEVICE="cuda:0"
    return 0
  fi

  best_idx=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | \
    awk -F',' '{
      gsub(/ /, "", $1);
      gsub(/ /, "", $2);
      if (NR == 1 || $2 + 0 < min_mem) {
        min_mem = $2 + 0;
        best_idx = $1;
      }
    } END { print best_idx }')

  if [ -n "${best_idx}" ]; then
    DEVICE="cuda:${best_idx}"
    return 0
  fi

  DEVICE="cuda:0"
}

resolve_timbre_side_only() {
  case "${TIMBRE_SIDE_ONLY}" in
    auto|AUTO|"")
      resolved_timbre_side_only=$("${PYTHON}" - "${MODEL_PATH}" <<'PY' || printf '0'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1]) / "timbre_memory_config.json"
if not config_path.exists():
    print("0")
    raise SystemExit(0)
try:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
except Exception:
    print("0")
    raise SystemExit(0)
print("1" if config.get("timbre_side_only", False) else "0")
PY
)
      TIMBRE_SIDE_ONLY="${resolved_timbre_side_only}"
      ;;
  esac
}

# 标记用户是否显式传了路径，避免 metadata 覆盖
case "${SOURCE_AUDIO}" in
  "") ;;
  *) SOURCE_AUDIO_SET_BY_USER=1 ;;
esac
case "${TIMBRE_REF_AUDIO}" in
  "") ;;
  *) TIMBRE_REF_AUDIO_SET_BY_USER=1 ;;
esac

resolve_from_metadata
resolve_device
resolve_default_model_path
resolve_timbre_side_only
resolve_output_dir

if [ ! -f "${SOURCE_AUDIO}" ]; then
  echo "SOURCE_AUDIO 不存在: ${SOURCE_AUDIO}" >&2
  exit 1
fi
if [ ! -f "${TIMBRE_REF_AUDIO}" ]; then
  echo "TIMBRE_REF_AUDIO 不存在: ${TIMBRE_REF_AUDIO}" >&2
  exit 1
fi

source_stem=$(basename "${SOURCE_AUDIO}")
source_stem=${source_stem%.*}
timbre_stem=$(basename "${TIMBRE_REF_AUDIO}")
timbre_stem=${timbre_stem%.*}

if [ -z "${OUTPUT_WAV}" ]; then
  OUTPUT_WAV="${OUTPUT_DIR}/${source_stem}_${timbre_stem}.wav"
fi

if [ "${SAVE_CODEC_INTERMEDIATES}" = "1" ]; then
  if [ -z "${OUTPUT_GENERATED_CODEC}" ]; then
    OUTPUT_GENERATED_CODEC="${OUTPUT_DIR}/${source_stem}_${timbre_stem}.generated_codec.pt"
  fi
  if [ -z "${OUTPUT_CODEC_JSONL}" ]; then
    OUTPUT_CODEC_JSONL="${OUTPUT_DIR}/${source_stem}_${timbre_stem}.codec_visualization.jsonl"
  fi
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname "${OUTPUT_WAV}")"

echo "[infer] MODE=${MODE}"
echo "[infer] MODEL_PATH=${MODEL_PATH}"
echo "[infer] BASE_MODEL_PATH=${BASE_MODEL_PATH}"
echo "[infer] DEVICE=${DEVICE}"
echo "[infer] MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-default}"
if [ "${MODE}" = "text" ]; then
  echo "[infer] TEXT_AUTO_MAX_NEW_TOKENS=${TEXT_AUTO_MAX_NEW_TOKENS}"
  echo "[infer] TEXT_CJK_CHARS_PER_SECOND=${TEXT_CJK_CHARS_PER_SECOND}"
  echo "[infer] TEXT_LATIN_WORDS_PER_SECOND=${TEXT_LATIN_WORDS_PER_SECOND}"
  echo "[infer] TEXT_DURATION_MARGIN=${TEXT_DURATION_MARGIN}"
  echo "[infer] TEXT_EXTRA_NEW_TOKENS=${TEXT_EXTRA_NEW_TOKENS}"
  echo "[infer] TEXT_MIN_NEW_TOKENS_FLOOR=${TEXT_MIN_NEW_TOKENS_FLOOR}"
fi
echo "[infer] NO_TEXT_MAX_TOKEN_MARGIN=${NO_TEXT_MAX_TOKEN_MARGIN}"
echo "[infer] DISABLE_TIMBRE_MEMORY=${DISABLE_TIMBRE_MEMORY}"
echo "[infer] TIMBRE_SIDE_ONLY=${TIMBRE_SIDE_ONLY}"
echo "[infer] SPEAKER_ENCODER_TYPE=${SPEAKER_ENCODER_TYPE:-adapter_config}"
echo "[infer] SPEAKER_ENCODER_PATH=${SPEAKER_ENCODER_PATH:-adapter_config}"
echo "[infer] SPEAKER_EMBEDDING_DIM=${SPEAKER_EMBEDDING_DIM:-adapter_config}"
echo "[infer] TIMBRE_REF_SPEAKER_EMBEDDING_PATH=${TIMBRE_REF_SPEAKER_EMBEDDING_PATH:-none}"
if [ "${MODE}" = "no_text" ]; then
  echo "[infer] NO_TEXT_AUDIO_TEMPERATURE=${NO_TEXT_AUDIO_TEMPERATURE}"
  echo "[infer] NO_TEXT_AUDIO_TOP_P=${NO_TEXT_AUDIO_TOP_P}"
  echo "[infer] NO_TEXT_AUDIO_TOP_K=${NO_TEXT_AUDIO_TOP_K}"
  echo "[infer] NO_TEXT_AUDIO_REPETITION_PENALTY=${NO_TEXT_AUDIO_REPETITION_PENALTY}"
  echo "[infer] NO_TEXT_SOURCE_GATE_FLOOR=${NO_TEXT_SOURCE_GATE_FLOOR}"
  echo "[infer] NO_TEXT_MIN_AUDIO_TOKENS=${NO_TEXT_MIN_AUDIO_TOKENS:-source_codec_len}"
fi
echo "[infer] SOURCE_AUDIO=${SOURCE_AUDIO}"
echo "[infer] TIMBRE_REF_AUDIO=${TIMBRE_REF_AUDIO}"
echo "[infer] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[infer] OUTPUT_WAV=${OUTPUT_WAV}"
echo "[infer] SAVE_CODEC_INTERMEDIATES=${SAVE_CODEC_INTERMEDIATES}"
echo "[infer] OUTPUT_GENERATED_CODEC=${OUTPUT_GENERATED_CODEC:-none}"
echo "[infer] OUTPUT_CODEC_JSONL=${OUTPUT_CODEC_JSONL:-none}"
echo "[infer] DEBUG_GENERATION_STRUCTURE=${DEBUG_GENERATION_STRUCTURE}"
echo "[infer] AUDIO_SEGMENT_POLICY=${AUDIO_SEGMENT_POLICY}"
echo "[infer] DISABLE_SOURCE_SEMANTIC_MEMORY=${DISABLE_SOURCE_SEMANTIC_MEMORY}"
echo "[infer] SOURCE_SEMANTIC_FEATURE_PATH=${SOURCE_SEMANTIC_FEATURE_PATH:-online}"
echo "[infer] SOURCE_SEMANTIC_MODEL_NAME_OR_PATH=${SOURCE_SEMANTIC_MODEL_NAME_OR_PATH}"
echo "[infer] SOURCE_SEMANTIC_LAYER=${SOURCE_SEMANTIC_LAYER}"
echo "[infer] SOURCE_SEMANTIC_DEVICE=${SOURCE_SEMANTIC_DEVICE}"
echo "[infer] SOURCE_SEMANTIC_ATTENTION_DEBUG_DIR=${SOURCE_SEMANTIC_ATTENTION_DEBUG_DIR:-none}"
echo "[infer] SOURCE_SEMANTIC_POSITION_SCALE=${SOURCE_SEMANTIC_POSITION_SCALE:-model_default}"
echo "[infer] SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH=${SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH:-model_default}"
echo "[infer] SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH=${SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH:-model_default}"
echo "[infer] SOURCE_CONTENT_TOKEN_IDS=${SOURCE_CONTENT_TOKEN_IDS:+provided}"
echo "[infer] SOURCE_CONTENT_TOKEN_IDS_PATH=${SOURCE_CONTENT_TOKEN_IDS_PATH:-none}"
echo "[infer] SOURCE_CONTENT_TEXT=${SOURCE_CONTENT_TEXT:+provided}"
echo "[infer] SOURCE_CONTENT_SPM_MODEL=${SOURCE_CONTENT_SPM_MODEL}"

if [ "${MODE}" = "text" ]; then
  if [ -z "${TEXT}" ]; then
    echo "MODE=text 时必须提供 TEXT，或者传 CASE_ID 让脚本从 metadata 里取默认文本。" >&2
    exit 1
  fi
  echo "[infer] TEXT=${TEXT}"
  set -- \
    --config "${CONFIG}" \
    --model-path "${MODEL_PATH}" \
    --base-model-path "${BASE_MODEL_PATH}" \
    --source-audio "${SOURCE_AUDIO}" \
    --timbre-ref-audio "${TIMBRE_REF_AUDIO}" \
    --text "${TEXT}" \
    --output-wav "${OUTPUT_WAV}" \
    --audio-segment-policy "${AUDIO_SEGMENT_POLICY}" \
    --device "${DEVICE}"
  if [ -n "${SEED}" ]; then
    set -- "$@" --seed "${SEED}"
  fi
  if [ -n "${MAX_NEW_TOKENS}" ]; then
    set -- "$@" --max-new-tokens "${MAX_NEW_TOKENS}"
  fi
  if [ "${TEXT_AUTO_MAX_NEW_TOKENS}" = "0" ]; then
    set -- "$@" --no-text-auto-max-new-tokens
  fi
  set -- "$@" \
    --text-cjk-chars-per-second "${TEXT_CJK_CHARS_PER_SECOND}" \
    --text-latin-words-per-second "${TEXT_LATIN_WORDS_PER_SECOND}" \
    --text-duration-margin "${TEXT_DURATION_MARGIN}" \
    --text-extra-new-tokens "${TEXT_EXTRA_NEW_TOKENS}" \
    --text-min-new-tokens-floor "${TEXT_MIN_NEW_TOKENS_FLOOR}"
  if [ "${DISABLE_TIMBRE_MEMORY}" = "1" ]; then
    set -- "$@" --disable-timbre-memory
  fi
  if [ "${TIMBRE_SIDE_ONLY}" = "1" ]; then
    set -- "$@" --timbre-side-only
  fi
  if [ -n "${SPEAKER_ENCODER_TYPE}" ]; then
    set -- "$@" --speaker-encoder-type "${SPEAKER_ENCODER_TYPE}"
  fi
  if [ -n "${SPEAKER_ENCODER_PATH}" ]; then
    set -- "$@" --speaker-encoder-path "${SPEAKER_ENCODER_PATH}"
  fi
  if [ -n "${SPEAKER_EMBEDDING_DIM}" ]; then
    set -- "$@" --speaker-embedding-dim "${SPEAKER_EMBEDDING_DIM}"
  fi
  if [ -n "${TIMBRE_REF_SPEAKER_EMBEDDING_PATH}" ]; then
    set -- "$@" --timbre-ref-speaker-embedding-path "${TIMBRE_REF_SPEAKER_EMBEDDING_PATH}"
  fi
  if [ "${DISABLE_SOURCE_SEMANTIC_MEMORY}" = "1" ]; then
    set -- "$@" --disable-source-semantic-memory
  fi
  if [ -n "${SOURCE_SEMANTIC_FEATURE_PATH}" ]; then
    set -- "$@" --source-semantic-feature-path "${SOURCE_SEMANTIC_FEATURE_PATH}"
  fi
  set -- "$@" \
    --source-semantic-model-name-or-path "${SOURCE_SEMANTIC_MODEL_NAME_OR_PATH}" \
    --source-semantic-cache-dir "${SOURCE_SEMANTIC_CACHE_DIR}" \
    --source-semantic-layer "${SOURCE_SEMANTIC_LAYER}" \
    --source-semantic-device "${SOURCE_SEMANTIC_DEVICE}" \
    --source-semantic-dtype "${SOURCE_SEMANTIC_DTYPE}" \
    --source-semantic-downsample-stride "${SOURCE_SEMANTIC_DOWNSAMPLE_STRIDE}"
  if [ -n "${SOURCE_SEMANTIC_POSITION_SCALE}" ]; then
    set -- "$@" --source-semantic-position-scale "${SOURCE_SEMANTIC_POSITION_SCALE}"
  fi
  if [ -n "${SOURCE_SEMANTIC_ATTENTION_DEBUG_DIR}" ]; then
    set -- "$@" \
      --source-semantic-attention-debug-dir "${SOURCE_SEMANTIC_ATTENTION_DEBUG_DIR}" \
      --source-semantic-attention-debug-max-tokens "${SOURCE_SEMANTIC_ATTENTION_DEBUG_MAX_TOKENS}"
  fi
  if [ -n "${SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH}" ]; then
    set -- "$@" --source-semantic-monotonic-bias-strength "${SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH}"
  fi
  if [ -n "${SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH}" ]; then
    set -- "$@" --source-semantic-monotonic-bias-width "${SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH}"
  fi
  if [ "${DISABLE_SOURCE_SEMANTIC_MONOTONIC_BIAS}" = "1" ]; then
    set -- "$@" --disable-source-semantic-monotonic-bias
  fi
  set -- "$@" --source-semantic-progress-clock "${SOURCE_SEMANTIC_PROGRESS_CLOCK}"
  if [ "${SOURCE_SEMANTIC_RELEASE_AFTER_PROGRESS}" = "1" ]; then
    set -- "$@" --source-semantic-release-after-progress
  fi
  set -- "$@" --source-semantic-release-start "${SOURCE_SEMANTIC_RELEASE_START}"
  if [ "${SOURCE_SEMANTIC_LOCAL_FILES_ONLY}" = "0" ]; then
    set -- "$@" --no-source-semantic-local-files-only
  fi
  if [ -n "${SOURCE_CONTENT_TOKEN_IDS}" ]; then
    set -- "$@" --source-content-token-ids "${SOURCE_CONTENT_TOKEN_IDS}"
  fi
  if [ -n "${SOURCE_CONTENT_TOKEN_IDS_PATH}" ]; then
    set -- "$@" --source-content-token-ids-path "${SOURCE_CONTENT_TOKEN_IDS_PATH}"
  fi
  if [ -n "${SOURCE_CONTENT_TEXT}" ]; then
    set -- "$@" --source-content-text "${SOURCE_CONTENT_TEXT}"
  fi
  if [ -n "${SOURCE_CONTENT_SPM_MODEL}" ]; then
    set -- "$@" --source-content-spm-model "${SOURCE_CONTENT_SPM_MODEL}"
  fi
  if [ -n "${TEMPERATURE}" ]; then
    set -- "$@" --temperature "${TEMPERATURE}"
  fi
  if [ -n "${TOP_P}" ]; then
    set -- "$@" --top-p "${TOP_P}"
  fi
  if [ -n "${TOP_K}" ]; then
    set -- "$@" --top-k "${TOP_K}"
  fi
  if [ -n "${AUDIO_TEMPERATURE}" ]; then
    set -- "$@" --audio-temperature "${AUDIO_TEMPERATURE}"
  fi
  if [ -n "${AUDIO_TOP_P}" ]; then
    set -- "$@" --audio-top-p "${AUDIO_TOP_P}"
  fi
  if [ -n "${AUDIO_TOP_K}" ]; then
    set -- "$@" --audio-top-k "${AUDIO_TOP_K}"
  fi
  if [ -n "${AUDIO_REPETITION_PENALTY}" ]; then
    set -- "$@" --audio-repetition-penalty "${AUDIO_REPETITION_PENALTY}"
  fi
  if [ -n "${OUTPUT_GENERATED_CODEC}" ]; then
    set -- "$@" --output-generated-codec "${OUTPUT_GENERATED_CODEC}"
  fi
  if [ -n "${OUTPUT_CODEC_JSONL}" ]; then
    set -- "$@" --output-codec-jsonl "${OUTPUT_CODEC_JSONL}"
  fi
  if [ "${DEBUG_GENERATION_STRUCTURE}" = "1" ]; then
    set -- "$@" --debug-generation-structure
  fi
  exec "${PYTHON}" "${PROJECT_ROOT}/scripts/003001_infer_moss_codecvc.py" "$@"
fi

if [ "${MODE}" = "no_text" ]; then
  set -- \
    --config "${CONFIG}" \
    --model-path "${MODEL_PATH}" \
    --base-model-path "${BASE_MODEL_PATH}" \
    --source-audio "${SOURCE_AUDIO}" \
    --timbre-ref-audio "${TIMBRE_REF_AUDIO}" \
    --no-text \
    --output-wav "${OUTPUT_WAV}" \
    --audio-segment-policy "${AUDIO_SEGMENT_POLICY}" \
    --device "${DEVICE}"
  if [ -n "${SEED}" ]; then
    set -- "$@" --seed "${SEED}"
  fi
  if [ -n "${MAX_NEW_TOKENS}" ]; then
    set -- "$@" --max-new-tokens "${MAX_NEW_TOKENS}"
  fi
  set -- "$@" \
    --no-text-max-token-margin "${NO_TEXT_MAX_TOKEN_MARGIN}" \
    --no-text-duration-budget-ratio "${NO_TEXT_DURATION_BUDGET_RATIO}"
  if [ "${NO_TEXT_SOFT_DURATION_BUDGET}" = "1" ]; then
    set -- "$@" \
      --no-text-soft-duration-budget \
      --no-text-soft-min-audio-ratio "${NO_TEXT_SOFT_MIN_AUDIO_RATIO}"
    if [ -n "${NO_TEXT_SOFT_EXTRA_TOKEN_MARGIN}" ]; then
      set -- "$@" --no-text-soft-extra-token-margin "${NO_TEXT_SOFT_EXTRA_TOKEN_MARGIN}"
    fi
  fi
  if [ "${DISABLE_TIMBRE_MEMORY}" = "1" ]; then
    set -- "$@" --disable-timbre-memory
  fi
  if [ "${TIMBRE_SIDE_ONLY}" = "1" ]; then
    set -- "$@" --timbre-side-only
  fi
  if [ -n "${SPEAKER_ENCODER_TYPE}" ]; then
    set -- "$@" --speaker-encoder-type "${SPEAKER_ENCODER_TYPE}"
  fi
  if [ -n "${SPEAKER_ENCODER_PATH}" ]; then
    set -- "$@" --speaker-encoder-path "${SPEAKER_ENCODER_PATH}"
  fi
  if [ -n "${SPEAKER_EMBEDDING_DIM}" ]; then
    set -- "$@" --speaker-embedding-dim "${SPEAKER_EMBEDDING_DIM}"
  fi
  if [ -n "${TIMBRE_REF_SPEAKER_EMBEDDING_PATH}" ]; then
    set -- "$@" --timbre-ref-speaker-embedding-path "${TIMBRE_REF_SPEAKER_EMBEDDING_PATH}"
  fi
  if [ "${DISABLE_SOURCE_SEMANTIC_MEMORY}" = "1" ]; then
    set -- "$@" --disable-source-semantic-memory
  fi
  if [ -n "${SOURCE_SEMANTIC_FEATURE_PATH}" ]; then
    set -- "$@" --source-semantic-feature-path "${SOURCE_SEMANTIC_FEATURE_PATH}"
  fi
  set -- "$@" \
    --source-semantic-model-name-or-path "${SOURCE_SEMANTIC_MODEL_NAME_OR_PATH}" \
    --source-semantic-cache-dir "${SOURCE_SEMANTIC_CACHE_DIR}" \
    --source-semantic-layer "${SOURCE_SEMANTIC_LAYER}" \
    --source-semantic-device "${SOURCE_SEMANTIC_DEVICE}" \
    --source-semantic-dtype "${SOURCE_SEMANTIC_DTYPE}" \
    --source-semantic-downsample-stride "${SOURCE_SEMANTIC_DOWNSAMPLE_STRIDE}"
  if [ -n "${SOURCE_SEMANTIC_POSITION_SCALE}" ]; then
    set -- "$@" --source-semantic-position-scale "${SOURCE_SEMANTIC_POSITION_SCALE}"
  fi
  if [ -n "${SOURCE_SEMANTIC_ATTENTION_DEBUG_DIR}" ]; then
    set -- "$@" \
      --source-semantic-attention-debug-dir "${SOURCE_SEMANTIC_ATTENTION_DEBUG_DIR}" \
      --source-semantic-attention-debug-max-tokens "${SOURCE_SEMANTIC_ATTENTION_DEBUG_MAX_TOKENS}"
  fi
  if [ -n "${SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH}" ]; then
    set -- "$@" --source-semantic-monotonic-bias-strength "${SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH}"
  fi
  if [ -n "${SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH}" ]; then
    set -- "$@" --source-semantic-monotonic-bias-width "${SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH}"
  fi
  if [ "${DISABLE_SOURCE_SEMANTIC_MONOTONIC_BIAS}" = "1" ]; then
    set -- "$@" --disable-source-semantic-monotonic-bias
  fi
  set -- "$@" --source-semantic-progress-clock "${SOURCE_SEMANTIC_PROGRESS_CLOCK}"
  if [ "${SOURCE_SEMANTIC_RELEASE_AFTER_PROGRESS}" = "1" ]; then
    set -- "$@" --source-semantic-release-after-progress
  fi
  set -- "$@" --source-semantic-release-start "${SOURCE_SEMANTIC_RELEASE_START}"
  if [ "${SOURCE_SEMANTIC_LOCAL_FILES_ONLY}" = "0" ]; then
    set -- "$@" --no-source-semantic-local-files-only
  fi
  if [ -n "${SOURCE_CONTENT_TOKEN_IDS}" ]; then
    set -- "$@" --source-content-token-ids "${SOURCE_CONTENT_TOKEN_IDS}"
  fi
  if [ -n "${SOURCE_CONTENT_TOKEN_IDS_PATH}" ]; then
    set -- "$@" --source-content-token-ids-path "${SOURCE_CONTENT_TOKEN_IDS_PATH}"
  fi
  if [ -n "${SOURCE_CONTENT_TEXT}" ]; then
    set -- "$@" --source-content-text "${SOURCE_CONTENT_TEXT}"
  fi
  if [ -n "${SOURCE_CONTENT_SPM_MODEL}" ]; then
    set -- "$@" --source-content-spm-model "${SOURCE_CONTENT_SPM_MODEL}"
  fi
  if [ -n "${TEMPERATURE}" ]; then
    set -- "$@" --temperature "${TEMPERATURE}"
  fi
  if [ -n "${TOP_P}" ]; then
    set -- "$@" --top-p "${TOP_P}"
  fi
  if [ -n "${TOP_K}" ]; then
    set -- "$@" --top-k "${TOP_K}"
  fi
  no_text_audio_temperature="${NO_TEXT_AUDIO_TEMPERATURE:-${AUDIO_TEMPERATURE:-}}"
  no_text_audio_top_p="${NO_TEXT_AUDIO_TOP_P:-${AUDIO_TOP_P:-}}"
  no_text_audio_top_k="${NO_TEXT_AUDIO_TOP_K:-${AUDIO_TOP_K:-}}"
  no_text_audio_repetition_penalty="${NO_TEXT_AUDIO_REPETITION_PENALTY:-${AUDIO_REPETITION_PENALTY:-}}"
  if [ -n "${no_text_audio_temperature}" ]; then
    set -- "$@" --audio-temperature "${no_text_audio_temperature}"
  fi
  if [ -n "${no_text_audio_top_p}" ]; then
    set -- "$@" --audio-top-p "${no_text_audio_top_p}"
  fi
  if [ -n "${no_text_audio_top_k}" ]; then
    set -- "$@" --audio-top-k "${no_text_audio_top_k}"
  fi
  if [ -n "${no_text_audio_repetition_penalty}" ]; then
    set -- "$@" --audio-repetition-penalty "${no_text_audio_repetition_penalty}"
  fi
  if [ -n "${NO_TEXT_SOURCE_GATE_FLOOR}" ]; then
    set -- "$@" --source-gate-floor "${NO_TEXT_SOURCE_GATE_FLOOR}"
  fi
  if [ -n "${NO_TEXT_MIN_AUDIO_TOKENS}" ]; then
    case "${NO_TEXT_MIN_AUDIO_TOKENS}" in
      source_codec_len|auto)
        # Leave --min-audio-tokens unset so the Python infer script can derive
        # the guard from the encoded source codec length after tokenization.
        ;;
      *)
        set -- "$@" --min-audio-tokens "${NO_TEXT_MIN_AUDIO_TOKENS}"
        ;;
    esac
  fi
  if [ -n "${OUTPUT_GENERATED_CODEC}" ]; then
    set -- "$@" --output-generated-codec "${OUTPUT_GENERATED_CODEC}"
  fi
  if [ -n "${OUTPUT_CODEC_JSONL}" ]; then
    set -- "$@" --output-codec-jsonl "${OUTPUT_CODEC_JSONL}"
  fi
  if [ "${DEBUG_GENERATION_STRUCTURE}" = "1" ]; then
    set -- "$@" --debug-generation-structure
  fi
  exec "${PYTHON}" "${PROJECT_ROOT}/scripts/003001_infer_moss_codecvc.py" "$@"
fi

echo "不支持的 MODE=${MODE}，请使用 MODE=text 或 MODE=no_text" >&2
exit 1

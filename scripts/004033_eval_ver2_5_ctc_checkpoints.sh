#!/usr/bin/env sh
set -eu

# Evaluate the Ver2.5 CTC follow-up runs on the fixed no-text sample.
#
# Default:
#   sh scripts/004033_eval_ver2_5_ctc_checkpoints.sh
#
# Useful overrides:
#   CHECKPOINT_STEPS="step-1000 step-2000" sh scripts/004033_eval_ver2_5_ctc_checkpoints.sh
#   RUN_ASR=0 sh scripts/004033_eval_ver2_5_ctc_checkpoints.sh

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd "${SCRIPT_DIR}/.." && pwd)

EVAL_SCRIPT="${EVAL_SCRIPT:-${PROJECT_ROOT}/scripts/004011_eval_moss_codecvc_checkpoints.sh}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/outputs/lora_runs/ver2_5_debug_5k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/testset/outputs/ver2_5_ctc_eval}"

CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-step-1000}"
EVAL_VARIANTS="${EVAL_VARIANTS:-topk1}"
SOURCE_AUDIO="${SOURCE_AUDIO:-${PROJECT_ROOT}/testset/source/media.wav}"
TIMBRE_REF_AUDIO="${TIMBRE_REF_AUDIO:-${PROJECT_ROOT}/testset/timbre_ref/zh_a_000001_timbre.wav}"
DEVICE="${DEVICE:-cuda:0}"
ASR_DEVICE="${ASR_DEVICE:-cuda:0}"
SOURCE_SEMANTIC_DEVICE="${SOURCE_SEMANTIC_DEVICE:-cpu}"
RUN_ASR="${RUN_ASR:-1}"
OVERWRITE_INFER="${OVERWRITE_INFER:-1}"
NO_TEXT_MIN_AUDIO_TOKENS="${NO_TEXT_MIN_AUDIO_TOKENS:-source_codec_len}"

if [ ! -x "${EVAL_SCRIPT}" ] && [ ! -f "${EVAL_SCRIPT}" ]; then
  echo "ERROR: missing eval script: ${EVAL_SCRIPT}" >&2
  exit 1
fi

has_checkpoint() {
  run_dir="$1"
  for step_name in ${CHECKPOINT_STEPS}; do
    ckpt_dir="${run_dir}/${step_name}"
    if [ -f "${ckpt_dir}/adapter_model.safetensors" ] && [ -f "${ckpt_dir}/timbre_memory_adapter.pt" ]; then
      return 0
    fi
  done
  return 1
}

run_eval() {
  run_name="$1"
  disable_source_semantic="$2"
  run_dir="${RUN_ROOT}/${run_name}"

  if [ ! -d "${run_dir}" ]; then
    echo "[ver2.5-ctc-eval] skip missing run_dir=${run_dir}" >&2
    return 1
  fi
  if ! has_checkpoint "${run_dir}"; then
    echo "[ver2.5-ctc-eval] skip ${run_name}: no complete checkpoint in CHECKPOINT_STEPS=${CHECKPOINT_STEPS}" >&2
    return 1
  fi

  echo "[ver2.5-ctc-eval] run=${run_name} disable_source_semantic=${disable_source_semantic}"
  RUN_DIR="${run_dir}" \
  CHECKPOINT_STEPS="${CHECKPOINT_STEPS}" \
  EVAL_VARIANTS="${EVAL_VARIANTS}" \
  EVAL_NAME="${run_name}_${CHECKPOINT_STEPS}_fixed_no_text" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  SOURCE_AUDIO="${SOURCE_AUDIO}" \
  TIMBRE_REF_AUDIO="${TIMBRE_REF_AUDIO}" \
  DEVICE="${DEVICE}" \
  ASR_DEVICE="${ASR_DEVICE}" \
  SOURCE_SEMANTIC_DEVICE="${SOURCE_SEMANTIC_DEVICE}" \
  DISABLE_SOURCE_SEMANTIC_MEMORY="${disable_source_semantic}" \
  RUN_ASR="${RUN_ASR}" \
  OVERWRITE_INFER="${OVERWRITE_INFER}" \
  NO_TEXT_MIN_AUDIO_TOKENS="${NO_TEXT_MIN_AUDIO_TOKENS}" \
  sh "${EVAL_SCRIPT}"
}

completed=0
if run_eval "ver2_5_ctc_only_headlr10" "1"; then
  completed=$((completed + 1))
fi
if run_eval "ver2_5_source_semantic_ctc_headlr10" "0"; then
  completed=$((completed + 1))
fi

if [ "${completed}" -eq 0 ]; then
  echo "[ver2.5-ctc-eval] no requested checkpoints were ready." >&2
  exit 1
fi

echo "[ver2.5-ctc-eval] completed_runs=${completed}"

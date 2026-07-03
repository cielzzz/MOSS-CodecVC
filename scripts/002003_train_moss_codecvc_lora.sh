#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-tts/bin/python}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/default.yaml}"
MODEL_PATH="${MODEL_PATH:-}"
CODEC_PATH="${CODEC_PATH:-}"
TRAIN_JSONL="${TRAIN_JSONL:-${PROJECT_ROOT}/trainset/legacy_outputs/existing_seedvc_zh_en_1000_vq32/sft/moss_codecvc_sft.mixed_mode.current.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/lora_runs/moss_codecvc_lora_pilot}"
TRAIN_LOG="${TRAIN_LOG:-${OUTPUT_DIR}/train.log}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
N_VQ="${N_VQ:-32}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-auto}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_ROWS="${MAX_ROWS:-0}"
SAVE_STEPS="${SAVE_STEPS:-0}"
EVAL_JSONL="${EVAL_JSONL:-}"
EVAL_JSONL_SPEC="${EVAL_JSONL_SPEC:-}"
EVAL_STEPS="${EVAL_STEPS:-0}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"

LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LM_HEADS_MODE="${LM_HEADS_MODE:-none}"
TRAINABLE_LORA_MODULES="${TRAINABLE_LORA_MODULES:-all}"
RESUME_ADAPTER_PATH="${RESUME_ADAPTER_PATH:-}"
SMOKE_TEST="${SMOKE_TEST:-0}"
PACK_ONLY="${PACK_ONLY:-0}"
EXTRA_ARGS_STR="${EXTRA_ARGS_STR:-}"

ARGS=(
  --config "${CONFIG}"
  --train-jsonl "${TRAIN_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --num-epochs "${NUM_EPOCHS}"
  --mixed-precision "${MIXED_PRECISION}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --num-workers "${NUM_WORKERS}"
  --n-vq "${N_VQ}"
  --max-rows "${MAX_ROWS}"
  --lora-r "${LORA_R}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
  --lm-heads-mode "${LM_HEADS_MODE}"
  --trainable-lora-modules "${TRAINABLE_LORA_MODULES}"
  --gradient-checkpointing
)

if [ -n "${MODEL_PATH}" ]; then
  ARGS+=(--model-path "${MODEL_PATH}")
fi
if [ -n "${CODEC_PATH}" ]; then
  ARGS+=(--codec-path "${CODEC_PATH}")
fi
if [ -n "${MAX_TRAIN_STEPS}" ] && [ "${MAX_TRAIN_STEPS}" != "0" ]; then
  ARGS+=(--max-train-steps "${MAX_TRAIN_STEPS}")
fi
if [ -n "${SAVE_STEPS}" ] && [ "${SAVE_STEPS}" != "0" ]; then
  ARGS+=(--save-steps "${SAVE_STEPS}")
fi
if [ -n "${EVAL_JSONL}" ]; then
  ARGS+=(--eval-jsonl "${EVAL_JSONL}")
fi
if [ -n "${EVAL_JSONL_SPEC}" ]; then
  ARGS+=(--eval-jsonl-spec "${EVAL_JSONL_SPEC}")
fi
if [ -n "${EVAL_STEPS}" ] && [ "${EVAL_STEPS}" != "0" ]; then
  ARGS+=(--eval-steps "${EVAL_STEPS}")
fi
if [ -n "${EVAL_MAX_BATCHES}" ] && [ "${EVAL_MAX_BATCHES}" != "0" ]; then
  ARGS+=(--eval-max-batches "${EVAL_MAX_BATCHES}")
fi
ARGS+=(--eval-num-workers "${EVAL_NUM_WORKERS}")
if [ -n "${RESUME_ADAPTER_PATH}" ]; then
  ARGS+=(--resume-adapter-path "${RESUME_ADAPTER_PATH}")
fi
if [ "${SMOKE_TEST}" = "1" ]; then
  ARGS+=(--smoke-test)
fi
if [ "${PACK_ONLY}" = "1" ]; then
  ARGS+=(--pack-only)
fi
if [ -n "${EXTRA_ARGS_STR}" ]; then
  read -r -a EXTRA_ARGS <<< "${EXTRA_ARGS_STR}"
  ARGS+=("${EXTRA_ARGS[@]}")
fi

mkdir -p "${OUTPUT_DIR}"

if [ "${ENABLE_TRAIN_LOG:-1}" = "1" ]; then
  mkdir -p "$(dirname "${TRAIN_LOG}")"
  exec > >(tee -a "${TRAIN_LOG}") 2>&1
fi

echo "[MOSS-CodecVC LoRA] config=${CONFIG}"
echo "[MOSS-CodecVC LoRA] train_jsonl=${TRAIN_JSONL}"
echo "[MOSS-CodecVC LoRA] output_dir=${OUTPUT_DIR}"
echo "[MOSS-CodecVC LoRA] eval_jsonl=${EVAL_JSONL} eval_jsonl_spec=${EVAL_JSONL_SPEC}"
echo "[MOSS-CodecVC LoRA] train_log=${TRAIN_LOG}"
echo "[MOSS-CodecVC LoRA] n_vq=${N_VQ} lora_r=${LORA_R} lm_heads_mode=${LM_HEADS_MODE}"
"${PYTHON}" "${SCRIPT_DIR}/002002_train_moss_codecvc_lora.py" "${ARGS[@]}"

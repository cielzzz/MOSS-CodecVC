#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

MOSS_TTS_ROOT="${MOSS_TTS_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-TTS}"
PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-tts/bin/python}"
MODEL_PATH="${MODEL_PATH:-OpenMOSS-Team/MOSS-TTS}"
CODEC_PATH="${CODEC_PATH:-${MOSS_TTS_ROOT}/MOSS-Audio-Tokenizer}"
TRAIN_JSONL="${TRAIN_JSONL:-${PROJECT_ROOT}/trainset/legacy_outputs/existing_seedvc_zh_en_1000_vq32/sft/moss_codecvc_sft.mixed_mode.current.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/sft/moss_codecvc_pilot}"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-}"
ACCELERATE_EXTRA_ARGS_STR="${ACCELERATE_EXTRA_ARGS_STR:-}"
TRAIN_EXTRA_ARGS_STR="${TRAIN_EXTRA_ARGS_STR:-}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
N_VQ="${N_VQ:-32}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-auto}"
CHANNELWISE_LOSS_WEIGHT="${CHANNELWISE_LOSS_WEIGHT:-1,32}"
NUM_WORKERS="${NUM_WORKERS:-0}"

cd "${MOSS_TTS_ROOT}"

ARGS=(
  --model-path "${MODEL_PATH}"
  --codec-path "${CODEC_PATH}"
  --train-jsonl "${TRAIN_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --warmup-ratio "${WARMUP_RATIO}"
  --num-epochs "${NUM_EPOCHS}"
  --mixed-precision "${MIXED_PRECISION}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --num-workers "${NUM_WORKERS}"
  --n-vq "${N_VQ}"
  --channelwise-loss-weight "${CHANNELWISE_LOSS_WEIGHT}"
  --gradient-checkpointing
)

if [ -n "${MAX_TRAIN_STEPS}" ] && [ "${MAX_TRAIN_STEPS}" != "0" ]; then
  ARGS+=(--max-train-steps "${MAX_TRAIN_STEPS}")
fi

if [ -n "${TRAIN_EXTRA_ARGS_STR}" ]; then
  read -r -a TRAIN_EXTRA_ARGS <<< "${TRAIN_EXTRA_ARGS_STR}"
  ARGS+=("${TRAIN_EXTRA_ARGS[@]}")
fi

echo "[MOSS-CodecVC] train_jsonl=${TRAIN_JSONL}"
echo "[MOSS-CodecVC] output_dir=${OUTPUT_DIR}"
echo "[MOSS-CodecVC] model_path=${MODEL_PATH}"
echo "[MOSS-CodecVC] codec_path=${CODEC_PATH}"
echo "[MOSS-CodecVC] n_vq=${N_VQ}"

if [ -n "${ACCELERATE_CONFIG_FILE}" ]; then
  LAUNCH_ARGS=(--config_file "${ACCELERATE_CONFIG_FILE}")
  if [ -n "${ACCELERATE_EXTRA_ARGS_STR}" ]; then
    read -r -a ACCELERATE_EXTRA_ARGS <<< "${ACCELERATE_EXTRA_ARGS_STR}"
    LAUNCH_ARGS+=("${ACCELERATE_EXTRA_ARGS[@]}")
  fi
  echo "[MOSS-CodecVC] accelerate_config=${ACCELERATE_CONFIG_FILE}"
  "${PYTHON}" -m accelerate.commands.launch "${LAUNCH_ARGS[@]}" \
    moss_tts_delay/finetuning/sft.py "${ARGS[@]}"
else
  "${PYTHON}" moss_tts_delay/finetuning/sft.py "${ARGS[@]}"
fi

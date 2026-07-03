#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
CONFIG="${CONFIG:-configs/remote_full.yaml}"
MODEL_PATH="${MODEL_PATH:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction/MOSS-TTS}"
TRAIN_JSONL="${TRAIN_JSONL:-${SCRIPT_DIR}/data/train_current_balanced_zh500_en500.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/lora_runs/formal_lora_zh1000_en1000_warmup}"

cd "${PROJECT_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
PYTHON="${PYTHON}" \
CONFIG="${CONFIG}" \
TRAIN_JSONL="${TRAIN_JSONL}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
MODEL_PATH="${MODEL_PATH}" \
N_VQ=32 \
MAX_TRAIN_STEPS=600 \
SAVE_STEPS=100 \
SMOKE_TEST=0 \
bash scripts/002003_train_moss_codecvc_lora.sh

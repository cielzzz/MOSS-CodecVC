#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

PY_MOSS="${PY_MOSS:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
if [ ! -x "${PY_MOSS}" ]; then
  PY_MOSS=python
fi
PY_SEEDVC="${PY_SEEDVC:-/inspire/ssd/project/embodied-multimodality/public/yqzhang/miniconda3/envs/contts-train/bin/python}"

DATA_ROOT="${DATA_ROOT:-/inspire/qb-ilm/project/embodied-multimodality/public/xyzhang/data/VC_train}"
INPUT_ROOT="${INPUT_ROOT:-${DATA_ROOT}/v2_real_target_pilot_20260706}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/v2_real_target_seedvc_triples_pilot_20260707}"
RUN_NAME="${RUN_NAME:-v2_real_target_seedvc_triples_pilot_20260707}"

PAIR_CONSTRUCTION_ROOT="${PAIR_CONSTRUCTION_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction}"
SEEDVC_ROUTE_ROOT="${SEEDVC_ROUTE_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction_prosody_routes}"
SEED_VC_DIR="${SEED_VC_DIR:-${SEEDVC_ROUTE_ROOT}/third_party/seed-vc}"
DEPS_DIR="${DEPS_DIR:-${SEEDVC_ROUTE_ROOT}/.deps/seedvc}"

JOBS_JSONL="${JOBS_JSONL:-${OUTPUT_ROOT}/source_seedvc_jobs.jsonl}"
RESULTS_JSONL="${RESULTS_JSONL:-${OUTPUT_ROOT}/source_seedvc_results.jsonl}"

RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_SEEDVC="${RUN_SEEDVC:-0}"
RUN_COLLECT="${RUN_COLLECT:-0}"
MAX_ROWS="${MAX_ROWS:-0}"

SEEDVC_GPU_IDS="${SEEDVC_GPU_IDS:-${GPU_IDS:-}}"
SEEDVC_SHARD_COUNT="${SEEDVC_SHARD_COUNT:-0}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-25}"
LENGTH_ADJUST="${LENGTH_ADJUST:-1.0}"
INFERENCE_CFG_RATE="${INFERENCE_CFG_RATE:-0.7}"
FP16="${FP16:-true}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MAX_JOBS="${MAX_JOBS:-0}"
FAIL_FAST="${FAIL_FAST:-0}"
SHOW_MODEL_OUTPUT="${SHOW_MODEL_OUTPUT:-0}"
MIN_SOURCE_AUDIO_BYTES="${MIN_SOURCE_AUDIO_BYTES:-4096}"

echo "[v2-real-target] root=${ROOT}"
echo "[v2-real-target] input_root=${INPUT_ROOT}"
echo "[v2-real-target] output_root=${OUTPUT_ROOT}"
echo "[v2-real-target] jobs=${JOBS_JSONL}"
echo "[v2-real-target] results=${RESULTS_JSONL}"
echo "[v2-real-target] run_prepare=${RUN_PREPARE} run_seedvc=${RUN_SEEDVC} run_collect=${RUN_COLLECT}"

cd "${ROOT}"
"${PY_MOSS}" -m py_compile \
  dataset_scripts/build_v2_seedvc_real_target_triples.py \
  dataset_scripts/build_v2_real_target_pilot_from_resultf.py

if [ "${RUN_PREPARE}" = "1" ]; then
  "${PY_MOSS}" dataset_scripts/build_v2_seedvc_real_target_triples.py \
    prepare \
    --input-root "${INPUT_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --run-name "${RUN_NAME}" \
    --max-rows "${MAX_ROWS}" \
    --overwrite
fi

if [ "${RUN_SEEDVC}" = "1" ]; then
  export PYTHONPATH="${DEPS_DIR}:${PAIR_CONSTRUCTION_ROOT}/scripts:${SEED_VC_DIR}:${PYTHONPATH:-}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export HF_HOME="${SEED_VC_DIR}/checkpoints/hf_cache"
  export HUGGINGFACE_HUB_CACHE="${SEED_VC_DIR}/checkpoints/hf_cache"
  export TRANSFORMERS_CACHE="${SEED_VC_DIR}/checkpoints/hf_cache"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

  PY="${PY_SEEDVC}" \
  JOBS_JSONL="${JOBS_JSONL}" \
  RESULTS_JSONL="${RESULTS_JSONL}" \
  SEED_VC_DIR="${SEED_VC_DIR}" \
  SEEDVC_GPU_IDS="${SEEDVC_GPU_IDS}" \
  SEEDVC_SHARD_COUNT="${SEEDVC_SHARD_COUNT}" \
  DIFFUSION_STEPS="${DIFFUSION_STEPS}" \
  LENGTH_ADJUST="${LENGTH_ADJUST}" \
  INFERENCE_CFG_RATE="${INFERENCE_CFG_RATE}" \
  FP16="${FP16}" \
  SKIP_EXISTING="${SKIP_EXISTING}" \
  MAX_JOBS="${MAX_JOBS}" \
  FAIL_FAST="${FAIL_FAST}" \
  SHOW_MODEL_OUTPUT="${SHOW_MODEL_OUTPUT}" \
    bash "${PAIR_CONSTRUCTION_ROOT}/scripts/run_seedvc_jobs_sharded.sh"
fi

if [ "${RUN_COLLECT}" = "1" ]; then
  "${PY_MOSS}" dataset_scripts/build_v2_seedvc_real_target_triples.py \
    collect \
    --output-root "${OUTPUT_ROOT}" \
    --results-jsonl "${RESULTS_JSONL}" \
    --min-source-audio-bytes "${MIN_SOURCE_AUDIO_BYTES}" \
    --overwrite
fi

echo "[v2-real-target] done"

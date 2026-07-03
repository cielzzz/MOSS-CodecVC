#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

PY_MOSS="${PY_MOSS:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
PY_SEEDVC="${PY_SEEDVC:-/inspire/ssd/project/embodied-multimodality/public/yqzhang/miniconda3/envs/contts-train/bin/python}"
PAIR_CONSTRUCTION_ROOT="${PAIR_CONSTRUCTION_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction}"
SEED_VC_DIR="${SEED_VC_DIR:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction_prosody_routes/third_party/seed-vc}"
DEPS_DIR="${DEPS_DIR:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction_prosody_routes/.deps/seedvc}"

RUN_NAME="${RUN_NAME:-text_prosody_mosstts_seedvc_independent_timbre_zh_en_0001_0004}"
WORK_ROOT="${WORK_ROOT:-${ROOT}/trainset/text_prosody_mosstts_seedvc/independent_timbre_zh_en_0001_0004}"
JOBS_JSONL="${JOBS_JSONL:-${WORK_ROOT}/text_seedvc_jobs.jsonl}"
RESULTS_JSONL="${RESULTS_JSONL:-${WORK_ROOT}/text_seedvc_results.jsonl}"
MANIFEST_JSONL="${MANIFEST_JSONL:-${WORK_ROOT}/vc_manifest.text_prosody.jsonl}"
TARGET_AUDIO_ROOT="${TARGET_AUDIO_ROOT:-${WORK_ROOT}/seedvc_targets}"

RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_SEEDVC="${RUN_SEEDVC:-1}"
RUN_COLLECT="${RUN_COLLECT:-1}"
PREPARE_REQUIRE_EXISTING_AUDIO="${PREPARE_REQUIRE_EXISTING_AUDIO:-1}"
PREPARE_PROGRESS_EVERY="${PREPARE_PROGRESS_EVERY:-1000}"

MAX_JOBS="${MAX_JOBS:-0}"
MAX_JOBS_PER_LANGUAGE="${MAX_JOBS_PER_LANGUAGE:-0}"
MAX_ROWS_PER_INPUT="${MAX_ROWS_PER_INPUT:-0}"
MIN_BEST_SIMILARITY="${MIN_BEST_SIMILARITY:-0.0}"
MIN_DNSMOS="${MIN_DNSMOS:-0.0}"
SKIP_FLAGS="${SKIP_FLAGS:-}"
LANGUAGES="${LANGUAGES:-zh,en}"
TIMBRE_REF_POLICY="${TIMBRE_REF_POLICY:-random_different_text}"
TIMBRE_REF_SEED="${TIMBRE_REF_SEED:-20260627}"

SEEDVC_GPU_IDS="${SEEDVC_GPU_IDS:-${GPU_IDS:-}}"
SEEDVC_SHARD_COUNT="${SEEDVC_SHARD_COUNT:-0}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-25}"
LENGTH_ADJUST="${LENGTH_ADJUST:-1.0}"
INFERENCE_CFG_RATE="${INFERENCE_CFG_RATE:-0.7}"
FP16="${FP16:-true}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FAIL_FAST="${FAIL_FAST:-0}"
SHOW_MODEL_OUTPUT="${SHOW_MODEL_OUTPUT:-0}"
MIN_TARGET_AUDIO_BYTES="${MIN_TARGET_AUDIO_BYTES:-4096}"

DEFAULT_VCDATA_JSONLS=(
  "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/outputs/mtd_pass_nonmulti_primary_le_0p3_en100_pair_20260611_run01/vcdata/en/en_slim_0001/merged.stepaudio_input.all.jsonl"
  "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/outputs/moss_tts_data_temp_zhen10k_qz_20260613_run03/vcdata/en/en_slim_0002/merged.stepaudio_input.all.jsonl"
  "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/outputs/moss_tts_data_temp_zhen10k_qz_20260613_run03/vcdata/en/en_slim_0003/merged.stepaudio_input.all.jsonl"
  "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/outputs/mtd_pass_nonmulti_primary_le_0p3_zh0004_en0004_ij_qz_20260621_run01/vcdata/en/en_slim_0004/merged.stepaudio_input.all.jsonl"
  "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/outputs/mtd_pass_nonmulti_primary_le_0p3_zh100_pair_20260611_run02/vcdata/zh/zh_slim_0001/merged.stepaudio_input.all.jsonl"
  "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/outputs/moss_tts_data_temp_zhen10k_qz_20260613_run03/vcdata/zh/zh_slim_0002/merged.stepaudio_input.all.jsonl"
  "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/outputs/moss_tts_data_temp_zhen10k_qz_20260613_run03/vcdata/zh/zh_slim_0003/merged.stepaudio_input.all.jsonl"
  "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/outputs/mtd_pass_nonmulti_primary_le_0p3_zh0004_en0004_ij_qz_20260621_run01/vcdata/zh/zh_slim_0004/merged.stepaudio_input.all.jsonl"
)

mkdir -p "${WORK_ROOT}"

echo "[text-prosody] run_name=${RUN_NAME}"
echo "[text-prosody] work_root=${WORK_ROOT}"
echo "[text-prosody] jobs=${JOBS_JSONL}"
echo "[text-prosody] results=${RESULTS_JSONL}"
echo "[text-prosody] manifest=${MANIFEST_JSONL}"
echo "[text-prosody] timbre_ref_policy=${TIMBRE_REF_POLICY} seed=${TIMBRE_REF_SEED}"

if [ "${RUN_PREPARE}" = "1" ]; then
  prepare_args=(
    "${SCRIPT_DIR}/001034_build_text_prosody_from_mosstts_vcdata.py"
    prepare
    --jobs-jsonl "${JOBS_JSONL}"
    --target-audio-root "${TARGET_AUDIO_ROOT}"
    --run-name "${RUN_NAME}"
    --languages "${LANGUAGES}"
    --max-rows-per-input "${MAX_ROWS_PER_INPUT}"
    --max-jobs "${MAX_JOBS}"
    --max-jobs-per-language "${MAX_JOBS_PER_LANGUAGE}"
    --timbre-ref-policy "${TIMBRE_REF_POLICY}"
    --timbre-ref-seed "${TIMBRE_REF_SEED}"
    --min-best-similarity "${MIN_BEST_SIMILARITY}"
    --min-dnsmos "${MIN_DNSMOS}"
    --progress-every "${PREPARE_PROGRESS_EVERY}"
    --summary-json "${WORK_ROOT}/text_seedvc_jobs.summary.json"
    --overwrite
  )
  if [ "${PREPARE_REQUIRE_EXISTING_AUDIO}" = "0" ]; then
    prepare_args+=(--no-require-existing-audio)
  fi
  if [ -n "${SKIP_FLAGS}" ]; then
    prepare_args+=(--skip-flags "${SKIP_FLAGS}")
  fi
  if [ -n "${VCDATA_JSONLS:-}" ]; then
    IFS=',' read -r -a custom_jsonls <<< "${VCDATA_JSONLS}"
    for jsonl in "${custom_jsonls[@]}"; do
      prepare_args+=(--vcdata-jsonl "${jsonl}")
    done
  else
    for jsonl in "${DEFAULT_VCDATA_JSONLS[@]}"; do
      prepare_args+=(--vcdata-jsonl "${jsonl}")
    done
  fi
  "${PY_MOSS}" "${prepare_args[@]}"
fi

if [ "${RUN_SEEDVC}" = "1" ]; then
  export PYTHONPATH="${DEPS_DIR}:${PAIR_CONSTRUCTION_ROOT}/scripts:${SEED_VC_DIR}:${PYTHONPATH:-}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  # Seed-VC keeps Whisper/BigVGAN in its own cache. Do not let a generic
  # outer HF_HOME override this, otherwise offline Qizhi jobs cannot load.
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
  "${PY_MOSS}" "${SCRIPT_DIR}/001034_build_text_prosody_from_mosstts_vcdata.py" \
    collect \
    --jobs-jsonl "${JOBS_JSONL}" \
    --results-jsonl "${RESULTS_JSONL}" \
    --output-jsonl "${MANIFEST_JSONL}" \
    --run-name "${RUN_NAME}" \
    --summary-json "${WORK_ROOT}/vc_manifest.text_prosody.summary.json" \
    --min-target-audio-bytes "${MIN_TARGET_AUDIO_BYTES}" \
    --overwrite
fi

echo "[text-prosody] done"

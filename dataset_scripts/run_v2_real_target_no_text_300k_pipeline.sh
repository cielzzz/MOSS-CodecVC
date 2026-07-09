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
RUN_NAME="${RUN_NAME:-v2_real_target_no_text_300k_zh_en_balanced_20260707}"
PAIR_ROOT="${PAIR_ROOT:-${DATA_ROOT}/${RUN_NAME}}"
TRIPLE_ROOT="${TRIPLE_ROOT:-${DATA_ROOT}/${RUN_NAME}_seedvc_triples}"

NUM_NO_TEXT="${NUM_NO_TEXT:-300000}"
MAX_PER_DATASET="${MAX_PER_DATASET:-60000}"
REF_MAX_SEC="${REF_MAX_SEC:-30.0}"
MIN_SIMILARITY="${MIN_SIMILARITY:-0.85}"
TARGET_MIN_SEC="${TARGET_MIN_SEC:-2.0}"
TARGET_MAX_SEC="${TARGET_MAX_SEC:-15.0}"
REF_MIN_SEC="${REF_MIN_SEC:-4.0}"
LANGUAGES="${LANGUAGES:-zh,en}"
BALANCE_LANGUAGES="${BALANCE_LANGUAGES:-1}"
DATASETS="${DATASETS:-apple_podcast_estwfiruchnonltzph,apple_podcast_josailczinplittrropkthmauavelyye,apple_podcast_vnjpidgrfrkemyptpededk,haitianruisheng_1,haitianruisheng_2,haitianruisheng_3,haitianruisheng_4,haitianruisheng_6,haitianruisheng_7,haitianruisheng_8,haitianruisheng_9,qingting_fm,rchive_rss_podcast_v2}"
RCLONE_TRANSFERS="${RCLONE_TRANSFERS:-32}"
RCLONE_CHECKERS="${RCLONE_CHECKERS:-64}"
DOWNLOAD_AUDIO="${DOWNLOAD_AUDIO:-1}"

RUN_PAIR_PREPARE="${RUN_PAIR_PREPARE:-1}"
RUN_TRIPLE_PREPARE="${RUN_TRIPLE_PREPARE:-1}"
RUN_SEEDVC="${RUN_SEEDVC:-1}"
RUN_COLLECT="${RUN_COLLECT:-1}"
RUN_REF_CHANNEL_AUGMENT="${RUN_REF_CHANNEL_AUGMENT:-1}"

PAIR_CONSTRUCTION_ROOT="${PAIR_CONSTRUCTION_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction}"
SEEDVC_ROUTE_ROOT="${SEEDVC_ROUTE_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction_prosody_routes}"
SEED_VC_DIR="${SEED_VC_DIR:-${SEEDVC_ROUTE_ROOT}/third_party/seed-vc}"
DEPS_DIR="${DEPS_DIR:-${SEEDVC_ROUTE_ROOT}/.deps/seedvc}"

JOBS_JSONL="${JOBS_JSONL:-${TRIPLE_ROOT}/source_seedvc_jobs.jsonl}"
RESULTS_JSONL="${RESULTS_JSONL:-${TRIPLE_ROOT}/source_seedvc_results.jsonl}"
NO_TEXT_MANIFEST="${NO_TEXT_MANIFEST:-${TRIPLE_ROOT}/no_text.train.manifest.jsonl}"
NO_TEXT_SIMPLE="${NO_TEXT_SIMPLE:-${TRIPLE_ROOT}/no_text.train.simple.jsonl}"
NO_TEXT_REF_AUG_MANIFEST="${NO_TEXT_REF_AUG_MANIFEST:-${TRIPLE_ROOT}/no_text.train.refmix.manifest.jsonl}"
NO_TEXT_REF_AUG_SIMPLE="${NO_TEXT_REF_AUG_SIMPLE:-${TRIPLE_ROOT}/no_text.train.refmix.simple.jsonl}"
REF_AUG_AUDIO_ROOT="${REF_AUG_AUDIO_ROOT:-${TRIPLE_ROOT}/ref_channel_augmented_timbre_refs}"
REF_AUG_RISK_MODE="${REF_AUG_RISK_MODE:-same_episode}"
REF_AUG_FRACTION="${REF_AUG_FRACTION:-0.3}"
REF_AUG_JOBS="${REF_AUG_JOBS:-16}"
REF_AUG_AUDIO_EXTENSION="${REF_AUG_AUDIO_EXTENSION:-.wav}"
REF_AUG_FFMPEG="${REF_AUG_FFMPEG:-/opt/conda/envs/speech/bin/ffmpeg}"
REF_AUG_LOUDNESS_MATCH="${REF_AUG_LOUDNESS_MATCH:-mean_volume}"

SEEDVC_GPU_IDS="${SEEDVC_GPU_IDS:-${GPU_IDS:-0,1,2,3,4,5,6,7}}"
SEEDVC_SHARD_COUNT="${SEEDVC_SHARD_COUNT:-8}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-25}"
LENGTH_ADJUST="${LENGTH_ADJUST:-1.0}"
INFERENCE_CFG_RATE="${INFERENCE_CFG_RATE:-0.7}"
FP16="${FP16:-true}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MAX_JOBS="${MAX_JOBS:-0}"
FAIL_FAST="${FAIL_FAST:-0}"
SHOW_MODEL_OUTPUT="${SHOW_MODEL_OUTPUT:-0}"
MIN_SOURCE_AUDIO_BYTES="${MIN_SOURCE_AUDIO_BYTES:-4096}"

mkdir -p "${DATA_ROOT}" "${PAIR_ROOT}" "${TRIPLE_ROOT}"

PIPELINE_LOCK_DIR="${PIPELINE_LOCK_DIR:-${TRIPLE_ROOT}/.v2_no_text_300k_pipeline.lock}"
if mkdir "${PIPELINE_LOCK_DIR}" 2>/dev/null; then
  {
    echo "pid=$$"
    echo "host=$(hostname)"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "triple_root=${TRIPLE_ROOT}"
  } > "${PIPELINE_LOCK_DIR}/owner.txt"
  cleanup_pipeline_lock() {
    rm -f "${PIPELINE_LOCK_DIR}/owner.txt"
    rmdir "${PIPELINE_LOCK_DIR}" 2>/dev/null || true
  }
  trap cleanup_pipeline_lock EXIT
else
  echo "ERROR: another v2 no-text pipeline appears to be running: ${PIPELINE_LOCK_DIR}" >&2
  if [ -f "${PIPELINE_LOCK_DIR}/owner.txt" ]; then
    cat "${PIPELINE_LOCK_DIR}/owner.txt" >&2 || true
  fi
  exit 75
fi

echo "[v2-no-text-300k] root=${ROOT}"
echo "[v2-no-text-300k] pair_root=${PAIR_ROOT}"
echo "[v2-no-text-300k] triple_root=${TRIPLE_ROOT}"
echo "[v2-no-text-300k] num_no_text=${NUM_NO_TEXT} max_per_dataset=${MAX_PER_DATASET} ref_max_sec=${REF_MAX_SEC}"
echo "[v2-no-text-300k] languages=${LANGUAGES} balance_languages=${BALANCE_LANGUAGES}"
echo "[v2-no-text-300k] datasets=${DATASETS}"
echo "[v2-no-text-300k] download_audio=${DOWNLOAD_AUDIO} rclone_transfers=${RCLONE_TRANSFERS} rclone_checkers=${RCLONE_CHECKERS}"
echo "[v2-no-text-300k] run_pair_prepare=${RUN_PAIR_PREPARE} run_triple_prepare=${RUN_TRIPLE_PREPARE} run_seedvc=${RUN_SEEDVC} run_collect=${RUN_COLLECT} run_ref_channel_augment=${RUN_REF_CHANNEL_AUGMENT}"
echo "[v2-no-text-300k] seedvc_gpu_ids=${SEEDVC_GPU_IDS} shard_count=${SEEDVC_SHARD_COUNT} max_jobs=${MAX_JOBS}"

if [ "${RUN_REF_CHANNEL_AUGMENT}" = "1" ]; then
  if [ ! -x "${REF_AUG_FFMPEG}" ]; then
    for candidate in /opt/conda/envs/speech/bin/ffmpeg /usr/bin/ffmpeg /usr/local/bin/ffmpeg /opt/conda/bin/ffmpeg; do
      if [ -x "${candidate}" ]; then
        REF_AUG_FFMPEG="${candidate}"
        break
      fi
    done
  fi
  if [ ! -x "${REF_AUG_FFMPEG}" ] && command -v ffmpeg >/dev/null 2>&1; then
    REF_AUG_FFMPEG="$(command -v ffmpeg)"
  fi
fi
echo "[v2-no-text-300k] ref_aug risk_mode=${REF_AUG_RISK_MODE} fraction=${REF_AUG_FRACTION} jobs=${REF_AUG_JOBS} loudness_match=${REF_AUG_LOUDNESS_MATCH} ffmpeg=${REF_AUG_FFMPEG} audio_root=${REF_AUG_AUDIO_ROOT}"

cd "${ROOT}"
"${PY_MOSS}" -m py_compile \
  dataset_scripts/build_v2_real_target_pilot_from_resultf.py \
  dataset_scripts/build_v2_seedvc_real_target_triples.py \
  dataset_scripts/augment_v2_ref_channel.py

if [ "${RUN_PAIR_PREPARE}" = "1" ]; then
  "${PY_MOSS}" dataset_scripts/build_v2_real_target_pilot_from_resultf.py \
    --output-root "${PAIR_ROOT}" \
    --run-name "${RUN_NAME}" \
    --num-no-text "${NUM_NO_TEXT}" \
    --num-text 0 \
    --max-per-dataset "${MAX_PER_DATASET}" \
    --min-similarity "${MIN_SIMILARITY}" \
    --target-min-sec "${TARGET_MIN_SEC}" \
    --target-max-sec "${TARGET_MAX_SEC}" \
    --ref-min-sec "${REF_MIN_SEC}" \
    --ref-max-sec "${REF_MAX_SEC}" \
    --languages "${LANGUAGES}" \
    "$([ "${BALANCE_LANGUAGES}" = "1" ] && printf '%s' '--balance-languages' || printf '%s' '--no-balance-languages')" \
    --datasets "${DATASETS}" \
    --no-load-newtrain-info \
    --materialize-audio \
    "$([ "${DOWNLOAD_AUDIO}" = "1" ] && printf '%s' '--download-audio' || printf '%s' '--no-download-audio')" \
    --rclone-transfers "${RCLONE_TRANSFERS}" \
    --rclone-checkers "${RCLONE_CHECKERS}"
fi

if [ "${RUN_TRIPLE_PREPARE}" = "1" ]; then
  "${PY_MOSS}" dataset_scripts/build_v2_seedvc_real_target_triples.py \
    prepare \
    --input-root "${PAIR_ROOT}" \
    --output-root "${TRIPLE_ROOT}" \
    --run-name "${RUN_NAME}_seedvc_triples" \
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
    --output-root "${TRIPLE_ROOT}" \
    --results-jsonl "${RESULTS_JSONL}" \
    --modes no_text \
    --require-result-ok \
    --min-source-audio-bytes "${MIN_SOURCE_AUDIO_BYTES}" \
    --overwrite
fi

if [ "${RUN_REF_CHANNEL_AUGMENT}" = "1" ] && [ -f "${NO_TEXT_MANIFEST}" ]; then
  "${PY_MOSS}" dataset_scripts/augment_v2_ref_channel.py \
    --input-jsonl "${NO_TEXT_MANIFEST}" \
    --output-jsonl "${NO_TEXT_REF_AUG_MANIFEST}" \
    --audio-output-root "${REF_AUG_AUDIO_ROOT}" \
    --summary-json "${TRIPLE_ROOT}/summary.ref_channel_augment.manifest.json" \
    --ffmpeg "${REF_AUG_FFMPEG}" \
    --risk-mode "${REF_AUG_RISK_MODE}" \
    --sample-fraction "${REF_AUG_FRACTION}" \
    --audio-extension "${REF_AUG_AUDIO_EXTENSION}" \
    --loudness-match "${REF_AUG_LOUDNESS_MATCH}" \
    --jobs "${REF_AUG_JOBS}" \
    --overwrite

  if [ -f "${NO_TEXT_SIMPLE}" ]; then
    "${PY_MOSS}" dataset_scripts/augment_v2_ref_channel.py \
      --input-jsonl "${NO_TEXT_SIMPLE}" \
      --output-jsonl "${NO_TEXT_REF_AUG_SIMPLE}" \
      --audio-output-root "${REF_AUG_AUDIO_ROOT}" \
      --summary-json "${TRIPLE_ROOT}/summary.ref_channel_augment.simple.json" \
      --ffmpeg "${REF_AUG_FFMPEG}" \
      --risk-mode "${REF_AUG_RISK_MODE}" \
      --sample-fraction "${REF_AUG_FRACTION}" \
      --audio-extension "${REF_AUG_AUDIO_EXTENSION}" \
      --loudness-match "${REF_AUG_LOUDNESS_MATCH}" \
      --jobs "${REF_AUG_JOBS}" \
      --overwrite
  fi
fi

echo "[v2-no-text-300k] final counts"
test -f "${PAIR_ROOT}/no_text.train.simple.jsonl" && wc -l "${PAIR_ROOT}/no_text.train.simple.jsonl" || true
test -f "${JOBS_JSONL}" && wc -l "${JOBS_JSONL}" || true
test -f "${RESULTS_JSONL}" && wc -l "${RESULTS_JSONL}" || true
test -f "${NO_TEXT_MANIFEST}" && wc -l "${NO_TEXT_MANIFEST}" || true
test -f "${NO_TEXT_REF_AUG_MANIFEST}" && wc -l "${NO_TEXT_REF_AUG_MANIFEST}" || true
echo "[v2-no-text-300k] done"

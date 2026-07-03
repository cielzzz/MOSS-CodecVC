#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-tts/bin/python}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/default.yaml}"
INPUT_JSONL="${INPUT_JSONL:?set INPUT_JSONL=/path/to/source_or_pair.jsonl}"
RUN_NAME="${RUN_NAME:-moss_codecvc_$(date +%Y%m%d_%H%M%S)}"
MODE="${MODE:-from_pairs}"
LIMIT="${LIMIT:-0}"
N_VQ="${N_VQ:-32}"
DEVICE="${DEVICE:-cuda:0}"
CODEC_DTYPE="${CODEC_DTYPE:-float32}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/trainset/${RUN_NAME}}"
MANIFEST_JSONL="${MANIFEST_JSONL:-${OUTPUT_ROOT}/manifests/vc_manifest.jsonl}"
ENCODED_JSONL="${ENCODED_JSONL:-${OUTPUT_ROOT}/encoded/vc_manifest.encoded.jsonl}"
CODES_DIR="${CODES_DIR:-${OUTPUT_ROOT}/codes}"
SFT_JSONL="${SFT_JSONL:-${OUTPUT_ROOT}/sft/moss_codecvc_sft.jsonl}"
TEXT_MODE="${TEXT_MODE:-target}"
TEXT_PLACEHOLDER="${TEXT_PLACEHOLDER:-<NO_TEXT>}"
EMIT_MODES="${EMIT_MODES:-text}"
NO_TEXT_TEXT_MODE="${NO_TEXT_TEXT_MODE:-placeholder}"
NO_TEXT_PLACEHOLDER="${NO_TEXT_PLACEHOLDER:-<NO_TEXT>}"
ENCODE_NO_REUSE="${ENCODE_NO_REUSE:-0}"

ENCODE_REUSE_ARGS=()
if [[ "${ENCODE_NO_REUSE}" == "1" || "${ENCODE_NO_REUSE}" == "true" ]]; then
  ENCODE_REUSE_ARGS+=(--no-reuse)
fi

mkdir -p "${OUTPUT_ROOT}/manifests" "${OUTPUT_ROOT}/encoded" "${OUTPUT_ROOT}/sft" "${CODES_DIR}"

echo "[prepare] project=${PROJECT_ROOT}"
echo "[prepare] input=${INPUT_JSONL}"
echo "[prepare] output_root=${OUTPUT_ROOT}"
echo "[prepare] mode=${MODE} n_vq=${N_VQ} device=${DEVICE} codec_dtype=${CODEC_DTYPE}"
echo "[prepare] text_mode=${TEXT_MODE}"
echo "[prepare] emit_modes=${EMIT_MODES}"
echo "[prepare] encode_no_reuse=${ENCODE_NO_REUSE}"

"${PYTHON}" "${SCRIPT_DIR}/001001_build_vc_manifest.py" \
  --config "${CONFIG}" \
  --input-jsonl "${INPUT_JSONL}" \
  --output-jsonl "${MANIFEST_JSONL}" \
  --mode "${MODE}" \
  --run-name "${RUN_NAME}" \
  --limit "${LIMIT}" \
  --require-target

"${PYTHON}" "${SCRIPT_DIR}/001002_encode_codec_tokens.py" \
  --config "${CONFIG}" \
  --input-jsonl "${MANIFEST_JSONL}" \
  --output-jsonl "${ENCODED_JSONL}" \
  --codes-dir "${CODES_DIR}" \
  --n-vq "${N_VQ}" \
  --device "${DEVICE}" \
  --dtype "${CODEC_DTYPE}" \
  "${ENCODE_REUSE_ARGS[@]}"

"${PYTHON}" "${SCRIPT_DIR}/001003_build_moss_sft_jsonl.py" \
  --input-jsonl "${ENCODED_JSONL}" \
  --output-jsonl "${SFT_JSONL}" \
  --text-mode "${TEXT_MODE}" \
  --text-placeholder "${TEXT_PLACEHOLDER}" \
  --emit-modes "${EMIT_MODES}" \
  --no-text-text-mode "${NO_TEXT_TEXT_MODE}" \
  --no-text-placeholder "${NO_TEXT_PLACEHOLDER}"

echo "[prepare] manifest=${MANIFEST_JSONL}"
echo "[prepare] encoded=${ENCODED_JSONL}"
echo "[prepare] sft=${SFT_JSONL}"

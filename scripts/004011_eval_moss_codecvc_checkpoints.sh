#!/usr/bin/env sh
set -eu

# Fixed-sample checkpoint evaluation for MOSS-CodecVC.
# Default use: sh scripts/004011_eval_moss_codecvc_checkpoints.sh
#
# The script runs no_text inference for selected checkpoints, then optionally
# runs Qwen-ASR to compute CER/WER/repeat_score/duration_ratio.

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd "${SCRIPT_DIR}/.." && pwd)

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
ASR_PYTHON="${ASR_PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
RUN_SCRIPT="${RUN_SCRIPT:-${PROJECT_ROOT}/scripts/003003_run_moss_codecvc_infer.sh}"
ASR_SCRIPT="${ASR_SCRIPT:-${PROJECT_ROOT}/scripts/001017_asr_content_filter.py}"

RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/outputs/lora_runs/ver2_1_68w_textrep3_lora_r16_a32_gbs64_syncfix}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-step-10000 step-11000}"
EVAL_VARIANTS="${EVAL_VARIANTS:-default lowrand}"

SOURCE_AUDIO="${SOURCE_AUDIO:-${PROJECT_ROOT}/testset/source/media.wav}"
TIMBRE_REF_AUDIO="${TIMBRE_REF_AUDIO:-${PROJECT_ROOT}/testset/timbre_ref/zh_a_000001_timbre.wav}"
# Qwen-ASR expects full language names, e.g. Chinese / English, not zh / en.
LANGUAGE="${LANGUAGE:-Chinese}"
SOURCE_ASR_TEXT="${SOURCE_ASR_TEXT:-}"

timestamp=$(date -u +%Y%m%d-%H%M%S)
run_name=$(basename "${RUN_DIR}")
EVAL_NAME="${EVAL_NAME:-${run_name}_media_no_text_${timestamp}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/testset/outputs/checkpoint_eval}"
EVAL_DIR="${EVAL_DIR:-${OUTPUT_ROOT}/${EVAL_NAME}}"

DEVICE="${DEVICE:-auto}"
DEBUG_GENERATION_STRUCTURE="${DEBUG_GENERATION_STRUCTURE:-1}"
NO_TEXT_MAX_TOKEN_MARGIN="${NO_TEXT_MAX_TOKEN_MARGIN:-0}"
NO_TEXT_MIN_AUDIO_TOKENS="${NO_TEXT_MIN_AUDIO_TOKENS:-}"
AUDIO_SEGMENT_POLICY="${AUDIO_SEGMENT_POLICY:-all}"

RUN_INFER="${RUN_INFER:-1}"
OVERWRITE_INFER="${OVERWRITE_INFER:-0}"
RUN_ASR="${RUN_ASR:-1}"
ASR_BACKEND="${ASR_BACKEND:-qwen_asr}"
QWEN_ASR_MODEL="${QWEN_ASR_MODEL:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/checkpoint/qwen-asr-1_7b}"
QWEN_ASR_DTYPE="${QWEN_ASR_DTYPE:-bfloat16}"
QWEN_ASR_MAX_BATCH_SIZE="${QWEN_ASR_MAX_BATCH_SIZE:-1}"
QWEN_ASR_MAX_NEW_TOKENS="${QWEN_ASR_MAX_NEW_TOKENS:-256}"
ASR_DEVICE="${ASR_DEVICE:-cuda:0}"
NO_TEXT_ZH_CER_THRESHOLD="${NO_TEXT_ZH_CER_THRESHOLD:-0.35}"
NO_TEXT_EN_WER_THRESHOLD="${NO_TEXT_EN_WER_THRESHOLD:-0.30}"
MAX_REPEAT_SCORE="${MAX_REPEAT_SCORE:-0.30}"

mkdir -p "${EVAL_DIR}"
input_jsonl="${EVAL_DIR}/asr_input.jsonl"
asr_jsonl="${EVAL_DIR}/asr_eval.jsonl"
metrics_csv="${EVAL_DIR}/metrics.csv"
summary_md="${EVAL_DIR}/SUMMARY.md"
: > "${input_jsonl}"

if [ ! -f "${SOURCE_AUDIO}" ]; then
  echo "SOURCE_AUDIO 不存在: ${SOURCE_AUDIO}" >&2
  exit 1
fi
if [ ! -f "${TIMBRE_REF_AUDIO}" ]; then
  echo "TIMBRE_REF_AUDIO 不存在: ${TIMBRE_REF_AUDIO}" >&2
  exit 1
fi

echo "[ckpt-eval] RUN_DIR=${RUN_DIR}"
echo "[ckpt-eval] CHECKPOINT_STEPS=${CHECKPOINT_STEPS}"
echo "[ckpt-eval] EVAL_VARIANTS=${EVAL_VARIANTS}"
echo "[ckpt-eval] SOURCE_AUDIO=${SOURCE_AUDIO}"
echo "[ckpt-eval] TIMBRE_REF_AUDIO=${TIMBRE_REF_AUDIO}"
echo "[ckpt-eval] EVAL_DIR=${EVAL_DIR}"

append_asr_input_row() {
  sample_id="$1"
  checkpoint_step="$2"
  variant="$3"
  target_audio="$4"
  SAMPLE_ID="${sample_id}" \
  CHECKPOINT_STEP="${checkpoint_step}" \
  VARIANT="${variant}" \
  SOURCE_AUDIO_VALUE="${SOURCE_AUDIO}" \
  TARGET_AUDIO_VALUE="${target_audio}" \
  LANGUAGE_VALUE="${LANGUAGE}" \
  SOURCE_ASR_TEXT_VALUE="${SOURCE_ASR_TEXT}" \
  "${PYTHON}" - <<'PY' >> "${input_jsonl}"
import json
import os

row = {
    "sample_id": os.environ["SAMPLE_ID"],
    "checkpoint_step": os.environ["CHECKPOINT_STEP"],
    "eval_variant": os.environ["VARIANT"],
    "mode": "no_text",
    "moss_codecvc_mode": "no_text",
    "language": os.environ["LANGUAGE_VALUE"],
    "source_audio": os.environ["SOURCE_AUDIO_VALUE"],
    "target_audio": os.environ["TARGET_AUDIO_VALUE"],
    "text": "<NO_TEXT>",
}
source_asr = os.environ.get("SOURCE_ASR_TEXT_VALUE", "").strip()
if source_asr:
    row["asr_src_text"] = source_asr
print(json.dumps(row, ensure_ascii=False))
PY
}

for step_name in ${CHECKPOINT_STEPS}; do
  model_path="${RUN_DIR}/${step_name}"
  if [ ! -d "${model_path}" ]; then
    echo "[ckpt-eval] skip missing checkpoint: ${model_path}" >&2
    continue
  fi
  if [ ! -f "${model_path}/adapter_model.safetensors" ] || [ ! -f "${model_path}/timbre_memory_adapter.pt" ]; then
    echo "[ckpt-eval] skip incomplete checkpoint: ${model_path}" >&2
    continue
  fi

  for variant in ${EVAL_VARIANTS}; do
    sample_id="${step_name}_${variant}"
    out_dir="${EVAL_DIR}/${sample_id}"
    source_stem=$(basename "${SOURCE_AUDIO}")
    source_stem=${source_stem%.*}
    timbre_stem=$(basename "${TIMBRE_REF_AUDIO}")
    timbre_stem=${timbre_stem%.*}
    output_wav="${out_dir}/${source_stem}_${timbre_stem}.wav"

    audio_temperature=""
    audio_top_p=""
    audio_top_k=""
    audio_repetition_penalty=""
    case "${variant}" in
      default)
        ;;
      lowrand)
        audio_temperature="1.0"
        audio_top_p="0.70"
        audio_top_k="20"
        audio_repetition_penalty="1.10"
        ;;
      topk1|near_greedy|greedy)
        audio_temperature="1.0"
        audio_top_p="1.0"
        audio_top_k="1"
        audio_repetition_penalty="1.05"
        ;;
      *)
        echo "[ckpt-eval] unknown variant=${variant}; use default sampling for this variant" >&2
        ;;
    esac

    if [ "${RUN_INFER}" = "1" ]; then
      if [ -f "${output_wav}" ] && [ "${OVERWRITE_INFER}" != "1" ]; then
        echo "[ckpt-eval] reuse existing wav: ${output_wav}"
      else
        echo "[ckpt-eval] infer sample=${sample_id} model=${model_path}"
        MODE="no_text" \
        MODEL_PATH="${model_path}" \
        SOURCE_AUDIO="${SOURCE_AUDIO}" \
        TIMBRE_REF_AUDIO="${TIMBRE_REF_AUDIO}" \
        OUTPUT_DIR="${out_dir}" \
        OUTPUT_WAV="${output_wav}" \
        DEVICE="${DEVICE}" \
        DEBUG_GENERATION_STRUCTURE="${DEBUG_GENERATION_STRUCTURE}" \
        NO_TEXT_MAX_TOKEN_MARGIN="${NO_TEXT_MAX_TOKEN_MARGIN}" \
        NO_TEXT_MIN_AUDIO_TOKENS="${NO_TEXT_MIN_AUDIO_TOKENS}" \
        AUDIO_SEGMENT_POLICY="${AUDIO_SEGMENT_POLICY}" \
        NO_TEXT_AUDIO_TEMPERATURE="${audio_temperature}" \
        NO_TEXT_AUDIO_TOP_P="${audio_top_p}" \
        NO_TEXT_AUDIO_TOP_K="${audio_top_k}" \
        NO_TEXT_AUDIO_REPETITION_PENALTY="${audio_repetition_penalty}" \
        sh "${RUN_SCRIPT}"
      fi
    fi

    if [ -f "${output_wav}" ]; then
      append_asr_input_row "${sample_id}" "${step_name}" "${variant}" "${output_wav}"
    else
      echo "[ckpt-eval] missing wav after inference, not adding ASR row: ${output_wav}" >&2
    fi
  done
done

if [ "${RUN_ASR}" = "1" ]; then
  if [ ! -s "${input_jsonl}" ]; then
    echo "[ckpt-eval] no ASR input rows: ${input_jsonl}" >&2
    exit 1
  fi
  echo "[ckpt-eval] ASR eval -> ${asr_jsonl}"
  "${ASR_PYTHON}" "${ASR_SCRIPT}" \
    --input-jsonl "${input_jsonl}" \
    --output-jsonl "${asr_jsonl}" \
    --asr-backend "${ASR_BACKEND}" \
    --qwen-asr-model "${QWEN_ASR_MODEL}" \
    --qwen-asr-dtype "${QWEN_ASR_DTYPE}" \
    --qwen-asr-max-batch-size "${QWEN_ASR_MAX_BATCH_SIZE}" \
    --qwen-asr-max-new-tokens "${QWEN_ASR_MAX_NEW_TOKENS}" \
    --device "${ASR_DEVICE}" \
    --language "${LANGUAGE}" \
    --no-text-zh-cer-threshold "${NO_TEXT_ZH_CER_THRESHOLD}" \
    --no-text-en-wer-threshold "${NO_TEXT_EN_WER_THRESHOLD}" \
    --max-repeat-score "${MAX_REPEAT_SCORE}" \
    --overwrite
else
  echo "[ckpt-eval] RUN_ASR=0, skip ASR."
fi

if [ -f "${asr_jsonl}" ]; then
  ASR_JSONL="${asr_jsonl}" \
  METRICS_CSV="${metrics_csv}" \
  SUMMARY_MD="${summary_md}" \
  RUN_DIR_VALUE="${RUN_DIR}" \
  SOURCE_AUDIO_VALUE="${SOURCE_AUDIO}" \
  TIMBRE_REF_AUDIO_VALUE="${TIMBRE_REF_AUDIO}" \
  "${PYTHON}" - <<'PY'
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

asr_jsonl = Path(os.environ["ASR_JSONL"])
metrics_csv = Path(os.environ["METRICS_CSV"])
summary_md = Path(os.environ["SUMMARY_MD"])

rows = []
for line in asr_jsonl.read_text(encoding="utf-8").splitlines():
    if line.strip():
        rows.append(json.loads(line))

fieldnames = [
    "sample_id",
    "checkpoint_step",
    "eval_variant",
    "cer_tgt",
    "wer_tgt",
    "repeat_score",
    "duration_ratio_tgt_src",
    "content_keep",
    "content_filter_reason",
    "target_audio",
    "asr_tgt_text",
]
with metrics_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})

def fnum(value):
    try:
        return f"{float(value):.4f}"
    except Exception:
        return ""

sorted_rows = sorted(rows, key=lambda item: (str(item.get("checkpoint_step", "")), str(item.get("eval_variant", ""))))
best = None
for row in sorted_rows:
    try:
        score = float(row.get("cer_tgt", 1e9)) + 0.5 * float(row.get("repeat_score", 1e9))
    except Exception:
        continue
    if best is None or score < best[0]:
        best = (score, row)

lines = [
    "# MOSS-CodecVC Fixed Checkpoint Eval",
    "",
    f"run_dir: `{os.environ['RUN_DIR_VALUE']}`",
    f"source_audio: `{os.environ['SOURCE_AUDIO_VALUE']}`",
    f"timbre_ref_audio: `{os.environ['TIMBRE_REF_AUDIO_VALUE']}`",
    "",
]
if best is not None:
    row = best[1]
    lines.extend(
        [
            "## Current Best By CER + 0.5 * Repeat",
            "",
            f"- sample: `{row.get('sample_id')}`",
            f"- checkpoint: `{row.get('checkpoint_step')}`",
            f"- variant: `{row.get('eval_variant')}`",
            f"- CER: `{fnum(row.get('cer_tgt'))}`",
            f"- repeat_score: `{fnum(row.get('repeat_score'))}`",
            f"- target_audio: `{row.get('target_audio')}`",
            "",
        ]
    )
lines.extend(
    [
        "## Metrics",
        "",
        "| sample | checkpoint | variant | CER | repeat | duration | keep | reason | ASR target |",
        "|---|---:|---|---:|---:|---:|---|---|---|",
    ]
)
for row in sorted_rows:
    text = str(row.get("asr_tgt_text") or "")
    if len(text) > 90:
        text = text[:90] + "..."
    text = text.replace("|", "/")
    lines.append(
        "| {sample} | {ckpt} | {variant} | {cer} | {rep} | {dur} | {keep} | {reason} | {text} |".format(
            sample=row.get("sample_id", ""),
            ckpt=row.get("checkpoint_step", ""),
            variant=row.get("eval_variant", ""),
            cer=fnum(row.get("cer_tgt")),
            rep=fnum(row.get("repeat_score")),
            dur=fnum(row.get("duration_ratio_tgt_src")),
            keep=row.get("content_keep", ""),
            reason=row.get("content_filter_reason", ""),
            text=text,
        )
    )
lines.extend(["", f"CSV: `{metrics_csv}`", f"ASR JSONL: `{asr_jsonl}`", ""])
summary_md.write_text("\n".join(lines), encoding="utf-8")
print(f"[ckpt-eval] metrics={metrics_csv}")
print(f"[ckpt-eval] summary={summary_md}")
PY
else
  echo "[ckpt-eval] ASR output missing, summary not generated: ${asr_jsonl}" >&2
fi

echo "[ckpt-eval] done: ${EVAL_DIR}"

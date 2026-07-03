#!/usr/bin/env sh
set -eu

# Run a fixed no-text SeedTTS validation subset for Ver2.5 checkpoints, then
# compute ASR/CER/WER/repeat metrics. Defaults are intentionally hardcoded so
# the script can be run as:
#   MODEL_PATH=/abs/path/to/step-500 sh scripts/004034_run_ver2_5_no_text_validation_eval.sh

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
ROOT=$(CDPATH= cd "${SCRIPT_DIR}/.." && pwd)

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
ASR_PYTHON="${ASR_PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
VALIDATION_JSONL="${VALIDATION_JSONL:-$ROOT/testset/validation/ver2_3_debug/seedtts_vc_ver2_3_benchmark_core128.jsonl}"
MODEL_PATH="${MODEL_PATH:-$ROOT/outputs/lora_runs/ver2_5_debug_5k/ver2_5_s1_from_a_posbias_2k/step-500}"
RUN_SCRIPT="${RUN_SCRIPT:-$ROOT/scripts/003003_run_moss_codecvc_infer.sh}"
VALID_INFER_SCRIPT="${VALID_INFER_SCRIPT:-$ROOT/scripts/004013_run_seedtts_validation_infer.py}"
BUILD_EVAL_SCRIPT="${BUILD_EVAL_SCRIPT:-$ROOT/scripts/004017_build_seedtts_generated_eval_jsonl.py}"
ASR_SCRIPT="${ASR_SCRIPT:-$ROOT/scripts/001017_asr_content_filter.py}"

MAX_CASES="${MAX_CASES:-48}"
PER_CELL="${PER_CELL:-6}"
DEVICE="${DEVICE:-auto}"
ASR_DEVICE="${ASR_DEVICE:-cuda:0}"
OVERWRITE_INFER="${OVERWRITE_INFER:-0}"
RUN_ASR="${RUN_ASR:-1}"

QWEN_ASR_MODEL="${QWEN_ASR_MODEL:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/checkpoint/qwen-asr-1_7b}"
QWEN_ASR_DTYPE="${QWEN_ASR_DTYPE:-bfloat16}"
QWEN_ASR_MAX_BATCH_SIZE="${QWEN_ASR_MAX_BATCH_SIZE:-1}"
QWEN_ASR_MAX_NEW_TOKENS="${QWEN_ASR_MAX_NEW_TOKENS:-256}"

NO_TEXT_ZH_CER_THRESHOLD="${NO_TEXT_ZH_CER_THRESHOLD:-0.35}"
NO_TEXT_EN_WER_THRESHOLD="${NO_TEXT_EN_WER_THRESHOLD:-0.30}"
MAX_REPEAT_SCORE="${MAX_REPEAT_SCORE:-0.30}"

if [ ! -f "$VALIDATION_JSONL" ]; then
  echo "missing validation jsonl: $VALIDATION_JSONL" >&2
  exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
  echo "missing MODEL_PATH directory: $MODEL_PATH" >&2
  exit 1
fi

model_parent=$(basename "$(dirname "$MODEL_PATH")")
model_leaf=$(basename "$MODEL_PATH")
run_id="${RUN_ID:-${model_parent}_${model_leaf}_notext_core${MAX_CASES}}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/testset/outputs/ver2_5_no_text_eval/$run_id}"
MANIFEST_JSONL="${MANIFEST_JSONL:-$OUTPUT_DIR/manifest.jsonl}"
EVAL_INPUT_JSONL="${EVAL_INPUT_JSONL:-$OUTPUT_DIR/${run_id}.generated_eval_input.jsonl}"
ASR_JSONL="${ASR_JSONL:-$OUTPUT_DIR/${run_id}.asr_eval.jsonl}"
METRICS_CSV="${METRICS_CSV:-$OUTPUT_DIR/${run_id}.metrics.csv}"
SUMMARY_MD="${SUMMARY_MD:-$OUTPUT_DIR/SUMMARY.md}"

mkdir -p "$OUTPUT_DIR"

echo "[ver2.5-no-text-eval] model=$MODEL_PATH"
echo "[ver2.5-no-text-eval] validation=$VALIDATION_JSONL"
echo "[ver2.5-no-text-eval] output_dir=$OUTPUT_DIR"
echo "[ver2.5-no-text-eval] max_cases=$MAX_CASES per_cell=$PER_CELL"

infer_args=""
if [ "$OVERWRITE_INFER" = "1" ]; then
  infer_args="$infer_args --overwrite"
fi

"$PYTHON" "$VALID_INFER_SCRIPT" \
  --validation-jsonl "$VALIDATION_JSONL" \
  --model-path "$MODEL_PATH" \
  --run-script "$RUN_SCRIPT" \
  --output-dir "$OUTPUT_DIR" \
  --manifest-jsonl "$MANIFEST_JSONL" \
  --mode no_text \
  --per-cell "$PER_CELL" \
  --max-cases "$MAX_CASES" \
  --device "$DEVICE" \
  $infer_args

"$PYTHON" "$BUILD_EVAL_SCRIPT" \
  --validation-jsonl "$VALIDATION_JSONL" \
  --output-dir "$OUTPUT_DIR" \
  --manifest-jsonl "$MANIFEST_JSONL" \
  --run-id "$run_id" \
  --output-jsonl "$EVAL_INPUT_JSONL" \
  --status "ok,ok_after_rerun,skipped_exists"

if [ "$RUN_ASR" = "1" ]; then
  "$ASR_PYTHON" "$ASR_SCRIPT" \
    --input-jsonl "$EVAL_INPUT_JSONL" \
    --output-jsonl "$ASR_JSONL" \
    --asr-backend qwen_asr \
    --qwen-asr-model "$QWEN_ASR_MODEL" \
    --qwen-asr-dtype "$QWEN_ASR_DTYPE" \
    --qwen-asr-max-batch-size "$QWEN_ASR_MAX_BATCH_SIZE" \
    --qwen-asr-max-new-tokens "$QWEN_ASR_MAX_NEW_TOKENS" \
    --device "$ASR_DEVICE" \
    --content-reference-mode source \
    --skip-source-asr \
    --no-text-zh-cer-threshold "$NO_TEXT_ZH_CER_THRESHOLD" \
    --no-text-en-wer-threshold "$NO_TEXT_EN_WER_THRESHOLD" \
    --max-repeat-score "$MAX_REPEAT_SCORE" \
    --overwrite
else
  echo "[ver2.5-no-text-eval] RUN_ASR=0, skip ASR metrics."
fi

if [ -f "$ASR_JSONL" ]; then
  ASR_JSONL="$ASR_JSONL" METRICS_CSV="$METRICS_CSV" SUMMARY_MD="$SUMMARY_MD" RUN_ID="$run_id" MODEL_PATH_VALUE="$MODEL_PATH" "$PYTHON" - <<'PY'
from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

asr_jsonl = Path(os.environ["ASR_JSONL"])
metrics_csv = Path(os.environ["METRICS_CSV"])
summary_md = Path(os.environ["SUMMARY_MD"])
rows = [json.loads(line) for line in asr_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]

fields = [
    "sample_id",
    "case_id",
    "cell",
    "source_lang",
    "ref_lang",
    "cer_tgt",
    "wer_tgt",
    "repeat_score",
    "duration_ratio_tgt_src",
    "content_keep",
    "content_filter_reason",
    "target_audio",
    "asr_tgt_text",
    "content_ref_text",
]
with metrics_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fields})

def finite(value):
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None

def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None

by_cell = defaultdict(list)
for row in rows:
    by_cell[str(row.get("cell") or "unknown")].append(row)

def fmt(value):
    return "" if value is None else f"{value:.4f}"

lines = [
    "# Ver2.5 No-Text Validation Eval",
    "",
    f"run_id: `{os.environ['RUN_ID']}`",
    f"model: `{os.environ['MODEL_PATH_VALUE']}`",
    f"rows: `{len(rows)}`",
    "",
    "## Overall",
    "",
]
lines.append(f"- mean CER: `{fmt(mean(finite(row.get('cer_tgt')) for row in rows))}`")
lines.append(f"- mean WER: `{fmt(mean(finite(row.get('wer_tgt')) for row in rows))}`")
lines.append(f"- mean repeat: `{fmt(mean(finite(row.get('repeat_score')) for row in rows))}`")
lines.append(f"- mean duration ratio: `{fmt(mean(finite(row.get('duration_ratio_tgt_src')) for row in rows))}`")
lines.append(f"- keep count: `{sum(1 for row in rows if row.get('content_keep') is True)}/{len(rows)}`")
lines.append("")
lines.extend(["## By Cell", "", "| cell | n | CER | WER | repeat | duration | keep |", "|---|---:|---:|---:|---:|---:|---:|"])
for cell, group in sorted(by_cell.items()):
    lines.append(
        "| {cell} | {n} | {cer} | {wer} | {rep} | {dur} | {keep} |".format(
            cell=cell,
            n=len(group),
            cer=fmt(mean(finite(row.get("cer_tgt")) for row in group)),
            wer=fmt(mean(finite(row.get("wer_tgt")) for row in group)),
            rep=fmt(mean(finite(row.get("repeat_score")) for row in group)),
            dur=fmt(mean(finite(row.get("duration_ratio_tgt_src")) for row in group)),
            keep=sum(1 for row in group if row.get("content_keep") is True),
        )
    )
reason_counts = Counter(str(row.get("content_filter_reason") or "keep") for row in rows)
lines.extend(["", "## Filter Reasons", ""])
for reason, count in reason_counts.most_common():
    lines.append(f"- `{reason}`: {count}")
lines.extend(["", f"CSV: `{metrics_csv}`", f"ASR JSONL: `{asr_jsonl}`", ""])
summary_md.write_text("\n".join(lines), encoding="utf-8")
print(f"[ver2.5-no-text-eval] metrics={metrics_csv}")
print(f"[ver2.5-no-text-eval] summary={summary_md}")
PY
fi

echo "[ver2.5-no-text-eval] done: $OUTPUT_DIR"

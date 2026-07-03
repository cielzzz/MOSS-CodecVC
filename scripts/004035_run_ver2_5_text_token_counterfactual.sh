#!/usr/bin/env sh
set -eu

# P0-A text-token memory counterfactual test.
# Same source/ref/model, two different SOURCE_CONTENT_TEXT values:
#   1. correct transcript aligned with source wav
#   2. deliberately wrong transcript
#
# If generated content follows the wrong transcript, P0-A text-token memory is
# actively controlling no-text content rather than being a no-op.

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
RUN_SCRIPT="${RUN_SCRIPT:-$PROJECT_ROOT/scripts/003003_run_moss_codecvc_infer.sh}"
ASR_SCRIPT="${ASR_SCRIPT:-$PROJECT_ROOT/scripts/001017_asr_content_filter.py}"
PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
ASR_PYTHON="${ASR_PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/outputs/lora_runs/ver2_5_debug_5k/ver2_5_p0a_text_token_memory_2k/step-1500}"
SOURCE_AUDIO="${SOURCE_AUDIO:-$PROJECT_ROOT/testset/source/media.wav}"
TIMBRE_REF_AUDIO="${TIMBRE_REF_AUDIO:-$PROJECT_ROOT/testset/timbre_ref/zh_a_000001_timbre.wav}"

CORRECT_SOURCE_CONTENT_TEXT="${CORRECT_SOURCE_CONTENT_TEXT:-你就一直没动过心？那倒也不是，我又不是圣人，所以说啊，你也是活生生的有血有肉的人啊，不是什么特殊材料制成的，很简单。}"
WRONG_SOURCE_CONTENT_TEXT="${WRONG_SOURCE_CONTENT_TEXT:-今天天气很好，我们一起去公园散步，然后买一杯咖啡，晚上再回家看电影。}"

RUN_ID="${RUN_ID:-ver2_5_p0a_step1500_text_token_counterfactual_media}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/testset/outputs/ver2_5_counterfactual/$RUN_ID}"
DEVICE="${DEVICE:-auto}"
ASR_DEVICE="${ASR_DEVICE:-cuda:1}"
RUN_ASR="${RUN_ASR:-1}"

QWEN_ASR_MODEL="${QWEN_ASR_MODEL:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/checkpoint/qwen-asr-1_7b}"
QWEN_ASR_DTYPE="${QWEN_ASR_DTYPE:-bfloat16}"
QWEN_ASR_MAX_BATCH_SIZE="${QWEN_ASR_MAX_BATCH_SIZE:-1}"
QWEN_ASR_MAX_NEW_TOKENS="${QWEN_ASR_MAX_NEW_TOKENS:-256}"

mkdir -p "$OUTPUT_DIR"

run_one() {
  variant="$1"
  transcript="$2"
  output_wav="$OUTPUT_DIR/${variant}.wav"
  echo "[counterfactual] variant=$variant"
  echo "[counterfactual] transcript=$transcript"
  MODE=no_text \
  MODEL_PATH="$MODEL_PATH" \
  SOURCE_AUDIO="$SOURCE_AUDIO" \
  TIMBRE_REF_AUDIO="$TIMBRE_REF_AUDIO" \
  SOURCE_CONTENT_TEXT="$transcript" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  OUTPUT_WAV="$output_wav" \
  DEVICE="$DEVICE" \
  DEBUG_GENERATION_STRUCTURE=1 \
  NO_TEXT_MAX_TOKEN_MARGIN=0 \
  NO_TEXT_AUDIO_TEMPERATURE=1.0 \
  NO_TEXT_AUDIO_TOP_P=1.0 \
  NO_TEXT_AUDIO_TOP_K=1 \
  NO_TEXT_AUDIO_REPETITION_PENALTY=1.0 \
  sh "$RUN_SCRIPT"
}

run_one "correct_transcript" "$CORRECT_SOURCE_CONTENT_TEXT"
run_one "wrong_transcript" "$WRONG_SOURCE_CONTENT_TEXT"

ASR_INPUT_JSONL="$OUTPUT_DIR/asr_input.jsonl"
ASR_OUTPUT_JSONL="$OUTPUT_DIR/asr_eval.jsonl"
SUMMARY_MD="$OUTPUT_DIR/SUMMARY.md"

"$PYTHON" - "$ASR_INPUT_JSONL" "$SOURCE_AUDIO" "$OUTPUT_DIR" "$CORRECT_SOURCE_CONTENT_TEXT" "$WRONG_SOURCE_CONTENT_TEXT" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
source_audio = sys.argv[2]
output_dir = Path(sys.argv[3])
correct = sys.argv[4]
wrong = sys.argv[5]
rows = [
    {
        "sample_id": "correct_transcript",
        "mode": "no_text",
        "moss_codecvc_mode": "no_text",
        "language": "Chinese",
        "source_audio": source_audio,
        "target_audio": str(output_dir / "correct_transcript.wav"),
        "text": "<NO_TEXT>",
        "content_ref_text": correct,
        "content_ref_text_source": "manual_correct_source_transcript",
        "content_asr_backend": "manual",
        "content_asr_model": "",
    },
    {
        "sample_id": "wrong_transcript",
        "mode": "no_text",
        "moss_codecvc_mode": "no_text",
        "language": "Chinese",
        "source_audio": source_audio,
        "target_audio": str(output_dir / "wrong_transcript.wav"),
        "text": "<NO_TEXT>",
        "content_ref_text": wrong,
        "content_ref_text_source": "manual_counterfactual_wrong_transcript",
        "content_asr_backend": "manual",
        "content_asr_model": "",
    },
]
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"[counterfactual] wrote {out}")
PY

if [ "$RUN_ASR" = "1" ]; then
  "$ASR_PYTHON" "$ASR_SCRIPT" \
    --input-jsonl "$ASR_INPUT_JSONL" \
    --output-jsonl "$ASR_OUTPUT_JSONL" \
    --asr-backend qwen_asr \
    --qwen-asr-model "$QWEN_ASR_MODEL" \
    --qwen-asr-dtype "$QWEN_ASR_DTYPE" \
    --qwen-asr-max-batch-size "$QWEN_ASR_MAX_BATCH_SIZE" \
    --qwen-asr-max-new-tokens "$QWEN_ASR_MAX_NEW_TOKENS" \
    --device "$ASR_DEVICE" \
    --content-reference-mode text \
    --skip-source-asr \
    --disable-duration-ratio-check \
    --zh-cer-threshold 1.0 \
    --overwrite
fi

if [ -f "$ASR_OUTPUT_JSONL" ]; then
  "$PYTHON" - "$ASR_OUTPUT_JSONL" "$SUMMARY_MD" "$RUN_ID" "$MODEL_PATH" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
summary = Path(sys.argv[2])
lines = [
    "# Ver2.5 Text Token Memory Counterfactual",
    "",
    f"run_id: `{sys.argv[3]}`",
    f"model: `{sys.argv[4]}`",
    "",
    "| variant | CER vs injected transcript | repeat | ASR target | injected transcript |",
    "|---|---:|---:|---|---|",
]
for row in rows:
    lines.append(
        "| {variant} | {cer:.4f} | {repeat:.4f} | {asr} | {ref} |".format(
            variant=row.get("sample_id", ""),
            cer=float(row.get("cer_tgt", 0.0)),
            repeat=float(row.get("repeat_score", 0.0)),
            asr=str(row.get("asr_tgt_text") or "").replace("|", "\\|"),
            ref=str(row.get("content_ref_text") or "").replace("|", "\\|"),
        )
    )
lines.append("")
summary.write_text("\n".join(lines), encoding="utf-8")
print(f"[counterfactual] summary={summary}")
PY
fi

echo "[counterfactual] done output_dir=$OUTPUT_DIR"

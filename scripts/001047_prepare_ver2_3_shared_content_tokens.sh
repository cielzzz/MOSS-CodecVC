#!/usr/bin/env bash
set -euo pipefail

# Prepare Ver2.3 shared multilingual TextCTC tokens for mixed text/no_text training.
# This fixes the unsafe Ver2.1 situation where no_text and text JSONLs used
# separately built char CTC vocabs.

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
cd "$PROJECT_ROOT"

PYTHON_MAIN="${PYTHON_MAIN:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"

NO_TEXT_DATASET_NAME="${NO_TEXT_DATASET_NAME:-zh45w_en22w_no_text}"
TEXT_DATASET_NAME="${TEXT_DATASET_NAME:-zh3w_en3w_text_prosody_independent_timbre}"

NO_TEXT_INPUT="${NO_TEXT_INPUT:-$PROJECT_ROOT/trainset/$NO_TEXT_DATASET_NAME/sft/moss_codecvc_sft.$NO_TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_asr_filter.with_content_tokens.with_hubert.jsonl}"
TEXT_INPUT="${TEXT_INPUT:-$PROJECT_ROOT/trainset/$TEXT_DATASET_NAME/sft/moss_codecvc_sft.$TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_target_asr.with_content_tokens.with_target_hubert.jsonl}"

TOKENIZER_ROOT="${TOKENIZER_ROOT:-$PROJECT_ROOT/trainset/shared_content_tokenizers}"
TOKENIZER_PREFIX="${TOKENIZER_PREFIX:-$TOKENIZER_ROOT/spm_multilingual_byte_fallback_v1}"
TOKENIZER_ID="${TOKENIZER_ID:-spm_multilingual_byte_fallback_v1}"

NO_TEXT_OUTPUT="${NO_TEXT_OUTPUT:-$PROJECT_ROOT/trainset/$NO_TEXT_DATASET_NAME/sft/moss_codecvc_sft.$NO_TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_asr_filter.with_hubert.with_spm_content_tokens.jsonl}"
TEXT_OUTPUT="${TEXT_OUTPUT:-$PROJECT_ROOT/trainset/$TEXT_DATASET_NAME/sft/moss_codecvc_sft.$TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_target_asr.with_content_tokens.with_target_hubert.with_spm_content_tokens.jsonl}"

VOCAB_SIZE="${VOCAB_SIZE:-8000}"
MODEL_TYPE="${MODEL_TYPE:-unigram}"
CHARACTER_COVERAGE="${CHARACTER_COVERAGE:-0.9995}"
BYTE_FALLBACK="${BYTE_FALLBACK:-1}"
LOWERCASE_LATIN="${LOWERCASE_LATIN:-0}"
STRIP_EXTRA_WHITESPACE="${STRIP_EXTRA_WHITESPACE:-1}"
REQUIRE_CONTENT_KEEP="${REQUIRE_CONTENT_KEEP:-1}"
MAX_ROWS_PER_SOURCE="${MAX_ROWS_PER_SOURCE:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100000}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

bool_flag() {
  local value="$1"
  local positive="$2"
  local negative="$3"
  case "$value" in
    1|true|TRUE|yes|YES|y|Y) printf "%s" "$positive" ;;
    *) printf "%s" "$negative" ;;
  esac
}

overwrite_flag=()
if [[ "$OVERWRITE" == "1" ]]; then
  overwrite_flag=(--overwrite)
fi
byte_fallback_flag=("$(bool_flag "$BYTE_FALLBACK" "--byte-fallback" "--no-byte-fallback")")
lowercase_flag=("$(bool_flag "$LOWERCASE_LATIN" "--lowercase-latin" "--no-lowercase-latin")")
whitespace_flag=("$(bool_flag "$STRIP_EXTRA_WHITESPACE" "--strip-extra-whitespace" "--no-strip-extra-whitespace")")
keep_flag=("$(bool_flag "$REQUIRE_CONTENT_KEEP" "--require-content-keep" "--no-require-content-keep")")

echo "[ver2.3-content] project=$PROJECT_ROOT"
echo "[ver2.3-content] no_text_input=$NO_TEXT_INPUT"
echo "[ver2.3-content] text_input=$TEXT_INPUT"
echo "[ver2.3-content] tokenizer_prefix=$TOKENIZER_PREFIX"
echo "[ver2.3-content] no_text_output=$NO_TEXT_OUTPUT"
echo "[ver2.3-content] text_output=$TEXT_OUTPUT"

test -f "$NO_TEXT_INPUT"
test -f "$TEXT_INPUT"
mkdir -p "$TOKENIZER_ROOT"

if [[ "$SKIP_EXISTING" == "1" && -f "$TOKENIZER_PREFIX.model" && "$OVERWRITE" != "1" ]]; then
  echo "[ver2.3-content] reuse tokenizer: $TOKENIZER_PREFIX.model"
else
  "$PYTHON_MAIN" scripts/001045_train_multilingual_content_tokenizer.py \
    --input-jsonl "$NO_TEXT_INPUT" \
    --input-jsonl "$TEXT_INPUT" \
    --output-prefix "$TOKENIZER_PREFIX" \
    --tokenizer-id "$TOKENIZER_ID" \
    --vocab-size "$VOCAB_SIZE" \
    --model-type "$MODEL_TYPE" \
    --character-coverage "$CHARACTER_COVERAGE" \
    "${byte_fallback_flag[@]}" \
    "${lowercase_flag[@]}" \
    "${whitespace_flag[@]}" \
    "${keep_flag[@]}" \
    --max-rows-per-source "$MAX_ROWS_PER_SOURCE" \
    --progress-every "$PROGRESS_EVERY" \
    "${overwrite_flag[@]}"
fi

if [[ "$SKIP_EXISTING" == "1" && -f "$NO_TEXT_OUTPUT" && "$OVERWRITE" != "1" ]]; then
  echo "[ver2.3-content] reuse no_text output: $NO_TEXT_OUTPUT"
else
  "$PYTHON_MAIN" scripts/001046_extract_multilingual_content_tokens.py \
    --input-jsonl "$NO_TEXT_INPUT" \
    --output-jsonl "$NO_TEXT_OUTPUT" \
    --spm-model "$TOKENIZER_PREFIX.model" \
    --tokenizer-meta "$TOKENIZER_PREFIX.json" \
    --tokenizer-id "$TOKENIZER_ID" \
    "${lowercase_flag[@]}" \
    "${whitespace_flag[@]}" \
    "${keep_flag[@]}" \
    --progress-every "$PROGRESS_EVERY" \
    "${overwrite_flag[@]}"
fi

if [[ "$SKIP_EXISTING" == "1" && -f "$TEXT_OUTPUT" && "$OVERWRITE" != "1" ]]; then
  echo "[ver2.3-content] reuse text output: $TEXT_OUTPUT"
else
  "$PYTHON_MAIN" scripts/001046_extract_multilingual_content_tokens.py \
    --input-jsonl "$TEXT_INPUT" \
    --output-jsonl "$TEXT_OUTPUT" \
    --spm-model "$TOKENIZER_PREFIX.model" \
    --tokenizer-meta "$TOKENIZER_PREFIX.json" \
    --tokenizer-id "$TOKENIZER_ID" \
    "${lowercase_flag[@]}" \
    "${whitespace_flag[@]}" \
    "${keep_flag[@]}" \
    --progress-every "$PROGRESS_EVERY" \
    "${overwrite_flag[@]}"
fi

echo "[ver2.3-content] done"
echo "  TOKENIZER=$TOKENIZER_PREFIX.model"
echo "  NO_TEXT_TRAIN_JSONL=$NO_TEXT_OUTPUT"
echo "  TEXT_TRAIN_JSONL=$TEXT_OUTPUT"

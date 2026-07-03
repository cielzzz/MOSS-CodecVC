#!/bin/sh
# Ver2.8: WavLM-BNF content memory + codec residual + timbre side-only VC experiment.
#
# This keeps the Ver2.3 clean data recipe, drops auxiliary CTC loss, removes
# S2/timbre codec from the AR prompt, and feeds no-text content through
# SourceSemanticMemory with continuous WavLM/BNF features.

set -eu

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
NO_TEXT_DATASET_NAME="${NO_TEXT_DATASET_NAME:-zh45w_en22w_no_text}"
TEXT_DATASET_NAME="${TEXT_DATASET_NAME:-zh3w_en3w_text_prosody_independent_timbre}"
# repeat is the total number of sampled copies. TEXT_REPEAT=5 means the text
# set is used as 1 original copy + 4 extra copies, which balances the current
# Ver2.8 split: 32903 * 5 ~= 165158 no-text train rows.
TEXT_REPEAT="${TEXT_REPEAT:-5}"
BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%d-%H%M%S)}"

NO_TEXT_HUBERT_CLEAN="$ROOT/trainset/$NO_TEXT_DATASET_NAME/sft/moss_codecvc_sft.$NO_TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_asr_filter.with_hubert.with_spm_content_tokens.ctc_clean.jsonl"
NO_TEXT_WAVLM_CLEAN="${NO_TEXT_WAVLM_CLEAN:-$ROOT/trainset/$NO_TEXT_DATASET_NAME/sft/moss_codecvc_sft.$NO_TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_asr_filter.with_wavlm_bnf.with_spm_content_tokens.ctc_clean.jsonl}"
NO_TEXT_CLEAN="${NO_TEXT_CLEAN:-$NO_TEXT_WAVLM_CLEAN}"
TEXT_HUBERT_CLEAN="$ROOT/trainset/$TEXT_DATASET_NAME/sft/moss_codecvc_sft.$TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_target_asr.with_content_tokens.with_target_hubert.with_spm_content_tokens.ctc_clean.jsonl"
TEXT_WAVLM_CLEAN="${TEXT_WAVLM_CLEAN:-$ROOT/trainset/$TEXT_DATASET_NAME/sft/moss_codecvc_sft.$TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_target_asr.with_content_tokens.with_target_wavlm_bnf.with_spm_content_tokens.ctc_clean.jsonl}"
TEXT_CLEAN="${TEXT_CLEAN:-$TEXT_WAVLM_CLEAN}"
SUBMIT_SCRIPT="$ROOT/scripts/002004_submit_ver2_lora_68w_h200_qz.sh"
PREPARED_DIR="${PREPARED_DIR:-$ROOT/trainset/ver2_8_prepared}"
AUTO_PREPARE_VER2_8="${AUTO_PREPARE_VER2_8:-1}"
PREPARED_NO_TEXT_TRAIN="$PREPARED_DIR/no_text.train.jsonl"
PREPARED_NO_TEXT_VALID="$PREPARED_DIR/no_text.valid.jsonl"
PREPARED_TEXT_TRAIN="$PREPARED_DIR/text.train.jsonl"
PREPARED_TEXT_VALID="$PREPARED_DIR/text.valid.jsonl"

print_wavlm_prepare_help() {
  echo "Prepare WavLM manifests first. Recommended H200/QZ entry:" >&2
  echo "  sh scripts/002015_submit_ver2_8_wavlm_bnf_data_h200_qz.sh" >&2
  echo "Local/manual fallback commands:" >&2
  echo "  $PY scripts/001020_extract_hubert_semantic_features.py --input-jsonl $NO_TEXT_HUBERT_CLEAN --output-jsonl $NO_TEXT_WAVLM_CLEAN --feature-root $ROOT/trainset/$NO_TEXT_DATASET_NAME/semantic_features/wavlm_bnf --extractor wavlm --source both --layer 9 --save-dtype float16 --overwrite" >&2
  echo "  $PY scripts/001020_extract_hubert_semantic_features.py --input-jsonl $TEXT_HUBERT_CLEAN --output-jsonl $TEXT_WAVLM_CLEAN --feature-root $ROOT/trainset/$TEXT_DATASET_NAME/semantic_features/wavlm_bnf --extractor wavlm --source target --layer 9 --save-dtype float16 --overwrite" >&2
}

if [ ! -f "$PREPARED_NO_TEXT_TRAIN" ] || [ ! -f "$PREPARED_TEXT_TRAIN" ] || [ ! -f "$PREPARED_NO_TEXT_VALID" ] || [ ! -f "$PREPARED_TEXT_VALID" ]; then
  if [ "$AUTO_PREPARE_VER2_8" = "1" ]; then
    if [ ! -f "$NO_TEXT_CLEAN" ] || [ ! -f "$TEXT_CLEAN" ]; then
      echo "ERROR: prepared Ver2.8 data is absent and WavLM manifests are missing." >&2
      echo "  no_text=$NO_TEXT_CLEAN" >&2
      echo "  text=$TEXT_CLEAN" >&2
      print_wavlm_prepare_help
      exit 1
    fi
    echo "[ver2.8-submit] prepared data missing; running prepare into $PREPARED_DIR"
    "$PY" "$ROOT/scripts/002013_prepare_ver2_8_train_valid.py" \
      --no-text-jsonl "$NO_TEXT_CLEAN" \
      --text-jsonl "$TEXT_CLEAN" \
      --output-dir "$PREPARED_DIR" \
      --semantic-kind wavlm \
      --text-repeat "$TEXT_REPEAT" \
      --overwrite
  else
    echo "ERROR: prepared Ver2.8 data is missing: $PREPARED_DIR" >&2
    echo "Run scripts/002013_prepare_ver2_8_train_valid.py or set AUTO_PREPARE_VER2_8=1." >&2
    exit 1
  fi
fi

for required_path in "$SUBMIT_SCRIPT" "$PREPARED_NO_TEXT_TRAIN" "$PREPARED_TEXT_TRAIN" "$PREPARED_NO_TEXT_VALID" "$PREPARED_TEXT_VALID"; do
  if [ ! -f "$required_path" ]; then
    echo "ERROR: required file not found: $required_path" >&2
    exit 1
  fi
done

export NO_TEXT_TRAIN_JSONL="${NO_TEXT_TRAIN_JSONL:-$PREPARED_NO_TEXT_TRAIN}"
export TEXT_TRAIN_JSONL="${TEXT_TRAIN_JSONL:-$PREPARED_TEXT_TRAIN}"
export PY
export TEXT_REPEAT
export TRAIN_JSONL_SPEC="${TRAIN_JSONL_SPEC:-$NO_TEXT_TRAIN_JSONL::repeat=1,$TEXT_TRAIN_JSONL::repeat=$TEXT_REPEAT}"
export OUT_DIR="${OUT_DIR:-$ROOT/outputs/lora_runs/ver2_8_sideonly_wavlmbnf_codecres_textrep${TEXT_REPEAT}_lora_r16_a32_gbs64}"
export JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codecvc-ver2-8-sideonly-wavlmbnf-codecres-textrep${TEXT_REPEAT}}"
export BATCH_ID

export NUM_EPOCHS="${NUM_EPOCHS:-6}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export LOGGING_STEPS="${LOGGING_STEPS:-20}"

export TIMBRE_SIDE_ONLY="${TIMBRE_SIDE_ONLY:-1}"
export TIMBRE_ADAPTER_INIT_GATE="${TIMBRE_ADAPTER_INIT_GATE:--2.0}"
export TIMBRE_ADAPTER_GATE_LR_MULTIPLIER="${TIMBRE_ADAPTER_GATE_LR_MULTIPLIER:-10.0}"

export CONTENT_CTC_WEIGHT="${CONTENT_CTC_WEIGHT:-0}"
export CONTENT_CTC_VOCAB_SIZE="${CONTENT_CTC_VOCAB_SIZE:-0}"
export SEMANTIC_LOSS_WEIGHT="${SEMANTIC_LOSS_WEIGHT:-0.03}"
export SEMANTIC_MODE="${SEMANTIC_MODE:-continuous}"
export SEMANTIC_SOURCE="${SEMANTIC_SOURCE:-mode_aware}"
export SEMANTIC_FEATURE_DIM="${SEMANTIC_FEATURE_DIM:-768}"
export SEMANTIC_FEATURE_LOSS_TYPE="${SEMANTIC_FEATURE_LOSS_TYPE:-cosine}"
export LAMBDA_CONTENT="${LAMBDA_CONTENT:-0}"

export ENABLE_SOURCE_SEMANTIC_MEMORY="${ENABLE_SOURCE_SEMANTIC_MEMORY:-1}"
export SOURCE_CONTENT_MEMORY_TYPE="${SOURCE_CONTENT_MEMORY_TYPE:-wavlm_bnf_continuous}"
export SOURCE_CONTENT_VOCAB_SIZE="${SOURCE_CONTENT_VOCAB_SIZE:-0}"
export SOURCE_CONTENT_PADDING_ID="${SOURCE_CONTENT_PADDING_ID:-0}"
export SOURCE_SEMANTIC_FEATURE_DIM="${SOURCE_SEMANTIC_FEATURE_DIM:-768}"
export SOURCE_SEMANTIC_NO_TEXT_GATE="${SOURCE_SEMANTIC_NO_TEXT_GATE:-1.0}"
export SOURCE_SEMANTIC_TEXT_GATE="${SOURCE_SEMANTIC_TEXT_GATE:-0.0}"
export SOURCE_SEMANTIC_PROGRESS_WEIGHT="${SOURCE_SEMANTIC_PROGRESS_WEIGHT:-0.02}"
export SOURCE_SEMANTIC_INIT_GATE="${SOURCE_SEMANTIC_INIT_GATE:--1.0}"
export SOURCE_SEMANTIC_LR_MULTIPLIER="${SOURCE_SEMANTIC_LR_MULTIPLIER:-1.0}"
export SOURCE_SEMANTIC_GATE_LR_MULTIPLIER="${SOURCE_SEMANTIC_GATE_LR_MULTIPLIER:-10.0}"
export SOURCE_SEMANTIC_POSITION_SCALE="${SOURCE_SEMANTIC_POSITION_SCALE:-0.10}"
export SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH="${SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH:-2.0}"
export SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH="${SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH:-0.25}"
export SOURCE_CONTENT_CODEC_BOTTLENECK_DIM="${SOURCE_CONTENT_CODEC_BOTTLENECK_DIM:-256}"
export SOURCE_CONTENT_CODEC_CODEBOOKS="${SOURCE_CONTENT_CODEC_CODEBOOKS:-first_4}"
export SOURCE_CODEC_RESIDUAL_MEMORY_WEIGHT="${SOURCE_CODEC_RESIDUAL_MEMORY_WEIGHT:-0.20}"
export SOURCE_CODEC_RESIDUAL_MEMORY_DETACH="${SOURCE_CODEC_RESIDUAL_MEMORY_DETACH:-0}"
export SOURCE_PROSODY_NO_TEXT_GATE="${SOURCE_PROSODY_NO_TEXT_GATE:-1.0}"
export SOURCE_PROSODY_TEXT_GATE="${SOURCE_PROSODY_TEXT_GATE:-0.0}"

export REF_CONTENT_SUPPRESSION_WEIGHT="${REF_CONTENT_SUPPRESSION_WEIGHT:-0.01}"
export REF_CONTENT_SUPPRESSION_MARGIN="${REF_CONTENT_SUPPRESSION_MARGIN:-0.10}"
export REF_CONTENT_SUPPRESSION_SOURCE="${REF_CONTENT_SUPPRESSION_SOURCE:-auto}"
export REF_CONTENT_SUPPRESSION_DETACH_REF="${REF_CONTENT_SUPPRESSION_DETACH_REF:-1}"

export LAMBDA_PROSODY="${LAMBDA_PROSODY:-0.05}"
export PROGRESS_LOSS_WEIGHT="${PROGRESS_LOSS_WEIGHT:-0.02}"
export STOP_LOSS_WEIGHT="${STOP_LOSS_WEIGHT:-0.05}"
export TARGET_SPK_WEIGHT="${TARGET_SPK_WEIGHT:-0.20}"
export SOURCE_SUPPRESS_WEIGHT="${SOURCE_SUPPRESS_WEIGHT:-0.10}"
export SPEAKER_LOSS_WARMUP_WEIGHT="${SPEAKER_LOSS_WARMUP_WEIGHT:-0.05}"

if [ -f "$ROOT/testset/validation/ver2_3_debug/moss_codecvc_ver2_3_loss_valid_160.jsonl" ]; then
  export EVAL_JSONL_SPEC="${EVAL_JSONL_SPEC:-$PREPARED_NO_TEXT_VALID::repeat=1,$PREPARED_TEXT_VALID::repeat=1}"
  export EVAL_STEPS="${EVAL_STEPS:-1000}"
fi

echo "[ver2.8-submit] TRAIN_JSONL_SPEC=$TRAIN_JSONL_SPEC"
echo "[ver2.8-submit] EVAL_JSONL_SPEC=${EVAL_JSONL_SPEC:-}"
echo "[ver2.8-submit] OUT_DIR=$OUT_DIR"
echo "[ver2.8-submit] CTC=$CONTENT_CTC_WEIGHT timbre_side_only=$TIMBRE_SIDE_ONLY source_content_memory=$SOURCE_CONTENT_MEMORY_TYPE"
echo "[ver2.8-submit] semantic_loss=$SEMANTIC_LOSS_WEIGHT mode=$SEMANTIC_MODE source=$SEMANTIC_SOURCE feature_dim=$SEMANTIC_FEATURE_DIM"
echo "[ver2.8-submit] source_semantic_feature_dim=$SOURCE_SEMANTIC_FEATURE_DIM codec_residual_weight=$SOURCE_CODEC_RESIDUAL_MEMORY_WEIGHT"
echo "[ver2.8-submit] source_prosody_gates no_text=$SOURCE_PROSODY_NO_TEXT_GATE text=$SOURCE_PROSODY_TEXT_GATE"
echo "[ver2.8-submit] ref_content_suppression=$REF_CONTENT_SUPPRESSION_WEIGHT margin=$REF_CONTENT_SUPPRESSION_MARGIN"
if [ -f "$ROOT/scripts/002017_summarize_ver2_8_data.py" ]; then
  "$PY" "$ROOT/scripts/002017_summarize_ver2_8_data.py" \
    --prepared-dir "$PREPARED_DIR" \
    --no-text-wavlm-jsonl "$NO_TEXT_CLEAN" \
    --text-wavlm-jsonl "$TEXT_CLEAN" \
    --text-repeat "$TEXT_REPEAT" \
    --output-json "$PREPARED_DIR/ver2_8_data_summary_for_submit.json"
fi

exec sh "$SUBMIT_SCRIPT" "$@"

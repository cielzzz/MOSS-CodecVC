#!/bin/sh
# Batch-28 path X dry-run wrapper:
# ver2.3 prompt sequence ([text?, C_src, C_ref]) + BNF content cross-attn.
# Defaults to DRY_RUN=1; set DRY_RUN=0 only after explicit user confirmation.

set -eu

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
BATCH_ID="${BATCH_ID:-ver23_content_side_dryrun_$(date -u +%Y%m%d_%H%M%S)}"
QZ_RECORD_ROOT="${QZ_RECORD_ROOT:-$ROOT/trainset/qz_jobs/$BATCH_ID}"

TRAINSET_DIR="${TRAINSET_DIR:-$ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709}"
NO_TEXT_TRAIN_JSONL="${NO_TEXT_TRAIN_JSONL:-$TRAINSET_DIR/no_text.train.jsonl}"
TEXT_TRAIN_JSONL="${TEXT_TRAIN_JSONL:-$TRAINSET_DIR/text.train.jsonl}"
TEXT_REPEAT="${TEXT_REPEAT:-10}"
TRAIN_JSONL_SPEC="${TRAIN_JSONL_SPEC:-$NO_TEXT_TRAIN_JSONL::repeat=1,$TEXT_TRAIN_JSONL::repeat=$TEXT_REPEAT}"

JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-ver23-content-side-3k}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/lora_runs/ver23_content_side_3k_olddata_textrep${TEXT_REPEAT}_$BATCH_ID}"

export ROOT
export BATCH_ID
export QZ_RECORD_ROOT
export NO_TEXT_TRAIN_JSONL
export TEXT_TRAIN_JSONL
export TEXT_REPEAT
export TRAIN_JSONL_SPEC
export JOB_NAME_PREFIX
export OUT_DIR

export DRY_RUN="${DRY_RUN:-1}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export EVAL_STEPS="${EVAL_STEPS:-500}"
export EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant_with_warmup}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export GPU_COUNT="${GPU_COUNT:-8}"
export POST_TRAIN_QUICK_EVAL="${POST_TRAIN_QUICK_EVAL:-0}"

# Path X keeps C_ref in the AR prompt and removes all speaker-side/timbre-memory injections.
export USE_TIMBRE_MEMORY=0
export TIMBRE_MEMORY_TOKENS=0
export TIMBRE_ADAPTER_LAYERS=""
export TIMBRE_SIDE_ONLY=0
export REF_PROMPT_CODEC_PERMUTATION=0
export USE_PERTURBED_SOURCE_PROMPT=0
export ENABLE_SPEAKER_SIDE_PATHWAY=0
export ENABLE_SPEAKER_CROSS_ATTN=0
export ENABLE_SOURCE_SEMANTIC_MEMORY=0
export SOURCE_SUPPRESS_WEIGHT=0.0
export TARGET_SPK_WEIGHT=0.0
export SPEAKER_INFONCE_WEIGHT=0.0
export SPEAKER_CONDITION_DROPOUT=0.0
export REF_AUDIO_CFG_DROPOUT="${REF_AUDIO_CFG_DROPOUT:-0.0}"
export REF_CONTENT_SUPPRESSION_WEIGHT=0.0
export REF_SPEAKER_PROMPT_TOKENS=0
export REF_SPEAKER_ADALN_WEIGHT=0.0

# Keep the proven content-stability pieces.
export TARGET_FRONT_CE_WEIGHT=4.0
export TARGET_FRONT_CE_SECONDS=0.75
export TARGET_FRONT_CE_FRAME_RATE=12.5
export PROGRESS_LOSS_WEIGHT=0.10
export STOP_LOSS_WEIGHT=0.20
export PROGRESS_NUM_BINS=32

# Disable unrelated auxiliary losses for a clean single-path content-side test.
export LAMBDA_ROUTE=0.0
export LAMBDA_PROSODY=0.0
export LAMBDA_CONTENT=0.0
export CONTENT_CTC_WEIGHT="${CONTENT_CTC_WEIGHT:-0.0}"
export CONTENT_TOKEN_WEIGHT=0.0
export CONTENT_SOURCE_CODEC_WEIGHT=0.0
export SEMANTIC_LOSS_WEIGHT=0.0

# New Batch-28 BNF content side pathway.
export ENABLE_CONTENT_CROSS_ATTN=1
export CONTENT_CROSS_ATTN_LAYERS="${CONTENT_CROSS_ATTN_LAYERS:-all}"
export CONTENT_CROSS_ATTN_FEATURE_DIM="${CONTENT_CROSS_ATTN_FEATURE_DIM:-768}"
export CONTENT_CROSS_ATTN_GATE_INIT="${CONTENT_CROSS_ATTN_GATE_INIT:--0.5}"
export CONTENT_CROSS_ATTN_OUTPUT_SCALE="${CONTENT_CROSS_ATTN_OUTPUT_SCALE:-0.3}"
export CONTENT_CROSS_ATTN_DROPOUT="${CONTENT_CROSS_ATTN_DROPOUT:-0.0}"
export CONTENT_ENCODER_HIDDEN_SIZE="${CONTENT_ENCODER_HIDDEN_SIZE:-0}"
export CONTENT_ENCODER_LAYERS="${CONTENT_ENCODER_LAYERS:-2}"
export CONTENT_ENCODER_CONV_KERNEL_SIZE="${CONTENT_ENCODER_CONV_KERNEL_SIZE:-7}"
export GUIDED_ATTN_LOSS_WEIGHT="${GUIDED_ATTN_LOSS_WEIGHT:-0.05}"
export GUIDED_ATTN_WARMUP_STEPS="${GUIDED_ATTN_WARMUP_STEPS:-1000}"
export GUIDED_ATTN_BAND_FRAMES="${GUIDED_ATTN_BAND_FRAMES:-3}"
export PHONEME_CLASSIFIER_LOSS_WEIGHT="${PHONEME_CLASSIFIER_LOSS_WEIGHT:-0.02}"
export CONTENT_CROSS_ATTN_LR_MULTIPLIER="${CONTENT_CROSS_ATTN_LR_MULTIPLIER:-1.0}"
export LORA_WARMUP_FREEZE_STEPS="${LORA_WARMUP_FREEZE_STEPS:-0}"
export SOURCE_CONTENT_MEMORY_TYPE=wavlm_bnf_continuous

BASE_ARGS=""
if [ "$DRY_RUN" = "1" ]; then
  BASE_ARGS="--dry-run"
fi
bash "$ROOT/scripts/002004_submit_ver2_lora_68w_h200_qz.sh" $BASE_ARGS

mkdir -p "$QZ_RECORD_ROOT"
CORE_JSON="$QZ_RECORD_ROOT/train_args_dry_run_core.json"
python - <<'PY'
import json
import os
from pathlib import Path

keys = [
    "BATCH_ID",
    "JOB_NAME_PREFIX",
    "TRAIN_JSONL_SPEC",
    "NO_TEXT_TRAIN_JSONL",
    "TEXT_TRAIN_JSONL",
    "TEXT_REPEAT",
    "OUT_DIR",
    "MAX_TRAIN_STEPS",
    "SAVE_STEPS",
    "EVAL_STEPS",
    "LEARNING_RATE",
    "LR_SCHEDULER_TYPE",
    "WARMUP_RATIO",
    "PER_DEVICE_BATCH_SIZE",
    "GRADIENT_ACCUMULATION_STEPS",
    "GPU_COUNT",
    "USE_TIMBRE_MEMORY",
    "TIMBRE_MEMORY_TOKENS",
    "TIMBRE_ADAPTER_LAYERS",
    "TIMBRE_SIDE_ONLY",
    "REF_PROMPT_CODEC_PERMUTATION",
    "ENABLE_SPEAKER_SIDE_PATHWAY",
    "ENABLE_SPEAKER_CROSS_ATTN",
    "ENABLE_SOURCE_SEMANTIC_MEMORY",
    "SOURCE_SUPPRESS_WEIGHT",
    "TARGET_SPK_WEIGHT",
    "SPEAKER_INFONCE_WEIGHT",
    "SPEAKER_CONDITION_DROPOUT",
    "REF_AUDIO_CFG_DROPOUT",
    "REF_CONTENT_SUPPRESSION_WEIGHT",
    "TARGET_FRONT_CE_WEIGHT",
    "TARGET_FRONT_CE_SECONDS",
    "PROGRESS_LOSS_WEIGHT",
    "STOP_LOSS_WEIGHT",
    "ENABLE_CONTENT_CROSS_ATTN",
    "CONTENT_CROSS_ATTN_LAYERS",
    "CONTENT_CROSS_ATTN_FEATURE_DIM",
    "CONTENT_CROSS_ATTN_GATE_INIT",
    "CONTENT_CROSS_ATTN_OUTPUT_SCALE",
    "CONTENT_ENCODER_HIDDEN_SIZE",
    "CONTENT_ENCODER_LAYERS",
    "GUIDED_ATTN_LOSS_WEIGHT",
    "GUIDED_ATTN_WARMUP_STEPS",
    "GUIDED_ATTN_BAND_FRAMES",
    "PHONEME_CLASSIFIER_LOSS_WEIGHT",
    "CONTENT_CROSS_ATTN_LR_MULTIPLIER",
    "TIMBRE_ADAPTER_GATE_LR_MULTIPLIER",
    "LORA_WARMUP_FREEZE_STEPS",
    "SOURCE_CONTENT_MEMORY_TYPE",
    "LAMBDA_ROUTE",
    "LAMBDA_PROSODY",
    "LAMBDA_CONTENT",
    "CONTENT_CTC_WEIGHT",
    "SEMANTIC_LOSS_WEIGHT",
]
payload = {key: os.environ.get(key, "") for key in keys}
payload["sequence_structure"] = "[text?, C_src frames, C_ref frames]"
payload["c_ref_in_sequence"] = os.environ.get("TIMBRE_SIDE_ONLY") == "0"
payload["speaker_side_pathway_closed"] = (
    os.environ.get("ENABLE_SPEAKER_SIDE_PATHWAY") == "0"
    and os.environ.get("ENABLE_SPEAKER_CROSS_ATTN") == "0"
)
payload["legacy_timbre_memory_closed"] = (
    os.environ.get("USE_TIMBRE_MEMORY") == "0"
    and os.environ.get("TIMBRE_MEMORY_TOKENS") == "0"
)
payload["content_cross_attn_enabled"] = os.environ.get("ENABLE_CONTENT_CROSS_ATTN") == "1"
path = Path(os.environ["QZ_RECORD_ROOT"]) / "train_args_dry_run_core.json"
path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[dry-run] wrote {path}")
PY

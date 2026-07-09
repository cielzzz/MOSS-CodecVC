#!/bin/sh
# Submit a single-node 8*H200 QZ/OpenI job for MOSS-CodecVC Ver2.x LoRA training.
#
# Usage:
#   sh scripts/002004_submit_ver2_lora_68w_h200_qz.sh
#   sh scripts/002004_submit_ver2_lora_68w_h200_qz.sh --dry-run
#
# Common overrides:
#   TRAIN_JSONL_SPEC='/path/no_text.jsonl::repeat=1,/path/text.jsonl::repeat=3' OUT_DIR=/path/to/out sh scripts/002004_submit_ver2_lora_68w_h200_qz.sh
#   NUM_EPOCHS=3 MAX_TRAIN_STEPS=0 JOB_NAME=my-job sh scripts/002004_submit_ver2_lora_68w_h200_qz.sh

set -eu

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
PAIR_CONSTRUCTION_ROOT="${PAIR_CONSTRUCTION_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction}"
QZCLI="${QZCLI:-$PAIR_CONSTRUCTION_ROOT/scripts/qzcli_with_deps.sh}"

WORKSPACE="${WORKSPACE:-CI-情境智能}"
PROJECT="${PROJECT:-CI-情境智能}"
COMPUTE_GROUP="${COMPUTE_GROUP:-lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122}"  # MTTS-3-2-0715
FRAMEWORK="${FRAMEWORK:-pytorch}"
INSTANCES="${INSTANCES:-1}"
SHM_GI="${SHM_GI:-1200}"
PRIORITY="${PRIORITY:-10}"
SPEC="${SPEC:-67b10bc6-78b0-41a3-aaf4-358eeeb99009}"
IMAGE="${IMAGE:-docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420}"
IMAGE_TYPE="${IMAGE_TYPE:-SOURCE_PRIVATE}"

DATASET_NAME="${DATASET_NAME:-zh45w_en22w_no_text}"
TEXT_DATASET_NAME="${TEXT_DATASET_NAME:-zh3w_en3w_text_prosody_independent_timbre}"
NO_TEXT_TRAIN_JSONL="${NO_TEXT_TRAIN_JSONL:-$ROOT/trainset/$DATASET_NAME/sft/moss_codecvc_sft.$DATASET_NAME.with_light_ecapa_spk.with_prosody.with_asr_filter.with_content_tokens.with_hubert.jsonl}"
TEXT_TRAIN_JSONL="${TEXT_TRAIN_JSONL:-$ROOT/trainset/$TEXT_DATASET_NAME/sft/moss_codecvc_sft.$TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_content_tokens.with_target_hubert.jsonl}"
TEXT_REPEAT="${TEXT_REPEAT:-3}"
TRAIN_JSONL="${TRAIN_JSONL:-$NO_TEXT_TRAIN_JSONL}"
TRAIN_JSONL_SPEC="${TRAIN_JSONL_SPEC:-$NO_TEXT_TRAIN_JSONL::repeat=1,$TEXT_TRAIN_JSONL::repeat=$TEXT_REPEAT}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/lora_runs/ver2_1_68w_textrep${TEXT_REPEAT}_lora_r16_a32_gbs64}"
PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download}"

BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%d-%H%M%S)}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codecvc-ver2-1-68w-textrep${TEXT_REPEAT}-train-lora}"
JOB_NAME="${JOB_NAME:-$JOB_NAME_PREFIX-$BATCH_ID}"
QZ_RECORD_ROOT="${QZ_RECORD_ROOT:-$ROOT/trainset/qz_jobs/$BATCH_ID}"
RUNNER="$QZ_RECORD_ROOT/run_train_entrypoint.sh"

NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
GPU_COUNT="${GPU_COUNT:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_fsdp_h200_8gpu_no_ckpt.yaml}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
SMOKE_TEST="${SMOKE_TEST:-0}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
RESUME_ADAPTER_PATH="${RESUME_ADAPTER_PATH:-}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
EVAL_JSONL="${EVAL_JSONL:-}"
EVAL_JSONL_SPEC="${EVAL_JSONL_SPEC:-}"
EVAL_SEEN_JSONL="${EVAL_SEEN_JSONL:-}"
EVAL_SEEN_JSONL_SPEC="${EVAL_SEEN_JSONL_SPEC:-}"
EVAL_UNSEEN_JSONL="${EVAL_UNSEEN_JSONL:-}"
EVAL_UNSEEN_JSONL_SPEC="${EVAL_UNSEEN_JSONL_SPEC:-}"
EVAL_STEPS="${EVAL_STEPS:-0}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"
POST_TRAIN_QUICK_EVAL="${POST_TRAIN_QUICK_EVAL:-0}"
POST_TRAIN_EVAL_SCRIPT="${POST_TRAIN_EVAL_SCRIPT:-$ROOT/scripts/004054_run_ver2_8_timbre_quick_eval.sh}"
POST_TRAIN_EVAL_LABEL="${POST_TRAIN_EVAL_LABEL:-$JOB_NAME_PREFIX}"
POST_TRAIN_EVAL_ROOT="${POST_TRAIN_EVAL_ROOT:-$ROOT/testset/outputs/ver2_8_timbre_repair_quick_eval}"
POST_TRAIN_EVAL_DOCS_MD="${POST_TRAIN_EVAL_DOCS_MD:-$ROOT/docs/ver2_8_timbre_repair_1a_1b_short_train_quick_eval_20260704.md}"
QUICK_VALIDATION_JSONL="${QUICK_VALIDATION_JSONL:-$ROOT/testset/validation/ver2_8_timbre_quick20_seedtts_no_text_20260704.jsonl}"
DOMAIN_VALIDATION_JSONL="${DOMAIN_VALIDATION_JSONL:-$ROOT/testset/validation/ver2_8_t11_domain_prepared_valid_no_text_50_20260704.jsonl}"
POST_TRAIN_QUICK_GPU_COUNT="${POST_TRAIN_QUICK_GPU_COUNT:-4}"
POST_TRAIN_QUICK_NUM_SHARDS="${POST_TRAIN_QUICK_NUM_SHARDS:-$POST_TRAIN_QUICK_GPU_COUNT}"
POST_TRAIN_QUICK_ASR_NUM_SHARDS="${POST_TRAIN_QUICK_ASR_NUM_SHARDS:-$POST_TRAIN_QUICK_GPU_COUNT}"
POST_TRAIN_RUN_T11="${POST_TRAIN_RUN_T11:-1}"

TARGET_SPK_WEIGHT="${TARGET_SPK_WEIGHT:-0.05}"
SOURCE_SUPPRESS_WEIGHT="${SOURCE_SUPPRESS_WEIGHT:-0.05}"
SPEAKER_LOSS_WARMUP_STEPS="${SPEAKER_LOSS_WARMUP_STEPS:-1000}"
SPEAKER_LOSS_WARMUP_WEIGHT="${SPEAKER_LOSS_WARMUP_WEIGHT:-0.02}"
SPEAKER_LOSS_SCHEDULE="${SPEAKER_LOSS_SCHEDULE:-step}"
SPEAKER_LOSS_MARGIN="${SPEAKER_LOSS_MARGIN:-0.10}"
SPEAKER_ENCODER_TYPE="${SPEAKER_ENCODER_TYPE:-embedding_loader}"
SPEAKER_ENCODER_PATH="${SPEAKER_ENCODER_PATH:-}"
SPEAKER_EMBEDDING_DIM="${SPEAKER_EMBEDDING_DIM:-192}"
REF_SPEAKER_PROMPT_TOKENS="${REF_SPEAKER_PROMPT_TOKENS:-0}"
REF_SPEAKER_PROMPT_DROPOUT="${REF_SPEAKER_PROMPT_DROPOUT:-0.0}"
REF_SPEAKER_PROMPT_MODE="${REF_SPEAKER_PROMPT_MODE:-memory}"
REF_SPEAKER_PROMPT_TOKEN_SOURCE="${REF_SPEAKER_PROMPT_TOKEN_SOURCE:-speaker_mlp}"
REF_SPEAKER_PROMPT_SLOT="${REF_SPEAKER_PROMPT_SLOT:-0}"
REF_SPEAKER_PROMPT_SLOT_CODE="${REF_SPEAKER_PROMPT_SLOT_CODE:--1}"
REF_SPEAKER_PROMPT_SLOT_PACK_MODE="${REF_SPEAKER_PROMPT_SLOT_PACK_MODE:-pad}"
REF_SPEAKER_PROMPT_OUTPUT_NORM="${REF_SPEAKER_PROMPT_OUTPUT_NORM:-0}"
REF_SPEAKER_PROMPT_OUTPUT_SCALE="${REF_SPEAKER_PROMPT_OUTPUT_SCALE:-1.0}"
REF_SPEAKER_PROMPT_LR_MULTIPLIER="${REF_SPEAKER_PROMPT_LR_MULTIPLIER:-1.0}"
REF_PROMPT_CODEC_PERMUTATION="${REF_PROMPT_CODEC_PERMUTATION:-0}"
REF_PROMPT_CODEC_PERMUTATION_MIN_SECONDS="${REF_PROMPT_CODEC_PERMUTATION_MIN_SECONDS:-2.0}"
REF_PROMPT_CODEC_PERMUTATION_MAX_SECONDS="${REF_PROMPT_CODEC_PERMUTATION_MAX_SECONDS:-4.0}"
REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE="${REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE:-12.5}"
REF_PROMPT_CODEC_PERMUTATION_SEED="${REF_PROMPT_CODEC_PERMUTATION_SEED:-1234}"
REF_PROMPT_CODEC_PERMUTATION_MODE="${REF_PROMPT_CODEC_PERMUTATION_MODE:-shuffle}"
REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS="${REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS:-0.4}"
POST_TRAIN_REF_PROMPT_CODEC_PERMUTATION="${POST_TRAIN_REF_PROMPT_CODEC_PERMUTATION:-$REF_PROMPT_CODEC_PERMUTATION}"
POST_TRAIN_REF_PROMPT_CODEC_PERMUTATION_MODE="${POST_TRAIN_REF_PROMPT_CODEC_PERMUTATION_MODE:-$REF_PROMPT_CODEC_PERMUTATION_MODE}"
POST_TRAIN_REF_PROMPT_CODEC_PERMUTATION_MIN_SECONDS="${POST_TRAIN_REF_PROMPT_CODEC_PERMUTATION_MIN_SECONDS:-$REF_PROMPT_CODEC_PERMUTATION_MIN_SECONDS}"
POST_TRAIN_REF_PROMPT_CODEC_PERMUTATION_MAX_SECONDS="${POST_TRAIN_REF_PROMPT_CODEC_PERMUTATION_MAX_SECONDS:-$REF_PROMPT_CODEC_PERMUTATION_MAX_SECONDS}"
POST_TRAIN_REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE="${POST_TRAIN_REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE:-$REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE}"
POST_TRAIN_REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS="${POST_TRAIN_REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS:-$REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS}"
POST_TRAIN_REF_PROMPT_CODEC_PERMUTATION_BOOTSTRAP="${POST_TRAIN_REF_PROMPT_CODEC_PERMUTATION_BOOTSTRAP:-}"
TARGET_FRONT_CE_WEIGHT="${TARGET_FRONT_CE_WEIGHT:-1.0}"
TARGET_FRONT_CE_SECONDS="${TARGET_FRONT_CE_SECONDS:-0.0}"
TARGET_FRONT_CE_FRAME_RATE="${TARGET_FRONT_CE_FRAME_RATE:-12.5}"
REF_SPEAKER_ADALN_WEIGHT="${REF_SPEAKER_ADALN_WEIGHT:-0.0}"
SPEAKER_INFONCE_WEIGHT="${SPEAKER_INFONCE_WEIGHT:-0.0}"
SPEAKER_INFONCE_TEMPERATURE="${SPEAKER_INFONCE_TEMPERATURE:-0.07}"
SPEAKER_INFONCE_NEGATIVE_POOL_SIZE="${SPEAKER_INFONCE_NEGATIVE_POOL_SIZE:-0}"
SPEAKER_INFONCE_NEGATIVE_POOL_SEED="${SPEAKER_INFONCE_NEGATIVE_POOL_SEED:-1234}"
SPEAKER_CONDITION_DROPOUT="${SPEAKER_CONDITION_DROPOUT:-0.0}"
ENABLE_SPEAKER_SIDE_PATHWAY="${ENABLE_SPEAKER_SIDE_PATHWAY:-0}"
SPEAKER_SIDE_PATHWAY_LAYERS="${SPEAKER_SIDE_PATHWAY_LAYERS:-all}"
SPEAKER_SIDE_PATHWAY_KV_BIAS="${SPEAKER_SIDE_PATHWAY_KV_BIAS:-1}"
SPEAKER_SIDE_PATHWAY_GATE_INIT="${SPEAKER_SIDE_PATHWAY_GATE_INIT:-0.0}"
SPEAKER_SIDE_PATHWAY_DROPOUT="${SPEAKER_SIDE_PATHWAY_DROPOUT:-0.15}"
ENABLE_SPEAKER_CROSS_ATTN="${ENABLE_SPEAKER_CROSS_ATTN:-0}"
SPEAKER_CROSS_ATTN_LAYERS="${SPEAKER_CROSS_ATTN_LAYERS:-all}"
SPEAKER_CROSS_ATTN_TOKENS="${SPEAKER_CROSS_ATTN_TOKENS:-8}"
SPEAKER_CROSS_ATTN_GATE_INIT="${SPEAKER_CROSS_ATTN_GATE_INIT:-0.0}"
SPEAKER_CROSS_ATTN_DROPOUT="${SPEAKER_CROSS_ATTN_DROPOUT:-0.0}"
SPEAKER_CROSS_ATTN_OUTPUT_SCALE="${SPEAKER_CROSS_ATTN_OUTPUT_SCALE:-1.0}"
SPEAKER_CROSS_ATTN_TOKEN_INIT_STD="${SPEAKER_CROSS_ATTN_TOKEN_INIT_STD:-}"
SPEAKER_CROSS_ATTN_ALPHA_WARMUP_STEPS="${SPEAKER_CROSS_ATTN_ALPHA_WARMUP_STEPS:-0}"
SPEAKER_CROSS_ATTN_SOURCE="${SPEAKER_CROSS_ATTN_SOURCE:-vector}"
SPEAKER_CROSS_ATTN_SEQ_DIM="${SPEAKER_CROSS_ATTN_SEQ_DIM:-0}"
USE_PERTURBED_SOURCE_PROMPT="${USE_PERTURBED_SOURCE_PROMPT:-0}"
LAMBDA_ROUTE="${LAMBDA_ROUTE:-0.01}"
ROUTING_GATE_LR_MULTIPLIER="${ROUTING_GATE_LR_MULTIPLIER:-10.0}"
CONTENT_CTC_HEAD_LR_MULTIPLIER="${CONTENT_CTC_HEAD_LR_MULTIPLIER:-1.0}"
TIMBRE_ADAPTER_INIT_GATE="${TIMBRE_ADAPTER_INIT_GATE:--4.0}"
TIMBRE_ADAPTER_GATE_LR_MULTIPLIER="${TIMBRE_ADAPTER_GATE_LR_MULTIPLIER:-1.0}"
USE_TIMBRE_MEMORY="${USE_TIMBRE_MEMORY:-1}"
TIMBRE_MEMORY_TOKENS="${TIMBRE_MEMORY_TOKENS:-16}"
TIMBRE_ADAPTER_LAYERS="${TIMBRE_ADAPTER_LAYERS-last_4}"
LAMBDA_PROSODY="${LAMBDA_PROSODY:-0.05}"
LAMBDA_CONTENT="${LAMBDA_CONTENT:-0.0}"
PROSODY_F0_WEIGHT="${PROSODY_F0_WEIGHT:-0.0}"
PROSODY_VOICED_WEIGHT="${PROSODY_VOICED_WEIGHT:-0.0}"
PROSODY_ENERGY_WEIGHT="${PROSODY_ENERGY_WEIGHT:-0.5}"
PROSODY_PAUSE_WEIGHT="${PROSODY_PAUSE_WEIGHT:-1.0}"
PROSODY_DURATION_WEIGHT="${PROSODY_DURATION_WEIGHT:-0.5}"
CONTENT_POSITIVE="${CONTENT_POSITIVE:-source}"
CONTENT_EMBEDDING_DIM="${CONTENT_EMBEDDING_DIM:-0}"
CONTENT_EMBEDDING_WEIGHT="${CONTENT_EMBEDDING_WEIGHT:-1.0}"
CONTENT_CTC_WEIGHT="${CONTENT_CTC_WEIGHT:-0.10}"
CONTENT_CTC_VOCAB_SIZE="${CONTENT_CTC_VOCAB_SIZE:-0}"
CONTENT_CTC_BLANK_ID="${CONTENT_CTC_BLANK_ID:-0}"
CONTENT_CTC_TOKEN_OFFSET="${CONTENT_CTC_TOKEN_OFFSET:-1}"
CONTENT_TOKEN_VOCAB_SIZE="${CONTENT_TOKEN_VOCAB_SIZE:-0}"
CONTENT_TOKEN_WEIGHT="${CONTENT_TOKEN_WEIGHT:-0.0}"
CONTENT_SOURCE_CODEC_WEIGHT="${CONTENT_SOURCE_CODEC_WEIGHT:-0.0}"
CONTENT_SOURCE_CODEC_CODEBOOKS="${CONTENT_SOURCE_CODEC_CODEBOOKS:-0,1,2,3}"
SEMANTIC_LOSS_WEIGHT="${SEMANTIC_LOSS_WEIGHT:-0.05}"
SEMANTIC_MODE="${SEMANTIC_MODE:-continuous}"
SEMANTIC_SOURCE="${SEMANTIC_SOURCE:-mode_aware}"
SEMANTIC_VOCAB_SIZE="${SEMANTIC_VOCAB_SIZE:-0}"
SEMANTIC_FEATURE_DIM="${SEMANTIC_FEATURE_DIM:-0}"
SEMANTIC_FEATURE_LOSS_TYPE="${SEMANTIC_FEATURE_LOSS_TYPE:-cosine}"
PROGRESS_LOSS_WEIGHT="${PROGRESS_LOSS_WEIGHT:-0.0}"
STOP_LOSS_WEIGHT="${STOP_LOSS_WEIGHT:-0.0}"
PROGRESS_NUM_BINS="${PROGRESS_NUM_BINS:-32}"
ENABLE_SOURCE_SEMANTIC_MEMORY="${ENABLE_SOURCE_SEMANTIC_MEMORY:-0}"
SOURCE_SEMANTIC_FEATURE_DIM="${SOURCE_SEMANTIC_FEATURE_DIM:-768}"
SOURCE_SEMANTIC_ADAPTER_LAYERS="${SOURCE_SEMANTIC_ADAPTER_LAYERS-28,30,32,34,35}"
SOURCE_SEMANTIC_NO_TEXT_GATE="${SOURCE_SEMANTIC_NO_TEXT_GATE:-1.0}"
SOURCE_SEMANTIC_TEXT_GATE="${SOURCE_SEMANTIC_TEXT_GATE:-0.0}"
SOURCE_SEMANTIC_LEARNED_TEXT_GATE="${SOURCE_SEMANTIC_LEARNED_TEXT_GATE:-0}"
SOURCE_SEMANTIC_PROGRESS_WEIGHT="${SOURCE_SEMANTIC_PROGRESS_WEIGHT:-0.0}"
SOURCE_SEMANTIC_DROPOUT="${SOURCE_SEMANTIC_DROPOUT:-0.1}"
SOURCE_SEMANTIC_INIT_GATE="${SOURCE_SEMANTIC_INIT_GATE:--2.0}"
SOURCE_SEMANTIC_POSITION_SCALE="${SOURCE_SEMANTIC_POSITION_SCALE:-0.0}"
SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH="${SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH:-0.0}"
SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH="${SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH:-0.25}"
SOURCE_CONTENT_MEMORY_TYPE="${SOURCE_CONTENT_MEMORY_TYPE:-hubert_continuous}"
SOURCE_CONTENT_VOCAB_SIZE="${SOURCE_CONTENT_VOCAB_SIZE:-0}"
SOURCE_CONTENT_PADDING_ID="${SOURCE_CONTENT_PADDING_ID:-0}"
SOURCE_CONTENT_CODEC_BOTTLENECK_DIM="${SOURCE_CONTENT_CODEC_BOTTLENECK_DIM:-256}"
SOURCE_CONTENT_CODEC_CODEBOOKS="${SOURCE_CONTENT_CODEC_CODEBOOKS:-first_4}"
SOURCE_CONTENT_DEDUP_UNITS="${SOURCE_CONTENT_DEDUP_UNITS:-0}"
SOURCE_CODEC_RESIDUAL_MEMORY_WEIGHT="${SOURCE_CODEC_RESIDUAL_MEMORY_WEIGHT:-0.0}"
SOURCE_CODEC_RESIDUAL_MEMORY_DETACH="${SOURCE_CODEC_RESIDUAL_MEMORY_DETACH:-0}"
ENABLE_CONTENT_CROSS_ATTN="${ENABLE_CONTENT_CROSS_ATTN:-0}"
CONTENT_CROSS_ATTN_LAYERS="${CONTENT_CROSS_ATTN_LAYERS:-all}"
CONTENT_CROSS_ATTN_FEATURE_DIM="${CONTENT_CROSS_ATTN_FEATURE_DIM:-768}"
CONTENT_CROSS_ATTN_GATE_INIT="${CONTENT_CROSS_ATTN_GATE_INIT:--0.5}"
CONTENT_CROSS_ATTN_DROPOUT="${CONTENT_CROSS_ATTN_DROPOUT:-0.0}"
CONTENT_CROSS_ATTN_OUTPUT_SCALE="${CONTENT_CROSS_ATTN_OUTPUT_SCALE:-0.3}"
CONTENT_ENCODER_LAYERS="${CONTENT_ENCODER_LAYERS:-2}"
CONTENT_ENCODER_CONV_KERNEL_SIZE="${CONTENT_ENCODER_CONV_KERNEL_SIZE:-7}"
GUIDED_ATTN_LOSS_WEIGHT="${GUIDED_ATTN_LOSS_WEIGHT:-0.0}"
GUIDED_ATTN_WARMUP_STEPS="${GUIDED_ATTN_WARMUP_STEPS:-1000}"
GUIDED_ATTN_BAND_FRAMES="${GUIDED_ATTN_BAND_FRAMES:-3}"
PHONEME_CLASSIFIER_LOSS_WEIGHT="${PHONEME_CLASSIFIER_LOSS_WEIGHT:-0.0}"
SOURCE_PROSODY_NO_TEXT_GATE="${SOURCE_PROSODY_NO_TEXT_GATE:-1.0}"
SOURCE_PROSODY_TEXT_GATE="${SOURCE_PROSODY_TEXT_GATE:-1.0}"
TIMBRE_SIDE_ONLY="${TIMBRE_SIDE_ONLY:-0}"
REF_CONTENT_SUPPRESSION_WEIGHT="${REF_CONTENT_SUPPRESSION_WEIGHT:-0.0}"
REF_CONTENT_SUPPRESSION_MARGIN="${REF_CONTENT_SUPPRESSION_MARGIN:-0.0}"
REF_CONTENT_SUPPRESSION_SOURCE="${REF_CONTENT_SUPPRESSION_SOURCE:-auto}"
REF_CONTENT_SUPPRESSION_DETACH_REF="${REF_CONTENT_SUPPRESSION_DETACH_REF:-1}"
SOURCE_SEMANTIC_LR_MULTIPLIER="${SOURCE_SEMANTIC_LR_MULTIPLIER:-1.0}"
SOURCE_SEMANTIC_GATE_LR_MULTIPLIER="${SOURCE_SEMANTIC_GATE_LR_MULTIPLIER:-10.0}"
TRAIN_SOURCE_SEMANTIC_ONLY="${TRAIN_SOURCE_SEMANTIC_ONLY:-0}"
FREEZE_LORA="${FREEZE_LORA:-0}"
FREEZE_ROLE_ROUTING="${FREEZE_ROLE_ROUTING:-0}"
FREEZE_TIMBRE_ADAPTER="${FREEZE_TIMBRE_ADAPTER:-0}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-}"
if [ -z "$CUDA_VISIBLE_DEVICES_VALUE" ]; then
  if [ "$GPU_COUNT" = "1" ]; then
    CUDA_VISIBLE_DEVICES_VALUE="0"
  elif [ "$GPU_COUNT" = "2" ]; then
    CUDA_VISIBLE_DEVICES_VALUE="0,1"
  elif [ "$GPU_COUNT" = "4" ]; then
    CUDA_VISIBLE_DEVICES_VALUE="0,1,2,3"
  else
    CUDA_VISIBLE_DEVICES_VALUE="0,1,2,3,4,5,6,7"
  fi
fi

DRY_RUN="${DRY_RUN:-0}"
SKIP_LOCAL_DATA_CHECK="${SKIP_LOCAL_DATA_CHECK:-0}"

usage() {
  cat <<EOF
Usage:
  sh scripts/002004_submit_ver2_lora_68w_h200_qz.sh [--dry-run] [--skip-local-data-check]

Important env overrides:
  TRAIN_JSONL_SPEC=...  default: $TRAIN_JSONL_SPEC
  OUT_DIR=...           default: $OUT_DIR
  JOB_NAME=...          default: $JOB_NAME
  NUM_EPOCHS=...        default: $NUM_EPOCHS
  MAX_TRAIN_STEPS=...   default: $MAX_TRAIN_STEPS (0 means epoch-based)
  EVAL_JSONL=...        optional train-ready valid JSONL for checkpoint eval loss
  COMPUTE_GROUP=...     default: $COMPUTE_GROUP
EOF
}

check_jsonl_spec_paths() {
  spec="$1"
  old_ifs="$IFS"
  IFS=','
  for item in $spec; do
    IFS="$old_ifs"
    path="${item%%::*}"
    if [ -n "$path" ] && [ ! -f "$path" ]; then
      echo "ERROR: TRAIN_JSONL_SPEC path does not exist: $path" >&2
      exit 1
    fi
    IFS=','
  done
  IFS="$old_ifs"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --skip-local-data-check)
      SKIP_LOCAL_DATA_CHECK=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [ ! -d "$ROOT" ]; then
  echo "ERROR: ROOT does not exist: $ROOT" >&2
  exit 1
fi
if [ ! -x "$QZCLI" ]; then
  echo "ERROR: qzcli wrapper is not executable: $QZCLI" >&2
  exit 1
fi
if [ ! -x "$PY" ]; then
  echo "ERROR: training python is not executable: $PY" >&2
  exit 1
fi
if [ "$DRY_RUN" -ne 1 ] && [ "$SKIP_LOCAL_DATA_CHECK" -ne 1 ]; then
  if [ ! -f "$NO_TEXT_TRAIN_JSONL" ]; then
    echo "ERROR: NO_TEXT_TRAIN_JSONL does not exist yet: $NO_TEXT_TRAIN_JSONL" >&2
    exit 1
  fi
  if [ ! -f "$TEXT_TRAIN_JSONL" ]; then
    echo "ERROR: TEXT_TRAIN_JSONL does not exist yet: $TEXT_TRAIN_JSONL" >&2
    exit 1
  fi
  if [ -n "$EVAL_JSONL" ] && [ ! -f "$EVAL_JSONL" ]; then
    echo "ERROR: EVAL_JSONL does not exist yet: $EVAL_JSONL" >&2
    exit 1
  fi
  check_jsonl_spec_paths "$TRAIN_JSONL_SPEC"
  if [ -n "$EVAL_JSONL_SPEC" ]; then
    check_jsonl_spec_paths "$EVAL_JSONL_SPEC"
  fi
  if [ -n "$EVAL_SEEN_JSONL" ] && [ ! -f "$EVAL_SEEN_JSONL" ]; then
    echo "ERROR: EVAL_SEEN_JSONL does not exist yet: $EVAL_SEEN_JSONL" >&2
    exit 1
  fi
  if [ -n "$EVAL_SEEN_JSONL_SPEC" ]; then
    check_jsonl_spec_paths "$EVAL_SEEN_JSONL_SPEC"
  fi
  if [ -n "$EVAL_UNSEEN_JSONL" ] && [ ! -f "$EVAL_UNSEEN_JSONL" ]; then
    echo "ERROR: EVAL_UNSEEN_JSONL does not exist yet: $EVAL_UNSEEN_JSONL" >&2
    exit 1
  fi
  if [ -n "$EVAL_UNSEEN_JSONL_SPEC" ]; then
    check_jsonl_spec_paths "$EVAL_UNSEEN_JSONL_SPEC"
  fi
fi

mkdir -p "$QZ_RECORD_ROOT"

cat > "$RUNNER" <<EOF
#!/bin/sh
set -eu

export ROOT="$ROOT"
export PY="$PY"
export DOWNLOAD_ROOT="$DOWNLOAD_ROOT"
export HF_HOME="\$DOWNLOAD_ROOT/huggingface"
export TRANSFORMERS_CACHE="\$DOWNLOAD_ROOT/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="\$DOWNLOAD_ROOT/huggingface/hub"
export HF_DATASETS_CACHE="\$DOWNLOAD_ROOT/huggingface/datasets"
export TORCH_HOME="\$DOWNLOAD_ROOT/torch"
export XDG_CACHE_HOME="\$DOWNLOAD_ROOT/cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=\${NCCL_DEBUG:-WARN}

cd "\$ROOT"

TRAIN_JSONL="$TRAIN_JSONL"
NO_TEXT_TRAIN_JSONL="$NO_TEXT_TRAIN_JSONL"
TEXT_TRAIN_JSONL="$TEXT_TRAIN_JSONL"
TRAIN_JSONL_SPEC="$TRAIN_JSONL_SPEC"
OUT_DIR="$OUT_DIR"
EVAL_JSONL="$EVAL_JSONL"
EVAL_JSONL_SPEC="$EVAL_JSONL_SPEC"
EVAL_SEEN_JSONL="$EVAL_SEEN_JSONL"
EVAL_SEEN_JSONL_SPEC="$EVAL_SEEN_JSONL_SPEC"
EVAL_UNSEEN_JSONL="$EVAL_UNSEEN_JSONL"
EVAL_UNSEEN_JSONL_SPEC="$EVAL_UNSEEN_JSONL_SPEC"
EVAL_STEPS="$EVAL_STEPS"
EVAL_MAX_BATCHES="$EVAL_MAX_BATCHES"
EVAL_NUM_WORKERS="$EVAL_NUM_WORKERS"
RESUME_ADAPTER_PATH="$RESUME_ADAPTER_PATH"

echo "[qz-train] date=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[qz-train] host=\$(hostname)"
echo "[qz-train] root=\$ROOT"
echo "[qz-train] no_text_train_jsonl=\$NO_TEXT_TRAIN_JSONL"
echo "[qz-train] text_train_jsonl=\$TEXT_TRAIN_JSONL"
echo "[qz-train] train_jsonl_spec=\$TRAIN_JSONL_SPEC"
echo "[qz-train] text_repeat=$TEXT_REPEAT"
echo "[qz-train] out_dir=\$OUT_DIR"
echo "[qz-train] python=\$PY"
echo "[qz-train] num_epochs=$NUM_EPOCHS max_train_steps=$MAX_TRAIN_STEPS"
echo "[qz-train] lr_scheduler_type=$LR_SCHEDULER_TYPE warmup_ratio=$WARMUP_RATIO"
echo "[qz-train] eval_jsonl=\$EVAL_JSONL eval_jsonl_spec=\$EVAL_JSONL_SPEC eval_seen_jsonl=\$EVAL_SEEN_JSONL eval_seen_jsonl_spec=\$EVAL_SEEN_JSONL_SPEC eval_unseen_jsonl=\$EVAL_UNSEEN_JSONL eval_unseen_jsonl_spec=\$EVAL_UNSEEN_JSONL_SPEC eval_steps=\$EVAL_STEPS eval_max_batches=\$EVAL_MAX_BATCHES"
echo "[qz-train] post_train_quick_eval=$POST_TRAIN_QUICK_EVAL post_train_eval_label=$POST_TRAIN_EVAL_LABEL"
echo "[qz-train] gpu_count=$GPU_COUNT cuda_visible_devices=\$CUDA_VISIBLE_DEVICES"
echo "[qz-train] global_batch_size=$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * GPU_COUNT))"
echo "[qz-train] routing_gate_lr_multiplier=$ROUTING_GATE_LR_MULTIPLIER"
echo "[qz-train] content_ctc_head_lr_multiplier=$CONTENT_CTC_HEAD_LR_MULTIPLIER"
echo "[qz-train] use_timbre_memory=$USE_TIMBRE_MEMORY timbre_memory_tokens=$TIMBRE_MEMORY_TOKENS timbre_adapter_layers='$TIMBRE_ADAPTER_LAYERS'"
echo "[qz-train] timbre_adapter_init_gate=$TIMBRE_ADAPTER_INIT_GATE timbre_adapter_gate_lr_multiplier=$TIMBRE_ADAPTER_GATE_LR_MULTIPLIER"
echo "[qz-train] speaker_encoder_type=$SPEAKER_ENCODER_TYPE speaker_encoder_path=$SPEAKER_ENCODER_PATH speaker_embedding_dim=$SPEAKER_EMBEDDING_DIM"
echo "[qz-train] speaker_repair ref_prompt_tokens=$REF_SPEAKER_PROMPT_TOKENS ref_prompt_mode=$REF_SPEAKER_PROMPT_MODE ref_prompt_source=$REF_SPEAKER_PROMPT_TOKEN_SOURCE ref_prompt_slot=$REF_SPEAKER_PROMPT_SLOT ref_prompt_slot_code=$REF_SPEAKER_PROMPT_SLOT_CODE ref_prompt_slot_pack_mode=$REF_SPEAKER_PROMPT_SLOT_PACK_MODE ref_prompt_output_norm=$REF_SPEAKER_PROMPT_OUTPUT_NORM ref_prompt_output_scale=$REF_SPEAKER_PROMPT_OUTPUT_SCALE ref_prompt_lr_multiplier=$REF_SPEAKER_PROMPT_LR_MULTIPLIER ref_prompt_codec_permutation=$REF_PROMPT_CODEC_PERMUTATION ref_prompt_codec_permutation_seconds=$REF_PROMPT_CODEC_PERMUTATION_MIN_SECONDS-$REF_PROMPT_CODEC_PERMUTATION_MAX_SECONDS ref_prompt_codec_permutation_mode=$REF_PROMPT_CODEC_PERMUTATION_MODE ref_prompt_codec_permutation_block_seconds=$REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS ref_prompt_codec_permutation_seed=$REF_PROMPT_CODEC_PERMUTATION_SEED target_front_ce_weight=$TARGET_FRONT_CE_WEIGHT target_front_ce_seconds=$TARGET_FRONT_CE_SECONDS target_front_ce_frame_rate=$TARGET_FRONT_CE_FRAME_RATE speaker_loss_schedule=$SPEAKER_LOSS_SCHEDULE speaker_loss_warmup_steps=$SPEAKER_LOSS_WARMUP_STEPS speaker_loss_warmup_weight=$SPEAKER_LOSS_WARMUP_WEIGHT ref_adaln_weight=$REF_SPEAKER_ADALN_WEIGHT infonce_weight=$SPEAKER_INFONCE_WEIGHT infonce_neg_pool=$SPEAKER_INFONCE_NEGATIVE_POOL_SIZE infonce_neg_pool_seed=$SPEAKER_INFONCE_NEGATIVE_POOL_SEED condition_dropout=$SPEAKER_CONDITION_DROPOUT use_perturbed_source_prompt=$USE_PERTURBED_SOURCE_PROMPT"
echo "[qz-train] speaker_side_pathway=$ENABLE_SPEAKER_SIDE_PATHWAY layers=$SPEAKER_SIDE_PATHWAY_LAYERS kv_bias=$SPEAKER_SIDE_PATHWAY_KV_BIAS gate_init=$SPEAKER_SIDE_PATHWAY_GATE_INIT dropout=$SPEAKER_SIDE_PATHWAY_DROPOUT"
echo "[qz-train] speaker_cross_attn=$ENABLE_SPEAKER_CROSS_ATTN source=$SPEAKER_CROSS_ATTN_SOURCE seq_dim=$SPEAKER_CROSS_ATTN_SEQ_DIM layers=$SPEAKER_CROSS_ATTN_LAYERS tokens=$SPEAKER_CROSS_ATTN_TOKENS gate_init=$SPEAKER_CROSS_ATTN_GATE_INIT dropout=$SPEAKER_CROSS_ATTN_DROPOUT output_scale=$SPEAKER_CROSS_ATTN_OUTPUT_SCALE token_init_std=$SPEAKER_CROSS_ATTN_TOKEN_INIT_STD"
echo "[qz-train] lambda_content=$LAMBDA_CONTENT content_source_codec_weight=$CONTENT_SOURCE_CODEC_WEIGHT content_token_vocab_size=$CONTENT_TOKEN_VOCAB_SIZE content_ctc_weight=$CONTENT_CTC_WEIGHT semantic_loss_weight=$SEMANTIC_LOSS_WEIGHT"
echo "[qz-train] progress_loss_weight=$PROGRESS_LOSS_WEIGHT stop_loss_weight=$STOP_LOSS_WEIGHT progress_num_bins=$PROGRESS_NUM_BINS"
echo "[qz-train] source_semantic_memory=$ENABLE_SOURCE_SEMANTIC_MEMORY source_semantic_layers=$SOURCE_SEMANTIC_ADAPTER_LAYERS source_semantic_progress_weight=$SOURCE_SEMANTIC_PROGRESS_WEIGHT source_semantic_lr_multiplier=$SOURCE_SEMANTIC_LR_MULTIPLIER source_semantic_gate_lr_multiplier=$SOURCE_SEMANTIC_GATE_LR_MULTIPLIER"
echo "[qz-train] source_content_memory_type=$SOURCE_CONTENT_MEMORY_TYPE source_content_vocab_size=$SOURCE_CONTENT_VOCAB_SIZE source_content_codec_bottleneck_dim=$SOURCE_CONTENT_CODEC_BOTTLENECK_DIM source_content_codec_codebooks=$SOURCE_CONTENT_CODEC_CODEBOOKS source_content_dedup_units=$SOURCE_CONTENT_DEDUP_UNITS source_codec_residual_memory_weight=$SOURCE_CODEC_RESIDUAL_MEMORY_WEIGHT source_codec_residual_memory_detach=$SOURCE_CODEC_RESIDUAL_MEMORY_DETACH"
echo "[qz-train] content_cross_attn=$ENABLE_CONTENT_CROSS_ATTN layers=$CONTENT_CROSS_ATTN_LAYERS feature_dim=$CONTENT_CROSS_ATTN_FEATURE_DIM gate_init=$CONTENT_CROSS_ATTN_GATE_INIT dropout=$CONTENT_CROSS_ATTN_DROPOUT output_scale=$CONTENT_CROSS_ATTN_OUTPUT_SCALE encoder_layers=$CONTENT_ENCODER_LAYERS guided_weight=$GUIDED_ATTN_LOSS_WEIGHT guided_warmup=$GUIDED_ATTN_WARMUP_STEPS guided_band=$GUIDED_ATTN_BAND_FRAMES phoneme_weight=$PHONEME_CLASSIFIER_LOSS_WEIGHT"
echo "[qz-train] source_prosody_gates no_text=$SOURCE_PROSODY_NO_TEXT_GATE text=$SOURCE_PROSODY_TEXT_GATE"
echo "[qz-train] timbre_side_only=$TIMBRE_SIDE_ONLY ref_content_suppression_weight=$REF_CONTENT_SUPPRESSION_WEIGHT margin=$REF_CONTENT_SUPPRESSION_MARGIN source=$REF_CONTENT_SUPPRESSION_SOURCE detach_ref=$REF_CONTENT_SUPPRESSION_DETACH_REF"
echo "[qz-train] source_semantic_position_scale=$SOURCE_SEMANTIC_POSITION_SCALE source_semantic_monotonic_bias_strength=$SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH source_semantic_monotonic_bias_width=$SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH"
echo "[qz-train] train_source_semantic_only=$TRAIN_SOURCE_SEMANTIC_ONLY freeze_lora=$FREEZE_LORA freeze_role_routing=$FREEZE_ROLE_ROUTING freeze_timbre_adapter=$FREEZE_TIMBRE_ADAPTER"
nvidia-smi

test -f "\$NO_TEXT_TRAIN_JSONL"
test -f "\$TEXT_TRAIN_JSONL"
if [ -n "\$EVAL_JSONL" ]; then
  test -f "\$EVAL_JSONL"
fi
if [ -n "\$EVAL_SEEN_JSONL" ]; then
  test -f "\$EVAL_SEEN_JSONL"
fi
if [ -n "\$EVAL_UNSEEN_JSONL" ]; then
  test -f "\$EVAL_UNSEEN_JSONL"
fi

GRADIENT_CHECKPOINTING_ARGS=""
if [ "$GRADIENT_CHECKPOINTING" = "1" ]; then
  GRADIENT_CHECKPOINTING_ARGS="--gradient-checkpointing"
fi
SMOKE_TEST_ARGS=""
if [ "$SMOKE_TEST" = "1" ]; then
  SMOKE_TEST_ARGS="--smoke-test"
fi
TIMBRE_MEMORY_ARGS=""
if [ "$USE_TIMBRE_MEMORY" = "1" ]; then
  TIMBRE_MEMORY_ARGS="--use-timbre-memory"
else
  TIMBRE_MEMORY_ARGS="--no-use-timbre-memory"
fi
TIMBRE_MEMORY_ARGS="\$TIMBRE_MEMORY_ARGS --timbre-memory-tokens $TIMBRE_MEMORY_TOKENS"
if [ -n "$TIMBRE_ADAPTER_LAYERS" ]; then
  TIMBRE_MEMORY_ARGS="\$TIMBRE_MEMORY_ARGS --timbre-adapter-layers $TIMBRE_ADAPTER_LAYERS"
fi

EVAL_ARGS=""
if [ -n "\$EVAL_JSONL" ]; then
  EVAL_ARGS="\$EVAL_ARGS --eval-jsonl \$EVAL_JSONL"
fi
if [ -n "\$EVAL_JSONL_SPEC" ]; then
  EVAL_ARGS="\$EVAL_ARGS --eval-jsonl-spec \$EVAL_JSONL_SPEC"
fi
if [ -n "\$EVAL_SEEN_JSONL" ]; then
  EVAL_ARGS="\$EVAL_ARGS --eval-seen-jsonl \$EVAL_SEEN_JSONL"
fi
if [ -n "\$EVAL_SEEN_JSONL_SPEC" ]; then
  EVAL_ARGS="\$EVAL_ARGS --eval-seen-jsonl-spec \$EVAL_SEEN_JSONL_SPEC"
fi
if [ -n "\$EVAL_UNSEEN_JSONL" ]; then
  EVAL_ARGS="\$EVAL_ARGS --eval-unseen-jsonl \$EVAL_UNSEEN_JSONL"
fi
if [ -n "\$EVAL_UNSEEN_JSONL_SPEC" ]; then
  EVAL_ARGS="\$EVAL_ARGS --eval-unseen-jsonl-spec \$EVAL_UNSEEN_JSONL_SPEC"
fi
if [ "\$EVAL_STEPS" != "0" ]; then
  EVAL_ARGS="\$EVAL_ARGS --eval-steps \$EVAL_STEPS"
fi
if [ "\$EVAL_MAX_BATCHES" != "0" ]; then
  EVAL_ARGS="\$EVAL_ARGS --eval-max-batches \$EVAL_MAX_BATCHES"
fi
EVAL_ARGS="\$EVAL_ARGS --eval-num-workers \$EVAL_NUM_WORKERS"

RESUME_ARGS=""
if [ -n "\$RESUME_ADAPTER_PATH" ]; then
  RESUME_ARGS="--resume-adapter-path \$RESUME_ADAPTER_PATH"
fi

SOURCE_SEMANTIC_ARGS=""
if [ "$ENABLE_SOURCE_SEMANTIC_MEMORY" = "1" ]; then
  SOURCE_SEMANTIC_ARGS="--enable-source-semantic-memory"
else
  SOURCE_SEMANTIC_ARGS="--no-enable-source-semantic-memory"
fi
if [ "$SOURCE_SEMANTIC_LEARNED_TEXT_GATE" = "1" ]; then
  SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --source-semantic-learned-text-gate"
else
  SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --no-source-semantic-learned-text-gate"
fi
SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --source-semantic-feature-dim $SOURCE_SEMANTIC_FEATURE_DIM"
if [ -n "$SOURCE_SEMANTIC_ADAPTER_LAYERS" ]; then
  SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --source-semantic-adapter-layers $SOURCE_SEMANTIC_ADAPTER_LAYERS"
fi
SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --source-semantic-no-text-gate $SOURCE_SEMANTIC_NO_TEXT_GATE"
SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --source-semantic-text-gate $SOURCE_SEMANTIC_TEXT_GATE"
SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --source-semantic-progress-weight $SOURCE_SEMANTIC_PROGRESS_WEIGHT"
SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --source-semantic-dropout $SOURCE_SEMANTIC_DROPOUT"
SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --source-semantic-init-gate $SOURCE_SEMANTIC_INIT_GATE"
SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --source-semantic-position-scale $SOURCE_SEMANTIC_POSITION_SCALE"
SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --source-semantic-monotonic-bias-strength $SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH"
SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --source-semantic-monotonic-bias-width $SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH"
SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --source-content-memory-type $SOURCE_CONTENT_MEMORY_TYPE"
SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --source-content-vocab-size $SOURCE_CONTENT_VOCAB_SIZE"
SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --source-content-padding-id $SOURCE_CONTENT_PADDING_ID"
SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --source-content-codec-bottleneck-dim $SOURCE_CONTENT_CODEC_BOTTLENECK_DIM"
SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --source-content-codec-codebooks $SOURCE_CONTENT_CODEC_CODEBOOKS"
if [ "$SOURCE_CONTENT_DEDUP_UNITS" = "1" ]; then
  SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --source-content-dedup-units"
else
  SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --no-source-content-dedup-units"
fi
SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --source-codec-residual-memory-weight $SOURCE_CODEC_RESIDUAL_MEMORY_WEIGHT"
if [ "$SOURCE_CODEC_RESIDUAL_MEMORY_DETACH" = "1" ]; then
  SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --source-codec-residual-memory-detach"
else
  SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --no-source-codec-residual-memory-detach"
fi
SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --source-semantic-gate-lr-multiplier $SOURCE_SEMANTIC_GATE_LR_MULTIPLIER"
SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --source-semantic-lr-multiplier $SOURCE_SEMANTIC_LR_MULTIPLIER"

CONTENT_CROSS_ATTN_ARGS=""
if [ "$ENABLE_CONTENT_CROSS_ATTN" = "1" ]; then
  CONTENT_CROSS_ATTN_ARGS="\$CONTENT_CROSS_ATTN_ARGS --enable-content-cross-attn"
else
  CONTENT_CROSS_ATTN_ARGS="\$CONTENT_CROSS_ATTN_ARGS --no-enable-content-cross-attn"
fi
CONTENT_CROSS_ATTN_ARGS="\$CONTENT_CROSS_ATTN_ARGS --content-cross-attn-layers $CONTENT_CROSS_ATTN_LAYERS"
CONTENT_CROSS_ATTN_ARGS="\$CONTENT_CROSS_ATTN_ARGS --content-cross-attn-feature-dim $CONTENT_CROSS_ATTN_FEATURE_DIM"
CONTENT_CROSS_ATTN_ARGS="\$CONTENT_CROSS_ATTN_ARGS --content-cross-attn-gate-init $CONTENT_CROSS_ATTN_GATE_INIT"
CONTENT_CROSS_ATTN_ARGS="\$CONTENT_CROSS_ATTN_ARGS --content-cross-attn-dropout $CONTENT_CROSS_ATTN_DROPOUT"
CONTENT_CROSS_ATTN_ARGS="\$CONTENT_CROSS_ATTN_ARGS --content-cross-attn-output-scale $CONTENT_CROSS_ATTN_OUTPUT_SCALE"
CONTENT_CROSS_ATTN_ARGS="\$CONTENT_CROSS_ATTN_ARGS --content-encoder-layers $CONTENT_ENCODER_LAYERS"
CONTENT_CROSS_ATTN_ARGS="\$CONTENT_CROSS_ATTN_ARGS --content-encoder-conv-kernel-size $CONTENT_ENCODER_CONV_KERNEL_SIZE"
CONTENT_CROSS_ATTN_ARGS="\$CONTENT_CROSS_ATTN_ARGS --guided-attn-loss-weight $GUIDED_ATTN_LOSS_WEIGHT"
CONTENT_CROSS_ATTN_ARGS="\$CONTENT_CROSS_ATTN_ARGS --guided-attn-warmup-steps $GUIDED_ATTN_WARMUP_STEPS"
CONTENT_CROSS_ATTN_ARGS="\$CONTENT_CROSS_ATTN_ARGS --guided-attn-band-frames $GUIDED_ATTN_BAND_FRAMES"
CONTENT_CROSS_ATTN_ARGS="\$CONTENT_CROSS_ATTN_ARGS --phoneme-classifier-loss-weight $PHONEME_CLASSIFIER_LOSS_WEIGHT"
FREEZE_ARGS=""
if [ "$TRAIN_SOURCE_SEMANTIC_ONLY" = "1" ]; then
  FREEZE_ARGS="\$FREEZE_ARGS --train-source-semantic-only"
fi
if [ "$FREEZE_LORA" = "1" ]; then
  FREEZE_ARGS="\$FREEZE_ARGS --freeze-lora"
fi
if [ "$FREEZE_ROLE_ROUTING" = "1" ]; then
  FREEZE_ARGS="\$FREEZE_ARGS --freeze-role-routing"
fi
if [ "$FREEZE_TIMBRE_ADAPTER" = "1" ]; then
  FREEZE_ARGS="\$FREEZE_ARGS --freeze-timbre-adapter"
fi

SPEAKER_SIDE_ARGS=""
if [ "$ENABLE_SPEAKER_SIDE_PATHWAY" = "1" ]; then
  SPEAKER_SIDE_ARGS="\$SPEAKER_SIDE_ARGS --enable-speaker-side-pathway"
else
  SPEAKER_SIDE_ARGS="\$SPEAKER_SIDE_ARGS --no-enable-speaker-side-pathway"
fi
if [ "$SPEAKER_SIDE_PATHWAY_KV_BIAS" = "1" ]; then
  SPEAKER_SIDE_ARGS="\$SPEAKER_SIDE_ARGS --speaker-side-pathway-kv-bias"
else
  SPEAKER_SIDE_ARGS="\$SPEAKER_SIDE_ARGS --no-speaker-side-pathway-kv-bias"
fi
SPEAKER_SIDE_ARGS="\$SPEAKER_SIDE_ARGS --speaker-side-pathway-layers $SPEAKER_SIDE_PATHWAY_LAYERS"
SPEAKER_SIDE_ARGS="\$SPEAKER_SIDE_ARGS --speaker-side-pathway-gate-init $SPEAKER_SIDE_PATHWAY_GATE_INIT"
SPEAKER_SIDE_ARGS="\$SPEAKER_SIDE_ARGS --speaker-side-pathway-dropout $SPEAKER_SIDE_PATHWAY_DROPOUT"
if [ "$ENABLE_SPEAKER_CROSS_ATTN" = "1" ]; then
  SPEAKER_SIDE_ARGS="\$SPEAKER_SIDE_ARGS --enable-speaker-cross-attn"
else
  SPEAKER_SIDE_ARGS="\$SPEAKER_SIDE_ARGS --no-enable-speaker-cross-attn"
fi
SPEAKER_SIDE_ARGS="\$SPEAKER_SIDE_ARGS --speaker-cross-attn-layers $SPEAKER_CROSS_ATTN_LAYERS"
SPEAKER_SIDE_ARGS="\$SPEAKER_SIDE_ARGS --speaker-cross-attn-tokens $SPEAKER_CROSS_ATTN_TOKENS"
SPEAKER_SIDE_ARGS="\$SPEAKER_SIDE_ARGS --speaker-cross-attn-gate-init $SPEAKER_CROSS_ATTN_GATE_INIT"
SPEAKER_SIDE_ARGS="\$SPEAKER_SIDE_ARGS --speaker-cross-attn-dropout $SPEAKER_CROSS_ATTN_DROPOUT"
SPEAKER_SIDE_ARGS="\$SPEAKER_SIDE_ARGS --speaker-cross-attn-output-scale $SPEAKER_CROSS_ATTN_OUTPUT_SCALE"
SPEAKER_SIDE_ARGS="\$SPEAKER_SIDE_ARGS --speaker-cross-attn-source $SPEAKER_CROSS_ATTN_SOURCE"
SPEAKER_SIDE_ARGS="\$SPEAKER_SIDE_ARGS --speaker-cross-attn-seq-dim $SPEAKER_CROSS_ATTN_SEQ_DIM"
if [ -n "$SPEAKER_CROSS_ATTN_TOKEN_INIT_STD" ]; then
  SPEAKER_SIDE_ARGS="\$SPEAKER_SIDE_ARGS --speaker-cross-attn-token-init-std $SPEAKER_CROSS_ATTN_TOKEN_INIT_STD"
fi
SPEAKER_SIDE_ARGS="\$SPEAKER_SIDE_ARGS --speaker-cross-attn-alpha-warmup-steps $SPEAKER_CROSS_ATTN_ALPHA_WARMUP_STEPS"

"\$PY" -m accelerate.commands.launch \\
  --config_file "$ACCELERATE_CONFIG" \\
  scripts/002002_train_moss_codecvc_lora.py \\
  --config configs/remote_full.yaml \\
  --train-jsonl-spec "\$TRAIN_JSONL_SPEC" \\
  --output-dir "\$OUT_DIR" \\
  --version ver2 \\
  \$TIMBRE_MEMORY_ARGS \\
  --enable-role-routing \\
  --enable-target-head-routing \\
  --timbre-encoder-type conformer \\
  --timbre-encoder-layers 2 \\
  --timbre-speaker-conditioning \\
  --source-prosody-encoder-type conformer \\
  --source-prosody-encoder-layers 2 \\
  --prosody-memory-tokens 8 \\
  --source-prosody-no-text-gate "$SOURCE_PROSODY_NO_TEXT_GATE" \\
  --source-prosody-text-gate "$SOURCE_PROSODY_TEXT_GATE" \\
  --speaker-encoder-type "$SPEAKER_ENCODER_TYPE" \\
  --speaker-encoder-path "$SPEAKER_ENCODER_PATH" \\
  --speaker-embedding-dim "$SPEAKER_EMBEDDING_DIM" \\
  --target-speaker-similarity-weight "$TARGET_SPK_WEIGHT" \\
  --source-speaker-suppression-weight "$SOURCE_SUPPRESS_WEIGHT" \\
  --speaker-loss-warmup-steps "$SPEAKER_LOSS_WARMUP_STEPS" \\
  --speaker-loss-warmup-weight "$SPEAKER_LOSS_WARMUP_WEIGHT" \\
  --speaker-loss-schedule "$SPEAKER_LOSS_SCHEDULE" \\
  --speaker-loss-margin "$SPEAKER_LOSS_MARGIN" \\
  --ref-speaker-prompt-tokens "$REF_SPEAKER_PROMPT_TOKENS" \\
  --ref-speaker-prompt-dropout "$REF_SPEAKER_PROMPT_DROPOUT" \\
  --ref-speaker-prompt-mode "$REF_SPEAKER_PROMPT_MODE" \\
  --ref-speaker-prompt-token-source "$REF_SPEAKER_PROMPT_TOKEN_SOURCE" \\
  --ref-speaker-prompt-slot-code "$REF_SPEAKER_PROMPT_SLOT_CODE" \\
  --ref-speaker-prompt-slot-pack-mode "$REF_SPEAKER_PROMPT_SLOT_PACK_MODE" \\
  --ref-speaker-prompt-output-scale "$REF_SPEAKER_PROMPT_OUTPUT_SCALE" \\
  --ref-speaker-prompt-lr-multiplier "$REF_SPEAKER_PROMPT_LR_MULTIPLIER" \\
  $( [ "$REF_PROMPT_CODEC_PERMUTATION" = "1" ] && printf '%s' '--ref-prompt-codec-permutation' || printf '%s' '--no-ref-prompt-codec-permutation' ) \\
  --ref-prompt-codec-permutation-min-seconds "$REF_PROMPT_CODEC_PERMUTATION_MIN_SECONDS" \\
  --ref-prompt-codec-permutation-max-seconds "$REF_PROMPT_CODEC_PERMUTATION_MAX_SECONDS" \\
  --ref-prompt-codec-permutation-frame-rate "$REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE" \\
  --ref-prompt-codec-permutation-mode "$REF_PROMPT_CODEC_PERMUTATION_MODE" \\
  --ref-prompt-codec-permutation-block-seconds "$REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS" \\
  --ref-prompt-codec-permutation-seed "$REF_PROMPT_CODEC_PERMUTATION_SEED" \\
  --target-front-ce-weight "$TARGET_FRONT_CE_WEIGHT" \\
  --target-front-ce-seconds "$TARGET_FRONT_CE_SECONDS" \\
  --target-front-ce-frame-rate "$TARGET_FRONT_CE_FRAME_RATE" \\
  $( [ "$REF_SPEAKER_PROMPT_OUTPUT_NORM" = "1" ] && printf '%s' '--ref-speaker-prompt-output-norm' || printf '%s' '--no-ref-speaker-prompt-output-norm' ) \\
  $( [ "$REF_SPEAKER_PROMPT_SLOT" = "1" ] && printf '%s' '--ref-speaker-prompt-slot' || printf '%s' '--no-ref-speaker-prompt-slot' ) \\
  --ref-speaker-adaln-weight "$REF_SPEAKER_ADALN_WEIGHT" \\
  --speaker-infonce-weight "$SPEAKER_INFONCE_WEIGHT" \\
  --speaker-infonce-temperature "$SPEAKER_INFONCE_TEMPERATURE" \\
  --speaker-infonce-negative-pool-size "$SPEAKER_INFONCE_NEGATIVE_POOL_SIZE" \\
  --speaker-infonce-negative-pool-seed "$SPEAKER_INFONCE_NEGATIVE_POOL_SEED" \\
  --speaker-condition-dropout "$SPEAKER_CONDITION_DROPOUT" \\
  $( [ "$USE_PERTURBED_SOURCE_PROMPT" = "1" ] && printf '%s' '--use-perturbed-source-prompt' || printf '%s' '--no-use-perturbed-source-prompt' ) \\
  --lambda-route "$LAMBDA_ROUTE" \\
  --routing-gate-lr-multiplier "$ROUTING_GATE_LR_MULTIPLIER" \\
  --content-ctc-head-lr-multiplier "$CONTENT_CTC_HEAD_LR_MULTIPLIER" \\
  --timbre-adapter-init-gate "$TIMBRE_ADAPTER_INIT_GATE" \\
  --timbre-adapter-gate-lr-multiplier "$TIMBRE_ADAPTER_GATE_LR_MULTIPLIER" \\
  --lambda-prosody "$LAMBDA_PROSODY" \\
  --prosody-f0-weight "$PROSODY_F0_WEIGHT" \\
  --prosody-voiced-weight "$PROSODY_VOICED_WEIGHT" \\
  --prosody-energy-weight "$PROSODY_ENERGY_WEIGHT" \\
  --prosody-pause-weight "$PROSODY_PAUSE_WEIGHT" \\
  --prosody-duration-weight "$PROSODY_DURATION_WEIGHT" \\
  --lambda-content "$LAMBDA_CONTENT" \\
  --content-positive "$CONTENT_POSITIVE" \\
  --content-embedding-dim "$CONTENT_EMBEDDING_DIM" \\
  --content-embedding-weight "$CONTENT_EMBEDDING_WEIGHT" \\
  --content-ctc-weight "$CONTENT_CTC_WEIGHT" \\
  --content-ctc-vocab-size "$CONTENT_CTC_VOCAB_SIZE" \\
  --content-ctc-blank-id "$CONTENT_CTC_BLANK_ID" \\
  --content-ctc-token-offset "$CONTENT_CTC_TOKEN_OFFSET" \\
  --content-token-vocab-size "$CONTENT_TOKEN_VOCAB_SIZE" \\
  --content-token-weight "$CONTENT_TOKEN_WEIGHT" \\
  --content-source-codec-weight "$CONTENT_SOURCE_CODEC_WEIGHT" \\
  --content-source-codec-codebooks "$CONTENT_SOURCE_CODEC_CODEBOOKS" \\
  --semantic-loss-weight "$SEMANTIC_LOSS_WEIGHT" \\
  --semantic-mode "$SEMANTIC_MODE" \\
  --semantic-source "$SEMANTIC_SOURCE" \\
  --semantic-vocab-size "$SEMANTIC_VOCAB_SIZE" \\
  --semantic-feature-dim "$SEMANTIC_FEATURE_DIM" \\
  --semantic-feature-loss-type "$SEMANTIC_FEATURE_LOSS_TYPE" \\
  $( [ "$TIMBRE_SIDE_ONLY" = "1" ] && printf '%s' '--timbre-side-only' || printf '%s' '--no-timbre-side-only' ) \\
  --ref-content-suppression-weight "$REF_CONTENT_SUPPRESSION_WEIGHT" \\
  --ref-content-suppression-margin "$REF_CONTENT_SUPPRESSION_MARGIN" \\
  --ref-content-suppression-source "$REF_CONTENT_SUPPRESSION_SOURCE" \\
  $( [ "$REF_CONTENT_SUPPRESSION_DETACH_REF" = "1" ] && printf '%s' '--ref-content-suppression-detach-ref' || printf '%s' '--no-ref-content-suppression-detach-ref' ) \\
  --progress-loss-weight "$PROGRESS_LOSS_WEIGHT" \\
  --stop-loss-weight "$STOP_LOSS_WEIGHT" \\
  --progress-num-bins "$PROGRESS_NUM_BINS" \\
  \$SPEAKER_SIDE_ARGS \\
  \$SOURCE_SEMANTIC_ARGS \\
  \$CONTENT_CROSS_ATTN_ARGS \\
  \$FREEZE_ARGS \\
  --learning-rate "$LEARNING_RATE" \\
  --weight-decay "$WEIGHT_DECAY" \\
  --warmup-ratio "$WARMUP_RATIO" \\
  --lr-scheduler-type "$LR_SCHEDULER_TYPE" \\
  --per-device-batch-size "$PER_DEVICE_BATCH_SIZE" \\
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \\
  --num-epochs "$NUM_EPOCHS" \\
  --max-train-steps "$MAX_TRAIN_STEPS" \\
  --mixed-precision "$MIXED_PRECISION" \\
  --attn-implementation auto \\
  \$GRADIENT_CHECKPOINTING_ARGS \\
  --lora-r "$LORA_R" \\
  --lora-alpha "$LORA_ALPHA" \\
  --lora-dropout "$LORA_DROPOUT" \\
  \$RESUME_ARGS \\
  --trainable-lora-modules all \\
  --lm-heads-mode none \\
  --channelwise-loss-weight 1,32 \\
  --logging-steps "$LOGGING_STEPS" \\
  --save-steps "$SAVE_STEPS" \\
  --num-workers "$NUM_WORKERS" \\
  --max-grad-norm "$MAX_GRAD_NORM" \\
  \$EVAL_ARGS \\
  \$SMOKE_TEST_ARGS

if [ "$POST_TRAIN_QUICK_EVAL" = "1" ]; then
  echo "[qz-train] starting post-train quick eval"
  RUN_DIR="\$OUT_DIR" \\
  RUN_LABEL="$POST_TRAIN_EVAL_LABEL" \\
  EVAL_ROOT="$POST_TRAIN_EVAL_ROOT" \\
  DOCS_MD="$POST_TRAIN_EVAL_DOCS_MD" \\
  QUICK_VALIDATION_JSONL="$QUICK_VALIDATION_JSONL" \\
  DOMAIN_VALIDATION_JSONL="$DOMAIN_VALIDATION_JSONL" \\
  QUICK_GPU_COUNT="$POST_TRAIN_QUICK_GPU_COUNT" \\
  QUICK_NUM_SHARDS="$POST_TRAIN_QUICK_NUM_SHARDS" \\
	  QUICK_ASR_NUM_SHARDS="$POST_TRAIN_QUICK_ASR_NUM_SHARDS" \\
	  RUN_T11="$POST_TRAIN_RUN_T11" \\
	  TIMBRE_SIDE_ONLY="$TIMBRE_SIDE_ONLY" \\
	  REF_PROMPT_CODEC_PERMUTATION="$POST_TRAIN_REF_PROMPT_CODEC_PERMUTATION" \\
	  REF_PROMPT_CODEC_PERMUTATION_MODE="$POST_TRAIN_REF_PROMPT_CODEC_PERMUTATION_MODE" \\
	  REF_PROMPT_CODEC_PERMUTATION_MIN_SECONDS="$POST_TRAIN_REF_PROMPT_CODEC_PERMUTATION_MIN_SECONDS" \\
	  REF_PROMPT_CODEC_PERMUTATION_MAX_SECONDS="$POST_TRAIN_REF_PROMPT_CODEC_PERMUTATION_MAX_SECONDS" \\
	  REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE="$POST_TRAIN_REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE" \\
	  REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS="$POST_TRAIN_REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS" \\
	  REF_PROMPT_CODEC_PERMUTATION_BOOTSTRAP="$POST_TRAIN_REF_PROMPT_CODEC_PERMUTATION_BOOTSTRAP" \\
	  REF_SPEAKER_PROMPT_ATTENTION_CAPTURE_FRAMES="${REF_SPEAKER_PROMPT_ATTENTION_CAPTURE_FRAMES:-10}" \\
	  PYTHON="\$PY" \\
	  bash "$POST_TRAIN_EVAL_SCRIPT"
fi
EOF
chmod +x "$RUNNER"

COMMAND="sh $RUNNER"

echo "=========================================="
echo "QZ submit: MOSS-CodecVC Ver2.x LoRA"
echo "  JOB_NAME=$JOB_NAME"
echo "  ROOT=$ROOT"
echo "  TRAIN_JSONL_SPEC=$TRAIN_JSONL_SPEC"
echo "  NO_TEXT_TRAIN_JSONL=$NO_TEXT_TRAIN_JSONL"
echo "  TEXT_TRAIN_JSONL=$TEXT_TRAIN_JSONL"
echo "  TEXT_REPEAT=$TEXT_REPEAT"
echo "  OUT_DIR=$OUT_DIR"
echo "  NUM_EPOCHS=$NUM_EPOCHS"
echo "  MAX_TRAIN_STEPS=$MAX_TRAIN_STEPS"
echo "  GPU_COUNT=$GPU_COUNT"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_VALUE"
echo "  ACCELERATE_CONFIG=$ACCELERATE_CONFIG"
echo "  GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"
echo "  SMOKE_TEST=$SMOKE_TEST"
echo "  EVAL_JSONL=$EVAL_JSONL"
echo "  EVAL_JSONL_SPEC=$EVAL_JSONL_SPEC"
echo "  EVAL_SEEN_JSONL=$EVAL_SEEN_JSONL"
echo "  EVAL_SEEN_JSONL_SPEC=$EVAL_SEEN_JSONL_SPEC"
echo "  EVAL_UNSEEN_JSONL=$EVAL_UNSEEN_JSONL"
echo "  EVAL_UNSEEN_JSONL_SPEC=$EVAL_UNSEEN_JSONL_SPEC"
echo "  POST_TRAIN_QUICK_EVAL=$POST_TRAIN_QUICK_EVAL"
echo "  POST_TRAIN_EVAL_LABEL=$POST_TRAIN_EVAL_LABEL"
echo "  QUICK_VALIDATION_JSONL=$QUICK_VALIDATION_JSONL"
echo "  DOMAIN_VALIDATION_JSONL=$DOMAIN_VALIDATION_JSONL"
echo "  ENABLE_SOURCE_SEMANTIC_MEMORY=$ENABLE_SOURCE_SEMANTIC_MEMORY"
echo "  SOURCE_SEMANTIC_PROGRESS_WEIGHT=$SOURCE_SEMANTIC_PROGRESS_WEIGHT"
echo "  SOURCE_SEMANTIC_INIT_GATE=$SOURCE_SEMANTIC_INIT_GATE"
echo "  SOURCE_CONTENT_MEMORY_TYPE=$SOURCE_CONTENT_MEMORY_TYPE"
echo "  SOURCE_CONTENT_VOCAB_SIZE=$SOURCE_CONTENT_VOCAB_SIZE"
echo "  SOURCE_CONTENT_CODEC_BOTTLENECK_DIM=$SOURCE_CONTENT_CODEC_BOTTLENECK_DIM"
echo "  SOURCE_CONTENT_CODEC_CODEBOOKS=$SOURCE_CONTENT_CODEC_CODEBOOKS"
echo "  SOURCE_CODEC_RESIDUAL_MEMORY_WEIGHT=$SOURCE_CODEC_RESIDUAL_MEMORY_WEIGHT"
echo "  SOURCE_CODEC_RESIDUAL_MEMORY_DETACH=$SOURCE_CODEC_RESIDUAL_MEMORY_DETACH"
echo "  ENABLE_CONTENT_CROSS_ATTN=$ENABLE_CONTENT_CROSS_ATTN"
echo "  CONTENT_CROSS_ATTN_LAYERS=$CONTENT_CROSS_ATTN_LAYERS"
echo "  CONTENT_CROSS_ATTN_FEATURE_DIM=$CONTENT_CROSS_ATTN_FEATURE_DIM"
echo "  CONTENT_CROSS_ATTN_GATE_INIT=$CONTENT_CROSS_ATTN_GATE_INIT"
echo "  CONTENT_CROSS_ATTN_OUTPUT_SCALE=$CONTENT_CROSS_ATTN_OUTPUT_SCALE"
echo "  CONTENT_ENCODER_LAYERS=$CONTENT_ENCODER_LAYERS"
echo "  GUIDED_ATTN_LOSS_WEIGHT=$GUIDED_ATTN_LOSS_WEIGHT"
echo "  GUIDED_ATTN_WARMUP_STEPS=$GUIDED_ATTN_WARMUP_STEPS"
echo "  GUIDED_ATTN_BAND_FRAMES=$GUIDED_ATTN_BAND_FRAMES"
echo "  PHONEME_CLASSIFIER_LOSS_WEIGHT=$PHONEME_CLASSIFIER_LOSS_WEIGHT"
echo "  SOURCE_SEMANTIC_POSITION_SCALE=$SOURCE_SEMANTIC_POSITION_SCALE"
echo "  SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH=$SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH"
echo "  SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH=$SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH"
echo "  SOURCE_SEMANTIC_LR_MULTIPLIER=$SOURCE_SEMANTIC_LR_MULTIPLIER"
echo "  SOURCE_SEMANTIC_GATE_LR_MULTIPLIER=$SOURCE_SEMANTIC_GATE_LR_MULTIPLIER"
echo "  TIMBRE_SIDE_ONLY=$TIMBRE_SIDE_ONLY"
echo "  REF_SPEAKER_PROMPT_TOKENS=$REF_SPEAKER_PROMPT_TOKENS"
echo "  REF_SPEAKER_PROMPT_MODE=$REF_SPEAKER_PROMPT_MODE"
echo "  REF_SPEAKER_PROMPT_TOKEN_SOURCE=$REF_SPEAKER_PROMPT_TOKEN_SOURCE"
echo "  REF_SPEAKER_PROMPT_SLOT=$REF_SPEAKER_PROMPT_SLOT"
echo "  REF_SPEAKER_PROMPT_SLOT_CODE=$REF_SPEAKER_PROMPT_SLOT_CODE"
echo "  REF_SPEAKER_PROMPT_SLOT_PACK_MODE=$REF_SPEAKER_PROMPT_SLOT_PACK_MODE"
echo "  REF_SPEAKER_PROMPT_OUTPUT_NORM=$REF_SPEAKER_PROMPT_OUTPUT_NORM"
echo "  REF_SPEAKER_PROMPT_OUTPUT_SCALE=$REF_SPEAKER_PROMPT_OUTPUT_SCALE"
echo "  REF_SPEAKER_PROMPT_LR_MULTIPLIER=$REF_SPEAKER_PROMPT_LR_MULTIPLIER"
echo "  SPEAKER_ENCODER_TYPE=$SPEAKER_ENCODER_TYPE"
echo "  SPEAKER_ENCODER_PATH=$SPEAKER_ENCODER_PATH"
echo "  SPEAKER_EMBEDDING_DIM=$SPEAKER_EMBEDDING_DIM"
echo "  ENABLE_SPEAKER_SIDE_PATHWAY=$ENABLE_SPEAKER_SIDE_PATHWAY"
echo "  SPEAKER_SIDE_PATHWAY_LAYERS=$SPEAKER_SIDE_PATHWAY_LAYERS"
echo "  SPEAKER_SIDE_PATHWAY_KV_BIAS=$SPEAKER_SIDE_PATHWAY_KV_BIAS"
echo "  SPEAKER_SIDE_PATHWAY_GATE_INIT=$SPEAKER_SIDE_PATHWAY_GATE_INIT"
echo "  SPEAKER_SIDE_PATHWAY_DROPOUT=$SPEAKER_SIDE_PATHWAY_DROPOUT"
echo "  ENABLE_SPEAKER_CROSS_ATTN=$ENABLE_SPEAKER_CROSS_ATTN"
echo "  SPEAKER_CROSS_ATTN_LAYERS=$SPEAKER_CROSS_ATTN_LAYERS"
echo "  SPEAKER_CROSS_ATTN_TOKENS=$SPEAKER_CROSS_ATTN_TOKENS"
echo "  SPEAKER_CROSS_ATTN_GATE_INIT=$SPEAKER_CROSS_ATTN_GATE_INIT"
echo "  SPEAKER_CROSS_ATTN_DROPOUT=$SPEAKER_CROSS_ATTN_DROPOUT"
echo "  SPEAKER_CROSS_ATTN_OUTPUT_SCALE=$SPEAKER_CROSS_ATTN_OUTPUT_SCALE"
echo "  SPEAKER_CROSS_ATTN_TOKEN_INIT_STD=$SPEAKER_CROSS_ATTN_TOKEN_INIT_STD"
echo "  SPEAKER_CROSS_ATTN_ALPHA_WARMUP_STEPS=$SPEAKER_CROSS_ATTN_ALPHA_WARMUP_STEPS"
echo "  SPEAKER_CROSS_ATTN_SOURCE=$SPEAKER_CROSS_ATTN_SOURCE"
echo "  SPEAKER_CROSS_ATTN_SEQ_DIM=$SPEAKER_CROSS_ATTN_SEQ_DIM"
echo "  REF_PROMPT_CODEC_PERMUTATION=$REF_PROMPT_CODEC_PERMUTATION"
echo "  REF_PROMPT_CODEC_PERMUTATION_MIN_SECONDS=$REF_PROMPT_CODEC_PERMUTATION_MIN_SECONDS"
echo "  REF_PROMPT_CODEC_PERMUTATION_MAX_SECONDS=$REF_PROMPT_CODEC_PERMUTATION_MAX_SECONDS"
echo "  REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE=$REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE"
echo "  REF_PROMPT_CODEC_PERMUTATION_MODE=$REF_PROMPT_CODEC_PERMUTATION_MODE"
echo "  REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS=$REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS"
echo "  REF_PROMPT_CODEC_PERMUTATION_SEED=$REF_PROMPT_CODEC_PERMUTATION_SEED"
echo "  TARGET_FRONT_CE_WEIGHT=$TARGET_FRONT_CE_WEIGHT"
echo "  TARGET_FRONT_CE_SECONDS=$TARGET_FRONT_CE_SECONDS"
echo "  TARGET_FRONT_CE_FRAME_RATE=$TARGET_FRONT_CE_FRAME_RATE"
echo "  REF_CONTENT_SUPPRESSION_WEIGHT=$REF_CONTENT_SUPPRESSION_WEIGHT"
echo "  REF_CONTENT_SUPPRESSION_SOURCE=$REF_CONTENT_SUPPRESSION_SOURCE"
echo "  USE_TIMBRE_MEMORY=$USE_TIMBRE_MEMORY"
echo "  TIMBRE_MEMORY_TOKENS=$TIMBRE_MEMORY_TOKENS"
echo "  TIMBRE_ADAPTER_LAYERS=$TIMBRE_ADAPTER_LAYERS"
echo "  TIMBRE_ADAPTER_INIT_GATE=$TIMBRE_ADAPTER_INIT_GATE"
echo "  TIMBRE_ADAPTER_GATE_LR_MULTIPLIER=$TIMBRE_ADAPTER_GATE_LR_MULTIPLIER"
echo "  TRAIN_SOURCE_SEMANTIC_ONLY=$TRAIN_SOURCE_SEMANTIC_ONLY"
echo "  QZCLI=$QZCLI"
echo "  WORKSPACE=$WORKSPACE"
echo "  PROJECT=$PROJECT"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP"
echo "  SPEC=$SPEC"
echo "  PRIORITY=$PRIORITY"
echo "  IMAGE=$IMAGE"
echo "  SHM_GI=$SHM_GI"
echo "  RUNNER=$RUNNER"
echo "  COMMAND=$COMMAND"
echo "=========================================="

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry-run] Runner script generated but no QZ job was submitted."
  echo "[dry-run] To inspect runner:"
  echo "  sed -n '1,220p' $RUNNER"
  exit 0
fi

TMP_OUTPUT="$QZ_RECORD_ROOT/submit_output.txt"
rm -f "$TMP_OUTPUT"

set +e
env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
  "$QZCLI" create-job \
  --name "$JOB_NAME" \
  --workspace "$WORKSPACE" \
  --project "$PROJECT" \
  --compute-group "$COMPUTE_GROUP" \
  --spec "$SPEC" \
  --framework "$FRAMEWORK" \
  --instances "$INSTANCES" \
  --shm "$SHM_GI" \
  --priority "$PRIORITY" \
  --image "$IMAGE" \
  --image-type "$IMAGE_TYPE" \
  --command "$COMMAND" >"$TMP_OUTPUT" 2>&1
STATUS=$?
set -e

cat "$TMP_OUTPUT"

if [ "$STATUS" -ne 0 ]; then
  if grep -q 'Cookie 已过期或无效' "$TMP_OUTPUT"; then
    echo "Cookie expired; running qzcli login and retrying once." >&2
    env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
      "$QZCLI" login
    set +e
    env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
      "$QZCLI" create-job \
      --name "$JOB_NAME" \
      --workspace "$WORKSPACE" \
      --project "$PROJECT" \
      --compute-group "$COMPUTE_GROUP" \
      --spec "$SPEC" \
      --framework "$FRAMEWORK" \
      --instances "$INSTANCES" \
      --shm "$SHM_GI" \
      --priority "$PRIORITY" \
      --image "$IMAGE" \
      --image-type "$IMAGE_TYPE" \
      --command "$COMMAND" >"$TMP_OUTPUT" 2>&1
    STATUS=$?
    set -e
    cat "$TMP_OUTPUT"
  fi
fi

if [ "$STATUS" -ne 0 ]; then
  echo "Submission failed. Output saved to $TMP_OUTPUT" >&2
  if grep -q 'Cookie 已过期或无效' "$TMP_OUTPUT"; then
    echo "Fix: qzcli login" >&2
  fi
  exit "$STATUS"
fi

JOB_ID=$(grep -Eo 'job-[0-9a-fA-F-]{36}' "$TMP_OUTPUT" | tail -n 1 || true)
if [ -z "$JOB_ID" ]; then
  JOB_UUID=$(grep -E '任务ID|job_id|Job ID' "$TMP_OUTPUT" | grep -Eo '[0-9a-fA-F-]{36}' | tail -n 1 || true)
  if [ -n "$JOB_UUID" ]; then
    JOB_ID="job-$JOB_UUID"
  fi
fi

{
  printf 'job_name\tjob_id\tcompute_group\trunner\tout_dir\n'
  printf '%s\t%s\t%s\t%s\t%s\n' "$JOB_NAME" "${JOB_ID:-}" "$COMPUTE_GROUP" "$RUNNER" "$OUT_DIR"
} > "$QZ_RECORD_ROOT/submitted_jobs.tsv"

echo "=========================================="
echo "Submission completed."
echo "  JOB_NAME=$JOB_NAME"
echo "  JOB_ID=${JOB_ID:-not parsed}"
echo "  RECORD=$QZ_RECORD_ROOT/submitted_jobs.tsv"
echo "  TENSORBOARD=$OUT_DIR/tensorboard"
echo "=========================================="

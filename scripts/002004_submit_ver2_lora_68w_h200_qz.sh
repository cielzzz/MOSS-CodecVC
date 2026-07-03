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
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_fsdp_h200_8gpu_no_ckpt.yaml}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
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
EVAL_STEPS="${EVAL_STEPS:-0}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"

TARGET_SPK_WEIGHT="${TARGET_SPK_WEIGHT:-0.05}"
SOURCE_SUPPRESS_WEIGHT="${SOURCE_SUPPRESS_WEIGHT:-0.05}"
SPEAKER_LOSS_WARMUP_STEPS="${SPEAKER_LOSS_WARMUP_STEPS:-1000}"
SPEAKER_LOSS_WARMUP_WEIGHT="${SPEAKER_LOSS_WARMUP_WEIGHT:-0.02}"
SPEAKER_LOSS_MARGIN="${SPEAKER_LOSS_MARGIN:-0.10}"
LAMBDA_ROUTE="${LAMBDA_ROUTE:-0.01}"
ROUTING_GATE_LR_MULTIPLIER="${ROUTING_GATE_LR_MULTIPLIER:-10.0}"
CONTENT_CTC_HEAD_LR_MULTIPLIER="${CONTENT_CTC_HEAD_LR_MULTIPLIER:-1.0}"
TIMBRE_ADAPTER_INIT_GATE="${TIMBRE_ADAPTER_INIT_GATE:--4.0}"
TIMBRE_ADAPTER_GATE_LR_MULTIPLIER="${TIMBRE_ADAPTER_GATE_LR_MULTIPLIER:-1.0}"
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
SOURCE_SEMANTIC_ADAPTER_LAYERS="${SOURCE_SEMANTIC_ADAPTER_LAYERS:-28,30,32,34,35}"
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
fi

mkdir -p "$QZ_RECORD_ROOT"

cat > "$RUNNER" <<EOF
#!/bin/sh
set -eu

export ROOT="$ROOT"
export PY="$PY"
export DOWNLOAD_ROOT="$DOWNLOAD_ROOT"
export HF_HOME="\$DOWNLOAD_ROOT/huggingface"
export TRANSFORMERS_CACHE="\$DOWNLOAD_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="\$DOWNLOAD_ROOT/huggingface/hub"
export HF_DATASETS_CACHE="\$DOWNLOAD_ROOT/huggingface/datasets"
export TORCH_HOME="\$DOWNLOAD_ROOT/torch"
export XDG_CACHE_HOME="\$DOWNLOAD_ROOT/cache"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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
echo "[qz-train] eval_jsonl=\$EVAL_JSONL eval_jsonl_spec=\$EVAL_JSONL_SPEC eval_steps=\$EVAL_STEPS eval_max_batches=\$EVAL_MAX_BATCHES"
echo "[qz-train] global_batch_size=$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * 8))"
echo "[qz-train] routing_gate_lr_multiplier=$ROUTING_GATE_LR_MULTIPLIER"
echo "[qz-train] content_ctc_head_lr_multiplier=$CONTENT_CTC_HEAD_LR_MULTIPLIER"
echo "[qz-train] timbre_adapter_init_gate=$TIMBRE_ADAPTER_INIT_GATE timbre_adapter_gate_lr_multiplier=$TIMBRE_ADAPTER_GATE_LR_MULTIPLIER"
echo "[qz-train] lambda_content=$LAMBDA_CONTENT content_source_codec_weight=$CONTENT_SOURCE_CODEC_WEIGHT content_token_vocab_size=$CONTENT_TOKEN_VOCAB_SIZE content_ctc_weight=$CONTENT_CTC_WEIGHT semantic_loss_weight=$SEMANTIC_LOSS_WEIGHT"
echo "[qz-train] progress_loss_weight=$PROGRESS_LOSS_WEIGHT stop_loss_weight=$STOP_LOSS_WEIGHT progress_num_bins=$PROGRESS_NUM_BINS"
echo "[qz-train] source_semantic_memory=$ENABLE_SOURCE_SEMANTIC_MEMORY source_semantic_layers=$SOURCE_SEMANTIC_ADAPTER_LAYERS source_semantic_progress_weight=$SOURCE_SEMANTIC_PROGRESS_WEIGHT source_semantic_lr_multiplier=$SOURCE_SEMANTIC_LR_MULTIPLIER source_semantic_gate_lr_multiplier=$SOURCE_SEMANTIC_GATE_LR_MULTIPLIER"
echo "[qz-train] source_content_memory_type=$SOURCE_CONTENT_MEMORY_TYPE source_content_vocab_size=$SOURCE_CONTENT_VOCAB_SIZE source_content_codec_bottleneck_dim=$SOURCE_CONTENT_CODEC_BOTTLENECK_DIM source_content_codec_codebooks=$SOURCE_CONTENT_CODEC_CODEBOOKS source_content_dedup_units=$SOURCE_CONTENT_DEDUP_UNITS source_codec_residual_memory_weight=$SOURCE_CODEC_RESIDUAL_MEMORY_WEIGHT source_codec_residual_memory_detach=$SOURCE_CODEC_RESIDUAL_MEMORY_DETACH"
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

GRADIENT_CHECKPOINTING_ARGS=""
if [ "$GRADIENT_CHECKPOINTING" = "1" ]; then
  GRADIENT_CHECKPOINTING_ARGS="--gradient-checkpointing"
fi

EVAL_ARGS=""
if [ -n "\$EVAL_JSONL" ]; then
  EVAL_ARGS="\$EVAL_ARGS --eval-jsonl \$EVAL_JSONL"
fi
if [ -n "\$EVAL_JSONL_SPEC" ]; then
  EVAL_ARGS="\$EVAL_ARGS --eval-jsonl-spec \$EVAL_JSONL_SPEC"
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
SOURCE_SEMANTIC_ARGS="\$SOURCE_SEMANTIC_ARGS --source-semantic-adapter-layers $SOURCE_SEMANTIC_ADAPTER_LAYERS"
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

"\$PY" -m accelerate.commands.launch \\
  --config_file "$ACCELERATE_CONFIG" \\
  scripts/002002_train_moss_codecvc_lora.py \\
  --config configs/remote_full.yaml \\
  --train-jsonl-spec "\$TRAIN_JSONL_SPEC" \\
  --output-dir "\$OUT_DIR" \\
  --version ver2 \\
  --use-timbre-memory \\
  --enable-role-routing \\
  --enable-target-head-routing \\
  --timbre-encoder-type conformer \\
  --timbre-encoder-layers 2 \\
  --timbre-memory-tokens 16 \\
  --timbre-speaker-conditioning \\
  --source-prosody-encoder-type conformer \\
  --source-prosody-encoder-layers 2 \\
  --prosody-memory-tokens 8 \\
  --source-prosody-no-text-gate "$SOURCE_PROSODY_NO_TEXT_GATE" \\
  --source-prosody-text-gate "$SOURCE_PROSODY_TEXT_GATE" \\
  --speaker-encoder-type embedding_loader \\
  --speaker-embedding-dim 192 \\
  --target-speaker-similarity-weight "$TARGET_SPK_WEIGHT" \\
  --source-speaker-suppression-weight "$SOURCE_SUPPRESS_WEIGHT" \\
  --speaker-loss-warmup-steps "$SPEAKER_LOSS_WARMUP_STEPS" \\
  --speaker-loss-warmup-weight "$SPEAKER_LOSS_WARMUP_WEIGHT" \\
  --speaker-loss-margin "$SPEAKER_LOSS_MARGIN" \\
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
  \$SOURCE_SEMANTIC_ARGS \\
  \$FREEZE_ARGS \\
  --learning-rate "$LEARNING_RATE" \\
  --weight-decay "$WEIGHT_DECAY" \\
  --warmup-ratio "$WARMUP_RATIO" \\
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
  \$EVAL_ARGS
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
echo "  EVAL_JSONL=$EVAL_JSONL"
echo "  EVAL_JSONL_SPEC=$EVAL_JSONL_SPEC"
echo "  ENABLE_SOURCE_SEMANTIC_MEMORY=$ENABLE_SOURCE_SEMANTIC_MEMORY"
echo "  SOURCE_SEMANTIC_PROGRESS_WEIGHT=$SOURCE_SEMANTIC_PROGRESS_WEIGHT"
echo "  SOURCE_SEMANTIC_INIT_GATE=$SOURCE_SEMANTIC_INIT_GATE"
echo "  SOURCE_CONTENT_MEMORY_TYPE=$SOURCE_CONTENT_MEMORY_TYPE"
echo "  SOURCE_CONTENT_VOCAB_SIZE=$SOURCE_CONTENT_VOCAB_SIZE"
echo "  SOURCE_CONTENT_CODEC_BOTTLENECK_DIM=$SOURCE_CONTENT_CODEC_BOTTLENECK_DIM"
echo "  SOURCE_CONTENT_CODEC_CODEBOOKS=$SOURCE_CONTENT_CODEC_CODEBOOKS"
echo "  SOURCE_CODEC_RESIDUAL_MEMORY_WEIGHT=$SOURCE_CODEC_RESIDUAL_MEMORY_WEIGHT"
echo "  SOURCE_CODEC_RESIDUAL_MEMORY_DETACH=$SOURCE_CODEC_RESIDUAL_MEMORY_DETACH"
echo "  SOURCE_SEMANTIC_POSITION_SCALE=$SOURCE_SEMANTIC_POSITION_SCALE"
echo "  SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH=$SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH"
echo "  SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH=$SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH"
echo "  SOURCE_SEMANTIC_LR_MULTIPLIER=$SOURCE_SEMANTIC_LR_MULTIPLIER"
echo "  SOURCE_SEMANTIC_GATE_LR_MULTIPLIER=$SOURCE_SEMANTIC_GATE_LR_MULTIPLIER"
echo "  TIMBRE_SIDE_ONLY=$TIMBRE_SIDE_ONLY"
echo "  REF_CONTENT_SUPPRESSION_WEIGHT=$REF_CONTENT_SUPPRESSION_WEIGHT"
echo "  REF_CONTENT_SUPPRESSION_SOURCE=$REF_CONTENT_SUPPRESSION_SOURCE"
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

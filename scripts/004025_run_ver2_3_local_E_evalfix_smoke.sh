#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_fsdp_4090_2gpu_smoke.yaml}"

NO_TEXT_DATASET_NAME="${NO_TEXT_DATASET_NAME:-zh45w_en22w_no_text}"
TEXT_DATASET_NAME="${TEXT_DATASET_NAME:-zh3w_en3w_text_prosody_independent_timbre}"
NO_TEXT_TRAIN_JSONL="${NO_TEXT_TRAIN_JSONL:-$ROOT/trainset/$NO_TEXT_DATASET_NAME/sft/moss_codecvc_sft.$NO_TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_asr_filter.with_hubert.with_spm_content_tokens.ctc_clean.jsonl}"
TEXT_TRAIN_JSONL="${TEXT_TRAIN_JSONL:-$ROOT/trainset/$TEXT_DATASET_NAME/sft/moss_codecvc_sft.$TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_target_asr.with_content_tokens.with_target_hubert.with_spm_content_tokens.ctc_clean.jsonl}"
VALID_JSONL="${VALID_JSONL:-$ROOT/testset/validation/ver2_3_debug/moss_codecvc_ver2_3_loss_valid_160.jsonl}"
RESUME_ADAPTER_PATH="${RESUME_ADAPTER_PATH:-$ROOT/outputs/lora_runs/ver2_3_debug_resume_verified/ablation_b_ctc/step-500}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/lora_runs/ver2_3_debug_local_E_evalfix_smoke_$RUN_STAMP}"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-40}"
SAVE_STEPS="${SAVE_STEPS:-20}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-2}"
NO_TEXT_MAX_ROWS="${NO_TEXT_MAX_ROWS:-64}"
TEXT_MAX_ROWS="${TEXT_MAX_ROWS:-64}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
DRY_RUN=0

usage() {
  cat >&2 <<EOF_USAGE
Usage: bash $0 [--dry-run]

Local E smoke: resume ablation_b_ctc, run on local 4090, save/eval every 20 steps by default.
Important env overrides:
  OUT_DIR=...                 default: timestamped ver2_3_debug_local_E_evalfix_smoke_*
  RESUME_ADAPTER_PATH=...     default: verified ablation_b_ctc/step-500
  MAX_TRAIN_STEPS=40          SAVE_STEPS=20
  CUDA_VISIBLE_DEVICES=0,1    EVAL_MAX_BATCHES=2
EOF_USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

require_file() {
  if [ ! -f "$1" ]; then
    echo "ERROR: missing required file: $1" >&2
    exit 1
  fi
}

require_file "$PY"
require_file "$NO_TEXT_TRAIN_JSONL"
require_file "$TEXT_TRAIN_JSONL"
require_file "$VALID_JSONL"
require_file "$RESUME_ADAPTER_PATH/adapter_model.safetensors"
require_file "$RESUME_ADAPTER_PATH/timbre_memory_adapter.pt"
require_file "$ROOT/$ACCELERATE_CONFIG"

TRAIN_JSONL_SPEC="$NO_TEXT_TRAIN_JSONL::repeat=1::max_rows=$NO_TEXT_MAX_ROWS,$TEXT_TRAIN_JSONL::repeat=1::max_rows=$TEXT_MAX_ROWS"

export ROOT PY DOWNLOAD_ROOT
export HF_HOME="$DOWNLOAD_ROOT/huggingface"
export TRANSFORMERS_CACHE="$DOWNLOAD_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$DOWNLOAD_ROOT/huggingface/hub"
export HF_DATASETS_CACHE="$DOWNLOAD_ROOT/huggingface/datasets"
export TORCH_HOME="$DOWNLOAD_ROOT/torch"
export XDG_CACHE_HOME="$DOWNLOAD_ROOT/cache"
export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

cd "$ROOT"
mkdir -p "$OUT_DIR"

cmd=(
  "$PY" -m accelerate.commands.launch
  --config_file "$ACCELERATE_CONFIG"
  scripts/002002_train_moss_codecvc_lora.py
  --config configs/remote_full.yaml
  --train-jsonl-spec "$TRAIN_JSONL_SPEC"
  --eval-jsonl "$VALID_JSONL"
  --eval-steps 0
  --eval-max-batches "$EVAL_MAX_BATCHES"
  --eval-num-workers 0
  --output-dir "$OUT_DIR"
  --version ver2
  --use-timbre-memory
  --enable-role-routing
  --enable-target-head-routing
  --timbre-encoder-type conformer
  --timbre-encoder-layers 2
  --timbre-memory-tokens 16
  --timbre-speaker-conditioning
  --source-prosody-encoder-type conformer
  --source-prosody-encoder-layers 2
  --prosody-memory-tokens 8
  --speaker-encoder-type embedding_loader
  --speaker-embedding-dim 192
  --target-speaker-similarity-weight 0.0
  --source-speaker-suppression-weight 0.0
  --speaker-loss-warmup-steps 1000
  --speaker-loss-warmup-weight 0.02
  --speaker-loss-margin 0.10
  --lambda-route 0.01
  --routing-gate-lr-multiplier 10.0
  --lambda-prosody 0.0
  --prosody-f0-weight 0.0
  --prosody-voiced-weight 0.0
  --prosody-energy-weight 0.5
  --prosody-pause-weight 1.0
  --prosody-duration-weight 0.5
  --lambda-content 0.0
  --content-positive source
  --content-embedding-dim 0
  --content-embedding-weight 1.0
  --content-ctc-weight 0.02
  --content-ctc-vocab-size 0
  --content-ctc-blank-id 0
  --content-ctc-token-offset 1
  --content-token-vocab-size 0
  --content-token-weight 0.0
  --content-source-codec-weight 0.0
  --content-source-codec-codebooks 0,1,2,3
  --semantic-loss-weight 0.0
  --semantic-mode continuous
  --semantic-source mode_aware
  --semantic-vocab-size 0
  --semantic-feature-dim 0
  --semantic-feature-loss-type cosine
  --progress-loss-weight 0.0
  --stop-loss-weight 0.0
  --progress-num-bins 32
  --learning-rate 1e-5
  --weight-decay 0.01
  --warmup-ratio 0.03
  --per-device-batch-size 1
  --gradient-accumulation-steps 1
  --max-train-steps "$MAX_TRAIN_STEPS"
  --mixed-precision bf16
  --attn-implementation auto
  --lora-r 16
  --lora-alpha 32
  --lora-dropout 0.05
  --resume-adapter-path "$RESUME_ADAPTER_PATH"
  --trainable-lora-modules all
  --lm-heads-mode none
  --channelwise-loss-weight 1,32
  --logging-steps "$LOGGING_STEPS"
  --save-steps "$SAVE_STEPS"
  --num-workers 0
  --max-grad-norm 1.0
)

echo "[local-E] root=$ROOT"
echo "[local-E] out_dir=$OUT_DIR"
echo "[local-E] resume=$RESUME_ADAPTER_PATH"
echo "[local-E] max_train_steps=$MAX_TRAIN_STEPS save_steps=$SAVE_STEPS eval_max_batches=$EVAL_MAX_BATCHES cuda=$CUDA_VISIBLE_DEVICES"
echo "[local-E] train_jsonl_spec=$TRAIN_JSONL_SPEC"

if [ "$DRY_RUN" = "1" ]; then
  printf '[dry-run]'; printf ' %q' "${cmd[@]}"; printf '\n'
  exit 0
fi

nvidia-smi || true
"${cmd[@]}" 2>&1 | tee "$OUT_DIR/local_E.log"

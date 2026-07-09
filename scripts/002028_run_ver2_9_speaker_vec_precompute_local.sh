#!/bin/sh
# Run ver2.9 WavLM-SV speaker-vector precompute locally.
#
# This is intended for resumable offline data preparation on local GPUs. It
# reuses existing .npy vectors when OVERWRITE=0, then merges shard manifests
# and verifies the prepared directory.

set -eu

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download}"
INPUT_PREPARED_DIR="${INPUT_PREPARED_DIR:-$ROOT/trainset/ver2_8_prepared_speaker_split_20260705}"
OUTPUT_PREPARED_DIR="${OUTPUT_PREPARED_DIR:-$ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_20260707}"
SPEAKER_VEC_DIR="${SPEAKER_VEC_DIR:-$OUTPUT_PREPARED_DIR/speaker_vecs}"
SPEAKER_ENCODER_PATH="${SPEAKER_ENCODER_PATH:-microsoft/wavlm-base-plus-sv}"
SPEAKER_EMBEDDING_DIM="${SPEAKER_EMBEDDING_DIM:-512}"
NUM_SHARDS="${NUM_SHARDS:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
OVERWRITE="${OVERWRITE:-0}"
TEXT_REPEAT="${TEXT_REPEAT:-10}"
GPUS="${GPUS:-0 1}"
DEFAULT_SPLITS="no_text.train.jsonl text.train.jsonl no_text.valid.jsonl text.valid.jsonl no_text.seen_valid.jsonl text.seen_valid.jsonl no_text.unseen_valid.jsonl text.unseen_valid.jsonl"
SPLITS="${SPLITS:-$DEFAULT_SPLITS}"
SPLIT_STAGES="${SPLIT_STAGES:-}"
if [ -z "$SPLIT_STAGES" ]; then
  if [ "$SPLITS" = "$DEFAULT_SPLITS" ]; then
    SPLIT_STAGES="no_text.train.jsonl|text.train.jsonl|no_text.valid.jsonl text.valid.jsonl no_text.seen_valid.jsonl text.seen_valid.jsonl no_text.unseen_valid.jsonl text.unseen_valid.jsonl"
  else
    SPLIT_STAGES="$SPLITS"
  fi
fi
BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$OUTPUT_PREPARED_DIR/logs/local_resume_$BATCH_ID}"

export ROOT PY DOWNLOAD_ROOT INPUT_PREPARED_DIR OUTPUT_PREPARED_DIR SPEAKER_VEC_DIR
export SPEAKER_ENCODER_PATH SPEAKER_EMBEDDING_DIM NUM_SHARDS BATCH_SIZE OVERWRITE TEXT_REPEAT SPLITS
export HF_HOME="$DOWNLOAD_ROOT/huggingface"
export TRANSFORMERS_CACHE="$DOWNLOAD_ROOT/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="$DOWNLOAD_ROOT/huggingface/hub"
export HF_DATASETS_CACHE="$DOWNLOAD_ROOT/huggingface/datasets"
export TORCH_HOME="$DOWNLOAD_ROOT/torch"
export XDG_CACHE_HOME="$DOWNLOAD_ROOT/cache"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LOCAL_FILES_ONLY=1

mkdir -p "$LOG_DIR" "$OUTPUT_PREPARED_DIR" "$SPEAKER_VEC_DIR"

echo "[local-speaker-vec] start date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[local-speaker-vec] root=$ROOT"
echo "[local-speaker-vec] input=$INPUT_PREPARED_DIR"
echo "[local-speaker-vec] output=$OUTPUT_PREPARED_DIR"
echo "[local-speaker-vec] log_dir=$LOG_DIR"
echo "[local-speaker-vec] shards=$NUM_SHARDS gpus=$GPUS batch_size=$BATCH_SIZE overwrite=$OVERWRITE"
echo "[local-speaker-vec] splits=$SPLITS"
echo "[local-speaker-vec] split_stages=$SPLIT_STAGES"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits || true

cd "$ROOT"

run_shard() {
  idx="$1"
  gpu="$2"
  stage_idx="$3"
  stage_splits="$4"
  log="$LOG_DIR/stage_${stage_idx}_shard_$(printf '%02d' "$idx").log"
  echo "[local-speaker-vec] launching stage=$stage_idx shard=$idx gpu=$gpu splits=$stage_splits log=$log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export DEVICE=cuda
    export SHARD_INDEX="$idx"
    export SPLITS="$stage_splits"
    sh "$ROOT/scripts/002023_prepare_ver2_9_speaker_vecs.sh"
  ) > "$log" 2>&1 &
  SHARD_PID="$!"
}

old_ifs="$IFS"
IFS='|'
set -- $SPLIT_STAGES
IFS="$old_ifs"
stage_idx=0
for stage_splits do
  stage_idx=$((stage_idx + 1))
  echo "[local-speaker-vec] stage=$stage_idx splits=$stage_splits start date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  idx=0
  while [ "$idx" -lt "$NUM_SHARDS" ]; do
    pids=""
    for gpu in $GPUS; do
      if [ "$idx" -ge "$NUM_SHARDS" ]; then
        break
      fi
      run_shard "$idx" "$gpu" "$stage_idx" "$stage_splits"
      pids="$pids $SHARD_PID"
      idx=$((idx + 1))
    done

    status=0
    for pid in $pids; do
      if ! wait "$pid"; then
        status=1
      fi
    done
    if [ "$status" != "0" ]; then
      echo "[local-speaker-vec] stage=$stage_idx shard batch failed; see $LOG_DIR" >&2
      exit 1
    fi
    echo "[local-speaker-vec] stage=$stage_idx completed up_to_shard=$idx date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  done
  echo "[local-speaker-vec] stage=$stage_idx done date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

echo "[local-speaker-vec] merging date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SHARD_INDEX=merge MERGE_SHARDS=1 sh "$ROOT/scripts/002023_prepare_ver2_9_speaker_vecs.sh" > "$LOG_DIR/merge.log" 2>&1
find "$OUTPUT_PREPARED_DIR" -maxdepth 1 -name "*.jsonl" -printf "%f\n" | sort > "$LOG_DIR/final_manifests.txt"

echo "[local-speaker-vec] verifying date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" "$ROOT/scripts/002027_verify_ver2_9_speaker_vecs.py" \
  --prepared-dir "$OUTPUT_PREPARED_DIR" \
  --input-prepared-dir "$INPUT_PREPARED_DIR" \
  --expected-dim 512 \
  --sample-per-split 200 > "$LOG_DIR/verify.log" 2>&1

echo "[local-speaker-vec] sanity_check date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" "$ROOT/scripts/002029_sanity_check_ver2_9_speaker_vecs.py" \
  --prepared-dir "$OUTPUT_PREPARED_DIR" \
  --split no_text.train.jsonl \
  --sample-size 100 \
  --seed 20260707 \
  --min-delta 0.15 > "$LOG_DIR/sanity_no_text_train_100.log" 2>&1

echo "[local-speaker-vec] done date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

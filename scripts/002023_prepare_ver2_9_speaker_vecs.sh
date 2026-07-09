#!/bin/sh
# Prepare ver2.9 speaker-side manifests by adding WavLM-SV speaker_vec_path.

set -eu

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
INPUT_PREPARED_DIR="${INPUT_PREPARED_DIR:-$ROOT/trainset/ver2_8_prepared_speaker_split_20260705}"
OUTPUT_PREPARED_DIR="${OUTPUT_PREPARED_DIR:-$ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_20260707}"
SPEAKER_VEC_DIR="${SPEAKER_VEC_DIR:-$OUTPUT_PREPARED_DIR/speaker_vecs}"
SPEAKER_ENCODER_PATH="${SPEAKER_ENCODER_PATH:-microsoft/wavlm-base-plus-sv}"
SPEAKER_EMBEDDING_DIM="${SPEAKER_EMBEDDING_DIM:-512}"
DEVICE="${DEVICE:-cuda}"
MAX_ROWS="${MAX_ROWS:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
OVERWRITE="${OVERWRITE:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-}"
MERGE_SHARDS="${MERGE_SHARDS:-0}"
SPLITS="${SPLITS:-no_text.train.jsonl text.train.jsonl no_text.valid.jsonl text.valid.jsonl no_text.seen_valid.jsonl text.seen_valid.jsonl no_text.unseen_valid.jsonl text.unseen_valid.jsonl}"

mkdir -p "$OUTPUT_PREPARED_DIR" "$SPEAKER_VEC_DIR"

extra_args=""
if [ "$OVERWRITE" = "1" ]; then
  extra_args="$extra_args --overwrite"
fi
if [ "$LOCAL_FILES_ONLY" = "1" ]; then
  extra_args="$extra_args --local-files-only"
fi
if [ "$MAX_ROWS" != "0" ]; then
  extra_args="$extra_args --max-rows $MAX_ROWS"
fi
extra_args="$extra_args --batch-size $BATCH_SIZE"

shard_suffix() {
  idx="$1"
  printf '.shard%05d-of%05d' "$idx" "$NUM_SHARDS"
}

merge_split_shards() {
  split="$1"
  final_jsonl="$OUTPUT_PREPARED_DIR/$split"
  tmp_jsonl="$final_jsonl.tmp"
  : > "$tmp_jsonl"
  idx=0
  while [ "$idx" -lt "$NUM_SHARDS" ]; do
    part_jsonl="$OUTPUT_PREPARED_DIR/$split$(shard_suffix "$idx").jsonl"
    if [ ! -f "$part_jsonl" ]; then
      echo "[ver2.9-speaker-vec] missing shard for merge: $part_jsonl" >&2
      rm -f "$tmp_jsonl"
      return 1
    fi
    cat "$part_jsonl" >> "$tmp_jsonl"
    idx=$((idx + 1))
  done
  mv "$tmp_jsonl" "$final_jsonl"
  echo "[ver2.9-speaker-vec] merged split=$split output=$final_jsonl"
}

if [ "$SHARD_INDEX" != "merge" ]; then
  for split in $SPLITS; do
    input_jsonl="$INPUT_PREPARED_DIR/$split"
    output_jsonl="$OUTPUT_PREPARED_DIR/$split"
    shard_args=""
    if [ -n "$SHARD_INDEX" ]; then
      output_jsonl="$OUTPUT_PREPARED_DIR/$split$(shard_suffix "$SHARD_INDEX").jsonl"
      shard_args="--shard-index $SHARD_INDEX --num-shards $NUM_SHARDS"
    fi
    if [ ! -f "$input_jsonl" ]; then
      echo "[ver2.9-speaker-vec] skip missing split: $input_jsonl" >&2
      continue
    fi
    echo "[ver2.9-speaker-vec] split=$split input=$input_jsonl output=$output_jsonl shard=${SHARD_INDEX:-all}/$NUM_SHARDS batch_size=$BATCH_SIZE"
    "$PY" "$ROOT/scripts/002022_precompute_ver2_9_speaker_vecs.py" \
      --input-jsonl "$input_jsonl" \
      --output-jsonl "$output_jsonl" \
      --speaker-vec-dir "$SPEAKER_VEC_DIR" \
      --speaker-encoder-path "$SPEAKER_ENCODER_PATH" \
      --speaker-embedding-dim "$SPEAKER_EMBEDDING_DIM" \
      --device "$DEVICE" \
      $shard_args \
      $extra_args
  done
fi

if { [ "$MERGE_SHARDS" = "1" ] || [ "$SHARD_INDEX" = "merge" ]; } && [ "$NUM_SHARDS" -gt 1 ]; then
  for split in $SPLITS; do
    input_jsonl="$INPUT_PREPARED_DIR/$split"
    if [ ! -f "$input_jsonl" ]; then
      continue
    fi
    merge_split_shards "$split"
  done
fi

if [ -f "$OUTPUT_PREPARED_DIR/no_text.train.jsonl" ] && [ -f "$OUTPUT_PREPARED_DIR/text.train.jsonl" ]; then
  {
    printf '%s::repeat=1\n' "$OUTPUT_PREPARED_DIR/no_text.train.jsonl"
    printf '%s::repeat=%s\n' "$OUTPUT_PREPARED_DIR/text.train.jsonl" "${TEXT_REPEAT:-10}"
  } > "$OUTPUT_PREPARED_DIR/mixed.train.spec.txt"
fi
if [ -f "$OUTPUT_PREPARED_DIR/no_text.valid.jsonl" ] && [ -f "$OUTPUT_PREPARED_DIR/text.valid.jsonl" ]; then
  {
    printf '%s::repeat=1\n' "$OUTPUT_PREPARED_DIR/no_text.valid.jsonl"
    printf '%s::repeat=1\n' "$OUTPUT_PREPARED_DIR/text.valid.jsonl"
  } > "$OUTPUT_PREPARED_DIR/mixed.valid.spec.txt"
fi
if [ -f "$OUTPUT_PREPARED_DIR/no_text.seen_valid.jsonl" ] && [ -f "$OUTPUT_PREPARED_DIR/text.seen_valid.jsonl" ]; then
  {
    printf '%s::repeat=1\n' "$OUTPUT_PREPARED_DIR/no_text.seen_valid.jsonl"
    printf '%s::repeat=1\n' "$OUTPUT_PREPARED_DIR/text.seen_valid.jsonl"
  } > "$OUTPUT_PREPARED_DIR/mixed.valid_seen.spec.txt"
fi
if [ -f "$OUTPUT_PREPARED_DIR/no_text.unseen_valid.jsonl" ] && [ -f "$OUTPUT_PREPARED_DIR/text.unseen_valid.jsonl" ]; then
  {
    printf '%s::repeat=1\n' "$OUTPUT_PREPARED_DIR/no_text.unseen_valid.jsonl"
    printf '%s::repeat=1\n' "$OUTPUT_PREPARED_DIR/text.unseen_valid.jsonl"
  } > "$OUTPUT_PREPARED_DIR/mixed.valid_unseen.spec.txt"
fi

echo "[ver2.9-speaker-vec] done output_prepared_dir=$OUTPUT_PREPARED_DIR speaker_vec_dir=$SPEAKER_VEC_DIR"

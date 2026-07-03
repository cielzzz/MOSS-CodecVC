#!/usr/bin/env bash
set -euo pipefail

# Local CPU-only watcher. It waits until vcdata_construction finishes all
# selected splits, merges shard manifests, then submits text/no_text branch
# pipelines to Qizhi. This avoids starting an H200 job just to wait.

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
VCDATA_CONSTRUCTION_ROOT="${VCDATA_CONSTRUCTION_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction}"
PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"

DATASET_NAME="${DATASET_NAME:-zh11w_en11w_0005_0015_vcdata_first}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/trainset/$DATASET_NAME}"
VCDATA_ROOT="${VCDATA_ROOT:-$DATASET_ROOT/vcdata}"
VCDATA_JSONLS_FILE="${VCDATA_JSONLS_FILE:-$DATASET_ROOT/vcdata_jsonls.txt}"

LANGUAGES="${LANGUAGES:-zh,en}"
START_SHARD="${START_SHARD:-0005}"
END_SHARD="${END_SHARD:-0015}"
EXPECTED_ROWS_PER_SPLIT="${EXPECTED_ROWS_PER_SPLIT:-10000}"

POLL_SECONDS="${POLL_SECONDS:-300}"
RUN_ONCE="${RUN_ONCE:-0}"
PRIORITY="${PRIORITY:-10}"
BRANCHES="${BRANCHES:-text,no_text}"
ENABLE_TRAIN_READY_GPU_KEEPALIVE="${ENABLE_TRAIN_READY_GPU_KEEPALIVE:-1}"
ENABLE_SEMANTIC_GPU_KEEPALIVE="${ENABLE_SEMANTIC_GPU_KEEPALIVE:-1}"
TEXT_SEMANTIC_GPU_KEEPALIVE="${TEXT_SEMANTIC_GPU_KEEPALIVE:-$ENABLE_SEMANTIC_GPU_KEEPALIVE}"
NO_TEXT_SEMANTIC_GPU_KEEPALIVE="${NO_TEXT_SEMANTIC_GPU_KEEPALIVE:-$ENABLE_SEMANTIC_GPU_KEEPALIVE}"

WATCH_ROOT="${WATCH_ROOT:-$DATASET_ROOT/qz_jobs/vcdata_branch_watch}"
LOG_FILE="${LOG_FILE:-$WATCH_ROOT/watch.log}"
SUBMIT_MARKER="${SUBMIT_MARKER:-$WATCH_ROOT/submitted.done}"

mkdir -p "$WATCH_ROOT"

timestamp_utc() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

count_done_rows() {
  local split_dir="$1"
  local manifests=("$split_dir"/manifest_shard*.jsonl)
  if [ ! -e "${manifests[0]}" ]; then
    echo 0
    return 0
  fi
  wc -l "${manifests[@]}" | awk 'END { print $1 + 0 }'
}

expected_split_names() {
  local start_num end_num lang shard_num shard
  start_num=$((10#$START_SHARD))
  end_num=$((10#$END_SHARD))
  IFS=',' read -r -a lang_array <<< "$LANGUAGES"
  for lang in "${lang_array[@]}"; do
    lang="$(printf '%s' "$lang" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [ -n "$lang" ] || continue
    for shard_num in $(seq "$start_num" "$end_num"); do
      shard=$(printf "%04d" "$shard_num")
      printf '%s_slim_%s\n' "$lang" "$shard"
    done
  done
}

check_complete() {
  local complete=1
  local total_rows=0
  local split_name split_dir rows
  : > "$WATCH_ROOT/progress.tsv.tmp"
  while IFS= read -r split_name; do
    [ -n "$split_name" ] || continue
    split_dir="$VCDATA_ROOT/$split_name"
    rows=0
    if [ -d "$split_dir" ]; then
      rows=$(count_done_rows "$split_dir")
    fi
    total_rows=$((total_rows + rows))
    if [ "$rows" -lt "$EXPECTED_ROWS_PER_SPLIT" ]; then
      complete=0
    fi
    printf '%s\t%s\t%s\n' "$split_name" "$rows" "$EXPECTED_ROWS_PER_SPLIT" >> "$WATCH_ROOT/progress.tsv.tmp"
  done < <(expected_split_names)
  mv "$WATCH_ROOT/progress.tsv.tmp" "$WATCH_ROOT/progress.tsv"
  printf '%s\n' "$total_rows" > "$WATCH_ROOT/total_rows.txt"
  return "$((1 - complete))"
}

merge_complete_splits() {
  local split_name split_dir merged
  : > "$VCDATA_JSONLS_FILE.tmp"
  while IFS= read -r split_name; do
    [ -n "$split_name" ] || continue
    split_dir="$VCDATA_ROOT/$split_name"
    merged="$split_dir/merged.stepaudio_input.all.jsonl"
    if [ ! -d "$split_dir" ]; then
      echo "ERROR: missing split dir: $split_dir" >&2
      exit 2
    fi
    "$PY" "$VCDATA_CONSTRUCTION_ROOT/merge_shards.py" \
      --input-dir "$split_dir" \
      --output "$split_dir/manifest_merged.jsonl" \
      --dedupe-key original_audio \
      --keep best_similarity \
      --order source_original
    ln -sfn manifest_merged.jsonl "$merged"
    printf '%s\n' "$merged" >> "$VCDATA_JSONLS_FILE.tmp"
  done < <(expected_split_names)
  mv "$VCDATA_JSONLS_FILE.tmp" "$VCDATA_JSONLS_FILE"
}

submit_branches() {
  if [ -s "$SUBMIT_MARKER" ]; then
    echo "[$(timestamp_utc)] already submitted: $SUBMIT_MARKER"
    cat "$SUBMIT_MARKER"
    return 0
  fi
  merge_complete_splits
  local submit_log="$WATCH_ROOT/submit_branches.$(date -u +%Y%m%d-%H%M%S).log"
  echo "[$(timestamp_utc)] submitting branches priority=$PRIORITY branches=$BRANCHES" | tee -a "$submit_log"
  env \
    DATASET_NAME="$DATASET_NAME" \
    DATASET_ROOT="$DATASET_ROOT" \
    VCDATA_JSONLS_FILE="$VCDATA_JSONLS_FILE" \
    PRIORITY="$PRIORITY" \
    BRANCHES="$BRANCHES" \
    ENABLE_TRAIN_READY_GPU_KEEPALIVE="$ENABLE_TRAIN_READY_GPU_KEEPALIVE" \
    ENABLE_SEMANTIC_GPU_KEEPALIVE="$ENABLE_SEMANTIC_GPU_KEEPALIVE" \
    TEXT_SEMANTIC_GPU_KEEPALIVE="$TEXT_SEMANTIC_GPU_KEEPALIVE" \
    NO_TEXT_SEMANTIC_GPU_KEEPALIVE="$NO_TEXT_SEMANTIC_GPU_KEEPALIVE" \
    bash "$ROOT/scripts/001051_submit_vcdata_branch_pipelines_qz.sh" | tee -a "$submit_log"
  {
    echo "submitted_utc=$(timestamp_utc)"
    echo "submit_log=$submit_log"
    echo "vcdata_jsonls_file=$VCDATA_JSONLS_FILE"
  } > "$SUBMIT_MARKER"
}

main_loop() {
  while true; do
    if check_complete; then
      total_rows="$(cat "$WATCH_ROOT/total_rows.txt")"
      echo "[$(timestamp_utc)] vcdata complete total_rows=$total_rows" | tee -a "$LOG_FILE"
      submit_branches | tee -a "$LOG_FILE"
      exit 0
    fi
    total_rows="$(cat "$WATCH_ROOT/total_rows.txt")"
    echo "[$(timestamp_utc)] vcdata incomplete total_rows=$total_rows; partial:" | tee -a "$LOG_FILE"
    awk -F'\t' '$2 < $3 { printf "  %s %s/%s\n", $1, $2, $3 }' "$WATCH_ROOT/progress.tsv" | tee -a "$LOG_FILE"
    if [ "$RUN_ONCE" = "1" ]; then
      exit 1
    fi
    sleep "$POLL_SECONDS"
  done
}

main_loop

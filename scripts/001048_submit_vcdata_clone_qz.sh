#!/usr/bin/env bash
set -euo pipefail

# Submit raw JSONL -> MOSS-TTS vcdata clone jobs.
# This is stage 1 shared by both text_prosody and no_text VC construction.

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
VCDATA_CONSTRUCTION_ROOT="${VCDATA_CONSTRUCTION_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction}"
RAW_ROOT="${RAW_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vc_data_temp/mtd_pass_nonmulti_primary_le_0p3_split_10k}"

QZCLI_TOOL="${QZCLI_TOOL:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/qzcli_tool}"
QZ_PY="${QZ_PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
QZCLI_HOME="${QZCLI_HOME:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/.codex/qzcli_home}"

WORKSPACE="${WORKSPACE:-ws-8207e9e2-e733-4eec-a475-cfa1c36480ba}"
PROJECT="${PROJECT:-project-c67c548f-f02c-453b-ba5b-8745db6886e7}"
COMPUTE_GROUP="${COMPUTE_GROUP:-lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122}"  # MTTS-3-2-0715
SPEC="${SPEC:-67b10bc6-78b0-41a3-aaf4-358eeeb99009}"
FRAMEWORK="${FRAMEWORK:-pytorch}"
INSTANCES="${INSTANCES:-1}"
SHM_GI="${SHM_GI:-1200}"
PRIORITY="${PRIORITY:-3}"
IMAGE="${IMAGE:-docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420}"
IMAGE_TYPE="${IMAGE_TYPE:-SOURCE_PRIVATE}"

DATASET_NAME="${DATASET_NAME:-zh11w_en11w_0005_0015_vcdata_first}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/trainset/$DATASET_NAME}"
RAW_LINK_DIR="${RAW_LINK_DIR:-$DATASET_ROOT/raw_inputs}"
VCDATA_OUTPUT_ROOT="${VCDATA_OUTPUT_ROOT:-$DATASET_ROOT/vcdata}"

LANGUAGES="${LANGUAGES:-zh,en}"
START_SHARD="${START_SHARD:-0005}"
END_SHARD="${END_SHARD:-0015}"

ACTIVATE_SCRIPT="${ACTIVATE_SCRIPT:-$VCDATA_CONSTRUCTION_ROOT/activate_moss_ttsd_vc.sh}"
MODEL_DIR="${MODEL_DIR:-$VCDATA_CONSTRUCTION_ROOT/MOSS-TTS}"
AUDIO_PATH_FIELD="${AUDIO_PATH_FIELD:-audio_path}"
TEXT_FIELD="${TEXT_FIELD:-mtd_transcript}"
NUM_CANDIDATES="${NUM_CANDIDATES:-16}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:-0.85}"
SEED_BASE="${SEED_BASE:-42}"
NUM_GPUS="${NUM_GPUS:-8}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
PARALLEL_SPLITS="${PARALLEL_SPLITS:-8}"
VCDATA_SCHEDULER_MODE="${VCDATA_SCHEDULER_MODE:-dir_shards}"
MAX_RETRIES="${MAX_RETRIES:-3}"
TASK_COUNT="${TASK_COUNT:-2}"
GPU_MONITOR_ENABLE="${GPU_MONITOR_ENABLE:-1}"
GPU_MONITOR_INTERVAL_SEC="${GPU_MONITOR_INTERVAL_SEC:-30}"

BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%d-%H%M%S)-$$}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codec-vc-data-process}"
QZ_RECORD_ROOT="${QZ_RECORD_ROOT:-$DATASET_ROOT/qz_jobs/vcdata_clone/$BATCH_ID}"

DRY_RUN=0

usage() {
  cat <<EOF
Usage:
  bash scripts/001048_submit_vcdata_clone_qz.sh [--dry-run]

Common overrides:
  START_SHARD=0005 END_SHARD=0015 TASK_COUNT=4 DATASET_NAME=zh11w_en11w_0005_0015_vcdata_first \\
    bash scripts/001048_submit_vcdata_clone_qz.sh

Outputs:
  raw symlinks: $RAW_LINK_DIR
  vcdata root:  $VCDATA_OUTPUT_ROOT
  qz records:   $QZ_RECORD_ROOT
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
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
if [ ! -d "$VCDATA_CONSTRUCTION_ROOT" ]; then
  echo "ERROR: VCDATA_CONSTRUCTION_ROOT does not exist: $VCDATA_CONSTRUCTION_ROOT" >&2
  exit 1
fi
if [ ! -x "$QZ_PY" ]; then
  echo "ERROR: QZ_PY is not executable: $QZ_PY" >&2
  exit 1
fi

mkdir -p "$RAW_LINK_DIR" "$VCDATA_OUTPUT_ROOT" "$QZ_RECORD_ROOT" "$QZCLI_HOME"

start_num=$((10#$START_SHARD))
end_num=$((10#$END_SHARD))
if [ "$end_num" -lt "$start_num" ]; then
  echo "ERROR: END_SHARD must be >= START_SHARD" >&2
  exit 2
fi

IFS=',' read -r -a LANG_ARRAY <<< "$LANGUAGES"
INPUT_LIST="$QZ_RECORD_ROOT/input_jsonls.txt"
: > "$INPUT_LIST"
for lang in "${LANG_ARRAY[@]}"; do
  lang="$(printf '%s' "$lang" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  [ -n "$lang" ] || continue
  for shard_num in $(seq "$start_num" "$end_num"); do
    shard=$(printf "%04d" "$shard_num")
    src="$RAW_ROOT/$lang/${lang}_slim_${shard}.jsonl"
    if [ ! -s "$src" ]; then
      echo "ERROR: missing input JSONL: $src" >&2
      exit 2
    fi
    ln -sfn "$src" "$RAW_LINK_DIR/${lang}_slim_${shard}.jsonl"
    printf '%s\n' "$RAW_LINK_DIR/${lang}_slim_${shard}.jsonl" >> "$INPUT_LIST"
  done
done

count_lines() {
  awk 'NF { c += 1 } END { print c + 0 }' "$1"
}

count_done_cases() {
  local split_dir="$1"
  local manifests=("$split_dir"/manifest_shard*.jsonl)
  if [ ! -e "${manifests[0]}" ]; then
    echo 0
    return 0
  fi
  wc -l "${manifests[@]}" | awk 'END { print $1 + 0 }'
}

PENDING_LIST="$QZ_RECORD_ROOT/pending_jsonls.txt"
: > "$PENDING_LIST"
total_splits=0
pending_splits=0
skipped_completed=0
while IFS= read -r jsonl; do
  [ -n "$jsonl" ] || continue
  total_splits=$((total_splits + 1))
  stem=$(basename "$jsonl" .jsonl)
  total_cases=$(count_lines "$jsonl")
  done_cases=0
  if [ -d "$VCDATA_OUTPUT_ROOT/$stem" ]; then
    done_cases=$(count_done_cases "$VCDATA_OUTPUT_ROOT/$stem")
  fi
  if [ "$total_cases" -gt 0 ] && [ "$done_cases" -ge "$total_cases" ] && [ -f "$VCDATA_OUTPUT_ROOT/$stem/.stage1_generate_state.json" ]; then
    skipped_completed=$((skipped_completed + 1))
    continue
  fi
  printf '%s\n' "$jsonl" >> "$PENDING_LIST"
  pending_splits=$((pending_splits + 1))
done < "$INPUT_LIST"

if [ "$pending_splits" -eq 0 ]; then
  echo "All selected vcdata clone splits are already complete."
  echo "  DATASET_ROOT=$DATASET_ROOT"
  echo "  VCDATA_OUTPUT_ROOT=$VCDATA_OUTPUT_ROOT"
  exit 0
fi

group_count="$TASK_COUNT"
if [ "$group_count" -gt "$pending_splits" ]; then
  group_count="$pending_splits"
fi
if [ "$group_count" -lt 1 ]; then
  group_count=1
fi

GROUP_ROOT="$QZ_RECORD_ROOT/groups"
rm -rf "$GROUP_ROOT"
mkdir -p "$GROUP_ROOT"
for group_idx in $(seq 0 $((group_count - 1))); do
  mkdir -p "$GROUP_ROOT/group_$(printf '%02d' "$group_idx")"
done

idx=0
while IFS= read -r jsonl; do
  group_idx=$((idx % group_count))
  group_dir="$GROUP_ROOT/group_$(printf '%02d' "$group_idx")"
  ln -sfn "$jsonl" "$group_dir/$(basename "$jsonl")"
  printf '%s\n' "$(basename "$jsonl")" >> "$group_dir/splits.txt"
  idx=$((idx + 1))
done < "$PENDING_LIST"

SUBMITTED_TSV="$QZ_RECORD_ROOT/submitted_jobs.tsv"
: > "$SUBMITTED_TSV"
printf 'job_name\tjob_id\tpriority\tcompute_group\trunner\tgroup_dir\tvcdata_output_root\n' > "$SUBMITTED_TSV"

echo "=========================================="
echo "QZ submit: vcdata clone stage"
echo "  DATASET_NAME=$DATASET_NAME"
echo "  DATASET_ROOT=$DATASET_ROOT"
echo "  RAW_ROOT=$RAW_ROOT"
echo "  RAW_LINK_DIR=$RAW_LINK_DIR"
echo "  VCDATA_OUTPUT_ROOT=$VCDATA_OUTPUT_ROOT"
echo "  LANGUAGES=$LANGUAGES SHARDS=$START_SHARD..$END_SHARD"
echo "  total_splits=$total_splits pending_splits=$pending_splits skipped_completed=$skipped_completed"
echo "  TASK_COUNT=$TASK_COUNT effective_group_count=$group_count"
echo "  TEXT_FIELD=$TEXT_FIELD AUDIO_PATH_FIELD=$AUDIO_PATH_FIELD"
echo "  NUM_CANDIDATES=$NUM_CANDIDATES BATCH_SIZE=$BATCH_SIZE SIMILARITY_THRESHOLD=$SIMILARITY_THRESHOLD"
echo "  VCDATA_SCHEDULER_MODE=$VCDATA_SCHEDULER_MODE NUM_GPUS=$NUM_GPUS WORKERS_PER_GPU=$WORKERS_PER_GPU PARALLEL_SPLITS=$PARALLEL_SPLITS"
echo "  PRIORITY=$PRIORITY COMPUTE_GROUP=$COMPUTE_GROUP"
echo "  QZ_RECORD_ROOT=$QZ_RECORD_ROOT"
echo "  DRY_RUN=$DRY_RUN"
echo "=========================================="

for group_idx in $(seq 0 $((group_count - 1))); do
  tag=$(printf "%02d" "$group_idx")
  group_dir="$GROUP_ROOT/group_$tag"
  if [ ! -s "$group_dir/splits.txt" ]; then
    continue
  fi
  job_name="${JOB_NAME_PREFIX}-${BATCH_ID}-g${tag}"
  runner="$QZ_RECORD_ROOT/run_vcdata_clone_g${tag}.sh"
  run_log="$QZ_RECORD_ROOT/run_vcdata_clone_g${tag}.log"
  gpu_monitor_log="$QZ_RECORD_ROOT/gpu_metrics_g${tag}.csv"
  worker_log_root="$QZ_RECORD_ROOT/worker_logs_g${tag}"
  attempt_metrics_log="$QZ_RECORD_ROOT/vcdata_attempt_metrics_g${tag}.tsv"

  cat > "$runner" <<EOF
#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$QZ_RECORD_ROOT" "$worker_log_root"
exec > >(tee -a "$run_log") 2>&1
set -x

export ACTIVATE_SCRIPT="$ACTIVATE_SCRIPT"
export MODEL_DIR="$MODEL_DIR"
export INPUT_DIR="$group_dir"
export OUTPUT_ROOT="$VCDATA_OUTPUT_ROOT"
export AUDIO_PATH_FIELD="$AUDIO_PATH_FIELD"
export TEXT_FIELD="$TEXT_FIELD"
export NUM_CANDIDATES="$NUM_CANDIDATES"
export BATCH_SIZE="$BATCH_SIZE"
export SIMILARITY_THRESHOLD="$SIMILARITY_THRESHOLD"
export SEED_BASE="$((SEED_BASE + group_idx))"
export NUM_GPUS="$NUM_GPUS"
export WORKERS_PER_GPU="$WORKERS_PER_GPU"
export PARALLEL_SPLITS="$PARALLEL_SPLITS"
export MAX_RETRIES="$MAX_RETRIES"
export SCHEDULER_MODE="$VCDATA_SCHEDULER_MODE"
export ALLOW_RESUME_SHARD_CHANGE=1
export GPU_MONITOR_ENABLE="$GPU_MONITOR_ENABLE"
export GPU_MONITOR_INTERVAL_SEC="$GPU_MONITOR_INTERVAL_SEC"
export GPU_MONITOR_LOG="$gpu_monitor_log"
export WORKER_LOG_ROOT="$worker_log_root"
export ATTEMPT_METRICS_LOG="$attempt_metrics_log"
export HF_HOME="$VCDATA_CONSTRUCTION_ROOT/.hf_cache"
export TRANSFORMERS_CACHE="$VCDATA_CONSTRUCTION_ROOT/.hf_cache/transformers"
export HUGGINGFACE_HUB_CACHE="$VCDATA_CONSTRUCTION_ROOT/.hf_cache/hub"
export TORCH_HOME="/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/torch"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="\${OMP_NUM_THREADS:-8}"

cd "$VCDATA_CONSTRUCTION_ROOT"
echo "[vcdata-clone] date=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[vcdata-clone] host=\$(hostname)"
echo "[vcdata-clone] group=$tag input_dir=\$INPUT_DIR output_root=\$OUTPUT_ROOT"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

bash run8gpu_retries.sh

while IFS= read -r split_name; do
  [ -n "\$split_name" ] || continue
  stem="\${split_name%.jsonl}"
  task_dir="$VCDATA_OUTPUT_ROOT/\$stem"
  if [ -d "\$task_dir" ]; then
    python merge_shards.py \\
      --input-dir "\$task_dir" \\
      --output "\$task_dir/manifest_merged.jsonl" \\
      --dedupe-key original_audio \\
      --keep best_similarity \\
      --order source_original
    ln -sfn manifest_merged.jsonl "\$task_dir/merged.stepaudio_input.all.jsonl"
  fi
done < "$group_dir/splits.txt"

find "$VCDATA_OUTPUT_ROOT" -mindepth 2 -maxdepth 2 -name merged.stepaudio_input.all.jsonl | sort > "$DATASET_ROOT/vcdata_jsonls.txt"
echo "[vcdata-clone] merged jsonl list: $DATASET_ROOT/vcdata_jsonls.txt"
EOF
  chmod +x "$runner"

  command_to_run="bash $runner"
  tmp_output="$QZ_RECORD_ROOT/submit_g${tag}.txt"
  qz_args=(
    -m qzcli.cli create-job
    --name "$job_name"
    --workspace "$WORKSPACE"
    --project "$PROJECT"
    --compute-group "$COMPUTE_GROUP"
    --spec "$SPEC"
    --framework "$FRAMEWORK"
    --instances "$INSTANCES"
    --shm "$SHM_GI"
    --priority "$PRIORITY"
    --image "$IMAGE"
    --image-type "$IMAGE_TYPE"
    --command "$command_to_run"
  )
  if [ "$DRY_RUN" -eq 1 ]; then
    qz_args+=(--dry-run)
  fi

  echo "------------------------------------------"
  echo "Group $tag"
  echo "  JOB_NAME=$job_name"
  echo "  GROUP_DIR=$group_dir"
  echo "  RUNNER=$runner"
  echo "  COMMAND=$command_to_run"

  set +e
  env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
    HOME="$QZCLI_HOME" \
    PYTHONPATH="$QZCLI_TOOL" \
    "$QZ_PY" "${qz_args[@]}" >"$tmp_output" 2>&1
  status=$?
  set -e
  cat "$tmp_output"
  if [ "$status" -ne 0 ]; then
    echo "Submission failed for $job_name. Output saved to $tmp_output" >&2
    exit "$status"
  fi

  job_id=$(grep -Eo 'job-[0-9a-fA-F-]{36}' "$tmp_output" | tail -n 1 || true)
  if [ -z "$job_id" ]; then
    job_uuid=$(grep -E '任务ID|job_id|Job ID' "$tmp_output" | grep -Eo '[0-9a-fA-F-]{36}' | tail -n 1 || true)
    if [ -n "$job_uuid" ]; then
      job_id="job-$job_uuid"
    fi
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$job_name" "${job_id:-}" "$PRIORITY" "$COMPUTE_GROUP" "$runner" "$group_dir" "$VCDATA_OUTPUT_ROOT" >> "$SUBMITTED_TSV"
done

echo "=========================================="
echo "vcdata clone submission finished"
echo "  submitted_jobs=$SUBMITTED_TSV"
echo "  vcdata_jsonls_after_merge=$DATASET_ROOT/vcdata_jsonls.txt"
echo "=========================================="

#!/usr/bin/env bash
set -euo pipefail

# Submit stage-2 branch pipelines after vcdata_construction is complete.
# Branches:
#   text    : vcdata -> text_prosody Seed-VC triples -> train-ready semantic JSONL
#   no_text : vcdata -> no_text Seed-VC triples -> train-ready semantic JSONL

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
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
PRIORITY="${PRIORITY:-10}"
IMAGE="${IMAGE:-docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420}"
IMAGE_TYPE="${IMAGE_TYPE:-SOURCE_PRIVATE}"

PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
PY_MOSS="${PY_MOSS:-$PY}"
PY_SEEDVC="${PY_SEEDVC:-/inspire/ssd/project/embodied-multimodality/public/yqzhang/miniconda3/envs/contts-train/bin/python}"
PYTHON_MAIN="${PYTHON_MAIN:-$PY}"
PYTHON_ASR="${PYTHON_ASR:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download}"

DATASET_NAME="${DATASET_NAME:-zh11w_en11w_0005_0015_vcdata_first}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/trainset/$DATASET_NAME}"
VCDATA_JSONLS_FILE="${VCDATA_JSONLS_FILE:-$DATASET_ROOT/vcdata_jsonls.txt}"
VCDATA_JSONLS="${VCDATA_JSONLS:-}"

BRANCHES="${BRANCHES:-text,no_text}"
BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%d-%H%M%S)-$$}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codecvc-vcdata-branch}"
QZ_RECORD_ROOT="${QZ_RECORD_ROOT:-$DATASET_ROOT/qz_jobs/vcdata_branches/$BATCH_ID}"

LANGUAGES="${LANGUAGES:-zh,en}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
SEEDVC_SHARD_COUNT="${SEEDVC_SHARD_COUNT:-8}"
CODEC_SHARD_COUNT="${CODEC_SHARD_COUNT:-8}"
SPEAKER_SHARD_COUNT="${SPEAKER_SHARD_COUNT:-8}"
PROSODY_SHARD_COUNT="${PROSODY_SHARD_COUNT:-16}"
N_VQ="${N_VQ:-32}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE="${FORCE:-0}"

# Keepalive is only used by known CPU/IO-heavy tail stages. Actual GPU-heavy
# stages such as Seed-VC, codec, ECAPA, ASR and HuBERT do real work.
ENABLE_TRAIN_READY_GPU_KEEPALIVE="${ENABLE_TRAIN_READY_GPU_KEEPALIVE:-1}"
ENABLE_SEMANTIC_GPU_KEEPALIVE="${ENABLE_SEMANTIC_GPU_KEEPALIVE:-1}"
TEXT_SEMANTIC_GPU_KEEPALIVE="${TEXT_SEMANTIC_GPU_KEEPALIVE:-$ENABLE_SEMANTIC_GPU_KEEPALIVE}"
NO_TEXT_SEMANTIC_GPU_KEEPALIVE="${NO_TEXT_SEMANTIC_GPU_KEEPALIVE:-$ENABLE_SEMANTIC_GPU_KEEPALIVE}"

TEXT_DATASET_NAME="${TEXT_DATASET_NAME:-${DATASET_NAME}_text_prosody}"
NO_TEXT_DATASET_NAME="${NO_TEXT_DATASET_NAME:-${DATASET_NAME}_no_text}"

DRY_RUN=0

usage() {
  cat <<EOF
Usage:
  bash scripts/001051_submit_vcdata_branch_pipelines_qz.sh [--dry-run]

Common overrides:
  DATASET_NAME=zh11w_en11w_0005_0015_vcdata_first PRIORITY=10 BRANCHES=text,no_text \\
    bash scripts/001051_submit_vcdata_branch_pipelines_qz.sh

Inputs:
  VCDATA_JSONLS_FILE=$VCDATA_JSONLS_FILE
  or VCDATA_JSONLS=/abs/a.jsonl,/abs/b.jsonl
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
if [ ! -x "$QZ_PY" ]; then
  echo "ERROR: QZ_PY is not executable: $QZ_PY" >&2
  exit 1
fi

resolve_vcdata_jsonls() {
  if [ -n "$VCDATA_JSONLS" ]; then
    printf '%s\n' "$VCDATA_JSONLS"
    return 0
  fi
  if [ -s "$VCDATA_JSONLS_FILE" ]; then
    paste -sd, "$VCDATA_JSONLS_FILE"
    return 0
  fi
  return 0
}

vcdata_csv="$(resolve_vcdata_jsonls)"
if [ -z "$vcdata_csv" ]; then
  echo "ERROR: no vcdata JSONLs found. Set VCDATA_JSONLS or create $VCDATA_JSONLS_FILE." >&2
  exit 2
fi
vcdata_count=$(printf '%s' "$vcdata_csv" | tr ',' '\n' | awk 'NF { c += 1 } END { print c + 0 }')
if [ "$vcdata_count" -lt 1 ]; then
  echo "ERROR: empty vcdata JSONL list." >&2
  exit 2
fi

mkdir -p "$QZ_RECORD_ROOT" "$QZCLI_HOME"
SUBMITTED_TSV="$QZ_RECORD_ROOT/submitted_jobs.tsv"
: > "$SUBMITTED_TSV"
printf 'branch\tjob_name\tjob_id\tpriority\tcompute_group\trunner\tdataset_name\n' > "$SUBMITTED_TSV"

submit_branch() {
  local branch="$1"
  local tag run_text run_no_text branch_dataset runner run_log job_name tmp_output command_to_run
  case "$branch" in
    text)
      tag="text"
      run_text=1
      run_no_text=0
      branch_dataset="$TEXT_DATASET_NAME"
      ;;
    no_text)
      tag="no_text"
      run_text=0
      run_no_text=1
      branch_dataset="$NO_TEXT_DATASET_NAME"
      ;;
    *)
      echo "ERROR: unsupported branch: $branch" >&2
      return 2
      ;;
  esac

  runner="$QZ_RECORD_ROOT/run_${tag}_pipeline.sh"
  run_log="$QZ_RECORD_ROOT/run_${tag}_pipeline.log"
  job_name="${JOB_NAME_PREFIX}-${DATASET_NAME}-${tag}-${BATCH_ID}"
  tmp_output="$QZ_RECORD_ROOT/submit_${tag}.txt"

  cat > "$runner" <<EOF
#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$QZ_RECORD_ROOT"
exec > >(tee -a "$run_log") 2>&1
set -x

export PY="$PY"
export PY_MOSS="$PY_MOSS"
export PY_SEEDVC="$PY_SEEDVC"
export PYTHON_MAIN="$PYTHON_MAIN"
export PYTHON_ASR="$PYTHON_ASR"
export DOWNLOAD_ROOT="$DOWNLOAD_ROOT"
export DATASET_NAME="$DATASET_NAME"
export DATASET_ROOT="$DATASET_ROOT"
export VCDATA_JSONLS="$vcdata_csv"
export LANGUAGES="$LANGUAGES"
export GPU_IDS="$GPU_IDS"
export SEEDVC_GPU_IDS="$GPU_IDS"
export CODEC_GPU_IDS="$GPU_IDS"
export SPEAKER_GPU_IDS="$GPU_IDS"
export SEEDVC_SHARD_COUNT="$SEEDVC_SHARD_COUNT"
export CODEC_SHARD_COUNT="$CODEC_SHARD_COUNT"
export SPEAKER_SHARD_COUNT="$SPEAKER_SHARD_COUNT"
export PROSODY_SHARD_COUNT="$PROSODY_SHARD_COUNT"
export N_VQ="$N_VQ"
export SKIP_EXISTING="$SKIP_EXISTING"
export FORCE="$FORCE"

export RUN_TEXT_BRANCH="$run_text"
export RUN_NO_TEXT_BRANCH="$run_no_text"
export TEXT_DATASET_NAME="$TEXT_DATASET_NAME"
export NO_TEXT_DATASET_NAME="$NO_TEXT_DATASET_NAME"

export TRAIN_READY_GPU_KEEPALIVE="$ENABLE_TRAIN_READY_GPU_KEEPALIVE"
export TEXT_TRAIN_READY_GPU_KEEPALIVE="$ENABLE_TRAIN_READY_GPU_KEEPALIVE"
export NO_TEXT_TRAIN_READY_GPU_KEEPALIVE="$ENABLE_TRAIN_READY_GPU_KEEPALIVE"
export TEXT_SEMANTIC_GPU_KEEPALIVE="$TEXT_SEMANTIC_GPU_KEEPALIVE"
export NO_TEXT_SEMANTIC_GPU_KEEPALIVE="$NO_TEXT_SEMANTIC_GPU_KEEPALIVE"

export TEXT_ASR_NUM_SHARDS=8
export TEXT_HUBERT_NUM_SHARDS=16
export NO_TEXT_SEMANTIC_ASR_NUM_SHARDS=8
export NO_TEXT_SEMANTIC_HUBERT_NUM_SHARDS=16
export NO_TEXT_SEMANTIC_HUBERT_SOURCE=both

export HF_HOME="\$DOWNLOAD_ROOT/huggingface"
export TRANSFORMERS_CACHE="\$DOWNLOAD_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="\$DOWNLOAD_ROOT/huggingface/hub"
export HF_DATASETS_CACHE="\$DOWNLOAD_ROOT/huggingface/datasets"
export TORCH_HOME="\$DOWNLOAD_ROOT/torch"
export XDG_CACHE_HOME="\$DOWNLOAD_ROOT/cache"
export HF_ENDPOINT="\${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="\${OMP_NUM_THREADS:-8}"

cd "$ROOT"
echo "[vcdata-branch] date=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[vcdata-branch] host=\$(hostname)"
echo "[vcdata-branch] branch=$branch dataset=$branch_dataset priority=$PRIORITY"
echo "[vcdata-branch] vcdata_count=$vcdata_count"
echo "[vcdata-branch] train_ready_gpu_keepalive=$ENABLE_TRAIN_READY_GPU_KEEPALIVE text_semantic_gpu_keepalive=$TEXT_SEMANTIC_GPU_KEEPALIVE no_text_semantic_gpu_keepalive=$NO_TEXT_SEMANTIC_GPU_KEEPALIVE"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

bash scripts/001050_run_vcdata_text_no_text_pipeline.sh
EOF
  chmod +x "$runner"

  command_to_run="bash $runner"
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
  echo "Submit branch=$branch"
  echo "  JOB_NAME=$job_name"
  echo "  DATASET=$branch_dataset"
  echo "  RUNNER=$runner"
  echo "  PRIORITY=$PRIORITY"

  set +e
  env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
    HOME="$QZCLI_HOME" \
    PYTHONPATH="$QZCLI_TOOL" \
    "$QZ_PY" "${qz_args[@]}" >"$tmp_output" 2>&1
  local status=$?
  set -e
  cat "$tmp_output"
  if [ "$status" -ne 0 ]; then
    echo "Submission failed for branch=$branch. Output saved to $tmp_output" >&2
    return "$status"
  fi

  local job_id
  job_id=$(grep -Eo 'job-[0-9a-fA-F-]{36}' "$tmp_output" | tail -n 1 || true)
  if [ -z "$job_id" ]; then
    local job_uuid
    job_uuid=$(grep -E '任务ID|job_id|Job ID' "$tmp_output" | grep -Eo '[0-9a-fA-F-]{36}' | tail -n 1 || true)
    if [ -n "$job_uuid" ]; then
      job_id="job-$job_uuid"
    fi
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$branch" "$job_name" "${job_id:-}" "$PRIORITY" "$COMPUTE_GROUP" "$runner" "$branch_dataset" >> "$SUBMITTED_TSV"
}

echo "=========================================="
echo "QZ submit: vcdata branch pipelines"
echo "  DATASET_NAME=$DATASET_NAME"
echo "  DATASET_ROOT=$DATASET_ROOT"
echo "  vcdata_count=$vcdata_count"
echo "  BRANCHES=$BRANCHES"
echo "  PRIORITY=$PRIORITY"
echo "  QZ_RECORD_ROOT=$QZ_RECORD_ROOT"
echo "  DRY_RUN=$DRY_RUN"
echo "=========================================="

IFS=',' read -r -a branch_array <<< "$BRANCHES"
for raw_branch in "${branch_array[@]}"; do
  branch="$(printf '%s' "$raw_branch" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  [ -n "$branch" ] || continue
  submit_branch "$branch"
done

echo "=========================================="
echo "vcdata branch submission finished"
echo "  submitted_jobs=$SUBMITTED_TSV"
echo "=========================================="

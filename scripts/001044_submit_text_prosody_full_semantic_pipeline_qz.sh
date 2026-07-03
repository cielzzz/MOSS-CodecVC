#!/usr/bin/env bash
set -euo pipefail

# Submit the full reusable text_prosody semantic pipeline:
# vcdata -> Seed-VC triples -> codec/SFT/ECAPA/prosody -> ASR/CTC/HuBERT semantic JSONL.

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
PRIORITY="${PRIORITY:-3}"
IMAGE="${IMAGE:-docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420}"
IMAGE_TYPE="${IMAGE_TYPE:-SOURCE_PRIVATE}"

PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
PY_MOSS="${PY_MOSS:-$PY}"
PY_SEEDVC="${PY_SEEDVC:-/inspire/ssd/project/embodied-multimodality/public/yqzhang/miniconda3/envs/contts-train/bin/python}"
PYTHON_MAIN="${PYTHON_MAIN:-$PY}"
PYTHON_ASR="${PYTHON_ASR:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download}"

DATASET_NAME="${DATASET_NAME:-zh3w_en3w_text_prosody_independent_timbre_semantic}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/trainset/$DATASET_NAME}"
RUN_NAME="${RUN_NAME:-$DATASET_NAME}"
VCDATA_JSONLS="${VCDATA_JSONLS:-}"

RUN_TEXT_TRIPLE_STAGE="${RUN_TEXT_TRIPLE_STAGE:-1}"
RUN_TRAIN_READY_STAGE="${RUN_TRAIN_READY_STAGE:-1}"
RUN_SEMANTIC_STAGE="${RUN_SEMANTIC_STAGE:-1}"
FORCE="${FORCE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

MAX_JOBS="${MAX_JOBS:-0}"
MAX_JOBS_PER_LANGUAGE="${MAX_JOBS_PER_LANGUAGE:-0}"
MAX_ROWS_PER_INPUT="${MAX_ROWS_PER_INPUT:-0}"
LANGUAGES="${LANGUAGES:-zh,en}"
TIMBRE_REF_POLICY="${TIMBRE_REF_POLICY:-random_different_text}"
TIMBRE_REF_SEED="${TIMBRE_REF_SEED:-20260627}"
SEEDVC_SHARD_COUNT="${SEEDVC_SHARD_COUNT:-8}"
SEEDVC_GPU_IDS="${SEEDVC_GPU_IDS:-0,1,2,3,4,5,6,7}"

N_VQ="${N_VQ:-32}"
CODEC_GPU_IDS="${CODEC_GPU_IDS:-0,1,2,3,4,5,6,7}"
SPEAKER_GPU_IDS="${SPEAKER_GPU_IDS:-0,1,2,3,4,5,6,7}"
CODEC_SHARD_COUNT="${CODEC_SHARD_COUNT:-8}"
SPEAKER_SHARD_COUNT="${SPEAKER_SHARD_COUNT:-8}"
PROSODY_SHARD_COUNT="${PROSODY_SHARD_COUNT:-16}"
TRAIN_READY_GPU_KEEPALIVE="${TRAIN_READY_GPU_KEEPALIVE:-0}"

ASR_BACKEND="${ASR_BACKEND:-qwen_asr}"
ASR_NUM_SHARDS="${ASR_NUM_SHARDS:-8}"
ASR_DEVICES="${ASR_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7}"
HUBERT_NUM_SHARDS="${HUBERT_NUM_SHARDS:-16}"
HUBERT_DEVICES="${HUBERT_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7,cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7}"
HUBERT_PRE_SPLIT="${HUBERT_PRE_SPLIT:-1}"
HUBERT_SOURCE="${HUBERT_SOURCE:-target}"
SEMANTIC_RUN_PARALLEL_BRANCHES="${SEMANTIC_RUN_PARALLEL_BRANCHES:-1}"
SEMANTIC_FILTER_CONTENT_KEEP="${SEMANTIC_FILTER_CONTENT_KEEP:-1}"
SEMANTIC_GPU_KEEPALIVE="${SEMANTIC_GPU_KEEPALIVE:-0}"
SEMANTIC_GPU_KEEPALIVE_GPU_IDS="${SEMANTIC_GPU_KEEPALIVE_GPU_IDS:-0,1,2,3,4,5,6,7}"
SEMANTIC_GPU_KEEPALIVE_MATMUL_SIZE="${SEMANTIC_GPU_KEEPALIVE_MATMUL_SIZE:-2048}"
SEMANTIC_GPU_KEEPALIVE_SLEEP_SEC="${SEMANTIC_GPU_KEEPALIVE_SLEEP_SEC:-0.05}"
SEMANTIC_GPU_KEEPALIVE_DTYPE="${SEMANTIC_GPU_KEEPALIVE_DTYPE:-float16}"

BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%d-%H%M%S)-$$}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codecvc-text-prosody-full-semantic}"
JOB_NAME="${JOB_NAME:-$JOB_NAME_PREFIX-$BATCH_ID}"
QZ_RECORD_ROOT="${QZ_RECORD_ROOT:-$ROOT/trainset/qz_jobs/$BATCH_ID}"
RUNNER="$QZ_RECORD_ROOT/run_text_prosody_full_semantic_pipeline_entrypoint.sh"

DRY_RUN=0

usage() {
  cat <<EOF
Usage:
  bash scripts/001044_submit_text_prosody_full_semantic_pipeline_qz.sh [--dry-run]

Common overrides:
  DATASET_NAME=my_text_prosody VCDATA_JSONLS=/abs/a.jsonl,/abs/b.jsonl bash scripts/001044_submit_text_prosody_full_semantic_pipeline_qz.sh
  MAX_JOBS_PER_LANGUAGE=1000 bash scripts/001044_submit_text_prosody_full_semantic_pipeline_qz.sh --dry-run

Default final semantic JSONL:
  $DATASET_ROOT/sft/moss_codecvc_sft.$DATASET_NAME.with_light_ecapa_spk.with_prosody.with_content_tokens.with_target_hubert.jsonl
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

mkdir -p "$QZ_RECORD_ROOT" "$QZCLI_HOME"

cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail

RUN_LOG="$QZ_RECORD_ROOT/run.log"
mkdir -p "$QZ_RECORD_ROOT"
exec > >(tee -a "\$RUN_LOG") 2>&1
set -x

export PY="$PY"
export PY_MOSS="$PY_MOSS"
export PY_SEEDVC="$PY_SEEDVC"
export PYTHON_MAIN="$PYTHON_MAIN"
export PYTHON_ASR="$PYTHON_ASR"
export DOWNLOAD_ROOT="$DOWNLOAD_ROOT"
export DATASET_NAME="$DATASET_NAME"
export DATASET_ROOT="$DATASET_ROOT"
export RUN_NAME="$RUN_NAME"
export VCDATA_JSONLS="$VCDATA_JSONLS"

export RUN_TEXT_TRIPLE_STAGE="$RUN_TEXT_TRIPLE_STAGE"
export RUN_TRAIN_READY_STAGE="$RUN_TRAIN_READY_STAGE"
export RUN_SEMANTIC_STAGE="$RUN_SEMANTIC_STAGE"
export FORCE="$FORCE"
export SKIP_EXISTING="$SKIP_EXISTING"

export MAX_JOBS="$MAX_JOBS"
export MAX_JOBS_PER_LANGUAGE="$MAX_JOBS_PER_LANGUAGE"
export MAX_ROWS_PER_INPUT="$MAX_ROWS_PER_INPUT"
export LANGUAGES="$LANGUAGES"
export TIMBRE_REF_POLICY="$TIMBRE_REF_POLICY"
export TIMBRE_REF_SEED="$TIMBRE_REF_SEED"
export SEEDVC_SHARD_COUNT="$SEEDVC_SHARD_COUNT"
export SEEDVC_GPU_IDS="$SEEDVC_GPU_IDS"

export N_VQ="$N_VQ"
export CODEC_GPU_IDS="$CODEC_GPU_IDS"
export SPEAKER_GPU_IDS="$SPEAKER_GPU_IDS"
export CODEC_SHARD_COUNT="$CODEC_SHARD_COUNT"
export SPEAKER_SHARD_COUNT="$SPEAKER_SHARD_COUNT"
export PROSODY_SHARD_COUNT="$PROSODY_SHARD_COUNT"
export TRAIN_READY_GPU_KEEPALIVE="$TRAIN_READY_GPU_KEEPALIVE"

export ASR_BACKEND="$ASR_BACKEND"
export ASR_NUM_SHARDS="$ASR_NUM_SHARDS"
export ASR_DEVICES="$ASR_DEVICES"
export HUBERT_NUM_SHARDS="$HUBERT_NUM_SHARDS"
export HUBERT_DEVICES="$HUBERT_DEVICES"
export HUBERT_PRE_SPLIT="$HUBERT_PRE_SPLIT"
export HUBERT_SOURCE="$HUBERT_SOURCE"
export SEMANTIC_RUN_PARALLEL_BRANCHES="$SEMANTIC_RUN_PARALLEL_BRANCHES"
export SEMANTIC_FILTER_CONTENT_KEEP="$SEMANTIC_FILTER_CONTENT_KEEP"
export SEMANTIC_GPU_KEEPALIVE="$SEMANTIC_GPU_KEEPALIVE"
export SEMANTIC_GPU_KEEPALIVE_GPU_IDS="$SEMANTIC_GPU_KEEPALIVE_GPU_IDS"
export SEMANTIC_GPU_KEEPALIVE_MATMUL_SIZE="$SEMANTIC_GPU_KEEPALIVE_MATMUL_SIZE"
export SEMANTIC_GPU_KEEPALIVE_SLEEP_SEC="$SEMANTIC_GPU_KEEPALIVE_SLEEP_SEC"
export SEMANTIC_GPU_KEEPALIVE_DTYPE="$SEMANTIC_GPU_KEEPALIVE_DTYPE"

export HF_HOME="\$DOWNLOAD_ROOT/huggingface"
export TRANSFORMERS_CACHE="\$DOWNLOAD_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="\$DOWNLOAD_ROOT/huggingface/hub"
export HF_DATASETS_CACHE="\$DOWNLOAD_ROOT/huggingface/datasets"
export TORCH_HOME="\$DOWNLOAD_ROOT/torch"
export XDG_CACHE_HOME="\$DOWNLOAD_ROOT/cache"
export DISABLE_SAFETENSORS_CONVERSION=1
export HF_ENDPOINT="\${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="\${OMP_NUM_THREADS:-8}"

cd "$ROOT"
echo "[qz-text-prosody-full-semantic] date=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[qz-text-prosody-full-semantic] host=\$(hostname)"
echo "[qz-text-prosody-full-semantic] dataset=\$DATASET_NAME"
echo "[qz-text-prosody-full-semantic] vcdata_jsonls=\${VCDATA_JSONLS:-<default>}"
echo "[qz-text-prosody-full-semantic] seedvc_shards=\$SEEDVC_SHARD_COUNT codec_shards=\$CODEC_SHARD_COUNT speaker_shards=\$SPEAKER_SHARD_COUNT"
echo "[qz-text-prosody-full-semantic] asr_shards=\$ASR_NUM_SHARDS hubert_shards=\$HUBERT_NUM_SHARDS hubert_pre_split=\$HUBERT_PRE_SPLIT"
echo "[qz-text-prosody-full-semantic] semantic_gpu_keepalive=\$SEMANTIC_GPU_KEEPALIVE ids=\$SEMANTIC_GPU_KEEPALIVE_GPU_IDS"
echo "[qz-text-prosody-full-semantic] priority=$PRIORITY"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

bash scripts/001043_run_text_prosody_full_semantic_pipeline.sh
EOF
chmod +x "$RUNNER"

COMMAND="bash $RUNNER"
TMP_OUTPUT="$QZ_RECORD_ROOT/submit_output.txt"
rm -f "$TMP_OUTPUT"

echo "=========================================="
echo "QZ submit: full text_prosody semantic pipeline"
echo "  JOB_NAME=$JOB_NAME"
echo "  DATASET_NAME=$DATASET_NAME"
echo "  DATASET_ROOT=$DATASET_ROOT"
echo "  VCDATA_JSONLS=${VCDATA_JSONLS:-<default 001035 list>}"
echo "  RUNNER=$RUNNER"
echo "  WORKSPACE=$WORKSPACE PROJECT=$PROJECT COMPUTE_GROUP=$COMPUTE_GROUP SPEC=$SPEC"
echo "  PRIORITY=$PRIORITY IMAGE=$IMAGE"
echo "  COMMAND=$COMMAND"
echo "=========================================="

qz_args=(
  -m qzcli.cli create-job
  --name "$JOB_NAME"
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
  --command "$COMMAND"
)
if [ "$DRY_RUN" -eq 1 ]; then
  qz_args+=(--dry-run)
fi

set +e
env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
  HOME="$QZCLI_HOME" \
  PYTHONPATH="$QZCLI_TOOL" \
  "$QZ_PY" "${qz_args[@]}" >"$TMP_OUTPUT" 2>&1
STATUS=$?
set -e

cat "$TMP_OUTPUT"

if [ "$STATUS" -ne 0 ]; then
  echo "Submission failed. Output saved to $TMP_OUTPUT" >&2
  exit "$STATUS"
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry-run] Runner generated but no QZ job was submitted."
  echo "[dry-run] Inspect: sed -n '1,240p' $RUNNER"
  exit 0
fi

JOB_ID=$(grep -Eo 'job-[0-9a-fA-F-]{36}' "$TMP_OUTPUT" | tail -n 1 || true)
if [ -z "$JOB_ID" ]; then
  JOB_UUID=$(grep -E '任务ID|job_id|Job ID' "$TMP_OUTPUT" | grep -Eo '[0-9a-fA-F-]{36}' | tail -n 1 || true)
  if [ -n "$JOB_UUID" ]; then
    JOB_ID="job-$JOB_UUID"
  fi
fi

{
  printf 'job_name\tjob_id\tpriority\tcompute_group\trunner\tdataset_root\tvcdata_jsonls\n'
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$JOB_NAME" "${JOB_ID:-}" "$PRIORITY" "$COMPUTE_GROUP" "$RUNNER" "$DATASET_ROOT" "${VCDATA_JSONLS:-<default>}"
} > "$QZ_RECORD_ROOT/submitted_jobs.tsv"

echo "=========================================="
echo "Submission completed."
echo "  JOB_NAME=$JOB_NAME"
echo "  JOB_ID=${JOB_ID:-not parsed}"
echo "  RECORD=$QZ_RECORD_ROOT/submitted_jobs.tsv"
echo "=========================================="

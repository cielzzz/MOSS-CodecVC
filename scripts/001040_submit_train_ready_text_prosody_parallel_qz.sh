#!/usr/bin/env bash
set -euo pipefail

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
DATASET_NAME="${DATASET_NAME:-zh3w_en3w_text_prosody_independent_timbre}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/trainset/$DATASET_NAME}"
INPUT_JSONL="${INPUT_JSONL:-$DATASET_ROOT/manifests/vc_manifest.$DATASET_NAME.jsonl}"
BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%d-%H%M%S)}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codec-vc-text-prosody-train-ready-parallel}"
JOB_NAME="${JOB_NAME:-$JOB_NAME_PREFIX-$BATCH_ID}"
QZ_RECORD_ROOT="${QZ_RECORD_ROOT:-$ROOT/trainset/qz_jobs/$BATCH_ID}"
RUNNER="$QZ_RECORD_ROOT/run_train_ready_text_prosody_parallel_entrypoint.sh"

N_VQ="${N_VQ:-32}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
CODEC_GPU_IDS="${CODEC_GPU_IDS:-0,1,2,3,4,5,6,7}"
SPEAKER_GPU_IDS="${SPEAKER_GPU_IDS:-0,1,2,3,4,5,6,7}"
CODEC_SHARD_COUNT="${CODEC_SHARD_COUNT:-8}"
SPEAKER_SHARD_COUNT="${SPEAKER_SHARD_COUNT:-8}"
PROSODY_SHARD_COUNT="${PROSODY_SHARD_COUNT:-16}"
EMIT_MODES="${EMIT_MODES:-text}"
MAX_ROWS="${MAX_ROWS:-0}"
FORCE="${FORCE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
GPU_KEEPALIVE="${GPU_KEEPALIVE:-1}"
WAIT_HEARTBEAT_SECS="${WAIT_HEARTBEAT_SECS:-60}"
WRITE_TRAIN_COMMAND="${WRITE_TRAIN_COMMAND:-1}"

DRY_RUN=0
usage() {
  cat <<EOF
Usage:
  bash scripts/001040_submit_train_ready_text_prosody_parallel_qz.sh [--dry-run]

Common overrides:
  PRIORITY=10 bash scripts/001040_submit_train_ready_text_prosody_parallel_qz.sh
  CODEC_GPU_IDS=0,1,2,3,4,5,6,7 SPEAKER_GPU_IDS=0,1,2,3,4,5,6,7 CODEC_SHARD_COUNT=8 SPEAKER_SHARD_COUNT=8 bash scripts/001040_submit_train_ready_text_prosody_parallel_qz.sh

Notes:
  - Default priority follows the global convention: 3.
  - Defaults target 8x H200: codec and ECAPA are both sharded over all 8 GPUs, while prosody is CPU-sharded.
  - This job runs codec/SFT, speaker extraction, and prosody sidecar in parallel, then merges final train JSONL.
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
if [ "$DRY_RUN" -ne 1 ] && [ ! -s "$INPUT_JSONL" ]; then
  echo "ERROR: INPUT_JSONL missing or empty: $INPUT_JSONL" >&2
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

export ROOT="$ROOT"
export PY="$PY"
export DATASET_NAME="$DATASET_NAME"
export DATASET_ROOT="$DATASET_ROOT"
export INPUT_JSONL="$INPUT_JSONL"
export N_VQ="$N_VQ"
export GPU_IDS="$GPU_IDS"
export CODEC_GPU_IDS="$CODEC_GPU_IDS"
export SPEAKER_GPU_IDS="$SPEAKER_GPU_IDS"
export CODEC_SHARD_COUNT="$CODEC_SHARD_COUNT"
export SPEAKER_SHARD_COUNT="$SPEAKER_SHARD_COUNT"
export PROSODY_SHARD_COUNT="$PROSODY_SHARD_COUNT"
export EMIT_MODES="$EMIT_MODES"
export MAX_ROWS="$MAX_ROWS"
export FORCE="$FORCE"
export SKIP_EXISTING="$SKIP_EXISTING"
export GPU_KEEPALIVE="$GPU_KEEPALIVE"
export WAIT_HEARTBEAT_SECS="$WAIT_HEARTBEAT_SECS"
export WRITE_TRAIN_COMMAND="$WRITE_TRAIN_COMMAND"

export HF_ENDPOINT="\${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="\${OMP_NUM_THREADS:-8}"

cd "\$ROOT"
echo "[qz-text-train-ready-parallel] date=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[qz-text-train-ready-parallel] host=\$(hostname)"
echo "[qz-text-train-ready-parallel] dataset_name=\$DATASET_NAME"
echo "[qz-text-train-ready-parallel] input=\$INPUT_JSONL"
echo "[qz-text-train-ready-parallel] codec_gpu_ids=\$CODEC_GPU_IDS speaker_gpu_ids=\$SPEAKER_GPU_IDS"
echo "[qz-text-train-ready-parallel] priority=$PRIORITY"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

bash scripts/001039_run_train_ready_text_prosody_parallel.sh
EOF
chmod +x "$RUNNER"

COMMAND="bash $RUNNER"
TMP_OUTPUT="$QZ_RECORD_ROOT/submit_output.txt"
rm -f "$TMP_OUTPUT"

echo "=========================================="
echo "QZ submit: text_prosody train-ready parallel preprocessing"
echo "  JOB_NAME=$JOB_NAME"
echo "  ROOT=$ROOT"
echo "  DATASET_NAME=$DATASET_NAME"
echo "  DATASET_ROOT=$DATASET_ROOT"
echo "  INPUT_JSONL=$INPUT_JSONL"
echo "  RUNNER=$RUNNER"
echo "  WORKSPACE=$WORKSPACE"
echo "  PROJECT=$PROJECT"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP"
echo "  SPEC=$SPEC"
echo "  PRIORITY=$PRIORITY"
echo "  IMAGE=$IMAGE"
echo "  CODEC_GPU_IDS=$CODEC_GPU_IDS CODEC_SHARD_COUNT=$CODEC_SHARD_COUNT"
echo "  SPEAKER_GPU_IDS=$SPEAKER_GPU_IDS SPEAKER_SHARD_COUNT=$SPEAKER_SHARD_COUNT"
echo "  PROSODY_SHARD_COUNT=$PROSODY_SHARD_COUNT"
echo "  GPU_KEEPALIVE=$GPU_KEEPALIVE"
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
  printf 'job_name\tjob_id\tpriority\tcompute_group\trunner\tdataset_root\tinput_jsonl\n'
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$JOB_NAME" "${JOB_ID:-}" "$PRIORITY" "$COMPUTE_GROUP" "$RUNNER" "$DATASET_ROOT" "$INPUT_JSONL"
} > "$QZ_RECORD_ROOT/submitted_jobs.tsv"

echo "=========================================="
echo "Submission completed."
echo "  JOB_NAME=$JOB_NAME"
echo "  JOB_ID=${JOB_ID:-not parsed}"
echo "  RECORD=$QZ_RECORD_ROOT/submitted_jobs.tsv"
echo "=========================================="

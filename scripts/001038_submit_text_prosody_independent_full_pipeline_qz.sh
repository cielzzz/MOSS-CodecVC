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
BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%d-%H%M%S)}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codec-vc-text-prosody-full-pipeline}"
JOB_NAME="${JOB_NAME:-$JOB_NAME_PREFIX-$BATCH_ID}"
QZ_RECORD_ROOT="${QZ_RECORD_ROOT:-$ROOT/trainset/qz_jobs/$BATCH_ID}"
RUNNER="$QZ_RECORD_ROOT/run_text_prosody_independent_full_pipeline_entrypoint.sh"

RUN_TEXT_TRIPLE_STAGE="${RUN_TEXT_TRIPLE_STAGE:-1}"
RUN_TRAIN_READY_STAGE="${RUN_TRAIN_READY_STAGE:-1}"
RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_SEEDVC="${RUN_SEEDVC:-1}"
RUN_COLLECT="${RUN_COLLECT:-1}"
PREPARE_REQUIRE_EXISTING_AUDIO="${PREPARE_REQUIRE_EXISTING_AUDIO:-1}"
PREPARE_PROGRESS_EVERY="${PREPARE_PROGRESS_EVERY:-1000}"
RUN_PROSODY_FEATURES="${RUN_PROSODY_FEATURES:-1}"

LANGUAGES="${LANGUAGES:-zh,en}"
TIMBRE_REF_POLICY="${TIMBRE_REF_POLICY:-random_different_text}"
TIMBRE_REF_SEED="${TIMBRE_REF_SEED:-20260627}"
MAX_JOBS="${MAX_JOBS:-0}"
MAX_JOBS_PER_LANGUAGE="${MAX_JOBS_PER_LANGUAGE:-0}"
MAX_ROWS_PER_INPUT="${MAX_ROWS_PER_INPUT:-0}"
MIN_BEST_SIMILARITY="${MIN_BEST_SIMILARITY:-0.0}"
MIN_DNSMOS="${MIN_DNSMOS:-0.0}"
SKIP_FLAGS="${SKIP_FLAGS:-}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
SEEDVC_GPU_IDS="${SEEDVC_GPU_IDS:-$GPU_IDS}"
SEEDVC_SHARD_COUNT="${SEEDVC_SHARD_COUNT:-8}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-25}"
LENGTH_ADJUST="${LENGTH_ADJUST:-1.0}"
INFERENCE_CFG_RATE="${INFERENCE_CFG_RATE:-0.7}"
FP16="${FP16:-true}"
SEEDVC_SKIP_EXISTING="${SEEDVC_SKIP_EXISTING:-1}"
FAIL_FAST="${FAIL_FAST:-0}"
SHOW_MODEL_OUTPUT="${SHOW_MODEL_OUTPUT:-0}"
MIN_TARGET_AUDIO_BYTES="${MIN_TARGET_AUDIO_BYTES:-4096}"

N_VQ="${N_VQ:-32}"
CODEC_SHARD_COUNT="${CODEC_SHARD_COUNT:-8}"
SPEAKER_SHARD_COUNT="${SPEAKER_SHARD_COUNT:-8}"
PROSODY_SHARD_COUNT="${PROSODY_SHARD_COUNT:-16}"
SPEAKER_DEVICE="${SPEAKER_DEVICE:-cuda:0}"
ATTACH_REQUIRE_EMBEDDING_EXISTS="${ATTACH_REQUIRE_EMBEDDING_EXISTS:-1}"
GPU_KEEPALIVE="${GPU_KEEPALIVE:-0}"
GPU_KEEPALIVE_STAGES="${GPU_KEEPALIVE_STAGES:-speaker_extract,attach,prosody_extract}"
GPU_KEEPALIVE_GPU_IDS="${GPU_KEEPALIVE_GPU_IDS:-$GPU_IDS}"
WAIT_HEARTBEAT_SECS="${WAIT_HEARTBEAT_SECS:-60}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE="${FORCE:-0}"
WRITE_TRAIN_COMMAND="${WRITE_TRAIN_COMMAND:-1}"
RUN_CODEC="${RUN_CODEC:-1}"
RUN_SFT="${RUN_SFT:-1}"
RUN_SPEAKER_PLAN="${RUN_SPEAKER_PLAN:-1}"
RUN_SPEAKER_EXTRACT="${RUN_SPEAKER_EXTRACT:-1}"
RUN_ATTACH="${RUN_ATTACH:-1}"

DRY_RUN=0

usage() {
  cat <<EOF
Usage:
  bash scripts/001038_submit_text_prosody_independent_full_pipeline_qz.sh [--dry-run]

Common overrides:
  MAX_JOBS_PER_LANGUAGE=5 SKIP_FLAGS=LOW_SIM bash scripts/001038_submit_text_prosody_independent_full_pipeline_qz.sh
  SKIP_FLAGS=LOW_SIM DATASET_NAME=zh_en_text_prosody_independent_ok_only bash scripts/001038_submit_text_prosody_independent_full_pipeline_qz.sh

Notes:
  - Default priority is 3.
  - Job names do not include an xyzhang- prefix.
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

export ROOT="$ROOT"
export PY="$PY"
export DATASET_NAME="$DATASET_NAME"
export DATASET_ROOT="$DATASET_ROOT"
export RUN_TEXT_TRIPLE_STAGE="$RUN_TEXT_TRIPLE_STAGE"
export RUN_TRAIN_READY_STAGE="$RUN_TRAIN_READY_STAGE"
export RUN_PREPARE="$RUN_PREPARE"
export RUN_SEEDVC="$RUN_SEEDVC"
export RUN_COLLECT="$RUN_COLLECT"
export PREPARE_REQUIRE_EXISTING_AUDIO="$PREPARE_REQUIRE_EXISTING_AUDIO"
export PREPARE_PROGRESS_EVERY="$PREPARE_PROGRESS_EVERY"
export RUN_PROSODY_FEATURES="$RUN_PROSODY_FEATURES"
export LANGUAGES="$LANGUAGES"
export TIMBRE_REF_POLICY="$TIMBRE_REF_POLICY"
export TIMBRE_REF_SEED="$TIMBRE_REF_SEED"
export MAX_JOBS="$MAX_JOBS"
export MAX_JOBS_PER_LANGUAGE="$MAX_JOBS_PER_LANGUAGE"
export MAX_ROWS_PER_INPUT="$MAX_ROWS_PER_INPUT"
export MIN_BEST_SIMILARITY="$MIN_BEST_SIMILARITY"
export MIN_DNSMOS="$MIN_DNSMOS"
export SKIP_FLAGS="$SKIP_FLAGS"
export GPU_IDS="$GPU_IDS"
export SEEDVC_GPU_IDS="$SEEDVC_GPU_IDS"
export SEEDVC_SHARD_COUNT="$SEEDVC_SHARD_COUNT"
export DIFFUSION_STEPS="$DIFFUSION_STEPS"
export LENGTH_ADJUST="$LENGTH_ADJUST"
export INFERENCE_CFG_RATE="$INFERENCE_CFG_RATE"
export FP16="$FP16"
export SEEDVC_SKIP_EXISTING="$SEEDVC_SKIP_EXISTING"
export FAIL_FAST="$FAIL_FAST"
export SHOW_MODEL_OUTPUT="$SHOW_MODEL_OUTPUT"
export MIN_TARGET_AUDIO_BYTES="$MIN_TARGET_AUDIO_BYTES"
export N_VQ="$N_VQ"
export CODEC_SHARD_COUNT="$CODEC_SHARD_COUNT"
export SPEAKER_SHARD_COUNT="$SPEAKER_SHARD_COUNT"
export PROSODY_SHARD_COUNT="$PROSODY_SHARD_COUNT"
export SPEAKER_DEVICE="$SPEAKER_DEVICE"
export ATTACH_REQUIRE_EMBEDDING_EXISTS="$ATTACH_REQUIRE_EMBEDDING_EXISTS"
export GPU_KEEPALIVE="$GPU_KEEPALIVE"
export GPU_KEEPALIVE_STAGES="$GPU_KEEPALIVE_STAGES"
export GPU_KEEPALIVE_GPU_IDS="$GPU_KEEPALIVE_GPU_IDS"
export WAIT_HEARTBEAT_SECS="$WAIT_HEARTBEAT_SECS"
export SKIP_EXISTING="$SKIP_EXISTING"
export FORCE="$FORCE"
export WRITE_TRAIN_COMMAND="$WRITE_TRAIN_COMMAND"
export RUN_CODEC="$RUN_CODEC"
export RUN_SFT="$RUN_SFT"
export RUN_SPEAKER_PLAN="$RUN_SPEAKER_PLAN"
export RUN_SPEAKER_EXTRACT="$RUN_SPEAKER_EXTRACT"
export RUN_ATTACH="$RUN_ATTACH"

export HF_ENDPOINT="\${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="\${OMP_NUM_THREADS:-8}"

cd "\$ROOT"
echo "[qz-text-prosody-full] date=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[qz-text-prosody-full] host=\$(hostname)"
echo "[qz-text-prosody-full] dataset_name=\$DATASET_NAME"
echo "[qz-text-prosody-full] dataset_root=\$DATASET_ROOT"
echo "[qz-text-prosody-full] priority=$PRIORITY"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

bash scripts/001037_run_text_prosody_independent_full_pipeline.sh
EOF
chmod +x "$RUNNER"

COMMAND="bash $RUNNER"
TMP_OUTPUT="$QZ_RECORD_ROOT/submit_output.txt"
rm -f "$TMP_OUTPUT"

echo "=========================================="
echo "QZ submit: text_prosody independent-timbre full pipeline"
echo "  JOB_NAME=$JOB_NAME"
echo "  ROOT=$ROOT"
echo "  DATASET_NAME=$DATASET_NAME"
echo "  DATASET_ROOT=$DATASET_ROOT"
echo "  RUNNER=$RUNNER"
echo "  WORKSPACE=$WORKSPACE"
echo "  PROJECT=$PROJECT"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP"
echo "  SPEC=$SPEC"
echo "  PRIORITY=$PRIORITY"
echo "  IMAGE=$IMAGE"
echo "  MAX_JOBS=$MAX_JOBS"
echo "  MAX_JOBS_PER_LANGUAGE=$MAX_JOBS_PER_LANGUAGE"
echo "  SKIP_FLAGS=$SKIP_FLAGS"
echo "  SEEDVC_SHARD_COUNT=$SEEDVC_SHARD_COUNT"
echo "  CODEC_SHARD_COUNT=$CODEC_SHARD_COUNT"
echo "  SPEAKER_SHARD_COUNT=$SPEAKER_SHARD_COUNT"
echo "  PROSODY_SHARD_COUNT=$PROSODY_SHARD_COUNT"
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
  printf 'job_name\tjob_id\tpriority\tcompute_group\trunner\tdataset_root\n'
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$JOB_NAME" "${JOB_ID:-}" "$PRIORITY" "$COMPUTE_GROUP" "$RUNNER" "$DATASET_ROOT"
} > "$QZ_RECORD_ROOT/submitted_jobs.tsv"

echo "=========================================="
echo "Submission completed."
echo "  JOB_NAME=$JOB_NAME"
echo "  JOB_ID=${JOB_ID:-not parsed}"
echo "  RECORD=$QZ_RECORD_ROOT/submitted_jobs.tsv"
echo "=========================================="

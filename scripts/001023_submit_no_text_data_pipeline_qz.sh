#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
QZCLI_TOOL="${QZCLI_TOOL:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/qzcli_tool}"
QZ_PY="${QZ_PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
QZCLI_HOME="${QZCLI_HOME:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/.codex/qzcli_home}"

WORKSPACE="${WORKSPACE:-ws-8207e9e2-e733-4eec-a475-cfa1c36480ba}"
PROJECT="${PROJECT:-project-c67c548f-f02c-453b-ba5b-8745db6886e7}"
COMPUTE_GROUP="${COMPUTE_GROUP:-lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122}"  # MTTS-3-2-0715
FRAMEWORK="${FRAMEWORK:-pytorch}"
INSTANCES="${INSTANCES:-1}"
SHM_GI="${SHM_GI:-1200}"
PRIORITY="${PRIORITY:-3}"
SPEC="${SPEC:-67b10bc6-78b0-41a3-aaf4-358eeeb99009}"
IMAGE="${IMAGE:-docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420}"
IMAGE_TYPE="${IMAGE_TYPE:-SOURCE_PRIVATE}"

DATASET_NAME="${DATASET_NAME:-zh45w_en22w_no_text}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/trainset/$DATASET_NAME}"
RAW_INPUT_ROOT="${RAW_INPUT_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vc_data_temp/mtd_pass_nonmulti_primary_le_0p3_split_10k}"
PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"

BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%d-%H%M%S)}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codec-vc-no-text-data-pipeline}"
JOB_NAME="${JOB_NAME:-$JOB_NAME_PREFIX-$BATCH_ID}"
QZ_RECORD_ROOT="${QZ_RECORD_ROOT:-$ROOT/trainset/qz_jobs/$BATCH_ID}"
RUNNER="$QZ_RECORD_ROOT/run_no_text_data_pipeline_entrypoint.sh"

RUN_TRIPLE_STAGE="${RUN_TRIPLE_STAGE:-1}"
RUN_TRAIN_READY_STAGE="${RUN_TRAIN_READY_STAGE:-1}"
RUN_SEEDVC="${RUN_SEEDVC:-true}"
RUN_PROSODY_FEATURES="${RUN_PROSODY_FEATURES:-1}"
RESUME_EXISTING="${RESUME_EXISTING:-true}"
LANGUAGES="${LANGUAGES:-zh,en}"
EMIT_PAIR_TYPES="${EMIT_PAIR_TYPES:-no_text}"
MAX_ROWS="${MAX_ROWS:-0}"
MAX_PAIRS="${MAX_PAIRS:-0}"
MIN_DURATION_SEC="${MIN_DURATION_SEC:-1.0}"
MAX_DURATION_SEC="${MAX_DURATION_SEC:-30.0}"
TEXT_SOURCE_POLICY="${TEXT_SOURCE_POLICY:-different_speaker}"
INFER_SPEAKER_FROM="${INFER_SPEAKER_FROM:-auto}"
ALLOW_MISSING_AUDIO="${ALLOW_MISSING_AUDIO:-false}"
ALLOW_MISSING_SPEAKER="${ALLOW_MISSING_SPEAKER:-false}"
AUGMENT_SOURCE_JSONL="${AUGMENT_SOURCE_JSONL:-true}"
REQUIRE_TARGET_AUDIO="${REQUIRE_TARGET_AUDIO:-true}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100000}"
SEED="${SEED:-42}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
SEEDVC_GPU_IDS="${SEEDVC_GPU_IDS:-$GPU_IDS}"
SEEDVC_SHARD_COUNT="${SEEDVC_SHARD_COUNT:-8}"
SEEDVC_MAX_JOBS="${SEEDVC_MAX_JOBS:-0}"
SEEDVC_SKIP_EXISTING="${SEEDVC_SKIP_EXISTING:-1}"
SEEDVC_FAIL_FAST="${SEEDVC_FAIL_FAST:-0}"
SEEDVC_SHOW_MODEL_OUTPUT="${SEEDVC_SHOW_MODEL_OUTPUT:-0}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-25}"
LENGTH_ADJUST="${LENGTH_ADJUST:-1.0}"
INFERENCE_CFG_RATE="${INFERENCE_CFG_RATE:-0.7}"
SEEDVC_FP16="${SEEDVC_FP16:-true}"
N_VQ="${N_VQ:-32}"
CODEC_SHARD_COUNT="${CODEC_SHARD_COUNT:-8}"
CODEC_DEVICE="${CODEC_DEVICE:-cuda:0}"
SPEAKER_SHARD_COUNT="${SPEAKER_SHARD_COUNT:-8}"
PROSODY_SHARD_COUNT="${PROSODY_SHARD_COUNT:-16}"
SPEAKER_DEVICE="${SPEAKER_DEVICE:-cuda:0}"
ATTACH_REQUIRE_EMBEDDING_EXISTS="${ATTACH_REQUIRE_EMBEDDING_EXISTS:-0}"
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
  bash scripts/001023_submit_no_text_data_pipeline_qz.sh [--dry-run]

Common overrides:
  DATASET_NAME=new_dataset RAW_INPUT_ROOT=/abs/raw/jsonl/root bash scripts/001023_submit_no_text_data_pipeline_qz.sh
  RUN_TRIPLE_STAGE=0 DATASET_NAME=existing_dataset bash scripts/001023_submit_no_text_data_pipeline_qz.sh
  MAX_ROWS=1000 MAX_PAIRS=1000 SEEDVC_MAX_JOBS=1000 DATASET_NAME=smoke_trainset bash scripts/001023_submit_no_text_data_pipeline_qz.sh
  SPEAKER_DEVICE=cpu GPU_KEEPALIVE=1 bash scripts/001023_submit_no_text_data_pipeline_qz.sh

Notes:
  - Default compute group is MTTS-3-2-0715.
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
export RAW_INPUT_ROOT="$RAW_INPUT_ROOT"
export RUN_TRIPLE_STAGE="$RUN_TRIPLE_STAGE"
export RUN_TRAIN_READY_STAGE="$RUN_TRAIN_READY_STAGE"
export RUN_SEEDVC="$RUN_SEEDVC"
export RUN_PROSODY_FEATURES="$RUN_PROSODY_FEATURES"
export RESUME_EXISTING="$RESUME_EXISTING"
export LANGUAGES="$LANGUAGES"
export EMIT_PAIR_TYPES="$EMIT_PAIR_TYPES"
export MAX_ROWS="$MAX_ROWS"
export MAX_PAIRS="$MAX_PAIRS"
export MIN_DURATION_SEC="$MIN_DURATION_SEC"
export MAX_DURATION_SEC="$MAX_DURATION_SEC"
export TEXT_SOURCE_POLICY="$TEXT_SOURCE_POLICY"
export INFER_SPEAKER_FROM="$INFER_SPEAKER_FROM"
export ALLOW_MISSING_AUDIO="$ALLOW_MISSING_AUDIO"
export ALLOW_MISSING_SPEAKER="$ALLOW_MISSING_SPEAKER"
export AUGMENT_SOURCE_JSONL="$AUGMENT_SOURCE_JSONL"
export REQUIRE_TARGET_AUDIO="$REQUIRE_TARGET_AUDIO"
export PROGRESS_EVERY="$PROGRESS_EVERY"
export SEED="$SEED"
export GPU_IDS="$GPU_IDS"
export SEEDVC_GPU_IDS="$SEEDVC_GPU_IDS"
export SEEDVC_SHARD_COUNT="$SEEDVC_SHARD_COUNT"
export SEEDVC_MAX_JOBS="$SEEDVC_MAX_JOBS"
export SEEDVC_SKIP_EXISTING="$SEEDVC_SKIP_EXISTING"
export SEEDVC_FAIL_FAST="$SEEDVC_FAIL_FAST"
export SEEDVC_SHOW_MODEL_OUTPUT="$SEEDVC_SHOW_MODEL_OUTPUT"
export DIFFUSION_STEPS="$DIFFUSION_STEPS"
export LENGTH_ADJUST="$LENGTH_ADJUST"
export INFERENCE_CFG_RATE="$INFERENCE_CFG_RATE"
export SEEDVC_FP16="$SEEDVC_FP16"
export N_VQ="$N_VQ"
export CODEC_SHARD_COUNT="$CODEC_SHARD_COUNT"
export CODEC_DEVICE="$CODEC_DEVICE"
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
echo "[qz-data-pipeline] date=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[qz-data-pipeline] host=\$(hostname)"
echo "[qz-data-pipeline] dataset_name=\$DATASET_NAME"
echo "[qz-data-pipeline] dataset_root=\$DATASET_ROOT"
echo "[qz-data-pipeline] raw_input_root=\$RAW_INPUT_ROOT"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

bash scripts/001024_run_no_text_data_pipeline_full.sh
EOF
chmod +x "$RUNNER"

COMMAND="bash $RUNNER"

echo "=========================================="
echo "QZ submit: no-text data pipeline"
echo "  JOB_NAME=$JOB_NAME"
echo "  ROOT=$ROOT"
echo "  DATASET_NAME=$DATASET_NAME"
echo "  DATASET_ROOT=$DATASET_ROOT"
echo "  RAW_INPUT_ROOT=$RAW_INPUT_ROOT"
echo "  RUNNER=$RUNNER"
echo "  WORKSPACE=$WORKSPACE"
echo "  PROJECT=$PROJECT"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP"
echo "  SPEC=$SPEC"
echo "  PRIORITY=$PRIORITY"
echo "  IMAGE=$IMAGE"
echo "  RUN_TRIPLE_STAGE=$RUN_TRIPLE_STAGE"
echo "  RUN_TRAIN_READY_STAGE=$RUN_TRAIN_READY_STAGE"
echo "  MAX_ROWS=$MAX_ROWS"
echo "  MAX_PAIRS=$MAX_PAIRS"
echo "  SEEDVC_MAX_JOBS=$SEEDVC_MAX_JOBS"
echo "  N_VQ=$N_VQ"
echo "  GPU_KEEPALIVE=$GPU_KEEPALIVE"
echo "  COMMAND=$COMMAND"
echo "=========================================="

TMP_OUTPUT="$QZ_RECORD_ROOT/submit_output.txt"
rm -f "$TMP_OUTPUT"

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
  echo "[dry-run] Inspect: sed -n '1,220p' $RUNNER"
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
  printf 'job_name\tjob_id\tcompute_group\trunner\tdataset_root\n'
  printf '%s\t%s\t%s\t%s\t%s\n' "$JOB_NAME" "${JOB_ID:-}" "$COMPUTE_GROUP" "$RUNNER" "$DATASET_ROOT"
} > "$QZ_RECORD_ROOT/submitted_jobs.tsv"

echo "=========================================="
echo "Submission completed."
echo "  JOB_NAME=$JOB_NAME"
echo "  JOB_ID=${JOB_ID:-not parsed}"
echo "  RECORD=$QZ_RECORD_ROOT/submitted_jobs.tsv"
echo "=========================================="

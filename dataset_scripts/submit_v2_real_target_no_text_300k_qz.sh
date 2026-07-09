#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
DATA_ROOT="${DATA_ROOT:-/inspire/qb-ilm/project/embodied-multimodality/public/xyzhang/data/VC_train}"
QZCLI_TOOL="${QZCLI_TOOL:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/qzcli_tool}"
QZ_PY="${QZ_PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
QZCLI_HOME="${QZCLI_HOME:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/.codex/qzcli_home}"

WORKSPACE="${WORKSPACE:-ws-8207e9e2-e733-4eec-a475-cfa1c36480ba}"  # CI-情境智能
PROJECT="${PROJECT:-project-c67c548f-f02c-453b-ba5b-8745db6886e7}"
COMPUTE_GROUP="${COMPUTE_GROUP:-lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122}"  # MTTS-3-2-0715
SPEC="${SPEC:-67b10bc6-78b0-41a3-aaf4-358eeeb99009}"  # 8x H200
FRAMEWORK="${FRAMEWORK:-pytorch}"
INSTANCES="${INSTANCES:-1}"
SHM_GI="${SHM_GI:-1200}"
PRIORITY="${PRIORITY:-10}"
IMAGE="${IMAGE:-docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420}"
IMAGE_TYPE="${IMAGE_TYPE:-SOURCE_PRIVATE}"
QZ_SUBMIT_API="${QZ_SUBMIT_API:-v2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%d-%H%M%S)}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codec-vc-v2-real-target-no-text-300k}"
JOB_NAME="${JOB_NAME:-${JOB_NAME_PREFIX}-${BATCH_ID}}"
QZ_RECORD_ROOT="${QZ_RECORD_ROOT:-${DATA_ROOT}/qz_jobs/${BATCH_ID}}"
RUNNER="${QZ_RECORD_ROOT}/run_v2_real_target_no_text_300k_entrypoint.sh"

RUN_NAME="${RUN_NAME:-v2_real_target_no_text_300k_zh_en_balanced_20260707}"
PAIR_ROOT="${PAIR_ROOT:-${DATA_ROOT}/${RUN_NAME}}"
TRIPLE_ROOT="${TRIPLE_ROOT:-${DATA_ROOT}/${RUN_NAME}_seedvc_triples}"
NUM_NO_TEXT="${NUM_NO_TEXT:-300000}"
MAX_PER_DATASET="${MAX_PER_DATASET:-60000}"
REF_MAX_SEC="${REF_MAX_SEC:-30.0}"
LANGUAGES="${LANGUAGES:-zh,en}"
BALANCE_LANGUAGES="${BALANCE_LANGUAGES:-1}"
DATASETS="${DATASETS:-apple_podcast_estwfiruchnonltzph,apple_podcast_josailczinplittrropkthmauavelyye,apple_podcast_vnjpidgrfrkemyptpededk,haitianruisheng_1,haitianruisheng_2,haitianruisheng_3,haitianruisheng_4,haitianruisheng_6,haitianruisheng_7,haitianruisheng_8,haitianruisheng_9,qingting_fm,rchive_rss_podcast_v2}"
SEEDVC_GPU_IDS="${SEEDVC_GPU_IDS:-0,1,2,3,4,5,6,7}"
SEEDVC_SHARD_COUNT="${SEEDVC_SHARD_COUNT:-8}"
MAX_JOBS="${MAX_JOBS:-0}"
RCLONE_TRANSFERS="${RCLONE_TRANSFERS:-32}"
RCLONE_CHECKERS="${RCLONE_CHECKERS:-64}"
DOWNLOAD_AUDIO="${DOWNLOAD_AUDIO:-0}"
RUN_PAIR_PREPARE="${RUN_PAIR_PREPARE:-0}"
RUN_TRIPLE_PREPARE="${RUN_TRIPLE_PREPARE:-0}"
RUN_SEEDVC="${RUN_SEEDVC:-1}"
RUN_COLLECT="${RUN_COLLECT:-1}"
RUN_REF_CHANNEL_AUGMENT="${RUN_REF_CHANNEL_AUGMENT:-1}"
REF_AUG_RISK_MODE="${REF_AUG_RISK_MODE:-same_episode}"
REF_AUG_FRACTION="${REF_AUG_FRACTION:-0.3}"
REF_AUG_JOBS="${REF_AUG_JOBS:-16}"
REF_AUG_AUDIO_EXTENSION="${REF_AUG_AUDIO_EXTENSION:-.wav}"
REF_AUG_FFMPEG="${REF_AUG_FFMPEG:-/opt/conda/envs/speech/bin/ffmpeg}"
REF_AUG_LOUDNESS_MATCH="${REF_AUG_LOUDNESS_MATCH:-mean_volume}"

DRY_RUN=0

usage() {
  cat <<EOF
Usage:
  bash dataset_scripts/submit_v2_real_target_no_text_300k_qz.sh [--dry-run]

Common overrides:
  COMPUTE_GROUP=qz_zxy_gpu_4090 bash dataset_scripts/submit_v2_real_target_no_text_300k_qz.sh
  MAX_JOBS=1000 JOB_NAME=codec-vc-v2-no-text-smoke bash dataset_scripts/submit_v2_real_target_no_text_300k_qz.sh
  RUN_PAIR_PREPARE=0 RUN_TRIPLE_PREPARE=0 bash dataset_scripts/submit_v2_real_target_no_text_300k_qz.sh

This job builds 300k V2 no-text real-target pairs, runs Seed-VC on one 8-GPU node,
and collects the ready no-text manifest.
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

mkdir -p "$QZ_RECORD_ROOT" "$QZCLI_HOME" "$DATA_ROOT"

cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail

export DATA_ROOT="$DATA_ROOT"
export RUN_NAME="$RUN_NAME"
export PAIR_ROOT="$PAIR_ROOT"
export TRIPLE_ROOT="$TRIPLE_ROOT"
export NUM_NO_TEXT="$NUM_NO_TEXT"
export MAX_PER_DATASET="$MAX_PER_DATASET"
export REF_MAX_SEC="$REF_MAX_SEC"
export LANGUAGES="$LANGUAGES"
export BALANCE_LANGUAGES="$BALANCE_LANGUAGES"
export DATASETS="$DATASETS"
export RCLONE_TRANSFERS="$RCLONE_TRANSFERS"
export RCLONE_CHECKERS="$RCLONE_CHECKERS"
export DOWNLOAD_AUDIO="$DOWNLOAD_AUDIO"
export RUN_PAIR_PREPARE="$RUN_PAIR_PREPARE"
export RUN_TRIPLE_PREPARE="$RUN_TRIPLE_PREPARE"
export RUN_SEEDVC="$RUN_SEEDVC"
export RUN_COLLECT="$RUN_COLLECT"
export SEEDVC_GPU_IDS="$SEEDVC_GPU_IDS"
export SEEDVC_SHARD_COUNT="$SEEDVC_SHARD_COUNT"
export MAX_JOBS="$MAX_JOBS"
export RUN_REF_CHANNEL_AUGMENT="$RUN_REF_CHANNEL_AUGMENT"
export REF_AUG_RISK_MODE="$REF_AUG_RISK_MODE"
export REF_AUG_FRACTION="$REF_AUG_FRACTION"
export REF_AUG_JOBS="$REF_AUG_JOBS"
export REF_AUG_AUDIO_EXTENSION="$REF_AUG_AUDIO_EXTENSION"
export REF_AUG_FFMPEG="$REF_AUG_FFMPEG"
export REF_AUG_LOUDNESS_MATCH="$REF_AUG_LOUDNESS_MATCH"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="\${OMP_NUM_THREADS:-8}"
export HF_ENDPOINT="\${HF_ENDPOINT:-https://hf-mirror.com}"

cd "$ROOT"
echo "[qz-v2-no-text-300k] date=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[qz-v2-no-text-300k] host=\$(hostname)"
echo "[qz-v2-no-text-300k] data_root=\$DATA_ROOT"
echo "[qz-v2-no-text-300k] pair_root=\$PAIR_ROOT"
echo "[qz-v2-no-text-300k] triple_root=\$TRIPLE_ROOT"
echo "[qz-v2-no-text-300k] seedvc_gpu_ids=\$SEEDVC_GPU_IDS shard_count=\$SEEDVC_SHARD_COUNT max_jobs=\$MAX_JOBS"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
bash dataset_scripts/run_v2_real_target_no_text_300k_pipeline.sh
EOF
chmod +x "$RUNNER"

COMMAND="bash $RUNNER"
TMP_OUTPUT="$QZ_RECORD_ROOT/submit_output.txt"

echo "=========================================="
echo "QZ submit: V2 real-target no-text 300k"
echo "  JOB_NAME=$JOB_NAME"
echo "  ROOT=$ROOT"
echo "  DATA_ROOT=$DATA_ROOT"
echo "  RUN_NAME=$RUN_NAME"
echo "  PAIR_ROOT=$PAIR_ROOT"
echo "  TRIPLE_ROOT=$TRIPLE_ROOT"
echo "  NUM_NO_TEXT=$NUM_NO_TEXT"
echo "  MAX_PER_DATASET=$MAX_PER_DATASET"
echo "  REF_MAX_SEC=$REF_MAX_SEC"
echo "  LANGUAGES=$LANGUAGES"
echo "  BALANCE_LANGUAGES=$BALANCE_LANGUAGES"
echo "  DATASETS=$DATASETS"
echo "  DOWNLOAD_AUDIO=$DOWNLOAD_AUDIO"
echo "  RUN_PAIR_PREPARE=$RUN_PAIR_PREPARE"
echo "  RUN_TRIPLE_PREPARE=$RUN_TRIPLE_PREPARE"
echo "  RUN_SEEDVC=$RUN_SEEDVC"
echo "  RUN_COLLECT=$RUN_COLLECT"
echo "  RUNNER=$RUNNER"
echo "  WORKSPACE=$WORKSPACE"
echo "  PROJECT=$PROJECT"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP"
echo "  SPEC=${SPEC:-auto}"
echo "  PRIORITY=$PRIORITY"
echo "  IMAGE=$IMAGE"
echo "  QZ_SUBMIT_API=$QZ_SUBMIT_API"
echo "  GPUS_PER_NODE=$GPUS_PER_NODE"
echo "  SEEDVC_GPU_IDS=$SEEDVC_GPU_IDS"
echo "  SEEDVC_SHARD_COUNT=$SEEDVC_SHARD_COUNT"
echo "  MAX_JOBS=$MAX_JOBS"
echo "  RUN_REF_CHANNEL_AUGMENT=$RUN_REF_CHANNEL_AUGMENT"
echo "  REF_AUG_RISK_MODE=$REF_AUG_RISK_MODE"
echo "  REF_AUG_FRACTION=$REF_AUG_FRACTION"
echo "  REF_AUG_JOBS=$REF_AUG_JOBS"
echo "  REF_AUG_AUDIO_EXTENSION=$REF_AUG_AUDIO_EXTENSION"
echo "  REF_AUG_LOUDNESS_MATCH=$REF_AUG_LOUDNESS_MATCH"
echo "  COMMAND=$COMMAND"
echo "=========================================="

rm -f "$TMP_OUTPUT"

if [ "$QZ_SUBMIT_API" = "v2" ]; then
  qz_args=(
    dataset_scripts/qz_v2_submit_train_job.py
    --name "$JOB_NAME"
    --workspace "$WORKSPACE"
    --project "$PROJECT"
    --compute-group "$COMPUTE_GROUP"
    --framework "$FRAMEWORK"
    --instances "$INSTANCES"
    --shm-gi "$SHM_GI"
    --priority "$PRIORITY"
    --image "$IMAGE"
    --image-type "$IMAGE_TYPE"
    --command "$COMMAND"
    --gpus-per-node "$GPUS_PER_NODE"
    --payload-json "$QZ_RECORD_ROOT/submit_payload.v2.json"
    --response-json "$QZ_RECORD_ROOT/submit_output.v2.json"
  )
  if [ -n "$SPEC" ]; then
    qz_args+=(--spec "$SPEC")
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    qz_args+=(--dry-run)
  fi

  set +e
  env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
    HOME="$QZCLI_HOME" \
    PYTHONPATH="$QZCLI_TOOL" \
    "$QZ_PY" "${qz_args[@]}" >"$TMP_OUTPUT" 2>&1
  status=$?
  set -e
else
  qz_args=(
    -m qzcli.cli create-job
    --name "$JOB_NAME"
    --workspace "$WORKSPACE"
    --project "$PROJECT"
    --compute-group "$COMPUTE_GROUP"
    --framework "$FRAMEWORK"
    --instances "$INSTANCES"
    --shm "$SHM_GI"
    --priority "$PRIORITY"
    --image "$IMAGE"
    --image-type "$IMAGE_TYPE"
    --command "$COMMAND"
  )
  if [ -n "$SPEC" ]; then
    qz_args+=(--spec "$SPEC")
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    qz_args+=(--dry-run)
  fi

  set +e
  env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
    HOME="$QZCLI_HOME" \
    PYTHONPATH="$QZCLI_TOOL" \
    "$QZ_PY" "${qz_args[@]}" >"$TMP_OUTPUT" 2>&1
  status=$?
  set -e
fi

cat "$TMP_OUTPUT"
if [ "$status" -ne 0 ]; then
  echo "Submission failed. Output saved to $TMP_OUTPUT" >&2
  exit "$status"
fi
if [ "$DRY_RUN" -eq 1 ]; then
  echo "Dry run completed. No job was submitted."
  exit 0
fi

job_id=$(grep -Eo 'job-[0-9a-fA-F-]{36}' "$TMP_OUTPUT" | tail -n 1 || true)
if [ -z "$job_id" ]; then
  job_uuid=$(grep -E '任务ID|job_id|Job ID' "$TMP_OUTPUT" | grep -Eo '[0-9a-fA-F-]{36}' | tail -n 1 || true)
  if [ -n "$job_uuid" ]; then
    job_id="job-$job_uuid"
  fi
fi

{
  printf 'job_name\tjob_id\tcompute_group\trunner\tpair_root\ttriple_root\n'
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$JOB_NAME" "${job_id:-}" "$COMPUTE_GROUP" "$RUNNER" "$PAIR_ROOT" "$TRIPLE_ROOT"
} > "$QZ_RECORD_ROOT/submitted_jobs.tsv"

echo "=========================================="
echo "Submission completed."
echo "  JOB_NAME=$JOB_NAME"
echo "  JOB_ID=${job_id:-not parsed}"
echo "  RECORD=$QZ_RECORD_ROOT/submitted_jobs.tsv"
echo "=========================================="

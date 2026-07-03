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

BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%d-%H%M%S)}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codec-vc-text-prosody-seedvc}"
JOB_NAME="${JOB_NAME:-$JOB_NAME_PREFIX-$BATCH_ID}"
QZ_RECORD_ROOT="${QZ_RECORD_ROOT:-$ROOT/trainset/qz_jobs/$BATCH_ID}"
RUNNER="$QZ_RECORD_ROOT/run_text_prosody_seedvc_entrypoint.sh"

RUN_NAME="${RUN_NAME:-text_prosody_mosstts_seedvc_zh_en_0001_0004}"
WORK_ROOT="${WORK_ROOT:-$ROOT/trainset/text_prosody_mosstts_seedvc/zh_en_0001_0004}"
JOBS_JSONL="${JOBS_JSONL:-$WORK_ROOT/text_seedvc_jobs.jsonl}"
RESULTS_JSONL="${RESULTS_JSONL:-$WORK_ROOT/text_seedvc_results.jsonl}"
MANIFEST_JSONL="${MANIFEST_JSONL:-$WORK_ROOT/vc_manifest.text_prosody.jsonl}"
TARGET_AUDIO_ROOT="${TARGET_AUDIO_ROOT:-$WORK_ROOT/seedvc_targets}"

SEEDVC_GPU_IDS="${SEEDVC_GPU_IDS:-0,1,2,3,4,5,6,7}"
SEEDVC_SHARD_COUNT="${SEEDVC_SHARD_COUNT:-16}"
MAX_JOBS="${MAX_JOBS:-0}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-25}"
LENGTH_ADJUST="${LENGTH_ADJUST:-1.0}"
INFERENCE_CFG_RATE="${INFERENCE_CFG_RATE:-0.7}"
FP16="${FP16:-true}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FAIL_FAST="${FAIL_FAST:-0}"
SHOW_MODEL_OUTPUT="${SHOW_MODEL_OUTPUT:-0}"
MIN_TARGET_AUDIO_BYTES="${MIN_TARGET_AUDIO_BYTES:-4096}"

DRY_RUN=0

usage() {
  cat <<EOF
Usage:
  bash scripts/001036_submit_text_prosody_seedvc_qz.sh [--dry-run]

Common overrides:
  SEEDVC_SHARD_COUNT=8 bash scripts/001036_submit_text_prosody_seedvc_qz.sh
  MAX_JOBS=1000 JOB_NAME=codec-vc-text-prosody-smoke bash scripts/001036_submit_text_prosody_seedvc_qz.sh

This submits only Seed-VC generation plus final collect. Prepare jobs locally first with:
  RUN_SEEDVC=0 RUN_COLLECT=0 bash scripts/001035_run_text_prosody_mosstts_seedvc_pipeline.sh
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
if [ ! -f "$JOBS_JSONL" ]; then
  echo "ERROR: JOBS_JSONL does not exist: $JOBS_JSONL" >&2
  exit 1
fi
if [ ! -x "$QZ_PY" ]; then
  echo "ERROR: QZ_PY is not executable: $QZ_PY" >&2
  exit 1
fi

mkdir -p "$QZ_RECORD_ROOT" "$QZCLI_HOME" "$TARGET_AUDIO_ROOT"

cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail

export ROOT="$ROOT"
export RUN_NAME="$RUN_NAME"
export WORK_ROOT="$WORK_ROOT"
export JOBS_JSONL="$JOBS_JSONL"
export RESULTS_JSONL="$RESULTS_JSONL"
export MANIFEST_JSONL="$MANIFEST_JSONL"
export TARGET_AUDIO_ROOT="$TARGET_AUDIO_ROOT"
export RUN_PREPARE=0
export RUN_SEEDVC=1
export RUN_COLLECT=1
export SEEDVC_GPU_IDS="$SEEDVC_GPU_IDS"
export SEEDVC_SHARD_COUNT="$SEEDVC_SHARD_COUNT"
export MAX_JOBS="$MAX_JOBS"
export DIFFUSION_STEPS="$DIFFUSION_STEPS"
export LENGTH_ADJUST="$LENGTH_ADJUST"
export INFERENCE_CFG_RATE="$INFERENCE_CFG_RATE"
export FP16="$FP16"
export SKIP_EXISTING="$SKIP_EXISTING"
export FAIL_FAST="$FAIL_FAST"
export SHOW_MODEL_OUTPUT="$SHOW_MODEL_OUTPUT"
export MIN_TARGET_AUDIO_BYTES="$MIN_TARGET_AUDIO_BYTES"
export HF_ENDPOINT="\${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="\${OMP_NUM_THREADS:-8}"

cd "\$ROOT"
echo "[qz-text-prosody-seedvc] date=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[qz-text-prosody-seedvc] host=\$(hostname)"
echo "[qz-text-prosody-seedvc] jobs_jsonl=\$JOBS_JSONL"
echo "[qz-text-prosody-seedvc] results_jsonl=\$RESULTS_JSONL"
echo "[qz-text-prosody-seedvc] manifest_jsonl=\$MANIFEST_JSONL"
echo "[qz-text-prosody-seedvc] target_audio_root=\$TARGET_AUDIO_ROOT"
echo "[qz-text-prosody-seedvc] seedvc_gpu_ids=\$SEEDVC_GPU_IDS shard_count=\$SEEDVC_SHARD_COUNT max_jobs=\$MAX_JOBS"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
wc -l "\$JOBS_JSONL" || true

bash scripts/001035_run_text_prosody_mosstts_seedvc_pipeline.sh

echo "[qz-text-prosody-seedvc] final counts"
test -f "\$RESULTS_JSONL" && wc -l "\$RESULTS_JSONL" || true
test -f "\$MANIFEST_JSONL" && wc -l "\$MANIFEST_JSONL" || true
EOF
chmod +x "$RUNNER"

COMMAND="bash $RUNNER"
TMP_OUTPUT="$QZ_RECORD_ROOT/submit_output.txt"

echo "=========================================="
echo "QZ submit: text_prosody Seed-VC pipeline"
echo "  JOB_NAME=$JOB_NAME"
echo "  ROOT=$ROOT"
echo "  RUN_NAME=$RUN_NAME"
echo "  WORK_ROOT=$WORK_ROOT"
echo "  JOBS_JSONL=$JOBS_JSONL"
echo "  RESULTS_JSONL=$RESULTS_JSONL"
echo "  MANIFEST_JSONL=$MANIFEST_JSONL"
echo "  RUNNER=$RUNNER"
echo "  WORKSPACE=$WORKSPACE"
echo "  PROJECT=$PROJECT"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP"
echo "  SPEC=$SPEC"
echo "  PRIORITY=$PRIORITY"
echo "  IMAGE=$IMAGE"
echo "  SEEDVC_GPU_IDS=$SEEDVC_GPU_IDS"
echo "  SEEDVC_SHARD_COUNT=$SEEDVC_SHARD_COUNT"
echo "  MAX_JOBS=$MAX_JOBS"
echo "  COMMAND=$COMMAND"
echo "=========================================="

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
  printf 'job_name\tjob_id\tcompute_group\trunner\twork_root\tjobs_jsonl\tresults_jsonl\tmanifest_jsonl\n'
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$JOB_NAME" "${JOB_ID:-}" "$COMPUTE_GROUP" "$RUNNER" "$WORK_ROOT" "$JOBS_JSONL" "$RESULTS_JSONL" "$MANIFEST_JSONL"
} > "$QZ_RECORD_ROOT/submitted_jobs.tsv"

echo "=========================================="
echo "Submission completed."
echo "  JOB_NAME=$JOB_NAME"
echo "  JOB_ID=${JOB_ID:-not parsed}"
echo "  RECORD=$QZ_RECORD_ROOT/submitted_jobs.tsv"
echo "=========================================="

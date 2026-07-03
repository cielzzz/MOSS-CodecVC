#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
QZCLI_TOOL="${QZCLI_TOOL:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/qzcli_tool}"
QZ_PY="${QZ_PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
QZCLI_HOME="${QZCLI_HOME:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/.codex/qzcli_home}"
if [ ! -x "$QZ_PY" ]; then
  QZ_PY=python
fi

WORKSPACE="${WORKSPACE:-ws-8207e9e2-e733-4eec-a475-cfa1c36480ba}"
PROJECT="${PROJECT:-project-c67c548f-f02c-453b-ba5b-8745db6886e7}"
COMPUTE_GROUP="${COMPUTE_GROUP:-lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122}"  # MTTS-3-2-0715
FRAMEWORK="${FRAMEWORK:-pytorch}"
INSTANCES="${INSTANCES:-1}"
SHM_GI="${SHM_GI:-1200}"
PRIORITY="${PRIORITY:-3}"
IMAGE="${IMAGE:-docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420}"
IMAGE_TYPE="${IMAGE_TYPE:-SOURCE_PRIVATE}"
SPEC="${SPEC:-67b10bc6-78b0-41a3-aaf4-358eeeb99009}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codec-vc-data-process}"
BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%d-%H%M%S)}"
JOB_NAME="${JOB_NAME:-$JOB_NAME_PREFIX-$BATCH_ID}"
RUNNER="${RUNNER:-$ROOT/scripts/001026_run_large_mixed_mode_manifest_full.sh}"
QZ_RECORD_ROOT="${QZ_RECORD_ROOT:-$ROOT/trainset/qz_jobs/$BATCH_ID}"
DRY_RUN=0

for arg in "$@"; do
  if [ "$arg" = "--dry-run" ]; then
    DRY_RUN=1
  fi
done

mkdir -p "$QZ_RECORD_ROOT"

COMMAND="bash -lc 'cd \"$ROOT\" && bash \"$RUNNER\"'"

echo "=========================================="
echo "QZ submit: large mixed-mode VC manifest"
echo "  JOB_NAME=$JOB_NAME"
echo "  ROOT=$ROOT"
echo "  QZCLI_TOOL=$QZCLI_TOOL"
echo "  QZ_PY=$QZ_PY"
echo "  QZCLI_HOME=$QZCLI_HOME"
echo "  WORKSPACE=$WORKSPACE"
echo "  PROJECT=$PROJECT"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP"
echo "  FRAMEWORK=$FRAMEWORK"
echo "  INSTANCES=$INSTANCES"
echo "  SHM_GI=$SHM_GI"
echo "  PRIORITY=$PRIORITY"
echo "  IMAGE=$IMAGE"
echo "  IMAGE_TYPE=$IMAGE_TYPE"
echo "  SPEC=${SPEC:-auto-select}"
echo "  QZ_RECORD_ROOT=$QZ_RECORD_ROOT"
echo "=========================================="

tmp_output=$(mktemp)
cleanup() {
  rm -f "$tmp_output"
}
trap cleanup EXIT INT TERM

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
qz_args+=("$@")

set +e
env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
  HOME="$QZCLI_HOME" \
  PYTHONPATH="$QZCLI_TOOL" \
  "$QZ_PY" "${qz_args[@]}" >"$tmp_output" 2>&1
status=$?
set -e

cat "$tmp_output"
cp "$tmp_output" "$QZ_RECORD_ROOT/submit_output.txt"

if [ "$status" -ne 0 ]; then
  echo "Submission failed. Output saved to $QZ_RECORD_ROOT/submit_output.txt" >&2
  exit "$status"
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "Dry run completed. No job was submitted."
  exit 0
fi

job_id=$(grep -Eo 'job-[0-9a-fA-F-]{36}' "$tmp_output" | tail -n 1 || true)
if [ -z "$job_id" ]; then
  job_uuid=$(grep -E '任务ID|job_id|Job ID' "$tmp_output" | grep -Eo '[0-9a-fA-F-]{36}' | tail -n 1 || true)
  if [ -n "$job_uuid" ]; then
    job_id="job-$job_uuid"
  fi
fi

{
  printf 'job_name\tjob_id\tcompute_group\trecord_root\n'
  printf '%s\t%s\t%s\t%s\n' "$JOB_NAME" "${job_id:-}" "$COMPUTE_GROUP" "$QZ_RECORD_ROOT"
} > "$QZ_RECORD_ROOT/submitted_jobs.tsv"

echo "=========================================="
echo "Submission completed."
echo "  JOB_NAME=$JOB_NAME"
echo "  JOB_ID=${job_id:-not parsed}"
echo "  RECORD=$QZ_RECORD_ROOT/submitted_jobs.tsv"
if [ -n "${job_id:-}" ]; then
  echo "  status: HOME=$QZCLI_HOME PYTHONPATH=$QZCLI_TOOL $QZ_PY -m qzcli.cli status $job_id"
  echo "  logs:   HOME=$QZCLI_HOME PYTHONPATH=$QZCLI_TOOL $QZ_PY -m qzcli.cli logs $job_id"
fi
echo "=========================================="

#!/usr/bin/env bash
# Submit v2 real-target no-text ver2.9 data preparation as one 8-GPU H200 QZ job.
#
# Default is DRY_RUN=1. Set DRY_RUN=0 and ALLOW_V2_REAL_NO_TEXT_PREP_SUBMIT=1
# to submit intentionally.

set -euo pipefail

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
PAIR_CONSTRUCTION_ROOT="${PAIR_CONSTRUCTION_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction}"
QZCLI="${QZCLI:-$PAIR_CONSTRUCTION_ROOT/scripts/qzcli_with_deps.sh}"

WORKSPACE="${WORKSPACE:-CI-情境智能}"
PROJECT="${PROJECT:-CI-情境智能}"
COMPUTE_GROUP="${COMPUTE_GROUP:-lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122}"
FRAMEWORK="${FRAMEWORK:-pytorch}"
INSTANCES="${INSTANCES:-1}"
SHM_GI="${SHM_GI:-1200}"
PRIORITY="${PRIORITY:-10}"
SPEC="${SPEC:-67b10bc6-78b0-41a3-aaf4-358eeeb99009}"
IMAGE="${IMAGE:-docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420}"
IMAGE_TYPE="${IMAGE_TYPE:-SOURCE_PRIVATE}"

PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download}"

SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-/inspire/qb-ilm/project/embodied-multimodality/public/xyzhang/data/VC_train/v2_real_target_no_text_300k_zh_en_balanced_20260707_seedvc_triples}"
TRAIN_INPUT_JSONL="${TRAIN_INPUT_JSONL:-$SOURCE_DATA_DIR/no_text.train.refdecorr.train_minus_valid.manifest.jsonl}"
VALID_ROOT="${VALID_ROOT:-$SOURCE_DATA_DIR/valid_ref_channel_heldout_2k_20260708}"
VALID_SAME_INPUT_JSONL="${VALID_SAME_INPUT_JSONL:-$VALID_ROOT/same_episode_near_original_valid.manifest.jsonl}"
VALID_CROSS_INPUT_JSONL="${VALID_CROSS_INPUT_JSONL:-$VALID_ROOT/heldout_refdecorr_cross_channel_valid.manifest.jsonl}"

WORK_ROOT="${WORK_ROOT:-$ROOT/trainset/v2_real_target_no_text_refdecorr_20260708}"
PREPARED_DIR="${PREPARED_DIR:-$ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_v2_data_20260708}"
TEXT_PREPARED_DIR="${TEXT_PREPARED_DIR:-$ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_20260707}"
TEXT_REPEAT="${TEXT_REPEAT:-10}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
CODEC_SHARD_COUNT="${CODEC_SHARD_COUNT:-8}"
ECAPA_SHARD_COUNT="${ECAPA_SHARD_COUNT:-8}"
PROSODY_SHARD_COUNT="${PROSODY_SHARD_COUNT:-16}"
WAVLM_BNF_SHARD_COUNT="${WAVLM_BNF_SHARD_COUNT:-8}"
WAVLM_SV_SHARD_COUNT="${WAVLM_SV_SHARD_COUNT:-8}"
WAVLM_LOCAL_FILES_ONLY="${WAVLM_LOCAL_FILES_ONLY:-1}"
WAVLM_SV_LOCAL_FILES_ONLY="${WAVLM_SV_LOCAL_FILES_ONLY:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE="${FORCE:-0}"
OVERWRITE="${OVERWRITE:-0}"
WAIT_HEARTBEAT_SECS="${WAIT_HEARTBEAT_SECS:-60}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1000}"

BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%d-%H%M%S)}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codecvc-ver2-9-v2real-no-text-prep}"
JOB_NAME="${JOB_NAME:-$JOB_NAME_PREFIX-$BATCH_ID}"
QZ_RECORD_ROOT="${QZ_RECORD_ROOT:-$ROOT/trainset/qz_jobs/$BATCH_ID}"
RUNNER="$QZ_RECORD_ROOT/run_v2_real_no_text_ver2_9_data_prep.sh"
DRY_RUN="${DRY_RUN:-1}"

if [ "$DRY_RUN" != "1" ] && [ "${ALLOW_V2_REAL_NO_TEXT_PREP_SUBMIT:-0}" != "1" ]; then
  echo "ERROR: guarded submit; set ALLOW_V2_REAL_NO_TEXT_PREP_SUBMIT=1 with DRY_RUN=0." >&2
  exit 1
fi
if [ ! -d "$ROOT" ]; then
  echo "ERROR: ROOT does not exist: $ROOT" >&2
  exit 1
fi
if [ ! -x "$QZCLI" ]; then
  echo "ERROR: qzcli wrapper is not executable: $QZCLI" >&2
  exit 1
fi
for path in "$TRAIN_INPUT_JSONL" "$VALID_SAME_INPUT_JSONL" "$VALID_CROSS_INPUT_JSONL" "$TEXT_PREPARED_DIR/text.train.jsonl"; do
  if [ ! -s "$path" ]; then
    echo "ERROR: required input is missing or empty: $path" >&2
    exit 1
  fi
done

mkdir -p "$QZ_RECORD_ROOT"
cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail

export ROOT="$ROOT"
export PY="$PY"
export DOWNLOAD_ROOT="$DOWNLOAD_ROOT"
export SOURCE_DATA_DIR="$SOURCE_DATA_DIR"
export TRAIN_INPUT_JSONL="$TRAIN_INPUT_JSONL"
export VALID_ROOT="$VALID_ROOT"
export VALID_SAME_INPUT_JSONL="$VALID_SAME_INPUT_JSONL"
export VALID_CROSS_INPUT_JSONL="$VALID_CROSS_INPUT_JSONL"
export WORK_ROOT="$WORK_ROOT"
export PREPARED_DIR="$PREPARED_DIR"
export TEXT_PREPARED_DIR="$TEXT_PREPARED_DIR"
export TEXT_REPEAT="$TEXT_REPEAT"

export GPU_IDS="$GPU_IDS"
export CODEC_GPU_IDS="$GPU_IDS"
export SPEAKER_GPU_IDS="$GPU_IDS"
export CODEC_SHARD_COUNT="$CODEC_SHARD_COUNT"
export ECAPA_SHARD_COUNT="$ECAPA_SHARD_COUNT"
export PROSODY_SHARD_COUNT="$PROSODY_SHARD_COUNT"
export WAVLM_BNF_SHARD_COUNT="$WAVLM_BNF_SHARD_COUNT"
export WAVLM_SV_SHARD_COUNT="$WAVLM_SV_SHARD_COUNT"
export WAVLM_DEVICES="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7"
export WAVLM_SV_DEVICES="\$WAVLM_DEVICES"
export WAVLM_LOCAL_FILES_ONLY="$WAVLM_LOCAL_FILES_ONLY"
export WAVLM_SV_LOCAL_FILES_ONLY="$WAVLM_SV_LOCAL_FILES_ONLY"
export SKIP_EXISTING="$SKIP_EXISTING"
export FORCE="$FORCE"
export OVERWRITE="$OVERWRITE"
export DRY_RUN=0
export WAIT_HEARTBEAT_SECS="$WAIT_HEARTBEAT_SECS"
export PROGRESS_EVERY="$PROGRESS_EVERY"

export HF_HOME="\$DOWNLOAD_ROOT/huggingface"
export TRANSFORMERS_CACHE="\$DOWNLOAD_ROOT/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="\$DOWNLOAD_ROOT/huggingface/hub"
export HF_DATASETS_CACHE="\$DOWNLOAD_ROOT/huggingface/datasets"
export TORCH_HOME="\$DOWNLOAD_ROOT/torch"
export XDG_CACHE_HOME="\$DOWNLOAD_ROOT/cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export DISABLE_SAFETENSORS_CONVERSION=1
export OMP_NUM_THREADS="\${OMP_NUM_THREADS:-8}"

cd "\$ROOT"
echo "[v2-real-no-text-qz] date=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[v2-real-no-text-qz] host=\$(hostname)"
echo "[v2-real-no-text-qz] train=\$TRAIN_INPUT_JSONL"
echo "[v2-real-no-text-qz] valid_same=\$VALID_SAME_INPUT_JSONL"
echo "[v2-real-no-text-qz] valid_cross=\$VALID_CROSS_INPUT_JSONL"
echo "[v2-real-no-text-qz] prepared_dir=\$PREPARED_DIR"
echo "[v2-real-no-text-qz] text_prepared_dir=\$TEXT_PREPARED_DIR text_repeat=\$TEXT_REPEAT"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

"\$PY" - <<'PY'
from __future__ import annotations
import json
from pathlib import Path

paths = {
    "train": Path("$TRAIN_INPUT_JSONL"),
    "same_valid": Path("$VALID_SAME_INPUT_JSONL"),
    "cross_valid": Path("$VALID_CROSS_INPUT_JSONL"),
}
for name, path in paths.items():
    rows = 0
    first = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            if first is None:
                first = json.loads(line)
    print(f"[v2-real-no-text-qz] {name} rows={rows} first_sample_id={first.get('sample_id') if first else None}", flush=True)
    for key in ("source_audio", "timbre_ref_audio", "target_audio"):
        if not first or not first.get(key):
            raise SystemExit(f"missing {key} in {name}: {path}")
PY

bash scripts/002034_prepare_v2_real_no_text_ver2_9_data.sh
EOF
chmod +x "$RUNNER"

COMMAND="bash $RUNNER"

echo "=========================================="
echo "QZ submit: v2 real-target no-text ver2.9 data preparation"
echo "  JOB_NAME=$JOB_NAME"
echo "  ROOT=$ROOT"
echo "  TRAIN_INPUT_JSONL=$TRAIN_INPUT_JSONL"
echo "  VALID_SAME_INPUT_JSONL=$VALID_SAME_INPUT_JSONL"
echo "  VALID_CROSS_INPUT_JSONL=$VALID_CROSS_INPUT_JSONL"
echo "  WORK_ROOT=$WORK_ROOT"
echo "  PREPARED_DIR=$PREPARED_DIR"
echo "  TEXT_PREPARED_DIR=$TEXT_PREPARED_DIR"
echo "  MIXED_SPEC=$PREPARED_DIR/mixed.train.spec.txt"
echo "  GPU_IDS=$GPU_IDS"
echo "  CODEC_SHARD_COUNT=$CODEC_SHARD_COUNT ECAPA_SHARD_COUNT=$ECAPA_SHARD_COUNT PROSODY_SHARD_COUNT=$PROSODY_SHARD_COUNT"
echo "  WAVLM_BNF_SHARD_COUNT=$WAVLM_BNF_SHARD_COUNT WAVLM_SV_SHARD_COUNT=$WAVLM_SV_SHARD_COUNT"
echo "  QZCLI=$QZCLI"
echo "  WORKSPACE=$WORKSPACE"
echo "  PROJECT=$PROJECT"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP"
echo "  SPEC=$SPEC"
echo "  PRIORITY=$PRIORITY"
echo "  IMAGE=$IMAGE"
echo "  RUNNER=$RUNNER"
echo "  COMMAND=$COMMAND"
echo "=========================================="

if [ "$DRY_RUN" = "1" ]; then
  echo "[dry-run] Runner script generated but no QZ job was submitted."
  echo "[dry-run] Inspect: sed -n '1,260p' $RUNNER"
  exit 0
fi

TMP_OUTPUT="$QZ_RECORD_ROOT/submit_output.txt"
rm -f "$TMP_OUTPUT"

set +e
env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
  "$QZCLI" create-job \
  --name "$JOB_NAME" \
  --workspace "$WORKSPACE" \
  --project "$PROJECT" \
  --compute-group "$COMPUTE_GROUP" \
  --spec "$SPEC" \
  --framework "$FRAMEWORK" \
  --instances "$INSTANCES" \
  --shm "$SHM_GI" \
  --priority "$PRIORITY" \
  --image "$IMAGE" \
  --image-type "$IMAGE_TYPE" \
  --command "$COMMAND" >"$TMP_OUTPUT" 2>&1
STATUS=$?
set -e

cat "$TMP_OUTPUT"

if [ "$STATUS" -ne 0 ] && grep -q 'Cookie 已过期或无效' "$TMP_OUTPUT"; then
  echo "Cookie expired; running qzcli login and retrying once." >&2
  env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy "$QZCLI" login
  set +e
  env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
    "$QZCLI" create-job \
    --name "$JOB_NAME" \
    --workspace "$WORKSPACE" \
    --project "$PROJECT" \
    --compute-group "$COMPUTE_GROUP" \
    --spec "$SPEC" \
    --framework "$FRAMEWORK" \
    --instances "$INSTANCES" \
    --shm "$SHM_GI" \
    --priority "$PRIORITY" \
    --image "$IMAGE" \
    --image-type "$IMAGE_TYPE" \
    --command "$COMMAND" >"$TMP_OUTPUT" 2>&1
  STATUS=$?
  set -e
  cat "$TMP_OUTPUT"
fi

if [ "$STATUS" -ne 0 ]; then
  echo "Submission failed. Output saved to $TMP_OUTPUT" >&2
  exit "$STATUS"
fi

JOB_ID=$(grep -Eo 'job-[0-9a-fA-F-]{36}' "$TMP_OUTPUT" | tail -n 1 || true)
if [ -z "$JOB_ID" ]; then
  JOB_UUID=$(grep -E '任务ID|job_id|Job ID' "$TMP_OUTPUT" | grep -Eo '[0-9a-fA-F-]{36}' | tail -n 1 || true)
  if [ -n "$JOB_UUID" ]; then
    JOB_ID="job-$JOB_UUID"
  fi
fi

{
  printf 'job_name\tjob_id\tcompute_group\trunner\tprepared_dir\n'
  printf '%s\t%s\t%s\t%s\t%s\n' "$JOB_NAME" "${JOB_ID:-}" "$COMPUTE_GROUP" "$RUNNER" "$PREPARED_DIR"
} > "$QZ_RECORD_ROOT/submitted_jobs.tsv"

echo "=========================================="
echo "Submission completed."
echo "  JOB_NAME=$JOB_NAME"
echo "  JOB_ID=${JOB_ID:-not parsed}"
echo "  RECORD=$QZ_RECORD_ROOT/submitted_jobs.tsv"
echo "  PREPARED_DIR=$PREPARED_DIR"
echo "=========================================="

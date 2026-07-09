#!/bin/sh
# Submit ver2.9 WavLM-SV speaker-vector precompute as one 8-GPU QZ job.
#
# Default is DRY_RUN=1. Set DRY_RUN=0 and ALLOW_VER2_9_SPEAKER_VEC_SUBMIT=1
# to submit intentionally.

set -eu

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
INPUT_PREPARED_DIR="${INPUT_PREPARED_DIR:-$ROOT/trainset/ver2_8_prepared_speaker_split_20260705}"
OUTPUT_PREPARED_DIR="${OUTPUT_PREPARED_DIR:-$ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_20260707}"
SPEAKER_VEC_DIR="${SPEAKER_VEC_DIR:-$OUTPUT_PREPARED_DIR/speaker_vecs}"
SPEAKER_ENCODER_PATH="${SPEAKER_ENCODER_PATH:-microsoft/wavlm-base-plus-sv}"
SPEAKER_EMBEDDING_DIM="${SPEAKER_EMBEDDING_DIM:-512}"
PRELOAD_MODEL="${PRELOAD_MODEL:-0}"
PRELOAD_LOCAL_FILES_ONLY="${PRELOAD_LOCAL_FILES_ONLY:-1}"
NUM_SHARDS="${NUM_SHARDS:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TEXT_REPEAT="${TEXT_REPEAT:-10}"
OVERWRITE="${OVERWRITE:-0}"
SPLITS="${SPLITS:-no_text.train.jsonl text.train.jsonl no_text.valid.jsonl text.valid.jsonl no_text.seen_valid.jsonl text.seen_valid.jsonl no_text.unseen_valid.jsonl text.unseen_valid.jsonl}"

BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%d-%H%M%S)}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codecvc-ver2-9-speaker-vec-precompute}"
JOB_NAME="${JOB_NAME:-$JOB_NAME_PREFIX-$BATCH_ID}"
QZ_RECORD_ROOT="${QZ_RECORD_ROOT:-$ROOT/trainset/qz_jobs/$BATCH_ID}"
RUNNER="$QZ_RECORD_ROOT/run_speaker_vec_precompute.sh"
DRY_RUN="${DRY_RUN:-1}"

if [ "$DRY_RUN" != "1" ] && [ "${ALLOW_VER2_9_SPEAKER_VEC_SUBMIT:-0}" != "1" ]; then
  echo "ERROR: guarded submit; set ALLOW_VER2_9_SPEAKER_VEC_SUBMIT=1 with DRY_RUN=0." >&2
  exit 1
fi
if [ ! -x "$QZCLI" ]; then
  echo "ERROR: qzcli wrapper is not executable: $QZCLI" >&2
  exit 1
fi
if [ ! -f "$INPUT_PREPARED_DIR/no_text.train.jsonl" ] || [ ! -f "$INPUT_PREPARED_DIR/text.train.jsonl" ]; then
  echo "ERROR: missing input prepared manifests under $INPUT_PREPARED_DIR" >&2
  exit 1
fi

mkdir -p "$QZ_RECORD_ROOT"
cat > "$RUNNER" <<EOF
#!/bin/sh
set -eu

export ROOT="$ROOT"
export PY="$PY"
export DOWNLOAD_ROOT="$DOWNLOAD_ROOT"
export HF_HOME="\$DOWNLOAD_ROOT/huggingface"
export TRANSFORMERS_CACHE="\$DOWNLOAD_ROOT/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="\$DOWNLOAD_ROOT/huggingface/hub"
export HF_DATASETS_CACHE="\$DOWNLOAD_ROOT/huggingface/datasets"
export TORCH_HOME="\$DOWNLOAD_ROOT/torch"
export XDG_CACHE_HOME="\$DOWNLOAD_ROOT/cache"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "\$ROOT"
mkdir -p "$OUTPUT_PREPARED_DIR" "$SPEAKER_VEC_DIR" "$OUTPUT_PREPARED_DIR/logs"

echo "[speaker-vec-qz] date=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[speaker-vec-qz] host=\$(hostname)"
echo "[speaker-vec-qz] input=$INPUT_PREPARED_DIR"
echo "[speaker-vec-qz] output=$OUTPUT_PREPARED_DIR"
echo "[speaker-vec-qz] shards=$NUM_SHARDS batch_size=$BATCH_SIZE speaker_encoder=$SPEAKER_ENCODER_PATH dim=$SPEAKER_EMBEDDING_DIM preload_model=$PRELOAD_MODEL preload_local_files_only=$PRELOAD_LOCAL_FILES_ONLY"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

if [ "$PRELOAD_MODEL" = "1" ]; then
  echo "[speaker-vec-qz] preloading WavLM-SV into HF cache"
  "\$PY" - <<'PY'
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
model_name = "$SPEAKER_ENCODER_PATH"
local_files_only = bool(int("$PRELOAD_LOCAL_FILES_ONLY"))
AutoFeatureExtractor.from_pretrained(model_name, local_files_only=local_files_only)
AutoModelForAudioXVector.from_pretrained(model_name, local_files_only=local_files_only)
print("[speaker-vec-qz] preload ok", model_name, "local_files_only=", local_files_only, flush=True)
PY
else
  echo "[speaker-vec-qz] skipping model preload; shards will load local cache directly"
fi

pids=""
idx=0
while [ "\$idx" -lt "$NUM_SHARDS" ]; do
  log_path="$OUTPUT_PREPARED_DIR/logs/precompute_shard_\$(printf '%02d' "\$idx").log"
  echo "[speaker-vec-qz] launching shard=\$idx log=\$log_path"
  (
    export CUDA_VISIBLE_DEVICES="\$idx"
    INPUT_PREPARED_DIR="$INPUT_PREPARED_DIR" \\
    OUTPUT_PREPARED_DIR="$OUTPUT_PREPARED_DIR" \\
    SPEAKER_VEC_DIR="$SPEAKER_VEC_DIR" \\
    SPEAKER_ENCODER_PATH="$SPEAKER_ENCODER_PATH" \\
    SPEAKER_EMBEDDING_DIM="$SPEAKER_EMBEDDING_DIM" \\
    DEVICE="cuda" \\
    LOCAL_FILES_ONLY="1" \\
    OVERWRITE="$OVERWRITE" \\
    BATCH_SIZE="$BATCH_SIZE" \\
    NUM_SHARDS="$NUM_SHARDS" \\
    SHARD_INDEX="\$idx" \\
    TEXT_REPEAT="$TEXT_REPEAT" \\
    SPLITS="$SPLITS" \\
      sh "\$ROOT/scripts/002023_prepare_ver2_9_speaker_vecs.sh"
  ) > "\$log_path" 2>&1 &
  pids="\$pids \$!"
  idx=\$((idx + 1))
done

status=0
for pid in \$pids; do
  if ! wait "\$pid"; then
    status=1
  fi
done
if [ "\$status" != "0" ]; then
  echo "[speaker-vec-qz] at least one shard failed; see $OUTPUT_PREPARED_DIR/logs" >&2
  exit 1
fi

echo "[speaker-vec-qz] merging shards"
INPUT_PREPARED_DIR="$INPUT_PREPARED_DIR" \\
OUTPUT_PREPARED_DIR="$OUTPUT_PREPARED_DIR" \\
SPEAKER_VEC_DIR="$SPEAKER_VEC_DIR" \\
NUM_SHARDS="$NUM_SHARDS" \\
SHARD_INDEX="merge" \\
MERGE_SHARDS="1" \\
TEXT_REPEAT="$TEXT_REPEAT" \\
SPLITS="$SPLITS" \\
  sh "\$ROOT/scripts/002023_prepare_ver2_9_speaker_vecs.sh"

echo "[speaker-vec-qz] verifying merged manifests"
"\$PY" - <<'PY'
import json
from pathlib import Path
root = Path("$OUTPUT_PREPARED_DIR")
required = ["no_text.train.jsonl", "text.train.jsonl", "no_text.valid.jsonl", "text.valid.jsonl"]
for name in required:
    path = root / name
    if not path.exists():
        raise SystemExit(f"missing merged manifest: {path}")
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            count += 1
            if count <= 3:
                row = json.loads(line)
                if not row.get("speaker_vec_path"):
                    raise SystemExit(f"missing speaker_vec_path in {path}:{count}")
    print(f"[speaker-vec-qz] {name} rows={count}", flush=True)
print("[speaker-vec-qz] verify ok", flush=True)
PY

echo "[speaker-vec-qz] done output=$OUTPUT_PREPARED_DIR"
EOF
chmod +x "$RUNNER"

COMMAND="sh $RUNNER"

echo "=========================================="
echo "QZ submit: ver2.9 speaker vec precompute"
echo "  JOB_NAME=$JOB_NAME"
echo "  ROOT=$ROOT"
echo "  INPUT_PREPARED_DIR=$INPUT_PREPARED_DIR"
echo "  OUTPUT_PREPARED_DIR=$OUTPUT_PREPARED_DIR"
echo "  SPEAKER_VEC_DIR=$SPEAKER_VEC_DIR"
echo "  SPEAKER_ENCODER_PATH=$SPEAKER_ENCODER_PATH"
echo "  SPEAKER_EMBEDDING_DIM=$SPEAKER_EMBEDDING_DIM"
echo "  PRELOAD_MODEL=$PRELOAD_MODEL"
echo "  PRELOAD_LOCAL_FILES_ONLY=$PRELOAD_LOCAL_FILES_ONLY"
echo "  NUM_SHARDS=$NUM_SHARDS"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  SPLITS=$SPLITS"
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
  echo "[dry-run] Inspect: sed -n '1,240p' $RUNNER"
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
  printf 'job_name\tjob_id\tcompute_group\trunner\toutput_prepared_dir\n'
  printf '%s\t%s\t%s\t%s\t%s\n' "$JOB_NAME" "${JOB_ID:-}" "$COMPUTE_GROUP" "$RUNNER" "$OUTPUT_PREPARED_DIR"
} > "$QZ_RECORD_ROOT/submitted_jobs.tsv"

echo "=========================================="
echo "Submission completed."
echo "  JOB_NAME=$JOB_NAME"
echo "  JOB_ID=${JOB_ID:-not parsed}"
echo "  RECORD=$QZ_RECORD_ROOT/submitted_jobs.tsv"
echo "  OUTPUT_PREPARED_DIR=$OUTPUT_PREPARED_DIR"
echo "=========================================="

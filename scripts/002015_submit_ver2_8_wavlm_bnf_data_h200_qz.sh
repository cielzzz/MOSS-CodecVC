#!/bin/sh
# Submit Ver2.8 WavLM-BNF feature extraction + train/valid preparation to QZ.

set -eu

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
PAIR_CONSTRUCTION_ROOT="${PAIR_CONSTRUCTION_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction}"
QZCLI="${QZCLI:-$PAIR_CONSTRUCTION_ROOT/scripts/qzcli_with_deps.sh}"

WORKSPACE="${WORKSPACE:-CI-情境智能}"
PROJECT="${PROJECT:-CI-情境智能}"
COMPUTE_GROUP="${COMPUTE_GROUP:-lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122}"  # MTTS-3-2-0715
FRAMEWORK="${FRAMEWORK:-pytorch}"
INSTANCES="${INSTANCES:-1}"
SHM_GI="${SHM_GI:-1200}"
PRIORITY="${PRIORITY:-10}"
SPEC="${SPEC:-67b10bc6-78b0-41a3-aaf4-358eeeb99009}"
IMAGE="${IMAGE:-docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420}"
IMAGE_TYPE="${IMAGE_TYPE:-SOURCE_PRIVATE}"

PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download}"
BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%d-%H%M%S)}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codecvc-ver2-8-wavlmbnf-data}"
JOB_NAME="${JOB_NAME:-$JOB_NAME_PREFIX-$BATCH_ID}"
QZ_RECORD_ROOT="${QZ_RECORD_ROOT:-$ROOT/trainset/qz_jobs/ver2_8_wavlmbnf_data_$BATCH_ID}"
RUNNER="$QZ_RECORD_ROOT/run_ver2_8_wavlm_bnf_data_entrypoint.sh"

NO_TEXT_DATASET_NAME="${NO_TEXT_DATASET_NAME:-zh45w_en22w_no_text}"
TEXT_DATASET_NAME="${TEXT_DATASET_NAME:-zh3w_en3w_text_prosody_independent_timbre}"
NO_TEXT_HUBERT_CLEAN="${NO_TEXT_HUBERT_CLEAN:-$ROOT/trainset/$NO_TEXT_DATASET_NAME/sft/moss_codecvc_sft.$NO_TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_asr_filter.with_hubert.with_spm_content_tokens.ctc_clean.jsonl}"
TEXT_HUBERT_CLEAN="${TEXT_HUBERT_CLEAN:-$ROOT/trainset/$TEXT_DATASET_NAME/sft/moss_codecvc_sft.$TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_target_asr.with_content_tokens.with_target_hubert.with_spm_content_tokens.ctc_clean.jsonl}"
NO_TEXT_WAVLM_CLEAN="${NO_TEXT_WAVLM_CLEAN:-$ROOT/trainset/$NO_TEXT_DATASET_NAME/sft/moss_codecvc_sft.$NO_TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_asr_filter.with_wavlm_bnf.with_spm_content_tokens.ctc_clean.jsonl}"
TEXT_WAVLM_CLEAN="${TEXT_WAVLM_CLEAN:-$ROOT/trainset/$TEXT_DATASET_NAME/sft/moss_codecvc_sft.$TEXT_DATASET_NAME.with_light_ecapa_spk.with_prosody.with_target_asr.with_content_tokens.with_target_wavlm_bnf.with_spm_content_tokens.ctc_clean.jsonl}"
PREPARED_DIR="${PREPARED_DIR:-$ROOT/trainset/ver2_8_prepared}"

WAVLM_NUM_SHARDS="${WAVLM_NUM_SHARDS:-8}"
WAVLM_DEVICES="${WAVLM_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7}"
NO_TEXT_WAVLM_NUM_SHARDS="${NO_TEXT_WAVLM_NUM_SHARDS:-$WAVLM_NUM_SHARDS}"
TEXT_WAVLM_NUM_SHARDS="${TEXT_WAVLM_NUM_SHARDS:-$WAVLM_NUM_SHARDS}"
NO_TEXT_WAVLM_DEVICES="${NO_TEXT_WAVLM_DEVICES:-$WAVLM_DEVICES}"
TEXT_WAVLM_DEVICES="${TEXT_WAVLM_DEVICES:-$WAVLM_DEVICES}"
WAVLM_LAYER="${WAVLM_LAYER:-9}"
WAVLM_DTYPE="${WAVLM_DTYPE:-auto}"
WAVLM_SAVE_DTYPE="${WAVLM_SAVE_DTYPE:-float16}"
WAVLM_USE_SAFETENSORS="${WAVLM_USE_SAFETENSORS:-false}"
MAX_ROWS="${MAX_ROWS:-0}"
# repeat is total sampled copies, not extra copies. 5 = original + 4 extra.
TEXT_REPEAT="${TEXT_REPEAT:-5}"
OVERWRITE="${OVERWRITE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
REUSE_EXISTING_FEATURES="${REUSE_EXISTING_FEATURES:-1}"
CHECK_FEATURE_FILES="${CHECK_FEATURE_FILES:-1}"

DRY_RUN="${DRY_RUN:-0}"
SKIP_LOCAL_DATA_CHECK="${SKIP_LOCAL_DATA_CHECK:-0}"

usage() {
  cat <<EOF
Usage:
  sh scripts/002015_submit_ver2_8_wavlm_bnf_data_h200_qz.sh [--dry-run] [--skip-local-data-check]

Important env overrides:
  PRIORITY=10
  COMPUTE_GROUP=$COMPUTE_GROUP
  WAVLM_NUM_SHARDS=$WAVLM_NUM_SHARDS
  WAVLM_DEVICES=$WAVLM_DEVICES
  OVERWRITE=$OVERWRITE
  REUSE_EXISTING_FEATURES=$REUSE_EXISTING_FEATURES
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --skip-local-data-check)
      SKIP_LOCAL_DATA_CHECK=1
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

if [ ! -x "$QZCLI" ]; then
  echo "ERROR: qzcli wrapper is not executable: $QZCLI" >&2
  exit 1
fi
if [ ! -x "$PY" ]; then
  echo "ERROR: python is not executable: $PY" >&2
  exit 1
fi
if [ "$DRY_RUN" -ne 1 ] && [ "$SKIP_LOCAL_DATA_CHECK" -ne 1 ]; then
  test -f "$NO_TEXT_HUBERT_CLEAN"
  test -f "$TEXT_HUBERT_CLEAN"
  test -f "$ROOT/scripts/002014_prepare_ver2_8_wavlm_bnf_data.sh"
fi

mkdir -p "$QZ_RECORD_ROOT"

cat > "$RUNNER" <<EOF
#!/bin/sh
set -eu

export ROOT="$ROOT"
export PY="$PY"
export DOWNLOAD_ROOT="$DOWNLOAD_ROOT"
export HF_HOME="\$DOWNLOAD_ROOT/huggingface"
export TRANSFORMERS_CACHE="\$DOWNLOAD_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="\$DOWNLOAD_ROOT/huggingface/hub"
export HF_DATASETS_CACHE="\$DOWNLOAD_ROOT/huggingface/datasets"
export TORCH_HOME="\$DOWNLOAD_ROOT/torch"
export XDG_CACHE_HOME="\$DOWNLOAD_ROOT/cache"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

export NO_TEXT_DATASET_NAME="$NO_TEXT_DATASET_NAME"
export TEXT_DATASET_NAME="$TEXT_DATASET_NAME"
export NO_TEXT_HUBERT_CLEAN="$NO_TEXT_HUBERT_CLEAN"
export TEXT_HUBERT_CLEAN="$TEXT_HUBERT_CLEAN"
export NO_TEXT_WAVLM_CLEAN="$NO_TEXT_WAVLM_CLEAN"
export TEXT_WAVLM_CLEAN="$TEXT_WAVLM_CLEAN"
export PREPARED_DIR="$PREPARED_DIR"
export NO_TEXT_WAVLM_NUM_SHARDS="$NO_TEXT_WAVLM_NUM_SHARDS"
export TEXT_WAVLM_NUM_SHARDS="$TEXT_WAVLM_NUM_SHARDS"
export NO_TEXT_WAVLM_DEVICES="$NO_TEXT_WAVLM_DEVICES"
export TEXT_WAVLM_DEVICES="$TEXT_WAVLM_DEVICES"
export WAVLM_LAYER="$WAVLM_LAYER"
export WAVLM_DTYPE="$WAVLM_DTYPE"
export WAVLM_SAVE_DTYPE="$WAVLM_SAVE_DTYPE"
export WAVLM_USE_SAFETENSORS="$WAVLM_USE_SAFETENSORS"
export MAX_ROWS="$MAX_ROWS"
export TEXT_REPEAT="$TEXT_REPEAT"
export OVERWRITE="$OVERWRITE"
export SKIP_EXISTING="$SKIP_EXISTING"
export REUSE_EXISTING_FEATURES="$REUSE_EXISTING_FEATURES"
export CHECK_FEATURE_FILES="$CHECK_FEATURE_FILES"
export RUN_NO_TEXT=1
export RUN_TEXT=1
export RUN_PREPARE=1

cd "\$ROOT"
echo "[qz-ver2.8-wavlm] date=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[qz-ver2.8-wavlm] host=\$(hostname)"
echo "[qz-ver2.8-wavlm] no_text_input=\$NO_TEXT_HUBERT_CLEAN"
echo "[qz-ver2.8-wavlm] text_input=\$TEXT_HUBERT_CLEAN"
echo "[qz-ver2.8-wavlm] no_text_output=\$NO_TEXT_WAVLM_CLEAN"
echo "[qz-ver2.8-wavlm] text_output=\$TEXT_WAVLM_CLEAN"
echo "[qz-ver2.8-wavlm] prepared_dir=\$PREPARED_DIR"
echo "[qz-ver2.8-wavlm] shards no_text=\$NO_TEXT_WAVLM_NUM_SHARDS text=\$TEXT_WAVLM_NUM_SHARDS"
echo "[qz-ver2.8-wavlm] devices no_text=\$NO_TEXT_WAVLM_DEVICES text=\$TEXT_WAVLM_DEVICES"
echo "[qz-ver2.8-wavlm] overwrite=\$OVERWRITE reuse_features=\$REUSE_EXISTING_FEATURES"
nvidia-smi

bash scripts/002014_prepare_ver2_8_wavlm_bnf_data.sh
EOF
chmod +x "$RUNNER"

COMMAND="sh $RUNNER"

echo "=========================================="
echo "QZ submit: Ver2.8 WavLM-BNF data preparation"
echo "  JOB_NAME=$JOB_NAME"
echo "  ROOT=$ROOT"
echo "  NO_TEXT_HUBERT_CLEAN=$NO_TEXT_HUBERT_CLEAN"
echo "  TEXT_HUBERT_CLEAN=$TEXT_HUBERT_CLEAN"
echo "  NO_TEXT_WAVLM_CLEAN=$NO_TEXT_WAVLM_CLEAN"
echo "  TEXT_WAVLM_CLEAN=$TEXT_WAVLM_CLEAN"
echo "  PREPARED_DIR=$PREPARED_DIR"
echo "  WAVLM_NUM_SHARDS=$WAVLM_NUM_SHARDS"
echo "  WAVLM_DEVICES=$WAVLM_DEVICES"
echo "  OVERWRITE=$OVERWRITE REUSE_EXISTING_FEATURES=$REUSE_EXISTING_FEATURES"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP"
echo "  SPEC=$SPEC"
echo "  PRIORITY=$PRIORITY"
echo "  IMAGE=$IMAGE"
echo "  QZ_RECORD_ROOT=$QZ_RECORD_ROOT"
echo "  RUNNER=$RUNNER"
echo "  COMMAND=$COMMAND"
echo "=========================================="

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry-run] Runner generated but no QZ job was submitted."
  echo "[dry-run] Inspect: sed -n '1,220p' $RUNNER"
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

if [ "$STATUS" -ne 0 ]; then
  if grep -q 'Cookie 已过期或无效' "$TMP_OUTPUT"; then
    echo "Cookie expired; running qzcli login and retrying once." >&2
    env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
      "$QZCLI" login
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

SUBMITTED_TSV="$QZ_RECORD_ROOT/submitted_jobs.tsv"
{
  printf 'job_name\tjob_id\tpriority\tcompute_group\trunner\tno_text_wavlm\ttext_wavlm\tprepared_dir\n'
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$JOB_NAME" "${JOB_ID:-UNKNOWN}" "$PRIORITY" "$COMPUTE_GROUP" "$RUNNER" \
    "$NO_TEXT_WAVLM_CLEAN" "$TEXT_WAVLM_CLEAN" "$PREPARED_DIR"
} >"$SUBMITTED_TSV"

echo "Submitted job_id=${JOB_ID:-UNKNOWN}"
echo "Records: $QZ_RECORD_ROOT"
echo "Status:  $QZCLI status ${JOB_ID:-<job-id>}"
echo "Logs:    $QZCLI logs ${JOB_ID:-<job-id>}"

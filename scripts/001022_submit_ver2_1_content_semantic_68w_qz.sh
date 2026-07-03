#!/usr/bin/env bash
set -euo pipefail

# Submit a single-node 8*H200 QZ/OpenI job for Ver2.1 content/semantic data preparation.
#
# Usage:
#   bash scripts/001022_submit_ver2_1_content_semantic_68w_qz.sh --dry-run
#   bash scripts/001022_submit_ver2_1_content_semantic_68w_qz.sh

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

DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download}"
PYTHON_MAIN="${PYTHON_MAIN:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
PYTHON_ASR="${PYTHON_ASR:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"

DATASET_NAME="${DATASET_NAME:-zh45w_en22w_no_text}"
TRAINSET_DIR="${TRAINSET_DIR:-$ROOT/trainset/$DATASET_NAME}"
SFT_DIR="${SFT_DIR:-$TRAINSET_DIR/sft}"
INPUT_JSONL="${INPUT_JSONL:-$SFT_DIR/moss_codecvc_sft.$DATASET_NAME.with_light_ecapa_spk.with_prosody.jsonl}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$SFT_DIR/moss_codecvc_sft.$DATASET_NAME.with_light_ecapa_spk.with_prosody}"
ASR_JSONL="${ASR_JSONL:-$OUTPUT_PREFIX.with_asr_filter.jsonl}"
ASR_KEPT_JSONL="${ASR_KEPT_JSONL:-$OUTPUT_PREFIX.with_asr_filter.keep.jsonl}"
CONTENT_JSONL="${CONTENT_JSONL:-$OUTPUT_PREFIX.with_asr_filter.with_content_tokens.jsonl}"
FINAL_JSONL="${FINAL_JSONL:-$OUTPUT_PREFIX.with_asr_filter.with_content_tokens.with_hubert.jsonl}"
CONTENT_VOCAB_JSON="${CONTENT_VOCAB_JSON:-$SFT_DIR/content_ctc_char_vocab.json}"
HUBERT_FEATURE_ROOT="${HUBERT_FEATURE_ROOT:-$TRAINSET_DIR/semantic_features/hubert}"

BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%d-%H%M%S)}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codecvc-ver2-1-content-semantic-68w}"
JOB_NAME="${JOB_NAME:-$JOB_NAME_PREFIX-$BATCH_ID}"
QZ_RECORD_ROOT="${QZ_RECORD_ROOT:-$ROOT/trainset/qz_jobs/$BATCH_ID}"
RUNNER="$QZ_RECORD_ROOT/run_ver2_1_content_semantic_entrypoint.sh"

RUN_P0="${RUN_P0:-1}"
FILTER_CONTENT_KEEP="${FILTER_CONTENT_KEEP:-1}"
RUN_P1="${RUN_P1:-1}"
RUN_P2="${RUN_P2:-1}"
OVERWRITE="${OVERWRITE:-0}"
RESUME_SHARDS="${RESUME_SHARDS:-1}"
MAX_ROWS="${MAX_ROWS:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1000}"

ASR_BACKEND="${ASR_BACKEND:-qwen_asr}"
QWEN_ASR_MODEL="${QWEN_ASR_MODEL:-$DOWNLOAD_ROOT/checkpoint/qwen-asr-1_7b}"
QWEN_ASR_DTYPE="${QWEN_ASR_DTYPE:-bfloat16}"
QWEN_ASR_MAX_BATCH_SIZE="${QWEN_ASR_MAX_BATCH_SIZE:-16}"
QWEN_ASR_MAX_NEW_TOKENS="${QWEN_ASR_MAX_NEW_TOKENS:-256}"
ASR_DEVICE="${ASR_DEVICE:-cuda:0}"
ASR_NUM_SHARDS="${ASR_NUM_SHARDS:-8}"
ASR_DEVICES="${ASR_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7}"
ASR_SHARD_DIR="${ASR_SHARD_DIR:-$ASR_JSONL.shards}"
ASR_MAP_JSONL="${ASR_MAP_JSONL:-}"
FASTER_WHISPER_MODEL="${FASTER_WHISPER_MODEL:-}"
WHISPER_MODEL="${WHISPER_MODEL:-small}"
LANGUAGE="${LANGUAGE:-}"
ZH_CER_THRESHOLD="${ZH_CER_THRESHOLD:-0.20}"
EN_WER_THRESHOLD="${EN_WER_THRESHOLD:-0.25}"
NO_TEXT_ZH_CER_THRESHOLD="${NO_TEXT_ZH_CER_THRESHOLD:-0.25}"
NO_TEXT_EN_WER_THRESHOLD="${NO_TEXT_EN_WER_THRESHOLD:-0.30}"
MAX_REPEAT_SCORE="${MAX_REPEAT_SCORE:-0.30}"
MIN_ASR_CHARS="${MIN_ASR_CHARS:-2}"
MIN_DURATION_RATIO="${MIN_DURATION_RATIO:-0.50}"
MAX_DURATION_RATIO="${MAX_DURATION_RATIO:-1.80}"
CONTENT_KEEP_MISSING_AS="${CONTENT_KEEP_MISSING_AS:-drop}"

HUBERT_CACHE_DIR="${HUBERT_CACHE_DIR:-$DOWNLOAD_ROOT/huggingface}"
DEFAULT_HUBERT_MODEL="$HUBERT_CACHE_DIR/models--facebook--hubert-base-ls960/snapshots/dba3bb02fda4248b6e082697eee756de8fe8aa8a"
if [[ ! -f "$DEFAULT_HUBERT_MODEL/config.json" ]]; then
  DEFAULT_HUBERT_MODEL="facebook/hubert-base-ls960"
fi
HUBERT_MODEL="${HUBERT_MODEL:-$DEFAULT_HUBERT_MODEL}"
HUBERT_DEVICE="${HUBERT_DEVICE:-cuda}"
HUBERT_DTYPE="${HUBERT_DTYPE:-auto}"
HUBERT_SAVE_DTYPE="${HUBERT_SAVE_DTYPE:-float16}"
HUBERT_LAYER="${HUBERT_LAYER:-9}"
HUBERT_DOWNSAMPLE_STRIDE="${HUBERT_DOWNSAMPLE_STRIDE:-1}"
HUBERT_USE_SAFETENSORS="${HUBERT_USE_SAFETENSORS:-false}"
HUBERT_LOCAL_FILES_ONLY="${HUBERT_LOCAL_FILES_ONLY:-1}"
HUBERT_SOURCE="${HUBERT_SOURCE:-both}"
HUBERT_NUM_SHARDS="${HUBERT_NUM_SHARDS:-8}"
HUBERT_DEVICES="${HUBERT_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7}"
HUBERT_SHARD_DIR="${HUBERT_SHARD_DIR:-$FINAL_JSONL.shards}"
P2_DRY_RUN="${P2_DRY_RUN:-0}"

DRY_RUN=0
SKIP_LOCAL_DATA_CHECK="${SKIP_LOCAL_DATA_CHECK:-0}"

usage() {
  cat <<EOF
Usage:
  bash scripts/001022_submit_ver2_1_content_semantic_68w_qz.sh [--dry-run] [--skip-local-data-check]

Common overrides:
  MAX_ROWS=1000 OVERWRITE=1 bash scripts/001022_submit_ver2_1_content_semantic_68w_qz.sh --dry-run
  INPUT_JSONL=/abs/train.jsonl OUTPUT_PREFIX=/abs/out_prefix bash scripts/001022_submit_ver2_1_content_semantic_68w_qz.sh

Default output:
  FINAL_JSONL=$FINAL_JSONL
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

if [ ! -d "$ROOT" ]; then
  echo "ERROR: ROOT does not exist: $ROOT" >&2
  exit 1
fi
if [ ! -x "$QZ_PY" ]; then
  echo "ERROR: QZ_PY is not executable: $QZ_PY" >&2
  exit 1
fi
if [ ! -x "$PYTHON_MAIN" ]; then
  echo "ERROR: PYTHON_MAIN is not executable: $PYTHON_MAIN" >&2
  exit 1
fi
if [ ! -x "$PYTHON_ASR" ]; then
  echo "ERROR: PYTHON_ASR is not executable: $PYTHON_ASR" >&2
  exit 1
fi
if [ "$DRY_RUN" -ne 1 ] && [ "$SKIP_LOCAL_DATA_CHECK" -ne 1 ] && [ ! -f "$INPUT_JSONL" ]; then
  echo "ERROR: INPUT_JSONL does not exist yet: $INPUT_JSONL" >&2
  echo "Use --dry-run to inspect the QZ command before the data is ready." >&2
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

export PROJECT_ROOT="$ROOT"
export PYTHON_MAIN="$PYTHON_MAIN"
export PYTHON_ASR="$PYTHON_ASR"
export DOWNLOAD_ROOT="$DOWNLOAD_ROOT"
export TRAINSET_DIR="$TRAINSET_DIR"
export SFT_DIR="$SFT_DIR"
export INPUT_JSONL="$INPUT_JSONL"
export OUTPUT_PREFIX="$OUTPUT_PREFIX"
export ASR_JSONL="$ASR_JSONL"
export ASR_KEPT_JSONL="$ASR_KEPT_JSONL"
export CONTENT_JSONL="$CONTENT_JSONL"
export FINAL_JSONL="$FINAL_JSONL"
export CONTENT_VOCAB_JSON="$CONTENT_VOCAB_JSON"
export HUBERT_FEATURE_ROOT="$HUBERT_FEATURE_ROOT"

export RUN_P0="$RUN_P0"
export FILTER_CONTENT_KEEP="$FILTER_CONTENT_KEEP"
export RUN_P1="$RUN_P1"
export RUN_P2="$RUN_P2"
export OVERWRITE="$OVERWRITE"
export RESUME_SHARDS="$RESUME_SHARDS"
export MAX_ROWS="$MAX_ROWS"
export PROGRESS_EVERY="$PROGRESS_EVERY"

export ASR_BACKEND="$ASR_BACKEND"
export QWEN_ASR_MODEL="$QWEN_ASR_MODEL"
export QWEN_ASR_DTYPE="$QWEN_ASR_DTYPE"
export QWEN_ASR_MAX_BATCH_SIZE="$QWEN_ASR_MAX_BATCH_SIZE"
export QWEN_ASR_MAX_NEW_TOKENS="$QWEN_ASR_MAX_NEW_TOKENS"
export ASR_DEVICE="$ASR_DEVICE"
export ASR_NUM_SHARDS="$ASR_NUM_SHARDS"
export ASR_DEVICES="$ASR_DEVICES"
export ASR_SHARD_DIR="$ASR_SHARD_DIR"
export ASR_MAP_JSONL="$ASR_MAP_JSONL"
export FASTER_WHISPER_MODEL="$FASTER_WHISPER_MODEL"
export WHISPER_MODEL="$WHISPER_MODEL"
export LANGUAGE="$LANGUAGE"
export ZH_CER_THRESHOLD="$ZH_CER_THRESHOLD"
export EN_WER_THRESHOLD="$EN_WER_THRESHOLD"
export NO_TEXT_ZH_CER_THRESHOLD="$NO_TEXT_ZH_CER_THRESHOLD"
export NO_TEXT_EN_WER_THRESHOLD="$NO_TEXT_EN_WER_THRESHOLD"
export MAX_REPEAT_SCORE="$MAX_REPEAT_SCORE"
export MIN_ASR_CHARS="$MIN_ASR_CHARS"
export MIN_DURATION_RATIO="$MIN_DURATION_RATIO"
export MAX_DURATION_RATIO="$MAX_DURATION_RATIO"
export CONTENT_KEEP_MISSING_AS="$CONTENT_KEEP_MISSING_AS"

export HUBERT_MODEL="$HUBERT_MODEL"
export HUBERT_CACHE_DIR="$HUBERT_CACHE_DIR"
export HUBERT_DEVICE="$HUBERT_DEVICE"
export HUBERT_DTYPE="$HUBERT_DTYPE"
export HUBERT_SAVE_DTYPE="$HUBERT_SAVE_DTYPE"
export HUBERT_LAYER="$HUBERT_LAYER"
export HUBERT_DOWNSAMPLE_STRIDE="$HUBERT_DOWNSAMPLE_STRIDE"
export HUBERT_USE_SAFETENSORS="$HUBERT_USE_SAFETENSORS"
export HUBERT_LOCAL_FILES_ONLY="$HUBERT_LOCAL_FILES_ONLY"
export HUBERT_SOURCE="$HUBERT_SOURCE"
export HUBERT_NUM_SHARDS="$HUBERT_NUM_SHARDS"
export HUBERT_DEVICES="$HUBERT_DEVICES"
export HUBERT_SHARD_DIR="$HUBERT_SHARD_DIR"
export P2_DRY_RUN="$P2_DRY_RUN"

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

cd "\$PROJECT_ROOT"
echo "[qz-ver2.1-data] date=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[qz-ver2.1-data] host=\$(hostname)"
echo "[qz-ver2.1-data] root=\$PROJECT_ROOT"
echo "[qz-ver2.1-data] input=\$INPUT_JSONL"
echo "[qz-ver2.1-data] final=\$FINAL_JSONL"
echo "[qz-ver2.1-data] asr_backend=\$ASR_BACKEND asr_device=\$ASR_DEVICE asr_num_shards=\$ASR_NUM_SHARDS asr_devices=\$ASR_DEVICES"
echo "[qz-ver2.1-data] hubert_model=\$HUBERT_MODEL hubert_device=\$HUBERT_DEVICE hubert_source=\$HUBERT_SOURCE hubert_num_shards=\$HUBERT_NUM_SHARDS hubert_devices=\$HUBERT_DEVICES"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

bash scripts/001018_prepare_ver2_1_content_semantic_68w.sh
EOF
chmod +x "$RUNNER"

COMMAND="bash $RUNNER"

echo "=========================================="
echo "QZ submit: Ver2.1 content/semantic data prep"
echo "  JOB_NAME=$JOB_NAME"
echo "  ROOT=$ROOT"
echo "  INPUT_JSONL=$INPUT_JSONL"
echo "  FINAL_JSONL=$FINAL_JSONL"
echo "  RUNNER=$RUNNER"
echo "  WORKSPACE=$WORKSPACE"
echo "  PROJECT=$PROJECT"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP"
echo "  SPEC=$SPEC"
echo "  PRIORITY=$PRIORITY"
echo "  IMAGE=$IMAGE"
echo "  RUN_P0=$RUN_P0 FILTER_CONTENT_KEEP=$FILTER_CONTENT_KEEP RUN_P1=$RUN_P1 RUN_P2=$RUN_P2"
echo "  MAX_ROWS=$MAX_ROWS OVERWRITE=$OVERWRITE RESUME_SHARDS=$RESUME_SHARDS"
echo "  ASR_BACKEND=$ASR_BACKEND ASR_NUM_SHARDS=$ASR_NUM_SHARDS ASR_DEVICES=$ASR_DEVICES"
echo "  HUBERT_MODEL=$HUBERT_MODEL HUBERT_SOURCE=$HUBERT_SOURCE HUBERT_NUM_SHARDS=$HUBERT_NUM_SHARDS HUBERT_DEVICES=$HUBERT_DEVICES"
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
  if grep -q 'Cookie 已过期或无效' "$TMP_OUTPUT"; then
    echo "Fix authentication, then resubmit:" >&2
    echo "  HOME=$QZCLI_HOME PYTHONPATH=$QZCLI_TOOL $QZ_PY -m qzcli.cli login" >&2
    echo "  bash scripts/001022_submit_ver2_1_content_semantic_68w_qz.sh" >&2
  fi
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
  printf 'job_name\tjob_id\tcompute_group\trunner\tinput_jsonl\tfinal_jsonl\n'
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$JOB_NAME" "${JOB_ID:-}" "$COMPUTE_GROUP" "$RUNNER" "$INPUT_JSONL" "$FINAL_JSONL"
} > "$QZ_RECORD_ROOT/submitted_jobs.tsv"

echo "=========================================="
echo "Submission completed."
echo "  JOB_NAME=$JOB_NAME"
echo "  JOB_ID=${JOB_ID:-not parsed}"
echo "  RECORD=$QZ_RECORD_ROOT/submitted_jobs.tsv"
echo "  FINAL_JSONL=$FINAL_JSONL"
echo "=========================================="

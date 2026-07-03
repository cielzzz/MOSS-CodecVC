#!/usr/bin/env bash
set -euo pipefail

# Submit one 8-GPU QZ job that evaluates the current Ver2.8 comparison
# checkpoints on the fixed 320-case SeedTTS validation set.

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
ROOT=$(CDPATH= cd "${SCRIPT_DIR}/.." && pwd)
PAIR_CONSTRUCTION_ROOT="${PAIR_CONSTRUCTION_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction}"
QZCLI="${QZCLI:-$PAIR_CONSTRUCTION_ROOT/scripts/qzcli_with_deps.sh}"

WORKSPACE="${WORKSPACE:-CI-情境智能}"
PROJECT="${PROJECT:-CI-情境智能}"
COMPUTE_GROUP="${COMPUTE_GROUP:-lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122}"  # MTTS-3-2-0715
SPEC="${SPEC:-67b10bc6-78b0-41a3-aaf4-358eeeb99009}"  # 8x H200
PRIORITY="${PRIORITY:-10}"
FRAMEWORK="${FRAMEWORK:-pytorch}"
INSTANCES="${INSTANCES:-1}"
SHM_GI="${SHM_GI:-1200}"
IMAGE="${IMAGE:-docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420}"
IMAGE_TYPE="${IMAGE_TYPE:-SOURCE_PRIVATE}"

NEW_STEP="${NEW_STEP:-16000}"
OLD_STEP="${OLD_STEP:-15000}"
BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%d-%H%M%S)}"
DECODING_PROFILE="${DECODING_PROFILE:-lowrand}"
JOB_NAME="${JOB_NAME:-codecvc-ver2-8-seedtts320-eval-new${NEW_STEP}-old${OLD_STEP}-${DECODING_PROFILE}-${BATCH_ID}}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/testset/outputs/ver2_8_seedtts320_eval/new${NEW_STEP}_old${OLD_STEP}_${DECODING_PROFILE}_${BATCH_ID}}"
QZ_RECORD_ROOT="${QZ_RECORD_ROOT:-$EVAL_ROOT/qz_submit}"
RUNNER="$QZ_RECORD_ROOT/run_ver2_8_seedtts320_eval.sh"
DRY_RUN="${DRY_RUN:-0}"

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
ASR_PYTHON="${ASR_PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"

NEW_RUN="$ROOT/outputs/lora_runs/ver2_8_sideonly_wavlmbnf_codecres_textrep10_mergednotext_modeawareprosody_lora_r16_a32_gbs64"
OLD_RUN="$ROOT/outputs/lora_runs/ver2_8_sideonly_wavlmbnf_codecres_textrep5_lora_r16_a32_gbs64"
NEW_MODEL_PATH="${NEW_MODEL_PATH:-$NEW_RUN/step-$NEW_STEP}"
OLD_MODEL_PATH="${OLD_MODEL_PATH:-$OLD_RUN/step-$OLD_STEP}"

for model_path in "$NEW_MODEL_PATH" "$OLD_MODEL_PATH"; do
  if [ ! -d "$model_path" ]; then
    echo "ERROR: missing checkpoint: $model_path" >&2
    exit 1
  fi
done
if [ ! -x "$QZCLI" ]; then
  echo "ERROR: qzcli wrapper not executable: $QZCLI" >&2
  exit 1
fi

mkdir -p "$QZ_RECORD_ROOT" "$EVAL_ROOT"

cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail

export ROOT="$ROOT"
export PYTHON="$PYTHON"
export ASR_PYTHON="$ASR_PYTHON"
export DOWNLOAD_ROOT="/inspire/ssd/project/embodied-multimodality/public/xyzhang/download"
export HF_HOME="\$DOWNLOAD_ROOT/huggingface"
export TRANSFORMERS_CACHE="\$DOWNLOAD_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="\$DOWNLOAD_ROOT/huggingface/hub"
export HF_DATASETS_CACHE="\$DOWNLOAD_ROOT/huggingface/datasets"
export TORCH_HOME="\$DOWNLOAD_ROOT/torch"
export XDG_CACHE_HOME="\$DOWNLOAD_ROOT/cache"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd "\$ROOT"
mkdir -p "$EVAL_ROOT"

run_eval() {
  local run_id="\$1"
  local label="\$2"
  local model_path="\$3"
  local output_dir="$EVAL_ROOT/\$run_id"
  RUN_ID="\$run_id" \
  RUN_LABEL="\$label" \
  MODEL_PATH="\$model_path" \
  OUTPUT_DIR="\$output_dir" \
  NUM_SHARDS=8 \
  ASR_NUM_SHARDS=8 \
  GPU_COUNT=8 \
  MODE=all \
  MAX_CASES=0 \
  PER_MODE=0 \
  PER_CELL=0 \
  PERSISTENT_INFER=1 \
  INFER_SHARD_START_DELAY_SEC="${INFER_SHARD_START_DELAY_SEC:-20}" \
  OVERWRITE_INFER=0 \
  RESET_MANIFESTS=1 \
  DECODING_PROFILE="$DECODING_PROFILE" \
  TIMBRE_SIDE_ONLY=1 \
  CONTENT_REFERENCE_MODE=text \
  BUILD_PAGE=1 \
  PAGE_DIR="$EVAL_ROOT/listening_page" \
  bash "\$ROOT/scripts/004039_run_seedtts_validation_eval.sh"
}

run_eval "ver2_8_merged_textrep10_step${NEW_STEP}_${DECODING_PROFILE}" "ver2.8 merged no-text + textrep10 step-${NEW_STEP}" "$NEW_MODEL_PATH"
run_eval "ver2_8_textrep5_step${OLD_STEP}_${DECODING_PROFILE}" "ver2.8 old textrep5 step-${OLD_STEP}" "$OLD_MODEL_PATH"

"\$PYTHON" "\$ROOT/scripts/004043_compare_seedtts_validation_runs.py" \
  --eval-root "$EVAL_ROOT" \
  --output-md "$EVAL_ROOT/COMPARE.md" \
  --output-csv "$EVAL_ROOT/compare_summary.csv"

echo "[ver2.8-seedtts320-qz] done: $EVAL_ROOT"
EOF
chmod +x "$RUNNER"

COMMAND="bash $RUNNER"

echo "QZ submit: Ver2.8 SeedTTS 320 eval"
echo "  JOB_NAME=$JOB_NAME"
echo "  NEW_MODEL_PATH=$NEW_MODEL_PATH"
echo "  OLD_MODEL_PATH=$OLD_MODEL_PATH"
echo "  EVAL_ROOT=$EVAL_ROOT"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP"
echo "  PRIORITY=$PRIORITY"
echo "  RUNNER=$RUNNER"

if [ "$DRY_RUN" = "1" ]; then
  echo "[dry-run] runner generated: $RUNNER"
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
{
  printf 'job_name\tjob_id\tcompute_group\tpriority\trunner\teval_root\tnew_model_path\told_model_path\n'
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$JOB_NAME" "${JOB_ID:-}" "$COMPUTE_GROUP" "$PRIORITY" "$RUNNER" "$EVAL_ROOT" "$NEW_MODEL_PATH" "$OLD_MODEL_PATH"
} > "$QZ_RECORD_ROOT/submitted_jobs.tsv"
echo "submitted_jobs=$QZ_RECORD_ROOT/submitted_jobs.tsv"

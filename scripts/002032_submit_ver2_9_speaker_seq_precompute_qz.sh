#!/bin/sh
# Submit ver2.9 Fix 6 WavLM sequence speaker-feature precompute as one 8-GPU QZ job.
#
# Default is DRY_RUN=1. Set DRY_RUN=0 and ALLOW_VER2_9_SPEAKER_SEQ_SUBMIT=1
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
INPUT_PREPARED_DIR="${INPUT_PREPARED_DIR:-$ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_20260707}"
OUTPUT_PREPARED_DIR="${OUTPUT_PREPARED_DIR:-$ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260708}"
SPEAKER_SEQ_DIR="${SPEAKER_SEQ_DIR:-$OUTPUT_PREPARED_DIR/speaker_seq_features}"
WAVLM_MODEL_PATH="${WAVLM_MODEL_PATH:-microsoft/wavlm-base-plus}"
WAVLM_LAYER="${WAVLM_LAYER:-9}"
DOWNSAMPLE_STRIDE="${DOWNSAMPLE_STRIDE:-2}"
SAVE_DTYPE="${SAVE_DTYPE:-float16}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
NUM_SHARDS="${NUM_SHARDS:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
OVERWRITE="${OVERWRITE:-0}"
TEXT_REPEAT="${TEXT_REPEAT:-10}"
SPLITS="${SPLITS:-no_text.train.jsonl text.train.jsonl no_text.valid.jsonl text.valid.jsonl no_text.seen_valid.jsonl text.seen_valid.jsonl no_text.unseen_valid.jsonl text.unseen_valid.jsonl}"

QUICK_VALIDATION_JSONL="${QUICK_VALIDATION_JSONL:-$ROOT/testset/validation/ver2_8_timbre_quick20_seedtts_no_text_20260704.jsonl}"
QUICK_SEQ_VALIDATION_JSONL="${QUICK_SEQ_VALIDATION_JSONL:-$ROOT/testset/validation/ver2_9_seq_ver2_8_timbre_quick20_seedtts_no_text_20260704.jsonl}"
RUN_DOMAIN_VALIDATION="${RUN_DOMAIN_VALIDATION:-1}"
DOMAIN_VALIDATION_JSONL="${DOMAIN_VALIDATION_JSONL:-$ROOT/testset/validation/ver2_8_t11_domain_prepared_valid_no_text_50_20260704.jsonl}"
DOMAIN_SEQ_VALIDATION_JSONL="${DOMAIN_SEQ_VALIDATION_JSONL:-$ROOT/testset/validation/ver2_9_seq_ver2_8_t11_domain_prepared_valid_no_text_50_20260704.jsonl}"

BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%d-%H%M%S)}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codecvc-ver2-9-speaker-seq-precompute}"
JOB_NAME="${JOB_NAME:-$JOB_NAME_PREFIX-$BATCH_ID}"
QZ_RECORD_ROOT="${QZ_RECORD_ROOT:-$ROOT/trainset/qz_jobs/$BATCH_ID}"
RUNNER="$QZ_RECORD_ROOT/run_speaker_seq_precompute.sh"
DRY_RUN="${DRY_RUN:-1}"

if [ "$DRY_RUN" != "1" ] && [ "${ALLOW_VER2_9_SPEAKER_SEQ_SUBMIT:-0}" != "1" ]; then
  echo "ERROR: guarded submit; set ALLOW_VER2_9_SPEAKER_SEQ_SUBMIT=1 with DRY_RUN=0." >&2
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
export HF_HUB_OFFLINE="$LOCAL_FILES_ONLY"
export HF_DATASETS_OFFLINE="$LOCAL_FILES_ONLY"
export TRANSFORMERS_OFFLINE="$LOCAL_FILES_ONLY"
export OMP_NUM_THREADS="\${OMP_NUM_THREADS:-4}"

cd "\$ROOT"
mkdir -p "$OUTPUT_PREPARED_DIR" "$SPEAKER_SEQ_DIR" "$OUTPUT_PREPARED_DIR/logs" "\$(dirname "$QUICK_SEQ_VALIDATION_JSONL")" "\$(dirname "$DOMAIN_SEQ_VALIDATION_JSONL")"

echo "[speaker-seq-qz] date=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[speaker-seq-qz] host=\$(hostname)"
echo "[speaker-seq-qz] input=$INPUT_PREPARED_DIR"
echo "[speaker-seq-qz] output=$OUTPUT_PREPARED_DIR"
echo "[speaker-seq-qz] speaker_seq_dir=$SPEAKER_SEQ_DIR"
echo "[speaker-seq-qz] model=$WAVLM_MODEL_PATH layer=$WAVLM_LAYER stride=$DOWNSAMPLE_STRIDE dtype=$SAVE_DTYPE local_files_only=$LOCAL_FILES_ONLY"
echo "[speaker-seq-qz] shards=$NUM_SHARDS batch_size=$BATCH_SIZE overwrite=$OVERWRITE"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

"\$PY" - <<'PY'
from transformers import AutoFeatureExtractor, AutoModel
model_name = "$WAVLM_MODEL_PATH"
local_files_only = bool(int("$LOCAL_FILES_ONLY"))
AutoFeatureExtractor.from_pretrained(model_name, local_files_only=local_files_only)
AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
print("[speaker-seq-qz] model load check ok", model_name, "local_files_only=", local_files_only, flush=True)
PY

pids=""
idx=0
while [ "\$idx" -lt "$NUM_SHARDS" ]; do
  log_path="$OUTPUT_PREPARED_DIR/logs/precompute_seq_shard_\$(printf '%02d' "\$idx").log"
  echo "[speaker-seq-qz] launching shard=\$idx log=\$log_path"
  (
    export CUDA_VISIBLE_DEVICES="\$idx"
    for split in $SPLITS; do
      input_jsonl="$INPUT_PREPARED_DIR/\$split"
      if [ ! -f "\$input_jsonl" ]; then
        echo "[speaker-seq-qz] skip missing split: \$input_jsonl" >&2
        continue
      fi
      shard_suffix=".shard\$(printf '%05d' "\$idx")-of\$(printf '%05d' "$NUM_SHARDS").jsonl"
      output_jsonl="$OUTPUT_PREPARED_DIR/\$split\$shard_suffix"
      split_dir="$SPEAKER_SEQ_DIR/\${split%.jsonl}"
      extra_args=""
      if [ "$OVERWRITE" = "1" ]; then
        extra_args="\$extra_args --overwrite"
      fi
      if [ "$LOCAL_FILES_ONLY" = "1" ]; then
        extra_args="\$extra_args --local-files-only"
      fi
      echo "[speaker-seq-qz] shard=\$idx split=\$split output=\$output_jsonl"
      "\$PY" "\$ROOT/scripts/002031_precompute_wavlm_seq_features.py" \\
        --input-jsonl "\$input_jsonl" \\
        --output-jsonl "\$output_jsonl" \\
        --speaker-seq-dir "\$split_dir" \\
        --model-name-or-path "$WAVLM_MODEL_PATH" \\
        --layer "$WAVLM_LAYER" \\
        --downsample-stride "$DOWNSAMPLE_STRIDE" \\
        --dtype "$SAVE_DTYPE" \\
        --batch-size "$BATCH_SIZE" \\
        --device cuda \\
        --shard-index "\$idx" \\
        --num-shards "$NUM_SHARDS" \\
        --summary-json "\$output_jsonl.summary.json" \\
        \$extra_args
    done
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
  echo "[speaker-seq-qz] at least one shard failed; see $OUTPUT_PREPARED_DIR/logs" >&2
  exit 1
fi

echo "[speaker-seq-qz] merging shards"
for split in $SPLITS; do
  input_jsonl="$INPUT_PREPARED_DIR/\$split"
  if [ ! -f "\$input_jsonl" ]; then
    continue
  fi
  final_jsonl="$OUTPUT_PREPARED_DIR/\$split"
  tmp_jsonl="\$final_jsonl.tmp"
  : > "\$tmp_jsonl"
  idx=0
  while [ "\$idx" -lt "$NUM_SHARDS" ]; do
    part_jsonl="$OUTPUT_PREPARED_DIR/\$split.shard\$(printf '%05d' "\$idx")-of\$(printf '%05d' "$NUM_SHARDS").jsonl"
    if [ ! -f "\$part_jsonl" ]; then
      echo "[speaker-seq-qz] missing shard for merge: \$part_jsonl" >&2
      rm -f "\$tmp_jsonl"
      exit 1
    fi
    cat "\$part_jsonl" >> "\$tmp_jsonl"
    idx=\$((idx + 1))
  done
  mv "\$tmp_jsonl" "\$final_jsonl"
  echo "[speaker-seq-qz] merged split=\$split output=\$final_jsonl"
done

if [ -f "$OUTPUT_PREPARED_DIR/no_text.train.jsonl" ] && [ -f "$OUTPUT_PREPARED_DIR/text.train.jsonl" ]; then
  {
    printf '%s::repeat=1\\n' "$OUTPUT_PREPARED_DIR/no_text.train.jsonl"
    printf '%s::repeat=%s\\n' "$OUTPUT_PREPARED_DIR/text.train.jsonl" "$TEXT_REPEAT"
  } > "$OUTPUT_PREPARED_DIR/mixed.train.spec.txt"
fi
if [ -f "$OUTPUT_PREPARED_DIR/no_text.valid.jsonl" ] && [ -f "$OUTPUT_PREPARED_DIR/text.valid.jsonl" ]; then
  {
    printf '%s::repeat=1\\n' "$OUTPUT_PREPARED_DIR/no_text.valid.jsonl"
    printf '%s::repeat=1\\n' "$OUTPUT_PREPARED_DIR/text.valid.jsonl"
  } > "$OUTPUT_PREPARED_DIR/mixed.valid.spec.txt"
fi
if [ -f "$OUTPUT_PREPARED_DIR/no_text.seen_valid.jsonl" ] && [ -f "$OUTPUT_PREPARED_DIR/text.seen_valid.jsonl" ]; then
  {
    printf '%s::repeat=1\\n' "$OUTPUT_PREPARED_DIR/no_text.seen_valid.jsonl"
    printf '%s::repeat=1\\n' "$OUTPUT_PREPARED_DIR/text.seen_valid.jsonl"
  } > "$OUTPUT_PREPARED_DIR/mixed.valid_seen.spec.txt"
fi
if [ -f "$OUTPUT_PREPARED_DIR/no_text.unseen_valid.jsonl" ] && [ -f "$OUTPUT_PREPARED_DIR/text.unseen_valid.jsonl" ]; then
  {
    printf '%s::repeat=1\\n' "$OUTPUT_PREPARED_DIR/no_text.unseen_valid.jsonl"
    printf '%s::repeat=1\\n' "$OUTPUT_PREPARED_DIR/text.unseen_valid.jsonl"
  } > "$OUTPUT_PREPARED_DIR/mixed.valid_unseen.spec.txt"
fi

echo "[speaker-seq-qz] preparing quick validation manifest"
extra_args=""
if [ "$LOCAL_FILES_ONLY" = "1" ]; then
  extra_args="\$extra_args --local-files-only"
fi
if [ "$OVERWRITE" = "1" ]; then
  extra_args="\$extra_args --overwrite"
fi
if [ -f "$QUICK_VALIDATION_JSONL" ]; then
  CUDA_VISIBLE_DEVICES=0 "\$PY" "\$ROOT/scripts/002031_precompute_wavlm_seq_features.py" \\
    --input-jsonl "$QUICK_VALIDATION_JSONL" \\
    --output-jsonl "$QUICK_SEQ_VALIDATION_JSONL" \\
    --speaker-seq-dir "$SPEAKER_SEQ_DIR/quick20" \\
    --model-name-or-path "$WAVLM_MODEL_PATH" \\
    --layer "$WAVLM_LAYER" \\
    --downsample-stride "$DOWNSAMPLE_STRIDE" \\
    --dtype "$SAVE_DTYPE" \\
    --batch-size "$BATCH_SIZE" \\
    --device cuda \\
    --summary-json "$QUICK_SEQ_VALIDATION_JSONL.summary.json" \\
    \$extra_args
fi
if [ "$RUN_DOMAIN_VALIDATION" = "1" ] && [ -f "$DOMAIN_VALIDATION_JSONL" ]; then
  CUDA_VISIBLE_DEVICES=0 "\$PY" "\$ROOT/scripts/002031_precompute_wavlm_seq_features.py" \\
    --input-jsonl "$DOMAIN_VALIDATION_JSONL" \\
    --output-jsonl "$DOMAIN_SEQ_VALIDATION_JSONL" \\
    --speaker-seq-dir "$SPEAKER_SEQ_DIR/domain50" \\
    --model-name-or-path "$WAVLM_MODEL_PATH" \\
    --layer "$WAVLM_LAYER" \\
    --downsample-stride "$DOWNSAMPLE_STRIDE" \\
    --dtype "$SAVE_DTYPE" \\
    --batch-size "$BATCH_SIZE" \\
    --device cuda \\
    --summary-json "$DOMAIN_SEQ_VALIDATION_JSONL.summary.json" \\
    \$extra_args
fi

echo "[speaker-seq-qz] verifying merged manifests"
"\$PY" - <<'PY'
import json
from pathlib import Path
root = Path("$OUTPUT_PREPARED_DIR")
required = ["no_text.train.jsonl", "text.train.jsonl"]
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
                if not row.get("speaker_seq_path"):
                    raise SystemExit(f"missing speaker_seq_path in {path}:{count}")
    print(f"[speaker-seq-qz] {name} rows={count}", flush=True)
for path_str in ["$QUICK_SEQ_VALIDATION_JSONL", "$DOMAIN_SEQ_VALIDATION_JSONL"]:
    path = Path(path_str)
    if not path.exists():
        continue
    with path.open("r", encoding="utf-8") as handle:
        first = next((line for line in handle if line.strip()), "")
    if first and not json.loads(first).get("speaker_seq_path"):
        raise SystemExit(f"missing speaker_seq_path in validation manifest: {path}")
    print(f"[speaker-seq-qz] validation ready: {path}", flush=True)
print("[speaker-seq-qz] verify ok", flush=True)
PY

echo "[speaker-seq-qz] done output=$OUTPUT_PREPARED_DIR quick=$QUICK_SEQ_VALIDATION_JSONL"
EOF
chmod +x "$RUNNER"

COMMAND="sh $RUNNER"

echo "=========================================="
echo "QZ submit: ver2.9 speaker sequence precompute"
echo "  JOB_NAME=$JOB_NAME"
echo "  ROOT=$ROOT"
echo "  INPUT_PREPARED_DIR=$INPUT_PREPARED_DIR"
echo "  OUTPUT_PREPARED_DIR=$OUTPUT_PREPARED_DIR"
echo "  SPEAKER_SEQ_DIR=$SPEAKER_SEQ_DIR"
echo "  WAVLM_MODEL_PATH=$WAVLM_MODEL_PATH"
echo "  WAVLM_LAYER=$WAVLM_LAYER"
echo "  DOWNSAMPLE_STRIDE=$DOWNSAMPLE_STRIDE"
echo "  SAVE_DTYPE=$SAVE_DTYPE"
echo "  LOCAL_FILES_ONLY=$LOCAL_FILES_ONLY"
echo "  NUM_SHARDS=$NUM_SHARDS"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  SPLITS=$SPLITS"
echo "  QUICK_VALIDATION_JSONL=$QUICK_VALIDATION_JSONL"
echo "  QUICK_SEQ_VALIDATION_JSONL=$QUICK_SEQ_VALIDATION_JSONL"
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

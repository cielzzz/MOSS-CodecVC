#!/usr/bin/env bash
# Ver2.8 pre-training checks. This script never submits QZ training.

set -euo pipefail

ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
PY="${PY:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
PREPARED_DIR="${PREPARED_DIR:-$ROOT/trainset/ver2_8_prepared}"
# repeat is total sampled copies. 5 = original text data + 4 extra copies.
TEXT_REPEAT="${TEXT_REPEAT:-5}"
MAX_ROWS_PER_SOURCE="${MAX_ROWS_PER_SOURCE:-16}"
AUDIT_MAX_ROWS="${AUDIT_MAX_ROWS:-256}"
RUN_SMOKE="${RUN_SMOKE:-0}"
CHECK_MIXED_PRECISION="${CHECK_MIXED_PRECISION:-no}"
BATCH_ID="${BATCH_ID:-$(date -u +%Y%m%d-%H%M%S)}"
CHECK_DIR="${CHECK_DIR:-$ROOT/outputs/debug_ver2_8_pretrain_check/$BATCH_ID}"

cd "$ROOT"
mkdir -p "$CHECK_DIR"

need_file() {
  local path="$1"
  if [ ! -s "$path" ]; then
    echo "ERROR: missing required prepared data file: $path" >&2
    exit 1
  fi
}

NO_TEXT_TRAIN="$PREPARED_DIR/no_text.train.jsonl"
NO_TEXT_VALID="$PREPARED_DIR/no_text.valid.jsonl"
TEXT_TRAIN="$PREPARED_DIR/text.train.jsonl"
TEXT_VALID="$PREPARED_DIR/text.valid.jsonl"
SUMMARY_JSON="$PREPARED_DIR/summary.json"

for path in "$NO_TEXT_TRAIN" "$NO_TEXT_VALID" "$TEXT_TRAIN" "$TEXT_VALID" "$SUMMARY_JSON"; do
  need_file "$path"
done

TRAIN_SPEC_FULL="$NO_TEXT_TRAIN::repeat=1,$TEXT_TRAIN::repeat=$TEXT_REPEAT"
VALID_SPEC_FULL="$NO_TEXT_VALID::repeat=1,$TEXT_VALID::repeat=1"
TRAIN_SPEC_LIMITED="$NO_TEXT_TRAIN::repeat=1::max_rows=$MAX_ROWS_PER_SOURCE,$TEXT_TRAIN::repeat=1::max_rows=$MAX_ROWS_PER_SOURCE"
VALID_SPEC_LIMITED="$NO_TEXT_VALID::repeat=1::max_rows=$MAX_ROWS_PER_SOURCE,$TEXT_VALID::repeat=1::max_rows=$MAX_ROWS_PER_SOURCE"
NO_TEXT_SPEC_LIMITED="$NO_TEXT_TRAIN::repeat=1::max_rows=$MAX_ROWS_PER_SOURCE"
TEXT_SPEC_LIMITED="$TEXT_TRAIN::repeat=1::max_rows=$MAX_ROWS_PER_SOURCE"

echo "=========================================="
echo "Ver2.8 pre-training checks"
echo "  ROOT=$ROOT"
echo "  PREPARED_DIR=$PREPARED_DIR"
echo "  CHECK_DIR=$CHECK_DIR"
echo "  TRAIN_SPEC_FULL=$TRAIN_SPEC_FULL"
echo "  VALID_SPEC_FULL=$VALID_SPEC_FULL"
echo "  MAX_ROWS_PER_SOURCE=$MAX_ROWS_PER_SOURCE"
echo "  RUN_SMOKE=$RUN_SMOKE"
echo "  CHECK_MIXED_PRECISION=$CHECK_MIXED_PRECISION"
echo "=========================================="

"$PY" -m py_compile \
  scripts/001020_extract_hubert_semantic_features.py \
  scripts/002013_prepare_ver2_8_train_valid.py \
  scripts/002017_summarize_ver2_8_data.py \
  scripts/004030_audit_source_semantic_features.py \
  scripts/002002_train_moss_codecvc_lora.py \
  scripts/003001_infer_moss_codecvc.py \
  scripts/004044_run_seedtts_validation_infer_persistent.py \
  moss_codecvc/models/source_semantic_memory.py \
  moss_codecvc/models/moss_codecvc_wrapper.py

"$PY" scripts/002017_summarize_ver2_8_data.py \
  --prepared-dir "$PREPARED_DIR" \
  --text-repeat "$TEXT_REPEAT" \
  --output-json "$CHECK_DIR/data_summary.json"

"$PY" scripts/004030_audit_source_semantic_features.py \
  --manifest "$NO_TEXT_TRAIN" \
  --max-rows "$AUDIT_MAX_ROWS" \
  --train-jsonl-spec "$TRAIN_SPEC_LIMITED" \
  --batch-size 4 \
  --max-batches 8 \
  --timbre-side-only \
  --output-dir "$CHECK_DIR/audit_mixed_pack"

COMMON_PACK_ARGS=(
  --config configs/remote_full.yaml
  --version ver2
  --use-timbre-memory
  --enable-role-routing
  --enable-target-head-routing
  --timbre-encoder-type conformer
  --timbre-encoder-layers 2
  --timbre-memory-tokens 16
  --timbre-speaker-conditioning
  --source-prosody-encoder-type conformer
  --source-prosody-encoder-layers 2
  --prosody-memory-tokens 8
  --source-prosody-no-text-gate 1.0
  --source-prosody-text-gate 0.0
  --speaker-encoder-type embedding_loader
  --speaker-embedding-dim 192
  --target-speaker-similarity-weight 0.05
  --source-speaker-suppression-weight 0.05
  --speaker-loss-warmup-steps 1000
  --speaker-loss-warmup-weight 0.02
  --speaker-loss-margin 0.10
  --lambda-route 0.01
  --routing-gate-lr-multiplier 10.0
  --content-ctc-head-lr-multiplier 1.0
  --timbre-adapter-init-gate -2.0
  --timbre-adapter-gate-lr-multiplier 10.0
  --lambda-prosody 0.05
  --prosody-f0-weight 0.0
  --prosody-voiced-weight 0.0
  --prosody-energy-weight 0.5
  --prosody-pause-weight 1.0
  --prosody-duration-weight 0.5
  --lambda-content 0
  --content-positive source
  --content-embedding-dim 0
  --content-embedding-weight 1.0
  --content-ctc-weight 0
  --content-ctc-vocab-size 0
  --content-ctc-blank-id 0
  --content-ctc-token-offset 1
  --content-token-vocab-size 0
  --content-token-weight 0.0
  --content-source-codec-weight 0.0
  --content-source-codec-codebooks 0,1,2,3
  --semantic-loss-weight 0.03
  --semantic-mode continuous
  --semantic-source mode_aware
  --semantic-vocab-size 0
  --semantic-feature-dim 768
  --semantic-feature-loss-type cosine
  --timbre-side-only
  --ref-content-suppression-weight 0.01
  --ref-content-suppression-margin 0.10
  --ref-content-suppression-source auto
  --ref-content-suppression-detach-ref
  --progress-loss-weight 0.02
  --stop-loss-weight 0.05
  --progress-num-bins 32
  --enable-source-semantic-memory
  --source-semantic-feature-dim 768
  --source-semantic-adapter-layers 28,30,32,34,35
  --source-semantic-no-text-gate 1.0
  --source-semantic-text-gate 0.0
  --no-source-semantic-learned-text-gate
  --source-semantic-progress-weight 0.02
  --source-semantic-dropout 0.1
  --source-semantic-init-gate -1.0
  --source-semantic-position-scale 0.10
  --source-semantic-monotonic-bias-strength 2.0
  --source-semantic-monotonic-bias-width 0.25
  --source-content-memory-type wavlm_bnf_continuous
  --source-content-vocab-size 0
  --source-content-padding-id 0
  --source-content-codec-bottleneck-dim 256
  --source-content-codec-codebooks first_4
  --no-source-content-dedup-units
  --source-codec-residual-memory-weight 0.20
  --no-source-codec-residual-memory-detach
  --source-semantic-gate-lr-multiplier 10.0
  --source-semantic-lr-multiplier 1.0
  --learning-rate 1e-5
  --weight-decay 0.01
  --warmup-ratio 0.03
  --per-device-batch-size 1
  --gradient-accumulation-steps 1
  --num-epochs 1
  --max-train-steps 0
  --mixed-precision "$CHECK_MIXED_PRECISION"
  --attn-implementation eager
  --lora-r 16
  --lora-alpha 32
  --lora-dropout 0.05
  --trainable-lora-modules all
  --lm-heads-mode none
  --channelwise-loss-weight 1,32
  --logging-steps 1
  --save-steps 0
  --num-workers 0
  --max-grad-norm 1.0
)

run_pack_only() {
  local name="$1"
  local spec="$2"
  shift 2
  local out_dir="$CHECK_DIR/pack_$name"
  mkdir -p "$out_dir"
  echo "[ver2.8-check] pack-only $name spec=$spec"
  "$PY" scripts/002002_train_moss_codecvc_lora.py \
    --train-jsonl-spec "$spec" \
    "${COMMON_PACK_ARGS[@]}" \
    --output-dir "$out_dir" \
    "$@" \
    --pack-only
}

run_pack_only no_text "$NO_TEXT_SPEC_LIMITED"
run_pack_only text "$TEXT_SPEC_LIMITED"
run_pack_only mixed "$TRAIN_SPEC_LIMITED" --eval-jsonl-spec "$VALID_SPEC_LIMITED" --eval-num-workers 0

if [ "$RUN_SMOKE" = "1" ]; then
  SMOKE_SPEC="${SMOKE_SPEC:-$NO_TEXT_SPEC_LIMITED}"
  SMOKE_OUT="${SMOKE_OUT:-$CHECK_DIR/smoke_out}"
  echo "[ver2.8-check] smoke-test spec=$SMOKE_SPEC out=$SMOKE_OUT"
  "$PY" scripts/002002_train_moss_codecvc_lora.py \
    --train-jsonl-spec "$SMOKE_SPEC" \
    "${COMMON_PACK_ARGS[@]}" \
    --output-dir "$SMOKE_OUT" \
    --max-train-steps 1 \
    --smoke-test
fi

echo "=========================================="
echo "Ver2.8 pre-training checks finished"
echo "  CHECK_DIR=$CHECK_DIR"
echo "  data_summary=$CHECK_DIR/data_summary.json"
echo "  audit=$CHECK_DIR/audit_mixed_pack/audit_report.json"
echo "=========================================="

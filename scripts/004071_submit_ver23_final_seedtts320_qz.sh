#!/usr/bin/env bash
# Submit one MTTS-only 8xH200 SeedTTS-320 evaluation for a Ver2.3 arm.
#
# Examples:
#   FAMILY=batch3436 ARM=B1 DRY_RUN=1 bash scripts/004071_submit_ver23_final_seedtts320_qz.sh
#   FAMILY=batch37 ARM=C2 REF_AUDIO_CFG_SCALE=1.4 DRY_RUN=0 bash scripts/004071_submit_ver23_final_seedtts320_qz.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
QZCLI="${QZCLI:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/qzcli_with_deps.sh}"
QZCLI_HOME="${QZCLI_HOME:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/.codex/qzcli_home}"

WORKSPACE="${WORKSPACE:-ws-8207e9e2-e733-4eec-a475-cfa1c36480ba}"
PROJECT="${PROJECT:-project-c67c548f-f02c-453b-ba5b-8745db6886e7}"
ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"  # MTTS-3-2-0715
COMPUTE_GROUP="${COMPUTE_GROUP:-$ALLOWED_COMPUTE_GROUP}"
SPEC="${SPEC:-67b10bc6-78b0-41a3-aaf4-358eeeb99009}"
QZCLI_GPU_TYPE_OVERRIDE="${QZCLI_GPU_TYPE_OVERRIDE:-NVIDIA_H200_SXM_141G}"
IMAGE="${IMAGE:-docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420}"
IMAGE_TYPE="${IMAGE_TYPE:-SOURCE_PRIVATE}"
FRAMEWORK="${FRAMEWORK:-pytorch}"
INSTANCES="${INSTANCES:-1}"
SHM_GI="${SHM_GI:-1200}"
PRIORITY="${PRIORITY:-10}"

FAMILY="${FAMILY:-batch3436}"
ARM="${ARM:-B1}"
TRAIN_STAMP="${TRAIN_STAMP:-20260710_mtts}"
EVAL_STEP="${EVAL_STEP:-3000}"
REF_AUDIO_CFG_SCALE="${REF_AUDIO_CFG_SCALE:-1.0}"
SEED="${SEED:-1234}"
RUN_TAG="${RUN_TAG:-20260711_mtts}"
DRY_RUN="${DRY_RUN:-1}"
FORCE="${FORCE:-0}"
ENTRYPOINT="${VER23_FINAL_320_ENTRYPOINT:-0}"

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
ASR_PYTHON="${ASR_PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
VALIDATION_JSONL="${VALIDATION_JSONL:-$PROJECT_ROOT/testset/validation/seedtts_vc_ver2_3_validation.jsonl}"
SPEECHBRAIN_ECAPA_MODEL_SOURCE="${SPEECHBRAIN_ECAPA_MODEL_SOURCE:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/models/speechbrain/spkrec-ecapa-voxceleb}"

if [ "$COMPUTE_GROUP" != "$ALLOWED_COMPUTE_GROUP" ]; then
  echo "ERROR: final SeedTTS-320 is restricted to MTTS-3-2-0715 ($ALLOWED_COMPUTE_GROUP); got $COMPUTE_GROUP" >&2
  exit 2
fi
case "$DRY_RUN:$FORCE:$ENTRYPOINT" in
  [01]:[01]:[01]) ;;
  *) echo "ERROR: DRY_RUN, FORCE, and VER23_FINAL_320_ENTRYPOINT must be 0 or 1" >&2; exit 2 ;;
esac
case "$EVAL_STEP" in
  3000) ;;
  *) echo "ERROR: this final-eval wrapper only accepts EVAL_STEP=3000" >&2; exit 2 ;;
esac

arm_info() {
  case "$FAMILY:$ARM" in
    batch3436:B1)
      printf '%s\t%s\t%s\n' \
        "ver23_content_side_batch3436_B1_ver23_bnf_last16_3k_$TRAIN_STAMP" \
        "ver23_batch3436_B1_bnf_last16" \
        "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch3436_eval_20260711_1092820"
      ;;
    batch3436:B2)
      printf '%s\t%s\t%s\n' \
        "ver23_content_side_batch3436_B2_ver23_text_r1_3k_$TRAIN_STAMP" \
        "ver23_batch3436_B2_text_r1" \
        "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch3436_eval_20260711_1092820"
      ;;
    batch3436:A1)
      printf '%s\t%s\t%s\n' \
        "ver23_content_side_batch3436_A1_ver23_stronger_decouple_3k_$TRAIN_STAMP" \
        "ver23_batch3436_A1_stronger" \
        "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch3436_eval_20260711_1092820"
      ;;
    batch3436:B3)
      printf '%s\t%s\t%s\n' \
        "ver23_content_side_batch3436_B3_ver23_weaker_decouple_3k_$TRAIN_STAMP" \
        "ver23_batch3436_B3_weaker" \
        "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch3436_eval_20260711_1092820"
      ;;
    batch3436:A2)
      printf '%s\t%s\t%s\n' \
        "ver23_content_side_batch3436_A2_ver23_ctc_3k_$TRAIN_STAMP" \
        "ver23_batch3436_A2_ctc" \
        "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch3436_eval_20260711_1092820"
      ;;
    batch37:C1)
      printf '%s\t%s\t%s\n' \
        "ver23_content_side_batch37_C1_ver23_compact_content_lr_warmup_3k_$TRAIN_STAMP" \
        "ver23_batch37_C1_compact" \
        "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch37_eval_20260711_1092820"
      ;;
    batch37:C2)
      printf '%s\t%s\t%s\n' \
        "ver23_content_side_batch37_C2_ver23_true_ref_audio_cfg_3k_$TRAIN_STAMP" \
        "ver23_batch37_C2_refcfg" \
        "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch37_eval_20260711_1092820"
      ;;
    *) echo "ERROR: unsupported FAMILY/ARM: $FAMILY/$ARM" >&2; return 2 ;;
  esac
}

IFS=$'\t' read -r RUN_DIR_NAME RUN_LABEL DEFAULT_CODE_ROOT <<<"$(arm_info)"
CODE_ROOT="${CODE_ROOT:-$DEFAULT_CODE_ROOT}"

if [ "$FAMILY:$ARM" = "batch37:C2" ]; then
  case "$REF_AUDIO_CFG_SCALE" in
    1.0|1.2|1.4|1.6) ;;
    *) echo "ERROR: C2 REF_AUDIO_CFG_SCALE must be 1.0, 1.2, 1.4, or 1.6" >&2; exit 2 ;;
  esac
elif [ "$REF_AUDIO_CFG_SCALE" != "1.0" ]; then
  echo "ERROR: REF_AUDIO_CFG_SCALE must be 1.0 outside Batch-37 C2" >&2
  exit 2
fi

CFG_TAG=$(printf '%s' "$REF_AUDIO_CFG_SCALE" | tr '.' 'p')
MODEL_PATH="$PROJECT_ROOT/outputs/lora_runs/$RUN_DIR_NAME/step-$EVAL_STEP"
RUN_ID="${RUN_LABEL}_step-${EVAL_STEP}_cfg${CFG_TAG}_seedtts320_d2d3_seed${SEED}"
RECORD_ROOT="${RECORD_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/ver23_final_seedtts320_${FAMILY}_${ARM}_cfg${CFG_TAG}_${RUN_TAG}}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/testset/outputs/ver23_final_seedtts320_${FAMILY}_${ARM}_cfg${CFG_TAG}_${RUN_TAG}}"
OUTPUT_DIR="$EVAL_ROOT/$RUN_ID"
JOB_NAME="${JOB_NAME:-ver23_final320_${FAMILY}_${ARM}_cfg${CFG_TAG}_${RUN_TAG}}"

validate_checkpoint() {
  local required=(adapter_model.safetensors adapter_config.json README.md timbre_memory_adapter.pt timbre_memory_config.json)
  local name
  if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: missing checkpoint: $MODEL_PATH" >&2
    return 1
  fi
  for name in "${required[@]}"; do
    if [ ! -s "$MODEL_PATH/$name" ]; then
      echo "ERROR: missing or empty checkpoint file: $MODEL_PATH/$name" >&2
      return 1
    fi
  done
  "$PYTHON" -c 'import json,sys; [json.load(open(p, encoding="utf-8")) for p in sys.argv[1:]]' \
    "$MODEL_PATH/adapter_config.json" "$MODEL_PATH/timbre_memory_config.json"
}

run_entrypoint() {
  validate_checkpoint
  mkdir -p "$RECORD_ROOT" "$EVAL_ROOT" "$OUTPUT_DIR"
  exec > >(tee -a "$RECORD_ROOT/run.log") 2>&1

  echo "[ver23-final320] start=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
  echo "[ver23-final320] family=$FAMILY arm=$ARM model=$MODEL_PATH"
  echo "[ver23-final320] run_id=$RUN_ID ref_audio_cfg_scale=$REF_AUDIO_CFG_SCALE"
  echo "[ver23-final320] code_root=$CODE_ROOT output=$OUTPUT_DIR"
  nvidia-smi || true

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=8 \
  HF_MODULES_CACHE_ROOT="$OUTPUT_DIR/.hf_modules_cache" \
  SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH=0.0 \
  TEMPERATURE=0.7 \
  NO_TEXT_AUDIO_TEMPERATURE=1.1 \
  NO_TEXT_AUDIO_TOP_P=0.7 \
  NO_TEXT_AUDIO_TOP_K=20 \
  AUDIO_TEMPERATURE=1.1 \
  AUDIO_TOP_P=0.7 \
  AUDIO_TOP_K=20 \
  SPEAKER_ENCODER_TYPE=embedding_loader \
  TIMBRE_SIDE_ONLY=0 \
  TIMBRE_CFG_SCALE=1.0 \
  REF_AUDIO_CFG_SCALE="$REF_AUDIO_CFG_SCALE" \
  REF_PROMPT_CODEC_PERMUTATION=0 \
  PYTHON="$PYTHON" \
  ASR_PYTHON="$ASR_PYTHON" \
  VALIDATION_JSONL="$VALIDATION_JSONL" \
  MODEL_PATH="$MODEL_PATH" \
  RUN_ID="$RUN_ID" \
  RUN_LABEL="$RUN_ID" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  MODE=all \
  MAX_CASES=0 \
  PER_MODE=0 \
  PER_CELL=0 \
  DECODING_PROFILE=default \
  PERSISTENT_INFER=1 \
  OVERWRITE_INFER=1 \
  RESET_MANIFESTS=1 \
  RUN_ASR=1 \
  RUN_SUMMARY=1 \
  BUILD_PAGE=0 \
  GPU_COUNT=8 \
  NUM_SHARDS=8 \
  ASR_NUM_SHARDS=8 \
  SEED="$SEED" \
  bash "$CODE_ROOT/scripts/004039_run_seedtts_validation_eval.sh"

  "$PYTHON" "$CODE_ROOT/scripts/004056_summarize_seedtts_ref_content_similarity.py" \
    --asr-jsonl "$OUTPUT_DIR/${RUN_ID}.asr_eval.jsonl" \
    --output-json "$OUTPUT_DIR/${RUN_ID}.ref_content_similarity_summary.json" \
    --output-md "$OUTPUT_DIR/${RUN_ID}.ref_content_similarity_summary.md"

  "$PYTHON" "$CODE_ROOT/scripts/004048_summarize_seedtts_ablation_metrics.py" \
    --validation-jsonl "$VALIDATION_JSONL" \
    --run "$RUN_ID=$OUTPUT_DIR" \
    --output-csv "$EVAL_ROOT/${RUN_ID}.dual_encoder_cases.csv" \
    --summary-json "$EVAL_ROOT/${RUN_ID}.dual_encoder_summary.json" \
    --summary-md "$EVAL_ROOT/${RUN_ID}.dual_encoder_summary.md" \
    --speaker-device cuda:0 \
    --extra-speaker-encoder speechbrain_ecapa \
    --extra-speaker-device cuda:1 \
    --speechbrain-ecapa-model-source "$SPEECHBRAIN_ECAPA_MODEL_SOURCE"

  "$PYTHON" "$CODE_ROOT/scripts/004063_analyze_seedtts320_diagnostics.py" \
    --validation-jsonl "$VALIDATION_JSONL" \
    --sim-cases-csv "$EVAL_ROOT/${RUN_ID}.dual_encoder_cases.csv" \
    --run "$RUN_ID=$OUTPUT_DIR" \
    --output-dir "$EVAL_ROOT/diagnostics" \
    --prefix "$RUN_ID"

  echo "[ver23-final320] complete=$(date -u +%Y-%m-%dT%H:%M:%SZ) output=$EVAL_ROOT"
}

if [ "$ENTRYPOINT" = "1" ]; then
  run_entrypoint
  exit 0
fi

if [ ! -d "$PROJECT_ROOT" ] || [ ! -d "$CODE_ROOT" ]; then
  echo "ERROR: missing PROJECT_ROOT or CODE_ROOT" >&2
  exit 1
fi
if [ ! -x "$QZCLI" ] || [ ! -x "$PYTHON" ] || [ ! -x "$ASR_PYTHON" ]; then
  echo "ERROR: missing qzcli or Python interpreter" >&2
  exit 1
fi
if [ ! -s "$VALIDATION_JSONL" ] || [ ! -d "$SPEECHBRAIN_ECAPA_MODEL_SOURCE" ]; then
  echo "ERROR: missing validation JSONL or local SpeechBrain ECAPA model" >&2
  exit 1
fi
if ! grep -q 'hf_modules_cache_root' "$CODE_ROOT/scripts/004039_run_seedtts_validation_eval.sh"; then
  echo "ERROR: CODE_ROOT lacks per-shard HF dynamic-module cache isolation: $CODE_ROOT" >&2
  exit 1
fi
validate_checkpoint
mkdir -p "$RECORD_ROOT" "$EVAL_ROOT" "$QZCLI_HOME"
if [ "$FORCE" != "1" ] && { [ -s "$EVAL_ROOT/${RUN_ID}.dual_encoder_summary.json" ] || [ -s "$RECORD_ROOT/submitted_jobs.tsv" ]; }; then
  echo "ERROR: existing final-eval output/submission; use FORCE=1 only for an intentional rerun" >&2
  exit 1
fi

COMMAND="env VER23_FINAL_320_ENTRYPOINT=1 FAMILY=$FAMILY ARM=$ARM TRAIN_STAMP=$TRAIN_STAMP EVAL_STEP=$EVAL_STEP REF_AUDIO_CFG_SCALE=$REF_AUDIO_CFG_SCALE SEED=$SEED RUN_TAG=$RUN_TAG FORCE=$FORCE PROJECT_ROOT=$PROJECT_ROOT CODE_ROOT=$CODE_ROOT RECORD_ROOT=$RECORD_ROOT EVAL_ROOT=$EVAL_ROOT VALIDATION_JSONL=$VALIDATION_JSONL bash $CODE_ROOT/scripts/004071_submit_ver23_final_seedtts320_qz.sh"
SUBMIT_OUTPUT="$RECORD_ROOT/submit_output.txt"

echo "=========================================="
echo "QZ submit: Ver2.3 final SeedTTS-320"
echo "  JOB_NAME=$JOB_NAME"
echo "  FAMILY=$FAMILY ARM=$ARM REF_AUDIO_CFG_SCALE=$REF_AUDIO_CFG_SCALE"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  CODE_ROOT=$CODE_ROOT"
echo "  RECORD_ROOT=$RECORD_ROOT"
echo "  EVAL_ROOT=$EVAL_ROOT"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP (MTTS-3-2-0715 only)"
echo "  DRY_RUN=$DRY_RUN"
echo "  COMMAND=$COMMAND"
echo "=========================================="

qz_args=(
  create-job
  --name "$JOB_NAME"
  --command "$COMMAND"
  --workspace "$WORKSPACE"
  --project "$PROJECT"
  --compute-group "$COMPUTE_GROUP"
  --spec "$SPEC"
  --image "$IMAGE"
  --image-type "$IMAGE_TYPE"
  --instances "$INSTANCES"
  --shm "$SHM_GI"
  --priority "$PRIORITY"
  --framework "$FRAMEWORK"
)
if [ "$DRY_RUN" = "1" ]; then
  qz_args+=(--dry-run)
fi

set +e
output=$(
  env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
    HOME="$QZCLI_HOME" \
    QZCLI_GPU_TYPE_OVERRIDE="$QZCLI_GPU_TYPE_OVERRIDE" \
    "$QZCLI" "${qz_args[@]}" 2>&1
)
status=$?
set -e
printf '%s\n' "$output" | tee "$SUBMIT_OUTPUT"
if [ "$status" -ne 0 ]; then
  echo "ERROR: QZ submission failed; see $SUBMIT_OUTPUT" >&2
  exit "$status"
fi
if [ "$DRY_RUN" = "1" ]; then
  echo "[ver23-final320] dry-run passed; no job submitted"
  exit 0
fi

job_id=$(printf '%s\n' "$output" | grep -Eo 'job-[0-9a-fA-F-]{36}' | tail -n 1 || true)
{
  printf 'job_name\tjob_id\tfamily\tarm\tref_audio_cfg_scale\tcompute_group\tmodel_path\trecord_root\teval_root\n'
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$JOB_NAME" "$job_id" "$FAMILY" "$ARM" "$REF_AUDIO_CFG_SCALE" "$COMPUTE_GROUP" "$MODEL_PATH" "$RECORD_ROOT" "$EVAL_ROOT"
} > "$RECORD_ROOT/submitted_jobs.tsv"
echo "[ver23-final320] submitted job_id=$job_id record=$RECORD_ROOT/submitted_jobs.tsv"

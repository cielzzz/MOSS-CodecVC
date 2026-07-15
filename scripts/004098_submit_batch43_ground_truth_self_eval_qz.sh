#!/usr/bin/env bash
# Prepare and (only when explicitly requested) submit Batch-43 Ground truth
# source-self calibration scoring through the existing 004091 scorer pipeline.
#
# Default behavior is a QZ dry-run.  To prepare inputs without contacting QZ:
#   PREPARE_ONLY=1 bash scripts/004098_submit_batch43_ground_truth_self_eval_qz.sh
#
# A real submission requires both DRY_RUN=0 and the explicit confirmation gate:
#   DRY_RUN=0 CONFIRM_GROUND_TRUTH_SELF_EVAL=1 \
#     bash scripts/004098_submit_batch43_ground_truth_self_eval_qz.sh
#
# This is NOT a target-speaker VC ground-truth row.  It scores the untouched
# field-5 source waveform against itself for speaker-scorer calibration and
# against field-4 text for the raw-source ASR ceiling.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
PYTHON="${PYTHON:-python3}"
PREP_SCRIPT="${PREP_SCRIPT:-$PROJECT_ROOT/scripts/004097_prepare_batch43_ground_truth_self_eval.py}"
SCORER_WRAPPER="${SCORER_WRAPPER:-$PROJECT_ROOT/scripts/004091_submit_batch42_unified_scorers_qz.sh}"

AUDIT_ROOT="${AUDIT_ROOT:-$PROJECT_ROOT/testset/outputs/batch42_seedtts_eval_audit_20260711}"
DATASET_ROOT="${DATASET_ROOT:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/batch42/datasets/seed-tts-eval/seedtts_testset}"
EN_MANIFEST="${EN_MANIFEST:-$AUDIT_ROOT/official_en_vc_minus_internal320_strict_case.lst}"
ZH_MANIFEST="${ZH_MANIFEST:-$AUDIT_ROOT/official_zh_vc_minus_internal320_strict_case.lst}"
EXPECTED_EN="${EXPECTED_EN:-567}"
EXPECTED_ZH="${EXPECTED_ZH:-1194}"
EXPECTED_EN_SHA256="${EXPECTED_EN_SHA256:-48549d8029e680d74656660191c4641ca5a8040ccbe3252ce89bfc3b0c9c75ae}"
EXPECTED_ZH_SHA256="${EXPECTED_ZH_SHA256:-4b637cc1cff33dc369954755538d12396fc92d439a52742103a29b7c563cf6df}"

RUN_TAG="${RUN_TAG:-20260712_mtts}"
INPUT_ROOT="${INPUT_ROOT:-$PROJECT_ROOT/testset/outputs/batch43_ground_truth_source_self_inputs_20260712}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/testset/outputs/batch43_ground_truth_source_self_scored_20260712_mtts}"
RECORD_ROOT="${RECORD_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/batch43_ground_truth_source_self_eval_${RUN_TAG}}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
DRY_RUN="${DRY_RUN:-1}"
FORCE="${FORCE:-0}"
ENABLE_QWEN_ASR="${ENABLE_QWEN_ASR:-0}"
CONFIRM_GROUND_TRUTH_SELF_EVAL="${CONFIRM_GROUND_TRUTH_SELF_EVAL:-0}"

# Hard QZ contract inherited by and explicitly passed to 004091.
WORKSPACE="${WORKSPACE:-ws-8207e9e2-e733-4eec-a475-cfa1c36480ba}"
PROJECT="${PROJECT:-project-c67c548f-f02c-453b-ba5b-8745db6886e7}"
COMPUTE_GROUP="${COMPUTE_GROUP:-lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122}"
SPEC="${SPEC:-67b10bc6-78b0-41a3-aaf4-358eeeb99009}"
INSTANCES="${INSTANCES:-1}"
QZCLI_GPU_TYPE_OVERRIDE="${QZCLI_GPU_TYPE_OVERRIDE:-NVIDIA_H200_SXM_141G}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

case "$PREPARE_ONLY:$DRY_RUN:$FORCE:$ENABLE_QWEN_ASR:$CONFIRM_GROUND_TRUTH_SELF_EVAL" in
  [01]:[01]:[01]:[01]:[01]) ;;
  *) die "PREPARE_ONLY, DRY_RUN, FORCE, ENABLE_QWEN_ASR, and CONFIRM_GROUND_TRUTH_SELF_EVAL must be 0 or 1" ;;
esac
[ -x "$PYTHON" ] || command -v "$PYTHON" >/dev/null 2>&1 || die "missing Python: $PYTHON"
[ -s "$PREP_SCRIPT" ] || die "missing preparation script: $PREP_SCRIPT"
[ "$COMPUTE_GROUP" = "lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122" ] || die "only MTTS-3-2-0715 is allowed"
[ "$SPEC" = "67b10bc6-78b0-41a3-aaf4-358eeeb99009" ] || die "unexpected QZ spec: $SPEC"
[ "$INSTANCES" = "1" ] || die "exactly one 8xH200 instance is required"
[ "$QZCLI_GPU_TYPE_OVERRIDE" = "NVIDIA_H200_SXM_141G" ] || die "only H200 is allowed"
if [ "$DRY_RUN" = "0" ] && [ "$CONFIRM_GROUND_TRUTH_SELF_EVAL" != "1" ]; then
  die "live submission requires CONFIRM_GROUND_TRUTH_SELF_EVAL=1"
fi

"$PYTHON" "$PREP_SCRIPT" \
  --dataset-root "$DATASET_ROOT" \
  --en-manifest "$EN_MANIFEST" \
  --zh-manifest "$ZH_MANIFEST" \
  --output-dir "$INPUT_ROOT" \
  --expected-en "$EXPECTED_EN" \
  --expected-zh "$EXPECTED_ZH" \
  --expected-en-sha256 "$EXPECTED_EN_SHA256" \
  --expected-zh-sha256 "$EXPECTED_ZH_SHA256"

EN_INPUT="$INPUT_ROOT/ground_truth_source_self.en.input.jsonl"
ZH_INPUT="$INPUT_ROOT/ground_truth_source_self.zh.input.jsonl"
AUDIT_JSON="$INPUT_ROOT/GROUND_TRUTH_SOURCE_SELF_AUDIT.json"
[ -s "$EN_INPUT" ] || die "preparation did not emit EN input: $EN_INPUT"
[ -s "$ZH_INPUT" ] || die "preparation did not emit ZH input: $ZH_INPUT"
[ -s "$AUDIT_JSON" ] || die "preparation did not emit audit: $AUDIT_JSON"

echo "[batch43-ground-truth] prepared calibration-only inputs"
echo "  EN_INPUT=$EN_INPUT"
echo "  ZH_INPUT=$ZH_INPUT"
echo "  AUDIT=$AUDIT_JSON"
echo "  WARNING=same-file source self-SIM is not target-speaker VC performance"
echo "  ZH_HARD=N/A (official hardcase.lst has no source/ground-truth target waveform)"

if [ "$PREPARE_ONLY" = "1" ]; then
  echo "[batch43-ground-truth] PREPARE_ONLY=1; no QZ command invoked"
  exit 0
fi

[ -s "$SCORER_WRAPPER" ] || die "missing unified scorer wrapper: $SCORER_WRAPPER"
env \
  SYSTEM_TAG=ground_truth \
  INPUT_SYSTEM_ID=ground_truth \
  EN_INPUT="$EN_INPUT" \
  ZH_INPUT="$ZH_INPUT" \
  EN_TEST_SET_ID=seedtts-vc-en-internal320-disjoint \
  ZH_TEST_SET_ID=seedtts-vc-zh-internal320-disjoint \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  RECORD_ROOT="$RECORD_ROOT" \
  RUN_TAG="$RUN_TAG" \
  JOB_NAME="batch43_ground_truth_source_self_eval_${RUN_TAG}" \
  DRY_RUN="$DRY_RUN" \
  FORCE="$FORCE" \
  ENABLE_QWEN_ASR="$ENABLE_QWEN_ASR" \
  WORKSPACE="$WORKSPACE" \
  PROJECT="$PROJECT" \
  COMPUTE_GROUP="$COMPUTE_GROUP" \
  SPEC="$SPEC" \
  INSTANCES="$INSTANCES" \
  QZCLI_GPU_TYPE_OVERRIDE="$QZCLI_GPU_TYPE_OVERRIDE" \
  bash "$SCORER_WRAPPER"

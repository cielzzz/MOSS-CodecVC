#!/usr/bin/env bash
# Run the registered Batch-44 r3/r5 paired full320 evaluation on the local
# two-RTX-4090 development host.  This wrapper has no remote-submit path.
#
# Safe default (prints the resolved plan and exits):
#   STEP=10000 bash scripts/004118_run_batch44_v1_paired_full320_local.sh
#
# Read-only-ish preflight (audits data/checkpoints/provenance and both GPUs,
# but does not start inference/scoring):
#   STEP=10000 ACTION=preflight \
#     bash scripts/004118_run_batch44_v1_paired_full320_local.sh
#
# Explicit long evaluation:
#   STEP=10000 ACTION=run CONFIRM_LOCAL_FULL320=1 \
#     bash scripts/004118_run_batch44_v1_paired_full320_local.sh

set -euo pipefail

CANONICAL_PROJECT_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
PROJECT_ROOT="${PROJECT_ROOT:-$CANONICAL_PROJECT_ROOT}"
CODE_ROOT="${CODE_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch37_eval_20260711_1092820}"
ENGINE_SOURCE="${BATCH44_LOCAL_FULL320_ENGINE:-$CANONICAL_PROJECT_ROOT/scripts/004112_submit_batch44_v1_paired_full320_qz.sh}"
RUNNER_SOURCE="${BATCH44_LOCAL_FULL320_RUNNER_SOURCE:-$(readlink -f "${BASH_SOURCE[0]}")}"

STAMP="20260713"
STEP="${STEP:-10000}"
SEED="${SEED:-1234}"
ACTION="${ACTION:-plan}"
CONFIRM_LOCAL_FULL320="${CONFIRM_LOCAL_FULL320:-0}"
TEST_MODE="${BATCH44_LOCAL_FULL320_TEST_MODE:-0}"
MIN_CHECKPOINT_AGE_SEC="${MIN_CHECKPOINT_AGE_SEC:-90}"

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
ASR_PYTHON="${ASR_PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
SPEAKER_SIM_ROOT="${SPEAKER_SIM_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction}"
SPEECHBRAIN_ECAPA_MODEL_SOURCE="${SPEECHBRAIN_ECAPA_MODEL_SOURCE:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/models/speechbrain/spkrec-ecapa-voxceleb}"
VALIDATION_JSONL="${VALIDATION_JSONL:-$PROJECT_ROOT/testset/validation/seedtts_vc_ver2_3_validation.jsonl}"
R3_RUN_DIR="${R3_RUN_DIR:-$PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r3_v1_30k}"
R5_RUN_DIR="${R5_RUN_DIR:-$PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r5_v1_30k}"
TRAIN_PAIR_LEDGER="$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713/submitted_pair.tsv"

RECORD_ROOT="${RECORD_ROOT:-$PROJECT_ROOT/trainset/local_jobs/ver23_batch44_paired_full320_step${STEP}_${STAMP}}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/testset/outputs/ver23_batch44_paired_full320_${STAMP}}"
STEP_ROOT="$EVAL_ROOT/step-$STEP"
FROZEN_RUNNER="$RECORD_ROOT/004118_run_batch44_v1_paired_full320_local.frozen.sh"
FROZEN_ENGINE="$RECORD_ROOT/004112_batch44_v1_paired_full320_engine.frozen.sh"
INPUTS_MANIFEST="$RECORD_ROOT/frozen_inputs.sha256"
RUN_LOCK="$RECORD_ROOT/.local_run.lock"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

case "$STEP" in
  10000|20000|26000|28000|30000) ;;
  *) die "STEP must be a registered Batch-44 full320 checkpoint: 10000/20000/26000/28000/30000; got $STEP" ;;
esac
case "$ACTION" in
  plan|preflight|run) ;;
  *) die "ACTION must be plan, preflight, or run; got $ACTION" ;;
esac
case "$CONFIRM_LOCAL_FULL320:$TEST_MODE" in
  [01]:[01]) ;;
  *) die "CONFIRM_LOCAL_FULL320 and BATCH44_LOCAL_FULL320_TEST_MODE must be 0 or 1" ;;
esac
case "$MIN_CHECKPOINT_AGE_SEC" in
  ''|*[!0-9]*) die "MIN_CHECKPOINT_AGE_SEC must be a non-negative integer" ;;
esac
if [ "$PROJECT_ROOT" != "$CANONICAL_PROJECT_ROOT" ] && [ "$TEST_MODE" != "1" ]; then
  die "non-canonical PROJECT_ROOT is allowed only in local full320 test mode"
fi
if [ "$TEST_MODE" = "1" ] && [ "$ACTION" != "plan" ]; then
  die "local full320 test mode may only exercise ACTION=plan"
fi
if [ "$ACTION" = "run" ] && [ "$CONFIRM_LOCAL_FULL320" != "1" ]; then
  die "ACTION=run requires CONFIRM_LOCAL_FULL320=1"
fi

expected_record="$PROJECT_ROOT/trainset/local_jobs/ver23_batch44_paired_full320_step${STEP}_${STAMP}"
expected_eval="$PROJECT_ROOT/testset/outputs/ver23_batch44_paired_full320_${STAMP}"
if [ "$TEST_MODE" != "1" ]; then
  [ "$PROJECT_ROOT" = "$CANONICAL_PROJECT_ROOT" ] || die "local run requires canonical PROJECT_ROOT"
  [ "$RECORD_ROOT" = "$expected_record" ] || die "local record root must be $expected_record"
  [ "$EVAL_ROOT" = "$expected_eval" ] || die "local eval root must be $expected_eval"
  [ "$RUNNER_SOURCE" = "$CANONICAL_PROJECT_ROOT/scripts/004118_run_batch44_v1_paired_full320_local.sh" ] \
    || die "local runner source is not canonical"
  [ "$ENGINE_SOURCE" = "$CANONICAL_PROJECT_ROOT/scripts/004112_submit_batch44_v1_paired_full320_qz.sh" ] \
    || die "local engine source is not canonical"
fi

[ -s "$RUNNER_SOURCE" ] || die "missing local runner source: $RUNNER_SOURCE"
[ -s "$ENGINE_SOURCE" ] || die "missing paired full320 engine: $ENGINE_SOURCE"
bash -n "$RUNNER_SOURCE"
bash -n "$ENGINE_SOURCE"

echo "=========================================="
echo "Batch-44 paired full320 local plan"
echo "  ACTION=$ACTION"
echo "  STEP=$STEP"
echo "  BACKEND=local"
echo "  HOST_REQUIREMENT=xyzhang-dev--*"
echo "  GPU_REQUIREMENT=2x NVIDIA GeForce RTX 4090 (indices 0,1)"
echo "  LANE_EXECUTION=sequential"
echo "  LANES=r3/no_text160,r3/text160,r5/no_text160,r5/text160"
echo "  GPU_PLAN=all lanes use local indices 0,1"
echo "  RECORD_ROOT=$RECORD_ROOT"
echo "  STEP_ROOT=$STEP_ROOT"
echo "  R3_CHECKPOINT=$R3_RUN_DIR/step-$STEP"
echo "  R5_CHECKPOINT=$R5_RUN_DIR/step-$STEP"
echo "  VALIDATION_JSONL=$VALIDATION_JSONL"
echo "  CODE_ROOT=$CODE_ROOT"
echo "=========================================="

if [ "$ACTION" = "plan" ]; then
  echo "[batch44-full320-local] plan complete; no GPU work was started"
  exit 0
fi

[ ! -L "$RECORD_ROOT" ] || die "local record root may not be a symlink: $RECORD_ROOT"
[ ! -L "$EVAL_ROOT" ] || die "local eval root may not be a symlink: $EVAL_ROOT"
[ ! -L "$STEP_ROOT" ] || die "local step root may not be a symlink: $STEP_ROOT"
mkdir -p "$RECORD_ROOT" "$STEP_ROOT/runs" "$STEP_ROOT/aggregate"
[ ! -e "$RECORD_ROOT/submitted_jobs.tsv" ] && [ ! -L "$RECORD_ROOT/submitted_jobs.tsv" ] \
  || die "local record must not contain submitted_jobs.tsv: $RECORD_ROOT"
[ ! -e "$RECORD_ROOT/COMPLETED.json" ] && [ ! -L "$RECORD_ROOT/COMPLETED.json" ] \
  || die "local full320 has existing completion evidence: $RECORD_ROOT/COMPLETED.json"
[ ! -e "$RECORD_ROOT/complete.marker" ] && [ ! -L "$RECORD_ROOT/complete.marker" ] \
  || die "local full320 has existing marker evidence: $RECORD_ROOT/complete.marker"
[ ! -e "$STEP_ROOT/aggregate/paired_metrics.tsv" ] && [ ! -L "$STEP_ROOT/aggregate/paired_metrics.tsv" ] \
  || die "paired metrics already exist; refusing to overwrite: $STEP_ROOT"

if [ -s "$FROZEN_RUNNER" ]; then
  cmp -s "$RUNNER_SOURCE" "$FROZEN_RUNNER" || die "frozen local runner drift: $FROZEN_RUNNER"
else
  cp "$RUNNER_SOURCE" "$FROZEN_RUNNER"
  chmod 0555 "$FROZEN_RUNNER"
fi
if [ -s "$FROZEN_ENGINE" ]; then
  cmp -s "$ENGINE_SOURCE" "$FROZEN_ENGINE" || die "frozen local engine drift: $FROZEN_ENGINE"
else
  cp "$ENGINE_SOURCE" "$FROZEN_ENGINE"
  chmod 0555 "$FROZEN_ENGINE"
fi

manifest_tmp="$INPUTS_MANIFEST.tmp.$$"
{
  sha256sum "$FROZEN_RUNNER" "$FROZEN_ENGINE" "$VALIDATION_JSONL" "$TRAIN_PAIR_LEDGER"
  sha256sum \
    "$CODE_ROOT/scripts/004039_run_seedtts_validation_eval.sh" \
    "$CODE_ROOT/scripts/004042_summarize_seedtts_validation_eval.py" \
    "$CODE_ROOT/scripts/004044_run_seedtts_validation_infer_persistent.py" \
    "$CODE_ROOT/scripts/004048_summarize_seedtts_ablation_metrics.py" \
    "$CODE_ROOT/scripts/004056_summarize_seedtts_ref_content_similarity.py" \
    "$CODE_ROOT/moss_codecvc/models/moss_codecvc_wrapper.py"
} > "$manifest_tmp"
mv "$manifest_tmp" "$INPUTS_MANIFEST"

if [ "$ACTION" = "run" ]; then
  if ! mkdir "$RUN_LOCK" 2>/dev/null; then
    die "persistent local-run lock exists; inspect local processes/artifacts before recovery: $RUN_LOCK"
  fi
  "$PYTHON" - "$RUN_LOCK/owner.json" "$STEP" <<'PY'
import datetime as dt
import json
import os
import socket
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "backend": "local",
    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "hostname": socket.gethostname(),
    "pid": os.getppid(),
    "step": int(sys.argv[2]),
    "policy": "persistent lock; inspect process and artifacts before manual recovery",
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
fi

preflight_only=0
[ "$ACTION" = "preflight" ] && preflight_only=1

env \
  BATCH44_PAIRED_FULL320_ENTRYPOINT=1 \
  BATCH44_FULL320_BACKEND=local \
  BATCH44_LOCAL_PREFLIGHT_ONLY="$preflight_only" \
  STEP="$STEP" \
  SEED="$SEED" \
  PROJECT_ROOT="$PROJECT_ROOT" \
  CODE_ROOT="$CODE_ROOT" \
  R3_RUN_DIR="$R3_RUN_DIR" \
  R5_RUN_DIR="$R5_RUN_DIR" \
  RECORD_ROOT="$RECORD_ROOT" \
  EVAL_ROOT="$EVAL_ROOT" \
  VALIDATION_JSONL="$VALIDATION_JSONL" \
  PYTHON="$PYTHON" \
  ASR_PYTHON="$ASR_PYTHON" \
  SPEAKER_SIM_ROOT="$SPEAKER_SIM_ROOT" \
  SPEECHBRAIN_ECAPA_MODEL_SOURCE="$SPEECHBRAIN_ECAPA_MODEL_SOURCE" \
  MIN_CHECKPOINT_AGE_SEC="$MIN_CHECKPOINT_AGE_SEC" \
  LOCAL_FROZEN_RUNNER="$FROZEN_RUNNER" \
  LOCAL_ENGINE_SOURCE="$FROZEN_ENGINE" \
  LOCAL_INPUTS_MANIFEST="$INPUTS_MANIFEST" \
  LOCAL_RUN_LOCK="$RUN_LOCK" \
  bash "$FROZEN_ENGINE"

if [ "$ACTION" = "preflight" ]; then
  echo "[batch44-full320-local] preflight complete; no inference/scoring was started"
else
  echo "[batch44-full320-local] complete: $RECORD_ROOT/COMPLETED.json"
fi

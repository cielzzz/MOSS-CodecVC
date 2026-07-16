#!/usr/bin/env bash
# Batch-48 wrapper: four bug fixes + final ver3.1 DDLFM no_text probe.
# The underlying submitter keeps the immutable MTTS-3-2-0715 contract and all
# safety checks; this wrapper only supplies a new branch/tag, job identity and
# output roots so Batch-47 artifacts can never be overwritten.
set -Eeuo pipefail

ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
export REQUIRED_BRANCH="feat/ver3_1_batch48_final"
export REQUIRED_READY_TAG="ver3_1_batch48_bugs_fixed"
export BATCH_ID="codecVC-ver3-1-batch48-ddlfm-no-text-3k-probe-20260717"
export JOB_NAME="$BATCH_ID"
export OUTPUT_ROOT="$ROOT/outputs/ver3_1_batch48_ddlfm_no_text_probe_20260717"
export TINY_GATE_REPORT="$ROOT/testset/outputs/ver3_1_batch48_endpoint_gate_20260717_3k/report.json"
export ALLOW_FAILED_TINY_GATE="${ALLOW_FAILED_TINY_GATE:-0}"
export LOCAL_QUICK20_STEPS="1500,3000"
export LOCAL_FULL_VALIDATION_AT="3000"
export ALLOW_CODECVC_BATCH48_SUBMIT="${ALLOW_CODECVC_BATCH48_SUBMIT:-0}"
# Do not let an old Batch-47 guard accidentally authorize this new arm.
unset ALLOW_CODECVC_BATCH47_SUBMIT
export DRY_RUN="${DRY_RUN:-1}"
unset RECORD_ROOT

exec bash "$ROOT/scripts/ver3_1/submit_batch47_ddlfm_probe_qz.sh" "$@"

#!/usr/bin/env bash
# Run one strict Batch-44 r3 continuation full320 evaluation using only the
# two local RTX 4090 GPUs.  Scientific EFFECTIVE_STEP is mapped to the physical
# continuation checkpoint as local_step=effective_step-10000.

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
CANONICAL_PROJECT_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
PROJECT_ROOT="${PROJECT_ROOT:-$CANONICAL_PROJECT_ROOT}"
CODE_ROOT="${CODE_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch37_eval_20260711_1092820}"
STAMP="20260713"
SEED="${SEED:-1234}"

EFFECTIVE_STEP="${EFFECTIVE_STEP:-20000}"
case "$EFFECTIVE_STEP" in
  20000|24000|26000|28000|30000) ;;
  *) echo "ERROR: EFFECTIVE_STEP must be 20000, 24000, 26000, 28000, or 30000" >&2; exit 2 ;;
esac
CONTINUATION_LOCAL_STEP=$((EFFECTIVE_STEP - 10000))

ACTION="${ACTION:-plan}"
CONFIRM_LOCAL_R3_FULL320="${CONFIRM_LOCAL_R3_FULL320:-0}"
CONFIRM_EFFECTIVE_STEP="${CONFIRM_EFFECTIVE_STEP:-}"
CONFIRM_LOCAL_ONLY="${CONFIRM_LOCAL_ONLY:-}"
MIN_CHECKPOINT_AGE_SEC="${MIN_CHECKPOINT_AGE_SEC:-90}"
MAX_INITIAL_GPU_MEMORY_MIB="${MAX_INITIAL_GPU_MEMORY_MIB:-2048}"

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
ASR_PYTHON="${ASR_PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
VALIDATION_JSONL="$PROJECT_ROOT/testset/validation/seedtts_vc_ver2_3_validation.jsonl"
VALIDATION_SHA256="725ee9d58a7e6066d2a7b79c858cb6ff4dd7292cc167c45dc6b6ebbeaff2fe14"
SPEAKER_SIM_ROOT="${SPEAKER_SIM_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction}"
SPEECHBRAIN_ECAPA_MODEL_SOURCE="${SPEECHBRAIN_ECAPA_MODEL_SOURCE:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/models/speechbrain/spkrec-ecapa-voxceleb}"

CONTINUATION_RUN_DIR="$PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r3_v1_warmstart10k_to30k"
CHECKPOINT="$CONTINUATION_RUN_DIR/step-$CONTINUATION_LOCAL_STEP"
TRAIN_JOB_ID="job-165f3b1d-8c7d-47d8-8882-bdb86ef642ab"
WARM_START_CONTRACT="$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_r3_v1_warmstart10k_to30k_${STAMP}/warm_start_contract.json"
WARM_START_CONTRACT_SHA256="2d686e5e57b70fcaa3db8c8eb2b306003a38599b2c9ac37023979d80b6d9fc34"

RUNNER_SOURCE="$SCRIPT_DIR/004127_run_batch44_r3_warmstart_full320_local.sh"
FINALIZER_SOURCE="$SCRIPT_DIR/batch44_r3_warmstart_full320_finalize.py"
PROVENANCE_HELPER_SOURCE="$SCRIPT_DIR/batch44_r3_warmstart_quick20_completion.py"

RECORD_ROOT="${RECORD_ROOT:-$PROJECT_ROOT/trainset/local_jobs/ver23_batch44_r3_warmstart_full320_step${EFFECTIVE_STEP}_${STAMP}}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/testset/outputs/ver23_batch44_r3_warmstart_full320_${STAMP}/step-${EFFECTIVE_STEP}}"
RUN_ID="ver2_9_5_final_r3_warmstart_effective_step-${EFFECTIVE_STEP}_seedtts320_all_d2d3_seed${SEED}"
OUTPUT_DIR="$EVAL_ROOT/$RUN_ID"
AGG_ROOT="$EVAL_ROOT/aggregate"
DIAG_ROOT="$EVAL_ROOT/diagnostics"

RUN_LOCK="$RECORD_ROOT/.local_full320.lock"
BINDING_JSON="$RECORD_ROOT/CONTINUATION_BINDING.json"
RUNTIME_JSON="$RECORD_ROOT/LOCAL_RUNTIME.json"
FROZEN_RUNNER="$RECORD_ROOT/004127_run_batch44_r3_warmstart_full320_local.frozen.sh"
FROZEN_FINALIZER="$RECORD_ROOT/batch44_r3_warmstart_full320_finalize.frozen.py"
FROZEN_PROVENANCE_HELPER="$RECORD_ROOT/batch44_r3_warmstart_provenance.frozen.py"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

case "$ACTION" in
  plan|preflight|run) ;;
  *) die "ACTION must be plan, preflight, or run; got $ACTION" ;;
esac
for value in "$MIN_CHECKPOINT_AGE_SEC" "$MAX_INITIAL_GPU_MEMORY_MIB"; do
  case "$value" in ''|*[!0-9]*) die "age/memory limits must be non-negative integers" ;; esac
done
[ "$MAX_INITIAL_GPU_MEMORY_MIB" -gt 0 ] || die "MAX_INITIAL_GPU_MEMORY_MIB must be positive"
if [ "$ACTION" = "run" ]; then
  [ "$CONFIRM_LOCAL_R3_FULL320" = "1" ] \
    || die "ACTION=run requires CONFIRM_LOCAL_R3_FULL320=1"
  [ "$CONFIRM_EFFECTIVE_STEP" = "$EFFECTIVE_STEP" ] \
    || die "ACTION=run requires CONFIRM_EFFECTIVE_STEP=$EFFECTIVE_STEP"
  [ "$CONFIRM_LOCAL_ONLY" = "RTX4090x2" ] \
    || die "ACTION=run requires CONFIRM_LOCAL_ONLY=RTX4090x2"
fi
if [ "$PROJECT_ROOT" != "$CANONICAL_PROJECT_ROOT" ]; then
  die "PROJECT_ROOT must be canonical: $CANONICAL_PROJECT_ROOT"
fi

echo "=========================================="
echo "Batch-44 r3 warm-start strict full320 local"
echo "  ACTION=$ACTION"
echo "  EFFECTIVE_STEP=$EFFECTIVE_STEP"
echo "  CONTINUATION_LOCAL_STEP=$CONTINUATION_LOCAL_STEP"
echo "  TRAIN_JOB_ID=$TRAIN_JOB_ID"
echo "  CHECKPOINT=$CHECKPOINT"
echo "  WARM_START_CONTRACT=$WARM_START_CONTRACT"
echo "  WARM_START_CONTRACT_SHA256=$WARM_START_CONTRACT_SHA256"
echo "  BACKEND=local-only"
echo "  GPU_REQUIREMENT=2x NVIDIA GeForce RTX 4090 (indices 0,1)"
echo "  VALIDATION=no_text160+text160"
echo "  RUN_ID=$RUN_ID"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  RECORD_ROOT=$RECORD_ROOT"
echo "=========================================="

if [ "$ACTION" = "plan" ]; then
  echo "[batch44-r3-warmstart-full320] plan complete; no files or GPU work started"
  exit 0
fi

[ -x "$PYTHON" ] || die "missing Python: $PYTHON"
[ -x "$ASR_PYTHON" ] || die "missing ASR Python: $ASR_PYTHON"
[ -s "$VALIDATION_JSONL" ] || die "missing validation JSONL"
[ -d "$CODE_ROOT" ] || die "missing evaluation code root: $CODE_ROOT"
[ -d "$SPEAKER_SIM_ROOT" ] || die "missing WavLM scorer root"
[ -d "$SPEECHBRAIN_ECAPA_MODEL_SOURCE" ] || die "missing SpeechBrain ECAPA model"
for path in "$RUNNER_SOURCE" "$FINALIZER_SOURCE" "$PROVENANCE_HELPER_SOURCE"; do
  [ -s "$path" ] || die "missing runner dependency: $path"
done
bash -n "$RUNNER_SOURCE"
"$PYTHON" -m py_compile "$FINALIZER_SOURCE" "$PROVENANCE_HELPER_SOURCE"

actual_validation_sha=$(sha256sum "$VALIDATION_JSONL" | awk '{print $1}')
[ "$actual_validation_sha" = "$VALIDATION_SHA256" ] \
  || die "validation SHA256 drift: $actual_validation_sha"
actual_contract_sha=$(sha256sum "$WARM_START_CONTRACT" | awk '{print $1}')
[ "$actual_contract_sha" = "$WARM_START_CONTRACT_SHA256" ] \
  || die "warm-start contract SHA256 drift: $actual_contract_sha"

"$PYTHON" "$FINALIZER_SOURCE" audit-binding \
  --provenance-helper "$PROVENANCE_HELPER_SOURCE" \
  --project-root "$PROJECT_ROOT" \
  --effective-step "$EFFECTIVE_STEP" \
  --checkpoint "$CHECKPOINT" \
  --train-job-id "$TRAIN_JOB_ID" \
  --warm-start-contract "$WARM_START_CONTRACT" \
  --warm-start-contract-sha256 "$WARM_START_CONTRACT_SHA256" \
  --min-checkpoint-age-sec "$MIN_CHECKPOINT_AGE_SEC"

"$PYTHON" - "$CODE_ROOT" <<'PY'
from __future__ import annotations
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
expected = {
    "scripts/004039_run_seedtts_validation_eval.sh": "94ee38a950691ddd22e9487c82821247447dc7ecf20813e94852c56687c727b4",
    "scripts/004044_run_seedtts_validation_infer_persistent.py": "22045797d68d54bc2b72c64773c43464e4164b19b3a29d97537149e15594fa1d",
    "scripts/004048_summarize_seedtts_ablation_metrics.py": "e1856c1a503a2101480323acaa9b0d231a6b28971377d47664f3fae02b1d7ca4",
    "scripts/004056_summarize_seedtts_ref_content_similarity.py": "42df1d42934bf3283975eda2bef773a53cafe2a75e4518432664f9373321c4a4",
    "scripts/004063_analyze_seedtts320_diagnostics.py": "ac73c3da45f94b133f334c3bf22e91511fe0e04adbb5bba45663feed3f4721cc",
    "moss_codecvc/models/moss_codecvc_wrapper.py": "1d32527ec29fada353dc70b88a11cff972da901c5830dfeafb3bcf9f067d3ae3",
}
errors = []
for relative, wanted in expected.items():
    path = root / relative
    if not path.is_file():
        errors.append(f"missing {path}")
        continue
    with path.open("rb") as handle:
        got = hashlib.file_digest(handle, "sha256").hexdigest()
    if got != wanted:
        errors.append(f"{relative}: {got} != {wanted}")
if errors:
    raise SystemExit("evaluation snapshot drift:\n- " + "\n- ".join(errors))
print(f"[batch44-r3-warmstart-full320] code snapshot PASS files={len(expected)}")
PY

case "$(hostname)" in
  xyzhang-dev--*) ;;
  *) die "full320 is restricted to the local xyzhang-dev host" ;;
esac
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi unavailable"
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l | tr -d ' ')
[ "$gpu_count" = "2" ] || die "local full320 requires exactly two visible GPUs; got $gpu_count"
gpu_indices=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | paste -sd, -)
[ "$gpu_indices" = "0,1" ] || die "local full320 requires GPU indices 0,1; got $gpu_indices"
gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)
[ "$gpu_names" = "NVIDIA GeForce RTX 4090" ] \
  || die "local full320 requires RTX 4090; got $gpu_names"
while IFS= read -r used; do
  [ "$used" -le "$MAX_INITIAL_GPU_MEMORY_MIB" ] \
    || die "local GPU memory is in use: ${used}MiB > ${MAX_INITIAL_GPU_MEMORY_MIB}MiB"
done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

if [ "$ACTION" = "preflight" ]; then
  echo "[batch44-r3-warmstart-full320] preflight PASS; no output directories or GPU work started"
  exit 0
fi

[ "$RUNNER_SOURCE" = "$PROJECT_ROOT/scripts/004127_run_batch44_r3_warmstart_full320_local.sh" ] \
  || die "live runner path drift"
[ "$FINALIZER_SOURCE" = "$PROJECT_ROOT/scripts/batch44_r3_warmstart_full320_finalize.py" ] \
  || die "live finalizer path drift"
[ "$PROVENANCE_HELPER_SOURCE" = "$PROJECT_ROOT/scripts/batch44_r3_warmstart_quick20_completion.py" ] \
  || die "live provenance helper path drift"
[ "$CHECKPOINT" = "$CONTINUATION_RUN_DIR/step-$CONTINUATION_LOCAL_STEP" ] \
  || die "live checkpoint path drift"
[ "$RECORD_ROOT" = "$PROJECT_ROOT/trainset/local_jobs/ver23_batch44_r3_warmstart_full320_step${EFFECTIVE_STEP}_${STAMP}" ] \
  || die "live record root drift"
[ "$EVAL_ROOT" = "$PROJECT_ROOT/testset/outputs/ver23_batch44_r3_warmstart_full320_${STAMP}/step-${EFFECTIVE_STEP}" ] \
  || die "live eval root drift"

if [ -e "$RECORD_ROOT" ] || [ -L "$RECORD_ROOT" ]; then
  die "record root already exists; refusing to overwrite: $RECORD_ROOT"
fi
if [ -e "$EVAL_ROOT" ] || [ -L "$EVAL_ROOT" ]; then
  die "eval root already exists; refusing to overwrite: $EVAL_ROOT"
fi
mkdir -p "$RECORD_ROOT" "$OUTPUT_DIR" "$AGG_ROOT" "$DIAG_ROOT"
mkdir "$RUN_LOCK" || die "failed to acquire persistent local full320 lock"
install -m 0555 "$RUNNER_SOURCE" "$FROZEN_RUNNER"
install -m 0444 "$FINALIZER_SOURCE" "$FROZEN_FINALIZER"
install -m 0444 "$PROVENANCE_HELPER_SOURCE" "$FROZEN_PROVENANCE_HELPER"

"$PYTHON" "$FROZEN_FINALIZER" audit-binding \
  --provenance-helper "$FROZEN_PROVENANCE_HELPER" \
  --project-root "$PROJECT_ROOT" \
  --effective-step "$EFFECTIVE_STEP" \
  --checkpoint "$CHECKPOINT" \
  --train-job-id "$TRAIN_JOB_ID" \
  --warm-start-contract "$WARM_START_CONTRACT" \
  --warm-start-contract-sha256 "$WARM_START_CONTRACT_SHA256" \
  --min-checkpoint-age-sec "$MIN_CHECKPOINT_AGE_SEC" \
  --output "$BINDING_JSON"

"$PYTHON" "$FROZEN_FINALIZER" capture-runtime \
  --output "$RUNTIME_JSON" \
  --binding "$BINDING_JSON" \
  --runner "$FROZEN_RUNNER" \
  --finalizer "$FROZEN_FINALIZER" \
  --provenance-helper "$FROZEN_PROVENANCE_HELPER" \
  --effective-step "$EFFECTIVE_STEP" \
  --max-initial-memory-mib "$MAX_INITIAL_GPU_MEMORY_MIB"

exec >>"$RECORD_ROOT/run.local.log" 2>&1
echo "[batch44-r3-warmstart-full320] start=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) effective=$EFFECTIVE_STEP local=$CONTINUATION_LOCAL_STEP"
nvidia-smi

CUDA_VISIBLE_DEVICES=0,1 \
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
REF_AUDIO_CFG_SCALE=1.0 \
REF_PROMPT_CODEC_PERMUTATION=0 \
PYTHON="$PYTHON" \
ASR_PYTHON="$ASR_PYTHON" \
VALIDATION_JSONL="$VALIDATION_JSONL" \
MODEL_PATH="$CHECKPOINT" \
RUN_ID="$RUN_ID" \
RUN_LABEL="Batch-44 r3 warm-start effective-$EFFECTIVE_STEP local-$CONTINUATION_LOCAL_STEP full320" \
OUTPUT_DIR="$OUTPUT_DIR" \
MODE=all \
MAX_CASES=0 \
PER_MODE=0 \
PER_CELL=0 \
DECODING_PROFILE=default \
PERSISTENT_INFER=1 \
INFER_SHARD_START_DELAY_SEC=3 \
OVERWRITE_INFER=1 \
RESET_MANIFESTS=1 \
RUN_ASR=1 \
RUN_SUMMARY=1 \
BUILD_PAGE=0 \
GPU_COUNT=2 \
NUM_SHARDS=2 \
ASR_NUM_SHARDS=2 \
SEED="$SEED" \
bash "$CODE_ROOT/scripts/004039_run_seedtts_validation_eval.sh"

"$PYTHON" "$CODE_ROOT/scripts/004056_summarize_seedtts_ref_content_similarity.py" \
  --asr-jsonl "$OUTPUT_DIR/${RUN_ID}.asr_eval.jsonl" \
  --output-json "$OUTPUT_DIR/${RUN_ID}.ref_content_similarity_summary.json" \
  --output-md "$OUTPUT_DIR/${RUN_ID}.ref_content_similarity_summary.md"

CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 \
"$PYTHON" "$CODE_ROOT/scripts/004048_summarize_seedtts_ablation_metrics.py" \
  --validation-jsonl "$VALIDATION_JSONL" \
  --run "$RUN_ID=$OUTPUT_DIR" \
  --output-csv "$AGG_ROOT/dual_encoder_cases.csv" \
  --summary-json "$AGG_ROOT/dual_encoder_summary.json" \
  --summary-md "$AGG_ROOT/dual_encoder_summary.md" \
  --speaker-device cuda:0 \
  --speaker-sim-root "$SPEAKER_SIM_ROOT" \
  --extra-speaker-encoder speechbrain_ecapa \
  --extra-speaker-device cuda:1 \
  --speechbrain-ecapa-model-source "$SPEECHBRAIN_ECAPA_MODEL_SOURCE"

"$PYTHON" "$CODE_ROOT/scripts/004063_analyze_seedtts320_diagnostics.py" \
  --validation-jsonl "$VALIDATION_JSONL" \
  --sim-cases-csv "$AGG_ROOT/dual_encoder_cases.csv" \
  --run "$RUN_ID=$OUTPUT_DIR" \
  --output-dir "$DIAG_ROOT" \
  --prefix "$RUN_ID"

"$PYTHON" "$FROZEN_FINALIZER" finalize \
  --project-root "$PROJECT_ROOT" \
  --record-root "$RECORD_ROOT" \
  --eval-root "$EVAL_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint "$CHECKPOINT" \
  --warm-start-contract "$WARM_START_CONTRACT" \
  --validation-jsonl "$VALIDATION_JSONL" \
  --code-root "$CODE_ROOT" \
  --binding "$BINDING_JSON" \
  --runtime "$RUNTIME_JSON" \
  --runner "$FROZEN_RUNNER" \
  --finalizer "$FROZEN_FINALIZER" \
  --provenance-helper "$FROZEN_PROVENANCE_HELPER" \
  --effective-step "$EFFECTIVE_STEP" \
  --train-job-id "$TRAIN_JOB_ID" \
  --warm-start-contract-sha256 "$WARM_START_CONTRACT_SHA256" \
  --validation-sha256 "$VALIDATION_SHA256" \
  --run-id "$RUN_ID"

rm -rf "$RUN_LOCK"
echo "[batch44-r3-warmstart-full320] complete=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[batch44-r3-warmstart-full320] metrics=$AGG_ROOT/metrics.md"
echo "[batch44-r3-warmstart-full320] unified_input=$AGG_ROOT/unified_eval_input.jsonl"
echo "[batch44-r3-warmstart-full320] completion=$RECORD_ROOT/COMPLETED.json"

#!/usr/bin/env bash
# Evaluate one Batch-44 early checkpoint (r3/r5 x 4k/6k/8k) on the fixed
# SeedTTS-derived 320 set using only the two local RTX 4090 GPUs.

set -euo pipefail

CANONICAL_PROJECT_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
PROJECT_ROOT="${PROJECT_ROOT:-$CANONICAL_PROJECT_ROOT}"
CODE_ROOT="${CODE_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch37_eval_20260711_1092820}"
ARM="${ARM:-r5}"
STEP="${STEP:-6000}"
ACTION="${ACTION:-plan}"
CONFIRM_LOCAL_EARLY_FULL320="${CONFIRM_LOCAL_EARLY_FULL320:-0}"
STAMP="20260713"
SEED="${SEED:-1234}"

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
ASR_PYTHON="${ASR_PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
VALIDATION_JSONL="$PROJECT_ROOT/testset/validation/seedtts_vc_ver2_3_validation.jsonl"
VALIDATION_SHA256="725ee9d58a7e6066d2a7b79c858cb6ff4dd7292cc167c45dc6b6ebbeaff2fe14"
SPEAKER_SIM_ROOT="${SPEAKER_SIM_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction}"
SPEECHBRAIN_ECAPA_MODEL_SOURCE="${SPEECHBRAIN_ECAPA_MODEL_SOURCE:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/models/speechbrain/spkrec-ecapa-voxceleb}"

R3_RUN_DIR="$PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r3_v1_30k"
R5_RUN_DIR="$PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r5_v1_30k"
R3_TRAIN_JOB_ID="job-2b91d332-d500-4279-84f9-0a6a81a376aa"
R5_TRAIN_JOB_ID="job-b8eb2f1f-a3eb-483b-a289-b4cce281525c"

RECORD_ROOT="${RECORD_ROOT:-$PROJECT_ROOT/trainset/local_jobs/ver23_batch44_early_best_full320_${ARM}_step${STEP}_${STAMP}}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/testset/outputs/ver23_batch44_early_best_full320_${STAMP}/${ARM}_step-${STEP}}"
RUN_ID="ver2_9_5_final_${ARM}_step-${STEP}_seedtts320_all_d2d3_seed${SEED}"
OUTPUT_DIR="$EVAL_ROOT/$RUN_ID"
AGG_ROOT="$EVAL_ROOT/aggregate"
DIAG_ROOT="$EVAL_ROOT/diagnostics"
COMPLETION_JSON="$RECORD_ROOT/COMPLETED.json"
COMPLETE_MARKER="$RECORD_ROOT/complete.marker"
RUN_LOCK="$RECORD_ROOT/.local_run.lock"
RUNTIME_JSON="$RECORD_ROOT/LOCAL_RUNTIME.json"
FROZEN_RUNNER="$RECORD_ROOT/004120_run_batch44_early_best_full320_local.frozen.sh"
FINALIZER_SOURCE="$PROJECT_ROOT/scripts/batch44_early_full320_finalize.py"
FROZEN_FINALIZER="$RECORD_ROOT/batch44_early_full320_finalize.frozen.py"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

case "$ARM" in
  r3)
    RUN_DIR="$R3_RUN_DIR"
    TRAIN_JOB_ID="$R3_TRAIN_JOB_ID"
    ;;
  r5)
    RUN_DIR="$R5_RUN_DIR"
    TRAIN_JOB_ID="$R5_TRAIN_JOB_ID"
    ;;
  *) die "ARM must be r3 or r5; got $ARM" ;;
esac
case "$STEP" in
  4000|6000|8000) ;;
  *) die "STEP must be 4000, 6000, or 8000; got $STEP" ;;
esac
case "$ACTION" in
  plan|preflight|run) ;;
  *) die "ACTION must be plan, preflight, or run; got $ACTION" ;;
esac
case "$CONFIRM_LOCAL_EARLY_FULL320" in
  0|1) ;;
  *) die "CONFIRM_LOCAL_EARLY_FULL320 must be 0 or 1" ;;
esac
if [ "$PROJECT_ROOT" != "$CANONICAL_PROJECT_ROOT" ]; then
  die "PROJECT_ROOT must be canonical: $CANONICAL_PROJECT_ROOT"
fi
if [ "$ACTION" = "run" ] && [ "$CONFIRM_LOCAL_EARLY_FULL320" != "1" ]; then
  die "ACTION=run requires CONFIRM_LOCAL_EARLY_FULL320=1"
fi

CHECKPOINT="$RUN_DIR/step-$STEP"

echo "=========================================="
echo "Batch-44 early-best full320 local"
echo "  ACTION=$ACTION"
echo "  CANDIDATE=$ARM step-$STEP"
echo "  TRAIN_JOB_ID=$TRAIN_JOB_ID"
echo "  CHECKPOINT=$CHECKPOINT"
echo "  BACKEND=local"
echo "  GPU_REQUIREMENT=2x NVIDIA GeForce RTX 4090 (indices 0,1)"
echo "  VALIDATION_JSONL=$VALIDATION_JSONL"
echo "  RUN_ID=$RUN_ID"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  RECORD_ROOT=$RECORD_ROOT"
echo "=========================================="

if [ "$ACTION" = "plan" ]; then
  echo "[batch44-early-full320] plan complete; no files or GPU work started"
  exit 0
fi

[ -x "$PYTHON" ] || die "missing Python: $PYTHON"
[ -x "$ASR_PYTHON" ] || die "missing ASR Python: $ASR_PYTHON"
[ -s "$VALIDATION_JSONL" ] || die "missing validation JSONL"
[ -d "$CODE_ROOT" ] || die "missing evaluation code root: $CODE_ROOT"
[ -d "$SPEAKER_SIM_ROOT" ] || die "missing WavLM scorer root"
[ -d "$SPEECHBRAIN_ECAPA_MODEL_SOURCE" ] || die "missing SpeechBrain ECAPA model"
[ -s "$FINALIZER_SOURCE" ] || die "missing finalizer: $FINALIZER_SOURCE"

actual_validation_sha=$(sha256sum "$VALIDATION_JSONL" | awk '{print $1}')
[ "$actual_validation_sha" = "$VALIDATION_SHA256" ] \
  || die "validation SHA256 drift: $actual_validation_sha"

required_checkpoint_files=(
  adapter_model.safetensors
  adapter_config.json
  README.md
  timbre_memory_adapter.pt
  timbre_memory_config.json
)
for name in "${required_checkpoint_files[@]}"; do
  [ -s "$CHECKPOINT/$name" ] || die "missing checkpoint artifact: $CHECKPOINT/$name"
done
"$PYTHON" -c 'import json,sys; [json.load(open(p, encoding="utf-8")) for p in sys.argv[1:]]' \
  "$CHECKPOINT/adapter_config.json" "$CHECKPOINT/timbre_memory_config.json"

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
print(f"[batch44-early-full320] code snapshot PASS files={len(expected)}")
PY

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi unavailable"
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l | tr -d ' ')
[ "$gpu_count" = "2" ] || die "local full320 requires exactly two visible GPUs; got $gpu_count"
gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)
[ "$gpu_names" = "NVIDIA GeForce RTX 4090" ] || die "local full320 requires RTX 4090; got $gpu_names"

if [ "$ACTION" = "preflight" ]; then
  echo "[batch44-early-full320] preflight PASS; no output directories or GPU work started"
  exit 0
fi

if [ -e "$RECORD_ROOT" ] || [ -L "$RECORD_ROOT" ]; then
  die "record root already exists; refusing to overwrite: $RECORD_ROOT"
fi
if [ -e "$EVAL_ROOT" ] || [ -L "$EVAL_ROOT" ]; then
  die "eval root already exists; refusing to overwrite: $EVAL_ROOT"
fi
mkdir -p "$RECORD_ROOT" "$OUTPUT_DIR" "$AGG_ROOT" "$DIAG_ROOT"
mkdir "$RUN_LOCK" || die "failed to acquire local run lock"
cleanup() {
  rmdir "$RUN_LOCK" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cp --reflink=auto "$PROJECT_ROOT/scripts/004120_run_batch44_early_best_full320_local.sh" "$FROZEN_RUNNER"
cp --reflink=auto "$FINALIZER_SOURCE" "$FROZEN_FINALIZER"

"$PYTHON" - "$RUNTIME_JSON" <<'PY'
from __future__ import annotations
import json
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

rows = []
output = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total", "--format=csv,noheader,nounits"],
    text=True,
)
for line in output.splitlines():
    index, name, uuid, memory = [piece.strip() for piece in line.split(",", 3)]
    rows.append({"index": int(index), "name": name, "uuid": uuid, "memory_total_mib": float(memory)})
payload = {
    "schema": "moss_codecvc.batch44_early_best_full320_runtime.v1",
    "status": "started",
    "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "hostname": socket.gethostname(),
    "backend": "local",
    "gpu_count": len(rows),
    "gpu_indices": [row["index"] for row in rows],
    "gpu_model": rows[0]["name"] if rows else "",
    "gpus": rows,
    "scheduling": "one run, two modulo shards on GPU indices 0 and 1",
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

exec >>"$RECORD_ROOT/run.local.log" 2>&1
echo "[batch44-early-full320] start=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
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
RUN_LABEL="Batch-44 $ARM step-$STEP early-best full320" \
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

"$PYTHON" "$FROZEN_FINALIZER" \
  --project-root "$PROJECT_ROOT" \
  --record-root "$RECORD_ROOT" \
  --eval-root "$EVAL_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --run-id "$RUN_ID" \
  --arm "$ARM" \
  --step "$STEP" \
  --checkpoint "$CHECKPOINT" \
  --validation-jsonl "$VALIDATION_JSONL" \
  --code-root "$CODE_ROOT" \
  --train-job-id "$TRAIN_JOB_ID" \
  --runtime-json "$RUNTIME_JSON" \
  --runner "$FROZEN_RUNNER"

echo "[batch44-early-full320] complete=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[batch44-early-full320] metrics=$AGG_ROOT/metrics.md"
echo "[batch44-early-full320] completion=$COMPLETION_JSON"

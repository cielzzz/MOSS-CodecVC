#!/usr/bin/env bash
# Batch-41: evaluate the repeat=3 B2-mixed step-3000 pilot on the complete
# SeedTTS-derived 320-case benchmark and emit the machine-readable gate used
# by 002053.  The scientific decoding protocol deliberately uses two shards,
# matching the Batch-33 / Batch-34+36 final comparison rather than changing
# the stochastic stream to eight shards.

set -euo pipefail

SELF_PATH=$(readlink -f "$0")
PROJECT_ROOT="${PROJECT_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
QZCLI="${QZCLI:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/qzcli_with_deps.sh}"
QZCLI_HOME="${QZCLI_HOME:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/.codex/qzcli_home}"

WORKSPACE="${WORKSPACE:-ws-8207e9e2-e733-4eec-a475-cfa1c36480ba}"
PROJECT="${PROJECT:-project-c67c548f-f02c-453b-ba5b-8745db6886e7}"
ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122" # MTTS-3-2-0715
ALLOWED_SPEC="67b10bc6-78b0-41a3-aaf4-358eeeb99009"
COMPUTE_GROUP="${COMPUTE_GROUP:-$ALLOWED_COMPUTE_GROUP}"
SPEC="${SPEC:-$ALLOWED_SPEC}"
QZCLI_GPU_TYPE_OVERRIDE="${QZCLI_GPU_TYPE_OVERRIDE:-NVIDIA_H200_SXM_141G}"
IMAGE="${IMAGE:-docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420}"
IMAGE_TYPE="${IMAGE_TYPE:-SOURCE_PRIVATE}"
FRAMEWORK="${FRAMEWORK:-pytorch}"
INSTANCES="${INSTANCES:-1}"
SHM_GI="${SHM_GI:-1200}"
PRIORITY="${PRIORITY:-10}"

DRY_RUN="${DRY_RUN:-1}"
FORCE="${FORCE:-0}"
ENTRYPOINT="${BATCH41_PILOT_FINAL320_ENTRYPOINT:-0}"
KEEPALIVE_UNUSED_GPUS="${KEEPALIVE_UNUSED_GPUS:-1}"
SEED="${SEED:-1234}"
EVAL_STEP="${EVAL_STEP:-3000}"
TEXT_REPEAT="${TEXT_REPEAT:-3}"

PILOT_JOB_ID="job-04c05174-9a20-4074-add4-2655293452ed"
TRAIN_RUN_NAME="ver23_content_side_batch41_b2_mixed_3k_probe_20260711"
CANONICAL_MODEL_PATH="$PROJECT_ROOT/outputs/lora_runs/$TRAIN_RUN_NAME/step-$EVAL_STEP"
MODEL_PATH="${MODEL_PATH:-$CANONICAL_MODEL_PATH}"

# This audited snapshot contains the text-row BNF inference bypass and the
# per-shard Hugging Face dynamic-module-cache isolation used by final320.
CODE_ROOT="${CODE_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch3436_eval_20260711_1092820}"
PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
ASR_PYTHON="${ASR_PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
CANONICAL_VALIDATION_JSONL="$PROJECT_ROOT/testset/validation/seedtts_vc_ver2_3_validation.jsonl"
CANONICAL_VALIDATION_SHA256="725ee9d58a7e6066d2a7b79c858cb6ff4dd7292cc167c45dc6b6ebbeaff2fe14"
VALIDATION_JSONL="${VALIDATION_JSONL:-$CANONICAL_VALIDATION_JSONL}"
SPEECHBRAIN_ECAPA_MODEL_SOURCE="${SPEECHBRAIN_ECAPA_MODEL_SOURCE:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/models/speechbrain/spkrec-ecapa-voxceleb}"

RUN_ID="ver23_batch41_b2_mixed_r3_step-${EVAL_STEP}_seedtts320_d2d3_seed${SEED}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/testset/outputs/ver23_batch41_b2_mixed_3k_probe_20260711}"
OUTPUT_DIR="$EVAL_ROOT/$RUN_ID"
RECORD_ROOT="${RECORD_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/ver23_batch41_b2_mixed_r3_final320_20260711}"
JOB_NAME="${JOB_NAME:-ver23_batch41_b2mixed_r3_final320}"
FROZEN_DRIVER="$RECORD_ROOT/004086_batch41_pilot_final320.frozen.sh"
FROZEN_GATE_BUILDER="$RECORD_ROOT/004088_batch41_pilot_gate.frozen.py"
GATE_BUILDER="${GATE_BUILDER:-$PROJECT_ROOT/scripts/004088_build_batch41_pilot_gate.py}"
SUBMISSION_LOCK="$RECORD_ROOT/.live_submission_lock"
SUBMISSION_LOCK_HELD=0
PILOT_GATE_JSON="$EVAL_ROOT/pilot_gate.json"
DUAL_CASES="$EVAL_ROOT/${RUN_ID}.dual_encoder_cases.csv"
DUAL_SUMMARY_JSON="$EVAL_ROOT/${RUN_ID}.dual_encoder_summary.json"
DUAL_SUMMARY_MD="$EVAL_ROOT/${RUN_ID}.dual_encoder_summary.md"
DIAGNOSTICS_ROOT="$EVAL_ROOT/diagnostics"

case "$DRY_RUN:$FORCE:$ENTRYPOINT:$KEEPALIVE_UNUSED_GPUS" in
  [01]:[01]:[01]:[01]) ;;
  *)
    echo "ERROR: DRY_RUN, FORCE, BATCH41_PILOT_FINAL320_ENTRYPOINT, and KEEPALIVE_UNUSED_GPUS must be 0 or 1" >&2
    exit 2
    ;;
esac
if [ "$COMPUTE_GROUP" != "$ALLOWED_COMPUTE_GROUP" ]; then
  echo "ERROR: only MTTS-3-2-0715 is allowed; got $COMPUTE_GROUP" >&2
  exit 2
fi
if [ "$SPEC" != "$ALLOWED_SPEC" ] || [ "$INSTANCES" != "1" ]; then
  echo "ERROR: Batch-41 final320 requires spec=$ALLOWED_SPEC and instances=1" >&2
  exit 2
fi
if [ "$QZCLI_GPU_TYPE_OVERRIDE" != "NVIDIA_H200_SXM_141G" ]; then
  echo "ERROR: Batch-41 final320 requires NVIDIA_H200_SXM_141G" >&2
  exit 2
fi
if [ "$EVAL_STEP" != "3000" ] || [ "$TEXT_REPEAT" != "3" ]; then
  echo "ERROR: this gate is registered only for step=3000 and text_repeat=3" >&2
  exit 2
fi
if [ "$DRY_RUN" = "0" ] && [ "$(readlink -f "$MODEL_PATH")" != "$(readlink -m "$CANONICAL_MODEL_PATH")" ]; then
  echo "ERROR: live evaluation cannot override canonical pilot MODEL_PATH=$CANONICAL_MODEL_PATH" >&2
  exit 2
fi

validate_checkpoint() {
  local required=(
    adapter_model.safetensors
    adapter_config.json
    README.md
    timbre_memory_adapter.pt
    timbre_memory_config.json
  )
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

validate_validation_manifest() {
  "$PYTHON" - \
    "$VALIDATION_JSONL" \
    "$CANONICAL_VALIDATION_JSONL" \
    "$CANONICAL_VALIDATION_SHA256" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

manifest_path = Path(sys.argv[1])
canonical_path = Path(sys.argv[2])
expected_sha256 = sys.argv[3]

try:
    manifest_realpath = manifest_path.resolve(strict=True)
    canonical_realpath = canonical_path.resolve(strict=True)
except FileNotFoundError as exc:
    raise SystemExit(f"missing validation manifest: {exc.filename}") from exc
if manifest_realpath != canonical_realpath:
    raise SystemExit(
        "Batch-41 final320 must use the canonical validation manifest: "
        f"got={manifest_realpath} expected={canonical_realpath}"
    )

digest = hashlib.sha256()
with manifest_realpath.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
actual_sha256 = digest.hexdigest()
if actual_sha256 != expected_sha256:
    raise SystemExit(
        "canonical validation manifest SHA256 mismatch: "
        f"got={actual_sha256} expected={expected_sha256}"
    )

rows = []
with manifest_realpath.open(encoding="utf-8") as handle:
    for line_no, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{manifest_realpath}:{line_no}: invalid JSONL") from exc
ids = [str(row.get("case_id") or "") for row in rows]
if len(rows) != 320 or len(set(ids)) != 320 or any(not case_id for case_id in ids):
    raise SystemExit(
        "canonical validation manifest must contain 320 unique non-empty case IDs: "
        f"rows={len(rows)} unique={len(set(ids))} blanks={sum(not item for item in ids)}"
    )
cells = (
    "en_src_en_ref_same_gender",
    "en_src_zh_ref_f2m",
    "en_src_zh_ref_m2f",
    "en_src_zh_ref_same_gender",
    "zh_src_en_ref_f2m",
    "zh_src_en_ref_m2f",
    "zh_src_en_ref_same_gender",
    "zh_src_zh_ref_same_gender",
)
expected_scope = Counter((mode, cell) for mode in ("no_text", "text") for cell in cells)
expected_scope = Counter({key: 20 for key in expected_scope})
actual_scope = Counter(
    (str(row.get("mode") or ""), str(row.get("cell") or "")) for row in rows
)
if actual_scope != expected_scope:
    raise SystemExit(f"unexpected canonical validation mode/cell scope: {dict(actual_scope)}")
print(
    "[batch41-final320-input] PASS "
    f"manifest={manifest_realpath} sha256={actual_sha256} rows=320"
)
PY
}

validate_static_inputs() {
  if [ ! -d "$PROJECT_ROOT" ] || [ ! -d "$CODE_ROOT" ]; then
    echo "ERROR: missing PROJECT_ROOT or CODE_ROOT" >&2
    return 1
  fi
  if [ ! -x "$QZCLI" ] || [ ! -x "$PYTHON" ] || [ ! -x "$ASR_PYTHON" ]; then
    echo "ERROR: missing qzcli or Python interpreter" >&2
    return 1
  fi
  if [ ! -s "$GATE_BUILDER" ]; then
    echo "ERROR: missing Batch-41 gate builder: $GATE_BUILDER" >&2
    return 1
  fi
  if [ ! -s "$VALIDATION_JSONL" ] || [ ! -d "$SPEECHBRAIN_ECAPA_MODEL_SOURCE" ]; then
    echo "ERROR: missing validation JSONL or local SpeechBrain ECAPA model" >&2
    return 1
  fi
  validate_validation_manifest
  if ! grep -q 'hf_modules_cache_root' "$CODE_ROOT/scripts/004039_run_seedtts_validation_eval.sh"; then
    echo "ERROR: evaluation snapshot lacks per-shard HF cache isolation: $CODE_ROOT" >&2
    return 1
  fi
  validate_checkpoint
}

release_submission_lock() {
  if [ "$SUBMISSION_LOCK_HELD" = "1" ]; then
    rmdir "$SUBMISSION_LOCK" 2>/dev/null || true
    SUBMISSION_LOCK_HELD=0
  fi
}

KEEPALIVE_PIDS=()

start_gpu_keepalive() {
  KEEPALIVE_PIDS=()
  if [ "$KEEPALIVE_UNUSED_GPUS" != "1" ]; then
    echo "[batch41-final320-keepalive] disabled"
    return 0
  fi
  mkdir -p "$RECORD_ROOT/keepalive_logs"
  local gpu
  for gpu in $(seq 2 7); do
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" - "$gpu" \
      >"$RECORD_ROOT/keepalive_logs/gpu${gpu}.log" 2>&1 <<'PY' &
import signal
import sys
import time

import torch

physical_gpu = sys.argv[1]
running = True

def stop(*_args):
    global running
    running = False

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
device = torch.device("cuda:0")
a = torch.randn((4096, 4096), device=device, dtype=torch.float16)
b = torch.randn((4096, 4096), device=device, dtype=torch.float16)
c = torch.empty_like(a)
print(f"keepalive started physical_gpu={physical_gpu}", flush=True)
with torch.inference_mode():
    while running:
        for _ in range(4):
            torch.mm(a, b, out=c)
        torch.cuda.synchronize()
        time.sleep(2.0)
print(f"keepalive stopped physical_gpu={physical_gpu}", flush=True)
PY
    KEEPALIVE_PIDS+=("$!")
  done
  sleep 1
  local pid
  for pid in "${KEEPALIVE_PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: a GPU keepalive process failed during startup" >&2
      stop_gpu_keepalive
      return 1
    fi
  done
  echo "[batch41-final320-keepalive] started GPUs 2-7 pids=${KEEPALIVE_PIDS[*]}"
}

stop_gpu_keepalive() {
  if [ "${#KEEPALIVE_PIDS[@]}" -eq 0 ]; then
    return 0
  fi
  kill "${KEEPALIVE_PIDS[@]}" 2>/dev/null || true
  local pid
  for pid in "${KEEPALIVE_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  KEEPALIVE_PIDS=()
  echo "[batch41-final320-keepalive] stopped"
}

run_seedtts320() {
  mkdir -p "$OUTPUT_DIR"
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
  REF_SPEAKER_PROMPT_ATTENTION_CAPTURE_FRAMES=0 \
  MOSS_TTS_ATTN_IMPLEMENTATION= \
  FILTER_V2_REAL_NO_TEXT_REF_CONTENT_LEAK=1 \
  PYTHON="$PYTHON" \
  ASR_PYTHON="$ASR_PYTHON" \
  VALIDATION_JSONL="$VALIDATION_JSONL" \
  MODEL_PATH="$MODEL_PATH" \
  RUN_ID="$RUN_ID" \
  RUN_LABEL="Batch-41 B2-mixed repeat=3 step-$EVAL_STEP" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  MODE=all \
  MAX_CASES=0 \
  PER_MODE=0 \
  PER_CELL=0 \
  DECODING_PROFILE=default \
  PERSISTENT_INFER=1 \
  INFER_SHARD_START_DELAY_SEC=0 \
  OVERWRITE_INFER=1 \
  RESET_MANIFESTS=1 \
  RUN_ASR=1 \
  RUN_SUMMARY=1 \
  BUILD_PAGE=0 \
  CONTENT_REFERENCE_MODE=text \
  GPU_COUNT=2 \
  NUM_SHARDS=2 \
  ASR_NUM_SHARDS=2 \
  SEED="$SEED" \
  bash "$CODE_ROOT/scripts/004039_run_seedtts_validation_eval.sh"
}

audit_and_score() {
  "$PYTHON" - "$OUTPUT_DIR" "$RUN_ID" "$VALIDATION_JSONL" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

run_dir = Path(sys.argv[1])
run_id = sys.argv[2]
validation_path = Path(sys.argv[3])
validation_rows = [
    json.loads(line)
    for line in validation_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
validation_ids = [str(row.get("case_id") or "") for row in validation_rows]
if len(validation_rows) != 320 or len(set(validation_ids)) != 320 or any(
    not case_id for case_id in validation_ids
):
    raise SystemExit("canonical validation manifest no longer contains 320 unique case IDs")
validation_by_id = {
    str(row["case_id"]): (str(row.get("mode") or ""), str(row.get("cell") or ""))
    for row in validation_rows
}
manifests = []
paths = sorted(run_dir.glob("manifest.shard*.jsonl"))
if len(paths) != 2:
    raise SystemExit(f"expected two inference manifests, got {len(paths)}")
for path in paths:
    manifests.extend(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
ids = [str(row.get("case_id") or "") for row in manifests]
modes = Counter(str(row.get("mode") or "") for row in manifests)
statuses = Counter(str(row.get("status") or "") for row in manifests)
if len(manifests) != 320 or len(set(ids)) != 320 or any(not case_id for case_id in ids):
    raise SystemExit(f"manifest rows/unique mismatch: {len(manifests)}/{len(set(ids))}")
if set(ids) != set(validation_ids):
    missing = sorted(set(validation_ids) - set(ids))[:5]
    extra = sorted(set(ids) - set(validation_ids))[:5]
    raise SystemExit(f"inference case set differs from canonical 320: missing={missing} extra={extra}")
scope_mismatches = [
    str(row.get("case_id") or "")
    for row in manifests
    if (str(row.get("mode") or ""), str(row.get("cell") or ""))
    != validation_by_id[str(row.get("case_id") or "")]
]
if scope_mismatches:
    raise SystemExit(f"inference mode/cell differs from canonical 320: {scope_mismatches[:5]}")
if modes != Counter({"no_text": 160, "text": 160}):
    raise SystemExit(f"unexpected mode counts: {dict(modes)}")
if statuses != Counter({"ok": 320}):
    raise SystemExit(f"unexpected inference statuses: {dict(statuses)}")
missing_audio = [
    str(row.get("case_id") or "")
    for row in manifests
    if not row.get("output_exists")
    or not Path(str(row.get("output_wav") or "")).is_file()
    or Path(str(row.get("output_wav") or "")).stat().st_size <= 0
]
if missing_audio:
    raise SystemExit(f"missing or empty generated audio for cases: {missing_audio[:5]}")
asr_path = run_dir / f"{run_id}.asr_eval.jsonl"
asr_rows = [
    json.loads(line)
    for line in asr_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
asr_ids = [str(row.get("case_id") or row.get("sample_id") or "") for row in asr_rows]
if (
    len(asr_rows) != 320
    or len(set(asr_ids)) != 320
    or any(not case_id for case_id in asr_ids)
    or set(asr_ids) != set(ids)
):
    raise SystemExit("ASR rows/case set do not match the 320 inference rows")
asr_scope_mismatches = [
    case_id
    for row, case_id in zip(asr_rows, asr_ids)
    if (str(row.get("mode") or ""), str(row.get("cell") or ""))
    != validation_by_id[case_id]
    or str(row.get("run_id") or "") != run_id
]
if asr_scope_mismatches:
    raise SystemExit(f"ASR run/mode/cell provenance mismatch: {asr_scope_mismatches[:5]}")
infer_logs = sorted((run_dir / "logs").glob("infer.shard*.log"))
if len(infer_logs) != 2:
    raise SystemExit(f"expected two inference logs, got {len(infer_logs)}")
bnf_lines = sum(
    path.read_text(encoding="utf-8", errors="replace").count("source semantic memory type=")
    for path in infer_logs
)
if bnf_lines != 160:
    raise SystemExit(
        f"text bypass audit failed: expected BNF extraction only on 160 no_text rows, got {bnf_lines}"
    )
print("[batch41-final320-audit] PASS rows=320 modes=160+160 bnf_extractions=160")
PY

  "$PYTHON" "$CODE_ROOT/scripts/004056_summarize_seedtts_ref_content_similarity.py" \
    --asr-jsonl "$OUTPUT_DIR/${RUN_ID}.asr_eval.jsonl" \
    --output-json "$OUTPUT_DIR/${RUN_ID}.ref_content_similarity_summary.json" \
    --output-md "$OUTPUT_DIR/${RUN_ID}.ref_content_similarity_summary.md"

  CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 \
  "$PYTHON" "$CODE_ROOT/scripts/004048_summarize_seedtts_ablation_metrics.py" \
    --validation-jsonl "$VALIDATION_JSONL" \
    --run "$RUN_ID=$OUTPUT_DIR" \
    --output-csv "$DUAL_CASES" \
    --summary-json "$DUAL_SUMMARY_JSON" \
    --summary-md "$DUAL_SUMMARY_MD" \
    --speaker-device cuda:0 \
    --extra-speaker-encoder speechbrain_ecapa \
    --extra-speaker-device cuda:1 \
    --speechbrain-ecapa-model-source "$SPEECHBRAIN_ECAPA_MODEL_SOURCE"

  mkdir -p "$DIAGNOSTICS_ROOT"
  "$PYTHON" "$CODE_ROOT/scripts/004063_analyze_seedtts320_diagnostics.py" \
    --validation-jsonl "$VALIDATION_JSONL" \
    --sim-cases-csv "$DUAL_CASES" \
    --run "$RUN_ID=$OUTPUT_DIR" \
    --output-dir "$DIAGNOSTICS_ROOT" \
    --prefix "$RUN_ID"
}

build_pilot_gate() {
  "$PYTHON" "$GATE_BUILDER" \
    --run-summary "$OUTPUT_DIR/${RUN_ID}.summary.json" \
    --dual-cases "$DUAL_CASES" \
    --output "$PILOT_GATE_JSON" \
    --pilot-job-id "$PILOT_JOB_ID" \
    --run-id "$RUN_ID" \
    --checkpoint-step "$EVAL_STEP" \
    --text-repeat "$TEXT_REPEAT"
}

run_entrypoint() {
  mkdir -p "$RECORD_ROOT" "$EVAL_ROOT" "$OUTPUT_DIR"
  exec > >(tee -a "$RECORD_ROOT/run.log") 2>&1
  echo "[batch41-final320] start=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
  echo "[batch41-final320] pilot_job=$PILOT_JOB_ID model=$MODEL_PATH"
  echo "[batch41-final320] run_id=$RUN_ID output=$OUTPUT_DIR"
  validate_static_inputs

  local gpu_count
  gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l | tr -d ' ')
  if [ "$gpu_count" != "8" ]; then
    echo "ERROR: expected exactly 8 GPUs in the MTTS whole node; got $gpu_count" >&2
    return 1
  fi
  if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -Evq 'H200'; then
    echo "ERROR: one or more allocated GPUs are not H200" >&2
    return 1
  fi
  nvidia-smi || true

  start_gpu_keepalive
  trap stop_gpu_keepalive EXIT
  trap 'trap - EXIT INT TERM; stop_gpu_keepalive; exit 130' INT
  trap 'trap - EXIT INT TERM; stop_gpu_keepalive; exit 143' TERM
  run_seedtts320
  audit_and_score
  build_pilot_gate
  stop_gpu_keepalive
  trap - EXIT INT TERM
  echo "[batch41-final320] complete=$(date -u +%Y-%m-%dT%H:%M:%SZ) gate=$PILOT_GATE_JSON"
}

if [ "$ENTRYPOINT" = "1" ]; then
  run_entrypoint
  exit 0
fi

validate_static_inputs
mkdir -p "$RECORD_ROOT" "$EVAL_ROOT" "$QZCLI_HOME"
if [ "$DRY_RUN" = "0" ]; then
  if ! mkdir "$SUBMISSION_LOCK" 2>/dev/null; then
    echo "ERROR: another live Batch-41 final320 submission holds $SUBMISSION_LOCK" >&2
    exit 1
  fi
  SUBMISSION_LOCK_HELD=1
  trap release_submission_lock EXIT
  trap 'release_submission_lock; exit 130' INT
  trap 'release_submission_lock; exit 143' TERM
fi
if [ "$FORCE" != "1" ] && {
  [ -s "$PILOT_GATE_JSON" ] || [ -s "$RECORD_ROOT/submitted_jobs.tsv" ];
}; then
  echo "ERROR: existing Batch-41 pilot gate/submission; use FORCE=1 only for an intentional rerun" >&2
  exit 1
fi

cp "$SELF_PATH" "$FROZEN_DRIVER"
chmod +x "$FROZEN_DRIVER"
cp "$GATE_BUILDER" "$FROZEN_GATE_BUILDER"
chmod +x "$FROZEN_GATE_BUILDER"
COMMAND="env BATCH41_PILOT_FINAL320_ENTRYPOINT=1 DRY_RUN=0 FORCE=$FORCE KEEPALIVE_UNUSED_GPUS=$KEEPALIVE_UNUSED_GPUS PROJECT_ROOT=$PROJECT_ROOT CODE_ROOT=$CODE_ROOT MODEL_PATH=$CANONICAL_MODEL_PATH EVAL_ROOT=$EVAL_ROOT RECORD_ROOT=$RECORD_ROOT GATE_BUILDER=$FROZEN_GATE_BUILDER VALIDATION_JSONL=$VALIDATION_JSONL PYTHON=$PYTHON ASR_PYTHON=$ASR_PYTHON bash $FROZEN_DRIVER"
SUBMIT_OUTPUT="$RECORD_ROOT/submit_output.txt"

echo "=========================================="
echo "QZ submit: Batch-41 B2-mixed repeat=3 final320"
echo "  JOB_NAME=$JOB_NAME"
echo "  PILOT_JOB_ID=$PILOT_JOB_ID"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  RUN_ID=$RUN_ID"
echo "  EVAL_ROOT=$EVAL_ROOT"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP (MTTS-3-2-0715 only)"
echo "  SHARDS=2 (Batch-33 comparable), keepalive_gpus=2-7"
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
  echo "[batch41-final320] dry-run passed; no job submitted"
  exit 0
fi

job_id=$(
  printf '%s\n' "$output" |
    grep -Eo 'job-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}' |
    tail -n 1 || true
)
case "$job_id" in
  job-????????-????-????-????-????????????) ;;
  *)
    echo "ERROR: QZ returned success but no valid job ID was parsed" >&2
    exit 1
    ;;
esac
{
  printf 'job_name\tjob_id\tpilot_job_id\tcompute_group\tmodel_path\trecord_root\teval_root\n'
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$JOB_NAME" "$job_id" "$PILOT_JOB_ID" "$COMPUTE_GROUP" "$MODEL_PATH" "$RECORD_ROOT" "$EVAL_ROOT"
} > "$RECORD_ROOT/submitted_jobs.tsv"
echo "[batch41-final320] submitted job_id=$job_id record=$RECORD_ROOT/submitted_jobs.tsv"

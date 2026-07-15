#!/usr/bin/env bash
# Score the prepared Ground Truth front/back-half calibration on the local
# dual-RTX4090 workstation.  This runner is speaker-only: the existing
# same-file Ground Truth pass remains the source of the raw-audio ASR floor.
#
# Safe default (no files, models, or GPUs touched):
#   bash scripts/004131_run_batch42_gt_half_split_local.sh
#
# Read-only preflight:
#   ACTION=preflight bash scripts/004131_run_batch42_gt_half_split_local.sh
#
# A real local pass requires all three confirmations:
#   ACTION=run \
#   CONFIRM_GT_HALF_SPLIT_LOCAL=1 \
#   CONFIRM_GT_HALF_SPLIT_SPEAKER_ONLY=1 \
#   CONFIRM_GT_HALF_SPLIT_NO_ASR=1 \
#     bash scripts/004131_run_batch42_gt_half_split_local.sh

set -euo pipefail

CANONICAL_PROJECT_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
PROJECT_ROOT="${PROJECT_ROOT:-$CANONICAL_PROJECT_ROOT}"
TEST_MODE="${BATCH42_GT_HALF_SPLIT_TEST_MODE:-0}"
ACTION="${ACTION:-plan}"
CONFIRM_LOCAL="${CONFIRM_GT_HALF_SPLIT_LOCAL:-0}"
CONFIRM_SPEAKER_ONLY="${CONFIRM_GT_HALF_SPLIT_SPEAKER_ONLY:-0}"
CONFIRM_NO_ASR="${CONFIRM_GT_HALF_SPLIT_NO_ASR:-0}"
MAX_INITIAL_GPU_MEMORY_MIB="${MAX_INITIAL_GPU_MEMORY_MIB:-2048}"

PREP_ROOT="${PREP_ROOT:-$PROJECT_ROOT/testset/outputs/batch42_ground_truth_half_split_inputs_20260713}"
PREP_AUDIT="${PREP_AUDIT:-$PREP_ROOT/GROUND_TRUTH_HALF_SPLIT_AUDIT.json}"
EN_INPUT="${EN_INPUT:-$PREP_ROOT/ground_truth_half_split.en.input.jsonl}"
ZH_INPUT="${ZH_INPUT:-$PREP_ROOT/ground_truth_half_split.zh.input.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/testset/outputs/batch42_ground_truth_half_split_scored_20260713_local}"
RECORD_ROOT="${RECORD_ROOT:-$PROJECT_ROOT/trainset/local_jobs/batch42_ground_truth_half_split_scored_20260713}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-ground_truth_half_split}"
SYSTEM_ID="ground_truth_half_split"
NUM_SHARDS=2

SCORER_PYTHON="${SCORER_PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$PROJECT_ROOT/scripts/004082_run_unified_vc_eval.py}"
BATCH42_PYTHON_DEPS="${BATCH42_PYTHON_DEPS:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/python_deps/batch42_eval}"
SPEECHBRAIN_PYTHON_DEPS="${SPEECHBRAIN_PYTHON_DEPS:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/python_deps/speechbrain_py312}"
SCORER_BASE_SITE="${SCORER_BASE_SITE:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/lib/python3.12/site-packages}"
SPEAKER_SIM_ROOT="${SPEAKER_SIM_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction}"
WAVLM_CHECKPOINT="${WAVLM_CHECKPOINT:-/inspire/hdd/project/embodied-multimodality/public/kxhuang/vcdata_construction/models/wavlm_large_finetune.pth}"
SEEDTTS_EVAL_ROOT="${SEEDTTS_EVAL_ROOT:-/inspire/hdd/project/embodied-multimodality/public/kxhuang/vcdata_construction/models/seed-tts-eval}"
WAVLM_MODEL_DIR="${WAVLM_MODEL_DIR:-/inspire/hdd/project/embodied-multimodality/public/kxhuang/vcdata_construction/models/wavlm-large}"
ERES2NET_MODEL="${ERES2NET_MODEL:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/batch42/models/modelscope/models/iic--speech_eres2net_sv_zh-cn_16k-common/snapshots/master}"
SPEECHBRAIN_MODEL="${SPEECHBRAIN_MODEL:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/models/speechbrain/spkrec-ecapa-voxceleb}"
HF_HOME="${HF_HOME:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/batch42/models/huggingface}"
MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/batch42/models/modelscope}"
TORCH_HOME="${TORCH_HOME:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/batch42/models/torch}"

RUNNER_SOURCE="$PROJECT_ROOT/scripts/004131_run_batch42_gt_half_split_local.sh"
LOCK_DIR="$RECORD_ROOT/.local_speaker_score.lock"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

case "$TEST_MODE:$CONFIRM_LOCAL:$CONFIRM_SPEAKER_ONLY:$CONFIRM_NO_ASR" in
  [01]:[01]:[01]:[01]) ;;
  *) die "boolean flags must be 0 or 1" ;;
esac
case "$ACTION" in
  plan|preflight|run) ;;
  *) die "ACTION must be plan, preflight, or run" ;;
esac
case "$MAX_INITIAL_GPU_MEMORY_MIB" in
  ''|*[!0-9]*) die "MAX_INITIAL_GPU_MEMORY_MIB must be a non-negative integer" ;;
esac
[ "$MAX_INITIAL_GPU_MEMORY_MIB" -gt 0 ] || die "MAX_INITIAL_GPU_MEMORY_MIB must be positive"
if [ "$PROJECT_ROOT" != "$CANONICAL_PROJECT_ROOT" ] && [ "$TEST_MODE" != "1" ]; then
  die "non-canonical PROJECT_ROOT is test-only"
fi
if [ "$ACTION" = "run" ]; then
  [ "$CONFIRM_LOCAL" = "1" ] || die "ACTION=run requires CONFIRM_GT_HALF_SPLIT_LOCAL=1"
  [ "$CONFIRM_SPEAKER_ONLY" = "1" ] || die "ACTION=run requires CONFIRM_GT_HALF_SPLIT_SPEAKER_ONLY=1"
  [ "$CONFIRM_NO_ASR" = "1" ] || die "ACTION=run requires CONFIRM_GT_HALF_SPLIT_NO_ASR=1"
fi
if [ "$TEST_MODE" = "0" ]; then
  [ "$PREP_ROOT" = "$CANONICAL_PROJECT_ROOT/testset/outputs/batch42_ground_truth_half_split_inputs_20260713" ] \
    || die "production prepared-input root is hard-locked"
  [ "$OUTPUT_ROOT" = "$CANONICAL_PROJECT_ROOT/testset/outputs/batch42_ground_truth_half_split_scored_20260713_local" ] \
    || die "production scorer output root is hard-locked"
  [ "$RECORD_ROOT" = "$CANONICAL_PROJECT_ROOT/trainset/local_jobs/batch42_ground_truth_half_split_scored_20260713" ] \
    || die "production record root is hard-locked"
  [ "$EVAL_SCRIPT" = "$CANONICAL_PROJECT_ROOT/scripts/004082_run_unified_vc_eval.py" ] \
    || die "production evaluator is hard-locked"
fi

echo "=========================================="
echo "Batch-42 Ground Truth half-split local speaker calibration"
echo "  ACTION=$ACTION"
echo "  BACKEND=local dual RTX4090"
echo "  INPUTS=$EN_INPUT,$ZH_INPUT"
echo "  SPEAKER_BACKENDS=wavlm_large_sv,eres2net,speechbrain_ecapa"
echo "  ASR=disabled; reuse existing same-file raw-source ASR floor"
echo "  OUTPUT_ROOT=$OUTPUT_ROOT"
echo "  RECORD_ROOT=$RECORD_ROOT"
echo "=========================================="

if [ "$ACTION" = "plan" ]; then
  echo "[gt-half-split-local] plan complete; no files, models, or GPUs were touched"
  exit 0
fi

[ -x "$SCORER_PYTHON" ] || die "missing scorer Python: $SCORER_PYTHON"
[ -s "$EVAL_SCRIPT" ] || die "missing unified evaluator: $EVAL_SCRIPT"
[ -s "$PREP_AUDIT" ] || die "missing preparation audit: $PREP_AUDIT"
[ -s "$EN_INPUT" ] || die "missing EN half-split input: $EN_INPUT"
[ -s "$ZH_INPUT" ] || die "missing ZH half-split input: $ZH_INPUT"
for path in \
  "$BATCH42_PYTHON_DEPS" \
  "$SPEECHBRAIN_PYTHON_DEPS" \
  "$SCORER_BASE_SITE" \
  "$SPEAKER_SIM_ROOT" \
  "$WAVLM_CHECKPOINT" \
  "$SEEDTTS_EVAL_ROOT" \
  "$WAVLM_MODEL_DIR" \
  "$ERES2NET_MODEL/pretrained_eres2net_aug.ckpt" \
  "$SPEECHBRAIN_MODEL/hyperparams.yaml"; do
  [ -e "$path" ] || die "missing scorer dependency/model asset: $path"
done

INPUT_AUDIT_JSON=$(
  "$SCORER_PYTHON" - "$PREP_AUDIT" "$EN_INPUT" "$ZH_INPUT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

audit_path = Path(sys.argv[1]).resolve()
inputs = {"en": Path(sys.argv[2]).resolve(), "zh": Path(sys.argv[3]).resolve()}
audit = json.loads(audit_path.read_text(encoding="utf-8"))
if audit.get("schema_version") != "moss_codecvc.batch42_gt_half_split_audit.v1":
    raise SystemExit("wrong half-split audit schema")
if audit.get("system_id") != "ground_truth_half_split":
    raise SystemExit("wrong half-split system identity")
contract = audit.get("scoring_contract") or {}
if contract.get("speaker_scorers") != ["wavlm_large_sv", "eres2net", "speechbrain_ecapa"]:
    raise SystemExit("speaker scorer contract drift")
if contract.get("asr_backends") != []:
    raise SystemExit("half-split audit unexpectedly requests content recognition")

result = {"audit": str(audit_path), "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(), "splits": {}}
for language, path in inputs.items():
    split = (audit.get("splits") or {}).get(language) or {}
    registration = split.get("input_jsonl") or {}
    if Path(str(registration.get("path") or "")).resolve() != path:
        raise SystemExit(f"{language}: prepared input path drift")
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if registration.get("sha256") != actual_sha:
        raise SystemExit(f"{language}: prepared input SHA drift")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != int(split.get("kept_rows") or -1) or not rows:
        raise SystemExit(f"{language}: input row count differs from audit")
    if [int(row.get("input_index", -1)) for row in rows] != list(range(len(rows))):
        raise SystemExit(f"{language}: input_index is not contiguous")
    for row in rows:
        if row.get("system_id") != "ground_truth_half_split" or row.get("language") != language:
            raise SystemExit(f"{language}: row identity drift")
        generated = Path(str(row.get("generated_audio") or "")).resolve()
        reference = Path(str(row.get("reference_audio") or "")).resolve()
        source = Path(str(row.get("source_audio") or "")).resolve()
        if generated == reference or not generated.is_file() or not reference.is_file() or not source.is_file():
            raise SystemExit(f"{language}: invalid front/back/full audio binding")
        half = row.get("half_split") or {}
        if half.get("overlap_frames") != 0 or half.get("gap_frames") != 0:
            raise SystemExit(f"{language}: half overlap/gap drift")
        if float(half.get("front_seconds") or 0) < float(half.get("min_half_seconds") or 0):
            raise SystemExit(f"{language}: front duration gate drift")
        if float(half.get("back_seconds") or 0) < float(half.get("min_half_seconds") or 0):
            raise SystemExit(f"{language}: back duration gate drift")
    result["splits"][language] = {
        "rows": len(rows),
        "test_set_id": split.get("test_set_id"),
        "input": str(path),
        "input_sha256": actual_sha,
    }
print(json.dumps(result, sort_keys=True))
PY
)

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used --format=csv,noheader,nounits)
[ "${#GPU_ROWS[@]}" = "2" ] || die "local scorer requires exactly two visible GPUs"
for row in "${GPU_ROWS[@]}"; do
  IFS=',' read -r index name uuid total used <<<"$row"
  name=$(echo "$name" | xargs)
  used=$(echo "$used" | xargs)
  [ "$name" = "NVIDIA GeForce RTX 4090" ] || die "local scorer requires RTX4090; got $name"
  [ "${used%.*}" -le "$MAX_INITIAL_GPU_MEMORY_MIB" ] \
    || die "GPU $index starts with ${used} MiB in use; limit=$MAX_INITIAL_GPU_MEMORY_MIB"
done

PYTHONPATH="$BATCH42_PYTHON_DEPS:$SCORER_BASE_SITE:$SPEECHBRAIN_PYTHON_DEPS${PYTHONPATH:+:$PYTHONPATH}" \
  "$SCORER_PYTHON" - <<'PY'
import importlib.util
required = ("torch", "torchaudio", "modelscope", "funasr", "speechbrain")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing scorer modules: {missing}")
print("[gt-half-split-local] Python dependency audit PASS")
PY

if [ "$ACTION" = "preflight" ]; then
  echo "[gt-half-split-local] preflight PASS; no scoring output was created"
  exit 0
fi

[ ! -e "$OUTPUT_ROOT" ] && [ ! -L "$OUTPUT_ROOT" ] || die "output root already exists: $OUTPUT_ROOT"
[ ! -e "$RECORD_ROOT" ] && [ ! -L "$RECORD_ROOT" ] || die "record root already exists: $RECORD_ROOT"
mkdir -p "$OUTPUT_ROOT" "$RECORD_ROOT"
mkdir "$LOCK_DIR" || die "failed to acquire persistent local lock"
printf '%s\n' "$INPUT_AUDIT_JSON" > "$RECORD_ROOT/input_audit.json"
cp --reflink=auto "$RUNNER_SOURCE" "$RECORD_ROOT/004131_run_batch42_gt_half_split_local.frozen.sh"
cp --reflink=auto "$EVAL_SCRIPT" "$RECORD_ROOT/004082_run_unified_vc_eval.frozen.py"
FROZEN_EVAL="$RECORD_ROOT/004082_run_unified_vc_eval.frozen.py"

nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv \
  > "$RECORD_ROOT/runtime_gpu_inventory.csv"

score_language() {
  local language=$1
  local input=$2
  local test_set_id=$3
  local language_root="$OUTPUT_ROOT/$language"
  local partial_root="$language_root/partials"
  local merged_root="$language_root/merged"
  local log_root="$language_root/logs"
  local run_id="${RUN_ID_PREFIX}_${language}_speaker_only"
  local shard stem log_path
  local -a pids=()
  mkdir -p "$partial_root" "$merged_root" "$log_root"
  for shard in 0 1; do
    stem=$(printf '%s.%s.shard-%05d-of-%05d' "$SYSTEM_ID" "$language" "$shard" "$NUM_SHARDS")
    log_path="$log_root/$stem.log"
    (
      export CUDA_VISIBLE_DEVICES="$shard"
      export PYTHONPATH="$BATCH42_PYTHON_DEPS:$SCORER_BASE_SITE:$SPEECHBRAIN_PYTHON_DEPS:$RECORD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
      export HF_HOME MODELSCOPE_CACHE TORCH_HOME
      export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
      export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
      "$SCORER_PYTHON" "$FROZEN_EVAL" evaluate \
        --input "$input" \
        --output-dir "$partial_root" \
        --output-stem "$stem" \
        --run-id "$run_id" \
        --system-id "$SYSTEM_ID" \
        --test-set-id "$test_set_id" \
        --input-profile official_seedtts_vc \
        --speaker-scorer all \
        --no-reuse-existing \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$shard" \
        --speaker-device cuda:0 \
        --speaker-sim-root "$SPEAKER_SIM_ROOT" \
        --wavlm-implementation seedtts_official \
        --wavlm-checkpoint "$WAVLM_CHECKPOINT" \
        --seedtts-eval-root "$SEEDTTS_EVAL_ROOT" \
        --wavlm-model-dir "$WAVLM_MODEL_DIR" \
        --eres2net-model "$ERES2NET_MODEL" \
        --speechbrain-model "$SPEECHBRAIN_MODEL"
    ) >"$log_path" 2>&1 &
    pids+=("$!")
  done
  for shard in 0 1; do
    if ! wait "${pids[$shard]}"; then
      tail -n 100 "$log_root"/*.log >&2 || true
      die "$language speaker shard $shard failed"
    fi
  done
  "$SCORER_PYTHON" "$FROZEN_EVAL" merge \
    --partial "$partial_root/$SYSTEM_ID.$language.shard-00000-of-00002.unified_eval.jsonl" \
    --partial "$partial_root/$SYSTEM_ID.$language.shard-00001-of-00002.unified_eval.jsonl" \
    --output-dir "$merged_root" \
    --output-stem "$SYSTEM_ID.$language.merged" \
    --run-id "${run_id}_merged"
}

readarray -t SPLIT_META < <(
  "$SCORER_PYTHON" - "$RECORD_ROOT/input_audit.json" <<'PY'
import json, sys
x=json.load(open(sys.argv[1], encoding="utf-8"))
for language in ("en", "zh"):
    row=x["splits"][language]
    print(f"{language}\t{row['input']}\t{row['test_set_id']}\t{row['rows']}")
PY
)
for item in "${SPLIT_META[@]}"; do
  IFS=$'\t' read -r language input test_set_id expected_rows <<<"$item"
  score_language "$language" "$input" "$test_set_id"
done

"$SCORER_PYTHON" - "$RECORD_ROOT" "$OUTPUT_ROOT" "$PREP_AUDIT" "$INPUT_AUDIT_JSON" <<'PY'
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

record = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
prep_audit = Path(sys.argv[3]).resolve()
input_audit = json.loads(sys.argv[4])

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

artifacts = {}
for language in ("en", "zh"):
    expected = int(input_audit["splits"][language]["rows"])
    merged = output / language / "merged" / f"ground_truth_half_split.{language}.merged.unified_eval.jsonl"
    rows = [json.loads(line) for line in merged.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected:
        raise SystemExit(f"{language}: merged rows={len(rows)}, expected={expected}")
    for row in rows:
        if row.get("content_asr") not in ({}, None):
            raise SystemExit(f"{language}/{row.get('case_id')}: content recognition was unexpectedly run")
        for backend in ("wavlm_large_sv", "eres2net", "speechbrain_ecapa"):
            result = (row.get("speaker_similarity") or {}).get(backend) or {}
            if result.get("status") != "ok" or result.get("error"):
                raise SystemExit(f"{language}/{row.get('case_id')}: {backend}={result}")
            for metric in ("sim_ref", "sim_src"):
                value = result.get(metric)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise SystemExit(f"{language}/{row.get('case_id')}: {backend}.{metric}={value!r}")
    artifacts[language] = {
        "merged_jsonl": str(merged),
        "merged_sha256": digest(merged),
        "rows": len(rows),
    }

completion = {
    "schema": "moss_codecvc.batch42_gt_half_split_local_completion.v1",
    "status": "complete",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "backend": "local_dual_rtx4090",
    "system_id": "ground_truth_half_split",
    "speaker_backends": ["wavlm_large_sv", "eres2net", "speechbrain_ecapa"],
    "asr_backends": [],
    "prep_audit": str(prep_audit),
    "prep_audit_sha256": digest(prep_audit),
    "outputs": artifacts,
}
path = record / "COMPLETED.json"
tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
tmp.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, path)
marker = {
    "schema": "moss_codecvc.batch42_gt_half_split_local_marker.v1",
    "status": "complete",
    "completion_sha256": digest(path),
}
(record / "complete.marker").write_text(json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8")
PY

rmdir "$LOCK_DIR"
echo "[gt-half-split-local] complete output=$OUTPUT_ROOT completion=$RECORD_ROOT/COMPLETED.json"

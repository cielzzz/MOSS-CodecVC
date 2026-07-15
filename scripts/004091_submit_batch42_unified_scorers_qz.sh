#!/usr/bin/env bash
# Run the Batch-42 paper-facing unified scorer pass for one baseline system.
#
# This wrapper consumes the successful EN/ZH JSONLs emitted after 004090's
# inference-shard merge.  One MTTS-only 8xH200 job runs EN in eight GPU-local
# workers, merges/audits it, then does the same for ZH.  Every worker sees one
# physical GPU as cuda:0 and evaluates all three speaker scorers plus the
# language-primary ASR backend.
#
# Required dry-run example (never submits by default):
#   SYSTEM_TAG=openvoice_v2 \
#   EN_INPUT=/path/to/openvoice_v2.en.successful.jsonl \
#   ZH_INPUT=/path/to/openvoice_v2.zh.successful.jsonl \
#   OUTPUT_ROOT=/path/to/batch42/scored/openvoice_v2 \
#   bash scripts/004091_submit_batch42_unified_scorers_qz.sh
#
# Actual submission must be an explicit later action:
#   ... DRY_RUN=0 bash scripts/004091_submit_batch42_unified_scorers_qz.sh

set -euo pipefail

SELF_PATH=$(readlink -f "$0")
PROJECT_ROOT="${PROJECT_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
QZCLI="${QZCLI:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/qzcli_with_deps.sh}"
QZCLI_HOME="${QZCLI_HOME:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/.codex/qzcli_home}"

ALLOWED_WORKSPACE="ws-8207e9e2-e733-4eec-a475-cfa1c36480ba" # CI-情境智能
ALLOWED_PROJECT="project-c67c548f-f02c-453b-ba5b-8745db6886e7" # CI-情境智能
WORKSPACE="${WORKSPACE:-$ALLOWED_WORKSPACE}"
PROJECT="${PROJECT:-$ALLOWED_PROJECT}"
ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122" # MTTS-3-2-0715
COMPUTE_GROUP="${COMPUTE_GROUP:-$ALLOWED_COMPUTE_GROUP}"
ALLOWED_SPEC="67b10bc6-78b0-41a3-aaf4-358eeeb99009"
SPEC="${SPEC:-$ALLOWED_SPEC}"
ALLOWED_GPU_TYPE="NVIDIA_H200_SXM_141G"
QZCLI_GPU_TYPE_OVERRIDE="${QZCLI_GPU_TYPE_OVERRIDE:-$ALLOWED_GPU_TYPE}"
IMAGE="${IMAGE:-docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420}"
IMAGE_TYPE="${IMAGE_TYPE:-SOURCE_PRIVATE}"
FRAMEWORK="${FRAMEWORK:-pytorch}"
INSTANCES="${INSTANCES:-1}"
SHM_GI="${SHM_GI:-1200}"
PRIORITY="${PRIORITY:-10}"

SYSTEM_TAG="${SYSTEM_TAG:-}"
INPUT_SYSTEM_ID="${INPUT_SYSTEM_ID:-}"
EN_INPUT="${EN_INPUT:-}"
ZH_INPUT="${ZH_INPUT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
EN_TEST_SET_ID="${EN_TEST_SET_ID:-seedtts-vc-en-internal320-disjoint}"
ZH_TEST_SET_ID="${ZH_TEST_SET_ID:-seedtts-vc-zh-internal320-disjoint}"
RUN_TAG="${RUN_TAG:-20260711_mtts}"
DRY_RUN="${DRY_RUN:-1}"
FORCE="${FORCE:-0}"
ENTRYPOINT="${BATCH42_UNIFIED_SCORERS_ENTRYPOINT:-0}"
ENABLE_QWEN_ASR="${ENABLE_QWEN_ASR:-0}"
NUM_SHARDS="${NUM_SHARDS:-8}"

EXPECTED_EN_CASES=567
EXPECTED_ZH_CASES=1194

SCORER_PYTHON="${SCORER_PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
BATCH42_PYTHON_DEPS="${BATCH42_PYTHON_DEPS:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/python_deps/batch42_eval}"
SPEECHBRAIN_PYTHON_DEPS="${SPEECHBRAIN_PYTHON_DEPS:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/python_deps/speechbrain_py312}"
SCORER_BASE_SITE="${SCORER_BASE_SITE:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/lib/python3.12/site-packages}"

SPEAKER_SIM_ROOT="${SPEAKER_SIM_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction}"
WAVLM_CHECKPOINT="${WAVLM_CHECKPOINT:-/inspire/hdd/project/embodied-multimodality/public/kxhuang/vcdata_construction/models/wavlm_large_finetune.pth}"
SEEDTTS_EVAL_ROOT="${SEEDTTS_EVAL_ROOT:-/inspire/hdd/project/embodied-multimodality/public/kxhuang/vcdata_construction/models/seed-tts-eval}"
WAVLM_MODEL_DIR="${WAVLM_MODEL_DIR:-/inspire/hdd/project/embodied-multimodality/public/kxhuang/vcdata_construction/models/wavlm-large}"
ERES2NET_MODEL="${ERES2NET_MODEL:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/batch42/models/modelscope/models/iic--speech_eres2net_sv_zh-cn_16k-common/snapshots/master}"
SPEECHBRAIN_MODEL="${SPEECHBRAIN_MODEL:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/models/speechbrain/spkrec-ecapa-voxceleb}"
PARAFORMER_MODEL="${PARAFORMER_MODEL:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/batch42/models/modelscope/models/damo--speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/snapshots/master}"
WHISPER_MODEL="${WHISPER_MODEL:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/batch42/models/huggingface/hub/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1}"
QWEN_ASR_MODEL="${QWEN_ASR_MODEL:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/checkpoint/qwen-asr-1_7b}"

HF_HOME="${HF_HOME:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/batch42/models/huggingface}"
MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/batch42/models/modelscope}"
TORCH_HOME="${TORCH_HOME:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/batch42/models/torch}"

SOURCE_EVAL_SCRIPT="$PROJECT_ROOT/scripts/004082_run_unified_vc_eval.py"
SOURCE_QWEN_ADAPTER="$PROJECT_ROOT/scripts/001017_asr_content_filter.py"
SOURCE_SCHEMA="$PROJECT_ROOT/docs/schemas/moss_codecvc_unified_vc_eval_v1.schema.json"
SOURCE_PROVENANCE_HELPER="$PROJECT_ROOT/scripts/batch42_scorer_provenance.py"
RECORD_ROOT="${RECORD_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/batch42_unified_scorers_${SYSTEM_TAG:-unset}_${RUN_TAG}}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-$RECORD_ROOT/record_snapshot}"
FROZEN_DRIVER="$SNAPSHOT_ROOT/scripts/004091_submit_batch42_unified_scorers_qz.sh"
FROZEN_EVAL_SCRIPT="$SNAPSHOT_ROOT/scripts/004082_run_unified_vc_eval.py"
FROZEN_QWEN_ADAPTER="$SNAPSHOT_ROOT/scripts/001017_asr_content_filter.py"
FROZEN_SCHEMA="$SNAPSHOT_ROOT/docs/schemas/moss_codecvc_unified_vc_eval_v1.schema.json"
FROZEN_PROVENANCE_HELPER="$SNAPSHOT_ROOT/scripts/batch42_scorer_provenance.py"
SNAPSHOT_SHA="$SNAPSHOT_ROOT/snapshot.sha256"
DEPENDENCY_REPORT="$RECORD_ROOT/dependency_report.json"
JOB_NAME="${JOB_NAME:-batch42_score_${SYSTEM_TAG:-unset}_${RUN_TAG}}"
SOURCE_INFERENCE_COMPLETION="${SOURCE_INFERENCE_COMPLETION:-}"
SOURCE_FINAL_SELECTION="${SOURCE_FINAL_SELECTION:-}"
INPUT_PROVENANCE="${INPUT_PROVENANCE:-$RECORD_ROOT/input_provenance.json}"
SUBMISSION_CONTRACT="${SUBMISSION_CONTRACT:-$RECORD_ROOT/submission_contract.json}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

validate_static_contract() {
  [ -n "$SYSTEM_TAG" ] || die "SYSTEM_TAG is required"
  if [ -z "$INPUT_SYSTEM_ID" ]; then
    if [ "$SYSTEM_TAG" = "seed_vc_v2_style" ]; then
      INPUT_SYSTEM_ID="seed_vc_v2"
    else
      INPUT_SYSTEM_ID="$SYSTEM_TAG"
    fi
  fi
  [ -n "$EN_INPUT" ] || die "EN_INPUT is required"
  [ -n "$ZH_INPUT" ] || die "ZH_INPUT is required"
  [ -n "$OUTPUT_ROOT" ] || die "OUTPUT_ROOT is required"
  case "$SYSTEM_TAG" in
    *[!A-Za-z0-9._-]*|'') die "SYSTEM_TAG must contain only A-Z, a-z, 0-9, dot, underscore, or dash" ;;
  esac
  case "$INPUT_SYSTEM_ID" in
    *[!A-Za-z0-9._-]*|'') die "INPUT_SYSTEM_ID must contain only A-Z, a-z, 0-9, dot, underscore, or dash" ;;
  esac
  [ -n "$EN_TEST_SET_ID" ] || die "EN_TEST_SET_ID must not be empty"
  [ -n "$ZH_TEST_SET_ID" ] || die "ZH_TEST_SET_ID must not be empty"
  case "$DRY_RUN:$FORCE:$ENTRYPOINT:$ENABLE_QWEN_ASR" in
    [01]:[01]:[01]:[01]) ;;
    *) die "DRY_RUN, FORCE, BATCH42_UNIFIED_SCORERS_ENTRYPOINT, and ENABLE_QWEN_ASR must be 0 or 1" ;;
  esac
  [ "$COMPUTE_GROUP" = "$ALLOWED_COMPUTE_GROUP" ] || \
    die "only MTTS-3-2-0715 ($ALLOWED_COMPUTE_GROUP) is allowed; got $COMPUTE_GROUP"
  [ "$WORKSPACE" = "$ALLOWED_WORKSPACE" ] || \
    die "only CI-情境智能 workspace $ALLOWED_WORKSPACE is allowed; got $WORKSPACE"
  [ "$PROJECT" = "$ALLOWED_PROJECT" ] || \
    die "only CI-情境智能 project $ALLOWED_PROJECT is allowed; got $PROJECT"
  [ "$SPEC" = "$ALLOWED_SPEC" ] || die "only spec $ALLOWED_SPEC is allowed; got $SPEC"
  [ "$QZCLI_GPU_TYPE_OVERRIDE" = "$ALLOWED_GPU_TYPE" ] || \
    die "only GPU type $ALLOWED_GPU_TYPE is allowed; got $QZCLI_GPU_TYPE_OVERRIDE"
  [ "$INSTANCES" = "1" ] || die "exactly one 8xH200 instance is required; got $INSTANCES"
  [ "$NUM_SHARDS" = "8" ] || die "NUM_SHARDS is fixed at 8; got $NUM_SHARDS"
  [ -s "$SOURCE_PROVENANCE_HELPER" ] || [ -s "$FROZEN_PROVENANCE_HELPER" ] || \
    die "missing scorer provenance helper"
  if [ -n "$SOURCE_INFERENCE_COMPLETION" ] || [ -n "$SOURCE_FINAL_SELECTION" ]; then
    [ -s "$SOURCE_INFERENCE_COMPLETION" ] || die "missing SOURCE_INFERENCE_COMPLETION"
    [ -s "$SOURCE_FINAL_SELECTION" ] || die "missing SOURCE_FINAL_SELECTION"
  fi
  if [ "$SYSTEM_TAG" = "path_x_final" ]; then
    [ "$INPUT_SYSTEM_ID" = "path_x_final" ] || \
      die "path_x_final requires INPUT_SYSTEM_ID=path_x_final"
    [ -s "$SOURCE_INFERENCE_COMPLETION" ] || \
      die "path_x_final requires strict inference COMPLETED binding"
    [ -s "$SOURCE_FINAL_SELECTION" ] || \
      die "path_x_final requires FINAL_SELECTION binding"
  fi
}

audit_gpu_inventory() {
  local -a gpu_names=()
  local name
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable in scorer job"
  mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
  [ "${#gpu_names[@]}" = "8" ] || \
    die "scorer job must expose exactly 8 GPUs; found ${#gpu_names[@]}"
  for name in "${gpu_names[@]}"; do
    case "$name" in
      *H200*) ;;
      *) die "scorer job requires all GPUs to be H200; found $name" ;;
    esac
  done
  printf '[batch42-scorer-gpu] PASS count=8 names=%s\n' "$(IFS=,; echo "${gpu_names[*]}")"
}

validate_runtime_assets() {
  local path
  [ -x "$SCORER_PYTHON" ] || die "missing scorer Python: $SCORER_PYTHON"
  [ -d "$BATCH42_PYTHON_DEPS" ] || die "missing Batch-42 Python deps: $BATCH42_PYTHON_DEPS"
  [ -d "$SPEECHBRAIN_PYTHON_DEPS" ] || die "missing SpeechBrain Python deps: $SPEECHBRAIN_PYTHON_DEPS"
  [ -d "$SCORER_BASE_SITE" ] || die "missing scorer base site-packages: $SCORER_BASE_SITE"
  [ -s "$SOURCE_EVAL_SCRIPT" ] || [ -s "$FROZEN_EVAL_SCRIPT" ] || die "missing 004082 evaluator"
  [ -s "$SOURCE_SCHEMA" ] || [ -s "$FROZEN_SCHEMA" ] || die "missing unified-eval JSON schema"
  for path in \
    "$SPEAKER_SIM_ROOT/speaker_similarity.py" \
    "$WAVLM_CHECKPOINT" \
    "$SEEDTTS_EVAL_ROOT" \
    "$WAVLM_MODEL_DIR" \
    "$ERES2NET_MODEL/pretrained_eres2net_aug.ckpt" \
    "$SPEECHBRAIN_MODEL/hyperparams.yaml" \
    "$PARAFORMER_MODEL/model.pt" \
    "$WHISPER_MODEL/model.safetensors"; do
    [ -e "$path" ] || die "missing scorer model asset: $path"
  done
  if [ "$ENABLE_QWEN_ASR" = "1" ]; then
    [ -d "$QWEN_ASR_MODEL" ] || die "ENABLE_QWEN_ASR=1 but model is missing: $QWEN_ASR_MODEL"
    [ -s "$SOURCE_QWEN_ADAPTER" ] || [ -s "$FROZEN_QWEN_ADAPTER" ] || \
      die "ENABLE_QWEN_ASR=1 but legacy adapter is missing: $SOURCE_QWEN_ADAPTER"
  fi
  PYTHONPATH="$BATCH42_PYTHON_DEPS:$SCORER_BASE_SITE:$SPEECHBRAIN_PYTHON_DEPS:$SNAPSHOT_ROOT:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$SCORER_PYTHON" - <<'PY'
import importlib.util
import huggingface_hub
import transformers

required = (
    "torch", "torchaudio", "transformers", "soundfile", "modelscope",
    "addict", "funasr", "zhconv", "jsonschema", "speechbrain",
)
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing scorer Python modules: {missing}")
hub_major = int(str(huggingface_hub.__version__).split(".", 1)[0])
if hub_major >= 1:
    raise SystemExit(
        "incompatible huggingface_hub selected before transformers: "
        f"{huggingface_hub.__version__} from {huggingface_hub.__file__}"
    )
print(
    "[batch42-scorer-runtime] transformers compatibility PASS "
    f"hub={huggingface_hub.__version__} transformers={transformers.__version__}"
)
print("[batch42-scorer-runtime] Python module audit PASS")
PY
}

audit_input_jsonl() {
  local language=$1
  local input=$2
  local expected=$3
  local expected_test_set_id=$4
  [ -s "$input" ] || die "$language input is missing/empty: $input"
  "$SCORER_PYTHON" - \
    "$input" "$language" "$expected" "$INPUT_SYSTEM_ID" "$expected_test_set_id" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

input_path = Path(sys.argv[1]).expanduser().resolve()
language = sys.argv[2]
expected = int(sys.argv[3])
input_system_id = sys.argv[4]
expected_test_set_id = sys.argv[5]
rows = []
with input_path.open(encoding="utf-8") as handle:
    for line_number, raw in enumerate(handle, start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{input_path}:{line_number}: invalid JSON: {exc}")
        if not isinstance(row, dict):
            raise SystemExit(f"{input_path}:{line_number}: row must be an object")
        rows.append(row)
if len(rows) != expected:
    raise SystemExit(f"{language}: expected {expected} rows, got {len(rows)}")
case_ids = [str(row.get("case_id") or "") for row in rows]
case_uids = [str(row.get("case_uid") or "") for row in rows]
if any(not value for value in case_ids) or len(set(case_ids)) != expected:
    raise SystemExit(f"{language}: missing/duplicate case_id")
if any(not value for value in case_uids) or len(set(case_uids)) != expected:
    raise SystemExit(f"{language}: missing/duplicate case_uid")
indices = [int(row.get("input_index", -1)) for row in rows]
if set(indices) != set(range(expected)):
    raise SystemExit(f"{language}: input_index must cover 0..{expected - 1}")
statuses = Counter(str(row.get("status") or "") for row in rows)
if set(statuses) - {"ok", "skipped_existing"}:
    raise SystemExit(f"{language}: non-success inference statuses: {dict(statuses)}")
test_sets = set()
for row in rows:
    case_id = str(row["case_id"])
    if str(row.get("system_id") or "") != input_system_id:
        raise SystemExit(
            f"{language}/{case_id}: system_id={row.get('system_id')!r}, "
            f"expected input system {input_system_id!r}"
        )
    row_language = str(row.get("language") or "").lower()
    if row_language != language:
        raise SystemExit(f"{language}/{case_id}: language={row_language!r}")
    test_sets.add(str(row.get("test_set_id") or ""))
    for field in ("generated_audio", "reference_audio", "source_audio"):
        path = Path(str(row.get(field) or ""))
        if not path.is_file() or path.stat().st_size < 44:
            raise SystemExit(f"{language}/{case_id}: missing/empty {field}: {path}")
    if not str(row.get("target_text") or row.get("reference_text") or "").strip():
        raise SystemExit(f"{language}/{case_id}: missing reference text")
if test_sets != {expected_test_set_id}:
    raise SystemExit(
        f"{language}: expected test_set_id={expected_test_set_id!r}, got {sorted(test_sets)}"
    )
print(
    f"[batch42-scorer-input] {language} PASS rows={expected} unique={expected} "
    f"status={dict(statuses)} input_system={input_system_id} "
    f"test_set={expected_test_set_id}"
)
PY
}

write_dependency_report() {
  local evaluator=$1
  mkdir -p "$RECORD_ROOT"
  PYTHONPATH="$BATCH42_PYTHON_DEPS:$SCORER_BASE_SITE:$SPEECHBRAIN_PYTHON_DEPS:$SNAPSHOT_ROOT:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$SCORER_PYTHON" "$evaluator" check \
      --output-json "$DEPENDENCY_REPORT" \
      --speaker-sim-root "$SPEAKER_SIM_ROOT" \
      --wavlm-implementation seedtts_official \
      --wavlm-checkpoint "$WAVLM_CHECKPOINT" \
      --seedtts-eval-root "$SEEDTTS_EVAL_ROOT" \
      --wavlm-model-dir "$WAVLM_MODEL_DIR" \
      --eres2net-model "$ERES2NET_MODEL" \
      --speechbrain-model "$SPEECHBRAIN_MODEL" \
      --paraformer-model "$PARAFORMER_MODEL" \
      --whisper-model "$WHISPER_MODEL" \
      --qwen-asr-model "$QWEN_ASR_MODEL"
}

prepare_snapshot() {
  mkdir -p "$SNAPSHOT_ROOT/scripts" "$SNAPSHOT_ROOT/docs/schemas" "$SNAPSHOT_ROOT/moss_codecvc"
  cp -p "$SELF_PATH" "$FROZEN_DRIVER"
  cp -p "$SOURCE_EVAL_SCRIPT" "$FROZEN_EVAL_SCRIPT"
  cp -p "$SOURCE_QWEN_ADAPTER" "$FROZEN_QWEN_ADAPTER"
  cp -p "$SOURCE_SCHEMA" "$FROZEN_SCHEMA"
  cp -p "$SOURCE_PROVENANCE_HELPER" "$FROZEN_PROVENANCE_HELPER"
  cp -a "$PROJECT_ROOT/moss_codecvc/." "$SNAPSHOT_ROOT/moss_codecvc/"
  (
    cd "$SNAPSHOT_ROOT"
    find scripts docs moss_codecvc -type f -print0 | sort -z | xargs -0 sha256sum
  ) > "$SNAPSHOT_SHA"
  write_dependency_report "$FROZEN_EVAL_SCRIPT"
  {
    printf 'key\tvalue\n'
    printf 'system_tag\t%s\n' "$SYSTEM_TAG"
    printf 'input_system_id\t%s\n' "$INPUT_SYSTEM_ID"
    printf 'en_input\t%s\n' "$EN_INPUT"
    printf 'zh_input\t%s\n' "$ZH_INPUT"
    printf 'output_root\t%s\n' "$OUTPUT_ROOT"
    printf 'scorer_python\t%s\n' "$SCORER_PYTHON"
    printf 'workspace\t%s\n' "$WORKSPACE"
    printf 'project\t%s\n' "$PROJECT"
    printf 'compute_group\t%s\n' "$COMPUTE_GROUP"
    printf 'spec\t%s\n' "$SPEC"
    printf 'instances\t%s\n' "$INSTANCES"
    printf 'num_shards\t%s\n' "$NUM_SHARDS"
    printf 'enable_qwen_asr\t%s\n' "$ENABLE_QWEN_ASR"
    printf 'source_inference_completion\t%s\n' "$SOURCE_INFERENCE_COMPLETION"
    printf 'source_final_selection\t%s\n' "$SOURCE_FINAL_SELECTION"
    printf 'input_provenance\t%s\n' "$INPUT_PROVENANCE"
    printf 'submission_contract\t%s\n' "$SUBMISSION_CONTRACT"
  } > "$RECORD_ROOT/resolved_config.tsv"
}

write_input_provenance() {
  "$SCORER_PYTHON" "$FROZEN_PROVENANCE_HELPER" write-input \
    --output "$INPUT_PROVENANCE" \
    --system-id "$SYSTEM_TAG" \
    --input-system-id "$INPUT_SYSTEM_ID" \
    --en-input "$EN_INPUT" \
    --zh-input "$ZH_INPUT" \
    --en-test-set-id "$EN_TEST_SET_ID" \
    --zh-test-set-id "$ZH_TEST_SET_ID" \
    --output-root "$OUTPUT_ROOT" \
    --snapshot-manifest "$SNAPSHOT_SHA" \
    --source-inference-completion "$SOURCE_INFERENCE_COMPLETION" \
    --source-final-selection "$SOURCE_FINAL_SELECTION"
}

wait_for_submission_contract() {
  "$SCORER_PYTHON" "$FROZEN_PROVENANCE_HELPER" wait-submission \
    --contract "$SUBMISSION_CONTRACT" \
    --input-provenance "$INPUT_PROVENANCE" \
    --system-id "$SYSTEM_TAG" \
    --output-root "$OUTPUT_ROOT" \
    --record-root "$RECORD_ROOT" \
    --snapshot-root "$SNAPSHOT_ROOT" \
    --timeout-seconds 300
}

strict_audit_merged() {
  local language=$1
  local merged_jsonl=$2
  local expected=$3
  local primary_asr=$4
  local test_set_id=$5
  local audit_json=$6
  local audit_md=$7
  "$SCORER_PYTHON" - \
    "$merged_jsonl" "$FROZEN_SCHEMA" "$language" "$expected" \
    "$SYSTEM_TAG" "$test_set_id" "$primary_asr" "$ENABLE_QWEN_ASR" \
    "$audit_json" "$audit_md" <<'PY'
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import jsonschema

(
    merged_path,
    schema_path,
    language,
    expected_raw,
    system_tag,
    test_set_id,
    primary_asr,
    qwen_raw,
    audit_json,
    audit_md,
) = sys.argv[1:]
expected = int(expected_raw)
qwen_enabled = qwen_raw == "1"
merged_path = Path(merged_path)
rows = [
    json.loads(line)
    for line in merged_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
validator = jsonschema.Draft202012Validator(schema)
if len(rows) != expected:
    raise SystemExit(f"{language}: merged rows={len(rows)}, expected={expected}")
case_ids = [str(row.get("case_id") or "") for row in rows]
if any(not value for value in case_ids) or len(set(case_ids)) != expected:
    raise SystemExit(f"{language}: missing/duplicate case_id after merge")
indices = [int((row.get("provenance") or {}).get("input_index", -1)) for row in rows]
if set(indices) != set(range(expected)):
    raise SystemExit(f"{language}: merged input_index coverage is not 0..{expected - 1}")

speaker_backends = ("wavlm_large_sv", "eres2net", "speechbrain_ecapa")
required_asr = [primary_asr] + (["qwen_asr"] if qwen_enabled else [])
speaker_status = {backend: Counter() for backend in speaker_backends}
asr_status = {backend: Counter() for backend in required_asr}
for row_number, row in enumerate(rows, start=1):
    errors = sorted(validator.iter_errors(row), key=lambda item: list(item.path))
    if errors:
        first = errors[0]
        raise SystemExit(
            f"{language}: schema failure row={row_number} case={row.get('case_id')}: {first.message}"
        )
    case_id = str(row["case_id"])
    if row.get("system_id") != system_tag or row.get("test_set_id") != test_set_id:
        raise SystemExit(
            f"{language}/{case_id}: identity mismatch system/test="
            f"{row.get('system_id')!r}/{row.get('test_set_id')!r}"
        )
    if row.get("language") != language:
        raise SystemExit(f"{language}/{case_id}: language={row.get('language')!r}")
    for field in ("generated", "reference", "source"):
        path = Path(str((row.get("audio") or {}).get(field) or ""))
        if not path.is_file() or path.stat().st_size < 44:
            raise SystemExit(f"{language}/{case_id}: missing/empty audio.{field}: {path}")
    for section_name in ("speaker_similarity", "content_asr"):
        for backend, result in (row.get(section_name) or {}).items():
            if not isinstance(result, dict):
                raise SystemExit(f"{language}/{case_id}: {section_name}.{backend} is not an object")
            if result.get("error"):
                raise SystemExit(
                    f"{language}/{case_id}: {section_name}.{backend} error={result.get('error')}"
                )
            if result.get("status") in {
                "error", "backend_unavailable", "missing_audio", "missing_reference"
            }:
                raise SystemExit(
                    f"{language}/{case_id}: {section_name}.{backend} status={result.get('status')}"
                )
    for backend in speaker_backends:
        result = (row.get("speaker_similarity") or {}).get(backend) or {}
        speaker_status[backend][str(result.get("status") or "missing")] += 1
        if result.get("status") != "ok":
            raise SystemExit(f"{language}/{case_id}: speaker {backend} status={result.get('status')}")
        for metric in ("sim_ref", "sim_src"):
            value = result.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise SystemExit(f"{language}/{case_id}: speaker {backend}.{metric}={value!r}")
    for backend in required_asr:
        result = (row.get("content_asr") or {}).get(backend) or {}
        asr_status[backend][str(result.get("status") or "missing")] += 1
        if result.get("status") != "ok":
            raise SystemExit(f"{language}/{case_id}: ASR {backend} status={result.get('status')}")
        value = result.get("primary_error")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise SystemExit(f"{language}/{case_id}: ASR {backend}.primary_error={value!r}")

payload = {
    "schema_version": "moss_codecvc.batch42_strict_scorer_audit.v1",
    "system_id": system_tag,
    "test_set_id": test_set_id,
    "language": language,
    "rows": len(rows),
    "unique_case_ids": len(set(case_ids)),
    "input_index_coverage": [min(indices), max(indices)],
    "speaker_status_counts": {
        backend: dict(counts) for backend, counts in speaker_status.items()
    },
    "asr_status_counts": {backend: dict(counts) for backend, counts in asr_status.items()},
    "all_ok": True,
    "merged_jsonl": str(merged_path.resolve()),
}

def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)

atomic_write(audit_json, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
lines = [
    f"# Batch-42 strict scorer audit: {system_tag} / {language}",
    "",
    f"- rows / unique: {len(rows)} / {len(set(case_ids))}",
    f"- speaker: `{json.dumps(payload['speaker_status_counts'], sort_keys=True)}`",
    f"- ASR: `{json.dumps(payload['asr_status_counts'], sort_keys=True)}`",
    "- all_ok: `true`",
    "",
]
atomic_write(audit_md, "\n".join(lines))
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
}

runtime_score_smoke() {
  local smoke_root="$RECORD_ROOT/runtime_smoke"
  local smoke_stem="${SYSTEM_TAG}.en.runtime-smoke"
  local smoke_jsonl="$smoke_root/$smoke_stem.unified_eval.jsonl"
  mkdir -p "$smoke_root"
  echo "[batch42-scorer-smoke] actual one-case start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  (
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH="$BATCH42_PYTHON_DEPS:$SCORER_BASE_SITE:$SPEECHBRAIN_PYTHON_DEPS:$SNAPSHOT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export HF_HOME TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
    export MODELSCOPE_CACHE TORCH_HOME
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    "$SCORER_PYTHON" "$FROZEN_EVAL_SCRIPT" evaluate \
      --input "$EN_INPUT" \
      --output-dir "$smoke_root" \
      --output-stem "$smoke_stem" \
      --run-id "${SYSTEM_TAG}_en_runtime_smoke" \
      --system-id "$SYSTEM_TAG" \
      --test-set-id "$EN_TEST_SET_ID" \
      --input-profile official_seedtts_vc \
      --metric-profile seedtts_official \
      --speaker-scorer all \
      --asr-backend whisper_large_v3 \
      --continue-on-error \
      --no-reuse-existing \
      --limit 1 \
      --speaker-device cuda:0 \
      --asr-device cuda:0 \
      --speaker-sim-root "$SPEAKER_SIM_ROOT" \
      --wavlm-implementation seedtts_official \
      --wavlm-checkpoint "$WAVLM_CHECKPOINT" \
      --seedtts-eval-root "$SEEDTTS_EVAL_ROOT" \
      --wavlm-model-dir "$WAVLM_MODEL_DIR" \
      --eres2net-model "$ERES2NET_MODEL" \
      --speechbrain-model "$SPEECHBRAIN_MODEL" \
      --paraformer-model "$PARAFORMER_MODEL" \
      --whisper-model "$WHISPER_MODEL" \
      --qwen-asr-model "$QWEN_ASR_MODEL"
  )
  "$SCORER_PYTHON" - "$smoke_jsonl" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != 1:
    raise SystemExit(f"runtime smoke expected one row, got {len(rows)}: {path}")
row = rows[0]
errors = []
for backend in ("wavlm_large_sv", "eres2net", "speechbrain_ecapa"):
    result = (row.get("speaker_similarity") or {}).get(backend) or {}
    if result.get("status") != "ok" or result.get("error"):
        errors.append(f"speaker.{backend}={result}")
    for metric in ("sim_ref", "sim_src"):
        value = result.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(f"speaker.{backend}.{metric}={value!r}")
asr = (row.get("content_asr") or {}).get("whisper_large_v3") or {}
if asr.get("status") != "ok" or asr.get("error"):
    errors.append(f"asr.whisper_large_v3={asr}")
if errors:
    raise SystemExit("runtime scorer smoke failed:\n- " + "\n- ".join(errors))
print(f"[batch42-scorer-smoke] PASS path={path}")
PY
}

run_language() {
  local language=$1
  local input=$2
  local expected=$3
  local primary_asr=$4
  local test_set_id=$5
  local lang_root="$OUTPUT_ROOT/$language"
  local partial_root="$lang_root/partials"
  local log_root="$lang_root/logs"
  local merged_root="$lang_root/merged"
  local run_id="${SYSTEM_TAG}_${language}_unified_scorers"
  local -a pids=()
  local -a logs=()
  local shard stem log_path

  mkdir -p "$partial_root" "$log_root" "$merged_root"
  echo "[batch42-scorer] language=$language rows=$expected primary_asr=$primary_asr start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for shard in 0 1 2 3 4 5 6 7; do
    stem=$(printf '%s.%s.shard-%05d-of-%05d' "$SYSTEM_TAG" "$language" "$shard" "$NUM_SHARDS")
    log_path="$log_root/$stem.log"
    logs+=("$log_path")
    (
      export CUDA_VISIBLE_DEVICES="$shard"
      export PYTHONPATH="$BATCH42_PYTHON_DEPS:$SCORER_BASE_SITE:$SPEECHBRAIN_PYTHON_DEPS:$SNAPSHOT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
      export HF_HOME TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
      export MODELSCOPE_CACHE TORCH_HOME
      export TOKENIZERS_PARALLELISM=false
      export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      worker_command=(
        "$SCORER_PYTHON" "$FROZEN_EVAL_SCRIPT" evaluate
        --input "$input"
        --output-dir "$partial_root"
        --output-stem "$stem"
        --run-id "$run_id"
        --system-id "$SYSTEM_TAG"
        --test-set-id "$test_set_id"
        --input-profile official_seedtts_vc
        --metric-profile seedtts_official
        --speaker-scorer all
        --asr-backend "$primary_asr"
        --continue-on-error
        --no-reuse-existing
        --num-shards "$NUM_SHARDS"
        --shard-index "$shard"
        --speaker-device cuda:0
        --asr-device cuda:0
        --speaker-sim-root "$SPEAKER_SIM_ROOT"
        --wavlm-implementation seedtts_official
        --wavlm-checkpoint "$WAVLM_CHECKPOINT"
        --seedtts-eval-root "$SEEDTTS_EVAL_ROOT"
        --wavlm-model-dir "$WAVLM_MODEL_DIR"
        --eres2net-model "$ERES2NET_MODEL"
        --speechbrain-model "$SPEECHBRAIN_MODEL"
        --paraformer-model "$PARAFORMER_MODEL"
        --whisper-model "$WHISPER_MODEL"
        --qwen-asr-model "$QWEN_ASR_MODEL"
      )
      if [ "$ENABLE_QWEN_ASR" = "1" ]; then
        worker_command+=(--asr-backend qwen_asr)
      fi
      "${worker_command[@]}"
    ) > "$log_path" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  for shard in 0 1 2 3 4 5 6 7; do
    if ! wait "${pids[$shard]}"; then
      echo "ERROR: $language shard $shard failed; log=${logs[$shard]}" >&2
      tail -n 120 "${logs[$shard]}" >&2 || true
      failed=1
    fi
  done
  [ "$failed" = "0" ] || return 1

  local -a merge_args=()
  local -a partials=()
  local partial_path
  for shard in 0 1 2 3 4 5 6 7; do
    stem=$(printf '%s.%s.shard-%05d-of-%05d' "$SYSTEM_TAG" "$language" "$shard" "$NUM_SHARDS")
    partial_path="$partial_root/$stem.unified_eval.jsonl"
    partials+=("$partial_path")
    [ -s "$partial_path" ] || die "$language shard $shard did not emit $partial_path"
    merge_args+=(--partial "$partial_path")
  done
  local merged_stem="${SYSTEM_TAG}.${language}.merged"
  PYTHONPATH="$BATCH42_PYTHON_DEPS:$SCORER_BASE_SITE:$SPEECHBRAIN_PYTHON_DEPS:$SNAPSHOT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$SCORER_PYTHON" "$FROZEN_EVAL_SCRIPT" merge \
      "${merge_args[@]}" \
      --output-dir "$merged_root" \
      --output-stem "$merged_stem" \
      --run-id "${run_id}_merged"

  strict_audit_merged \
    "$language" \
    "$merged_root/$merged_stem.unified_eval.jsonl" \
    "$expected" \
    "$primary_asr" \
    "$test_set_id" \
    "$merged_root/$merged_stem.strict_audit.json" \
    "$merged_root/$merged_stem.strict_audit.md"
  echo "[batch42-scorer] language=$language complete=$(date -u +%Y-%m-%dT%H:%M:%SZ) summary=$merged_root/$merged_stem.summary.json"
}

write_combined_summary() {
  "$SCORER_PYTHON" - \
    "$SYSTEM_TAG" "$OUTPUT_ROOT" \
    "$OUTPUT_ROOT/en/merged/${SYSTEM_TAG}.en.merged.summary.json" \
    "$OUTPUT_ROOT/en/merged/${SYSTEM_TAG}.en.merged.strict_audit.json" \
    "$OUTPUT_ROOT/zh/merged/${SYSTEM_TAG}.zh.merged.summary.json" \
    "$OUTPUT_ROOT/zh/merged/${SYSTEM_TAG}.zh.merged.strict_audit.json" <<'PY'
import json
import os
import sys
from pathlib import Path

system_tag, output_root, en_summary, en_audit, zh_summary, zh_audit = sys.argv[1:]
output_root = Path(output_root)

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

payload = {
    "schema_version": "moss_codecvc.batch42_system_unified_summary.v1",
    "system_id": system_tag,
    "en": {
        "summary_path": str(Path(en_summary).resolve()),
        "strict_audit_path": str(Path(en_audit).resolve()),
        "strict_audit": load(en_audit),
        "group_all": load(en_summary)["groups"]["all"],
    },
    "zh": {
        "summary_path": str(Path(zh_summary).resolve()),
        "strict_audit_path": str(Path(zh_audit).resolve()),
        "strict_audit": load(zh_audit),
        "group_all": load(zh_summary)["groups"]["all"],
    },
}
json_path = output_root / f"{system_tag}.en_zh.summary.json"
md_path = output_root / f"{system_tag}.en_zh.summary.md"

def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)

atomic_write(json_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
lines = [f"# Batch-42 unified scorer summary: {system_tag}", ""]
for language in ("en", "zh"):
    item = payload[language]
    group = item["group_all"]
    lines.extend([f"## {language.upper()}", ""])
    lines.append(f"- cases: {group['n_cases']}")
    lines.append(f"- strict audit: `{item['strict_audit_path']}`")
    lines.append("")
    lines.append("| speaker scorer | n | SIM(ref) | SIM(src) |")
    lines.append("|---|---:|---:|---:|")
    for backend in ("wavlm_large_sv", "eres2net", "speechbrain_ecapa"):
        metric = group["speaker_similarity"][backend]
        lines.append(
            f"| {backend} | {metric['sim_ref']['n']} | "
            f"{metric['sim_ref']['mean']:.6f} | {metric['sim_src']['mean']:.6f} |"
        )
    lines.append("")
    lines.append("| ASR | n | primary error |")
    lines.append("|---|---:|---:|")
    for backend, metric in group["content_asr"].items():
        if metric["primary_error"]["n"]:
            lines.append(
                f"| {backend} | {metric['primary_error']['n']} | "
                f"{metric['primary_error']['mean']:.6f} |"
            )
    lines.append("")
atomic_write(md_path, "\n".join(lines) + "\n")
print(json.dumps({"summary_json": str(json_path), "summary_md": str(md_path)}, indent=2))
PY

  "$SCORER_PYTHON" "$FROZEN_PROVENANCE_HELPER" bind-combined \
    --combined-summary "$OUTPUT_ROOT/${SYSTEM_TAG}.en_zh.summary.json" \
    --system-id "$SYSTEM_TAG" \
    --input-provenance "$INPUT_PROVENANCE" \
    --submission-contract "$SUBMISSION_CONTRACT" \
    --en-summary "$OUTPUT_ROOT/en/merged/${SYSTEM_TAG}.en.merged.summary.json" \
    --en-audit "$OUTPUT_ROOT/en/merged/${SYSTEM_TAG}.en.merged.strict_audit.json" \
    --en-merged-jsonl "$OUTPUT_ROOT/en/merged/${SYSTEM_TAG}.en.merged.unified_eval.jsonl" \
    --zh-summary "$OUTPUT_ROOT/zh/merged/${SYSTEM_TAG}.zh.merged.summary.json" \
    --zh-audit "$OUTPUT_ROOT/zh/merged/${SYSTEM_TAG}.zh.merged.strict_audit.json" \
    --zh-merged-jsonl "$OUTPUT_ROOT/zh/merged/${SYSTEM_TAG}.zh.merged.unified_eval.jsonl"
}

write_completion() {
  "$SCORER_PYTHON" "$FROZEN_PROVENANCE_HELPER" write-completion \
    --output "$OUTPUT_ROOT/completion.json" \
    --system-id "$SYSTEM_TAG" \
    --output-root "$OUTPUT_ROOT" \
    --input-provenance "$INPUT_PROVENANCE" \
    --submission-contract "$SUBMISSION_CONTRACT"
}

run_entrypoint() {
  validate_static_contract
  exec > >(tee -a "$RECORD_ROOT/job.log") 2>&1
  echo "[batch42-scorer] start=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) system=$SYSTEM_TAG"
  echo "[batch42-scorer] en=$EN_INPUT zh=$ZH_INPUT output=$OUTPUT_ROOT"
  (
    cd "$SNAPSHOT_ROOT"
    sha256sum -c "$SNAPSHOT_SHA"
  )
  wait_for_submission_contract
  validate_runtime_assets
  audit_input_jsonl en "$EN_INPUT" "$EXPECTED_EN_CASES" "$EN_TEST_SET_ID"
  audit_input_jsonl zh "$ZH_INPUT" "$EXPECTED_ZH_CASES" "$ZH_TEST_SET_ID"
  write_dependency_report "$FROZEN_EVAL_SCRIPT"
  mkdir -p "$OUTPUT_ROOT"
  audit_gpu_inventory
  nvidia-smi
  runtime_score_smoke
  run_language en "$EN_INPUT" "$EXPECTED_EN_CASES" whisper_large_v3 "$EN_TEST_SET_ID"
  run_language zh "$ZH_INPUT" "$EXPECTED_ZH_CASES" paraformer_zh "$ZH_TEST_SET_ID"
  write_combined_summary
  write_completion
  echo "[batch42-scorer] complete=$(date -u +%Y-%m-%dT%H:%M:%SZ) summary=$OUTPUT_ROOT/${SYSTEM_TAG}.en_zh.summary.json"
}

validate_static_contract

if [ "$ENTRYPOINT" = "1" ]; then
  run_entrypoint
  exit 0
fi

[ -d "$PROJECT_ROOT" ] || die "missing PROJECT_ROOT: $PROJECT_ROOT"
[ -x "$QZCLI" ] || die "missing qzcli wrapper: $QZCLI"
validate_runtime_assets
audit_input_jsonl en "$EN_INPUT" "$EXPECTED_EN_CASES" "$EN_TEST_SET_ID"
audit_input_jsonl zh "$ZH_INPUT" "$EXPECTED_ZH_CASES" "$ZH_TEST_SET_ID"
mkdir -p "$RECORD_ROOT" "$QZCLI_HOME"

if [ "$FORCE" != "1" ] && {
  [ -s "$RECORD_ROOT/submitted_jobs.tsv" ] ||
  [ -s "$SUBMISSION_CONTRACT" ] ||
  [ -s "$OUTPUT_ROOT/completion.json" ] ||
  [ -s "$OUTPUT_ROOT/${SYSTEM_TAG}.en_zh.summary.json" ];
}; then
  die "existing submission/completion detected; set FORCE=1 only for an intentional rerun"
fi

prepare_snapshot
write_input_provenance
if [ -e "$SUBMISSION_CONTRACT" ]; then
  if [ "$FORCE" != "1" ]; then
    die "stale submission contract detected: $SUBMISSION_CONTRACT"
  fi
  "$SCORER_PYTHON" - "$SUBMISSION_CONTRACT" <<'PY'
import sys
from pathlib import Path

Path(sys.argv[1]).unlink(missing_ok=True)
PY
fi

command_parts=(
  env
  BATCH42_UNIFIED_SCORERS_ENTRYPOINT=1
  SYSTEM_TAG="$SYSTEM_TAG"
  INPUT_SYSTEM_ID="$INPUT_SYSTEM_ID"
  EN_INPUT="$EN_INPUT"
  ZH_INPUT="$ZH_INPUT"
  OUTPUT_ROOT="$OUTPUT_ROOT"
  EN_TEST_SET_ID="$EN_TEST_SET_ID"
  ZH_TEST_SET_ID="$ZH_TEST_SET_ID"
  RUN_TAG="$RUN_TAG"
  FORCE="$FORCE"
  ENABLE_QWEN_ASR="$ENABLE_QWEN_ASR"
  PROJECT_ROOT="$PROJECT_ROOT"
  RECORD_ROOT="$RECORD_ROOT"
  SNAPSHOT_ROOT="$SNAPSHOT_ROOT"
  SOURCE_INFERENCE_COMPLETION="$SOURCE_INFERENCE_COMPLETION"
  SOURCE_FINAL_SELECTION="$SOURCE_FINAL_SELECTION"
  INPUT_PROVENANCE="$INPUT_PROVENANCE"
  SUBMISSION_CONTRACT="$SUBMISSION_CONTRACT"
  SCORER_PYTHON="$SCORER_PYTHON"
  BATCH42_PYTHON_DEPS="$BATCH42_PYTHON_DEPS"
  SPEECHBRAIN_PYTHON_DEPS="$SPEECHBRAIN_PYTHON_DEPS"
  SPEAKER_SIM_ROOT="$SPEAKER_SIM_ROOT"
  WAVLM_CHECKPOINT="$WAVLM_CHECKPOINT"
  SEEDTTS_EVAL_ROOT="$SEEDTTS_EVAL_ROOT"
  WAVLM_MODEL_DIR="$WAVLM_MODEL_DIR"
  ERES2NET_MODEL="$ERES2NET_MODEL"
  SPEECHBRAIN_MODEL="$SPEECHBRAIN_MODEL"
  PARAFORMER_MODEL="$PARAFORMER_MODEL"
  WHISPER_MODEL="$WHISPER_MODEL"
  QWEN_ASR_MODEL="$QWEN_ASR_MODEL"
  HF_HOME="$HF_HOME"
  MODELSCOPE_CACHE="$MODELSCOPE_CACHE"
  TORCH_HOME="$TORCH_HOME"
  bash "$FROZEN_DRIVER"
)
printf -v COMMAND '%q ' "${command_parts[@]}"
COMMAND=${COMMAND% }
SUBMIT_OUTPUT="$RECORD_ROOT/submit_output.txt"

echo "=========================================="
echo "QZ submit: Batch-42 unified scorers"
echo "  JOB_NAME=$JOB_NAME"
echo "  SYSTEM_TAG=$SYSTEM_TAG"
echo "  INPUT_SYSTEM_ID=$INPUT_SYSTEM_ID"
echo "  EN_INPUT=$EN_INPUT ($EXPECTED_EN_CASES rows)"
echo "  ZH_INPUT=$ZH_INPUT ($EXPECTED_ZH_CASES rows)"
echo "  OUTPUT_ROOT=$OUTPUT_ROOT"
echo "  SNAPSHOT_ROOT=$SNAPSHOT_ROOT"
echo "  INPUT_PROVENANCE=$INPUT_PROVENANCE"
echo "  SOURCE_INFERENCE_COMPLETION=${SOURCE_INFERENCE_COMPLETION:-none}"
echo "  SOURCE_FINAL_SELECTION=${SOURCE_FINAL_SELECTION:-none}"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP (MTTS-3-2-0715 only)"
echo "  SPEC=$SPEC INSTANCES=$INSTANCES GPU_TYPE=$QZCLI_GPU_TYPE_OVERRIDE"
echo "  DRY_RUN=$DRY_RUN ENABLE_QWEN_ASR=$ENABLE_QWEN_ASR"
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
  echo "ERROR: QZ create-job failed; see $SUBMIT_OUTPUT" >&2
  exit "$status"
fi
if [ "$DRY_RUN" = "1" ]; then
  echo "[batch42-scorer] dry-run passed; no job submitted"
  exit 0
fi

job_id=$(printf '%s\n' "$output" | grep -Eo 'job-[0-9a-fA-F-]{36}' | tail -n 1 || true)
[ -n "$job_id" ] || die "submission returned no job ID; see $SUBMIT_OUTPUT"
"$SCORER_PYTHON" "$FROZEN_PROVENANCE_HELPER" write-submission \
  --output "$SUBMISSION_CONTRACT" \
  --input-provenance "$INPUT_PROVENANCE" \
  --job-id "$job_id" \
  --job-name "$JOB_NAME" \
  --system-id "$SYSTEM_TAG" \
  --output-root "$OUTPUT_ROOT" \
  --record-root "$RECORD_ROOT" \
  --snapshot-root "$SNAPSHOT_ROOT" \
  --submit-output "$SUBMIT_OUTPUT"
"$SCORER_PYTHON" - \
  "$INPUT_PROVENANCE" "$SUBMISSION_CONTRACT" "$RECORD_ROOT/submitted_jobs.tsv" <<'PY'
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

input_path, submission_path, ledger_path = map(Path, sys.argv[1:])
input_payload = json.loads(input_path.read_text(encoding="utf-8"))
submission = json.loads(submission_path.read_text(encoding="utf-8"))

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

resource = submission["resource_contract"]
inputs = input_payload["inputs"]
upstream = input_payload["upstream"]
inference = upstream.get("strict_inference_completion") or {}
selection = upstream.get("final_selection") or {}
submit_output = submission["submit_output"]
fields = [
    "job_name", "job_id", "system_tag", "compute_group",
    "compute_group_name", "spec", "instances", "gpu_type", "gpus",
    "en_input", "en_input_sha256", "zh_input", "zh_input_sha256",
    "source_inference_completion", "source_inference_completion_sha256",
    "source_final_selection", "source_final_selection_sha256", "output_root",
    "snapshot_root", "input_provenance", "input_provenance_sha256",
    "submission_contract", "submission_contract_sha256", "submit_output",
    "submit_output_sha256",
]
row = {
    "job_name": submission["job_name"],
    "job_id": submission["job_id"],
    "system_tag": submission["system_id"],
    "compute_group": resource["compute_group_id"],
    "compute_group_name": resource["compute_group_name"],
    "spec": resource["spec_id"],
    "instances": resource["instances"],
    "gpu_type": resource["gpu_type"],
    "gpus": resource["gpus"],
    "en_input": inputs["en"]["path"],
    "en_input_sha256": inputs["en"]["sha256"],
    "zh_input": inputs["zh"]["path"],
    "zh_input_sha256": inputs["zh"]["sha256"],
    "source_inference_completion": inference.get("path", ""),
    "source_inference_completion_sha256": inference.get("sha256", ""),
    "source_final_selection": selection.get("path", ""),
    "source_final_selection_sha256": selection.get("sha256", ""),
    "output_root": submission["output_root"],
    "snapshot_root": submission["snapshot_root"],
    "input_provenance": str(input_path.resolve()),
    "input_provenance_sha256": sha256(input_path),
    "submission_contract": str(submission_path.resolve()),
    "submission_contract_sha256": sha256(submission_path),
    "submit_output": submit_output["path"],
    "submit_output_sha256": submit_output["sha256"],
}
ledger_path.parent.mkdir(parents=True, exist_ok=True)
temporary = ledger_path.with_name(f".{ledger_path.name}.tmp-{os.getpid()}")
with temporary.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
os.replace(temporary, ledger_path)
PY
echo "[batch42-scorer] submitted job_id=$job_id record=$RECORD_ROOT/submitted_jobs.tsv"

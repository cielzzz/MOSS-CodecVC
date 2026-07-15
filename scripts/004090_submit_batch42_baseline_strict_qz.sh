#!/bin/sh
# Submit one Batch-42 baseline system as an independent strict QZ job.
#
# Default is a platform dry-run.  No live QZ job is created unless DRY_RUN=0
# is supplied explicitly after review.
#
# Supported SYSTEM values:
#   openvoice_v2
#   freevc_v1
#   seed_vc_v2
#   seed_vc_v2_style
#   cosyvoice2_vc
#   vevo_timbre
#
# Each live job is one MTTS-3-2-0715 instance with 8 H200 GPUs.  GPU i runs
# modulo shard i, first on strict EN567 and then on strict ZH1194.  The job
# refuses to publish a completion marker unless every inference row succeeds,
# both 8-shard merges pass, and both 004082 schema-only checks preserve the
# registered denominators.
#
# For a real single-case MTTS gate before a full rerun, set
# BATCH42_BASELINE_SMOKE_ONLY=1 with a dedicated RUN_TAG.  The job audits all
# eight allocated H200s, performs one actual EN forward on GPU 0, requires the
# sole manifest row to be ok, writes SMOKE_COMPLETED.json, and exits before the
# eight-worker full pass.  Submit-side preflight remains schema/runtime dry-run.

set -eu

SELF_PATH=$(readlink -f "$0")
PROJECT_ROOT="${PROJECT_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
QZCLI="${QZCLI:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/qzcli_with_deps.sh}"
QZCLI_HOME="${QZCLI_HOME:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/.codex/qzcli_home}"

ALLOWED_WORKSPACE="ws-8207e9e2-e733-4eec-a475-cfa1c36480ba"
ALLOWED_PROJECT="project-c67c548f-f02c-453b-ba5b-8745db6886e7"
ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"
ALLOWED_SPEC="67b10bc6-78b0-41a3-aaf4-358eeeb99009"
ALLOWED_GPU_TYPE="NVIDIA_H200_SXM_141G"
ALLOWED_INSTANCES="1"
ALLOWED_GPUS_PER_INSTANCE="8"

WORKSPACE="${WORKSPACE:-$ALLOWED_WORKSPACE}"
PROJECT="${PROJECT:-$ALLOWED_PROJECT}"
COMPUTE_GROUP="${COMPUTE_GROUP:-$ALLOWED_COMPUTE_GROUP}"
SPEC="${SPEC:-$ALLOWED_SPEC}"
QZCLI_GPU_TYPE_OVERRIDE="${QZCLI_GPU_TYPE_OVERRIDE:-$ALLOWED_GPU_TYPE}"
INSTANCES="${INSTANCES:-$ALLOWED_INSTANCES}"
GPUS_PER_INSTANCE="${GPUS_PER_INSTANCE:-$ALLOWED_GPUS_PER_INSTANCE}"
NUM_SHARDS="${NUM_SHARDS:-8}"

IMAGE="${IMAGE:-docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420}"
IMAGE_TYPE="${IMAGE_TYPE:-SOURCE_PRIVATE}"
FRAMEWORK="${FRAMEWORK:-pytorch}"
SHM_GI="${SHM_GI:-1200}"
PRIORITY="${PRIORITY:-10}"

SYSTEM="${SYSTEM:-openvoice_v2}"
RUN_TAG="${RUN_TAG:-20260711_mtts}"
DRY_RUN="${DRY_RUN:-1}"
FORCE="${FORCE:-0}"
ENTRYPOINT="${BATCH42_BASELINE_ENTRYPOINT:-0}"
SMOKE_ONLY="${BATCH42_BASELINE_SMOKE_ONLY:-0}"
SMOKE_GATE_JSON="${SMOKE_GATE_JSON:-}"
SMOKE_GATE_SHA256="${SMOKE_GATE_SHA256:-}"

AUDIT_ROOT="$PROJECT_ROOT/testset/outputs/batch42_seedtts_eval_audit_20260711"
DOWNLOAD_ROOT="/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/batch42"
ALLOWED_EN_MANIFEST="$AUDIT_ROOT/official_en_vc_minus_internal320_strict_case.lst"
ALLOWED_ZH_MANIFEST="$AUDIT_ROOT/official_zh_vc_minus_internal320_strict_case.lst"
ALLOWED_EN_INPUT_ROOT="$DOWNLOAD_ROOT/datasets/seed-tts-eval/seedtts_testset/en"
ALLOWED_ZH_INPUT_ROOT="$DOWNLOAD_ROOT/datasets/seed-tts-eval/seedtts_testset/zh"
ALLOWED_EN_MANIFEST_SHA256="48549d8029e680d74656660191c4641ca5a8040ccbe3252ce89bfc3b0c9c75ae"
ALLOWED_ZH_MANIFEST_SHA256="4b637cc1cff33dc369954755538d12396fc92d439a52742103a29b7c563cf6df"
EN_MANIFEST="${EN_MANIFEST:-$ALLOWED_EN_MANIFEST}"
ZH_MANIFEST="${ZH_MANIFEST:-$ALLOWED_ZH_MANIFEST}"
EN_INPUT_ROOT="${EN_INPUT_ROOT:-$ALLOWED_EN_INPUT_ROOT}"
ZH_INPUT_ROOT="${ZH_INPUT_ROOT:-$ALLOWED_ZH_INPUT_ROOT}"

EN_EXPECTED="567"
ZH_EXPECTED="1194"
EN_TEST_SET_ID="seedtts-vc-en-internal320-disjoint"
ZH_TEST_SET_ID="seedtts-vc-zh-internal320-disjoint"

RECORD_ROOT="${RECORD_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/batch42_baseline_strict_${SYSTEM}_${RUN_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/testset/outputs/batch42_baseline_strict_${SYSTEM}_${RUN_TAG}}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-$RECORD_ROOT/snapshot}"
RUNNER="${RUNNER:-$RECORD_ROOT/run_batch42_${SYSTEM}_strict_entrypoint.sh}"
JOB_NAME="${JOB_NAME:-batch42_strict_${SYSTEM}_${RUN_TAG}}"
FINAL_MARKER="$OUTPUT_ROOT/COMPLETED.json"
SMOKE_MARKER="$OUTPUT_ROOT/SMOKE_COMPLETED.json"
SUBMISSION_LOCK="$RECORD_ROOT/.live_submission_lock"
SUBMISSION_LOCK_HELD=0

BASE_PY="/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python"
VC_PY="/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/vc-benchmark/bin/python"
OPENVOICE_PY="$DOWNLOAD_ROOT/envs/openvoice-v2/bin/python"
FREEVC_PY="$DOWNLOAD_ROOT/envs/freevc-v1/bin/python"
BATCH42_DEPS="/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/python_deps/batch42_eval"
COSY_SITE="$DOWNLOAD_ROOT/envs/cosyvoice2-v2.0-py310/site-packages"
VC_LIB="/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/vc-benchmark/lib"
VC_SITE="$VC_LIB/python3.10/site-packages"
COSY_ORT_CAPI="$COSY_SITE/onnxruntime/capi"
VC_PIP_NVIDIA_LD="$VC_SITE/nvidia/cublas/lib:$VC_SITE/nvidia/cuda_cupti/lib:$VC_SITE/nvidia/cuda_nvrtc/lib:$VC_SITE/nvidia/cuda_runtime/lib:$VC_SITE/nvidia/cudnn/lib:$VC_SITE/nvidia/cufft/lib:$VC_SITE/nvidia/curand/lib:$VC_SITE/nvidia/cusolver/lib:$VC_SITE/nvidia/cusparse/lib:$VC_SITE/nvidia/nccl/lib:$VC_SITE/nvidia/nvjitlink/lib:$VC_SITE/nvidia/nvtx/lib"
VC_RUNTIME_LD="$VC_PIP_NVIDIA_LD:$VC_LIB:/usr/local/cuda-12.6/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu"
COSY_RUNTIME_LD="$COSY_ORT_CAPI:$VC_RUNTIME_LD"
TORCH_HOME_PATH="$DOWNLOAD_ROOT/models/torch"
HF_HOME_PATH="$DOWNLOAD_ROOT/models/huggingface"

case "$SYSTEM" in
  openvoice_v2)
    INFERENCE_SCRIPT_NAME="004084_run_batch42_openvoice_freevc.py"
    INTERNAL_SYSTEM_ID="openvoice_v2"
    PUBLIC_SYSTEM_ID="openvoice_v2"
    ;;
  freevc_v1)
    INFERENCE_SCRIPT_NAME="004084_run_batch42_openvoice_freevc.py"
    INTERNAL_SYSTEM_ID="freevc_v1"
    PUBLIC_SYSTEM_ID="freevc_v1"
    ;;
  seed_vc_v2)
    INFERENCE_SCRIPT_NAME="004085_run_batch42_seedvc_cosyvoice.py"
    INTERNAL_SYSTEM_ID="seed_vc_v2"
    PUBLIC_SYSTEM_ID="seed_vc_v2"
    ;;
  seed_vc_v2_style)
    INFERENCE_SCRIPT_NAME="004085_run_batch42_seedvc_cosyvoice.py"
    INTERNAL_SYSTEM_ID="seed_vc_v2"
    PUBLIC_SYSTEM_ID="seed_vc_v2_style"
    ;;
  cosyvoice2_vc)
    INFERENCE_SCRIPT_NAME="004085_run_batch42_seedvc_cosyvoice.py"
    INTERNAL_SYSTEM_ID="cosyvoice2_vc"
    PUBLIC_SYSTEM_ID="cosyvoice2_vc"
    ;;
  vevo_timbre)
    INFERENCE_SCRIPT_NAME="004087_run_batch42_vevo_timbre.py"
    INTERNAL_SYSTEM_ID="vevo_timbre"
    PUBLIC_SYSTEM_ID="vevo_timbre"
    ;;
  *)
    echo "ERROR: unsupported SYSTEM=$SYSTEM" >&2
    exit 2
    ;;
esac

INFERENCE_SCRIPT="$SNAPSHOT_ROOT/scripts/$INFERENCE_SCRIPT_NAME"
MERGE_SCRIPT="$SNAPSHOT_ROOT/scripts/004089_merge_batch42_baseline_shards.py"
SCHEMA_SCRIPT="$SNAPSHOT_ROOT/scripts/004082_run_unified_vc_eval.py"

validate_fixed_configuration() {
  case "$DRY_RUN:$FORCE:$ENTRYPOINT:$SMOKE_ONLY" in
    [01]:[01]:[01]:[01]) ;;
    *)
      echo "ERROR: DRY_RUN, FORCE, BATCH42_BASELINE_ENTRYPOINT, and BATCH42_BASELINE_SMOKE_ONLY must be 0 or 1" >&2
      return 2
      ;;
  esac
  if [ "$WORKSPACE" != "$ALLOWED_WORKSPACE" ] || [ "$PROJECT" != "$ALLOWED_PROJECT" ]; then
    echo "ERROR: Batch-42 is restricted to workspace/project CI-情境智能" >&2
    return 2
  fi
  if [ "$COMPUTE_GROUP" != "$ALLOWED_COMPUTE_GROUP" ]; then
    echo "ERROR: only MTTS-3-2-0715 is allowed; got COMPUTE_GROUP=$COMPUTE_GROUP" >&2
    return 2
  fi
  if [ "$SPEC" != "$ALLOWED_SPEC" ]; then
    echo "ERROR: Batch-42 requires MTTS spec $ALLOWED_SPEC; got SPEC=$SPEC" >&2
    return 2
  fi
  if [ "$INSTANCES" != "$ALLOWED_INSTANCES" ]; then
    echo "ERROR: Batch-42 requires exactly one instance; got INSTANCES=$INSTANCES" >&2
    return 2
  fi
  if [ "$GPUS_PER_INSTANCE" != "$ALLOWED_GPUS_PER_INSTANCE" ]; then
    echo "ERROR: Batch-42 requires 8 GPUs per instance; got GPUS_PER_INSTANCE=$GPUS_PER_INSTANCE" >&2
    return 2
  fi
  if [ "$NUM_SHARDS" != "$ALLOWED_GPUS_PER_INSTANCE" ]; then
    echo "ERROR: Batch-42 strict inference requires NUM_SHARDS=8; got NUM_SHARDS=$NUM_SHARDS" >&2
    return 2
  fi
  if [ "$QZCLI_GPU_TYPE_OVERRIDE" != "$ALLOWED_GPU_TYPE" ]; then
    echo "ERROR: Batch-42 requires GPU type $ALLOWED_GPU_TYPE; got $QZCLI_GPU_TYPE_OVERRIDE" >&2
    return 2
  fi
  if [ "$(readlink -f "$EN_MANIFEST")" != "$(readlink -f "$ALLOWED_EN_MANIFEST")" ] || \
     [ "$(readlink -f "$ZH_MANIFEST")" != "$(readlink -f "$ALLOWED_ZH_MANIFEST")" ]; then
    echo "ERROR: Batch-42 strict jobs must use the registered EN567/ZH1194 strict_case manifests" >&2
    return 2
  fi
  if [ "$(readlink -f "$EN_INPUT_ROOT")" != "$(readlink -f "$ALLOWED_EN_INPUT_ROOT")" ] || \
     [ "$(readlink -f "$ZH_INPUT_ROOT")" != "$(readlink -f "$ALLOWED_ZH_INPUT_ROOT")" ]; then
    echo "ERROR: Batch-42 strict jobs must use the registered Seed-TTS EN/ZH input roots" >&2
    return 2
  fi
}

require_file() {
  if [ ! -s "$1" ]; then
    echo "ERROR: missing or empty file: $1" >&2
    return 1
  fi
}

require_executable() {
  if [ ! -x "$1" ]; then
    echo "ERROR: missing executable: $1" >&2
    return 1
  fi
}

prepare_smoke_gate() {
  if [ -z "$SMOKE_GATE_JSON" ]; then
    if [ -n "$SMOKE_GATE_SHA256" ]; then
      echo "ERROR: SMOKE_GATE_SHA256 requires SMOKE_GATE_JSON" >&2
      return 2
    fi
    return 0
  fi
  require_file "$SMOKE_GATE_JSON"
  SMOKE_GATE_JSON=$(readlink -f "$SMOKE_GATE_JSON")
  actual_sha256=$(sha256sum "$SMOKE_GATE_JSON" | awk '{print $1}')
  if [ "$ENTRYPOINT" = "1" ]; then
    if [ -z "$SMOKE_GATE_SHA256" ]; then
      echo "ERROR: job entrypoint requires the submit-locked SMOKE_GATE_SHA256" >&2
      return 2
    fi
    if [ "$actual_sha256" != "$SMOKE_GATE_SHA256" ]; then
      echo "ERROR: smoke gate SHA256 changed after submission" >&2
      echo "  expected=$SMOKE_GATE_SHA256" >&2
      echo "  actual=$actual_sha256" >&2
      return 2
    fi
  else
    SMOKE_GATE_SHA256="$actual_sha256"
  fi
}

audit_smoke_gate() {
  gate_audit_output=$1
  if [ -z "$SMOKE_GATE_JSON" ]; then
    return 0
  fi
  "$BASE_PY" - \
    "$SMOKE_GATE_JSON" "$SMOKE_GATE_SHA256" "$PUBLIC_SYSTEM_ID" \
    "$gate_audit_output" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

marker_path = Path(sys.argv[1]).resolve()
expected_sha256 = sys.argv[2]
expected_system = sys.argv[3]
audit_path = Path(sys.argv[4])

raw = marker_path.read_bytes()
actual_sha256 = hashlib.sha256(raw).hexdigest()
if actual_sha256 != expected_sha256:
    raise SystemExit(
        f"smoke gate SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
    )
marker = json.loads(raw)
errors = []
if marker.get("schema_version") != "moss_codecvc.batch42_baseline_strict_smoke_completion.v1":
    errors.append(f"unexpected schema_version={marker.get('schema_version')!r}")
if marker.get("status") != "smoke_complete":
    errors.append(f"unexpected status={marker.get('status')!r}")
if marker.get("public_system_id") != expected_system:
    errors.append(
        f"public_system_id={marker.get('public_system_id')!r}, expected {expected_system!r}"
    )
resource = marker.get("resource_contract") or {}
expected_resource = {
    "compute_group": "MTTS-3-2-0715",
    "instances": 1,
    "gpus_per_instance": 8,
    "gpu_type": "NVIDIA_H200_SXM_141G",
}
for key, expected in expected_resource.items():
    if resource.get(key) != expected:
        errors.append(f"resource_contract.{key}={resource.get(key)!r}, expected {expected!r}")
actual_case = marker.get("actual_one_case") or {}
generated = Path(str(actual_case.get("generated_audio") or ""))
if not generated.is_file():
    errors.append(f"generated_audio missing: {generated}")
elif generated.stat().st_size <= 1024:
    errors.append(f"generated_audio too small: {generated.stat().st_size} bytes")
if errors:
    raise SystemExit("invalid smoke gate:\n- " + "\n- ".join(errors))

payload = {
    "schema_version": "moss_codecvc.batch42_smoke_gate_audit.v1",
    "status": "pass",
    "marker_path": str(marker_path),
    "marker_sha256": actual_sha256,
    "public_system_id": expected_system,
    "resource_contract": expected_resource,
    "actual_one_case": {
        "case_id": actual_case.get("case_id"),
        "generated_audio": str(generated.resolve()),
        "output_bytes": generated.stat().st_size,
    },
}
audit_path.parent.mkdir(parents=True, exist_ok=True)
temporary = audit_path.with_name(f".{audit_path.name}.tmp-{os.getpid()}")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, audit_path)
print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
PY
}

release_submission_lock() {
  if [ "$SUBMISSION_LOCK_HELD" = "1" ]; then
    rmdir "$SUBMISSION_LOCK" 2>/dev/null || true
    SUBMISSION_LOCK_HELD=0
  fi
}

audit_strict_inputs() {
  require_file "$EN_MANIFEST"
  require_file "$ZH_MANIFEST"
  if [ ! -d "$EN_INPUT_ROOT" ] || [ ! -d "$ZH_INPUT_ROOT" ]; then
    echo "ERROR: strict Seed-TTS input root is missing" >&2
    return 1
  fi
  "$BASE_PY" - \
    "$EN_MANIFEST" "$EN_INPUT_ROOT" "$EN_EXPECTED" "$ALLOWED_EN_MANIFEST_SHA256" \
    "$ZH_MANIFEST" "$ZH_INPUT_ROOT" "$ZH_EXPECTED" "$ALLOWED_ZH_MANIFEST_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def audit(manifest_arg, root_arg, expected_arg, expected_sha256):
    manifest = Path(manifest_arg).resolve()
    root = Path(root_arg).resolve()
    expected = int(expected_arg)
    rows = []
    missing = []
    malformed = []
    outside = []
    with manifest.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.rstrip("\r\n")
            if not raw:
                continue
            fields = raw.split("|")
            if len(fields) != 5:
                malformed.append((line_number, len(fields)))
                continue
            case_id = fields[0].strip()
            ref = (root / fields[2].strip()).resolve()
            src = (root / fields[4].strip()).resolve()
            for role, path in (("reference", ref), ("source", src)):
                try:
                    path.relative_to(root)
                except ValueError:
                    outside.append((line_number, role, str(path)))
                if not path.is_file():
                    missing.append((line_number, role, str(path)))
            rows.append(case_id)
    if len(rows) != expected:
        raise SystemExit(f"{manifest}: expected {expected} valid rows, got {len(rows)}")
    if malformed:
        raise SystemExit(f"{manifest}: malformed five-column rows: {malformed[:5]}")
    if missing:
        raise SystemExit(f"{manifest}: missing audio: {missing[:5]}")
    if outside:
        raise SystemExit(f"{manifest}: audio outside registered root: {outside[:5]}")
    if len(set(rows)) != len(rows):
        raise SystemExit(f"{manifest}: duplicate case IDs")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise SystemExit(
            f"{manifest}: SHA256 mismatch: expected {expected_sha256}, got {digest}"
        )
    return {
        "manifest": str(manifest),
        "input_root": str(root),
        "rows": len(rows),
        "unique_case_ids": len(set(rows)),
        "sha256": digest,
        "audio_files_checked": len(rows) * 2,
    }

report = {
    "schema_version": "moss_codecvc.batch42_strict_input_audit.v1",
    "en": audit(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]),
    "zh": audit(sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8]),
}
print(json.dumps(report, ensure_ascii=False, indent=2))
PY
}

run_system_python() {
  run_gpu=$1
  shift
  case "$SYSTEM" in
    openvoice_v2)
      env \
        CUDA_VISIBLE_DEVICES="$run_gpu" \
        PYTHONNOUSERSITE=1 \
        PYTHONPATH= \
        TORCH_HOME="$TORCH_HOME_PATH" \
        "$OPENVOICE_PY" "$@"
      ;;
    freevc_v1)
      env \
        CUDA_VISIBLE_DEVICES="$run_gpu" \
        PYTHONNOUSERSITE=1 \
        PYTHONPATH= \
        "$FREEVC_PY" "$@"
      ;;
    seed_vc_v2|seed_vc_v2_style)
      env \
        CUDA_VISIBLE_DEVICES="$run_gpu" \
        PYTHONNOUSERSITE=1 \
        PYTHONPATH= \
        HF_HOME="$HF_HOME_PATH" \
        HUGGINGFACE_HUB_CACHE="$HF_HOME_PATH/hub" \
        TRANSFORMERS_CACHE="$HF_HOME_PATH/transformers" \
        HF_HUB_DISABLE_XET=1 \
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
        TORCH_HOME="$TORCH_HOME_PATH" \
        LD_LIBRARY_PATH="$VC_RUNTIME_LD" \
        "$VC_PY" "$@"
      ;;
    cosyvoice2_vc)
      env \
        CUDA_VISIBLE_DEVICES="$run_gpu" \
        PYTHONNOUSERSITE=1 \
        PYTHONPATH="$COSY_SITE" \
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
        TORCH_HOME="$TORCH_HOME_PATH" \
        LD_LIBRARY_PATH="$COSY_RUNTIME_LD" \
        "$VC_PY" "$@"
      ;;
    vevo_timbre)
      env \
        CUDA_VISIBLE_DEVICES="$run_gpu" \
        PYTHONNOUSERSITE=1 \
        PYTHONPATH="$BATCH42_DEPS" \
        TORCH_HOME="$TORCH_HOME_PATH" \
        LD_LIBRARY_PATH="$VC_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$BASE_PY" "$@"
      ;;
  esac
}

run_inference_case() {
  infer_gpu=$1
  infer_input=$2
  infer_root=$3
  infer_test_set=$4
  infer_output_dir=$5
  infer_manifest=$6
  infer_summary=$7
  infer_shard=$8
  infer_mode=$9
  case "$infer_mode" in
    preflight_dry)
      preflight_run_flags="--dry-run --continue-on-error"
      ;;
    preflight_actual)
      preflight_run_flags="--no-continue-on-error --fail-if-any-error"
      ;;
    full)
      preflight_run_flags=""
      ;;
    *)
      echo "ERROR: unsupported inference mode: $infer_mode" >&2
      return 2
      ;;
  esac

  case "$SYSTEM" in
    openvoice_v2)
      if [ "$infer_mode" != "full" ]; then
        run_system_python "$infer_gpu" "$INFERENCE_SCRIPT" \
          --system openvoice_v2 \
          --input "$infer_input" --input-format lst --input-root "$infer_root" \
          --test-set-id "$infer_test_set" --output-dir "$infer_output_dir" \
          --manifest-jsonl "$infer_manifest" --summary-json "$infer_summary" \
          --num-shards 1 --shard-index 0 --max-cases 1 \
          --no-resume --no-skip-existing $preflight_run_flags \
          --device cuda:0 --openvoice-segmentation upstream_silero_vad \
          --openvoice-silero-short-retry-split-seconds 2.0 \
          --no-openvoice-enable-watermark
      else
        run_system_python "$infer_gpu" "$INFERENCE_SCRIPT" \
          --system openvoice_v2 \
          --input "$infer_input" --input-format lst --input-root "$infer_root" \
          --test-set-id "$infer_test_set" --output-dir "$infer_output_dir" \
          --manifest-jsonl "$infer_manifest" --summary-json "$infer_summary" \
          --num-shards "$NUM_SHARDS" --shard-index "$infer_shard" \
          --resume --skip-existing --continue-on-error --fail-if-any-error \
          --device cuda:0 --openvoice-segmentation upstream_silero_vad \
          --openvoice-silero-short-retry-split-seconds 2.0 \
          --no-openvoice-enable-watermark
      fi
      ;;
    freevc_v1)
      if [ "$infer_mode" != "full" ]; then
        run_system_python "$infer_gpu" "$INFERENCE_SCRIPT" \
          --system freevc_v1 \
          --input "$infer_input" --input-format lst --input-root "$infer_root" \
          --test-set-id "$infer_test_set" --output-dir "$infer_output_dir" \
          --manifest-jsonl "$infer_manifest" --summary-json "$infer_summary" \
          --num-shards 1 --shard-index 0 --max-cases 1 \
          --no-resume --no-skip-existing $preflight_run_flags \
          --device cuda:0
      else
        run_system_python "$infer_gpu" "$INFERENCE_SCRIPT" \
          --system freevc_v1 \
          --input "$infer_input" --input-format lst --input-root "$infer_root" \
          --test-set-id "$infer_test_set" --output-dir "$infer_output_dir" \
          --manifest-jsonl "$infer_manifest" --summary-json "$infer_summary" \
          --num-shards "$NUM_SHARDS" --shard-index "$infer_shard" \
          --resume --skip-existing --continue-on-error --fail-if-any-error \
          --device cuda:0
      fi
      ;;
    seed_vc_v2)
      if [ "$infer_mode" != "full" ]; then
        run_system_python "$infer_gpu" "$INFERENCE_SCRIPT" \
          --system seed_vc_v2 \
          --input "$infer_input" --input-format lst --input-root "$infer_root" \
          --test-set-id "$infer_test_set" --output-dir "$infer_output_dir" \
          --manifest-jsonl "$infer_manifest" --summary-json "$infer_summary" \
          --num-shards 1 --shard-index 0 --limit 1 \
          --no-resume --no-skip-existing $preflight_run_flags \
          --device cuda:0 --offline --no-seed-convert-style --no-seed-disable-cudnn
      else
        run_system_python "$infer_gpu" "$INFERENCE_SCRIPT" \
          --system seed_vc_v2 \
          --input "$infer_input" --input-format lst --input-root "$infer_root" \
          --test-set-id "$infer_test_set" --output-dir "$infer_output_dir" \
          --manifest-jsonl "$infer_manifest" --summary-json "$infer_summary" \
          --num-shards "$NUM_SHARDS" --shard-index "$infer_shard" \
          --resume --skip-existing --continue-on-error --fail-if-any-error \
          --device cuda:0 --offline --no-seed-convert-style --no-seed-disable-cudnn
      fi
      ;;
    seed_vc_v2_style)
      if [ "$infer_mode" != "full" ]; then
        run_system_python "$infer_gpu" "$INFERENCE_SCRIPT" \
          --system seed_vc_v2 \
          --input "$infer_input" --input-format lst --input-root "$infer_root" \
          --test-set-id "$infer_test_set" --output-dir "$infer_output_dir" \
          --manifest-jsonl "$infer_manifest" --summary-json "$infer_summary" \
          --num-shards 1 --shard-index 0 --limit 1 \
          --no-resume --no-skip-existing $preflight_run_flags \
          --device cuda:0 --offline --seed-convert-style --no-seed-disable-cudnn
      else
        run_system_python "$infer_gpu" "$INFERENCE_SCRIPT" \
          --system seed_vc_v2 \
          --input "$infer_input" --input-format lst --input-root "$infer_root" \
          --test-set-id "$infer_test_set" --output-dir "$infer_output_dir" \
          --manifest-jsonl "$infer_manifest" --summary-json "$infer_summary" \
          --num-shards "$NUM_SHARDS" --shard-index "$infer_shard" \
          --resume --skip-existing --continue-on-error --fail-if-any-error \
          --device cuda:0 --offline --seed-convert-style --no-seed-disable-cudnn
      fi
      ;;
    cosyvoice2_vc)
      if [ "$infer_mode" != "full" ]; then
        run_system_python "$infer_gpu" "$INFERENCE_SCRIPT" \
          --system cosyvoice2_vc \
          --input "$infer_input" --input-format lst --input-root "$infer_root" \
          --test-set-id "$infer_test_set" --output-dir "$infer_output_dir" \
          --manifest-jsonl "$infer_manifest" --summary-json "$infer_summary" \
          --num-shards 1 --shard-index 0 --limit 1 \
          --no-resume --no-skip-existing $preflight_run_flags \
          --device cuda:0 --offline --no-cosy-fp16 \
          --cosy-speech-tokenizer-provider cuda
      else
        run_system_python "$infer_gpu" "$INFERENCE_SCRIPT" \
          --system cosyvoice2_vc \
          --input "$infer_input" --input-format lst --input-root "$infer_root" \
          --test-set-id "$infer_test_set" --output-dir "$infer_output_dir" \
          --manifest-jsonl "$infer_manifest" --summary-json "$infer_summary" \
          --num-shards "$NUM_SHARDS" --shard-index "$infer_shard" \
          --resume --skip-existing --continue-on-error --fail-if-any-error \
          --device cuda:0 --offline --no-cosy-fp16 \
          --cosy-speech-tokenizer-provider cuda
      fi
      ;;
    vevo_timbre)
      if [ "$infer_mode" != "full" ]; then
        run_system_python "$infer_gpu" "$INFERENCE_SCRIPT" \
          --input "$infer_input" --input-format lst --input-root "$infer_root" \
          --test-set-id "$infer_test_set" --output-dir "$infer_output_dir" \
          --manifest-jsonl "$infer_manifest" --summary-json "$infer_summary" \
          --num-shards 1 --shard-index 0 --max-cases 1 \
          --no-resume --no-skip-existing $preflight_run_flags \
          --device cuda:0 --flow-matching-steps 32 --target-db -25 \
          --verify-checkpoint-sha256
      else
        run_system_python "$infer_gpu" "$INFERENCE_SCRIPT" \
          --input "$infer_input" --input-format lst --input-root "$infer_root" \
          --test-set-id "$infer_test_set" --output-dir "$infer_output_dir" \
          --manifest-jsonl "$infer_manifest" --summary-json "$infer_summary" \
          --num-shards "$NUM_SHARDS" --shard-index "$infer_shard" \
          --resume --skip-existing --continue-on-error --fail-if-any-error \
          --device cuda:0 --flow-matching-steps 32 --target-db -25
      fi
      ;;
  esac
}

runtime_preflight() {
  preflight_root=$1
  mkdir -p "$preflight_root"
  if [ "$ENTRYPOINT" = "1" ]; then
    preflight_mode="preflight_actual"
  else
    preflight_mode="preflight_dry"
  fi

  case "$SYSTEM" in
    openvoice_v2)
      run_system_python 0 -c \
        'import json, os; from pathlib import Path; import torch, whisper_timestamped; p=Path(os.environ["TORCH_HOME"])/"hub/snakers4_silero-vad_master"; assert (p/"hubconf.py").is_file(), p; assert (p/"src/silero_vad/data/silero_vad.jit").is_file(), p; print(json.dumps({"torch_home": os.environ["TORCH_HOME"], "torch_hub_dir": torch.hub.get_dir(), "silero_local_cache": str(p), "silero_local_ready": True}, sort_keys=True))' \
        > "$preflight_root/accelerator_runtime.json"
      ;;
    seed_vc_v2|seed_vc_v2_style)
      run_system_python 0 -c \
        'import json, os, torch; assert torch.cuda.is_available(); assert torch.backends.cudnn.enabled; layer=torch.nn.Conv1d(4, 4, 3, padding=1).cuda(); out=layer(torch.zeros(1,4,32,device="cuda")); torch.cuda.synchronize(); print(json.dumps({"torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_device": torch.cuda.get_device_name(0), "cudnn_version": torch.backends.cudnn.version(), "cudnn_enabled": torch.backends.cudnn.enabled, "conv1d_shape": list(out.shape), "ld_library_path": os.environ.get("LD_LIBRARY_PATH", "")}, sort_keys=True))' \
        > "$preflight_root/accelerator_runtime.json"
      ;;
    cosyvoice2_vc)
      run_system_python 0 -c \
        'import json, os, pyworld, hyperpyyaml, onnxruntime, torch; providers=onnxruntime.get_available_providers(); assert "CUDAExecutionProvider" in providers, providers; assert torch.cuda.is_available(); print(json.dumps({"onnxruntime": onnxruntime.__version__, "onnxruntime_file": onnxruntime.__file__, "providers": providers, "torch": torch.__version__, "cuda_device": torch.cuda.get_device_name(0), "cudnn_version": torch.backends.cudnn.version(), "ld_library_path": os.environ.get("LD_LIBRARY_PATH", "")}, sort_keys=True))' \
        > "$preflight_root/accelerator_runtime.json"
      ;;
  esac

  run_inference_case \
    0 "$EN_MANIFEST" "$EN_INPUT_ROOT" "$EN_TEST_SET_ID" \
    "$preflight_root" "$preflight_root/manifest.jsonl" \
    "$preflight_root/summary.json" 0 "$preflight_mode"

  "$BASE_PY" - \
    "$preflight_root/runtime_audit.json" \
    "$preflight_root/summary.json" \
    "$SYSTEM" "$preflight_mode" <<'PY'
import json
import sys
from pathlib import Path

audit_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
selector = sys.argv[3]
mode = sys.argv[4]
audit = json.loads(audit_path.read_text(encoding="utf-8"))
summary = json.loads(summary_path.read_text(encoding="utf-8"))
errors = []
if audit.get("ready") is not True:
    errors.append(f"runtime audit not ready: {audit.get('blocking_reasons') or audit}")
if summary.get("runtime_ready") is not True:
    errors.append(f"summary runtime_ready={summary.get('runtime_ready')!r}")
if summary.get("selected_valid_cases") != 1:
    errors.append(f"selected_valid_cases={summary.get('selected_valid_cases')!r}")
if summary.get("selected_input_errors") != 0:
    errors.append(f"selected_input_errors={summary.get('selected_input_errors')!r}")
expected_status = "complete" if mode == "preflight_actual" else "dry_run_complete"
expected_action = "ok" if mode == "preflight_actual" else "dry_run"
expected_counts = {expected_action: 1}
if summary.get("status") != expected_status:
    errors.append(f"summary status={summary.get('status')!r}, expected {expected_status!r}")
if summary.get("manifest_status_counts") != expected_counts:
    errors.append(
        "manifest_status_counts="
        f"{summary.get('manifest_status_counts')!r}, expected {expected_counts!r}"
    )
if summary.get("run_action_counts") != expected_counts:
    errors.append(
        f"run_action_counts={summary.get('run_action_counts')!r}, "
        f"expected {expected_counts!r}"
    )
if selector in {"seed_vc_v2", "seed_vc_v2_style"}:
    actual = (summary.get("inference_config") or {}).get("convert_style")
    expected = selector == "seed_vc_v2_style"
    if actual is not expected:
        errors.append(f"Seed-VC convert_style={actual!r}, expected {expected!r}")
    if (summary.get("inference_config") or {}).get("disable_cudnn") is not False:
        errors.append("Seed-VC official preflight must keep cuDNN enabled")
if selector == "openvoice_v2":
    config = summary.get("inference_config") or {}
    if config.get("speaker_embedding_segmentation") != "upstream_silero_vad":
        errors.append(f"OpenVoice segmentation not registered: {config!r}")
    if config.get("upstream_silero_vad") is not True:
        errors.append(f"OpenVoice upstream_silero_vad not true: {config!r}")
    if config.get("upstream_silero_short_audio_retry_split_seconds") != 2.0:
        errors.append(f"OpenVoice short-audio Silero retry not 2.0 s: {config!r}")
if selector == "cosyvoice2_vc":
    config = summary.get("inference_config") or {}
    if config.get("speech_tokenizer_onnx_provider") != "cuda":
        errors.append(f"Cosy speech tokenizer provider not cuda: {config!r}")
if errors:
    raise SystemExit("\n".join(errors))
print(f"runtime preflight ready: selector={selector} mode={mode} audit={audit_path}")
PY
}

snapshot_scripts() {
  mkdir -p "$SNAPSHOT_ROOT/scripts" "$RECORD_ROOT" "$OUTPUT_ROOT"
  for snapshot_name in \
    004082_run_unified_vc_eval.py \
    004084_run_batch42_openvoice_freevc.py \
    004085_run_batch42_seedvc_cosyvoice.py \
    004087_run_batch42_vevo_timbre.py \
    004089_merge_batch42_baseline_shards.py; do
    require_file "$PROJECT_ROOT/scripts/$snapshot_name"
    cp "$PROJECT_ROOT/scripts/$snapshot_name" "$SNAPSHOT_ROOT/scripts/$snapshot_name"
    chmod 0555 "$SNAPSHOT_ROOT/scripts/$snapshot_name"
  done
  cp "$SELF_PATH" "$RUNNER"
  chmod 0555 "$RUNNER"
  sh -n "$RUNNER"
  sha256sum \
    "$SNAPSHOT_ROOT/scripts/004082_run_unified_vc_eval.py" \
    "$SNAPSHOT_ROOT/scripts/004084_run_batch42_openvoice_freevc.py" \
    "$SNAPSHOT_ROOT/scripts/004085_run_batch42_seedvc_cosyvoice.py" \
    "$SNAPSHOT_ROOT/scripts/004087_run_batch42_vevo_timbre.py" \
    "$SNAPSHOT_ROOT/scripts/004089_merge_batch42_baseline_shards.py" \
    "$RUNNER" > "$RECORD_ROOT/sha256sums.txt"
  sha256sum "$EN_MANIFEST" "$ZH_MANIFEST" > "$RECORD_ROOT/strict_manifest_sha256sums.txt"
}

write_submission_plan() {
  {
    printf 'job_name\tsystem\tinternal_system_id\tcompute_group\tspec\tgpu_type\tinstances\tgpus_per_instance\tshards\ten_cases\tzh_cases\tsmoke_only\tsmoke_gate_json\tsmoke_gate_sha256\toutput_root\tentrypoint\n'
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$JOB_NAME" "$PUBLIC_SYSTEM_ID" "$INTERNAL_SYSTEM_ID" \
      "$COMPUTE_GROUP" "$SPEC" "$QZCLI_GPU_TYPE_OVERRIDE" \
      "$INSTANCES" "$GPUS_PER_INSTANCE" "$NUM_SHARDS" \
      "$EN_EXPECTED" "$ZH_EXPECTED" "$SMOKE_ONLY" \
      "$SMOKE_GATE_JSON" "$SMOKE_GATE_SHA256" "$OUTPUT_ROOT" "$RUNNER"
  } > "$RECORD_ROOT/submission_plan.tsv"
}

audit_allocated_gpus() {
  allocated_count=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l | tr -d ' ')
  if [ "$allocated_count" != "$ALLOWED_GPUS_PER_INSTANCE" ]; then
    echo "ERROR: expected exactly 8 allocated GPUs, got $allocated_count" >&2
    return 1
  fi
  if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -Evq 'H200'; then
    echo "ERROR: one or more allocated GPUs are not H200" >&2
    return 1
  fi
  nvidia-smi
}

run_language_shard() {
  shard_index=$1
  language_id=$2
  if [ "$language_id" = "en" ]; then
    shard_input="$EN_MANIFEST"
    shard_root="$EN_INPUT_ROOT"
    shard_test_set="$EN_TEST_SET_ID"
  else
    shard_input="$ZH_MANIFEST"
    shard_root="$ZH_INPUT_ROOT"
    shard_test_set="$ZH_TEST_SET_ID"
  fi
  shard_output="$OUTPUT_ROOT/$language_id"
  shard_tag=$(printf '%05d' "$shard_index")
  shard_manifest="$shard_output/manifest.shard-${shard_tag}-of-00008.jsonl"
  shard_summary="$shard_output/summary.shard-${shard_tag}-of-00008.json"
  mkdir -p "$shard_output"
  run_inference_case \
    "$shard_index" "$shard_input" "$shard_root" "$shard_test_set" \
    "$shard_output" "$shard_manifest" "$shard_summary" \
    "$shard_index" full
}

run_all_workers() {
  mkdir -p "$RECORD_ROOT/worker_logs" "$OUTPUT_ROOT/en" "$OUTPUT_ROOT/zh"
  worker_pids=""
  shard=0
  while [ "$shard" -lt "$NUM_SHARDS" ]; do
    (
      echo "[worker-$shard] start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "[worker-$shard] physical_gpu=$shard"
      run_language_shard "$shard" en
      run_language_shard "$shard" zh
      echo "[worker-$shard] complete=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    ) > "$RECORD_ROOT/worker_logs/shard-$(printf '%05d' "$shard").log" 2>&1 &
    worker_pids="$worker_pids $!"
    shard=$((shard + 1))
  done

  worker_failed=0
  for worker_pid in $worker_pids; do
    if ! wait "$worker_pid"; then
      echo "ERROR: worker pid $worker_pid failed; preserving all shard ledgers" >&2
      worker_failed=1
    fi
  done
  if [ "$worker_failed" != "0" ]; then
    return 1
  fi
}

merge_one_language() {
  merge_lang=$1
  if [ "$merge_lang" = "en" ]; then
    merge_expected="$EN_EXPECTED"
    merge_test_set="$EN_TEST_SET_ID"
  else
    merge_expected="$ZH_EXPECTED"
    merge_test_set="$ZH_TEST_SET_ID"
  fi
  merge_root="$OUTPUT_ROOT/$merge_lang"
  "$BASE_PY" "$MERGE_SCRIPT" \
    --input "$merge_root/manifest.shard-00000-of-00008.jsonl" \
    --input "$merge_root/manifest.shard-00001-of-00008.jsonl" \
    --input "$merge_root/manifest.shard-00002-of-00008.jsonl" \
    --input "$merge_root/manifest.shard-00003-of-00008.jsonl" \
    --input "$merge_root/manifest.shard-00004-of-00008.jsonl" \
    --input "$merge_root/manifest.shard-00005-of-00008.jsonl" \
    --input "$merge_root/manifest.shard-00006-of-00008.jsonl" \
    --input "$merge_root/manifest.shard-00007-of-00008.jsonl" \
    --merged-manifest "$merge_root/merged_manifest.jsonl" \
    --successful-jsonl "$merge_root/successful.jsonl" \
    --summary-json "$merge_root/merge_summary.json" \
    --expected-shards "$NUM_SHARDS" \
    --expected-cases "$merge_expected" \
    --system-id "$INTERNAL_SYSTEM_ID" \
    --test-set-id "$merge_test_set" \
    --require-all-ok
}

schema_check_one_language() {
  schema_lang=$1
  if [ "$schema_lang" = "en" ]; then
    schema_expected="$EN_EXPECTED"
    schema_test_set="$EN_TEST_SET_ID"
  else
    schema_expected="$ZH_EXPECTED"
    schema_test_set="$ZH_TEST_SET_ID"
  fi
  schema_root="$OUTPUT_ROOT/$schema_lang/schema"
  schema_stem="${PUBLIC_SYSTEM_ID}_${schema_lang}_strict"
  mkdir -p "$schema_root"
  env PYTHONPATH="$BATCH42_DEPS" "$BASE_PY" "$SCHEMA_SCRIPT" evaluate \
    --input "$OUTPUT_ROOT/$schema_lang/successful.jsonl" \
    --output-dir "$schema_root" \
    --output-stem "$schema_stem" \
    --run-id "batch42_${PUBLIC_SYSTEM_ID}_${schema_lang}_${RUN_TAG}" \
    --system-id "$PUBLIC_SYSTEM_ID" \
    --test-set-id "$schema_test_set" \
    --input-profile official_seedtts_vc \
    --metric-profile seedtts_official \
    --schema-only
  schema_jsonl="$schema_root/${schema_stem}.unified_eval.jsonl"
  schema_rows=$(wc -l < "$schema_jsonl" | tr -d ' ')
  if [ "$schema_rows" != "$schema_expected" ]; then
    echo "ERROR: schema-only $schema_lang expected $schema_expected rows, got $schema_rows" >&2
    return 1
  fi
}

write_completion_marker() {
  "$BASE_PY" - \
    "$OUTPUT_ROOT" "$RECORD_ROOT" "$PUBLIC_SYSTEM_ID" "$INTERNAL_SYSTEM_ID" "$RUN_TAG" \
    "$OUTPUT_ROOT/en/merge_summary.json" \
    "$OUTPUT_ROOT/zh/merge_summary.json" \
    "$OUTPUT_ROOT/en/schema/${PUBLIC_SYSTEM_ID}_en_strict.unified_eval.jsonl" \
    "$OUTPUT_ROOT/zh/schema/${PUBLIC_SYSTEM_ID}_zh_strict.unified_eval.jsonl" \
    "$FINAL_MARKER" "$SMOKE_GATE_JSON" "$SMOKE_GATE_SHA256" \
    "$RECORD_ROOT/smoke_gate_audit.complete.json" <<'PY'
import datetime
import json
import os
import sys
from pathlib import Path

(
    output_root,
    record_root,
    public_system,
    internal_system,
    run_tag,
    en_merge_path,
    zh_merge_path,
    en_schema_path,
    zh_schema_path,
    marker_path,
    smoke_gate_path,
    smoke_gate_sha256,
    smoke_gate_audit_path,
) = sys.argv[1:]

en_merge = json.loads(Path(en_merge_path).read_text(encoding="utf-8"))
zh_merge = json.loads(Path(zh_merge_path).read_text(encoding="utf-8"))
if en_merge.get("all_ok") is not True or en_merge.get("rows") != 567:
    raise SystemExit(f"invalid EN merge summary: {en_merge}")
if zh_merge.get("all_ok") is not True or zh_merge.get("rows") != 1194:
    raise SystemExit(f"invalid ZH merge summary: {zh_merge}")

def audit_schema(path_arg, expected, expected_system):
    path = Path(path_arg)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                rows.append(json.loads(raw))
    if len(rows) != expected:
        raise SystemExit(f"{path}: expected {expected} rows, got {len(rows)}")
    systems = {row.get("system_id") for row in rows}
    if systems != {expected_system}:
        raise SystemExit(f"{path}: unexpected system IDs {systems}")
    return str(path.resolve())

smoke_gate = None
if smoke_gate_path:
    gate_audit = json.loads(Path(smoke_gate_audit_path).read_text(encoding="utf-8"))
    if gate_audit.get("status") != "pass":
        raise SystemExit(f"invalid final smoke gate audit: {gate_audit}")
    if gate_audit.get("marker_sha256") != smoke_gate_sha256:
        raise SystemExit(f"final smoke gate SHA mismatch: {gate_audit}")
    smoke_gate = {
        "marker_path": str(Path(smoke_gate_path).resolve()),
        "marker_sha256": smoke_gate_sha256,
        "audit_json": str(Path(smoke_gate_audit_path).resolve()),
        "actual_one_case": gate_audit.get("actual_one_case"),
    }

payload = {
    "schema_version": "moss_codecvc.batch42_baseline_strict_completion.v1",
    "status": "complete",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "public_system_id": public_system,
    "internal_runner_system_id": internal_system,
    "run_tag": run_tag,
    "resource_contract": {
        "compute_group": "MTTS-3-2-0715",
        "instances": 1,
        "gpus_per_instance": 8,
        "gpu_type": "NVIDIA_H200_SXM_141G",
    },
    "smoke_gate": smoke_gate,
    "strict_sets": {
        "en": {
            "registered_cases": 567,
            "merge_summary": str(Path(en_merge_path).resolve()),
            "schema_jsonl": audit_schema(en_schema_path, 567, public_system),
        },
        "zh": {
            "registered_cases": 1194,
            "merge_summary": str(Path(zh_merge_path).resolve()),
            "schema_jsonl": audit_schema(zh_schema_path, 1194, public_system),
        },
    },
    "output_root": str(Path(output_root).resolve()),
    "record_root": str(Path(record_root).resolve()),
}
marker = Path(marker_path)
marker.parent.mkdir(parents=True, exist_ok=True)
temporary = marker.with_name(f".{marker.name}.tmp-{os.getpid()}")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, marker)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
  cp "$FINAL_MARKER" "$RECORD_ROOT/completion.json"
}

write_smoke_completion_marker() {
  smoke_root="$RECORD_ROOT/preflight_job"
  "$BASE_PY" - \
    "$smoke_root/manifest.jsonl" \
    "$smoke_root/summary.json" \
    "$SMOKE_MARKER" \
    "$PUBLIC_SYSTEM_ID" "$INTERNAL_SYSTEM_ID" \
    "$OUTPUT_ROOT" "$RECORD_ROOT" <<'PY'
import datetime as dt
import json
import os
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
marker_path = Path(sys.argv[3])
public_system = sys.argv[4]
internal_system = sys.argv[5]
output_root = Path(sys.argv[6])
record_root = Path(sys.argv[7])

rows = [
    json.loads(line)
    for line in manifest_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if len(rows) != 1 or rows[0].get("status") != "ok":
    raise SystemExit(f"actual smoke must contain exactly one ok row; got {rows!r}")
if summary.get("manifest_status_counts") != {"ok": 1}:
    raise SystemExit(f"invalid smoke summary counts: {summary!r}")
generated = Path(str(rows[0].get("generated_audio") or ""))
if not generated.is_file() or generated.stat().st_size <= 0:
    raise SystemExit(f"actual smoke generated audio is missing/empty: {generated}")
backend_details = rows[0].get("backend_details") or {}
inference_config = summary.get("inference_config") or {}
if public_system == "openvoice_v2":
    if backend_details.get("upstream_silero_vad") is not True:
        raise SystemExit(f"OpenVoice smoke did not use upstream Silero: {backend_details}")
    if int(backend_details.get("upstream_silero_calls") or 0) < 1:
        raise SystemExit(f"OpenVoice smoke registered no upstream Silero call: {backend_details}")
    if backend_details.get("upstream_silero_short_audio_retry_split_seconds") != 2.0:
        raise SystemExit(f"OpenVoice smoke retry split is not 2.0 s: {backend_details}")
if public_system.startswith("seed_vc_v2"):
    if inference_config.get("disable_cudnn") is not False:
        raise SystemExit(f"Seed-VC smoke disabled cuDNN: {inference_config}")
    if backend_details.get("cudnn_enabled") is not True:
        raise SystemExit(f"Seed-VC smoke did not execute with cuDNN enabled: {backend_details}")
if public_system == "cosyvoice2_vc":
    providers = backend_details.get("speech_tokenizer_providers_actual") or []
    if not providers or providers[0] != "CUDAExecutionProvider":
        raise SystemExit(f"CosyVoice smoke did not use CUDA EP: {backend_details}")

accelerator_runtime_path = record_root / "preflight_job/accelerator_runtime.json"
accelerator_runtime = json.loads(
    accelerator_runtime_path.read_text(encoding="utf-8")
)

payload = {
    "schema_version": "moss_codecvc.batch42_baseline_strict_smoke_completion.v1",
    "status": "smoke_complete",
    "smoke_only": True,
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "public_system_id": public_system,
    "internal_runner_system_id": internal_system,
    "resource_contract": {
        "compute_group": "MTTS-3-2-0715",
        "instances": 1,
        "gpus_per_instance": 8,
        "gpu_type": "NVIDIA_H200_SXM_141G",
    },
    "actual_one_case": {
        "case_id": rows[0].get("case_id"),
        "generated_audio": str(generated.resolve()),
        "output_bytes": generated.stat().st_size,
        "runtime_seconds": rows[0].get("runtime_seconds"),
        "backend_details": backend_details,
        "manifest_jsonl": str(manifest_path.resolve()),
        "summary_json": str(summary_path.resolve()),
        "inference_config": inference_config,
        "accelerator_runtime": accelerator_runtime,
        "accelerator_runtime_json": str(accelerator_runtime_path.resolve()),
    },
    "output_root": str(output_root.resolve()),
    "record_root": str(record_root.resolve()),
}
marker_path.parent.mkdir(parents=True, exist_ok=True)
temporary = marker_path.with_name(f".{marker_path.name}.tmp-{os.getpid()}")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, marker_path)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
  cp "$SMOKE_MARKER" "$RECORD_ROOT/smoke_completion.json"
}

run_job_entrypoint() {
  mkdir -p "$RECORD_ROOT" "$OUTPUT_ROOT"
  exec >> "$RECORD_ROOT/run.log" 2>&1
  echo "[batch42-strict] start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[batch42-strict] system=$SYSTEM internal_system=$INTERNAL_SYSTEM_ID"
  echo "[batch42-strict] smoke_only=$SMOKE_ONLY"
  echo "[batch42-strict] output_root=$OUTPUT_ROOT"
  echo "[batch42-strict] snapshot_root=$SNAPSHOT_ROOT"

  require_file "$INFERENCE_SCRIPT"
  require_file "$MERGE_SCRIPT"
  require_file "$SCHEMA_SCRIPT"
  audit_strict_inputs > "$RECORD_ROOT/input_audit.job.json"
  if [ -n "$SMOKE_GATE_JSON" ]; then
    audit_smoke_gate "$RECORD_ROOT/smoke_gate_audit.job.json"
  fi
  audit_allocated_gpus
  runtime_preflight "$RECORD_ROOT/preflight_job"
  if [ "$SMOKE_ONLY" = "1" ]; then
    write_smoke_completion_marker
    echo "[batch42-strict] smoke_complete=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[batch42-strict] smoke_marker=$SMOKE_MARKER"
    return 0
  fi
  run_all_workers
  merge_one_language en
  merge_one_language zh
  schema_check_one_language en
  schema_check_one_language zh
  if [ -n "$SMOKE_GATE_JSON" ]; then
    audit_smoke_gate "$RECORD_ROOT/smoke_gate_audit.complete.json"
  fi
  write_completion_marker
  echo "[batch42-strict] complete=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[batch42-strict] marker=$FINAL_MARKER"
}

validate_fixed_configuration
prepare_smoke_gate

if [ "$ENTRYPOINT" = "1" ]; then
  run_job_entrypoint
  exit 0
fi

require_executable "$QZCLI"
require_executable "$BASE_PY"
require_executable "$VC_PY"
case "$SYSTEM" in
  openvoice_v2) require_executable "$OPENVOICE_PY" ;;
  freevc_v1) require_executable "$FREEVC_PY" ;;
esac

mkdir -p "$RECORD_ROOT" "$OUTPUT_ROOT" "$QZCLI_HOME"
if [ "$DRY_RUN" = "0" ]; then
  if ! mkdir "$SUBMISSION_LOCK" 2>/dev/null; then
    echo "ERROR: another live submission attempt holds $SUBMISSION_LOCK" >&2
    exit 1
  fi
  SUBMISSION_LOCK_HELD=1
  trap release_submission_lock EXIT
  trap 'release_submission_lock; exit 130' INT
  trap 'release_submission_lock; exit 143' TERM
  if [ "$FORCE" != "1" ]; then
    if [ -s "$RECORD_ROOT/submitted_jobs.tsv" ] || \
       [ -s "$FINAL_MARKER" ] || [ -s "$SMOKE_MARKER" ]; then
      echo "ERROR: existing live submission/completion for $SYSTEM; use FORCE=1 only for an intentional rerun" >&2
      exit 1
    fi
  fi
fi

snapshot_scripts
audit_strict_inputs > "$RECORD_ROOT/input_audit.submit.json"
if [ -n "$SMOKE_GATE_JSON" ]; then
  audit_smoke_gate "$RECORD_ROOT/smoke_gate_audit.submit.json"
fi
runtime_preflight "$RECORD_ROOT/preflight_submit"
write_submission_plan

COMMAND="env BATCH42_BASELINE_ENTRYPOINT=1 BATCH42_BASELINE_SMOKE_ONLY=$SMOKE_ONLY SYSTEM=$SYSTEM RUN_TAG=$RUN_TAG FORCE=$FORCE PROJECT_ROOT=$PROJECT_ROOT RECORD_ROOT=$RECORD_ROOT OUTPUT_ROOT=$OUTPUT_ROOT SNAPSHOT_ROOT=$SNAPSHOT_ROOT RUNNER=$RUNNER EN_MANIFEST=$EN_MANIFEST ZH_MANIFEST=$ZH_MANIFEST EN_INPUT_ROOT=$EN_INPUT_ROOT ZH_INPUT_ROOT=$ZH_INPUT_ROOT SMOKE_GATE_JSON=$SMOKE_GATE_JSON SMOKE_GATE_SHA256=$SMOKE_GATE_SHA256 WORKSPACE=$WORKSPACE PROJECT=$PROJECT COMPUTE_GROUP=$COMPUTE_GROUP SPEC=$SPEC INSTANCES=$INSTANCES GPUS_PER_INSTANCE=$GPUS_PER_INSTANCE NUM_SHARDS=$NUM_SHARDS QZCLI_GPU_TYPE_OVERRIDE=$QZCLI_GPU_TYPE_OVERRIDE sh $RUNNER"
SUBMIT_OUTPUT="$RECORD_ROOT/submit_output.txt"

echo "=========================================="
echo "Batch-42 strict baseline QZ plan"
echo "  SYSTEM=$SYSTEM"
echo "  INTERNAL_SYSTEM_ID=$INTERNAL_SYSTEM_ID"
echo "  JOB_NAME=$JOB_NAME"
echo "  SMOKE_ONLY=$SMOKE_ONLY"
echo "  SMOKE_GATE_JSON=${SMOKE_GATE_JSON:-none}"
echo "  SMOKE_GATE_SHA256=${SMOKE_GATE_SHA256:-none}"
echo "  STRICT_EN=$EN_EXPECTED STRICT_ZH=$ZH_EXPECTED"
echo "  WORKERS=8 (one modulo shard per H200; EN then ZH)"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP (MTTS-3-2-0715 only)"
echo "  SPEC=$SPEC"
echo "  INSTANCES=$INSTANCES"
echo "  GPU_TYPE=$QZCLI_GPU_TYPE_OVERRIDE"
echo "  OUTPUT_ROOT=$OUTPUT_ROOT"
echo "  RECORD_ROOT=$RECORD_ROOT"
echo "  SNAPSHOT_ROOT=$SNAPSHOT_ROOT"
echo "  DRY_RUN=$DRY_RUN"
echo "  COMMAND=$COMMAND"
echo "=========================================="

set +e
if [ "$DRY_RUN" = "1" ]; then
  qz_output=$(
    env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
      HOME="$QZCLI_HOME" \
      QZCLI_GPU_TYPE_OVERRIDE="$QZCLI_GPU_TYPE_OVERRIDE" \
      "$QZCLI" create-job \
        --name "$JOB_NAME" \
        --command "$COMMAND" \
        --workspace "$WORKSPACE" \
        --project "$PROJECT" \
        --compute-group "$COMPUTE_GROUP" \
        --spec "$SPEC" \
        --image "$IMAGE" \
        --image-type "$IMAGE_TYPE" \
        --instances "$INSTANCES" \
        --shm "$SHM_GI" \
        --priority "$PRIORITY" \
        --framework "$FRAMEWORK" \
        --dry-run 2>&1
  )
  qz_status=$?
else
  qz_output=$(
    env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
      HOME="$QZCLI_HOME" \
      QZCLI_GPU_TYPE_OVERRIDE="$QZCLI_GPU_TYPE_OVERRIDE" \
      "$QZCLI" create-job \
        --name "$JOB_NAME" \
        --command "$COMMAND" \
        --workspace "$WORKSPACE" \
        --project "$PROJECT" \
        --compute-group "$COMPUTE_GROUP" \
        --spec "$SPEC" \
        --image "$IMAGE" \
        --image-type "$IMAGE_TYPE" \
        --instances "$INSTANCES" \
        --shm "$SHM_GI" \
        --priority "$PRIORITY" \
        --framework "$FRAMEWORK" 2>&1
  )
  qz_status=$?
fi
set -e

printf '%s\n' "$qz_output" > "$SUBMIT_OUTPUT"
printf '%s\n' "$qz_output"
if [ "$qz_status" -ne 0 ]; then
  echo "ERROR: QZ create-job command failed; see $SUBMIT_OUTPUT" >&2
  exit "$qz_status"
fi

if [ "$DRY_RUN" = "1" ]; then
  {
    printf 'job_name\tsystem\tcompute_group\tspec\tinstances\tgpu_type\tsmoke_only\tsmoke_gate_json\tsmoke_gate_sha256\tstatus\n'
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tdry_run_only\n' \
      "$JOB_NAME" "$PUBLIC_SYSTEM_ID" "$COMPUTE_GROUP" "$SPEC" \
      "$INSTANCES" "$QZCLI_GPU_TYPE_OVERRIDE" "$SMOKE_ONLY" \
      "$SMOKE_GATE_JSON" "$SMOKE_GATE_SHA256"
  } > "$RECORD_ROOT/dry_run_jobs.tsv"
  echo "[batch42-strict] dry-run passed; no QZ job submitted"
  exit 0
fi

job_id=$(
  printf '%s\n' "$qz_output" |
    grep -Eo 'job-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}' |
    head -n 1 || true
)
case "$job_id" in
  job-????????-????-????-????-????????????) ;;
  *)
    echo "ERROR: QZ returned success but no valid job ID was parsed" >&2
    exit 1
    ;;
esac
{
  printf 'job_name\tjob_id\tsystem\tcompute_group\tspec\tinstances\tgpu_type\tsmoke_only\tsmoke_gate_json\tsmoke_gate_sha256\toutput_root\trecord_root\tentrypoint\n'
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$JOB_NAME" "$job_id" "$PUBLIC_SYSTEM_ID" "$COMPUTE_GROUP" "$SPEC" \
    "$INSTANCES" "$QZCLI_GPU_TYPE_OVERRIDE" "$SMOKE_ONLY" \
    "$SMOKE_GATE_JSON" "$SMOKE_GATE_SHA256" \
    "$OUTPUT_ROOT" "$RECORD_ROOT" "$RUNNER"
} > "$RECORD_ROOT/submitted_jobs.tsv"
echo "[batch42-strict] submitted job_id=$job_id record=$RECORD_ROOT/submitted_jobs.tsv"

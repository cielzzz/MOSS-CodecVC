#!/usr/bin/env bash
# Submit the registered Batch-33 / Path X 3k checkpoint on the strict
# Batch-42 EN567 and ZH1194 VC sets.
#
# Two explicit modes are supported:
#   MODE=smoke  - one real EN case on GPU 0, after auditing the allocated
#                 1x8 H200 node; writes the registered smoke gate marker.
#   MODE=full   - eight GPU-local workers, each running one modulo shard for
#                 EN and then ZH.  A validated smoke marker is mandatory for
#                 every live submission and again inside the full job.
#
# Default is a QZ dry-run.  This script never creates a live task unless
# DRY_RUN=0 is supplied explicitly.

set -euo pipefail

SELF_PATH=$(readlink -f "$0")
ALLOWED_PROJECT_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
ALLOWED_QZCLI="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/qzcli_with_deps.sh"
ALLOWED_QZCLI_HOME="/inspire/ssd/project/embodied-multimodality/public/xyzhang/.codex/qzcli_home"
PROJECT_ROOT="${PROJECT_ROOT:-$ALLOWED_PROJECT_ROOT}"
QZCLI="${QZCLI:-$ALLOWED_QZCLI}"
QZCLI_HOME="${QZCLI_HOME:-$ALLOWED_QZCLI_HOME}"

ALLOWED_WORKSPACE="ws-8207e9e2-e733-4eec-a475-cfa1c36480ba"
ALLOWED_PROJECT="project-c67c548f-f02c-453b-ba5b-8745db6886e7"
ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"
ALLOWED_SPEC="67b10bc6-78b0-41a3-aaf4-358eeeb99009"
ALLOWED_GPU_TYPE="NVIDIA_H200_SXM_141G"
ALLOWED_INSTANCES="1"
ALLOWED_GPUS="8"

WORKSPACE="${WORKSPACE:-$ALLOWED_WORKSPACE}"
PROJECT="${PROJECT:-$ALLOWED_PROJECT}"
COMPUTE_GROUP="${COMPUTE_GROUP:-$ALLOWED_COMPUTE_GROUP}"
SPEC="${SPEC:-$ALLOWED_SPEC}"
QZCLI_GPU_TYPE_OVERRIDE="${QZCLI_GPU_TYPE_OVERRIDE:-$ALLOWED_GPU_TYPE}"
INSTANCES="${INSTANCES:-$ALLOWED_INSTANCES}"
GPUS_PER_INSTANCE="${GPUS_PER_INSTANCE:-$ALLOWED_GPUS}"
NUM_SHARDS="${NUM_SHARDS:-$ALLOWED_GPUS}"

ALLOWED_IMAGE="docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420"
ALLOWED_IMAGE_TYPE="SOURCE_PRIVATE"
ALLOWED_FRAMEWORK="pytorch"
ALLOWED_SHM_GI="1200"
IMAGE="${IMAGE:-$ALLOWED_IMAGE}"
IMAGE_TYPE="${IMAGE_TYPE:-$ALLOWED_IMAGE_TYPE}"
FRAMEWORK="${FRAMEWORK:-$ALLOWED_FRAMEWORK}"
SHM_GI="${SHM_GI:-$ALLOWED_SHM_GI}"
PRIORITY="${PRIORITY:-10}"

MODE="${MODE:-smoke}"
RUN_TAG="${RUN_TAG:-20260711_mtts}"
SMOKE_GATE_TAG="${SMOKE_GATE_TAG:-20260711_mtts}"
DRY_RUN="${DRY_RUN:-1}"
FORCE="${FORCE:-0}"
ENTRYPOINT="${BATCH42_PATHX_STRICT_ENTRYPOINT:-0}"

SYSTEM_ID="path_x_3k"
REGISTERED_MODEL_PATH="$PROJECT_ROOT/outputs/lora_runs/ver23_content_side_3k_olddata_textrep10_ver23_content_side_text_bypass_3k_20260710/step-3000"
REGISTERED_CODE_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch3436_eval_20260711_1092820"
REGISTERED_ENGINE="$REGISTERED_CODE_ROOT/scripts/004044_run_seedtts_validation_infer_persistent.py"
REGISTERED_WAVLM="/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/huggingface/models--microsoft--wavlm-base-plus/snapshots/4c66d4806a428f2e922ccfa1a962776e232d487b"
REGISTERED_WAVLM_CACHE="/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/huggingface"
REGISTERED_BASE_MODEL="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction/MOSS-TTS"
REGISTERED_BASE_CONFIG_SHA256="214fc997d98f51ab57925a5939afc6280e76044198b664221622e70d098ed06e"
REGISTERED_ENGINE_SHA256="c9dec31f4155d39cdbd02069dd8b91677ff5dee03e98d441377d949135a8e709"

AUDIT_ROOT="$PROJECT_ROOT/testset/outputs/batch42_seedtts_eval_audit_20260711"
EN_MANIFEST="$AUDIT_ROOT/official_en_vc_minus_internal320_strict_case.lst"
ZH_MANIFEST="$AUDIT_ROOT/official_zh_vc_minus_internal320_strict_case.lst"
EN_INPUT_ROOT="/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/batch42/datasets/seed-tts-eval/seedtts_testset/en"
ZH_INPUT_ROOT="/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/batch42/datasets/seed-tts-eval/seedtts_testset/zh"
EN_MANIFEST_SHA256="48549d8029e680d74656660191c4641ca5a8040ccbe3252ce89bfc3b0c9c75ae"
ZH_MANIFEST_SHA256="4b637cc1cff33dc369954755538d12396fc92d439a52742103a29b7c563cf6df"
EN_EXPECTED="567"
ZH_EXPECTED="1194"
EN_TEST_SET_ID="seedtts-vc-en-internal320-disjoint"
ZH_TEST_SET_ID="seedtts-vc-zh-internal320-disjoint"

ALLOWED_BASE_PY="/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python"
ALLOWED_MOSS_PY="/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python"
ALLOWED_BATCH42_DEPS="/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/python_deps/batch42_eval"
BASE_PY="${BASE_PY:-$ALLOWED_BASE_PY}"
MOSS_PY="${MOSS_PY:-$ALLOWED_MOSS_PY}"
BATCH42_DEPS="${BATCH42_DEPS:-$ALLOWED_BATCH42_DEPS}"

SMOKE_OUTPUT_ROOT="$PROJECT_ROOT/testset/outputs/batch42_pathx_strict_smoke_gate_${SMOKE_GATE_TAG}"
SMOKE_MARKER="$SMOKE_OUTPUT_ROOT/SMOKE_COMPLETED.json"
if [ "$MODE" = "smoke" ]; then
  RECORD_ROOT="${RECORD_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/batch42_pathx_strict_smoke_${SMOKE_GATE_TAG}}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-$SMOKE_OUTPUT_ROOT}"
  JOB_NAME="${JOB_NAME:-batch42_pathx_strict_smoke_${SMOKE_GATE_TAG}}"
  FINAL_MARKER="$SMOKE_MARKER"
else
  RECORD_ROOT="${RECORD_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/batch42_pathx_strict_full_${RUN_TAG}}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/testset/outputs/batch42_pathx_strict_${SYSTEM_ID}_${RUN_TAG}}"
  JOB_NAME="${JOB_NAME:-batch42_pathx_strict_${SYSTEM_ID}_${RUN_TAG}}"
  FINAL_MARKER="$OUTPUT_ROOT/COMPLETED.json"
fi

SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-$RECORD_ROOT/record_snapshot}"
FROZEN_DRIVER="$SNAPSHOT_ROOT/scripts/004094_submit_batch42_pathx_strict_qz.sh"
PATHX_SCRIPT="$SNAPSHOT_ROOT/scripts/004093_run_batch42_pathx_strict.py"
MERGE_SCRIPT="$SNAPSHOT_ROOT/scripts/004089_merge_batch42_baseline_shards.py"
SCHEMA_SCRIPT="$SNAPSHOT_ROOT/scripts/004082_run_unified_vc_eval.py"
RUNNER="${RUNNER:-$RECORD_ROOT/run_batch42_pathx_strict_entrypoint.sh}"
SOURCE_PATHX_SCRIPT="$PROJECT_ROOT/scripts/004093_run_batch42_pathx_strict.py"
SOURCE_MERGE_SCRIPT="$PROJECT_ROOT/scripts/004089_merge_batch42_baseline_shards.py"
SOURCE_SCHEMA_SCRIPT="$PROJECT_ROOT/scripts/004082_run_unified_vc_eval.py"

INPUT_DIR="$RECORD_ROOT/inputs"
EN_CANONICAL="$INPUT_DIR/en.canonical.jsonl"
ZH_CANONICAL="$INPUT_DIR/zh.canonical.jsonl"
EN_PREPARE_SUMMARY="$INPUT_DIR/en.prepare_summary.json"
ZH_PREPARE_SUMMARY="$INPUT_DIR/zh.prepare_summary.json"
IDENTITY_AUDIT_SUBMIT="$RECORD_ROOT/registered_identity.submit.json"
IDENTITY_AUDIT_JOB="$RECORD_ROOT/registered_identity.job.json"
SUBMISSION_LOCK="$RECORD_ROOT/.live_submission_lock"
SUBMISSION_LOCK_HELD=0

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_file() {
  [ -s "$1" ] || die "missing or empty file: $1"
}

require_dir() {
  [ -d "$1" ] || die "missing directory: $1"
}

require_executable() {
  [ -x "$1" ] || die "missing executable: $1"
}

release_submission_lock() {
  if [ "$SUBMISSION_LOCK_HELD" = "1" ]; then
    rmdir "$SUBMISSION_LOCK" 2>/dev/null || true
    SUBMISSION_LOCK_HELD=0
  fi
}

validate_fixed_configuration() {
  case "$MODE" in
    smoke|full) ;;
    *) die "MODE must be smoke or full; got $MODE" ;;
  esac
  case "$DRY_RUN:$FORCE:$ENTRYPOINT" in
    [01]:[01]:[01]) ;;
    *) die "DRY_RUN, FORCE, and BATCH42_PATHX_STRICT_ENTRYPOINT must be 0 or 1" ;;
  esac
  [ "$(readlink -m "$PROJECT_ROOT")" = "$ALLOWED_PROJECT_ROOT" ] || \
    die "PROJECT_ROOT is hard-locked to $ALLOWED_PROJECT_ROOT"
  [ "$(readlink -f "$QZCLI")" = "$(readlink -f "$ALLOWED_QZCLI")" ] || \
    die "QZCLI must use the registered qzcli_with_deps.sh wrapper"
  [ "$(readlink -m "$QZCLI_HOME")" = "$ALLOWED_QZCLI_HOME" ] || \
    die "QZCLI_HOME must use the registered Codex credential home"
  [ "$WORKSPACE" = "$ALLOWED_WORKSPACE" ] || die "workspace must be CI-情境智能"
  [ "$PROJECT" = "$ALLOWED_PROJECT" ] || die "project must be CI-情境智能"
  [ "$COMPUTE_GROUP" = "$ALLOWED_COMPUTE_GROUP" ] || \
    die "only MTTS-3-2-0715 is allowed; got $COMPUTE_GROUP"
  [ "$SPEC" = "$ALLOWED_SPEC" ] || die "only registered MTTS spec is allowed"
  [ "$QZCLI_GPU_TYPE_OVERRIDE" = "$ALLOWED_GPU_TYPE" ] || die "only H200 is allowed"
  [ "$INSTANCES" = "$ALLOWED_INSTANCES" ] || die "exactly one instance is required"
  [ "$GPUS_PER_INSTANCE" = "$ALLOWED_GPUS" ] || die "exactly 8 GPUs are required"
  [ "$NUM_SHARDS" = "$ALLOWED_GPUS" ] || die "full protocol requires 8 shards"
  [ "$IMAGE" = "$ALLOWED_IMAGE" ] || die "QZ image is hard-locked for Path X strict evaluation"
  [ "$IMAGE_TYPE" = "$ALLOWED_IMAGE_TYPE" ] || die "QZ image type is hard-locked"
  [ "$FRAMEWORK" = "$ALLOWED_FRAMEWORK" ] || die "QZ framework is hard-locked"
  [ "$SHM_GI" = "$ALLOWED_SHM_GI" ] || die "QZ shared-memory request is hard-locked"
  [ "$BASE_PY" = "$ALLOWED_BASE_PY" ] || die "BASE_PY is hard-locked"
  [ "$MOSS_PY" = "$ALLOWED_MOSS_PY" ] || die "MOSS_PY is hard-locked"
  [ "$BATCH42_DEPS" = "$ALLOWED_BATCH42_DEPS" ] || die "BATCH42_DEPS is hard-locked"
  case "$RUN_TAG:$SMOKE_GATE_TAG" in
    *[!A-Za-z0-9_.:-]*|:*|*:) die "RUN_TAG and SMOKE_GATE_TAG must be non-empty safe path components" ;;
  esac
  record_base=$(readlink -m "$PROJECT_ROOT/trainset/qz_jobs")
  output_base=$(readlink -m "$PROJECT_ROOT/testset/outputs")
  record_resolved=$(readlink -m "$RECORD_ROOT")
  output_resolved=$(readlink -m "$OUTPUT_ROOT")
  snapshot_resolved=$(readlink -m "$SNAPSHOT_ROOT")
  runner_resolved=$(readlink -m "$RUNNER")
  case "$record_resolved" in
    "$record_base"/*) ;;
    *) die "RECORD_ROOT must stay under $record_base" ;;
  esac
  case "$output_resolved" in
    "$output_base"/*) ;;
    *) die "OUTPUT_ROOT must stay under $output_base" ;;
  esac
  case "$snapshot_resolved" in
    "$record_resolved"/*) ;;
    *) die "SNAPSHOT_ROOT must stay under RECORD_ROOT" ;;
  esac
  case "$runner_resolved" in
    "$record_resolved"/*) ;;
    *) die "RUNNER must stay under RECORD_ROOT" ;;
  esac
  [ "$(readlink -f "$REGISTERED_MODEL_PATH")" = "$(readlink -f "$PROJECT_ROOT/outputs/lora_runs/ver23_content_side_3k_olddata_textrep10_ver23_content_side_text_bypass_3k_20260710/step-3000")" ] || \
    die "Batch-33 model path identity changed"
  [ "$(readlink -f "$REGISTERED_CODE_ROOT")" = "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch3436_eval_20260711_1092820" ] || \
    die "frozen eval code root identity changed"
  [ "$(readlink -f "$REGISTERED_ENGINE")" = "$(readlink -f "$REGISTERED_CODE_ROOT/scripts/004044_run_seedtts_validation_infer_persistent.py")" ] || \
    die "registered 004044 engine mismatch"
  if [ "$MODE" = "smoke" ] && [ "$(readlink -m "$OUTPUT_ROOT")" != "$(readlink -m "$SMOKE_OUTPUT_ROOT")" ]; then
    die "smoke output must be the registered shared smoke-gate root"
  fi
  if [ "$MODE" = "full" ] && [ "$(readlink -m "$OUTPUT_ROOT")" = "$(readlink -m "$SMOKE_OUTPUT_ROOT")" ]; then
    die "full output must be separate from the registered smoke-gate root"
  fi
}

audit_strict_inputs() {
  "$BASE_PY" - \
    "$EN_MANIFEST" "$EN_INPUT_ROOT" "$EN_EXPECTED" "$EN_MANIFEST_SHA256" \
    "$ZH_MANIFEST" "$ZH_INPUT_ROOT" "$ZH_EXPECTED" "$ZH_MANIFEST_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def audit(manifest_arg, root_arg, expected_arg, expected_sha):
    manifest = Path(manifest_arg).resolve()
    root = Path(root_arg).resolve()
    expected = int(expected_arg)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if digest != expected_sha:
        raise SystemExit(f"{manifest}: SHA256 mismatch {digest}")
    rows = []
    for line_number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split("|")
        if len(fields) != 5 or any(not item.strip() for item in fields):
            raise SystemExit(f"{manifest}:{line_number}: malformed/empty five-column row")
        case_id, _, ref_value, _, src_value = (item.strip() for item in fields)
        ref = (root / ref_value).resolve()
        src = (root / src_value).resolve()
        for role, path in (("reference", ref), ("source", src)):
            try:
                path.relative_to(root)
            except ValueError:
                raise SystemExit(f"{manifest}:{line_number}: {role} escapes root: {path}")
            if not path.is_file() or path.stat().st_size < 44:
                raise SystemExit(f"{manifest}:{line_number}: missing/empty {role}: {path}")
        if ref == src:
            raise SystemExit(f"{manifest}:{line_number}: source equals reference")
        rows.append(case_id)
    if len(rows) != expected or len(set(rows)) != expected:
        raise SystemExit(f"{manifest}: expected {expected} unique rows, got {len(rows)}")
    return {"manifest": str(manifest), "input_root": str(root), "rows": len(rows), "sha256": digest}

print(json.dumps({
    "schema_version": "moss_codecvc.batch42_pathx_strict_input_audit.v1",
    "en": audit(*sys.argv[1:5]),
    "zh": audit(*sys.argv[5:9]),
}, ensure_ascii=False, indent=2))
PY
}

audit_registered_identity() {
  output_json=$1
  "$BASE_PY" - \
    "$REGISTERED_MODEL_PATH" "$REGISTERED_CODE_ROOT" "$REGISTERED_ENGINE" \
    "$REGISTERED_WAVLM" "$REGISTERED_BASE_MODEL" \
    "$REGISTERED_BASE_CONFIG_SHA256" "$REGISTERED_ENGINE_SHA256" "$output_json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

model, code, engine, wavlm, base_model = map(Path, sys.argv[1:6])
expected_base_config_sha, expected_engine_sha, output_arg = sys.argv[6:9]
output = Path(output_arg)
expected_model = {
    "README.md": (179, "4d45f7d68a88a39671cc0cbc86f1acdfbee5351401eee2a97df253f0d077717f"),
    "adapter_config.json": (1179, "06530eac22376a6befd9e81c95c333e4bb1c889de96e9059c2d5498cd90a7aee"),
    "adapter_model.safetensors": (87366096, "3a51162fc7ccf1b9e1aa477ad7c44fa64390d109b8b63765a9cd636f090f4b25"),
    "timbre_memory_config.json": (5026, "5c8842d87327c2cf1af2697725a19bf2b53ba654fa0a6b3f68b6a42fd50e9970"),
    "timbre_memory_adapter.pt": (1697093491, "020a16ad4bba5a812b2f62e29cb68dcec9d4055344e02de01555be8afd9d6895"),
}
expected_code = {
    "scripts/004044_run_seedtts_validation_infer_persistent.py": "c9dec31f4155d39cdbd02069dd8b91677ff5dee03e98d441377d949135a8e709",
    "scripts/003001_infer_moss_codecvc.py": "d9a3426a3668a4bdd95a81fdf86b02e32d774b4893ba0428e5b1c6fba4f5ce73",
    "moss_codecvc/models/moss_codecvc_wrapper.py": "5815c8ab5e0aab69d19328fd01782620064327eaf5f39cc4923df8ce3ae9ca42",
    "moss_codecvc/models/content_cross_attn.py": "a8e4cd12d279cfff7c38e3e2d8b21b55d70c403cec654edf7ef77de58acba66a",
}
expected_wavlm = {
    "config.json": (2232, "fea6df1c2700a3954fc07e70588aecc9055eeb28db2ff57151a2db0d19180ed4"),
    "pytorch_model.bin": (377617425, "3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51"),
}
if engine.resolve() != (code / "scripts/004044_run_seedtts_validation_infer_persistent.py").resolve():
    raise SystemExit("engine is outside registered frozen code root")
if not base_model.is_dir():
    raise SystemExit(f"registered base model is missing: {base_model}")

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

model_files = {}
for name, (expected_size, expected_sha) in expected_model.items():
    path = model / name
    if not path.is_file() or path.stat().st_size != expected_size:
        raise SystemExit(f"checkpoint file mismatch: {path}")
    actual_sha = sha(path)
    if actual_sha != expected_sha:
        raise SystemExit(f"checkpoint hash mismatch: {path}: {actual_sha}")
    model_files[name] = {"path": str(path.resolve()), "size": expected_size, "sha256": actual_sha}

code_files = {}
for relative, expected_sha in expected_code.items():
    path = code / relative
    if not path.is_file():
        raise SystemExit(f"frozen code file missing: {path}")
    actual_sha = sha(path)
    if actual_sha != expected_sha:
        raise SystemExit(f"frozen code hash mismatch: {path}: {actual_sha}")
    code_files[relative] = {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": actual_sha}

if code_files["scripts/004044_run_seedtts_validation_infer_persistent.py"]["sha256"] != expected_engine_sha:
    raise SystemExit("registered engine SHA256 mismatch")

wavlm_files = {}
for name, (expected_size, expected_sha) in expected_wavlm.items():
    path = wavlm / name
    if not path.is_file() or path.stat().st_size != expected_size:
        raise SystemExit(f"WavLM snapshot file mismatch: {path}")
    actual_sha = sha(path)
    if actual_sha != expected_sha:
        raise SystemExit(f"WavLM snapshot hash mismatch: {path}: {actual_sha}")
    wavlm_files[name] = {"path": str(path.resolve()), "size": expected_size, "sha256": actual_sha}

adapter_config = json.loads((model / "adapter_config.json").read_text(encoding="utf-8"))
adapter_base_model = Path(str(adapter_config.get("base_model_name_or_path") or "")).resolve()
if adapter_base_model != base_model.resolve():
    raise SystemExit(
        f"adapter base model mismatch: expected {base_model.resolve()}, got {adapter_base_model}"
    )
base_config = base_model / "config.json"
if not base_config.is_file():
    raise SystemExit(f"registered base config is missing: {base_config}")
base_config_sha = sha(base_config)
if base_config_sha != expected_base_config_sha:
    raise SystemExit(f"registered base config hash mismatch: {base_config_sha}")

config = json.loads((model / "timbre_memory_config.json").read_text(encoding="utf-8"))
registered = {
    "content_cross_attn_enabled": True,
    "content_cross_attn_layers": "all",
    "content_cross_attn_feature_dim": 768,
    "content_cross_attn_gate_init": -0.5,
    "content_cross_attn_output_scale": 0.3,
    "content_encoder_layers": 2,
    "guided_attn_loss_weight": 0.05,
    "phoneme_classifier_loss_weight": 0.02,
    "progress_loss_weight": 0.1,
    "stop_loss_weight": 0.2,
    "source_semantic_memory_enabled": False,
    "speaker_side_pathway_enabled": False,
    "speaker_cross_attn_enabled": False,
    "source_content_memory_type": "wavlm_bnf_continuous",
}
mismatches = {key: [value, config.get(key)] for key, value in registered.items() if config.get(key) != value}
if mismatches:
    raise SystemExit(f"Batch-33 config mismatch: {mismatches}")
payload = {
    "schema_version": "moss_codecvc.batch42_pathx_registered_identity.v1",
    "status": "verified",
    "model_path": str(model.resolve()),
    "code_root": str(code.resolve()),
    "engine": str(engine.resolve()),
    "wavlm": str(wavlm.resolve()),
    "base_model": str(base_model.resolve()),
    "base_model_config": {
        "path": str(base_config.resolve()),
        "size": base_config.stat().st_size,
        "sha256": base_config_sha,
    },
    "model_files": model_files,
    "code_files": code_files,
    "wavlm_files": wavlm_files,
    "registered_config": registered,
    "actual_base_language_layers": 36,
}
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
}

snapshot_scripts() {
  mkdir -p "$SNAPSHOT_ROOT/scripts" "$RECORD_ROOT" "$OUTPUT_ROOT"
  require_file "$SOURCE_PATHX_SCRIPT"
  require_file "$SOURCE_MERGE_SCRIPT"
  require_file "$SOURCE_SCHEMA_SCRIPT"
  cp "$SOURCE_PATHX_SCRIPT" "$PATHX_SCRIPT"
  cp "$SOURCE_MERGE_SCRIPT" "$MERGE_SCRIPT"
  cp "$SOURCE_SCHEMA_SCRIPT" "$SCHEMA_SCRIPT"
  cp "$SELF_PATH" "$FROZEN_DRIVER"
  chmod 0555 "$PATHX_SCRIPT" "$MERGE_SCRIPT" "$SCHEMA_SCRIPT" "$FROZEN_DRIVER"
  {
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    printf 'exec env BATCH42_PATHX_STRICT_ENTRYPOINT=1 MODE=%q RUN_TAG=%q SMOKE_GATE_TAG=%q PROJECT_ROOT=%q RECORD_ROOT=%q OUTPUT_ROOT=%q SNAPSHOT_ROOT=%q RUNNER=%q WORKSPACE=%q PROJECT=%q COMPUTE_GROUP=%q SPEC=%q INSTANCES=%q GPUS_PER_INSTANCE=%q NUM_SHARDS=%q QZCLI_GPU_TYPE_OVERRIDE=%q bash %q\n' \
      "$MODE" "$RUN_TAG" "$SMOKE_GATE_TAG" "$PROJECT_ROOT" "$RECORD_ROOT" "$OUTPUT_ROOT" "$SNAPSHOT_ROOT" "$RUNNER" \
      "$WORKSPACE" "$PROJECT" "$COMPUTE_GROUP" "$SPEC" "$INSTANCES" "$GPUS_PER_INSTANCE" "$NUM_SHARDS" "$QZCLI_GPU_TYPE_OVERRIDE" "$FROZEN_DRIVER"
  } > "$RUNNER"
  chmod 0555 "$RUNNER"
  bash -n "$FROZEN_DRIVER"
  bash -n "$RUNNER"
  sha256sum "$PATHX_SCRIPT" "$MERGE_SCRIPT" "$SCHEMA_SCRIPT" "$FROZEN_DRIVER" "$RUNNER" > "$RECORD_ROOT/sha256sums.txt"
}

prepare_inputs() {
  mkdir -p "$INPUT_DIR"
  "$BASE_PY" "$PATHX_SCRIPT" prepare \
    --input "$EN_MANIFEST" --input-root "$EN_INPUT_ROOT" --language en \
    --expected-cases "$EN_EXPECTED" --expected-sha256 "$EN_MANIFEST_SHA256" \
    --test-set-id "$EN_TEST_SET_ID" --output-jsonl "$EN_CANONICAL" \
    --summary-json "$EN_PREPARE_SUMMARY" > "$INPUT_DIR/en.prepare.log"
  "$BASE_PY" "$PATHX_SCRIPT" prepare \
    --input "$ZH_MANIFEST" --input-root "$ZH_INPUT_ROOT" --language zh \
    --expected-cases "$ZH_EXPECTED" --expected-sha256 "$ZH_MANIFEST_SHA256" \
    --test-set-id "$ZH_TEST_SET_ID" --output-jsonl "$ZH_CANONICAL" \
    --summary-json "$ZH_PREPARE_SUMMARY" > "$INPUT_DIR/zh.prepare.log"
}

canonical_sha() {
  "$BASE_PY" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["output_sha256"])' "$1"
}

validate_audio_file() {
  "$BASE_PY" - "$1" <<'PY'
import sys
import wave
from pathlib import Path

path = Path(sys.argv[1]).resolve()
if not path.is_file() or path.stat().st_size < 1024:
    raise SystemExit(f"generated audio is missing/too small: {path}")
errors = []
try:
    import soundfile as sf

    info = sf.info(str(path))
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise RuntimeError(f"invalid soundfile metadata: {info!r}")
except Exception as exc:
    errors.append(f"soundfile={exc}")
    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getnframes() <= 0 or handle.getframerate() <= 0 or handle.getnchannels() <= 0:
                raise RuntimeError("invalid wave metadata")
    except Exception as wave_exc:
        errors.append(f"wave={wave_exc}")
        raise SystemExit(f"generated audio cannot be decoded: {path}: {'; '.join(errors)}")
print(path)
PY
}

engine_schema_preflight() {
  preflight_root="$RECORD_ROOT/preflight_submit"
  mkdir -p "$preflight_root"
  en_sha=$(canonical_sha "$EN_PREPARE_SUMMARY")
  "$MOSS_PY" "$PATHX_SCRIPT" run \
    --python "$MOSS_PY" --engine-script "$REGISTERED_ENGINE" \
    --code-root "$REGISTERED_CODE_ROOT" --model-path "$REGISTERED_MODEL_PATH" \
    --canonical-jsonl "$EN_CANONICAL" --expected-canonical-sha256 "$en_sha" \
    --expected-cases "$EN_EXPECTED" --input-root "$EN_INPUT_ROOT" \
    --output-dir "$preflight_root/audio" --raw-manifest "$preflight_root/raw.jsonl" \
    --manifest-jsonl "$preflight_root/manifest.jsonl" --summary-json "$preflight_root/summary.json" \
    --system-id "$SYSTEM_ID" --test-set-id "$EN_TEST_SET_ID" --language en \
    --num-shards 1 --shard-index 0 --max-cases 1 --device cuda:0 \
    --seed 1234 --temperature 0.7 --audio-temperature 1.1 --audio-top-p 0.7 \
    --audio-top-k 20 --audio-repetition-penalty 1.0 \
    --no-text-duration-budget-ratio 1.0 --no-text-max-token-margin 0 \
    --timbre-cfg-scale 1.0 --source-semantic-model "$REGISTERED_WAVLM" \
    --source-semantic-cache "$REGISTERED_WAVLM_CACHE" --source-semantic-layer 9 \
    --source-semantic-downsample-stride 1 --engine-dry-run
}

run_actual_node_preflight() {
  preflight_root="$RECORD_ROOT/preflight_job_actual"
  mkdir -p "$preflight_root/audio" "$preflight_root/hf_modules"
  rm -f "$preflight_root/raw.jsonl" "$preflight_root/manifest.jsonl" "$preflight_root/summary.json"
  en_sha=$(canonical_sha "$EN_PREPARE_SUMMARY")
  CUDA_VISIBLE_DEVICES=0 \
  HF_MODULES_CACHE="$preflight_root/hf_modules" \
  HF_MODULES_CACHE_ROOT="$preflight_root/hf_modules" \
    "$MOSS_PY" "$PATHX_SCRIPT" run \
      --python "$MOSS_PY" --engine-script "$REGISTERED_ENGINE" \
      --code-root "$REGISTERED_CODE_ROOT" --model-path "$REGISTERED_MODEL_PATH" \
      --canonical-jsonl "$EN_CANONICAL" --expected-canonical-sha256 "$en_sha" \
      --expected-cases "$EN_EXPECTED" --input-root "$EN_INPUT_ROOT" \
      --output-dir "$preflight_root/audio" --raw-manifest "$preflight_root/raw.jsonl" \
      --manifest-jsonl "$preflight_root/manifest.jsonl" --summary-json "$preflight_root/summary.json" \
      --system-id "$SYSTEM_ID" --test-set-id "$EN_TEST_SET_ID" --language en \
      --num-shards 1 --shard-index 0 --max-cases 1 --device cuda:0 \
      --seed 1234 --temperature 0.7 --audio-temperature 1.1 --audio-top-p 0.7 \
      --audio-top-k 20 --audio-repetition-penalty 1.0 \
      --no-text-duration-budget-ratio 1.0 --no-text-max-token-margin 0 \
      --timbre-cfg-scale 1.0 --source-semantic-model "$REGISTERED_WAVLM" \
      --source-semantic-cache "$REGISTERED_WAVLM_CACHE" --source-semantic-layer 9 \
      --source-semantic-downsample-stride 1
  generated=$(
    "$BASE_PY" - "$preflight_root/manifest.jsonl" "$preflight_root/summary.json" <<'PY'
import json
import sys
from pathlib import Path

manifest_path, summary_path = map(Path, sys.argv[1:])
rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if len(rows) != 1 or rows[0].get("status") != "ok":
    raise SystemExit(f"actual node preflight requires one newly generated ok row: {rows!r}")
if summary.get("status") != "complete" or summary.get("status_counts") != {"ok": 1}:
    raise SystemExit(f"actual node preflight summary failed: {summary!r}")
print(rows[0].get("generated_audio") or "")
PY
  )
  validate_audio_file "$generated" > "$preflight_root/audio_decode.txt"
  echo "[pathx-preflight] actual one-case forward passed: $generated"
}

audit_allocated_gpus() {
  gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l | tr -d ' ')
  [ "$gpu_count" = "$ALLOWED_GPUS" ] || die "expected exactly 8 allocated GPUs, got $gpu_count"
  if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -Evq 'H200'; then
    die "all allocated GPUs must be H200"
  fi
  nvidia-smi
}

validate_smoke_marker() {
  require_file "$SMOKE_MARKER"
  "$BASE_PY" - \
    "$SMOKE_MARKER" "$REGISTERED_MODEL_PATH" "$REGISTERED_CODE_ROOT" \
    "$REGISTERED_WAVLM" "$REGISTERED_ENGINE_SHA256" \
    "$REGISTERED_BASE_CONFIG_SHA256" "$EN_MANIFEST_SHA256" "$ZH_MANIFEST_SHA256" <<'PY'
import hashlib
import json
import sys
import wave
from pathlib import Path

marker_path = Path(sys.argv[1])
marker = json.loads(marker_path.read_text(encoding="utf-8"))
errors = []
if marker.get("schema_version") != "moss_codecvc.batch42_pathx_strict_smoke_completion.v1":
    errors.append("wrong schema_version")
if marker.get("status") != "smoke_complete":
    errors.append(f"status={marker.get('status')!r}")
identity = marker.get("registered_identity") or {}
if str(Path(identity.get("model_path", "")).resolve()) != str(Path(sys.argv[2]).resolve()):
    errors.append("model identity mismatch")
if str(Path(identity.get("code_root", "")).resolve()) != str(Path(sys.argv[3]).resolve()):
    errors.append("code identity mismatch")
if str(Path(identity.get("wavlm", "")).resolve()) != str(Path(sys.argv[4]).resolve()):
    errors.append("WavLM identity mismatch")
strict = marker.get("strict_inputs") or {}
if (strict.get("en") or {}).get("sha256") != sys.argv[7]:
    errors.append("EN strict manifest hash mismatch")
if (strict.get("zh") or {}).get("sha256") != sys.argv[8]:
    errors.append("ZH strict manifest hash mismatch")
resource = marker.get("resource_contract") or {}
if resource != {"compute_group": "MTTS-3-2-0715", "gpu_type": "NVIDIA_H200_SXM_141G", "gpus": 8, "instances": 1}:
    errors.append(f"resource contract mismatch: {resource!r}")
actual = marker.get("actual_one_case") or {}
generated = Path(str(actual.get("generated_audio") or ""))
if not generated.is_file() or generated.stat().st_size < 1024:
    errors.append(f"smoke generated audio missing/empty: {generated}")
else:
    decoded = False
    decode_errors = []
    try:
        import soundfile as sf

        info = sf.info(str(generated))
        decoded = info.frames > 0 and info.samplerate > 0 and info.channels > 0
        if not decoded:
            decode_errors.append(f"invalid soundfile metadata: {info!r}")
    except Exception as exc:
        decode_errors.append(f"soundfile={exc}")
    if not decoded:
        try:
            with wave.open(str(generated), "rb") as handle:
                decoded = (
                    handle.getnframes() > 0
                    and handle.getframerate() > 0
                    and handle.getnchannels() > 0
                )
        except Exception as exc:
            decode_errors.append(f"wave={exc}")
    if not decoded:
        errors.append(f"smoke generated audio is not decodable: {decode_errors}")

inference = actual.get("inference_config") or {}
expected_inference = {
    "mode": "no_text",
    "engine_sha256": sys.argv[5],
    "source_semantic_model": str(Path(sys.argv[4]).resolve()),
    "source_semantic_layer": 9,
    "source_semantic_downsample_stride": 1,
    "source_semantic_local_files_only": True,
    "temperature": 0.7,
    "audio_temperature": 1.1,
    "audio_top_p": 0.7,
    "audio_top_k": 20,
    "audio_repetition_penalty": 1.0,
    "no_text_duration_budget_ratio": 1.0,
    "no_text_max_token_margin": 0,
    "no_text_soft_duration_budget": False,
    "filter_ref_content_leak": False,
    "timbre_cfg_scale": 1.0,
    "ref_audio_cfg_scale": 1.0,
    "ref_prompt_codec_permutation": False,
    "ref_speaker_prompt_slot": False,
    "timbre_side_only": False,
    "audio_segment_policy": "all",
    "source_semantic_monotonic_bias_strength": 0.0,
    "source_semantic_progress_clock": "decode_step",
    "source_semantic_release_after_progress": False,
    "source_semantic_release_start": 1.0,
    "seed": 1234,
    "model_path": str(Path(sys.argv[2]).resolve()),
    "code_root": str(Path(sys.argv[3]).resolve()),
    "base_model_path": "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction/MOSS-TTS",
}
for key, expected in expected_inference.items():
    if inference.get(key) != expected:
        errors.append(f"inference contract mismatch for {key}: {inference.get(key)!r} != {expected!r}")

model_files = identity.get("model_files") or {}
code_files = identity.get("code_files") or {}
wavlm_files = identity.get("wavlm_files") or {}
asset_hashes = {
    "adapter_model": (model_files.get("adapter_model.safetensors") or {}).get("sha256"),
    "timbre_adapter": (model_files.get("timbre_memory_adapter.pt") or {}).get("sha256"),
    "timbre_config": (model_files.get("timbre_memory_config.json") or {}).get("sha256"),
    "engine": (code_files.get("scripts/004044_run_seedtts_validation_infer_persistent.py") or {}).get("sha256"),
    "wavlm_model": (wavlm_files.get("pytorch_model.bin") or {}).get("sha256"),
    "base_config": (identity.get("base_model_config") or {}).get("sha256"),
}
expected_asset_hashes = {
    "adapter_model": "3a51162fc7ccf1b9e1aa477ad7c44fa64390d109b8b63765a9cd636f090f4b25",
    "timbre_adapter": "020a16ad4bba5a812b2f62e29cb68dcec9d4055344e02de01555be8afd9d6895",
    "timbre_config": "5c8842d87327c2cf1af2697725a19bf2b53ba654fa0a6b3f68b6a42fd50e9970",
    "engine": sys.argv[5],
    "wavlm_model": "3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51",
    "base_config": sys.argv[6],
}
if asset_hashes != expected_asset_hashes:
    errors.append(f"registered asset hash mismatch: {asset_hashes!r}")

protocol_contract = {
    "system_id": "path_x_3k",
    "inference": {key: inference.get(key) for key in sorted(expected_inference)},
    "asset_hashes": asset_hashes,
    "strict_manifest_sha256": {
        "en": (strict.get("en") or {}).get("sha256"),
        "zh": (strict.get("zh") or {}).get("sha256"),
    },
}
stored_contract = marker.get("protocol_contract")
if stored_contract != protocol_contract:
    errors.append("stored smoke protocol contract does not match marker payload")
fingerprint = hashlib.sha256(
    json.dumps(protocol_contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
if marker.get("protocol_fingerprint_sha256") != fingerprint:
    errors.append("smoke protocol fingerprint mismatch")
if errors:
    raise SystemExit("; ".join(errors))
print(
    f"[pathx-smoke-gate] PASS marker={marker_path.resolve()} "
    f"case={actual.get('case_id')} fingerprint={fingerprint}"
)
PY
}

run_pathx_shard() {
  language_id=$1
  shard_index=$2
  run_mode=$3
  if [ "$language_id" = "en" ]; then
    canonical="$EN_CANONICAL"
    canonical_summary="$EN_PREPARE_SUMMARY"
    expected="$EN_EXPECTED"
    input_root="$EN_INPUT_ROOT"
    test_set="$EN_TEST_SET_ID"
  else
    canonical="$ZH_CANONICAL"
    canonical_summary="$ZH_PREPARE_SUMMARY"
    expected="$ZH_EXPECTED"
    input_root="$ZH_INPUT_ROOT"
    test_set="$ZH_TEST_SET_ID"
  fi
  canonical_digest=$(canonical_sha "$canonical_summary")
  shard_tag=$(printf '%05d' "$shard_index")
  audio_dir="$OUTPUT_ROOT/$language_id/audio"
  raw_manifest="$RECORD_ROOT/raw/$language_id/raw.shard-${shard_tag}.jsonl"
  manifest="$OUTPUT_ROOT/$language_id/manifests/manifest.shard-${shard_tag}-of-00008.jsonl"
  summary="$RECORD_ROOT/summaries/$language_id/summary.shard-${shard_tag}.json"
  mkdir -p "$audio_dir" "$(dirname "$raw_manifest")" "$(dirname "$manifest")" "$(dirname "$summary")"
  if [ "$run_mode" = "smoke" ]; then
    shard_count=1
    max_args=(--max-cases 1)
    resume_args=()
    manifest="$OUTPUT_ROOT/$language_id/manifests/manifest.smoke.jsonl"
    raw_manifest="$RECORD_ROOT/raw/$language_id/raw.smoke.jsonl"
    summary="$RECORD_ROOT/summaries/$language_id/summary.smoke.json"
  else
    shard_count="$NUM_SHARDS"
    max_args=()
    resume_args=(--resume)
  fi
  "$MOSS_PY" "$PATHX_SCRIPT" run \
    --python "$MOSS_PY" --engine-script "$REGISTERED_ENGINE" \
    --code-root "$REGISTERED_CODE_ROOT" --model-path "$REGISTERED_MODEL_PATH" \
    --canonical-jsonl "$canonical" --expected-canonical-sha256 "$canonical_digest" \
    --expected-cases "$expected" --input-root "$input_root" \
    --output-dir "$audio_dir" --raw-manifest "$raw_manifest" \
    --manifest-jsonl "$manifest" --summary-json "$summary" \
    --system-id "$SYSTEM_ID" --test-set-id "$test_set" --language "$language_id" \
    --num-shards "$shard_count" --shard-index "$shard_index" --device cuda:0 \
    --seed 1234 --temperature 0.7 --audio-temperature 1.1 --audio-top-p 0.7 \
    --audio-top-k 20 --audio-repetition-penalty 1.0 \
    --no-text-duration-budget-ratio 1.0 --no-text-max-token-margin 0 \
    --timbre-cfg-scale 1.0 --source-semantic-model "$REGISTERED_WAVLM" \
    --source-semantic-cache "$REGISTERED_WAVLM_CACHE" --source-semantic-layer 9 \
    --source-semantic-downsample-stride 1 \
    "${max_args[@]}" "${resume_args[@]}"
}

run_smoke() {
  mkdir -p "$RECORD_ROOT/worker_logs"
  CUDA_VISIBLE_DEVICES=0 \
  HF_MODULES_CACHE="$RECORD_ROOT/hf_modules/smoke" \
  HF_MODULES_CACHE_ROOT="$RECORD_ROOT/hf_modules/smoke" \
    run_pathx_shard en 0 smoke > "$RECORD_ROOT/worker_logs/smoke.log" 2>&1
}

run_all_workers() {
  mkdir -p "$RECORD_ROOT/worker_logs" "$RECORD_ROOT/hf_modules"
  pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    (
      export CUDA_VISIBLE_DEVICES="$shard"
      export HF_MODULES_CACHE="$RECORD_ROOT/hf_modules/worker-$shard"
      export HF_MODULES_CACHE_ROOT="$RECORD_ROOT/hf_modules/worker-$shard"
      echo "[pathx-worker-$shard] start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      run_pathx_shard en "$shard" full
      run_pathx_shard zh "$shard" full
      echo "[pathx-worker-$shard] complete=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    ) > "$RECORD_ROOT/worker_logs/shard-$(printf '%05d' "$shard").log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "ERROR: Path X worker pid $pid failed; shard artifacts are preserved" >&2
      failed=1
    fi
  done
  [ "$failed" = "0" ] || return 1
}

merge_language() {
  language_id=$1
  if [ "$language_id" = "en" ]; then
    expected="$EN_EXPECTED"
    test_set="$EN_TEST_SET_ID"
  else
    expected="$ZH_EXPECTED"
    test_set="$ZH_TEST_SET_ID"
  fi
  root="$OUTPUT_ROOT/$language_id"
  args=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    args+=(--input "$root/manifests/manifest.shard-$(printf '%05d' "$shard")-of-00008.jsonl")
  done
  "$BASE_PY" "$MERGE_SCRIPT" "${args[@]}" \
    --merged-manifest "$root/merged_manifest.jsonl" \
    --successful-jsonl "$root/successful.jsonl" \
    --summary-json "$root/merge_summary.json" \
    --expected-shards "$NUM_SHARDS" --expected-cases "$expected" \
    --system-id "$SYSTEM_ID" --test-set-id "$test_set" --require-all-ok
}

schema_check_language() {
  language_id=$1
  if [ "$language_id" = "en" ]; then
    expected="$EN_EXPECTED"
    test_set="$EN_TEST_SET_ID"
  else
    expected="$ZH_EXPECTED"
    test_set="$ZH_TEST_SET_ID"
  fi
  root="$OUTPUT_ROOT/$language_id"
  schema_root="$root/schema"
  stem="${SYSTEM_ID}_${language_id}_strict"
  mkdir -p "$schema_root"
  env PYTHONPATH="$BATCH42_DEPS" "$BASE_PY" "$SCHEMA_SCRIPT" evaluate \
    --input "$root/successful.jsonl" --output-dir "$schema_root" --output-stem "$stem" \
    --run-id "batch42_${SYSTEM_ID}_${language_id}_${RUN_TAG}" \
    --system-id "$SYSTEM_ID" --test-set-id "$test_set" \
    --input-profile official_seedtts_vc --metric-profile seedtts_official --schema-only
  rows=$(wc -l < "$schema_root/${stem}.unified_eval.jsonl" | tr -d ' ')
  [ "$rows" = "$expected" ] || die "schema-only $language_id expected $expected rows, got $rows"
}

write_smoke_marker() {
  "$BASE_PY" - \
    "$OUTPUT_ROOT/en/manifests/manifest.smoke.jsonl" \
    "$RECORD_ROOT/summaries/en/summary.smoke.json" \
    "$IDENTITY_AUDIT_JOB" "$RECORD_ROOT/input_audit.job.json" "$SMOKE_MARKER" <<'PY'
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

manifest_path, summary_path, identity_path, inputs_path, marker_path = map(Path, sys.argv[1:])
rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
summary = json.loads(summary_path.read_text(encoding="utf-8"))
identity = json.loads(identity_path.read_text(encoding="utf-8"))
inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
if len(rows) != 1 or rows[0].get("status") != "ok":
    raise SystemExit(f"smoke requires exactly one newly generated ok row: {rows!r}")
if summary.get("status") != "complete" or summary.get("status_counts") != {"ok": 1}:
    raise SystemExit(f"invalid smoke summary: {summary!r}")
generated = Path(str(rows[0].get("generated_audio") or ""))
if not generated.is_file() or generated.stat().st_size < 1024:
    raise SystemExit(f"smoke generated audio missing/empty: {generated}")
inference = summary.get("inference_config") or {}
protocol_keys = (
    "mode",
    "engine_sha256",
    "source_semantic_model",
    "source_semantic_layer",
    "source_semantic_downsample_stride",
    "source_semantic_local_files_only",
    "temperature",
    "audio_temperature",
    "audio_top_p",
    "audio_top_k",
    "audio_repetition_penalty",
    "no_text_duration_budget_ratio",
    "no_text_max_token_margin",
    "no_text_soft_duration_budget",
    "filter_ref_content_leak",
    "timbre_cfg_scale",
    "ref_audio_cfg_scale",
    "ref_prompt_codec_permutation",
    "ref_speaker_prompt_slot",
    "timbre_side_only",
    "audio_segment_policy",
    "source_semantic_monotonic_bias_strength",
    "source_semantic_progress_clock",
    "source_semantic_release_after_progress",
    "source_semantic_release_start",
    "seed",
    "model_path",
    "code_root",
    "base_model_path",
)
model_files = identity.get("model_files") or {}
code_files = identity.get("code_files") or {}
wavlm_files = identity.get("wavlm_files") or {}
protocol_contract = {
    "system_id": "path_x_3k",
    "inference": {key: inference.get(key) for key in sorted(protocol_keys)},
    "asset_hashes": {
        "adapter_model": (model_files.get("adapter_model.safetensors") or {}).get("sha256"),
        "timbre_adapter": (model_files.get("timbre_memory_adapter.pt") or {}).get("sha256"),
        "timbre_config": (model_files.get("timbre_memory_config.json") or {}).get("sha256"),
        "engine": (code_files.get("scripts/004044_run_seedtts_validation_infer_persistent.py") or {}).get("sha256"),
        "wavlm_model": (wavlm_files.get("pytorch_model.bin") or {}).get("sha256"),
        "base_config": (identity.get("base_model_config") or {}).get("sha256"),
    },
    "strict_manifest_sha256": {
        "en": (inputs.get("en") or {}).get("sha256"),
        "zh": (inputs.get("zh") or {}).get("sha256"),
    },
}
protocol_fingerprint = hashlib.sha256(
    json.dumps(protocol_contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
payload = {
    "schema_version": "moss_codecvc.batch42_pathx_strict_smoke_completion.v1",
    "status": "smoke_complete",
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "system_id": "path_x_3k",
    "resource_contract": {"compute_group": "MTTS-3-2-0715", "gpu_type": "NVIDIA_H200_SXM_141G", "gpus": 8, "instances": 1},
    "registered_identity": identity,
    "strict_inputs": inputs,
    "protocol_contract": protocol_contract,
    "protocol_fingerprint_sha256": protocol_fingerprint,
    "actual_one_case": {
        "case_id": rows[0].get("case_id"),
        "generated_audio": str(generated.resolve()),
        "output_bytes": generated.stat().st_size,
        "runtime_seconds": rows[0].get("runtime_seconds"),
        "manifest_jsonl": str(manifest_path.resolve()),
        "summary_json": str(summary_path.resolve()),
        "inference_config": summary.get("inference_config"),
    },
}
marker_path.parent.mkdir(parents=True, exist_ok=True)
temporary = marker_path.with_name(f".{marker_path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, marker_path)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
  if ! validate_smoke_marker; then
    rm -f "$SMOKE_MARKER"
    return 1
  fi
  cp "$SMOKE_MARKER" "$RECORD_ROOT/smoke_completion.json"
}

write_full_marker() {
  "$BASE_PY" - "$OUTPUT_ROOT" "$RECORD_ROOT" "$IDENTITY_AUDIT_JOB" "$SMOKE_MARKER" "$FINAL_MARKER" <<'PY'
import datetime as dt
import json
import os
import sys
from pathlib import Path

output_root, record_root, identity_path, smoke_path, marker_path = map(Path, sys.argv[1:])
identity = json.loads(identity_path.read_text(encoding="utf-8"))
smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
sets = {}
for language, expected in (("en", 567), ("zh", 1194)):
    root = output_root / language
    merge = json.loads((root / "merge_summary.json").read_text(encoding="utf-8"))
    if merge.get("all_ok") is not True or merge.get("rows") != expected:
        raise SystemExit(f"invalid {language} merge: {merge!r}")
    successful = root / "successful.jsonl"
    schema = root / "schema" / f"path_x_3k_{language}_strict.unified_eval.jsonl"
    if sum(1 for line in successful.open(encoding="utf-8") if line.strip()) != expected:
        raise SystemExit(f"invalid {language} successful denominator")
    if sum(1 for line in schema.open(encoding="utf-8") if line.strip()) != expected:
        raise SystemExit(f"invalid {language} schema denominator")
    sets[language] = {
        "registered_cases": expected,
        "successful_jsonl": str(successful.resolve()),
        "schema_jsonl": str(schema.resolve()),
        "merge_summary": str((root / "merge_summary.json").resolve()),
    }
payload = {
    "schema_version": "moss_codecvc.batch42_pathx_strict_completion.v1",
    "status": "complete",
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "system_id": "path_x_3k",
    "resource_contract": {"compute_group": "MTTS-3-2-0715", "gpu_type": "NVIDIA_H200_SXM_141G", "gpus": 8, "instances": 1},
    "registered_identity": identity,
    "smoke_gate": {
        "marker": str(smoke_path.resolve()),
        "completed_utc": smoke.get("completed_utc"),
        "case_id": (smoke.get("actual_one_case") or {}).get("case_id"),
        "protocol_fingerprint_sha256": smoke.get("protocol_fingerprint_sha256"),
    },
    "strict_sets": sets,
    "output_root": str(output_root.resolve()),
    "record_root": str(record_root.resolve()),
}
marker_path.parent.mkdir(parents=True, exist_ok=True)
temporary = marker_path.with_name(f".{marker_path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, marker_path)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
  cp "$FINAL_MARKER" "$RECORD_ROOT/completion.json"
}

run_job_entrypoint() {
  mkdir -p "$RECORD_ROOT" "$OUTPUT_ROOT"
  exec >> "$RECORD_ROOT/run.log" 2>&1
  echo "[batch42-pathx] start=$(date -u +%Y-%m-%dT%H:%M:%SZ) mode=$MODE"
  echo "[batch42-pathx] model=$REGISTERED_MODEL_PATH"
  echo "[batch42-pathx] code_root=$REGISTERED_CODE_ROOT"
  echo "[batch42-pathx] output_root=$OUTPUT_ROOT"
  require_file "$PATHX_SCRIPT"
  require_file "$MERGE_SCRIPT"
  require_file "$SCHEMA_SCRIPT"
  require_file "$RECORD_ROOT/sha256sums.txt"
  sha256sum -c "$RECORD_ROOT/sha256sums.txt"
  rm -f "$FINAL_MARKER" "$RECORD_ROOT/completion.json"
  if [ "$MODE" = "smoke" ]; then
    rm -f "$RECORD_ROOT/smoke_completion.json"
  fi
  audit_allocated_gpus
  audit_strict_inputs > "$RECORD_ROOT/input_audit.job.json"
  audit_registered_identity "$IDENTITY_AUDIT_JOB" > "$RECORD_ROOT/identity_audit.job.log"
  prepare_inputs
  if [ "$MODE" = "smoke" ]; then
    run_smoke
    write_smoke_marker
    echo "[batch42-pathx] smoke_complete marker=$SMOKE_MARKER"
    return 0
  fi
  validate_smoke_marker
  run_actual_node_preflight > "$RECORD_ROOT/preflight_job_actual.log" 2>&1
  echo "[batch42-pathx] actual node preflight passed"
  run_all_workers
  merge_language en
  merge_language zh
  schema_check_language en
  schema_check_language zh
  write_full_marker
  echo "[batch42-pathx] complete marker=$FINAL_MARKER"
}

write_submission_plan() {
  smoke_gate_state="not_required"
  if [ "$MODE" = "full" ]; then
    if [ -s "$SMOKE_MARKER" ]; then
      smoke_gate_state="present_and_validated"
    else
      smoke_gate_state="deferred_for_dry_run_only"
    fi
  fi
  {
    printf 'job_name\tmode\tsystem\tcompute_group\tspec\tgpu_type\tinstances\tgpus\tshards\ten_cases\tzh_cases\tsmoke_gate\toutput_root\tentrypoint\n'
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$JOB_NAME" "$MODE" "$SYSTEM_ID" "$COMPUTE_GROUP" "$SPEC" \
      "$QZCLI_GPU_TYPE_OVERRIDE" "$INSTANCES" "$GPUS_PER_INSTANCE" "$NUM_SHARDS" \
      "$EN_EXPECTED" "$ZH_EXPECTED" "$smoke_gate_state" "$OUTPUT_ROOT" "$RUNNER"
  } > "$RECORD_ROOT/submission_plan.tsv"
}

validate_fixed_configuration

if [ "$ENTRYPOINT" = "1" ]; then
  run_job_entrypoint
  exit 0
fi

require_executable "$QZCLI"
require_executable "$BASE_PY"
require_executable "$MOSS_PY"
require_file "$SOURCE_PATHX_SCRIPT"
require_file "$SOURCE_MERGE_SCRIPT"
require_file "$SOURCE_SCHEMA_SCRIPT"
require_dir "$REGISTERED_MODEL_PATH"
require_dir "$REGISTERED_CODE_ROOT"
require_dir "$REGISTERED_WAVLM"
require_dir "$EN_INPUT_ROOT"
require_dir "$ZH_INPUT_ROOT"

mkdir -p "$RECORD_ROOT" "$OUTPUT_ROOT" "$QZCLI_HOME"
if [ "$DRY_RUN" = "0" ]; then
  if ! mkdir "$SUBMISSION_LOCK" 2>/dev/null; then
    die "another live submission attempt holds $SUBMISSION_LOCK"
  fi
  SUBMISSION_LOCK_HELD=1
  trap release_submission_lock EXIT
  trap 'release_submission_lock; exit 130' INT
  trap 'release_submission_lock; exit 143' TERM
  if [ "$MODE" = "full" ]; then
    validate_smoke_marker
  fi
  if [ "$FORCE" != "1" ] && { [ -s "$RECORD_ROOT/submitted_jobs.tsv" ] || [ -s "$FINAL_MARKER" ]; }; then
    die "existing live submission/completion; use FORCE=1 only for an intentional rerun"
  fi
elif [ "$MODE" = "full" ] && [ -s "$SMOKE_MARKER" ]; then
  validate_smoke_marker
fi

snapshot_scripts
audit_strict_inputs > "$RECORD_ROOT/input_audit.submit.json"
audit_registered_identity "$IDENTITY_AUDIT_SUBMIT" > "$RECORD_ROOT/identity_audit.submit.log"
prepare_inputs
engine_schema_preflight > "$RECORD_ROOT/engine_preflight.log" 2>&1
write_submission_plan

COMMAND="bash $RUNNER"
SUBMIT_OUTPUT="$RECORD_ROOT/submit_output.txt"

echo "Batch-42 Path X strict QZ plan"
echo "  MODE=$MODE"
echo "  SYSTEM_ID=$SYSTEM_ID"
echo "  JOB_NAME=$JOB_NAME"
echo "  MODEL=$REGISTERED_MODEL_PATH"
echo "  CODE_ROOT=$REGISTERED_CODE_ROOT"
echo "  EN=$EN_EXPECTED ZH=$ZH_EXPECTED"
echo "  WORKERS=8 (one shard per H200; EN then ZH)"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP (MTTS-3-2-0715 only)"
echo "  SPEC=$SPEC INSTANCES=$INSTANCES GPU_TYPE=$QZCLI_GPU_TYPE_OVERRIDE"
echo "  SMOKE_MARKER=$SMOKE_MARKER"
echo "  OUTPUT_ROOT=$OUTPUT_ROOT"
echo "  RECORD_ROOT=$RECORD_ROOT"
echo "  DRY_RUN=$DRY_RUN"

set +e
if [ "$DRY_RUN" = "1" ]; then
  qz_output=$(
    env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
      HOME="$QZCLI_HOME" QZCLI_GPU_TYPE_OVERRIDE="$QZCLI_GPU_TYPE_OVERRIDE" \
      "$QZCLI" create-job \
        --name "$JOB_NAME" --command "$COMMAND" --workspace "$WORKSPACE" \
        --project "$PROJECT" --compute-group "$COMPUTE_GROUP" --spec "$SPEC" \
        --image "$IMAGE" --image-type "$IMAGE_TYPE" --instances "$INSTANCES" \
        --shm "$SHM_GI" --priority "$PRIORITY" --framework "$FRAMEWORK" --dry-run 2>&1
  )
  qz_status=$?
else
  qz_output=$(
    env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
      HOME="$QZCLI_HOME" QZCLI_GPU_TYPE_OVERRIDE="$QZCLI_GPU_TYPE_OVERRIDE" \
      "$QZCLI" create-job \
        --name "$JOB_NAME" --command "$COMMAND" --workspace "$WORKSPACE" \
        --project "$PROJECT" --compute-group "$COMPUTE_GROUP" --spec "$SPEC" \
        --image "$IMAGE" --image-type "$IMAGE_TYPE" --instances "$INSTANCES" \
        --shm "$SHM_GI" --priority "$PRIORITY" --framework "$FRAMEWORK" 2>&1
  )
  qz_status=$?
fi
set -e

printf '%s\n' "$qz_output" > "$SUBMIT_OUTPUT"
printf '%s\n' "$qz_output"
[ "$qz_status" = "0" ] || die "QZ create-job failed; see $SUBMIT_OUTPUT"

if [ "$DRY_RUN" = "1" ]; then
  {
    printf 'job_name\tmode\tsystem\tcompute_group\tspec\tinstances\tgpu_type\tstatus\n'
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\tdry_run_only\n' \
      "$JOB_NAME" "$MODE" "$SYSTEM_ID" "$COMPUTE_GROUP" "$SPEC" "$INSTANCES" "$QZCLI_GPU_TYPE_OVERRIDE"
  } > "$RECORD_ROOT/dry_run_jobs.tsv"
  echo "[batch42-pathx] dry-run passed; no QZ job submitted"
  exit 0
fi

mapfile -t job_ids < <(
  printf '%s\n' "$qz_output" |
    grep -Eo 'job-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}' |
    sort -u || true
)
[ "${#job_ids[@]}" = "1" ] || \
  die "QZ returned success but expected exactly one unique complete UUID job ID; got ${#job_ids[@]}"
job_id=${job_ids[0]}
case "$job_id" in
  job-????????-????-????-????-????????????) ;;
  *) die "QZ returned success but no complete UUID job ID was parsed" ;;
esac
submitted_tmp="$RECORD_ROOT/.submitted_jobs.tsv.tmp-$$"
{
  printf 'job_name\tjob_id\tmode\tsystem\tcompute_group\tspec\tinstances\tgpu_type\toutput_root\trecord_root\tentrypoint\n'
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$JOB_NAME" "$job_id" "$MODE" "$SYSTEM_ID" "$COMPUTE_GROUP" "$SPEC" \
    "$INSTANCES" "$QZCLI_GPU_TYPE_OVERRIDE" "$OUTPUT_ROOT" "$RECORD_ROOT" "$RUNNER"
} > "$submitted_tmp"
mv "$submitted_tmp" "$RECORD_ROOT/submitted_jobs.tsv"
echo "[batch42-pathx] submitted job_id=$job_id"

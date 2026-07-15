#!/bin/sh
# Batch-44 (2026-07-13): paired ver2.9.5 v1-data 30k controls.
#
# Both arms use the exact 20260709 v1 no_text/text manifests and the frozen
# Batch-43 Path-X recipe.  The only scientific difference inside this pair is
# TEXT_REPEAT=3 versus TEXT_REPEAT=5.
#
# Safe default: generate/audit both runners and execute QZ payload dry-runs.
#   sh scripts/002056_submit_ver23_batch44_v1_r3_r5_30k_qz.sh
#
# Live requires two explicit switches; DRY_RUN=0 alone is rejected.
#   LIVE=1 DRY_RUN=0 sh scripts/002056_submit_ver23_batch44_v1_r3_r5_30k_qz.sh

set -eu

PROJECT_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
FROZEN_CODE_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch3436_20260710_1092820"
BASE_MODEL_CONFIG="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction/MOSS-TTS/config.json"
QZCLI="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/qzcli_with_deps.sh"
QZCLI_HOME="/inspire/ssd/project/embodied-multimodality/public/xyzhang/.codex/qzcli_home"

STAMP="20260713"
DRY_RUN="${DRY_RUN:-1}"
LIVE="${LIVE:-0}"
if [ "$LIVE" = "1" ]; then
  VERIFY_FULL_SHA256="${VERIFY_FULL_SHA256:-1}"
else
  VERIFY_FULL_SHA256="${VERIFY_FULL_SHA256:-0}"
fi

WORKSPACE="CI-情境智能"
PROJECT="CI-情境智能"
EXPECTED_WORKSPACE_ID="ws-8207e9e2-e733-4eec-a475-cfa1c36480ba"
EXPECTED_PROJECT_ID="project-c67c548f-f02c-453b-ba5b-8745db6886e7"
ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122" # MTTS-3-2-0715
ALLOWED_SPEC="67b10bc6-78b0-41a3-aaf4-358eeeb99009"
ALLOWED_INSTANCES="1"
ALLOWED_GPU_TYPE="NVIDIA_H200_SXM_141G"
ALLOWED_ACCELERATE_CONFIG="configs/accelerate_fsdp_h200_8gpu_no_ckpt.yaml"
COMPUTE_GROUP="${COMPUTE_GROUP:-$ALLOWED_COMPUTE_GROUP}"
SPEC="${SPEC:-$ALLOWED_SPEC}"
INSTANCES="${INSTANCES:-$ALLOWED_INSTANCES}"
QZCLI_GPU_TYPE_OVERRIDE="${QZCLI_GPU_TYPE_OVERRIDE:-$ALLOWED_GPU_TYPE}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$ALLOWED_ACCELERATE_CONFIG}"
IMAGE="docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420"
IMAGE_TYPE="SOURCE_PRIVATE"
FRAMEWORK="pytorch"
SHM_GI="1200"
PRIORITY="10"

PAIR_ID="ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_${STAMP}"
PAIR_RECORD_ROOT="$PROJECT_ROOT/trainset/qz_jobs/$PAIR_ID"
R3_RECORD_ROOT="$PAIR_RECORD_ROOT/r3"
R5_RECORD_ROOT="$PAIR_RECORD_ROOT/r5"
R3_JOB_NAME="ver2_9_5_final_r3_v1_30k"
R5_JOB_NAME="ver2_9_5_final_r5_v1_30k"
R3_BATCH_ID="ver23_batch44_${R3_JOB_NAME}_${STAMP}"
R5_BATCH_ID="ver23_batch44_${R5_JOB_NAME}_${STAMP}"
R3_OUT_DIR="$PROJECT_ROOT/outputs/lora_runs/$R3_JOB_NAME"
R5_OUT_DIR="$PROJECT_ROOT/outputs/lora_runs/$R5_JOB_NAME"

# Exact v1 inputs requested for both arms.
V1_DIR="$PROJECT_ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709"
NO_TEXT_TRAIN_JSONL="$V1_DIR/no_text.train.jsonl"
TEXT_TRAIN_JSONL="$V1_DIR/text.train.jsonl"
NO_TEXT_EXPECTED_ROWS="310420"
NO_TEXT_EXPECTED_BYTES="18048211813"
NO_TEXT_EXPECTED_SHA256="c4b061f0a968e73710dc86d81478483a9195e8a053f510f09be7952d60c3d279"
TEXT_EXPECTED_ROWS="32419"
TEXT_EXPECTED_BYTES="2196087856"
TEXT_EXPECTED_SHA256="c6632888d08e79382001909a65951d6ce7bab80d7fb585cf7729e0a9188a9a80"

# Exact Batch-43 r3/r5 evidence used to prove recipe equivalence.
BATCH43_PAIR_ROOT="$PROJECT_ROOT/trainset/qz_jobs/ver23_batch43_ver2_9_5_final_r3_r5_v2_30k_20260712"
B43_R3_CORE="$BATCH43_PAIR_ROOT/r3/train_args_dry_run_core.json"
B43_R3_RUNNER="$BATCH43_PAIR_ROOT/r3/run_train_entrypoint.sh"
B43_R5_CORE="$BATCH43_PAIR_ROOT/r5/train_args_dry_run_core.json"
B43_R5_RUNNER="$BATCH43_PAIR_ROOT/r5/run_train_entrypoint.sh"
B43_R3_CORE_SHA256="78525edc2e039e3f2c68dd845aa716966b4c11c560697a6d126a9ec12d17724c"
B43_R3_RUNNER_SHA256="ed8aab069163bf3f6de21b23cbe3f4a4babb5dae3362132b9e69bb83491cf17e"
B43_R5_CORE_SHA256="d863f7579dfab905e99f3a0b9980abe310cc51c3a4aadc0292ce5ca6f4ebba9f"
B43_R5_RUNNER_SHA256="87911d4c63035b3fffb945d50a64cd255acff72392b783a7ae1256830aee56c6"
B43_V2_DIR="$PROJECT_ROOT/trainset/ver2_9_prepared_v2_real_no_text_refdecorr_wavlm_sv_20260708"
B43_NO_TEXT="$B43_V2_DIR/no_text.v2.train.jsonl"
B43_TEXT="$B43_V2_DIR/text.train.jsonl"
B43_R3_JOB="ver2_9_5_final_r3_v2_30k"
B43_R5_JOB="ver2_9_5_final_r5_v2_30k"
B43_R3_OUT="$PROJECT_ROOT/outputs/lora_runs/$B43_R3_JOB"
B43_R5_OUT="$PROJECT_ROOT/outputs/lora_runs/$B43_R5_JOB"

case "$DRY_RUN:$LIVE:$VERIFY_FULL_SHA256" in
  [01]:[01]:[01]) ;;
  *) echo "ERROR: DRY_RUN, LIVE, and VERIFY_FULL_SHA256 must be 0 or 1" >&2; exit 2 ;;
esac
if [ "$LIVE" = "0" ] && [ "$DRY_RUN" = "0" ]; then
  echo "ERROR: DRY_RUN=0 alone is insufficient; live pair requires LIVE=1 DRY_RUN=0" >&2
  exit 2
fi
if [ "$LIVE" = "1" ] && [ "$DRY_RUN" != "0" ]; then
  echo "ERROR: LIVE=1 requires DRY_RUN=0" >&2
  exit 2
fi
if [ "$LIVE" = "1" ] && [ "$VERIFY_FULL_SHA256" != "1" ]; then
  echo "ERROR: live pair requires full SHA256 verification" >&2
  exit 2
fi
if [ "$COMPUTE_GROUP" != "$ALLOWED_COMPUTE_GROUP" ]; then
  echo "ERROR: Batch-44 may only use MTTS-3-2-0715 ($ALLOWED_COMPUTE_GROUP)" >&2
  exit 2
fi
if [ "$SPEC" != "$ALLOWED_SPEC" ] || [ "$INSTANCES" != "$ALLOWED_INSTANCES" ]; then
  echo "ERROR: each Batch-44 arm requires spec=$ALLOWED_SPEC and instances=1" >&2
  exit 2
fi
if [ "$QZCLI_GPU_TYPE_OVERRIDE" != "$ALLOWED_GPU_TYPE" ]; then
  echo "ERROR: each Batch-44 arm requires $ALLOWED_GPU_TYPE" >&2
  exit 2
fi
if [ "$ACCELERATE_CONFIG" != "$ALLOWED_ACCELERATE_CONFIG" ]; then
  echo "ERROR: Batch-44 requires $ALLOWED_ACCELERATE_CONFIG" >&2
  exit 2
fi
if [ ! -x "$QZCLI" ] || [ ! -d "$QZCLI_HOME" ]; then
  echo "ERROR: fixed qzcli-local wrapper/HOME is unavailable" >&2
  exit 1
fi

mkdir -p "$PAIR_RECORD_ROOT" "$R3_RECORD_ROOT" "$R5_RECORD_ROOT"

audit_frozen_baseline() {
  python - \
    "$FROZEN_CODE_ROOT" "$BASE_MODEL_CONFIG" \
    "$B43_R3_CORE" "$B43_R3_CORE_SHA256" "$B43_R3_RUNNER" "$B43_R3_RUNNER_SHA256" \
    "$B43_R5_CORE" "$B43_R5_CORE_SHA256" "$B43_R5_RUNNER" "$B43_R5_RUNNER_SHA256" \
    "$BATCH43_PAIR_ROOT/pair_control_audit.json" "$PAIR_RECORD_ROOT/frozen_baseline_audit.json" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
base_config = Path(sys.argv[2])
r3_core, r3_core_sha, r3_runner, r3_runner_sha = Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5]), sys.argv[6]
r5_core, r5_core_sha, r5_runner, r5_runner_sha = Path(sys.argv[7]), sys.argv[8], Path(sys.argv[9]), sys.argv[10]
pair_audit, output = Path(sys.argv[11]), Path(sys.argv[12])
expected_files = {
    "scripts/002049_submit_ver23_content_side_3k_qz.sh": "e25e4a0d0774181191e07d764b188fa6dae5a6517792be665f9fe101e381d356",
    "scripts/002004_submit_ver2_lora_68w_h200_qz.sh": "3eab2e399b5af8925a36fc8a7b7a39c537b822531c42fa850fd5dd49d3a1ab16",
    "scripts/002002_train_moss_codecvc_lora.py": "3ed70e6c1cb08f81f0c5accd46efa4340df710f92c880601b282fdd38714d7c5",
    "moss_codecvc/models/moss_codecvc_wrapper.py": "5815c8ab5e0aab69d19328fd01782620064327eaf5f39cc4923df8ce3ae9ca42",
    "moss_codecvc/models/content_cross_attn.py": "a8e4cd12d279cfff7c38e3e2d8b21b55d70c403cec654edf7ef77de58acba66a",
    "configs/remote_full.yaml": "0cb3bb7c9d2fdaa050e30bbb06ff55e654b2b411c6d83325daca03c49c9245fd",
    "configs/accelerate_fsdp_h200_8gpu_no_ckpt.yaml": "438c5f5f9dc66f081fa9cc8e9861c4709ef3bc4e3fd6de9532c79f3210e7322e",
}

def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

errors, actual = [], {}
for relative, wanted in expected_files.items():
    path = root / relative
    if not path.is_file():
        errors.append(f"missing frozen file: {path}")
        continue
    got = digest(path)
    actual[relative] = got
    if got != wanted:
        errors.append(f"frozen drift {relative}: {got} != {wanted}")
for path, wanted, label in (
    (base_config, "214fc997d98f51ab57925a5939afc6280e76044198b664221622e70d098ed06e", "base config"),
    (r3_core, r3_core_sha, "Batch-43 r3 core"),
    (r3_runner, r3_runner_sha, "Batch-43 r3 runner"),
    (r5_core, r5_core_sha, "Batch-43 r5 core"),
    (r5_runner, r5_runner_sha, "Batch-43 r5 runner"),
):
    if not path.is_file() or digest(path) != wanted:
        errors.append(f"{label} missing or hash drifted: {path}")
pair = json.loads(pair_audit.read_text(encoding="utf-8"))
if pair.get("status") != "pass" or pair.get("only_intended_scientific_delta") != {"TEXT_REPEAT": {"r3": 3, "r5": 5}}:
    errors.append("Batch-43 pair audit no longer proves r3/r5 repeat-only control")
training = (root / "scripts/002002_train_moss_codecvc_lora.py").read_text(encoding="utf-8")
if not re.search(r'add_argument\("--seed",\s*type=int,\s*default=42\)', training):
    errors.append("frozen parser no longer proves training seed=42")
language = json.loads(base_config.read_text(encoding="utf-8")).get("language_config") or {}
for key, wanted in {"hidden_size": 4096, "num_attention_heads": 32, "num_hidden_layers": 36}.items():
    if language.get(key) != wanted:
        errors.append(f"language_config.{key} drifted")
if errors:
    raise SystemExit("Batch-44 frozen baseline audit failed:\n- " + "\n- ".join(errors))
payload = {
    "status": "pass",
    "frozen_code_root": str(root),
    "registered_git_head": "109282047f19886b8f2b38c4cec67092386c172f",
    "frozen_files_sha256": actual,
    "batch43_r3": {"core": str(r3_core), "core_sha256": r3_core_sha, "runner": str(r3_runner), "runner_sha256": r3_runner_sha},
    "batch43_r5": {"core": str(r5_core), "core_sha256": r5_core_sha, "runner": str(r5_runner), "runner_sha256": r5_runner_sha},
    "training_seed": 42,
    "base_language_config": {"hidden_size": 4096, "num_attention_heads": 32, "num_hidden_layers": 36},
    "content_cross_attn_layers_all_resolves_to": 36,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[batch44-code-audit] PASS output={output}")
PY
}

audit_v1_inputs() {
  python - \
    "$NO_TEXT_TRAIN_JSONL" "$V1_DIR/no_text.train.jsonl.offsets.u64.json" \
    "$NO_TEXT_EXPECTED_ROWS" "$NO_TEXT_EXPECTED_BYTES" "$NO_TEXT_EXPECTED_SHA256" \
    "$TEXT_TRAIN_JSONL" "$V1_DIR/text.train.jsonl.offsets.u64.json" \
    "$TEXT_EXPECTED_ROWS" "$TEXT_EXPECTED_BYTES" "$TEXT_EXPECTED_SHA256" \
    "$VERIFY_FULL_SHA256" "$PAIR_RECORD_ROOT/input_identity.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

no_text, no_index = Path(sys.argv[1]), Path(sys.argv[2])
no_rows, no_bytes, no_sha = int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
text, text_index = Path(sys.argv[6]), Path(sys.argv[7])
text_rows, text_bytes, text_sha = int(sys.argv[8]), int(sys.argv[9]), sys.argv[10]
verify_full = sys.argv[11] == "1"
output = Path(sys.argv[12])
errors = []
no_meta = json.loads(no_index.read_text(encoding="utf-8"))
text_meta = json.loads(text_index.read_text(encoding="utf-8"))
for label, path, meta, rows, size in (
    ("no_text", no_text, no_meta, no_rows, no_bytes),
    ("text", text, text_meta, text_rows, text_bytes),
):
    if not path.is_file() or path.stat().st_size != size:
        errors.append(f"{label} missing/size mismatch: {path}")
    if meta.get("rows") != rows or meta.get("source_size") != size:
        errors.append(f"{label} index row/size mismatch: {meta}")
    if Path(str(meta.get("source_path") or "")).resolve() != path.resolve():
        errors.append(f"{label} index source path mismatch")
actual = {}
if verify_full:
    for label, path, wanted in (("no_text", no_text, no_sha), ("text", text, text_sha)):
        with path.open("rb") as handle:
            got = hashlib.file_digest(handle, "sha256").hexdigest()
        actual[label] = got
        if got != wanted:
            errors.append(f"{label} SHA256={got}, expected {wanted}")
with no_text.open(encoding="utf-8") as handle:
    first_no = json.loads(handle.readline())
with text.open(encoding="utf-8") as handle:
    first_text = json.loads(handle.readline())
if first_no.get("moss_codecvc_mode") != "no_text" or first_text.get("moss_codecvc_mode") != "text":
    errors.append("first-row moss_codecvc_mode mismatch")
for key in (
    "reference_audio_codes", "audio_codes", "content_token_ids",
    "source_wavlm_bnf_features_path", "speaker_vec_path",
    "source_speaker_embedding_path", "timbre_ref_speaker_embedding_path",
    "target_speaker_embedding_path",
):
    if first_no.get(key) in (None, "", []):
        errors.append(f"v1 no_text first row missing {key}")
if errors:
    raise SystemExit("Batch-44 v1 input audit failed:\n- " + "\n- ".join(errors))
payload = {
    "status": "pass",
    "full_sha256_verified": verify_full,
    "no_text": {"path": str(no_text), "rows": no_rows, "bytes": no_bytes, "sha256": actual.get("no_text", no_sha), "repeat": 1},
    "text": {"path": str(text), "rows": text_rows, "bytes": text_bytes, "sha256": actual.get("text", text_sha)},
    "arms": {
        "r3": {"text_repeat": 3, "effective_rows": no_rows + text_rows * 3, "text_share": (text_rows * 3) / (no_rows + text_rows * 3)},
        "r5": {"text_repeat": 5, "effective_rows": no_rows + text_rows * 5, "text_share": (text_rows * 5) / (no_rows + text_rows * 5)},
    },
}
rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
output.write_text(rendered, encoding="utf-8")
if verify_full:
    output.with_name("input_identity.full_sha256.json").write_text(rendered, encoding="utf-8")
print(f"[batch44-input-audit] PASS no_text={no_rows} text={text_rows} sha256={'verified' if verify_full else 'registered'}")
PY
}

arm_values() {
  case "$1" in
    3)
      ARM_KEY="r3"; ARM_JOB_NAME="$R3_JOB_NAME"; ARM_BATCH_ID="$R3_BATCH_ID"
      ARM_RECORD_ROOT="$R3_RECORD_ROOT"; ARM_OUT_DIR="$R3_OUT_DIR"
      ARM_B43_CORE="$B43_R3_CORE"; ARM_B43_RUNNER="$B43_R3_RUNNER"
      ARM_B43_JOB="$B43_R3_JOB"; ARM_B43_OUT="$B43_R3_OUT"
      ;;
    5)
      ARM_KEY="r5"; ARM_JOB_NAME="$R5_JOB_NAME"; ARM_BATCH_ID="$R5_BATCH_ID"
      ARM_RECORD_ROOT="$R5_RECORD_ROOT"; ARM_OUT_DIR="$R5_OUT_DIR"
      ARM_B43_CORE="$B43_R5_CORE"; ARM_B43_RUNNER="$B43_R5_RUNNER"
      ARM_B43_JOB="$B43_R5_JOB"; ARM_B43_OUT="$B43_R5_OUT"
      ;;
    *) echo "ERROR: unsupported text repeat $1" >&2; exit 2 ;;
  esac
  ARM_TEXT_REPEAT="$1"
  ARM_TRAIN_JSONL_SPEC="$NO_TEXT_TRAIN_JSONL::repeat=1,$TEXT_TRAIN_JSONL::repeat=$ARM_TEXT_REPEAT"
}

run_arm() {
  arm_values "$1"
  requested_dry_run="$2"
  env -i \
    PATH="$PATH" HOME="$QZCLI_HOME" QZCLI="$QZCLI" \
    ROOT="$FROZEN_CODE_ROOT" DRY_RUN="$requested_dry_run" \
    WORKSPACE="$WORKSPACE" PROJECT="$PROJECT" COMPUTE_GROUP="$COMPUTE_GROUP" \
    SPEC="$SPEC" INSTANCES="$INSTANCES" QZCLI_GPU_TYPE_OVERRIDE="$QZCLI_GPU_TYPE_OVERRIDE" \
    ACCELERATE_CONFIG="$ACCELERATE_CONFIG" \
    TRAINSET_DIR="$V1_DIR" TRAIN_JSONL="$NO_TEXT_TRAIN_JSONL" \
    NO_TEXT_TRAIN_JSONL="$NO_TEXT_TRAIN_JSONL" TEXT_TRAIN_JSONL="$TEXT_TRAIN_JSONL" \
    TEXT_REPEAT="$ARM_TEXT_REPEAT" TRAIN_JSONL_SPEC="$ARM_TRAIN_JSONL_SPEC" \
    BATCH_ID="$ARM_BATCH_ID" JOB_NAME="$ARM_JOB_NAME" JOB_NAME_PREFIX="$ARM_JOB_NAME" \
    QZ_RECORD_ROOT="$ARM_RECORD_ROOT" OUT_DIR="$ARM_OUT_DIR" \
    NUM_EPOCHS="6" MAX_TRAIN_STEPS="30000" SAVE_STEPS="2000" EVAL_STEPS="2000" \
    EVAL_MAX_BATCHES="0" EVAL_NUM_WORKERS="0" LEARNING_RATE="1e-5" \
    LR_SCHEDULER_TYPE="constant_with_warmup" WARMUP_RATIO="0.03" WEIGHT_DECAY="0.01" \
    PER_DEVICE_BATCH_SIZE="1" GRADIENT_ACCUMULATION_STEPS="8" GPU_COUNT="8" \
    MIXED_PRECISION="bf16" GRADIENT_CHECKPOINTING="0" \
    LORA_R="16" LORA_ALPHA="32" LORA_DROPOUT="0.05" \
    LOGGING_STEPS="20" NUM_WORKERS="4" MAX_GRAD_NORM="1.0" POST_TRAIN_QUICK_EVAL="0" \
    RESUME_ADAPTER_PATH="" TRAIN_SOURCE_SEMANTIC_ONLY="0" FREEZE_LORA="0" \
    FREEZE_ROLE_ROUTING="0" FREEZE_TIMBRE_ADAPTER="0" \
    EVAL_JSONL="" EVAL_JSONL_SPEC="" EVAL_SEEN_JSONL="" EVAL_SEEN_JSONL_SPEC="" \
    EVAL_UNSEEN_JSONL="" EVAL_UNSEEN_JSONL_SPEC="" \
    CONTENT_CROSS_ATTN_LAYERS="all" CONTENT_CROSS_ATTN_FEATURE_DIM="768" \
    CONTENT_CROSS_ATTN_GATE_INIT="-0.5" CONTENT_CROSS_ATTN_OUTPUT_SCALE="0.3" \
    CONTENT_CROSS_ATTN_DROPOUT="0.0" CONTENT_ENCODER_LAYERS="2" \
    CONTENT_ENCODER_CONV_KERNEL_SIZE="7" GUIDED_ATTN_LOSS_WEIGHT="0.05" \
    GUIDED_ATTN_WARMUP_STEPS="1000" GUIDED_ATTN_BAND_FRAMES="3" \
    PHONEME_CLASSIFIER_LOSS_WEIGHT="0.02" CONTENT_CTC_WEIGHT="0.0" \
    TIMBRE_ADAPTER_GATE_LR_MULTIPLIER="1.0" \
    sh "$FROZEN_CODE_ROOT/scripts/002049_submit_ver23_content_side_3k_qz.sh"
}

audit_pair_config() {
  python - \
    "$R3_RECORD_ROOT/train_args_dry_run_core.json" "$R5_RECORD_ROOT/train_args_dry_run_core.json" \
    "$R3_RECORD_ROOT/run_train_entrypoint.sh" "$R5_RECORD_ROOT/run_train_entrypoint.sh" \
    "$B43_R3_CORE" "$B43_R5_CORE" "$B43_R3_RUNNER" "$B43_R5_RUNNER" \
    "$NO_TEXT_TRAIN_JSONL" "$TEXT_TRAIN_JSONL" "$B43_NO_TEXT" "$B43_TEXT" \
    "$R3_JOB_NAME" "$R5_JOB_NAME" "$B43_R3_JOB" "$B43_R5_JOB" \
    "$R3_OUT_DIR" "$R5_OUT_DIR" "$B43_R3_OUT" "$B43_R5_OUT" \
    "$PAIR_RECORD_ROOT/pair_control_audit.json" "$PAIR_RECORD_ROOT/evaluation_contract.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

r3_core_path, r5_core_path, r3_runner_path, r5_runner_path = map(Path, sys.argv[1:5])
b43_r3_core_path, b43_r5_core_path, b43_r3_runner_path, b43_r5_runner_path = map(Path, sys.argv[5:9])
v1_no, v1_text, v2_no, v2_text = sys.argv[9:13]
r3_job, r5_job, b43_r3_job, b43_r5_job = sys.argv[13:17]
r3_out, r5_out, b43_r3_out, b43_r5_out = sys.argv[17:21]
output, eval_output = map(Path, sys.argv[21:23])
r3 = json.loads(r3_core_path.read_text(encoding="utf-8"))
r5 = json.loads(r5_core_path.read_text(encoding="utf-8"))
b43_r3 = json.loads(b43_r3_core_path.read_text(encoding="utf-8"))
b43_r5 = json.loads(b43_r5_core_path.read_text(encoding="utf-8"))
errors = []
common = {
    "NO_TEXT_TRAIN_JSONL": v1_no, "TEXT_TRAIN_JSONL": v1_text,
    "MAX_TRAIN_STEPS": "30000", "SAVE_STEPS": "2000", "EVAL_STEPS": "2000",
    "LEARNING_RATE": "1e-5", "LR_SCHEDULER_TYPE": "constant_with_warmup", "WARMUP_RATIO": "0.03",
    "PER_DEVICE_BATCH_SIZE": "1", "GRADIENT_ACCUMULATION_STEPS": "8", "GPU_COUNT": "8",
    "USE_TIMBRE_MEMORY": "0", "ENABLE_SPEAKER_SIDE_PATHWAY": "0", "ENABLE_SPEAKER_CROSS_ATTN": "0",
    "ENABLE_SOURCE_SEMANTIC_MEMORY": "0", "TARGET_FRONT_CE_WEIGHT": "4.0", "TARGET_FRONT_CE_SECONDS": "0.75",
    "PROGRESS_LOSS_WEIGHT": "0.10", "STOP_LOSS_WEIGHT": "0.20", "ENABLE_CONTENT_CROSS_ATTN": "1",
    "CONTENT_CROSS_ATTN_LAYERS": "all", "CONTENT_CROSS_ATTN_FEATURE_DIM": "768",
    "CONTENT_CROSS_ATTN_GATE_INIT": "-0.5", "CONTENT_CROSS_ATTN_OUTPUT_SCALE": "0.3",
    "CONTENT_ENCODER_LAYERS": "2", "GUIDED_ATTN_LOSS_WEIGHT": "0.05", "GUIDED_ATTN_WARMUP_STEPS": "1000",
    "GUIDED_ATTN_BAND_FRAMES": "3", "PHONEME_CLASSIFIER_LOSS_WEIGHT": "0.02", "CONTENT_CTC_WEIGHT": "0.0",
    "SOURCE_CONTENT_MEMORY_TYPE": "wavlm_bnf_continuous", "sequence_structure": "[text?, C_src frames, C_ref frames]",
}
for label, payload in (("r3", r3), ("r5", r5)):
    for key, wanted in common.items():
        if payload.get(key) != wanted:
            errors.append(f"{label}: {key}={payload.get(key)!r}, expected {wanted!r}")
for label, payload, repeat, job, out in (
    ("r3", r3, 3, r3_job, r3_out), ("r5", r5, 5, r5_job, r5_out)
):
    spec = f"{v1_no}::repeat=1,{v1_text}::repeat={repeat}"
    for key, wanted in {"TEXT_REPEAT": str(repeat), "TRAIN_JSONL_SPEC": spec, "JOB_NAME_PREFIX": job, "OUT_DIR": out}.items():
        if payload.get(key) != wanted:
            errors.append(f"{label}: {key} mismatch")

pair_allowed = sorted(["BATCH_ID", "JOB_NAME_PREFIX", "OUT_DIR", "TEXT_REPEAT", "TRAIN_JSONL_SPEC"])
pair_diffs = sorted(key for key in set(r3) | set(r5) if r3.get(key) != r5.get(key))
if pair_diffs != pair_allowed:
    errors.append(f"v1 pair unexpected deltas={pair_diffs}; allowed={pair_allowed}")
def drop(payload, keys):
    result = dict(payload)
    for key in keys:
        result.pop(key, None)
    return result
if drop(r3, pair_allowed) != drop(r5, pair_allowed):
    errors.append("v1 pair normalized core configs differ")

baseline_allowed = sorted(["BATCH_ID", "JOB_NAME_PREFIX", "NO_TEXT_TRAIN_JSONL", "TEXT_TRAIN_JSONL", "OUT_DIR", "TRAIN_JSONL_SPEC"])
for label, current, baseline in (("r3", r3, b43_r3), ("r5", r5, b43_r5)):
    diffs = sorted(key for key in set(current) | set(baseline) if current.get(key) != baseline.get(key))
    if diffs != baseline_allowed:
        errors.append(f"{label} vs Batch-43 unexpected core deltas={diffs}; allowed={baseline_allowed}")
    if drop(current, baseline_allowed) != drop(baseline, baseline_allowed):
        errors.append(f"{label} vs Batch-43 normalized core differs")

def normalize_pair_runner(path, spec, out_dir, job, repeat):
    text = Path(path).read_text(encoding="utf-8")
    for old, new in ((spec, "<SPEC>"), (out_dir, "<OUT>"), (job, "<JOB>"), (f"text_repeat={repeat}", "text_repeat=<R>")):
        text = text.replace(old, new)
    return text
r3_spec = f"{v1_no}::repeat=1,{v1_text}::repeat=3"
r5_spec = f"{v1_no}::repeat=1,{v1_text}::repeat=5"
nr3 = normalize_pair_runner(r3_runner_path, r3_spec, r3_out, r3_job, 3)
nr5 = normalize_pair_runner(r5_runner_path, r5_spec, r5_out, r5_job, 5)
if nr3 != nr5:
    errors.append("v1 generated runners differ beyond repeat/job/output identity")

def normalize_vs_baseline(path, no_path, text_path, out_dir, job):
    text = Path(path).read_text(encoding="utf-8")
    for old, new in ((no_path, "<NO>"), (text_path, "<TEXT>"), (out_dir, "<OUT>"), (job, "<JOB>")):
        text = text.replace(old, new)
    return text
for label, current_path, baseline_path, current_out, baseline_out, current_job, baseline_job in (
    ("r3", r3_runner_path, b43_r3_runner_path, r3_out, b43_r3_out, r3_job, b43_r3_job),
    ("r5", r5_runner_path, b43_r5_runner_path, r5_out, b43_r5_out, r5_job, b43_r5_job),
):
    current_norm = normalize_vs_baseline(current_path, v1_no, v1_text, current_out, current_job)
    baseline_norm = normalize_vs_baseline(baseline_path, v2_no, v2_text, baseline_out, baseline_job)
    if current_norm != baseline_norm:
        errors.append(f"{label} runner differs from corresponding Batch-43 runner beyond v1 inputs/job/output")
    if "--seed" in current_norm or "--seed" in baseline_norm:
        errors.append(f"{label} runner unexpectedly overrides training seed")
if errors:
    raise SystemExit("Batch-44 pair config audit failed:\n- " + "\n- ".join(errors))

normalized_core = json.dumps(drop(r3, pair_allowed), sort_keys=True, separators=(",", ":")).encode()
payload = {
    "status": "pass",
    "pair": {"r3": r3_job, "r5": r5_job},
    "only_intended_scientific_delta_within_batch44": {"TEXT_REPEAT": {"r3": 3, "r5": 5}},
    "shared_v1_inputs": {"no_text": v1_no, "text": v1_text},
    "batch43_to_batch44_common_scientific_change": {"data_version": {"from": "v2 no_text + 20260707 text", "to": "v1 20260709 no_text + text"}},
    "identity_only_arm_fields": ["BATCH_ID", "JOB_NAME_PREFIX", "OUT_DIR"],
    "normalized_core_sha256": hashlib.sha256(normalized_core).hexdigest(),
    "normalized_runner_sha256": hashlib.sha256(nr3.encode()).hexdigest(),
    "training_seed": 42,
    "content_cross_attn_layers": "all = 36 transformer layers",
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
evaluation = {
    "schema": "batch44_v1_r3_r5_eval_contract_v1", "status": "registered_not_run",
    "quick20_steps": list(range(2000, 30001, 2000)),
    "first_mandatory_full320_step": 10000,
    "full320_steps_if_healthy": [10000, 20000, 30000],
    "protocol": "same-step r3/r5, no_text160+text160, Batch-43 D2/D3 two-shard decoding, ASR + WavLM-SV + SpeechBrain ECAPA",
    "stop_rules": {"loss": "NaN/Inf/divergence", "no_text_cer": ">0.20", "no_text_wavlm_margin": "<0.02"},
    "primary_comparisons": ["v1 r3 vs v1 r5", "v1 r3 vs Batch-43 v2 r3", "v1 r5 vs Batch-43 v2 r5"],
}
eval_output.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[batch44-pair-audit] PASS output={output}")
PY
}

qz_payload_dry_run() {
  arm_values "$1"
  runner="$ARM_RECORD_ROOT/run_train_entrypoint.sh"
  command="sh $runner"
  output_path="$ARM_RECORD_ROOT/qz_payload_dry_run.txt"
  set +e
  output=$(
    env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
      HOME="$QZCLI_HOME" QZCLI_GPU_TYPE_OVERRIDE="$QZCLI_GPU_TYPE_OVERRIDE" \
      "$QZCLI" create-job --name "$ARM_JOB_NAME" --command "$command" \
        --workspace "$WORKSPACE" --project "$PROJECT" --compute-group "$COMPUTE_GROUP" \
        --spec "$SPEC" --image "$IMAGE" --image-type "$IMAGE_TYPE" --instances "$INSTANCES" \
        --shm "$SHM_GI" --priority "$PRIORITY" --framework "$FRAMEWORK" --dry-run 2>&1
  )
  status=$?
  set -e
  printf '%s\n' "$output" | tee "$output_path"
  if [ "$status" -ne 0 ]; then
    echo "ERROR: qz payload dry-run failed for $ARM_KEY" >&2
    return "$status"
  fi
  python - "$output_path" "$ARM_JOB_NAME" "$command" "$EXPECTED_WORKSPACE_ID" "$EXPECTED_PROJECT_ID" \
    "$ALLOWED_COMPUTE_GROUP" "$ALLOWED_SPEC" "$ALLOWED_GPU_TYPE" "$ARM_RECORD_ROOT/qz_payload.json" <<'PY'
import json
import re
import sys
from pathlib import Path
source = Path(sys.argv[1])
job, command, workspace, project, group, spec, gpu, output = *sys.argv[2:9], Path(sys.argv[9])
text = source.read_text(encoding="utf-8")
start = text.find("{")
if start < 0:
    raise SystemExit("QZ dry-run lacks JSON payload")
payload, _ = json.JSONDecoder().raw_decode(text[start:])
errors = []
for key, wanted in {"name": job, "command": command, "workspace_id": workspace, "project_id": project, "logic_compute_group_id": group, "framework": "pytorch"}.items():
    if payload.get(key) != wanted:
        errors.append(f"{key}={payload.get(key)!r}, expected {wanted!r}")
configs = payload.get("framework_config") or []
if len(configs) != 1:
    errors.append(f"framework_config count={len(configs)}")
else:
    cfg, resource = configs[0], configs[0].get("resource_spec_price") or {}
    if cfg.get("instance_count") != 1 or cfg.get("gpu_count") != 8:
        errors.append(f"shape={cfg.get('instance_count')}x{cfg.get('gpu_count')}")
    if resource.get("gpu_type") != gpu or resource.get("gpu_count") != 8:
        errors.append(f"gpu resource={resource}")
    if resource.get("logic_compute_group_id") != group or resource.get("quota_id") != spec:
        errors.append(f"compute/spec resource={resource}")
if re.search(r"job-[0-9a-f-]{36}", text, re.I):
    errors.append("dry-run unexpectedly returned job ID")
if errors:
    raise SystemExit("Batch-44 qz payload audit failed:\n- " + "\n- ".join(errors))
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[batch44-qz-dry-run] PASS job={job} output={output}")
PY
}

refuse_duplicate_live() {
  if [ -s "$PAIR_RECORD_ROOT/submitted_pair.tsv" ] || [ -s "$R3_RECORD_ROOT/submitted_jobs.tsv" ] || [ -s "$R5_RECORD_ROOT/submitted_jobs.tsv" ]; then
    echo "ERROR: Batch-44 v1 pair already has a submission ledger" >&2
    exit 1
  fi
  for out_dir in "$R3_OUT_DIR" "$R5_OUT_DIR"; do
    if [ -d "$out_dir" ] && find "$out_dir" -mindepth 1 -print -quit | grep -q .; then
      echo "ERROR: non-empty output would mix runs: $out_dir" >&2
      exit 1
    fi
  done
}

verify_live_arm() {
  arm_values "$1"
  ledger="$ARM_RECORD_ROOT/submitted_jobs.tsv"
  python - "$ledger" "$ARM_JOB_NAME" "$COMPUTE_GROUP" "$ARM_OUT_DIR" <<'PY'
import csv
import re
import sys
from pathlib import Path
path, job, group, out_dir = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
if not path.is_file():
    raise SystemExit(f"missing live ledger: {path}")
rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
if len(rows) != 1:
    raise SystemExit(f"expected one ledger row, got {len(rows)}")
row = rows[0]
if row.get("job_name") != job or row.get("compute_group") != group or row.get("out_dir") != out_dir:
    raise SystemExit(f"live ledger identity mismatch: {row}")
if not re.fullmatch(r"job-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", row.get("job_id", ""), re.I):
    raise SystemExit(f"invalid job ID: {row.get('job_id')}")
print(f"[batch44-live-audit] PASS {job} {row['job_id']}")
PY
}

build_pair_ledger() {
  python - "$R3_RECORD_ROOT/submitted_jobs.tsv" "$R5_RECORD_ROOT/submitted_jobs.tsv" "$PAIR_RECORD_ROOT/submitted_pair.tsv" <<'PY'
import csv
import sys
from pathlib import Path
r3_path, r5_path, output = map(Path, sys.argv[1:])
rows = []
for arm, path in (("r3", r3_path), ("r5", r5_path)):
    item = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
    if len(item) != 1:
        raise SystemExit(f"{arm} ledger row count != 1")
    rows.append((arm, item[0]))
with output.open("w", encoding="utf-8", newline="") as handle:
    fields = ["arm", "job_name", "job_id", "compute_group", "runner", "out_dir"]
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    for arm, row in rows:
        writer.writerow({"arm": arm, **{key: row[key] for key in fields[1:]}})
print(f"[batch44-pair-ledger] wrote {output}")
PY
}

echo "=========================================="
echo "Batch-44 v1 r3/r5 paired 30k"
echo "  r3=$R3_JOB_NAME text_repeat=3"
echo "  r5=$R5_JOB_NAME text_repeat=5"
echo "  shared_no_text=$NO_TEXT_TRAIN_JSONL"
echo "  shared_text=$TEXT_TRAIN_JSONL"
echo "  output_r3=$R3_OUT_DIR"
echo "  output_r5=$R5_OUT_DIR"
echo "  record=$PAIR_RECORD_ROOT"
echo "  max_steps=30000 save/eval=2000 gbs=64"
echo "  compute=$COMPUTE_GROUP (MTTS-3-2-0715 only), each arm=1x8 H200"
echo "  LIVE=$LIVE DRY_RUN=$DRY_RUN VERIFY_FULL_SHA256=$VERIFY_FULL_SHA256"
echo "=========================================="

audit_frozen_baseline
audit_v1_inputs
run_arm 3 1
run_arm 5 1
audit_pair_config
qz_payload_dry_run 3
qz_payload_dry_run 5

if [ "$LIVE" = "0" ]; then
  echo "[batch44] audited pair dry-run complete; no QZ job submitted"
  exit 0
fi

refuse_duplicate_live
LIVE_LOCK="$PAIR_RECORD_ROOT/.live_pair_submit.lock"
if ! mkdir "$LIVE_LOCK" 2>/dev/null; then
  echo "ERROR: Batch-44 live pair lock already exists: $LIVE_LOCK" >&2
  exit 1
fi
trap 'rmdir "$LIVE_LOCK" 2>/dev/null || true' EXIT INT TERM
echo "[batch44] LIVE=1 DRY_RUN=0: submitting r3 then r5"
run_arm 3 0
verify_live_arm 3
run_arm 5 0
verify_live_arm 5
audit_pair_config
build_pair_ledger
rmdir "$LIVE_LOCK"
trap - EXIT INT TERM
echo "[batch44] submitted pair ledger=$PAIR_RECORD_ROOT/submitted_pair.tsv"

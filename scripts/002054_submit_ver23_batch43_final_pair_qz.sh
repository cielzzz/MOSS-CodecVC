#!/bin/sh
# Batch-43: submit the paired ver2.9.5-final data-v2 30k runs.
#
# The two generated jobs are a strict controlled pair.  Their only scientific
# difference is TEXT_REPEAT (r3 versus r5); both use the same frozen Path-X
# code, canonical v2 no_text/text inputs, optimizer, model recipe, step budget,
# and one 8xH200 node from MTTS-3-2-0715.
#
# Safe default (build and compare both runners, submit nothing):
#   sh scripts/002054_submit_ver23_batch43_final_pair_qz.sh
#
# User-authorized Batch-43 live submission:
#   DRY_RUN=0 sh scripts/002054_submit_ver23_batch43_final_pair_qz.sh

set -eu

PROJECT_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
FROZEN_CODE_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch3436_20260710_1092820"
BASE_MODEL_CONFIG="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction/MOSS-TTS/config.json"
BASE_MODEL_CONFIG_SHA256="214fc997d98f51ab57925a5939afc6280e76044198b664221622e70d098ed06e"
QZCLI="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/qzcli_with_deps.sh"
QZCLI_HOME="/inspire/ssd/project/embodied-multimodality/public/xyzhang/.codex/qzcli_home"
STAMP="20260712"
DRY_RUN="${DRY_RUN:-1}"

WORKSPACE="CI-情境智能"
PROJECT="CI-情境智能"
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

PAIR_ID="ver23_batch43_ver2_9_5_final_r3_r5_v2_30k_${STAMP}"
PAIR_RECORD_ROOT="$PROJECT_ROOT/trainset/qz_jobs/$PAIR_ID"
R3_RECORD_ROOT="$PAIR_RECORD_ROOT/r3"
R5_RECORD_ROOT="$PAIR_RECORD_ROOT/r5"
R3_JOB_NAME="ver2_9_5_final_r3_v2_30k"
R5_JOB_NAME="ver2_9_5_final_r5_v2_30k"
R3_BATCH_ID="ver23_batch43_${R3_JOB_NAME}_${STAMP}"
R5_BATCH_ID="ver23_batch43_${R5_JOB_NAME}_${STAMP}"
R3_OUT_DIR="$PROJECT_ROOT/outputs/lora_runs/$R3_JOB_NAME"
R5_OUT_DIR="$PROJECT_ROOT/outputs/lora_runs/$R5_JOB_NAME"

# Canonical data-v2 pair.  no_text is the user-confirmed U1/U1-prime/U2
# reference-channel-decorrelated data.  text.train.jsonl is a canonical symlink
# to the same 32,419-row text split for both arms.
V2_DIR="$PROJECT_ROOT/trainset/ver2_9_prepared_v2_real_no_text_refdecorr_wavlm_sv_20260708"
NO_TEXT_TRAIN_JSONL="$V2_DIR/no_text.v2.train.jsonl"
TEXT_TRAIN_JSONL="$V2_DIR/text.train.jsonl"
NO_TEXT_EXPECTED_ROWS="295632"
NO_TEXT_EXPECTED_BYTES="23967514378"
NO_TEXT_EXPECTED_SHA256="de2e6ca854c8054445739ea831641b0f138893f2ec9ba8dbfd7b0a5760dda5eb"
TEXT_EXPECTED_ROWS="32419"
TEXT_EXPECTED_BYTES="2186751184"
TEXT_EXPECTED_SHA256="b30d07b16b8f86df5b725e9d07f7d8499cb00d6b4662f5b28b65027aa6295993"

# This is not a generic pilot-gate bypass.  Batch-43 beta is authorized only
# for the exact completed Batch-41 r=3 edge-fail artifact and exact metrics
# that the user accepted on 2026-07-12.
BATCH43_BETA_OVERRIDE_ID="batch43_beta_accept_exact_batch41_r3_edge_fail_20260712"
PILOT_GATE_JSON="$PROJECT_ROOT/testset/outputs/ver23_batch41_b2_mixed_3k_probe_20260711/pilot_gate.json"
PILOT_GATE_EXPECTED_SHA256="0256a042c0a7f7486a3aaab350fce9219aa91dbb04c56730c5e11a68fde95880"

if [ "$DRY_RUN" = "0" ]; then
  VERIFY_FULL_SHA256="${VERIFY_FULL_SHA256:-1}"
else
  VERIFY_FULL_SHA256="${VERIFY_FULL_SHA256:-0}"
fi

case "$DRY_RUN" in
  0|1) ;;
  *) echo "ERROR: DRY_RUN must be 0 or 1, got $DRY_RUN" >&2; exit 2 ;;
esac
case "$VERIFY_FULL_SHA256" in
  0|1) ;;
  *) echo "ERROR: VERIFY_FULL_SHA256 must be 0 or 1" >&2; exit 2 ;;
esac
if [ "$DRY_RUN" = "0" ] && [ "$VERIFY_FULL_SHA256" != "1" ]; then
  echo "ERROR: live Batch-43 pair submission requires full SHA256 verification" >&2
  exit 2
fi
if [ "$COMPUTE_GROUP" != "$ALLOWED_COMPUTE_GROUP" ]; then
  echo "ERROR: Batch-43 may only use MTTS-3-2-0715 ($ALLOWED_COMPUTE_GROUP)" >&2
  exit 2
fi
if [ "$SPEC" != "$ALLOWED_SPEC" ] || [ "$INSTANCES" != "$ALLOWED_INSTANCES" ]; then
  echo "ERROR: Batch-43 requires spec=$ALLOWED_SPEC and instances=1 per arm" >&2
  exit 2
fi
if [ "$QZCLI_GPU_TYPE_OVERRIDE" != "$ALLOWED_GPU_TYPE" ]; then
  echo "ERROR: Batch-43 requires $ALLOWED_GPU_TYPE" >&2
  exit 2
fi
if [ "$ACCELERATE_CONFIG" != "$ALLOWED_ACCELERATE_CONFIG" ]; then
  echo "ERROR: Batch-43 requires $ALLOWED_ACCELERATE_CONFIG" >&2
  exit 2
fi
if [ ! -x "$QZCLI" ]; then
  echo "ERROR: fixed qzcli-local wrapper is not executable: $QZCLI" >&2
  exit 1
fi
if [ ! -d "$QZCLI_HOME" ]; then
  echo "ERROR: fixed Codex qzcli HOME is absent: $QZCLI_HOME" >&2
  exit 1
fi

mkdir -p "$PAIR_RECORD_ROOT" "$R3_RECORD_ROOT" "$R5_RECORD_ROOT"

audit_frozen_code() {
  python - \
    "$FROZEN_CODE_ROOT" "$BASE_MODEL_CONFIG" "$BASE_MODEL_CONFIG_SHA256" \
    "$PAIR_RECORD_ROOT/frozen_code_audit.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
base_model_config = Path(sys.argv[2])
base_model_config_sha256 = sys.argv[3]
output = Path(sys.argv[4])
expected = {
    "scripts/002049_submit_ver23_content_side_3k_qz.sh": "e25e4a0d0774181191e07d764b188fa6dae5a6517792be665f9fe101e381d356",
    "scripts/002004_submit_ver2_lora_68w_h200_qz.sh": "3eab2e399b5af8925a36fc8a7b7a39c537b822531c42fa850fd5dd49d3a1ab16",
    "scripts/002002_train_moss_codecvc_lora.py": "3ed70e6c1cb08f81f0c5accd46efa4340df710f92c880601b282fdd38714d7c5",
    "moss_codecvc/models/moss_codecvc_wrapper.py": "5815c8ab5e0aab69d19328fd01782620064327eaf5f39cc4923df8ce3ae9ca42",
    "moss_codecvc/models/content_cross_attn.py": "a8e4cd12d279cfff7c38e3e2d8b21b55d70c403cec654edf7ef77de58acba66a",
    "configs/remote_full.yaml": "0cb3bb7c9d2fdaa050e30bbb06ff55e654b2b411c6d83325daca03c49c9245fd",
    "configs/accelerate_fsdp_h200_8gpu_no_ckpt.yaml": "438c5f5f9dc66f081fa9cc8e9861c4709ef3bc4e3fd6de9532c79f3210e7322e",
}
actual = {}
errors = []
for relative, wanted in expected.items():
    path = root / relative
    if not path.is_file():
        errors.append(f"missing frozen file: {path}")
        continue
    digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    actual[relative] = digest
    if digest != wanted:
        errors.append(f"frozen file drift: {relative}: expected {wanted}, got {digest}")
if errors:
    raise SystemExit("Batch-43 frozen-code audit failed:\n- " + "\n- ".join(errors))
with base_model_config.open("rb") as handle:
    actual_base_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
if actual_base_sha256 != base_model_config_sha256:
    raise SystemExit(
        "Batch-43 base-model config drift: "
        f"expected {base_model_config_sha256}, got {actual_base_sha256}"
    )
base_config = json.loads(base_model_config.read_text(encoding="utf-8"))
language = base_config.get("language_config") or {}
expected_language = {
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
}
for key, wanted in expected_language.items():
    if language.get(key) != wanted:
        errors.append(f"base language_config.{key}={language.get(key)!r}, expected {wanted!r}")
if errors:
    raise SystemExit("Batch-43 base-model architecture audit failed:\n- " + "\n- ".join(errors))
payload = {
    "status": "pass",
    "frozen_code_root": str(root),
    "registered_git_head": "109282047f19886b8f2b38c4cec67092386c172f",
    "files_sha256": actual,
    "base_model_config": str(base_model_config),
    "base_model_config_sha256": actual_base_sha256,
    "base_language_config": expected_language,
    "content_cross_attn_layers_all_resolves_to": 36,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[batch43-code-audit] PASS root={root}")
PY
}

audit_v2_inputs() {
  python - \
    "$V2_DIR/summary.json" "$NO_TEXT_TRAIN_JSONL" "$TEXT_TRAIN_JSONL" \
    "$NO_TEXT_EXPECTED_ROWS" "$NO_TEXT_EXPECTED_BYTES" "$NO_TEXT_EXPECTED_SHA256" \
    "$TEXT_EXPECTED_ROWS" "$TEXT_EXPECTED_BYTES" "$TEXT_EXPECTED_SHA256" \
    "$VERIFY_FULL_SHA256" "$PAIR_RECORD_ROOT/input_identity.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
no_text_path = Path(sys.argv[2])
text_path = Path(sys.argv[3])
no_text_rows = int(sys.argv[4])
no_text_bytes = int(sys.argv[5])
no_text_sha = sys.argv[6]
text_rows = int(sys.argv[7])
text_bytes = int(sys.argv[8])
text_sha = sys.argv[9]
verify_full = sys.argv[10] == "1"
output = Path(sys.argv[11])

summary = json.loads(summary_path.read_text(encoding="utf-8"))
no_text = summary["splits"]["no_text.v2.train.jsonl"]
text = summary["splits"]["text.train.jsonl"]
errors = []
if summary.get("status") != "complete":
    errors.append(f"summary status={summary.get('status')!r}")
if no_text.get("rows") != no_text_rows:
    errors.append(f"no_text rows={no_text.get('rows')!r}")
if text.get("rows") != text_rows:
    errors.append(f"text rows={text.get('rows')!r}")
if no_text.get("missing") != {} or no_text.get("ref_content_leaks") != 0:
    errors.append("no_text has missing fields or ref-content leaks")
if text.get("missing") != {} or text.get("ref_content_leaks") != 0:
    errors.append("text has missing fields or ref-content leaks")
if no_text.get("language_counts") != {"en": 146940, "zh": 148692}:
    errors.append(f"unexpected no_text language counts={no_text.get('language_counts')!r}")
if text.get("language_counts") != {"en": 17649, "zh": 14770}:
    errors.append(f"unexpected text language counts={text.get('language_counts')!r}")
expected_profiles = {
    "near_flat": 192276,
    "mild_eq": 65119,
    "room_eq": 20515,
    "codec_eq": 13285,
    "phone_band": 4437,
}
if no_text.get("ref_channel_profile_counts") != expected_profiles:
    errors.append(f"unexpected no_text channel profiles={no_text.get('ref_channel_profile_counts')!r}")
if Path(no_text.get("path", "")).resolve() != no_text_path.resolve():
    errors.append("summary no_text path mismatch")
if Path(text.get("path", "")).resolve() != text_path.resolve():
    errors.append("summary text path mismatch")
for path, wanted_bytes in ((no_text_path, no_text_bytes), (text_path, text_bytes)):
    if not path.is_file():
        errors.append(f"missing input {path}")
    elif path.stat().st_size != wanted_bytes:
        errors.append(f"size mismatch for {path}: expected {wanted_bytes}, got {path.stat().st_size}")

actual_sha = {}
if verify_full:
    for label, path, wanted in (
        ("no_text", no_text_path, no_text_sha),
        ("text", text_path, text_sha),
    ):
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        actual_sha[label] = digest
        if digest != wanted:
            errors.append(f"{label} SHA256 mismatch: expected {wanted}, got {digest}")

with no_text_path.open(encoding="utf-8") as handle:
    first_no_text = json.loads(handle.readline())
with text_path.open(encoding="utf-8") as handle:
    first_text = json.loads(handle.readline())
if first_no_text.get("moss_codecvc_mode") != "no_text":
    errors.append("first v2 no_text row has wrong mode")
if first_text.get("moss_codecvc_mode") != "text":
    errors.append("first text row has wrong mode")
roles = first_no_text.get("v2_real_target") or {}
if roles.get("target_is_real_audio") is not True or roles.get("source_is_seedvc_output") is not True:
    errors.append(f"unexpected U1/U1-prime roles={roles!r}")
if not first_no_text.get("timbre_ref_channel_augmented"):
    errors.append("first no_text row lacks U2 reference-side channel decorrelation")
for key in (
    "reference_audio_codes",
    "audio_codes",
    "content_token_ids",
    "source_wavlm_bnf_features_path",
    "speaker_vec_path",
    "source_speaker_embedding_path",
    "timbre_ref_speaker_embedding_path",
    "target_speaker_embedding_path",
):
    if first_no_text.get(key) in (None, "", []):
        errors.append(f"first no_text row missing {key}")
if errors:
    raise SystemExit("Batch-43 v2 input audit failed:\n- " + "\n- ".join(errors))

payload = {
    "status": "pass",
    "full_sha256_verified": verify_full,
    "no_text": {
        "path": str(no_text_path),
        "resolved_path": str(no_text_path.resolve()),
        "rows": no_text_rows,
        "bytes": no_text_bytes,
        "sha256": actual_sha.get("no_text", no_text_sha),
        "roles": "source=U1-prime Seed-VC, timbre_ref=U2 channel-decorrelated, target=U1 real",
    },
    "text": {
        "path": str(text_path),
        "resolved_path": str(text_path.resolve()),
        "rows": text_rows,
        "bytes": text_bytes,
        "sha256": actual_sha.get("text", text_sha),
    },
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(
    "[batch43-input-audit] PASS "
    f"no_text={no_text_rows} text={text_rows} "
    f"sha256={'verified' if verify_full else 'registered'}"
)
PY
}

audit_batch43_beta_override() {
  python - \
    "$PILOT_GATE_JSON" "$PILOT_GATE_EXPECTED_SHA256" \
    "$BATCH43_BETA_OVERRIDE_ID" "$PAIR_RECORD_ROOT/batch43_beta_override.json" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_sha256 = sys.argv[2]
override_id = sys.argv[3]
output = Path(sys.argv[4])
if not path.is_file():
    raise SystemExit(f"Batch-43 beta override requires exact pilot gate: {path}")
with path.open("rb") as handle:
    actual_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
if actual_sha256 != expected_sha256:
    raise SystemExit(
        "Batch-43 beta override pilot gate drift: "
        f"expected SHA256 {expected_sha256}, got {actual_sha256}"
    )
gate = json.loads(path.read_text(encoding="utf-8"))
exact = {
    "schema_version": "moss_codecvc.batch41_pilot_gate.v1",
    "decision": "fail",
    "pilot_job_id": "job-04c05174-9a20-4074-add4-2655293452ed",
    "checkpoint_step": 3000,
    "text_repeat": 3,
    "run_id": "ver23_batch41_b2_mixed_r3_step-3000_seedtts320_d2d3_seed1234",
}
errors = [
    f"{key}={gate.get(key)!r}, expected {wanted!r}"
    for key, wanted in exact.items()
    if gate.get(key) != wanted
]
exact_metrics = {
    "no_text_cer": 0.11701921379434184,
    "no_text_fail": 0.15625,
    "text_cer": 0.06924000057563053,
    "text_fail": 0.1875,
    "text_en_src_fail": 0.1875,
    "wavlm_sim_ref": 0.4412522044032812,
    "wavlm_sim_src": 0.40955235734581946,
    "wavlm_ref_bound": 0.40625,
    "speechbrain_ecapa_sim_ref": 0.4827294389717281,
}
for key, wanted in exact_metrics.items():
    try:
        actual = float(gate[key])
    except (KeyError, TypeError, ValueError):
        errors.append(f"missing/non-numeric exact metric {key}")
        continue
    if not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=0.0):
        errors.append(f"{key}={actual!r}, expected exact {wanted!r}")
expected_checks = {
    "no_text_cer_lt_0p12": True,
    "text_en_src_fail_lt_0p15": False,
    "text_cer_lt_0p06": False,
    "wavlm_sim_ref_ge_0p42": True,
}
if gate.get("checks") != expected_checks:
    errors.append(f"checks={gate.get('checks')!r}, expected {expected_checks!r}")
scope = gate.get("metric_scope") or {}
if scope.get("text_en_src_fail") != "text en_src cells, n=80, official content_keep definition":
    errors.append(f"unexpected text_en_src metric scope={scope.get('text_en_src_fail')!r}")
if errors:
    raise SystemExit("Batch-43 beta override rejected:\n- " + "\n- ".join(errors))
payload = {
    "status": "pass",
    "override_id": override_id,
    "scope": "Batch-43 r3/r5 v2 30k pair only; not a reusable gate bypass",
    "pilot_gate_path": str(path),
    "pilot_gate_sha256": actual_sha256,
    "accepted_gate_decision": "fail",
    "accepted_pilot_job_id": exact["pilot_job_id"],
    "accepted_checkpoint_step": 3000,
    "accepted_text_repeat": 3,
    "accepted_metrics": exact_metrics,
    "user_decision": "direction beta: run r3 v2 30k and paired r5 v2 30k despite edge fail",
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(
    "[batch43-beta-override] PASS exact Batch-41 gate decision=fail "
    "step=3000 text_repeat=3 metrics locked"
)
PY
}

arm_values() {
  case "$1" in
    3)
      ARM_JOB_NAME="$R3_JOB_NAME"
      ARM_BATCH_ID="$R3_BATCH_ID"
      ARM_RECORD_ROOT="$R3_RECORD_ROOT"
      ARM_OUT_DIR="$R3_OUT_DIR"
      ;;
    5)
      ARM_JOB_NAME="$R5_JOB_NAME"
      ARM_BATCH_ID="$R5_BATCH_ID"
      ARM_RECORD_ROOT="$R5_RECORD_ROOT"
      ARM_OUT_DIR="$R5_OUT_DIR"
      ;;
    *) echo "ERROR: unsupported Batch-43 TEXT_REPEAT=$1" >&2; exit 2 ;;
  esac
  ARM_TEXT_REPEAT="$1"
  ARM_TRAIN_JSONL_SPEC="$NO_TEXT_TRAIN_JSONL::repeat=1,$TEXT_TRAIN_JSONL::repeat=$ARM_TEXT_REPEAT"
}

run_arm() {
  arm_values "$1"
  requested_dry_run="$2"

  # env -i is deliberate: inherited experiment knobs cannot silently change
  # either arm.  The frozen, hash-locked wrappers supply all remaining defaults.
  env -i \
    PATH="$PATH" \
    HOME="$QZCLI_HOME" \
    QZCLI="$QZCLI" \
    ROOT="$FROZEN_CODE_ROOT" \
    DRY_RUN="$requested_dry_run" \
    WORKSPACE="$WORKSPACE" \
    PROJECT="$PROJECT" \
    COMPUTE_GROUP="$COMPUTE_GROUP" \
    SPEC="$SPEC" \
    INSTANCES="$INSTANCES" \
    QZCLI_GPU_TYPE_OVERRIDE="$QZCLI_GPU_TYPE_OVERRIDE" \
    ACCELERATE_CONFIG="$ACCELERATE_CONFIG" \
    TRAINSET_DIR="$V2_DIR" \
    TRAIN_JSONL="$NO_TEXT_TRAIN_JSONL" \
    NO_TEXT_TRAIN_JSONL="$NO_TEXT_TRAIN_JSONL" \
    TEXT_TRAIN_JSONL="$TEXT_TRAIN_JSONL" \
    TEXT_REPEAT="$ARM_TEXT_REPEAT" \
    TRAIN_JSONL_SPEC="$ARM_TRAIN_JSONL_SPEC" \
    BATCH_ID="$ARM_BATCH_ID" \
    JOB_NAME="$ARM_JOB_NAME" \
    JOB_NAME_PREFIX="$ARM_JOB_NAME" \
    QZ_RECORD_ROOT="$ARM_RECORD_ROOT" \
    OUT_DIR="$ARM_OUT_DIR" \
    NUM_EPOCHS="6" \
    MAX_TRAIN_STEPS="30000" \
    SAVE_STEPS="2000" \
    EVAL_STEPS="2000" \
    EVAL_MAX_BATCHES="0" \
    EVAL_NUM_WORKERS="0" \
    LEARNING_RATE="1e-5" \
    LR_SCHEDULER_TYPE="constant_with_warmup" \
    WARMUP_RATIO="0.03" \
    WEIGHT_DECAY="0.01" \
    PER_DEVICE_BATCH_SIZE="1" \
    GRADIENT_ACCUMULATION_STEPS="8" \
    GPU_COUNT="8" \
    MIXED_PRECISION="bf16" \
    GRADIENT_CHECKPOINTING="0" \
    LORA_R="16" \
    LORA_ALPHA="32" \
    LORA_DROPOUT="0.05" \
    LOGGING_STEPS="20" \
    NUM_WORKERS="4" \
    MAX_GRAD_NORM="1.0" \
    POST_TRAIN_QUICK_EVAL="0" \
    RESUME_ADAPTER_PATH="" \
    TRAIN_SOURCE_SEMANTIC_ONLY="0" \
    FREEZE_LORA="0" \
    FREEZE_ROLE_ROUTING="0" \
    FREEZE_TIMBRE_ADAPTER="0" \
    EVAL_JSONL="" \
    EVAL_JSONL_SPEC="" \
    EVAL_SEEN_JSONL="" \
    EVAL_SEEN_JSONL_SPEC="" \
    EVAL_UNSEEN_JSONL="" \
    EVAL_UNSEEN_JSONL_SPEC="" \
    CONTENT_CROSS_ATTN_LAYERS="all" \
    CONTENT_CROSS_ATTN_FEATURE_DIM="768" \
    CONTENT_CROSS_ATTN_GATE_INIT="-0.5" \
    CONTENT_CROSS_ATTN_OUTPUT_SCALE="0.3" \
    CONTENT_CROSS_ATTN_DROPOUT="0.0" \
    CONTENT_ENCODER_LAYERS="2" \
    CONTENT_ENCODER_CONV_KERNEL_SIZE="7" \
    GUIDED_ATTN_LOSS_WEIGHT="0.05" \
    GUIDED_ATTN_WARMUP_STEPS="1000" \
    GUIDED_ATTN_BAND_FRAMES="3" \
    PHONEME_CLASSIFIER_LOSS_WEIGHT="0.02" \
    CONTENT_CTC_WEIGHT="0.0" \
    TIMBRE_ADAPTER_GATE_LR_MULTIPLIER="1.0" \
    sh "$FROZEN_CODE_ROOT/scripts/002049_submit_ver23_content_side_3k_qz.sh"
}

audit_pair_config() {
  python - \
    "$R3_RECORD_ROOT/train_args_dry_run_core.json" \
    "$R5_RECORD_ROOT/train_args_dry_run_core.json" \
    "$R3_RECORD_ROOT/run_train_entrypoint.sh" \
    "$R5_RECORD_ROOT/run_train_entrypoint.sh" \
    "$NO_TEXT_TRAIN_JSONL" "$TEXT_TRAIN_JSONL" \
    "$R3_JOB_NAME" "$R5_JOB_NAME" \
    "$R3_BATCH_ID" "$R5_BATCH_ID" \
    "$R3_OUT_DIR" "$R5_OUT_DIR" \
    "$PAIR_RECORD_ROOT/pair_control_audit.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

r3_core_path, r5_core_path, r3_runner_path, r5_runner_path = map(Path, sys.argv[1:5])
no_text_path, text_path = sys.argv[5:7]
r3_job, r5_job, r3_batch, r5_batch, r3_out, r5_out = sys.argv[7:13]
output = Path(sys.argv[13])
r3 = json.loads(r3_core_path.read_text(encoding="utf-8"))
r5 = json.loads(r5_core_path.read_text(encoding="utf-8"))

r3_spec = f"{no_text_path}::repeat=1,{text_path}::repeat=3"
r5_spec = f"{no_text_path}::repeat=1,{text_path}::repeat=5"
common_required = {
    "NO_TEXT_TRAIN_JSONL": no_text_path,
    "TEXT_TRAIN_JSONL": text_path,
    "MAX_TRAIN_STEPS": "30000",
    "SAVE_STEPS": "2000",
    "EVAL_STEPS": "2000",
    "LEARNING_RATE": "1e-5",
    "LR_SCHEDULER_TYPE": "constant_with_warmup",
    "WARMUP_RATIO": "0.03",
    "PER_DEVICE_BATCH_SIZE": "1",
    "GRADIENT_ACCUMULATION_STEPS": "8",
    "GPU_COUNT": "8",
    "USE_TIMBRE_MEMORY": "0",
    "TIMBRE_MEMORY_TOKENS": "0",
    "TIMBRE_SIDE_ONLY": "0",
    "REF_PROMPT_CODEC_PERMUTATION": "0",
    "ENABLE_SPEAKER_SIDE_PATHWAY": "0",
    "ENABLE_SPEAKER_CROSS_ATTN": "0",
    "ENABLE_SOURCE_SEMANTIC_MEMORY": "0",
    "TARGET_FRONT_CE_WEIGHT": "4.0",
    "TARGET_FRONT_CE_SECONDS": "0.75",
    "PROGRESS_LOSS_WEIGHT": "0.10",
    "STOP_LOSS_WEIGHT": "0.20",
    "ENABLE_CONTENT_CROSS_ATTN": "1",
    "CONTENT_CROSS_ATTN_LAYERS": "all",
    "CONTENT_CROSS_ATTN_FEATURE_DIM": "768",
    "CONTENT_CROSS_ATTN_GATE_INIT": "-0.5",
    "CONTENT_CROSS_ATTN_OUTPUT_SCALE": "0.3",
    "CONTENT_ENCODER_LAYERS": "2",
    "GUIDED_ATTN_LOSS_WEIGHT": "0.05",
    "GUIDED_ATTN_WARMUP_STEPS": "1000",
    "GUIDED_ATTN_BAND_FRAMES": "3",
    "PHONEME_CLASSIFIER_LOSS_WEIGHT": "0.02",
    "CONTENT_CTC_WEIGHT": "0.0",
    "SOURCE_CONTENT_MEMORY_TYPE": "wavlm_bnf_continuous",
    "LAMBDA_ROUTE": "0.0",
    "LAMBDA_PROSODY": "0.0",
    "LAMBDA_CONTENT": "0.0",
    "SEMANTIC_LOSS_WEIGHT": "0.0",
    "sequence_structure": "[text?, C_src frames, C_ref frames]",
}
errors = []
for label, payload in (("r3", r3), ("r5", r5)):
    for key, wanted in common_required.items():
        if payload.get(key) != wanted:
            errors.append(f"{label}: {key}={payload.get(key)!r}, expected {wanted!r}")
expected_arm = {
    "r3": (r3, "3", r3_spec, r3_batch, r3_job, r3_out),
    "r5": (r5, "5", r5_spec, r5_batch, r5_job, r5_out),
}
for label, (payload, repeat, spec, batch, job, out_dir) in expected_arm.items():
    wanted = {
        "TEXT_REPEAT": repeat,
        "TRAIN_JSONL_SPEC": spec,
        "BATCH_ID": batch,
        "JOB_NAME_PREFIX": job,
        "OUT_DIR": out_dir,
    }
    for key, value in wanted.items():
        if payload.get(key) != value:
            errors.append(f"{label}: {key}={payload.get(key)!r}, expected {value!r}")

different_keys = sorted(key for key in set(r3) | set(r5) if r3.get(key) != r5.get(key))
allowed_different_keys = sorted(["BATCH_ID", "JOB_NAME_PREFIX", "OUT_DIR", "TEXT_REPEAT", "TRAIN_JSONL_SPEC"])
if different_keys != allowed_different_keys:
    errors.append(f"unexpected core config deltas={different_keys}; allowed={allowed_different_keys}")

def normalized_core(payload):
    payload = dict(payload)
    for key in allowed_different_keys:
        payload.pop(key, None)
    return payload

if normalized_core(r3) != normalized_core(r5):
    errors.append("normalized core configs differ")

def normalize_runner(path, spec, out_dir, job, repeat):
    text = path.read_text(encoding="utf-8")
    replacements = (
        (spec, "<TRAIN_JSONL_SPEC>"),
        (out_dir, "<OUT_DIR>"),
        (job, "<ARM_NAME>"),
        (f"text_repeat={repeat}", "text_repeat=<TEXT_REPEAT>"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text

r3_runner = normalize_runner(r3_runner_path, r3_spec, r3_out, r3_job, "3")
r5_runner = normalize_runner(r5_runner_path, r5_spec, r5_out, r5_job, "5")
if r3_runner != r5_runner:
    errors.append("generated runners differ beyond arm name/output/train repeat")

for label, runner_path, repeat in (("r3", r3_runner_path, "3"), ("r5", r5_runner_path, "5")):
    runner = runner_path.read_text(encoding="utf-8")
    for needle in (
        f"[qz-train] text_repeat={repeat}",
        "[qz-train] num_epochs=6 max_train_steps=30000",
        "[qz-train] global_batch_size=64",
        '--config_file "configs/accelerate_fsdp_h200_8gpu_no_ckpt.yaml"',
        '--max-train-steps "30000"',
        '--save-steps "2000"',
        '--learning-rate "1e-5"',
        '--lr-scheduler-type "constant_with_warmup"',
        '--content-cross-attn-layers all',
        '--content-cross-attn-gate-init -0.5',
        '--content-cross-attn-output-scale 0.3',
        '--guided-attn-loss-weight 0.05',
        '--phoneme-classifier-loss-weight 0.02',
        '--no-use-timbre-memory',
        '--no-enable-speaker-side-pathway',
        '--no-enable-speaker-cross-attn',
        '--no-enable-source-semantic-memory',
        'train_source_semantic_only=0 freeze_lora=0 freeze_role_routing=0 freeze_timbre_adapter=0',
    ):
        if needle not in runner:
            errors.append(f"{label} runner missing {needle!r}")

if errors:
    raise SystemExit("Batch-43 pair control audit failed:\n- " + "\n- ".join(errors))

normalized_runner_sha = hashlib.sha256(r3_runner.encode("utf-8")).hexdigest()
normalized_core_bytes = json.dumps(normalized_core(r3), sort_keys=True, separators=(",", ":")).encode("utf-8")
payload = {
    "status": "pass",
    "decision_provenance": {
        "batch43_override_id": "batch43_beta_accept_exact_batch41_r3_edge_fail_20260712",
        "batch41_repeat3_pilot_gate": "exact decision=fail artifact accepted by user on 2026-07-12",
        "pilot_data": "old_no_text_310420_plus_text_32419_repeat3",
        "batch43_data": "v2_no_text_295632_plus_same_text_32419",
    },
    "only_intended_scientific_delta": {
        "TEXT_REPEAT": {"r3": 3, "r5": 5},
    },
    "arm_specific_non_scientific_fields": ["BATCH_ID", "JOB_NAME_PREFIX", "OUT_DIR"],
    "arm_specific_record_roots": {"r3": str(r3_core_path.parent), "r5": str(r5_core_path.parent)},
    "shared_no_text_path": no_text_path,
    "shared_text_path": text_path,
    "content_cross_attn_layers": "all (36 transformer layers; 32 is attention-head/n_vq count)",
    "normalized_core_sha256": hashlib.sha256(normalized_core_bytes).hexdigest(),
    "normalized_runner_sha256": normalized_runner_sha,
    "r3_effective_rows": 295632 + 32419 * 3,
    "r5_effective_rows": 295632 + 32419 * 5,
    "r3_text_share": (32419 * 3) / (295632 + 32419 * 3),
    "r5_text_share": (32419 * 5) / (295632 + 32419 * 5),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(
    "[batch43-pair-control] PASS only scientific delta: TEXT_REPEAT 3 vs 5; "
    f"normalized_runner_sha256={normalized_runner_sha}"
)
PY
}

refuse_duplicate_live_submission() {
  python - \
    "$R3_RECORD_ROOT/submitted_jobs.tsv" "$R5_RECORD_ROOT/submitted_jobs.tsv" \
    "$R3_OUT_DIR" "$R5_OUT_DIR" <<'PY'
import sys
from pathlib import Path

r3_ledger, r5_ledger, r3_out, r5_out = map(Path, sys.argv[1:])
errors = []
for ledger in (r3_ledger, r5_ledger):
    if ledger.exists() and ledger.stat().st_size > 0:
        errors.append(f"existing live submission ledger: {ledger}")
for out_dir in (r3_out, r5_out):
    if out_dir.exists() and any(out_dir.iterdir()):
        errors.append(f"refusing to reuse non-empty output directory: {out_dir}")
if errors:
    raise SystemExit("Batch-43 duplicate fence rejected live submission:\n- " + "\n- ".join(errors))
print("[batch43-live-duplicate-fence] PASS")
PY
}

acquire_atomic_live_lock() {
  LIVE_LOCK_DIR="$PAIR_RECORD_ROOT/.live_submit.lock"
  if ! mkdir "$LIVE_LOCK_DIR" 2>/dev/null; then
    echo "ERROR: Batch-43 atomic live-submit lock already exists: $LIVE_LOCK_DIR" >&2
    echo "Inspect submitted_pair.tsv and both arm ledgers before any manual recovery." >&2
    exit 4
  fi
  python - "$LIVE_LOCK_DIR/owner.json" "$BATCH43_BETA_OVERRIDE_ID" <<'PY'
import datetime as dt
import json
import os
import socket
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "host": socket.gethostname(),
    "pid": os.getppid(),
    "override_id": sys.argv[2],
    "policy": "persistent lock; never auto-remove after a partial or complete submission",
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[batch43-atomic-live-lock] acquired {path.parent}")
PY
}

verify_arm_live_submission() {
  arm_values "$1"
  python - \
    "$ARM_RECORD_ROOT/submit_output.txt" "$ARM_RECORD_ROOT/submitted_jobs.tsv" \
    "$ARM_RECORD_ROOT/arm_submission.tsv" "$ARM_JOB_NAME" \
    "$COMPUTE_GROUP" "$ARM_OUT_DIR" "$ARM_RECORD_ROOT/run_train_entrypoint.sh" <<'PY'
import csv
import re
import sys
from pathlib import Path

submit_output, upstream_ledger, arm_ledger = map(Path, sys.argv[1:4])
expected_job, expected_compute, expected_out, expected_runner = sys.argv[4:8]
uuid_pattern = re.compile(
    r"job-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
if not submit_output.is_file():
    raise SystemExit(f"missing qzcli submit output: {submit_output}")
job_ids = sorted(set(uuid_pattern.findall(submit_output.read_text(encoding="utf-8", errors="replace"))))
if len(job_ids) != 1:
    raise SystemExit(f"expected exactly one complete QZ job UUID in {submit_output}, got {job_ids}")
job_id = job_ids[0]
if not upstream_ledger.is_file():
    raise SystemExit(f"missing upstream arm ledger: {upstream_ledger}")
with upstream_ledger.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if len(rows) != 1:
    raise SystemExit(f"expected one upstream arm ledger row, got {rows}")
row = rows[0]
expected = {
    "job_name": expected_job,
    "job_id": job_id,
    "compute_group": expected_compute,
    "runner": expected_runner,
    "out_dir": expected_out,
}
errors = [f"{key}={row.get(key)!r}, expected {wanted!r}" for key, wanted in expected.items() if row.get(key) != wanted]
if errors:
    raise SystemExit("arm submission provenance mismatch:\n- " + "\n- ".join(errors))
tmp = arm_ledger.with_suffix(arm_ledger.suffix + ".tmp")
with tmp.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=expected.keys(), delimiter="\t")
    writer.writeheader()
    writer.writerow(expected)
tmp.replace(arm_ledger)
print(f"[batch43-arm-ledger] verified unique job UUID {job_id}; wrote {arm_ledger}")
PY
}

build_pair_submission_ledger() {
  python - \
    "$R3_RECORD_ROOT/arm_submission.tsv" "$R5_RECORD_ROOT/arm_submission.tsv" \
    "$PAIR_RECORD_ROOT/submitted_pair.tsv" <<'PY'
import csv
import sys
from pathlib import Path

inputs = [Path(sys.argv[1]), Path(sys.argv[2])]
output = Path(sys.argv[3])
rows = []
for path in inputs:
    if not path.is_file():
        raise SystemExit(f"missing arm submission ledger: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        parsed = list(csv.DictReader(handle, delimiter="\t"))
    if len(parsed) != 1 or not parsed[0].get("job_id", "").startswith("job-"):
        raise SystemExit(f"invalid arm submission ledger: {path}: {parsed}")
    rows.append(parsed[0])
if rows[0]["job_id"] == rows[1]["job_id"]:
    raise SystemExit("r3 and r5 unexpectedly share one job ID")
with output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
print(f"[batch43-pair-ledger] wrote {output}")
PY
}

echo "=========================================="
echo "Batch-43 ver2.9.5-final controlled 30k pair"
echo "  r3 job=$R3_JOB_NAME out=$R3_OUT_DIR"
echo "  r5 job=$R5_JOB_NAME out=$R5_OUT_DIR"
echo "  no_text=$NO_TEXT_TRAIN_JSONL repeat=1"
echo "  text=$TEXT_TRAIN_JSONL repeat={3,5}"
echo "  frozen_code=$FROZEN_CODE_ROOT"
echo "  qzcli=$QZCLI HOME=$QZCLI_HOME (proxies cleared by env -i)"
echo "  compute_group=$COMPUTE_GROUP (MTTS-3-2-0715 only)"
echo "  spec=$SPEC instances=1 per arm GPU=8xH200"
echo "  max_steps=30000 save/eval_steps=2000 gbs=64 lr=1e-5"
echo "  DRY_RUN=$DRY_RUN"
echo "=========================================="

audit_frozen_code
audit_v2_inputs
audit_batch43_beta_override

# Always build both plans first and prove the controlled comparison before any
# job is allowed to reach qzcli.
run_arm 3 1
run_arm 5 1
audit_pair_config

if [ "$DRY_RUN" = "1" ]; then
  echo "[batch43] paired dry-run passed; no QZ job submitted"
  echo "[batch43] audit=$PAIR_RECORD_ROOT/pair_control_audit.json"
  exit 0
fi

acquire_atomic_live_lock
refuse_duplicate_live_submission
echo "[batch43] full input hashes and paired configs verified; submitting r3 then r5"
run_arm 3 0
verify_arm_live_submission 3
run_arm 5 0
verify_arm_live_submission 5
audit_pair_config
build_pair_submission_ledger
echo "[batch43] submission ledger: $PAIR_RECORD_ROOT/submitted_pair.tsv"

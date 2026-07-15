#!/usr/bin/env bash
# Batch-44 r3/r5 paired full SeedTTS-derived 320 evaluation at one exact step.
#
# The two arms are always evaluated at the same checkpoint step.  One MTTS
# 8xH200 node is split into four independent two-GPU lanes:
#   r3 no_text160 -> GPUs 0,1      r3 text160 -> GPUs 2,3
#   r5 no_text160 -> GPUs 4,5      r5 text160 -> GPUs 6,7
#
# Safe defaults never submit a live job.  The registered paired schedule is
# exactly 10k/20k/30k; no other checkpoint may be evaluated by this wrapper.
#
#   STEP=20000 STATIC_AUDIT_ONLY=1 bash scripts/004112_submit_batch44_v1_paired_full320_qz.sh
#   STEP=20000 DRY_RUN=1 bash scripts/004112_submit_batch44_v1_paired_full320_qz.sh
#
# A live submission is intentionally double-gated:
#   STEP=20000 DRY_RUN=0 CONFIRM_BATCH44_FULL320=1 \
#     bash scripts/004112_submit_batch44_v1_paired_full320_qz.sh

set -euo pipefail

CANONICAL_PROJECT_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
PROJECT_ROOT="${PROJECT_ROOT:-$CANONICAL_PROJECT_ROOT}"
CODE_ROOT="${CODE_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch37_eval_20260711_1092820}"
QZCLI="${QZCLI:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/qzcli_with_deps.sh}"
QZCLI_HOME="${QZCLI_HOME:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/.codex/qzcli_home}"

ALLOWED_WORKSPACE="ws-8207e9e2-e733-4eec-a475-cfa1c36480ba"       # CI-情境智能
ALLOWED_PROJECT="project-c67c548f-f02c-453b-ba5b-8745db6886e7"   # CI-情境智能
ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122" # MTTS-3-2-0715
ALLOWED_SPEC="67b10bc6-78b0-41a3-aaf4-358eeeb99009"              # 8xH200
ALLOWED_GPU_TYPE="NVIDIA_H200_SXM_141G"
WORKSPACE="${WORKSPACE:-$ALLOWED_WORKSPACE}"
PROJECT="${PROJECT:-$ALLOWED_PROJECT}"
COMPUTE_GROUP="${COMPUTE_GROUP:-$ALLOWED_COMPUTE_GROUP}"
SPEC="${SPEC:-$ALLOWED_SPEC}"
QZCLI_GPU_TYPE_OVERRIDE="${QZCLI_GPU_TYPE_OVERRIDE:-$ALLOWED_GPU_TYPE}"
IMAGE="${IMAGE:-docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420}"
IMAGE_TYPE="${IMAGE_TYPE:-SOURCE_PRIVATE}"
FRAMEWORK="${FRAMEWORK:-pytorch}"
INSTANCES="${INSTANCES:-1}"
SHM_GI="${SHM_GI:-1200}"
PRIORITY="${PRIORITY:-10}"

STAMP="20260713"
STEP="${STEP:-20000}"
SEED="${SEED:-1234}"
DRY_RUN="${DRY_RUN:-1}"
CONFIRM_BATCH44_FULL320="${CONFIRM_BATCH44_FULL320:-0}"
ENTRYPOINT="${BATCH44_PAIRED_FULL320_ENTRYPOINT:-0}"
STATIC_AUDIT_ONLY="${STATIC_AUDIT_ONLY:-0}"
TEST_MODE="${BATCH44_FULL320_TEST_MODE:-0}"
MIN_CHECKPOINT_AGE_SEC="${MIN_CHECKPOINT_AGE_SEC:-90}"
BACKEND="${BATCH44_FULL320_BACKEND:-qz}"
LOCAL_PREFLIGHT_ONLY="${BATCH44_LOCAL_PREFLIGHT_ONLY:-0}"
LOCAL_FROZEN_RUNNER="${LOCAL_FROZEN_RUNNER:-}"
LOCAL_ENGINE_SOURCE="${LOCAL_ENGINE_SOURCE:-$PROJECT_ROOT/scripts/004112_submit_batch44_v1_paired_full320_qz.sh}"
LOCAL_INPUTS_MANIFEST="${LOCAL_INPUTS_MANIFEST:-}"

R3_RUN_DIR="${R3_RUN_DIR:-$PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r3_v1_30k}"
R5_RUN_DIR="${R5_RUN_DIR:-$PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r5_v1_30k}"
R3_TRAIN_JOB_ID="job-2b91d332-d500-4279-84f9-0a6a81a376aa"
R5_TRAIN_JOB_ID="job-b8eb2f1f-a3eb-483b-a289-b4cce281525c"
TRAIN_PAIR_LEDGER="$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713/submitted_pair.tsv"
TRAIN_IDENTITY_ROOT="$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713"

CANONICAL_VALIDATION_JSONL="$PROJECT_ROOT/testset/validation/seedtts_vc_ver2_3_validation.jsonl"
VALIDATION_JSONL="${VALIDATION_JSONL:-$CANONICAL_VALIDATION_JSONL}"
VALIDATION_SHA256="725ee9d58a7e6066d2a7b79c858cb6ff4dd7292cc167c45dc6b6ebbeaff2fe14"
PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
ASR_PYTHON="${ASR_PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
SPEAKER_SIM_ROOT="${SPEAKER_SIM_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction}"
SPEECHBRAIN_ECAPA_MODEL_SOURCE="${SPEECHBRAIN_ECAPA_MODEL_SOURCE:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/models/speechbrain/spkrec-ecapa-voxceleb}"

RECORD_ROOT="${RECORD_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_paired_full320_step${STEP}_${STAMP}}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/testset/outputs/ver23_batch44_paired_full320_${STAMP}}"
STEP_ROOT="$EVAL_ROOT/step-$STEP"
RUNS_ROOT="$STEP_ROOT/runs"
AGG_ROOT="$STEP_ROOT/aggregate"
JOB_NAME="${JOB_NAME:-ver23_batch44_r3r5_full320_step${STEP}_${STAMP}}"
FROZEN_RUNNER="$RECORD_ROOT/004112_submit_batch44_v1_paired_full320_qz.frozen.sh"
SUBMISSION_LOCK="$RECORD_ROOT/.live_submit.lock"
COMPLETION_JSON="$RECORD_ROOT/COMPLETED.json"
LOCAL_RUN_LOCK="${LOCAL_RUN_LOCK:-$RECORD_ROOT/.local_run.lock}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

if [ "$BACKEND" = "local" ]; then
  case "$STEP" in
    10000|20000|26000|28000|30000) ;;
    *) die "local STEP must be one of 10000/20000/26000/28000/30000; got $STEP" ;;
  esac
else
  case "$STEP" in
    10000|20000|30000) ;;
    *) die "STEP must be one of the registered Batch-44 full320 checkpoints 10000/20000/30000; got $STEP" ;;
  esac
fi
case "$DRY_RUN:$CONFIRM_BATCH44_FULL320:$ENTRYPOINT:$STATIC_AUDIT_ONLY:$TEST_MODE:$LOCAL_PREFLIGHT_ONLY" in
  [01]:[01]:[01]:[01]:[01]:[01]) ;;
  *) die "boolean flags must be 0 or 1" ;;
esac
case "$BACKEND" in
  qz|local) ;;
  *) die "BATCH44_FULL320_BACKEND must be qz or local; got $BACKEND" ;;
esac
case "$MIN_CHECKPOINT_AGE_SEC" in
  ''|*[!0-9]*) die "MIN_CHECKPOINT_AGE_SEC must be a non-negative integer" ;;
esac
if [ "$PROJECT_ROOT" != "$CANONICAL_PROJECT_ROOT" ] && [ "$TEST_MODE" != "1" ]; then
  die "non-canonical PROJECT_ROOT is allowed only with BATCH44_FULL320_TEST_MODE=1"
fi
if [ "$TEST_MODE" = "1" ] && { [ "$DRY_RUN" = "0" ] || [ "$ENTRYPOINT" = "1" ]; }; then
  die "test mode may not submit or execute a QZ entrypoint"
fi
if [ "$BACKEND" = "local" ] && [ "$ENTRYPOINT" != "1" ]; then
  die "local backend is entrypoint-only; use scripts/004118_run_batch44_v1_paired_full320_local.sh"
fi
if [ "$LOCAL_PREFLIGHT_ONLY" = "1" ] && [ "$BACKEND" != "local" ]; then
  die "BATCH44_LOCAL_PREFLIGHT_ONLY is valid only for the local backend"
fi
if [ "$BACKEND" = "qz" ]; then
  if [ "$WORKSPACE" != "$ALLOWED_WORKSPACE" ] || [ "$PROJECT" != "$ALLOWED_PROJECT" ]; then
    die "Batch-44 full320 is restricted to CI-情境智能"
  fi
  if [ "$COMPUTE_GROUP" != "$ALLOWED_COMPUTE_GROUP" ]; then
    die "Batch-44 full320 may only use MTTS-3-2-0715 ($ALLOWED_COMPUTE_GROUP)"
  fi
  if [ "$SPEC" != "$ALLOWED_SPEC" ] || [ "$QZCLI_GPU_TYPE_OVERRIDE" != "$ALLOWED_GPU_TYPE" ]; then
    die "Batch-44 full320 requires spec=$ALLOWED_SPEC and GPU=$ALLOWED_GPU_TYPE"
  fi
  if [ "$INSTANCES" != "1" ]; then
    die "Batch-44 full320 requires exactly one 8xH200 instance"
  fi
  if [ "$DRY_RUN" = "0" ] && [ "$CONFIRM_BATCH44_FULL320" != "1" ]; then
    die "live submission requires CONFIRM_BATCH44_FULL320=1"
  fi
elif [ "$TEST_MODE" != "1" ]; then
  [ "$RECORD_ROOT" = "$CANONICAL_PROJECT_ROOT/trainset/local_jobs/ver23_batch44_paired_full320_step${STEP}_${STAMP}" ] \
    || die "local backend record root is not canonical: $RECORD_ROOT"
  [ "$EVAL_ROOT" = "$CANONICAL_PROJECT_ROOT/testset/outputs/ver23_batch44_paired_full320_${STAMP}" ] \
    || die "local backend eval root is not canonical: $EVAL_ROOT"
  [ ! -L "$RECORD_ROOT" ] || die "local backend record root may not be a symlink: $RECORD_ROOT"
  [ ! -L "$EVAL_ROOT" ] || die "local backend eval root may not be a symlink: $EVAL_ROOT"
  [ ! -L "$STEP_ROOT" ] || die "local backend step root may not be a symlink: $STEP_ROOT"
fi

arm_run_dir() {
  case "$1" in
    r3) printf '%s\n' "$R3_RUN_DIR" ;;
    r5) printf '%s\n' "$R5_RUN_DIR" ;;
    *) die "unsupported Batch-44 arm: $1" ;;
  esac
}

arm_label() {
  case "$1" in
    # Keep the evaluation run-id namespace aligned with the registered
    # Best3/blind/final consumers.  The training run directory carries
    # `_v1_30k`, but the canonical evaluation run-id intentionally does not.
    r3) printf '%s\n' "ver2_9_5_final_r3" ;;
    r5) printf '%s\n' "ver2_9_5_final_r5" ;;
    *) die "unsupported Batch-44 arm: $1" ;;
  esac
}

arm_checkpoint() {
  printf '%s/step-%s\n' "$(arm_run_dir "$1")" "$STEP"
}

run_id_for() {
  local arm="$1"
  local mode="$2"
  printf '%s_step-%s_%s_seedtts160_d2d3_seed%s\n' "$(arm_label "$arm")" "$STEP" "$mode" "$SEED"
}

output_dir_for() {
  printf '%s/%s\n' "$RUNS_ROOT" "$(run_id_for "$1" "$2")"
}

audit_code_root() {
  "$PYTHON" - "$CODE_ROOT" <<'PY'
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {
    "scripts/004039_run_seedtts_validation_eval.sh": "94ee38a950691ddd22e9487c82821247447dc7ecf20813e94852c56687c727b4",
    "scripts/004042_summarize_seedtts_validation_eval.py": "815975fe4b9e8cf51ab5eef919418528dc088be53a3179b5e5a50e1e14b0dd20",
    "scripts/004044_run_seedtts_validation_infer_persistent.py": "22045797d68d54bc2b72c64773c43464e4164b19b3a29d97537149e15594fa1d",
    "scripts/004048_summarize_seedtts_ablation_metrics.py": "e1856c1a503a2101480323acaa9b0d231a6b28971377d47664f3fae02b1d7ca4",
    "scripts/004056_summarize_seedtts_ref_content_similarity.py": "42df1d42934bf3283975eda2bef773a53cafe2a75e4518432664f9373321c4a4",
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
        errors.append(f"{relative} SHA256={got}, expected {wanted}")

if not errors:
    eval_text = (root / "scripts/004039_run_seedtts_validation_eval.sh").read_text(encoding="utf-8")
    infer_text = (root / "scripts/004044_run_seedtts_validation_infer_persistent.py").read_text(encoding="utf-8")
    wrapper_text = (root / "moss_codecvc/models/moss_codecvc_wrapper.py").read_text(encoding="utf-8")
    for needle in (
        'export HF_MODULES_CACHE="$hf_modules_cache_root/shard${shard}"',
        'mkdir -p "$HF_MODULES_CACHE"',
    ):
        if needle not in eval_text:
            errors.append(f"004039 missing shard-cache isolation: {needle}")
    if "content_cross_attn_needs_features = content_cross_attn_encoder is not None and no_text" not in infer_text:
        errors.append("004044 does not bypass BNF feature preparation for text inference")
    for needle in ("_content_cross_attn_active_sample_mask", "content_cross_attn_text_bypass_samples"):
        if needle not in wrapper_text:
            errors.append(f"wrapper missing text-row BNF bypass: {needle}")
if errors:
    raise SystemExit("Batch-44 frozen eval snapshot audit failed:\n- " + "\n- ".join(errors))
print(f"[batch44-full320-code] PASS root={root} files={len(expected)}")
PY
  bash -n "$CODE_ROOT/scripts/004039_run_seedtts_validation_eval.sh"
}

audit_validation_manifest() {
  "$PYTHON" - "$VALIDATION_JSONL" "$CANONICAL_VALIDATION_JSONL" "$VALIDATION_SHA256" "$TEST_MODE" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
canonical = Path(sys.argv[2])
wanted_sha = sys.argv[3]
test_mode = sys.argv[4] == "1"
if not path.is_file():
    raise SystemExit(f"missing validation manifest: {path}")
if not test_mode and path.resolve() != canonical.resolve():
    raise SystemExit(f"full320 requires canonical validation manifest: {canonical}")
with path.open("rb") as handle:
    got_sha = hashlib.file_digest(handle, "sha256").hexdigest()
if not test_mode and got_sha != wanted_sha:
    raise SystemExit(f"validation SHA256={got_sha}, expected {wanted_sha}")
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
ids = [str(row.get("case_id") or "") for row in rows]
modes = Counter(str(row.get("mode") or "") for row in rows)
mode_cells = Counter((str(row.get("mode") or ""), str(row.get("cell") or "")) for row in rows)
if len(rows) != 320 or len(set(ids)) != 320 or any(not value for value in ids):
    raise SystemExit(f"validation rows/unique/blanks={len(rows)}/{len(set(ids))}/{sum(not x for x in ids)}")
if modes != Counter({"no_text": 160, "text": 160}):
    raise SystemExit(f"validation mode counts={dict(modes)}")
if len(mode_cells) != 16 or any(value != 20 for value in mode_cells.values()):
    raise SystemExit(f"validation must contain 20 cases per mode/cell: {dict(mode_cells)}")
text_en = [row for row in rows if row.get("mode") == "text" and str(row.get("cell") or "").startswith("en_src_")]
if len(text_en) != 80:
    raise SystemExit(f"text en_src scope must contain 80 cases, got {len(text_en)}")
print(f"[batch44-full320-data] PASS sha256={got_sha} no_text=160 text=160 text_en_src=80")
PY
}

audit_training_pair() {
  if [ "$TEST_MODE" = "1" ]; then
    return 0
  fi
  "$PYTHON" - "$TRAIN_PAIR_LEDGER" "$R3_RUN_DIR" "$R5_RUN_DIR" \
    "$R3_TRAIN_JOB_ID" "$R5_TRAIN_JOB_ID" "$ALLOWED_COMPUTE_GROUP" \
    "$TRAIN_IDENTITY_ROOT" <<'PY'
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ledger = Path(sys.argv[1])
r3_out, r5_out, r3_job, r5_job, compute = sys.argv[2:7]
identity_root = Path(sys.argv[7])
if not ledger.is_file():
    raise SystemExit(f"missing training pair ledger: {ledger}")
with ledger.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
expected = {
    "ver2_9_5_final_r3_v1_30k": ("r3", r3_job, r3_out),
    "ver2_9_5_final_r5_v1_30k": ("r5", r5_job, r5_out),
}
errors = []
if len(rows) != 2:
    errors.append(f"expected two rows, got {len(rows)}")
seen = set()
for row in rows:
    name = row.get("job_name", "")
    seen.add(name)
    if name not in expected:
        errors.append(f"unexpected training job {name!r}")
        continue
    arm, job, out_dir = expected[name]
    if row.get("arm") != arm:
        errors.append(f"{name} arm={row.get('arm')!r}, expected {arm!r}")
    if row.get("job_id") != job:
        errors.append(f"{name} job_id={row.get('job_id')!r}, expected {job}")
    if row.get("out_dir") != out_dir:
        errors.append(f"{name} out_dir={row.get('out_dir')!r}, expected {out_dir}")
    if row.get("compute_group") != compute:
        errors.append(f"{name} compute_group={row.get('compute_group')!r}, expected {compute}")
    wanted_runner = str(identity_root / arm / "run_train_entrypoint.sh")
    if row.get("runner") != wanted_runner:
        errors.append(f"{name} runner={row.get('runner')!r}, expected {wanted_runner!r}")
if seen != set(expected):
    errors.append(f"training pair names={sorted(seen)}, expected {sorted(expected)}")

immutable = {
    identity_root / "evaluation_contract.json": "cd41b1f1cb97fb7bd50b5939a6825ca55143835f4c7ad164af42260551b946c1",
    identity_root / "input_identity.full_sha256.json": "accefffbac9aa78b499c5938c9842040dbe44eb6a6188b8a4c444db1b57566b4",
    identity_root / "r3/train_args_dry_run_core.json": "3637b90fe14bbda7c4915362d5bd726b072833f07853df4fc555fde0d2c32d7b",
    identity_root / "r5/train_args_dry_run_core.json": "161f05b766454ce7d3e38af1772a1126ec612eaeb0682678bd6a46ca1d62daff",
    identity_root / "r3/run_train_entrypoint.sh": "09492f6304287918115b01bc0c582c2394d8ba417636c665a57bd30152386b1a",
    identity_root / "r5/run_train_entrypoint.sh": "193e23ce50c5d46ccecaccfb493871d5ddcb773ef2482ca53a0b3fd7b170c207",
}
for path, wanted in immutable.items():
    if not path.is_file():
        errors.append(f"missing immutable training artifact: {path}")
        continue
    with path.open("rb") as handle:
        got = hashlib.file_digest(handle, "sha256").hexdigest()
    if got != wanted:
        errors.append(f"immutable training artifact SHA256={got}, expected {wanted}: {path}")

contract_path = identity_root / "evaluation_contract.json"
if contract_path.is_file():
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != "batch44_v1_r3_r5_eval_contract_v1":
        errors.append(f"evaluation contract schema={contract.get('schema')!r}")
    if contract.get("full320_steps_if_healthy") != [10000, 20000, 30000]:
        errors.append(f"evaluation contract steps={contract.get('full320_steps_if_healthy')!r}")
    if contract.get("stop_rules") != {
        "loss": "NaN/Inf/divergence",
        "no_text_cer": ">0.20",
        "no_text_wavlm_margin": "<0.02",
    }:
        errors.append(f"evaluation contract stop_rules={contract.get('stop_rules')!r}")
if errors:
    raise SystemExit("Batch-44 training provenance audit failed:\n- " + "\n- ".join(errors))
with ledger.open("rb") as handle:
    digest = hashlib.file_digest(handle, "sha256").hexdigest()
print(f"[batch44-full320-training] PASS ledger_sha256={digest} r3={r3_job} r5={r5_job}")
PY
}

validate_checkpoint() {
  local arm="$1"
  local expected_repeat checkpoint run_dir provenance_out
  run_dir="$(arm_run_dir "$arm")"
  checkpoint="$(arm_checkpoint "$arm")"
  case "$arm" in r3) expected_repeat=3 ;; r5) expected_repeat=5 ;; esac
  provenance_out="$RECORD_ROOT/checkpoint_${arm}_step${STEP}.json"
  "$PYTHON" - "$arm" "$expected_repeat" "$run_dir" "$checkpoint" "$PROJECT_ROOT" \
    "$STEP" "$MIN_CHECKPOINT_AGE_SEC" "$provenance_out" "$TEST_MODE" "$BACKEND" <<'PY'
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

arm = sys.argv[1]
repeat = int(sys.argv[2])
run_dir = Path(sys.argv[3])
checkpoint = Path(sys.argv[4])
project_root = Path(sys.argv[5])
step = int(sys.argv[6])
min_age = int(sys.argv[7])
provenance_out = Path(sys.argv[8])
test_mode = sys.argv[9] == "1"
backend = sys.argv[10]
required = {
    "adapter_model.safetensors": 1_000_000,
    "adapter_config.json": 1,
    "README.md": 1,
    "timbre_memory_adapter.pt": 1_000_000,
    "timbre_memory_config.json": 1,
}
errors = []
inventory = {}
newest = 0.0
if checkpoint.name != f"step-{step}":
    errors.append(f"checkpoint name={checkpoint.name!r}")
for name, minimum in required.items():
    path = checkpoint / name
    if not path.is_file() or path.stat().st_size < minimum:
        errors.append(f"missing/small checkpoint file: {path}")
        continue
    stat = path.stat()
    newest = max(newest, stat.st_mtime)
    item = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    # Local evaluation is the final consumer of mutable training outputs, so
    # bind every checkpoint byte (including both large weight files).  The QZ
    # compatibility path retains its historical lightweight JSON-only audit.
    if name.endswith(".json") or backend == "local":
        with path.open("rb") as handle:
            item["sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    inventory[name] = item
age = time.time() - newest if newest else -1.0
if not test_mode and age < min_age:
    errors.append(f"checkpoint newest-file age={age:.1f}s, required >= {min_age}s")

for name in ("adapter_config.json", "timbre_memory_config.json"):
    path = checkpoint / name
    if path.is_file():
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"invalid JSON {path}: {exc}")

cfg_path = checkpoint / "timbre_memory_config.json"
if cfg_path.is_file():
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    expected = {
        "content_cross_attn_enabled": True,
        "content_cross_attn_layers": "all",
        "content_cross_attn_feature_dim": 768,
        "content_cross_attn_gate_init": -0.5,
        "content_cross_attn_output_scale": 0.3,
        "content_encoder_layers": 2,
        "guided_attn_loss_weight": 0.05,
        "phoneme_classifier_loss_weight": 0.02,
        "content_ctc_weight": 0.0,
        "progress_loss_weight": 0.1,
        "stop_loss_weight": 0.2,
        "target_front_ce_weight": 4.0,
        "target_front_ce_seconds": 0.75,
        "use_role_routing": True,
        "num_memory_tokens": 0,
        "timbre_side_only": False,
        "source_semantic_memory_enabled": False,
        "speaker_side_pathway_enabled": False,
        "speaker_cross_attn_enabled": False,
    }
    for key, wanted in expected.items():
        got = cfg.get(key)
        if isinstance(wanted, float):
            try:
                ok = math.isclose(float(got), wanted, rel_tol=0.0, abs_tol=1e-9)
            except (TypeError, ValueError):
                ok = False
        else:
            ok = got == wanted
        if not ok:
            errors.append(f"checkpoint config {key}={got!r}, expected {wanted!r}")

identity_path = (
    project_root
    / "trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713"
    / arm
    / "train_args_dry_run_core.json"
)
identity_sha = ""
if not identity_path.is_file():
    errors.append(f"missing training identity: {identity_path}")
else:
    with identity_path.open("rb") as handle:
        identity_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    if not test_mode:
        wanted_sha = {
            "r3": "3637b90fe14bbda7c4915362d5bd726b072833f07853df4fc555fde0d2c32d7b",
            "r5": "161f05b766454ce7d3e38af1772a1126ec612eaeb0682678bd6a46ca1d62daff",
        }[arm]
        if identity_sha != wanted_sha:
            errors.append(f"training identity SHA256={identity_sha}, expected {wanted_sha}")
    args = json.loads(identity_path.read_text(encoding="utf-8"))
    no_text = project_root / "trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/no_text.train.jsonl"
    text = project_root / "trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/text.train.jsonl"
    expected_args = {
        "OUT_DIR": str(run_dir),
        "TRAIN_JSONL_SPEC": f"{no_text}::repeat=1,{text}::repeat={repeat}",
        "TEXT_REPEAT": str(repeat),
        "MAX_TRAIN_STEPS": "30000",
        "SAVE_STEPS": "2000",
        "EVAL_STEPS": "2000",
        "LEARNING_RATE": "1e-5",
        "LR_SCHEDULER_TYPE": "constant_with_warmup",
        "WARMUP_RATIO": "0.03",
        "ENABLE_CONTENT_CROSS_ATTN": "1",
        "CONTENT_CROSS_ATTN_LAYERS": "all",
        "GUIDED_ATTN_LOSS_WEIGHT": "0.05",
        "PHONEME_CLASSIFIER_LOSS_WEIGHT": "0.02",
        "CONTENT_CTC_WEIGHT": "0.0",
    }
    for key, wanted in expected_args.items():
        if args.get(key) != wanted:
            errors.append(f"training identity {key}={args.get(key)!r}, expected {wanted!r}")
if errors:
    raise SystemExit(f"Batch-44 {arm} step-{step} checkpoint audit failed:\n- " + "\n- ".join(errors))

payload = {
    "arm": arm,
    "text_repeat": repeat,
    "training_job_id": {
        "r3": "job-2b91d332-d500-4279-84f9-0a6a81a376aa",
        "r5": "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c",
    }[arm],
    "step": step,
    "run_dir": str(run_dir),
    "checkpoint": str(checkpoint),
    "newest_file_age_sec": age,
    "training_identity": str(identity_path),
    "training_identity_sha256": identity_sha,
    "checkpoint_inventory": inventory,
}
provenance_out.parent.mkdir(parents=True, exist_ok=True)
provenance_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[batch44-full320-checkpoint] PASS arm={arm} step={step} repeat={repeat} age={age:.0f}s")
PY
}

audit_static_inputs() {
  [ -x "$PYTHON" ] || die "Python interpreter is not executable: $PYTHON"
  [ -x "$ASR_PYTHON" ] || die "ASR Python interpreter is not executable: $ASR_PYTHON"
  [ -d "$SPEAKER_SIM_ROOT" ] || die "missing WavLM speaker scorer root: $SPEAKER_SIM_ROOT"
  [ -d "$SPEECHBRAIN_ECAPA_MODEL_SOURCE" ] || die "missing SpeechBrain ECAPA model: $SPEECHBRAIN_ECAPA_MODEL_SOURCE"
  audit_code_root
  audit_validation_manifest
  audit_training_pair
}

write_resolved_runs() {
  mkdir -p "$RECORD_ROOT" "$RUNS_ROOT" "$AGG_ROOT"
  {
    printf 'arm\ttext_repeat\ttrain_job_id\tstep\tmode\tgpu_pair\tcheckpoint\trun_id\toutput_dir\n'
    local arm mode repeat job gpu
    for arm in r3 r5; do
      case "$arm" in
        r3) repeat=3; job="$R3_TRAIN_JOB_ID" ;;
        r5) repeat=5; job="$R5_TRAIN_JOB_ID" ;;
      esac
      for mode in no_text text; do
        if [ "$BACKEND" = "local" ]; then
          gpu="0,1"
        else
          case "$arm:$mode" in
            r3:no_text) gpu="0,1" ;;
            r3:text) gpu="2,3" ;;
            r5:no_text) gpu="4,5" ;;
            r5:text) gpu="6,7" ;;
          esac
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$arm" "$repeat" "$job" "$STEP" "$mode" "$gpu" "$(arm_checkpoint "$arm")" \
          "$(run_id_for "$arm" "$mode")" "$(output_dir_for "$arm" "$mode")"
      done
    done
  } > "$RECORD_ROOT/resolved_runs.tsv"
}

ensure_evaluation_is_new() {
  if [ -e "$COMPLETION_JSON" ] || [ -L "$COMPLETION_JSON" ] \
    || [ -e "$RECORD_ROOT/complete.marker" ] || [ -L "$RECORD_ROOT/complete.marker" ] \
    || [ -e "$AGG_ROOT/paired_metrics.tsv" ] || [ -L "$AGG_ROOT/paired_metrics.tsv" ]; then
    die "step-$STEP full320 has existing/partial completion evidence; refusing to overwrite"
  fi
  if [ "$ENTRYPOINT" != "1" ] && [ "$DRY_RUN" = "0" ]; then
    if [ -s "$RECORD_ROOT/submitted_jobs.tsv" ]; then
      die "step-$STEP already has a live submission ledger"
    fi
    if [ -d "$SUBMISSION_LOCK" ]; then
      die "persistent live-submit lock exists; inspect QZ state before recovery: $SUBMISSION_LOCK"
    fi
  fi
  if [ "$BACKEND" = "local" ] && { [ -e "$RECORD_ROOT/submitted_jobs.tsv" ] || [ -L "$RECORD_ROOT/submitted_jobs.tsv" ]; }; then
    die "local full320 record must not contain a QZ submission ledger: $RECORD_ROOT/submitted_jobs.tsv"
  fi
  local arm mode run_id output_dir
  for arm in r3 r5; do
    for mode in no_text text; do
      run_id="$(run_id_for "$arm" "$mode")"
      output_dir="$(output_dir_for "$arm" "$mode")"
      if [ -e "$output_dir" ] || [ -L "$output_dir" ]; then
        die "partial/existing evaluation requires manual inspection: $output_dir"
      fi
    done
  done
}

audit_runtime_gpus() {
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable in full320 entrypoint"
  local gpu_count
  gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l | tr -d ' ')
  if [ "$BACKEND" = "local" ]; then
    [ "$gpu_count" = "2" ] || die "local full320 requires exactly two visible GPUs; got $gpu_count"
    if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -Evq '^NVIDIA GeForce RTX 4090$'; then
      die "local full320 requires two NVIDIA GeForce RTX 4090 GPUs"
    fi
    local hostname_value
    hostname_value=$(hostname)
    case "$hostname_value" in
      xyzhang-dev--*) ;;
      *) die "local full320 must run on the registered development host; got $hostname_value" ;;
    esac
    nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version \
      --format=csv,noheader,nounits > "$RECORD_ROOT/runtime_gpu_inventory.csv"
    "$PYTHON" - "$RECORD_ROOT/runtime_gpu_inventory.csv" \
      "$RECORD_ROOT/runtime_gpu_inventory.json" "$hostname_value" <<'PY'
import csv
import datetime as dt
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
output = Path(sys.argv[2])
host = sys.argv[3]
rows = []
with source.open(encoding="utf-8", newline="") as handle:
    for raw in csv.reader(handle):
        if len(raw) != 5:
            raise SystemExit(f"invalid nvidia-smi inventory row: {raw!r}")
        index, uuid, name, memory, driver = (value.strip() for value in raw)
        rows.append({
            "index": int(index),
            "uuid": uuid,
            "name": name,
            "memory_total_mib": int(memory),
            "driver_version": driver,
        })
if [row["index"] for row in rows] != [0, 1]:
    raise SystemExit(f"local GPU indices must be [0, 1], got {rows!r}")
if any(row["name"] != "NVIDIA GeForce RTX 4090" for row in rows):
    raise SystemExit(f"local GPU model drift: {rows!r}")
if any(row["memory_total_mib"] < 48_000 for row in rows):
    raise SystemExit(f"local GPU memory is below 48,000 MiB: {rows!r}")
if len({row["uuid"] for row in rows}) != 2 or any(not row["uuid"].startswith("GPU-") for row in rows):
    raise SystemExit(f"local GPU UUID inventory invalid: {rows!r}")
payload = {
    "schema": "batch44_local_gpu_inventory_v1",
    "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "hostname": host,
    "gpus": rows,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    echo "[batch44-full320-gpu] PASS backend=local count=2 type=RTX4090 host=$hostname_value"
    return 0
  fi
  [ "$gpu_count" = "8" ] || die "QZ entrypoint must expose exactly 8 GPUs; got $gpu_count"
  if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -Evq 'H200'; then
    die "QZ entrypoint contains a non-H200 GPU"
  fi
  echo "[batch44-full320-gpu] PASS backend=qz count=8 type=H200"
}

run_eval_lane() {
  local arm="$1"
  local mode="$2"
  local gpu_pair="$3"
  local checkpoint run_id output_dir log
  checkpoint="$(arm_checkpoint "$arm")"
  run_id="$(run_id_for "$arm" "$mode")"
  output_dir="$(output_dir_for "$arm" "$mode")"
  log="$RECORD_ROOT/eval_${arm}_${mode}_step${STEP}.log"
  mkdir -p "$output_dir"
  (
    set -euo pipefail
    echo "[batch44-full320] arm=$arm mode=$mode gpu_pair=$gpu_pair checkpoint=$checkpoint run_id=$run_id"
    CUDA_VISIBLE_DEVICES="$gpu_pair" \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=8 \
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
    HF_MODULES_CACHE_ROOT="$output_dir/.hf_modules_cache" \
    INFER_SHARD_START_DELAY_SEC=3 \
    PYTHON="$PYTHON" \
    ASR_PYTHON="$ASR_PYTHON" \
    VALIDATION_JSONL="$VALIDATION_JSONL" \
    MODEL_PATH="$checkpoint" \
    RUN_ID="$run_id" \
    RUN_LABEL="Batch-44 $arm step-$STEP $mode full160" \
    OUTPUT_DIR="$output_dir" \
    MODE="$mode" \
    MAX_CASES=0 \
    PER_MODE=0 \
    PER_CELL=0 \
    DECODING_PROFILE=default \
    PERSISTENT_INFER=1 \
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

    "$PYTHON" "$CODE_ROOT/scripts/004056_summarize_seedtts_ref_content_similarity.py" \
      --asr-jsonl "$output_dir/${run_id}.asr_eval.jsonl" \
      --output-json "$output_dir/${run_id}.ref_content_similarity_summary.json" \
      --output-md "$output_dir/${run_id}.ref_content_similarity_summary.md"
    echo "[batch44-full320] lane complete arm=$arm mode=$mode output=$output_dir"
  ) > >(tee -a "$log") 2>&1
}

run_four_lanes() {
  if [ "$BACKEND" = "local" ]; then
    # The two local RTX 4090s are reused lane-by-lane.  Sequential execution
    # avoids four model replicas competing for the same two devices and keeps
    # every lane on the exact same scientific two-shard path as the QZ run.
    run_eval_lane r3 no_text 0,1
    run_eval_lane r3 text 0,1
    run_eval_lane r5 no_text 0,1
    run_eval_lane r5 text 0,1
    return 0
  fi
  local pids=()
  local failed=0
  run_eval_lane r3 no_text 0,1 & pids+=("$!")
  run_eval_lane r3 text 2,3 & pids+=("$!")
  run_eval_lane r5 no_text 4,5 & pids+=("$!")
  run_eval_lane r5 text 6,7 & pids+=("$!")
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  [ "$failed" = "0" ] || die "one or more Batch-44 full320 lanes failed"
}

audit_lane_outputs() {
  "$PYTHON" - "$VALIDATION_JSONL" "$RECORD_ROOT/resolved_runs.tsv" "$AGG_ROOT/completeness.json" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

validation_path = Path(sys.argv[1])
resolved_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
validation = [json.loads(line) for line in validation_path.read_text(encoding="utf-8").splitlines() if line.strip()]
expected_by_mode = {
    mode: {str(row["case_id"]) for row in validation if row.get("mode") == mode}
    for mode in ("no_text", "text")
}
with resolved_path.open(encoding="utf-8", newline="") as handle:
    configs = list(csv.DictReader(handle, delimiter="\t"))
if len(configs) != 4:
    raise SystemExit(f"expected four resolved lanes, got {len(configs)}")

payload = {"lanes": []}
for cfg in configs:
    mode = cfg["mode"]
    run_id = cfg["run_id"]
    run_dir = Path(cfg["output_dir"])
    manifests = []
    manifest_paths = sorted(run_dir.glob("manifest.shard*.jsonl"))
    if len(manifest_paths) != 2:
        raise SystemExit(f"{run_id}: expected two inference manifests, got {len(manifest_paths)}")
    for path in manifest_paths:
        manifests.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    ids = [str(row.get("case_id") or "") for row in manifests]
    statuses = Counter(str(row.get("status") or "") for row in manifests)
    modes = Counter(str(row.get("mode") or "") for row in manifests)
    if len(manifests) != 160 or len(set(ids)) != 160 or set(ids) != expected_by_mode[mode]:
        raise SystemExit(f"{run_id}: inference set is not the canonical {mode}160")
    if statuses != Counter({"ok": 160}) or modes != Counter({mode: 160}):
        raise SystemExit(f"{run_id}: statuses={dict(statuses)} modes={dict(modes)}")
    missing_audio = []
    for row in manifests:
        path = Path(str(row.get("output_wav") or ""))
        if not row.get("output_exists") or not path.is_file() or path.stat().st_size <= 1024:
            missing_audio.append(str(row.get("case_id") or ""))
    if missing_audio:
        raise SystemExit(f"{run_id}: missing/small generated audio: {missing_audio[:5]}")

    asr_path = run_dir / f"{run_id}.asr_eval.jsonl"
    asr_rows = [json.loads(line) for line in asr_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    asr_ids = [str(row.get("case_id") or row.get("sample_id") or "") for row in asr_rows]
    if len(asr_rows) != 160 or len(set(asr_ids)) != 160 or set(asr_ids) != expected_by_mode[mode]:
        raise SystemExit(f"{run_id}: ASR set is not the canonical {mode}160")
    if any(str(row.get("mode") or "") != mode or str(row.get("run_id") or "") != run_id for row in asr_rows):
        raise SystemExit(f"{run_id}: ASR mode/run provenance mismatch")

    summary = json.loads((run_dir / f"{run_id}.summary.json").read_text(encoding="utf-8"))
    if int(summary["overall"]["n"]) != 160 or int(summary["by_mode"][mode]["n"]) != 160:
        raise SystemExit(f"{run_id}: merged summary is not n=160")
    infer_logs = sorted((run_dir / "logs").glob("infer.shard*.log"))
    if len(infer_logs) != 2:
        raise SystemExit(f"{run_id}: expected two inference logs")
    bnf_lines = sum(
        path.read_text(encoding="utf-8", errors="replace").count("source semantic memory type=")
        for path in infer_logs
    )
    expected_bnf = 160 if mode == "no_text" else 0
    if bnf_lines != expected_bnf:
        raise SystemExit(f"{run_id}: expected BNF extraction count {expected_bnf}, got {bnf_lines}")
    payload["lanes"].append({
        "arm": cfg["arm"],
        "mode": mode,
        "run_id": run_id,
        "rows": 160,
        "asr_rows": 160,
        "bnf_extraction_lines": bnf_lines,
        "checkpoint": cfg["checkpoint"],
        "training_job_id": cfg["train_job_id"],
    })
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[batch44-full320-completeness] PASS lanes=4 rows=640 output={output_path}")
PY
}

run_dual_encoder_scoring() {
  local run_args=()
  local arm repeat job step mode gpu checkpoint run_id output_dir
  while IFS=$'\t' read -r arm repeat job step mode gpu checkpoint run_id output_dir; do
    [ "$arm" = "arm" ] && continue
    run_args+=(--run "$run_id=$output_dir")
  done < "$RECORD_ROOT/resolved_runs.tsv"
  CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 \
    "$PYTHON" "$CODE_ROOT/scripts/004048_summarize_seedtts_ablation_metrics.py" \
      --validation-jsonl "$VALIDATION_JSONL" \
      "${run_args[@]}" \
      --output-csv "$AGG_ROOT/dual_encoder_cases.csv" \
      --summary-json "$AGG_ROOT/dual_encoder_summary.json" \
      --summary-md "$AGG_ROOT/dual_encoder_summary.md" \
      --speaker-device cuda:0 \
      --speaker-sim-root "$SPEAKER_SIM_ROOT" \
      --extra-speaker-encoder speechbrain_ecapa \
      --extra-speaker-device cuda:1 \
      --speechbrain-ecapa-model-source "$SPEECHBRAIN_ECAPA_MODEL_SOURCE"
}

build_paired_metrics() {
  "$PYTHON" - "$STEP" "$RECORD_ROOT/resolved_runs.tsv" "$AGG_ROOT/dual_encoder_cases.csv" \
    "$AGG_ROOT/paired_metrics" "$R3_TRAIN_JOB_ID" "$R5_TRAIN_JOB_ID" <<'PY'
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

step = int(sys.argv[1])
resolved_path = Path(sys.argv[2])
dual_path = Path(sys.argv[3])
out_prefix = Path(sys.argv[4])
train_jobs = {"r3": sys.argv[5], "r5": sys.argv[6]}
with resolved_path.open(encoding="utf-8", newline="") as handle:
    configs = list(csv.DictReader(handle, delimiter="\t"))
with dual_path.open(encoding="utf-8", newline="") as handle:
    dual_rows = list(csv.DictReader(handle))
if len(configs) != 4 or len(dual_rows) != 640:
    raise SystemExit(f"expected 4 configs and 640 dual rows, got {len(configs)} and {len(dual_rows)}")

def finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None

def mean(values):
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None

def truth(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "keep"}

def normalize(text):
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum() or "\u3400" <= ch <= "\u9fff")

def lcs_len(a, b):
    if not a or not b:
        return 0
    if len(a) < len(b):
        short, long = a, b
    else:
        short, long = b, a
    prev = [0] * (len(short) + 1)
    for ch in long:
        cur = [0]
        for index, other in enumerate(short, start=1):
            cur.append(prev[index - 1] + 1 if ch == other else max(prev[index], cur[-1]))
        prev = cur
    return prev[-1]

def ref_f1(row):
    generated = normalize(row.get("asr_tgt_text"))
    reference = normalize(row.get("timbre_ref_text"))
    hit = lcs_len(generated, reference)
    precision = hit / max(1, len(generated))
    recall = hit / max(1, len(reference))
    return 0.0 if precision + recall <= 0 else 2 * precision * recall / (precision + recall)

by_run = {}
for row in dual_rows:
    by_run.setdefault(row["run"], []).append(row)
by_arm_mode = {}
for cfg in configs:
    run_id = cfg["run_id"]
    rows = by_run.get(run_id, [])
    if len(rows) != 160 or len({row["case_id"] for row in rows}) != 160:
        raise SystemExit(f"{run_id}: dual rows/unique != 160")
    if any(row.get("mode") != cfg["mode"] for row in rows):
        raise SystemExit(f"{run_id}: dual rows contain wrong mode")
    for row in rows:
        for field in ("sim_gen_ref", "sim_gen_source", "ecapa_sim_gen_ref", "ecapa_sim_gen_source", "cer_tgt"):
            if finite(row.get(field)) is None:
                raise SystemExit(f"{run_id}/{row.get('case_id')}: missing {field}")
    run_dir = Path(cfg["output_dir"])
    summary = json.loads((run_dir / f"{run_id}.summary.json").read_text(encoding="utf-8"))["overall"]
    summary_keep = int(summary["keep"])
    dual_keep = sum(truth(row.get("content_keep")) for row in rows)
    if int(summary["n"]) != 160 or summary_keep != dual_keep:
        raise SystemExit(f"{run_id}: 004042 summary and dual-case keep counts disagree")
    asr_path = run_dir / f"{run_id}.asr_eval.jsonl"
    asr_rows = [json.loads(line) for line in asr_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(asr_rows) != 160:
        raise SystemExit(f"{run_id}: ASR rows != 160")
    by_id = {str(row.get("case_id") or row.get("sample_id") or ""): row for row in asr_rows}
    if set(by_id) != {row["case_id"] for row in rows}:
        raise SystemExit(f"{run_id}: ASR/dual case sets disagree")
    for row in rows:
        asr = by_id[row["case_id"]]
        row["timbre_ref_text"] = asr.get("timbre_ref_text")
    by_arm_mode[(cfg["arm"], cfg["mode"])] = rows

metrics = []
for arm in ("r3", "r5"):
    for scope in ("no_text", "text", "all"):
        rows = (
            by_arm_mode[(arm, "no_text")] + by_arm_mode[(arm, "text")]
            if scope == "all"
            else by_arm_mode[(arm, scope)]
        )
        n = len(rows)
        expected_n = 320 if scope == "all" else 160
        if n != expected_n:
            raise SystemExit(f"{arm}/{scope}: n={n}, expected {expected_n}")
        keep = sum(truth(row.get("content_keep")) for row in rows)
        wavlm_ref = [finite(row.get("sim_gen_ref")) for row in rows]
        wavlm_src = [finite(row.get("sim_gen_source")) for row in rows]
        sb_ref = [finite(row.get("ecapa_sim_gen_ref")) for row in rows]
        sb_src = [finite(row.get("ecapa_sim_gen_source")) for row in rows]
        en_src = [row for row in rows if row.get("mode") == "text" and str(row.get("cell") or "").startswith("en_src_")]
        if scope == "text" and len(en_src) != 80:
            raise SystemExit(f"{arm}/text: text en_src n={len(en_src)}, expected 80")
        if scope == "no_text" and en_src:
            raise SystemExit(f"{arm}/no_text unexpectedly contains text en_src")
        metrics.append({
            "step": step,
            "arm": arm,
            "text_repeat": 3 if arm == "r3" else 5,
            "train_job_id": train_jobs[arm],
            "scope": scope,
            "n": n,
            "keep": keep,
            "fail_count": n - keep,
            "fail_rate": (n - keep) / n,
            "cer": mean([finite(row.get("cer_tgt")) for row in rows]),
            "wavlm_sim_ref": mean(wavlm_ref),
            "wavlm_sim_src": mean(wavlm_src),
            "wavlm_margin": mean(wavlm_ref) - mean(wavlm_src),
            "wavlm_ref_bound": sum(ref - src > 0.05 for ref, src in zip(wavlm_ref, wavlm_src)) / n,
            "speechbrain_sim_ref": mean(sb_ref),
            "speechbrain_sim_src": mean(sb_src),
            "speechbrain_margin": mean(sb_ref) - mean(sb_src),
            "speechbrain_ref_bound": sum(ref - src > 0.05 for ref, src in zip(sb_ref, sb_src)) / n,
            "ref_content_lcs_f1": mean([ref_f1(row) for row in rows]),
            "text_en_src_n": len(en_src) if en_src else "",
            "text_en_src_fail_count": sum(not truth(row.get("content_keep")) for row in en_src) if en_src else "",
            "text_en_src_fail_rate": (sum(not truth(row.get("content_keep")) for row in en_src) / len(en_src)) if en_src else "",
            "text_en_src_cer": mean([finite(row.get("cer_tgt")) for row in en_src]) if en_src else "",
        })

out_prefix.parent.mkdir(parents=True, exist_ok=True)
json_path = Path(str(out_prefix) + ".json")
tsv_path = Path(str(out_prefix) + ".tsv")
md_path = Path(str(out_prefix) + ".md")
json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
with tsv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(metrics[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(metrics)

def fmt(value, percent=False):
    if value == "" or value is None:
        return "—"
    return f"{100 * value:.2f}%" if percent else f"{value:.4f}"

lines = [
    f"# Batch-44 r3/r5 paired full320 step-{step}",
    "",
    "Each arm uses the canonical no_text160 + text160 benchmark. `fail` is official `content_keep=False`.",
    "Text `en_src` is the complete 80-case subset, not a quick proxy.",
    "",
    "| Arm | Scope | n | fail | CER | WavLM ref | WavLM src | WavLM margin | WavLM ref-bound | SpB ref | SpB src | SpB ref-bound | F1(ref-content) | text en_src fail | text en_src CER |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in metrics:
    lines.append(
        f"| {row['arm']} | {row['scope']} | {row['n']} | {fmt(row['fail_rate'], True)} | {fmt(row['cer'])} | "
        f"{fmt(row['wavlm_sim_ref'])} | {fmt(row['wavlm_sim_src'])} | {fmt(row['wavlm_margin'])} | "
        f"{fmt(row['wavlm_ref_bound'], True)} | {fmt(row['speechbrain_sim_ref'])} | "
        f"{fmt(row['speechbrain_sim_src'])} | {fmt(row['speechbrain_ref_bound'], True)} | "
        f"{fmt(row['ref_content_lcs_f1'])} | {fmt(row['text_en_src_fail_rate'], True)} | "
        f"{fmt(row['text_en_src_cer'])} |"
    )
lines.extend([
    "",
    "## r5 - r3",
    "",
    "| Scope | Δfail | ΔCER | ΔWavLM ref | ΔWavLM src | ΔWavLM margin | ΔWavLM ref-bound | ΔSpB ref | ΔF1 | Δtext en_src fail |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
])
indexed = {(row["arm"], row["scope"]): row for row in metrics}
for scope in ("no_text", "text", "all"):
    a = indexed[("r3", scope)]
    b = indexed[("r5", scope)]
    en_delta = "—" if scope == "no_text" else fmt(b["text_en_src_fail_rate"] - a["text_en_src_fail_rate"], True)
    lines.append(
        f"| {scope} | {b['fail_rate'] - a['fail_rate']:+.2%} | {b['cer'] - a['cer']:+.4f} | "
        f"{b['wavlm_sim_ref'] - a['wavlm_sim_ref']:+.4f} | {b['wavlm_sim_src'] - a['wavlm_sim_src']:+.4f} | "
        f"{b['wavlm_margin'] - a['wavlm_margin']:+.4f} | {b['wavlm_ref_bound'] - a['wavlm_ref_bound']:+.2%} | "
        f"{b['speechbrain_sim_ref'] - a['speechbrain_sim_ref']:+.4f} | "
        f"{b['ref_content_lcs_f1'] - a['ref_content_lcs_f1']:+.4f} | {en_delta} |"
    )
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[batch44-full320-metrics] PASS rows={len(metrics)} output={md_path}")
PY
}

write_local_completion() {
  [ "$BACKEND" = "local" ] || die "write_local_completion called for backend=$BACKEND"
  [ -s "$LOCAL_FROZEN_RUNNER" ] || die "missing frozen local runner: $LOCAL_FROZEN_RUNNER"
  [ -s "$LOCAL_ENGINE_SOURCE" ] || die "missing local full320 engine: $LOCAL_ENGINE_SOURCE"
  [ -s "$LOCAL_INPUTS_MANIFEST" ] || die "missing local frozen-input manifest: $LOCAL_INPUTS_MANIFEST"
  [ -s "$RECORD_ROOT/runtime_gpu_inventory.json" ] || die "missing local GPU inventory JSON"
  [ ! -e "$RECORD_ROOT/submitted_jobs.tsv" ] && [ ! -L "$RECORD_ROOT/submitted_jobs.tsv" ] \
    || die "local completion forbids submitted_jobs.tsv"
  "$PYTHON" - "$COMPLETION_JSON" "$STEP" "$RECORD_ROOT" "$STEP_ROOT" "$CODE_ROOT" \
    "$VALIDATION_JSONL" "$TRAIN_PAIR_LEDGER" "$R3_TRAIN_JOB_ID" "$R5_TRAIN_JOB_ID" \
    "$LOCAL_FROZEN_RUNNER" "$LOCAL_ENGINE_SOURCE" "$LOCAL_INPUTS_MANIFEST" \
    "$RECORD_ROOT/runtime_gpu_inventory.json" "${RUN_STARTED_UTC:-}" <<'PY'
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

output = Path(sys.argv[1])
step = int(sys.argv[2])
record_root = Path(sys.argv[3]).resolve()
step_root = Path(sys.argv[4]).resolve()
code_root = Path(sys.argv[5]).resolve()
validation = Path(sys.argv[6]).resolve()
train_ledger = Path(sys.argv[7]).resolve()
r3_job, r5_job = sys.argv[8:10]
runner = Path(sys.argv[10]).resolve()
engine = Path(sys.argv[11]).resolve()
inputs_manifest = Path(sys.argv[12]).resolve()
gpu_inventory_path = Path(sys.argv[13]).resolve()
started_utc = sys.argv[14]

def sha256(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"missing completion input: {path}")
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

if runner.parent != record_root or runner.name != "004118_run_batch44_v1_paired_full320_local.frozen.sh":
    raise SystemExit(f"unregistered frozen local runner path: {runner}")
if os.path.lexists(record_root / "submitted_jobs.tsv"):
    raise SystemExit("local completion forbids submitted_jobs.tsv")

gpu_inventory = json.loads(gpu_inventory_path.read_text(encoding="utf-8"))
gpus = gpu_inventory.get("gpus") if isinstance(gpu_inventory, dict) else None
hostname = socket.gethostname()
if not hostname.startswith("xyzhang-dev--") or gpu_inventory.get("hostname") != hostname:
    raise SystemExit(f"local host provenance drift: {hostname!r}/{gpu_inventory.get('hostname')!r}")
if not isinstance(gpus, list) or len(gpus) != 2:
    raise SystemExit("local completion requires exactly two inventoried GPUs")
if [row.get("index") for row in gpus] != [0, 1]:
    raise SystemExit(f"local GPU indices drift: {gpus!r}")
if any(row.get("name") != "NVIDIA GeForce RTX 4090" for row in gpus):
    raise SystemExit(f"local GPU model drift: {gpus!r}")
if any(int(row.get("memory_total_mib") or 0) < 48_000 for row in gpus):
    raise SystemExit(f"local GPU memory drift: {gpus!r}")
if len({str(row.get("uuid") or "") for row in gpus}) != 2:
    raise SystemExit(f"local GPU UUID drift: {gpus!r}")

artifact_paths = {
    "completeness_json": step_root / "aggregate/completeness.json",
    "dual_encoder_cases_csv": step_root / "aggregate/dual_encoder_cases.csv",
    "paired_metrics_tsv": step_root / "aggregate/paired_metrics.tsv",
    "paired_metrics_json": step_root / "aggregate/paired_metrics.json",
    "paired_metrics_md": step_root / "aggregate/paired_metrics.md",
}
artifacts = {
    key: {"path": str(path), "sha256": sha256(path)}
    for key, path in artifact_paths.items()
}
checkpoint_provenance = {
    arm: {
        "path": str(record_root / f"checkpoint_{arm}_step{step}.json"),
        "sha256": sha256(record_root / f"checkpoint_{arm}_step{step}.json"),
    }
    for arm in ("r3", "r5")
}
resolved_runs = record_root / "resolved_runs.tsv"
completed_utc = dt.datetime.now(dt.timezone.utc).isoformat()
payload = {
    "schema": "batch44_v1_paired_full320_v1",
    "status": "complete",
    "backend": "local",
    "completed_utc": completed_utc,
    "step": step,
    "training_jobs": {"r3": r3_job, "r5": r5_job},
    "training_pair_ledger": str(train_ledger),
    "training_pair_ledger_sha256": sha256(train_ledger),
    "validation_jsonl": str(validation),
    "validation_sha256": sha256(validation),
    "code_root": str(code_root),
    "record_root": str(record_root),
    "step_root": str(step_root),
    **{key: value["path"] for key, value in artifacts.items()},
    "scope": {"r3": {"no_text": 160, "text": 160}, "r5": {"no_text": 160, "text": 160}},
    "gpu_plan": {
        "r3_no_text": "0,1", "r3_text": "0,1", "r5_no_text": "0,1", "r5_text": "0,1"
    },
    "lane_execution": "sequential",
    "execution": {
        "hostname": hostname,
        "pid": os.getppid(),
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "gpu_count": 2,
        "gpu_indices": [0, 1],
        "gpu_models": [str(row["name"]) for row in gpus],
        "gpu_uuids": [str(row["uuid"]) for row in gpus],
        "gpu_memory_total_mib": [int(row["memory_total_mib"]) for row in gpus],
        "gpu_inventory": str(gpu_inventory_path),
        "gpu_inventory_sha256": sha256(gpu_inventory_path),
    },
    "runner": {"path": str(runner), "sha256": sha256(runner)},
    "engine": {"path": str(engine), "sha256": sha256(engine)},
    "inputs_manifest": {"path": str(inputs_manifest), "sha256": sha256(inputs_manifest)},
    "resolved_runs": {"path": str(resolved_runs), "sha256": sha256(resolved_runs)},
    "checkpoint_provenance": checkpoint_provenance,
    "artifacts": artifacts,
}
temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
completion_sha = sha256(output)
marker = record_root / "complete.marker"
marker_temp = marker.with_name(f".{marker.name}.tmp-{os.getpid()}")
marker_temp.write_text(f"COMPLETED.json sha256\t{completion_sha}\n", encoding="utf-8")
os.replace(marker_temp, marker)
print(f"[batch44-full320-complete] backend=local sha256={completion_sha} wrote {output}")
PY
}

write_completion() {
  if [ "$BACKEND" = "local" ]; then
    write_local_completion
    return 0
  fi
  "$PYTHON" - "$COMPLETION_JSON" "$STEP" "$RECORD_ROOT" "$STEP_ROOT" "$CODE_ROOT" \
    "$VALIDATION_JSONL" "$TRAIN_PAIR_LEDGER" "$R3_TRAIN_JOB_ID" "$R5_TRAIN_JOB_ID" \
    "$ALLOWED_COMPUTE_GROUP" "$ALLOWED_SPEC" "$JOB_NAME" <<'PY'
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

output = Path(sys.argv[1])
step = int(sys.argv[2])
record_root = Path(sys.argv[3]).resolve()
step_root = Path(sys.argv[4]).resolve()
code_root = Path(sys.argv[5]).resolve()
validation = Path(sys.argv[6]).resolve()
train_ledger = Path(sys.argv[7]).resolve()
r3_job, r5_job = sys.argv[8:10]
compute_group, spec, job_name = sys.argv[10:13]
submission_ledger = record_root / "submitted_jobs.tsv"
job_re = re.compile(
    r"^job-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

with submission_ledger.open(encoding="utf-8", newline="") as handle:
    submitted_rows = list(csv.DictReader(handle, delimiter="\t"))
if len(submitted_rows) != 1:
    raise SystemExit(f"{submission_ledger}: expected exactly one submission row")
submitted = submitted_rows[0]
expected_ledger = {
    "job_name": job_name,
    "step": str(step),
    "compute_group": compute_group,
    "spec": spec,
    "r3_train_job_id": r3_job,
    "r5_train_job_id": r5_job,
}
bad_ledger = {
    key: {"expected": wanted, "actual": submitted.get(key)}
    for key, wanted in expected_ledger.items()
    if submitted.get(key) != wanted
}
if bad_ledger:
    raise SystemExit(f"submission ledger drift: {bad_ledger}")
expected_ledger_paths = {
    "record_root": record_root,
    "step_root": step_root,
    "code_root": code_root,
}
bad_ledger_paths = {}
for key, wanted in expected_ledger_paths.items():
    actual = Path(str(submitted.get(key) or "")).expanduser().resolve()
    if actual != wanted.resolve():
        bad_ledger_paths[key] = {
            "expected": str(wanted.resolve()),
            "actual": str(actual),
        }
if bad_ledger_paths:
    raise SystemExit(f"submission ledger path drift: {bad_ledger_paths}")
evaluation_job_id = str(submitted.get("job_id") or "")
if not job_re.fullmatch(evaluation_job_id):
    raise SystemExit(f"invalid evaluation QZ job id: {evaluation_job_id!r}")

payload = {
    "schema": "batch44_v1_paired_full320_v1",
    "status": "complete",
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "step": step,
    "training_jobs": {"r3": r3_job, "r5": r5_job},
    "training_pair_ledger": str(train_ledger),
    "training_pair_ledger_sha256": sha256(train_ledger),
    "validation_jsonl": str(validation),
    "validation_sha256": sha256(validation),
    "code_root": str(code_root),
    "record_root": str(record_root),
    "step_root": str(step_root),
    "evaluation_job": {
        "job_name": job_name,
        "job_id": evaluation_job_id,
        "submission_ledger": str(submission_ledger),
        "submission_ledger_sha256": sha256(submission_ledger),
    },
    "completeness_json": str(step_root / "aggregate/completeness.json"),
    "dual_encoder_cases_csv": str(step_root / "aggregate/dual_encoder_cases.csv"),
    "paired_metrics_tsv": str(step_root / "aggregate/paired_metrics.tsv"),
    "paired_metrics_json": str(step_root / "aggregate/paired_metrics.json"),
    "paired_metrics_md": str(step_root / "aggregate/paired_metrics.md"),
    "scope": {"r3": {"no_text": 160, "text": 160}, "r5": {"no_text": 160, "text": 160}},
    "gpu_plan": {
        "r3_no_text": "0,1", "r3_text": "2,3", "r5_no_text": "4,5", "r5_text": "6,7"
    },
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(record_root / "complete.marker").write_text(payload["completed_utc"] + "\n", encoding="utf-8")
print(f"[batch44-full320-complete] wrote {output}")
PY
}

run_entrypoint() {
  mkdir -p "$RECORD_ROOT" "$RUNS_ROOT" "$AGG_ROOT"
  exec > >(tee -a "$RECORD_ROOT/run.log") 2>&1
  RUN_STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  export RUN_STARTED_UTC
  echo "[batch44-full320] start backend=$BACKEND step=$STEP date=$RUN_STARTED_UTC host=$(hostname)"
  echo "[batch44-full320] training_jobs=r3:$R3_TRAIN_JOB_ID,r5:$R5_TRAIN_JOB_ID"
  if [ "$BACKEND" = "local" ]; then
    echo "[batch44-full320] gpu_plan=sequential:r3-no_text:0,1+r3-text:0,1+r5-no_text:0,1+r5-text:0,1"
  else
    echo "[batch44-full320] gpu_plan=parallel:r3-no_text:0,1+r3-text:2,3+r5-no_text:4,5+r5-text:6,7"
  fi
  audit_static_inputs
  validate_checkpoint r3
  validate_checkpoint r5
  ensure_evaluation_is_new
  write_resolved_runs
  audit_runtime_gpus
  nvidia-smi || true
  if [ "$LOCAL_PREFLIGHT_ONLY" = "1" ]; then
    echo "[batch44-full320] local preflight passed; inference/scoring was not started"
    exit 0
  fi
  run_four_lanes
  audit_lane_outputs
  run_dual_encoder_scoring
  build_paired_metrics
  write_completion
  echo "[batch44-full320] complete step=$STEP metrics=$AGG_ROOT/paired_metrics.tsv"
}

if [ "$ENTRYPOINT" = "1" ]; then
  run_entrypoint
  exit 0
fi

[ -x "$QZCLI" ] || die "qzcli-local wrapper is not executable: $QZCLI"
[ -d "$QZCLI_HOME" ] || die "qzcli-local HOME is missing: $QZCLI_HOME"
mkdir -p "$RECORD_ROOT" "$RUNS_ROOT" "$AGG_ROOT"
audit_static_inputs
write_resolved_runs
if [ "$STATIC_AUDIT_ONLY" = "1" ]; then
  echo "[batch44-full320] static audit passed; checkpoints and QZ were not touched"
  exit 0
fi
validate_checkpoint r3
validate_checkpoint r5
ensure_evaluation_is_new

RUNNER_SOURCE="$PROJECT_ROOT/scripts/004112_submit_batch44_v1_paired_full320_qz.sh"
[ -s "$RUNNER_SOURCE" ] || die "missing Batch-44 full320 wrapper: $RUNNER_SOURCE"
if [ -s "$FROZEN_RUNNER" ]; then
  cmp -s "$RUNNER_SOURCE" "$FROZEN_RUNNER" || die "frozen runner drift at $FROZEN_RUNNER"
else
  cp "$RUNNER_SOURCE" "$FROZEN_RUNNER"
  chmod 0555 "$FROZEN_RUNNER"
fi
{
  sha256sum "$FROZEN_RUNNER" "$VALIDATION_JSONL" "$TRAIN_PAIR_LEDGER" "$RECORD_ROOT/resolved_runs.tsv"
  sha256sum \
    "$CODE_ROOT/scripts/004039_run_seedtts_validation_eval.sh" \
    "$CODE_ROOT/scripts/004042_summarize_seedtts_validation_eval.py" \
    "$CODE_ROOT/scripts/004044_run_seedtts_validation_infer_persistent.py" \
    "$CODE_ROOT/scripts/004048_summarize_seedtts_ablation_metrics.py" \
    "$CODE_ROOT/scripts/004056_summarize_seedtts_ref_content_similarity.py" \
    "$CODE_ROOT/moss_codecvc/models/moss_codecvc_wrapper.py"
} > "$RECORD_ROOT/frozen_inputs.sha256"

if [ "$DRY_RUN" = "0" ]; then
  [ "$PROJECT_ROOT" = "$CANONICAL_PROJECT_ROOT" ] || die "live submission requires canonical PROJECT_ROOT"
  [ "$R3_RUN_DIR" = "$CANONICAL_PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r3_v1_30k" ] || die "live r3 run dir is not canonical"
  [ "$R5_RUN_DIR" = "$CANONICAL_PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r5_v1_30k" ] || die "live r5 run dir is not canonical"
  [ "$RECORD_ROOT" = "$CANONICAL_PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_paired_full320_step${STEP}_${STAMP}" ] || die "live record root is not canonical"
  [ "$EVAL_ROOT" = "$CANONICAL_PROJECT_ROOT/testset/outputs/ver23_batch44_paired_full320_${STAMP}" ] || die "live eval root is not canonical"
  if ! mkdir "$SUBMISSION_LOCK" 2>/dev/null; then
    die "persistent live-submit lock already exists: $SUBMISSION_LOCK"
  fi
  "$PYTHON" - "$SUBMISSION_LOCK/owner.json" "$STEP" <<'PY'
import datetime as dt
import json
import os
import socket
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "host": socket.gethostname(),
    "pid": os.getppid(),
    "step": int(sys.argv[2]),
    "policy": "persistent lock; inspect QZ state before any manual recovery",
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
fi

COMMAND="env BATCH44_PAIRED_FULL320_ENTRYPOINT=1 STEP=$STEP SEED=$SEED PROJECT_ROOT=$PROJECT_ROOT CODE_ROOT=$CODE_ROOT R3_RUN_DIR=$R3_RUN_DIR R5_RUN_DIR=$R5_RUN_DIR RECORD_ROOT=$RECORD_ROOT EVAL_ROOT=$EVAL_ROOT VALIDATION_JSONL=$VALIDATION_JSONL PYTHON=$PYTHON ASR_PYTHON=$ASR_PYTHON SPEAKER_SIM_ROOT=$SPEAKER_SIM_ROOT SPEECHBRAIN_ECAPA_MODEL_SOURCE=$SPEECHBRAIN_ECAPA_MODEL_SOURCE MIN_CHECKPOINT_AGE_SEC=$MIN_CHECKPOINT_AGE_SEC bash $FROZEN_RUNNER"
if [ "$DRY_RUN" = "1" ]; then
  SUBMIT_OUTPUT="$RECORD_ROOT/submit_output.dry_run.txt"
else
  SUBMIT_OUTPUT="$RECORD_ROOT/submit_output.txt"
fi

echo "=========================================="
echo "QZ submit: Batch-44 paired full320"
echo "  JOB_NAME=$JOB_NAME"
echo "  STEP=$STEP (same checkpoint step for r3/r5)"
echo "  TRAINING_JOBS=r3:$R3_TRAIN_JOB_ID r5:$R5_TRAIN_JOB_ID"
echo "  SCOPE=each arm no_text160 + text160; text en_src full n=80"
echo "  GPU_PLAN=r3-no_text:0,1 r3-text:2,3 r5-no_text:4,5 r5-text:6,7"
echo "  SCORERS=WavLM-large-SV + SpeechBrain ECAPA; ASR + ref-content F1"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP (MTTS-3-2-0715 only)"
echo "  SPEC=$SPEC (8x $ALLOWED_GPU_TYPE) INSTANCES=$INSTANCES"
echo "  RECORD_ROOT=$RECORD_ROOT"
echo "  STEP_ROOT=$STEP_ROOT"
echo "  CODE_ROOT=$CODE_ROOT"
echo "  DRY_RUN=$DRY_RUN CONFIRM_BATCH44_FULL320=$CONFIRM_BATCH44_FULL320"
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
  die "QZ dry-run/submission failed; inspect $SUBMIT_OUTPUT"
fi
if [ "$DRY_RUN" = "1" ]; then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RECORD_ROOT/dry_run.ok"
  echo "[batch44-full320] platform dry-run passed; no job submitted"
  exit 0
fi

"$PYTHON" - "$SUBMIT_OUTPUT" "$RECORD_ROOT/submitted_jobs.tsv" "$JOB_NAME" "$STEP" \
  "$COMPUTE_GROUP" "$SPEC" "$RECORD_ROOT" "$STEP_ROOT" "$CODE_ROOT" \
  "$R3_TRAIN_JOB_ID" "$R5_TRAIN_JOB_ID" <<'PY'
import re
import sys
from pathlib import Path

submit_output = Path(sys.argv[1])
ledger = Path(sys.argv[2])
values = sys.argv[3:]
pattern = re.compile(
    r"job-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
job_ids = sorted(set(pattern.findall(submit_output.read_text(encoding="utf-8", errors="replace"))))
if len(job_ids) != 1:
    raise SystemExit(f"expected exactly one complete QZ job UUID, got {job_ids}")
fields = [
    "job_name", "job_id", "step", "compute_group", "spec", "record_root", "step_root",
    "code_root", "r3_train_job_id", "r5_train_job_id",
]
row = [values[0], job_ids[0], *values[1:]]
ledger.write_text("\t".join(fields) + "\n" + "\t".join(row) + "\n", encoding="utf-8")
print(f"[batch44-full320] submitted job_id={job_ids[0]} ledger={ledger}")
PY

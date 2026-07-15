#!/usr/bin/env bash
# Batch-44 r3/r5 paired quick20 evaluation for one exact 2k checkpoint.
#
# One MTTS-3-2-0715 node is split into four independent 2-GPU evaluations:
#   r3 no_text -> GPUs 0,1       r3 text -> GPUs 2,3
#   r5 no_text -> GPUs 4,5       r5 text -> GPUs 6,7
#
# This wrapper never waits for a checkpoint while occupying a QZ node.  Both
# arms must expose a complete checkpoint at the same STEP before submission.
# It defaults to a platform dry-run.  A live submission additionally requires
# CONFIRM_BATCH44_QUICK20=1.
#
#   STEP=2000 DRY_RUN=1 bash scripts/004110_submit_batch44_v1_quick20_qz.sh
#   STEP=2000 DRY_RUN=0 CONFIRM_BATCH44_QUICK20=1 \
#     bash scripts/004110_submit_batch44_v1_quick20_qz.sh

set -euo pipefail

CANONICAL_PROJECT_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
PROJECT_ROOT="${PROJECT_ROOT:-$CANONICAL_PROJECT_ROOT}"
CODE_ROOT="${CODE_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch37_eval_20260711_1092820}"
QZCLI="${QZCLI:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/qzcli_with_deps.sh}"
QZCLI_HOME="${QZCLI_HOME:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/.codex/qzcli_home}"

ALLOWED_WORKSPACE="ws-8207e9e2-e733-4eec-a475-cfa1c36480ba"           # CI-情境智能
ALLOWED_PROJECT="project-c67c548f-f02c-453b-ba5b-8745db6886e7"       # CI-情境智能
WORKSPACE="${WORKSPACE:-$ALLOWED_WORKSPACE}"
PROJECT="${PROJECT:-$ALLOWED_PROJECT}"
ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122" # MTTS-3-2-0715
ALLOWED_SPEC="67b10bc6-78b0-41a3-aaf4-358eeeb99009"              # 8xH200
ALLOWED_GPU_TYPE="NVIDIA_H200_SXM_141G"
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
STEP="${STEP:-2000}"
DRY_RUN="${DRY_RUN:-1}"
CONFIRM_BATCH44_QUICK20="${CONFIRM_BATCH44_QUICK20:-0}"
ENTRYPOINT="${BATCH44_QUICK20_ENTRYPOINT:-0}"
STATIC_AUDIT_ONLY="${STATIC_AUDIT_ONLY:-0}"
COLLECT_ONLY="${COLLECT_ONLY:-0}"
TEST_MODE="${BATCH44_QUICK20_TEST_MODE:-0}"
LIBRARY_MODE="${BATCH44_QUICK20_LIBRARY_MODE:-0}"

R3_RUN_DIR="${R3_RUN_DIR:-$PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r3_v1_30k}"
R5_RUN_DIR="${R5_RUN_DIR:-$PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r5_v1_30k}"
R3_TRAIN_JOB_ID="job-2b91d332-d500-4279-84f9-0a6a81a376aa"
R5_TRAIN_JOB_ID="job-b8eb2f1f-a3eb-483b-a289-b4cce281525c"
TRAIN_PAIR_LEDGER="$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713/submitted_pair.tsv"
RECORD_ROOT="${RECORD_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_quick20_step${STEP}_${STAMP}}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/testset/outputs/ver23_batch44_quick20_${STAMP}}"
JOB_NAME="${JOB_NAME:-ver23_batch44_quick20_step${STEP}_${STAMP}}"

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
ASR_PYTHON="${ASR_PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
NO_TEXT20_JSONL="${NO_TEXT20_JSONL:-$PROJECT_ROOT/testset/validation/ver2_8_timbre_quick20_seedtts_no_text_20260704.jsonl}"
NO_TEXT20_SHA256="f28de52e87b8c422380fe22052039ce48d59a1662a8bf7b137ce67405c35fba0"
TEXT_SOURCE_JSONL="${TEXT_SOURCE_JSONL:-$PROJECT_ROOT/testset/validation/seedtts_vc_ver2_3_validation.jsonl}"
TEXT_SOURCE_SHA256="725ee9d58a7e6066d2a7b79c858cb6ff4dd7292cc167c45dc6b6ebbeaff2fe14"
TEXT20_JSONL="${TEXT20_JSONL:-$RECORD_ROOT/ver23_batch44_text_quick20_8cell_20260713.jsonl}"
TEXT20_SHA256="0952c4162e7ff7a9c2850f1f76f572f2f710e205b222c874016b05564f21bea8"
COMPLETION_JSON="$RECORD_ROOT/COMPLETED.json"
COMPLETE_MARKER="$RECORD_ROOT/complete.marker"
FROZEN_RUNNER_PATH="$RECORD_ROOT/004110_submit_batch44_v1_quick20_qz.frozen.sh"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

case "$STEP" in
  2000|4000|6000|8000|10000|12000|14000|16000|18000|20000|22000|24000|26000|28000|30000) ;;
  *) die "STEP must be one of 2000,4000,...,30000; got $STEP" ;;
esac
case "$DRY_RUN:$CONFIRM_BATCH44_QUICK20:$ENTRYPOINT:$STATIC_AUDIT_ONLY:$COLLECT_ONLY:$TEST_MODE" in
  [01]:[01]:[01]:[01]:[01]:[01]) ;;
  *) die "boolean flags must be 0 or 1" ;;
esac
case "$LIBRARY_MODE" in
  0|1) ;;
  *) die "BATCH44_QUICK20_LIBRARY_MODE must be 0 or 1" ;;
esac
if [ "$COLLECT_ONLY" = "1" ] && { [ "$ENTRYPOINT" = "1" ] || [ "$STATIC_AUDIT_ONLY" = "1" ]; }; then
  die "COLLECT_ONLY may not be combined with ENTRYPOINT or STATIC_AUDIT_ONLY"
fi
if [ "$PROJECT_ROOT" != "$CANONICAL_PROJECT_ROOT" ] && [ "$TEST_MODE" != "1" ]; then
  die "non-canonical PROJECT_ROOT is allowed only with BATCH44_QUICK20_TEST_MODE=1"
fi
if [ "$TEST_MODE" = "1" ] && { [ "$DRY_RUN" = "0" ] || [ "$ENTRYPOINT" = "1" ]; }; then
  die "test mode may not submit or run a QZ entrypoint"
fi
if [ "$COMPUTE_GROUP" != "$ALLOWED_COMPUTE_GROUP" ]; then
  die "Batch-44 quick20 may only use MTTS-3-2-0715 ($ALLOWED_COMPUTE_GROUP)"
fi
if [ "$WORKSPACE" != "$ALLOWED_WORKSPACE" ] || [ "$PROJECT" != "$ALLOWED_PROJECT" ]; then
  die "Batch-44 quick20 is restricted to the CI-情境智能 workspace/project"
fi
if [ "$SPEC" != "$ALLOWED_SPEC" ]; then
  die "Batch-44 quick20 requires the registered 8xH200 spec $ALLOWED_SPEC"
fi
if [ "$QZCLI_GPU_TYPE_OVERRIDE" != "$ALLOWED_GPU_TYPE" ]; then
  die "Batch-44 quick20 requires GPU type $ALLOWED_GPU_TYPE"
fi
if [ "$INSTANCES" != "1" ]; then
  die "Batch-44 quick20 requires exactly one 8xH200 instance"
fi
if [ "$DRY_RUN" = "0" ] && [ "$CONFIRM_BATCH44_QUICK20" != "1" ]; then
  die "live submission requires CONFIRM_BATCH44_QUICK20=1"
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
    # Best3/final consumers.  The training run directory carries `_v1_30k`,
    # but the canonical evaluation run-id intentionally does not.
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
  printf '%s_step-%s_%s_quick20_d2d3_seed1234\n' "$(arm_label "$arm")" "$STEP" "$mode"
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
    "scripts/004044_run_seedtts_validation_infer_persistent.py": "22045797d68d54bc2b72c64773c43464e4164b19b3a29d97537149e15594fa1d",
    "scripts/004050_summarize_seedtts_speaker_sim_only.py": "420c63c59f1b430a6daa546a2ed7874cb407aedf390b88dabf21801d44dd1e42",
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
if errors:
    raise SystemExit("Batch-44 eval snapshot audit failed:\n- " + "\n- ".join(errors))

eval_text = (root / "scripts/004039_run_seedtts_validation_eval.sh").read_text(encoding="utf-8")
wrapper_text = (root / "moss_codecvc/models/moss_codecvc_wrapper.py").read_text(encoding="utf-8")
infer_text = (root / "scripts/004044_run_seedtts_validation_infer_persistent.py").read_text(encoding="utf-8")
for needle in (
    'export HF_MODULES_CACHE="$hf_modules_cache_root/shard${shard}"',
    'mkdir -p "$HF_MODULES_CACHE"',
):
    if needle not in eval_text:
        errors.append(f"004039 missing shard-cache isolation: {needle}")
for needle in (
    "_content_cross_attn_active_sample_mask",
    "content_cross_attn_text_bypass_samples",
):
    if needle not in wrapper_text:
        errors.append(f"wrapper missing text-row BNF bypass: {needle}")
if "content_cross_attn_needs_features = content_cross_attn_encoder is not None and no_text" not in infer_text:
    errors.append("004044 does not bypass BNF feature preparation for text inference")
if errors:
    raise SystemExit("Batch-44 eval behavior audit failed:\n- " + "\n- ".join(errors))
print(f"[batch44-quick20-code] PASS root={root}")
PY
  bash -n "$CODE_ROOT/scripts/004039_run_seedtts_validation_eval.sh"
}

audit_training_pair() {
  if [ "$TEST_MODE" = "1" ]; then
    return 0
  fi
  "$PYTHON" - "$TRAIN_PAIR_LEDGER" "$R3_RUN_DIR" "$R5_RUN_DIR" \
    "$R3_TRAIN_JOB_ID" "$R5_TRAIN_JOB_ID" "$ALLOWED_COMPUTE_GROUP" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

ledger = Path(sys.argv[1])
r3_out, r5_out, r3_job, r5_job, compute = sys.argv[2:7]
if not ledger.is_file():
    raise SystemExit(f"missing Batch-44 training-pair ledger: {ledger}")
with ledger.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
expected = {
    "ver2_9_5_final_r3_v1_30k": (r3_job, r3_out),
    "ver2_9_5_final_r5_v1_30k": (r5_job, r5_out),
}
errors = []
if len(rows) != 2:
    errors.append(f"expected two training rows, got {len(rows)}")
seen = set()
for row in rows:
    name = row.get("job_name", "")
    seen.add(name)
    if name not in expected:
        errors.append(f"unexpected training job name={name!r}")
        continue
    wanted_job, wanted_out = expected[name]
    if row.get("job_id") != wanted_job:
        errors.append(f"{name} job_id={row.get('job_id')!r}, expected {wanted_job}")
    if row.get("out_dir") != wanted_out:
        errors.append(f"{name} out_dir={row.get('out_dir')!r}, expected {wanted_out}")
    if row.get("compute_group") != compute:
        errors.append(f"{name} compute_group={row.get('compute_group')!r}, expected {compute}")
if seen != set(expected):
    errors.append(f"training pair names={sorted(seen)}, expected {sorted(expected)}")
if errors:
    raise SystemExit("Batch-44 training provenance audit failed:\n- " + "\n- ".join(errors))
print(f"[batch44-quick20-training] PASS r3={r3_job} r5={r5_job}")
PY
}

prepare_and_validate_quick_sets() {
  mkdir -p "$RECORD_ROOT"
  "$PYTHON" - \
    "$NO_TEXT20_JSONL" "$NO_TEXT20_SHA256" \
    "$TEXT_SOURCE_JSONL" "$TEXT_SOURCE_SHA256" \
    "$TEXT20_JSONL" "$TEXT20_SHA256" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

no_text_path = Path(sys.argv[1])
no_text_sha = sys.argv[2]
text_source = Path(sys.argv[3])
text_source_sha = sys.argv[4]
output = Path(sys.argv[5])
output_sha = sys.argv[6]

def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

for path, wanted in ((no_text_path, no_text_sha), (text_source, text_source_sha)):
    if not path.is_file():
        raise SystemExit(f"missing quick20 input: {path}")
    got = sha256(path)
    if got != wanted:
        raise SystemExit(f"quick20 input drift: {path}: {got}, expected {wanted}")

no_text_rows = [json.loads(line) for line in no_text_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(no_text_rows) != 20:
    raise SystemExit(f"no_text quick20 must contain 20 rows, got {len(no_text_rows)}")
bad = [row.get("case_id") for row in no_text_rows if row.get("mode") != "no_text"]
if bad:
    raise SystemExit(f"no_text quick20 contains wrong modes: {bad}")

case_ids = [
    "seedtts_text_en_src_zh_ref_m2f_000000",
    "seedtts_text_en_src_zh_ref_m2f_000001",
    "seedtts_text_en_src_zh_ref_m2f_000002",
    "seedtts_text_en_src_zh_ref_f2m_000000",
    "seedtts_text_en_src_zh_ref_f2m_000001",
    "seedtts_text_en_src_zh_ref_f2m_000002",
    "seedtts_text_zh_src_en_ref_m2f_000000",
    "seedtts_text_zh_src_en_ref_m2f_000001",
    "seedtts_text_zh_src_en_ref_f2m_000000",
    "seedtts_text_zh_src_en_ref_f2m_000001",
    "seedtts_text_en_src_en_ref_same_gender_000000",
    "seedtts_text_en_src_en_ref_same_gender_000001",
    "seedtts_text_en_src_en_ref_same_gender_000002",
    "seedtts_text_zh_src_zh_ref_same_gender_000000",
    "seedtts_text_zh_src_zh_ref_same_gender_000001",
    "seedtts_text_en_src_zh_ref_same_gender_000000",
    "seedtts_text_en_src_zh_ref_same_gender_000001",
    "seedtts_text_en_src_zh_ref_same_gender_000002",
    "seedtts_text_zh_src_en_ref_same_gender_000000",
    "seedtts_text_zh_src_en_ref_same_gender_000001",
]
selected_by_id = {}
wanted_ids = set(case_ids)
for line in text_source.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    case_id = str(row.get("case_id") or "")
    if case_id in wanted_ids:
        if case_id in selected_by_id:
            raise SystemExit(f"duplicate text quick20 case: {case_id}")
        selected_by_id[case_id] = row
missing = [case_id for case_id in case_ids if case_id not in selected_by_id]
if missing:
    raise SystemExit(f"missing text quick20 cases: {missing}")
selected = [selected_by_id[case_id] for case_id in case_ids]
counts = Counter(str(row.get("cell") or "") for row in selected)
if len(selected) != 20 or sorted(counts.values()) != [2, 2, 2, 2, 3, 3, 3, 3]:
    raise SystemExit(f"unexpected text quick20 cells: {dict(counts)}")
en_src_n = sum(str(row.get("cell") or "").startswith("en_src_") for row in selected)
if en_src_n != 12:
    raise SystemExit(f"text quick20 en_src proxy must contain 12 rows, got {en_src_n}")
payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in selected)
got_output_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
if got_output_sha != output_sha:
    raise SystemExit(f"text quick20 payload drift: {got_output_sha}, expected {output_sha}")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(payload, encoding="utf-8")
print(
    "[batch44-quick20-data] PASS "
    f"no_text=20 text=20 text_en_src_proxy_n={en_src_n} text_sha256={got_output_sha}"
)
PY
}

validate_checkpoint() {
  local arm="$1"
  local run_dir checkpoint expected_repeat
  run_dir="$(arm_run_dir "$arm")"
  checkpoint="$(arm_checkpoint "$arm")"
  case "$arm" in r3) expected_repeat=3 ;; r5) expected_repeat=5 ;; esac
  "$PYTHON" - "$arm" "$expected_repeat" "$run_dir" "$checkpoint" "$PROJECT_ROOT" "$STEP" <<'PY'
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

arm = sys.argv[1]
repeat = int(sys.argv[2])
run_dir = Path(sys.argv[3])
checkpoint = Path(sys.argv[4])
project_root = Path(sys.argv[5])
step = int(sys.argv[6])
required = {
    "adapter_model.safetensors": 1_000_000,
    "adapter_config.json": 1,
    "README.md": 1,
    "timbre_memory_adapter.pt": 1_000_000,
    "timbre_memory_config.json": 1,
}
errors = []
if checkpoint.name != f"step-{step}":
    errors.append(f"checkpoint name={checkpoint.name!r}")
for name, minimum_size in required.items():
    path = checkpoint / name
    if not path.is_file() or path.stat().st_size < minimum_size:
        errors.append(f"missing/small checkpoint file: {path}")
for name in ("adapter_config.json", "timbre_memory_config.json"):
    path = checkpoint / name
    if path.is_file():
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - audit message preserves the parser error
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

identity_root = (
    project_root
    / "trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713"
)
args_path = identity_root / arm / "train_args_dry_run_core.json"
if not args_path.is_file():
    errors.append(f"missing run identity: {args_path}")
else:
    expected_identity_sha = {
        "r3": "3637b90fe14bbda7c4915362d5bd726b072833f07853df4fc555fde0d2c32d7b",
        "r5": "161f05b766454ce7d3e38af1772a1126ec612eaeb0682678bd6a46ca1d62daff",
    }[arm]
    canonical_root = Path(
        "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
    )
    if project_root.resolve() == canonical_root:
        with args_path.open("rb") as handle:
            actual_identity_sha = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual_identity_sha != expected_identity_sha:
            errors.append(
                f"run identity SHA256={actual_identity_sha}, expected {expected_identity_sha}"
            )
    args = json.loads(args_path.read_text(encoding="utf-8"))
    no_text = project_root / "trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/no_text.train.jsonl"
    text = project_root / "trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/text.train.jsonl"
    expected_spec = f"{no_text}::repeat=1,{text}::repeat={repeat}"
    expected_args = {
        "OUT_DIR": str(run_dir),
        "TRAIN_JSONL_SPEC": expected_spec,
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
        got = args.get(key)
        ok = got == wanted
        if not ok:
            errors.append(f"run identity {key}={got!r}, expected {wanted!r}")
if errors:
    raise SystemExit(f"Batch-44 {arm} step-{step} checkpoint audit failed:\n- " + "\n- ".join(errors))
print(f"[batch44-quick20-checkpoint] PASS arm={arm} repeat={repeat} path={checkpoint}")
PY
}

ensure_evaluation_is_new() {
  local completion_artifact
  for completion_artifact in \
    "$COMPLETION_JSON" "$COMPLETE_MARKER" \
    "$RECORD_ROOT/metrics.json" "$RECORD_ROOT/metrics.tsv" "$RECORD_ROOT/metrics.md"; do
    [ ! -e "$completion_artifact" ] || \
      die "step-$STEP has existing/partial completion evidence; inspect before recovery: $completion_artifact"
  done
  if [ "$ENTRYPOINT" != "1" ] && [ "$DRY_RUN" = "0" ]; then
    if [ -s "$RECORD_ROOT/submitted_jobs.tsv" ] && grep -Eq 'job-[0-9a-fA-F-]{36}' "$RECORD_ROOT/submitted_jobs.tsv"; then
      die "step-$STEP quick20 already has a live job ledger: $RECORD_ROOT/submitted_jobs.tsv"
    fi
    if [ -d "$RECORD_ROOT/.live_submit.lock" ]; then
      die "step-$STEP has a persistent live-submit lock; inspect before recovery: $RECORD_ROOT/.live_submit.lock"
    fi
  fi
  local arm mode run_id output_dir
  for arm in r3 r5; do
    for mode in no_text text; do
      run_id="$(run_id_for "$arm" "$mode")"
      output_dir="$EVAL_ROOT/$run_id"
      if [ -s "$output_dir/${run_id}.summary.json" ] || compgen -G "$output_dir/manifest.shard*.jsonl" >/dev/null; then
        die "partial/existing eval output requires inspection, not overwrite: $output_dir"
      fi
    done
  done
}

audit_runtime_gpus() {
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable in QZ entrypoint"
  local gpu_count
  gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l | tr -d ' ')
  [ "$gpu_count" = "8" ] || die "QZ entrypoint must expose exactly 8 GPUs, got $gpu_count"
  if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -Evq 'H200'; then
    die "QZ entrypoint contains a non-H200 GPU"
  fi
  echo "[batch44-quick20-gpu] PASS count=8 type=H200"
}

run_eval() {
  local arm="$1"
  local mode="$2"
  local gpu_pair="$3"
  local checkpoint validation_jsonl run_id output_dir log
  checkpoint="$(arm_checkpoint "$arm")"
  case "$mode" in
    no_text) validation_jsonl="$NO_TEXT20_JSONL" ;;
    text) validation_jsonl="$TEXT20_JSONL" ;;
    *) die "unsupported quick20 mode: $mode" ;;
  esac
  run_id="$(run_id_for "$arm" "$mode")"
  output_dir="$EVAL_ROOT/$run_id"
  log="$RECORD_ROOT/eval_${arm}_${mode}_step${STEP}.log"

  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu_pair"
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=8
    export SPEAKER_ENCODER_TYPE=embedding_loader
    echo "[batch44-quick20] arm=$arm mode=$mode gpu_pair=$gpu_pair checkpoint=$checkpoint run_id=$run_id"

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
    REF_AUDIO_CFG_SCALE=1.0 \
    HF_MODULES_CACHE_ROOT="$output_dir/.hf_modules_cache" \
    INFER_SHARD_START_DELAY_SEC=3 \
    PYTHON="$PYTHON" \
    ASR_PYTHON="$ASR_PYTHON" \
    VALIDATION_JSONL="$validation_jsonl" \
    MODEL_PATH="$checkpoint" \
    RUN_ID="$run_id" \
    RUN_LABEL="$run_id" \
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
    GPU_COUNT=2 \
    NUM_SHARDS=2 \
    ASR_NUM_SHARDS=2 \
    SEED=1234 \
    bash "$CODE_ROOT/scripts/004039_run_seedtts_validation_eval.sh"

    "$PYTHON" "$CODE_ROOT/scripts/004056_summarize_seedtts_ref_content_similarity.py" \
      --asr-jsonl "$output_dir/${run_id}.asr_eval.jsonl" \
      --output-json "$output_dir/${run_id}.ref_content_similarity_summary.json" \
      --output-md "$output_dir/${run_id}.ref_content_similarity_summary.md"

    HF_MODULES_CACHE="$output_dir/.hf_modules_cache/speaker_summary" \
    "$PYTHON" "$CODE_ROOT/scripts/004050_summarize_seedtts_speaker_sim_only.py" \
      --validation-jsonl "$validation_jsonl" \
      --run "$run_id=$output_dir" \
      --output-csv "$output_dir/${run_id}.speaker_sim.csv" \
      --summary-json "$output_dir/${run_id}.speaker_sim_summary.json" \
      --summary-md "$output_dir/${run_id}.speaker_sim_summary.md" \
      --speaker-device cuda:0

    echo "[batch44-quick20] arm=$arm mode=$mode complete output=$output_dir"
  ) > >(tee -a "$log") 2>&1
}

collect_metrics() {
  "$PYTHON" - "$EVAL_ROOT" "$RECORD_ROOT" "$STEP" \
    "$(arm_label r3)" "$(arm_label r5)" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

eval_root = Path(sys.argv[1])
record_root = Path(sys.argv[2])
step = int(sys.argv[3])
labels = {"r3": sys.argv[4], "r5": sys.argv[5]}
train_jobs = {
    "r3": "job-2b91d332-d500-4279-84f9-0a6a81a376aa",
    "r5": "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c",
}
rows_out = []
for arm in ("r3", "r5"):
    for mode in ("no_text", "text"):
        run_id = f"{labels[arm]}_step-{step}_{mode}_quick20_d2d3_seed1234"
        out_dir = eval_root / run_id
        summary_path = out_dir / f"{run_id}.summary.json"
        speaker_path = out_dir / f"{run_id}.speaker_sim.csv"
        ref_content_path = out_dir / f"{run_id}.ref_content_similarity_summary.json"
        asr_path = out_dir / f"{run_id}.asr_eval.jsonl"
        for path in (summary_path, speaker_path, ref_content_path, asr_path):
            if not path.is_file():
                raise SystemExit(f"missing metric input: {path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))["overall"]
        speaker_rows = list(csv.DictReader(speaker_path.open(encoding="utf-8")))
        valid_speaker = [
            row for row in speaker_rows
            if row.get("status") in {"ok", "ok_after_rerun", "skipped_exists"}
        ]
        asr_rows = [json.loads(line) for line in asr_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        n = int(summary["n"])
        keep = int(summary["keep"])
        if n != 20 or len(valid_speaker) != 20 or len(asr_rows) != 20:
            raise SystemExit(
                f"incomplete quick20 run={run_id}: summary={n} speaker={len(valid_speaker)} asr={len(asr_rows)}"
            )
        case_ids = [str(row.get("case_id") or row.get("sample_id") or "") for row in asr_rows]
        if len(set(case_ids)) != 20 or any(not case_id for case_id in case_ids):
            raise SystemExit(f"duplicate/empty ASR case IDs: {run_id}")
        speaker_case_ids = [str(row.get("case_id") or "") for row in valid_speaker]
        if len(set(speaker_case_ids)) != 20 or any(not case_id for case_id in speaker_case_ids):
            raise SystemExit(f"duplicate/empty speaker case IDs: {run_id}")
        if set(speaker_case_ids) != set(case_ids):
            only_speaker = sorted(set(speaker_case_ids) - set(case_ids))
            only_asr = sorted(set(case_ids) - set(speaker_case_ids))
            raise SystemExit(
                f"speaker/ASR case-id set mismatch: {run_id}: "
                f"only_speaker={only_speaker[:5]} only_asr={only_asr[:5]}"
            )
        wrong_modes = [row.get("case_id") for row in asr_rows if row.get("mode") != mode]
        if wrong_modes:
            raise SystemExit(f"wrong ASR modes in {run_id}: {wrong_modes}")
        sim_ref = sum(float(row["sim_gen_ref"]) for row in valid_speaker) / 20
        sim_src = sum(float(row["sim_gen_source"]) for row in valid_speaker) / 20
        ref_bound_count = sum(
            float(row["sim_gen_ref"]) - float(row["sim_gen_source"]) > 0.05
            for row in valid_speaker
        )
        ref_content = json.loads(ref_content_path.read_text(encoding="utf-8"))["overall"]
        en_src_rows = (
            [row for row in asr_rows if str(row.get("cell") or "").startswith("en_src_")]
            if mode == "text"
            else []
        )
        if mode == "text" and len(en_src_rows) != 12:
            raise SystemExit(f"text en_src quick proxy must have n=12, got {len(en_src_rows)}: {run_id}")
        en_src_fail = (
            sum(row.get("content_keep") is not True for row in en_src_rows) / len(en_src_rows)
            if en_src_rows else None
        )
        row_out = {
            "step": step,
            "arm": arm,
            "train_job_id": train_jobs[arm],
            "mode": mode,
            "n": n,
            "keep": keep,
            "fail": (n - keep) / n,
            "cer": float(summary["cer"]),
            "sim_ref": sim_ref,
            "sim_src": sim_src,
            "margin": sim_ref - sim_src,
            "ref_bound_count": ref_bound_count,
            "ref_bound": ref_bound_count / 20,
            "ref_content_f1": float(ref_content["ref_content_lcs_f1_mean"]),
            "text_en_src_quick_n": len(en_src_rows) if en_src_rows else "",
            "text_en_src_quick_fail": en_src_fail if en_src_fail is not None else "",
            "text_en_src_scope": "quick20 proxy n=12; not the full text en_src n=80 gate" if en_src_rows else "",
            "run_id": run_id,
            "output_dir": str(out_dir),
        }
        rows_out.append(row_out)

record_root.mkdir(parents=True, exist_ok=True)
(record_root / "metrics.json").write_text(
    json.dumps(rows_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
fields = list(rows_out[0])
with (record_root / "metrics.tsv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows_out)

lines = [
    f"# Batch-44 r3/r5 quick20 step-{step}",
    "",
    "Training provenance: r3 `job-2b91d332-d500-4279-84f9-0a6a81a376aa`; r5 `job-b8eb2f1f-a3eb-483b-a289-b4cce281525c`.",
    "",
    "text en_src fail is a fixed **12-case quick20 proxy**, not the full 80-case gate.",
    "",
    "| Arm | Mode | fail | CER | sim(ref) | sim(src) | margin | ref-bound | F1(ref-content) | text en_src quick fail |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows_out:
    en_src = "—" if row["text_en_src_quick_fail"] == "" else f"{row['text_en_src_quick_fail']:.1%} (n=12)"
    lines.append(
        f"| {row['arm']} | {row['mode']} | {row['fail']:.1%} | {row['cer']:.4f} | "
        f"{row['sim_ref']:.4f} | {row['sim_src']:.4f} | {row['margin']:.4f} | "
        f"{row['ref_bound']:.1%} | {row['ref_content_f1']:.4f} | {en_src} |"
    )
lines.extend([
    "",
    "## r5 - r3",
    "",
    "| Mode | Δfail | ΔCER | Δsim(ref) | Δsim(src) | Δmargin | Δref-bound | ΔF1 | Δtext en_src quick fail |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
])
indexed = {(row["arm"], row["mode"]): row for row in rows_out}
for mode in ("no_text", "text"):
    a = indexed[("r3", mode)]
    b = indexed[("r5", mode)]
    if mode == "text":
        en_delta = f"{b['text_en_src_quick_fail'] - a['text_en_src_quick_fail']:+.1%}"
    else:
        en_delta = "—"
    lines.append(
        f"| {mode} | {b['fail'] - a['fail']:+.1%} | {b['cer'] - a['cer']:+.4f} | "
        f"{b['sim_ref'] - a['sim_ref']:+.4f} | {b['sim_src'] - a['sim_src']:+.4f} | "
        f"{b['margin'] - a['margin']:+.4f} | {b['ref_bound'] - a['ref_bound']:+.1%} | "
        f"{b['ref_content_f1'] - a['ref_content_f1']:+.4f} | {en_delta} |"
    )
(record_root / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
for row in rows_out:
    en_src = "n/a" if row["text_en_src_quick_fail"] == "" else f"{row['text_en_src_quick_fail']:.1%}/n12"
    print(
        "[batch44-quick20-metric] "
        f"step={step} arm={row['arm']} mode={row['mode']} fail={row['fail']:.1%} "
        f"CER={row['cer']:.4f} sim_ref={row['sim_ref']:.4f} sim_src={row['sim_src']:.4f} "
        f"margin={row['margin']:.4f} ref_bound={row['ref_bound']:.1%} "
        f"F1={row['ref_content_f1']:.4f} text_en_src_quick={en_src}"
    )
PY
}

finalize_completion() {
  "$PYTHON" - \
    "$RECORD_ROOT" "$EVAL_ROOT" "$STEP" \
    "$(arm_label r3)" "$(arm_label r5)" \
    "$R3_TRAIN_JOB_ID" "$R5_TRAIN_JOB_ID" \
    "$(arm_checkpoint r3)" "$(arm_checkpoint r5)" \
    "$NO_TEXT20_JSONL" "$NO_TEXT20_SHA256" \
    "$TEXT_SOURCE_JSONL" "$TEXT_SOURCE_SHA256" \
    "$TEXT20_JSONL" "$TEXT20_SHA256" \
    "$FROZEN_RUNNER_PATH" "$CODE_ROOT" \
    "$ALLOWED_COMPUTE_GROUP" "$ALLOWED_SPEC" "$JOB_NAME" <<'PY'
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path

record_root = Path(sys.argv[1]).resolve()
eval_root = Path(sys.argv[2]).resolve()
step = int(sys.argv[3])
labels = {"r3": sys.argv[4], "r5": sys.argv[5]}
training_jobs = {"r3": sys.argv[6], "r5": sys.argv[7]}
checkpoints = {"r3": Path(sys.argv[8]).resolve(), "r5": Path(sys.argv[9]).resolve()}
fixed_input_specs = {
    "no_text20": (Path(sys.argv[10]).resolve(), sys.argv[11]),
    "text_source": (Path(sys.argv[12]).resolve(), sys.argv[13]),
    "text20": (Path(sys.argv[14]).resolve(), sys.argv[15]),
}
frozen_runner = Path(sys.argv[16]).resolve()
code_root = Path(sys.argv[17]).resolve()
compute_group, spec, job_name = sys.argv[18:21]
completion_path = record_root / "COMPLETED.json"
marker_path = record_root / "complete.marker"
metrics_paths = {
    "json": record_root / "metrics.json",
    "tsv": record_root / "metrics.tsv",
    "md": record_root / "metrics.md",
}
ledger_path = record_root / "submitted_jobs.tsv"
job_re = re.compile(
    r"^job-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def sha256(path: Path) -> str:
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing/empty completion input: {path}")
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256(path),
    }


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


for arm, checkpoint in checkpoints.items():
    if not checkpoint.is_dir() or checkpoint.name != f"step-{step}":
        raise SystemExit(f"invalid {arm} checkpoint binding: {checkpoint}")

if frozen_runner != (record_root / "004110_submit_batch44_v1_quick20_qz.frozen.sh").resolve():
    raise SystemExit(f"unexpected frozen runner path: {frozen_runner}")
frozen_runner_artifact = artifact(frozen_runner)

fixed_inputs: dict[str, dict[str, object]] = {}
for name, (path, expected_sha) in fixed_input_specs.items():
    actual_sha = sha256(path)
    if actual_sha != expected_sha:
        raise SystemExit(f"fixed input SHA drift: {name}={actual_sha}, expected {expected_sha}")
    fixed_inputs[name] = {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": actual_sha,
    }

metrics_json = json.loads(metrics_paths["json"].read_text(encoding="utf-8"))
if not isinstance(metrics_json, list) or len(metrics_json) != 4:
    raise SystemExit("metrics.json must contain exactly four rows")
with metrics_paths["tsv"].open(encoding="utf-8", newline="") as handle:
    metrics_tsv = list(csv.DictReader(handle, delimiter="\t"))
if len(metrics_tsv) != 4:
    raise SystemExit("metrics.tsv must contain exactly four rows")

expected_keys = {(arm, mode) for arm in ("r3", "r5") for mode in ("no_text", "text")}
json_by_key = {}
tsv_by_key = {}
for source, rows, destination in (
    ("metrics.json", metrics_json, json_by_key),
    ("metrics.tsv", metrics_tsv, tsv_by_key),
):
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit(f"{source}: metric row must be an object")
        key = (str(row.get("arm") or ""), str(row.get("mode") or ""))
        if key not in expected_keys or key in destination:
            raise SystemExit(f"{source}: invalid/duplicate row identity {key}")
        arm, mode = key
        expected_run_id = f"{labels[arm]}_step-{step}_{mode}_quick20_d2d3_seed1234"
        if int(row.get("step", -1)) != step:
            raise SystemExit(f"{source}: {key} step drift")
        if str(row.get("train_job_id") or "") != training_jobs[arm]:
            raise SystemExit(f"{source}: {key} training-job drift")
        if str(row.get("run_id") or "") != expected_run_id:
            raise SystemExit(f"{source}: {key} run-id drift")
        destination[key] = row
if set(json_by_key) != expected_keys or set(tsv_by_key) != expected_keys:
    raise SystemExit("metric row identity set is incomplete")
for key in expected_keys:
    for field in ("step", "arm", "train_job_id", "mode", "n", "keep", "run_id", "output_dir"):
        if str(json_by_key[key].get(field, "")) != str(tsv_by_key[key].get(field, "")):
            raise SystemExit(f"metrics JSON/TSV disagree for {key}.{field}")

with ledger_path.open(encoding="utf-8", newline="") as handle:
    ledger_rows = list(csv.DictReader(handle, delimiter="\t"))
if len(ledger_rows) != 1:
    raise SystemExit(f"{ledger_path}: expected exactly one submission row")
submitted = ledger_rows[0]
expected_ledger = {
    "job_name": job_name,
    "step": str(step),
    "compute_group": compute_group,
    "spec": spec,
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
    "eval_root": eval_root,
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

runs = []
for arm in ("r3", "r5"):
    for mode in ("no_text", "text"):
        runs.append({
            "arm": arm,
            "mode": mode,
            "run_id": f"{labels[arm]}_step-{step}_{mode}_quick20_d2d3_seed1234",
            "training_job_id": training_jobs[arm],
            "checkpoint": str(checkpoints[arm]),
        })

completed_utc = dt.datetime.now(dt.timezone.utc).isoformat()
payload = {
    "schema": "moss_codecvc.batch44_v1_quick20_completion.v1",
    "status": "complete",
    "step": step,
    "completed_utc": completed_utc,
    "record_root": str(record_root),
    "eval_root": str(eval_root),
    "training_jobs": training_jobs,
    "evaluation_job": {
        "job_name": job_name,
        "job_id": evaluation_job_id,
        "submission_ledger": artifact(ledger_path),
    },
    "resource_contract": {
        "compute_group": "MTTS-3-2-0715",
        "compute_group_id": compute_group,
        "spec": spec,
        "instances": 1,
        "gpus": 8,
        "gpu_type": "NVIDIA_H200_SXM_141G",
    },
    "fixed_inputs": fixed_inputs,
    "frozen_runner": frozen_runner_artifact,
    "metrics": {name: artifact(path) for name, path in metrics_paths.items()},
    "runs": runs,
}

# A stale marker must never coexist with an in-progress completion rewrite.
marker_path.unlink(missing_ok=True)
atomic_write(completion_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
completion_sha = sha256(completion_path)
marker_payload = {
    "schema": "moss_codecvc.batch44_v1_quick20_complete_marker.v1",
    "status": "complete",
    "step": step,
    "completed_utc": completed_utc,
    "completed_json_sha256": completion_sha,
}
# The marker is deliberately the last completion artifact written.
atomic_write(marker_path, json.dumps(marker_payload, indent=2, sort_keys=True) + "\n")
print(
    f"[batch44-quick20-completion] PASS step={step} runs=4 "
    f"job={evaluation_job_id} completed_sha256={completion_sha}"
)
PY
}

run_entrypoint() {
  audit_code_root
  audit_training_pair
  prepare_and_validate_quick_sets
  validate_checkpoint r3
  validate_checkpoint r5
  ensure_evaluation_is_new
  audit_runtime_gpus
  mkdir -p "$RECORD_ROOT" "$EVAL_ROOT"
  exec > >(tee -a "$RECORD_ROOT/run.log") 2>&1
  echo "[batch44-quick20] start step=$STEP date=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
  echo "[batch44-quick20] training_jobs=r3:$R3_TRAIN_JOB_ID,r5:$R5_TRAIN_JOB_ID"
  echo "[batch44-quick20] compute_group=MTTS-3-2-0715 gpu_plan=r3-no_text:0,1+r3-text:2,3+r5-no_text:4,5+r5-text:6,7"
  nvidia-smi || true

  local pids=()
  local failed=0
  run_eval r3 no_text 0,1 & pids+=("$!")
  run_eval r3 text 2,3 & pids+=("$!")
  run_eval r5 no_text 4,5 & pids+=("$!")
  run_eval r5 text 6,7 & pids+=("$!")
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  [ "$failed" = "0" ] || die "one or more Batch-44 quick20 shards failed"

  collect_metrics
  finalize_completion
  echo "[batch44-quick20] complete step=$STEP metrics=$RECORD_ROOT/metrics.tsv"
}

if [ "$LIBRARY_MODE" = "1" ]; then
  # 004117 sources the audited data/checkpoint/evaluation functions below, but
  # owns its local-only runtime gate and completion provenance.  Returning here
  # guarantees that sourcing this file cannot reach any platform submission
  # code.
  return 0 2>/dev/null || exit 0
fi

if [ "$ENTRYPOINT" = "1" ]; then
  run_entrypoint
  exit 0
fi

[ -x "$PYTHON" ] || die "Python interpreter is not executable: $PYTHON"
[ -x "$ASR_PYTHON" ] || die "ASR Python interpreter is not executable: $ASR_PYTHON"
[ -x "$QZCLI" ] || die "qzcli-local wrapper is not executable: $QZCLI"
[ -d "$QZCLI_HOME" ] || die "qzcli-local HOME is missing: $QZCLI_HOME"
audit_code_root
audit_training_pair
prepare_and_validate_quick_sets
if [ "$COLLECT_ONLY" = "1" ]; then
  validate_checkpoint r3
  validate_checkpoint r5
  collect_metrics
  finalize_completion
  echo "[batch44-quick20] collect-only recovered step=$STEP metrics=$RECORD_ROOT/metrics.tsv"
  exit 0
fi
if [ "$STATIC_AUDIT_ONLY" = "1" ]; then
  echo "[batch44-quick20] static audit passed; checkpoints and QZ were not touched"
  exit 0
fi
validate_checkpoint r3
validate_checkpoint r5
ensure_evaluation_is_new

mkdir -p "$RECORD_ROOT"
RUNNER_SOURCE="$PROJECT_ROOT/scripts/004110_submit_batch44_v1_quick20_qz.sh"
FROZEN_RUNNER="$FROZEN_RUNNER_PATH"
[ -s "$RUNNER_SOURCE" ] || die "missing Batch-44 quick20 wrapper: $RUNNER_SOURCE"
if [ -s "$FROZEN_RUNNER" ]; then
  cmp -s "$RUNNER_SOURCE" "$FROZEN_RUNNER" || die "frozen runner drift at $FROZEN_RUNNER"
else
  cp "$RUNNER_SOURCE" "$FROZEN_RUNNER"
  chmod 0555 "$FROZEN_RUNNER"
fi
sha256sum "$FROZEN_RUNNER" > "$RECORD_ROOT/frozen_runner.sha256"

if [ "$DRY_RUN" = "0" ]; then
  [ "$R3_RUN_DIR" = "$CANONICAL_PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r3_v1_30k" ] || \
    die "live r3 run directory is not canonical: $R3_RUN_DIR"
  [ "$R5_RUN_DIR" = "$CANONICAL_PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r5_v1_30k" ] || \
    die "live r5 run directory is not canonical: $R5_RUN_DIR"
  [ "$RECORD_ROOT" = "$CANONICAL_PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_quick20_step${STEP}_${STAMP}" ] || \
    die "live record root is not canonical: $RECORD_ROOT"
  [ "$EVAL_ROOT" = "$CANONICAL_PROJECT_ROOT/testset/outputs/ver23_batch44_quick20_${STAMP}" ] || \
    die "live eval root is not canonical: $EVAL_ROOT"
  if ! mkdir "$RECORD_ROOT/.live_submit.lock" 2>/dev/null; then
    die "persistent live-submit lock already exists: $RECORD_ROOT/.live_submit.lock"
  fi
  "$PYTHON" - "$RECORD_ROOT/.live_submit.lock/owner.json" "$STEP" <<'PY'
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

COMMAND="env BATCH44_QUICK20_ENTRYPOINT=1 STEP=$STEP PROJECT_ROOT=$PROJECT_ROOT CODE_ROOT=$CODE_ROOT R3_RUN_DIR=$R3_RUN_DIR R5_RUN_DIR=$R5_RUN_DIR RECORD_ROOT=$RECORD_ROOT EVAL_ROOT=$EVAL_ROOT PYTHON=$PYTHON ASR_PYTHON=$ASR_PYTHON bash $FROZEN_RUNNER"
if [ "$DRY_RUN" = "1" ]; then
  SUBMIT_OUTPUT="$RECORD_ROOT/submit_output.dry_run.txt"
else
  SUBMIT_OUTPUT="$RECORD_ROOT/submit_output.txt"
fi

echo "=========================================="
echo "QZ submit: Batch-44 paired quick20"
echo "  JOB_NAME=$JOB_NAME"
echo "  STEP=$STEP"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP (MTTS-3-2-0715 only)"
echo "  SPEC=$SPEC (8x $ALLOWED_GPU_TYPE) INSTANCES=$INSTANCES"
echo "  R3_CHECKPOINT=$(arm_checkpoint r3)"
echo "  R5_CHECKPOINT=$(arm_checkpoint r5)"
echo "  TRAINING_JOBS=r3:$R3_TRAIN_JOB_ID r5:$R5_TRAIN_JOB_ID"
echo "  GPU_PLAN=r3-no_text:0,1 r3-text:2,3 r5-no_text:4,5 r5-text:6,7"
echo "  QUICK_SET=no_text20 + text20; text_en_src proxy n=12"
echo "  RECORD_ROOT=$RECORD_ROOT"
echo "  EVAL_ROOT=$EVAL_ROOT"
echo "  CODE_ROOT=$CODE_ROOT"
echo "  DRY_RUN=$DRY_RUN CONFIRM_BATCH44_QUICK20=$CONFIRM_BATCH44_QUICK20"
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
  echo "[batch44-quick20] platform dry-run passed; no job submitted"
  exit 0
fi

"$PYTHON" - "$SUBMIT_OUTPUT" "$RECORD_ROOT/submitted_jobs.tsv" \
  "$JOB_NAME" "$STEP" "$COMPUTE_GROUP" "$SPEC" "$RECORD_ROOT" "$EVAL_ROOT" "$CODE_ROOT" <<'PY'
import re
import sys
from pathlib import Path

submit_output = Path(sys.argv[1])
ledger = Path(sys.argv[2])
job_name, step, compute_group, spec, record_root, eval_root, code_root = sys.argv[3:10]
pattern = re.compile(
    r"job-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
job_ids = sorted(set(pattern.findall(submit_output.read_text(encoding="utf-8", errors="replace"))))
if len(job_ids) != 1:
    raise SystemExit(f"expected exactly one complete QZ job UUID, got {job_ids}")
fields = ["job_name", "job_id", "step", "compute_group", "spec", "record_root", "eval_root", "code_root"]
values = [job_name, job_ids[0], step, compute_group, spec, record_root, eval_root, code_root]
ledger.write_text("\t".join(fields) + "\n" + "\t".join(values) + "\n", encoding="utf-8")
print(f"[batch44-quick20] submitted job_id={job_ids[0]} ledger={ledger}")
PY

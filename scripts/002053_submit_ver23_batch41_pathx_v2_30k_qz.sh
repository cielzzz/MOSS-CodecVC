#!/bin/sh
# Batch-41 Track 2: Path X data-v2 30k.  Live submission is impossible until
# a machine-readable step-3000 pilot gate proves all four registered criteria.

set -eu

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
FROZEN_CODE_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch3436_20260710_1092820"
CODE_ROOT="${CODE_ROOT:-$FROZEN_CODE_ROOT}"
STAMP="${STAMP:-20260711}"
DRY_RUN="${DRY_RUN:-1}"

ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122" # MTTS-3-2-0715
ALLOWED_SPEC="67b10bc6-78b0-41a3-aaf4-358eeeb99009"
ALLOWED_INSTANCES="1"
ALLOWED_ACCELERATE_CONFIG="configs/accelerate_fsdp_h200_8gpu_no_ckpt.yaml"
COMPUTE_GROUP="${COMPUTE_GROUP:-$ALLOWED_COMPUTE_GROUP}"
SPEC="${SPEC:-$ALLOWED_SPEC}"
INSTANCES="${INSTANCES:-$ALLOWED_INSTANCES}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$ALLOWED_ACCELERATE_CONFIG}"
QZCLI_GPU_TYPE_OVERRIDE="${QZCLI_GPU_TYPE_OVERRIDE:-NVIDIA_H200_SXM_141G}"

JOB_NAME="ver23_pathX_v2_30k_final"
BATCH_ID="ver23_content_side_batch41_pathX_v2_30k_final_${STAMP}"
RECORD_ROOT="${RECORD_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/$BATCH_ID}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs/lora_runs/$BATCH_ID}"

# User-confirmed data-v2 route: source=u1-prime (Seed-VC perturbed),
# timbre_ref=u2 (random ref-side channel decorrelation), target=u1 (real audio).
V2_DIR="$PROJECT_ROOT/trainset/ver2_9_prepared_v2_real_no_text_refdecorr_wavlm_sv_20260708"
NO_TEXT_TRAIN_JSONL="$V2_DIR/no_text.v2.train.jsonl"
TEXT_TRAIN_JSONL="$V2_DIR/text.train.jsonl"
NO_TEXT_EXPECTED_BYTES="23967514378"
NO_TEXT_EXPECTED_SHA256="de2e6ca854c8054445739ea831641b0f138893f2ec9ba8dbfd7b0a5760dda5eb"
TEXT_REPEAT=3
TRAIN_JSONL_SPEC="$NO_TEXT_TRAIN_JSONL::repeat=1,$TEXT_TRAIN_JSONL::repeat=$TEXT_REPEAT"

# Full hashing reads 23.97 GB.  It is mandatory immediately before a live
# submission; dry-runs still enforce the exact canonical path, summary and size.
if [ "$DRY_RUN" = "0" ]; then
  VERIFY_V2_FULL_SHA256="${VERIFY_V2_FULL_SHA256:-1}"
else
  VERIFY_V2_FULL_SHA256="${VERIFY_V2_FULL_SHA256:-0}"
fi

PILOT_JOB_ID="job-04c05174-9a20-4074-add4-2655293452ed"
PILOT_GATE_JSON="${PILOT_GATE_JSON:-$PROJECT_ROOT/testset/outputs/ver23_batch41_b2_mixed_3k_probe_20260711/pilot_gate.json}"

case "$DRY_RUN" in
  0|1) ;;
  *) echo "ERROR: DRY_RUN must be 0 or 1, got $DRY_RUN" >&2; exit 2 ;;
esac
case "$VERIFY_V2_FULL_SHA256" in
  0|1) ;;
  *) echo "ERROR: VERIFY_V2_FULL_SHA256 must be 0 or 1" >&2; exit 2 ;;
esac
if [ "$DRY_RUN" = "0" ] && [ "$VERIFY_V2_FULL_SHA256" != "1" ]; then
  echo "ERROR: live Batch-41 30k submission requires full v2 SHA256 verification" >&2
  exit 2
fi
if [ "$COMPUTE_GROUP" != "$ALLOWED_COMPUTE_GROUP" ]; then
  echo "ERROR: only MTTS-3-2-0715 is allowed; got $COMPUTE_GROUP" >&2
  exit 2
fi
if [ "$SPEC" != "$ALLOWED_SPEC" ] || [ "$INSTANCES" != "$ALLOWED_INSTANCES" ]; then
  echo "ERROR: Batch-41 30k requires spec=$ALLOWED_SPEC and instances=1" >&2
  exit 2
fi
if [ "$ACCELERATE_CONFIG" != "$ALLOWED_ACCELERATE_CONFIG" ]; then
  echo "ERROR: Batch-41 30k requires $ALLOWED_ACCELERATE_CONFIG" >&2
  exit 2
fi
if [ "$QZCLI_GPU_TYPE_OVERRIDE" != "NVIDIA_H200_SXM_141G" ]; then
  echo "ERROR: Batch-41 30k requires NVIDIA_H200_SXM_141G" >&2
  exit 2
fi
if [ ! -x "$CODE_ROOT/scripts/002049_submit_ver23_content_side_3k_qz.sh" ]; then
  echo "ERROR: missing Path X submit wrapper under CODE_ROOT=$CODE_ROOT" >&2
  exit 1
fi

mkdir -p "$RECORD_ROOT"

audit_v2_inputs() {
  python - \
    "$V2_DIR/summary.json" "$NO_TEXT_TRAIN_JSONL" "$TEXT_TRAIN_JSONL" \
    "$NO_TEXT_EXPECTED_BYTES" "$NO_TEXT_EXPECTED_SHA256" \
    "$VERIFY_V2_FULL_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

summary_path, no_text_path, text_path = map(Path, sys.argv[1:4])
expected_no_text_bytes = int(sys.argv[4])
expected_no_text_sha256 = sys.argv[5]
verify_full_sha256 = sys.argv[6] == "1"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
no_text = summary["splits"]["no_text.v2.train.jsonl"]
text = summary["splits"]["text.train.jsonl"]
errors = []
if summary.get("status") != "complete":
    errors.append(f"summary status={summary.get('status')!r}")
if no_text.get("rows") != 295632:
    errors.append(f"no_text rows={no_text.get('rows')!r}")
if no_text.get("missing") != {} or no_text.get("ref_content_leaks") != 0:
    errors.append("no_text has missing fields or ref-content leaks")
if no_text.get("language_counts") != {"en": 146940, "zh": 148692}:
    errors.append(f"unexpected language counts={no_text.get('language_counts')!r}")
expected_profiles = {
    "near_flat": 192276,
    "mild_eq": 65119,
    "room_eq": 20515,
    "codec_eq": 13285,
    "phone_band": 4437,
}
if no_text.get("ref_channel_profile_counts") != expected_profiles:
    errors.append(f"unexpected channel profiles={no_text.get('ref_channel_profile_counts')!r}")
if text.get("rows") != 32419 or text.get("missing") != {}:
    errors.append(f"unexpected text summary={text!r}")
if Path(no_text.get("path", "")).resolve() != no_text_path.resolve():
    errors.append("summary no_text path mismatch")
if Path(text.get("path", "")).resolve() != text_path.resolve():
    errors.append("summary text path mismatch")
for path in (no_text_path, text_path):
    if not path.is_file() or path.stat().st_size <= 0:
        errors.append(f"missing input {path}")
if no_text_path.is_file() and no_text_path.stat().st_size != expected_no_text_bytes:
    errors.append(
        f"no_text size mismatch: expected {expected_no_text_bytes}, "
        f"got {no_text_path.stat().st_size}"
    )
actual_no_text_sha256 = None
if verify_full_sha256 and no_text_path.is_file():
    with no_text_path.open("rb") as handle:
        actual_no_text_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual_no_text_sha256 != expected_no_text_sha256:
        errors.append(
            "no_text SHA256 mismatch: "
            f"expected {expected_no_text_sha256}, got {actual_no_text_sha256}"
        )
with no_text_path.open(encoding="utf-8") as handle:
    row = json.loads(handle.readline())
if row.get("moss_codecvc_mode") != "no_text":
    errors.append("first v2 row is not no_text")
roles = row.get("v2_real_target") or {}
if roles.get("target_is_real_audio") is not True or roles.get("source_is_seedvc_output") is not True:
    errors.append(f"unexpected u1/u1-prime roles={roles!r}")
if not row.get("timbre_ref_channel_augmented"):
    errors.append("first v2 row lacks ref-side channel decorrelation")
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
    if row.get(key) in (None, "", []):
        errors.append(f"first v2 row missing {key}")
if errors:
    raise SystemExit("Batch-41 v2 input audit failed:\n- " + "\n- ".join(errors))
effective_text = 32419 * 3
effective_total = 295632 + effective_text
print(
    "[batch41-v2-input-audit] PASS "
    f"no_text=295632 text=32419 repeat=3 total={effective_total} "
    f"text_share={effective_text/effective_total:.4%} "
    f"bytes={expected_no_text_bytes} "
    f"sha256={'verified:' + actual_no_text_sha256 if actual_no_text_sha256 else 'registered:' + expected_no_text_sha256}"
)
PY
}

audit_pilot_gate() {
  if [ ! -s "$PILOT_GATE_JSON" ]; then
    echo "ERROR: pilot gate is absent: $PILOT_GATE_JSON" >&2
    echo "Run and evaluate the complete step-3000 pilot before any live 30k submission." >&2
    exit 3
  fi
  python - "$PILOT_GATE_JSON" "$PILOT_JOB_ID" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_job = sys.argv[2]
gate = json.loads(path.read_text(encoding="utf-8"))
checks = {
    "decision=pass": gate.get("decision") == "pass",
    "pilot_job_id": gate.get("pilot_job_id") == expected_job,
    "step=3000": int(gate.get("checkpoint_step", -1)) == 3000,
    "text_repeat=3": int(gate.get("text_repeat", -1)) == 3,
    "no_text_cer<0.12": float(gate.get("no_text_cer", 999)) < 0.12,
    "text_en_src_fail<0.15": float(gate.get("text_en_src_fail", 999)) < 0.15,
    "text_cer<0.06": float(gate.get("text_cer", 999)) < 0.06,
    "wavlm_sim_ref>=0.42": float(gate.get("wavlm_sim_ref", -999)) >= 0.42,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"pilot gate rejected ({path}): {failed}; payload={gate}")
print(f"[batch41-pilot-gate] PASS path={path}")
PY
}

run_path_x_wrapper() {
  requested_dry_run="$1"
  (
    export ROOT="$CODE_ROOT"
    export DRY_RUN="$requested_dry_run"
    export COMPUTE_GROUP="$COMPUTE_GROUP"
    export SPEC="$SPEC"
    export INSTANCES="$INSTANCES"
    export ACCELERATE_CONFIG="$ACCELERATE_CONFIG"
    export QZCLI_GPU_TYPE_OVERRIDE="$QZCLI_GPU_TYPE_OVERRIDE"

    export TRAINSET_DIR="$V2_DIR"
    export NO_TEXT_TRAIN_JSONL="$NO_TEXT_TRAIN_JSONL"
    export TEXT_TRAIN_JSONL="$TEXT_TRAIN_JSONL"
    export TEXT_REPEAT=3
    export TRAIN_JSONL_SPEC="$TRAIN_JSONL_SPEC"
    export BATCH_ID="$BATCH_ID"
    export JOB_NAME="$JOB_NAME"
    export JOB_NAME_PREFIX="$JOB_NAME"
    export QZ_RECORD_ROOT="$RECORD_ROOT"
    export OUT_DIR="$OUT_DIR"

    export NUM_EPOCHS=6
    export MAX_TRAIN_STEPS=30000
    export SAVE_STEPS=2000
    export EVAL_STEPS=2000
    export EVAL_MAX_BATCHES=0
    export LEARNING_RATE=1e-5
    export LR_SCHEDULER_TYPE=constant_with_warmup
    export WARMUP_RATIO=0.03
    export WEIGHT_DECAY=0.01
    export PER_DEVICE_BATCH_SIZE=1
    export GRADIENT_ACCUMULATION_STEPS=8
    export GPU_COUNT=8
    export MIXED_PRECISION=bf16
    export GRADIENT_CHECKPOINTING=0
    export LORA_R=16
    export LORA_ALPHA=32
    export LORA_DROPOUT=0.05
    export LOGGING_STEPS=20
    export NUM_WORKERS=4
    export MAX_GRAD_NORM=1.0
    export POST_TRAIN_QUICK_EVAL=0

    export CONTENT_CROSS_ATTN_LAYERS=all
    export CONTENT_CROSS_ATTN_FEATURE_DIM=768
    export CONTENT_CROSS_ATTN_GATE_INIT=-0.5
    export CONTENT_CROSS_ATTN_OUTPUT_SCALE=0.3
    export CONTENT_CROSS_ATTN_DROPOUT=0.0
    export CONTENT_ENCODER_HIDDEN_SIZE=0
    export CONTENT_ENCODER_LAYERS=2
    export CONTENT_ENCODER_CONV_KERNEL_SIZE=7
    export CONTENT_CROSS_ATTN_LR_MULTIPLIER=1.0
    export GUIDED_ATTN_LOSS_WEIGHT=0.05
    export GUIDED_ATTN_WARMUP_STEPS=1000
    export GUIDED_ATTN_BAND_FRAMES=3
    export PHONEME_CLASSIFIER_LOSS_WEIGHT=0.02
    export CONTENT_CTC_WEIGHT=0.0
    export LORA_WARMUP_FREEZE_STEPS=0
    export REF_AUDIO_CFG_DROPOUT=0.0
    export TIMBRE_ADAPTER_GATE_LR_MULTIPLIER=1.0

    bash "$CODE_ROOT/scripts/002049_submit_ver23_content_side_3k_qz.sh"
  )
}

audit_generated_config() {
  python - "$RECORD_ROOT/train_args_dry_run_core.json" "$RECORD_ROOT/run_train_entrypoint.sh" "$TRAIN_JSONL_SPEC" "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

core_path, runner_path = map(Path, sys.argv[1:3])
expected_spec, out_dir = sys.argv[3:]
payload = json.loads(core_path.read_text(encoding="utf-8"))
required = {
    "TRAIN_JSONL_SPEC": expected_spec,
    "TEXT_REPEAT": "3",
    "OUT_DIR": out_dir,
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
    "ENABLE_SPEAKER_SIDE_PATHWAY": "0",
    "ENABLE_SPEAKER_CROSS_ATTN": "0",
    "ENABLE_SOURCE_SEMANTIC_MEMORY": "0",
    "TARGET_FRONT_CE_WEIGHT": "4.0",
    "TARGET_FRONT_CE_SECONDS": "0.75",
    "PROGRESS_LOSS_WEIGHT": "0.10",
    "STOP_LOSS_WEIGHT": "0.20",
    "ENABLE_CONTENT_CROSS_ATTN": "1",
    "CONTENT_CROSS_ATTN_LAYERS": "all",
    "CONTENT_CROSS_ATTN_GATE_INIT": "-0.5",
    "CONTENT_CROSS_ATTN_OUTPUT_SCALE": "0.3",
    "GUIDED_ATTN_LOSS_WEIGHT": "0.05",
    "PHONEME_CLASSIFIER_LOSS_WEIGHT": "0.02",
    "CONTENT_CTC_WEIGHT": "0.0",
    "SOURCE_CONTENT_MEMORY_TYPE": "wavlm_bnf_continuous",
}
errors = [f"{key}={payload.get(key)!r}, expected {value!r}" for key, value in required.items() if payload.get(key) != value]
if payload.get("sequence_structure") != "[text?, C_src frames, C_ref frames]":
    errors.append(f"sequence_structure={payload.get('sequence_structure')!r}")
runner = runner_path.read_text(encoding="utf-8")
for needle in (
    '[qz-train] num_epochs=6 max_train_steps=30000',
    '[qz-train] global_batch_size=64',
    '--config_file "configs/accelerate_fsdp_h200_8gpu_no_ckpt.yaml"',
    '--max-train-steps "30000"',
    '--save-steps "2000"',
):
    if needle not in runner:
        errors.append(f"runner missing {needle!r}")
if errors:
    raise SystemExit("Batch-41 v2 30k config audit failed:\n- " + "\n- ".join(errors))
print(f"[batch41-v2-config-audit] PASS core={core_path}")
PY
}

echo "=========================================="
echo "Batch-41 Path X data-v2 30k"
echo "  JOB_NAME=$JOB_NAME"
echo "  TRAIN_JSONL_SPEC=$TRAIN_JSONL_SPEC"
echo "  OUT_DIR=$OUT_DIR"
echo "  MAX_TRAIN_STEPS=30000 SAVE/EVAL=2000"
echo "  GLOBAL_BATCH_SIZE=64"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP (MTTS-3-2-0715 only)"
echo "  CODE_ROOT=$CODE_ROOT"
echo "  PILOT_GATE_JSON=$PILOT_GATE_JSON"
echo "  DRY_RUN=$DRY_RUN"
echo "=========================================="

audit_v2_inputs
run_path_x_wrapper 1
audit_generated_config

if [ "$DRY_RUN" = "1" ]; then
  echo "[batch41-v2] dry-run passed; live submission remains gated by step-3000 pilot metrics"
  exit 0
fi

audit_pilot_gate
echo "[batch41-v2] pilot gate passed; proceeding with explicit live submission"
run_path_x_wrapper 0
audit_generated_config
echo "[batch41-v2] submission record: $RECORD_ROOT/submitted_jobs.tsv"

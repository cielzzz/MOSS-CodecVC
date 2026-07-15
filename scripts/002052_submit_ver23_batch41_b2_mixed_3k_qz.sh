#!/bin/sh
# Batch-41 single-arm pilot: Path X / ver2.9.5 B2 with the old Batch-34+36
# training data and only text repeat changed from 1 to 3.
#
# Safe default (generates and audits the QZ runner, submits nothing):
#   sh scripts/002052_submit_ver23_batch41_b2_mixed_3k_qz.sh
#
# Future explicit submission (not run by this script's creation/validation):
#   DRY_RUN=0 sh scripts/002052_submit_ver23_batch41_b2_mixed_3k_qz.sh

set -eu

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
FROZEN_CODE_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch3436_20260710_1092820"
CODE_ROOT="${CODE_ROOT:-$FROZEN_CODE_ROOT}"
STAMP="${STAMP:-20260711}"
DRY_RUN="${DRY_RUN:-1}"

# Hard resource fence: Batch-41 may only use MTTS-3-2-0715, one 8xH200 node.
ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"  # MTTS-3-2-0715
ALLOWED_SPEC="67b10bc6-78b0-41a3-aaf4-358eeeb99009"
ALLOWED_INSTANCES="1"
ALLOWED_ACCELERATE_CONFIG="configs/accelerate_fsdp_h200_8gpu_no_ckpt.yaml"
COMPUTE_GROUP="${COMPUTE_GROUP:-$ALLOWED_COMPUTE_GROUP}"
SPEC="${SPEC:-$ALLOWED_SPEC}"
INSTANCES="${INSTANCES:-$ALLOWED_INSTANCES}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$ALLOWED_ACCELERATE_CONFIG}"
QZCLI_GPU_TYPE_OVERRIDE="${QZCLI_GPU_TYPE_OVERRIDE:-NVIDIA_H200_SXM_141G}"

JOB_NAME="ver23_b2_mixed_3k_probe"
BATCH_ID="ver23_content_side_batch41_b2_mixed_3k_probe_${STAMP}"
RECORD_ROOT="${RECORD_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/$BATCH_ID}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs/lora_runs/$BATCH_ID}"

# Track 1 intentionally keeps the old 310,420-row Batch-34+36 no-text split.
# The separate 295,632-row u1/u1'/u2 real-target v2 split is reserved for the
# later Track 2 30k run and must not be substituted into this pilot.
TRAINSET_DIR="$PROJECT_ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709"
NO_TEXT_TRAIN_JSONL="$TRAINSET_DIR/no_text.train.jsonl"
TEXT_TRAIN_JSONL="$TRAINSET_DIR/text.train.jsonl"
TEXT_REPEAT=3
TRAIN_JSONL_SPEC="$NO_TEXT_TRAIN_JSONL::repeat=1,$TEXT_TRAIN_JSONL::repeat=$TEXT_REPEAT"

case "$DRY_RUN" in
  0|1) ;;
  *)
    echo "ERROR: DRY_RUN must be 0 or 1, got: $DRY_RUN" >&2
    exit 2
    ;;
esac
if [ "$COMPUTE_GROUP" != "$ALLOWED_COMPUTE_GROUP" ]; then
  echo "ERROR: Batch-41 is restricted to MTTS-3-2-0715 ($ALLOWED_COMPUTE_GROUP); got $COMPUTE_GROUP" >&2
  exit 2
fi
if [ "$SPEC" != "$ALLOWED_SPEC" ]; then
  echo "ERROR: Batch-41 must use spec $ALLOWED_SPEC; got $SPEC" >&2
  exit 2
fi
if [ "$INSTANCES" != "$ALLOWED_INSTANCES" ]; then
  echo "ERROR: Batch-41 must use exactly one 8xH200 instance; got INSTANCES=$INSTANCES" >&2
  exit 2
fi
if [ "$ACCELERATE_CONFIG" != "$ALLOWED_ACCELERATE_CONFIG" ]; then
  echo "ERROR: Batch-41 must use $ALLOWED_ACCELERATE_CONFIG; got ACCELERATE_CONFIG=$ACCELERATE_CONFIG" >&2
  exit 2
fi
if [ "$QZCLI_GPU_TYPE_OVERRIDE" != "NVIDIA_H200_SXM_141G" ]; then
  echo "ERROR: Batch-41 must use NVIDIA_H200_SXM_141G; got $QZCLI_GPU_TYPE_OVERRIDE" >&2
  exit 2
fi
if [ ! -x "$CODE_ROOT/scripts/002049_submit_ver23_content_side_3k_qz.sh" ]; then
  echo "ERROR: missing executable Path X submit wrapper under CODE_ROOT: $CODE_ROOT" >&2
  exit 1
fi
for input_path in "$NO_TEXT_TRAIN_JSONL" "$TEXT_TRAIN_JSONL"; do
  if [ ! -s "$input_path" ]; then
    echo "ERROR: missing or empty Batch-41 input: $input_path" >&2
    exit 1
  fi
done

mkdir -p "$RECORD_ROOT"

audit_input_manifests() {
  python - "$TRAINSET_DIR" "$NO_TEXT_TRAIN_JSONL" "$TEXT_TRAIN_JSONL" <<'PY'
import json
import sys
from pathlib import Path

prepared_dir = Path(sys.argv[1])
no_text_path = Path(sys.argv[2])
text_path = Path(sys.argv[3])
no_text_index_path = prepared_dir / "no_text.train.jsonl.offsets.u64.json"
text_index_path = prepared_dir / "text.train.jsonl.offsets.u64.json"

no_text_index = json.loads(no_text_index_path.read_text(encoding="utf-8"))
text_index = json.loads(text_index_path.read_text(encoding="utf-8"))
if no_text_index.get("rows") != 310420 or Path(str(no_text_index.get("source_path") or "")).resolve() != no_text_path.resolve():
    raise SystemExit(f"unexpected old no-text index summary: {no_text_index}")
if text_index.get("rows") != 32419 or Path(str(text_index.get("source_path") or "")).resolve() != text_path.resolve():
    raise SystemExit(f"unexpected text index summary: {text_index}")

with no_text_path.open(encoding="utf-8") as handle:
    no_text = json.loads(handle.readline())
with text_path.open(encoding="utf-8") as handle:
    text = json.loads(handle.readline())

required_no_text = (
    "reference_audio_codes",
    "audio_codes",
    "content_token_ids",
    "source_wavlm_bnf_features_path",
    "speaker_vec_path",
    "source_speaker_embedding_path",
    "timbre_ref_speaker_embedding_path",
    "target_speaker_embedding_path",
)
missing_fields = [key for key in required_no_text if no_text.get(key) in (None, "", [])]
if missing_fields:
    raise SystemExit(f"old no-text first row lacks training fields: {missing_fields}")
if no_text.get("moss_codecvc_mode") != "no_text" or text.get("moss_codecvc_mode") != "text":
    raise SystemExit("unexpected moss_codecvc_mode in old no-text/text inputs")
for key in (
    "source_wavlm_bnf_features_path",
    "speaker_vec_path",
    "source_speaker_embedding_path",
    "timbre_ref_speaker_embedding_path",
    "target_speaker_embedding_path",
):
    path = Path(str(no_text[key]))
    if not path.is_file():
        raise SystemExit(f"old no-text first-row dependency is missing: {key}={path}")

effective_text = 32419 * 3
effective_total = 310420 + effective_text
print(
    "[batch41-input-audit] PASS "
    f"old_no_text=310420 text=32419 text_repeat=3 "
    f"effective_total={effective_total} text_share={effective_text/effective_total:.4%}"
)
PY
}

run_path_x_wrapper() {
  requested_dry_run="$1"
  (
    # Every overrideable recipe field below is assigned explicitly so caller
    # state cannot silently change the registered B2 recipe.
    export ROOT="$CODE_ROOT"
    export DRY_RUN="$requested_dry_run"
    export COMPUTE_GROUP="$COMPUTE_GROUP"
    export SPEC="$SPEC"
    export INSTANCES="$INSTANCES"
    export ACCELERATE_CONFIG="$ACCELERATE_CONFIG"
    export QZCLI_GPU_TYPE_OVERRIDE="$QZCLI_GPU_TYPE_OVERRIDE"

    export TRAINSET_DIR="$TRAINSET_DIR"
    export NO_TEXT_TRAIN_JSONL="$NO_TEXT_TRAIN_JSONL"
    export TEXT_TRAIN_JSONL="$TEXT_TRAIN_JSONL"
    export TEXT_REPEAT=3
    export TRAIN_JSONL_SPEC="$TRAIN_JSONL_SPEC"

    export BATCH_ID="$BATCH_ID"
    export JOB_NAME="$JOB_NAME"
    export JOB_NAME_PREFIX="$JOB_NAME"
    export QZ_RECORD_ROOT="$RECORD_ROOT"
    export OUT_DIR="$OUT_DIR"

    # Match the completed Batch-34+36 B2 recipe exactly except TEXT_REPEAT.
    export NUM_EPOCHS=1
    export MAX_TRAIN_STEPS=3000
    export SAVE_STEPS=500
    export EVAL_STEPS=500
    export EVAL_MAX_BATCHES=0
    export LEARNING_RATE=1e-5
    # Existing B2's recorded "constant" recipe is constant_with_warmup + 3% warmup.
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
  core_json="$RECORD_ROOT/train_args_dry_run_core.json"
  runner="$RECORD_ROOT/run_train_entrypoint.sh"
  python - "$core_json" "$runner" "$TRAIN_JSONL_SPEC" "$JOB_NAME" "$COMPUTE_GROUP" "$SPEC" "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

core_path = Path(sys.argv[1])
runner_path = Path(sys.argv[2])
expected_spec = sys.argv[3]
job_name = sys.argv[4]
compute_group = sys.argv[5]
qz_spec = sys.argv[6]
out_dir = sys.argv[7]

if not core_path.is_file():
    raise SystemExit(f"missing dry-run core JSON: {core_path}")
if not runner_path.is_file():
    raise SystemExit(f"missing generated runner: {runner_path}")

payload = json.loads(core_path.read_text(encoding="utf-8"))
required = {
    "TRAIN_JSONL_SPEC": expected_spec,
    "TEXT_REPEAT": "3",
    "OUT_DIR": out_dir,
    "MAX_TRAIN_STEPS": "3000",
    "SAVE_STEPS": "500",
    "EVAL_STEPS": "500",
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
}
errors = [
    f"{key}={payload.get(key)!r}, expected {value!r}"
    for key, value in required.items()
    if payload.get(key) != value
]

# The frozen Batch-34+36 training snapshot predates these four explicit
# knobs.  Their absence is the exact historical B2 behavior: no ref-CFG
# dropout, model-default content width, the base content-path LR, and no LoRA
# freeze.  Newer CODE_ROOTs record the equivalent explicit values.  Accept
# either representation, but reject any non-default value.
snapshot_compatible_defaults = {
    "REF_AUDIO_CFG_DROPOUT": "0.0",
    "CONTENT_ENCODER_HIDDEN_SIZE": "0",
    "CONTENT_CROSS_ATTN_LR_MULTIPLIER": "1.0",
    "LORA_WARMUP_FREEZE_STEPS": "0",
}
for key, value in snapshot_compatible_defaults.items():
    actual = payload.get(key)
    if actual not in (None, value):
        errors.append(f"{key}={actual!r}, expected absent historical default or {value!r}")

if payload.get("sequence_structure") != "[text?, C_src frames, C_ref frames]":
    errors.append(f"unexpected sequence_structure={payload.get('sequence_structure')!r}")
for flag in ("c_ref_in_sequence", "speaker_side_pathway_closed", "legacy_timbre_memory_closed", "content_cross_attn_enabled"):
    if payload.get(flag) is not True:
        errors.append(f"{flag}={payload.get(flag)!r}, expected True")

global_batch_size = int(required["PER_DEVICE_BATCH_SIZE"]) * int(required["GRADIENT_ACCUMULATION_STEPS"]) * int(required["GPU_COUNT"])
if global_batch_size != 64:
    errors.append(f"global_batch_size={global_batch_size}, expected 64")

runner = runner_path.read_text(encoding="utf-8")
runner_needles = [
    f'TRAIN_JSONL_SPEC="{expected_spec}"',
    f'OUT_DIR="{out_dir}"',
    "[qz-train] num_epochs=1 max_train_steps=3000",
    "[qz-train] lr_scheduler_type=constant_with_warmup warmup_ratio=0.03",
    "[qz-train] gpu_count=8",
    "[qz-train] global_batch_size=64",
    '--config_file "configs/accelerate_fsdp_h200_8gpu_no_ckpt.yaml"',
    '--learning-rate "1e-5"',
    '--gradient-accumulation-steps "8"',
    '--max-train-steps "3000"',
]
for needle in runner_needles:
    if needle not in runner:
        errors.append(f"runner missing expected text: {needle!r}")

if errors:
    print("[batch41-audit] FAILED", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)

summary = {
    "batch": "Batch-41",
    "job_name": job_name,
    "compute_group": compute_group,
    "compute_group_name": "MTTS-3-2-0715",
    "spec": qz_spec,
    "gpu_type": "NVIDIA_H200_SXM_141G",
    "gpu_count": 8,
    "global_batch_size": global_batch_size,
    "only_intended_recipe_delta_from_batch3436_B2": {"TEXT_REPEAT": {"from": 1, "to": 3}},
    "snapshot_compatible_implicit_defaults": snapshot_compatible_defaults,
    "effective_train_jsonl_spec": expected_spec,
    "core_config": payload,
    "runner": str(runner_path),
}
summary_path = core_path.parent / "batch41_dry_run_config_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[batch41-audit] PASS summary={summary_path}")
PY
}

echo "=========================================="
echo "Batch-41 Path X B2 mixed-ratio 3k pilot"
echo "  JOB_NAME=$JOB_NAME"
echo "  TRAIN_JSONL_SPEC=$TRAIN_JSONL_SPEC"
echo "  OUT_DIR=$OUT_DIR"
echo "  RECORD_ROOT=$RECORD_ROOT"
echo "  MAX_TRAIN_STEPS=3000"
echo "  GLOBAL_BATCH_SIZE=64 (1 x grad_accum 8 x 8 GPUs)"
echo "  LR=1e-5 scheduler=constant_with_warmup warmup_ratio=0.03"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP (MTTS-3-2-0715 only)"
echo "  SPEC=$SPEC"
echo "  INSTANCES=$INSTANCES"
echo "  ACCELERATE_CONFIG=$ACCELERATE_CONFIG"
echo "  CODE_ROOT=$CODE_ROOT"
echo "  DRY_RUN=$DRY_RUN"
echo "=========================================="

audit_input_manifests

# Always generate and audit the exact runner before any possible live submission.
run_path_x_wrapper 1
audit_generated_config

if [ "$DRY_RUN" = "1" ]; then
  echo "[batch41] dry-run passed; no QZ job submitted"
  exit 0
fi

echo "[batch41] audited config passed; proceeding with the explicit live submission"
run_path_x_wrapper 0
audit_generated_config
echo "[batch41] submission record: $RECORD_ROOT/submitted_jobs.tsv"

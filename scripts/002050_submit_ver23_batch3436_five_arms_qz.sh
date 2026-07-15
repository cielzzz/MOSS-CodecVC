#!/bin/sh
# Batch-34+36 five-arm ablation wrapper.
#
# Default mode is configuration-only dry-run. After inspecting the five generated
# runners/core JSON files, submit with:
#   DRY_RUN=0 sh scripts/002050_submit_ver23_batch3436_five_arms_qz.sh
#
# Submit a subset while preserving the registered priority order with:
#   ARMS="B1 B2" DRY_RUN=0 sh scripts/002050_submit_ver23_batch3436_five_arms_qz.sh

set -eu

PROJECT_ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
CODE_ROOT="${CODE_ROOT:-$PROJECT_ROOT}"
STAMP="${STAMP:-20260710}"
DRY_RUN="${DRY_RUN:-1}"
ARMS="${ARMS:-B1 B2 A1 B3 A2}"
ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"  # MTTS-3-2-0715
COMPUTE_GROUP="${COMPUTE_GROUP:-$ALLOWED_COMPUTE_GROUP}"
SPEC="${SPEC:-67b10bc6-78b0-41a3-aaf4-358eeeb99009}"
QZCLI_GPU_TYPE_OVERRIDE="${QZCLI_GPU_TYPE_OVERRIDE:-NVIDIA_H200_SXM_141G}"
AUDIT_ROOT="${AUDIT_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/ver23_content_side_batch3436_$STAMP}"

if [ "$COMPUTE_GROUP" != "$ALLOWED_COMPUTE_GROUP" ]; then
  echo "ERROR: Batch-34+36 is restricted to MTTS-3-2-0715 ($ALLOWED_COMPUTE_GROUP); got $COMPUTE_GROUP" >&2
  exit 2
fi

case "$DRY_RUN" in
  0|1) ;;
  *)
    echo "ERROR: DRY_RUN must be 0 or 1, got: $DRY_RUN" >&2
    exit 2
    ;;
esac

arm_selected() {
  wanted="$1"
  for selected in $ARMS; do
    if [ "$selected" = "$wanted" ]; then
      return 0
    fi
  done
  return 1
}

submit_arm() {
  tag="$1"
  arm_slug="$2"
  text_repeat="$3"
  layers="$4"
  phoneme_weight="$5"
  guided_weight="$6"
  ctc_weight="$7"

  if ! arm_selected "$tag"; then
    return 0
  fi

  batch_id="ver23_content_side_batch3436_${tag}_${arm_slug}_${STAMP}"
  job_name="ver23_batch3436_${tag}_${arm_slug}_${STAMP}"
  record_root="$AUDIT_ROOT/$tag"
  out_dir="$PROJECT_ROOT/outputs/lora_runs/$batch_id"

  echo "[batch3436] arm=$tag slug=$arm_slug dry_run=$DRY_RUN code_root=$CODE_ROOT"
  (
    unset TRAIN_JSONL_SPEC QZ_RECORD_ROOT JOB_NAME BATCH_ID OUT_DIR
    export ROOT="$CODE_ROOT"
    export DRY_RUN
    export COMPUTE_GROUP
    export SPEC
    export QZCLI_GPU_TYPE_OVERRIDE
    export TRAINSET_DIR="$PROJECT_ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709"
    export NO_TEXT_TRAIN_JSONL="$PROJECT_ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/no_text.train.jsonl"
    export TEXT_TRAIN_JSONL="$PROJECT_ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/text.train.jsonl"
    export BATCH_ID="$batch_id"
    export JOB_NAME="$job_name"
    export JOB_NAME_PREFIX="ver23_batch3436_$tag"
    export QZ_RECORD_ROOT="$record_root"
    export OUT_DIR="$out_dir"
    export TEXT_REPEAT="$text_repeat"
    export CONTENT_CROSS_ATTN_LAYERS="$layers"
    export PHONEME_CLASSIFIER_LOSS_WEIGHT="$phoneme_weight"
    export GUIDED_ATTN_LOSS_WEIGHT="$guided_weight"
    export CONTENT_CTC_WEIGHT="$ctc_weight"
    export MAX_TRAIN_STEPS=3000
    export SAVE_STEPS=500
    export EVAL_STEPS=500
    export LEARNING_RATE=1e-5
    export PER_DEVICE_BATCH_SIZE=1
    export GRADIENT_ACCUMULATION_STEPS=8
    export GPU_COUNT=8
    export POST_TRAIN_QUICK_EVAL=0
    bash "$CODE_ROOT/scripts/002049_submit_ver23_content_side_3k_qz.sh"
  )
}

mkdir -p "$AUDIT_ROOT"

# Registered fallback priority when fewer than five whole H200 nodes are free.
submit_arm B1 ver23_bnf_last16_3k 10 last_16 0.02 0.05 0.0
submit_arm B2 ver23_text_r1_3k 1 all 0.02 0.05 0.0
submit_arm A1 ver23_stronger_decouple_3k 10 all 0.05 0.10 0.0
submit_arm B3 ver23_weaker_decouple_3k 10 all 0.01 0.02 0.0
submit_arm A2 ver23_ctc_3k 10 all 0.02 0.05 0.10

if [ "$DRY_RUN" = "1" ]; then
  python - "$AUDIT_ROOT" $ARMS <<'PY'
import json
import sys
from pathlib import Path

audit_root = Path(sys.argv[1])
selected = sys.argv[2:]
expected = {
    "B1": {"TEXT_REPEAT": "10", "CONTENT_CROSS_ATTN_LAYERS": "last_16", "PHONEME_CLASSIFIER_LOSS_WEIGHT": "0.02", "GUIDED_ATTN_LOSS_WEIGHT": "0.05", "CONTENT_CTC_WEIGHT": "0.0"},
    "B2": {"TEXT_REPEAT": "1", "CONTENT_CROSS_ATTN_LAYERS": "all", "PHONEME_CLASSIFIER_LOSS_WEIGHT": "0.02", "GUIDED_ATTN_LOSS_WEIGHT": "0.05", "CONTENT_CTC_WEIGHT": "0.0"},
    "A1": {"TEXT_REPEAT": "10", "CONTENT_CROSS_ATTN_LAYERS": "all", "PHONEME_CLASSIFIER_LOSS_WEIGHT": "0.05", "GUIDED_ATTN_LOSS_WEIGHT": "0.10", "CONTENT_CTC_WEIGHT": "0.0"},
    "B3": {"TEXT_REPEAT": "10", "CONTENT_CROSS_ATTN_LAYERS": "all", "PHONEME_CLASSIFIER_LOSS_WEIGHT": "0.01", "GUIDED_ATTN_LOSS_WEIGHT": "0.02", "CONTENT_CTC_WEIGHT": "0.0"},
    "A2": {"TEXT_REPEAT": "10", "CONTENT_CROSS_ATTN_LAYERS": "all", "PHONEME_CLASSIFIER_LOSS_WEIGHT": "0.02", "GUIDED_ATTN_LOSS_WEIGHT": "0.05", "CONTENT_CTC_WEIGHT": "0.10"},
}
required_shared = {
    "MAX_TRAIN_STEPS": "3000",
    "SAVE_STEPS": "500",
    "EVAL_STEPS": "500",
    "LEARNING_RATE": "1e-5",
    "USE_TIMBRE_MEMORY": "0",
    "ENABLE_SPEAKER_SIDE_PATHWAY": "0",
    "ENABLE_SPEAKER_CROSS_ATTN": "0",
    "ENABLE_SOURCE_SEMANTIC_MEMORY": "0",
    "TARGET_FRONT_CE_WEIGHT": "4.0",
    "TARGET_FRONT_CE_SECONDS": "0.75",
    "PROGRESS_LOSS_WEIGHT": "0.10",
    "STOP_LOSS_WEIGHT": "0.20",
    "CONTENT_CROSS_ATTN_GATE_INIT": "-0.5",
    "CONTENT_CROSS_ATTN_OUTPUT_SCALE": "0.3",
}

errors = []
rows = {}
for tag in selected:
    path = audit_root / tag / "train_args_dry_run_core.json"
    if not path.is_file():
        errors.append(f"{tag}: missing {path}")
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows[tag] = payload
    for key, value in required_shared.items():
        if payload.get(key) != value:
            errors.append(f"{tag}: {key}={payload.get(key)!r}, expected {value!r}")
    for key, value in expected[tag].items():
        if payload.get(key) != value:
            errors.append(f"{tag}: {key}={payload.get(key)!r}, expected {value!r}")
    spec = payload.get("TRAIN_JSONL_SPEC", "")
    expected_repeat = f"text.train.jsonl::repeat={expected[tag]['TEXT_REPEAT']}"
    if expected_repeat not in spec:
        errors.append(f"{tag}: TRAIN_JSONL_SPEC does not contain {expected_repeat!r}: {spec!r}")

if errors:
    print("[batch3436-audit] FAILED", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)

summary_path = audit_root / "dry_run_config_summary.json"
summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[batch3436-audit] PASS arms={','.join(selected)} summary={summary_path}")
PY
else
  {
    printf 'arm\tjob_name\tjob_id\tcompute_group\trunner\tout_dir\n'
    for tag in B1 B2 A1 B3 A2; do
      if ! arm_selected "$tag"; then
        continue
      fi
      record="$AUDIT_ROOT/$tag/submitted_jobs.tsv"
      if [ -f "$record" ]; then
        tail -n 1 "$record" | awk -v arm="$tag" 'BEGIN { FS=OFS="\t" } { print arm, $1, $2, $3, $4, $5 }'
      fi
    done
  } > "$AUDIT_ROOT/submitted_jobs.tsv"
  echo "[batch3436] submission summary: $AUDIT_ROOT/submitted_jobs.tsv"
fi

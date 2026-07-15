#!/bin/sh
# Batch-37 C1/C2 two-arm probe. Default is configuration-only dry-run.
# Submit both arms only after audit passes:
#   DRY_RUN=0 sh scripts/002051_submit_ver23_batch37_two_arms_qz.sh

set -eu

PROJECT_ROOT="${ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
CODE_ROOT="${CODE_ROOT:-$PROJECT_ROOT}"
STAMP="${STAMP:-20260710_mtts}"
DRY_RUN="${DRY_RUN:-1}"
ARMS="${ARMS:-C1 C2}"
ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"  # MTTS-3-2-0715
COMPUTE_GROUP="${COMPUTE_GROUP:-$ALLOWED_COMPUTE_GROUP}"
SPEC="${SPEC:-67b10bc6-78b0-41a3-aaf4-358eeeb99009}"
QZCLI_GPU_TYPE_OVERRIDE="${QZCLI_GPU_TYPE_OVERRIDE:-NVIDIA_H200_SXM_141G}"
AUDIT_ROOT="${AUDIT_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/ver23_content_side_batch37_$STAMP}"

if [ "$COMPUTE_GROUP" != "$ALLOWED_COMPUTE_GROUP" ]; then
  echo "ERROR: Batch-37 is restricted to MTTS-3-2-0715 ($ALLOWED_COMPUTE_GROUP); got $COMPUTE_GROUP" >&2
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
  slug="$2"
  layers="$3"
  encoder_hidden_size="$4"
  content_lr_multiplier="$5"
  gate_lr_multiplier="$6"
  lora_warmup_freeze_steps="$7"
  ref_audio_cfg_dropout="$8"

  if ! arm_selected "$tag"; then
    return 0
  fi

  batch_id="ver23_content_side_batch37_${tag}_${slug}_${STAMP}"
  job_name="ver23_batch37_${tag}_${slug}_${STAMP}"
  record_root="$AUDIT_ROOT/$tag"
  out_dir="$PROJECT_ROOT/outputs/lora_runs/$batch_id"

  echo "[batch37] arm=$tag slug=$slug dry_run=$DRY_RUN code_root=$CODE_ROOT"
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
    export JOB_NAME_PREFIX="ver23_batch37_$tag"
    export QZ_RECORD_ROOT="$record_root"
    export OUT_DIR="$out_dir"
    export TEXT_REPEAT=10
    export CONTENT_CROSS_ATTN_LAYERS="$layers"
    export CONTENT_ENCODER_HIDDEN_SIZE="$encoder_hidden_size"
    export CONTENT_CROSS_ATTN_LR_MULTIPLIER="$content_lr_multiplier"
    export TIMBRE_ADAPTER_GATE_LR_MULTIPLIER="$gate_lr_multiplier"
    export LORA_WARMUP_FREEZE_STEPS="$lora_warmup_freeze_steps"
    export REF_AUDIO_CFG_DROPOUT="$ref_audio_cfg_dropout"
    export CONTENT_ENCODER_LAYERS=2
    export PHONEME_CLASSIFIER_LOSS_WEIGHT=0.02
    export GUIDED_ATTN_LOSS_WEIGHT=0.05
    export CONTENT_CTC_WEIGHT=0.0
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

submit_arm C1 ver23_compact_content_lr_warmup_3k "7,15,23,31,35" 512 5.0 10.0 500 0.0
submit_arm C2 ver23_true_ref_audio_cfg_3k all 0 1.0 1.0 0 0.15

if [ "$DRY_RUN" = "1" ]; then
  python - "$AUDIT_ROOT" $ARMS <<'PY'
import json
import sys
from pathlib import Path

audit_root = Path(sys.argv[1])
selected = sys.argv[2:]
expected = {
    "C1": {
        "CONTENT_CROSS_ATTN_LAYERS": "7,15,23,31,35",
        "CONTENT_ENCODER_HIDDEN_SIZE": "512",
        "CONTENT_CROSS_ATTN_LR_MULTIPLIER": "5.0",
        "TIMBRE_ADAPTER_GATE_LR_MULTIPLIER": "10.0",
        "LORA_WARMUP_FREEZE_STEPS": "500",
        "REF_AUDIO_CFG_DROPOUT": "0.0",
    },
    "C2": {
        "CONTENT_CROSS_ATTN_LAYERS": "all",
        "CONTENT_ENCODER_HIDDEN_SIZE": "0",
        "CONTENT_CROSS_ATTN_LR_MULTIPLIER": "1.0",
        "TIMBRE_ADAPTER_GATE_LR_MULTIPLIER": "1.0",
        "LORA_WARMUP_FREEZE_STEPS": "0",
        "REF_AUDIO_CFG_DROPOUT": "0.15",
    },
}
shared = {
    "TEXT_REPEAT": "10",
    "MAX_TRAIN_STEPS": "3000",
    "SAVE_STEPS": "500",
    "EVAL_STEPS": "500",
    "LEARNING_RATE": "1e-5",
    "TARGET_FRONT_CE_WEIGHT": "4.0",
    "TARGET_FRONT_CE_SECONDS": "0.75",
    "PROGRESS_LOSS_WEIGHT": "0.10",
    "STOP_LOSS_WEIGHT": "0.20",
    "CONTENT_CROSS_ATTN_GATE_INIT": "-0.5",
    "CONTENT_CROSS_ATTN_OUTPUT_SCALE": "0.3",
    "CONTENT_ENCODER_LAYERS": "2",
    "PHONEME_CLASSIFIER_LOSS_WEIGHT": "0.02",
    "GUIDED_ATTN_LOSS_WEIGHT": "0.05",
    "CONTENT_CTC_WEIGHT": "0.0",
    "USE_TIMBRE_MEMORY": "0",
    "ENABLE_SPEAKER_SIDE_PATHWAY": "0",
    "ENABLE_SPEAKER_CROSS_ATTN": "0",
    "ENABLE_SOURCE_SEMANTIC_MEMORY": "0",
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
    for key, value in {**shared, **expected[tag]}.items():
        if payload.get(key) != value:
            errors.append(f"{tag}: {key}={payload.get(key)!r}, expected {value!r}")
    spec = payload.get("TRAIN_JSONL_SPEC", "")
    if "no_text.train.jsonl::repeat=1" not in spec or "text.train.jsonl::repeat=10" not in spec:
        errors.append(f"{tag}: unexpected TRAIN_JSONL_SPEC={spec!r}")

if errors:
    print("[batch37-audit] FAILED", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)

summary_path = audit_root / "dry_run_config_summary.json"
summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[batch37-audit] PASS arms={','.join(selected)} summary={summary_path}")
PY
else
  {
    printf 'arm\tjob_name\tjob_id\tcompute_group\trunner\tout_dir\n'
    for tag in C1 C2; do
      if ! arm_selected "$tag"; then
        continue
      fi
      record="$AUDIT_ROOT/$tag/submitted_jobs.tsv"
      if [ -f "$record" ]; then
        tail -n 1 "$record" | awk -v arm="$tag" 'BEGIN { FS=OFS="\t" } { print arm, $1, $2, $3, $4, $5 }'
      fi
    done
  } > "$AUDIT_ROOT/submitted_jobs.tsv"
  echo "[batch37] submission summary: $AUDIT_ROOT/submitted_jobs.tsv"
fi

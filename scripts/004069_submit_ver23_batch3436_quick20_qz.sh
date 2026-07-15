#!/usr/bin/env bash
# Submit one MTTS-only job that evaluates all selected Batch-34+36 arms.
#
# The job keeps the Batch-33 comparison protocol at two inference shards and
# two ASR shards per arm. B1/B2/A1/B3 run concurrently on four GPU pairs, then
# A2 runs on one pair. This fills one 8xH200 node without changing the
# shard-count-dependent generation sequence.
#
# Dry-run:
#   STEP=500 DRY_RUN=1 bash scripts/004069_submit_ver23_batch3436_quick20_qz.sh
#
# Submit after all selected checkpoints are complete:
#   STEP=500 DRY_RUN=0 bash scripts/004069_submit_ver23_batch3436_quick20_qz.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
CODE_ROOT="${CODE_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch3436_20260710_1092820}"
QZCLI="${QZCLI:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/qzcli_with_deps.sh}"
QZCLI_HOME="${QZCLI_HOME:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/.codex/qzcli_home}"

WORKSPACE="${WORKSPACE:-ws-8207e9e2-e733-4eec-a475-cfa1c36480ba}"
PROJECT="${PROJECT:-project-c67c548f-f02c-453b-ba5b-8745db6886e7}"
ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"  # MTTS-3-2-0715
COMPUTE_GROUP="${COMPUTE_GROUP:-$ALLOWED_COMPUTE_GROUP}"
SPEC="${SPEC:-67b10bc6-78b0-41a3-aaf4-358eeeb99009}"
QZCLI_GPU_TYPE_OVERRIDE="${QZCLI_GPU_TYPE_OVERRIDE:-NVIDIA_H200_SXM_141G}"
IMAGE="${IMAGE:-docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420}"
IMAGE_TYPE="${IMAGE_TYPE:-SOURCE_PRIVATE}"
FRAMEWORK="${FRAMEWORK:-pytorch}"
INSTANCES="${INSTANCES:-1}"
SHM_GI="${SHM_GI:-1200}"
PRIORITY="${PRIORITY:-10}"

STEP="${STEP:-500}"
STAMP="${STAMP:-20260710_mtts}"
ARMS_CSV="${ARMS_CSV:-B1,B2,A1,B3,A2}"
DRY_RUN="${DRY_RUN:-1}"
FORCE="${FORCE:-0}"
ENTRYPOINT="${BATCH3436_QUICK20_ENTRYPOINT:-0}"
WAIT_FOR_CHECKPOINTS="${WAIT_FOR_CHECKPOINTS:-1}"
CHECKPOINT_WAIT_TIMEOUT="${CHECKPOINT_WAIT_TIMEOUT:-3600}"
CHECKPOINT_POLL_SECONDS="${CHECKPOINT_POLL_SECONDS:-15}"
RECORD_ROOT="${RECORD_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/ver23_content_side_batch3436_quick20_step${STEP}_$STAMP}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/testset/outputs/ver23_content_side_batch3436_quick_eval_$STAMP}"
JOB_NAME="${JOB_NAME:-ver23_batch3436_quick20_step${STEP}_$STAMP}"

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
ASR_PYTHON="${ASR_PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
NO_TEXT20_JSONL="${NO_TEXT20_JSONL:-$PROJECT_ROOT/testset/validation/ver2_8_timbre_quick20_seedtts_no_text_20260704.jsonl}"
TEXT_SOURCE_JSONL="${TEXT_SOURCE_JSONL:-$PROJECT_ROOT/testset/validation/seedtts_vc_ver2_3_validation.jsonl}"
TEXT20_JSONL="${TEXT20_JSONL:-$RECORD_ROOT/ver23_batch3436_text_quick20_8cell_20260710.jsonl}"

if [ "$COMPUTE_GROUP" != "$ALLOWED_COMPUTE_GROUP" ]; then
  echo "ERROR: quick20 is restricted to MTTS-3-2-0715 ($ALLOWED_COMPUTE_GROUP); got $COMPUTE_GROUP" >&2
  exit 2
fi
case "$STEP" in
  500|1000|1500|2000|2500|3000) ;;
  *)
    echo "ERROR: STEP must be one of 500,1000,1500,2000,2500,3000; got $STEP" >&2
    exit 2
    ;;
esac
case "$DRY_RUN:$FORCE:$ENTRYPOINT:$WAIT_FOR_CHECKPOINTS" in
  [01]:[01]:[01]:[01]) ;;
  *)
    echo "ERROR: DRY_RUN, FORCE, BATCH3436_QUICK20_ENTRYPOINT, and WAIT_FOR_CHECKPOINTS must be 0 or 1" >&2
    exit 2
    ;;
esac
if [ "$CHECKPOINT_WAIT_TIMEOUT" -le 0 ] || [ "$CHECKPOINT_POLL_SECONDS" -le 0 ]; then
  echo "ERROR: checkpoint wait timeout and poll interval must be positive" >&2
  exit 2
fi

arm_selected() {
  case ",$ARMS_CSV," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

arm_info() {
  case "$1" in
    B1) printf '%s\t%s\n' "ver23_bnf_last16_3k" "ver23_batch3436_B1_bnf_last16" ;;
    B2) printf '%s\t%s\n' "ver23_text_r1_3k" "ver23_batch3436_B2_text_r1" ;;
    A1) printf '%s\t%s\n' "ver23_stronger_decouple_3k" "ver23_batch3436_A1_stronger" ;;
    B3) printf '%s\t%s\n' "ver23_weaker_decouple_3k" "ver23_batch3436_B3_weaker" ;;
    A2) printf '%s\t%s\n' "ver23_ctc_3k" "ver23_batch3436_A2_ctc" ;;
    *)
      echo "ERROR: unsupported arm: $1" >&2
      return 2
      ;;
  esac
}

arm_checkpoint() {
  local arm="$1"
  local suffix label
  IFS=$'\t' read -r suffix label <<<"$(arm_info "$arm")"
  printf '%s\n' "$PROJECT_ROOT/outputs/lora_runs/ver23_content_side_batch3436_${arm}_${suffix}_$STAMP/step-$STEP"
}

validate_checkpoint() {
  local arm="$1"
  local checkpoint
  checkpoint="$(arm_checkpoint "$arm")"
  local required=(
    adapter_model.safetensors
    adapter_config.json
    README.md
    timbre_memory_adapter.pt
    timbre_memory_config.json
  )
  if [ ! -d "$checkpoint" ]; then
    echo "ERROR: missing checkpoint directory for $arm: $checkpoint" >&2
    return 1
  fi
  local name
  for name in "${required[@]}"; do
    if [ ! -s "$checkpoint/$name" ]; then
      echo "ERROR: missing or empty checkpoint file for $arm: $checkpoint/$name" >&2
      return 1
    fi
  done
  "$PYTHON" - "$checkpoint/adapter_config.json" "$checkpoint/timbre_memory_config.json" <<'PY'
import json
import sys
for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as handle:
        json.load(handle)
PY
}

checkpoint_ready() {
  validate_checkpoint "$1" >/dev/null 2>&1
}

wait_for_checkpoint() {
  local arm="$1"
  if [ "$WAIT_FOR_CHECKPOINTS" != "1" ]; then
    validate_checkpoint "$arm"
    return
  fi
  local start now elapsed checkpoint
  start=$(date +%s)
  checkpoint="$(arm_checkpoint "$arm")"
  while ! checkpoint_ready "$arm"; do
    now=$(date +%s)
    elapsed=$((now - start))
    if [ "$elapsed" -ge "$CHECKPOINT_WAIT_TIMEOUT" ]; then
      echo "ERROR: timed out waiting for $arm checkpoint: $checkpoint" >&2
      return 1
    fi
    echo "[batch3436-quick20] waiting arm=$arm checkpoint=$checkpoint elapsed=${elapsed}s"
    sleep "$CHECKPOINT_POLL_SECONDS"
  done
  echo "[batch3436-quick20] checkpoint ready arm=$arm path=$checkpoint"
}

prepare_text20() {
  mkdir -p "$RECORD_ROOT"
  "$PYTHON" - "$TEXT_SOURCE_JSONL" "$TEXT20_JSONL" \
    seedtts_text_en_src_zh_ref_m2f_000000 \
    seedtts_text_en_src_zh_ref_m2f_000001 \
    seedtts_text_en_src_zh_ref_m2f_000002 \
    seedtts_text_en_src_zh_ref_f2m_000000 \
    seedtts_text_en_src_zh_ref_f2m_000001 \
    seedtts_text_en_src_zh_ref_f2m_000002 \
    seedtts_text_zh_src_en_ref_m2f_000000 \
    seedtts_text_zh_src_en_ref_m2f_000001 \
    seedtts_text_zh_src_en_ref_f2m_000000 \
    seedtts_text_zh_src_en_ref_f2m_000001 \
    seedtts_text_en_src_en_ref_same_gender_000000 \
    seedtts_text_en_src_en_ref_same_gender_000001 \
    seedtts_text_en_src_en_ref_same_gender_000002 \
    seedtts_text_zh_src_zh_ref_same_gender_000000 \
    seedtts_text_zh_src_zh_ref_same_gender_000001 \
    seedtts_text_en_src_zh_ref_same_gender_000000 \
    seedtts_text_en_src_zh_ref_same_gender_000001 \
    seedtts_text_en_src_zh_ref_same_gender_000002 \
    seedtts_text_zh_src_en_ref_same_gender_000000 \
    seedtts_text_zh_src_en_ref_same_gender_000001 <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

source = Path(sys.argv[1])
output = Path(sys.argv[2])
case_ids = sys.argv[3:]
rows = {}
for line in source.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    case_id = str(row.get("case_id") or "")
    if case_id in case_ids:
        if case_id in rows:
            raise SystemExit(f"duplicate case_id in source: {case_id}")
        rows[case_id] = row
missing = [case_id for case_id in case_ids if case_id not in rows]
if missing:
    raise SystemExit(f"missing text20 cases: {missing}")
selected = [rows[case_id] for case_id in case_ids]
counts = Counter(str(row.get("cell") or "") for row in selected)
expected = sorted([2, 2, 2, 2, 3, 3, 3, 3])
if len(selected) != 20 or sorted(counts.values()) != expected:
    raise SystemExit(f"unexpected text20 distribution: n={len(selected)} counts={dict(counts)}")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in selected),
    encoding="utf-8",
)
print(f"[batch3436-quick20] text20={output} n={len(selected)} cells={dict(counts)}")
PY
}

run_arm() {
  local arm="$1"
  local gpu_pair="$2"
  local suffix label run_dir checkpoint
  IFS=$'\t' read -r suffix label <<<"$(arm_info "$arm")"
  run_dir="$PROJECT_ROOT/outputs/lora_runs/ver23_content_side_batch3436_${arm}_${suffix}_$STAMP"
  checkpoint="$run_dir/step-$STEP"
  local text_run_id="${label}_step-${STEP}_text_quick20_d2d3_seed1234"
  local text_out="$EVAL_ROOT/$text_run_id"
  local log="$RECORD_ROOT/eval_${arm}_step${STEP}.log"

  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu_pair"
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=8
    echo "[batch3436-quick20] arm=$arm gpu_pair=$gpu_pair checkpoint=$checkpoint"

    PYTHON="$PYTHON" \
    ASR_PYTHON="$ASR_PYTHON" \
    RUN_DIR="$run_dir" \
    RUN_LABEL="$label" \
    EVAL_ROOT="$EVAL_ROOT" \
    DOCS_MD="$EVAL_ROOT/${label}_no_text_quick20_rollup.md" \
    QUICK_VALIDATION_JSONL="$NO_TEXT20_JSONL" \
    CHECKPOINTS="step-$STEP" \
    SEED=1234 \
    QUICK_GPU_COUNT=2 \
    QUICK_NUM_SHARDS=2 \
    QUICK_ASR_NUM_SHARDS=2 \
    SPEAKER_ENCODER_TYPE=embedding_loader \
    TIMBRE_SIDE_ONLY=0 \
    REF_SPEAKER_PROMPT_ATTENTION_CAPTURE_FRAMES=10 \
    RUN_QUICK=1 \
    RUN_T11=0 \
    RUN_ASR=1 \
    BUILD_PAGE=0 \
    bash "$CODE_ROOT/scripts/004054_run_ver2_8_timbre_quick_eval.sh"

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
    PYTHON="$PYTHON" \
    ASR_PYTHON="$ASR_PYTHON" \
    VALIDATION_JSONL="$TEXT20_JSONL" \
    MODEL_PATH="$checkpoint" \
    RUN_ID="$text_run_id" \
    RUN_LABEL="$text_run_id" \
    OUTPUT_DIR="$text_out" \
    MODE=text \
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
      --asr-jsonl "$text_out/${text_run_id}.asr_eval.jsonl" \
      --output-json "$text_out/${text_run_id}.ref_content_similarity_summary.json" \
      --output-md "$text_out/${text_run_id}.ref_content_similarity_summary.md"

    "$PYTHON" "$CODE_ROOT/scripts/004050_summarize_seedtts_speaker_sim_only.py" \
      --validation-jsonl "$TEXT20_JSONL" \
      --run "$text_run_id=$text_out" \
      --output-csv "$text_out/${text_run_id}.speaker_sim.csv" \
      --summary-json "$text_out/${text_run_id}.speaker_sim_summary.json" \
      --summary-md "$text_out/${text_run_id}.speaker_sim_summary.md" \
      --speaker-device cuda:0

    echo "[batch3436-quick20] arm=$arm complete"
  ) > >(tee -a "$log") 2>&1
}

collect_metrics() {
  "$PYTHON" - "$EVAL_ROOT" "$RECORD_ROOT" "$STEP" "$ARMS_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

eval_root = Path(sys.argv[1])
record_root = Path(sys.argv[2])
step = int(sys.argv[3])
arms = [item for item in sys.argv[4].split(",") if item]
labels = {
    "B1": "ver23_batch3436_B1_bnf_last16",
    "B2": "ver23_batch3436_B2_text_r1",
    "A1": "ver23_batch3436_A1_stronger",
    "B3": "ver23_batch3436_B3_weaker",
    "A2": "ver23_batch3436_A2_ctc",
}

rows_out = []
for arm in arms:
    label = labels[arm]
    for mode in ("no_text", "text"):
        middle = "text_" if mode == "text" else ""
        run_id = f"{label}_step-{step}_{middle}quick20_d2d3_seed1234"
        out_dir = eval_root / run_id
        summary = json.loads((out_dir / f"{run_id}.summary.json").read_text(encoding="utf-8"))["overall"]
        speaker_rows = list(csv.DictReader((out_dir / f"{run_id}.speaker_sim.csv").open(encoding="utf-8")))
        valid = [row for row in speaker_rows if row.get("status") == "ok"]
        if not valid:
            raise SystemExit(f"no valid speaker rows: {out_dir}")
        sim_ref = sum(float(row["sim_gen_ref"]) for row in valid) / len(valid)
        sim_src = sum(float(row["sim_gen_source"]) for row in valid) / len(valid)
        ref_bound_count = sum(
            float(row["sim_gen_ref"]) - float(row["sim_gen_source"]) > 0.05
            for row in valid
        )
        ref_content = json.loads(
            (out_dir / f"{run_id}.ref_content_similarity_summary.json").read_text(encoding="utf-8")
        )["overall"]["ref_content_lcs_f1_mean"]
        n = int(summary["n"])
        keep = int(summary["keep"])
        rows_out.append({
            "arm": arm,
            "mode": mode,
            "n": n,
            "keep": keep,
            "fail": (n - keep) / n,
            "cer": float(summary["cer"]),
            "sim_ref": sim_ref,
            "sim_src": sim_src,
            "ref_bound_count": ref_bound_count,
            "ref_bound": ref_bound_count / len(valid),
            "ref_content_f1": float(ref_content),
            "run_id": run_id,
            "output_dir": str(out_dir),
        })

record_root.mkdir(parents=True, exist_ok=True)
(record_root / "metrics.json").write_text(
    json.dumps(rows_out, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
fields = [
    "arm", "mode", "n", "keep", "fail", "cer", "sim_ref", "sim_src",
    "ref_bound_count", "ref_bound", "ref_content_f1", "run_id", "output_dir",
]
with (record_root / "metrics.tsv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows_out)
for row in rows_out:
    print(
        "[batch3436-quick20-metric] "
        f"{row['arm']} {row['mode']} CER={row['cer']:.4f} fail={row['fail']:.1%} "
        f"sim_ref={row['sim_ref']:.4f} sim_src={row['sim_src']:.4f} "
        f"ref_bound={row['ref_bound']:.1%} ref_content_f1={row['ref_content_f1']:.4f}"
    )
PY
}

run_entrypoint() {
  mkdir -p "$RECORD_ROOT" "$EVAL_ROOT"
  exec > >(tee -a "$RECORD_ROOT/run.log") 2>&1
  echo "[batch3436-quick20] start date=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
  echo "[batch3436-quick20] step=$STEP arms=$ARMS_CSV code_root=$CODE_ROOT"
  echo "[batch3436-quick20] compute_group=MTTS-3-2-0715"
  nvidia-smi || true

  prepare_text20
  local pids=()
  local failed=0
  if arm_selected B1 || arm_selected A2; then
    (
      set -euo pipefail
      if arm_selected B1; then
        wait_for_checkpoint B1
        run_arm B1 0,1
      fi
      if arm_selected A2; then
        wait_for_checkpoint A2
        run_arm A2 0,1
      fi
    ) &
    pids+=("$!")
  fi
  if arm_selected B2; then
    (wait_for_checkpoint B2 && run_arm B2 2,3) &
    pids+=("$!")
  fi
  if arm_selected A1; then
    (wait_for_checkpoint A1 && run_arm A1 4,5) &
    pids+=("$!")
  fi
  if arm_selected B3; then
    (wait_for_checkpoint B3 && run_arm B3 6,7) &
    pids+=("$!")
  fi
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ]; then
    echo "ERROR: one or more arm evaluations failed" >&2
    exit 1
  fi

  collect_metrics
  echo "[batch3436-quick20] complete date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

if [ "$ENTRYPOINT" = "1" ]; then
  run_entrypoint
  exit 0
fi

if [ ! -d "$PROJECT_ROOT" ] || [ ! -d "$CODE_ROOT" ]; then
  echo "ERROR: missing PROJECT_ROOT or CODE_ROOT" >&2
  exit 1
fi
if [ ! -x "$QZCLI" ] || [ ! -x "$PYTHON" ]; then
  echo "ERROR: missing qzcli wrapper or Python interpreter" >&2
  exit 1
fi
if [ ! -s "$NO_TEXT20_JSONL" ] || [ ! -s "$TEXT_SOURCE_JSONL" ]; then
  echo "ERROR: missing validation JSONL" >&2
  exit 1
fi

mkdir -p "$RECORD_ROOT" "$QZCLI_HOME"
prepare_text20
if [ "$FORCE" != "1" ] && [ -s "$RECORD_ROOT/metrics.tsv" ]; then
  echo "ERROR: metrics already exist; use FORCE=1 only for an intentional rerun: $RECORD_ROOT/metrics.tsv" >&2
  exit 1
fi

if [ "$DRY_RUN" = "0" ]; then
  if [ "$WAIT_FOR_CHECKPOINTS" = "1" ]; then
    ready_count=0
    for arm in B1 B2 A1 B3 A2; do
      if arm_selected "$arm" && checkpoint_ready "$arm"; then
        ready_count=$((ready_count + 1))
        echo "[batch3436-quick20] submit-ready arm=$arm checkpoint=$(arm_checkpoint "$arm")"
      fi
    done
    if [ "$ready_count" -eq 0 ]; then
      echo "ERROR: no selected checkpoint is ready; refusing to reserve an idle MTTS node" >&2
      exit 1
    fi
  else
    for arm in B1 B2 A1 B3 A2; do
      if arm_selected "$arm"; then
        validate_checkpoint "$arm"
      fi
    done
  fi
fi

COMMAND="env BATCH3436_QUICK20_ENTRYPOINT=1 STEP=$STEP STAMP=$STAMP ARMS_CSV=$ARMS_CSV FORCE=$FORCE WAIT_FOR_CHECKPOINTS=$WAIT_FOR_CHECKPOINTS CHECKPOINT_WAIT_TIMEOUT=$CHECKPOINT_WAIT_TIMEOUT CHECKPOINT_POLL_SECONDS=$CHECKPOINT_POLL_SECONDS PROJECT_ROOT=$PROJECT_ROOT CODE_ROOT=$CODE_ROOT RECORD_ROOT=$RECORD_ROOT EVAL_ROOT=$EVAL_ROOT bash $PROJECT_ROOT/scripts/004069_submit_ver23_batch3436_quick20_qz.sh"
SUBMIT_OUTPUT="$RECORD_ROOT/submit_output.txt"

echo "=========================================="
echo "QZ submit: Batch-34+36 five-arm quick20"
echo "  JOB_NAME=$JOB_NAME"
echo "  STEP=$STEP"
echo "  ARMS_CSV=$ARMS_CSV"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP (MTTS-3-2-0715 only)"
echo "  SPEC=$SPEC"
echo "  RECORD_ROOT=$RECORD_ROOT"
echo "  EVAL_ROOT=$EVAL_ROOT"
echo "  CODE_ROOT=$CODE_ROOT"
echo "  WAIT_FOR_CHECKPOINTS=$WAIT_FOR_CHECKPOINTS timeout=$CHECKPOINT_WAIT_TIMEOUT poll=$CHECKPOINT_POLL_SECONDS"
echo "  DRY_RUN=$DRY_RUN"
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
  echo "ERROR: QZ submission failed; see $SUBMIT_OUTPUT" >&2
  exit "$status"
fi
if [ "$DRY_RUN" = "1" ]; then
  echo "[batch3436-quick20] dry-run passed; no job submitted"
  exit 0
fi

job_id=$(printf '%s\n' "$output" | grep -Eo 'job-[0-9a-fA-F-]{36}' | tail -n 1 || true)
{
  printf 'job_name\tjob_id\tstep\tarms\tcompute_group\trecord_root\teval_root\n'
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$JOB_NAME" "$job_id" "$STEP" "$ARMS_CSV" "$COMPUTE_GROUP" "$RECORD_ROOT" "$EVAL_ROOT"
} > "$RECORD_ROOT/submitted_jobs.tsv"
echo "[batch3436-quick20] submitted job_id=$job_id record=$RECORD_ROOT/submitted_jobs.tsv"

#!/usr/bin/env bash
# MTTS-only 8-way CTC greedy probe for Batch-34+36 A2.
#
# The eight existing no_text validation shards are mapped one-to-one to the
# eight H200 GPUs. DRY_RUN defaults to 1; this script never submits unless the
# caller explicitly sets DRY_RUN=0.
#
# Dry-run:
#   bash scripts/004073_submit_ver23_a2_ctc_greedy_probe_qz.sh
#
# Submit after explicit approval:
#   DRY_RUN=0 bash scripts/004073_submit_ver23_a2_ctc_greedy_probe_qz.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC}"
CODE_ROOT="${CODE_ROOT:-/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch3436_eval_20260711_1092820}"
TRAINING_CODE_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch3436_20260710_1092820"
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

STEP="${STEP:-3000}"
TRAIN_STAMP="${TRAIN_STAMP:-20260710_mtts}"
EVAL_STAMP="${EVAL_STAMP:-20260711_mtts}"
DRY_RUN="${DRY_RUN:-1}"
FORCE="${FORCE:-0}"
ENTRYPOINT="${BATCH3436_A2_CTC_PROBE_ENTRYPOINT:-0}"
MAX_ROWS_PER_SHARD="${MAX_ROWS_PER_SHARD:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EXPECTED_ROWS="${EXPECTED_ROWS:-1000}"

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
CONFIG="${CONFIG:-$CODE_ROOT/configs/remote_full.yaml}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/outputs/lora_runs/ver23_content_side_batch3436_A2_ver23_ctc_3k_${TRAIN_STAMP}/step-${STEP}}"
INPUT_ROOT="${INPUT_ROOT:-$PROJECT_ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/testset/outputs/ver23_batch3436_A2_ctc_greedy_probe_step${STEP}_${EVAL_STAMP}}"
RECORD_ROOT="${RECORD_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/ver23_batch3436_A2_ctc_greedy_probe_step${STEP}_${EVAL_STAMP}}"
JOB_NAME="${JOB_NAME:-ver23_batch3436_A2_ctc_probe_step${STEP}_${EVAL_STAMP}}"

if [ "$COMPUTE_GROUP" != "$ALLOWED_COMPUTE_GROUP" ]; then
  echo "ERROR: this probe is restricted to MTTS-3-2-0715 ($ALLOWED_COMPUTE_GROUP); got $COMPUTE_GROUP" >&2
  exit 2
fi
if [ "$INSTANCES" != "1" ]; then
  echo "ERROR: this wrapper requires exactly one 8xH200 instance; got INSTANCES=$INSTANCES" >&2
  exit 2
fi
case "$DRY_RUN:$FORCE:$ENTRYPOINT" in
  [01]:[01]:[01]) ;;
  *)
    echo "ERROR: DRY_RUN, FORCE, and BATCH3436_A2_CTC_PROBE_ENTRYPOINT must be 0 or 1" >&2
    exit 2
    ;;
esac
for value_name in STEP MAX_ROWS_PER_SHARD BATCH_SIZE NUM_WORKERS EXPECTED_ROWS; do
  value="${!value_name}"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $value_name must be a non-negative integer; got $value" >&2
    exit 2
  fi
done
if [ "$BATCH_SIZE" -le 0 ] || [ "$EXPECTED_ROWS" -le 0 ]; then
  echo "ERROR: BATCH_SIZE and EXPECTED_ROWS must be positive" >&2
  exit 2
fi
if [ "$CODE_ROOT" = "$TRAINING_CODE_ROOT" ]; then
  echo "ERROR: refusing to modify/use the frozen Batch-34+36 training snapshot for this new probe" >&2
  exit 2
fi

probe_script="$CODE_ROOT/scripts/004019_ctc_greedy_decode_probe.py"

validate_inputs() {
  if [ ! -d "$PROJECT_ROOT" ] || [ ! -d "$CODE_ROOT" ]; then
    echo "ERROR: missing PROJECT_ROOT or safe CODE_ROOT" >&2
    return 1
  fi
  if [ ! -x "$PYTHON" ] || [ ! -x "$QZCLI" ]; then
    echo "ERROR: missing Python or qzcli wrapper" >&2
    return 1
  fi
  if [ ! -s "$probe_script" ] || [ ! -s "$CONFIG" ]; then
    echo "ERROR: missing CTC probe or runtime config in safe eval snapshot: $probe_script $CONFIG" >&2
    return 1
  fi
  if ! grep -q 'source_bnf_content_cross_attn_memory' "$probe_script"; then
    echo "ERROR: safe eval snapshot does not contain the corrected BNF-memory CTC probe" >&2
    return 1
  fi
  local required
  for required in adapter_model.safetensors adapter_config.json README.md timbre_memory_adapter.pt timbre_memory_config.json; do
    if [ ! -s "$MODEL_PATH/$required" ]; then
      echo "ERROR: missing or empty A2 checkpoint file: $MODEL_PATH/$required" >&2
      return 1
    fi
  done
  "$PYTHON" - "$MODEL_PATH/timbre_memory_config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    cfg = json.load(handle)
checks = {
    "content_cross_attn_enabled": cfg.get("content_cross_attn_enabled") is True,
    "content_ctc_weight": float(cfg.get("content_ctc_weight") or 0.0) > 0.0,
    "content_ctc_vocab_size": int(cfg.get("content_ctc_vocab_size") or 0) > 1,
    "content_ctc_blank_id": int(cfg.get("content_ctc_blank_id") or 0) == 0,
    "content_ctc_token_offset": int(cfg.get("content_ctc_token_offset") or 0) == 1,
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit(f"invalid A2 CTC checkpoint config: failed={failed} cfg={cfg}")
print(
    "[a2-ctc-probe] checkpoint config PASS "
    f"weight={cfg['content_ctc_weight']} vocab={cfg['content_ctc_vocab_size']} "
    f"layers={cfg.get('content_cross_attn_layers')}"
)
PY
  local shard
  for shard in $(seq 0 7); do
    input_jsonl=$(printf '%s/no_text.valid.jsonl.shard%05d-of00008.jsonl' "$INPUT_ROOT" "$shard")
    if [ ! -s "$input_jsonl" ]; then
      echo "ERROR: missing no_text validation shard: $input_jsonl" >&2
      return 1
    fi
  done
  "$PYTHON" - "$INPUT_ROOT" "$EXPECTED_ROWS" "$MAX_ROWS_PER_SHARD" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_rows = int(sys.argv[2])
max_rows = int(sys.argv[3])
rows = 0
missing_bnf = 0
missing_tokens = 0
wrong_mode = 0
for shard in range(8):
    path = root / f"no_text.valid.jsonl.shard{shard:05d}-of00008.jsonl"
    shard_rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if max_rows > 0 and shard_rows >= max_rows:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            shard_rows += 1
            rows += 1
            wrong_mode += str(row.get("moss_codecvc_mode") or row.get("mode") or "").strip() != "no_text"
            missing_bnf += not bool(
                row.get("source_semantic_features_path")
                or row.get("source_wavlm_bnf_features_path")
                or row.get("source_asr_bnf_feature_path")
            )
            missing_tokens += not bool(row.get("content_token_ids"))
if max_rows == 0 and rows != expected_rows:
    raise SystemExit(f"row-count mismatch: rows={rows} expected={expected_rows}")
if wrong_mode or missing_bnf or missing_tokens:
    raise SystemExit(
        f"invalid probe input: rows={rows} wrong_mode={wrong_mode} "
        f"missing_bnf={missing_bnf} missing_tokens={missing_tokens}"
    )
print(f"[a2-ctc-probe] input audit PASS rows={rows} no_text={rows} shards=8")
PY
}

merge_results() {
  "$PYTHON" - "$OUTPUT_ROOT" "$EXPECTED_ROWS" "$MAX_ROWS_PER_SHARD" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_rows = int(sys.argv[2])
max_rows = int(sys.argv[3])
summaries = []
row_paths = []
for shard in range(8):
    prefix = root / f"shard_{shard:02d}"
    summary_path = prefix.with_suffix(".summary.json")
    row_path = prefix.with_suffix(".jsonl")
    if not summary_path.is_file() or not row_path.is_file():
        raise SystemExit(f"missing shard output: {summary_path} or {row_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("content_ctc_input_source") != "source_bnf_content_cross_attn_memory":
        raise SystemExit(f"wrong CTC input source in {summary_path}: {summary.get('content_ctc_input_source')}")
    if int(summary.get("text_bypassed_rows") or 0) != 0:
        raise SystemExit(f"unexpected text rows in no_text shard: {summary_path}")
    summaries.append(summary)
    row_paths.append(row_path)

sum_keys = (
    "input_rows", "rows", "no_text_rows", "text_bypassed_rows", "exact_matches",
    "ter_sum", "ter_count", "edit_distance", "target_tokens", "blank_frames",
    "nonblank_frames", "total_frames", "blank_posterior_sum", "nonblank_posterior_sum",
    "collapsed_tokens", "collapsed_punctuation_tokens", "empty_predictions",
    "punctuation_only_collapses", "single_token_collapses", "severe_length_collapses",
    "blank_dominant_rows", "dominant_nonblank_frame_share_sum",
)
overall = {
    "schema_version": 2,
    "content_ctc_input_source": "source_bnf_content_cross_attn_memory",
    "content_cross_attn_mode": True,
    "shards": 8,
}
for key in sum_keys:
    overall[key] = sum(float(item.get(key) or 0.0) for item in summaries)
for key in ("input_rows", "rows", "no_text_rows", "text_bypassed_rows", "exact_matches", "ter_count",
            "edit_distance", "target_tokens", "blank_frames", "nonblank_frames", "total_frames",
            "collapsed_tokens", "collapsed_punctuation_tokens", "empty_predictions",
            "punctuation_only_collapses", "single_token_collapses", "severe_length_collapses",
            "blank_dominant_rows"):
    overall[key] = int(overall[key])
overall["collapsed_len_min"] = min(int(item["collapsed_len_min"]) for item in summaries)
overall["collapsed_len_max"] = max(int(item["collapsed_len_max"]) for item in summaries)
rows = max(1, overall["rows"])
frames = max(1, overall["total_frames"])
targets = max(1, overall["target_tokens"])
collapsed = max(1, overall["collapsed_tokens"])
ter_count = max(1, overall["ter_count"])
overall.update({
    "exact_rate": overall["exact_matches"] / rows,
    "mean_row_greedy_ter": overall["ter_sum"] / ter_count,
    "greedy_ter": overall["edit_distance"] / targets,
    "blank_frame_rate": overall["blank_frames"] / frames,
    "nonblank_frame_rate": overall["nonblank_frames"] / frames,
    "blank_posterior_mean": overall["blank_posterior_sum"] / frames,
    "nonblank_posterior_mean": overall["nonblank_posterior_sum"] / frames,
    "collapsed_len_mean": overall["collapsed_tokens"] / rows,
    "collapsed_to_target_token_ratio": overall["collapsed_tokens"] / targets,
    "collapsed_punctuation_share": overall["collapsed_punctuation_tokens"] / collapsed,
    "empty_prediction_rate": overall["empty_predictions"] / rows,
    "punctuation_only_collapse_rate": overall["punctuation_only_collapses"] / rows,
    "single_token_collapse_rate": overall["single_token_collapses"] / rows,
    "severe_length_collapse_rate": overall["severe_length_collapses"] / rows,
    "blank_dominant_rate": overall["blank_dominant_rows"] / rows,
    "dominant_nonblank_frame_share_mean": overall["dominant_nonblank_frame_share_sum"] / rows,
})
overall["collapse_diagnosis"] = {
    "blank_collapse": bool(overall["nonblank_frame_rate"] < 0.01 or overall["empty_prediction_rate"] >= 0.5),
    "punctuation_collapse": bool(
        overall["punctuation_only_collapse_rate"] >= 0.1
        or overall["collapsed_punctuation_share"] >= 0.5
    ),
    "dominant_token_collapse": bool(
        overall["dominant_nonblank_frame_share_mean"] >= 0.9
        and overall["collapsed_to_target_token_ratio"] < 0.25
    ),
}
if max_rows == 0 and overall["rows"] != expected_rows:
    raise SystemExit(f"merged row-count mismatch: rows={overall['rows']} expected={expected_rows}")

with (root / "all_rows.jsonl").open("w", encoding="utf-8") as output:
    for path in row_paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                output.write(line)
(root / "overall_summary.json").write_text(
    json.dumps(overall, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

fields = [
    "shard", "rows", "greedy_ter", "mean_row_greedy_ter", "exact_rate",
    "blank_frame_rate", "nonblank_frame_rate", "nonblank_posterior_mean",
    "collapsed_len_mean", "empty_prediction_rate", "punctuation_only_collapse_rate",
    "dominant_nonblank_frame_share_mean",
]
with (root / "shard_metrics.tsv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    for shard, summary in enumerate(summaries):
        writer.writerow({"shard": shard, **{key: summary[key] for key in fields if key != "shard"}})

flags = overall["collapse_diagnosis"]
lines = [
    "# Batch-34+36 A2 CTC greedy probe",
    "",
    f"- CTC input: `{overall['content_ctc_input_source']}`",
    f"- Rows: {overall['rows']}",
    f"- Corpus greedy TER: {overall['greedy_ter']:.6f}",
    f"- Mean row TER: {overall['mean_row_greedy_ter']:.6f}",
    f"- Exact rate: {overall['exact_rate']:.6f}",
    f"- Argmax blank / nonblank: {overall['blank_frame_rate']:.6f} / {overall['nonblank_frame_rate']:.6f}",
    f"- Posterior blank / nonblank: {overall['blank_posterior_mean']:.6f} / {overall['nonblank_posterior_mean']:.6f}",
    f"- Collapsed length mean/min/max: {overall['collapsed_len_mean']:.3f} / {overall['collapsed_len_min']} / {overall['collapsed_len_max']}",
    f"- Collapsed/target token ratio: {overall['collapsed_to_target_token_ratio']:.6f}",
    f"- Empty / punctuation-only / severe-short: {overall['empty_prediction_rate']:.6f} / {overall['punctuation_only_collapse_rate']:.6f} / {overall['severe_length_collapse_rate']:.6f}",
    f"- Collapsed punctuation share: {overall['collapsed_punctuation_share']:.6f}",
    f"- Dominant nonblank frame-token share mean: {overall['dominant_nonblank_frame_share_mean']:.6f}",
    f"- Collapse flags: blank={flags['blank_collapse']} punctuation={flags['punctuation_collapse']} dominant_token={flags['dominant_token_collapse']}",
    "",
]
(root / "overall_summary.md").write_text("\n".join(lines), encoding="utf-8")
print(
    "[a2-ctc-probe] overall "
    f"rows={overall['rows']} TER={overall['greedy_ter']:.6f} exact={overall['exact_rate']:.6f} "
    f"blank={overall['blank_frame_rate']:.6f} nonblank={overall['nonblank_frame_rate']:.6f} "
    f"nonblank_post={overall['nonblank_posterior_mean']:.6f} flags={flags}"
)
PY
}

run_entrypoint() {
  validate_inputs
  mkdir -p "$OUTPUT_ROOT" "$RECORD_ROOT"
  exec > >(tee -a "$RECORD_ROOT/run.log") 2>&1
  echo "[a2-ctc-probe] start date=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
  echo "[a2-ctc-probe] model=$MODEL_PATH code_root=$CODE_ROOT output=$OUTPUT_ROOT"
  echo "[a2-ctc-probe] compute_group=MTTS-3-2-0715 mapping=8_shards_to_8_H200"
  nvidia-smi || true

  local pids=()
  local shard
  for shard in $(seq 0 7); do
    input_jsonl=$(printf '%s/no_text.valid.jsonl.shard%05d-of00008.jsonl' "$INPUT_ROOT" "$shard")
    output_jsonl=$(printf '%s/shard_%02d.jsonl' "$OUTPUT_ROOT" "$shard")
    summary_json=$(printf '%s/shard_%02d.summary.json' "$OUTPUT_ROOT" "$shard")
    log=$(printf '%s/shard_%02d.log' "$RECORD_ROOT" "$shard")
    (
      set -euo pipefail
      export CUDA_VISIBLE_DEVICES="$shard"
      export TOKENIZERS_PARALLELISM=false
      export HF_HUB_OFFLINE=1
      export TRANSFORMERS_OFFLINE=1
      export OMP_NUM_THREADS=8
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      export HF_MODULES_CACHE="$RECORD_ROOT/hf_modules_cache/shard_$shard"
      mkdir -p "$HF_MODULES_CACHE"
      "$PYTHON" "$probe_script" \
        --config "$CONFIG" \
        --model-path "$MODEL_PATH" \
        --jsonl "$input_jsonl" \
        --output-jsonl "$output_jsonl" \
        --summary-json "$summary_json" \
        --max-rows "$MAX_ROWS_PER_SHARD" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --device cuda:0 \
        --speaker-encoder-type embedding_loader
    ) > >(tee -a "$log") 2>&1 &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ]; then
    echo "ERROR: one or more A2 CTC probe shards failed" >&2
    exit 1
  fi
  merge_results
  echo "[a2-ctc-probe] complete date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

if [ "$ENTRYPOINT" = "1" ]; then
  run_entrypoint
  exit 0
fi

validate_inputs
mkdir -p "$RECORD_ROOT" "$QZCLI_HOME"
if [ "$FORCE" != "1" ] && [ -s "$OUTPUT_ROOT/overall_summary.json" ]; then
  echo "ERROR: completed output already exists; set FORCE=1 only for an intentional rerun: $OUTPUT_ROOT" >&2
  exit 1
fi

COMMAND="env BATCH3436_A2_CTC_PROBE_ENTRYPOINT=1 STEP=$STEP TRAIN_STAMP=$TRAIN_STAMP EVAL_STAMP=$EVAL_STAMP FORCE=$FORCE MAX_ROWS_PER_SHARD=$MAX_ROWS_PER_SHARD BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=$NUM_WORKERS EXPECTED_ROWS=$EXPECTED_ROWS PROJECT_ROOT=$PROJECT_ROOT CODE_ROOT=$CODE_ROOT CONFIG=$CONFIG MODEL_PATH=$MODEL_PATH INPUT_ROOT=$INPUT_ROOT OUTPUT_ROOT=$OUTPUT_ROOT RECORD_ROOT=$RECORD_ROOT bash $CODE_ROOT/scripts/004073_submit_ver23_a2_ctc_greedy_probe_qz.sh"
SUBMIT_OUTPUT="$RECORD_ROOT/submit_output.txt"

echo "=========================================="
echo "QZ submit: Batch-34+36 A2 CTC greedy probe"
echo "  JOB_NAME=$JOB_NAME"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  INPUT=8 x no_text.valid shard"
echo "  PARALLELISM=8 processes / 8 H200"
echo "  COMPUTE_GROUP=$COMPUTE_GROUP (MTTS-3-2-0715 only)"
echo "  SPEC=$SPEC"
echo "  CODE_ROOT=$CODE_ROOT"
echo "  CONFIG=$CONFIG"
echo "  OUTPUT_ROOT=$OUTPUT_ROOT"
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
  echo "ERROR: QZ submission/dry-run failed; see $SUBMIT_OUTPUT" >&2
  exit "$status"
fi
if [ "$DRY_RUN" = "1" ]; then
  echo "[a2-ctc-probe] dry-run passed; no job submitted"
  exit 0
fi

job_id=$(printf '%s\n' "$output" | grep -Eo 'job-[0-9a-fA-F-]{36}' | tail -n 1 || true)
{
  printf 'job_name\tjob_id\tstep\tcompute_group\tmodel_path\toutput_root\n'
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$JOB_NAME" "$job_id" "$STEP" "$COMPUTE_GROUP" "$MODEL_PATH" "$OUTPUT_ROOT"
} > "$RECORD_ROOT/submitted_jobs.tsv"
echo "[a2-ctc-probe] submitted job_id=$job_id record=$RECORD_ROOT/submitted_jobs.tsv"

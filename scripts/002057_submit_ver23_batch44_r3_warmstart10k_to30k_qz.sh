#!/bin/sh
# Batch-44 recovery: continue the selected r3 arm from the complete step-10000
# weights for another 20k optimizer updates.
#
# This is intentionally recorded as a weights-only warm start, not an exact
# resume: the source checkpoint has no optimizer/scheduler/RNG/trainer state.
# The new run therefore uses a separate output directory and record root.
#
# Safe default (audit + generated runner + QZ payload dry-run only):
#   sh scripts/002057_submit_ver23_batch44_r3_warmstart10k_to30k_qz.sh
#
# Live submit requires all three explicit gates:
#   LIVE=1 DRY_RUN=0 CONFIRM_WARMSTART=1 \
#     sh scripts/002057_submit_ver23_batch44_r3_warmstart10k_to30k_qz.sh

set -eu

PROJECT_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
FROZEN_CODE_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC_snapshots/ver23_batch3436_20260710_1092820"
QZCLI="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/qzcli_with_deps.sh"
QZCLI_HOME="/inspire/ssd/project/embodied-multimodality/public/xyzhang/.codex/qzcli_home"

STAMP="20260713"
DRY_RUN="${DRY_RUN:-1}"
LIVE="${LIVE:-0}"
CONFIRM_WARMSTART="${CONFIRM_WARMSTART:-0}"

WORKSPACE="CI-情境智能"
PROJECT="CI-情境智能"
EXPECTED_WORKSPACE_ID="ws-8207e9e2-e733-4eec-a475-cfa1c36480ba"
EXPECTED_PROJECT_ID="project-c67c548f-f02c-453b-ba5b-8745db6886e7"
ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122" # MTTS-3-2-0715
ALLOWED_SPEC="67b10bc6-78b0-41a3-aaf4-358eeeb99009"
ALLOWED_GPU_TYPE="NVIDIA_H200_SXM_141G"
ALLOWED_ACCELERATE_CONFIG="configs/accelerate_fsdp_h200_8gpu_no_ckpt.yaml"
COMPUTE_GROUP="${COMPUTE_GROUP:-$ALLOWED_COMPUTE_GROUP}"
SPEC="${SPEC:-$ALLOWED_SPEC}"
INSTANCES="${INSTANCES:-1}"
QZCLI_GPU_TYPE_OVERRIDE="${QZCLI_GPU_TYPE_OVERRIDE:-$ALLOWED_GPU_TYPE}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$ALLOWED_ACCELERATE_CONFIG}"
IMAGE="docker.sii.shaipower.online/inspire-studio/ngc-pytorch-25.10:25_patch_20260420"
IMAGE_TYPE="SOURCE_PRIVATE"
FRAMEWORK="pytorch"
SHM_GI="1200"
PRIORITY="10"

JOB_NAME="ver2_9_5_final_r3_v1_warmstart10k_to30k"
BATCH_ID="ver23_batch44_${JOB_NAME}_${STAMP}"
RECORD_ROOT="$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_r3_v1_warmstart10k_to30k_${STAMP}"
OUT_DIR="$PROJECT_ROOT/outputs/lora_runs/$JOB_NAME"
BASE_RUN_DIR="$PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r3_v1_30k"
RESUME_ADAPTER_PATH="$BASE_RUN_DIR/step-10000"
BASE_EFFECTIVE_STEP="10000"
CONTINUATION_STEPS="20000"
EFFECTIVE_TARGET_STEP="30000"

V1_DIR="$PROJECT_ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709"
NO_TEXT_TRAIN_JSONL="$V1_DIR/no_text.train.jsonl"
TEXT_TRAIN_JSONL="$V1_DIR/text.train.jsonl"
TRAIN_JSONL_SPEC="$NO_TEXT_TRAIN_JSONL::repeat=1,$TEXT_TRAIN_JSONL::repeat=3"
SOURCE_R3_RECORD="$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713/r3"
SOURCE_R3_CORE="$SOURCE_R3_RECORD/train_args_dry_run_core.json"

case "$DRY_RUN:$LIVE:$CONFIRM_WARMSTART" in
  [01]:[01]:[01]) ;;
  *) echo "ERROR: DRY_RUN, LIVE and CONFIRM_WARMSTART must be 0 or 1" >&2; exit 2 ;;
esac
if [ "$LIVE" = "0" ] && [ "$DRY_RUN" = "0" ]; then
  echo "ERROR: DRY_RUN=0 requires LIVE=1 and CONFIRM_WARMSTART=1" >&2
  exit 2
fi
if [ "$LIVE" = "1" ] && { [ "$DRY_RUN" != "0" ] || [ "$CONFIRM_WARMSTART" != "1" ]; }; then
  echo "ERROR: live submit requires LIVE=1 DRY_RUN=0 CONFIRM_WARMSTART=1" >&2
  exit 2
fi
if [ "$COMPUTE_GROUP" != "$ALLOWED_COMPUTE_GROUP" ] || [ "$SPEC" != "$ALLOWED_SPEC" ]; then
  echo "ERROR: continuation may only use MTTS-3-2-0715 and the registered spec" >&2
  exit 2
fi
if [ "$INSTANCES" != "1" ] || [ "$QZCLI_GPU_TYPE_OVERRIDE" != "$ALLOWED_GPU_TYPE" ]; then
  echo "ERROR: continuation requires one 8xH200 node" >&2
  exit 2
fi
if [ "$ACCELERATE_CONFIG" != "$ALLOWED_ACCELERATE_CONFIG" ]; then
  echo "ERROR: continuation requires $ALLOWED_ACCELERATE_CONFIG" >&2
  exit 2
fi
[ -x "$QZCLI" ] || { echo "ERROR: qzcli wrapper unavailable" >&2; exit 1; }
[ -d "$QZCLI_HOME" ] || { echo "ERROR: qzcli HOME unavailable" >&2; exit 1; }
[ -d "$FROZEN_CODE_ROOT" ] || { echo "ERROR: frozen code root unavailable" >&2; exit 1; }

# A submitted continuation is immutable.  Refuse every later invocation before
# regenerating the runner, core config, payload, or warm-start contract.
if [ -s "$RECORD_ROOT/submitted_jobs.tsv" ]; then
  echo "ERROR: continuation is already submitted; refusing to rewrite provenance: $RECORD_ROOT" >&2
  exit 1
fi

mkdir -p "$RECORD_ROOT"

audit_warmstart_source() {
  python - \
    "$RESUME_ADAPTER_PATH" "$NO_TEXT_TRAIN_JSONL" "$TEXT_TRAIN_JSONL" \
    "$SOURCE_R3_CORE" "$OUT_DIR" "$JOB_NAME" "$LIVE" "$RECORD_ROOT/warm_start_contract.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

checkpoint, no_text, text, source_core, out_dir, job_name, verify_full, output = (
    Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]),
    Path(sys.argv[5]), sys.argv[6], sys.argv[7] == "1", Path(sys.argv[8])
)
expected_files = {
    "adapter_model.safetensors": (87366096, "a8f552419d5eec2fdd6e1e87f210ef745a45e220cc4d1ae7b37060dd36c6adda"),
    "adapter_config.json": (1179, "fab62263e36848a1c4ab20c573dd99f6d644ac0ee8c6cb39349435649eb3fb16"),
    "README.md": (179, "4d45f7d68a88a39671cc0cbc86f1acdfbee5351401eee2a97df253f0d077717f"),
    "timbre_memory_adapter.pt": (1697093491, "4892a136665bcf5ae78068ef39ca7e01ece426109f333d6b7572bb84cba714d0"),
    "timbre_memory_config.json": (5026, "5c8842d87327c2cf1af2697725a19bf2b53ba654fa0a6b3f68b6a42fd50e9970"),
}
errors = []
files = {}
for name, (size, wanted_sha) in expected_files.items():
    path = checkpoint / name
    if not path.is_file() or path.is_symlink():
        errors.append(f"missing or symlink checkpoint file: {path}")
        continue
    actual_size = path.stat().st_size
    with path.open("rb") as handle:
        actual_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    files[name] = {"path": str(path), "bytes": actual_size, "sha256": actual_sha}
    if actual_size != size or actual_sha != wanted_sha:
        errors.append(f"checkpoint identity drift for {name}: {actual_size}/{actual_sha}")

for forbidden in (
    "optimizer.bin", "optimizer.pt", "scheduler.bin", "scheduler.pt",
    "trainer_state.json", "random_states_0.pkl",
):
    if (checkpoint / forbidden).exists():
        errors.append(f"unexpected strict-resume state appeared: {forbidden}")

if not no_text.is_file() or no_text.stat().st_size != 18048211813:
    errors.append(f"v1 no_text missing/size drift: {no_text}")
if not text.is_file() or text.stat().st_size != 2196087856:
    errors.append(f"v1 text missing/size drift: {text}")
data_hashes = {}
if verify_full:
    for label, path, wanted_sha in (
        ("no_text", no_text, "c4b061f0a968e73710dc86d81478483a9195e8a053f510f09be7952d60c3d279"),
        ("text", text, "c6632888d08e79382001909a65951d6ce7bab80d7fb585cf7729e0a9188a9a80"),
    ):
        with path.open("rb") as handle:
            actual_sha = hashlib.file_digest(handle, "sha256").hexdigest()
        data_hashes[label] = actual_sha
        if actual_sha != wanted_sha:
            errors.append(f"{label} full SHA drift: {actual_sha}")
if not source_core.is_file():
    errors.append(f"missing source r3 core identity: {source_core}")
else:
    with source_core.open("rb") as handle:
        core_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    if core_sha != "3637b90fe14bbda7c4915362d5bd726b072833f07853df4fc555fde0d2c32d7b":
        errors.append(f"source r3 core SHA drift: {core_sha}")

try:
    cfg = json.loads((checkpoint / "timbre_memory_config.json").read_text(encoding="utf-8"))
except Exception as exc:
    errors.append(f"invalid timbre config: {exc}")
    cfg = {}
expected_cfg = {
    "content_cross_attn_enabled": True,
    "content_cross_attn_layers": "all",
    "content_cross_attn_feature_dim": 768,
    "content_cross_attn_gate_init": -0.5,
    "content_cross_attn_output_scale": 0.3,
    "content_encoder_layers": 2,
    "guided_attn_loss_weight": 0.05,
    "guided_attn_warmup_steps": 1000,
    "phoneme_classifier_loss_weight": 0.02,
    "content_ctc_weight": 0.0,
    "progress_loss_weight": 0.1,
    "stop_loss_weight": 0.2,
    "target_front_ce_weight": 4.0,
    "target_front_ce_seconds": 0.75,
    "source_semantic_memory_enabled": False,
    "speaker_side_pathway_enabled": False,
    "speaker_cross_attn_enabled": False,
}
for key, wanted in expected_cfg.items():
    if cfg.get(key) != wanted:
        errors.append(f"base config {key}={cfg.get(key)!r}, expected {wanted!r}")
if errors:
    raise SystemExit("warm-start source audit failed:\n- " + "\n- ".join(errors))

payload = {
    "schema": "batch44_r3_weights_only_warm_start_v1",
    "status": "audited_pre_submit",
    "job_name": job_name,
    "output_dir": str(out_dir),
    "source_checkpoint": str(checkpoint),
    "source_checkpoint_files": files,
    "source_effective_step": 10000,
    "continuation_local_target_step": 20000,
    "effective_target_step": 30000,
    "effective_step_offset": 10000,
    "step_mapping": "effective_step = 10000 + continuation_local_step",
    "resume_semantics": "weights_only_warm_start_not_exact_resume",
    "state_resets": ["optimizer", "scheduler", "rng", "global_step", "data_iterator"],
    "lost_unsaved_source_updates": 660,
    "mechanical_recovery_overrides": {
        "warmup_ratio": 0.0,
        "guided_attn_warmup_steps": 0,
        "reason": "base step 10000 had already completed both warmups",
    },
    "data": {
        "no_text": {"path": str(no_text), "bytes": no_text.stat().st_size, "repeat": 1,
                    "registered_sha256": "c4b061f0a968e73710dc86d81478483a9195e8a053f510f09be7952d60c3d279",
                    "actual_sha256": data_hashes.get("no_text")},
        "text": {"path": str(text), "bytes": text.stat().st_size, "repeat": 3,
                 "registered_sha256": "c6632888d08e79382001909a65951d6ce7bab80d7fb585cf7729e0a9188a9a80",
                 "actual_sha256": data_hashes.get("text")},
    },
    "full_data_sha256_verified": verify_full,
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[batch44-r3-warmstart] source audit PASS contract={output}")
PY
}

generate_runner() {
  requested_dry_run="$1"
  env -i \
    PATH="$PATH" HOME="$QZCLI_HOME" QZCLI="$QZCLI" \
    ROOT="$FROZEN_CODE_ROOT" DRY_RUN="$requested_dry_run" \
    WORKSPACE="$WORKSPACE" PROJECT="$PROJECT" COMPUTE_GROUP="$COMPUTE_GROUP" \
    SPEC="$SPEC" INSTANCES="$INSTANCES" QZCLI_GPU_TYPE_OVERRIDE="$QZCLI_GPU_TYPE_OVERRIDE" \
    ACCELERATE_CONFIG="$ACCELERATE_CONFIG" \
    TRAINSET_DIR="$V1_DIR" TRAIN_JSONL="$NO_TEXT_TRAIN_JSONL" \
    NO_TEXT_TRAIN_JSONL="$NO_TEXT_TRAIN_JSONL" TEXT_TRAIN_JSONL="$TEXT_TRAIN_JSONL" \
    TEXT_REPEAT="3" TRAIN_JSONL_SPEC="$TRAIN_JSONL_SPEC" \
    BATCH_ID="$BATCH_ID" JOB_NAME="$JOB_NAME" JOB_NAME_PREFIX="$JOB_NAME" \
    QZ_RECORD_ROOT="$RECORD_ROOT" OUT_DIR="$OUT_DIR" \
    NUM_EPOCHS="6" MAX_TRAIN_STEPS="$CONTINUATION_STEPS" SAVE_STEPS="2000" EVAL_STEPS="2000" \
    EVAL_MAX_BATCHES="0" EVAL_NUM_WORKERS="0" LEARNING_RATE="1e-5" \
    LR_SCHEDULER_TYPE="constant_with_warmup" WARMUP_RATIO="0.0" WEIGHT_DECAY="0.01" \
    PER_DEVICE_BATCH_SIZE="1" GRADIENT_ACCUMULATION_STEPS="8" GPU_COUNT="8" \
    MIXED_PRECISION="bf16" GRADIENT_CHECKPOINTING="0" \
    LORA_R="16" LORA_ALPHA="32" LORA_DROPOUT="0.05" \
    LOGGING_STEPS="20" NUM_WORKERS="4" MAX_GRAD_NORM="1.0" POST_TRAIN_QUICK_EVAL="0" \
    RESUME_ADAPTER_PATH="$RESUME_ADAPTER_PATH" TRAIN_SOURCE_SEMANTIC_ONLY="0" FREEZE_LORA="0" \
    FREEZE_ROLE_ROUTING="0" FREEZE_TIMBRE_ADAPTER="0" \
    EVAL_JSONL="" EVAL_JSONL_SPEC="" EVAL_SEEN_JSONL="" EVAL_SEEN_JSONL_SPEC="" \
    EVAL_UNSEEN_JSONL="" EVAL_UNSEEN_JSONL_SPEC="" \
    CONTENT_CROSS_ATTN_LAYERS="all" CONTENT_CROSS_ATTN_FEATURE_DIM="768" \
    CONTENT_CROSS_ATTN_GATE_INIT="-0.5" CONTENT_CROSS_ATTN_OUTPUT_SCALE="0.3" \
    CONTENT_CROSS_ATTN_DROPOUT="0.0" CONTENT_ENCODER_LAYERS="2" \
    CONTENT_ENCODER_CONV_KERNEL_SIZE="7" GUIDED_ATTN_LOSS_WEIGHT="0.05" \
    GUIDED_ATTN_WARMUP_STEPS="0" GUIDED_ATTN_BAND_FRAMES="3" \
    PHONEME_CLASSIFIER_LOSS_WEIGHT="0.02" CONTENT_CTC_WEIGHT="0.0" \
    TIMBRE_ADAPTER_GATE_LR_MULTIPLIER="1.0" \
    sh "$FROZEN_CODE_ROOT/scripts/002049_submit_ver23_content_side_3k_qz.sh"
}

audit_generated_config() {
  python - \
    "$RECORD_ROOT/train_args_dry_run_core.json" "$SOURCE_R3_CORE" \
    "$RECORD_ROOT/run_train_entrypoint.sh" "$RESUME_ADAPTER_PATH" \
    "$OUT_DIR" "$JOB_NAME" "$TRAIN_JSONL_SPEC" "$RECORD_ROOT/generated_config_audit.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

current_path, source_path, runner, resume, out_dir, job, spec, output = (
    Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4],
    sys.argv[5], sys.argv[6], sys.argv[7], Path(sys.argv[8])
)
current = json.loads(current_path.read_text(encoding="utf-8"))
source = json.loads(source_path.read_text(encoding="utf-8"))
expected = {
    "JOB_NAME_PREFIX": job,
    "OUT_DIR": out_dir,
    "TRAIN_JSONL_SPEC": spec,
    "TEXT_REPEAT": "3",
    "MAX_TRAIN_STEPS": "20000",
    "SAVE_STEPS": "2000",
    "EVAL_STEPS": "2000",
    "LEARNING_RATE": "1e-5",
    "LR_SCHEDULER_TYPE": "constant_with_warmup",
    "WARMUP_RATIO": "0.0",
    "GUIDED_ATTN_WARMUP_STEPS": "0",
    "GUIDED_ATTN_LOSS_WEIGHT": "0.05",
    "PHONEME_CLASSIFIER_LOSS_WEIGHT": "0.02",
    "CONTENT_CTC_WEIGHT": "0.0",
    "GPU_COUNT": "8",
    "GRADIENT_ACCUMULATION_STEPS": "8",
    "CONTENT_CROSS_ATTN_LAYERS": "all",
    "ENABLE_CONTENT_CROSS_ATTN": "1",
    "USE_TIMBRE_MEMORY": "0",
    "ENABLE_SOURCE_SEMANTIC_MEMORY": "0",
    "ENABLE_SPEAKER_SIDE_PATHWAY": "0",
    "ENABLE_SPEAKER_CROSS_ATTN": "0",
    "TARGET_FRONT_CE_WEIGHT": "4.0",
    "TARGET_FRONT_CE_SECONDS": "0.75",
    "PROGRESS_LOSS_WEIGHT": "0.10",
    "STOP_LOSS_WEIGHT": "0.20",
}
errors = [f"{key}={current.get(key)!r}, expected {wanted!r}" for key, wanted in expected.items() if current.get(key) != wanted]
allowed_diffs = {
    "BATCH_ID", "JOB_NAME_PREFIX", "OUT_DIR", "MAX_TRAIN_STEPS",
    "WARMUP_RATIO", "GUIDED_ATTN_WARMUP_STEPS",
}
diffs = {key for key in set(current) | set(source) if current.get(key) != source.get(key)}
if diffs != allowed_diffs:
    errors.append(f"unexpected deltas versus source r3 config: {sorted(diffs)}; allowed={sorted(allowed_diffs)}")
text = runner.read_text(encoding="utf-8")
for needle in (resume, f'OUT_DIR="{out_dir}"', '--resume-adapter-path $RESUME_ADAPTER_PATH'):
    if needle not in text:
        errors.append(f"generated runner missing {needle!r}")
if errors:
    raise SystemExit("generated continuation config audit failed:\n- " + "\n- ".join(errors))
payload = {
    "status": "pass",
    "current_core_sha256": hashlib.sha256(current_path.read_bytes()).hexdigest(),
    "source_core_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
    "allowed_deltas_vs_source": sorted(allowed_diffs),
    "actual_deltas_vs_source": sorted(diffs),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[batch44-r3-warmstart] generated config PASS audit={output}")
PY
}

qz_payload_dry_run() {
  runner="$RECORD_ROOT/run_train_entrypoint.sh"
  command="sh $runner"
  output_path="$RECORD_ROOT/qz_payload_dry_run.txt"
  set +e
  output=$(
    env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
      HOME="$QZCLI_HOME" QZCLI_GPU_TYPE_OVERRIDE="$QZCLI_GPU_TYPE_OVERRIDE" \
      "$QZCLI" create-job --name "$JOB_NAME" --command "$command" \
        --workspace "$WORKSPACE" --project "$PROJECT" --compute-group "$COMPUTE_GROUP" \
        --spec "$SPEC" --image "$IMAGE" --image-type "$IMAGE_TYPE" --instances "$INSTANCES" \
        --shm "$SHM_GI" --priority "$PRIORITY" --framework "$FRAMEWORK" --dry-run 2>&1
  )
  status=$?
  set -e
  printf '%s\n' "$output" | tee "$output_path"
  [ "$status" -eq 0 ] || { echo "ERROR: qz payload dry-run failed" >&2; return "$status"; }
  python - "$output_path" "$JOB_NAME" "$command" "$EXPECTED_WORKSPACE_ID" "$EXPECTED_PROJECT_ID" \
    "$ALLOWED_COMPUTE_GROUP" "$ALLOWED_SPEC" "$ALLOWED_GPU_TYPE" "$RECORD_ROOT/qz_payload.json" <<'PY'
import json
import re
import sys
from pathlib import Path

source = Path(sys.argv[1])
job, command, workspace, project, group, spec, gpu = sys.argv[2:9]
output = Path(sys.argv[9])
text = source.read_text(encoding="utf-8")
start = text.find("{")
if start < 0:
    raise SystemExit("QZ dry-run lacks JSON payload")
payload, _ = json.JSONDecoder().raw_decode(text[start:])
errors = []
for key, wanted in {
    "name": job, "command": command, "workspace_id": workspace,
    "project_id": project, "logic_compute_group_id": group, "framework": "pytorch",
}.items():
    if payload.get(key) != wanted:
        errors.append(f"{key}={payload.get(key)!r}, expected {wanted!r}")
configs = payload.get("framework_config") or []
if len(configs) != 1:
    errors.append(f"framework_config count={len(configs)}")
else:
    cfg = configs[0]
    resource = cfg.get("resource_spec_price") or {}
    if cfg.get("instance_count") != 1 or cfg.get("gpu_count") != 8:
        errors.append(f"shape={cfg.get('instance_count')}x{cfg.get('gpu_count')}")
    if resource.get("gpu_type") != gpu or resource.get("gpu_count") != 8:
        errors.append(f"gpu resource={resource}")
    if resource.get("logic_compute_group_id") != group or resource.get("quota_id") != spec:
        errors.append(f"compute/spec resource={resource}")
if re.search(r"job-[0-9a-f-]{36}", text, re.I):
    errors.append("dry-run unexpectedly returned job ID")
if errors:
    raise SystemExit("QZ continuation payload audit failed:\n- " + "\n- ".join(errors))
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[batch44-r3-warmstart] QZ payload PASS output={output}")
PY
}

refuse_duplicate_live() {
  if [ -s "$RECORD_ROOT/submitted_jobs.tsv" ]; then
    echo "ERROR: continuation already has a submission ledger" >&2
    exit 1
  fi
  if [ -d "$OUT_DIR" ] && find "$OUT_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "ERROR: continuation output directory is non-empty: $OUT_DIR" >&2
    exit 1
  fi
}

verify_live_ledger() {
  python - "$RECORD_ROOT/submitted_jobs.tsv" "$JOB_NAME" "$COMPUTE_GROUP" "$OUT_DIR" \
    "$RECORD_ROOT/warm_start_contract.json" <<'PY'
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ledger, job, group, out_dir, contract = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], Path(sys.argv[5])
rows = list(csv.DictReader(ledger.open(encoding="utf-8"), delimiter="\t"))
if len(rows) != 1:
    raise SystemExit(f"expected one live ledger row, got {len(rows)}")
row = rows[0]
if row.get("job_name") != job or row.get("compute_group") != group or row.get("out_dir") != out_dir:
    raise SystemExit(f"live ledger identity mismatch: {row}")
job_id = row.get("job_id", "")
if not re.fullmatch(r"job-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", job_id, re.I):
    raise SystemExit(f"invalid job ID: {job_id}")
payload = json.loads(contract.read_text(encoding="utf-8"))
payload["status"] = "submitted"
payload["job_id"] = job_id
payload["warm_start_contract_pre_submit_sha256"] = hashlib.sha256(contract.read_bytes()).hexdigest()
contract.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[batch44-r3-warmstart] live ledger PASS job_id={job_id}")
PY
}

echo "=========================================="
echo "Batch-44 r3 weights-only warm start"
echo "  source=$RESUME_ADAPTER_PATH (effective step $BASE_EFFECTIVE_STEP)"
echo "  job=$JOB_NAME"
echo "  output=$OUT_DIR"
echo "  continuation_steps=$CONTINUATION_STEPS effective_target=$EFFECTIVE_TARGET_STEP"
echo "  data=v1 no_text repeat1 + text repeat3"
echo "  lr=1e-5 scheduler=constant_with_warmup warmup_ratio=0"
echo "  guided_attn_warmup_steps=0"
echo "  compute=MTTS-3-2-0715 only, 1x8 H200"
echo "  LIVE=$LIVE DRY_RUN=$DRY_RUN CONFIRM_WARMSTART=$CONFIRM_WARMSTART"
echo "=========================================="

audit_warmstart_source
generate_runner 1
audit_generated_config
qz_payload_dry_run

if [ "$LIVE" = "0" ]; then
  echo "[batch44-r3-warmstart] audited dry-run complete; no QZ job submitted"
  exit 0
fi

refuse_duplicate_live
LIVE_LOCK="$RECORD_ROOT/.live_submit.lock"
if ! mkdir "$LIVE_LOCK" 2>/dev/null; then
  echo "ERROR: continuation live submit lock exists: $LIVE_LOCK" >&2
  exit 1
fi
trap 'rmdir "$LIVE_LOCK" 2>/dev/null || true' EXIT INT TERM
generate_runner 0
verify_live_ledger
rmdir "$LIVE_LOCK"
trap - EXIT INT TERM
echo "[batch44-r3-warmstart] submitted ledger=$RECORD_ROOT/submitted_jobs.tsv"

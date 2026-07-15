#!/usr/bin/env bash
# Schedule the registered Batch-44 paired full320 checkpoints on the local
# two-RTX-4090 development host.  This watcher has no remote-submit surface.
#
# Safe one-shot status/plan (default):
#   bash scripts/004122_watch_batch44_v1_paired_full320_local.sh
#
# Explicit live monitor.  ACTION=run and the confirmation flag are two
# independent gates.  At most one blocking 004118 evaluation runs at a time:
#   MODE=monitor ACTION=run CONFIRM_LOCAL_FULL320_WATCHER=1 \
#     bash scripts/004122_watch_batch44_v1_paired_full320_local.sh

set -euo pipefail

CANONICAL_PROJECT_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
PROJECT_ROOT="${PROJECT_ROOT:-$CANONICAL_PROJECT_ROOT}"
TEST_MODE="${BATCH44_LOCAL_FULL320_WATCHER_TEST_MODE:-0}"
STAMP="20260713"
MODE="${MODE:-once}"
ACTION="${ACTION:-plan}"
CONFIRM_LOCAL_FULL320_WATCHER="${CONFIRM_LOCAL_FULL320_WATCHER:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_SCANS="${MAX_SCANS:-0}"
MIN_CHECKPOINT_AGE_SEC="${MIN_CHECKPOINT_AGE_SEC:-90}"
STOP_WHEN_COMPLETE="${STOP_WHEN_COMPLETE:-1}"

R3_RUN_DIR="${R3_RUN_DIR:-$PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r3_v1_30k}"
R5_RUN_DIR="${R5_RUN_DIR:-$PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r5_v1_30k}"
R3_TRAIN_JOB_ID="job-2b91d332-d500-4279-84f9-0a6a81a376aa"
R5_TRAIN_JOB_ID="job-b8eb2f1f-a3eb-483b-a289-b4cce281525c"
TRAIN_IDENTITY_ROOT="$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713"
TRAIN_PAIR_LEDGER="$TRAIN_IDENTITY_ROOT/submitted_pair.tsv"

STATE_ROOT="${STATE_ROOT:-$PROJECT_ROOT/trainset/local_jobs/ver23_batch44_paired_full320_scheduler_${STAMP}}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/testset/outputs/ver23_batch44_paired_full320_${STAMP}}"
LOCAL_RUNNER="${LOCAL_RUNNER:-$PROJECT_ROOT/scripts/004118_run_batch44_v1_paired_full320_local.sh}"
QUICK20_VALIDATOR="${QUICK20_VALIDATOR:-$PROJECT_ROOT/scripts/004103_select_batch43_best3.py}"
FULL320_VALIDATOR="${FULL320_VALIDATOR:-$PROJECT_ROOT/scripts/004107_finalize_batch43_pathx_final.py}"
PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
STEPS="10000 20000 30000"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

case "$TEST_MODE:$CONFIRM_LOCAL_FULL320_WATCHER:$STOP_WHEN_COMPLETE" in
  [01]:[01]:[01]) ;;
  *) die "boolean flags must be 0 or 1" ;;
esac
case "$MODE" in once|monitor) ;; *) die "MODE must be once or monitor" ;; esac
case "$ACTION" in plan|run) ;; *) die "ACTION must be plan or run" ;; esac
for value in "$POLL_SECONDS" "$MAX_SCANS" "$MIN_CHECKPOINT_AGE_SEC"; do
  case "$value" in
    ''|*[!0-9]*) die "POLL_SECONDS, MAX_SCANS and MIN_CHECKPOINT_AGE_SEC must be non-negative integers" ;;
  esac
done
[ "$POLL_SECONDS" -gt 0 ] || die "POLL_SECONDS must be positive"
[ -x "$PYTHON" ] || die "Python interpreter is not executable: $PYTHON"
[ -s "$LOCAL_RUNNER" ] || die "missing local paired full320 runner: $LOCAL_RUNNER"
[ -s "$QUICK20_VALIDATOR" ] || die "missing strict quick20 validator: $QUICK20_VALIDATOR"
[ -s "$FULL320_VALIDATOR" ] || die "missing strict full320 validator: $FULL320_VALIDATOR"
bash -n "$LOCAL_RUNNER"

if [ "$PROJECT_ROOT" != "$CANONICAL_PROJECT_ROOT" ] && [ "$TEST_MODE" != "1" ]; then
  die "non-canonical PROJECT_ROOT is allowed only in watcher test mode"
fi
if [ "$ACTION" = "run" ]; then
  [ "$CONFIRM_LOCAL_FULL320_WATCHER" = "1" ] || \
    die "ACTION=run requires CONFIRM_LOCAL_FULL320_WATCHER=1"
  if [ "$TEST_MODE" = "0" ]; then
    [ "$PROJECT_ROOT" = "$CANONICAL_PROJECT_ROOT" ] || die "live watcher requires canonical PROJECT_ROOT"
    [ "$LOCAL_RUNNER" = "$PROJECT_ROOT/scripts/004118_run_batch44_v1_paired_full320_local.sh" ] || \
      die "production local runner is hard-locked to canonical 004118"
  fi
fi

expected_state="$PROJECT_ROOT/trainset/local_jobs/ver23_batch44_paired_full320_scheduler_${STAMP}"
expected_eval="$PROJECT_ROOT/testset/outputs/ver23_batch44_paired_full320_${STAMP}"
if [ "$TEST_MODE" = "0" ]; then
  [ "$STATE_ROOT" = "$expected_state" ] || die "state root must be $expected_state"
  [ "$EVAL_ROOT" = "$expected_eval" ] || die "eval root must be $expected_eval"
fi
[ ! -L "$STATE_ROOT" ] || die "state root may not be a symlink: $STATE_ROOT"
[ ! -L "$EVAL_ROOT" ] || die "eval root may not be a symlink: $EVAL_ROOT"
mkdir -p "$STATE_ROOT" "$EVAL_ROOT"

audit_training_pair() {
  if [ "$TEST_MODE" = "1" ]; then
    return 0
  fi
  "$PYTHON" - "$TRAIN_PAIR_LEDGER" "$R3_RUN_DIR" "$R5_RUN_DIR" \
    "$R3_TRAIN_JOB_ID" "$R5_TRAIN_JOB_ID" "$TRAIN_IDENTITY_ROOT" <<'PY'
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

ledger = Path(sys.argv[1])
r3_dir, r5_dir, r3_job, r5_job = sys.argv[2:6]
identity_root = Path(sys.argv[6])
if not ledger.is_file():
    raise SystemExit(f"missing Batch-44 training pair ledger: {ledger}")
with ledger.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
expected = {
    "ver2_9_5_final_r3_v1_30k": ("r3", r3_job, r3_dir),
    "ver2_9_5_final_r5_v1_30k": ("r5", r5_job, r5_dir),
}
errors: list[str] = []
if len(rows) != 2:
    errors.append(f"expected two training rows, got {len(rows)}")
seen: set[str] = set()
for row in rows:
    name = str(row.get("job_name") or "")
    seen.add(name)
    if name not in expected:
        errors.append(f"unexpected training job {name!r}")
        continue
    arm, job, out_dir = expected[name]
    wanted = {
        "arm": arm,
        "job_id": job,
        "out_dir": out_dir,
        "runner": str(identity_root / arm / "run_train_entrypoint.sh"),
    }
    for key, value in wanted.items():
        if row.get(key) != value:
            errors.append(f"{name} {key}={row.get(key)!r}, expected {value!r}")
if seen != set(expected):
    errors.append(f"training names={sorted(seen)}, expected={sorted(expected)}")

immutable = {
    identity_root / "evaluation_contract.json": "cd41b1f1cb97fb7bd50b5939a6825ca55143835f4c7ad164af42260551b946c1",
    identity_root / "input_identity.full_sha256.json": "accefffbac9aa78b499c5938c9842040dbe44eb6a6188b8a4c444db1b57566b4",
    identity_root / "r3/train_args_dry_run_core.json": "3637b90fe14bbda7c4915362d5bd726b072833f07853df4fc555fde0d2c32d7b",
    identity_root / "r5/train_args_dry_run_core.json": "161f05b766454ce7d3e38af1772a1126ec612eaeb0682678bd6a46ca1d62daff",
    identity_root / "r3/run_train_entrypoint.sh": "09492f6304287918115b01bc0c582c2394d8ba417636c665a57bd30152386b1a",
    identity_root / "r5/run_train_entrypoint.sh": "193e23ce50c5d46ccecaccfb493871d5ddcb773ef2482ca53a0b3fd7b170c207",
}
for path, wanted_sha in immutable.items():
    if not path.is_file():
        errors.append(f"missing immutable training artifact: {path}")
        continue
    with path.open("rb") as handle:
        actual_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual_sha != wanted_sha:
        errors.append(f"training artifact SHA256={actual_sha}, expected {wanted_sha}: {path}")
if errors:
    raise SystemExit("Batch-44 local full320 watcher training audit failed:\n- " + "\n- ".join(errors))
print(f"[batch44-local-full320-watch] training provenance r3={r3_job} r5={r5_job}")
PY
}

checkpoint_probe() {
  local step="$1"
  "$PYTHON" - "$R3_RUN_DIR" "$R5_RUN_DIR" "$PROJECT_ROOT" "$step" \
    "$MIN_CHECKPOINT_AGE_SEC" "$TEST_MODE" <<'PY'
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

r3, r5, project = map(Path, sys.argv[1:4])
step = int(sys.argv[4])
min_age = int(sys.argv[5])
test_mode = sys.argv[6] == "1"
minimum_large = 1 if test_mode else 1_000_000
required = {
    "adapter_model.safetensors": minimum_large,
    "adapter_config.json": 1,
    "README.md": 1,
    "timbre_memory_adapter.pt": minimum_large,
    "timbre_memory_config.json": 1,
}


def equal(got: object, wanted: object) -> bool:
    if isinstance(wanted, float):
        try:
            return math.isclose(float(got), wanted, rel_tol=0.0, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return got == wanted


def probe(arm: str, run_dir: Path, repeat: int) -> tuple[bool, str]:
    checkpoint = run_dir / f"step-{step}"
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        return False, f"{arm}:missing_or_symlink"
    newest = 0.0
    for name, minimum in required.items():
        path = checkpoint / name
        if not path.is_file() or path.is_symlink():
            return False, f"{arm}:missing_or_symlink:{name}"
        stat = path.stat()
        newest = max(newest, stat.st_mtime)
        if stat.st_size < minimum:
            return False, f"{arm}:small:{name}:{stat.st_size}"
    age = time.time() - newest
    if age < min_age:
        return False, f"{arm}:settling:{age:.0f}s<{min_age}s"
    try:
        json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
        cfg = json.loads((checkpoint / "timbre_memory_config.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, f"{arm}:invalid_json:{exc}"
    expected_cfg = {
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
    bad_cfg = [key for key, wanted in expected_cfg.items() if not equal(cfg.get(key), wanted)]
    if bad_cfg:
        return False, f"{arm}:config:" + ",".join(bad_cfg)
    args_path = (
        project
        / "trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713"
        / arm
        / "train_args_dry_run_core.json"
    )
    if not args_path.is_file():
        return False, f"{arm}:missing:train_args_dry_run_core.json"
    with args_path.open("rb") as handle:
        actual_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    if not test_mode:
        wanted_sha = {
            "r3": "3637b90fe14bbda7c4915362d5bd726b072833f07853df4fc555fde0d2c32d7b",
            "r5": "161f05b766454ce7d3e38af1772a1126ec612eaeb0682678bd6a46ca1d62daff",
        }[arm]
        if actual_sha != wanted_sha:
            return False, f"{arm}:identity_sha256_mismatch:{actual_sha}"
    try:
        args = json.loads(args_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, f"{arm}:invalid_args:{exc}"
    no_text = project / "trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/no_text.train.jsonl"
    text = project / "trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/text.train.jsonl"
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
    bad_args = [key for key, wanted in expected_args.items() if args.get(key) != wanted]
    if bad_args:
        return False, f"{arm}:identity:" + ",".join(bad_args)
    return True, f"{arm}:ready:age={age:.0f}s"


r3_ok, r3_reason = probe("r3", r3, 3)
r5_ok, r5_reason = probe("r5", r5, 5)
print("ready" if r3_ok and r5_ok else "waiting")
print(r3_reason)
print(r5_reason)
PY
}

scan_evidence() {
  "$PYTHON" - "$PROJECT_ROOT" "$STATE_ROOT" "$EVAL_ROOT" "$STAMP" "$STEPS" \
    "$QUICK20_VALIDATOR" "$FULL320_VALIDATOR" \
    "$R3_TRAIN_JOB_ID" "$R5_TRAIN_JOB_ID" <<'PY'
from __future__ import annotations

import csv
import datetime as dt
import importlib.util
import json
import os
import sys
from pathlib import Path

project = Path(sys.argv[1]).resolve()
state_root = Path(sys.argv[2]).resolve()
eval_root = Path(sys.argv[3]).resolve()
stamp = sys.argv[4]
steps = [int(value) for value in sys.argv[5].split()]
quick_path = Path(sys.argv[6]).resolve()
full_path = Path(sys.argv[7]).resolve()
r3_job, r5_job = sys.argv[8:10]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


quick = load_module("batch44_local_full320_quick_validator", quick_path)
full = load_module("batch44_local_full320_completion_validator", full_path)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def lexists(path: Path) -> bool:
    return os.path.lexists(path)


rows: list[dict[str, object]] = []
all_full_metrics: list[dict[str, object]] = []
negative: list[dict[str, object]] = []
earlier_incomplete = False
for step in steps:
    quick_local = project / f"trainset/local_jobs/ver23_batch44_quick20_step{step}_{stamp}"
    quick_remote_legacy = project / f"trainset/qz_jobs/ver23_batch44_quick20_step{step}_{stamp}"
    full_local = project / f"trainset/local_jobs/ver23_batch44_paired_full320_step{step}_{stamp}"
    full_remote_legacy = project / f"trainset/qz_jobs/ver23_batch44_paired_full320_step{step}_{stamp}"
    step_root = eval_root / f"step-{step}"

    if lexists(quick_remote_legacy):
        raise SystemExit(
            f"step-{step} quick20 must use local_jobs; unexpected legacy/remote record: {quick_remote_legacy}"
        )
    if lexists(full_remote_legacy):
        raise SystemExit(
            f"step-{step} full320 must use local_jobs; unexpected legacy/remote record: {full_remote_legacy}"
        )
    for root, label in ((quick_local, "quick20"), (full_local, "full320"), (step_root, "full320 step root")):
        if root.is_symlink():
            raise SystemExit(f"step-{step} {label} root may not be a symlink: {root}")

    # Strictly classify the local quick20 prerequisite.
    quick_parts = [
        quick_local / "COMPLETED.json",
        quick_local / "complete.marker",
        quick_local / "metrics.json",
        quick_local / "metrics.tsv",
        quick_local / "metrics.md",
    ]
    quick_forbidden = [quick_local / "submitted_jobs.tsv", quick_local / ".live_submit.lock"]
    if any(lexists(path) for path in quick_forbidden):
        raise SystemExit(f"step-{step} local quick20 contains remote ledger/lock evidence: {quick_local}")
    quick_any = any(lexists(path) for path in quick_parts)
    quick_all = all(path.is_file() and path.stat().st_size > 0 for path in quick_parts)
    quick_lock = quick_local / ".local_quick20.lock"
    if quick_all:
        metrics = quick.load_metrics(quick_local / "metrics.json", project_root=project, step=step)
        provenance = quick.audit_quick20_provenance(
            quick_local / "metrics.json", project_root=project, step=step
        )
        if provenance.get("backend") != "local":
            raise SystemExit(f"step-{step} quick20 backend is not local")
        quick_status = "complete"
        for arm in ("r3", "r5"):
            item = metrics[(arm, "no_text")]
            margin = float(item["margin"])
            if margin < 0.0:
                negative.append({
                    "step": step,
                    "arm": arm,
                    "margin": margin,
                    "sim_ref": float(item["sim_ref"]),
                    "sim_src": float(item["sim_src"]),
                    "cer": float(item["cer"]),
                    "quick20_completion": provenance["completion_json"],
                    "training_job_id": r3_job if arm == "r3" else r5_job,
                })
    elif quick_lock.is_dir():
        required_running = [
            quick_local / "LOCAL_RUNTIME.json",
            quick_local / "004117_run_batch44_v1_quick20_local.frozen.sh",
        ]
        if not all(path.is_file() and path.stat().st_size > 0 for path in required_running):
            raise SystemExit(f"step-{step} quick20 lock lacks local runtime/runner provenance")
        quick_status = "running"
    elif quick_any:
        raise SystemExit(f"step-{step} has partial quick20 completion evidence: {quick_local}")
    elif lexists(quick_local) and (not quick_local.is_dir() or any(quick_local.iterdir())):
        raise SystemExit(f"step-{step} has unbound quick20 record evidence: {quick_local}")
    else:
        quick_status = "pending"

    # Strictly classify the paired full320 result.  A persistent local lock is
    # running/manual-audit evidence; it never authorizes a second invocation.
    full_parts = [
        full_local / "COMPLETED.json",
        full_local / "complete.marker",
        step_root / "aggregate/paired_metrics.json",
        step_root / "aggregate/paired_metrics.tsv",
        step_root / "aggregate/paired_metrics.md",
    ]
    full_forbidden = [full_local / "submitted_jobs.tsv", full_local / ".live_submit.lock"]
    if any(lexists(path) for path in full_forbidden):
        raise SystemExit(f"step-{step} local full320 contains remote ledger/lock evidence: {full_local}")
    full_any = any(lexists(path) for path in full_parts)
    full_all = all(path.is_file() and path.stat().st_size > 0 for path in full_parts)
    full_lock = full_local / ".local_run.lock"
    if full_all:
        completion, indexed = full.validate_full320_step(
            step=step,
            completion_path=full_local / "COMPLETED.json",
            metrics_path=step_root / "aggregate/paired_metrics.json",
            project_root=project,
        )
        if completion.get("backend") != "local":
            raise SystemExit(f"step-{step} full320 backend is not local")
        full_status = "complete"
        for arm in ("r3", "r5"):
            for scope in ("no_text", "text", "all"):
                all_full_metrics.append(dict(indexed[(arm, scope)]))
    elif full_lock.is_dir():
        owner = full_lock / "owner.json"
        if not owner.is_file() or owner.stat().st_size <= 0:
            raise SystemExit(f"step-{step} full320 lock lacks owner provenance: {full_lock}")
        full_status = "running"
    elif full_any:
        raise SystemExit(f"step-{step} has partial full320 completion evidence: {full_local}")
    elif lexists(full_local) and (not full_local.is_dir() or any(full_local.iterdir())):
        raise SystemExit(f"step-{step} has unbound full320 record evidence: {full_local}")
    else:
        full_status = "pending"

    if earlier_incomplete and full_status == "complete":
        raise SystemExit(f"step-{step} completed before an earlier registered full320 step")
    if full_status != "complete":
        earlier_incomplete = True
    rows.append({
        "step": step,
        "quick20_status": quick_status,
        "full320_status": full_status,
        "quick20_record": str(quick_local),
        "full320_record": str(full_local),
    })

if negative:
    generated = dt.datetime.now(dt.timezone.utc).isoformat()
    alert = {
        "schema": "batch44_v1_local_full320_quick20_negative_margin_alert_v1",
        "status": "alert",
        "generated_utc": generated,
        "trigger": "strictly validated same-step local quick20 no_text WavLM margin < 0",
        "scheduler_action": "do not start this or any later paired full320 evaluation",
        "training_action": "report only; watcher never stops or mutates training jobs",
        "training_jobs": {"r3": r3_job, "r5": r5_job},
        "alerts": negative,
    }
    atomic_json(state_root / "ALERT_NEGATIVE_NO_TEXT_MARGIN.json", alert)
    (state_root / "STOP_RECOMMENDATION.md").write_text(
        "# Batch-44 evaluation stop recommendation\n\n"
        "A strictly validated local quick20 no_text margin is negative.\n"
        "The local full320 watcher will not schedule this or later evaluations.\n"
        "Neither training job was stopped or modified.\n",
        encoding="utf-8",
    )

first_incomplete = next((row for row in rows if row["full320_status"] != "complete"), None)
summary = {
    "schema": "batch44_v1_local_paired_full320_watcher_scan_v1",
    "status": "alert" if negative else "ok",
    "steps": steps,
    "completed": sum(row["full320_status"] == "complete" for row in rows),
    "total": len(rows),
    "alert": bool(negative),
    "first_incomplete": first_incomplete,
}
atomic_json(state_root / "scan_latest.json", rows)
atomic_json(state_root / "scan_summary.json", summary)
with (state_root / "scan_latest.tsv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
if all_full_metrics:
    atomic_json(eval_root / "paired_metrics_all.json", all_full_metrics)
    with (eval_root / "paired_metrics_all.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_full_metrics[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(all_full_metrics)
print(
    f"[batch44-local-full320-audit] complete={summary['completed']}/{summary['total']} "
    f"alert={summary['alert']}"
)
PY
}

summary_value() {
  local key="$1"
  "$PYTHON" - "$STATE_ROOT/scan_summary.json" "$key" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))[sys.argv[2]]
print("1" if value is True else "0" if value is False else value)
PY
}

first_incomplete() {
  "$PYTHON" - "$STATE_ROOT/scan_summary.json" <<'PY'
import json
import sys
from pathlib import Path

row = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("first_incomplete")
if isinstance(row, dict):
    print(row["step"])
    print(row["quick20_status"])
    print(row["full320_status"])
PY
}

append_action() {
  local step="$1" result="$2"
  local ledger="$STATE_ROOT/actions.tsv"
  if [ ! -s "$ledger" ]; then
    printf 'utc\tstep\taction\tresult\n' > "$ledger"
  fi
  printf '%s\t%s\trun_local_full320\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$step" "$result" >> "$ledger"
}

run_scan() {
  local scan="$1" first step quick_status full_status probe ready detail_r3 detail_r5 post
  scan_evidence
  cp "$STATE_ROOT/scan_latest.tsv" "$STATE_ROOT/scan_${scan}.tsv"
  echo "[batch44-local-full320-watch] scan=$scan action=$ACTION utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  cat "$STATE_ROOT/scan_latest.tsv"
  if [ "$(summary_value alert)" = "1" ]; then
    echo "[batch44-local-full320-watch] ALERT: negative quick20 margin; full320 scheduling stopped; training untouched"
    return 0
  fi
  first="$(first_incomplete)"
  step="$(printf '%s\n' "$first" | sed -n '1p')"
  quick_status="$(printf '%s\n' "$first" | sed -n '2p')"
  full_status="$(printf '%s\n' "$first" | sed -n '3p')"
  if [ -z "$step" ]; then
    echo "[batch44-local-full320-watch] all registered paired full320 checkpoints complete"
    return 0
  fi
  if [ "$full_status" = "running" ]; then
    echo "[batch44-local-full320-watch] step-$step local full320 is running or locked for manual audit"
    return 0
  fi
  [ "$full_status" = "pending" ] || die "unexpected full320 status=$full_status step=$step"
  case "$quick_status" in
    pending)
      echo "[batch44-local-full320-watch] waiting for strict local quick20 completion at step-$step"
      return 0
      ;;
    running)
      echo "[batch44-local-full320-watch] step-$step local quick20 is still running"
      return 0
      ;;
    complete) ;;
    *) die "unexpected quick20 status=$quick_status step=$step" ;;
  esac

  probe="$(checkpoint_probe "$step")"
  ready="$(printf '%s\n' "$probe" | sed -n '1p')"
  detail_r3="$(printf '%s\n' "$probe" | sed -n '2p')"
  detail_r5="$(printf '%s\n' "$probe" | sed -n '3p')"
  if [ "$ready" != "ready" ]; then
    echo "[batch44-local-full320-watch] waiting checkpoint step=$step $detail_r3 $detail_r5"
    return 0
  fi
  if [ "$ACTION" = "plan" ]; then
    echo "[batch44-local-full320-watch] next_ready_step=$step backend=local (plan only)"
    echo "[batch44-local-full320-watch] would run: STEP=$step ACTION=run CONFIRM_LOCAL_FULL320=1 bash $LOCAL_RUNNER"
    return 0
  fi

  echo "[batch44-local-full320-watch] RUN step=$step via canonical 004118; training jobs are read-only"
  local runner_rc=0
  STEP="$step" ACTION=run CONFIRM_LOCAL_FULL320=1 \
    MIN_CHECKPOINT_AGE_SEC="$MIN_CHECKPOINT_AGE_SEC" \
    PROJECT_ROOT="$PROJECT_ROOT" R3_RUN_DIR="$R3_RUN_DIR" R5_RUN_DIR="$R5_RUN_DIR" \
    EVAL_ROOT="$EVAL_ROOT" bash "$LOCAL_RUNNER" || runner_rc=$?
  if [ "$runner_rc" -ne 0 ]; then
    append_action "$step" "failed_rc_$runner_rc"
    die "004118 failed rc=$runner_rc at step-$step; persistent local evidence requires manual audit"
  fi
  scan_evidence
  post="$($PYTHON - "$STATE_ROOT/scan_latest.json" "$step" <<'PY'
import json
import sys
from pathlib import Path

rows = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
step = int(sys.argv[2])
matches = [row for row in rows if int(row["step"]) == step]
if len(matches) != 1:
    raise SystemExit(f"post-run step identity drift: {matches}")
print(matches[0]["full320_status"])
PY
)"
  if [ "$post" != "complete" ]; then
    append_action "$step" "zero_exit_without_strict_completion"
    die "004118 returned zero without a strictly validated local completion at step-$step"
  fi
  append_action "$step" success
  echo "[batch44-local-full320-watch] strictly accepted step-$step local completion"
}

audit_training_pair

LOCK_DIR="$STATE_ROOT/.watch.lock"
PID_FILE="$STATE_ROOT/monitor.pid"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  die "another local paired-full320 watcher appears active: $LOCK_DIR"
fi
cleanup() {
  if [ -s "$PID_FILE" ] && [ "$(cat "$PID_FILE" 2>/dev/null || true)" = "$$" ]; then
    rm -f "$PID_FILE"
  fi
  rm -f "$LOCK_DIR/owner.txt"
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
printf 'pid=%s host=%s mode=%s action=%s started=%s\n' \
  "$$" "$(hostname)" "$MODE" "$ACTION" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCK_DIR/owner.txt"
printf '%s\n' "$$" > "$PID_FILE"

scan=0
while :; do
  scan=$((scan + 1))
  run_scan "$scan"
  if [ "$(summary_value alert)" = "1" ]; then
    break
  fi
  if [ "$STOP_WHEN_COMPLETE" = "1" ] && [ "$(summary_value completed)" = "3" ]; then
    break
  fi
  if [ "$MODE" = "once" ]; then
    break
  fi
  if [ "$MAX_SCANS" -gt 0 ] && [ "$scan" -ge "$MAX_SCANS" ]; then
    echo "[batch44-local-full320-watch] reached MAX_SCANS=$MAX_SCANS"
    break
  fi
  sleep "$POLL_SECONDS"
done

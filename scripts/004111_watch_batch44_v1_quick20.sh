#!/usr/bin/env bash
# Discover and schedule Batch-44 r3/r5 paired quick20 checkpoints.
#
# Safe defaults:
#   MODE=once ACTION=plan bash scripts/004111_watch_batch44_v1_quick20.sh
#
# Platform dry-run every ready 2k checkpoint, serialized in step order:
#   MODE=monitor ACTION=dry-run bash scripts/004111_watch_batch44_v1_quick20.sh
#
# Live monitoring/submission requires two explicit gates.  It submits at most
# one QZ evaluation at a time and will not advance until its complete.marker
# appears.  Every evaluation is delegated to 004110, which hard-locks the
# MTTS-3-2-0715 1x8-H200 resource.
#   MODE=monitor ACTION=submit ALLOW_LIVE_SUBMIT=1 \
#     bash scripts/004111_watch_batch44_v1_quick20.sh
#
# Detached launch (the script maintains monitor.pid while alive):
#   STATE_ROOT=/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/trainset/qz_jobs/ver23_batch44_quick20_scheduler_20260713
#   mkdir -p "$STATE_ROOT"
#   setsid env MODE=monitor ACTION=submit ALLOW_LIVE_SUBMIT=1 \
#     bash scripts/004111_watch_batch44_v1_quick20.sh \
#     >> "$STATE_ROOT/monitor.log" 2>&1 < /dev/null &

set -euo pipefail

CANONICAL_PROJECT_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
PROJECT_ROOT="${PROJECT_ROOT:-$CANONICAL_PROJECT_ROOT}"
TEST_MODE="${BATCH44_QUICK20_TEST_MODE:-0}"
STAMP="20260713"
MODE="${MODE:-once}"
ACTION="${ACTION:-plan}"
ALLOW_LIVE_SUBMIT="${ALLOW_LIVE_SUBMIT:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_SCANS="${MAX_SCANS:-0}"
MIN_CHECKPOINT_AGE_SEC="${MIN_CHECKPOINT_AGE_SEC:-90}"
STOP_WHEN_COMPLETE="${STOP_WHEN_COMPLETE:-1}"
ALERT_TRIGGERED=0
VALIDATED_COMPLETION_STEPS=""

R3_RUN_DIR="${R3_RUN_DIR:-$PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r3_v1_30k}"
R5_RUN_DIR="${R5_RUN_DIR:-$PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r5_v1_30k}"
R3_TRAIN_JOB_ID="job-2b91d332-d500-4279-84f9-0a6a81a376aa"
R5_TRAIN_JOB_ID="job-b8eb2f1f-a3eb-483b-a289-b4cce281525c"
TRAIN_PAIR_LEDGER="$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713/submitted_pair.tsv"
STATE_ROOT="${STATE_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_quick20_scheduler_${STAMP}}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/testset/outputs/ver23_batch44_quick20_${STAMP}}"
SUBMIT_WRAPPER="${SUBMIT_WRAPPER:-$PROJECT_ROOT/scripts/004110_submit_batch44_v1_quick20_qz.sh}"
PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
STEPS="2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 22000 24000 26000 28000 30000"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

case "$TEST_MODE:$ALLOW_LIVE_SUBMIT:$STOP_WHEN_COMPLETE" in
  [01]:[01]:[01]) ;;
  *) die "boolean flags must be 0 or 1" ;;
esac
case "$MODE" in once|monitor) ;; *) die "MODE must be once or monitor" ;; esac
case "$ACTION" in plan|dry-run|submit) ;; *) die "ACTION must be plan, dry-run, or submit" ;; esac
for numeric_value in "$POLL_SECONDS" "$MAX_SCANS" "$MIN_CHECKPOINT_AGE_SEC"; do
  case "$numeric_value" in
    ''|*[!0-9]*) die "POLL_SECONDS, MAX_SCANS, and MIN_CHECKPOINT_AGE_SEC must be non-negative integers" ;;
  esac
done
[ "$POLL_SECONDS" -gt 0 ] || die "POLL_SECONDS must be positive"
if [ "$PROJECT_ROOT" != "$CANONICAL_PROJECT_ROOT" ] && [ "$TEST_MODE" != "1" ]; then
  die "non-canonical PROJECT_ROOT is allowed only with BATCH44_QUICK20_TEST_MODE=1"
fi
if [ "$ACTION" = "submit" ]; then
  [ "$ALLOW_LIVE_SUBMIT" = "1" ] || die "ACTION=submit requires ALLOW_LIVE_SUBMIT=1"
  [ "$TEST_MODE" = "0" ] || die "test mode may not submit live jobs"
  [ "$PROJECT_ROOT" = "$CANONICAL_PROJECT_ROOT" ] || die "live submission requires canonical PROJECT_ROOT"
fi
[ -x "$PYTHON" ] || die "Python interpreter is not executable: $PYTHON"
if [ "$ACTION" != "plan" ]; then
  [ -s "$SUBMIT_WRAPPER" ] || die "missing Batch-44 quick20 submit wrapper: $SUBMIT_WRAPPER"
  bash -n "$SUBMIT_WRAPPER"
fi

mkdir -p "$STATE_ROOT" "$EVAL_ROOT"
if [ -s "$STATE_ROOT/QZ_SUBMISSIONS_DISABLED.json" ] && [ "$ACTION" != "plan" ]; then
  die "QZ quick20 scheduling is disabled by the local-4090 evaluation policy: $STATE_ROOT/QZ_SUBMISSIONS_DISABLED.json"
fi

audit_training_pair() {
  if [ "$TEST_MODE" = "1" ]; then
    return 0
  fi
  "$PYTHON" - "$TRAIN_PAIR_LEDGER" "$R3_RUN_DIR" "$R5_RUN_DIR" \
    "$R3_TRAIN_JOB_ID" "$R5_TRAIN_JOB_ID" <<'PY'
import csv
import sys
from pathlib import Path

ledger = Path(sys.argv[1])
r3_out, r5_out, r3_job, r5_job = sys.argv[2:6]
if not ledger.is_file():
    raise SystemExit(f"missing Batch-44 training pair ledger: {ledger}")
with ledger.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
expected = {
    "ver2_9_5_final_r3_v1_30k": (r3_job, r3_out),
    "ver2_9_5_final_r5_v1_30k": (r5_job, r5_out),
}
errors = []
if len(rows) != 2:
    errors.append(f"expected two rows, got {len(rows)}")
seen = set()
for row in rows:
    name = row.get("job_name", "")
    seen.add(name)
    if name not in expected:
        errors.append(f"unexpected job name={name!r}")
        continue
    job, out_dir = expected[name]
    if row.get("job_id") != job or row.get("out_dir") != out_dir:
        errors.append(f"training provenance drift for {name}: {row}")
if seen != set(expected):
    errors.append(f"training names={sorted(seen)}, expected {sorted(expected)}")
if errors:
    raise SystemExit("Batch-44 watcher training audit failed:\n- " + "\n- ".join(errors))
print(f"[batch44-quick20-watch] training provenance r3={r3_job} r5={r5_job}")
PY
}

audit_training_pair

record_root_for() {
  printf '%s/trainset/qz_jobs/ver23_batch44_quick20_step%s_%s\n' "$PROJECT_ROOT" "$1" "$STAMP"
}

validate_completion_record() {
  local step="$1"
  local record
  record="$(record_root_for "$step")"
  "$PYTHON" - "$record" "$step" "$STAMP" <<'PY'
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

record = Path(sys.argv[1]).resolve()
step = int(sys.argv[2])
stamp = sys.argv[3]
completion_path = record / "COMPLETED.json"
marker_path = record / "complete.marker"
ledger_path = record / "submitted_jobs.tsv"
job_re = re.compile(
    r"^job-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def load_object(path: Path, label: str) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing/empty {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - fail closed with source path
        raise SystemExit(f"invalid {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing/empty completion artifact: {path}")
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_artifact(meta: object, expected_path: Path, label: str) -> None:
    if not isinstance(meta, dict):
        raise SystemExit(f"{label} artifact metadata must be an object")
    try:
        recorded_path = Path(str(meta["path"])).resolve()
        recorded_size = int(meta["size"])
        recorded_sha = str(meta["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid {label} artifact metadata: {meta!r}") from exc
    expected_path = expected_path.resolve()
    if recorded_path != expected_path:
        raise SystemExit(
            f"{label} artifact path mismatch: {recorded_path} != {expected_path}"
        )
    if not expected_path.is_file() or expected_path.stat().st_size <= 0:
        raise SystemExit(f"missing/empty {label} artifact: {expected_path}")
    actual_size = expected_path.stat().st_size
    if recorded_size != actual_size:
        raise SystemExit(
            f"{label} artifact size mismatch: completion={recorded_size} actual={actual_size}"
        )
    actual_sha = sha256(expected_path)
    if recorded_sha != actual_sha:
        raise SystemExit(
            f"{label} artifact SHA mismatch: completion={recorded_sha} actual={actual_sha}"
        )


completion = load_object(completion_path, "COMPLETED.json")
expected_completion = {
    "schema": "moss_codecvc.batch44_v1_quick20_completion.v1",
    "status": "complete",
    "step": step,
    "record_root": str(record),
}
bad_completion = {
    key: {"expected": wanted, "actual": completion.get(key)}
    for key, wanted in expected_completion.items()
    if completion.get(key) != wanted
}
if bad_completion:
    raise SystemExit(f"COMPLETED.json identity mismatch: {bad_completion}")

marker = load_object(marker_path, "complete.marker")
expected_marker = {
    "schema": "moss_codecvc.batch44_v1_quick20_complete_marker.v1",
    "status": "complete",
    "step": step,
}
bad_marker = {
    key: {"expected": wanted, "actual": marker.get(key)}
    for key, wanted in expected_marker.items()
    if marker.get(key) != wanted
}
if bad_marker:
    raise SystemExit(f"complete.marker identity mismatch: {bad_marker}")
completion_sha = sha256(completion_path)
if marker.get("completed_json_sha256") != completion_sha:
    raise SystemExit(
        "complete.marker COMPLETED.json SHA mismatch: "
        f"marker={marker.get('completed_json_sha256')!r} actual={completion_sha}"
    )

metrics = completion.get("metrics")
if not isinstance(metrics, dict) or set(metrics) != {"json", "tsv", "md"}:
    raise SystemExit(f"COMPLETED.json metrics artifact set is invalid: {metrics!r}")
for suffix in ("json", "tsv", "md"):
    validate_artifact(metrics[suffix], record / f"metrics.{suffix}", f"metrics.{suffix}")

if not ledger_path.is_file() or ledger_path.stat().st_size <= 0:
    raise SystemExit(f"missing/empty submission ledger: {ledger_path}")
with ledger_path.open(encoding="utf-8", newline="") as handle:
    ledger_rows = list(csv.DictReader(handle, delimiter="\t"))
if len(ledger_rows) != 1:
    raise SystemExit(f"submission ledger must contain exactly one row, got {len(ledger_rows)}")
ledger = ledger_rows[0]
expected_job_name = f"ver23_batch44_quick20_step{step}_{stamp}"
if ledger.get("step") != str(step):
    raise SystemExit(
        f"submission ledger step mismatch: {ledger.get('step')!r} != {step}"
    )
if ledger.get("job_name") != expected_job_name:
    raise SystemExit(
        f"submission ledger job_name mismatch: {ledger.get('job_name')!r} != {expected_job_name!r}"
    )
ledger_job_id = str(ledger.get("job_id") or "")
if not job_re.fullmatch(ledger_job_id):
    raise SystemExit(f"invalid submission ledger job_id: {ledger_job_id!r}")

evaluation_job = completion.get("evaluation_job")
if not isinstance(evaluation_job, dict):
    raise SystemExit("COMPLETED.json evaluation_job must be an object")
if evaluation_job.get("job_name") != ledger.get("job_name"):
    raise SystemExit("completion/ledger job_name mismatch")
if evaluation_job.get("job_id") != ledger_job_id:
    raise SystemExit("completion/ledger job_id mismatch")
validate_artifact(
    evaluation_job.get("submission_ledger"),
    ledger_path,
    "submission ledger",
)

print(f"[batch44-quick20-completion-audit] PASS step={step} job={ledger_job_id}")
PY
}

validate_completion_snapshot() {
  local step record
  VALIDATED_COMPLETION_STEPS=""
  for step in $STEPS; do
    record="$(record_root_for "$step")"
    # The producer writes complete.marker last.  Once it exists, every bound
    # completion artifact must validate before this watcher may advance.
    if [ -e "$record/complete.marker" ]; then
      if ! validate_completion_record "$step"; then
        die "invalid Batch-44 quick20 completion record at step-$step; refusing to advance"
      fi
      VALIDATED_COMPLETION_STEPS="$VALIDATED_COMPLETION_STEPS $step"
    fi
  done
}

is_validated_completion_step() {
  case " $VALIDATED_COMPLETION_STEPS " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

checkpoint_probe() {
  local step="$1"
  "$PYTHON" - "$R3_RUN_DIR" "$R5_RUN_DIR" "$PROJECT_ROOT" "$step" "$MIN_CHECKPOINT_AGE_SEC" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

r3, r5, project_root = map(Path, sys.argv[1:4])
step = int(sys.argv[4])
min_age = int(sys.argv[5])
now = time.time()
required = {
    "adapter_model.safetensors": 1_000_000,
    "adapter_config.json": 1,
    "README.md": 1,
    "timbre_memory_adapter.pt": 1_000_000,
    "timbre_memory_config.json": 1,
}

def probe(arm: str, run_dir: Path, repeat: int) -> tuple[bool, str]:
    checkpoint = run_dir / f"step-{step}"
    if not checkpoint.is_dir():
        return False, f"{arm}:missing"
    newest_mtime = 0.0
    for name, minimum_size in required.items():
        path = checkpoint / name
        if not path.is_file():
            return False, f"{arm}:missing:{name}"
        stat = path.stat()
        newest_mtime = max(newest_mtime, stat.st_mtime)
        if stat.st_size < minimum_size:
            return False, f"{arm}:small:{name}:{stat.st_size}"
    age = now - newest_mtime
    if age < min_age:
        return False, f"{arm}:settling:{age:.0f}s<{min_age}s"
    try:
        adapter_cfg = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
        cfg = json.loads((checkpoint / "timbre_memory_config.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - probe reports parse errors verbatim
        return False, f"{arm}:invalid_json:{exc}"
    del adapter_cfg
    expected_cfg = {
        "content_cross_attn_enabled": True,
        "content_cross_attn_layers": "all",
        "content_cross_attn_feature_dim": 768,
        "content_encoder_layers": 2,
        "num_memory_tokens": 0,
        "timbre_side_only": False,
        "source_semantic_memory_enabled": False,
        "speaker_side_pathway_enabled": False,
        "speaker_cross_attn_enabled": False,
    }
    bad = [f"{key}={cfg.get(key)!r}" for key, wanted in expected_cfg.items() if cfg.get(key) != wanted]
    if bad:
        return False, f"{arm}:config:" + ",".join(bad)
    identity_root = (
        project_root
        / "trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713"
    )
    args_path = identity_root / arm / "train_args_dry_run_core.json"
    if not args_path.is_file():
        return False, f"{arm}:missing:train_args_dry_run_core.json"
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
            return False, f"{arm}:identity_sha256_mismatch:{actual_identity_sha}"
    try:
        args = json.loads(args_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, f"{arm}:invalid_args:{exc}"
    no_text = project_root / "trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/no_text.train.jsonl"
    text = project_root / "trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/text.train.jsonl"
    wanted_spec = f"{no_text}::repeat=1,{text}::repeat={repeat}"
    if args.get("TRAIN_JSONL_SPEC") != wanted_spec:
        return False, f"{arm}:train_spec_mismatch"
    expected_identity = {
        "OUT_DIR": str(run_dir),
        "TEXT_REPEAT": str(repeat),
        "MAX_TRAIN_STEPS": "30000",
        "SAVE_STEPS": "2000",
        "EVAL_STEPS": "2000",
        "LEARNING_RATE": "1e-5",
        "LR_SCHEDULER_TYPE": "constant_with_warmup",
        "WARMUP_RATIO": "0.03",
    }
    if any(args.get(key) != wanted for key, wanted in expected_identity.items()):
        return False, f"{arm}:schedule_mismatch"
    return True, f"{arm}:ready:age={age:.0f}s"

r3_ok, r3_reason = probe("r3", r3, 3)
r5_ok, r5_reason = probe("r5", r5, 5)
print("ready" if r3_ok and r5_ok else "waiting")
print(r3_reason)
print(r5_reason)
PY
}

refresh_rollup() {
  "$PYTHON" - "$PROJECT_ROOT" "$STATE_ROOT" "$EVAL_ROOT" "$STAMP" "$STEPS" \
    "$VALIDATED_COMPLETION_STEPS" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

project_root = Path(sys.argv[1])
state_root = Path(sys.argv[2])
eval_root = Path(sys.argv[3])
stamp = sys.argv[4]
steps = [int(value) for value in sys.argv[5].split()]
validated_steps = {int(value) for value in sys.argv[6].split()}
rows = []
status_rows = []
seen = set()
for step in steps:
    record = project_root / f"trainset/qz_jobs/ver23_batch44_quick20_step{step}_{stamp}"
    complete = step in validated_steps
    submitted = False
    ledger = record / "submitted_jobs.tsv"
    if ledger.is_file():
        submitted = "job-" in ledger.read_text(encoding="utf-8", errors="replace")
    dry_run = (record / "dry_run.ok").is_file()
    locked = (record / ".live_submit.lock").is_dir()
    status_rows.append({
        "step": step,
        "r3_train_job_id": "job-2b91d332-d500-4279-84f9-0a6a81a376aa",
        "r5_train_job_id": "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c",
        "complete": complete,
        "submitted": submitted,
        "dry_run": dry_run,
        "live_lock": locked,
        "record_root": str(record),
    })
    if not complete:
        continue
    with (record / "metrics.tsv").open(encoding="utf-8", newline="") as handle:
        step_rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(step_rows) != 4:
        raise SystemExit(f"complete step-{step} must have exactly four metric rows, got {len(step_rows)}")
    for row in step_rows:
        row_step = int(row["step"])
        key = (row_step, row["arm"], row["mode"])
        if row_step != step or key in seen:
            raise SystemExit(f"duplicate/mismatched Batch-44 quick20 metric key: {key}")
        if row["arm"] not in {"r3", "r5"} or row["mode"] not in {"no_text", "text"}:
            raise SystemExit(f"invalid Batch-44 quick20 metric key: {key}")
        seen.add(key)
        rows.append(row)

state_root.mkdir(parents=True, exist_ok=True)
(state_root / "status.json").write_text(json.dumps(status_rows, indent=2) + "\n", encoding="utf-8")
with (state_root / "status.tsv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(status_rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(status_rows)

if rows:
    fields = list(rows[0])
    with (eval_root / "metrics_all.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (eval_root / "metrics_all.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
else:
    for name in ("metrics_all.tsv", "metrics_all.json"):
        path = eval_root / name
        if path.exists():
            path.unlink()

def f(row, key):
    value = row.get(key, "")
    return None if value == "" else float(value)

lines = [
    "# Batch-44 r3/r5 per-2k quick20 rollup",
    "",
    "Protocol: fixed no_text20 + text20. `text en_src fail` is a 12-case quick20 proxy, not the full 80-case gate.",
    "",
    "Training provenance: r3 `job-2b91d332-d500-4279-84f9-0a6a81a376aa`; r5 `job-b8eb2f1f-a3eb-483b-a289-b4cce281525c`.",
    "",
    f"Completed paired checkpoints: {len(rows) // 4}/15.",
    "",
    "| Step | Mode | r3 fail | r5 fail | r3 CER | r5 CER | r3 sim(ref) | r5 sim(ref) | r3 sim(src) | r5 sim(src) | r3 ref-bound | r5 ref-bound | r3 F1 | r5 F1 | r3/r5 text en_src quick fail | Flags |",
    "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]
index = {(int(row["step"]), row["arm"], row["mode"]): row for row in rows}
for step in steps:
    for mode in ("no_text", "text"):
        r3 = index.get((step, "r3", mode))
        r5 = index.get((step, "r5", mode))
        if not r3 or not r5:
            continue
        flags = []
        for arm, row in (("r3", r3), ("r5", r5)):
            if f(row, "cer") > 0.30:
                flags.append(f"{arm}:CER>0.30")
            if f(row, "ref_content_f1") > 0.20:
                flags.append(f"{arm}:F1>0.20")
            if f(row, "margin") < 0.02:
                flags.append(f"{arm}:margin<0.02")
            if mode == "text" and f(row, "text_en_src_quick_fail") > 0.25:
                flags.append(f"{arm}:text_en_src_quick>25%")
        if mode == "text":
            en_src = f"{f(r3, 'text_en_src_quick_fail'):.1%} / {f(r5, 'text_en_src_quick_fail'):.1%} (n=12 each)"
        else:
            en_src = "—"
        lines.append(
            f"| {step} | {mode} | {f(r3, 'fail'):.1%} | {f(r5, 'fail'):.1%} | "
            f"{f(r3, 'cer'):.4f} | {f(r5, 'cer'):.4f} | "
            f"{f(r3, 'sim_ref'):.4f} | {f(r5, 'sim_ref'):.4f} | "
            f"{f(r3, 'sim_src'):.4f} | {f(r5, 'sim_src'):.4f} | "
            f"{f(r3, 'ref_bound'):.1%} | {f(r5, 'ref_bound'):.1%} | "
            f"{f(r3, 'ref_content_f1'):.4f} | {f(r5, 'ref_content_f1'):.4f} | "
            f"{en_src} | {', '.join(flags) if flags else '—'} |"
        )
(eval_root / "metrics_all.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[batch44-quick20-rollup] complete_steps={len(rows) // 4}/15 path={eval_root / 'metrics_all.md'}")
PY
}

detect_negative_no_text_margin() {
  "$PYTHON" - \
    "$PROJECT_ROOT" "$STATE_ROOT" "$EVAL_ROOT" "$STAMP" "$STEPS" \
    "$VALIDATED_COMPLETION_STEPS" "$R3_TRAIN_JOB_ID" "$R5_TRAIN_JOB_ID" <<'PY'
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
from pathlib import Path

project_root = Path(sys.argv[1])
state_root = Path(sys.argv[2])
eval_root = Path(sys.argv[3])
stamp = sys.argv[4]
steps = [int(value) for value in sys.argv[5].split()]
validated_steps = {int(value) for value in sys.argv[6].split()}
r3_job, r5_job = sys.argv[7:9]
alerts = []
for step in steps:
    if step not in validated_steps:
        continue
    record = project_root / f"trainset/qz_jobs/ver23_batch44_quick20_step{step}_{stamp}"
    metrics = record / "metrics.tsv"
    marker = record / "complete.marker"
    if not metrics.is_file() or not marker.is_file():
        raise SystemExit(f"validated completion artifacts disappeared for step-{step}")
    with metrics.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 4:
        raise SystemExit(
            f"completed step-{step} must contain four metric rows, got {len(rows)}"
        )
    for row in rows:
        if row.get("mode") != "no_text":
            continue
        arm = row.get("arm", "")
        if arm not in {"r3", "r5"}:
            raise SystemExit(f"invalid no_text arm in {metrics}: {arm!r}")
        try:
            margin = float(row["margin"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"invalid margin in {metrics}: {row}") from exc
        if margin < 0.0:
            alerts.append({
                "step": step,
                "arm": arm,
                "margin": margin,
                "sim_ref": float(row["sim_ref"]),
                "sim_src": float(row["sim_src"]),
                "cer": float(row["cer"]),
                "record_root": str(record),
                "metrics_tsv": str(metrics),
                "training_job_id": r3_job if arm == "r3" else r5_job,
            })
if not alerts:
    raise SystemExit(1)

generated = dt.datetime.now(dt.timezone.utc).isoformat()
payload = {
    "schema": "batch44_v1_quick20_negative_margin_alert_v1",
    "status": "alert",
    "generated_utc": generated,
    "trigger": "any completed quick20 no_text WavLM margin < 0",
    "scheduler_action": "stop scheduling further quick20 evaluations",
    "training_action": "recommend stop only; watcher does not stop training jobs",
    "training_jobs": {"r3": r3_job, "r5": r5_job},
    "alerts": alerts,
}
state_root.mkdir(parents=True, exist_ok=True)
eval_root.mkdir(parents=True, exist_ok=True)
rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
for root in (state_root, eval_root):
    (root / "ALERT_NEGATIVE_NO_TEXT_MARGIN.json").write_text(rendered, encoding="utf-8")

lines = [
    "# ALERT: Batch-44 no_text WavLM margin below zero",
    "",
    f"Generated: `{generated}`",
    "",
    "The quick20 scheduler has stopped and will not submit later checkpoints.",
    "It has **not** called QZ stop and has not changed either training job.",
    "",
    "| Step | Arm | margin | sim(ref) | sim(src) | CER | training job |",
    "|---:|---|---:|---:|---:|---:|---|",
]
for item in alerts:
    lines.append(
        f"| {item['step']} | {item['arm']} | {item['margin']:.4f} | "
        f"{item['sim_ref']:.4f} | {item['sim_src']:.4f} | "
        f"{item['cer']:.4f} | `{item['training_job_id']}` |"
    )
alert_md = "\n".join(lines) + "\n"
recommendation = "\n".join([
    "# Batch-44 stop recommendation",
    "",
    "A completed no_text quick20 has WavLM `sim(ref)-sim(src) < 0`.",
    "Per the registered policy, stop further quick20 scheduling immediately.",
    "The main thread should verify the complete metrics and QZ training status,",
    "then decide whether to stop the affected arm or both paired training arms.",
    "",
    "This watcher deliberately performs no automatic training stop operation.",
    "",
])
for root in (state_root, eval_root):
    (root / "ALERT_NEGATIVE_NO_TEXT_MARGIN.md").write_text(alert_md, encoding="utf-8")
    (root / "STOP_RECOMMENDATION.md").write_text(recommendation, encoding="utf-8")
print(
    "[batch44-quick20-ALERT] negative no_text margin; "
    f"alerts={len(alerts)} state={state_root}"
)
PY
}

write_scan_status() {
  local scan="$1"
  local output="$STATE_ROOT/scan_latest.tsv"
  : > "$output"
  printf 'step\tstatus\tr3_train_job_id\tr5_train_job_id\tdetail_r3\tdetail_r5\trecord_root\n' >> "$output"
  local step record probe first r3_detail r5_detail status job_id
  for step in $STEPS; do
    record="$(record_root_for "$step")"
    if is_validated_completion_step "$step"; then
      status="complete"
      r3_detail="metrics"
      r5_detail="metrics"
    elif [ -s "$record/submitted_jobs.tsv" ] && grep -Eq 'job-[0-9a-fA-F-]{36}' "$record/submitted_jobs.tsv"; then
      status="submitted"
      job_id="$(grep -Eo 'job-[0-9a-fA-F-]{36}' "$record/submitted_jobs.tsv" | tail -n 1)"
      r3_detail="$job_id:waiting_for_complete.marker"
      r5_detail="$job_id:waiting_for_complete.marker"
    elif [ -d "$record/.live_submit.lock" ]; then
      status="locked"
      r3_detail="manual_QZ_audit_required"
      r5_detail="manual_QZ_audit_required"
    elif [ "$ACTION" = "dry-run" ] && [ -s "$record/dry_run.ok" ]; then
      status="dry_run_passed"
      r3_detail="platform_dry_run"
      r5_detail="platform_dry_run"
    else
      probe="$(checkpoint_probe "$step")"
      first="$(printf '%s\n' "$probe" | sed -n '1p')"
      r3_detail="$(printf '%s\n' "$probe" | sed -n '2p')"
      r5_detail="$(printf '%s\n' "$probe" | sed -n '3p')"
      if [ "$first" = "ready" ]; then status="ready"; else status="waiting"; fi
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$step" "$status" "$R3_TRAIN_JOB_ID" "$R5_TRAIN_JOB_ID" "$r3_detail" "$r5_detail" "$record" >> "$output"
  done
  cp "$output" "$STATE_ROOT/scan_${scan}.tsv"
  "$PYTHON" - "$output" "$STATE_ROOT/scan_latest.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

source, output = map(Path, sys.argv[1:])
with source.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
PY
}

select_next_step() {
  "$PYTHON" - "$STATE_ROOT/scan_latest.tsv" "$ACTION" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
action = sys.argv[2]
with path.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
for row in rows:
    status = row["status"]
    if status == "complete":
        continue
    if action == "dry-run" and status == "dry_run_passed":
        continue
    if status == "ready":
        print(row["step"])
    # Serialize strictly: do not skip a missing/submitted/locked earlier step.
    break
PY
}

all_complete() {
  "$PYTHON" - "$STATE_ROOT/scan_latest.tsv" <<'PY'
import csv
import sys
from pathlib import Path

with Path(sys.argv[1]).open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
raise SystemExit(0 if len(rows) == 15 and all(row["status"] == "complete" for row in rows) else 1)
PY
}

run_scan() {
  local scan="$1"
  local next_step=""
  ALERT_TRIGGERED=0
  validate_completion_snapshot
  refresh_rollup
  if detect_negative_no_text_margin; then
    ALERT_TRIGGERED=1
    write_scan_status "$scan"
    echo "[batch44-quick20-watch] ALERT active: scheduling stopped; training jobs were not stopped"
    return 0
  fi
  write_scan_status "$scan"
  echo "[batch44-quick20-watch] scan=$scan action=$ACTION time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  awk -F '\t' 'NR==1 || $2 != "waiting" {print}' "$STATE_ROOT/scan_latest.tsv"

  if [ "$ACTION" = "plan" ]; then
    next_step="$(select_next_step || true)"
    if [ -n "$next_step" ]; then
      echo "[batch44-quick20-watch] next_ready_step=$next_step (plan only; no qzcli call)"
    else
      echo "[batch44-quick20-watch] no schedulable paired checkpoint"
    fi
    return 0
  fi

  next_step="$(select_next_step || true)"
  if [ -z "$next_step" ]; then
    echo "[batch44-quick20-watch] no schedulable paired checkpoint"
    return 0
  fi
  if [ "$ACTION" = "dry-run" ]; then
    echo "[batch44-quick20-watch] platform dry-run step=$next_step"
    STEP="$next_step" DRY_RUN=1 \
      PROJECT_ROOT="$PROJECT_ROOT" R3_RUN_DIR="$R3_RUN_DIR" R5_RUN_DIR="$R5_RUN_DIR" EVAL_ROOT="$EVAL_ROOT" \
      bash "$SUBMIT_WRAPPER"
  else
    echo "[batch44-quick20-watch] LIVE submit step=$next_step"
    STEP="$next_step" DRY_RUN=0 CONFIRM_BATCH44_QUICK20=1 \
      PROJECT_ROOT="$PROJECT_ROOT" R3_RUN_DIR="$R3_RUN_DIR" R5_RUN_DIR="$R5_RUN_DIR" EVAL_ROOT="$EVAL_ROOT" \
      bash "$SUBMIT_WRAPPER"
  fi
}

LOCK_DIR="$STATE_ROOT/.watch.lock"
PID_FILE="$STATE_ROOT/monitor.pid"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  die "another Batch-44 quick20 watcher appears active: $LOCK_DIR"
fi
cleanup() {
  if [ -s "$PID_FILE" ] && [ "$(cat "$PID_FILE" 2>/dev/null || true)" = "$$" ]; then
    rm -f "$PID_FILE"
  fi
  rm -f "$LOCK_DIR/owner.txt"
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
printf '%s\n' "pid=$$ host=$(hostname) mode=$MODE action=$ACTION started=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCK_DIR/owner.txt"
printf '%s\n' "$$" > "$PID_FILE"

scan=0
while :; do
  scan=$((scan + 1))
  run_scan "$scan"
  if [ "$ALERT_TRIGGERED" = "1" ]; then
    echo "[batch44-quick20-watch] exiting monitor because no_text margin<0 alert is active"
    break
  fi
  if [ "$STOP_WHEN_COMPLETE" = "1" ] && all_complete; then
    echo "[batch44-quick20-watch] all 15 paired checkpoints complete"
    break
  fi
  if [ "$MODE" = "once" ]; then
    break
  fi
  if [ "$MAX_SCANS" -gt 0 ] && [ "$scan" -ge "$MAX_SCANS" ]; then
    echo "[batch44-quick20-watch] reached MAX_SCANS=$MAX_SCANS"
    break
  fi
  sleep "$POLL_SECONDS"
done

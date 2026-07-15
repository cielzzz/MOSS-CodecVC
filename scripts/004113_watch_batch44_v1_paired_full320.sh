#!/usr/bin/env bash
# Discover and safely schedule the paired Batch-44 full320 evaluations.
#
# The registered schedule contains step-10000, step-20000, and step-30000. A later
# step is never scheduled before the earlier step is complete.  Both r3 and
# r5 checkpoints must be complete, configuration-compatible, and settled at
# the exact same step.  004112 then performs all scientific and QZ audits.
#
# Safe one-shot status/plan:
#   MODE=once ACTION=plan bash scripts/004113_watch_batch44_v1_paired_full320.sh
#
# Platform dry-run when the next paired checkpoint is ready:
#   MODE=monitor ACTION=dry-run bash scripts/004113_watch_batch44_v1_paired_full320.sh
#
# Live operation is double-gated and submits at most one evaluation at once:
#   MODE=monitor ACTION=submit ALLOW_LIVE_SUBMIT=1 \
#     bash scripts/004113_watch_batch44_v1_paired_full320.sh

set -euo pipefail

CANONICAL_PROJECT_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
PROJECT_ROOT="${PROJECT_ROOT:-$CANONICAL_PROJECT_ROOT}"
TEST_MODE="${BATCH44_FULL320_TEST_MODE:-0}"
STAMP="20260713"
MODE="${MODE:-once}"
ACTION="${ACTION:-plan}"
ALLOW_LIVE_SUBMIT="${ALLOW_LIVE_SUBMIT:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_SCANS="${MAX_SCANS:-0}"
MIN_CHECKPOINT_AGE_SEC="${MIN_CHECKPOINT_AGE_SEC:-90}"
STOP_WHEN_COMPLETE="${STOP_WHEN_COMPLETE:-1}"

R3_RUN_DIR="${R3_RUN_DIR:-$PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r3_v1_30k}"
R5_RUN_DIR="${R5_RUN_DIR:-$PROJECT_ROOT/outputs/lora_runs/ver2_9_5_final_r5_v1_30k}"
R3_TRAIN_JOB_ID="job-2b91d332-d500-4279-84f9-0a6a81a376aa"
R5_TRAIN_JOB_ID="job-b8eb2f1f-a3eb-483b-a289-b4cce281525c"
ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122" # MTTS-3-2-0715
TRAIN_IDENTITY_ROOT="$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713"
TRAIN_PAIR_LEDGER="$TRAIN_IDENTITY_ROOT/submitted_pair.tsv"
STATE_ROOT="${STATE_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_paired_full320_scheduler_${STAMP}}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/testset/outputs/ver23_batch44_paired_full320_${STAMP}}"
SUBMIT_WRAPPER="${SUBMIT_WRAPPER:-$PROJECT_ROOT/scripts/004112_submit_batch44_v1_paired_full320_qz.sh}"
PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python}"
STEPS="10000 20000 30000"
MONITOR_LOCK="$STATE_ROOT/.monitor.lock"

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
for value in "$POLL_SECONDS" "$MAX_SCANS" "$MIN_CHECKPOINT_AGE_SEC"; do
  case "$value" in
    ''|*[!0-9]*) die "POLL_SECONDS, MAX_SCANS, and MIN_CHECKPOINT_AGE_SEC must be non-negative integers" ;;
  esac
done
[ "$POLL_SECONDS" -gt 0 ] || die "POLL_SECONDS must be positive"
if [ "$PROJECT_ROOT" != "$CANONICAL_PROJECT_ROOT" ] && [ "$TEST_MODE" != "1" ]; then
  die "non-canonical PROJECT_ROOT is allowed only with BATCH44_FULL320_TEST_MODE=1"
fi
if [ "$ACTION" = "submit" ]; then
  [ "$ALLOW_LIVE_SUBMIT" = "1" ] || die "ACTION=submit requires ALLOW_LIVE_SUBMIT=1"
  [ "$TEST_MODE" = "0" ] || die "test mode may not submit live jobs"
  [ "$PROJECT_ROOT" = "$CANONICAL_PROJECT_ROOT" ] || die "live submission requires canonical PROJECT_ROOT"
fi
[ -x "$PYTHON" ] || die "Python interpreter is not executable: $PYTHON"
if [ "$ACTION" != "plan" ]; then
  [ -s "$SUBMIT_WRAPPER" ] || die "missing Batch-44 full320 submit wrapper: $SUBMIT_WRAPPER"
  bash -n "$SUBMIT_WRAPPER"
fi
mkdir -p "$STATE_ROOT" "$EVAL_ROOT"
if [ -s "$STATE_ROOT/QZ_SUBMISSIONS_DISABLED.json" ] && [ "$ACTION" != "plan" ]; then
  die "QZ full320 scheduling is disabled by the local-4090 evaluation policy: $STATE_ROOT/QZ_SUBMISSIONS_DISABLED.json"
fi

audit_training_pair() {
  if [ "$TEST_MODE" = "1" ]; then
    return 0
  fi
  "$PYTHON" - "$TRAIN_PAIR_LEDGER" "$R3_RUN_DIR" "$R5_RUN_DIR" \
    "$R3_TRAIN_JOB_ID" "$R5_TRAIN_JOB_ID" "$ALLOWED_COMPUTE_GROUP" \
    "$TRAIN_IDENTITY_ROOT" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

ledger = Path(sys.argv[1])
r3_out, r5_out, r3_job, r5_job, compute = sys.argv[2:7]
identity_root = Path(sys.argv[7])
if not ledger.is_file():
    raise SystemExit(f"missing Batch-44 training pair ledger: {ledger}")
with ledger.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
expected = {
    "ver2_9_5_final_r3_v1_30k": ("r3", r3_job, r3_out),
    "ver2_9_5_final_r5_v1_30k": ("r5", r5_job, r5_out),
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
    arm, job, out_dir = expected[name]
    if row.get("arm") != arm:
        errors.append(f"{name} arm={row.get('arm')!r}, expected {arm!r}")
    if row.get("job_id") != job or row.get("out_dir") != out_dir:
        errors.append(f"training provenance drift for {name}: {row}")
    if row.get("compute_group") != compute:
        errors.append(
            f"{name} compute_group={row.get('compute_group')!r}, expected {compute!r}"
        )
    wanted_runner = str(identity_root / arm / "run_train_entrypoint.sh")
    if row.get("runner") != wanted_runner:
        errors.append(f"{name} runner={row.get('runner')!r}, expected {wanted_runner!r}")
if seen != set(expected):
    errors.append(f"training names={sorted(seen)}, expected {sorted(expected)}")

immutable = {
    identity_root / "evaluation_contract.json": "cd41b1f1cb97fb7bd50b5939a6825ca55143835f4c7ad164af42260551b946c1",
    identity_root / "input_identity.full_sha256.json": "accefffbac9aa78b499c5938c9842040dbe44eb6a6188b8a4c444db1b57566b4",
    identity_root / "r3/train_args_dry_run_core.json": "3637b90fe14bbda7c4915362d5bd726b072833f07853df4fc555fde0d2c32d7b",
    identity_root / "r5/train_args_dry_run_core.json": "161f05b766454ce7d3e38af1772a1126ec612eaeb0682678bd6a46ca1d62daff",
    identity_root / "r3/run_train_entrypoint.sh": "09492f6304287918115b01bc0c582c2394d8ba417636c665a57bd30152386b1a",
    identity_root / "r5/run_train_entrypoint.sh": "193e23ce50c5d46ccecaccfb493871d5ddcb773ef2482ca53a0b3fd7b170c207",
}
for path, wanted in immutable.items():
    if not path.is_file():
        errors.append(f"missing immutable training artifact: {path}")
        continue
    with path.open("rb") as handle:
        got = hashlib.file_digest(handle, "sha256").hexdigest()
    if got != wanted:
        errors.append(f"immutable training artifact SHA256={got}, expected {wanted}: {path}")

contract_path = identity_root / "evaluation_contract.json"
if contract_path.is_file():
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != "batch44_v1_r3_r5_eval_contract_v1":
        errors.append(f"evaluation contract schema={contract.get('schema')!r}")
    if contract.get("full320_steps_if_healthy") != [10000, 20000, 30000]:
        errors.append(f"evaluation contract steps={contract.get('full320_steps_if_healthy')!r}")
    if contract.get("stop_rules") != {
        "loss": "NaN/Inf/divergence",
        "no_text_cer": ">0.20",
        "no_text_wavlm_margin": "<0.02",
    }:
        errors.append(f"evaluation contract stop_rules={contract.get('stop_rules')!r}")
if errors:
    raise SystemExit("Batch-44 full320 watcher training audit failed:\n- " + "\n- ".join(errors))
print(
    f"[batch44-full320-watch] training provenance r3={r3_job} "
    f"r5={r5_job} compute_group={compute}"
)
PY
}

record_root_for() {
  printf '%s/trainset/qz_jobs/ver23_batch44_paired_full320_step%s_%s\n' "$PROJECT_ROOT" "$1" "$STAMP"
}

step_root_for() {
  printf '%s/step-%s\n' "$EVAL_ROOT" "$1"
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

r3, r5, project_root = map(Path, sys.argv[1:4])
step = int(sys.argv[4])
min_age = int(sys.argv[5])
test_mode = sys.argv[6] == "1"
now = time.time()
required = {
    "adapter_model.safetensors": 1_000_000,
    "adapter_config.json": 1,
    "README.md": 1,
    "timbre_memory_adapter.pt": 1_000_000,
    "timbre_memory_config.json": 1,
}

def equal(got, wanted):
    if isinstance(wanted, float):
        try:
            return math.isclose(float(got), wanted, rel_tol=0.0, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return got == wanted

def probe(arm: str, run_dir: Path, repeat: int):
    checkpoint = run_dir / f"step-{step}"
    if not checkpoint.is_dir():
        return False, f"{arm}:missing"
    newest = 0.0
    for name, minimum in required.items():
        path = checkpoint / name
        if not path.is_file():
            return False, f"{arm}:missing:{name}"
        stat = path.stat()
        newest = max(newest, stat.st_mtime)
        if stat.st_size < minimum:
            return False, f"{arm}:small:{name}:{stat.st_size}"
    age = now - newest
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
    bad = [f"{key}={cfg.get(key)!r}" for key, wanted in expected_cfg.items() if not equal(cfg.get(key), wanted)]
    if bad:
        return False, f"{arm}:config:" + ",".join(bad)

    identity = (
        project_root
        / "trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713"
        / arm
        / "train_args_dry_run_core.json"
    )
    if not identity.is_file():
        return False, f"{arm}:missing:train_args_dry_run_core.json"
    with identity.open("rb") as handle:
        identity_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    if not test_mode:
        wanted_sha = {
            "r3": "3637b90fe14bbda7c4915362d5bd726b072833f07853df4fc555fde0d2c32d7b",
            "r5": "161f05b766454ce7d3e38af1772a1126ec612eaeb0682678bd6a46ca1d62daff",
        }[arm]
        if identity_sha != wanted_sha:
            return False, f"{arm}:identity_sha256_mismatch:{identity_sha}"
    try:
        args = json.loads(identity.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, f"{arm}:invalid_args:{exc}"
    no_text = project_root / "trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/no_text.train.jsonl"
    text = project_root / "trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/text.train.jsonl"
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
        return False, f"{arm}:identity_mismatch:" + ",".join(bad_args)
    return True, f"{arm}:ready:age={age:.0f}s"

r3_ok, r3_reason = probe("r3", r3, 3)
r5_ok, r5_reason = probe("r5", r5, 5)
print("ready" if r3_ok and r5_ok else "waiting")
print(r3_reason)
print(r5_reason)
PY
}

refresh_rollup() {
  "$PYTHON" - "$PROJECT_ROOT" "$STATE_ROOT" "$EVAL_ROOT" "$STAMP" "$STEPS" <<'PY'
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
metrics = []
status_rows = []
seen = set()
for step in steps:
    record = project_root / f"trainset/qz_jobs/ver23_batch44_paired_full320_step{step}_{stamp}"
    step_root = eval_root / f"step-{step}"
    completion = record / "COMPLETED.json"
    metrics_path = step_root / "aggregate/paired_metrics.tsv"
    complete = completion.is_file() and metrics_path.is_file() and (record / "complete.marker").is_file()
    submitted = (record / "submitted_jobs.tsv").is_file() and "job-" in (record / "submitted_jobs.tsv").read_text(encoding="utf-8", errors="replace")
    dry_run = (record / "dry_run.ok").is_file()
    locked = (record / ".live_submit.lock").is_dir()
    status_rows.append({
        "step": step,
        "complete": complete,
        "submitted": submitted,
        "dry_run": dry_run,
        "live_lock": locked,
        "record_root": str(record),
        "step_root": str(step_root),
    })
    if not complete:
        continue
    payload = json.loads(completion.read_text(encoding="utf-8"))
    if payload.get("schema") != "batch44_v1_paired_full320_v1" or payload.get("status") != "complete" or int(payload.get("step", -1)) != step:
        raise SystemExit(f"invalid completion artifact: {completion}")
    expected_jobs = {
        "r3": "job-2b91d332-d500-4279-84f9-0a6a81a376aa",
        "r5": "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c",
    }
    if payload.get("training_jobs") != expected_jobs:
        raise SystemExit(
            f"completion training_jobs={payload.get('training_jobs')!r}, "
            f"expected={expected_jobs!r}: {completion}"
        )
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        step_rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(step_rows) != 6:
        raise SystemExit(f"complete step-{step} must contain six metric rows, got {len(step_rows)}")
    expected_keys = {(arm, scope) for arm in ("r3", "r5") for scope in ("no_text", "text", "all")}
    actual_keys = {(row["arm"], row["scope"]) for row in step_rows}
    if actual_keys != expected_keys:
        raise SystemExit(f"step-{step} metric keys={actual_keys}, expected={expected_keys}")
    for row in step_rows:
        key = (int(row["step"]), row["arm"], row["scope"])
        if key in seen or key[0] != step:
            raise SystemExit(f"duplicate/mismatched metric key: {key}")
        expected_repeat = "3" if row["arm"] == "r3" else "5"
        expected_job = expected_jobs[row["arm"]]
        expected_n = "320" if row["scope"] == "all" else "160"
        if row.get("text_repeat") != expected_repeat:
            raise SystemExit(f"{key}: text_repeat={row.get('text_repeat')!r}")
        if row.get("train_job_id") != expected_job:
            raise SystemExit(f"{key}: train_job_id={row.get('train_job_id')!r}")
        if row.get("n") != expected_n:
            raise SystemExit(f"{key}: n={row.get('n')!r}, expected={expected_n}")
        if row["scope"] == "no_text" and row.get("text_en_src_n") not in {"", None}:
            raise SystemExit(f"{key}: no_text row unexpectedly has text_en_src_n")
        if row["scope"] in {"text", "all"} and row.get("text_en_src_n") != "80":
            raise SystemExit(f"{key}: text_en_src_n={row.get('text_en_src_n')!r}")
        seen.add(key)
        metrics.append(row)

state_root.mkdir(parents=True, exist_ok=True)
(state_root / "status.json").write_text(json.dumps(status_rows, indent=2) + "\n", encoding="utf-8")
with (state_root / "status.tsv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(status_rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(status_rows)

if metrics:
    with (eval_root / "paired_metrics_all.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(metrics)
(eval_root / "paired_metrics_all.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

completed = sum(row["complete"] for row in status_rows)
lines = [
    "# Batch-44 paired full320 schedule",
    "",
    f"Completed paired checkpoints: {completed}/{len(steps)}.",
    "",
    "Each completed step contains r3/r5 × no_text160/text160, full text en_src n=80, WavLM and SpeechBrain speaker metrics, ASR, and ref-content F1.",
    "",
    "| Step | complete | submitted | dry-run | live lock |",
    "|---:|---|---|---|---|",
]
for row in status_rows:
    lines.append(f"| {row['step']} | {row['complete']} | {row['submitted']} | {row['dry_run']} | {row['live_lock']} |")
if metrics:
    lines.extend([
        "",
        "## Metrics",
        "",
        "| Step | Arm | Scope | fail | CER | WavLM ref | WavLM src | WavLM margin | SpB ref | ref-bound | F1 | text en_src fail |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in metrics:
        en = "—" if row["text_en_src_fail_rate"] == "" else f"{float(row['text_en_src_fail_rate']):.2%}"
        lines.append(
            f"| {row['step']} | {row['arm']} | {row['scope']} | {float(row['fail_rate']):.2%} | "
            f"{float(row['cer']):.4f} | {float(row['wavlm_sim_ref']):.4f} | "
            f"{float(row['wavlm_sim_src']):.4f} | {float(row['wavlm_margin']):.4f} | "
            f"{float(row['speechbrain_sim_ref']):.4f} | {float(row['wavlm_ref_bound']):.2%} | "
            f"{float(row['ref_content_lcs_f1']):.4f} | {en} |"
        )
(eval_root / "paired_metrics_all.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[batch44-full320-watch] rollup completed={completed}/{len(steps)} metrics_rows={len(metrics)}")
PY
}

detect_full320_red_flags() {
  "$PYTHON" - \
    "$PROJECT_ROOT" "$STATE_ROOT" "$EVAL_ROOT" "$STAMP" "$STEPS" \
    "$R3_TRAIN_JOB_ID" "$R5_TRAIN_JOB_ID" <<'PY'
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import sys
from pathlib import Path

project_root = Path(sys.argv[1])
state_root = Path(sys.argv[2])
eval_root = Path(sys.argv[3])
stamp = sys.argv[4]
steps = [int(value) for value in sys.argv[5].split()]
r3_job, r5_job = sys.argv[6:8]
jobs = {"r3": r3_job, "r5": r5_job}
alerts: list[dict[str, object]] = []


def number(row: dict[str, str], field: str, source: Path) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid {field} in {source}: {row}") from exc
    if not math.isfinite(value):
        raise SystemExit(f"non-finite {field} in {source}: {row}")
    return value


def add_alert(
    *,
    step: int,
    arm: str,
    scope: str,
    code: str,
    value: float,
    threshold: float,
    relation: str,
    metrics: Path,
    row: dict[str, str],
) -> None:
    alerts.append(
        {
            "step": step,
            "arm": arm,
            "scope": scope,
            "code": code,
            "value": value,
            "threshold": threshold,
            "relation": relation,
            "cer": number(row, "cer", metrics),
            "wavlm_sim_ref": number(row, "wavlm_sim_ref", metrics),
            "wavlm_sim_src": number(row, "wavlm_sim_src", metrics),
            "wavlm_margin": number(row, "wavlm_margin", metrics),
            "ref_content_lcs_f1": number(row, "ref_content_lcs_f1", metrics),
            "training_job_id": jobs[arm],
            "metrics_tsv": str(metrics),
        }
    )


for step in steps:
    record = project_root / f"trainset/qz_jobs/ver23_batch44_paired_full320_step{step}_{stamp}"
    metrics = eval_root / f"step-{step}/aggregate/paired_metrics.tsv"
    if not (record / "complete.marker").is_file() or not metrics.is_file():
        continue
    with metrics.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 6:
        raise SystemExit(f"completed step-{step} must contain six metric rows, got {len(rows)}")
    by_key = {(row.get("arm", ""), row.get("scope", "")): row for row in rows}
    expected = {(arm, scope) for arm in ("r3", "r5") for scope in ("no_text", "text", "all")}
    if set(by_key) != expected:
        raise SystemExit(f"invalid metric keys in {metrics}: {sorted(by_key)}")
    for arm in ("r3", "r5"):
        row = by_key[(arm, "no_text")]
        margin = number(row, "wavlm_margin", metrics)
        cer = number(row, "cer", metrics)
        f1 = number(row, "ref_content_lcs_f1", metrics)
        if margin < 0.0:
            add_alert(
                step=step,
                arm=arm,
                scope="no_text",
                code="negative_no_text_margin",
                value=margin,
                threshold=0.0,
                relation="<",
                metrics=metrics,
                row=row,
            )
        elif margin < 0.02:
            add_alert(
                step=step,
                arm=arm,
                scope="no_text",
                code="no_text_margin_lt_0p02",
                value=margin,
                threshold=0.02,
                relation="<",
                metrics=metrics,
                row=row,
            )
        if cer > 0.20:
            add_alert(
                step=step,
                arm=arm,
                scope="no_text",
                code="no_text_cer_gt_0p20",
                value=cer,
                threshold=0.20,
                relation=">",
                metrics=metrics,
                row=row,
            )
        if f1 > 0.20:
            add_alert(
                step=step,
                arm=arm,
                scope="no_text",
                code="no_text_ref_content_f1_gt_0p20",
                value=f1,
                threshold=0.20,
                relation=">",
                metrics=metrics,
                row=row,
            )

        text_row = by_key[(arm, "text")]
        text_cer = number(text_row, "cer", metrics)
        if text_cer > 0.30:
            add_alert(
                step=step,
                arm=arm,
                scope="text",
                code="text_cer_gt_0p30",
                value=text_cer,
                threshold=0.30,
                relation=">",
                metrics=metrics,
                row=text_row,
            )
        if step >= 20000:
            en_src_fail = number(text_row, "text_en_src_fail_rate", metrics)
            if en_src_fail > 0.25:
                add_alert(
                    step=step,
                    arm=arm,
                    scope="text",
                    code="text_en_src_fail_gt_0p25_at_or_after_20k",
                    value=en_src_fail,
                    threshold=0.25,
                    relation=">",
                    metrics=metrics,
                    row=text_row,
                )

if not alerts:
    raise SystemExit(1)

generated = dt.datetime.now(dt.timezone.utc).isoformat()
payload = {
    "schema": "batch44_v1_full320_red_flag_alert_v1",
    "status": "alert",
    "generated_utc": generated,
    "policy": {
        "negative_no_text_margin": "any completed full320 no_text WavLM margin < 0",
        "no_text_margin": "any completed full320 no_text WavLM margin < 0.02",
        "content": "any completed full320 no_text CER > 0.20; text CER > 0.30",
        "reference_copy": "no_text F1(ref-content) > 0.20",
        "text_en_src": "step >= 20k and full text en_src fail > 25%",
    },
    "scheduler_action": "stop scheduling later full320 evaluations pending manual review",
    "training_action": "recommend stop only; watcher does not call QZ stop or mutate training jobs",
    "training_jobs": jobs,
    "alerts": alerts,
}
state_root.mkdir(parents=True, exist_ok=True)
eval_root.mkdir(parents=True, exist_ok=True)
rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
for root in (state_root, eval_root):
    (root / "ALERT_FULL320_RED_FLAGS.json").write_text(rendered, encoding="utf-8")
if any(item["code"] == "negative_no_text_margin" for item in alerts):
    for root in (state_root, eval_root):
        (root / "ALERT_NEGATIVE_NO_TEXT_MARGIN.json").write_text(rendered, encoding="utf-8")

lines = [
    "# ALERT: Batch-44 v1 paired full320 red flag",
    "",
    f"Generated: `{generated}`",
    "",
    "Later full320 scheduling is paused pending manual review.",
    "No QZ stop command was issued and neither training job was modified.",
    "",
    "| Step | Arm | Scope | Code | Value | Rule | CER | WavLM margin | F1(ref-content) | Job |",
    "|---:|---|---|---|---:|---:|---:|---:|---:|---|",
]
for item in alerts:
    lines.append(
        f"| {item['step']} | {item['arm']} | {item['scope']} | {item['code']} | "
        f"{item['value']:.4f} | {item['relation']} {item['threshold']:.4f} | "
        f"{item['cer']:.4f} | {item['wavlm_margin']:.4f} | "
        f"{item['ref_content_lcs_f1']:.4f} | `{item['training_job_id']}` |"
    )
alert_md = "\n".join(lines) + "\n"
recommendation = "\n".join(
    [
        "# Batch-44 stop recommendation",
        "",
        "A completed paired full320 result crossed a preregistered red flag.",
        "Inspect `ALERT_FULL320_RED_FLAGS.json` and the exact per-case metrics before deciding",
        "whether to stop one or both training arms.",
        "",
        "This watcher deliberately performs no automatic QZ or training stop operation.",
        "",
    ]
)
for root in (state_root, eval_root):
    (root / "ALERT_FULL320_RED_FLAGS.md").write_text(alert_md, encoding="utf-8")
    (root / "STOP_RECOMMENDATION.md").write_text(recommendation, encoding="utf-8")
print(f"[batch44-full320-ALERT] red_flags={len(alerts)} state={state_root}")
PY
}

append_action() {
  local step="$1"
  local action="$2"
  local result="$3"
  local ledger="$STATE_ROOT/actions.tsv"
  if [ ! -s "$ledger" ]; then
    printf 'utc\tstep\taction\tresult\n' > "$ledger"
  fi
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$step" "$action" "$result" >> "$ledger"
}

scan_once() {
  refresh_rollup || return $?
  ALERT_TRIGGERED=0
  if detect_full320_red_flags; then
    ALERT_TRIGGERED=1
    echo "[batch44-full320-watch] ALERT active: later scheduling paused; training jobs were not stopped"
    return 0
  fi
  local step record completion marker metrics submitted active_step="" next_step="" probe status detail_r3 detail_r5
  : > "$STATE_ROOT/scan_latest.tsv"
  printf 'step\tstate\tdetail_r3\tdetail_r5\trecord_root\n' > "$STATE_ROOT/scan_latest.tsv"

  # Strict chronology: only the earliest incomplete registered step can run.
  for step in $STEPS; do
    record="$(record_root_for "$step")"
    completion="$record/COMPLETED.json"
    marker="$record/complete.marker"
    metrics="$(step_root_for "$step")/aggregate/paired_metrics.tsv"
    if [ -s "$completion" ] && [ -s "$marker" ] && [ -s "$metrics" ]; then
      printf '%s\tcomplete\t—\t—\t%s\n' "$step" "$record" >> "$STATE_ROOT/scan_latest.tsv"
      continue
    fi
    if [ -s "$record/submitted_jobs.tsv" ] && grep -Eq 'job-[0-9a-fA-F-]{36}' "$record/submitted_jobs.tsv"; then
      active_step="$step"
      printf '%s\tsubmitted_waiting\t—\t—\t%s\n' "$step" "$record" >> "$STATE_ROOT/scan_latest.tsv"
      break
    fi
    if [ -d "$record/.live_submit.lock" ]; then
      active_step="$step"
      printf '%s\tlocked_manual_audit\t—\t—\t%s\n' "$step" "$record" >> "$STATE_ROOT/scan_latest.tsv"
      break
    fi
    if [ -e "$completion" ] || [ -e "$marker" ] || [ -e "$metrics" ]; then
      active_step="$step"
      printf '%s\tinconsistent_partial_completion\t—\t—\t%s\n' "$step" "$record" >> "$STATE_ROOT/scan_latest.tsv"
      echo "ERROR: step-$step has partial completion artifacts; manual audit is required before any later full320 scheduling" >&2
      return 21
    fi
    probe="$(checkpoint_probe "$step")"
    status="$(printf '%s\n' "$probe" | sed -n '1p')"
    detail_r3="$(printf '%s\n' "$probe" | sed -n '2p')"
    detail_r5="$(printf '%s\n' "$probe" | sed -n '3p')"
    printf '%s\t%s\t%s\t%s\t%s\n' "$step" "$status" "$detail_r3" "$detail_r5" "$record" >> "$STATE_ROOT/scan_latest.tsv"
    if [ "$status" = "ready" ]; then
      next_step="$step"
    fi
    break
  done

  if [ -n "$active_step" ]; then
    echo "[batch44-full320-watch] active_step=$active_step waiting for completion; no new submission"
    return 0
  fi
  if [ -z "$next_step" ]; then
    local completed_count
    completed_count=$(awk -F '\t' 'NR>1 && $2=="complete" {n++} END {print n+0}' "$STATE_ROOT/scan_latest.tsv")
    if [ "$completed_count" = "3" ]; then
      echo "[batch44-full320-watch] all registered full320 checkpoints are complete"
      return 10
    fi
    echo "[batch44-full320-watch] no paired checkpoint ready"
    return 0
  fi

  echo "[batch44-full320-watch] next_ready_step=$next_step"
  case "$ACTION" in
    plan)
      echo "[batch44-full320-watch] plan only; would run STEP=$next_step via $SUBMIT_WRAPPER"
      ;;
    dry-run)
      local dry_marker
      dry_marker="$(record_root_for "$next_step")/dry_run.ok"
      if [ -s "$dry_marker" ]; then
        echo "[batch44-full320-watch] step-$next_step already passed platform dry-run"
      else
        local wrapper_rc=0
        STEP="$next_step" DRY_RUN=1 MIN_CHECKPOINT_AGE_SEC="$MIN_CHECKPOINT_AGE_SEC" \
          PROJECT_ROOT="$PROJECT_ROOT" R3_RUN_DIR="$R3_RUN_DIR" R5_RUN_DIR="$R5_RUN_DIR" \
          EVAL_ROOT="$EVAL_ROOT" \
          bash "$SUBMIT_WRAPPER" || wrapper_rc=$?
        if [ "$wrapper_rc" -ne 0 ]; then
          append_action "$next_step" dry-run "failed_rc_$wrapper_rc"
          echo "ERROR: Batch-44 full320 dry-run wrapper failed rc=$wrapper_rc; no success was recorded" >&2
          return 1
        fi
        append_action "$next_step" dry-run success
      fi
      ;;
    submit)
      local wrapper_rc=0
      STEP="$next_step" DRY_RUN=0 CONFIRM_BATCH44_FULL320=1 \
        MIN_CHECKPOINT_AGE_SEC="$MIN_CHECKPOINT_AGE_SEC" \
        PROJECT_ROOT="$PROJECT_ROOT" R3_RUN_DIR="$R3_RUN_DIR" R5_RUN_DIR="$R5_RUN_DIR" \
        EVAL_ROOT="$EVAL_ROOT" bash "$SUBMIT_WRAPPER" || wrapper_rc=$?
      if [ "$wrapper_rc" -ne 0 ]; then
        append_action "$next_step" submit "failed_rc_$wrapper_rc"
        echo "ERROR: Batch-44 full320 submit wrapper failed rc=$wrapper_rc; inspect the persistent per-step lock and QZ state before recovery" >&2
        return 1
      fi
      append_action "$next_step" submit success
      ;;
  esac
  return 0
}

release_monitor_lock() {
  if [ -d "$MONITOR_LOCK" ]; then
    rm -f "$MONITOR_LOCK/owner.json"
    rmdir "$MONITOR_LOCK" 2>/dev/null || true
  fi
}

audit_training_pair
if [ "$MODE" = "once" ]; then
  set +e
  scan_once
  rc=$?
  set -e
  [ "$rc" = "10" ] && exit 0
  exit "$rc"
fi

if ! mkdir "$MONITOR_LOCK" 2>/dev/null; then
  die "another monitor or a stale monitor lock exists: $MONITOR_LOCK"
fi
"$PYTHON" - "$MONITOR_LOCK/owner.json" "$ACTION" <<'PY'
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
    "action": sys.argv[2],
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
trap release_monitor_lock EXIT
trap 'release_monitor_lock; exit 130' INT
trap 'release_monitor_lock; exit 143' TERM

scan_count=0
ALERT_TRIGGERED=0
while true; do
  scan_count=$((scan_count + 1))
  echo "[batch44-full320-watch] scan=$scan_count utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) action=$ACTION"
  set +e
  scan_once
  rc=$?
  set -e
  if [ "$ALERT_TRIGGERED" = "1" ]; then
    echo "[batch44-full320-watch] exiting monitor because a full320 red-flag alert is active"
    break
  fi
  if [ "$rc" = "10" ] && [ "$STOP_WHEN_COMPLETE" = "1" ]; then
    echo "[batch44-full320-watch] registered schedule complete; stopping monitor"
    break
  fi
  if [ "$rc" != "0" ] && [ "$rc" != "10" ]; then
    exit "$rc"
  fi
  if [ "$MAX_SCANS" -gt 0 ] && [ "$scan_count" -ge "$MAX_SCANS" ]; then
    echo "[batch44-full320-watch] MAX_SCANS=$MAX_SCANS reached"
    break
  fi
  sleep "$POLL_SECONDS"
done

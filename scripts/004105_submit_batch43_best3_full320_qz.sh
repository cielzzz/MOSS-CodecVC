#!/usr/bin/env bash
# Submit the Batch-44 v1 quick20-selected Best3 for full SeedTTS-derived 320.
#
# 004101 deliberately hard-locks the milestone schedule to paired step-20k and
# step-30k.  Best3 additionally needs 26k/28k.  This wrapper materializes an
# audited copy of 004101 which changes only that accepted-step set and its
# self-snapshot source.  The scientific evaluation, checkpoint audits, four
# r3/r5 lanes, metrics schema, MTTS-only resource contract and live fences all
# remain byte-for-byte inherited from 004101.
#
# Because 004101 preserves same-step pairing, this may evaluate an unselected
# counterpart at a selected step.  Only candidate ids recorded in the 004103
# selection advance to blind20.
#
# Safe default (QZ platform dry-runs only; no live jobs):
#   bash scripts/004105_submit_batch43_best3_full320_qz.sh
#
# Plan/materialization only (does not touch QZ or checkpoints):
#   PLAN_ONLY=1 bash scripts/004105_submit_batch43_best3_full320_qz.sh
#
# A future live submission requires both explicit switches:
#   DRY_RUN=0 CONFIRM_BATCH44_BEST3_FULL320=1 \
#     bash scripts/004105_submit_batch43_best3_full320_qz.sh

set -euo pipefail

CANONICAL_PROJECT_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
PROJECT_ROOT="${PROJECT_ROOT:-$CANONICAL_PROJECT_ROOT}"
SELECTION_JSON="${SELECTION_JSON:-$PROJECT_ROOT/testset/outputs/batch44_best3_20260713/best3_selection.json}"
SOURCE_004101="${SOURCE_004101:-$PROJECT_ROOT/scripts/004101_submit_batch43_paired_full320_qz.sh}"
PLAN_ROOT="${PLAN_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_best3_full320_plan_20260713}"
MATERIALIZED_004101="$PLAN_ROOT/004101_batch44_v1_best3_steps.materialized.sh"
PLAN_JSON="$PLAN_ROOT/best3_full320_plan.json"
PLAN_TSV="$PLAN_ROOT/best3_full320_plan.tsv"

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
DRY_RUN="${DRY_RUN:-1}"
CONFIRM_BATCH44_BEST3_FULL320="${CONFIRM_BATCH44_BEST3_FULL320:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"
FORCE_MATERIALIZE="${FORCE_MATERIALIZE:-0}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

case "$DRY_RUN:$CONFIRM_BATCH44_BEST3_FULL320:$PLAN_ONLY:$FORCE_MATERIALIZE" in
  [01]:[01]:[01]:[01]) ;;
  *) die "DRY_RUN, CONFIRM_BATCH44_BEST3_FULL320, PLAN_ONLY and FORCE_MATERIALIZE must be 0 or 1" ;;
esac
[ "$PROJECT_ROOT" = "$CANONICAL_PROJECT_ROOT" ] || die "PROJECT_ROOT is hard-locked to $CANONICAL_PROJECT_ROOT"
[ -x "$PYTHON" ] || die "Python is not executable: $PYTHON"
[ -s "$SELECTION_JSON" ] || die "missing Best3 selection: $SELECTION_JSON"
[ -s "$SOURCE_004101" ] || die "missing 004101 source: $SOURCE_004101"
if [ "$DRY_RUN" = "0" ] && [ "$CONFIRM_BATCH44_BEST3_FULL320" != "1" ]; then
  die "live submission requires CONFIRM_BATCH44_BEST3_FULL320=1"
fi

mkdir -p "$PLAN_ROOT"

"$PYTHON" - "$SELECTION_JSON" "$SOURCE_004101" "$MATERIALIZED_004101" \
  "$PLAN_JSON" "$PLAN_TSV" "$FORCE_MATERIALIZE" <<'PY'
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

selection_path, source_path, materialized_path, plan_json, plan_tsv = map(Path, sys.argv[1:6])
force = sys.argv[6] == "1"

selection = json.loads(selection_path.read_text(encoding="utf-8"))
if selection.get("schema_version") != "moss_codecvc.batch44_v1_best3_selection.v1":
    raise SystemExit(f"wrong selection schema: {selection.get('schema_version')!r}")
if selection.get("status") != "selected":
    raise SystemExit(f"Best3 is not selected: status={selection.get('status')!r}")
if selection.get("experiment_id") != "batch44_v1" or selection.get("data_version") != "v1_20260709":
    raise SystemExit("Best3 selection is not the registered Batch-44 v1 experiment")
selected = selection.get("selected_candidate_ids")
if not isinstance(selected, list) or len(selected) != 3 or len(set(selected)) != 3:
    raise SystemExit(f"expected exactly three selected candidate ids, got {selected!r}")

candidate_rows = {
    str(row.get("candidate_id")): row
    for row in selection.get("candidates", [])
    if isinstance(row, dict)
}
resolved = []
expected_jobs = {
    "r3": "job-2b91d332-d500-4279-84f9-0a6a81a376aa",
    "r5": "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c",
}
expected_repeats = {"r3": 3, "r5": 5}
run_dirs = {
    "r3": "ver2_9_5_final_r3_v1_30k",
    "r5": "ver2_9_5_final_r5_v1_30k",
}
for candidate_id in selected:
    row = candidate_rows.get(candidate_id)
    if not row or row.get("selected_for_full320") is not True:
        raise SystemExit(f"selected candidate missing/flag drift: {candidate_id}")
    arm = row.get("arm")
    step = row.get("step")
    if arm not in {"r3", "r5"} or step not in {26000, 28000, 30000}:
        raise SystemExit(f"invalid Best3 candidate {candidate_id}: arm={arm!r} step={step!r}")
    if candidate_id != f"{arm}_step-{step}":
        raise SystemExit(f"candidate id mismatch: {candidate_id}")
    if row.get("train_job_id") != expected_jobs[arm]:
        raise SystemExit(f"Batch-44 training job mismatch: {candidate_id}")
    if row.get("text_repeat") != expected_repeats[arm]:
        raise SystemExit(f"Batch-44 text repeat mismatch: {candidate_id}")
    checkpoint = Path(str((row.get("checkpoint") or {}).get("path") or "")).resolve()
    expected_checkpoint = (
        Path("/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC")
        / "outputs/lora_runs" / run_dirs[arm] / f"step-{step}"
    ).resolve()
    if checkpoint != expected_checkpoint:
        raise SystemExit(f"Batch-44 checkpoint path mismatch: {candidate_id}")
    resolved.append({"candidate_id": candidate_id, "arm": arm, "step": step})

steps = sorted({row["step"] for row in resolved})
extras = [
    f"{arm}_step-{step}"
    for step in steps
    for arm in ("r3", "r5")
    if f"{arm}_step-{step}" not in set(selected)
]

source = source_path.read_text(encoding="utf-8")
accepted_old = 'case "$STEP" in\n  10000|20000|30000) ;;'
accepted_new = 'case "$STEP" in\n  10000|20000|26000|28000|30000) ;;'
message_old = 'STEP must be diagnostic 10000 or registered 20000/30000; got $STEP'
message_new = 'STEP must be diagnostic 10000, milestone 20000/30000, or Best3 26000/28000; got $STEP'
for old, label in ((accepted_old, "accepted-step case"), (message_old, "accepted-step message")):
    if source.count(old) != 1:
        raise SystemExit(f"004101 semantic anchor drift for {label}: count={source.count(old)}")

materialized = source.replace(accepted_old, accepted_new).replace(message_old, message_new)
identity_replacements = {
    "ver23_batch43_ver2_9_5_final_r3_r5_v2_30k_20260712":
        "ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713",
    "ver2_9_5_final_r3_v2_30k": "ver2_9_5_final_r3_v1_30k",
    "ver2_9_5_final_r5_v2_30k": "ver2_9_5_final_r5_v1_30k",
    "job-a34d84d4-59cc-4824-b197-0829bfe79004":
        "job-2b91d332-d500-4279-84f9-0a6a81a376aa",
    "job-aef79753-7fcd-444e-b94d-3e21eedb2394":
        "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c",
    "78525edc2e039e3f2c68dd845aa716966b4c11c560697a6d126a9ec12d17724c":
        "3637b90fe14bbda7c4915362d5bd726b072833f07853df4fc555fde0d2c32d7b",
    "d863f7579dfab905e99f3a0b9980abe310cc51c3a4aadc0292ce5ca6f4ebba9f":
        "161f05b766454ce7d3e38af1772a1126ec612eaeb0682678bd6a46ca1d62daff",
    "ver2_9_prepared_v2_real_no_text_refdecorr_wavlm_sv_20260708/no_text.v2.train.jsonl":
        "ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/no_text.train.jsonl",
    "ver2_9_prepared_v2_real_no_text_refdecorr_wavlm_sv_20260708/text.train.jsonl":
        "ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/text.train.jsonl",
}
for old, new in identity_replacements.items():
    if old not in materialized:
        raise SystemExit(f"004101 Batch-43 identity anchor absent: {old}")
    materialized = materialized.replace(old, new)

materialized = materialized.replace("Batch-43", "Batch-44 v1")
materialized = materialized.replace("BATCH43", "BATCH44_V1")
materialized = materialized.replace("batch43", "batch44")
materialized = materialized.replace("20260712", "20260713")
materialized = materialized.replace(
    '"schema": "batch44_paired_full320_v1"',
    '"schema": "batch44_v1_paired_full320_v1"',
)
self_old = 'RUNNER_SOURCE="$PROJECT_ROOT/scripts/004101_submit_batch44_paired_full320_qz.sh"'
self_new = f'RUNNER_SOURCE="{materialized_path.resolve()}"'
if materialized.count(self_old) != 1:
    raise SystemExit(
        f"004101 Batch-44 self snapshot anchor drift: count={materialized.count(self_old)}"
    )
materialized = materialized.replace(self_old, self_new)

for forbidden in (
    "ver2_9_5_final_r3_v2_30k",
    "ver2_9_5_final_r5_v2_30k",
    "job-a34d84d4-59cc-4824-b197-0829bfe79004",
    "job-aef79753-7fcd-444e-b94d-3e21eedb2394",
    "batch43_paired_full320_v1",
):
    if forbidden in materialized:
        raise SystemExit(f"materialized Batch-44 wrapper retains stopped v2 identity: {forbidden}")
source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
materialized_sha = hashlib.sha256(materialized.encode("utf-8")).hexdigest()

if materialized_path.exists() and materialized_path.read_text(encoding="utf-8") != materialized:
    if not force:
        raise SystemExit(
            f"materialized 004101 drift at {materialized_path}; inspect or set FORCE_MATERIALIZE=1"
        )
if not materialized_path.exists() or force:
    temporary = materialized_path.with_name(f".{materialized_path.name}.tmp-{os.getpid()}")
    temporary.write_text(materialized, encoding="utf-8")
    os.chmod(temporary, 0o555)
    os.replace(temporary, materialized_path)

payload = {
    "schema_version": "moss_codecvc.batch44_v1_best3_full320_plan.v1",
    "experiment_id": "batch44_v1",
    "data_version": "v1_20260709",
    "status": "ready",
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "selection_json": str(selection_path.resolve()),
    "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
    "selected_candidates": resolved,
    "selected_steps": steps,
    "paired_extra_counterparts": extras,
    "source_004101": str(source_path.resolve()),
    "source_004101_sha256": source_sha,
    "materialized_004101": str(materialized_path.resolve()),
    "materialized_004101_sha256": materialized_sha,
    "semantic_delta": [
        "historical 004101 Batch-43 v2 identities are replaced by registered Batch-44 v1 identities",
        "accepted STEP set extends from {10000,20000,30000} to {10000,20000,26000,28000,30000}",
        "self snapshot source points to this materialized copy",
    ],
    "inherited_contract": {
        "compute_group": "MTTS-3-2-0715",
        "gpu_type": "NVIDIA_H200_SXM_141G",
        "instances": 1,
        "gpus": 8,
        "per_step_scope": "paired r3/r5 x no_text160/text160",
        "metrics": "WavLM-large-SV + SpeechBrain ECAPA + ASR + ref-content F1",
    },
}
plan_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = ["step\tselected_candidates\textra_counterparts\tcommand"]
for step in steps:
    chosen = ",".join(row["candidate_id"] for row in resolved if row["step"] == step)
    extra = ",".join(item for item in extras if item.endswith(f"step-{step}"))
    command = f"STEP={step} DRY_RUN=1 bash {materialized_path.resolve()}"
    lines.append(f"{step}\t{chosen}\t{extra}\t{command}")
plan_tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

bash -n "$MATERIALIZED_004101"
echo "[batch44-v1-best3-full320] plan=$PLAN_JSON"
echo "[batch44-v1-best3-full320] steps=$(tail -n +2 "$PLAN_TSV" | cut -f1 | paste -sd, -)"
echo "[batch44-v1-best3-full320] DRY_RUN=$DRY_RUN PLAN_ONLY=$PLAN_ONLY"

if [ "$PLAN_ONLY" = "1" ]; then
  echo "[batch44-v1-best3-full320] plan-only complete; QZ was not touched"
  exit 0
fi

while IFS=$'\t' read -r step selected_candidates extra_counterparts _command; do
  [ "$step" = "step" ] && continue
  echo "[batch44-v1-best3-full320] step=$step selected=$selected_candidates extra=${extra_counterparts:-none}"
  completion="$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_paired_full320_step${step}_20260713/COMPLETED.json"
  metrics="$PROJECT_ROOT/testset/outputs/ver23_batch44_paired_full320_20260713/step-${step}/aggregate/paired_metrics.json"
  marker="$PROJECT_ROOT/trainset/qz_jobs/ver23_batch44_paired_full320_step${step}_20260713/complete.marker"
  if [ -s "$completion" ] && [ -s "$metrics" ] && [ -s "$marker" ]; then
    "$PYTHON" - "$PROJECT_ROOT/scripts/004107_finalize_batch43_pathx_final.py" \
      "$completion" "$metrics" "$step" "$PROJECT_ROOT" <<'PY'
import importlib.util
import sys
from pathlib import Path

validator_path = Path(sys.argv[1])
completion_path = Path(sys.argv[2])
metrics_path = Path(sys.argv[3])
step = int(sys.argv[4])
project_root = Path(sys.argv[5])
spec = importlib.util.spec_from_file_location("batch44_v1_final_validator", validator_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot import full320 validator: {validator_path}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.validate_full320_step(
    step=step,
    completion_path=completion_path,
    metrics_path=metrics_path,
    project_root=project_root,
)
print(f"[batch44-v1-best3-full320] reuse complete step={step} metrics={metrics_path}")
PY
    continue
  fi
  STEP="$step" \
  DRY_RUN="$DRY_RUN" \
  CONFIRM_BATCH44_V1_FULL320="$CONFIRM_BATCH44_BEST3_FULL320" \
  bash "$MATERIALIZED_004101"
done < "$PLAN_TSV"

if [ "$DRY_RUN" = "1" ]; then
  echo "[batch44-v1-best3-full320] all selected-step platform dry-runs passed; no live job submitted"
else
  echo "[batch44-v1-best3-full320] live submissions completed; inspect per-step submitted_jobs.tsv ledgers"
fi

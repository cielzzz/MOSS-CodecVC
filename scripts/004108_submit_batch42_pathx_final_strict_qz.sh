#!/usr/bin/env bash
# Materialize and invoke the Batch-42 strict Path-X wrapper for path_x_final.
#
# 004094 is intentionally immutable for the published 3k probe.  This script
# derives a final-specific copy whose semantic delta is audited and recorded:
#   * system_id path_x_3k -> path_x_final;
#   * checkpoint identity -> FINAL_SELECTION.json exact path/hashes;
#   * frozen inference code -> Batch37 eval snapshot used by Batch-44 v1 full320;
#   * path adapter -> 004106 plus an immutable 004093 base copy.
# Everything else (EN567/ZH1194 manifests, decode knobs, smoke gate, 8 H200
# workers, merge/schema fences and QZ resource locks) stays inherited from
# 004094.
#
# Default is a platform dry-run in smoke mode; no live task is submitted.
#   MODE=smoke bash scripts/004108_submit_batch42_pathx_final_strict_qz.sh
#   MODE=full  bash scripts/004108_submit_batch42_pathx_final_strict_qz.sh
#
# Future live execution is explicitly double-gated:
#   MODE=smoke DRY_RUN=0 CONFIRM_BATCH44_FINAL_STRICT=1 bash scripts/004108_...

set -euo pipefail

CANONICAL_PROJECT_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
PROJECT_ROOT="${PROJECT_ROOT:-$CANONICAL_PROJECT_ROOT}"
FINAL_SELECTION_JSON="${FINAL_SELECTION_JSON:-$PROJECT_ROOT/testset/outputs/batch44_best3_20260713/FINAL_SELECTION.json}"
SOURCE_004094="${SOURCE_004094:-$PROJECT_ROOT/scripts/004094_submit_batch42_pathx_strict_qz.sh}"
SOURCE_004106="${SOURCE_004106:-$PROJECT_ROOT/scripts/004106_run_batch42_pathx_final_strict.py}"
SOURCE_004107="${SOURCE_004107:-$PROJECT_ROOT/scripts/004107_finalize_batch43_pathx_final.py}"
SOURCE_004093="${SOURCE_004093:-$PROJECT_ROOT/scripts/004093_run_batch42_pathx_strict.py}"
PLAN_ROOT="${PLAN_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/batch42_pathx_final_batch44_v1_materialized_20260713}"
MATERIALIZED_004094="$PLAN_ROOT/004094_pathx_final.materialized.sh"
MATERIALIZATION_JSON="$PLAN_ROOT/materialization.json"

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
MODE="${MODE:-smoke}"
DRY_RUN="${DRY_RUN:-1}"
CONFIRM_BATCH44_FINAL_STRICT="${CONFIRM_BATCH44_FINAL_STRICT:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"
FORCE_MATERIALIZE="${FORCE_MATERIALIZE:-0}"
RUN_TAG="${RUN_TAG:-20260713_mtts}"
SMOKE_GATE_TAG="${SMOKE_GATE_TAG:-20260713_mtts}"

die() { echo "ERROR: $*" >&2; exit 2; }

case "$MODE" in smoke|full) ;; *) die "MODE must be smoke or full" ;; esac
case "$DRY_RUN:$CONFIRM_BATCH44_FINAL_STRICT:$PLAN_ONLY:$FORCE_MATERIALIZE" in
  [01]:[01]:[01]:[01]) ;;
  *) die "boolean flags must be 0 or 1" ;;
esac
[ "$PROJECT_ROOT" = "$CANONICAL_PROJECT_ROOT" ] || die "PROJECT_ROOT is hard-locked"
[ -x "$PYTHON" ] || die "Python is not executable: $PYTHON"
for path in "$FINAL_SELECTION_JSON" "$SOURCE_004094" "$SOURCE_004106" "$SOURCE_004107" "$SOURCE_004093"; do
  [ -s "$path" ] || die "missing input: $path"
done
if [ "$DRY_RUN" = "0" ] && [ "$CONFIRM_BATCH44_FINAL_STRICT" != "1" ]; then
  die "live submission requires CONFIRM_BATCH44_FINAL_STRICT=1"
fi

mkdir -p "$PLAN_ROOT"

"$PYTHON" - "$FINAL_SELECTION_JSON" "$SOURCE_004094" "$SOURCE_004106" \
  "$SOURCE_004107" "$SOURCE_004093" "$MATERIALIZED_004094" "$MATERIALIZATION_JSON" \
  "$FORCE_MATERIALIZE" <<'PY'
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

final_path, source_94, source_106, source_107, source_93, output_path, audit_path = map(Path, sys.argv[1:8])
force = sys.argv[8] == "1"
final = json.loads(final_path.read_text(encoding="utf-8"))
specification = importlib.util.spec_from_file_location(
    "batch44_final_selection_strict_validator", source_106
)
if specification is None or specification.loader is None:
    raise SystemExit(f"cannot import FINAL_SELECTION validator: {source_106}")
validator = importlib.util.module_from_spec(specification)
sys.modules[specification.name] = validator
specification.loader.exec_module(validator)
final = validator.load_final_selection(final_path, verify_checkpoint_hashes=True)
for key, expected in {
    "schema_version": "moss_codecvc.batch44_v1_final_selection.v1",
    "status": "final",
    "system_id": "path_x_final",
}.items():
    if final.get(key) != expected:
        raise SystemExit(f"FINAL_SELECTION {key}={final.get(key)!r}, expected {expected!r}")
if final.get("experiment_id") != "batch44_v1" or final.get("data_version") != "v1_20260709":
    raise SystemExit("FINAL_SELECTION is not the registered Batch-44 v1 experiment")
candidate = final.get("candidate") or {}
arm = candidate.get("arm")
step = candidate.get("step")
run_dirs = {
    "r3": "ver2_9_5_final_r3_v1_30k",
    "r5": "ver2_9_5_final_r5_v1_30k",
}
expected_repeats = {"r3": 3, "r5": 5}
expected_train_jobs = {
    "r3": "job-2b91d332-d500-4279-84f9-0a6a81a376aa",
    "r5": "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c",
}
if arm not in run_dirs or step not in {26000, 28000, 30000}:
    raise SystemExit(f"FINAL_SELECTION arm/step is not a registered Best3 point: {arm!r}/{step!r}")
candidate_id = f"{arm}_step-{step}"
if candidate.get("candidate_id") != candidate_id:
    raise SystemExit(f"FINAL_SELECTION candidate_id must be {candidate_id!r}")
if candidate.get("text_repeat") != expected_repeats[arm]:
    raise SystemExit("FINAL_SELECTION text_repeat does not match the registered arm")
if candidate.get("train_job_id") != expected_train_jobs[arm]:
    raise SystemExit("FINAL_SELECTION train_job_id does not match the registered arm")
checkpoint = Path(str(candidate.get("checkpoint_path") or "")).resolve()
expected_checkpoint = (
    Path("/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC")
    / "outputs/lora_runs" / run_dirs[arm] / f"step-{step}"
).resolve()
if checkpoint != expected_checkpoint:
    raise SystemExit(f"FINAL_SELECTION checkpoint={checkpoint}, expected {expected_checkpoint}")
model_files = candidate.get("model_files") or {}
required_files = (
    "README.md", "adapter_config.json", "adapter_model.safetensors",
    "timbre_memory_config.json", "timbre_memory_adapter.pt",
)
if set(model_files) != set(required_files):
    raise SystemExit("FINAL_SELECTION model_files are incomplete")
for name in required_files:
    item = checkpoint / name
    registered = model_files[name]
    if not item.is_file() or item.stat().st_size != registered.get("size"):
        raise SystemExit(f"checkpoint size/path drift: {item}")
    if not isinstance(registered.get("sha256"), str) or len(registered["sha256"]) != 64:
        raise SystemExit(f"invalid registered SHA256 for {name}")
    with item.open("rb") as handle:
        actual_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual_sha != registered["sha256"]:
        raise SystemExit(f"checkpoint SHA256 drift for {name}: {actual_sha}")

text = source_94.read_text(encoding="utf-8")

def replace_once(old: str, new: str, label: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"004094 anchor drift for {label}: count={count}")
    text = text.replace(old, new)

# Identity strings which intentionally occur many times are global changes.
old_checkpoint = (
    "$PROJECT_ROOT/outputs/lora_runs/ver23_content_side_3k_olddata_textrep10_"
    "ver23_content_side_text_bypass_3k_20260710/step-3000"
)
if text.count(old_checkpoint) != 2:
    raise SystemExit("004094 old checkpoint anchor is absent")
text = text.replace(old_checkpoint, str(checkpoint))
text = text.replace("path_x_3k", "path_x_final")

old_code_root = (
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
    "MOSS-CodecVC_snapshots/ver23_batch3436_eval_20260711_1092820"
)
new_code_root = (
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
    "MOSS-CodecVC_snapshots/ver23_batch37_eval_20260711_1092820"
)
if old_code_root not in text:
    raise SystemExit("004094 old code-root anchor is absent")
text = text.replace(old_code_root, new_code_root)

hash_replacements = {
    "c9dec31f4155d39cdbd02069dd8b91677ff5dee03e98d441377d949135a8e709":
        "22045797d68d54bc2b72c64773c43464e4164b19b3a29d97537149e15594fa1d",
    "5815c8ab5e0aab69d19328fd01782620064327eaf5f39cc4923df8ce3ae9ca42":
        "1d32527ec29fada353dc70b88a11cff972da901c5830dfeafb3bcf9f067d3ae3",
    "a8e4cd12d279cfff7c38e3e2d8b21b55d70c403cec654edf7ef77de58acba66a":
        "2be7b4cdf24c18df773b215ad3afe8682a65e519dee6ea81515ac4dd8b44ed1a",
}
for old, new in hash_replacements.items():
    if old not in text:
        raise SystemExit(f"004094 code hash anchor absent: {old}")
    text = text.replace(old, new)

old_model_hashes = {
    "README.md": "4d45f7d68a88a39671cc0cbc86f1acdfbee5351401eee2a97df253f0d077717f",
    "adapter_config.json": "06530eac22376a6befd9e81c95c333e4bb1c889de96e9059c2d5498cd90a7aee",
    "adapter_model.safetensors": "3a51162fc7ccf1b9e1aa477ad7c44fa64390d109b8b63765a9cd636f090f4b25",
    "timbre_memory_config.json": "5c8842d87327c2cf1af2697725a19bf2b53ba654fa0a6b3f68b6a42fd50e9970",
    "timbre_memory_adapter.pt": "020a16ad4bba5a812b2f62e29cb68dcec9d4055344e02de01555be8afd9d6895",
}
for name, old_sha in old_model_hashes.items():
    if old_sha not in text:
        raise SystemExit(f"004094 old model hash anchor absent for {name}")
    text = text.replace(old_sha, str(model_files[name]["sha256"]))

model_block_pattern = re.compile(r"expected_model = \{\n.*?\n\}\nexpected_code = \{", re.DOTALL)
matches = list(model_block_pattern.finditer(text))
if len(matches) != 1:
    raise SystemExit(f"004094 expected_model block count={len(matches)}")
model_lines = ["expected_model = {"]
for name in required_files:
    registered = model_files[name]
    model_lines.append(
        f"    {name!r}: ({int(registered['size'])}, {str(registered['sha256'])!r}),"
    )
model_lines.extend(["}", "expected_code = {"])
text = model_block_pattern.sub("\n".join(model_lines), text, count=1)

replace_once(
    'SOURCE_PATHX_SCRIPT="$PROJECT_ROOT/scripts/004093_run_batch42_pathx_strict.py"',
    'SOURCE_PATHX_SCRIPT="$PROJECT_ROOT/scripts/004106_run_batch42_pathx_final_strict.py"\n'
    'SOURCE_BASE_PATHX_SCRIPT="$PROJECT_ROOT/scripts/004093_run_batch42_pathx_strict.py"\n'
    'SOURCE_FINALIZER_SCRIPT="$PROJECT_ROOT/scripts/004107_finalize_batch43_pathx_final.py"\n'
    f'SOURCE_FINAL_SELECTION={str(final_path.resolve())!r}',
    "final adapter sources",
)
replace_once(
    'PATHX_SCRIPT="$SNAPSHOT_ROOT/scripts/004093_run_batch42_pathx_strict.py"',
    'PATHX_SCRIPT="$SNAPSHOT_ROOT/scripts/004093_run_batch42_pathx_strict.py"\n'
    'BASE_PATHX_SCRIPT="$SNAPSHOT_ROOT/scripts/004093_run_batch42_pathx_strict_base.py"\n'
    'FINALIZER_SCRIPT="$SNAPSHOT_ROOT/scripts/004107_finalize_batch43_pathx_final.py"\n'
    'FINAL_SELECTION_SNAPSHOT="$SNAPSHOT_ROOT/FINAL_SELECTION.json"\n'
    'export BATCH44_FINAL_SELECTION="$FINAL_SELECTION_SNAPSHOT"',
    "snapshot adapter paths",
)
replace_once(
    '  require_file "$SOURCE_PATHX_SCRIPT"\n  require_file "$SOURCE_MERGE_SCRIPT"',
        '  require_file "$SOURCE_PATHX_SCRIPT"\n'
        '  require_file "$SOURCE_BASE_PATHX_SCRIPT"\n'
        '  require_file "$SOURCE_FINALIZER_SCRIPT"\n'
        '  require_file "$SOURCE_FINAL_SELECTION"\n'
    '  require_file "$SOURCE_MERGE_SCRIPT"',
    "snapshot source requirements",
)
replace_once(
    '  cp "$SOURCE_PATHX_SCRIPT" "$PATHX_SCRIPT"\n  cp "$SOURCE_MERGE_SCRIPT" "$MERGE_SCRIPT"',
        '  cp "$SOURCE_PATHX_SCRIPT" "$PATHX_SCRIPT"\n'
        '  cp "$SOURCE_BASE_PATHX_SCRIPT" "$BASE_PATHX_SCRIPT"\n'
        '  cp "$SOURCE_FINALIZER_SCRIPT" "$FINALIZER_SCRIPT"\n'
        '  cp "$SOURCE_FINAL_SELECTION" "$FINAL_SELECTION_SNAPSHOT"\n'
    '  cp "$SOURCE_MERGE_SCRIPT" "$MERGE_SCRIPT"',
    "snapshot copies",
)
replace_once(
    '  chmod 0555 "$PATHX_SCRIPT" "$MERGE_SCRIPT" "$SCHEMA_SCRIPT" "$FROZEN_DRIVER"',
    '  chmod 0555 "$PATHX_SCRIPT" "$BASE_PATHX_SCRIPT" "$FINALIZER_SCRIPT" "$FINAL_SELECTION_SNAPSHOT" '
    '"$MERGE_SCRIPT" "$SCHEMA_SCRIPT" "$FROZEN_DRIVER"',
    "snapshot modes",
)
replace_once(
    '  sha256sum "$PATHX_SCRIPT" "$MERGE_SCRIPT" "$SCHEMA_SCRIPT" "$FROZEN_DRIVER" "$RUNNER" > "$RECORD_ROOT/sha256sums.txt"',
    '  sha256sum "$PATHX_SCRIPT" "$BASE_PATHX_SCRIPT" "$FINALIZER_SCRIPT" "$FINAL_SELECTION_SNAPSHOT" '
    '"$MERGE_SCRIPT" "$SCHEMA_SCRIPT" "$FROZEN_DRIVER" "$RUNNER" > "$RECORD_ROOT/sha256sums.txt"',
    "snapshot hashes",
)
replace_once(
    'require_file "$SOURCE_PATHX_SCRIPT"\nrequire_file "$SOURCE_MERGE_SCRIPT"',
        'require_file "$SOURCE_PATHX_SCRIPT"\n'
        'require_file "$SOURCE_BASE_PATHX_SCRIPT"\n'
        'require_file "$SOURCE_FINALIZER_SCRIPT"\n'
        'require_file "$SOURCE_FINAL_SELECTION"\n'
    'require_file "$SOURCE_MERGE_SCRIPT"',
    "submit source requirements",
)

if text.count("path_x_3k"):
    raise SystemExit("materialized wrapper still contains path_x_3k")
resource_needles = (
    'ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"',
    'ALLOWED_SPEC="67b10bc6-78b0-41a3-aaf4-358eeeb99009"',
    'ALLOWED_GPU_TYPE="NVIDIA_H200_SXM_141G"',
    'ALLOWED_INSTANCES="1"',
    'ALLOWED_GPUS="8"',
    'die "only MTTS-3-2-0715 is allowed; got $COMPUTE_GROUP"',
)
missing_resource = [needle for needle in resource_needles if needle not in text]
if missing_resource:
    raise SystemExit(f"materialized wrapper lost MTTS resource locks: {missing_resource}")
materialized_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
if output_path.exists() and output_path.read_text(encoding="utf-8") != text and not force:
    raise SystemExit(f"materialized wrapper drift: {output_path}; use FORCE_MATERIALIZE=1 after audit")
if not output_path.exists() or force:
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.chmod(temporary, 0o555)
    os.replace(temporary, output_path)

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

audit = {
    "schema_version": "moss_codecvc.batch42_pathx_final_batch44_v1_materialization.v1",
    "status": "ready",
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "system_id": "path_x_final",
    "experiment_id": "batch44_v1",
    "data_version": "v1_20260709",
    "final_selection": str(final_path.resolve()),
    "final_selection_sha256": sha(final_path),
    "checkpoint": str(checkpoint),
    "source_004094": str(source_94.resolve()),
    "source_004094_sha256": sha(source_94),
    "source_004106": str(source_106.resolve()),
    "source_004106_sha256": sha(source_106),
    "source_004107": str(source_107.resolve()),
    "source_004107_sha256": sha(source_107),
    "source_004093_base": str(source_93.resolve()),
    "source_004093_base_sha256": sha(source_93),
    "materialized_004094": str(output_path.resolve()),
    "materialized_004094_sha256": materialized_sha,
    "frozen_eval_code_root": new_code_root,
    "inherited_protocol": "004094 strict EN567/ZH1194 smoke->full contract",
    "resource_contract": {
        "compute_group": "MTTS-3-2-0715",
        "compute_group_id": "lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122",
        "spec": "67b10bc6-78b0-41a3-aaf4-358eeeb99009",
        "gpu_type": "NVIDIA_H200_SXM_141G",
        "instances": 1,
        "gpus": 8,
    },
}
audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
PY

bash -n "$MATERIALIZED_004094"
"$PYTHON" -m py_compile "$SOURCE_004106"
echo "[batch42-pathx-final] materialization=$MATERIALIZATION_JSON"
echo "[batch42-pathx-final] mode=$MODE dry_run=$DRY_RUN plan_only=$PLAN_ONLY"

if [ "$PLAN_ONLY" = "1" ]; then
  echo "[batch42-pathx-final] plan-only complete; QZ was not touched"
  exit 0
fi

MODE="$MODE" \
DRY_RUN="$DRY_RUN" \
RUN_TAG="$RUN_TAG" \
SMOKE_GATE_TAG="$SMOKE_GATE_TAG" \
bash "$MATERIALIZED_004094"

if [ "$DRY_RUN" = "1" ]; then
  echo "[batch42-pathx-final] platform dry-run passed; no live job submitted"
fi

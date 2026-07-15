#!/usr/bin/env bash
# Materialize and submit Batch-42 strict Path-X inference for effective-30k.
#
# This derives a 30k-effective copy from the audited 004094 Batch-42 Path-X
# submitter.  The protocol remains EN567/ZH1194 strict VC; the deliberate
# changes are:
#   * system_id path_x_3k -> path_x_final;
#   * checkpoint -> r3 weights-only warm-start local step 20000/effective 30000;
#   * frozen code root -> Batch37 eval snapshot;
#   * Path-X adapter -> 004132, plus a frozen 004093 base copy.
#
# Default is a platform dry-run:
#   MODE=smoke bash scripts/004133_submit_batch42_pathx_effective30k_qz.sh
#   MODE=full  bash scripts/004133_submit_batch42_pathx_effective30k_qz.sh
#
# Live submission:
#   MODE=smoke DRY_RUN=0 bash scripts/004133_submit_batch42_pathx_effective30k_qz.sh
#   MODE=full  DRY_RUN=0 bash scripts/004133_submit_batch42_pathx_effective30k_qz.sh

set -euo pipefail

CANONICAL_PROJECT_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
PROJECT_ROOT="${PROJECT_ROOT:-$CANONICAL_PROJECT_ROOT}"
SOURCE_004094="${SOURCE_004094:-$PROJECT_ROOT/scripts/004094_submit_batch42_pathx_strict_qz.sh}"
SOURCE_ADAPTER="${SOURCE_ADAPTER:-$PROJECT_ROOT/scripts/004132_run_batch42_pathx_effective30k_strict.py}"
SOURCE_BASE_004093="${SOURCE_BASE_004093:-$PROJECT_ROOT/scripts/004093_run_batch42_pathx_strict.py}"
PLAN_ROOT="${PLAN_ROOT:-$PROJECT_ROOT/trainset/qz_jobs/batch42_pathx_effective30k_materialized_20260714}"
MATERIALIZED_004094="$PLAN_ROOT/004094_pathx_effective30k.materialized.sh"
MATERIALIZATION_JSON="$PLAN_ROOT/materialization.json"

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
MODE="${MODE:-smoke}"
DRY_RUN="${DRY_RUN:-1}"
FORCE="${FORCE:-0}"
FORCE_MATERIALIZE="${FORCE_MATERIALIZE:-0}"
RUN_TAG="${RUN_TAG:-20260714_mtts_effective30k}"
SMOKE_GATE_TAG="${SMOKE_GATE_TAG:-20260714_mtts_effective30k}"

die() { echo "ERROR: $*" >&2; exit 2; }

case "$MODE" in smoke|full) ;; *) die "MODE must be smoke or full" ;; esac
case "$DRY_RUN:$FORCE:$FORCE_MATERIALIZE" in [01]:[01]:[01]) ;; *) die "boolean flags must be 0 or 1" ;; esac
[ "$PROJECT_ROOT" = "$CANONICAL_PROJECT_ROOT" ] || die "PROJECT_ROOT is hard-locked"
[ -x "$PYTHON" ] || die "Python is not executable: $PYTHON"
for path in "$SOURCE_004094" "$SOURCE_ADAPTER" "$SOURCE_BASE_004093"; do
  [ -s "$path" ] || die "missing input: $path"
done

mkdir -p "$PLAN_ROOT"

"$PYTHON" - "$SOURCE_004094" "$SOURCE_ADAPTER" "$SOURCE_BASE_004093" \
  "$MATERIALIZED_004094" "$MATERIALIZATION_JSON" "$FORCE_MATERIALIZE" <<'PY'
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path

source_94, source_adapter, source_base, output_path, audit_path = map(Path, sys.argv[1:6])
force = sys.argv[6] == "1"
old_model = (
    "$PROJECT_ROOT/outputs/lora_runs/ver23_content_side_3k_olddata_textrep10_"
    "ver23_content_side_text_bypass_3k_20260710/step-3000"
)
new_model = (
    "$PROJECT_ROOT/outputs/lora_runs/"
    "ver2_9_5_final_r3_v1_warmstart10k_to30k/step-20000"
)
old_code_root = (
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
    "MOSS-CodecVC_snapshots/ver23_batch3436_eval_20260711_1092820"
)
new_code_root = (
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
    "MOSS-CodecVC_snapshots/ver23_batch37_eval_20260711_1092820"
)

model_files = {
    "README.md": (179, "4d45f7d68a88a39671cc0cbc86f1acdfbee5351401eee2a97df253f0d077717f"),
    "adapter_config.json": (1179, "9d5fba10346b9b894b0e1eff6afddd0661f2bb3c5ae038a62d318ab1c40381f1"),
    "adapter_model.safetensors": (87366096, "b0df7ee1b39d4dfa5e513e569df7ef275f30dd043359eaa785572c288f7b0264"),
    "timbre_memory_config.json": (5023, "d8dad69f6523c67ecee9cce5900ae9809099b68655bf2e945713fb39f8271519"),
    "timbre_memory_adapter.pt": (1697093491, "f22ecf7dddd8f7994d4083af6e26afbe819ac56f402e8c490b19e1e4036b02ef"),
}
hash_replacements = {
    "06530eac22376a6befd9e81c95c333e4bb1c889de96e9059c2d5498cd90a7aee":
        model_files["adapter_config.json"][1],
    "3a51162fc7ccf1b9e1aa477ad7c44fa64390d109b8b63765a9cd636f090f4b25":
        model_files["adapter_model.safetensors"][1],
    "5c8842d87327c2cf1af2697725a19bf2b53ba654fa0a6b3f68b6a42fd50e9970":
        model_files["timbre_memory_config.json"][1],
    "020a16ad4bba5a812b2f62e29cb68dcec9d4055344e02de01555be8afd9d6895":
        model_files["timbre_memory_adapter.pt"][1],
    "c9dec31f4155d39cdbd02069dd8b91677ff5dee03e98d441377d949135a8e709":
        "22045797d68d54bc2b72c64773c43464e4164b19b3a29d97537149e15594fa1d",
    "5815c8ab5e0aab69d19328fd01782620064327eaf5f39cc4923df8ce3ae9ca42":
        "1d32527ec29fada353dc70b88a11cff972da901c5830dfeafb3bcf9f067d3ae3",
    "a8e4cd12d279cfff7c38e3e2d8b21b55d70c403cec654edf7ef77de58acba66a":
        "2be7b4cdf24c18df773b215ad3afe8682a65e519dee6ea81515ac4dd8b44ed1a",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"004094 anchor drift for {label}: count={count}")
    return text.replace(old, new)


if output_path.exists() and not force:
    existing = output_path.read_text(encoding="utf-8")
    expected_marker = "ver2_9_5_final_r3_v1_warmstart10k_to30k/step-20000"
    if expected_marker in existing and "SYSTEM_ID=\"path_x_final\"" in existing:
        raise SystemExit(0)
    raise SystemExit(f"materialized script exists but is not this route: {output_path}")

text = source_94.read_text(encoding="utf-8")
if text.count(old_model) != 2:
    raise SystemExit(f"old model anchor count drift: {text.count(old_model)}")
if old_code_root not in text:
    raise SystemExit("old code-root anchor absent")
text = text.replace(old_model, new_model)
text = text.replace(old_code_root, new_code_root)
text = text.replace("path_x_3k", "path_x_final")
for old, new in hash_replacements.items():
    if old not in text:
        raise SystemExit(f"hash anchor absent: {old}")
    text = text.replace(old, new)

model_block_pattern = re.compile(r"expected_model = \{\n.*?\n\}\nexpected_code = \{", re.DOTALL)
if len(model_block_pattern.findall(text)) != 1:
    raise SystemExit("expected_model block count drift")
model_lines = ["expected_model = {"]
for name, (size, sha) in model_files.items():
    model_lines.append(f"    {name!r}: ({size}, {sha!r}),")
model_lines.extend(["}", "expected_code = {"])
text = model_block_pattern.sub("\n".join(model_lines), text, count=1)

text = replace_once(
    text,
    'PATHX_SCRIPT="$SNAPSHOT_ROOT/scripts/004093_run_batch42_pathx_strict.py"',
    'PATHX_SCRIPT="$SNAPSHOT_ROOT/scripts/004093_run_batch42_pathx_strict.py"\n'
    'BASE_PATHX_SCRIPT="$SNAPSHOT_ROOT/scripts/004093_run_batch42_pathx_strict_base.py"',
    "snapshot base path",
)
text = replace_once(
    text,
    'SOURCE_PATHX_SCRIPT="$PROJECT_ROOT/scripts/004093_run_batch42_pathx_strict.py"',
    'SOURCE_PATHX_SCRIPT="$PROJECT_ROOT/scripts/004132_run_batch42_pathx_effective30k_strict.py"\n'
    'SOURCE_BASE_PATHX_SCRIPT="$PROJECT_ROOT/scripts/004093_run_batch42_pathx_strict.py"',
    "adapter source path",
)
text = replace_once(
    text,
    '  require_file "$SOURCE_PATHX_SCRIPT"\n  require_file "$SOURCE_MERGE_SCRIPT"',
    '  require_file "$SOURCE_PATHX_SCRIPT"\n'
    '  require_file "$SOURCE_BASE_PATHX_SCRIPT"\n'
    '  require_file "$SOURCE_MERGE_SCRIPT"',
    "snapshot source requirements",
)
text = replace_once(
    text,
    '  cp "$SOURCE_PATHX_SCRIPT" "$PATHX_SCRIPT"\n  cp "$SOURCE_MERGE_SCRIPT" "$MERGE_SCRIPT"',
    '  cp "$SOURCE_PATHX_SCRIPT" "$PATHX_SCRIPT"\n'
    '  cp "$SOURCE_BASE_PATHX_SCRIPT" "$BASE_PATHX_SCRIPT"\n'
    '  cp "$SOURCE_MERGE_SCRIPT" "$MERGE_SCRIPT"',
    "snapshot copies",
)
text = replace_once(
    text,
    '  chmod 0555 "$PATHX_SCRIPT" "$MERGE_SCRIPT" "$SCHEMA_SCRIPT" "$FROZEN_DRIVER"',
    '  chmod 0555 "$PATHX_SCRIPT" "$BASE_PATHX_SCRIPT" "$MERGE_SCRIPT" "$SCHEMA_SCRIPT" "$FROZEN_DRIVER"',
    "snapshot chmod",
)
text = replace_once(
    text,
    '  sha256sum "$PATHX_SCRIPT" "$MERGE_SCRIPT" "$SCHEMA_SCRIPT" "$FROZEN_DRIVER" "$RUNNER" > "$RECORD_ROOT/sha256sums.txt"',
    '  sha256sum "$PATHX_SCRIPT" "$BASE_PATHX_SCRIPT" "$MERGE_SCRIPT" "$SCHEMA_SCRIPT" "$FROZEN_DRIVER" "$RUNNER" > "$RECORD_ROOT/sha256sums.txt"',
    "snapshot sha",
)
text = replace_once(
    text,
    'require_file "$SOURCE_PATHX_SCRIPT"\nrequire_file "$SOURCE_MERGE_SCRIPT"',
    'require_file "$SOURCE_PATHX_SCRIPT"\n'
    'require_file "$SOURCE_BASE_PATHX_SCRIPT"\n'
    'require_file "$SOURCE_MERGE_SCRIPT"',
    "top-level source requirements",
)

output_path.parent.mkdir(parents=True, exist_ok=True)
tmp = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
tmp.write_text(text, encoding="utf-8")
os.replace(tmp, output_path)
output_path.chmod(0o555)
audit = {
    "schema_version": "moss_codecvc.batch42_pathx_effective30k_materialization.v1",
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "source_004094": str(source_94.resolve()),
    "source_004094_sha256": sha256_file(source_94),
    "source_adapter": str(source_adapter.resolve()),
    "source_adapter_sha256": sha256_file(source_adapter),
    "source_base_004093": str(source_base.resolve()),
    "source_base_004093_sha256": sha256_file(source_base),
    "materialized": str(output_path.resolve()),
    "materialized_sha256": sha256_file(output_path),
    "system_id": "path_x_final",
    "checkpoint": str(Path(new_model.replace("$PROJECT_ROOT", str(source_94.parents[1]))).resolve()),
    "checkpoint_semantics": "r3 weights-only warm-start local step 20000 / nominal effective step 30000",
    "model_files": {name: {"size": size, "sha256": sha} for name, (size, sha) in model_files.items()},
}
tmp_audit = audit_path.with_name(f".{audit_path.name}.tmp-{os.getpid()}")
tmp_audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp_audit, audit_path)
print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
PY

exec env \
  MODE="$MODE" \
  RUN_TAG="$RUN_TAG" \
  SMOKE_GATE_TAG="$SMOKE_GATE_TAG" \
  DRY_RUN="$DRY_RUN" \
  FORCE="$FORCE" \
  PROJECT_ROOT="$PROJECT_ROOT" \
  bash "$MATERIALIZED_004094"

#!/usr/bin/env bash
# Score path_x_final under Batch-42's strict three-scorer/two-ASR protocol,
# then (in a later invocation) publish the completed 8/8 paper table.
#
# Stage score (default dry-run, no live task):
#   STAGE=score bash scripts/004109_score_and_publish_batch42_pathx_final.sh
#
# Stage table (local, only after the scorer job completed):
#   STAGE=table bash scripts/004109_score_and_publish_batch42_pathx_final.sh
#
# Future live scorer submission is explicitly gated:
#   STAGE=score DRY_RUN=0 CONFIRM_BATCH44_FINAL_SCORERS=1 bash scripts/004109_...

set -euo pipefail

CANONICAL_PROJECT_ROOT="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
PROJECT_ROOT="${PROJECT_ROOT:-$CANONICAL_PROJECT_ROOT}"
RUN_TAG="${RUN_TAG:-20260713_mtts}"
STAGE="${STAGE:-score}"
DRY_RUN="${DRY_RUN:-1}"
CONFIRM_BATCH44_FINAL_SCORERS="${CONFIRM_BATCH44_FINAL_SCORERS:-0}"

PYTHON="${PYTHON:-/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python}"
INFERENCE_ROOT="${INFERENCE_ROOT:-$PROJECT_ROOT/testset/outputs/batch42_pathx_strict_path_x_final_${RUN_TAG}}"
INFERENCE_COMPLETION="${INFERENCE_COMPLETION:-$INFERENCE_ROOT/COMPLETED.json}"
EN_INPUT="$INFERENCE_ROOT/en/successful.jsonl"
ZH_INPUT="$INFERENCE_ROOT/zh/successful.jsonl"
SCORER_OUTPUT="${SCORER_OUTPUT:-$PROJECT_ROOT/testset/outputs/batch42_unified_scorers_path_x_final_${RUN_TAG}}"
SCORER_RECORD="${SCORER_RECORD:-$PROJECT_ROOT/trainset/qz_jobs/batch42_unified_scorers_path_x_final_${RUN_TAG}}"
TABLE_PREFIX="${TABLE_PREFIX:-$PROJECT_ROOT/testset/outputs/batch42_baseline_tables_20260711/batch42_baseline_final}"
FINAL_SELECTION_JSON="${FINAL_SELECTION_JSON:-$PROJECT_ROOT/testset/outputs/batch44_best3_20260713/FINAL_SELECTION.json}"
INTERIM_TABLE_JSON="${INTERIM_TABLE_JSON:-$PROJECT_ROOT/testset/outputs/batch42_baseline_tables_20260711/batch42_baseline_interim.json}"

SCORER_WRAPPER="$PROJECT_ROOT/scripts/004091_submit_batch42_unified_scorers_qz.sh"
TABLE_BUILDER="$PROJECT_ROOT/scripts/004092_build_batch42_baseline_tables.py"
FINAL_VALIDATOR="$PROJECT_ROOT/scripts/004106_run_batch42_pathx_final_strict.py"
PROVENANCE_HELPER="$PROJECT_ROOT/scripts/batch42_scorer_provenance.py"
EN_SUMMARY="$SCORER_OUTPUT/en/merged/path_x_final.en.merged.summary.json"
ZH_SUMMARY="$SCORER_OUTPUT/zh/merged/path_x_final.zh.merged.summary.json"
COMBINED_SUMMARY="$SCORER_OUTPUT/path_x_final.en_zh.summary.json"
EXPECTED_INTERIM_SHA256="76f89a236a17be9a944fdfbdf10f8aa58c436caefd9bde9486710158d6d00e9e"

die() { echo "ERROR: $*" >&2; exit 2; }
case "$STAGE" in score|table) ;; *) die "STAGE must be score or table" ;; esac
case "$DRY_RUN:$CONFIRM_BATCH44_FINAL_SCORERS" in [01]:[01]) ;; *) die "boolean flags must be 0 or 1" ;; esac
[ "$PROJECT_ROOT" = "$CANONICAL_PROJECT_ROOT" ] || die "PROJECT_ROOT is hard-locked"
[ -x "$PYTHON" ] || die "Python is not executable: $PYTHON"
[ -s "$SCORER_WRAPPER" ] || die "missing 004091 scorer wrapper"
[ -s "$TABLE_BUILDER" ] || die "missing 004092 table builder"
[ -s "$FINAL_VALIDATOR" ] || die "missing 004106 final provenance validator"
[ -s "$PROVENANCE_HELPER" ] || die "missing Batch-42 scorer provenance helper"
grep -Fq 'ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"' "$SCORER_WRAPPER" || \
  die "004091 lost the MTTS-3-2-0715 compute-group lock"
grep -Fq 'ALLOWED_SPEC="67b10bc6-78b0-41a3-aaf4-358eeeb99009"' "$SCORER_WRAPPER" || \
  die "004091 lost the registered 8xH200 spec lock"
grep -Fq 'ALLOWED_GPU_TYPE="NVIDIA_H200_SXM_141G"' "$SCORER_WRAPPER" || \
  die "004091 lost the H200 GPU-type lock"
if [ "$STAGE" = "score" ] && [ "$DRY_RUN" = "0" ] && [ "$CONFIRM_BATCH44_FINAL_SCORERS" != "1" ]; then
  die "live scorer submission requires CONFIRM_BATCH44_FINAL_SCORERS=1"
fi

validate_inference() {
  "$PYTHON" - "$INFERENCE_COMPLETION" "$INFERENCE_ROOT" "$EN_INPUT" "$ZH_INPUT" \
    "$FINAL_SELECTION_JSON" "$FINAL_VALIDATOR" <<'PY'
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

completion_path, root, en_path, zh_path, final_path, validator_path = map(Path, sys.argv[1:])
if not completion_path.is_file():
    raise SystemExit(f"missing strict inference completion: {completion_path}")
payload = json.loads(completion_path.read_text(encoding="utf-8"))
specification = importlib.util.spec_from_file_location(
    "batch44_final_publish_provenance_validator", validator_path
)
if specification is None or specification.loader is None:
    raise SystemExit(f"cannot import FINAL_SELECTION validator: {validator_path}")
validator = importlib.util.module_from_spec(specification)
sys.modules[specification.name] = validator
specification.loader.exec_module(validator)
final = validator.load_final_selection(final_path, verify_checkpoint_hashes=True)
if final.get("schema_version") != "moss_codecvc.batch44_v1_final_selection.v1" or final.get("status") != "final":
    raise SystemExit("invalid Batch-44 v1 FINAL_SELECTION.json")
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
    raise SystemExit("FINAL_SELECTION is not a registered 26k/28k/30k Best3 point")
if candidate.get("candidate_id") != f"{arm}_step-{step}":
    raise SystemExit("FINAL_SELECTION candidate_id drift")
if candidate.get("text_repeat") != expected_repeats[arm]:
    raise SystemExit("FINAL_SELECTION text_repeat drift")
if candidate.get("train_job_id") != expected_train_jobs[arm]:
    raise SystemExit("FINAL_SELECTION training job drift")
checkpoint = Path(str(candidate.get("checkpoint_path") or "")).resolve()
expected_checkpoint = (
    Path("/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC")
    / "outputs/lora_runs" / run_dirs[arm] / f"step-{step}"
).resolve()
if checkpoint != expected_checkpoint:
    raise SystemExit("FINAL_SELECTION checkpoint path drift")
required_model_files = {
    "README.md", "adapter_config.json", "adapter_model.safetensors",
    "timbre_memory_config.json", "timbre_memory_adapter.pt",
}
candidate_files = candidate.get("model_files") or {}
if set(candidate_files) != required_model_files:
    raise SystemExit("FINAL_SELECTION model file registration is incomplete")
final_sha = hashlib.sha256(final_path.read_bytes()).hexdigest()
if payload.get("schema_version") != "moss_codecvc.batch42_pathx_strict_completion.v1":
    raise SystemExit(f"wrong inference completion schema: {payload.get('schema_version')!r}")
if payload.get("status") != "complete" or payload.get("system_id") != "path_x_final":
    raise SystemExit("strict inference is not a complete path_x_final result")
if Path(str(payload.get("output_root") or "")).resolve() != root.resolve():
    raise SystemExit("strict inference output_root drift")
if payload.get("resource_contract") != {
    "compute_group": "MTTS-3-2-0715",
    "gpu_type": "NVIDIA_H200_SXM_141G",
    "gpus": 8,
    "instances": 1,
}:
    raise SystemExit("strict inference resource contract is not MTTS 1x8 H200")
identity = payload.get("registered_identity") or {}
if identity.get("schema_version") != "moss_codecvc.batch42_pathx_registered_identity.v1" or identity.get("status") != "verified":
    raise SystemExit("strict inference registered identity is invalid")
if Path(str(identity.get("model_path") or "")).resolve() != checkpoint:
    raise SystemExit("strict inference checkpoint differs from FINAL_SELECTION")
expected_code_root = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
    "MOSS-CodecVC_snapshots/ver23_batch37_eval_20260711_1092820"
)
if Path(str(identity.get("code_root") or "")).resolve() != expected_code_root:
    raise SystemExit("strict inference frozen code root drift")
registered_files = identity.get("model_files") or {}
if set(registered_files) != required_model_files:
    raise SystemExit("strict inference registered model file set drift")
for name, registration in candidate_files.items():
    actual = registered_files.get(name) or {}
    if actual.get("size") != registration.get("size") or actual.get("sha256") != registration.get("sha256"):
        raise SystemExit(f"strict inference model identity drift for {name}")
smoke = payload.get("smoke_gate") or {}
marker = Path(str(smoke.get("marker") or "")).resolve()
fingerprint = str(smoke.get("protocol_fingerprint_sha256") or "")
if not marker.is_file() or len(fingerprint) != 64:
    raise SystemExit("strict inference smoke gate is missing")
smoke_payload = json.loads(marker.read_text(encoding="utf-8"))
if (
    smoke_payload.get("schema_version") != "moss_codecvc.batch42_pathx_strict_smoke_completion.v1"
    or smoke_payload.get("status") != "smoke_complete"
    or smoke_payload.get("system_id") != "path_x_final"
    or smoke_payload.get("protocol_fingerprint_sha256") != fingerprint
    or smoke_payload.get("resource_contract") != payload.get("resource_contract")
):
    raise SystemExit("strict inference smoke protocol/resource identity drift")
contract = smoke_payload.get("protocol_contract") or {}
if contract.get("system_id") != "path_x_final" or (contract.get("strict_manifest_sha256") or {}) != {
    "en": "48549d8029e680d74656660191c4641ca5a8040ccbe3252ce89bfc3b0c9c75ae",
    "zh": "4b637cc1cff33dc369954755538d12396fc92d439a52742103a29b7c563cf6df",
}:
    raise SystemExit("strict inference manifest fingerprint drift")
for language, expected, path in (("en", 567, en_path), ("zh", 1194, zh_path)):
    item = (payload.get("strict_sets") or {}).get(language) or {}
    if item.get("registered_cases") != expected:
        raise SystemExit(f"{language} completion denominator drift")
    if Path(str(item.get("successful_jsonl") or "")).resolve() != path.resolve():
        raise SystemExit(f"{language} successful_jsonl path drift")
    if not path.is_file():
        raise SystemExit(f"missing {language} successful JSONL: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected or len({row.get("case_id") for row in rows}) != expected:
        raise SystemExit(f"{language} rows/unique != {expected}")
    if any(row.get("system_id") != "path_x_final" or row.get("status") not in {"ok", "skipped_existing"} for row in rows):
        raise SystemExit(f"{language} contains wrong system/status rows")
    for row in rows:
        inference = ((row.get("provenance") or {}).get("inference_config") or {})
        if (
            Path(str(inference.get("model_path") or "")).resolve() != checkpoint
            or Path(str(inference.get("code_root") or "")).resolve() != expected_code_root
            or inference.get("final_selection_sha256") != final_sha
            or float(inference.get("ref_audio_cfg_scale", -1)) != 1.0
        ):
            raise SystemExit(f"{language}/{row.get('case_id')}: row provenance differs from final selection")
print("[batch42-pathx-final-input] PASS EN567 ZH1194 system_id=path_x_final")
PY
}

if [ "$STAGE" = "score" ]; then
  validate_inference
  SYSTEM_TAG=path_x_final \
  INPUT_SYSTEM_ID=path_x_final \
  EN_INPUT="$EN_INPUT" \
  ZH_INPUT="$ZH_INPUT" \
  OUTPUT_ROOT="$SCORER_OUTPUT" \
  RECORD_ROOT="$SCORER_RECORD" \
  SOURCE_INFERENCE_COMPLETION="$INFERENCE_COMPLETION" \
  SOURCE_FINAL_SELECTION="$FINAL_SELECTION_JSON" \
  RUN_TAG="$RUN_TAG" \
  ENABLE_QWEN_ASR=0 \
  DRY_RUN="$DRY_RUN" \
  bash "$SCORER_WRAPPER"
  if [ "$DRY_RUN" = "1" ]; then
    echo "[batch42-pathx-final-score] platform dry-run passed; no live job submitted"
  fi
  exit 0
fi

validate_inference

for path in \
  "$SCORER_OUTPUT/completion.json" "$COMBINED_SUMMARY" \
  "$EN_SUMMARY" "$ZH_SUMMARY" "$SCORER_RECORD/submitted_jobs.tsv"; do
  [ -s "$path" ] || die "missing completed final scorer artifact: $path"
done

"$PYTHON" - "$TABLE_BUILDER" "$INTERIM_TABLE_JSON" "$EN_SUMMARY" "$ZH_SUMMARY" \
  "$TABLE_PREFIX" "$SCORER_RECORD" "$SCORER_OUTPUT/completion.json" \
  "$EN_INPUT" "$ZH_INPUT" "$SCORER_OUTPUT" "$PROVENANCE_HELPER" \
  "$INFERENCE_COMPLETION" "$FINAL_SELECTION_JSON" "$EXPECTED_INTERIM_SHA256" <<'PY'
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

builder_path = Path(sys.argv[1])
interim_path = Path(sys.argv[2])
en_final = Path(sys.argv[3])
zh_final = Path(sys.argv[4])
output_prefix = Path(sys.argv[5])
scorer_record = Path(sys.argv[6])
scorer_completion_path = Path(sys.argv[7])
expected_en_input = Path(sys.argv[8]).resolve()
expected_zh_input = Path(sys.argv[9]).resolve()
expected_scorer_output = Path(sys.argv[10]).resolve()
provenance_helper_path = Path(sys.argv[11])
expected_inference_completion = Path(sys.argv[12]).resolve()
expected_final_selection = Path(sys.argv[13]).resolve()
expected_interim_sha256 = sys.argv[14]
ledger = scorer_record / "submitted_jobs.tsv"

spec = importlib.util.spec_from_file_location(
    "batch42_pathx_final_scorer_provenance", provenance_helper_path
)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot import scorer provenance helper: {provenance_helper_path}")
provenance = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = provenance
spec.loader.exec_module(provenance)
try:
    scorer_attestation = provenance.verify_final_bundle(
        completion_path=scorer_completion_path,
        ledger_path=ledger,
        expected_output_root=expected_scorer_output,
        expected_en_input=expected_en_input,
        expected_zh_input=expected_zh_input,
        expected_inference_completion=expected_inference_completion,
        expected_final_selection=expected_final_selection,
    )
except provenance.ProvenanceError as exc:
    raise SystemExit(f"path_x_final scorer provenance verification failed: {exc}")

if provenance.sha256_file(interim_path) != expected_interim_sha256:
    raise SystemExit("frozen Batch-42 7/8 interim table SHA256 drift")
interim = json.loads(interim_path.read_text(encoding="utf-8"))
if interim.get("status") != "interim" or interim.get("counts") != {
    "systems": 8, "complete": 7, "partial": 0, "pending": 1
}:
    raise SystemExit("the frozen Batch-42 interim table is not 7/8")
old_rows = {row["system_id"]: row for row in interim.get("systems", [])}
if old_rows.get("path_x_final", {}).get("status") != "pending":
    raise SystemExit("the frozen interim path_x_final row is not pending")

builder_spec = importlib.util.spec_from_file_location(
    "batch42_table_builder_frozen_inputs", builder_path
)
if builder_spec is None or builder_spec.loader is None:
    raise SystemExit(f"cannot import table builder: {builder_path}")
module = importlib.util.module_from_spec(builder_spec)
sys.modules[builder_spec.name] = module
builder_spec.loader.exec_module(module)
stage_root = output_prefix.parent / f".{output_prefix.name}.candidate-{os.getpid()}"
stage_prefix = stage_root / output_prefix.name
stage_paths = [
    stage_prefix.with_suffix(".md"),
    stage_prefix.with_suffix(".json"),
    stage_prefix.with_suffix(".tsv"),
    stage_prefix.with_name(stage_prefix.name + ".cross_validation.tsv"),
]
for path in stage_paths:
    path.unlink(missing_ok=True)
argv = ["--no-discovery", "--allow-path-x-final", "--output-prefix", str(stage_prefix)]
for system_id, row in old_rows.items():
    if system_id == "path_x_final":
        continue
    if row.get("status") != "complete":
        raise SystemExit(f"frozen interim row is not complete: {system_id}")
    for language in ("en", "zh"):
        summary = ((row.get("metrics") or {}).get(language) or {}).get("summary_path")
        if not summary:
            raise SystemExit(f"frozen interim summary is missing: {system_id}/{language}")
        argv.extend([f"--{language}-summary", f"{system_id}={summary}"])
argv.extend(["--en-summary", f"path_x_final={en_final}"])
argv.extend(["--zh-summary", f"path_x_final={zh_final}"])
args = module.build_parser().parse_args(argv)
payload, outputs = module.run(args)
if payload.get("status") != "complete" or payload.get("counts") != {
    "systems": 8, "complete": 8, "partial": 0, "pending": 0
}:
    raise SystemExit(f"Batch-42 table is not complete 8/8: {payload.get('counts')}")
rows = {row["system_id"]: row for row in payload.get("systems", [])}
if [row.get("system_id") for row in payload.get("systems", [])] != [
    row.get("system_id") for row in interim.get("systems", [])
]:
    raise SystemExit("Batch-42 system row order drift while publishing final")
for system_id, old in old_rows.items():
    if system_id == "path_x_final":
        continue
    new = rows.get(system_id) or {}
    if new != old:
        raise SystemExit(f"frozen Batch-42 row drift while publishing final: {system_id}")
old_main = {row["system_id"]: row for row in interim.get("main_table", [])}
new_main = {row["system_id"]: row for row in payload.get("main_table", [])}
old_cross = {
    (row["system_id"], row["split"]): row
    for row in interim.get("cross_validation_table", [])
}
new_cross = {
    (row["system_id"], row["split"]): row
    for row in payload.get("cross_validation_table", [])
}
for system_id in old_rows:
    if system_id == "path_x_final":
        continue
    if new_main.get(system_id) != old_main.get(system_id):
        raise SystemExit(f"frozen Batch-42 main-table row drift: {system_id}")
    for split in ("EN567", "ZH1194"):
        if new_cross.get((system_id, split)) != old_cross.get((system_id, split)):
            raise SystemExit(
                f"frozen Batch-42 cross-validation row drift: {system_id}/{split}"
            )
final = rows.get("path_x_final") or {}
if final.get("status") != "complete":
    raise SystemExit("path_x_final row is not complete")
for language, expected in (("en", 567), ("zh", 1194)):
    item = (final.get("metrics") or {}).get(language) or {}
    if item.get("status") != "complete" or item.get("n_cases") != expected:
        raise SystemExit(f"path_x_final {language} denominator/status drift")
expected_summary_paths = {
    "en": en_final.resolve(),
    "zh": zh_final.resolve(),
}
for language, path in expected_summary_paths.items():
    registered = Path(final["metrics"][language]["summary_path"]).resolve()
    if registered != path:
        raise SystemExit(f"path_x_final {language} summary-path drift")

final_paths = {
    "markdown": output_prefix.with_suffix(".md"),
    "json": output_prefix.with_suffix(".json"),
    "main_tsv": output_prefix.with_suffix(".tsv"),
    "cross_validation_tsv": output_prefix.with_name(
        output_prefix.name + ".cross_validation.tsv"
    ),
}
for key, stage_path_raw in outputs.items():
    stage_path = Path(stage_path_raw)
    if stage_path.resolve() != stage_paths[
        {"markdown": 0, "json": 1, "main_tsv": 2, "cross_validation_tsv": 3}[key]
    ].resolve():
        raise SystemExit(f"unexpected staged table path for {key}: {stage_path}")

def staged_ref(stage_path, final_path):
    stage_path = Path(stage_path)
    return {
        "path": str(Path(final_path).resolve()),
        "size": stage_path.stat().st_size,
        "sha256": provenance.sha256_file(stage_path),
    }

table_refs = {
    key: staged_ref(Path(outputs[key]), final_paths[key]) for key in final_paths
}
frozen_rows = [
    row for row in interim.get("systems", []) if row.get("system_id") != "path_x_final"
]
published_rows = [
    row for row in payload.get("systems", []) if row.get("system_id") != "path_x_final"
]
canonical_frozen = json.dumps(
    frozen_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
).encode("utf-8")
canonical_published = json.dumps(
    published_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
).encode("utf-8")
if canonical_frozen != canonical_published:
    raise SystemExit("canonical frozen 7-row payload drift")
publication = {
    "schema_version": "moss_codecvc.batch42_pathx_final_table_publication.v1",
    "status": "complete",
    "system_id": "path_x_final",
    "counts": payload["counts"],
    "scorer_attestation": scorer_attestation,
    "frozen_interim": provenance.file_ref(interim_path),
    "frozen_nonfinal_rows_sha256": hashlib.sha256(canonical_frozen).hexdigest(),
    "published_nonfinal_rows_sha256": hashlib.sha256(canonical_published).hexdigest(),
    "table_artifacts": table_refs,
}

# The builder only wrote hidden candidate files.  Promote all four verified
# tables atomically one-by-one, then write the provenance marker last.  A
# consumer must require the marker and verify its hashes before using a table.
for key, final_path in final_paths.items():
    final_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(Path(outputs[key]), final_path)
stage_root.rmdir()
provenance_path = output_prefix.with_suffix(".provenance.json")
provenance.atomic_write_json(provenance_path, publication)
for key, ref in table_refs.items():
    provenance._require_ref(ref, label=f"published table {key}")
print(
    f"[batch42-pathx-final-table] PASS complete=8/8 "
    f"output={final_paths['json']} provenance={provenance_path}"
)
PY

echo "[batch42-pathx-final-table] markdown=${TABLE_PREFIX}.md"
echo "[batch42-pathx-final-table] provenance=${TABLE_PREFIX}.provenance.json"

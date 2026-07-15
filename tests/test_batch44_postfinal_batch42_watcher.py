from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/004115_watch_batch44_postfinal_batch42_publish.py"
COMPUTE_ID = "lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"
SPEC = "67b10bc6-78b0-41a3-aaf4-358eeeb99009"
GPU = "NVIDIA_H200_SXM_141G"
WORKSPACE = "ws-8207e9e2-e733-4eec-a475-cfa1c36480ba"
PROJECT_ID = "project-c67c548f-f02c-453b-ba5b-8745db6886e7"
SMOKE_JOB = "job-11111111-1111-1111-1111-111111111111"
FULL_JOB = "job-22222222-2222-2222-2222-222222222222"
SCORE_JOB = "job-33333333-3333-3333-3333-333333333333"
EN_N = 3
ZH_N = 4
EN_HASH = "a" * 64
ZH_HASH = "b" * 64


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def final_payload(project: Path) -> dict:
    checkpoint = project / "outputs/lora_runs/ver2_9_5_final_r3_v1_30k/step-26000"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model_files = {}
    for index, name in enumerate(
        (
            "README.md",
            "adapter_config.json",
            "adapter_model.safetensors",
            "timbre_memory_config.json",
            "timbre_memory_adapter.pt",
        ),
        start=1,
    ):
        path = checkpoint / name
        path.write_bytes(bytes([index]) * (10 + index))
        model_files[name] = {"size": path.stat().st_size, "sha256": sha(path)}
    return {
        "schema_version": "moss_codecvc.batch44_v1_final_selection.v1",
        "experiment_id": "batch44_v1",
        "data_version": "v1_20260709",
        "status": "final",
        "system_id": "path_x_final",
        "candidate": {
            "candidate_id": "r3_step-26000",
            "arm": "r3",
            "text_repeat": 3,
            "step": 26000,
            "train_job_id": "job-2b91d332-d500-4279-84f9-0a6a81a376aa",
            "checkpoint_path": str(checkpoint.resolve()),
            "model_files": model_files,
        },
    }


def registered_identity(final: dict, code_root: Path) -> dict:
    return {
        "schema_version": "moss_codecvc.batch42_pathx_registered_identity.v1",
        "status": "verified",
        "model_path": final["candidate"]["checkpoint_path"],
        "code_root": str(code_root.resolve()),
        "model_files": final["candidate"]["model_files"],
    }


def make_interim(path: Path) -> None:
    rows = []
    for index in range(7):
        rows.append(
            {
                "system_id": f"baseline_{index}",
                "status": "complete",
                "metrics": {
                    "en": {"status": "complete", "n_cases": EN_N, "value": index},
                    "zh": {"status": "complete", "n_cases": ZH_N, "value": index},
                },
            }
        )
    rows.append({"system_id": "path_x_final", "status": "pending", "metrics": {}})
    write_json(
        path,
        {
            "status": "interim",
            "counts": {"systems": 8, "complete": 7, "partial": 0, "pending": 1},
            "systems": rows,
        },
    )


def make_tools(project: Path, final_path: Path, code_root: Path) -> dict[str, Path]:
    tools = project / "tools"
    validator = tools / "validator.py"
    inference = tools / "inference.sh"
    score = tools / "score.sh"
    qz = tools / "qzcli"
    scorer_provenance = tools / "batch42_scorer_provenance.py"

    write_executable(
        validator,
        """from pathlib import Path
import json
def validate_final_selection_provenance(path, **kwargs):
    return json.loads(Path(path).read_text())
""",
    )
    write_executable(
        scorer_provenance,
        '''from pathlib import Path
import hashlib, json
LEDGER_FIELDS = [
    "job_name", "job_id", "system_tag", "compute_group", "compute_group_name",
    "spec", "instances", "gpu_type", "gpus", "en_input", "en_input_sha256",
    "zh_input", "zh_input_sha256", "source_inference_completion",
    "source_inference_completion_sha256", "source_final_selection",
    "source_final_selection_sha256", "output_root", "snapshot_root",
    "input_provenance", "input_provenance_sha256", "submission_contract",
    "submission_contract_sha256", "submit_output", "submit_output_sha256",
]
class ProvenanceError(ValueError): pass
def sha256_file(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def validate_input_provenance(path, **kwargs): return json.loads(Path(path).read_text())
def validate_submission_contract(path, **kwargs): return json.loads(Path(path).read_text())
def verify_final_bundle(completion_path, **kwargs):
    payload = json.loads(Path(completion_path).read_text())
    return {
      "schema_version": "moss_codecvc.batch42_pathx_final_scorer_attestation.v1",
      "status": "verified", "job_id": payload["job_id"],
      "resource_contract": payload["resource_contract"],
    }
''',
    )
    write_executable(
        inference,
        r'''#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$MODE" >> "${FAKE_WRAPPER_CALLS:?}"
if [ "$MODE" = smoke ]; then
  record=${SMOKE_RECORD_ROOT:?}
  output=${SMOKE_OUTPUT_ROOT:?}
  name="batch42_pathx_strict_smoke_${SMOKE_GATE_TAG}"
  job=job-11111111-1111-1111-1111-111111111111
else
  record=${FULL_RECORD_ROOT:?}
  output=${INFERENCE_ROOT:?}
  name="batch42_pathx_strict_path_x_final_${RUN_TAG}"
  job=job-22222222-2222-2222-2222-222222222222
fi
mkdir -p "$record"
entrypoint="$record/run_batch42_pathx_strict_entrypoint.sh"
printf '#!/usr/bin/env bash\n' > "$entrypoint"
chmod +x "$entrypoint"
{
  printf 'job_name\tjob_id\tmode\tsystem\tcompute_group\tspec\tinstances\tgpu_type\toutput_root\trecord_root\tentrypoint\n'
  printf '%s\t%s\t%s\tpath_x_final\tlcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122\t67b10bc6-78b0-41a3-aaf4-358eeeb99009\t1\tNVIDIA_H200_SXM_141G\t%s\t%s\t%s\n' \
    "$name" "$job" "$MODE" "$output" "$record" "$entrypoint"
} > "$record/submitted_jobs.tsv"
''',
    )
    write_executable(
        score,
        r'''#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$STAGE" >> "${FAKE_WRAPPER_CALLS:?}"
if [ "$STAGE" = score ]; then
  record=${SCORER_RECORD:?}
  mkdir -p "$record/record_snapshot/scripts"
  frozen="$record/record_snapshot/scripts/004091_submit_batch42_unified_scorers_qz.sh"
  printf '#!/usr/bin/env bash\n' > "$frozen"
  chmod +x "$frozen"
  cp "${SCORER_PROVENANCE_HELPER:?}" "$record/record_snapshot/scripts/batch42_scorer_provenance.py"
  printf 'submitted job-33333333-3333-3333-3333-333333333333\n' > "$record/submit_output.txt"
  "${PYTHON:?}" - "$record" "$INFERENCE_ROOT" "$SCORER_OUTPUT" "$RUN_TAG" \
    "$FINAL_SELECTION_JSON" "$SCORER_PROVENANCE_HELPER" <<'PY'
import csv, hashlib, json, sys
from pathlib import Path
record, inference, output = map(Path, sys.argv[1:4])
run_tag, final, helper = sys.argv[4], Path(sys.argv[5]), Path(sys.argv[6])
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
resource = {
 "workspace_id": "ws-8207e9e2-e733-4eec-a475-cfa1c36480ba",
 "project_id": "project-c67c548f-f02c-453b-ba5b-8745db6886e7",
 "compute_group_id": "lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122",
 "compute_group_name": "MTTS-3-2-0715",
 "spec_id": "67b10bc6-78b0-41a3-aaf4-358eeeb99009",
 "gpu_type": "NVIDIA_H200_SXM_141G", "instances": 1, "gpus": 8, "shards": 8,
}
en, zh = inference / "en/successful.jsonl", inference / "zh/successful.jsonl"
full = inference / "COMPLETED.json"
input_path = record / "input_provenance.json"
submission_path = record / "submission_contract.json"
submit_output = record / "submit_output.txt"
input_path.write_text(json.dumps({"resource_contract": resource}) + "\n")
submission_path.write_text(json.dumps({
 "job_id": "job-33333333-3333-3333-3333-333333333333",
 "job_name": f"batch42_score_path_x_final_{run_tag}",
 "resource_contract": resource,
}) + "\n")
fields = [
 "job_name", "job_id", "system_tag", "compute_group", "compute_group_name",
 "spec", "instances", "gpu_type", "gpus", "en_input", "en_input_sha256",
 "zh_input", "zh_input_sha256", "source_inference_completion",
 "source_inference_completion_sha256", "source_final_selection",
 "source_final_selection_sha256", "output_root", "snapshot_root",
 "input_provenance", "input_provenance_sha256", "submission_contract",
 "submission_contract_sha256", "submit_output", "submit_output_sha256",
]
row = {
 "job_name": f"batch42_score_path_x_final_{run_tag}",
 "job_id": "job-33333333-3333-3333-3333-333333333333",
 "system_tag": "path_x_final", "compute_group": resource["compute_group_id"],
 "compute_group_name": resource["compute_group_name"], "spec": resource["spec_id"],
 "instances": "1", "gpu_type": resource["gpu_type"], "gpus": "8",
 "en_input": str(en.resolve()), "en_input_sha256": sha(en),
 "zh_input": str(zh.resolve()), "zh_input_sha256": sha(zh),
 "source_inference_completion": str(full.resolve()),
 "source_inference_completion_sha256": sha(full),
 "source_final_selection": str(final.resolve()), "source_final_selection_sha256": sha(final),
 "output_root": str(output.resolve()), "snapshot_root": str((record / "record_snapshot").resolve()),
 "input_provenance": str(input_path.resolve()), "input_provenance_sha256": sha(input_path),
 "submission_contract": str(submission_path.resolve()), "submission_contract_sha256": sha(submission_path),
 "submit_output": str(submit_output.resolve()), "submit_output_sha256": sha(submit_output),
}
with (record / "submitted_jobs.tsv").open("w", newline="") as handle:
 writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
 writer.writeheader(); writer.writerow(row)
PY
  exit 0
fi
if [ "${FAKE_TABLE_FAIL:-0}" = 1 ]; then
  exit 19
fi
"${PYTHON:?}" - "$INTERIM_TABLE_JSON" "$TABLE_PREFIX" "${BATCH44_POSTFINAL_TEST_EN_EXPECTED:?}" "${BATCH44_POSTFINAL_TEST_ZH_EXPECTED:?}" <<'PY'
import json, os, sys
from pathlib import Path
interim, prefix = Path(sys.argv[1]), Path(sys.argv[2])
en_n, zh_n = int(sys.argv[3]), int(sys.argv[4])
payload = json.loads(interim.read_text())
payload["status"] = "complete"
payload["counts"] = {"systems": 8, "complete": 8, "partial": 0, "pending": 0}
for row in payload["systems"]:
    if row["system_id"] == "path_x_final":
        row["status"] = "complete"
        row["metrics"] = {
            "en": {"status": "complete", "n_cases": en_n},
            "zh": {"status": "complete", "n_cases": zh_n},
        }
prefix.parent.mkdir(parents=True, exist_ok=True)
paths = {
 "json": prefix.with_suffix(".json"),
 "markdown": prefix.with_suffix(".md"),
 "main_tsv": prefix.with_suffix(".tsv"),
 "cross_validation_tsv": prefix.with_name(prefix.name + ".cross_validation.tsv"),
}
paths["json"].write_text(json.dumps(payload, indent=2) + "\n")
paths["markdown"].write_text("# complete 8/8\n")
paths["main_tsv"].write_text("system\tstatus\npath_x_final\tcomplete\n")
paths["cross_validation_tsv"].write_text("system\tsplit\npath_x_final\tEN567\n")
def ref(path):
 import hashlib
 return {"path": str(path.resolve()), "size": path.stat().st_size,
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
resource = {
 "workspace_id": "ws-8207e9e2-e733-4eec-a475-cfa1c36480ba",
 "project_id": "project-c67c548f-f02c-453b-ba5b-8745db6886e7",
 "compute_group_id": "lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122",
 "compute_group_name": "MTTS-3-2-0715",
 "spec_id": "67b10bc6-78b0-41a3-aaf4-358eeeb99009",
 "gpu_type": "NVIDIA_H200_SXM_141G", "instances": 1, "gpus": 8, "shards": 8,
}
scorer = json.loads((Path(os.environ["SCORER_OUTPUT"]) / "completion.json").read_text())
frozen = ref(interim)
marker = {
 "schema_version": "moss_codecvc.batch42_pathx_final_table_publication.v1",
 "status": "complete", "system_id": "path_x_final", "counts": payload["counts"],
 "scorer_attestation": {"status": "verified", "job_id": scorer["job_id"],
                         "resource_contract": resource},
 "frozen_interim": frozen,
 "frozen_nonfinal_rows_sha256": "f" * 64,
 "published_nonfinal_rows_sha256": "f" * 64,
 "table_artifacts": {key: ref(path) for key, path in paths.items()},
}
prefix.with_suffix(".provenance.json").write_text(json.dumps(marker, indent=2) + "\n")
PY
''',
    )
    write_executable(
        qz,
        r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
assert sys.argv[1] == "status" and sys.argv[3] == "--json"
job = sys.argv[2]
statuses = json.loads(Path(os.environ["FAKE_QZ_STATUSES"]).read_text())
project = Path(os.environ["PROJECT_ROOT"])
run_tag = os.environ["RUN_TAG"]
smoke_tag = os.environ["SMOKE_GATE_TAG"]
if job.endswith("111111111111"):
    name = f"batch42_pathx_strict_smoke_{smoke_tag}"
    command = f"bash {os.environ['SMOKE_RECORD_ROOT']}/run_batch42_pathx_strict_entrypoint.sh"
elif job.endswith("222222222222"):
    name = f"batch42_pathx_strict_path_x_final_{run_tag}"
    command = f"bash {os.environ['FULL_RECORD_ROOT']}/run_batch42_pathx_strict_entrypoint.sh"
else:
    name = f"batch42_score_path_x_final_{run_tag}"
    command = f"env X=1 bash {os.environ['SCORER_RECORD']}/record_snapshot/scripts/004091_submit_batch42_unified_scorers_qz.sh"
payload = {
  "job_id": job,
  "name": name,
  "workspace_id": "ws-8207e9e2-e733-4eec-a475-cfa1c36480ba",
  "project_id": "project-c67c548f-f02c-453b-ba5b-8745db6886e7",
  "logic_compute_group_id": "lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122",
  "logic_compute_group_name": "MTTS-3-2-0715",
  "status": statuses[job],
  "command": command,
  "framework_config": [{
    "instance_count": 1,
    "gpu_count": 8,
    "instance_spec_price_info": {
      "quota_id": "67b10bc6-78b0-41a3-aaf4-358eeeb99009",
      "gpu_info": {"gpu_type": "NVIDIA_H200_SXM_141G"},
    },
  }],
}
with Path(os.environ["FAKE_QZ_CALLS"]).open("a") as handle:
    handle.write(job + "\n")
print("rich panel before json")
print(json.dumps(payload, indent=2))
''',
    )
    return {
        "validator": validator,
        "inference": inference,
        "score": score,
        "qz": qz,
        "scorer_provenance": scorer_provenance,
    }


def make_materialization(env: dict[str, str], final: dict) -> None:
    plan = Path(env["PLAN_ROOT"])
    materialized = plan / "004094_pathx_final.materialized.sh"
    write_executable(materialized, "#!/usr/bin/env bash\n")
    final_path = Path(env["FINAL_SELECTION_JSON"])
    write_json(
        plan / "materialization.json",
        {
            "schema_version": "moss_codecvc.batch42_pathx_final_batch44_v1_materialization.v1",
            "status": "ready",
            "system_id": "path_x_final",
            "experiment_id": "batch44_v1",
            "data_version": "v1_20260709",
            "final_selection": str(final_path.resolve()),
            "final_selection_sha256": sha(final_path),
            "checkpoint": final["candidate"]["checkpoint_path"],
            "frozen_eval_code_root": env["EXPECTED_EVAL_CODE_ROOT"],
            "materialized_004094": str(materialized.resolve()),
            "materialized_004094_sha256": sha(materialized),
            "resource_contract": {
                "compute_group": "MTTS-3-2-0715",
                "compute_group_id": COMPUTE_ID,
                "spec": SPEC,
                "gpu_type": GPU,
                "instances": 1,
                "gpus": 8,
            },
        },
    )


def setup_project(tmp_path: Path, *, with_final: bool = True) -> tuple[Path, dict[str, str], dict]:
    project = tmp_path / "project"
    project.mkdir()
    final = final_payload(project)
    final_path = project / "testset/outputs/batch44_best3_20260713/FINAL_SELECTION.json"
    if with_final:
        write_json(final_path, final)
    code_root = project / "eval_snapshot"
    code_root.mkdir()
    tools = make_tools(project, final_path, code_root)
    interim = project / "testset/outputs/tables/interim.json"
    make_interim(interim)
    qz_home = project / "qz_home"
    qz_home.mkdir()
    status_path = project / "statuses.json"
    write_json(status_path, {})
    calls = project / "wrapper_calls.txt"
    qz_calls = project / "qz_calls.txt"
    calls.write_text("")
    qz_calls.write_text("")
    env = os.environ.copy()
    env.update(
        {
            "BATCH44_POSTFINAL_TEST_MODE": "1",
            "PROJECT_ROOT": str(project),
            "FINAL_SELECTION_JSON": str(final_path),
            "FINAL_VALIDATOR": str(tools["validator"]),
            "INFERENCE_WRAPPER": str(tools["inference"]),
            "SCORE_WRAPPER": str(tools["score"]),
            "SCORER_PROVENANCE_HELPER": str(tools["scorer_provenance"]),
            "QZCLI": str(tools["qz"]),
            "QZCLI_HOME": str(qz_home),
            "RUN_TAG": "test_run",
            "SMOKE_GATE_TAG": "test_smoke",
            "STATE_ROOT": str(project / "state"),
            "PLAN_ROOT": str(project / "plan"),
            "SMOKE_OUTPUT_ROOT": str(project / "outputs/smoke"),
            "SMOKE_RECORD_ROOT": str(project / "records/smoke"),
            "INFERENCE_ROOT": str(project / "outputs/full"),
            "FULL_RECORD_ROOT": str(project / "records/full"),
            "SCORER_OUTPUT": str(project / "outputs/score"),
            "SCORER_RECORD": str(project / "records/score"),
            "TABLE_PREFIX": str(project / "outputs/tables/final"),
            "INTERIM_TABLE_JSON": str(interim),
            "EXPECTED_EVAL_CODE_ROOT": str(code_root),
            "BATCH44_POSTFINAL_TEST_EN_EXPECTED": str(EN_N),
            "BATCH44_POSTFINAL_TEST_ZH_EXPECTED": str(ZH_N),
            "BATCH44_POSTFINAL_TEST_EN_MANIFEST_SHA256": EN_HASH,
            "BATCH44_POSTFINAL_TEST_ZH_MANIFEST_SHA256": ZH_HASH,
            "FAKE_WRAPPER_CALLS": str(calls),
            "FAKE_QZ_CALLS": str(qz_calls),
            "FAKE_QZ_STATUSES": str(status_path),
            "PYTHON": sys.executable,
            "POLL_SECONDS": "1",
        }
    )
    if with_final:
        make_materialization(env, final)
    return project, env, final


def run_watcher(env: dict[str, str], *, action: str = "submit") -> subprocess.CompletedProcess[str]:
    call_env = env.copy()
    if action == "submit":
        call_env.update(
            {
                "ALLOW_LIVE_SUBMIT": "1",
                "CONFIRM_BATCH44_POSTFINAL_ORCHESTRATOR": "1",
            }
        )
    else:
        call_env.update(
            {
                "ALLOW_LIVE_SUBMIT": "0",
                "CONFIRM_BATCH44_POSTFINAL_ORCHESTRATOR": "0",
            }
        )
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--mode", "once", "--action", action],
        text=True,
        capture_output=True,
        env=call_env,
        check=False,
    )


def set_statuses(env: dict[str, str], **statuses: str) -> None:
    mapping = {
        {"smoke": SMOKE_JOB, "full": FULL_JOB, "score": SCORE_JOB}[key]: value
        for key, value in statuses.items()
    }
    write_json(Path(env["FAKE_QZ_STATUSES"]), mapping)


def make_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 1600)


def make_smoke(env: dict[str, str], final: dict) -> dict:
    root = Path(env["SMOKE_OUTPUT_ROOT"])
    wav = root / "en/audio/case.wav"
    make_wav(wav)
    contract = {
        "system_id": "path_x_final",
        "strict_manifest_sha256": {"en": EN_HASH, "zh": ZH_HASH},
        "inference": {"mode": "no_text"},
        "asset_hashes": {},
    }
    fingerprint = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload = {
        "schema_version": "moss_codecvc.batch42_pathx_strict_smoke_completion.v1",
        "status": "smoke_complete",
        "system_id": "path_x_final",
        "completed_utc": "2026-07-12T00:00:00+00:00",
        "resource_contract": {
            "compute_group": "MTTS-3-2-0715",
            "gpu_type": GPU,
            "gpus": 8,
            "instances": 1,
        },
        "registered_identity": registered_identity(final, Path(env["EXPECTED_EVAL_CODE_ROOT"])),
        "strict_inputs": {
            "schema_version": "moss_codecvc.batch42_pathx_strict_input_audit.v1",
            "en": {"rows": EN_N, "sha256": EN_HASH},
            "zh": {"rows": ZH_N, "sha256": ZH_HASH},
        },
        "protocol_contract": contract,
        "protocol_fingerprint_sha256": fingerprint,
        "actual_one_case": {
            "case_id": "case",
            "generated_audio": str(wav.resolve()),
            "inference_config": {
                "model_path": final["candidate"]["checkpoint_path"],
                "code_root": env["EXPECTED_EVAL_CODE_ROOT"],
                "final_selection_sha256": sha(Path(env["FINAL_SELECTION_JSON"])),
                "ref_audio_cfg_scale": 1.0,
            },
        },
    }
    write_json(root / "SMOKE_COMPLETED.json", payload)
    return payload


def write_inference_rows(path: Path, n: int, final: dict, env: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(n):
        rows.append(
            {
                "case_id": f"case-{index}",
                "system_id": "path_x_final",
                "status": "ok",
                "provenance": {
                    "inference_config": {
                        "model_path": final["candidate"]["checkpoint_path"],
                        "code_root": env["EXPECTED_EVAL_CODE_ROOT"],
                        "final_selection_sha256": sha(Path(env["FINAL_SELECTION_JSON"])),
                        "ref_audio_cfg_scale": 1.0,
                    }
                },
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_full(env: dict[str, str], final: dict, smoke: dict) -> dict:
    root = Path(env["INFERENCE_ROOT"])
    record = Path(env["FULL_RECORD_ROOT"])
    sets = {}
    for language, n in (("en", EN_N), ("zh", ZH_N)):
        successful = root / language / "successful.jsonl"
        schema = root / language / "schema" / f"path_x_final_{language}_strict.unified_eval.jsonl"
        merge = root / language / "merge_summary.json"
        write_inference_rows(successful, n, final, env)
        schema.parent.mkdir(parents=True, exist_ok=True)
        schema.write_text("".join("{}\n" for _ in range(n)), encoding="utf-8")
        write_json(merge, {"all_ok": True, "rows": n})
        sets[language] = {
            "registered_cases": n,
            "successful_jsonl": str(successful.resolve()),
            "schema_jsonl": str(schema.resolve()),
            "merge_summary": str(merge.resolve()),
        }
    payload = {
        "schema_version": "moss_codecvc.batch42_pathx_strict_completion.v1",
        "status": "complete",
        "system_id": "path_x_final",
        "completed_utc": "2026-07-12T00:01:00+00:00",
        "resource_contract": {
            "compute_group": "MTTS-3-2-0715",
            "gpu_type": GPU,
            "gpus": 8,
            "instances": 1,
        },
        "registered_identity": registered_identity(final, Path(env["EXPECTED_EVAL_CODE_ROOT"])),
        "smoke_gate": {
            "marker": str((Path(env["SMOKE_OUTPUT_ROOT"]) / "SMOKE_COMPLETED.json").resolve()),
            "protocol_fingerprint_sha256": smoke["protocol_fingerprint_sha256"],
        },
        "strict_sets": sets,
        "output_root": str(root.resolve()),
        "record_root": str(record.resolve()),
    }
    marker = root / "COMPLETED.json"
    write_json(marker, payload)
    record.mkdir(parents=True, exist_ok=True)
    (record / "completion.json").write_bytes(marker.read_bytes())
    return payload


def metric_group(n: int, primary_asr: str) -> dict:
    speakers = {}
    for backend in ("wavlm_large_sv", "eres2net", "speechbrain_ecapa"):
        speakers[backend] = {
            "status_counts": {"ok": n},
            "sim_ref": {"n": n, "mean": 0.5},
            "sim_src": {"n": n, "mean": 0.4},
        }
    return {
        "n_cases": n,
        "speaker_similarity": speakers,
        "content_asr": {
            primary_asr: {
                "status_counts": {"ok": n},
                "primary_error": {"n": n, "mean": 0.05},
            }
        },
    }


def make_scorer(env: dict[str, str]) -> None:
    root = Path(env["SCORER_OUTPUT"])
    combined = {
        "schema_version": "moss_codecvc.batch42_system_unified_summary.v1",
        "system_id": "path_x_final",
    }
    for language, n, asr, test_set in (
        ("en", EN_N, "whisper_large_v3", "seedtts-vc-en-internal320-disjoint"),
        ("zh", ZH_N, "paraformer_zh", "seedtts-vc-zh-internal320-disjoint"),
    ):
        merged_root = root / language / "merged"
        merged = merged_root / f"path_x_final.{language}.merged.unified_eval.jsonl"
        merged.parent.mkdir(parents=True, exist_ok=True)
        merged.write_text("".join("{}\n" for _ in range(n)), encoding="utf-8")
        group = metric_group(n, asr)
        summary = {
            "schema_version": "moss_codecvc.unified_vc_eval.v1",
            "record_type": "vc_eval_summary",
            "groups": {"all": group},
            "per_case_jsonl": str(merged.resolve()),
        }
        summary_path = merged_root / f"path_x_final.{language}.merged.summary.json"
        write_json(summary_path, summary)
        audit = {
            "schema_version": "moss_codecvc.batch42_strict_scorer_audit.v1",
            "system_id": "path_x_final",
            "test_set_id": test_set,
            "language": language,
            "rows": n,
            "unique_case_ids": n,
            "speaker_status_counts": {
                backend: {"ok": n}
                for backend in ("wavlm_large_sv", "eres2net", "speechbrain_ecapa")
            },
            "asr_status_counts": {asr: {"ok": n}},
            "all_ok": True,
            "merged_jsonl": str(merged.resolve()),
        }
        audit_path = merged_root / f"path_x_final.{language}.merged.strict_audit.json"
        write_json(audit_path, audit)
        combined[language] = {
            "summary_path": str(summary_path.resolve()),
            "strict_audit_path": str(audit_path.resolve()),
            "strict_audit": audit,
            "group_all": group,
        }
    write_json(root / "path_x_final.en_zh.summary.json", combined)
    write_json(
        root / "completion.json",
        {
            "schema_version": "moss_codecvc.batch42_unified_scorer_completion.v2",
            "system_id": "path_x_final",
            "status": "complete",
            "completed_at_utc": "2026-07-12T00:02:00Z",
            "en_cases": EN_N,
            "zh_cases": ZH_N,
            "output_root": str(root.resolve()),
            "job_id": SCORE_JOB,
            "resource_contract": {
                "workspace_id": WORKSPACE,
                "project_id": PROJECT_ID,
                "compute_group_id": COMPUTE_ID,
                "compute_group_name": "MTTS-3-2-0715",
                "spec_id": SPEC,
                "gpu_type": GPU,
                "instances": 1,
                "gpus": 8,
                "shards": 8,
            },
        },
    )


def test_source_has_one_way_fail_closed_contract() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "validate_final_selection_provenance" in text
    assert "CONFIRM_BATCH44_POSTFINAL_ORCHESTRATOR" in text
    assert "ALLOW_LIVE_SUBMIT" in text
    assert '"CONFIRM_BATCH44_FINAL_STRICT": "1"' in text
    assert '"CONFIRM_BATCH44_FINAL_SCORERS": "1"' in text
    assert "job_succeeded" in text
    assert "HALTED.json" in text
    assert "no automatic retry" in text.lower()
    assert "MTTS-3-2-0715" in text
    assert SPEC in text
    assert "NVIDIA_H200_SXM_141G" in text
    assert "--winner" not in text
    assert "004107_finalize_batch43_pathx_final.py\", \"--" not in text


def test_missing_final_is_pending_and_never_touches_qz(tmp_path: Path) -> None:
    project, env, _ = setup_project(tmp_path, with_final=False)
    result = run_watcher(env, action="plan")
    assert result.returncode == 3, result.stderr + result.stdout
    state = json.loads((project / "state/scan_latest.json").read_text())
    assert state["state"] == "WAITING_FINAL_SELECTION"
    assert Path(env["FAKE_QZ_CALLS"]).read_text() == ""
    assert Path(env["FAKE_WRAPPER_CALLS"]).read_text() == ""
    assert not (project / "state/HALTED.json").exists()


def test_submit_requires_all_watcher_live_gates(tmp_path: Path) -> None:
    _, env, _ = setup_project(tmp_path)
    env["ALLOW_LIVE_SUBMIT"] = "1"
    env["CONFIRM_BATCH44_POSTFINAL_ORCHESTRATOR"] = "0"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--mode", "once", "--action", "submit"],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    assert result.returncode == 2
    assert "requires ALLOW_LIVE_SUBMIT=1" in result.stderr
    assert Path(env["FAKE_WRAPPER_CALLS"]).read_text() == ""


def test_complete_serial_pipeline_submits_once_per_stage_and_publishes_8_of_8(
    tmp_path: Path,
) -> None:
    project, env, final = setup_project(tmp_path)

    first = run_watcher(env)
    assert first.returncode == 3, first.stderr + first.stdout
    assert Path(env["FAKE_WRAPPER_CALLS"]).read_text().splitlines() == ["smoke"]
    assert (Path(env["SMOKE_RECORD_ROOT"]) / "submitted_jobs.tsv").is_file()
    assert not (Path(env["FULL_RECORD_ROOT"]) / "submitted_jobs.tsv").exists()

    set_statuses(env, smoke="job_running")
    running = run_watcher(env)
    assert running.returncode == 3, running.stderr + running.stdout
    assert "WAITING_SMOKE_QZ" in running.stdout
    assert Path(env["FAKE_WRAPPER_CALLS"]).read_text().splitlines() == ["smoke"]

    smoke = make_smoke(env, final)
    set_statuses(env, smoke="job_succeeded")
    after_smoke = run_watcher(env)
    assert after_smoke.returncode == 3, after_smoke.stderr + after_smoke.stdout
    assert Path(env["FAKE_WRAPPER_CALLS"]).read_text().splitlines() == ["smoke", "full"]

    make_full(env, final, smoke)
    set_statuses(env, smoke="job_succeeded", full="job_succeeded")
    after_full = run_watcher(env)
    assert after_full.returncode == 3, after_full.stderr + after_full.stdout
    assert Path(env["FAKE_WRAPPER_CALLS"]).read_text().splitlines() == [
        "smoke",
        "full",
        "score",
    ]

    make_scorer(env)
    set_statuses(
        env,
        smoke="job_succeeded",
        full="job_succeeded",
        score="job_succeeded",
    )
    completed = run_watcher(env)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert Path(env["FAKE_WRAPPER_CALLS"]).read_text().splitlines() == [
        "smoke",
        "full",
        "score",
        "table",
    ]
    marker = json.loads((project / "state/PIPELINE_COMPLETE.json").read_text())
    assert marker["status"] == "complete"
    assert marker["candidate_id"] == "r3_step-26000"
    assert marker["jobs"] == {"smoke": SMOKE_JOB, "full": FULL_JOB, "score": SCORE_JOB}
    table = json.loads((project / "outputs/tables/final.json").read_text())
    assert table["counts"] == {"systems": 8, "complete": 8, "partial": 0, "pending": 0}
    assert not (project / "state/HALTED.json").exists()

    # Idempotent verification must not submit or publish anything again.
    again = run_watcher(env)
    assert again.returncode == 0, again.stderr + again.stdout
    assert Path(env["FAKE_WRAPPER_CALLS"]).read_text().splitlines() == [
        "smoke",
        "full",
        "score",
        "table",
    ]


def test_failed_qz_job_halts_and_never_retries(tmp_path: Path) -> None:
    project, env, _ = setup_project(tmp_path)
    submitted = run_watcher(env)
    assert submitted.returncode == 3
    set_statuses(env, smoke="job_failed")
    failed = run_watcher(env)
    assert failed.returncode == 2
    halt = json.loads((project / "state/HALTED.json").read_text())
    assert halt["status"] == "halted"
    assert halt["automatic_retry"] is False
    assert "job_failed" in halt["reason"]
    calls = Path(env["FAKE_WRAPPER_CALLS"]).read_text()
    repeated = run_watcher(env)
    assert repeated.returncode == 2
    assert Path(env["FAKE_WRAPPER_CALLS"]).read_text() == calls
    assert not (Path(env["FULL_RECORD_ROOT"]) / "submitted_jobs.tsv").exists()


def test_qz_success_without_atomic_marker_halts(tmp_path: Path) -> None:
    project, env, _ = setup_project(tmp_path)
    assert run_watcher(env).returncode == 3
    set_statuses(env, smoke="job_succeeded")
    result = run_watcher(env)
    assert result.returncode == 2
    halt = json.loads((project / "state/HALTED.json").read_text())
    assert "atomic completion artifact is missing" in halt["reason"]
    assert not (Path(env["FULL_RECORD_ROOT"]) / "submitted_jobs.tsv").exists()


def test_out_of_order_downstream_ledger_halts_before_any_submission(tmp_path: Path) -> None:
    project, env, _ = setup_project(tmp_path)
    full = Path(env["FULL_RECORD_ROOT"])
    full.mkdir(parents=True)
    (full / "submitted_jobs.tsv").write_text("forged\n", encoding="utf-8")
    result = run_watcher(env)
    assert result.returncode == 2
    halt = json.loads((project / "state/HALTED.json").read_text())
    assert "downstream artifacts exist before smoke submission" in halt["reason"]
    assert Path(env["FAKE_WRAPPER_CALLS"]).read_text() == ""


def test_plan_with_existing_ledger_never_queries_qz(tmp_path: Path) -> None:
    _, env, _ = setup_project(tmp_path)
    assert run_watcher(env).returncode == 3
    Path(env["FAKE_QZ_CALLS"]).write_text("")
    planned = run_watcher(env, action="plan")
    assert planned.returncode == 3, planned.stderr + planned.stdout
    assert "SMOKE_NEEDS_QZ_AUDIT" in planned.stdout
    assert Path(env["FAKE_QZ_CALLS"]).read_text() == ""
    assert Path(env["FAKE_WRAPPER_CALLS"]).read_text().splitlines() == ["smoke"]


def test_qz_resource_drift_halts_before_full(tmp_path: Path) -> None:
    project, env, final = setup_project(tmp_path)
    assert run_watcher(env).returncode == 3
    make_smoke(env, final)
    set_statuses(env, smoke="job_succeeded")
    qz = Path(env["QZCLI"])
    qz.write_text(qz.read_text().replace('"gpu_count": 8', '"gpu_count": 4'), encoding="utf-8")
    result = run_watcher(env)
    assert result.returncode == 2
    halt = json.loads((project / "state/HALTED.json").read_text())
    assert "one 8xH200" in halt["reason"]
    assert not (Path(env["FULL_RECORD_ROOT"]) / "submitted_jobs.tsv").exists()

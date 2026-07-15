from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts/batch42_scorer_provenance.py"
SPEC = importlib.util.spec_from_file_location(
    "batch42_scorer_provenance_under_test", HELPER_PATH
)
assert SPEC is not None and SPEC.loader is not None
provenance = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = provenance
SPEC.loader.exec_module(provenance)


JOB_ID = "job-11111111-2222-3333-4444-555555555555"


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_json(path: Path, payload: object) -> Path:
    return _write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_ledger(
    path: Path,
    *,
    input_path: Path,
    submission_path: Path,
) -> None:
    input_payload = json.loads(input_path.read_text(encoding="utf-8"))
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    resource = submission["resource_contract"]
    inputs = input_payload["inputs"]
    upstream = input_payload["upstream"]
    inference = upstream["strict_inference_completion"]
    selection = upstream["final_selection"]
    submit = submission["submit_output"]
    row = {
        "job_name": submission["job_name"],
        "job_id": submission["job_id"],
        "system_tag": submission["system_id"],
        "compute_group": resource["compute_group_id"],
        "compute_group_name": resource["compute_group_name"],
        "spec": resource["spec_id"],
        "instances": resource["instances"],
        "gpu_type": resource["gpu_type"],
        "gpus": resource["gpus"],
        "en_input": inputs["en"]["path"],
        "en_input_sha256": inputs["en"]["sha256"],
        "zh_input": inputs["zh"]["path"],
        "zh_input_sha256": inputs["zh"]["sha256"],
        "source_inference_completion": inference["path"],
        "source_inference_completion_sha256": inference["sha256"],
        "source_final_selection": selection["path"],
        "source_final_selection_sha256": selection["sha256"],
        "output_root": submission["output_root"],
        "snapshot_root": submission["snapshot_root"],
        "input_provenance": str(input_path.resolve()),
        "input_provenance_sha256": provenance.sha256_file(input_path),
        "submission_contract": str(submission_path.resolve()),
        "submission_contract_sha256": provenance.sha256_file(submission_path),
        "submit_output": submit["path"],
        "submit_output_sha256": submit["sha256"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=provenance.LEDGER_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(row)


def _fixture(root: Path) -> dict[str, Path]:
    record = root / "record"
    snapshot = record / "record_snapshot"
    output = root / "scored"
    en_input = _write(root / "inference/en/successful.jsonl", '{"case_id":"en"}\n')
    zh_input = _write(root / "inference/zh/successful.jsonl", '{"case_id":"zh"}\n')
    inference_completion = _write_json(
        root / "inference/COMPLETED.json", {"status": "complete"}
    )
    final_selection = _write_json(
        root / "selection/FINAL_SELECTION.json", {"status": "final"}
    )
    snapshot_manifest = _write(snapshot / "snapshot.sha256", "fixture snapshot\n")
    submit_output = _write(record / "submit_output.txt", f"created {JOB_ID}\n")
    input_path = record / "input_provenance.json"
    submission_path = record / "submission_contract.json"
    provenance.write_input_provenance(
        output=input_path,
        system_id="path_x_final",
        input_system_id="path_x_final",
        en_input=en_input,
        zh_input=zh_input,
        en_test_set_id="seedtts-vc-en-internal320-disjoint",
        zh_test_set_id="seedtts-vc-zh-internal320-disjoint",
        output_root=output,
        snapshot_manifest=snapshot_manifest,
        source_inference_completion=inference_completion,
        source_final_selection=final_selection,
    )
    provenance.write_submission_contract(
        output=submission_path,
        input_provenance=input_path,
        job_id=JOB_ID,
        job_name="batch42_score_path_x_final_fixture",
        system_id="path_x_final",
        output_root=output,
        record_root=record,
        snapshot_root=snapshot,
        submit_output=submit_output,
    )
    artifact_paths: dict[str, Path] = {}
    for language in ("en", "zh"):
        stem = output / language / "merged" / f"path_x_final.{language}.merged"
        artifact_paths[f"{language}_summary"] = _write_json(
            Path(f"{stem}.summary.json"), {"language": language}
        )
        artifact_paths[f"{language}_audit"] = _write_json(
            Path(f"{stem}.strict_audit.json"), {"all_ok": True}
        )
        artifact_paths[f"{language}_merged"] = _write(
            Path(f"{stem}.unified_eval.jsonl"), '{"status":"ok"}\n'
        )
    combined = _write_json(
        output / "path_x_final.en_zh.summary.json",
        {
            "schema_version": provenance.COMBINED_SCHEMA,
            "system_id": "path_x_final",
            "en": {},
            "zh": {},
        },
    )
    provenance.bind_combined_summary(
        combined_summary=combined,
        system_id="path_x_final",
        input_provenance=input_path,
        submission_contract=submission_path,
        en_summary=artifact_paths["en_summary"],
        en_audit=artifact_paths["en_audit"],
        en_merged_jsonl=artifact_paths["en_merged"],
        zh_summary=artifact_paths["zh_summary"],
        zh_audit=artifact_paths["zh_audit"],
        zh_merged_jsonl=artifact_paths["zh_merged"],
    )
    completion = output / "completion.json"
    provenance.write_completion(
        output=completion,
        system_id="path_x_final",
        output_root=output,
        input_provenance=input_path,
        submission_contract=submission_path,
        completed_at_utc="2026-07-13T00:00:00Z",
    )
    ledger = record / "submitted_jobs.tsv"
    _write_ledger(ledger, input_path=input_path, submission_path=submission_path)
    return {
        "record": record,
        "output": output,
        "en_input": en_input,
        "zh_input": zh_input,
        "inference_completion": inference_completion,
        "final_selection": final_selection,
        "input_provenance": input_path,
        "submission_contract": submission_path,
        "completion": completion,
        "combined": combined,
        "ledger": ledger,
        **artifact_paths,
    }


def _verify(paths: dict[str, Path]) -> dict[str, object]:
    return provenance.verify_final_bundle(
        completion_path=paths["completion"],
        ledger_path=paths["ledger"],
        expected_output_root=paths["output"],
        expected_en_input=paths["en_input"],
        expected_zh_input=paths["zh_input"],
        expected_inference_completion=paths["inference_completion"],
        expected_final_selection=paths["final_selection"],
    )


def test_complete_final_scorer_bundle_binds_all_required_inputs(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    result = _verify(paths)
    assert result["status"] == "verified"
    assert result["job_id"] == JOB_ID
    assert result["resource_contract"] == provenance.EXPECTED_RESOURCE_CONTRACT
    assert result["inputs"]["en"]["sha256"] == provenance.sha256_file(
        paths["en_input"]
    )
    assert result["inputs"]["zh"]["sha256"] == provenance.sha256_file(
        paths["zh_input"]
    )
    assert result["upstream"]["strict_inference_completion"][
        "sha256"
    ] == provenance.sha256_file(paths["inference_completion"])
    assert result["upstream"]["final_selection"][
        "sha256"
    ] == provenance.sha256_file(paths["final_selection"])


@pytest.mark.parametrize(
    ("key", "replacement", "match"),
    [
        ("en_input", "replaced EN input\n", "EN successful.jsonl SHA256 drift"),
        ("zh_input", "replaced ZH input\n", "ZH successful.jsonl SHA256 drift"),
        (
            "inference_completion",
            '{"status":"replaced"}\n',
            "strict inference COMPLETED SHA256 drift",
        ),
        (
            "final_selection",
            '{"status":"replaced"}\n',
            "FINAL_SELECTION SHA256 drift",
        ),
        ("en_summary", '{"metric":"forged"}\n', "scorer en/summary SHA256 drift"),
        ("combined", '{"system_id":"forged"}\n', "combined scorer summary SHA256 drift"),
    ],
)
def test_bundle_rejects_post_score_artifact_or_input_replacement(
    tmp_path: Path, key: str, replacement: str, match: str
) -> None:
    paths = _fixture(tmp_path)
    paths[key].write_text(replacement, encoding="utf-8")
    with pytest.raises(provenance.ProvenanceError, match=match):
        _verify(paths)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("job_id", "job-00000000-0000-0000-0000-000000000000"),
        ("compute_group_name", "H200-3-2"),
        ("spec", "wrong-spec"),
        ("gpu_type", "NVIDIA_A100"),
        ("gpus", "1"),
        ("output_root", "/tmp/forged-output"),
        ("en_input_sha256", "0" * 64),
        ("source_final_selection_sha256", "0" * 64),
    ],
)
def test_bundle_rejects_forged_qz_or_hash_ledger_fields(
    tmp_path: Path, field: str, replacement: str
) -> None:
    paths = _fixture(tmp_path)
    with paths["ledger"].open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    rows[0][field] = replacement
    with paths["ledger"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=provenance.LEDGER_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(provenance.ProvenanceError, match=f"ledger drift: {field}"):
        _verify(paths)


def test_bundle_rejects_legacy_minimal_completion(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _write_json(
        paths["completion"],
        {
            "system_id": "path_x_final",
            "status": "complete",
            "en_cases": 567,
            "zh_cases": 1194,
        },
    )
    with pytest.raises(provenance.ProvenanceError, match="schema_version"):
        _verify(paths)


def test_publish_script_hard_locks_interim_and_stages_before_promotion() -> None:
    publish = (ROOT / "scripts/004109_score_and_publish_batch42_pathx_final.sh").read_text(
        encoding="utf-8"
    )
    interim = ROOT / "testset/outputs/batch42_baseline_tables_20260711/batch42_baseline_interim.json"
    assert provenance.sha256_file(interim) in publish
    assert "verify_final_bundle(" in publish
    assert "if new != old:" in publish
    assert "main-table row drift" in publish
    assert "cross-validation row drift" in publish
    assert ".candidate-{os.getpid()}" in publish
    assert "os.replace(Path(outputs[key]), final_path)" in publish
    assert "batch42_pathx_final_table_publication.v1" in publish


def test_scorer_job_writes_runtime_bound_v2_completion() -> None:
    scorer = (ROOT / "scripts/004091_submit_batch42_unified_scorers_qz.sh").read_text(
        encoding="utf-8"
    )
    assert "wait_for_submission_contract" in scorer
    assert "audit_gpu_inventory" in scorer
    assert "bind-combined" in scorer
    assert "write-completion" in scorer
    assert "source_inference_completion_sha256" in scorer
    assert "source_final_selection_sha256" in scorer
    assert "submit_output_sha256" in scorer
    assert "compute_group_name" in scorer
    assert 'ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"' in scorer
    assert 'ALLOWED_GPU_TYPE="NVIDIA_H200_SXM_141G"' in scorer

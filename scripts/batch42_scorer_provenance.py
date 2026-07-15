#!/usr/bin/env python3
"""Create and verify fail-closed provenance for Batch-42 scorer jobs.

The generic scorer wrapper runs in two trust domains: the local submitter knows
the QZ job id, while the frozen in-job entrypoint can prove that the scoring
actually ran with the registered inputs and runtime GPU audit.  This helper
joins those domains with immutable SHA256-bound contracts and is intentionally
usable from both shell scripts and focused unit tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


INPUT_SCHEMA = "moss_codecvc.batch42_scorer_input_provenance.v1"
SUBMISSION_SCHEMA = "moss_codecvc.batch42_scorer_submission_contract.v1"
COMPLETION_SCHEMA = "moss_codecvc.batch42_unified_scorer_completion.v2"
COMBINED_SCHEMA = "moss_codecvc.batch42_system_unified_summary.v1"
JOB_ID_RE = re.compile(
    r"job-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

EXPECTED_RESOURCE_CONTRACT = {
    "workspace_id": "ws-8207e9e2-e733-4eec-a475-cfa1c36480ba",
    "project_id": "project-c67c548f-f02c-453b-ba5b-8745db6886e7",
    "compute_group_id": "lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122",
    "compute_group_name": "MTTS-3-2-0715",
    "spec_id": "67b10bc6-78b0-41a3-aaf4-358eeeb99009",
    "gpu_type": "NVIDIA_H200_SXM_141G",
    "instances": 1,
    "gpus": 8,
    "shards": 8,
}

LEDGER_FIELDS = [
    "job_name",
    "job_id",
    "system_tag",
    "compute_group",
    "compute_group_name",
    "spec",
    "instances",
    "gpu_type",
    "gpus",
    "en_input",
    "en_input_sha256",
    "zh_input",
    "zh_input_sha256",
    "source_inference_completion",
    "source_inference_completion_sha256",
    "source_final_selection",
    "source_final_selection_sha256",
    "output_root",
    "snapshot_root",
    "input_provenance",
    "input_provenance_sha256",
    "submission_contract",
    "submission_contract_sha256",
    "submit_output",
    "submit_output_sha256",
]


class ProvenanceError(ValueError):
    """A scorer provenance contract is missing, inconsistent, or stale."""


def sha256_file(path: Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ProvenanceError(f"missing provenance file: {resolved}")
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _load_json(path: Path) -> Any:
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProvenanceError(f"missing JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProvenanceError(f"invalid JSON {path}: {exc}") from exc


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require_ref(
    ref: Any,
    *,
    label: str,
    expected_path: Path | None = None,
) -> Path:
    if not isinstance(ref, Mapping):
        raise ProvenanceError(f"{label} must be a file reference")
    path = Path(str(ref.get("path") or "")).expanduser().resolve()
    if expected_path is not None and path != Path(expected_path).expanduser().resolve():
        raise ProvenanceError(f"{label} path drift: {path} != {expected_path}")
    if not path.is_file():
        raise ProvenanceError(f"{label} is missing: {path}")
    actual_sha = sha256_file(path)
    if ref.get("sha256") != actual_sha:
        raise ProvenanceError(f"{label} SHA256 drift")
    if ref.get("size") != path.stat().st_size:
        raise ProvenanceError(f"{label} size drift")
    return path


def _optional_ref(path: Path | None) -> dict[str, Any] | None:
    return None if path is None else file_ref(path)


def _assert_resource(resource: Any, *, label: str) -> None:
    if resource != EXPECTED_RESOURCE_CONTRACT:
        raise ProvenanceError(
            f"{label} is not the registered MTTS-3-2-0715 one-node 8xH200 contract"
        )


def write_input_provenance(
    *,
    output: Path,
    system_id: str,
    input_system_id: str,
    en_input: Path,
    zh_input: Path,
    en_test_set_id: str,
    zh_test_set_id: str,
    output_root: Path,
    snapshot_manifest: Path,
    source_inference_completion: Path | None,
    source_final_selection: Path | None,
) -> dict[str, Any]:
    if system_id == "path_x_final" and (
        source_inference_completion is None or source_final_selection is None
    ):
        raise ProvenanceError(
            "path_x_final requires strict inference COMPLETED and FINAL_SELECTION bindings"
        )
    payload = {
        "schema_version": INPUT_SCHEMA,
        "system_id": system_id,
        "input_system_id": input_system_id,
        "inputs": {
            "en": {
                **file_ref(en_input),
                "expected_cases": 567,
                "test_set_id": en_test_set_id,
            },
            "zh": {
                **file_ref(zh_input),
                "expected_cases": 1194,
                "test_set_id": zh_test_set_id,
            },
        },
        "upstream": {
            "strict_inference_completion": _optional_ref(
                source_inference_completion
            ),
            "final_selection": _optional_ref(source_final_selection),
        },
        "output_root": str(Path(output_root).expanduser().resolve()),
        "resource_contract": dict(EXPECTED_RESOURCE_CONTRACT),
        "snapshot_manifest": file_ref(snapshot_manifest),
    }
    atomic_write_json(output, payload)
    return payload


def validate_input_provenance(
    path: Path,
    *,
    expected_system_id: str | None = None,
    expected_en_input: Path | None = None,
    expected_zh_input: Path | None = None,
    expected_output_root: Path | None = None,
    expected_inference_completion: Path | None = None,
    expected_final_selection: Path | None = None,
) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != INPUT_SCHEMA:
        raise ProvenanceError("invalid scorer input-provenance schema")
    system_id = str(payload.get("system_id") or "")
    if expected_system_id is not None and system_id != expected_system_id:
        raise ProvenanceError("scorer input-provenance system drift")
    _assert_resource(payload.get("resource_contract"), label="input resource contract")
    if expected_output_root is not None and Path(
        str(payload.get("output_root") or "")
    ).resolve() != Path(expected_output_root).resolve():
        raise ProvenanceError("scorer input-provenance output_root drift")
    inputs = payload.get("inputs") or {}
    en = inputs.get("en") or {}
    zh = inputs.get("zh") or {}
    _require_ref(en, label="EN successful.jsonl", expected_path=expected_en_input)
    _require_ref(zh, label="ZH successful.jsonl", expected_path=expected_zh_input)
    if en.get("expected_cases") != 567 or zh.get("expected_cases") != 1194:
        raise ProvenanceError("scorer input denominators drift")
    _require_ref(payload.get("snapshot_manifest"), label="snapshot manifest")
    upstream = payload.get("upstream") or {}
    inference_ref = upstream.get("strict_inference_completion")
    selection_ref = upstream.get("final_selection")
    if system_id == "path_x_final" and (
        not isinstance(inference_ref, Mapping) or not isinstance(selection_ref, Mapping)
    ):
        raise ProvenanceError("path_x_final upstream provenance is incomplete")
    if inference_ref is not None:
        _require_ref(
            inference_ref,
            label="strict inference COMPLETED",
            expected_path=expected_inference_completion,
        )
    elif expected_inference_completion is not None:
        raise ProvenanceError("strict inference COMPLETED binding is missing")
    if selection_ref is not None:
        _require_ref(
            selection_ref,
            label="FINAL_SELECTION",
            expected_path=expected_final_selection,
        )
    elif expected_final_selection is not None:
        raise ProvenanceError("FINAL_SELECTION binding is missing")
    return dict(payload)


def write_submission_contract(
    *,
    output: Path,
    input_provenance: Path,
    job_id: str,
    job_name: str,
    system_id: str,
    output_root: Path,
    record_root: Path,
    snapshot_root: Path,
    submit_output: Path,
) -> dict[str, Any]:
    if not JOB_ID_RE.fullmatch(job_id):
        raise ProvenanceError(f"invalid QZ job id: {job_id!r}")
    input_payload = validate_input_provenance(
        input_provenance,
        expected_system_id=system_id,
        expected_output_root=output_root,
    )
    submit_ref = file_ref(submit_output)
    if job_id not in Path(submit_output).read_text(encoding="utf-8", errors="replace"):
        raise ProvenanceError("QZ submit output does not contain the returned job id")
    payload = {
        "schema_version": SUBMISSION_SCHEMA,
        "job_id": job_id,
        "job_name": job_name,
        "system_id": system_id,
        "input_provenance": file_ref(input_provenance),
        "output_root": str(Path(output_root).expanduser().resolve()),
        "record_root": str(Path(record_root).expanduser().resolve()),
        "snapshot_root": str(Path(snapshot_root).expanduser().resolve()),
        "resource_contract": input_payload["resource_contract"],
        "submit_output": submit_ref,
    }
    atomic_write_json(output, payload)
    return payload


def validate_submission_contract(
    path: Path,
    *,
    expected_input_provenance: Path | None = None,
    expected_system_id: str | None = None,
    expected_output_root: Path | None = None,
    expected_record_root: Path | None = None,
    expected_snapshot_root: Path | None = None,
) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SUBMISSION_SCHEMA:
        raise ProvenanceError("invalid scorer submission-contract schema")
    job_id = str(payload.get("job_id") or "")
    if not JOB_ID_RE.fullmatch(job_id):
        raise ProvenanceError("scorer submission contract has invalid QZ job id")
    if expected_system_id is not None and payload.get("system_id") != expected_system_id:
        raise ProvenanceError("scorer submission system drift")
    for key, expected in (
        ("output_root", expected_output_root),
        ("record_root", expected_record_root),
        ("snapshot_root", expected_snapshot_root),
    ):
        if expected is not None and Path(str(payload.get(key) or "")).resolve() != Path(
            expected
        ).resolve():
            raise ProvenanceError(f"scorer submission {key} drift")
    _assert_resource(
        payload.get("resource_contract"), label="submission resource contract"
    )
    input_path = _require_ref(
        payload.get("input_provenance"),
        label="input-provenance contract",
        expected_path=expected_input_provenance,
    )
    validate_input_provenance(
        input_path,
        expected_system_id=str(payload.get("system_id") or ""),
        expected_output_root=Path(str(payload.get("output_root") or "")),
    )
    submit_path = _require_ref(payload.get("submit_output"), label="QZ submit output")
    if job_id not in submit_path.read_text(encoding="utf-8", errors="replace"):
        raise ProvenanceError("QZ submit output/job-id binding drift")
    return dict(payload)


def wait_for_submission_contract(
    path: Path,
    *,
    timeout_seconds: int,
    **expected: Any,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return validate_submission_contract(path, **expected)
        except (OSError, ProvenanceError) as exc:
            last_error = exc
            time.sleep(1)
    raise ProvenanceError(
        f"timed out waiting for valid submission contract {path}: {last_error}"
    )


def bind_combined_summary(
    *,
    combined_summary: Path,
    system_id: str,
    input_provenance: Path,
    submission_contract: Path,
    en_summary: Path,
    en_audit: Path,
    en_merged_jsonl: Path,
    zh_summary: Path,
    zh_audit: Path,
    zh_merged_jsonl: Path,
) -> dict[str, Any]:
    payload = _load_json(combined_summary)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != COMBINED_SCHEMA:
        raise ProvenanceError("invalid combined scorer summary schema")
    if payload.get("system_id") != system_id:
        raise ProvenanceError("combined scorer summary system drift")
    input_payload = validate_input_provenance(
        input_provenance, expected_system_id=system_id
    )
    submission = validate_submission_contract(
        submission_contract,
        expected_input_provenance=input_provenance,
        expected_system_id=system_id,
        expected_output_root=Path(str(input_payload["output_root"])),
    )
    mutable = dict(payload)
    mutable["provenance"] = {
        "input_provenance": file_ref(input_provenance),
        "submission_contract": file_ref(submission_contract),
        "job_id": submission["job_id"],
        "resource_contract": submission["resource_contract"],
        "inputs": input_payload["inputs"],
        "upstream": input_payload["upstream"],
        "artifacts": {
            "en": {
                "summary": file_ref(en_summary),
                "strict_audit": file_ref(en_audit),
                "merged_jsonl": file_ref(en_merged_jsonl),
            },
            "zh": {
                "summary": file_ref(zh_summary),
                "strict_audit": file_ref(zh_audit),
                "merged_jsonl": file_ref(zh_merged_jsonl),
            },
        },
    }
    atomic_write_json(combined_summary, mutable)
    return mutable


def _expected_artifact_paths(output_root: Path, system_id: str) -> dict[str, Any]:
    root = Path(output_root).resolve()
    result: dict[str, Any] = {}
    for language in ("en", "zh"):
        stem = root / language / "merged" / f"{system_id}.{language}.merged"
        result[language] = {
            "summary": Path(f"{stem}.summary.json"),
            "strict_audit": Path(f"{stem}.strict_audit.json"),
            "merged_jsonl": Path(f"{stem}.unified_eval.jsonl"),
        }
    result["combined_summary"] = root / f"{system_id}.en_zh.summary.json"
    return result


def write_completion(
    *,
    output: Path,
    system_id: str,
    output_root: Path,
    input_provenance: Path,
    submission_contract: Path,
    completed_at_utc: str | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    input_payload = validate_input_provenance(
        input_provenance,
        expected_system_id=system_id,
        expected_output_root=output_root,
    )
    submission = validate_submission_contract(
        submission_contract,
        expected_input_provenance=input_provenance,
        expected_system_id=system_id,
        expected_output_root=output_root,
    )
    expected = _expected_artifact_paths(output_root, system_id)
    combined_path = expected.pop("combined_summary")
    combined = _load_json(combined_path)
    if not isinstance(combined, Mapping) or combined.get("system_id") != system_id:
        raise ProvenanceError("combined summary identity drift before completion")
    bound = (combined.get("provenance") or {}).get("artifacts") or {}
    artifacts: dict[str, Any] = {}
    for language in ("en", "zh"):
        artifacts[language] = {}
        for kind, path in expected[language].items():
            expected_ref = file_ref(path)
            if (bound.get(language) or {}).get(kind) != expected_ref:
                raise ProvenanceError(
                    f"combined summary {language}/{kind} provenance drift"
                )
            artifacts[language][kind] = expected_ref
    artifacts["combined_summary"] = file_ref(combined_path)
    timestamp = completed_at_utc or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    payload = {
        "schema_version": COMPLETION_SCHEMA,
        "system_id": system_id,
        "status": "complete",
        "completed_at_utc": timestamp,
        "en_cases": 567,
        "zh_cases": 1194,
        "output_root": str(output_root),
        "job_id": submission["job_id"],
        "resource_contract": submission["resource_contract"],
        "input_provenance": file_ref(input_provenance),
        "submission_contract": file_ref(submission_contract),
        "inputs": input_payload["inputs"],
        "upstream": input_payload["upstream"],
        "artifacts": artifacts,
    }
    atomic_write_json(output, payload)
    return payload


def _validate_artifact_refs(
    completion: Mapping[str, Any], *, output_root: Path, system_id: str
) -> dict[str, Any]:
    expected = _expected_artifact_paths(output_root, system_id)
    artifacts = completion.get("artifacts") or {}
    for language in ("en", "zh"):
        for kind, expected_path in expected[language].items():
            _require_ref(
                (artifacts.get(language) or {}).get(kind),
                label=f"scorer {language}/{kind}",
                expected_path=expected_path,
            )
    combined_path = _require_ref(
        artifacts.get("combined_summary"),
        label="combined scorer summary",
        expected_path=expected["combined_summary"],
    )
    return _load_json(combined_path)


def verify_final_bundle(
    *,
    completion_path: Path,
    ledger_path: Path,
    expected_output_root: Path,
    expected_en_input: Path,
    expected_zh_input: Path,
    expected_inference_completion: Path,
    expected_final_selection: Path,
) -> dict[str, Any]:
    """Verify the entire path_x_final scorer bundle and return an attestation."""

    completion_path = Path(completion_path).resolve()
    output_root = Path(expected_output_root).resolve()
    ledger_path = Path(ledger_path).resolve()
    expected_record_root = ledger_path.parent
    expected_snapshot_root = expected_record_root / "record_snapshot"
    completion = _load_json(completion_path)
    if not isinstance(completion, Mapping):
        raise ProvenanceError("scorer completion must be an object")
    expected_header = {
        "schema_version": COMPLETION_SCHEMA,
        "system_id": "path_x_final",
        "status": "complete",
        "en_cases": 567,
        "zh_cases": 1194,
        "output_root": str(output_root),
    }
    for key, expected in expected_header.items():
        if completion.get(key) != expected:
            raise ProvenanceError(
                f"path_x_final scorer completion drift: {key}={completion.get(key)!r}"
            )
    if not str(completion.get("completed_at_utc") or ""):
        raise ProvenanceError("path_x_final scorer completion timestamp is missing")
    _assert_resource(
        completion.get("resource_contract"), label="completion resource contract"
    )
    job_id = str(completion.get("job_id") or "")
    if not JOB_ID_RE.fullmatch(job_id):
        raise ProvenanceError("completion has invalid QZ job id")
    input_path = _require_ref(
        completion.get("input_provenance"), label="completion input provenance"
    )
    input_payload = validate_input_provenance(
        input_path,
        expected_system_id="path_x_final",
        expected_en_input=expected_en_input,
        expected_zh_input=expected_zh_input,
        expected_output_root=output_root,
        expected_inference_completion=expected_inference_completion,
        expected_final_selection=expected_final_selection,
    )
    if completion.get("inputs") != input_payload.get("inputs"):
        raise ProvenanceError("completion input SHA bindings drift")
    if completion.get("upstream") != input_payload.get("upstream"):
        raise ProvenanceError("completion upstream SHA bindings drift")
    submission_path = _require_ref(
        completion.get("submission_contract"), label="completion submission contract"
    )
    submission = validate_submission_contract(
        submission_path,
        expected_input_provenance=input_path,
        expected_system_id="path_x_final",
        expected_output_root=output_root,
        expected_record_root=expected_record_root,
        expected_snapshot_root=expected_snapshot_root,
    )
    if submission.get("job_id") != job_id:
        raise ProvenanceError("completion/submission QZ job-id drift")
    combined = _validate_artifact_refs(
        completion, output_root=output_root, system_id="path_x_final"
    )
    if (
        not isinstance(combined, Mapping)
        or combined.get("schema_version") != COMBINED_SCHEMA
        or combined.get("system_id") != "path_x_final"
    ):
        raise ProvenanceError("combined summary identity/schema drift")
    combined_provenance = combined.get("provenance") or {}
    if combined_provenance.get("input_provenance") != completion.get(
        "input_provenance"
    ):
        raise ProvenanceError("combined summary input-provenance drift")
    if combined_provenance.get("submission_contract") != completion.get(
        "submission_contract"
    ):
        raise ProvenanceError("combined summary submission-contract drift")
    if combined_provenance.get("job_id") != job_id:
        raise ProvenanceError("combined summary QZ job-id drift")
    if combined_provenance.get("resource_contract") != EXPECTED_RESOURCE_CONTRACT:
        raise ProvenanceError("combined summary resource-contract drift")
    if combined_provenance.get("inputs") != input_payload.get("inputs"):
        raise ProvenanceError("combined summary input SHA bindings drift")
    if combined_provenance.get("upstream") != input_payload.get("upstream"):
        raise ProvenanceError("combined summary upstream SHA bindings drift")
    for language in ("en", "zh"):
        if (combined_provenance.get("artifacts") or {}).get(language) != (
            completion.get("artifacts") or {}
        ).get(language):
            raise ProvenanceError(f"combined summary {language} artifact bindings drift")

    try:
        with ledger_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != LEDGER_FIELDS:
                raise ProvenanceError("path_x_final scorer ledger schema drift")
            rows = list(reader)
    except FileNotFoundError as exc:
        raise ProvenanceError(f"missing scorer ledger: {ledger_path}") from exc
    if len(rows) != 1:
        raise ProvenanceError("path_x_final scorer must have exactly one QZ ledger row")
    row = rows[0]
    resource = EXPECTED_RESOURCE_CONTRACT
    expected_ledger = {
        "job_name": str(submission.get("job_name") or ""),
        "job_id": job_id,
        "system_tag": "path_x_final",
        "compute_group": resource["compute_group_id"],
        "compute_group_name": resource["compute_group_name"],
        "spec": resource["spec_id"],
        "instances": str(resource["instances"]),
        "gpu_type": resource["gpu_type"],
        "gpus": str(resource["gpus"]),
        "en_input": str(Path(expected_en_input).resolve()),
        "en_input_sha256": sha256_file(Path(expected_en_input)),
        "zh_input": str(Path(expected_zh_input).resolve()),
        "zh_input_sha256": sha256_file(Path(expected_zh_input)),
        "source_inference_completion": str(
            Path(expected_inference_completion).resolve()
        ),
        "source_inference_completion_sha256": sha256_file(
            Path(expected_inference_completion)
        ),
        "source_final_selection": str(Path(expected_final_selection).resolve()),
        "source_final_selection_sha256": sha256_file(
            Path(expected_final_selection)
        ),
        "output_root": str(output_root),
        "snapshot_root": str(Path(str(submission.get("snapshot_root") or "")).resolve()),
        "input_provenance": str(input_path),
        "input_provenance_sha256": sha256_file(input_path),
        "submission_contract": str(submission_path),
        "submission_contract_sha256": sha256_file(submission_path),
        "submit_output": str(
            Path(str((submission.get("submit_output") or {}).get("path") or "")).resolve()
        ),
        "submit_output_sha256": str(
            (submission.get("submit_output") or {}).get("sha256") or ""
        ),
    }
    for key, expected in expected_ledger.items():
        if row.get(key) != expected:
            raise ProvenanceError(f"path_x_final scorer ledger drift: {key}")
    return {
        "schema_version": "moss_codecvc.batch42_pathx_final_scorer_attestation.v1",
        "status": "verified",
        "job_id": job_id,
        "resource_contract": dict(EXPECTED_RESOURCE_CONTRACT),
        "completion": file_ref(completion_path),
        "combined_summary": completion["artifacts"]["combined_summary"],
        "ledger": file_ref(ledger_path),
        "input_provenance": completion["input_provenance"],
        "submission_contract": completion["submission_contract"],
        "inputs": completion["inputs"],
        "upstream": completion["upstream"],
        "output_root": str(output_root),
    }


def _optional_path(value: str) -> Path | None:
    return Path(value) if value else None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    item = sub.add_parser("write-input")
    item.add_argument("--output", required=True, type=Path)
    item.add_argument("--system-id", required=True)
    item.add_argument("--input-system-id", required=True)
    item.add_argument("--en-input", required=True, type=Path)
    item.add_argument("--zh-input", required=True, type=Path)
    item.add_argument("--en-test-set-id", required=True)
    item.add_argument("--zh-test-set-id", required=True)
    item.add_argument("--output-root", required=True, type=Path)
    item.add_argument("--snapshot-manifest", required=True, type=Path)
    item.add_argument("--source-inference-completion", default="")
    item.add_argument("--source-final-selection", default="")

    item = sub.add_parser("write-submission")
    item.add_argument("--output", required=True, type=Path)
    item.add_argument("--input-provenance", required=True, type=Path)
    item.add_argument("--job-id", required=True)
    item.add_argument("--job-name", required=True)
    item.add_argument("--system-id", required=True)
    item.add_argument("--output-root", required=True, type=Path)
    item.add_argument("--record-root", required=True, type=Path)
    item.add_argument("--snapshot-root", required=True, type=Path)
    item.add_argument("--submit-output", required=True, type=Path)

    item = sub.add_parser("wait-submission")
    item.add_argument("--contract", required=True, type=Path)
    item.add_argument("--input-provenance", required=True, type=Path)
    item.add_argument("--system-id", required=True)
    item.add_argument("--output-root", required=True, type=Path)
    item.add_argument("--record-root", required=True, type=Path)
    item.add_argument("--snapshot-root", required=True, type=Path)
    item.add_argument("--timeout-seconds", type=int, default=300)

    item = sub.add_parser("bind-combined")
    item.add_argument("--combined-summary", required=True, type=Path)
    item.add_argument("--system-id", required=True)
    item.add_argument("--input-provenance", required=True, type=Path)
    item.add_argument("--submission-contract", required=True, type=Path)
    for language in ("en", "zh"):
        item.add_argument(f"--{language}-summary", required=True, type=Path)
        item.add_argument(f"--{language}-audit", required=True, type=Path)
        item.add_argument(f"--{language}-merged-jsonl", required=True, type=Path)

    item = sub.add_parser("write-completion")
    item.add_argument("--output", required=True, type=Path)
    item.add_argument("--system-id", required=True)
    item.add_argument("--output-root", required=True, type=Path)
    item.add_argument("--input-provenance", required=True, type=Path)
    item.add_argument("--submission-contract", required=True, type=Path)
    item.add_argument("--completed-at-utc", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "write-input":
        payload = write_input_provenance(
            output=args.output,
            system_id=args.system_id,
            input_system_id=args.input_system_id,
            en_input=args.en_input,
            zh_input=args.zh_input,
            en_test_set_id=args.en_test_set_id,
            zh_test_set_id=args.zh_test_set_id,
            output_root=args.output_root,
            snapshot_manifest=args.snapshot_manifest,
            source_inference_completion=_optional_path(
                args.source_inference_completion
            ),
            source_final_selection=_optional_path(args.source_final_selection),
        )
    elif args.command == "write-submission":
        payload = write_submission_contract(
            output=args.output,
            input_provenance=args.input_provenance,
            job_id=args.job_id,
            job_name=args.job_name,
            system_id=args.system_id,
            output_root=args.output_root,
            record_root=args.record_root,
            snapshot_root=args.snapshot_root,
            submit_output=args.submit_output,
        )
    elif args.command == "wait-submission":
        payload = wait_for_submission_contract(
            args.contract,
            timeout_seconds=args.timeout_seconds,
            expected_input_provenance=args.input_provenance,
            expected_system_id=args.system_id,
            expected_output_root=args.output_root,
            expected_record_root=args.record_root,
            expected_snapshot_root=args.snapshot_root,
        )
    elif args.command == "bind-combined":
        payload = bind_combined_summary(
            combined_summary=args.combined_summary,
            system_id=args.system_id,
            input_provenance=args.input_provenance,
            submission_contract=args.submission_contract,
            en_summary=args.en_summary,
            en_audit=args.en_audit,
            en_merged_jsonl=args.en_merged_jsonl,
            zh_summary=args.zh_summary,
            zh_audit=args.zh_audit,
            zh_merged_jsonl=args.zh_merged_jsonl,
        )
    else:
        payload = write_completion(
            output=args.output,
            system_id=args.system_id,
            output_root=args.output_root,
            input_provenance=args.input_provenance,
            submission_contract=args.submission_contract,
            completed_at_utc=args.completed_at_utc or None,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

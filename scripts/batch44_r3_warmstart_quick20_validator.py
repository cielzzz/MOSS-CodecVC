#!/usr/bin/env python3
"""Strict validator for a completed Batch-44 r3 warm-start local quick20."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


def _load_helper():
    sibling = Path(__file__).with_name(
        "batch44_r3_warmstart_quick20_completion.frozen.py"
    )
    if not sibling.is_file():
        sibling = Path(__file__).with_name(
            "batch44_r3_warmstart_quick20_completion.py"
        )
    spec = importlib.util.spec_from_file_location(
        "batch44_r3_warmstart_quick20_completion_bound", sibling
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load bound completion helper: {sibling}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_HELPER = _load_helper()
BASE_EFFECTIVE_STEP = _HELPER.BASE_EFFECTIVE_STEP
CHECKPOINT_FILES = _HELPER.CHECKPOINT_FILES
COMPLETION_SCHEMA = _HELPER.COMPLETION_SCHEMA
MARKER_SCHEMA = _HELPER.MARKER_SCHEMA
artifact = _HELPER.artifact
checkpoint_manifest_sha256 = _HELPER.checkpoint_manifest_sha256
require_artifact_matches = _HELPER.require_artifact_matches
run_id = _HELPER.run_id
sha256_file = _HELPER.sha256_file
validate_step_mapping = _HELPER.validate_step_mapping


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _validate_metrics(
    *,
    completion: Mapping[str, Any],
    record_root: Path,
    output_root: Path,
    effective_step: int,
    local_step: int,
    checkpoint: Path,
    train_job_id: str,
    contract: Path,
) -> None:
    metric_specs = completion.get("metrics")
    if not isinstance(metric_specs, dict) or set(metric_specs) != {"json", "tsv", "md"}:
        raise ValueError("completion metric artifact set drift")
    metric_paths = {
        name: require_artifact_matches(
            metric_specs[name], record_root / f"metrics.{name}", f"metrics {name}"
        )
        for name in ("json", "tsv", "md")
    }
    payload = json.loads(metric_paths["json"].read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 2:
        raise ValueError("metrics.json must contain exactly two rows")
    with metric_paths["tsv"].open(encoding="utf-8", newline="") as handle:
        tsv = list(csv.DictReader(handle, delimiter="\t"))
    if len(tsv) != 2:
        raise ValueError("metrics.tsv must contain exactly two rows")
    contract_sha = sha256_file(contract)
    checkpoint_files = completion.get("checkpoint_files")
    if not isinstance(checkpoint_files, dict):
        raise ValueError("completion checkpoint_files must be an object")
    manifest_sha = checkpoint_manifest_sha256(checkpoint_files)
    expected_modes = {"no_text", "text"}
    by_mode: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("metrics rows must be objects")
        mode = str(row.get("mode") or "")
        if mode not in expected_modes or mode in by_mode:
            raise ValueError(f"invalid/duplicate metric mode: {mode!r}")
        identity = run_id(effective_step, mode)
        expected = {
            "step": effective_step,
            "effective_step": effective_step,
            "base_effective_step": BASE_EFFECTIVE_STEP,
            "continuation_local_step": local_step,
            "arm": "r3",
            "train_job_id": train_job_id,
            "mode": mode,
            "n": 20,
            "checkpoint": str(checkpoint),
            "checkpoint_manifest_sha256": manifest_sha,
            "warm_start_contract": str(contract),
            "warm_start_contract_sha256": contract_sha,
            "run_id": identity,
            "output_dir": str(output_root / identity),
        }
        drift = {
            key: {"expected": wanted, "actual": row.get(key)}
            for key, wanted in expected.items()
            if row.get(key) != wanted
        }
        if drift:
            raise ValueError(f"metric {mode} identity drift: {drift}")
        numbers: dict[str, float] = {}
        for field in (
            "fail", "cer", "sim_ref", "sim_src", "margin", "ref_bound",
            "ref_content_f1",
        ):
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"metric {mode}.{field} is not numeric") from exc
            if not math.isfinite(value):
                raise ValueError(f"metric {mode}.{field} is non-finite")
            numbers[field] = value
        keep = row.get("keep")
        if isinstance(keep, bool) or not isinstance(keep, int) or not 0 <= keep <= 20:
            raise ValueError(f"metric {mode}.keep invalid: {keep!r}")
        if not math.isclose(numbers["fail"], (20 - keep) / 20, abs_tol=1e-12):
            raise ValueError(f"metric {mode} fail/keep mismatch")
        if not math.isclose(
            numbers["margin"], numbers["sim_ref"] - numbers["sim_src"], abs_tol=1e-12
        ):
            raise ValueError(f"metric {mode} margin mismatch")
        by_mode[mode] = row
    if set(by_mode) != expected_modes:
        raise ValueError(f"metric mode set drift: {set(by_mode)}")
    for row in tsv:
        mode = str(row.get("mode") or "")
        if mode not in by_mode:
            raise ValueError(f"unexpected metrics.tsv mode: {mode!r}")
        for field in (
            "step", "effective_step", "base_effective_step", "continuation_local_step",
            "arm", "train_job_id", "mode", "n", "keep", "checkpoint",
            "checkpoint_manifest_sha256", "warm_start_contract",
            "warm_start_contract_sha256", "run_id", "output_dir",
        ):
            if str(row.get(field, "")) != str(by_mode[mode].get(field, "")):
                raise ValueError(f"metrics JSON/TSV disagree for {mode}.{field}")


def validate_completion(
    record_root: Path,
    *,
    expected_effective_step: int | None = None,
    expected_continuation_local_step: int | None = None,
    expected_train_job_id: str | None = None,
) -> dict[str, Any]:
    record_root = record_root.expanduser().resolve()
    completion_path = record_root / "COMPLETED.json"
    marker_path = record_root / "complete.marker"
    completion = _json_object(completion_path, "completion")
    marker = _json_object(marker_path, "completion marker")
    effective_step = int(completion.get("effective_step") or -1)
    local_step = int(completion.get("continuation_local_step") or -1)
    validate_step_mapping(effective_step, local_step)
    if expected_effective_step is not None and effective_step != expected_effective_step:
        raise ValueError(
            f"effective step={effective_step}, expected {expected_effective_step}"
        )
    if (
        expected_continuation_local_step is not None
        and local_step != expected_continuation_local_step
    ):
        raise ValueError(
            f"continuation local step={local_step}, "
            f"expected {expected_continuation_local_step}"
        )
    train_job_id = str(completion.get("train_job_id") or "")
    if expected_train_job_id is not None and train_job_id != expected_train_job_id:
        raise ValueError(f"train job={train_job_id!r}, expected {expected_train_job_id!r}")
    expected_top = {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "backend": "local",
        "step": effective_step,
        "effective_step": effective_step,
        "base_effective_step": BASE_EFFECTIVE_STEP,
        "continuation_local_step": local_step,
        "record_root": str(record_root),
    }
    top_drift = {
        key: {"expected": wanted, "actual": completion.get(key)}
        for key, wanted in expected_top.items()
        if completion.get(key) != wanted
    }
    if top_drift:
        raise ValueError(f"completion identity drift: {top_drift}")
    output_root = Path(str(completion.get("output_root") or "")).expanduser().resolve()
    checkpoint = Path(str(completion.get("checkpoint") or "")).expanduser().resolve()
    contract = Path(str(completion.get("warm_start_contract") or "")).expanduser().resolve()
    if checkpoint.name != f"step-{local_step}" or not checkpoint.is_dir():
        raise ValueError(f"completion checkpoint binding drift: {checkpoint}")
    if completion.get("warm_start_contract_sha256") != sha256_file(contract):
        raise ValueError("completion warm-start contract SHA drift")

    checkpoint_specs = completion.get("checkpoint_files")
    if not isinstance(checkpoint_specs, dict) or set(checkpoint_specs) != set(CHECKPOINT_FILES):
        raise ValueError("completion checkpoint artifact set drift")
    for name in CHECKPOINT_FILES:
        require_artifact_matches(
            checkpoint_specs[name], checkpoint / name, f"checkpoint {name}"
        )
    manifest_sha = checkpoint_manifest_sha256(checkpoint_specs)
    if completion.get("checkpoint_manifest_sha256") != manifest_sha:
        raise ValueError("completion checkpoint manifest SHA drift")

    completion_sha = sha256_file(completion_path)
    expected_marker = {
        "schema": MARKER_SCHEMA,
        "status": "complete",
        "backend": "local",
        "step": effective_step,
        "effective_step": effective_step,
        "base_effective_step": BASE_EFFECTIVE_STEP,
        "continuation_local_step": local_step,
        "completed_json_sha256": completion_sha,
    }
    marker_drift = {
        key: {"expected": wanted, "actual": marker.get(key)}
        for key, wanted in expected_marker.items()
        if marker.get(key) != wanted
    }
    if marker_drift:
        raise ValueError(f"completion marker drift: {marker_drift}")

    for key, label in (
        ("runner", "runner"),
        ("common_library", "common library"),
        ("completion_helper", "completion helper"),
        ("validator", "validator"),
    ):
        spec = completion.get(key)
        if not isinstance(spec, dict):
            raise ValueError(f"completion {label} artifact missing")
        require_artifact_matches(spec, Path(str(spec.get("path") or "")), label)
    execution = completion.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("completion execution object missing")
    runtime = execution.get("runtime_manifest")
    if not isinstance(runtime, dict):
        raise ValueError("completion runtime artifact missing")
    require_artifact_matches(runtime, Path(str(runtime.get("path") or "")), "runtime")

    fixed = completion.get("fixed_inputs")
    if not isinstance(fixed, dict) or set(fixed) != {"no_text20", "text_source", "text20"}:
        raise ValueError("completion fixed input set drift")
    for name, spec in fixed.items():
        if not isinstance(spec, dict):
            raise ValueError(f"fixed input {name} artifact invalid")
        require_artifact_matches(spec, Path(str(spec.get("path") or "")), f"fixed input {name}")

    training = completion.get("training_provenance")
    if not isinstance(training, dict):
        raise ValueError("completion training provenance missing")
    for name, spec in training.items():
        if not isinstance(spec, dict):
            raise ValueError(f"training provenance {name} artifact invalid")
        require_artifact_matches(spec, Path(str(spec.get("path") or "")), f"training {name}")

    _validate_metrics(
        completion=completion,
        record_root=record_root,
        output_root=output_root,
        effective_step=effective_step,
        local_step=local_step,
        checkpoint=checkpoint,
        train_job_id=train_job_id,
        contract=contract,
    )

    runs = completion.get("runs")
    if not isinstance(runs, list) or len(runs) != 2:
        raise ValueError("completion runs must contain exactly two rows")
    seen: set[str] = set()
    for row in runs:
        if not isinstance(row, dict):
            raise ValueError("completion run rows must be objects")
        mode = str(row.get("mode") or "")
        if mode not in {"no_text", "text"} or mode in seen:
            raise ValueError(f"invalid/duplicate completion run mode: {mode!r}")
        identity = run_id(effective_step, mode)
        output_dir = output_root / identity
        expected = {
            "arm": "r3",
            "mode": mode,
            "run_id": identity,
            "effective_step": effective_step,
            "continuation_local_step": local_step,
            "checkpoint": str(checkpoint),
            "train_job_id": train_job_id,
            "output_dir": str(output_dir),
        }
        drift = {
            key: {"expected": wanted, "actual": row.get(key)}
            for key, wanted in expected.items()
            if row.get(key) != wanted
        }
        if drift:
            raise ValueError(f"completion run {mode} drift: {drift}")
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != {
            "summary", "asr", "speaker", "ref_content"
        }:
            raise ValueError(f"completion run {mode} artifact set drift")
        expected_paths = {
            "summary": output_dir / f"{identity}.summary.json",
            "asr": output_dir / f"{identity}.asr_eval.jsonl",
            "speaker": output_dir / f"{identity}.speaker_sim.csv",
            "ref_content": output_dir / f"{identity}.ref_content_similarity_summary.json",
        }
        for name, path in expected_paths.items():
            require_artifact_matches(artifacts[name], path, f"run {mode} {name}")
        seen.add(mode)
    if seen != {"no_text", "text"}:
        raise ValueError(f"completion run mode set drift: {seen}")

    if os.path.lexists(record_root / "submitted_jobs.tsv"):
        raise ValueError("local quick20 record unexpectedly contains a QZ submission ledger")
    return completion


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--record-root", type=Path, required=True)
    result.add_argument("--expected-effective-step", type=int)
    result.add_argument("--expected-continuation-local-step", type=int)
    result.add_argument("--expected-train-job-id")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    payload = validate_completion(
        args.record_root,
        expected_effective_step=args.expected_effective_step,
        expected_continuation_local_step=args.expected_continuation_local_step,
        expected_train_job_id=args.expected_train_job_id,
    )
    print(
        "[batch44-r3-warmstart-quick20-validate] PASS "
        f"effective={payload['effective_step']} "
        f"local={payload['continuation_local_step']} runs={len(payload['runs'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

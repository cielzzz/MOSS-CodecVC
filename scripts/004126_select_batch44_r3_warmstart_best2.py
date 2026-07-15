#!/usr/bin/env python3
"""Select the Batch-44 r3 warm-start Best2 from strict quick20 evidence.

This selector is deliberately local and read-only with respect to training.  It
never submits, stops, or starts a task.  All four registered effective
checkpoints must have a completion accepted by
``batch44_r3_warmstart_quick20_validator.py`` before selection can occur.

The continuation reset its local step counter after loading effective step
10,000, so the registered candidate mapping is:

    effective 24k/26k/28k/30k -> continuation local 14k/16k/18k/20k

Missing future evidence produces a ``pending`` report and exit code 3.  A
partial or mutated completion fails closed instead of being treated as
pending.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "moss_codecvc.batch44_r3_warmstart_best2_selection.v1"
EXPERIMENT_ID = "batch44_r3_v1_weights_only_warmstart"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAMP = "20260713"
BASE_EFFECTIVE_STEP = 10_000
REGISTERED_EFFECTIVE_STEPS = (24_000, 26_000, 28_000, 30_000)
EXPECTED_TRAIN_JOB_ID = "job-165f3b1d-8c7d-47d8-8882-bdb86ef642ab"
EXPECTED_RUN_DIR_REL = Path(
    "outputs/lora_runs/ver2_9_5_final_r3_v1_warmstart10k_to30k"
)
EXPECTED_CONTRACT_REL = Path(
    "trainset/qz_jobs/"
    "ver23_batch44_r3_v1_warmstart10k_to30k_20260713/"
    "warm_start_contract.json"
)
EXPECTED_QUICK20_OUTPUT_REL = Path(
    "testset/outputs/ver23_batch44_r3_warmstart_quick20_20260713"
)
EXPECTED_CONTRACT_SHA256 = (
    "2d686e5e57b70fcaa3db8c8eb2b306003a38599b2c9ac37023979d80b6d9fc34"
)
DEFAULT_OUTPUT_REL = Path(
    "testset/outputs/batch44_closure_20260713"
)
SELECTION_FILENAME = "best2_r3_selection.json"
SUMMARY_FILENAME = "best2_r3_summary.md"
EXPECTED_CHECKPOINT_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "README.md",
    "timbre_memory_adapter.pt",
    "timbre_memory_config.json",
)
EXPECTED_STATE_RESETS = [
    "optimizer",
    "scheduler",
    "rng",
    "global_step",
    "data_iterator",
]


class PendingEvidence(RuntimeError):
    """Raised when one or more registered future artifacts do not exist yet."""

    def __init__(
        self,
        missing: Sequence[str],
        *,
        available_effective_steps: Sequence[int] = (),
        contract_binding: Mapping[str, Any] | None = None,
    ) -> None:
        self.missing = list(missing)
        self.available_effective_steps = list(available_effective_steps)
        self.contract_binding = dict(contract_binding or {})
        super().__init__("; ".join(self.missing))


class InvalidEvidence(RuntimeError):
    """Raised when present evidence violates the registered contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise InvalidEvidence(f"missing/empty provenance artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_text(path: Path, value: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidEvidence(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise InvalidEvidence(f"{label} must be a JSON object: {path}")
    return payload


def load_validator():
    path = Path(__file__).with_name("batch44_r3_warmstart_quick20_validator.py")
    if not path.is_file():
        raise InvalidEvidence(f"missing registered strict validator: {path}")
    name = "moss_codecvc_batch44_r3_best2_strict_validator"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise InvalidEvidence(f"cannot load registered strict validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, path.resolve()


def continuation_local_step(effective_step: int) -> int:
    if effective_step not in REGISTERED_EFFECTIVE_STEPS:
        raise InvalidEvidence(
            f"effective step must be one of {REGISTERED_EFFECTIVE_STEPS}, "
            f"got {effective_step}"
        )
    local_step = effective_step - BASE_EFFECTIVE_STEP
    if local_step not in {14_000, 16_000, 18_000, 20_000}:
        raise InvalidEvidence(
            f"registered effective/local mapping drift: {effective_step}/{local_step}"
        )
    return local_step


def registered_space() -> dict[str, Any]:
    return {
        "arm": "r3",
        "candidate_count": len(REGISTERED_EFFECTIVE_STEPS),
        "effective_steps": list(REGISTERED_EFFECTIVE_STEPS),
        "effective_to_continuation_local": {
            str(step): continuation_local_step(step)
            for step in REGISTERED_EFFECTIVE_STEPS
        },
        "base_effective_step": BASE_EFFECTIVE_STEP,
        "mapping": "effective_step = 10000 + continuation_local_step",
    }


def audit_contract(project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    project_root = project_root.expanduser().resolve()
    contract = (project_root / EXPECTED_CONTRACT_REL).resolve()
    run_dir = (project_root / EXPECTED_RUN_DIR_REL).resolve()
    if not contract.is_file():
        raise PendingEvidence([f"missing warm-start contract: {contract}"])
    actual_sha = sha256_file(contract)
    if actual_sha != EXPECTED_CONTRACT_SHA256:
        raise InvalidEvidence(
            f"warm-start contract SHA256={actual_sha}, "
            f"expected {EXPECTED_CONTRACT_SHA256}"
        )
    payload = load_json_object(contract, "warm-start contract")
    expected = {
        "schema": "batch44_r3_weights_only_warm_start_v1",
        "status": "submitted",
        "job_id": EXPECTED_TRAIN_JOB_ID,
        "output_dir": str(run_dir),
        "source_effective_step": BASE_EFFECTIVE_STEP,
        "effective_step_offset": BASE_EFFECTIVE_STEP,
        "continuation_local_target_step": 20_000,
        "effective_target_step": 30_000,
        "resume_semantics": "weights_only_warm_start_not_exact_resume",
        "step_mapping": "effective_step = 10000 + continuation_local_step",
        "state_resets": EXPECTED_STATE_RESETS,
        "full_data_sha256_verified": True,
    }
    drift = {
        key: {"expected": wanted, "actual": payload.get(key)}
        for key, wanted in expected.items()
        if payload.get(key) != wanted
    }
    data = payload.get("data")
    if not isinstance(data, dict):
        drift["data"] = {"expected": "registered old-v1 r3 mix", "actual": data}
    else:
        no_text = data.get("no_text")
        text = data.get("text")
        if not isinstance(no_text, dict) or no_text.get("repeat") != 1:
            drift["data.no_text.repeat"] = {
                "expected": 1,
                "actual": no_text.get("repeat") if isinstance(no_text, dict) else None,
            }
        if not isinstance(text, dict) or text.get("repeat") != 3:
            drift["data.text.repeat"] = {
                "expected": 3,
                "actual": text.get("repeat") if isinstance(text, dict) else None,
            }
    if drift:
        raise InvalidEvidence(f"warm-start contract identity drift: {drift}")
    if not run_dir.is_dir():
        raise PendingEvidence(
            [f"missing continuation run directory: {run_dir}"],
            contract_binding={
                "artifact": artifact(contract),
                "job_id": EXPECTED_TRAIN_JOB_ID,
                "run_dir": str(run_dir),
            },
        )
    binding = {
        "artifact": artifact(contract),
        "job_id": EXPECTED_TRAIN_JOB_ID,
        "run_dir": str(run_dir),
        "resume_semantics": payload["resume_semantics"],
        "source_effective_step": BASE_EFFECTIVE_STEP,
        "effective_target_step": 30_000,
        "state_resets": list(EXPECTED_STATE_RESETS),
    }
    return payload, binding


def quick20_record_root(project_root: Path, effective_step: int) -> Path:
    return (
        project_root
        / "trainset/local_jobs"
        / f"ver23_batch44_r3_warmstart_quick20_step{effective_step}_{STAMP}"
    ).resolve()


def expected_checkpoint(project_root: Path, effective_step: int) -> Path:
    return (
        project_root
        / EXPECTED_RUN_DIR_REL
        / f"step-{continuation_local_step(effective_step)}"
    ).resolve()


def _validate_present_record(record_root: Path, effective_step: int) -> None:
    completion = record_root / "COMPLETED.json"
    marker = record_root / "complete.marker"
    completion_exists = completion.is_file()
    marker_exists = marker.is_file()
    if not completion_exists and not marker_exists:
        raise PendingEvidence(
            [
                f"effective-{effective_step}: missing strict completion evidence "
                f"at {record_root}"
            ]
        )
    if completion_exists != marker_exists:
        raise InvalidEvidence(
            f"effective-{effective_step}: partial completion evidence at {record_root}"
        )


def _metric_rows(completion: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = completion.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != {"json", "tsv", "md"}:
        raise InvalidEvidence("strict completion metric artifact set drift")
    metrics_json = Path(str(metrics["json"].get("path") or "")).resolve()
    try:
        rows = json.loads(metrics_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidEvidence(f"cannot read strict quick20 metrics {metrics_json}: {exc}") from exc
    if not isinstance(rows, list) or len(rows) != 2:
        raise InvalidEvidence(f"strict quick20 metrics must contain two rows: {metrics_json}")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise InvalidEvidence("strict quick20 metric row must be an object")
        mode = str(row.get("mode") or "")
        if mode not in {"no_text", "text"} or mode in indexed:
            raise InvalidEvidence(f"invalid/duplicate strict metric mode: {mode!r}")
        indexed[mode] = dict(row)
    if set(indexed) != {"no_text", "text"}:
        raise InvalidEvidence(f"strict metric mode set drift: {set(indexed)}")
    return indexed["no_text"], indexed["text"]


def audit_candidate(
    project_root: Path,
    *,
    effective_step: int,
    validator: Any,
    contract_binding: Mapping[str, Any],
) -> dict[str, Any]:
    local_step = continuation_local_step(effective_step)
    record_root = quick20_record_root(project_root, effective_step)
    _validate_present_record(record_root, effective_step)
    try:
        completion = validator.validate_completion(
            record_root,
            expected_effective_step=effective_step,
            expected_continuation_local_step=local_step,
            expected_train_job_id=EXPECTED_TRAIN_JOB_ID,
        )
    except Exception as exc:  # noqa: BLE001 - preserve validator context
        raise InvalidEvidence(
            f"effective-{effective_step}: strict completion validation failed: {exc}"
        ) from exc

    expected_contract = (project_root / EXPECTED_CONTRACT_REL).resolve()
    expected_run_dir = (project_root / EXPECTED_RUN_DIR_REL).resolve()
    checkpoint = expected_checkpoint(project_root, effective_step)
    expected_output_root = (project_root / EXPECTED_QUICK20_OUTPUT_REL).resolve()
    expected = {
        "checkpoint": str(checkpoint),
        "warm_start_contract": str(expected_contract),
        "warm_start_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "train_job_id": EXPECTED_TRAIN_JOB_ID,
        "output_root": str(expected_output_root),
    }
    drift: dict[str, dict[str, Any]] = {}
    for key, wanted in expected.items():
        actual = completion.get(key)
        if key in {"checkpoint", "warm_start_contract", "output_root"}:
            actual = str(Path(str(actual or "")).expanduser().resolve())
        if actual != wanted:
            drift[key] = {"expected": wanted, "actual": actual}
    if drift:
        raise InvalidEvidence(
            f"effective-{effective_step}: strict completion binding drift: {drift}"
        )
    if checkpoint.parent != expected_run_dir:
        raise InvalidEvidence(
            f"effective-{effective_step}: checkpoint escaped continuation run: {checkpoint}"
        )

    checkpoint_files = completion.get("checkpoint_files")
    if not isinstance(checkpoint_files, dict) or set(checkpoint_files) != set(
        EXPECTED_CHECKPOINT_FILES
    ):
        raise InvalidEvidence(
            f"effective-{effective_step}: five-file checkpoint artifact set drift"
        )
    for name in EXPECTED_CHECKPOINT_FILES:
        spec = checkpoint_files[name]
        expected_path = (checkpoint / name).resolve()
        if not isinstance(spec, dict) or Path(str(spec.get("path") or "")).resolve() != expected_path:
            raise InvalidEvidence(
                f"effective-{effective_step}: checkpoint artifact path drift: {name}"
            )
        if not expected_path.is_file():
            raise InvalidEvidence(
                f"effective-{effective_step}: completed checkpoint file missing: {expected_path}"
            )
    no_text, text = _metric_rows(completion)
    for mode, row in (("no_text", no_text), ("text", text)):
        if row.get("warm_start_contract_sha256") != EXPECTED_CONTRACT_SHA256:
            raise InvalidEvidence(
                f"effective-{effective_step}: {mode} metric contract SHA drift"
            )
        if row.get("train_job_id") != EXPECTED_TRAIN_JOB_ID:
            raise InvalidEvidence(
                f"effective-{effective_step}: {mode} metric train job drift"
            )
        for field in ("sim_ref", "margin", "cer"):
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise InvalidEvidence(
                    f"effective-{effective_step}: {mode}.{field} is invalid"
                ) from exc
            if not math.isfinite(value):
                raise InvalidEvidence(
                    f"effective-{effective_step}: {mode}.{field} is non-finite"
                )

    candidate_id = f"r3_effective-{effective_step}"
    return {
        "candidate_id": candidate_id,
        "arm": "r3",
        "effective_step": effective_step,
        "continuation_local_step": local_step,
        "train_job_id": EXPECTED_TRAIN_JOB_ID,
        "warm_start_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "checkpoint": {
            "path": str(checkpoint),
            "manifest_sha256": completion["checkpoint_manifest_sha256"],
            "files": checkpoint_files,
        },
        "completion": {
            "record_root": str(record_root),
            "completed_json": artifact(record_root / "COMPLETED.json"),
            "complete_marker": artifact(record_root / "complete.marker"),
            "completed_utc": completion.get("completed_utc"),
            "output_root": completion["output_root"],
        },
        "quick20": {
            "metrics": completion["metrics"],
            "no_text": no_text,
            "text": text,
        },
        "resume_semantics": contract_binding["resume_semantics"],
    }


def ranking_key(candidate: Mapping[str, Any]) -> tuple[float, float, float, float, int]:
    quick20 = candidate["quick20"]
    no_text = quick20["no_text"]
    text = quick20["text"]
    return (
        -float(no_text["sim_ref"]),
        -float(no_text["margin"]),
        float(no_text["cer"]),
        float(text["cer"]),
        -int(candidate["effective_step"]),
    )


def build_selection(project_root: Path) -> dict[str, Any]:
    project_root = project_root.expanduser().resolve()
    _, contract_binding = audit_contract(project_root)
    validator, validator_path = load_validator()
    validator_checkpoint_files = tuple(validator.CHECKPOINT_FILES)
    if validator_checkpoint_files != EXPECTED_CHECKPOINT_FILES:
        raise InvalidEvidence(
            "registered validator checkpoint file contract drift: "
            f"{validator_checkpoint_files}"
        )

    candidates: list[dict[str, Any]] = []
    missing: list[str] = []
    for effective_step in REGISTERED_EFFECTIVE_STEPS:
        try:
            candidates.append(
                audit_candidate(
                    project_root,
                    effective_step=effective_step,
                    validator=validator,
                    contract_binding=contract_binding,
                )
            )
        except PendingEvidence as exc:
            missing.extend(exc.missing)
    if missing:
        raise PendingEvidence(
            missing,
            available_effective_steps=[
                int(candidate["effective_step"]) for candidate in candidates
            ],
            contract_binding=contract_binding,
        )
    if len(candidates) != len(REGISTERED_EFFECTIVE_STEPS):
        raise InvalidEvidence(
            f"expected four strictly validated candidates, got {len(candidates)}"
        )

    ranked = sorted(candidates, key=ranking_key)
    selected = ranked[:2]
    selected_ids = [str(candidate["candidate_id"]) for candidate in selected]
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
        candidate["selected_for_full320"] = candidate["candidate_id"] in selected_ids
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "selected",
        "generated_at_utc": utc_now(),
        "project_root": str(project_root),
        "read_only_contract": {
            "training_mutation": False,
            "task_submission": False,
            "purpose": "select two checkpoints for later full320 evaluation",
        },
        "registered_candidate_space": registered_space(),
        "warm_start": contract_binding,
        "strict_validator": artifact(validator_path),
        "ranking": {
            "primary_scope": "fixed no_text quick20 WavLM-large-SV proxy",
            "ordered_criteria": [
                "no_text WavLM SIM(ref) descending",
                "no_text WavLM SIM(ref)-SIM(src) margin descending",
                "no_text CER ascending",
                "text CER ascending",
                "effective step descending",
            ],
            "warning": (
                "Best2 is an evaluation shortlist, not a final model decision. "
                "Both selected checkpoints still require strict full320 evaluation."
            ),
        },
        "selected_candidate_ids": selected_ids,
        "best2": [
            {
                "rank": candidate["rank"],
                "candidate_id": candidate["candidate_id"],
                "effective_step": candidate["effective_step"],
                "continuation_local_step": candidate["continuation_local_step"],
                "checkpoint": candidate["checkpoint"]["path"],
                "no_text_wavlm_sim_ref": float(
                    candidate["quick20"]["no_text"]["sim_ref"]
                ),
                "no_text_margin": float(candidate["quick20"]["no_text"]["margin"]),
                "no_text_cer": float(candidate["quick20"]["no_text"]["cer"]),
                "text_cer": float(candidate["quick20"]["text"]["cer"]),
            }
            for candidate in selected
        ],
        "candidates": ranked,
    }


def pending_payload(project_root: Path, exc: PendingEvidence) -> dict[str, Any]:
    project_root = project_root.expanduser().resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "pending",
        "generated_at_utc": utc_now(),
        "project_root": str(project_root),
        "read_only_contract": {
            "training_mutation": False,
            "task_submission": False,
            "purpose": "wait for all four strict quick20 completions",
        },
        "registered_candidate_space": registered_space(),
        "warm_start": exc.contract_binding or {
            "job_id": EXPECTED_TRAIN_JOB_ID,
            "run_dir": str((project_root / EXPECTED_RUN_DIR_REL).resolve()),
            "contract": str((project_root / EXPECTED_CONTRACT_REL).resolve()),
            "expected_contract_sha256": EXPECTED_CONTRACT_SHA256,
        },
        "available_effective_steps": sorted(exc.available_effective_steps),
        "missing_evidence": exc.missing,
        "selected_candidate_ids": [],
        "best2": [],
        "ranking": {
            "status": "not_run_until_all_four_candidates_validate",
            "ordered_criteria": [
                "no_text WavLM SIM(ref) descending",
                "no_text margin descending",
                "no_text CER ascending",
                "text CER ascending",
                "effective step descending",
            ],
        },
    }


def render_summary(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Batch-44 r3 warm-start Best2",
        "",
        f"Status: **{payload['status']}**",
        "",
        "Training semantics: weights-only continuation from effective step 10,000; "
        "optimizer/scheduler/RNG/data position were reset.",
        "",
    ]
    if payload["status"] == "pending":
        lines.extend(
            [
                "Selection has not run because the registered candidate set is incomplete.",
                "",
                "## Missing evidence",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in payload["missing_evidence"])
        lines.extend(
            [
                "",
                "Available effective steps: "
                + ", ".join(str(step) for step in payload["available_effective_steps"])
                if payload["available_effective_steps"]
                else "Available effective steps: none",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "Ranking: no_text SIM(ref) descending, then no_text margin descending, "
            "no_text CER ascending, text CER ascending, and effective step descending.",
            "",
            "| Rank | Candidate | Effective | Local | no_text SIM(ref) | "
            "no_text margin | no_text CER | text CER | Best2 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for candidate in payload["candidates"]:
        no_text = candidate["quick20"]["no_text"]
        text = candidate["quick20"]["text"]
        lines.append(
            f"| {candidate['rank']} | {candidate['candidate_id']} | "
            f"{candidate['effective_step']} | {candidate['continuation_local_step']} | "
            f"{float(no_text['sim_ref']):.6f} | {float(no_text['margin']):+.6f} | "
            f"{float(no_text['cer']):.6f} | {float(text['cer']):.6f} | "
            f"{'yes' if candidate['selected_for_full320'] else 'no'} |"
        )
    lines.extend(["", "## Selected for strict full320", ""])
    for winner in payload["best2"]:
        lines.append(
            f"- Winner{winner['rank']}: `{winner['candidate_id']}` — "
            f"SIM(ref)={winner['no_text_wavlm_sim_ref']:.6f}, "
            f"margin={winner['no_text_margin']:+.6f}, "
            f"no_text CER={winner['no_text_cer']:.6f}, "
            f"text CER={winner['text_cer']:.6f}."
        )
    lines.extend(
        [
            "",
            "These are shortlist winners only; neither is the final model until both "
            "strict full320 evaluations complete.",
            "",
        ]
    )
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    result.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "testset/outputs/batch44_closure_20260713 under project root."
        ),
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (project_root / DEFAULT_OUTPUT_REL).resolve()
    )
    selection_path = output_dir / SELECTION_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    try:
        payload = build_selection(project_root)
    except PendingEvidence as exc:
        payload = pending_payload(project_root, exc)
        atomic_json(selection_path, payload)
        atomic_text(summary_path, render_summary(payload))
        print(
            "[batch44-r3-best2] pending: " + "; ".join(exc.missing),
            file=sys.stderr,
        )
        print(f"[batch44-r3-best2] report={selection_path}")
        return 3
    except (InvalidEvidence, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: Batch-44 r3 Best2 evidence validation failed: {exc}", file=sys.stderr)
        return 1
    atomic_json(selection_path, payload)
    atomic_text(summary_path, render_summary(payload))
    print(
        "[batch44-r3-best2] selected="
        + ",".join(payload["selected_candidate_ids"])
    )
    print(f"[batch44-r3-best2] report={selection_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

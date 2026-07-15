#!/usr/bin/env python3
"""Serialize the human-approved Batch-44 winner into Batch-42's final 8/8 table.

This watcher starts *after* the three Best3 listening reviews and an explicit
004107 finalization have produced the canonical ``FINAL_SELECTION.json``.  It
never generates reviews, chooses a winner, or invokes 004107's finalization
CLI.  Its only use of 004107 is the strict, read-only provenance replay.

The live state machine is deliberately one-way and fail closed::

    strict 004107 FINAL_SELECTION validation
      -> 004108 live smoke -> QZ succeeded + SMOKE_COMPLETED.json
      -> 004108 live full  -> QZ succeeded + EN567/ZH1194 COMPLETED.json
      -> 004109 live score -> QZ succeeded + complete unified scorer artifacts
      -> 004109 STAGE=table -> verified Batch-42 8/8 publication

Every remote stage has exactly one accepted ledger row.  A failed/stopped QZ
job, a successful job without its atomic completion artifact, a submission
attempt without a ledger, or any provenance drift writes ``HALTED.json`` and
stops.  There is no automatic retry.  Removing/overriding that halt is an
explicit manual recovery action outside this program.

Safe readiness check (no qzcli call, no writes outside the watcher state)::

    python scripts/004115_watch_batch44_postfinal_batch42_publish.py \
      --mode once --action plan

Live monitoring is intentionally triple-gated at this layer, while 004108 and
004109 retain their own independent live confirmation gates::

    ALLOW_LIVE_SUBMIT=1 CONFIRM_BATCH44_POSTFINAL_ORCHESTRATOR=1 \
      python scripts/004115_watch_batch44_postfinal_batch42_publish.py \
      --mode monitor --action submit

Do not start this watcher until human review has finished and 004107 has
created the canonical final-selection file.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CANONICAL_PROJECT_ROOT = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
    "MOSS-CodecVC"
)
CANONICAL_QZCLI = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
    "pair_construction/scripts/qzcli_with_deps.sh"
)
CANONICAL_QZCLI_HOME = Path(
    "/inspire/ssd/project/embodied-multimodality/public/xyzhang/.codex/qzcli_home"
)

ALLOWED_WORKSPACE = "ws-8207e9e2-e733-4eec-a475-cfa1c36480ba"
ALLOWED_PROJECT = "project-c67c548f-f02c-453b-ba5b-8745db6886e7"
ALLOWED_COMPUTE_GROUP_ID = "lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"
ALLOWED_COMPUTE_GROUP_NAME = "MTTS-3-2-0715"
ALLOWED_SPEC = "67b10bc6-78b0-41a3-aaf4-358eeeb99009"
ALLOWED_GPU_TYPE = "NVIDIA_H200_SXM_141G"
ALLOWED_INSTANCES = 1
ALLOWED_GPUS = 8

FINAL_SCHEMA = "moss_codecvc.batch44_v1_final_selection.v1"
SMOKE_SCHEMA = "moss_codecvc.batch42_pathx_strict_smoke_completion.v1"
FULL_SCHEMA = "moss_codecvc.batch42_pathx_strict_completion.v1"
SCORER_SUMMARY_SCHEMA = "moss_codecvc.batch42_system_unified_summary.v1"
SCORER_COMPLETION_SCHEMA = "moss_codecvc.batch42_unified_scorer_completion.v2"
STRICT_SCORER_AUDIT_SCHEMA = "moss_codecvc.batch42_strict_scorer_audit.v1"
UNIFIED_EVAL_SCHEMA = "moss_codecvc.unified_vc_eval.v1"

JOB_RE = re.compile(
    r"^job-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
MODEL_FILES = {
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "timbre_memory_config.json",
    "timbre_memory_adapter.pt",
}
RESOURCE_CONTRACT = {
    "compute_group": ALLOWED_COMPUTE_GROUP_NAME,
    "gpu_type": ALLOWED_GPU_TYPE,
    "gpus": ALLOWED_GPUS,
    "instances": ALLOWED_INSTANCES,
}
SCORER_RESOURCE_CONTRACT = {
    "workspace_id": ALLOWED_WORKSPACE,
    "project_id": ALLOWED_PROJECT,
    "compute_group_id": ALLOWED_COMPUTE_GROUP_ID,
    "compute_group_name": ALLOWED_COMPUTE_GROUP_NAME,
    "spec_id": ALLOWED_SPEC,
    "gpu_type": ALLOWED_GPU_TYPE,
    "instances": ALLOWED_INSTANCES,
    "gpus": ALLOWED_GPUS,
    "shards": 8,
}
STRICT_MANIFEST_SHA256 = {
    "en": "48549d8029e680d74656660191c4641ca5a8040ccbe3252ce89bfc3b0c9c75ae",
    "zh": "4b637cc1cff33dc369954755538d12396fc92d439a52742103a29b7c563cf6df",
}
EXPECTED_RUN_DIRS = {
    "r3": "ver2_9_5_final_r3_v1_30k",
    "r5": "ver2_9_5_final_r5_v1_30k",
}
EXPECTED_REPEATS = {"r3": 3, "r5": 5}
EXPECTED_TRAIN_JOBS = {
    "r3": "job-2b91d332-d500-4279-84f9-0a6a81a376aa",
    "r5": "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c",
}
EXPECTED_STEPS = {26000, 28000, 30000}


class ContractError(RuntimeError):
    """A persistent contract violation which must halt the pipeline."""


class Pending(RuntimeError):
    """A normal not-ready state which should be polled later."""

    def __init__(self, state: str, detail: str) -> None:
        super().__init__(detail)
        self.state = state
        self.detail = detail


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"JSON object required: {path}")
    return payload


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def ensure_relative(child: Path, parent: Path, label: str) -> Path:
    child = child.expanduser().resolve()
    parent = parent.expanduser().resolve()
    try:
        child.relative_to(parent)
    except ValueError as exc:
        raise ContractError(f"{label} escapes registered root: {child} not under {parent}") from exc
    return child


def count_nonempty_lines(path: Path) -> int:
    if not path.is_file():
        raise ContractError(f"missing line-oriented artifact: {path}")
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def assert_finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ContractError(f"{label} is not finite: {number}")
    return number


@dataclass(frozen=True)
class Config:
    project_root: Path
    test_mode: bool
    mode: str
    action: str
    allow_live_submit: bool
    confirm_orchestrator: bool
    poll_seconds: int
    max_scans: int
    stop_when_complete: bool
    run_tag: str
    smoke_gate_tag: str
    python: Path
    final_validator: Path
    inference_wrapper: Path
    score_wrapper: Path
    scorer_provenance_helper: Path
    qzcli: Path
    qzcli_home: Path
    final_selection: Path
    plan_root: Path
    state_root: Path
    smoke_output: Path
    smoke_record: Path
    full_output: Path
    full_record: Path
    scorer_output: Path
    scorer_record: Path
    table_prefix: Path
    interim_table: Path
    expected_eval_code_root: Path
    en_expected: int
    zh_expected: int
    strict_manifest_sha256: Mapping[str, str]

    @property
    def smoke_marker(self) -> Path:
        return self.smoke_output / "SMOKE_COMPLETED.json"

    @property
    def full_marker(self) -> Path:
        return self.full_output / "COMPLETED.json"

    @property
    def scorer_marker(self) -> Path:
        return self.scorer_output / "completion.json"

    @property
    def table_json(self) -> Path:
        return self.table_prefix.with_suffix(".json")

    @property
    def table_md(self) -> Path:
        return self.table_prefix.with_suffix(".md")

    @property
    def table_tsv(self) -> Path:
        return self.table_prefix.with_suffix(".tsv")

    @property
    def table_cross_validation_tsv(self) -> Path:
        return self.table_prefix.with_name(self.table_prefix.name + ".cross_validation.tsv")

    @property
    def table_provenance(self) -> Path:
        return self.table_prefix.with_suffix(".provenance.json")


def env_bool(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    if value not in {"0", "1"}:
        raise ContractError(f"{name} must be 0 or 1")
    return value == "1"


def safe_component(value: str, label: str) -> str:
    if not value or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ContractError(f"{label} must be a non-empty safe path component")
    return value


def build_config(args: argparse.Namespace) -> Config:
    test_mode = env_bool("BATCH44_POSTFINAL_TEST_MODE")
    canonical = CANONICAL_PROJECT_ROOT.resolve()
    project = Path(os.environ.get("PROJECT_ROOT", str(canonical))).resolve()
    if project != canonical and not test_mode:
        raise ContractError("non-canonical PROJECT_ROOT is allowed only in test mode")

    run_tag = safe_component(os.environ.get("RUN_TAG", "20260713_mtts"), "RUN_TAG")
    smoke_tag = safe_component(
        os.environ.get("SMOKE_GATE_TAG", "20260713_mtts"), "SMOKE_GATE_TAG"
    )
    mode = args.mode or os.environ.get("MODE", "once")
    action = args.action or os.environ.get("ACTION", "plan")
    if mode not in {"once", "monitor"}:
        raise ContractError("MODE must be once or monitor")
    if action not in {"plan", "submit"}:
        raise ContractError("ACTION must be plan or submit")
    allow = env_bool("ALLOW_LIVE_SUBMIT")
    confirm = env_bool("CONFIRM_BATCH44_POSTFINAL_ORCHESTRATOR")
    if action == "submit" and not (allow and confirm):
        raise ContractError(
            "ACTION=submit requires ALLOW_LIVE_SUBMIT=1 and "
            "CONFIRM_BATCH44_POSTFINAL_ORCHESTRATOR=1"
        )

    poll = args.poll_seconds
    if poll is None:
        poll = int(os.environ.get("POLL_SECONDS", "60"))
    max_scans = args.max_scans
    if max_scans is None:
        max_scans = int(os.environ.get("MAX_SCANS", "0"))
    if poll <= 0 or max_scans < 0:
        raise ContractError("POLL_SECONDS must be positive and MAX_SCANS non-negative")

    final_selection = Path(
        os.environ.get(
            "FINAL_SELECTION_JSON",
            str(project / "testset/outputs/batch44_best3_20260713/FINAL_SELECTION.json"),
        )
    ).resolve()
    plan_root = Path(
        os.environ.get(
            "PLAN_ROOT",
            str(project / "trainset/qz_jobs/batch42_pathx_final_batch44_v1_materialized_20260713"),
        )
    ).resolve()
    state_root = Path(
        os.environ.get(
            "STATE_ROOT",
            str(project / "trainset/qz_jobs/batch44_postfinal_batch42_publish_20260713"),
        )
    ).resolve()
    smoke_output = Path(
        os.environ.get(
            "SMOKE_OUTPUT_ROOT",
            str(project / f"testset/outputs/batch42_pathx_strict_smoke_gate_{smoke_tag}"),
        )
    ).resolve()
    smoke_record = Path(
        os.environ.get(
            "SMOKE_RECORD_ROOT",
            str(project / f"trainset/qz_jobs/batch42_pathx_strict_smoke_{smoke_tag}"),
        )
    ).resolve()
    full_output = Path(
        os.environ.get(
            "INFERENCE_ROOT",
            str(project / f"testset/outputs/batch42_pathx_strict_path_x_final_{run_tag}"),
        )
    ).resolve()
    full_record = Path(
        os.environ.get(
            "FULL_RECORD_ROOT",
            str(project / f"trainset/qz_jobs/batch42_pathx_strict_full_{run_tag}"),
        )
    ).resolve()
    scorer_output = Path(
        os.environ.get(
            "SCORER_OUTPUT",
            str(project / f"testset/outputs/batch42_unified_scorers_path_x_final_{run_tag}"),
        )
    ).resolve()
    scorer_record = Path(
        os.environ.get(
            "SCORER_RECORD",
            str(project / f"trainset/qz_jobs/batch42_unified_scorers_path_x_final_{run_tag}"),
        )
    ).resolve()
    table_prefix = Path(
        os.environ.get(
            "TABLE_PREFIX",
            str(project / "testset/outputs/batch42_baseline_tables_20260711/batch42_baseline_final"),
        )
    ).resolve()
    interim_table = Path(
        os.environ.get(
            "INTERIM_TABLE_JSON",
            str(project / "testset/outputs/batch42_baseline_tables_20260711/batch42_baseline_interim.json"),
        )
    ).resolve()

    expected_eval = Path(
        os.environ.get(
            "EXPECTED_EVAL_CODE_ROOT",
            "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
            "MOSS-CodecVC_snapshots/ver23_batch37_eval_20260711_1092820",
        )
    ).resolve()
    en_expected = int(os.environ.get("BATCH44_POSTFINAL_TEST_EN_EXPECTED", "567"))
    zh_expected = int(os.environ.get("BATCH44_POSTFINAL_TEST_ZH_EXPECTED", "1194"))
    manifest_hashes = dict(STRICT_MANIFEST_SHA256)
    if test_mode:
        manifest_hashes = {
            "en": os.environ.get("BATCH44_POSTFINAL_TEST_EN_MANIFEST_SHA256", manifest_hashes["en"]),
            "zh": os.environ.get("BATCH44_POSTFINAL_TEST_ZH_MANIFEST_SHA256", manifest_hashes["zh"]),
        }
    elif en_expected != 567 or zh_expected != 1194 or expected_eval != Path(
        "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
        "MOSS-CodecVC_snapshots/ver23_batch37_eval_20260711_1092820"
    ):
        raise ContractError("production EN/ZH denominators and eval snapshot are hard-locked")

    return Config(
        project_root=project,
        test_mode=test_mode,
        mode=mode,
        action=action,
        allow_live_submit=allow,
        confirm_orchestrator=confirm,
        poll_seconds=poll,
        max_scans=max_scans,
        stop_when_complete=env_bool("STOP_WHEN_COMPLETE", "1"),
        run_tag=run_tag,
        smoke_gate_tag=smoke_tag,
        python=Path(os.environ.get("PYTHON", sys.executable)).resolve(),
        final_validator=Path(
            os.environ.get(
                "FINAL_VALIDATOR",
                str(project / "scripts/004107_finalize_batch43_pathx_final.py"),
            )
        ).resolve(),
        inference_wrapper=Path(
            os.environ.get(
                "INFERENCE_WRAPPER",
                str(project / "scripts/004108_submit_batch42_pathx_final_strict_qz.sh"),
            )
        ).resolve(),
        score_wrapper=Path(
            os.environ.get(
                "SCORE_WRAPPER",
                str(project / "scripts/004109_score_and_publish_batch42_pathx_final.sh"),
            )
        ).resolve(),
        scorer_provenance_helper=Path(
            os.environ.get(
                "SCORER_PROVENANCE_HELPER",
                str(project / "scripts/batch42_scorer_provenance.py"),
            )
        ).resolve(),
        qzcli=Path(os.environ.get("QZCLI", str(CANONICAL_QZCLI))).resolve(),
        qzcli_home=Path(os.environ.get("QZCLI_HOME", str(CANONICAL_QZCLI_HOME))).resolve(),
        final_selection=final_selection,
        plan_root=plan_root,
        state_root=state_root,
        smoke_output=smoke_output,
        smoke_record=smoke_record,
        full_output=full_output,
        full_record=full_record,
        scorer_output=scorer_output,
        scorer_record=scorer_record,
        table_prefix=table_prefix,
        interim_table=interim_table,
        expected_eval_code_root=expected_eval,
        en_expected=en_expected,
        zh_expected=zh_expected,
        strict_manifest_sha256=manifest_hashes,
    )


def validate_static_config(config: Config) -> None:
    for label, path in (
        ("Python", config.python),
        ("004107 validator", config.final_validator),
        ("004108 inference wrapper", config.inference_wrapper),
        ("004109 score/table wrapper", config.score_wrapper),
        ("Batch-42 scorer provenance helper", config.scorer_provenance_helper),
    ):
        if not path.is_file():
            raise ContractError(f"missing {label}: {path}")
    if not os.access(config.python, os.X_OK):
        raise ContractError(f"Python is not executable: {config.python}")

    if config.test_mode:
        if config.qzcli == CANONICAL_QZCLI.resolve() and config.action == "submit":
            raise ContractError("test mode may not use the canonical qzcli for submit simulation")
        return

    expected = {
        "final_validator": config.project_root / "scripts/004107_finalize_batch43_pathx_final.py",
        "inference_wrapper": config.project_root / "scripts/004108_submit_batch42_pathx_final_strict_qz.sh",
        "score_wrapper": config.project_root / "scripts/004109_score_and_publish_batch42_pathx_final.sh",
        "scorer_provenance_helper": config.project_root / "scripts/batch42_scorer_provenance.py",
    }
    for name, path in expected.items():
        if getattr(config, name) != path.resolve():
            raise ContractError(f"production {name} is hard-locked to {path}")
    if config.qzcli != CANONICAL_QZCLI.resolve() or config.qzcli_home != CANONICAL_QZCLI_HOME.resolve():
        raise ContractError("production qzcli wrapper/HOME are hard-locked")
    if not config.qzcli.is_file() or not os.access(config.qzcli, os.X_OK):
        raise ContractError(f"qzcli-local wrapper is not executable: {config.qzcli}")
    if not config.qzcli_home.is_dir():
        raise ContractError(f"qzcli-local HOME is missing: {config.qzcli_home}")

    inference_text = config.inference_wrapper.read_text(encoding="utf-8")
    score_text = config.score_wrapper.read_text(encoding="utf-8")
    for needle in (
        'CONFIRM_BATCH44_FINAL_STRICT="${CONFIRM_BATCH44_FINAL_STRICT:-0}"',
        '"spec": "67b10bc6-78b0-41a3-aaf4-358eeeb99009"',
        '"compute_group": "MTTS-3-2-0715"',
        '"gpu_type": "NVIDIA_H200_SXM_141G"',
        '"instances": 1',
        '"gpus": 8',
    ):
        if needle not in inference_text:
            raise ContractError(f"004108 lost registered live/resource gate: {needle}")
    for needle in (
        'CONFIRM_BATCH44_FINAL_SCORERS="${CONFIRM_BATCH44_FINAL_SCORERS:-0}"',
        'ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"',
        'ALLOWED_SPEC="67b10bc6-78b0-41a3-aaf4-358eeeb99009"',
        'ALLOWED_GPU_TYPE="NVIDIA_H200_SXM_141G"',
        'PROVENANCE_HELPER="$PROJECT_ROOT/scripts/batch42_scorer_provenance.py"',
    ):
        if needle not in score_text:
            raise ContractError(f"004109/004091 provenance gate is absent: {needle}")


def import_validator(path: Path):
    specification = importlib.util.spec_from_file_location(
        "batch44_postfinal_strict_004107", path
    )
    if specification is None or specification.loader is None:
        raise ContractError(f"cannot import 004107 validator: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    validator = getattr(module, "validate_final_selection_provenance", None)
    if not callable(validator):
        raise ContractError("004107 lost validate_final_selection_provenance")
    return validator


def import_scorer_provenance(path: Path):
    specification = importlib.util.spec_from_file_location(
        "batch44_postfinal_scorer_provenance", path
    )
    if specification is None or specification.loader is None:
        raise ContractError(f"cannot import scorer provenance helper: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    for name in (
        "LEDGER_FIELDS",
        "ProvenanceError",
        "validate_input_provenance",
        "validate_submission_contract",
        "verify_final_bundle",
        "sha256_file",
    ):
        if not hasattr(module, name):
            raise ContractError(f"scorer provenance helper lost {name}")
    return module


def audit_final_scalars(config: Config, final: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": FINAL_SCHEMA,
        "experiment_id": "batch44_v1",
        "data_version": "v1_20260709",
        "status": "final",
        "system_id": "path_x_final",
    }
    bad = {key: final.get(key) for key, value in expected.items() if final.get(key) != value}
    if bad:
        raise ContractError(f"invalid canonical FINAL_SELECTION scalars: {bad}")
    candidate = final.get("candidate")
    if not isinstance(candidate, dict):
        raise ContractError("FINAL_SELECTION candidate is missing")
    arm, step = candidate.get("arm"), candidate.get("step")
    if arm not in EXPECTED_RUN_DIRS or step not in EXPECTED_STEPS:
        raise ContractError(f"FINAL_SELECTION is outside registered Batch-44 Best3: {arm}/{step}")
    expected_checkpoint = (
        config.project_root
        / "outputs/lora_runs"
        / EXPECTED_RUN_DIRS[str(arm)]
        / f"step-{step}"
    ).resolve()
    expected_candidate = {
        "candidate_id": f"{arm}_step-{step}",
        "text_repeat": EXPECTED_REPEATS[str(arm)],
        "train_job_id": EXPECTED_TRAIN_JOBS[str(arm)],
    }
    for key, value in expected_candidate.items():
        if candidate.get(key) != value:
            raise ContractError(f"FINAL_SELECTION candidate {key} drift")
    if Path(str(candidate.get("checkpoint_path") or "")).resolve() != expected_checkpoint:
        raise ContractError("FINAL_SELECTION checkpoint path drift")
    model_files = candidate.get("model_files")
    if not isinstance(model_files, dict) or set(model_files) != MODEL_FILES:
        raise ContractError("FINAL_SELECTION model file registration is incomplete")


def validate_final_selection(config: Config, *, force_strict: bool) -> dict[str, Any]:
    if not config.final_selection.is_file():
        raise Pending(
            "WAITING_FINAL_SELECTION",
            f"canonical FINAL_SELECTION.json is not ready: {config.final_selection}",
        )
    cache_path = config.state_root / "final_selection_audit.json"
    final_sha = sha256_file(config.final_selection)
    validator_sha = sha256_file(config.final_validator)
    if not force_strict and cache_path.is_file():
        cached = load_json(cache_path)
        if (
            cached.get("status") == "verified"
            and cached.get("final_selection") == str(config.final_selection)
            and cached.get("final_selection_sha256") == final_sha
            and cached.get("validator") == str(config.final_validator)
            and cached.get("validator_sha256") == validator_sha
        ):
            final = load_json(config.final_selection)
            audit_final_scalars(config, final)
            return final

    validator = import_validator(config.final_validator)
    try:
        final = validator(
            config.final_selection,
            project_root=config.project_root,
            verify_checkpoint_hashes=True,
        )
    except Exception as exc:  # 004107 deliberately raises several exception types.
        raise ContractError(f"004107 strict FINAL_SELECTION validation failed: {exc}") from exc
    if not isinstance(final, dict):
        raise ContractError("004107 strict validator did not return a JSON object")
    audit_final_scalars(config, final)
    candidate = final["candidate"]
    atomic_write_json(
        cache_path,
        {
            "schema_version": "moss_codecvc.batch44_postfinal_final_audit.v1",
            "status": "verified",
            "verified_at_utc": utc_now(),
            "validator_method": "004107.validate_final_selection_provenance",
            "validator": str(config.final_validator),
            "validator_sha256": validator_sha,
            "final_selection": str(config.final_selection),
            "final_selection_sha256": final_sha,
            "candidate_id": candidate["candidate_id"],
            "arm": candidate["arm"],
            "step": candidate["step"],
            "checkpoint_path": candidate["checkpoint_path"],
        },
    )
    return final


def read_single_ledger(path: Path) -> tuple[list[str], dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise Pending("MISSING_LEDGER", f"submission ledger is absent: {path}")
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            rows = list(reader)
            fields = list(reader.fieldnames or [])
    except OSError as exc:
        raise ContractError(f"cannot read ledger {path}: {exc}") from exc
    if len(rows) != 1:
        raise ContractError(f"exactly one QZ submission is required in {path}; got {len(rows)}")
    return fields, rows[0]


def audit_materialization(config: Config, final: Mapping[str, Any]) -> None:
    path = config.plan_root / "materialization.json"
    payload = load_json(path)
    expected = {
        "schema_version": "moss_codecvc.batch42_pathx_final_batch44_v1_materialization.v1",
        "status": "ready",
        "system_id": "path_x_final",
        "experiment_id": "batch44_v1",
        "data_version": "v1_20260709",
        "final_selection": str(config.final_selection),
        "final_selection_sha256": sha256_file(config.final_selection),
        "checkpoint": str(Path(str(final["candidate"]["checkpoint_path"])).resolve()),
        "frozen_eval_code_root": str(config.expected_eval_code_root),
    }
    bad = {key: payload.get(key) for key, value in expected.items() if payload.get(key) != value}
    if bad:
        raise ContractError(f"004108 materialization provenance drift: {bad}")
    resource = payload.get("resource_contract")
    if resource != {
        "compute_group": ALLOWED_COMPUTE_GROUP_NAME,
        "compute_group_id": ALLOWED_COMPUTE_GROUP_ID,
        "spec": ALLOWED_SPEC,
        "gpu_type": ALLOWED_GPU_TYPE,
        "instances": ALLOWED_INSTANCES,
        "gpus": ALLOWED_GPUS,
    }:
        raise ContractError("004108 materialization is not MTTS-3-2-0715 / 1x8 H200")
    materialized = Path(str(payload.get("materialized_004094") or "")).resolve()
    if not materialized.is_file() or sha256_file(materialized) != payload.get(
        "materialized_004094_sha256"
    ):
        raise ContractError("004108 materialized wrapper path/SHA drift")


def audit_inference_ledger(
    config: Config, *, stage: str, final: Mapping[str, Any]
) -> dict[str, str]:
    if stage == "smoke":
        record, output, mode = config.smoke_record, config.smoke_output, "smoke"
        expected_name = f"batch42_pathx_strict_smoke_{config.smoke_gate_tag}"
    elif stage == "full":
        record, output, mode = config.full_record, config.full_output, "full"
        expected_name = f"batch42_pathx_strict_path_x_final_{config.run_tag}"
    else:  # pragma: no cover - internal programming error
        raise AssertionError(stage)
    ledger = record / "submitted_jobs.tsv"
    fields, row = read_single_ledger(ledger)
    expected_fields = [
        "job_name",
        "job_id",
        "mode",
        "system",
        "compute_group",
        "spec",
        "instances",
        "gpu_type",
        "output_root",
        "record_root",
        "entrypoint",
    ]
    if fields != expected_fields:
        raise ContractError(f"{stage} ledger header drift: {fields}")
    expected = {
        "job_name": expected_name,
        "mode": mode,
        "system": "path_x_final",
        "compute_group": ALLOWED_COMPUTE_GROUP_ID,
        "spec": ALLOWED_SPEC,
        "instances": "1",
        "gpu_type": ALLOWED_GPU_TYPE,
        "output_root": str(output),
        "record_root": str(record),
    }
    bad = {key: row.get(key) for key, value in expected.items() if row.get(key) != value}
    if bad:
        raise ContractError(f"{stage} MTTS/spec/path ledger provenance drift: {bad}")
    if not JOB_RE.fullmatch(row.get("job_id", "")):
        raise ContractError(f"{stage} ledger has invalid QZ job id")
    entrypoint = ensure_relative(Path(row.get("entrypoint", "")), record, f"{stage} entrypoint")
    if not entrypoint.is_file():
        raise ContractError(f"{stage} frozen entrypoint is missing: {entrypoint}")
    audit_materialization(config, final)
    atomic_write_json(
        config.state_root / f"{stage}_ledger_audit.json",
        {
            "schema_version": "moss_codecvc.batch44_postfinal_ledger_audit.v1",
            "status": "verified",
            "stage": stage,
            "verified_at_utc": utc_now(),
            "ledger": str(ledger),
            "ledger_sha256": sha256_file(ledger),
            "job_id": row["job_id"],
            "resource_contract": RESOURCE_CONTRACT,
            "spec": ALLOWED_SPEC,
            "entrypoint": str(entrypoint),
        },
    )
    return row


def audit_scorer_ledger(config: Config) -> dict[str, str]:
    ledger = config.scorer_record / "submitted_jobs.tsv"
    fields, row = read_single_ledger(ledger)
    provenance = import_scorer_provenance(config.scorer_provenance_helper)
    if fields != list(provenance.LEDGER_FIELDS):
        raise ContractError(f"scorer ledger header drift: {fields}")
    en_input = config.full_output / "en/successful.jsonl"
    zh_input = config.full_output / "zh/successful.jsonl"
    input_provenance = config.scorer_record / "input_provenance.json"
    submission_contract = config.scorer_record / "submission_contract.json"
    submit_output = config.scorer_record / "submit_output.txt"
    expected = {
        "job_name": f"batch42_score_path_x_final_{config.run_tag}",
        "system_tag": "path_x_final",
        "compute_group": ALLOWED_COMPUTE_GROUP_ID,
        "compute_group_name": ALLOWED_COMPUTE_GROUP_NAME,
        "spec": ALLOWED_SPEC,
        "instances": "1",
        "gpu_type": ALLOWED_GPU_TYPE,
        "gpus": "8",
        "en_input": str(en_input),
        "en_input_sha256": sha256_file(en_input),
        "zh_input": str(zh_input),
        "zh_input_sha256": sha256_file(zh_input),
        "source_inference_completion": str(config.full_marker),
        "source_inference_completion_sha256": sha256_file(config.full_marker),
        "source_final_selection": str(config.final_selection),
        "source_final_selection_sha256": sha256_file(config.final_selection),
        "output_root": str(config.scorer_output),
        "snapshot_root": str(config.scorer_record / "record_snapshot"),
        "input_provenance": str(input_provenance),
        "input_provenance_sha256": sha256_file(input_provenance),
        "submission_contract": str(submission_contract),
        "submission_contract_sha256": sha256_file(submission_contract),
        "submit_output": str(submit_output),
        "submit_output_sha256": sha256_file(submit_output),
    }
    bad = {key: row.get(key) for key, value in expected.items() if row.get(key) != value}
    if bad:
        raise ContractError(f"scorer MTTS/spec/input ledger provenance drift: {bad}")
    if not JOB_RE.fullmatch(row.get("job_id", "")):
        raise ContractError("scorer ledger has invalid QZ job id")
    try:
        input_payload = provenance.validate_input_provenance(
            input_provenance,
            expected_system_id="path_x_final",
            expected_en_input=en_input,
            expected_zh_input=zh_input,
            expected_output_root=config.scorer_output,
            expected_inference_completion=config.full_marker,
            expected_final_selection=config.final_selection,
        )
        submission = provenance.validate_submission_contract(
            submission_contract,
            expected_input_provenance=input_provenance,
            expected_system_id="path_x_final",
            expected_output_root=config.scorer_output,
            expected_record_root=config.scorer_record,
            expected_snapshot_root=config.scorer_record / "record_snapshot",
        )
    except provenance.ProvenanceError as exc:
        raise ContractError(f"scorer submit provenance contract failed: {exc}") from exc
    if input_payload.get("resource_contract") != SCORER_RESOURCE_CONTRACT:
        raise ContractError("scorer input provenance is not MTTS/spec/1x8H200")
    if submission.get("job_id") != row["job_id"]:
        raise ContractError("scorer submission contract/ledger QZ job-id drift")
    frozen = Path(row["snapshot_root"]) / "scripts/004091_submit_batch42_unified_scorers_qz.sh"
    if not frozen.is_file():
        raise ContractError(f"scorer frozen entrypoint is missing: {frozen}")
    frozen_helper = Path(row["snapshot_root"]) / "scripts/batch42_scorer_provenance.py"
    if not frozen_helper.is_file():
        raise ContractError(f"scorer frozen provenance helper is missing: {frozen_helper}")
    atomic_write_json(
        config.state_root / "score_ledger_audit.json",
        {
            "schema_version": "moss_codecvc.batch44_postfinal_ledger_audit.v1",
            "status": "verified",
            "stage": "score",
            "verified_at_utc": utc_now(),
            "ledger": str(ledger),
            "ledger_sha256": sha256_file(ledger),
            "job_id": row["job_id"],
            "resource_contract": RESOURCE_CONTRACT,
            "spec": ALLOWED_SPEC,
            "entrypoint": str(frozen),
            "input_provenance": str(input_provenance),
            "input_provenance_sha256": sha256_file(input_provenance),
            "submission_contract": str(submission_contract),
            "submission_contract_sha256": sha256_file(submission_contract),
        },
    )
    return row


def extract_qz_json(text: str, expected_job: str) -> dict[str, Any]:
    # qzcli prints a Rich detail panel before the requested JSON.  Decode every
    # possible JSON object and select the unique API object for this job.
    decoder = json.JSONDecoder()
    matches: list[dict[str, Any]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("job_id") == expected_job:
            matches.append(payload)
    unique = {json.dumps(item, sort_keys=True): item for item in matches}
    if len(unique) != 1:
        raise ContractError(
            f"qzcli status output must contain one API JSON for {expected_job}; got {len(unique)}"
        )
    return next(iter(unique.values()))


def audit_qz_payload(
    payload: Mapping[str, Any], *, row: Mapping[str, str], command_needle: str
) -> str:
    expected = {
        "job_id": row["job_id"],
        "name": row["job_name"],
        "workspace_id": ALLOWED_WORKSPACE,
        "project_id": ALLOWED_PROJECT,
        "logic_compute_group_id": ALLOWED_COMPUTE_GROUP_ID,
        "logic_compute_group_name": ALLOWED_COMPUTE_GROUP_NAME,
    }
    bad = {key: payload.get(key) for key, value in expected.items() if payload.get(key) != value}
    if bad:
        raise ContractError(f"QZ job identity/workspace/compute-group drift: {bad}")
    framework = payload.get("framework_config")
    if not isinstance(framework, list) or len(framework) != 1:
        raise ContractError("QZ job must have exactly one framework_config entry")
    item = framework[0]
    if not isinstance(item, dict):
        raise ContractError("QZ framework_config entry must be an object")
    info = item.get("instance_spec_price_info") or {}
    gpu = info.get("gpu_info") or {}
    if (
        item.get("instance_count") != ALLOWED_INSTANCES
        or item.get("gpu_count") != ALLOWED_GPUS
        or info.get("quota_id") != ALLOWED_SPEC
        or gpu.get("gpu_type") != ALLOWED_GPU_TYPE
    ):
        raise ContractError("QZ job is not registered spec / one 8xH200 instance")
    command = str(payload.get("command") or "")
    if command_needle not in command:
        raise ContractError(f"QZ command is not bound to frozen entrypoint: {command_needle}")
    status = str(payload.get("status") or "").strip().lower()
    accepted = {
        "job_pending",
        "job_queued",
        "job_running",
        "job_succeeded",
        "job_failed",
        "job_stopped",
    }
    if status not in accepted:
        raise ContractError(f"unrecognized QZ status (fail closed): {status!r}")
    return status


def query_qz_status(
    config: Config, *, stage: str, row: Mapping[str, str], command_needle: str
) -> str:
    if config.action != "submit":
        raise Pending(
            f"{stage.upper()}_NEEDS_QZ_AUDIT",
            f"{stage} ledger exists; ACTION=plan never calls qzcli",
        )
    if not config.qzcli.is_file() or not os.access(config.qzcli, os.X_OK):
        raise ContractError(f"qzcli wrapper is not executable: {config.qzcli}")
    environment = os.environ.copy()
    for key in (
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        environment.pop(key, None)
    environment["HOME"] = str(config.qzcli_home)
    result = subprocess.run(
        [str(config.qzcli), "status", row["job_id"], "--json"],
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    raw = result.stdout + ("\n" + result.stderr if result.stderr else "")
    atomic_write_text(config.state_root / f"qz_status_{stage}_latest.txt", raw)
    if result.returncode != 0:
        raise ContractError(
            f"qzcli status failed for {stage}/{row['job_id']} rc={result.returncode}; no retry"
        )
    payload = extract_qz_json(raw, row["job_id"])
    status = audit_qz_payload(payload, row=row, command_needle=command_needle)
    audit = {
        "schema_version": "moss_codecvc.batch44_postfinal_qz_status_audit.v1",
        "status": status,
        "stage": stage,
        "queried_at_utc": utc_now(),
        "job_id": row["job_id"],
        "job_name": row["job_name"],
        "workspace_id": payload["workspace_id"],
        "project_id": payload["project_id"],
        "compute_group_id": payload["logic_compute_group_id"],
        "compute_group_name": payload["logic_compute_group_name"],
        "spec": ALLOWED_SPEC,
        "instances": ALLOWED_INSTANCES,
        "gpus": ALLOWED_GPUS,
        "gpu_type": ALLOWED_GPU_TYPE,
        "command": payload.get("command"),
    }
    atomic_write_json(config.state_root / f"qz_status_{stage}_latest.json", audit)
    history = config.state_root / "qz_status_history.jsonl"
    with history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(audit, ensure_ascii=False, sort_keys=True) + "\n")
    return status


def audit_registered_identity(
    config: Config, identity: Mapping[str, Any], final: Mapping[str, Any], label: str
) -> None:
    if (
        identity.get("schema_version") != "moss_codecvc.batch42_pathx_registered_identity.v1"
        or identity.get("status") != "verified"
    ):
        raise ContractError(f"{label}: registered identity schema/status drift")
    candidate = final["candidate"]
    if Path(str(identity.get("model_path") or "")).resolve() != Path(
        candidate["checkpoint_path"]
    ).resolve():
        raise ContractError(f"{label}: checkpoint differs from FINAL_SELECTION")
    if Path(str(identity.get("code_root") or "")).resolve() != config.expected_eval_code_root:
        raise ContractError(f"{label}: frozen eval code root drift")
    actual_files = identity.get("model_files")
    if not isinstance(actual_files, dict) or set(actual_files) != MODEL_FILES:
        raise ContractError(f"{label}: registered model file set drift")
    for name, registration in candidate["model_files"].items():
        actual = actual_files.get(name) or {}
        if actual.get("size") != registration.get("size") or actual.get("sha256") != registration.get(
            "sha256"
        ):
            raise ContractError(f"{label}: model identity drift for {name}")


def validate_wav(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 1024:
        raise ContractError(f"smoke generated WAV is missing/empty: {path}")
    try:
        with wave.open(str(path), "rb") as handle:
            valid = (
                handle.getnframes() > 0
                and handle.getframerate() > 0
                and handle.getnchannels() > 0
            )
    except (wave.Error, OSError) as exc:
        raise ContractError(f"smoke generated WAV is not decodable: {path}: {exc}") from exc
    if not valid:
        raise ContractError(f"smoke generated WAV metadata is invalid: {path}")


def audit_smoke_marker(config: Config, final: Mapping[str, Any]) -> dict[str, Any]:
    marker = config.smoke_marker
    payload = load_json(marker)
    if (
        payload.get("schema_version") != SMOKE_SCHEMA
        or payload.get("status") != "smoke_complete"
        or payload.get("system_id") != "path_x_final"
        or payload.get("resource_contract") != RESOURCE_CONTRACT
    ):
        raise ContractError("SMOKE_COMPLETED schema/system/resource contract drift")
    audit_registered_identity(config, payload.get("registered_identity") or {}, final, "smoke")
    strict = payload.get("strict_inputs") or {}
    if strict.get("schema_version") != "moss_codecvc.batch42_pathx_strict_input_audit.v1":
        raise ContractError("smoke strict-input audit schema drift")
    for language, expected in (("en", config.en_expected), ("zh", config.zh_expected)):
        item = strict.get(language) or {}
        if item.get("rows") != expected or item.get("sha256") != config.strict_manifest_sha256[language]:
            raise ContractError(f"smoke {language} strict manifest rows/SHA drift")
    contract = payload.get("protocol_contract")
    if not isinstance(contract, dict) or contract.get("system_id") != "path_x_final":
        raise ContractError("smoke protocol contract is not path_x_final")
    if contract.get("strict_manifest_sha256") != dict(config.strict_manifest_sha256):
        raise ContractError("smoke protocol strict-manifest fingerprint drift")
    fingerprint = hashlib.sha256(
        json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if payload.get("protocol_fingerprint_sha256") != fingerprint:
        raise ContractError("smoke protocol fingerprint is not reproducible")
    actual = payload.get("actual_one_case") or {}
    generated = ensure_relative(
        Path(str(actual.get("generated_audio") or "")), config.smoke_output, "smoke WAV"
    )
    validate_wav(generated)
    inference = actual.get("inference_config") or {}
    if (
        Path(str(inference.get("model_path") or "")).resolve()
        != Path(final["candidate"]["checkpoint_path"]).resolve()
        or Path(str(inference.get("code_root") or "")).resolve()
        != config.expected_eval_code_root
        or inference.get("final_selection_sha256") != sha256_file(config.final_selection)
        or assert_finite(inference.get("ref_audio_cfg_scale"), "smoke ref_audio_cfg_scale")
        != 1.0
    ):
        raise ContractError("smoke inference provenance differs from FINAL_SELECTION")
    atomic_write_json(
        config.state_root / "smoke_artifact_audit.json",
        {
            "schema_version": "moss_codecvc.batch44_postfinal_artifact_audit.v1",
            "status": "verified",
            "stage": "smoke",
            "verified_at_utc": utc_now(),
            "marker": str(marker),
            "marker_sha256": sha256_file(marker),
            "protocol_fingerprint_sha256": fingerprint,
            "generated_audio": str(generated),
            "generated_audio_sha256": sha256_file(generated),
        },
    )
    return payload


def audit_successful_jsonl(
    path: Path,
    *,
    expected: int,
    final: Mapping[str, Any],
    config: Config,
    label: str,
) -> None:
    seen: set[str] = set()
    count = 0
    final_sha = sha256_file(config.final_selection)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            count += 1
            row = json.loads(line)
            case_id = str(row.get("case_id") or "")
            if not case_id or case_id in seen:
                raise ContractError(f"{label}: missing/duplicate case_id")
            seen.add(case_id)
            if row.get("system_id") != "path_x_final" or row.get("status") not in {
                "ok",
                "skipped_existing",
            }:
                raise ContractError(f"{label}/{case_id}: wrong system/status")
            inference = ((row.get("provenance") or {}).get("inference_config") or {})
            if (
                Path(str(inference.get("model_path") or "")).resolve()
                != Path(final["candidate"]["checkpoint_path"]).resolve()
                or Path(str(inference.get("code_root") or "")).resolve()
                != config.expected_eval_code_root
                or inference.get("final_selection_sha256") != final_sha
                or assert_finite(
                    inference.get("ref_audio_cfg_scale"), f"{label}/{case_id} ref CFG"
                )
                != 1.0
            ):
                raise ContractError(f"{label}/{case_id}: FINAL_SELECTION provenance drift")
    if count != expected or len(seen) != expected:
        raise ContractError(f"{label}: rows/unique={count}/{len(seen)}, expected {expected}")


def audit_full_marker(
    config: Config, final: Mapping[str, Any], smoke: Mapping[str, Any]
) -> dict[str, Any]:
    marker = config.full_marker
    payload = load_json(marker)
    if (
        payload.get("schema_version") != FULL_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("system_id") != "path_x_final"
        or payload.get("resource_contract") != RESOURCE_CONTRACT
        or Path(str(payload.get("output_root") or "")).resolve() != config.full_output
        or Path(str(payload.get("record_root") or "")).resolve() != config.full_record
    ):
        raise ContractError("full COMPLETED schema/system/resource/path contract drift")
    audit_registered_identity(config, payload.get("registered_identity") or {}, final, "full")
    smoke_gate = payload.get("smoke_gate") or {}
    if (
        Path(str(smoke_gate.get("marker") or "")).resolve() != config.smoke_marker
        or smoke_gate.get("protocol_fingerprint_sha256")
        != smoke.get("protocol_fingerprint_sha256")
    ):
        raise ContractError("full completion is not bound to the verified smoke gate")
    strict_sets = payload.get("strict_sets") or {}
    for language, expected in (("en", config.en_expected), ("zh", config.zh_expected)):
        item = strict_sets.get(language) or {}
        successful = config.full_output / language / "successful.jsonl"
        if (
            item.get("registered_cases") != expected
            or Path(str(item.get("successful_jsonl") or "")).resolve() != successful
        ):
            raise ContractError(f"full {language} denominator/successful path drift")
        schema_path = ensure_relative(
            Path(str(item.get("schema_jsonl") or "")),
            config.full_output / language,
            f"full {language} schema",
        )
        merge_path = ensure_relative(
            Path(str(item.get("merge_summary") or "")),
            config.full_output / language,
            f"full {language} merge",
        )
        merge = load_json(merge_path)
        if merge.get("all_ok") is not True or merge.get("rows") != expected:
            raise ContractError(f"full {language} merge is not all-ok/{expected}")
        if count_nonempty_lines(schema_path) != expected:
            raise ContractError(f"full {language} schema denominator drift")
        audit_successful_jsonl(
            successful,
            expected=expected,
            final=final,
            config=config,
            label=f"full/{language}",
        )
    record_copy = config.full_record / "completion.json"
    if not record_copy.is_file() or record_copy.read_bytes() != marker.read_bytes():
        raise ContractError("full completion record copy is missing or differs")
    atomic_write_json(
        config.state_root / "full_artifact_audit.json",
        {
            "schema_version": "moss_codecvc.batch44_postfinal_artifact_audit.v1",
            "status": "verified",
            "stage": "full",
            "verified_at_utc": utc_now(),
            "marker": str(marker),
            "marker_sha256": sha256_file(marker),
            "en_cases": config.en_expected,
            "zh_cases": config.zh_expected,
            "smoke_marker": str(config.smoke_marker),
            "smoke_marker_sha256": sha256_file(config.smoke_marker),
        },
    )
    return payload


def audit_metric_block(block: Mapping[str, Any], expected: int, label: str) -> None:
    speakers = block.get("speaker_similarity") or {}
    if set(speakers) != {"wavlm_large_sv", "eres2net", "speechbrain_ecapa"}:
        raise ContractError(f"{label}: three-speaker-scorer set drift")
    for backend, metric in speakers.items():
        if (metric.get("status_counts") or {}).get("ok") != expected:
            raise ContractError(f"{label}/{backend}: scorer status denominator drift")
        for side in ("sim_ref", "sim_src"):
            item = metric.get(side) or {}
            if item.get("n") != expected:
                raise ContractError(f"{label}/{backend}/{side}: denominator drift")
            assert_finite(item.get("mean"), f"{label}/{backend}/{side}.mean")


def audit_scorer_artifacts(config: Config) -> dict[str, Any]:
    provenance = import_scorer_provenance(config.scorer_provenance_helper)
    try:
        attestation = provenance.verify_final_bundle(
            completion_path=config.scorer_marker,
            ledger_path=config.scorer_record / "submitted_jobs.tsv",
            expected_output_root=config.scorer_output,
            expected_en_input=config.full_output / "en/successful.jsonl",
            expected_zh_input=config.full_output / "zh/successful.jsonl",
            expected_inference_completion=config.full_marker,
            expected_final_selection=config.final_selection,
        )
    except provenance.ProvenanceError as exc:
        raise ContractError(f"unified scorer provenance bundle failed: {exc}") from exc
    if (
        not isinstance(attestation, dict)
        or attestation.get("status") != "verified"
        or attestation.get("resource_contract") != SCORER_RESOURCE_CONTRACT
    ):
        raise ContractError("unified scorer helper returned an invalid attestation")
    completion = load_json(config.scorer_marker)
    required_completion = {
        "schema_version": SCORER_COMPLETION_SCHEMA,
        "system_id": "path_x_final",
        "status": "complete",
        "en_cases": config.en_expected,
        "zh_cases": config.zh_expected,
        "output_root": str(config.scorer_output),
        "job_id": attestation.get("job_id"),
        "resource_contract": SCORER_RESOURCE_CONTRACT,
    }
    bad = {
        key: completion.get(key)
        for key, value in required_completion.items()
        if completion.get(key) != value
    }
    if bad or not completion.get("completed_at_utc"):
        raise ContractError(f"unified scorer completion contract drift: {bad}")
    combined_path = config.scorer_output / "path_x_final.en_zh.summary.json"
    combined = load_json(combined_path)
    if (
        combined.get("schema_version") != SCORER_SUMMARY_SCHEMA
        or combined.get("system_id") != "path_x_final"
    ):
        raise ContractError("combined scorer summary schema/system drift")
    summary_paths: dict[str, Path] = {}
    for language, expected, primary_asr, test_set in (
        ("en", config.en_expected, "whisper_large_v3", "seedtts-vc-en-internal320-disjoint"),
        ("zh", config.zh_expected, "paraformer_zh", "seedtts-vc-zh-internal320-disjoint"),
    ):
        root = config.scorer_output / language / "merged"
        summary_path = root / f"path_x_final.{language}.merged.summary.json"
        audit_path = root / f"path_x_final.{language}.merged.strict_audit.json"
        summary = load_json(summary_path)
        audit = load_json(audit_path)
        if (
            summary.get("schema_version") != UNIFIED_EVAL_SCHEMA
            or summary.get("record_type") != "vc_eval_summary"
        ):
            raise ContractError(f"{language} scorer summary schema/type drift")
        group = (summary.get("groups") or {}).get("all") or {}
        if group.get("n_cases") != expected:
            raise ContractError(f"{language} scorer summary denominator drift")
        audit_metric_block(group, expected, f"scorer/{language}")
        asr = (group.get("content_asr") or {}).get(primary_asr) or {}
        if (asr.get("status_counts") or {}).get("ok") != expected:
            raise ContractError(f"{language}/{primary_asr}: ASR status denominator drift")
        error = asr.get("primary_error") or {}
        if error.get("n") != expected:
            raise ContractError(f"{language}/{primary_asr}: ASR denominator drift")
        assert_finite(error.get("mean"), f"{language}/{primary_asr}.mean")
        if (
            audit.get("schema_version") != STRICT_SCORER_AUDIT_SCHEMA
            or audit.get("system_id") != "path_x_final"
            or audit.get("language") != language
            or audit.get("test_set_id") != test_set
            or audit.get("rows") != expected
            or audit.get("unique_case_ids") != expected
            or audit.get("all_ok") is not True
        ):
            raise ContractError(f"{language} strict scorer audit drift")
        for backend in ("wavlm_large_sv", "eres2net", "speechbrain_ecapa"):
            if ((audit.get("speaker_status_counts") or {}).get(backend) or {}).get("ok") != expected:
                raise ContractError(f"{language}/{backend}: strict scorer status drift")
        if ((audit.get("asr_status_counts") or {}).get(primary_asr) or {}).get("ok") != expected:
            raise ContractError(f"{language}/{primary_asr}: strict ASR status drift")
        merged = ensure_relative(
            Path(str(audit.get("merged_jsonl") or "")), root, f"{language} merged scorer JSONL"
        )
        if count_nonempty_lines(merged) != expected:
            raise ContractError(f"{language} merged scorer JSONL denominator drift")
        combined_item = combined.get(language) or {}
        if (
            Path(str(combined_item.get("summary_path") or "")).resolve() != summary_path
            or Path(str(combined_item.get("strict_audit_path") or "")).resolve() != audit_path
            or combined_item.get("strict_audit") != audit
            or combined_item.get("group_all") != group
        ):
            raise ContractError(f"combined scorer summary {language} provenance drift")
        summary_paths[language] = summary_path
    atomic_write_json(
        config.state_root / "score_artifact_audit.json",
        {
            "schema_version": "moss_codecvc.batch44_postfinal_artifact_audit.v1",
            "status": "verified",
            "stage": "score",
            "verified_at_utc": utc_now(),
            "completion": str(config.scorer_marker),
            "completion_sha256": sha256_file(config.scorer_marker),
            "combined_summary": str(combined_path),
            "combined_summary_sha256": sha256_file(combined_path),
            "en_summary": str(summary_paths["en"]),
            "en_summary_sha256": sha256_file(summary_paths["en"]),
            "zh_summary": str(summary_paths["zh"]),
            "zh_summary_sha256": sha256_file(summary_paths["zh"]),
            "scorer_attestation": attestation,
        },
    )
    return completion


def audit_final_table(config: Config) -> dict[str, Any]:
    provenance = import_scorer_provenance(config.scorer_provenance_helper)
    payload = load_json(config.table_json)
    if payload.get("status") != "complete" or payload.get("counts") != {
        "systems": 8,
        "complete": 8,
        "partial": 0,
        "pending": 0,
    }:
        raise ContractError("Batch-42 final table is not complete 8/8")
    rows = payload.get("systems")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ContractError("Batch-42 final table must contain exactly eight systems")
    indexed = {str(row.get("system_id") or ""): row for row in rows if isinstance(row, dict)}
    if len(indexed) != 8 or "path_x_final" not in indexed:
        raise ContractError("Batch-42 final table system IDs are missing/duplicated")
    final = indexed["path_x_final"]
    if final.get("status") != "complete":
        raise ContractError("Batch-42 path_x_final table row is not complete")
    for language, expected in (("en", config.en_expected), ("zh", config.zh_expected)):
        item = ((final.get("metrics") or {}).get(language) or {})
        if item.get("status") != "complete" or item.get("n_cases") != expected:
            raise ContractError(f"Batch-42 path_x_final {language} denominator/status drift")
    interim = load_json(config.interim_table)
    if interim.get("status") != "interim" or interim.get("counts") != {
        "systems": 8,
        "complete": 7,
        "partial": 0,
        "pending": 1,
    }:
        raise ContractError("frozen Batch-42 interim table is not trustworthy 7/8")
    old = {row["system_id"]: row for row in interim.get("systems", [])}
    if (old.get("path_x_final") or {}).get("status") != "pending":
        raise ContractError("frozen interim path_x_final row is not pending")
    for system_id, old_row in old.items():
        if system_id == "path_x_final":
            continue
        new_row = indexed.get(system_id) or {}
        if new_row.get("status") != old_row.get("status") or new_row.get("metrics") != old_row.get(
            "metrics"
        ):
            raise ContractError(f"published table changed frozen baseline row: {system_id}")
    publication = load_json(config.table_provenance)
    if (
        publication.get("schema_version")
        != "moss_codecvc.batch42_pathx_final_table_publication.v1"
        or publication.get("status") != "complete"
        or publication.get("system_id") != "path_x_final"
        or publication.get("counts")
        != {"systems": 8, "complete": 8, "partial": 0, "pending": 0}
    ):
        raise ContractError("Batch-42 table publication marker schema/status drift")
    scorer_attestation = publication.get("scorer_attestation") or {}
    if (
        scorer_attestation.get("status") != "verified"
        or scorer_attestation.get("resource_contract") != SCORER_RESOURCE_CONTRACT
        or scorer_attestation.get("job_id")
        != load_json(config.scorer_marker).get("job_id")
    ):
        raise ContractError("Batch-42 table publication scorer attestation drift")
    frozen_ref = publication.get("frozen_interim") or {}
    if (
        Path(str(frozen_ref.get("path") or "")).resolve() != config.interim_table
        or frozen_ref.get("size") != config.interim_table.stat().st_size
        or frozen_ref.get("sha256") != sha256_file(config.interim_table)
    ):
        raise ContractError("Batch-42 table publication frozen-interim binding drift")
    expected_artifacts = {
        "markdown": config.table_md,
        "json": config.table_json,
        "main_tsv": config.table_tsv,
        "cross_validation_tsv": config.table_cross_validation_tsv,
    }
    refs = publication.get("table_artifacts") or {}
    for key, path in expected_artifacts.items():
        ref = refs.get(key) or {}
        if (
            Path(str(ref.get("path") or "")).resolve() != path
            or not path.is_file()
            or ref.get("size") != path.stat().st_size
            or ref.get("sha256") != sha256_file(path)
        ):
            raise ContractError(f"Batch-42 published table artifact drift: {key}")
    if publication.get("frozen_nonfinal_rows_sha256") != publication.get(
        "published_nonfinal_rows_sha256"
    ):
        raise ContractError("Batch-42 published non-final rows differ from frozen 7/8")
    return payload


def write_state(config: Config, state: str, detail: str, **extra: Any) -> None:
    payload = {
        "schema_version": "moss_codecvc.batch44_postfinal_batch42_state.v1",
        "updated_at_utc": utc_now(),
        "state": state,
        "detail": detail,
        "mode": config.mode,
        "action": config.action,
        **extra,
    }
    atomic_write_json(config.state_root / "scan_latest.json", payload)
    print(f"[batch44-postfinal] state={state} detail={detail}")


def write_halt(config: Config, stage: str, reason: str) -> None:
    path = config.state_root / "HALTED.json"
    if not path.exists():
        atomic_write_json(
            path,
            {
                "schema_version": "moss_codecvc.batch44_postfinal_halt.v1",
                "status": "halted",
                "halted_at_utc": utc_now(),
                "stage": stage,
                "reason": reason,
                "automatic_retry": False,
                "recovery": (
                    "inspect the QZ job, ledger, artifact, and watcher logs; recovery is an "
                    "explicit manual action and must not silently remove this marker"
                ),
            },
        )
    write_state(config, "HALTED", reason, stage=stage, halt_marker=str(path))


def attempt_path(config: Config, stage: str) -> Path:
    return config.state_root / f"attempt_{stage}.json"


def run_logged_command(
    config: Config,
    *,
    stage: str,
    command: Sequence[str],
    environment: Mapping[str, str],
) -> None:
    attempted = attempt_path(config, stage)
    if attempted.exists():
        raise ContractError(
            f"{stage} has a prior attempt marker but no accepted completion; no automatic retry"
        )
    atomic_write_json(
        attempted,
        {
            "schema_version": "moss_codecvc.batch44_postfinal_stage_attempt.v1",
            "status": "started",
            "stage": stage,
            "started_at_utc": utc_now(),
            "command": list(command),
            "final_selection": str(config.final_selection),
            "final_selection_sha256": sha256_file(config.final_selection),
        },
    )
    result = subprocess.run(
        list(command), text=True, capture_output=True, check=False, env=dict(environment)
    )
    log = (
        f"command={json.dumps(list(command), ensure_ascii=False)}\n"
        f"returncode={result.returncode}\n"
        "--- stdout ---\n"
        f"{result.stdout}\n"
        "--- stderr ---\n"
        f"{result.stderr}\n"
    )
    atomic_write_text(config.state_root / f"{stage}_command.log", log)
    if result.returncode != 0:
        raise ContractError(f"{stage} command failed rc={result.returncode}; no automatic retry")


def wrapper_environment(config: Config) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PROJECT_ROOT": str(config.project_root),
            "FINAL_SELECTION_JSON": str(config.final_selection),
            "PLAN_ROOT": str(config.plan_root),
            "RUN_TAG": config.run_tag,
            "SMOKE_GATE_TAG": config.smoke_gate_tag,
            "INFERENCE_ROOT": str(config.full_output),
            "INFERENCE_COMPLETION": str(config.full_marker),
            "SCORER_OUTPUT": str(config.scorer_output),
            "SCORER_RECORD": str(config.scorer_record),
            "TABLE_PREFIX": str(config.table_prefix),
            "INTERIM_TABLE_JSON": str(config.interim_table),
            "PYTHON": str(config.python),
        }
    )
    if config.test_mode:
        environment.update(
            {
                "SMOKE_OUTPUT_ROOT": str(config.smoke_output),
                "SMOKE_RECORD_ROOT": str(config.smoke_record),
                "FULL_RECORD_ROOT": str(config.full_record),
                "EXPECTED_EVAL_CODE_ROOT": str(config.expected_eval_code_root),
                "BATCH44_POSTFINAL_TEST_EN_EXPECTED": str(config.en_expected),
                "BATCH44_POSTFINAL_TEST_ZH_EXPECTED": str(config.zh_expected),
            }
        )
    return environment


def submit_remote_stage(
    config: Config, *, stage: str, final: Mapping[str, Any]
) -> None:
    # Re-run the full 004107 replay immediately before every irreversible QZ
    # submission.  This is deliberately stronger than the lightweight polling
    # cache used while a registered job is merely running.
    validate_final_selection(config, force_strict=True)
    environment = wrapper_environment(config)
    if stage in {"smoke", "full"}:
        environment.update(
            {
                "MODE": stage,
                "DRY_RUN": "0",
                "PLAN_ONLY": "0",
                "CONFIRM_BATCH44_FINAL_STRICT": "1",
            }
        )
        command = ["bash", str(config.inference_wrapper)]
    elif stage == "score":
        environment.update(
            {
                "STAGE": "score",
                "DRY_RUN": "0",
                "CONFIRM_BATCH44_FINAL_SCORERS": "1",
            }
        )
        command = ["bash", str(config.score_wrapper)]
    else:  # pragma: no cover
        raise AssertionError(stage)
    run_logged_command(config, stage=stage, command=command, environment=environment)
    # A wrapper returning zero without its unique ledger is an ambiguous remote
    # mutation.  Halt instead of trying the create-job call again.
    if stage in {"smoke", "full"}:
        audit_inference_ledger(config, stage=stage, final=final)
    else:
        audit_scorer_ledger(config)


def publish_table(config: Config) -> None:
    validate_final_selection(config, force_strict=True)
    environment = wrapper_environment(config)
    environment.update(
        {
            "STAGE": "table",
            "DRY_RUN": "1",
            "CONFIRM_BATCH44_FINAL_SCORERS": "0",
        }
    )
    run_logged_command(
        config,
        stage="table",
        command=["bash", str(config.score_wrapper)],
        environment=environment,
    )
    audit_final_table(config)


def remote_stage(
    config: Config,
    *,
    stage: str,
    final: Mapping[str, Any],
    marker: Path,
    downstream_present: bool,
) -> tuple[dict[str, str], str]:
    record = config.smoke_record if stage == "smoke" else config.full_record
    if stage == "score":
        record = config.scorer_record
    ledger = record / "submitted_jobs.tsv"
    if not ledger.is_file():
        if marker.exists():
            raise ContractError(f"{stage} completion exists without a QZ submission ledger")
        if downstream_present:
            raise ContractError(f"downstream artifacts exist before {stage} submission")
        if attempt_path(config, stage).exists():
            raise ContractError(
                f"{stage} has an attempt marker but no ledger; no automatic retry"
            )
        if config.action == "plan":
            raise Pending(
                f"READY_{stage.upper()}",
                f"{stage} is ready; ACTION=plan made no qzcli call",
            )
        submit_remote_stage(config, stage=stage, final=final)
        raise Pending(
            f"{stage.upper()}_SUBMITTED",
            f"{stage} submitted exactly once; waiting for QZ terminal success",
        )

    if stage in {"smoke", "full"}:
        row = audit_inference_ledger(config, stage=stage, final=final)
        command_needle = row["entrypoint"]
    else:
        row = audit_scorer_ledger(config)
        command_needle = str(
            Path(row["snapshot_root"]) / "scripts/004091_submit_batch42_unified_scorers_qz.sh"
        )
    status = query_qz_status(config, stage=stage, row=row, command_needle=command_needle)
    if status in {"job_pending", "job_queued", "job_running"}:
        if downstream_present:
            raise ContractError(f"downstream stage exists while {stage} QZ job is still {status}")
        raise Pending(
            f"WAITING_{stage.upper()}_QZ",
            f"{stage} {row['job_id']} is {status}",
        )
    if status in {"job_failed", "job_stopped"}:
        raise ContractError(f"{stage} QZ job {row['job_id']} ended as {status}; no retry")
    if status != "job_succeeded":  # pragma: no cover - audit_qz_payload fences this.
        raise ContractError(f"unexpected {stage} QZ status: {status}")
    if not marker.is_file() or marker.stat().st_size == 0:
        raise ContractError(
            f"{stage} QZ job succeeded but atomic completion artifact is missing: {marker}"
        )
    return row, status


def write_pipeline_complete(
    config: Config,
    *,
    final: Mapping[str, Any],
    smoke_row: Mapping[str, str],
    full_row: Mapping[str, str],
    score_row: Mapping[str, str],
) -> None:
    path = config.state_root / "PIPELINE_COMPLETE.json"
    payload = {
        "schema_version": "moss_codecvc.batch44_postfinal_batch42_complete.v1",
        "status": "complete",
        "completed_at_utc": utc_now(),
        "pipeline": "004107 verify -> 004108 smoke -> 004108 full -> 004109 score -> 004109 table",
        "final_selection": str(config.final_selection),
        "final_selection_sha256": sha256_file(config.final_selection),
        "candidate_id": final["candidate"]["candidate_id"],
        "jobs": {
            "smoke": smoke_row["job_id"],
            "full": full_row["job_id"],
            "score": score_row["job_id"],
        },
        "resource_contract": {
            **RESOURCE_CONTRACT,
            "compute_group_id": ALLOWED_COMPUTE_GROUP_ID,
            "spec": ALLOWED_SPEC,
        },
        "artifacts": {
            "smoke": {"path": str(config.smoke_marker), "sha256": sha256_file(config.smoke_marker)},
            "full": {"path": str(config.full_marker), "sha256": sha256_file(config.full_marker)},
            "score": {"path": str(config.scorer_marker), "sha256": sha256_file(config.scorer_marker)},
            "table_json": {"path": str(config.table_json), "sha256": sha256_file(config.table_json)},
            "table_md": {"path": str(config.table_md), "sha256": sha256_file(config.table_md)},
            "table_tsv": {"path": str(config.table_tsv), "sha256": sha256_file(config.table_tsv)},
            "table_cross_validation_tsv": {
                "path": str(config.table_cross_validation_tsv),
                "sha256": sha256_file(config.table_cross_validation_tsv),
            },
            "table_publication": {
                "path": str(config.table_provenance),
                "sha256": sha256_file(config.table_provenance),
            },
        },
    }
    if path.exists():
        old = load_json(path)
        for key in ("final_selection", "final_selection_sha256", "candidate_id", "jobs", "resource_contract", "artifacts"):
            if old.get(key) != payload.get(key):
                raise ContractError(f"existing PIPELINE_COMPLETE provenance drift: {key}")
        return
    atomic_write_json(path, payload)


def run_once(config: Config) -> bool:
    halt = config.state_root / "HALTED.json"
    if halt.is_file():
        payload = load_json(halt)
        raise ContractError(
            f"persistent HALTED marker requires manual audit: {payload.get('stage')}: {payload.get('reason')}"
        )
    final = validate_final_selection(config, force_strict=False)

    smoke_row, _ = remote_stage(
        config,
        stage="smoke",
        final=final,
        marker=config.smoke_marker,
        downstream_present=(config.full_record / "submitted_jobs.tsv").exists()
        or (config.scorer_record / "submitted_jobs.tsv").exists()
        or config.full_marker.exists()
        or config.scorer_marker.exists()
        or config.table_json.exists(),
    )
    smoke = audit_smoke_marker(config, final)

    full_row, _ = remote_stage(
        config,
        stage="full",
        final=final,
        marker=config.full_marker,
        downstream_present=(config.scorer_record / "submitted_jobs.tsv").exists()
        or config.scorer_marker.exists()
        or config.table_json.exists(),
    )
    audit_full_marker(config, final, smoke)

    score_row, _ = remote_stage(
        config,
        stage="score",
        final=final,
        marker=config.scorer_marker,
        downstream_present=config.table_json.exists(),
    )
    audit_scorer_artifacts(config)

    complete = config.state_root / "PIPELINE_COMPLETE.json"
    if complete.exists():
        audit_final_table(config)
        write_pipeline_complete(
            config,
            final=final,
            smoke_row=smoke_row,
            full_row=full_row,
            score_row=score_row,
        )
        write_state(config, "COMPLETE", "verified existing Batch-42 final table 8/8")
        return True
    if attempt_path(config, "table").exists():
        raise ContractError("table has a prior attempt without PIPELINE_COMPLETE; no automatic retry")
    if config.action == "plan":
        raise Pending("READY_TABLE", "scorers are complete; ACTION=plan did not publish the table")
    publish_table(config)
    write_pipeline_complete(
        config,
        final=final,
        smoke_row=smoke_row,
        full_row=full_row,
        score_row=score_row,
    )
    write_state(config, "COMPLETE", "Batch-42 final table published and verified complete=8/8")
    return True


def acquire_monitor_lock(config: Config) -> Path | None:
    if config.mode != "monitor":
        return None
    lock = config.state_root / ".monitor.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise ContractError(f"another post-final watcher holds {lock}") from exc
    atomic_write_json(
        lock / "owner.json",
        {"pid": os.getpid(), "started_at_utc": utc_now(), "action": config.action},
    )
    return lock


def run(config: Config) -> int:
    config.state_root.mkdir(parents=True, exist_ok=True)
    validate_static_config(config)
    lock = acquire_monitor_lock(config)
    scans = 0
    try:
        while True:
            scans += 1
            try:
                complete = run_once(config)
            except Pending as pending:
                write_state(config, pending.state, pending.detail, scan=scans)
                complete = False
            except ContractError as exc:
                # A pre-existing HALTED marker is reported, not overwritten.
                if not (config.state_root / "HALTED.json").exists():
                    stage = "unknown"
                    text = str(exc).lower()
                    for candidate in ("smoke", "full", "score", "table", "final"):
                        if candidate in text:
                            stage = candidate
                            break
                    write_halt(config, stage, str(exc))
                else:
                    write_state(config, "HALTED", str(exc))
                return 2
            if complete and (config.stop_when_complete or config.mode == "once"):
                return 0
            if config.mode == "once":
                return 3 if not complete else 0
            if config.max_scans and scans >= config.max_scans:
                write_state(config, "MAX_SCANS_REACHED", f"stopped after {scans} scans")
                return 3
            time.sleep(config.poll_seconds)
    finally:
        if lock is not None:
            shutil.rmtree(lock, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("once", "monitor"), default=None)
    parser.add_argument("--action", choices=("plan", "submit"), default=None)
    parser.add_argument("--poll-seconds", type=int, default=None)
    parser.add_argument("--max-scans", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = build_config(build_parser().parse_args(argv))
        return run(config)
    except (ContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("ERROR: interrupted", file=sys.stderr)
        return 130
    except Exception:  # Defensive: unexpected code errors must never advance stages.
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())

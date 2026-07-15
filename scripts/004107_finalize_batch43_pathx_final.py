#!/usr/bin/env python3
"""Audit Batch-44 v1 Best3 evidence and register ``path_x_final``.

The script first decodes each anonymous review with its private page manifest,
then validates the paired full320 evidence for the selected arm/step.  Without
``--winner`` it only writes a review/objective report and exits 3; no model is
silently promoted.  ``--winner auto`` accepts the unique best subjective net
score (candidate wins minus Batch-33 wins), while ``--winner CANDIDATE_ID``
records an explicit human decision.

Finalization hashes all five checkpoint files and emits the immutable
``moss_codecvc.batch44_v1_final_selection.v1`` manifest consumed by 004106/004108.
It never submits a QZ task.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


BEST3_SCHEMA = "moss_codecvc.batch44_v1_best3_selection.v1"
BLIND_SCHEMA = "moss_codecvc.batch44_v1_best3_blind20.v1"
FULL320_COMPLETION_SCHEMA = "batch44_v1_paired_full320_v1"
FINAL_SCHEMA = "moss_codecvc.batch44_v1_final_selection.v1"
DECISION_SCHEMA = "moss_codecvc.batch44_v1_final_decision_report.v1"
EXPERIMENT_ID = "batch44_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEST3 = PROJECT_ROOT / "testset/outputs/batch44_best3_20260713/best3_selection.json"
DEFAULT_BLIND_READY = (
    PROJECT_ROOT
    / "testset/outputs/batch44_best3_20260713/private_blind20/BLIND20_READY.json"
)
DEFAULT_REPORT = PROJECT_ROOT / "testset/outputs/batch44_best3_20260713/final_decision_report.json"
DEFAULT_FINAL = PROJECT_ROOT / "testset/outputs/batch44_best3_20260713/FINAL_SELECTION.json"
EXPECTED_JOBS = {
    "r3": "job-2b91d332-d500-4279-84f9-0a6a81a376aa",
    "r5": "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c",
}
EXPECTED_REPEATS = {"r3": 3, "r5": 5}
ALLOWED_STEPS = {26000, 28000, 30000}
RUN_DIRS = {
    "r3": "ver2_9_5_final_r3_v1_30k",
    "r5": "ver2_9_5_final_r5_v1_30k",
}
REJECTED_BATCH43_V2_JOBS = {
    "job-a34d84d4-59cc-4824-b197-0829bfe79004",
    "job-aef79753-7fcd-444e-b94d-3e21eedb2394",
}
ALLOWED_COMPUTE_GROUP = "lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"
ALLOWED_SPEC = "67b10bc6-78b0-41a3-aaf4-358eeeb99009"
EXPECTED_CODE_ROOT = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
    "MOSS-CodecVC_snapshots/ver23_batch37_eval_20260711_1092820"
)
JOB_ID_RE = re.compile(
    r"^job-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
MODEL_FILES = (
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "timbre_memory_config.json",
    "timbre_memory_adapter.pt",
)
METRIC_KEYS = {
    "step", "arm", "text_repeat", "train_job_id", "scope", "n", "keep",
    "fail_count", "fail_rate", "cer", "wavlm_sim_ref", "wavlm_sim_src",
    "wavlm_margin", "wavlm_ref_bound", "speechbrain_sim_ref",
    "speechbrain_sim_src", "speechbrain_margin", "speechbrain_ref_bound",
    "ref_content_lcs_f1", "text_en_src_n", "text_en_src_fail_count",
    "text_en_src_fail_rate", "text_en_src_cer",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_binding(raw: str, *, option: str) -> tuple[str, Path]:
    key, separator, value = raw.partition("=")
    if not separator or not key.strip() or not value.strip():
        raise ValueError(f"{option} must use KEY=PATH, got {raw!r}")
    return key.strip(), Path(value.strip()).expanduser().resolve()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def load_best3_selector(project_root: Path):
    candidates = (
        Path(__file__).with_name("004103_select_batch43_best3.py"),
        project_root / "scripts/004103_select_batch43_best3.py",
    )
    selector_path = next((path for path in candidates if path.is_file()), None)
    if selector_path is None:
        raise FileNotFoundError("missing 004103 Best3 selector for provenance replay")
    name = "moss_codecvc_batch44_best3_selector_replay"
    spec = importlib.util.spec_from_file_location(name, selector_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import Best3 selector: {selector_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def revalidate_best3_quick20(payload: Mapping[str, Any], *, project_root: Path) -> None:
    if Path(str(payload.get("project_root") or "")).resolve() != project_root.resolve():
        raise ValueError("Best3 project_root is not the registered project")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 6:
        raise ValueError("Best3 quick20 replay requires all six candidates")
    stamps: set[str] = set()
    for row in candidates:
        quick = row.get("quick20") if isinstance(row, dict) else None
        if not isinstance(quick, dict):
            raise ValueError("Best3 candidate lacks quick20 provenance")
        metrics_path = Path(str(quick.get("metrics_json") or "")).resolve()
        expected_prefix = f"ver23_batch44_quick20_step{row.get('step')}_"
        if not metrics_path.parent.name.startswith(expected_prefix):
            raise ValueError(f"Best3 quick20 path drift: {metrics_path}")
        stamps.add(metrics_path.parent.name[len(expected_prefix):])
        if quick.get("metrics_sha256") != sha256_file(metrics_path):
            raise ValueError(f"Best3 quick20 SHA256 drift: {metrics_path}")
    if len(stamps) != 1 or not next(iter(stamps)):
        raise ValueError(f"Best3 quick20 stamps disagree: {sorted(stamps)}")
    selector = load_best3_selector(project_root)
    replay = selector.build_plan(project_root, stamp=next(iter(stamps)))
    semantic_keys = (
        "schema_version", "experiment_id", "data_version", "status",
        "project_root", "registered_candidate_space", "ranking", "candidates",
        "selected_candidate_ids", "paired_full320_plan",
    )
    expected = {key: replay.get(key) for key in semantic_keys}
    actual = {key: payload.get(key) for key in semantic_keys}
    if actual != expected:
        raise ValueError(
            "Best3 selection differs from replayed 004103 quick20 ranking; "
            "the shortlist may have been edited"
        )


def load_best3(
    path: Path, *, project_root: Path = PROJECT_ROOT, replay_quick20: bool = True
) -> tuple[dict[str, Any], list[str], dict[str, dict[str, Any]]]:
    payload = load_json(path)
    if payload.get("schema_version") != BEST3_SCHEMA or payload.get("status") != "selected":
        raise ValueError(f"invalid Best3 selection: {path}")
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("data_version") != "v1_20260709"
    ):
        raise ValueError(f"Best3 selection is not registered Batch-44 v1: {path}")
    expected_space = {
        "arms": ["r3", "r5"],
        "steps": sorted(ALLOWED_STEPS),
        "candidate_count": 6,
    }
    if payload.get("registered_candidate_space") != expected_space:
        raise ValueError(
            f"Best3 registered candidate space drift: "
            f"{payload.get('registered_candidate_space')!r}"
        )
    selected = payload.get("selected_candidate_ids")
    if not isinstance(selected, list) or len(selected) != 3 or len(set(selected)) != 3:
        raise ValueError(f"Best3 must contain three candidates: {selected!r}")
    candidate_rows = payload.get("candidates")
    if not isinstance(candidate_rows, list) or len(candidate_rows) != 6:
        raise ValueError("Best3 must contain the complete six-candidate space")
    candidates: dict[str, dict[str, Any]] = {}
    for row in candidate_rows:
        if not isinstance(row, dict):
            raise ValueError("Best3 candidate must be an object")
        arm, step = row.get("arm"), row.get("step")
        if arm not in EXPECTED_JOBS or step not in ALLOWED_STEPS:
            raise ValueError(
                f"Best3 candidate is outside registered 30k checkpoints: "
                f"{arm!r}/step-{step!r}"
            )
        candidate_id = f"{arm}_step-{step}"
        if row.get("candidate_id") != candidate_id:
            raise ValueError(
                f"Best3 candidate_id={row.get('candidate_id')!r}, "
                f"expected {candidate_id!r}"
            )
        if row.get("text_repeat") != EXPECTED_REPEATS[str(arm)]:
            raise ValueError(f"Best3 {candidate_id} text repeat drift")
        if row.get("train_job_id") in REJECTED_BATCH43_V2_JOBS:
            raise ValueError(f"Best3 {candidate_id} references stopped Batch-43 v2")
        if row.get("train_job_id") != EXPECTED_JOBS[str(arm)]:
            raise ValueError(f"Best3 {candidate_id} training job drift")
        if candidate_id in candidates:
            raise ValueError(f"duplicate Best3 candidate: {candidate_id}")
        candidates[candidate_id] = row
    expected_candidates = {
        f"{arm}_step-{step}"
        for arm in ("r3", "r5")
        for step in ALLOWED_STEPS
    }
    if set(candidates) != expected_candidates:
        raise ValueError("Best3 does not contain the registered six-candidate space")
    if any(candidate_id not in candidates for candidate_id in selected):
        raise ValueError("Best3 selected ids are missing from candidates")
    selected_flags = {
        candidate_id
        for candidate_id, row in candidates.items()
        if row.get("selected_for_full320") is True
    }
    if selected_flags != set(selected):
        raise ValueError("Best3 selected ids and selected_for_full320 flags disagree")
    if replay_quick20:
        revalidate_best3_quick20(payload, project_root=project_root)
    return payload, [str(item) for item in selected], candidates


def default_full320_paths(project_root: Path, steps: set[int]) -> dict[int, tuple[Path, Path]]:
    resolved: dict[int, tuple[Path, Path]] = {}
    for step in steps:
        local = (
            project_root
            / "trainset/local_jobs"
            / f"ver23_batch44_paired_full320_step{step}_20260713/COMPLETED.json"
        )
        qz = (
            project_root
            / "trainset/qz_jobs"
            / f"ver23_batch44_paired_full320_step{step}_20260713/COMPLETED.json"
        )
        if local.is_file() and qz.is_file():
            raise ValueError(
                f"ambiguous full320 provenance for step-{step}: both local and QZ completions exist"
            )
        completion = local if local.is_file() or not qz.is_file() else qz
        metrics = (
            project_root
            / "testset/outputs/ver23_batch44_paired_full320_20260713"
            / f"step-{step}/aggregate/paired_metrics.json"
        )
        resolved[step] = completion, metrics
    return resolved


def validate_local_full320_execution(
    *,
    completion: Mapping[str, Any],
    completion_path: Path,
    expected_record: Path,
    expected_paths: Mapping[str, Path],
) -> None:
    """Validate local 2x4090 execution without manufacturing QZ provenance."""

    submit_ledger = expected_record / "submitted_jobs.tsv"
    if submit_ledger.exists():
        raise ValueError(f"{completion_path}: local completion forbids {submit_ledger}")
    if "evaluation_job" in completion:
        raise ValueError(f"{completion_path}: local completion must not declare evaluation_job")

    def require_sha_bound(
        value: Any, *, expected: Path, label: str
    ) -> Path:
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise ValueError(f"{completion_path}: {label} must contain exactly path/sha256")
        actual = Path(str(value.get("path") or "")).expanduser().resolve()
        if actual != expected.resolve():
            raise ValueError(f"{completion_path}: {label} path={actual}, expected {expected}")
        if not actual.is_file():
            raise FileNotFoundError(actual)
        if value.get("sha256") != sha256_file(actual):
            raise ValueError(f"{completion_path}: {label} SHA256 drift")
        return actual

    runner_path = require_sha_bound(
        completion.get("runner"),
        expected=expected_record / "004118_run_batch44_v1_paired_full320_local.frozen.sh",
        label="runner",
    )
    engine_path = require_sha_bound(
        completion.get("engine"),
        expected=expected_record / "004112_batch44_v1_paired_full320_engine.frozen.sh",
        label="engine",
    )
    require_sha_bound(
        completion.get("inputs_manifest"),
        expected=expected_record / "frozen_inputs.sha256",
        label="inputs_manifest",
    )
    require_sha_bound(
        completion.get("resolved_runs"),
        expected=expected_record / "resolved_runs.tsv",
        label="resolved_runs",
    )
    if runner_path == engine_path:
        raise ValueError(f"{completion_path}: local runner and engine must be separately frozen")

    checkpoint_provenance = completion.get("checkpoint_provenance")
    if not isinstance(checkpoint_provenance, dict) or set(checkpoint_provenance) != {"r3", "r5"}:
        raise ValueError(f"{completion_path}: local checkpoint provenance registry drift")
    project_root = expected_record.parents[2]
    required_checkpoint_files = {
        "adapter_model.safetensors",
        "adapter_config.json",
        "README.md",
        "timbre_memory_adapter.pt",
        "timbre_memory_config.json",
    }
    for arm in ("r3", "r5"):
        provenance_path = require_sha_bound(
            checkpoint_provenance[arm],
            expected=expected_record / f"checkpoint_{arm}_step{completion.get('step')}.json",
            label=f"checkpoint_provenance.{arm}",
        )
        provenance = load_json(provenance_path)
        expected_checkpoint = (
            project_root
            / "outputs/lora_runs"
            / RUN_DIRS[arm]
            / f"step-{completion.get('step')}"
        ).resolve()
        if (
            provenance.get("arm") != arm
            or provenance.get("step") != completion.get("step")
            or provenance.get("text_repeat") != EXPECTED_REPEATS[arm]
            or provenance.get("training_job_id") != EXPECTED_JOBS[arm]
            or Path(str(provenance.get("checkpoint") or "")).resolve() != expected_checkpoint
        ):
            raise ValueError(f"{completion_path}: {arm} checkpoint provenance identity drift")
        inventory = provenance.get("checkpoint_inventory")
        if not isinstance(inventory, dict) or set(inventory) != required_checkpoint_files:
            raise ValueError(f"{completion_path}: {arm} checkpoint inventory drift")
        for name, item in inventory.items():
            path = expected_checkpoint / name
            if not isinstance(item, dict) or not path.is_file():
                raise ValueError(f"{completion_path}: {arm} checkpoint file missing: {name}")
            if item.get("bytes") != path.stat().st_size or item.get("sha256") != sha256_file(path):
                raise ValueError(f"{completion_path}: {arm} checkpoint file SHA/size drift: {name}")

    artifacts = completion.get("artifacts")
    expected_artifacts = {
        key: expected_paths[key]
        for key in (
            "completeness_json",
            "dual_encoder_cases_csv",
            "paired_metrics_tsv",
            "paired_metrics_json",
            "paired_metrics_md",
        )
    }
    if not isinstance(artifacts, dict) or set(artifacts) != set(expected_artifacts):
        raise ValueError(f"{completion_path}: local artifact registry drift")
    for key, expected in expected_artifacts.items():
        require_sha_bound(artifacts[key], expected=expected, label=f"artifacts.{key}")

    execution = completion.get("execution")
    if not isinstance(execution, dict):
        raise ValueError(f"{completion_path}: missing local execution provenance")
    hostname = str(execution.get("hostname") or "")
    if not hostname.startswith("xyzhang-dev--"):
        raise ValueError(f"{completion_path}: unregistered local hostname {hostname!r}")
    if execution.get("gpu_count") != 2 or execution.get("gpu_indices") != [0, 1]:
        raise ValueError(f"{completion_path}: local GPU count/indices drift")
    if execution.get("gpu_models") != [
        "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce RTX 4090",
    ]:
        raise ValueError(f"{completion_path}: local GPU model drift")
    memories = execution.get("gpu_memory_total_mib")
    if (
        not isinstance(memories, list)
        or len(memories) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 48_000 for value in memories)
    ):
        raise ValueError(f"{completion_path}: local GPU memory inventory drift")
    uuid_re = re.compile(
        r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    )
    uuids = execution.get("gpu_uuids")
    if (
        not isinstance(uuids, list)
        or len(uuids) != 2
        or len(set(uuids)) != 2
        or any(not uuid_re.fullmatch(str(value)) for value in uuids)
    ):
        raise ValueError(f"{completion_path}: local GPU UUID inventory drift")
    pid = execution.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError(f"{completion_path}: invalid local execution pid")
    for key in ("started_utc", "completed_utc"):
        raw = str(execution.get(key) or "")
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{completion_path}: invalid execution {key}={raw!r}") from exc
        if parsed.tzinfo is None:
            raise ValueError(f"{completion_path}: execution {key} must be timezone-aware")
    if execution.get("completed_utc") != completion.get("completed_utc"):
        raise ValueError(f"{completion_path}: completion/execution timestamp drift")

    inventory_path = Path(str(execution.get("gpu_inventory") or "")).expanduser().resolve()
    expected_inventory = (expected_record / "runtime_gpu_inventory.json").resolve()
    if inventory_path != expected_inventory or not inventory_path.is_file():
        raise ValueError(f"{completion_path}: GPU inventory path drift")
    if execution.get("gpu_inventory_sha256") != sha256_file(inventory_path):
        raise ValueError(f"{completion_path}: GPU inventory SHA256 drift")
    inventory = load_json(inventory_path)
    if (
        not isinstance(inventory, dict)
        or inventory.get("schema") != "batch44_local_gpu_inventory_v1"
        or inventory.get("hostname") != hostname
    ):
        raise ValueError(f"{completion_path}: GPU inventory identity drift")
    gpu_rows = inventory.get("gpus")
    if not isinstance(gpu_rows, list) or len(gpu_rows) != 2:
        raise ValueError(f"{completion_path}: GPU inventory must contain two rows")
    for index, row in enumerate(gpu_rows):
        if not isinstance(row, dict):
            raise ValueError(f"{completion_path}: invalid GPU inventory row")
        expected = {
            "index": index,
            "uuid": uuids[index],
            "name": execution["gpu_models"][index],
            "memory_total_mib": memories[index],
        }
        bad = {key: row.get(key) for key, wanted in expected.items() if row.get(key) != wanted}
        if bad or not str(row.get("driver_version") or ""):
            raise ValueError(f"{completion_path}: GPU inventory row-{index} drift: {bad}")

    marker = expected_record / "complete.marker"
    if not marker.is_file():
        raise FileNotFoundError(marker)
    expected_marker = f"COMPLETED.json sha256\t{sha256_file(completion_path)}\n"
    if marker.read_text(encoding="utf-8") != expected_marker:
        raise ValueError(f"{completion_path}: complete.marker COMPLETED.json SHA mismatch")


def validate_full320_provenance(
    *, step: int, completion_path: Path, metrics_path: Path, project_root: Path
) -> dict[str, Any]:
    project_root = project_root.resolve()
    expected_local_record = (
        project_root
        / "trainset/local_jobs"
        / f"ver23_batch44_paired_full320_step{step}_20260713"
    ).resolve()
    expected_qz_record = (
        project_root
        / "trainset/qz_jobs"
        / f"ver23_batch44_paired_full320_step{step}_20260713"
    ).resolve()
    expected_step_root = (
        project_root
        / "testset/outputs/ver23_batch44_paired_full320_20260713"
        / f"step-{step}"
    ).resolve()
    resolved_completion = completion_path.resolve()
    if resolved_completion == expected_local_record / "COMPLETED.json":
        backend = "local"
        expected_record = expected_local_record
    elif resolved_completion == expected_qz_record / "COMPLETED.json":
        backend = "qz"
        expected_record = expected_qz_record
    else:
        raise ValueError(
            f"full320 completion path={resolved_completion}, expected local "
            f"{expected_local_record / 'COMPLETED.json'} or QZ "
            f"{expected_qz_record / 'COMPLETED.json'}"
        )
    expected_completion = expected_record / "COMPLETED.json"
    expected_metrics = expected_step_root / "aggregate/paired_metrics.json"
    if metrics_path.resolve() != expected_metrics:
        raise ValueError(
            f"full320 metrics path={metrics_path.resolve()}, expected {expected_metrics}"
        )

    completion = load_json(completion_path)
    declared_backend = completion.get("backend")
    if backend == "local" and declared_backend != "local":
        raise ValueError(f"{completion_path}: local completion must declare backend='local'")
    if backend == "qz" and declared_backend not in {None, "qz"}:
        raise ValueError(f"{completion_path}: QZ completion backend drift: {declared_backend!r}")
    expected_paths = {
        "record_root": expected_record,
        "step_root": expected_step_root,
        "paired_metrics_json": expected_metrics,
        "paired_metrics_tsv": expected_step_root / "aggregate/paired_metrics.tsv",
        "paired_metrics_md": expected_step_root / "aggregate/paired_metrics.md",
        "dual_encoder_cases_csv": expected_step_root / "aggregate/dual_encoder_cases.csv",
        "completeness_json": expected_step_root / "aggregate/completeness.json",
    }
    for key, expected in expected_paths.items():
        actual = Path(str(completion.get(key) or "")).expanduser().resolve()
        if actual != expected.resolve():
            raise ValueError(f"{completion_path}: {key}={actual}, expected {expected}")
        if key not in {"record_root", "step_root"} and not actual.is_file():
            raise FileNotFoundError(actual)

    completeness = load_json(expected_paths["completeness_json"])
    lanes = completeness.get("lanes") if isinstance(completeness, dict) else None
    if not isinstance(lanes, list) or len(lanes) != 4:
        raise ValueError(f"{expected_paths['completeness_json']}: expected four lanes")
    lane_index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for lane in lanes:
        if not isinstance(lane, dict):
            raise ValueError("full320 completeness lane must be an object")
        arm, mode = str(lane.get("arm") or ""), str(lane.get("mode") or "")
        key = (arm, mode)
        if arm not in EXPECTED_JOBS or mode not in {"no_text", "text"} or key in lane_index:
            raise ValueError(f"invalid/duplicate full320 lane: {key}")
        expected_run = f"ver2_9_5_final_{arm}_step-{step}_{mode}_seedtts160_d2d3_seed1234"
        expected_checkpoint = (
            project_root / "outputs/lora_runs" / RUN_DIRS[arm] / f"step-{step}"
        ).resolve()
        if (
            lane.get("run_id") != expected_run
            or lane.get("training_job_id") != EXPECTED_JOBS[arm]
            or Path(str(lane.get("checkpoint") or "")).resolve() != expected_checkpoint
            or lane.get("rows") != 160
            or lane.get("asr_rows") != 160
            or lane.get("bnf_extraction_lines") != (160 if mode == "no_text" else 0)
        ):
            raise ValueError(f"full320 lane provenance/completeness drift: {key}")
        lane_index[key] = lane
    if set(lane_index) != {(arm, mode) for arm in ("r3", "r5") for mode in ("no_text", "text")}:
        raise ValueError("full320 completeness is missing an arm/mode lane")

    dual_path = expected_paths["dual_encoder_cases_csv"]
    with dual_path.open(encoding="utf-8", newline="") as handle:
        dual_rows = list(csv.DictReader(handle))
    if len(dual_rows) != 640:
        raise ValueError(f"{dual_path}: expected 640 dual-encoder rows, got {len(dual_rows)}")
    dual_counts: Counter[tuple[str, str]] = Counter()
    dual_ids: set[tuple[str, str]] = set()
    for row in dual_rows:
        run, mode, case_id = str(row.get("run") or ""), str(row.get("mode") or ""), str(row.get("case_id") or "")
        matched = [
            key
            for key in lane_index
            if run == f"ver2_9_5_final_{key[0]}_step-{step}_{key[1]}_seedtts160_d2d3_seed1234"
            and mode == key[1]
        ]
        if len(matched) != 1 or not case_id or (run, case_id) in dual_ids:
            raise ValueError(f"{dual_path}: invalid/duplicate dual row {run}/{case_id}")
        for field in ("sim_gen_ref", "sim_gen_source", "ecapa_sim_gen_ref", "ecapa_sim_gen_source", "cer_tgt"):
            try:
                value = float(row.get(field))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{dual_path}:{run}/{case_id}.{field} is not numeric") from exc
            if not math.isfinite(value):
                raise ValueError(f"{dual_path}:{run}/{case_id}.{field} is not finite")
        dual_ids.add((run, case_id))
        dual_counts[matched[0]] += 1
    if any(dual_counts[key] != 160 for key in lane_index):
        raise ValueError(f"{dual_path}: dual row counts drift: {dict(dual_counts)}")

    if completion.get("training_jobs") != EXPECTED_JOBS:
        raise ValueError(f"{completion_path}: training job provenance drift")
    if completion.get("scope") != {
        "r3": {"no_text": 160, "text": 160},
        "r5": {"no_text": 160, "text": 160},
    }:
        raise ValueError(f"{completion_path}: full320 scope drift")
    expected_gpu_plan = (
        {
            "r3_no_text": "0,1",
            "r3_text": "0,1",
            "r5_no_text": "0,1",
            "r5_text": "0,1",
        }
        if backend == "local"
        else {
            "r3_no_text": "0,1",
            "r3_text": "2,3",
            "r5_no_text": "4,5",
            "r5_text": "6,7",
        }
    )
    if completion.get("gpu_plan") != expected_gpu_plan:
        raise ValueError(f"{completion_path}: GPU lane plan drift")
    if backend == "local" and completion.get("lane_execution") != "sequential":
        raise ValueError(f"{completion_path}: local lanes must execute sequentially")
    if Path(str(completion.get("code_root") or "")).resolve() != EXPECTED_CODE_ROOT:
        raise ValueError(f"{completion_path}: frozen code root drift")

    validation = (
        project_root / "testset/validation/seedtts_vc_ver2_3_validation.jsonl"
    ).resolve()
    if Path(str(completion.get("validation_jsonl") or "")).resolve() != validation:
        raise ValueError(f"{completion_path}: validation manifest path drift")
    if not validation.is_file() or completion.get("validation_sha256") != sha256_file(validation):
        raise ValueError(f"{completion_path}: validation manifest hash drift")

    training_ledger = (
        project_root
        / "trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713"
        / "submitted_pair.tsv"
    ).resolve()
    if Path(str(completion.get("training_pair_ledger") or "")).resolve() != training_ledger:
        raise ValueError(f"{completion_path}: training pair ledger path drift")
    if not training_ledger.is_file() or completion.get("training_pair_ledger_sha256") != sha256_file(training_ledger):
        raise ValueError(f"{completion_path}: training pair ledger hash drift")

    submit_ledger = expected_record / "submitted_jobs.tsv"
    if backend == "qz":
        if not submit_ledger.is_file():
            raise FileNotFoundError(submit_ledger)
        with submit_ledger.open(encoding="utf-8", newline="") as handle:
            submitted = list(csv.DictReader(handle, delimiter="\t"))
        if len(submitted) != 1:
            raise ValueError(f"{submit_ledger}: expected one QZ submission row")
        row = submitted[0]
        expected_submit = {
            "step": str(step),
            "compute_group": ALLOWED_COMPUTE_GROUP,
            "spec": ALLOWED_SPEC,
            "r3_train_job_id": EXPECTED_JOBS["r3"],
            "r5_train_job_id": EXPECTED_JOBS["r5"],
        }
        bad = {key: row.get(key) for key, wanted in expected_submit.items() if row.get(key) != wanted}
        if bad:
            raise ValueError(f"{submit_ledger}: MTTS/QZ provenance drift: {bad}")
        expected_submit_paths = {
            "record_root": expected_record,
            "step_root": expected_step_root,
            "code_root": EXPECTED_CODE_ROOT,
        }
        for key, wanted in expected_submit_paths.items():
            if Path(str(row.get(key) or "")).resolve() != wanted.resolve():
                raise ValueError(f"{submit_ledger}: MTTS/QZ path provenance drift: {key}")
        if not JOB_ID_RE.fullmatch(str(row.get("job_id") or "")):
            raise ValueError(f"{submit_ledger}: invalid QZ job id {row.get('job_id')!r}")
    else:
        validate_local_full320_execution(
            completion=completion,
            completion_path=completion_path,
            expected_record=expected_record,
            expected_paths=expected_paths,
        )
    return completion


def validate_full320_step(
    *, step: int, completion_path: Path, metrics_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    completion = validate_full320_provenance(
        step=step,
        completion_path=completion_path,
        metrics_path=metrics_path,
        project_root=project_root,
    )
    if completion.get("schema") != FULL320_COMPLETION_SCHEMA:
        raise ValueError(f"{completion_path}: wrong schema {completion.get('schema')!r}")
    if completion.get("status") != "complete" or completion.get("step") != step:
        raise ValueError(f"{completion_path}: incomplete/wrong step")
    if Path(str(completion.get("paired_metrics_json") or "")).resolve() != metrics_path.resolve():
        raise ValueError(f"{completion_path}: paired_metrics_json path drift")
    if Path(str(completion.get("dual_encoder_cases_csv") or "")).is_file() is False:
        raise FileNotFoundError(f"{completion_path}: missing dual_encoder_cases_csv")
    metrics = load_json(metrics_path)
    if not isinstance(metrics, list) or len(metrics) != 6:
        raise ValueError(f"{metrics_path}: expected six metric rows")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in metrics:
        if not isinstance(row, dict) or set(row) != METRIC_KEYS:
            raise ValueError(f"{metrics_path}: metric key/schema drift")
        arm, scope = row.get("arm"), row.get("scope")
        if arm not in {"r3", "r5"} or scope not in {"no_text", "text", "all"}:
            raise ValueError(f"{metrics_path}: bad identity {arm!r}/{scope!r}")
        key = (str(arm), str(scope))
        if key in indexed:
            raise ValueError(f"{metrics_path}: duplicate {key}")
        if row.get("step") != step:
            raise ValueError(f"{metrics_path}: {key} wrong step")
        if row.get("text_repeat") != EXPECTED_REPEATS[str(arm)]:
            raise ValueError(f"{metrics_path}: {key} wrong repeat")
        if row.get("train_job_id") != EXPECTED_JOBS[str(arm)]:
            raise ValueError(f"{metrics_path}: {key} wrong train job")
        expected_n = 320 if scope == "all" else 160
        if row.get("n") != expected_n:
            raise ValueError(f"{metrics_path}: {key} n={row.get('n')!r}")
        keep = row.get("keep")
        fail_count = row.get("fail_count")
        if (
            isinstance(keep, bool)
            or not isinstance(keep, int)
            or not 0 <= keep <= expected_n
        ):
            raise ValueError(f"{metrics_path}: {key} invalid keep={keep!r}")
        if (
            isinstance(fail_count, bool)
            or not isinstance(fail_count, int)
            or fail_count != expected_n - keep
        ):
            raise ValueError(
                f"{metrics_path}: {key} fail_count={fail_count!r}, expected {expected_n - keep}"
            )
        for metric in (
            "fail_rate", "cer", "wavlm_sim_ref", "wavlm_sim_src", "wavlm_margin",
            "wavlm_ref_bound", "speechbrain_sim_ref", "speechbrain_sim_src",
            "speechbrain_margin", "speechbrain_ref_bound", "ref_content_lcs_f1",
        ):
            finite(row.get(metric), label=f"{metrics_path}:{key}.{metric}")
        if not math.isclose(
            float(row["fail_rate"]), fail_count / expected_n, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{metrics_path}: {key} fail_rate/count mismatch")
        for metric in ("fail_rate", "wavlm_ref_bound", "speechbrain_ref_bound"):
            if not 0.0 <= float(row[metric]) <= 1.0:
                raise ValueError(f"{metrics_path}: {key} {metric} outside [0,1]")
        if float(row["cer"]) < 0.0 or float(row["ref_content_lcs_f1"]) < 0.0:
            raise ValueError(f"{metrics_path}: {key} negative CER/F1")
        if not math.isclose(
            float(row["wavlm_margin"]),
            float(row["wavlm_sim_ref"]) - float(row["wavlm_sim_src"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{metrics_path}: {key} WavLM margin mismatch")
        if not math.isclose(
            float(row["speechbrain_margin"]),
            float(row["speechbrain_sim_ref"]) - float(row["speechbrain_sim_src"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{metrics_path}: {key} SpeechBrain margin mismatch")
        if scope == "no_text":
            if any(row.get(field) not in {"", None} for field in (
                "text_en_src_n", "text_en_src_fail_count", "text_en_src_fail_rate", "text_en_src_cer"
            )):
                raise ValueError(f"{metrics_path}: {key} unexpectedly contains text en_src metrics")
        else:
            en_n = row.get("text_en_src_n")
            en_fail = row.get("text_en_src_fail_count")
            if en_n != 80 or isinstance(en_fail, bool) or not isinstance(en_fail, int) or not 0 <= en_fail <= 80:
                raise ValueError(f"{metrics_path}: {key} invalid text en_src counts")
            en_rate = finite(
                row.get("text_en_src_fail_rate"), label=f"{metrics_path}:{key}.text_en_src_fail_rate"
            )
            en_cer = finite(
                row.get("text_en_src_cer"), label=f"{metrics_path}:{key}.text_en_src_cer"
            )
            if not math.isclose(en_rate, en_fail / 80, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{metrics_path}: {key} text en_src fail-rate mismatch")
            if en_cer < 0.0:
                raise ValueError(f"{metrics_path}: {key} negative text en_src CER")
        indexed[key] = dict(row)
    if len(indexed) != 6:
        raise ValueError(f"{metrics_path}: missing arm/scope rows")
    averaged_metrics = (
        "cer", "wavlm_sim_ref", "wavlm_sim_src", "wavlm_margin", "wavlm_ref_bound",
        "speechbrain_sim_ref", "speechbrain_sim_src", "speechbrain_margin",
        "speechbrain_ref_bound", "ref_content_lcs_f1",
    )
    for arm in ("r3", "r5"):
        no_text = indexed[(arm, "no_text")]
        text = indexed[(arm, "text")]
        combined = indexed[(arm, "all")]
        if combined["keep"] != no_text["keep"] + text["keep"]:
            raise ValueError(f"{metrics_path}: {arm}/all keep is not the mode sum")
        if combined["fail_count"] != no_text["fail_count"] + text["fail_count"]:
            raise ValueError(f"{metrics_path}: {arm}/all fail_count is not the mode sum")
        for metric in averaged_metrics:
            expected = (float(no_text[metric]) + float(text[metric])) / 2.0
            if not math.isclose(float(combined[metric]), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{metrics_path}: {arm}/all {metric} is not the mode mean")
        for metric in (
            "text_en_src_n", "text_en_src_fail_count", "text_en_src_fail_rate", "text_en_src_cer"
        ):
            if combined[metric] != text[metric]:
                raise ValueError(f"{metrics_path}: {arm}/all {metric} differs from text")
    return completion, indexed


def require_registered_file(
    *, value: Any, expected_path: Path | None, expected_sha256: Any, label: str
) -> Path:
    path = Path(str(value or "")).expanduser().resolve()
    if expected_path is not None and path != expected_path.expanduser().resolve():
        raise ValueError(f"{label} path={path}, expected {expected_path.resolve()}")
    if not path.is_file():
        raise FileNotFoundError(path)
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"{label} has invalid registered SHA256")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA256={actual}, expected {expected_sha256}")
    return path


def load_blind_ready(
    path: Path,
    selected: list[str],
    *,
    best3_path: Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = load_json(path)
    if payload.get("schema_version") != BLIND_SCHEMA or payload.get("status") != "complete":
        raise ValueError(f"invalid BLIND20_READY: {path}")
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("data_version") != "v1_20260709"
    ):
        raise ValueError(f"BLIND20_READY is not registered Batch-44 v1: {path}")
    blind_producer = (
        project_root / "scripts/004104_build_batch43_best3_blind20.py"
    ).resolve()
    if not blind_producer.is_file():
        raise FileNotFoundError(f"missing registered 004104 blind builder: {blind_producer}")
    expected_blind_producer = {
        "script": str(blind_producer),
        "script_sha256": sha256_file(blind_producer),
        "entrypoint": blind_producer.name,
    }
    if payload.get("producer") != expected_blind_producer:
        raise ValueError("BLIND20_READY is not bound to the registered 004104 producer")
    if payload.get("selected_candidate_ids") != selected:
        raise ValueError("BLIND20_READY candidate order/identity differs from Best3")
    case_ids_by_candidate = payload.get("case_ids_by_candidate")
    if not isinstance(case_ids_by_candidate, dict) or set(case_ids_by_candidate) != set(selected):
        raise ValueError("BLIND20_READY must contain one case list per Best3 candidate")
    all_case_ids: set[str] = set()
    for candidate_id in selected:
        case_ids = case_ids_by_candidate[candidate_id]
        if not isinstance(case_ids, list):
            raise ValueError(f"BLIND20_READY {candidate_id} case list is invalid")
        normalized = [str(item or "") for item in case_ids]
        if len(normalized) != 20 or len(set(normalized)) != 20 or any(not item for item in normalized):
            raise ValueError(f"BLIND20_READY {candidate_id} must contain 20 unique cases")
        if all_case_ids.intersection(normalized):
            raise ValueError("BLIND20_READY pages must use disjoint case sets")
        all_case_ids.update(normalized)
    if best3_path is not None:
        best3_path = best3_path.expanduser().resolve()
        require_registered_file(
            value=payload.get("selection_json"),
            expected_path=best3_path,
            expected_sha256=payload.get("selection_sha256"),
            label="BLIND20_READY Best3 selection",
        )
    batch33_path = require_registered_file(
        value=payload.get("batch33_diagnostics_csv"),
        expected_path=None,
        expected_sha256=payload.get("batch33_diagnostics_sha256"),
        label="BLIND20_READY Batch-33 diagnostics",
    )
    if payload.get("batch33_run_id") != "Batch33":
        raise ValueError("BLIND20_READY Batch-33 run identity drift")
    with batch33_path.open(encoding="utf-8", newline="") as handle:
        batch33_rows = [
            row for row in csv.DictReader(handle)
            if str(row.get("run") or row.get("run_id") or "") == "Batch33"
            and str(row.get("mode") or "no_text") == "no_text"
        ]
    if not batch33_rows:
        raise ValueError("BLIND20_READY Batch-33 diagnostics has no no_text rows")
    batch33_by_case = {
        str(row.get("case_id") or ""): row for row in batch33_rows
    }
    if len(batch33_by_case) != len(batch33_rows) or "" in batch33_by_case:
        raise ValueError("BLIND20_READY Batch-33 diagnostics has duplicate/empty case ids")

    evidence_by_candidate = payload.get("candidate_evidence_by_candidate")
    if not isinstance(evidence_by_candidate, dict) or set(evidence_by_candidate) != set(selected):
        raise ValueError("BLIND20_READY must bind full320 evidence for every Best3 candidate")
    validated_evidence: dict[str, dict[str, Any]] = {}
    for candidate_id in selected:
        evidence = evidence_by_candidate[candidate_id]
        if not isinstance(evidence, dict):
            raise ValueError(f"BLIND20_READY {candidate_id} evidence must be an object")
        arm, separator, step_text = candidate_id.partition("_step-")
        if not separator or arm not in EXPECTED_JOBS:
            raise ValueError(f"invalid blind candidate id: {candidate_id}")
        try:
            step = int(step_text)
        except ValueError as exc:
            raise ValueError(f"invalid blind candidate step: {candidate_id}") from exc
        if step not in ALLOWED_STEPS:
            raise ValueError(f"blind candidate is outside Best3 steps: {candidate_id}")
        expected_completion, expected_metrics = default_full320_paths(
            project_root, {step}
        )[step]
        completion_path = require_registered_file(
            value=evidence.get("completion_json"),
            expected_path=expected_completion,
            expected_sha256=evidence.get("completion_sha256"),
            label=f"BLIND20_READY {candidate_id} COMPLETED.json",
        )
        metrics_path = require_registered_file(
            value=evidence.get("paired_metrics_json"),
            expected_path=expected_metrics,
            expected_sha256=evidence.get("paired_metrics_sha256"),
            label=f"BLIND20_READY {candidate_id} paired_metrics.json",
        )
        completion, indexed = validate_full320_step(
            step=step,
            completion_path=completion_path,
            metrics_path=metrics_path,
            project_root=project_root,
        )
        dual_path = require_registered_file(
            value=evidence.get("dual_encoder_cases_csv"),
            expected_path=Path(str(completion["dual_encoder_cases_csv"])),
            expected_sha256=evidence.get("dual_encoder_cases_sha256"),
            label=f"BLIND20_READY {candidate_id} dual_encoder_cases.csv",
        )
        expected_run = (
            f"ver2_9_5_final_{arm}_step-{step}_no_text_seedtts160_d2d3_seed1234"
        )
        with dual_path.open(encoding="utf-8", newline="") as handle:
            lane_rows = [
                row for row in csv.DictReader(handle)
                if str(row.get("run") or row.get("run_id") or "") == expected_run
                and str(row.get("mode") or "") == "no_text"
            ]
        lane_by_case = {str(row.get("case_id") or ""): row for row in lane_rows}
        if len(lane_rows) != 160 or len(lane_by_case) != 160 or "" in lane_by_case:
            raise ValueError(
                f"BLIND20_READY {candidate_id} diagnostics is not the registered no_text160 lane"
            )
        expected_target_root = (
            Path(str(completion["step_root"])) / "runs" / expected_run
        ).resolve()
        expected_scalars = {
            "candidate_id": candidate_id,
            "arm": arm,
            "step": step,
            "run_id": expected_run,
            "target_audio_root": str(expected_target_root),
            "dual_encoder_cases_csv": str(dual_path),
        }
        bad = {
            key: evidence.get(key)
            for key, expected in expected_scalars.items()
            if evidence.get(key) != expected
        }
        if bad:
            raise ValueError(f"BLIND20_READY {candidate_id} full320 identity drift: {bad}")
        if evidence.get("objective_no_text") != indexed[(arm, "no_text")]:
            raise ValueError(f"BLIND20_READY {candidate_id} objective row drift")
        validated_evidence[candidate_id] = {
            **evidence,
            "dual_path": dual_path,
            "target_root": expected_target_root,
            "lane_by_case": lane_by_case,
        }
    pages = {
        str(row.get("candidate_id")): row
        for row in payload.get("pages", [])
        if isinstance(row, dict)
    }
    if set(pages) != set(selected):
        raise ValueError("BLIND20_READY pages do not exactly match Best3")
    for candidate_id, page in pages.items():
        manifest_path = require_registered_file(
            value=page.get("manifest"),
            expected_path=None,
            expected_sha256=page.get("manifest_sha256"),
            label=f"BLIND20_READY {candidate_id} private manifest",
        )
        require_registered_file(
            value=page.get("index"),
            expected_path=None,
            expected_sha256=page.get("index_sha256"),
            label=f"BLIND20_READY {candidate_id} listening page",
        )
        manifest = load_json(manifest_path)
        evidence = validated_evidence[candidate_id]
        if (
            manifest.get("schema_version") != BLIND_SCHEMA
            or manifest.get("candidate_id") != candidate_id
            or manifest.get("page_id") != page.get("page_id")
            or manifest.get("candidate_run_id") != evidence["run_id"]
            or Path(str(manifest.get("candidate_diagnostics_csv") or "")).resolve()
            != evidence["dual_path"]
            or manifest.get("candidate_diagnostics_sha256")
            != evidence["dual_encoder_cases_sha256"]
            or manifest.get("candidate_evidence") != evidence_by_candidate[candidate_id]
            or manifest.get("producer") != expected_blind_producer
        ):
            raise ValueError(f"invalid private blind manifest: {manifest_path}")
        manifest_ids = [str(row.get("case_id") or "") for row in manifest.get("cases", [])]
        if manifest_ids != [str(item) for item in case_ids_by_candidate[candidate_id]]:
            raise ValueError(f"BLIND20_READY/manifest case order drift: {candidate_id}")
        for row in manifest.get("cases", []):
            case_id = str(row.get("case_id") or "")
            candidate_row = evidence["lane_by_case"].get(case_id)
            batch33_row = batch33_by_case.get(case_id)
            if not isinstance(candidate_row, dict) or not isinstance(batch33_row, dict):
                raise ValueError(f"{manifest_path}: {case_id} absent from bound diagnostics")
            mapping = row.get("mapping")
            if not isinstance(mapping, dict) or set(mapping) != {"A", "B"}:
                raise ValueError(f"{manifest_path}: {case_id} invalid A/B mapping")
            roles = {str(item.get("role") or "") for item in mapping.values() if isinstance(item, dict)}
            if roles != {"candidate", "batch33"}:
                raise ValueError(f"{manifest_path}: {case_id} invalid A/B roles")
            for letter, item in mapping.items():
                if not isinstance(item, dict):
                    raise ValueError(f"{manifest_path}: {case_id}/{letter} mapping is invalid")
                audio = require_registered_file(
                    value=item.get("audio"),
                    expected_path=None,
                    expected_sha256=item.get("audio_sha256"),
                    label=f"{manifest_path}:{case_id}/{letter} audio",
                )
                if item.get("role") == "candidate":
                    expected_audio = (evidence["target_root"] / f"{case_id}.wav").resolve()
                    if audio != expected_audio:
                        raise ValueError(
                            f"{manifest_path}: {case_id} candidate audio={audio}, "
                            f"expected registered full320 target {expected_audio}"
                        )
                    if audio != Path(str(candidate_row.get("target_audio") or "")).resolve():
                        raise ValueError(
                            f"{manifest_path}: {case_id} candidate mapping differs from dual CSV"
                        )
                else:
                    expected_audio = Path(
                        str(batch33_row.get("target_audio") or "")
                    ).resolve()
                    if audio != expected_audio:
                        raise ValueError(
                            f"{manifest_path}: {case_id} Batch-33 mapping differs from diagnostics"
                        )
            for anchor in ("source", "reference"):
                item = row.get(anchor)
                if not isinstance(item, dict):
                    raise ValueError(f"{manifest_path}: {case_id} missing {anchor} registration")
                audio = require_registered_file(
                    value=item.get("audio"),
                    expected_path=None,
                    expected_sha256=item.get("audio_sha256"),
                    label=f"{manifest_path}:{case_id}/{anchor} audio",
                )
                field = "source_audio" if anchor == "source" else "timbre_ref_audio"
                candidate_anchor = Path(str(candidate_row.get(field) or "")).resolve()
                baseline_anchor = Path(str(batch33_row.get(field) or "")).resolve()
                if audio != candidate_anchor or audio != baseline_anchor:
                    raise ValueError(
                        f"{manifest_path}: {case_id} {anchor} differs from bound diagnostics"
                    )
    return payload, pages


def decode_review(
    *, candidate_id: str, page: Mapping[str, Any], review_path: Path
) -> dict[str, Any]:
    manifest_path = Path(str(page.get("manifest") or "")).resolve()
    manifest = load_json(manifest_path)
    review = load_json(review_path)
    if manifest.get("schema_version") != BLIND_SCHEMA:
        raise ValueError(f"{manifest_path}: wrong blind manifest schema")
    page_id = manifest.get("page_id")
    if review.get("page_id") != page_id:
        raise ValueError(f"{review_path}: page identity mismatch")
    if review.get("candidate_id") not in (None, "", candidate_id):
        raise ValueError(f"{review_path}: unexpected leaked/wrong candidate identity")
    if review.get("complete") is not True:
        raise ValueError(f"{review_path}: review is not complete")
    private = {row["case_id"]: row for row in manifest.get("cases", [])}
    items = review.get("items")
    if not isinstance(items, list) or len(items) != 20:
        raise ValueError(f"{review_path}: expected 20 review items")
    if len(private) != 20:
        raise ValueError(f"{manifest_path}: expected 20 private cases")
    counts: Counter[str] = Counter()
    decoded = []
    seen: set[str] = set()
    for item in items:
        case_id = str(item.get("case_id") or "")
        judgment = str(item.get("judgment") or "")
        if case_id in seen or case_id not in private:
            raise ValueError(f"{review_path}: duplicate/unknown case {case_id!r}")
        seen.add(case_id)
        if judgment in {"A", "B"}:
            outcome = private[case_id]["mapping"][judgment]["role"]
            if outcome not in {"candidate", "batch33"}:
                raise ValueError(f"{manifest_path}: invalid private role {outcome!r}")
        elif judgment in {"tie", "neither"}:
            outcome = judgment
        else:
            raise ValueError(f"{review_path}: invalid judgment {judgment!r}")
        counts[outcome] += 1
        decoded.append({"case_id": case_id, "anonymous_judgment": judgment, "outcome": outcome})
    if seen != set(private):
        raise ValueError(f"{review_path}: case set differs from private manifest")
    return {
        "candidate_id": candidate_id,
        "review_json": str(review_path),
        "review_sha256": sha256_file(review_path),
        "manifest_json": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "counts": {
            "candidate_wins": counts["candidate"],
            "batch33_wins": counts["batch33"],
            "tie": counts["tie"],
            "neither": counts["neither"],
        },
        "subjective_net": counts["candidate"] - counts["batch33"],
        "decoded_items": decoded,
    }


def recommendation_key(row: Mapping[str, Any], objective: Mapping[str, Any]) -> tuple[Any, ...]:
    counts = row["counts"]
    return (
        -int(row["subjective_net"]),
        -int(counts["candidate_wins"]),
        int(counts["batch33_wins"]),
        -float(objective["all"]["wavlm_sim_ref"]),
        str(row["candidate_id"]),
    )


def evaluate_objective_gate(objective: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    no_text = objective["no_text"]
    text = objective["text"]
    checks = {
        "no_text_cer_le_0p08": float(no_text["cer"]) <= 0.08,
        "no_text_wavlm_sim_ref_ge_0p45": float(no_text["wavlm_sim_ref"]) >= 0.45,
        "no_text_wavlm_margin_ge_0p02": float(no_text["wavlm_margin"]) >= 0.02,
        "no_text_ref_content_f1_le_0p20": float(no_text["ref_content_lcs_f1"]) <= 0.20,
        "text_cer_le_0p05": float(text["cer"]) <= 0.05,
        "text_en_src_fail_le_0p10": float(text["text_en_src_fail_rate"]) <= 0.10,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "source": "Batch-44 v1 registered 30k gates plus the SIM-margin/ref-content red flags",
    }


def choose_auto_winner(
    ranked: Sequence[Mapping[str, Any]], objective_gates: Mapping[str, Mapping[str, Any]]
) -> str | None:
    eligible = [
        row
        for row in ranked
        if objective_gates[str(row["candidate_id"])]["pass"]
        and int(row["subjective_net"]) >= 0
    ]
    if not eligible:
        return None
    if len(eligible) > 1 and int(eligible[0]["subjective_net"]) <= int(eligible[1]["subjective_net"]):
        return None
    return str(eligible[0]["candidate_id"])


def hash_checkpoint(
    candidate: Mapping[str, Any], *, project_root: Path = PROJECT_ROOT
) -> tuple[Path, dict[str, dict[str, Any]]]:
    arm, step = str(candidate["arm"]), int(candidate["step"])
    checkpoint = (
        project_root / "outputs/lora_runs" / RUN_DIRS[arm] / f"step-{step}"
    ).resolve()
    if Path(str(candidate["checkpoint"]["path"])).resolve() != checkpoint:
        raise ValueError(f"Best3 checkpoint path drift for {arm}_step-{step}")
    files = {}
    for name in MODEL_FILES:
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
        files[name] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    return checkpoint, files


def producer_registration(project_root: Path = PROJECT_ROOT) -> dict[str, str]:
    producer = (project_root / "scripts/004107_finalize_batch43_pathx_final.py").resolve()
    if not producer.is_file():
        # A frozen copy may call the verifier, but the registration remains the
        # canonical source path whose bytes must match the frozen verifier.
        producer = Path(__file__).resolve()
    return {
        "script": str(producer),
        "script_sha256": sha256_file(producer),
        "entrypoint": "004107_finalize_batch43_pathx_final.py",
    }


def validate_final_selection_provenance(
    final_path: Path,
    *,
    payload: Mapping[str, Any] | None = None,
    project_root: Path = PROJECT_ROOT,
    verify_checkpoint_hashes: bool = True,
) -> dict[str, Any]:
    """Replay the complete 004103->004104->004107 final decision chain."""
    final_path = final_path.expanduser().resolve()
    final = dict(payload) if payload is not None else load_json(final_path)
    expected_scalars = {
        "schema_version": FINAL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "data_version": "v1_20260709",
        "status": "final",
        "system_id": "path_x_final",
    }
    bad = {
        key: final.get(key)
        for key, expected in expected_scalars.items()
        if final.get(key) != expected
    }
    if bad:
        raise ValueError(f"invalid Batch-44 v1 FINAL_SELECTION scalars: {bad}")
    producer = final.get("producer")
    expected_producer = producer_registration(project_root)
    if producer != expected_producer:
        raise ValueError(
            "FINAL_SELECTION is not bound to the current registered 004107 producer"
        )

    report_path = require_registered_file(
        value=final.get("decision_report"),
        expected_path=None,
        expected_sha256=final.get("decision_report_sha256"),
        label="FINAL_SELECTION decision report",
    )
    report = load_json(report_path)
    if (
        report.get("schema_version") != DECISION_SCHEMA
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("data_version") != "v1_20260709"
        or report.get("status") != "finalized"
        or report.get("producer") != expected_producer
    ):
        raise ValueError("FINAL_SELECTION decision report is not a finalized 004107 report")
    best3_path = require_registered_file(
        value=report.get("best3_selection"),
        expected_path=None,
        expected_sha256=report.get("best3_selection_sha256"),
        label="decision report Best3 selection",
    )
    best3, selected, candidate_rows = load_best3(
        best3_path, project_root=project_root, replay_quick20=True
    )
    blind_path = require_registered_file(
        value=report.get("blind_ready"),
        expected_path=None,
        expected_sha256=report.get("blind_ready_sha256"),
        label="decision report BLIND20_READY",
    )
    blind, pages = load_blind_ready(
        blind_path,
        selected,
        best3_path=best3_path,
        project_root=project_root,
    )
    decision = report.get("finalization_decision")
    candidate = final.get("candidate")
    if not isinstance(decision, dict) or not isinstance(candidate, dict):
        raise ValueError("FINAL_SELECTION lacks candidate/finalization decision")
    winner = str(candidate.get("candidate_id") or "")
    method = str(final.get("selection_method") or "")
    if (
        winner not in selected
        or decision.get("winner_candidate_id") != winner
        or decision.get("selection_method") != method
    ):
        raise ValueError("FINAL_SELECTION winner differs from finalized decision report")
    if method == "auto_unique_subjective_net" and report.get("auto_winner") != winner:
        raise ValueError("FINAL_SELECTION auto winner differs from decision report")
    if method not in {"auto_unique_subjective_net", "explicit_human_selection"}:
        raise ValueError(f"invalid FINAL_SELECTION method: {method!r}")

    full_provenance = report.get("full320_provenance")
    objective = report.get("full320_objective")
    gates = report.get("objective_gates")
    if not isinstance(full_provenance, dict) or not isinstance(objective, dict) or not isinstance(gates, dict):
        raise ValueError("decision report lacks full320 evidence/gates")
    validated_by_step: dict[int, dict[tuple[str, str], dict[str, Any]]] = {}
    for step in {int(candidate_rows[item]["step"]) for item in selected}:
        registration = full_provenance.get(str(step))
        if not isinstance(registration, dict):
            raise ValueError(f"decision report lacks full320 provenance for step-{step}")
        defaults = default_full320_paths(project_root, {step})[step]
        completion_path = require_registered_file(
            value=registration.get("completion_json"),
            expected_path=defaults[0],
            expected_sha256=registration.get("completion_sha256"),
            label=f"decision report step-{step} COMPLETED.json",
        )
        metrics_path = require_registered_file(
            value=registration.get("metrics_json"),
            expected_path=defaults[1],
            expected_sha256=registration.get("metrics_sha256"),
            label=f"decision report step-{step} paired_metrics.json",
        )
        completion, indexed = validate_full320_step(
            step=step,
            completion_path=completion_path,
            metrics_path=metrics_path,
            project_root=project_root,
        )
        require_registered_file(
            value=registration.get("dual_encoder_cases_csv"),
            expected_path=Path(str(completion["dual_encoder_cases_csv"])),
            expected_sha256=registration.get("dual_encoder_cases_sha256"),
            label=f"decision report step-{step} dual_encoder_cases.csv",
        )
        validated_by_step[step] = indexed

    decoded_by_candidate: dict[str, dict[str, Any]] = {}
    subjective_rows = report.get("subjective")
    if not isinstance(subjective_rows, list) or len(subjective_rows) != 3:
        raise ValueError("decision report must contain three subjective results")
    reported_subjective = {
        str(row.get("candidate_id") or ""): row
        for row in subjective_rows if isinstance(row, dict)
    }
    if set(reported_subjective) != set(selected):
        raise ValueError("decision report subjective candidates differ from Best3")
    for candidate_id in selected:
        row = candidate_rows[candidate_id]
        arm, step = str(row["arm"]), int(row["step"])
        expected_objective = {
            scope: validated_by_step[step][(arm, scope)]
            for scope in ("no_text", "text", "all")
        }
        if objective.get(candidate_id) != expected_objective:
            raise ValueError(f"decision report objective drift: {candidate_id}")
        recomputed_gate = evaluate_objective_gate(expected_objective)
        if gates.get(candidate_id) != recomputed_gate:
            raise ValueError(f"decision report objective gate drift: {candidate_id}")
        review_path = Path(str(reported_subjective[candidate_id].get("review_json") or "")).resolve()
        decoded = decode_review(
            candidate_id=candidate_id,
            page=pages[candidate_id],
            review_path=review_path,
        )
        decoded["rank"] = reported_subjective[candidate_id].get("rank")
        if decoded != reported_subjective[candidate_id]:
            raise ValueError(f"decision report subjective decode drift: {candidate_id}")
        decoded_by_candidate[candidate_id] = decoded
    ranked = sorted(
        decoded_by_candidate.values(),
        key=lambda row: recommendation_key(row, objective[row["candidate_id"]]),
    )
    for rank, row in enumerate(ranked, start=1):
        if row.get("rank") != rank:
            raise ValueError("decision report subjective ranking drift")
    if choose_auto_winner(ranked, gates) != report.get("auto_winner"):
        raise ValueError("decision report auto-winner calculation drift")
    if gates.get(winner, {}).get("pass") is not True:
        raise ValueError("FINAL_SELECTION winner fails recomputed objective gate")
    if final.get("full320_metrics") != objective[winner]:
        raise ValueError("FINAL_SELECTION full320 metrics differ from decision report")
    if final.get("subjective_result") != reported_subjective[winner]:
        raise ValueError("FINAL_SELECTION subjective result differs from decision report")

    winner_row = candidate_rows[winner]
    arm, step = str(winner_row["arm"]), int(winner_row["step"])
    expected_identity = {
        "candidate_id": winner,
        "arm": arm,
        "text_repeat": EXPECTED_REPEATS[arm],
        "step": step,
        "train_job_id": EXPECTED_JOBS[arm],
    }
    identity_bad = {
        key: candidate.get(key)
        for key, expected in expected_identity.items()
        if candidate.get(key) != expected
    }
    if identity_bad:
        raise ValueError(f"FINAL_SELECTION candidate identity drift: {identity_bad}")
    checkpoint, actual_files = hash_checkpoint(winner_row, project_root=project_root)
    if Path(str(candidate.get("checkpoint_path") or "")).resolve() != checkpoint:
        raise ValueError("FINAL_SELECTION checkpoint differs from Best3 winner")
    registered_files = candidate.get("model_files")
    if not isinstance(registered_files, dict) or set(registered_files) != set(MODEL_FILES):
        raise ValueError("FINAL_SELECTION model file registration is incomplete")
    for name in MODEL_FILES:
        registration = registered_files[name]
        if registration.get("size") != actual_files[name]["size"]:
            raise ValueError(f"FINAL_SELECTION checkpoint size drift: {name}")
        if verify_checkpoint_hashes and registration.get("sha256") != actual_files[name]["sha256"]:
            raise ValueError(f"FINAL_SELECTION checkpoint SHA256 drift: {name}")

    winner_review = reported_subjective[winner]
    winner_step_provenance = full_provenance[str(step)]
    expected_provenance = {
        "producer": expected_producer,
        "best3_selection": str(best3_path),
        "best3_selection_sha256": sha256_file(best3_path),
        "blind_ready": str(blind_path),
        "blind_ready_sha256": sha256_file(blind_path),
        "decision_report": str(report_path),
        "decision_report_sha256": sha256_file(report_path),
        "winner_review": winner_review["review_json"],
        "winner_review_sha256": winner_review["review_sha256"],
        "winner_manifest": winner_review["manifest_json"],
        "winner_manifest_sha256": winner_review["manifest_sha256"],
        "winner_full320": winner_step_provenance,
    }
    if final.get("provenance") != expected_provenance:
        raise ValueError("FINAL_SELECTION upstream provenance bundle drift")
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best3-selection", type=Path, default=DEFAULT_BEST3)
    parser.add_argument("--blind-ready", type=Path, default=DEFAULT_BLIND_READY)
    parser.add_argument(
        "--review", action="append", default=[], metavar="CANDIDATE=JSON",
        help="exported complete review JSON; repeat exactly three times",
    )
    parser.add_argument(
        "--completion", action="append", default=[], metavar="STEP=JSON",
        help="override a paired full320 COMPLETED.json for a selected step",
    )
    parser.add_argument(
        "--metrics", action="append", default=[], metavar="STEP=JSON",
        help="override paired_metrics.json for a selected step",
    )
    parser.add_argument(
        "--winner",
        default="",
        help="explicit Best3 candidate id, or 'auto'; empty writes report only and exits 3",
    )
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--final-json", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    best3_path = args.best3_selection.expanduser().resolve()
    blind_path = args.blind_ready.expanduser().resolve()
    best3, selected, candidate_rows = load_best3(
        best3_path, project_root=PROJECT_ROOT, replay_quick20=True
    )
    blind, pages = load_blind_ready(
        blind_path,
        selected,
        best3_path=best3_path,
        project_root=PROJECT_ROOT,
    )
    reviews = dict(parse_binding(raw, option="--review") for raw in args.review)
    if set(reviews) != set(selected):
        raise ValueError(f"review bindings must exactly match Best3: {selected}")

    steps = {int(candidate_rows[candidate_id]["step"]) for candidate_id in selected}
    defaults = default_full320_paths(PROJECT_ROOT, steps)
    completions = {int(key): path for key, path in (parse_binding(raw, option="--completion") for raw in args.completion)}
    metrics = {int(key): path for key, path in (parse_binding(raw, option="--metrics") for raw in args.metrics)}
    if set(completions) - steps or set(metrics) - steps:
        raise ValueError("full320 overrides contain a non-selected step")
    full_by_step = {}
    for step in steps:
        completion_path = completions.get(step, defaults[step][0]).resolve()
        metrics_path = metrics.get(step, defaults[step][1]).resolve()
        completion, indexed = validate_full320_step(
            step=step, completion_path=completion_path, metrics_path=metrics_path
        )
        full_by_step[step] = {
            "completion": completion,
            "completion_path": completion_path,
            "metrics_path": metrics_path,
            "metrics": indexed,
        }

    objective: dict[str, Any] = {}
    objective_gates: dict[str, Any] = {}
    subjective = []
    for candidate_id in selected:
        candidate = candidate_rows[candidate_id]
        arm, step = str(candidate["arm"]), int(candidate["step"])
        indexed = full_by_step[step]["metrics"]
        objective[candidate_id] = {
            scope: indexed[(arm, scope)] for scope in ("no_text", "text", "all")
        }
        objective_gates[candidate_id] = evaluate_objective_gate(objective[candidate_id])
        subjective.append(
            decode_review(
                candidate_id=candidate_id,
                page=pages[candidate_id],
                review_path=reviews[candidate_id],
            )
        )
    ranked = sorted(
        subjective,
        key=lambda row: recommendation_key(row, objective[row["candidate_id"]]),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    auto_winner = choose_auto_winner(ranked, objective_gates)
    producer = producer_registration(PROJECT_ROOT)
    report = {
        "schema_version": DECISION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "data_version": "v1_20260709",
        "status": "winner_ready" if auto_winner else "human_decision_required",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "producer": producer,
        "best3_selection": str(best3_path),
        "best3_selection_sha256": sha256_file(best3_path),
        "blind_ready": str(blind_path),
        "blind_ready_sha256": sha256_file(blind_path),
        "subjective_ranking_rule": (
            "candidate_wins - Batch33_wins descending; then candidate wins descending; "
            "Batch33 wins ascending; full320 all WavLM SIM(ref) descending"
        ),
        "auto_winner_requires": (
            "registered full320 objective gate pass, subjective_net >= 0 versus Batch-33, "
            "and a unique best subjective_net among eligible candidates"
        ),
        "auto_winner": auto_winner,
        "subjective": ranked,
        "full320_objective": objective,
        "objective_gates": objective_gates,
        "full320_provenance": {
            str(step): {
                "completion_json": str(value["completion_path"]),
                "completion_sha256": sha256_file(value["completion_path"]),
                "metrics_json": str(value["metrics_path"]),
                "metrics_sha256": sha256_file(value["metrics_path"]),
                "dual_encoder_cases_csv": str(
                    Path(str(value["completion"]["dual_encoder_cases_csv"])).resolve()
                ),
                "dual_encoder_cases_sha256": sha256_file(
                    Path(str(value["completion"]["dual_encoder_cases_csv"])).resolve()
                ),
            }
            for step, value in full_by_step.items()
        },
    }
    atomic_json(args.report_json.expanduser().resolve(), report)

    if not args.winner:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 3
    if args.winner == "auto":
        if auto_winner is None:
            raise ValueError("auto winner is not unique; provide an explicit --winner")
        winner = auto_winner
        method = "auto_unique_subjective_net"
    else:
        winner = args.winner
        method = "explicit_human_selection"
    if winner not in selected:
        raise ValueError(f"winner {winner!r} is not in Best3 {selected}")
    if objective_gates[winner]["pass"] is not True:
        raise ValueError(
            f"winner {winner!r} fails the registered full320 objective gate; "
            "do not publish it as path_x_final"
        )
    report["status"] = "finalized"
    report["finalization_decision"] = {
        "winner_candidate_id": winner,
        "selection_method": method,
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.report_json.expanduser().resolve(), report)
    final_path = args.final_json.expanduser().resolve()
    if final_path.exists() and not args.force:
        raise FileExistsError(f"final selection exists (use --force only intentionally): {final_path}")
    winner_row = candidate_rows[winner]
    checkpoint, model_files = hash_checkpoint(winner_row, project_root=PROJECT_ROOT)
    arm, step = str(winner_row["arm"]), int(winner_row["step"])
    winner_subjective = next(row for row in ranked if row["candidate_id"] == winner)
    winner_full320 = report["full320_provenance"][str(step)]
    report_path = args.report_json.expanduser().resolve()
    final = {
        "schema_version": FINAL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "data_version": "v1_20260709",
        "status": "final",
        "system_id": "path_x_final",
        "display_name": "ver2.9.5-final (ours 30k)",
        "version_name": f"ver2.9.5-final-{arm}",
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_method": method,
        "producer": producer,
        "candidate": {
            "candidate_id": winner,
            "arm": arm,
            "text_repeat": EXPECTED_REPEATS[arm],
            "step": step,
            "train_job_id": EXPECTED_JOBS[arm],
            "checkpoint_path": str(checkpoint),
            "model_files": model_files,
        },
        "full320_metrics": objective[winner],
        "subjective_result": winner_subjective,
        "decision_report": str(report_path),
        "decision_report_sha256": sha256_file(report_path),
        "provenance": {
            "producer": producer,
            "best3_selection": str(best3_path),
            "best3_selection_sha256": sha256_file(best3_path),
            "blind_ready": str(blind_path),
            "blind_ready_sha256": sha256_file(blind_path),
            "decision_report": str(report_path),
            "decision_report_sha256": sha256_file(report_path),
            "winner_review": winner_subjective["review_json"],
            "winner_review_sha256": winner_subjective["review_sha256"],
            "winner_manifest": winner_subjective["manifest_json"],
            "winner_manifest_sha256": winner_subjective["manifest_sha256"],
            "winner_full320": winner_full320,
        },
        "next_protocol": {
            "strict_inference_system_id": "path_x_final",
            "datasets": ["EN567", "ZH1194"],
            "scorers": ["WavLM-large-SV", "ERes2Net", "SpeechBrain ECAPA"],
            "asr": {"en": "Whisper-large-v3", "zh": "Paraformer-zh"},
        },
    }
    atomic_json(final_path, final)
    validate_final_selection_provenance(
        final_path,
        payload=final,
        project_root=PROJECT_ROOT,
        verify_checkpoint_hashes=True,
    )
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

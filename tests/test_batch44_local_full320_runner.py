from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUNNER = ROOT / "scripts/004118_run_batch44_v1_paired_full320_local.sh"
ENGINE = ROOT / "scripts/004112_submit_batch44_v1_paired_full320_qz.sh"
VALIDATOR = ROOT / "scripts/004107_finalize_batch43_pathx_final.py"


def load_validator():
    spec = importlib.util.spec_from_file_location("batch44_local_full320_validator", VALIDATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_rows(module, step: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arm in ("r3", "r5"):
        for scope, n, fail in (("no_text", 160, 8), ("text", 160, 8), ("all", 320, 16)):
            has_text = scope != "no_text"
            rows.append(
                {
                    "step": step,
                    "arm": arm,
                    "text_repeat": module.EXPECTED_REPEATS[arm],
                    "train_job_id": module.EXPECTED_JOBS[arm],
                    "scope": scope,
                    "n": n,
                    "keep": n - fail,
                    "fail_count": fail,
                    "fail_rate": fail / n,
                    "cer": 0.05,
                    "wavlm_sim_ref": 0.45,
                    "wavlm_sim_src": 0.39,
                    "wavlm_margin": 0.06,
                    "wavlm_ref_bound": 0.55,
                    "speechbrain_sim_ref": 0.50,
                    "speechbrain_sim_src": 0.30,
                    "speechbrain_margin": 0.20,
                    "speechbrain_ref_bound": 0.70,
                    "ref_content_lcs_f1": 0.04,
                    "text_en_src_n": 80 if has_text else "",
                    "text_en_src_fail_count": 4 if has_text else "",
                    "text_en_src_fail_rate": 0.05 if has_text else "",
                    "text_en_src_cer": 0.04 if has_text else "",
                }
            )
    return rows


def test_local_runner_is_plan_only_by_default_and_has_no_submit_surface(tmp_path: Path) -> None:
    for script in (LOCAL_RUNNER, ENGINE):
        result = subprocess.run(
            ["bash", "-n", str(script)], text=True, capture_output=True, check=False
        )
        assert result.returncode == 0, result.stderr

    local_source = LOCAL_RUNNER.read_text(encoding="utf-8")
    engine_source = ENGINE.read_text(encoding="utf-8")
    assert 'ACTION="${ACTION:-plan}"' in local_source
    assert "CONFIRM_LOCAL_FULL320" in local_source
    assert "create-job" not in local_source
    assert "qzcli_with_deps" not in local_source
    assert "QZCLI" not in local_source
    assert "submitted_jobs.tsv" in local_source  # only a fail-closed prohibition
    assert "local full320 record must not contain" in engine_source
    for lane in (
        "run_eval_lane r3 no_text 0,1",
        "run_eval_lane r3 text 0,1",
        "run_eval_lane r5 no_text 0,1",
        "run_eval_lane r5 text 0,1",
    ):
        assert lane in engine_source
    assert '"backend": "local"' in engine_source
    assert '"lane_execution": "sequential"' in engine_source

    project = tmp_path / "MOSS-CodecVC"
    result = subprocess.run(
        ["bash", str(LOCAL_RUNNER)],
        cwd=ROOT,
        env={
            **os.environ,
            "PROJECT_ROOT": str(project),
            "BATCH44_LOCAL_FULL320_TEST_MODE": "1",
            "STEP": "10000",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert "plan complete; no GPU work was started" in result.stdout
    assert "BACKEND=local" in result.stdout
    assert "LANE_EXECUTION=sequential" in result.stdout
    assert not (project / "trainset/local_jobs").exists()


def test_local_completion_and_strict_validator_bind_dual_4090_provenance(
    tmp_path: Path,
) -> None:
    module = load_validator()
    step = 26000
    project = tmp_path / "MOSS-CodecVC"
    record = (
        project
        / f"trainset/local_jobs/ver23_batch44_paired_full320_step{step}_20260713"
    )
    step_root = (
        project
        / f"testset/outputs/ver23_batch44_paired_full320_20260713/step-{step}"
    )
    aggregate = step_root / "aggregate"
    aggregate.mkdir(parents=True)
    record.mkdir(parents=True)

    metrics_path = aggregate / "paired_metrics.json"
    metrics_path.write_text(json.dumps(metric_rows(module, step)) + "\n", encoding="utf-8")
    (aggregate / "paired_metrics.tsv").write_text("fixture\n", encoding="utf-8")
    (aggregate / "paired_metrics.md").write_text("fixture\n", encoding="utf-8")

    completeness = {
        "lanes": [
            {
                "arm": arm,
                "mode": mode,
                "run_id": f"ver2_9_5_final_{arm}_step-{step}_{mode}_seedtts160_d2d3_seed1234",
                "checkpoint": str(
                    project
                    / "outputs/lora_runs"
                    / module.RUN_DIRS[arm]
                    / f"step-{step}"
                ),
                "training_job_id": module.EXPECTED_JOBS[arm],
                "rows": 160,
                "asr_rows": 160,
                "bnf_extraction_lines": 160 if mode == "no_text" else 0,
            }
            for arm in ("r3", "r5")
            for mode in ("no_text", "text")
        ]
    }
    (aggregate / "completeness.json").write_text(
        json.dumps(completeness) + "\n", encoding="utf-8"
    )
    dual_path = aggregate / "dual_encoder_cases.csv"
    with dual_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run",
                "mode",
                "case_id",
                "sim_gen_ref",
                "sim_gen_source",
                "ecapa_sim_gen_ref",
                "ecapa_sim_gen_source",
                "cer_tgt",
            ],
        )
        writer.writeheader()
        for arm in ("r3", "r5"):
            for mode in ("no_text", "text"):
                run = f"ver2_9_5_final_{arm}_step-{step}_{mode}_seedtts160_d2d3_seed1234"
                for index in range(160):
                    writer.writerow(
                        {
                            "run": run,
                            "mode": mode,
                            "case_id": f"case-{index:03d}",
                            "sim_gen_ref": 0.45,
                            "sim_gen_source": 0.39,
                            "ecapa_sim_gen_ref": 0.50,
                            "ecapa_sim_gen_source": 0.30,
                            "cer_tgt": 0.05,
                        }
                    )

    validation = project / "testset/validation/seedtts_vc_ver2_3_validation.jsonl"
    validation.parent.mkdir(parents=True)
    validation.write_text("{}\n", encoding="utf-8")
    train_ledger = (
        project
        / "trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713/submitted_pair.tsv"
    )
    train_ledger.parent.mkdir(parents=True)
    train_ledger.write_text("fixture\n", encoding="utf-8")

    runner = record / "004118_run_batch44_v1_paired_full320_local.frozen.sh"
    engine = record / "004112_batch44_v1_paired_full320_engine.frozen.sh"
    inputs = record / "frozen_inputs.sha256"
    runner.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    engine.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    inputs.write_text("fixture\n", encoding="utf-8")
    (record / "resolved_runs.tsv").write_text("fixture\n", encoding="utf-8")

    required_checkpoint_files = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "README.md",
        "timbre_memory_adapter.pt",
        "timbre_memory_config.json",
    )
    for arm in ("r3", "r5"):
        checkpoint = (
            project
            / "outputs/lora_runs"
            / module.RUN_DIRS[arm]
            / f"step-{step}"
        )
        checkpoint.mkdir(parents=True)
        for name in required_checkpoint_files:
            (checkpoint / name).write_bytes(f"{arm}:{name}\n".encode())
        inventory = {
            name: {
                "bytes": (checkpoint / name).stat().st_size,
                "sha256": sha256(checkpoint / name),
            }
            for name in required_checkpoint_files
        }
        (record / f"checkpoint_{arm}_step{step}.json").write_text(
            json.dumps(
                {
                    "arm": arm,
                    "step": step,
                    "text_repeat": module.EXPECTED_REPEATS[arm],
                    "training_job_id": module.EXPECTED_JOBS[arm],
                    "checkpoint": str(checkpoint),
                    "checkpoint_inventory": inventory,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    host = socket.gethostname()
    assert host.startswith("xyzhang-dev--")
    uuids = [
        "GPU-11111111-1111-1111-1111-111111111111",
        "GPU-22222222-2222-2222-2222-222222222222",
    ]
    inventory = record / "runtime_gpu_inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema": "batch44_local_gpu_inventory_v1",
                "captured_utc": "2026-07-13T00:00:00+00:00",
                "hostname": host,
                "gpus": [
                    {
                        "index": index,
                        "uuid": uuids[index],
                        "name": "NVIDIA GeForce RTX 4090",
                        "memory_total_mib": 49140,
                        "driver_version": "550.163.01",
                    }
                    for index in (0, 1)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    source = ENGINE.read_text(encoding="utf-8")
    match = re.search(
        r"write_local_completion\(\) \{.*?<<'PY'\n(?P<body>.*?)\nPY\n\}",
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    completion_path = record / "COMPLETED.json"
    result = subprocess.run(
        [
            sys.executable,
            "-",
            str(completion_path),
            str(step),
            str(record),
            str(step_root),
            str(module.EXPECTED_CODE_ROOT),
            str(validation),
            str(train_ledger),
            module.EXPECTED_JOBS["r3"],
            module.EXPECTED_JOBS["r5"],
            str(runner),
            str(engine),
            str(inputs),
            str(inventory),
            "2026-07-13T00:00:00+00:00",
        ],
        input=match.group("body"),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert completion["backend"] == "local"
    assert completion["lane_execution"] == "sequential"
    assert completion["execution"]["gpu_indices"] == [0, 1]
    assert completion["execution"]["gpu_memory_total_mib"] == [49140, 49140]
    assert set(completion["artifacts"]) == {
        "completeness_json",
        "dual_encoder_cases_csv",
        "paired_metrics_json",
        "paired_metrics_tsv",
        "paired_metrics_md",
    }
    assert not (record / "submitted_jobs.tsv").exists()
    assert (record / "complete.marker").read_text(encoding="utf-8") == (
        f"COMPLETED.json sha256\t{sha256(completion_path)}\n"
    )

    selected_completion, indexed = module.validate_full320_step(
        step=step,
        completion_path=completion_path,
        metrics_path=metrics_path,
        project_root=project,
    )
    assert selected_completion["backend"] == "local"
    assert len(indexed) == 6
    assert module.default_full320_paths(project, {step})[step][0] == completion_path

    runner.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runner SHA256 drift"):
        module.validate_full320_step(
            step=step,
            completion_path=completion_path,
            metrics_path=metrics_path,
            project_root=project,
        )

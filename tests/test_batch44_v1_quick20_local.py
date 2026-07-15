from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/004117_run_batch44_v1_quick20_local.sh"
COMMON = ROOT / "scripts/004110_submit_batch44_v1_quick20_qz.sh"
HELPER_PATH = ROOT / "scripts/batch44_quick20_local_completion.py"
SPEC = importlib.util.spec_from_file_location("batch44_local_completion", HELPER_PATH)
assert SPEC is not None and SPEC.loader is not None
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, text: str = "fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_completion_fixture(tmp_path: Path) -> dict[str, object]:
    project = tmp_path / "MOSS-CodecVC"
    record = (
        project
        / "trainset/qz_jobs/ver23_batch44_quick20_step8000_20260713"
    )
    eval_root = project / "testset/outputs/ver23_batch44_quick20_20260713"
    code_root = tmp_path / "snapshot"
    record.mkdir(parents=True)
    code_root.mkdir()

    runner = write(record / "004117_run_batch44_v1_quick20_local.frozen.sh")
    common = write(record / "004110_batch44_quick20_common.frozen.sh")
    completion_helper = write(
        record / "batch44_quick20_local_completion.frozen.py"
    )
    runtime = {
        "schema": HELPER.RUNTIME_SCHEMA,
        "backend": "local",
        "status": "started",
        "started_utc": "2026-07-13T00:00:00+00:00",
        "hostname": "xyzhang-dev--pytest",
        "pid": 123,
        "gpu_count": 2,
        "gpu_indices": [0, 1],
        "gpu_model": HELPER.GPU_MODEL,
        "gpus": [
            {
                "index": index,
                "uuid": f"GPU-00000000-0000-0000-0000-00000000000{index}",
                "name": HELPER.GPU_MODEL,
                "memory_total_mib": 49140,
                "memory_used_mib_at_start": 460,
                "driver_version": "550.163.01",
            }
            for index in (0, 1)
        ],
        "max_initial_gpu_memory_mib": 2048,
        "scheduling": "four lanes sequential; each lane uses GPUs 0,1 with two shards",
        "runner": HELPER.artifact(runner),
        "common_library": HELPER.artifact(common),
        "completion_helper": HELPER.artifact(completion_helper),
    }
    checkpoints: dict[str, Path] = {}
    for arm in ("r3", "r5"):
        checkpoint = project / "outputs/lora_runs" / arm / "step-8000"
        for name in HELPER.CHECKPOINT_FILES:
            write(checkpoint / name, f"{arm}:{name}\n")
        checkpoints[arm] = checkpoint
    runtime["checkpoints"] = {
        arm: {
            "path": str(checkpoint.resolve()),
            "step": 8000,
            "training_job_id": HELPER.TRAINING_JOBS[arm],
            "files": {
                name: HELPER.artifact(checkpoint / name)
                for name in HELPER.CHECKPOINT_FILES
            },
        }
        for arm, checkpoint in checkpoints.items()
    }
    runtime_path = record / "LOCAL_RUNTIME.json"
    runtime_path.write_text(json.dumps(runtime) + "\n", encoding="utf-8")

    identity = (
        project
        / "trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713"
    )
    write(identity / "submitted_pair.tsv", "arm\tjob_id\nr3\tjob-r3\n")
    for arm in ("r3", "r5"):
        write(identity / arm / "train_args_dry_run_core.json", "{}\n")

    fixed = {
        "no_text20": write(project / "no_text20.jsonl", '{"case_id":"n"}\n'),
        "text_source": write(project / "text_source.jsonl", '{"case_id":"s"}\n'),
        "text20": write(record / "text20.jsonl", '{"case_id":"t"}\n'),
    }

    rows: list[dict[str, object]] = []
    for arm in ("r3", "r5"):
        for mode in ("no_text", "text"):
            run_id = (
                f"{HELPER.LABELS[arm]}_step-8000_{mode}_quick20_d2d3_seed1234"
            )
            output = eval_root / run_id
            for suffix, content in (
                ("summary.json", '{"overall":{"n":20}}\n'),
                ("asr_eval.jsonl", '{"case_id":"x"}\n'),
                ("speaker_sim.csv", "case_id,status\nx,ok\n"),
                ("ref_content_similarity_summary.json", '{"overall":{}}\n'),
            ):
                write(output / f"{run_id}.{suffix}", content)
            rows.append(
                {
                    "step": 8000,
                    "arm": arm,
                    "train_job_id": HELPER.TRAINING_JOBS[arm],
                    "mode": mode,
                    "n": 20,
                    "keep": 18,
                    "fail": 0.1,
                    "cer": 0.05,
                    "sim_ref": 0.45,
                    "sim_src": 0.40,
                    "margin": 0.05,
                    "ref_bound_count": 10,
                    "ref_bound": 0.5,
                    "ref_content_f1": 0.06,
                    "text_en_src_quick_n": 12 if mode == "text" else "",
                    "text_en_src_quick_fail": 0.1 if mode == "text" else "",
                    "text_en_src_scope": "proxy" if mode == "text" else "",
                    "run_id": run_id,
                    "output_dir": str(output.resolve()),
                }
            )
    (record / "metrics.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    with (record / "metrics.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    write(record / "metrics.md", "# Batch-44 local quick20\n")

    return {
        "record_root": record,
        "eval_root": eval_root,
        "project_root": project,
        "code_root": code_root,
        "step": 8000,
        "r3_checkpoint": checkpoints["r3"],
        "r5_checkpoint": checkpoints["r5"],
        "no_text20": fixed["no_text20"],
        "no_text20_sha256": sha256(fixed["no_text20"]),
        "text_source": fixed["text_source"],
        "text_source_sha256": sha256(fixed["text_source"]),
        "text20": fixed["text20"],
        "text20_sha256": sha256(fixed["text20"]),
        "runner": runner,
        "common_library": common,
        "completion_helper": completion_helper,
        "runtime_manifest": runtime_path,
    }


def test_shell_syntax_local_only_and_sequential_contract() -> None:
    for script in (RUNNER, COMMON):
        result = subprocess.run(
            ["bash", "-n", str(script)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert result.returncode == 0, result.stdout
    source = RUNNER.read_text(encoding="utf-8")
    common = COMMON.read_text(encoding="utf-8")
    assert 'BATCH44_QUICK20_LIBRARY_MODE=1' in source
    assert 'if [ "$LIBRARY_MODE" = "1" ]; then' in common
    assert "qzcli" not in source.lower()
    assert "create-job" not in source
    assert "submitted_jobs.tsv" not in source
    assert ".live_submit.lock" not in source
    assert 'run_eval "$arm" "$mode" 0,1' in source
    lanes = [
        "run_lane 1 r3 no_text",
        "run_lane 2 r3 text",
        "run_lane 3 r5 no_text",
        "run_lane 4 r5 text",
    ]
    assert [source.index(item) for item in lanes] == sorted(
        source.index(item) for item in lanes
    )
    assert 'exec >> "$RECORD_ROOT/run.local.log" 2>&1' in source
    assert 'exec > >(tee -a "$RECORD_ROOT/run.local.log")' not in source
    assert "2000|4000|6000|8000|10000" in common


def test_local_completion_binds_runtime_runner_checkpoints_and_metrics(
    tmp_path: Path,
) -> None:
    kwargs = make_completion_fixture(tmp_path)
    payload = HELPER.finalize_completion(**kwargs)
    record = Path(kwargs["record_root"])
    completion = json.loads((record / "COMPLETED.json").read_text(encoding="utf-8"))
    marker = json.loads((record / "complete.marker").read_text(encoding="utf-8"))
    assert payload == completion
    assert completion["schema"] == HELPER.COMPLETION_SCHEMA
    assert completion["backend"] == "local"
    assert completion["execution"]["hostname"] == "xyzhang-dev--pytest"
    assert completion["execution"]["gpu_indices"] == [0, 1]
    assert completion["runner"]["sha256"] == sha256(Path(kwargs["runner"]))
    assert completion["metrics"]["json"]["sha256"] == sha256(
        record / "metrics.json"
    )
    assert set(completion["checkpoints"]) == {"r3", "r5"}
    assert len(completion["runs"]) == 4
    assert marker["schema"] == HELPER.MARKER_SCHEMA
    assert marker["backend"] == "local"
    assert marker["completed_json_sha256"] == sha256(record / "COMPLETED.json")
    assert not (record / "submitted_jobs.tsv").exists()


def test_local_completion_rejects_runner_drift(tmp_path: Path) -> None:
    kwargs = make_completion_fixture(tmp_path)
    runner = Path(kwargs["runner"])
    runner.write_text("changed after capture\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runner changed after runtime capture"):
        HELPER.finalize_completion(**kwargs)
    record = Path(kwargs["record_root"])
    assert not (record / "COMPLETED.json").exists()
    assert not (record / "complete.marker").exists()


def test_local_completion_rejects_qz_submission_ledger(tmp_path: Path) -> None:
    kwargs = make_completion_fixture(tmp_path)
    record = Path(kwargs["record_root"])
    write(record / "submitted_jobs.tsv", "forged\n")
    with pytest.raises(ValueError, match="forbids submitted_jobs.tsv"):
        HELPER.finalize_completion(**kwargs)
    assert not (record / "COMPLETED.json").exists()
    assert not (record / "complete.marker").exists()


def test_local_completion_rejects_metrics_json_tsv_disagreement(
    tmp_path: Path,
) -> None:
    kwargs = make_completion_fixture(tmp_path)
    record = Path(kwargs["record_root"])
    tsv_path = record / "metrics.tsv"
    with tsv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    rows[0]["sim_ref"] = str(float(rows[0]["sim_ref"]) + 0.01)
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="metrics JSON/TSV disagree"):
        HELPER.finalize_completion(**kwargs)
    assert not (record / "COMPLETED.json").exists()
    assert not (record / "complete.marker").exists()


def test_local_dry_run_supports_step8000_without_execution_artifacts(
    tmp_path: Path,
) -> None:
    record = tmp_path / "record"
    eval_root = tmp_path / "eval"
    env = os.environ.copy()
    env.update(
        {
            "STEP": "8000",
            "DRY_RUN": "1",
            "RECORD_ROOT": str(record),
            "EVAL_ROOT": str(eval_root),
            # This test verifies identity, not workstation idleness.
            "MAX_INITIAL_GPU_MEMORY_MIB": "50000",
        }
    )
    result = subprocess.run(
        ["bash", str(RUNNER)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert "STEP=8000" in result.stdout
    assert "backend=local" in result.stdout.lower()
    assert "dry-run passed; no inference started" in result.stdout
    assert sorted(path.name for path in record.iterdir()) == [
        "ver23_batch44_text_quick20_8cell_20260713.jsonl"
    ]
    assert not eval_root.exists()


def test_existing_partial_record_fails_closed(tmp_path: Path) -> None:
    record = tmp_path / "record"
    write(record / "run.local.log", "partial\n")
    env = os.environ.copy()
    env.update(
        {
            "STEP": "8000",
            "DRY_RUN": "1",
            "RECORD_ROOT": str(record),
            "EVAL_ROOT": str(tmp_path / "eval"),
        }
    )
    result = subprocess.run(
        ["bash", str(RUNNER)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode != 0
    assert "existing/partial local quick20 artifact requires manual audit" in result.stdout
    assert not (record / "COMPLETED.json").exists()
    assert not (record / "complete.marker").exists()

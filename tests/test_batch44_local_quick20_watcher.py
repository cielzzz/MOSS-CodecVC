from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WATCHER = ROOT / "scripts/004119_watch_batch44_v1_quick20_local.sh"
VALIDATOR = ROOT / "scripts/004103_select_batch43_best3.py"
PYTHON = "/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python"
R3_JOB = "job-2b91d332-d500-4279-84f9-0a6a81a376aa"
R5_JOB = "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c"
COMPUTE = "lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"
SPEC = "67b10bc6-78b0-41a3-aaf4-358eeeb99009"
CODE_ROOT = (
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
    "MOSS-CodecVC_snapshots/ver23_batch37_eval_20260711_1092820"
)


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def write(path: Path, text: str = "fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def metrics_rows(step: int, *, negative_arm: str | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    jobs = {"r3": R3_JOB, "r5": R5_JOB}
    for arm in ("r3", "r5"):
        for mode in ("no_text", "text"):
            sim_ref = 0.45 if arm == "r3" else 0.46
            sim_src = 0.40 if mode == "no_text" else 0.28
            if arm == negative_arm and mode == "no_text":
                sim_ref = 0.35
                sim_src = 0.41
            rows.append(
                {
                    "step": step,
                    "arm": arm,
                    "train_job_id": jobs[arm],
                    "mode": mode,
                    "n": 20,
                    "keep": 19,
                    "fail": 0.05,
                    "cer": 0.04,
                    "sim_ref": sim_ref,
                    "sim_src": sim_src,
                    "margin": sim_ref - sim_src,
                    "ref_bound_count": 12,
                    "ref_bound": 0.6,
                    "ref_content_f1": 0.05,
                    "text_en_src_quick_n": 12 if mode == "text" else "",
                    "text_en_src_quick_fail": 1 / 12 if mode == "text" else "",
                    "text_en_src_scope": "proxy" if mode == "text" else "",
                    "run_id": (
                        f"ver2_9_5_final_{arm}_step-{step}_{mode}_"
                        "quick20_d2d3_seed1234"
                    ),
                    "output_dir": str(Path("/tmp") / f"{arm}-{step}-{mode}"),
                }
            )
    return rows


def write_completion(
    project: Path,
    step: int,
    *,
    backend: str,
    location: str,
    negative_arm: str | None = None,
) -> Path:
    record = (
        project
        / f"trainset/{location}/ver23_batch44_quick20_step{step}_20260713"
    )
    record.mkdir(parents=True)
    rows = metrics_rows(step, negative_arm=negative_arm)
    (record / "metrics.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    with (record / "metrics.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    write(record / "metrics.md", "# fixture\n")
    eval_root = project / "testset/outputs/ver23_batch44_quick20_20260713"
    jobs = {"r3": R3_JOB, "r5": R5_JOB}
    if backend == "qz":
        runner = write(record / "004110_submit_batch44_v1_quick20_qz.frozen.sh")
        ledger = record / "submitted_jobs.tsv"
        fields = (
            "job_name",
            "job_id",
            "step",
            "compute_group",
            "spec",
            "record_root",
            "eval_root",
            "code_root",
        )
        row = {
            "job_name": f"ver23_batch44_quick20_step{step}_20260713",
            "job_id": f"job-00000000-0000-0000-0000-{step:012d}",
            "step": str(step),
            "compute_group": COMPUTE,
            "spec": SPEC,
            "record_root": str(record.resolve()),
            "eval_root": str(eval_root.resolve()),
            "code_root": CODE_ROOT,
        }
        with ledger.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerow(row)
        completion_payload: dict[str, object] = {
            "schema": "moss_codecvc.batch44_v1_quick20_completion.v1",
            "status": "complete",
            "step": step,
            "record_root": str(record.resolve()),
            "eval_root": str(eval_root.resolve()),
            "training_jobs": jobs,
            "resource_contract": {
                "compute_group": "MTTS-3-2-0715",
                "compute_group_id": COMPUTE,
                "spec": SPEC,
                "instances": 1,
                "gpus": 8,
                "gpu_type": "NVIDIA_H200_SXM_141G",
            },
            "evaluation_job": {
                "job_name": row["job_name"],
                "job_id": row["job_id"],
                "submission_ledger": artifact(ledger),
            },
            "frozen_runner": artifact(runner),
            "metrics": {
                name: artifact(record / f"metrics.{name}")
                for name in ("json", "tsv", "md")
            },
        }
        marker_schema = "moss_codecvc.batch44_v1_quick20_complete_marker.v1"
    else:
        runtime = write(record / "LOCAL_RUNTIME.json", '{"backend":"local"}\n')
        runner = write(record / "004117_run_batch44_v1_quick20_local.frozen.sh")
        common = write(record / "004110_batch44_quick20_common.frozen.sh")
        helper = write(record / "batch44_quick20_local_completion.frozen.py")
        fixed_paths = {
            "no_text20": write(
                project
                / "testset/validation/ver2_8_timbre_quick20_seedtts_no_text_20260704.jsonl"
            ),
            "text_source": write(
                project / "testset/validation/seedtts_vc_ver2_3_validation.jsonl"
            ),
            "text20": write(
                record / "ver23_batch44_text_quick20_8cell_20260713.jsonl"
            ),
        }
        checkpoint_rows: dict[str, object] = {}
        run_rows: list[dict[str, object]] = []
        identity = (
            project
            / "trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713"
        )
        pair_ledger = write(identity / "submitted_pair.tsv", "fixture\n")
        train_args: dict[str, Path] = {}
        for arm, repeat in (("r3", 3), ("r5", 5)):
            checkpoint = (
                project
                / f"outputs/lora_runs/ver2_9_5_final_{arm}_v1_30k/step-{step}"
            )
            files = {
                name: write(checkpoint / name)
                for name in (
                    "adapter_model.safetensors",
                    "adapter_config.json",
                    "README.md",
                    "timbre_memory_adapter.pt",
                    "timbre_memory_config.json",
                )
            }
            checkpoint_rows[arm] = {
                "path": str(checkpoint.resolve()),
                "step": step,
                "training_job_id": jobs[arm],
                "files": {name: artifact(path) for name, path in files.items()},
            }
            no_text = project / "trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/no_text.train.jsonl"
            text = project / "trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/text.train.jsonl"
            args_payload = {
                "OUT_DIR": str(checkpoint.parent),
                "TRAIN_JSONL_SPEC": f"{no_text}::repeat=1,{text}::repeat={repeat}",
                "TEXT_REPEAT": str(repeat),
                "MAX_TRAIN_STEPS": "30000",
                "SAVE_STEPS": "2000",
                "EVAL_STEPS": "2000",
                "LEARNING_RATE": "1e-5",
                "LR_SCHEDULER_TYPE": "constant_with_warmup",
                "WARMUP_RATIO": "0.03",
            }
            train_args[arm] = write(
                identity / arm / "train_args_dry_run_core.json",
                json.dumps(args_payload) + "\n",
            )
            for mode in ("no_text", "text"):
                run_id = (
                    f"ver2_9_5_final_{arm}_step-{step}_{mode}_"
                    "quick20_d2d3_seed1234"
                )
                output_dir = eval_root / run_id
                output_artifacts = {
                    "summary": write(output_dir / f"{run_id}.summary.json"),
                    "asr": write(output_dir / f"{run_id}.asr_eval.jsonl"),
                    "speaker": write(output_dir / f"{run_id}.speaker_sim.csv"),
                    "ref_content": write(
                        output_dir / f"{run_id}.ref_content_similarity_summary.json"
                    ),
                }
                run_rows.append(
                    {
                        "arm": arm,
                        "mode": mode,
                        "run_id": run_id,
                        "training_job_id": jobs[arm],
                        "checkpoint": str(checkpoint.resolve()),
                        "output_dir": str(output_dir.resolve()),
                        "artifacts": {
                            name: artifact(path)
                            for name, path in output_artifacts.items()
                        },
                    }
                )
        gpu_rows = [
            {
                "index": index,
                "uuid": f"GPU-00000000-0000-0000-0000-{index:012d}",
                "name": "NVIDIA GeForce RTX 4090",
                "memory_total_mib": 49140,
            }
            for index in (0, 1)
        ]
        scheduling = "four lanes sequential; each lane uses GPUs 0,1 with two shards"
        runtime_payload = {
            "schema": "moss_codecvc.batch44_v1_quick20_local_runtime.v1",
            "backend": "local",
            "status": "started",
            "hostname": "xyzhang-dev--pytest",
            "gpu_count": 2,
            "gpu_indices": [0, 1],
            "gpu_model": "NVIDIA GeForce RTX 4090",
            "gpus": gpu_rows,
            "scheduling": scheduling,
            "runner": artifact(runner),
            "common_library": artifact(common),
            "completion_helper": artifact(helper),
        }
        runtime.write_text(json.dumps(runtime_payload) + "\n", encoding="utf-8")
        completion_payload = {
            "schema": "moss_codecvc.batch44_v1_quick20_completion.v2",
            "status": "complete",
            "backend": "local",
            "step": step,
            "record_root": str(record.resolve()),
            "eval_root": str(eval_root.resolve()),
            "code_root": CODE_ROOT,
            "training_jobs": jobs,
            "execution": {
                "hostname": "xyzhang-dev--pytest",
                "gpu_count": 2,
                "gpu_indices": [0, 1],
                "gpu_model": "NVIDIA GeForce RTX 4090",
                "gpus": gpu_rows,
                "scheduling": scheduling,
                "runtime_manifest": artifact(runtime),
            },
            "runner": artifact(runner),
            "common_library": artifact(common),
            "completion_helper": artifact(helper),
            "fixed_inputs": {
                name: artifact(path) for name, path in fixed_paths.items()
            },
            "checkpoints": checkpoint_rows,
            "training_provenance": {
                "pair_ledger": artifact(pair_ledger),
                "train_args": {
                    arm: artifact(path) for arm, path in train_args.items()
                },
            },
            "runs": run_rows,
            "metrics": {
                name: artifact(record / f"metrics.{name}")
                for name in ("json", "tsv", "md")
            },
        }
        marker_schema = "moss_codecvc.batch44_v1_quick20_complete_marker.v2"
    completion = record / "COMPLETED.json"
    completion.write_text(
        json.dumps(completion_payload, indent=2) + "\n", encoding="utf-8"
    )
    marker = {
        "schema": marker_schema,
        "status": "complete",
        "step": step,
        "completed_json_sha256": hashlib.sha256(completion.read_bytes()).hexdigest(),
    }
    if backend == "local":
        marker["backend"] = "local"
    (record / "complete.marker").write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8"
    )
    return record


def write_history(project: Path, *, negative_arm: str | None = None) -> None:
    for step in (2000, 4000, 6000):
        write_completion(project, step, backend="qz", location="qz_jobs")
    write_completion(
        project,
        8000,
        backend="local",
        location="qz_jobs",
        negative_arm=negative_arm,
    )


def write_ready_checkpoint(project: Path, step: int) -> tuple[Path, Path]:
    run_dirs = {
        "r3": project / "outputs/lora_runs/ver2_9_5_final_r3_v1_30k",
        "r5": project / "outputs/lora_runs/ver2_9_5_final_r5_v1_30k",
    }
    cfg = {
        "content_cross_attn_enabled": True,
        "content_cross_attn_layers": "all",
        "content_cross_attn_feature_dim": 768,
        "content_cross_attn_gate_init": -0.5,
        "content_cross_attn_output_scale": 0.3,
        "content_encoder_layers": 2,
        "guided_attn_loss_weight": 0.05,
        "phoneme_classifier_loss_weight": 0.02,
        "content_ctc_weight": 0.0,
        "progress_loss_weight": 0.1,
        "stop_loss_weight": 0.2,
        "target_front_ce_weight": 4.0,
        "target_front_ce_seconds": 0.75,
        "use_role_routing": True,
        "num_memory_tokens": 0,
        "timbre_side_only": False,
        "source_semantic_memory_enabled": False,
        "speaker_side_pathway_enabled": False,
        "speaker_cross_attn_enabled": False,
    }
    identity = (
        project
        / "trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713"
    )
    no_text = project / "trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/no_text.train.jsonl"
    text = project / "trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/text.train.jsonl"
    for arm, repeat in (("r3", 3), ("r5", 5)):
        checkpoint = run_dirs[arm] / f"step-{step}"
        write(checkpoint / "adapter_model.safetensors")
        write(checkpoint / "adapter_config.json", "{}\n")
        write(checkpoint / "README.md")
        write(checkpoint / "timbre_memory_adapter.pt")
        write(checkpoint / "timbre_memory_config.json", json.dumps(cfg) + "\n")
        args = {
            "OUT_DIR": str(run_dirs[arm]),
            "TRAIN_JSONL_SPEC": f"{no_text}::repeat=1,{text}::repeat={repeat}",
            "TEXT_REPEAT": str(repeat),
            "MAX_TRAIN_STEPS": "30000",
            "SAVE_STEPS": "2000",
            "EVAL_STEPS": "2000",
            "LEARNING_RATE": "1e-5",
            "LR_SCHEDULER_TYPE": "constant_with_warmup",
            "WARMUP_RATIO": "0.03",
        }
        write(identity / arm / "train_args_dry_run_core.json", json.dumps(args) + "\n")
    return run_dirs["r3"], run_dirs["r5"]


def make_fake_runner(tmp_path: Path) -> tuple[Path, Path]:
    log = tmp_path / "runner_calls.tsv"
    runner = tmp_path / "fake_local_runner.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"printf '%s\\t%s\\t%s\\t%s\\t%s\\n' \"$STEP\" \"$RECORD_ROOT\" \"$DRY_RUN\" \"$CONFIRM_LOCAL_QUICK20\" \"$EVAL_ROOT\" >> {log}\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    return runner, log


def run_watcher(
    project: Path,
    runner: Path,
    *,
    action: str = "plan",
    confirm: str = "0",
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "PROJECT_ROOT": str(project),
            "BATCH44_LOCAL_QUICK20_WATCHER_TEST_MODE": "1",
            "ACTION": action,
            "MODE": "once",
            "CONFIRM_LOCAL_QUICK20_WATCHER": confirm,
            "MIN_CHECKPOINT_AGE_SEC": "0",
            "LOCAL_RUNNER": str(runner),
            "PROVENANCE_VALIDATOR": str(VALIDATOR),
            "PYTHON": PYTHON,
        }
    )
    return subprocess.run(
        ["bash", str(WATCHER)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_default_plan_accepts_qz_history_local_8k_and_does_not_run(tmp_path: Path) -> None:
    project = tmp_path / "project"
    write_history(project)
    write_ready_checkpoint(project, 10000)
    runner, log = make_fake_runner(tmp_path)
    result = run_watcher(project, runner)
    assert result.returncode == 0, result.stdout
    assert "complete=4/15" in result.stdout
    assert "next_ready_step=10000 backend=local" in result.stdout
    assert not log.exists()
    state = project / "trainset/local_jobs/ver23_batch44_quick20_scheduler_20260713"
    statuses = json.loads((state / "scan_latest.json").read_text(encoding="utf-8"))
    assert [row["backend"] for row in statuses[:4]] == ["qz", "qz", "qz", "local"]


def test_live_gate_and_strictly_serial_first_local_step(tmp_path: Path) -> None:
    project = tmp_path / "project"
    write_history(project)
    r3, r5 = write_ready_checkpoint(project, 10000)
    write_ready_checkpoint(project, 12000)
    runner, log = make_fake_runner(tmp_path)
    denied = run_watcher(project, runner, action="run", confirm="0")
    assert denied.returncode != 0
    assert "requires CONFIRM_LOCAL_QUICK20_WATCHER=1" in denied.stdout
    result = run_watcher(project, runner, action="run", confirm="1")
    assert result.returncode == 0, result.stdout
    calls = log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 1
    fields = calls[0].split("\t")
    assert fields[0] == "10000"
    assert fields[1] == str(
        project / "trainset/local_jobs/ver23_batch44_quick20_step10000_20260713"
    )
    assert fields[2:4] == ["0", "1"]
    assert str(r3) not in fields[1] and str(r5) not in fields[1]
    assert not (
        project / "trainset/qz_jobs/ver23_batch44_quick20_step10000_20260713"
    ).exists()


def test_negative_margin_stops_scheduling_without_training_mutation(tmp_path: Path) -> None:
    project = tmp_path / "project"
    write_history(project, negative_arm="r3")
    write_ready_checkpoint(project, 10000)
    runner, log = make_fake_runner(tmp_path)
    result = run_watcher(project, runner, action="run", confirm="1")
    assert result.returncode == 0, result.stdout
    assert "margin<0" in result.stdout
    assert not log.exists()
    alert = (
        project
        / "trainset/local_jobs/ver23_batch44_quick20_scheduler_20260713/"
        "ALERT_NEGATIVE_NO_TEXT_MARGIN.json"
    )
    payload = json.loads(alert.read_text(encoding="utf-8"))
    assert payload["training_action"].startswith("report only")
    assert payload["alerts"][0]["arm"] == "r3"


def test_dual_record_roots_fail_closed(tmp_path: Path) -> None:
    project = tmp_path / "project"
    write_history(project)
    (project / "trainset/qz_jobs/ver23_batch44_quick20_step10000_20260713").mkdir(
        parents=True
    )
    (project / "trainset/local_jobs/ver23_batch44_quick20_step10000_20260713").mkdir(
        parents=True
    )
    runner, _ = make_fake_runner(tmp_path)
    result = run_watcher(project, runner)
    assert result.returncode != 0
    assert "conflicting QZ/local record roots" in result.stdout


def test_local_completion_with_qz_ledger_is_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    write_history(project)
    record = project / "trainset/qz_jobs/ver23_batch44_quick20_step8000_20260713"
    write(record / "submitted_jobs.tsv", "forged\n")
    runner, _ = make_fake_runner(tmp_path)
    result = run_watcher(project, runner)
    assert result.returncode != 0
    assert "must not contain a QZ submission ledger" in result.stdout


def test_source_has_no_remote_submit_path() -> None:
    source = WATCHER.read_text(encoding="utf-8")
    assert "004117_run_batch44_v1_quick20_local.sh" in source
    assert 'CONFIRM_LOCAL_QUICK20_WATCHER="${CONFIRM_LOCAL_QUICK20_WATCHER:-0}"' in source
    assert 'RECORD_ROOT="$record_root"' in source
    assert "004110_submit_batch44_v1_quick20_qz.sh" not in source
    assert "CONFIRM_BATCH44_QUICK20" not in source
    assert "ALLOW_LIVE_SUBMIT" not in source
    assert 'bash "$LOCAL_RUNNER"' in source

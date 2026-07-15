from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WATCHER = ROOT / "scripts/004124_watch_batch44_r3_warmstart_quick20_local.sh"
STRICT_VALIDATOR = ROOT / "scripts/batch44_r3_warmstart_quick20_validator.py"
JOB_ID = "job-165f3b1d-8c7d-47d8-8882-bdb86ef642ab"
COMPUTE_GROUP = "lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, text: str = "fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_contract(project: Path) -> tuple[Path, Path, Path]:
    run_dir = project / "outputs/lora_runs/ver2_9_5_final_r3_v1_warmstart10k_to30k"
    record = project / "trainset/qz_jobs/ver23_batch44_r3_v1_warmstart10k_to30k_20260713"
    contract = record / "warm_start_contract.json"
    payload = {
        "schema": "batch44_r3_weights_only_warm_start_v1",
        "status": "submitted",
        "job_id": JOB_ID,
        "job_name": "ver2_9_5_final_r3_v1_warmstart10k_to30k",
        "output_dir": str(run_dir.resolve()),
        "source_effective_step": 10000,
        "effective_step_offset": 10000,
        "continuation_local_target_step": 20000,
        "effective_target_step": 30000,
        "resume_semantics": "weights_only_warm_start_not_exact_resume",
        "step_mapping": "effective_step = 10000 + continuation_local_step",
        "mechanical_recovery_overrides": {
            "warmup_ratio": 0,
            "guided_attn_warmup_steps": 0,
        },
        "data": {"no_text": {"repeat": 1}, "text": {"repeat": 3}},
        "full_data_sha256_verified": True,
        "state_resets": ["optimizer", "scheduler", "rng", "global_step", "data_iterator"],
    }
    write(contract, json.dumps(payload, indent=2) + "\n")
    ledger = record / "submitted_jobs.tsv"
    with ledger.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("job_name", "job_id", "compute_group", "runner", "out_dir"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "job_name": payload["job_name"],
                "job_id": JOB_ID,
                "compute_group": COMPUTE_GROUP,
                "runner": str(record / "run_train_entrypoint.sh"),
                "out_dir": str(run_dir.resolve()),
            }
        )
    return contract, ledger, run_dir


def checkpoint_config() -> dict[str, object]:
    return {
        "content_cross_attn_enabled": True,
        "content_cross_attn_layers": "all",
        "content_cross_attn_feature_dim": 768,
        "content_cross_attn_gate_init": -0.5,
        "content_cross_attn_output_scale": 0.3,
        "content_encoder_layers": 2,
        "guided_attn_loss_weight": 0.05,
        "guided_attn_warmup_steps": 0,
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


def write_checkpoint(run_dir: Path, local_step: int) -> Path:
    checkpoint = run_dir / f"step-{local_step}"
    write(checkpoint / "adapter_model.safetensors")
    write(checkpoint / "adapter_config.json", "{}\n")
    write(checkpoint / "README.md")
    write(checkpoint / "timbre_memory_adapter.pt")
    write(checkpoint / "timbre_memory_config.json", json.dumps(checkpoint_config()) + "\n")
    return checkpoint


def make_runner(tmp_path: Path) -> tuple[Path, Path]:
    calls = tmp_path / "runner_calls.tsv"
    runner = tmp_path / "004123_fake_runner.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"printf '%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' "
        '"$EFFECTIVE_STEP" "$CONTINUATION_LOCAL_STEP" "$CHECKPOINT" '
        '"$TRAIN_JOB_ID" "$RECORD_ROOT" "$DRY_RUN" "$CONFIRM_RUN" '
        f">> {calls}\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    return runner, calls


def make_nvidia_smi(tmp_path: Path, *, used: int = 100, gpu_count: int = 2) -> Path:
    tool = tmp_path / f"nvidia-smi-{used}-{gpu_count}"
    rows = [f"{index}, NVIDIA GeForce RTX 4090, {used}, 49140" for index in range(gpu_count)]
    tool.write_text("#!/usr/bin/env bash\nprintf '%s\\n' " + " ".join(repr(row) for row in rows) + "\n", encoding="utf-8")
    tool.chmod(0o755)
    return tool


def metrics_rows(
    effective: int,
    checkpoint: Path,
    contract_sha: str,
    *,
    negative: bool = False,
) -> list[dict[str, object]]:
    local_step = effective - 10000
    rows: list[dict[str, object]] = []
    for mode in ("no_text", "text"):
        sim_ref = 0.35 if negative and mode == "no_text" else 0.45
        sim_src = 0.41 if mode == "no_text" else 0.28
        rows.append(
            {
                "step": effective,
                "base_effective_step": 10000,
                "continuation_local_step": local_step,
                "effective_step": effective,
                "checkpoint": str(checkpoint.resolve()),
                "warm_start_contract_sha256": contract_sha,
                "train_job_id": JOB_ID,
                "arm": "r3",
                "mode": mode,
                "n": 20,
                "keep": 19,
                "fail": 0.05,
                "cer": 0.04,
                "sim_ref": sim_ref,
                "sim_src": sim_src,
                "margin": sim_ref - sim_src,
                "ref_bound": 0.6,
                "ref_content_f1": 0.05,
            }
        )
    return rows


def write_completion(
    project: Path,
    run_dir: Path,
    contract: Path,
    effective: int,
    *,
    negative: bool = False,
) -> Path:
    local_step = effective - 10000
    checkpoint = write_checkpoint(run_dir, local_step)
    record = project / (
        "trainset/local_jobs/"
        f"ver23_batch44_r3_warmstart_quick20_step{effective}_20260713"
    )
    record.mkdir(parents=True)
    rows = metrics_rows(effective, checkpoint, sha256(contract), negative=negative)
    metrics_json = write(record / "metrics.json", json.dumps(rows, indent=2) + "\n")
    with (record / "metrics.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    write(record / "metrics.md", "# fixture\n")
    completion = {
        "schema": "moss_codecvc.batch44_r3_warmstart_quick20_completion.v1",
        "status": "complete",
        "backend": "local",
        "step": effective,
        "effective_step": effective,
        "base_effective_step": 10000,
        "continuation_local_step": local_step,
        "checkpoint": str(checkpoint.resolve()),
        "train_job_id": JOB_ID,
        "warm_start_contract_sha256": sha256(contract),
        "runs": [{"arm": "r3", "mode": mode} for mode in ("no_text", "text")],
        "metrics": {
            "json": {
                "path": str(metrics_json.resolve()),
                "size": metrics_json.stat().st_size,
                "sha256": sha256(metrics_json),
            }
        },
    }
    completed = write(record / "COMPLETED.json", json.dumps(completion, indent=2) + "\n")
    marker = {
        "schema": "moss_codecvc.batch44_r3_warmstart_quick20_complete_marker.v1",
        "status": "complete",
        "step": effective,
        "effective_step": effective,
        "base_effective_step": 10000,
        "continuation_local_step": local_step,
        "completed_json_sha256": sha256(completed),
    }
    write(record / "complete.marker", json.dumps(marker, indent=2) + "\n")
    return record


def run_watcher(
    project: Path,
    contract: Path,
    ledger: Path,
    run_dir: Path,
    runner: Path,
    nvidia_smi: Path,
    *,
    action: str = "plan",
    confirm_watcher: str = "0",
    confirm_run: str = "0",
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "PROJECT_ROOT": str(project),
            "BATCH44_R3_WARMSTART_QUICK20_WATCHER_TEST_MODE": "1",
            "MODE": "once",
            "ACTION": action,
            "CONFIRM_R3_WARMSTART_QUICK20_WATCHER": confirm_watcher,
            "CONFIRM_R3_WARMSTART_QUICK20_RUN": confirm_run,
            "MIN_CHECKPOINT_AGE_SEC": "0",
            "CONTINUATION_RUN_DIR": str(run_dir),
            "TRAIN_JOB_ID": JOB_ID,
            "WARM_START_RECORD_ROOT": str(contract.parent),
            "WARM_START_CONTRACT": str(contract),
            "TRAIN_LEDGER": str(ledger),
            "EXPECTED_CONTRACT_SHA256": sha256(contract),
            "EXPECTED_LEDGER_SHA256": sha256(ledger),
            "LOCAL_RUNNER": str(runner),
            "STRICT_VALIDATOR": str(STRICT_VALIDATOR),
            "PYTHON": sys.executable,
            "NVIDIA_SMI": str(nvidia_smi),
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


def fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    project = tmp_path / "project"
    contract, ledger, run_dir = write_contract(project)
    runner, calls = make_runner(tmp_path)
    nvidia_smi = make_nvidia_smi(tmp_path)
    return project, contract, ledger, run_dir, runner, calls, nvidia_smi


def test_default_plan_maps_effective_12k_to_local_2k_without_running(tmp_path: Path) -> None:
    project, contract, ledger, run_dir, runner, calls, nvidia_smi = fixture(tmp_path)
    write_checkpoint(run_dir, 2000)
    result = run_watcher(project, contract, ledger, run_dir, runner, nvidia_smi)
    assert result.returncode == 0, result.stdout
    assert "complete=0/10" in result.stdout
    assert "next_ready effective=12000 local=2000" in result.stdout
    assert not calls.exists()
    state = project / "trainset/local_jobs/ver23_batch44_r3_warmstart_quick20_scheduler_20260713"
    summary = json.loads((state / "scan_summary.json").read_text(encoding="utf-8"))
    assert summary["total"] == 10
    assert summary["first_incomplete"]["continuation_local_step"] == 2000


def test_run_requires_both_confirmations_and_schedules_only_first_ready_step(tmp_path: Path) -> None:
    project, contract, ledger, run_dir, runner, calls, nvidia_smi = fixture(tmp_path)
    write_checkpoint(run_dir, 2000)
    write_checkpoint(run_dir, 4000)
    denied_first = run_watcher(
        project, contract, ledger, run_dir, runner, nvidia_smi,
        action="run", confirm_watcher="0", confirm_run="1",
    )
    assert denied_first.returncode != 0
    assert "CONFIRM_R3_WARMSTART_QUICK20_WATCHER=1" in denied_first.stdout
    denied_second = run_watcher(
        project, contract, ledger, run_dir, runner, nvidia_smi,
        action="run", confirm_watcher="1", confirm_run="0",
    )
    assert denied_second.returncode != 0
    assert "CONFIRM_R3_WARMSTART_QUICK20_RUN=1" in denied_second.stdout
    result = run_watcher(
        project, contract, ledger, run_dir, runner, nvidia_smi,
        action="run", confirm_watcher="1", confirm_run="1",
    )
    assert result.returncode == 0, result.stdout
    rows = calls.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    fields = rows[0].split("\t")
    assert fields[0:2] == ["12000", "2000"]
    assert fields[2] == str((run_dir / "step-2000"))
    assert fields[3] == JOB_ID
    assert fields[5:7] == ["0", "1"]


def test_gpu_preflight_failure_blocks_runner(tmp_path: Path) -> None:
    project, contract, ledger, run_dir, runner, calls, _ = fixture(tmp_path)
    write_checkpoint(run_dir, 2000)
    busy = make_nvidia_smi(tmp_path, used=4096)
    result = run_watcher(
        project, contract, ledger, run_dir, runner, busy,
        action="run", confirm_watcher="1", confirm_run="1",
    )
    assert result.returncode != 0
    assert "GPU memory preflight exceeds" in result.stdout
    assert not calls.exists()


def test_negative_no_text_margin_alerts_without_invoking_runner(tmp_path: Path) -> None:
    project, contract, ledger, run_dir, runner, calls, nvidia_smi = fixture(tmp_path)
    write_completion(project, run_dir, contract, 12000, negative=True)
    write_checkpoint(run_dir, 4000)
    result = run_watcher(
        project, contract, ledger, run_dir, runner, nvidia_smi,
        action="run", confirm_watcher="1", confirm_run="1",
    )
    assert result.returncode == 0, result.stdout
    assert "training untouched" in result.stdout
    assert not calls.exists()
    alert_path = (
        project
        / "trainset/local_jobs/ver23_batch44_r3_warmstart_quick20_scheduler_20260713"
        / "ALERT_NEGATIVE_NO_TEXT_MARGIN.json"
    )
    alert = json.loads(alert_path.read_text(encoding="utf-8"))
    assert alert["training_action"].startswith("alert/report only")
    assert alert["train_job_id"] == JOB_ID
    assert alert["alerts"][0]["effective_step"] == 12000


def test_partial_completion_fails_closed(tmp_path: Path) -> None:
    project, contract, ledger, run_dir, runner, _, nvidia_smi = fixture(tmp_path)
    record = (
        project
        / "trainset/local_jobs/ver23_batch44_r3_warmstart_quick20_step12000_20260713"
    )
    write(record / "metrics.json", "[]\n")
    result = run_watcher(project, contract, ledger, run_dir, runner, nvidia_smi)
    assert result.returncode != 0
    assert "partial completion evidence" in result.stdout


def test_out_of_order_active_record_fails_serialization(tmp_path: Path) -> None:
    project, contract, ledger, run_dir, runner, _, nvidia_smi = fixture(tmp_path)
    write_checkpoint(run_dir, 2000)
    later = (
        project
        / "trainset/local_jobs/ver23_batch44_r3_warmstart_quick20_step14000_20260713"
    )
    (later / ".local_quick20.lock").mkdir(parents=True)
    write(later / "LOCAL_RUNTIME.json", "{}\n")
    write(later / "004123_run_batch44_r3_warmstart_quick20_local.frozen.sh")
    result = run_watcher(project, contract, ledger, run_dir, runner, nvidia_smi)
    assert result.returncode != 0
    assert "out-of-order quick20 state violates strict serialization" in result.stdout


def test_source_is_local_only_and_defaults_to_90_second_settle() -> None:
    source = WATCHER.read_text(encoding="utf-8")
    assert 'MIN_CHECKPOINT_AGE_SEC="${MIN_CHECKPOINT_AGE_SEC:-90}"' in source
    assert "004123_run_batch44_r3_warmstart_quick20_local.sh" in source
    assert "EFFECTIVE_STEPS=\"12000 14000 16000 18000 20000 22000 24000 26000 28000 30000\"" in source
    assert "effective_step = 10000 + continuation_local_step" in source
    assert '"$QZCLI"' not in source
    assert "create-job --" not in source
    assert "kill " not in source.lower()
    assert "CONFIRM_R3_WARMSTART_QUICK20_WATCHER" in source
    assert "CONFIRM_R3_WARMSTART_QUICK20_RUN" in source

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts/004125_watch_batch44_r3_warmstart_training_health.py"


def load_module():
    spec = importlib.util.spec_from_file_location("batch44_r3_warmstart_health_004125", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


def iso(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_log(
    path: Path,
    *,
    latest_step: int = 100,
    target: int = 20_000,
    age_seconds: int = 10,
    extra: str = "",
) -> None:
    now = dt.datetime.now(dt.timezone.utc)
    steps = [max(0, latest_step - 40), max(0, latest_step - 20), latest_step]
    ages = [200, 105, age_seconds]
    rows = [
        f"{iso(now - dt.timedelta(seconds=age))} step={step}/{target} epoch=0 "
        "loss=5.9000 lr=1.00e-05 lora_grad_norm=0.1000"
        for age, step in zip(ages, steps)
    ]
    if extra:
        rows.append(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    timestamp = (now - dt.timedelta(seconds=age_seconds)).timestamp()
    os.utime(path, (timestamp, timestamp))


def static_qz_payload(runner: Path) -> dict:
    return {
        "name": MODULE.JOB_NAME,
        "workspace_id": MODULE.ALLOWED_WORKSPACE_ID,
        "project_id": MODULE.ALLOWED_PROJECT_ID,
        "logic_compute_group_id": MODULE.ALLOWED_COMPUTE_GROUP_ID,
        "command": f"sh {runner}",
        "framework_config": [
            {
                "instance_count": 1,
                "gpu_count": 8,
                "resource_spec_price": {
                    "quota_id": MODULE.ALLOWED_SPEC_ID,
                    "gpu_type": MODULE.ALLOWED_GPU_TYPE,
                },
            }
        ],
    }


def write_fake_qz(path: Path) -> None:
    path.write_text(
        f"""#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

call_path = Path(os.environ["FAKE_QZ_CALL_LOG"])
with call_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({{
        "argv": sys.argv[1:],
        "home": os.environ.get("HOME"),
        "proxies": {{key: os.environ.get(key) for key in (
            "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"
        ) if key in os.environ}},
    }}, sort_keys=True) + "\\n")

job = sys.argv[2]
if os.environ.get("FAKE_QZ_FAIL") == "auth":
    print("Cookie expired; please login", file=sys.stderr)
    raise SystemExit(7)
if os.environ.get("FAKE_QZ_FAIL") == "generic":
    print("temporary gateway error", file=sys.stderr)
    raise SystemExit(8)

record = Path(os.environ["RECORD_ROOT"])
payload = {{
    "job_id": job,
    "name": "{MODULE.JOB_NAME}",
    "status": os.environ.get("FAKE_QZ_STATUS", "job_running"),
    "workspace_id": "{MODULE.ALLOWED_WORKSPACE_ID}",
    "project_id": "{MODULE.ALLOWED_PROJECT_ID}",
    "logic_compute_group_id": "{MODULE.ALLOWED_COMPUTE_GROUP_ID}",
    "logic_compute_group_name": os.environ.get("FAKE_QZ_GROUP", "{MODULE.ALLOWED_COMPUTE_GROUP_NAME}"),
    "command": f"sh {{record / 'run_train_entrypoint.sh'}}",
    "framework_config": [{{
        "instance_count": 1,
        "gpu_count": int(os.environ.get("FAKE_QZ_GPUS", "8")),
        "instance_spec_price_info": {{
            "quota_id": "{MODULE.ALLOWED_SPEC_ID}",
            "gpu_info": {{"gpu_type": "{MODULE.ALLOWED_GPU_TYPE}"}},
        }},
    }}],
}}
print("qz detail panel")
print(json.dumps(payload, ensure_ascii=False))
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def make_checkpoint(run_dir: Path, local_step: int | str) -> Path:
    directory = run_dir / ("final" if local_step == "final" else f"step-{local_step}")
    directory.mkdir(parents=True, exist_ok=True)
    for name in MODULE.EXPECTED_MODEL_FILES:
        (directory / name).write_bytes(b"model")
    return directory


def setup_fixture(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    project = tmp_path / "MOSS-CodecVC"
    project.mkdir()
    record = project / MODULE.RECORD_RELATIVE
    state = project / MODULE.STATE_RELATIVE
    run_dir = project / MODULE.RUN_RELATIVE
    source_checkpoint = project / MODULE.SOURCE_CHECKPOINT_RELATIVE
    record.mkdir(parents=True)
    source_checkpoint.mkdir(parents=True)
    home = tmp_path / "qzhome"
    home.mkdir()
    fake_qz = tmp_path / "qzcli"
    write_fake_qz(fake_qz)

    runner = record / "run_train_entrypoint.sh"
    runner.write_text(
        "#!/bin/sh\n"
        f'OUT_DIR="{run_dir}"\n'
        f'RESUME_ADAPTER_PATH="{source_checkpoint}"\n'
        f'TRAIN_JSONL_SPEC="{project}/trainset/v1/no_text.train.jsonl::repeat=1,'
        f'{project}/trainset/v1/text.train.jsonl::repeat=3"\n'
        'echo "lr_scheduler_type=constant_with_warmup warmup_ratio=0.0"\n'
        'echo "guided_weight=0.05 guided_warmup=0"\n'
        'RESUME_ARGS="--resume-adapter-path $RESUME_ADAPTER_PATH"\n'
        'python train.py --max-train-steps "20000" --save-steps "2000" $RESUME_ARGS\n',
        encoding="utf-8",
    )
    runner.chmod(0o755)
    contract = {
        "schema": "batch44_r3_weights_only_warm_start_v1",
        "status": "submitted",
        "job_id": MODULE.JOB_ID,
        "job_name": MODULE.JOB_NAME,
        "output_dir": str(run_dir),
        "source_checkpoint": str(source_checkpoint),
        "source_effective_step": 10_000,
        "continuation_local_target_step": 20_000,
        "effective_step_offset": 10_000,
        "effective_target_step": 30_000,
        "resume_semantics": "weights_only_warm_start_not_exact_resume",
        "full_data_sha256_verified": True,
        "mechanical_recovery_overrides": {
            "warmup_ratio": 0.0,
            "guided_attn_warmup_steps": 0,
        },
    }
    contract_path = record / "warm_start_contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    (record / "submitted_jobs.tsv").write_text(
        "job_name\tjob_id\tcompute_group\trunner\tout_dir\n"
        f"{MODULE.JOB_NAME}\t{MODULE.JOB_ID}\t{MODULE.ALLOWED_COMPUTE_GROUP_ID}\t"
        f"{runner}\t{run_dir}\n",
        encoding="utf-8",
    )
    (record / "qz_payload.json").write_text(
        json.dumps(static_qz_payload(runner)), encoding="utf-8"
    )
    write_log(run_dir / "train.log")

    env = os.environ.copy()
    env.update(
        {
            "BATCH44_WARMSTART_HEALTH_TEST_MODE": "1",
            "PROJECT_ROOT": str(project),
            "RECORD_ROOT": str(record),
            "STATE_ROOT": str(state),
            "RUN_DIR": str(run_dir),
            "SOURCE_CHECKPOINT": str(source_checkpoint),
            "QZCLI": str(fake_qz),
            "QZCLI_HOME": str(home),
            "WARM_START_CONTRACT_SHA256": contract_sha,
            "FAKE_QZ_CALL_LOG": str(tmp_path / "qz_calls.jsonl"),
            "BATCH44_WARMSTART_HEALTH_QZ_TIMEOUT_SECONDS": "5",
        }
    )
    for key in (
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        env[key] = "http://must-not-reach-qzcli.invalid:9999"
    return project, env


def invoke(env: dict[str, str], *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--mode", "once", *extra],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def read_status(project: Path) -> dict:
    return json.loads(
        (project / MODULE.STATE_RELATIVE / "STATUS.json").read_text(encoding="utf-8")
    )


def alert_ids(status: dict) -> set[str]:
    return {item["id"] for item in status["alerts"]}


def test_static_contract_is_single_job_hard_bound_and_observation_only() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert MODULE.JOB_ID in source
    assert MODULE.WARM_START_CONTRACT_SHA256 in source
    assert "LOCAL_TARGET_STEP = 20_000" in source
    assert "EFFECTIVE_TARGET_STEP = 30_000" in source
    assert 'command = [str(config.qzcli), "status", JOB_ID, "--json"]' in source
    assert "create-job" not in source
    assert 'subprocess.run([str(config.qzcli), "stop"' not in source
    assert 'subprocess.run([str(config.qzcli), "restart"' not in source


def test_clean_scan_binds_contract_maps_steps_and_removes_proxies(tmp_path: Path) -> None:
    project, env = setup_fixture(tmp_path)
    result = invoke(env)
    assert result.returncode == 0, result.stderr + result.stdout
    status = read_status(project)
    assert status["status"] == "healthy"
    assert status["contract"]["warm_start_contract_sha256"] == env["WARM_START_CONTRACT_SHA256"]
    assert status["run"]["latest_local_step"] == 100
    assert status["run"]["latest_effective_step"] == 10_100
    assert status["step_mapping"]["effective_target_step"] == 30_000
    assert status["run"]["train_log"]["seconds_per_step"] == pytest.approx(4.75)
    calls = [json.loads(line) for line in Path(env["FAKE_QZ_CALL_LOG"]).read_text().splitlines()]
    assert len(calls) == 1
    assert calls[0]["argv"] == ["status", MODULE.JOB_ID, "--json"]
    assert calls[0]["home"] == env["QZCLI_HOME"]
    assert calls[0]["proxies"] == {}


def test_contract_sha_drift_fails_closed_before_qz_query(tmp_path: Path) -> None:
    project, env = setup_fixture(tmp_path)
    contract = Path(env["RECORD_ROOT"]) / "warm_start_contract.json"
    contract.write_text(contract.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    result = invoke(env)
    assert result.returncode == 2
    assert "warm_start_contract SHA drift" in result.stderr
    assert not Path(env["FAKE_QZ_CALL_LOG"]).exists()
    alert = json.loads(
        (project / MODULE.STATE_RELATIVE / "ALERT.json").read_text(encoding="utf-8")
    )
    assert alert["automatic_action_taken"] is False
    assert alert["alerts"][0]["id"] == "watcher:static_contract_error"


def test_terminal_before_local_20k_is_critical_without_remote_action(tmp_path: Path) -> None:
    project, env = setup_fixture(tmp_path)
    env["FAKE_QZ_STATUS"] = "job_stopped"
    result = invoke(env)
    assert result.returncode == 1
    status = read_status(project)
    assert "warmstart:terminal_before_local_20000" in alert_ids(status)
    alert = json.loads(
        (project / MODULE.STATE_RELATIVE / "ALERT.json").read_text(encoding="utf-8")
    )
    assert alert["automatic_action_taken"] is False
    recommendation = (
        project / MODULE.STATE_RELATIVE / "STOP_OR_RECOVERY_RECOMMENDATION.md"
    ).read_text(encoding="utf-8")
    assert "has **not** stopped, restarted, logged in to, or submitted" in recommendation


def test_live_resource_drift_is_rejected(tmp_path: Path) -> None:
    project, env = setup_fixture(tmp_path)
    env["FAKE_QZ_GPUS"] = "4"
    result = invoke(env)
    assert result.returncode == 1
    status = read_status(project)
    assert "warmstart:qz_resource_or_identity_drift" in alert_ids(status)


def test_complete_local_checkpoint_reports_effective_mapping(tmp_path: Path) -> None:
    project, env = setup_fixture(tmp_path)
    run_dir = Path(env["RUN_DIR"])
    write_log(run_dir / "train.log", latest_step=2300)
    make_checkpoint(run_dir, 2000)
    result = invoke(env)
    assert result.returncode == 0, result.stderr + result.stdout
    status = read_status(project)
    checkpoints = status["run"]["checkpoints"]
    assert checkpoints["complete_local_steps"] == [2000]
    assert checkpoints["complete_effective_steps"] == [12000]
    assert not checkpoints["overdue_local_steps"]


def test_succeeded_requires_all_local_checkpoints_and_final(tmp_path: Path) -> None:
    project, env = setup_fixture(tmp_path)
    run_dir = Path(env["RUN_DIR"])
    write_log(run_dir / "train.log", latest_step=20_000)
    for step in range(2000, 20_001, 2000):
        make_checkpoint(run_dir, step)
    make_checkpoint(run_dir, "final")
    env["FAKE_QZ_STATUS"] = "job_succeeded"
    result = invoke(env)
    assert result.returncode == 0, result.stderr + result.stdout
    status = read_status(project)
    assert status["status"] == "complete"
    assert status["run"]["latest_effective_step"] == 30_000
    assert status["run"]["checkpoints"]["final_complete"] is True


def test_wrong_train_log_target_is_critical(tmp_path: Path) -> None:
    project, env = setup_fixture(tmp_path)
    write_log(Path(env["RUN_DIR"]) / "train.log", target=30_000)
    result = invoke(env)
    assert result.returncode == 1
    assert "warmstart:local_target_drift" in alert_ids(read_status(project))

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts/004116_watch_batch44_training_health.py"


def load_module():
    spec = importlib.util.spec_from_file_location("batch44_health_004116", SCRIPT)
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
    age_seconds: int = 10,
    extra: str = "",
) -> None:
    now = dt.datetime.now(dt.timezone.utc)
    rows = []
    for offset, step in ((200, latest_step - 40), (105, latest_step - 20), (age_seconds, latest_step)):
        rows.append(
            f"{iso(now - dt.timedelta(seconds=offset))} step={step}/30000 epoch=0 "
            "loss=5.9000 lr=1.00e-05 lora_grad_norm=0.1000"
        )
    if extra:
        rows.append(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    timestamp = (now - dt.timedelta(seconds=age_seconds)).timestamp()
    os.utime(path, (timestamp, timestamp))


def qz_payload(root: Path, arm: str, job_name: str) -> dict:
    return {
        "name": job_name,
        "workspace_id": MODULE.ALLOWED_WORKSPACE_ID,
        "project_id": MODULE.ALLOWED_PROJECT_ID,
        "logic_compute_group_id": MODULE.ALLOWED_COMPUTE_GROUP_ID,
        "command": f"sh {root / arm / 'run_train_entrypoint.sh'}",
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
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

job = sys.argv[2]
call_path = Path(os.environ["FAKE_QZ_CALL_LOG"])
with call_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "argv": sys.argv[1:],
        "home": os.environ.get("HOME"),
        "proxies": {key: os.environ.get(key) for key in (
            "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"
        ) if key in os.environ},
    }, sort_keys=True) + "\\n")
if os.environ.get("FAKE_QZ_AUTH_JOB") == job:
    print("Cookie expired; please login", file=sys.stderr)
    raise SystemExit(7)
if os.environ.get("FAKE_QZ_FAIL_JOB") == job:
    print("temporary gateway error", file=sys.stderr)
    raise SystemExit(8)

arms = {
    "job-2b91d332-d500-4279-84f9-0a6a81a376aa": ("r3", "ver2_9_5_final_r3_v1_30k", os.environ.get("FAKE_QZ_STATUS_R3", "job_running")),
    "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c": ("r5", "ver2_9_5_final_r5_v1_30k", os.environ.get("FAKE_QZ_STATUS_R5", "job_running")),
}
arm, name, status = arms[job]
root = Path(os.environ["PAIR_RECORD_ROOT"])
payload = {
    "job_id": job,
    "name": name,
    "status": status,
    "workspace_id": "ws-8207e9e2-e733-4eec-a475-cfa1c36480ba",
    "project_id": "project-c67c548f-f02c-453b-ba5b-8745db6886e7",
    "logic_compute_group_id": "lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122",
    "logic_compute_group_name": os.environ.get("FAKE_QZ_GROUP_NAME", "MTTS-3-2-0715"),
    "command": f"sh {root / arm / 'run_train_entrypoint.sh'}",
    "framework_config": [{
        "instance_count": 1,
        "gpu_count": int(os.environ.get("FAKE_QZ_GPUS", "8")),
        "instance_spec_price_info": {
            "quota_id": "67b10bc6-78b0-41a3-aaf4-358eeeb99009",
            "gpu_info": {"gpu_type": "NVIDIA_H200_SXM_141G"},
        },
    }],
}
print("qz detail panel")
print(json.dumps(payload, ensure_ascii=False))
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def setup_fixture(
    tmp_path: Path, *, symlink_trainset: bool = False
) -> tuple[Path, dict[str, str]]:
    project = tmp_path / "MOSS-CodecVC"
    project.mkdir()
    if symlink_trainset:
        physical_trainset = tmp_path / "physical_trainset"
        physical_trainset.mkdir()
        (project / "trainset").symlink_to(physical_trainset, target_is_directory=True)
    pair = project / MODULE.PAIR_RECORD_RELATIVE
    state = project / MODULE.DEFAULT_STATE_RELATIVE
    home = tmp_path / "qzhome"
    home.mkdir(parents=True)
    fake_qz = tmp_path / "qzcli"
    write_fake_qz(fake_qz)

    ledger_lines = ["arm\tjob_name\tjob_id\tcompute_group\trunner\tout_dir"]
    for arm in MODULE.ARMS:
        record = pair / arm.arm
        record.mkdir(parents=True)
        runner = record / "run_train_entrypoint.sh"
        run_dir = project / "outputs/lora_runs" / arm.run_name
        runner.write_text(
            "#!/bin/sh\n"
            f'TRAIN_JSONL_SPEC="{project}/trainset/v1/no_text.train.jsonl::repeat=1,'
            f'{project}/trainset/v1/text.train.jsonl::repeat={arm.repeat}"\n'
            f'OUT_DIR="{run_dir}"\n'
            'EVAL_STEPS="2000"\n'
            'python train.py --max-train-steps "30000" --save-steps "2000"\n',
            encoding="utf-8",
        )
        runner.chmod(0o755)
        (record / "qz_payload.json").write_text(
            json.dumps(qz_payload(pair, arm.arm, arm.job_name)), encoding="utf-8"
        )
        write_log(run_dir / "train.log")
        ledger_lines.append(
            "\t".join(
                (
                    arm.arm,
                    arm.job_name,
                    arm.job_id,
                    MODULE.ALLOWED_COMPUTE_GROUP_ID,
                    str(runner),
                    str(run_dir),
                )
            )
        )
    (pair / "submitted_pair.tsv").write_text(
        "\n".join(ledger_lines) + "\n", encoding="utf-8"
    )
    env = os.environ.copy()
    env.update(
        {
            "BATCH44_HEALTH_TEST_MODE": "1",
            "PROJECT_ROOT": str(project),
            "PAIR_RECORD_ROOT": str(pair),
            "STATE_ROOT": str(state),
            "QZCLI": str(fake_qz),
            "QZCLI_HOME": str(home),
            "FAKE_QZ_CALL_LOG": str(tmp_path / "qz_calls.jsonl"),
            "BATCH44_HEALTH_QZ_TIMEOUT_SECONDS": "5",
            "BATCH44_HEALTH_SPACE_WARN_BYTES": "0",
            "BATCH44_HEALTH_SPACE_CRITICAL_BYTES": "0",
            "BATCH44_HEALTH_INODE_WARN_COUNT": "0",
            "BATCH44_HEALTH_INODE_CRITICAL_COUNT": "0",
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
        (project / MODULE.DEFAULT_STATE_RELATIVE / "STATUS.json").read_text(encoding="utf-8")
    )


def alert_ids(status: dict) -> set[str]:
    return {item["id"] for item in status["alerts"]}


def test_static_contract_is_hard_bound_and_observation_only() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "MTTS-3-2-0715" in source
    assert "job-2b91d332-d500-4279-84f9-0a6a81a376aa" in source
    assert "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c" in source
    assert "POLL_SECONDS = 300" in source
    assert 'command = [str(config.qzcli), "status", arm.job_id, "--json"]' in source
    assert "create-job" not in source
    assert "subprocess.run([str(config.qzcli), \"stop\"" not in source
    assert "subprocess.run([str(config.qzcli), \"restart\"" not in source


def test_clean_scan_queries_r3_then_r5_with_shared_home_and_no_proxy(tmp_path: Path) -> None:
    project, env = setup_fixture(tmp_path)
    result = invoke(env)
    assert result.returncode == 0, result.stderr + result.stdout
    status = read_status(project)
    assert status["status"] == "healthy"
    assert not status["alerts"]
    assert status["poll_seconds"] == 300
    assert status["resume_risk"]["strict_resume_available"] is False
    assert status["arms"]["r3"]["train_log"]["eta_utc"]
    assert status["arms"]["r5"]["train_log"]["seconds_per_step"] == pytest.approx(4.75)
    state = project / MODULE.DEFAULT_STATE_RELATIVE
    assert (state / "STATUS.md").is_file()
    assert not (state / "ALERT.json").exists()
    calls = [json.loads(line) for line in Path(env["FAKE_QZ_CALL_LOG"]).read_text().splitlines()]
    assert [item["argv"][1] for item in calls] == [
        MODULE.ARMS[0].job_id,
        MODULE.ARMS[1].job_id,
    ]
    assert all(item["argv"][0] == "status" and item["argv"][2] == "--json" for item in calls)
    assert all(item["home"] == env["QZCLI_HOME"] for item in calls)
    assert all(item["proxies"] == {} for item in calls)


def test_trainset_symlink_does_not_rewrite_ledger_or_remote_command_identity(
    tmp_path: Path,
) -> None:
    project, env = setup_fixture(tmp_path, symlink_trainset=True)
    result = invoke(env)
    assert result.returncode == 0, result.stderr + result.stdout
    status = read_status(project)
    assert status["status"] == "healthy"
    calls = [json.loads(line) for line in Path(env["FAKE_QZ_CALL_LOG"]).read_text().splitlines()]
    assert len(calls) == 2


def test_second_scan_records_step_growth_and_eta(tmp_path: Path) -> None:
    project, env = setup_fixture(tmp_path)
    assert invoke(env).returncode == 0
    for arm in MODULE.ARMS:
        write_log(project / "outputs/lora_runs" / arm.run_name / "train.log", latest_step=140)
    assert invoke(env).returncode == 0
    status = read_status(project)
    for arm in ("r3", "r5"):
        log = status["arms"][arm]["train_log"]
        assert log["previous_observed_step"] == 100
        assert log["step_delta_since_previous_scan"] == 40
        assert log["progressing_since_previous_scan"] is True
        assert log["eta_utc"]


def test_running_but_stale_log_writes_alert_and_manual_recommendation(tmp_path: Path) -> None:
    project, env = setup_fixture(tmp_path)
    env["BATCH44_HEALTH_LOG_STALL_SECONDS"] = "30"
    for arm in MODULE.ARMS:
        write_log(
            project / "outputs/lora_runs" / arm.run_name / "train.log",
            latest_step=100,
            age_seconds=120,
        )
    result = invoke(env)
    assert result.returncode == 1
    status = read_status(project)
    assert {"r3:training_stalled", "r5:training_stalled"} <= alert_ids(status)
    state = project / MODULE.DEFAULT_STATE_RELATIVE
    recommendation = (state / "STOP_OR_RECOVERY_RECOMMENDATION.md").read_text()
    assert "has **not** stopped, restarted, logged in to, or submitted" in recommendation


def test_nan_fatal_checkpoint_error_and_stale_tmp_are_detected(tmp_path: Path) -> None:
    project, env = setup_fixture(tmp_path)
    env["BATCH44_HEALTH_TMP_WARN_SECONDS"] = "1"
    env["BATCH44_HEALTH_TMP_CRITICAL_SECONDS"] = "2"
    run = project / "outputs/lora_runs" / MODULE.ARMS[0].run_name
    extra = (
        f"{iso(dt.datetime.now(dt.timezone.utc))} step=120/30000 epoch=0 "
        "loss=nan lr=1e-5\nTraceback (most recent call last): RuntimeError: CUDA out of memory"
    )
    write_log(run / "train.log", latest_step=100, extra=extra)
    error = run / "step-2000/checkpoint_save_error.txt"
    error.parent.mkdir(parents=True)
    error.write_text("OSError: no space left on device\n", encoding="utf-8")
    hidden = run / ".step-2000.tmp-123"
    hidden.mkdir()
    old = dt.datetime.now(dt.timezone.utc).timestamp() - 10
    os.utime(hidden, (old, old))
    result = invoke(env)
    assert result.returncode == 1
    ids = alert_ids(read_status(project))
    assert "r3:nonfinite_training_metric" in ids
    assert "r3:fatal_log_signature" in ids
    assert "r3:checkpoint_save_error" in ids
    assert "r3:checkpoint_incomplete" in ids
    assert "r3:stale_hidden_checkpoint_tmp" in ids


def test_checkpoint_missing_after_training_passes_save_boundary_is_critical(tmp_path: Path) -> None:
    project, env = setup_fixture(tmp_path)
    run = project / "outputs/lora_runs" / MODULE.ARMS[0].run_name
    write_log(run / "train.log", latest_step=2020)
    result = invoke(env)
    assert result.returncode == 1
    status = read_status(project)
    assert "r3:checkpoint_overdue" in alert_ids(status)
    alert = next(item for item in status["alerts"] if item["id"] == "r3:checkpoint_overdue")
    assert alert["evidence"]["overdue_steps"] == [2000]


def test_complete_inference_checkpoint_is_reported_but_not_strict_resume(tmp_path: Path) -> None:
    project, env = setup_fixture(tmp_path)
    run = project / "outputs/lora_runs" / MODULE.ARMS[0].run_name
    write_log(run / "train.log", latest_step=2020)
    checkpoint = run / "step-2000"
    checkpoint.mkdir()
    for name in MODULE.EXPECTED_MODEL_FILES:
        (checkpoint / name).write_bytes(b"model")
    result = invoke(env)
    assert result.returncode == 0, result.stderr + result.stdout
    status = read_status(project)
    inventory = status["arms"]["r3"]["checkpoints"]
    assert inventory["complete_steps"] == [2000]
    assert inventory["overdue_steps"] == []
    assert inventory["strict_resume_available"] is False
    assert "inference_only" in inventory["resume_contract"]


@pytest.mark.parametrize("terminal", ["job_failed", "job_stopped", "job_succeeded"])
def test_any_terminal_state_before_30k_is_a_critical_alert(
    tmp_path: Path, terminal: str
) -> None:
    project, env = setup_fixture(tmp_path)
    env["FAKE_QZ_STATUS_R3"] = terminal
    result = invoke(env)
    assert result.returncode == 1
    status = read_status(project)
    assert "r3:terminal_before_30000" in alert_ids(status)
    alert = next(item for item in status["alerts"] if item["id"] == "r3:terminal_before_30000")
    assert alert["severity"] == "critical"


def test_explicit_auth_error_is_conservative_and_does_not_block_second_query(tmp_path: Path) -> None:
    project, env = setup_fixture(tmp_path)
    env["FAKE_QZ_AUTH_JOB"] = MODULE.ARMS[0].job_id
    result = invoke(env)
    assert result.returncode == 1
    status = read_status(project)
    ids = alert_ids(status)
    assert "r3:qz_authentication_required" in ids
    assert "r3:qz_query_unavailable" not in ids
    assert status["arms"]["r3"]["qz"]["authentication_error"] is True
    assert status["arms"]["r5"]["qz"]["status"] == "job_running"
    calls = [json.loads(line) for line in Path(env["FAKE_QZ_CALL_LOG"]).read_text().splitlines()]
    assert len(calls) == 2
    assert all(item["argv"][0] == "status" for item in calls)


def test_generic_qz_failure_is_not_mislabeled_as_authentication(tmp_path: Path) -> None:
    project, env = setup_fixture(tmp_path)
    env["FAKE_QZ_FAIL_JOB"] = MODULE.ARMS[0].job_id
    assert invoke(env).returncode == 1
    status = read_status(project)
    assert "r3:qz_query_unavailable" in alert_ids(status)
    assert "r3:qz_authentication_required" not in alert_ids(status)


def test_live_resource_drift_is_rejected(tmp_path: Path) -> None:
    project, env = setup_fixture(tmp_path)
    env["FAKE_QZ_GPUS"] = "4"
    assert invoke(env).returncode == 1
    status = read_status(project)
    assert {
        "r3:qz_resource_or_identity_drift",
        "r5:qz_resource_or_identity_drift",
    } <= alert_ids(status)


def test_static_ledger_drift_writes_contract_alert_before_qz(tmp_path: Path) -> None:
    project, env = setup_fixture(tmp_path)
    ledger = Path(env["PAIR_RECORD_ROOT"]) / "submitted_pair.tsv"
    ledger.write_text(ledger.read_text().replace(MODULE.ARMS[0].job_id, MODULE.ARMS[1].job_id))
    result = invoke(env)
    assert result.returncode == 2
    state = project / MODULE.DEFAULT_STATE_RELATIVE
    alert = json.loads((state / "ALERT.json").read_text())
    assert alert["alerts"][0]["id"] == "watcher:static_contract_error"
    assert not Path(env["FAKE_QZ_CALL_LOG"]).exists()


def test_storage_and_inode_thresholds_write_alerts_without_deleting(tmp_path: Path) -> None:
    project, env = setup_fixture(tmp_path)
    huge = str(10**30)
    env.update(
        {
            "BATCH44_HEALTH_SPACE_WARN_BYTES": huge,
            "BATCH44_HEALTH_SPACE_CRITICAL_BYTES": huge,
            "BATCH44_HEALTH_INODE_WARN_COUNT": huge,
            "BATCH44_HEALTH_INODE_CRITICAL_COUNT": huge,
        }
    )
    assert invoke(env).returncode == 1
    ids = alert_ids(read_status(project))
    assert "storage:available_bytes_low" in ids
    assert "storage:available_inodes_low" in ids


def test_production_poll_override_is_rejected_before_any_qz_call(tmp_path: Path) -> None:
    # This exercises build_config directly so the canonical production paths
    # are not required to run a subprocess in the test.
    args = MODULE.parser().parse_args(["--mode", "monitor", "--poll-seconds", "60"])
    old = os.environ.pop("BATCH44_HEALTH_TEST_MODE", None)
    try:
        with pytest.raises(MODULE.ContractError, match="hard-locked to 300"):
            MODULE.build_config(args)
    finally:
        if old is not None:
            os.environ["BATCH44_HEALTH_TEST_MODE"] = old

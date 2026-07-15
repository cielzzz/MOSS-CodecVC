from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WATCHER = ROOT / "scripts/004122_watch_batch44_v1_paired_full320_local.sh"
RUNNER = ROOT / "scripts/004118_run_batch44_v1_paired_full320_local.sh"
QUICK_VALIDATOR = ROOT / "scripts/004103_select_batch43_best3.py"
FULL_VALIDATOR = ROOT / "scripts/004107_finalize_batch43_pathx_final.py"


def run_plan(project: Path, **extra: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PROJECT_ROOT": str(project),
        "BATCH44_LOCAL_FULL320_WATCHER_TEST_MODE": "1",
        "LOCAL_RUNNER": str(RUNNER),
        "QUICK20_VALIDATOR": str(QUICK_VALIDATOR),
        "FULL320_VALIDATOR": str(FULL_VALIDATOR),
        "MODE": "once",
        "ACTION": "plan",
        **extra,
    }
    return subprocess.run(
        ["bash", str(WATCHER)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_watcher_is_local_plan_only_by_default() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(WATCHER)], text=True, capture_output=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    source = WATCHER.read_text(encoding="utf-8")
    assert 'ACTION="${ACTION:-plan}"' in source
    assert 'STEPS="10000 20000 30000"' in source
    assert "CONFIRM_LOCAL_FULL320_WATCHER" in source
    assert 'STEP="$step" ACTION=run CONFIRM_LOCAL_FULL320=1' in source
    assert "004118_run_batch44_v1_paired_full320_local.sh" in source
    assert "qzcli" not in source.lower()
    assert "create-job" not in source
    assert "kill " not in source.lower()
    assert "submitted_jobs.tsv" in source  # fail-closed prohibition only


def test_empty_fixture_waits_for_local_quick20(tmp_path: Path) -> None:
    project = tmp_path / "MOSS-CodecVC"
    result = run_plan(project)
    assert result.returncode == 0, result.stdout
    assert "waiting for strict local quick20 completion at step-10000" in result.stdout
    assert "complete=0/3" in result.stdout
    assert not (project / "trainset/qz_jobs").exists()


def test_live_action_requires_second_confirmation(tmp_path: Path) -> None:
    project = tmp_path / "MOSS-CodecVC"
    result = run_plan(project, ACTION="run", CONFIRM_LOCAL_FULL320_WATCHER="0")
    assert result.returncode != 0
    assert "requires CONFIRM_LOCAL_FULL320_WATCHER=1" in result.stdout


def test_remote_record_conflict_fails_closed(tmp_path: Path) -> None:
    project = tmp_path / "MOSS-CodecVC"
    remote = (
        project
        / "trainset/qz_jobs/ver23_batch44_quick20_step10000_20260713"
    )
    remote.mkdir(parents=True)
    result = run_plan(project)
    assert result.returncode != 0
    assert "quick20 must use local_jobs" in result.stdout


def test_partial_quick20_completion_fails_closed(tmp_path: Path) -> None:
    project = tmp_path / "MOSS-CodecVC"
    record = (
        project
        / "trainset/local_jobs/ver23_batch44_quick20_step10000_20260713"
    )
    record.mkdir(parents=True)
    (record / "metrics.json").write_text("[]\n", encoding="utf-8")
    result = run_plan(project)
    assert result.returncode != 0
    assert "partial quick20 completion evidence" in result.stdout


def test_partial_full320_completion_fails_closed(tmp_path: Path) -> None:
    project = tmp_path / "MOSS-CodecVC"
    record = (
        project
        / "trainset/local_jobs/ver23_batch44_paired_full320_step10000_20260713"
    )
    record.mkdir(parents=True)
    (record / "COMPLETED.json").write_text("{}\n", encoding="utf-8")
    result = run_plan(project)
    assert result.returncode != 0
    assert "partial full320 completion evidence" in result.stdout

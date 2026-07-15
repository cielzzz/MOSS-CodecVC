from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WATCHER = ROOT / "scripts/004129_watch_batch44_r3_warmstart_closure_local.sh"
PYTHON = "/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python"
JOB_ID = "job-165f3b1d-8c7d-47d8-8882-bdb86ef642ab"
CONTRACT_SHA = "2d686e5e57b70fcaa3db8c8eb2b306003a38599b2c9ac37023979d80b6d9fc34"


def write(path: Path, text: str = "x\n", *, mode: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if mode is not None:
        path.chmod(mode)
    return path


def base_env(tmp_path: Path) -> tuple[dict[str, str], dict[str, Path]]:
    project = tmp_path / "project"
    state = tmp_path / "state"
    closure = tmp_path / "closure"
    invoked = tmp_path / "runner.invoked"
    report_invoked = tmp_path / "report.invoked"
    runner = write(
        tmp_path / "fake_runner.sh",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"printf '%s\\n' \"${{ACTION:-unset}}:${{EFFECTIVE_STEP:-unset}}\" >> {invoked}\n",
        mode=0o755,
    )
    selector = write(
        tmp_path / "fake_selector.py",
        "from __future__ import annotations\n"
        "import argparse,json\n"
        "from pathlib import Path\n"
        "def audit_contract(project): return ({}, {})\n"
        "def build_selection(project):\n"
        " return {'status':'selected','selected_candidate_ids':['r3_effective-24000','r3_effective-26000'],"
        "'best2':[{'effective_step':24000},{'effective_step':26000}]}\n"
        "if __name__ == '__main__':\n"
        " p=argparse.ArgumentParser(); p.add_argument('--project-root'); p.add_argument('--output-dir'); a=p.parse_args()\n"
        " o=Path(a.output_dir); o.mkdir(parents=True,exist_ok=True); s=build_selection(Path(a.project_root))\n"
        " (o/'best2_r3_selection.json').write_text(json.dumps(s)+'\\n')\n"
        " (o/'best2_r3_summary.md').write_text('selected\\n')\n",
    )
    report = write(
        tmp_path / "fake_report.py",
        "from __future__ import annotations\n"
        "import argparse,json\n"
        "from pathlib import Path\n"
        f"Path({str(report_invoked)!r}).write_text('yes\\n')\n"
        "p=argparse.ArgumentParser(); p.add_argument('--project-root'); p.add_argument('--output-dir');"
        "p.add_argument('--best2-selection'); p.add_argument('--final-selection'); a=p.parse_args()\n"
        "o=Path(a.output_dir); o.mkdir(parents=True,exist_ok=True)\n"
        "(o/'closure_manifest.json').write_text(json.dumps({'status':'interim'})+'\\n')\n",
    )
    validator = write(
        tmp_path / "fake_quick_validator.py",
        "def validate_completion(record, **kwargs):\n"
        f" return {{'warm_start_contract_sha256': {CONTRACT_SHA!r}}}\n",
    )
    env = os.environ.copy()
    env.update(
        {
            "BATCH44_R3_WARMSTART_CLOSURE_WATCHER_TEST_MODE": "1",
            "PROJECT_ROOT": str(project),
            "STATE_ROOT": str(state),
            "CLOSURE_ROOT": str(closure),
            "FULL320_RUNNER": str(runner),
            "BEST2_SELECTOR": str(selector),
            "REPORT_BUILDER": str(report),
            "QUICK20_VALIDATOR": str(validator),
            "PYTHON": PYTHON,
            "POLL_SECONDS": "1",
            "MAX_SCANS": "1",
            "MIN_CHECKPOINT_AGE_SEC": "0",
        }
    )
    return env, {
        "project": project,
        "state": state,
        "closure": closure,
        "runner": runner,
        "invoked": invoked,
        "report_invoked": report_invoked,
    }


def run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(WATCHER)],
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def make_checkpoint(project: Path, effective: int) -> Path:
    local = effective - 10000
    checkpoint = (
        project
        / "outputs/lora_runs/ver2_9_5_final_r3_v1_warmstart10k_to30k"
        / f"step-{local}"
    )
    write(checkpoint / "adapter_model.safetensors")
    write(checkpoint / "adapter_config.json", "{}\n")
    write(checkpoint / "README.md")
    write(checkpoint / "timbre_memory_adapter.pt")
    write(checkpoint / "timbre_memory_config.json", "{}\n")
    return checkpoint


def make_full_completion(project: Path, effective: int) -> None:
    local = effective - 10000
    record = (
        project
        / "trainset/local_jobs"
        / f"ver23_batch44_r3_warmstart_full320_step{effective}_20260713"
    )
    eval_root = (
        project
        / "testset/outputs/ver23_batch44_r3_warmstart_full320_20260713"
        / f"step-{effective}"
    )
    checkpoint = make_checkpoint(project, effective)
    generic = write(eval_root / "generic.json", "{}\n")
    unified = eval_root / "aggregate/unified_eval_input.jsonl"
    rows = []
    for index in range(320):
        rows.append(
            {
                "case_id": f"case-{index}",
                "status": "ok",
                "metadata": {"mode": "no_text" if index < 160 else "text"},
            }
        )
    write(unified, "".join(json.dumps(row) + "\n" for row in rows))
    metrics = []
    for scope, n in (("no_text", 160), ("text", 160), ("all", 320)):
        metrics.append(
            {
                "scope": scope,
                "effective_step": effective,
                "continuation_local_step": local,
                "n": n,
                "fail_rate": 0.1,
                "qwen_primary_error": 0.1,
                "qwen_cer": 0.1,
                "qwen_wer": 0.1,
                "wavlm_sim_ref": 0.45,
                "wavlm_sim_src": 0.40,
                "wavlm_margin": 0.05,
                "speechbrain_sim_ref": 0.50,
                "speechbrain_sim_src": 0.25,
                "speechbrain_margin": 0.25,
            }
        )
    metrics_path = eval_root / "aggregate/metrics.json"
    write(metrics_path, json.dumps(metrics) + "\n")
    required_names = {
        "validation", "binding", "runtime", "summary", "qwen_asr",
        "dual_cases", "dual_summary", "diagnostics", "metrics_tsv", "metrics_md",
        "fail_reasons_json", "fail_reasons_md", "unified_eval_input_summary",
        "manifest_shard0", "manifest_shard1", "infer_log_shard0", "infer_log_shard1",
    }
    artifacts = {name: artifact(generic) for name in required_names}
    artifacts["metrics_json"] = artifact(metrics_path)
    artifacts["unified_eval_input"] = artifact(unified)
    checkpoint_files = {
        name: artifact(checkpoint / name)
        for name in (
            "adapter_model.safetensors", "adapter_config.json", "README.md",
            "timbre_memory_adapter.pt", "timbre_memory_config.json",
        )
    }
    completion = {
        "schema": "moss_codecvc.batch44_r3_warmstart_full320_local.v1",
        "status": "complete",
        "backend": "local",
        "base_effective_step": 10000,
        "effective_step": effective,
        "continuation_local_step": local,
        "arm": "r3",
        "text_repeat": 3,
        "train_job_id": JOB_ID,
        "record_root": str(record.resolve()),
        "eval_root": str(eval_root.resolve()),
        "expected_warm_start_contract_sha256": CONTRACT_SHA,
        "checkpoint": {"path": str(checkpoint.resolve()), "files": checkpoint_files},
        "run": {
            "validation_rows": 320,
            "inference_rows": 320,
            "qwen_asr_rows": 320,
            "audio_rows": 320,
            "bnf_audit": {
                "run_case_counts": {"no_text": 160, "text": 160},
                "bnf_extraction_counts": {"no_text": 160, "text": 0},
            },
        },
        "metrics": metrics,
        "artifacts": artifacts,
    }
    completion_path = write(record / "COMPLETED.json", json.dumps(completion) + "\n")
    marker = {
        "schema": "moss_codecvc.batch44_r3_warmstart_full320_marker.v1",
        "status": "complete",
        "backend": "local",
        "effective_step": effective,
        "continuation_local_step": local,
        "completion_json": str(completion_path.resolve()),
        "completion_sha256": hashlib.sha256(completion_path.read_bytes()).hexdigest(),
    }
    write(record / "complete.marker", json.dumps(marker) + "\n")


def test_watcher_syntax_and_registered_chain() -> None:
    source = WATCHER.read_text(encoding="utf-8")
    assert subprocess.run(["bash", "-n", str(WATCHER)], check=False).returncode == 0
    for token in (
        "EARLY_FULL320_STEP=\"20000\"",
        'BEST2_CANDIDATE_STEPS="24000 26000 28000 30000"',
        "004127_run_batch44_r3_warmstart_full320_local.sh",
        "004126_select_batch44_r3_warmstart_best2.py",
        "004128_build_batch44_completion_reports.py",
        "CONFIRM_BATCH44_CLOSURE_WATCHER",
        "CONFIRM_BATCH44_LOCAL_EVALUATIONS",
        "CONFIRM_BATCH44_MONITOR_LOOP",
        "FINAL_SELECTION_DISABLED_DO_NOT_CREATE.json",
    ):
        assert token in source
    assert "004107" not in source
    assert "FINAL_SELECTION.json" not in source


def test_default_plan_creates_no_state(tmp_path: Path) -> None:
    env, paths = base_env(tmp_path)
    env.update({"ACTION": "plan", "MODE": "once"})
    result = run(env)
    assert result.returncode == 0, result.stderr
    assert "no state, GPU work, or monitor started" in result.stdout
    assert not paths["state"].exists()
    assert not paths["closure"].exists()


def test_run_requires_two_confirms(tmp_path: Path) -> None:
    env, _ = base_env(tmp_path)
    env.update({"ACTION": "run", "MODE": "once"})
    result = run(env)
    assert result.returncode != 0
    assert "CONFIRM_BATCH44_CLOSURE_WATCHER=1" in result.stderr


def test_monitor_requires_third_confirm(tmp_path: Path) -> None:
    env, _ = base_env(tmp_path)
    env.update(
        {
            "ACTION": "run",
            "MODE": "monitor",
            "CONFIRM_BATCH44_CLOSURE_WATCHER": "1",
            "CONFIRM_BATCH44_LOCAL_EVALUATIONS": "1",
        }
    )
    result = run(env)
    assert result.returncode != 0
    assert "CONFIRM_BATCH44_MONITOR_LOOP=1" in result.stderr


def test_once_waits_for_effective20k_checkpoint_without_dispatch(tmp_path: Path) -> None:
    env, paths = base_env(tmp_path)
    env.update(
        {
            "ACTION": "run",
            "MODE": "once",
            "CONFIRM_BATCH44_CLOSURE_WATCHER": "1",
            "CONFIRM_BATCH44_LOCAL_EVALUATIONS": "1",
        }
    )
    result = run(env)
    assert result.returncode == 0, result.stderr
    scan = json.loads((paths["state"] / "scan_latest.json").read_text())
    assert scan["next_action"] == "run_full320:20000"
    assert not paths["invoked"].exists()


def test_partial_full320_evidence_halts_fail_closed(tmp_path: Path) -> None:
    env, paths = base_env(tmp_path)
    partial = (
        paths["project"]
        / "trainset/local_jobs/ver23_batch44_r3_warmstart_full320_step20000_20260713"
    )
    write(partial / "COMPLETED.json", "{}\n")
    env.update(
        {
            "ACTION": "run",
            "MODE": "once",
            "CONFIRM_BATCH44_CLOSURE_WATCHER": "1",
            "CONFIRM_BATCH44_LOCAL_EVALUATIONS": "1",
        }
    )
    result = run(env)
    assert result.returncode != 0
    halted = json.loads((paths["state"] / "HALTED.json").read_text())
    assert halted["status"] == "halted"
    assert "partial full320" in halted["reason"]


def test_busy_gpu_is_waiting_not_halt_or_dispatch(tmp_path: Path) -> None:
    env, paths = base_env(tmp_path)
    make_checkpoint(paths["project"], 20000)
    fake_smi = write(
        tmp_path / "fake-nvidia-smi",
        "#!/usr/bin/env bash\n"
        "case \"$*\" in\n"
        "  *query-gpu=index*) printf '0\\n1\\n' ;;\n"
        "  *query-gpu=name*) printf 'NVIDIA GeForce RTX 4090\\nNVIDIA GeForce RTX 4090\\n' ;;\n"
        "  *query-gpu=memory.used*) printf '4096\\n4096\\n' ;;\n"
        "  *) exit 1 ;;\n"
        "esac\n",
        mode=0o755,
    )
    env.update(
        {
            "ACTION": "run",
            "MODE": "once",
            "CONFIRM_BATCH44_CLOSURE_WATCHER": "1",
            "CONFIRM_BATCH44_LOCAL_EVALUATIONS": "1",
            "NVIDIA_SMI": str(fake_smi),
            "MAX_INITIAL_GPU_MEMORY_MIB": "2048",
        }
    )
    result = run(env)
    assert result.returncode == 0, result.stderr
    waiting = json.loads((paths["state"] / "WAITING_GPU.json").read_text())
    assert waiting["effective_step"] == 20000
    assert waiting["automatic_retry"] is True
    assert not paths["invoked"].exists()
    assert not (paths["state"] / "HALTED.json").exists()


def test_accepted_full320_refreshes_004128_once(tmp_path: Path) -> None:
    env, paths = base_env(tmp_path)
    make_full_completion(paths["project"], 20000)
    env.update(
        {
            "ACTION": "run",
            "MODE": "once",
            "CONFIRM_BATCH44_CLOSURE_WATCHER": "1",
            "CONFIRM_BATCH44_LOCAL_EVALUATIONS": "1",
        }
    )
    result = run(env)
    assert result.returncode == 0, result.stderr
    marker = paths["state"] / "report_refreshed_effective-20000.json"
    assert marker.is_file()
    assert paths["report_invoked"].is_file()
    assert (paths["closure"] / "closure_manifest.json").is_file()
    scan = json.loads((paths["state"] / "scan_latest.json").read_text())
    assert scan["next_action"] == "wait_quick20_candidates"

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "scripts/004110_submit_batch44_v1_quick20_qz.sh"
WATCH = ROOT / "scripts/004111_watch_batch44_v1_quick20.sh"
PYTHON = Path(
    "/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python"
)


def run_bash(script: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    merged.update(env)
    return subprocess.run(
        ["bash", str(script)],
        cwd=ROOT,
        env=merged,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def write_checkpoint(project: Path, run_dir: Path, step: int, repeat: int) -> None:
    checkpoint = run_dir / f"step-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"a" * 1_000_001)
    (checkpoint / "timbre_memory_adapter.pt").write_bytes(b"b" * 1_000_001)
    (checkpoint / "README.md").write_text("checkpoint\n", encoding="utf-8")
    (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    config = {
        "content_cross_attn_enabled": True,
        "content_cross_attn_layers": "all",
        "content_cross_attn_feature_dim": 768,
        "content_encoder_layers": 2,
        "num_memory_tokens": 0,
        "timbre_side_only": False,
        "source_semantic_memory_enabled": False,
        "speaker_side_pathway_enabled": False,
        "speaker_cross_attn_enabled": False,
    }
    (checkpoint / "timbre_memory_config.json").write_text(
        json.dumps(config) + "\n", encoding="utf-8"
    )
    no_text = (
        project
        / "trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/no_text.train.jsonl"
    )
    text = (
        project
        / "trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/text.train.jsonl"
    )
    arm = "r3" if repeat == 3 else "r5"
    identity_dir = (
        project
        / "trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713"
        / arm
    )
    identity_dir.mkdir(parents=True, exist_ok=True)
    args = {
        "OUT_DIR": str(run_dir),
        "TRAIN_JSONL_SPEC": f"{no_text}::repeat=1,{text}::repeat={repeat}",
        "TEXT_REPEAT": str(repeat),
        "MAX_TRAIN_STEPS": "30000",
        "SAVE_STEPS": "2000",
        "EVAL_STEPS": "2000",
        "LEARNING_RATE": "1e-5",
        "LR_SCHEDULER_TYPE": "constant_with_warmup",
        "WARMUP_RATIO": "0.03",
    }
    (identity_dir / "train_args_dry_run_core.json").write_text(
        json.dumps(args) + "\n", encoding="utf-8"
    )


def watcher_env(tmp_path: Path) -> dict[str, str]:
    project = tmp_path / "MOSS-CodecVC"
    r3 = project / "outputs/lora_runs/ver2_9_5_final_r3_v1_30k"
    r5 = project / "outputs/lora_runs/ver2_9_5_final_r5_v1_30k"
    project.mkdir(parents=True)
    return {
        "BATCH44_QUICK20_TEST_MODE": "1",
        "PROJECT_ROOT": str(project),
        "R3_RUN_DIR": str(r3),
        "R5_RUN_DIR": str(r5),
        "STATE_ROOT": str(project / "scheduler_state"),
        "EVAL_ROOT": str(project / "eval"),
        "PYTHON": str(PYTHON),
        "MODE": "once",
        "ACTION": "plan",
        "MIN_CHECKPOINT_AGE_SEC": "0",
        "POLL_SECONDS": "1",
    }


def metric_rows(step: int) -> list[dict[str, object]]:
    rows = []
    for arm, base in (("r3", 0.10), ("r5", 0.09)):
        for mode in ("no_text", "text"):
            rows.append(
                {
                    "step": step,
                    "arm": arm,
                    "mode": mode,
                    "n": 20,
                    "keep": 18,
                    "fail": 0.10,
                    "cer": base,
                    "sim_ref": 0.45 if arm == "r3" else 0.46,
                    "sim_src": 0.39,
                    "margin": 0.06 if arm == "r3" else 0.07,
                    "ref_bound_count": 11,
                    "ref_bound": 0.55,
                    "ref_content_f1": 0.05,
                    "text_en_src_quick_n": 12 if mode == "text" else "",
                    "text_en_src_quick_fail": 0.1667 if mode == "text" else "",
                    "text_en_src_scope": (
                        "quick20 proxy n=12; not the full text en_src n=80 gate"
                        if mode == "text"
                        else ""
                    ),
                    "run_id": f"{arm}_{mode}_{step}",
                    "output_dir": f"/fake/{arm}/{mode}/{step}",
                }
            )
    return rows


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def write_valid_completion(
    record: Path,
    rows: list[dict[str, object]],
    step: int,
    *,
    ledger_step: int | None = None,
    ledger_job_id: str = "job-11111111-1111-1111-1111-111111111111",
    completion_job_id: str | None = None,
) -> None:
    record.mkdir(parents=True, exist_ok=True)
    metrics_json = record / "metrics.json"
    metrics_tsv = record / "metrics.tsv"
    metrics_md = record / "metrics.md"
    metrics_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with metrics_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    metrics_md.write_text(f"# quick20 step-{step}\n", encoding="utf-8")

    ledger = record / "submitted_jobs.tsv"
    job_name = f"ver23_batch44_quick20_step{step}_20260713"
    with ledger.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("job_name", "job_id", "step"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "job_name": job_name,
                "job_id": ledger_job_id,
                "step": ledger_step if ledger_step is not None else step,
            }
        )

    completion = record / "COMPLETED.json"
    completion.write_text(
        json.dumps(
            {
                "schema": "moss_codecvc.batch44_v1_quick20_completion.v1",
                "status": "complete",
                "step": step,
                "record_root": str(record.resolve()),
                "evaluation_job": {
                    "job_name": job_name,
                    "job_id": completion_job_id or ledger_job_id,
                    "submission_ledger": artifact(ledger),
                },
                "metrics": {
                    "json": artifact(metrics_json),
                    "tsv": artifact(metrics_tsv),
                    "md": artifact(metrics_md),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    marker = {
        "schema": "moss_codecvc.batch44_v1_quick20_complete_marker.v1",
        "status": "complete",
        "step": step,
        "completed_json_sha256": hashlib.sha256(completion.read_bytes()).hexdigest(),
    }
    (record / "complete.marker").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_shell_syntax_and_safety_contract() -> None:
    for script in (SUBMIT, WATCH):
        result = subprocess.run(
            ["bash", "-n", str(script)], cwd=ROOT, text=True, capture_output=True, check=False
        )
        assert result.returncode == 0, result.stderr
    submit_text = SUBMIT.read_text(encoding="utf-8")
    watch_text = WATCH.read_text(encoding="utf-8")
    assert 'DRY_RUN="${DRY_RUN:-1}"' in submit_text
    assert 'CONFIRM_BATCH44_QUICK20="${CONFIRM_BATCH44_QUICK20:-0}"' in submit_text
    assert 'COLLECT_ONLY="${COLLECT_ONLY:-0}"' in submit_text
    assert "lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122" in submit_text
    assert "67b10bc6-78b0-41a3-aaf4-358eeeb99009" in submit_text
    assert "NVIDIA_H200_SXM_141G" in submit_text
    assert "job-2b91d332-d500-4279-84f9-0a6a81a376aa" in submit_text
    assert "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c" in submit_text
    assert "ver2_9_5_final_r3_v1_30k" in submit_text
    assert "ver2_9_5_final_r5_v1_30k" in submit_text
    assert 'r3) printf \'%s\\n\' "ver2_9_5_final_r3" ;;' in submit_text
    assert 'r5) printf \'%s\\n\' "ver2_9_5_final_r5" ;;' in submit_text
    assert 'r3) printf \'%s\\n\' "ver2_9_5_final_r3_v1" ;;' not in submit_text
    assert 'r5) printf \'%s\\n\' "ver2_9_5_final_r5_v1" ;;' not in submit_text
    assert '"$(arm_label r3)" "$(arm_label r5)" <<\'PY\'' in submit_text
    assert 'labels = {"r3": sys.argv[4], "r5": sys.argv[5]}' in submit_text
    assert 'labels = {"r3": "ver2_9_5_final_r3"' not in submit_text
    assert 'expected_ledger_paths = {' in submit_text
    assert 'Path(str(submitted.get(key) or "")).expanduser().resolve()' in submit_text
    assert "3637b90fe14bbda7c4915362d5bd726b072833f07853df4fc555fde0d2c32d7b" in submit_text
    assert "161f05b766454ce7d3e38af1772a1126ec612eaeb0682678bd6a46ca1d62daff" in submit_text
    for line in (
        "run_eval r3 no_text 0,1",
        "run_eval r3 text 2,3",
        "run_eval r5 no_text 4,5",
        "run_eval r5 text 6,7",
    ):
        assert line in submit_text
    assert "quick20 proxy n=12; not the full text en_src n=80 gate" in submit_text
    assert 'if mode == "text"' in submit_text
    assert "no_text quick20 unexpectedly contains en_src cells" not in submit_text
    assert "2000 4000 6000" in watch_text
    assert 'ACTION="${ACTION:-plan}"' in watch_text
    assert 'ALLOW_LIVE_SUBMIT="${ALLOW_LIVE_SUBMIT:-0}"' in watch_text
    assert "ALERT_NEGATIVE_NO_TEXT_MARGIN.json" in watch_text
    assert "STOP_RECOMMENDATION.md" in watch_text
    assert "watcher does not stop training jobs" in watch_text
    assert 'if margin < 0.0:' in watch_text


def test_static_protocol_audit_does_not_need_checkpoints(tmp_path: Path) -> None:
    result = run_bash(
        SUBMIT,
        {
            "STATIC_AUDIT_ONLY": "1",
            "DRY_RUN": "1",
            "STEP": "2000",
            "RECORD_ROOT": str(tmp_path / "record"),
            "EVAL_ROOT": str(tmp_path / "eval"),
        },
    )
    assert result.returncode == 0, result.stdout
    assert "static audit passed" in result.stdout
    text20 = tmp_path / "record/ver23_batch44_text_quick20_8cell_20260713.jsonl"
    assert len(text20.read_text(encoding="utf-8").splitlines()) == 20


def test_watcher_discovers_only_same_step_ready_pair(tmp_path: Path) -> None:
    env = watcher_env(tmp_path)
    project = Path(env["PROJECT_ROOT"])
    r3 = Path(env["R3_RUN_DIR"])
    r5 = Path(env["R5_RUN_DIR"])
    write_checkpoint(project, r3, 2000, 3)
    write_checkpoint(project, r5, 2000, 5)
    write_checkpoint(project, r3, 4000, 3)

    result = run_bash(WATCH, env)
    assert result.returncode == 0, result.stdout
    assert "next_ready_step=2000" in result.stdout
    rows = list(
        csv.DictReader(
            (Path(env["STATE_ROOT"]) / "scan_latest.tsv").open(encoding="utf-8"), delimiter="\t"
        )
    )
    by_step = {int(row["step"]): row for row in rows}
    assert by_step[2000]["status"] == "ready"
    assert by_step[4000]["status"] == "waiting"
    assert "r5:missing" in by_step[4000]["detail_r5"]


def test_rollup_deduplicates_and_labels_en_src_proxy(tmp_path: Path) -> None:
    env = watcher_env(tmp_path)
    project = Path(env["PROJECT_ROOT"])
    record = project / "trainset/qz_jobs/ver23_batch44_quick20_step2000_20260713"
    rows = metric_rows(2000)
    write_valid_completion(record, rows, 2000)

    result = run_bash(WATCH, env)
    assert result.returncode == 0, result.stdout
    rollup = Path(env["EVAL_ROOT"]) / "metrics_all.md"
    text = rollup.read_text(encoding="utf-8")
    assert "Completed paired checkpoints: 1/15" in text
    assert "12-case quick20 proxy" in text
    all_rows = list(
        csv.DictReader(
            (Path(env["EVAL_ROOT"]) / "metrics_all.tsv").open(encoding="utf-8"), delimiter="\t"
        )
    )
    assert len(all_rows) == 4
    assert len({(row["step"], row["arm"], row["mode"]) for row in all_rows}) == 4


def test_test_mode_cannot_submit_live(tmp_path: Path) -> None:
    env = watcher_env(tmp_path)
    env.update({"ACTION": "submit", "ALLOW_LIVE_SUBMIT": "1"})
    result = run_bash(WATCH, env)
    assert result.returncode != 0
    assert "test mode may not submit live jobs" in result.stdout


def test_child_wrapper_failure_propagates_without_false_completion(tmp_path: Path) -> None:
    env = watcher_env(tmp_path)
    project = Path(env["PROJECT_ROOT"])
    write_checkpoint(project, Path(env["R3_RUN_DIR"]), 2000, 3)
    write_checkpoint(project, Path(env["R5_RUN_DIR"]), 2000, 5)
    failing_wrapper = tmp_path / "fail_quick20_wrapper.sh"
    failing_wrapper.write_text(
        "#!/usr/bin/env bash\necho fake-quick20-wrapper-failure >&2\nexit 7\n",
        encoding="utf-8",
    )
    failing_wrapper.chmod(0o755)
    env.update({"ACTION": "dry-run", "SUBMIT_WRAPPER": str(failing_wrapper)})

    result = run_bash(WATCH, env)
    assert result.returncode == 7, result.stdout
    assert "fake-quick20-wrapper-failure" in result.stdout
    record = project / "trainset/qz_jobs/ver23_batch44_quick20_step2000_20260713"
    assert not (record / "dry_run.ok").exists()
    assert not (record / "complete.marker").exists()


def test_negative_no_text_margin_stops_scheduler_and_writes_recommendation(
    tmp_path: Path,
) -> None:
    env = watcher_env(tmp_path)
    project = Path(env["PROJECT_ROOT"])
    record = project / "trainset/qz_jobs/ver23_batch44_quick20_step2000_20260713"
    rows = metric_rows(2000)
    for row in rows:
        if row["arm"] == "r5" and row["mode"] == "no_text":
            row["sim_ref"] = 0.38
            row["sim_src"] = 0.44
            row["margin"] = -0.06
    write_valid_completion(record, rows, 2000)

    # A later pair is ready, but the negative completed margin must prevent it
    # from being selected or submitted.
    write_checkpoint(project, Path(env["R3_RUN_DIR"]), 4000, 3)
    write_checkpoint(project, Path(env["R5_RUN_DIR"]), 4000, 5)
    result = run_bash(WATCH, env)
    assert result.returncode == 0, result.stdout
    assert "ALERT active: scheduling stopped" in result.stdout
    assert "next_ready_step=4000" not in result.stdout

    state = Path(env["STATE_ROOT"])
    payload = json.loads(
        (state / "ALERT_NEGATIVE_NO_TEXT_MARGIN.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "alert"
    assert payload["scheduler_action"] == "stop scheduling further quick20 evaluations"
    assert payload["training_action"] == (
        "recommend stop only; watcher does not stop training jobs"
    )
    assert payload["alerts"] == [
        {
            "arm": "r5",
            "cer": 0.09,
            "margin": -0.06,
            "metrics_tsv": str(record / "metrics.tsv"),
            "record_root": str(record),
            "sim_ref": 0.38,
            "sim_src": 0.44,
            "step": 2000,
            "training_job_id": "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c",
        }
    ]
    recommendation = (state / "STOP_RECOMMENDATION.md").read_text(encoding="utf-8")
    assert "performs no automatic training stop operation" in recommendation
    assert not (record.parent / "ver23_batch44_quick20_step4000_20260713/submitted_jobs.tsv").exists()


def test_marker_without_completed_json_fails_closed(tmp_path: Path) -> None:
    env = watcher_env(tmp_path)
    project = Path(env["PROJECT_ROOT"])
    record = project / "trainset/qz_jobs/ver23_batch44_quick20_step2000_20260713"
    rows = metric_rows(2000)
    write_valid_completion(record, rows, 2000)
    (record / "COMPLETED.json").unlink()

    result = run_bash(WATCH, env)
    assert result.returncode != 0
    assert "missing/empty COMPLETED.json" in result.stdout
    assert "refusing to advance" in result.stdout


def test_marker_completed_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    env = watcher_env(tmp_path)
    project = Path(env["PROJECT_ROOT"])
    record = project / "trainset/qz_jobs/ver23_batch44_quick20_step2000_20260713"
    write_valid_completion(record, metric_rows(2000), 2000)
    marker = json.loads((record / "complete.marker").read_text(encoding="utf-8"))
    marker["completed_json_sha256"] = "0" * 64
    (record / "complete.marker").write_text(json.dumps(marker) + "\n", encoding="utf-8")

    result = run_bash(WATCH, env)
    assert result.returncode != 0
    assert "complete.marker COMPLETED.json SHA mismatch" in result.stdout


def test_completed_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    for name, field, value in (
        ("schema", "schema", "wrong.schema"),
        ("status", "status", "partial"),
        ("step", "step", 4000),
    ):
        case = tmp_path / name
        env = watcher_env(case)
        project = Path(env["PROJECT_ROOT"])
        record = project / "trainset/qz_jobs/ver23_batch44_quick20_step2000_20260713"
        write_valid_completion(record, metric_rows(2000), 2000)
        completion_path = record / "COMPLETED.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion[field] = value
        completion_path.write_text(json.dumps(completion) + "\n", encoding="utf-8")
        marker_path = record / "complete.marker"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["completed_json_sha256"] = hashlib.sha256(
            completion_path.read_bytes()
        ).hexdigest()
        marker_path.write_text(json.dumps(marker) + "\n", encoding="utf-8")

        result = run_bash(WATCH, env)
        assert result.returncode != 0
        assert "COMPLETED.json identity mismatch" in result.stdout


def test_metrics_artifact_tamper_fails_closed(tmp_path: Path) -> None:
    env = watcher_env(tmp_path)
    project = Path(env["PROJECT_ROOT"])
    record = project / "trainset/qz_jobs/ver23_batch44_quick20_step2000_20260713"
    write_valid_completion(record, metric_rows(2000), 2000)
    with (record / "metrics.md").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")

    result = run_bash(WATCH, env)
    assert result.returncode != 0
    assert "metrics.md artifact size mismatch" in result.stdout


def test_metrics_artifact_same_size_sha_tamper_fails_closed(tmp_path: Path) -> None:
    env = watcher_env(tmp_path)
    project = Path(env["PROJECT_ROOT"])
    record = project / "trainset/qz_jobs/ver23_batch44_quick20_step2000_20260713"
    write_valid_completion(record, metric_rows(2000), 2000)
    metrics_md = record / "metrics.md"
    original = metrics_md.read_text(encoding="utf-8")
    metrics_md.write_text("X" + original[1:], encoding="utf-8")

    result = run_bash(WATCH, env)
    assert result.returncode != 0
    assert "metrics.md artifact SHA mismatch" in result.stdout


def test_ledger_step_or_job_mismatch_fails_closed(tmp_path: Path) -> None:
    for name, kwargs, expected in (
        ("step", {"ledger_step": 4000}, "submission ledger step mismatch"),
        (
            "job",
            {"completion_job_id": "job-22222222-2222-2222-2222-222222222222"},
            "completion/ledger job_id mismatch",
        ),
    ):
        case = tmp_path / name
        env = watcher_env(case)
        project = Path(env["PROJECT_ROOT"])
        record = project / "trainset/qz_jobs/ver23_batch44_quick20_step2000_20260713"
        write_valid_completion(record, metric_rows(2000), 2000, **kwargs)

        result = run_bash(WATCH, env)
        assert result.returncode != 0
        assert expected in result.stdout

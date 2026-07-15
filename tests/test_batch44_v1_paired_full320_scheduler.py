from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "scripts/004112_submit_batch44_v1_paired_full320_qz.sh"
WATCH = ROOT / "scripts/004113_watch_batch44_v1_paired_full320.sh"
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
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"a" * 1_000_001)
    (checkpoint / "timbre_memory_adapter.pt").write_bytes(b"b" * 1_000_001)
    (checkpoint / "README.md").write_text("checkpoint\n", encoding="utf-8")
    (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    config = {
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
        "ENABLE_CONTENT_CROSS_ATTN": "1",
        "CONTENT_CROSS_ATTN_LAYERS": "all",
        "GUIDED_ATTN_LOSS_WEIGHT": "0.05",
        "PHONEME_CLASSIFIER_LOSS_WEIGHT": "0.02",
        "CONTENT_CTC_WEIGHT": "0.0",
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
        "BATCH44_FULL320_TEST_MODE": "1",
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
    rows: list[dict[str, object]] = []
    for arm, repeat in (("r3", 3), ("r5", 5)):
        for scope, n in (("no_text", 160), ("text", 160), ("all", 320)):
            en_scope = scope in {"text", "all"}
            rows.append(
                {
                    "step": step,
                    "arm": arm,
                    "text_repeat": repeat,
                    "train_job_id": (
                        "job-2b91d332-d500-4279-84f9-0a6a81a376aa"
                        if arm == "r3"
                        else "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c"
                    ),
                    "scope": scope,
                    "n": n,
                    "keep": n - 16,
                    "fail_count": 16,
                    "fail_rate": 16 / n,
                    "cer": 0.08,
                    "wavlm_sim_ref": 0.45,
                    "wavlm_sim_src": 0.39,
                    "wavlm_margin": 0.06,
                    "wavlm_ref_bound": 0.55,
                    "speechbrain_sim_ref": 0.49,
                    "speechbrain_sim_src": 0.30,
                    "speechbrain_margin": 0.19,
                    "speechbrain_ref_bound": 0.70,
                    "ref_content_lcs_f1": 0.05,
                    "text_en_src_n": 80 if en_scope else "",
                    "text_en_src_fail_count": 8 if en_scope else "",
                    "text_en_src_fail_rate": 0.10 if en_scope else "",
                    "text_en_src_cer": 0.05 if en_scope else "",
                }
            )
    return rows


def mark_complete(
    project: Path,
    eval_root: Path,
    step: int,
    rows: list[dict[str, object]] | None = None,
) -> None:
    record = (
        project
        / f"trainset/qz_jobs/ver23_batch44_paired_full320_step{step}_20260713"
    )
    aggregate = eval_root / f"step-{step}/aggregate"
    record.mkdir(parents=True, exist_ok=True)
    aggregate.mkdir(parents=True, exist_ok=True)
    rows = metric_rows(step) if rows is None else rows
    with (aggregate / "paired_metrics.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (record / "COMPLETED.json").write_text(
        json.dumps(
            {
                "schema": "batch44_v1_paired_full320_v1",
                "status": "complete",
                "step": step,
                "training_jobs": {
                    "r3": "job-2b91d332-d500-4279-84f9-0a6a81a376aa",
                    "r5": "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (record / "complete.marker").write_text("done\n", encoding="utf-8")


def test_shell_syntax_and_safety_contract() -> None:
    for script in (SUBMIT, WATCH):
        result = subprocess.run(
            ["bash", "-n", str(script)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    submit_text = SUBMIT.read_text(encoding="utf-8")
    watch_text = WATCH.read_text(encoding="utf-8")
    assert 'DRY_RUN="${DRY_RUN:-1}"' in submit_text
    assert 'CONFIRM_BATCH44_FULL320="${CONFIRM_BATCH44_FULL320:-0}"' in submit_text
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
    assert "3637b90fe14bbda7c4915362d5bd726b072833f07853df4fc555fde0d2c32d7b" in submit_text
    assert "161f05b766454ce7d3e38af1772a1126ec612eaeb0682678bd6a46ca1d62daff" in submit_text
    assert "ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709" in submit_text
    for forbidden in (
        "ver2_9_prepared_v2_real_no_text_refdecorr_wavlm_sv_20260708",
        "ver2_9_5_final_r3_v2_30k",
        "ver2_9_5_final_r5_v2_30k",
        "job-a34d84d4-59cc-4824-b197-0829bfe79004",
        "job-aef79753-7fcd-444e-b94d-3e21eedb2394",
    ):
        assert forbidden not in submit_text
        assert forbidden not in watch_text
    for lane in (
        "run_eval_lane r3 no_text 0,1",
        "run_eval_lane r3 text 2,3",
        "run_eval_lane r5 no_text 4,5",
        "run_eval_lane r5 text 6,7",
    ):
        assert lane in submit_text
    assert '"schema": "batch44_v1_paired_full320_v1"' in submit_text
    assert 'expected_ledger_paths = {' in submit_text
    assert 'Path(str(submitted.get(key) or "")).expanduser().resolve()' in submit_text
    assert '"text_en_src_n": len(en_src)' in submit_text
    assert '"speechbrain_sim_ref"' in submit_text
    assert '"ref_content_lcs_f1"' in submit_text
    assert "10000|20000|30000" in submit_text
    assert 'STEPS="10000 20000 30000"' in watch_text
    assert 'ACTION="${ACTION:-plan}"' in watch_text
    assert 'ALLOW_LIVE_SUBMIT="${ALLOW_LIVE_SUBMIT:-0}"' in watch_text
    assert "ALERT_FULL320_RED_FLAGS.json" in watch_text
    assert "ALERT_NEGATIVE_NO_TEXT_MARGIN.json" in watch_text
    assert "STOP_RECOMMENDATION.md" in watch_text
    assert "watcher does not call QZ stop or mutate training jobs" in watch_text
    assert "qzcli stop" not in watch_text.lower()
    assert "stop-job" not in watch_text.lower()
    assert "stop_job" not in watch_text.lower()
    assert "004103" not in submit_text
    assert "004105" not in submit_text


def test_completion_accepts_symlink_spelling_for_submission_ledger_paths(
    tmp_path: Path,
) -> None:
    source = SUBMIT.read_text(encoding="utf-8")
    match = re.search(
        r"write_completion\(\) \{.*?<<'PY'\n(?P<body>.*?)\nPY\n\}",
        source,
        flags=re.DOTALL,
    )
    assert match is not None

    physical_qz = tmp_path / "physical_qz"
    physical_qz.mkdir()
    logical_qz = tmp_path / "logical_qz"
    logical_qz.symlink_to(physical_qz, target_is_directory=True)
    record = logical_qz / "record"
    record.mkdir()
    step_root = tmp_path / "eval/step-10000"
    step_root.mkdir(parents=True)
    code_root = tmp_path / "code"
    code_root.mkdir()
    validation = tmp_path / "validation.jsonl"
    validation.write_text("{}\n", encoding="utf-8")
    training_ledger = tmp_path / "training.tsv"
    training_ledger.write_text("fixture\n", encoding="utf-8")
    completion = record / "COMPLETED.json"
    job_name = "ver23_batch44_r3r5_full320_step10000_20260713"
    job_id = "job-11111111-2222-3333-4444-555555555555"
    compute_group = "lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"
    spec = "67b10bc6-78b0-41a3-aaf4-358eeeb99009"
    r3_job = "job-2b91d332-d500-4279-84f9-0a6a81a376aa"
    r5_job = "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c"
    fields = [
        "job_name",
        "job_id",
        "step",
        "compute_group",
        "spec",
        "record_root",
        "step_root",
        "code_root",
        "r3_train_job_id",
        "r5_train_job_id",
    ]
    values = [
        job_name,
        job_id,
        "10000",
        compute_group,
        spec,
        str(record),
        str(step_root),
        str(code_root),
        r3_job,
        r5_job,
    ]
    (record / "submitted_jobs.tsv").write_text(
        "\t".join(fields) + "\n" + "\t".join(values) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-",
            str(completion),
            "10000",
            str(record.resolve()),
            str(step_root),
            str(code_root),
            str(validation),
            str(training_ledger),
            r3_job,
            r5_job,
            compute_group,
            spec,
            job_name,
        ],
        input=match.group("body"),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(completion.read_text(encoding="utf-8"))
    assert payload["record_root"] == str(record.resolve())
    assert payload["evaluation_job"]["job_id"] == job_id
    assert (record / "complete.marker").is_file()


def test_static_protocol_audit_needs_no_future_checkpoint(tmp_path: Path) -> None:
    result = run_bash(
        SUBMIT,
        {
            "STEP": "20000",
            "STATIC_AUDIT_ONLY": "1",
            "DRY_RUN": "1",
            "RECORD_ROOT": str(tmp_path / "record"),
            "EVAL_ROOT": str(tmp_path / "eval"),
        },
    )
    assert result.returncode == 0, result.stdout
    assert "static audit passed; checkpoints and QZ were not touched" in result.stdout
    assert "no_text=160 text=160 text_en_src=80" in result.stdout
    resolved = (tmp_path / "record/resolved_runs.tsv").read_text(encoding="utf-8")
    assert len(resolved.splitlines()) == 5


def test_watcher_requires_same_step_pair_and_preserves_chronology(tmp_path: Path) -> None:
    env = watcher_env(tmp_path)
    project = Path(env["PROJECT_ROOT"])
    r3 = Path(env["R3_RUN_DIR"])
    r5 = Path(env["R5_RUN_DIR"])

    # Later checkpoints are ready, but step-10k lacks r5. The watcher must
    # neither skip the registered 10k node nor mix steps between arms.
    write_checkpoint(project, r3, 10000, 3)
    write_checkpoint(project, r3, 20000, 3)
    write_checkpoint(project, r5, 20000, 5)
    write_checkpoint(project, r3, 30000, 3)
    write_checkpoint(project, r5, 30000, 5)
    result = run_bash(WATCH, env)
    assert result.returncode == 0, result.stdout
    assert "no paired checkpoint ready" in result.stdout
    assert "next_ready_step=20000" not in result.stdout
    assert "next_ready_step=30000" not in result.stdout

    # Once r5 step-10k appears, the exact same-step pair is selected first.
    write_checkpoint(project, r5, 10000, 5)
    result = run_bash(WATCH, env)
    assert result.returncode == 0, result.stdout
    assert "next_ready_step=10000" in result.stdout

    # Each later node is gated by verified completion of the preceding node.
    mark_complete(project, Path(env["EVAL_ROOT"]), 10000)
    result = run_bash(WATCH, env)
    assert result.returncode == 0, result.stdout
    assert "next_ready_step=20000" in result.stdout
    mark_complete(project, Path(env["EVAL_ROOT"]), 20000)
    result = run_bash(WATCH, env)
    assert result.returncode == 0, result.stdout
    assert "next_ready_step=30000" in result.stdout


def test_rollup_accepts_exact_six_row_schema(tmp_path: Path) -> None:
    env = watcher_env(tmp_path)
    project = Path(env["PROJECT_ROOT"])
    eval_root = Path(env["EVAL_ROOT"])
    mark_complete(project, eval_root, 10000)
    result = run_bash(WATCH, env)
    assert result.returncode == 0, result.stdout
    text = (eval_root / "paired_metrics_all.md").read_text(encoding="utf-8")
    assert "Completed paired checkpoints: 1/3" in text
    assert "full text en_src n=80" in text
    rows = list(
        csv.DictReader(
            (eval_root / "paired_metrics_all.tsv").open(encoding="utf-8"),
            delimiter="\t",
        )
    )
    assert len(rows) == 6
    assert {(row["arm"], row["scope"]) for row in rows} == {
        (arm, scope)
        for arm in ("r3", "r5")
        for scope in ("no_text", "text", "all")
    }


def test_test_mode_cannot_submit_live(tmp_path: Path) -> None:
    env = watcher_env(tmp_path)
    env.update({"ACTION": "submit", "ALLOW_LIVE_SUBMIT": "1"})
    result = run_bash(WATCH, env)
    assert result.returncode != 0
    assert "test mode may not submit live jobs" in result.stdout


def test_unregistered_step_is_rejected_before_qz() -> None:
    result = run_bash(
        SUBMIT,
        {
            "STEP": "12000",
            "STATIC_AUDIT_ONLY": "1",
            "DRY_RUN": "1",
        },
    )
    assert result.returncode != 0
    assert "registered Batch-44 full320 checkpoints 10000/20000/30000" in result.stdout


def test_step_10000_is_static_audit_only_compatible(tmp_path: Path) -> None:
    result = run_bash(
        SUBMIT,
        {
            "STEP": "10000",
            "STATIC_AUDIT_ONLY": "1",
            "DRY_RUN": "1",
            "RECORD_ROOT": str(tmp_path / "record10k"),
            "EVAL_ROOT": str(tmp_path / "eval10k"),
        },
    )
    assert result.returncode == 0, result.stdout
    assert "static audit passed; checkpoints and QZ were not touched" in result.stdout
    # The formal watcher includes all three preregistered paired nodes.
    assert 'STEPS="10000 20000 30000"' in WATCH.read_text(encoding="utf-8")


def test_persistent_live_lock_blocks_resubmission(tmp_path: Path) -> None:
    env = watcher_env(tmp_path)
    project = Path(env["PROJECT_ROOT"])
    write_checkpoint(project, Path(env["R3_RUN_DIR"]), 10000, 3)
    write_checkpoint(project, Path(env["R5_RUN_DIR"]), 10000, 5)
    record = project / "trainset/qz_jobs/ver23_batch44_paired_full320_step10000_20260713"
    (record / ".live_submit.lock").mkdir(parents=True)

    result = run_bash(WATCH, env)
    assert result.returncode == 0, result.stdout
    assert "active_step=10000 waiting for completion" in result.stdout
    assert "next_ready_step=10000" not in result.stdout
    scan = (Path(env["STATE_ROOT"]) / "scan_latest.tsv").read_text(encoding="utf-8")
    assert "10000\tlocked_manual_audit" in scan


def test_child_wrapper_failure_is_fail_closed_and_never_recorded_as_success(
    tmp_path: Path,
) -> None:
    env = watcher_env(tmp_path)
    project = Path(env["PROJECT_ROOT"])
    write_checkpoint(project, Path(env["R3_RUN_DIR"]), 10000, 3)
    write_checkpoint(project, Path(env["R5_RUN_DIR"]), 10000, 5)
    failing_wrapper = tmp_path / "fail_full320_wrapper.sh"
    failing_wrapper.write_text(
        "#!/usr/bin/env bash\necho fake-full320-wrapper-failure >&2\nexit 7\n",
        encoding="utf-8",
    )
    failing_wrapper.chmod(0o755)
    env.update({"ACTION": "dry-run", "SUBMIT_WRAPPER": str(failing_wrapper)})

    result = run_bash(WATCH, env)
    assert result.returncode != 0, result.stdout
    assert "dry-run wrapper failed rc=7" in result.stdout
    actions = (Path(env["STATE_ROOT"]) / "actions.tsv").read_text(encoding="utf-8")
    assert "10000\tdry-run\tfailed_rc_7" in actions
    assert "10000\tdry-run\tsuccess" not in actions


def test_missing_completion_marker_never_advances_chronology(tmp_path: Path) -> None:
    env = watcher_env(tmp_path)
    project = Path(env["PROJECT_ROOT"])
    eval_root = Path(env["EVAL_ROOT"])
    mark_complete(project, eval_root, 10000)
    record = project / "trainset/qz_jobs/ver23_batch44_paired_full320_step10000_20260713"
    (record / "complete.marker").unlink()
    write_checkpoint(project, Path(env["R3_RUN_DIR"]), 20000, 3)
    write_checkpoint(project, Path(env["R5_RUN_DIR"]), 20000, 5)

    result = run_bash(WATCH, env)
    assert result.returncode == 21, result.stdout
    assert "partial completion artifacts; manual audit is required" in result.stdout
    assert "next_ready_step=20000" not in result.stdout
    scan = (Path(env["STATE_ROOT"]) / "scan_latest.tsv").read_text(encoding="utf-8")
    assert "10000\tinconsistent_partial_completion" in scan


def test_negative_no_text_margin_alerts_without_stopping_training(tmp_path: Path) -> None:
    env = watcher_env(tmp_path)
    project = Path(env["PROJECT_ROOT"])
    rows = metric_rows(10000)
    for row in rows:
        if row["arm"] == "r5" and row["scope"] == "no_text":
            row["wavlm_sim_ref"] = 0.38
            row["wavlm_sim_src"] = 0.44
            row["wavlm_margin"] = -0.06
    mark_complete(project, Path(env["EVAL_ROOT"]), 10000, rows)
    write_checkpoint(project, Path(env["R3_RUN_DIR"]), 20000, 3)
    write_checkpoint(project, Path(env["R5_RUN_DIR"]), 20000, 5)

    result = run_bash(WATCH, env)
    assert result.returncode == 0, result.stdout
    assert "ALERT active: later scheduling paused; training jobs were not stopped" in result.stdout
    assert "next_ready_step=20000" not in result.stdout

    state = Path(env["STATE_ROOT"])
    payload = json.loads((state / "ALERT_FULL320_RED_FLAGS.json").read_text(encoding="utf-8"))
    assert payload["status"] == "alert"
    assert payload["training_action"] == (
        "recommend stop only; watcher does not call QZ stop or mutate training jobs"
    )
    assert payload["scheduler_action"] == (
        "stop scheduling later full320 evaluations pending manual review"
    )
    matching = [item for item in payload["alerts"] if item["code"] == "negative_no_text_margin"]
    assert matching == [
        {
            "arm": "r5",
            "cer": 0.08,
            "code": "negative_no_text_margin",
            "metrics_tsv": str(
                Path(env["EVAL_ROOT"]) / "step-10000/aggregate/paired_metrics.tsv"
            ),
            "ref_content_lcs_f1": 0.05,
            "relation": "<",
            "scope": "no_text",
            "step": 10000,
            "threshold": 0.0,
            "training_job_id": "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c",
            "value": -0.06,
            "wavlm_margin": -0.06,
            "wavlm_sim_ref": 0.38,
            "wavlm_sim_src": 0.44,
        }
    ]
    assert (state / "ALERT_NEGATIVE_NO_TEXT_MARGIN.json").is_file()
    recommendation = (state / "STOP_RECOMMENDATION.md").read_text(encoding="utf-8")
    assert "no automatic QZ or training stop operation" in recommendation


def test_20k_registered_red_flags_pause_30k_schedule(tmp_path: Path) -> None:
    env = watcher_env(tmp_path)
    project = Path(env["PROJECT_ROOT"])
    mark_complete(project, Path(env["EVAL_ROOT"]), 10000)
    rows = metric_rows(20000)
    for row in rows:
        if row["arm"] == "r3" and row["scope"] == "no_text":
            row["wavlm_sim_ref"] = 0.405
            row["wavlm_sim_src"] = 0.39
            row["wavlm_margin"] = 0.015
        if row["arm"] == "r5" and row["scope"] == "text":
            row["text_en_src_fail_rate"] = 0.30
            row["text_en_src_fail_count"] = 24
    mark_complete(project, Path(env["EVAL_ROOT"]), 20000, rows)
    write_checkpoint(project, Path(env["R3_RUN_DIR"]), 30000, 3)
    write_checkpoint(project, Path(env["R5_RUN_DIR"]), 30000, 5)

    result = run_bash(WATCH, env)
    assert result.returncode == 0, result.stdout
    assert "ALERT active" in result.stdout
    assert "next_ready_step=30000" not in result.stdout
    payload = json.loads(
        (Path(env["STATE_ROOT"]) / "ALERT_FULL320_RED_FLAGS.json").read_text(
            encoding="utf-8"
        )
    )
    assert {(item["arm"], item["code"]) for item in payload["alerts"]} == {
        ("r3", "no_text_margin_lt_0p02"),
        ("r5", "text_en_src_fail_gt_0p25_at_or_after_20k"),
    }
    assert not (Path(env["STATE_ROOT"]) / "ALERT_NEGATIVE_NO_TEXT_MARGIN.json").exists()


def test_rollup_rejects_forged_training_job_provenance(tmp_path: Path) -> None:
    env = watcher_env(tmp_path)
    project = Path(env["PROJECT_ROOT"])
    rows = metric_rows(10000)
    rows[0]["train_job_id"] = "job-00000000-0000-0000-0000-000000000000"
    mark_complete(project, Path(env["EVAL_ROOT"]), 10000, rows)
    result = run_bash(WATCH, env)
    assert result.returncode != 0
    assert "train_job_id='job-00000000-0000-0000-0000-000000000000'" in result.stdout


def test_checkpoint_identity_must_use_old_v1_data(tmp_path: Path) -> None:
    env = watcher_env(tmp_path)
    project = Path(env["PROJECT_ROOT"])
    r3 = Path(env["R3_RUN_DIR"])
    r5 = Path(env["R5_RUN_DIR"])
    write_checkpoint(project, r3, 10000, 3)
    write_checkpoint(project, r5, 10000, 5)
    args_path = (
        project
        / "trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713/r3/train_args_dry_run_core.json"
    )
    args = json.loads(args_path.read_text(encoding="utf-8"))
    args["TRAIN_JSONL_SPEC"] = args["TRAIN_JSONL_SPEC"].replace(
        "ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/no_text.train.jsonl",
        "ver2_9_prepared_v2_real_no_text_refdecorr_wavlm_sv_20260708/no_text.v2.train.jsonl",
    )
    args_path.write_text(json.dumps(args) + "\n", encoding="utf-8")

    result = run_bash(WATCH, env)
    assert result.returncode == 0, result.stdout
    assert "no paired checkpoint ready" in result.stdout
    scan = (Path(env["STATE_ROOT"]) / "scan_latest.tsv").read_text(encoding="utf-8")
    assert "r3:identity_mismatch:TRAIN_JSONL_SPEC" in scan


def test_checkpoint_stability_age_is_enforced(tmp_path: Path) -> None:
    env = watcher_env(tmp_path)
    env["MIN_CHECKPOINT_AGE_SEC"] = "3600"
    project = Path(env["PROJECT_ROOT"])
    write_checkpoint(project, Path(env["R3_RUN_DIR"]), 10000, 3)
    write_checkpoint(project, Path(env["R5_RUN_DIR"]), 10000, 5)
    result = run_bash(WATCH, env)
    assert result.returncode == 0, result.stdout
    assert "no paired checkpoint ready" in result.stdout
    scan = (Path(env["STATE_ROOT"]) / "scan_latest.tsv").read_text(encoding="utf-8")
    assert "settling:" in scan

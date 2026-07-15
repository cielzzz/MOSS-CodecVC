from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "scripts/004101_submit_batch43_paired_full320_qz.sh"
WATCH = ROOT / "scripts/004102_watch_batch43_paired_full320.sh"
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
        / "trainset/ver2_9_prepared_v2_real_no_text_refdecorr_wavlm_sv_20260708/no_text.v2.train.jsonl"
    )
    text = (
        project
        / "trainset/ver2_9_prepared_v2_real_no_text_refdecorr_wavlm_sv_20260708/text.train.jsonl"
    )
    arm = "r3" if repeat == 3 else "r5"
    identity_dir = (
        project
        / "trainset/qz_jobs/ver23_batch43_ver2_9_5_final_r3_r5_v2_30k_20260712"
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
    r3 = project / "outputs/lora_runs/ver2_9_5_final_r3_v2_30k"
    r5 = project / "outputs/lora_runs/ver2_9_5_final_r5_v2_30k"
    project.mkdir(parents=True)
    return {
        "BATCH43_FULL320_TEST_MODE": "1",
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
                    "train_job_id": f"job-{arm}",
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


def mark_complete(project: Path, eval_root: Path, step: int) -> None:
    record = (
        project
        / f"trainset/qz_jobs/ver23_batch43_paired_full320_step{step}_20260712"
    )
    aggregate = eval_root / f"step-{step}/aggregate"
    record.mkdir(parents=True, exist_ok=True)
    aggregate.mkdir(parents=True, exist_ok=True)
    rows = metric_rows(step)
    with (aggregate / "paired_metrics.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (record / "COMPLETED.json").write_text(
        json.dumps(
            {
                "schema": "batch43_paired_full320_v1",
                "status": "complete",
                "step": step,
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
    assert 'CONFIRM_BATCH43_FULL320="${CONFIRM_BATCH43_FULL320:-0}"' in submit_text
    assert "lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122" in submit_text
    assert "67b10bc6-78b0-41a3-aaf4-358eeeb99009" in submit_text
    assert "NVIDIA_H200_SXM_141G" in submit_text
    for lane in (
        "run_eval_lane r3 no_text 0,1",
        "run_eval_lane r3 text 2,3",
        "run_eval_lane r5 no_text 4,5",
        "run_eval_lane r5 text 6,7",
    ):
        assert lane in submit_text
    assert '"schema": "batch43_paired_full320_v1"' in submit_text
    assert '"text_en_src_n": len(en_src)' in submit_text
    assert '"speechbrain_sim_ref"' in submit_text
    assert '"ref_content_lcs_f1"' in submit_text
    assert "10000|20000|30000" in submit_text
    assert 'STEPS="20000 30000"' in watch_text
    assert 'ACTION="${ACTION:-plan}"' in watch_text
    assert 'ALLOW_LIVE_SUBMIT="${ALLOW_LIVE_SUBMIT:-0}"' in watch_text


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

    # step-30k is complete on disk, but step-20k lacks r5.  The watcher must
    # not skip the registered 20k midpoint or mix steps between arms.
    write_checkpoint(project, r3, 20000, 3)
    write_checkpoint(project, r3, 30000, 3)
    write_checkpoint(project, r5, 30000, 5)
    result = run_bash(WATCH, env)
    assert result.returncode == 0, result.stdout
    assert "no paired checkpoint ready" in result.stdout
    assert "next_ready_step=30000" not in result.stdout

    # Once r5 step-20k appears, the exact same-step pair is selected first.
    write_checkpoint(project, r5, 20000, 5)
    result = run_bash(WATCH, env)
    assert result.returncode == 0, result.stdout
    assert "next_ready_step=20000" in result.stdout

    # Only after verified 20k completion may 30k be selected.
    mark_complete(project, Path(env["EVAL_ROOT"]), 20000)
    result = run_bash(WATCH, env)
    assert result.returncode == 0, result.stdout
    assert "next_ready_step=30000" in result.stdout


def test_rollup_accepts_exact_six_row_schema(tmp_path: Path) -> None:
    env = watcher_env(tmp_path)
    project = Path(env["PROJECT_ROOT"])
    eval_root = Path(env["EVAL_ROOT"])
    mark_complete(project, eval_root, 20000)
    result = run_bash(WATCH, env)
    assert result.returncode == 0, result.stdout
    text = (eval_root / "paired_metrics_all.md").read_text(encoding="utf-8")
    assert "Completed paired checkpoints: 1/2" in text
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
    assert "STEP must be diagnostic 10000 or registered 20000/30000" in result.stdout


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
    # The formal watcher must remain exactly 20k/30k despite diagnostic support.
    assert 'STEPS="20000 30000"' in WATCH.read_text(encoding="utf-8")

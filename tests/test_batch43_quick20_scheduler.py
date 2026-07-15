from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "scripts/004099_submit_batch43_quick20_qz.sh"
WATCH = ROOT / "scripts/004100_watch_batch43_quick20.sh"
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
        "BATCH43_QUICK20_TEST_MODE": "1",
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


def test_shell_syntax_and_safety_contract() -> None:
    for script in (SUBMIT, WATCH):
        result = subprocess.run(
            ["bash", "-n", str(script)], cwd=ROOT, text=True, capture_output=True, check=False
        )
        assert result.returncode == 0, result.stderr
    submit_text = SUBMIT.read_text(encoding="utf-8")
    watch_text = WATCH.read_text(encoding="utf-8")
    assert 'DRY_RUN="${DRY_RUN:-1}"' in submit_text
    assert 'CONFIRM_BATCH43_QUICK20="${CONFIRM_BATCH43_QUICK20:-0}"' in submit_text
    assert 'COLLECT_ONLY="${COLLECT_ONLY:-0}"' in submit_text
    assert "lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122" in submit_text
    assert "67b10bc6-78b0-41a3-aaf4-358eeeb99009" in submit_text
    assert "NVIDIA_H200_SXM_141G" in submit_text
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
    text20 = tmp_path / "record/ver23_batch43_text_quick20_8cell_20260712.jsonl"
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
    record = project / "trainset/qz_jobs/ver23_batch43_quick20_step2000_20260712"
    record.mkdir(parents=True)
    rows = metric_rows(2000)
    with (record / "metrics.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (record / "complete.marker").write_text("done\n", encoding="utf-8")

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

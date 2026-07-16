from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.ver3_1.run_batch46_local_quick20 import (
    READY_SCHEMA,
    VARIANTS,
    ema_lag_warning_path,
    evaluate_step500_gate,
    loss_trend,
    ready_marker,
    red_flag_path,
    variant_completion,
)


def test_ready_marker_selects_immutable_inference_checkpoint(tmp_path: Path) -> None:
    inference = tmp_path / "step-000500.infer.pt"
    inference.write_bytes(b"raw-and-ema-without-optimizer")
    (tmp_path / "last.pt").write_bytes(b"mutable-last")
    marker = {
        "schema": READY_SCHEMA,
        "status": "ready",
        "step": 500,
        "inference_checkpoint": str(inference),
        "inference_checkpoint_size_bytes": inference.stat().st_size,
    }
    marker_path = tmp_path / "step-000500.ready.json"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    selected = ready_marker(SimpleNamespace(checkpoint_root=tmp_path), 500)
    assert selected is not None
    assert selected[0] == marker_path
    assert selected[2] == inference.resolve()
    assert selected[2].name != "last.pt"


def test_ready_marker_rejects_last_pt_alias(tmp_path: Path) -> None:
    last = tmp_path / "last.pt"
    last.write_bytes(b"mutable-last")
    marker = {
        "schema": READY_SCHEMA,
        "status": "ready",
        "step": 500,
        "inference_checkpoint": str(last),
        "inference_checkpoint_size_bytes": last.stat().st_size,
    }
    (tmp_path / "step-000500.ready.json").write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="inference path"):
        ready_marker(SimpleNamespace(checkpoint_root=tmp_path), 500)


def test_loss_trend_is_checkpoint_bounded(tmp_path: Path) -> None:
    rows = [
        {"step": 1, "loss": 1.0},
        {"step": 100, "loss": 0.8},
        {"step": 200, "loss": 0.6},
        {"step": 300, "loss": 0.5},
        {"step": 400, "loss": 0.4},
        {"step": 500, "loss": 0.3},
        {"step": 600, "loss": 99.0},
    ]
    with (tmp_path / "train_log.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    result = loss_trend(tmp_path, 500)
    assert result["points"] == 6
    assert result["last"] == pytest.approx(0.3)
    assert result["direction"] == "down"
    assert result["slope_per_100_steps"] < 0


def write_step500_completions(args: SimpleNamespace, fail_rates: tuple[float, float]) -> None:
    for variant, fail_rate in zip(VARIANTS, fail_rates):
        completion = variant_completion(args, 500, variant)
        completion.parent.mkdir(parents=True, exist_ok=True)
        completion.write_text(
            json.dumps({
                "status": "completed",
                "metrics": {
                    "variant": variant.key,
                    "fail_rate": fail_rate,
                    "sim_ref": 0.1,
                    "sim_src": 0.1,
                    "margin": 0.0,
                    "cer": 1.0,
                },
            }),
            encoding="utf-8",
        )


def test_step500_ema_only_failure_is_lag_warning_and_continues(tmp_path: Path) -> None:
    args = SimpleNamespace(
        task_name="codecVC-test",
        output_root=tmp_path / "outputs",
        record_root=tmp_path / "records",
    )
    write_step500_completions(args, (1.0, 0.75))
    assert evaluate_step500_gate(args) == "ema_lag_warning"
    marker = json.loads(ema_lag_warning_path(args).read_text())
    assert marker["status"] == "ema_lag_warning"
    assert "continue" in marker["action"]
    assert not red_flag_path(args).exists()


def test_step500_both_lanes_failure_is_hard_red_flag(tmp_path: Path) -> None:
    args = SimpleNamespace(
        task_name="codecVC-test",
        output_root=tmp_path / "outputs",
        record_root=tmp_path / "records",
    )
    write_step500_completions(args, (1.0, 1.0))
    assert evaluate_step500_gate(args) == "hard_red_flag"
    marker = json.loads(red_flag_path(args).read_text())
    assert marker["status"] == "red_flag"
    assert "both EMA+CFG1.5 and raw+CFG1.5" in marker["reason"]
    assert "training is not stopped or signalled" in marker["action"]

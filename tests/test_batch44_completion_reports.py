from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/004128_build_batch44_completion_reports.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "test_batch44_completion_reports_module", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def metric_values(*, arm: str, mode: str, step: int) -> dict:
    if arm == "r3":
        sim_ref = 0.44 + step / 1_000_000
        sim_src = 0.40
    else:
        sim_ref = 0.45
        sim_src = 0.41
    if mode == "text":
        sim_ref -= 0.03
        sim_src -= 0.12
    keep = 18 if arm == "r3" else 19
    return {
        "n": 20,
        "keep": keep,
        "fail": (20 - keep) / 20,
        "cer": 0.08 if mode == "no_text" else 0.05,
        "sim_ref": sim_ref,
        "sim_src": sim_src,
        "margin": sim_ref - sim_src,
        "ref_bound": 0.50,
        "ref_content_f1": 0.06,
    }


def write_original(path: Path) -> None:
    rows = []
    for step in (2_000, 10_000):
        for arm in ("r3", "r5"):
            for mode in ("no_text", "text"):
                values = metric_values(arm=arm, mode=mode, step=step)
                rows.append(
                    {
                        "step": step,
                        "arm": arm,
                        "train_job_id": MODULE.ORIGINAL_TRAIN_JOBS[arm],
                        "mode": mode,
                        **values,
                        "text_en_src_quick_n": 12 if mode == "text" else "",
                        "text_en_src_quick_fail": 1 / 12 if mode == "text" else "",
                        "text_en_src_scope": "proxy" if mode == "text" else "",
                        "run_id": f"{arm}-{step}-{mode}",
                        "output_dir": f"/tmp/{arm}-{step}-{mode}",
                    }
                )
    write_tsv(path, rows)


def write_continuation(path: Path, *, local_step: int = 2_000) -> None:
    effective = 12_000
    rows = []
    for mode in ("no_text", "text"):
        values = metric_values(arm="r3", mode=mode, step=effective)
        rows.append(
            {
                "arm": "r3",
                "base_effective_step": 10_000,
                **values,
                "checkpoint": f"/tmp/warmstart/step-{local_step}",
                "checkpoint_manifest_sha256": "a" * 64,
                "continuation_local_step": local_step,
                "effective_step": effective,
                "mode": mode,
                "output_dir": f"/tmp/r3-{effective}-{mode}",
                "run_id": f"r3-{effective}-{mode}",
                "step": effective,
                "text_en_src_quick_fail": 1 / 12 if mode == "text" else "",
                "text_en_src_quick_n": 12 if mode == "text" else "",
                "text_en_src_scope": "proxy" if mode == "text" else "",
                "train_job_id": MODULE.CONTINUATION_TRAIN_JOB,
                "warm_start_contract": "/tmp/warm_start_contract.json",
                "warm_start_contract_sha256": MODULE.CONTINUATION_CONTRACT_SHA256,
            }
        )
    write_tsv(path, rows)


def full_row(arm: str, mode: str) -> dict:
    n = 160
    keep = 140 if arm == "r3" else 145
    sim_ref = 0.445 if arm == "r3" else 0.438
    sim_src = 0.397 if arm == "r3" else 0.406
    if mode == "text":
        sim_ref -= 0.03
        sim_src -= 0.12
    return {
        "step": 10_000,
        "arm": arm,
        "text_repeat": 3 if arm == "r3" else 5,
        "train_job_id": MODULE.ORIGINAL_TRAIN_JOBS[arm],
        "scope": mode,
        "n": n,
        "keep": keep,
        "fail_count": n - keep,
        "fail_rate": (n - keep) / n,
        "cer": 0.12 if mode == "no_text" else 0.06,
        "wavlm_sim_ref": sim_ref,
        "wavlm_sim_src": sim_src,
        "wavlm_margin": sim_ref - sim_src,
        "wavlm_ref_bound": 0.50,
        "speechbrain_sim_ref": 0.49,
        "speechbrain_sim_src": 0.25,
        "speechbrain_margin": 0.24,
        "speechbrain_ref_bound": 0.85,
        "ref_content_lcs_f1": 0.06,
        "text_en_src_n": 80 if mode == "text" else "",
        "text_en_src_fail_count": 10 if mode == "text" else "",
        "text_en_src_fail_rate": 0.125 if mode == "text" else "",
        "text_en_src_cer": 0.05 if mode == "text" else "",
    }


def write_full320(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        full_row(arm, mode)
        for arm in ("r3", "r5")
        for mode in ("no_text", "text")
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")


def write_baseline(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    systems = [
        ("ground_truth", "Ground truth (self-eval)", "complete"),
        ("seed_vc_v2", "Seed-VC V2", "complete"),
        ("cosyvoice2_vc", "CosyVoice 2 VC", "complete"),
        ("openvoice_v2", "OpenVoice V2", "complete"),
        ("vevo_timbre", "Vevo-Timbre", "complete"),
        ("freevc_v1", "FreeVC V1", "complete"),
        ("path_x_3k", "ver2.9.5-probe (ours 3k)", "complete"),
        ("path_x_final", "ver2.9.5-final (ours 30k)", "pending"),
    ]
    main = []
    cross = []
    for index, (system_id, display, status) in enumerate(systems):
        value = None if status == "pending" else 0.50 + index / 100
        main.append(
            {
                "system_id": system_id,
                "system": display,
                "type": "fixture",
                "status": status,
                "en567_wavlm_sim_ref": value,
                "en567_whisper_wer_fraction": None if value is None else 0.04,
                "zh1194_wavlm_sim_ref": value,
                "zh1194_paraformer_cer_fraction": None if value is None else 0.03,
            }
        )
        for split in ("EN567", "ZH1194"):
            cross.append(
                {
                    "system_id": system_id,
                    "system": display,
                    "split": split,
                    "wavlm_large_sv_sim_ref": value,
                    "eres2net_sim_ref": None if value is None else value + (0.20 if system_id == "seed_vc_v2" and split == "EN567" else 0.02),
                    "speechbrain_ecapa_sim_ref": None if value is None else value + 0.01,
                    "status": status,
                }
            )
    path.write_text(
        json.dumps(
            {
                "status": "interim",
                "counts": {"systems": 8, "complete": 7, "pending": 1},
                "main_table": main,
                "cross_validation_table": cross,
            }
        ),
        encoding="utf-8",
    )


def test_builds_interim_closure_without_inventing_future_results(tmp_path: Path) -> None:
    original = tmp_path / "original.tsv"
    continuation = tmp_path / "continuation.tsv"
    full320 = tmp_path / "full320.json"
    baseline = tmp_path / "baseline.json"
    output = tmp_path / "closure"
    write_original(original)
    write_continuation(continuation)
    write_full320(full320)
    write_baseline(baseline)

    assert (
        MODULE.main(
            [
                "--project-root",
                str(tmp_path),
                "--original-quick20",
                str(original),
                "--continuation-quick20",
                str(continuation),
                "--full320-metrics",
                str(full320),
                "--baseline-table",
                str(baseline),
                "--best2-selection",
                str(tmp_path / "missing_best2.json"),
                "--final-selection",
                str(tmp_path / "missing_final.json"),
                "--mos-summary",
                str(tmp_path / "missing_mos.json"),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    expected = {
        "learning_curves_r3.tsv",
        "learning_curves_r3.json",
        "learning_curves_r3.png",
        "paired_metrics_r3_full320.md",
        "batch44_task1_r3_report.md",
        "batch44_task2_r5_report.md",
        "batch44_task3_baselines_report.md",
        "closure_manifest.json",
    }
    assert expected == {path.name for path in output.iterdir()}
    assert (output / "learning_curves_r3.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    paired = (output / "paired_metrics_r3_full320.md").read_text()
    assert "Evidence type: strict full320 only" in paired
    assert "quick20 rows are intentionally excluded" in paired
    assert "Effective steps with accepted full320 evidence: 10000" in paired
    assert "| 10000 | no_text | 160 |" in paired

    curve = json.loads((output / "learning_curves_r3.json").read_text())
    assert curve["status"] == "interim"
    assert curve["warm_start_boundary"]["effective_step"] == 10_000
    assert curve["warm_start_boundary"]["semantics"] == "weights_only_warm_start_not_exact_resume"
    assert curve["observed_quick20_steps"] == [2_000, 10_000, 12_000]
    assert 14_000 in curve["missing_quick20_steps"]
    assert {row["evidence_type"] for row in curve["rows"]} == {"quick20", "full320"}
    continuation_rows = [row for row in curve["rows"] if row["step"] == 12_000]
    assert {row["continuation_local_step"] for row in continuation_rows} == {2_000}
    assert {row["phase"] for row in continuation_rows} == {
        "weights_only_warm_start_continuation"
    }

    task1 = (output / "batch44_task1_r3_report.md").read_text()
    assert "Report status: **interim**" in task1
    assert "Best2: **pending**" in task1
    assert "30k FINAL_SELECTION: **pending**" in task1
    assert "ERes2Net: pending" in task1
    assert "weights-only warm-start" in task1
    assert "| 10000 | no_text | 160 |" in task1

    task2 = (output / "batch44_task2_r5_report.md").read_text()
    assert "Report status: **stopped_at_10k**" in task2
    assert "| 12000 | N/A | N/A | N/A | N/A | N/A | arm terminated |" in task2
    assert "r5 30k 三判据：N/A" in task2

    task3 = (output / "batch44_task3_baselines_report.md").read_text()
    assert "Baseline table completion: **7/8**" in task3
    assert "test-zh-hard: **N/A**" in task3
    assert "前半段 vs 后半段 speaker calibration：pending" in task3
    assert "SMOS/CMOS 40 cases × 5 raters: **pending**" in task3
    assert "scorer disagreement" in task3
    assert "ver2.9.5-final (30k): **pending**" in task3


def test_rejects_invalid_warmstart_effective_local_mapping(tmp_path: Path) -> None:
    continuation = tmp_path / "continuation.tsv"
    write_continuation(continuation, local_step=4_000)
    with pytest.raises(ValueError, match="invalid effective/local mapping"):
        MODULE.load_continuation_quick20(continuation)


def test_missing_best2_final_and_mos_are_pending_not_errors(tmp_path: Path) -> None:
    original = tmp_path / "original.tsv"
    continuation = tmp_path / "continuation.tsv"
    baseline = tmp_path / "baseline.json"
    output = tmp_path / "closure"
    write_original(original)
    write_continuation(continuation)
    write_baseline(baseline)

    args = MODULE.parse_args(
        [
            "--project-root",
            str(tmp_path),
            "--original-quick20",
            str(original),
            "--continuation-quick20",
            str(continuation),
            "--baseline-table",
            str(baseline),
            "--best2-selection",
            str(tmp_path / "absent-best2.json"),
            "--final-selection",
            str(tmp_path / "absent-final.json"),
            "--mos-summary",
            str(tmp_path / "absent-mos.json"),
            "--output-dir",
            str(output),
        ]
    )
    manifest = MODULE.build(args)
    assert manifest["status"] == "interim"
    assert manifest["inputs"]["best2"]["status"] == "missing"
    assert manifest["inputs"]["final_selection"]["status"] == "missing"
    assert manifest["inputs"]["mos"]["status"] == "missing"
    assert "pending" in (output / "batch44_task1_r3_report.md").read_text()

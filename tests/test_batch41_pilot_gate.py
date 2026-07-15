from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/004088_build_batch41_pilot_gate.py"
SPEC = importlib.util.spec_from_file_location("batch41_pilot_gate", SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot import {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


FIELDS = (
    "run",
    "case_id",
    "mode",
    "cell",
    "cer_tgt",
    "content_keep",
    "sim_gen_ref",
    "sim_gen_source",
    "ecapa_sim_gen_ref",
)


def write_fixture(
    root: Path,
    *,
    run_id: str = "pilot",
    no_text_cer: float = 0.10,
    text_cer: float = 0.05,
    text_en_src_fail: int = 8,
    wavlm_ref: float = 0.43,
) -> tuple[Path, Path]:
    rows = []
    no_text_keep = 150
    text_keep = 150 - text_en_src_fail
    for index in range(160):
        rows.append(
            {
                "run": run_id,
                "case_id": f"no-text-{index:03d}",
                "mode": "no_text",
                "cell": "en_src_en_ref_same_gender" if index < 80 else "zh_src_zh_ref_same_gender",
                "cer_tgt": no_text_cer,
                "content_keep": index < no_text_keep,
                "sim_gen_ref": wavlm_ref,
                "sim_gen_source": 0.36,
                "ecapa_sim_gen_ref": 0.49,
            }
        )
    for index in range(160):
        is_en_src = index < 80
        keep = index >= text_en_src_fail if is_en_src else index < 80 + (text_keep - (80 - text_en_src_fail))
        rows.append(
            {
                "run": run_id,
                "case_id": f"text-{index:03d}",
                "mode": "text",
                "cell": "en_src_en_ref_same_gender" if is_en_src else "zh_src_zh_ref_same_gender",
                "cer_tgt": text_cer,
                "content_keep": keep,
                "sim_gen_ref": 0.41,
                "sim_gen_source": 0.31,
                "ecapa_sim_gen_ref": 0.48,
            }
        )
    cases = root / "cases.csv"
    with cases.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = root / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "by_mode": {
                    "no_text": {"n": 160, "cer": no_text_cer, "keep": no_text_keep},
                    "text": {"n": 160, "cer": text_cer, "keep": text_keep},
                }
            }
        ),
        encoding="utf-8",
    )
    return summary, cases


class Batch41PilotGateTest(unittest.TestCase):
    def test_all_registered_thresholds_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary, cases = write_fixture(Path(tmp))
            gate = MODULE.build_gate(
                summary_path=summary,
                cases_path=cases,
                pilot_job_id="job-fixture",
                run_id="pilot",
            )
        self.assertEqual(gate["decision"], "pass")
        self.assertAlmostEqual(gate["text_en_src_fail"], 0.10)
        self.assertTrue(all(gate["checks"].values()))

    def test_failed_en_src_threshold_blocks_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary, cases = write_fixture(Path(tmp), text_en_src_fail=16)
            gate = MODULE.build_gate(
                summary_path=summary,
                cases_path=cases,
                pilot_job_id="job-fixture",
                run_id="pilot",
            )
        self.assertEqual(gate["decision"], "fail")
        self.assertAlmostEqual(gate["text_en_src_fail"], 0.20)
        self.assertFalse(gate["checks"]["text_en_src_fail_lt_0p15"])

    def test_summary_and_csv_keep_counts_must_agree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary, cases = write_fixture(Path(tmp))
            payload = json.loads(summary.read_text(encoding="utf-8"))
            payload["by_mode"]["text"]["keep"] -= 1
            summary.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "text content_keep"):
                MODULE.build_gate(
                    summary_path=summary,
                    cases_path=cases,
                    pilot_job_id="job-fixture",
                    run_id="pilot",
                )

    def test_summary_run_id_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary, cases = write_fixture(Path(tmp))
            payload = json.loads(summary.read_text(encoding="utf-8"))
            payload["run_id"] = "different-run"
            summary.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                MODULE.build_gate(
                    summary_path=summary,
                    cases_path=cases,
                    pilot_job_id="job-fixture",
                    run_id="pilot",
                )

    def test_summary_and_csv_cer_must_agree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary, cases = write_fixture(Path(tmp))
            payload = json.loads(summary.read_text(encoding="utf-8"))
            payload["by_mode"]["no_text"]["cer"] = 0.01
            summary.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no_text CER disagrees"):
                MODULE.build_gate(
                    summary_path=summary,
                    cases_path=cases,
                    pilot_job_id="job-fixture",
                    run_id="pilot",
                )

    def test_all_four_gate_boundaries(self) -> None:
        cases_to_check = (
            ({"no_text_cer": 0.12}, "no_text_cer_lt_0p12", False),
            ({"text_cer": 0.06}, "text_cer_lt_0p06", False),
            ({"text_en_src_fail": 12}, "text_en_src_fail_lt_0p15", False),
            ({"wavlm_ref": 0.42}, "wavlm_sim_ref_ge_0p42", True),
            ({"wavlm_ref": 0.419999}, "wavlm_sim_ref_ge_0p42", False),
        )
        for fixture_kwargs, check_name, expected in cases_to_check:
            with self.subTest(check_name=check_name, fixture_kwargs=fixture_kwargs):
                with tempfile.TemporaryDirectory() as tmp:
                    summary, cases = write_fixture(Path(tmp), **fixture_kwargs)
                    gate = MODULE.build_gate(
                        summary_path=summary,
                        cases_path=cases,
                        pilot_job_id="job-fixture",
                        run_id="pilot",
                    )
                self.assertEqual(gate["checks"][check_name], expected)

    def test_invalid_content_keep_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary, cases = write_fixture(Path(tmp))
            with cases.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["content_keep"] = "unknown"
            with cases.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "invalid boolean"):
                MODULE.build_gate(
                    summary_path=summary,
                    cases_path=cases,
                    pilot_job_id="job-fixture",
                    run_id="pilot",
                )


if __name__ == "__main__":
    unittest.main()

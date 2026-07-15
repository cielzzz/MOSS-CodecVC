from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/004092_build_batch42_baseline_tables.py"
SPEC = importlib.util.spec_from_file_location("batch42_baseline_tables", SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot import {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def metric(n: int, mean: float) -> dict:
    return {"n": n, "mean": mean, "std": 0.01, "min": mean, "max": mean}


def write_summary(
    output_root: Path,
    *,
    system_id: str,
    language: str,
    wavlm: float,
    eres2net: float,
    speechbrain: float,
    error: float,
    count_override: int | None = None,
) -> Path:
    spec = MODULE.DATASETS[language]
    count = spec.expected_cases if count_override is None else count_override
    group = {
        "n_cases": count,
        "speaker_similarity": {},
        "content_asr": {},
    }
    for backend, value in (
        ("wavlm_large_sv", wavlm),
        ("eres2net", eres2net),
        ("speechbrain_ecapa", speechbrain),
    ):
        group["speaker_similarity"][backend] = {
            "status_counts": {"ok": count},
            "sim_ref": metric(count, value),
            "sim_src": metric(count, value - 0.1),
        }
    group["content_asr"][spec.asr_backend] = {
        "status_counts": {"ok": count},
        "primary_error": metric(count, error),
        spec.error_metric: metric(count, error),
    }
    identity = (
        f"system_test_set_language:{system_id}:{spec.test_set_id}:{language}"
    )
    summary = {
        "schema_version": MODULE.UNIFIED_SUMMARY_SCHEMA,
        "record_type": "vc_eval_summary",
        "groups": {"all": group, identity: group},
        "run_id": f"{system_id}_{language}_merged",
    }
    merged = output_root / language / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    summary_path = merged / f"{system_id}.{language}.merged.summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    audit = {
        "schema_version": MODULE.STRICT_AUDIT_SCHEMA,
        "system_id": system_id,
        "test_set_id": spec.test_set_id,
        "language": language,
        "rows": count,
        "unique_case_ids": count,
        "input_index_coverage": [0, count - 1],
        "speaker_status_counts": {
            backend: {"ok": count} for backend, _display in MODULE.SPEAKER_SCORERS
        },
        "asr_status_counts": {spec.asr_backend: {"ok": count}},
        "all_ok": True,
    }
    audit_path = MODULE._audit_path_for_summary(summary_path)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    return summary_path


class Batch42BaselineTablesTest(unittest.TestCase):
    def test_default_type_labels_match_the_asset_audit(self) -> None:
        labels = {item.system_id: item.system_type for item in MODULE.DEFAULT_SYSTEMS}
        names = {item.system_id: item.display_name for item in MODULE.DEFAULT_SYSTEMS}
        self.assertEqual(
            names["ground_truth"], "Ground truth (self-eval)"
        )
        self.assertEqual(names["path_x_3k"], "ver2.9.5-probe (ours 3k)")
        self.assertEqual(names["path_x_final"], "ver2.9.5-final (ours 30k)")
        self.assertEqual(labels["ground_truth"], "metric calibration, not VC")
        self.assertEqual(labels["seed_vc_v2"], "conditional flow matching")
        self.assertEqual(names["cosyvoice2_vc"], "CosyVoice 2 VC")
        self.assertEqual(
            labels["vevo_timbre"], "content-style tokenizer + flow matching"
        )
        self.assertEqual(labels["freevc_v1"], "VITS + WavLM bottleneck")
        self.assertNotIn("AutoVC", labels.values())

    def test_explicit_complete_system_writes_all_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "scored"
            en = write_summary(
                run_root,
                system_id="fixture",
                language="en",
                wavlm=0.51,
                eres2net=0.61,
                speechbrain=0.41,
                error=0.034,
            )
            zh = write_summary(
                run_root,
                system_id="fixture",
                language="zh",
                wavlm=0.52,
                eres2net=0.62,
                speechbrain=0.42,
                error=0.056,
            )
            prefix = root / "tables/interim"
            status = MODULE.main(
                [
                    "--no-discovery",
                    "--en-summary",
                    f"fixture={en}",
                    "--zh-summary",
                    f"fixture={zh}",
                    "--expected-system",
                    "fixture",
                    "--system-meta",
                    "fixture|Fixture VC|Test",
                    "--output-prefix",
                    str(prefix),
                ]
            )
            payload = json.loads(prefix.with_suffix(".json").read_text())
            markdown = prefix.with_suffix(".md").read_text()
            main_tsv = prefix.with_suffix(".tsv").read_text()
            cross_tsv = prefix.with_name(
                prefix.name + ".cross_validation.tsv"
            ).read_text()

        self.assertEqual(status, 0)
        self.assertEqual(payload["protocol"]["label"], MODULE.PROTOCOL_LABEL)
        self.assertEqual(payload["status"], "complete")
        row = payload["main_table"][0]
        self.assertEqual(row["system"], "Fixture VC")
        self.assertAlmostEqual(row["zh1194_wavlm_sim_ref"], 0.52)
        self.assertAlmostEqual(row["zh1194_paraformer_cer_fraction"], 0.056)
        self.assertAlmostEqual(row["zh1194_paraformer_cer_percent"], 5.6)
        self.assertAlmostEqual(row["en567_whisper_wer_fraction"], 0.034)
        self.assertAlmostEqual(row["en567_whisper_wer_percent"], 3.4)
        self.assertIsNone(row["zh_hard_wavlm_sim_ref"])
        self.assertEqual(row["zh_hard_status"], "N/A")
        self.assertIn(MODULE.PROTOCOL_LABEL, markdown)
        self.assertIn("same-file scorer calibration", markdown)
        self.assertIn("paired waveform", markdown)
        self.assertIn("| Fixture VC | Test | 0.5100 | 3.40 | 0.5200 | 5.60", markdown)
        self.assertIn("no pure-VC ZH-hard manifest", markdown)
        self.assertIn("fixture\tFixture VC\tTest", main_tsv)
        self.assertIn("fixture\tFixture VC\tZH1194", cross_tsv)
        self.assertIn("fixture\tFixture VC\tEN567", cross_tsv)

    def test_missing_language_is_partial_and_never_filled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            en = write_summary(
                root / "scored",
                system_id="fixture",
                language="en",
                wavlm=0.51,
                eres2net=0.61,
                speechbrain=0.41,
                error=0.034,
            )
            item = MODULE.validate_summary(en)
            payload = MODULE.build_payload(
                selected={"fixture": {"en": item}},
                system_specs=[MODULE.SystemSpec("fixture", "Fixture", "Test")],
                search_roots=[],
                explicit_summaries={},
                rejected_candidates=[],
            )
            markdown = MODULE.render_markdown(payload)
            row = payload["main_table"][0]

        self.assertEqual(payload["status"], "interim")
        self.assertEqual(row["status"], "partial")
        self.assertIsNone(row["zh1194_wavlm_sim_ref"])
        self.assertIn("| Fixture | Test | 0.5100 | 3.40 | pending | pending", markdown)
        self.assertNotIn("0.750", markdown)

    def test_discovery_prefers_complete_pair_over_newer_partial_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            complete = root / "complete"
            old_en = write_summary(
                complete,
                system_id="fixture",
                language="en",
                wavlm=0.51,
                eres2net=0.61,
                speechbrain=0.41,
                error=0.034,
            )
            write_summary(
                complete,
                system_id="fixture",
                language="zh",
                wavlm=0.52,
                eres2net=0.62,
                speechbrain=0.42,
                error=0.056,
            )
            partial = root / "newer_partial"
            newer_en = write_summary(
                partial,
                system_id="fixture",
                language="en",
                wavlm=0.91,
                eres2net=0.91,
                speechbrain=0.91,
                error=0.01,
            )
            newer_ns = old_en.stat().st_mtime_ns + 10_000_000_000
            os.utime(newer_en, ns=(newer_ns, newer_ns))
            selected, rejected = MODULE.discover_summaries([root])

        self.assertFalse(rejected)
        self.assertEqual(set(selected["fixture"]), {"en", "zh"})
        self.assertEqual(selected["fixture"]["en"].output_root.name, "complete")
        self.assertAlmostEqual(
            selected["fixture"]["en"].group_all["speaker_similarity"]
            ["wavlm_large_sv"]["sim_ref"]["mean"],
            0.51,
        )

    def test_wrong_case_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_summary(
                Path(tmp) / "bad",
                system_id="fixture",
                language="en",
                wavlm=0.51,
                eres2net=0.61,
                speechbrain=0.41,
                error=0.034,
                count_override=566,
            )
            with self.assertRaisesRegex(ValueError, "n_cases=566, expected 567"):
                MODULE.validate_summary(path)

    def test_explicit_system_identity_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_summary(
                Path(tmp) / "scored",
                system_id="actual",
                language="zh",
                wavlm=0.52,
                eres2net=0.62,
                speechbrain=0.42,
                error=0.056,
            )
            with self.assertRaisesRegex(ValueError, "system='actual', expected 'other'"):
                MODULE.validate_summary(path, expected_system="other")

    def test_audit_must_match_strict_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_summary(
                Path(tmp) / "scored",
                system_id="fixture",
                language="zh",
                wavlm=0.52,
                eres2net=0.62,
                speechbrain=0.42,
                error=0.056,
            )
            audit_path = MODULE._audit_path_for_summary(path)
            audit = json.loads(audit_path.read_text())
            audit["test_set_id"] = "official-unqualified"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "test_set_id=.*expected"):
                MODULE.validate_summary(path)

    def test_path_x_final_is_never_promoted_by_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scored = root / "scored"
            write_summary(
                scored,
                system_id="path_x_final",
                language="en",
                wavlm=0.51,
                eres2net=0.61,
                speechbrain=0.41,
                error=0.034,
            )
            write_summary(
                scored,
                system_id="path_x_final",
                language="zh",
                wavlm=0.52,
                eres2net=0.62,
                speechbrain=0.42,
                error=0.056,
            )
            args = MODULE.build_parser().parse_args(
                [
                    "--search-root",
                    str(root),
                    "--output-prefix",
                    str(root / "table"),
                ]
            )
            payload, _outputs = MODULE.run(args)

        final = next(
            row for row in payload["systems"] if row["system_id"] == "path_x_final"
        )
        self.assertEqual(final["status"], "pending")
        self.assertTrue(
            any(
                "cannot be promoted by recursive discovery" in item["reason"]
                for item in payload["discovery"]["rejected_candidates"]
            )
        )

    def test_explicit_path_x_final_requires_dedicated_gate_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scored = root / "scored"
            en = write_summary(
                scored,
                system_id="path_x_final",
                language="en",
                wavlm=0.51,
                eres2net=0.61,
                speechbrain=0.41,
                error=0.034,
            )
            zh = write_summary(
                scored,
                system_id="path_x_final",
                language="zh",
                wavlm=0.52,
                eres2net=0.62,
                speechbrain=0.42,
                error=0.056,
            )
            base = [
                "--no-discovery",
                "--en-summary",
                f"path_x_final={en}",
                "--zh-summary",
                f"path_x_final={zh}",
                "--output-prefix",
                str(root / "table"),
            ]
            args = MODULE.build_parser().parse_args(base)
            with self.assertRaisesRegex(ValueError, "--allow-path-x-final"):
                MODULE.run(args)
            args = MODULE.build_parser().parse_args(
                ["--allow-path-x-final", *base]
            )
            payload, _outputs = MODULE.run(args)

        final = next(
            row for row in payload["systems"] if row["system_id"] == "path_x_final"
        )
        self.assertEqual(final["status"], "complete")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/004082_run_unified_vc_eval.py"
SPEC = importlib.util.spec_from_file_location("batch42_unified_eval", SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot import {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class UnifiedVCEvalSchemaTest(unittest.TestCase):
    def test_whisper_decodes_with_soundfile_before_pipeline(self) -> None:
        import numpy as np
        import soundfile as sf

        class FakePipeline:
            feature_extractor = type("FeatureExtractor", (), {"sampling_rate": 16000})()

            def __init__(self) -> None:
                self.payload = None
                self.generate_kwargs = None

            def __call__(self, payload, *, generate_kwargs):
                self.payload = payload
                self.generate_kwargs = generate_kwargs
                return {"text": " decoded "}

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "eight_k.wav"
            sf.write(wav, np.zeros(800, dtype=np.float32), 8000)
            backend = MODULE.WhisperLargeV3ASR.__new__(MODULE.WhisperLargeV3ASR)
            backend.model_id = "fixture"
            backend.pipeline = FakePipeline()
            self.assertEqual(backend.transcribe(str(wav), "en"), "decoded")
            self.assertIsInstance(backend.pipeline.payload, dict)
            self.assertEqual(backend.pipeline.payload["sampling_rate"], 16000)
            self.assertEqual(backend.pipeline.payload["array"].ndim, 1)
            self.assertGreaterEqual(len(backend.pipeline.payload["array"]), 1590)
            self.assertEqual(
                backend.pipeline.generate_kwargs,
                {"task": "transcribe", "language": "english"},
            )

    def test_legacy_aliases_are_preserved_without_recompute(self) -> None:
        case = MODULE.normalize_case(
            {
                "case_id": "legacy-1",
                "run": "old-run",
                "mode": "no_text",
                "source_lang": "zh-CN",
                "target_audio": "/tmp/generated.wav",
                "timbre_ref_audio": "/tmp/reference.wav",
                "source_audio": "/tmp/source.wav",
                "content_ref_text": "你好世界",
                "sim_gen_ref": 0.42,
                "sim_gen_source": 0.31,
                "ecapa_sim_gen_ref": 0.51,
                "ecapa_sim_gen_source": 0.22,
                "asr_tgt_text": "你好世界",
                "cer_tgt": 0.0,
                "wer_tgt": 0.0,
            },
            input_index=0,
            input_path=Path("legacy.jsonl"),
            run_id="batch42",
            system_override="",
            test_set_override="internal320",
            legacy_asr_backend="qwen_asr",
        )
        self.assertEqual(case["schema_version"], MODULE.SCHEMA_VERSION)
        self.assertEqual(case["language"], "zh")
        self.assertEqual(case["system_id"], "old-run")
        self.assertEqual(
            case["speaker_similarity"]["wavlm_large_sv"]["status"], "precomputed"
        )
        self.assertEqual(
            case["speaker_similarity"]["speechbrain_ecapa"]["sim_ref"], 0.51
        )
        self.assertEqual(case["content_asr"]["qwen_asr"]["primary_error"], 0.0)

    def test_schema_only_language_routing_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "cases.jsonl"
            rows = [
                {
                    "case_id": "zh-1",
                    "language": "zh",
                    "generated_audio": "/not/loaded/zh.wav",
                    "reference_audio": "/not/loaded/zh-ref.wav",
                    "reference_text": "统一协议",
                },
                {
                    "case_id": "en-1",
                    "language": "en",
                    "generated_audio": "/not/loaded/en.wav",
                    "reference_audio": "/not/loaded/en-ref.wav",
                    "reference_text": "unified protocol",
                },
            ]
            input_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            status = MODULE.main(
                [
                    "evaluate",
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(root / "out"),
                    "--run-id",
                    "schema_smoke",
                    "--system-id",
                    "dummy-system",
                    "--test-set-id",
                    "dummy-set",
                    "--speaker-scorer",
                    "all",
                    "--asr-backend",
                    "all",
                    "--schema-only",
                ]
            )
            self.assertEqual(status, 0)
            output = root / "out/schema_smoke.unified_eval.jsonl"
            cases = {row["case_id"]: row for row in read_jsonl(output)}
            self.assertEqual(len(cases), 2)
            for backend in MODULE.SPEAKER_BACKENDS:
                self.assertEqual(cases["zh-1"]["speaker_similarity"][backend]["status"], "pending")
            self.assertEqual(cases["zh-1"]["content_asr"]["paraformer_zh"]["status"], "pending")
            self.assertEqual(
                cases["zh-1"]["content_asr"]["whisper_large_v3"]["status"],
                "skipped_language",
            )
            self.assertEqual(
                cases["en-1"]["content_asr"]["paraformer_zh"]["status"],
                "skipped_language",
            )
            self.assertEqual(cases["en-1"]["content_asr"]["whisper_large_v3"]["status"], "pending")
            self.assertEqual(cases["en-1"]["content_asr"]["qwen_asr"]["status"], "pending")
            self.assertTrue((root / "out/schema_smoke.unified_eval.csv").exists())
            self.assertTrue((root / "out/schema_smoke.summary.json").exists())
            self.assertTrue((root / "out/schema_smoke.summary.md").exists())

    def test_partial_merge_prefers_concrete_result_over_pending(self) -> None:
        base = {
            "schema_version": MODULE.SCHEMA_VERSION,
            "record_type": "vc_eval_case",
            "run_id": "partial",
            "system_id": "system",
            "test_set_id": "set",
            "case_id": "case",
            "language": "en",
            "audio": {"generated": "g.wav", "reference": "r.wav", "source": "s.wav"},
            "reference_text": "hello world",
            "metadata": {},
            "speaker_similarity": {"wavlm_large_sv": {"status": "pending", "backend": "wavlm_large_sv"}},
            "content_asr": {},
            "provenance": {},
        }
        concrete = json.loads(json.dumps(base))
        concrete["speaker_similarity"]["wavlm_large_sv"] = {
            "status": "ok",
            "backend": "wavlm_large_sv",
            "model_id": "fixture",
            "sim_ref": 0.75,
            "sim_src": 0.25,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / "left.jsonl"
            right = root / "right.jsonl"
            left.write_text(json.dumps(base) + "\n", encoding="utf-8")
            right.write_text(json.dumps(concrete) + "\n", encoding="utf-8")
            merged = MODULE.merge_cases([left, right], run_id="merged")
        self.assertEqual(len(merged), 1)
        result = merged[0]["speaker_similarity"]["wavlm_large_sv"]
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["sim_ref"], 0.75)

    def test_canonical_schema_can_be_reused_as_backend_input(self) -> None:
        original = {
            "schema_version": MODULE.SCHEMA_VERSION,
            "record_type": "vc_eval_case",
            "run_id": "schema",
            "system_id": "system",
            "test_set_id": "set",
            "case_id": "case",
            "language": "zh",
            "audio": {"generated": "g.wav", "reference": "r.wav", "source": "s.wav"},
            "reference_text": "测试",
            "metadata": {},
            "speaker_similarity": {
                "wavlm_large_sv": {"status": "pending", "backend": "wavlm_large_sv"}
            },
            "content_asr": {},
            "provenance": {},
        }
        reused = MODULE.normalize_case(
            original,
            input_index=3,
            input_path=Path("canonical.jsonl"),
            run_id="actual-run",
            system_override="",
            test_set_override="",
            legacy_asr_backend="qwen_asr",
        )
        self.assertEqual(reused["run_id"], "actual-run")
        self.assertEqual(reused["audio"], original["audio"])
        self.assertEqual(reused["speaker_similarity"]["wavlm_large_sv"]["status"], "pending")
        self.assertEqual(reused["provenance"]["input_index"], 3)

    def test_error_rates_match_language_units(self) -> None:
        self.assertEqual(MODULE.cer("你号", "你好"), 0.5)
        self.assertEqual(MODULE.wer("hello brave world", "hello world"), 0.5)

    def test_seedtts_official_normalization_matches_run_wer_profile(self) -> None:
        # ASCII underscore is removed by string.punctuation, while the ASCII
        # apostrophe is the one punctuation mark Seed-TTS explicitly preserves.
        self.assertEqual(
            MODULE.seedtts_official_error_rate("你_好", "你好", "zh"), 0.0
        )
        self.assertEqual(
            MODULE.seedtts_official_error_rate("你'好", "你好", "zh"), 0.5
        )
        # Mandarin scoring is character-wise and case-sensitive for embedded
        # Latin text; English scoring is word-wise and lower-cased.
        self.assertEqual(
            MODULE.seedtts_official_error_rate("A你", "a你", "zh"), 0.5
        )
        self.assertEqual(
            MODULE.seedtts_official_error_rate("HELLO, world!", "hello world", "en"),
            0.0,
        )
        payload = MODULE.error_rate_payload(
            "hello brave world", "hello world", "en", "seedtts_official"
        )
        self.assertEqual(payload["metric_profile"], "seedtts_eval_run_wer.py")
        self.assertEqual(payload["wer"], 0.5)
        self.assertIsNone(payload["cer"])

    def test_official_vc_profile_rejects_ambiguous_target_audio(self) -> None:
        row = {
            "case_id": "official-1",
            "system_id": "system",
            "test_set_id": "seedtts-vc-zh",
            "language": "zh",
            "target_audio": "infer-or-ground-truth.wav",
            "reference_audio": "prompt.wav",
            "source_audio": "source.wav",
            "reference_text": "测试",
        }
        with self.assertRaisesRegex(ValueError, "generated_audio"):
            MODULE.normalize_case(
                row,
                input_index=0,
                input_path=Path("official.jsonl"),
                run_id="official",
                system_override="",
                test_set_override="",
                legacy_asr_backend="qwen_asr",
                input_profile="official_seedtts_vc",
            )
        unsafe_canonical = MODULE.normalize_case(
            row,
            input_index=0,
            input_path=Path("legacy.jsonl"),
            run_id="legacy",
            system_override="",
            test_set_override="",
            legacy_asr_backend="qwen_asr",
        )
        with self.assertRaisesRegex(ValueError, "legacy target_audio"):
            MODULE.normalize_case(
                unsafe_canonical,
                input_index=0,
                input_path=Path("canonical.jsonl"),
                run_id="official",
                system_override="",
                test_set_override="",
                legacy_asr_backend="qwen_asr",
                input_profile="official_seedtts_vc",
            )
        row["generated_audio"] = "system-output.wav"
        case = MODULE.normalize_case(
            row,
            input_index=0,
            input_path=Path("official.jsonl"),
            run_id="official",
            system_override="",
            test_set_override="",
            legacy_asr_backend="qwen_asr",
            input_profile="official_seedtts_vc",
        )
        self.assertEqual(case["audio"]["generated"], "system-output.wav")
        self.assertEqual(
            case["provenance"]["audio_field_mapping"]["generated"],
            "generated_audio",
        )

    def test_default_shard_outputs_do_not_overwrite_each_other(self) -> None:
        args = MODULE.build_parser().parse_args(
            [
                "evaluate",
                "--input",
                "cases.jsonl",
                "--output-dir",
                "out",
                "--run-id",
                "sharded",
                "--num-shards",
                "4",
                "--shard-index",
                "2",
            ]
        )
        jsonl_path, _csv_path, _summary_json, _summary_md = MODULE.resolve_output_paths(args)
        self.assertEqual(
            jsonl_path.name,
            "sharded.shard-00002-of-00004.unified_eval.jsonl",
        )

    def test_summary_exposes_missing_coverage_and_metric_profile(self) -> None:
        case = MODULE.normalize_case(
            {
                "case_id": "coverage-1",
                "language": "en",
                "generated_audio": "g.wav",
                "reference_audio": "r.wav",
                "source_audio": "s.wav",
                "reference_text": "hello",
                "asr_tgt_text": "hello",
                "wer_tgt": 0.0,
            },
            input_index=0,
            input_path=Path("fixture.jsonl"),
            run_id="coverage",
            system_override="system",
            test_set_override="set",
            legacy_asr_backend="qwen_asr",
        )
        group = MODULE.summarize_cases([case])["groups"]["all"]
        self.assertEqual(
            group["speaker_similarity"]["eres2net"]["status_counts"],
            {"not_present": 1},
        )
        self.assertEqual(
            group["content_asr"]["qwen_asr"]["metric_profile_counts"],
            {"legacy_precomputed_unspecified": 1},
        )

    def test_jsonl_writer_rejects_non_finite_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                MODULE.write_jsonl(
                    Path(tmp) / "invalid.jsonl",
                    [{"sim_ref": float("nan")}],
                )

    def test_merge_rejects_unknown_or_mislabeled_backend(self) -> None:
        base = {
            "schema_version": MODULE.SCHEMA_VERSION,
            "record_type": "vc_eval_case",
            "run_id": "partial",
            "system_id": "system",
            "test_set_id": "set",
            "case_id": "case",
            "language": "en",
            "audio": {"generated": "g.wav", "reference": "r.wav", "source": "s.wav"},
            "reference_text": "hello",
            "metadata": {},
            "speaker_similarity": {
                "wavlm_large_sv": {
                    "status": "ok",
                    "backend": "speechbrain_ecapa",
                    "sim_ref": 0.5,
                }
            },
            "content_asr": {},
            "provenance": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text(json.dumps(base) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "backend must equal"):
                MODULE.merge_cases([path], run_id="merged")

    def test_emitted_record_validates_against_json_schema(self) -> None:
        record = MODULE.normalize_case(
            {
                "case_id": "schema-1",
                "language": "en",
                "generated_audio": "generated.wav",
                "reference_audio": "reference.wav",
                "source_audio": "source.wav",
                "reference_text": "schema validation",
                "sim_gen_ref": 0.6,
                "asr_tgt_text": "schema validation",
                "wer_tgt": 0.0,
            },
            input_index=0,
            input_path=Path("fixture.jsonl"),
            run_id="schema-test",
            system_override="system",
            test_set_override="test-set",
            legacy_asr_backend="qwen_asr",
        )
        schema = json.loads(
            (ROOT / "docs/schemas/moss_codecvc_unified_vc_eval_v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        jsonschema.Draft202012Validator(schema).validate(record)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/004084_run_batch42_openvoice_freevc.py"
SPEC = importlib.util.spec_from_file_location("batch42_baseline_infer", SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot import {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_test_wav(path: Path, *, frames: int = 1600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * frames)


class FakeBackend:
    def __init__(self, fail_case_ids: set[str] | None = None):
        self.fail_case_ids = fail_case_ids or set()
        self.calls: list[str] = []

    def convert(self, case, output_path: Path) -> dict:
        self.calls.append(case.case_id)
        if case.case_id in self.fail_case_ids:
            raise RuntimeError(f"fixture failure for {case.case_id}")
        write_test_wav(output_path)
        return {"backend": "fixture"}


class Batch42BaselineInferTest(unittest.TestCase):
    def test_seedtts_lst_uses_field5_as_source_and_field3_as_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "wavs/source.wav"
            reference = root / "prompt-wavs/reference.wav"
            write_test_wav(source)
            write_test_wav(reference)
            manifest = root / "vc.lst"
            manifest.write_text(
                "case-1|prompt text|prompt-wavs/reference.wav|target text|wavs/source.wav\n",
                encoding="utf-8",
            )
            cases, issues = MODULE.read_input(manifest)
        self.assertEqual(issues, [])
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].source_audio, source.resolve())
        self.assertEqual(cases[0].reference_audio, reference.resolve())
        self.assertEqual(
            cases[0].field_mapping["source_audio"], "field_5/source_audio"
        )
        self.assertEqual(
            cases[0].field_mapping["reference_audio"], "field_3/prompt_audio"
        )

    def test_four_column_hardcase_is_recorded_as_input_issue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hardcase.lst"
            path.write_text(
                "hard-1|prompt|prompt-wavs/ref.wav|target without source wav\n",
                encoding="utf-8",
            )
            cases, issues = MODULE.read_input(path)
        self.assertEqual(cases, [])
        self.assertEqual(len(issues), 1)
        self.assertIn("got 4", issues[0].message)
        self.assertIn("Column 5", issues[0].message)

    def test_jsonl_supports_direct_and_unified_audio_fields_without_target_guess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "cases.jsonl"
            rows = [
                {
                    "case_id": "direct",
                    "source_audio": "source.wav",
                    "timbre_ref_audio": "ref.wav",
                    "text": "你好",
                },
                {
                    "case_id": "nested",
                    "audio": {"source": "source2.wav", "reference": "ref2.wav"},
                    "reference_text": "hello",
                },
                {
                    "case_id": "ambiguous",
                    "target_audio": "must-not-be-used-as-source.wav",
                    "reference_audio": "ref3.wav",
                },
            ]
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            cases, issues = MODULE.read_input(path)
        self.assertEqual([case.case_id for case in cases], ["direct", "nested"])
        self.assertEqual(cases[0].language, "zh")
        self.assertEqual(cases[1].language, "en")
        self.assertEqual(cases[1].field_mapping["source_audio"], "audio.source")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].case_id, "ambiguous")
        self.assertIn("source_audio", issues[0].message)

    def test_deterministic_filename_and_modulo_sharding(self) -> None:
        case = MODULE.VCCase(
            input_index=5,
            input_line=6,
            case_id="unsafe / case 中文",
            case_uid="0123456789abcdef0123",
            source_audio=Path("source.wav"),
            reference_audio=Path("ref.wav"),
        )
        self.assertEqual(
            MODULE.deterministic_wav_name(case),
            "unsafe-case__0123456789ab.wav",
        )
        self.assertTrue(MODULE.shard_selected(5, 4, 1))
        self.assertFalse(MODULE.shard_selected(5, 4, 2))
        parsed = MODULE.build_parser().parse_args(
            [
                "--system",
                "openvoice_v2",
                "--input",
                "fixture.lst",
                "--test-set-id",
                "fixture",
            ]
        )
        self.assertTrue(parsed.continue_on_error)
        self.assertFalse(parsed.openvoice_enable_watermark)
        self.assertIsNone(parsed.openvoice_vad)
        self.assertEqual(
            MODULE.effective_openvoice_segmentation(parsed), "upstream_silero_vad"
        )
        config = MODULE.registered_inference_config(
            parsed, MODULE.resolve_asset_paths(parsed)
        )
        self.assertEqual(
            config["speaker_embedding_api"], "openvoice.se_extractor.get_se"
        )
        self.assertEqual(
            config["speaker_embedding_vad_implementation"],
            "whisper_timestamped.get_vad_segments(method=silero)",
        )
        self.assertTrue(config["upstream_silero_vad"])
        self.assertEqual(config["network_access"], "disabled")
        parsed.num_shards = 4
        parsed.shard_index = 2
        _output, shard_manifest, shard_summary = MODULE.resolve_outputs(parsed)
        self.assertEqual(
            shard_manifest.name, "manifest.shard-00002-of-00004.jsonl"
        )
        self.assertEqual(shard_summary.name, "summary.shard-00002-of-00004.json")

    def test_dry_run_writes_manifest_even_when_model_assets_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_test_wav(root / "source.wav")
            write_test_wav(root / "ref.wav")
            manifest = root / "cases.lst"
            manifest.write_text(
                "case-1|prompt|ref.wav|target|source.wav\n", encoding="utf-8"
            )
            output = root / "out"
            status = MODULE.main(
                [
                    "--system",
                    "openvoice_v2",
                    "--input",
                    str(manifest),
                    "--test-set-id",
                    "fixture",
                    "--output-dir",
                    str(output),
                    "--model-root",
                    str(root / "missing-model"),
                    "--dry-run",
                ]
            )
            rows = read_jsonl(output / "manifest.jsonl")
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(status, 0)
        self.assertEqual(rows[0]["status"], "dry_run")
        self.assertFalse(rows[0]["runtime_ready"])
        self.assertEqual(summary["status"], "dry_run_complete")

    def test_resume_skips_prior_success_without_loading_backend_again(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_test_wav(root / "source.wav")
            write_test_wav(root / "ref.wav")
            manifest = root / "cases.lst"
            manifest.write_text(
                "case-1|prompt|ref.wav|target|source.wav\n", encoding="utf-8"
            )
            output = root / "out"
            args = MODULE.build_parser().parse_args(
                [
                    "--system",
                    "freevc_v1",
                    "--input",
                    str(manifest),
                    "--test-set-id",
                    "fixture",
                    "--output-dir",
                    str(output),
                    "--model-root",
                    str(root / "missing-model"),
                    "--min-output-bytes",
                    "100",
                ]
            )
            first = FakeBackend()
            status_first = MODULE.execute(
                args, backend_factory=lambda _args, _paths, _out: first
            )
            second = FakeBackend()
            status_second = MODULE.execute(
                args, backend_factory=lambda _args, _paths, _out: second
            )
            rows = read_jsonl(output / "manifest.jsonl")
        self.assertEqual(status_first, 0)
        self.assertEqual(status_second, 0)
        self.assertEqual(first.calls, ["case-1"])
        self.assertEqual(second.calls, [])
        self.assertEqual(rows[0]["status"], "ok")

    def test_case_failure_does_not_block_later_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("source1.wav", "ref1.wav", "source2.wav", "ref2.wav"):
                write_test_wav(root / name)
            manifest = root / "cases.lst"
            manifest.write_text(
                "bad|prompt|ref1.wav|target|source1.wav\n"
                "good|prompt|ref2.wav|target|source2.wav\n",
                encoding="utf-8",
            )
            output = root / "out"
            args = MODULE.build_parser().parse_args(
                [
                    "--system",
                    "openvoice_v2",
                    "--input",
                    str(manifest),
                    "--test-set-id",
                    "fixture",
                    "--output-dir",
                    str(output),
                    "--model-root",
                    str(root / "missing-model"),
                    "--min-output-bytes",
                    "100",
                ]
            )
            backend = FakeBackend({"bad"})
            status = MODULE.execute(
                args, backend_factory=lambda _args, _paths, _out: backend
            )
            rows = {row["case_id"]: row for row in read_jsonl(output / "manifest.jsonl")}
        self.assertEqual(status, 0)
        self.assertEqual(backend.calls, ["bad", "good"])
        self.assertEqual(rows["bad"]["status"], "error")
        self.assertEqual(rows["good"]["status"], "ok")

    def test_openvoice_full_audio_segmentation_is_local_and_deterministic(self) -> None:
        import librosa
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "speaker.wav"
            write_test_wav(audio, frames=40_000)
            backend = MODULE.OpenVoiceBackend.__new__(MODULE.OpenVoiceBackend)
            backend.librosa = librosa
            backend.sf = sf
            backend.segmentation = "full_audio"
            backend.max_segment_seconds = 1.0
            backend.vad_top_db = 40.0
            backend.energy_vad_fallbacks = 0
            backend.offline_segments_written = 0
            backend.work_dir = root / "work"
            segments = backend._offline_segments(audio, root / "cache-key.pth")

        self.assertEqual(len(segments), 3)
        self.assertEqual([path.name for path in segments], [
            "segment-00000.wav",
            "segment-00001.wav",
            "segment-00002.wav",
        ])
        self.assertEqual(backend.offline_segments_written, 3)
        self.assertEqual(backend.energy_vad_fallbacks, 0)

    def test_openvoice_upstream_silero_short_audio_retry_stays_upstream(self) -> None:
        import torch

        class FakeSEExtractor:
            def __init__(self) -> None:
                self.get_se_calls = 0
                self.split_seconds: list[float] = []

            def split_audio_vad(
                self, audio_path, audio_name, target_dir, split_seconds=10.0
            ):
                del audio_path, audio_name, target_dir
                self.split_seconds.append(float(split_seconds))
                return "fixture-wavs"

            def get_se(self, audio_path, converter, target_dir, vad=True):
                del audio_path, converter, target_dir
                self.get_se_calls += 1
                self.assert_vad = vad
                if self.get_se_calls == 1:
                    raise AssertionError("input audio is too short")
                self.split_audio_vad("audio.wav", "speaker", "work")
                return torch.ones(1, 4), "speaker"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "speaker.wav"
            write_test_wav(audio, frames=40_000)
            backend = MODULE.OpenVoiceBackend.__new__(MODULE.OpenVoiceBackend)
            backend.torch = torch
            backend.device = "cpu"
            backend.converter = object()
            backend.se_extractor = FakeSEExtractor()
            backend.segmentation = "upstream_silero_vad"
            backend.vad_top_db = 40.0
            backend.max_segment_seconds = 20.0
            backend.silero_short_retry_split_seconds = 2.0
            backend.upstream_silero_calls = 0
            backend.upstream_silero_short_audio_retries = 0
            backend.cache_dir = root / "cache"
            backend.work_dir = root / "work"
            backend.cache_dir.mkdir()
            backend.work_dir.mkdir()
            embedding = backend._speaker_embedding(audio)

        self.assertEqual(tuple(embedding.shape), (1, 4))
        self.assertTrue(backend.se_extractor.assert_vad)
        self.assertEqual(backend.se_extractor.get_se_calls, 2)
        self.assertEqual(backend.se_extractor.split_seconds, [2.0])
        self.assertEqual(backend.upstream_silero_calls, 2)
        self.assertEqual(backend.upstream_silero_short_audio_retries, 1)


if __name__ == "__main__":
    unittest.main()

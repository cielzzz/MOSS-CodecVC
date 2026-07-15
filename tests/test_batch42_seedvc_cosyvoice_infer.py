from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/004085_run_batch42_seedvc_cosyvoice.py"
SPEC = importlib.util.spec_from_file_location("batch42_seedvc_cosyvoice", SCRIPT)
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


class Batch42SeedVCCosyVoiceInferTest(unittest.TestCase):
    def test_seedtts_lst_uses_field5_source_and_field3_reference(self) -> None:
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

    def test_four_column_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hard.lst"
            path.write_text(
                "hard|prompt|prompt-wavs/ref.wav|text without source\n",
                encoding="utf-8",
            )
            cases, issues = MODULE.read_input(path)
        self.assertEqual(cases, [])
        self.assertEqual(len(issues), 1)
        self.assertIn("got 4", issues[0].message)
        self.assertIn("Field 5", issues[0].message)

    def test_canonical_jsonl_never_guesses_target_audio_as_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "cases.jsonl"
            rows = [
                {
                    "case_id": "nested",
                    "audio": {"source": "source.wav", "reference": "ref.wav"},
                    "reference_text": "hello",
                },
                {
                    "case_id": "ambiguous",
                    "target_audio": "must-not-be-source.wav",
                    "reference_audio": "ref2.wav",
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            cases, issues = MODULE.read_input(path)
        self.assertEqual([case.case_id for case in cases], ["nested"])
        self.assertEqual(cases[0].language, "en")
        self.assertEqual(cases[0].field_mapping["source_audio"], "audio.source")
        self.assertEqual(len(issues), 1)
        self.assertIn("intentionally not guessed", issues[0].message)

    def test_registered_defaults_are_v2_timbre_only_and_direct_vc(self) -> None:
        parser = MODULE.build_parser()
        seed_args = parser.parse_args(
            [
                "--system",
                "seed_vc_v2",
                "--input",
                "cases.lst",
                "--test-set-id",
                "fixture",
            ]
        )
        seed_paths = MODULE.resolve_asset_paths(seed_args)
        seed_config = MODULE.registered_inference_config(seed_args, seed_paths)
        self.assertFalse(seed_args.seed_convert_style)
        self.assertEqual(seed_paths["ar_checkpoint"].name, "ar_base.pth")
        self.assertEqual(seed_paths["ar_checkpoint"].parent.name, "v2")
        self.assertEqual(seed_paths["cfm_checkpoint"].name, "cfm_small.pth")
        self.assertEqual(
            seed_config["api"], "VoiceConversionWrapper.convert_voice_with_streaming"
        )
        self.assertFalse(seed_args.seed_disable_cudnn)
        self.assertFalse(seed_config["disable_cudnn"])

        seed_ar_args = parser.parse_args(
            [
                "--system",
                "seed_vc_v2",
                "--input",
                "cases.lst",
                "--test-set-id",
                "fixture",
                "--seed-convert-style",
            ]
        )
        self.assertTrue(seed_ar_args.seed_convert_style)

        seed_h200_args = parser.parse_args(
            [
                "--system",
                "seed_vc_v2",
                "--input",
                "cases.lst",
                "--test-set-id",
                "fixture",
                "--seed-disable-cudnn",
            ]
        )
        seed_h200_paths = MODULE.resolve_asset_paths(seed_h200_args)
        seed_h200_config = MODULE.registered_inference_config(
            seed_h200_args, seed_h200_paths
        )
        self.assertTrue(seed_h200_args.seed_disable_cudnn)
        self.assertTrue(seed_h200_config["disable_cudnn"])

        cosy_args = parser.parse_args(
            [
                "--system",
                "cosyvoice2_vc",
                "--input",
                "cases.lst",
                "--test-set-id",
                "fixture",
            ]
        )
        cosy_paths = MODULE.resolve_asset_paths(cosy_args)
        cosy_config = MODULE.registered_inference_config(cosy_args, cosy_paths)
        self.assertEqual(cosy_config["api"], "CosyVoice2.inference_vc")
        self.assertTrue(cosy_config["direct_vc"])
        self.assertEqual(cosy_args.cosy_speech_tokenizer_provider, "cuda")
        self.assertEqual(cosy_config["speech_tokenizer_onnx_provider"], "cuda")
        self.assertFalse(cosy_config["speech_tokenizer_cpu_fallback"])
        self.assertEqual(cosy_config["main_model_device"], "cuda:0")
        self.assertEqual(
            MODULE.EXPECTED_CODE_REVISIONS["cosyvoice2_vc"],
            "8555549e882236e6541748b1042d95693caa82ba",
        )

    def test_cosy_cpu_provider_overrides_only_speech_tokenizer(self) -> None:
        calls: list[tuple[str, list[str] | None]] = []

        class FakeSession:
            def __init__(self, providers):
                self.providers = list(providers or [])

            def get_providers(self):
                return self.providers

        def original(model_path, *unused_args, **kwargs):
            del unused_args
            providers = kwargs.get("providers")
            calls.append((str(model_path), providers))
            return FakeSession(providers)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer = root / "speech_tokenizer_v2.onnx"
            campplus = root / "campplus.onnx"
            events: list[dict] = []
            factory = MODULE.cosy_onnx_inference_session_factory(
                original,
                speech_tokenizer_path=tokenizer,
                provider_mode="cpu",
                events=events,
            )
            tokenizer_session = factory(
                str(tokenizer), providers=["CUDAExecutionProvider"]
            )
            factory(str(campplus), providers=["CPUExecutionProvider"])

        self.assertEqual(
            tokenizer_session.get_providers(), ["CPUExecutionProvider"]
        )
        self.assertEqual(calls[0][1], ["CPUExecutionProvider"])
        self.assertEqual(calls[1][1], ["CPUExecutionProvider"])
        self.assertEqual(events[-1]["event"], "speech_tokenizer_session_created")

    def test_cosy_auto_provider_explicitly_falls_back_to_cpu(self) -> None:
        calls: list[list[str]] = []

        class FakeSession:
            def __init__(self, providers):
                self.providers = list(providers)

            def get_providers(self):
                return self.providers

        def original(_model_path, *unused_args, **kwargs):
            del unused_args
            providers = list(kwargs["providers"])
            calls.append(providers)
            if providers[0] == "CUDAExecutionProvider":
                raise RuntimeError("CUDNN_STATUS_NOT_INITIALIZED")
            return FakeSession(providers)

        events: list[dict] = []
        factory = MODULE.cosy_onnx_inference_session_factory(
            original,
            speech_tokenizer_path=Path("/models/speech_tokenizer_v2.onnx"),
            provider_mode="auto",
            events=events,
        )
        session = factory("/models/speech_tokenizer_v2.onnx")

        self.assertEqual(calls, [
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CPUExecutionProvider"],
        ])
        self.assertEqual(session.get_providers(), ["CPUExecutionProvider"])
        self.assertEqual(events[0]["fallback"], "CPUExecutionProvider")
        self.assertEqual(events[1]["providers"], ["CPUExecutionProvider"])

    def test_seed_cudnn_compatibility_policy(self) -> None:
        cudnn = SimpleNamespace(enabled=True)
        torch_module = SimpleNamespace(
            backends=SimpleNamespace(cudnn=cudnn)
        )
        state = MODULE.configure_seed_cudnn(torch_module, disable=True)
        self.assertFalse(state)
        self.assertFalse(cudnn.enabled)

        no_cudnn = SimpleNamespace(backends=SimpleNamespace())
        self.assertIsNone(
            MODULE.configure_seed_cudnn(no_cudnn, disable=True)
        )

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
            MODULE.deterministic_wav_name(case), "unsafe-case__0123456789ab.wav"
        )
        self.assertTrue(MODULE.shard_selected(5, 4, 1))
        self.assertFalse(MODULE.shard_selected(5, 4, 2))

    def test_dry_run_writes_manifest_with_missing_models(self) -> None:
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
                    "seed_vc_v2",
                    "--input",
                    str(manifest),
                    "--test-set-id",
                    "fixture",
                    "--output-dir",
                    str(output),
                    "--repo-root",
                    str(root / "missing-repo"),
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
        self.assertFalse(
            rows[0]["provenance"]["inference_config"]["convert_style"]
        )
        self.assertEqual(summary["status"], "dry_run_complete")

    def test_resume_keeps_prior_success_without_constructing_backend(self) -> None:
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
                    "seed_vc_v2",
                    "--input",
                    str(manifest),
                    "--test-set-id",
                    "fixture",
                    "--output-dir",
                    str(output),
                    "--repo-root",
                    str(root / "missing-repo"),
                    "--model-root",
                    str(root / "missing-model"),
                    "--min-output-bytes",
                    "100",
                ]
            )
            first = FakeBackend()
            first_status = MODULE.execute(
                args, backend_factory=lambda _args, _paths, _out: first
            )

            # No injected factory on the second pass: resume must not be blocked
            # merely because model assets are currently unavailable.
            second_status = MODULE.execute(args)
            rows = read_jsonl(output / "manifest.jsonl")
        self.assertEqual(first_status, 0)
        self.assertEqual(second_status, 0)
        self.assertEqual(first.calls, ["case-1"])
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["resume_action"], "kept_prior_success")

    def test_continue_on_error_processes_later_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("s1.wav", "r1.wav", "s2.wav", "r2.wav"):
                write_test_wav(root / name)
            manifest = root / "cases.lst"
            manifest.write_text(
                "bad|p|r1.wav|t|s1.wav\n"
                "good|p|r2.wav|t|s2.wav\n",
                encoding="utf-8",
            )
            output = root / "out"
            args = MODULE.build_parser().parse_args(
                [
                    "--system",
                    "cosyvoice2_vc",
                    "--input",
                    str(manifest),
                    "--test-set-id",
                    "fixture",
                    "--output-dir",
                    str(output),
                    "--repo-root",
                    str(root / "missing-repo"),
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

    def test_no_continue_on_error_stops_after_first_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("s1.wav", "r1.wav", "s2.wav", "r2.wav"):
                write_test_wav(root / name)
            manifest = root / "cases.lst"
            manifest.write_text(
                "bad|p|r1.wav|t|s1.wav\n"
                "never|p|r2.wav|t|s2.wav\n",
                encoding="utf-8",
            )
            output = root / "out"
            args = MODULE.build_parser().parse_args(
                [
                    "--system",
                    "seed_vc_v2",
                    "--input",
                    str(manifest),
                    "--test-set-id",
                    "fixture",
                    "--output-dir",
                    str(output),
                    "--repo-root",
                    str(root / "missing-repo"),
                    "--model-root",
                    str(root / "missing-model"),
                    "--min-output-bytes",
                    "100",
                    "--no-continue-on-error",
                ]
            )
            backend = FakeBackend({"bad"})
            status = MODULE.execute(
                args, backend_factory=lambda _args, _paths, _out: backend
            )
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(status, 1)
        self.assertEqual(backend.calls, ["bad"])
        self.assertEqual(summary["status"], "stopped_on_error")


if __name__ == "__main__":
    unittest.main()

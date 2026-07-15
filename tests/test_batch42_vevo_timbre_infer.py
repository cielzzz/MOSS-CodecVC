from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/004087_run_batch42_vevo_timbre.py"
SPEC = importlib.util.spec_from_file_location("batch42_vevo_timbre", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


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
        return {"backend": "fixture-vevo"}


class Batch42VevoTimbreInferTest(unittest.TestCase):
    def test_common_runner_is_reused_for_protocol_sensitive_helpers(self) -> None:
        self.assertIs(MODULE.read_input, MODULE.COMMON.read_input)
        self.assertIs(
            MODULE.deterministic_wav_name, MODULE.COMMON.deterministic_wav_name
        )
        self.assertIs(MODULE.shard_selected, MODULE.COMMON.shard_selected)

    def test_seedtts_lst_uses_field5_source_and_field3_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            reference = root / "reference.wav"
            write_test_wav(source)
            write_test_wav(reference)
            manifest = root / "vc.lst"
            manifest.write_text(
                "case-1|prompt|reference.wav|target|source.wav\n",
                encoding="utf-8",
            )
            cases, issues = MODULE.read_input(manifest)
        self.assertEqual(issues, [])
        self.assertEqual(cases[0].source_audio, source.resolve())
        self.assertEqual(cases[0].reference_audio, reference.resolve())
        self.assertEqual(
            cases[0].field_mapping["source_audio"], "field_5/source_audio"
        )
        self.assertEqual(
            cases[0].field_mapping["reference_audio"], "field_3/prompt_audio"
        )

    def test_four_columns_and_ambiguous_target_audio_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "bad.lst"
            manifest.write_text(
                "bad|prompt|reference.wav|target-without-source\n",
                encoding="utf-8",
            )
            cases, issues = MODULE.read_input(manifest)
            self.assertEqual(cases, [])
            self.assertIn("got 4", issues[0].message)

            canonical = root / "bad.jsonl"
            canonical.write_text(
                json.dumps(
                    {
                        "case_id": "ambiguous",
                        "target_audio": "must-not-be-source.wav",
                        "reference_audio": "reference.wav",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            cases, issues = MODULE.read_input(canonical)
        self.assertEqual(cases, [])
        self.assertEqual(issues[0].case_id, "ambiguous")
        self.assertIn("source_audio", issues[0].message)

    def test_deterministic_filename_modulo_sharding_and_registered_defaults(self) -> None:
        case = MODULE.VCCase(
            input_index=5,
            input_line=6,
            case_id="unsafe / case 中文",
            case_uid="0123456789abcdef0123",
            source_audio=Path("source.wav"),
            reference_audio=Path("reference.wav"),
        )
        self.assertEqual(
            MODULE.deterministic_wav_name(case),
            "unsafe-case__0123456789ab.wav",
        )
        self.assertTrue(MODULE.shard_selected(5, 4, 1))
        self.assertFalse(MODULE.shard_selected(5, 4, 2))
        parsed = MODULE.build_parser().parse_args(
            ["--input", "fixture.lst", "--test-set-id", "fixture"]
        )
        self.assertEqual(parsed.flow_matching_steps, 32)
        self.assertEqual(parsed.target_db, -25.0)
        self.assertEqual(parsed.model_revision, MODULE.DEFAULT_MODEL_REVISION)
        self.assertTrue(parsed.continue_on_error)

    def test_safetensors_header_probe_rejects_corruption(self) -> None:
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "valid.safetensors"
            invalid = root / "invalid.safetensors"
            save_file({"fixture.weight": torch.zeros(2, 3)}, str(valid))
            invalid.write_bytes(b"not-a-safetensors-file")
            valid_state = MODULE.safetensors_header_state(valid)
            invalid_state = MODULE.safetensors_header_state(invalid)
        self.assertTrue(valid_state["ready"])
        self.assertEqual(valid_state["tensor_count"], 1)
        self.assertEqual(valid_state["first_tensor_keys"], ["fixture.weight"])
        self.assertFalse(invalid_state["ready"])

    def test_dry_run_writes_manifest_and_incomplete_runtime_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_test_wav(root / "source.wav")
            write_test_wav(root / "reference.wav")
            manifest = root / "cases.lst"
            manifest.write_text(
                "case-1|prompt|reference.wav|target|source.wav\n",
                encoding="utf-8",
            )
            output = root / "out"
            original_hooks = (
                MODULE.COMMON.resolve_asset_paths,
                MODULE.COMMON.runtime_audit,
                MODULE.COMMON.default_backend_factory,
            )
            status = MODULE.main(
                [
                    "--input",
                    str(manifest),
                    "--test-set-id",
                    "fixture",
                    "--output-dir",
                    str(output),
                    "--model-root",
                    str(root / "missing-model"),
                    "--torch-home",
                    str(root / "missing-torch"),
                    "--device",
                    "cpu",
                    "--dry-run",
                ]
            )
            rows = read_jsonl(output / "manifest.jsonl")
            audit = json.loads(
                (output / "runtime_audit.json").read_text(encoding="utf-8")
            )
        self.assertEqual(status, 0)
        self.assertIs(MODULE.COMMON.resolve_asset_paths, original_hooks[0])
        self.assertIs(MODULE.COMMON.runtime_audit, original_hooks[1])
        self.assertIs(MODULE.COMMON.default_backend_factory, original_hooks[2])
        self.assertEqual(rows[0]["system_id"], MODULE.SYSTEM_ID)
        self.assertEqual(rows[0]["status"], "dry_run")
        self.assertFalse(rows[0]["runtime_ready"])
        self.assertFalse(audit["ready"])
        self.assertIn("file:tokenizer_checkpoint", "\n".join(audit["blocking_reasons"]))
        self.assertEqual(
            rows[0]["provenance"]["field_mapping"]["source_audio"],
            "field_5/source_audio",
        )

    def test_resume_and_per_case_error_come_from_shared_execute_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (
                "source1.wav",
                "reference1.wav",
                "source2.wav",
                "reference2.wav",
            ):
                write_test_wav(root / name)
            manifest = root / "cases.lst"
            manifest.write_text(
                "bad|prompt|reference1.wav|target|source1.wav\n"
                "good|prompt|reference2.wav|target|source2.wav\n",
                encoding="utf-8",
            )
            output = root / "out"
            args = MODULE.build_parser().parse_args(
                [
                    "--input",
                    str(manifest),
                    "--test-set-id",
                    "fixture",
                    "--output-dir",
                    str(output),
                    "--model-root",
                    str(root / "missing-model"),
                    "--torch-home",
                    str(root / "missing-torch"),
                    "--device",
                    "cpu",
                    "--min-output-bytes",
                    "100",
                ]
            )
            first = FakeBackend({"bad"})
            first_status = MODULE.execute(
                args, backend_factory=lambda _args, _paths, _out: first
            )
            rows = {row["case_id"]: row for row in read_jsonl(output / "manifest.jsonl")}
            second = FakeBackend()
            second_status = MODULE.execute(
                args, backend_factory=lambda _args, _paths, _out: second
            )
        self.assertEqual(first_status, 0)
        self.assertEqual(first.calls, ["bad", "good"])
        self.assertEqual(rows["bad"]["status"], "error")
        self.assertEqual(rows["good"]["status"], "ok")
        self.assertEqual(second_status, 0)
        self.assertEqual(second.calls, ["bad"])

    def test_backend_passes_strict_source_reference_and_official_defaults(self) -> None:
        class FakePipeline:
            def __init__(self) -> None:
                self.calls = []

            def inference_fm(self, **kwargs):
                self.calls.append(kwargs)
                return object()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            reference = root / "reference.wav"
            output = root / "generated.wav"
            write_test_wav(source)
            write_test_wav(reference)
            case = MODULE.VCCase(
                input_index=0,
                input_line=1,
                case_id="case-1",
                case_uid="0123456789abcdef0123",
                source_audio=source,
                reference_audio=reference,
            )
            backend = object.__new__(MODULE.VevoTimbreBackend)
            import torch

            backend.torch = torch
            backend.pipeline = FakePipeline()
            backend.flow_matching_steps = 32
            backend.target_db = -25.0

            def fake_save_audio(_waveform, *, output_path, target_db):
                self.assertEqual(target_db, -25.0)
                write_test_wav(Path(output_path))

            backend.save_audio = fake_save_audio
            details = backend.convert(case, output)
            self.assertEqual(len(backend.pipeline.calls), 1)
            call = backend.pipeline.calls[0]
            self.assertEqual(call["src_wav_path"], str(source))
            self.assertEqual(call["timbre_ref_wav_path"], str(reference))
            self.assertEqual(call["flow_matching_steps"], 32)
            self.assertEqual(
                details["official_entry"],
                "models/vc/vevo/infer_vevotimbre.py",
            )
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()

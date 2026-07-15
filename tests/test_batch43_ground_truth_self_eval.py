from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/004097_prepare_batch43_ground_truth_self_eval.py"
WRAPPER = ROOT / "scripts/004098_submit_batch43_ground_truth_self_eval_qz.sh"
SPEC = importlib.util.spec_from_file_location("batch43_ground_truth_self_eval", SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot import {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_wav(path: Path, *, sample: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(int(sample).to_bytes(2, "little", signed=True) * 320)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Batch43GroundTruthSelfEvalTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        dataset = root / "dataset"
        manifests = root / "manifests"
        manifests.mkdir(parents=True)
        for language, offset in (("en", 0), ("zh", 10)):
            write_wav(dataset / language / "prompt-wavs/prompt.wav", sample=1 + offset)
            write_wav(dataset / language / "wavs/source.wav", sample=2 + offset)
        en = manifests / "en.lst"
        zh = manifests / "zh.lst"
        en.write_text(
            "en-case|prompt words|prompt-wavs/prompt.wav|source words|wavs/source.wav\n",
            encoding="utf-8",
        )
        zh.write_text(
            "zh-case|提示文本|prompt-wavs/prompt.wav|源文本|wavs/source.wav\n",
            encoding="utf-8",
        )
        return dataset, en, zh

    def test_build_all_uses_source_same_file_and_preserves_real_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, en, zh = self.make_fixture(root)
            output = root / "output"
            args = MODULE.build_parser().parse_args(
                [
                    "--dataset-root",
                    str(dataset),
                    "--en-manifest",
                    str(en),
                    "--zh-manifest",
                    str(zh),
                    "--output-dir",
                    str(output),
                    "--expected-en",
                    "1",
                    "--expected-zh",
                    "1",
                    "--expected-en-sha256",
                    file_sha(en),
                    "--expected-zh-sha256",
                    file_sha(zh),
                ]
            )
            audit = MODULE.build_all(args)
            row = json.loads(
                (output / "ground_truth_source_self.en.input.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )

        self.assertEqual(row["system_id"], "ground_truth")
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["input_index"], 0)
        self.assertEqual(row["generated_audio"], row["reference_audio"])
        self.assertEqual(row["generated_audio"], row["source_audio"])
        self.assertNotEqual(row["generated_audio"], row["target_reference_audio"])
        self.assertEqual(row["reference_text"], "source words")
        self.assertTrue(row["evaluation_contract"]["calibration_only"])
        self.assertFalse(
            row["evaluation_contract"]["paired_target_speaker_ground_truth_available"]
        )
        self.assertTrue(audit["calibration_only"])
        self.assertEqual(audit["splits"]["en"]["generated_equals_reference_rows"], 1)
        self.assertEqual(audit["zh_hard"]["status"], "not_applicable")
        self.assertIn("no source/ground-truth waveform", audit["zh_hard"]["reason"])

    def test_wrong_manifest_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, en, _zh = self.make_fixture(root)
            spec = MODULE.SplitSpec(
                language="en",
                expected_rows=1,
                manifest_sha256="0" * 64,
                test_set_id="fixture",
                manifest_name="en.lst",
            )
            with self.assertRaisesRegex(ValueError, "manifest SHA256"):
                MODULE.build_split(
                    spec=spec,
                    manifest=en,
                    dataset_root=dataset,
                    output_jsonl=root / "out.jsonl",
                )

    def test_four_field_tts_or_hard_row_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, en, _zh = self.make_fixture(root)
            en.write_text(
                "hard|prompt text|prompt-wavs/prompt.wav|hard target text\n",
                encoding="utf-8",
            )
            spec = MODULE.SplitSpec("en", 1, file_sha(en), "fixture", "en.lst")
            with self.assertRaisesRegex(ValueError, "exactly 5 fields"):
                MODULE.build_split(
                    spec=spec,
                    manifest=en,
                    dataset_root=dataset,
                    output_jsonl=root / "out.jsonl",
                )

    def test_source_and_prompt_must_be_nonparallel_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, en, _zh = self.make_fixture(root)
            en.write_text(
                "same|prompt|prompt-wavs/prompt.wav|text|prompt-wavs/prompt.wav\n",
                encoding="utf-8",
            )
            spec = MODULE.SplitSpec("en", 1, file_sha(en), "fixture", "en.lst")
            with self.assertRaisesRegex(ValueError, "same path"):
                MODULE.build_split(
                    spec=spec,
                    manifest=en,
                    dataset_root=dataset,
                    output_jsonl=root / "out.jsonl",
                )

    def test_prepare_only_wrapper_never_invokes_qz_scorer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, en, zh = self.make_fixture(root)
            marker = root / "scorer-called"
            fake_scorer = root / "fake_scorer.sh"
            fake_scorer.write_text(
                f"#!/usr/bin/env bash\ntouch {marker}\n",
                encoding="utf-8",
            )
            fake_scorer.chmod(0o755)
            env = os.environ.copy()
            env.update(
                {
                    "PROJECT_ROOT": str(ROOT),
                    "DATASET_ROOT": str(dataset),
                    "EN_MANIFEST": str(en),
                    "ZH_MANIFEST": str(zh),
                    "EXPECTED_EN": "1",
                    "EXPECTED_ZH": "1",
                    "EXPECTED_EN_SHA256": file_sha(en),
                    "EXPECTED_ZH_SHA256": file_sha(zh),
                    "INPUT_ROOT": str(root / "inputs"),
                    "OUTPUT_ROOT": str(root / "scores"),
                    "RECORD_ROOT": str(root / "records"),
                    "SCORER_WRAPPER": str(fake_scorer),
                    "PREPARE_ONLY": "1",
                }
            )
            result = subprocess.run(
                ["bash", str(WRAPPER)],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            scorer_was_called = marker.exists()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PREPARE_ONLY=1; no QZ command invoked", result.stdout)
        self.assertFalse(scorer_was_called)

    def test_live_wrapper_requires_explicit_confirmation(self) -> None:
        env = os.environ.copy()
        env.update({"DRY_RUN": "0", "CONFIRM_GROUND_TRUTH_SELF_EVAL": "0"})
        result = subprocess.run(
            ["bash", str(WRAPPER)],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires CONFIRM_GROUND_TRUTH_SELF_EVAL=1", result.stderr)


if __name__ == "__main__":
    unittest.main()

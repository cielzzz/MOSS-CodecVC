from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/004093_run_batch42_pathx_strict.py"
SPEC = importlib.util.spec_from_file_location("batch42_pathx_strict", SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot import {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_wav(path: Path, *, frames: int = 1_600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * frames)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    MODULE.atomic_jsonl(path, rows)


def option_value(command: list[str], option: str) -> str:
    index = command.index(option)
    return command[index + 1]


def make_case(
    input_root: Path,
    *,
    case_id: str = "case-1",
    index: int = 0,
    language: str = "en",
) -> MODULE.StrictCase:
    source = input_root / f"source-{index}.wav"
    reference = input_root / f"reference-{index}.wav"
    write_wav(source)
    write_wav(reference)
    return MODULE.StrictCase(
        input_index=index,
        input_line=index + 1,
        case_id=case_id,
        case_uid=MODULE.stable_case_uid(case_id, source, reference),
        prompt_text=f"prompt {index}",
        target_text=f"target {index}",
        source_audio=source.resolve(),
        reference_audio=reference.resolve(),
        language=language,
    )


def make_run_args(
    root: Path,
    rows: list[dict],
    *,
    resume: bool = False,
    engine_dry_run: bool = False,
    max_cases: int = 0,
) -> tuple[object, Path]:
    input_root = root / "input"
    input_root.mkdir(parents=True, exist_ok=True)
    canonical = root / "canonical.jsonl"
    write_jsonl(canonical, rows)
    argv = [
        "run",
        "--canonical-jsonl",
        str(canonical),
        "--expected-canonical-sha256",
        MODULE.sha256_file(canonical),
        "--expected-cases",
        str(len(rows)),
        "--input-root",
        str(input_root),
        "--output-dir",
        str(root / "wav"),
        "--raw-manifest",
        str(root / "raw.jsonl"),
        "--manifest-jsonl",
        str(root / "manifest.jsonl"),
        "--summary-json",
        str(root / "summary.json"),
        "--test-set-id",
        "strict-en",
        "--language",
        "en",
        "--min-output-bytes",
        "100",
    ]
    if resume:
        argv.append("--resume")
    if engine_dry_run:
        argv.append("--engine-dry-run")
    if max_cases:
        argv.extend(("--max-cases", str(max_cases)))
    return MODULE.build_parser().parse_args(argv), canonical


class Batch42PathXStrictInputTest(unittest.TestCase):
    def test_field5_is_content_source_field3_is_timbre_ref_and_all_rows_are_no_text(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "seedtts"
            source = input_root / "wavs/source.wav"
            reference = input_root / "prompt-wavs/reference.wav"
            write_wav(source)
            write_wav(reference)
            strict = root / "strict.lst"
            strict.write_text(
                "case-1|prompt words|prompt-wavs/reference.wav|target words|wavs/source.wav\n",
                encoding="utf-8",
            )
            cases = MODULE.parse_strict_lst(
                strict,
                input_root=input_root,
                language="en",
                expected_cases=1,
                expected_sha256=MODULE.sha256_file(strict),
            )
            row = MODULE.canonical_record(cases[0], test_set_id="strict-en")

        self.assertEqual(cases[0].source_audio, source.resolve())
        self.assertEqual(cases[0].reference_audio, reference.resolve())
        self.assertEqual(row["source_audio"], str(source.resolve()))
        self.assertEqual(row["timbre_ref_audio"], str(reference.resolve()))
        self.assertEqual(row["reference_audio"], str(reference.resolve()))
        self.assertEqual(row["target_text"], "target words")
        self.assertEqual(row["source_text"], "target words")
        self.assertEqual(row["content_ref_text"], "target words")
        self.assertEqual(row["timbre_ref_text"], "prompt words")
        self.assertEqual(row["text"], "<NO_TEXT>")
        self.assertEqual(row["mode"], "no_text")
        self.assertEqual(row["moss_codecvc_mode"], "no_text")
        self.assertEqual(
            row["batch42_field_mapping"],
            {
                "source_audio": "field_5/source_audio",
                "timbre_ref_audio": "field_3/prompt_audio",
                "target_text": "field_4/target_text; scorer-only in no_text mode",
            },
        )

    def test_prepare_writes_audited_canonical_jsonl_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "seedtts"
            write_wav(input_root / "source.wav")
            write_wav(input_root / "ref.wav")
            strict = root / "strict.lst"
            strict.write_text(
                "case-1|prompt|ref.wav|target|source.wav\n", encoding="utf-8"
            )
            output = root / "canonical.jsonl"
            summary = root / "summary.json"
            with contextlib.redirect_stdout(io.StringIO()):
                status = MODULE.main(
                    [
                        "prepare",
                        "--input",
                        str(strict),
                        "--input-root",
                        str(input_root),
                        "--language",
                        "en",
                        "--expected-cases",
                        "1",
                        "--expected-sha256",
                        MODULE.sha256_file(strict),
                        "--test-set-id",
                        "strict-en",
                        "--output-jsonl",
                        str(output),
                        "--summary-json",
                        str(summary),
                    ]
                )
            rows = MODULE.read_jsonl(output)
            metadata = json.loads(summary.read_text(encoding="utf-8"))
            output_sha256 = MODULE.sha256_file(output)

        self.assertEqual(status, 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(metadata["cases"], 1)
        self.assertEqual(metadata["mode_counts"], {"no_text": 1})
        self.assertEqual(metadata["output_sha256"], output_sha256)

    def test_strict_manifest_rejects_wrong_hash_and_wrong_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_wav(root / "source.wav")
            write_wav(root / "ref.wav")
            strict = root / "strict.lst"
            strict.write_text("case|prompt|ref.wav|target|source.wav\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                MODULE.parse_strict_lst(
                    strict,
                    input_root=root,
                    language="en",
                    expected_cases=1,
                    expected_sha256="0" * 64,
                )
            with self.assertRaisesRegex(ValueError, "expected 2 strict cases, got 1"):
                MODULE.parse_strict_lst(
                    strict,
                    input_root=root,
                    language="en",
                    expected_cases=2,
                    expected_sha256=MODULE.sha256_file(strict),
                )

    def test_strict_manifest_rejects_blank_rows_duplicate_ids_and_safe_stem_collision(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("s0.wav", "r0.wav", "s1.wav", "r1.wav"):
                write_wav(root / name)

            blank = root / "blank.lst"
            blank.write_text(
                "first|p|r0.wav|t|s0.wav\n\nsecond|p|r1.wav|t|s1.wav\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "blank rows are forbidden"):
                MODULE.parse_strict_lst(
                    blank,
                    input_root=root,
                    language="en",
                    expected_cases=3,
                    expected_sha256=MODULE.sha256_file(blank),
                )

            duplicate = root / "duplicate.lst"
            duplicate.write_text(
                "same|p|r0.wav|t|s0.wav\nsame|p|r1.wav|t|s1.wav\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate case_id"):
                MODULE.parse_strict_lst(
                    duplicate,
                    input_root=root,
                    language="en",
                    expected_cases=2,
                    expected_sha256=MODULE.sha256_file(duplicate),
                )

            collision = root / "collision.lst"
            collision.write_text(
                "a/b|p|r0.wav|t|s0.wav\na b|p|r1.wav|t|s1.wav\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "safe_stem collision"):
                MODULE.parse_strict_lst(
                    collision,
                    input_root=root,
                    language="en",
                    expected_cases=2,
                    expected_sha256=MODULE.sha256_file(collision),
                )

    def test_strict_manifest_rejects_root_escape_missing_audio_and_same_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_wav(input_root / "ref.wav")
            outside = root / "outside.wav"
            write_wav(outside)

            escaped = root / "escaped.lst"
            escaped.write_text(
                f"case|p|ref.wav|t|{outside}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "escapes registered input root"):
                MODULE.parse_strict_lst(
                    escaped,
                    input_root=input_root,
                    language="en",
                    expected_cases=1,
                    expected_sha256=MODULE.sha256_file(escaped),
                )

            missing = root / "missing.lst"
            missing.write_text("case|p|ref.wav|t|missing.wav\n", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "missing source/reference"):
                MODULE.parse_strict_lst(
                    missing,
                    input_root=input_root,
                    language="en",
                    expected_cases=1,
                    expected_sha256=MODULE.sha256_file(missing),
                )

            same = root / "same.lst"
            same.write_text("case|p|ref.wav|t|ref.wav\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "same audio"):
                MODULE.parse_strict_lst(
                    same,
                    input_root=input_root,
                    language="en",
                    expected_cases=1,
                    expected_sha256=MODULE.sha256_file(same),
                )

    def test_canonical_audit_locks_hash_uid_indices_roots_and_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            case = make_case(input_root)
            row = MODULE.canonical_record(case, test_set_id="strict-en")
            canonical = root / "canonical.jsonl"
            write_jsonl(canonical, [row])
            audit = MODULE.audit_canonical_rows(
                [row],
                canonical_path=canonical,
                expected_cases=1,
                expected_sha256=MODULE.sha256_file(canonical),
                test_set_id="strict-en",
                language="en",
                input_root=input_root,
            )
            self.assertEqual(audit["rows"], 1)

            with self.assertRaisesRegex(ValueError, "canonical JSONL SHA256 mismatch"):
                MODULE.audit_canonical_rows(
                    [row],
                    canonical_path=canonical,
                    expected_cases=1,
                    expected_sha256="0" * 64,
                    test_set_id="strict-en",
                    language="en",
                    input_root=input_root,
                )

            bad = copy.deepcopy(row)
            bad["source_text"] = "not field four"
            write_jsonl(canonical, [bad])
            with self.assertRaisesRegex(ValueError, "source_text must preserve field 4"):
                MODULE.audit_canonical_rows(
                    [bad],
                    canonical_path=canonical,
                    expected_cases=1,
                    expected_sha256=MODULE.sha256_file(canonical),
                    test_set_id="strict-en",
                    language="en",
                    input_root=input_root,
                )

            bad = copy.deepcopy(row)
            bad["case_uid"] = "tampered"
            write_jsonl(canonical, [bad])
            with self.assertRaisesRegex(ValueError, "case_uid mismatch"):
                MODULE.audit_canonical_rows(
                    [bad],
                    canonical_path=canonical,
                    expected_cases=1,
                    expected_sha256=MODULE.sha256_file(canonical),
                    test_set_id="strict-en",
                    language="en",
                    input_root=input_root,
                )


class Batch42PathXRegisteredIdentityTest(unittest.TestCase):
    @staticmethod
    def registered_args() -> object:
        return MODULE.build_parser().parse_args(
            [
                "run",
                "--canonical-jsonl",
                "fixture.jsonl",
                "--expected-canonical-sha256",
                "0" * 64,
                "--expected-cases",
                "1",
                "--input-root",
                ".",
                "--output-dir",
                "out",
                "--raw-manifest",
                "raw.jsonl",
                "--manifest-jsonl",
                "manifest.jsonl",
                "--summary-json",
                "summary.json",
                "--test-set-id",
                "strict-en",
                "--language",
                "en",
            ]
        )

    def test_registered_checkpoint_base_model_and_wavlm_snapshot_are_complete(self) -> None:
        identity = MODULE.validate_registered_assets(self.registered_args())
        self.assertEqual(
            identity["adapter_base_model_name_or_path"],
            str(MODULE.REGISTERED_BASE_MODEL_PATH.resolve()),
        )
        self.assertEqual(
            identity["base_model_config"]["sha256"],
            MODULE.REGISTERED_BASE_MODEL_CONFIG_SHA256,
        )
        self.assertEqual(
            set(identity["source_semantic_files"]),
            {"config.json", "preprocessor_config.json", "pytorch_model.bin"},
        )
        self.assertEqual(
            identity["source_semantic_files"]["pytorch_model.bin"]["sha256"],
            "size_only_in_worker",
        )

    def test_registered_identity_rejects_base_model_override(self) -> None:
        args = self.registered_args()
        args.base_model_path = Path("/tmp/not-the-registered-base")
        with self.assertRaisesRegex(ValueError, "identity mismatch for base_model_path"):
            MODULE.validate_registered_assets(args)

    def test_registered_identity_requires_wavlm_files_and_exact_base_config_hash(self) -> None:
        args = self.registered_args()
        extra = dict(MODULE.REGISTERED_SOURCE_SEMANTIC_FILES)
        extra["missing-required.bin"] = {"size": 1, "sha256": "0" * 64}
        with mock.patch.object(MODULE, "REGISTERED_SOURCE_SEMANTIC_FILES", extra):
            with self.assertRaises(FileNotFoundError):
                MODULE.validate_registered_assets(args)
        with mock.patch.object(
            MODULE, "REGISTERED_BASE_MODEL_CONFIG_SHA256", "0" * 64
        ):
            with self.assertRaisesRegex(ValueError, "base-model config SHA256 mismatch"):
                MODULE.validate_registered_assets(args)


class Batch42PathXExecutionContractTest(unittest.TestCase):
    def test_engine_command_registers_all_behavior_knobs_and_resume_overwrite_semantics(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            case = make_case(input_root)
            row = MODULE.canonical_record(case, test_set_id="strict-en")
            args, _canonical = make_run_args(
                root, [row], engine_dry_run=True, max_cases=1
            )
            command = MODULE.build_engine_command(args)

            self.assertEqual(option_value(command, "--mode"), "no_text")
            self.assertEqual(option_value(command, "--seed"), "1234")
            self.assertEqual(option_value(command, "--temperature"), "0.7")
            self.assertEqual(option_value(command, "--audio-temperature"), "1.1")
            self.assertEqual(option_value(command, "--audio-top-p"), "0.7")
            self.assertEqual(option_value(command, "--audio-top-k"), "20")
            self.assertEqual(option_value(command, "--source-semantic-layer"), "9")
            self.assertEqual(
                option_value(command, "--source-semantic-downsample-stride"), "1"
            )
            self.assertEqual(
                Path(option_value(command, "--base-model-path")).resolve(),
                MODULE.REGISTERED_BASE_MODEL_PATH.resolve(),
            )
            self.assertIn("--source-semantic-local-files-only", command)
            self.assertIn("--no-filter-v2-real-no-text-ref-content-leak", command)
            self.assertIn("--no-ref-prompt-codec-permutation", command)
            self.assertIn("--no-ref-speaker-prompt-slot", command)
            self.assertIn("--no-timbre-side-only", command)
            self.assertIn("--overwrite", command)
            self.assertIn("--dry-run", command)

            args.resume = True
            resumed = MODULE.build_engine_command(args)
            self.assertNotIn("--overwrite", resumed)

    def test_engine_environment_removes_every_frozen_engine_env_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            row = MODULE.canonical_record(
                make_case(input_root), test_set_id="strict-en"
            )
            args, _canonical = make_run_args(root, [row])
            polluted = {key: "polluted" for key in MODULE.INFERENCE_ENV_KEYS}
            polluted["PYTHONPATH"] = "/old/pythonpath"
            with mock.patch.dict(os.environ, polluted, clear=True):
                env = MODULE.sanitized_engine_environment(args)

        explicit_zeroes = {
            "NO_TEXT_SOFT_DURATION_BUDGET",
            "DISABLE_MODE_TOKEN",
            "DISABLE_SOURCE_SEMANTIC_MEMORY",
            "SOURCE_SEMANTIC_RELEASE_AFTER_PROGRESS",
        }
        for key in MODULE.INFERENCE_ENV_KEYS - explicit_zeroes:
            self.assertNotIn(key, env, key)
        for key in explicit_zeroes:
            self.assertEqual(env[key], "0")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(
            env["PYTHONPATH"],
            f"{MODULE.REGISTERED_CODE_ROOT.resolve()}{os.pathsep}/old/pythonpath",
        )

    def test_raw_ledger_is_last_row_wins_and_duplicate_count_is_audited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            case = make_case(input_root)
            canonical = MODULE.canonical_record(case, test_set_id="strict-en")
            args, _canonical_path = make_run_args(root, [canonical])
            generated = args.output_dir / f"{MODULE.safe_stem(case.case_id)}.wav"
            write_wav(generated)
            raw_rows = [
                {
                    "case_id": case.case_id,
                    "status": "failed",
                    "output_wav": str(generated),
                    "error": "stale failure",
                },
                {
                    "case_id": case.case_id,
                    "status": "ok",
                    "output_wav": str(generated),
                    "elapsed_sec": 1.25,
                },
            ]
            records, counts, audit = MODULE.convert_raw_manifest(
                args, [canonical], raw_rows
            )

        self.assertEqual(counts, {"ok": 1})
        self.assertEqual(records[0]["status"], "ok")
        self.assertEqual(records[0]["runtime_seconds"], 1.25)
        self.assertNotIn("error", records[0])
        self.assertEqual(audit["raw_rows"], 2)
        self.assertEqual(audit["raw_unique_case_ids"], 1)
        self.assertEqual(audit["raw_duplicate_rows"], 1)

    def test_resume_rebuilds_raw_ledger_but_reuses_existing_wav(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            case = make_case(input_root)
            canonical = MODULE.canonical_record(case, test_set_id="strict-en")
            args, _canonical_path = make_run_args(root, [canonical], resume=True)
            generated = args.output_dir / f"{MODULE.safe_stem(case.case_id)}.wav"
            write_wav(generated)
            args.raw_manifest.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(
                args.raw_manifest,
                [
                    {"case_id": case.case_id, "status": "failed"},
                    {"case_id": case.case_id, "status": "failed"},
                ],
            )
            observed: dict[str, object] = {}

            def fake_run(command, *, cwd, env, check):
                self.assertFalse(args.raw_manifest.exists())
                self.assertNotIn("--overwrite", command)
                self.assertEqual(cwd, MODULE.REGISTERED_CODE_ROOT.resolve())
                self.assertFalse(check)
                observed["command"] = command
                observed["env"] = env
                write_jsonl(
                    args.raw_manifest,
                    [
                        {
                            "case_id": case.case_id,
                            "status": "skipped_exists",
                            "output_wav": str(generated),
                            "elapsed_sec": 0.0,
                        }
                    ],
                )
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(
                MODULE,
                "validate_registered_assets",
                return_value={"identity": "fixture"},
            ), mock.patch.object(MODULE.subprocess, "run", side_effect=fake_run):
                with contextlib.redirect_stdout(io.StringIO()):
                    status = MODULE.run_command(args)

            raw = MODULE.read_jsonl(args.raw_manifest)
            records = MODULE.read_jsonl(args.manifest_jsonl)
            summary = json.loads(args.summary_json.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(len(raw), 1)
        self.assertEqual(records[0]["status"], "skipped_existing")
        self.assertEqual(summary["raw_manifest_audit"]["raw_duplicate_rows"], 0)
        self.assertEqual(summary["status"], "complete")
        self.assertIn("command", observed)


if __name__ == "__main__":
    unittest.main()

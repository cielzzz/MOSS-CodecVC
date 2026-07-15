from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PREP_SCRIPT = ROOT / "scripts/004130_prepare_batch42_gt_half_split.py"
RUNNER = ROOT / "scripts/004131_run_batch42_gt_half_split_local.sh"
EVAL_SCRIPT = ROOT / "scripts/004082_run_unified_vc_eval.py"


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREP = load_script(PREP_SCRIPT, "test_batch42_gt_half_split_prep")
EVAL = load_script(EVAL_SCRIPT, "test_batch42_gt_half_split_eval")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_wav(path: Path, *, frames: int, sample_rate: int = 1000, offset: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = bytearray()
    for index in range(frames):
        value = ((index + offset) % 2000) - 1000
        payload.extend(int(value).to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(payload))


def make_parent_fixture(root: Path, *, bad_role: bool = False) -> dict[str, Path]:
    parent = root / "parent"
    inputs: dict[str, Path] = {}
    split_payloads = {}
    for language, offset in (("en", 0), ("zh", 100)):
        long_wav = root / "audio" / language / "long.wav"
        short_wav = root / "audio" / language / "short.wav"
        prompt = root / "audio" / language / "prompt.wav"
        write_wav(long_wav, frames=1001, offset=offset)
        write_wav(short_wav, frames=500, offset=offset + 10)
        write_wav(prompt, frames=1000, offset=offset + 20)
        rows = []
        for index, source in enumerate((long_wav, short_wav)):
            source_text = "hello world" if language == "en" else "你好世界"
            rows.append(
                {
                    "schema_version": PREP.PARENT_INPUT_SCHEMA,
                    "record_type": "ground_truth_source_self_eval_input",
                    "status": "ok",
                    "system_id": "ground_truth",
                    "test_set_id": f"fixture-{language}",
                    "case_id": f"{language}-case-{index}",
                    "case_uid": f"{language}-uid-{index}",
                    "input_index": index,
                    "input_line": index + 1,
                    "language": language,
                    "generated_audio": str(source.resolve()),
                    "reference_audio": (
                        str(prompt.resolve())
                        if bad_role and language == "en" and index == 0
                        else str(source.resolve())
                    ),
                    "source_audio": str(source.resolve()),
                    "target_reference_audio": str(prompt.resolve()),
                    "reference_text": source_text,
                    "target_text": source_text,
                    "evaluation_contract": {"calibration_only": True},
                    "provenance": {},
                }
            )
        input_path = parent / f"ground_truth_source_self.{language}.input.jsonl"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        inputs[language] = input_path
        split_payloads[language] = {
            "language": language,
            "test_set_id": f"fixture-{language}",
            "rows": 2,
            "output_jsonl": str(input_path.resolve()),
            "output_jsonl_sha256": sha(input_path),
            "generated_equals_reference_rows": 2,
        }
    audit = parent / "GROUND_TRUTH_SOURCE_SELF_AUDIT.json"
    audit.write_text(
        json.dumps(
            {
                "schema_version": PREP.PARENT_AUDIT_SCHEMA,
                "system_id": "ground_truth",
                "calibration_only": True,
                "splits": split_payloads,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {"en": inputs["en"], "zh": inputs["zh"], "audit": audit}


def build_args(paths: dict[str, Path], output: Path):
    return PREP.build_parser().parse_args(
        [
            "--en-input",
            str(paths["en"]),
            "--zh-input",
            str(paths["zh"]),
            "--parent-audit",
            str(paths["audit"]),
            "--output-dir",
            str(output),
            "--expected-en",
            "2",
            "--expected-zh",
            "2",
            "--expected-en-sha256",
            sha(paths["en"]),
            "--expected-zh-sha256",
            sha(paths["zh"]),
            "--expected-parent-audit-sha256",
            sha(paths["audit"]),
            "--min-half-seconds",
            "0.4",
        ]
    )


def test_prepare_builds_nonoverlapping_balanced_halves_and_skip_ledger(
    tmp_path: Path,
) -> None:
    paths = make_parent_fixture(tmp_path)
    parent_hashes = {name: sha(path) for name, path in paths.items()}
    output = tmp_path / "half_split"
    audit = PREP.build_all(build_args(paths, output))

    assert audit["schema_version"] == PREP.AUDIT_SCHEMA
    assert audit["scoring_contract"]["asr_backends"] == []
    assert audit["existing_same_file_results"]["status"] == "untouched"
    for name, path in paths.items():
        assert sha(path) == parent_hashes[name]

    for language in ("en", "zh"):
        split = audit["splits"][language]
        assert split["parent_rows"] == 2
        assert split["kept_rows"] == 1
        assert split["skipped_rows"] == 1
        assert split["reason_counts"] == {
            "front_below_min_half_seconds": 1,
            "keep": 1,
        }
        assert split["max_balance_delta_frames"] == 1
        input_path = Path(split["input_jsonl"]["path"])
        ledger_path = Path(split["ledger_jsonl"]["path"])
        assert input_path.is_file() and ledger_path.is_file()
        row = json.loads(input_path.read_text(encoding="utf-8").strip())
        ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
        assert [item["status"] for item in ledger] == ["kept", "skipped"]
        assert ledger[1]["reason"] == "front_below_min_half_seconds"
        front = Path(row["generated_audio"])
        back = Path(row["reference_audio"])
        source = Path(row["source_audio"])
        assert front.is_file() and back.is_file() and source.is_file()
        assert front != back and front != source and back != source
        with wave.open(str(source), "rb") as handle:
            source_frames = handle.getnframes()
        with wave.open(str(front), "rb") as handle:
            front_frames = handle.getnframes()
        with wave.open(str(back), "rb") as handle:
            back_frames = handle.getnframes()
        assert front_frames + back_frames == source_frames
        assert abs(front_frames - back_frames) == 1
        half = row["half_split"]
        assert half["front_frame_range"] == [0, front_frames]
        assert half["back_frame_range"] == [front_frames, source_frames]
        assert half["overlap_frames"] == 0
        assert half["gap_frames"] == 0
        assert row["evaluation_contract"]["run_asr"] is False

        canonical = EVAL.normalize_case(
            row,
            input_index=0,
            input_path=input_path,
            run_id="fixture",
            system_override=PREP.SYSTEM_ID,
            test_set_override=row["test_set_id"],
            legacy_asr_backend="qwen_asr",
            input_profile="official_seedtts_vc",
        )
        assert canonical["audio"]["generated"] == str(front)
        assert canonical["audio"]["reference"] == str(back)
        assert canonical["content_asr"] == {}

    with pytest.raises(FileExistsError, match="output directory already exists"):
        PREP.build_all(build_args(paths, output))


def test_prepare_rejects_parent_that_is_not_same_file_calibration(
    tmp_path: Path,
) -> None:
    paths = make_parent_fixture(tmp_path, bad_role=True)
    # Keep the parent audit internally consistent so the row-level role check
    # is the condition that fails.
    audit = json.loads(paths["audit"].read_text())
    audit["splits"]["en"]["output_jsonl_sha256"] = sha(paths["en"])
    paths["audit"].write_text(json.dumps(audit, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="not the registered same-file"):
        PREP.build_all(build_args(paths, tmp_path / "output"))


def test_runner_safe_plan_and_confirmation_gates(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "PROJECT_ROOT": str(tmp_path),
            "BATCH42_GT_HALF_SPLIT_TEST_MODE": "1",
            "PREP_ROOT": str(tmp_path / "prepared"),
            "OUTPUT_ROOT": str(tmp_path / "scores"),
            "RECORD_ROOT": str(tmp_path / "records"),
        }
    )
    plan = subprocess.run(
        ["bash", str(RUNNER)], env=env, text=True, capture_output=True, check=False
    )
    assert plan.returncode == 0, plan.stderr
    assert "ACTION=plan" in plan.stdout
    assert "ASR=disabled" in plan.stdout
    assert "no files, models, or GPUs were touched" in plan.stdout
    assert not (tmp_path / "scores").exists()
    assert not (tmp_path / "records").exists()

    env["ACTION"] = "run"
    denied = subprocess.run(
        ["bash", str(RUNNER)], env=env, text=True, capture_output=True, check=False
    )
    assert denied.returncode == 2
    assert "requires CONFIRM_GT_HALF_SPLIT_LOCAL=1" in denied.stderr


def test_runner_is_local_speaker_only_by_construction() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "NVIDIA GeForce RTX 4090" in source
    assert "--speaker-scorer all" in source
    assert "--asr-backend" not in source
    assert "create-job" not in source
    assert "qzcli" not in source.lower()
    assert "CONFIRM_GT_HALF_SPLIT_LOCAL" in source
    assert "CONFIRM_GT_HALF_SPLIT_SPEAKER_ONLY" in source
    assert "CONFIRM_GT_HALF_SPLIT_NO_ASR" in source

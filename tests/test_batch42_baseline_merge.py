from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/004089_merge_batch42_baseline_shards.py"
SPEC = importlib.util.spec_from_file_location("batch42_baseline_merge", SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot import {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 1600)


def row(root: Path, index: int, shard: int, *, status: str = "ok") -> dict:
    generated = root / f"generated-{index}.wav"
    if status == "ok":
        write_wav(generated)
    return {
        "schema_version": "moss_codecvc.baseline_vc_infer.v1",
        "record_type": "baseline_vc_inference",
        "system_id": "fixture",
        "test_set_id": "strict",
        "case_id": f"case-{index}",
        "case_uid": f"uid-{index}",
        "input_index": index,
        "status": status,
        "generated_audio": str(generated),
        "source_audio": str(root / f"source-{index}.wav"),
        "reference_audio": str(root / f"ref-{index}.wav"),
        "target_text": "text",
        "provenance": {"num_shards": 2, "shard_index": shard},
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
    )


class Batch42BaselineMergeTest(unittest.TestCase):
    def test_merges_complete_shards_and_emits_scorer_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard0, shard1 = root / "m0.jsonl", root / "m1.jsonl"
            write_jsonl(shard0, [row(root, 0, 0), row(root, 2, 0)])
            write_jsonl(shard1, [row(root, 1, 1), row(root, 3, 1)])
            status = MODULE.main(
                [
                    "--input", str(shard0), "--input", str(shard1),
                    "--merged-manifest", str(root / "merged.jsonl"),
                    "--successful-jsonl", str(root / "successful.jsonl"),
                    "--summary-json", str(root / "summary.json"),
                    "--expected-shards", "2", "--expected-cases", "4",
                    "--system-id", "fixture", "--test-set-id", "strict",
                    "--require-all-ok",
                ]
            )
            merged = (root / "merged.jsonl").read_text(encoding="utf-8").splitlines()
            successful = (root / "successful.jsonl").read_text(encoding="utf-8").splitlines()
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(status, 0)
        self.assertEqual(len(merged), 4)
        self.assertEqual(len(successful), 4)
        self.assertTrue(summary["all_ok"])

    def test_errors_remain_in_ledger_but_not_scorer_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard0, shard1 = root / "m0.jsonl", root / "m1.jsonl"
            write_jsonl(shard0, [row(root, 0, 0)])
            write_jsonl(shard1, [row(root, 1, 1, status="error")])
            status = MODULE.main(
                [
                    "--input", str(shard0), "--input", str(shard1),
                    "--merged-manifest", str(root / "merged.jsonl"),
                    "--successful-jsonl", str(root / "successful.jsonl"),
                    "--summary-json", str(root / "summary.json"),
                    "--expected-shards", "2", "--expected-cases", "2",
                    "--require-all-ok",
                ]
            )
            merged = (root / "merged.jsonl").read_text(encoding="utf-8").splitlines()
            successful = (root / "successful.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(status, 1)
        self.assertEqual(len(merged), 2)
        self.assertEqual(len(successful), 1)

    def test_duplicate_case_uid_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = row(root, 0, 0)
            second = row(root, 1, 1)
            second["case_uid"] = first["case_uid"]
            with self.assertRaisesRegex(ValueError, "duplicate case_uid"):
                MODULE.audit_rows(
                    [first, second],
                    expected_shards=2,
                    expected_cases=2,
                    system_id=None,
                    test_set_id=None,
                )

    def test_wrong_modulo_shard_assignment_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = row(root, 0, 1)
            good = row(root, 1, 0)
            with self.assertRaisesRegex(ValueError, "not assigned"):
                MODULE.audit_rows(
                    [bad, good],
                    expected_shards=2,
                    expected_cases=2,
                    system_id=None,
                    test_set_id=None,
                )


if __name__ == "__main__":
    unittest.main()

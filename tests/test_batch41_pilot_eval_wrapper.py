from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/004086_submit_batch41_b2_mixed_pilot_seedtts320_qz.sh"
SOURCE_VALIDATION = ROOT / "testset/validation/seedtts_vc_ver2_3_validation.jsonl"
CODE_ROOT = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
    "MOSS-CodecVC_snapshots/ver23_batch3436_eval_20260711_1092820"
)
SPEECHBRAIN_MODEL = Path(
    "/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/models/"
    "speechbrain/spkrec-ecapa-voxceleb"
)
TRAIN_RUN_NAME = "ver23_content_side_batch41_b2_mixed_3k_probe_20260711"
VALID_JOB_ID = "job-12345678-1234-1234-1234-123456789abc"


class Batch41PilotEvalWrapperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        self.record = self.root / "record"
        self.eval_root = self.root / "eval"
        self.qz_home = self.root / "qz-home"
        self.call_log = self.root / "qz-calls.log"

        validation = self.project / "testset/validation/seedtts_vc_ver2_3_validation.jsonl"
        validation.parent.mkdir(parents=True)
        shutil.copyfile(SOURCE_VALIDATION, validation)
        self.validation = validation

        checkpoint = (
            self.project
            / "outputs/lora_runs"
            / TRAIN_RUN_NAME
            / "step-3000"
        )
        checkpoint.mkdir(parents=True)
        for name in ("adapter_config.json", "timbre_memory_config.json"):
            (checkpoint / name).write_text("{}\n", encoding="utf-8")
        for name in ("adapter_model.safetensors", "README.md", "timbre_memory_adapter.pt"):
            (checkpoint / name).write_bytes(b"fixture\n")

        self.qzcli = self.root / "fake-qzcli.sh"
        self.qzcli.write_text(
            "#!/bin/sh\n"
            "printf 'called\\n' >> \"$QZ_CALL_LOG\"\n"
            "printf '%s\\n' \"${FAKE_QZ_OUTPUT:-}\"\n"
            "exit \"${FAKE_QZ_STATUS:-0}\"\n",
            encoding="utf-8",
        )
        self.qzcli.chmod(0o755)

        self.env = {
            **os.environ,
            "PROJECT_ROOT": str(self.project),
            "CODE_ROOT": str(CODE_ROOT),
            "PYTHON": sys.executable,
            "ASR_PYTHON": sys.executable,
            "QZCLI": str(self.qzcli),
            "QZCLI_HOME": str(self.qz_home),
            "QZ_CALL_LOG": str(self.call_log),
            "RECORD_ROOT": str(self.record),
            "EVAL_ROOT": str(self.eval_root),
            "GATE_BUILDER": str(ROOT / "scripts/004088_build_batch41_pilot_gate.py"),
            "SPEECHBRAIN_ECAPA_MODEL_SOURCE": str(SPEECHBRAIN_MODEL),
            "DRY_RUN": "0",
            "FORCE": "0",
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_wrapper(self, **overrides: str) -> subprocess.CompletedProcess[str]:
        env = {**self.env, **overrides}
        return subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

    def qz_call_count(self) -> int:
        if not self.call_log.exists():
            return 0
        return len(self.call_log.read_text(encoding="utf-8").splitlines())

    def test_manifest_realpath_is_hard_locked(self) -> None:
        alternate = self.root / "alternate.jsonl"
        shutil.copyfile(self.validation, alternate)
        result = self.run_wrapper(VALIDATION_JSONL=str(alternate))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must use the canonical validation manifest", result.stderr)
        self.assertEqual(self.qz_call_count(), 0)

    def test_manifest_sha256_is_hard_locked(self) -> None:
        with self.validation.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        result = self.run_wrapper()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SHA256 mismatch", result.stderr)
        self.assertEqual(self.qz_call_count(), 0)

    def test_mtts_compute_group_is_hard_locked(self) -> None:
        result = self.run_wrapper(COMPUTE_GROUP="forbidden-pool")
        self.assertEqual(result.returncode, 2)
        self.assertIn("only MTTS-3-2-0715 is allowed", result.stderr)
        self.assertEqual(self.qz_call_count(), 0)

    def test_existing_atomic_lock_blocks_before_qz(self) -> None:
        self.record.mkdir(parents=True)
        (self.record / ".live_submission_lock").mkdir()
        result = self.run_wrapper(FAKE_QZ_OUTPUT=VALID_JOB_ID)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("another live Batch-41 final320 submission", result.stderr)
        self.assertEqual(self.qz_call_count(), 0)

    def test_success_without_valid_job_id_is_rejected_and_unlocks(self) -> None:
        result = self.run_wrapper(FAKE_QZ_OUTPUT="create-job returned success")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no valid job ID was parsed", result.stderr)
        self.assertEqual(self.qz_call_count(), 1)
        self.assertFalse((self.record / "submitted_jobs.tsv").exists())
        self.assertFalse((self.record / ".live_submission_lock").exists())

    def test_valid_job_id_is_recorded_and_duplicate_is_fenced(self) -> None:
        first = self.run_wrapper(FAKE_QZ_OUTPUT=f"created {VALID_JOB_ID}")
        self.assertEqual(first.returncode, 0, first.stderr)
        submitted = self.record / "submitted_jobs.tsv"
        self.assertIn(VALID_JOB_ID, submitted.read_text(encoding="utf-8"))
        self.assertFalse((self.record / ".live_submission_lock").exists())
        self.assertEqual(self.qz_call_count(), 1)

        second = self.run_wrapper(FAKE_QZ_OUTPUT=f"created {VALID_JOB_ID}")
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("existing Batch-41 pilot gate/submission", second.stderr)
        self.assertEqual(self.qz_call_count(), 1)
        self.assertFalse((self.record / ".live_submission_lock").exists())


if __name__ == "__main__":
    unittest.main()

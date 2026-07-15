from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/004094_submit_batch42_pathx_strict_qz.sh"


class Batch42PathXStrictSubmitWrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def run_rejected_configuration(self, **overrides: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = os.environ.copy()
            env.update(
                {
                    "MODE": "smoke",
                    "DRY_RUN": "1",
                    "RUN_TAG": "unit_rejected",
                    "SMOKE_GATE_TAG": "unit_rejected",
                    "RECORD_ROOT": str(root / "record"),
                    # Smoke output is hard-bound to its registered project-root path,
                    # so do not override OUTPUT_ROOT here.
                    **overrides,
                }
            )
            completed = subprocess.run(
                ["bash", str(SCRIPT)],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse((root / "record" / "submitted_jobs.tsv").exists())
            self.assertFalse((root / "record" / "dry_run_jobs.tsv").exists())
            return completed

    def test_bash_syntax_and_embedded_python(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True, cwd=ROOT)
        lines = self.source.splitlines()
        blocks = []
        index = 0
        while index < len(lines):
            if "<<'PY'" not in lines[index]:
                index += 1
                continue
            end = index + 1
            while end < len(lines) and lines[end] != "PY":
                end += 1
            self.assertLess(end, len(lines), f"unclosed Python heredoc at line {index + 1}")
            source = "\n".join(lines[index + 1 : end]) + "\n"
            compile(source, f"{SCRIPT}:heredoc@{index + 2}", "exec")
            blocks.append(source)
            index = end + 1
        self.assertEqual(len(blocks), 7)

    def test_wrong_resource_contracts_are_rejected_before_any_qz_artifact(self) -> None:
        cases = (
            ({"WORKSPACE": "wrong-workspace"}, "workspace"),
            ({"PROJECT": "wrong-project"}, "project"),
            ({"COMPUTE_GROUP": "H200-3-2"}, "MTTS-3-2-0715"),
            ({"SPEC": "wrong-spec"}, "registered MTTS spec"),
            ({"QZCLI_GPU_TYPE_OVERRIDE": "NVIDIA_A100"}, "only H200"),
            ({"INSTANCES": "2"}, "one instance"),
            ({"GPUS_PER_INSTANCE": "4"}, "8 GPUs"),
            ({"NUM_SHARDS": "4"}, "8 shards"),
            ({"PROJECT_ROOT": str(ROOT.parent)}, "PROJECT_ROOT is hard-locked"),
        )
        for overrides, expected_message in cases:
            with self.subTest(overrides=overrides):
                completed = self.run_rejected_configuration(**overrides)
                self.assertIn(expected_message, completed.stderr)

    def test_qzcli_wrapper_home_and_proxy_cleanup_are_hard_bound(self) -> None:
        self.assertIn('ALLOWED_QZCLI="/inspire/qb-ilm2/project/', self.source)
        self.assertIn("pair_construction/scripts/qzcli_with_deps.sh", self.source)
        self.assertIn(
            'ALLOWED_QZCLI_HOME="/inspire/ssd/project/embodied-multimodality/public/xyzhang/.codex/qzcli_home"',
            self.source,
        )
        self.assertEqual(
            self.source.count(
                "env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy"
            ),
            2,
        )

    def test_snapshot_is_verified_inside_job_before_forward(self) -> None:
        job_body = self.source.split("run_job_entrypoint() {", 1)[1].split(
            "write_submission_plan() {", 1
        )[0]
        checksum = job_body.index('sha256sum -c "$RECORD_ROOT/sha256sums.txt"')
        gpu_audit = job_body.index("audit_allocated_gpus")
        worker_launch = job_body.index("run_all_workers")
        self.assertLess(checksum, gpu_audit)
        self.assertLess(gpu_audit, worker_launch)

    def test_full_job_requires_smoke_and_real_node_preflight(self) -> None:
        job_body = self.source.split("run_job_entrypoint() {", 1)[1].split(
            "write_submission_plan() {", 1
        )[0]
        smoke_gate = job_body.index("validate_smoke_marker")
        actual_preflight = job_body.index("run_actual_node_preflight")
        workers = job_body.index("run_all_workers")
        self.assertLess(smoke_gate, actual_preflight)
        self.assertLess(actual_preflight, workers)
        self.assertIn("--max-cases 1", self.source)
        self.assertIn("actual node preflight requires one newly generated ok row", self.source)

    def test_smoke_marker_is_audio_decoded_and_protocol_fingerprinted(self) -> None:
        self.assertIn("soundfile as sf", self.source)
        self.assertIn("wave.open", self.source)
        self.assertIn('"protocol_contract": protocol_contract', self.source)
        self.assertIn('"protocol_fingerprint_sha256": protocol_fingerprint', self.source)
        self.assertIn("smoke protocol fingerprint mismatch", self.source)
        self.assertIn('"ref_audio_cfg_scale": 1.0', self.source)

    def test_live_submission_fences_and_uuid_parser_are_strict(self) -> None:
        self.assertIn('SUBMISSION_LOCK="$RECORD_ROOT/.live_submission_lock"', self.source)
        self.assertIn('mkdir "$SUBMISSION_LOCK"', self.source)
        self.assertRegex(
            self.source,
            re.compile(
                r'if \[ "\$MODE" = "full" \]; then\n\s+validate_smoke_marker',
                re.MULTILINE,
            ),
        )
        self.assertIn("mapfile -t job_ids", self.source)
        self.assertIn("sort -u", self.source)
        self.assertIn('expected exactly one unique complete UUID job ID', self.source)
        self.assertIn('mv "$submitted_tmp" "$RECORD_ROOT/submitted_jobs.tsv"', self.source)

    def test_completion_marker_follows_all_workers_merges_and_schema_checks(self) -> None:
        job_body = self.source.split("run_job_entrypoint() {", 1)[1].split(
            "write_submission_plan() {", 1
        )[0]
        ordered = [
            job_body.index("run_all_workers"),
            job_body.index("merge_language en"),
            job_body.index("merge_language zh"),
            job_body.index("schema_check_language en"),
            job_body.index("schema_check_language zh"),
            job_body.index("write_full_marker"),
        ]
        self.assertEqual(ordered, sorted(ordered))
        self.assertIn("set -euo pipefail", self.source)
        self.assertIn('rm -f "$FINAL_MARKER"', job_body)


if __name__ == "__main__":
    unittest.main()

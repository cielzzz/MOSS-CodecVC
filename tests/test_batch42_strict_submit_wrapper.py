from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/004090_submit_batch42_baseline_strict_qz.sh"


class Batch42StrictSubmitWrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_job_preflight_is_actual_and_full_workers_use_full_mode(self) -> None:
        self.assertIn('preflight_mode="preflight_actual"', self.source)
        self.assertIn('preflight_mode="preflight_dry"', self.source)
        self.assertIn(
            'preflight_run_flags="--no-continue-on-error --fail-if-any-error"',
            self.source,
        )
        self.assertRegex(
            self.source,
            re.compile(
                r'run_inference_case \\\n'
                r'\s+"\$shard_index".*?"\$shard_index" full',
                re.DOTALL,
            ),
        )
        self.assertNotIn('"$shard_index" live', self.source)

    def test_smoke_only_contract_is_propagated_and_fenced(self) -> None:
        self.assertIn(
            'SMOKE_ONLY="${BATCH42_BASELINE_SMOKE_ONLY:-0}"', self.source
        )
        self.assertIn('SMOKE_MARKER="$OUTPUT_ROOT/SMOKE_COMPLETED.json"', self.source)
        self.assertIn("write_smoke_completion_marker", self.source)
        self.assertIn("moss_codecvc.batch42_baseline_strict_smoke_completion.v1", self.source)
        self.assertIn("BATCH42_BASELINE_SMOKE_ONLY=$SMOKE_ONLY", self.source)
        self.assertIn('[ -s "$SMOKE_MARKER" ]', self.source)

    def test_optional_smoke_gate_is_sha_locked_and_provenance_preserved(self) -> None:
        self.assertIn('SMOKE_GATE_JSON="${SMOKE_GATE_JSON:-}"', self.source)
        self.assertIn('SMOKE_GATE_SHA256="${SMOKE_GATE_SHA256:-}"', self.source)
        self.assertIn("prepare_smoke_gate", self.source)
        self.assertIn("audit_smoke_gate", self.source)
        self.assertIn(
            'moss_codecvc.batch42_baseline_strict_smoke_completion.v1',
            self.source,
        )
        self.assertIn('marker.get("public_system_id") != expected_system', self.source)
        self.assertIn('generated.stat().st_size <= 1024', self.source)
        self.assertIn('SMOKE_GATE_JSON=$SMOKE_GATE_JSON', self.source)
        self.assertIn('SMOKE_GATE_SHA256=$SMOKE_GATE_SHA256', self.source)
        self.assertIn('"smoke_gate": smoke_gate', self.source)

    def test_openvoice_and_cosy_official_runtime_paths_are_explicit(self) -> None:
        self.assertGreaterEqual(
            self.source.count("--openvoice-segmentation upstream_silero_vad"), 2
        )
        self.assertGreaterEqual(
            self.source.count("--cosy-speech-tokenizer-provider cuda"), 2
        )
        self.assertIn('config.get("upstream_silero_vad") is not True', self.source)
        self.assertIn(
            'config.get("speech_tokenizer_onnx_provider") != "cuda"', self.source
        )
        self.assertIn('TORCH_HOME="$TORCH_HOME_PATH"', self.source)
        self.assertEqual(
            self.source.count("--openvoice-silero-short-retry-split-seconds 2.0"),
            2,
        )
        self.assertIn('COSY_ORT_CAPI="$COSY_SITE/onnxruntime/capi"', self.source)
        self.assertIn('VC_PIP_NVIDIA_LD=', self.source)

    def test_seed_official_path_keeps_cudnn_enabled_in_all_paths(self) -> None:
        self.assertEqual(self.source.count("--no-seed-disable-cudnn"), 4)
        self.assertIn(
            'get("disable_cudnn") is not False', self.source
        )


if __name__ == "__main__":
    unittest.main()

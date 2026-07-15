from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/ver3_1/run_step1_zq_extract.sh"
SUBMITTER = ROOT / "scripts/ver3_1/submit_step1_zq_extract_qz.sh"
ALLOWED_POOL = "lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"
ALLOWED_SPEC = "67b10bc6-78b0-41a3-aaf4-358eeeb99009"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_wrappers_are_valid_bash() -> None:
    for path in (RUNNER, SUBMITTER):
        completed = subprocess.run(
            ["bash", "-n", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr


def test_runner_hard_locks_authoritative_v1_inputs_and_totals() -> None:
    script = _text(RUNNER)
    assert "ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/no_text.train.jsonl" in script
    assert "ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709/text.train.jsonl" in script
    assert "c4b061f0a968e73710dc86d81478483a9195e8a053f510f09be7952d60c3d279" in script
    assert "c6632888d08e79382001909a65951d6ce7bab80d7fb585cf7729e0a9188a9a80" in script
    assert "NO_TEXT_ROWS=310420" in script
    assert "NO_TEXT_FRAMES=31089741" in script
    assert "TEXT_ROWS=32419" in script
    assert "TEXT_FRAMES=4008719" in script
    assert "EXPECTED_UTTERANCES=342839" in script
    assert "EXPECTED_FRAMES=35098460" in script
    assert 'OUTPUT_ROOT="$ROOT/prepared/zq_targets_v1"' in script
    assert "RUN_CONTRACT_PATH=\"$OUTPUT_PARENT/zq_targets_v1.RUN_CONTRACT.json\"" in script
    assert 'EXTRACTOR_CONTRACT_PATH="$OUTPUT_ROOT/CONTRACT.json"' in script
    assert 'LOG_ROOT="$OUTPUT_PARENT/zq_targets_v1_logs/$RUN_ID"' in script


def test_runner_uses_eight_real_one_gpu_workers_with_full_contract() -> None:
    script = _text(RUNNER)
    assert "NUM_SHARDS=8" in script
    assert "for shard in 0 1 2 3 4 5 6 7" in script
    assert 'CUDA_VISIBLE_DEVICES="$shard"' in script
    assert "--device cuda:0" in script
    assert script.count('--input "no_text=$NO_TEXT_MANIFEST"') == 1
    assert script.count('--input "text=$TEXT_MANIFEST"') == 1
    assert '--codes-source "$CODES_SOURCE"' in script
    assert "CODES_SOURCE=manifest" in script
    assert "NUM_QUANTIZERS=32" in script
    assert "LATENT_DIM=768" in script
    assert "OUTPUT_DTYPE=float32" in script
    assert "CODEC_DTYPE=float32" in script
    assert 'ZQ_BATCH_SIZE="${ZQ_BATCH_SIZE:-32}"' in script
    assert "finalize" in script
    assert "--num-shards 8" in script
    assert "--expected-total-utterances" in script
    assert "--expected-total-frames" in script


def test_runner_guards_environment_keepalive_heartbeat_and_completion() -> None:
    script = _text(RUNNER)
    assert "torch.cuda.device_count() != 8" in script
    assert '"H200" not in name.upper()' in script
    assert "HF_HUB_OFFLINE=1" in script
    assert "TRANSFORMERS_OFFLINE=1" in script
    assert "required_sizes" in script
    assert "model.safetensors.index.json" in script
    assert "for gpu in 0 1 2 3 4 5 6 7" in script
    assert "KEEPALIVE_PIDS" in script
    assert "trap cleanup EXIT" in script
    assert "ver3.1-step1-heartbeat" in script
    assert "RUN_CONTRACT.json" in script
    assert "mixed extraction contract" in script
    assert "already completed; refusing a duplicate" in script
    assert "COMPLETED.json" in script
    assert "VERIFIED_COMPLETED.json" in script
    assert '"total_utterances": 342839' in script
    assert '"total_frames": 35098460' in script
    for guard in (
        "EXPECTED_RUNNER_SHA256",
        "EXPECTED_EXTRACTOR_SHA256",
        "EXPECTED_CONFIG_SHA256",
        "EXPECTED_MOSS_CODEC_SHA256",
    ):
        assert guard in script


def test_submitter_is_dry_guarded_snapshotted_and_ledgered() -> None:
    script = _text(SUBMITTER)
    assert 'DRY_RUN="${DRY_RUN:-1}"' in script
    assert "ALLOW_CODECVC_VER3_1_STEP1_SUBMIT" in script
    assert '[[ "$JOB_NAME" == codecVC-* ]]' in script
    assert '[[ "$BATCH_ID" == codecVC-* ]]' in script
    assert 'QZ_RECORD_ROOT must live under the shared project trainset/qz_jobs root' in script
    assert "record_snapshot" in script
    assert "SHA256SUMS" in script
    assert "SOURCE_GIT_SHA" in script
    assert "live_submission.lock" in script
    assert "GLOBAL_LEDGER" in script
    assert "submitted_jobs.tsv" in script
    assert "submission_plan.tsv" in script
    assert '"$QZCLI" create-job' in script
    assert script.count('"$QZCLI" login') == 1
    assert "one retry" in script


def test_submitter_allows_only_mtts_one_node_contract() -> None:
    script = _text(SUBMITTER)
    assert ALLOWED_POOL in script
    assert ALLOWED_SPEC in script
    pools = set(re.findall(r"lcg-[0-9a-f-]{36}", script))
    assert pools == {ALLOWED_POOL}
    specs = set(re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", script))
    assert ALLOWED_SPEC in specs
    assert 'INSTANCES="${INSTANCES:-1}"' in script
    assert '[ "$INSTANCES" = "1" ]' in script
    assert "--instances 1" in script
    assert "gpus\": 8" in script
    assert "MTTS-3-2-0715" in script
    assert "H200" in script
    assert "qzcli_with_deps.sh" in script

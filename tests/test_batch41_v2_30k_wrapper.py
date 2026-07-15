from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/002053_submit_ver23_batch41_pathx_v2_30k_qz.sh"


def test_v2_no_text_identity_is_registered() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'NO_TEXT_EXPECTED_BYTES="23967514378"' in text
    assert (
        'NO_TEXT_EXPECTED_SHA256="de2e6ca854c8054445739ea831641b0f'
        '138893f2ec9ba8dbfd7b0a5760dda5eb"'
    ) in text
    assert 'NO_TEXT_TRAIN_JSONL="$V2_DIR/no_text.v2.train.jsonl"' in text


def test_live_submission_cannot_skip_full_hash() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'if [ "$DRY_RUN" = "0" ]' in text
    assert 'VERIFY_V2_FULL_SHA256="${VERIFY_V2_FULL_SHA256:-1}"' in text
    assert (
        'live Batch-41 30k submission requires full v2 SHA256 verification'
        in text
    )
    assert 'hashlib.file_digest(handle, "sha256")' in text


def test_registered_training_mix_remains_repeat_three() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "TEXT_REPEAT=3" in text
    assert (
        'TRAIN_JSONL_SPEC="$NO_TEXT_TRAIN_JSONL::repeat=1,'
        '$TEXT_TRAIN_JSONL::repeat=$TEXT_REPEAT"'
    ) in text

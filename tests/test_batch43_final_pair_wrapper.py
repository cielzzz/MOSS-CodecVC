from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/002054_submit_ver23_batch43_final_pair_qz.sh"


def _text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_pair_uses_the_same_registered_v2_inputs_and_hashes() -> None:
    text = _text()
    assert 'NO_TEXT_TRAIN_JSONL="$V2_DIR/no_text.v2.train.jsonl"' in text
    assert 'TEXT_TRAIN_JSONL="$V2_DIR/text.train.jsonl"' in text
    assert 'NO_TEXT_EXPECTED_ROWS="295632"' in text
    assert 'TEXT_EXPECTED_ROWS="32419"' in text
    assert (
        'NO_TEXT_EXPECTED_SHA256="de2e6ca854c8054445739ea831641b0f'
        '138893f2ec9ba8dbfd7b0a5760dda5eb"'
    ) in text
    assert (
        'TEXT_EXPECTED_SHA256="b30d07b16b8f86df5b725e9d07f7d849'
        '9cb00d6b4662f5b28b65027aa6295993"'
    ) in text


def test_live_pair_requires_full_hashing_and_has_duplicate_fence() -> None:
    text = _text()
    assert 'VERIFY_FULL_SHA256="${VERIFY_FULL_SHA256:-1}"' in text
    assert "live Batch-43 pair submission requires full SHA256 verification" in text
    assert 'hashlib.file_digest(handle, "sha256")' in text
    assert "refuse_duplicate_live_submission" in text
    assert "existing live submission ledger" in text
    assert "acquire_atomic_live_lock" in text
    assert 'mkdir "$LIVE_LOCK_DIR"' in text
    assert "persistent lock; never auto-remove" in text


def test_beta_override_accepts_only_the_exact_failed_r3_pilot() -> None:
    text = _text()
    assert (
        'BATCH43_BETA_OVERRIDE_ID="batch43_beta_accept_exact_batch41_r3_edge_fail_20260712"'
        in text
    )
    assert (
        'PILOT_GATE_EXPECTED_SHA256="0256a042c0a7f7486a3aaab350fce9219'
        'aa91dbb04c56730c5e11a68fde95880"'
    ) in text
    assert '"decision": "fail"' in text
    assert '"checkpoint_step": 3000' in text
    assert '"text_repeat": 3' in text
    assert '"text_en_src_fail": 0.1875' in text
    assert '"text_cer": 0.06924000057563053' in text
    assert "not a reusable gate bypass" in text
    assert "audit_batch43_beta_override" in text


def test_resource_and_frozen_code_fences_are_hard_locked() -> None:
    text = _text()
    assert 'ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"' in text
    assert 'ALLOWED_SPEC="67b10bc6-78b0-41a3-aaf4-358eeeb99009"' in text
    assert 'ALLOWED_GPU_TYPE="NVIDIA_H200_SXM_141G"' in text
    assert 'ALLOWED_INSTANCES="1"' in text
    assert "ver23_batch3436_20260710_1092820" in text
    assert "frozen file drift" in text
    assert 'QZCLI="/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/qzcli_with_deps.sh"' in text
    assert 'QZCLI_HOME="/inspire/ssd/project/embodied-multimodality/public/xyzhang/.codex/qzcli_home"' in text
    assert 'HOME="$QZCLI_HOME"' in text
    assert 'QZCLI="$QZCLI"' in text


def test_base_architecture_and_all_layer_semantics_are_audited() -> None:
    text = _text()
    assert '"hidden_size": 4096' in text
    assert '"num_attention_heads": 32' in text
    assert '"num_hidden_layers": 36' in text
    assert '"content_cross_attn_layers_all_resolves_to": 36' in text
    assert "all (36 transformer layers; 32 is attention-head/n_vq count)" in text


def test_only_scientific_delta_is_text_repeat() -> None:
    text = _text()
    assert 'run_arm 3 1' in text
    assert 'run_arm 5 1' in text
    assert '"only_intended_scientific_delta": {' in text
    assert '"TEXT_REPEAT": {"r3": 3, "r5": 5}' in text
    assert "unexpected core config deltas" in text
    assert "generated runners differ beyond arm name/output/train repeat" in text
    assert 'ARM_TRAIN_JSONL_SPEC="$NO_TEXT_TRAIN_JSONL::repeat=1,$TEXT_TRAIN_JSONL::repeat=$ARM_TEXT_REPEAT"' in text


def test_requested_30k_recipe_is_explicit() -> None:
    text = _text()
    for expected in (
        'MAX_TRAIN_STEPS="30000"',
        'SAVE_STEPS="2000"',
        'EVAL_STEPS="2000"',
        'LEARNING_RATE="1e-5"',
        'GRADIENT_ACCUMULATION_STEPS="8"',
        'GPU_COUNT="8"',
        'CONTENT_CROSS_ATTN_LAYERS="all"',
        'CONTENT_CROSS_ATTN_GATE_INIT="-0.5"',
        'CONTENT_CROSS_ATTN_OUTPUT_SCALE="0.3"',
        'GUIDED_ATTN_LOSS_WEIGHT="0.05"',
        'PHONEME_CLASSIFIER_LOSS_WEIGHT="0.02"',
        'CONTENT_CTC_WEIGHT="0.0"',
        'RESUME_ADAPTER_PATH=""',
        'FREEZE_LORA="0"',
        'FREEZE_ROLE_ROUTING="0"',
        'FREEZE_TIMBRE_ADAPTER="0"',
    ):
        assert expected in text


def test_safe_default_builds_both_plans_before_live_submission() -> None:
    text = _text()
    assert 'DRY_RUN="${DRY_RUN:-1}"' in text
    dry_r3 = text.index("run_arm 3 1")
    dry_r5 = text.index("run_arm 5 1")
    pair_audit = text.index("audit_pair_config", dry_r5)
    live_r3 = text.index("run_arm 3 0")
    live_r5 = text.index("run_arm 5 0")
    assert dry_r3 < dry_r5 < pair_audit < live_r3 < live_r5


def test_live_submission_requires_one_complete_job_uuid_per_arm() -> None:
    text = _text()
    assert "verify_arm_live_submission" in text
    assert "expected exactly one complete QZ job UUID" in text
    assert '"$ARM_RECORD_ROOT/arm_submission.tsv"' in text
    assert 'verify_arm_live_submission 3' in text
    assert 'verify_arm_live_submission 5' in text

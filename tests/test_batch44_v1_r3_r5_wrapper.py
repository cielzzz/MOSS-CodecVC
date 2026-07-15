from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/002056_submit_ver23_batch44_v1_r3_r5_30k_qz.sh"


def source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_exact_arm_names_and_20260713_identity() -> None:
    text = source()
    assert 'STAMP="20260713"' in text
    assert 'R3_JOB_NAME="ver2_9_5_final_r3_v1_30k"' in text
    assert 'R5_JOB_NAME="ver2_9_5_final_r5_v1_30k"' in text
    assert 'PAIR_ID="ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_${STAMP}"' in text


def test_v1_inputs_are_exactly_hash_and_size_locked() -> None:
    text = source()
    assert 'V1_DIR="$PROJECT_ROOT/trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709"' in text
    assert 'NO_TEXT_EXPECTED_ROWS="310420"' in text
    assert 'NO_TEXT_EXPECTED_BYTES="18048211813"' in text
    assert 'NO_TEXT_EXPECTED_SHA256="c4b061f0a968e73710dc86d81478483a9195e8a053f510f09be7952d60c3d279"' in text
    assert 'TEXT_EXPECTED_ROWS="32419"' in text
    assert 'TEXT_EXPECTED_BYTES="2196087856"' in text
    assert 'TEXT_EXPECTED_SHA256="c6632888d08e79382001909a65951d6ce7bab80d7fb585cf7729e0a9188a9a80"' in text
    assert "input_identity.full_sha256.json" in text


def test_pair_only_changes_text_repeat() -> None:
    text = source()
    assert 'ARM_TRAIN_JSONL_SPEC="$NO_TEXT_TRAIN_JSONL::repeat=1,$TEXT_TRAIN_JSONL::repeat=$ARM_TEXT_REPEAT"' in text
    assert '"only_intended_scientific_delta_within_batch44": {"TEXT_REPEAT": {"r3": 3, "r5": 5}}' in text
    assert 'pair_allowed = sorted(["BATCH_ID", "JOB_NAME_PREFIX", "OUT_DIR", "TEXT_REPEAT", "TRAIN_JSONL_SPEC"])' in text
    assert "v1 pair normalized core configs differ" in text


def test_each_arm_normalizes_to_corresponding_batch43_recipe() -> None:
    text = source()
    assert 'B43_R3_CORE_SHA256="78525edc2e039e3f2c68dd845aa716966b4c11c560697a6d126a9ec12d17724c"' in text
    assert 'B43_R5_CORE_SHA256="d863f7579dfab905e99f3a0b9980abe310cc51c3a4aadc0292ce5ca6f4ebba9f"' in text
    assert 'B43_R3_RUNNER_SHA256="ed8aab069163bf3f6de21b23cbe3f4a4babb5dae3362132b9e69bb83491cf17e"' in text
    assert 'B43_R5_RUNNER_SHA256="87911d4c63035b3fffb945d50a64cd255acff72392b783a7ae1256830aee56c6"' in text
    assert "runner differs from corresponding Batch-43 runner beyond v1 inputs/job/output" in text
    assert 'NUM_EPOCHS="6"' in text
    assert 'MAX_TRAIN_STEPS="30000"' in text
    assert 'SAVE_STEPS="2000"' in text
    assert 'EVAL_STEPS="2000"' in text


def test_live_is_double_gated_atomic_and_duplicate_fenced() -> None:
    text = source()
    assert 'LIVE="${LIVE:-0}"' in text
    assert 'DRY_RUN="${DRY_RUN:-1}"' in text
    assert "DRY_RUN=0 alone is insufficient" in text
    assert "refuse_duplicate_live" in text
    assert ".live_pair_submit.lock" in text
    assert "run_arm 3 0" in text
    assert "verify_live_arm 3" in text
    assert "run_arm 5 0" in text
    assert "verify_live_arm 5" in text
    assert "build_pair_ledger" in text


def test_qz_payloads_are_exact_live_command_and_mtts_8h200() -> None:
    text = source()
    assert 'command="sh $runner"' in text
    assert 'ALLOWED_COMPUTE_GROUP="lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"' in text
    assert 'ALLOWED_SPEC="67b10bc6-78b0-41a3-aaf4-358eeeb99009"' in text
    assert 'ALLOWED_INSTANCES="1"' in text
    assert 'ALLOWED_GPU_TYPE="NVIDIA_H200_SXM_141G"' in text
    assert "qz_payload_dry_run 3" in text
    assert "qz_payload_dry_run 5" in text
    assert 'cfg.get("instance_count") != 1 or cfg.get("gpu_count") != 8' in text


def test_evaluation_contract_registers_same_step_pair_monitoring() -> None:
    text = source()
    assert '"first_mandatory_full320_step": 10000' in text
    assert '"full320_steps_if_healthy": [10000, 20000, 30000]' in text
    assert '"status": "registered_not_run"' in text

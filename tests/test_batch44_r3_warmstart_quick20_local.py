from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/004123_run_batch44_r3_warmstart_quick20_local.sh"
HELPER_PATH = ROOT / "scripts/batch44_r3_warmstart_quick20_completion.py"
VALIDATOR_PATH = ROOT / "scripts/batch44_r3_warmstart_quick20_validator.py"
JOB_ID = "job-165f3b1d-8c7d-47d8-8882-bdb86ef642ab"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = load(HELPER_PATH, "batch44_r3_warmstart_helper_test")
validator = load(VALIDATOR_PATH, "batch44_r3_warmstart_validator_test")


def write(path: Path, text: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def checkpoint_config() -> dict[str, object]:
    return {
        "content_cross_attn_enabled": True,
        "content_cross_attn_layers": "all",
        "content_cross_attn_feature_dim": 768,
        "content_cross_attn_gate_init": -0.5,
        "content_cross_attn_output_scale": 0.3,
        "content_encoder_layers": 2,
        "guided_attn_loss_weight": 0.05,
        "guided_attn_warmup_steps": 0,
        "phoneme_classifier_loss_weight": 0.02,
        "content_ctc_weight": 0.0,
        "progress_loss_weight": 0.1,
        "stop_loss_weight": 0.2,
        "target_front_ce_weight": 4.0,
        "target_front_ce_seconds": 0.75,
        "use_role_routing": True,
        "num_memory_tokens": 0,
        "timbre_side_only": False,
        "source_semantic_memory_enabled": False,
        "speaker_side_pathway_enabled": False,
        "speaker_cross_attn_enabled": False,
    }


def make_checkpoint(path: Path, *, large: bool = True) -> Path:
    path.mkdir(parents=True)
    size = 1_000_001 if large else 1
    (path / "adapter_model.safetensors").write_bytes(b"a" * size)
    (path / "timbre_memory_adapter.pt").write_bytes(b"b" * size)
    write(path / "README.md", "checkpoint\n")
    write(path / "adapter_config.json", "{}\n")
    write(
        path / "timbre_memory_config.json",
        json.dumps(checkpoint_config(), sort_keys=True) + "\n",
    )
    return path


def make_fixture(tmp_path: Path) -> dict[str, Path | str]:
    project = tmp_path / "project"
    warm_record = project / "trainset/qz_jobs/warm"
    output_dir = project / "outputs/lora_runs/continuation"
    checkpoint = make_checkpoint(output_dir / "step-2000")
    source = make_checkpoint(project / "outputs/lora_runs/base/step-10000", large=False)
    source_files = {}
    for name in helper.CHECKPOINT_FILES:
        item = source / name
        source_files[name] = {
            "path": str(item.resolve()),
            "bytes": item.stat().st_size,
            "sha256": helper.sha256_file(item),
        }
    contract = warm_record / "warm_start_contract.json"
    payload = {
        "schema": helper.WARM_START_SCHEMA,
        "status": "submitted",
        "job_id": JOB_ID,
        "job_name": "continuation",
        "output_dir": str(output_dir.resolve()),
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_files": source_files,
        "source_effective_step": 10000,
        "effective_step_offset": 10000,
        "continuation_local_target_step": 20000,
        "effective_target_step": 30000,
        "resume_semantics": "weights_only_warm_start_not_exact_resume",
    }
    write(contract, json.dumps(payload, sort_keys=True) + "\n")
    write(
        warm_record / "submitted_jobs.tsv",
        "job_name\tjob_id\tcompute_group\trunner\tout_dir\n"
        f"continuation\t{JOB_ID}\t{helper.ALLOWED_COMPUTE_GROUP}\t"
        f"{warm_record / 'run_train_entrypoint.sh'}\t{output_dir.resolve()}\n",
    )
    core = {
        "OUT_DIR": str(output_dir.resolve()),
        "TEXT_REPEAT": "3",
        "MAX_TRAIN_STEPS": "20000",
        "SAVE_STEPS": "2000",
        "EVAL_STEPS": "2000",
        "LEARNING_RATE": "1e-5",
        "LR_SCHEDULER_TYPE": "constant_with_warmup",
        "WARMUP_RATIO": "0.0",
        "GUIDED_ATTN_WARMUP_STEPS": "0",
        "GUIDED_ATTN_LOSS_WEIGHT": "0.05",
        "PHONEME_CLASSIFIER_LOSS_WEIGHT": "0.02",
        "CONTENT_CTC_WEIGHT": "0.0",
        "ENABLE_CONTENT_CROSS_ATTN": "1",
        "CONTENT_CROSS_ATTN_LAYERS": "all",
    }
    write(warm_record / "train_args_dry_run_core.json", json.dumps(core) + "\n")
    write(
        warm_record / "run_train_entrypoint.sh",
        f"RESUME_ADAPTER_PATH={source.resolve()}\nOUT_DIR={output_dir.resolve()}\n"
        "--resume-adapter-path $RESUME_ADAPTER_PATH\n",
    )
    write(warm_record / "generated_config_audit.json", "{}\n")
    write(warm_record / "qz_payload.json", "{}\n")

    record = project / "trainset/local_jobs/quick12000"
    output_root = project / "testset/outputs/quick"
    for mode in ("no_text", "text"):
        identity = helper.run_id(12000, mode)
        out = output_root / identity
        case_ids = [f"{mode}-{index}" for index in range(20)]
        cells = ["en_src_x" if mode == "text" and index < 12 else "other" for index in range(20)]
        write(out / f"{identity}.summary.json", json.dumps({"overall": {"n": 20, "keep": 18, "cer": 0.1}}))
        out.mkdir(parents=True, exist_ok=True)
        with (out / f"{identity}.speaker_sim.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["case_id", "status", "sim_gen_ref", "sim_gen_source"])
            writer.writeheader()
            for case_id in case_ids:
                writer.writerow({"case_id": case_id, "status": "ok", "sim_gen_ref": 0.45, "sim_gen_source": 0.40})
        rows = [
            {"case_id": case_id, "mode": mode, "cell": cell, "content_keep": True}
            for case_id, cell in zip(case_ids, cells)
        ]
        write(out / f"{identity}.asr_eval.jsonl", "".join(json.dumps(row) + "\n" for row in rows))
        write(
            out / f"{identity}.ref_content_similarity_summary.json",
            json.dumps({"overall": {"ref_content_lcs_f1_mean": 0.05}}),
        )

    runner = write(record / "004123_run_batch44_r3_warmstart_quick20_local.frozen.sh")
    common = write(record / "004110_batch44_quick20_common.frozen.sh")
    completion_helper = write(record / "batch44_r3_warmstart_quick20_completion.frozen.py")
    completion_validator = write(record / "batch44_r3_warmstart_quick20_validator.frozen.py")
    runtime = record / "LOCAL_RUNTIME.json"
    checkpoint_files = {name: helper.artifact(checkpoint / name) for name in helper.CHECKPOINT_FILES}
    runtime_payload = {
        "schema": helper.RUNTIME_SCHEMA,
        "status": "started",
        "backend": "local",
        "hostname": "xyzhang-dev--test",
        "base_effective_step": 10000,
        "effective_step": 12000,
        "continuation_local_step": 2000,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_files": checkpoint_files,
        "checkpoint_manifest_sha256": helper.checkpoint_manifest_sha256(checkpoint_files),
        "train_job_id": JOB_ID,
        "warm_start_contract": helper.artifact(contract),
        "gpu_count": 2,
        "gpu_indices": [0, 1],
        "gpu_model": helper.GPU_MODEL,
        "gpus": [],
        "scheduling": "test",
        "runner": helper.artifact(runner),
        "common_library": helper.artifact(common),
        "completion_helper": helper.artifact(completion_helper),
        "validator": helper.artifact(completion_validator),
    }
    write(runtime, json.dumps(runtime_payload, sort_keys=True) + "\n")
    fixed = [write(project / f"fixed-{index}.jsonl") for index in range(3)]
    return {
        "project": project,
        "record": record,
        "output_root": output_root,
        "checkpoint": checkpoint,
        "contract": contract,
        "runner": runner,
        "common": common,
        "helper": completion_helper,
        "validator": completion_validator,
        "runtime": runtime,
        "no_text": fixed[0],
        "text_source": fixed[1],
        "text20": fixed[2],
    }


def test_step_mapping_is_dual_identity() -> None:
    helper.validate_step_mapping(12000, 2000)
    helper.validate_step_mapping(30000, 20000)
    with pytest.raises(ValueError, match="mapping drift"):
        helper.validate_step_mapping(12000, 4000)


def test_runner_interface_and_local_only_contract() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for name in (
        "EFFECTIVE_STEP", "CONTINUATION_LOCAL_STEP", "CHECKPOINT", "TRAIN_JOB_ID",
        "WARM_START_CONTRACT", "RECORD_ROOT", "OUTPUT_ROOT", "DRY_RUN", "CONFIRM_RUN",
    ):
        assert name in source
    assert ".local_quick20.lock" in source
    assert "004123_run_batch44_r3_warmstart_quick20_local.frozen.sh" in source
    assert "create-job" not in source
    assert subprocess.run(["bash", "-n", str(RUNNER)], check=False).returncode == 0


def test_finalize_and_validator_reject_marker_mutation(tmp_path: Path) -> None:
    item = make_fixture(tmp_path)
    helper.collect_metrics(
        record_root=item["record"], output_root=item["output_root"],
        effective_step=12000, continuation_local_step=2000,
        checkpoint=item["checkpoint"], train_job_id=JOB_ID,
        warm_start_contract=item["contract"],
    )
    helper.finalize_completion(
        record_root=item["record"], output_root=item["output_root"],
        project_root=item["project"], code_root=item["project"],
        effective_step=12000, continuation_local_step=2000,
        checkpoint=item["checkpoint"], train_job_id=JOB_ID,
        warm_start_contract=item["contract"],
        no_text20=item["no_text"], no_text20_sha256=helper.sha256_file(item["no_text"]),
        text_source=item["text_source"], text_source_sha256=helper.sha256_file(item["text_source"]),
        text20=item["text20"], text20_sha256=helper.sha256_file(item["text20"]),
        runner=item["runner"], common_library=item["common"],
        completion_helper=item["helper"], validator=item["validator"],
        runtime_manifest=item["runtime"],
    )
    payload = validator.validate_completion(
        item["record"], expected_effective_step=12000,
        expected_continuation_local_step=2000, expected_train_job_id=JOB_ID,
    )
    assert payload["step"] == 12000
    assert len(payload["runs"]) == 2
    marker_path = item["record"] / "complete.marker"
    marker = json.loads(marker_path.read_text())
    marker["continuation_local_step"] = 4000
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="marker drift"):
        validator.validate_completion(item["record"])

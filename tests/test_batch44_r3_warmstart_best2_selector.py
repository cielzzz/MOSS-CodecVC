from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/004126_select_batch44_r3_warmstart_best2.py"
HELPER_PATH = ROOT / "scripts/batch44_r3_warmstart_quick20_completion.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


selector = load(SCRIPT, "batch44_r3_warmstart_best2_selector_test")
helper = load(HELPER_PATH, "batch44_r3_warmstart_best2_helper_test")


def write(path: Path, text: str = "fixture\n") -> Path:
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


def make_project(tmp_path: Path, monkeypatch) -> dict[str, Path]:
    project = tmp_path / "project"
    run_dir = project / selector.EXPECTED_RUN_DIR_REL
    run_dir.mkdir(parents=True)
    source = make_checkpoint(project / "outputs/lora_runs/base/step-10000", large=False)
    source_files = {}
    for name in helper.CHECKPOINT_FILES:
        item = source / name
        source_files[name] = {
            "path": str(item.resolve()),
            "bytes": item.stat().st_size,
            "sha256": helper.sha256_file(item),
        }

    contract = project / selector.EXPECTED_CONTRACT_REL
    payload = {
        "schema": helper.WARM_START_SCHEMA,
        "status": "submitted",
        "job_id": selector.EXPECTED_TRAIN_JOB_ID,
        "job_name": "ver2_9_5_final_r3_v1_warmstart10k_to30k",
        "output_dir": str(run_dir.resolve()),
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_files": source_files,
        "source_effective_step": 10000,
        "effective_step_offset": 10000,
        "continuation_local_target_step": 20000,
        "effective_target_step": 30000,
        "resume_semantics": "weights_only_warm_start_not_exact_resume",
        "step_mapping": "effective_step = 10000 + continuation_local_step",
        "state_resets": [
            "optimizer", "scheduler", "rng", "global_step", "data_iterator"
        ],
        "full_data_sha256_verified": True,
        "data": {"no_text": {"repeat": 1}, "text": {"repeat": 3}},
    }
    write(contract, json.dumps(payload, sort_keys=True) + "\n")
    monkeypatch.setattr(
        selector, "EXPECTED_CONTRACT_SHA256", helper.sha256_file(contract)
    )

    warm_record = contract.parent
    write(
        warm_record / "submitted_jobs.tsv",
        "job_name\tjob_id\tcompute_group\trunner\tout_dir\n"
        f"{payload['job_name']}\t{selector.EXPECTED_TRAIN_JOB_ID}\t"
        f"{helper.ALLOWED_COMPUTE_GROUP}\t{warm_record / 'run_train_entrypoint.sh'}\t"
        f"{run_dir.resolve()}\n",
    )
    core = {
        "OUT_DIR": str(run_dir.resolve()),
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
        f"RESUME_ADAPTER_PATH={source.resolve()}\nOUT_DIR={run_dir.resolve()}\n"
        "--resume-adapter-path $RESUME_ADAPTER_PATH\n",
    )
    write(warm_record / "generated_config_audit.json", "{}\n")
    write(warm_record / "qz_payload.json", "{}\n")

    fixed = [write(project / f"fixed-{index}.jsonl") for index in range(3)]
    return {
        "project": project,
        "run_dir": run_dir,
        "contract": contract,
        "output_root": project / selector.EXPECTED_QUICK20_OUTPUT_REL,
        "no_text": fixed[0],
        "text_source": fixed[1],
        "text20": fixed[2],
    }


def add_strict_completion(
    item: dict[str, Path],
    effective_step: int,
    *,
    no_text_sim_ref: float,
    no_text_margin: float,
    no_text_cer: float,
    text_cer: float,
) -> Path:
    local_step = selector.continuation_local_step(effective_step)
    checkpoint = make_checkpoint(item["run_dir"] / f"step-{local_step}")
    record = selector.quick20_record_root(item["project"], effective_step)
    record.mkdir(parents=True)
    output_root = item["output_root"]

    for mode in ("no_text", "text"):
        identity = helper.run_id(effective_step, mode)
        output = output_root / identity
        output.mkdir(parents=True)
        cer = no_text_cer if mode == "no_text" else text_cer
        sim_ref = no_text_sim_ref if mode == "no_text" else 0.42
        sim_src = sim_ref - no_text_margin if mode == "no_text" else 0.28
        case_ids = [f"{effective_step}-{mode}-{index}" for index in range(20)]
        cells = [
            "en_src_fixture" if mode == "text" and index < 12 else "other"
            for index in range(20)
        ]
        write(
            output / f"{identity}.summary.json",
            json.dumps({"overall": {"n": 20, "keep": 18, "cer": cer}}) + "\n",
        )
        with (output / f"{identity}.speaker_sim.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["case_id", "status", "sim_gen_ref", "sim_gen_source"],
            )
            writer.writeheader()
            for case_id in case_ids:
                writer.writerow(
                    {
                        "case_id": case_id,
                        "status": "ok",
                        "sim_gen_ref": sim_ref,
                        "sim_gen_source": sim_src,
                    }
                )
        asr_rows = [
            {
                "case_id": case_id,
                "mode": mode,
                "cell": cell,
                "content_keep": True,
            }
            for case_id, cell in zip(case_ids, cells)
        ]
        write(
            output / f"{identity}.asr_eval.jsonl",
            "".join(json.dumps(row) + "\n" for row in asr_rows),
        )
        write(
            output / f"{identity}.ref_content_similarity_summary.json",
            json.dumps({"overall": {"ref_content_lcs_f1_mean": 0.05}}) + "\n",
        )

    runner = write(record / "004123_run_batch44_r3_warmstart_quick20_local.frozen.sh")
    common = write(record / "004110_batch44_quick20_common.frozen.sh")
    completion_helper = write(
        record / "batch44_r3_warmstart_quick20_completion.frozen.py"
    )
    completion_validator = write(
        record / "batch44_r3_warmstart_quick20_validator.frozen.py"
    )
    checkpoint_files = {
        name: helper.artifact(checkpoint / name) for name in helper.CHECKPOINT_FILES
    }
    runtime = record / "LOCAL_RUNTIME.json"
    runtime_payload = {
        "schema": helper.RUNTIME_SCHEMA,
        "status": "started",
        "backend": "local",
        "hostname": "xyzhang-dev--best2-test",
        "base_effective_step": 10000,
        "effective_step": effective_step,
        "continuation_local_step": local_step,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_files": checkpoint_files,
        "checkpoint_manifest_sha256": helper.checkpoint_manifest_sha256(
            checkpoint_files
        ),
        "train_job_id": selector.EXPECTED_TRAIN_JOB_ID,
        "warm_start_contract": helper.artifact(item["contract"]),
        "gpu_count": 2,
        "gpu_indices": [0, 1],
        "gpu_model": helper.GPU_MODEL,
        "gpus": [],
        "scheduling": "fixture",
        "runner": helper.artifact(runner),
        "common_library": helper.artifact(common),
        "completion_helper": helper.artifact(completion_helper),
        "validator": helper.artifact(completion_validator),
    }
    write(runtime, json.dumps(runtime_payload, sort_keys=True) + "\n")
    helper.collect_metrics(
        record_root=record,
        output_root=output_root,
        effective_step=effective_step,
        continuation_local_step=local_step,
        checkpoint=checkpoint,
        train_job_id=selector.EXPECTED_TRAIN_JOB_ID,
        warm_start_contract=item["contract"],
    )
    helper.finalize_completion(
        record_root=record,
        output_root=output_root,
        project_root=item["project"],
        code_root=item["project"],
        effective_step=effective_step,
        continuation_local_step=local_step,
        checkpoint=checkpoint,
        train_job_id=selector.EXPECTED_TRAIN_JOB_ID,
        warm_start_contract=item["contract"],
        no_text20=item["no_text"],
        no_text20_sha256=helper.sha256_file(item["no_text"]),
        text_source=item["text_source"],
        text_source_sha256=helper.sha256_file(item["text_source"]),
        text20=item["text20"],
        text20_sha256=helper.sha256_file(item["text20"]),
        runner=runner,
        common_library=common,
        completion_helper=completion_helper,
        validator=completion_validator,
        runtime_manifest=runtime,
    )
    return record


def test_missing_registered_evidence_writes_pending_and_exits_three(
    tmp_path: Path, monkeypatch
) -> None:
    item = make_project(tmp_path, monkeypatch)
    output = tmp_path / "selection"
    rc = selector.main(
        ["--project-root", str(item["project"]), "--output-dir", str(output)]
    )
    assert rc == 3
    payload = json.loads((output / selector.SELECTION_FILENAME).read_text())
    assert payload["status"] == "pending"
    assert payload["available_effective_steps"] == []
    assert len(payload["missing_evidence"]) == 4
    assert all(
        any(f"effective-{step}" in message for message in payload["missing_evidence"])
        for step in selector.REGISTERED_EFFECTIVE_STEPS
    )
    assert "Status: **pending**" in (output / selector.SUMMARY_FILENAME).read_text()


def test_strict_four_candidate_selection_uses_effective_local_mapping(
    tmp_path: Path, monkeypatch
) -> None:
    item = make_project(tmp_path, monkeypatch)
    values = {
        24000: (0.48, 0.02, 0.20, 0.20),
        26000: (0.47, 0.10, 0.10, 0.10),
        28000: (0.47, 0.10, 0.05, 0.03),
        30000: (0.47, 0.10, 0.05, 0.03),
    }
    for effective_step, metrics in values.items():
        add_strict_completion(
            item,
            effective_step,
            no_text_sim_ref=metrics[0],
            no_text_margin=metrics[1],
            no_text_cer=metrics[2],
            text_cer=metrics[3],
        )
    output = tmp_path / "selection"
    rc = selector.main(
        ["--project-root", str(item["project"]), "--output-dir", str(output)]
    )
    assert rc == 0
    payload = json.loads((output / selector.SELECTION_FILENAME).read_text())
    assert payload["status"] == "selected"
    assert payload["selected_candidate_ids"] == [
        "r3_effective-24000",
        "r3_effective-30000",
    ]
    assert [row["effective_step"] for row in payload["candidates"]] == [
        24000, 30000, 28000, 26000
    ]
    assert payload["registered_candidate_space"][
        "effective_to_continuation_local"
    ] == {"24000": 14000, "26000": 16000, "28000": 18000, "30000": 20000}
    assert payload["warm_start"]["artifact"]["sha256"] == selector.EXPECTED_CONTRACT_SHA256
    assert all(
        set(row["checkpoint"]["files"]) == set(selector.EXPECTED_CHECKPOINT_FILES)
        for row in payload["candidates"]
    )
    summary = (output / selector.SUMMARY_FILENAME).read_text(encoding="utf-8")
    assert "Winner1: `r3_effective-24000`" in summary
    assert "Winner2: `r3_effective-30000`" in summary


def synthetic_candidate(
    *, step: int = 24000, sim_ref: float = 0.45, margin: float = 0.04,
    no_text_cer: float = 0.08, text_cer: float = 0.05,
) -> dict[str, object]:
    return {
        "effective_step": step,
        "quick20": {
            "no_text": {
                "sim_ref": sim_ref,
                "margin": margin,
                "cer": no_text_cer,
            },
            "text": {"cer": text_cer},
        },
    }


def test_ranking_key_implements_every_registered_tie_break() -> None:
    base = synthetic_candidate()
    assert selector.ranking_key(synthetic_candidate(sim_ref=0.46)) < selector.ranking_key(base)
    assert selector.ranking_key(synthetic_candidate(margin=0.05)) < selector.ranking_key(base)
    assert selector.ranking_key(synthetic_candidate(no_text_cer=0.07)) < selector.ranking_key(base)
    assert selector.ranking_key(synthetic_candidate(text_cer=0.04)) < selector.ranking_key(base)
    assert selector.ranking_key(synthetic_candidate(step=30000)) < selector.ranking_key(base)


def test_present_partial_completion_is_invalid_not_pending(tmp_path: Path, monkeypatch) -> None:
    item = make_project(tmp_path, monkeypatch)
    record = selector.quick20_record_root(item["project"], 24000)
    write(record / "COMPLETED.json", "{}\n")
    output = tmp_path / "selection"
    rc = selector.main(
        ["--project-root", str(item["project"]), "--output-dir", str(output)]
    )
    assert rc == 1
    assert not (output / selector.SELECTION_FILENAME).exists()


def test_checkpoint_tamper_fails_strict_validator(tmp_path: Path, monkeypatch) -> None:
    item = make_project(tmp_path, monkeypatch)
    for effective_step in selector.REGISTERED_EFFECTIVE_STEPS:
        add_strict_completion(
            item,
            effective_step,
            no_text_sim_ref=0.45 + effective_step / 1_000_000,
            no_text_margin=0.04,
            no_text_cer=0.08,
            text_cer=0.05,
        )
    checkpoint = item["run_dir"] / "step-14000" / "adapter_config.json"
    checkpoint.write_text('{"tampered": true}\n', encoding="utf-8")
    output = tmp_path / "selection"
    rc = selector.main(
        ["--project-root", str(item["project"]), "--output-dir", str(output)]
    )
    assert rc == 1
    assert not (output / selector.SELECTION_FILENAME).exists()


def test_contract_output_binding_is_strict(tmp_path: Path, monkeypatch) -> None:
    item = make_project(tmp_path, monkeypatch)
    contract = json.loads(item["contract"].read_text(encoding="utf-8"))
    contract["output_dir"] = str((tmp_path / "wrong-run").resolve())
    item["contract"].write_text(json.dumps(contract, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        selector, "EXPECTED_CONTRACT_SHA256", helper.sha256_file(item["contract"])
    )
    rc = selector.main(
        ["--project-root", str(item["project"]), "--output-dir", str(tmp_path / "out")]
    )
    assert rc == 1


def test_selector_source_is_read_only_and_keeps_old_best3_untouched() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "REGISTERED_EFFECTIVE_STEPS = (24_000, 26_000, 28_000, 30_000)" in source
    assert "batch44_r3_warmstart_quick20_validator.py" in source
    assert selector.SELECTION_FILENAME in source
    assert selector.SUMMARY_FILENAME in source
    assert "create-job" not in source
    assert "qzcli" not in source.lower()
    assert "subprocess" not in source

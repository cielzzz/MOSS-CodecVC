from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import wave
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"test_{name.replace('.', '_')}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def valid_best3_payload(module) -> dict:
    candidates = []
    selected = ["r3_step-26000", "r5_step-28000", "r3_step-30000"]
    for step in sorted(module.ALLOWED_STEPS):
        for arm in ("r3", "r5"):
            candidate_id = f"{arm}_step-{step}"
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "arm": arm,
                    "step": step,
                    "text_repeat": module.EXPECTED_REPEATS[arm],
                    "train_job_id": module.EXPECTED_JOBS[arm],
                    "selected_for_full320": candidate_id in selected,
                }
            )
    return {
        "schema_version": module.BEST3_SCHEMA,
        "experiment_id": module.EXPERIMENT_ID,
        "data_version": "v1_20260709",
        "status": "selected",
        "registered_candidate_space": {
            "arms": ["r3", "r5"],
            "steps": sorted(module.ALLOWED_STEPS),
            "candidate_count": 6,
        },
        "selected_candidate_ids": selected,
        "candidates": candidates,
    }


def write_checkpoint(root: Path, module, arm: str, step: int) -> Path:
    checkpoint = root / "outputs/lora_runs" / module.RUN_DIRS[arm] / f"step-{step}"
    checkpoint.mkdir(parents=True)
    config = {
        "content_cross_attn_enabled": True,
        "content_cross_attn_layers": "all",
        "content_cross_attn_feature_dim": 768,
        "content_cross_attn_gate_init": -0.5,
        "content_cross_attn_output_scale": 0.3,
        "content_encoder_layers": 2,
        "guided_attn_loss_weight": 0.05,
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
        "speaker_condition_dropout": 0.0,
    }
    (checkpoint / "timbre_memory_config.json").write_text(json.dumps(config))
    (checkpoint / "adapter_config.json").write_text("{}")
    for name in ("README.md", "adapter_model.safetensors", "timbre_memory_adapter.pt"):
        (checkpoint / name).write_bytes(b"x")
    return checkpoint


def quick_rows(module, step: int, scores: dict[str, tuple[float, float]]) -> list[dict]:
    rows = []
    for arm in module.REGISTERED_ARMS:
        for mode, offset in (("no_text", 0.0), ("text", -0.01)):
            sim_ref, sim_src = scores[arm]
            rows.append(
                {
                    "step": step,
                    "arm": arm,
                    "train_job_id": module.EXPECTED_TRAIN_JOBS[arm],
                    "mode": mode,
                    "n": 20,
                    "keep": 18,
                    "fail": 0.1,
                    "cer": 0.08,
                    "sim_ref": sim_ref + offset,
                    "sim_src": sim_src,
                    "margin": sim_ref + offset - sim_src,
                    "ref_bound_count": 10,
                    "ref_bound": 0.5,
                    "ref_content_f1": 0.05,
                    "text_en_src_quick_n": 12 if mode == "text" else "",
                    "text_en_src_quick_fail": 0.1 if mode == "text" else "",
                    "text_en_src_scope": "proxy" if mode == "text" else "",
                    "run_id": f"ver2_9_5_final_{arm}_step-{step}_{mode}_quick20_d2d3_seed1234",
                    "output_dir": f"/tmp/{arm}-{step}-{mode}",
                }
            )
    return rows


def write_quick20_artifact(
    root: Path, module, *, step: int, stamp: str, rows: list[dict]
) -> Path:
    path = module.metrics_path(root, step, stamp)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(rows))
    metrics_tsv = path.parent / "metrics.tsv"
    metrics_md = path.parent / "metrics.md"
    metrics_tsv.write_text("fixture\n")
    metrics_md.write_text("fixture\n")
    ledger = path.parent / "submitted_jobs.tsv"
    ledger.write_text(
        "job_name\tjob_id\tstep\tcompute_group\tspec\trecord_root\teval_root\tcode_root\n"
        f"fixture\tjob-11111111-1111-1111-1111-111111111111\t{step}\t"
        f"{module.ALLOWED_COMPUTE_GROUP}\t{module.ALLOWED_SPEC}\t{path.parent.resolve()}\t"
        f"{(root / f'testset/outputs/ver23_batch44_quick20_{stamp}').resolve()}\t"
        f"{module.EXPECTED_EVAL_CODE_ROOT}\n"
    )
    runner = path.parent / "004110_submit_batch44_v1_quick20_qz.frozen.sh"
    runner.write_text("#!/usr/bin/env bash\nexit 0\n")

    def artifact(item: Path) -> dict:
        return {
            "path": str(item.resolve()),
            "size": item.stat().st_size,
            "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
        }

    completion = path.parent / "COMPLETED.json"
    completion.write_text(
        json.dumps(
            {
                "schema": module.QZ_QUICK20_COMPLETION_SCHEMA,
                "status": "complete",
                "step": step,
                "record_root": str(path.parent.resolve()),
                "eval_root": str(
                    (root / f"testset/outputs/ver23_batch44_quick20_{stamp}").resolve()
                ),
                "training_jobs": module.EXPECTED_TRAIN_JOBS,
                "evaluation_job": {
                    "job_name": "fixture",
                    "job_id": "job-11111111-1111-1111-1111-111111111111",
                    "submission_ledger": artifact(ledger),
                },
                "resource_contract": {
                    "compute_group": "MTTS-3-2-0715",
                    "compute_group_id": module.ALLOWED_COMPUTE_GROUP,
                    "spec": module.ALLOWED_SPEC,
                    "instances": 1,
                    "gpus": 8,
                    "gpu_type": "NVIDIA_H200_SXM_141G",
                },
                "frozen_runner": artifact(runner),
                "metrics": {
                    "json": artifact(path),
                    "tsv": artifact(metrics_tsv),
                    "md": artifact(metrics_md),
                },
            }
        )
    )
    (path.parent / "complete.marker").write_text(
        json.dumps(
            {
                "schema": module.QZ_QUICK20_MARKER_SCHEMA,
                "status": "complete",
                "step": step,
                "completed_json_sha256": hashlib.sha256(
                    completion.read_bytes()
                ).hexdigest(),
            }
        )
    )
    return path


def convert_quick20_to_local(
    path: Path, module, *, root: Path, step: int, stamp: str
) -> Path:
    record = path.parent
    if step >= 10000:
        local_record = (
            root
            / "trainset/local_jobs"
            / f"ver23_batch44_quick20_step{step}_{stamp}"
        )
        local_record.parent.mkdir(parents=True, exist_ok=True)
        record.rename(local_record)
        record = local_record
        path = record / "metrics.json"
    (record / "submitted_jobs.tsv").unlink()
    (record / "004110_submit_batch44_v1_quick20_qz.frozen.sh").unlink()
    runtime = record / "LOCAL_RUNTIME.json"
    runner = record / "004117_run_batch44_v1_quick20_local.frozen.sh"
    runner.write_text("#!/usr/bin/env bash\nexit 0\n")
    common = record / "004110_batch44_quick20_common.frozen.sh"
    common.write_text("fixture\n")
    helper = record / "batch44_quick20_local_completion.frozen.py"
    helper.write_text("fixture\n")

    def artifact(item: Path) -> dict:
        return {
            "path": str(item.resolve()),
            "size": item.stat().st_size,
            "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
        }

    fixed_paths = {
        "no_text20": root
        / "testset/validation/ver2_8_timbre_quick20_seedtts_no_text_20260704.jsonl",
        "text_source": root
        / "testset/validation/seedtts_vc_ver2_3_validation.jsonl",
        "text20": record / "ver23_batch44_text_quick20_8cell_20260713.jsonl",
    }
    for item in fixed_paths.values():
        item.parent.mkdir(parents=True, exist_ok=True)
        item.write_text("fixture\n")
    checkpoints = {
        arm: write_checkpoint(root, module, arm, step) for arm in module.REGISTERED_ARMS
    }
    checkpoint_payload = {
        arm: {
            "path": str(checkpoint.resolve()),
            "step": step,
            "training_job_id": module.EXPECTED_TRAIN_JOBS[arm],
            "files": {
                name: artifact(checkpoint / name)
                for name in module.REQUIRED_CHECKPOINT_FILES
            },
        }
        for arm, checkpoint in checkpoints.items()
    }
    identity = (
        root
        / "trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713"
    )
    pair_ledger = identity / "submitted_pair.tsv"
    pair_ledger.parent.mkdir(parents=True, exist_ok=True)
    pair_ledger.write_text("fixture\n")
    train_args = {}
    for arm in module.REGISTERED_ARMS:
        item = identity / arm / "train_args_dry_run_core.json"
        item.parent.mkdir(parents=True, exist_ok=True)
        item.write_text("{}\n")
        train_args[arm] = item
    eval_root = root / f"testset/outputs/ver23_batch44_quick20_{stamp}"
    runs = []
    for arm in module.REGISTERED_ARMS:
        for mode in ("no_text", "text"):
            run_id = f"ver2_9_5_final_{arm}_step-{step}_{mode}_quick20_d2d3_seed1234"
            output = eval_root / run_id
            outputs = {
                "summary": output / f"{run_id}.summary.json",
                "asr": output / f"{run_id}.asr_eval.jsonl",
                "speaker": output / f"{run_id}.speaker_sim.csv",
                "ref_content": output / f"{run_id}.ref_content_similarity_summary.json",
            }
            for item in outputs.values():
                item.parent.mkdir(parents=True, exist_ok=True)
                item.write_text("fixture\n")
            runs.append({
                "arm": arm,
                "mode": mode,
                "run_id": run_id,
                "training_job_id": module.EXPECTED_TRAIN_JOBS[arm],
                "checkpoint": str(checkpoints[arm].resolve()),
                "output_dir": str(output.resolve()),
                "artifacts": {name: artifact(item) for name, item in outputs.items()},
            })
    gpu_rows = [
        {
            "index": index,
            "uuid": f"GPU-00000000-0000-0000-0000-{index:012d}",
            "name": module.LOCAL_GPU_MODEL,
            "memory_total_mib": 49140,
            "driver_version": "550.163.01",
        }
        for index in (0, 1)
    ]
    scheduling = "four lanes sequential; each lane uses GPUs 0,1 with two shards"
    runtime.write_text(json.dumps({
        "schema": "moss_codecvc.batch44_v1_quick20_local_runtime.v1",
        "backend": "local",
        "status": "started",
        "hostname": "xyzhang-dev--fixture",
        "gpu_count": 2,
        "gpu_indices": [0, 1],
        "gpu_model": module.LOCAL_GPU_MODEL,
        "gpus": gpu_rows,
        "scheduling": scheduling,
        "runner": artifact(runner),
        "common_library": artifact(common),
        "completion_helper": artifact(helper),
    }) + "\n")
    completion = record / "COMPLETED.json"
    completion.write_text(
        json.dumps(
            {
                "schema": module.LOCAL_QUICK20_COMPLETION_SCHEMA,
                "status": "complete",
                "backend": "local",
                "step": step,
                "record_root": str(record.resolve()),
                "eval_root": str(
                    eval_root.resolve()
                ),
                "code_root": module.EXPECTED_EVAL_CODE_ROOT,
                "training_jobs": module.EXPECTED_TRAIN_JOBS,
                "execution": {
                    "hostname": "xyzhang-dev--fixture",
                    "gpu_count": 2,
                    "gpu_indices": [0, 1],
                    "gpu_model": module.LOCAL_GPU_MODEL,
                    "gpus": gpu_rows,
                    "scheduling": scheduling,
                    "runtime_manifest": artifact(runtime),
                },
                "runner": artifact(runner),
                "common_library": artifact(common),
                "completion_helper": artifact(helper),
                "fixed_inputs": {
                    name: artifact(item) for name, item in fixed_paths.items()
                },
                "checkpoints": checkpoint_payload,
                "training_provenance": {
                    "pair_ledger": artifact(pair_ledger),
                    "train_args": {
                        arm: artifact(item) for arm, item in train_args.items()
                    },
                },
                "runs": runs,
                "metrics": {
                    "json": artifact(path),
                    "tsv": artifact(record / "metrics.tsv"),
                    "md": artifact(record / "metrics.md"),
                },
            }
        )
    )
    (record / "complete.marker").write_text(
        json.dumps(
            {
                "schema": module.LOCAL_QUICK20_MARKER_SCHEMA,
                "status": "complete",
                "backend": "local",
                "step": step,
                "completed_json_sha256": hashlib.sha256(
                    completion.read_bytes()
                ).hexdigest(),
            }
        )
    )
    return path


def test_best3_selector_accepts_strict_local_4090_quick20(tmp_path: Path):
    module = load_script("004103_select_batch43_best3.py")
    step = 26000
    path = write_quick20_artifact(
        tmp_path,
        module,
        step=step,
        stamp="local",
        rows=quick_rows(module, step, {"r3": (0.45, 0.38), "r5": (0.44, 0.37)}),
    )
    path = convert_quick20_to_local(
        path, module, root=tmp_path, step=step, stamp="local"
    )

    provenance = module.audit_quick20_provenance(
        path, project_root=tmp_path, step=step
    )
    assert provenance["backend"] == "local"
    assert provenance["evaluation_id"] == "local:xyzhang-dev--fixture"
    assert set(module.load_metrics(path, project_root=tmp_path, step=step)) == {
        ("r3", "no_text"), ("r3", "text"), ("r5", "no_text"), ("r5", "text")
    }


def test_best3_selector_rejects_local_quick20_with_qz_ledger(tmp_path: Path):
    module = load_script("004103_select_batch43_best3.py")
    step = 26000
    path = write_quick20_artifact(
        tmp_path,
        module,
        step=step,
        stamp="local-ledger",
        rows=quick_rows(module, step, {"r3": (0.45, 0.38), "r5": (0.44, 0.37)}),
    )
    path = convert_quick20_to_local(
        path, module, root=tmp_path, step=step, stamp="local-ledger"
    )
    (path.parent / "submitted_jobs.tsv").write_text("forged\n")
    with pytest.raises(ValueError, match="must not contain a QZ submission ledger"):
        module.audit_quick20_provenance(path, project_root=tmp_path, step=step)


def test_best3_selector_rejects_local_quick20_runner_sha_drift(tmp_path: Path):
    module = load_script("004103_select_batch43_best3.py")
    step = 26000
    path = write_quick20_artifact(
        tmp_path,
        module,
        step=step,
        stamp="local-sha",
        rows=quick_rows(module, step, {"r3": (0.45, 0.38), "r5": (0.44, 0.37)}),
    )
    path = convert_quick20_to_local(
        path, module, root=tmp_path, step=step, stamp="local-sha"
    )
    (path.parent / "004117_run_batch44_v1_quick20_local.frozen.sh").write_text(
        "#!/usr/bin/env bash\nexit 1\n"
    )
    with pytest.raises(ValueError, match="local frozen runner SHA256"):
        module.audit_quick20_provenance(path, project_root=tmp_path, step=step)


@pytest.mark.parametrize(
    "target",
    (
        "checkpoint",
        "common_library",
        "completion_helper",
        "fixed_input",
        "training_provenance",
        "run_artifact",
    ),
)
def test_best3_selector_rejects_same_size_local_provenance_tamper(
    tmp_path: Path, target: str
) -> None:
    module = load_script("004103_select_batch43_best3.py")
    step = 26000
    path = write_quick20_artifact(
        tmp_path,
        module,
        step=step,
        stamp=f"tamper-{target}",
        rows=quick_rows(module, step, {"r3": (0.45, 0.38), "r5": (0.44, 0.37)}),
    )
    path = convert_quick20_to_local(
        path,
        module,
        root=tmp_path,
        step=step,
        stamp=f"tamper-{target}",
    )
    record = path.parent
    run_id = f"ver2_9_5_final_r3_step-{step}_no_text_quick20_d2d3_seed1234"
    targets = {
        "checkpoint": (
            tmp_path
            / "outputs/lora_runs"
            / module.RUN_DIRS["r3"]
            / f"step-{step}/adapter_model.safetensors"
        ),
        "common_library": record / "004110_batch44_quick20_common.frozen.sh",
        "completion_helper": record / "batch44_quick20_local_completion.frozen.py",
        "fixed_input": (
            tmp_path
            / "testset/validation/ver2_8_timbre_quick20_seedtts_no_text_20260704.jsonl"
        ),
        "training_provenance": (
            tmp_path
            / "trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713/"
            "r3/train_args_dry_run_core.json"
        ),
        "run_artifact": (
            tmp_path
            / f"testset/outputs/ver23_batch44_quick20_tamper-{target}"
            / run_id
            / f"{run_id}.speaker_sim.csv"
        ),
    }
    victim = targets[target]
    original = victim.read_bytes()
    assert original
    replacement = bytes([original[0] ^ 1]) + original[1:]
    assert len(replacement) == len(original)
    victim.write_bytes(replacement)
    with pytest.raises(ValueError, match="SHA256"):
        module.audit_quick20_provenance(path, project_root=tmp_path, step=step)


def test_best3_selector_ranks_all_six_and_plans_paired_steps(tmp_path: Path):
    module = load_script("004103_select_batch43_best3.py")
    scores = {
        26000: {"r3": (0.43, 0.36), "r5": (0.44, 0.37)},
        28000: {"r3": (0.47, 0.38), "r5": (0.42, 0.36)},
        30000: {"r3": (0.45, 0.37), "r5": (0.46, 0.38)},
    }
    for step in module.REGISTERED_STEPS:
        for arm in module.REGISTERED_ARMS:
            write_checkpoint(tmp_path, module, arm, step)
        write_quick20_artifact(
            tmp_path,
            module,
            step=step,
            stamp="fixture",
            rows=quick_rows(module, step, scores[step]),
        )
    plan = module.build_plan(tmp_path, stamp="fixture")
    assert plan["status"] == "selected"
    assert len(plan["candidates"]) == 6
    assert plan["selected_candidate_ids"] == [
        "r3_step-28000",
        "r5_step-30000",
        "r3_step-30000",
    ]
    assert plan["paired_full320_plan"]["selected_steps"] == [28000, 30000]
    assert plan["paired_full320_plan"]["extra_counterparts_evaluated_by_paired_wrapper"] == [
        "r5_step-28000"
    ]


def test_best3_selector_rejects_stopped_batch43_v2_training_job(tmp_path: Path):
    module = load_script("004103_select_batch43_best3.py")
    step = 26000
    rows = quick_rows(
        module,
        step,
        {"r3": (0.45, 0.38), "r5": (0.44, 0.37)},
    )
    rows[0]["train_job_id"] = sorted(module.REJECTED_BATCH43_V2_TRAIN_JOBS)[0]
    path = write_quick20_artifact(
        tmp_path, module, step=step, stamp="legacy", rows=rows
    )
    with pytest.raises(ValueError, match="stopped Batch-43 v2 training"):
        module.load_metrics(path, project_root=tmp_path, step=step)


def test_best3_selector_uses_explicit_project_root_with_real_qz_jobs_symlink(
    tmp_path: Path,
) -> None:
    module = load_script("004103_select_batch43_best3.py")
    project_root = tmp_path / "MOSS-CodecVC"
    real_qz_jobs = project_root / "trainset/zh45w_en22w_no_text/qz_jobs"
    real_qz_jobs.mkdir(parents=True)
    (project_root / "trainset/qz_jobs").symlink_to(
        Path("zh45w_en22w_no_text/qz_jobs")
    )
    step = 26000
    path = write_quick20_artifact(
        project_root,
        module,
        step=step,
        stamp="symlink",
        rows=quick_rows(
            module,
            step,
            {"r3": (0.45, 0.38), "r5": (0.44, 0.37)},
        ),
    )

    assert path.parent.parent == real_qz_jobs.resolve()
    rows = module.load_metrics(
        path,
        project_root=project_root,
        step=step,
    )
    assert set(rows) == {
        ("r3", "no_text"),
        ("r3", "text"),
        ("r5", "no_text"),
        ("r5", "text"),
    }


def test_best3_selector_rejects_metrics_outside_explicit_project_root(
    tmp_path: Path,
) -> None:
    module = load_script("004103_select_batch43_best3.py")
    source_root = tmp_path / "source"
    claimed_root = tmp_path / "claimed"
    step = 26000
    path = write_quick20_artifact(
        source_root,
        module,
        step=step,
        stamp="foreign",
        rows=quick_rows(
            module,
            step,
            {"r3": (0.45, 0.38), "r5": (0.44, 0.37)},
        ),
    )

    with pytest.raises(ValueError, match="quick20 metrics path=.*expected"):
        module.load_metrics(
            path,
            project_root=claimed_root,
            step=step,
        )


def test_finalizer_replays_quick20_and_rejects_edited_best3(tmp_path: Path):
    selector = load_script("004103_select_batch43_best3.py")
    finalizer = load_script("004107_finalize_batch43_pathx_final.py")
    scores = {
        26000: {"r3": (0.43, 0.36), "r5": (0.44, 0.37)},
        28000: {"r3": (0.47, 0.38), "r5": (0.42, 0.36)},
        30000: {"r3": (0.45, 0.37), "r5": (0.46, 0.38)},
    }
    for step in selector.REGISTERED_STEPS:
        for arm in selector.REGISTERED_ARMS:
            write_checkpoint(tmp_path, selector, arm, step)
        write_quick20_artifact(
            tmp_path,
            selector,
            step=step,
            stamp="replay",
            rows=quick_rows(selector, step, scores[step]),
        )
    plan = selector.build_plan(tmp_path, stamp="replay")
    # Keep all identities, flags and source hashes plausible, but hand-edit the
    # derived ranking score.  004107 must replay 004103 and reject the plan.
    plan["candidates"][0]["quick20"]["pooled_wavlm_sim_ref"] += 0.123
    path = tmp_path / "best3_selection.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="differs from replayed 004103"):
        finalizer.load_best3(path, project_root=tmp_path, replay_quick20=True)


def write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 160)


def diagnostic_row(case_id: str, bucket: str, audio_root: Path, run: str, cell: str) -> dict:
    source = audio_root / f"{case_id}.source.wav"
    reference = audio_root / f"{case_id}.reference.wav"
    target = audio_root / f"{case_id}.{run}.wav"
    for path in (source, reference, target):
        write_wav(path)
    values = {
        "ref-bound": (0.60, 0.40),
        "src-bound": (0.35, 0.60),
        "ambiguous": (0.50, 0.49),
    }[bucket]
    return {
        "run": run,
        "mode": "no_text",
        "case_id": case_id,
        "cell": cell,
        "content_keep": "True",
        "cer_tgt": "0.05",
        "target_audio": str(target),
        "source_audio": str(source),
        "timbre_ref_audio": str(reference),
        "sim_gen_ref": values[0],
        "sim_gen_source": values[1],
        "ecapa_sim_gen_ref": values[0],
        "ecapa_sim_gen_source": values[1],
    }


def test_blind20_uses_shared_stratified_cases_and_balanced_positions(tmp_path: Path):
    module = load_script("004104_build_batch43_best3_blind20.py")
    candidate_ids = ["r3_step-26000", "r5_step-28000", "r3_step-30000"]
    buckets = ["ref-bound"] * 6 + ["src-bound"] * 6 + ["ambiguous"] * 12
    systems = {"batch33": {}, **{candidate_id: {} for candidate_id in candidate_ids}}
    for index, bucket in enumerate(buckets):
        case_id = f"case-{index:03d}"
        cell = f"cell-{index % 8}"
        for system_id in systems:
            row = diagnostic_row(case_id, bucket, tmp_path / "audio", system_id, cell)
            label, metrics = module.dual_label(row, margin=0.05)
            systems[system_id][case_id] = {
                "case_id": case_id,
                "cell": cell,
                "content_keep": True,
                "cer": 0.05,
                "generated_audio": Path(row["target_audio"]),
                "source_audio": Path(row["source_audio"]),
                "reference_audio": Path(row["timbre_ref_audio"]),
                "metrics": metrics,
            }
            assert label == bucket
    eligible, skipped = module.common_eligible(
        {candidate_id: systems[candidate_id] for candidate_id in candidate_ids},
        systems["batch33"],
    )
    assert not skipped
    selected, audit = module.select_cases(eligible, seed=43)
    assert len(selected) == 20
    assert audit["selected_by_bucket"] == {
        "ref-bound": 3,
        "src-bound": 3,
        "ambiguous": 4,
        "random": 10,
    }
    (tmp_path / "diag.csv").write_text("run,case_id\n")
    page = module.stage_page(
        candidate_id=candidate_ids[0],
        rows=selected,
        output_dir=tmp_path / "page",
        manifest_path=tmp_path / "private/page.manifest.json",
        seed=44,
        diagnostics_csv=tmp_path / "diag.csv",
        run_id="run",
        force=False,
    )
    manifest = json.loads(Path(page["manifest"]).read_text())
    assert manifest["experiment_id"] == module.EXPERIMENT_ID
    assert manifest["data_version"] == "v1_20260709"
    assert manifest["position_balance"] == {"batch33": 10, "candidate": 10}
    assert candidate_ids[0] not in (tmp_path / "page/index.html").read_text()
    assert candidate_ids[0] not in page["index"]
    assert not list((tmp_path / "page").glob("*.json"))
    assert Path(page["manifest"]).is_relative_to(tmp_path / "private")


def test_blind20_binds_registered_full320_csv_run_and_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = load_script("004104_build_batch43_best3_blind20.py")
    step = 26000
    candidate = {"candidate_id": "r3_step-26000", "arm": "r3", "step": step}
    completion_path = module.default_completion_path(tmp_path, step)
    completion_path.parent.mkdir(parents=True)
    step_root = (
        tmp_path
        / f"testset/outputs/ver23_batch44_paired_full320_20260713/step-{step}"
    )
    aggregate = step_root / "aggregate"
    aggregate.mkdir(parents=True)
    metrics_path = aggregate / "paired_metrics.json"
    metrics_path.write_text("[]")
    dual_path = aggregate / "dual_encoder_cases.csv"
    dual_path.write_text("run,mode,case_id\n")
    completion = {
        "paired_metrics_json": str(metrics_path.resolve()),
        "dual_encoder_cases_csv": str(dual_path.resolve()),
        "step_root": str(step_root.resolve()),
    }
    completion_path.write_text(json.dumps(completion))
    objective = {"arm": "r3", "scope": "no_text", "step": step}

    class DummyValidator:
        @staticmethod
        def validate_full320_step(**kwargs):
            assert kwargs["project_root"] == tmp_path
            return completion, {("r3", "no_text"): objective}

    monkeypatch.setattr(module, "load_full320_validator", lambda: DummyValidator)
    wrong_csv = aggregate / "replacement.csv"
    wrong_csv.write_text("run,mode,case_id\n")
    expected_run = (
        "ver2_9_5_final_r3_step-26000_no_text_seedtts160_d2d3_seed1234"
    )
    with pytest.raises(ValueError, match="must be the COMPLETED dual_encoder_cases_csv"):
        module.bind_candidate_full320_evidence(
            candidate=candidate,
            diagnostics_csv=wrong_csv,
            run_id=expected_run,
            completion_path=completion_path,
            project_root=tmp_path,
        )
    with pytest.raises(ValueError, match="expected registered lane"):
        module.bind_candidate_full320_evidence(
            candidate=candidate,
            diagnostics_csv=dual_path,
            run_id="ver2_9_5_final_r5_step-26000_no_text_seedtts160_d2d3_seed1234",
            completion_path=completion_path,
            project_root=tmp_path,
        )
    evidence = module.bind_candidate_full320_evidence(
        candidate=candidate,
        diagnostics_csv=dual_path,
        run_id=expected_run,
        completion_path=completion_path,
        project_root=tmp_path,
    )
    assert evidence["dual_encoder_cases_sha256"] == module.sha256_file(dual_path)

    replacement_audio = tmp_path / "replacement/case-000.wav"
    source_audio = tmp_path / "anchors/source.wav"
    reference_audio = tmp_path / "anchors/reference.wav"
    for path in (replacement_audio, source_audio, reference_audio):
        write_wav(path)
    with dual_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run", "mode", "case_id", "cell", "content_keep", "cer_tgt",
                "target_audio", "source_audio", "timbre_ref_audio", "sim_gen_ref",
                "sim_gen_source", "ecapa_sim_gen_ref", "ecapa_sim_gen_source",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run": expected_run, "mode": "no_text", "case_id": "case-000",
                "cell": "cell", "content_keep": "True", "cer_tgt": "0.0",
                "target_audio": replacement_audio, "source_audio": source_audio,
                "timbre_ref_audio": reference_audio, "sim_gen_ref": "0.5",
                "sim_gen_source": "0.4", "ecapa_sim_gen_ref": "0.5",
                "ecapa_sim_gen_source": "0.4",
            }
        )
    with pytest.raises(ValueError, match="target_audio=.*expected"):
        module.load_diagnostics(
            dual_path,
            run_id=expected_run,
            margin=0.05,
            expected_target_root=Path(evidence["target_audio_root"]),
            expected_n=1,
        )


def metric_rows(module, step: int) -> list[dict]:
    rows = []
    for arm in ("r3", "r5"):
        base = {
            "step": step,
            "arm": arm,
            "text_repeat": module.EXPECTED_REPEATS[arm],
            "train_job_id": module.EXPECTED_JOBS[arm],
        }
        no_text = {
            **base, "scope": "no_text", "n": 160, "keep": 150,
            "fail_count": 10, "fail_rate": 10 / 160, "cer": 0.10,
            "wavlm_sim_ref": 0.45, "wavlm_sim_src": 0.40, "wavlm_margin": 0.05,
            "wavlm_ref_bound": 0.50, "speechbrain_sim_ref": 0.50,
            "speechbrain_sim_src": 0.30, "speechbrain_margin": 0.20,
            "speechbrain_ref_bound": 0.70, "ref_content_lcs_f1": 0.05,
            "text_en_src_n": "", "text_en_src_fail_count": "",
            "text_en_src_fail_rate": "", "text_en_src_cer": "",
        }
        text = {
            **base, "scope": "text", "n": 160, "keep": 152,
            "fail_count": 8, "fail_rate": 8 / 160, "cer": 0.06,
            "wavlm_sim_ref": 0.43, "wavlm_sim_src": 0.28, "wavlm_margin": 0.15,
            "wavlm_ref_bound": 0.60, "speechbrain_sim_ref": 0.52,
            "speechbrain_sim_src": 0.18, "speechbrain_margin": 0.34,
            "speechbrain_ref_bound": 0.90, "ref_content_lcs_f1": 0.04,
            "text_en_src_n": 80, "text_en_src_fail_count": 6,
            "text_en_src_fail_rate": 6 / 80, "text_en_src_cer": 0.05,
        }
        combined = {
            **base, "scope": "all", "n": 320, "keep": 302,
            "fail_count": 18, "fail_rate": 18 / 320, "cer": 0.08,
            "wavlm_sim_ref": 0.44, "wavlm_sim_src": 0.34, "wavlm_margin": 0.10,
            "wavlm_ref_bound": 0.55, "speechbrain_sim_ref": 0.51,
            "speechbrain_sim_src": 0.24, "speechbrain_margin": 0.27,
            "speechbrain_ref_bound": 0.80, "ref_content_lcs_f1": 0.045,
            "text_en_src_n": 80, "text_en_src_fail_count": 6,
            "text_en_src_fail_rate": 6 / 80, "text_en_src_cer": 0.05,
        }
        rows.extend((no_text, text, combined))
    return rows


def test_full320_validator_requires_exact_six_row_contract(tmp_path: Path):
    module = load_script("004107_finalize_batch43_pathx_final.py")
    step = 26000
    record = tmp_path / f"trainset/qz_jobs/ver23_batch44_paired_full320_step{step}_20260713"
    step_root = tmp_path / f"testset/outputs/ver23_batch44_paired_full320_20260713/step-{step}"
    aggregate = step_root / "aggregate"
    aggregate.mkdir(parents=True)
    record.mkdir(parents=True)
    metrics_path = aggregate / "paired_metrics.json"
    metrics_path.write_text(json.dumps(metric_rows(module, step)))
    dual = aggregate / "dual_encoder_cases.csv"
    with dual.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run", "mode", "case_id", "sim_gen_ref", "sim_gen_source",
                "ecapa_sim_gen_ref", "ecapa_sim_gen_source", "cer_tgt",
            ],
        )
        writer.writeheader()
        for arm in ("r3", "r5"):
            for mode in ("no_text", "text"):
                run = f"ver2_9_5_final_{arm}_step-{step}_{mode}_seedtts160_d2d3_seed1234"
                for index in range(160):
                    writer.writerow(
                        {
                            "run": run, "mode": mode, "case_id": f"case-{index:03d}",
                            "sim_gen_ref": 0.45, "sim_gen_source": 0.40,
                            "ecapa_sim_gen_ref": 0.50, "ecapa_sim_gen_source": 0.30,
                            "cer_tgt": 0.05,
                        }
                    )
    for name in ("paired_metrics.tsv", "paired_metrics.md"):
        (aggregate / name).write_text("fixture\n")
    completeness = {
        "lanes": [
            {
                "arm": arm,
                "mode": mode,
                "run_id": f"ver2_9_5_final_{arm}_step-{step}_{mode}_seedtts160_d2d3_seed1234",
                "checkpoint": str(
                    (tmp_path / "outputs/lora_runs" / module.RUN_DIRS[arm] / f"step-{step}").resolve()
                ),
                "training_job_id": module.EXPECTED_JOBS[arm],
                "rows": 160,
                "asr_rows": 160,
                "bnf_extraction_lines": 160 if mode == "no_text" else 0,
            }
            for arm in ("r3", "r5")
            for mode in ("no_text", "text")
        ]
    }
    (aggregate / "completeness.json").write_text(json.dumps(completeness))
    validation = tmp_path / "testset/validation/seedtts_vc_ver2_3_validation.jsonl"
    validation.parent.mkdir(parents=True)
    validation.write_text("{}\n")
    train_ledger = (
        tmp_path
        / "trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713/submitted_pair.tsv"
    )
    train_ledger.parent.mkdir(parents=True)
    train_ledger.write_text("fixture\n")
    (record / "submitted_jobs.tsv").write_text(
        "job_name\tjob_id\tstep\tcompute_group\tspec\trecord_root\tstep_root\tcode_root\tr3_train_job_id\tr5_train_job_id\n"
        f"fixture\tjob-11111111-1111-1111-1111-111111111111\t{step}\t{module.ALLOWED_COMPUTE_GROUP}\t"
        f"{module.ALLOWED_SPEC}\t{record.resolve()}\t{step_root.resolve()}\t{module.EXPECTED_CODE_ROOT}\t"
        f"{module.EXPECTED_JOBS['r3']}\t{module.EXPECTED_JOBS['r5']}\n"
    )
    completion_path = record / "COMPLETED.json"
    completion_path.write_text(
        json.dumps(
            {
                "schema": module.FULL320_COMPLETION_SCHEMA,
                "status": "complete",
                "step": step,
                "training_jobs": module.EXPECTED_JOBS,
                "training_pair_ledger": str(train_ledger.resolve()),
                "training_pair_ledger_sha256": module.sha256_file(train_ledger),
                "validation_jsonl": str(validation.resolve()),
                "validation_sha256": module.sha256_file(validation),
                "code_root": str(module.EXPECTED_CODE_ROOT),
                "record_root": str(record.resolve()),
                "step_root": str(step_root.resolve()),
                "completeness_json": str((aggregate / "completeness.json").resolve()),
                "paired_metrics_json": str(metrics_path),
                "paired_metrics_tsv": str((aggregate / "paired_metrics.tsv").resolve()),
                "paired_metrics_md": str((aggregate / "paired_metrics.md").resolve()),
                "dual_encoder_cases_csv": str(dual),
                "scope": {"r3": {"no_text": 160, "text": 160}, "r5": {"no_text": 160, "text": 160}},
                "gpu_plan": {"r3_no_text": "0,1", "r3_text": "2,3", "r5_no_text": "4,5", "r5_text": "6,7"},
            }
        )
    )
    completion, indexed = module.validate_full320_step(
        step=step, completion_path=completion_path, metrics_path=metrics_path,
        project_root=tmp_path,
    )
    assert completion["status"] == "complete"
    assert len(indexed) == 6
    broken = json.loads(metrics_path.read_text())
    broken[0]["n"] = 159
    metrics_path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="n=159"):
        module.validate_full320_step(
            step=step, completion_path=completion_path, metrics_path=metrics_path,
            project_root=tmp_path,
        )
    broken = metric_rows(module, step)
    broken[0]["keep"] = -999
    broken[0]["fail_count"] = 999999
    metrics_path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="invalid keep"):
        module.validate_full320_step(
            step=step, completion_path=completion_path, metrics_path=metrics_path,
            project_root=tmp_path,
        )
    metrics_path.write_text(json.dumps(metric_rows(module, step)))
    submit_path = record / "submitted_jobs.tsv"
    submit_path.write_text(
        submit_path.read_text().replace(module.ALLOWED_COMPUTE_GROUP, "H200-3-2")
    )
    with pytest.raises(ValueError, match="MTTS/QZ provenance drift"):
        module.validate_full320_step(
            step=step, completion_path=completion_path, metrics_path=metrics_path,
            project_root=tmp_path,
        )


def test_pathx_final_adapter_accepts_only_registered_manifest_and_patches_identity(tmp_path: Path):
    module = load_script("004106_run_batch42_pathx_final_strict.py")
    module.PROJECT_ROOT = tmp_path
    checkpoint = tmp_path / "outputs/lora_runs" / module.ALLOWED_RUN_DIRS["r3"] / "step-26000"
    checkpoint.mkdir(parents=True)
    timbre = {
        "content_cross_attn_enabled": True,
        "content_cross_attn_layers": "all",
        "content_cross_attn_feature_dim": 768,
        "content_cross_attn_gate_init": -0.5,
        "content_cross_attn_output_scale": 0.3,
        "content_encoder_layers": 2,
        "guided_attn_loss_weight": 0.05,
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
        "speaker_condition_dropout": 0.0,
    }
    (checkpoint / "timbre_memory_config.json").write_text(json.dumps(timbre))
    (checkpoint / "adapter_config.json").write_text("{}")
    for name in ("README.md", "adapter_model.safetensors", "timbre_memory_adapter.pt"):
        (checkpoint / name).write_bytes(name.encode())
    files = {
        name: {
            "size": (checkpoint / name).stat().st_size,
            "sha256": module.sha256_file(checkpoint / name),
        }
        for name in module.REQUIRED_FILES
    }
    selection_path = tmp_path / "FINAL_SELECTION.json"
    valid_selection = {
        "schema_version": module.FINAL_SCHEMA,
        "experiment_id": module.EXPERIMENT_ID,
        "data_version": "v1_20260709",
        "status": "final",
        "system_id": module.SYSTEM_ID,
        "candidate": {
            "candidate_id": "r3_step-26000",
            "arm": "r3",
            "step": 26000,
            "text_repeat": module.EXPECTED_REPEATS["r3"],
            "train_job_id": module.EXPECTED_TRAIN_JOBS["r3"],
            "checkpoint_path": str(checkpoint),
            "model_files": files,
        },
    }
    selection_path.write_text(json.dumps(valid_selection))
    with pytest.raises(ValueError, match="not bound to the current registered 004107"):
        module.load_final_selection(selection_path, verify_checkpoint_hashes=True)
    forged = json.loads(json.dumps(valid_selection))
    validator = module.load_final_provenance_validator()
    forged["producer"] = validator.producer_registration(tmp_path)
    selection_path.write_text(json.dumps(forged))
    with pytest.raises(FileNotFoundError):
        module.load_final_selection(selection_path, verify_checkpoint_hashes=True)
    selection_path.write_text(json.dumps(valid_selection))
    selection = {
        **valid_selection,
        "_selection_path": str(selection_path.resolve()),
        "_selection_sha256": module.sha256_file(selection_path),
    }

    class Dummy:
        def inference_config(self, _args):
            return {"ref_audio_cfg_scale": 1.0}

    dummy = Dummy()
    module.configure_upstream(dummy, selection)
    assert dummy.SYSTEM_ID == "path_x_final"
    assert dummy.REGISTERED_MODEL_PATH == checkpoint
    assert dummy.REGISTERED_CODE_ROOT == module.FINAL_CODE_ROOT
    assert dummy.REGISTERED_MODEL_FILES["adapter_model.safetensors"] == files["adapter_model.safetensors"]
    assert dummy.inference_config(None)["final_selection_sha256"] == module.sha256_file(selection_path)

    broken = json.loads(selection_path.read_text())
    broken["candidate"]["step"] = 10000
    broken["candidate"]["candidate_id"] = "r3_step-10000"
    selection_path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="invalid final arm/step"):
        module.load_final_selection(selection_path)

    legacy_schema = json.loads(json.dumps(valid_selection))
    legacy_schema["schema_version"] = "moss_codecvc.batch43_final_selection.v1"
    selection_path.write_text(json.dumps(legacy_schema))
    with pytest.raises(ValueError, match="schema_version"):
        module.load_final_selection(selection_path)

    legacy_checkpoint = json.loads(json.dumps(valid_selection))
    legacy_checkpoint["candidate"]["checkpoint_path"] = str(
        tmp_path / "outputs/lora_runs/ver2_9_5_final_r3_v2_30k/step-26000"
    )
    selection_path.write_text(json.dumps(legacy_checkpoint))
    with pytest.raises(ValueError, match="checkpoint path"):
        module.load_final_selection(selection_path)


def test_finalizer_rejects_stopped_10k_outside_registered_best3(tmp_path: Path):
    module = load_script("004107_finalize_batch43_pathx_final.py")
    payload = valid_best3_payload(module)
    payload["candidates"][0]["step"] = 10000
    payload["candidates"][0]["candidate_id"] = "r3_step-10000"
    path = tmp_path / "best3.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="outside registered 30k checkpoints"):
        module.load_best3(path)


def test_finalizer_rejects_legacy_batch43_schema_and_training_job(tmp_path: Path):
    module = load_script("004107_finalize_batch43_pathx_final.py")
    path = tmp_path / "best3.json"

    legacy_schema = valid_best3_payload(module)
    legacy_schema["schema_version"] = "moss_codecvc.batch43_best3_selection.v1"
    path.write_text(json.dumps(legacy_schema))
    with pytest.raises(ValueError, match="invalid Best3 selection"):
        module.load_best3(path)

    legacy_job = valid_best3_payload(module)
    legacy_job["candidates"][0]["train_job_id"] = sorted(
        module.REJECTED_BATCH43_V2_JOBS
    )[0]
    path.write_text(json.dumps(legacy_job))
    with pytest.raises(ValueError, match="references stopped Batch-43 v2"):
        module.load_best3(path)


def test_objective_gate_returns_verdict_and_rejects_stopped_10k_profile():
    module = load_script("004107_finalize_batch43_pathx_final.py")
    passing = {
        "no_text": {
            "cer": 0.08,
            "wavlm_sim_ref": 0.45,
            "wavlm_margin": 0.02,
            "ref_content_lcs_f1": 0.20,
        },
        "text": {"cer": 0.05, "text_en_src_fail_rate": 0.10},
    }
    verdict = module.evaluate_objective_gate(passing)
    assert verdict["pass"] is True
    stopped_10k = json.loads(json.dumps(passing))
    stopped_10k["no_text"].update(
        {"cer": 0.1967, "wavlm_sim_ref": 0.3866, "wavlm_margin": -0.1415}
    )
    verdict = module.evaluate_objective_gate(stopped_10k)
    assert verdict["pass"] is False
    assert verdict["checks"]["no_text_wavlm_margin_ge_0p02"] is False


def test_final_auto_winner_requires_objective_pass_and_nonnegative_subjective_result():
    module = load_script("004107_finalize_batch43_pathx_final.py")
    ranked = [
        {"candidate_id": "a", "subjective_net": -1},
        {"candidate_id": "b", "subjective_net": -2},
        {"candidate_id": "c", "subjective_net": -3},
    ]
    gates = {candidate: {"pass": True} for candidate in ("a", "b", "c")}
    assert module.choose_auto_winner(ranked, gates) is None
    ranked[0]["subjective_net"] = 2
    assert module.choose_auto_winner(ranked, gates) == "a"
    gates["a"]["pass"] = False
    assert module.choose_auto_winner(ranked, gates) is None


def test_blind_ready_requires_private_disjoint_case_sets(tmp_path: Path):
    module = load_script("004107_finalize_batch43_pathx_final.py")
    selected = ["r3_step-26000", "r5_step-28000", "r3_step-30000"]
    pages = []
    case_ids_by_candidate = {}
    for page_index, candidate_id in enumerate(selected):
        case_ids = [f"case-{page_index}-{index:02d}" for index in range(20)]
        case_ids_by_candidate[candidate_id] = case_ids
        manifest = tmp_path / "private" / f"page-{page_index}.manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": module.BLIND_SCHEMA,
                    "experiment_id": module.EXPERIMENT_ID,
                    "data_version": "v1_20260709",
                    "candidate_id": candidate_id,
                    "page_id": f"opaque-{page_index}",
                    "cases": [{"case_id": case_id} for case_id in case_ids],
                }
            )
        )
        pages.append(
            {"candidate_id": candidate_id, "page_id": f"opaque-{page_index}", "manifest": str(manifest)}
        )
    ready = tmp_path / "private/BLIND20_READY.json"
    payload = {
        "schema_version": module.BLIND_SCHEMA,
        "experiment_id": module.EXPERIMENT_ID,
        "data_version": "v1_20260709",
        "status": "complete",
        "selected_candidate_ids": selected,
        "case_ids_by_candidate": case_ids_by_candidate,
        "pages": pages,
    }
    ready.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="registered 004104 producer"):
        module.load_blind_ready(ready, selected)
    blind_builder = ROOT / "scripts/004104_build_batch43_best3_blind20.py"
    payload["producer"] = {
        "script": str(blind_builder.resolve()),
        "script_sha256": module.sha256_file(blind_builder),
        "entrypoint": blind_builder.name,
    }
    payload["case_ids_by_candidate"][selected[1]][0] = payload["case_ids_by_candidate"][selected[0]][0]
    ready.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="disjoint"):
        module.load_blind_ready(ready, selected)


def test_best3_materializer_extends_only_registered_step_gate(tmp_path: Path):
    module = load_script("004103_select_batch43_best3.py")
    selected = ["r3_step-26000", "r5_step-28000", "r3_step-30000"]
    selection = {
        "schema_version": module.SCHEMA_VERSION,
        "experiment_id": module.EXPERIMENT_ID,
        "data_version": "v1_20260709",
        "status": "selected",
        "selected_candidate_ids": selected,
        "candidates": [
            {
                "candidate_id": candidate_id,
                "arm": arm,
                "step": step,
                "text_repeat": module.TEXT_REPEATS[arm],
                "train_job_id": module.EXPECTED_TRAIN_JOBS[arm],
                "checkpoint": {
                    "path": str(
                        (
                            ROOT
                            / "outputs/lora_runs"
                            / module.RUN_DIRS[arm]
                            / f"step-{step}"
                        ).resolve()
                    )
                },
                "selected_for_full320": True,
            }
            for candidate_id, arm, step in (
                ("r3_step-26000", "r3", 26000),
                ("r5_step-28000", "r5", 28000),
                ("r3_step-30000", "r3", 30000),
            )
        ],
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts/004105_submit_batch43_best3_full320_qz.sh")],
        cwd=ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "SELECTION_JSON": str(selection_path),
            "PLAN_ROOT": str(tmp_path / "plan"),
            "PLAN_ONLY": "1",
            "PYTHON": sys.executable,
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    materialized = (
        tmp_path / "plan/004101_batch44_v1_best3_steps.materialized.sh"
    ).read_text()
    assert "10000|20000|26000|28000|30000" in materialized
    assert "10000|20000|30000" not in materialized
    assert module.EXPECTED_TRAIN_JOBS["r3"] in materialized
    assert module.EXPECTED_TRAIN_JOBS["r5"] in materialized
    assert module.RUN_DIRS["r3"] in materialized
    assert module.RUN_DIRS["r5"] in materialized
    assert '"schema": "batch44_v1_paired_full320_v1"' in materialized
    for legacy in (
        "ver2_9_5_final_r3_v2_30k",
        "ver2_9_5_final_r5_v2_30k",
        "job-a34d84d4-59cc-4824-b197-0829bfe79004",
        "job-aef79753-7fcd-444e-b94d-3e21eedb2394",
        '"schema": "batch43_paired_full320_v1"',
    ):
        assert legacy not in materialized
    plan = json.loads((tmp_path / "plan/best3_full320_plan.json").read_text())
    assert plan["schema_version"] == "moss_codecvc.batch44_v1_best3_full320_plan.v1"
    assert plan["experiment_id"] == "batch44_v1"
    assert plan["data_version"] == "v1_20260709"
    assert plan["selected_steps"] == [26000, 28000, 30000]
    assert plan["paired_extra_counterparts"] == [
        "r5_step-26000", "r3_step-28000", "r5_step-30000"
    ]


def test_final_publish_chain_revalidates_registered_checkpoint_before_table() -> None:
    submit = (ROOT / "scripts/004108_submit_batch42_pathx_final_strict_qz.sh").read_text()
    publish = (ROOT / "scripts/004109_score_and_publish_batch42_pathx_final.sh").read_text()
    assert "step not in {26000, 28000, 30000}" in submit
    assert "candidate_id = f\"{arm}_step-{step}\"" in submit
    assert "FINAL_SELECTION train_job_id does not match the registered arm" in submit
    assert "CONFIRM_BATCH44_FINAL_STRICT" in submit
    assert "moss_codecvc.batch44_v1_final_selection.v1" in submit
    assert "ver2_9_5_final_r3_v1_30k" in submit
    assert "ver2_9_5_final_r5_v1_30k" in submit
    assert "job-2b91d332-d500-4279-84f9-0a6a81a376aa" in submit
    assert "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c" in submit
    assert "validator.load_final_selection(final_path, verify_checkpoint_hashes=True)" in submit
    assert "SOURCE_FINALIZER_SCRIPT" in submit
    assert "FINALIZER_SCRIPT" in submit
    assert publish.count("validate_inference") == 3  # definition + score + table
    assert "step not in {26000, 28000, 30000}" in publish
    assert "CONFIRM_BATCH44_FINAL_SCORERS" in publish
    assert "moss_codecvc.batch44_v1_final_selection.v1" in publish
    assert "ver2_9_5_final_r3_v1_30k" in publish
    assert "ver2_9_5_final_r5_v1_30k" in publish
    assert "job-2b91d332-d500-4279-84f9-0a6a81a376aa" in publish
    assert "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c" in publish
    assert "validator.load_final_selection(final_path, verify_checkpoint_hashes=True)" in publish
    assert "validate_inference\n\nfor path in" in publish
    assert '"--allow-path-x-final"' in publish

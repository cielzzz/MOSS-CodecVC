from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/004114_watch_batch44_best3_full320.sh"
STEPS = tuple(range(2000, 30001, 2000))
R3_JOB = "job-2b91d332-d500-4279-84f9-0a6a81a376aa"
R5_JOB = "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c"
COMPUTE = "lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122"
SPEC = "67b10bc6-78b0-41a3-aaf4-358eeeb99009"
CODE_ROOT = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
    "MOSS-CodecVC_snapshots/ver23_batch37_eval_20260711_1092820"
)


def write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def make_quick20(project: Path) -> None:
    for step in STEPS:
        record = (
            project
            / f"trainset/qz_jobs/ver23_batch44_quick20_step{step}_20260713"
        )
        record.mkdir(parents=True)
        rows = []
        for arm, train_job in (("r3", R3_JOB), ("r5", R5_JOB)):
            for mode in ("no_text", "text"):
                sim_ref = 0.46 if arm == "r3" else 0.45
                sim_src = 0.40
                rows.append(
                    {
                        "step": step,
                        "arm": arm,
                        "train_job_id": train_job,
                        "mode": mode,
                        "n": 20,
                        "keep": 19,
                        "fail": 0.05,
                        "cer": 0.04,
                        "sim_ref": sim_ref,
                        "sim_src": sim_src,
                        "margin": sim_ref - sim_src,
                        "ref_bound_count": 12,
                        "ref_bound": 0.6,
                        "ref_content_f1": 0.05,
                        "text_en_src_quick_n": 12 if mode == "text" else "",
                        "text_en_src_quick_fail": 0.08 if mode == "text" else "",
                        "text_en_src_scope": "proxy" if mode == "text" else "",
                        "run_id": (
                            f"ver2_9_5_final_{arm}_step-{step}_{mode}_"
                            "quick20_d2d3_seed1234"
                        ),
                        "output_dir": str(project / "unused"),
                    }
                )
        (record / "metrics.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )
        with (record / "metrics.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        (record / "complete.marker").write_text("complete\n", encoding="utf-8")
        fields = (
            "job_name",
            "job_id",
            "step",
            "compute_group",
            "spec",
            "record_root",
            "eval_root",
            "code_root",
        )
        values = {
            "job_name": f"quick-{step}",
            "job_id": f"job-00000000-0000-0000-0000-{step:012d}",
            "step": str(step),
            "compute_group": COMPUTE,
            "spec": SPEC,
            "record_root": str(record.resolve()),
            "eval_root": str(
                (project / "testset/outputs/ver23_batch44_quick20_20260713").resolve()
            ),
            "code_root": str(CODE_ROOT),
        }
        with (record / "submitted_jobs.tsv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerow(values)
        (record / "metrics.md").write_text("fixture\n", encoding="utf-8")
        runner = record / "004110_submit_batch44_v1_quick20_qz.frozen.sh"
        runner.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")

        def artifact(path: Path) -> dict:
            return {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

        completion = record / "COMPLETED.json"
        completion.write_text(
            json.dumps(
                {
                    "schema": "moss_codecvc.batch44_v1_quick20_completion.v1",
                    "status": "complete",
                    "step": step,
                    "record_root": str(record.resolve()),
                    "eval_root": values["eval_root"],
                    "training_jobs": {"r3": R3_JOB, "r5": R5_JOB},
                    "evaluation_job": {
                        "job_name": values["job_name"],
                        "job_id": values["job_id"],
                        "submission_ledger": artifact(record / "submitted_jobs.tsv"),
                    },
                    "resource_contract": {
                        "compute_group": "MTTS-3-2-0715",
                        "compute_group_id": COMPUTE,
                        "spec": SPEC,
                        "instances": 1,
                        "gpus": 8,
                        "gpu_type": "NVIDIA_H200_SXM_141G",
                    },
                    "frozen_runner": artifact(runner),
                    "metrics": {
                        "json": artifact(record / "metrics.json"),
                        "tsv": artifact(record / "metrics.tsv"),
                        "md": artifact(record / "metrics.md"),
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (record / "complete.marker").write_text(
            json.dumps(
                {
                    "schema": "moss_codecvc.batch44_v1_quick20_complete_marker.v1",
                    "status": "complete",
                    "step": step,
                    "completed_json_sha256": hashlib.sha256(
                        completion.read_bytes()
                    ).hexdigest(),
                }
            )
            + "\n",
            encoding="utf-8",
        )


def convert_quick20_record_to_local(project: Path, step: int) -> None:
    record = project / f"trainset/qz_jobs/ver23_batch44_quick20_step{step}_20260713"
    if step >= 10000:
        local_record = (
            project
            / f"trainset/local_jobs/ver23_batch44_quick20_step{step}_20260713"
        )
        local_record.parent.mkdir(parents=True, exist_ok=True)
        record.rename(local_record)
        record = local_record
    (record / "submitted_jobs.tsv").unlink()
    (record / "004110_submit_batch44_v1_quick20_qz.frozen.sh").unlink()
    runtime = record / "LOCAL_RUNTIME.json"
    runner = record / "004117_run_batch44_v1_quick20_local.frozen.sh"
    runner.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    common = record / "004110_batch44_quick20_common.frozen.sh"
    common.write_text("fixture\n", encoding="utf-8")
    helper = record / "batch44_quick20_local_completion.frozen.py"
    helper.write_text("fixture\n", encoding="utf-8")

    def artifact(path: Path) -> dict:
        return {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    fixed_paths = {
        "no_text20": project
        / "testset/validation/ver2_8_timbre_quick20_seedtts_no_text_20260704.jsonl",
        "text_source": project
        / "testset/validation/seedtts_vc_ver2_3_validation.jsonl",
        "text20": record / "ver23_batch44_text_quick20_8cell_20260713.jsonl",
    }
    for item in fixed_paths.values():
        item.parent.mkdir(parents=True, exist_ok=True)
        item.write_text("fixture\n", encoding="utf-8")
    jobs = {"r3": R3_JOB, "r5": R5_JOB}
    checkpoint_payload = {}
    checkpoint_paths = {}
    for arm in ("r3", "r5"):
        checkpoint = (
            project
            / f"outputs/lora_runs/ver2_9_5_final_{arm}_v1_30k/step-{step}"
        )
        files = {}
        for name in (
            "adapter_model.safetensors",
            "adapter_config.json",
            "README.md",
            "timbre_memory_adapter.pt",
            "timbre_memory_config.json",
        ):
            item = checkpoint / name
            item.parent.mkdir(parents=True, exist_ok=True)
            item.write_text("fixture\n", encoding="utf-8")
            files[name] = artifact(item)
        checkpoint_paths[arm] = checkpoint
        checkpoint_payload[arm] = {
            "path": str(checkpoint.resolve()),
            "step": step,
            "training_job_id": jobs[arm],
            "files": files,
        }
    identity = (
        project
        / "trainset/qz_jobs/ver23_batch44_ver2_9_5_final_r3_r5_v1_30k_20260713"
    )
    pair_ledger = identity / "submitted_pair.tsv"
    pair_ledger.parent.mkdir(parents=True, exist_ok=True)
    pair_ledger.write_text("fixture\n", encoding="utf-8")
    train_args = {}
    for arm in ("r3", "r5"):
        item = identity / arm / "train_args_dry_run_core.json"
        item.parent.mkdir(parents=True, exist_ok=True)
        item.write_text("{}\n", encoding="utf-8")
        train_args[arm] = item
    eval_root = project / "testset/outputs/ver23_batch44_quick20_20260713"
    runs = []
    for arm in ("r3", "r5"):
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
                item.write_text("fixture\n", encoding="utf-8")
            runs.append({
                "arm": arm,
                "mode": mode,
                "run_id": run_id,
                "training_job_id": jobs[arm],
                "checkpoint": str(checkpoint_paths[arm].resolve()),
                "output_dir": str(output.resolve()),
                "artifacts": {name: artifact(item) for name, item in outputs.items()},
            })
    gpu_rows = [
        {
            "index": index,
            "uuid": f"GPU-00000000-0000-0000-0000-{index:012d}",
            "name": "NVIDIA GeForce RTX 4090",
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
        "gpu_model": "NVIDIA GeForce RTX 4090",
        "gpus": gpu_rows,
        "scheduling": scheduling,
        "runner": artifact(runner),
        "common_library": artifact(common),
        "completion_helper": artifact(helper),
    }) + "\n", encoding="utf-8")
    completion = record / "COMPLETED.json"
    completion.write_text(
        json.dumps(
            {
                "schema": "moss_codecvc.batch44_v1_quick20_completion.v2",
                "status": "complete",
                "backend": "local",
                "step": step,
                "record_root": str(record.resolve()),
                "eval_root": str(eval_root.resolve()),
                "code_root": str(CODE_ROOT),
                "training_jobs": jobs,
                "execution": {
                    "hostname": "xyzhang-dev--fixture",
                    "gpu_count": 2,
                    "gpu_indices": [0, 1],
                    "gpu_model": "NVIDIA GeForce RTX 4090",
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
                    "json": artifact(record / "metrics.json"),
                    "tsv": artifact(record / "metrics.tsv"),
                    "md": artifact(record / "metrics.md"),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (record / "complete.marker").write_text(
        json.dumps(
            {
                "schema": "moss_codecvc.batch44_v1_quick20_complete_marker.v2",
                "status": "complete",
                "backend": "local",
                "step": step,
                "completed_json_sha256": hashlib.sha256(
                    completion.read_bytes()
                ).hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def make_tools(project: Path) -> dict[str, Path]:
    tools = project / "tools"
    selector = tools / "selector.py"
    submit = tools / "submit.sh"
    builder = tools / "builder.py"
    validator = tools / "validator.py"
    write_executable(
        selector,
        r'''#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--project-root", type=Path, required=True)
p.add_argument("--quick20-stamp")
p.add_argument("--output-json", type=Path, required=True)
p.add_argument("--output-md", type=Path, required=True)
p.add_argument("--no-pending-output", action="store_true")
a = p.parse_args()
marker = os.environ.get("FAKE_SELECTOR_CALLED")
if marker:
    Path(marker).write_text("called\n")
runs = {"r3": "ver2_9_5_final_r3_v1_30k", "r5": "ver2_9_5_final_r5_v1_30k"}
jobs = {"r3": "job-2b91d332-d500-4279-84f9-0a6a81a376aa", "r5": "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c"}
repeats = {"r3": 3, "r5": 5}
selected = ["r3_step-26000", "r5_step-26000", "r3_step-28000"]
rows = []
rank = 0
for step in (26000, 28000, 30000):
    for arm in ("r3", "r5"):
        rank += 1
        cid = f"{arm}_step-{step}"
        rows.append({
            "candidate_id": cid,
            "arm": arm,
            "step": step,
            "rank": rank,
            "text_repeat": repeats[arm],
            "train_job_id": jobs[arm],
            "checkpoint": {"path": str((a.project_root / "outputs/lora_runs" / runs[arm] / f"step-{step}").resolve())},
            "selected_for_full320": cid in selected,
            "quick20": {},
        })
payload = {
    "schema_version": "moss_codecvc.batch44_v1_best3_selection.v1",
    "experiment_id": "batch44_v1",
    "data_version": "v1_20260709",
    "status": "selected",
    "registered_candidate_space": {"arms": ["r3", "r5"], "steps": [26000, 28000, 30000], "candidate_count": 6},
    "selected_candidate_ids": selected,
    "candidates": rows,
}
a.output_json.parent.mkdir(parents=True, exist_ok=True)
a.output_json.write_text(json.dumps(payload, indent=2) + "\n")
a.output_md.write_text("ready\n")
''',
    )
    write_executable(
        submit,
        r'''#!/usr/bin/env bash
set -euo pipefail
printf 'ACTION=%s STEP=%s CONFIRM=%s\n' "${ACTION:-}" "${STEP:-}" "${CONFIRM_LOCAL_FULL320:-}" >> "${FAKE_SUBMIT_LOG:?}"
''',
    )
    write_executable(
        validator,
        r'''from pathlib import Path
import json
def load_best3(path, **kwargs):
    payload = json.loads(Path(path).read_text())
    selected = payload["selected_candidate_ids"]
    rows = {row["candidate_id"]: row for row in payload["candidates"]}
    return payload, selected, rows
def validate_full320_step(**kwargs):
    assert Path(kwargs["completion_path"]).is_file()
    assert Path(kwargs["metrics_path"]).is_file()
    return {}, {}
def load_blind_ready(path, selected, **kwargs):
    payload = json.loads(Path(path).read_text())
    assert payload["selected_candidate_ids"] == selected
    return payload, {}
''',
    )
    write_executable(
        builder,
        r'''#!/usr/bin/env python3
import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--selection", type=Path, required=True)
p.add_argument("--output-root", type=Path, required=True)
p.add_argument("--private-root", type=Path, required=True)
p.add_argument("--candidate-diagnostics", action="append", default=[])
p.add_argument("--candidate-run", action="append", default=[])
p.add_argument("--candidate-completion", action="append", default=[])
a = p.parse_args()
selection = json.loads(a.selection.read_text())
assert len(a.candidate_diagnostics) == 3
assert len(a.candidate_run) == 3
assert len(a.candidate_completion) == 3
a.output_root.mkdir(parents=True, exist_ok=True)
(a.output_root / "index.html").write_text("blind")
a.private_root.mkdir(parents=True, exist_ok=True)
(a.private_root / "BLIND20_READY.json").write_text(json.dumps({"selected_candidate_ids": selection["selected_candidate_ids"]}) + "\n")
''',
    )
    return {
        "selector": selector,
        "submit": submit,
        "builder": builder,
        "validator": validator,
    }


def base_env(project: Path, tools: dict[str, Path]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "BATCH44_BEST3_TEST_MODE": "1",
            "PROJECT_ROOT": str(project),
            "PYTHON": sys.executable,
            "SELECTOR": str(tools["selector"]),
            "LOCAL_FULL_WRAPPER": str(tools["submit"]),
            "BLIND_BUILDER": str(tools["builder"]),
            "FULL_VALIDATOR": str(tools["validator"]),
            "STATE_ROOT": str(project / "state"),
            "BEST3_ROOT": str(project / "best3"),
            "PLAN_ROOT": str(project / "plan"),
            "BLIND_OUTPUT_ROOT": str(project / "blind_web"),
            "BLIND_PRIVATE_ROOT": str(project / "blind_private"),
            "FAKE_SELECTOR_CALLED": str(project / "selector.called"),
            "FAKE_SUBMIT_LOG": str(project / "submit.log"),
            "MODE": "once",
            "ACTION": "plan",
        }
    )
    return env


def run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def make_full320_complete(project: Path, step: int) -> None:
    make_full320_submitted(project, step)
    record = (
        project
        / f"trainset/qz_jobs/ver23_batch44_paired_full320_step{step}_20260713"
    )
    aggregate = (
        project
        / f"testset/outputs/ver23_batch44_paired_full320_20260713/step-{step}/aggregate"
    )
    record.mkdir(parents=True, exist_ok=True)
    aggregate.mkdir(parents=True, exist_ok=True)
    (record / "COMPLETED.json").write_text("{}\n")
    (record / "complete.marker").write_text("complete\n")
    (aggregate / "paired_metrics.json").write_text("[]\n")
    (aggregate / "dual_encoder_cases.csv").write_text("run,mode,case_id\n")


def make_full320_submitted(project: Path, step: int) -> None:
    record = (
        project
        / f"trainset/qz_jobs/ver23_batch44_paired_full320_step{step}_20260713"
    )
    step_root = (
        project
        / f"testset/outputs/ver23_batch44_paired_full320_20260713/step-{step}"
    )
    record.mkdir(parents=True, exist_ok=True)
    fields = (
        "job_name",
        "job_id",
        "step",
        "compute_group",
        "spec",
        "record_root",
        "step_root",
        "code_root",
        "r3_train_job_id",
        "r5_train_job_id",
    )
    row = {
        "job_name": f"full-{step}",
        "job_id": f"job-11111111-1111-1111-1111-{step:012d}",
        "step": str(step),
        "compute_group": COMPUTE,
        "spec": SPEC,
        "record_root": str(record.resolve()),
        "step_root": str(step_root.resolve()),
        "code_root": str(CODE_ROOT),
        "r3_train_job_id": R3_JOB,
        "r5_train_job_id": R5_JOB,
    }
    with (record / "submitted_jobs.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow(row)


def make_local_full320_complete(project: Path, step: int) -> None:
    record = (
        project
        / f"trainset/local_jobs/ver23_batch44_paired_full320_step{step}_20260713"
    )
    aggregate = (
        project
        / f"testset/outputs/ver23_batch44_paired_full320_20260713/step-{step}/aggregate"
    )
    record.mkdir(parents=True, exist_ok=True)
    aggregate.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "completeness_json": aggregate / "completeness.json",
        "dual_encoder_cases_csv": aggregate / "dual_encoder_cases.csv",
        "paired_metrics_json": aggregate / "paired_metrics.json",
        "paired_metrics_tsv": aggregate / "paired_metrics.tsv",
        "paired_metrics_md": aggregate / "paired_metrics.md",
    }
    for name, path in artifacts.items():
        path.write_text("[]\n" if name.endswith("_json") else "fixture\n")
    runner = record / "004118_run_batch44_v1_paired_full320_local.frozen.sh"
    runner.write_text("#!/usr/bin/env bash\nexit 0\n")
    inventory = record / "runtime_gpu_inventory.json"
    inventory.write_text('{"schema":"batch44_local_gpu_inventory_v1"}\n')

    def artifact(path: Path) -> dict:
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    completion = record / "COMPLETED.json"
    completion.write_text(
        json.dumps(
            {
                "schema": "batch44_v1_paired_full320_v1",
                "backend": "local",
                "status": "complete",
                "step": step,
                "execution": {
                    "hostname": "xyzhang-dev--fixture",
                    "gpu_count": 2,
                    "gpu_indices": [0, 1],
                    "gpu_models": [
                        "NVIDIA GeForce RTX 4090",
                        "NVIDIA GeForce RTX 4090",
                    ],
                    "gpu_memory_total_mib": [49140, 49140],
                    "gpu_uuids": [
                        "GPU-00000000-0000-0000-0000-000000000000",
                        "GPU-00000000-0000-0000-0000-000000000001",
                    ],
                    "gpu_inventory": str(inventory.resolve()),
                    "gpu_inventory_sha256": hashlib.sha256(
                        inventory.read_bytes()
                    ).hexdigest(),
                },
                "runner": artifact(runner),
                "artifacts": {
                    name: artifact(path) for name, path in artifacts.items()
                },
            }
        )
        + "\n"
    )
    (record / "complete.marker").write_text(
        "COMPLETED.json sha256\t"
        + hashlib.sha256(completion.read_bytes()).hexdigest()
        + "\n"
    )


def test_waits_for_all_fifteen_quick20_before_selector(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    env = base_env(tmp_path, tools)
    result = run(env)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "selector.called").exists()
    assert not (tmp_path / "submit.log").exists()
    state = json.loads((tmp_path / "state/scan_latest.json").read_text())
    assert state["state"] == "waiting_quick20"


def test_plan_selects_best3_without_touching_qz(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    make_quick20(tmp_path)
    env = base_env(tmp_path, tools)
    result = run(env)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "selector.called").is_file()
    assert not (tmp_path / "submit.log").exists()
    assert (tmp_path / "state/PLAN_READY.json").is_file()
    audit = json.loads((tmp_path / "state/selection_audit.json").read_text())
    assert audit["selected_steps"] == [26000, 28000]
    assert len(audit["selected_candidate_ids"]) == 3


def test_plan_accepts_historical_qz_then_local_4090_quick20_mix(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    make_quick20(tmp_path)
    for step in STEPS[3:]:
        convert_quick20_record_to_local(tmp_path, step)
    env = base_env(tmp_path, tools)
    result = run(env)
    assert result.returncode == 0, result.stderr
    quick_audit = json.loads((tmp_path / "state/quick20_audit.json").read_text())
    assert quick_audit["backend_counts"] == {"local": 12, "qz": 3}
    assert [row["backend"] for row in quick_audit["records"][:3]] == ["qz"] * 3
    assert all(row["backend"] == "local" for row in quick_audit["records"][3:])


def test_local_quick20_with_qz_ledger_is_rejected_before_selector(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    make_quick20(tmp_path)
    convert_quick20_record_to_local(tmp_path, 8000)
    record = tmp_path / "trainset/qz_jobs/ver23_batch44_quick20_step8000_20260713"
    (record / "submitted_jobs.tsv").write_text("forged\n", encoding="utf-8")
    env = base_env(tmp_path, tools)
    result = run(env)
    assert result.returncode == 2
    assert "must not contain a QZ submission ledger" in result.stderr
    assert not (tmp_path / "selector.called").exists()


def test_registered_alert_blocks_before_selection_or_submission(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    make_quick20(tmp_path)
    alert = (
        tmp_path
        / "trainset/qz_jobs/ver23_batch44_quick20_scheduler_20260713/"
        "ALERT_NEGATIVE_NO_TEXT_MARGIN.json"
    )
    alert.parent.mkdir(parents=True)
    alert.write_text("{}\n")
    env = base_env(tmp_path, tools)
    result = run(env)
    assert result.returncode == 20
    assert not (tmp_path / "selector.called").exists()
    assert not (tmp_path / "submit.log").exists()


def test_existing_selected_step_submission_is_never_duplicated(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    make_quick20(tmp_path)
    make_full320_submitted(tmp_path, 26000)
    env = base_env(tmp_path, tools)
    env["ACTION"] = "preflight"
    result = run(env)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "submit.log").exists()
    state = json.loads((tmp_path / "state/scan_latest.json").read_text())
    assert state["state"] == "waiting_existing_full320"


def test_partial_selected_full320_requires_manual_audit_and_never_dispatches(
    tmp_path: Path,
) -> None:
    tools = make_tools(tmp_path)
    make_quick20(tmp_path)
    make_full320_complete(tmp_path, 26000)
    record = tmp_path / "trainset/qz_jobs/ver23_batch44_paired_full320_step26000_20260713"
    (record / "complete.marker").unlink()
    env = base_env(tmp_path, tools)
    env["ACTION"] = "preflight"

    result = run(env)
    assert result.returncode == 21, result.stderr
    assert not (tmp_path / "submit.log").exists()
    state = json.loads((tmp_path / "state/scan_latest.json").read_text())
    assert state["state"] == "manual_full320_audit_required"
    full_state = json.loads((tmp_path / "state/full320_state.json").read_text())
    by_step = {row["step"]: row["state"] for row in full_state["steps"]}
    assert by_step[26000] == "inconsistent_partial_completion"


def test_preflight_calls_local_wrapper_for_each_missing_step_without_live_confirmation(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    make_quick20(tmp_path)
    env = base_env(tmp_path, tools)
    env["ACTION"] = "preflight"
    result = run(env)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "submit.log").read_text().splitlines() == [
        "ACTION=preflight STEP=26000 CONFIRM=",
        "ACTION=preflight STEP=28000 CONFIRM=",
    ]
    assert (tmp_path / "state/PREFLIGHT_COMPLETE.json").is_file()


def test_failed_child_wrapper_never_writes_terminal_success(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    make_quick20(tmp_path)
    write_executable(
        tools["submit"],
        "#!/usr/bin/env bash\necho fake-best3-wrapper-failure >&2\nexit 7\n",
    )
    env = base_env(tmp_path, tools)
    env["ACTION"] = "preflight"

    result = run(env)
    assert result.returncode != 0
    assert "local full320 preflight step-26000 failed rc=7" in result.stderr
    state_root = tmp_path / "state"
    assert not (state_root / "DRY_RUN_COMPLETE.json").exists()
    assert not (state_root / "PREFLIGHT_COMPLETE.json").exists()
    state = json.loads((state_root / "scan_latest.json").read_text())
    assert state["state"] == "full320_preflight_failed"


def test_blind_only_strictly_validates_completed_full320_then_builds_pages(
    tmp_path: Path,
) -> None:
    tools = make_tools(tmp_path)
    make_quick20(tmp_path)
    make_full320_complete(tmp_path, 26000)
    make_full320_complete(tmp_path, 28000)
    env = base_env(tmp_path, tools)
    env["ACTION"] = "blind-only"
    result = run(env)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "submit.log").exists()
    assert (tmp_path / "blind_web/index.html").is_file()
    assert (tmp_path / "blind_private/BLIND20_READY.json").is_file()
    assert (tmp_path / "state/BLIND20_READY.json").is_file()
    state = json.loads((tmp_path / "state/scan_latest.json").read_text())
    assert state["state"] == "blind20_ready"


def test_blind_only_accepts_local_4090_full320_and_binds_local_completions(
    tmp_path: Path,
) -> None:
    tools = make_tools(tmp_path)
    make_quick20(tmp_path)
    make_local_full320_complete(tmp_path, 26000)
    make_local_full320_complete(tmp_path, 28000)
    env = base_env(tmp_path, tools)
    env["ACTION"] = "blind-only"
    result = run(env)
    assert result.returncode == 0, result.stderr
    full_state = json.loads((tmp_path / "state/full320_state.json").read_text())
    assert {row["backend"] for row in full_state["steps"]} == {"local"}
    bindings = (tmp_path / "state/blind20_bindings.tsv").read_text()
    assert "trainset/local_jobs/ver23_batch44_paired_full320_step26000_20260713" in bindings
    assert "trainset/local_jobs/ver23_batch44_paired_full320_step28000_20260713" in bindings


def test_local_full320_with_qz_submission_ledger_requires_manual_audit(
    tmp_path: Path,
) -> None:
    tools = make_tools(tmp_path)
    make_quick20(tmp_path)
    make_local_full320_complete(tmp_path, 26000)
    record = tmp_path / "trainset/local_jobs/ver23_batch44_paired_full320_step26000_20260713"
    (record / "submitted_jobs.tsv").write_text("forged\n")
    env = base_env(tmp_path, tools)
    env["ACTION"] = "blind-only"
    result = run(env)
    assert result.returncode == 21, result.stderr
    state = json.loads((tmp_path / "state/scan_latest.json").read_text())
    assert state["state"] == "manual_full320_audit_required"


def test_test_mode_can_never_enter_local_run_branch(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    make_quick20(tmp_path)
    env = base_env(tmp_path, tools)
    env.update(
        {
            "ACTION": "run",
            "CONFIRM_LOCAL_FULL320_ORCHESTRATOR": "1",
        }
    )
    result = run(env)
    assert result.returncode == 2
    assert "test mode may not execute the local-run branch" in result.stderr
    assert not (tmp_path / "submit.log").exists()


def test_static_contract_keeps_scope_and_mtts_gates() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "004103_select_batch43_best3.py" in text
    assert '--project-root "$PROJECT_ROOT"' in text
    assert "004118_run_batch44_v1_paired_full320_local.sh" in text
    assert "004104_build_batch43_best3_blind20.py" in text
    assert "004107_finalize_batch43_pathx_final.py" in text
    assert "CONFIRM_LOCAL_FULL320_ORCHESTRATOR=1" in text
    assert "CONFIRM_LOCAL_FULL320=1" in text
    assert "ACTION=preflight" in text
    assert "ACTION=run" in text
    assert "two RTX 4090" in text
    assert "lcg-0d3f8d0a-789c-491a-ae24-3f8f2b2f8122" in text
    assert "67b10bc6-78b0-41a3-aaf4-358eeeb99009" in text
    assert "never creates FINAL_SELECTION.json" in text

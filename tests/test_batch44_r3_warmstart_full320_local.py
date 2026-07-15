from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/004127_run_batch44_r3_warmstart_full320_local.sh"
FINALIZER = ROOT / "scripts/batch44_r3_warmstart_full320_finalize.py"
JOB_ID = "job-165f3b1d-8c7d-8882-bdb86ef642ab"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


finalizer = load(FINALIZER, "batch44_r3_warmstart_full320_test")


def write(path: Path, text: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("effective", "local"),
    ((20000, 10000), (24000, 14000), (26000, 16000), (28000, 18000), (30000, 20000)),
)
def test_supported_effective_to_local_mapping(effective: int, local: int) -> None:
    assert finalizer.continuation_local_step(effective) == local
    finalizer.validate_step_mapping(effective, local)


def test_mapping_rejects_unregistered_or_physical_step_drift() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        finalizer.continuation_local_step(22000)
    with pytest.raises(ValueError, match="mapping drift"):
        finalizer.validate_step_mapping(24000, 16000)


def test_runner_is_local_only_confirm_gated_and_syntax_valid() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for token in (
        'ACTION="${ACTION:-plan}"',
        "EFFECTIVE_STEP",
        "CONTINUATION_LOCAL_STEP=$((EFFECTIVE_STEP - 10000))",
        "CONFIRM_LOCAL_R3_FULL320",
        "CONFIRM_EFFECTIVE_STEP",
        "CONFIRM_LOCAL_ONLY",
        "RTX4090x2",
        "WARM_START_CONTRACT_SHA256",
        "NUM_SHARDS=2",
        "ASR_NUM_SHARDS=2",
        "MODE=all",
        "unified_eval_input.jsonl",
    ):
        assert token in source
    assert "qzcli" not in source
    assert "create-job" not in source
    assert subprocess.run(["bash", "-n", str(RUNNER)], check=False).returncode == 0


def test_runner_default_plan_has_no_live_side_effect_gate() -> None:
    result = subprocess.run(
        ["bash", str(RUNNER)],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PATH": "/usr/bin:/bin", "ACTION": "plan", "EFFECTIVE_STEP": "30000"},
    )
    assert result.returncode == 0, result.stderr
    assert "CONTINUATION_LOCAL_STEP=20000" in result.stdout
    assert "no files or GPU work started" in result.stdout


def test_runner_live_requires_all_three_confirms() -> None:
    result = subprocess.run(
        ["bash", str(RUNNER)],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PATH": "/usr/bin:/bin", "ACTION": "run", "EFFECTIVE_STEP": "20000"},
    )
    assert result.returncode != 0
    assert "CONFIRM_LOCAL_R3_FULL320=1" in result.stderr


def make_infer_logs(tmp_path: Path, *, text_bnf: bool = False) -> tuple[list[Path], set[str]]:
    logs = [tmp_path / "infer.shard0.log", tmp_path / "infer.shard1.log"]
    lines: list[list[str]] = [[], []]
    expected_ids: set[str] = set()
    for mode in ("no_text", "text"):
        for index in range(160):
            case_id = f"{mode}-{index:03d}"
            expected_ids.add(case_id)
            shard = index % 2
            lines[shard].append(f"[persistent-valid] run {case_id} mode={mode}")
            if mode == "no_text" or (text_bnf and index == 0):
                lines[shard].append(
                    "[persistent-infer] source semantic memory type=wavlm_bnf_continuous"
                )
            lines[shard].append(f"[persistent-valid] done {case_id} status=ok")
    for path, content in zip(logs, lines):
        write(path, "\n".join(content) + "\n")
    return logs, expected_ids


def test_bnf_audit_proves_no_text_160_and_text_zero(tmp_path: Path) -> None:
    logs, expected_ids = make_infer_logs(tmp_path)
    result = finalizer.parse_bnf_by_mode(logs, expected_ids)
    assert result["run_case_counts"] == {"no_text": 160, "text": 160}
    assert result["bnf_extraction_counts"] == {"no_text": 160, "text": 0}


def test_bnf_audit_rejects_one_text_side_extraction(tmp_path: Path) -> None:
    logs, expected_ids = make_infer_logs(tmp_path, text_bnf=True)
    with pytest.raises(ValueError, match="BNF bypass drift"):
        finalizer.parse_bnf_by_mode(logs, expected_ids)


def metric_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mode in ("no_text", "text"):
        for index in range(160):
            keep = index % 10 != 0
            rows.append(
                {
                    "mode": mode,
                    "cell": (
                        "en_src_ref" if mode == "text" and index < 80 else "zh_src_ref"
                    ),
                    "language": "en" if index % 2 == 0 else "zh",
                    "content_keep": keep,
                    "content_filter_reason": "keep" if keep else "target_too_long",
                    "cer_tgt": 0.1,
                    "wer_tgt": 0.2,
                    "sim_gen_ref": 0.45,
                    "sim_gen_source": 0.40,
                    "ecapa_sim_gen_ref": 0.50,
                    "ecapa_sim_gen_source": 0.25,
                    "asr_tgt_text": "generated",
                    "timbre_ref_text": "reference",
                }
            )
    return rows


def test_metrics_include_qwen_dual_encoder_and_fail_reasons(tmp_path: Path) -> None:
    metrics, reasons = finalizer.build_metrics(
        metric_rows(),
        effective_step=24000,
        local_step=14000,
        train_job_id=JOB_ID,
        checkpoint=tmp_path / "step-14000",
        contract=tmp_path / "warm_start_contract.json",
    )
    assert [row["scope"] for row in metrics] == ["no_text", "text", "all"]
    assert metrics[0]["wavlm_margin"] == pytest.approx(0.05)
    assert metrics[0]["speechbrain_margin"] == pytest.approx(0.25)
    assert metrics[0]["qwen_primary_error"] == pytest.approx(0.15)
    assert metrics[1]["text_en_src_n"] == 80
    assert reasons["all"]["failed_reason_counts"] == {"target_too_long": 32}


def test_unified_input_is_004082_ready_and_preserves_modes(tmp_path: Path) -> None:
    source = write(tmp_path / "source.wav", "a" * 64)
    reference = write(tmp_path / "reference.wav", "b" * 64)
    generated = write(tmp_path / "generated.wav", "c" * 64)
    validation = []
    manifests = {}
    asr = {}
    for index in range(320):
        mode = "no_text" if index < 160 else "text"
        case_id = f"case-{index:03d}"
        validation.append(
            {
                "case_id": case_id,
                "mode": mode,
                "cell": "en_src_ref",
                "source_lang": "en",
                "source_audio": str(source),
                "timbre_ref_audio": str(reference),
                "content_ref_text": "hello",
            }
        )
        manifests[case_id] = {"case_id": case_id, "output_wav": str(generated)}
        asr[case_id] = {
            "case_id": case_id,
            "language": "en",
            "content_ref_text": "hello",
            "target_asr_backend": "qwen_asr",
            "content_keep": True,
            "content_filter_reason": "keep",
        }
    rows = finalizer.build_unified_input_rows(
        validation_rows=validation,
        manifest_by_id=manifests,
        asr_by_id=asr,
        system_id="r3-effective-24000",
        run_id="run",
        effective_step=24000,
        local_step=14000,
        checkpoint=tmp_path / "step-14000",
        contract=tmp_path / "contract.json",
    )
    assert len(rows) == 320
    assert rows[0]["generated_audio"] == str(generated)
    assert rows[0]["reference_audio"] == str(reference)
    assert rows[0]["source_audio"] == str(source)
    assert rows[0]["reference_text"] == "hello"
    assert rows[0]["metadata"]["mode"] == "no_text"
    assert rows[-1]["metadata"]["mode"] == "text"
    assert {row["input_index"] for row in rows} == set(range(320))


def test_contract_sha_is_checked_before_helper_import(tmp_path: Path) -> None:
    contract = write(tmp_path / "contract.json", json.dumps({"x": 1}) + "\n")
    with pytest.raises(ValueError, match="contract SHA256 drift"):
        finalizer.audit_continuation_binding(
            provenance_helper=tmp_path / "does-not-exist.py",
            project_root=tmp_path,
            effective_step=20000,
            checkpoint=tmp_path / "step-10000",
            train_job_id=JOB_ID,
            warm_start_contract=contract,
            expected_contract_sha256="0" * 64,
            min_checkpoint_age_sec=0,
            test_mode=True,
        )


def test_runtime_contract_requires_two_idle_4090s(monkeypatch: pytest.MonkeyPatch) -> None:
    output = "\n".join(
        (
            "0, GPU-a, NVIDIA GeForce RTX 4090, 49140, 460, 575.57",
            "1, GPU-b, NVIDIA GeForce RTX 4090, 49140, 461, 575.57",
        )
    )
    monkeypatch.setattr(finalizer.socket, "gethostname", lambda: "xyzhang-dev--test")
    monkeypatch.setattr(
        finalizer.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=output, stderr=""),
    )
    runtime = finalizer.query_local_runtime(max_initial_memory_mib=2048)
    assert runtime["gpu_indices"] == [0, 1]
    assert runtime["gpu_model"] == "NVIDIA GeForce RTX 4090"

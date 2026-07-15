from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import wave
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/004121_atomic_update_seedtts_listening_page.py"
PREFIX = "window.SEEDTTS_BENCHMARK = "


def _write_wav(path: Path, value: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        sample = int(value).to_bytes(2, "little", signed=True)
        handle.writeframes(sample * 80)


def _payload_bytes(payload: dict) -> bytes:
    return (PREFIX + json.dumps(payload, ensure_ascii=False, indent=2) + ";\n").encode()


def _load_payload(path: Path) -> dict:
    text = path.read_text(encoding="utf-8").strip()
    assert text.startswith(PREFIX)
    body = text[len(PREFIX) :]
    if body.endswith(";"):
        body = body[:-1]
    return json.loads(body)


def _samples(case_ids: list[str], run_id: str) -> list[dict]:
    return [
        {
            "index": index,
            "case_id": case_id,
            "mode": "no_text" if index <= 160 else "text",
            "cell": f"cell_{(index - 1) // 20:02d}",
            "target_audio": f"assets/runs/{run_id}/target/{case_id}.wav",
        }
        for index, case_id in enumerate(case_ids, start=1)
    ]


def _make_fixture(tmp_path: Path) -> dict[str, object]:
    page_dir = tmp_path / "page"
    page_dir.mkdir()
    index_path = page_dir / "index.html"
    index_path.write_text("<html>custom live index</html>\n", encoding="utf-8")

    source = tmp_path / "inputs/source.wav"
    timbre = tmp_path / "inputs/timbre.wav"
    _write_wav(source, 1)
    _write_wav(timbre, 2)
    validation = tmp_path / "fixed320.jsonl"
    validation_rows = []
    case_ids = []
    with validation.open("w", encoding="utf-8") as handle:
        for index in range(320):
            case_id = f"case_{index:06d}"
            mode = "no_text" if index < 160 else "text"
            row = {
                "case_id": case_id,
                "mode": mode,
                "cell": f"cell_{index // 20:02d}",
                "source_audio": str(source),
                "timbre_ref_audio": str(timbre),
                "source_text": f"source {index}",
                "timbre_ref_text": f"reference {index}",
                "text": f"input {index}" if mode == "text" else "<NO_TEXT>",
                "content_ref_text": f"content {index}",
                "eval_text_source": "input_text" if mode == "text" else "source_text",
                "source_lang": "en" if index % 2 == 0 else "zh",
                "ref_lang": "zh" if index % 2 == 0 else "en",
                "source_id": f"src-{index}",
                "ref_id": f"ref-{index}",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            validation_rows.append(row)
            case_ids.append(case_id)
    validation_sha = hashlib.sha256(validation.read_bytes()).hexdigest()

    for kind, target in (("source", source), ("timbre", timbre)):
        anchor_dir = page_dir / "assets" / kind
        anchor_dir.mkdir(parents=True)
        for case_id in case_ids:
            os.symlink(target, anchor_dir / f"{case_id}.wav")
    old_asset = page_dir / "assets/runs/ver23_content_old/target"
    old_asset.mkdir(parents=True)
    (old_asset / "keep.txt").write_text("rollback material\n", encoding="utf-8")

    run_ids = ["ver2_3", "baseline_a", "ver23_content_old", "baseline_b"]
    runs = [
        {
            "run_id": run_id,
            "label": run_id,
            "sentinel": f"preserve-{run_id}",
            "samples": _samples(case_ids, run_id),
        }
        for run_id in run_ids
    ]
    payload = {"dataset": {"name": "fixed", "total": 320}, "runs": runs}
    data_path = page_dir / "benchmark_data.js"
    old_data = _payload_bytes(payload)
    data_path.write_bytes(old_data)

    output_dir = tmp_path / "latest-output"
    output_dir.mkdir()
    manifests = [output_dir / "manifest.shard0.jsonl", output_dir / "manifest.shard1.jsonl"]
    handles = [path.open("w", encoding="utf-8") for path in manifests]
    try:
        for index, case_id in enumerate(case_ids):
            target = output_dir / f"{case_id}.wav"
            _write_wav(target, index % 100)
            row = {
                "case_id": case_id,
                "status": "ok",
                "output_wav": str(target),
                "returncode": 0,
                "elapsed_sec": index / 10,
            }
            handles[index % 2].write(json.dumps(row) + "\n")
    finally:
        for handle in handles:
            handle.close()
    model_path = tmp_path / "model/step-30000"
    model_path.mkdir(parents=True)
    (model_path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    return {
        "page_dir": page_dir,
        "index_path": index_path,
        "data_path": data_path,
        "old_data": old_data,
        "old_payload": payload,
        "validation": validation,
        "validation_sha": validation_sha,
        "case_ids": case_ids,
        "output_dir": output_dir,
        "model_path": model_path,
        "old_asset": old_asset,
    }


def _run(fixture: dict[str, object], *extra: str) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--page-dir",
        str(fixture["page_dir"]),
        "--validation-jsonl",
        str(fixture["validation"]),
        "--expected-validation-sha256",
        str(fixture["validation_sha"]),
        "--output-dir",
        str(fixture["output_dir"]),
        "--run-id",
        "latest_full320",
        "--label",
        "Latest full320",
        "--model-path",
        str(fixture["model_path"]),
        *extra,
    ]
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_plan_is_read_only_and_has_registered_order(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    before_tree = sorted(str(path.relative_to(fixture["page_dir"])) for path in fixture["page_dir"].rglob("*"))
    index_before = fixture["index_path"].read_bytes()
    result = _run(fixture)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["mode"] == "plan"
    assert plan["live_mutations"] is False
    assert plan["case_count"] == 320
    assert plan["removed_runs"] == ["ver23_content_old"]
    assert plan["baseline_relative_order"] == ["baseline_a", "baseline_b"]
    assert plan["final_run_order"] == ["baseline_a", "baseline_b", "ver2_3", "latest_full320"]
    assert plan["commit_order"] == [
        "backup_old_benchmark_data",
        "publish_content_addressed_asset",
        "atomic_replace_benchmark_data",
    ]
    assert plan["invariants"]["all_staged_target_links_resolve"] is True
    assert fixture["data_path"].read_bytes() == fixture["old_data"]
    assert fixture["index_path"].read_bytes() == index_before
    after_tree = sorted(str(path.relative_to(fixture["page_dir"])) for path in fixture["page_dir"].rglob("*"))
    assert after_tree == before_tree


def test_apply_backs_up_then_publishes_assets_and_payload(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    index_before = fixture["index_path"].read_bytes()
    old_by_id = {run["run_id"]: deepcopy(run) for run in fixture["old_payload"]["runs"]}
    result = _run(fixture, "--apply", "--confirm-apply", "ATOMIC_REPLACE")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["mode"] == "applied"
    assert report["live_mutations"] is True
    assert report["asset_installed"] is True
    assert fixture["index_path"].read_bytes() == index_before
    assert not (fixture["page_dir"] / ".benchmark_data.update.lock").exists()

    backup = Path(report["backup_path"])
    assert backup.read_bytes() == fixture["old_data"]
    assert Path(report["live_asset_dir"]).is_dir()
    assert (fixture["old_asset"] / "keep.txt").read_text() == "rollback material\n"

    payload = _load_payload(fixture["data_path"])
    run_ids = [run["run_id"] for run in payload["runs"]]
    assert run_ids == ["baseline_a", "baseline_b", "ver2_3", "latest_full320"]
    by_id = {run["run_id"]: run for run in payload["runs"]}
    assert by_id["baseline_a"] == old_by_id["baseline_a"]
    assert by_id["baseline_b"] == old_by_id["baseline_b"]
    assert by_id["ver2_3"] == old_by_id["ver2_3"]
    latest = by_id["latest_full320"]
    assert latest["publication"]["role"] == "ver23_content_latest"
    assert latest["publication"]["target_audio_count"] == 320
    assert [sample["case_id"] for sample in latest["samples"]] == fixture["case_ids"]
    target_files = list(Path(report["live_asset_dir"]).joinpath("target").iterdir())
    assert len(target_files) == 320
    assert all(path.is_file() and not path.is_symlink() for path in target_files)
    assert (Path(report["live_asset_dir"]) / "ASSET_MANIFEST.json").is_file()


def test_missing_target_fails_before_any_live_mutation(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    missing = fixture["output_dir"] / f"{fixture['case_ids'][17]}.wav"
    missing.unlink()
    before_data = fixture["data_path"].read_bytes()
    before_index = fixture["index_path"].read_bytes()
    before_tree = sorted(str(path.relative_to(fixture["page_dir"])) for path in fixture["page_dir"].rglob("*"))
    result = _run(fixture, "--apply", "--confirm-apply", "ATOMIC_REPLACE")
    assert result.returncode == 2
    assert "missing generated target" in result.stderr
    assert fixture["data_path"].read_bytes() == before_data
    assert fixture["index_path"].read_bytes() == before_index
    after_tree = sorted(str(path.relative_to(fixture["page_dir"])) for path in fixture["page_dir"].rglob("*"))
    assert after_tree == before_tree
    assert not (fixture["page_dir"] / "backups").exists()


def test_broken_shared_anchor_fails_without_relinking_live_page(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    broken = fixture["page_dir"] / "assets/source" / f"{fixture['case_ids'][3]}.wav"
    broken.unlink()
    os.symlink(tmp_path / "does-not-exist.wav", broken)
    before_data = fixture["data_path"].read_bytes()
    before_target = os.readlink(broken)
    result = _run(fixture, "--apply", "--confirm-apply", "ATOMIC_REPLACE")
    assert result.returncode == 2
    assert "broken live source anchor" in result.stderr
    assert fixture["data_path"].read_bytes() == before_data
    assert os.readlink(broken) == before_target
    assert not (fixture["page_dir"] / "backups").exists()


def test_apply_requires_explicit_confirmation(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    result = _run(fixture, "--apply")
    assert result.returncode == 2
    assert "--confirm-apply ATOMIC_REPLACE" in result.stderr
    assert fixture["data_path"].read_bytes() == fixture["old_data"]

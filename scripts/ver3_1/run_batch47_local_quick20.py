#!/usr/bin/env python
"""Run/watch the fixed Batch-47 no_text quick20 evaluation on local RTX 4090s.

The watcher is deliberately checkpoint-marker driven.  It evaluates a step
only after ``step-XXXXXX.ready.json`` is atomically published by the trainer,
and it loads the marker's immutable ``inference_checkpoint`` rather than
``last.pt`` or the optimizer-bearing training checkpoint.

For every 500-step checkpoint it runs two paired lanes with identical cases,
noise seeds, Euler steps and CFG scale:

* primary: EMA weights + speaker CFG 1.5;
* diagnostic: raw weights + speaker CFG 1.5.

The fixed Batch-45 no_text quick20 condition tensors are reused, so evaluating
six checkpoints does not repeatedly extract WavLM semantics or ECAPA speaker
embeddings.  ASR and speaker metrics use the existing repository scripts.

Safety properties
-----------------
* ``plan`` is the default action and starts no GPU process;
* ``once``/``watch`` require ``--allow-run``;
* no QZ command is imported or invoked;
* no signal is ever sent to the training process;
* step-500 is judged only after both lanes finish: EMA-only 100% fail writes
  an EMA-lag warning and continues; EMA+raw both at 100% writes a hard red
  flag and stops this watcher only.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shlex
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
STEPS = (500, 1000, 1500, 2000, 2500, 3000)
READY_SCHEMA = "ver3_1_ddlfm_checkpoint_ready_v1"
TASK_PREFIX = "codecVC-"
QUICK20_SHA256 = "bf69f38ba6f35fe36fcacb229fc4a3633c955a64ac06561ae3ac74cbe4b3c4f2"
SEMANTIC_MANIFEST_SHA256 = "7feabe54cfcf1ee85a3e15b4a3d39432cab4099d3f51022a478e8c4fd6502633"


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    use_ema: bool


VARIANTS = (
    Variant("primary_ema_cfg1p5", "primary EMA + CFG 1.5", True),
    Variant("diagnostic_raw_cfg1p5", "diagnostic raw + CFG 1.5", False),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            yield row


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.fmean(items) if items else None


def parse_args() -> argparse.Namespace:
    conditions = ROOT / "testset/outputs/ver3_1_step4_ddlfm_eval_20260715/conditions"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--action", choices=("plan", "once", "watch"), default="plan")
    ap.add_argument("--allow-run", action="store_true")
    ap.add_argument(
        "--task-name",
        default="codecVC-ver3-1-batch47-local-quick20-20260716",
    )
    ap.add_argument(
        "--checkpoint-root",
        default=str(ROOT / "outputs/ver3_1_batch47_ddlfm_no_text_probe_20260716"),
    )
    ap.add_argument(
        "--output-root",
        default=str(ROOT / "testset/outputs/codecVC-ver3-1-batch47-local-quick20-20260716"),
    )
    ap.add_argument(
        "--record-root",
        default=str(ROOT / "trainset/local_jobs/codecVC-ver3-1-batch47-local-quick20-20260716"),
    )
    ap.add_argument(
        "--validation-jsonl",
        default=str(conditions / "validation_quick20_no_text.jsonl"),
    )
    ap.add_argument(
        "--semantic-manifest",
        default=str(conditions / "no_text_semantic/manifest.jsonl"),
    )
    ap.add_argument(
        "--ecapa-embedding-dir",
        default=str(conditions / "ecapa/embeddings"),
    )
    ap.add_argument(
        "--zq-channel-stats",
        default=str(ROOT / "prepared/zq_targets_v1/channel_stats.pt"),
    )
    ap.add_argument(
        "--adapter-checkpoint",
        default=str(ROOT / "outputs/ver3_1_content_adapter_probe_20260715/step-003000"),
    )
    ap.add_argument(
        "--moss-python",
        default="/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-vc/bin/python",
    )
    ap.add_argument(
        "--asr-python",
        default="/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/bin/python",
    )
    ap.add_argument(
        "--qwen-asr-model",
        default="/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/checkpoint/qwen-asr-1_7b",
    )
    ap.add_argument("--inference-device", default="cuda:0")
    ap.add_argument("--condition-device", default="cuda:1")
    ap.add_argument("--asr-device", default="cuda:0")
    ap.add_argument("--speaker-device", default="cuda:1")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--sampling-steps", type=int, default=20)
    ap.add_argument("--cfg-scale", type=float, default=2.5, help="speaker CFG scale")
    ap.add_argument("--semantic-cfg-scale", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=20260715)
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    ap.add_argument("--max-scans", type=int, default=0, help="0 means no limit in watch mode")
    ap.add_argument("--failure-cer-threshold", type=float, default=0.30)
    ap.add_argument(
        "--continue-after-step500-red-flag",
        action="store_true",
        help="Manual diagnostic override; default watcher stops itself after the red flag.",
    )
    return ap.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "checkpoint_root",
        "output_root",
        "record_root",
        "validation_jsonl",
        "semantic_manifest",
        "ecapa_embedding_dir",
        "zq_channel_stats",
        "adapter_checkpoint",
        "moss_python",
        "asr_python",
        "qwen_asr_model",
    ):
        setattr(args, name, Path(getattr(args, name)).expanduser().resolve())
    return args


def validate_fixed_conditions(args: argparse.Namespace) -> dict[str, Any]:
    if not str(args.task_name).startswith(TASK_PREFIX):
        raise ValueError(f"task name must start with {TASK_PREFIX}")
    if not args.validation_jsonl.is_file():
        raise FileNotFoundError(args.validation_jsonl)
    if sha256_file(args.validation_jsonl) != QUICK20_SHA256:
        raise ValueError("fixed no_text quick20 JSONL SHA256 changed")
    rows = list(iter_jsonl(args.validation_jsonl))
    case_ids = [str(row.get("case_id") or "") for row in rows]
    if len(rows) != 20 or len(set(case_ids)) != 20 or any(not item for item in case_ids):
        raise ValueError("fixed quick20 must contain exactly 20 unique case IDs")
    if any(str(row.get("mode") or "") != "no_text" for row in rows):
        raise ValueError("Batch-47 quick20 must be no_text-only")

    if not args.semantic_manifest.is_file():
        raise FileNotFoundError(args.semantic_manifest)
    if sha256_file(args.semantic_manifest) != SEMANTIC_MANIFEST_SHA256:
        raise ValueError("fixed semantic manifest SHA256 changed")
    semantic: dict[str, Path] = {}
    for row in iter_jsonl(args.semantic_manifest):
        key = str(row.get("case_id") or row.get("sample_id") or row.get("utt_id") or "")
        value = row.get("semantic_v3_1_path") or row.get("semantic_path")
        if key and value:
            semantic[key] = Path(str(value)).expanduser().resolve()
    missing_semantic = [case_id for case_id in case_ids if not semantic.get(case_id, Path()).is_file()]
    if missing_semantic:
        raise FileNotFoundError(f"quick20 semantic conditions missing: {missing_semantic[:3]}")

    if not args.ecapa_embedding_dir.is_dir():
        raise FileNotFoundError(args.ecapa_embedding_dir)
    missing_ecapa = [case_id for case_id in case_ids if not (args.ecapa_embedding_dir / f"{case_id}.pt").is_file()]
    if missing_ecapa:
        raise FileNotFoundError(f"quick20 ECAPA conditions missing: {missing_ecapa[:3]}")
    return {
        "rows": len(rows),
        "case_ids": case_ids,
        "validation_sha256": QUICK20_SHA256,
        "semantic_manifest_sha256": SEMANTIC_MANIFEST_SHA256,
        "semantic_coverage": len(rows),
        "ecapa_coverage": len(rows),
    }


def dependency_status(args: argparse.Namespace) -> dict[str, Any]:
    files = {
        "moss_python": args.moss_python,
        "asr_python": args.asr_python,
        "evaluator": ROOT / "scripts/ver3_1/evaluate_ddlfm_validation.py",
        "eval_input_builder": ROOT / "scripts/004017_build_seedtts_generated_eval_jsonl.py",
        "asr_filter": ROOT / "scripts/001017_asr_content_filter.py",
        "asr_summary": ROOT / "scripts/004042_summarize_seedtts_validation_eval.py",
        "speaker_summary": ROOT / "scripts/004048_summarize_seedtts_ablation_metrics.py",
        "qwen_asr_model": args.qwen_asr_model,
        "adapter_checkpoint": args.adapter_checkpoint,
        "zq_channel_stats": args.zq_channel_stats,
        "moss_audio_tokenizer": Path(
            "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
            "vcdata_construction/MOSS-Audio-Tokenizer"
        ),
        "moss_tts_root": Path(
            "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-TTS"
        ),
        "wavlm_speaker_scorer_root": Path(
            "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction"
        ),
    }
    status = {name: {"path": str(path), "exists": path.exists()} for name, path in files.items()}
    status["checkpoint_root"] = {
        "path": str(args.checkpoint_root),
        "exists": args.checkpoint_root.is_dir(),
    }
    return status


def ready_marker(args: argparse.Namespace, step: int) -> tuple[Path, dict[str, Any], Path] | None:
    marker_path = args.checkpoint_root / f"step-{step:06d}.ready.json"
    if not marker_path.is_file():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    checks = {
        "schema": marker.get("schema") == READY_SCHEMA,
        "status": marker.get("status") == "ready",
        "step": int(marker.get("step", -1)) == int(step),
    }
    checkpoint_value = str(marker.get("inference_checkpoint") or "")
    infer_path = Path(checkpoint_value).expanduser().resolve() if checkpoint_value else Path()
    expected_path = (args.checkpoint_root / f"step-{step:06d}.infer.pt").resolve()
    checks["inference path"] = bool(checkpoint_value) and infer_path == expected_path
    checks["inference file"] = infer_path.is_file()
    if infer_path.is_file():
        checks["inference size"] = int(marker.get("inference_checkpoint_size_bytes", -1)) == infer_path.stat().st_size
    else:
        checks["inference size"] = False
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"invalid ready marker {marker_path}: {', '.join(failed)}")
    return marker_path, marker, infer_path


def run_id(step: int, variant: Variant) -> str:
    return f"codecVC-ver3-1-batch47-step{step:06d}-{variant.key.replace('_', '-')}"


def variant_dir(args: argparse.Namespace, step: int, variant: Variant) -> Path:
    return args.output_root / f"step-{step:06d}" / variant.key


def variant_completion(args: argparse.Namespace, step: int, variant: Variant) -> Path:
    return variant_dir(args, step, variant) / "QUICK20_COMPLETED.json"


def quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in command)


def command_plan(args: argparse.Namespace, checkpoint: Path, step: int, variant: Variant) -> dict[str, list[str]]:
    directory = variant_dir(args, step, variant)
    rid = run_id(step, variant)
    generated = directory / f"{rid}.generated_eval_input.jsonl"
    asr_jsonl = directory / f"{rid}.asr_eval.jsonl"
    metrics_csv = directory / f"{rid}.metrics.csv"
    summary_json = directory / f"{rid}.summary.json"
    paired_csv = directory / f"{rid}.paired_metrics.csv"
    paired_json = directory / f"{rid}.paired_metrics.summary.json"
    paired_md = directory / f"{rid}.paired_metrics.summary.md"
    inference = [
        str(args.moss_python),
        str(ROOT / "scripts/ver3_1/evaluate_ddlfm_validation.py"),
        "--validation-jsonl", str(args.validation_jsonl),
        "--checkpoint", str(checkpoint),
        "--adapter-checkpoint", str(args.adapter_checkpoint),
        "--output-dir", str(directory),
        "--mode", "no_text",
        "--max-cases", "20",
        "--batch-size", str(args.batch_size),
        "--sampling-steps", str(args.sampling_steps),
        "--cfg-scale", str(args.cfg_scale),
        "--semantic-cfg-scale", str(args.semantic_cfg_scale),
        "--zq-channel-stats", str(args.zq_channel_stats),
        "--seed", str(args.seed),
        "--device", str(args.inference_device),
        "--ecapa-device", str(args.condition_device),
        "--semantic-manifest", str(args.semantic_manifest),
        "--ecapa-embedding-dir", str(args.ecapa_embedding_dir),
    ]
    if not variant.use_ema:
        inference.append("--no-ema")
    build = [
        str(args.moss_python),
        str(ROOT / "scripts/004017_build_seedtts_generated_eval_jsonl.py"),
        "--validation-jsonl", str(args.validation_jsonl),
        "--output-dir", str(directory),
        "--manifest-jsonl", str(directory / "manifest.jsonl"),
        "--run-id", rid,
        "--output-jsonl", str(generated),
        "--status", "ok",
    ]
    asr = [
        str(args.asr_python),
        str(ROOT / "scripts/001017_asr_content_filter.py"),
        "--input-jsonl", str(generated),
        "--output-jsonl", str(asr_jsonl),
        "--asr-backend", "qwen_asr",
        "--qwen-asr-model", str(args.qwen_asr_model),
        "--qwen-asr-dtype", "bfloat16",
        "--qwen-asr-max-batch-size", "8",
        "--qwen-asr-max-new-tokens", "256",
        "--device", str(args.asr_device),
        "--content-reference-mode", "text",
        "--skip-source-asr",
        "--zh-cer-threshold", "0.20",
        "--en-wer-threshold", "0.25",
        "--no-text-zh-cer-threshold", "0.35",
        "--no-text-en-wer-threshold", "0.30",
        "--max-repeat-score", "0.30",
        "--progress-every", "5",
        "--overwrite",
    ]
    summarize = [
        str(args.moss_python),
        str(ROOT / "scripts/004042_summarize_seedtts_validation_eval.py"),
        "--asr-jsonl", str(asr_jsonl),
        "--metrics-csv", str(metrics_csv),
        "--summary-md", str(directory / "SUMMARY.md"),
        "--summary-json", str(summary_json),
        "--run-id", rid,
        "--run-label", f"Batch-47 step {step} {variant.label}",
        "--model-path", str(checkpoint),
    ]
    speaker = [
        str(args.asr_python),
        str(ROOT / "scripts/004048_summarize_seedtts_ablation_metrics.py"),
        "--validation-jsonl", str(args.validation_jsonl),
        "--run", f"{rid}={directory}",
        "--output-csv", str(paired_csv),
        "--summary-json", str(paired_json),
        "--summary-md", str(paired_md),
        "--speaker-device", str(args.speaker_device),
        "--failure-cer-threshold", str(args.failure_cer_threshold),
    ]
    return {
        "inference": inference,
        "build_eval_input": build,
        "asr": asr,
        "asr_summary": summarize,
        "speaker_summary": speaker,
    }


def child_environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    sox_lib = Path("/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/vc-benchmark/lib")
    if sox_lib.is_dir():
        env["LD_LIBRARY_PATH"] = str(sox_lib) + (
            os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
    return env


def run_command(args: argparse.Namespace, step: int, variant: Variant, stage: str, command: list[str]) -> None:
    directory = variant_dir(args, step, variant)
    logs = directory / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{stage}.log"
    record = {
        "utc": utc_now(),
        "step": step,
        "variant": variant.key,
        "stage": stage,
        "command": command,
        "log": str(log_path),
    }
    with (logs / "commands.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=child_environment(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(f"{stage} failed with code {process.returncode}; see {log_path}")


def row_count(path: Path) -> int:
    return sum(1 for _ in iter_jsonl(path)) if path.is_file() else 0


def validate_case_set(path: Path, expected: set[str]) -> None:
    rows = list(iter_jsonl(path))
    case_ids = [str(row.get("case_id") or row.get("sample_id") or "") for row in rows]
    if len(rows) != len(expected) or len(set(case_ids)) != len(expected) or set(case_ids) != expected:
        raise ValueError(f"case-ID contract mismatch: {path}")


def loss_trend(checkpoint_root: Path, step: int) -> dict[str, Any]:
    log_path = checkpoint_root / "train_log.jsonl"
    rows: list[tuple[int, float]] = []
    if log_path.is_file():
        for row in iter_jsonl(log_path):
            row_step = int(row.get("step", -1))
            loss = finite(row.get("loss"))
            if 0 < row_step <= step and loss is not None:
                rows.append((row_step, loss))
    start = max(1, step - 499)
    interval = [(s, value) for s, value in rows if start <= s <= step]
    previous = [(s, value) for s, value in rows if max(1, start - 500) <= s < start]
    values = [value for _, value in interval]
    previous_values = [value for _, value in previous]
    slope_per_100: float | None = None
    if len(interval) >= 2:
        xs = [float(s) for s, _ in interval]
        ys = values
        x_mean = statistics.fmean(xs)
        y_mean = statistics.fmean(ys)
        denom = sum((x - x_mean) ** 2 for x in xs)
        if denom > 0:
            slope_per_100 = 100.0 * sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    current_mean = mean(values)
    previous_mean = mean(previous_values)
    delta = current_mean - previous_mean if current_mean is not None and previous_mean is not None else None
    if slope_per_100 is None:
        direction = "insufficient"
    elif slope_per_100 < -1.0e-4:
        direction = "down"
    elif slope_per_100 > 1.0e-4:
        direction = "up"
    else:
        direction = "flat"
    return {
        "train_log": str(log_path),
        "interval": [start, step],
        "points": len(interval),
        "mean": current_mean,
        "last": values[-1] if values else None,
        "previous_interval_mean": previous_mean,
        "delta_vs_previous_interval": delta,
        "slope_per_100_steps": slope_per_100,
        "direction": direction,
    }


def load_variant_metrics(args: argparse.Namespace, step: int, variant: Variant) -> dict[str, Any]:
    directory = variant_dir(args, step, variant)
    rid = run_id(step, variant)
    asr_summary_path = directory / f"{rid}.summary.json"
    paired_path = directory / f"{rid}.paired_metrics.summary.json"
    asr_summary = json.loads(asr_summary_path.read_text(encoding="utf-8"))
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    run = (paired.get("runs") or {}).get(rid)
    if not isinstance(run, dict):
        raise ValueError(f"paired summary has no run {rid}")
    sim_ref = finite(run.get("sim_gen_ref_mean"))
    sim_src = finite(run.get("sim_gen_source_mean"))
    failure = finite(run.get("failure_rate_cer_gt_threshold"))
    cer = finite((asr_summary.get("overall") or {}).get("cer"))
    if None in (sim_ref, sim_src, failure, cer):
        raise ValueError(f"incomplete metrics for {rid}")
    return {
        "run_id": rid,
        "step": step,
        "variant": variant.key,
        "label": variant.label,
        "weights": "ema" if variant.use_ema else "raw",
        "cfg_scale": float(args.cfg_scale),
        "semantic_cfg_scale": float(args.semantic_cfg_scale),
        "n": int(run.get("n", 0)),
        "sim_ref": sim_ref,
        "sim_src": sim_src,
        "margin": sim_ref - sim_src,
        "cer": cer,
        "fail_rate": failure,
        "failure_cer_threshold": float(args.failure_cer_threshold),
        "loss_trend": loss_trend(args.checkpoint_root, step),
        "asr_summary": str(asr_summary_path),
        "paired_summary": str(paired_path),
    }


def run_variant(
    args: argparse.Namespace,
    marker_path: Path,
    marker: dict[str, Any],
    checkpoint: Path,
    step: int,
    variant: Variant,
) -> dict[str, Any]:
    completion = variant_completion(args, step, variant)
    if completion.is_file():
        payload = json.loads(completion.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            raise ValueError(f"invalid existing completion: {completion}")
        if Path(str(payload.get("inference_checkpoint") or "")).expanduser().resolve() != checkpoint:
            raise ValueError(f"existing completion checkpoint mismatch: {completion}")
        if payload.get("ready_marker_sha256") != sha256_file(marker_path):
            raise ValueError(f"existing completion ready-marker mismatch: {completion}")
        return payload["metrics"]

    directory = variant_dir(args, step, variant)
    directory.mkdir(parents=True, exist_ok=True)
    evaluator_completion = directory / "COMPLETED.json"
    commands = command_plan(args, checkpoint, step, variant)
    expected_cases = {
        str(row.get("case_id") or "")
        for row in iter_jsonl(args.validation_jsonl)
    }
    if not evaluator_completion.is_file():
        if (directory / "manifest.jsonl").exists():
            raise RuntimeError(
                f"partial inference output exists without COMPLETED.json: {directory}; "
                "preserve it for diagnosis and choose an explicit recovery action"
            )
        run_command(args, step, variant, "inference", commands["inference"])
    if row_count(directory / "manifest.jsonl") != 20:
        raise ValueError(f"inference manifest is not 20 rows: {directory}")
    validate_case_set(directory / "manifest.jsonl", expected_cases)

    rid = run_id(step, variant)
    generated = directory / f"{rid}.generated_eval_input.jsonl"
    asr_jsonl = directory / f"{rid}.asr_eval.jsonl"
    summary_json = directory / f"{rid}.summary.json"
    paired_json = directory / f"{rid}.paired_metrics.summary.json"
    if row_count(generated) != 20:
        run_command(args, step, variant, "build_eval_input", commands["build_eval_input"])
    validate_case_set(generated, expected_cases)
    if row_count(asr_jsonl) != 20:
        run_command(args, step, variant, "asr", commands["asr"])
    validate_case_set(asr_jsonl, expected_cases)
    if not summary_json.is_file():
        run_command(args, step, variant, "asr_summary", commands["asr_summary"])
    if not paired_json.is_file():
        run_command(args, step, variant, "speaker_summary", commands["speaker_summary"])

    metrics = load_variant_metrics(args, step, variant)
    if metrics["n"] != 20:
        raise ValueError(f"speaker summary is not 20 rows: {metrics['n']}")
    evaluator = json.loads(evaluator_completion.read_text(encoding="utf-8"))
    evaluator_checkpoint = Path(str(evaluator.get("checkpoint") or "")).expanduser().resolve()
    evaluator_validation = Path(str(evaluator.get("validation_jsonl") or "")).expanduser().resolve()
    if evaluator_checkpoint != checkpoint:
        raise ValueError(f"evaluator checkpoint mismatch: {evaluator_checkpoint} != {checkpoint}")
    if evaluator_validation != args.validation_jsonl:
        raise ValueError(f"evaluator validation mismatch: {evaluator_validation} != {args.validation_jsonl}")
    if int(evaluator.get("rows", -1)) != 20 or int((evaluator.get("by_mode") or {}).get("no_text", -1)) != 20:
        raise ValueError("evaluator completion does not contain exactly 20 no_text rows")
    if int(evaluator.get("sampling_steps", -1)) != int(args.sampling_steps):
        raise ValueError("Euler sampling-step mismatch in evaluator completion")
    if int(evaluator.get("seed", -1)) != int(args.seed):
        raise ValueError("sampling seed mismatch in evaluator completion")
    expected_ema = bool(variant.use_ema)
    if bool(evaluator.get("using_ema")) != expected_ema:
        raise ValueError(
            f"weight lane mismatch for {variant.key}: using_ema={evaluator.get('using_ema')}"
        )
    if abs(float(evaluator.get("speaker_cfg_scale", evaluator.get("cfg_scale"))) - float(args.cfg_scale)) > 1.0e-9:
        raise ValueError("CFG scale mismatch in evaluator completion")
    if abs(float(evaluator.get("semantic_cfg_scale", 0.0)) - float(args.semantic_cfg_scale)) > 1.0e-9:
        raise ValueError("semantic CFG scale mismatch in evaluator completion")
    payload = {
        "schema": "ver3_1_batch47_local_quick20_completion_v1",
        "status": "completed",
        "completed_at_utc": utc_now(),
        "task_name": args.task_name,
        "ready_marker_path": str(marker_path),
        "ready_marker_sha256": sha256_file(marker_path),
        "ready_marker": marker,
        "inference_checkpoint": str(checkpoint),
        "inference_checkpoint_size_bytes": checkpoint.stat().st_size,
        "metrics": metrics,
    }
    atomic_json(completion, payload)
    return metrics


def completed_metrics(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in STEPS:
        for variant in VARIANTS:
            path = variant_completion(args, step, variant)
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("status") == "completed" and isinstance(payload.get("metrics"), dict):
                    rows.append(payload["metrics"])
    return rows


def write_aggregate(args: argparse.Namespace) -> None:
    rows = completed_metrics(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "ver3_1_batch47_local_quick20_aggregate_v1",
        "updated_at_utc": utc_now(),
        "task_name": args.task_name,
        "checkpoint_root": str(args.checkpoint_root),
        "validation_jsonl": str(args.validation_jsonl),
        "steps": list(STEPS),
        "variants": [variant.key for variant in VARIANTS],
        "rows": rows,
    }
    atomic_json(args.output_root / "summary.json", payload)
    with (args.output_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "step", "variant", "weights", "cfg_scale", "semantic_cfg_scale", "n", "sim_ref", "sim_src",
            "margin", "cer", "fail_rate", "loss_mean", "loss_last",
            "loss_slope_per_100", "loss_direction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (int(item["step"]), str(item["variant"]))):
            trend = row.get("loss_trend") or {}
            writer.writerow({
                "step": row.get("step"),
                "variant": row.get("variant"),
                "weights": row.get("weights"),
                "cfg_scale": row.get("cfg_scale"),
                "semantic_cfg_scale": row.get("semantic_cfg_scale"),
                "n": row.get("n"),
                "sim_ref": row.get("sim_ref"),
                "sim_src": row.get("sim_src"),
                "margin": row.get("margin"),
                "cer": row.get("cer"),
                "fail_rate": row.get("fail_rate"),
                "loss_mean": trend.get("mean"),
                "loss_last": trend.get("last"),
                "loss_slope_per_100": trend.get("slope_per_100_steps"),
                "loss_direction": trend.get("direction"),
            })
    lines = [
        "# Batch-47 local quick20",
        "",
        f"Task: `{args.task_name}`  ",
        "Protocol: fixed 20 no_text cases, Euler-20, speaker CFG 2.5 + semantic CFG 2.0, primary EMA + raw diagnostic.",
        "",
        "| Step | Lane | SIM(ref) | SIM(src) | Margin | CER | Fail | Loss mean | Loss slope/100 | Trend |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda item: (int(item["step"]), str(item["variant"]))):
        trend = row.get("loss_trend") or {}
        fmt = lambda value: "" if value is None else f"{float(value):.4f}"
        lines.append(
            "| {step} | {lane} | {ref} | {src} | {margin} | {cer} | {fail:.1%} | {loss} | {slope} | {direction} |".format(
                step=row["step"], lane=row["weights"], ref=fmt(row["sim_ref"]),
                src=fmt(row["sim_src"]), margin=fmt(row["margin"]), cer=fmt(row["cer"]),
                fail=float(row["fail_rate"]), loss=fmt(trend.get("mean")),
                slope=fmt(trend.get("slope_per_100_steps")), direction=trend.get("direction", ""),
            )
        )
    (args.output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def red_flag_path(args: argparse.Namespace) -> Path:
    return args.record_root / "RED_FLAG_STEP500_100PCT_FAIL.json"


def ema_lag_warning_path(args: argparse.Namespace) -> Path:
    return args.record_root / "EMA_LAG_WARNING_STEP500.json"


def evaluate_step500_gate(args: argparse.Namespace) -> str:
    metrics: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        path = variant_completion(args, 500, variant)
        if not path.is_file():
            return "incomplete"
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics[variant.key] = payload["metrics"]
    primary = metrics[VARIANTS[0].key]
    diagnostic = metrics[VARIANTS[1].key]
    ema_all_fail = float(primary["fail_rate"]) >= 1.0 - 1.0e-12
    raw_all_fail = float(diagnostic["fail_rate"]) >= 1.0 - 1.0e-12
    if not ema_all_fail:
        return "pass"
    if not raw_all_fail:
        payload = {
        "schema": "ver3_1_batch47_step500_ema_lag_warning_v1",
            "status": "ema_lag_warning",
            "created_at_utc": utc_now(),
            "reason": "primary EMA+CFG1.5 is 100% fail but raw+CFG1.5 is not",
            "interpretation": "short-probe EMA lag; preserve the raw diagnostic and continue the watcher",
            "action": "continue later checkpoints; training is not stopped or signalled",
            "task_name": args.task_name,
            "step": 500,
            "metrics": metrics,
        }
        atomic_json(ema_lag_warning_path(args), payload)
        atomic_json(args.output_root / "EMA_LAG_WARNING_STEP500.json", payload)
        return "ema_lag_warning"
    payload = {
        "schema": "ver3_1_batch47_step500_red_flag_v1",
        "status": "red_flag",
        "created_at_utc": utc_now(),
        "reason": "both EMA+speakerCFG2.5+semanticCFG2.0 and raw lanes have 100% fail at step 500",
        "action": "stop this evaluation watcher and report; training is not stopped or signalled",
        "task_name": args.task_name,
        "step": 500,
        "metrics": metrics,
    }
    atomic_json(red_flag_path(args), payload)
    atomic_json(args.output_root / "RED_FLAG_STEP500_100PCT_FAIL.json", payload)
    return "hard_red_flag"


def plan_payload(args: argparse.Namespace, conditions: dict[str, Any]) -> dict[str, Any]:
    dependencies = dependency_status(args)
    steps: list[dict[str, Any]] = []
    for step in STEPS:
        ready = ready_marker(args, step)
        checkpoint = ready[2] if ready else args.checkpoint_root / f"step-{step:06d}.infer.pt"
        steps.append({
            "step": step,
            "ready_marker": str(args.checkpoint_root / f"step-{step:06d}.ready.json"),
            "ready": ready is not None,
            "inference_checkpoint": str(checkpoint),
            "lanes": [
                {
                    "variant": variant.key,
                    "weights": "ema" if variant.use_ema else "raw",
                    "completed": variant_completion(args, step, variant).is_file(),
                    "output_dir": str(variant_dir(args, step, variant)),
                    "commands": {
                        name: quote_command(command)
                        for name, command in command_plan(args, checkpoint, step, variant).items()
                    },
                }
                for variant in VARIANTS
            ],
        })
    return {
        "schema": "ver3_1_batch47_local_quick20_plan_v1",
        "generated_at_utc": utc_now(),
        "action": args.action,
        "dry_run": args.action == "plan",
        "task_name": args.task_name,
        "backend": "local RTX4090 only",
        "checkpoint_trigger": "step-XXXXXX.ready.json -> inference_checkpoint",
        "never_uses": ["last.pt", "QZ evaluation", "training process signals"],
        "conditions": conditions,
        "dependencies": dependencies,
        "steps": steps,
        "step500_red_flag": {
            "hard_criterion": "EMA+CFG1.5 and raw+CFG1.5 both have fail_rate == 100%",
            "ema_lag_warning": "EMA fail_rate == 100% but raw fail_rate < 100%; continue watcher",
            "requires_both_lanes": True,
            "stops_watcher_only": True,
            "kills_training": False,
        },
    }


def acquire_lock(args: argparse.Namespace):
    args.record_root.mkdir(parents=True, exist_ok=True)
    path = args.record_root / "watcher.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"another Batch-47 local quick20 watcher holds {path}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "started_at_utc": utc_now(), "task": args.task_name}) + "\n")
    handle.flush()
    return handle


def scan_once(args: argparse.Namespace) -> str:
    if red_flag_path(args).is_file() and not args.continue_after_step500_red_flag:
        print(f"[batch47-local-quick20] red flag already exists: {red_flag_path(args)}", flush=True)
        return "red_flag"
    if not args.zq_channel_stats.is_file():
        print(
            f"[batch47-local-quick20] waiting for canonical zq stats {args.zq_channel_stats}",
            flush=True,
        )
        return "waiting"
    for step in STEPS:
        if all(variant_completion(args, step, variant).is_file() for variant in VARIANTS):
            continue
        ready = ready_marker(args, step)
        if ready is None:
            print(
                f"[batch47-local-quick20] waiting for atomic ready marker "
                f"{args.checkpoint_root / f'step-{step:06d}.ready.json'}",
                flush=True,
            )
            return "waiting"
        marker_path, marker, checkpoint = ready
        print(f"[batch47-local-quick20] step={step} checkpoint={checkpoint}", flush=True)
        for variant in VARIANTS:
            if variant_completion(args, step, variant).is_file():
                continue
            print(f"[batch47-local-quick20] run {variant.label}", flush=True)
            metrics = run_variant(args, marker_path, marker, checkpoint, step, variant)
            trend = metrics.get("loss_trend") or {}
            print(
                "[batch47-local-quick20] "
                f"step={step} weights={metrics['weights']} "
                f"sim_ref={float(metrics['sim_ref']):.4f} "
                f"sim_src={float(metrics['sim_src']):.4f} "
                f"margin={float(metrics['margin']):+.4f} "
                f"cer={float(metrics['cer']):.4f} "
                f"fail={float(metrics['fail_rate']):.1%} "
                f"loss_mean={trend.get('mean')} "
                f"loss_slope_per_100={trend.get('slope_per_100_steps')} "
                f"loss_trend={trend.get('direction')}",
                flush=True,
            )
        write_aggregate(args)
        if step == 500:
            gate = evaluate_step500_gate(args)
            if gate == "ema_lag_warning":
                print(
                    "[batch47-local-quick20] EMA_LAG_WARNING: step500 EMA fail=100% "
                    "but raw<100%; watcher continues",
                    flush=True,
                )
            if gate == "hard_red_flag" and not args.continue_after_step500_red_flag:
                print(
                    "[batch47-local-quick20] HARD RED FLAG: step500 EMA and raw both fail=100%; "
                    "watcher stops itself, training remains untouched",
                    flush=True,
                )
                return "red_flag"
        return "evaluated"
    write_aggregate(args)
    return "complete"


def main() -> int:
    args = resolve_args(parse_args())
    conditions = validate_fixed_conditions(args)
    plan = plan_payload(args, conditions)
    if args.action == "plan":
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if not args.allow_run:
        raise PermissionError("once/watch requires --allow-run")
    missing = [
        name for name, item in dependency_status(args).items()
        if name not in {"checkpoint_root", "zq_channel_stats"} and not bool(item.get("exists"))
    ]
    if missing:
        raise FileNotFoundError("runtime dependencies are missing: " + ", ".join(missing))
    lock = acquire_lock(args)
    try:
        atomic_json(args.record_root / "plan.json", plan)
        if args.action == "once":
            state = scan_once(args)
            print(f"[batch47-local-quick20] state={state}", flush=True)
            return 0
        scans = 0
        while True:
            scans += 1
            state = scan_once(args)
            print(f"[batch47-local-quick20] scan={scans} state={state}", flush=True)
            if state in {"complete", "red_flag"}:
                return 0
            if args.max_scans > 0 and scans >= args.max_scans:
                return 0
            time.sleep(max(1.0, float(args.poll_seconds)))
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())

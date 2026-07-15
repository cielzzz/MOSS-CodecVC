#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import soundfile as sf


REQUIRED_CHECKPOINT_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "README.md",
    "timbre_memory_adapter.pt",
    "timbre_memory_config.json",
)


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(row)
    return rows


def truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "keep"}


def finite(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: non-numeric value {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label}: non-finite value {value!r}")
    return result


def normalize(text: Any) -> str:
    return "".join(str(text or "").lower().split())


def lcs_len(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for index, cb in enumerate(b, start=1):
            cur.append(prev[index - 1] + 1 if ca == cb else max(prev[index], cur[-1]))
        prev = cur
    return prev[-1]


def ref_f1(row: dict[str, Any]) -> float:
    generated = normalize(row.get("asr_tgt_text"))
    reference = normalize(row.get("timbre_ref_text"))
    hit = lcs_len(generated, reference)
    precision = hit / max(1, len(generated))
    recall = hit / max(1, len(reference))
    return 0.0 if precision + recall <= 0 else 2 * precision * recall / (precision + recall)


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ValueError(f"missing/empty artifact: {path}")
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def validate_audio(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 1024:
        raise ValueError(f"missing/small generated audio: {path}")
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != 24000 or data.shape[1] != 1 or data.shape[0] <= 0:
        raise ValueError(
            f"unexpected audio format {path}: sr={sample_rate} shape={data.shape}"
        )
    if not np.isfinite(data).all():
        raise ValueError(f"non-finite audio samples: {path}")
    return {"frames": int(data.shape[0]), "sample_rate": sample_rate, "channels": 1}


def build_metrics(
    rows: list[dict[str, Any]],
    *,
    arm: str,
    step: int,
    train_job_id: str,
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for scope in ("no_text", "text", "all"):
        selected = rows if scope == "all" else [row for row in rows if row["mode"] == scope]
        expected = 320 if scope == "all" else 160
        if len(selected) != expected:
            raise ValueError(f"{scope}: rows={len(selected)}, expected {expected}")
        keep = sum(truth(row.get("content_keep")) for row in selected)
        wavlm_ref = [finite(row.get("sim_gen_ref"), label="sim_gen_ref") for row in selected]
        wavlm_src = [finite(row.get("sim_gen_source"), label="sim_gen_source") for row in selected]
        sb_ref = [finite(row.get("ecapa_sim_gen_ref"), label="ecapa_sim_gen_ref") for row in selected]
        sb_src = [finite(row.get("ecapa_sim_gen_source"), label="ecapa_sim_gen_source") for row in selected]
        en_src = [
            row
            for row in selected
            if row.get("mode") == "text" and str(row.get("cell") or "").startswith("en_src_")
        ]
        if scope == "text" and len(en_src) != 80:
            raise ValueError(f"text en_src rows={len(en_src)}, expected 80")
        metrics.append(
            {
                "arm": arm,
                "step": step,
                "text_repeat": 3 if arm == "r3" else 5,
                "train_job_id": train_job_id,
                "scope": scope,
                "n": len(selected),
                "keep": keep,
                "fail_count": len(selected) - keep,
                "fail_rate": (len(selected) - keep) / len(selected),
                "cer": mean(finite(row.get("cer_tgt"), label="cer_tgt") for row in selected),
                "wavlm_sim_ref": mean(wavlm_ref),
                "wavlm_sim_src": mean(wavlm_src),
                "wavlm_margin": mean(wavlm_ref) - mean(wavlm_src),
                "wavlm_ref_bound": sum(ref - src > 0.05 for ref, src in zip(wavlm_ref, wavlm_src)) / len(selected),
                "speechbrain_sim_ref": mean(sb_ref),
                "speechbrain_sim_src": mean(sb_src),
                "speechbrain_margin": mean(sb_ref) - mean(sb_src),
                "speechbrain_ref_bound": sum(ref - src > 0.05 for ref, src in zip(sb_ref, sb_src)) / len(selected),
                "ref_content_lcs_f1": mean(ref_f1(row) for row in selected),
                "text_en_src_n": len(en_src) if en_src else "",
                "text_en_src_fail_count": sum(not truth(row.get("content_keep")) for row in en_src) if en_src else "",
                "text_en_src_fail_rate": (sum(not truth(row.get("content_keep")) for row in en_src) / len(en_src)) if en_src else "",
                "text_en_src_cer": mean(finite(row.get("cer_tgt"), label="en_src.cer_tgt") for row in en_src) if en_src else "",
            }
        )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--record-root", required=True)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm", choices=("r3", "r5"), required=True)
    parser.add_argument("--step", type=int, choices=(4000, 6000, 8000), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--train-job-id", required=True)
    parser.add_argument("--runtime-json", required=True)
    parser.add_argument("--runner", required=True)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve(strict=True)
    record_root = Path(args.record_root).resolve(strict=True)
    eval_root = Path(args.eval_root).resolve(strict=True)
    output_dir = Path(args.output_dir).resolve(strict=True)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    validation_path = Path(args.validation_jsonl).resolve(strict=True)
    code_root = Path(args.code_root).resolve(strict=True)
    runtime_path = Path(args.runtime_json).resolve(strict=True)
    runner_path = Path(args.runner).resolve(strict=True)
    completion_path = record_root / "COMPLETED.json"
    marker_path = record_root / "complete.marker"
    if completion_path.exists() or completion_path.is_symlink() or marker_path.exists() or marker_path.is_symlink():
        raise ValueError("completion evidence already exists")

    validation = read_jsonl(validation_path)
    expected_ids = {str(row.get("case_id") or "") for row in validation}
    expected_modes = Counter(str(row.get("mode") or "") for row in validation)
    if len(validation) != 320 or len(expected_ids) != 320 or "" in expected_ids:
        raise ValueError("validation set is not unique 320")
    if expected_modes != Counter({"no_text": 160, "text": 160}):
        raise ValueError(f"validation mode drift: {expected_modes}")

    manifest_paths = sorted(output_dir.glob("manifest.shard*.jsonl"))
    if len(manifest_paths) != 2:
        raise ValueError(f"expected two manifests, got {manifest_paths}")
    manifest_rows = [row for path in manifest_paths for row in read_jsonl(path)]
    manifest_ids = [str(row.get("case_id") or "") for row in manifest_rows]
    statuses = Counter(str(row.get("status") or "") for row in manifest_rows)
    if len(manifest_rows) != 320 or len(set(manifest_ids)) != 320 or set(manifest_ids) != expected_ids:
        raise ValueError("inference manifest is not the canonical 320")
    if statuses != Counter({"ok": 320}):
        raise ValueError(f"inference statuses are not all ok: {statuses}")
    audio_frames = 0
    for row in manifest_rows:
        wav = Path(str(row.get("output_wav") or ""))
        info = validate_audio(wav)
        audio_frames += info["frames"]

    asr_path = output_dir / f"{args.run_id}.asr_eval.jsonl"
    asr_rows = read_jsonl(asr_path)
    asr_by_id = {str(row.get("case_id") or row.get("sample_id") or ""): row for row in asr_rows}
    if len(asr_rows) != 320 or len(asr_by_id) != 320 or set(asr_by_id) != expected_ids:
        raise ValueError("ASR result is not the canonical 320")

    summary_path = output_dir / f"{args.run_id}.summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary["overall"]["n"]) != 320:
        raise ValueError("summary overall n != 320")
    for mode in ("no_text", "text"):
        if int(summary["by_mode"][mode]["n"]) != 160:
            raise ValueError(f"summary {mode} n != 160")

    aggregate = eval_root / "aggregate"
    dual_path = aggregate / "dual_encoder_cases.csv"
    with dual_path.open(encoding="utf-8", newline="") as handle:
        dual_rows = list(csv.DictReader(handle))
    dual_ids = {str(row.get("case_id") or "") for row in dual_rows}
    if len(dual_rows) != 320 or len(dual_ids) != 320 or dual_ids != expected_ids:
        raise ValueError("dual-encoder rows are not the canonical 320")
    if any(row.get("run") != args.run_id for row in dual_rows):
        raise ValueError("dual-encoder run-id drift")
    for row in dual_rows:
        row["timbre_ref_text"] = asr_by_id[row["case_id"]].get("timbre_ref_text")

    metrics = build_metrics(
        dual_rows,
        arm=args.arm,
        step=args.step,
        train_job_id=args.train_job_id,
    )
    metrics_json = aggregate / "metrics.json"
    metrics_tsv = aggregate / "metrics.tsv"
    metrics_md = aggregate / "metrics.md"
    write_atomic(metrics_json, json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    with metrics_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(metrics)

    def fmt(value: Any, *, percent: bool = False) -> str:
        if value == "" or value is None:
            return "—"
        return f"{100 * float(value):.2f}%" if percent else f"{float(value):.4f}"

    lines = [
        f"# Batch-44 early-best full320: {args.arm} step-{args.step}",
        "",
        "| Scope | n | fail | CER | WavLM ref | WavLM src | margin | ref-bound | SpB ref | SpB src | SpB ref-bound | F1(ref-content) | en_src fail | en_src CER |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['scope']} | {row['n']} | {fmt(row['fail_rate'], percent=True)} | {fmt(row['cer'])} | "
            f"{fmt(row['wavlm_sim_ref'])} | {fmt(row['wavlm_sim_src'])} | {fmt(row['wavlm_margin'])} | "
            f"{fmt(row['wavlm_ref_bound'], percent=True)} | {fmt(row['speechbrain_sim_ref'])} | "
            f"{fmt(row['speechbrain_sim_src'])} | {fmt(row['speechbrain_ref_bound'], percent=True)} | "
            f"{fmt(row['ref_content_lcs_f1'])} | {fmt(row['text_en_src_fail_rate'], percent=True)} | "
            f"{fmt(row['text_en_src_cer'])} |"
        )
    write_atomic(metrics_md, "\n".join(lines) + "\n")

    infer_logs = sorted((output_dir / "logs").glob("infer.shard*.log"))
    if len(infer_logs) != 2:
        raise ValueError("expected two inference logs")
    bnf_lines = sum(
        path.read_text(encoding="utf-8", errors="replace").count("source semantic memory type=")
        for path in infer_logs
    )
    if bnf_lines != 160:
        raise ValueError(f"expected 160 no_text BNF extractions, got {bnf_lines}")
    fatal_tokens = ("Traceback (most recent call last)", "CUDA out of memory", "NCCL", "Killed", "NaN")
    fatal_hits = {
        path.name: [token for token in fatal_tokens if token in path.read_text(encoding="utf-8", errors="replace")]
        for path in sorted((output_dir / "logs").glob("*.log"))
    }
    fatal_hits = {key: value for key, value in fatal_hits.items() if value}
    if fatal_hits:
        raise ValueError(f"fatal log signatures: {fatal_hits}")

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if runtime.get("gpu_count") != 2 or runtime.get("gpu_model") != "NVIDIA GeForce RTX 4090":
        raise ValueError(f"runtime GPU contract drift: {runtime}")
    if runtime.get("hostname") != socket.gethostname():
        raise ValueError("runtime hostname drift")

    checkpoint_files = {name: artifact(checkpoint / name) for name in REQUIRED_CHECKPOINT_FILES}
    artifacts = {
        "validation": artifact(validation_path),
        "runtime": artifact(runtime_path),
        "runner": artifact(runner_path),
        "summary": artifact(summary_path),
        "asr": artifact(asr_path),
        "ref_content": artifact(output_dir / f"{args.run_id}.ref_content_similarity_summary.json"),
        "dual_cases": artifact(dual_path),
        "dual_summary": artifact(aggregate / "dual_encoder_summary.json"),
        "diagnostics": artifact(eval_root / "diagnostics" / f"{args.run_id}.summary.json"),
        "metrics_json": artifact(metrics_json),
        "metrics_tsv": artifact(metrics_tsv),
        "metrics_md": artifact(metrics_md),
    }
    for index, path in enumerate(manifest_paths):
        artifacts[f"manifest_shard{index}"] = artifact(path)

    completion = {
        "schema": "moss_codecvc.batch44_early_best_full320_local.v1",
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": "local",
        "selection": {
            "arm": args.arm,
            "step": args.step,
            "candidate_id": f"{args.arm}_step-{args.step}",
            "selection_scope": "r3/r5 x 4k/6k/8k strict quick20 ranking",
        },
        "train_job_id": args.train_job_id,
        "checkpoint": {"path": str(checkpoint), "files": checkpoint_files},
        "run": {
            "run_id": args.run_id,
            "output_dir": str(output_dir),
            "validation_rows": 320,
            "audio_rows": 320,
            "audio_frames": audio_frames,
            "bnf_extraction_lines": bnf_lines,
        },
        "code_root": str(code_root),
        "runtime": runtime,
        "metrics": metrics,
        "artifacts": artifacts,
    }
    write_atomic(completion_path, json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    completion_sha = sha256_file(completion_path)
    marker = {
        "schema": "moss_codecvc.batch44_early_best_full320_marker.v1",
        "status": "complete",
        "completion_json": str(completion_path),
        "completion_sha256": completion_sha,
    }
    write_atomic(marker_path, json.dumps(marker, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "completion": str(completion_path),
                "completion_sha256": completion_sha,
                "metrics": str(metrics_md),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Strict local full320 finalizer for the Batch-44 r3 continuation.

The continuation restarted its physical checkpoint counter at zero after a
weights-only warm start from effective step 10,000.  This module keeps the
physical continuation step and the scientific/effective step bound together,
verifies the immutable warm-start contract, and accepts a full320 result only
after all inference, Qwen-ASR, WavLM, SpeechBrain, diagnostics, audio, and BNF
bypass checks pass.

This file never submits, stops, or queries a remote job.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import socket
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf


BASE_EFFECTIVE_STEP = 10_000
SUPPORTED_EFFECTIVE_STEPS = (20_000, 24_000, 26_000, 28_000, 30_000)
EXPECTED_WARM_START_CONTRACT_SHA256 = (
    "2d686e5e57b70fcaa3db8c8eb2b306003a38599b2c9ac37023979d80b6d9fc34"
)
GPU_MODEL = "NVIDIA GeForce RTX 4090"
HOST_PREFIX = "xyzhang-dev--"
CHECKPOINT_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "README.md",
    "timbre_memory_adapter.pt",
    "timbre_memory_config.json",
)
COMPLETION_SCHEMA = "moss_codecvc.batch44_r3_warmstart_full320_local.v1"
MARKER_SCHEMA = "moss_codecvc.batch44_r3_warmstart_full320_marker.v1"
BINDING_SCHEMA = "moss_codecvc.batch44_r3_warmstart_full320_binding.v1"
RUNTIME_SCHEMA = "moss_codecvc.batch44_r3_warmstart_full320_runtime.v1"
UNIFIED_INPUT_SCHEMA = "moss_codecvc.batch44_r3_warmstart_full320_unified_input.v1"


def sha256_file(path: Path) -> str:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing/empty artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ValueError(f"missing/empty artifact: {path}")
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def require_artifact_matches(
    spec: Mapping[str, Any], path: Path, label: str
) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise ValueError(f"{label} artifact spec must be an object")
    actual = artifact(path)
    for key in ("path", "size", "sha256"):
        if spec.get(key) != actual[key]:
            raise ValueError(
                f"{label} artifact drift for {key}: "
                f"captured={spec.get(key)!r} actual={actual[key]!r}"
            )
    return actual


def continuation_local_step(effective_step: int) -> int:
    if effective_step not in SUPPORTED_EFFECTIVE_STEPS:
        raise ValueError(
            f"effective step must be one of {SUPPORTED_EFFECTIVE_STEPS}, "
            f"got {effective_step}"
        )
    local_step = effective_step - BASE_EFFECTIVE_STEP
    if local_step <= 0 or local_step % 2_000:
        raise ValueError(f"invalid continuation local step derived from {effective_step}")
    return local_step


def validate_step_mapping(effective_step: int, local_step: int) -> None:
    wanted = continuation_local_step(effective_step)
    if local_step != wanted:
        raise ValueError(
            f"effective/local mapping drift: {effective_step}-{BASE_EFFECTIVE_STEP}="
            f"{wanted}, got {local_step}"
        )


def load_provenance_helper(path: Path):
    resolved = path.expanduser().resolve(strict=True)
    spec = importlib.util.spec_from_file_location(
        "batch44_r3_warmstart_full320_bound_provenance", resolved
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import continuation provenance helper: {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def audit_continuation_binding(
    *,
    provenance_helper: Path,
    project_root: Path,
    effective_step: int,
    checkpoint: Path,
    train_job_id: str,
    warm_start_contract: Path,
    expected_contract_sha256: str,
    min_checkpoint_age_sec: int,
    test_mode: bool = False,
) -> dict[str, Any]:
    local_step = continuation_local_step(effective_step)
    contract = warm_start_contract.expanduser().resolve(strict=True)
    actual_contract_sha = sha256_file(contract)
    if actual_contract_sha != expected_contract_sha256:
        raise ValueError(
            "warm-start contract SHA256 drift: "
            f"{actual_contract_sha} != {expected_contract_sha256}"
        )
    helper = load_provenance_helper(provenance_helper)
    payload = helper.audit_binding(
        project_root=project_root,
        effective_step=effective_step,
        continuation_local_step=local_step,
        checkpoint=checkpoint,
        train_job_id=train_job_id,
        warm_start_contract=contract,
        min_checkpoint_age_sec=min_checkpoint_age_sec,
        test_mode=test_mode,
    )
    expected = {
        "base_effective_step": BASE_EFFECTIVE_STEP,
        "effective_step": effective_step,
        "continuation_local_step": local_step,
        "checkpoint": str(checkpoint.expanduser().resolve()),
        "train_job_id": train_job_id,
    }
    drift = {
        key: {"expected": wanted, "actual": payload.get(key)}
        for key, wanted in expected.items()
        if payload.get(key) != wanted
    }
    if drift:
        raise ValueError(f"continuation helper binding drift: {drift}")
    contract_spec = payload.get("warm_start_contract")
    if not isinstance(contract_spec, Mapping):
        raise ValueError("continuation helper omitted warm-start contract artifact")
    require_artifact_matches(contract_spec, contract, "warm-start contract")
    if contract_spec.get("sha256") != expected_contract_sha256:
        raise ValueError("continuation helper contract SHA binding drift")
    checkpoint_files = payload.get("checkpoint_files")
    if not isinstance(checkpoint_files, Mapping) or set(checkpoint_files) != set(
        CHECKPOINT_FILES
    ):
        raise ValueError("continuation checkpoint five-file set drift")
    for name in CHECKPOINT_FILES:
        require_artifact_matches(
            checkpoint_files[name], checkpoint / name, f"checkpoint {name}"
        )
    return {
        "schema": BINDING_SCHEMA,
        "status": "pass",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": "local",
        "base_effective_step": BASE_EFFECTIVE_STEP,
        "effective_step": effective_step,
        "continuation_local_step": local_step,
        "train_job_id": train_job_id,
        "checkpoint": str(checkpoint.expanduser().resolve()),
        "checkpoint_files": checkpoint_files,
        "checkpoint_manifest_sha256": payload["checkpoint_manifest_sha256"],
        "warm_start_contract": contract_spec,
        "expected_warm_start_contract_sha256": expected_contract_sha256,
        "warm_start_contract_payload": payload.get("warm_start_contract_payload"),
        "training_provenance": payload.get("training_provenance"),
        "provenance_helper": artifact(provenance_helper),
    }


def _parse_gpu_row(line: str) -> list[str]:
    row = [piece.strip() for piece in next(csv.reader([line]))]
    if len(row) != 6:
        raise ValueError(f"unexpected nvidia-smi row: {line!r}")
    return row


def query_local_runtime(
    *, max_initial_memory_mib: int, allow_any_host: bool = False
) -> dict[str, Any]:
    hostname = socket.gethostname()
    if not allow_any_host and not hostname.startswith(HOST_PREFIX):
        raise ValueError(
            f"full320 is restricted to local host {HOST_PREFIX}*, got {hostname!r}"
        )
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if query.returncode != 0:
        raise ValueError(f"nvidia-smi failed: {query.stderr.strip()}")
    rows = [_parse_gpu_row(line) for line in query.stdout.splitlines() if line.strip()]
    if len(rows) != 2:
        raise ValueError(f"local full320 requires exactly two GPUs, got {len(rows)}")
    gpus = [
        {
            "index": int(index),
            "uuid": uuid,
            "name": name,
            "memory_total_mib": int(total),
            "memory_used_mib_at_start": int(used),
            "driver_version": driver,
        }
        for index, uuid, name, total, used, driver in rows
    ]
    gpus.sort(key=lambda item: item["index"])
    if [item["index"] for item in gpus] != [0, 1]:
        raise ValueError(f"local full320 requires GPU indices [0,1], got {gpus}")
    if any(item["name"] != GPU_MODEL for item in gpus):
        raise ValueError(f"local full320 requires two {GPU_MODEL} GPUs, got {gpus}")
    if any(item["memory_used_mib_at_start"] > max_initial_memory_mib for item in gpus):
        raise ValueError(
            f"local GPUs are not idle below {max_initial_memory_mib} MiB: {gpus}"
        )
    return {
        "hostname": hostname,
        "backend": "local",
        "gpu_count": 2,
        "gpu_indices": [0, 1],
        "gpu_model": GPU_MODEL,
        "gpus": gpus,
    }


def capture_runtime(
    *,
    output: Path,
    binding: Path,
    runner: Path,
    finalizer: Path,
    provenance_helper: Path,
    effective_step: int,
    max_initial_memory_mib: int,
    allow_any_host: bool = False,
) -> dict[str, Any]:
    local_step = continuation_local_step(effective_step)
    binding_payload = load_json(binding, "continuation binding")
    if binding_payload.get("schema") != BINDING_SCHEMA:
        raise ValueError("runtime binding schema drift")
    runtime = query_local_runtime(
        max_initial_memory_mib=max_initial_memory_mib,
        allow_any_host=allow_any_host,
    )
    payload = {
        "schema": RUNTIME_SCHEMA,
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **runtime,
        "base_effective_step": BASE_EFFECTIVE_STEP,
        "effective_step": effective_step,
        "continuation_local_step": local_step,
        "scheduling": "one full320 run, two modulo shards on local GPU indices 0 and 1",
        "binding": artifact(binding),
        "runner": artifact(runner),
        "finalizer": artifact(finalizer),
        "provenance_helper": artifact(provenance_helper),
    }
    atomic_json(output, payload)
    return payload


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
    previous = [0] * (len(b) + 1)
    for char_a in a:
        current = [0]
        for index, char_b in enumerate(b, start=1):
            current.append(
                previous[index - 1] + 1
                if char_a == char_b
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def ref_f1(row: Mapping[str, Any]) -> float:
    generated = normalize(row.get("asr_tgt_text"))
    reference = normalize(row.get("timbre_ref_text"))
    hit = lcs_len(generated, reference)
    precision = hit / max(1, len(generated))
    recall = hit / max(1, len(reference))
    return 0.0 if precision + recall <= 0 else 2 * precision * recall / (precision + recall)


def validate_audio(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 1024:
        raise ValueError(f"missing/small generated audio: {path}")
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != 24_000 or data.shape[1] != 1 or data.shape[0] <= 0:
        raise ValueError(
            f"unexpected audio format {path}: sr={sample_rate} shape={data.shape}"
        )
    if not np.isfinite(data).all():
        raise ValueError(f"non-finite audio samples: {path}")
    return {"frames": int(data.shape[0]), "sample_rate": sample_rate, "channels": 1}


def build_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    effective_step: int,
    local_step: int,
    train_job_id: str,
    checkpoint: Path,
    contract: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_step_mapping(effective_step, local_step)
    metrics: list[dict[str, Any]] = []
    fail_reasons: dict[str, Any] = {}
    for scope in ("no_text", "text", "all"):
        selected = list(rows) if scope == "all" else [row for row in rows if row["mode"] == scope]
        expected = 320 if scope == "all" else 160
        if len(selected) != expected:
            raise ValueError(f"{scope}: rows={len(selected)}, expected {expected}")
        keep = sum(truth(row.get("content_keep")) for row in selected)
        wavlm_ref = [finite(row.get("sim_gen_ref"), label="sim_gen_ref") for row in selected]
        wavlm_src = [finite(row.get("sim_gen_source"), label="sim_gen_source") for row in selected]
        speechbrain_ref = [
            finite(row.get("ecapa_sim_gen_ref"), label="ecapa_sim_gen_ref")
            for row in selected
        ]
        speechbrain_src = [
            finite(row.get("ecapa_sim_gen_source"), label="ecapa_sim_gen_source")
            for row in selected
        ]
        qwen_cer = [finite(row.get("cer_tgt"), label="cer_tgt") for row in selected]
        qwen_wer = [finite(row.get("wer_tgt"), label="wer_tgt") for row in selected]
        qwen_primary = [
            finite(
                row.get("wer_tgt") if row.get("language") == "en" else row.get("cer_tgt"),
                label="qwen_primary_error",
            )
            for row in selected
        ]
        en_src = [
            row
            for row in selected
            if row.get("mode") == "text" and str(row.get("cell") or "").startswith("en_src_")
        ]
        if scope == "text" and len(en_src) != 80:
            raise ValueError(f"text en_src rows={len(en_src)}, expected 80")
        all_reason_counts = Counter(str(row.get("content_filter_reason") or "missing") for row in selected)
        failed_reason_counts = Counter(
            str(row.get("content_filter_reason") or "missing")
            for row in selected
            if not truth(row.get("content_keep"))
        )
        fail_reasons[scope] = {
            "n": len(selected),
            "keep": keep,
            "fail": len(selected) - keep,
            "all_reason_counts": dict(sorted(all_reason_counts.items())),
            "failed_reason_counts": dict(sorted(failed_reason_counts.items())),
        }
        metrics.append(
            {
                "arm": "r3",
                "step": effective_step,
                "effective_step": effective_step,
                "base_effective_step": BASE_EFFECTIVE_STEP,
                "continuation_local_step": local_step,
                "text_repeat": 3,
                "train_job_id": train_job_id,
                "checkpoint": str(checkpoint),
                "warm_start_contract": str(contract),
                "scope": scope,
                "n": len(selected),
                "keep": keep,
                "fail_count": len(selected) - keep,
                "fail_rate": (len(selected) - keep) / len(selected),
                "qwen_primary_error": mean(qwen_primary),
                "qwen_cer": mean(qwen_cer),
                "qwen_wer": mean(qwen_wer),
                # Compatibility alias consumed by the registered 004128
                # learning-curve/report schema.  The scorer identity remains
                # explicit in the qwen_* fields above.
                "cer": mean(qwen_cer),
                "wavlm_sim_ref": mean(wavlm_ref),
                "wavlm_sim_src": mean(wavlm_src),
                "wavlm_margin": mean(wavlm_ref) - mean(wavlm_src),
                "wavlm_ref_bound": sum(
                    ref - src > 0.05 for ref, src in zip(wavlm_ref, wavlm_src)
                )
                / len(selected),
                "speechbrain_sim_ref": mean(speechbrain_ref),
                "speechbrain_sim_src": mean(speechbrain_src),
                "speechbrain_margin": mean(speechbrain_ref) - mean(speechbrain_src),
                "speechbrain_ref_bound": sum(
                    ref - src > 0.05
                    for ref, src in zip(speechbrain_ref, speechbrain_src)
                )
                / len(selected),
                "ref_content_lcs_f1": mean(ref_f1(row) for row in selected),
                "text_en_src_n": len(en_src) if en_src else "",
                "text_en_src_fail_count": (
                    sum(not truth(row.get("content_keep")) for row in en_src) if en_src else ""
                ),
                "text_en_src_fail_rate": (
                    sum(not truth(row.get("content_keep")) for row in en_src) / len(en_src)
                    if en_src
                    else ""
                ),
                "text_en_src_qwen_cer": (
                    mean(finite(row.get("cer_tgt"), label="en_src.cer_tgt") for row in en_src)
                    if en_src
                    else ""
                ),
                "failed_reason_counts_json": json.dumps(
                    dict(sorted(failed_reason_counts.items())),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
    return metrics, fail_reasons


RUN_RE = re.compile(r"\[persistent-valid\] run (\S+) mode=(no_text|text)\b")
DONE_RE = re.compile(r"\[persistent-valid\] done (\S+) status=(\S+)")
BNF_TOKEN = "source semantic memory type="


def parse_bnf_by_mode(
    log_paths: Sequence[Path], expected_ids: set[str]
) -> dict[str, Any]:
    run_modes: Counter[str] = Counter()
    bnf_modes: Counter[str] = Counter()
    seen_runs: set[str] = set()
    seen_done: set[str] = set()
    for path in log_paths:
        current_case: str | None = None
        current_mode: str | None = None
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            run_match = RUN_RE.search(line)
            if run_match:
                if current_case is not None:
                    raise ValueError(f"nested inference run in {path}: {current_case}")
                current_case, current_mode = run_match.groups()
                if current_case in seen_runs:
                    raise ValueError(f"duplicate inference run case: {current_case}")
                seen_runs.add(current_case)
                run_modes[current_mode] += 1
                continue
            if BNF_TOKEN in line:
                if current_case is None or current_mode is None:
                    raise ValueError(f"BNF extraction outside a case in {path}")
                bnf_modes[current_mode] += 1
                continue
            done_match = DONE_RE.search(line)
            if done_match:
                done_case, done_status = done_match.groups()
                if done_case != current_case:
                    raise ValueError(
                        f"inference done/run mismatch in {path}: {done_case}/{current_case}"
                    )
                if done_status != "ok":
                    raise ValueError(f"non-ok inference log status: {done_case}={done_status}")
                seen_done.add(done_case)
                current_case = None
                current_mode = None
        if current_case is not None:
            raise ValueError(f"unterminated inference case in {path}: {current_case}")
    if seen_runs != expected_ids or seen_done != expected_ids:
        raise ValueError(
            "inference log case-id coverage drift: "
            f"runs={len(seen_runs)} done={len(seen_done)} expected={len(expected_ids)}"
        )
    expected_runs = Counter({"no_text": 160, "text": 160})
    if run_modes != expected_runs:
        raise ValueError(f"inference log mode counts drift: {run_modes}")
    expected_bnf = Counter({"no_text": 160})
    if bnf_modes != expected_bnf:
        raise ValueError(
            "BNF bypass drift: expected no_text=160,text=0; "
            f"got no_text={bnf_modes['no_text']},text={bnf_modes['text']}"
        )
    return {
        "run_case_counts": {"no_text": 160, "text": 160},
        "bnf_extraction_counts": {
            "no_text": bnf_modes["no_text"],
            "text": bnf_modes["text"],
        },
        "case_ids": len(seen_runs),
    }


def stable_case_uid(case_id: str, source: str, reference: str) -> str:
    payload = f"{case_id}\0{source}\0{reference}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_unified_input_rows(
    *,
    validation_rows: Sequence[Mapping[str, Any]],
    manifest_by_id: Mapping[str, Mapping[str, Any]],
    asr_by_id: Mapping[str, Mapping[str, Any]],
    system_id: str,
    run_id: str,
    effective_step: int,
    local_step: int,
    checkpoint: Path,
    contract: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, validation in enumerate(validation_rows):
        case_id = str(validation.get("case_id") or "")
        manifest = manifest_by_id[case_id]
        asr = asr_by_id[case_id]
        source = str(validation.get("source_audio") or "")
        reference = str(validation.get("timbre_ref_audio") or "")
        generated = str(manifest.get("output_wav") or "")
        language = str(asr.get("language") or validation.get("source_lang") or "").lower()
        if language not in {"en", "zh"}:
            raise ValueError(f"{case_id}: unsupported content language {language!r}")
        reference_text = str(
            asr.get("content_ref_text") or validation.get("content_ref_text") or ""
        ).strip()
        if not reference_text:
            raise ValueError(f"{case_id}: missing unified evaluator reference text")
        for label, raw_path in (
            ("source_audio", source),
            ("reference_audio", reference),
            ("generated_audio", generated),
        ):
            path = Path(raw_path)
            if not path.is_file() or path.stat().st_size < 44:
                raise ValueError(f"{case_id}: missing/empty {label}: {path}")
        rows.append(
            {
                "schema_version": UNIFIED_INPUT_SCHEMA,
                "record_type": "baseline_vc_inference",
                "system_id": system_id,
                "test_set_id": "seedtts-derived-vc-internal320",
                "case_id": case_id,
                "case_uid": stable_case_uid(case_id, source, reference),
                "input_index": index,
                "language": language,
                "source_audio": source,
                "reference_audio": reference,
                "generated_audio": generated,
                "target_text": reference_text,
                "reference_text": reference_text,
                "status": "ok",
                "metadata": {
                    "mode": validation.get("mode"),
                    "moss_codecvc_mode": validation.get("mode"),
                    "cell": validation.get("cell"),
                    "effective_step": effective_step,
                    "continuation_local_step": local_step,
                    "existing_qwen_asr_backend": asr.get("target_asr_backend"),
                    "existing_qwen_content_keep": truth(asr.get("content_keep")),
                    "existing_qwen_content_filter_reason": asr.get(
                        "content_filter_reason"
                    ),
                },
                "provenance": {
                    "run_id": run_id,
                    "checkpoint": str(checkpoint),
                    "warm_start_contract": str(contract),
                    "purpose": "input for scripts/004082_run_unified_vc_eval.py three-scorer/two-ASR pass",
                },
            }
        )
    if len(rows) != 320 or len({row["case_id"] for row in rows}) != 320:
        raise ValueError("unified evaluator input is not unique 320")
    return rows


def _fmt(value: Any, *, percent: bool = False) -> str:
    if value == "" or value is None:
        return "—"
    return f"{100 * float(value):.2f}%" if percent else f"{float(value):.4f}"


def write_metric_artifacts(
    aggregate: Path,
    metrics: Sequence[Mapping[str, Any]],
    fail_reasons: Mapping[str, Any],
    *,
    effective_step: int,
    local_step: int,
) -> dict[str, Path]:
    metrics_json = aggregate / "metrics.json"
    metrics_tsv = aggregate / "metrics.tsv"
    metrics_md = aggregate / "metrics.md"
    fail_json = aggregate / "fail_reasons.json"
    fail_md = aggregate / "fail_reasons.md"
    atomic_json(metrics_json, list(metrics))
    temporary_tsv = metrics_tsv.with_name(f".{metrics_tsv.name}.tmp-{os.getpid()}")
    with temporary_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(metrics)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_tsv, metrics_tsv)
    lines = [
        f"# Batch-44 r3 continuation full320: effective-{effective_step} (local-{local_step})",
        "",
        "| Scope | n | fail | Qwen primary | Qwen CER | Qwen WER | WavLM ref | WavLM src | margin | ref-bound | SpB ref | SpB src | SpB ref-bound | F1(ref-content) | en_src fail | en_src CER |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['scope']} | {row['n']} | {_fmt(row['fail_rate'], percent=True)} | "
            f"{_fmt(row['qwen_primary_error'])} | {_fmt(row['qwen_cer'])} | "
            f"{_fmt(row['qwen_wer'])} | {_fmt(row['wavlm_sim_ref'])} | "
            f"{_fmt(row['wavlm_sim_src'])} | {_fmt(row['wavlm_margin'])} | "
            f"{_fmt(row['wavlm_ref_bound'], percent=True)} | "
            f"{_fmt(row['speechbrain_sim_ref'])} | {_fmt(row['speechbrain_sim_src'])} | "
            f"{_fmt(row['speechbrain_ref_bound'], percent=True)} | "
            f"{_fmt(row['ref_content_lcs_f1'])} | "
            f"{_fmt(row['text_en_src_fail_rate'], percent=True)} | "
            f"{_fmt(row['text_en_src_qwen_cer'])} |"
        )
    atomic_text(metrics_md, "\n".join(lines) + "\n")
    atomic_json(fail_json, fail_reasons)
    fail_lines = ["# Qwen content-filter failure reasons", ""]
    for scope in ("no_text", "text", "all"):
        fail_lines.extend(
            [
                f"## {scope}",
                "",
                f"- keep: {fail_reasons[scope]['keep']}",
                f"- fail: {fail_reasons[scope]['fail']}",
                f"- failed reasons: `{json.dumps(fail_reasons[scope]['failed_reason_counts'], ensure_ascii=False, sort_keys=True)}`",
                "",
            ]
        )
    atomic_text(fail_md, "\n".join(fail_lines))
    return {
        "metrics_json": metrics_json,
        "metrics_tsv": metrics_tsv,
        "metrics_md": metrics_md,
        "fail_reasons_json": fail_json,
        "fail_reasons_md": fail_md,
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    effective_step = int(args.effective_step)
    local_step = continuation_local_step(effective_step)
    project_root = args.project_root.expanduser().resolve(strict=True)
    record_root = args.record_root.expanduser().resolve(strict=True)
    eval_root = args.eval_root.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    contract = args.warm_start_contract.expanduser().resolve(strict=True)
    validation_path = args.validation_jsonl.expanduser().resolve(strict=True)
    code_root = args.code_root.expanduser().resolve(strict=True)
    binding_path = args.binding.expanduser().resolve(strict=True)
    runtime_path = args.runtime.expanduser().resolve(strict=True)
    runner_path = args.runner.expanduser().resolve(strict=True)
    finalizer_path = args.finalizer.expanduser().resolve(strict=True)
    helper_path = args.provenance_helper.expanduser().resolve(strict=True)
    completion_path = record_root / "COMPLETED.json"
    marker_path = record_root / "complete.marker"
    if completion_path.exists() or completion_path.is_symlink():
        raise ValueError("completion JSON already exists")
    if marker_path.exists() or marker_path.is_symlink():
        raise ValueError("completion marker already exists")
    if checkpoint.name != f"step-{local_step}":
        raise ValueError(f"checkpoint/local-step drift: {checkpoint}")
    if sha256_file(validation_path) != args.validation_sha256:
        raise ValueError("validation SHA256 drift")

    current_binding = audit_continuation_binding(
        provenance_helper=helper_path,
        project_root=project_root,
        effective_step=effective_step,
        checkpoint=checkpoint,
        train_job_id=args.train_job_id,
        warm_start_contract=contract,
        expected_contract_sha256=args.warm_start_contract_sha256,
        min_checkpoint_age_sec=0,
        test_mode=args.test_mode,
    )
    captured_binding = load_json(binding_path, "captured binding")
    for key in (
        "schema",
        "status",
        "base_effective_step",
        "effective_step",
        "continuation_local_step",
        "train_job_id",
        "checkpoint",
        "checkpoint_manifest_sha256",
        "expected_warm_start_contract_sha256",
    ):
        if captured_binding.get(key) != current_binding.get(key):
            raise ValueError(f"captured/current continuation binding drift for {key}")
    for name in CHECKPOINT_FILES:
        require_artifact_matches(
            captured_binding["checkpoint_files"][name],
            checkpoint / name,
            f"captured checkpoint {name}",
        )

    runtime = load_json(runtime_path, "local runtime")
    expected_runtime = {
        "schema": RUNTIME_SCHEMA,
        "status": "started",
        "backend": "local",
        "gpu_count": 2,
        "gpu_indices": [0, 1],
        "gpu_model": GPU_MODEL,
        "base_effective_step": BASE_EFFECTIVE_STEP,
        "effective_step": effective_step,
        "continuation_local_step": local_step,
    }
    runtime_drift = {
        key: {"expected": wanted, "actual": runtime.get(key)}
        for key, wanted in expected_runtime.items()
        if runtime.get(key) != wanted
    }
    if runtime_drift:
        raise ValueError(f"local runtime contract drift: {runtime_drift}")
    if not args.test_mode and runtime.get("hostname") != socket.gethostname():
        raise ValueError("runtime hostname drift")
    require_artifact_matches(runtime["binding"], binding_path, "runtime binding")
    for key, path in (
        ("runner", runner_path),
        ("finalizer", finalizer_path),
        ("provenance_helper", helper_path),
    ):
        require_artifact_matches(runtime[key], path, f"runtime {key}")

    validation = read_jsonl(validation_path)
    expected_ids = {str(row.get("case_id") or "") for row in validation}
    expected_modes = Counter(str(row.get("mode") or "") for row in validation)
    if len(validation) != 320 or len(expected_ids) != 320 or "" in expected_ids:
        raise ValueError("validation set is not unique 320")
    if expected_modes != Counter({"no_text": 160, "text": 160}):
        raise ValueError(f"validation mode drift: {expected_modes}")

    manifest_paths = sorted(output_dir.glob("manifest.shard*.jsonl"))
    if len(manifest_paths) != 2:
        raise ValueError(f"expected two inference manifests, got {manifest_paths}")
    manifest_rows = [row for path in manifest_paths for row in read_jsonl(path)]
    manifest_ids = [str(row.get("case_id") or "") for row in manifest_rows]
    statuses = Counter(str(row.get("status") or "") for row in manifest_rows)
    if (
        len(manifest_rows) != 320
        or len(set(manifest_ids)) != 320
        or set(manifest_ids) != expected_ids
    ):
        raise ValueError("inference manifest is not the canonical 320")
    if statuses != Counter({"ok": 320}):
        raise ValueError(f"inference statuses are not all ok: {statuses}")
    manifest_by_id = {str(row["case_id"]): row for row in manifest_rows}
    output_wavs = [Path(str(row.get("output_wav") or "")) for row in manifest_rows]
    if len({str(path.resolve(strict=False)) for path in output_wavs}) != 320:
        raise ValueError("generated WAV paths are not unique 320")
    audio_frames = 0
    for wav in output_wavs:
        audio_frames += validate_audio(wav)["frames"]

    asr_path = output_dir / f"{args.run_id}.asr_eval.jsonl"
    asr_rows = read_jsonl(asr_path)
    asr_by_id = {
        str(row.get("case_id") or row.get("sample_id") or ""): row for row in asr_rows
    }
    if len(asr_rows) != 320 or len(asr_by_id) != 320 or set(asr_by_id) != expected_ids:
        raise ValueError("Qwen ASR result is not the canonical 320")
    qwen_backends = Counter(str(row.get("target_asr_backend") or "") for row in asr_rows)
    if qwen_backends != Counter({"qwen_asr": 320}):
        raise ValueError(f"Qwen ASR backend coverage drift: {qwen_backends}")
    for row in asr_rows:
        case_id = str(row.get("case_id") or row.get("sample_id") or "")
        if str(row.get("manifest_status") or "") != "ok":
            raise ValueError(f"{case_id}: non-ok ASR manifest status")
        finite(row.get("cer_tgt"), label=f"{case_id}.cer_tgt")
        finite(row.get("wer_tgt"), label=f"{case_id}.wer_tgt")
        if not str(row.get("content_filter_reason") or ""):
            raise ValueError(f"{case_id}: missing content_filter_reason")

    summary_path = output_dir / f"{args.run_id}.summary.json"
    summary = load_json(summary_path, "full320 summary")
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
        asr = asr_by_id[row["case_id"]]
        for key in (
            "timbre_ref_text",
            "language",
            "content_filter_reason",
            "content_keep",
            "cer_tgt",
            "wer_tgt",
            "asr_tgt_text",
        ):
            row[key] = asr.get(key)

    metrics, fail_reasons = build_metrics(
        dual_rows,
        effective_step=effective_step,
        local_step=local_step,
        train_job_id=args.train_job_id,
        checkpoint=checkpoint,
        contract=contract,
    )
    metric_paths = write_metric_artifacts(
        aggregate,
        metrics,
        fail_reasons,
        effective_step=effective_step,
        local_step=local_step,
    )

    infer_logs = sorted((output_dir / "logs").glob("infer.shard*.log"))
    if len(infer_logs) != 2:
        raise ValueError("expected two inference logs")
    bnf_audit = parse_bnf_by_mode(infer_logs, expected_ids)
    fatal_tokens = (
        "Traceback (most recent call last)",
        "CUDA out of memory",
        "NCCL",
        "Killed",
        "NaN",
    )
    fatal_hits = {
        path.name: [
            token
            for token in fatal_tokens
            if token in path.read_text(encoding="utf-8", errors="replace")
        ]
        for path in sorted((output_dir / "logs").glob("*.log"))
    }
    fatal_hits = {name: hits for name, hits in fatal_hits.items() if hits}
    if fatal_hits:
        raise ValueError(f"fatal log signatures: {fatal_hits}")

    system_id = f"ver2_9_5_final_r3_warmstart_effective_{effective_step}"
    unified_rows = build_unified_input_rows(
        validation_rows=validation,
        manifest_by_id=manifest_by_id,
        asr_by_id=asr_by_id,
        system_id=system_id,
        run_id=args.run_id,
        effective_step=effective_step,
        local_step=local_step,
        checkpoint=checkpoint,
        contract=contract,
    )
    unified_path = aggregate / "unified_eval_input.jsonl"
    atomic_jsonl(unified_path, unified_rows)
    unified_summary_path = aggregate / "unified_eval_input.summary.json"
    atomic_json(
        unified_summary_path,
        {
            "schema": UNIFIED_INPUT_SCHEMA,
            "status": "ready",
            "rows": 320,
            "mode_counts": {"no_text": 160, "text": 160},
            "language_counts": dict(
                sorted(Counter(row["language"] for row in unified_rows).items())
            ),
            "system_id": system_id,
            "run_id": args.run_id,
            "effective_step": effective_step,
            "continuation_local_step": local_step,
            "consumer": "scripts/004082_run_unified_vc_eval.py",
        },
    )

    diagnostics_path = eval_root / "diagnostics" / f"{args.run_id}.summary.json"
    code_artifacts = {
        name: artifact(code_root / name)
        for name in (
            "scripts/004039_run_seedtts_validation_eval.sh",
            "scripts/004044_run_seedtts_validation_infer_persistent.py",
            "scripts/004048_summarize_seedtts_ablation_metrics.py",
            "scripts/004056_summarize_seedtts_ref_content_similarity.py",
            "scripts/004063_analyze_seedtts320_diagnostics.py",
        )
    }
    result_artifacts: dict[str, Any] = {
        "validation": artifact(validation_path),
        "binding": artifact(binding_path),
        "runtime": artifact(runtime_path),
        "runner": artifact(runner_path),
        "finalizer": artifact(finalizer_path),
        "provenance_helper": artifact(helper_path),
        "summary": artifact(summary_path),
        "qwen_asr": artifact(asr_path),
        "ref_content": artifact(
            output_dir / f"{args.run_id}.ref_content_similarity_summary.json"
        ),
        "dual_cases": artifact(dual_path),
        "dual_summary": artifact(aggregate / "dual_encoder_summary.json"),
        "diagnostics": artifact(diagnostics_path),
        "unified_eval_input": artifact(unified_path),
        "unified_eval_input_summary": artifact(unified_summary_path),
    }
    result_artifacts.update({name: artifact(path) for name, path in metric_paths.items()})
    for index, path in enumerate(manifest_paths):
        result_artifacts[f"manifest_shard{index}"] = artifact(path)
    for index, path in enumerate(infer_logs):
        result_artifacts[f"infer_log_shard{index}"] = artifact(path)

    completed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    completion = {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "completed_at_utc": completed_at,
        "backend": "local",
        "base_effective_step": BASE_EFFECTIVE_STEP,
        "effective_step": effective_step,
        "continuation_local_step": local_step,
        "arm": "r3",
        "text_repeat": 3,
        "train_job_id": args.train_job_id,
        "record_root": str(record_root),
        "eval_root": str(eval_root),
        "checkpoint": {
            "path": str(checkpoint),
            "files": current_binding["checkpoint_files"],
            "manifest_sha256": current_binding["checkpoint_manifest_sha256"],
        },
        "warm_start_contract": current_binding["warm_start_contract"],
        "expected_warm_start_contract_sha256": args.warm_start_contract_sha256,
        "binding": captured_binding,
        "run": {
            "run_id": args.run_id,
            "output_dir": str(output_dir),
            "validation_rows": 320,
            "inference_rows": 320,
            "qwen_asr_rows": 320,
            "audio_rows": 320,
            "audio_frames": audio_frames,
            "bnf_audit": bnf_audit,
            "unified_eval_input": str(unified_path),
        },
        "runtime": runtime,
        "metrics": metrics,
        "fail_reasons": fail_reasons,
        "code_root": str(code_root),
        "code_artifacts": code_artifacts,
        "artifacts": result_artifacts,
    }
    atomic_json(completion_path, completion)
    completion_sha = sha256_file(completion_path)
    marker = {
        "schema": MARKER_SCHEMA,
        "status": "complete",
        "backend": "local",
        "effective_step": effective_step,
        "continuation_local_step": local_step,
        "completed_at_utc": completed_at,
        "completion_json": str(completion_path),
        "completion_sha256": completion_sha,
    }
    atomic_json(marker_path, marker)
    return completion


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit-binding")
    audit.add_argument("--provenance-helper", type=Path, required=True)
    audit.add_argument("--project-root", type=Path, required=True)
    audit.add_argument("--effective-step", type=int, required=True)
    audit.add_argument("--checkpoint", type=Path, required=True)
    audit.add_argument("--train-job-id", required=True)
    audit.add_argument("--warm-start-contract", type=Path, required=True)
    audit.add_argument(
        "--warm-start-contract-sha256",
        default=EXPECTED_WARM_START_CONTRACT_SHA256,
    )
    audit.add_argument("--min-checkpoint-age-sec", type=int, default=90)
    audit.add_argument("--output", type=Path)
    audit.add_argument("--test-mode", action="store_true")

    runtime = commands.add_parser("capture-runtime")
    for name in ("output", "binding", "runner", "finalizer", "provenance-helper"):
        runtime.add_argument(f"--{name}", type=Path, required=True)
    runtime.add_argument("--effective-step", type=int, required=True)
    runtime.add_argument("--max-initial-memory-mib", type=int, default=2048)
    runtime.add_argument("--allow-any-host", action="store_true")

    finish = commands.add_parser("finalize")
    for name in (
        "project-root",
        "record-root",
        "eval-root",
        "output-dir",
        "checkpoint",
        "warm-start-contract",
        "validation-jsonl",
        "code-root",
        "binding",
        "runtime",
        "runner",
        "finalizer",
        "provenance-helper",
    ):
        finish.add_argument(f"--{name}", type=Path, required=True)
    finish.add_argument("--effective-step", type=int, required=True)
    finish.add_argument("--train-job-id", required=True)
    finish.add_argument("--warm-start-contract-sha256", required=True)
    finish.add_argument("--validation-sha256", required=True)
    finish.add_argument("--run-id", required=True)
    finish.add_argument("--test-mode", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "audit-binding":
        payload = audit_continuation_binding(
            provenance_helper=args.provenance_helper,
            project_root=args.project_root,
            effective_step=args.effective_step,
            checkpoint=args.checkpoint,
            train_job_id=args.train_job_id,
            warm_start_contract=args.warm_start_contract,
            expected_contract_sha256=args.warm_start_contract_sha256,
            min_checkpoint_age_sec=args.min_checkpoint_age_sec,
            test_mode=args.test_mode,
        )
        if args.output:
            atomic_json(args.output, payload)
        print(
            "[batch44-r3-warmstart-full320-binding] PASS "
            f"effective={payload['effective_step']} "
            f"local={payload['continuation_local_step']}"
        )
        return 0
    if args.command == "capture-runtime":
        payload = capture_runtime(
            output=args.output,
            binding=args.binding,
            runner=args.runner,
            finalizer=args.finalizer,
            provenance_helper=args.provenance_helper,
            effective_step=args.effective_step,
            max_initial_memory_mib=args.max_initial_memory_mib,
            allow_any_host=args.allow_any_host,
        )
        print(
            "[batch44-r3-warmstart-full320-runtime] PASS "
            f"host={payload['hostname']} effective={payload['effective_step']}"
        )
        return 0
    payload = finalize(args)
    print(
        json.dumps(
            {
                "status": "PASS",
                "effective_step": payload["effective_step"],
                "continuation_local_step": payload["continuation_local_step"],
                "completion": str(args.record_root / "COMPLETED.json"),
                "metrics": str(args.eval_root / "aggregate/metrics.md"),
                "unified_eval_input": str(
                    args.eval_root / "aggregate/unified_eval_input.jsonl"
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

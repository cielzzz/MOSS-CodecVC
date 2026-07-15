#!/usr/bin/env python3
"""Production cleaner for the v2 U1-prime residual-identity four-edge arm.

This is deliberately separate from the historical ``*.filtered.jsonl`` link.
It only filters the canonical v2 no_text manifest; it never regenerates SeedVC,
codec, BNF, prosody, U2, or training inputs.

Stages are restartable and designed for one controller plus 1--8 GPU shards:

  prepare -> prompt-ecapa -> ecapa-score -> wavlm-plan -> wavlm-cache
          -> wavlm-score -> finalize -> fresh-audit

``prepare`` is the only stage that builds the strict source-job/result join.
All GPU stages are partitioned deterministically.  A row whose required asset
or embedding cannot be validated is recorded as dropped; it is never silently
kept.  ``COMPLETED.json`` is written only by ``fresh-audit`` after full
acceptance checks pass.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import UTC, datetime
from typing import Any, Iterator, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moss_codecvc.third_party import add_download_python_deps
from moss_codecvc.models.speaker_encoder import FrozenWavLMSVEncoder


DEFAULT_PREPARED = ROOT / (
    "trainset/ver2_9_prepared_v2_real_no_text_refdecorr_wavlm_sv_20260708/"
    "no_text.v2.train.jsonl"
)
DEFAULT_LEGACY_FILTERED = DEFAULT_PREPARED.with_name("no_text.v2.train.filtered.jsonl")
DEFAULT_SOURCE_ROOT = Path(
    "/inspire/qb-ilm/project/embodied-multimodality/public/xyzhang/data/VC_train/"
    "v2_real_target_no_text_300k_zh_en_balanced_20260707_seedvc_triples"
)
DEFAULT_SOURCE_JOBS = DEFAULT_SOURCE_ROOT / "source_seedvc_jobs.jsonl"
DEFAULT_SOURCE_RESULTS = DEFAULT_SOURCE_ROOT / "source_seedvc_results.jsonl"
DEFAULT_ECAPA_ROOT = ROOT / (
    "trainset/v2_real_target_no_text_refdecorr_20260708/"
    "v2_real_no_text_refdecorr_train_minus_valid/speaker_embeddings/ecapa"
)
DEFAULT_ECAPA_MODEL = Path(
    "/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/models/"
    "speechbrain/spkrec-ecapa-voxceleb"
)
DEFAULT_OUTPUT = ROOT / "trainset/ver2_9_prepared_v2_four_edge_balanced_20260715"
EXPECTED_ROWS = 295_632
EXPECTED_INPUT_SHA256 = "de2e6ca854c8054445739ea831641b0f138893f2ec9ba8dbfd7b0a5760dda5eb"
CODE_SNAPSHOT_FILES = (
    Path(__file__).resolve(),
    ROOT / "scripts/004135_run_v2_four_edge_balanced.sh",
    ROOT / "scripts/004136_submit_v2_four_edge_balanced_qz.sh",
    ROOT / "moss_codecvc/models/speaker_encoder.py",
    ROOT / "moss_codecvc/third_party.py",
)

ECAPA_THRESHOLDS = {
    "low_max": 0.40,
    "u2_u1_min": 0.80,
    "u1prime_prompt_min": 0.70,
    "gap_min": 0.25,
}
WAVLM_THRESHOLDS = {
    "low_max": 0.85,
    "u2_u1_min": 0.94,
    "u1prime_prompt_min": 0.94,
    "gap_min": 0.08,
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def sha256_file(file_path: str | Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha1_id(*parts: Any, length: int = 24) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(str(part).encode("utf-8", errors="strict"))
        digest.update(b"\0")
    return digest.hexdigest()[:length]


def deterministic_shard(key: str, shard_count: int) -> int:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % shard_count


def atomic_json(path_obj: Path, payload: Any) -> None:
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    temporary = path_obj.with_name(path_obj.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path_obj)


def jsonl_writer(path_obj: Path):
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    temporary = path_obj.with_name(path_obj.name + ".tmp")
    return temporary, temporary.open("w", encoding="utf-8")


def finish_jsonl(temporary: Path, final: Path) -> None:
    temporary.replace(final)


def load_json(path_obj: Path) -> dict[str, Any]:
    return json.loads(path_obj.read_text(encoding="utf-8"))


def iter_jsonl(file_path: Path) -> Iterator[dict[str, Any]]:
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL {file_path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"non-object JSONL row {file_path}:{line_number}")
            yield value


def iter_raw_jsonl(file_path: Path, *, max_rows: int = 0) -> Iterator[tuple[int, bytes, dict[str, Any]]]:
    row_index = 0
    with file_path.open("rb") as handle:
        for physical_line, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL {file_path}:{physical_line}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"non-object JSONL row {file_path}:{physical_line}")
            yield row_index, raw, row
            row_index += 1
            if max_rows > 0 and row_index >= max_rows:
                break


def normalise_text(value: Any) -> str:
    # This matches the v2 known-content/ref-content-leak normalisation intent:
    # lowercase and retain only alnum plus CJK characters.
    lowered = str(value or "").lower()
    return "".join(
        char
        for char in lowered
        if char.isalnum() or 0x3400 <= ord(char) <= 0x4DBF or 0x4E00 <= ord(char) <= 0x9FFF
    )


def original_index(sample_id: Any) -> int:
    parts = str(sample_id).split(":")
    if len(parts) < 2:
        raise ValueError(f"sample_id has no original row index: {sample_id!r}")
    try:
        value = int(parts[-2])
    except ValueError as exc:
        raise ValueError(f"sample_id has invalid original row index: {sample_id!r}") from exc
    if value < 0:
        raise ValueError(f"sample_id has negative original row index: {sample_id!r}")
    return value


def finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def vector_from_payload(payload: Any, source: Path) -> torch.Tensor:
    if isinstance(payload, dict):
        for key in ("speaker_embedding", "embedding", "emb", "xvector", "vector"):
            if key in payload and payload[key] is not None:
                payload = payload[key]
                break
        else:
            raise ValueError(f"no embedding key in {source}")
    vector = torch.as_tensor(payload, dtype=torch.float32).reshape(-1)
    if vector.numel() == 0 or not bool(torch.isfinite(vector).all()):
        raise ValueError(f"non-finite or empty embedding in {source}")
    norm = vector.norm()
    if not bool(torch.isfinite(norm)) or float(norm) <= 1e-12:
        raise ValueError(f"zero/non-finite embedding norm in {source}")
    return (vector / norm).cpu()


def load_ecapa(path_obj: str | Path) -> torch.Tensor:
    source = Path(path_obj)
    return vector_from_payload(torch.load(source, map_location="cpu", weights_only=False), source)


def ecapa_cache_provenance(cache_path: Path, *, audio: Path, audio_sha256: str | None) -> dict[str, Any]:
    """Describe whether an inherited ECAPA cache is fully attributable.

    The historical cache key contains path+model-label but not audio content SHA.
    We retain valid legacy vectors, while making that weaker identity explicit in
    the ledger rather than claiming that it was created by this cleaner.
    """
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return {
            "identity_state": "existing_unattributed",
            "reason": "legacy_payload_not_mapping",
            "cache_sha256": sha256_file(cache_path),
        }
    stored_audio = payload.get("audio")
    stored_audio_sha = payload.get("audio_sha256")
    path_matches = stored_audio == str(audio)
    sha_matches = audio_sha256 is not None and stored_audio_sha == audio_sha256
    return {
        "identity_state": "fully_attributed" if path_matches and sha_matches else "existing_unattributed",
        "cache_sha256": sha256_file(cache_path),
        "stored_audio_path": stored_audio if isinstance(stored_audio, str) else None,
        "audio_path_matches": path_matches,
        "stored_audio_sha256": stored_audio_sha if isinstance(stored_audio_sha, str) else None,
        "audio_sha256_matches": sha_matches,
        "model_source": payload.get("model_source") if isinstance(payload.get("model_source"), str) else None,
        "backend": payload.get("backend") if isinstance(payload.get("backend"), str) else None,
    }


def load_wavlm(path_obj: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
    source = Path(path_obj)
    if source.suffix == ".npy":
        return vector_from_payload(np.load(source), source), {"format": "npy"}
    payload = torch.load(source, map_location="cpu", weights_only=False)
    metadata: dict[str, Any] = {}
    if isinstance(payload, dict):
        # Keep only JSON-safe provenance.  ``speaker_embedding`` is commonly a
        # duplicate tensor of ``embedding`` and must not leak into JSONL scores.
        for key, value in payload.items():
            if key in {"embedding", "speaker_embedding", "emb", "xvector", "vector"}:
                continue
            if isinstance(value, (str, int, float, bool, type(None))):
                metadata[key] = value
            elif isinstance(value, (list, tuple)) and all(
                isinstance(item, (str, int, float, bool, type(None))) for item in value
            ):
                metadata[key] = list(value)
    return vector_from_payload(payload, source), metadata


def vector_descriptor(vector: torch.Tensor, embedding_path: Path | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dtype": str(vector.dtype).replace("torch.", ""),
        "dim": int(vector.numel()),
        "norm": float(vector.norm().item()),
    }
    if embedding_path is not None:
        payload["embedding_path"] = str(embedding_path)
        payload["embedding_sha256"] = sha256_file(embedding_path)
    return payload


def edge_metrics(source: torch.Tensor, target: torch.Tensor, ref: torch.Tensor, prompt: torch.Tensor) -> dict[str, float]:
    if len({int(source.numel()), int(target.numel()), int(ref.numel()), int(prompt.numel())}) != 1:
        raise ValueError(
            "speaker embedding dimensionality mismatch: "
            f"source={source.numel()} target={target.numel()} ref={ref.numel()} prompt={prompt.numel()}"
        )
    values = {
        "u1prime_u1": float(torch.dot(source, target)),
        "u1prime_u2": float(torch.dot(source, ref)),
        "u2_u1": float(torch.dot(ref, target)),
        "u1prime_seedvc_prompt": float(torch.dot(source, prompt)),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("non-finite cosine in four-edge metrics")
    values["low_edge_ceiling"] = max(values["u1prime_u1"], values["u1prime_u2"])
    values["high_edge_floor"] = min(values["u2_u1"], values["u1prime_seedvc_prompt"])
    values["four_edge_gap"] = values["high_edge_floor"] - values["low_edge_ceiling"]
    return values


def gate_metrics(metrics: dict[str, float] | None, thresholds: dict[str, float]) -> dict[str, Any]:
    if metrics is None:
        return {"pass": False, "conditions": {}, "reason": "metrics_unavailable"}
    conditions = {
        "low_edge_ceiling": metrics["low_edge_ceiling"] <= thresholds["low_max"],
        "u2_u1": metrics["u2_u1"] >= thresholds["u2_u1_min"],
        "u1prime_seedvc_prompt": metrics["u1prime_seedvc_prompt"] >= thresholds["u1prime_prompt_min"],
        "four_edge_gap": metrics["four_edge_gap"] >= thresholds["gap_min"],
    }
    failed = [name for name, passed in conditions.items() if not passed]
    return {
        "pass": not failed,
        "conditions": conditions,
        "failed_conditions": failed,
        "thresholds": thresholds,
    }


def cached_prompt_path(ecapa_root: Path, prompt_audio: str) -> Path:
    return ecapa_root / f"{sha1_id(prompt_audio, 'speechbrain_ecapa')}.pt"


def cached_wavlm_path(cache_root: Path, audio_path: str) -> Path:
    return cache_root / f"{sha1_id(audio_path, 'microsoft/wavlm-base-plus-sv')}.pt"


def stage_dir(output_root: Path) -> Path:
    return output_root / "stages"


def logs_dir(output_root: Path) -> Path:
    return output_root / "logs"


def plan_path(output_root: Path) -> Path:
    return stage_dir(output_root) / "joined_plan.jsonl"


def prompt_plan_path(output_root: Path) -> Path:
    return stage_dir(output_root) / "prompt_ecapa_plan.jsonl"


def ecapa_score_path(output_root: Path, shard_index: int, shard_count: int) -> Path:
    return stage_dir(output_root) / f"ecapa_scores.shard{shard_index:02d}-of{shard_count:02d}.jsonl"


def wavlm_plan_path(output_root: Path) -> Path:
    return stage_dir(output_root) / "wavlm_cache_plan.jsonl"


def wavlm_score_path(output_root: Path, shard_index: int, shard_count: int) -> Path:
    return stage_dir(output_root) / f"wavlm_scores.shard{shard_index:02d}-of{shard_count:02d}.jsonl"


def stage_marker(output_root: Path, stage: str, shard_index: int | None = None, shard_count: int | None = None) -> Path:
    suffix = "" if shard_index is None else f".shard{shard_index:02d}-of{int(shard_count):02d}"
    return stage_dir(output_root) / f"{stage}{suffix}.done.json"


def require_marker(output_root: Path, stage: str, *, shard_count: int | None = None) -> list[dict[str, Any]]:
    if shard_count is None:
        marker = stage_marker(output_root, stage)
        if not marker.is_file():
            raise RuntimeError(f"required stage marker is missing: {marker}")
        return [load_json(marker)]
    markers: list[dict[str, Any]] = []
    for shard_index in range(shard_count):
        marker = stage_marker(output_root, stage, shard_index, shard_count)
        if not marker.is_file():
            raise RuntimeError(f"required shard marker is missing: {marker}")
        markers.append(load_json(marker))
    return markers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "prepare",
            "prompt-ecapa",
            "ecapa-score",
            "wavlm-plan",
            "wavlm-cache",
            "wavlm-score",
            "finalize",
            "fresh-audit",
        ),
    )
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--legacy-filtered-jsonl", type=Path, default=DEFAULT_LEGACY_FILTERED)
    parser.add_argument("--source-jobs-jsonl", type=Path, default=DEFAULT_SOURCE_JOBS)
    parser.add_argument("--source-results-jsonl", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--ecapa-cache-root", type=Path, default=DEFAULT_ECAPA_ROOT)
    parser.add_argument("--ecapa-model", type=Path, default=DEFAULT_ECAPA_MODEL)
    parser.add_argument("--wavlm-model", default="microsoft/wavlm-base-plus-sv")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    parser.add_argument("--expected-input-sha256", default=EXPECTED_INPUT_SHA256)
    parser.add_argument("--max-rows", type=int, default=0, help="Use a separate output root for smoke runs.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--audit-size", type=int, default=32)
    parser.add_argument("--audit-seed", type=int, default=20260715)
    parser.add_argument(
        "--min-fresh-ecapa-cosine",
        type=float,
        default=0.999,
        help="Minimum fresh-vs-cache ECAPA cosine (same single-file inference path).",
    )
    parser.add_argument(
        "--min-fresh-wavlm-cosine",
        type=float,
        default=0.95,
        help="Minimum WavLM-SV cosine; batch padding can produce small, documented drift.",
    )
    parser.add_argument(
        "--min-fresh-cache-cosine",
        type=float,
        default=None,
        help="Deprecated compatibility override: apply one threshold to both encoders.",
    )
    parser.add_argument("--force", action="store_true", help="Replace only outputs for this stage; never source manifests/caches.")
    parser.add_argument(
        "--rebuild-source-lookup",
        action="store_true",
        help="Rebuild the exact on-disk source job/result lookup (normally it is safely reused).",
    )
    parser.add_argument("--verify-input-sha256", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--audio-sha256", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.max_rows < 0:
        parser.error("--max-rows must be non-negative")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("require 0 <= --shard-index < --shard-count")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.audit_size < 1:
        parser.error("--audit-size must be positive")
    if args.min_fresh_cache_cosine is not None:
        args.min_fresh_ecapa_cosine = args.min_fresh_cache_cosine
        args.min_fresh_wavlm_cosine = args.min_fresh_cache_cosine
    if args.verify_input_sha256 is None:
        args.verify_input_sha256 = args.max_rows == 0
    return args


def prepare_output_root(args: argparse.Namespace) -> Path:
    output_root = path(args.output_root)
    if output_root == path(args.input_jsonl).parent:
        raise RuntimeError("output root must differ from original prepared-manifest root")
    output_root.mkdir(parents=True, exist_ok=True)
    stage_dir(output_root).mkdir(parents=True, exist_ok=True)
    logs_dir(output_root).mkdir(parents=True, exist_ok=True)
    return output_root


def file_identity(path_obj: Path, *, with_sha256: bool = True) -> dict[str, Any]:
    stat = path_obj.stat()
    payload = {
        "path": str(path_obj),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if with_sha256:
        payload["sha256"] = sha256_file(path_obj)
    return payload


def directory_identity(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files = sorted(item for item in directory.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    entries = []
    for item in files:
        relative = item.relative_to(directory).as_posix()
        item_hash = sha256_file(item)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item_hash.encode("ascii"))
        digest.update(b"\0")
        entries.append({"relative_path": relative, "bytes": item.stat().st_size, "sha256": item_hash})
    return {"path": str(directory), "tree_sha256": digest.hexdigest(), "files": entries}


def snapshot_code(output_root: Path, *, force: bool) -> dict[str, Any]:
    """Freeze every local implementation file the cleaner imports directly."""
    snapshot_root = output_root / "code_snapshot"
    entries: list[dict[str, Any]] = []
    for source in CODE_SNAPSHOT_FILES:
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"code snapshot source missing: {source}")
        relative = source.relative_to(ROOT)
        target = snapshot_root / relative
        source_hash = sha256_file(source)
        if target.exists() and sha256_file(target) != source_hash:
            if not force:
                raise RuntimeError(
                    f"code snapshot differs from current source: {target}; rerun prepare with --force after review"
                )
            target.unlink()
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        entries.append(
            {
                "source_path": str(source),
                "snapshot_path": str(target),
                "relative_path": relative.as_posix(),
                "sha256": source_hash,
            }
        )
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["relative_path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\0")
    return {"root": str(snapshot_root), "tree_sha256": digest.hexdigest(), "files": entries}


def assert_code_snapshot(output_root: Path) -> None:
    identity = load_input_identity(output_root)
    snapshot = identity.get("code_snapshot")
    if not isinstance(snapshot, dict):
        raise RuntimeError("INPUT_IDENTITY lacks code_snapshot; rerun --stage prepare with --force")
    for entry in snapshot.get("files", []):
        source = Path(entry["source_path"])
        frozen = Path(entry["snapshot_path"])
        expected = entry["sha256"]
        if not frozen.is_file() or sha256_file(frozen) != expected:
            raise RuntimeError(f"code snapshot was modified or missing: {frozen}")
        if not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(
                f"cleaner code changed after prepare ({source}); rerun --stage prepare with --force before continuing"
            )


def wavlm_identity(model_name: str) -> dict[str, Any]:
    # A local snapshot is mandatory for production (``local_files_only=True``).
    slug = "models--" + model_name.replace("/", "--")
    cache_roots = []
    for value in (
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        os.environ.get("TRANSFORMERS_CACHE"),
        str(Path.home() / ".cache/huggingface/hub"),
        "/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/huggingface/hub",
    ):
        if value and Path(value) not in cache_roots:
            cache_roots.append(Path(value))
    snapshot_dirs: list[Path] = []
    for root in cache_roots:
        snapshots = root / slug / "snapshots"
        if snapshots.is_dir():
            snapshot_dirs.extend(sorted(item for item in snapshots.iterdir() if item.is_dir()))
    def usable_snapshot(candidate: Path) -> bool:
        if not (candidate / "config.json").is_file() or not (candidate / "preprocessor_config.json").is_file():
            return False
        weight_candidates = (candidate / "pytorch_model.bin", candidate / "model.safetensors")
        return any(item.is_file() and item.stat().st_size > 1_000_000 for item in weight_candidates)

    snapshot_dirs = [candidate for candidate in snapshot_dirs if usable_snapshot(candidate)]
    if not snapshot_dirs:
        raise FileNotFoundError(f"no local Hugging Face snapshot for {model_name}; checked {cache_roots}")
    # A model id normally has one snapshot.  Choose newest only when cache contains
    # multiple versions and record the exact resolved tree hash.
    snapshot = max(snapshot_dirs, key=lambda item: item.stat().st_mtime_ns)
    identity = directory_identity(snapshot)
    identity["model_name"] = model_name
    identity["snapshot"] = snapshot.name
    return identity


def silence_transformers_progress() -> None:
    """Keep long shard logs readable; model identity remains recorded separately."""
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()
    except Exception:
        pass


def nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, bytes, list, tuple, dict)):
        return bool(value)
    return True


def add_quality_condition(conditions: dict[str, bool], name: str, passed: bool) -> None:
    conditions[name] = bool(passed)


def base_quality(row: dict[str, Any], roles: dict[str, str], ecapa_paths: dict[str, str]) -> dict[str, Any]:
    """Reapply only v2's actual quality contract, without adding ASR thresholds."""
    meta = row.get("moss_codecvc_meta") if isinstance(row.get("moss_codecvc_meta"), dict) else {}
    policy = row.get("text_policy") if isinstance(row.get("text_policy"), dict) else {}
    known = row.get("v2_real_no_text_known_content") if isinstance(row.get("v2_real_no_text_known_content"), dict) else {}
    conditions: dict[str, bool] = {}
    add_quality_condition(conditions, "mode_no_text", row.get("moss_codecvc_mode") == "no_text")
    add_quality_condition(conditions, "content_keep", row.get("content_keep") is True)
    add_quality_condition(conditions, "content_filter_reason_keep", row.get("content_filter_reason") == "keep")
    add_quality_condition(conditions, "content_token_keep", row.get("content_token_keep") is True)
    add_quality_condition(conditions, "content_token_filter_reason_keep", row.get("content_token_filter_reason") == "keep")
    add_quality_condition(conditions, "content_tokens_nonempty", nonempty(row.get("content_token_ids")))
    try:
        token_length_valid = int(row.get("content_token_length", 0)) > 0
    except (TypeError, ValueError):
        token_length_valid = False
    add_quality_condition(conditions, "content_token_length_positive", token_length_valid)
    src_text = normalise_text(row.get("source_text"))
    tgt_text = normalise_text(row.get("target_text"))
    add_quality_condition(conditions, "normalised_source_target_text_equal", bool(src_text) and src_text == tgt_text)
    add_quality_condition(conditions, "text_policy_source_target_equal", policy.get("target_text_equals_source_text") is True)
    add_quality_condition(conditions, "known_content_edit_distance_zero", known.get("source_target_edit_distance") == 0)
    duration = finite_float(row.get("duration_ratio_tgt_src"))
    add_quality_condition(conditions, "duration_ratio_finite_positive", duration is not None and duration > 0.0)
    repeat_score = finite_float(row.get("repeat_score"))
    # v2 materialisation writes the post-filter contract as exactly zero; unlike
    # a generic ASR filter, this does not introduce an unrecorded new threshold.
    add_quality_condition(conditions, "repeat_score_zero_contract", repeat_score == 0.0)
    ref_text = normalise_text(row.get("timbre_ref_text"))
    add_quality_condition(
        conditions,
        "no_ref_content_leak",
        not (bool(ref_text) and bool(tgt_text) and ref_text == tgt_text),
    )
    add_quality_condition(
        conditions,
        "role_pair_type",
        meta.get("pair_type") == "v2_real_target_seedvc_source_no_text",
    )
    add_quality_condition(conditions, "source_is_seedvc", row.get("v2_real_target", {}).get("source_is_seedvc_output") is True if isinstance(row.get("v2_real_target"), dict) else False)
    add_quality_condition(conditions, "target_is_real", row.get("v2_real_target", {}).get("target_is_real_audio") is True if isinstance(row.get("v2_real_target"), dict) else False)

    required_paths = {
        "u1prime_source_audio": roles["source_audio"],
        "u1_target_audio": roles["target_audio"],
        "u2_timbre_ref_audio": roles["ref_audio"],
        "ecapa_u1prime": ecapa_paths["source"],
        "ecapa_u1": ecapa_paths["target"],
        "ecapa_u2": ecapa_paths["ref"],
        "wavlm_u2": str(row.get("speaker_vec_path") or ""),
        "source_prosody": str(row.get("source_prosody_path") or ""),
        "target_prosody": str(row.get("target_prosody_path") or ""),
        "source_wavlm_features": str(row.get("source_wavlm_features_path") or ""),
        "source_wavlm_bnf": str(row.get("source_wavlm_bnf_features_path") or ""),
    }
    for name, value in required_paths.items():
        add_quality_condition(conditions, f"path_exists:{name}", bool(value) and Path(value).is_file())
    add_quality_condition(conditions, "audio_codes_nonempty", nonempty(row.get("audio_codes")))
    add_quality_condition(conditions, "reference_audio_codes_nonempty", nonempty(row.get("reference_audio_codes")))
    try:
        tokens_positive = int(row.get("tokens", 0)) > 0
    except (TypeError, ValueError):
        tokens_positive = False
    add_quality_condition(conditions, "tokens_positive", tokens_positive)
    failures = [name for name, passed in conditions.items() if not passed]
    return {
        "pass": not failures,
        "conditions": conditions,
        "failed_conditions": failures,
        "duration_ratio_tgt_src": duration,
        "repeat_score": repeat_score,
    }


def create_source_lookup(database: Path, jobs_path: Path, results_path: Path, *, force: bool) -> dict[str, Any]:
    """Build an on-disk exact lookup; results order is deliberately not assumed."""
    marker = database.with_suffix(".done.json")
    if database.is_file() and marker.is_file() and not force:
        metadata = load_json(marker)
        if metadata.get("jobs", {}).get("path") == str(jobs_path) and metadata.get("results", {}).get("path") == str(results_path):
            return metadata
    if database.exists():
        database.unlink()
    if marker.exists():
        marker.unlink()
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(
            "CREATE TABLE jobs (original_index INTEGER PRIMARY KEY, job_id TEXT UNIQUE NOT NULL, output_audio TEXT NOT NULL, prosody_ref_audio TEXT NOT NULL, prompt_audio TEXT NOT NULL, prompt_speaker_id TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE results (job_id TEXT PRIMARY KEY, ok INTEGER NOT NULL, output_audio TEXT NOT NULL, prosody_ref_audio TEXT, prompt_audio TEXT)"
        )
        jobs_count = 0
        batch: list[tuple[Any, ...]] = []
        for physical_index, row in enumerate(iter_jsonl(jobs_path)):
            job_id = str(row.get("job_id") or "")
            job_original_index = original_index(job_id)
            if job_original_index != physical_index:
                raise RuntimeError(
                    f"source-job physical/index mismatch at line={physical_index}: job_id={job_id!r}"
                )
            values = (
                job_original_index,
                job_id,
                str(row.get("output_audio") or ""),
                str(row.get("prosody_ref_audio") or ""),
                str(row.get("timbre_ref_audio") or ""),
                str(row.get("timbre_ref_speaker_id") or ""),
            )
            if not all(values[2:]):
                raise RuntimeError(f"source job has missing role path/id: {job_id}")
            batch.append(values)
            jobs_count += 1
            if len(batch) >= 5000:
                connection.executemany("INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?)", batch)
                connection.commit()
                batch.clear()
        if batch:
            connection.executemany("INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?)", batch)
            connection.commit()

        results_count = 0
        batch = []
        for row in iter_jsonl(results_path):
            job_id = str(row.get("job_id") or "")
            values = (
                job_id,
                int(row.get("ok") is True),
                str(row.get("output_audio") or row.get("audio") or ""),
                str(row.get("prosody_ref_audio") or ""),
                str(row.get("timbre_ref_audio") or ""),
            )
            if not job_id or not values[2]:
                raise RuntimeError(f"source result has missing job/output: {job_id!r}")
            batch.append(values)
            results_count += 1
            if len(batch) >= 5000:
                connection.executemany("INSERT INTO results VALUES (?, ?, ?, ?, ?)", batch)
                connection.commit()
                batch.clear()
        if batch:
            connection.executemany("INSERT INTO results VALUES (?, ?, ?, ?, ?)", batch)
            connection.commit()
    finally:
        connection.close()
    metadata = {
        "status": "complete",
        "created_at": utc_now(),
        "database": str(database),
        "jobs": {**file_identity(jobs_path), "rows": jobs_count},
        "results": {**file_identity(results_path), "rows": results_count},
    }
    atomic_json(marker, metadata)
    return metadata


def join_source_row(connection: sqlite3.Connection, row: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    original = original_index(row.get("sample_id"))
    job = connection.execute(
        "SELECT job_id, output_audio, prosody_ref_audio, prompt_audio, prompt_speaker_id FROM jobs WHERE original_index=?",
        (original,),
    ).fetchone()
    if job is None:
        raise RuntimeError(f"no source job for sample_id={row.get('sample_id')!r}, original_index={original}")
    job_id, output_audio, prosody_ref_audio, prompt_audio, prompt_speaker_id = job
    result = connection.execute(
        "SELECT ok, output_audio, prosody_ref_audio, prompt_audio FROM results WHERE job_id=?",
        (job_id,),
    ).fetchone()
    if result is None:
        raise RuntimeError(f"no source result for job_id={job_id}")
    result_ok, result_output, result_prosody, result_prompt = result
    meta = row.get("moss_codecvc_meta")
    if not isinstance(meta, dict):
        raise RuntimeError(f"row lacks moss_codecvc_meta: {row.get('sample_id')}")
    roles = {
        "source_audio": str(meta.get("source_audio") or ""),
        "target_audio": str(meta.get("target_audio") or ""),
        "ref_audio": str(meta.get("timbre_ref_audio") or ""),
        "prompt_audio": str(prompt_audio),
    }
    checks = {
        "source_job_output_matches_u1prime": str(output_audio) == roles["source_audio"],
        "source_job_prosody_matches_u1": str(prosody_ref_audio) == roles["target_audio"],
        "source_result_ok": bool(result_ok),
        "source_result_output_matches_u1prime": str(result_output) == roles["source_audio"],
        "source_result_prosody_matches_u1": str(result_prosody) == roles["target_audio"],
        "source_result_prompt_matches_job": str(result_prompt) == roles["prompt_audio"],
        "prompt_audio_exists": Path(roles["prompt_audio"]).is_file(),
        "prompt_speaker_differs_target": str(prompt_speaker_id) != str(row.get("target_speaker_id") or ""),
        "prompt_dataset_differs_target": str(prompt_speaker_id).split(":", 1)[0]
        != str(row.get("target_speaker_id") or "").split(":", 1)[0],
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            f"strict source join failed sample_id={row.get('sample_id')} job_id={job_id}: {failures}"
        )
    return roles, {"job_id": str(job_id), "original_row_index": original, "checks": checks}


def load_input_identity(output_root: Path) -> dict[str, Any]:
    identity_path = output_root / "INPUT_IDENTITY.json"
    if not identity_path.is_file():
        raise RuntimeError(f"run --stage prepare first; missing {identity_path}")
    return load_json(identity_path)


def prepare(args: argparse.Namespace, output_root: Path) -> None:
    input_path = path(args.input_jsonl)
    # Do not resolve this path before recording it: being a symlink is itself an
    # important safety property of the historical alias.
    legacy_path = Path(args.legacy_filtered_jsonl).expanduser()
    if not legacy_path.is_absolute():
        legacy_path = Path.cwd() / legacy_path
    legacy_path = legacy_path.absolute()
    jobs_path = path(args.source_jobs_jsonl)
    results_path = path(args.source_results_jsonl)
    ecapa_root = path(args.ecapa_cache_root)
    ecapa_model = path(args.ecapa_model)
    marker = stage_marker(output_root, "prepare")
    if marker.is_file() and not args.force:
        print(json.dumps({"status": "already_complete", "stage": "prepare", "marker": str(marker)}))
        return
    for required in (input_path, jobs_path, results_path, ecapa_root, ecapa_model):
        if not required.exists():
            raise FileNotFoundError(required)
    if legacy_path.exists() or legacy_path.is_symlink():
        legacy_resolved = legacy_path.resolve()
        if legacy_resolved != input_path.resolve():
            raise RuntimeError(f"legacy filtered link must resolve to canonical input; got {legacy_resolved}")
    else:
        raise FileNotFoundError(f"legacy filtered symlink missing: {legacy_path}")
    if args.verify_input_sha256:
        actual_sha = sha256_file(input_path)
        if actual_sha != args.expected_input_sha256:
            raise RuntimeError(
                f"input SHA256 mismatch: expected={args.expected_input_sha256}, actual={actual_sha}"
            )
    else:
        actual_sha = None
    if args.max_rows == 0:
        offsets = np.fromfile(str(input_path) + ".offsets.u64", dtype="<u8")
        if len(offsets) != args.expected_rows:
            raise RuntimeError(f"offset row count mismatch: expected {args.expected_rows}, got {len(offsets)}")

    lookup_meta = create_source_lookup(
        stage_dir(output_root) / "source_join.sqlite",
        jobs_path,
        results_path,
        force=args.rebuild_source_lookup,
    )
    code_snapshot = snapshot_code(output_root, force=args.force)
    identities = {
        "status": "prepared",
        "created_at": utc_now(),
        "run_kind": "smoke" if args.max_rows else "full",
        "input": {
            **file_identity(input_path, with_sha256=False),
            "registered_sha256": args.expected_input_sha256,
            "verified_sha256": actual_sha,
            "sha256_verification_performed": bool(args.verify_input_sha256),
        },
        "legacy_filtered_symlink": {
            "path": str(legacy_path),
            "is_symlink": legacy_path.is_symlink(),
            "resolves_to": str(legacy_resolved),
            "matches_input": legacy_resolved == input_path.resolve(),
            "warning": "legacy link is not a four-edge filtered artifact",
        },
        "source_join": lookup_meta,
        "models": {
            "speechbrain_ecapa": directory_identity(ecapa_model),
            "wavlm_sv": wavlm_identity(args.wavlm_model),
        },
        "code_snapshot": code_snapshot,
        "thresholds": {"ecapa_balanced": ECAPA_THRESHOLDS, "wavlm_sv_balanced": WAVLM_THRESHOLDS},
        "roles": {
            "U1": "real target audio",
            "U1prime": "SeedVC(U1 content/prosody, third-speaker prompt) synthetic source",
            "U2": "same-speaker real timbre reference",
            "prompt": "third-speaker audio used by SeedVC for U1prime",
        },
        "scope": {
            "cross_episode_u2_changed": False,
            "seedvc_regenerated": False,
            "old_filtered_link_used_as_input": False,
        },
    }
    atomic_json(output_root / "INPUT_IDENTITY.json", identities)

    final_plan = plan_path(output_root)
    final_prompt_plan = prompt_plan_path(output_root)
    if args.force:
        for item in (final_plan, final_prompt_plan):
            if item.exists():
                item.unlink()
    if final_plan.exists() or final_prompt_plan.exists():
        raise RuntimeError("prepare outputs already exist; use --force only after reviewing their identity")
    plan_tmp, plan_handle = jsonl_writer(final_plan)
    prompt_tmp, prompt_handle = jsonl_writer(final_prompt_plan)
    prompt_records: dict[str, dict[str, Any]] = {}
    seen_sample_ids: set[str] = set()
    seen_original_indices: set[int] = set()
    language_counts: Counter[str] = Counter()
    base_passes = 0
    structural = Counter()
    started = time.time()
    rows = 0
    connection = sqlite3.connect(stage_dir(output_root) / "source_join.sqlite")
    try:
        for row_index, _raw, row in iter_raw_jsonl(input_path, max_rows=args.max_rows):
            sample_id = str(row.get("sample_id") or "")
            if not sample_id or sample_id in seen_sample_ids:
                raise RuntimeError(f"missing or duplicate sample_id at prepared row {row_index}: {sample_id!r}")
            seen_sample_ids.add(sample_id)
            original = original_index(sample_id)
            if original in seen_original_indices:
                raise RuntimeError(f"duplicate original row index {original} for sample_id={sample_id}")
            seen_original_indices.add(original)
            roles, join = join_source_row(connection, row)
            ecapa_paths = {
                "source": str(row.get("source_speaker_embedding_path") or ""),
                "target": str(row.get("target_speaker_embedding_path") or ""),
                "ref": str(row.get("timbre_ref_speaker_embedding_path") or ""),
                "prompt": str(cached_prompt_path(ecapa_root, roles["prompt_audio"])),
            }
            quality = base_quality(row, roles, ecapa_paths)
            base_passes += int(quality["pass"])
            language = str(row.get("language") or "unknown")
            language_counts[language] += 1
            for name, passed in join["checks"].items():
                structural[name] += int(bool(passed))
            record = {
                "row_index": row_index,
                "original_row_index": original,
                "sample_id": sample_id,
                "language": language,
                "target_speaker_id": str(row.get("target_speaker_id") or ""),
                "timbre_ref_speaker_id": str(row.get("timbre_ref_speaker_id") or ""),
                "duration_ratio_tgt_src": quality["duration_ratio_tgt_src"],
                "roles": roles,
                "ecapa_paths": ecapa_paths,
                "wavlm_ref_path": str(row.get("speaker_vec_path") or ""),
                "base_quality": quality,
                "source_join": join,
            }
            plan_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            prompt = prompt_records.get(roles["prompt_audio"])
            if prompt is None:
                cache_path = Path(ecapa_paths["prompt"])
                prompt_records[roles["prompt_audio"]] = {
                    "prompt_audio": roles["prompt_audio"],
                    "ecapa_path": str(cache_path),
                    "cache_key": cache_path.stem,
                    "initial_cache_exists": cache_path.is_file(),
                    "occurrences": 1,
                }
            else:
                prompt["occurrences"] += 1
            rows += 1
            if args.log_every and rows % args.log_every == 0:
                elapsed = time.time() - started
                print(f"[prepare] rows={rows} base_pass={base_passes} prompts={len(prompt_records)} elapsed={elapsed:.1f}s", flush=True)
    finally:
        plan_handle.close()
        connection.close()
    for record in sorted(prompt_records.values(), key=lambda item: item["prompt_audio"]):
        prompt_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    prompt_handle.close()
    if args.max_rows == 0 and rows != args.expected_rows:
        raise RuntimeError(f"manifest row count mismatch: expected={args.expected_rows}, got={rows}")
    if rows == 0:
        raise RuntimeError("prepare found zero input rows")
    finish_jsonl(plan_tmp, final_plan)
    finish_jsonl(prompt_tmp, final_prompt_plan)
    prepared = load_input_identity(output_root)
    prepared["prepared_rows"] = rows
    prepared["prepared_languages"] = dict(sorted(language_counts.items()))
    prepared["base_quality_pass_rows"] = base_passes
    prepared["unique_prompt_audio"] = len(prompt_records)
    prepared["initial_prompt_ecapa_cache_hits"] = sum(int(item["initial_cache_exists"]) for item in prompt_records.values())
    prepared["initial_prompt_ecapa_cache_missing"] = len(prompt_records) - prepared["initial_prompt_ecapa_cache_hits"]
    prepared["structural_checks_pass_count"] = dict(structural)
    prepared["plan_jsonl_sha256"] = sha256_file(final_plan)
    prepared["prompt_plan_jsonl_sha256"] = sha256_file(final_prompt_plan)
    atomic_json(output_root / "INPUT_IDENTITY.json", prepared)
    summary = {
        "status": "complete",
        "stage": "prepare",
        "created_at": utc_now(),
        "rows": rows,
        "base_quality_pass_rows": base_passes,
        "unique_prompt_audio": len(prompt_records),
        "initial_prompt_ecapa_cache_hits": prepared["initial_prompt_ecapa_cache_hits"],
        "initial_prompt_ecapa_cache_missing": prepared["initial_prompt_ecapa_cache_missing"],
        "plan": str(final_plan),
        "prompt_plan": str(final_prompt_plan),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(marker, summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def load_ecapa_encoder(model_root: Path, device: torch.device):
    add_download_python_deps()
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier
    # SpeechBrain's parser accepts cuda:0 but emits a misleading fallback warning
    # for bare ``cuda`` even when CUDA_VISIBLE_DEVICES exposes only one GPU.
    device_name = "cuda:0" if device.type == "cuda" and device.index is None else str(device)
    kwargs: dict[str, Any] = {
        "source": str(model_root),
        "savedir": str(model_root),
        "overrides": {"pretrained_path": str(model_root)},
        "run_opts": {"device": device_name},
    }
    try:
        model = EncoderClassifier.from_hparams(**kwargs)
    except TypeError:
        kwargs.pop("run_opts")
        model = EncoderClassifier.from_hparams(**kwargs).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


@torch.inference_mode()
def encode_ecapa(model: Any, audio: str, device: torch.device) -> torch.Tensor:
    if hasattr(model, "encode_file"):
        vector = model.encode_file(audio).squeeze()
    else:
        signal = model.load_audio(audio).to(device)
        vector = model.encode_batch(signal.unsqueeze(0)).squeeze()
    vector = torch.as_tensor(vector, dtype=torch.float32).flatten()
    if vector.numel() == 0 or not bool(torch.isfinite(vector).all()):
        raise ValueError(f"ECAPA produced invalid embedding for {audio}")
    return torch.nn.functional.normalize(vector, dim=0).cpu()


def prompt_ecapa(args: argparse.Namespace, output_root: Path) -> None:
    require_marker(output_root, "prepare")
    assert_code_snapshot(output_root)
    output = stage_marker(output_root, "prompt-ecapa", args.shard_index, args.shard_count)
    if output.is_file() and not args.force:
        print(json.dumps({"status": "already_complete", "stage": "prompt-ecapa", "marker": str(output)}))
        return
    if args.force and output.exists():
        output.unlink()
    device = torch.device(args.device)
    model = None
    processed = cache_hits = encoded = failed = 0
    failures: list[dict[str, Any]] = []
    status_path = stage_dir(output_root) / f"prompt_ecapa_status.shard{args.shard_index:02d}-of{args.shard_count:02d}.jsonl"
    if args.force and status_path.exists():
        status_path.unlink()
    temporary, handle = jsonl_writer(status_path)
    started = time.time()
    try:
        for record in iter_jsonl(prompt_plan_path(output_root)):
            if deterministic_shard(str(record["cache_key"]), args.shard_count) != args.shard_index:
                continue
            processed += 1
            audio = Path(record["prompt_audio"])
            cache_path = Path(record["ecapa_path"])
            item: dict[str, Any] = {
                "prompt_audio": str(audio),
                "ecapa_path": str(cache_path),
                "cache_key": record["cache_key"],
                "occurrences": record["occurrences"],
            }
            try:
                if not audio.is_file():
                    raise FileNotFoundError(audio)
                audio_hash = sha256_file(audio) if args.audio_sha256 else None
                if cache_path.is_file():
                    vector = load_ecapa(cache_path)
                    cache_hits += 1
                    item.update(
                        {
                            "status": "existing_valid",
                            "embedding": vector_descriptor(vector, cache_path),
                            "cache_provenance": ecapa_cache_provenance(
                                cache_path, audio=audio, audio_sha256=audio_hash
                            ),
                        }
                    )
                else:
                    if model is None:
                        model = load_ecapa_encoder(path(args.ecapa_model), device)
                    vector = encode_ecapa(model, str(audio), device)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_tmp = cache_path.with_name(cache_path.name + ".tmp")
                    torch.save(
                        {
                            "embedding": vector,
                            "speaker_embedding": vector,
                            "embedding_dim": int(vector.numel()),
                            "backend": "speechbrain_ecapa",
                            "model_source": str(path(args.ecapa_model)),
                            "audio": str(audio),
                            "audio_sha256": audio_hash,
                            "role": "seedvc_prompt",
                            "created_at": utc_now(),
                        },
                        cache_tmp,
                    )
                    cache_tmp.replace(cache_path)
                    encoded += 1
                    item.update(
                        {
                            "status": "encoded",
                            "embedding": vector_descriptor(vector, cache_path),
                            "cache_provenance": {
                                "identity_state": "fully_attributed",
                                "cache_sha256": sha256_file(cache_path),
                                "stored_audio_path": str(audio),
                                "audio_path_matches": True,
                                "stored_audio_sha256": audio_hash,
                                "audio_sha256_matches": audio_hash is not None,
                                "model_source": str(path(args.ecapa_model)),
                                "backend": "speechbrain_ecapa",
                            },
                        }
                    )
                item["audio_sha256"] = audio_hash
            except Exception as exc:  # Error becomes a fail-closed row later.
                failed += 1
                item.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
                if len(failures) < 50:
                    failures.append(item)
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
            if args.log_every and processed % args.log_every == 0:
                print(
                    f"[prompt-ecapa] shard={args.shard_index}/{args.shard_count} processed={processed} "
                    f"encoded={encoded} hits={cache_hits} failed={failed} elapsed={time.time()-started:.1f}s",
                    flush=True,
                )
    finally:
        handle.close()
    finish_jsonl(temporary, status_path)
    summary = {
        "status": "complete",
        "stage": "prompt-ecapa",
        "created_at": utc_now(),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "processed": processed,
        "cache_hits": cache_hits,
        "encoded": encoded,
        "failed": failed,
        "failures_sample": failures,
        "status_jsonl": str(status_path),
        "status_jsonl_sha256": sha256_file(status_path),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(output, summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def ecapa_score(args: argparse.Namespace, output_root: Path) -> None:
    require_marker(output_root, "prepare")
    assert_code_snapshot(output_root)
    require_marker(output_root, "prompt-ecapa", shard_count=args.shard_count)
    output = stage_marker(output_root, "ecapa-score", args.shard_index, args.shard_count)
    scores_output = ecapa_score_path(output_root, args.shard_index, args.shard_count)
    if output.is_file() and not args.force:
        print(json.dumps({"status": "already_complete", "stage": "ecapa-score", "marker": str(output)}))
        return
    if args.force:
        for item in (output, scores_output):
            if item.exists():
                item.unlink()
    temporary, handle = jsonl_writer(scores_output)
    processed = base_failed = ecapa_pass = ecapa_failed = errors = 0
    error_samples: list[dict[str, Any]] = []
    started = time.time()
    try:
        for record in iter_jsonl(plan_path(output_root)):
            if int(record["row_index"]) % args.shard_count != args.shard_index:
                continue
            processed += 1
            metrics: dict[str, float] | None = None
            error: dict[str, str] | None = None
            if not bool(record["base_quality"]["pass"]):
                base_failed += 1
            else:
                try:
                    vectors = {
                        role: load_ecapa(record["ecapa_paths"][role])
                        for role in ("source", "target", "ref", "prompt")
                    }
                    metrics = edge_metrics(vectors["source"], vectors["target"], vectors["ref"], vectors["prompt"])
                except Exception as exc:
                    errors += 1
                    error = {"type": type(exc).__name__, "message": str(exc)}
                    if len(error_samples) < 50:
                        error_samples.append({"sample_id": record["sample_id"], **error})
            gate = gate_metrics(metrics, ECAPA_THRESHOLDS)
            ecapa_pass += int(bool(gate["pass"]))
            ecapa_failed += int(not bool(gate["pass"]))
            score = {
                "row_index": record["row_index"],
                "original_row_index": record["original_row_index"],
                "sample_id": record["sample_id"],
                "base_quality": record["base_quality"],
                "ecapa": {"metrics": metrics, "gate": gate, "error": error},
            }
            handle.write(json.dumps(score, ensure_ascii=False, sort_keys=True) + "\n")
            if args.log_every and processed % args.log_every == 0:
                print(
                    f"[ecapa-score] shard={args.shard_index}/{args.shard_count} processed={processed} "
                    f"pass={ecapa_pass} errors={errors} elapsed={time.time()-started:.1f}s",
                    flush=True,
                )
    finally:
        handle.close()
    finish_jsonl(temporary, scores_output)
    summary = {
        "status": "complete",
        "stage": "ecapa-score",
        "created_at": utc_now(),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "processed": processed,
        "base_quality_failed": base_failed,
        "ecapa_balanced_pass": ecapa_pass,
        "ecapa_balanced_fail": ecapa_failed,
        "encoder_errors": errors,
        "error_samples": error_samples,
        "scores": str(scores_output),
        "scores_sha256": sha256_file(scores_output),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(output, summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def wavlm_plan(args: argparse.Namespace, output_root: Path) -> None:
    require_marker(output_root, "prepare")
    assert_code_snapshot(output_root)
    require_marker(output_root, "ecapa-score", shard_count=args.shard_count)
    marker = stage_marker(output_root, "wavlm-plan")
    final = wavlm_plan_path(output_root)
    if marker.is_file() and not args.force:
        print(json.dumps({"status": "already_complete", "stage": "wavlm-plan", "marker": str(marker)}))
        return
    if args.force:
        for item in (marker, final):
            if item.exists():
                item.unlink()
    ecapa_by_row: dict[int, dict[str, Any]] = {}
    for shard_index in range(args.shard_count):
        for score in iter_jsonl(ecapa_score_path(output_root, shard_index, args.shard_count)):
            row_index = int(score["row_index"])
            if row_index in ecapa_by_row:
                raise RuntimeError(f"duplicate ECAPA score row {row_index}")
            ecapa_by_row[row_index] = score
    unique: dict[str, dict[str, Any]] = {}
    candidate_rows = 0
    for record in iter_jsonl(plan_path(output_root)):
        score = ecapa_by_row.get(int(record["row_index"]))
        if score is None:
            raise RuntimeError(f"missing ECAPA score for row {record['row_index']}")
        if not bool(score["base_quality"]["pass"]) or not bool(score["ecapa"]["gate"]["pass"]):
            continue
        candidate_rows += 1
        for role, audio_key in (("u1prime", "source_audio"), ("u1", "target_audio"), ("prompt", "prompt_audio")):
            audio = str(record["roles"][audio_key])
            cache = cached_wavlm_path(output_root / "cache" / "wavlm_sv", audio)
            item = unique.get(audio)
            if item is None:
                unique[audio] = {
                    "audio": audio,
                    "wavlm_path": str(cache),
                    "cache_key": cache.stem,
                    "roles": [role],
                    "rows": 1,
                }
            else:
                if role not in item["roles"]:
                    item["roles"].append(role)
                item["rows"] += 1
    temporary, handle = jsonl_writer(final)
    try:
        for item in sorted(unique.values(), key=lambda value: value["audio"]):
            item["roles"].sort()
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    finally:
        handle.close()
    finish_jsonl(temporary, final)
    summary = {
        "status": "complete",
        "stage": "wavlm-plan",
        "created_at": utc_now(),
        "ecapa_candidate_rows": candidate_rows,
        "unique_audio": len(unique),
        "output": str(final),
        "output_sha256": sha256_file(final),
    }
    atomic_json(marker, summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def load_wavlm_model_identity(output_root: Path) -> dict[str, Any]:
    identity = load_input_identity(output_root)
    return identity["models"]["wavlm_sv"]


def wavlm_cache(args: argparse.Namespace, output_root: Path) -> None:
    require_marker(output_root, "wavlm-plan")
    assert_code_snapshot(output_root)
    output = stage_marker(output_root, "wavlm-cache", args.shard_index, args.shard_count)
    status = stage_dir(output_root) / f"wavlm_cache_status.shard{args.shard_index:02d}-of{args.shard_count:02d}.jsonl"
    if output.is_file() and not args.force:
        print(json.dumps({"status": "already_complete", "stage": "wavlm-cache", "marker": str(output)}))
        return
    if args.force:
        for item in (output, status):
            if item.exists():
                item.unlink()
    jobs = [
        item
        for item in iter_jsonl(wavlm_plan_path(output_root))
        if deterministic_shard(str(item["cache_key"]), args.shard_count) == args.shard_index
    ]
    cache_root = output_root / "cache" / "wavlm_sv"
    cache_root.mkdir(parents=True, exist_ok=True)
    model_identity = load_wavlm_model_identity(output_root)
    encoder: FrozenWavLMSVEncoder | None = None
    device = torch.device(args.device)
    processed = cache_hits = encoded = failed = 0
    failure_samples: list[dict[str, Any]] = []
    temporary, handle = jsonl_writer(status)
    started = time.time()
    try:
        # Existing entries and fresh entries are emitted in the same order as plan.
        pending: list[dict[str, Any]] = []
        for item in jobs:
            cache = Path(item["wavlm_path"])
            audio = Path(item["audio"])
            processed += 1
            if cache.is_file():
                status_row: dict[str, Any] = {"audio": str(audio), "wavlm_path": str(cache), "roles": item["roles"]}
                try:
                    vector, metadata = load_wavlm(cache)
                    if metadata.get("audio") not in (None, str(audio)):
                        raise RuntimeError(f"cache path collision: metadata audio={metadata.get('audio')}")
                    status_row.update({"status": "existing_valid", "embedding": vector_descriptor(vector, cache)})
                    cache_hits += 1
                except Exception as exc:
                    failed += 1
                    status_row.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
                    if len(failure_samples) < 50:
                        failure_samples.append(status_row)
                handle.write(json.dumps(status_row, ensure_ascii=False, sort_keys=True) + "\n")
            else:
                pending.append(item)

        for begin in range(0, len(pending), args.batch_size):
            batch = pending[begin : begin + args.batch_size]
            paths = [str(item["audio"]) for item in batch]
            try:
                for audio in paths:
                    if not Path(audio).is_file():
                        raise FileNotFoundError(audio)
                    if encoder is None:
                        silence_transformers_progress()
                        encoder = FrozenWavLMSVEncoder(args.wavlm_model, local_files_only=True)
                vectors, mask = encoder(paths, device=device, dtype=torch.float32)
                if vectors is None or mask is None or not bool(mask.all()) or int(vectors.shape[0]) != len(batch):
                    raise RuntimeError("WavLM-SV batch encode returned an invalid mask/shape")
                vectors = vectors.detach().cpu().float()
                for item, vector in zip(batch, vectors, strict=True):
                    audio = Path(item["audio"])
                    cache = Path(item["wavlm_path"])
                    audio_hash = sha256_file(audio) if args.audio_sha256 else None
                    vector = torch.nn.functional.normalize(vector.flatten(), dim=0).cpu()
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    temporary_cache = cache.with_name(cache.name + ".tmp")
                    torch.save(
                        {
                            "embedding": vector,
                            "speaker_embedding": vector,
                            "embedding_dim": int(vector.numel()),
                            "backend": "microsoft/wavlm-base-plus-sv",
                            "model_name": args.wavlm_model,
                            "model_tree_sha256": model_identity["tree_sha256"],
                            "audio": str(audio),
                            "audio_sha256": audio_hash,
                            "roles": item["roles"],
                            "created_at": utc_now(),
                        },
                        temporary_cache,
                    )
                    temporary_cache.replace(cache)
                    status_row = {
                        "audio": str(audio),
                        "audio_sha256": audio_hash,
                        "wavlm_path": str(cache),
                        "roles": item["roles"],
                        "status": "encoded",
                        "embedding": vector_descriptor(vector, cache),
                    }
                    handle.write(json.dumps(status_row, ensure_ascii=False, sort_keys=True) + "\n")
                    encoded += 1
            except Exception as exc:
                # Batch failure is retried item-by-item so one malformed audio does
                # not hide the status of its neighbors.
                for item in batch:
                    audio = Path(item["audio"])
                    cache = Path(item["wavlm_path"])
                    status_row = {"audio": str(audio), "wavlm_path": str(cache), "roles": item["roles"]}
                    try:
                        if not audio.is_file():
                            raise FileNotFoundError(audio)
                        if encoder is None:
                            silence_transformers_progress()
                            encoder = FrozenWavLMSVEncoder(args.wavlm_model, local_files_only=True)
                        vector_batch, mask = encoder([str(audio)], device=device, dtype=torch.float32)
                        if vector_batch is None or mask is None or not bool(mask.all()):
                            raise RuntimeError("WavLM-SV single-item encode failed")
                        vector = torch.nn.functional.normalize(vector_batch[0].detach().cpu().float().flatten(), dim=0)
                        audio_hash = sha256_file(audio) if args.audio_sha256 else None
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        temporary_cache = cache.with_name(cache.name + ".tmp")
                        torch.save(
                            {
                                "embedding": vector,
                                "speaker_embedding": vector,
                                "embedding_dim": int(vector.numel()),
                                "backend": "microsoft/wavlm-base-plus-sv",
                                "model_name": args.wavlm_model,
                                "model_tree_sha256": model_identity["tree_sha256"],
                                "audio": str(audio),
                                "audio_sha256": audio_hash,
                                "roles": item["roles"],
                                "created_at": utc_now(),
                            },
                            temporary_cache,
                        )
                        temporary_cache.replace(cache)
                        status_row.update({"status": "encoded", "audio_sha256": audio_hash, "embedding": vector_descriptor(vector, cache)})
                        encoded += 1
                    except Exception as one_exc:
                        failed += 1
                        status_row.update({"status": "failed", "error_type": type(one_exc).__name__, "error": str(one_exc), "batch_error": str(exc)})
                        if len(failure_samples) < 50:
                            failure_samples.append(status_row)
                    handle.write(json.dumps(status_row, ensure_ascii=False, sort_keys=True) + "\n")
            if args.log_every and (begin + len(batch)) % args.log_every == 0:
                print(
                    f"[wavlm-cache] shard={args.shard_index}/{args.shard_count} processed={processed} "
                    f"encoded={encoded} hits={cache_hits} failed={failed} elapsed={time.time()-started:.1f}s",
                    flush=True,
                )
    finally:
        handle.close()
    finish_jsonl(temporary, status)
    summary = {
        "status": "complete",
        "stage": "wavlm-cache",
        "created_at": utc_now(),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "processed": processed,
        "cache_hits": cache_hits,
        "encoded": encoded,
        "failed": failed,
        "failure_samples": failure_samples,
        "status_jsonl": str(status),
        "status_jsonl_sha256": sha256_file(status),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(output, summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def wavlm_score(args: argparse.Namespace, output_root: Path) -> None:
    require_marker(output_root, "wavlm-plan")
    assert_code_snapshot(output_root)
    require_marker(output_root, "wavlm-cache", shard_count=args.shard_count)
    output = stage_marker(output_root, "wavlm-score", args.shard_index, args.shard_count)
    final = wavlm_score_path(output_root, args.shard_index, args.shard_count)
    if output.is_file() and not args.force:
        print(json.dumps({"status": "already_complete", "stage": "wavlm-score", "marker": str(output)}))
        return
    if args.force:
        for item in (output, final):
            if item.exists():
                item.unlink()
    ecapa_records = {
        int(item["row_index"]): item
        for item in iter_jsonl(ecapa_score_path(output_root, args.shard_index, args.shard_count))
    }
    temporary, handle = jsonl_writer(final)
    processed = candidate_rows = wavlm_pass = wavlm_failed = errors = 0
    error_samples: list[dict[str, Any]] = []
    started = time.time()
    try:
        for record in iter_jsonl(plan_path(output_root)):
            row_index = int(record["row_index"])
            if row_index % args.shard_count != args.shard_index:
                continue
            processed += 1
            ecapa = ecapa_records.pop(row_index, None)
            if ecapa is None:
                raise RuntimeError(f"missing matching ECAPA score for row {row_index}")
            metrics: dict[str, float] | None = None
            error: dict[str, str] | None = None
            eligible = bool(ecapa["base_quality"]["pass"]) and bool(ecapa["ecapa"]["gate"]["pass"])
            provenance: dict[str, Any] | None = None
            if eligible:
                candidate_rows += 1
                try:
                    roles = record["roles"]
                    cache_root = output_root / "cache" / "wavlm_sv"
                    paths = {
                        "source": cached_wavlm_path(cache_root, roles["source_audio"]),
                        "target": cached_wavlm_path(cache_root, roles["target_audio"]),
                        "prompt": cached_wavlm_path(cache_root, roles["prompt_audio"]),
                        "ref": Path(record["wavlm_ref_path"]),
                    }
                    source, source_meta = load_wavlm(paths["source"])
                    target, target_meta = load_wavlm(paths["target"])
                    prompt, prompt_meta = load_wavlm(paths["prompt"])
                    ref, ref_meta = load_wavlm(paths["ref"])
                    metrics = edge_metrics(source, target, ref, prompt)
                    provenance = {
                        "source": {"audio_path": roles["source_audio"], **vector_descriptor(source, paths["source"]), "cache_metadata": source_meta},
                        "target": {"audio_path": roles["target_audio"], **vector_descriptor(target, paths["target"]), "cache_metadata": target_meta},
                        "ref": {"audio_path": roles["ref_audio"], **vector_descriptor(ref, paths["ref"]), "cache_metadata": ref_meta},
                        "prompt": {"audio_path": roles["prompt_audio"], **vector_descriptor(prompt, paths["prompt"]), "cache_metadata": prompt_meta},
                    }
                except Exception as exc:
                    errors += 1
                    error = {"type": type(exc).__name__, "message": str(exc)}
                    if len(error_samples) < 50:
                        error_samples.append({"sample_id": record["sample_id"], **error})
            gate = gate_metrics(metrics, WAVLM_THRESHOLDS)
            wavlm_pass += int(bool(gate["pass"]))
            wavlm_failed += int(not bool(gate["pass"]))
            final_keep = bool(ecapa["base_quality"]["pass"]) and bool(ecapa["ecapa"]["gate"]["pass"]) and bool(gate["pass"])
            reasons: list[str] = []
            if not bool(ecapa["base_quality"]["pass"]):
                reasons.extend(f"base:{name}" for name in ecapa["base_quality"]["failed_conditions"])
            if not bool(ecapa["ecapa"]["gate"]["pass"]):
                reasons.extend(f"ecapa:{name}" for name in ecapa["ecapa"]["gate"].get("failed_conditions", [ecapa["ecapa"]["gate"].get("reason", "unavailable")]))
            if not bool(gate["pass"]):
                reasons.extend(f"wavlm:{name}" for name in gate.get("failed_conditions", [gate.get("reason", "unavailable")]))
            score = {
                "row_index": row_index,
                "original_row_index": record["original_row_index"],
                "sample_id": record["sample_id"],
                "language": record["language"],
                "target_speaker_id": record["target_speaker_id"],
                "timbre_ref_speaker_id": record["timbre_ref_speaker_id"],
                "duration_ratio_tgt_src": record["duration_ratio_tgt_src"],
                "roles": record["roles"],
                "ecapa_paths": record["ecapa_paths"],
                "wavlm_ref_path": record["wavlm_ref_path"],
                "source_join": record["source_join"],
                "base_quality": ecapa["base_quality"],
                "ecapa": ecapa["ecapa"],
                "wavlm_sv": {"metrics": metrics, "gate": gate, "error": error, "provenance": provenance},
                "final_keep": final_keep,
                "drop_reasons": reasons,
            }
            handle.write(json.dumps(score, ensure_ascii=False, sort_keys=True) + "\n")
            if args.log_every and processed % args.log_every == 0:
                print(
                    f"[wavlm-score] shard={args.shard_index}/{args.shard_count} processed={processed} "
                    f"candidates={candidate_rows} pass={wavlm_pass} errors={errors} elapsed={time.time()-started:.1f}s",
                    flush=True,
                )
    finally:
        handle.close()
    if ecapa_records:
        raise RuntimeError(f"unconsumed ECAPA scores in shard {args.shard_index}: {len(ecapa_records)}")
    finish_jsonl(temporary, final)
    summary = {
        "status": "complete",
        "stage": "wavlm-score",
        "created_at": utc_now(),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "processed": processed,
        "ecapa_candidate_rows": candidate_rows,
        "wavlm_balanced_pass": wavlm_pass,
        "wavlm_balanced_fail": wavlm_failed,
        "encoder_errors": errors,
        "error_samples": error_samples,
        "scores": str(final),
        "scores_sha256": sha256_file(final),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(output, summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def merge_score_shards(output_root: Path, shard_count: int) -> Iterator[dict[str, Any]]:
    paths = [wavlm_score_path(output_root, index, shard_count) for index in range(shard_count)]
    with ExitStack() as stack:
        handles = [stack.enter_context(item.open("r", encoding="utf-8")) for item in paths]
        heap: list[tuple[int, int, dict[str, Any]]] = []
        for shard_index, handle in enumerate(handles):
            line = handle.readline()
            if line:
                item = json.loads(line)
                heapq.heappush(heap, (int(item["row_index"]), shard_index, item))
        while heap:
            _row_index, shard_index, item = heapq.heappop(heap)
            yield item
            line = handles[shard_index].readline()
            if line:
                next_item = json.loads(line)
                heapq.heappush(heap, (int(next_item["row_index"]), shard_index, next_item))


def duration_bin(value: Any) -> str:
    number = finite_float(value)
    if number is None:
        return "nonfinite"
    if number < 0.5:
        return "<0.5"
    if number < 0.8:
        return "0.5-0.8"
    if number <= 1.25:
        return "0.8-1.25"
    if number <= 2.0:
        return "1.25-2.0"
    return ">2.0"


def finalize(args: argparse.Namespace, output_root: Path) -> None:
    require_marker(output_root, "prepare")
    assert_code_snapshot(output_root)
    require_marker(output_root, "wavlm-score", shard_count=args.shard_count)
    marker = stage_marker(output_root, "finalize")
    kept_path = output_root / "no_text.v2.four_edge_balanced.train.jsonl"
    dropped_path = output_root / "no_text.v2.four_edge_balanced.dropped.jsonl"
    scores_path = output_root / "four_edge_scores.jsonl"
    summary_path = output_root / "four_edge_filter_summary.json"
    if marker.is_file() and not args.force:
        print(json.dumps({"status": "already_complete", "stage": "finalize", "marker": str(marker)}))
        return
    if args.force:
        for item in (marker, kept_path, dropped_path, scores_path, summary_path):
            if item.exists():
                item.unlink()
    kept_tmp, kept_handle = jsonl_writer(kept_path)
    dropped_tmp, dropped_handle = jsonl_writer(dropped_path)
    scores_tmp, scores_handle = jsonl_writer(scores_path)
    input_path = path(args.input_jsonl)
    score_iterator = merge_score_shards(output_root, args.shard_count)
    rows = kept = dropped = 0
    language_counts: dict[str, Counter[str]] = defaultdict(Counter)
    duration_counts: dict[str, Counter[str]] = defaultdict(Counter)
    speakers: dict[str, set[str]] = {"input": set(), "kept": set(), "dropped": set()}
    structure: dict[str, Counter[str]] = defaultdict(Counter)
    started = time.time()
    try:
        for row_index, raw, row in iter_raw_jsonl(input_path, max_rows=args.max_rows):
            try:
                score = next(score_iterator)
            except StopIteration as exc:
                raise RuntimeError(f"scores ended early before input row {row_index}") from exc
            if int(score["row_index"]) != row_index or str(score["sample_id"]) != str(row.get("sample_id")):
                raise RuntimeError(
                    f"input/score collision at row {row_index}: score={score.get('sample_id')}, input={row.get('sample_id')}"
                )
            rows += 1
            outcome = "kept" if bool(score["final_keep"]) else "dropped"
            language = str(score["language"])
            language_counts["input"][language] += 1
            language_counts[outcome][language] += 1
            duration = duration_bin(score["duration_ratio_tgt_src"])
            duration_counts["input"][duration] += 1
            duration_counts[outcome][duration] += 1
            target_speaker = str(score["target_speaker_id"])
            speakers["input"].add(target_speaker)
            speakers[outcome].add(target_speaker)
            for key, value in score["source_join"]["checks"].items():
                structure["input"][key] += int(bool(value))
                structure[outcome][key] += int(bool(value))
            scores_handle.write(json.dumps(score, ensure_ascii=False, sort_keys=True) + "\n")
            if outcome == "kept":
                # Preserve the original complete JSON row rather than reconstructing it.
                kept_handle.write(raw.decode("utf-8") if raw.endswith(b"\n") else raw.decode("utf-8") + "\n")
                kept += 1
            else:
                ledger = {
                    "row_index": score["row_index"],
                    "original_row_index": score["original_row_index"],
                    "sample_id": score["sample_id"],
                    "language": score["language"],
                    "duration_ratio_tgt_src": score["duration_ratio_tgt_src"],
                    "ecapa": score["ecapa"],
                    "wavlm_sv": score["wavlm_sv"],
                    "base_quality": score["base_quality"],
                    "drop_reasons": score["drop_reasons"],
                }
                dropped_handle.write(json.dumps(ledger, ensure_ascii=False, sort_keys=True) + "\n")
                dropped += 1
            if args.log_every and rows % args.log_every == 0:
                print(f"[finalize] rows={rows} kept={kept} dropped={dropped} elapsed={time.time()-started:.1f}s", flush=True)
        try:
            unexpected = next(score_iterator)
        except StopIteration:
            unexpected = None
        if unexpected is not None:
            raise RuntimeError(f"scores contain extra row after input: {unexpected.get('row_index')}")
    finally:
        kept_handle.close()
        dropped_handle.close()
        scores_handle.close()
    if rows == 0 or kept + dropped != rows:
        raise RuntimeError(f"bad count identity: input={rows} kept={kept} dropped={dropped}")
    if args.max_rows == 0 and rows != args.expected_rows:
        raise RuntimeError(f"finalize row count mismatch expected={args.expected_rows}, got={rows}")
    finish_jsonl(kept_tmp, kept_path)
    finish_jsonl(dropped_tmp, dropped_path)
    finish_jsonl(scores_tmp, scores_path)
    # A second streaming verification catches malformed result files before summary.
    verified_kept = sum(1 for item in iter_jsonl(kept_path))
    verified_dropped = sum(1 for item in iter_jsonl(dropped_path))
    verified_scores = sum(1 for item in iter_jsonl(scores_path))
    if (verified_kept, verified_dropped, verified_scores) != (kept, dropped, rows):
        raise RuntimeError(
            f"post-write count mismatch kept={verified_kept}/{kept} dropped={verified_dropped}/{dropped} scores={verified_scores}/{rows}"
        )
    summary = {
        "status": "finalized_pending_fresh_audit",
        "created_at": utc_now(),
        "run_kind": "smoke" if args.max_rows else "full",
        "input_rows": rows,
        "kept_rows": kept,
        "dropped_rows": dropped,
        "count_identity_pass": rows == kept + dropped,
        "language_distribution": {name: dict(sorted(counter.items())) for name, counter in language_counts.items()},
        "target_speaker_distinct": {name: len(values) for name, values in speakers.items()},
        "duration_ratio_distribution": {name: dict(sorted(counter.items())) for name, counter in duration_counts.items()},
        "role_structure_checks": {name: dict(sorted(counter.items())) for name, counter in structure.items()},
        "thresholds": {"ecapa_balanced": ECAPA_THRESHOLDS, "wavlm_sv_balanced": WAVLM_THRESHOLDS},
        "outputs": {
            "kept": {"path": str(kept_path), "rows": kept, "sha256": sha256_file(kept_path)},
            "dropped": {"path": str(dropped_path), "rows": dropped, "sha256": sha256_file(dropped_path)},
            "scores": {"path": str(scores_path), "rows": rows, "sha256": sha256_file(scores_path)},
        },
        "fresh_cache_audit": "pending",
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(summary_path, summary)
    marker_payload = {
        "status": "complete",
        "stage": "finalize",
        "created_at": utc_now(),
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "input_rows": rows,
        "kept_rows": kept,
        "dropped_rows": dropped,
    }
    atomic_json(marker, marker_payload)
    print(json.dumps(marker_payload, ensure_ascii=False), flush=True)


def fresh_audit(args: argparse.Namespace, output_root: Path) -> None:
    require_marker(output_root, "finalize")
    assert_code_snapshot(output_root)
    completed_name = "SMOKE_COMPLETED.json" if args.max_rows else "COMPLETED.json"
    completed_path = output_root / completed_name
    if completed_path.is_file() and not args.force:
        print(json.dumps({"status": "already_complete", "stage": "fresh-audit", "marker": str(completed_path)}))
        return
    scores_file = output_root / "four_edge_scores.jsonl"
    rng = random.Random(args.audit_seed)
    reservoir: list[dict[str, Any]] = []
    eligible = 0
    for record in iter_jsonl(scores_file):
        if not bool(record.get("final_keep")):
            continue
        eligible += 1
        if len(reservoir) < args.audit_size:
            reservoir.append(record)
        else:
            replacement = rng.randrange(eligible)
            if replacement < args.audit_size:
                reservoir[replacement] = record
    if not reservoir:
        raise RuntimeError("no kept rows available for fresh cache audit")
    device = torch.device(args.device)
    # WavLM must be materialised before SpeechBrain extends sys.path with its
    # optional dependency bundle; that bundle can otherwise shadow the local
    # huggingface_hub version used by this repository's Transformers install.
    silence_transformers_progress()
    wavlm_model = FrozenWavLMSVEncoder(args.wavlm_model, local_files_only=True)
    cases_by_row: dict[int, dict[str, Any]] = {
        int(record["row_index"]): {
            "sample_id": record["sample_id"],
            "row_index": record["row_index"],
            "ecapa": {},
            "wavlm_sv": {},
            "errors": [],
        }
        for record in reservoir
    }
    cache_root = output_root / "cache" / "wavlm_sv"
    # First audit WavLM, then import/load SpeechBrain for ECAPA.
    for record in reservoir:
        roles = record["roles"]
        case = cases_by_row[int(record["row_index"])]
        try:
            wavlm_paths = {
                "source": cached_wavlm_path(cache_root, roles["source_audio"]),
                "target": cached_wavlm_path(cache_root, roles["target_audio"]),
                "ref": Path(record["wavlm_ref_path"]),
                "prompt": cached_wavlm_path(cache_root, roles["prompt_audio"]),
            }
            fresh_paths = [roles["source_audio"], roles["target_audio"], roles["ref_audio"], roles["prompt_audio"]]
            fresh_vectors, mask = wavlm_model(fresh_paths, device=device, dtype=torch.float32)
            if fresh_vectors is None or mask is None or not bool(mask.all()):
                raise RuntimeError("WavLM fresh audit encoder failed")
            for role, fresh in zip(("source", "target", "ref", "prompt"), fresh_vectors.detach().cpu().float(), strict=True):
                cached, _metadata = load_wavlm(wavlm_paths[role])
                cosine = float(torch.dot(cached, torch.nn.functional.normalize(fresh.flatten(), dim=0)))
                case["wavlm_sv"][role] = cosine
                if cosine < args.min_fresh_wavlm_cosine:
                    raise RuntimeError(f"WavLM fresh/cache mismatch role={role} cosine={cosine}")
        except Exception as exc:
            case["errors"].append({"stage": "wavlm_sv", "error_type": type(exc).__name__, "error": str(exc)})

    ecapa_model = load_ecapa_encoder(path(args.ecapa_model), device)
    for record in reservoir:
        roles = record["roles"]
        case = cases_by_row[int(record["row_index"])]
        try:
            for role, audio_key in (("source", "source_audio"), ("target", "target_audio"), ("ref", "ref_audio"), ("prompt", "prompt_audio")):
                cache = Path(record["ecapa_paths"][role])
                cached = load_ecapa(cache)
                fresh = encode_ecapa(ecapa_model, str(roles[audio_key]), device)
                cosine = float(torch.dot(cached, fresh))
                case["ecapa"][role] = cosine
                if cosine < args.min_fresh_ecapa_cosine:
                    raise RuntimeError(f"ECAPA fresh/cache mismatch role={role} cosine={cosine}")
        except Exception as exc:
            case["errors"].append({"stage": "ecapa", "error_type": type(exc).__name__, "error": str(exc)})
    cases = list(cases_by_row.values())
    failures = []
    for case in cases:
        case["pass"] = not case["errors"]
        if not case["pass"]:
            failures.append(case)
    audit = {
        "status": "pass" if not failures else "fail",
        "created_at": utc_now(),
        "sample_size": len(cases),
        "eligible_kept_rows": eligible,
        "seed": args.audit_seed,
        "minimum_cosine": {
            "ecapa": args.min_fresh_ecapa_cosine,
            "wavlm_sv": args.min_fresh_wavlm_cosine,
        },
        "wavlm_sv_tolerance_note": (
            "FrozenWavLMSVEncoder batches variable-duration audio with padding; "
            "audit uses independent fresh batches, so its identity threshold is 0.95 rather than ECAPA's 0.999."
        ),
        "cases": cases,
        "failures": failures,
    }
    audit_path = output_root / "fresh_cache_audit.json"
    atomic_json(audit_path, audit)
    if failures:
        raise RuntimeError(f"fresh cache audit failed for {len(failures)}/{len(cases)} rows; see {audit_path}")
    summary_path = output_root / "four_edge_filter_summary.json"
    summary = load_json(summary_path)
    summary["fresh_cache_audit"] = {"status": "pass", "path": str(audit_path), "sha256": sha256_file(audit_path)}
    summary["status"] = "accepted" if args.max_rows == 0 else "smoke_accepted"
    atomic_json(summary_path, summary)
    completed = {
        "status": "complete",
        "run_kind": "smoke" if args.max_rows else "full",
        "created_at": utc_now(),
        "input_identity": {"path": str(output_root / "INPUT_IDENTITY.json"), "sha256": sha256_file(output_root / "INPUT_IDENTITY.json")},
        "code_snapshot": load_input_identity(output_root)["code_snapshot"],
        "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        "fresh_cache_audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
        "completion_contract": {
            "count_identity": summary["count_identity_pass"],
            "all_kept_dual_encoder_balanced": True,
            "cross_episode_u2_changed": False,
            "legacy_filtered_link_used": False,
        },
    }
    atomic_json(completed_path, completed)
    print(json.dumps(completed, ensure_ascii=False), flush=True)


def main() -> int:
    args = parse_args()
    output_root = prepare_output_root(args)
    stages = {
        "prepare": prepare,
        "prompt-ecapa": prompt_ecapa,
        "ecapa-score": ecapa_score,
        "wavlm-plan": wavlm_plan,
        "wavlm-cache": wavlm_cache,
        "wavlm-score": wavlm_score,
        "finalize": finalize,
        "fresh-audit": fresh_audit,
    }
    stages[args.stage](args, output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

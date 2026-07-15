#!/usr/bin/env python
"""Extract decoder-native dequantized codec latents for DDLFM training.

The extractor is deliberately restartable and shard-safe.  Each worker reads
only its byte range from every input JSONL, writes one atomic ``.npy`` per
utterance, and publishes its shard completion marker last.  ``finalize`` then
validates all shard records and writes the dataset-level manifest/statistics.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moss_codecvc.config import deep_get, load_config
from moss_codecvc.moss_codec import MossCodec


DEFAULT_CONFIG = ROOT / "configs/remote_full.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "prepared/zq_targets_v1"
CONTRACT_FILENAME = "CONTRACT.json"
ID_KEYS = ("utt_id", "sample_id", "pair_id", "id", "case_id")
SAFE_SPLIT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class InputSpec:
    split: str
    manifest: Path
    size_bytes: int
    mtime_ns: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "manifest": str(self.manifest),
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
        }


@dataclass
class PendingItem:
    split: str
    manifest: Path
    byte_offset: int
    utterance_id: str
    record_id: str
    row_sha256: str
    output_path: Path
    codes: torch.Tensor
    target_audio: str | None
    codes_source: str


class RunningStats:
    """Mergeable scalar statistics over stored latent values."""

    def __init__(self) -> None:
        self.value_count = 0
        self.sum = 0.0
        self.sum_squares = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update_array(self, value: np.ndarray) -> None:
        array = np.asarray(value)
        if array.size == 0:
            raise ValueError("cannot collect statistics from an empty array")
        if not np.isfinite(array).all():
            raise ValueError("latent contains non-finite values")
        work = array.astype(np.float64, copy=False)
        self.value_count += int(work.size)
        self.sum += float(work.sum(dtype=np.float64))
        self.sum_squares += float(np.square(work).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(work.min()))
        self.maximum = max(self.maximum, float(work.max()))

    def merge_dict(self, payload: dict[str, Any]) -> None:
        count = int(payload.get("value_count", 0))
        if count <= 0:
            return
        self.value_count += count
        self.sum += float(payload["sum"])
        self.sum_squares += float(payload["sum_squares"])
        self.minimum = min(self.minimum, float(payload["min"]))
        self.maximum = max(self.maximum, float(payload["max"]))

    def as_dict(self) -> dict[str, Any]:
        if self.value_count <= 0:
            return {
                "value_count": 0,
                "sum": 0.0,
                "sum_squares": 0.0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        mean = self.sum / self.value_count
        variance = max(0.0, self.sum_squares / self.value_count - mean * mean)
        return {
            "value_count": self.value_count,
            "sum": self.sum,
            "sum_squares": self.sum_squares,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
        }


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    identity: dict[str, Any] = {
        "name": path.name,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    # Hash control-plane files, but avoid hashing multi-GB weight shards once
    # per extraction worker.  Weight shard names/sizes/mtimes plus the hashed
    # safetensors index still make the checkpoint identity restart-safe.
    if stat.st_size <= 4 * 1024 * 1024 or path.name in {
        "config.json",
        "configuration_moss_audio_tokenizer.py",
        "modeling_moss_audio_tokenizer.py",
        "model.safetensors.index.json",
    }:
        identity["sha256"] = sha256_file(path)
    return identity


def codec_provenance(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    codec_path_value = deep_get(config, "moss.codec_path")
    if not codec_path_value:
        raise ValueError(f"moss.codec_path is missing from {config_path}")
    codec_path = Path(str(codec_path_value)).expanduser().resolve()
    if not codec_path.is_dir():
        raise FileNotFoundError(f"codec checkpoint directory does not exist: {codec_path}")
    checkpoint_config_path = codec_path / "config.json"
    if not checkpoint_config_path.is_file():
        raise FileNotFoundError(f"codec checkpoint config is missing: {checkpoint_config_path}")
    checkpoint_config = json.loads(checkpoint_config_path.read_text(encoding="utf-8"))
    if not isinstance(checkpoint_config, dict):
        raise ValueError(f"codec checkpoint config must be an object: {checkpoint_config_path}")
    quantizer_kwargs = checkpoint_config.get("quantizer_kwargs")
    if not isinstance(quantizer_kwargs, dict):
        raise ValueError(f"codec checkpoint config has no quantizer_kwargs: {checkpoint_config_path}")

    sampling_rate = int(checkpoint_config.get("sampling_rate", 0))
    downsample_rate = int(checkpoint_config.get("downsample_rate", 0))
    codebook_size = int(quantizer_kwargs.get("codebook_size", 0))
    max_quantizers = int(quantizer_kwargs.get("num_quantizers", 0))
    latent_dim = int(quantizer_kwargs.get("output_dim", 0))
    if min(sampling_rate, downsample_rate, codebook_size, max_quantizers, latent_dim) <= 0:
        raise ValueError(f"invalid codec contract in {checkpoint_config_path}")

    files = [
        _checkpoint_file_identity(path)
        for path in sorted(codec_path.iterdir(), key=lambda value: value.name)
        if path.is_file()
    ]
    if not files:
        raise ValueError(f"codec checkpoint directory has no files: {codec_path}")
    identity = {
        "codec_path": str(codec_path),
        "sampling_rate": sampling_rate,
        "downsample_rate": downsample_rate,
        "frame_rate_hz": sampling_rate / downsample_rate,
        "codebook_size": codebook_size,
        "max_quantizers": max_quantizers,
        "latent_dim": latent_dim,
        "files": files,
    }
    return {**identity, "fingerprint": canonical_json_sha256(identity)}


def parse_input_spec(value: str) -> InputSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--input must be SPLIT=MANIFEST")
    split, manifest_text = value.split("=", 1)
    split = split.strip()
    manifest_text = manifest_text.strip()
    if not SAFE_SPLIT_RE.fullmatch(split) or split in {".", ".."}:
        raise argparse.ArgumentTypeError(f"invalid split name: {split!r}")
    if not manifest_text:
        raise argparse.ArgumentTypeError("manifest path cannot be empty")
    manifest = Path(manifest_text).expanduser().resolve()
    if not manifest.is_file():
        raise argparse.ArgumentTypeError(f"manifest does not exist: {manifest}")
    stat = manifest.stat()
    return InputSpec(split=split, manifest=manifest, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def byte_range(size_bytes: int, shard_id: int, num_shards: int) -> tuple[int, int]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")
    if size_bytes < 0:
        raise ValueError("size_bytes cannot be negative")
    return size_bytes * shard_id // num_shards, size_bytes * (shard_id + 1) // num_shards


def iter_jsonl_byte_range(
    path: str | Path,
    *,
    shard_id: int,
    num_shards: int,
) -> Iterator[tuple[int, bytes]]:
    """Yield complete JSONL lines whose first byte belongs to this shard.

    This seeks directly to the worker's range.  A shard beginning in the middle
    of a (potentially very large) inline-code row discards only that partial row;
    it never scans the preceding portion of the file.
    """

    manifest = Path(path)
    size_bytes = manifest.stat().st_size
    start, end = byte_range(size_bytes, shard_id, num_shards)
    with manifest.open("rb") as handle:
        _seek_to_first_owned_line(handle, start)
        while True:
            offset = handle.tell()
            if offset >= end or offset >= size_bytes:
                break
            line = handle.readline()
            if not line:
                break
            if line.strip():
                yield offset, line


def _seek_to_first_owned_line(handle: BinaryIO, start: int) -> None:
    if start <= 0:
        handle.seek(0)
        return
    handle.seek(start - 1)
    preceding = handle.read(1)
    if preceding == b"\n":
        handle.seek(start)
    else:
        handle.readline()


def row_sha256(raw_line: bytes) -> str:
    return hashlib.sha256(raw_line.rstrip(b"\r\n")).hexdigest()


def _first_nonempty(mapping: dict[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def resolve_utterance_id(row: dict[str, Any], raw_digest: str) -> str:
    utterance_id = _first_nonempty(row, ID_KEYS)
    if utterance_id is None:
        meta = row.get("moss_codecvc_meta")
        if isinstance(meta, dict):
            utterance_id = _first_nonempty(meta, ID_KEYS)
    return utterance_id or f"row-{raw_digest[:24]}"


def safe_filename(split: str, utterance_id: str, raw_digest: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", utterance_id).strip("._-") or "utterance"
    stem = stem[:80]
    collision_digest = hashlib.sha256(
        json.dumps([split, utterance_id, raw_digest], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return f"{stem}.{collision_digest}.npy"


def logical_record_id(split: str, utterance_id: str) -> str:
    return json.dumps([split, utterance_id], ensure_ascii=False, separators=(",", ":"))


def validate_codes(
    value: Any,
    *,
    expected_n_vq: int,
    codebook_size: int,
    source: str,
) -> torch.Tensor:
    raw_codes = torch.as_tensor(value)
    if raw_codes.dtype == torch.bool:
        raise ValueError(f"{source} codes must be integer token IDs, not booleans")
    if raw_codes.is_floating_point():
        if not bool(torch.isfinite(raw_codes).all()) or not bool(torch.equal(raw_codes, raw_codes.round())):
            raise ValueError(f"{source} codes must contain finite integer token IDs")
    codes = raw_codes.to(dtype=torch.long)
    if codes.ndim != 2:
        raise ValueError(f"{source} codes must be (T, NQ), got {tuple(codes.shape)}")
    if codes.shape[0] <= 0 or codes.shape[1] <= 0:
        raise ValueError(f"{source} codes cannot be empty, got {tuple(codes.shape)}")
    if int(codes.shape[1]) != int(expected_n_vq):
        raise ValueError(
            f"{source} codes must use exactly {expected_n_vq} quantizers, got {codes.shape[1]}"
        )
    if bool((codes < 0).any()):
        raise ValueError(f"{source} codes contains negative token IDs")
    if bool((codes >= int(codebook_size)).any()):
        maximum = int(codes.max().item())
        raise ValueError(
            f"{source} codes contains token ID {maximum} outside codebook size {codebook_size}"
        )
    return codes.contiguous().cpu()


def manifest_codes(
    row: dict[str, Any],
    *,
    expected_n_vq: int,
    codebook_size: int,
) -> torch.Tensor:
    if "audio_codes" not in row or row["audio_codes"] is None:
        raise KeyError("manifest row has no audio_codes")
    return validate_codes(
        row["audio_codes"],
        expected_n_vq=expected_n_vq,
        codebook_size=codebook_size,
        source="manifest audio_codes",
    )


def target_audio_from_row(row: dict[str, Any], *, required: bool = True) -> str | None:
    meta = row.get("moss_codecvc_meta")
    value = meta.get("target_audio") if isinstance(meta, dict) else None
    if value is None or not str(value).strip():
        value = row.get("target_audio")
    if value is None or not str(value).strip():
        if required:
            raise KeyError("encode/verify requires target_audio (moss_codecvc_meta or top-level)")
        return None
    return str(Path(str(value)).expanduser())


def target_codec_frames_from_row(row: dict[str, Any]) -> int:
    meta = row.get("moss_codecvc_meta")
    value = meta.get("target_codec_frames") if isinstance(meta, dict) else None
    if value is None:
        value = row.get("target_codec_frames")
    if value is None:
        raise KeyError("manifest row has no target_codec_frames")
    try:
        frames = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"target_codec_frames must be an integer, got {value!r}") from exc
    if frames <= 0:
        raise ValueError(f"target_codec_frames must be positive, got {frames}")
    return frames


def acquire_codes(
    row: dict[str, Any],
    *,
    codes_source: str,
    codec: Any | None,
    n_vq: int | None,
    codebook_size: int,
) -> tuple[torch.Tensor, str | None]:
    if n_vq is None:
        raise ValueError("n_vq must be resolved before acquiring codes")
    if codes_source == "manifest":
        codes = manifest_codes(row, expected_n_vq=n_vq, codebook_size=codebook_size)
        target_audio = target_audio_from_row(row, required=False)
    else:
        if codec is None:
            raise RuntimeError(f"codes_source={codes_source} requires a codec")
        target_audio = target_audio_from_row(row, required=True)
        assert target_audio is not None
        if codes_source == "encode":
            encoded = codec.encode_path(target_audio, n_vq=n_vq)
            codes = validate_codes(
                encoded["codes"],
                expected_n_vq=n_vq,
                codebook_size=codebook_size,
                source="codec encode",
            )
        elif codes_source == "verify":
            expected = manifest_codes(row, expected_n_vq=n_vq, codebook_size=codebook_size)
            encoded = codec.encode_path(target_audio, n_vq=n_vq)
            actual = validate_codes(
                encoded["codes"],
                expected_n_vq=n_vq,
                codebook_size=codebook_size,
                source="codec encode",
            )
            if actual.shape != expected.shape:
                raise ValueError(
                    "encoded codes do not match manifest: "
                    f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
                )
            mismatches = int(torch.count_nonzero(actual != expected).item())
            if mismatches:
                raise ValueError(
                    f"encoded codes do not match manifest: {mismatches}/{expected.numel()} tokens differ"
                )
            codes = expected
        else:
            raise ValueError(f"unsupported codes_source: {codes_source!r}")

    expected_frames = target_codec_frames_from_row(row)
    if int(codes.shape[0]) != expected_frames:
        raise ValueError(
            f"target_codec_frames mismatch: manifest={expected_frames}, codes={codes.shape[0]}"
        )
    return codes, target_audio


def resolve_quantizer(codec: Any) -> Any:
    candidate = codec
    seen: set[int] = set()
    while id(candidate) not in seen:
        seen.add(id(candidate))
        quantizer = getattr(candidate, "quantizer", None)
        if quantizer is not None and callable(getattr(quantizer, "decode_codes", None)):
            return quantizer
        if hasattr(candidate, "module"):
            candidate = candidate.module
            continue
        if hasattr(candidate, "model"):
            candidate = candidate.model
            continue
        break
    raise TypeError("MossCodec model has no quantizer.decode_codes(codes) method")


@torch.inference_mode()
def decode_codes_batch(
    quantizer: Any,
    codes_list: Sequence[torch.Tensor],
    *,
    device: torch.device,
    expected_dim: int,
) -> list[torch.Tensor]:
    if not codes_list:
        return []
    normalized = [torch.as_tensor(codes, dtype=torch.long) for codes in codes_list]
    nqs = {int(codes.shape[1]) for codes in normalized if codes.ndim == 2}
    if len(nqs) != 1 or any(codes.ndim != 2 for codes in normalized):
        raise ValueError("one decode batch must contain (T, NQ) codes with a common NQ")
    max_frames = max(int(codes.shape[0]) for codes in normalized)
    nq = next(iter(nqs))
    batch = torch.zeros(nq, len(normalized), max_frames, dtype=torch.long, device=device)
    for index, codes in enumerate(normalized):
        batch[:, index, : codes.shape[0]] = codes.transpose(0, 1).to(device)
    decoded = quantizer.decode_codes(batch)
    decoded = torch.as_tensor(decoded)
    if decoded.ndim != 3 or decoded.shape[0] != len(normalized):
        raise ValueError(
            "quantizer.decode_codes must return (B, D, T), "
            f"got {tuple(decoded.shape)} for B={len(normalized)}"
        )
    if int(decoded.shape[1]) != int(expected_dim):
        raise ValueError(f"decoder latent dim must be {expected_dim}, got {decoded.shape[1]}")
    if int(decoded.shape[2]) < max_frames:
        raise ValueError(f"decoded latent has only {decoded.shape[2]} frames, expected at least {max_frames}")
    if not bool(torch.isfinite(decoded).all()):
        raise ValueError("quantizer.decode_codes returned non-finite values")
    return [
        decoded[index, :, : codes.shape[0]].detach().to(dtype=torch.float32, device="cpu").contiguous()
        for index, codes in enumerate(normalized)
    ]


def atomic_save_npy(path: str | Path, array: np.ndarray) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(array), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output.stat().st_size


def validate_npy(
    path: str | Path,
    *,
    expected_dim: int,
    expected_dtype: str,
    expected_frames: int | None = None,
    load_values: bool = True,
) -> np.ndarray:
    target = Path(path)
    try:
        array = np.load(target, allow_pickle=False, mmap_mode="r")
    except Exception as exc:
        raise ValueError(f"invalid npy file {target}: {exc}") from exc
    expected_np_dtype = np.dtype(expected_dtype)
    if array.ndim != 2:
        raise ValueError(f"latent must be (D, T), got {array.shape} in {target}")
    if int(array.shape[0]) != int(expected_dim):
        raise ValueError(f"latent dim must be {expected_dim}, got {array.shape[0]} in {target}")
    if int(array.shape[1]) <= 0:
        raise ValueError(f"latent has no frames in {target}")
    if expected_frames is not None and int(array.shape[1]) != int(expected_frames):
        raise ValueError(
            f"latent frame count must be {expected_frames}, got {array.shape[1]} in {target}"
        )
    if array.dtype != expected_np_dtype:
        raise ValueError(f"latent dtype must be {expected_np_dtype.name}, got {array.dtype.name} in {target}")
    if load_values and not np.isfinite(array).all():
        raise ValueError(f"latent contains non-finite values in {target}")
    return array


def atomic_write_json(path: str | Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_text(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def make_extraction_contract(
    args: argparse.Namespace,
    *,
    inputs: Sequence[InputSpec],
    config: dict[str, Any],
    provenance: dict[str, Any],
    n_vq: int,
) -> dict[str, Any]:
    contract = {
        "schema_version": 1,
        "inputs": [spec.as_dict() for spec in inputs],
        "codec_provenance": provenance,
        "codes_source": args.codes_source,
        "n_vq": int(n_vq),
        "codebook_size": int(provenance["codebook_size"]),
        "codec_dtype": args.codec_dtype,
        "output_dtype": np.dtype(args.output_dtype).name,
        "expected_dim": int(args.expected_dim),
        "frame_rate_hz": float(provenance["frame_rate_hz"]),
        "num_shards": int(args.num_shards),
        "max_rows_per_input_per_shard": int(args.max_rows),
        "partial": bool(args.max_rows > 0),
        "config_path": str(Path(args.config).expanduser().resolve()),
        "config_default_n_vq": int(deep_get(config, "moss.default_n_vq", 32)),
    }
    return {**contract, "contract_sha256": canonical_json_sha256(contract)}


def ensure_output_contract(output_root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Atomically publish or validate one immutable output-root contract."""

    output_root.mkdir(parents=True, exist_ok=True)
    contract_path = output_root / CONTRACT_FILENAME
    lock_path = output_root / ".CONTRACT.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            if contract_path.exists():
                actual = json.loads(contract_path.read_text(encoding="utf-8"))
                if not isinstance(actual, dict):
                    raise ValueError(f"output contract is not a JSON object: {contract_path}")
                if actual != contract:
                    raise ValueError(
                        "output-root contract mismatch; use a new output root instead of reusing incompatible zq files: "
                        f"{contract_path}"
                    )
                return actual

            preexisting = [
                path
                for path in output_root.iterdir()
                if path.name not in {lock_path.name}
                and not path.name.startswith(f".{CONTRACT_FILENAME}.tmp-")
            ]
            if preexisting:
                raise RuntimeError(
                    "refusing to adopt a non-empty output root without CONTRACT.json; "
                    f"found {preexisting[:3]}"
                )
            atomic_write_json(contract_path, contract)
            return contract
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def acquire_shard_lock(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"another extractor already owns this shard: {path}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "started_at_unix": time.time()}) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def release_shard_lock(handle: Any) -> None:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def shard_prefix(output_root: Path, shard_id: int, num_shards: int) -> Path:
    return output_root / "_shards" / f"shard-{shard_id:05d}-of-{num_shards:05d}"


def make_record(
    item: PendingItem,
    array: np.ndarray,
    *,
    reused: bool,
    output_size: int,
    output_sha256: str,
    frame_rate_hz: float,
) -> dict[str, Any]:
    work = np.asarray(array).astype(np.float64, copy=False)
    return {
        "record_id": item.record_id,
        "utterance_id": item.utterance_id,
        "split": item.split,
        "manifest": str(item.manifest),
        "manifest_byte_offset": item.byte_offset,
        "row_sha256": item.row_sha256,
        "target_audio": item.target_audio,
        "codes_source": item.codes_source,
        "num_quantizers": int(item.codes.shape[1]),
        "num_frames": int(item.codes.shape[0]),
        "latent_dim": int(array.shape[0]),
        "dtype": np.dtype(array.dtype).name,
        "frame_rate_hz": float(frame_rate_hz),
        "duration_sec": float(item.codes.shape[0] / frame_rate_hz),
        "output_path": str(item.output_path.resolve()),
        "output_size_bytes": int(output_size),
        "output_sha256": output_sha256,
        "reused": bool(reused),
        "mean": float(work.mean()),
        "std": float(work.std()),
        "min": float(work.min()),
        "max": float(work.max()),
    }


def _json_row(raw_line: bytes, manifest: Path, byte_offset: int) -> dict[str, Any]:
    try:
        value = json.loads(raw_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON row at {manifest}:{byte_offset}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON row must be an object at {manifest}:{byte_offset}")
    return value


def _codec_from_args(args: argparse.Namespace) -> MossCodec:
    config = load_config(args.config)
    codec_path = deep_get(config, "moss.codec_path")
    moss_root = deep_get(config, "moss.root")
    if not codec_path:
        raise ValueError(f"moss.codec_path is missing from {args.config}")
    return MossCodec(
        codec_path,
        moss_root=moss_root,
        device=args.device,
        dtype=args.codec_dtype,
    )


def _manifest_unchanged(spec: InputSpec) -> None:
    stat = spec.manifest.stat()
    if stat.st_size != spec.size_bytes or stat.st_mtime_ns != spec.mtime_ns:
        raise RuntimeError(f"input manifest changed while extracting: {spec.manifest}")


def run_extract(args: argparse.Namespace) -> int:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.expected_dim <= 0:
        raise ValueError("--expected-dim must be positive")
    if args.max_rows < 0:
        raise ValueError("--max-rows cannot be negative")
    byte_range(0, args.shard_id, args.num_shards)
    inputs: list[InputSpec] = list(args.input)
    split_names = [spec.split for spec in inputs]
    if len(split_names) != len(set(split_names)):
        raise ValueError("each --input split name must be unique")

    output_root = Path(args.output_root).expanduser().resolve()
    config = load_config(args.config)
    provenance = codec_provenance(config, args.config)
    config_n_vq = int(deep_get(config, "moss.default_n_vq", 32))
    n_vq = int(args.n_vq) if args.n_vq is not None else config_n_vq
    if n_vq <= 0:
        raise ValueError("--n-vq (or moss.default_n_vq) must be positive")
    if n_vq > int(provenance["max_quantizers"]):
        raise ValueError(
            f"requested {n_vq} quantizers, checkpoint supports only {provenance['max_quantizers']}"
        )
    if int(args.expected_dim) != int(provenance["latent_dim"]):
        raise ValueError(
            f"--expected-dim={args.expected_dim} does not match codec latent dim {provenance['latent_dim']}"
        )
    contract = make_extraction_contract(
        args,
        inputs=inputs,
        config=config,
        provenance=provenance,
        n_vq=n_vq,
    )
    ensure_output_contract(output_root, contract)
    prefix = shard_prefix(output_root, args.shard_id, args.num_shards)
    lock_handle = acquire_shard_lock(prefix.with_suffix(".lock"))
    try:
        return _run_extract_locked(
            args,
            inputs=inputs,
            output_root=output_root,
            prefix=prefix,
            contract=contract,
            n_vq=n_vq,
        )
    finally:
        release_shard_lock(lock_handle)


def _run_extract_locked(
    args: argparse.Namespace,
    *,
    inputs: Sequence[InputSpec],
    output_root: Path,
    prefix: Path,
    contract: dict[str, Any],
    n_vq: int,
) -> int:
    (output_root / "COMPLETED.json").unlink(missing_ok=True)
    records_path = prefix.with_suffix(".records.jsonl")
    errors_path = prefix.with_suffix(".errors.jsonl")
    stats_path = prefix.with_suffix(".stats.json")
    completed_path = prefix.with_suffix(".COMPLETED.json")
    completed_path.unlink(missing_ok=True)

    output_dtype = np.dtype(args.output_dtype)
    codebook_size = int(contract["codebook_size"])
    frame_rate_hz = float(contract["frame_rate_hz"])
    codec: Any | None = None
    quantizer: Any | None = None
    device: torch.device | None = None

    def ensure_codec() -> tuple[Any, Any, torch.device]:
        nonlocal codec, quantizer, device
        if codec is None:
            codec = _codec_from_args(args)
            quantizer = resolve_quantizer(codec)
            device = torch.device(getattr(codec, "device", args.device))
            actual_n_vq = getattr(quantizer, "num_quantizers", None)
            actual_codebook_size = getattr(quantizer, "codebook_size", None)
            if actual_n_vq is not None and int(actual_n_vq) < n_vq:
                raise ValueError(f"loaded quantizer exposes only {actual_n_vq} quantizers, need {n_vq}")
            if actual_codebook_size is not None and int(actual_codebook_size) != codebook_size:
                raise ValueError(
                    f"loaded quantizer codebook size {actual_codebook_size} != contract {codebook_size}"
                )
        assert quantizer is not None and device is not None
        return codec, quantizer, device

    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    pending: list[PendingItem] = []
    outcome_keys: set[tuple[str, int]] = set()
    aggregate = RunningStats()
    processed_rows = 0
    reused_rows = 0
    written_rows = 0
    total_frames = 0
    total_size_bytes = 0
    input_progress: list[dict[str, Any]] = []
    started = time.time()

    def item_key(item: PendingItem) -> tuple[str, int]:
        return str(item.manifest), int(item.byte_offset)

    def record_success(item: PendingItem, array: np.ndarray, *, reused: bool) -> None:
        nonlocal reused_rows, written_rows, total_frames, total_size_bytes
        key = item_key(item)
        if key in outcome_keys:
            raise RuntimeError(f"duplicate row outcome for {key}")
        size_bytes = item.output_path.stat().st_size
        output_sha256 = sha256_file(item.output_path)
        record = make_record(
            item,
            array,
            reused=reused,
            output_size=size_bytes,
            output_sha256=output_sha256,
            frame_rate_hz=frame_rate_hz,
        )
        aggregate.update_array(array)
        records.append(record)
        outcome_keys.add(key)
        reused_rows += int(reused)
        written_rows += int(not reused)
        total_frames += int(array.shape[1])
        total_size_bytes += int(size_bytes)

    def record_error(item: PendingItem | None, exc: Exception, **context: Any) -> None:
        manifest = str(item.manifest) if item is not None else str(context.get("manifest", ""))
        byte_offset = int(item.byte_offset) if item is not None else int(context.get("manifest_byte_offset", -1))
        key = (manifest, byte_offset)
        if not manifest or byte_offset < 0:
            raise RuntimeError("row error is missing manifest/byte offset") from exc
        if key in outcome_keys:
            return
        error = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            **context,
        }
        if item is not None:
            error.update(
                {
                    "record_id": item.record_id,
                    "utterance_id": item.utterance_id,
                    "split": item.split,
                    "manifest": str(item.manifest),
                    "manifest_byte_offset": item.byte_offset,
                    "output_path": str(item.output_path),
                }
            )
        errors.append(error)
        outcome_keys.add(key)

    def save_decoded(item: PendingItem, latent: torch.Tensor) -> Exception | None:
        try:
            stored = latent.numpy().astype(output_dtype, copy=False)
            atomic_save_npy(item.output_path, stored)
            array = validate_npy(
                item.output_path,
                expected_dim=args.expected_dim,
                expected_dtype=output_dtype.name,
                expected_frames=int(item.codes.shape[0]),
            )
            record_success(item, array, reused=False)
            return None
        except Exception as exc:
            record_error(item, exc, stage="decode_or_save")
            return exc

    def flush_pending() -> None:
        nonlocal pending
        if not pending:
            return
        batch = pending
        pending = []
        decoder: Any | None = None
        decoder_device: torch.device | None = None
        try:
            _, decoder, decoder_device = ensure_codec()
            latents = decode_codes_batch(
                decoder,
                [item.codes for item in batch],
                device=decoder_device,
                expected_dim=args.expected_dim,
            )
        except Exception as exc:
            if args.strict:
                for item in batch:
                    record_error(item, exc, stage="decode_batch")
                raise
            if decoder is None or decoder_device is None:
                for item in batch:
                    record_error(item, exc, stage="decode_batch")
                return
            for item in batch:
                try:
                    one = decode_codes_batch(
                        decoder,
                        [item.codes],
                        device=decoder_device,
                        expected_dim=args.expected_dim,
                    )[0]
                    save_decoded(item, one)
                except Exception as one_exc:
                    record_error(item, one_exc, stage="decode_or_save")
            return
        first_error: Exception | None = None
        for item, latent in zip(batch, latents, strict=True):
            error = save_decoded(item, latent)
            if first_error is None and error is not None:
                first_error = error
        if first_error is not None and args.strict:
            raise first_error

    fatal: Exception | None = None
    try:
        for spec in inputs:
            range_start, range_end = byte_range(spec.size_bytes, args.shard_id, args.num_shards)
            progress = {
                "split": spec.split,
                "manifest": str(spec.manifest),
                "range_start": range_start,
                "range_end": range_end,
                "processed_rows": 0,
                "first_byte_offset": None,
                "last_byte_offset": None,
                "cap_reached": False,
                "range_exhausted": False,
            }
            input_progress.append(progress)
            for byte_offset, raw_line in iter_jsonl_byte_range(
                spec.manifest,
                shard_id=args.shard_id,
                num_shards=args.num_shards,
            ):
                if args.max_rows > 0 and int(progress["processed_rows"]) >= args.max_rows:
                    progress["cap_reached"] = True
                    break
                processed_rows += 1
                progress["processed_rows"] = int(progress["processed_rows"]) + 1
                if progress["first_byte_offset"] is None:
                    progress["first_byte_offset"] = byte_offset
                progress["last_byte_offset"] = byte_offset
                item: PendingItem | None = None
                try:
                    digest = row_sha256(raw_line)
                    row = _json_row(raw_line, spec.manifest, byte_offset)
                    utterance_id = resolve_utterance_id(row, digest)
                    record_id = logical_record_id(spec.split, utterance_id)
                    output_path = output_root / spec.split / safe_filename(spec.split, utterance_id, digest)
                    current_codec = None
                    if args.codes_source in {"encode", "verify"}:
                        current_codec, _, _ = ensure_codec()
                    codes, target_audio = acquire_codes(
                        row,
                        codes_source=args.codes_source,
                        codec=current_codec,
                        n_vq=n_vq,
                        codebook_size=codebook_size,
                    )
                    item = PendingItem(
                        split=spec.split,
                        manifest=spec.manifest,
                        byte_offset=byte_offset,
                        utterance_id=utterance_id,
                        record_id=record_id,
                        row_sha256=digest,
                        output_path=output_path,
                        codes=codes,
                        target_audio=target_audio,
                        codes_source=args.codes_source,
                    )
                    if output_path.exists() and not args.overwrite:
                        try:
                            array = validate_npy(
                                output_path,
                                expected_dim=args.expected_dim,
                                expected_dtype=output_dtype.name,
                                expected_frames=int(codes.shape[0]),
                            )
                        except ValueError:
                            pass
                        else:
                            record_success(item, array, reused=True)
                            continue
                    if pending and int(pending[0].codes.shape[1]) != int(codes.shape[1]):
                        flush_pending()
                    pending.append(item)
                    if len(pending) >= args.batch_size:
                        flush_pending()
                except Exception as exc:
                    if (str(spec.manifest), int(byte_offset)) not in outcome_keys:
                        record_error(
                            item,
                            exc,
                            stage="prepare",
                            split=spec.split,
                            manifest=str(spec.manifest),
                            manifest_byte_offset=byte_offset,
                        )
                    if args.strict:
                        raise
                if args.log_every > 0 and processed_rows % args.log_every == 0:
                    print(
                        f"[zq-extract] shard={args.shard_id}/{args.num_shards} "
                        f"rows={processed_rows} written={written_rows} reused={reused_rows} "
                        f"errors={len(errors)} elapsed={time.time() - started:.1f}s",
                        flush=True,
                    )
            else:
                progress["range_exhausted"] = True
            _manifest_unchanged(spec)
        flush_pending()
    except Exception as exc:
        fatal = exc

    if pending:
        try:
            flush_pending()
        except Exception as exc:
            if fatal is None:
                fatal = exc

    if len(records) + len(errors) != processed_rows or len(outcome_keys) != processed_rows:
        invariant_error = RuntimeError(
            "row outcome invariant failed: "
            f"processed={processed_rows}, records={len(records)}, errors={len(errors)}, "
            f"unique_outcomes={len(outcome_keys)}"
        )
        if fatal is None:
            fatal = invariant_error

    atomic_write_jsonl(records_path, records)
    atomic_write_jsonl(errors_path, errors)
    scalar_stats = aggregate.as_dict()
    summary = {
        "schema_version": 1,
        "status": "failed" if fatal is not None else "completed",
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "inputs": [spec.as_dict() for spec in inputs],
        "input_progress": input_progress,
        "contract_path": str(output_root / CONTRACT_FILENAME),
        "contract_sha256": contract["contract_sha256"],
        "codes_source": args.codes_source,
        "n_vq": n_vq,
        "codebook_size": codebook_size,
        "output_dtype": output_dtype.name,
        "expected_dim": args.expected_dim,
        "frame_rate_hz": frame_rate_hz,
        "partial": bool(contract["partial"]),
        "max_rows_per_input_per_shard": int(args.max_rows),
        "processed_rows": processed_rows,
        "records": len(records),
        "written": written_rows,
        "reused": reused_rows,
        "errors": len(errors),
        "total_frames": total_frames,
        "total_duration_sec": total_frames / frame_rate_hz,
        "total_size_bytes": total_size_bytes,
        "latent_stats": scalar_stats,
        "mean": scalar_stats["mean"],
        "std": scalar_stats["std"],
        "min": scalar_stats["min"],
        "max": scalar_stats["max"],
        "records_path": str(records_path),
        "errors_path": str(errors_path),
        "elapsed_sec": time.time() - started,
    }
    if fatal is not None:
        summary["fatal_error_type"] = type(fatal).__name__
        summary["fatal_error"] = str(fatal)
    atomic_write_json(stats_path, summary)
    if fatal is not None:
        raise fatal
    completion = {
        **summary,
        "status": "completed",
        "stats_path": str(stats_path),
        "completed_at_unix": time.time(),
    }
    atomic_write_json(completed_path, completion)
    print(json.dumps(completion, ensure_ascii=False, sort_keys=True))
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _iter_jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"record is not an object at {path}:{line_no}")
            yield row


def run_finalize(args: argparse.Namespace) -> int:
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.expected_total_utterances is not None and args.expected_total_utterances < 0:
        raise ValueError("--expected-total-utterances cannot be negative")
    if args.expected_total_frames is not None and args.expected_total_frames < 0:
        raise ValueError("--expected-total-frames cannot be negative")
    output_root = Path(args.output_root).expanduser().resolve()
    final_manifest = output_root / "manifest.jsonl"
    final_errors = output_root / "errors.jsonl"
    final_stats = output_root / "stats.json"
    final_completed = output_root / "COMPLETED.json"
    final_completed.unlink(missing_ok=True)

    contract_path = output_root / CONTRACT_FILENAME
    if not contract_path.is_file():
        raise FileNotFoundError(f"missing output contract: {contract_path}")
    contract = _read_json(contract_path)
    stored_contract_hash = str(contract.get("contract_sha256", ""))
    contract_body = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if not SHA256_RE.fullmatch(stored_contract_hash) or canonical_json_sha256(contract_body) != stored_contract_hash:
        raise ValueError(f"invalid or modified output contract: {contract_path}")
    if int(contract.get("num_shards", -1)) != args.num_shards:
        raise ValueError(
            f"--num-shards={args.num_shards} does not match output contract {contract.get('num_shards')}"
        )
    if bool(contract.get("partial")) and not args.allow_partial:
        raise ValueError("output contract is partial because --max-rows was used; pass --allow-partial explicitly")

    contract_inputs = contract.get("inputs")
    if not isinstance(contract_inputs, list) or not contract_inputs:
        raise ValueError("output contract has no inputs")
    input_by_manifest: dict[str, dict[str, Any]] = {}
    for value in contract_inputs:
        if not isinstance(value, dict):
            raise ValueError("invalid input entry in output contract")
        manifest = str(value.get("manifest", ""))
        split = str(value.get("split", ""))
        if not manifest or not split or manifest in input_by_manifest:
            raise ValueError("invalid or duplicate manifest entry in output contract")
        stat = Path(manifest).stat()
        if stat.st_size != int(value.get("size_bytes", -1)) or stat.st_mtime_ns != int(value.get("mtime_ns", -1)):
            raise RuntimeError(f"input manifest changed after extraction: {manifest}")
        input_by_manifest[manifest] = value

    completions: list[dict[str, Any]] = []
    aggregate = RunningStats()
    seen_ids: dict[tuple[str, str], str] = {}
    seen_paths: set[str] = set()
    common_dim: int | None = None
    common_dtype: str | None = None
    common_n_vq: int | None = None
    total_utterances = 0
    total_errors = 0
    total_frames = 0
    total_size_bytes = 0

    unexpected = sorted((output_root / "_shards").glob("shard-*-of-*.COMPLETED.json"))
    expected_completion_paths = {
        shard_prefix(output_root, shard_id, args.num_shards).with_suffix(".COMPLETED.json")
        for shard_id in range(args.num_shards)
    }
    foreign = [path for path in unexpected if path not in expected_completion_paths]
    if foreign:
        raise ValueError(f"found completion markers for a different shard contract: {foreign[:3]}")

    manifest_tmp = final_manifest.with_name(f".{final_manifest.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    errors_tmp = final_errors.with_name(f".{final_errors.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with manifest_tmp.open("w", encoding="utf-8") as manifest_handle, errors_tmp.open(
            "w", encoding="utf-8"
        ) as errors_handle:
            for shard_id in range(args.num_shards):
                prefix = shard_prefix(output_root, shard_id, args.num_shards)
                completion_path = prefix.with_suffix(".COMPLETED.json")
                if not completion_path.is_file():
                    raise FileNotFoundError(f"missing shard completion: {completion_path}")
                completion = _read_json(completion_path)
                if completion.get("status") != "completed":
                    raise ValueError(f"shard {shard_id} is not completed")
                if (
                    int(completion.get("shard_id", -1)) != shard_id
                    or int(completion.get("num_shards", -1)) != args.num_shards
                ):
                    raise ValueError(f"shard identity mismatch in {completion_path}")
                if str(completion.get("contract_sha256", "")) != stored_contract_hash:
                    raise ValueError(f"shard contract hash mismatch in {completion_path}")
                if bool(completion.get("partial")) != bool(contract.get("partial")):
                    raise ValueError(f"shard partial flag mismatch in {completion_path}")
                expected_records_path = prefix.with_suffix(".records.jsonl").resolve()
                expected_errors_path = prefix.with_suffix(".errors.jsonl").resolve()
                records_path = Path(str(completion.get("records_path", ""))).resolve()
                errors_path = Path(str(completion.get("errors_path", ""))).resolve()
                if records_path != expected_records_path or errors_path != expected_errors_path:
                    raise ValueError(f"shard records/errors path mismatch for shard {shard_id}")
                if not records_path.is_file() or not errors_path.is_file():
                    raise FileNotFoundError(f"shard records/errors missing for shard {shard_id}")
                expected_record_count = int(completion.get("records", -1))
                expected_error_count = int(completion.get("errors", -1))
                processed_count = int(completion.get("processed_rows", -1))
                if expected_record_count < 0 or expected_error_count < 0:
                    raise ValueError(f"invalid shard counts for shard {shard_id}")
                if expected_record_count + expected_error_count != processed_count:
                    raise ValueError(f"row outcome count mismatch in shard completion {shard_id}")
                if expected_error_count and not args.allow_errors:
                    raise ValueError(f"shard {shard_id} has {expected_error_count} extraction errors")

                progress_rows = completion.get("input_progress")
                if not isinstance(progress_rows, list) or len(progress_rows) != len(contract_inputs):
                    raise ValueError(f"invalid input progress in shard completion {shard_id}")
                progress_by_manifest: dict[str, dict[str, Any]] = {}
                for progress in progress_rows:
                    if not isinstance(progress, dict):
                        raise ValueError(f"invalid input progress entry in shard completion {shard_id}")
                    manifest = str(progress.get("manifest", ""))
                    input_contract = input_by_manifest.get(manifest)
                    if input_contract is None or manifest in progress_by_manifest:
                        raise ValueError(f"unknown or duplicate input progress in shard completion {shard_id}")
                    range_start, range_end = byte_range(
                        int(input_contract["size_bytes"]), shard_id, args.num_shards
                    )
                    if (
                        str(progress.get("split", "")) != str(input_contract["split"])
                        or int(progress.get("range_start", -1)) != range_start
                        or int(progress.get("range_end", -1)) != range_end
                    ):
                        raise ValueError(f"input progress range mismatch in shard completion {shard_id}")
                    if not bool(contract.get("partial")) and not bool(progress.get("range_exhausted")):
                        raise ValueError(f"full extraction did not exhaust an input range in shard {shard_id}")
                    progress_by_manifest[manifest] = progress
                if sum(int(value.get("processed_rows", -1)) for value in progress_rows) != processed_count:
                    raise ValueError(f"input progress count mismatch in shard completion {shard_id}")
                outcome_count_by_manifest = {manifest: 0 for manifest in input_by_manifest}

                shard_record_count = 0
                for record in _iter_jsonl_records(records_path):
                    shard_record_count += 1
                    split = str(record.get("split", ""))
                    utterance_id = str(record.get("utterance_id", ""))
                    manifest = str(record.get("manifest", ""))
                    byte_offset = int(record.get("manifest_byte_offset", -1))
                    input_contract = input_by_manifest.get(manifest)
                    if not split or not utterance_id or input_contract is None:
                        raise ValueError(f"record has invalid split/utterance/manifest in {records_path}")
                    if split != str(input_contract.get("split", "")):
                        raise ValueError(f"record split does not match input contract in {records_path}")
                    range_start, range_end = byte_range(
                        int(input_contract["size_bytes"]), shard_id, args.num_shards
                    )
                    if byte_offset < range_start or byte_offset >= range_end:
                        raise ValueError(
                            f"record byte offset {byte_offset} is outside shard {shard_id} range "
                            f"[{range_start}, {range_end})"
                        )
                    outcome_count_by_manifest[manifest] += 1
                    identity = (split, utterance_id)
                    if identity in seen_ids:
                        raise ValueError(
                            f"duplicate utterance ID {identity!r}: {seen_ids[identity]} and {records_path}"
                        )
                    seen_ids[identity] = str(records_path)
                    output_path = str(record.get("output_path", ""))
                    output_file = Path(output_path).resolve()
                    try:
                        output_file.relative_to(output_root)
                    except ValueError as exc:
                        raise ValueError(f"output path escapes output root: {output_file}") from exc
                    canonical_output_path = str(output_file)
                    if canonical_output_path in seen_paths:
                        raise ValueError(f"duplicate output_path in shard records: {output_path}")
                    seen_paths.add(canonical_output_path)
                    dim = int(record.get("latent_dim", -1))
                    dtype = str(record.get("dtype", ""))
                    frames = int(record.get("num_frames", -1))
                    record_n_vq = int(record.get("num_quantizers", -1))
                    if common_dim is None:
                        common_dim = dim
                        common_dtype = dtype
                        common_n_vq = record_n_vq
                    elif dim != common_dim or dtype != common_dtype or record_n_vq != common_n_vq:
                        raise ValueError(
                            "non-uniform latent/code contract: "
                            f"expected ({common_dim}, {common_dtype}, nq={common_n_vq}), "
                            f"got ({dim}, {dtype}, nq={record_n_vq})"
                        )
                    if record_n_vq != int(contract["n_vq"]):
                        raise ValueError(f"record num_quantizers does not match contract: {output_path}")
                    if str(record.get("codes_source", "")) != str(contract["codes_source"]):
                        raise ValueError(f"record codes_source does not match contract: {output_path}")
                    if float(record.get("frame_rate_hz", -1.0)) != float(contract["frame_rate_hz"]):
                        raise ValueError(f"record frame rate does not match contract: {output_path}")
                    array = validate_npy(
                        output_file,
                        expected_dim=dim,
                        expected_dtype=dtype,
                        expected_frames=frames,
                        load_values=False,
                    )
                    actual_size = output_file.stat().st_size
                    if actual_size != int(record.get("output_size_bytes", -1)):
                        raise ValueError(f"output size mismatch for {output_path}")
                    expected_sha256 = str(record.get("output_sha256", ""))
                    if not SHA256_RE.fullmatch(expected_sha256):
                        raise ValueError(f"invalid output sha256 in record: {output_path}")
                    actual_sha256 = sha256_file(output_file)
                    if actual_sha256 != expected_sha256:
                        raise ValueError(f"output sha256 mismatch for {output_path}")
                    if tuple(array.shape) != (dim, frames):
                        raise ValueError(f"output shape mismatch for {output_path}")
                    total_utterances += 1
                    total_frames += frames
                    total_size_bytes += actual_size
                    manifest_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

                shard_error_count = 0
                for error in _iter_jsonl_records(errors_path):
                    shard_error_count += 1
                    manifest = str(error.get("manifest", ""))
                    byte_offset = int(error.get("manifest_byte_offset", -1))
                    input_contract = input_by_manifest.get(manifest)
                    if input_contract is None:
                        raise ValueError(f"error record has unknown manifest in {errors_path}")
                    range_start, range_end = byte_range(
                        int(input_contract["size_bytes"]), shard_id, args.num_shards
                    )
                    if byte_offset < range_start or byte_offset >= range_end:
                        raise ValueError(f"error byte offset is outside shard range in {errors_path}")
                    outcome_count_by_manifest[manifest] += 1
                    errors_handle.write(json.dumps(error, ensure_ascii=False, sort_keys=True) + "\n")
                if shard_record_count != expected_record_count:
                    raise ValueError(f"record count mismatch for shard {shard_id}")
                if shard_error_count != expected_error_count:
                    raise ValueError(f"error count mismatch for shard {shard_id}")
                for manifest, progress in progress_by_manifest.items():
                    if outcome_count_by_manifest[manifest] != int(progress.get("processed_rows", -1)):
                        raise ValueError(
                            f"per-input row outcome mismatch for {manifest} in shard {shard_id}"
                        )
                total_errors += shard_error_count
                completions.append(completion)
                aggregate.merge_dict(dict(completion.get("latent_stats") or {}))

            manifest_handle.flush()
            os.fsync(manifest_handle.fileno())
            errors_handle.flush()
            os.fsync(errors_handle.fileno())

        if total_utterances <= 0:
            raise ValueError("cannot finalize an empty zq dataset")
        if common_dim != int(contract.get("expected_dim", -1)):
            raise ValueError("record latent dim does not match output contract")
        if common_dtype != str(contract.get("output_dtype", "")):
            raise ValueError("record dtype does not match output contract")
        if common_n_vq != int(contract.get("n_vq", -1)):
            raise ValueError("record num_quantizers does not match output contract")
        completion_frames = sum(int(value.get("total_frames", 0)) for value in completions)
        completion_size = sum(int(value.get("total_size_bytes", 0)) for value in completions)
        completion_records = sum(int(value.get("records", 0)) for value in completions)
        completion_errors = sum(int(value.get("errors", 0)) for value in completions)
        if (
            completion_frames != total_frames
            or completion_size != total_size_bytes
            or completion_records != total_utterances
            or completion_errors != total_errors
        ):
            raise ValueError("merged totals do not match shard completion totals")
        if args.expected_total_utterances is not None and total_utterances != args.expected_total_utterances:
            raise ValueError(
                f"total utterance gate failed: expected {args.expected_total_utterances}, got {total_utterances}"
            )
        if args.expected_total_frames is not None and total_frames != args.expected_total_frames:
            raise ValueError(f"total frame gate failed: expected {args.expected_total_frames}, got {total_frames}")

        scalar_stats = aggregate.as_dict()
        expected_value_count = total_frames * int(common_dim)
        if int(scalar_stats.get("value_count", 0)) != expected_value_count:
            raise ValueError(
                f"latent statistics count mismatch: expected {expected_value_count}, "
                f"got {scalar_stats.get('value_count')}"
            )
        os.replace(manifest_tmp, final_manifest)
        os.replace(errors_tmp, final_errors)
    finally:
        manifest_tmp.unlink(missing_ok=True)
        errors_tmp.unlink(missing_ok=True)

    frame_rate = float(contract["frame_rate_hz"])
    scalar_stats = aggregate.as_dict()
    summary = {
        "schema_version": 1,
        "status": "completed",
        "num_shards": args.num_shards,
        "contract_path": str(contract_path),
        "contract_sha256": stored_contract_hash,
        "inputs": contract_inputs,
        "codes_source": contract["codes_source"],
        "n_vq": contract["n_vq"],
        "codebook_size": contract["codebook_size"],
        "partial": bool(contract["partial"]),
        "latent_dim": common_dim,
        "dtype": common_dtype,
        "frame_rate_hz": frame_rate,
        "total_utterances": total_utterances,
        "total_frames": total_frames,
        "total_duration_sec": total_frames / frame_rate,
        "total_size_bytes": total_size_bytes,
        "errors": total_errors,
        "expected_total_utterances": args.expected_total_utterances,
        "expected_total_frames": args.expected_total_frames,
        "latent_stats": scalar_stats,
        "mean": scalar_stats["mean"],
        "std": scalar_stats["std"],
        "min": scalar_stats["min"],
        "max": scalar_stats["max"],
        "manifest_path": str(final_manifest),
        "errors_path": str(final_errors),
        "shard_completion_paths": [
            str(shard_prefix(output_root, shard_id, args.num_shards).with_suffix(".COMPLETED.json"))
            for shard_id in range(args.num_shards)
        ],
    }
    atomic_write_json(final_stats, summary)
    atomic_write_json(
        final_completed,
        {
            **summary,
            "stats_path": str(final_stats),
            "completed_at_unix": time.time(),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract MOSS decoder-native zq targets for ver3.1 DDLFM.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="extract one byte-range shard")
    extract.add_argument(
        "--input",
        action="append",
        required=True,
        type=parse_input_spec,
        metavar="SPLIT=MANIFEST",
        help="repeat for each dataset split",
    )
    extract.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    extract.add_argument("--config", default=str(DEFAULT_CONFIG))
    extract.add_argument("--codes-source", choices=("manifest", "encode", "verify"), default="manifest")
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument("--codec-dtype", choices=("float16", "bfloat16", "float32"), default="float32")
    extract.add_argument("--output-dtype", "--dtype", dest="output_dtype", choices=("float16", "float32"), default="float32")
    extract.add_argument("--n-vq", type=int, default=None)
    extract.add_argument("--batch-size", type=int, default=8)
    extract.add_argument("--expected-dim", type=int, default=768)
    extract.add_argument("--num-shards", type=int, default=1)
    extract.add_argument("--shard-id", type=int, default=0)
    extract.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="debug cap applied independently to each input in this shard; marks output partial",
    )
    extract.add_argument("--log-every", type=int, default=100)
    extract.add_argument("--overwrite", action="store_true")
    extract.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    extract.set_defaults(func=run_extract)

    finalize = subparsers.add_parser("finalize", help="validate and merge completed shards")
    finalize.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    finalize.add_argument("--num-shards", type=int, required=True)
    finalize.add_argument("--allow-errors", action="store_true")
    finalize.add_argument("--allow-partial", action="store_true")
    finalize.add_argument("--expected-total-utterances", type=int, default=None)
    finalize.add_argument("--expected-total-frames", type=int, default=None)
    finalize.set_defaults(func=run_finalize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

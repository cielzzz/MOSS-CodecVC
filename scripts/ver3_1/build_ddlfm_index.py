#!/usr/bin/env python
"""Build a compact, speaker-safe training index for ver3.1 DDLFM.

The zq target manifest is the authoritative row order and target latent
contract.  Its records retain the source manifest path and byte offset, so we
can recover only the conditioning fields needed by the CFM loader without
duplicating the very large ``audio_codes`` payloads.

Conditioning policy:

* ``no_text``: use the verified WavLM Content-Adapter semantic ``.npy``.
* ``text``: use manifest ``content_token_ids`` only.  Never use target BNF,
  target WavLM features, or source-BNF from the text prosody carrier.
* speaker: use the existing 192-D timbre-reference embedding sidecar.  The
  current v1 sidecars are SpeechBrain ECAPA; the index records this fact.

The resulting JSONL is intentionally compact and is suitable for sharding to
QZ workers.  It does not contain audio codes or credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--zq-manifest",
        default=str(ROOT / "prepared/zq_targets_v1/manifest.jsonl"),
    )
    ap.add_argument(
        "--semantic-manifest",
        default=str(ROOT / "prepared/semantic_v1_v3_1_step3_no_text_20260715/manifest.jsonl"),
    )
    ap.add_argument(
        "--output",
        default=str(ROOT / "prepared/ddlfm_v1_index.jsonl"),
    )
    ap.add_argument(
        "--summary",
        default="",
        help="Optional summary path; defaults to <output>.summary.json",
    )
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def nested(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if value not in (None, ""):
        return value
    meta = row.get("moss_codecvc_meta")
    if isinstance(meta, dict):
        return meta.get(key)
    return None


def mode_of(row: dict[str, Any], split: str) -> str:
    value = row.get("moss_codecvc_mode") or nested(row, "moss_codecvc_mode") or split
    return str(value).strip().lower()


def token_ids(row: dict[str, Any]) -> list[int]:
    value = row.get("content_token_ids")
    if not isinstance(value, list):
        value = row.get("content_ref_token_ids")
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError("boolean content token id")
        result.append(int(item))
    return result


def read_at_handle(handle: Any, path: Path, offset: int) -> dict[str, Any]:
    handle.seek(int(offset))
    line = handle.readline()
    if not line:
        raise ValueError(f"empty source row at {path}:{offset}")
    row = json.loads(line)
    if not isinstance(row, dict):
        raise ValueError(f"source row is not an object at {path}:{offset}")
    return row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_semantic_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("utterance_id") or row.get("sample_id") or "")
            if not key:
                raise ValueError(f"semantic manifest row {line_no} has no utterance_id")
            if key in result:
                raise ValueError(f"duplicate semantic utterance_id: {key}")
            result[key] = row
    return result


def compact_record(
    zq: dict[str, Any], source: dict[str, Any], semantic: dict[str, Any] | None
) -> dict[str, Any]:
    split = str(zq.get("split") or "")
    uid = str(zq.get("utterance_id") or "")
    mode = mode_of(source, split)
    if mode not in {"no_text", "text"}:
        raise ValueError(f"unsupported moss_codecvc_mode={mode!r} for {uid}")

    zq_path = Path(str(zq.get("output_path") or "")).resolve()
    if not zq_path.is_file():
        raise FileNotFoundError(zq_path)
    target_frames = int(zq.get("num_frames") or 0)
    if target_frames <= 0:
        raise ValueError(f"invalid zq num_frames for {uid}: {target_frames}")

    speaker_path = source.get("timbre_ref_speaker_embedding_path")
    if not speaker_path:
        raise ValueError(f"missing timbre reference speaker embedding for {uid}")
    speaker_path = str(Path(str(speaker_path)).expanduser().resolve())
    if not Path(speaker_path).is_file():
        raise FileNotFoundError(speaker_path)

    out: dict[str, Any] = {
        "utterance_id": uid,
        "split": split,
        "moss_codecvc_mode": mode,
        "language": source.get("language"),
        "zq_path": str(zq_path),
        "zq_frames": target_frames,
        "zq_dim": int(zq.get("latent_dim") or 0),
        "zq_rate_hz": float(zq.get("frame_rate_hz") or 0.0),
        "duration_sec": float(zq.get("duration_sec") or 0.0),
        "speaker_embedding_path": speaker_path,
        "speaker_embedding_backend": "speechbrain_ecapa_192d_sidecar",
        "source_codec_frames": int(nested(source, "source_codec_frames") or 0),
        "target_codec_frames": int(nested(source, "target_codec_frames") or target_frames),
        "text": source.get("text"),
        "content_ref_text": source.get("content_ref_text"),
    }

    if mode == "no_text":
        if semantic is None:
            raise ValueError(f"missing no_text semantic row for {uid}")
        semantic_path = Path(str(semantic.get("semantic_v3_1_path") or "")).resolve()
        if not semantic_path.is_file():
            raise FileNotFoundError(semantic_path)
        semantic_frames = int(semantic.get("semantic_v3_1_frames") or 0)
        semantic_dim = int(semantic.get("semantic_v3_1_dim") or 0)
        if semantic_dim != 512:
            raise ValueError(f"unexpected no_text semantic dim={semantic_dim} for {uid}")
        out.update(
            {
                "semantic_kind": "wavlm_adapter_v3_1",
                "semantic_path": str(semantic_path),
                "semantic_frames": semantic_frames,
                "semantic_dim": semantic_dim,
                "semantic_rate_hz": float(semantic.get("semantic_v3_1_rate_hz") or 12.5),
            }
        )
    else:
        ids = token_ids(source)
        if not ids:
            raise ValueError(f"text row has no content_token_ids for {uid}")
        if any(item <= 0 or item >= 8001 for item in ids):
            raise ValueError(f"text content token outside [1,8000] for {uid}")
        out.update(
            {
                "semantic_kind": "text_tokens_source_token_memory",
                "content_token_ids": ids,
                "content_token_length": len(ids),
                "content_token_vocab_size": 8001,
                "content_token_padding_id": 0,
                "semantic_dim": 512,
                "semantic_rate_hz": None,
            }
        )
    return out


def main() -> int:
    args = parse_args()
    zq_path = Path(args.zq_manifest).expanduser().resolve()
    semantic_path = Path(args.semantic_manifest).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    summary_path = Path(args.summary).expanduser().resolve() if args.summary else output.with_suffix(".summary.json")
    if not zq_path.is_file():
        raise FileNotFoundError(zq_path)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")

    semantic_map = load_semantic_map(semantic_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    counts: Counter[str] = Counter()
    missing_semantic = 0
    rows_written = 0
    started = time.time()
    source_handles: dict[Path, Any] = {}
    try:
      with zq_path.open("r", encoding="utf-8") as source_handle, tmp.open("w", encoding="utf-8") as out_handle:
        for line_no, line in enumerate(source_handle, 1):
            if not line.strip():
                continue
            zq = json.loads(line)
            source_manifest = Path(str(zq.get("manifest") or "")).expanduser().resolve()
            if source_manifest not in source_handles:
                source_handles[source_manifest] = source_manifest.open("rb")
            source = read_at_handle(
                source_handles[source_manifest],
                source_manifest,
                int(zq.get("manifest_byte_offset") or 0),
            )
            uid = str(zq.get("utterance_id") or "")
            source_uid = str(source.get("sample_id") or "")
            if source_uid != uid:
                raise ValueError(
                    f"source/zq utterance mismatch at zq line {line_no}: {source_uid!r} != {uid!r}"
                )
            mode = mode_of(source, str(zq.get("split") or ""))
            semantic = semantic_map.get(uid) if mode == "no_text" else None
            if mode == "no_text" and semantic is None:
                missing_semantic += 1
            record = compact_record(zq, source, semantic)
            out_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            rows_written += 1
            counts[mode] += 1
            if args.max_rows and rows_written >= int(args.max_rows):
                break
    finally:
      for handle in source_handles.values():
        handle.close()
    os.replace(tmp, output)
    payload = {
        "schema": "ver3_1_ddlfm_index_v1",
        "status": "completed",
        "generated_at_unix": time.time(),
        "elapsed_sec": time.time() - started,
        "rows": rows_written,
        "mode_counts": dict(counts),
        "missing_no_text_semantic": missing_semantic,
        "zq_manifest": str(zq_path),
        "zq_manifest_sha256": sha256_file(zq_path),
        "semantic_manifest": str(semantic_path),
        "semantic_manifest_sha256": sha256_file(semantic_path),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "text_conditioning": "content_token_ids via SourceTokenMemoryEncoder; no target/source BNF",
        "speaker_conditioning": "timbre_ref_speaker_embedding_path; existing ECAPA 192-D sidecar",
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

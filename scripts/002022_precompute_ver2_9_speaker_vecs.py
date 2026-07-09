#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moss_codecvc.models.speaker_encoder import FrozenWavLMSVEncoder


def safe_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text[:180] or "row"


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc


def record_value(record: dict[str, Any], key: str) -> Any | None:
    if record.get(key) is not None:
        return record[key]
    meta = record.get("moss_codecvc_meta")
    if isinstance(meta, dict):
        return meta.get(key)
    return None


def record_id(record: dict[str, Any], fallback: str) -> str:
    for key in ("pair_id", "sample_id", "case_id", "utt_id", "id"):
        value = record_value(record, key)
        if value not in (None, ""):
            return safe_id(str(value))
    return safe_id(fallback)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Precompute ver2.9 frozen WavLM-SV speaker vectors.")
    ap.add_argument("--input-jsonl", required=True, help="Input train-ready JSONL manifest.")
    ap.add_argument("--output-jsonl", required=True, help="Output JSONL with speaker_vec_path added.")
    ap.add_argument("--speaker-vec-dir", default="", help="Directory for .npy vectors. Defaults to output parent/speaker_vecs.")
    ap.add_argument("--audio-key", default="timbre_ref_audio")
    ap.add_argument("--speaker-encoder-path", default="microsoft/wavlm-base-plus-sv")
    ap.add_argument("--speaker-embedding-dim", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--shard-index", type=int, default=-1, help="0-based shard index. Negative disables sharding.")
    ap.add_argument("--num-shards", type=int, default=1)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl).expanduser()
    output_jsonl = Path(args.output_jsonl).expanduser()
    speaker_vec_dir = Path(args.speaker_vec_dir).expanduser() if args.speaker_vec_dir else output_jsonl.parent / "speaker_vecs"
    batch_size = max(1, int(args.batch_size))
    num_shards = max(1, int(args.num_shards))
    shard_index = int(args.shard_index)
    if shard_index >= num_shards:
        raise ValueError(f"shard-index must be < num-shards, got {shard_index} >= {num_shards}")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    speaker_vec_dir.mkdir(parents=True, exist_ok=True)

    encoder = FrozenWavLMSVEncoder(
        args.speaker_encoder_path,
        embedding_dim=int(args.speaker_embedding_dim) if int(args.speaker_embedding_dim) > 0 else None,
        local_files_only=bool(args.local_files_only),
    )
    device = torch.device(args.device)
    rows = 0
    scanned = 0
    written = 0
    reused = 0
    missing = 0

    def flush_batch(out, batch: list[dict[str, Any]]) -> None:
        nonlocal written, reused, missing
        encode_paths: list[str] = []
        encode_items: list[tuple[dict[str, Any], Path, str]] = []
        for item in batch:
            record = item["record"]
            audio_path = item["audio_path"]
            vec_path = item["vec_path"]
            if not audio_path:
                missing += 1
                record["speaker_vec_path"] = None
                continue
            record["speaker_vec_path"] = str(vec_path)
            if args.overwrite or not vec_path.exists():
                encode_paths.append(str(audio_path))
                encode_items.append((record, vec_path, str(audio_path)))
            else:
                reused += 1
        if encode_paths:
            emb, mask = encoder(encode_paths, device=device, dtype=torch.float32)
            if emb is None or mask is None:
                raise RuntimeError("failed to compute speaker vec batch")
            for row_idx, (_record, vec_path, audio_path) in enumerate(encode_items):
                if not bool(mask[row_idx].item()):
                    raise RuntimeError(f"failed to compute speaker vec for {audio_path}")
                tmp_path = vec_path.with_name(f"{vec_path.name}.tmp{os.getpid()}.npy")
                np.save(tmp_path, emb[row_idx].detach().cpu().numpy().astype(np.float32))
                tmp_path.replace(vec_path)
                written += 1
        for item in batch:
            out.write(json.dumps(item["record"], ensure_ascii=False) + "\n")

    batch: list[dict[str, Any]] = []
    with output_jsonl.open("w", encoding="utf-8") as out:
        for line_no, record in iter_jsonl(input_jsonl):
            scanned += 1
            if shard_index >= 0 and (line_no - 1) % num_shards != shard_index:
                continue
            if args.max_rows > 0 and rows >= args.max_rows:
                break
            rows += 1
            audio_path = record_value(record, args.audio_key)
            pair_id = record_id(record, f"{input_jsonl.stem}_{line_no}")
            vec_path = speaker_vec_dir / f"{pair_id}.npy"
            batch.append({"record": record, "audio_path": audio_path, "vec_path": vec_path})
            if len(batch) >= batch_size:
                flush_batch(out, batch)
                batch.clear()
            if rows % 1000 == 0:
                print(
                    f"[speaker-vec] rows={rows} scanned={scanned} written={written} "
                    f"reused={reused} missing={missing}",
                    flush=True,
                )
        if batch:
            flush_batch(out, batch)
            batch.clear()
    print(
        f"[speaker-vec] done input={input_jsonl} output={output_jsonl} rows={rows} scanned={scanned} "
        f"written={written} reused={reused} missing_audio={missing} speaker_vec_dir={speaker_vec_dir} "
        f"batch_size={batch_size} shard_index={shard_index} num_shards={num_shards}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

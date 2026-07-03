#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.config import deep_get, load_config
from moss_codecvc.io_utils import iter_jsonl, load_torch_file, safe_stem, stable_id, write_jsonl
from moss_codecvc.moss_codec import MossCodec


FIELD_TO_PATH = {
    "source": "source_audio",
    "timbre": "timbre_ref_audio",
    "target": "target_audio",
}


def encode_one(codec: MossCodec, audio_path: str, out_dir: Path, n_vq: int, reuse: bool = True) -> dict[str, Any]:
    key = stable_id(audio_path, n_vq, length=20)
    out_path = out_dir / f"{safe_stem(Path(audio_path).stem)}.{key}.pt"
    if reuse and out_path.exists():
        payload = load_torch_file(out_path)
        return {
            "codes_path": str(out_path),
            "num_frames": int(payload.get("num_frames", payload["codes"].shape[0])),
            "n_vq": int(payload.get("n_vq", payload["codes"].shape[1])),
            "duration_sec": payload.get("duration_sec"),
            "reused": True,
        }
    result = codec.encode_path(audio_path, n_vq=n_vq)
    payload = {
        "codes": result["codes"],
        "audio_path": audio_path,
        "num_frames": result["num_frames"],
        "n_vq": result["n_vq"],
        "duration_sec": result["duration_sec"],
        "sample_rate": result["sample_rate"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return {
        "codes_path": str(out_path),
        "num_frames": result["num_frames"],
        "n_vq": result["n_vq"],
        "duration_sec": result["duration_sec"],
        "reused": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--codes-dir", required=True)
    ap.add_argument("--n-vq", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--fields", default="source,timbre,target")
    ap.add_argument("--no-reuse", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    moss_root = deep_get(cfg, "moss.root")
    codec_path = deep_get(cfg, "moss.codec_path")
    n_vq = args.n_vq or int(deep_get(cfg, "moss.default_n_vq", 32))
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]

    codec = MossCodec(codec_path, moss_root=moss_root, device=args.device, dtype=args.dtype)
    codes_dir = Path(args.codes_dir).expanduser()

    out_rows = []
    encoded = 0
    skipped = 0
    failed = 0
    for idx, row in enumerate(iter_jsonl(args.input_jsonl)):
        new_row = dict(row)
        meta = dict(new_row.get("codec_meta") or {})
        for field in fields:
            path_key = FIELD_TO_PATH[field]
            audio_path = new_row.get(path_key)
            if not audio_path:
                skipped += 1
                continue
            field_dir = codes_dir / field
            try:
                result = encode_one(codec, audio_path, field_dir, n_vq, reuse=not args.no_reuse)
            except Exception as exc:
                failed += 1
                print(
                    f"encode error row={idx} field={field} sample_id={new_row.get('sample_id')} "
                    f"err={type(exc).__name__}: {exc}",
                    flush=True,
                )
                new_row.setdefault("codec_errors", {})[field] = f"{type(exc).__name__}: {exc}"
                continue
            prefix = "timbre_ref" if field == "timbre" else field
            new_row[f"{prefix}_audio_codes_path"] = result["codes_path"]
            new_row[f"{prefix}_codec_frames"] = result["num_frames"]
            new_row[f"{prefix}_duration_sec"] = result["duration_sec"]
            encoded += 0 if result["reused"] else 1
            meta[f"{field}_reused"] = result["reused"]
        new_row["codec_meta"] = {
            **meta,
            "n_vq": n_vq,
            "codec_path": str(codec_path),
        }
        out_rows.append(new_row)
        if (idx + 1) % 50 == 0:
            print(f"processed {idx + 1} rows encoded_new={encoded} failed_fields={failed}", flush=True)

    n = write_jsonl(args.output_jsonl, out_rows)
    print(
        f"wrote {n} encoded rows -> {Path(args.output_jsonl).resolve()} "
        f"encoded_new={encoded} skipped_fields={skipped} failed_fields={failed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

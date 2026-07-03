#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.config import deep_get, load_config
from moss_codecvc.io_utils import load_torch_file
from moss_codecvc.moss_codec import MossCodec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--codes-pt", required=True)
    ap.add_argument("--output-wav", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    cfg = load_config(args.config)
    codec = MossCodec(
        deep_get(cfg, "moss.codec_path"),
        moss_root=deep_get(cfg, "moss.root"),
        device=args.device,
        dtype=args.dtype,
    )
    payload = load_torch_file(args.codes_pt)
    codes = payload["codes"] if isinstance(payload, dict) else payload
    wav = codec.decode_codes(codes)
    codec.save_wav(args.output_wav, wav)
    print(f"wrote {Path(args.output_wav).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

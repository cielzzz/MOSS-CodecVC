#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.config import deep_get, load_config


def format_model_ref(value: Any) -> tuple[str, bool]:
    ref = str(value or "")
    if not ref:
        return "", False
    return ref, Path(ref).expanduser().exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    moss_root = Path(deep_get(cfg, "moss.root", ""))
    model_ref, model_ref_exists = format_model_ref(deep_get(cfg, "moss.model_path", ""))
    codec_path = Path(deep_get(cfg, "moss.codec_path", ""))
    expected_n_vq = int(deep_get(cfg, "training.n_vq", deep_get(cfg, "moss.default_n_vq", 0)) or 0)

    print(f"project_root={ROOT}")
    print(f"moss_root={moss_root} exists={moss_root.exists()}")
    print(f"model_path={model_ref} local_exists={model_ref_exists}")
    print(f"codec_path={codec_path} exists={codec_path.exists()}")

    if str(moss_root) not in sys.path:
        sys.path.insert(0, str(moss_root))

    checks = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("torchaudio", "torchaudio"),
        ("moss_tts_delay", "moss_tts_delay.processing_moss_tts"),
    ]
    ok = True
    for label, module_name in checks:
        try:
            __import__(module_name)
            print(f"import {label}: ok")
        except Exception as exc:
            ok = False
            print(f"import {label}: failed: {type(exc).__name__}: {exc}")

    if not moss_root.exists() or not codec_path.exists():
        ok = False

    try:
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(
            model_ref,
            trust_remote_code=True,
            local_files_only=True,
        )
        model_n_vq = int(getattr(model_config, "n_vq", 0) or 0)
        print(
            "model_config: ok "
            f"model_type={getattr(model_config, 'model_type', None)} "
            f"n_vq={model_n_vq}"
        )
        if expected_n_vq and model_n_vq and expected_n_vq != model_n_vq:
            ok = False
            print(f"n_vq mismatch: config training.n_vq={expected_n_vq} model n_vq={model_n_vq}")
    except Exception as exc:
        ok = False
        print(f"model_config: failed: {type(exc).__name__}: {exc}")

    try:
        from transformers import AutoConfig

        codec_config = AutoConfig.from_pretrained(
            str(codec_path),
            trust_remote_code=True,
            local_files_only=True,
        )
        print(f"codec_config: ok model_type={getattr(codec_config, 'model_type', None)}")
    except Exception as exc:
        ok = False
        print(f"codec_config: failed: {type(exc).__name__}: {exc}")

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

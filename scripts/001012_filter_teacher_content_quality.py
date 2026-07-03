#!/usr/bin/env python
from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> int:
    impl_path = Path(__file__).resolve().with_name("001017_asr_content_filter.py")
    spec = importlib.util.spec_from_file_location("moss_codecvc_asr_content_filter", impl_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load ASR content filter implementation: {impl_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).with_name("004027_audit_ctc_content_targets.py")
    runpy.run_path(str(target), run_name="__main__")

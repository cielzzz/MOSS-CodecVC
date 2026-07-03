from __future__ import annotations

import os
import sys
from pathlib import Path


DEFAULT_DOWNLOAD_ROOT = Path("/inspire/ssd/project/embodied-multimodality/public/xyzhang/download")
DEFAULT_SPEECHBRAIN_DEPS = DEFAULT_DOWNLOAD_ROOT / "python_deps" / "speechbrain_py312"
DEFAULT_SPEECHBRAIN_ECAPA_DIR = DEFAULT_DOWNLOAD_ROOT / "models" / "speechbrain" / "spkrec-ecapa-voxceleb"


def add_download_python_deps(download_root: str | Path | None = None) -> list[Path]:
    """Add project-managed third-party dependency directories to sys.path."""
    root = Path(download_root or os.environ.get("MOSS_CODECVC_DOWNLOAD_ROOT") or DEFAULT_DOWNLOAD_ROOT).expanduser()
    candidates = [
        root / "python_deps" / "speechbrain_py312",
    ]
    added: list[Path] = []
    for path in candidates:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
            added.append(path)
    return added


def default_speechbrain_ecapa_dir(download_root: str | Path | None = None) -> Path:
    root = Path(download_root or os.environ.get("MOSS_CODECVC_DOWNLOAD_ROOT") or DEFAULT_DOWNLOAD_ROOT).expanduser()
    return root / "models" / "speechbrain" / "spkrec-ecapa-voxceleb"

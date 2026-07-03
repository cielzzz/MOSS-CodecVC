from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(p)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "PyYAML is required for YAML configs. Install pyyaml or pass a JSON config."
        ) from exc
    data = yaml.safe_load(text)
    return data or {}


def deep_get(data: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = data
    for key in dotted_key.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

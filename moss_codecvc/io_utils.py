from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def pick_first(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def stable_id(*parts: Any, length: int = 16) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(str(part).encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:length]


def safe_stem(path_or_id: str, fallback: str = "item") -> str:
    text = str(path_or_id).strip() or fallback
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out[:120] or fallback


def load_torch_file(path: str | Path, *, map_location: str = "cpu") -> Any:
    """Load a local torch payload with the safer weights-only mode when available."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

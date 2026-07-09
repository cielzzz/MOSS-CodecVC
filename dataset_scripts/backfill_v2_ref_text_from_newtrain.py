#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    tmp = path.with_name(path.name + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(path)
    return count


def dataset_name(row: dict[str, Any]) -> str:
    return str(
        row.get("dataset_name")
        or row.get("meta", {}).get("source_fields", {}).get("dataset_name")
        or ""
    )


def reference_path(row: dict[str, Any]) -> str:
    return str(
        row.get("reference")
        or row.get("meta", {}).get("source_fields", {}).get("reference")
        or ""
    )


def set_if_present(row: dict[str, Any], key: str, value: str) -> None:
    if key in row:
        row[key] = value


def backfill_row(row: dict[str, Any], text: str) -> bool:
    if not text:
        return False
    changed = False
    before = json.dumps(row, ensure_ascii=False, sort_keys=True)

    set_if_present(row, "u2_text", text)
    set_if_present(row, "timbre_ref_text", text)

    meta = row.get("meta")
    if isinstance(meta, dict):
        u2 = meta.get("u2")
        if isinstance(u2, dict) and "text" in u2:
            u2["text"] = text
        source_job = meta.get("source_seedvc_job")
        if isinstance(source_job, dict):
            metadata = source_job.get("metadata")
            if isinstance(metadata, dict) and "final_timbre_ref_text" in metadata:
                metadata["final_timbre_ref_text"] = text

    after = json.dumps(row, ensure_ascii=False, sort_keys=True)
    changed = before != after
    return changed


def load_rows(paths: list[Path]) -> tuple[dict[tuple[str, str], str], dict[Path, list[dict[str, Any]]]]:
    wanted: dict[tuple[str, str], str] = {}
    rows_by_path: dict[Path, list[dict[str, Any]]] = {}
    for path in paths:
        rows = list(iter_jsonl(path))
        rows_by_path[path] = rows
        for row in rows:
            ds = dataset_name(row)
            ref = reference_path(row)
            if ds and ref:
                wanted[(ds, ref)] = ""
    return wanted, rows_by_path


def run_rg(patterns: list[str], files: list[Path]) -> list[dict[str, Any]]:
    if not patterns or not files:
        return []
    rg = shutil.which("rg")
    if not rg:
        raise SystemExit("rg is required for fast targeted lookup but was not found in PATH")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=True) as handle:
        handle.write("\n".join(sorted(set(patterns))) + "\n")
        handle.flush()
        cmd = [rg, "--json", "-F", "-f", handle.name, *[str(path) for path in files]]
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode not in (0, 1):
        raise SystemExit(f"rg failed with exit={result.returncode}\n{result.stderr}")
    out: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        event = json.loads(line)
        if event.get("type") != "match":
            continue
        text = event.get("data", {}).get("lines", {}).get("text") or ""
        path = event.get("data", {}).get("path", {}).get("text") or ""
        if not text.strip():
            continue
        out.append({"path": path, "line": json.loads(text)})
    return out


def build_lookup(mapping_json: Path, wanted: dict[tuple[str, str], str]) -> dict[tuple[str, str], str]:
    mapping = json.loads(mapping_json.read_text(encoding="utf-8"))
    refs_by_dataset: dict[str, set[str]] = defaultdict(set)
    for ds, ref in wanted:
        refs_by_dataset[ds].add(ref)

    file_to_dataset: dict[str, str] = {}
    files: list[Path] = []
    patterns: list[str] = []
    for ds, refs in refs_by_dataset.items():
        item = mapping.get(ds) or {}
        newtrain = Path(str(item.get("newtrain_jsonl") or ""))
        if not newtrain.exists():
            continue
        file_to_dataset[str(newtrain)] = ds
        files.append(newtrain)
        patterns.extend(refs)

    found: dict[tuple[str, str], str] = {}
    for match in run_rg(patterns, files):
        ds = file_to_dataset.get(match["path"], "")
        line = match["line"]
        ref = str(line.get("audio_segment_path") or "")
        text = str(line.get("asr_text") or "").strip()
        if ds and ref and text:
            found[(ds, ref)] = text
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping-json", required=True)
    parser.add_argument("--jsonl", action="append", required=True)
    parser.add_argument("--in-place", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    paths = [Path(path).expanduser().resolve(strict=False) for path in args.jsonl]
    wanted, rows_by_path = load_rows(paths)
    lookup = build_lookup(Path(args.mapping_json).expanduser().resolve(strict=False), wanted)

    total_changed = 0
    for path, rows in rows_by_path.items():
        changed = 0
        for row in rows:
            text = lookup.get((dataset_name(row), reference_path(row)), "")
            if backfill_row(row, text):
                changed += 1
        if args.in_place:
            write_jsonl(path, rows)
        total_changed += changed
        print(f"{path}\trows={len(rows)}\tchanged={changed}")
    print(f"wanted={len(wanted)} found={len(lookup)} changed={total_changed}")
    missing = sorted(set(wanted) - set(lookup))
    if missing:
        print("missing:")
        for ds, ref in missing[:20]:
            print(f"  {ds}\t{ref}")
    return 0 if lookup else 1


if __name__ == "__main__":
    raise SystemExit(main())

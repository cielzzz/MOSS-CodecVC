#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract a small Emilia WebDataset tar subset to source JSONL.")
    parser.add_argument("--tar", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--limit", type=int, default=240)
    parser.add_argument("--min-duration", type=float, default=3.0)
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument("--min-dnsmos", type=float, default=3.0)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    pending: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    def maybe_flush(key: str) -> None:
        item = pending.get(key) or {}
        meta = item.get("json")
        audio_payload = item.get("audio_payload")
        audio_suffix = item.get("audio_suffix")
        if not meta or not audio_payload or not audio_suffix:
            return
        duration = float(meta.get("duration") or 0.0)
        if duration < args.min_duration or duration > args.max_duration:
            pending.pop(key, None)
            return
        dnsmos = meta.get("dnsmos")
        if dnsmos is not None and float(dnsmos) < args.min_dnsmos:
            pending.pop(key, None)
            return
        text = meta.get("text")
        if not text:
            pending.pop(key, None)
            return
        audio_path = audio_dir / f"{Path(key).name}{audio_suffix}"
        if not audio_path.exists():
            audio_path.write_bytes(audio_payload)
        rows.append(
            {
                "audio": str(audio_path),
                "text": text,
                "language": args.language,
                "duration": duration,
                "speaker": meta.get("speaker"),
                "dnsmos": dnsmos,
                "recording_id": meta.get("id") or Path(key).name,
                "source": "emilia_webdataset",
                "source_tar": str(Path(args.tar).resolve()),
            }
        )
        pending.pop(key, None)

    with tarfile.open(args.tar, "r") as tar:
        for member in tar:
            if len(rows) >= args.limit:
                break
            if not member.isfile():
                continue
            suffix = Path(member.name).suffix.lower()
            if suffix not in {".json", ".mp3", ".wav", ".flac"}:
                continue
            key = str(Path(member.name).with_suffix(""))
            item = pending.setdefault(key, {})
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            payload = fileobj.read()
            if suffix == ".json":
                item["json"] = json.loads(payload.decode("utf-8"))
            else:
                item["audio_payload"] = payload
                item["audio_suffix"] = suffix
            maybe_flush(key)

    n = write_jsonl(args.output_jsonl, rows)
    summary_path = Path(args.output_jsonl).with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "tar": str(Path(args.tar).resolve()),
                "output_jsonl": str(Path(args.output_jsonl).resolve()),
                "audio_dir": str(audio_dir.resolve()),
                "rows_written": n,
                "language": args.language,
                "min_duration": args.min_duration,
                "max_duration": args.max_duration,
                "min_dnsmos": args.min_dnsmos,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote {n} Emilia WebDataset source rows -> {Path(args.output_jsonl).resolve()}")
    print(f"summary -> {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

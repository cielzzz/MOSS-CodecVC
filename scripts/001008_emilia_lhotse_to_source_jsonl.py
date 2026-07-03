#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Iterator


def iter_jsonl_gz(path: str | Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def first_audio_source(recording: dict[str, Any]) -> str | None:
    sources = recording.get("sources") or []
    if not sources:
        return None
    source = sources[0].get("source")
    return str(source) if source else None


def maybe_rewrite_audio_path(audio: str, audio_cache_root: str, cache_language: str) -> str:
    if not audio_cache_root:
        return audio
    marker = "/fc71e07/"
    if marker not in audio:
        return audio
    rel = audio.split(marker, 1)[1]
    if audio.endswith(".mp3"):
        rel = rel[:-4] + ".flac"
    return str(Path(audio_cache_root) / "emilia" / cache_language / "24000" / rel)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Emilia Lhotse manifests to source JSONL for Seed-VC pairing.")
    parser.add_argument("--recordings-gz", required=True)
    parser.add_argument("--supervisions-gz", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--language", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-duration", type=float, default=3.0)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--min-dnsmos", type=float, default=3.0)
    parser.add_argument("--audio-cache-root", default="")
    parser.add_argument("--cache-language", default="zh")
    parser.add_argument("--require-existing-audio", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    seen_speakers: dict[str, int] = {}
    scanned = 0
    rec_iter = iter_jsonl_gz(args.recordings_gz)
    sup_iter = iter_jsonl_gz(args.supervisions_gz)
    for rec, sup in zip(rec_iter, sup_iter):
        scanned += 1
        rec_id = rec.get("id")
        sup_rec_id = sup.get("recording_id")
        if rec_id != sup_rec_id:
            raise RuntimeError(
                f"recording/supervision order mismatch at line {scanned}: "
                f"recording_id={rec_id!r} supervision_recording_id={sup_rec_id!r}"
            )
        audio = first_audio_source(rec)
        if not audio:
            continue
        audio = maybe_rewrite_audio_path(audio, args.audio_cache_root, args.cache_language)
        if args.require_existing_audio and not Path(audio).exists():
            continue
        duration = float(sup.get("duration") or rec.get("duration") or 0.0)
        if duration < args.min_duration or duration > args.max_duration:
            continue
        custom = sup.get("custom") or {}
        dnsmos = custom.get("dnsmos")
        if dnsmos is not None and float(dnsmos) < args.min_dnsmos:
            continue
        text = sup.get("text") or custom.get("raw_text")
        if not text:
            continue
        speaker = sup.get("speaker") or ""
        rows.append(
            {
                "audio": audio,
                "text": text,
                "language": args.language or sup.get("language"),
                "duration": duration,
                "speaker": speaker,
                "dnsmos": dnsmos,
                "recording_id": sup.get("recording_id"),
                "supervision_id": sup.get("id"),
                "sampling_rate": rec.get("sampling_rate"),
                "source": "emilia_lhotse",
                "source_jsonl": custom.get("source_jsonl"),
            }
        )
        if speaker:
            seen_speakers[speaker] = seen_speakers.get(speaker, 0) + 1
        if args.limit > 0 and len(rows) >= args.limit:
            break

    n = write_jsonl(args.output_jsonl, rows)
    summary_path = Path(args.output_jsonl).with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "recordings_gz": str(Path(args.recordings_gz).resolve()),
                "supervisions_gz": str(Path(args.supervisions_gz).resolve()),
                "output_jsonl": str(Path(args.output_jsonl).resolve()),
                "scanned_supervisions": scanned,
                "rows_written": n,
                "unique_speakers": len(seen_speakers),
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
    print(f"wrote {n} Emilia source rows -> {Path(args.output_jsonl).resolve()}")
    print(f"summary -> {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

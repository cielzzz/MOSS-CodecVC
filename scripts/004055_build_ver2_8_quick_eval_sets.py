#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION = ROOT / "testset/validation/seedtts_vc_ver2_3_validation.jsonl"
DEFAULT_PREPARED = (
    ROOT
    / "trainset/ver2_8_prepared_zh45w_en22w_plus_zh11w_en11w_0005_0015_merged_no_text_plus_zh3w_text_textrep10"
)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())[:160] or "case"


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def duration_sec(row: dict[str, Any], role: str, frame_rate: float) -> float | None:
    meta = row.get("moss_codecvc_meta") or {}
    for container in (row, meta):
        for key in (f"{role}_duration_sec", f"{role}_duration", f"{role}_audio_duration_sec"):
            value = finite_float(container.get(key))
            if value is not None and value > 0:
                return value
    frames = finite_float(row.get(f"{role}_codec_frames"))
    if frames is None:
        frames = finite_float(meta.get(f"{role}_codec_frames"))
    if frames is not None and frames > 0 and frame_rate > 0:
        return frames / frame_rate
    return None


def select_balanced_no_text(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("mode") == "no_text":
            by_cell[str(row.get("cell") or "unknown")].append(row)
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for cell in sorted(by_cell):
            bucket = by_cell[cell]
            if bucket:
                selected.append(bucket.pop(0))
                progressed = True
                if len(selected) >= count:
                    break
        if not progressed:
            break
    return selected


def prepared_row_to_validation(row: dict[str, Any], index: int, source_duration: float | None) -> dict[str, Any] | None:
    meta = row.get("moss_codecvc_meta") or {}
    source_audio = str(meta.get("source_audio") or row.get("source_audio") or "")
    timbre_ref_audio = str(meta.get("timbre_ref_audio") or row.get("timbre_ref_audio") or "")
    if not source_audio or not timbre_ref_audio:
        return None
    content_text = str(
        row.get("content_ref_text")
        or row.get("asr_src_text")
        or row.get("source_text")
        or row.get("text")
        or ""
    ).strip()
    sample_id = str(row.get("sample_id") or f"prepared_valid_{index:06d}")
    return {
        "case_id": f"ver2_8_t11_domain_no_text_{index:06d}_{safe_id(sample_id)}",
        "mode": "no_text",
        "cell": "prepared_valid_no_text_2_8s",
        "source_audio": source_audio,
        "source_text": content_text,
        "source_lang": row.get("language"),
        "source_id": sample_id,
        "source_duration_sec": source_duration,
        "timbre_ref_audio": timbre_ref_audio,
        "timbre_ref_text": row.get("timbre_ref_text") or "",
        "ref_lang": row.get("language"),
        "ref_id": safe_id(str(meta.get("timbre_ref_audio") or "ref")),
        "text": "<NO_TEXT>",
        "content_ref_text": content_text,
        "eval_text_source": "content_ref_text",
        "prepared_sample_id": sample_id,
        "teacher_target_audio": meta.get("target_audio"),
        "teacher_asr_tgt_text": row.get("asr_tgt_text"),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build fixed Ver2.8 timbre quick-eval and T11 domain eval sets.")
    ap.add_argument("--seedtts-validation-jsonl", default=str(DEFAULT_VALIDATION))
    ap.add_argument("--prepared-dir", default=str(DEFAULT_PREPARED))
    ap.add_argument("--quick-output-jsonl", required=True)
    ap.add_argument("--domain-output-jsonl", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--quick-count", type=int, default=20)
    ap.add_argument("--domain-count", type=int, default=50)
    ap.add_argument("--min-duration-sec", type=float, default=2.0)
    ap.add_argument("--max-duration-sec", type=float, default=8.0)
    ap.add_argument("--frame-rate", type=float, default=12.5)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    seedtts_rows = list(iter_jsonl(Path(args.seedtts_validation_jsonl)))
    quick_rows = select_balanced_no_text(seedtts_rows, int(args.quick_count))

    prepared_no_text_valid = Path(args.prepared_dir) / "no_text.valid.jsonl"
    domain_rows: list[dict[str, Any]] = []
    scanned = 0
    for row in iter_jsonl(prepared_no_text_valid):
        scanned += 1
        if row.get("moss_codecvc_mode") != "no_text" and row.get("text") != "<NO_TEXT>":
            continue
        if row.get("content_keep") is False:
            continue
        dur = duration_sec(row, "source", float(args.frame_rate))
        if dur is None or dur < float(args.min_duration_sec) or dur > float(args.max_duration_sec):
            continue
        converted = prepared_row_to_validation(row, len(domain_rows), dur)
        if converted is None:
            continue
        domain_rows.append(converted)
        if len(domain_rows) >= int(args.domain_count):
            break

    write_jsonl(Path(args.quick_output_jsonl), quick_rows)
    write_jsonl(Path(args.domain_output_jsonl), domain_rows)
    payload = {
        "seedtts_validation_jsonl": str(Path(args.seedtts_validation_jsonl)),
        "prepared_dir": str(Path(args.prepared_dir)),
        "quick_output_jsonl": str(Path(args.quick_output_jsonl)),
        "domain_output_jsonl": str(Path(args.domain_output_jsonl)),
        "quick_rows": len(quick_rows),
        "domain_rows": len(domain_rows),
        "domain_scanned_rows": scanned,
        "domain_duration_sec": {
            "min": float(args.min_duration_sec),
            "max": float(args.max_duration_sec),
            "frame_rate": float(args.frame_rate),
        },
        "quick_case_ids": [row.get("case_id") for row in quick_rows],
        "domain_case_ids": [row.get("case_id") for row in domain_rows],
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[quick-eval-sets] quick={len(quick_rows)} domain={len(domain_rows)} summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

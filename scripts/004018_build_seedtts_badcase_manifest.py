#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_JSONL = ROOT / "testset/validation/seedtts_vc_ver2_3_validation.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "testset/outputs/ver2_3_ctc_clean_seedtts_valid_full"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge manual review and ASR eval into a SeedTTS badcase manifest.")
    ap.add_argument("--validation-jsonl", default=str(DEFAULT_VALIDATION_JSONL))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--manifest-jsonl", action="append", default=[])
    ap.add_argument("--asr-jsonl", default="")
    ap.add_argument("--manual-review-json", default="", help="JSON exported by the listening page.")
    ap.add_argument("--output-jsonl", default="")
    ap.add_argument("--output-summary", default="")
    ap.add_argument("--include-asr-filtered", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include-manual-bad", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-cer", type=float, default=0.20)
    ap.add_argument("--max-wer", type=float, default=0.25)
    ap.add_argument("--max-repeat-score", type=float, default=0.30)
    ap.add_argument("--max-duration-ratio", type=float, default=1.80)
    ap.add_argument("--min-duration-ratio", type=float, default=0.50)
    return ap.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def expected_text(row: dict[str, Any]) -> str:
    if str(row.get("mode") or "") == "text":
        text = str(row.get("text") or "").strip()
        if text and text != "<NO_TEXT>":
            return text
    return str(row.get("content_ref_text") or row.get("source_text") or "").strip()


def load_manual(path: str) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    obj = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    items = obj.get("items") if isinstance(obj, dict) else obj
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        case_id = str(item.get("case_id") or "")
        if case_id:
            out[case_id] = item
    return out


def main() -> int:
    args = parse_args()
    validation_jsonl = Path(args.validation_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifests = [Path(item).expanduser().resolve() for item in args.manifest_jsonl]
    if not manifests:
        manifests = [
            output_dir / "manifest.shard0.jsonl",
            output_dir / "manifest.shard1.jsonl",
            output_dir / "manifest.rerun_failed.jsonl",
            output_dir / "manifest.jsonl",
        ]

    validation = {str(row.get("case_id") or ""): row for row in iter_jsonl(validation_jsonl)}
    manifest_by_case: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        if not manifest.exists():
            continue
        for row in iter_jsonl(manifest):
            case_id = str(row.get("case_id") or "")
            if case_id:
                manifest_by_case[case_id] = row

    asr_by_case: dict[str, dict[str, Any]] = {}
    if args.asr_jsonl:
        for row in iter_jsonl(Path(args.asr_jsonl).expanduser()):
            case_id = str(row.get("case_id") or row.get("sample_id") or "")
            if case_id:
                asr_by_case[case_id] = row
    manual_by_case = load_manual(args.manual_review_json)

    bad_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for case_id, val in sorted(validation.items()):
        asr = asr_by_case.get(case_id, {})
        manual = manual_by_case.get(case_id, {})
        reasons: list[str] = []
        issues = list(manual.get("issues") or []) if isinstance(manual.get("issues"), list) else []
        if args.include_manual_bad and manual.get("rating") == "bad":
            reasons.append("manual_bad")
            reasons.extend(f"manual:{issue}" for issue in issues)
        if args.include_asr_filtered and asr:
            if asr.get("content_keep") is False:
                reasons.append("asr_filtered")
                for item in str(asr.get("content_filter_reason") or "").split(","):
                    item = item.strip()
                    if item and item != "keep":
                        reasons.append(f"asr:{item}")
            cer = finite_float(asr.get("cer_tgt"))
            wer = finite_float(asr.get("wer_tgt"))
            repeat = finite_float(asr.get("repeat_score"))
            duration = finite_float(asr.get("duration_ratio_tgt_src"))
            if cer is not None and cer > args.max_cer:
                reasons.append("metric:cer")
            if wer is not None and wer > args.max_wer:
                reasons.append("metric:wer")
            if repeat is not None and repeat > args.max_repeat_score:
                reasons.append("metric:repeat")
            if duration is not None and duration > args.max_duration_ratio:
                reasons.append("metric:target_too_long")
            if duration is not None and duration < args.min_duration_ratio:
                reasons.append("metric:target_too_short")
        reasons = sorted(set(reasons))
        if not reasons:
            continue
        for reason in reasons:
            reason_counts[reason] += 1
        manifest = manifest_by_case.get(case_id, {})
        out = {
            "case_id": case_id,
            "mode": val.get("mode"),
            "cell": val.get("cell"),
            "source_lang": val.get("source_lang"),
            "ref_lang": val.get("ref_lang"),
            "source_audio": val.get("source_audio"),
            "timbre_ref_audio": val.get("timbre_ref_audio"),
            "target_audio": asr.get("target_audio") or manifest.get("output_wav"),
            "expected_text": expected_text(val),
            "source_text": val.get("source_text"),
            "timbre_ref_text": val.get("timbre_ref_text"),
            "input_text": val.get("text"),
            "badcase_reasons": reasons,
            "manual_rating": manual.get("rating"),
            "manual_issues": issues,
            "manual_note": manual.get("note"),
            "asr_tgt_text": asr.get("asr_tgt_text"),
            "cer_tgt": asr.get("cer_tgt"),
            "wer_tgt": asr.get("wer_tgt"),
            "repeat_score": asr.get("repeat_score"),
            "duration_ratio_tgt_src": asr.get("duration_ratio_tgt_src"),
            "content_keep": asr.get("content_keep"),
            "content_filter_reason": asr.get("content_filter_reason"),
        }
        bad_rows.append(out)

    output_jsonl = Path(args.output_jsonl).expanduser() if args.output_jsonl else output_dir / "seedtts_badcases.jsonl"
    summary_path = Path(args.output_summary).expanduser() if args.output_summary else Path(str(output_jsonl) + ".summary.json")
    write_jsonl(output_jsonl, bad_rows)
    summary = {
        "validation_rows": len(validation),
        "asr_rows": len(asr_by_case),
        "manual_rows": len(manual_by_case),
        "badcase_rows": len(bad_rows),
        "mode_counts": dict(Counter(str(row.get("mode") or "") for row in bad_rows)),
        "reason_counts": dict(reason_counts.most_common()),
        "output_jsonl": str(output_jsonl),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[badcase] rows={len(bad_rows)} output={output_jsonl}")
    print(f"[badcase] summary={summary_path}")
    print(f"[badcase] reasons={dict(reason_counts.most_common(20))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

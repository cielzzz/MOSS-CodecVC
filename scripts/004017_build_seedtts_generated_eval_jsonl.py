#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_JSONL = ROOT / "testset/validation/seedtts_vc_ver2_3_validation.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "testset/outputs/ver2_3_ctc_clean_seedtts_valid_full"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build ASR/content-eval input JSONL from SeedTTS validation inference outputs.")
    ap.add_argument("--validation-jsonl", default=str(DEFAULT_VALIDATION_JSONL))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--manifest-jsonl", action="append", default=[])
    ap.add_argument("--run-id", default="ver2_3")
    ap.add_argument("--output-jsonl", default="")
    ap.add_argument("--status", default="ok,ok_after_rerun", help="Comma-separated manifest statuses to keep; empty keeps all.")
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


def safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return stem[:180] or "case"


def expected_text(row: dict[str, Any]) -> str:
    mode = str(row.get("mode") or "")
    if mode == "text":
        text = str(row.get("text") or "").strip()
        if text and text != "<NO_TEXT>":
            return text
    return str(row.get("content_ref_text") or row.get("source_text") or "").strip()


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
    keep_status = {item.strip() for item in str(args.status or "").split(",") if item.strip()}

    validation_rows = {str(row.get("case_id") or ""): row for row in iter_jsonl(validation_jsonl)}
    manifest_rows: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        if not manifest.exists():
            continue
        for row in iter_jsonl(manifest):
            case_id = str(row.get("case_id") or "")
            if case_id:
                manifest_rows[case_id] = row

    rows: list[dict[str, Any]] = []
    for case_id, val in sorted(validation_rows.items()):
        manifest = manifest_rows.get(case_id, {})
        output_wav = str(manifest.get("output_wav") or output_dir / f"{safe_stem(case_id)}.wav")
        status = str(manifest.get("status") or ("ok" if Path(output_wav).exists() else "missing"))
        if status != "ok" and Path(output_wav).exists():
            status = "ok_after_rerun"
        if keep_status and status not in keep_status:
            continue
        exp_text = expected_text(val)
        source_lang = str(val.get("source_lang") or "")
        row = {
            "sample_id": case_id,
            "case_id": case_id,
            "run_id": args.run_id,
            "mode": val.get("mode"),
            "moss_codecvc_mode": val.get("mode"),
            "cell": val.get("cell"),
            "language": source_lang,
            "source_lang": source_lang,
            "ref_lang": val.get("ref_lang"),
            "source_audio": val.get("source_audio"),
            "timbre_ref_audio": val.get("timbre_ref_audio"),
            "target_audio": output_wav,
            "text": exp_text,
            "target_text": exp_text,
            "content_ref_text": exp_text,
            "content_ref_text_source": val.get("content_ref_text_source")
            or val.get("eval_text_source")
            or manifest.get("content_ref_text_source"),
            "source_content_text": manifest.get("source_content_text"),
            "source_content_text_key": manifest.get("source_content_text_key"),
            "content_asr_backend": val.get("content_asr_backend")
            or manifest.get("content_asr_backend"),
            "content_asr_model": val.get("content_asr_model")
            or manifest.get("content_asr_model"),
            "source_asr_backend": val.get("source_asr_backend")
            or manifest.get("source_asr_backend"),
            "source_asr_model": val.get("source_asr_model")
            or manifest.get("source_asr_model"),
            "asr_src_text": exp_text,
            "source_text": val.get("source_text"),
            "timbre_ref_text": val.get("timbre_ref_text"),
            "input_text": val.get("text"),
            "manifest_status": status,
            "returncode": manifest.get("returncode"),
            "elapsed_sec": manifest.get("elapsed_sec"),
        }
        rows.append(row)

    output_jsonl = Path(args.output_jsonl).expanduser() if args.output_jsonl else output_dir / f"{args.run_id}.generated_eval_input.jsonl"
    write_jsonl(output_jsonl, rows)
    print(f"[generated-eval] validation_rows={len(validation_rows)} manifest_rows={len(manifest_rows)} output_rows={len(rows)}")
    print(f"[generated-eval] output={output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

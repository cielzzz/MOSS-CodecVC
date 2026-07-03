#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import re
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDTTS_VALID = ROOT / "testset/validation/seedtts_vc_ver2_3_validation.jsonl"
DEFAULT_METADATA = ROOT / "testset/metadata.tsv"
DEFAULT_OUTPUT_DIR = ROOT / "testset/validation/ver2_3_switch_ablation"
FORCED_BADCASE_TEXT = "我待在公司里，真的只是加班工作，你为什么不相信我"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small fixed inference set for Ver2.3 switch ablations.")
    parser.add_argument("--seedtts-valid-jsonl", default=str(DEFAULT_SEEDTTS_VALID))
    parser.add_argument("--metadata-tsv", default=str(DEFAULT_METADATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-jsonl", default="")
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--no-text-per-bucket", type=int, default=10)
    parser.add_argument("--text-per-bucket", type=int, default=10)
    parser.add_argument("--forced-source-audio", default=str(ROOT / "testset/source/zh_a_000003_source.flac"))
    parser.add_argument("--forced-timbre-ref-audio", default=str(ROOT / "testset/timbre_ref/zh_a_000001_timbre.wav"))
    parser.add_argument("--forced-text", default=FORCED_BADCASE_TEXT)
    return parser.parse_args()


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


def abs_path(value: str | None) -> str:
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return str(path.resolve(strict=False))


def infer_mode(row: dict[str, Any]) -> str:
    mode = str(row.get("mode") or row.get("moss_codecvc_mode") or "").strip().lower()
    if mode in {"no_text", "text"}:
        return mode
    text = str(row.get("text") or "").strip()
    return "no_text" if text in {"", "<NO_TEXT>"} else "text"


def infer_lang(row: dict[str, Any], mode: str) -> str:
    raw = str(row.get("source_lang") if mode == "no_text" else row.get("language") or row.get("source_lang") or "").lower()
    if raw in {"zh", "zho", "cn", "chinese", "mandarin", "中文"}:
        return "zh"
    if raw in {"en", "eng", "english"}:
        return "en"
    text = str(row.get("content_ref_text") or row.get("text") or row.get("source_text") or "")
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return "unknown"


def text_length_score(text: str) -> int:
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
    latin_words = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+", text or ""))
    return cjk + latin_words


def normalize_row(row: dict[str, Any], *, bucket: str, index: int) -> dict[str, Any]:
    mode = infer_mode(row)
    text = str(row.get("text") or "").strip()
    content_ref_text = str(row.get("content_ref_text") or row.get("source_text") or text).strip()
    if mode == "no_text":
        text = "<NO_TEXT>"
    language = infer_lang(row, mode)
    return {
        "eval_id": f"{mode}_{bucket}_{index:03d}",
        "case_id": row.get("case_id") or row.get("sample_id") or f"{mode}_{bucket}_{index:03d}",
        "mode": mode,
        "bucket": bucket,
        "language": language,
        "source_audio": abs_path(row.get("source_audio")),
        "timbre_ref_audio": abs_path(row.get("timbre_ref_audio")),
        "text": text,
        "content_ref_text": content_ref_text,
        "source_text": row.get("source_text") or content_ref_text,
        "timbre_ref_text": row.get("timbre_ref_text") or "",
        "source_lang": row.get("source_lang") or language,
        "ref_lang": row.get("ref_lang") or "",
        "length_score": text_length_score(content_ref_text if mode == "no_text" else text),
        "source": "seedtts_valid",
    }


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            case_id = str(row.get("case_id") or "").strip()
            if case_id:
                rows[case_id] = row
    return rows


def bucket_select(rows: list[dict[str, Any]], *, mode: str, buckets: list[str], per_bucket: int, rng: random.Random) -> list[dict[str, Any]]:
    candidates = [row for row in rows if infer_mode(row) == mode and Path(abs_path(row.get("source_audio"))).exists() and Path(abs_path(row.get("timbre_ref_audio"))).exists()]
    candidates.sort(key=lambda row: (text_length_score(str(row.get("content_ref_text") or row.get("text") or row.get("source_text") or "")), str(row.get("case_id") or "")))
    if len(candidates) < per_bucket * len(buckets):
        raise RuntimeError(f"Not enough {mode} candidates: have={len(candidates)} need={per_bucket * len(buckets)}")

    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    n = len(candidates)
    for bucket_index, bucket in enumerate(buckets):
        start = int(round(n * bucket_index / len(buckets)))
        end = int(round(n * (bucket_index + 1) / len(buckets)))
        pool = candidates[start:end]
        rng.shuffle(pool)
        bucket_rows: list[dict[str, Any]] = []
        for row in pool:
            key = str(row.get("case_id") or row.get("source_audio") or "")
            if key in used:
                continue
            used.add(key)
            bucket_rows.append(row)
            if len(bucket_rows) >= per_bucket:
                break
        if len(bucket_rows) < per_bucket:
            raise RuntimeError(f"Not enough rows for {mode}:{bucket}; got={len(bucket_rows)} need={per_bucket}")
        selected.extend(normalize_row(row, bucket=bucket, index=len(selected)) for row in bucket_rows)
    return selected


def forced_badcase(args: argparse.Namespace, metadata: dict[str, dict[str, str]]) -> dict[str, Any]:
    meta = metadata.get("zh_a_000003", {})
    source_audio = abs_path(args.forced_source_audio or meta.get("source_audio"))
    timbre_ref_audio = abs_path(args.forced_timbre_ref_audio or meta.get("timbre_ref_audio"))
    for label, path in (("forced source", source_audio), ("forced timbre ref", timbre_ref_audio)):
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing {label} audio: {path}")
    text = str(args.forced_text or FORCED_BADCASE_TEXT).strip()
    return {
        "eval_id": "text_forced_company_overtime_000",
        "case_id": "forced_text_zh_company_overtime",
        "mode": "text",
        "bucket": "forced_badcase",
        "language": "zh",
        "source_audio": source_audio,
        "timbre_ref_audio": timbre_ref_audio,
        "text": text,
        "content_ref_text": text,
        "source_text": meta.get("source_text") or text,
        "timbre_ref_text": meta.get("timbre_text") or "",
        "source_lang": "zh",
        "ref_lang": "zh",
        "length_score": text_length_score(text),
        "source": "forced_local_badcase",
    }


def main() -> int:
    args = parse_args()
    rng = random.Random(int(args.seed))
    valid_path = Path(args.seedtts_valid_jsonl).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_jsonl = Path(args.output_jsonl).expanduser() if args.output_jsonl else output_dir / "moss_codecvc_ver2_3_switch_fixed_infer_51.jsonl"
    rows = list(iter_jsonl(valid_path))
    selected = []
    selected.extend(
        bucket_select(
            rows,
            mode="no_text",
            buckets=["short", "medium", "long"],
            per_bucket=int(args.no_text_per_bucket),
            rng=rng,
        )
    )
    selected.extend(
        bucket_select(
            rows,
            mode="text",
            buckets=["short", "medium"],
            per_bucket=int(args.text_per_bucket),
            rng=rng,
        )
    )
    selected.append(forced_badcase(args, load_metadata(Path(args.metadata_tsv).expanduser())))

    write_jsonl(output_jsonl, selected)
    summary = {
        "output_jsonl": str(output_jsonl),
        "rows": len(selected),
        "counts": dict(Counter(f"{row['mode']}:{row['bucket']}" for row in selected)),
        "forced_case": "forced_text_zh_company_overtime",
        "seed": int(args.seed),
    }
    summary_path = output_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

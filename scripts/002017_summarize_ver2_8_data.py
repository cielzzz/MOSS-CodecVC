#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NO_TEXT_WAVLM = (
    ROOT
    / "trainset/zh45w_en22w_no_text/sft/"
    / "moss_codecvc_sft.zh45w_en22w_no_text.with_light_ecapa_spk.with_prosody.with_asr_filter.with_wavlm_bnf.with_spm_content_tokens.ctc_clean.jsonl"
)
DEFAULT_TEXT_WAVLM = (
    ROOT
    / "trainset/zh3w_en3w_text_prosody_independent_timbre/sft/"
    / "moss_codecvc_sft.zh3w_en3w_text_prosody_independent_timbre.with_light_ecapa_spk.with_prosody.with_target_asr.with_content_tokens.with_target_wavlm_bnf.with_spm_content_tokens.ctc_clean.jsonl"
)
DEFAULT_NO_TEXT_FEATURE_ROOT = ROOT / "trainset/zh45w_en22w_no_text/semantic_features/wavlm_bnf"
DEFAULT_TEXT_FEATURE_ROOT = ROOT / "trainset/zh3w_en3w_text_prosody_independent_timbre/semantic_features/wavlm_bnf"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize Ver2.8 WavLM-BNF data before training submission.")
    ap.add_argument("--prepared-dir", default=str(ROOT / "trainset/ver2_8_prepared"))
    ap.add_argument("--no-text-wavlm-jsonl", default=str(DEFAULT_NO_TEXT_WAVLM))
    ap.add_argument("--text-wavlm-jsonl", default=str(DEFAULT_TEXT_WAVLM))
    ap.add_argument("--no-text-feature-root", default=str(DEFAULT_NO_TEXT_FEATURE_ROOT))
    ap.add_argument("--text-feature-root", default=str(DEFAULT_TEXT_FEATURE_ROOT))
    ap.add_argument("--text-repeat", type=int, default=5)
    ap.add_argument("--output-json", default="")
    ap.add_argument("--skip-feature-scan", action="store_true")
    return ap.parse_args()


def count_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    count = 0
    with path.open("rb") as handle:
        for _ in handle:
            count += 1
    return count


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def scan_tree(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "files": 0, "bytes": 0, "by_subdir": {}}
    files = 0
    total_bytes = 0
    by_subdir: dict[str, dict[str, int]] = {}
    for root, _, filenames in os.walk(path):
        root_path = Path(root)
        rel = root_path.relative_to(path)
        top = str(rel.parts[0]) if rel.parts else "."
        item = by_subdir.setdefault(top, {"files": 0, "bytes": 0})
        for name in filenames:
            file_path = root_path / name
            try:
                size = file_path.stat().st_size
            except OSError:
                size = 0
            files += 1
            total_bytes += size
            item["files"] += 1
            item["bytes"] += size
    return {"exists": True, "files": files, "bytes": total_bytes, "by_subdir": by_subdir}


def fmt_count(value: int | None) -> str:
    return "missing" if value is None else str(value)


def fmt_bytes(value: int | None) -> str:
    if value is None:
        return "missing"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.1f}{unit}" if unit != "B" else f"{int(amount)}B"
        amount /= 1024.0
    return f"{value}B"


def split_summary(summary: dict[str, Any] | None, mode: str) -> dict[str, Any]:
    if not summary:
        return {}
    return dict(((summary.get("split") or {}).get(mode) or {}))


def clean_summary(summary: dict[str, Any] | None, mode: str) -> dict[str, Any]:
    if not summary:
        return {}
    return dict(((summary.get("clean") or {}).get(mode) or {}))


def main() -> int:
    args = parse_args()
    prepared_dir = Path(args.prepared_dir).expanduser()
    no_text_train = prepared_dir / "no_text.train.jsonl"
    no_text_valid = prepared_dir / "no_text.valid.jsonl"
    text_train = prepared_dir / "text.train.jsonl"
    text_valid = prepared_dir / "text.valid.jsonl"
    no_text_rejects = prepared_dir / "temp/asr_rejects.no_text.jsonl"
    text_rejects = prepared_dir / "temp/asr_rejects.text.jsonl"
    summary_json = prepared_dir / "summary.json"

    counts = {
        "raw_wavlm": {
            "no_text": count_lines(Path(args.no_text_wavlm_jsonl).expanduser()),
            "text": count_lines(Path(args.text_wavlm_jsonl).expanduser()),
        },
        "prepared": {
            "no_text_train": count_lines(no_text_train),
            "no_text_valid": count_lines(no_text_valid),
            "text_train": count_lines(text_train),
            "text_valid": count_lines(text_valid),
            "no_text_rejects": count_lines(no_text_rejects),
            "text_rejects": count_lines(text_rejects),
        },
    }
    no_text_effective_train_rows = None
    text_effective_train_rows = None
    effective_train_rows = None
    text_no_text_balance_ratio = None
    if counts["prepared"]["no_text_train"] is not None and counts["prepared"]["text_train"] is not None:
        no_text_effective_train_rows = int(counts["prepared"]["no_text_train"])
        text_effective_train_rows = int(args.text_repeat) * int(counts["prepared"]["text_train"])
        effective_train_rows = no_text_effective_train_rows + text_effective_train_rows
        if no_text_effective_train_rows > 0:
            text_no_text_balance_ratio = text_effective_train_rows / float(no_text_effective_train_rows)

    summary = load_json(summary_json)
    feature_scan = {}
    if not args.skip_feature_scan:
        feature_scan = {
            "no_text": scan_tree(Path(args.no_text_feature_root).expanduser()),
            "text": scan_tree(Path(args.text_feature_root).expanduser()),
        }

    payload = {
        "prepared_dir": str(prepared_dir),
        "summary_json": str(summary_json),
        "text_repeat": int(args.text_repeat),
        "text_extra_copies": max(0, int(args.text_repeat) - 1),
        "counts": counts,
        "no_text_effective_train_rows": no_text_effective_train_rows,
        "text_effective_train_rows": text_effective_train_rows,
        "effective_train_rows_after_repeat": effective_train_rows,
        "text_no_text_balance_ratio": text_no_text_balance_ratio,
        "summary_status": (summary or {}).get("status"),
        "clean": {
            "no_text": clean_summary(summary, "no_text"),
            "text": clean_summary(summary, "text"),
        },
        "split": {
            "no_text": split_summary(summary, "no_text"),
            "text": split_summary(summary, "text"),
        },
        "features": feature_scan,
    }

    print("[ver2.8-data] prepared_dir=" + str(prepared_dir))
    print(
        "[ver2.8-data] raw_wavlm_rows "
        f"no_text={fmt_count(counts['raw_wavlm']['no_text'])} "
        f"text={fmt_count(counts['raw_wavlm']['text'])}"
    )
    balance_text = (
        f"balance_ratio={text_no_text_balance_ratio:.4f} "
        if text_no_text_balance_ratio is not None
        else ""
    )
    print(
        "[ver2.8-data] train_rows "
        f"no_text={fmt_count(counts['prepared']['no_text_train'])} "
        f"text={fmt_count(counts['prepared']['text_train'])} "
        f"text_repeat={int(args.text_repeat)} "
        f"text_extra_copies={max(0, int(args.text_repeat) - 1)} "
        f"text_effective={fmt_count(text_effective_train_rows)} "
        f"{balance_text}"
        f"effective={fmt_count(effective_train_rows)}"
    )
    print(
        "[ver2.8-data] valid_rows "
        f"no_text={fmt_count(counts['prepared']['no_text_valid'])} "
        f"text={fmt_count(counts['prepared']['text_valid'])}"
    )
    print(
        "[ver2.8-data] rejects "
        f"no_text={fmt_count(counts['prepared']['no_text_rejects'])} "
        f"text={fmt_count(counts['prepared']['text_rejects'])}"
    )
    for mode in ("no_text", "text"):
        clean = payload["clean"][mode]
        split = payload["split"][mode]
        if clean or split:
            print(
                f"[ver2.8-data] summary.{mode} "
                f"kept={clean.get('kept', 'n/a')} rejected={clean.get('rejected', 'n/a')} "
                f"languages={clean.get('languages', {})} "
                f"train={split.get('train_rows', 'n/a')} valid={split.get('valid_rows', 'n/a')}"
            )
    for mode, item in feature_scan.items():
        by_subdir = item.get("by_subdir") or {}
        subdir_text = ", ".join(
            f"{key}:{value.get('files', 0)}" for key, value in sorted(by_subdir.items()) if key != "."
        )
        print(
            f"[ver2.8-data] wavlm_features.{mode} "
            f"files={item.get('files', 0)} size={fmt_bytes(item.get('bytes', 0))} "
            f"subdirs={{{subdir_text}}}"
        )

    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ver2.8-data] output_json={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

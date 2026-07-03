#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_JSONL = ROOT / "testset/validation/seedtts_vc_ver2_3_validation.jsonl"
DEFAULT_MODEL_PATH = (
    ROOT
    / "outputs/lora_runs/ver2_3_ctc_clean_textrep5_spm_lora_r16_a32_gbs64/final"
)
DEFAULT_RUN_SCRIPT = ROOT / "scripts/003003_run_moss_codecvc_infer.sh"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run MOSS-CodecVC inference on the fixed Seed-TTS validation JSONL."
    )
    ap.add_argument("--validation-jsonl", default=str(DEFAULT_VALIDATION_JSONL))
    ap.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    ap.add_argument("--run-script", default=str(DEFAULT_RUN_SCRIPT))
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "testset/outputs/ver2_3_ctc_clean_seedtts_valid_smoke"),
    )
    ap.add_argument("--mode", choices=("all", "no_text", "text"), default="all")
    ap.add_argument(
        "--per-mode",
        type=int,
        default=0,
        help="Limit selected rows per mode. 0 means no per-mode limit.",
    )
    ap.add_argument(
        "--per-cell",
        type=int,
        default=0,
        help="Limit selected rows per mode:cell. 0 means no per-cell limit.",
    )
    ap.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Global selected row limit after filtering. 0 means no limit.",
    )
    ap.add_argument("--num-shards", type=int, default=1, help="Split selected rows into N shards.")
    ap.add_argument("--shard-index", type=int, default=0, help="Run shard index in [0, N).")
    ap.add_argument("--case-id", action="append", default=[], help="Run one explicit case id.")
    ap.add_argument(
        "--manifest-jsonl",
        default="",
        help="Optional manifest path. Defaults to <output-dir>/manifest.jsonl.",
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--python", default="", help="Optional PYTHON override for the infer shell script.")
    ap.add_argument("--max-new-tokens", default="", help="Optional MAX_NEW_TOKENS override.")
    ap.add_argument("--debug-generation-structure", action="store_true")
    ap.add_argument("--save-codec-intermediates", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.num_shards < 1:
        ap.error("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        ap.error("--shard-index must be in [0, --num-shards)")
    return args


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


def safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return stem[:180] or "case"


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    wanted_case_ids = set(args.case_id)
    selected: list[dict[str, Any]] = []
    mode_counts: Counter[str] = Counter()
    cell_counts: Counter[str] = Counter()

    for row in rows:
        case_id = str(row.get("case_id") or "")
        mode = str(row.get("mode") or "")
        cell = str(row.get("cell") or "")
        if wanted_case_ids and case_id not in wanted_case_ids:
            continue
        if args.mode != "all" and mode != args.mode:
            continue
        if mode not in {"no_text", "text"}:
            continue
        cell_key = f"{mode}:{cell}"
        if args.per_mode > 0 and mode_counts[mode] >= args.per_mode:
            continue
        if args.per_cell > 0 and cell_counts[cell_key] >= args.per_cell:
            continue

        selected.append(row)
        mode_counts[mode] += 1
        cell_counts[cell_key] += 1
        if args.max_cases > 0 and len(selected) >= args.max_cases:
            break

    if args.num_shards > 1:
        selected = [
            row for idx, row in enumerate(selected) if idx % args.num_shards == args.shard_index
        ]

    return selected


def required_text(row: dict[str, Any]) -> str:
    text = str(row.get("text") or "").strip()
    if text and text != "<NO_TEXT>":
        return text
    return str(row.get("content_ref_text") or row.get("source_text") or "").strip()


def source_content_text_with_key(row: dict[str, Any]) -> tuple[str, str]:
    for key in ("source_text", "content_ref_text", "asr_src_text", "source_asr_text"):
        text = str(row.get(key) or "").strip()
        if text and text != "<NO_TEXT>":
            return text, key
    return "", ""


def source_content_text(row: dict[str, Any]) -> str:
    text, _key = source_content_text_with_key(row)
    return text


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_case(row: dict[str, Any], args: argparse.Namespace, output_dir: Path, manifest: Path) -> int:
    case_id = str(row.get("case_id") or "")
    mode = str(row.get("mode") or "")
    output_wav = output_dir / f"{safe_stem(case_id)}.wav"
    start = time.time()

    content_text, content_text_key = source_content_text_with_key(row)
    manifest_row = {
        "case_id": case_id,
        "mode": mode,
        "cell": row.get("cell"),
        "source_audio": row.get("source_audio"),
        "timbre_ref_audio": row.get("timbre_ref_audio"),
        "text": row.get("text"),
        "content_ref_text": row.get("content_ref_text"),
        "source_content_text": content_text,
        "source_content_text_key": content_text_key,
        "content_ref_text_source": row.get("content_ref_text_source") or row.get("eval_text_source"),
        "content_asr_backend": row.get("content_asr_backend"),
        "content_asr_model": row.get("content_asr_model"),
        "source_asr_backend": row.get("source_asr_backend"),
        "source_asr_model": row.get("source_asr_model"),
        "output_wav": str(output_wav),
    }

    if output_wav.exists() and not args.overwrite:
        manifest_row.update({"status": "skipped_exists", "elapsed_sec": 0.0})
        append_jsonl(manifest, manifest_row)
        print(f"[valid] skip existing {case_id} -> {output_wav}", flush=True)
        return 0

    env = os.environ.copy()
    env.update(
        {
            "MODEL_PATH": str(Path(args.model_path).expanduser()),
            "MODE": mode,
            "SOURCE_AUDIO": str(row.get("source_audio") or ""),
            "TIMBRE_REF_AUDIO": str(row.get("timbre_ref_audio") or ""),
            "OUTPUT_DIR": str(output_dir),
            "OUTPUT_WAV": str(output_wav),
            "DEVICE": args.device,
            "DEBUG_GENERATION_STRUCTURE": "1" if args.debug_generation_structure else "0",
            "SAVE_CODEC_INTERMEDIATES": "1" if args.save_codec_intermediates else "0",
        }
    )
    if args.python:
        env["PYTHON"] = args.python
    if args.max_new_tokens:
        env["MAX_NEW_TOKENS"] = args.max_new_tokens
    if mode == "text":
        env["TEXT"] = required_text(row)
    else:
        env.pop("TEXT", None)
    if content_text:
        env["SOURCE_CONTENT_TEXT"] = content_text
    else:
        env.pop("SOURCE_CONTENT_TEXT", None)

    print(f"[valid] run {case_id} mode={mode}", flush=True)
    proc = subprocess.run(["sh", str(Path(args.run_script).expanduser())], env=env)
    elapsed = round(time.time() - start, 3)
    status = "ok" if proc.returncode == 0 and output_wav.exists() else "failed"
    manifest_row.update(
        {
            "status": status,
            "returncode": proc.returncode,
            "elapsed_sec": elapsed,
            "output_exists": output_wav.exists(),
        }
    )
    append_jsonl(manifest, manifest_row)
    print(
        f"[valid] done {case_id} status={status} returncode={proc.returncode} elapsed={elapsed}s",
        flush=True,
    )
    return proc.returncode


def main() -> int:
    args = parse_args()
    validation_jsonl = Path(args.validation_jsonl).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    manifest = Path(args.manifest_jsonl).expanduser() if args.manifest_jsonl else output_dir / "manifest.jsonl"
    rows = list(iter_jsonl(validation_jsonl))
    selected = select_rows(rows, args)
    if not selected:
        print("[valid] no rows selected", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[valid] selected={len(selected)} shard={args.shard_index}/{args.num_shards} "
        f"output_dir={output_dir} manifest={manifest}",
        flush=True,
    )
    for row in selected:
        print(
            f"[valid] selected case_id={row.get('case_id')} mode={row.get('mode')} cell={row.get('cell')}",
            flush=True,
        )

    if args.dry_run:
        return 0

    failures = 0
    for row in selected:
        ret = run_case(row, args, output_dir, manifest)
        if ret != 0:
            failures += 1
            if args.fail_fast:
                break
    print(f"[valid] complete total={len(selected)} failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

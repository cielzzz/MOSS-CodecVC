#!/usr/bin/env python
from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
import json
import random
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_JSONL = ROOT / "testset/validation/seedtts_vc_ver2_3_validation.jsonl"
DEFAULT_NO_TEXT_TRAIN_JSONL = (
    ROOT
    / "trainset/zh45w_en22w_no_text/sft/"
    / "moss_codecvc_sft.zh45w_en22w_no_text.with_light_ecapa_spk.with_prosody.with_asr_filter.with_hubert.with_spm_content_tokens.ctc_clean.jsonl"
)
DEFAULT_TEXT_TRAIN_JSONL = (
    ROOT
    / "trainset/zh3w_en3w_text_prosody_independent_timbre/sft/"
    / "moss_codecvc_sft.zh3w_en3w_text_prosody_independent_timbre.with_light_ecapa_spk.with_prosody.with_target_asr.with_content_tokens.with_target_hubert.with_spm_content_tokens.ctc_clean.jsonl"
)
DEFAULT_TRAIN_JSONL_SPEC = f"{DEFAULT_NO_TEXT_TRAIN_JSONL}::repeat=1,{DEFAULT_TEXT_TRAIN_JSONL}::repeat=1"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build fixed Ver2.3 debug benchmark/loss-valid splits.")
    ap.add_argument("--benchmark-jsonl", default=str(DEFAULT_BENCHMARK_JSONL))
    ap.add_argument("--train-jsonl-spec", default=DEFAULT_TRAIN_JSONL_SPEC)
    ap.add_argument("--output-dir", default=str(ROOT / "testset/validation/ver2_3_debug"))
    ap.add_argument("--seed", type=int, default=20260630)
    ap.add_argument("--benchmark-per-mode-cell", type=int, default=8)
    ap.add_argument("--loss-valid-total", type=int, default=160)
    ap.add_argument("--loss-valid-no-text", type=int, default=0)
    ap.add_argument("--loss-valid-text", type=int, default=0)
    ap.add_argument("--max-random-probes-per-source", type=int, default=8000)
    ap.add_argument("--max-repeat-score", type=float, default=0.30)
    ap.add_argument("--zh-cer-threshold", type=float, default=0.15)
    ap.add_argument("--en-wer-threshold", type=float, default=0.20)
    ap.add_argument("--min-duration-ratio", type=float, default=0.50)
    ap.add_argument("--max-duration-ratio", type=float, default=2.00)
    ap.add_argument("--tiny-sizes", default="32,128")
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


def parse_train_jsonl_paths(spec: str) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for line in str(spec or "").splitlines():
        for chunk in line.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            path = Path(chunk.split("::", 1)[0]).expanduser()
            key = str(path)
            if key not in seen:
                paths.append(path)
                seen.add(key)
    if not paths:
        raise ValueError("--train-jsonl-spec produced no paths")
    return paths


def load_offsets(path: Path) -> array:
    offsets_path = Path(str(path) + ".offsets.u64")
    offsets = array("Q")
    if offsets_path.exists():
        with offsets_path.open("rb") as handle:
            rows = offsets_path.stat().st_size // offsets.itemsize
            offsets.fromfile(handle, rows)
        return offsets
    with path.open("rb") as handle:
        while True:
            pos = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.strip():
                offsets.append(pos)
    return offsets


def read_row_at(handle, offset: int) -> dict[str, Any]:
    handle.seek(int(offset))
    line = handle.readline()
    if not line:
        raise ValueError(f"empty line at offset {offset}")
    return json.loads(line.decode("utf-8"))


def nested_get(row: dict[str, Any], key: str) -> Any:
    if key in row and row[key] not in (None, ""):
        return row[key]
    meta = row.get("moss_codecvc_meta")
    if isinstance(meta, dict) and meta.get(key) not in (None, ""):
        return meta[key]
    return None


def bool_value(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "keep", "kept"}


def float_value(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def list_nonempty(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None


def infer_mode(row: dict[str, Any]) -> str:
    mode = str(nested_get(row, "moss_codecvc_mode") or nested_get(row, "mode") or "").strip().lower()
    if mode in {"no_text", "text"}:
        return mode
    text = str(nested_get(row, "text") or "").strip()
    return "no_text" if text in {"", "<NO_TEXT>"} else "text"


def has_zh(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def infer_language(row: dict[str, Any]) -> str:
    raw = str(nested_get(row, "language") or nested_get(row, "lang") or "").strip().lower()
    if raw in {"zh", "zho", "cn", "chinese", "mandarin", "中文"}:
        return "zh"
    if raw in {"en", "eng", "english"}:
        return "en"
    text = " ".join(
        str(nested_get(row, key) or "")
        for key in ("content_ref_text", "asr_src_text", "asr_tgt_text", "source_text", "target_text", "text")
    )
    if has_zh(text):
        return "zh"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return "other"


def passes_quality(row: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    quality = str(nested_get(row, "quality") or "").strip().lower()
    if quality and quality not in {"high", "good", "keep", "kept", "ok"}:
        return False, f"quality={quality}"
    if not bool_value(nested_get(row, "content_keep"), default=False):
        return False, "content_keep=false"
    if not bool_value(nested_get(row, "content_token_keep"), default=False):
        return False, "content_token_keep=false"
    token_len = float_value(nested_get(row, "content_token_length"))
    token_ids = nested_get(row, "content_token_ids") or nested_get(row, "content_ref_token_ids")
    if (token_len is None or token_len <= 0) and not list_nonempty(token_ids):
        return False, "missing_content_tokens"
    vocab_size = float_value(nested_get(row, "content_ctc_vocab_size"))
    if vocab_size is not None and vocab_size <= 1:
        return False, f"bad_vocab={vocab_size}"
    if not list_nonempty(nested_get(row, "audio_codes")) and not list_nonempty(nested_get(row, "audio_codes_path")):
        return False, "missing_audio_codes"
    if not list_nonempty(nested_get(row, "reference_audio_codes")) and not list_nonempty(nested_get(row, "reference_audio_codes_path")):
        return False, "missing_reference_audio_codes"
    repeat = float_value(nested_get(row, "repeat_score"))
    if repeat is not None and repeat > float(args.max_repeat_score):
        return False, f"repeat_score={repeat:.4f}"
    lang = infer_language(row)
    cer = float_value(nested_get(row, "cer_tgt"))
    wer = float_value(nested_get(row, "wer_tgt"))
    if lang == "zh" and cer is not None and cer > float(args.zh_cer_threshold):
        return False, f"cer_tgt={cer:.4f}"
    if lang == "en" and wer is not None and wer > float(args.en_wer_threshold):
        return False, f"wer_tgt={wer:.4f}"
    duration_ratio = float_value(nested_get(row, "duration_ratio_tgt_src"))
    if duration_ratio is not None:
        if duration_ratio < float(args.min_duration_ratio) or duration_ratio > float(args.max_duration_ratio):
            return False, f"duration_ratio={duration_ratio:.4f}"
    return True, "keep"


def make_targets(args: argparse.Namespace) -> dict[tuple[str, str], int]:
    if args.loss_valid_no_text > 0 or args.loss_valid_text > 0:
        no_text = int(args.loss_valid_no_text)
        text = int(args.loss_valid_text)
    else:
        no_text = int(args.loss_valid_total) // 2
        text = int(args.loss_valid_total) - no_text
    targets: dict[tuple[str, str], int] = {}
    for mode, total in (("no_text", no_text), ("text", text)):
        zh = total // 2
        en = total - zh
        targets[(mode, "zh")] = zh
        targets[(mode, "en")] = en
    return targets


def targets_complete(selected: dict[tuple[str, str], list[dict[str, Any]]], targets: dict[tuple[str, str], int]) -> bool:
    return all(len(selected[key]) >= limit for key, limit in targets.items())


def build_benchmark_subset(args: argparse.Namespace, rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = list(iter_jsonl(Path(args.benchmark_jsonl).expanduser()))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        mode = str(row.get("mode") or "").strip()
        cell = str(row.get("cell") or "unknown").strip() or "unknown"
        groups[(mode, cell)].append(row)
    selected: list[dict[str, Any]] = []
    group_counts: dict[str, int] = {}
    for key in sorted(groups):
        group = list(groups[key])
        rng.shuffle(group)
        take = min(int(args.benchmark_per_mode_cell), len(group))
        selected.extend(group[:take])
        group_counts[f"{key[0]}:{key[1]}"] = take
    selected.sort(key=lambda row: (str(row.get("mode") or ""), str(row.get("cell") or ""), str(row.get("case_id") or "")))
    return selected, {"source_rows": len(rows), "selected_rows": len(selected), "group_counts": group_counts}


def build_loss_valid(args: argparse.Namespace, rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets = make_targets(args)
    selected: dict[tuple[str, str], list[dict[str, Any]]] = {key: [] for key in targets}
    selected_ids: set[str] = set()
    reasons: Counter[str] = Counter()
    scanned: Counter[str] = Counter()
    kept_seen: Counter[str] = Counter()
    paths = parse_train_jsonl_paths(args.train_jsonl_spec)

    for path in paths:
        path = path.expanduser()
        offsets = load_offsets(path)
        probe_count = min(len(offsets), int(args.max_random_probes_per_source))
        if probe_count <= 0:
            continue
        indices = rng.sample(range(len(offsets)), probe_count)
        with path.open("rb") as handle:
            for index in indices:
                row = read_row_at(handle, int(offsets[index]))
                source_key = path.name
                scanned[source_key] += 1
                ok, reason = passes_quality(row, args)
                if not ok:
                    reasons[reason] += 1
                    continue
                mode = infer_mode(row)
                lang = infer_language(row)
                kept_seen[f"{mode}:{lang}"] += 1
                key = (mode, lang)
                if key not in targets:
                    reasons[f"unwanted_bucket={mode}:{lang}"] += 1
                    continue
                if len(selected[key]) >= targets[key]:
                    continue
                row_id = str(nested_get(row, "sample_id") or nested_get(row, "utt_id") or nested_get(row, "target_audio") or id(row))
                if row_id in selected_ids:
                    continue
                selected[key].append(row)
                selected_ids.add(row_id)
                if targets_complete(selected, targets):
                    break
        if targets_complete(selected, targets):
            break

    rows: list[dict[str, Any]] = []
    shortages: dict[str, dict[str, int]] = {}
    for key, target in targets.items():
        bucket_rows = selected[key]
        rows.extend(bucket_rows)
        if len(bucket_rows) < target:
            shortages[f"{key[0]}:{key[1]}"] = {"target": target, "selected": len(bucket_rows)}
    rng.shuffle(rows)
    summary = {
        "targets": {f"{mode}:{lang}": count for (mode, lang), count in targets.items()},
        "selected_rows": len(rows),
        "selected_counts": dict(Counter(f"{infer_mode(row)}:{infer_language(row)}" for row in rows)),
        "shortages": shortages,
        "scanned_by_source": dict(scanned),
        "kept_seen_by_bucket": dict(kept_seen),
        "reject_reasons_top20": dict(reasons.most_common(20)),
        "train_jsonl_paths": [str(path) for path in paths],
    }
    return rows, summary


def select_tiny(rows: list[dict[str, Any]], size: int, rng: random.Random) -> list[dict[str, Any]]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[infer_mode(row)].append(row)
    for bucket in by_mode.values():
        rng.shuffle(bucket)
    no_text_target = size // 2
    text_target = size - no_text_target
    selected = by_mode.get("no_text", [])[:no_text_target] + by_mode.get("text", [])[:text_target]
    if len(selected) < size:
        used = {id(row) for row in selected}
        rest = [row for row in rows if id(row) not in used]
        rng.shuffle(rest)
        selected.extend(rest[: size - len(selected)])
    rng.shuffle(selected)
    return selected[:size]


def select_tiny_mode(rows: list[dict[str, Any]], size: int, mode: str, rng: random.Random) -> list[dict[str, Any]]:
    selected = [row for row in rows if infer_mode(row) == mode]
    rng.shuffle(selected)
    return selected[:size]


def build_tiny_mode_from_sources(args: argparse.Namespace, rng: random.Random, *, size: int, mode: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    paths = parse_train_jsonl_paths(args.train_jsonl_spec)
    for path in paths:
        path = path.expanduser()
        offsets = load_offsets(path)
        probe_count = min(len(offsets), max(int(args.max_random_probes_per_source), int(size) * 8))
        if probe_count <= 0:
            continue
        indices = rng.sample(range(len(offsets)), probe_count)
        with path.open("rb") as handle:
            for index in indices:
                row = read_row_at(handle, int(offsets[index]))
                if infer_mode(row) != mode:
                    continue
                ok, _ = passes_quality(row, args)
                if not ok:
                    continue
                row_id = str(nested_get(row, "sample_id") or nested_get(row, "utt_id") or nested_get(row, "target_audio") or id(row))
                if row_id in selected_ids:
                    continue
                selected.append(row)
                selected_ids.add(row_id)
                if len(selected) >= int(size):
                    rng.shuffle(selected)
                    return selected[:size]
    rng.shuffle(selected)
    return selected[:size]


def main() -> int:
    args = parse_args()
    rng = random.Random(int(args.seed))
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_rows, benchmark_summary = build_benchmark_subset(args, rng)
    benchmark_path = output_dir / f"seedtts_vc_ver2_3_benchmark_core{len(benchmark_rows)}.jsonl"
    write_jsonl(benchmark_path, benchmark_rows)

    loss_rows, loss_summary = build_loss_valid(args, rng)
    loss_path = output_dir / f"moss_codecvc_ver2_3_loss_valid_{len(loss_rows)}.jsonl"
    write_jsonl(loss_path, loss_rows)
    write_jsonl(output_dir / f"moss_codecvc_ver2_3_loss_valid_{len(loss_rows)}.no_text.jsonl", [row for row in loss_rows if infer_mode(row) == "no_text"])
    write_jsonl(output_dir / f"moss_codecvc_ver2_3_loss_valid_{len(loss_rows)}.text.jsonl", [row for row in loss_rows if infer_mode(row) == "text"])

    tiny_outputs: dict[str, str] = {}
    for item in str(args.tiny_sizes or "").split(","):
        item = item.strip()
        if not item:
            continue
        size = int(item)
        if size <= 0:
            continue
        tiny_rows = select_tiny(loss_rows, size, rng)
        tiny_path = output_dir / f"moss_codecvc_ver2_3_tiny_overfit_{len(tiny_rows)}.jsonl"
        write_jsonl(tiny_path, tiny_rows)
        tiny_outputs[str(len(tiny_rows))] = str(tiny_path)

        tiny_no_text_rows = build_tiny_mode_from_sources(args, rng, size=size, mode="no_text")
        tiny_no_text_path = output_dir / f"moss_codecvc_ver2_3_tiny_overfit_no_text_{len(tiny_no_text_rows)}.jsonl"
        write_jsonl(tiny_no_text_path, tiny_no_text_rows)
        tiny_outputs[f"no_text_{len(tiny_no_text_rows)}"] = str(tiny_no_text_path)

    summary = {
        "seed": int(args.seed),
        "benchmark": {"path": str(benchmark_path), **benchmark_summary},
        "loss_valid": {"path": str(loss_path), **loss_summary},
        "tiny_overfit": tiny_outputs,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[valid-split] benchmark={benchmark_path} rows={len(benchmark_rows)}")
    print(f"[valid-split] loss_valid={loss_path} rows={len(loss_rows)} counts={loss_summary['selected_counts']}")
    if loss_summary["shortages"]:
        print(f"[valid-split] WARNING shortages={loss_summary['shortages']}")
    print(f"[valid-split] summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

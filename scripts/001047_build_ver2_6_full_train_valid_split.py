#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_NO_TEXT_JSONL = (
    ROOT
    / "trainset/zh45w_en22w_no_text/sft/"
    / "moss_codecvc_sft.zh45w_en22w_no_text.with_light_ecapa_spk.with_prosody.with_asr_filter.with_hubert.with_spm_content_tokens.jsonl"
)
DEFAULT_TEXT_OLD_JSONL = (
    ROOT
    / "trainset/zh3w_en3w_text_prosody_independent_timbre/sft/"
    / "moss_codecvc_sft.zh3w_en3w_text_prosody_independent_timbre.with_light_ecapa_spk.with_prosody.with_target_asr.with_content_tokens.with_target_hubert.with_spm_content_tokens.jsonl"
)
DEFAULT_TEXT_NEW_JSONL = (
    ROOT
    / "trainset/zh11w_en11w_0005_0015_vcdata_first_text_prosody/sft/"
    / "moss_codecvc_sft.zh11w_en11w_0005_0015_vcdata_first_text_prosody.with_light_ecapa_spk.with_prosody.with_target_asr.with_content_tokens.with_target_hubert.with_spm_content_tokens.jsonl"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build deterministic Ver2.6 full-data train/validation JSONL splits.")
    ap.add_argument("--no-text-jsonl", default=str(DEFAULT_NO_TEXT_JSONL))
    ap.add_argument("--text-old-jsonl", default=str(DEFAULT_TEXT_OLD_JSONL))
    ap.add_argument("--text-new-jsonl", default=str(DEFAULT_TEXT_NEW_JSONL))
    ap.add_argument("--output-dir", default=str(ROOT / "testset/validation/ver2_6"))
    ap.add_argument("--seed", type=int, default=20260701)
    ap.add_argument("--no-text-valid-size", type=int, default=5000)
    ap.add_argument("--text-old-valid-size", type=int, default=2000)
    ap.add_argument("--text-new-valid-size", type=int, default=5000)
    ap.add_argument("--progress-every", type=int, default=50000)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def nested_get(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if value not in (None, ""):
        return value
    meta = row.get("moss_codecvc_meta")
    if isinstance(meta, dict):
        value = meta.get(key)
        if value not in (None, ""):
            return value
    return None


def bool_value(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "keep", "kept"}


def infer_mode(row: dict[str, Any], dataset_name: str) -> str:
    mode = str(nested_get(row, "moss_codecvc_mode") or nested_get(row, "mode") or "").strip().lower()
    if mode in {"no_text", "text"}:
        return mode
    if dataset_name == "no_text":
        return "no_text"
    if dataset_name in {"text_old", "text_new"}:
        return "text"
    text = str(nested_get(row, "text") or "").strip()
    return "no_text" if text in {"", "<NO_TEXT>"} else "text"


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
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return "zh"
    if any(("a" <= char.lower() <= "z") for char in text):
        return "en"
    return "other"


def stable_record_key(row: dict[str, Any]) -> str:
    for key in ("sample_id", "id", "uid", "utt_id"):
        value = nested_get(row, key)
        if value not in (None, ""):
            return str(value)
    payload = {
        key: nested_get(row, key)
        for key in (
            "source_audio",
            "source_wav",
            "source_audio_path",
            "timbre_ref_audio",
            "timbre_audio",
            "target_audio",
            "target_wav",
            "text",
            "content_ref_text",
        )
        if nested_get(row, key) not in (None, "")
    }
    if payload:
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def hash_score(seed: int, dataset_name: str, record_key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{dataset_name}:{record_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def train_output_path(path: Path, seed: int) -> Path:
    if path.name.endswith(".jsonl"):
        return path.with_name(path.name[: -len(".jsonl")] + f".train_seed{seed}.jsonl")
    return path.with_name(path.name + f".train_seed{seed}.jsonl")


def valid_output_path(output_dir: Path, dataset_name: str, seed: int) -> Path:
    if dataset_name == "no_text":
        return output_dir / f"no_text_full_seed{seed}.jsonl"
    if dataset_name == "text_old":
        return output_dir / f"text_old_seed{seed}.jsonl"
    if dataset_name == "text_new":
        return output_dir / f"text_new_seed{seed}.jsonl"
    return output_dir / f"{dataset_name}_seed{seed}.jsonl"


def read_json_line(path: Path, line_no: int, line: str) -> dict[str, Any]:
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc


def select_validation_keys(
    *,
    dataset_name: str,
    path: Path,
    valid_size: int,
    seed: int,
    progress_every: int,
) -> tuple[set[str], dict[str, Any]]:
    if valid_size <= 0:
        return set(), {}
    heap: list[tuple[int, str]] = []
    stats: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    tokenizer_counts: Counter[str] = Counter()
    duplicate_keys: Counter[str] = Counter()
    seen_keys: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = read_json_line(path, line_no, line)
            record_key = stable_record_key(row)
            if record_key in seen_keys:
                duplicate_keys[record_key] += 1
            seen_keys.add(record_key)
            score = hash_score(seed, dataset_name, record_key)
            item = (-score, record_key)
            if len(heap) < valid_size:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)

            stats["rows"] += 1
            if bool_value(nested_get(row, "content_keep"), default=False):
                stats["content_keep_true"] += 1
            if bool_value(nested_get(row, "content_token_keep"), default=False):
                stats["content_token_keep_true"] += 1
            if nested_get(row, "content_token_ids"):
                stats["has_content_token_ids"] += 1
            vocab = nested_get(row, "content_ctc_vocab_size")
            if vocab not in (None, ""):
                tokenizer_counts[f"vocab={vocab}"] += 1
            tokenizer_id = nested_get(row, "content_tokenizer_id") or nested_get(row, "content_tokenizer")
            if tokenizer_id not in (None, ""):
                tokenizer_counts[str(tokenizer_id)] += 1
            mode_counts[infer_mode(row, dataset_name)] += 1
            language_counts[infer_language(row)] += 1
            if progress_every > 0 and stats["rows"] % progress_every == 0:
                print(
                    f"[ver2.6-split:scan] dataset={dataset_name} rows={stats['rows']} "
                    f"valid_heap={len(heap)} content_token_keep={stats['content_token_keep_true']}",
                    flush=True,
                )
    selected = {record_key for _, record_key in heap}
    summary = {
        "rows": int(stats["rows"]),
        "selected_valid_keys": len(selected),
        "content_keep_true": int(stats["content_keep_true"]),
        "content_token_keep_true": int(stats["content_token_keep_true"]),
        "has_content_token_ids": int(stats["has_content_token_ids"]),
        "mode_counts": dict(mode_counts),
        "language_counts": dict(language_counts),
        "tokenizer_counts": dict(tokenizer_counts),
        "duplicate_key_count": sum(duplicate_keys.values()),
        "duplicate_key_examples": duplicate_keys.most_common(10),
    }
    return selected, summary


def write_split(
    *,
    dataset_name: str,
    path: Path,
    train_path: Path,
    valid_path: Path,
    combined_valid_handle,
    valid_keys: set[str],
    progress_every: int,
    overwrite: bool,
) -> dict[str, Any]:
    if not overwrite:
        for out_path in (train_path, valid_path):
            if out_path.exists():
                raise FileExistsError(f"output exists, pass --overwrite: {out_path}")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    train_tmp = train_path.with_name(train_path.name + ".tmp")
    valid_tmp = valid_path.with_name(valid_path.name + ".tmp")
    stats: Counter[str] = Counter()
    train_mode_counts: Counter[str] = Counter()
    valid_mode_counts: Counter[str] = Counter()
    train_language_counts: Counter[str] = Counter()
    valid_language_counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as src, train_tmp.open("w", encoding="utf-8") as train_out, valid_tmp.open(
        "w", encoding="utf-8"
    ) as valid_out:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            row = read_json_line(path, line_no, line)
            record_key = stable_record_key(row)
            is_valid = record_key in valid_keys
            mode = infer_mode(row, dataset_name)
            language = infer_language(row)
            if is_valid:
                valid_out.write(line)
                combined_valid_handle.write(line)
                stats["valid_rows"] += 1
                valid_mode_counts[mode] += 1
                valid_language_counts[language] += 1
            else:
                train_out.write(line)
                stats["train_rows"] += 1
                train_mode_counts[mode] += 1
                train_language_counts[language] += 1
            stats["rows"] += 1
            if progress_every > 0 and stats["rows"] % progress_every == 0:
                print(
                    f"[ver2.6-split:write] dataset={dataset_name} rows={stats['rows']} "
                    f"train={stats['train_rows']} valid={stats['valid_rows']}",
                    flush=True,
                )
    train_tmp.replace(train_path)
    valid_tmp.replace(valid_path)
    return {
        "rows": int(stats["rows"]),
        "train_rows": int(stats["train_rows"]),
        "valid_rows": int(stats["valid_rows"]),
        "train_mode_counts": dict(train_mode_counts),
        "valid_mode_counts": dict(valid_mode_counts),
        "train_language_counts": dict(train_language_counts),
        "valid_language_counts": dict(valid_language_counts),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_valid = output_dir / f"ver2_6_full_valid_seed{args.seed}.jsonl"
    if combined_valid.exists() and not args.overwrite:
        raise FileExistsError(f"output exists, pass --overwrite: {combined_valid}")
    specs = [
        ("no_text", Path(args.no_text_jsonl).expanduser(), int(args.no_text_valid_size)),
        ("text_old", Path(args.text_old_jsonl).expanduser(), int(args.text_old_valid_size)),
        ("text_new", Path(args.text_new_jsonl).expanduser(), int(args.text_new_valid_size)),
    ]
    for _, path, _ in specs:
        if not path.exists():
            raise FileNotFoundError(path)

    summary: dict[str, Any] = {
        "status": "running",
        "seed": int(args.seed),
        "output_dir": str(output_dir.resolve(strict=False)),
        "combined_valid_jsonl": str(combined_valid.resolve(strict=False)),
        "datasets": {},
    }
    selected_by_dataset: dict[str, set[str]] = {}
    for dataset_name, path, valid_size in specs:
        selected, scan_summary = select_validation_keys(
            dataset_name=dataset_name,
            path=path,
            valid_size=valid_size,
            seed=int(args.seed),
            progress_every=int(args.progress_every),
        )
        selected_by_dataset[dataset_name] = selected
        summary["datasets"][dataset_name] = {
            "input_jsonl": str(path.resolve(strict=False)),
            "requested_valid_size": int(valid_size),
            "scan": scan_summary,
        }

    combined_tmp = combined_valid.with_name(combined_valid.name + ".tmp")
    with combined_tmp.open("w", encoding="utf-8") as combined_handle:
        for dataset_name, path, _ in specs:
            train_path = train_output_path(path, int(args.seed))
            valid_path = valid_output_path(output_dir, dataset_name, int(args.seed))
            write_summary = write_split(
                dataset_name=dataset_name,
                path=path,
                train_path=train_path,
                valid_path=valid_path,
                combined_valid_handle=combined_handle,
                valid_keys=selected_by_dataset[dataset_name],
                progress_every=int(args.progress_every),
                overwrite=bool(args.overwrite),
            )
            summary["datasets"][dataset_name].update(
                {
                    "train_jsonl": str(train_path.resolve(strict=False)),
                    "valid_jsonl": str(valid_path.resolve(strict=False)),
                    "write": write_summary,
                }
            )
    combined_tmp.replace(combined_valid)
    summary["status"] = "complete"
    summary["combined_valid_rows"] = sum(
        int(item["write"]["valid_rows"]) for item in summary["datasets"].values()
    )
    summary["combined_train_rows"] = sum(
        int(item["write"]["train_rows"]) for item in summary["datasets"].values()
    )
    write_json(output_dir / f"ver2_6_full_split_seed{args.seed}.summary.json", summary)
    write_json(output_dir / f"ver2_6_full_split_seed{args.seed}.done.json", summary)
    print(
        f"[ver2.6-split] complete seed={args.seed} "
        f"train_rows={summary['combined_train_rows']} valid_rows={summary['combined_valid_rows']} "
        f"summary={output_dir / f'ver2_6_full_split_seed{args.seed}.summary.json'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

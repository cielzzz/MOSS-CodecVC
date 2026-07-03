#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NO_TEXT_JSONL = (
    ROOT
    / "trainset/zh45w_en22w_no_text/sft/"
    / "moss_codecvc_sft.zh45w_en22w_no_text.with_light_ecapa_spk.with_prosody.with_asr_filter.with_wavlm_bnf.with_spm_content_tokens.ctc_clean.jsonl"
)
DEFAULT_TEXT_JSONL = (
    ROOT
    / "trainset/zh3w_en3w_text_prosody_independent_timbre/sft/"
    / "moss_codecvc_sft.zh3w_en3w_text_prosody_independent_timbre.with_light_ecapa_spk.with_prosody.with_target_asr.with_content_tokens.with_target_wavlm_bnf.with_spm_content_tokens.ctc_clean.jsonl"
)


SOURCE_FEATURE_KEYS = {
    "wavlm": (
        "source_wavlm_bnf_features_path",
        "source_wavlm_feature_path",
        "source_wavlm_features_path",
    ),
    "hubert": (
        "source_hubert_feature_path",
        "source_hubert_features_path",
    ),
    "asr_bnf": (
        "source_asr_bnf_feature_path",
        "source_asr_bnf_features_path",
        "source_bnf_feature_path",
        "source_bnf_features_path",
    ),
    "any": (
        "source_wavlm_bnf_features_path",
        "source_asr_bnf_feature_path",
        "source_asr_bnf_features_path",
        "source_bnf_feature_path",
        "source_bnf_features_path",
        "source_wavlm_feature_path",
        "source_wavlm_features_path",
        "source_semantic_feature_path",
        "source_semantic_features_path",
        "source_hubert_feature_path",
        "source_hubert_features_path",
    ),
}

TARGET_FEATURE_KEYS = {
    "wavlm": (
        "target_wavlm_bnf_features_path",
        "target_wavlm_feature_path",
        "target_wavlm_features_path",
    ),
    "hubert": (
        "target_hubert_feature_path",
        "target_hubert_features_path",
    ),
    "asr_bnf": (
        "target_asr_bnf_feature_path",
        "target_asr_bnf_features_path",
        "target_bnf_feature_path",
        "target_bnf_features_path",
    ),
    "any": (
        "target_wavlm_bnf_features_path",
        "target_asr_bnf_feature_path",
        "target_asr_bnf_features_path",
        "target_bnf_feature_path",
        "target_bnf_features_path",
        "target_wavlm_feature_path",
        "target_wavlm_features_path",
        "target_semantic_feature_path",
        "target_semantic_features_path",
        "target_hubert_feature_path",
        "target_hubert_features_path",
    ),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Filter ASR-clean MOSS-CodecVC text/no-text manifests and build "
            "Ver2.8 train/valid splits."
        )
    )
    ap.add_argument("--no-text-jsonl", default=str(DEFAULT_NO_TEXT_JSONL))
    ap.add_argument("--text-jsonl", default=str(DEFAULT_TEXT_JSONL))
    ap.add_argument("--output-dir", default=str(ROOT / "trainset/ver2_8_prepared"))
    ap.add_argument(
        "--semantic-kind",
        choices=("wavlm", "asr_bnf", "hubert", "any"),
        default="wavlm",
        help="Required continuous feature family. Use any only for smoke tests or backward-compatible HuBERT runs.",
    )
    ap.add_argument("--check-feature-files", action="store_true")
    ap.add_argument("--require-no-text-target-feature", action="store_true")
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--valid-ratio", type=float, default=0.01)
    ap.add_argument("--valid-min-no-text", type=int, default=500)
    ap.add_argument("--valid-min-text", type=int, default=200)
    ap.add_argument("--valid-max-no-text", type=int, default=1000)
    ap.add_argument("--valid-max-text", type=int, default=1000)
    ap.add_argument("--valid-count-no-text", type=int, default=0)
    ap.add_argument("--valid-count-text", type=int, default=0)
    ap.add_argument(
        "--text-repeat",
        type=int,
        default=5,
        help=(
            "Total sampled copies for text train rows. The default 5 means "
            "1 original copy + 4 extra copies, balancing the current Ver2.8 "
            "text split against no-text."
        ),
    )
    ap.add_argument("--max-rows-per-source", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
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
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


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


def has_value(row: dict[str, Any], key: str) -> bool:
    value = nested_get(row, key)
    if value in (None, ""):
        return False
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    return True


def first_path(row: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, str] | tuple[None, None]:
    for key in keys:
        value = nested_get(row, key)
        if value not in (None, ""):
            return key, str(value)
    return None, None


def feature_exists(path: str | None) -> bool:
    return bool(path) and Path(str(path)).expanduser().exists()


def infer_mode(row: dict[str, Any]) -> str:
    mode = str(nested_get(row, "moss_codecvc_mode") or nested_get(row, "mode") or "").strip().lower()
    if mode in {"text", "no_text"}:
        return mode
    text = str(nested_get(row, "text") or "").strip()
    return "no_text" if text in {"", "<NO_TEXT>"} else "text"


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def infer_language(row: dict[str, Any]) -> str:
    raw = str(nested_get(row, "language") or nested_get(row, "lang") or "").strip().lower()
    if raw in {"zh", "zho", "cn", "chinese", "mandarin", "中文"}:
        return "zh"
    if raw in {"en", "eng", "english"}:
        return "en"
    text = " ".join(
        str(nested_get(row, key) or "")
        for key in ("content_ref_text", "asr_src_text", "asr_tgt_text", "text")
    )
    if has_cjk(text):
        return "zh"
    if any(("a" <= char.lower() <= "z") for char in text):
        return "en"
    return "other"


def group_key(row: dict[str, Any], mode: str) -> str:
    for key in (
        "target_speaker_id",
        "speaker_id",
        "target_speaker_embedding_path",
        "timbre_ref_speaker_embedding_path",
        "source_speaker_embedding_path",
        "target_audio",
        "audio",
        "sample_id",
    ):
        value = nested_get(row, key)
        if value not in (None, ""):
            return f"{mode}:{key}:{value}"
    return f"{mode}:row:{hashlib.sha1(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]}"


def validate_row(
    row: dict[str, Any],
    *,
    expected_mode: str,
    semantic_kind: str,
    check_feature_files: bool,
    require_no_text_target_feature: bool,
) -> tuple[bool, str, dict[str, str]]:
    mode = infer_mode(row)
    if mode != expected_mode:
        return False, f"mode={mode}", {}
    if not bool_value(nested_get(row, "content_keep"), default=False):
        return False, "content_keep=false", {}
    if not bool_value(nested_get(row, "content_token_keep"), default=False):
        return False, "content_token_keep=false", {}
    if not has_value(row, "content_token_ids") and not has_value(row, "content_token_ids_path"):
        return False, "missing_content_token_ids", {}
    if not has_value(row, "audio_codes") and not has_value(row, "audio_codes_path"):
        return False, "missing_target_audio_codes", {}
    refs = nested_get(row, "reference_audio_codes")
    if isinstance(refs, list):
        if len(refs) < 2 or refs[0] in (None, []) or refs[1] in (None, []):
            return False, "missing_reference_audio_codes_pair", {}
    elif not has_value(row, "reference_audio_codes_path"):
        return False, "missing_reference_audio_codes", {}

    asr_tgt = str(nested_get(row, "asr_tgt_text") or "").strip()
    if not asr_tgt:
        return False, "missing_asr_tgt_text", {}
    if expected_mode == "no_text":
        asr_src = str(nested_get(row, "asr_src_text") or "").strip()
        if not asr_src:
            return False, "missing_asr_src_text", {}
        source_key, source_path = first_path(row, SOURCE_FEATURE_KEYS[semantic_kind])
        if not source_path:
            return False, f"missing_{semantic_kind}_source_feature", {}
        if check_feature_files and not feature_exists(source_path):
            return False, f"missing_source_feature_file:{source_key}", {}
        target_key, target_path = first_path(row, TARGET_FEATURE_KEYS[semantic_kind])
        if require_no_text_target_feature and not target_path:
            return False, f"missing_{semantic_kind}_target_feature", {}
        if require_no_text_target_feature and check_feature_files and not feature_exists(target_path):
            return False, f"missing_target_feature_file:{target_key}", {}
        return True, "keep", {"source_feature_key": str(source_key), "target_feature_key": str(target_key or "")}

    text = str(nested_get(row, "text") or nested_get(row, "content_ref_text") or "").strip()
    if not text or text == "<NO_TEXT>":
        return False, "missing_text", {}
    target_key, target_path = first_path(row, TARGET_FEATURE_KEYS[semantic_kind])
    if not target_path:
        return False, f"missing_{semantic_kind}_target_feature", {}
    if check_feature_files and not feature_exists(target_path):
        return False, f"missing_target_feature_file:{target_key}", {}
    return True, "keep", {"target_feature_key": str(target_key)}


def clean_rows(
    path: Path,
    *,
    expected_mode: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    feature_keys: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    for idx, row in enumerate(iter_jsonl(path)):
        if args.max_rows_per_source > 0 and idx >= int(args.max_rows_per_source):
            break
        ok, reason, meta = validate_row(
            row,
            expected_mode=expected_mode,
            semantic_kind=args.semantic_kind,
            check_feature_files=bool(args.check_feature_files),
            require_no_text_target_feature=bool(args.require_no_text_target_feature),
        )
        if ok:
            out = dict(row)
            out["ver2_8_prepared"] = {
                "mode": expected_mode,
                "semantic_kind": args.semantic_kind,
                "asr_filter_pass": True,
                **meta,
            }
            kept.append(out)
            languages[infer_language(out)] += 1
            for value in meta.values():
                if value:
                    feature_keys[value] += 1
        else:
            rej = dict(row)
            rej["ver2_8_reject_reason"] = reason
            rejected.append(rej)
            reasons[reason] += 1
    summary = {
        "input": str(path),
        "mode": expected_mode,
        "kept": len(kept),
        "rejected": len(rejected),
        "reject_reasons": dict(reasons.most_common()),
        "languages": dict(languages),
        "feature_keys": dict(feature_keys),
    }
    return kept, rejected, summary


def valid_target_count(rows: list[dict[str, Any]], *, mode: str, args: argparse.Namespace) -> int:
    override = int(args.valid_count_no_text if mode == "no_text" else args.valid_count_text)
    if override > 0:
        return min(max(0, override), max(0, len(rows) - 1))
    if len(rows) < 2:
        return 0
    ratio_count = int(round(len(rows) * float(args.valid_ratio)))
    min_count = int(args.valid_min_no_text if mode == "no_text" else args.valid_min_text)
    max_count = int(args.valid_max_no_text if mode == "no_text" else args.valid_max_text)
    if len(rows) < min_count * 2:
        floor = max(1, len(rows) // 20)
    else:
        floor = min_count
    return min(max(ratio_count, floor), max_count, len(rows) - 1)


def split_rows(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    target = valid_target_count(rows, mode=mode, args=args)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row, mode)].append(row)
    group_items = list(groups.items())
    rng = random.Random(int(args.seed) + (17 if mode == "text" else 0))
    rng.shuffle(group_items)
    valid: list[dict[str, Any]] = []
    valid_groups: set[str] = set()
    lang_counts: Counter[str] = Counter()
    for key, group in group_items:
        if len(valid) >= target:
            break
        valid_groups.add(key)
        valid.extend(group)
        for row in group:
            lang_counts[infer_language(row)] += 1
    train = [row for key, group in group_items if key not in valid_groups for row in group]
    for split_name, split_rows_ in (("train", train), ("valid", valid)):
        for row in split_rows_:
            prepared = dict(row.get("ver2_8_prepared") or {})
            prepared["split"] = split_name
            row["ver2_8_prepared"] = prepared
    summary = {
        "mode": mode,
        "input_rows": len(rows),
        "train_rows": len(train),
        "valid_rows": len(valid),
        "valid_target_rows": target,
        "groups": len(groups),
        "valid_groups": len(valid_groups),
        "valid_languages": dict(lang_counts),
    }
    return train, valid, summary


def ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists, pass --overwrite: {path}")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser()
    temp_dir = out_dir / "temp"
    for rel in (
        "no_text.train.jsonl",
        "no_text.valid.jsonl",
        "text.train.jsonl",
        "text.valid.jsonl",
        "mixed.train.spec.txt",
        "mixed.valid.spec.txt",
        "summary.json",
    ):
        ensure_writable(out_dir / rel, bool(args.overwrite))

    no_text_rows, no_text_rejects, no_text_clean_summary = clean_rows(
        Path(args.no_text_jsonl).expanduser(),
        expected_mode="no_text",
        args=args,
    )
    text_rows, text_rejects, text_clean_summary = clean_rows(
        Path(args.text_jsonl).expanduser(),
        expected_mode="text",
        args=args,
    )
    if not no_text_rows:
        raise RuntimeError(f"no usable no-text rows after filtering; summary={no_text_clean_summary}")
    if not text_rows:
        raise RuntimeError(f"no usable text rows after filtering; summary={text_clean_summary}")

    no_text_train, no_text_valid, no_text_split_summary = split_rows(no_text_rows, mode="no_text", args=args)
    text_train, text_valid, text_split_summary = split_rows(text_rows, mode="text", args=args)

    no_text_train_path = out_dir / "no_text.train.jsonl"
    no_text_valid_path = out_dir / "no_text.valid.jsonl"
    text_train_path = out_dir / "text.train.jsonl"
    text_valid_path = out_dir / "text.valid.jsonl"
    write_jsonl(no_text_train_path, no_text_train)
    write_jsonl(no_text_valid_path, no_text_valid)
    write_jsonl(text_train_path, text_train)
    write_jsonl(text_valid_path, text_valid)
    write_jsonl(temp_dir / "asr_rejects.no_text.jsonl", no_text_rejects)
    write_jsonl(temp_dir / "asr_rejects.text.jsonl", text_rejects)

    train_spec = f"{no_text_train_path}::repeat=1,{text_train_path}::repeat={int(args.text_repeat)}"
    valid_spec = f"{no_text_valid_path}::repeat=1,{text_valid_path}::repeat=1"
    (out_dir / "mixed.train.spec.txt").write_text(train_spec + "\n", encoding="utf-8")
    (out_dir / "mixed.valid.spec.txt").write_text(valid_spec + "\n", encoding="utf-8")
    no_text_effective_train_rows = len(no_text_train)
    text_effective_train_rows = len(text_train) * int(args.text_repeat)
    text_no_text_balance_ratio = (
        text_effective_train_rows / float(no_text_effective_train_rows)
        if no_text_effective_train_rows > 0
        else None
    )

    summary = {
        "status": "complete",
        "semantic_kind": args.semantic_kind,
        "check_feature_files": bool(args.check_feature_files),
        "require_no_text_target_feature": bool(args.require_no_text_target_feature),
        "seed": int(args.seed),
        "text_repeat": int(args.text_repeat),
        "text_extra_copies": max(0, int(args.text_repeat) - 1),
        "no_text_effective_train_rows": no_text_effective_train_rows,
        "text_effective_train_rows": text_effective_train_rows,
        "effective_train_rows_after_repeat": no_text_effective_train_rows + text_effective_train_rows,
        "text_no_text_balance_ratio": text_no_text_balance_ratio,
        "clean": {
            "no_text": no_text_clean_summary,
            "text": text_clean_summary,
        },
        "split": {
            "no_text": no_text_split_summary,
            "text": text_split_summary,
        },
        "outputs": {
            "no_text_train": str(no_text_train_path),
            "no_text_valid": str(no_text_valid_path),
            "text_train": str(text_train_path),
            "text_valid": str(text_valid_path),
            "train_spec": train_spec,
            "valid_spec": valid_spec,
            "no_text_rejects": str(temp_dir / "asr_rejects.no_text.jsonl"),
            "text_rejects": str(temp_dir / "asr_rejects.text.jsonl"),
        },
    }
    write_json(out_dir / "summary.json", summary)
    balance_ratio_text = (
        f"{text_no_text_balance_ratio:.4f}"
        if text_no_text_balance_ratio is not None
        else "n/a"
    )
    print(
        "[ver2.8-prepare] "
        f"no_text train={len(no_text_train)} valid={len(no_text_valid)} rejects={len(no_text_rejects)}; "
        f"text train={len(text_train)} valid={len(text_valid)} rejects={len(text_rejects)}",
        flush=True,
    )
    print(
        "[ver2.8-prepare] "
        f"text_repeat={int(args.text_repeat)} text_extra_copies={max(0, int(args.text_repeat) - 1)} "
        f"text_effective={text_effective_train_rows} "
        f"no_text_effective={no_text_effective_train_rows} "
        f"balance_ratio={balance_ratio_text}",
        flush=True,
    )
    print(f"[ver2.8-prepare] train_spec={train_spec}", flush=True)
    print(f"[ver2.8-prepare] valid_spec={valid_spec}", flush=True)
    print(f"[ver2.8-prepare] summary={out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

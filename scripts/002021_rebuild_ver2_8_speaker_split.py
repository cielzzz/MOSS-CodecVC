#!/usr/bin/env python
from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
import hashlib
import heapq
import json
import os
from pathlib import Path
import random
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OLD_PREPARED_DIR = (
    ROOT
    / "trainset/ver2_8_prepared_zh45w_en22w_plus_zh11w_en11w_0005_0015_merged_no_text_plus_zh3w_text_textrep10"
)
DEFAULT_NO_TEXT_JSONL = OLD_PREPARED_DIR / "temp/no_text.merged.source_clean.jsonl"
DEFAULT_TEXT_JSONL = (
    ROOT
    / "trainset/zh3w_en3w_text_prosody_independent_timbre/sft/"
    / (
        "moss_codecvc_sft.zh3w_en3w_text_prosody_independent_timbre.with_light_ecapa_spk.with_prosody."
        "with_target_asr.with_content_tokens.with_target_wavlm_bnf.with_spm_content_tokens.ctc_clean.jsonl"
    )
)
DEFAULT_MANIFESTS = (
    ROOT / "trainset/zh45w_en22w_no_text/manifests/vc_manifest.zh45w_en22w_no_text.jsonl",
    ROOT
    / "trainset/zh11w_en11w_0005_0015_vcdata_first_no_text/manifests/"
    / "vc_manifest.zh11w_en11w_0005_0015_vcdata_first_no_text.jsonl",
    ROOT
    / "trainset/zh3w_en3w_text_prosody_independent_timbre/manifests/"
    / "vc_manifest.zh3w_en3w_text_prosody_independent_timbre.jsonl",
)

ROLE_KEYS = {
    "source": ("source_speaker_id", "source_pseudo_speaker_id", "source_speaker_pseudo_id"),
    "timbre_ref": ("timbre_ref_speaker_id", "timbre_ref_pseudo_speaker_id", "timbre_ref_speaker_pseudo_id"),
    "target": ("target_speaker_id", "target_pseudo_speaker_id", "target_speaker_pseudo_id"),
}
MANIFEST_ROLE_KEYS = {
    "source": "source_speaker_id",
    "timbre_ref": "timbre_ref_speaker_id",
    "target": "target_speaker_id",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rebuild Ver2.8 full train/valid split with speaker isolation.")
    ap.add_argument("--no-text-jsonl", default=str(DEFAULT_NO_TEXT_JSONL))
    ap.add_argument("--text-jsonl", default=str(DEFAULT_TEXT_JSONL))
    ap.add_argument("--output-dir", default=str(ROOT / "trainset/ver2_8_prepared_speaker_split_20260705"))
    ap.add_argument("--old-seen-valid-no-text-jsonl", default=str(OLD_PREPARED_DIR / "no_text.valid.jsonl"))
    ap.add_argument("--old-seen-valid-text-jsonl", default=str(OLD_PREPARED_DIR / "text.valid.jsonl"))
    ap.add_argument("--speaker-manifest-jsonl", action="append", default=[])
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--unseen-valid-no-text-count", type=int, default=1000)
    ap.add_argument("--unseen-valid-text-count", type=int, default=300)
    ap.add_argument("--target-speaker-ratio", type=float, default=0.01)
    ap.add_argument("--duration-frame-rate", type=float, default=12.5)
    ap.add_argument("--text-repeat", type=int, default=10)
    ap.add_argument("--docs-md", default=str(ROOT / "docs/ver2_8_full_speaker_split_20260705.md"))
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
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_offsets(path: Path) -> dict[str, Any]:
    offsets = array("Q")
    with path.open("rb") as handle:
        while True:
            pos = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.strip():
                offsets.append(pos)
    index_path = Path(str(path) + ".offsets.u64")
    meta_path = Path(str(index_path) + ".json")
    tmp_index = Path(str(index_path) + f".tmp.{os.getpid()}")
    tmp_meta = Path(str(meta_path) + f".tmp.{os.getpid()}")
    with tmp_index.open("wb") as handle:
        offsets.tofile(handle)
    stat = path.stat()
    meta = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "index_path": str(index_path.resolve()),
        "rows": int(len(offsets)),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "source_path": str(path.resolve()),
        "source_size": int(stat.st_size),
    }
    with tmp_meta.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
    tmp_index.replace(index_path)
    tmp_meta.replace(meta_path)
    return meta


def nested_get(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if value not in (None, ""):
        return value
    for meta_key in ("moss_codecvc_meta", "meta"):
        meta = row.get(meta_key)
        if isinstance(meta, dict):
            value = meta.get(key)
            if value not in (None, ""):
                return value
    return None


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def infer_language(row: dict[str, Any]) -> str:
    raw = str(nested_get(row, "language") or nested_get(row, "lang") or "").strip().lower()
    if raw in {"zh", "zho", "cn", "chinese", "mandarin"}:
        return "zh"
    if raw in {"en", "eng", "english"}:
        return "en"
    text = " ".join(
        str(nested_get(row, key) or "")
        for key in ("content_ref_text", "asr_src_text", "asr_tgt_text", "text")
    )
    if has_cjk(text):
        return "zh"
    if any("a" <= char.lower() <= "z" for char in text):
        return "en"
    return "other"


def duration_seconds(row: dict[str, Any], frame_rate: float) -> float:
    for key in ("target_duration_sec", "duration_sec", "duration", "audio_duration_sec"):
        value = nested_get(row, key)
        if value not in (None, ""):
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                return parsed
    for key in ("target_codec_frames", "audio_codec_frames", "target_frames", "num_target_frames"):
        value = nested_get(row, key)
        if value not in (None, ""):
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0 and frame_rate > 0:
                return parsed / float(frame_rate)
    return 0.0


def duration_bin(seconds: float) -> str:
    if seconds <= 0:
        return "unknown"
    if seconds < 4.0:
        return "short"
    if seconds < 8.0:
        return "medium"
    return "long"


def row_key(row: dict[str, Any]) -> str:
    sample_id = str(row.get("sample_id") or "").strip()
    if sample_id:
        return sample_id
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def manifest_value(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value not in (None, ""):
        return str(value)
    meta = row.get("meta")
    if isinstance(meta, dict):
        value = meta.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def load_manifest_lookup(paths: list[Path]) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    lookup: dict[str, dict[str, str]] = {}
    audits: list[dict[str, Any]] = []
    all_speakers: set[str] = set()
    conflicts = 0
    for path in paths:
        rows = 0
        sample_ids = 0
        role_counts: Counter[str] = Counter()
        role_speakers: dict[str, set[str]] = {role: set() for role in ROLE_KEYS}
        speakers: set[str] = set()
        source_counts: Counter[str] = Counter()
        session_field_counts: Counter[str] = Counter()
        language_counts: Counter[str] = Counter()
        for row in iter_jsonl(path):
            rows += 1
            sid = str(row.get("sample_id") or "").strip()
            if sid:
                sample_ids += 1
            values: dict[str, str] = {}
            for role, key in MANIFEST_ROLE_KEYS.items():
                value = manifest_value(row, key)
                if value:
                    values[role] = value
                    role_counts[key] += 1
                    role_speakers[role].add(value)
                    speakers.add(value)
                    all_speakers.add(value)
            meta = row.get("meta")
            if isinstance(meta, dict):
                for key, value in meta.items():
                    if key.endswith("_speaker_source") and value not in (None, ""):
                        source_counts[str(value)] += 1
                    if "session" in key.lower() and value not in (None, ""):
                        session_field_counts[key] += 1
            for key, value in row.items():
                if "session" in key.lower() and value not in (None, ""):
                    session_field_counts[key] += 1
            lang = str(row.get("language") or "").strip().lower()
            if lang:
                language_counts[lang] += 1
            if sid and set(values) == set(ROLE_KEYS):
                existing = lookup.get(sid)
                if existing is not None and existing != values:
                    conflicts += 1
                lookup.setdefault(sid, values)
        audits.append(
            {
                "path": str(path),
                "rows": rows,
                "sample_ids": sample_ids,
                "lookup_rows": sample_ids,
                "unique_speakers": len(speakers),
                "role_field_counts": dict(role_counts),
                "role_unique_speakers": {role: len(values) for role, values in role_speakers.items()},
                "speaker_source_counts": dict(source_counts.most_common()),
                "session_field_counts": dict(session_field_counts.most_common()),
                "languages": dict(language_counts.most_common()),
            }
        )
    return lookup, {
        "manifest_files": audits,
        "lookup_sample_ids": len(lookup),
        "unique_speakers_all_manifests": len(all_speakers),
        "sample_id_conflicts": conflicts,
    }


def direct_speaker_value(row: dict[str, Any], role: str) -> str | None:
    for key in ROLE_KEYS[role]:
        value = nested_get(row, key)
        if value not in (None, ""):
            return str(value)
    return None


def attach_speaker_ids(
    row: dict[str, Any],
    *,
    mode: str,
    split: str,
    lookup: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], dict[str, str], str]:
    out = dict(row)
    sid = str(out.get("sample_id") or "").strip()
    manifest_ids = lookup.get(sid) or {}
    ids: dict[str, str] = {}
    source = "row"
    for role in ROLE_KEYS:
        value = direct_speaker_value(out, role)
        if value is None:
            value = manifest_ids.get(role)
            if value is not None:
                source = "manifest" if source == "row" else source
        if value is None:
            source = "missing"
        else:
            ids[role] = value
            out[f"{role}_speaker_id"] = value
    prepared = dict(out.get("ver2_8_prepared") or {})
    prepared.update({"mode": mode, "split": split, "speaker_id_source": source})
    out["ver2_8_prepared"] = prepared
    out["ver2_8_speaker_split"] = {
        "mode": mode,
        "split": split,
        "speaker_id_source": source,
        "source_speaker_id": ids.get("source"),
        "timbre_ref_speaker_id": ids.get("timbre_ref"),
        "target_speaker_id": ids.get("target"),
    }
    return out, ids, source


def load_rows_with_speakers(
    path: Path,
    *,
    mode: str,
    split: str,
    lookup: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    missing_examples: list[str] = []
    sample_id_present = 0
    row_direct_counts: Counter[str] = Counter()
    meta_direct_counts: Counter[str] = Counter()
    for raw in iter_jsonl(path):
        if raw.get("sample_id") not in (None, ""):
            sample_id_present += 1
        meta = raw.get("moss_codecvc_meta")
        for role, keys in ROLE_KEYS.items():
            if any(raw.get(key) not in (None, "") for key in keys):
                row_direct_counts[role] += 1
            if isinstance(meta, dict) and any(meta.get(key) not in (None, "") for key in keys):
                meta_direct_counts[role] += 1
        row, ids, source = attach_speaker_ids(raw, mode=mode, split=split, lookup=lookup)
        source_counts[source] += 1
        if set(ids) != set(ROLE_KEYS) and len(missing_examples) < 10:
            missing_examples.append(str(raw.get("sample_id") or ""))
        rows.append(row)
    if source_counts.get("missing", 0) > 0:
        raise RuntimeError(
            f"{path} has unresolved speaker ids: {source_counts['missing']} rows; examples={missing_examples}"
        )
    return rows, {
        "path": str(path),
        "rows": len(rows),
        "sample_id_present": sample_id_present,
        "row_direct_speaker_counts": dict(row_direct_counts),
        "moss_codecvc_meta_direct_speaker_counts": dict(meta_direct_counts),
        "speaker_id_source_counts": dict(source_counts),
    }


def speaker_set(row: dict[str, Any]) -> set[str]:
    speakers = set()
    for role in ROLE_KEYS:
        value = row.get(f"{role}_speaker_id")
        if value not in (None, ""):
            speakers.add(str(value))
    return speakers


def split_stats(rows: list[dict[str, Any]], *, frame_rate: float) -> dict[str, Any]:
    langs: Counter[str] = Counter()
    bins: Counter[str] = Counter()
    lang_bins: Counter[str] = Counter()
    target_speakers: Counter[str] = Counter()
    seconds_values: list[float] = []
    speakers: set[str] = set()
    for row in rows:
        lang = infer_language(row)
        sec = duration_seconds(row, frame_rate)
        bin_name = duration_bin(sec)
        langs[lang] += 1
        bins[bin_name] += 1
        lang_bins[f"{lang}/{bin_name}"] += 1
        if sec > 0:
            seconds_values.append(sec)
        speakers.update(speaker_set(row))
        target = row.get("target_speaker_id")
        if target not in (None, ""):
            target_speakers[str(target)] += 1
    seconds_values.sort()

    def percentile(q: float) -> float | None:
        if not seconds_values:
            return None
        idx = min(len(seconds_values) - 1, max(0, int(round((len(seconds_values) - 1) * q))))
        return round(float(seconds_values[idx]), 3)

    return {
        "rows": len(rows),
        "speaker_count": len(speakers),
        "target_speaker_count": len(target_speakers),
        "languages": dict(langs.most_common()),
        "duration_bins": dict(bins.most_common()),
        "language_duration_bins": dict(lang_bins.most_common()),
        "duration_seconds": {
            "min": percentile(0.0),
            "p50": percentile(0.5),
            "p90": percentile(0.9),
            "max": percentile(1.0),
        },
    }


class StatsAccumulator:
    def __init__(self, *, frame_rate: float) -> None:
        self.frame_rate = float(frame_rate)
        self.rows = 0
        self.languages: Counter[str] = Counter()
        self.duration_bins: Counter[str] = Counter()
        self.language_duration_bins: Counter[str] = Counter()
        self.speakers: set[str] = set()
        self.target_speakers: Counter[str] = Counter()
        self.duration_values: list[float] = []

    def add(self, row: dict[str, Any]) -> None:
        self.rows += 1
        lang = infer_language(row)
        seconds = duration_seconds(row, self.frame_rate)
        bin_name = duration_bin(seconds)
        self.languages[lang] += 1
        self.duration_bins[bin_name] += 1
        self.language_duration_bins[f"{lang}/{bin_name}"] += 1
        self.speakers.update(speaker_set(row))
        target = row.get("target_speaker_id")
        if target not in (None, ""):
            self.target_speakers[str(target)] += 1
        if seconds > 0:
            self.duration_values.append(seconds)

    def to_dict(self) -> dict[str, Any]:
        values = sorted(self.duration_values)

        def percentile(q: float) -> float | None:
            if not values:
                return None
            idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
            return round(float(values[idx]), 3)

        return {
            "rows": int(self.rows),
            "speaker_count": len(self.speakers),
            "target_speaker_count": len(self.target_speakers),
            "languages": dict(self.languages.most_common()),
            "duration_bins": dict(self.duration_bins.most_common()),
            "language_duration_bins": dict(self.language_duration_bins.most_common()),
            "duration_seconds": {
                "min": percentile(0.0),
                "p50": percentile(0.5),
                "p90": percentile(0.9),
                "max": percentile(1.0),
            },
        }


def combine_stats_dicts(left: dict[str, Any], right: dict[str, Any], *, speaker_count: int) -> dict[str, Any]:
    left_duration = left.get("duration_seconds") or {}
    right_duration = right.get("duration_seconds") or {}
    mins = [value for value in (left_duration.get("min"), right_duration.get("min")) if value is not None]
    maxes = [value for value in (left_duration.get("max"), right_duration.get("max")) if value is not None]
    return {
        "rows": int(left.get("rows", 0)) + int(right.get("rows", 0)),
        "speaker_count": int(speaker_count),
        "target_speaker_count_upper_bound": int(left.get("target_speaker_count", 0))
        + int(right.get("target_speaker_count", 0)),
        "languages": dict((Counter(left.get("languages") or {}) + Counter(right.get("languages") or {})).most_common()),
        "duration_bins": dict(
            (Counter(left.get("duration_bins") or {}) + Counter(right.get("duration_bins") or {})).most_common()
        ),
        "language_duration_bins": dict(
            (
                Counter(left.get("language_duration_bins") or {})
                + Counter(right.get("language_duration_bins") or {})
            ).most_common()
        ),
        "duration_seconds": {
            "min": min(mins) if mins else None,
            "p50": None,
            "p90": None,
            "max": max(maxes) if maxes else None,
        },
    }


def stable_score(seed: int, key: str) -> float:
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def push_top_random(
    heap: list[tuple[float, str, dict[str, Any]]],
    *,
    limit: int,
    score: float,
    key: str,
    row: dict[str, Any],
) -> None:
    item = (-float(score), key, row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def scan_and_sample_unseen(
    path: Path,
    *,
    mode: str,
    lookup: dict[str, dict[str, str]],
    seen_keys: set[str],
    seen_speakers: set[str],
    target_count: int,
    seed: int,
    frame_rate: float,
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    source_counts: Counter[str] = Counter()
    row_direct_counts: Counter[str] = Counter()
    meta_direct_counts: Counter[str] = Counter()
    sample_id_present = 0
    input_speakers: set[str] = set()
    candidate_count = 0
    bucket_best: dict[tuple[str, str], tuple[float, str, dict[str, Any]]] = {}
    heap: list[tuple[float, str, dict[str, Any]]] = []
    heap_limit = max(int(target_count) + 32, int(target_count) * 2)
    for raw in iter_jsonl(path):
        if raw.get("sample_id") not in (None, ""):
            sample_id_present += 1
        meta = raw.get("moss_codecvc_meta")
        for role, keys in ROLE_KEYS.items():
            if any(raw.get(key) not in (None, "") for key in keys):
                row_direct_counts[role] += 1
            if isinstance(meta, dict) and any(meta.get(key) not in (None, "") for key in keys):
                meta_direct_counts[role] += 1
        row, ids, source = attach_speaker_ids(raw, mode=mode, split="all", lookup=lookup)
        source_counts[source] += 1
        if set(ids) != set(ROLE_KEYS):
            raise RuntimeError(f"{path} has unresolved speaker ids at sample_id={raw.get('sample_id')}")
        speakers = speaker_set(row)
        input_speakers.update(speakers)
        key = row_key(row)
        if key in seen_keys or (speakers & seen_speakers):
            continue
        candidate_count += 1
        score = stable_score(seed, key)
        bucket = (infer_language(row), duration_bin(duration_seconds(row, frame_rate)))
        existing = bucket_best.get(bucket)
        if existing is None or score < existing[0]:
            bucket_best[bucket] = (score, key, row)
        push_top_random(heap, limit=heap_limit, score=score, key=key, row=row)

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    for _, key, row in sorted(bucket_best.values(), key=lambda item: (item[0], item[1])):
        if len(selected) >= target_count:
            break
        if key not in selected_keys:
            selected.append(row)
            selected_keys.add(key)
    for neg_score, key, row in sorted(heap, reverse=True):
        if len(selected) >= target_count:
            break
        if key not in selected_keys:
            selected.append(row)
            selected_keys.add(key)
    if len(selected) < target_count:
        raise RuntimeError(f"{path} only has {len(selected)} unseen candidates; target={target_count}")
    audit = {
        "path": str(path),
        "rows": sum(source_counts.values()),
        "sample_id_present": sample_id_present,
        "row_direct_speaker_counts": dict(row_direct_counts),
        "moss_codecvc_meta_direct_speaker_counts": dict(meta_direct_counts),
        "speaker_id_source_counts": dict(source_counts),
        "unseen_candidate_rows": candidate_count,
        "unseen_bucket_count": len(bucket_best),
    }
    return selected[:target_count], input_speakers, audit


def write_split_streaming(
    path: Path,
    *,
    mode: str,
    lookup: dict[str, dict[str, str]],
    train_path: Path,
    unused_path: Path,
    seen_keys: set[str],
    unseen_keys: set[str],
    unseen_speakers: set[str],
    frame_rate: float,
) -> tuple[dict[str, Any], dict[str, Any], set[str], set[str]]:
    train_path.parent.mkdir(parents=True, exist_ok=True)
    unused_path.parent.mkdir(parents=True, exist_ok=True)
    train_tmp = train_path.with_name(train_path.name + f".tmp.{os.getpid()}")
    unused_tmp = unused_path.with_name(unused_path.name + f".tmp.{os.getpid()}")
    train_stats = StatsAccumulator(frame_rate=frame_rate)
    unused_stats = StatsAccumulator(frame_rate=frame_rate)
    with train_tmp.open("w", encoding="utf-8") as train_handle, unused_tmp.open("w", encoding="utf-8") as unused_handle:
        for raw in iter_jsonl(path):
            row, ids, _ = attach_speaker_ids(raw, mode=mode, split="all", lookup=lookup)
            if set(ids) != set(ROLE_KEYS):
                raise RuntimeError(f"{path} has unresolved speaker ids at sample_id={raw.get('sample_id')}")
            key = row_key(row)
            if key in seen_keys or key in unseen_keys:
                continue
            if speaker_set(row) & unseen_speakers:
                row = mark_split([row], "unseen_unused")[0]
                unused_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                unused_stats.add(row)
            else:
                row = mark_split([row], "train")[0]
                train_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                train_stats.add(row)
    train_tmp.replace(train_path)
    unused_tmp.replace(unused_path)
    return train_stats.to_dict(), unused_stats.to_dict(), set(train_stats.speakers), set(unused_stats.speakers)


def mark_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    marked: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        prepared = dict(out.get("ver2_8_prepared") or {})
        prepared["split"] = split
        out["ver2_8_prepared"] = prepared
        split_meta = dict(out.get("ver2_8_speaker_split") or {})
        split_meta["split"] = split
        out["ver2_8_speaker_split"] = split_meta
        marked.append(out)
    return marked


def select_unseen_valid(
    rows: list[dict[str, Any]],
    *,
    target_count: int,
    seed: int,
    forbidden_speakers: set[str],
    forbidden_keys: set[str],
    frame_rate: float,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row_key(row) in forbidden_keys:
            continue
        if speaker_set(row) & forbidden_speakers:
            continue
        buckets[(infer_language(row), duration_bin(duration_seconds(row, frame_rate)))].append(row)
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    selected_targets: set[str] = set()

    def add(row: dict[str, Any]) -> bool:
        key = row_key(row)
        if key in selected_keys:
            return False
        selected.append(row)
        selected_keys.add(key)
        target = row.get("target_speaker_id")
        if target not in (None, ""):
            selected_targets.add(str(target))
        return True

    for bucket_key in sorted(buckets):
        if len(selected) >= target_count:
            break
        for row in buckets[bucket_key]:
            if add(row):
                break
    shuffled = [row for bucket_rows in buckets.values() for row in bucket_rows]
    rng.shuffle(shuffled)
    for prefer_new_target in (True, False):
        if len(selected) >= target_count:
            break
        for row in shuffled:
            if len(selected) >= target_count:
                break
            target = row.get("target_speaker_id")
            if prefer_new_target and target not in (None, "") and str(target) in selected_targets:
                continue
            add(row)
    if len(selected) < target_count:
        raise RuntimeError(f"could only select {len(selected)} unseen valid rows, target={target_count}")
    return selected[:target_count]


def write_specs(out_dir: Path, *, text_repeat: int) -> dict[str, str]:
    no_text_train = out_dir / "no_text.train.jsonl"
    text_train = out_dir / "text.train.jsonl"
    no_text_valid = out_dir / "no_text.valid.jsonl"
    text_valid = out_dir / "text.valid.jsonl"
    no_text_seen = out_dir / "no_text.seen_valid.jsonl"
    text_seen = out_dir / "text.seen_valid.jsonl"
    no_text_unseen = out_dir / "no_text.unseen_valid.jsonl"
    text_unseen = out_dir / "text.unseen_valid.jsonl"
    specs = {
        "train_spec": f"{no_text_train.resolve()}::repeat=1,{text_train.resolve()}::repeat={int(text_repeat)}",
        "valid_spec": f"{no_text_valid.resolve()}::repeat=1,{text_valid.resolve()}::repeat=1",
        "valid_seen_spec": f"{no_text_seen.resolve()}::repeat=1,{text_seen.resolve()}::repeat=1",
        "valid_unseen_spec": f"{no_text_unseen.resolve()}::repeat=1,{text_unseen.resolve()}::repeat=1",
    }
    (out_dir / "mixed.train.spec.txt").write_text(specs["train_spec"] + "\n", encoding="utf-8")
    (out_dir / "mixed.valid.spec.txt").write_text(specs["valid_spec"] + "\n", encoding="utf-8")
    (out_dir / "mixed.valid_seen.spec.txt").write_text(specs["valid_seen_spec"] + "\n", encoding="utf-8")
    (out_dir / "mixed.valid_unseen.spec.txt").write_text(specs["valid_unseen_spec"] + "\n", encoding="utf-8")
    return specs


def ensure_outputs(out_dir: Path, overwrite: bool) -> None:
    targets = [
        "no_text.train.jsonl",
        "text.train.jsonl",
        "no_text.valid.jsonl",
        "text.valid.jsonl",
        "no_text.unseen_valid.jsonl",
        "text.unseen_valid.jsonl",
        "no_text.seen_valid.jsonl",
        "text.seen_valid.jsonl",
        "mixed.train.spec.txt",
        "mixed.valid.spec.txt",
        "mixed.valid_seen.spec.txt",
        "mixed.valid_unseen.spec.txt",
        "summary.json",
    ]
    for name in targets:
        path = out_dir / name
        if path.exists() and not overwrite:
            raise FileExistsError(f"output exists, pass --overwrite: {path}")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    split = summary["split"]
    audit = summary["speaker_audit"]
    lines = [
        "# Ver2.8 Full Speaker Split Rebuild",
        "",
        f"- output_dir: `{summary['output_dir']}`",
        f"- speaker source: `{audit['speaker_source_decision']}`",
        f"- ECAPA clustering: `{audit['ecapa_clustering']}`",
        f"- manifest sample ids resolved: {audit['lookup_sample_ids']}",
        f"- manifest unique speaker keys: {audit['unique_speakers_all_manifests']}",
        "",
        "## Split Counts",
        "",
        "| split | no-text rows | text rows | speakers | zh | en | short | medium | long |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("train", "seen_valid", "unseen_valid", "unseen_unused"):
        no_stats = split["no_text"][name]
        text_stats = split["text"][name]
        speakers = split["combined"][name]["speaker_count"]
        langs = Counter(no_stats["languages"]) + Counter(text_stats["languages"])
        bins = Counter(no_stats["duration_bins"]) + Counter(text_stats["duration_bins"])
        lines.append(
            f"| {name} | {no_stats['rows']} | {text_stats['rows']} | {speakers} | "
            f"{langs.get('zh', 0)} | {langs.get('en', 0)} | "
            f"{bins.get('short', 0)} | {bins.get('medium', 0)} | {bins.get('long', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Isolation Checks",
            "",
            f"- unseen valid speaker ratio: {split['unseen_speaker_ratio']:.6f} "
            f"(target about {split['target_speaker_ratio']:.4f})",
            f"- train/unseen speaker overlap: {split['train_unseen_speaker_overlap_count']}",
            f"- old seen valid rows excluded from train: {split['seen_valid_rows_excluded_from_train']}",
            f"- rows quarantined because they contain unseen speakers: "
            f"no-text {split['no_text']['unseen_unused']['rows']}, text {split['text']['unseen_unused']['rows']}",
            "",
            "## Notes",
            "",
            "- `no_text.valid.jsonl` and `text.valid.jsonl` are the new unseen-valid aliases.",
            "- `no_text.seen_valid.jsonl` and `text.seen_valid.jsonl` preserve the old valid rows for historical loss curves.",
            "- Manifest fields are usable, so ECAPA clustering was not run. The upstream ids include filename/vcdata-derived "
            "speaker surrogates; review this granularity before approving the final recipe run.",
            "",
            "## Specs",
            "",
        ]
    )
    for key, value in summary["outputs"].items():
        if key.endswith("_spec"):
            lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser()
    ensure_outputs(out_dir, bool(args.overwrite))
    manifest_paths = [Path(p).expanduser() for p in (args.speaker_manifest_jsonl or [str(p) for p in DEFAULT_MANIFESTS])]
    lookup, manifest_audit = load_manifest_lookup(manifest_paths)
    no_text_path = Path(args.no_text_jsonl).expanduser()
    text_path = Path(args.text_jsonl).expanduser()
    frame_rate = float(args.duration_frame_rate)

    seen_no_text_rows, seen_no_text_audit = load_rows_with_speakers(
        Path(args.old_seen_valid_no_text_jsonl).expanduser(),
        mode="no_text",
        split="seen_valid",
        lookup=lookup,
    )
    seen_text_rows, seen_text_audit = load_rows_with_speakers(
        Path(args.old_seen_valid_text_jsonl).expanduser(),
        mode="text",
        split="seen_valid",
        lookup=lookup,
    )

    seen_rows_all = seen_no_text_rows + seen_text_rows
    seen_keys = {row_key(row) for row in seen_rows_all}
    seen_speakers = set().union(*(speaker_set(row) for row in seen_rows_all)) if seen_rows_all else set()

    unseen_no_text, no_text_input_speakers, no_text_audit = scan_and_sample_unseen(
        no_text_path,
        mode="no_text",
        lookup=lookup,
        seen_keys=seen_keys,
        seen_speakers=seen_speakers,
        target_count=int(args.unseen_valid_no_text_count),
        seed=int(args.seed),
        frame_rate=frame_rate,
    )
    unseen_text, text_input_speakers, text_audit = scan_and_sample_unseen(
        text_path,
        mode="text",
        lookup=lookup,
        seen_keys=seen_keys,
        seen_speakers=seen_speakers,
        target_count=int(args.unseen_valid_text_count),
        seed=int(args.seed) + 17,
        frame_rate=frame_rate,
    )
    unseen_keys = {row_key(row) for row in unseen_no_text + unseen_text}
    unseen_speakers = set().union(*(speaker_set(row) for row in unseen_no_text + unseen_text)) if unseen_keys else set()

    unseen_no_text = mark_split(unseen_no_text, "unseen_valid")
    unseen_text = mark_split(unseen_text, "unseen_valid")
    seen_no_text_rows = mark_split(seen_no_text_rows, "seen_valid")
    seen_text_rows = mark_split(seen_text_rows, "seen_valid")

    paths = {
        "no_text_train": out_dir / "no_text.train.jsonl",
        "text_train": out_dir / "text.train.jsonl",
        "no_text_valid": out_dir / "no_text.valid.jsonl",
        "text_valid": out_dir / "text.valid.jsonl",
        "no_text_unseen_valid": out_dir / "no_text.unseen_valid.jsonl",
        "text_unseen_valid": out_dir / "text.unseen_valid.jsonl",
        "no_text_seen_valid": out_dir / "no_text.seen_valid.jsonl",
        "text_seen_valid": out_dir / "text.seen_valid.jsonl",
        "no_text_unseen_unused": out_dir / "temp/no_text.unseen_unused.jsonl",
        "text_unseen_unused": out_dir / "temp/text.unseen_unused.jsonl",
    }
    write_jsonl(paths["no_text_valid"], unseen_no_text)
    write_jsonl(paths["text_valid"], unseen_text)
    write_jsonl(paths["no_text_unseen_valid"], unseen_no_text)
    write_jsonl(paths["text_unseen_valid"], unseen_text)
    write_jsonl(paths["no_text_seen_valid"], seen_no_text_rows)
    write_jsonl(paths["text_seen_valid"], seen_text_rows)

    (
        no_text_train_stats,
        no_text_unused_stats,
        no_text_train_speakers,
        no_text_unused_speakers,
    ) = write_split_streaming(
        no_text_path,
        mode="no_text",
        lookup=lookup,
        train_path=paths["no_text_train"],
        unused_path=paths["no_text_unseen_unused"],
        seen_keys=seen_keys,
        unseen_keys=unseen_keys,
        unseen_speakers=unseen_speakers,
        frame_rate=frame_rate,
    )
    text_train_stats, text_unused_stats, text_train_speakers, text_unused_speakers = write_split_streaming(
        text_path,
        mode="text",
        lookup=lookup,
        train_path=paths["text_train"],
        unused_path=paths["text_unseen_unused"],
        seen_keys=seen_keys,
        unseen_keys=unseen_keys,
        unseen_speakers=unseen_speakers,
        frame_rate=frame_rate,
    )
    specs = write_specs(out_dir, text_repeat=int(args.text_repeat))

    offset_meta = {}
    for name, path in paths.items():
        offset_meta[name] = write_offsets(path)

    train_speakers = set(no_text_train_speakers) | set(text_train_speakers)
    seen_valid_speakers = set().union(*(speaker_set(row) for row in seen_no_text_rows + seen_text_rows))
    all_input_speakers = set(no_text_input_speakers) | set(text_input_speakers)
    train_unseen_overlap = train_speakers & unseen_speakers
    seen_keys_in_train: set[str] = set()

    seen_no_text_stats = split_stats(seen_no_text_rows, frame_rate=frame_rate)
    seen_text_stats = split_stats(seen_text_rows, frame_rate=frame_rate)
    unseen_no_text_stats = split_stats(unseen_no_text, frame_rate=frame_rate)
    unseen_text_stats = split_stats(unseen_text, frame_rate=frame_rate)
    combined_seen_stats = split_stats(seen_no_text_rows + seen_text_rows, frame_rate=frame_rate)
    combined_unseen_stats = split_stats(unseen_no_text + unseen_text, frame_rate=frame_rate)
    unused_speakers = set(no_text_unused_speakers) | set(text_unused_speakers)
    split_summary = {
        "target_speaker_ratio": float(args.target_speaker_ratio),
        "all_input_speaker_count": len(all_input_speakers),
        "train_speaker_count": len(train_speakers),
        "seen_valid_speaker_count": len(seen_valid_speakers),
        "unseen_valid_speaker_count": len(unseen_speakers),
        "unseen_speaker_ratio": len(unseen_speakers) / max(1, len(all_input_speakers)),
        "train_unseen_speaker_overlap_count": len(train_unseen_overlap),
        "train_unseen_speaker_overlap_examples": sorted(train_unseen_overlap)[:10],
        "seen_valid_rows_excluded_from_train": len(seen_keys) - len(seen_keys_in_train),
        "seen_valid_rows_found_in_train": len(seen_keys_in_train),
        "no_text": {
            "train": no_text_train_stats,
            "seen_valid": seen_no_text_stats,
            "unseen_valid": unseen_no_text_stats,
            "unseen_unused": no_text_unused_stats,
        },
        "text": {
            "train": text_train_stats,
            "seen_valid": seen_text_stats,
            "unseen_valid": unseen_text_stats,
            "unseen_unused": text_unused_stats,
        },
        "combined": {
            "train": combine_stats_dicts(no_text_train_stats, text_train_stats, speaker_count=len(train_speakers)),
            "seen_valid": combined_seen_stats,
            "unseen_valid": combined_unseen_stats,
            "unseen_unused": combine_stats_dicts(
                no_text_unused_stats,
                text_unused_stats,
                speaker_count=len(unused_speakers),
            ),
        },
    }

    if train_unseen_overlap:
        raise RuntimeError(f"unseen/train speaker overlap is nonzero: {sorted(train_unseen_overlap)[:10]}")
    if seen_keys_in_train:
        raise RuntimeError(f"seen-valid rows leaked into train: {sorted(seen_keys_in_train)[:10]}")

    outputs = {name: str(path) for name, path in paths.items()}
    outputs.update(specs)
    summary = {
        "status": "complete",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_dir": str(out_dir),
        "seed": int(args.seed),
        "text_repeat": int(args.text_repeat),
        "speaker_audit": {
            **manifest_audit,
            "prepared_no_text": no_text_audit,
            "prepared_text": text_audit,
            "seen_valid_no_text": seen_no_text_audit,
            "seen_valid_text": seen_text_audit,
            "speaker_source_decision": "manifest_sample_id",
            "speaker_granularity_caveat": (
                "Upstream speaker ids are parseable but include filename_hash/vcdata-derived surrogate ids."
            ),
            "ecapa_clustering": "skipped_manifest_speaker_fields_resolved_all_rows",
        },
        "split": split_summary,
        "outputs": outputs,
        "offsets": offset_meta,
    }
    write_json(out_dir / "summary.json", summary)
    write_report(Path(args.docs_md).expanduser(), summary)
    print(
        "[speaker-split] "
        f"no_text train={no_text_train_stats['rows']} unseen={len(unseen_no_text)} seen={len(seen_no_text_rows)} "
        f"unused={no_text_unused_stats['rows']}; "
        f"text train={text_train_stats['rows']} unseen={len(unseen_text)} seen={len(seen_text_rows)} "
        f"unused={text_unused_stats['rows']}",
        flush=True,
    )
    print(
        "[speaker-split] "
        f"speakers all={len(all_input_speakers)} train={len(train_speakers)} "
        f"unseen={len(unseen_speakers)} ratio={split_summary['unseen_speaker_ratio']:.6f} "
        f"train_unseen_overlap={len(train_unseen_overlap)}",
        flush=True,
    )
    print(f"[speaker-split] summary={out_dir / 'summary.json'}", flush=True)
    print(f"[speaker-split] docs={Path(args.docs_md).expanduser()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

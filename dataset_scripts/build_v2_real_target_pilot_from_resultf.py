#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RESULTF_DIR = Path(
    "/inspire/hdd/project/embodied-multimodality/public/btjiang/VoiceClone/resultf"
)
DEFAULT_PREPARE_DIR = Path(
    "/inspire/hdd/project/embodied-multimodality/public/btjiang/VoiceClone/prepare"
)
DEFAULT_META_DIR = Path(
    "/inspire/hdd/project/embodied-multimodality/public/btjiang/VoiceClone/meta"
)
DEFAULT_NEWTRAIN_DIR = Path(
    "/inspire/hdd/project/embodied-multimodality/public/btjiang/WinterisComing/Data/newtrain"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/inspire/qb-ilm/project/embodied-multimodality/public/xyzhang/data/"
    "VC_train/v2_real_target_pilot_20260706"
)
DEFAULT_FEISHU_TABLE_CSV = Path(
    "/inspire/hdd/project/embodied-multimodality/public/btjiang/misc/dataassistant/data/PREDATATOTAL.csv"
)
LOCAL_TOKENIZE_ROOT = Path(
    "/inspire/qb-ilm/project/embodied-multimodality/public/speech_generation/data/tokenize"
)


NO_TEXT_INSTRUCTION = (
    "Voice conversion task. [S1] is the source speech carrying content, pauses, duration and prosody. "
    "[S2] is the target timbre reference. Generate the same content as S1 with S2 timbre while preserving "
    "S1 timing and prosody."
)
TEXT_INSTRUCTION = (
    "Text-guided voice conversion task. Use the provided text as lexical content. "
    "[S1] carries rhythm, pauses, speaking rate and duration hints. [S2] is the target timbre reference. "
    "Generate speech whose lexical content follows the text and whose speaker identity follows [S2]."
)


DATASET_TABLES: dict[str, str] = {
    "apple_podcast_ar": "dwd_audio_apple_podcast_ar_denoise_stdz_diariz_tts_segs_langdetect_speechscore",
    "apple_podcast_au_1": "dwd_audio_apple_podcast_au_1_denoise_stdz_diariz_tts_segs_langdetect_speechscore",
    "apple_podcast_au_2": "dwd_audio_apple_podcast_au_2_denoise_stdz_diariz_tts_segs_langdetect_speechscore",
    "apple_podcast_au_3": "dwd_audio_apple_podcast_au_3_denoise_stdz_diariz_tts_segs_langdetect_speechscore",
    "apple_podcast_estwfiruchnonltzph": (
        "dwd_audio_apple_podcast_estwfiruchnonltzph_denoise_stdz_diariz_tts_raw_segs_langdetect_speechscore"
    ),
    "apple_podcast_josailczinplittrropkthmauavelyye": (
        "dwd_audio_apple_podcast_josailczinplittrropkthmauavelyye_denoise_stdz_diariz_tts_raw_segs_langdetect_speechscore"
    ),
    "apple_podcast_vnjpidgrfrkemyptpededk": (
        "dwd_audio_apple_podcast_vnjpidgrfrkemyptpededk_denoise_stdz_diariz_tts_raw_segs_langdetect_speechscore"
    ),
    "haitianruisheng_1": "dwd_audio_haitianruisheng_1_v3_denoise_stdz_diariz_tts_raw_segs_langdetect_speechscore",
    "haitianruisheng_2": "dwd_audio_haitianruisheng_2_v2_denoise_stdz_diariz_tts_raw_segs_langdetect_speechscore",
    "haitianruisheng_3": "dwd_audio_haitianruisheng_3_v3_denoise_stdz_diariz_tts_raw_segs_langdetect_speechscore",
    "haitianruisheng_4": "dwd_audio_haitianruisheng_4_denoise_stdz_diariz_tts_raw_segs_langdetect_speechscore",
    "haitianruisheng_6": "dwd_audio_haitianruisheng_6_denoise_stdz_diariz_tts_raw_segs_langdetect_speechscore",
    "haitianruisheng_7": "dwd_audio_haitianruisheng_7_v3_denoise_stdz_diariz_tts_raw_segs_langdetect_speechscore",
    "haitianruisheng_8": "dwd_audio_haitianruisheng_8_v3_denoise_stdz_diariz_tts_raw_segs_langdetect_speechscore",
    "haitianruisheng_9": "dwd_audio_haitianruisheng_9_v3_denoise_stdz_diariz_tts_raw_segs_langdetect_speechscore",
    "qingting_fm": "dwd_audio_qingting_fm_denoise_stdz_diariz_tts_raw_segs_langdetect_speechscore",
    "rchive_rss_podcast_v2": "dwd_audio_rchive_rss_podcast_v2_denoise_stdz_diariz_tts_raw_segs_langdetect_speechscore",
}

DATASET_LANGUAGE_HINTS: dict[str, set[str]] = {
    "apple_podcast_estwfiruchnonltzph": {"en"},
    "apple_podcast_josailczinplittrropkthmauavelyye": {"en"},
    "apple_podcast_vnjpidgrfrkemyptpededk": {"en"},
    "haitianruisheng_1": {"en"},
    "haitianruisheng_2": {"zh"},
    "haitianruisheng_3": {"zh"},
    "haitianruisheng_4": {"zh"},
    "haitianruisheng_6": {"en"},
    "haitianruisheng_7": {"en"},
    "haitianruisheng_8": {"en"},
    "haitianruisheng_9": {"en"},
    "qingting_fm": {"zh"},
    "rchive_rss_podcast_v2": {"en"},
}


@dataclass(frozen=True)
class DatasetMapping:
    dataset_name: str
    table_name: str
    lance_table_uri: str
    feishu_record_id: str
    feishu_table_csv: str
    newtrain_jsonl: str
    local_tokenize_root: str
    local_tokenize_exists: bool


@dataclass
class PairRow:
    dataset_name: str
    source_resultf_jsonl: str
    source_prepare_jsonl: str
    source_line_index: int
    lance_table_uri: str
    label_relative_path: str
    reference_relative_path: str
    target_audio: str
    timbre_ref_audio: str
    target_audio_source_uri: str
    timbre_ref_audio_source_uri: str
    target_audio_materialized: bool
    timbre_ref_audio_materialized: bool
    target_text: str
    timbre_ref_text: str
    language: str
    similarity: float
    target_duration_sec: float | None
    timbre_ref_duration_sec: float | None
    source_episode_id: str
    timbre_ref_episode_id: str
    source_speaker_id: str
    timbre_ref_speaker_id: str
    source_metadata_id: str
    timbre_ref_metadata_id: str
    label_idx: int | None
    ref_idx: int | None
    segs: Any


def stable_id(*values: Any, length: int = 16) -> str:
    payload = "\x1f".join(str(value) for value in values)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(path)
    return count


def write_mapping_tsv(path: Path, mappings: dict[str, DatasetMapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(
            "dataset_name\ttable_name\tlance_table_uri\tfeishu_record_id\t"
            "local_tokenize_exists\tlocal_tokenize_root\tnewtrain_jsonl\n"
        )
        for dataset_name in sorted(mappings):
            mapping = mappings[dataset_name]
            handle.write(
                "\t".join(
                    [
                        mapping.dataset_name,
                        mapping.table_name,
                        mapping.lance_table_uri,
                        mapping.feishu_record_id,
                        str(mapping.local_tokenize_exists),
                        mapping.local_tokenize_root,
                        mapping.newtrain_jsonl,
                    ]
                )
                + "\n"
            )
    tmp.replace(path)


def load_feishu_lance_table_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Feishu table CSV not found: {path}")

    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            table_name = str(row.get("数据集名称") or "").strip()
            lance_uri = str(row.get("lance 信息表路径") or "").strip()
            if not table_name or not lance_uri:
                continue
            rows[table_name] = {
                "lance_table_uri": lance_uri.rstrip("/"),
                "feishu_record_id": str(row.get("record_id") or "").strip(),
            }
    return rows


def dataset_mapping(
    dataset_name: str,
    newtrain_dir: Path,
    feishu_table_csv: Path,
    feishu_rows: dict[str, dict[str, str]],
) -> DatasetMapping:
    table_name = DATASET_TABLES[dataset_name]
    feishu_row = feishu_rows.get(table_name)
    if not feishu_row:
        raise SystemExit(f"missing lance table path for {dataset_name} / {table_name} in Feishu CSV")
    local_root = LOCAL_TOKENIZE_ROOT / table_name
    return DatasetMapping(
        dataset_name=dataset_name,
        table_name=table_name,
        lance_table_uri=feishu_row["lance_table_uri"],
        feishu_record_id=feishu_row["feishu_record_id"],
        feishu_table_csv=path_resolve(feishu_table_csv),
        newtrain_jsonl=str((newtrain_dir / f"{table_name}.jsonl").resolve(strict=False)),
        local_tokenize_root=str(local_root.resolve(strict=False)),
        local_tokenize_exists=local_root.exists(),
    )


def path_resolve(path: Path) -> str:
    return str(path.expanduser().resolve(strict=False))


def resolve_lance_audio_uri(mapping: DatasetMapping, relative_path: str) -> str:
    value = str(relative_path)
    if value.startswith(("s3://", "tos://", "oss://", "qz_oss2://", "http://", "https://", "/")):
        return value
    if value.startswith("speech.db/"):
        table_prefix = f"speech.db/{mapping.table_name}.lance/"
        if value.startswith(table_prefix):
            return f"{mapping.lance_table_uri.rstrip('/')}/{value[len(table_prefix):].lstrip('/')}"
        lance_parent = mapping.lance_table_uri.split("/speech.db/", 1)[0]
        return f"{lance_parent.rstrip('/')}/{value.lstrip('/')}"
    return f"{mapping.lance_table_uri.rstrip('/')}/{value.lstrip('/')}"


def rclone_path_from_uri(uri: str) -> str:
    if uri.startswith(("qz_oss2://", "qz_oss://")):
        scheme, rest = uri.split("://", 1)
        return f"{scheme}:{rest.lstrip('/')}"
    raise ValueError(f"unsupported rclone URI: {uri}")


def materialized_relative_path(mapping: DatasetMapping, relative_path: str) -> str:
    value = str(relative_path).lstrip("/")
    table_prefix = f"speech.db/{mapping.table_name}.lance/"
    if value.startswith(table_prefix):
        value = value[len(table_prefix) :]
    parts = Path(value).parts
    if not value or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe relative audio path: {relative_path}")
    return value


def rclone_download_env(disable_proxy: bool) -> dict[str, str]:
    env = os.environ.copy()
    if disable_proxy:
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
            env.pop(key, None)
        env["NO_PROXY"] = "*"
    return env


def materialize_pair_audio(
    pairs: list[PairRow],
    mappings: dict[str, DatasetMapping],
    *,
    output_root: Path,
    materialized_audio_dir: Path | None,
    download_audio: bool,
    rclone_bin: str,
    rclone_transfers: int,
    rclone_checkers: int,
    rclone_disable_proxy: bool,
) -> dict[str, Any]:
    audio_root = materialized_audio_dir or (output_root / "materialized_lance_audio")
    audio_root = audio_root.expanduser().resolve(strict=False)
    filelist_dir = output_root / "download_filelists"
    filelist_dir.mkdir(parents=True, exist_ok=True)

    rels_by_dataset: dict[str, set[str]] = defaultdict(set)
    manifest_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    local_path_by_key: dict[tuple[str, str], str] = {}

    for row in pairs:
        mapping = mappings[row.dataset_name]
        for role, relative_path, source_uri in (
            ("u1_target", row.label_relative_path, row.target_audio_source_uri),
            ("u2_timbre_ref", row.reference_relative_path, row.timbre_ref_audio_source_uri),
        ):
            rel = materialized_relative_path(mapping, relative_path)
            local_path = audio_root / row.dataset_name / rel
            key = (row.dataset_name, rel)
            rels_by_dataset[row.dataset_name].add(rel)
            local_path_by_key[key] = str(local_path.resolve(strict=False))
            manifest_by_key.setdefault(
                key,
                {
                    "dataset_name": row.dataset_name,
                    "role_first_seen": role,
                    "lance_table_uri": mapping.lance_table_uri,
                    "source_uri": f"{mapping.lance_table_uri.rstrip('/')}/{rel}",
                    "relative_path": rel,
                    "local_path": str(local_path.resolve(strict=False)),
                },
            )

        target_rel = materialized_relative_path(mapping, row.label_relative_path)
        ref_rel = materialized_relative_path(mapping, row.reference_relative_path)
        row.target_audio = local_path_by_key[(row.dataset_name, target_rel)]
        row.timbre_ref_audio = local_path_by_key[(row.dataset_name, ref_rel)]
        row.target_audio_materialized = True
        row.timbre_ref_audio_materialized = True

    download_manifest = sorted(manifest_by_key.values(), key=lambda item: (item["dataset_name"], item["relative_path"]))
    download_manifest_path = output_root / "audio_download_manifest.jsonl"
    write_jsonl(download_manifest_path, download_manifest)

    filelist_paths: dict[str, str] = {}
    commands: list[list[str]] = []
    if download_audio:
        rclone_path = shutil.which(rclone_bin) or rclone_bin
        if not shutil.which(rclone_path) and not Path(rclone_path).exists():
            raise SystemExit(f"rclone binary not found: {rclone_bin}")
        env = rclone_download_env(rclone_disable_proxy)
        for dataset_name in sorted(rels_by_dataset):
            mapping = mappings[dataset_name]
            filelist_path = filelist_dir / f"{dataset_name}.txt"
            filelist_path.write_text("\n".join(sorted(rels_by_dataset[dataset_name])) + "\n", encoding="utf-8")
            filelist_paths[dataset_name] = str(filelist_path.resolve(strict=False))
            local_dataset_root = audio_root / dataset_name
            local_dataset_root.mkdir(parents=True, exist_ok=True)
            cmd = [
                rclone_path,
                "copy",
                rclone_path_from_uri(mapping.lance_table_uri),
                str(local_dataset_root),
                "--files-from",
                str(filelist_path),
                "--ignore-existing",
                "--transfers",
                str(rclone_transfers),
                "--checkers",
                str(rclone_checkers),
                "--no-traverse",
                "--stats",
                "30s",
            ]
            commands.append(cmd)
            print(f"materializing {dataset_name}: {len(rels_by_dataset[dataset_name])} files")
            result = subprocess.run(cmd, env=env, text=True)
            if result.returncode != 0:
                raise SystemExit(f"rclone failed for {dataset_name}: exit={result.returncode}")

    missing = [item for item in download_manifest if not Path(item["local_path"]).exists()]
    if download_audio and missing:
        missing_preview = [item["local_path"] for item in missing[:10]]
        raise SystemExit(f"missing {len(missing)} materialized audio files after rclone; examples={missing_preview}")

    return {
        "materialized_audio_root": str(audio_root),
        "download_audio": download_audio,
        "download_manifest": str(download_manifest_path.resolve(strict=False)),
        "filelist_dir": str(filelist_dir.resolve(strict=False)),
        "filelists": filelist_paths,
        "unique_audio_files": len(download_manifest),
        "missing_after_download": len(missing),
        "rclone_disable_proxy": rclone_disable_proxy,
        "rclone_commands": [" ".join(cmd) for cmd in commands],
    }


SEGMENT_RE = re.compile(r"segment_(?P<start>\d+(?:_\d+)?)-(?P<end>\d+(?:_\d+)?)\.flac$")


def parse_segment_duration(path: str) -> float | None:
    match = SEGMENT_RE.search(path)
    if not match:
        return None
    start = float(match.group("start").replace("_", "."))
    end = float(match.group("end").replace("_", "."))
    duration = end - start
    return duration if duration >= 0 else None


def episode_id(path: str) -> str:
    marker = "/tts_audio_segment/"
    value = path
    if marker in value:
        value = value.split(marker, 1)[1]
    parts = [part for part in value.split("/") if part]
    if len(parts) >= 2:
        return parts[0]
    return Path(path).parent.name or "unknown_episode"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


EN_STOPWORDS = {
    "a",
    "about",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "by",
    "can",
    "do",
    "for",
    "from",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "is",
    "it",
    "like",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "was",
    "we",
    "what",
    "with",
    "you",
    "your",
}
EN_CORE_STOPWORDS = EN_STOPWORDS - {"a", "an", "i", "is"}


def infer_text_language(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return "unknown"
    cjk_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    letters = [ch for ch in text if ch.isalpha()]
    if cjk_count >= 2 and cjk_count / max(1, len(letters)) >= 0.25:
        return "zh"

    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())
    if not tokens:
        return "unknown"
    non_ascii_letters = sum(1 for ch in letters if ord(ch) > 127 and not ("\u4e00" <= ch <= "\u9fff"))
    ascii_letter_count = sum(1 for ch in letters if "A" <= ch <= "Z" or "a" <= ch <= "z")
    if non_ascii_letters / max(1, len(letters)) > 0.03:
        return "other"
    if ascii_letter_count / max(1, len(letters)) < 0.95:
        return "other"
    stopword_hits = sum(1 for token in tokens if token in EN_STOPWORDS)
    core_hits = sum(1 for token in tokens if token in EN_CORE_STOPWORDS)
    if core_hits >= 2 and stopword_hits >= 3 and stopword_hits / max(1, len(tokens)) >= 0.08:
        return "en"
    if len(tokens) <= 6 and core_hits >= 1 and stopword_hits >= 2:
        return "en"
    return "other"


def infer_language(text: str, row: dict[str, Any] | None = None) -> str:
    if row:
        lang = str(row.get("lang") or row.get("language") or "").strip().lower()
        if lang.startswith("zh") or lang in {"cmn", "zho", "chinese", "mandarin"}:
            return "zh"
        if lang.startswith("en") or lang in {"eng", "english"}:
            return "en"
    return infer_text_language(text)


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def row_passes_filters(
    row: dict[str, Any],
    *,
    min_similarity: float,
    target_min_sec: float,
    target_max_sec: float,
    ref_min_sec: float,
    ref_max_sec: float,
    require_known_duration: bool,
    require_text: bool,
    stats: Counter,
) -> bool:
    label = str(row.get("label") or "")
    reference = str(row.get("reference") or "")
    if not label or not reference or label == reference:
        stats["skip_bad_pair_paths"] += 1
        return False
    sim = as_float(row.get("similarity"))
    if sim is None or sim < min_similarity:
        stats["skip_low_similarity"] += 1
        return False
    text = normalize_text(row.get("text"))
    if require_text and not text:
        stats["skip_empty_text"] += 1
        return False
    target_duration = parse_segment_duration(label)
    ref_duration = parse_segment_duration(reference)
    if target_duration is None or ref_duration is None:
        stats["unknown_duration"] += 1
        if require_known_duration:
            stats["skip_unknown_duration"] += 1
            return False
    if target_duration is not None and not (target_min_sec <= target_duration <= target_max_sec):
        stats["skip_target_duration"] += 1
        return False
    if ref_duration is not None and ref_duration < ref_min_sec:
        stats["skip_ref_duration"] += 1
        return False
    if ref_duration is not None and ref_max_sec > 0 and ref_duration > ref_max_sec:
        stats["skip_ref_duration"] += 1
        return False
    return True


def discover_dataset_files(prepare_dir: Path, only_datasets: set[str]) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for path in sorted(prepare_dir.glob("*.jsonl")):
        dataset_name = path.stem
        if dataset_name not in DATASET_TABLES:
            continue
        if only_datasets and dataset_name not in only_datasets:
            continue
        found.append((dataset_name, path))
    return found


def dataset_has_parseable_segments(path: Path, probe_rows: int) -> bool:
    for row_idx, (_, row) in enumerate(iter_jsonl(path)):
        if row_idx >= probe_rows:
            break
        label = str(row.get("label") or "")
        reference = str(row.get("reference") or "")
        if parse_segment_duration(label) is not None and parse_segment_duration(reference) is not None:
            return True
    return False


def select_pairs(args: argparse.Namespace, mappings: dict[str, DatasetMapping]) -> tuple[list[PairRow], dict[str, Any]]:
    files = discover_dataset_files(Path(args.prepare_dir), set(args.datasets or []))
    if not files:
        raise SystemExit(f"no prepare JSONLs found under {args.prepare_dir}")
    skipped_no_parseable_duration: list[str] = []
    if args.require_known_duration:
        filtered_files: list[tuple[str, Path]] = []
        for dataset_name, prepare_path in files:
            if dataset_has_parseable_segments(prepare_path, args.duration_probe_rows):
                filtered_files.append((dataset_name, prepare_path))
            else:
                skipped_no_parseable_duration.append(dataset_name)
        files = filtered_files
    if not files:
        raise SystemExit("no datasets left after duration probe filtering")

    requested = max(args.num_no_text, args.num_text)
    quota = args.max_per_dataset
    if quota <= 0:
        quota = max(1, math.ceil(requested / max(1, len(files))))

    allowed_languages = set(args.languages or [])
    language_quotas: dict[str, int] = {}
    language_counts: Counter = Counter()
    if args.balance_languages:
        if not allowed_languages:
            raise SystemExit("--balance-languages requires --languages")
        base = requested // len(args.languages)
        remainder = requested % len(args.languages)
        language_quotas = {
            language: base + (1 if idx < remainder else 0)
            for idx, language in enumerate(args.languages)
        }

    def language_allowed(language: str, stats: Counter) -> bool:
        if allowed_languages and language not in allowed_languages:
            stats[f"skip_language_{language or 'unknown'}"] += 1
            return False
        if language_quotas and language_counts[language] >= language_quotas.get(language, 0):
            stats[f"skip_language_quota_full_{language}"] += 1
            return False
        return True

    def all_requested_languages_full() -> bool:
        if not language_quotas:
            return False
        return all(language_counts[language] >= quota for language, quota in language_quotas.items())

    def dataset_can_help(dataset_name: str) -> bool:
        if not language_quotas:
            return True
        remaining = {
            language
            for language, quota in language_quotas.items()
            if language_counts[language] < quota
        }
        hints = DATASET_LANGUAGE_HINTS.get(dataset_name)
        if not hints:
            return True
        return bool(hints & remaining)

    def make_pair(
        dataset_name: str,
        mapping: DatasetMapping,
        prepare_path: Path,
        resultf_path: Path,
        line_no: int,
        row: dict[str, Any],
        language: str,
    ) -> PairRow:
        label = str(row["label"])
        reference = str(row["reference"])
        target_text = normalize_text(row.get("text"))
        source_episode = episode_id(label)
        ref_episode = episode_id(reference)
        target_audio_uri = resolve_lance_audio_uri(mapping, label)
        timbre_ref_audio_uri = resolve_lance_audio_uri(mapping, reference)
        return PairRow(
            dataset_name=dataset_name,
            source_resultf_jsonl=str(resultf_path.resolve(strict=False)),
            source_prepare_jsonl=str(prepare_path.resolve(strict=False)),
            source_line_index=line_no,
            lance_table_uri=mapping.lance_table_uri,
            label_relative_path=label,
            reference_relative_path=reference,
            target_audio=target_audio_uri,
            timbre_ref_audio=timbre_ref_audio_uri,
            target_audio_source_uri=target_audio_uri,
            timbre_ref_audio_source_uri=timbre_ref_audio_uri,
            target_audio_materialized=False,
            timbre_ref_audio_materialized=False,
            target_text=target_text,
            timbre_ref_text="",
            language=language,
            similarity=float(row["similarity"]),
            target_duration_sec=parse_segment_duration(label),
            timbre_ref_duration_sec=parse_segment_duration(reference),
            source_episode_id=source_episode,
            timbre_ref_episode_id=ref_episode,
            source_speaker_id=f"{dataset_name}:{source_episode}",
            timbre_ref_speaker_id=f"{dataset_name}:{ref_episode}",
            source_metadata_id="",
            timbre_ref_metadata_id="",
            label_idx=as_int(row.get("label_idx")),
            ref_idx=as_int(row.get("ref_idx")),
            segs=row.get("segs"),
        )

    selected: list[PairRow] = []
    stats_by_dataset: dict[str, dict[str, int]] = {}
    seen_pairs: set[tuple[str, str, str]] = set()
    for dataset_name, prepare_path in files:
        if len(selected) >= requested or all_requested_languages_full():
            break
        if not dataset_can_help(dataset_name):
            stats_by_dataset[dataset_name] = {"skip_dataset_language_quota_full": 1}
            continue
        mapping = mappings[dataset_name]
        stats: Counter = Counter()
        kept = 0
        resultf_path = Path(args.resultf_dir) / f"{dataset_name}.jsonl"
        for line_no, row in iter_jsonl(prepare_path):
            stats["rows_read"] += 1
            if len(selected) >= requested or all_requested_languages_full():
                break
            if kept >= quota:
                break
            if not row_passes_filters(
                row,
                min_similarity=args.min_similarity,
                target_min_sec=args.target_min_sec,
                target_max_sec=args.target_max_sec,
                ref_min_sec=args.ref_min_sec,
                ref_max_sec=args.ref_max_sec,
                require_known_duration=args.require_known_duration,
                require_text=args.require_text,
                stats=stats,
            ):
                continue
            label = str(row["label"])
            reference = str(row["reference"])
            pair_key = (dataset_name, label, reference)
            if pair_key in seen_pairs:
                stats["skip_duplicate_pair"] += 1
                continue
            target_text = normalize_text(row.get("text"))
            language = infer_language(target_text, row)
            if not language_allowed(language, stats):
                continue
            seen_pairs.add(pair_key)
            selected.append(make_pair(dataset_name, mapping, prepare_path, resultf_path, line_no, row, language))
            language_counts[language] += 1
            kept += 1
            stats["kept"] += 1
        stats_by_dataset[dataset_name] = dict(stats)
        if len(selected) >= requested or all_requested_languages_full():
            break

    if len(selected) < requested and not all_requested_languages_full():
        for dataset_name, prepare_path in files:
            if len(selected) >= requested or all_requested_languages_full():
                break
            if not dataset_can_help(dataset_name):
                stats_by_dataset[dataset_name] = {
                    **stats_by_dataset.get(dataset_name, {}),
                    "fill_skip_dataset_language_quota_full": 1,
                }
                continue
            mapping = mappings[dataset_name]
            stats = Counter(stats_by_dataset.get(dataset_name, {}))
            resultf_path = Path(args.resultf_dir) / f"{dataset_name}.jsonl"
            for line_no, row in iter_jsonl(prepare_path):
                if len(selected) >= requested or all_requested_languages_full():
                    break
                stats["fill_rows_read"] += 1
                if not row_passes_filters(
                    row,
                    min_similarity=args.min_similarity,
                    target_min_sec=args.target_min_sec,
                    target_max_sec=args.target_max_sec,
                    ref_min_sec=args.ref_min_sec,
                    ref_max_sec=args.ref_max_sec,
                    require_known_duration=args.require_known_duration,
                    require_text=args.require_text,
                    stats=stats,
                ):
                    continue
                label = str(row["label"])
                reference = str(row["reference"])
                pair_key = (dataset_name, label, reference)
                if pair_key in seen_pairs:
                    stats["fill_skip_duplicate_pair"] += 1
                    continue
                target_text = normalize_text(row.get("text"))
                language = infer_language(target_text, row)
                if not language_allowed(language, stats):
                    continue
                seen_pairs.add(pair_key)
                selected.append(make_pair(dataset_name, mapping, prepare_path, resultf_path, line_no, row, language))
                language_counts[language] += 1
                stats["fill_kept"] += 1
            stats_by_dataset[dataset_name] = dict(stats)
            if len(selected) >= requested or all_requested_languages_full():
                break

    if len(selected) < requested:
        raise SystemExit(
            f"only selected {len(selected)} pairs, requested {requested}; "
            "relax filters or lower --num-no-text/--num-text"
        )
    selected = selected[:requested]
    summary = {
        "requested_pairs": requested,
        "selected_pairs": len(selected),
        "per_dataset_quota": quota,
        "allowed_languages": args.languages,
        "balance_languages": args.balance_languages,
        "language_quotas": language_quotas,
        "language_counts": dict(language_counts),
        "dataset_language_hints": {key: sorted(value) for key, value in DATASET_LANGUAGE_HINTS.items()},
        "skipped_no_parseable_duration": skipped_no_parseable_duration,
        "stats_by_dataset": stats_by_dataset,
    }
    return selected, summary


def load_newtrain_info(
    pairs: list[PairRow],
    mappings: dict[str, DatasetMapping],
) -> dict[str, dict[str, dict[str, Any]]]:
    wanted: dict[str, set[str]] = defaultdict(set)
    for row in pairs:
        wanted[row.dataset_name].add(row.label_relative_path)
        wanted[row.dataset_name].add(row.reference_relative_path)

    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for dataset_name, wanted_paths in wanted.items():
        mapping = mappings[dataset_name]
        path = Path(mapping.newtrain_jsonl)
        if not path.exists():
            continue
        remaining = set(wanted_paths)
        for _, row in iter_jsonl(path):
            rel = str(row.get("audio_segment_path") or "")
            if rel not in remaining:
                continue
            out[dataset_name][rel] = row
            remaining.remove(rel)
            if not remaining:
                break
    return out


def load_meta_info(pairs: list[PairRow], meta_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    wanted: dict[str, set[str]] = defaultdict(set)
    for row in pairs:
        wanted[row.dataset_name].add(row.label_relative_path)
        wanted[row.dataset_name].add(row.reference_relative_path)

    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for dataset_name, wanted_paths in wanted.items():
        path = meta_dir / f"{dataset_name}.jsonl"
        if not path.exists():
            continue
        remaining = set(wanted_paths)
        for _, row in iter_jsonl(path):
            rel = str(row.get("audio_segment_path") or "")
            if rel not in remaining:
                continue
            out[dataset_name][rel] = row
            remaining.remove(rel)
            if not remaining:
                break
    return out


def enrich_pairs(
    pairs: list[PairRow],
    newtrain_info: dict[str, dict[str, dict[str, Any]]],
    meta_info: dict[str, dict[str, dict[str, Any]]],
) -> Counter:
    stats: Counter = Counter()
    for row in pairs:
        target_info = newtrain_info.get(row.dataset_name, {}).get(row.label_relative_path, {})
        ref_info = newtrain_info.get(row.dataset_name, {}).get(row.reference_relative_path, {})
        if not row.target_text:
            row.target_text = normalize_text(target_info.get("asr_text"))
        ref_text = normalize_text(ref_info.get("asr_text"))
        if ref_text:
            row.timbre_ref_text = ref_text
        if row.target_duration_sec is None:
            row.target_duration_sec = as_float(target_info.get("duration"))
        if row.timbre_ref_duration_sec is None:
            row.timbre_ref_duration_sec = as_float(ref_info.get("duration"))
        row.language = infer_language(row.target_text, target_info)

        target_meta = meta_info.get(row.dataset_name, {}).get(row.label_relative_path, {})
        ref_meta = meta_info.get(row.dataset_name, {}).get(row.reference_relative_path, {})
        if target_meta:
            row.source_metadata_id = str(target_meta.get("metadata_id") or "")
            target_spk = str(target_meta.get("speaker_id") or "").strip()
            if target_spk:
                row.source_speaker_id = f"{row.dataset_name}:{row.source_metadata_id}:{target_spk}"
        else:
            stats["missing_target_meta"] += 1
        if ref_meta:
            row.timbre_ref_metadata_id = str(ref_meta.get("metadata_id") or "")
            ref_spk = str(ref_meta.get("speaker_id") or "").strip()
            if ref_spk:
                row.timbre_ref_speaker_id = f"{row.dataset_name}:{row.timbre_ref_metadata_id}:{ref_spk}"
        else:
            stats["missing_ref_meta"] += 1
        if not row.timbre_ref_text:
            stats["missing_timbre_ref_text"] += 1
        if row.source_speaker_id != row.timbre_ref_speaker_id:
            stats["speaker_id_mismatch_after_meta"] += 1
    return stats


def fill_reference_text_from_selected_pairs(pairs: list[PairRow]) -> Counter:
    text_by_audio: dict[tuple[str, str], str] = {}
    for row in pairs:
        if row.target_text:
            text_by_audio[(row.dataset_name, row.label_relative_path)] = row.target_text

    stats: Counter = Counter()
    for row in pairs:
        if row.timbre_ref_text:
            continue
        text = text_by_audio.get((row.dataset_name, row.reference_relative_path))
        if text:
            row.timbre_ref_text = text
            stats["filled_timbre_ref_text_from_selected_label"] += 1
        else:
            stats["missing_timbre_ref_text_in_selected_labels"] += 1
    return stats


def choose_perturb_donors(pairs: list[PairRow], seed: int) -> dict[int, PairRow]:
    rng = random.Random(seed)
    by_language: dict[str, list[PairRow]] = defaultdict(list)
    for row in pairs:
        by_language[row.language].append(row)
    all_rows = list(pairs)
    donors: dict[int, PairRow] = {}

    def valid_donor(row: PairRow, candidate: PairRow, *, require_cross_dataset: bool) -> bool:
        if require_cross_dataset and candidate.dataset_name == row.dataset_name:
            return False
        return (
            candidate.source_speaker_id != row.source_speaker_id
            and candidate.target_audio != row.target_audio
            and candidate.timbre_ref_audio != row.timbre_ref_audio
        )

    def pick_from_pool(row: PairRow, pool: list[PairRow], *, require_cross_dataset: bool) -> PairRow | None:
        if not pool:
            return None
        for _ in range(min(128, max(16, len(pool)))):
            candidate = rng.choice(pool)
            if valid_donor(row, candidate, require_cross_dataset=require_cross_dataset):
                return candidate
        step = rng.randrange(len(pool))
        for offset in range(len(pool)):
            candidate = pool[(step + offset) % len(pool)]
            if valid_donor(row, candidate, require_cross_dataset=require_cross_dataset):
                return candidate
        return None

    for idx, row in enumerate(pairs):
        pool = by_language.get(row.language) or all_rows
        donor = pick_from_pool(row, pool, require_cross_dataset=True)
        if donor is None and pool is not all_rows:
            donor = pick_from_pool(row, all_rows, require_cross_dataset=True)
        if donor is None:
            donor = pick_from_pool(row, pool, require_cross_dataset=False)
        if donor is None and pool is not all_rows:
            donor = pick_from_pool(row, all_rows, require_cross_dataset=False)
        donors[idx] = donor or rng.choice(all_rows)
    return donors


def source_output_audio(output_root: Path, row: PairRow, ordinal: int) -> str:
    digest = stable_id(row.dataset_name, row.label_relative_path, row.reference_relative_path, ordinal, length=12)
    return str((output_root / "seedvc_sources" / row.dataset_name / f"{ordinal:08d}_{digest}.wav").resolve(strict=False))


def build_source_jobs(
    pairs: list[PairRow],
    donors: dict[int, PairRow],
    *,
    output_root: Path,
    run_name: str,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for idx, row in enumerate(pairs):
        donor = donors[idx]
        output_audio = source_output_audio(output_root, row, idx)
        job_id = f"{run_name}:source_seedvc:{idx:08d}:{stable_id(row.target_audio, donor.timbre_ref_audio, output_audio, length=12)}"
        jobs.append(
            {
                "job_id": job_id,
                "pair_type": "v2_real_target_source_perturb",
                "prosody_ref_audio": row.target_audio,
                "source_audio": row.target_audio,
                "source_text": row.target_text,
                "timbre_ref_audio": donor.timbre_ref_audio,
                "timbre_ref_text": donor.timbre_ref_text,
                "target_audio": output_audio,
                "output_audio": output_audio,
                "target_text": row.target_text,
                "language": row.language,
                "source_speaker_id": row.source_speaker_id,
                "timbre_ref_speaker_id": donor.timbre_ref_speaker_id,
                "target_speaker_id": donor.timbre_ref_speaker_id,
                "metadata": {
                    "route": "v2_real_target_back_translation_source_generation",
                    "construction_rule": "u1_real_to_random_other_timbre_source; final_target_is_u1_real",
                    "real_target_audio": row.target_audio,
                    "real_target_audio_source_uri": row.target_audio_source_uri,
                    "real_target_text": row.target_text,
                    "final_timbre_ref_audio": row.timbre_ref_audio,
                    "final_timbre_ref_audio_source_uri": row.timbre_ref_audio_source_uri,
                    "final_timbre_ref_speaker_id": row.timbre_ref_speaker_id,
                    "perturb_timbre_dataset": donor.dataset_name,
                    "perturb_timbre_relative_path": donor.reference_relative_path,
                    "seedvc_backend": "seed_vc_v1_zero_shot_voice_conversion",
                    "source_audio_uri_kind": "local_materialized_file",
                    "requires_audio_materialization": True,
                },
            }
        )
    return jobs


def manifest_row(
    row: PairRow,
    source_job: dict[str, Any],
    *,
    ordinal: int,
    run_name: str,
    mode: str,
) -> dict[str, Any]:
    sample_id = f"{run_name}:{mode}:{ordinal:08d}:{stable_id(row.target_audio, row.timbre_ref_audio, mode, length=12)}"
    text = "<NO_TEXT>" if mode == "no_text" else row.target_text
    pair_type = "v2_real_target_no_text" if mode == "no_text" else "v2_real_target_text"
    return {
        "sample_id": sample_id,
        "source_audio": source_job["output_audio"],
        "source_text": row.target_text,
        "timbre_ref_audio": row.timbre_ref_audio,
        "timbre_ref_text": row.timbre_ref_text,
        "target_audio": row.target_audio,
        "target_text": row.target_text,
        "text": text,
        "language": row.language,
        "source_speaker_id": f"synthetic_source:{source_job['target_speaker_id']}",
        "timbre_ref_speaker_id": row.timbre_ref_speaker_id,
        "target_speaker_id": row.source_speaker_id,
        "source_gender": "unknown",
        "timbre_ref_gender": "unknown",
        "target_gender": "unknown",
        "pair_type": pair_type,
        "instruction": NO_TEXT_INSTRUCTION if mode == "no_text" else TEXT_INSTRUCTION,
        "text_prosody_instruction": TEXT_INSTRUCTION if mode == "text" else None,
        "preferred_emit_mode": mode,
        "source_audio_pending": True,
        "source_generation_job_id": source_job["job_id"],
        "source_generation_jobs_jsonl": "source_seedvc_jobs.jsonl",
        "v2_real_target": {
            "target_is_real_audio": True,
            "source_is_planned_seedvc_output": True,
            "teacher_pollution_side": "source_only",
            "source_generation_route": "Seed-VC(target_real -> random_other_timbre)",
        },
        "meta": {
            **asdict(row),
            "source_seedvc_job": {
                "job_id": source_job["job_id"],
                "prosody_ref_audio": source_job["prosody_ref_audio"],
                "perturb_timbre_ref_audio": source_job["timbre_ref_audio"],
                "output_audio": source_job["output_audio"],
            },
        },
    }


def simple_row(
    row: PairRow,
    source_job: dict[str, Any],
    *,
    ordinal: int,
    run_name: str,
    mode: str,
) -> dict[str, Any]:
    return {
        "sample_id": f"{run_name}:{mode}:{ordinal:08d}:{stable_id(row.target_audio, row.timbre_ref_audio, mode, length=12)}",
        "mode": mode,
        "dataset_name": row.dataset_name,
        "language": row.language,
        "lance_table_uri": row.lance_table_uri,
        "u1_target_audio_path": row.target_audio,
        "u1_target_audio_source_uri": row.target_audio_source_uri,
        "u1_target_audio_materialized": row.target_audio_materialized,
        "u1_text": row.target_text,
        "u1_duration_sec": row.target_duration_sec,
        "u2_timbre_ref_audio_path": row.timbre_ref_audio,
        "u2_timbre_ref_audio_source_uri": row.timbre_ref_audio_source_uri,
        "u2_timbre_ref_audio_materialized": row.timbre_ref_audio_materialized,
        "u2_text": row.timbre_ref_text,
        "u2_duration_sec": row.timbre_ref_duration_sec,
        "u1_prime_source_audio_path": source_job["output_audio"],
        "u1_prime_pending": True,
        "u1_prime_source_seedvc_job_id": source_job["job_id"],
        "u1_prime_seedvc_input_u1_path": source_job["prosody_ref_audio"],
        "u1_prime_seedvc_perturb_timbre_ref_path": source_job["timbre_ref_audio"],
        "label": row.label_relative_path,
        "reference": row.reference_relative_path,
        "segs": row.segs,
        "similarity": row.similarity,
        "label_idx": row.label_idx,
        "ref_idx": row.ref_idx,
        "source_line_index": row.source_line_index,
        "source_prepare_jsonl": row.source_prepare_jsonl,
        "source_resultf_jsonl": row.source_resultf_jsonl,
    }


def write_readme(output_root: Path, summary: dict[str, Any]) -> None:
    path = output_root / "README.md"
    lines = [
        "# MOSS-CodecVC V2 Real-Target Pilot",
        "",
        "Generated from `/inspire/hdd/project/embodied-multimodality/public/btjiang/VoiceClone/prepare`.",
        "",
        "Files:",
        "- `no_text.train.manifest.jsonl`: 10k no-text V2 manifest rows.",
        "- `text.train.manifest.jsonl`: 10k text V2 manifest rows.",
        "- `no_text.train.simple.jsonl`: compact no-text view with only u1/u2/u1' paths, text, label/segs/similarity.",
        "- `text.train.simple.jsonl`: compact text view with only u1/u2/u1' paths, text, label/segs/similarity.",
        "- `source_seedvc_jobs.jsonl`: planned Seed-VC jobs for generating `source_audio` (`u1'`).",
        "- `audio_download_manifest.jsonl`: qz_oss2 source URI to local materialized audio mapping.",
        "- `pair_manifest.jsonl`: shared real `(target=u1, timbre_ref=u2)` pair inventory.",
        "- `dataset_lance_mapping.json`: dataset name to lance table mapping.",
        "- `summary.json`: counts and filter settings.",
        "",
        "Lance table roots are read from the Feishu `PREDATATOTAL` table cache:",
        f"`{summary['input']['feishu_table_csv']}`.",
        "",
        "Important: `source_audio` in the two train manifests is a planned local output path under",
        "`seedvc_sources/`; those wavs must be materialized before codec/SFT feature extraction.",
        "`target_audio` and `timbre_ref_audio` point to local materialized files by default; their",
        "original qz_oss2 lance URIs are kept in `*_source_uri` fields.",
        "",
        f"Rows: no_text={summary['outputs']['no_text_rows']}, text={summary['outputs']['text_rows']}, "
        f"source_jobs={summary['outputs']['source_jobs']}.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resultf-dir", default=str(DEFAULT_RESULTF_DIR))
    parser.add_argument("--prepare-dir", default=str(DEFAULT_PREPARE_DIR))
    parser.add_argument("--meta-dir", default=str(DEFAULT_META_DIR))
    parser.add_argument("--newtrain-dir", default=str(DEFAULT_NEWTRAIN_DIR))
    parser.add_argument("--feishu-table-csv", default=str(DEFAULT_FEISHU_TABLE_CSV))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--materialize-audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download-audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--materialized-audio-dir", default="")
    parser.add_argument("--rclone-bin", default="rclone")
    parser.add_argument("--rclone-transfers", type=int, default=32)
    parser.add_argument("--rclone-checkers", type=int, default=64)
    parser.add_argument("--rclone-disable-proxy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-name", default="v2_real_target_pilot_20260706")
    parser.add_argument("--num-no-text", type=int, default=10000)
    parser.add_argument("--num-text", type=int, default=10000)
    parser.add_argument("--max-per-dataset", type=int, default=0)
    parser.add_argument("--min-similarity", type=float, default=0.85)
    parser.add_argument("--target-min-sec", type=float, default=2.0)
    parser.add_argument("--target-max-sec", type=float, default=15.0)
    parser.add_argument("--ref-min-sec", type=float, default=4.0)
    parser.add_argument("--ref-max-sec", type=float, default=30.0, help="0 disables max-length filtering for timbre refs.")
    parser.add_argument("--require-known-duration", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--duration-probe-rows", type=int, default=1000)
    parser.add_argument("--require-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-meta", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-newtrain-info", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--perturb-seed", type=int, default=20260706)
    parser.add_argument("--languages", default="", help="Comma-separated target languages, for example zh,en. Empty keeps all inferred languages.")
    parser.add_argument("--balance-languages", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--datasets", action="append", default=[], help="Optional dataset names. Repeat or comma-separate.")
    return parser.parse_args()


def expand_dataset_args(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out


def expand_csv_arg(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def main() -> int:
    args = parse_args()
    args.datasets = expand_dataset_args(args.datasets)
    args.languages = expand_csv_arg(args.languages)
    output_root = Path(args.output_root).expanduser().resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_names = sorted(DATASET_TABLES)
    if args.datasets:
        missing = sorted(set(args.datasets) - set(DATASET_TABLES))
        if missing:
            raise SystemExit(f"unknown dataset names: {missing}")
        dataset_names = sorted(args.datasets)

    feishu_table_csv = Path(args.feishu_table_csv).expanduser().resolve(strict=False)
    feishu_rows = load_feishu_lance_table_rows(feishu_table_csv)
    mappings = {
        name: dataset_mapping(name, Path(args.newtrain_dir), feishu_table_csv, feishu_rows)
        for name in dataset_names
    }
    write_json(output_root / "dataset_lance_mapping.json", {name: asdict(mapping) for name, mapping in mappings.items()})
    write_mapping_tsv(output_root / "dataset_lance_mapping.tsv", mappings)

    pairs, select_summary = select_pairs(args, mappings)
    selected_text_stats = fill_reference_text_from_selected_pairs(pairs)

    newtrain_info: dict[str, dict[str, dict[str, Any]]] = {}
    if args.load_newtrain_info:
        newtrain_info = load_newtrain_info(pairs, mappings)
    meta_info: dict[str, dict[str, dict[str, Any]]] = {}
    if args.load_meta:
        meta_info = load_meta_info(pairs, Path(args.meta_dir))
    enrich_stats = enrich_pairs(pairs, newtrain_info, meta_info)

    materialize_summary: dict[str, Any] = {
        "materialize_audio": False,
        "download_audio": False,
        "unique_audio_files": 0,
        "missing_after_download": 0,
    }
    if args.materialize_audio:
        materialized_audio_dir = Path(args.materialized_audio_dir) if args.materialized_audio_dir else None
        materialize_summary = {
            "materialize_audio": True,
            **materialize_pair_audio(
                pairs,
                mappings,
                output_root=output_root,
                materialized_audio_dir=materialized_audio_dir,
                download_audio=args.download_audio,
                rclone_bin=args.rclone_bin,
                rclone_transfers=args.rclone_transfers,
                rclone_checkers=args.rclone_checkers,
                rclone_disable_proxy=args.rclone_disable_proxy,
            ),
        }

    donors = choose_perturb_donors(pairs, args.perturb_seed)
    source_jobs = build_source_jobs(pairs, donors, output_root=output_root, run_name=args.run_name)
    no_text_pairs = pairs[: args.num_no_text]
    text_pairs = pairs[: args.num_text]
    source_jobs_by_idx = {idx: job for idx, job in enumerate(source_jobs)}

    pair_rows = [asdict(row) for row in pairs]
    no_text_rows = [
        manifest_row(row, source_jobs_by_idx[idx], ordinal=idx, run_name=args.run_name, mode="no_text")
        for idx, row in enumerate(no_text_pairs)
    ]
    text_rows = [
        manifest_row(row, source_jobs_by_idx[idx], ordinal=idx, run_name=args.run_name, mode="text")
        for idx, row in enumerate(text_pairs)
    ]
    no_text_simple_rows = [
        simple_row(row, source_jobs_by_idx[idx], ordinal=idx, run_name=args.run_name, mode="no_text")
        for idx, row in enumerate(no_text_pairs)
    ]
    text_simple_rows = [
        simple_row(row, source_jobs_by_idx[idx], ordinal=idx, run_name=args.run_name, mode="text")
        for idx, row in enumerate(text_pairs)
    ]

    pair_count = write_jsonl(output_root / "pair_manifest.jsonl", pair_rows)
    source_job_count = write_jsonl(output_root / "source_seedvc_jobs.jsonl", source_jobs)
    no_text_count = write_jsonl(output_root / "no_text.train.manifest.jsonl", no_text_rows)
    text_count = write_jsonl(output_root / "text.train.manifest.jsonl", text_rows)
    no_text_simple_count = write_jsonl(output_root / "no_text.train.simple.jsonl", no_text_simple_rows)
    text_simple_count = write_jsonl(output_root / "text.train.simple.jsonl", text_simple_rows)

    summary = {
        "run_name": args.run_name,
        "output_root": str(output_root),
        "input": {
            "resultf_dir": str(Path(args.resultf_dir).resolve(strict=False)),
            "prepare_dir": str(Path(args.prepare_dir).resolve(strict=False)),
            "meta_dir": str(Path(args.meta_dir).resolve(strict=False)),
            "newtrain_dir": str(Path(args.newtrain_dir).resolve(strict=False)),
            "feishu_table_csv": str(feishu_table_csv),
        },
        "filters": {
            "min_similarity": args.min_similarity,
            "target_min_sec": args.target_min_sec,
            "target_max_sec": args.target_max_sec,
            "ref_min_sec": args.ref_min_sec,
            "ref_max_sec": args.ref_max_sec,
            "require_known_duration": args.require_known_duration,
            "require_text": args.require_text,
        },
        "selection": select_summary,
        "enrichment_stats": dict(enrich_stats),
        "selected_pair_text_fill_stats": dict(selected_text_stats),
        "materialization": materialize_summary,
        "language_counts": dict(Counter(row.language for row in pairs)),
        "dataset_counts": dict(Counter(row.dataset_name for row in pairs)),
        "outputs": {
            "pair_rows": pair_count,
            "source_jobs": source_job_count,
            "no_text_rows": no_text_count,
            "text_rows": text_count,
            "no_text_simple_rows": no_text_simple_count,
            "text_simple_rows": text_simple_count,
            "pair_manifest": str((output_root / "pair_manifest.jsonl").resolve(strict=False)),
            "source_seedvc_jobs": str((output_root / "source_seedvc_jobs.jsonl").resolve(strict=False)),
            "no_text_manifest": str((output_root / "no_text.train.manifest.jsonl").resolve(strict=False)),
            "text_manifest": str((output_root / "text.train.manifest.jsonl").resolve(strict=False)),
            "no_text_simple": str((output_root / "no_text.train.simple.jsonl").resolve(strict=False)),
            "text_simple": str((output_root / "text.train.simple.jsonl").resolve(strict=False)),
            "audio_download_manifest": str((output_root / "audio_download_manifest.jsonl").resolve(strict=False)),
        },
        "caveat": (
            "source_audio values in train manifests are planned Seed-VC output paths and do not exist yet; "
            "materialize source_seedvc_jobs.jsonl before codec/SFT feature extraction."
        ),
    }
    write_json(output_root / "summary.json", summary)
    write_readme(output_root, summary)
    print(json.dumps(summary["outputs"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

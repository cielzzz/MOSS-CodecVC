#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OLD_NO_TEXT_JSONL = (
    ROOT
    / "trainset/zh45w_en22w_no_text/sft/"
    / "moss_codecvc_sft.zh45w_en22w_no_text.with_light_ecapa_spk.with_prosody.with_asr_filter.with_wavlm_bnf.with_spm_content_tokens.ctc_clean.jsonl"
)
DEFAULT_NEW_NO_TEXT_JSONL = (
    ROOT
    / "trainset/zh11w_en11w_0005_0015_vcdata_first_no_text/sft/"
    / "moss_codecvc_sft.zh11w_en11w_0005_0015_vcdata_first_no_text.with_light_ecapa_spk.with_prosody.with_asr_filter.with_wavlm_bnf.with_spm_content_tokens.ctc_clean.jsonl"
)
DEFAULT_TEXT_JSONL = (
    ROOT
    / "trainset/zh3w_en3w_text_prosody_independent_timbre/sft/"
    / "moss_codecvc_sft.zh3w_en3w_text_prosody_independent_timbre.with_light_ecapa_spk.with_prosody.with_target_asr.with_content_tokens.with_target_wavlm_bnf.with_spm_content_tokens.ctc_clean.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "trainset/ver2_8_prepared_zh45w_en22w_plus_zh11w_en11w_0005_0015_merged_no_text_plus_zh3w_text_textrep10"
)
DEFAULT_OLD_ASR_SUMMARY = (
    ROOT
    / "trainset/zh45w_en22w_no_text/sft/"
    / "moss_codecvc_sft.zh45w_en22w_no_text.with_light_ecapa_spk.with_prosody.with_asr_filter.with_wavlm_bnf.with_spm_content_tokens.ctc_clean.jsonl.summary.json"
)
DEFAULT_NEW_ASR_SUMMARY = (
    ROOT
    / "trainset/zh11w_en11w_0005_0015_vcdata_first_no_text/sft/"
    / "moss_codecvc_sft.zh11w_en11w_0005_0015_vcdata_first_no_text.with_light_ecapa_spk.with_prosody.with_asr_filter.keep.jsonl.summary.json"
)
DEFAULT_NEW_WAVLM_SUMMARY = (
    ROOT
    / "trainset/zh11w_en11w_0005_0015_vcdata_first_no_text/sft/"
    / "moss_codecvc_sft.zh11w_en11w_0005_0015_vcdata_first_no_text.with_light_ecapa_spk.with_prosody.with_asr_filter.with_wavlm_bnf.with_spm_content_tokens.ctc_clean.jsonl.summary.json"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Merge old and newly processed no-text Ver2.8 manifests, then build "
            "standard train/valid prepared splits with the existing Ver2.8 prepare script."
        )
    )
    ap.add_argument("--old-no-text-jsonl", default=str(DEFAULT_OLD_NO_TEXT_JSONL))
    ap.add_argument("--new-no-text-jsonl", default=str(DEFAULT_NEW_NO_TEXT_JSONL))
    ap.add_argument("--text-jsonl", default=str(DEFAULT_TEXT_JSONL))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--merged-no-text-jsonl", default="")
    ap.add_argument("--old-asr-summary", default=str(DEFAULT_OLD_ASR_SUMMARY))
    ap.add_argument("--new-asr-summary", default=str(DEFAULT_NEW_ASR_SUMMARY))
    ap.add_argument("--new-wavlm-summary", default=str(DEFAULT_NEW_WAVLM_SUMMARY))
    ap.add_argument("--prepare-script", default=str(ROOT / "scripts/002013_prepare_ver2_8_train_valid.py"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--semantic-kind", choices=("wavlm", "asr_bnf", "hubert", "any"), default="wavlm")
    ap.add_argument("--text-repeat", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--valid-count-no-text", type=int, default=1000)
    ap.add_argument("--valid-count-text", type=int, default=332)
    ap.add_argument("--check-feature-files", dest="check_feature_files", action="store_true")
    ap.add_argument("--no-check-feature-files", dest="check_feature_files", action="store_false")
    ap.set_defaults(check_feature_files=False)
    ap.add_argument("--require-no-text-target-feature", dest="require_no_text_target_feature", action="store_true")
    ap.add_argument("--no-require-no-text-target-feature", dest="require_no_text_target_feature", action="store_false")
    ap.set_defaults(require_no_text_target_feature=True)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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


def infer_mode(row: dict[str, Any]) -> str:
    mode = str(nested_get(row, "moss_codecvc_mode") or nested_get(row, "mode") or "").strip().lower()
    if mode in {"text", "no_text"}:
        return mode
    text = str(nested_get(row, "text") or "").strip()
    return "no_text" if text in {"", "<NO_TEXT>"} else "text"


def first_nonempty(row: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, str] | tuple[None, None]:
    for key in keys:
        value = nested_get(row, key)
        if value not in (None, ""):
            return key, str(value)
    return None, None


def file_exists(path: str | None) -> bool:
    return bool(path) and Path(str(path)).expanduser().exists()


def stable_hash(value: Any) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def dedup_key(row: dict[str, Any]) -> str:
    for key in ("sample_id", "uid", "id"):
        value = nested_get(row, key)
        if value not in (None, ""):
            return f"{key}:{value}"
    source_key, source_path = first_nonempty(
        row,
        (
            "source_wavlm_bnf_features_path",
            "source_wavlm_features_path",
            "source_hubert_features_path",
            "source_audio",
            "source_audio_path",
        ),
    )
    target_key, target_path = first_nonempty(
        row,
        (
            "target_wavlm_bnf_features_path",
            "target_wavlm_features_path",
            "target_hubert_features_path",
            "target_audio",
            "target_audio_path",
            "audio",
        ),
    )
    if source_path or target_path:
        return f"paths:{source_key}={source_path}|{target_key}={target_path}"
    return "row:" + stable_hash(row)


def validate_no_text_row(
    row: dict[str, Any],
    *,
    check_feature_files: bool,
    require_no_text_target_feature: bool,
) -> str | None:
    if infer_mode(row) != "no_text":
        return "mode_not_no_text"
    if not bool_value(nested_get(row, "content_keep"), default=False):
        return "content_keep_false"
    if not bool_value(nested_get(row, "content_token_keep"), default=False):
        return "content_token_keep_false"
    for key in ("asr_src_text", "asr_tgt_text"):
        if not str(nested_get(row, key) or "").strip():
            return f"missing_{key}"
    if not has_value(row, "content_token_ids") and not has_value(row, "content_token_ids_path"):
        return "missing_content_token_ids"
    if not has_value(row, "audio_codes") and not has_value(row, "audio_codes_path"):
        return "missing_target_audio_codes"
    refs = nested_get(row, "reference_audio_codes")
    if isinstance(refs, list):
        if len(refs) < 2 or refs[0] in (None, []) or refs[1] in (None, []):
            return "missing_reference_audio_codes_pair"
    elif not has_value(row, "reference_audio_codes_path"):
        return "missing_reference_audio_codes"
    for key in ("source_prosody_path", "target_prosody_path", "source_speaker_embedding_path", "target_speaker_embedding_path"):
        if not has_value(row, key):
            return f"missing_{key}"
        if check_feature_files and not file_exists(str(nested_get(row, key))):
            return f"missing_file:{key}"
    source_key, source_path = first_nonempty(
        row,
        (
            "source_wavlm_bnf_features_path",
            "source_wavlm_features_path",
            "source_semantic_features_path",
        ),
    )
    if not source_path:
        return "missing_source_wavlm_feature"
    if check_feature_files and not file_exists(source_path):
        return f"missing_file:{source_key}"
    target_key, target_path = first_nonempty(
        row,
        (
            "target_wavlm_bnf_features_path",
            "target_wavlm_features_path",
            "target_semantic_features_path",
        ),
    )
    if require_no_text_target_feature and not target_path:
        return "missing_target_wavlm_feature"
    if require_no_text_target_feature and check_feature_files and not file_exists(target_path):
        return f"missing_file:{target_key}"
    return None


def ensure_output_paths(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("output exists, pass --overwrite:\n  " + "\n  ".join(existing))


def merge_no_text_sources(
    sources: list[tuple[str, Path]],
    *,
    merged_path: Path,
    temp_dir: Path,
    check_feature_files: bool,
    require_no_text_target_feature: bool,
) -> dict[str, Any]:
    temp_dir.mkdir(parents=True, exist_ok=True)
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = merged_path.with_name(merged_path.name + ".tmp")
    seen: set[str] = set()
    stats: dict[str, Any] = {
        "output": str(merged_path),
        "inputs": [],
        "duplicates": 0,
        "written": 0,
        "rejects": 0,
        "reject_reasons": {},
    }
    reject_reasons: Counter[str] = Counter()
    reject_rows_by_source: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in sources}
    dup_examples: list[dict[str, Any]] = []

    with tmp_path.open("w", encoding="utf-8") as out:
        for source_name, source_path in sources:
            source_stats = {
                "name": source_name,
                "path": str(source_path),
                "rows": 0,
                "written": 0,
                "duplicates": 0,
                "rejects": 0,
                "reject_reasons": {},
            }
            source_reasons: Counter[str] = Counter()
            for line_no, row in iter_jsonl(source_path):
                source_stats["rows"] += 1
                reason = validate_no_text_row(
                    row,
                    check_feature_files=check_feature_files,
                    require_no_text_target_feature=require_no_text_target_feature,
                )
                if reason is not None:
                    source_stats["rejects"] += 1
                    source_reasons[reason] += 1
                    reject_reasons[reason] += 1
                    rejected = dict(row)
                    rejected["ver2_8_merge_reject_reason"] = reason
                    rejected["ver2_8_merge_source"] = source_name
                    rejected["ver2_8_merge_line_no"] = line_no
                    reject_rows_by_source[source_name].append(rejected)
                    continue

                key = dedup_key(row)
                if key in seen:
                    source_stats["duplicates"] += 1
                    stats["duplicates"] += 1
                    if len(dup_examples) < 20:
                        dup_examples.append(
                            {
                                "source": source_name,
                                "line_no": line_no,
                                "dedup_key": key,
                                "sample_id": nested_get(row, "sample_id"),
                            }
                        )
                    continue
                seen.add(key)
                output_row = dict(row)
                output_row["ver2_8_merge_source"] = source_name
                out.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                source_stats["written"] += 1
                stats["written"] += 1

            source_stats["reject_reasons"] = dict(source_reasons.most_common())
            stats["rejects"] += source_stats["rejects"]
            stats["inputs"].append(source_stats)
            write_jsonl(temp_dir / f"merge_rejects.{source_name}.jsonl", reject_rows_by_source[source_name])

    tmp_path.replace(merged_path)
    stats["reject_reasons"] = dict(reject_reasons.most_common())
    stats["duplicate_examples"] = dup_examples
    if stats["rejects"]:
        raise RuntimeError(f"no-text merge found rejected rows: {stats['reject_reasons']}")
    return stats


def run_prepare(args: argparse.Namespace, merged_path: Path, out_dir: Path) -> None:
    cmd = [
        args.python,
        str(Path(args.prepare_script).expanduser()),
        "--no-text-jsonl",
        str(merged_path),
        "--text-jsonl",
        str(Path(args.text_jsonl).expanduser()),
        "--output-dir",
        str(out_dir),
        "--semantic-kind",
        str(args.semantic_kind),
        "--text-repeat",
        str(args.text_repeat),
        "--seed",
        str(args.seed),
        "--valid-count-no-text",
        str(args.valid_count_no_text),
        "--valid-count-text",
        str(args.valid_count_text),
    ]
    if args.check_feature_files:
        cmd.append("--check-feature-files")
    if args.require_no_text_target_feature:
        cmd.append("--require-no-text-target-feature")
    if args.overwrite:
        cmd.append("--overwrite")
    print("[ver2.8-merge] running prepare: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def update_summary(out_dir: Path, merge_stats: dict[str, Any], args: argparse.Namespace) -> None:
    summary_path = out_dir / "summary.json"
    summary = load_json(summary_path)
    if summary is None:
        raise FileNotFoundError(f"prepare summary missing: {summary_path}")
    old_asr_summary = load_json(Path(args.old_asr_summary).expanduser())
    new_asr_summary = load_json(Path(args.new_asr_summary).expanduser())
    new_wavlm_summary = load_json(Path(args.new_wavlm_summary).expanduser())
    summary["merge"] = {
        "status": "complete",
        "text_repeat_recommended_for_balance": int(args.text_repeat),
        "old_no_text_asr_summary": old_asr_summary,
        "new_no_text_asr_summary": new_asr_summary,
        "new_no_text_wavlm_summary": new_wavlm_summary,
        "stats": merge_stats,
    }
    write_json(summary_path, summary)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser()
    temp_dir = out_dir / "temp"
    merged_path = Path(args.merged_no_text_jsonl).expanduser() if args.merged_no_text_jsonl else temp_dir / "no_text.merged.source_clean.jsonl"

    ensure_output_paths(
        [
            merged_path,
            out_dir / "no_text.train.jsonl",
            out_dir / "no_text.valid.jsonl",
            out_dir / "text.train.jsonl",
            out_dir / "text.valid.jsonl",
            out_dir / "mixed.train.spec.txt",
            out_dir / "mixed.valid.spec.txt",
            out_dir / "summary.json",
        ],
        bool(args.overwrite),
    )

    sources = [
        ("zh45w_en22w_no_text", Path(args.old_no_text_jsonl).expanduser()),
        ("zh11w_en11w_0005_0015_vcdata_first_no_text", Path(args.new_no_text_jsonl).expanduser()),
    ]
    for _, path in sources:
        if not path.exists():
            raise FileNotFoundError(path)
    if not Path(args.text_jsonl).expanduser().exists():
        raise FileNotFoundError(args.text_jsonl)

    print("[ver2.8-merge] merging no-text sources", flush=True)
    merge_stats = merge_no_text_sources(
        sources,
        merged_path=merged_path,
        temp_dir=temp_dir,
        check_feature_files=bool(args.check_feature_files),
        require_no_text_target_feature=bool(args.require_no_text_target_feature),
    )
    print(
        "[ver2.8-merge] merged no_text "
        f"written={merge_stats['written']} duplicates={merge_stats['duplicates']} rejects={merge_stats['rejects']}",
        flush=True,
    )

    run_prepare(args, merged_path, out_dir)
    update_summary(out_dir, merge_stats, args)
    print(f"[ver2.8-merge] prepared_dir={out_dir}", flush=True)
    print(f"[ver2.8-merge] summary={out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

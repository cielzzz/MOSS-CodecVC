#!/usr/bin/env python3
"""Audit overlap between the official Seed-TTS-Eval archive and our 320 cases.

The official archive contains separate manifests for zero-shot TTS and VC.  This
script extracts only those small manifests, records their exact counts, and
checks whether the source/reference IDs and texts in our internal JSONL are
drawn from the public benchmark.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any


MANIFEST_MEMBERS = {
    "test_en_tts": "seedtts_testset/en/meta.lst",
    "test_en_vc": "seedtts_testset/en/non_para_reconstruct_meta.lst",
    "test_zh_tts": "seedtts_testset/zh/meta.lst",
    "test_zh_hard_tts": "seedtts_testset/zh/hardcase.lst",
    "test_zh_vc": "seedtts_testset/zh/non_para_reconstruct_meta.lst",
}

SPLIT_DIRS = {
    "test_en_tts": "en",
    "test_zh_tts": "zh",
    "test_zh_hard_tts": "zh",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-tar", type=Path, required=True)
    parser.add_argument("--official-root", type=Path)
    parser.add_argument("--internal-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_manifest(tf: tarfile.TarFile, member: str) -> list[dict[str, Any]]:
    extracted = tf.extractfile(member)
    if extracted is None:
        raise FileNotFoundError(f"manifest not found in archive: {member}")
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(extracted.read().decode("utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split("|")
        if len(fields) not in {4, 5}:
            raise ValueError(f"{member}:{line_no}: expected 4 or 5 fields, got {len(fields)}")
        row = {
            "id": fields[0],
            "prompt_text": fields[1],
            "prompt_audio": fields[2],
            "target_text": fields[3],
        }
        if len(fields) == 5:
            row["target_audio"] = fields[4]
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def manifest_line(row: dict[str, Any]) -> str:
    fields = [
        str(row["id"]),
        str(row["prompt_text"]),
        str(row["prompt_audio"]),
        str(row["target_text"]),
    ]
    if "target_audio" in row:
        fields.append(str(row["target_audio"]))
    return "|".join(fields)


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(manifest_line(row) for row in rows) + "\n", encoding="utf-8")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def sha256_file(path: Path, cache: dict[Path, str]) -> str:
    path = path.resolve()
    if path not in cache:
        with path.open("rb") as handle:
            cache[path] = hashlib.file_digest(handle, "sha256").hexdigest()
    return cache[path]


def select_prompt_entry(
    entries: list[tuple[str, dict[str, Any]]], prompt_text: str
) -> tuple[str, dict[str, Any]] | None:
    exact = [entry for entry in entries if str(entry[1]["prompt_text"]).strip() == prompt_text]
    if len(exact) == 1:
        return exact[0]
    if exact:
        return exact[0]
    return entries[0] if entries else None


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(args.official_tar, "r:") as tf:
        archive_members = set(tf.getnames())
        manifests = {
            name: read_manifest(tf, member) for name, member in MANIFEST_MEMBERS.items()
        }

    for name, rows in manifests.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", rows)

    tts_splits = {name: rows for name, rows in manifests.items() if name.endswith("_tts")}
    id_to_entries: dict[str, list[tuple[str, dict[str, Any]]]] = collections.defaultdict(list)
    for split, rows in tts_splits.items():
        for row in rows:
            id_to_entries[str(row["id"])].append((split, row))

    internal = read_jsonl(args.internal_jsonl)
    details: list[dict[str, Any]] = []
    hash_cache: dict[Path, str] = {}
    for row in internal:
        source_entries = id_to_entries.get(str(row.get("source_id") or ""), [])
        ref_entries = id_to_entries.get(str(row.get("ref_id") or ""), [])
        source_text = str(row.get("source_text") or "").strip()
        ref_text = str(row.get("timbre_ref_text") or "").strip()
        input_text = str(row.get("text") or "").strip()
        detail = {
            "case_id": row.get("case_id"),
            "mode": row.get("mode"),
            "source_id": row.get("source_id"),
            "source_official_splits": sorted(split for split, _ in source_entries),
            "source_id_match": bool(source_entries),
            "source_prompt_text_match": any(
                source_text == str(entry["prompt_text"]).strip() for _, entry in source_entries
            ),
            "ref_id": row.get("ref_id"),
            "ref_official_splits": sorted(split for split, _ in ref_entries),
            "ref_id_match": bool(ref_entries),
            "ref_prompt_text_match": any(
                ref_text == str(entry["prompt_text"]).strip() for _, entry in ref_entries
            ),
            "text_matches_source_target": any(
                input_text == str(entry["target_text"]).strip() for _, entry in source_entries
            )
            if row.get("mode") == "text"
            else None,
        }
        if args.official_root is not None:
            for side, entries, prompt_text, audio_field in (
                ("source", source_entries, source_text, "source_audio"),
                ("ref", ref_entries, ref_text, "timbre_ref_audio"),
            ):
                selected = select_prompt_entry(entries, prompt_text)
                local_audio = Path(str(row.get(audio_field) or ""))
                official_audio: Path | None = None
                if selected is not None:
                    split, entry = selected
                    official_audio = (
                        args.official_root
                        / SPLIT_DIRS[split]
                        / str(entry["prompt_audio"])
                    )
                detail[f"{side}_local_audio_exists"] = local_audio.is_file()
                detail[f"{side}_official_audio"] = (
                    str(official_audio.resolve()) if official_audio is not None else None
                )
                detail[f"{side}_official_audio_exists"] = bool(
                    official_audio is not None and official_audio.is_file()
                )
                detail[f"{side}_prompt_audio_sha256_match"] = (
                    sha256_file(local_audio, hash_cache)
                    == sha256_file(official_audio, hash_cache)
                    if local_audio.is_file()
                    and official_audio is not None
                    and official_audio.is_file()
                    else None
                )
        details.append(detail)

    all_targets = {
        str(entry["target_text"]).strip()
        for rows in tts_splits.values()
        for entry in rows
    }
    text_rows = [row for row in internal if row.get("mode") == "text"]
    source_ids = {str(row.get("source_id") or "") for row in internal}
    ref_ids = {str(row.get("ref_id") or "") for row in internal}
    summary = {
        "official_tar": str(args.official_tar.resolve()),
        "internal_jsonl": str(args.internal_jsonl.resolve()),
        "official_manifest_members": MANIFEST_MEMBERS,
        "official_zh_hard_vc_manifest_candidates": sorted(
            member
            for member in archive_members
            if "hard" in member.lower()
            and "non_para_reconstruct_meta" in member.lower()
        ),
        "official_counts": {name: len(rows) for name, rows in manifests.items()},
        "official_rows_with_target_audio": {
            name: sum("target_audio" in row for row in rows) for name, rows in manifests.items()
        },
        "internal_rows": len(internal),
        "internal_source_unique": len(source_ids),
        "internal_source_unique_official_id_matches": sum(i in id_to_entries for i in source_ids),
        "internal_ref_unique": len(ref_ids),
        "internal_ref_unique_official_id_matches": sum(i in id_to_entries for i in ref_ids),
        "internal_rows_source_and_ref_id_match": sum(
            detail["source_id_match"] and detail["ref_id_match"] for detail in details
        ),
        "internal_rows_source_prompt_text_match": sum(
            detail["source_prompt_text_match"] for detail in details
        ),
        "internal_rows_ref_prompt_text_match": sum(
            detail["ref_prompt_text_match"] for detail in details
        ),
        "internal_text_rows": len(text_rows),
        "internal_text_rows_target_in_any_official_tts_manifest": sum(
            str(row.get("text") or "").strip() in all_targets for row in text_rows
        ),
        "internal_text_rows_target_matches_source_id_target": sum(
            detail["text_matches_source_target"] is True for detail in details
        ),
    }
    if args.official_root is not None:
        official_audio_audit: dict[str, dict[str, int]] = {}
        for name, rows in manifests.items():
            language_dir = "en" if name.startswith("test_en") else "zh"
            prompt_paths = [
                args.official_root / language_dir / str(row["prompt_audio"])
                for row in rows
            ]
            target_paths = [
                args.official_root / language_dir / str(row["target_audio"])
                for row in rows
                if "target_audio" in row
            ]
            official_audio_audit[name] = {
                "prompt_rows": len(prompt_paths),
                "prompt_missing_rows": sum(not path.is_file() for path in prompt_paths),
                "target_audio_rows": len(target_paths),
                "target_audio_missing_rows": sum(not path.is_file() for path in target_paths),
                "unique_prompt_files": len({path.resolve() for path in prompt_paths}),
                "unique_target_audio_files": len({path.resolve() for path in target_paths}),
            }

        internal_source_audio_hashes = {
            sha256_file(Path(str(row["source_audio"])), hash_cache) for row in internal
        }
        internal_ref_audio_hashes = {
            sha256_file(Path(str(row["timbre_ref_audio"])), hash_cache) for row in internal
        }
        internal_audio_hashes = internal_source_audio_hashes | internal_ref_audio_hashes
        internal_ordered_source_ref_pairs = {
            (
                sha256_file(Path(str(row["source_audio"])), hash_cache),
                sha256_file(Path(str(row["timbre_ref_audio"])), hash_cache),
            )
            for row in internal
        }
        internal_texts = {
            normalize_text(row.get(field))
            for row in internal
            for field in ("source_text", "timbre_ref_text", "content_ref_text", "text")
            if normalize_text(row.get(field)) not in {"", "<no_text>"}
        }

        vc_overlap_details: list[dict[str, Any]] = []
        vc_overlap_summary: dict[str, dict[str, Any]] = {}
        for name in ("test_en_vc", "test_zh_vc"):
            language_dir = "en" if name.startswith("test_en") else "zh"
            strict_audio_rows: list[dict[str, Any]] = []
            strict_case_rows: list[dict[str, Any]] = []
            split_details: list[dict[str, Any]] = []
            for row in manifests[name]:
                prompt_path = args.official_root / language_dir / str(row["prompt_audio"])
                source_path = args.official_root / language_dir / str(row["target_audio"])
                prompt_hash = sha256_file(prompt_path, hash_cache)
                source_hash = sha256_file(source_path, hash_cache)
                prompt_overlaps_source = prompt_hash in internal_source_audio_hashes
                prompt_overlaps_ref = prompt_hash in internal_ref_audio_hashes
                source_overlaps_source = source_hash in internal_source_audio_hashes
                source_overlaps_ref = source_hash in internal_ref_audio_hashes
                prompt_audio_overlap = prompt_hash in internal_audio_hashes
                source_audio_overlap = source_hash in internal_audio_hashes
                prompt_text_overlap = normalize_text(row["prompt_text"]) in internal_texts
                target_text_overlap = normalize_text(row["target_text"]) in internal_texts
                exact_ordered_pair = (
                    source_hash,
                    prompt_hash,
                ) in internal_ordered_source_ref_pairs
                keep_audio = not (prompt_audio_overlap or source_audio_overlap)
                keep_case = keep_audio and not (prompt_text_overlap or target_text_overlap)
                reasons = []
                if prompt_audio_overlap:
                    reasons.append("reference_prompt_audio")
                if source_audio_overlap:
                    reasons.append("source_content_audio")
                if prompt_text_overlap:
                    reasons.append("reference_prompt_text")
                if target_text_overlap:
                    reasons.append("source_content_text")
                detail = {
                    "manifest": name,
                    "id": row["id"],
                    "prompt_audio": row["prompt_audio"],
                    "source_audio": row["target_audio"],
                    "prompt_audio_sha256": prompt_hash,
                    "source_audio_sha256": source_hash,
                    "prompt_overlaps_internal_source_audio": prompt_overlaps_source,
                    "prompt_overlaps_internal_ref_audio": prompt_overlaps_ref,
                    "source_overlaps_internal_source_audio": source_overlaps_source,
                    "source_overlaps_internal_ref_audio": source_overlaps_ref,
                    "prompt_audio_overlap_internal_any": prompt_audio_overlap,
                    "source_audio_overlap_internal_any": source_audio_overlap,
                    "prompt_text_overlap_internal_any": prompt_text_overlap,
                    "target_text_overlap_internal_any": target_text_overlap,
                    "exact_ordered_source_ref_pair_match": exact_ordered_pair,
                    "keep_strict_audio_disjoint": keep_audio,
                    "keep_strict_audio_and_text_disjoint": keep_case,
                    "overlap_reasons": reasons,
                }
                split_details.append(detail)
                vc_overlap_details.append(detail)
                if keep_audio:
                    strict_audio_rows.append(row)
                if keep_case:
                    strict_case_rows.append(row)

            audio_stem = f"official_{language_dir}_vc_minus_internal320_strict_audio"
            case_stem = f"official_{language_dir}_vc_minus_internal320_strict_case"
            write_manifest(args.output_dir / f"{audio_stem}.lst", strict_audio_rows)
            write_jsonl(args.output_dir / f"{audio_stem}.jsonl", strict_audio_rows)
            write_manifest(args.output_dir / f"{case_stem}.lst", strict_case_rows)
            write_jsonl(args.output_dir / f"{case_stem}.jsonl", strict_case_rows)

            audio_manifest_path = args.output_dir / f"{audio_stem}.lst"
            case_manifest_path = args.output_dir / f"{case_stem}.lst"

            prompt_hashes = {detail["prompt_audio_sha256"] for detail in split_details}
            source_hashes = {detail["source_audio_sha256"] for detail in split_details}
            vc_overlap_summary[name] = {
                "total_rows": len(split_details),
                "prompt_audio_overlap_internal_any_rows": sum(
                    detail["prompt_audio_overlap_internal_any"] for detail in split_details
                ),
                "source_audio_overlap_internal_any_rows": sum(
                    detail["source_audio_overlap_internal_any"] for detail in split_details
                ),
                "either_audio_overlap_rows": sum(
                    detail["prompt_audio_overlap_internal_any"]
                    or detail["source_audio_overlap_internal_any"]
                    for detail in split_details
                ),
                "prompt_overlap_internal_source_rows": sum(
                    detail["prompt_overlaps_internal_source_audio"] for detail in split_details
                ),
                "prompt_overlap_internal_ref_rows": sum(
                    detail["prompt_overlaps_internal_ref_audio"] for detail in split_details
                ),
                "source_overlap_internal_source_rows": sum(
                    detail["source_overlaps_internal_source_audio"] for detail in split_details
                ),
                "source_overlap_internal_ref_rows": sum(
                    detail["source_overlaps_internal_ref_audio"] for detail in split_details
                ),
                "exact_ordered_source_ref_pair_matches": sum(
                    detail["exact_ordered_source_ref_pair_match"] for detail in split_details
                ),
                "unique_prompt_audio_assets": len(prompt_hashes),
                "unique_prompt_audio_assets_overlapping_internal": len(
                    prompt_hashes & internal_audio_hashes
                ),
                "unique_source_audio_assets": len(source_hashes),
                "unique_source_audio_assets_overlapping_internal": len(
                    source_hashes & internal_audio_hashes
                ),
                "prompt_text_overlap_internal_rows": sum(
                    detail["prompt_text_overlap_internal_any"] for detail in split_details
                ),
                "target_text_overlap_internal_rows": sum(
                    detail["target_text_overlap_internal_any"] for detail in split_details
                ),
                "either_text_overlap_rows": sum(
                    detail["prompt_text_overlap_internal_any"]
                    or detail["target_text_overlap_internal_any"]
                    for detail in split_details
                ),
                "strict_audio_disjoint_rows": len(strict_audio_rows),
                "strict_audio_disjoint_manifest": str(audio_manifest_path.resolve()),
                "strict_audio_disjoint_manifest_sha256": sha256_file(
                    audio_manifest_path, hash_cache
                ),
                "strict_audio_and_text_disjoint_rows": len(strict_case_rows),
                "strict_audio_and_text_disjoint_manifest": str(case_manifest_path.resolve()),
                "strict_audio_and_text_disjoint_manifest_sha256": sha256_file(
                    case_manifest_path, hash_cache
                ),
            }

        write_jsonl(
            args.output_dir / "official_vc_vs_internal320_overlap_details.jsonl",
            vc_overlap_details,
        )
        summary.update(
            {
                "official_root": str(args.official_root.resolve()),
                "official_audio_audit": official_audio_audit,
                "internal_audio_asset_counts": {
                    "unique_source_audio_sha256": len(internal_source_audio_hashes),
                    "unique_ref_audio_sha256": len(internal_ref_audio_hashes),
                    "unique_union_audio_sha256": len(internal_audio_hashes),
                    "source_ref_shared_audio_sha256": len(
                        internal_source_audio_hashes & internal_ref_audio_hashes
                    ),
                    "unique_ordered_source_ref_pairs": len(
                        internal_ordered_source_ref_pairs
                    ),
                },
                "official_vc_internal320_overlap": vc_overlap_summary,
                "internal_rows_source_prompt_audio_sha256_match": sum(
                    detail["source_prompt_audio_sha256_match"] is True for detail in details
                ),
                "internal_rows_ref_prompt_audio_sha256_match": sum(
                    detail["ref_prompt_audio_sha256_match"] is True for detail in details
                ),
            }
        )

    write_jsonl(args.output_dir / "internal320_overlap_details.jsonl", details)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    counts = summary["official_counts"]
    lines = [
        "# Batch-42 Seed-TTS-Eval overlap audit",
        "",
        "## Official archive manifests",
        "",
    ]
    if args.official_root is not None:
        lines.extend(
            [
                "| Manifest | Rows | Source/target audio rows | Missing prompt rows | Missing source/target rows |",
                "|---|---:|---:|---:|---:|",
            ]
        )
    else:
        lines.extend(
            [
                "| Manifest | Rows | Rows with source/target audio field |",
                "|---|---:|---:|",
            ]
        )
    for name in MANIFEST_MEMBERS:
        if args.official_root is not None:
            audio = summary["official_audio_audit"][name]
            lines.append(
                f"| `{name}` | {counts[name]} | {audio['target_audio_rows']} | "
                f"{audio['prompt_missing_rows']} | {audio['target_audio_missing_rows']} |"
            )
        else:
            lines.append(
                f"| `{name}` | {counts[name]} | "
                f"{summary['official_rows_with_target_audio'][name]} |"
            )
    lines.extend(
        [
            "",
            "ZH-hard VC manifest candidates in archive: "
            f"{len(summary['official_zh_hard_vc_manifest_candidates'])}.",
        ]
    )
    lines.extend(
        [
            "",
            "## Internal 320 overlap",
            "",
            f"- Rows with both source and reference IDs in official TTS manifests: "
            f"{summary['internal_rows_source_and_ref_id_match']}/{summary['internal_rows']}.",
            f"- Source prompt text exact matches: "
            f"{summary['internal_rows_source_prompt_text_match']}/{summary['internal_rows']}.",
            f"- Reference prompt text exact matches: "
            f"{summary['internal_rows_ref_prompt_text_match']}/{summary['internal_rows']}.",
            f"- Text-mode target text found in an official TTS manifest: "
            f"{summary['internal_text_rows_target_in_any_official_tts_manifest']}/"
            f"{summary['internal_text_rows']}.",
            "",
            "## Interpretation",
            "",
            "The internal 320 is an official-derived recombination/subset, not an independent "
            "test set. Existing 320 results must not be relabeled as a complete official "
            "Seed-TTS-Eval run. Audio-to-audio VC systems should use the official "
            "`non_para_reconstruct_meta.lst` manifests, not the text-only TTS manifests.",
            "",
        ]
    )
    if args.official_root is not None:
        lines[lines.index("## Interpretation"):lines.index("## Interpretation")] = [
            f"- Source prompt audio SHA-256 exact matches: "
            f"{summary['internal_rows_source_prompt_audio_sha256_match']}/"
            f"{summary['internal_rows']}.",
            f"- Reference prompt audio SHA-256 exact matches: "
            f"{summary['internal_rows_ref_prompt_audio_sha256_match']}/"
            f"{summary['internal_rows']}.",
            "",
        ]
        lines.extend(
            [
                "## Official VC versus internal 320",
                "",
                "| Split | Total | Prompt-audio overlap | Source-audio overlap | Exact ordered pair | Strict audio-disjoint | Strict audio+text-disjoint |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in ("test_en_vc", "test_zh_vc"):
            audit = summary["official_vc_internal320_overlap"][name]
            lines.append(
                f"| `{name}` | {audit['total_rows']} | "
                f"{audit['prompt_audio_overlap_internal_any_rows']} | "
                f"{audit['source_audio_overlap_internal_any_rows']} | "
                f"{audit['exact_ordered_source_ref_pair_matches']} | "
                f"{audit['strict_audio_disjoint_rows']} | "
                f"{audit['strict_audio_and_text_disjoint_rows']} |"
            )
        lines.extend(
            [
                "",
                "For the strongest model-selection firewall, use the strict "
                "audio+text-disjoint manifests. The strict audio-disjoint manifests "
                "are the direct answer to the no-shared-audio requirement.",
                "",
            ]
        )
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare the Batch-43 Ground truth *source-self* calibration inputs.

The strict Batch-42/43 benchmark is derived from Seed-TTS-Eval's
``non_para_reconstruct_meta.lst`` VC manifests.  Each five-column row contains
the target-speaker prompt in field 3 and a *different-speaker* source/content
recording in field 5.  The public package does not contain a paired recording
of the target speaker reading field 4, so a conventional target-speaker
``Ground truth`` VC waveform does not exist.

This script therefore prepares a deliberately narrower calibration row:

* generated_audio = field 5 source/content recording;
* reference_audio = the same field 5 recording;
* source_audio = the same field 5 recording;
* reference_text = field 4 transcript;
* target_reference_audio = field 3 prompt, retained for provenance only.

The resulting same-file speaker scores are scorer self-similarity checks (near
or exactly 1.0), not target-speaker VC scores.  ASR on field 5 is a useful raw
recording ceiling for the exact content transcript used by every converted
system.  The generated audit and Markdown repeat this warning so the row cannot
silently be presented as a comparable VC system result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "moss_codecvc.batch43_ground_truth_source_self_input.v1"
AUDIT_SCHEMA_VERSION = "moss_codecvc.batch43_ground_truth_source_self_audit.v1"
SYSTEM_ID = "ground_truth"
DISPLAY_NAME = "Ground truth (self-eval)"
CALIBRATION_WARNING = (
    "Same-file source self-evaluation only: speaker SIM is a scorer calibration, "
    "not target-speaker VC performance. The non-parallel VC manifest provides no "
    "paired waveform of the target prompt speaker reading the source transcript."
)

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_AUDIT_ROOT = (
    PROJECT_ROOT / "testset/outputs/batch42_seedtts_eval_audit_20260711"
)
DEFAULT_DATASET_ROOT = Path(
    "/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/"
    "batch42/datasets/seed-tts-eval/seedtts_testset"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "testset/outputs/batch43_ground_truth_source_self_inputs_20260712"
)


@dataclass(frozen=True)
class SplitSpec:
    language: str
    expected_rows: int
    manifest_sha256: str
    test_set_id: str
    manifest_name: str


SPLITS: dict[str, SplitSpec] = {
    "en": SplitSpec(
        language="en",
        expected_rows=567,
        manifest_sha256=(
            "48549d8029e680d74656660191c4641ca5a8040ccbe3252ce89bfc3b0c9c75ae"
        ),
        test_set_id="seedtts-vc-en-internal320-disjoint",
        manifest_name="official_en_vc_minus_internal320_strict_case.lst",
    ),
    "zh": SplitSpec(
        language="zh",
        expected_rows=1194,
        manifest_sha256=(
            "4b637cc1cff33dc369954755538d12396fc92d439a52742103a29b7c563cf6df"
        ),
        test_set_id="seedtts-vc-zh-internal320-disjoint",
        manifest_name="official_zh_vc_minus_internal320_strict_case.lst",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def require_relative_audio(
    dataset_language_root: Path,
    raw_path: str,
    *,
    context: str,
) -> Path:
    raw_path = raw_path.strip()
    candidate = Path(raw_path)
    if not raw_path or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{context}: audio path must be a safe relative path: {raw_path!r}")
    language_root = dataset_language_root.expanduser().resolve()
    resolved = (language_root / candidate).resolve()
    try:
        resolved.relative_to(language_root)
    except ValueError as exc:
        raise ValueError(f"{context}: audio escapes dataset root: {resolved}") from exc
    if not resolved.is_file() or resolved.stat().st_size < 44:
        raise ValueError(f"{context}: missing/empty audio: {resolved}")
    if resolved.suffix.lower() != ".wav":
        raise ValueError(f"{context}: expected a WAV asset: {resolved}")
    return resolved


def stable_case_uid(
    language: str,
    case_id: str,
    source_audio: Path,
    target_reference_audio: Path,
) -> str:
    material = "\0".join(
        (
            "ground_truth_source_self",
            language,
            case_id,
            str(source_audio),
            str(target_reference_audio),
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def build_split(
    *,
    spec: SplitSpec,
    manifest: Path,
    dataset_root: Path,
    output_jsonl: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = manifest.expanduser().resolve()
    dataset_root = dataset_root.expanduser().resolve()
    if not manifest.is_file():
        raise ValueError(f"{spec.language}: missing strict manifest: {manifest}")
    actual_manifest_sha = sha256_file(manifest)
    if spec.manifest_sha256 and actual_manifest_sha != spec.manifest_sha256:
        raise ValueError(
            f"{spec.language}: manifest SHA256={actual_manifest_sha}, "
            f"expected {spec.manifest_sha256}"
        )

    language_root = dataset_root / spec.language
    if not language_root.is_dir():
        raise ValueError(f"{spec.language}: missing dataset language root: {language_root}")

    rows: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    seen_case_uids: set[str] = set()
    unique_sources: set[Path] = set()
    unique_prompts: set[Path] = set()
    audio_sha_cache: dict[Path, str] = {}
    paired_same_sha = 0

    def audio_sha(path: Path) -> str:
        if path not in audio_sha_cache:
            audio_sha_cache[path] = sha256_file(path)
        return audio_sha_cache[path]

    with manifest.open(encoding="utf-8") as handle:
        for input_line, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                raise ValueError(f"{manifest}:{input_line}: blank rows are forbidden")
            fields = [value.strip() for value in line.split("|")]
            if len(fields) != 5:
                raise ValueError(
                    f"{manifest}:{input_line}: strict VC row must have exactly 5 fields; "
                    f"got {len(fields)}"
                )
            case_id, prompt_text, prompt_audio_raw, target_text, source_audio_raw = fields
            if any(not value for value in fields):
                raise ValueError(f"{manifest}:{input_line}: empty field in strict VC row")
            if case_id in seen_case_ids:
                raise ValueError(f"{manifest}:{input_line}: duplicate case_id={case_id!r}")

            prompt_audio = require_relative_audio(
                language_root,
                prompt_audio_raw,
                context=f"{manifest}:{input_line}:field3/prompt",
            )
            source_audio = require_relative_audio(
                language_root,
                source_audio_raw,
                context=f"{manifest}:{input_line}:field5/source",
            )
            if prompt_audio == source_audio:
                raise ValueError(
                    f"{manifest}:{input_line}: non-parallel source and prompt resolve "
                    f"to the same path: {source_audio}"
                )
            if audio_sha(prompt_audio) == audio_sha(source_audio):
                paired_same_sha += 1
                raise ValueError(
                    f"{manifest}:{input_line}: non-parallel source and prompt have "
                    "identical WAV bytes"
                )

            case_uid = stable_case_uid(
                spec.language, case_id, source_audio, prompt_audio
            )
            if case_uid in seen_case_uids:
                raise ValueError(f"{manifest}:{input_line}: duplicate case_uid={case_uid}")
            seen_case_ids.add(case_id)
            seen_case_uids.add(case_uid)
            unique_sources.add(source_audio)
            unique_prompts.add(prompt_audio)

            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "ground_truth_source_self_eval_input",
                    "status": "ok",
                    "system_id": SYSTEM_ID,
                    "test_set_id": spec.test_set_id,
                    "case_id": case_id,
                    "case_uid": case_uid,
                    "input_index": len(rows),
                    "input_line": input_line,
                    "language": spec.language,
                    # These three intentionally use the exact same path.  004082
                    # caches by path, so the speaker score is a true same-embedding
                    # scorer calibration rather than a second stochastic decode.
                    "generated_audio": str(source_audio),
                    "reference_audio": str(source_audio),
                    "source_audio": str(source_audio),
                    # Keep the actual VC target-speaker reference visible in the
                    # preparation ledger, but do not feed it to the self-SIM cell.
                    "target_reference_audio": str(prompt_audio),
                    "prompt_text": prompt_text,
                    "target_text": target_text,
                    "reference_text": target_text,
                    "evaluation_contract": {
                        "display_name": DISPLAY_NAME,
                        "calibration_only": True,
                        "paired_target_speaker_ground_truth_available": False,
                        "same_file_speaker_self_similarity": True,
                        "asr_raw_source_ceiling": True,
                        "target_reference_audio_used_for_speaker_score": False,
                        "warning": CALIBRATION_WARNING,
                    },
                    "provenance": {
                        "input": str(manifest),
                        "input_manifest_sha256": actual_manifest_sha,
                        "input_format": "seedtts_vc_non_parallel_five_field_lst",
                        "field_mapping": {
                            "generated_audio": (
                                "field_5/source_content_audio_raw_ground_truth"
                            ),
                            "reference_audio": (
                                "field_5/source_content_audio_same_file_self_sim"
                            ),
                            "source_audio": "field_5/source_content_audio",
                            "target_reference_audio": (
                                "field_3/prompt_target_speaker_provenance_only"
                            ),
                            "reference_text": "field_4/source_transcript",
                        },
                    },
                }
            )

    if len(rows) != spec.expected_rows:
        raise ValueError(
            f"{spec.language}: rows={len(rows)}, expected {spec.expected_rows}"
        )
    expected_indices = list(range(spec.expected_rows))
    if [row["input_index"] for row in rows] != expected_indices:
        raise AssertionError(f"{spec.language}: input_index coverage is not contiguous")

    rendered = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    atomic_write(output_jsonl, rendered)
    output_sha = sha256_file(output_jsonl)
    split_audit = {
        "language": spec.language,
        "test_set_id": spec.test_set_id,
        "rows": len(rows),
        "unique_case_ids": len(seen_case_ids),
        "unique_case_uids": len(seen_case_uids),
        "input_index_coverage": [0, len(rows) - 1],
        "manifest": str(manifest),
        "manifest_sha256": actual_manifest_sha,
        "output_jsonl": str(output_jsonl.resolve()),
        "output_jsonl_sha256": output_sha,
        "unique_source_audio": len(unique_sources),
        "unique_target_reference_audio": len(unique_prompts),
        "generated_equals_reference_rows": sum(
            row["generated_audio"] == row["reference_audio"] for row in rows
        ),
        "generated_equals_source_rows": sum(
            row["generated_audio"] == row["source_audio"] for row in rows
        ),
        "prompt_equals_source_path_rows": 0,
        "prompt_equals_source_sha256_rows": paired_same_sha,
        "all_audio_present": True,
        "all_rows_calibration_only": True,
    }
    return rows, split_audit


def render_audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Batch-43 Ground truth source-self evaluation input audit",
        "",
        f"Display label: **{DISPLAY_NAME}**",
        "",
        f"> **Calibration only.** {CALIBRATION_WARNING}",
        "",
        "## Exact role mapping",
        "",
        "| canonical field | strict VC asset | interpretation |",
        "|---|---|---|",
        "| generated_audio | field 5 source/content WAV | untouched public recording; no model output |",
        "| reference_audio | same field 5 WAV | same-file speaker-scorer self check |",
        "| source_audio | same field 5 WAV | makes SIM(src) the same calibration value |",
        "| reference_text | field 4 transcript | ASR ceiling on the exact VC source content |",
        "| target_reference_audio | field 3 prompt WAV | provenance only; not used for self-SIM |",
        "",
        "Speaker SIM from this row must not be ranked against converted systems: it does "
        "not measure transfer to the target prompt speaker. ASR is comparable only as a "
        "raw-recording ceiling/lower bound on recognition error.",
        "",
        "## Splits",
        "",
        "| split | rows | unique source | unique target prompt | manifest SHA256 | output |",
        "|---|---:|---:|---:|---|---|",
    ]
    for language in ("en", "zh"):
        item = audit["splits"][language]
        lines.append(
            f"| {language.upper()} | {item['rows']} | {item['unique_source_audio']} | "
            f"{item['unique_target_reference_audio']} | `{item['manifest_sha256']}` | "
            f"`{item['output_jsonl']}` |"
        )
    lines.extend(
        [
            "",
            "## ZH-hard",
            "",
            "**N/A.** Official `zh/hardcase.lst` has only four fields: prompt text, "
            "prompt WAV, and target text, but no field-5 source/ground-truth waveform "
            "for the hard target text. Scoring the prompt WAV against prompt text would "
            "be a different task and is intentionally not fabricated here.",
            "",
        ]
    )
    return "\n".join(lines)


def build_all(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_audits: dict[str, Any] = {}
    output_paths: dict[str, str] = {}
    for language in ("en", "zh"):
        base = SPLITS[language]
        spec = SplitSpec(
            language=language,
            expected_rows=(args.expected_en if language == "en" else args.expected_zh),
            manifest_sha256=(
                args.expected_en_sha256
                if language == "en"
                else args.expected_zh_sha256
            ),
            test_set_id=base.test_set_id,
            manifest_name=base.manifest_name,
        )
        manifest = args.en_manifest if language == "en" else args.zh_manifest
        output_jsonl = output_dir / f"ground_truth_source_self.{language}.input.jsonl"
        _rows, split_audit = build_split(
            spec=spec,
            manifest=manifest,
            dataset_root=args.dataset_root,
            output_jsonl=output_jsonl,
        )
        split_audits[language] = split_audit
        output_paths[language] = str(output_jsonl)

    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "system_id": SYSTEM_ID,
        "display_name": DISPLAY_NAME,
        "protocol_label": (
            "Seed-TTS-Eval-derived VC EN567/ZH1194 internal-320-disjoint; "
            "source-self calibration"
        ),
        "calibration_only": True,
        "paired_target_speaker_ground_truth_available": False,
        "speaker_similarity_interpretation": (
            "generated/reference/source are the same field-5 source WAV; SIM(ref) "
            "and SIM(src) are same-file scorer self-similarity checks, not VC scores"
        ),
        "asr_interpretation": (
            "ASR transcribes the untouched field-5 source WAV against field-4; this "
            "is the raw-recording recognition-error ceiling for system content"
        ),
        "warning": CALIBRATION_WARNING,
        "splits": split_audits,
        "zh_hard": {
            "status": "not_applicable",
            "reason": (
                "Official hardcase.lst has four fields and no source/ground-truth "
                "waveform for the hard target text"
            ),
        },
    }
    audit_json = output_dir / "GROUND_TRUTH_SOURCE_SELF_AUDIT.json"
    audit_md = output_dir / "GROUND_TRUTH_SOURCE_SELF_AUDIT.md"
    atomic_write(
        audit_json,
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write(audit_md, render_audit_markdown(audit))
    print(
        json.dumps(
            {
                "status": "prepared",
                "system_id": SYSTEM_ID,
                "calibration_only": True,
                "inputs": output_paths,
                "audit_json": str(audit_json),
                "audit_markdown": str(audit_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--en-manifest",
        type=Path,
        default=DEFAULT_AUDIT_ROOT / SPLITS["en"].manifest_name,
    )
    parser.add_argument(
        "--zh-manifest",
        type=Path,
        default=DEFAULT_AUDIT_ROOT / SPLITS["zh"].manifest_name,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-en", type=int, default=SPLITS["en"].expected_rows)
    parser.add_argument("--expected-zh", type=int, default=SPLITS["zh"].expected_rows)
    parser.add_argument(
        "--expected-en-sha256", default=SPLITS["en"].manifest_sha256
    )
    parser.add_argument(
        "--expected-zh-sha256", default=SPLITS["zh"].manifest_sha256
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_en <= 0 or args.expected_zh <= 0:
        raise SystemExit("expected split sizes must be positive")
    try:
        build_all(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

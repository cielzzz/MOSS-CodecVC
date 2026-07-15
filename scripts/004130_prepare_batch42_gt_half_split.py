#!/usr/bin/env python3
"""Prepare front/back-half Ground Truth speaker-calibration inputs.

This is a distinct calibration from the existing Batch-42 same-file Ground
Truth row.  For every accepted source waveform:

* ``generated_audio`` is the non-overlapping front half;
* ``reference_audio`` is the non-overlapping back half;
* ``source_audio`` remains the untouched full source waveform;
* no ASR result is generated or copied into the new input.

Rows whose two halves do not both meet ``--min-half-seconds`` are recorded in
the per-language ledger with an explicit skip reason.  The scorer input JSONL
contains accepted rows only and is suitable for
``004082_run_unified_vc_eval.py evaluate --input-profile official_seedtts_vc
--speaker-scorer all``.

The output directory is new and must not already exist.  Existing same-file
inputs and scores are read-only provenance and are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "moss_codecvc.batch42_gt_half_split_input.v1"
LEDGER_SCHEMA = "moss_codecvc.batch42_gt_half_split_ledger.v1"
AUDIT_SCHEMA = "moss_codecvc.batch42_gt_half_split_audit.v1"
PARENT_INPUT_SCHEMA = "moss_codecvc.batch43_ground_truth_source_self_input.v1"
PARENT_AUDIT_SCHEMA = "moss_codecvc.batch43_ground_truth_source_self_audit.v1"
SYSTEM_ID = "ground_truth_half_split"
DISPLAY_NAME = "Ground truth (front/back half calibration)"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARENT_ROOT = (
    PROJECT_ROOT
    / "testset/outputs/batch43_ground_truth_source_self_inputs_20260712"
)
DEFAULT_EN_INPUT = PARENT_ROOT / "ground_truth_source_self.en.input.jsonl"
DEFAULT_ZH_INPUT = PARENT_ROOT / "ground_truth_source_self.zh.input.jsonl"
DEFAULT_PARENT_AUDIT = PARENT_ROOT / "GROUND_TRUTH_SOURCE_SELF_AUDIT.json"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "testset/outputs/batch42_ground_truth_half_split_inputs_20260713"
)

EXPECTED_PARENT_SHA256 = {
    "en": "f27c28e886f46e65bf31c877c3b7612b78648e4edd656b557053507cd4503504",
    "zh": "740dcb220a6e71737dd102b5be5a30d36dd713ab1dcc1e21bf6eaa3218a4a399",
    "audit": "8e3a23cf7784ba49428b0ab83a3ba1b78da090d844245148015bbbda9e9f3e5e",
}
EXPECTED_ROWS = {"en": 567, "zh": 1194}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing/empty artifact: {path}")
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def staged_artifact(path: Path, final_path: Path) -> dict[str, Any]:
    """Hash a staging file while registering its post-rename destination."""

    spec = artifact(path)
    spec["path"] = str(final_path.resolve())
    return spec


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def render_jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def stable_half_uid(parent_uid: str, source_sha256: str, split_frame: int) -> str:
    material = f"gt_half_split\0{parent_uid}\0{source_sha256}\0{split_frame}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def validate_parent_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    language: str,
    expected_rows: int,
    input_path: Path,
    expected_test_set_id: str,
) -> None:
    if len(rows) != expected_rows:
        raise ValueError(
            f"{language}: parent rows={len(rows)}, expected {expected_rows}"
        )
    case_ids: set[str] = set()
    case_uids: set[str] = set()
    indices: list[int] = []
    for line_number, row in enumerate(rows, start=1):
        context = f"{input_path}:{line_number}"
        expected_scalars = {
            "schema_version": PARENT_INPUT_SCHEMA,
            "record_type": "ground_truth_source_self_eval_input",
            "status": "ok",
            "system_id": "ground_truth",
            "language": language,
            "test_set_id": expected_test_set_id,
        }
        drift = {
            key: {"expected": value, "actual": row.get(key)}
            for key, value in expected_scalars.items()
            if row.get(key) != value
        }
        if drift:
            raise ValueError(f"{context}: parent identity drift: {drift}")
        source = str(row.get("source_audio") or "")
        if (
            not source
            or row.get("generated_audio") != source
            or row.get("reference_audio") != source
        ):
            raise ValueError(
                f"{context}: parent is not the registered same-file source-self row"
            )
        case_id = str(row.get("case_id") or "")
        case_uid = str(row.get("case_uid") or "")
        if not case_id or case_id in case_ids:
            raise ValueError(f"{context}: missing/duplicate case_id={case_id!r}")
        if not case_uid or case_uid in case_uids:
            raise ValueError(f"{context}: missing/duplicate case_uid={case_uid!r}")
        case_ids.add(case_id)
        case_uids.add(case_uid)
        try:
            indices.append(int(row.get("input_index")))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}: invalid input_index") from exc
    if indices != list(range(expected_rows)):
        raise ValueError(f"{language}: parent input_index is not contiguous")


def validate_parent_audit(
    audit: Mapping[str, Any],
    *,
    audit_path: Path,
    inputs: Mapping[str, Path],
    expected_rows: Mapping[str, int],
    expected_hashes: Mapping[str, str],
) -> dict[str, str]:
    if audit.get("schema_version") != PARENT_AUDIT_SCHEMA:
        raise ValueError(f"wrong parent audit schema: {audit.get('schema_version')!r}")
    if audit.get("system_id") != "ground_truth" or audit.get("calibration_only") is not True:
        raise ValueError("parent audit is not the registered Ground Truth calibration")
    splits = audit.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("parent audit lacks split registrations")
    test_set_ids: dict[str, str] = {}
    for language in ("en", "zh"):
        split = splits.get(language)
        if not isinstance(split, dict):
            raise ValueError(f"parent audit lacks {language} split")
        actual_path = inputs[language].resolve()
        actual_sha = sha256_file(actual_path)
        expected_sha = expected_hashes[language]
        if actual_sha != expected_sha:
            raise ValueError(
                f"{language}: parent input SHA256={actual_sha}, expected {expected_sha}"
            )
        if Path(str(split.get("output_jsonl") or "")).resolve() != actual_path:
            raise ValueError(f"{language}: parent audit input path drift")
        if split.get("output_jsonl_sha256") != actual_sha:
            raise ValueError(f"{language}: parent audit input SHA registration drift")
        if int(split.get("rows") or -1) != expected_rows[language]:
            raise ValueError(f"{language}: parent audit row-count drift")
        if int(split.get("generated_equals_reference_rows") or -1) != expected_rows[language]:
            raise ValueError(f"{language}: parent audit is not same-file calibration")
        test_set_id = str(split.get("test_set_id") or "")
        if not test_set_id:
            raise ValueError(f"{language}: missing parent test_set_id")
        test_set_ids[language] = test_set_id
    actual_audit_sha = sha256_file(audit_path)
    if actual_audit_sha != expected_hashes["audit"]:
        raise ValueError(
            f"parent audit SHA256={actual_audit_sha}, expected {expected_hashes['audit']}"
        )
    return test_set_ids


def split_pcm_wave(
    source: Path,
    *,
    front_output: Path,
    back_output: Path,
    min_half_seconds: float,
) -> tuple[dict[str, Any] | None, str]:
    if not source.is_file() or source.stat().st_size < 44:
        return None, "source_missing_or_empty"
    if source.suffix.lower() != ".wav":
        return None, "source_not_wav"
    try:
        with wave.open(str(source), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            compression = handle.getcomptype()
            frames = handle.readframes(frame_count)
    except (wave.Error, EOFError, OSError) as exc:
        return None, f"invalid_wave:{type(exc).__name__}"
    if compression != "NONE":
        return None, f"unsupported_compression:{compression}"
    if channels < 1 or sample_width not in {1, 2, 3, 4} or sample_rate <= 0:
        return None, "invalid_wave_parameters"
    frame_width = channels * sample_width
    if len(frames) != frame_count * frame_width:
        return None, "truncated_wave_payload"
    split_frame = frame_count // 2
    front_frames = split_frame
    back_frames = frame_count - split_frame
    if front_frames <= 0 or back_frames <= 0:
        return None, "insufficient_frames_for_two_halves"
    front_seconds = front_frames / sample_rate
    back_seconds = back_frames / sample_rate
    if front_seconds + 1e-12 < min_half_seconds:
        return None, "front_below_min_half_seconds"
    if back_seconds + 1e-12 < min_half_seconds:
        return None, "back_below_min_half_seconds"
    boundary_byte = split_frame * frame_width
    front_bytes = frames[:boundary_byte]
    back_bytes = frames[boundary_byte:]

    def write_half(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(sample_width)
            handle.setframerate(sample_rate)
            handle.setcomptype("NONE", "not compressed")
            handle.writeframes(data)

    write_half(front_output, front_bytes)
    write_half(back_output, back_bytes)
    return (
        {
            "sample_rate": sample_rate,
            "channels": channels,
            "sample_width_bytes": sample_width,
            "source_frames": frame_count,
            "split_frame": split_frame,
            "front_frame_range": [0, split_frame],
            "back_frame_range": [split_frame, frame_count],
            "front_frames": front_frames,
            "back_frames": back_frames,
            "front_seconds": front_seconds,
            "back_seconds": back_seconds,
            "balance_delta_frames": abs(front_frames - back_frames),
            "overlap_frames": 0,
            "gap_frames": 0,
        },
        "keep",
    )


def build_language(
    *,
    language: str,
    rows: Sequence[Mapping[str, Any]],
    parent_input: Path,
    parent_input_sha256: str,
    parent_audit: Path,
    parent_audit_sha256: str,
    test_set_id: str,
    staging_root: Path,
    final_root: Path,
    min_half_seconds: float,
) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    source_hash_cache: dict[Path, str] = {}
    for parent_index, row in enumerate(rows):
        case_id = str(row["case_id"])
        parent_uid = str(row["case_uid"])
        source = Path(str(row["source_audio"])).expanduser().resolve()
        source_sha = ""
        if source.is_file():
            if source not in source_hash_cache:
                source_hash_cache[source] = sha256_file(source)
            source_sha = source_hash_cache[source]
        provisional_uid = hashlib.sha256(
            f"{language}\0{parent_uid}\0{source}".encode("utf-8")
        ).hexdigest()[:24]
        relative_front = Path("audio") / language / "front" / f"{parent_index:06d}_{provisional_uid}.wav"
        relative_back = Path("audio") / language / "back" / f"{parent_index:06d}_{provisional_uid}.wav"
        front = staging_root / relative_front
        back = staging_root / relative_back
        final_front = final_root / relative_front
        final_back = final_root / relative_back
        split, reason = split_pcm_wave(
            source,
            front_output=front,
            back_output=back,
            min_half_seconds=min_half_seconds,
        )
        ledger_row: dict[str, Any] = {
            "schema_version": LEDGER_SCHEMA,
            "language": language,
            "parent_input_index": parent_index,
            "case_id": case_id,
            "parent_case_uid": parent_uid,
            "source_audio": str(source),
            "source_audio_sha256": source_sha,
            "status": "kept" if split is not None else "skipped",
            "reason": reason,
        }
        reasons[reason] += 1
        if split is None:
            ledger.append(ledger_row)
            continue
        half_uid = stable_half_uid(parent_uid, source_sha, int(split["split_frame"]))
        # The provisional name is deterministic from the parent identity; keep
        # half_uid separately so file paths never depend on audio bytes written
        # during the same transaction.
        front_artifact = staged_artifact(front, final_front)
        back_artifact = staged_artifact(back, final_back)
        kept_index = len(accepted)
        output_row = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "ground_truth_half_split_eval_input",
            "status": "ok",
            "system_id": SYSTEM_ID,
            "display_name": DISPLAY_NAME,
            "test_set_id": test_set_id,
            "case_id": case_id,
            "case_uid": half_uid,
            "input_index": kept_index,
            "parent_input_index": parent_index,
            "language": language,
            "generated_audio": str(final_front.resolve()),
            "reference_audio": str(final_back.resolve()),
            "source_audio": str(source),
            "reference_text": str(row.get("reference_text") or ""),
            "target_reference_audio": str(row.get("target_reference_audio") or ""),
            "half_split": {
                **split,
                "min_half_seconds": min_half_seconds,
                "front_audio": front_artifact,
                "back_audio": back_artifact,
            },
            "evaluation_contract": {
                "calibration_only": True,
                "speaker_similarity_interpretation": (
                    "SIM(ref) compares the front half to the non-overlapping back half "
                    "of the same untouched source recording"
                ),
                "source_similarity_interpretation": (
                    "SIM(src) compares the front half to the full untouched source recording"
                ),
                "same_file_calibration_replaced": False,
                "run_speaker_backends": [
                    "wavlm_large_sv",
                    "eres2net",
                    "speechbrain_ecapa",
                ],
                "run_asr": False,
                "asr_floor_source": (
                    "reuse existing Batch-43 same-file Ground Truth raw-source ASR floor"
                ),
            },
            "provenance": {
                "parent_input": str(parent_input.resolve()),
                "parent_input_sha256": parent_input_sha256,
                "parent_audit": str(parent_audit.resolve()),
                "parent_audit_sha256": parent_audit_sha256,
                "parent_case_uid": parent_uid,
                "parent_source_audio_sha256": source_sha,
                "split_policy": "frame_midpoint; [0,mid) and [mid,n); no overlap/no gap",
            },
        }
        accepted.append(output_row)
        ledger_row.update(
            {
                "kept_input_index": kept_index,
                "case_uid": half_uid,
                "front_audio": front_artifact,
                "back_audio": back_artifact,
                "half_split": split,
            }
        )
        ledger.append(ledger_row)

    input_jsonl = staging_root / f"ground_truth_half_split.{language}.input.jsonl"
    ledger_jsonl = staging_root / f"ground_truth_half_split.{language}.ledger.jsonl"
    write_text(input_jsonl, render_jsonl(accepted))
    write_text(ledger_jsonl, render_jsonl(ledger))
    return {
        "language": language,
        "test_set_id": test_set_id,
        "parent_rows": len(rows),
        "kept_rows": len(accepted),
        "skipped_rows": len(rows) - len(accepted),
        "reason_counts": dict(sorted(reasons.items())),
        "input_index_coverage": [0, len(accepted) - 1] if accepted else [],
        "input_jsonl": staged_artifact(
            input_jsonl,
            final_root / f"ground_truth_half_split.{language}.input.jsonl",
        ),
        "ledger_jsonl": staged_artifact(
            ledger_jsonl,
            final_root / f"ground_truth_half_split.{language}.ledger.jsonl",
        ),
        "audio_files": len(accepted) * 2,
        "all_kept_halves_non_overlapping": all(
            row["half_split"]["overlap_frames"] == 0
            and row["half_split"]["gap_frames"] == 0
            for row in accepted
        ),
        "max_balance_delta_frames": max(
            (int(row["half_split"]["balance_delta_frames"]) for row in accepted),
            default=0,
        ),
    }


def render_audit_markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Batch-42 Ground Truth front/back-half calibration preparation",
        "",
        "- This is a separate speaker-scorer calibration; existing same-file Ground Truth outputs are untouched.",
        f"- Minimum duration per half: `{audit['min_half_seconds']:.3f}` seconds.",
        "- Split policy: frame midpoint, front `[0, mid)`, back `[mid, n)`, no overlap and no gap.",
        "- Scoring contract: three speaker backends only; no ASR rerun.",
        "",
        "| Language | Parent rows | Kept | Skipped | Audio halves | Max balance delta (frames) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for language in ("en", "zh"):
        split = audit["splits"][language]
        lines.append(
            f"| {language} | {split['parent_rows']} | {split['kept_rows']} | "
            f"{split['skipped_rows']} | {split['audio_files']} | "
            f"{split['max_balance_delta_frames']} |"
        )
    lines.extend(["", "## Skip reasons", ""])
    for language in ("en", "zh"):
        lines.append(f"- {language}: `{json.dumps(audit['splits'][language]['reason_counts'], sort_keys=True)}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `sim_ref`: front half versus back half of the same source recording.",
            "- `sim_src`: front half versus the full source recording.",
            "- ASR CER/WER floor remains the already completed same-file raw-source evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def build_all(args: argparse.Namespace) -> dict[str, Any]:
    if not math.isfinite(args.min_half_seconds) or args.min_half_seconds <= 0:
        raise ValueError("--min-half-seconds must be a positive finite number")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    parent_audit_path = args.parent_audit.expanduser().resolve(strict=True)
    inputs = {
        "en": args.en_input.expanduser().resolve(strict=True),
        "zh": args.zh_input.expanduser().resolve(strict=True),
    }
    expected_rows = {"en": args.expected_en, "zh": args.expected_zh}
    expected_hashes = {
        "en": args.expected_en_sha256,
        "zh": args.expected_zh_sha256,
        "audit": args.expected_parent_audit_sha256,
    }
    parent_audit = read_json(parent_audit_path, label="parent audit")
    test_set_ids = validate_parent_audit(
        parent_audit,
        audit_path=parent_audit_path,
        inputs=inputs,
        expected_rows=expected_rows,
        expected_hashes=expected_hashes,
    )
    parent_rows = {language: read_jsonl(path) for language, path in inputs.items()}
    for language in ("en", "zh"):
        validate_parent_rows(
            parent_rows[language],
            language=language,
            expected_rows=expected_rows[language],
            input_path=inputs[language],
            expected_test_set_id=test_set_ids[language],
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.staging-{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"staging path already exists: {staging}")
    staging.mkdir()
    try:
        parent_audit_sha = sha256_file(parent_audit_path)
        splits = {
            language: build_language(
                language=language,
                rows=parent_rows[language],
                parent_input=inputs[language],
                parent_input_sha256=expected_hashes[language],
                parent_audit=parent_audit_path,
                parent_audit_sha256=parent_audit_sha,
                test_set_id=test_set_ids[language],
                staging_root=staging,
                final_root=output_dir,
                min_half_seconds=args.min_half_seconds,
            )
            for language in ("en", "zh")
        }
        audit = {
            "schema_version": AUDIT_SCHEMA,
            "status": "prepared",
            "generated_at_utc": utc_now(),
            "system_id": SYSTEM_ID,
            "display_name": DISPLAY_NAME,
            "calibration_only": True,
            "min_half_seconds": args.min_half_seconds,
            "split_policy": "frame_midpoint; [0,mid) + [mid,n); no overlap/no gap",
            "existing_same_file_results": {
                "status": "untouched",
                "parent_audit": artifact(parent_audit_path),
                "inputs": {language: artifact(path) for language, path in inputs.items()},
            },
            "scoring_contract": {
                "evaluator": "scripts/004082_run_unified_vc_eval.py",
                "input_profile": "official_seedtts_vc",
                "speaker_scorers": [
                    "wavlm_large_sv",
                    "eres2net",
                    "speechbrain_ecapa",
                ],
                "asr_backends": [],
                "asr_policy": "do not rerun; retain existing same-file raw-source ASR floor",
            },
            "splits": splits,
            "zh_hard": {
                "status": "not_applicable",
                "reason": "no audited pure-VC ZH-hard source waveform",
            },
        }
        audit_json = staging / "GROUND_TRUTH_HALF_SPLIT_AUDIT.json"
        audit_md = staging / "GROUND_TRUTH_HALF_SPLIT_AUDIT.md"
        write_text(audit_md, render_audit_markdown(audit) + "\n")
        audit["outputs"] = {
            "audit_json": {
                "path": str((output_dir / audit_json.name).resolve()),
                "note": "self-describing audit; no recursive self-hash",
            },
            "audit_markdown": staged_artifact(
                audit_md, output_dir / audit_md.name
            ),
        }
        write_json(audit_json, audit)
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return read_json(output_dir / "GROUND_TRUTH_HALF_SPLIT_AUDIT.json", label="output audit")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--en-input", type=Path, default=DEFAULT_EN_INPUT)
    parser.add_argument("--zh-input", type=Path, default=DEFAULT_ZH_INPUT)
    parser.add_argument("--parent-audit", type=Path, default=DEFAULT_PARENT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-en", type=int, default=EXPECTED_ROWS["en"])
    parser.add_argument("--expected-zh", type=int, default=EXPECTED_ROWS["zh"])
    parser.add_argument(
        "--expected-en-sha256", default=EXPECTED_PARENT_SHA256["en"]
    )
    parser.add_argument(
        "--expected-zh-sha256", default=EXPECTED_PARENT_SHA256["zh"]
    )
    parser.add_argument(
        "--expected-parent-audit-sha256", default=EXPECTED_PARENT_SHA256["audit"]
    )
    parser.add_argument("--min-half-seconds", type=float, default=1.5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit = build_all(args)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

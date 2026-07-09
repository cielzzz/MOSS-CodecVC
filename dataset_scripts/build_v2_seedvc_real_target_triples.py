#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT_ROOT = Path(
    "/inspire/qb-ilm/project/embodied-multimodality/public/xyzhang/data/"
    "VC_train/v2_real_target_pilot_20260706"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/inspire/qb-ilm/project/embodied-multimodality/public/xyzhang/data/"
    "VC_train/v2_real_target_seedvc_triples_pilot_20260707"
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

CHANNEL_RISK_NOTE = (
    "u1 and u2 are cut from the same long recording when their episode ids match, so they can share microphone, "
    "room, noise, EQ and codec characteristics. A model may copy reference-channel cues instead of learning "
    "speaker identity."
)
CHANNEL_MITIGATIONS = [
    "Apply random ref-side channel augmentation to u2 before training or online during training.",
    "Prefer cross-recording u2 for the same speaker when a reliable speaker/session id is available.",
    "Keep a cross-channel validation subset where source, ref and target sessions differ.",
]


def stable_id(*values: Any, length: int = 12) -> str:
    payload = "\x1f".join(str(value) for value in values)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    return re.sub(r"\s+", " ", text)


def text_key(value: Any) -> str:
    text = normalize_text(value).lower()
    chars = [ch for ch in text if not unicodedata.category(ch).startswith("P") and not ch.isspace()]
    return "".join(chars)


def texts_differ(left: Any, right: Any) -> bool:
    left_key = text_key(left)
    right_key = text_key(right)
    return bool(left_key and right_key and left_key != right_key)


def path_ok(path: str, min_bytes: int) -> bool:
    if not path:
        return False
    try:
        item = Path(path)
        return item.exists() and item.stat().st_size >= min_bytes
    except OSError:
        return False


def rebase_path(value: Any, replacements: list[tuple[str, str]]) -> str:
    text = str(value or "")
    for old, new in replacements:
        if old and text.startswith(old):
            return new + text[len(old) :]
    return text


def dataset_from_row(row: dict[str, Any]) -> str:
    return str(row.get("dataset_name") or "unknown_dataset")


def extract_episode_id(relative_path: Any) -> str:
    text = str(relative_path or "")
    match = re.search(r"(?:^|/)tts_audio_segment/([^/]+)/", text)
    if match:
        return match.group(1)
    parts = [part for part in text.split("/") if part]
    for idx, part in enumerate(parts):
        if part in {"tts_audio_segment", "audio_segment"} and idx + 1 < len(parts):
            return parts[idx + 1]
    if len(parts) >= 2:
        return parts[-2]
    return ""


def speaker_id(dataset_name: str, episode_id: str, fallback_audio: str) -> str:
    if episode_id:
        return f"{dataset_name}:{episode_id}"
    return f"{dataset_name}:audio:{stable_id(fallback_audio, length=16)}"


def seedvc_output_path(output_root: Path, row: dict[str, Any], ordinal: int, *, use_input_path: bool) -> str:
    input_path = str(row.get("u1_prime_source_audio_path") or "")
    if use_input_path and input_path:
        return input_path
    dataset_name = dataset_from_row(row)
    digest = stable_id(
        row.get("u1_target_audio_path"),
        row.get("u2_timbre_ref_audio_path"),
        row.get("u1_prime_source_seedvc_job_id"),
        ordinal,
        length=12,
    )
    return str((output_root / "seedvc_sources" / dataset_name / f"{ordinal:08d}_{digest}.wav").resolve(strict=False))


def load_seedvc_jobs(path: Path, replacements: list[tuple[str, str]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_output: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return by_id, by_output
    for job in iter_jsonl(path):
        job = rebase_row_paths(job, replacements)
        job_id = str(job.get("job_id") or "")
        output = str(job.get("output_audio") or job.get("target_audio") or "")
        if job_id:
            by_id[job_id] = job
        if output:
            by_output[output] = job
    return by_id, by_output


def rebase_row_paths(row: dict[str, Any], replacements: list[tuple[str, str]]) -> dict[str, Any]:
    if not replacements:
        return row
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, str):
            out[key] = rebase_path(value, replacements)
        elif isinstance(value, dict):
            out[key] = rebase_row_paths(value, replacements)
        elif isinstance(value, list):
            out[key] = [rebase_row_paths(item, replacements) if isinstance(item, dict) else rebase_path(item, replacements) if isinstance(item, str) else item for item in value]
        else:
            out[key] = value
    return out


def find_old_job(row: dict[str, Any], jobs_by_id: dict[str, dict[str, Any]], jobs_by_output: dict[str, dict[str, Any]]) -> dict[str, Any]:
    job_id = str(row.get("u1_prime_source_seedvc_job_id") or "")
    if job_id and job_id in jobs_by_id:
        return jobs_by_id[job_id]
    old_output = str(row.get("u1_prime_source_audio_path") or "")
    if old_output and old_output in jobs_by_output:
        return jobs_by_output[old_output]
    return {}


def build_seedvc_job(
    row: dict[str, Any],
    old_job: dict[str, Any],
    *,
    output_audio: str,
    ordinal: int,
    run_name: str,
) -> dict[str, Any]:
    u1_audio = str(row.get("u1_target_audio_path") or "")
    u1_text = normalize_text(row.get("u1_text"))
    donor_audio = str(row.get("u1_prime_seedvc_perturb_timbre_ref_path") or old_job.get("timbre_ref_audio") or "")
    donor_text = normalize_text(old_job.get("timbre_ref_text"))
    source_speaker = str(old_job.get("source_speaker_id") or "")
    donor_speaker = str(old_job.get("timbre_ref_speaker_id") or old_job.get("target_speaker_id") or "")
    job_id = f"{run_name}:source_seedvc:{ordinal:08d}:{stable_id(u1_audio, donor_audio, output_audio)}"
    return {
        "job_id": job_id,
        "pair_type": "v2_real_target_source_perturb",
        "prosody_ref_audio": u1_audio,
        "prosody_ref_text": u1_text,
        "source_audio": u1_audio,
        "source_text": u1_text,
        "timbre_ref_audio": donor_audio,
        "timbre_ref_text": donor_text,
        "target_audio": output_audio,
        "output_audio": output_audio,
        "target_text": u1_text,
        "language": row.get("language") or "",
        "source_speaker_id": source_speaker,
        "timbre_ref_speaker_id": donor_speaker,
        "target_speaker_id": donor_speaker,
        "metadata": {
            "route": "v2_real_target_back_translation_source_generation",
            "construction_rule": "u1_real_to_random_other_timbre_source; final_target_is_u1_real",
            "seedvc_backend": "seed_vc_v1_zero_shot_voice_conversion",
            "real_target_audio": u1_audio,
            "real_target_text": u1_text,
            "final_timbre_ref_audio": row.get("u2_timbre_ref_audio_path") or "",
            "final_timbre_ref_text": normalize_text(row.get("u2_text")),
            "source_audio_uri_kind": "local_materialized_file",
            "requires_audio_materialization": True,
            "old_source_seedvc_job_id": row.get("u1_prime_source_seedvc_job_id") or "",
            "old_source_seedvc_output_audio": row.get("u1_prime_source_audio_path") or "",
        },
    }


def channel_risk(row: dict[str, Any]) -> dict[str, Any]:
    u1_episode = extract_episode_id(row.get("label"))
    u2_episode = extract_episode_id(row.get("reference"))
    same_episode = bool(u1_episode and u2_episode and u1_episode == u2_episode)
    return {
        "same_long_audio_channel_risk": same_episode,
        "u1_episode_id": u1_episode,
        "u2_episode_id": u2_episode,
        "risk_note": CHANNEL_RISK_NOTE if same_episode else "",
        "applied_mitigations": {
            "source_side_seedvc_perturbation": True,
            "ref_side_channel_augmentation": False,
            "cross_recording_ref": False,
        },
        "recommended_mitigations": CHANNEL_MITIGATIONS if same_episode else [],
    }


def base_manifest_row(
    row: dict[str, Any],
    job: dict[str, Any],
    *,
    mode: str,
    ordinal: int,
    run_name: str,
    pending: bool,
) -> dict[str, Any]:
    dataset_name = dataset_from_row(row)
    u1_audio = str(row.get("u1_target_audio_path") or "")
    u2_audio = str(row.get("u2_timbre_ref_audio_path") or "")
    u1_prime = str(job.get("output_audio") or "")
    u1_text = normalize_text(row.get("u1_text"))
    u2_text = normalize_text(row.get("u2_text"))
    label_episode = extract_episode_id(row.get("label"))
    ref_episode = extract_episode_id(row.get("reference"))
    target_speaker = speaker_id(dataset_name, label_episode, u1_audio)
    timbre_speaker = speaker_id(dataset_name, ref_episode, u2_audio)
    source_speaker = f"synthetic_source:{job.get('target_speaker_id') or stable_id(u1_prime, length=16)}"
    sample_id = f"{run_name}:{mode}:{ordinal:08d}:{stable_id(u1_audio, u2_audio, u1_prime, mode)}"
    text = "<NO_TEXT>" if mode == "no_text" else u1_text
    return {
        "sample_id": sample_id,
        "source_audio": u1_prime,
        "source_text": u1_text,
        "timbre_ref_audio": u2_audio,
        "timbre_ref_text": u2_text,
        "target_audio": u1_audio,
        "target_text": u1_text,
        "text": text,
        "language": row.get("language") or "",
        "source_speaker_id": source_speaker,
        "timbre_ref_speaker_id": timbre_speaker,
        "target_speaker_id": target_speaker,
        "source_gender": "unknown",
        "timbre_ref_gender": "unknown",
        "target_gender": "unknown",
        "pair_type": "v2_real_target_seedvc_source_no_text" if mode == "no_text" else "v2_real_target_seedvc_source_text",
        "instruction": NO_TEXT_INSTRUCTION if mode == "no_text" else TEXT_INSTRUCTION,
        "text_prosody_instruction": TEXT_INSTRUCTION if mode == "text" else None,
        "preferred_emit_mode": mode,
        "source_audio_pending": bool(pending),
        "source_generation_job_id": job.get("job_id"),
        "source_generation_jobs_jsonl": "source_seedvc_jobs.jsonl",
        "v2_real_target": {
            "target_is_real_audio": True,
            "source_is_seedvc_output": True,
            "teacher_pollution_side": "source_only",
            "source_generation_route": "Seed-VC(u1 real -> random other timbre) = u1_prime",
        },
        "text_policy": {
            "target_text_equals_source_text": text_key(u1_text) == text_key(row.get("u1_text")),
            "target_text_differs_from_timbre_text": texts_differ(u1_text, u2_text),
            "target_text_differs_from_source_text": False,
            "source_target_text_difference_is_impossible_for_seedvc_u1_prime": True,
        },
        "channel_shortcut_risk": channel_risk(row),
        "meta": {
            "u1": {
                "role": "real_target",
                "audio": u1_audio,
                "source_uri": row.get("u1_target_audio_source_uri") or "",
                "text": u1_text,
                "duration_sec": row.get("u1_duration_sec"),
            },
            "u2": {
                "role": "same_speaker_timbre_ref",
                "audio": u2_audio,
                "source_uri": row.get("u2_timbre_ref_audio_source_uri") or "",
                "text": u2_text,
                "duration_sec": row.get("u2_duration_sec"),
            },
            "u1_prime": {
                "role": "seedvc_source",
                "audio": u1_prime,
                "pending": bool(pending),
                "seedvc_input_u1_audio": job.get("prosody_ref_audio") or "",
                "seedvc_perturb_timbre_ref_audio": job.get("timbre_ref_audio") or "",
                "seedvc_perturb_timbre_ref_text": job.get("timbre_ref_text") or "",
            },
            "source_fields": {
                "dataset_name": row.get("dataset_name") or "",
                "lance_table_uri": row.get("lance_table_uri") or "",
                "label": row.get("label") or "",
                "reference": row.get("reference") or "",
                "segs": row.get("segs"),
                "similarity": row.get("similarity"),
                "label_idx": row.get("label_idx"),
                "ref_idx": row.get("ref_idx"),
                "source_line_index": row.get("source_line_index"),
                "source_prepare_jsonl": row.get("source_prepare_jsonl") or "",
                "source_resultf_jsonl": row.get("source_resultf_jsonl") or "",
            },
            "source_seedvc_job": job,
        },
    }


def simple_row(manifest: dict[str, Any], source_row: dict[str, Any], *, mode: str, ready: bool) -> dict[str, Any]:
    meta = manifest["meta"]
    u1 = meta["u1"]
    u2 = meta["u2"]
    u1_prime = meta["u1_prime"]
    source_fields = meta["source_fields"]
    return {
        "sample_id": manifest["sample_id"],
        "mode": mode,
        "ready": bool(ready),
        "dataset_name": source_fields["dataset_name"],
        "language": manifest.get("language") or "",
        "u1_target_audio_path": u1["audio"],
        "u1_text": u1["text"],
        "u2_timbre_ref_audio_path": u2["audio"],
        "u2_text": u2["text"],
        "u1_prime_source_audio_path": u1_prime["audio"],
        "u1_prime_pending": bool(u1_prime["pending"]),
        "source_audio": manifest["source_audio"],
        "timbre_ref_audio": manifest["timbre_ref_audio"],
        "target_audio": manifest["target_audio"],
        "target_text": manifest["target_text"],
        "text": manifest["text"],
        "label": source_fields["label"],
        "reference": source_fields["reference"],
        "segs": source_fields["segs"],
        "similarity": source_fields["similarity"],
        "label_idx": source_fields["label_idx"],
        "ref_idx": source_fields["ref_idx"],
        "same_long_audio_channel_risk": manifest["channel_shortcut_risk"]["same_long_audio_channel_risk"],
        "u1_episode_id": manifest["channel_shortcut_risk"]["u1_episode_id"],
        "u2_episode_id": manifest["channel_shortcut_risk"]["u2_episode_id"],
        "text_target_differs_from_timbre_text": manifest["text_policy"]["target_text_differs_from_timbre_text"],
        "text_target_differs_from_source_text": manifest["text_policy"]["target_text_differs_from_source_text"],
        "source_seedvc_job_id": manifest.get("source_generation_job_id") or "",
        "source_resultf_jsonl": source_fields["source_resultf_jsonl"],
        "source_prepare_jsonl": source_fields["source_prepare_jsonl"],
        "u1_target_audio_source_uri": u1["source_uri"],
        "u2_timbre_ref_audio_source_uri": u2["source_uri"],
        "lance_table_uri": source_fields["lance_table_uri"],
        "source_row_sample_id": source_row.get("sample_id") or "",
    }


def text_row_allowed(row: dict[str, Any], args: argparse.Namespace, stats: Counter) -> bool:
    if not args.emit_v2_u1_text:
        stats["text_skip_disabled_for_v2_seedvc_u1_route"] += 1
        return False
    u1_text = normalize_text(row.get("u1_text"))
    u2_text = normalize_text(row.get("u2_text"))
    if not u1_text:
        stats["text_skip_missing_target_text"] += 1
        return False
    if args.require_timbre_text_for_text and not u2_text:
        stats["text_skip_missing_timbre_text"] += 1
        return False
    if args.require_target_text_diff_timbre and u2_text and not texts_differ(u1_text, u2_text):
        stats["text_skip_target_text_not_diff_timbre"] += 1
        return False
    if args.require_target_text_diff_source:
        stats["text_skip_target_text_not_diff_source"] += 1
        return False
    return True


def build_replacements(args: argparse.Namespace) -> list[tuple[str, str]]:
    replacements: list[tuple[str, str]] = []
    for item in args.path_rewrite:
        if "=" not in item:
            raise SystemExit(f"--path-rewrite expects OLD=NEW, got: {item}")
        old, new = item.split("=", 1)
        replacements.append((old, new))
    return replacements


def prepare(args: argparse.Namespace) -> int:
    input_root = Path(args.input_root).expanduser().resolve(strict=False)
    output_root = Path(args.output_root).expanduser().resolve(strict=False)
    input_simple = Path(args.input_simple_jsonl).expanduser().resolve(strict=False) if args.input_simple_jsonl else input_root / "no_text.train.simple.jsonl"
    input_jobs = Path(args.input_seedvc_jobs_jsonl).expanduser().resolve(strict=False) if args.input_seedvc_jobs_jsonl else input_root / "source_seedvc_jobs.jsonl"
    replacements = build_replacements(args)

    if not input_simple.exists():
        raise SystemExit(f"input simple JSONL not found: {input_simple}")

    jobs_by_id, jobs_by_output = load_seedvc_jobs(input_jobs, replacements)
    stats: Counter = Counter()
    source_jobs: list[dict[str, Any]] = []
    pending_no_text: list[dict[str, Any]] = []
    pending_text: list[dict[str, Any]] = []
    simple_no_text: list[dict[str, Any]] = []
    simple_text: list[dict[str, Any]] = []

    rows_seen = 0
    for row in iter_jsonl(input_simple):
        rows_seen += 1
        if args.max_rows > 0 and rows_seen > args.max_rows:
            break
        row = rebase_row_paths(row, replacements)
        stats["input_rows"] += 1
        if args.require_existing_u1_u2:
            if not path_ok(str(row.get("u1_target_audio_path") or ""), 1):
                stats["skip_missing_u1_audio"] += 1
                continue
            if not path_ok(str(row.get("u2_timbre_ref_audio_path") or ""), 1):
                stats["skip_missing_u2_audio"] += 1
                continue
        risk = channel_risk(row)
        if args.require_cross_recording_ref and risk["same_long_audio_channel_risk"]:
            stats["skip_same_episode_ref"] += 1
            continue
        old_job = find_old_job(row, jobs_by_id, jobs_by_output)
        output_audio = seedvc_output_path(output_root, row, len(source_jobs), use_input_path=args.use_input_u1_prime_paths)
        job = build_seedvc_job(row, old_job, output_audio=output_audio, ordinal=len(source_jobs), run_name=args.run_name)
        if not job.get("timbre_ref_audio"):
            stats["skip_missing_seedvc_perturb_timbre_ref_audio"] += 1
            continue

        source_jobs.append(job)
        no_text_manifest = base_manifest_row(
            row,
            job,
            mode="no_text",
            ordinal=len(pending_no_text),
            run_name=args.run_name,
            pending=True,
        )
        pending_no_text.append(no_text_manifest)
        simple_no_text.append(simple_row(no_text_manifest, row, mode="no_text", ready=False))
        stats["no_text_rows"] += 1
        if risk["same_long_audio_channel_risk"]:
            stats["same_long_audio_channel_risk_rows"] += 1

        if text_row_allowed(row, args, stats):
            text_manifest = base_manifest_row(
                row,
                job,
                mode="text",
                ordinal=len(pending_text),
                run_name=args.run_name,
                pending=True,
            )
            pending_text.append(text_manifest)
            simple_text.append(simple_row(text_manifest, row, mode="text", ready=False))
            stats["text_rows"] += 1

    output_root.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        for path in [
            output_root / "source_seedvc_jobs.jsonl",
            output_root / "no_text.train.pending.manifest.jsonl",
            output_root / "text.train.pending.manifest.jsonl",
        ]:
            if path.exists():
                raise SystemExit(f"output exists, pass --overwrite to replace: {path}")

    counts = {
        "source_seedvc_jobs": write_jsonl(output_root / "source_seedvc_jobs.jsonl", source_jobs),
        "no_text_pending_manifest": write_jsonl(output_root / "no_text.train.pending.manifest.jsonl", pending_no_text),
        "text_pending_manifest": write_jsonl(output_root / "text.train.pending.manifest.jsonl", pending_text),
        "no_text_pending_simple": write_jsonl(output_root / "no_text.train.pending.simple.jsonl", simple_no_text),
        "text_pending_simple": write_jsonl(output_root / "text.train.pending.simple.jsonl", simple_text),
    }
    summary = {
        "stage": "prepare",
        "run_name": args.run_name,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "input_simple_jsonl": str(input_simple),
        "input_seedvc_jobs_jsonl": str(input_jobs),
        "counts": counts,
        "stats": dict(stats),
        "schema": {
            "final_training_triple": "{source_audio=u1_prime, timbre_ref_audio=u2, target_audio=u1}",
            "seedvc_source_generation": "{prosody_ref_audio=u1, timbre_ref_audio=random_other_speaker, output_audio=u1_prime}",
            "no_text_text_field": "<NO_TEXT>",
            "text_text_field": "u1_text",
        },
        "text_constraint_note": (
            "For this Seed-VC-u1-prime route, source_audio u1_prime speaks u1_text, so target_text cannot differ "
            "from source_text without making the manifest untruthful. Therefore text rows are disabled by default. "
            "Use scripts/001034_build_text_prosody_from_mosstts_vcdata.py for the established text_prosody route "
            "where source_text, timbre_ref_text and target_text are deliberately different."
        ),
        "channel_shortcut_risk_note": CHANNEL_RISK_NOTE,
        "recommended_mitigations": CHANNEL_MITIGATIONS,
        "seedvc_runner_hint": {
            "runner": "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction/scripts/run_seedvc_jobs_sharded.sh",
            "jobs_jsonl": str(output_root / "source_seedvc_jobs.jsonl"),
            "results_jsonl": str(output_root / "source_seedvc_results.jsonl"),
        },
    }
    write_json(output_root / "summary.prepare.json", summary)
    write_readme(output_root, summary)
    print(f"[prepare-v2-triples] output_root={output_root}")
    print(f"[prepare-v2-triples] jobs={counts['source_seedvc_jobs']} no_text={counts['no_text_pending_manifest']} text={counts['text_pending_manifest']}")
    return 0


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        job_id = str(row.get("job_id") or "")
        if job_id:
            out[job_id] = row
    return out


def result_accepts(row: dict[str, Any], results: dict[str, dict[str, Any]], args: argparse.Namespace, stats: Counter) -> tuple[bool, dict[str, Any] | None]:
    job_id = str(row.get("source_generation_job_id") or "")
    result = results.get(job_id)
    if args.require_result_ok and result is None:
        stats["skip_missing_seedvc_result"] += 1
        return False, result
    if result is not None and not result.get("ok"):
        stats["skip_bad_seedvc_result"] += 1
        return False, result
    audio = str((result or {}).get("audio") or (result or {}).get("output_audio") or row.get("source_audio") or "")
    if not path_ok(audio, args.min_source_audio_bytes):
        stats["skip_missing_or_small_source_audio"] += 1
        return False, result
    return True, result


def collect_one_mode(args: argparse.Namespace, mode: str, results: dict[str, dict[str, Any]]) -> tuple[int, int, Counter]:
    output_root = Path(args.output_root).expanduser().resolve(strict=False)
    pending_path = Path(args.pending_jsonl).expanduser().resolve(strict=False) if args.pending_jsonl else output_root / f"{mode}.train.pending.manifest.jsonl"
    final_path = Path(args.output_jsonl).expanduser().resolve(strict=False) if args.output_jsonl else output_root / f"{mode}.train.manifest.jsonl"
    simple_path = final_path.with_name(final_path.name.replace(".manifest.jsonl", ".simple.jsonl"))
    stats: Counter = Counter()
    final_rows: list[dict[str, Any]] = []
    simple_rows: list[dict[str, Any]] = []
    if not pending_path.exists():
        raise SystemExit(f"pending manifest not found: {pending_path}")
    for row in iter_jsonl(pending_path):
        stats["pending_rows"] += 1
        ok, result = result_accepts(row, results, args, stats)
        if not ok:
            continue
        row["source_audio_pending"] = False
        row["source_audio_materialized"] = True
        if result is not None:
            row["source_seedvc_result"] = result
            row["meta"]["u1_prime"]["seedvc_result"] = result
        row["meta"]["u1_prime"]["pending"] = False
        final_rows.append(row)
        simple_rows.append(simple_row(row, {"sample_id": row.get("sample_id")}, mode=mode, ready=True))
    if final_path.exists() and not args.overwrite:
        raise SystemExit(f"output exists, pass --overwrite to replace: {final_path}")
    written_final = write_jsonl(final_path, final_rows)
    written_simple = write_jsonl(simple_path, simple_rows)
    stats["written_final"] = written_final
    stats["written_simple"] = written_simple
    return written_final, written_simple, stats


def collect(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).expanduser().resolve(strict=False)
    results_path = Path(args.results_jsonl).expanduser().resolve(strict=False) if args.results_jsonl else output_root / "source_seedvc_results.jsonl"
    results = load_results(results_path)
    mode_counts: dict[str, Any] = {}
    total_stats: dict[str, Any] = {}
    for mode in [item.strip() for item in args.modes.split(",") if item.strip()]:
        if mode not in {"no_text", "text"}:
            raise SystemExit(f"unsupported mode: {mode}")
        written_final, written_simple, stats = collect_one_mode(args, mode, results)
        mode_counts[mode] = {"manifest": written_final, "simple": written_simple}
        total_stats[mode] = dict(stats)
    summary = {
        "stage": "collect",
        "output_root": str(output_root),
        "results_jsonl": str(results_path),
        "results_loaded": len(results),
        "mode_counts": mode_counts,
        "stats": total_stats,
    }
    write_json(output_root / "summary.collect.json", summary)
    print(f"[collect-v2-triples] output_root={output_root}")
    print(json.dumps({"mode_counts": mode_counts, "results_loaded": len(results)}, ensure_ascii=False, sort_keys=True))
    return 0


def write_readme(output_root: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V2 real-target Seed-VC triples",
        "",
        "This directory is prepared from the V2 real-target pilot data.",
        "",
        "Core triple:",
        "",
        "```text",
        "source_audio = u1_prime = Seed-VC(u1 real target -> random other speaker)",
        "timbre_ref_audio = u2 = same-speaker real reference",
        "target_audio = u1 = real target audio",
        "```",
        "",
        "Files:",
        "",
        "- `source_seedvc_jobs.jsonl`: jobs to materialize `u1_prime`.",
        "- `no_text.train.pending.manifest.jsonl`: no-text manifest before `u1_prime` wavs exist.",
        "- `text.train.pending.manifest.jsonl`: text manifest before `u1_prime` wavs exist.",
        "- `*.pending.simple.jsonl`: compact rows for inspecting `u1/u2/u1_prime`, text, label, segs and similarity.",
        "- After `collect`: `no_text.train.manifest.jsonl` and `text.train.manifest.jsonl` contain only rows whose `u1_prime` wav exists.",
        "",
        "Seed-VC runner:",
        "",
        "```bash",
        "PAIR_CONSTRUCTION_ROOT=/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction",
        "SEEDVC_ROUTE_ROOT=/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction_prosody_routes",
        "PY=/inspire/ssd/project/embodied-multimodality/public/yqzhang/miniconda3/envs/contts-train/bin/python \\",
        f"JOBS_JSONL={output_root / 'source_seedvc_jobs.jsonl'} \\",
        f"RESULTS_JSONL={output_root / 'source_seedvc_results.jsonl'} \\",
        "SEED_VC_DIR=$SEEDVC_ROUTE_ROOT/third_party/seed-vc \\",
        "SEEDVC_GPU_IDS=0 \\",
        "bash $PAIR_CONSTRUCTION_ROOT/scripts/run_seedvc_jobs_sharded.sh",
        "```",
        "",
        "Collect after Seed-VC:",
        "",
        "```bash",
        "python dataset_scripts/build_v2_seedvc_real_target_triples.py collect \\",
        f"  --output-root {output_root} \\",
        f"  --results-jsonl {output_root / 'source_seedvc_results.jsonl'} \\",
        "  --overwrite",
        "```",
        "",
        "Channel shortcut risk:",
        "",
        CHANNEL_RISK_NOTE,
        "",
        "Recommended mitigations:",
        "",
    ]
    lines.extend(f"- {item}" for item in CHANNEL_MITIGATIONS)
    lines.extend(["", "Prepare summary:", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), "```", ""])
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V2 real-target Seed-VC source triples from u1/u2 pilot rows.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="Write Seed-VC jobs and pending no-text/text manifests.")
    prep.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    prep.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    prep.add_argument("--input-simple-jsonl", default="")
    prep.add_argument("--input-seedvc-jobs-jsonl", default="")
    prep.add_argument("--run-name", default="v2_real_target_seedvc_triples_pilot_20260707")
    prep.add_argument("--max-rows", type=int, default=0)
    prep.add_argument("--use-input-u1-prime-paths", action=argparse.BooleanOptionalAction, default=False)
    prep.add_argument("--require-existing-u1-u2", action=argparse.BooleanOptionalAction, default=True)
    prep.add_argument("--require-cross-recording-ref", action=argparse.BooleanOptionalAction, default=False)
    prep.add_argument("--require-timbre-text-for-text", action=argparse.BooleanOptionalAction, default=True)
    prep.add_argument("--require-target-text-diff-timbre", action=argparse.BooleanOptionalAction, default=True)
    prep.add_argument("--require-target-text-diff-source", action=argparse.BooleanOptionalAction, default=False)
    prep.add_argument(
        "--emit-v2-u1-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Deprecated/debug only. Emits text rows with source_text==target_text because u1_prime speaks u1_text. "
            "Keep disabled for train data; use the old text_prosody construction for proper text supervision."
        ),
    )
    prep.add_argument("--path-rewrite", action="append", default=[], help="Rewrite path prefix while reading, format OLD=NEW.")
    prep.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    prep.set_defaults(func=prepare)

    coll = sub.add_parser("collect", help="Collect ready rows after Seed-VC has materialized u1_prime wavs.")
    coll.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    coll.add_argument("--results-jsonl", default="")
    coll.add_argument("--modes", default="no_text,text")
    coll.add_argument("--pending-jsonl", default="", help="Optional single pending JSONL; mainly for collecting one mode.")
    coll.add_argument("--output-jsonl", default="", help="Optional single output JSONL; mainly for collecting one mode.")
    coll.add_argument("--min-source-audio-bytes", type=int, default=4096)
    coll.add_argument("--require-result-ok", action=argparse.BooleanOptionalAction, default=False)
    coll.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    coll.set_defaults(func=collect)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build strict Batch-42 paper tables from 004091 per-language summaries.

The paper-facing protocol in this script is intentionally narrow:

* Seed-TTS-Eval-derived VC EN567 internal-320-disjoint
* Seed-TTS-Eval-derived VC ZH1194 internal-320-disjoint
* ZH-hard is N/A because the audited archive has no pure-VC hard manifest

Only a merged summary accompanied by a successful strict audit can contribute
numeric values. Missing systems or languages remain explicitly ``pending``.

Examples:

    python scripts/004092_build_batch42_baseline_tables.py \
      --output-prefix testset/outputs/batch42_baseline_tables_20260711/interim

    python scripts/004092_build_batch42_baseline_tables.py \
      --no-discovery \
      --en-summary freevc_v1=/path/freevc_v1.en.merged.summary.json \
      --zh-summary freevc_v1=/path/freevc_v1.zh.merged.summary.json \
      --output-prefix /tmp/batch42_tables
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "moss_codecvc.batch42_paper_tables.v1"
UNIFIED_SUMMARY_SCHEMA = "moss_codecvc.unified_vc_eval.v1"
STRICT_AUDIT_SCHEMA = "moss_codecvc.batch42_strict_scorer_audit.v1"
PROTOCOL_LABEL = (
    "Seed-TTS-Eval-derived VC EN567/ZH1194 internal-320-disjoint"
)
FINAL_SYSTEM_ID = "path_x_final"
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_SEARCH_ROOT = PROJECT_ROOT / "testset/outputs"


@dataclass(frozen=True)
class DatasetSpec:
    language: str
    test_set_id: str
    paper_label: str
    expected_cases: int
    asr_backend: str
    error_metric: str
    error_label: str


DATASETS: dict[str, DatasetSpec] = {
    "en": DatasetSpec(
        language="en",
        test_set_id="seedtts-vc-en-internal320-disjoint",
        paper_label="Seed-TTS-Eval-derived VC EN567 internal-320-disjoint",
        expected_cases=567,
        asr_backend="whisper_large_v3",
        error_metric="wer",
        error_label="Whisper-large-v3 WER",
    ),
    "zh": DatasetSpec(
        language="zh",
        test_set_id="seedtts-vc-zh-internal320-disjoint",
        paper_label="Seed-TTS-Eval-derived VC ZH1194 internal-320-disjoint",
        expected_cases=1194,
        asr_backend="paraformer_zh",
        error_metric="cer",
        error_label="Paraformer-zh CER",
    ),
}


@dataclass(frozen=True)
class SystemSpec:
    system_id: str
    display_name: str
    system_type: str


# Keep the requested paper-table order. Values remain pending until a strict
# 004091 summary is present; in particular, no paper/template number is copied
# into Ground truth or either Path X row.
DEFAULT_SYSTEMS: tuple[SystemSpec, ...] = (
    SystemSpec(
        "ground_truth",
        "Ground truth (self-eval)",
        "metric calibration, not VC",
    ),
    SystemSpec("seed_vc_v2", "Seed-VC V2", "conditional flow matching"),
    SystemSpec(
        "cosyvoice2_vc", "CosyVoice 2 VC", "speech-token LM + flow matching"
    ),
    SystemSpec(
        "openvoice_v2", "OpenVoice V2", "VITS tone-color converter"
    ),
    SystemSpec(
        "vevo_timbre", "Vevo-Timbre", "content-style tokenizer + flow matching"
    ),
    SystemSpec("freevc_v1", "FreeVC V1", "VITS + WavLM bottleneck"),
    SystemSpec(
        "path_x_3k", "ver2.9.5-probe (ours 3k)", "AR + BNF cross-attn"
    ),
    SystemSpec(
        "path_x_final", "ver2.9.5-final (ours 30k)", "AR + BNF cross-attn"
    ),
)


SPEAKER_SCORERS: tuple[tuple[str, str], ...] = (
    ("wavlm_large_sv", "WavLM-large-SV"),
    ("eres2net", "ERes2Net"),
    ("speechbrain_ecapa", "SpeechBrain ECAPA"),
)


@dataclass(frozen=True)
class ValidatedSummary:
    system_id: str
    language: str
    summary_path: Path
    audit_path: Path
    output_root: Path
    mtime_ns: int
    group_all: Mapping[str, Any]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON {path}: {exc}") from exc


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def _require_counted_metric(
    payload: Mapping[str, Any],
    *,
    expected: int,
    label: str,
) -> float:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    if payload.get("n") != expected:
        raise ValueError(f"{label}.n={payload.get('n')!r}, expected {expected}")
    return _finite_number(payload.get("mean"), label=f"{label}.mean")


def _audit_path_for_summary(summary_path: Path) -> Path:
    suffix = ".summary.json"
    if not summary_path.name.endswith(suffix):
        raise ValueError(f"summary filename must end with {suffix}: {summary_path}")
    return summary_path.with_name(
        summary_path.name[: -len(suffix)] + ".strict_audit.json"
    )


def _infer_identity(
    groups: Mapping[str, Any], summary_path: Path
) -> tuple[str, str]:
    matches: list[tuple[str, str]] = []
    prefix = "system_test_set_language:"
    for language, spec in DATASETS.items():
        suffix = f":{spec.test_set_id}:{language}"
        for key in groups:
            if isinstance(key, str) and key.startswith(prefix) and key.endswith(suffix):
                system_id = key[len(prefix) : -len(suffix)]
                if system_id:
                    matches.append((system_id, language))
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise ValueError(
            f"{summary_path}: expected exactly one strict EN/ZH identity group, "
            f"found {unique}"
        )
    return unique[0]


def _validate_status_counts(
    counts: Any, *, expected: int, label: str
) -> None:
    if not isinstance(counts, Mapping):
        raise ValueError(f"{label} must be an object")
    total = 0
    for key, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label}.{key} is not a non-negative integer")
        total += value
    if counts.get("ok") != expected or total != expected:
        raise ValueError(
            f"{label} must contain exactly ok={expected}; got {dict(counts)}"
        )


def validate_summary(
    summary_path: Path,
    *,
    expected_system: str | None = None,
    expected_language: str | None = None,
) -> ValidatedSummary:
    summary_path = summary_path.expanduser().resolve()
    payload = _load_json(summary_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{summary_path}: summary root must be an object")
    if payload.get("schema_version") != UNIFIED_SUMMARY_SCHEMA:
        raise ValueError(
            f"{summary_path}: schema_version={payload.get('schema_version')!r}, "
            f"expected {UNIFIED_SUMMARY_SCHEMA!r}"
        )
    if payload.get("record_type") != "vc_eval_summary":
        raise ValueError(f"{summary_path}: record_type must be vc_eval_summary")
    groups = payload.get("groups")
    if not isinstance(groups, Mapping) or not isinstance(groups.get("all"), Mapping):
        raise ValueError(f"{summary_path}: groups.all is missing")
    system_id, language = _infer_identity(groups, summary_path)
    if expected_system is not None and system_id != expected_system:
        raise ValueError(
            f"{summary_path}: system={system_id!r}, expected {expected_system!r}"
        )
    if expected_language is not None and language != expected_language:
        raise ValueError(
            f"{summary_path}: language={language!r}, expected {expected_language!r}"
        )

    spec = DATASETS[language]
    identity_key = (
        f"system_test_set_language:{system_id}:{spec.test_set_id}:{language}"
    )
    identity_group = groups.get(identity_key)
    group_all = groups["all"]
    if identity_group != group_all:
        raise ValueError(
            f"{summary_path}: groups.all is not the single strict identity group"
        )
    if group_all.get("n_cases") != spec.expected_cases:
        raise ValueError(
            f"{summary_path}: n_cases={group_all.get('n_cases')!r}, "
            f"expected {spec.expected_cases}"
        )

    speaker = group_all.get("speaker_similarity")
    if not isinstance(speaker, Mapping):
        raise ValueError(f"{summary_path}: speaker_similarity is missing")
    for backend, _display in SPEAKER_SCORERS:
        metric = speaker.get(backend)
        if not isinstance(metric, Mapping):
            raise ValueError(f"{summary_path}: missing speaker scorer {backend}")
        _validate_status_counts(
            metric.get("status_counts"),
            expected=spec.expected_cases,
            label=f"{summary_path}:{backend}.status_counts",
        )
        _require_counted_metric(
            metric.get("sim_ref", {}),
            expected=spec.expected_cases,
            label=f"{summary_path}:{backend}.sim_ref",
        )
        _require_counted_metric(
            metric.get("sim_src", {}),
            expected=spec.expected_cases,
            label=f"{summary_path}:{backend}.sim_src",
        )

    content_asr = group_all.get("content_asr")
    if not isinstance(content_asr, Mapping):
        raise ValueError(f"{summary_path}: content_asr is missing")
    asr = content_asr.get(spec.asr_backend)
    if not isinstance(asr, Mapping):
        raise ValueError(f"{summary_path}: missing ASR backend {spec.asr_backend}")
    _validate_status_counts(
        asr.get("status_counts"),
        expected=spec.expected_cases,
        label=f"{summary_path}:{spec.asr_backend}.status_counts",
    )
    error_mean = _require_counted_metric(
        asr.get(spec.error_metric, {}),
        expected=spec.expected_cases,
        label=f"{summary_path}:{spec.asr_backend}.{spec.error_metric}",
    )
    primary_mean = _require_counted_metric(
        asr.get("primary_error", {}),
        expected=spec.expected_cases,
        label=f"{summary_path}:{spec.asr_backend}.primary_error",
    )
    if not math.isclose(error_mean, primary_mean, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{summary_path}: {spec.error_metric} mean {error_mean} does not "
            f"match primary_error {primary_mean}"
        )

    audit_path = _audit_path_for_summary(summary_path)
    audit = _load_json(audit_path)
    if not isinstance(audit, Mapping):
        raise ValueError(f"{audit_path}: audit root must be an object")
    expected_audit = {
        "schema_version": STRICT_AUDIT_SCHEMA,
        "system_id": system_id,
        "test_set_id": spec.test_set_id,
        "language": language,
        "rows": spec.expected_cases,
        "unique_case_ids": spec.expected_cases,
        "input_index_coverage": [0, spec.expected_cases - 1],
        "all_ok": True,
    }
    for key, expected_value in expected_audit.items():
        if audit.get(key) != expected_value:
            raise ValueError(
                f"{audit_path}: {key}={audit.get(key)!r}, expected {expected_value!r}"
            )
    for backend, _display in SPEAKER_SCORERS:
        _validate_status_counts(
            (audit.get("speaker_status_counts") or {}).get(backend),
            expected=spec.expected_cases,
            label=f"{audit_path}:speaker_status_counts.{backend}",
        )
    _validate_status_counts(
        (audit.get("asr_status_counts") or {}).get(spec.asr_backend),
        expected=spec.expected_cases,
        label=f"{audit_path}:asr_status_counts.{spec.asr_backend}",
    )

    if summary_path.parent.name == "merged" and summary_path.parent.parent.name == language:
        output_root = summary_path.parent.parent.parent
    else:
        output_root = summary_path.parent
    return ValidatedSummary(
        system_id=system_id,
        language=language,
        summary_path=summary_path,
        audit_path=audit_path,
        output_root=output_root,
        mtime_ns=summary_path.stat().st_mtime_ns,
        group_all=group_all,
    )


def discover_summaries(
    search_roots: Sequence[Path],
) -> tuple[dict[str, dict[str, ValidatedSummary]], list[dict[str, str]]]:
    candidates: dict[str, dict[Path, dict[str, ValidatedSummary]]] = {}
    rejected: list[dict[str, str]] = []
    seen_paths: set[Path] = set()
    for root in search_roots:
        root = root.expanduser().resolve()
        if not root.exists():
            rejected.append({"path": str(root), "reason": "search root does not exist"})
            continue
        for path in sorted(root.rglob("*.merged.summary.json")):
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            try:
                item = validate_summary(resolved)
            except ValueError as exc:
                rejected.append({"path": str(resolved), "reason": str(exc)})
                continue
            candidates.setdefault(item.system_id, {}).setdefault(
                item.output_root, {}
            )[item.language] = item

    selected: dict[str, dict[str, ValidatedSummary]] = {}
    for system_id, roots in candidates.items():
        def root_score(entry: tuple[Path, dict[str, ValidatedSummary]]) -> tuple[Any, ...]:
            root, languages = entry
            complete_marker = int((root / "completion.json").is_file())
            newest = max(item.mtime_ns for item in languages.values())
            return (len(languages), complete_marker, newest, str(root))

        _root, languages = max(roots.items(), key=root_score)
        selected[system_id] = dict(languages)
    return selected, rejected


def _parse_binding(raw: str, *, option: str) -> tuple[str, Path]:
    system_id, separator, path_raw = raw.partition("=")
    system_id = system_id.strip()
    path_raw = path_raw.strip()
    if not separator or not system_id or not path_raw:
        raise ValueError(f"{option} must use SYSTEM=PATH, got {raw!r}")
    return system_id, Path(path_raw)


def _parse_system_meta(raw: str) -> SystemSpec:
    parts = [item.strip() for item in raw.split("|")]
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            f"--system-meta must use SYSTEM|DISPLAY_NAME|TYPE, got {raw!r}"
        )
    return SystemSpec(parts[0], parts[1], parts[2])


def _metric_mean(group: Mapping[str, Any], backend: str, metric: str) -> float:
    return float(group["speaker_similarity"][backend][metric]["mean"])


def _language_metrics(item: ValidatedSummary) -> dict[str, Any]:
    spec = DATASETS[item.language]
    group = item.group_all
    asr = group["content_asr"][spec.asr_backend]
    error_fraction = float(asr[spec.error_metric]["mean"])
    return {
        "status": "complete",
        "dataset_id": spec.test_set_id,
        "dataset_label": spec.paper_label,
        "n_cases": spec.expected_cases,
        "summary_path": str(item.summary_path),
        "strict_audit_path": str(item.audit_path),
        "wavlm_large_sv_sim_ref": _metric_mean(
            group, "wavlm_large_sv", "sim_ref"
        ),
        "eres2net_sim_ref": _metric_mean(group, "eres2net", "sim_ref"),
        "speechbrain_ecapa_sim_ref": _metric_mean(
            group, "speechbrain_ecapa", "sim_ref"
        ),
        "primary_asr_backend": spec.asr_backend,
        "primary_error_metric": spec.error_metric,
        "primary_error_fraction": error_fraction,
        "primary_error_percent": error_fraction * 100.0,
    }


def _pending_language(language: str) -> dict[str, Any]:
    spec = DATASETS[language]
    return {
        "status": "pending",
        "dataset_id": spec.test_set_id,
        "dataset_label": spec.paper_label,
        "n_cases": None,
        "summary_path": None,
        "strict_audit_path": None,
        "wavlm_large_sv_sim_ref": None,
        "eres2net_sim_ref": None,
        "speechbrain_ecapa_sim_ref": None,
        "primary_asr_backend": spec.asr_backend,
        "primary_error_metric": spec.error_metric,
        "primary_error_fraction": None,
        "primary_error_percent": None,
    }


def build_payload(
    *,
    selected: Mapping[str, Mapping[str, ValidatedSummary]],
    system_specs: Sequence[SystemSpec],
    search_roots: Sequence[Path],
    explicit_summaries: Mapping[str, Mapping[str, Path]],
    rejected_candidates: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for system in system_specs:
        available = selected.get(system.system_id, {})
        metrics = {
            language: (
                _language_metrics(available[language])
                if language in available
                else _pending_language(language)
            )
            for language in ("en", "zh")
        }
        complete_languages = sum(
            item["status"] == "complete" for item in metrics.values()
        )
        status = (
            "complete"
            if complete_languages == 2
            else "partial"
            if complete_languages == 1
            else "pending"
        )
        rows.append(
            {
                "system_id": system.system_id,
                "display_name": system.display_name,
                "type": system.system_type,
                "status": status,
                "metrics": metrics,
                "zh_hard": {
                    "status": "not_applicable",
                    "sim_ref": None,
                    "cer_fraction": None,
                    "cer_percent": None,
                    "reason": "No pure-VC ZH-hard manifest exists in the audited Seed-TTS-Eval-derived inputs.",
                },
            }
        )

    main_table = []
    cross_validation = []
    for row in rows:
        en = row["metrics"]["en"]
        zh = row["metrics"]["zh"]
        main_table.append(
            {
                "system_id": row["system_id"],
                "system": row["display_name"],
                "type": row["type"],
                "status": row["status"],
                "zh1194_wavlm_sim_ref": zh["wavlm_large_sv_sim_ref"],
                "zh1194_paraformer_cer_fraction": zh["primary_error_fraction"],
                "zh1194_paraformer_cer_percent": zh["primary_error_percent"],
                "en567_wavlm_sim_ref": en["wavlm_large_sv_sim_ref"],
                "en567_whisper_wer_fraction": en["primary_error_fraction"],
                "en567_whisper_wer_percent": en["primary_error_percent"],
                "zh_hard_wavlm_sim_ref": None,
                "zh_hard_cer_fraction": None,
                "zh_hard_cer_percent": None,
                "zh_hard_status": "N/A",
            }
        )
        for language in ("en", "zh"):
            item = row["metrics"][language]
            cross_validation.append(
                {
                    "system_id": row["system_id"],
                    "system": row["display_name"],
                    "split": "ZH1194" if language == "zh" else "EN567",
                    "dataset_label": DATASETS[language].paper_label,
                    "status": item["status"],
                    "n_cases": item["n_cases"],
                    "wavlm_large_sv_sim_ref": item["wavlm_large_sv_sim_ref"],
                    "eres2net_sim_ref": item["eres2net_sim_ref"],
                    "speechbrain_ecapa_sim_ref": item[
                        "speechbrain_ecapa_sim_ref"
                    ],
                }
            )

    complete_systems = sum(row["status"] == "complete" for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if complete_systems == len(rows) else "interim",
        "protocol": {
            "label": PROTOCOL_LABEL,
            "ground_truth_source_self_eval": {
                "display_name": "Ground truth (self-eval)",
                "calibration_only": True,
                "generated_audio": "field-5 source/content waveform",
                "reference_audio": "same field-5 waveform",
                "source_audio": "same field-5 waveform",
                "reference_text": "field-4 source transcript",
                "paired_target_speaker_ground_truth_available": False,
                "warning": (
                    "Same-file SIM is a scorer calibration, not target-speaker "
                    "VC performance; ASR is the raw-source recognition ceiling."
                ),
            },
            "en": {
                "label": DATASETS["en"].paper_label,
                "test_set_id": DATASETS["en"].test_set_id,
                "expected_cases": DATASETS["en"].expected_cases,
            },
            "zh": {
                "label": DATASETS["zh"].paper_label,
                "test_set_id": DATASETS["zh"].test_set_id,
                "expected_cases": DATASETS["zh"].expected_cases,
            },
            "zh_hard": {
                "status": "not_applicable",
                "reason": "No pure-VC ZH-hard manifest exists in the audited Seed-TTS-Eval-derived inputs.",
            },
            "error_units": {
                "json_fraction": "raw error ratio from 004091; insertions may make it exceed 1",
                "paper_table": "percent",
            },
        },
        "discovery": {
            "search_roots": [str(path.expanduser().resolve()) for path in search_roots],
            "explicit_summaries": {
                system_id: {language: str(path.expanduser().resolve()) for language, path in langs.items()}
                for system_id, langs in explicit_summaries.items()
            },
            "rejected_candidates": list(rejected_candidates),
        },
        "counts": {
            "systems": len(rows),
            "complete": complete_systems,
            "partial": sum(row["status"] == "partial" for row in rows),
            "pending": sum(row["status"] == "pending" for row in rows),
        },
        "systems": rows,
        "main_table": main_table,
        "cross_validation_table": cross_validation,
    }


def _format_sim(value: Any) -> str:
    return "pending" if value is None else f"{float(value):.4f}"


def _format_percent(value: Any) -> str:
    return "pending" if value is None else f"{float(value):.2f}"


def render_markdown(payload: Mapping[str, Any]) -> str:
    counts = payload["counts"]
    lines = [
        "# Batch-42 baseline VC unified evaluation tables",
        "",
        f"- Protocol: **{PROTOCOL_LABEL}**.",
        "- EN uses WavLM-large-SV SIM(ref) and Whisper-large-v3 WER; ZH uses WavLM-large-SV SIM(ref) and Paraformer-zh CER.",
        "- WER/CER are displayed as percentages; JSON also preserves raw fractions.",
        "- ZH-hard: **N/A**, because the audited archive has no pure-VC ZH-hard manifest.",
        "- **Ground truth (self-eval)** maps generated/reference/source to the "
        "same raw field-5 source WAV. Its SIM is only a same-file scorer calibration "
        "and must not be ranked as target-speaker VC performance; its ASR error is a "
        "raw-source ceiling. The non-parallel VC manifests contain no paired waveform "
        "of the target prompt speaker reading the source transcript.",
        f"- Status: `{payload['status']}`; complete systems {counts['complete']}/{counts['systems']}. Pending cells contain no substituted paper/template values.",
        "",
        "## Paper main table",
        "",
        "| System | Type | EN567 WavLM SIM ↑ | EN567 Whisper WER (%) ↓ | ZH1194 WavLM SIM ↑ | ZH1194 Paraformer CER (%) ↓ | ZH-hard SIM | ZH-hard CER | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["main_table"]:
        lines.append(
            "| {system} | {type} | {en_sim} | {en_wer} | {zh_sim} | "
            "{zh_cer} | N/A | N/A | {status} |".format(
                system=row["system"],
                type=row["type"],
                zh_sim=_format_sim(row["zh1194_wavlm_sim_ref"]),
                zh_cer=_format_percent(row["zh1194_paraformer_cer_percent"]),
                en_sim=_format_sim(row["en567_wavlm_sim_ref"]),
                en_wer=_format_percent(row["en567_whisper_wer_percent"]),
                status=row["status"],
            )
        )

    lines.extend(
        [
            "",
            "## SIM scorer cross-validation",
            "",
            "The language splits remain separate; no synthetic EN/ZH aggregate is reported.",
            "",
            "| System | Split | n | WavLM-large-SV SIM(ref) | ERes2Net SIM(ref) | SpeechBrain ECAPA SIM(ref) | Status |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["cross_validation_table"]:
        n_cases = "pending" if row["n_cases"] is None else str(row["n_cases"])
        lines.append(
            "| {system} | {split} | {n} | {wavlm} | {eres} | {speechbrain} | {status} |".format(
                system=row["system"],
                split=row["split"],
                n=n_cases,
                wavlm=_format_sim(row["wavlm_large_sv_sim_ref"]),
                eres=_format_sim(row["eres2net_sim_ref"]),
                speechbrain=_format_sim(row["speechbrain_ecapa_sim_ref"]),
                status=row["status"],
            )
        )

    lines.extend(["", "## Provenance", ""])
    for row in payload["systems"]:
        paths = [
            item["summary_path"]
            for item in row["metrics"].values()
            if item["summary_path"] is not None
        ]
        if paths:
            lines.append(f"- {row['display_name']}:")
            lines.extend(f"  - `{path}`" for path in paths)
    if not any(
        item["summary_path"]
        for row in payload["systems"]
        for item in row["metrics"].values()
    ):
        lines.append("- No complete strict summaries discovered yet.")
    lines.append("")
    return "\n".join(lines)


def _tsv_text(rows: Iterable[Sequence[Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerows(rows)
    return buffer.getvalue()


def render_main_tsv(payload: Mapping[str, Any]) -> str:
    rows: list[list[Any]] = [[
        "system_id",
        "system",
        "type",
        "en567_wavlm_sim_ref",
        "en567_whisper_wer_percent",
        "zh1194_wavlm_sim_ref",
        "zh1194_paraformer_cer_percent",
        "zh_hard_wavlm_sim_ref",
        "zh_hard_cer_percent",
        "status",
    ]]
    for row in payload["main_table"]:
        rows.append(
            [
                row["system_id"],
                row["system"],
                row["type"],
                "pending" if row["en567_wavlm_sim_ref"] is None else row["en567_wavlm_sim_ref"],
                "pending" if row["en567_whisper_wer_percent"] is None else row["en567_whisper_wer_percent"],
                "pending" if row["zh1194_wavlm_sim_ref"] is None else row["zh1194_wavlm_sim_ref"],
                "pending" if row["zh1194_paraformer_cer_percent"] is None else row["zh1194_paraformer_cer_percent"],
                "N/A",
                "N/A",
                row["status"],
            ]
        )
    return _tsv_text(rows)


def render_cross_validation_tsv(payload: Mapping[str, Any]) -> str:
    rows: list[list[Any]] = [[
        "system_id",
        "system",
        "split",
        "dataset_label",
        "n_cases",
        "wavlm_large_sv_sim_ref",
        "eres2net_sim_ref",
        "speechbrain_ecapa_sim_ref",
        "status",
    ]]
    for row in payload["cross_validation_table"]:
        rows.append(
            [
                row["system_id"],
                row["system"],
                row["split"],
                row["dataset_label"],
                "pending" if row["n_cases"] is None else row["n_cases"],
                "pending" if row["wavlm_large_sv_sim_ref"] is None else row["wavlm_large_sv_sim_ref"],
                "pending" if row["eres2net_sim_ref"] is None else row["eres2net_sim_ref"],
                "pending" if row["speechbrain_ecapa_sim_ref"] is None else row["speechbrain_ecapa_sim_ref"],
                row["status"],
            ]
        )
    return _tsv_text(rows)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_outputs(prefix: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    prefix = prefix.expanduser().resolve()
    paths = {
        "markdown": prefix.with_suffix(".md"),
        "json": prefix.with_suffix(".json"),
        "main_tsv": prefix.with_suffix(".tsv"),
        "cross_validation_tsv": prefix.with_name(
            prefix.name + ".cross_validation.tsv"
        ),
    }
    _atomic_write(paths["markdown"], render_markdown(payload))
    _atomic_write(
        paths["json"],
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(paths["main_tsv"], render_main_tsv(payload))
    _atomic_write(
        paths["cross_validation_tsv"], render_cross_validation_tsv(payload)
    )
    return {key: str(value) for key, value in paths.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        type=Path,
        help=(
            "root recursively searched for *.merged.summary.json; repeatable "
            f"(default: {DEFAULT_SEARCH_ROOT})"
        ),
    )
    parser.add_argument(
        "--no-discovery",
        action="store_true",
        help="disable recursive discovery and use only explicit summaries",
    )
    parser.add_argument(
        "--allow-path-x-final",
        action="store_true",
        help=(
            "allow explicit path_x_final EN/ZH summaries after the separate "
            "Batch-43 final-selection/inference/scorer gates have passed; "
            "path_x_final is never accepted through recursive discovery"
        ),
    )
    parser.add_argument(
        "--en-summary",
        action="append",
        default=[],
        metavar="SYSTEM=PATH",
        help="explicit strict EN merged summary; repeatable",
    )
    parser.add_argument(
        "--zh-summary",
        action="append",
        default=[],
        metavar="SYSTEM=PATH",
        help="explicit strict ZH merged summary; repeatable",
    )
    parser.add_argument(
        "--expected-system",
        action="append",
        default=[],
        metavar="SYSTEM",
        help="replace the default paper-system list; repeatable",
    )
    parser.add_argument(
        "--system-meta",
        action="append",
        default=[],
        metavar="SYSTEM|DISPLAY_NAME|TYPE",
        help="override/add display metadata; repeatable",
    )
    parser.add_argument("--output-prefix", required=True, type=Path)
    return parser


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str]]:
    search_roots = list(args.search_root)
    if not args.no_discovery and not search_roots:
        search_roots = [DEFAULT_SEARCH_ROOT]
    if args.no_discovery:
        search_roots = []

    selected: dict[str, dict[str, ValidatedSummary]] = {}
    rejected: list[dict[str, str]] = []
    if search_roots:
        selected, rejected = discover_summaries(search_roots)
        discovered_final = selected.pop(FINAL_SYSTEM_ID, {})
        for item in discovered_final.values():
            rejected.append(
                {
                    "path": str(item.summary_path),
                    "reason": (
                        "path_x_final cannot be promoted by recursive discovery; "
                        "use explicit summaries through the gated final publisher"
                    ),
                }
            )

    explicit: dict[str, dict[str, Path]] = {}
    for language, values in (("en", args.en_summary), ("zh", args.zh_summary)):
        for raw in values:
            system_id, path = _parse_binding(raw, option=f"--{language}-summary")
            if system_id == FINAL_SYSTEM_ID and not args.allow_path_x_final:
                raise ValueError(
                    "explicit path_x_final summaries require --allow-path-x-final "
                    "after the separate final-selection/inference/scorer gates pass"
                )
            if language in explicit.setdefault(system_id, {}):
                raise ValueError(
                    f"duplicate explicit {language} summary for {system_id}"
                )
            item = validate_summary(
                path, expected_system=system_id, expected_language=language
            )
            explicit[system_id][language] = path
            selected.setdefault(system_id, {})[language] = item

    default_meta = {item.system_id: item for item in DEFAULT_SYSTEMS}
    for raw in args.system_meta:
        item = _parse_system_meta(raw)
        default_meta[item.system_id] = item
    if args.expected_system:
        ordered_ids = list(dict.fromkeys(args.expected_system))
    else:
        ordered_ids = [item.system_id for item in DEFAULT_SYSTEMS]
    for system_id in sorted(selected):
        if system_id not in ordered_ids:
            ordered_ids.append(system_id)
    system_specs = [
        default_meta.get(system_id, SystemSpec(system_id, system_id, "unspecified"))
        for system_id in ordered_ids
    ]

    payload = build_payload(
        selected=selected,
        system_specs=system_specs,
        search_roots=search_roots,
        explicit_summaries=explicit,
        rejected_candidates=rejected,
    )
    outputs = write_outputs(args.output_prefix, payload)
    return payload, outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload, outputs = run(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "protocol": payload["protocol"]["label"],
                "counts": payload["counts"],
                "outputs": outputs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Build the Batch-34/36/37 seven-arm final decision report.

The report deliberately remains useful while the two official SeedTTS-320
families are still running.  Missing inputs and missing metrics are emitted as
explicit placeholders; this script never substitutes remembered numbers.

The two speaker-similarity backends have intentionally verbose canonical
names because both contain an ECAPA component but are not interchangeable:

* ``seedtts_wavlm_ecapa_*``: SeedTTSEval WavLM-Large + ECAPA-TDNN score.
* ``speechbrain_ecapa_*``: the independent SpeechBrain ECAPA score.

Typical use (all inputs are auto-discovered when they exist)::

    python scripts/004075_build_ver23_batch343637_decision_report.py \
      --output-prefix docs/assets/ver23_batch343637_final_decision_20260711

Explicit inputs can be JSON or TSV official matrices::

    python scripts/004075_build_ver23_batch343637_decision_report.py \
      --batch3436-matrix /path/batch3436_step3000.official_matrix.json \
      --batch37-matrix /path/batch37_step3000.official_matrix.tsv \
      --batch33-baseline /path/batch33_matrix.json \
      --subjective /path/listening_votes.tsv \
      --output-prefix /path/report

The subjective TSV/JSON accepts ``config_key`` or ``arm`` plus
``subjective_vs_batch33``.  Canonical values are ``clearly_better``,
``slightly_better``, ``same``, ``worse``, and ``pending``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]

SCOPES = ("no_text", "text", "all")
FAMILY_CONFIGS = {
    "batch3436": ("B1", "B2", "A1", "B3", "A2"),
    "batch37": ("C1", "C2L10", "C2L12", "C2L14", "C2L16"),
}
ARM_ORDER = ("B1", "B2", "A1", "B3", "A2", "C1", "C2")
CONFIG_ORDER = ("B1", "B2", "A1", "B3", "A2", "C1", "C2L10", "C2L12", "C2L14", "C2L16")

ARM_META = {
    "B1": {
        "title": "BNF last_16",
        "change": "CONTENT_CROSS_ATTN_LAYERS=last_16",
        "hypothesis": "A: all-layer BNF attention dilutes C_ref",
    },
    "B2": {
        "title": "text repeat 1",
        "change": "text repeat 10 -> 1",
        "hypothesis": "B: text-bypass training-mix mismatch",
    },
    "A1": {
        "title": "stronger decouple",
        "change": "phoneme 0.05; guided 0.10",
        "hypothesis": "reverse control for C",
    },
    "B3": {
        "title": "weaker decouple",
        "change": "phoneme 0.01; guided 0.02",
        "hypothesis": "C: auxiliary attention constraints are too strong",
    },
    "A2": {
        "title": "content CTC",
        "change": "CONTENT_CTC_WEIGHT=0.10",
        "hypothesis": "stronger BNF content supervision",
    },
    "C1": {
        "title": "compact content path",
        "change": "512d / five injection layers / independent LR / LoRA warmup",
        "hypothesis": "content path is oversized and under-trained",
    },
    "C2": {
        "title": "true ref-audio CFG",
        "change": "C_ref dropout 0.15; lambda scan 1.0/1.2/1.4/1.6",
        "hypothesis": "inference guidance can amplify reference acoustics",
    },
}

METRIC_FIELDS = (
    "n",
    "keep",
    "fail_count",
    "fail_rate",
    "cer",
    "primary_error",
    "seedtts_wavlm_ecapa_sim_ref",
    "seedtts_wavlm_ecapa_sim_src",
    "seedtts_wavlm_ecapa_ref_bound",
    "speechbrain_ecapa_sim_ref",
    "speechbrain_ecapa_sim_src",
    "speechbrain_ecapa_ref_bound",
    "ref_content_lcs_f1",
)

DECISION_REQUIRED_FIELDS = (
    "fail_rate",
    "cer",
    "seedtts_wavlm_ecapa_sim_ref",
    "seedtts_wavlm_ecapa_sim_src",
    "seedtts_wavlm_ecapa_ref_bound",
    "speechbrain_ecapa_sim_ref",
    "speechbrain_ecapa_sim_src",
    "speechbrain_ecapa_ref_bound",
    "ref_content_lcs_f1",
)

OUTPUT_TSV_FIELDS = (
    "family",
    "arm",
    "arm_title",
    "config_key",
    "ref_audio_cfg_scale",
    "scope",
    "is_baseline",
    *METRIC_FIELDS,
    "speechbrain_ref_ge_0p48",
    "seedtts_src_le_0p40",
    "objective_joint_pass",
    "case1_objective_pass",
    "subjective_vs_batch33",
    "subjective_notes",
    "red_flags",
    "warnings",
    "missing_fields",
    "data_status",
    "c2_objective_selected",
    "source_path",
)


def finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        output = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def finite_int(value: Any) -> int | None:
    number = finite_float(value)
    if number is None:
        return None
    return int(round(number))


def bool_value(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on", "pass", "keep"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "fail", "drop"}:
        return False
    return None


def mean(values: Iterable[float | None]) -> float | None:
    usable = [value for value in values if value is not None and math.isfinite(value)]
    return sum(usable) / len(usable) if usable else None


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def unique_strings(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = clean_text(value)
        if text and text not in seen:
            output.append(text)
            seen.add(text)
    return output


def as_posix(path: Path | None) -> str:
    return str(path.resolve()) if path is not None else ""


def config_arm(config_key: str) -> str:
    key = clean_text(config_key).upper().replace("-", "").replace("_", "")
    if key in {"BATCH33", "BASELINE", "VER295", "VER2.9.5"}:
        return "Batch-33"
    if key.startswith("C2") or "TRUEREFAUDIOCFG" in key or "REFCFG" in key:
        return "C2"
    for arm in ARM_ORDER:
        if key == arm or key.startswith(arm) or f"BATCH3436{arm}" in key or f"BATCH37{arm}" in key:
            return arm
    return clean_text(config_key) or "unknown"


def config_scale(config_key: str, value: Any = None) -> float | None:
    explicit = finite_float(value)
    if explicit is not None:
        return explicit
    normalized = clean_text(config_key).upper().replace(".", "P")
    match = re.search(r"C2L(10|12|14|16)\b", normalized)
    if match:
        return int(match.group(1)) / 10.0
    match = re.search(r"CFG(?:SCALE)?[_-]?(1P[0246])\b", normalized)
    if match:
        return float(match.group(1).replace("P", "."))
    return 1.0 if config_arm(config_key) != "Batch-33" else None


def canonical_config(config_key: str, scale: float | None = None) -> str:
    arm = config_arm(config_key)
    if arm == "C2":
        actual = scale if scale is not None else config_scale(config_key)
        if actual is not None:
            return f"C2L{int(round(actual * 10)):02d}"
    if arm in ARM_ORDER:
        return arm
    if arm == "Batch-33":
        return "Batch-33"
    return clean_text(config_key) or "unknown"


def row_alias(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row.get(name) not in (None, ""):
            return row.get(name)
    return None


def normalize_metric_row(
    raw: Mapping[str, Any],
    *,
    family_hint: str = "",
    config_hint: str = "",
    source_path: Path | None = None,
    is_baseline: bool = False,
) -> dict[str, Any]:
    raw_key = clean_text(row_alias(raw, "config_key", "config", "arm", "label", "name")) or config_hint
    scale = config_scale(raw_key, row_alias(raw, "ref_audio_cfg_scale", "cfg_scale", "lambda", "scale"))
    config_key = "Batch-33" if is_baseline else canonical_config(raw_key, scale)
    arm = "Batch-33" if is_baseline else config_arm(config_key)
    scope = clean_text(row_alias(raw, "scope", "mode")) or "all"
    if scope not in SCOPES:
        scope = scope.lower().replace("-", "_")

    n = finite_int(row_alias(raw, "n", "count", "num_samples"))
    keep = finite_int(row_alias(raw, "keep", "keep_count"))
    fail_count = finite_int(row_alias(raw, "fail_count", "failed", "num_failed"))
    fail_rate = finite_float(row_alias(raw, "fail_rate", "failure_rate", "fail"))
    if fail_rate is not None and fail_rate > 1.0 and fail_rate <= 100.0:
        fail_rate /= 100.0
    if fail_count is None and n is not None and keep is not None:
        fail_count = n - keep
    if fail_rate is None and fail_count is not None and n:
        fail_rate = fail_count / n
    if fail_count is None and fail_rate is not None and n is not None:
        fail_count = int(round(fail_rate * n))
    if keep is None and fail_count is not None and n is not None:
        keep = n - fail_count

    normalized: dict[str, Any] = {
        "family": clean_text(row_alias(raw, "family")) or family_hint,
        "arm": arm,
        "arm_title": "Batch-33 baseline" if is_baseline else ARM_META.get(arm, {}).get("title", arm),
        "config_key": config_key,
        "ref_audio_cfg_scale": scale,
        "scope": scope,
        "is_baseline": bool(is_baseline),
        "n": n,
        "keep": keep,
        "fail_count": fail_count,
        "fail_rate": fail_rate,
        "cer": finite_float(row_alias(raw, "cer")),
        "primary_error": finite_float(row_alias(raw, "primary_error", "wer_or_cer")),
        "seedtts_wavlm_ecapa_sim_ref": finite_float(
            row_alias(raw, "seedtts_wavlm_ecapa_sim_ref", "seedtts_sim_ref", "wavlm_sim_ref", "sim_ref")
        ),
        "seedtts_wavlm_ecapa_sim_src": finite_float(
            row_alias(raw, "seedtts_wavlm_ecapa_sim_src", "seedtts_sim_src", "wavlm_sim_src", "sim_src")
        ),
        "seedtts_wavlm_ecapa_ref_bound": finite_float(
            row_alias(raw, "seedtts_wavlm_ecapa_ref_bound", "seedtts_ref_bound", "wavlm_ref_bound", "ref_bound")
        ),
        "speechbrain_ecapa_sim_ref": finite_float(
            row_alias(raw, "speechbrain_ecapa_sim_ref", "speechbrain_sim_ref", "ecapa_sim_ref")
        ),
        "speechbrain_ecapa_sim_src": finite_float(
            row_alias(raw, "speechbrain_ecapa_sim_src", "speechbrain_sim_src", "ecapa_sim_src")
        ),
        "speechbrain_ecapa_ref_bound": finite_float(
            row_alias(raw, "speechbrain_ecapa_ref_bound", "speechbrain_ref_bound", "ecapa_ref_bound")
        ),
        "ref_content_lcs_f1": finite_float(row_alias(raw, "ref_content_lcs_f1", "ref_content_f1", "f1")),
        "subjective_vs_batch33": "pending",
        "subjective_notes": "",
        "red_flags": [],
        "warnings": [],
        "missing_fields": [],
        "data_status": "loaded",
        "c2_objective_selected": False,
        "source_path": as_posix(source_path),
    }
    return normalized


def placeholder_row(family: str, config_key: str, scope: str, *, baseline: bool = False) -> dict[str, Any]:
    row = normalize_metric_row(
        {"config_key": config_key, "scope": scope},
        family_hint=family,
        config_hint=config_key,
        is_baseline=baseline,
    )
    row["data_status"] = "missing"
    row["missing_fields"] = list(METRIC_FIELDS)
    return row


def read_table(path: Path) -> list[dict[str, Any]]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]


def extract_json_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    if isinstance(payload.get("rows"), list):
        return [dict(row) for row in payload["rows"] if isinstance(row, Mapping)]
    if isinstance(payload.get("matrix"), list):
        return [dict(row) for row in payload["matrix"] if isinstance(row, Mapping)]
    output: list[dict[str, Any]] = []
    for scope in SCOPES:
        value = payload.get(scope)
        if isinstance(value, Mapping):
            output.append({"scope": scope, **dict(value)})
    if output:
        return output
    if any(key in payload for key in ("scope", "mode", "cer", "fail_rate", "sim_ref")):
        return [dict(payload)]
    return []


def read_matrix_file(path: Path, family: str, *, baseline: bool = False) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    if not path.exists():
        return [], [f"missing file: {path}"]
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_rows = extract_json_rows(payload)
        elif path.suffix.lower() in {".tsv", ".tab", ".csv"}:
            raw_rows = read_table(path)
        else:
            return [], [f"unsupported matrix extension: {path}"]
    except Exception as exc:  # report malformed external results without hiding the rest
        return [], [f"failed to parse {path}: {type(exc).__name__}: {exc}"]
    if not raw_rows:
        errors.append(f"no metric rows in {path}")
    rows = [
        normalize_metric_row(
            row,
            family_hint=family,
            source_path=path,
            is_baseline=baseline,
        )
        for row in raw_rows
    ]
    return rows, errors


def normalize_text_for_lcs(text: Any) -> str:
    chars: list[str] = []
    for char in clean_text(text).lower():
        code = ord(char)
        if char.isalnum() or 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF:
            chars.append(char)
    return "".join(chars)


def lcs_len(first: str, second: str) -> int:
    if not first or not second:
        return 0
    short, long = (first, second) if len(first) <= len(second) else (second, first)
    previous = [0] * (len(short) + 1)
    for char in long:
        current = [0]
        for index, other in enumerate(short, start=1):
            current.append(previous[index - 1] + 1 if char == other else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def ref_content_f1(raw: Mapping[str, Any]) -> float:
    generated = normalize_text_for_lcs(raw.get("asr_tgt_text"))
    reference = normalize_text_for_lcs(raw.get("timbre_ref_text"))
    hit = lcs_len(generated, reference)
    precision = hit / max(1, len(generated))
    recall = hit / max(1, len(reference))
    return 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)


def primary_error(raw: Mapping[str, Any]) -> float | None:
    cer = finite_float(raw.get("cer_tgt"))
    wer = finite_float(raw.get("wer_tgt"))
    source_lang = clean_text(row_alias(raw, "source_lang", "language")).lower()
    if source_lang.startswith("zh"):
        return cer if cer is not None else wer
    if source_lang.startswith("en"):
        return wer if wer is not None else cer
    if cer is not None and wer is not None:
        return min(cer, wer)
    return cer if cer is not None else wer


def read_dual_encoder_baseline_dir(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load the Batch-33 baseline from the same-case dual-encoder evaluation."""

    errors: list[str] = []
    cases_path = path / "dual_encoder_cases.csv"
    asr_path = path / "run_views" / "Batch33" / "Batch33.asr_eval.jsonl"
    try:
        all_case_rows = read_table(cases_path)
    except Exception as exc:
        return [], [f"failed to parse enriched baseline cases {cases_path}: {exc}"]
    case_rows = [row for row in all_case_rows if clean_text(row.get("run")) == "Batch33"]
    if not case_rows:
        return [], [f"enriched baseline cases contain no Batch33 rows: {cases_path}"]
    if not asr_path.exists():
        return [], [f"enriched baseline ASR rows missing: {asr_path}"]
    try:
        asr_rows = [
            json.loads(line)
            for line in asr_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception as exc:
        return [], [f"failed to parse enriched baseline ASR rows {asr_path}: {exc}"]

    case_ids = {clean_text(row.get("case_id")) for row in case_rows}
    asr_ids = {clean_text(row.get("case_id")) for row in asr_rows}
    if case_ids != asr_ids:
        errors.append(
            "enriched Batch-33 case/ASR IDs differ: "
            f"speaker_only={len(case_ids - asr_ids)} asr_only={len(asr_ids - case_ids)}"
        )

    rows: list[dict[str, Any]] = []
    for scope in SCOPES:
        scoped_cases = case_rows if scope == "all" else [row for row in case_rows if row.get("mode") == scope]
        scoped_asr = asr_rows if scope == "all" else [row for row in asr_rows if row.get("mode") == scope]
        if len(scoped_cases) != len(scoped_asr):
            errors.append(
                f"enriched Batch-33 scope={scope} count mismatch: "
                f"speaker={len(scoped_cases)} asr={len(scoped_asr)}"
            )

        keep_values = [bool_value(row.get("content_keep")) for row in scoped_asr]
        if any(value is None for value in keep_values):
            errors.append(f"enriched Batch-33 scope={scope} has missing content_keep values")
        keep = sum(value is True for value in keep_values)

        seed_ref = [finite_float(row.get("sim_gen_ref")) for row in scoped_cases]
        seed_src = [finite_float(row.get("sim_gen_source")) for row in scoped_cases]
        speechbrain_ref = [finite_float(row.get("ecapa_sim_gen_ref")) for row in scoped_cases]
        speechbrain_src = [finite_float(row.get("ecapa_sim_gen_source")) for row in scoped_cases]

        def bound_rate(ref_values: Sequence[float | None], src_values: Sequence[float | None]) -> float | None:
            pairs = [
                (ref, src)
                for ref, src in zip(ref_values, src_values)
                if ref is not None and src is not None
            ]
            return sum((ref - src) > 0.05 for ref, src in pairs) / len(pairs) if pairs else None

        n = len(scoped_asr)
        row = normalize_metric_row(
            {
                "config_key": "Batch-33",
                "scope": scope,
                "n": n,
                "keep": keep,
                "fail_count": n - keep,
                "cer": mean(finite_float(raw.get("cer_tgt")) for raw in scoped_asr),
                "primary_error": mean(primary_error(raw) for raw in scoped_asr),
                "seedtts_wavlm_ecapa_sim_ref": mean(seed_ref),
                "seedtts_wavlm_ecapa_sim_src": mean(seed_src),
                "seedtts_wavlm_ecapa_ref_bound": bound_rate(seed_ref, seed_src),
                "speechbrain_ecapa_sim_ref": mean(speechbrain_ref),
                "speechbrain_ecapa_sim_src": mean(speechbrain_src),
                "speechbrain_ecapa_ref_bound": bound_rate(speechbrain_ref, speechbrain_src),
                "ref_content_lcs_f1": mean(ref_content_f1(raw) for raw in scoped_asr),
            },
            family_hint="baseline",
            source_path=path,
            is_baseline=True,
        )
        rows.append(row)
    return rows, errors


def find_single(path: Path, pattern: str, *, reject: Sequence[str] = ()) -> Path | None:
    candidates = [candidate for candidate in path.glob(pattern) if not any(token in candidate.name for token in reject)]
    return max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None


def read_legacy_baseline_dir(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    summary_path = find_single(path, "*.summary.json", reject=("shard", "speaker_sim", "ref_content"))
    speaker_path = find_single(path, "*.speaker_sim.csv")
    asr_path = find_single(path, "*.asr_eval.jsonl", reject=("shard",))
    if summary_path is None:
        return [], [f"baseline directory lacks run summary: {path}"]
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], [f"failed to parse baseline summary {summary_path}: {exc}"]

    speaker_rows: list[dict[str, Any]] = []
    if speaker_path is not None:
        speaker_rows = read_table(speaker_path)
    else:
        errors.append(f"baseline SeedTTS speaker rows missing under {path}")
    asr_rows: list[dict[str, Any]] = []
    if asr_path is not None:
        try:
            asr_rows = [json.loads(line) for line in asr_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except Exception as exc:
            errors.append(f"failed to parse baseline ASR rows {asr_path}: {exc}")
    else:
        errors.append(f"baseline ASR rows missing under {path}")

    rows: list[dict[str, Any]] = []
    for scope in SCOPES:
        official = summary.get("overall") if scope == "all" else (summary.get("by_mode") or {}).get(scope)
        if not isinstance(official, Mapping):
            errors.append(f"baseline summary missing scope={scope}: {summary_path}")
            continue
        scoped_speaker = speaker_rows if scope == "all" else [row for row in speaker_rows if row.get("mode") == scope]
        scoped_asr = asr_rows if scope == "all" else [row for row in asr_rows if row.get("mode") == scope]
        sim_ref = [finite_float(row.get("sim_gen_ref")) for row in scoped_speaker]
        sim_src = [finite_float(row.get("sim_gen_source")) for row in scoped_speaker]
        valid_pairs = [(ref, src) for ref, src in zip(sim_ref, sim_src) if ref is not None and src is not None]
        ref_bound = (
            sum((ref - src) > 0.05 for ref, src in valid_pairs) / len(valid_pairs)
            if valid_pairs
            else None
        )
        n = finite_int(official.get("n"))
        keep = finite_int(official.get("keep"))
        row = normalize_metric_row(
            {
                "config_key": "Batch-33",
                "scope": scope,
                "n": n,
                "keep": keep,
                "fail_count": n - keep if n is not None and keep is not None else None,
                "cer": official.get("cer"),
                "primary_error": official.get("primary_error"),
                "seedtts_wavlm_ecapa_sim_ref": mean(sim_ref),
                "seedtts_wavlm_ecapa_sim_src": mean(sim_src),
                "seedtts_wavlm_ecapa_ref_bound": ref_bound,
                "ref_content_lcs_f1": mean([ref_content_f1(raw) for raw in scoped_asr]),
            },
            family_hint="baseline",
            source_path=path,
            is_baseline=True,
        )
        rows.append(row)
    errors.append(
        "legacy Batch-33 directory has no independent SpeechBrain ECAPA fields; "
        "pass an enriched JSON/TSV baseline to fill them"
    )
    return rows, errors


def read_baseline(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if path.is_dir():
        if (path / "dual_encoder_cases.csv").exists():
            return read_dual_encoder_baseline_dir(path)
        return read_legacy_baseline_dir(path)
    return read_matrix_file(path, "baseline", baseline=True)


def newest_existing(paths: Iterable[Path]) -> Path | None:
    candidates = [path for path in paths if path.exists()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def discover_matrix(project_root: Path, family: str) -> Path | None:
    base = project_root / "testset" / "outputs"
    patterns = (
        f"ver23_family_final_seedtts320_{family}_step3000_*/aggregate/{family}_step3000.official_matrix.json",
        f"ver23_family_final_seedtts320_{family}_step3000_*/aggregate/{family}_step3000.official_matrix.tsv",
        f"**/{family}_step3000.official_matrix.json",
        f"**/{family}_step3000.official_matrix.tsv",
    )
    for pattern in patterns:
        selected = newest_existing(base.glob(pattern))
        if selected is not None:
            return selected
    return None


def discover_baseline(project_root: Path) -> Path | None:
    enriched_cases = newest_existing(
        (project_root / "testset" / "outputs").glob(
            "seedtts_baseline_dual_encoder_*/dual_encoder_cases.csv"
        )
    )
    if enriched_cases is not None:
        return enriched_cases.parent
    preferred = (
        project_root
        / "testset/outputs/ver23_content_side_text_bypass_3k_seedtts320_20260710"
        / "ver23_content_side_text_bypass_3k_step-3000_seedtts320_all_d2d3_seed1234"
    )
    if preferred.exists():
        return preferred
    candidates = (project_root / "testset" / "outputs").glob(
        "**/ver23_content_side_text_bypass_3k_step-3000*seedtts320*"
    )
    return newest_existing(candidate for candidate in candidates if candidate.is_dir())


def merge_expected_rows(
    loaded_rows: Sequence[dict[str, Any]],
    *,
    family: str,
    expected_configs: Sequence[str],
    baseline: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in loaded_rows:
        scale = row.get("ref_audio_cfg_scale")
        config_key = "Batch-33" if baseline else canonical_config(row.get("config_key", ""), scale)
        scope = clean_text(row.get("scope"))
        key = (config_key, scope)
        if key in by_key:
            errors.append(f"duplicate matrix row {family}/{config_key}/{scope}; last row retained")
        row["config_key"] = config_key
        row["arm"] = "Batch-33" if baseline else config_arm(config_key)
        by_key[key] = row

    output: list[dict[str, Any]] = []
    for config_key in expected_configs:
        for scope in SCOPES:
            key = (config_key, scope)
            row = by_key.get(key)
            if row is None:
                row = placeholder_row(family, config_key, scope, baseline=baseline)
                errors.append(f"missing matrix row: {family}/{config_key}/{scope}")
            output.append(row)
    unexpected = sorted(set(by_key) - {(config, scope) for config in expected_configs for scope in SCOPES})
    for config_key, scope in unexpected:
        errors.append(f"unexpected matrix row ignored: {family}/{config_key}/{scope}")
    return output, errors


def subjective_state(value: Any) -> str:
    text = clean_text(value)
    normalized = text.lower().replace("-", "_").replace(" ", "_")
    if not normalized or normalized in {"pending", "todo", "tbd", "na", "n/a", "待盲听", "待定"}:
        return "pending"
    if normalized in {"clearly_better", "much_better", "明显更像", "明显更好"} or "明显更像" in text:
        return "clearly_better"
    if normalized in {"slightly_better", "better", "略好", "更像"} or "略好" in text:
        return "slightly_better"
    if normalized in {"same", "similar", "tie", "差不多", "相当"} or "差不多" in text:
        return "same"
    if normalized in {"worse", "更不像", "更差"} or "更不像" in text:
        return "worse"
    return "pending"


def read_subjective(path: Path | None) -> tuple[dict[str, dict[str, str]], list[str]]:
    if path is None:
        return {}, ["subjective listening input not provided; placeholders retained"]
    if not path.exists():
        return {}, [f"missing subjective input: {path}"]
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping) and not isinstance(payload.get("rows"), list):
                raw_rows = []
                for key, value in payload.items():
                    if isinstance(value, Mapping):
                        raw_rows.append({"config_key": key, **dict(value)})
                    else:
                        raw_rows.append({"config_key": key, "subjective_vs_batch33": value})
            else:
                raw_rows = extract_json_rows(payload)
        else:
            raw_rows = read_table(path)
    except Exception as exc:
        return {}, [f"failed to parse subjective input {path}: {exc}"]
    output: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    for raw in raw_rows:
        key = clean_text(row_alias(raw, "config_key", "arm", "config", "name"))
        if not key:
            errors.append(f"subjective row without arm/config ignored: {raw}")
            continue
        explicit_scale = row_alias(raw, "ref_audio_cfg_scale", "lambda", "scale")
        if key.strip().upper() == "C2" and explicit_scale in (None, ""):
            scale = None
            canonical = "C2"
        else:
            scale = config_scale(key, explicit_scale)
            canonical = canonical_config(key, scale)
        state = subjective_state(row_alias(raw, "subjective_vs_batch33", "subjective", "result", "vote"))
        output[canonical] = {
            "state": state,
            "notes": clean_text(row_alias(raw, "subjective_notes", "notes", "comment")),
        }
        arm = config_arm(canonical)
        if canonical == arm and arm in ARM_ORDER and arm not in output:
            output[arm] = output[canonical]
    return output, errors


def apply_subjective(rows: Sequence[dict[str, Any]], subjective: Mapping[str, Mapping[str, str]]) -> None:
    for row in rows:
        config_key = clean_text(row.get("config_key"))
        arm = clean_text(row.get("arm"))
        value = subjective.get(config_key) or subjective.get(arm)
        if value:
            row["subjective_vs_batch33"] = value.get("state", "pending")
            row["subjective_notes"] = value.get("notes", "")


def quick20_step(path: Path, row: Mapping[str, Any]) -> int | None:
    for text in (path.as_posix(), clean_text(row.get("run_id"))):
        match = re.search(r"step[-_]?([0-9]+)", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def discover_quick20(project_root: Path) -> list[Path]:
    base = project_root / "trainset" / "qz_jobs"
    paths = list(base.glob("ver23_content_side_batch3436_quick20_step*/metrics.tsv"))
    paths.extend(base.glob("ver23_content_side_batch37_quick20_step*/metrics.tsv"))
    return sorted(set(paths))


def expand_input_paths(paths: Sequence[Path], default_name: str = "metrics.tsv") -> list[Path]:
    output: list[Path] = []
    for path in paths:
        if path.is_dir():
            direct = path / default_name
            if direct.exists():
                output.append(direct)
            else:
                output.extend(path.rglob(default_name))
        else:
            output.append(path)
    return sorted(set(output))


def quick_candidate_rank(path: Path) -> tuple[int, float, str]:
    name = path.as_posix().lower()
    safety = 2 if "cachefix" in name else 1 if "20260711" in name else 0
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return safety, mtime, name


def read_quick20(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    candidates: dict[tuple[str, str, int], tuple[Path, dict[str, Any]]] = {}
    for path in paths:
        if not path.exists():
            errors.append(f"missing quick20 metrics: {path}")
            continue
        try:
            raw_rows = read_table(path)
        except Exception as exc:
            errors.append(f"failed to parse quick20 metrics {path}: {exc}")
            continue
        for raw in raw_rows:
            arm = config_arm(clean_text(row_alias(raw, "arm", "config_key", "run_id")))
            mode = clean_text(row_alias(raw, "mode", "scope"))
            step = quick20_step(path, raw)
            if arm not in ARM_ORDER or mode not in {"no_text", "text"} or step is None:
                errors.append(f"ignored malformed quick20 row in {path}: arm={arm} mode={mode} step={step}")
                continue
            key = (arm, mode, step)
            existing = candidates.get(key)
            if existing is None or quick_candidate_rank(path) > quick_candidate_rank(existing[0]):
                if existing is not None:
                    errors.append(f"quick20 duplicate {key}: preferred {path} over {existing[0]}")
                candidates[key] = (path, raw)
            else:
                errors.append(f"quick20 duplicate {key}: retained {existing[0]}, ignored {path}")

    output: list[dict[str, Any]] = []
    for (arm, mode, step), (path, raw) in sorted(candidates.items(), key=lambda item: (ARM_ORDER.index(item[0][0]), item[0][2], item[0][1])):
        row = normalize_metric_row(raw, family_hint="quick20", config_hint=arm, source_path=path)
        output.append(
            {
                "arm": arm,
                "mode": mode,
                "step": step,
                "n": row["n"],
                "fail_rate": row["fail_rate"],
                "cer": row["cer"],
                "sim_ref": row["seedtts_wavlm_ecapa_sim_ref"],
                "sim_src": row["seedtts_wavlm_ecapa_sim_src"],
                "ref_bound": row["seedtts_wavlm_ecapa_ref_bound"],
                "ref_content_f1": row["ref_content_lcs_f1"],
                "source_path": as_posix(path),
            }
        )
    return output, errors


def summarize_quick20(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    by_arm: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[clean_text(row.get("arm"))].append(row)
    output: dict[str, dict[str, Any]] = {}
    for arm in ARM_ORDER:
        arm_rows = by_arm.get(arm, [])
        no_text = [row for row in arm_rows if row.get("mode") == "no_text"]
        latest_step = max((finite_int(row.get("step")) or -1 for row in arm_rows), default=-1)
        latest_rows = [dict(row) for row in arm_rows if finite_int(row.get("step")) == latest_step]
        flags: list[str] = []
        for row in arm_rows:
            cer = finite_float(row.get("cer"))
            f1 = finite_float(row.get("ref_content_f1"))
            if cer is not None and cer > 0.30:
                flags.append(f"quick20_CER_gt_0.30@{row.get('step')}/{row.get('mode')}={cer:.4f}")
            if f1 is not None and f1 > 0.20:
                flags.append(f"quick20_F1_gt_0.20@{row.get('step')}/{row.get('mode')}={f1:.4f}")
        expected_steps = {500, 1000, 1500, 2000, 2500, 3000}
        observed_steps = {finite_int(row.get("step")) for row in no_text}
        missing_steps = sorted(expected_steps - {step for step in observed_steps if step is not None})
        observed_pairs = {
            (finite_int(row.get("step")), clean_text(row.get("mode")))
            for row in arm_rows
        }
        expected_pairs = {(step, mode) for step in expected_steps for mode in ("no_text", "text")}
        missing_pairs = sorted(expected_pairs - observed_pairs)
        output[arm] = {
            "rows": len(arm_rows),
            "latest_step": None if latest_step < 0 else latest_step,
            "latest": latest_rows,
            "observed_no_text_steps": sorted(step for step in observed_steps if step is not None),
            "missing_no_text_steps": missing_steps,
            "missing_step_mode_pairs": [[step, mode] for step, mode in missing_pairs],
            "max_no_text_cer": max((finite_float(row.get("cer")) for row in no_text if finite_float(row.get("cer")) is not None), default=None),
            "max_no_text_f1": max((finite_float(row.get("ref_content_f1")) for row in no_text if finite_float(row.get("ref_content_f1")) is not None), default=None),
            "red_flags": unique_strings(flags),
            "warnings": [f"missing quick20 step/mode pairs: {missing_pairs}"] if missing_pairs else [],
        }
    return output


STEP_RE = re.compile(r"\bstep=(\d+)/(\d+)\b")
KV_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
FATAL_PATTERNS = {
    "traceback": re.compile(r"Traceback \(most recent call last\)"),
    "cuda_oom": re.compile(r"CUDA out of memory|OutOfMemoryError", re.IGNORECASE),
    "nccl_error": re.compile(r"NCCL[^\n]*(?:error|failed|abort)", re.IGNORECASE),
    "nonfinite_literal": re.compile(r"(?:loss|grad|ratio|metric)=[+-]?(?:nan|inf)\b", re.IGNORECASE),
}


def infer_arm_from_path(path: Path) -> str:
    name = path.as_posix()
    for pattern, arm in (
        (r"batch3436[_/-]B1(?:_|/)", "B1"),
        (r"batch3436[_/-]B2(?:_|/)", "B2"),
        (r"batch3436[_/-]A1(?:_|/)", "A1"),
        (r"batch3436[_/-]B3(?:_|/)", "B3"),
        (r"batch3436[_/-]A2(?:_|/)", "A2"),
        (r"batch37[_/-]C1(?:_|/)", "C1"),
        (r"batch37[_/-]C2(?:_|/)", "C2"),
    ):
        if re.search(pattern, name, flags=re.IGNORECASE):
            return arm
    return config_arm(path.parent.name)


def discover_training_logs(project_root: Path) -> list[Path]:
    base = project_root / "outputs" / "lora_runs"
    paths = list(base.glob("ver23_content_side_batch3436_*_20260710_mtts/train.log"))
    paths.extend(base.glob("ver23_content_side_batch37_*_20260710_mtts/train.log"))
    return sorted(set(paths))


def discover_structured_diagnostics(project_root: Path) -> list[Path]:
    base = project_root / "testset" / "outputs"
    paths = list(base.glob("ver23_batch3436_A2_ctc_greedy_probe_step3000_*/overall_summary.json"))
    paths.extend(base.glob("ver23*A2*ctc*probe*/overall_summary.json"))
    return sorted(set(paths))


def parse_metric_line(line: str) -> tuple[int, int, dict[str, float]] | None:
    step_match = STEP_RE.search(line)
    if step_match is None:
        return None
    values: dict[str, float] = {}
    for key, raw in KV_RE.findall(line):
        value = finite_float(raw)
        if value is not None:
            values[key] = value
    return int(step_match.group(1)), int(step_match.group(2)), values


def metric_summary(records: Sequence[tuple[int, dict[str, float]]], key: str) -> dict[str, float | int | None]:
    values = [(step, data[key]) for step, data in records if key in data and math.isfinite(data[key])]
    if not values:
        return {"n": 0, "mean": None, "min": None, "max": None, "first": None, "last": None, "last500_mean": None}
    last_step = max(step for step, _ in values)
    last500 = [value for step, value in values if step > last_step - 500]
    return {
        "n": len(values),
        "mean": mean(value for _, value in values),
        "min": min(value for _, value in values),
        "max": max(value for _, value in values),
        "first": values[0][1],
        "last": values[-1][1],
        "last500_mean": mean(last500),
    }


def parse_training_log(path: Path) -> dict[str, Any]:
    arm = infer_arm_from_path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    records: list[tuple[int, dict[str, float]]] = []
    max_target = 0
    for line in text.splitlines():
        parsed = parse_metric_line(line)
        if parsed is not None:
            step, target, metrics = parsed
            records.append((step, metrics))
            max_target = max(max_target, target)
    finished_matches = [int(value) for value in re.findall(r"\bfinished global_step=(\d+)\b", text)]
    finished_step = max(finished_matches, default=None)
    fatal = [name for name, pattern in FATAL_PATTERNS.items() if pattern.search(text)]
    summaries = {
        key: metric_summary(records, key)
        for key in (
            "loss",
            "content_guided_attn_loss",
            "content_phoneme_loss",
            "content_phoneme_classifier_acc",
            "content_cross_attn_delta_ratio",
            "content_cross_attn_gate_mean",
            "content_ctc_loss_raw",
            "content_ctc_nonblank_post",
            "ref_audio_cfg_dropped_rows",
            "lora_warmup_active",
            "content_cross_attn_memory_dim",
        )
    }
    flags = list(fatal)
    warnings: list[str] = []
    if finished_step is not None and finished_step < 3000:
        flags.append(f"training_incomplete_step_{finished_step}")
    if finished_step is None:
        warnings.append("missing finished global_step marker")
    loss_values = [data["loss"] for _, data in records if "loss" in data]
    if loss_values:
        median_loss = statistics.median(loss_values)
        if max(loss_values) > max(median_loss * 3.0, median_loss + 5.0):
            flags.append(f"loss_divergence_heuristic_max_{max(loss_values):.4f}_median_{median_loss:.4f}")
    if arm == "A1":
        guided = summaries["content_guided_attn_loss"]
        if guided["max"] is not None and guided["max"] > 2.0:
            flags.append(f"guided_attention_explosion_max_{guided['max']:.4f}")
        elif guided["last500_mean"] is not None and not 0.5 <= guided["last500_mean"] <= 0.7:
            warnings.append(f"A1 guided loss did not enter expected 0.5-0.7 band: {guided['last500_mean']:.4f}")
    if arm == "A2":
        ctc = summaries["content_ctc_loss_raw"]
        ctc_values = [data["content_ctc_loss_raw"] for step, data in records if step >= 120 and "content_ctc_loss_raw" in data]
        if ctc_values:
            ctc_median = statistics.median(ctc_values)
            if max(ctc_values) > max(ctc_median * 5.0, ctc_median + 50.0):
                flags.append(f"ctc_loss_divergence_max_{max(ctc_values):.4f}_median_{ctc_median:.4f}")
        elif ctc["n"] == 0:
            warnings.append("A2 training log has no CTC raw-loss samples")
    if arm == "C1":
        phoneme_acc = summaries["content_phoneme_classifier_acc"]["last500_mean"]
        if phoneme_acc is None:
            warnings.append("C1 phoneme classifier accuracy missing")
        elif phoneme_acc < 0.10:
            warnings.append(f"C1 last-500 phoneme accuracy below 0.10: {phoneme_acc:.4f}")
        memory_values = {
            int(round(data["content_cross_attn_memory_dim"]))
            for _, data in records
            if "content_cross_attn_memory_dim" in data
        }
        if memory_values and memory_values != {512}:
            flags.append(f"C1_unexpected_content_memory_dims_{sorted(memory_values)}")
        warmup_bad = [step for step, data in records if step > 500 and data.get("lora_warmup_active", 0.0) > 0.5]
        if warmup_bad:
            flags.append(f"C1_LoRA_still_frozen_after_500_first_step_{min(warmup_bad)}")
    if arm == "C2":
        dropped = [data.get("ref_audio_cfg_dropped_rows", 0.0) for _, data in records if "ref_audio_cfg_dropped_rows" in data]
        positive_rate = sum(value > 0 for value in dropped) / len(dropped) if dropped else None
        if positive_rate is None:
            warnings.append("C2 ref-audio dropout diagnostics missing")
        elif not 0.05 <= positive_rate <= 0.30:
            warnings.append(f"C2 logged dropout-event rate outside broad 5%-30% audit band: {positive_rate:.3f}")
    else:
        positive_rate = None
    return {
        "arm": arm,
        "source_path": as_posix(path),
        "metric_lines": len(records),
        "target_step": max_target or None,
        "finished_global_step": finished_step,
        "fatal_patterns": fatal,
        "summaries": summaries,
        "c2_logged_dropout_event_rate": positive_rate,
        "red_flags": unique_strings(flags),
        "warnings": unique_strings(warnings),
    }


def read_external_diagnostics(paths: Sequence[Path]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    output: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for path in paths:
        if not path.exists():
            errors.append(f"missing training diagnostic input: {path}")
            continue
        if path.name == "train.log" or path.suffix.lower() == ".log":
            try:
                diagnostic = parse_training_log(path)
            except Exception as exc:
                errors.append(f"failed to parse training log {path}: {exc}")
                continue
            existing = output.get(diagnostic["arm"])
            if existing:
                diagnostic.setdefault("external", {}).update(existing.get("external", {}))
                diagnostic.setdefault("external_summary", {}).update(existing.get("external_summary", {}))
                diagnostic["red_flags"] = unique_strings([*diagnostic.get("red_flags", []), *existing.get("red_flags", [])])
                diagnostic["warnings"] = unique_strings([*diagnostic.get("warnings", []), *existing.get("warnings", [])])
                diagnostic["source_path"] = " | ".join(unique_strings([diagnostic.get("source_path"), existing.get("source_path")]))
            output[diagnostic["arm"]] = diagnostic
            continue
        try:
            if path.suffix.lower() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                raw_rows = extract_json_rows(payload)
                if not raw_rows and isinstance(payload, Mapping):
                    if any(
                        key in payload
                        for key in (
                            "arm",
                            "config_key",
                            "config",
                            "greedy_ter",
                            "ctc_greedy_ter",
                            "blank_frame_rate",
                            "nonblank_frame_rate",
                            "collapse_diagnosis",
                        )
                    ):
                        raw_rows = [dict(payload)]
                    else:
                        raw_rows = []
                        arms = payload.get("arms") if isinstance(payload.get("arms"), Mapping) else payload
                        for arm, values in arms.items():
                            if isinstance(values, Mapping) and config_arm(clean_text(arm)) in ARM_ORDER:
                                raw_rows.append({"arm": arm, **dict(values)})
            else:
                raw_rows = read_table(path)
        except Exception as exc:
            errors.append(f"failed to parse training diagnostic {path}: {exc}")
            continue
        for raw in raw_rows:
            arm = config_arm(clean_text(row_alias(raw, "arm", "config_key", "config", "name")))
            if arm not in ARM_ORDER:
                arm = infer_arm_from_path(path)
            if arm not in ARM_ORDER:
                errors.append(f"diagnostic row with unknown arm ignored in {path}: {arm}")
                continue
            target = output.setdefault(
                arm,
                {
                    "arm": arm,
                    "source_path": as_posix(path),
                    "metric_lines": None,
                    "target_step": None,
                    "finished_global_step": None,
                    "fatal_patterns": [],
                    "summaries": {},
                    "red_flags": [],
                    "warnings": [],
                },
            )
            target.setdefault("external", {}).update(dict(raw))
            target["source_path"] = " | ".join(unique_strings([target.get("source_path"), as_posix(path)]))
            raw_flags = row_alias(raw, "red_flags", "red_flag", "flags")
            if bool_value(raw_flags) is True:
                target["red_flags"].append(f"external_red_flag:{path.name}")
            elif clean_text(raw_flags) and bool_value(raw_flags) is None:
                target["red_flags"].extend(re.split(r"[;,|]", clean_text(raw_flags)))
            blank_rate = finite_float(row_alias(raw, "blank_frame_rate", "ctc_blank_frame_rate", "blank_rate"))
            nonblank_rate = finite_float(row_alias(raw, "nonblank_frame_rate", "nonblank_rate", "ctc_nonblank_rate"))
            ter = finite_float(row_alias(raw, "greedy_ter", "ctc_greedy_ter", "ter"))
            punctuation_collapse = bool_value(row_alias(raw, "punctuation_collapse", "ctc_punctuation_collapse"))
            collapse_diagnosis = raw.get("collapse_diagnosis") if isinstance(raw.get("collapse_diagnosis"), Mapping) else {}
            if punctuation_collapse is None:
                punctuation_collapse = bool_value(collapse_diagnosis.get("punctuation_collapse"))
            punctuation_only_rate = finite_float(row_alias(raw, "punctuation_only_collapse_rate"))
            exact_rate = finite_float(row_alias(raw, "exact_rate"))
            if blank_rate is not None and blank_rate > 0.98:
                target["red_flags"].append(f"ctc_blank_collapse_blank_rate_{blank_rate:.4f}")
            if nonblank_rate is not None and nonblank_rate < 0.01:
                target["red_flags"].append(f"ctc_blank_collapse_nonblank_rate_{nonblank_rate:.4f}")
            if punctuation_collapse is True:
                target["red_flags"].append("ctc_punctuation_collapse")
            external_summary = target.setdefault("external_summary", {})
            for key, value in (
                ("ctc_greedy_ter", ter),
                ("ctc_blank_frame_rate", blank_rate),
                ("ctc_nonblank_frame_rate", nonblank_rate),
                ("ctc_exact_rate", exact_rate),
                ("ctc_punctuation_only_collapse_rate", punctuation_only_rate),
            ):
                if value is not None:
                    external_summary[key] = value
            if collapse_diagnosis:
                external_summary["ctc_collapse_diagnosis"] = dict(collapse_diagnosis)
            target["red_flags"] = unique_strings(target["red_flags"])
    return output, errors


def training_diagnostics(paths: Sequence[Path]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    diagnostics, errors = read_external_diagnostics(paths)
    for arm in ARM_ORDER:
        if arm not in diagnostics:
            diagnostics[arm] = {
                "arm": arm,
                "source_path": "",
                "metric_lines": None,
                "target_step": None,
                "finished_global_step": None,
                "fatal_patterns": [],
                "summaries": {},
                "red_flags": [],
                "warnings": ["training diagnostics missing"],
            }
            errors.append(f"training diagnostics missing for arm={arm}")
    return diagnostics, errors


def add_row_flags(
    rows: Sequence[dict[str, Any]],
    quick: Mapping[str, Mapping[str, Any]],
    diagnostics: Mapping[str, Mapping[str, Any]],
) -> None:
    for row in rows:
        missing = [field for field in METRIC_FIELDS if row.get(field) is None]
        row["missing_fields"] = missing
        row["data_status"] = "complete" if not missing else "partial" if any(row.get(field) is not None for field in METRIC_FIELDS) else "missing"
        if row.get("is_baseline"):
            continue
        arm = clean_text(row.get("arm"))
        quick_flags = list(quick.get(arm, {}).get("red_flags", []))
        # Batch-37 quick20 evaluated the C2 checkpoint only at lambda=1.0.
        # Do not incorrectly propagate that inference-scale-specific content
        # result to the independently evaluated lambda 1.2/1.4/1.6 rows.
        if arm == "C2" and clean_text(row.get("config_key")) != "C2L10":
            quick_flags = [flag for flag in quick_flags if not flag.startswith("quick20_")]
        flags = quick_flags
        flags.extend(diagnostics.get(arm, {}).get("red_flags", []))
        warnings = list(quick.get(arm, {}).get("warnings", []))
        warnings.extend(diagnostics.get(arm, {}).get("warnings", []))
        cer = finite_float(row.get("cer"))
        f1 = finite_float(row.get("ref_content_lcs_f1"))
        if cer is not None and cer > 0.30:
            flags.append(f"final320_CER_gt_0.30/{row.get('scope')}={cer:.4f}")
        if f1 is not None and f1 > 0.20:
            flags.append(f"final320_F1_gt_0.20/{row.get('scope')}={f1:.4f}")
        row["red_flags"] = unique_strings(flags)
        row["warnings"] = unique_strings(warnings)


def apply_thresholds(rows: Sequence[dict[str, Any]]) -> None:
    for row in rows:
        sb_ref = finite_float(row.get("speechbrain_ecapa_sim_ref"))
        seed_src = finite_float(row.get("seedtts_wavlm_ecapa_sim_src"))
        seed_ref = finite_float(row.get("seedtts_wavlm_ecapa_sim_ref"))
        seed_bound = finite_float(row.get("seedtts_wavlm_ecapa_ref_bound"))
        sb_pass = sb_ref is not None and sb_ref >= 0.48
        src_pass = seed_src is not None and seed_src <= 0.40
        row["speechbrain_ref_ge_0p48"] = sb_pass if sb_ref is not None else None
        row["seedtts_src_le_0p40"] = src_pass if seed_src is not None else None
        row["objective_joint_pass"] = sb_pass and src_pass if sb_ref is not None and seed_src is not None else None
        row["case1_objective_pass"] = (
            sb_pass
            and src_pass
            and seed_ref is not None
            and seed_ref > 0.44
            and seed_bound is not None
            and seed_bound > 0.50
        ) if all(value is not None for value in (sb_ref, seed_src, seed_ref, seed_bound)) else None


def no_text_rows(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        clean_text(row.get("config_key")): row
        for row in rows
        if row.get("scope") == "no_text" and not row.get("is_baseline")
    }


def c2_rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    sb_ref = finite_float(row.get("speechbrain_ecapa_sim_ref"))
    seed_src = finite_float(row.get("seedtts_wavlm_ecapa_sim_src"))
    required = [sb_ref, seed_src]
    complete = all(row.get(field) is not None for field in DECISION_REQUIRED_FIELDS)
    joint = sb_ref is not None and seed_src is not None and sb_ref >= 0.48 and seed_src <= 0.40
    threshold_count = int(sb_ref is not None and sb_ref >= 0.48) + int(seed_src is not None and seed_src <= 0.40)
    worst_margin = min(sb_ref - 0.48, 0.40 - seed_src) if all(value is not None for value in required) else -math.inf
    red_flag_free = not bool(row.get("red_flags"))
    def score(value: Any, *, lower_is_better: bool = False) -> float:
        number = finite_float(value)
        if number is None:
            return -math.inf
        return -number if lower_is_better else number

    return (
        int(complete),
        int(red_flag_free),
        int(joint),
        threshold_count,
        worst_margin,
        sb_ref if sb_ref is not None else -math.inf,
        -(seed_src if seed_src is not None else math.inf),
        score(row.get("seedtts_wavlm_ecapa_ref_bound")),
        score(row.get("seedtts_wavlm_ecapa_sim_ref")),
        score(row.get("fail_rate"), lower_is_better=True),
        score(row.get("cer"), lower_is_better=True),
        score(row.get("ref_content_lcs_f1"), lower_is_better=True),
        score(row.get("ref_audio_cfg_scale"), lower_is_better=True),
    )


def select_c2(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("arm") == "C2" and row.get("scope") == "no_text"]
    available = [row for row in candidates if any(row.get(field) is not None for field in DECISION_REQUIRED_FIELDS)]
    fully_complete = [row for row in candidates if all(row.get(field) is not None for field in DECISION_REQUIRED_FIELDS)]
    if not available:
        return {
            "status": "missing",
            "selected_config_key": None,
            "selected_scale": None,
            "reason": "all four C2 lambda no_text rows are missing",
            "ranking": [],
        }
    ranked = sorted(available, key=c2_rank_key, reverse=True)
    selected = ranked[0]
    for row in rows:
        if row.get("config_key") == selected.get("config_key"):
            row["c2_objective_selected"] = True
    status = "complete" if len(fully_complete) == 4 else "provisional"
    ranking = []
    for index, row in enumerate(ranked, start=1):
        ranking.append(
            {
                "rank": index,
                "config_key": row.get("config_key"),
                "lambda": row.get("ref_audio_cfg_scale"),
                "objective_joint_pass": row.get("objective_joint_pass"),
                "speechbrain_sim_ref": row.get("speechbrain_ecapa_sim_ref"),
                "seedtts_sim_src": row.get("seedtts_wavlm_ecapa_sim_src"),
                "seedtts_sim_ref": row.get("seedtts_wavlm_ecapa_sim_ref"),
                "seedtts_ref_bound": row.get("seedtts_wavlm_ecapa_ref_bound"),
                "fail_rate": row.get("fail_rate"),
                "cer": row.get("cer"),
                "red_flags": row.get("red_flags", []),
            }
        )
    return {
        "status": status,
        "selected_config_key": selected.get("config_key"),
        "selected_scale": selected.get("ref_audio_cfg_scale"),
        "reason": (
            "lexicographic objective rank: complete data, no red flag, joint threshold, "
            "threshold count, worst threshold margin, SpeechBrain ref, lower SeedTTS src, "
            "ref-bound, SeedTTS ref, lower fail/CER/F1, lower lambda tie-break"
        ),
        "ranking": ranking,
    }


def conceptual_rows(rows: Sequence[dict[str, Any]], c2_selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    lookup = no_text_rows(rows)
    output: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        config_key = clean_text(c2_selection.get("selected_config_key")) if arm == "C2" else arm
        row = lookup.get(config_key)
        if row is None:
            row = placeholder_row("batch37" if arm in {"C1", "C2"} else "batch3436", arm if arm != "C2" else "C2L10", "no_text")
            row["arm"] = arm
            row["arm_title"] = ARM_META[arm]["title"]
            if arm == "C2":
                row["config_key"] = "C2(best lambda pending)"
                row["ref_audio_cfg_scale"] = None
                row["warnings"] = ["C2 objective-best lambda cannot be selected until final metrics exist"]
        output.append(row)
    return output


def row_complete(row: Mapping[str, Any] | None, fields: Sequence[str] = DECISION_REQUIRED_FIELDS) -> bool:
    return row is not None and all(row.get(field) is not None for field in fields)


def case_entry(status: str, evidence: Sequence[str], recommendation: str) -> dict[str, Any]:
    return {"status": status, "evidence": list(evidence), "recommendation": recommendation}


def build_decision(
    rows: Sequence[dict[str, Any]],
    baseline_rows: Sequence[dict[str, Any]],
    c2_selection: Mapping[str, Any],
    diagnostics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    arms = conceptual_rows(rows, c2_selection)
    by_arm = {clean_text(row.get("arm")): row for row in arms}
    baseline = next((row for row in baseline_rows if row.get("scope") == "no_text"), None)
    missing_arms = [arm for arm in ARM_ORDER if arm not in by_arm]
    incomplete_arms = [arm for arm, row in by_arm.items() if not row_complete(row)]
    diagnostic_missing = [
        arm for arm in ARM_ORDER
        if not diagnostics.get(arm, {}).get("source_path")
    ]

    def evidence_known(arm: str) -> bool:
        row = by_arm.get(arm, {})
        quick_missing = any("missing quick20" in clean_text(warning) for warning in row.get("warnings", []))
        return bool(diagnostics.get(arm, {}).get("source_path")) and not quick_missing

    case1_candidates = [
        arm for arm, row in by_arm.items()
        if row.get("case1_objective_pass") is True and not row.get("red_flags")
    ]
    case1_subjective = [
        arm for arm in case1_candidates
        if by_arm[arm].get("subjective_vs_batch33") == "clearly_better" and evidence_known(arm)
    ]
    if case1_subjective:
        case1_status = "PASS"
        case1_evidence = [f"{arm} clears all objective thresholds and is subjectively clearly better" for arm in case1_subjective]
    elif case1_candidates and any(
        by_arm[arm].get("subjective_vs_batch33") == "pending" or not evidence_known(arm)
        for arm in case1_candidates
    ):
        case1_status = "PENDING"
        case1_evidence = [f"objective candidate awaiting listening: {arm}" for arm in case1_candidates]
    elif missing_arms or incomplete_arms:
        case1_status = "PENDING"
        case1_evidence = [f"missing/incomplete no_text final rows: {missing_arms + incomplete_arms}"]
    else:
        case1_status = "FAIL"
        case1_evidence = ["no red-flag-free arm satisfies SpeechBrain ref>=0.48, SeedTTS src<=0.40, SeedTTS ref>0.44, ref-bound>50%, and clearly-better listening"]

    case2_objective = [
        arm for arm, row in by_arm.items()
        if (finite_float(row.get("seedtts_wavlm_ecapa_sim_ref")) is not None)
        and 0.43 <= float(row["seedtts_wavlm_ecapa_sim_ref"]) <= 0.45
        and not row.get("red_flags")
    ]
    case2_subjective = [
        arm for arm in case2_objective
        if by_arm[arm].get("subjective_vs_batch33") in {"slightly_better", "clearly_better"}
        and evidence_known(arm)
    ]
    if len(case2_subjective) >= 2:
        case2_status = "PASS"
        case2_evidence = [f"multiple mild improvements: {case2_subjective}"]
    elif len(case2_objective) >= 2 and any(
        by_arm[arm].get("subjective_vs_batch33") == "pending" or not evidence_known(arm)
        for arm in case2_objective
    ):
        case2_status = "PENDING"
        case2_evidence = [f"multiple 0.43-0.45 objective candidates await listening: {case2_objective}"]
    elif missing_arms or incomplete_arms:
        case2_status = "PENDING"
        case2_evidence = [f"missing/incomplete no_text final rows: {missing_arms + incomplete_arms}"]
    else:
        case2_status = "FAIL"
        case2_evidence = [f"fewer than two red-flag-free 0.43-0.45 arms with positive listening: objective={case2_objective}, positive={case2_subjective}"]

    comparable: list[str] = []
    worse: list[str] = []
    comparison_pending = False
    if baseline is None or not all(baseline.get(field) is not None for field in ("fail_rate", "cer", "seedtts_wavlm_ecapa_sim_ref", "seedtts_wavlm_ecapa_ref_bound")):
        comparison_pending = True
    else:
        for arm, row in by_arm.items():
            required = ("fail_rate", "cer", "seedtts_wavlm_ecapa_sim_ref", "seedtts_wavlm_ecapa_ref_bound")
            if not all(row.get(field) is not None for field in required):
                comparison_pending = True
                continue
            sim_delta = float(row["seedtts_wavlm_ecapa_sim_ref"]) - float(baseline["seedtts_wavlm_ecapa_sim_ref"])
            bound_delta = float(row["seedtts_wavlm_ecapa_ref_bound"]) - float(baseline["seedtts_wavlm_ecapa_ref_bound"])
            fail_delta = float(row["fail_rate"]) - float(baseline["fail_rate"])
            cer_delta = float(row["cer"]) - float(baseline["cer"])
            if abs(sim_delta) <= 0.01 and abs(bound_delta) <= 0.10 and abs(fail_delta) <= 0.05 and abs(cer_delta) <= 0.05:
                comparable.append(arm)
            not_improved = sim_delta <= 0.005 and bound_delta <= 0.05
            materially_degraded = sim_delta <= -0.01 or bound_delta <= -0.10 or fail_delta >= 0.05 or cer_delta >= 0.05
            if not_improved and materially_degraded:
                worse.append(arm)

    if (
        len(comparable) == len(ARM_ORDER)
        and all(by_arm[arm].get("subjective_vs_batch33") == "same" for arm in comparable)
        and all(evidence_known(arm) for arm in comparable)
    ):
        case3_status = "PASS"
        case3_evidence = ["all seven conceptual arms are within the registered Batch-33 comparison tolerances and listening says same"]
    elif len(comparable) == len(ARM_ORDER) and any(
        by_arm[arm].get("subjective_vs_batch33") == "pending" or not evidence_known(arm)
        for arm in comparable
    ):
        case3_status = "PENDING"
        case3_evidence = ["all arms are objectively comparable; listening is incomplete"]
    elif comparison_pending or missing_arms:
        case3_status = "PENDING"
        case3_evidence = ["Batch-33 comparison or one/more arm rows are incomplete"]
    else:
        case3_status = "FAIL"
        case3_evidence = [f"objectively comparable arms={comparable}; expected all {len(ARM_ORDER)}"]

    if (
        len(worse) == len(ARM_ORDER)
        and all(by_arm[arm].get("subjective_vs_batch33") == "worse" for arm in worse)
        and all(evidence_known(arm) for arm in worse)
    ):
        case4_status = "PASS"
        case4_evidence = ["all seven conceptual arms are materially worse and listening agrees"]
    elif len(worse) == len(ARM_ORDER) and any(
        by_arm[arm].get("subjective_vs_batch33") == "pending" or not evidence_known(arm)
        for arm in worse
    ):
        case4_status = "PENDING"
        case4_evidence = ["all arms are objectively worse; listening is incomplete"]
    elif comparison_pending or missing_arms:
        case4_status = "PENDING"
        case4_evidence = ["Batch-33 comparison or one/more arm rows are incomplete"]
    else:
        case4_status = "FAIL"
        case4_evidence = [f"materially worse arms={worse}; expected all {len(ARM_ORDER)}"]

    cases = {
        "Case 1": case_entry(case1_status, case1_evidence, "Use the winning arm as the mainline; proceed to 30k plus data v2."),
        "Case 2": case_entry(case2_status, case2_evidence, "Combine the two best isolated variables and rerun a 3k probe."),
        "Case 3": case_entry(case3_status, case3_evidence, "The tested hypotheses are not primary; consider ref-audio CFG, 30k+data-v2, or an FM decoder."),
        "Case 4": case_entry(case4_status, case4_evidence, "Treat AR+LoRA as the bottleneck and move to a flow-matching decoder."),
    }
    selected_case = next((name for name in ("Case 1", "Case 2", "Case 3", "Case 4") if cases[name]["status"] == "PASS"), None)
    pending_reasons: list[str] = []
    if missing_arms:
        pending_reasons.append(f"missing conceptual arms: {missing_arms}")
    if incomplete_arms:
        pending_reasons.append(f"incomplete no_text metrics: {incomplete_arms}")
    if c2_selection.get("status") != "complete":
        pending_reasons.append(f"C2 lambda selection is {c2_selection.get('status')}")
    subjective_pending = [arm for arm, row in by_arm.items() if row.get("subjective_vs_batch33") == "pending"]
    if subjective_pending:
        pending_reasons.append(f"subjective listening pending: {subjective_pending}")
    if diagnostic_missing:
        pending_reasons.append(f"training diagnostics missing: {diagnostic_missing}")
    if selected_case is None and not pending_reasons and all(entry["status"] == "FAIL" for entry in cases.values()):
        pending_reasons.append("complete measurements fall outside the four pre-registered case definitions")
    return {
        "selected_case": selected_case or "PENDING",
        "selected_recommendation": cases[selected_case]["recommendation"] if selected_case else "Wait for missing objective/listening evidence; do not force a Case 1-4 label.",
        "conceptual_arm_configs": {
            arm: by_arm.get(arm, {}).get("config_key") for arm in ARM_ORDER
        },
        "case_definitions": {
            "Case 1": "joint threshold + SeedTTS ref>0.44 + ref-bound>50% + clearly-better listening + no red flag",
            "Case 2": "at least two red-flag-free arms at SeedTTS ref 0.43-0.45 with positive listening",
            "Case 3": "all conceptual arms comparable to Batch-33 within registered tolerances and listening says same",
            "Case 4": "all conceptual arms materially worse than Batch-33 and listening says worse",
        },
        "comparison_tolerances": {
            "case3_abs_seedtts_sim_ref": 0.01,
            "case3_abs_seedtts_ref_bound": 0.10,
            "case3_abs_fail_rate": 0.05,
            "case3_abs_cer": 0.05,
            "case4_seedtts_sim_ref_degradation": 0.01,
            "case4_seedtts_ref_bound_degradation": 0.10,
            "case4_fail_or_cer_degradation": 0.05,
        },
        "cases": cases,
        "pending_reasons": pending_reasons,
    }


def fmt_number(value: Any, digits: int = 4) -> str:
    number = finite_float(value)
    return "MISSING" if number is None else f"{number:.{digits}f}"


def fmt_percent(value: Any, digits: int = 1) -> str:
    number = finite_float(value)
    return "MISSING" if number is None else f"{100.0 * number:.{digits}f}%"


def fmt_bool(value: Any) -> str:
    return "MISSING" if value is None else "PASS" if bool(value) else "FAIL"


def tsv_value(value: Any) -> Any:
    if isinstance(value, list):
        return " | ".join(clean_text(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else value


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_TSV_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: tsv_value(row.get(field)) for field in OUTPUT_TSV_FIELDS})


def latest_quick_row(summary: Mapping[str, Any], mode: str) -> Mapping[str, Any] | None:
    return next((row for row in summary.get("latest", []) if row.get("mode") == mode), None)


def render_markdown(payload: Mapping[str, Any]) -> str:
    rows = payload["rows"]
    baseline_rows = [row for row in rows if row.get("is_baseline")]
    experiment_rows = [row for row in rows if not row.get("is_baseline")]
    c2 = payload["c2_selection"]
    decision = payload["decision"]
    conceptual = conceptual_rows(experiment_rows, c2)
    lines = [
        "# Batch-34+36+37 七臂最终汇总与决策",
        "",
        f"- 生成时间：`{payload['generated_at_utc']}`",
        f"- 数据状态：`{payload['status']}`",
        f"- 最终 Case：`{decision['selected_case']}`",
        f"- 建议：{decision['selected_recommendation']}",
        "- `fail` 使用 004042 的 official `content_keep=False` 定义。",
        "- SeedTTS 是 WavLM-Large + ECAPA-TDNN 口径；SpeechBrain ECAPA 是独立交叉编码器，二者不可互换。",
        "- ref-bound 定义为 `sim(gen,ref) - sim(gen,src) > 0.05`。",
        "",
        "## 输入与完整性",
        "",
        "| input | status | path |",
        "|---|---|---|",
    ]
    for name, info in payload["inputs"].items():
        lines.append(f"| {name} | {info.get('status')} | `{info.get('path') or 'MISSING'}` |")
    if payload["missing"]:
        lines.extend(["", "当前缺失/警告（不会用记忆数字填补）：", ""])
        lines.extend(f"- {item}" for item in payload["missing"])

    lines.extend([
        "",
        "## 七臂 no_text 主表",
        "",
        "C2 仅用自动选择的 objective 最优 lambda 进入七臂 Case 判定；四个 lambda 仍在后续明细中独立保留。",
        "",
        "| Arm | config | fail | CER | SeedTTS ref | SeedTTS src | SeedTTS ref-bound | SpeechBrain ref | SpeechBrain src | SpeechBrain ref-bound | F1 | 主观 vs Batch-33 | 红旗 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for row in conceptual:
        lines.append(
            "| {arm} {title} | {config} | {fail} | {cer} | {sref} | {ssrc} | {sbound} | {eref} | {esrc} | {ebound} | {f1} | {subjective} | {flags} |".format(
                arm=row["arm"], title=row.get("arm_title", ""), config=row["config_key"],
                fail=fmt_percent(row.get("fail_rate")), cer=fmt_number(row.get("cer")),
                sref=fmt_number(row.get("seedtts_wavlm_ecapa_sim_ref")),
                ssrc=fmt_number(row.get("seedtts_wavlm_ecapa_sim_src")),
                sbound=fmt_percent(row.get("seedtts_wavlm_ecapa_ref_bound")),
                eref=fmt_number(row.get("speechbrain_ecapa_sim_ref")),
                esrc=fmt_number(row.get("speechbrain_ecapa_sim_src")),
                ebound=fmt_percent(row.get("speechbrain_ecapa_ref_bound")),
                f1=fmt_number(row.get("ref_content_lcs_f1")),
                subjective=row.get("subjective_vs_batch33", "pending"),
                flags="<br>".join(row.get("red_flags", [])) or "none",
            )
        )
    if baseline_rows:
        baseline = next((row for row in baseline_rows if row.get("scope") == "no_text"), baseline_rows[0])
        lines.append(
            "| Batch-33 baseline | Batch-33 | {fail} | {cer} | {sref} | {ssrc} | {sbound} | {eref} | {esrc} | {ebound} | {f1} | baseline | none |".format(
                fail=fmt_percent(baseline.get("fail_rate")), cer=fmt_number(baseline.get("cer")),
                sref=fmt_number(baseline.get("seedtts_wavlm_ecapa_sim_ref")),
                ssrc=fmt_number(baseline.get("seedtts_wavlm_ecapa_sim_src")),
                sbound=fmt_percent(baseline.get("seedtts_wavlm_ecapa_ref_bound")),
                eref=fmt_number(baseline.get("speechbrain_ecapa_sim_ref")),
                esrc=fmt_number(baseline.get("speechbrain_ecapa_sim_src")),
                ebound=fmt_percent(baseline.get("speechbrain_ecapa_ref_bound")),
                f1=fmt_number(baseline.get("ref_content_lcs_f1")),
            )
        )

    lines.extend([
        "",
        "## C2 lambda objective 选择",
        "",
        f"- 状态：`{c2.get('status')}`",
        f"- 选中：`{c2.get('selected_config_key') or 'MISSING'}`，lambda=`{c2.get('selected_scale') if c2.get('selected_scale') is not None else 'MISSING'}`",
        f"- 规则：{c2.get('reason')}",
        "",
        "| rank | config | lambda | joint | SpeechBrain ref | SeedTTS src | SeedTTS ref | SeedTTS ref-bound | fail | CER | red flags |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    if c2.get("ranking"):
        for rank in c2["ranking"]:
            lines.append(
                f"| {rank['rank']} | {rank['config_key']} | {fmt_number(rank['lambda'], 1)} | {fmt_bool(rank['objective_joint_pass'])} | "
                f"{fmt_number(rank['speechbrain_sim_ref'])} | {fmt_number(rank['seedtts_sim_src'])} | "
                f"{fmt_number(rank['seedtts_sim_ref'])} | {fmt_percent(rank['seedtts_ref_bound'])} | "
                f"{fmt_percent(rank['fail_rate'])} | {fmt_number(rank['cer'])} | {'<br>'.join(rank['red_flags']) or 'none'} |"
            )
    else:
        lines.append("| - | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | final matrix missing |")

    lines.extend([
        "",
        "## 全配置 × scope 正式矩阵",
        "",
        "| config | cfg | scope | n | fail | CER | SeedTTS ref/src/bound | SpeechBrain ref/src/bound | F1 | joint | 主观 | red flags | status |",
        "|---|---:|---|---:|---:|---:|---|---|---:|---|---|---|---|",
    ])
    ordered = sorted(
        experiment_rows,
        key=lambda row: (
            CONFIG_ORDER.index(row["config_key"]) if row["config_key"] in CONFIG_ORDER else 999,
            SCOPES.index(row["scope"]) if row["scope"] in SCOPES else 999,
        ),
    )
    for row in ordered:
        lines.append(
            f"| {row['config_key']} | {fmt_number(row.get('ref_audio_cfg_scale'), 1)} | {row['scope']} | {row.get('n') if row.get('n') is not None else 'MISSING'} | "
            f"{fmt_percent(row.get('fail_rate'))} | {fmt_number(row.get('cer'))} | "
            f"{fmt_number(row.get('seedtts_wavlm_ecapa_sim_ref'))}/{fmt_number(row.get('seedtts_wavlm_ecapa_sim_src'))}/{fmt_percent(row.get('seedtts_wavlm_ecapa_ref_bound'))} | "
            f"{fmt_number(row.get('speechbrain_ecapa_sim_ref'))}/{fmt_number(row.get('speechbrain_ecapa_sim_src'))}/{fmt_percent(row.get('speechbrain_ecapa_ref_bound'))} | "
            f"{fmt_number(row.get('ref_content_lcs_f1'))} | {fmt_bool(row.get('objective_joint_pass'))} | {row.get('subjective_vs_batch33')} | "
            f"{'<br>'.join(row.get('red_flags', [])) or 'none'} | {row.get('data_status')} |"
        )

    lines.extend([
        "",
        "## Quick20 与训练诊断",
        "",
        "| Arm | latest step | latest no_text fail/CER/sim(ref)/ref-bound/F1 | max no_text CER/F1 | training step | guided last500 | phoneme acc last500 | delta/gate last500 | CTC raw/nonblank-post last500 | CTC probe TER/blank/nonblank/punct-only | red flags / warnings |",
        "|---|---:|---|---|---:|---:|---:|---|---|---:|---|",
    ])
    quick = payload["quick20_summary"]
    diagnostics = payload["training_diagnostics"]
    for arm in ARM_ORDER:
        q = quick.get(arm, {})
        latest = latest_quick_row(q, "no_text")
        diag = diagnostics.get(arm, {})
        summaries = diag.get("summaries", {})
        guided = summaries.get("content_guided_attn_loss", {}).get("last500_mean")
        phoneme_acc = summaries.get("content_phoneme_classifier_acc", {}).get("last500_mean")
        ctc = summaries.get("content_ctc_loss_raw", {}).get("last500_mean")
        ctc_nonblank = summaries.get("content_ctc_nonblank_post", {}).get("last500_mean")
        delta = summaries.get("content_cross_attn_delta_ratio", {}).get("last500_mean")
        gate = summaries.get("content_cross_attn_gate_mean", {}).get("last500_mean")
        external_summary = diag.get("external_summary", {})
        ctc_ter = external_summary.get("ctc_greedy_ter")
        ctc_blank = external_summary.get("ctc_blank_frame_rate")
        ctc_frame_nonblank = external_summary.get("ctc_nonblank_frame_rate")
        ctc_punct = external_summary.get("ctc_punctuation_only_collapse_rate")
        latest_text = "MISSING" if latest is None else "/".join(
            [
                fmt_percent(latest.get("fail_rate")),
                fmt_number(latest.get("cer")),
                fmt_number(latest.get("sim_ref")),
                fmt_percent(latest.get("ref_bound")),
                fmt_number(latest.get("ref_content_f1")),
            ]
        )
        flags = unique_strings([*q.get("red_flags", []), *diag.get("red_flags", []), *q.get("warnings", []), *diag.get("warnings", [])])
        lines.append(
            f"| {arm} | {q.get('latest_step') if q.get('latest_step') is not None else 'MISSING'} | {latest_text} | "
            f"{fmt_number(q.get('max_no_text_cer'))}/{fmt_number(q.get('max_no_text_f1'))} | "
            f"{diag.get('finished_global_step') if diag.get('finished_global_step') is not None else 'MISSING'} | "
            f"{fmt_number(guided)} | {fmt_number(phoneme_acc)} | {fmt_number(delta)}/{fmt_number(gate)} | "
            f"{fmt_number(ctc)}/{fmt_number(ctc_nonblank)} | "
            f"{fmt_number(ctc_ter)}/{fmt_percent(ctc_blank)}/{fmt_percent(ctc_frame_nonblank)}/{fmt_percent(ctc_punct)} | "
            f"{'<br>'.join(flags) or 'none'} |"
        )

    lines.extend([
        "",
        "## Case 1-4 决策矩阵",
        "",
        "联合客观门槛：no_text SpeechBrain ECAPA sim(ref) >= 0.48 且 SeedTTS sim(src) <= 0.40。Case 1 还要求 SeedTTS sim(ref) > 0.44、SeedTTS ref-bound > 50%、主观明显更像且无红旗。",
        "",
        "| Case | status | evidence | action |",
        "|---|---|---|---|",
    ])
    for name in ("Case 1", "Case 2", "Case 3", "Case 4"):
        case = decision["cases"][name]
        lines.append(f"| {name} | {case['status']} | {'<br>'.join(case['evidence'])} | {case['recommendation']} |")
    if decision["pending_reasons"]:
        lines.extend(["", "阻止最终 Case 落锤的项目：", ""])
        lines.extend(f"- {reason}" for reason in decision["pending_reasons"])

    lines.extend([
        "",
        "## 主观占位填写",
        "",
        "在 TSV/JSON 输入中按 arm 或具体 C2 config 填 `subjective_vs_batch33`：",
        "",
        "- `clearly_better`：比 Batch-33 明显更像 ref",
        "- `slightly_better`：比 Batch-33 略好",
        "- `same`：跟 Batch-33 差不多",
        "- `worse`：比 Batch-33 更不像",
        "- `pending`：尚未盲听",
        "",
        "## 论文占位",
        "",
        f"- §5 BNF side-pathway ablation 主因：`{'待定' if decision['selected_case'] == 'PENDING' else decision['selected_case']}`",
        "- §4 ECAPA-TDNN metric saturation：待结合盲听结果量化双编码器与主观 gap。",
        "",
        f"JSON: `{payload['output_paths']['json']}`",
        f"TSV: `{payload['output_paths']['tsv']}`",
        "",
    ])
    return "\n".join(lines)


def resolve_input(explicit: Path | None, discovered: Path | None) -> tuple[Path | None, str]:
    if explicit is not None:
        return explicit, "explicit"
    if discovered is not None:
        return discovered, "auto-discovered"
    return None, "missing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Batch-34+36+37 seven-arm final Markdown/TSV/JSON report. "
            "Incomplete inputs produce explicit MISSING rows instead of invented metrics."
        )
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--batch3436-matrix", type=Path, help="Official matrix JSON/TSV for B1/B2/A1/B3/A2.")
    parser.add_argument("--batch37-matrix", type=Path, help="Official matrix JSON/TSV for C1 and C2 lambda scan.")
    parser.add_argument(
        "--batch33-baseline",
        type=Path,
        help=(
            "Batch-33 matrix JSON/TSV, enriched dual-encoder output directory, "
            "or legacy SeedTTS-320 run directory."
        ),
    )
    parser.add_argument(
        "--quick20",
        type=Path,
        action="append",
        default=[],
        help="Quick20 metrics.tsv or containing directory; repeatable. Auto-discovered when omitted.",
    )
    parser.add_argument(
        "--training-diagnostics",
        type=Path,
        action="append",
        default=[],
        help="train.log or structured JSON/TSV diagnostic; repeatable. Training logs are auto-discovered when omitted.",
    )
    parser.add_argument(
        "--subjective",
        type=Path,
        help="Optional JSON/TSV subjective votes with arm/config_key and subjective_vs_batch33.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "docs/assets/ver23_batch343637_final_decision_20260711",
        help="Output path without extension; .md/.tsv/.json are written.",
    )
    parser.add_argument("--no-auto-discover", action="store_true", help="Do not search project-standard output locations.")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Write the report, then exit 3 if any required final input/metric remains missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    auto = not args.no_auto_discover

    batch3436_path, batch3436_origin = resolve_input(
        args.batch3436_matrix,
        discover_matrix(project_root, "batch3436") if auto else None,
    )
    batch37_path, batch37_origin = resolve_input(
        args.batch37_matrix,
        discover_matrix(project_root, "batch37") if auto else None,
    )
    baseline_path, baseline_origin = resolve_input(
        args.batch33_baseline,
        discover_baseline(project_root) if auto else None,
    )

    missing: list[str] = []
    input_info: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    for family, path, origin in (
        ("batch3436", batch3436_path, batch3436_origin),
        ("batch37", batch37_path, batch37_origin),
    ):
        if path is None:
            loaded: list[dict[str, Any]] = []
            errors = [f"{family} official matrix not found"]
        else:
            loaded, errors = read_matrix_file(path, family)
        merged, merge_errors = merge_expected_rows(
            loaded,
            family=family,
            expected_configs=FAMILY_CONFIGS[family],
        )
        all_rows.extend(merged)
        missing.extend(errors)
        missing.extend(merge_errors)
        input_info[family] = {
            "status": "loaded" if path is not None and not errors else "partial" if path is not None else "missing",
            "origin": origin,
            "path": as_posix(path),
        }

    if baseline_path is None:
        baseline_loaded: list[dict[str, Any]] = []
        baseline_errors = ["Batch-33 baseline not found"]
    else:
        baseline_loaded, baseline_errors = read_baseline(baseline_path)
    baseline_rows, baseline_merge_errors = merge_expected_rows(
        baseline_loaded,
        family="baseline",
        expected_configs=("Batch-33",),
        baseline=True,
    )
    all_rows.extend(baseline_rows)
    missing.extend(baseline_errors)
    missing.extend(baseline_merge_errors)
    input_info["batch33_baseline"] = {
        "status": "loaded" if baseline_path is not None and not baseline_errors else "partial" if baseline_path is not None else "missing",
        "origin": baseline_origin,
        "path": as_posix(baseline_path),
    }

    quick_paths = expand_input_paths(args.quick20) if args.quick20 else (discover_quick20(project_root) if auto else [])
    quick_rows, quick_errors = read_quick20(quick_paths)
    quick_summary = summarize_quick20(quick_rows)
    missing.extend(quick_errors)
    if not quick_paths:
        missing.append("quick20 metrics not found")
    input_info["quick20"] = {
        "status": "loaded" if quick_rows else "missing",
        "origin": "explicit" if args.quick20 else "auto-discovered" if quick_paths else "missing",
        "path": " | ".join(as_posix(path) for path in quick_paths),
    }

    explicit_diagnostics = expand_input_paths(args.training_diagnostics, "train.log")
    automatic_diagnostics = (
        [*discover_training_logs(project_root), *discover_structured_diagnostics(project_root)]
        if auto
        else []
    )
    diagnostic_paths = sorted(
        set([*automatic_diagnostics, *explicit_diagnostics]),
        key=lambda path: (0 if path.suffix.lower() == ".log" else 1, path.as_posix()),
    )
    diagnostics, diagnostic_errors = training_diagnostics(diagnostic_paths)
    missing.extend(diagnostic_errors)
    input_info["training_diagnostics"] = {
        "status": "loaded" if diagnostic_paths else "missing",
        "origin": "explicit" if args.training_diagnostics else "auto-discovered" if diagnostic_paths else "missing",
        "path": " | ".join(as_posix(path) for path in diagnostic_paths),
    }

    subjective, subjective_errors = read_subjective(args.subjective)
    missing.extend(subjective_errors)
    input_info["subjective"] = {
        "status": "loaded" if subjective else "missing",
        "origin": "explicit" if args.subjective else "missing",
        "path": as_posix(args.subjective),
    }
    apply_subjective(all_rows, subjective)
    add_row_flags(all_rows, quick_summary, diagnostics)
    apply_thresholds(all_rows)
    experiment_rows = [row for row in all_rows if not row.get("is_baseline")]
    c2_selection = select_c2(experiment_rows)
    decision = build_decision(experiment_rows, baseline_rows, c2_selection, diagnostics)

    required_final_missing = [
        f"{row['config_key']}/{row['scope']}:{','.join(row['missing_fields'])}"
        for row in all_rows
        if row.get("missing_fields")
    ]
    status = "complete" if not required_final_missing and not decision["pending_reasons"] else "incomplete"
    output_prefix = args.output_prefix.resolve()
    output_paths = {
        "md": str(output_prefix.with_suffix(".md")),
        "tsv": str(output_prefix.with_suffix(".tsv")),
        "json": str(output_prefix.with_suffix(".json")),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "status": status,
        "inputs": input_info,
        "thresholds": {
            "speechbrain_ecapa_sim_ref_min": 0.48,
            "seedtts_wavlm_ecapa_sim_src_max": 0.40,
            "case1_seedtts_wavlm_ecapa_sim_ref_strict_min": 0.44,
            "case1_seedtts_wavlm_ecapa_ref_bound_strict_min": 0.50,
            "red_flag_cer_strict_max": 0.30,
            "red_flag_ref_content_lcs_f1_strict_max": 0.20,
        },
        "missing": unique_strings([*missing, *required_final_missing]),
        "rows": all_rows,
        "quick20_rows": quick_rows,
        "quick20_summary": quick_summary,
        "training_diagnostics": diagnostics,
        "c2_selection": c2_selection,
        "decision": decision,
        "output_paths": output_paths,
    }

    json_path = Path(output_paths["json"])
    tsv_path = Path(output_paths["tsv"])
    md_path = Path(output_paths["md"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    write_tsv(tsv_path, all_rows)
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"[batch343637-decision] status={status}")
    print(f"[batch343637-decision] c2={c2_selection['status']} selected={c2_selection['selected_config_key']}")
    print(f"[batch343637-decision] case={decision['selected_case']}")
    print(f"[batch343637-decision] wrote {md_path}")
    print(f"[batch343637-decision] wrote {tsv_path}")
    print(f"[batch343637-decision] wrote {json_path}")
    if args.require_complete and status != "complete":
        print("[batch343637-decision] incomplete input/decision state", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a blinded no-text listening page for Ver2.3 ablation runs.

The page compares three generated candidates for the same SeedTTS case:
Ver2.3, Batch-33, and one current arm.  Candidate identities are randomized
to A/B/C independently per case.  The identity mapping is written only to
``manifest.json`` and is deliberately not embedded in ``index.html``.

Typical output is a child of the existing listening frontend root, for
example::

    outputs/listening_frontend/seedtts_valid_benchmark/\
        batch343637_subjective_20260711/B3

All audio URLs in the page are relative.  The default ``symlink`` asset mode
creates opaque links under ``assets/`` so Python's static ``http.server`` can
serve audio that lives elsewhere on the shared filesystem.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
ROLE_ORDER = ("ver2_3", "batch33", "current")
BLIND_LETTERS = ("A", "B", "C")
VALID_BINDINGS = {"ref-bound", "src-bound", "ambiguous"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build a static, per-arm, blinded no_text listening page comparing "
            "Ver2.3, Batch-33, and the current run."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--current-run-dir", required=True, help="Current arm (or selected C2 lambda) run directory.")
    ap.add_argument(
        "--ver2-3-run-dir",
        "--ver23-run-dir",
        dest="ver2_3_run_dir",
        required=True,
        help="Ver2.3 same-protocol SeedTTS run directory.",
    )
    ap.add_argument("--batch33-run-dir", required=True, help="Batch-33 / Ver2.9.5 run directory.")
    ap.add_argument(
        "--diagnostics",
        "--diagnostics-path",
        dest="diagnostics",
        default="",
        help=(
            "004063 per-sample JSONL/summary JSON, dual-encoder cases CSV, speaker_sim CSV, "
            "or a directory containing one of them. Empty means auto-discover in the current run."
        ),
    )
    ap.add_argument(
        "--diagnostics-run",
        default="",
        help="Run name to select when a diagnostics file contains multiple runs.",
    )
    ap.add_argument("--output-dir", required=True, help="New page directory; may be a child of the port-18603 web root.")
    ap.add_argument("--repo-root", default=str(ROOT), help="Root used to resolve repository-relative audio paths.")
    ap.add_argument(
        "--profile",
        choices=("auto", "batch3436", "batch37"),
        default="auto",
        help="batch3436 selects 20 keep-first cases; batch37 selects equal dual-encoder strata.",
    )
    ap.add_argument(
        "--sample-count",
        type=int,
        default=0,
        help="0 uses 20 for batch3436 and 12 (4 per stratum) for batch37.",
    )
    ap.add_argument("--selection-seed", type=int, default=20260711, help="Deterministic case-selection seed.")
    ap.add_argument("--blind-seed", type=int, default=20260710, help="Deterministic per-case A/B/C mapping seed.")
    ap.add_argument(
        "--batch37-selection-policy",
        choices=("consensus_v2", "legacy_v1"),
        default="consensus_v2",
        help=(
            "Selection namespace for Batch-37. legacy_v1 reproduces the original C1 page exactly, "
            "including page_id/case order, so browser-local manual reviews remain valid."
        ),
    )
    ap.add_argument("--binding-margin", type=float, default=0.05, help="Ref/src similarity margin for each encoder.")
    ap.add_argument(
        "--allow-single-encoder-strata",
        action="store_true",
        help="Allow Batch-37 strata from one encoder if dual-encoder metrics are unavailable (not recommended).",
    )
    ap.add_argument(
        "--asset-mode",
        choices=("symlink", "relative"),
        default="symlink",
        help=(
            "symlink creates opaque assets inside the page; relative references original audio via ../ paths "
            "and only works when those paths are under the HTTP server's reachable tree."
        ),
    )
    ap.add_argument("--current-label", default="", help="Human-readable current arm label.")
    ap.add_argument("--page-title", default="", help="HTML page title.")
    ap.add_argument("--force", action="store_true", help="Replace this generator's index/manifest and conflicting asset links.")
    return ap.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            yield row


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def parse_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "keep"}:
        return True
    if text in {"0", "false", "no", "n", "off", "drop", "fail"}:
        return False
    return None


def pick(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return ""


def safe_name(value: str, max_len: int = 96) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    value = re.sub(r"_+", "_", value).strip("._")
    return value[:max_len] or "item"


def resolve_audio_path(value: Any, *, run_dir: Path, repo_root: Path) -> Path | None:
    if value in (None, ""):
        return None
    raw = Path(str(value)).expanduser()
    if raw.is_absolute():
        return raw.resolve(strict=False)
    candidates = [repo_root / raw, run_dir / raw, Path.cwd() / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve(strict=False)


def manifest_paths(run_dir: Path) -> list[Path]:
    paths = sorted(run_dir.glob("manifest*.jsonl"))
    # Shards are the canonical source. A rerun manifest, if present, should
    # overwrite the corresponding failed shard row.
    return sorted(paths, key=lambda p: ("rerun" in p.name, p.name))


def merged_asr_paths(run_dir: Path) -> list[Path]:
    return [path for path in sorted(run_dir.glob("*.asr_eval.jsonl")) if ".shard" not in path.name]


def index_rows(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or row.get("sample_id") or "")
        if case_id:
            out[case_id] = row
    return out


def load_run(run_dir: Path, repo_root: Path) -> dict[str, Any]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    manifests: dict[str, dict[str, Any]] = {}
    paths = manifest_paths(run_dir)
    if not paths:
        raise FileNotFoundError(f"No manifest*.jsonl under {run_dir}")
    for path in paths:
        manifests.update(index_rows(iter_jsonl(path)))

    asr_rows: dict[str, dict[str, Any]] = {}
    for path in merged_asr_paths(run_dir):
        asr_rows.update(index_rows(iter_jsonl(path)))

    sim_rows: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("*.speaker_sim.csv")):
        sim_rows.update(index_rows(read_csv(path)))

    samples: dict[str, dict[str, Any]] = {}
    run_ids: Counter[str] = Counter()
    for case_id in sorted(set(manifests) | set(asr_rows) | set(sim_rows)):
        manifest = manifests.get(case_id, {})
        asr = asr_rows.get(case_id, {})
        sim = sim_rows.get(case_id, {})
        run_id = str(pick(asr.get("run_id"), sim.get("run"), manifest.get("run_id"), run_dir.name))
        run_ids[run_id] += 1
        generated_raw = pick(manifest.get("output_wav"), asr.get("target_audio"), sim.get("target_audio"))
        generated = resolve_audio_path(generated_raw, run_dir=run_dir, repo_root=repo_root)
        if generated is None:
            generated = (run_dir / f"{case_id}.wav").resolve(strict=False)
        samples[case_id] = {
            "case_id": case_id,
            "run_id": run_id,
            "mode": str(pick(asr.get("mode"), manifest.get("mode"), sim.get("mode"))),
            "cell": str(pick(asr.get("cell"), manifest.get("cell"), sim.get("cell"))),
            "language": str(pick(asr.get("language"), asr.get("source_lang"), manifest.get("source_lang"))),
            "source_lang": str(pick(asr.get("source_lang"), manifest.get("source_lang"))),
            "ref_lang": str(pick(asr.get("ref_lang"), manifest.get("ref_lang"))),
            "generated_audio": generated,
            "source_audio": resolve_audio_path(
                pick(manifest.get("source_audio"), asr.get("source_audio"), sim.get("source_audio")),
                run_dir=run_dir,
                repo_root=repo_root,
            ),
            "reference_audio": resolve_audio_path(
                pick(manifest.get("timbre_ref_audio"), asr.get("timbre_ref_audio"), sim.get("timbre_ref_audio")),
                run_dir=run_dir,
                repo_root=repo_root,
            ),
            "content_text": str(
                pick(
                    asr.get("content_ref_text"),
                    asr.get("target_text"),
                    manifest.get("content_ref_text"),
                    manifest.get("source_content_text"),
                )
            ),
            "source_text": str(pick(asr.get("source_text"), asr.get("asr_src_text"), manifest.get("source_content_text"))),
            "reference_text": str(pick(asr.get("timbre_ref_text"), manifest.get("timbre_ref_text"))),
            "asr_text": str(asr.get("asr_tgt_text") or ""),
            "content_keep": parse_bool(asr.get("content_keep")),
            "content_filter_reason": str(asr.get("content_filter_reason") or ""),
            "cer_tgt": finite(asr.get("cer_tgt")),
            "wer_tgt": finite(asr.get("wer_tgt")),
            "manifest_status": str(pick(manifest.get("status"), asr.get("manifest_status"))),
        }

    run_id = run_ids.most_common(1)[0][0] if run_ids else run_dir.name
    return {
        "run_dir": run_dir,
        "run_id": run_id,
        "samples": samples,
        "manifest_paths": paths,
        "asr_paths": merged_asr_paths(run_dir),
    }


def load_rows_from_diagnostics_file(path: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    suffixes = path.suffixes
    if path.suffix.lower() == ".csv":
        return read_csv(path), [path]
    if path.suffix.lower() == ".jsonl":
        return list(iter_jsonl(path)), [path]
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("per_sample_jsonl"):
            target = Path(str(payload["per_sample_jsonl"])).expanduser()
            if not target.is_absolute():
                target = path.parent / target
            if not target.is_file():
                # 004063 often records an absolute path, but moving the
                # aggregate directory should still work by basename.
                basename_target = path.parent / Path(str(payload["per_sample_jsonl"])).name
                if basename_target.is_file():
                    target = basename_target
            return list(iter_jsonl(target)), [path, target]
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            rows = [row for row in payload["rows"] if isinstance(row, dict)]
            return rows, [path]
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)], [path]
        raise ValueError(f"Unsupported diagnostics JSON payload: {path}")
    raise ValueError(f"Unsupported diagnostics extension {suffixes}: {path}")


def diagnostics_candidates(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Diagnostics path does not exist: {path}")
    patterns = ("*.per_sample.jsonl", "*dual_encoder_cases.csv", "*.speaker_sim.csv", "*.summary.json")
    found: list[Path] = []
    for pattern in patterns:
        found.extend(sorted(path.rglob(pattern)))
    # Keep preference order and remove duplicates.
    return list(dict.fromkeys(found))


def choose_diagnostics_rows(
    rows: list[dict[str, Any]],
    *,
    desired_run: str,
    explicit_run: bool,
) -> list[dict[str, Any]]:
    run_values = sorted({str(row.get("run") or row.get("run_id") or "") for row in rows if row.get("run") or row.get("run_id")})
    if not run_values:
        return rows
    exact = [row for row in rows if str(row.get("run") or row.get("run_id") or "") == desired_run]
    if exact:
        return exact
    if explicit_run:
        return []
    if len(run_values) == 1:
        return rows
    return []


def load_diagnostics(
    diagnostics_arg: str,
    *,
    current_run: dict[str, Any],
    diagnostics_run: str,
) -> tuple[dict[str, dict[str, Any]], list[str], str]:
    desired_run = diagnostics_run or str(current_run["run_id"])
    explicit_run = bool(diagnostics_run)
    if diagnostics_arg:
        root = Path(diagnostics_arg).expanduser().resolve()
        candidates = diagnostics_candidates(root)
    else:
        run_dir = Path(current_run["run_dir"])
        candidates = sorted(run_dir.glob("*.per_sample.jsonl"))
        candidates += sorted(run_dir.glob("*dual_encoder_cases.csv"))
        candidates += sorted(run_dir.glob("*.speaker_sim.csv"))

    errors: list[str] = []
    for candidate in candidates:
        try:
            rows, sources = load_rows_from_diagnostics_file(candidate)
        except Exception as exc:  # Continue to a lower-priority compatible file.
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
            continue
        selected = choose_diagnostics_rows(rows, desired_run=desired_run, explicit_run=explicit_run)
        selected = [row for row in selected if str(row.get("mode") or "no_text") == "no_text"]
        indexed = index_rows(selected)
        if indexed:
            return indexed, [str(path.resolve()) for path in sources], desired_run
        errors.append(f"{candidate}: no rows for run={desired_run!r}")

    if diagnostics_arg:
        details = "\n  ".join(errors) if errors else "no compatible files found"
        raise ValueError(f"Could not load diagnostics for run={desired_run!r}:\n  {details}")
    return {}, [], desired_run


def classify_binding(sim_ref: float | None, sim_src: float | None, margin: float) -> str:
    if sim_ref is None or sim_src is None:
        return "missing"
    delta = sim_ref - sim_src
    if delta > margin:
        return "ref-bound"
    if delta < -margin:
        return "src-bound"
    return "ambiguous"


def normalize_diagnostics(row: dict[str, Any], margin: float) -> dict[str, Any]:
    wavlm_ref = finite(row.get("sim_gen_ref"))
    wavlm_src = finite(row.get("sim_gen_source"))
    ecapa_ref = finite(row.get("ecapa_sim_gen_ref"))
    ecapa_src = finite(row.get("ecapa_sim_gen_source"))
    wavlm_binding = str(row.get("wavlm_binding") or classify_binding(wavlm_ref, wavlm_src, margin))
    ecapa_binding = str(row.get("ecapa_binding") or classify_binding(ecapa_ref, ecapa_src, margin))
    if wavlm_binding not in VALID_BINDINGS:
        wavlm_binding = classify_binding(wavlm_ref, wavlm_src, margin)
    if ecapa_binding not in VALID_BINDINGS:
        ecapa_binding = classify_binding(ecapa_ref, ecapa_src, margin)

    available = [binding for binding in (wavlm_binding, ecapa_binding) if binding != "missing"]
    if len(available) == 2:
        dual_stratum = available[0] if available[0] == available[1] else "ambiguous"
    elif len(available) == 1:
        dual_stratum = available[0]
    else:
        dual_stratum = "missing"
    return {
        "wavlm_sim_ref": wavlm_ref,
        "wavlm_sim_src": wavlm_src,
        "wavlm_delta_ref_minus_src": wavlm_ref - wavlm_src if wavlm_ref is not None and wavlm_src is not None else None,
        "wavlm_binding": wavlm_binding,
        "speechbrain_ecapa_sim_ref": ecapa_ref,
        "speechbrain_ecapa_sim_src": ecapa_src,
        "speechbrain_ecapa_delta_ref_minus_src": (
            ecapa_ref - ecapa_src if ecapa_ref is not None and ecapa_src is not None else None
        ),
        "speechbrain_ecapa_binding": ecapa_binding,
        "dual_encoder_available": len(available) == 2,
        "dual_encoder_stratum": dual_stratum,
        "content_keep": parse_bool(row.get("content_keep")),
    }


def select_anchor(samples: list[dict[str, Any]], key: str) -> Path | None:
    for sample in samples:
        path = sample.get(key)
        if isinstance(path, Path) and path.exists():
            return path
    return None


def build_eligible_cases(
    current: dict[str, Any],
    ver2_3: dict[str, Any],
    batch33: dict[str, Any],
    diagnostics: dict[str, dict[str, Any]],
    *,
    margin: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    eligible: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    current_samples = current["samples"]
    ver2_3_samples = ver2_3["samples"]
    batch33_samples = batch33["samples"]

    for case_id, current_sample in current_samples.items():
        if current_sample.get("mode") != "no_text":
            skipped["not_no_text"] += 1
            continue
        if case_id not in ver2_3_samples:
            skipped["missing_ver2_3_case"] += 1
            continue
        if case_id not in batch33_samples:
            skipped["missing_batch33_case"] += 1
            continue
        v23_sample = ver2_3_samples[case_id]
        b33_sample = batch33_samples[case_id]
        if v23_sample.get("mode") not in ("", "no_text") or b33_sample.get("mode") not in ("", "no_text"):
            skipped["baseline_mode_mismatch"] += 1
            continue

        generated = {
            "ver2_3": v23_sample.get("generated_audio"),
            "batch33": b33_sample.get("generated_audio"),
            "current": current_sample.get("generated_audio"),
        }
        missing_generated = [role for role, path in generated.items() if not isinstance(path, Path) or not path.exists()]
        if missing_generated:
            skipped["missing_generated_" + "_".join(missing_generated)] += 1
            continue

        source = select_anchor([current_sample, b33_sample, v23_sample], "source_audio")
        reference = select_anchor([current_sample, b33_sample, v23_sample], "reference_audio")
        if source is None:
            skipped["missing_source_anchor"] += 1
            continue
        if reference is None:
            skipped["missing_reference_anchor"] += 1
            continue

        diag_raw = diagnostics.get(case_id, {})
        diag = normalize_diagnostics(diag_raw, margin)
        content_keep = diag.get("content_keep")
        if content_keep is None:
            content_keep = current_sample.get("content_keep")
        eligible.append(
            {
                "case_id": case_id,
                "cell": str(pick(current_sample.get("cell"), b33_sample.get("cell"), v23_sample.get("cell"))),
                "language": str(pick(current_sample.get("language"), current_sample.get("source_lang"))),
                "source_lang": str(current_sample.get("source_lang") or ""),
                "ref_lang": str(current_sample.get("ref_lang") or ""),
                "content_text": str(pick(current_sample.get("content_text"), b33_sample.get("content_text"), v23_sample.get("content_text"))),
                "source_text": str(pick(current_sample.get("source_text"), b33_sample.get("source_text"), v23_sample.get("source_text"))),
                "reference_text": str(pick(current_sample.get("reference_text"), b33_sample.get("reference_text"), v23_sample.get("reference_text"))),
                "content_keep": content_keep,
                "content_filter_reason": str(current_sample.get("content_filter_reason") or ""),
                "current_asr_text": str(current_sample.get("asr_text") or ""),
                "current_cer": current_sample.get("cer_tgt"),
                "current_wer": current_sample.get("wer_tgt"),
                "source_audio": source,
                "reference_audio": reference,
                "generated": generated,
                "diagnostics": diag,
                "diagnostics_present": bool(diag_raw),
            }
        )
    return eligible, skipped


def diverse_pick(rows: list[dict[str, Any]], count: int, *, seed: int, salt: str) -> list[dict[str, Any]]:
    if count <= 0:
        return []

    def round_robin(pool: list[dict[str, Any]], wanted: int, pool_salt: str) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in pool:
            grouped[str(row.get("cell") or "unknown")].append(row)
        rng = random.Random(f"{seed}:{salt}:{pool_salt}")
        keys = sorted(grouped)
        rng.shuffle(keys)
        for key in keys:
            grouped[key].sort(key=lambda item: item["case_id"])
            rng.shuffle(grouped[key])
        picked: list[dict[str, Any]] = []
        while keys and len(picked) < wanted:
            next_keys: list[str] = []
            for key in keys:
                if grouped[key] and len(picked) < wanted:
                    picked.append(grouped[key].pop())
                if grouped[key]:
                    next_keys.append(key)
            keys = next_keys
        return picked

    keep = [row for row in rows if row.get("content_keep") is True]
    other = [row for row in rows if row.get("content_keep") is not True]
    selected = round_robin(keep, min(count, len(keep)), "keep")
    selected_ids = {row["case_id"] for row in selected}
    if len(selected) < count:
        remaining = [row for row in other if row["case_id"] not in selected_ids]
        selected.extend(round_robin(remaining, count - len(selected), "fallback"))
    return selected


def select_cases(
    eligible: list[dict[str, Any]],
    *,
    profile: str,
    sample_count: int,
    seed: int,
    allow_single_encoder: bool,
    batch37_selection_policy: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if profile == "batch3436":
        if len(eligible) < sample_count:
            raise ValueError(f"Need {sample_count} eligible no_text cases, found {len(eligible)}")
        selected = diverse_pick(eligible, sample_count, seed=seed, salt="batch3436")
        return selected, {
            "strategy": "content_keep_first_cell_round_robin",
            "requested": sample_count,
            "selected": len(selected),
            "selected_keep": sum(row.get("content_keep") is True for row in selected),
        }

    if sample_count % 3:
        raise ValueError("Batch-37 sample count must be divisible by 3 for equal ref/src/ambiguous strata")
    per_stratum = sample_count // 3
    strata: dict[str, list[dict[str, Any]]] = {name: [] for name in VALID_BINDINGS}
    excluded_missing_dual = 0
    for row in eligible:
        diag = row["diagnostics"]
        if not diag.get("dual_encoder_available") and not allow_single_encoder:
            excluded_missing_dual += 1
            continue
        stratum = str(diag.get("dual_encoder_stratum") or "missing")
        if stratum in strata:
            strata[stratum].append(row)

    counts = {name: len(rows) for name, rows in strata.items()}
    short = {name: count for name, count in counts.items() if count < per_stratum}
    if short and not allow_single_encoder:
        raise ValueError(
            "Insufficient Batch-37 dual-encoder strata: "
            f"need {per_stratum} each, counts={counts}, missing_dual_excluded={excluded_missing_dual}. "
            "Use --allow-single-encoder-strata only for an explicitly non-canonical fallback."
        )

    selected: list[dict[str, Any]] = []
    selected_by_stratum: dict[str, int] = {}
    fallback_selected: dict[str, int] = {name: 0 for name in VALID_BINDINGS}
    used_ids: set[str] = set()

    def tagged(items: list[dict[str, Any]], *, stratum: str, source: str) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for item in items:
            copy = dict(item)
            copy["selection_stratum"] = stratum
            copy["selection_stratum_source"] = source
            output.append(copy)
        return output

    # Preserve strict consensus whenever possible.  If a CFG sweep removes
    # almost every consensus src-bound sample, the explicit fallback uses the
    # primary WavLM-SV classification first (the source-leakage-sensitive
    # encoder), then SpeechBrain, while recording the provenance in manifest.
    legacy_selection = batch37_selection_policy == "legacy_v1"
    selection_order = (
        ("ref-bound", "src-bound", "ambiguous")
        if legacy_selection
        else ("src-bound", "ref-bound", "ambiguous")
    )
    for stratum in selection_order:
        strict_pool = [row for row in strata[stratum] if row["case_id"] not in used_ids]
        strict_pick = diverse_pick(
            strict_pool,
            min(per_stratum, len(strict_pool)),
            seed=seed,
            salt=f"batch37:{stratum}" if legacy_selection else f"batch37:{stratum}:consensus",
        )
        chosen = tagged(strict_pick, stratum=stratum, source="dual_encoder_consensus")
        used_ids.update(row["case_id"] for row in strict_pick)

        missing = per_stratum - len(chosen)
        if missing > 0 and allow_single_encoder:
            primary = [
                row
                for row in eligible
                if row["case_id"] not in used_ids and row["diagnostics"].get("wavlm_binding") == stratum
            ]
            secondary = [
                row
                for row in eligible
                if row["case_id"] not in used_ids
                and row not in primary
                and row["diagnostics"].get("speechbrain_ecapa_binding") == stratum
            ]
            primary_pick = diverse_pick(
                primary,
                min(missing, len(primary)),
                seed=seed,
                salt=f"batch37:{stratum}:wavlm-fallback",
            )
            chosen.extend(tagged(primary_pick, stratum=stratum, source="wavlm_single_encoder_fallback"))
            used_ids.update(row["case_id"] for row in primary_pick)
            fallback_selected[stratum] += len(primary_pick)
            missing = per_stratum - len(chosen)
            if missing > 0:
                secondary = [row for row in secondary if row["case_id"] not in used_ids]
                secondary_pick = diverse_pick(
                    secondary,
                    min(missing, len(secondary)),
                    seed=seed,
                    salt=f"batch37:{stratum}:ecapa-fallback",
                )
                chosen.extend(
                    tagged(secondary_pick, stratum=stratum, source="speechbrain_single_encoder_fallback")
                )
                used_ids.update(row["case_id"] for row in secondary_pick)
                fallback_selected[stratum] += len(secondary_pick)

        if len(chosen) < per_stratum:
            raise ValueError(
                "Insufficient Batch-37 stratum even after the explicit single-encoder fallback: "
                f"stratum={stratum} selected={len(chosen)} need={per_stratum} strict_counts={counts}"
            )
        selected.extend(chosen)
        selected_by_stratum[stratum] = len(chosen)
    random.Random(f"{seed}:batch37:page-order").shuffle(selected)
    return selected, {
        "strategy": (
            "dual_encoder_consensus_with_explicit_single_encoder_shortfall_fallback"
            if any(fallback_selected.values())
            else "dual_encoder_consensus_equal_strata_content_keep_first_cell_round_robin"
        ),
        "dual_encoder_consensus": "ref/src only when both encoders agree; all disagreements are ambiguous",
        "single_encoder_fallback_priority": "WavLM-SV first, then SpeechBrain ECAPA",
        "requested": sample_count,
        "per_stratum": per_stratum,
        "eligible_by_stratum": counts,
        "selected_by_stratum": selected_by_stratum,
        "fallback_selected_by_stratum": fallback_selected,
        "selected_keep": sum(row.get("content_keep") is True for row in selected),
        "missing_dual_excluded": excluded_missing_dual,
        "allow_single_encoder_strata": allow_single_encoder,
        "batch37_selection_policy": batch37_selection_policy,
    }


def relative_href(path: Path, output_dir: Path) -> str:
    return Path(os.path.relpath(path, output_dir)).as_posix()


def stage_audio(
    source: Path,
    *,
    destination: Path,
    output_dir: Path,
    asset_mode: str,
    force: bool,
) -> str:
    if not source.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {source}")
    if asset_mode == "relative":
        return relative_href(source, output_dir)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        same = False
        if destination.is_symlink():
            try:
                target = Path(os.readlink(destination))
                if not target.is_absolute():
                    target = destination.parent / target
                same = target.resolve(strict=False) == source.resolve(strict=False)
            except OSError:
                same = False
        if not same:
            if not force:
                raise FileExistsError(f"Conflicting asset exists (use --force): {destination}")
            destination.unlink()
    if not os.path.lexists(destination):
        os.symlink(str(source.resolve()), str(destination))
    return destination.relative_to(output_dir).as_posix()


def audio_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    return suffix if suffix in {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"} else ".wav"


def blind_mapping(case_id: str, seed: int) -> dict[str, str]:
    roles = list(ROLE_ORDER)
    random.Random(f"{seed}:{case_id}").shuffle(roles)
    return dict(zip(BLIND_LETTERS, roles))


def build_outputs(
    selected: list[dict[str, Any]],
    *,
    output_dir: Path,
    asset_mode: str,
    force: bool,
    blind_seed: int,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    assets_dir = output_dir / "assets"
    display_cases: list[dict[str, Any]] = []
    manifest_cases: list[dict[str, Any]] = []

    for index, row in enumerate(selected, start=1):
        prefix = f"case_{index:03d}"
        source = row["source_audio"]
        reference = row["reference_audio"]
        source_href = stage_audio(
            source,
            destination=assets_dir / f"{prefix}_anchor_source{audio_suffix(source)}",
            output_dir=output_dir,
            asset_mode=asset_mode,
            force=force,
        )
        reference_href = stage_audio(
            reference,
            destination=assets_dir / f"{prefix}_anchor_reference{audio_suffix(reference)}",
            output_dir=output_dir,
            asset_mode=asset_mode,
            force=force,
        )

        # Namespace the shuffle by page so revealing one arm cannot disclose
        # the A/B/C identities for the same case on another arm's page.
        mapping = blind_mapping(f"{metadata['page_id']}:{row['case_id']}", blind_seed)
        candidate_display: list[dict[str, str]] = []
        candidate_manifest: dict[str, dict[str, Any]] = {}
        for letter in BLIND_LETTERS:
            role = mapping[letter]
            source_path = row["generated"][role]
            href = stage_audio(
                source_path,
                destination=assets_dir / f"{prefix}_candidate_{letter}{audio_suffix(source_path)}",
                output_dir=output_dir,
                asset_mode=asset_mode,
                force=force,
            )
            candidate_display.append({"letter": letter, "audio": href})
            candidate_manifest[letter] = {
                "role": role,
                "label": metadata["roles"][role]["label"],
                "run_dir": metadata["roles"][role]["run_dir"],
                "source_path": str(source_path),
                "href": href,
            }

        display_cases.append(
            {
                "index": index,
                "case_id": row["case_id"],
                "cell": row["cell"],
                "language": row["language"],
                "source_lang": row["source_lang"],
                "ref_lang": row["ref_lang"],
                "content_text": row["content_text"],
                "reference_text": row["reference_text"],
                "source_audio": source_href,
                "reference_audio": reference_href,
                "candidates": candidate_display,
            }
        )
        manifest_cases.append(
            {
                "index": index,
                "case_id": row["case_id"],
                "mode": "no_text",
                "cell": row["cell"],
                "language": row["language"],
                "source_lang": row["source_lang"],
                "ref_lang": row["ref_lang"],
                "content_keep": row["content_keep"],
                "content_filter_reason": row["content_filter_reason"],
                "content_text": row["content_text"],
                "source_text": row["source_text"],
                "reference_text": row["reference_text"],
                "current_asr_text": row["current_asr_text"],
                "current_cer": row["current_cer"],
                "current_wer": row["current_wer"],
                "diagnostics_present": row["diagnostics_present"],
                "diagnostics": row["diagnostics"],
                "selection_stratum": row.get(
                    "selection_stratum", row["diagnostics"].get("dual_encoder_stratum")
                ),
                "selection_stratum_source": row.get("selection_stratum_source", "dual_encoder_consensus"),
                "anchors": {
                    "source": {"source_path": str(source), "href": source_href},
                    "reference": {"source_path": str(reference), "href": reference_href},
                },
                "candidate_mapping": candidate_manifest,
            }
        )

    display = {
        "schema_version": 1,
        "page_id": metadata["page_id"],
        "title": metadata["title"],
        "current_label": metadata["roles"]["current"]["label"],
        "cases": display_cases,
    }
    manifest = {
        **metadata,
        "mapping_visibility": "manifest_only_not_embedded_in_index_html",
        "response_semantics": {
            "current_better": "current 比 Batch-33 更像 ref",
            "same": "current 与 Batch-33 差不多",
            "current_worse": "current 比 Batch-33 更不像 ref",
        },
        "cases": manifest_cases,
    }
    return display, manifest


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE_HTML__</title>
  <style>
    :root { --bg:#f4f6f8; --panel:#fff; --line:#d8dee8; --ink:#172033; --muted:#68758b; --accent:#175cd3; --ok:#067647; --warn:#b54708; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI","PingFang SC",sans-serif; line-height:1.45; }
    .top { position:sticky; top:0; z-index:5; padding:16px 22px; background:rgba(244,246,248,.96); border-bottom:1px solid var(--line); backdrop-filter:blur(8px); }
    h1 { margin:0 0 5px; font-size:21px; }
    .sub { color:var(--muted); font-size:13px; }
    .toolbar { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:12px; }
    button { border:1px solid var(--line); border-radius:7px; background:#fff; color:var(--ink); padding:8px 11px; cursor:pointer; font-weight:650; }
    button:hover { border-color:#98a2b3; }
    button:disabled { opacity:.48; cursor:not-allowed; }
    button.active { border-color:var(--accent); background:#eff4ff; color:var(--accent); }
    button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
    .progress { font-weight:750; color:var(--ok); }
    main { max-width:1500px; margin:0 auto; padding:18px 22px 50px; }
    .case { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:15px; margin-bottom:15px; box-shadow:0 1px 2px rgba(16,24,40,.05); }
    .case-head { display:flex; justify-content:space-between; gap:12px; align-items:flex-start; }
    .case-title { font-size:15px; font-weight:800; overflow-wrap:anywhere; }
    .tag { color:var(--muted); font-size:12px; }
    .audio-grid { display:grid; grid-template-columns:repeat(5,minmax(170px,1fr)); gap:10px; margin-top:12px; }
    .audio-card { border:1px solid var(--line); border-radius:8px; padding:9px; min-width:0; background:#fbfcfe; }
    .audio-card.anchor { background:#f0f9ff; border-color:#b9e6fe; }
    .audio-card.candidate { background:#fffcf5; border-color:#fedf89; }
    .audio-label { font-size:12px; font-weight:800; margin-bottom:7px; color:var(--accent); }
    .candidate .audio-label { color:var(--warn); font-size:14px; }
    audio { width:100%; height:36px; }
    .text-grid { display:grid; grid-template-columns:2fr 1fr; gap:10px; margin-top:10px; }
    .text-box { border-top:1px solid var(--line); padding-top:8px; color:#344054; font-size:13px; }
    .text-box b { display:block; color:var(--muted); font-size:11px; text-transform:uppercase; margin-bottom:3px; }
    .review { margin-top:12px; border-top:1px solid var(--line); padding-top:12px; display:grid; gap:10px; }
    .review-row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
    .review-label { min-width:190px; font-size:13px; font-weight:750; }
    .reveal-map { width:100%; padding:9px 10px; border-radius:7px; background:#f8f9fc; border:1px dashed #98a2b3; color:#344054; font-size:13px; }
    textarea { width:100%; min-height:54px; resize:vertical; border:1px solid var(--line); border-radius:7px; padding:8px; font:inherit; }
    .unanswered { border-left:4px solid #f79009; }
    .answered { border-left:4px solid #12b76a; }
    .footnote { margin-top:14px; color:var(--muted); font-size:12px; }
    @media(max-width:1100px) { .audio-grid { grid-template-columns:repeat(2,minmax(180px,1fr)); } .text-grid { grid-template-columns:1fr; } }
    @media(max-width:620px) { main,.top { padding-left:11px; padding-right:11px; } .audio-grid { grid-template-columns:1fr; } .review-label { min-width:100%; } }
  </style>
</head>
<body>
  <header class="top">
    <h1>__TITLE_HTML__</h1>
    <div class="sub">仅 no_text，同一 case 对齐。请先盲选 A/B/C；盲选锁定后才可逐条解盲并填写 current vs Batch-33 结论。映射不嵌入本页面。</div>
    <div class="toolbar">
      <span id="progress" class="progress"></span>
      <button id="export" class="primary">导出 JSON</button>
      <button id="clear">清空本页选择</button>
    </div>
  </header>
  <main>
    <div id="cases"></div>
    <div class="footnote">选择会自动保存在当前浏览器 localStorage。导出的 case_id 可与同目录 manifest.json 离线合并解盲。</div>
  </main>
  <script>
  const PAGE = __DISPLAY_JSON__;
  const STORAGE_KEY = `ver23-subjective:${PAGE.page_id}`;
  const judgmentLabels = {
    current_better: "current 比 Batch-33 更像 ref",
    same: "current 与 Batch-33 差不多",
    current_worse: "current 比 Batch-33 更不像 ref"
  };
  let state = loadState();
  let revealMap = null;
  function esc(x) { return String(x ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }
  function loadState() { try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); } catch (_) { return {}; } }
  function saveState() { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); updateProgress(); }
  function answer(caseId) { return state[caseId] || {judgment:"", blind_winner:"", note:"", revealed:false, revealed_at:""}; }
  function audioCard(label, src, cls) { return `<div class="audio-card ${cls}"><div class="audio-label">${esc(label)}</div><audio controls preload="none" src="${esc(src)}"></audio></div>`; }
  function choiceButton(caseId, field, value, label, disabled=false) { const active = answer(caseId)[field] === value ? "active" : ""; return `<button class="${active}" data-case="${esc(caseId)}" data-field="${field}" data-value="${value}" ${disabled ? "disabled" : ""}>${esc(label)}</button>`; }
  async function ensureManifest() {
    if (revealMap) return revealMap;
    const response = await fetch("manifest.json", {cache:"no-store"});
    if (!response.ok) throw new Error(`manifest HTTP ${response.status}`);
    const payload = await response.json();
    revealMap = Object.fromEntries((payload.cases || []).map(x => [x.case_id, x.candidate_mapping || {}]));
    return revealMap;
  }
  function mappedLabel(caseId, letter) {
    const item = revealMap && revealMap[caseId] && revealMap[caseId][letter];
    return item ? String(item.label || item.role || letter) : letter;
  }
  async function revealCase(caseId) {
    const old = answer(caseId);
    if (!old.blind_winner) { alert("请先完成本条 A/B/C 盲选。"); return; }
    try { await ensureManifest(); }
    catch (err) { alert(`读取解盲映射失败：${err}`); return; }
    state[caseId] = {...old, blind_winner_locked:old.blind_winner, revealed:true, revealed_at:new Date().toISOString()};
    saveState(); render();
  }
  function renderCase(s) {
    const a = answer(s.case_id);
    const revealed = Boolean(a.revealed && revealMap && revealMap[s.case_id]);
    const candidates = s.candidates.map(x => audioCard(revealed ? `候选 ${x.letter} · ${mappedLabel(s.case_id,x.letter)}` : `候选 ${x.letter}`, x.audio, "candidate")).join("");
    const revealDetails = revealed
      ? `<div class="reveal-map">已锁定盲选：${esc(a.blind_winner_locked || a.blind_winner)}。映射：A=${esc(mappedLabel(s.case_id,"A"))}；B=${esc(mappedLabel(s.case_id,"B"))}；C=${esc(mappedLabel(s.case_id,"C"))}</div>`
      : `<button data-reveal="${esc(s.case_id)}" ${a.blind_winner ? "" : "disabled"}>锁定盲选并显示身份</button>`;
    const judgmentRow = revealed ? `<div class="review-row"><span class="review-label">核心结论：current vs Batch-33</span>
          ${choiceButton(s.case_id,"judgment","current_better",judgmentLabels.current_better)}
          ${choiceButton(s.case_id,"judgment","same",judgmentLabels.same)}
          ${choiceButton(s.case_id,"judgment","current_worse",judgmentLabels.current_worse)}
        </div>` : "";
    const langs = [s.source_lang, s.ref_lang].filter(Boolean).join("→");
    return `<article class="case ${a.judgment ? "answered" : "unanswered"}" id="case-${s.index}">
      <div class="case-head"><div><div class="tag">Case ${String(s.index).padStart(2,"0")} · ${esc(s.cell)} · ${esc(langs || s.language)}</div><div class="case-title">${esc(s.case_id)}</div></div></div>
      <div class="audio-grid">
        ${audioCard("Source 锚点", s.source_audio, "anchor")}
        ${audioCard("Reference 音色锚点", s.reference_audio, "anchor")}
        ${candidates}
      </div>
      <div class="text-grid">
        <div class="text-box"><b>目标内容</b>${esc(s.content_text || "")}</div>
        <div class="text-box"><b>Reference 语音文本</b>${esc(s.reference_text || "")}</div>
      </div>
      <div class="review">
        <div class="review-row"><span class="review-label">盲听：哪个候选最像 Reference？</span>
          ${choiceButton(s.case_id,"blind_winner","A","A",revealed)}${choiceButton(s.case_id,"blind_winner","B","B",revealed)}${choiceButton(s.case_id,"blind_winner","C","C",revealed)}${choiceButton(s.case_id,"blind_winner","tie","难分",revealed)}
        </div>
        <div class="review-row"><span class="review-label">两阶段解盲</span>${revealDetails}</div>
        ${judgmentRow}
        <textarea data-note="${esc(s.case_id)}" placeholder="可选备注">${esc(a.note || "")}</textarea>
      </div>
    </article>`;
  }
  function render() {
    document.getElementById("cases").innerHTML = PAGE.cases.map(renderCase).join("");
    document.querySelectorAll("button[data-case]:not([disabled])").forEach(btn => btn.onclick = () => {
      const old = answer(btn.dataset.case);
      state[btn.dataset.case] = {...old, [btn.dataset.field]: btn.dataset.value};
      saveState(); render();
    });
    document.querySelectorAll("button[data-reveal]:not([disabled])").forEach(btn => btn.onclick = () => revealCase(btn.dataset.reveal));
    document.querySelectorAll("textarea[data-note]").forEach(box => box.onchange = () => {
      const old = answer(box.dataset.note);
      state[box.dataset.note] = {...old, note: box.value};
      saveState();
    });
    document.querySelectorAll("audio").forEach(player => player.addEventListener("play", () => {
      document.querySelectorAll("audio").forEach(other => { if (other !== player) other.pause(); });
    }));
    updateProgress();
  }
  function updateProgress() {
    const done = PAGE.cases.filter(s => answer(s.case_id).judgment).length;
    document.getElementById("progress").textContent = `已完成 ${done}/${PAGE.cases.length}`;
  }
  function exportPayload() {
    return {
      schema_version: 1,
      page_id: PAGE.page_id,
      current_label: PAGE.current_label,
      exported_at: new Date().toISOString(),
      items: PAGE.cases.map(s => {
        const a = answer(s.case_id);
        return {case_id:s.case_id, judgment:a.judgment || "", judgment_label:judgmentLabels[a.judgment] || "", blind_winner:a.blind_winner || "", blind_winner_locked:a.blind_winner_locked || "", revealed:Boolean(a.revealed), revealed_at:a.revealed_at || "", note:a.note || ""};
      })
    };
  }
  function downloadJSON() {
    const blob = new Blob([JSON.stringify(exportPayload(), null, 2)], {type:"application/json"});
    const url = URL.createObjectURL(blob); const a = document.createElement("a");
    a.href = url; a.download = `${PAGE.page_id}.review.json`; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
  }
  document.getElementById("export").onclick = downloadJSON;
  document.getElementById("clear").onclick = () => { if (confirm("确认清空本页全部选择？")) { state = {}; localStorage.removeItem(STORAGE_KEY); render(); } };
  async function bootstrap() {
    if (Object.values(state).some(x => x && x.revealed)) {
      try { await ensureManifest(); } catch (_) { /* keep the page usable for fresh blind choices */ }
    }
    render();
  }
  bootstrap();
  </script>
</body>
</html>
'''


def render_html(display: dict[str, Any]) -> str:
    data = json.dumps(display, ensure_ascii=False, separators=(",", ":"))
    # Do not let user-controlled text terminate the script element.
    data = data.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return HTML_TEMPLATE.replace("__TITLE_HTML__", html.escape(display["title"])).replace("__DISPLAY_JSON__", data)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    current_dir = Path(args.current_run_dir).expanduser().resolve()
    ver2_3_dir = Path(args.ver2_3_run_dir).expanduser().resolve()
    batch33_dir = Path(args.batch33_run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.sample_count < 0:
        raise ValueError("--sample-count must be >= 0")
    if args.binding_margin < 0:
        raise ValueError("--binding-margin must be >= 0")

    current = load_run(current_dir, repo_root)
    ver2_3 = load_run(ver2_3_dir, repo_root)
    batch33 = load_run(batch33_dir, repo_root)
    profile = args.profile
    if profile == "auto":
        hint = f"{current_dir.name} {args.current_label}".lower()
        profile = "batch37" if any(token in hint for token in ("batch37", "compact", "refcfg", "true_ref_audio_cfg")) else "batch3436"
    sample_count = args.sample_count or (12 if profile == "batch37" else 20)
    if sample_count <= 0:
        raise ValueError("Resolved sample count must be positive")

    diagnostics, diagnostics_sources, diagnostics_run = load_diagnostics(
        args.diagnostics,
        current_run=current,
        diagnostics_run=args.diagnostics_run,
    )
    if profile == "batch37" and not diagnostics:
        raise ValueError("Batch-37 canonical stratification requires 004063 or dual-encoder diagnostics")

    eligible, skipped = build_eligible_cases(
        current,
        ver2_3,
        batch33,
        diagnostics,
        margin=args.binding_margin,
    )
    selected, selection_summary = select_cases(
        eligible,
        profile=profile,
        sample_count=sample_count,
        seed=args.selection_seed,
        allow_single_encoder=args.allow_single_encoder_strata,
        batch37_selection_policy=args.batch37_selection_policy,
    )

    current_label = args.current_label or current["run_id"] or current_dir.name
    title = args.page_title or f"{current_label} · no_text 主观音色盲听"
    page_hash = hashlib.sha256(
        "\n".join(
            [
                str(current_dir),
                str(ver2_3_dir),
                str(batch33_dir),
                str(args.selection_seed),
                str(args.blind_seed),
                *[row["case_id"] for row in selected],
            ]
        ).encode("utf-8")
    ).hexdigest()[:12]
    page_id = f"ver23_subjective_{safe_name(current_label, 60)}_{page_hash}"

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.html"
    manifest_path = output_dir / "manifest.json"
    for path in (index_path, manifest_path):
        if path.exists() and not args.force:
            raise FileExistsError(f"Output exists (use --force): {path}")

    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "page_id": page_id,
        "title": title,
        "profile": profile,
        "mode": "no_text",
        "asset_mode": args.asset_mode,
        "selection_seed": args.selection_seed,
        "blind_seed": args.blind_seed,
        "binding_margin": args.binding_margin,
        "diagnostics_run": diagnostics_run,
        "diagnostics_sources": diagnostics_sources,
        "selection": selection_summary,
        "eligibility": {
            "eligible": len(eligible),
            "skipped": dict(sorted(skipped.items())),
        },
        "roles": {
            "ver2_3": {"label": "ver2.3", "run_id": ver2_3["run_id"], "run_dir": str(ver2_3_dir)},
            "batch33": {"label": "Batch-33", "run_id": batch33["run_id"], "run_dir": str(batch33_dir)},
            "current": {"label": current_label, "run_id": current["run_id"], "run_dir": str(current_dir)},
        },
    }
    display, manifest = build_outputs(
        selected,
        output_dir=output_dir,
        asset_mode=args.asset_mode,
        force=args.force,
        blind_seed=args.blind_seed,
        metadata=metadata,
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    index_path.write_text(render_html(display), encoding="utf-8")

    print(f"[listening-page] wrote {index_path}")
    print(f"[listening-page] manifest {manifest_path}")
    print(f"[listening-page] profile={profile} selected={len(selected)} eligible={len(eligible)}")
    print(f"[listening-page] selection={json.dumps(selection_summary, ensure_ascii=False, sort_keys=True)}")
    print(f"[listening-page] diagnostics={diagnostics_sources or ['none; current ASR content_keep only']}")
    print(f"[listening-page] skipped={json.dumps(dict(sorted(skipped.items())), ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

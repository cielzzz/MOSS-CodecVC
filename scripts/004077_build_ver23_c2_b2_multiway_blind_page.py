#!/usr/bin/env python3
"""Build a five-candidate, never-revealed, stratified no_text listening page.

The page contains ten content-keep cases selected from the C2 lambda=1.6
dual-encoder binding diagnostics: three ref-bound, three src-bound, and four
ambiguous.  Strict dual-encoder strata are preferred.  If the strict stratum
cannot satisfy the requested count without a content failure, a content-keep
WavLM-only fallback is used and recorded in ``manifest.json``.

The browser-facing HTML contains only Source, Reference, and anonymous
candidate letters A-E.  Real candidate identities and selection strata live
only in ``manifest.json``; the page never fetches that file and never reveals
identities after a vote is locked.  Generated audio is exposed through opaque
symlink names under the page-local ``assets`` directory.
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
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/listening_frontend/seedtts_valid_benchmark"
    / "batch343637_subjective_20260711/C2_B2_multiway_blind10"
)
DEFAULT_VER23 = ROOT / "testset/outputs/ver2_3_ctc_clean_seedtts_valid_full"
DEFAULT_BATCH33 = (
    ROOT
    / "testset/outputs/ver23_content_side_text_bypass_3k_seedtts320_20260710"
    / "ver23_content_side_text_bypass_3k_step-3000_seedtts320_all_d2d3_seed1234"
)
DEFAULT_C2_ROOT = ROOT / "testset/outputs/ver23_family_final_seedtts320_batch37_step3000_20260711_mtts"
DEFAULT_C2_L14 = (
    DEFAULT_C2_ROOT
    / "runs/ver23_batch37_C2_refcfg_step-3000_cfg1p4_seedtts320_d2d3_seed1234"
)
DEFAULT_C2_L16 = (
    DEFAULT_C2_ROOT
    / "runs/ver23_batch37_C2_refcfg_step-3000_cfg1p6_seedtts320_d2d3_seed1234"
)
DEFAULT_B2 = (
    ROOT
    / "testset/outputs/ver23_family_final_seedtts320_batch3436_step3000_20260711_mtts"
    / "runs/ver23_batch3436_B2_text_r1_step-3000_cfg1p0_seedtts320_d2d3_seed1234"
)
DEFAULT_DIAGNOSTICS = DEFAULT_C2_ROOT / "aggregate/batch37_step3000.dual_encoder_cases.csv"

ROLE_ORDER = ("ver2_3", "batch33", "c2_lambda_1_4", "c2_lambda_1_6", "b2")
BLIND_LETTERS = ("A", "B", "C", "D", "E")
VALID_BINDINGS = {"ref-bound", "src-bound", "ambiguous"}
REQUESTED_BY_STRATUM = {"ref-bound": 3, "src-bound": 3, "ambiguous": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the ten-case anonymous five-way no_text listening page.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ver2-3-run-dir", default=str(DEFAULT_VER23))
    parser.add_argument("--batch33-run-dir", default=str(DEFAULT_BATCH33))
    parser.add_argument("--c2-lambda-1-4-run-dir", default=str(DEFAULT_C2_L14))
    parser.add_argument("--c2-lambda-1-6-run-dir", default=str(DEFAULT_C2_L16))
    parser.add_argument("--b2-run-dir", default=str(DEFAULT_B2))
    parser.add_argument("--diagnostics", default=str(DEFAULT_DIAGNOSTICS))
    parser.add_argument(
        "--diagnostics-run",
        default="ver23_batch37_C2_refcfg_step-3000_cfg1p6_seedtts320_d2d3_seed1234",
        help="Run name to select from the multi-run dual-encoder CSV.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--selection-seed", type=int, default=20260711)
    parser.add_argument("--blind-seed", type=int, default=20260712)
    parser.add_argument("--binding-margin", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield row


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on", "keep"}:
        return True
    if lowered in {"0", "false", "no", "n", "off", "drop", "fail"}:
        return False
    return None


def pick(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return ""


def resolve_audio_path(value: Any, *, run_dir: Path, repo_root: Path) -> Path | None:
    if value in (None, ""):
        return None
    raw = Path(str(value)).expanduser()
    if raw.is_absolute():
        return raw.resolve(strict=False)
    candidates = (repo_root / raw, run_dir / raw, Path.cwd() / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve(strict=False)


def index_rows(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or row.get("sample_id") or "")
        if case_id:
            indexed[case_id] = row
    return indexed


def manifest_paths(run_dir: Path) -> list[Path]:
    paths = sorted(run_dir.glob("manifest*.jsonl"))
    return sorted(paths, key=lambda path: ("rerun" in path.name, path.name))


def merged_asr_paths(run_dir: Path) -> list[Path]:
    return [path for path in sorted(run_dir.glob("*.asr_eval.jsonl")) if ".shard" not in path.name]


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

    samples: dict[str, dict[str, Any]] = {}
    run_ids: Counter[str] = Counter()
    for case_id in sorted(set(manifests) | set(asr_rows)):
        manifest = manifests.get(case_id, {})
        asr = asr_rows.get(case_id, {})
        run_id = str(pick(asr.get("run_id"), manifest.get("run_id"), run_dir.name))
        run_ids[run_id] += 1
        generated = resolve_audio_path(
            pick(manifest.get("output_wav"), asr.get("target_audio")),
            run_dir=run_dir,
            repo_root=repo_root,
        )
        if generated is None:
            generated = (run_dir / f"{case_id}.wav").resolve(strict=False)
        samples[case_id] = {
            "case_id": case_id,
            "run_id": run_id,
            "mode": str(pick(asr.get("mode"), manifest.get("mode"))),
            "cell": str(pick(asr.get("cell"), manifest.get("cell"))),
            "source_lang": str(pick(asr.get("source_lang"), manifest.get("source_lang"))),
            "ref_lang": str(pick(asr.get("ref_lang"), manifest.get("ref_lang"))),
            "generated_audio": generated,
            "source_audio": resolve_audio_path(
                pick(manifest.get("source_audio"), asr.get("source_audio")),
                run_dir=run_dir,
                repo_root=repo_root,
            ),
            "reference_audio": resolve_audio_path(
                pick(manifest.get("timbre_ref_audio"), asr.get("timbre_ref_audio")),
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
            "content_keep": parse_bool(asr.get("content_keep")),
        }
    return {
        "run_dir": run_dir,
        "run_id": run_ids.most_common(1)[0][0] if run_ids else run_dir.name,
        "samples": samples,
    }


def classify_binding(sim_ref: float | None, sim_src: float | None, margin: float) -> str:
    if sim_ref is None or sim_src is None:
        return "missing"
    delta = sim_ref - sim_src
    if delta > margin:
        return "ref-bound"
    if delta < -margin:
        return "src-bound"
    return "ambiguous"


def load_diagnostics(path: Path, *, run_name: str, margin: float) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Diagnostics file does not exist: {path}")
    rows = [
        row
        for row in read_csv(path)
        if str(row.get("run") or row.get("run_id") or "") == run_name
        and str(row.get("mode") or "no_text") == "no_text"
    ]
    if not rows:
        raise ValueError(f"No no_text diagnostics rows for run={run_name!r} in {path}")

    diagnostics: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "")
        wavlm_ref = finite(row.get("sim_gen_ref"))
        wavlm_src = finite(row.get("sim_gen_source"))
        ecapa_ref = finite(row.get("ecapa_sim_gen_ref"))
        ecapa_src = finite(row.get("ecapa_sim_gen_source"))
        wavlm_binding = classify_binding(wavlm_ref, wavlm_src, margin)
        ecapa_binding = classify_binding(ecapa_ref, ecapa_src, margin)
        if wavlm_binding == ecapa_binding and wavlm_binding in VALID_BINDINGS:
            dual_stratum = wavlm_binding
            dual_rule = "strict_encoder_agreement"
        elif wavlm_binding != "missing" and ecapa_binding != "missing":
            # A disagreement is not a strict ambiguous consensus.  Keep it
            # outside every canonical stratum; it may only enter through an
            # explicitly recorded single-encoder shortfall fallback.
            dual_stratum = "encoder-disagreement"
            dual_rule = "encoder_disagreement_excluded_from_strict_strata"
        else:
            dual_stratum = "missing"
            dual_rule = "missing_encoder_metric"
        diagnostics[case_id] = {
            "wavlm_sim_ref": wavlm_ref,
            "wavlm_sim_src": wavlm_src,
            "wavlm_delta_ref_minus_src": (
                wavlm_ref - wavlm_src if wavlm_ref is not None and wavlm_src is not None else None
            ),
            "wavlm_binding": wavlm_binding,
            "speechbrain_ecapa_sim_ref": ecapa_ref,
            "speechbrain_ecapa_sim_src": ecapa_src,
            "speechbrain_ecapa_delta_ref_minus_src": (
                ecapa_ref - ecapa_src if ecapa_ref is not None and ecapa_src is not None else None
            ),
            "speechbrain_ecapa_binding": ecapa_binding,
            "dual_encoder_stratum": dual_stratum,
            "dual_encoder_rule": dual_rule,
            "content_keep": parse_bool(row.get("content_keep")),
            "content_filter_reason": str(row.get("content_filter_reason") or ""),
            "cer_tgt": finite(row.get("cer_tgt")),
            "wer_tgt": finite(row.get("wer_tgt")),
        }
    return diagnostics


def choose_anchor(samples: Iterable[dict[str, Any]], key: str) -> Path | None:
    for sample in samples:
        path = sample.get(key)
        if isinstance(path, Path) and path.is_file():
            return path
    return None


def build_eligible(
    runs: dict[str, dict[str, Any]],
    diagnostics: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    primary_samples = runs["c2_lambda_1_6"]["samples"]
    eligible: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for case_id, primary in primary_samples.items():
        if primary.get("mode") != "no_text":
            skipped["not_no_text"] += 1
            continue
        if case_id not in diagnostics:
            skipped["missing_diagnostics"] += 1
            continue
        missing_roles = [role for role, run in runs.items() if case_id not in run["samples"]]
        if missing_roles:
            skipped["missing_role_case:" + ",".join(missing_roles)] += 1
            continue
        aligned = [runs[role]["samples"][case_id] for role in ROLE_ORDER]
        if any(sample.get("mode") not in {"", "no_text"} for sample in aligned):
            skipped["mode_mismatch"] += 1
            continue
        generated = {role: runs[role]["samples"][case_id]["generated_audio"] for role in ROLE_ORDER}
        missing_audio = [role for role, path in generated.items() if not isinstance(path, Path) or not path.is_file()]
        if missing_audio:
            skipped["missing_generated:" + ",".join(missing_audio)] += 1
            continue
        source = choose_anchor(reversed(aligned), "source_audio")
        reference = choose_anchor(reversed(aligned), "reference_audio")
        if source is None or reference is None:
            skipped["missing_anchor"] += 1
            continue
        diag = diagnostics[case_id]
        eligible.append(
            {
                "case_id": case_id,
                "cell": str(primary.get("cell") or ""),
                "source_lang": str(primary.get("source_lang") or ""),
                "ref_lang": str(primary.get("ref_lang") or ""),
                "content_text": str(primary.get("content_text") or ""),
                "content_keep": diag.get("content_keep"),
                "source_audio": source,
                "reference_audio": reference,
                "generated": generated,
                "diagnostics": diag,
            }
        )
    return eligible, skipped


def diverse_pick(rows: list[dict[str, Any]], count: int, *, seed: int, salt: str) -> list[dict[str, Any]]:
    """Prefer content-keep rows while round-robining SeedTTS cells."""

    def round_robin(pool: list[dict[str, Any]], wanted: int, suffix: str) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in pool:
            grouped[str(row.get("cell") or "unknown")].append(row)
        rng = random.Random(f"{seed}:{salt}:{suffix}")
        keys = sorted(grouped)
        rng.shuffle(keys)
        for key in keys:
            grouped[key].sort(key=lambda item: item["case_id"])
            rng.shuffle(grouped[key])
        selected: list[dict[str, Any]] = []
        while keys and len(selected) < wanted:
            remaining: list[str] = []
            for key in keys:
                if grouped[key] and len(selected) < wanted:
                    selected.append(grouped[key].pop())
                if grouped[key]:
                    remaining.append(key)
            keys = remaining
        return selected

    keep = [row for row in rows if row.get("content_keep") is True]
    other = [row for row in rows if row.get("content_keep") is not True]
    picked = round_robin(keep, min(count, len(keep)), "content_keep")
    used = {row["case_id"] for row in picked}
    if len(picked) < count:
        picked.extend(round_robin([row for row in other if row["case_id"] not in used], count - len(picked), "fallback"))
    return picked


def tag_selection(row: dict[str, Any], *, stratum: str, source: str) -> dict[str, Any]:
    tagged = dict(row)
    tagged["selection_stratum"] = stratum
    tagged["selection_stratum_source"] = source
    return tagged


def select_cases(eligible: list[dict[str, Any]], *, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strict = {stratum: [] for stratum in VALID_BINDINGS}
    for row in eligible:
        stratum = row["diagnostics"].get("dual_encoder_stratum")
        if stratum in strict:
            strict[stratum].append(row)
    strict_counts = {stratum: len(rows) for stratum, rows in strict.items()}
    strict_keep_counts = {
        stratum: sum(row.get("content_keep") is True for row in rows)
        for stratum, rows in strict.items()
    }
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    fallback_counts = {stratum: 0 for stratum in VALID_BINDINGS}
    selected_sources: Counter[str] = Counter()

    # Select the short src-bound stratum first so its one required fallback
    # cannot later be consumed by the ambiguous pool.
    for stratum in ("src-bound", "ref-bound", "ambiguous"):
        wanted = REQUESTED_BY_STRATUM[stratum]
        pool = [
            row
            for row in strict[stratum]
            if row["case_id"] not in used and row.get("content_keep") is True
        ]
        strict_pick = diverse_pick(pool, min(wanted, len(pool)), seed=seed, salt=f"strict:{stratum}")
        chosen = [
            tag_selection(row, stratum=stratum, source="dual_encoder_consensus") for row in strict_pick
        ]
        used.update(row["case_id"] for row in strict_pick)

        shortfall = wanted - len(chosen)
        if shortfall:
            wavlm_pool = [
                row
                for row in eligible
                if row["case_id"] not in used
                and row.get("content_keep") is True
                and row["diagnostics"].get("wavlm_binding") == stratum
                and (
                    stratum != "src-bound"
                    or row["diagnostics"].get("speechbrain_ecapa_binding") == "ambiguous"
                )
            ]
            wavlm_pick = diverse_pick(
                wavlm_pool,
                min(shortfall, len(wavlm_pool)),
                seed=seed,
                salt=f"wavlm-fallback:{stratum}",
            )
            chosen.extend(
                tag_selection(
                    row,
                    stratum=stratum,
                    source=(
                        "wavlm_src_speechbrain_ambiguous_content_keep_fallback"
                        if stratum == "src-bound"
                        else "wavlm_single_encoder_content_keep_fallback"
                    ),
                )
                for row in wavlm_pick
            )
            used.update(row["case_id"] for row in wavlm_pick)
            fallback_counts[stratum] += len(wavlm_pick)

        if len(chosen) != wanted:
            raise ValueError(
                f"Could not fill {stratum}: selected={len(chosen)} requested={wanted}; "
                f"strict_counts={strict_counts} strict_keep_counts={strict_keep_counts}"
            )
        selected.extend(chosen)
        selected_sources.update(row["selection_stratum_source"] for row in chosen)

    random.Random(f"{seed}:page-order").shuffle(selected)
    return selected, {
        "strategy": "dual_encoder_consensus_with_content_keep_wavlm_shortfall_fallback",
        "stratification_reference": "C2 lambda=1.6 diagnostics",
        "binding_rule": "margin=0.05 per encoder; both encoders must have the same label, including ambiguous",
        "requested_by_stratum": REQUESTED_BY_STRATUM,
        "eligible_by_stratum": strict_counts,
        "eligible_content_keep_by_stratum": strict_keep_counts,
        "selected_by_stratum": dict(Counter(row["selection_stratum"] for row in selected)),
        "selected_by_source": dict(selected_sources),
        "fallback_selected_by_stratum": fallback_counts,
        "selected_content_keep": sum(row.get("content_keep") is True for row in selected),
        "selected_total": len(selected),
        "content_quality_policy": (
            "all selected cases require C2 lambda=1.6 content_keep=True; the second strict src-bound "
            "case is excluded because its CER is 0.588"
        ),
    }


def audio_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    return suffix if suffix in {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"} else ".wav"


def stage_audio(source: Path, destination: Path, *, output_dir: Path, force: bool) -> str:
    if not source.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        same = False
        if destination.is_symlink():
            target = Path(os.readlink(destination))
            if not target.is_absolute():
                target = destination.parent / target
            same = target.resolve(strict=False) == source.resolve(strict=False)
        if not same:
            if not force:
                raise FileExistsError(f"Conflicting asset exists (use --force): {destination}")
            destination.unlink()
    if not os.path.lexists(destination):
        os.symlink(str(source.resolve()), str(destination))
    return destination.relative_to(output_dir).as_posix()


def safe_name(value: str, max_len: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return re.sub(r"_+", "_", cleaned).strip("._")[:max_len] or "item"


def balanced_blind_mappings(
    page_id: str,
    case_ids: list[str],
    seed: int,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Build two randomized Latin squares for exact A-E position balance.

    Ten cases and five systems imply that each real system must occupy every
    letter exactly twice.  Each five-row Latin square contributes one
    occurrence per system/letter.  The second square is sampled until its five
    full mappings are disjoint from the first, so all ten cases receive a
    different mapping rather than merely repeating the first square.
    """

    if len(case_ids) != 10:
        raise ValueError(f"Balanced blind design requires exactly 10 cases, got {len(case_ids)}")
    rng = random.Random(f"{seed}:{page_id}:balanced-latin")

    def make_square(base: list[str], step: int) -> list[tuple[str, ...]]:
        width = len(base)
        return [tuple(base[(column + step * row) % width] for column in range(width)) for row in range(width)]

    first_base = list(ROLE_ORDER)
    rng.shuffle(first_base)
    first_step = rng.choice((1, 2, 3, 4))
    first = make_square(first_base, first_step)
    first_set = set(first)

    second: list[tuple[str, ...]] | None = None
    second_base: list[str] = []
    second_step = 0
    for _ in range(1000):
        candidate_base = list(ROLE_ORDER)
        rng.shuffle(candidate_base)
        candidate_step = rng.choice((1, 2, 3, 4))
        candidate = make_square(candidate_base, candidate_step)
        if first_set.isdisjoint(candidate):
            second = candidate
            second_base = candidate_base
            second_step = candidate_step
            break
    if second is None:
        raise RuntimeError("Could not construct a disjoint second Latin square")

    mapping_rows = first + second
    rng.shuffle(mapping_rows)
    mappings = {
        case_id: dict(zip(BLIND_LETTERS, mapping_row))
        for case_id, mapping_row in zip(case_ids, mapping_rows)
    }
    unique_count = len({tuple(mappings[case_id][letter] for letter in BLIND_LETTERS) for case_id in case_ids})
    position_counts = {
        letter: dict(Counter(mappings[case_id][letter] for case_id in case_ids))
        for letter in BLIND_LETTERS
    }
    expected = set(ROLE_ORDER)
    for letter, counts in position_counts.items():
        if set(counts) != expected or any(count != 2 for count in counts.values()):
            raise AssertionError(f"Unbalanced blind mapping at letter {letter}: {counts}")
    if unique_count != len(case_ids):
        raise AssertionError(f"Expected ten unique per-case mappings, got {unique_count}")
    return mappings, {
        "strategy": "two_disjoint_randomized_5x5_latin_squares",
        "cases": len(case_ids),
        "letters": list(BLIND_LETTERS),
        "roles": list(ROLE_ORDER),
        "occurrences_per_role_per_letter": 2,
        "unique_mapping_count": unique_count,
        "position_counts": position_counts,
        "latin_square_parameters": {
            "first_base": first_base,
            "first_step": first_step,
            "second_base": second_base,
            "second_step": second_step,
        },
    }


def build_page_data(
    selected: list[dict[str, Any]],
    *,
    output_dir: Path,
    page_id: str,
    mappings: dict[str, dict[str, str]],
    role_metadata: dict[str, dict[str, str]],
    force: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    display_cases: list[dict[str, Any]] = []
    manifest_cases: list[dict[str, Any]] = []
    assets = output_dir / "assets"
    for index, row in enumerate(selected, start=1):
        prefix = f"case_{index:03d}"
        source_path = row["source_audio"]
        reference_path = row["reference_audio"]
        source_href = stage_audio(
            source_path,
            assets / f"{prefix}_anchor_1{audio_suffix(source_path)}",
            output_dir=output_dir,
            force=force,
        )
        reference_href = stage_audio(
            reference_path,
            assets / f"{prefix}_anchor_2{audio_suffix(reference_path)}",
            output_dir=output_dir,
            force=force,
        )
        mapping = mappings[row["case_id"]]
        display_candidates: list[dict[str, str]] = []
        manifest_mapping: dict[str, dict[str, str]] = {}
        for letter in BLIND_LETTERS:
            role = mapping[letter]
            audio_path = row["generated"][role]
            href = stage_audio(
                audio_path,
                assets / f"{prefix}_candidate_{letter}{audio_suffix(audio_path)}",
                output_dir=output_dir,
                force=force,
            )
            display_candidates.append({"letter": letter, "audio": href})
            manifest_mapping[letter] = {
                "role": role,
                "label": role_metadata[role]["label"],
                "run_id": role_metadata[role]["run_id"],
                "run_dir": role_metadata[role]["run_dir"],
                "source_path": str(audio_path),
                "href": href,
            }

        display_cases.append(
            {
                "index": index,
                "case_id": row["case_id"],
                "source_audio": source_href,
                "reference_audio": reference_href,
                "candidates": display_candidates,
            }
        )
        manifest_cases.append(
            {
                "index": index,
                "case_id": row["case_id"],
                "mode": "no_text",
                "cell": row["cell"],
                "source_lang": row["source_lang"],
                "ref_lang": row["ref_lang"],
                "content_text": row["content_text"],
                "content_keep": row["content_keep"],
                "selection_stratum": row["selection_stratum"],
                "selection_stratum_source": row["selection_stratum_source"],
                "diagnostics": row["diagnostics"],
                "anchors": {
                    "source": {"source_path": str(source_path), "href": source_href},
                    "reference": {"source_path": str(reference_path), "href": reference_href},
                },
                "candidate_mapping": manifest_mapping,
            }
        )
    display = {"schema_version": 1, "page_id": page_id, "cases": display_cases}
    return display, manifest_cases


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>五候选音色盲听</title>
  <style>
    :root { --bg:#f3f5f8; --panel:#fff; --line:#d8dee8; --ink:#172033; --muted:#667085; --accent:#175cd3; --ok:#067647; --warn:#b54708; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI","PingFang SC",sans-serif; line-height:1.45; }
    .top { position:sticky; top:0; z-index:5; padding:16px 22px; background:rgba(243,245,248,.96); border-bottom:1px solid var(--line); backdrop-filter:blur(8px); }
    h1 { margin:0 0 5px; font-size:21px; }
    .sub { color:var(--muted); font-size:13px; }
    .toolbar { display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-top:12px; }
    .progress { color:var(--ok); font-weight:800; }
    button { border:1px solid var(--line); border-radius:7px; padding:8px 11px; background:#fff; color:var(--ink); font-weight:700; cursor:pointer; }
    button:hover { border-color:#98a2b3; }
    button:disabled { opacity:.48; cursor:not-allowed; }
    button.active { border-color:var(--accent); background:#eff4ff; color:var(--accent); }
    button.primary { border-color:var(--accent); background:var(--accent); color:#fff; }
    main { max-width:1780px; margin:0 auto; padding:18px 22px 50px; }
    .case { margin-bottom:15px; padding:15px; border:1px solid var(--line); border-left:4px solid #f79009; border-radius:10px; background:var(--panel); box-shadow:0 1px 2px rgba(16,24,40,.05); }
    .case.locked { border-left-color:#12b76a; }
    .case-title { font-size:15px; font-weight:800; overflow-wrap:anywhere; }
    .case-number { color:var(--muted); font-size:12px; }
    .audio-grid { display:grid; grid-template-columns:repeat(7,minmax(145px,1fr)); gap:9px; margin-top:12px; }
    .audio-card { min-width:0; padding:9px; border:1px solid var(--line); border-radius:8px; background:#fbfcfe; }
    .audio-card.anchor { border-color:#b9e6fe; background:#f0f9ff; }
    .audio-card.candidate { border-color:#fedf89; background:#fffcf5; }
    .audio-label { margin-bottom:7px; color:var(--accent); font-size:12px; font-weight:800; }
    .candidate .audio-label { color:var(--warn); font-size:14px; }
    audio { width:100%; height:36px; }
    .review { display:flex; align-items:center; gap:9px; flex-wrap:wrap; margin-top:12px; padding-top:12px; border-top:1px solid var(--line); }
    .question { margin-right:6px; font-size:13px; font-weight:800; }
    .status { color:var(--muted); font-size:12px; }
    textarea { width:100%; min-height:48px; margin-top:9px; resize:vertical; padding:8px; border:1px solid var(--line); border-radius:7px; font:inherit; }
    .footnote { margin-top:14px; color:var(--muted); font-size:12px; }
    @media(max-width:1380px) { .audio-grid { grid-template-columns:repeat(4,minmax(160px,1fr)); } }
    @media(max-width:850px) { .audio-grid { grid-template-columns:repeat(2,minmax(160px,1fr)); } }
    @media(max-width:520px) { main,.top { padding-left:11px; padding-right:11px; } .audio-grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header class="top">
    <h1>五候选音色盲听</h1>
    <div class="sub">共 10 条 no_text。每条先听 Source 与 Reference，再从匿名候选 A-E 中选择最像 Reference 的一条。确认后不会解盲，仍可点“修改选择”重新填写。</div>
    <div class="toolbar">
      <span id="progress" class="progress"></span>
      <button id="export" class="primary">导出 review JSON</button>
      <button id="clear">清空全部选择</button>
    </div>
  </header>
  <main>
    <div id="cases"></div>
    <div class="footnote">选择自动保存在当前浏览器。页面始终保持匿名；导出文件只记录 case ID 与候选字母。</div>
  </main>
  <script>
  const PAGE = __DISPLAY_JSON__;
  const LETTERS = ["A","B","C","D","E"];
  const STORAGE_KEY = `multiway-blind:${PAGE.page_id}`;
  let state = loadState();
  function esc(x) { return String(x ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }
  function loadState() { try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); } catch (_) { return {}; } }
  function answer(caseId) { return state[caseId] || {winner:"",locked:false,note:"",locked_at:""}; }
  function saveState() { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); updateProgress(); }
  function audioCard(label, src, cls) { return `<div class="audio-card ${cls}"><div class="audio-label">${esc(label)}</div><audio controls preload="none" src="${esc(src)}"></audio></div>`; }
  function choiceButton(caseId, letter, locked) { const active = answer(caseId).winner === letter ? "active" : ""; return `<button class="${active}" data-choice="${esc(caseId)}" data-letter="${letter}" ${locked ? "disabled" : ""}>${letter}</button>`; }
  function renderCase(item) {
    const a = answer(item.case_id);
    const candidates = item.candidates.map(x => audioCard(`候选 ${x.letter}`, x.audio, "candidate")).join("");
    const action = a.locked
      ? `<button data-edit="${esc(item.case_id)}">修改选择</button><span class="status">已确认：候选 ${esc(a.winner)}</span>`
      : `<button class="primary" data-lock="${esc(item.case_id)}" ${a.winner ? "" : "disabled"}>确认本条选择</button><span class="status">确认后仍保持匿名</span>`;
    return `<article class="case ${a.locked ? "locked" : ""}">
      <div class="case-number">Case ${String(item.index).padStart(2,"0")}</div>
      <div class="case-title">${esc(item.case_id)}</div>
      <div class="audio-grid">
        ${audioCard("Source", item.source_audio, "anchor")}
        ${audioCard("Reference", item.reference_audio, "anchor")}
        ${candidates}
      </div>
      <div class="review">
        <span class="question">哪个候选最像 Reference？</span>
        ${LETTERS.map(letter => choiceButton(item.case_id, letter, a.locked)).join("")}
        ${action}
      </div>
      <textarea data-note="${esc(item.case_id)}" placeholder="可选备注">${esc(a.note || "")}</textarea>
    </article>`;
  }
  function render() {
    document.getElementById("cases").innerHTML = PAGE.cases.map(renderCase).join("");
    document.querySelectorAll("button[data-choice]:not([disabled])").forEach(button => button.onclick = () => {
      const old = answer(button.dataset.choice);
      state[button.dataset.choice] = {...old,winner:button.dataset.letter,locked:false,locked_at:""};
      saveState(); render();
    });
    document.querySelectorAll("button[data-lock]:not([disabled])").forEach(button => button.onclick = () => {
      const old = answer(button.dataset.lock);
      state[button.dataset.lock] = {...old,locked:true,locked_at:new Date().toISOString()};
      saveState(); render();
    });
    document.querySelectorAll("button[data-edit]").forEach(button => button.onclick = () => {
      const old = answer(button.dataset.edit);
      state[button.dataset.edit] = {...old,locked:false,locked_at:""};
      saveState(); render();
    });
    document.querySelectorAll("textarea[data-note]").forEach(box => box.onchange = () => {
      const old = answer(box.dataset.note);
      state[box.dataset.note] = {...old,note:box.value};
      saveState();
    });
    document.querySelectorAll("audio").forEach(player => player.addEventListener("play", () => {
      document.querySelectorAll("audio").forEach(other => { if (other !== player) other.pause(); });
    }));
    updateProgress();
  }
  function updateProgress() {
    const locked = PAGE.cases.filter(item => answer(item.case_id).locked).length;
    document.getElementById("progress").textContent = `已确认 ${locked}/${PAGE.cases.length}`;
  }
  function exportPayload() {
    return {
      schema_version:1,
      page_id:PAGE.page_id,
      exported_at:new Date().toISOString(),
      complete:PAGE.cases.every(item => answer(item.case_id).locked),
      items:PAGE.cases.map(item => {
        const a = answer(item.case_id);
        return {case_id:item.case_id,winner:a.winner || "",locked:Boolean(a.locked),locked_at:a.locked_at || "",note:a.note || ""};
      })
    };
  }
  function downloadJSON() {
    const blob = new Blob([JSON.stringify(exportPayload(), null, 2)], {type:"application/json"});
    const url = URL.createObjectURL(blob); const anchor = document.createElement("a");
    anchor.href = url; anchor.download = `${PAGE.page_id}.review.json`; document.body.appendChild(anchor); anchor.click(); anchor.remove(); URL.revokeObjectURL(url);
  }
  document.getElementById("export").onclick = downloadJSON;
  document.getElementById("clear").onclick = () => { if (confirm("确认清空本页全部选择？")) { state = {}; localStorage.removeItem(STORAGE_KEY); render(); } };
  render();
  </script>
</body>
</html>
'''


def render_html(display: dict[str, Any]) -> str:
    data = json.dumps(display, ensure_ascii=False, separators=(",", ":"))
    data = data.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return HTML_TEMPLATE.replace("__DISPLAY_JSON__", data)


def main() -> int:
    args = parse_args()
    if args.binding_margin < 0:
        raise ValueError("--binding-margin must be >= 0")
    repo_root = Path(args.repo_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    run_dirs = {
        "ver2_3": Path(args.ver2_3_run_dir).expanduser().resolve(),
        "batch33": Path(args.batch33_run_dir).expanduser().resolve(),
        "c2_lambda_1_4": Path(args.c2_lambda_1_4_run_dir).expanduser().resolve(),
        "c2_lambda_1_6": Path(args.c2_lambda_1_6_run_dir).expanduser().resolve(),
        "b2": Path(args.b2_run_dir).expanduser().resolve(),
    }
    runs = {role: load_run(path, repo_root) for role, path in run_dirs.items()}
    diagnostics_path = Path(args.diagnostics).expanduser().resolve()
    diagnostics = load_diagnostics(
        diagnostics_path,
        run_name=args.diagnostics_run,
        margin=args.binding_margin,
    )
    eligible, skipped = build_eligible(runs, diagnostics)
    selected, selection_summary = select_cases(eligible, seed=args.selection_seed)
    if len(selected) != 10:
        raise AssertionError(f"Expected exactly 10 cases, got {len(selected)}")

    page_hash = hashlib.sha256(
        "\n".join(
            [
                str(args.selection_seed),
                str(args.blind_seed),
                *[str(run_dirs[role]) for role in ROLE_ORDER],
                *[row["case_id"] for row in selected],
            ]
        ).encode("utf-8")
    ).hexdigest()[:12]
    page_id = f"multiway_blind10_{page_hash}"
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.html"
    manifest_path = output_dir / "manifest.json"
    for path in (index_path, manifest_path):
        if path.exists() and not args.force:
            raise FileExistsError(f"Output exists (use --force): {path}")

    role_labels = {
        "ver2_3": "ver2.3 baseline",
        "batch33": "Batch-33",
        "c2_lambda_1_4": "C2 lambda=1.4",
        "c2_lambda_1_6": "C2 lambda=1.6",
        "b2": "B2",
    }
    role_metadata = {
        role: {
            "label": role_labels[role],
            "run_id": str(runs[role]["run_id"]),
            "run_dir": str(run_dirs[role]),
        }
        for role in ROLE_ORDER
    }
    mappings, blind_design = balanced_blind_mappings(
        page_id,
        [row["case_id"] for row in selected],
        args.blind_seed,
    )
    display, manifest_cases = build_page_data(
        selected,
        output_dir=output_dir,
        page_id=page_id,
        mappings=mappings,
        role_metadata=role_metadata,
        force=args.force,
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "page_id": page_id,
        "mode": "no_text",
        "mapping_visibility": "manifest_only_never_embedded_or_fetched_by_index_html",
        "response_semantics": "single anonymous A-E winner most similar to reference",
        "selection_seed": args.selection_seed,
        "blind_seed": args.blind_seed,
        "blind_design": blind_design,
        "binding_margin": args.binding_margin,
        "diagnostics_run": args.diagnostics_run,
        "diagnostics_source": str(diagnostics_path),
        "selection": selection_summary,
        "eligibility": {"eligible": len(eligible), "skipped": dict(sorted(skipped.items()))},
        "roles": role_metadata,
        "cases": manifest_cases,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    index_path.write_text(render_html(display), encoding="utf-8")

    print(f"[multiway-blind] wrote {index_path}")
    print(f"[multiway-blind] manifest {manifest_path}")
    print(f"[multiway-blind] page_id={safe_name(page_id)} eligible={len(eligible)} selected={len(selected)}")
    print(f"[multiway-blind] selection={json.dumps(selection_summary, ensure_ascii=False, sort_keys=True)}")
    print(f"[multiway-blind] skipped={json.dumps(dict(sorted(skipped.items())), ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

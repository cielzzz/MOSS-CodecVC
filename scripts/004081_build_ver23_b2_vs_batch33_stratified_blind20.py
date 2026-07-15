#!/usr/bin/env python3
"""Build a jointly stratified, never-revealed B2 versus Batch-33 blind20 page.

Each system first receives its own dual-encoder coarse label: ref/src requires
WavLM and SpeechBrain agreement, while all other combinations are ambiguous.
The private 4+4+4 block uses only cases where B2 and Batch-33 have the same
coarse label.  The random block samples the entire remaining fresh population,
one case per benchmark cell, and records its label-transition mix.  Both
generated systems must have content_keep=True and CER <= 0.30.

Previously reviewed cases are excluded whenever each strict pool remains
sufficient.  Any unavoidable exclusion fallback is minimized and recorded in
the private manifest.  Browser HTML contains only opaque candidates A/B.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LISTENING_ROOT = ROOT / "outputs/listening_frontend/seedtts_valid_benchmark/batch343637_subjective_20260711"
B2_DIAGNOSTICS = (
    ROOT
    / "testset/outputs/ver23_family_final_seedtts320_batch3436_step3000_20260711_mtts"
    / "aggregate/batch3436_step3000.dual_encoder_cases.csv"
)
B2_RUN = "ver23_batch3436_B2_text_r1_step-3000_cfg1p0_seedtts320_d2d3_seed1234"
BATCH33_DIAGNOSTICS = ROOT / "testset/outputs/seedtts_baseline_dual_encoder_20260711_mtts/dual_encoder_cases.csv"
BATCH33_RUN = "Batch33"
DEFAULT_EXCLUSIONS = (
    LISTENING_ROOT / "C2_B2_multiway_blind10/manifest.json",
    LISTENING_ROOT / "C2_lambda14_vs16_pairwise_blind10/manifest.json",
    LISTENING_ROOT / "C2L14_vs_Batch33_pairwise_blind20/manifest.json",
)
DEFAULT_OUTPUT = LISTENING_ROOT / "B2_vs_Batch33_stratified_blind20"
ROLES = ("b2", "batch33")
LETTERS = ("A", "B")
STRICT_BUCKETS = ("ref-bound", "src-bound", "ambiguous")
STRICT_PER_BUCKET = 4
RANDOM_COUNT = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 4+4+4+8 anonymous B2 versus Batch-33 blind20 page.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--b2-diagnostics", default=str(B2_DIAGNOSTICS))
    parser.add_argument("--b2-run", default=B2_RUN)
    parser.add_argument("--batch33-diagnostics", default=str(BATCH33_DIAGNOSTICS))
    parser.add_argument("--batch33-run", default=BATCH33_RUN)
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        default=[],
        help="Prior listening manifest to exclude; repeatable. Defaults to the three existing comparison pages.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--selection-seed", type=int, default=20260716)
    parser.add_argument("--blind-seed", type=int, default=20260717)
    parser.add_argument("--binding-margin", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Diagnostics file does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "keep"}:
        return True
    if text in {"0", "false", "no", "n", "off", "drop", "fail"}:
        return False
    return None


def finite(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def require_audio(value: Any, *, label: str) -> Path:
    path = Path(str(value)).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def classify(sim_ref: float | None, sim_src: float | None, margin: float) -> str:
    if sim_ref is None or sim_src is None:
        return "missing"
    delta = sim_ref - sim_src
    if delta > margin:
        return "ref-bound"
    if delta < -margin:
        return "src-bound"
    return "ambiguous"


def load_run_diagnostics(path: Path, run_name: str, margin: float) -> dict[str, dict[str, Any]]:
    selected = [
        row
        for row in read_csv(path)
        if str(row.get("run") or row.get("run_id") or "") == run_name
        and str(row.get("mode") or "no_text") == "no_text"
    ]
    if not selected:
        raise ValueError(f"No no_text rows for run={run_name!r} in {path}")
    indexed: dict[str, dict[str, Any]] = {}
    for row in selected:
        case_id = str(row.get("case_id") or "")
        wavlm_ref = finite(row.get("sim_gen_ref"))
        wavlm_src = finite(row.get("sim_gen_source"))
        ecapa_ref = finite(row.get("ecapa_sim_gen_ref"))
        ecapa_src = finite(row.get("ecapa_sim_gen_source"))
        wavlm_binding = classify(wavlm_ref, wavlm_src, margin)
        ecapa_binding = classify(ecapa_ref, ecapa_src, margin)
        if wavlm_binding == ecapa_binding and wavlm_binding in {"ref-bound", "src-bound"}:
            dual_stratum = wavlm_binding
            dual_rule = "encoder_agreement"
        elif wavlm_binding != "missing" and ecapa_binding != "missing":
            dual_stratum = "ambiguous"
            dual_rule = (
                "both_ambiguous" if wavlm_binding == ecapa_binding else "encoder_disagreement_mapped_to_ambiguous"
            )
        else:
            dual_stratum = "missing"
            dual_rule = "missing_encoder_metric"
        indexed[case_id] = {
            "case_id": case_id,
            "mode": str(row.get("mode") or "no_text"),
            "cell": str(row.get("cell") or ""),
            "content_keep": parse_bool(row.get("content_keep")),
            "content_filter_reason": str(row.get("content_filter_reason") or ""),
            "cer_tgt": finite(row.get("cer_tgt")),
            "wer_tgt": finite(row.get("wer_tgt")),
            "generated_audio": require_audio(row.get("target_audio"), label=f"generated audio for {case_id}"),
            "source_audio": require_audio(row.get("source_audio"), label=f"source anchor for {case_id}"),
            "reference_audio": require_audio(row.get("timbre_ref_audio"), label=f"reference anchor for {case_id}"),
            "wavlm_sim_ref": wavlm_ref,
            "wavlm_sim_src": wavlm_src,
            "wavlm_binding": wavlm_binding,
            "speechbrain_ecapa_sim_ref": ecapa_ref,
            "speechbrain_ecapa_sim_src": ecapa_src,
            "speechbrain_ecapa_binding": ecapa_binding,
            "coarse_label": dual_stratum,
            "coarse_label_rule": dual_rule,
        }
    return indexed


def load_exclusions(paths: list[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    union: set[str] = set()
    audit: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Exclusion manifest does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases = payload.get("cases") if isinstance(payload, dict) else None
        if not isinstance(cases, list):
            raise ValueError(f"No cases list in exclusion manifest: {path}")
        ids = {str(row.get("case_id") or "") for row in cases if isinstance(row, dict)}
        ids.discard("")
        union.update(ids)
        audit.append(
            {
                "manifest": str(path),
                "page_id": payload.get("page_id"),
                "case_count": len(ids),
            }
        )
    return union, audit


def build_eligible(
    b2_rows: dict[str, dict[str, Any]],
    batch33_rows: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    eligible: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for case_id in sorted(set(b2_rows) & set(batch33_rows)):
        b2 = b2_rows[case_id]
        batch33 = batch33_rows[case_id]
        if b2.get("content_keep") is not True:
            skipped["b2_content_not_keep"] += 1
            continue
        if batch33.get("content_keep") is not True:
            skipped["batch33_content_not_keep"] += 1
            continue
        if b2.get("cer_tgt") is None or b2["cer_tgt"] > 0.30:
            skipped["b2_cer_over_0p30"] += 1
            continue
        if batch33.get("cer_tgt") is None or batch33["cer_tgt"] > 0.30:
            skipped["batch33_cer_over_0p30"] += 1
            continue
        b2_coarse = str(b2["coarse_label"])
        batch33_coarse = str(batch33["coarse_label"])
        joint_stratum = b2_coarse if b2_coarse == batch33_coarse else "label-migration"
        eligible.append(
            {
                "case_id": case_id,
                "cell": str(batch33.get("cell") or b2.get("cell") or "unknown"),
                "source_audio": batch33["source_audio"],
                "reference_audio": batch33["reference_audio"],
                "generated": {"b2": b2["generated_audio"], "batch33": batch33["generated_audio"]},
                "b2_coarse_label": b2_coarse,
                "batch33_coarse_label": batch33_coarse,
                "joint_stratum": joint_stratum,
                "label_transition": f"{b2_coarse}->{batch33_coarse}",
                "diagnostics": {
                    "b2": {
                        key: value
                        for key, value in b2.items()
                        if key not in {"generated_audio", "source_audio", "reference_audio"}
                    },
                    "batch33": {
                        key: value
                        for key, value in batch33.items()
                        if key not in {"generated_audio", "source_audio", "reference_audio"}
                    },
                },
            }
        )
    return eligible, skipped


def diverse_pick(rows: list[dict[str, Any]], count: int, *, seed: int, salt: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("cell") or "unknown")].append(row)
    rng = random.Random(f"{seed}:{salt}")
    cells = sorted(grouped)
    rng.shuffle(cells)
    for cell in cells:
        grouped[cell].sort(key=lambda row: row["case_id"])
        rng.shuffle(grouped[cell])
    picked: list[dict[str, Any]] = []
    active = list(cells)
    while active and len(picked) < count:
        remaining: list[str] = []
        for cell in active:
            if grouped[cell] and len(picked) < count:
                picked.append(grouped[cell].pop())
            if grouped[cell]:
                remaining.append(cell)
        active = remaining
    return picked


def tagged(row: dict[str, Any], *, bucket: str, source: str) -> dict[str, Any]:
    result = dict(row)
    result["selection_bucket"] = bucket
    result["selection_source"] = source
    return result


def select_cases(
    eligible: list[dict[str, Any]],
    *,
    excluded_ids: set[str],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    available = [row for row in eligible if row["case_id"] not in excluded_ids]
    excluded_eligible = [row for row in eligible if row["case_id"] in excluded_ids]
    before_counts = Counter(row["joint_stratum"] for row in eligible)
    after_counts = Counter(row["joint_stratum"] for row in available)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    prior_page_fallback: Counter[str] = Counter()

    for bucket in STRICT_BUCKETS:
        pool = [row for row in available if row["joint_stratum"] == bucket]
        chosen = diverse_pick(pool, min(STRICT_PER_BUCKET, len(pool)), seed=seed, salt=f"strict:{bucket}")
        chosen_tagged = [
            tagged(row, bucket=bucket, source="joint_coarse_label_after_prior_page_exclusion")
            for row in chosen
        ]
        used.update(row["case_id"] for row in chosen)
        shortfall = STRICT_PER_BUCKET - len(chosen_tagged)
        if shortfall:
            fallback_pool = [
                row
                for row in excluded_eligible
                if row["joint_stratum"] == bucket and row["case_id"] not in used
            ]
            fallback = diverse_pick(
                fallback_pool,
                min(shortfall, len(fallback_pool)),
                seed=seed,
                salt=f"prior-page-fallback:{bucket}",
            )
            chosen_tagged.extend(
                tagged(row, bucket=bucket, source="minimal_prior_page_case_fallback")
                for row in fallback
            )
            used.update(row["case_id"] for row in fallback)
            prior_page_fallback[bucket] += len(fallback)
        if len(chosen_tagged) != STRICT_PER_BUCKET:
            raise ValueError(
                f"Insufficient strict {bucket} pool: after_exclusion={len(pool)} "
                f"before_exclusion={before_counts[bucket]} requested={STRICT_PER_BUCKET}"
            )
        selected.extend(chosen_tagged)

    # The random block represents the full remaining fresh population rather
    # than conditioning on agreement or migration, with one case per cell.
    remaining = [
        row
        for row in available
        if row["case_id"] not in used
    ]
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in remaining:
        by_cell[row["cell"]].append(row)
    if len(by_cell) < RANDOM_COUNT:
        raise ValueError(f"Random block needs eight cells, found {len(by_cell)}")
    random_rng = random.Random(f"{seed}:random-one-per-cell")
    cells = sorted(by_cell)
    random_rng.shuffle(cells)
    random_chosen: list[dict[str, Any]] = []
    for cell in cells[:RANDOM_COUNT]:
        pool = sorted(by_cell[cell], key=lambda row: row["case_id"])
        random_chosen.append(random_rng.choice(pool))
    if len({row["case_id"] for row in random_chosen}) != RANDOM_COUNT:
        raise AssertionError("Random block contains duplicate cases")
    selected.extend(
        tagged(row, bucket="random", source="random_one_per_cell_from_all_remaining_fresh_cases")
        for row in random_chosen
    )
    used.update(row["case_id"] for row in random_chosen)

    if len(selected) != 20 or len(used) != 20:
        raise AssertionError(f"Expected 20 unique selected cases, got selected={len(selected)} unique={len(used)}")
    random.Random(f"{seed}:page-order").shuffle(selected)
    random_rows = [row for row in selected if row["selection_bucket"] == "random"]
    return selected, {
        "strategy": "joint_coarse_label_4_4_4_plus_remaining_distribution_random8",
        "binding_rule": (
            "within each system, WavLM and SpeechBrain agreement defines ref/src at margin 0.05; "
            "all other combinations map to ambiguous; 4/4/4 requires equal B2 and Batch-33 coarse labels"
        ),
        "content_quality_gate": "both B2 and Batch-33 require content_keep=True and CER<=0.30",
        "requested": {"ref-bound": 4, "src-bound": 4, "ambiguous": 4, "random": 8},
        "eligible_total": len(eligible),
        "prior_page_exclusion_union": len(excluded_ids),
        "eligible_after_prior_page_exclusion": len(available),
        "strata_pool_before_exclusion": dict(before_counts),
        "strata_pool_after_exclusion": dict(after_counts),
        "prior_page_fallback_by_bucket": dict(prior_page_fallback),
        "selected_by_bucket": dict(Counter(row["selection_bucket"] for row in selected)),
        "selected_by_source": dict(Counter(row["selection_source"] for row in selected)),
        "random_policy": "one uniformly sampled fresh case from each of the eight benchmark cells, without label conditioning",
        "random_cells": sorted(row["cell"] for row in random_rows),
        "random_label_transitions": dict(Counter(row["label_transition"] for row in random_rows)),
        "random_first12_overlap": len(
            {row["case_id"] for row in random_rows}
            & {row["case_id"] for row in selected if row["selection_bucket"] != "random"}
        ),
        "selected_prior_page_overlap": len(
            {row["case_id"] for row in selected} & excluded_ids
        ),
    }


def balanced_mappings(
    case_ids: list[str],
    *,
    seed: int,
    namespace: str,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    if len(case_ids) != 20:
        raise ValueError(f"Balanced mapping requires 20 cases, got {len(case_ids)}")
    roles_at_a = [ROLES[0]] * 10 + [ROLES[1]] * 10
    random.Random(f"{seed}:{namespace}:balanced-ab").shuffle(roles_at_a)
    mappings: dict[str, dict[str, str]] = {}
    for case_id, role_at_a in zip(case_ids, roles_at_a):
        role_at_b = ROLES[1] if role_at_a == ROLES[0] else ROLES[0]
        mappings[case_id] = {"A": role_at_a, "B": role_at_b}
    counts = {
        letter: dict(Counter(mappings[case_id][letter] for case_id in case_ids))
        for letter in LETTERS
    }
    for letter, value in counts.items():
        if set(value) != set(ROLES) or set(value.values()) != {10}:
            raise AssertionError(f"Unbalanced mapping at {letter}: {value}")
    return mappings, {
        "strategy": "deterministic_random_assignment_with_exact_10_10_position_balance",
        "position_counts": counts,
    }


def audio_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    return suffix if suffix in {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"} else ".wav"


def stage_audio(source: Path, destination: Path, *, output_dir: Path, force: bool) -> str:
    if not source.is_file():
        raise FileNotFoundError(f"Audio does not exist: {source}")
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


def build_outputs(
    selected: list[dict[str, Any]],
    *,
    mappings: dict[str, dict[str, str]],
    role_metadata: dict[str, dict[str, str]],
    output_dir: Path,
    page_id: str,
    force: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    display_cases: list[dict[str, Any]] = []
    manifest_cases: list[dict[str, Any]] = []
    assets = output_dir / "assets"
    for index, row in enumerate(selected, start=1):
        prefix = f"case_{index:03d}"
        case_id = row["case_id"]
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
        display_candidates: list[dict[str, str]] = []
        private_mapping: dict[str, dict[str, str]] = {}
        for letter in LETTERS:
            role = mappings[case_id][letter]
            audio_path = row["generated"][role]
            href = stage_audio(
                audio_path,
                assets / f"{prefix}_candidate_{letter}{audio_suffix(audio_path)}",
                output_dir=output_dir,
                force=force,
            )
            display_candidates.append({"letter": letter, "audio": href})
            private_mapping[letter] = {
                "role": role,
                "label": role_metadata[role]["label"],
                "run_id": role_metadata[role]["run_id"],
                "diagnostics_source": role_metadata[role]["diagnostics_source"],
                "source_path": str(audio_path),
                "href": href,
            }
        display_cases.append(
            {
                "index": index,
                "case_id": case_id,
                "source_audio": source_href,
                "reference_audio": reference_href,
                "candidates": display_candidates,
            }
        )
        manifest_cases.append(
            {
                "index": index,
                "case_id": case_id,
                "mode": "no_text",
                "cell": row["cell"],
                "both_content_keep": True,
                "selection_bucket": row["selection_bucket"],
                "selection_source": row["selection_source"],
                "b2_coarse_label": row["b2_coarse_label"],
                "batch33_coarse_label": row["batch33_coarse_label"],
                "joint_stratum": row["joint_stratum"],
                "label_transition": row["label_transition"],
                "diagnostics": row["diagnostics"],
                "anchors": {
                    "source": {"source_path": str(source_path), "href": source_href},
                    "reference": {"source_path": str(reference_path), "href": reference_href},
                },
                "candidate_mapping": private_mapping,
            }
        )
    return {"schema_version": 1, "page_id": page_id, "cases": display_cases}, manifest_cases


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>双候选音色盲听</title>
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
    button.active { border-color:var(--accent); background:#eff4ff; color:var(--accent); }
    button.primary { border-color:var(--accent); background:var(--accent); color:#fff; }
    main { max-width:1260px; margin:0 auto; padding:18px 22px 50px; }
    .case { margin-bottom:15px; padding:15px; border:1px solid var(--line); border-left:4px solid #f79009; border-radius:10px; background:var(--panel); box-shadow:0 1px 2px rgba(16,24,40,.05); }
    .case.answered { border-left-color:#12b76a; }
    .case-title { font-size:15px; font-weight:800; overflow-wrap:anywhere; }
    .case-number { color:var(--muted); font-size:12px; }
    .audio-grid { display:grid; grid-template-columns:repeat(4,minmax(190px,1fr)); gap:10px; margin-top:12px; }
    .audio-card { min-width:0; padding:10px; border:1px solid var(--line); border-radius:8px; background:#fbfcfe; }
    .audio-card.anchor { border-color:#b9e6fe; background:#f0f9ff; }
    .audio-card.candidate { border-color:#fedf89; background:#fffcf5; }
    .audio-label { margin-bottom:7px; color:var(--accent); font-size:12px; font-weight:800; }
    .candidate .audio-label { color:var(--warn); font-size:14px; }
    audio { width:100%; height:36px; }
    .review { display:flex; align-items:center; gap:9px; flex-wrap:wrap; margin-top:12px; padding-top:12px; border-top:1px solid var(--line); }
    .question { margin-right:6px; font-size:13px; font-weight:800; }
    textarea { width:100%; min-height:48px; margin-top:9px; resize:vertical; padding:8px; border:1px solid var(--line); border-radius:7px; font:inherit; }
    .footnote { margin-top:14px; color:var(--muted); font-size:12px; }
    @media(max-width:900px) { .audio-grid { grid-template-columns:repeat(2,minmax(170px,1fr)); } }
    @media(max-width:520px) { main,.top { padding-left:11px; padding-right:11px; } .audio-grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header class="top">
    <h1>双候选音色盲听</h1>
    <div class="sub">共 20 条 no_text。每条先听 Source 与 Reference，再比较匿名候选 A、B；页面始终不会解盲。</div>
    <div class="toolbar">
      <span id="progress" class="progress"></span>
      <button id="export" class="primary">导出 review JSON</button>
      <button id="clear">清空全部选择</button>
    </div>
  </header>
  <main>
    <div id="cases"></div>
    <div class="footnote">选择与备注自动保存在当前浏览器，可随时修改。导出文件只记录 case ID 和匿名判断。</div>
  </main>
  <script>
  const PAGE = __DISPLAY_JSON__;
  const STORAGE_KEY = `balanced-pair-blind:${PAGE.page_id}`;
  const CHOICES = [
    ["A", "A 更像 Reference"],
    ["tie", "两者差不多"],
    ["B", "B 更像 Reference"],
    ["neither", "两个都不像 Reference"]
  ];
  let state = loadState();
  function esc(x) { return String(x ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }
  function loadState() { try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); } catch (_) { return {}; } }
  function answer(caseId) { return state[caseId] || {judgment:"",note:"",updated_at:""}; }
  function saveState() { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); updateProgress(); }
  function audioCard(label, src, cls) { return `<div class="audio-card ${cls}"><div class="audio-label">${esc(label)}</div><audio controls preload="none" src="${esc(src)}"></audio></div>`; }
  function choiceButton(caseId, value, label) { const active = answer(caseId).judgment === value ? "active" : ""; return `<button class="${active}" data-choice="${esc(caseId)}" data-value="${value}">${esc(label)}</button>`; }
  function renderCase(item) {
    const a = answer(item.case_id);
    const candidates = item.candidates.map(x => audioCard(`候选 ${x.letter}`, x.audio, "candidate")).join("");
    return `<article class="case ${a.judgment ? "answered" : ""}">
      <div class="case-number">Case ${String(item.index).padStart(2,"0")}</div>
      <div class="case-title">${esc(item.case_id)}</div>
      <div class="audio-grid">
        ${audioCard("Source", item.source_audio, "anchor")}
        ${audioCard("Reference", item.reference_audio, "anchor")}
        ${candidates}
      </div>
      <div class="review"><span class="question">哪个更像 Reference？</span>${CHOICES.map(x => choiceButton(item.case_id,x[0],x[1])).join("")}</div>
      <textarea data-note="${esc(item.case_id)}" placeholder="可选备注">${esc(a.note || "")}</textarea>
    </article>`;
  }
  function render() {
    document.getElementById("cases").innerHTML = PAGE.cases.map(renderCase).join("");
    document.querySelectorAll("button[data-choice]").forEach(button => button.onclick = () => {
      const old = answer(button.dataset.choice);
      state[button.dataset.choice] = {...old,judgment:button.dataset.value,updated_at:new Date().toISOString()};
      saveState(); render();
    });
    document.querySelectorAll("textarea[data-note]").forEach(box => box.onchange = () => {
      const old = answer(box.dataset.note);
      state[box.dataset.note] = {...old,note:box.value,updated_at:new Date().toISOString()};
      saveState();
    });
    document.querySelectorAll("audio").forEach(player => player.addEventListener("play", () => {
      document.querySelectorAll("audio").forEach(other => { if (other !== player) other.pause(); });
    }));
    updateProgress();
  }
  function updateProgress() {
    const done = PAGE.cases.filter(item => answer(item.case_id).judgment).length;
    document.getElementById("progress").textContent = `已完成 ${done}/${PAGE.cases.length}`;
  }
  function exportPayload() {
    return {
      schema_version:1,
      page_id:PAGE.page_id,
      exported_at:new Date().toISOString(),
      complete:PAGE.cases.every(item => Boolean(answer(item.case_id).judgment)),
      items:PAGE.cases.map(item => {
        const a = answer(item.case_id);
        return {case_id:item.case_id,judgment:a.judgment || "",updated_at:a.updated_at || "",note:a.note || ""};
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
    b2_path = Path(args.b2_diagnostics).expanduser().resolve()
    batch33_path = Path(args.batch33_diagnostics).expanduser().resolve()
    exclusion_paths = (
        [Path(value).expanduser().resolve() for value in args.exclude_manifest]
        if args.exclude_manifest
        else [path.resolve() for path in DEFAULT_EXCLUSIONS]
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    b2_rows = load_run_diagnostics(b2_path, args.b2_run, args.binding_margin)
    batch33_rows = load_run_diagnostics(batch33_path, args.batch33_run, args.binding_margin)
    excluded_ids, exclusion_audit = load_exclusions(exclusion_paths)
    eligible, skipped = build_eligible(b2_rows, batch33_rows)
    selected, selection_audit = select_cases(eligible, excluded_ids=excluded_ids, seed=args.selection_seed)
    selected_ids = [row["case_id"] for row in selected]

    page_hash = hashlib.sha256(
        "\n".join(
            [
                str(args.selection_seed),
                str(args.blind_seed),
                str(b2_path),
                str(batch33_path),
                *selected_ids,
            ]
        ).encode("utf-8")
    ).hexdigest()[:12]
    page_id = f"balanced_pair_blind20_{page_hash}"
    mappings, blind_balance = balanced_mappings(selected_ids, seed=args.blind_seed, namespace=page_id)
    role_metadata = {
        "b2": {
            "label": "B2",
            "run_id": args.b2_run,
            "diagnostics_source": str(b2_path),
        },
        "batch33": {
            "label": "Batch-33",
            "run_id": args.batch33_run,
            "diagnostics_source": str(batch33_path),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.html"
    manifest_path = output_dir / "manifest.json"
    for path in (index_path, manifest_path):
        if path.exists() and not args.force:
            raise FileExistsError(f"Output exists (use --force): {path}")
    display, manifest_cases = build_outputs(
        selected,
        mappings=mappings,
        role_metadata=role_metadata,
        output_dir=output_dir,
        page_id=page_id,
        force=args.force,
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "page_id": page_id,
        "mode": "no_text",
        "mapping_visibility": "private_manifest_only_never_embedded_or_fetched_by_index_html",
        "response_semantics": {
            "A": "anonymous candidate A is more similar to reference",
            "tie": "the two anonymous candidates are equally similar to reference",
            "B": "anonymous candidate B is more similar to reference",
            "neither": "neither anonymous candidate resembles reference",
        },
        "decision_rules": {
            "strategy": "highest_priority_match",
            "total_cases": 20,
            "require_complete": True,
            "rules": [
                {
                    "id": "b2_clearly_better",
                    "priority": 30,
                    "minimum_count": 12,
                    "count": {"role_wins": ["b2"], "outcomes": []},
                    "result": {"verdict": "B2明显更好", "action": "可进30k"},
                },
                {
                    "id": "b2_clearly_worse",
                    "priority": 20,
                    "minimum_count": 12,
                    "count": {"role_wins": ["batch33"], "outcomes": []},
                    "result": {"verdict": "B2明显更差", "action": "重新分析"},
                },
                {
                    "id": "b2_not_clearly_worse",
                    "priority": 10,
                    "minimum_count": 12,
                    "count": {"role_wins": ["b2"], "outcomes": ["tie"]},
                    "result": {"verdict": "B2不明显更差", "action": "作为主线"},
                },
            ],
            "default_result": {"verdict": "未命中", "action": "继续分析"},
        },
        "selection_seed": args.selection_seed,
        "blind_seed": args.blind_seed,
        "binding_margin": args.binding_margin,
        "roles": role_metadata,
        "exclusions": {
            "sources": exclusion_audit,
            "unique_case_union": len(excluded_ids),
        },
        "eligibility": {
            "both_content_keep_and_cer_le_0p30": len(eligible),
            "skipped": dict(sorted(skipped.items())),
        },
        "selection_audit": selection_audit,
        "strata_audit": {
            "stratification_role": "joint_b2_batch33",
            "definition": selection_audit["binding_rule"],
            "content_quality_gate": selection_audit["content_quality_gate"],
            "before_prior_page_exclusion": selection_audit["strata_pool_before_exclusion"],
            "after_prior_page_exclusion": selection_audit["strata_pool_after_exclusion"],
            "selected": selection_audit["selected_by_bucket"],
            "prior_page_fallback": selection_audit["prior_page_fallback_by_bucket"],
        },
        "blind_balance": blind_balance,
        "cases": manifest_cases,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    index_path.write_text(render_html(display), encoding="utf-8")

    print(f"[stratified-pair] wrote {index_path}")
    print(f"[stratified-pair] manifest {manifest_path}")
    print(f"[stratified-pair] page_id={page_id} eligible={len(eligible)} selected={len(selected)}")
    print(f"[stratified-pair] selection={json.dumps(selection_audit, ensure_ascii=False, sort_keys=True)}")
    print(f"[stratified-pair] balance={json.dumps(blind_balance, ensure_ascii=False, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

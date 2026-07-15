#!/usr/bin/env python3
"""Build a fresh 20-case, never-revealed A/B timbre listening page.

Selection is performed from the formal no_text outputs of the two requested
systems.  A case is eligible only when both systems have ``content_keep=True``,
both generated files and the Source/Reference anchors exist, and the case was
not used by the earlier ten-case page.  Eligible cases are sampled with a
deterministic cell round-robin to maximize coverage and balance.

The HTML contains only opaque A/B candidates.  Real identities, selection
audit data, and the offline decision rule are stored only in ``manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
LISTENING_ROOT = ROOT / "outputs/listening_frontend/seedtts_valid_benchmark/batch343637_subjective_20260711"
DEFAULT_CANDIDATE_RUN = (
    ROOT
    / "testset/outputs/ver23_family_final_seedtts320_batch37_step3000_20260711_mtts"
    / "runs/ver23_batch37_C2_refcfg_step-3000_cfg1p4_seedtts320_d2d3_seed1234"
)
DEFAULT_BASELINE_RUN = (
    ROOT
    / "testset/outputs/ver23_content_side_text_bypass_3k_seedtts320_20260710"
    / "ver23_content_side_text_bypass_3k_step-3000_seedtts320_all_d2d3_seed1234"
)
DEFAULT_EXCLUSION_MANIFEST = LISTENING_ROOT / "C2_B2_multiway_blind10/manifest.json"
DEFAULT_OUTPUT = LISTENING_ROOT / "C2L14_vs_Batch33_pairwise_blind20"
ROLES = ("candidate", "baseline")
LETTERS = ("A", "B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the fresh 20-case anonymous A/B timbre listening page.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidate-run-dir", default=str(DEFAULT_CANDIDATE_RUN))
    parser.add_argument("--baseline-run-dir", default=str(DEFAULT_BASELINE_RUN))
    parser.add_argument("--exclusion-manifest", default=str(DEFAULT_EXCLUSION_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--selection-seed", type=int, default=20260714)
    parser.add_argument("--blind-seed", type=int, default=20260715)
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


def pick(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return ""


def resolve_audio(value: Any, *, run_dir: Path, repo_root: Path) -> Path | None:
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


def load_run(run_dir: Path, repo_root: Path) -> dict[str, Any]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    manifest_paths = sorted(
        run_dir.glob("manifest*.jsonl"),
        key=lambda path: ("rerun" in path.name, path.name),
    )
    if not manifest_paths:
        raise FileNotFoundError(f"No manifest*.jsonl under {run_dir}")
    manifests: dict[str, dict[str, Any]] = {}
    for path in manifest_paths:
        manifests.update(index_rows(iter_jsonl(path)))
    asr_rows: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("*.asr_eval.jsonl")):
        if ".shard" not in path.name:
            asr_rows.update(index_rows(iter_jsonl(path)))

    samples: dict[str, dict[str, Any]] = {}
    run_ids: Counter[str] = Counter()
    for case_id in sorted(set(manifests) | set(asr_rows)):
        manifest = manifests.get(case_id, {})
        asr = asr_rows.get(case_id, {})
        run_id = str(pick(asr.get("run_id"), manifest.get("run_id"), run_dir.name))
        run_ids[run_id] += 1
        generated = resolve_audio(
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
            "source_audio": resolve_audio(
                pick(manifest.get("source_audio"), asr.get("source_audio")),
                run_dir=run_dir,
                repo_root=repo_root,
            ),
            "reference_audio": resolve_audio(
                pick(manifest.get("timbre_ref_audio"), asr.get("timbre_ref_audio")),
                run_dir=run_dir,
                repo_root=repo_root,
            ),
            "content_keep": parse_bool(asr.get("content_keep")),
            "content_filter_reason": str(asr.get("content_filter_reason") or ""),
            "cer_tgt": finite(asr.get("cer_tgt")),
            "wer_tgt": finite(asr.get("wer_tgt")),
        }
    return {
        "run_dir": run_dir,
        "run_id": run_ids.most_common(1)[0][0] if run_ids else run_dir.name,
        "samples": samples,
        "manifest_paths": [str(path.resolve()) for path in manifest_paths],
    }


def load_exclusions(path: Path) -> tuple[set[str], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Exclusion manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list):
        raise ValueError(f"No cases list in exclusion manifest: {path}")
    case_ids = {str(row.get("case_id") or "") for row in cases if isinstance(row, dict)}
    case_ids.discard("")
    return case_ids, payload


def choose_anchor(samples: Iterable[dict[str, Any]], key: str) -> Path | None:
    for sample in samples:
        path = sample.get(key)
        if isinstance(path, Path) and path.is_file():
            return path
    return None


def build_eligible(
    candidate_run: dict[str, Any],
    baseline_run: dict[str, Any],
    excluded_ids: set[str],
) -> tuple[list[dict[str, Any]], Counter[str], dict[str, Any]]:
    candidate_samples = candidate_run["samples"]
    baseline_samples = baseline_run["samples"]
    skipped: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    audit_counts = {
        "candidate_total": len(candidate_samples),
        "baseline_total": len(baseline_samples),
        "intersection_total": len(set(candidate_samples) & set(baseline_samples)),
        "exclusion_manifest_case_count": len(excluded_ids),
    }
    for case_id in sorted(set(candidate_samples) & set(baseline_samples)):
        candidate = candidate_samples[case_id]
        baseline = baseline_samples[case_id]
        if candidate.get("mode") != "no_text" or baseline.get("mode") != "no_text":
            skipped["not_no_text"] += 1
            continue
        if case_id in excluded_ids:
            skipped["excluded_previous_blind10"] += 1
            continue
        if candidate.get("content_keep") is not True:
            skipped["candidate_content_not_keep"] += 1
            continue
        if baseline.get("content_keep") is not True:
            skipped["baseline_content_not_keep"] += 1
            continue
        candidate_audio = candidate.get("generated_audio")
        baseline_audio = baseline.get("generated_audio")
        if not isinstance(candidate_audio, Path) or not candidate_audio.is_file():
            skipped["missing_candidate_audio"] += 1
            continue
        if not isinstance(baseline_audio, Path) or not baseline_audio.is_file():
            skipped["missing_baseline_audio"] += 1
            continue
        source = choose_anchor((candidate, baseline), "source_audio")
        reference = choose_anchor((candidate, baseline), "reference_audio")
        if source is None:
            skipped["missing_source_anchor"] += 1
            continue
        if reference is None:
            skipped["missing_reference_anchor"] += 1
            continue
        cell = str(pick(candidate.get("cell"), baseline.get("cell")))
        eligible.append(
            {
                "case_id": case_id,
                "cell": cell,
                "source_lang": str(pick(candidate.get("source_lang"), baseline.get("source_lang"))),
                "ref_lang": str(pick(candidate.get("ref_lang"), baseline.get("ref_lang"))),
                "source_audio": source,
                "reference_audio": reference,
                "generated": {
                    "candidate": candidate_audio,
                    "baseline": baseline_audio,
                },
                "content_audit": {
                    "candidate": {
                        "content_keep": candidate.get("content_keep"),
                        "content_filter_reason": candidate.get("content_filter_reason"),
                        "cer_tgt": candidate.get("cer_tgt"),
                        "wer_tgt": candidate.get("wer_tgt"),
                    },
                    "baseline": {
                        "content_keep": baseline.get("content_keep"),
                        "content_filter_reason": baseline.get("content_filter_reason"),
                        "cer_tgt": baseline.get("cer_tgt"),
                        "wer_tgt": baseline.get("wer_tgt"),
                    },
                },
            }
        )
    audit_counts["eligible_total"] = len(eligible)
    audit_counts["eligible_by_cell"] = dict(sorted(Counter(row["cell"] for row in eligible).items()))
    return eligible, skipped, audit_counts


def select_round_robin(
    eligible: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(eligible) < count:
        raise ValueError(f"Need {count} eligible cases, found {len(eligible)}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[str(row.get("cell") or "unknown")].append(row)
    rng = random.Random(f"{seed}:cell-round-robin")
    cells = sorted(grouped)
    rng.shuffle(cells)
    for cell in cells:
        grouped[cell].sort(key=lambda row: row["case_id"])
        rng.shuffle(grouped[cell])

    selected: list[dict[str, Any]] = []
    active = list(cells)
    while active and len(selected) < count:
        remaining: list[str] = []
        for cell in active:
            if grouped[cell] and len(selected) < count:
                selected.append(grouped[cell].pop())
            if grouped[cell]:
                remaining.append(cell)
        active = remaining
    if len(selected) != count:
        raise ValueError(f"Cell round-robin selected only {len(selected)} of {count}")
    random.Random(f"{seed}:page-order").shuffle(selected)
    selected_counts = dict(sorted(Counter(row["cell"] for row in selected).items()))
    return selected, {
        "strategy": "both_content_keep_then_seeded_cell_round_robin",
        "requested": count,
        "selected": len(selected),
        "eligible_cell_count": len(cells),
        "selected_cell_count": len(selected_counts),
        "selected_by_cell": selected_counts,
        "max_min_selected_cell_gap": max(selected_counts.values()) - min(selected_counts.values()),
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
    position_counts = {
        letter: dict(Counter(mappings[case_id][letter] for case_id in case_ids))
        for letter in LETTERS
    }
    for letter, counts in position_counts.items():
        if set(counts) != set(ROLES) or set(counts.values()) != {10}:
            raise AssertionError(f"Unbalanced mapping at {letter}: {counts}")
    return mappings, {
        "strategy": "deterministic_random_assignment_with_exact_10_10_position_balance",
        "cases": 20,
        "position_counts": position_counts,
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
        case_id = str(row["case_id"])
        source_path: Path = row["source_audio"]
        reference_path: Path = row["reference_audio"]
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
            audio_path: Path = row["generated"][role]
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
                "run_dir": role_metadata[role]["run_dir"],
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
                "source_lang": row["source_lang"],
                "ref_lang": row["ref_lang"],
                "both_content_keep": True,
                "content_audit": row["content_audit"],
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
    <div class="sub">共 20 条全新 no_text。每条先听 Source 与 Reference，再比较匿名候选 A、B；页面始终不会解盲。</div>
    <div class="toolbar">
      <span id="progress" class="progress"></span>
      <button id="export" class="primary">导出 review JSON</button>
      <button id="clear">清空全部选择</button>
    </div>
  </header>
  <main>
    <div id="cases"></div>
    <div class="footnote">选择与备注会自动保存在当前浏览器，可随时修改。导出文件只记录 case ID 和匿名判断。</div>
  </main>
  <script>
  const PAGE = __DISPLAY_JSON__;
  const STORAGE_KEY = `fresh-pair-blind:${PAGE.page_id}`;
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
    repo_root = Path(args.repo_root).expanduser().resolve()
    candidate_dir = Path(args.candidate_run_dir).expanduser().resolve()
    baseline_dir = Path(args.baseline_run_dir).expanduser().resolve()
    exclusion_path = Path(args.exclusion_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    candidate_run = load_run(candidate_dir, repo_root)
    baseline_run = load_run(baseline_dir, repo_root)
    excluded_ids, exclusion_payload = load_exclusions(exclusion_path)
    eligible, skipped, eligibility_audit = build_eligible(candidate_run, baseline_run, excluded_ids)
    selected, selection_audit = select_round_robin(eligible, count=20, seed=args.selection_seed)
    selected_ids = [str(row["case_id"]) for row in selected]
    if set(selected_ids) & excluded_ids:
        raise AssertionError("Selected cases overlap the exclusion manifest")

    page_hash = hashlib.sha256(
        "\n".join(
            [
                str(args.selection_seed),
                str(args.blind_seed),
                str(candidate_dir),
                str(baseline_dir),
                *selected_ids,
            ]
        ).encode("utf-8")
    ).hexdigest()[:12]
    page_id = f"fresh_pair_blind20_{page_hash}"
    mappings, blind_balance = balanced_mappings(selected_ids, seed=args.blind_seed, namespace=page_id)
    role_metadata = {
        "candidate": {
            "label": "C2 lambda=1.4",
            "run_id": str(candidate_run["run_id"]),
            "run_dir": str(candidate_dir),
        },
        "baseline": {
            "label": "Batch-33",
            "run_id": str(baseline_run["run_id"]),
            "run_dir": str(baseline_dir),
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
        "decision_rule": {
            "target_role": "candidate",
            "target_label": "C2 lambda=1.4",
            "minimum_wins": 12,
            "total_cases": 20,
            "tie_and_neither_are_nonwins": True,
            "denominator": 20,
            "pass_if_target_wins_at_least": 12,
            "pass_expression": "decoded target wins >= 12/20",
            "non_win_categories": ["tie", "neither", "decoded baseline win"],
            "tie_and_neither_policy": "both count as target non-wins",
        },
        "selection_seed": args.selection_seed,
        "blind_seed": args.blind_seed,
        "roles": role_metadata,
        "exclusion": {
            "manifest": str(exclusion_path),
            "source_page_id": exclusion_payload.get("page_id"),
            "excluded_case_count": len(excluded_ids),
            "overlap_with_selected": 0,
        },
        "eligibility_audit": eligibility_audit,
        "skipped": dict(sorted(skipped.items())),
        "selection_audit": selection_audit,
        "blind_balance": blind_balance,
        "cases": manifest_cases,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    index_path.write_text(render_html(display), encoding="utf-8")

    print(f"[fresh-pair-blind] wrote {index_path}")
    print(f"[fresh-pair-blind] manifest {manifest_path}")
    print(f"[fresh-pair-blind] page_id={page_id} eligible={len(eligible)} selected={len(selected)}")
    print(f"[fresh-pair-blind] selection={json.dumps(selection_audit, ensure_ascii=False, sort_keys=True)}")
    print(f"[fresh-pair-blind] skipped={json.dumps(dict(sorted(skipped.items())), ensure_ascii=False)}")
    print(f"[fresh-pair-blind] balance={json.dumps(blind_balance, ensure_ascii=False, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

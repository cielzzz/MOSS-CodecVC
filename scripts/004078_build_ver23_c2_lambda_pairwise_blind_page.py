#!/usr/bin/env python3
"""Build a never-revealed A/B listening page from the frozen blind-10 set.

The input is the private manifest produced by 004077.  This generator keeps
the same ten cases, order, anchors, and selection strata, but stages only the
two requested generated systems as anonymous candidates A and B.  Assignment
is deterministic and exactly balanced: each real system appears in position A
five times and position B five times.

Real identities and strata are written only to the new private manifest.  The
browser-facing HTML never fetches that manifest and contains no system, run,
parameter, or stratum names.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LISTENING_ROOT = ROOT / "outputs/listening_frontend/seedtts_valid_benchmark/batch343637_subjective_20260711"
DEFAULT_SOURCE_MANIFEST = LISTENING_ROOT / "C2_B2_multiway_blind10/manifest.json"
DEFAULT_OUTPUT = LISTENING_ROOT / "C2_lambda14_vs16_pairwise_blind10"
PAIR_ROLES = ("c2_lambda_1_4", "c2_lambda_1_6")
LETTERS = ("A", "B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an anonymous, exactly balanced ten-case A/B listening page.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--blind-seed", type=int, default=20260713)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def require_file(value: Any, *, label: str) -> Path:
    path = Path(str(value)).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


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


def find_role_candidate(case: dict[str, Any], role: str) -> dict[str, Any]:
    matches = [
        dict(candidate)
        for candidate in (case.get("candidate_mapping") or {}).values()
        if candidate.get("role") == role
    ]
    if len(matches) != 1:
        raise ValueError(
            f"case={case.get('case_id')!r}: expected one candidate for role={role!r}, found {len(matches)}"
        )
    return matches[0]


def load_source_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {path}")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 10:
        raise ValueError(f"Expected exactly ten source cases in {path}")

    normalized: list[dict[str, Any]] = []
    for expected_index, raw in enumerate(cases, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Source case {expected_index} is not an object")
        case_id = str(raw.get("case_id") or "")
        if not case_id:
            raise ValueError(f"Source case {expected_index} has no case_id")
        anchors = raw.get("anchors") or {}
        source = require_file((anchors.get("source") or {}).get("source_path"), label=f"source anchor for {case_id}")
        reference = require_file(
            (anchors.get("reference") or {}).get("source_path"),
            label=f"reference anchor for {case_id}",
        )
        candidates = {role: find_role_candidate(raw, role) for role in PAIR_ROLES}
        for role, candidate in candidates.items():
            candidate["audio_path"] = require_file(candidate.get("source_path"), label=f"{role} audio for {case_id}")
        normalized.append(
            {
                "index": expected_index,
                "case_id": case_id,
                "source_audio": source,
                "reference_audio": reference,
                "selection_stratum": raw.get("selection_stratum"),
                "selection_stratum_source": raw.get("selection_stratum_source"),
                "content_keep": raw.get("content_keep"),
                "diagnostics": raw.get("diagnostics"),
                "candidates": candidates,
            }
        )
    return payload, normalized


def balanced_mappings(
    case_ids: list[str],
    *,
    seed: int,
    namespace: str,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    if len(case_ids) != 10:
        raise ValueError(f"Balanced pairwise mapping requires ten cases, got {len(case_ids)}")
    orientation = [PAIR_ROLES[0]] * 5 + [PAIR_ROLES[1]] * 5
    random.Random(f"{seed}:{namespace}:balanced-ab").shuffle(orientation)
    mappings: dict[str, dict[str, str]] = {}
    for case_id, role_at_a in zip(case_ids, orientation):
        role_at_b = PAIR_ROLES[1] if role_at_a == PAIR_ROLES[0] else PAIR_ROLES[0]
        mappings[case_id] = {"A": role_at_a, "B": role_at_b}
    position_counts = {
        letter: dict(Counter(mappings[case_id][letter] for case_id in case_ids))
        for letter in LETTERS
    }
    expected = set(PAIR_ROLES)
    for letter, counts in position_counts.items():
        if set(counts) != expected or set(counts.values()) != {5}:
            raise AssertionError(f"Unbalanced {letter} mapping: {counts}")
    return mappings, {
        "strategy": "deterministic_random_assignment_with_exact_5_5_position_balance",
        "cases": 10,
        "position_counts": position_counts,
    }


def build_outputs(
    cases: list[dict[str, Any]],
    *,
    mappings: dict[str, dict[str, str]],
    output_dir: Path,
    page_id: str,
    force: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    display_cases: list[dict[str, Any]] = []
    manifest_cases: list[dict[str, Any]] = []
    assets = output_dir / "assets"
    for case in cases:
        index = int(case["index"])
        case_id = str(case["case_id"])
        prefix = f"case_{index:03d}"
        source_path: Path = case["source_audio"]
        reference_path: Path = case["reference_audio"]
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
        private_mapping: dict[str, dict[str, Any]] = {}
        for letter in LETTERS:
            role = mappings[case_id][letter]
            candidate = case["candidates"][role]
            audio_path: Path = candidate["audio_path"]
            href = stage_audio(
                audio_path,
                assets / f"{prefix}_candidate_{letter}{audio_suffix(audio_path)}",
                output_dir=output_dir,
                force=force,
            )
            display_candidates.append({"letter": letter, "audio": href})
            private_mapping[letter] = {
                "role": role,
                "label": candidate.get("label"),
                "run_id": candidate.get("run_id"),
                "run_dir": candidate.get("run_dir"),
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
                "content_keep": case.get("content_keep"),
                "selection_stratum": case.get("selection_stratum"),
                "selection_stratum_source": case.get("selection_stratum_source"),
                "diagnostics": case.get("diagnostics"),
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
    <div class="sub">共 10 条 no_text。每条先听 Source 与 Reference，再判断匿名候选 A、B 哪个更像 Reference；页面始终不会解盲。</div>
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
  const STORAGE_KEY = `pair-blind:${PAGE.page_id}`;
  const CHOICES = [
    ["A", "A 更像 Reference"],
    ["tie", "两者差不多"],
    ["B", "B 更像 Reference"]
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
    source_manifest = Path(args.source_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    source_payload, cases = load_source_manifest(source_manifest)
    source_case_ids = [str(case["case_id"]) for case in cases]
    source_digest = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    page_hash = hashlib.sha256(
        "\n".join([str(args.blind_seed), source_digest, *source_case_ids]).encode("utf-8")
    ).hexdigest()[:12]
    page_id = f"pair_blind10_{page_hash}"
    mappings, balance = balanced_mappings(source_case_ids, seed=args.blind_seed, namespace=page_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.html"
    manifest_path = output_dir / "manifest.json"
    for path in (index_path, manifest_path):
        if path.exists() and not args.force:
            raise FileExistsError(f"Output exists (use --force): {path}")

    display, manifest_cases = build_outputs(
        cases,
        mappings=mappings,
        output_dir=output_dir,
        page_id=page_id,
        force=args.force,
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "page_id": page_id,
        "mode": "no_text",
        "source_manifest": str(source_manifest),
        "source_page_id": source_payload.get("page_id"),
        "source_selection": source_payload.get("selection"),
        "case_order_policy": "identical_to_source_manifest",
        "mapping_visibility": "private_manifest_only_never_embedded_or_fetched_by_index_html",
        "response_semantics": {
            "A": "anonymous candidate A is more similar to reference",
            "tie": "the two anonymous candidates are equally similar to reference",
            "B": "anonymous candidate B is more similar to reference",
        },
        "blind_seed": args.blind_seed,
        "blind_balance": balance,
        "roles": {
            role: source_payload.get("roles", {}).get(role, {})
            for role in PAIR_ROLES
        },
        "cases": manifest_cases,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    index_path.write_text(render_html(display), encoding="utf-8")

    print(f"[pair-blind] wrote {index_path}")
    print(f"[pair-blind] manifest {manifest_path}")
    print(f"[pair-blind] page_id={page_id} cases={len(cases)}")
    print(f"[pair-blind] balance={json.dumps(balance, ensure_ascii=False, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

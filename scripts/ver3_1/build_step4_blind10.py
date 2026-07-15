#!/usr/bin/env python3
"""Build a sanitized 10-case listening page for the Step-4 full320 output.

The published page and manifest intentionally contain only anonymous case labels
and relative asset paths.  Selection is deterministic and metric-independent:
one row is sampled from each no-text validation cell, with one additional row
in each cross-language direction to balance source/reference language counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "testset/outputs/ver3_1_step4_ddlfm_eval_20260715/full320/manifest.jsonl"
DEFAULT_OUTPUT = ROOT / "testset/outputs/ver3_1_step4_ddlfm_eval_20260715/blind10"

PRIMARY_CELLS = (
    "en_src_en_ref_same_gender",
    "en_src_zh_ref_f2m",
    "en_src_zh_ref_m2f",
    "en_src_zh_ref_same_gender",
    "zh_src_en_ref_f2m",
    "zh_src_en_ref_m2f",
    "zh_src_en_ref_same_gender",
    "zh_src_zh_ref_same_gender",
)

# These two extras keep source and reference languages at 5 EN / 5 ZH while
# making the gender-relation coverage 4 same-gender / 3 f2m / 3 m2f.
EXTRA_CELLS = (
    "en_src_zh_ref_f2m",
    "zh_src_en_ref_m2f",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
    return rows


def stable_index(seed: int, cell: str, slot: str, count: int) -> int:
    digest = hashlib.sha256(f"{seed}:{cell}:{slot}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count


def choose_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("mode") != "no_text" or row.get("status") != "ok":
            continue
        grouped[str(row.get("cell") or "")].append(row)

    for values in grouped.values():
        values.sort(key=lambda item: str(item.get("case_id") or ""))

    missing = [cell for cell in PRIMARY_CELLS if not grouped.get(cell)]
    if missing:
        raise RuntimeError(f"missing required no_text cells: {missing}")

    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    for cell in PRIMARY_CELLS:
        candidates = grouped[cell]
        row = candidates[stable_index(seed, cell, "primary", len(candidates))]
        selected.append(row)
        used_ids.add(str(row["case_id"]))

    for cell in EXTRA_CELLS:
        candidates = grouped[cell]
        start = stable_index(seed, cell, "extra", len(candidates))
        for offset in range(len(candidates)):
            row = candidates[(start + offset) % len(candidates)]
            if str(row["case_id"]) not in used_ids:
                selected.append(row)
                used_ids.add(str(row["case_id"]))
                break
        else:
            raise RuntimeError(f"could not choose a distinct extra row for {cell}")

    random.Random(seed).shuffle(selected)
    if len(selected) != 10 or len(used_ids) != 10:
        raise AssertionError("blind10 selection must contain 10 unique rows")
    return selected


def parse_cell(cell: str) -> tuple[str, str, str]:
    parts = cell.split("_")
    if len(parts) < 5 or parts[1] != "src" or parts[3] != "ref":
        raise ValueError(f"unexpected cell name: {cell}")
    return parts[0], parts[2], "_".join(parts[4:])


def copy_audio(source: str | Path, destination: Path) -> None:
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>语音转换抽听</title>
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
    main { max-width:1120px; margin:0 auto; padding:18px 22px 50px; }
    .case { margin-bottom:15px; padding:15px; border:1px solid var(--line); border-left:4px solid #f79009; border-radius:10px; background:var(--panel); box-shadow:0 1px 2px rgba(16,24,40,.05); }
    .case.answered { border-left-color:#12b76a; }
    .case-title { color:var(--muted); font-size:13px; font-weight:800; }
    .audio-grid { display:grid; grid-template-columns:repeat(3,minmax(210px,1fr)); gap:10px; margin-top:12px; }
    .audio-card { min-width:0; padding:10px; border:1px solid var(--line); border-radius:8px; background:#fbfcfe; }
    .audio-card.anchor { border-color:#b9e6fe; background:#f0f9ff; }
    .audio-card.candidate { border-color:#fedf89; background:#fffcf5; }
    .audio-label { margin-bottom:7px; color:var(--accent); font-size:12px; font-weight:800; }
    .candidate .audio-label { color:var(--warn); }
    audio { width:100%; height:36px; }
    .review { display:flex; align-items:center; gap:9px; flex-wrap:wrap; margin-top:12px; padding-top:12px; border-top:1px solid var(--line); }
    .question { width:100%; font-size:13px; font-weight:800; }
    textarea { width:100%; min-height:48px; margin-top:9px; resize:vertical; padding:8px; border:1px solid var(--line); border-radius:7px; font:inherit; }
    .footnote { margin-top:14px; color:var(--muted); font-size:12px; }
    @media(max-width:760px) { .audio-grid { grid-template-columns:1fr; } main,.top { padding-left:11px; padding-right:11px; } }
  </style>
</head>
<body>
  <header class="top">
    <h1>语音转换抽听</h1>
    <div class="sub">共 10 条。每条依次听 Source、Reference 和匿名 Candidate，判断 Candidate 的音色归属。页面不展示系统名、指标或内部运行信息。</div>
    <div class="toolbar">
      <span id="progress" class="progress"></span>
      <button id="export" class="primary">导出 review JSON</button>
      <button id="clear">清空全部选择</button>
    </div>
  </header>
  <main>
    <div id="cases"></div>
    <div class="footnote">选择与备注会自动保存在当前浏览器。导出文件只记录匿名 case 编号和主观判断。</div>
  </main>
  <script>
  const PAGE = __PAGE_JSON__;
  const STORAGE_KEY = `vc-listening:${PAGE.page_id}`;
  const CHOICES = [
    ["reference", "更像 Reference"],
    ["source", "更像 Source"],
    ["neither", "两边都不像"],
    ["uncertain", "介于两者 / 不确定"]
  ];
  let state = loadState();
  function esc(x) { return String(x ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }
  function loadState() { try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); } catch (_) { return {}; } }
  function answer(id) { return state[id] || {judgment:"",note:"",updated_at:""}; }
  function saveState() { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); updateProgress(); }
  function audioCard(label, src, cls) { return `<div class="audio-card ${cls}"><div class="audio-label">${esc(label)}</div><audio controls preload="none" src="${esc(src)}"></audio></div>`; }
  function choiceButton(id, value, label) { const active = answer(id).judgment === value ? "active" : ""; return `<button class="${active}" data-choice="${esc(id)}" data-value="${value}">${esc(label)}</button>`; }
  function renderCase(item) {
    const a = answer(item.case);
    return `<article class="case ${a.judgment ? "answered" : ""}">
      <div class="case-title">${esc(item.case)}</div>
      <div class="audio-grid">
        ${audioCard("Source", item.audio.source, "anchor")}
        ${audioCard("Reference", item.audio.reference, "anchor")}
        ${audioCard("Candidate", item.audio.candidate, "candidate")}
      </div>
      <div class="review"><span class="question">Candidate 的音色更接近哪一边？</span>${CHOICES.map(x => choiceButton(item.case,x[0],x[1])).join("")}</div>
      <textarea data-note="${esc(item.case)}" placeholder="可选备注（例如内容不可辨、噪声、音色特征）">${esc(a.note || "")}</textarea>
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
    const done = PAGE.cases.filter(item => answer(item.case).judgment).length;
    document.getElementById("progress").textContent = `已完成 ${done}/${PAGE.cases.length}`;
  }
  function exportPayload() {
    return {
      schema_version:1,
      page_id:PAGE.page_id,
      exported_at:new Date().toISOString(),
      complete:PAGE.cases.every(item => Boolean(answer(item.case).judgment)),
      items:PAGE.cases.map(item => {
        const a = answer(item.case);
        return {case:item.case,judgment:a.judgment || "",updated_at:a.updated_at || "",note:a.note || ""};
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


def build(input_manifest: Path, output_dir: Path, seed: int, *, force: bool) -> dict[str, Any]:
    selected = choose_rows(read_jsonl(input_manifest), seed)
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"output already exists: {output_dir}; pass --force to rebuild")
        shutil.rmtree(output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True)

    page_cases: list[dict[str, Any]] = []
    source_lang_counts: Counter[str] = Counter()
    ref_lang_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    language_pair_counts: Counter[str] = Counter()

    for index, row in enumerate(selected, start=1):
        case = f"Case {index:02d}"
        stem = f"case_{index:02d}"
        audio = {
            "source": f"assets/{stem}_source.wav",
            "reference": f"assets/{stem}_reference.wav",
            "candidate": f"assets/{stem}_candidate.wav",
        }
        copy_audio(row["source_audio"], output_dir / audio["source"])
        copy_audio(row["timbre_ref_audio"], output_dir / audio["reference"])
        copy_audio(row.get("output_wav") or row["target_audio"], output_dir / audio["candidate"])
        page_cases.append({"case": case, "audio": audio})

        source_lang, ref_lang, relation = parse_cell(str(row["cell"]))
        source_lang_counts[source_lang] += 1
        ref_lang_counts[ref_lang] += 1
        relation_counts[relation] += 1
        language_pair_counts["same_language" if source_lang == ref_lang else "cross_language"] += 1

    page_id = f"vc_blind10_{hashlib.sha256(str(seed).encode()).hexdigest()[:12]}"
    manifest = {
        "schema_version": 1,
        "page_id": page_id,
        "mode": "no_text",
        "response_semantics": "single anonymous candidate compared with source and reference",
        "privacy": "sanitized_relative_paths_only",
        "selection": {
            "strategy": "deterministic_metric_independent_cell_stratified_sample",
            "seed": seed,
            "case_count": len(page_cases),
            "source_language_counts": dict(sorted(source_lang_counts.items())),
            "reference_language_counts": dict(sorted(ref_lang_counts.items())),
            "language_pair_counts": dict(sorted(language_pair_counts.items())),
            "gender_relation_counts": dict(sorted(relation_counts.items())),
        },
        "cases": page_cases,
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    (output_dir / "manifest.json").write_text(manifest_text, encoding="utf-8")
    page_payload = {"schema_version": 1, "page_id": page_id, "cases": page_cases}
    html = HTML_TEMPLATE.replace("__PAGE_JSON__", json.dumps(page_payload, ensure_ascii=False, separators=(",", ":")))
    (output_dir / "index.html").write_text(html, encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = build(
        args.input_manifest.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        args.seed,
        force=args.force,
    )
    print(f"wrote: {args.output_dir / 'index.html'}")
    print(json.dumps(manifest["selection"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

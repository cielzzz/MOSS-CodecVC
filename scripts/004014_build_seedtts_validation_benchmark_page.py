#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_JSONL = ROOT / "testset/validation/seedtts_vc_ver2_3_validation.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "testset/outputs/ver2_3_ctc_clean_seedtts_valid_full"
DEFAULT_PAGE_DIR = ROOT / "outputs/listening_frontend/seedtts_valid_benchmark"


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc


def safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return stem[:180] or "item"


def load_existing_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    prefix = "window.SEEDTTS_BENCHMARK = "
    if not text.startswith(prefix):
        return None
    if text.endswith(";"):
        text = text[:-1]
    return json.loads(text[len(prefix) :])


def link_file(src: str | Path | None, dst: Path) -> str | None:
    if not src:
        return None
    src_path = Path(src)
    if not src_path.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src_path, dst)
    return dst.as_posix()


def copy_data_file(src: Path, dst_dir: Path) -> str | None:
    if not src.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst.name


def mode_display_text(row: dict[str, Any]) -> str:
    mode = row.get("mode")
    if mode == "text":
        return str(row.get("text") or row.get("content_ref_text") or "")
    return str(row.get("content_ref_text") or row.get("source_text") or "")


def build_run(args: argparse.Namespace, validation_rows: list[dict[str, Any]], page_dir: Path) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifests = [Path(p).expanduser().resolve() for p in args.manifest_jsonl]
    manifest_by_case: dict[str, dict[str, Any]] = {}
    manifest_counts: Counter[str] = Counter()

    for manifest in manifests:
        if not manifest.exists():
            continue
        for row in iter_jsonl(manifest):
            case_id = str(row.get("case_id") or "")
            if not case_id:
                continue
            manifest_by_case[case_id] = row
            manifest_counts[str(row.get("status") or "unknown")] += 1

    assets_dir = page_dir / "assets"
    source_dir = assets_dir / "source"
    timbre_dir = assets_dir / "timbre"
    target_dir = assets_dir / "runs" / safe_stem(args.run_id) / "target"

    samples: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    cell_counts: Counter[str] = Counter()
    missing_audio: Counter[str] = Counter()

    for index, row in enumerate(validation_rows, start=1):
        case_id = str(row.get("case_id") or "")
        manifest_row = manifest_by_case.get(case_id, {})
        output_wav = manifest_row.get("output_wav") or str(output_dir / f"{safe_stem(case_id)}.wav")
        status = str(manifest_row.get("status") or ("ok" if Path(output_wav).exists() else "missing"))
        output_exists = Path(output_wav).exists()
        if status != "ok" and output_exists:
            status = "ok_after_rerun"

        source_rel = link_file(
            row.get("source_audio"),
            source_dir / f"{safe_stem(case_id)}.wav",
        )
        timbre_rel = link_file(
            row.get("timbre_ref_audio"),
            timbre_dir / f"{safe_stem(case_id)}.wav",
        )
        target_rel = link_file(
            output_wav,
            target_dir / f"{safe_stem(case_id)}.wav",
        )
        if source_rel is None:
            missing_audio["source"] += 1
        if timbre_rel is None:
            missing_audio["timbre"] += 1
        if target_rel is None:
            missing_audio["target"] += 1

        rel = lambda p: str(Path(p).relative_to(page_dir)) if p else None
        mode = str(row.get("mode") or "")
        cell = str(row.get("cell") or "")
        sample = {
            "index": index,
            "case_id": case_id,
            "mode": mode,
            "cell": cell,
            "source_lang": row.get("source_lang"),
            "ref_lang": row.get("ref_lang"),
            "source_id": row.get("source_id"),
            "ref_id": row.get("ref_id"),
            "source_audio": rel(source_rel),
            "timbre_audio": rel(timbre_rel),
            "target_audio": rel(target_rel),
            "source_text": row.get("source_text"),
            "timbre_ref_text": row.get("timbre_ref_text"),
            "input_text": row.get("text") if mode == "text" else "",
            "content_text": row.get("content_ref_text"),
            "display_text": mode_display_text(row),
            "eval_text_source": row.get("eval_text_source"),
            "source_path": row.get("source_audio"),
            "timbre_path": row.get("timbre_ref_audio"),
            "target_path": output_wav,
            "status": status,
            "returncode": manifest_row.get("returncode"),
            "elapsed_sec": manifest_row.get("elapsed_sec"),
            "output_exists": output_exists,
        }
        samples.append(sample)
        status_counts[status] += 1
        mode_counts[mode] += 1
        cell_counts[cell] += 1

    return {
        "run_id": args.run_id,
        "label": args.run_label,
        "model_path": str(Path(args.model_path).expanduser()) if args.model_path else "",
        "output_dir": str(output_dir),
        "manifest_jsonl": [str(p) for p in manifests],
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": {
            "samples": len(samples),
            "status": dict(status_counts),
            "mode": dict(mode_counts),
            "cell": dict(cell_counts),
            "manifest_status_raw": dict(manifest_counts),
            "missing_audio_links": dict(missing_audio),
        },
        "samples": samples,
    }


def build_dataset(validation_rows: list[dict[str, Any]], validation_jsonl: Path) -> dict[str, Any]:
    modes = Counter(str(r.get("mode") or "") for r in validation_rows)
    cells_by_mode: dict[str, Counter[str]] = defaultdict(Counter)
    source_lang = Counter(str(r.get("source_lang") or "") for r in validation_rows)
    ref_lang = Counter(str(r.get("ref_lang") or "") for r in validation_rows)
    for row in validation_rows:
        cells_by_mode[str(row.get("mode") or "")][str(row.get("cell") or "")] += 1
    return {
        "name": "SeedTTS VC ver2.3 validation benchmark",
        "validation_jsonl": str(validation_jsonl),
        "total": len(validation_rows),
        "modes": dict(modes),
        "cells_by_mode": {k: dict(v) for k, v in cells_by_mode.items()},
        "source_lang": dict(source_lang),
        "ref_lang": dict(ref_lang),
    }


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SeedTTS Valid Benchmark</title>
  <script src="benchmark_data.js"></script>
  <style>
    :root { color-scheme: light; --bg:#f6f8fa; --panel:#ffffff; --ink:#172026; --muted:#58636d; --line:#d8dee4; --soft:#eef2f4; --accent:#0f766e; --blue:#1d4ed8; --ok:#0f766e; --bad:#b91c1c; --warn:#b45309; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    header { position:sticky; top:0; z-index:8; background:rgba(255,255,255,.96); border-bottom:1px solid var(--line); backdrop-filter:saturate(160%) blur(10px); }
    .bar { max-width:1680px; margin:0 auto; padding:18px 24px; display:grid; grid-template-columns:minmax(320px,1fr) minmax(520px,1.4fr); gap:18px; align-items:start; }
    h1 { margin:0; font-size:22px; line-height:1.2; letter-spacing:0; }
    .sub { color:var(--muted); font-size:13px; line-height:1.45; margin-top:6px; }
    main { max-width:1680px; margin:0 auto; padding:18px 24px 64px; }
    .controls { display:flex; flex-wrap:wrap; gap:8px; justify-content:flex-end; }
    button, select, input { height:34px; border:1px solid var(--line); background:white; color:var(--ink); border-radius:8px; padding:0 11px; font-weight:700; }
    input { min-width:260px; font-weight:600; }
    button { cursor:pointer; }
    button.active { background:#172026; color:white; border-color:#172026; }
    button.danger { border-color:#fecaca; color:#991b1b; }
    .summary { display:grid; grid-template-columns:repeat(6, minmax(0,1fr)); gap:10px; margin-bottom:14px; }
    .kpi { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; min-width:0; }
    .kpi b { display:block; font-size:12px; color:var(--muted); margin-bottom:5px; }
    .kpi strong { display:block; font-size:20px; line-height:1.2; }
    .kpi span { display:block; color:var(--muted); font-size:12px; margin-top:4px; overflow-wrap:anywhere; }
    .sample { background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; margin-bottom:14px; }
    .sample.review-bad { border-color:#dc2626; box-shadow:0 0 0 1px #fecaca inset; }
    .sample.review-ok { border-color:#86efac; }
    .sample-head { padding:13px 15px; background:#fbfcfd; border-bottom:1px solid var(--line); display:grid; grid-template-columns:1fr auto; gap:12px; align-items:start; }
    .title { display:flex; flex-wrap:wrap; align-items:center; gap:8px; min-width:0; }
    .idx { font-weight:850; }
    .pill { min-height:24px; display:inline-flex; align-items:center; border-radius:999px; background:var(--soft); color:var(--muted); font-size:12px; font-weight:850; padding:2px 9px; white-space:nowrap; }
    .pill.ok { color:var(--ok); background:#dcfce7; } .pill.bad { color:var(--bad); background:#fee2e2; } .pill.warn { color:var(--warn); background:#ffedd5; }
    .pill.text { color:#7c2d12; background:#ffedd5; } .pill.no_text { color:#075985; background:#e0f2fe; }
    .ids { color:var(--muted); font-size:12px; line-height:1.45; text-align:right; overflow-wrap:anywhere; }
    .grid { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); }
    .panel { min-width:0; padding:14px; border-right:1px solid var(--line); }
    .panel:last-child { border-right:0; }
    .label { min-height:22px; display:flex; justify-content:space-between; gap:8px; color:#26323b; font-size:12px; font-weight:850; margin-bottom:8px; }
    audio { width:100%; height:34px; display:block; margin-bottom:10px; }
    .target-list { display:grid; gap:10px; margin-bottom:12px; }
    .target-item { border:1px solid var(--line); border-radius:8px; background:#fbfcfd; padding:10px; min-width:0; }
    .target-item-head { min-height:24px; display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:8px; }
    .target-title { font-size:12px; font-weight:900; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .target-meta { color:var(--muted); font-size:11px; line-height:1.45; overflow-wrap:anywhere; }
    .text-box { border:1px solid var(--line); border-radius:8px; background:white; padding:9px 10px; font-size:13px; line-height:1.5; white-space:pre-wrap; overflow-wrap:anywhere; max-height:138px; overflow:auto; }
    .text-box.compact { max-height:70px; font-size:12px; color:var(--muted); }
    .mono { font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size:11px; line-height:1.4; color:#34404a; }
    details { margin-top:10px; }
    summary { cursor:pointer; color:var(--muted); font-size:12px; font-weight:800; }
    code { display:block; margin-top:7px; padding:8px; background:#f1f4f6; border:1px solid var(--line); border-radius:6px; white-space:pre-wrap; overflow-wrap:anywhere; }
    .review-box { margin-top:10px; border-top:1px solid var(--line); padding-top:10px; }
    .review-row { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px; }
    .review-row button { height:30px; padding:0 9px; font-size:12px; }
    .review-row button.active.bad { background:#dc2626; border-color:#dc2626; color:white; }
    .review-row button.active.ok { background:#15803d; border-color:#15803d; color:white; }
    textarea.review-note { width:100%; min-height:52px; resize:vertical; border:1px solid var(--line); border-radius:8px; padding:8px 9px; font:600 12px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    .empty { padding:28px; color:var(--muted); text-align:center; border:1px dashed var(--line); border-radius:8px; background:white; }
    @media (max-width: 1280px) { .bar { grid-template-columns:1fr; } .controls { justify-content:flex-start; } .summary { grid-template-columns:repeat(2, minmax(0,1fr)); } .grid { grid-template-columns:repeat(2, minmax(0,1fr)); } .panel:nth-child(2) { border-right:0; } .panel:nth-child(-n+2) { border-bottom:1px solid var(--line); } }
    @media (max-width: 760px) { main,.bar { padding-left:14px; padding-right:14px; } .summary,.grid,.sample-head { grid-template-columns:1fr; } .panel { border-right:0; border-bottom:1px solid var(--line); } .panel:last-child { border-bottom:0; } .ids { text-align:left; } input { min-width:100%; } }
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <div>
        <h1>SeedTTS Valid Benchmark</h1>
        <div class="sub">Source audio/text, timbre reference audio/text, text-mode input text, and side-by-side generated target audio for the fixed 320-case validation set.</div>
      </div>
      <div class="controls">
        <select id="runSelect"></select>
        <input id="searchBox" placeholder="Search case id, cell, text, path">
        <span id="filters"></span>
        <button id="exportBadBtn">Export Badcases</button>
        <button id="exportAllBtn">Export Review</button>
        <button id="clearReviewBtn" class="danger">Clear Review</button>
      </div>
    </div>
  </header>
  <main>
    <section class="summary" id="summary"></section>
    <section id="app"></section>
  </main>
<script>
const payload = window.SEEDTTS_BENCHMARK || {dataset:{}, runs:[]};
const COMPARE_RUN_ID = "__compare_all__";
const samplesByRun = new Map(payload.runs.map(run => [run.run_id, new Map((run.samples || []).map(sample => [sample.case_id, sample]))]));
let activeFilter = "all";
let activeRun = payload.runs.length > 1 ? COMPARE_RUN_ID : (payload.runs[0]?.run_id || "");
const reviewIssueLabels = {
  omission: "漏读",
  misread: "错读",
  misalign: "错位",
  tail_extra: "句末多读",
  repeat: "重复",
  speaker: "音色问题",
  other: "其他"
};
const reviewStoragePrefix = "seedtts_valid_review_v1";

function esc(value) {
  return String(value ?? "").replace(/[&<>"]/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;"}[ch] || ch));
}
function fmt(value, digits = 1) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "N/A";
}
function isCompareView() {
  return activeRun === COMPARE_RUN_ID && payload.runs.length > 1;
}
function baseRun() {
  return payload.runs[0] || {run_id:"", label:"", samples:[], counts:{}};
}
function selectedRun() {
  if (isCompareView()) return baseRun();
  return payload.runs.find(r => r.run_id === activeRun) || payload.runs[0] || {samples:[], counts:{}};
}
function selectedSamples() {
  return selectedRun().samples || [];
}
function activeRunLabel() {
  if (isCompareView()) return "Compare all runs";
  const run = selectedRun();
  return run.label || run.run_id || "run";
}
function sampleForRun(run, caseId) {
  return samplesByRun.get(run.run_id)?.get(caseId) || null;
}
function targetsForCase(caseId) {
  return payload.runs.map(run => {
    const sample = sampleForRun(run, caseId);
    return {run, sample, status: sample?.status || "missing"};
  });
}
function statusClass(status) {
  if (status === "ok" || status === "ok_after_rerun" || status === "skipped_exists") return "ok";
  if (status === "missing" || status === "unsupported_text_mode") return "warn";
  return "bad";
}
function reviewStorageKey(runId = activeRun) {
  return `${reviewStoragePrefix}:${runId || "default"}`;
}
function loadReview(runId = activeRun) {
  try { return JSON.parse(localStorage.getItem(reviewStorageKey(runId)) || "{}"); }
  catch (_err) { return {}; }
}
function saveReview(data, runId = activeRun) {
  localStorage.setItem(reviewStorageKey(runId), JSON.stringify(data));
}
function reviewForCase(caseId) {
  return loadReview()[caseId] || {rating:"", issues:[], note:""};
}
function setReview(caseId, patch) {
  const data = loadReview();
  const old = data[caseId] || {rating:"", issues:[], note:""};
  data[caseId] = {...old, ...patch, updated_at:new Date().toISOString()};
  if (!data[caseId].rating && !(data[caseId].issues || []).length && !data[caseId].note) delete data[caseId];
  saveReview(data);
}
function filtersForSamples(samples) {
  const cells = [...new Set(samples.map(s => s.cell).filter(Boolean))].sort();
  const reviewFilters = [["review:bad","manual bad"],["review:ok","manual ok"], ...Object.entries(reviewIssueLabels).map(([key,label]) => [`issue:${key}`, label])];
  return [["all","All"],["mode:no_text","no_text"],["mode:text","text"], ...reviewFilters, ...cells.map(c => [`cell:${c}`, c])];
}
function reviewPasses(sample) {
  if (!activeFilter.startsWith("review:") && !activeFilter.startsWith("issue:")) return true;
  const review = reviewForCase(sample.case_id);
  if (activeFilter === "review:bad") return review.rating === "bad";
  if (activeFilter === "review:ok") return review.rating === "ok";
  if (activeFilter.startsWith("issue:")) return (review.issues || []).includes(activeFilter.slice(6));
  return true;
}
function passes(sample, query) {
  if (activeFilter.startsWith("mode:") && sample.mode !== activeFilter.slice(5)) return false;
  if (activeFilter.startsWith("cell:") && sample.cell !== activeFilter.slice(5)) return false;
  if (!reviewPasses(sample)) return false;
  if (query) {
    const targetBlob = isCompareView()
      ? targetsForCase(sample.case_id).map(({run, sample: targetSample, status}) => [run.run_id, run.label, status, targetSample?.target_path, targetSample?.target_audio].join("\\n")).join("\\n")
      : [sample.target_path, sample.target_audio, sample.status].join("\\n");
    const blob = [sample.case_id, sample.mode, sample.cell, sample.source_text, sample.timbre_ref_text, sample.input_text, sample.content_text, sample.source_path, sample.timbre_path, targetBlob].join("\\n").toLowerCase();
    if (!blob.includes(query.toLowerCase())) return false;
  }
  return true;
}
function audioTag(src) {
  return src ? `<audio controls preload="none" src="${esc(src)}"></audio>` : `<div class="text-box">missing audio link</div>`;
}
function renderRunSelect() {
  const select = document.getElementById("runSelect");
  const compareOption = payload.runs.length > 1 ? `<option value="${COMPARE_RUN_ID}">Compare all runs</option>` : "";
  select.innerHTML = compareOption + payload.runs.map(r => `<option value="${esc(r.run_id)}">${esc(r.label || r.run_id)}</option>`).join("");
  select.value = activeRun;
  select.onchange = () => { activeRun = select.value; activeFilter = "all"; render(); };
}
function renderFilters(samples) {
  const el = document.getElementById("filters");
  el.innerHTML = filtersForSamples(samples).map(([key,label]) => `<button data-filter="${esc(key)}" class="${activeFilter === key ? "active" : ""}">${esc(label)}</button>`).join("");
  el.querySelectorAll("button").forEach(btn => btn.onclick = () => { activeFilter = btn.dataset.filter; render(); });
}
function addCounts(dst, src) {
  for (const [key, value] of Object.entries(src || {})) dst[key] = (dst[key] || 0) + Number(value || 0);
  return dst;
}
function summaryCounts() {
  if (!isCompareView()) return selectedRun().counts || {};
  const counts = {samples: payload.dataset.total || selectedSamples().length, status:{}, mode:{...(baseRun().counts?.mode || {})}, missing_audio_links:{}};
  for (const run of payload.runs) {
    addCounts(counts.status, run.counts?.status);
    addCounts(counts.missing_audio_links, run.counts?.missing_audio_links);
  }
  return counts;
}
function renderSummary(visibleCount) {
  const counts = summaryCounts();
  const status = counts.status || {};
  const mode = counts.mode || {};
  const missing = counts.missing_audio_links || {};
  const playable = (status.ok || 0) + (status.ok_after_rerun || 0) + (status.skipped_exists || 0);
  const targetSlots = isCompareView() ? (payload.runs.length * (payload.dataset.total || 0)) : (counts.samples || 0);
  const review = loadReview();
  const reviewItems = Object.values(review);
  const manualBad = reviewItems.filter(item => item.rating === "bad").length;
  const manualOk = reviewItems.filter(item => item.rating === "ok").length;
  const runSummary = isCompareView() ? payload.runs.map(r => r.label || r.run_id).join(", ") : `bad / ok: ${manualOk}`;
  document.getElementById("summary").innerHTML = `
    <div class="kpi"><b>Dataset</b><strong>${payload.dataset.total || 0}</strong><span>fixed validation cases</span></div>
    <div class="kpi"><b>Visible</b><strong>${visibleCount}</strong><span>after filters/search</span></div>
    <div class="kpi"><b>${isCompareView() ? "Target audio" : "Run status"}</b><strong>${playable}</strong><span>${isCompareView() ? `playable / ${targetSlots} target slots` : `ok ${status.ok || 0}, skipped ${status.skipped_exists || 0}`}; unsupported ${status.unsupported_text_mode || 0}</span></div>
    <div class="kpi"><b>${isCompareView() ? "Runs" : "Manual review"}</b><strong>${isCompareView() ? payload.runs.length : manualBad}</strong><span>${esc(runSummary)}</span></div>
    <div class="kpi"><b>Mode</b><strong>${mode.no_text || 0} / ${mode.text || 0}</strong><span>no_text / text</span></div>
    <div class="kpi"><b>Missing links</b><strong>${(missing.source || 0) + (missing.timbre || 0) + (missing.target || 0)}</strong><span>source ${missing.source || 0}, timbre ${missing.timbre || 0}, target ${missing.target || 0}</span></div>
  `;
}
function renderReviewBox(s) {
  const review = reviewForCase(s.case_id);
  const issues = review.issues || [];
  const issueButtons = Object.entries(reviewIssueLabels).map(([key,label]) => `<button data-case-id="${esc(s.case_id)}" data-review-issue="${esc(key)}" class="${issues.includes(key) ? "active" : ""}">${esc(label)}</button>`).join("");
  return `<div class="review-box">
    <div class="label"><span>Manual Review</span><span>${esc(review.rating || "unmarked")}</span></div>
    <div class="review-row">
      <button data-case-id="${esc(s.case_id)}" data-review-rating="ok" class="${review.rating === "ok" ? "active ok" : ""}">OK</button>
      <button data-case-id="${esc(s.case_id)}" data-review-rating="bad" class="${review.rating === "bad" ? "active bad" : ""}">Bad</button>
      <button data-case-id="${esc(s.case_id)}" data-review-rating="" class="${review.rating ? "" : "active"}">Clear</button>
    </div>
    <div class="review-row">${issueButtons}</div>
    <textarea class="review-note" data-case-id="${esc(s.case_id)}" placeholder="note">${esc(review.note || "")}</textarea>
  </div>`;
}
function targetFallback(sample, status) {
  if (status === "unsupported_text_mode") return "unsupported_text_mode";
  if (!sample) return "missing run sample";
  return "missing target audio";
}
function renderTargetItem(target) {
  const run = target.run;
  const sample = target.sample;
  const status = target.status;
  const label = run.label || run.run_id;
  const audio = sample?.target_audio ? audioTag(sample.target_audio) : `<div class="text-box compact">${esc(targetFallback(sample, status))}</div>`;
  return `<div class="target-item">
    <div class="target-item-head">
      <div class="target-title">${esc(label)}</div>
      <span class="pill ${statusClass(status)}">${esc(status)}</span>
    </div>
    ${audio}
    <div class="target-meta">elapsed ${esc(sample?.elapsed_sec ?? "N/A")}s · ${esc(run.run_id || "")}</div>
    <details><summary>target path</summary><code class="mono">${esc(sample?.target_path || "")}</code></details>
  </div>`;
}
function renderTargetPanel(s) {
  if (isCompareView()) {
    const targets = targetsForCase(s.case_id);
    return `<section class="panel">
      <div class="label"><span>Targets</span><span>${targets.length} runs</span></div>
      <div class="target-list">${targets.map(renderTargetItem).join("")}</div>
      <div class="label"><span>Expected Eval Text</span></div>
      <div class="text-box">${esc(s.content_text || s.display_text || "")}</div>
      ${renderReviewBox(s)}
    </section>`;
  }
  const targetLabel = `${selectedRun().label || selectedRun().run_id} Target`;
  return `<section class="panel">
    <div class="label"><span>${esc(targetLabel)}</span><span>generated</span></div>
    ${audioTag(s.target_audio)}
    <div class="label"><span>Expected Eval Text</span></div>
    <div class="text-box">${esc(s.content_text || s.display_text || "")}</div>
    <details><summary>target path</summary><code class="mono">${esc(s.target_path || "")}</code></details>
    ${renderReviewBox(s)}
  </section>`;
}
function renderCard(s) {
  const modeText = s.mode === "text" ? "Input Text" : "Content Text";
  const inputText = s.mode === "text" ? s.input_text : s.display_text;
  const review = reviewForCase(s.case_id);
  const reviewClass = review.rating === "bad" ? " review-bad" : (review.rating === "ok" ? " review-ok" : "");
  const statusPill = isCompareView()
    ? `<span class="pill">compare</span>`
    : `<span class="pill ${statusClass(s.status)}">${esc(s.status)}</span>`;
  const generationFields = isCompareView()
    ? `view=compare_all\\nruns=${payload.runs.length}`
    : `status=${esc(s.status)}\\nelapsed_sec=${esc(s.elapsed_sec ?? "")}\\nreturncode=${esc(s.returncode ?? "")}`;
  return `<article class="sample${reviewClass}">
    <div class="sample-head">
      <div class="title">
        <span class="idx">#${s.index}</span>
        <span class="pill ${esc(s.mode)}">${esc(s.mode)}</span>
        <span class="pill">${esc(s.cell)}</span>
        ${statusPill}
        <span class="pill">${esc(s.source_lang || "?")}→${esc(s.ref_lang || "?")}</span>
      </div>
      <div class="ids">${esc(s.case_id)}<br>${esc(s.source_id || "")} / ${esc(s.ref_id || "")}</div>
    </div>
    <div class="grid">
      <section class="panel">
        <div class="label"><span>Source Wav</span><span>${esc(s.source_lang || "")}</span></div>
        ${audioTag(s.source_audio)}
        <div class="label"><span>Source Text</span></div>
        <div class="text-box">${esc(s.source_text || "")}</div>
        <details><summary>source path</summary><code class="mono">${esc(s.source_path || "")}</code></details>
      </section>
      <section class="panel">
        <div class="label"><span>Timbre Wav</span><span>${esc(s.ref_lang || "")}</span></div>
        ${audioTag(s.timbre_audio)}
        <div class="label"><span>Timbre Text</span></div>
        <div class="text-box">${esc(s.timbre_ref_text || "")}</div>
        <details><summary>timbre path</summary><code class="mono">${esc(s.timbre_path || "")}</code></details>
      </section>
      <section class="panel">
        <div class="label"><span>${modeText}</span><span>${esc(s.eval_text_source || "")}</span></div>
        <div class="text-box">${esc(inputText || "")}</div>
        <details open><summary>generation fields</summary><code class="mono">mode=${esc(s.mode)}
${generationFields}</code></details>
      </section>
      ${renderTargetPanel(s)}
    </div>
  </article>`;
}
function render() {
  renderRunSelect();
  const baseSamples = selectedSamples();
  renderFilters(baseSamples);
  const query = document.getElementById("searchBox").value.trim();
  const samples = baseSamples.filter(s => passes(s, query));
  renderSummary(samples.length);
  document.getElementById("app").innerHTML = samples.length ? samples.map(renderCard).join("") : `<div class="empty">No samples match the current filter.</div>`;
  bindReviewEvents();
}
function bindReviewEvents() {
  document.querySelectorAll("[data-review-rating]").forEach(btn => btn.onclick = () => {
    const caseId = btn.dataset.caseId;
    setReview(caseId, {rating: btn.dataset.reviewRating || ""});
    render();
  });
  document.querySelectorAll("[data-review-issue]").forEach(btn => btn.onclick = () => {
    const caseId = btn.dataset.caseId;
    const issue = btn.dataset.reviewIssue;
    const old = reviewForCase(caseId);
    const issues = new Set(old.issues || []);
    if (issues.has(issue)) issues.delete(issue); else issues.add(issue);
    setReview(caseId, {issues: [...issues], rating: old.rating || "bad"});
    render();
  });
  document.querySelectorAll("textarea.review-note").forEach(box => box.onchange = () => {
    setReview(box.dataset.caseId, {note: box.value});
    renderSummary(selectedSamples().filter(s => passes(s, document.getElementById("searchBox").value.trim())).length);
  });
}
function reviewExportPayload(onlyBad) {
  const review = loadReview();
  const byCase = new Map(selectedSamples().map(sample => [sample.case_id, sample]));
  const items = Object.entries(review)
    .filter(([_caseId, item]) => !onlyBad || item.rating === "bad")
    .map(([caseId, item]) => {
      const base = byCase.get(caseId) || {};
      const targets = isCompareView()
        ? targetsForCase(caseId).map(({run, sample, status}) => ({
            run_id: run.run_id,
            run_label: run.label || run.run_id,
            status,
            target_audio: sample?.target_audio || "",
            target_path: sample?.target_path || ""
          }))
        : undefined;
      return targets ? {case_id: caseId, ...base, targets, ...item} : {case_id: caseId, ...base, ...item};
    });
  return {run_id: activeRun, run_label: activeRunLabel(), exported_at: new Date().toISOString(), items};
}
function downloadJSON(name, obj) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], {type:"application/json"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
document.getElementById("searchBox").addEventListener("input", render);
document.getElementById("exportBadBtn").onclick = () => downloadJSON(`${activeRun || "run"}.badcases.json`, reviewExportPayload(true));
document.getElementById("exportAllBtn").onclick = () => downloadJSON(`${activeRun || "run"}.review.json`, reviewExportPayload(false));
document.getElementById("clearReviewBtn").onclick = () => {
  if (confirm("Clear manual review for current run?")) {
    localStorage.removeItem(reviewStorageKey());
    render();
  }
};
render();
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a static listening page for SeedTTS validation inference outputs.")
    ap.add_argument("--validation-jsonl", default=str(DEFAULT_VALIDATION_JSONL))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--page-dir", default=str(DEFAULT_PAGE_DIR))
    ap.add_argument("--manifest-jsonl", action="append", default=[])
    ap.add_argument("--run-id", default="ver2_3")
    ap.add_argument("--run-label", default="ver2.3 ctc-clean final")
    ap.add_argument("--model-path", default=str(ROOT / "outputs/lora_runs/ver2_3_ctc_clean_textrep5_spm_lora_r16_a32_gbs64/final"))
    ap.add_argument("--append", action="store_true", help="Keep existing runs in benchmark_data.js and replace only --run-id.")
    args = ap.parse_args()

    validation_jsonl = Path(args.validation_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    page_dir = Path(args.page_dir).expanduser().resolve()
    if not args.manifest_jsonl:
        args.manifest_jsonl = [
            str(output_dir / "manifest.shard0.jsonl"),
            str(output_dir / "manifest.shard1.jsonl"),
            str(output_dir / "manifest.rerun_failed.jsonl"),
        ]

    validation_rows = list(iter_jsonl(validation_jsonl))
    page_dir.mkdir(parents=True, exist_ok=True)
    data_dir = page_dir / "data"
    copied_validation = copy_data_file(validation_jsonl, data_dir)
    copied_manifests = [copy_data_file(Path(p).expanduser().resolve(), data_dir) for p in args.manifest_jsonl]

    data_path = page_dir / "benchmark_data.js"
    payload = load_existing_payload(data_path) if args.append else None
    if payload is None:
        payload = {"dataset": build_dataset(validation_rows, validation_jsonl), "runs": []}
    else:
        payload["dataset"] = build_dataset(validation_rows, validation_jsonl)

    run = build_run(args, validation_rows, page_dir)
    run["downloads"] = {
        "validation_jsonl": copied_validation,
        "manifest_jsonl": [x for x in copied_manifests if x],
    }
    payload["runs"] = [r for r in payload.get("runs", []) if r.get("run_id") != args.run_id]
    payload["runs"].append(run)

    data_js = "window.SEEDTTS_BENCHMARK = " + json.dumps(payload, ensure_ascii=False, indent=2) + ";\n"
    data_path.write_text(data_js, encoding="utf-8")
    (page_dir / "index.html").write_text(HTML, encoding="utf-8")
    print(f"wrote page: {page_dir / 'index.html'}")
    print(f"runs: {', '.join(r.get('run_id', '') for r in payload['runs'])}")
    print(f"samples: {len(run['samples'])}")
    print(f"status: {run['counts']['status']}")
    print(f"missing_audio_links: {run['counts']['missing_audio_links']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

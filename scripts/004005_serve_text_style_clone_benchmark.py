#!/usr/bin/env python3
"""Serve an interactive listening UI for text style-clone benchmark outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
import socket
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


DEFAULT_INDEX = (
    "outputs/text_style_clone_benchmark_20260626/listening_index.jsonl"
)
BACKEND_ORDER = [
    "moss_tts",
    "f5_tts",
    "cosyvoice2_zero_shot",
    "cosyvoice3_zero_shot",
    "cosyvoice3",
    "styletts2",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def short_hash(*parts: str) -> str:
    key = "\n".join(parts).encode("utf-8")
    return hashlib.sha1(key).hexdigest()[:12]


def build_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    audio_map: dict[str, Path] = {}
    backend_counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}

    for row in rows:
        language = row.get("language") or "unknown"
        source_style_wav = row.get("source_style_wav") or ""
        timbre_ref_wav = row.get("timbre_ref_wav") or ""
        input_text = row.get("input_text") or ""
        group_id = short_hash(language, source_style_wav, timbre_ref_wav, input_text)
        group = grouped.setdefault(
            group_id,
            {
                "group_id": group_id,
                "language": language,
                "input_text": input_text,
                "source_style_text": row.get("source_style_text") or "",
                "source_style_wav": source_style_wav,
                "source_audio_url": f"audio/{group_id}/source",
                "timbre_ref_wav": timbre_ref_wav,
                "timbre_audio_url": f"audio/{group_id}/timbre",
                "outputs": [],
            },
        )
        backend = row.get("backend") or "unknown"
        sample_id = row.get("sample_id") or f"{backend}_{group_id}"
        carrier_wav = row.get("style_carrier_wav") or ""
        output_key = f"{group_id}:carrier:{backend}"
        group["outputs"].append(
            {
                "backend": backend,
                "sample_id": sample_id,
                "status": row.get("status") or "",
                "style_carrier_wav": carrier_wav,
                "audio_url": f"audio/{group_id}/carrier/{backend}",
            }
        )

        audio_map[f"{group_id}:source"] = Path(source_style_wav)
        audio_map[f"{group_id}:timbre"] = Path(timbre_ref_wav)
        audio_map[output_key] = Path(carrier_wav)
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
        language_counts[language] = language_counts.get(language, 0) + 1

    groups = list(grouped.values())
    for idx, group in enumerate(sorted(groups, key=group_sort_key), 1):
        group["display_index"] = idx
        group["outputs"] = sorted(
            group["outputs"],
            key=lambda item: (
                BACKEND_ORDER.index(item["backend"])
                if item["backend"] in BACKEND_ORDER
                else len(BACKEND_ORDER),
                item["backend"],
            ),
        )
    groups = sorted(groups, key=group_sort_key)
    return {
        "groups": groups,
        "audio_map": audio_map,
        "summary": {
            "rows": len(rows),
            "groups": len(groups),
            "backend_counts": backend_counts,
            "language_counts": language_counts,
            "backends": sorted(backend_counts),
            "languages": sorted(language_counts),
        },
    }


def group_sort_key(group: dict[str, Any]) -> tuple[str, str]:
    return (str(group.get("language") or ""), str(group.get("input_text") or ""))


def find_free_port(start: int = 18600, end: int = 18750) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free port found in {start}-{end}")


APP_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Text Style Clone Benchmark</title>
<style>
:root {
  --bg: #f4f5f2;
  --panel: #ffffff;
  --panel-2: #f9faf8;
  --ink: #20242a;
  --muted: #667085;
  --line: #d8ddd2;
  --green: #2e7d55;
  --blue: #2563a9;
  --red: #b13a2f;
  --gold: #9a6a12;
  --shadow: 0 10px 28px rgba(24, 32, 38, 0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
  letter-spacing: 0;
}
.app {
  min-height: 100vh;
  display: grid;
  grid-template-columns: minmax(300px, 360px) minmax(0, 1fr);
}
.sidebar {
  border-right: 1px solid var(--line);
  background: #fbfcfa;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}
.title {
  padding: 18px 18px 14px;
  border-bottom: 1px solid var(--line);
}
h1 {
  margin: 0 0 12px;
  font-size: 20px;
  line-height: 1.2;
  font-weight: 760;
}
.stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
}
.stat {
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 8px;
  padding: 8px;
}
.stat b { display: block; font-size: 18px; line-height: 1.1; }
.stat span { color: var(--muted); font-size: 12px; }
.controls {
  padding: 12px 18px;
  border-bottom: 1px solid var(--line);
  display: grid;
  gap: 10px;
}
.control-row { display: grid; gap: 6px; }
label {
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}
input, select {
  width: 100%;
  height: 36px;
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 7px;
  padding: 0 10px;
  color: var(--ink);
  font-size: 14px;
}
.sample-list {
  overflow: auto;
  padding: 10px;
  display: grid;
  gap: 8px;
}
.sample-button {
  width: 100%;
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 8px;
  padding: 10px;
  text-align: left;
  color: var(--ink);
  cursor: pointer;
}
.sample-button:hover { border-color: #aab5a5; }
.sample-button.active {
  outline: 2px solid var(--green);
  border-color: transparent;
}
.sample-top {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  align-items: center;
  margin-bottom: 6px;
}
.sample-name { font-weight: 760; font-size: 13px; }
.pill {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 2px 8px;
  color: var(--muted);
  font-size: 12px;
  white-space: nowrap;
}
.sample-text {
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
  overflow: hidden;
  color: #3f4650;
  font-size: 13px;
  line-height: 1.35;
}
.main {
  min-width: 0;
  padding: 22px;
  overflow: auto;
}
.toolbar {
  display: flex;
  gap: 10px;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
}
.selection h2 {
  margin: 0;
  font-size: 22px;
  line-height: 1.2;
}
.selection p {
  margin: 5px 0 0;
  color: var(--muted);
  font-size: 13px;
}
.backend-tabs {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  justify-content: flex-end;
}
.backend-tab {
  border: 1px solid var(--line);
  background: var(--panel);
  color: var(--ink);
  border-radius: 999px;
  padding: 7px 11px;
  font-size: 13px;
  cursor: pointer;
}
.backend-tab.off {
  color: #9aa1a9;
  background: #eef0ec;
}
.text-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 12px;
  margin-bottom: 14px;
}
.text-panel, .audio-panel, .result-panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
}
.text-panel {
  padding: 14px;
  min-height: 116px;
}
.text-panel h3, .audio-panel h3, .result-panel h3 {
  margin: 0 0 8px;
  font-size: 13px;
  color: var(--muted);
  text-transform: uppercase;
}
.text-content {
  font-size: 18px;
  line-height: 1.45;
  overflow-wrap: anywhere;
}
.reference-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 14px;
}
.audio-panel {
  padding: 12px;
  min-width: 0;
}
.path {
  margin-top: 8px;
  color: var(--muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;
  overflow-wrap: anywhere;
}
audio {
  width: 100%;
  height: 34px;
  display: block;
}
canvas {
  display: block;
  width: 100%;
  height: 56px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel-2);
  margin: 8px 0;
}
.results {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 12px;
}
.result-panel {
  padding: 12px;
  min-width: 0;
}
.result-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 8px;
}
.result-title {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}
.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  flex: 0 0 auto;
}
.backend {
  font-weight: 760;
  white-space: nowrap;
}
.sample-id {
  color: var(--muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.status {
  font-size: 12px;
  color: var(--muted);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 2px 8px;
  flex: 0 0 auto;
}
.empty {
  border: 1px dashed #b9c0b5;
  border-radius: 8px;
  padding: 30px;
  color: var(--muted);
  text-align: center;
  background: rgba(255,255,255,.6);
}
@media (max-width: 900px) {
  .app { grid-template-columns: 1fr; }
  .sidebar { min-height: auto; border-right: 0; border-bottom: 1px solid var(--line); }
  .sample-list { max-height: 280px; }
  .main { padding: 16px; }
  .text-grid, .reference-grid { grid-template-columns: 1fr; }
  .toolbar { align-items: flex-start; flex-direction: column; }
  .backend-tabs { justify-content: flex-start; }
}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="title">
      <h1>Text Style Clone Benchmark</h1>
      <div class="stats">
        <div class="stat"><b id="statGroups">0</b><span>Samples</span></div>
        <div class="stat"><b id="statRows">0</b><span>Wavs</span></div>
        <div class="stat"><b id="statBackends">0</b><span>Backends</span></div>
      </div>
    </div>
    <div class="controls">
      <div class="control-row">
        <label for="languageFilter">Language</label>
        <select id="languageFilter"></select>
      </div>
      <div class="control-row">
        <label for="queryFilter">Search</label>
        <input id="queryFilter" type="search" placeholder="Text or sample id">
      </div>
    </div>
    <div id="sampleList" class="sample-list"></div>
  </aside>
  <main class="main">
    <div id="content" class="empty">Loading</div>
  </main>
</div>
<script>
const colors = {
  moss_tts: "#2e7d55",
  f5_tts: "#2563a9",
  cosyvoice2_zero_shot: "#6f43b5",
  cosyvoice3_zero_shot: "#b85c00",
  cosyvoice3: "#9a6a12",
  styletts2: "#b13a2f"
};
const state = {
  data: null,
  groups: [],
  selectedId: null,
  activeBackends: new Set(),
  query: "",
  language: "all"
};

function $(id) { return document.getElementById(id); }
function esc(text) {
  return String(text ?? "").replace(/[&<>"']/g, ch => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[ch]));
}
function backendColor(name) { return colors[name] || "#536471"; }

async function loadData() {
  const response = await fetch("api/data", {cache: "no-store"});
  if (!response.ok) throw new Error(await response.text());
  state.data = await response.json();
  state.groups = state.data.groups;
  for (const backend of state.data.summary.backends) state.activeBackends.add(backend);
  state.selectedId = state.groups[0]?.group_id || null;
  initializeControls();
  render();
}

function initializeControls() {
  $("statGroups").textContent = state.data.summary.groups;
  $("statRows").textContent = state.data.summary.rows;
  $("statBackends").textContent = state.data.summary.backends.length;
  const languageFilter = $("languageFilter");
  languageFilter.innerHTML = '<option value="all">All</option>' +
    state.data.summary.languages.map(lang => `<option value="${esc(lang)}">${esc(lang)}</option>`).join("");
  languageFilter.addEventListener("change", () => {
    state.language = languageFilter.value;
    const first = filteredGroups()[0];
    state.selectedId = first ? first.group_id : null;
    render();
  });
  $("queryFilter").addEventListener("input", event => {
    state.query = event.target.value.trim().toLowerCase();
    const groups = filteredGroups();
    if (!groups.some(group => group.group_id === state.selectedId)) {
      state.selectedId = groups[0]?.group_id || null;
    }
    render();
  });
}

function filteredGroups() {
  return state.groups.filter(group => {
    if (state.language !== "all" && group.language !== state.language) return false;
    if (!state.query) return true;
    const haystack = [
      group.display_index,
      group.language,
      group.input_text,
      group.source_style_text,
      group.source_style_wav,
      group.timbre_ref_wav,
      ...group.outputs.flatMap(item => [item.backend, item.sample_id, item.style_carrier_wav])
    ].join(" ").toLowerCase();
    return haystack.includes(state.query);
  });
}

function render() {
  renderSampleList();
  renderContent();
}

function renderSampleList() {
  const groups = filteredGroups();
  const list = $("sampleList");
  if (!groups.length) {
    list.innerHTML = '<div class="empty">No matching samples</div>';
    return;
  }
  list.innerHTML = groups.map(group => `
    <button class="sample-button ${group.group_id === state.selectedId ? "active" : ""}" data-id="${esc(group.group_id)}">
      <div class="sample-top">
        <span class="sample-name">Sample ${group.display_index}</span>
        <span class="pill">${esc(group.language)} · ${group.outputs.length}</span>
      </div>
      <div class="sample-text">${esc(group.input_text)}</div>
    </button>
  `).join("");
  list.querySelectorAll(".sample-button").forEach(button => {
    button.addEventListener("click", () => {
      state.selectedId = button.dataset.id;
      render();
    });
  });
}

function renderContent() {
  const group = state.groups.find(item => item.group_id === state.selectedId);
  const content = $("content");
  if (!group) {
    content.className = "empty";
    content.textContent = "No sample selected";
    return;
  }
  content.className = "";
  const outputs = group.outputs.filter(item => state.activeBackends.has(item.backend));
  content.innerHTML = `
    <div class="toolbar">
      <div class="selection">
        <h2>Sample ${group.display_index} <span class="pill">${esc(group.language)}</span></h2>
        <p>${esc(group.group_id)}</p>
      </div>
      <div class="backend-tabs">
        ${state.data.summary.backends.map(backend => `
          <button class="backend-tab ${state.activeBackends.has(backend) ? "" : "off"}" data-backend="${esc(backend)}">
            ${esc(backend)}
          </button>
        `).join("")}
      </div>
    </div>
    <section class="text-grid">
      <div class="text-panel">
        <h3>Input Text</h3>
        <div class="text-content">${esc(group.input_text)}</div>
      </div>
      <div class="text-panel">
        <h3>Source Style Text</h3>
        <div class="text-content">${esc(group.source_style_text)}</div>
      </div>
    </section>
    <section class="reference-grid">
      ${audioPanel("Source Style", group.source_audio_url, group.source_style_wav, "source")}
      ${audioPanel("Timbre Reference", group.timbre_audio_url, group.timbre_ref_wav, "timbre")}
    </section>
    <section class="results">
      ${outputs.map(item => resultPanel(item)).join("") || '<div class="empty">No active backend</div>'}
    </section>
  `;
  content.querySelectorAll(".backend-tab").forEach(button => {
    button.addEventListener("click", () => {
      const backend = button.dataset.backend;
      if (state.activeBackends.has(backend)) {
        state.activeBackends.delete(backend);
      } else {
        state.activeBackends.add(backend);
      }
      renderContent();
    });
  });
  drawVisibleWaveforms(content);
}

function audioPanel(title, url, path, key) {
  return `
    <div class="audio-panel">
      <h3>${esc(title)}</h3>
      <canvas data-audio-url="${esc(url)}" data-wave-key="${esc(key)}"></canvas>
      <audio controls preload="metadata" src="${esc(url)}"></audio>
      <div class="path">${esc(path)}</div>
    </div>
  `;
}

function resultPanel(item) {
  return `
    <div class="result-panel">
      <div class="result-head">
        <div class="result-title">
          <span class="dot" style="background:${backendColor(item.backend)}"></span>
          <div style="min-width:0">
            <div class="backend">${esc(item.backend)}</div>
            <div class="sample-id">${esc(item.sample_id)}</div>
          </div>
        </div>
        <span class="status">${esc(item.status || "ok")}</span>
      </div>
      <canvas data-audio-url="${esc(item.audio_url)}" data-wave-key="${esc(item.sample_id)}"></canvas>
      <audio controls preload="metadata" src="${esc(item.audio_url)}"></audio>
      <div class="path">${esc(item.style_carrier_wav)}</div>
    </div>
  `;
}

async function drawVisibleWaveforms(root) {
  const canvases = [...root.querySelectorAll("canvas[data-audio-url]")];
  for (const canvas of canvases) {
    drawPlaceholder(canvas);
    try {
      const response = await fetch(canvas.dataset.audioUrl);
      const buffer = await response.arrayBuffer();
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const decoded = await audioContext.decodeAudioData(buffer.slice(0));
      drawWaveform(canvas, decoded);
      audioContext.close();
    } catch (error) {
      drawPlaceholder(canvas, true);
    }
  }
}

function drawPlaceholder(canvas, muted=false) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.strokeStyle = muted ? "#c6cbc2" : "#d8ddd2";
  ctx.beginPath();
  ctx.moveTo(0, rect.height / 2);
  ctx.lineTo(rect.width, rect.height / 2);
  ctx.stroke();
}

function drawWaveform(canvas, audioBuffer) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, rect.width, rect.height);
  const data = audioBuffer.getChannelData(0);
  const step = Math.max(1, Math.floor(data.length / rect.width));
  const mid = rect.height / 2;
  ctx.fillStyle = "#eef2eb";
  ctx.fillRect(0, 0, rect.width, rect.height);
  ctx.strokeStyle = "#2e7d55";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < rect.width; x++) {
    let min = 1, max = -1;
    const start = x * step;
    for (let i = 0; i < step && start + i < data.length; i++) {
      const value = data[start + i];
      if (value < min) min = value;
      if (value > max) max = value;
    }
    ctx.moveTo(x, mid + min * mid * 0.9);
    ctx.lineTo(x, mid + max * mid * 0.9);
  }
  ctx.stroke();
}

loadData().catch(error => {
  const content = $("content");
  content.className = "empty";
  content.textContent = error.message;
});
</script>
</body>
</html>
"""


class BenchmarkServer(BaseHTTPRequestHandler):
    payload: dict[str, Any] = {}
    audio_map: dict[str, Path] = {}

    def do_GET(self) -> None:
        route = self.normalized_path()
        if route in {"", "/"}:
            self.send_bytes(
                APP_HTML.encode("utf-8"),
                "text/html; charset=utf-8",
                cache=False,
            )
            return
        if route == "/api/data":
            data = {
                "groups": self.payload["groups"],
                "summary": self.payload["summary"],
            }
            self.send_bytes(
                json.dumps(data, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
                cache=False,
            )
            return
        if route.startswith("/audio/"):
            self.serve_audio(route)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")

    def normalized_path(self) -> str:
        parsed = urlparse(self.path)
        route = unquote(parsed.path)
        match = re.search(r"/proxy/\d+(/.*)?$", route)
        if match:
            route = match.group(1) or "/"
        if not route.startswith("/"):
            route = "/" + route
        return route

    def serve_audio(self, route: str) -> None:
        parts = route.strip("/").split("/")
        if len(parts) < 3:
            self.send_error(HTTPStatus.BAD_REQUEST, "Bad audio route")
            return
        group_id = parts[1]
        role = parts[2]
        if role == "carrier" and len(parts) >= 4:
            key = f"{group_id}:carrier:{parts[3]}"
        elif role in {"source", "timbre"}:
            key = f"{group_id}:{role}"
        else:
            self.send_error(HTTPStatus.BAD_REQUEST, "Bad audio route")
            return
        path = self.audio_map.get(key)
        if path is None or not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Audio not found")
            return
        self.send_file(path)

    def send_file(self, path: Path) -> None:
        size = path.stat().st_size
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        range_header = self.headers.get("Range")
        start, end = 0, size - 1
        status = HTTPStatus.OK
        if range_header:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if match:
                if match.group(1):
                    start = int(match.group(1))
                if match.group(2):
                    end = int(match.group(2))
                end = min(end, size - 1)
                if start > end or start >= size:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                status = HTTPStatus.PARTIAL_CONTENT

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()

        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def send_bytes(self, body: bytes, content_type: str, cache: bool = True) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-jsonl",
        type=Path,
        default=Path(DEFAULT_INDEX),
        help="Path to listening_index.jsonl.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=0, help="0 chooses a free port.")
    parser.add_argument("--port-start", type=int, default=18600)
    parser.add_argument("--port-end", type=int, default=18750)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_jsonl = args.index_jsonl.resolve()
    rows = read_jsonl(index_jsonl)
    payload = build_payload(rows)
    BenchmarkServer.payload = {
        "groups": payload["groups"],
        "summary": payload["summary"],
    }
    BenchmarkServer.audio_map = payload["audio_map"]

    port = args.port or find_free_port(args.port_start, args.port_end)
    server = ThreadingHTTPServer((args.host, port), BenchmarkServer)
    print(f"Serving {index_jsonl}")
    print(f"Listening on http://{args.host}:{port}/")
    print(f"Rows={payload['summary']['rows']} groups={payload['summary']['groups']}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

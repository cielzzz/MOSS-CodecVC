#!/usr/bin/env python3
"""Build a combined listening page for v2 pilot Case A/B/C evals."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
from pathlib import Path
from typing import Any


CASE_RUNS = [
    (
        "A",
        "Case A: old SeedTTS quick20",
        "v1_v2pilot_r1_caseA_oldquick20_step-1500_quick20_d2d3_seed1234",
    ),
    (
        "B",
        "Case B: v2 same-episode quick20",
        "v1_v2pilot_r1_caseB_v2quick20_step-1500_quick20_d2d3_seed1234",
    ),
    (
        "C",
        "Case C: v2 heldout cross-channel quick20",
        "v1_v2pilot_r1_caseC_v2heldout_quick20_step-1500_quick20_d2d3_seed1234",
    ),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def index_by_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        case_id = row.get("case_id") or row.get("sample_id")
        if case_id:
            out[str(case_id)] = row
    return out


def fmt_float(value: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def safe_name(text: str, max_len: int = 96) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text[:max_len] or "audio"


def resolve_path(path_value: Any, repo_root: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return repo_root / path


def link_audio(
    path_value: Any,
    *,
    role: str,
    case_key: str,
    idx: int,
    case_id: str,
    repo_root: Path,
    out_dir: Path,
    assets_dir: Path,
) -> dict[str, Any]:
    path = resolve_path(path_value, repo_root)
    if path is None:
        return {"href": "", "path": "", "exists": False}
    exists = path.exists()
    suffix = path.suffix if path.suffix else ".wav"
    filename = f"{case_key}_{idx:02d}_{role}_{safe_name(case_id)}{suffix}"
    dest = assets_dir / filename
    if exists:
        if dest.exists() or dest.is_symlink():
            current_target = None
            if dest.is_symlink():
                try:
                    current_target = os.readlink(dest)
                except OSError:
                    current_target = None
            if current_target != str(path):
                dest.unlink()
        if not dest.exists():
            os.symlink(str(path), str(dest))
    return {
        "href": str(dest.relative_to(out_dir)) if exists else "",
        "path": str(path),
        "exists": exists,
    }


def pick(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return ""


def load_case(case_key: str, title: str, run_name: str, root: Path, repo_root: Path, out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_dir = root / run_name
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    manifest_rows: list[dict[str, Any]] = []
    for shard in sorted(run_dir.glob("manifest.shard*.jsonl")):
        manifest_rows.extend(read_jsonl(shard))
    manifest_by_case = index_by_case(manifest_rows)

    asr_rows = read_jsonl(run_dir / f"{run_name}.asr_eval.jsonl")
    asr_by_case = index_by_case(asr_rows)
    sim_by_case = index_by_case(read_csv(run_dir / f"{run_name}.speaker_sim.csv"))
    metrics_by_case = index_by_case(read_csv(run_dir / f"{run_name}.metrics.csv"))

    summary = read_json(run_dir / f"{run_name}.summary.json")
    sim_summary = read_json(run_dir / f"{run_name}.speaker_sim_summary.json")

    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, manifest in enumerate(manifest_rows):
        case_id = str(manifest.get("case_id") or manifest.get("sample_id") or f"{case_key}_{idx:02d}")
        asr = asr_by_case.get(case_id, {})
        sim = sim_by_case.get(case_id, {})
        metrics = metrics_by_case.get(case_id, {})

        generated_path = pick(manifest.get("output_wav"), asr.get("target_audio"), sim.get("target_audio"))
        original_target_path = manifest.get("target_audio")
        source_path = pick(manifest.get("source_audio"), asr.get("source_audio"), sim.get("source_audio"))
        ref_path = pick(manifest.get("timbre_ref_audio"), asr.get("timbre_ref_audio"), sim.get("timbre_ref_audio"))

        gen_struct = manifest.get("generation_structure") or {}
        stop_stats = manifest.get("progress_stop_infer_stats") or {}
        ref_perm = manifest.get("ref_prompt_codec_permutation") or {}

        row = {
            "case_key": case_key,
            "case_title": title,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "idx": idx,
            "case_id": case_id,
            "cell": pick(asr.get("cell"), manifest.get("cell"), sim.get("cell")),
            "language": pick(asr.get("language"), manifest.get("language")),
            "status": pick(manifest.get("status"), asr.get("manifest_status"), sim.get("status")),
            "content_ref_text": pick(asr.get("content_ref_text"), manifest.get("content_ref_text"), asr.get("target_text")),
            "source_text": pick(asr.get("source_content_text"), manifest.get("source_content_text"), asr.get("source_text")),
            "timbre_ref_text": pick(asr.get("timbre_ref_text"), manifest.get("timbre_ref_text")),
            "asr_hyp": pick(asr.get("asr_tgt_text"), metrics.get("asr_tgt_text")),
            "cer": pick(asr.get("cer_tgt"), metrics.get("cer_tgt")),
            "wer": pick(asr.get("wer_tgt"), metrics.get("wer_tgt")),
            "repeat_score": pick(asr.get("repeat_score"), metrics.get("repeat_score")),
            "duration_ratio": pick(asr.get("duration_ratio_tgt_src"), metrics.get("duration_ratio_tgt_src")),
            "content_keep": str(pick(asr.get("content_keep"), metrics.get("content_keep"))),
            "filter_reason": pick(asr.get("content_filter_reason"), metrics.get("content_filter_reason")),
            "sim_ref": sim.get("sim_gen_ref", ""),
            "sim_source": sim.get("sim_gen_source", ""),
            "ref_bound": "",
            "source_frames": pick(ref_perm.get("source_frames"), ""),
            "prompt_frames": pick(ref_perm.get("prompt_frames"), ""),
            "gen_slots": pick(gen_struct.get("gen_slot_count"), ""),
            "delay_slots": pick(gen_struct.get("delay_slot_count"), ""),
            "dedelayed_lengths": gen_struct.get("dedelayed_segment_lengths") or [],
            "stop_prob_max": pick(stop_stats.get("stop_prob_max_max"), ""),
            "progress_max": pick(stop_stats.get("progress_value_max_max"), ""),
            "elapsed_sec": pick(manifest.get("elapsed_sec"), asr.get("elapsed_sec")),
            "source_audio": link_audio(
                source_path,
                role="source",
                case_key=case_key,
                idx=idx,
                case_id=case_id,
                repo_root=repo_root,
                out_dir=out_dir,
                assets_dir=assets_dir,
            ),
            "ref_audio": link_audio(
                ref_path,
                role="ref",
                case_key=case_key,
                idx=idx,
                case_id=case_id,
                repo_root=repo_root,
                out_dir=out_dir,
                assets_dir=assets_dir,
            ),
            "target_audio": link_audio(
                original_target_path,
                role="target",
                case_key=case_key,
                idx=idx,
                case_id=case_id,
                repo_root=repo_root,
                out_dir=out_dir,
                assets_dir=assets_dir,
            ),
            "generated_audio": link_audio(
                generated_path,
                role="generated",
                case_key=case_key,
                idx=idx,
                case_id=case_id,
                repo_root=repo_root,
                out_dir=out_dir,
                assets_dir=assets_dir,
            ),
        }
        try:
            row["ref_bound"] = bool(float(row["sim_ref"]) > float(row["sim_source"]))
        except (TypeError, ValueError):
            row["ref_bound"] = ""
        rows.append(row)

    card = {
        "case_key": case_key,
        "title": title,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "summary": summary.get("overall", {}),
        "speaker_summary": next(iter((sim_summary.get("runs") or {}).values()), {}).get("all", {}),
    }
    return rows, card


def audio_block(label: str, audio: dict[str, Any]) -> str:
    if audio.get("exists") and audio.get("href"):
        src = html.escape(audio["href"])
        path = html.escape(audio.get("path", ""))
        return f"""
          <div class="audio-block">
            <div class="audio-label">{html.escape(label)}</div>
            <audio controls preload="none" src="{src}"></audio>
            <div class="path" title="{path}">{path}</div>
          </div>
        """
    path = html.escape(audio.get("path", ""))
    return f"""
      <div class="audio-block missing">
        <div class="audio-label">{html.escape(label)}</div>
        <div class="missing-text">missing</div>
        <div class="path" title="{path}">{path}</div>
      </div>
    """


def metric_chip(label: str, value: Any, cls: str = "") -> str:
    value_text = html.escape(str(value)) if value not in (None, "") else "NA"
    return f'<span class="chip {cls}"><b>{html.escape(label)}</b>{value_text}</span>'


def row_html(row: dict[str, Any]) -> str:
    ref_bound_cls = "good" if row.get("ref_bound") is True else "warn"
    keep_cls = "good" if row.get("content_keep") == "True" else "bad"
    chips = [
        metric_chip("CER", fmt_float(row.get("cer")), "bad" if row.get("content_keep") != "True" else "good"),
        metric_chip("dur/src", fmt_float(row.get("duration_ratio"))),
        metric_chip("repeat", fmt_float(row.get("repeat_score"))),
        metric_chip("sim(ref)", fmt_float(row.get("sim_ref")), "good"),
        metric_chip("sim(src)", fmt_float(row.get("sim_source")), "warn"),
        metric_chip("ref-bound", row.get("ref_bound"), ref_bound_cls),
        metric_chip("keep", row.get("content_keep"), keep_cls),
        metric_chip("reason", row.get("filter_reason") or "keep"),
        metric_chip("src_fr", row.get("source_frames")),
        metric_chip("ref_fr", row.get("prompt_frames")),
        metric_chip("gen_slots", row.get("gen_slots")),
        metric_chip("stop_p", fmt_float(row.get("stop_prob_max"))),
    ]
    audio = "\n".join(
        [
            audio_block("source", row["source_audio"]),
            audio_block("timbre ref", row["ref_audio"]),
            audio_block("real target", row["target_audio"]),
            audio_block("generated", row["generated_audio"]),
        ]
    )
    dedelayed = ", ".join(map(str, row.get("dedelayed_lengths") or []))
    return f"""
      <article class="item" data-case="{html.escape(row['case_key'])}" data-keep="{html.escape(row.get('content_keep', ''))}">
        <header>
          <div>
            <div class="case-tag">{html.escape(row['case_key'])} #{row['idx']:02d} · {html.escape(row.get('cell') or '')} · {html.escape(row.get('language') or '')}</div>
            <h2>{html.escape(row['case_id'])}</h2>
          </div>
        </header>
        <div class="metrics">{"".join(chips)}</div>
        <div class="audio-grid">{audio}</div>
        <section class="texts">
          <div><h3>content ref / target text</h3><p>{html.escape(row.get('content_ref_text') or '')}</p></div>
          <div><h3>ASR hypothesis</h3><p>{html.escape(row.get('asr_hyp') or '')}</p></div>
          <div><h3>source text</h3><p>{html.escape(row.get('source_text') or '')}</p></div>
          <div><h3>timbre ref text</h3><p>{html.escape(row.get('timbre_ref_text') or '')}</p></div>
        </section>
        <details>
          <summary>Paths and generation stats</summary>
          <pre>{html.escape(json.dumps({
              "run_dir": row.get("run_dir"),
              "status": row.get("status"),
              "source_audio": row["source_audio"].get("path"),
              "timbre_ref_audio": row["ref_audio"].get("path"),
              "target_audio": row["target_audio"].get("path"),
              "generated_audio": row["generated_audio"].get("path"),
              "source_frames": row.get("source_frames"),
              "prompt_frames": row.get("prompt_frames"),
              "gen_slots": row.get("gen_slots"),
              "delay_slots": row.get("delay_slots"),
              "dedelayed_lengths": dedelayed,
              "progress_max": row.get("progress_max"),
              "elapsed_sec": row.get("elapsed_sec"),
          }, ensure_ascii=False, indent=2))}</pre>
        </details>
      </article>
    """


def summary_card(card: dict[str, Any]) -> str:
    overall = card.get("summary") or {}
    speaker = card.get("speaker_summary") or {}
    chips = [
        metric_chip("n", overall.get("n")),
        metric_chip("primary", fmt_float(overall.get("primary_error"), 4)),
        metric_chip("CER", fmt_float(overall.get("cer"), 4)),
        metric_chip("keep", f"{overall.get('keep', '')}/{overall.get('n', '')}"),
        metric_chip("dur", fmt_float(overall.get("duration"), 4)),
        metric_chip("sim(ref)", fmt_float(speaker.get("sim_gen_ref_mean"), 4)),
        metric_chip("sim(src)", fmt_float(speaker.get("sim_gen_source_mean"), 4)),
    ]
    return f"""
      <button class="summary-card" data-filter="{html.escape(card['case_key'])}">
        <div class="summary-title">{html.escape(card['title'])}</div>
        <div class="summary-run">{html.escape(card['run_name'])}</div>
        <div class="metrics mini">{"".join(chips)}</div>
      </button>
    """


def write_page(out_dir: Path, rows: list[dict[str, Any]], cards: list[dict[str, Any]], page_title: str) -> Path:
    rows_html = "\n".join(row_html(row) for row in rows)
    cards_html = "\n".join(summary_card(card) for card in cards)
    escaped_title = html.escape(page_title)
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #151923;
      --muted: #657084;
      --line: #d9dee8;
      --good: #0f7b45;
      --warn: #9a5b00;
      --bad: #b42318;
      --accent: #1955a6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }}
    header.page {{
      position: sticky;
      top: 0;
      z-index: 2;
      padding: 18px 24px 14px;
      background: rgba(246, 247, 249, 0.96);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(8px);
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .subtitle {{ margin: 0; color: var(--muted); font-size: 13px; }}
    .toolbar {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 14px;
    }}
    .toolbar button, .summary-card {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      border-radius: 6px;
      cursor: pointer;
    }}
    .toolbar button {{
      height: 34px;
      padding: 0 12px;
      font-weight: 650;
    }}
    .toolbar button.active {{ border-color: var(--accent); color: var(--accent); }}
    .toolbar input {{
      height: 34px;
      min-width: 260px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      background: white;
    }}
    main {{ padding: 18px 24px 40px; }}
    .summaries {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .summary-card {{
      padding: 12px;
      text-align: left;
      min-height: 120px;
    }}
    .summary-title {{ font-weight: 750; margin-bottom: 4px; }}
    .summary-run {{ color: var(--muted); font-size: 11px; overflow-wrap: anywhere; }}
    .item {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin: 0 0 14px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }}
    .item header {{
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }}
    .case-tag {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    h2 {{
      margin: 2px 0 0;
      font-size: 15px;
      overflow-wrap: anywhere;
      letter-spacing: 0;
    }}
    .metrics {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      margin: 8px 0 12px;
    }}
    .metrics.mini {{ margin-bottom: 0; }}
    .chip {{
      display: inline-flex;
      gap: 5px;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      white-space: nowrap;
      background: #f8fafc;
    }}
    .chip b {{ color: var(--muted); font-weight: 650; }}
    .chip.good {{ border-color: rgba(15, 123, 69, 0.35); color: var(--good); }}
    .chip.warn {{ border-color: rgba(154, 91, 0, 0.35); color: var(--warn); }}
    .chip.bad {{ border-color: rgba(180, 35, 24, 0.35); color: var(--bad); }}
    .audio-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(180px, 1fr));
      gap: 10px;
      margin: 10px 0 12px;
    }}
    .audio-block {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      min-width: 0;
      background: #fbfcfe;
    }}
    .audio-label {{
      font-size: 12px;
      font-weight: 750;
      margin-bottom: 6px;
      color: var(--accent);
    }}
    audio {{ width: 100%; height: 34px; }}
    .path {{
      margin-top: 6px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 10px;
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .missing-text {{
      height: 34px;
      display: grid;
      place-items: center;
      color: var(--bad);
      border: 1px dashed rgba(180, 35, 24, 0.35);
      border-radius: 4px;
      font-size: 12px;
    }}
    .texts {{
      display: grid;
      grid-template-columns: repeat(2, minmax(240px, 1fr));
      gap: 10px;
    }}
    .texts div {{
      border-top: 1px solid var(--line);
      padding-top: 8px;
      min-width: 0;
    }}
    h3 {{
      margin: 0 0 4px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    p {{ margin: 0; overflow-wrap: anywhere; }}
    details {{
      margin-top: 10px;
      border-top: 1px solid var(--line);
      padding-top: 8px;
    }}
    summary {{ cursor: pointer; color: var(--muted); font-size: 12px; font-weight: 700; }}
    pre {{
      overflow: auto;
      padding: 10px;
      background: #f1f4f8;
      border-radius: 6px;
      font-size: 11px;
      white-space: pre-wrap;
    }}
    .hidden {{ display: none; }}
    @media (max-width: 1000px) {{
      .audio-grid {{ grid-template-columns: repeat(2, minmax(180px, 1fr)); }}
      .texts {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 560px) {{
      main, header.page {{ padding-left: 12px; padding-right: 12px; }}
      .audio-grid {{ grid-template-columns: 1fr; }}
      .toolbar input {{ min-width: 100%; }}
    }}
  </style>
</head>
<body>
  <header class="page">
    <h1>{escaped_title}</h1>
    <p class="subtitle">Each row contains source, timbre ref, real target when available, generated output, ASR text, CER, duration ratio, repeat score, and ECAPA sim(ref/src).</p>
    <div class="toolbar">
      <button class="active" data-filter="ALL">All</button>
      <button data-filter="A">Case A</button>
      <button data-filter="B">Case B</button>
      <button data-filter="C">Case C</button>
      <button data-filter="FAIL">Non-keep only</button>
      <input id="search" placeholder="filter case id / text / ASR hyp">
    </div>
  </header>
  <main>
    <section class="summaries">{cards_html}</section>
    <section id="items">{rows_html}</section>
  </main>
  <script>
    const buttons = [...document.querySelectorAll('[data-filter]')];
    const search = document.getElementById('search');
    let active = 'ALL';
    function applyFilter() {{
      const q = (search.value || '').toLowerCase();
      for (const item of document.querySelectorAll('.item')) {{
        const caseKey = item.dataset.case;
        const keep = item.dataset.keep;
        const matchFilter = active === 'ALL' || active === caseKey || (active === 'FAIL' && keep !== 'True');
        const matchText = !q || item.innerText.toLowerCase().includes(q);
        item.classList.toggle('hidden', !(matchFilter && matchText));
      }}
      for (const btn of buttons) {{
        btn.classList.toggle('active', btn.dataset.filter === active);
      }}
    }}
    for (const btn of buttons) {{
      btn.addEventListener('click', () => {{
        active = btn.dataset.filter;
        applyFilter();
        window.scrollTo({{top: 0, behavior: 'smooth'}});
      }});
    }}
    search.addEventListener('input', applyFilter);
  </script>
</body>
</html>
"""
    out_path = out_dir / "index.html"
    out_path.write_text(page, encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("testset/outputs/ver2_9_v2pilot_r1_step1500_dual_quick_eval"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("testset/outputs/ver2_9_v2pilot_r1_step1500_dual_quick_eval/listening_page_abc"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case spec as KEY|Title|RunDirectoryName. Can be repeated.",
    )
    parser.add_argument("--page-title", default="v2 pilot r1 step-1500 Case A/B/C listening page")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    root = (repo_root / args.root).resolve() if not args.root.is_absolute() else args.root.resolve()
    out_dir = (repo_root / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    case_runs = CASE_RUNS
    if args.case:
        case_runs = []
        for spec in args.case:
            parts = spec.split("|", 2)
            if len(parts) != 3:
                raise ValueError(f"--case must be KEY|Title|RunDirectoryName, got: {spec}")
            case_runs.append((parts[0], parts[1], parts[2]))
    for case_key, title, run_name in case_runs:
        rows, card = load_case(case_key, title, run_name, root, repo_root, out_dir)
        all_rows.extend(rows)
        cards.append(card)

    page = write_page(out_dir, all_rows, cards, args.page_title)
    missing = 0
    for row in all_rows:
        for role in ("source_audio", "ref_audio", "target_audio", "generated_audio"):
            if not row[role].get("exists"):
                missing += 1
    print(f"wrote: {page}")
    print(f"rows: {len(all_rows)}")
    print(f"missing_audio_slots: {missing}")


if __name__ == "__main__":
    main()

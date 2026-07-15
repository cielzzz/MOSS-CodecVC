#!/usr/bin/env python3
"""Build three comparable Batch-44 v1 Best3-versus-Batch-33 blind20 pages.

The script consumes the 004103 Best3 selection and one full-320 diagnostics
CSV/run id for each selected candidate.  Each page uses its own disjoint 20
no_text cases so the repeated Batch-33 waveform cannot reveal the baseline
side across pages.  Each pair is gated on successful/content-safe output for
Batch-33 and that candidate, then stratified by Batch-33's dual-encoder binding
label: 3 ref-bound + 3 src-bound + 4 ambiguous + 10 random.

Each page independently randomizes exact 10/10 A/B positions and exposes only
Source, Reference and anonymous Candidate A/B.  The mapping manifests and
BLIND20_READY registry are written outside the listening webroot, so the
browser cannot retrieve them by guessing a sibling URL.  Choices are A, B,
tie, and neither.
This script is local-only and never submits a QZ task.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import secrets
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "moss_codecvc.batch44_v1_best3_blind20.v1"
SELECTION_SCHEMA = "moss_codecvc.batch44_v1_best3_selection.v1"
EXPERIMENT_ID = "batch44_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = (
    PROJECT_ROOT / "testset/outputs/batch44_best3_20260713/best3_selection.json"
)
DEFAULT_BATCH33_CSV = (
    PROJECT_ROOT
    / "testset/outputs/seedtts_baseline_dual_encoder_20260711_mtts/dual_encoder_cases.csv"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/listening_frontend/seedtts_valid_benchmark/batch44_best3_20260713"
)
DEFAULT_PRIVATE_ROOT = (
    PROJECT_ROOT / "testset/outputs/batch44_best3_20260713/private_blind20"
)
DEFAULT_BATCH33_RUN = "Batch33"
BUCKETS = ("ref-bound", "src-bound", "ambiguous")
BUCKET_COUNTS = {"ref-bound": 3, "src-bound": 3, "ambiguous": 4}
FULL320_VALIDATOR = Path(__file__).with_name(
    "004107_finalize_batch43_pathx_final.py"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def producer_registration() -> dict[str, str]:
    path = Path(__file__).resolve()
    return {
        "script": str(path),
        "script_sha256": sha256_file(path),
        "entrypoint": path.name,
    }


def load_full320_validator():
    if not FULL320_VALIDATOR.is_file():
        raise FileNotFoundError(f"missing full320 validator: {FULL320_VALIDATOR}")
    name = "moss_codecvc_batch44_blind20_full320_validator"
    spec = importlib.util.spec_from_file_location(name, FULL320_VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import full320 validator: {FULL320_VALIDATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "keep", "ok"}:
        return True
    if text in {"0", "false", "no", "drop", "fail"}:
        return False
    return None


def require_audio(value: Any, *, label: str) -> Path:
    path = Path(str(value or "")).expanduser().resolve(strict=False)
    if not path.is_file() or path.stat().st_size < 44:
        raise FileNotFoundError(f"missing/empty {label}: {path}")
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


def dual_label(row: Mapping[str, Any], *, margin: float) -> tuple[str, dict[str, Any]]:
    wavlm_ref = finite(row.get("sim_gen_ref"))
    wavlm_src = finite(row.get("sim_gen_source"))
    sb_ref = finite(row.get("ecapa_sim_gen_ref"))
    sb_src = finite(row.get("ecapa_sim_gen_source"))
    wavlm = classify(wavlm_ref, wavlm_src, margin)
    speechbrain = classify(sb_ref, sb_src, margin)
    if wavlm == speechbrain and wavlm in {"ref-bound", "src-bound"}:
        label = wavlm
        rule = "encoder_agreement"
    elif wavlm != "missing" and speechbrain != "missing":
        label = "ambiguous"
        rule = "ambiguous_or_encoder_disagreement"
    else:
        label = "missing"
        rule = "missing_encoder_metric"
    return label, {
        "wavlm_sim_ref": wavlm_ref,
        "wavlm_sim_src": wavlm_src,
        "wavlm_label": wavlm,
        "speechbrain_sim_ref": sb_ref,
        "speechbrain_sim_src": sb_src,
        "speechbrain_label": speechbrain,
        "dual_label": label,
        "dual_rule": rule,
    }


def load_diagnostics(
    path: Path, *, run_id: str, margin: float,
    expected_target_root: Path | None = None,
    expected_n: int | None = None,
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"diagnostics CSV does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in raw_rows
        if str(row.get("run") or row.get("run_id") or "") == run_id
        and str(row.get("mode") or "no_text") == "no_text"
    ]
    if not selected:
        raise ValueError(f"no no_text rows for run={run_id!r} in {path}")
    if expected_n is not None and len(selected) != expected_n:
        raise ValueError(
            f"{path}: run={run_id!r} has {len(selected)} rows, expected {expected_n}"
        )
    indexed: dict[str, dict[str, Any]] = {}
    for row in selected:
        case_id = str(row.get("case_id") or "")
        if not case_id:
            raise ValueError(f"empty case_id in {path}")
        if "/" in case_id or "\\" in case_id or case_id in {".", ".."}:
            raise ValueError(f"unsafe case_id={case_id!r} in {path}")
        if case_id in indexed:
            raise ValueError(f"duplicate case_id={case_id!r} in {path}")
        label, metrics = dual_label(row, margin=margin)
        generated_audio = require_audio(
            row.get("target_audio"), label=f"generated audio for {case_id}"
        )
        if expected_target_root is not None:
            expected_audio = (expected_target_root / f"{case_id}.wav").resolve()
            if generated_audio != expected_audio:
                raise ValueError(
                    f"{path}: {run_id}/{case_id} target_audio={generated_audio}, "
                    f"expected {expected_audio}"
                )
        indexed[case_id] = {
            "case_id": case_id,
            "cell": str(row.get("cell") or "unknown"),
            "content_keep": boolean(row.get("content_keep")),
            "cer": finite(row.get("cer_tgt")),
            "generated_audio": generated_audio,
            "source_audio": require_audio(
                row.get("source_audio"), label=f"source audio for {case_id}"
            ),
            "reference_audio": require_audio(
                row.get("timbre_ref_audio"), label=f"reference audio for {case_id}"
            ),
            "metrics": metrics,
        }
    return indexed


def default_completion_path(project_root: Path, step: int) -> Path:
    local = (
        project_root
        / "trainset/local_jobs"
        / f"ver23_batch44_paired_full320_step{step}_20260713/COMPLETED.json"
    ).resolve()
    qz = (
        project_root
        / "trainset/qz_jobs"
        / f"ver23_batch44_paired_full320_step{step}_20260713/COMPLETED.json"
    ).resolve()
    if local.is_file() and qz.is_file():
        raise ValueError(
            f"ambiguous full320 provenance for step-{step}: both local and QZ completions exist"
        )
    return local if local.is_file() or not qz.is_file() else qz


def bind_candidate_full320_evidence(
    *,
    candidate: Mapping[str, Any],
    diagnostics_csv: Path,
    run_id: str,
    completion_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Bind one blind candidate to the registered, completed full320 lane."""
    candidate_id = str(candidate.get("candidate_id") or "")
    arm = str(candidate.get("arm") or "")
    step = candidate.get("step")
    if arm not in {"r3", "r5"} or step not in {26000, 28000, 30000}:
        raise ValueError(f"{candidate_id}: invalid registered arm/step {arm!r}/{step!r}")
    if candidate_id != f"{arm}_step-{step}":
        raise ValueError(f"candidate identity drift: {candidate_id!r}")
    expected_completion = default_completion_path(project_root, int(step))
    completion_path = completion_path.expanduser().resolve()
    if completion_path != expected_completion:
        raise ValueError(
            f"{candidate_id}: completion={completion_path}, expected {expected_completion}"
        )
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    metrics_path = Path(str(completion.get("paired_metrics_json") or "")).resolve()
    dual_path = Path(str(completion.get("dual_encoder_cases_csv") or "")).resolve()
    diagnostics_csv = diagnostics_csv.expanduser().resolve()
    if diagnostics_csv != dual_path:
        raise ValueError(
            f"{candidate_id}: diagnostics CSV must be the COMPLETED dual_encoder_cases_csv; "
            f"got {diagnostics_csv}, expected {dual_path}"
        )
    validator = load_full320_validator()
    validated_completion, indexed = validator.validate_full320_step(
        step=int(step),
        completion_path=completion_path,
        metrics_path=metrics_path,
        project_root=project_root,
    )
    if validated_completion != completion:
        raise ValueError(f"{candidate_id}: COMPLETED.json changed during validation")
    expected_run = (
        f"ver2_9_5_final_{arm}_step-{step}_no_text_seedtts160_d2d3_seed1234"
    )
    if run_id != expected_run:
        raise ValueError(
            f"{candidate_id}: run_id={run_id!r}, expected registered lane {expected_run!r}"
        )
    objective_row = indexed.get((arm, "no_text"))
    if not isinstance(objective_row, dict):
        raise ValueError(f"{candidate_id}: paired_metrics lacks registered no_text row")
    step_root = Path(str(completion.get("step_root") or "")).resolve()
    target_root = (step_root / "runs" / expected_run).resolve()
    return {
        "candidate_id": candidate_id,
        "arm": arm,
        "step": int(step),
        "run_id": expected_run,
        "completion_json": str(completion_path),
        "completion_sha256": sha256_file(completion_path),
        "paired_metrics_json": str(metrics_path),
        "paired_metrics_sha256": sha256_file(metrics_path),
        "dual_encoder_cases_csv": str(dual_path),
        "dual_encoder_cases_sha256": sha256_file(dual_path),
        "target_audio_root": str(target_root),
        "objective_no_text": objective_row,
    }


def parse_binding(raw: str, *, option: str) -> tuple[str, str]:
    key, separator, value = raw.partition("=")
    key, value = key.strip(), value.strip()
    if not separator or not key or not value:
        raise ValueError(f"{option} must use CANDIDATE=VALUE, got {raw!r}")
    return key, value


def load_selection(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Best3 selection does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SELECTION_SCHEMA:
        raise ValueError(f"wrong Best3 selection schema: {payload.get('schema_version')!r}")
    if payload.get("experiment_id") != EXPERIMENT_ID or payload.get("data_version") != "v1_20260709":
        raise ValueError("Best3 selection is not the registered Batch-44 v1 experiment")
    if payload.get("status") != "selected":
        raise ValueError(f"Best3 selection status must be selected, got {payload.get('status')!r}")
    selected = payload.get("selected_candidate_ids")
    if not isinstance(selected, list) or len(selected) != 3 or len(set(selected)) != 3:
        raise ValueError(f"selection must contain exactly three unique candidates: {selected!r}")
    known = {
        str(row.get("candidate_id"))
        for row in payload.get("candidates", [])
        if isinstance(row, dict) and row.get("selected_for_full320") is True
    }
    if set(selected) != known:
        raise ValueError(f"selected_candidate_ids disagree with ranked candidates: {selected!r} vs {known!r}")
    return payload, [str(item) for item in selected]


def common_eligible(
    candidate_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    batch33: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    common_ids = set(batch33)
    for rows in candidate_rows.values():
        common_ids &= set(rows)
    skipped: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    for case_id in sorted(common_ids):
        baseline = batch33[case_id]
        systems = {candidate_id: rows[case_id] for candidate_id, rows in candidate_rows.items()}
        all_rows = {"batch33": baseline, **systems}
        rejected = False
        for system_id, row in all_rows.items():
            if row.get("content_keep") is not True:
                skipped[f"{system_id}:content_not_keep"] += 1
                rejected = True
                break
            cer = row.get("cer")
            if cer is None or float(cer) > 0.30:
                skipped[f"{system_id}:cer_over_0p30_or_missing"] += 1
                rejected = True
                break
            if row["metrics"]["dual_label"] == "missing":
                skipped[f"{system_id}:missing_dual_metric"] += 1
                rejected = True
                break
        if rejected:
            continue
        source = baseline["source_audio"]
        reference = baseline["reference_audio"]
        for system_id, row in systems.items():
            if row["source_audio"].resolve() != source.resolve():
                raise ValueError(f"{case_id}: source mismatch for {system_id}")
            if row["reference_audio"].resolve() != reference.resolve():
                raise ValueError(f"{case_id}: reference mismatch for {system_id}")
        eligible.append(
            {
                "case_id": case_id,
                "cell": baseline["cell"],
                "bucket": baseline["metrics"]["dual_label"],
                "source_audio": source,
                "reference_audio": reference,
                "batch33": baseline,
                "candidates": systems,
            }
        )
    return eligible, skipped


def diverse_pick(
    rows: list[dict[str, Any]], count: int, *, rng: random.Random
) -> list[dict[str, Any]]:
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[str(row["cell"])].append(row)
    cells = sorted(by_cell)
    rng.shuffle(cells)
    for cell in cells:
        by_cell[cell].sort(key=lambda item: item["case_id"])
        rng.shuffle(by_cell[cell])
    result: list[dict[str, Any]] = []
    active = cells
    while active and len(result) < count:
        next_active: list[str] = []
        for cell in active:
            if by_cell[cell] and len(result) < count:
                result.append(by_cell[cell].pop())
            if by_cell[cell]:
                next_active.append(cell)
        active = next_active
    return result


def select_cases(
    eligible: list[dict[str, Any]], *, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    pool_counts = Counter(row["bucket"] for row in eligible)
    for bucket in BUCKETS:
        pool = [row for row in eligible if row["bucket"] == bucket]
        wanted = BUCKET_COUNTS[bucket]
        chosen = diverse_pick(pool, wanted, rng=random.Random(f"{seed}:{bucket}"))
        if len(chosen) != wanted:
            raise ValueError(f"insufficient common {bucket} pool: {len(pool)}")
        for row in chosen:
            row = dict(row)
            row["selection_bucket"] = bucket
            selected.append(row)
            used.add(row["case_id"])
    remaining = [row for row in eligible if row["case_id"] not in used]
    random_rows = diverse_pick(remaining, 10, rng=random.Random(f"{seed}:random"))
    if len(random_rows) != 10:
        raise ValueError(f"insufficient random common pool: {len(remaining)}")
    for row in random_rows:
        row = dict(row)
        row["selection_bucket"] = "random"
        selected.append(row)
        used.add(row["case_id"])
    if len(selected) != 20 or len(used) != 20:
        raise AssertionError("blind20 selection is not exactly 20 unique cases")
    rng.shuffle(selected)
    return selected, {
        "strategy": "pairwise_disjoint_batch33_dual_encoder_3_3_4_plus_random10",
        "content_gate": "Batch-33 and all Best3 require content_keep=True and CER<=0.30",
        "pool_count": len(eligible),
        "pool_by_batch33_dual_label": dict(pool_counts),
        "selected_by_bucket": dict(Counter(row["selection_bucket"] for row in selected)),
        "selected_cells": dict(Counter(row["cell"] for row in selected)),
    }


def suffix(path: Path) -> str:
    value = path.suffix.lower()
    return value if value in {".wav", ".flac", ".mp3", ".m4a", ".ogg"} else ".wav"


def symlink_audio(source: Path, destination: Path, *, root: Path, force: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        same = destination.is_symlink() and destination.resolve(strict=False) == source.resolve()
        if not same:
            if not force:
                raise FileExistsError(f"conflicting staged asset: {destination}")
            destination.unlink()
    if not os.path.lexists(destination):
        os.symlink(str(source.resolve()), str(destination))
    return destination.relative_to(root).as_posix()


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Batch-44 v1 双候选盲听</title><style>
body{margin:0;background:#f4f6f8;color:#172033;font-family:system-ui,-apple-system,"PingFang SC",sans-serif}.top{position:sticky;top:0;background:#f4f6f8ee;border-bottom:1px solid #d8dee8;padding:15px 20px;z-index:3}h1{font-size:20px;margin:0 0 5px}.sub{font-size:13px;color:#667085}.bar{margin-top:10px;display:flex;gap:10px;align-items:center}.progress{font-weight:700;color:#067647}button{border:1px solid #cbd5e1;background:#fff;border-radius:7px;padding:7px 10px;cursor:pointer}button.active{border-color:#175cd3;background:#eff4ff;color:#175cd3}button.primary{background:#175cd3;color:#fff;border-color:#175cd3}main{max-width:1250px;margin:auto;padding:18px}.case{background:#fff;border:1px solid #d8dee8;border-left:4px solid #f79009;border-radius:10px;padding:14px;margin-bottom:14px}.case.answered{border-left-color:#12b76a}.title{font-weight:750}.grid{display:grid;grid-template-columns:repeat(4,minmax(180px,1fr));gap:9px;margin-top:10px}.card{border:1px solid #d8dee8;border-radius:8px;padding:9px;background:#fbfcfe}.anchor{background:#f0f9ff;border-color:#b9e6fe}.candidate{background:#fffcf5;border-color:#fedf89}.label{font-size:12px;font-weight:750;color:#175cd3;margin-bottom:6px}.candidate .label{color:#b54708}audio{width:100%;height:35px}.choices{display:flex;gap:8px;flex-wrap:wrap;border-top:1px solid #e4e7ec;margin-top:11px;padding-top:11px}.question{font-size:13px;font-weight:750;margin-right:4px}textarea{width:100%;box-sizing:border-box;margin-top:9px;border:1px solid #d8dee8;border-radius:7px;padding:7px;min-height:45px}@media(max-width:850px){.grid{grid-template-columns:repeat(2,minmax(160px,1fr))}}@media(max-width:500px){.grid{grid-template-columns:1fr}}
</style></head><body><header class="top"><h1>双候选音色盲听</h1><div class="sub">20 条相同 no_text case；先听 Source/Reference，再比较匿名 A/B。页面不会解盲。</div><div class="bar"><span id="progress" class="progress"></span><button id="export" class="primary">导出 review JSON</button><button id="clear">清空</button></div></header><main id="cases"></main><script>
const PAGE=__PAGE__;const KEY=`batch44-v1-best3:${PAGE.page_id}`;let state=load();const CHOICES=[["A","A 更像 Reference"],["tie","两者差不多"],["B","B 更像 Reference"],["neither","两组都不像"]];function esc(x){return String(x??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}function load(){try{return JSON.parse(localStorage.getItem(KEY)||"{}")}catch(_){return {}}}function get(id){return state[id]||{judgment:"",note:"",updated_at:""}}function save(){localStorage.setItem(KEY,JSON.stringify(state));progress()}function card(label,src,kind){return `<div class="card ${kind}"><div class="label">${esc(label)}</div><audio controls preload="none" src="${esc(src)}"></audio></div>`}function render(){document.getElementById("cases").innerHTML=PAGE.cases.map((x,i)=>{const a=get(x.case_id);return `<section class="case ${a.judgment?"answered":""}"><div class="title">Case ${String(i+1).padStart(2,"0")} · ${esc(x.case_id)}</div><div class="grid">${card("Source",x.source,"anchor")}${card("Reference",x.reference,"anchor")}${card("候选 A",x.A,"candidate")}${card("候选 B",x.B,"candidate")}</div><div class="choices"><span class="question">哪个更像 Reference？</span>${CHOICES.map(c=>`<button data-id="${esc(x.case_id)}" data-v="${c[0]}" class="${a.judgment===c[0]?"active":""}">${c[1]}</button>`).join("")}</div><textarea data-note="${esc(x.case_id)}" placeholder="可选备注">${esc(a.note)}</textarea></section>`}).join("");document.querySelectorAll("button[data-id]").forEach(b=>b.onclick=()=>{state[b.dataset.id]={...get(b.dataset.id),judgment:b.dataset.v,updated_at:new Date().toISOString()};save();render()});document.querySelectorAll("textarea[data-note]").forEach(t=>t.onchange=()=>{state[t.dataset.note]={...get(t.dataset.note),note:t.value,updated_at:new Date().toISOString()};save()});document.querySelectorAll("audio").forEach(p=>p.onplay=()=>document.querySelectorAll("audio").forEach(q=>{if(q!==p)q.pause()}));progress()}function progress(){const n=PAGE.cases.filter(x=>get(x.case_id).judgment).length;document.getElementById("progress").textContent=`已完成 ${n}/${PAGE.cases.length}`}document.getElementById("export").onclick=()=>{const payload={schema_version:1,page_id:PAGE.page_id,exported_at:new Date().toISOString(),complete:PAGE.cases.every(x=>get(x.case_id).judgment),items:PAGE.cases.map(x=>({case_id:x.case_id,...get(x.case_id)}))};const blob=new Blob([JSON.stringify(payload,null,2)],{type:"application/json"});const u=URL.createObjectURL(blob),a=document.createElement("a");a.href=u;a.download=`${PAGE.page_id}.review.json`;a.click();URL.revokeObjectURL(u)};document.getElementById("clear").onclick=()=>{if(confirm("确认清空？")){state={};localStorage.removeItem(KEY);render()}};render();
</script></body></html>'''


def stage_page(
    *,
    candidate_id: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
    manifest_path: Path,
    seed: int,
    diagnostics_csv: Path,
    run_id: str,
    force: bool,
    candidate_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    case_ids = [row["case_id"] for row in rows]
    positions = ["candidate"] * 10 + ["batch33"] * 10
    random.Random(f"{seed}:{candidate_id}:positions").shuffle(positions)
    mapping: dict[str, dict[str, str]] = {}
    display_cases: list[dict[str, str]] = []
    private_cases: list[dict[str, Any]] = []
    assets = output_dir / "assets"
    for index, (row, role_a) in enumerate(zip(rows, positions), start=1):
        role_b = "batch33" if role_a == "candidate" else "candidate"
        generated = {
            "candidate": row["candidates"][candidate_id]["generated_audio"],
            "batch33": row["batch33"]["generated_audio"],
        }
        prefix = f"case_{index:03d}"
        source = symlink_audio(
            row["source_audio"], assets / f"{prefix}_source{suffix(row['source_audio'])}",
            root=output_dir, force=force,
        )
        reference = symlink_audio(
            row["reference_audio"], assets / f"{prefix}_reference{suffix(row['reference_audio'])}",
            root=output_dir, force=force,
        )
        a_href = symlink_audio(
            generated[role_a], assets / f"{prefix}_A{suffix(generated[role_a])}",
            root=output_dir, force=force,
        )
        b_href = symlink_audio(
            generated[role_b], assets / f"{prefix}_B{suffix(generated[role_b])}",
            root=output_dir, force=force,
        )
        mapping[row["case_id"]] = {"A": role_a, "B": role_b}
        display_cases.append(
            {"case_id": row["case_id"], "source": source, "reference": reference, "A": a_href, "B": b_href}
        )
        private_cases.append(
            {
                "index": index,
                "case_id": row["case_id"],
                "cell": row["cell"],
                "selection_bucket": row["selection_bucket"],
                "mapping": {
                    "A": {
                        "role": role_a,
                        "audio": str(generated[role_a]),
                        "audio_sha256": sha256_file(generated[role_a]),
                    },
                    "B": {
                        "role": role_b,
                        "audio": str(generated[role_b]),
                        "audio_sha256": sha256_file(generated[role_b]),
                    },
                },
                "source": {
                    "audio": str(row["source_audio"]),
                    "audio_sha256": sha256_file(row["source_audio"]),
                },
                "reference": {
                    "audio": str(row["reference_audio"]),
                    "audio_sha256": sha256_file(row["reference_audio"]),
                },
                "candidate_metrics": row["candidates"][candidate_id]["metrics"],
                "batch33_metrics": row["batch33"]["metrics"],
            }
        )
    counts = Counter(role for item in mapping.values() for letter, role in item.items() if letter == "A")
    if counts != {"candidate": 10, "batch33": 10}:
        raise AssertionError(f"unbalanced A positions: {counts}")
    # The page id must not be a deterministic function of candidate identity;
    # otherwise the six registered arm/step candidates can be brute-forced.
    page_id = f"batch44_v1_best3_blind20_{secrets.token_hex(6)}"
    display = {
        "schema_version": 1,
        "page_id": page_id,
        "cases": display_cases,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.html"
    for path in (index_path, manifest_path):
        if path.exists() and not force:
            raise FileExistsError(f"output exists (use --force): {path}")
    encoded = json.dumps(display, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    index_path.write_text(HTML.replace("__PAGE__", encoded), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "data_version": "v1_20260709",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "producer": producer_registration(),
        "page_id": page_id,
        "candidate_id": candidate_id,
        "candidate_run_id": run_id,
        "candidate_diagnostics_csv": str(diagnostics_csv),
        "candidate_diagnostics_sha256": sha256_file(diagnostics_csv),
        "candidate_evidence": dict(candidate_evidence or {}),
        "batch33_run_id": DEFAULT_BATCH33_RUN,
        "mapping_visibility": "private_manifest_only; index.html never fetches this file",
        "response_semantics": {
            "A": "anonymous A is more reference-like",
            "B": "anonymous B is more reference-like",
            "tie": "equally reference-like",
            "neither": "neither resembles reference",
        },
        "position_balance": dict(counts),
        "cases": private_cases,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "candidate_id": candidate_id,
        "page_id": page_id,
        "index": str(index_path),
        "index_sha256": sha256_file(index_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def write_landing(output_root: Path, pages: list[Mapping[str, Any]]) -> None:
    links = "\n".join(
        f'<li><a href="{Path(page["index"]).parent.name}/">Blind20 page {index}</a></li>'
        for index, page in enumerate(pages, start=1)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Batch-44 v1 Best3 blind20</title>"
        "<h1>Batch-44 v1 Best3 vs Batch-33</h1><p>候选身份保持匿名；三页使用互不重叠的 20 条 case。</p><ol>"
        + links
        + "</ol>",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument(
        "--candidate-diagnostics", action="append", default=[], metavar="ID=CSV",
        help="full320 diagnostics CSV for one Best3 candidate; repeat exactly 3 times",
    )
    parser.add_argument(
        "--candidate-run", action="append", default=[], metavar="ID=RUN_ID",
        help="run id inside a candidate diagnostics CSV; repeat exactly 3 times",
    )
    parser.add_argument(
        "--candidate-completion", action="append", default=[], metavar="ID=COMPLETED_JSON",
        help=(
            "registered paired-full320 COMPLETED.json for one Best3 candidate; "
            "defaults to the canonical Batch-44 v1 step path"
        ),
    )
    parser.add_argument("--batch33-diagnostics", type=Path, default=DEFAULT_BATCH33_CSV)
    parser.add_argument("--batch33-run", default=DEFAULT_BATCH33_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--private-root",
        type=Path,
        default=DEFAULT_PRIVATE_ROOT,
        help="non-webroot directory for A/B mappings and BLIND20_READY.json",
    )
    parser.add_argument("--selection-seed", type=int, default=20260744)
    parser.add_argument("--blind-seed", type=int, default=20260745)
    parser.add_argument("--binding-margin", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    selection_path = args.selection.expanduser().resolve()
    selection, selected_ids = load_selection(selection_path)
    diagnostics = dict(parse_binding(raw, option="--candidate-diagnostics") for raw in args.candidate_diagnostics)
    runs = dict(parse_binding(raw, option="--candidate-run") for raw in args.candidate_run)
    completion_overrides = dict(
        parse_binding(raw, option="--candidate-completion")
        for raw in args.candidate_completion
    )
    if set(diagnostics) != set(selected_ids) or set(runs) != set(selected_ids):
        raise ValueError(
            "candidate diagnostics/run bindings must exactly match Best3: "
            f"selected={selected_ids}, diagnostics={sorted(diagnostics)}, runs={sorted(runs)}"
        )
    if set(completion_overrides) - set(selected_ids):
        raise ValueError(
            "candidate completion bindings contain non-Best3 ids: "
            f"{sorted(set(completion_overrides) - set(selected_ids))}"
        )
    candidate_index = {
        str(row.get("candidate_id")): row
        for row in selection.get("candidates", [])
        if isinstance(row, dict)
    }
    candidate_evidence = {}
    for candidate_id in selected_ids:
        candidate = candidate_index.get(candidate_id)
        if not isinstance(candidate, dict):
            raise ValueError(f"Best3 candidate row is missing: {candidate_id}")
        step = int(candidate["step"])
        completion_path = Path(
            completion_overrides.get(
                candidate_id, str(default_completion_path(PROJECT_ROOT, step))
            )
        )
        candidate_evidence[candidate_id] = bind_candidate_full320_evidence(
            candidate=candidate,
            diagnostics_csv=Path(diagnostics[candidate_id]),
            run_id=runs[candidate_id],
            completion_path=completion_path,
            project_root=PROJECT_ROOT,
        )
    candidate_rows = {
        candidate_id: load_diagnostics(
            Path(diagnostics[candidate_id]).expanduser().resolve(),
            run_id=runs[candidate_id],
            margin=args.binding_margin,
            expected_target_root=Path(
                candidate_evidence[candidate_id]["target_audio_root"]
            ),
            expected_n=160,
        )
        for candidate_id in selected_ids
    }
    batch33 = load_diagnostics(
        args.batch33_diagnostics.expanduser().resolve(),
        run_id=args.batch33_run,
        margin=args.binding_margin,
    )
    output_root = args.output_root.expanduser().resolve()
    private_root = args.private_root.expanduser().resolve()
    if private_root == output_root or private_root.is_relative_to(output_root):
        raise ValueError("private-root must not be inside the listening webroot")
    private_root.mkdir(parents=True, exist_ok=True)
    pages = []
    used_case_ids: set[str] = set()
    selection_by_candidate: dict[str, Any] = {}
    skipped_by_candidate: dict[str, Any] = {}
    case_ids_by_candidate: dict[str, list[str]] = {}
    for page_index, candidate_id in enumerate(selected_ids, start=1):
        eligible, skipped = common_eligible(
            {candidate_id: candidate_rows[candidate_id]}, batch33
        )
        eligible = [row for row in eligible if row["case_id"] not in used_case_ids]
        rows, selection_audit = select_cases(
            eligible, seed=args.selection_seed + page_index - 1
        )
        page_case_ids = [row["case_id"] for row in rows]
        if used_case_ids.intersection(page_case_ids):
            raise AssertionError("blind20 pages are not case-disjoint")
        used_case_ids.update(page_case_ids)
        selection_by_candidate[candidate_id] = selection_audit
        skipped_by_candidate[candidate_id] = dict(skipped)
        case_ids_by_candidate[candidate_id] = page_case_ids
        # The URL/directory is deliberately opaque.  Candidate identity exists
        # only in BLIND20_READY.json and the private page manifest.
        page_dir = output_root / f"comparison_{page_index:02d}"
        pages.append(
            stage_page(
                candidate_id=candidate_id,
                rows=rows,
                output_dir=page_dir,
                manifest_path=private_root / f"comparison_{page_index:02d}.manifest.json",
                seed=args.blind_seed,
                diagnostics_csv=Path(diagnostics[candidate_id]).expanduser().resolve(),
                run_id=runs[candidate_id],
                candidate_evidence=candidate_evidence[candidate_id],
                force=args.force,
            )
        )
    write_landing(output_root, pages)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "data_version": "v1_20260709",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "producer": producer_registration(),
        "selection_json": str(selection_path),
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "selected_candidate_ids": selected_ids,
        "case_overlap_policy": "three disjoint 20-case pages",
        "case_ids_by_candidate": case_ids_by_candidate,
        "candidate_evidence_by_candidate": candidate_evidence,
        "batch33_diagnostics_csv": str(args.batch33_diagnostics.expanduser().resolve()),
        "batch33_diagnostics_sha256": sha256_file(
            args.batch33_diagnostics.expanduser().resolve()
        ),
        "batch33_run_id": args.batch33_run,
        "selection_by_candidate": selection_by_candidate,
        "skipped_by_candidate": skipped_by_candidate,
        "pages": pages,
    }
    (private_root / "BLIND20_READY.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

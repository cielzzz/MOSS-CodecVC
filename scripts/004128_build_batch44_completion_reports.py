#!/usr/bin/env python3
"""Build evidence-only Batch-44 closure curves and interim reports.

The original Batch-44 r3/r5 jobs stopped after physical step 10k.  The r3
mainline then continued through a weights-only warm start whose local step
counter restarted at zero::

    effective_step = 10000 + continuation_local_step

This script joins those two quick20 namespaces without pretending that the
optimizer state was resumed.  It may also overlay registered full320 metrics,
but quick20 and full320 remain distinct evidence types in both the TSV/JSON
and the plot.

The report builder is deliberately tolerant of *absent future evidence*:
Best2, FINAL_SELECTION, Batch-42 final and MOS inputs remain explicit
``pending`` entries.  A present but malformed artifact is an error; no metric
is copied from memory or inferred from a neighbouring checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "moss_codecvc.batch44_completion_reports.v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "testset/outputs/batch44_closure_20260713"
DEFAULT_ORIGINAL_QUICK20 = (
    PROJECT_ROOT
    / "testset/outputs/ver23_batch44_quick20_20260713/metrics_all.tsv"
)
DEFAULT_CONTINUATION_QUICK20 = (
    PROJECT_ROOT
    / "testset/outputs/ver23_batch44_r3_warmstart_quick20_20260713/metrics_all.tsv"
)
DEFAULT_BASELINE_TABLE = (
    PROJECT_ROOT
    / "testset/outputs/batch42_baseline_tables_20260711/batch42_baseline_interim.json"
)
DEFAULT_BASELINE_FINAL = (
    PROJECT_ROOT
    / "testset/outputs/batch42_baseline_tables_20260711/batch42_baseline_final.json"
)
DEFAULT_BEST2 = DEFAULT_OUTPUT_DIR / "best2_r3_selection.json"
DEFAULT_FINAL_SELECTION = DEFAULT_OUTPUT_DIR / "FINAL_SELECTION.json"
DEFAULT_MOS_SUMMARY = DEFAULT_OUTPUT_DIR / "batch42_mos_summary.json"

BASE_EFFECTIVE_STEP = 10_000
EXPECTED_STEPS = tuple(range(2_000, 30_001, 2_000))
ORIGINAL_MAX_STEP = 10_000
ORIGINAL_TRAIN_JOBS = {
    "r3": "job-2b91d332-d500-4279-84f9-0a6a81a376aa",
    "r5": "job-b8eb2f1f-a3eb-483b-a289-b4cce281525c",
}
CONTINUATION_TRAIN_JOB = "job-165f3b1d-8c7d-47d8-8882-bdb86ef642ab"
CONTINUATION_CONTRACT_SHA256 = (
    "2d686e5e57b70fcaa3db8c8eb2b306003a38599b2c9ac37023979d80b6d9fc34"
)

CURVE_FIELDS = (
    "step",
    "arm",
    "mode",
    "evidence_type",
    "phase",
    "n",
    "keep",
    "fail",
    "cer",
    "sim_ref",
    "sim_src",
    "margin",
    "ref_bound",
    "ref_content_f1",
    "text_en_src_fail",
    "continuation_local_step",
    "train_job_id",
    "source_path",
    "source_sha256",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_snapshot(path: Path) -> tuple[bytes, dict[str, Any]]:
    path = path.expanduser().resolve()
    data = path.read_bytes()
    if not data:
        raise ValueError(f"empty input artifact: {path}")
    return data, {
        "path": str(path),
        "size": len(data),
        "sha256": sha256_bytes(data),
    }


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def finite(value: Any, *, label: str, optional: bool = False) -> float | None:
    if value in (None, ""):
        if optional:
            return None
        raise ValueError(f"{label} is missing")
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, got bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def integer(value: Any, *, label: str, optional: bool = False) -> int | None:
    number = finite(value, label=label, optional=optional)
    if number is None:
        return None
    result = int(number)
    if not math.isclose(number, result, abs_tol=1e-12):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    return result


def _metric_value(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if row.get(name) not in (None, ""):
            return row[name]
    return None


def _validate_common_metric(row: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    arm = str(row.get("arm") or "")
    mode = str(row.get("mode") or row.get("scope") or "")
    if arm not in {"r3", "r5"}:
        raise ValueError(f"{context}: invalid arm={arm!r}")
    if mode not in {"no_text", "text"}:
        raise ValueError(f"{context}: invalid mode/scope={mode!r}")
    step = integer(_metric_value(row, "effective_step", "step"), label=f"{context}.step")
    n = integer(row.get("n"), label=f"{context}.n")
    keep = integer(row.get("keep"), label=f"{context}.keep")
    assert step is not None and n is not None and keep is not None
    if step not in EXPECTED_STEPS:
        raise ValueError(f"{context}: step={step} is outside 2k..30k/2k")
    if not 0 <= keep <= n:
        raise ValueError(f"{context}: keep={keep}, n={n}")
    fail = finite(_metric_value(row, "fail", "fail_rate"), label=f"{context}.fail")
    cer = finite(row.get("cer"), label=f"{context}.cer")
    sim_ref = finite(
        _metric_value(row, "sim_ref", "wavlm_sim_ref"),
        label=f"{context}.sim_ref",
    )
    sim_src = finite(
        _metric_value(row, "sim_src", "wavlm_sim_src"),
        label=f"{context}.sim_src",
    )
    margin = finite(
        _metric_value(row, "margin", "wavlm_margin"),
        label=f"{context}.margin",
    )
    ref_bound = finite(
        _metric_value(row, "ref_bound", "wavlm_ref_bound"),
        label=f"{context}.ref_bound",
    )
    ref_content = finite(
        _metric_value(row, "ref_content_f1", "ref_content_lcs_f1"),
        label=f"{context}.ref_content_f1",
    )
    assert None not in (fail, cer, sim_ref, sim_src, margin, ref_bound, ref_content)
    if not math.isclose(fail, (n - keep) / n, abs_tol=1e-9):
        raise ValueError(f"{context}: fail/keep mismatch")
    if not math.isclose(margin, sim_ref - sim_src, abs_tol=1e-9):
        raise ValueError(f"{context}: margin != sim_ref-sim_src")
    for label, value in (("fail", fail), ("sim_ref", sim_ref), ("sim_src", sim_src), ("ref_bound", ref_bound)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{context}: {label}={value} is outside [0,1]")
    if cer < 0 or ref_content < 0:
        raise ValueError(f"{context}: negative CER/ref-content")
    return {
        "step": step,
        "arm": arm,
        "mode": mode,
        "n": n,
        "keep": keep,
        "fail": fail,
        "cer": cer,
        "sim_ref": sim_ref,
        "sim_src": sim_src,
        "margin": margin,
        "ref_bound": ref_bound,
        "ref_content_f1": ref_content,
        "text_en_src_fail": finite(
            _metric_value(row, "text_en_src_quick_fail", "text_en_src_fail_rate"),
            label=f"{context}.text_en_src_fail",
            optional=True,
        ),
        "train_job_id": str(row.get("train_job_id") or ""),
    }


def parse_tsv_snapshot(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    data, artifact = read_snapshot(path)
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: TSV is not UTF-8") from exc
    rows = [dict(row) for row in csv.DictReader(io.StringIO(text), delimiter="\t")]
    if not rows:
        raise ValueError(f"{path}: TSV has no data rows")
    return rows, artifact


def load_original_quick20(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows, artifact = parse_tsv_snapshot(path)
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for index, raw in enumerate(raw_rows, start=2):
        context = f"{path}:{index}"
        row = _validate_common_metric(raw, context=context)
        key = (row["arm"], row["step"], row["mode"])
        if key in seen:
            raise ValueError(f"{context}: duplicate metric identity {key}")
        seen.add(key)
        if row["step"] > ORIGINAL_MAX_STEP:
            raise ValueError(f"{context}: original job evidence extends beyond stopped 10k")
        expected_job = ORIGINAL_TRAIN_JOBS[row["arm"]]
        if row["train_job_id"] != expected_job:
            raise ValueError(
                f"{context}: train job={row['train_job_id']!r}, expected {expected_job!r}"
            )
        if row["n"] != 20:
            raise ValueError(f"{context}: quick20 n={row['n']}, expected 20")
        row.update(
            {
                "evidence_type": "quick20",
                "phase": "original_r3_r5_training",
                "continuation_local_step": None,
                "source_path": artifact["path"],
                "source_sha256": artifact["sha256"],
            }
        )
        output.append(row)
    return output, artifact


def load_continuation_quick20(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows, artifact = parse_tsv_snapshot(path)
    output: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for index, raw in enumerate(raw_rows, start=2):
        context = f"{path}:{index}"
        row = _validate_common_metric(raw, context=context)
        if row["arm"] != "r3":
            raise ValueError(f"{context}: continuation contains non-r3 arm")
        if row["step"] <= BASE_EFFECTIVE_STEP:
            raise ValueError(f"{context}: continuation effective step must exceed 10k")
        base = integer(raw.get("base_effective_step"), label=f"{context}.base_effective_step")
        local = integer(
            raw.get("continuation_local_step"),
            label=f"{context}.continuation_local_step",
        )
        assert base is not None and local is not None
        if base != BASE_EFFECTIVE_STEP or row["step"] != base + local:
            raise ValueError(
                f"{context}: invalid effective/local mapping "
                f"effective={row['step']} base={base} local={local}"
            )
        if row["train_job_id"] != CONTINUATION_TRAIN_JOB:
            raise ValueError(f"{context}: continuation training job drift")
        contract_sha = str(raw.get("warm_start_contract_sha256") or "")
        if contract_sha != CONTINUATION_CONTRACT_SHA256:
            raise ValueError(f"{context}: warm-start contract SHA256 drift")
        checkpoint = Path(str(raw.get("checkpoint") or ""))
        if checkpoint.name != f"step-{local}":
            raise ValueError(f"{context}: checkpoint/local-step binding drift")
        if row["n"] != 20:
            raise ValueError(f"{context}: quick20 n={row['n']}, expected 20")
        key = (row["step"], row["mode"])
        if key in seen:
            raise ValueError(f"{context}: duplicate continuation identity {key}")
        seen.add(key)
        row.update(
            {
                "evidence_type": "quick20",
                "phase": "weights_only_warm_start_continuation",
                "continuation_local_step": local,
                "source_path": artifact["path"],
                "source_sha256": artifact["sha256"],
            }
        )
        output.append(row)
    return output, artifact


def load_quick20(
    original_path: Path, continuation_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {}
    if original_path.expanduser().is_file():
        original, inputs["original_quick20"] = load_original_quick20(original_path)
        rows.extend(original)
    else:
        inputs["original_quick20"] = {
            "path": str(original_path.expanduser().resolve()),
            "status": "missing",
        }
    if continuation_path.expanduser().is_file():
        continuation, inputs["continuation_quick20"] = load_continuation_quick20(
            continuation_path
        )
        rows.extend(continuation)
    else:
        inputs["continuation_quick20"] = {
            "path": str(continuation_path.expanduser().resolve()),
            "status": "missing",
        }
    seen: set[tuple[str, int, str]] = set()
    for row in rows:
        key = (row["arm"], row["step"], row["mode"])
        if key in seen:
            raise ValueError(f"quick20 evidence overlaps across phases: {key}")
        seen.add(key)
    return sorted(rows, key=lambda row: (row["step"], row["arm"], row["mode"])), inputs


def _load_json_snapshot(path: Path) -> tuple[Any, dict[str, Any]]:
    data, artifact = read_snapshot(path)
    try:
        return json.loads(data), artifact
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc


def load_full320(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for path in sorted({item.expanduser().resolve() for item in paths}):
        if not path.is_file():
            continue
        payload, artifact = _load_json_snapshot(path)
        inputs.append(artifact)
        raw_rows = payload.get("metrics") if isinstance(payload, dict) else payload
        if not isinstance(raw_rows, list):
            raise ValueError(f"{path}: full320 JSON must contain a metric-row list")
        for index, raw in enumerate(raw_rows):
            if not isinstance(raw, dict):
                raise ValueError(f"{path}: full320 row {index} is not an object")
            scope = str(raw.get("scope") or raw.get("mode") or "")
            if scope == "all":
                continue
            context = f"{path}:row-{index}"
            row = _validate_common_metric(raw, context=context)
            if row["n"] != 160:
                raise ValueError(f"{context}: full320 mode n={row['n']}, expected 160")
            local = integer(
                raw.get("continuation_local_step"),
                label=f"{context}.continuation_local_step",
                optional=True,
            )
            if row["step"] > BASE_EFFECTIVE_STEP and row["arm"] == "r3":
                if local is None or row["step"] != BASE_EFFECTIVE_STEP + local:
                    raise ValueError(
                        f"{context}: post-10k r3 full320 lacks effective/local mapping"
                    )
                phase = "weights_only_warm_start_continuation"
            else:
                phase = "original_r3_r5_training"
            key = (row["arm"], row["step"], row["mode"])
            if key in seen:
                raise ValueError(f"duplicate full320 metric identity: {key}")
            seen.add(key)
            row.update(
                {
                    "evidence_type": "full320",
                    "phase": phase,
                    "continuation_local_step": local,
                    "source_path": artifact["path"],
                    "source_sha256": artifact["sha256"],
                    "eres2net_sim_ref": finite(
                        _metric_value(raw, "eres2net_sim_ref", "eres2net_sim_ref_mean"),
                        label=f"{context}.eres2net_sim_ref",
                        optional=True,
                    ),
                    "speechbrain_sim_ref": finite(
                        _metric_value(raw, "speechbrain_sim_ref", "speechbrain_ecapa_sim_ref"),
                        label=f"{context}.speechbrain_sim_ref",
                        optional=True,
                    ),
                }
            )
            output.append(row)
    return sorted(output, key=lambda row: (row["step"], row["arm"], row["mode"])), inputs


def discover_full320(project_root: Path, output_dir: Path) -> list[Path]:
    patterns = (
        "testset/outputs/ver23_batch44_paired_full320_20260713/step-*/aggregate/paired_metrics.json",
        "testset/outputs/ver23_batch44_r3_warmstart_full320_20260713/**/metrics.json",
    )
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(project_root.glob(pattern))
    if output_dir.is_dir():
        paths.extend(output_dir.glob("full320/**/metrics.json"))
    return paths


def load_optional_json(path: Path, *, label: str) -> tuple[Any | None, dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        return None, {"path": str(path), "status": "missing", "label": label}
    payload, artifact = _load_json_snapshot(path)
    artifact["label"] = label
    return payload, artifact


def fmt(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return "pending"
    return f"{float(value):.{digits}f}"


def fmt_percent(value: Any, digits: int = 2) -> str:
    if value in (None, ""):
        return "pending"
    return f"{100.0 * float(value):.{digits}f}%"


def quick_index(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    return {
        (str(row["arm"]), int(row["step"]), str(row["mode"])): row
        for row in rows
        if row.get("evidence_type") == "quick20"
    }


def full_index(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    return {
        (str(row["arm"]), int(row["step"]), str(row["mode"])): row
        for row in rows
        if row.get("evidence_type") == "full320"
    }


def margin_judgment(value: float | None) -> str:
    if value is None:
        return "pending"
    if value < 0:
        return "fail"
    if value < 0.02:
        return "warn"
    return "pass"


def curve_tsv(rows: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CURVE_FIELDS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in CURVE_FIELDS})
    return buffer.getvalue()


def render_plot(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
    except Exception as exc:  # pragma: no cover - Pillow is part of both project runtimes
        raise RuntimeError("Pillow is required for learning_curves_r3.png") from exc

    # Pillow avoids a known Matplotlib/NumPy ABI split between the reporting
    # and training environments while keeping the plot deterministic.
    width, height = 1680, 1500
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    regular_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    bold_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if regular_path.is_file() and bold_path.is_file():
        font = ImageFont.truetype(str(regular_path), 22)
        small = ImageFont.truetype(str(regular_path), 18)
        tiny = ImageFont.truetype(str(regular_path), 15)
        bold = ImageFont.truetype(str(bold_path), 27)
    else:  # pragma: no cover - standard DejaVu fonts exist on the target host
        font = small = tiny = bold = ImageFont.load_default()

    r3_rows = [row for row in rows if row["arm"] == "r3"]
    quick = [row for row in r3_rows if row["evidence_type"] == "quick20"]
    full = [row for row in r3_rows if row["evidence_type"] == "full320"]
    colors = {"no_text": "#1f77b4", "text": "#d97706"}
    fields = (
        ("cer", "CER (lower is better)"),
        ("sim_ref", "WavLM SIM(ref) (higher is better)"),
        ("margin", "SIM(ref) - SIM(src) (higher is better)"),
    )
    left, right = 155, width - 55
    panel_top = (150, 585, 1020)
    panel_height = 350

    def x_pos(step: int) -> int:
        return int(left + (step / 1000 - 2) / 28 * (right - left))

    def y_bounds(field: str) -> tuple[float, float]:
        values = [float(row[field]) for row in r3_rows if row.get(field) is not None]
        if not values:
            return 0.0, 1.0
        low, high = min(values), max(values)
        span = max(high - low, 0.02)
        padding = span * 0.18
        return max(0.0, low - padding), high + padding

    def y_pos(value: float, low: float, high: float, top: int) -> int:
        return int(top + panel_height - (value - low) / max(high - low, 1e-9) * panel_height)

    def draw_marker(x: int, y: int, *, mode: str, full320: bool) -> None:
        color = colors[mode]
        if full320:
            size = 10
            draw.line((x - size, y - size, x + size, y + size), fill=color, width=5)
            draw.line((x - size, y + size, x + size, y - size), fill=color, width=5)
            draw.ellipse((x - 12, y - 12, x + 12, y + 12), outline="#222222", width=1)
        elif mode == "no_text":
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline="white", width=1)
        else:
            draw.rectangle((x - 6, y - 6, x + 6, y + 6), fill=color, outline="white", width=1)

    draw.text(
        (width // 2, 35),
        "Batch-44 r3 learning curves",
        font=bold,
        fill="#111111",
        anchor="ma",
    )
    draw.text(
        (width // 2, 78),
        "lines/solid markers: fixed quick20 | X markers: formal full320",
        font=small,
        fill="#444444",
        anchor="ma",
    )
    # Legend independent of whether all evidence types already exist.
    legend_y = 115
    for index, (label, mode, is_full) in enumerate(
        (
            ("no_text quick20", "no_text", False),
            ("text quick20", "text", False),
            ("no_text full320", "no_text", True),
            ("text full320", "text", True),
        )
    ):
        x = 345 + index * 275
        draw_marker(x, legend_y, mode=mode, full320=is_full)
        draw.text((x + 18, legend_y), label, font=tiny, fill="#222222", anchor="lm")

    for top, (field, ylabel) in zip(panel_top, fields):
        low, high = y_bounds(field)
        draw.rectangle((left, top, right, top + panel_height), outline="#333333", width=2)
        for tick in range(6):
            ratio = tick / 5
            y = int(top + panel_height - ratio * panel_height)
            value = low + ratio * (high - low)
            draw.line((left, y, right, y), fill="#dddddd", width=1)
            draw.text((left - 12, y), f"{value:.3f}", font=tiny, fill="#444444", anchor="rm")
        draw.text((35, top + panel_height // 2), ylabel, font=small, fill="#222222", anchor="lm")

        for step in EXPECTED_STEPS:
            if step % 4_000 == 0 or step in {2_000, 10_000, 30_000}:
                x = x_pos(step)
                draw.line((x, top + panel_height, x, top + panel_height + 6), fill="#333333", width=1)
                if field == "margin":
                    draw.text((x, top + panel_height + 12), f"{step // 1000}k", font=tiny, fill="#333333", anchor="ma")

        boundary_x = x_pos(BASE_EFFECTIVE_STEP)
        for y in range(top, top + panel_height, 14):
            draw.line((boundary_x, y, boundary_x, min(y + 7, top + panel_height)), fill="#222222", width=2)
        draw.text(
            (boundary_x + 8, top + 8),
            "10k weights-only boundary\noptimizer/scheduler/RNG reset",
            font=tiny,
            fill="#333333",
        )

        for mode in ("no_text", "text"):
            mode_quick = sorted(
                (row for row in quick if row["mode"] == mode),
                key=lambda row: row["step"],
            )
            points = [
                (x_pos(int(row["step"])), y_pos(float(row[field]), low, high, top))
                for row in mode_quick
            ]
            if len(points) > 1:
                draw.line(points, fill=colors[mode], width=3)
            for x, y in points:
                draw_marker(x, y, mode=mode, full320=False)
            for row in sorted(
                (item for item in full if item["mode"] == mode),
                key=lambda item: item["step"],
            ):
                draw_marker(
                    x_pos(int(row["step"])),
                    y_pos(float(row[field]), low, high, top),
                    mode=mode,
                    full320=True,
                )

    draw.text(
        (width // 2, height - 30),
        "Effective training step",
        font=font,
        fill="#222222",
        anchor="ma",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp-{os.getpid()}.png")
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Software", "MOSS-CodecVC")
    metadata.add_text(
        "WarmStartBoundary",
        "effective_step=10000; weights-only; optimizer/scheduler/RNG reset",
    )
    image.save(temporary, format="PNG", pnginfo=metadata, optimize=True)
    os.replace(temporary, output)


def _full_table(rows: Sequence[Mapping[str, Any]], arm: str) -> list[str]:
    selected = [row for row in rows if row["arm"] == arm]
    if not selected:
        return ["- 当前没有该 arm 的正式 full320 证据。"]
    lines = [
        "| Step | Mode | n | Fail | CER | WavLM ref | WavLM src | Margin | SpB ref | ERes2Net ref | en_src fail |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(selected, key=lambda item: (item["step"], item["mode"])):
        lines.append(
            f"| {row['step']} | {row['mode']} | {row['n']} | {fmt_percent(row['fail'])} | "
            f"{fmt(row['cer'])} | {fmt(row['sim_ref'])} | {fmt(row['sim_src'])} | "
            f"{fmt(row['margin'])} | {fmt(row.get('speechbrain_sim_ref'))} | "
            f"{fmt(row.get('eres2net_sim_ref'))} | {fmt_percent(row.get('text_en_src_fail'))} |"
        )
    return lines


def render_paired_metrics_r3_full320(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """Render the standalone r3 full320 evidence requested by Batch-44.

    This file deliberately contains full320 rows only.  It never promotes a
    quick20 result to full-set evidence, and it remains explicitly interim
    until an effective-30k full320 row is present.
    """

    r3_rows = sorted(
        (row for row in rows if row.get("arm") == "r3"),
        key=lambda row: (int(row["step"]), str(row["mode"])),
    )
    observed_steps = sorted({int(row["step"]) for row in r3_rows})
    final_ready = all(
        any(
            int(row["step"]) == 30_000 and row.get("mode") == mode
            for row in r3_rows
        )
        for mode in ("no_text", "text")
    )
    lines = [
        "# Batch-44 r3 paired full320 metrics",
        "",
        f"- Status: **{'complete_at_30k' if final_ready else 'interim'}**.",
        "- Evidence type: strict full320 only (no_text 160 + text 160).",
        "- quick20 rows are intentionally excluded from this file.",
        "- Effective steps with accepted full320 evidence: "
        + (", ".join(str(step) for step in observed_steps) if observed_steps else "none"),
        "",
        *_full_table(r3_rows, "r3"),
        "",
    ]
    return "\n".join(lines)


def _extract_final_metrics(final_payload: Any) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(final_payload, dict):
        return None, "pending"
    candidate = final_payload.get("candidate")
    metrics = final_payload.get("full320_metrics")
    if final_payload.get("status") != "final" or not isinstance(candidate, dict):
        return None, "pending"
    if candidate.get("arm") != "r3" or not isinstance(metrics, dict):
        return None, "incompatible"
    return {"candidate": candidate, "metrics": metrics}, "available"


def _gate_line(label: str, value: Any, predicate: Any, formatter=fmt) -> str:
    if value in (None, ""):
        return f"- {label}: **pending**"
    passed = bool(predicate(float(value)))
    return f"- {label}: **{'Y' if passed else 'N'}**（实际值 {formatter(value)}）"


def render_task1(
    quick_rows: Sequence[Mapping[str, Any]],
    full_rows: Sequence[Mapping[str, Any]],
    *,
    best2: Any,
    final_payload: Any,
    output_dir: Path,
) -> tuple[str, str]:
    final, final_state = _extract_final_metrics(final_payload)
    best2_ready = isinstance(best2, dict) and best2.get("status") in {"selected", "complete", "final"}
    final_metrics = final["metrics"] if final else {}
    no_text = final_metrics.get("no_text") if isinstance(final_metrics, dict) else None
    text = final_metrics.get("text") if isinstance(final_metrics, dict) else None
    no_text = no_text if isinstance(no_text, dict) else {}
    text = text if isinstance(text, dict) else {}
    three_scorers = (
        _metric_value(no_text, "wavlm_sim_ref", "sim_ref"),
        _metric_value(no_text, "eres2net_sim_ref", "eres2net_sim_ref_mean"),
        _metric_value(no_text, "speechbrain_sim_ref", "speechbrain_ecapa_sim_ref"),
    )
    gate_values = (
        no_text.get("cer"),
        three_scorers[0],
        text.get("cer"),
        text.get("text_en_src_fail_rate"),
    )
    report_status = (
        "complete"
        if final_state == "available" and all(value not in (None, "") for value in gate_values + three_scorers)
        else "interim"
    )
    qindex = quick_index(quick_rows)
    lines = [
        "# Batch-44 Task 1 — r3 v1 30k closure report",
        "",
        f"- Report status: **{report_status}**.",
        "- 10k 之后是 weights-only warm-start，不是 optimizer/scheduler/RNG/data-position 的严格 resume。",
        "- quick20 仅用于每 2k 趋势；30k 判据必须来自 final full320。",
        f"- Best2: **{'available' if best2_ready else 'pending'}**.",
        f"- 30k FINAL_SELECTION: **{final_state}**.",
        "",
        "## A. 30k 三判据",
        "",
        _gate_line("no_text CER ≤ 0.08", no_text.get("cer"), lambda value: value <= 0.08),
        _gate_line(
            "no_text WavLM SIM(ref) ≥ 0.45",
            three_scorers[0],
            lambda value: value >= 0.45,
        ),
        _gate_line("text CER ≤ 0.05", text.get("cer"), lambda value: value <= 0.05),
        _gate_line(
            "text en_src fail ≤ 10%",
            text.get("text_en_src_fail_rate"),
            lambda value: value <= 0.10,
            fmt_percent,
        ),
        "",
        "三 scorer（no_text final）：",
        "",
        f"- WavLM-large-SV: {fmt(three_scorers[0])}",
        f"- ERes2Net: {fmt(three_scorers[1])}",
        f"- SpeechBrain ECAPA: {fmt(three_scorers[2])}",
        "",
        "## B. 每 2000 步 quick20 曲线",
        "",
        "| Effective step | no_text SIM(ref) | SIM(src) | margin | margin 判读 | no_text CER | text SIM(ref) | text CER | Evidence |",
        "|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for step in EXPECTED_STEPS:
        no = qindex.get(("r3", step, "no_text"))
        tx = qindex.get(("r3", step, "text"))
        evidence = (
            "original quick20"
            if step <= BASE_EFFECTIVE_STEP and no
            else "warm-start quick20"
            if no
            else "pending"
        )
        lines.append(
            f"| {step} | {fmt(no.get('sim_ref') if no else None)} | "
            f"{fmt(no.get('sim_src') if no else None)} | {fmt(no.get('margin') if no else None)} | "
            f"{margin_judgment(no.get('margin') if no else None)} | "
            f"{fmt(no.get('cer') if no else None)} | {fmt(tx.get('sim_ref') if tx else None)} | "
            f"{fmt(tx.get('cer') if tx else None)} | {evidence} |"
        )
    lines.extend(
        [
            "",
            "红旗规则：margin < 0 为 fail；margin < 0.02 为 warn。图中 10k 竖线表示 weights-only 断点。",
            "",
            "## C. Best2",
            "",
        ]
    )
    if best2_ready:
        selected = best2.get("selected") or best2.get("selected_candidate_ids") or []
        lines.append(f"- 已登记候选：`{selected}`")
    else:
        lines.append("- pending：24k/26k/28k/30k 候选尚未全部完成或 Best2 尚未登记。")
    lines.extend(
        [
            "",
            "## D. 当前正式 full320（不是自动视为 30k final）",
            "",
            *_full_table(full_rows, "r3"),
            "",
            "## E. 论文结论状态",
            "",
            "- final full320 尚未登记时，不对 Path X 30k 成功/失败做预判。",
            "- 主观 blind20 尚未完成时，不宣布 ver2.9.5-final 定稿版本。",
            "",
            "## F. 产物",
            "",
            f"- `{output_dir / 'learning_curves_r3.tsv'}`",
            f"- `{output_dir / 'learning_curves_r3.json'}`",
            f"- `{output_dir / 'learning_curves_r3.png'}`",
            f"- `{output_dir / 'paired_metrics_r3_full320.md'}`",
            f"- `{output_dir / 'batch44_task1_r3_report.md'}`",
            "",
        ]
    )
    return "\n".join(lines), report_status


def render_task2(
    quick_rows: Sequence[Mapping[str, Any]], full_rows: Sequence[Mapping[str, Any]]
) -> tuple[str, str]:
    qindex = quick_index(quick_rows)
    findex = full_index(full_rows)
    lines = [
        "# Batch-44 Task 2 — r5 stopped-arm report",
        "",
        "- Report status: **stopped_at_10k**.",
        "- r5 原训练已终止，最后完整 checkpoint 为 step-10000；10k 后没有训练或评测证据。",
        "- 因此 12k–30k 必须显示 N/A，不能插值、外推或称为 r5 30k。",
        "",
        "## A. quick20 学习曲线",
        "",
        "| Step | no_text CER | no_text SIM(ref) | no_text margin | text CER | text SIM(ref) | Status |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for step in EXPECTED_STEPS:
        no = qindex.get(("r5", step, "no_text"))
        tx = qindex.get(("r5", step, "text"))
        if step > ORIGINAL_MAX_STEP:
            lines.append(f"| {step} | N/A | N/A | N/A | N/A | N/A | arm terminated |")
        elif no and tx:
            lines.append(
                f"| {step} | {fmt(no['cer'])} | {fmt(no['sim_ref'])} | {fmt(no['margin'])} | "
                f"{fmt(tx['cer'])} | {fmt(tx['sim_ref'])} | complete quick20 |"
            )
        else:
            lines.append(f"| {step} | pending | pending | pending | pending | pending | missing evidence |")
    lines.extend(
        [
            "",
            "## B. r5 正式 full320",
            "",
            *_full_table(full_rows, "r5"),
            "",
            "## C. 共同 10k 的 r3 vs r5 对比",
            "",
            "| Mode | Metric | r3 | r5 | r5-r3 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for mode in ("no_text", "text"):
        r3 = findex.get(("r3", 10_000, mode))
        r5 = findex.get(("r5", 10_000, mode))
        for metric, label in (("cer", "CER"), ("sim_ref", "WavLM SIM(ref)"), ("margin", "WavLM margin")):
            left = r3.get(metric) if r3 else None
            right = r5.get(metric) if r5 else None
            delta = None if left is None or right is None else float(right) - float(left)
            lines.append(
                f"| {mode} | {label} | {fmt(left)} | {fmt(right)} | "
                f"{'pending' if delta is None else f'{delta:+.4f}'} |"
            )
    lines.extend(
        [
            "",
            "## D. 不可交付项",
            "",
            "- r5 24k/26k/28k/30k Best2：N/A（arm terminated）。",
            "- r5 30k 三判据：N/A。",
            "- r3-final vs r5 的后续盲听若使用 r5-10k，只能标注为不等训练量历史对照。",
            "- MOS/盲听结果：pending。",
            "",
        ]
    )
    return "\n".join(lines), "stopped_at_10k"


def _baseline_value(row: Mapping[str, Any], name: str) -> Any:
    return row.get(name)


def render_task3(baseline: Any, mos: Any) -> tuple[str, str]:
    if not isinstance(baseline, dict):
        return (
            "# Batch-44 Task 3 — Batch-42 baselines report\n\n"
            "- Report status: **pending**.\n"
            "- Baseline table JSON is unavailable; no remembered numbers were substituted.\n"
            "- test-zh-hard: **N/A** under the current pure-VC protocol.\n"
            "- Ground-truth half-split calibration: pending.\n"
            "- SMOS/CMOS: pending.\n",
            "pending",
        )
    counts = baseline.get("counts") if isinstance(baseline.get("counts"), dict) else {}
    complete = counts.get("complete")
    total = counts.get("systems")
    main_rows = baseline.get("main_table") if isinstance(baseline.get("main_table"), list) else []
    cross_rows = (
        baseline.get("cross_validation_table")
        if isinstance(baseline.get("cross_validation_table"), list)
        else []
    )
    mos_complete = isinstance(mos, dict) and mos.get("status") == "complete"
    final_complete = any(
        isinstance(row, dict)
        and row.get("system_id") == "path_x_final"
        and row.get("status") == "complete"
        for row in main_rows
    )
    report_status = "complete" if final_complete and mos_complete else "interim"
    lines = [
        "# Batch-44 Task 3 — Batch-42 baselines report",
        "",
        f"- Report status: **{report_status}**.",
        f"- Baseline table completion: **{complete}/{total}**.",
        "- test-zh-hard: **N/A**；当前审计输入没有 pure-VC source manifest，不是待补的普通任务。",
        "- 当前 Ground Truth 是 same-file self-eval（SIM 约 1.0）的 scorer calibration；用户要求的前后半段 calibration 仍 pending。",
        f"- SMOS/CMOS 40 cases × 5 raters: **{'complete' if mos_complete else 'pending'}**.",
        "",
        "## A. Paper main table",
        "",
        "| System | Type | EN WavLM SIM | EN WER | ZH WavLM SIM | ZH CER | ZH-hard | Status |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in main_rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"| {row.get('system', row.get('system_id', ''))} | {row.get('type', '')} | "
            f"{fmt(_baseline_value(row, 'en567_wavlm_sim_ref'))} | "
            f"{fmt_percent(_baseline_value(row, 'en567_whisper_wer_fraction'))} | "
            f"{fmt(_baseline_value(row, 'zh1194_wavlm_sim_ref'))} | "
            f"{fmt_percent(_baseline_value(row, 'zh1194_paraformer_cer_fraction'))} | "
            f"N/A | {row.get('status', 'pending')} |"
        )
    lines.extend(
        [
            "",
            "## B. 三 scorer 交叉验证与 spread",
            "",
            "`spread = max(WavLM, ERes2Net, SpB) - min(...)`。spread > 0.15 标记为 scorer disagreement。",
            "",
            "| System | Split | WavLM | ERes2Net | SpB ECAPA | Spread | 判读 |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    spread_flags = 0
    for row in cross_rows:
        if not isinstance(row, dict):
            continue
        values = [
            finite(row.get("wavlm_large_sv_sim_ref"), label="baseline.wavlm", optional=True),
            finite(row.get("eres2net_sim_ref"), label="baseline.eres2net", optional=True),
            finite(row.get("speechbrain_ecapa_sim_ref"), label="baseline.speechbrain", optional=True),
        ]
        spread = max(values) - min(values) if all(value is not None for value in values) else None
        judgment = "pending" if spread is None else "disagreement" if spread > 0.15 else "aligned"
        if judgment == "disagreement":
            spread_flags += 1
        lines.append(
            f"| {row.get('system', row.get('system_id', ''))} | {row.get('split', '')} | "
            f"{fmt(values[0])} | {fmt(values[1])} | {fmt(values[2])} | {fmt(spread)} | {judgment} |"
        )
    lines.extend(
        [
            "",
            f"- 当前 spread > 0.15 的 system/split 单元：**{spread_flags}**。",
            "",
            "## C. Ground Truth calibration",
            "",
            "- same-file self-eval：已完成，只能解释 scorer 自相似度与原始音频 ASR floor。",
            "- 前半段 vs 后半段 speaker calibration：pending；不得用 same-file 1.0 替代。",
            "- target-speaker 朗读 source transcript 的真实平行 ground truth：数据集中不存在。",
            "",
            "## D. 主观 MOS 与相关性",
            "",
        ]
    )
    if mos_complete:
        lines.append(f"- MOS summary: `{mos}`")
    else:
        lines.extend(
            [
                "- 40-case SMOS：pending。",
                "- 40-case CMOS vs ver2.9.5-final：pending。",
                "- 5 raters/case 完整性、95% CI、Pearson correlation：pending。",
                "- 在评分文件完成前，不推断哪个客观 scorer 与主观最相关。",
            ]
        )
    lines.extend(
        [
            "",
            "## E. Final row",
            "",
            f"- ver2.9.5-final (30k): **{'complete' if final_complete else 'pending'}**。",
            "- 现有七行数字保持原表，不用论文/template 数字填空。",
            "",
        ]
    )
    return "\n".join(lines), report_status


def build(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    quick_rows, quick_inputs = load_quick20(
        args.original_quick20, args.continuation_quick20
    )
    full_paths = (
        [path.expanduser().resolve() for path in args.full320_metrics]
        if args.full320_metrics
        else discover_full320(project_root, output_dir)
    )
    full_rows, full_inputs = load_full320(full_paths)
    all_curve_rows = sorted(
        [row for row in quick_rows + full_rows if row["arm"] == "r3"],
        key=lambda row: (row["step"], 0 if row["evidence_type"] == "quick20" else 1, row["mode"]),
    )

    best2, best2_input = load_optional_json(args.best2_selection, label="Best2 selection")
    final_payload, final_input = load_optional_json(
        args.final_selection, label="FINAL_SELECTION"
    )
    baseline_path = args.baseline_table.expanduser().resolve()
    if not args.baseline_table_explicit and DEFAULT_BASELINE_FINAL.is_file():
        baseline_path = DEFAULT_BASELINE_FINAL.resolve()
    baseline, baseline_input = load_optional_json(baseline_path, label="Batch-42 table")
    mos, mos_input = load_optional_json(args.mos_summary, label="MOS summary")

    output_dir.mkdir(parents=True, exist_ok=True)
    curve_tsv_path = output_dir / "learning_curves_r3.tsv"
    curve_json_path = output_dir / "learning_curves_r3.json"
    curve_png_path = output_dir / "learning_curves_r3.png"
    paired_r3_path = output_dir / "paired_metrics_r3_full320.md"
    task1_path = output_dir / "batch44_task1_r3_report.md"
    task2_path = output_dir / "batch44_task2_r5_report.md"
    task3_path = output_dir / "batch44_task3_baselines_report.md"
    manifest_path = output_dir / "closure_manifest.json"

    quick_r3_steps = sorted(
        {row["step"] for row in quick_rows if row["arm"] == "r3" and row["mode"] == "no_text"}
    )
    curve_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "status": "interim" if max(quick_r3_steps, default=0) < 30_000 else "curve_complete",
        "warm_start_boundary": {
            "effective_step": BASE_EFFECTIVE_STEP,
            "semantics": "weights_only_warm_start_not_exact_resume",
            "reset_state": ["optimizer", "scheduler", "rng", "global_step", "data_iterator"],
            "mapping": "effective_step = 10000 + continuation_local_step",
        },
        "expected_steps": list(EXPECTED_STEPS),
        "observed_quick20_steps": quick_r3_steps,
        "missing_quick20_steps": [step for step in EXPECTED_STEPS if step not in quick_r3_steps],
        "inputs": {**quick_inputs, "full320": full_inputs},
        "rows": all_curve_rows,
    }
    atomic_text(curve_tsv_path, curve_tsv(all_curve_rows))
    atomic_json(curve_json_path, curve_payload)
    render_plot(all_curve_rows, curve_png_path)
    atomic_text(paired_r3_path, render_paired_metrics_r3_full320(full_rows) + "\n")

    task1_text, task1_status = render_task1(
        quick_rows,
        full_rows,
        best2=best2,
        final_payload=final_payload,
        output_dir=output_dir,
    )
    task2_text, task2_status = render_task2(quick_rows, full_rows)
    task3_text, task3_status = render_task3(baseline, mos)
    atomic_text(task1_path, task1_text + "\n")
    atomic_text(task2_path, task2_text + "\n")
    atomic_text(task3_path, task3_text + "\n")

    outputs = {
        "learning_curves_tsv": str(curve_tsv_path),
        "learning_curves_json": str(curve_json_path),
        "learning_curves_png": str(curve_png_path),
        "paired_metrics_r3_full320": str(paired_r3_path),
        "task1_report": str(task1_path),
        "task2_report": str(task2_path),
        "task3_report": str(task3_path),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "status": "interim" if "interim" in {task1_status, task3_status} else "complete",
        "reports": {
            "task1": task1_status,
            "task2": task2_status,
            "task3": task3_status,
        },
        "inputs": {
            **quick_inputs,
            "full320": full_inputs,
            "best2": best2_input,
            "final_selection": final_input,
            "baseline_table": baseline_input,
            "mos": mos_input,
        },
        "outputs": {**outputs, "closure_manifest": str(manifest_path)},
    }
    atomic_json(manifest_path, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    tokens = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--original-quick20", type=Path, default=DEFAULT_ORIGINAL_QUICK20)
    parser.add_argument(
        "--continuation-quick20", type=Path, default=DEFAULT_CONTINUATION_QUICK20
    )
    parser.add_argument("--full320-metrics", type=Path, action="append", default=[])
    parser.add_argument("--best2-selection", type=Path, default=DEFAULT_BEST2)
    parser.add_argument("--final-selection", type=Path, default=DEFAULT_FINAL_SELECTION)
    parser.add_argument("--baseline-table", type=Path, default=DEFAULT_BASELINE_TABLE)
    parser.add_argument("--mos-summary", type=Path, default=DEFAULT_MOS_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(tokens)
    args.baseline_table_explicit = "--baseline-table" in tokens
    if "--best2-selection" not in tokens:
        args.best2_selection = args.output_dir / "best2_r3_selection.json"
    if "--mos-summary" not in tokens:
        args.mos_summary = args.output_dir / "batch42_mos_summary.json"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

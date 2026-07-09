#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build SeedTTS-320 per-sample diagnostics.")
    ap.add_argument("--validation-jsonl", required=True)
    ap.add_argument("--sim-cases-csv", required=True)
    ap.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--prefix", default="seedtts320_diagnostics")
    ap.add_argument("--binding-margin", type=float, default=0.05)
    ap.add_argument("--text-margin", type=float, default=0.05)
    ap.add_argument("--text-good-threshold", type=float, default=0.50)
    return ap.parse_args()


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


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value).expanduser()
        return path.name, path
    name, raw_path = value.split("=", 1)
    return name.strip(), Path(raw_path).expanduser()


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "keep"}


def normalize_chars(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def text_units(text: str, lang: str = "") -> list[str]:
    raw = str(text or "").lower()
    if str(lang).lower().startswith("zh"):
        return list(normalize_chars(raw))
    units = re.findall(r"[\w']+", raw, flags=re.UNICODE)
    return units if units else list(normalize_chars(raw))


def edit_distance(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        short, long = a, b
    else:
        short, long = b, a
    prev = list(range(len(short) + 1))
    for i, x in enumerate(long, start=1):
        cur = [i]
        for j, y in enumerate(short, start=1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (0 if x == y else 1)))
        prev = cur
    return prev[-1]


def error_rate(hyp: str, ref: str, lang: str = "") -> float | None:
    ref_units = text_units(ref, lang)
    hyp_units = text_units(hyp, lang)
    if not ref_units:
        return None
    return edit_distance(hyp_units, ref_units) / max(1, len(ref_units))


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        short, long = a, b
    else:
        short, long = b, a
    prev = [0] * (len(short) + 1)
    for item in long:
        cur = [0]
        for idx, other in enumerate(short, start=1):
            cur.append(prev[idx - 1] + 1 if item == other else max(prev[idx], cur[-1]))
        prev = cur
    return prev[-1]


def lcs_f1(a_text: str, b_text: str, lang: str = "") -> float | None:
    a = text_units(a_text, lang)
    b = text_units(b_text, lang)
    if not a or not b:
        return None
    hit = lcs_len(a, b)
    precision = hit / max(1, len(a))
    recall = hit / max(1, len(b))
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def recall(reference_units: list[str], hyp_units: list[str]) -> float | None:
    if not reference_units:
        return None
    return lcs_len(reference_units, hyp_units) / max(1, len(reference_units))


def repeated_ngram_ratio_from_units(units: list[str], max_n: int = 4) -> float:
    if len(units) < 4:
        return 0.0
    best = 0.0
    for n in range(2, max_n + 1):
        grams = [tuple(units[i : i + n]) for i in range(0, len(units) - n + 1)]
        if not grams:
            continue
        counts = Counter(grams)
        repeated = sum(count - 1 for count in counts.values() if count > 1)
        best = max(best, repeated / max(1, len(grams)))
    return float(best)


def split_thirds(units: list[str]) -> list[list[str]]:
    n = len(units)
    a = n // 3
    b = (2 * n) // 3
    return [units[:a], units[a:b], units[b:]]


def classify_binding(sim_ref: float | None, sim_src: float | None, margin: float) -> str:
    if sim_ref is None or sim_src is None:
        return "missing"
    delta = sim_ref - sim_src
    if delta > margin:
        return "ref-bound"
    if delta < -margin:
        return "src-bound"
    return "ambiguous"


def primary_error(row: dict[str, Any]) -> float | None:
    lang = str(row.get("language") or row.get("source_lang") or "").lower()
    cer = finite(row.get("cer_tgt"))
    wer = finite(row.get("wer_tgt"))
    if lang.startswith("zh"):
        return cer
    if lang.startswith("en"):
        return wer
    if cer is not None and wer is not None:
        return min(cer, wer)
    return cer if cer is not None else wer


def find_asr_jsonl(run_dir: Path) -> Path:
    candidates = [p for p in sorted(run_dir.glob("*.asr_eval.jsonl")) if ".shard" not in p.name]
    if not candidates:
        raise FileNotFoundError(f"No merged *.asr_eval.jsonl under {run_dir}")
    return candidates[0]


def read_manifests(run_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("manifest*.jsonl")):
        for row in iter_jsonl(path):
            case_id = str(row.get("case_id") or "")
            if case_id:
                rows[case_id] = row
    return rows


def load_sim_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[(str(row.get("run") or ""), str(row.get("case_id") or ""))] = row
    return rows


def text_confusion(row: dict[str, Any], margin: float, good_threshold: float) -> dict[str, Any]:
    lang = str(row.get("language") or row.get("source_lang") or "")
    hyp = str(row.get("asr_tgt_text") or "")
    given = str(row.get("input_text") or row.get("text") or "")
    source = str(row.get("source_text") or row.get("asr_src_text") or row.get("source_content_text") or "")
    sim_text = lcs_f1(hyp, given, lang)
    sim_source = lcs_f1(hyp, source, lang)
    sane_distinct = normalize_chars(given) != normalize_chars(source)
    label = "garbage"
    if sim_text is not None and sim_source is not None:
        if sim_text >= good_threshold and sim_text >= sim_source + margin:
            label = "follow-text"
        elif sim_source >= good_threshold and sim_source >= sim_text + margin:
            label = "copy-source"
    return {
        "text_sim_to_given": sim_text,
        "text_sim_to_source": sim_source,
        "text_given_source_distinct": sane_distinct,
        "text_confusion": label,
    }


def no_text_position_recall(row: dict[str, Any]) -> dict[str, Any]:
    lang = str(row.get("language") or row.get("source_lang") or "")
    ref_units = text_units(str(row.get("source_text") or row.get("content_ref_text") or ""), lang)
    hyp_units = text_units(str(row.get("asr_tgt_text") or ""), lang)
    thirds = split_thirds(ref_units)
    values = [recall(part, hyp_units) for part in thirds]
    return {
        "pos_recall_head": values[0],
        "pos_recall_middle": values[1],
        "pos_recall_tail": values[2],
    }


def subsequence_cut(hyp_units: list[str], ref_units: list[str]) -> tuple[list[str], float]:
    if not hyp_units or not ref_units:
        return hyp_units, 0.0
    ref_idx = 0
    last_hyp_idx = -1
    for hyp_idx, unit in enumerate(hyp_units):
        if ref_idx < len(ref_units) and unit == ref_units[ref_idx]:
            ref_idx += 1
            last_hyp_idx = hyp_idx
        if ref_idx >= len(ref_units):
            break
    coverage = ref_idx / max(1, len(ref_units))
    if last_hyp_idx < 0:
        return hyp_units, coverage
    return hyp_units[: last_hyp_idx + 1], coverage


def primary_threshold(*, mode: str, lang: str) -> float:
    is_no_text = mode == "no_text"
    if str(lang).lower().startswith("zh"):
        return 0.35 if is_no_text else 0.20
    return 0.30 if is_no_text else 0.25


def tail_ledger(row: dict[str, Any]) -> dict[str, Any]:
    lang = str(row.get("language") or row.get("source_lang") or "")
    mode = str(row.get("mode") or "")
    ref = str(row.get("content_ref_text") or row.get("source_text") or "")
    hyp = str(row.get("asr_tgt_text") or "")
    ref_units = text_units(ref, lang)
    hyp_units = text_units(hyp, lang)
    cropped_units, subseq_cov = subsequence_cut(hyp_units, ref_units)
    duration_ratio = finite(row.get("duration_ratio_tgt_src"))
    prefix_complete = (
        duration_ratio is not None
        and duration_ratio > 1.1
        and subseq_cov >= 0.95
        and len(hyp_units) > max(len(ref_units) + 2, int(len(ref_units) * 1.05))
    )
    cropped_text = " ".join(cropped_units)
    cropped_error = error_rate(cropped_text, ref, lang)
    cropped_repeat = repeated_ngram_ratio_from_units(cropped_units)
    cropped_keep_upper_bound = (
        cropped_error is not None
        and cropped_error <= primary_threshold(mode=mode, lang=lang)
        and cropped_repeat <= 0.30
        and len(normalize_chars(cropped_text)) >= 2
    )
    return {
        "tail_prefix_complete": bool(prefix_complete),
        "tail_subsequence_coverage": subseq_cov,
        "tail_cropped_error": cropped_error,
        "tail_cropped_repeat_score": cropped_repeat,
        "tail_cropped_keep_upper_bound": bool(cropped_keep_upper_bound),
    }


def auto_failure_class(row: dict[str, Any]) -> str:
    if bool_value(row.get("content_keep")):
        return "keep"
    reason = str(row.get("content_filter_reason") or "")
    mode = str(row.get("mode") or "")
    if "lang_mismatch" in reason:
        return "lang-mismatch"
    if mode == "text" and row.get("text_confusion") == "copy-source":
        return "copy-source(text)"
    if "repeat_score" in reason:
        return "repeat-loop"
    if row.get("tail_prefix_complete"):
        return "tail-append"
    if "target_too_short" in reason or "empty_or_too_short_target_asr" in reason:
        return "truncation-silence"
    if mode == "no_text":
        head = finite(row.get("pos_recall_head"))
        mid = finite(row.get("pos_recall_middle"))
        tail = finite(row.get("pos_recall_tail"))
        if mid is not None and head is not None and tail is not None and mid + 0.20 < min(head, tail):
            return "middle-skip"
    if "cer" in reason or "wer" in reason:
        return "other"
    return "other"


def write_scatter(rows: list[dict[str, Any]], output_png: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    runs = sorted({str(row.get("run")) for row in rows})
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * max(1, len(runs)), 4), squeeze=False)
    for ax, run in zip(axes[0], runs):
        subset = [row for row in rows if row.get("run") == run]
        colors = ["#1f77b4" if row.get("mode") == "no_text" else "#d62728" for row in subset]
        xs = [finite(row.get("sim_gen_source")) for row in subset]
        ys = [finite(row.get("sim_gen_ref")) for row in subset]
        points = [(x, y, c) for x, y, c in zip(xs, ys, colors) if x is not None and y is not None]
        if points:
            ax.scatter([p[0] for p in points], [p[1] for p in points], c=[p[2] for p in points], s=14, alpha=0.70)
        ax.axline((0, 0), slope=1, color="#888888", linewidth=1, linestyle="--")
        ax.set_title(run)
        ax.set_xlabel("sim(gen,src)")
        ax.set_ylabel("sim(gen,ref)")
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)
    return True


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = output_dir / args.prefix

    validation = {str(row.get("case_id") or ""): row for row in iter_jsonl(Path(args.validation_jsonl))}
    sim_rows = load_sim_rows(Path(args.sim_cases_csv))
    runs = [parse_run(item) for item in args.run]

    per_sample: list[dict[str, Any]] = []
    sanity = {
        "primary_error_formula": "if language/source_lang startswith zh: CER; elif startswith en: WER; else: min(CER, WER) when both exist, otherwise whichever exists",
        "text_given_source_equal_normalized": 0,
        "text_rows": 0,
        "text_timbre_prompt_missing": {},
        "generation_min_audio_tokens": {},
    }

    for run_name, run_dir in runs:
        manifests = read_manifests(run_dir)
        asr_rows = list(iter_jsonl(find_asr_jsonl(run_dir)))
        min_audio_counter: Counter[str] = Counter()
        text_missing_prompt = 0
        text_prompt_checked = 0
        for asr in asr_rows:
            case_id = str(asr.get("case_id") or asr.get("sample_id") or "")
            manifest = manifests.get(case_id, {})
            val = validation.get(case_id, {})
            sim = sim_rows.get((run_name, case_id), {})
            row: dict[str, Any] = {
                "run": run_name,
                "case_id": case_id,
                "mode": asr.get("mode") or val.get("mode"),
                "cell": asr.get("cell") or val.get("cell"),
                "language": asr.get("language") or asr.get("source_lang") or val.get("source_lang"),
                "source_lang": asr.get("source_lang") or val.get("source_lang"),
                "ref_lang": asr.get("ref_lang") or val.get("ref_lang"),
                "content_keep": bool_value(asr.get("content_keep")),
                "content_filter_reason": asr.get("content_filter_reason"),
                "primary_error": primary_error(asr),
                "cer_tgt": finite(asr.get("cer_tgt")),
                "wer_tgt": finite(asr.get("wer_tgt")),
                "repeat_score": finite(asr.get("repeat_score")),
                "duration_ratio_tgt_src": finite(asr.get("duration_ratio_tgt_src")),
                "asr_tgt_text": asr.get("asr_tgt_text"),
                "content_ref_text": asr.get("content_ref_text") or val.get("content_ref_text"),
                "input_text": asr.get("input_text") or val.get("text"),
                "source_text": asr.get("source_text") or asr.get("asr_src_text") or val.get("source_text"),
                "target_audio": asr.get("target_audio") or manifest.get("output_wav"),
                "source_audio": asr.get("source_audio") or val.get("source_audio"),
                "timbre_ref_audio": asr.get("timbre_ref_audio") or val.get("timbre_ref_audio"),
                "generation_min_audio_tokens": manifest.get("generation_min_audio_tokens"),
                "generation_max_new_tokens": manifest.get("generation_max_new_tokens"),
                "generation_min_new_tokens": manifest.get("generation_min_new_tokens"),
                "generation_structure": manifest.get("generation_structure"),
                "ref_prompt_codec_permutation": manifest.get("ref_prompt_codec_permutation"),
            }
            for key in ("sim_gen_ref", "sim_gen_source", "ecapa_sim_gen_ref", "ecapa_sim_gen_source"):
                row[key] = finite(sim.get(key))
            row["wavlm_delta_ref_minus_src"] = (
                row["sim_gen_ref"] - row["sim_gen_source"]
                if row["sim_gen_ref"] is not None and row["sim_gen_source"] is not None
                else None
            )
            row["ecapa_delta_ref_minus_src"] = (
                row["ecapa_sim_gen_ref"] - row["ecapa_sim_gen_source"]
                if row["ecapa_sim_gen_ref"] is not None and row["ecapa_sim_gen_source"] is not None
                else None
            )
            row["wavlm_binding"] = classify_binding(row["sim_gen_ref"], row["sim_gen_source"], args.binding_margin)
            row["ecapa_binding"] = classify_binding(row["ecapa_sim_gen_ref"], row["ecapa_sim_gen_source"], args.binding_margin)
            if row["mode"] == "text":
                row.update(text_confusion(row, args.text_margin, args.text_good_threshold))
                sanity["text_rows"] += 1
                if not row["text_given_source_distinct"]:
                    sanity["text_given_source_equal_normalized"] += 1
                stats = row.get("ref_prompt_codec_permutation")
                if isinstance(stats, dict) and "prompt_frames" in stats:
                    text_prompt_checked += 1
                    if finite(stats.get("prompt_frames")) is None or float(stats.get("prompt_frames") or 0) <= 0:
                        text_missing_prompt += 1
            if row["mode"] == "no_text":
                row.update(no_text_position_recall(row))
            row.update(tail_ledger(row))
            row["failure_class_auto"] = auto_failure_class(row)
            mat = row.get("generation_min_audio_tokens")
            min_audio_counter[str(mat if mat is not None else "missing")] += 1
            per_sample.append(row)
        sanity["text_timbre_prompt_missing"][run_name] = {
            "checked_rows_with_prompt_stats": text_prompt_checked,
            "missing_or_zero_prompt_frames": text_missing_prompt,
        }
        sanity["generation_min_audio_tokens"][run_name] = dict(min_audio_counter)

    per_sample_jsonl = out_prefix.with_suffix(".per_sample.jsonl")
    with per_sample_jsonl.open("w", encoding="utf-8") as handle:
        for row in per_sample:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    binding: dict[str, Any] = {}
    for encoder, field in (("wavlm", "wavlm_binding"), ("ecapa", "ecapa_binding")):
        enc_payload: dict[str, Any] = {}
        for (run, mode), group in group_by(per_sample, lambda r: (str(r.get("run")), str(r.get("mode")))).items():
            counts = Counter(str(row.get(field) or "missing") for row in group)
            total = sum(counts.values())
            enc_payload[f"{run}:{mode}"] = {
                "n": total,
                "counts": dict(counts),
                "rates": {key: value / total for key, value in counts.items()} if total else {},
                "delta_mean": mean([
                    finite(row.get("wavlm_delta_ref_minus_src" if encoder == "wavlm" else "ecapa_delta_ref_minus_src"))
                    for row in group
                ]),
            }
        binding[encoder] = enc_payload

    text_cross: dict[str, Any] = {}
    for (run, mode), group in group_by([r for r in per_sample if r.get("mode") == "text"], lambda r: (str(r.get("run")), str(r.get("mode")))).items():
        table: Counter[str] = Counter()
        for row in group:
            table[f"{row.get('text_confusion')}|{row.get('wavlm_binding')}"] += 1
        text_cross[f"{run}:{mode}"] = dict(table)

    pos_recall: dict[str, Any] = {}
    for (run, mode), group in group_by([r for r in per_sample if r.get("mode") == "no_text"], lambda r: (str(r.get("run")), str(r.get("mode")))).items():
        pos_recall[f"{run}:{mode}"] = {
            "n": len(group),
            "head": mean([finite(r.get("pos_recall_head")) for r in group]),
            "middle": mean([finite(r.get("pos_recall_middle")) for r in group]),
            "tail": mean([finite(r.get("pos_recall_tail")) for r in group]),
        }

    tail_rows = [r for r in per_sample if r.get("tail_prefix_complete")]
    tail_summary: dict[str, Any] = {}
    for (run, mode), group in group_by(tail_rows, lambda r: (str(r.get("run")), str(r.get("mode")))).items():
        recovered = [r for r in group if bool_value(r.get("tail_cropped_keep_upper_bound"))]
        recovered_from_fail = [
            r for r in recovered
            if not bool_value(r.get("content_keep"))
        ]
        tail_summary[f"{run}:{mode}"] = {
            "n": len(group),
            "original_primary_mean": mean([finite(r.get("primary_error")) for r in group]),
            "tail_cropped_error_mean": mean([finite(r.get("tail_cropped_error")) for r in group]),
            "tail_cropped_keep_upper_bound": len(recovered),
            "tail_cropped_recovered_from_fail": len(recovered_from_fail),
        }

    failure_taxonomy: dict[str, Any] = {}
    fail_rows = [r for r in per_sample if not bool_value(r.get("content_keep"))]
    for (run, mode), group in group_by(fail_rows, lambda r: (str(r.get("run")), str(r.get("mode")))).items():
        failure_taxonomy[f"{run}:{mode}"] = dict(Counter(str(r.get("failure_class_auto")) for r in group))

    scatter_png = out_prefix.with_suffix(".wavlm_scatter.png")
    scatter_written = write_scatter(per_sample, scatter_png)

    payload = {
        "runs": [name for name, _ in runs],
        "per_sample_jsonl": str(per_sample_jsonl),
        "scatter_png": str(scatter_png) if scatter_written else None,
        "binding_margin": args.binding_margin,
        "text_confusion_margin": args.text_margin,
        "text_confusion_good_threshold": args.text_good_threshold,
        "binding": binding,
        "text_confusion_x_wavlm_binding": text_cross,
        "no_text_position_recall": pos_recall,
        "tail_overflow_ledger": tail_summary,
        "sanity": sanity,
        "failure_taxonomy_auto": failure_taxonomy,
    }
    summary_json = out_prefix.with_suffix(".summary.json")
    summary_md = out_prefix.with_suffix(".summary.md")
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(render_md(payload), encoding="utf-8")
    print(f"[seedtts320-diagnostics] wrote {summary_md}")
    return 0


def group_by(rows: list[dict[str, Any]], key_fn):
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    return grouped


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def render_md(payload: dict[str, Any]) -> str:
    lines = ["# SeedTTS-320 Diagnostics", ""]
    lines.append(f"per-sample JSONL: `{payload['per_sample_jsonl']}`")
    if payload.get("scatter_png"):
        lines.append(f"scatter: `{payload['scatter_png']}`")
    lines.extend(["", "Primary error formula: " + payload["sanity"]["primary_error_formula"], ""])
    lines.extend(["## Binding Rate", ""])
    rows = []
    for encoder, groups in payload["binding"].items():
        for group, item in groups.items():
            counts = item["counts"]
            rates = item["rates"]
            rows.append([
                encoder,
                group,
                item["n"],
                f"{counts.get('ref-bound', 0)} ({fmt(rates.get('ref-bound'))})",
                f"{counts.get('src-bound', 0)} ({fmt(rates.get('src-bound'))})",
                f"{counts.get('ambiguous', 0)} ({fmt(rates.get('ambiguous'))})",
                fmt(item.get("delta_mean")),
            ])
    lines.extend(md_table(["encoder", "group", "n", "ref-bound", "src-bound", "ambiguous", "delta mean"], rows))
    lines.extend(["", "## Text Confusion x WavLM Binding", ""])
    rows = [[group, json.dumps(table, ensure_ascii=False)] for group, table in payload["text_confusion_x_wavlm_binding"].items()]
    lines.extend(md_table(["group", "counts"], rows))
    lines.extend(["", "## No-Text Position Recall", ""])
    rows = [[group, item["n"], fmt(item["head"]), fmt(item["middle"]), fmt(item["tail"])] for group, item in payload["no_text_position_recall"].items()]
    lines.extend(md_table(["group", "n", "head", "middle", "tail"], rows))
    lines.extend(["", "## Tail Overflow Ledger", ""])
    rows = [
        [
            group,
            item["n"],
            fmt(item["original_primary_mean"]),
            fmt(item["tail_cropped_error_mean"]),
            item.get("tail_cropped_keep_upper_bound", 0),
            item.get("tail_cropped_recovered_from_fail", 0),
        ]
        for group, item in payload["tail_overflow_ledger"].items()
    ]
    lines.extend(md_table(["group", "n", "orig primary", "tail-cropped error", "crop keep upper", "recover fail"], rows))
    lines.extend(["", "## Assertions / Sanity", ""])
    lines.append(f"- text rows: `{payload['sanity']['text_rows']}`")
    lines.append(f"- normalized given text == source text: `{payload['sanity']['text_given_source_equal_normalized']}`")
    lines.append(f"- text timbre prompt frame checks: `{json.dumps(payload['sanity']['text_timbre_prompt_missing'], ensure_ascii=False)}`")
    lines.append(f"- generation_min_audio_tokens buckets: `{json.dumps(payload['sanity']['generation_min_audio_tokens'], ensure_ascii=False)}`")
    lines.extend(["", "## Failure Taxonomy Auto", ""])
    rows = [[group, json.dumps(table, ensure_ascii=False)] for group, table in payload["failure_taxonomy_auto"].items()]
    lines.extend(md_table(["group", "counts"], rows))
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

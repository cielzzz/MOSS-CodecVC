#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_JSONL = ROOT / "testset/validation/seedtts_vc_ver2_3_validation.jsonl"
DEFAULT_SPEAKER_SIM_ROOT = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/vcdata_construction"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize generated speaker similarity from manifests only.")
    ap.add_argument("--validation-jsonl", default=str(DEFAULT_VALIDATION_JSONL))
    ap.add_argument("--run", action="append", required=True, help="NAME=EVAL_DIR. May be repeated.")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--summary-md", required=True)
    ap.add_argument("--matrix-csv", default="")
    ap.add_argument("--pairwise-csv", default="")
    ap.add_argument("--speaker-device", default="cuda:0")
    ap.add_argument("--speaker-sim-root", default=str(DEFAULT_SPEAKER_SIM_ROOT))
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
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def std(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value).expanduser()
        return path.name, path
    name, raw_path = value.split("=", 1)
    return name.strip(), Path(raw_path).expanduser()


def read_manifests(run_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("manifest*.jsonl")):
        for row in iter_jsonl(path):
            case_id = str(row.get("case_id") or "")
            if case_id and str(row.get("status") or "") in {"ok", "ok_after_rerun", "skipped_exists"}:
                rows[case_id] = row
    return rows


class SpeakerScorer:
    def __init__(self, root: Path, device: str) -> None:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from speaker_similarity import SpeakerSimilarity

        self.backend = SpeakerSimilarity(device=device)
        self.cache: dict[str, Any] = {}
        self.sim_cache: dict[tuple[str, str], float | None] = {}

    def embed(self, path: str | Path):
        key = str(path)
        if key not in self.cache:
            self.cache[key] = self.backend.embed_from_file(key)
        return self.cache[key]

    def similarity(self, a: str | Path, b: str | Path) -> float | None:
        key = (str(a), str(b))
        rev = (key[1], key[0])
        if key in self.sim_cache:
            return self.sim_cache[key]
        if rev in self.sim_cache:
            return self.sim_cache[rev]
        try:
            value = float(self.backend.compute_similarity(self.embed(a), self.embed(b)))
        except Exception as exc:
            print(f"[speaker-sim] failed {a} vs {b}: {type(exc).__name__}: {exc}", file=sys.stderr)
            value = None
        self.sim_cache[key] = value
        return value


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[str(row.get("mode") or "unknown")].append(row)

    def one(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(items),
            "sim_gen_ref_mean": mean([finite(row.get("sim_gen_ref")) for row in items]),
            "sim_gen_ref_std": std([finite(row.get("sim_gen_ref")) for row in items]),
            "sim_gen_source_mean": mean([finite(row.get("sim_gen_source")) for row in items]),
            "sim_gen_source_std": std([finite(row.get("sim_gen_source")) for row in items]),
        }

    out = {"all": one(rows)}
    for mode, items in sorted(by_mode.items()):
        out[mode] = one(items)
    return out


def main() -> int:
    args = parse_args()
    validation_rows = list(iter_jsonl(Path(args.validation_jsonl)))
    validation = {str(row.get("case_id") or ""): row for row in validation_rows}
    scorer = SpeakerScorer(Path(args.speaker_sim_root), args.speaker_device)

    per_case: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    run_summaries: dict[str, Any] = {}

    for run_spec in args.run:
        run_name, run_dir = parse_run(run_spec)
        manifests = read_manifests(run_dir)
        rows: list[dict[str, Any]] = []
        for case_id, manifest in sorted(manifests.items()):
            val = validation.get(case_id, {})
            target_audio = str(manifest.get("output_wav") or val.get("output_wav") or run_dir / f"{case_id}.wav")
            source_audio = str(manifest.get("source_audio") or val.get("source_audio") or "")
            timbre_ref_audio = str(manifest.get("timbre_ref_audio") or val.get("timbre_ref_audio") or "")
            item = {
                "run": run_name,
                "case_id": case_id,
                "mode": manifest.get("mode") or val.get("mode"),
                "cell": manifest.get("cell") or val.get("cell"),
                "source_case_id": val.get("source_case_id") or val.get("counterfactual_group") or case_id,
                "ref_case_id": val.get("ref_case_id"),
                "ref_swap_index": val.get("ref_swap_index"),
                "ref_swap_tag": val.get("ref_swap_tag"),
                "source_audio": source_audio,
                "timbre_ref_audio": timbre_ref_audio,
                "target_audio": target_audio,
                "seed": manifest.get("seed"),
                "status": manifest.get("status"),
            }
            item["sim_gen_ref"] = (
                scorer.similarity(target_audio, timbre_ref_audio)
                if target_audio and timbre_ref_audio and Path(target_audio).exists()
                else None
            )
            item["sim_gen_source"] = (
                scorer.similarity(target_audio, source_audio)
                if target_audio and source_audio and Path(target_audio).exists()
                else None
            )
            rows.append(item)
            per_case.append(item)

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get("source_case_id") or row.get("case_id"))].append(row)
        for group_id, items in sorted(grouped.items()):
            refs = sorted(
                [
                    {
                        "ref_swap_index": item.get("ref_swap_index"),
                        "ref_swap_tag": item.get("ref_swap_tag"),
                        "ref_case_id": item.get("ref_case_id"),
                        "timbre_ref_audio": item.get("timbre_ref_audio"),
                    }
                    for item in items
                    if item.get("timbre_ref_audio")
                ],
                key=lambda row: str(row.get("ref_swap_index")),
            )
            seen_refs: dict[str, dict[str, Any]] = {}
            for ref in refs:
                seen_refs[str(ref["timbre_ref_audio"])] = ref
            refs = list(seen_refs.values())
            if len(refs) >= 2:
                for item in items:
                    for ref in refs:
                        sim = scorer.similarity(str(item["target_audio"]), str(ref["timbre_ref_audio"]))
                        matrix_rows.append(
                            {
                                "run": run_name,
                                "source_case_id": group_id,
                                "generated_case_id": item.get("case_id"),
                                "generated_ref_swap_index": item.get("ref_swap_index"),
                                "generated_ref_swap_tag": item.get("ref_swap_tag"),
                                "comparison_ref_swap_index": ref.get("ref_swap_index"),
                                "comparison_ref_swap_tag": ref.get("ref_swap_tag"),
                                "comparison_ref_case_id": ref.get("ref_case_id"),
                                "is_selected_ref": str(item.get("timbre_ref_audio")) == str(ref.get("timbre_ref_audio")),
                                "sim_gen_comparison_ref": sim,
                            }
                        )
                for left_idx, left in enumerate(items):
                    for right in items[left_idx + 1 :]:
                        sim = scorer.similarity(str(left["target_audio"]), str(right["target_audio"]))
                        pairwise_rows.append(
                            {
                                "run": run_name,
                                "source_case_id": group_id,
                                "left_case_id": left.get("case_id"),
                                "left_ref_swap_index": left.get("ref_swap_index"),
                                "right_case_id": right.get("case_id"),
                                "right_ref_swap_index": right.get("ref_swap_index"),
                                "sim_output_pair": sim,
                            }
                        )

        summary = summarize(rows)
        selected = [finite(row.get("sim_gen_comparison_ref")) for row in matrix_rows if row["run"] == run_name and row["is_selected_ref"]]
        nonselected = [
            finite(row.get("sim_gen_comparison_ref"))
            for row in matrix_rows
            if row["run"] == run_name and not row["is_selected_ref"]
        ]
        summary["counterfactual"] = {
            "selected_ref_mean": mean(selected),
            "selected_ref_std": std(selected),
            "nonselected_ref_mean": mean(nonselected),
            "nonselected_ref_std": std(nonselected),
            "selected_minus_nonselected_mean": (
                mean(selected) - mean(nonselected)
                if mean(selected) is not None and mean(nonselected) is not None
                else None
            ),
            "output_pair_similarity_mean": mean(
                [finite(row.get("sim_output_pair")) for row in pairwise_rows if row["run"] == run_name]
            ),
            "output_pair_similarity_std": std(
                [finite(row.get("sim_output_pair")) for row in pairwise_rows if row["run"] == run_name]
            ),
        }
        summary["run_dir"] = str(run_dir)
        run_summaries[run_name] = summary

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run",
        "case_id",
        "mode",
        "cell",
        "source_case_id",
        "ref_case_id",
        "ref_swap_index",
        "ref_swap_tag",
        "sim_gen_ref",
        "sim_gen_source",
        "seed",
        "status",
        "target_audio",
        "source_audio",
        "timbre_ref_audio",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in per_case:
            writer.writerow(row)

    matrix_csv = Path(args.matrix_csv).expanduser() if args.matrix_csv else None
    if matrix_csv is not None:
        matrix_csv.parent.mkdir(parents=True, exist_ok=True)
        matrix_fields = [
            "run",
            "source_case_id",
            "generated_case_id",
            "generated_ref_swap_index",
            "generated_ref_swap_tag",
            "comparison_ref_swap_index",
            "comparison_ref_swap_tag",
            "comparison_ref_case_id",
            "is_selected_ref",
            "sim_gen_comparison_ref",
        ]
        with matrix_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=matrix_fields, extrasaction="ignore")
            writer.writeheader()
            for row in matrix_rows:
                writer.writerow(row)

    pairwise_csv = Path(args.pairwise_csv).expanduser() if args.pairwise_csv else None
    if pairwise_csv is not None:
        pairwise_csv.parent.mkdir(parents=True, exist_ok=True)
        pairwise_fields = [
            "run",
            "source_case_id",
            "left_case_id",
            "left_ref_swap_index",
            "right_case_id",
            "right_ref_swap_index",
            "sim_output_pair",
        ]
        with pairwise_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=pairwise_fields, extrasaction="ignore")
            writer.writeheader()
            for row in pairwise_rows:
                writer.writerow(row)

    payload = {
        "runs": run_summaries,
        "per_case_csv": str(output_csv),
        "matrix_csv": str(matrix_csv) if matrix_csv else None,
        "pairwise_csv": str(pairwise_csv) if pairwise_csv else None,
    }
    Path(args.summary_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# SeedTTS Speaker Similarity Only",
        "",
        "| run | scope | n | sim gen-ref | sim gen-source | selected ref | non-selected ref | selected-nonselected | output pair sim |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run_name, summary in run_summaries.items():
        for scope in ("all", "no_text", "text"):
            if scope not in summary:
                continue
            cur = summary[scope]
            cf = summary.get("counterfactual", {})
            lines.append(
                "| {run} | {scope} | {n} | {ref}±{ref_s} | {src}±{src_s} | {sel} | {non} | {gap} | {pair} |".format(
                    run=run_name,
                    scope=scope,
                    n=cur["n"],
                    ref=fmt(cur["sim_gen_ref_mean"]),
                    ref_s=fmt(cur["sim_gen_ref_std"]),
                    src=fmt(cur["sim_gen_source_mean"]),
                    src_s=fmt(cur["sim_gen_source_std"]),
                    sel=fmt(cf.get("selected_ref_mean")),
                    non=fmt(cf.get("nonselected_ref_mean")),
                    gap=fmt(cf.get("selected_minus_nonselected_mean")),
                    pair=fmt(cf.get("output_pair_similarity_mean")),
                )
            )
    lines.extend(["", f"per-case CSV: `{output_csv}`"])
    if matrix_csv:
        lines.append(f"matrix CSV: `{matrix_csv}`")
    if pairwise_csv:
        lines.append(f"pairwise CSV: `{pairwise_csv}`")
    lines.append(f"summary JSON: `{args.summary_json}`")
    lines.append("")
    Path(args.summary_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"[speaker-sim-only] wrote {args.summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

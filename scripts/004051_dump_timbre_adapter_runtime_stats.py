#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
PERSISTENT_SCRIPT = ROOT / "scripts/004044_run_seedtts_validation_infer_persistent.py"


def load_persistent_module():
    spec = importlib.util.spec_from_file_location("seedtts_persistent_infer", PERSISTENT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {PERSISTENT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Run a small SeedTTS inference pass while dumping TargetOnlyTimbreAdapter "
            "runtime gate/delta-ratio stats. All remaining args are forwarded to 004044."
        )
    )
    ap.add_argument("--stats-json", required=True)
    ap.add_argument("--stats-md", required=True)
    own_args, remaining = ap.parse_known_args()
    if not remaining:
        ap.error("pass 004044 inference args after --stats-json/--stats-md")
    return own_args, remaining


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def std(values: list[float]) -> float | None:
    if not values:
        return None
    mu = sum(values) / len(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * p))))
    return vals[idx]


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


class RuntimeTimbreStats:
    def __init__(self, layer_adapters: torch.nn.ModuleDict) -> None:
        self.layer_adapters = layer_adapters
        self.current_case_id = ""
        self.case_values: dict[str, dict[str, list[float]]] = {}
        self.handles = []

    def install(self) -> None:
        for name, module in self.layer_adapters.items():
            self.handles.append(module.register_forward_hook(self._make_hook(str(name))))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def start_case(self, case_id: str) -> None:
        self.current_case_id = case_id
        self.case_values = defaultdict(lambda: defaultdict(list))

    def _make_hook(self, layer_name: str):
        def hook(module, inputs, output):
            if not self.current_case_id:
                return
            if len(inputs) < 3:
                return
            hidden = inputs[0]
            target_mask = inputs[2]
            if not torch.is_tensor(hidden) or not torch.is_tensor(output) or not torch.is_tensor(target_mask):
                return
            try:
                mask = target_mask.to(device=hidden.device, dtype=torch.bool)
                if mask.shape != hidden.shape[:2] or not bool(mask.any().item()):
                    return
                delta = (output - hidden).detach().float()
                hidden_f = hidden.detach().float()
                mask_f = mask.unsqueeze(-1).to(dtype=torch.float32)
                delta_norm = float((delta * mask_f).norm().item())
                hidden_norm = float((hidden_f * mask_f).norm().item())
                ratio = delta_norm / max(hidden_norm, 1.0e-8)
                gate = float(torch.sigmoid(module.gate.detach().float()).item())
            except Exception:
                return
            bucket = self.case_values[layer_name]
            bucket["gate"].append(gate)
            bucket["delta_norm"].append(delta_norm)
            bucket["hidden_norm"].append(hidden_norm)
            bucket["delta_ratio"].append(ratio)

        return hook

    def summarize_case(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for layer_name, values in sorted(self.case_values.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
            ratios = [v for v in values.get("delta_ratio", []) if finite(v) is not None]
            gates = [v for v in values.get("gate", []) if finite(v) is not None]
            out[layer_name] = {
                "calls": len(ratios),
                "gate_mean": mean(gates),
                "delta_ratio_mean": mean(ratios),
                "delta_ratio_std": std(ratios),
                "delta_ratio_p50": percentile(ratios, 0.50),
                "delta_ratio_p90": percentile(ratios, 0.90),
                "delta_ratio_max": max(ratios) if ratios else None,
                "delta_norm_mean": mean([v for v in values.get("delta_norm", []) if finite(v) is not None]),
                "hidden_norm_mean": mean([v for v in values.get("hidden_norm", []) if finite(v) is not None]),
            }
        return out


def aggregate_cases(case_stats: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in case_stats:
        for layer, stats in case.get("layers", {}).items():
            by_layer[layer].append(stats)
    out: dict[str, Any] = {}
    for layer, items in sorted(by_layer.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
        out[layer] = {
            "cases": len(items),
            "gate_mean": mean([float(item["gate_mean"]) for item in items if item.get("gate_mean") is not None]),
            "delta_ratio_mean": mean(
                [float(item["delta_ratio_mean"]) for item in items if item.get("delta_ratio_mean") is not None]
            ),
            "delta_ratio_p90_mean": mean(
                [float(item["delta_ratio_p90"]) for item in items if item.get("delta_ratio_p90") is not None]
            ),
            "delta_ratio_max": max(
                [float(item["delta_ratio_max"]) for item in items if item.get("delta_ratio_max") is not None],
                default=None,
            ),
        }
    return out


def main() -> int:
    own_args, forwarded = parse_args()
    pmod = load_persistent_module()
    old_argv = sys.argv
    sys.argv = [str(PERSISTENT_SCRIPT), *forwarded]
    try:
        args = pmod.parse_args()
    finally:
        sys.argv = old_argv

    validation_jsonl = Path(args.validation_jsonl).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    manifest = Path(args.manifest_jsonl).expanduser() if args.manifest_jsonl else output_dir / "manifest.jsonl"
    rows = list(pmod.iter_jsonl(validation_jsonl))
    selected = pmod.select_rows(rows, args)
    if not selected:
        print("[timbre-runtime] no rows selected", file=sys.stderr)
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    engine = pmod.PersistentCodecVCInfer(args)
    layer_adapters = getattr(engine.model, "layer_adapters", None)
    if not isinstance(layer_adapters, torch.nn.ModuleDict) or not layer_adapters:
        raise RuntimeError("model has no layer_adapters ModuleDict to hook")
    collector = RuntimeTimbreStats(layer_adapters)
    collector.install()

    failures = 0
    case_stats: list[dict[str, Any]] = []
    try:
        for row in selected:
            case_id = str(row.get("case_id") or "")
            output_wav = output_dir / f"{pmod.safe_stem(case_id)}.wav"
            content_text, content_text_key = pmod.source_content_text_with_key(row)
            manifest_row = {
                "case_id": case_id,
                "mode": row.get("mode"),
                "cell": row.get("cell"),
                "source_audio": row.get("source_audio"),
                "timbre_ref_audio": row.get("timbre_ref_audio"),
                "text": row.get("text"),
                "content_ref_text": row.get("content_ref_text"),
                "source_content_text": content_text,
                "source_content_text_key": content_text_key,
                "output_wav": str(output_wav),
                "seed": args.seed,
            }
            if output_wav.exists() and not args.overwrite:
                manifest_row.update({"status": "skipped_exists", "elapsed_sec": 0.0})
                pmod.append_jsonl(manifest, manifest_row)
                continue
            start = time.time()
            collector.start_case(case_id)
            try:
                run_stats = engine.run_case(row, output_wav)
                elapsed = round(time.time() - start, 3)
                layers = collector.summarize_case()
                case_stats.append({"case_id": case_id, "layers": layers})
                manifest_row.update(
                    {
                        "status": "ok" if output_wav.exists() else "failed",
                        "returncode": 0 if output_wav.exists() else 1,
                        "elapsed_sec": elapsed,
                        "output_exists": output_wav.exists(),
                        "timbre_runtime_stats": layers,
                    }
                )
                manifest_row.update(run_stats)
                print(f"[timbre-runtime] done {case_id} elapsed={elapsed}s", flush=True)
            except Exception as exc:
                elapsed = round(time.time() - start, 3)
                failures += 1
                manifest_row.update(
                    {
                        "status": "failed",
                        "returncode": 1,
                        "elapsed_sec": elapsed,
                        "output_exists": output_wav.exists(),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"[timbre-runtime] failed {case_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
                if args.fail_fast:
                    pmod.append_jsonl(manifest, manifest_row)
                    break
            pmod.append_jsonl(manifest, manifest_row)
    finally:
        collector.close()

    aggregate = aggregate_cases(case_stats)
    payload = {
        "validation_jsonl": str(validation_jsonl),
        "model_path": str(args.model_path),
        "output_dir": str(output_dir),
        "manifest": str(manifest),
        "cases": case_stats,
        "aggregate_by_layer": aggregate,
        "failures": failures,
    }
    stats_json = Path(own_args.stats_json).expanduser()
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    stats_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Timbre Adapter Runtime Stats",
        "",
        f"model: `{args.model_path}`",
        f"cases: `{len(case_stats)}` failures: `{failures}`",
        "",
        "| layer | cases | sigmoid(gate) | delta_ratio mean | delta_ratio p90 mean | delta_ratio max |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, stats in aggregate.items():
        lines.append(
            "| {layer} | {cases} | {gate} | {ratio} | {p90} | {mx} |".format(
                layer=layer,
                cases=stats.get("cases"),
                gate=fmt(stats.get("gate_mean")),
                ratio=fmt(stats.get("delta_ratio_mean")),
                p90=fmt(stats.get("delta_ratio_p90_mean")),
                mx=fmt(stats.get("delta_ratio_max")),
            )
        )
    lines.extend(["", f"stats JSON: `{stats_json}`", ""])
    stats_md = Path(own_args.stats_md).expanduser()
    stats_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[timbre-runtime] wrote {stats_md}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

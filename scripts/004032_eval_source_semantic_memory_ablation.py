#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


METRIC_PATTERN = re.compile(r"([A-Za-z0-9_]+)=(-?(?:inf|nan|[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:e[-+]?[0-9]+)?)", re.I)


def parse_train_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"train_log_found": False}
    last_metrics: dict[str, float] = {}
    steps = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "step=" not in line:
                continue
            steps += 1
            for key, raw in METRIC_PATTERN.findall(line):
                try:
                    last_metrics[key] = float(raw)
                except ValueError:
                    if raw.lower() == "inf":
                        last_metrics[key] = math.inf
                    elif raw.lower() == "nan":
                        last_metrics[key] = math.nan
    return {"train_log_found": True, "logged_steps": steps, "train_last": last_metrics}


def parse_eval_jsonl(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"eval_found": False}
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return {"eval_found": True, "eval_count": len(rows), "eval_last": rows[-1] if rows else {}}


def summarize_run(run_dir: Path) -> dict[str, Any]:
    summary = {"run_dir": str(run_dir), "name": run_dir.name}
    summary.update(parse_train_log(run_dir / "train.log"))
    summary.update(parse_eval_jsonl(run_dir / "eval_loss.jsonl"))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize Ver2.5 SourceSemanticMemory ablation logs.")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--output-json", default="")
    args = ap.parse_args()

    rows = [summarize_run(Path(item).expanduser()) for item in args.run_dirs]
    output_json = Path(args.output_json).expanduser() if args.output_json else None
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

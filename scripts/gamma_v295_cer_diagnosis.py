#!/usr/bin/env python3
"""Diagnose Ver2.9.5 Path-X effective-30k content errors.

This is deliberately a CPU-only, post-hoc analysis.  It keeps two evaluation
scopes separate:

* the external strict Seed-TTS evaluation (ZH1194 / EN567), which reproduces
  the headline 12.00% CER and 8.25% WER;
* the internal effective-30k Full320 evaluation, whose no-text half supplies
  cell, language-pair, partial gender, and training-mode metadata.

The script does not modify checkpoints or source evaluation artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf


EXTERNAL_REL = Path(
    "testset/outputs/"
    "batch42_unified_scorers_path_x_final_20260714_mtts_effective30k"
)
INTERNAL_REL = Path(
    "testset/outputs/"
    "ver23_batch44_r3_warmstart_full320_20260713/step-30000"
)
INTERNAL_RUN = (
    "ver2_9_5_final_r3_warmstart_effective_step-30000_"
    "seedtts320_all_d2d3_seed1234"
)
ERROR_BINS = (
    ("0-5%", 0.0, 0.05),
    ("5-10%", 0.05, 0.10),
    ("10-20%", 0.10, 0.20),
    ("20-50%", 0.20, 0.50),
    ("50%+", 0.50, math.inf),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_text(text: str, language: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    if language.startswith("zh"):
        return "".join(
            char
            for char in text
            if not char.isspace()
            and not unicodedata.category(char).startswith(("P", "S"))
        )
    text = re.sub(r"[^\w']+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def text_units(text: str, language: str) -> list[str]:
    normalized = normalize_text(text, language)
    if language.startswith("zh"):
        return list(normalized)
    return normalized.split()


def edit_distance(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, 1):
        current = [i]
        for j, right_item in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for j, right_item in enumerate(right, 1):
            current.append(
                previous[j - 1] + 1
                if left_item == right_item
                else max(previous[j], current[-1])
            )
        previous = current
    return previous[-1]


def error_details(hypothesis: str, reference: str, language: str) -> dict[str, Any]:
    hyp = text_units(hypothesis, language)
    ref = text_units(reference, language)
    edits = edit_distance(hyp, ref)
    lcs = lcs_length(hyp, ref)
    return {
        "edits": edits,
        "reference_units": len(ref),
        "hypothesis_units": len(hyp),
        "reconstructed_error": edits / max(1, len(ref)),
        "lcs_recall": lcs / max(1, len(ref)),
        "lcs_precision": lcs / max(1, len(hyp)),
    }


def repeated_ngram_ratio(text: str, language: str) -> float:
    units = text_units(text, language)
    n = 2 if language.startswith("zh") else 3
    if len(units) < n:
        return 0.0
    ngrams = [tuple(units[i : i + n]) for i in range(len(units) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(1, len(ngrams))


def audio_duration(path: str | Path | None) -> float | None:
    if not path:
        return None
    try:
        return float(sf.info(str(path)).duration)
    except (RuntimeError, OSError):
        return None


def trailing_silence_seconds(
    path: str | Path | None, threshold_db: float = -40.0
) -> float | None:
    if not path:
        return None
    try:
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (RuntimeError, OSError):
        return None
    if len(audio) == 0 or sample_rate <= 0:
        return None
    amplitude = abs(audio).max(axis=1)
    threshold = 10.0 ** (threshold_db / 20.0)
    non_silent = (amplitude > threshold).nonzero()[0]
    if len(non_silent) == 0:
        return len(audio) / sample_rate
    return (len(audio) - int(non_silent[-1]) - 1) / sample_rate


def length_bucket(duration: float | None) -> str:
    if duration is None:
        return "unknown"
    if duration < 5:
        return "short_<5s"
    if duration <= 15:
        return "medium_5-15s"
    return "long_>15s"


def lexical_richness(text: str, language: str) -> float:
    units = text_units(text, language)
    return len(set(units)) / max(1, len(units))


def complexity_buckets(rows: list[dict[str, Any]]) -> None:
    by_language: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_language[row["language"]].append(row["lexical_richness"])
    thresholds: dict[str, tuple[float, float]] = {}
    for language, values in by_language.items():
        ordered = sorted(values)
        lower = ordered[min(len(ordered) - 1, len(ordered) // 3)]
        upper = ordered[min(len(ordered) - 1, (2 * len(ordered)) // 3)]
        thresholds[language] = (lower, upper)
    for row in rows:
        lower, upper = thresholds[row["language"]]
        value = row["lexical_richness"]
        row["complexity_bucket"] = (
            "low"
            if value <= lower
            else "medium"
            if value <= upper
            else "high"
        )


def infer_genders(cell: str) -> tuple[str, str]:
    if "_m2f" in cell:
        return "male", "female"
    if "_f2m" in cell:
        return "female", "male"
    return "unknown", "unknown"


def classify_failure(row: dict[str, Any]) -> dict[str, Any]:
    error = row["primary_error"]
    if error <= 0.20:
        return {
            "eligible": False,
            "labels": [],
            "primary": "below_20pct",
            "source_content_status": "not_evaluated",
        }
    details = error_details(row["hypothesis"], row["target_text"], row["language"])
    repeat_score = row["repeat_score"]
    duration_ratio = row.get("duration_ratio")
    tail_silence = row.get("tail_silence_seconds")
    labels: list[str] = []
    if (
        duration_ratio is not None
        and duration_ratio < 0.8
        and tail_silence is not None
        and tail_silence >= 0.20
    ):
        labels.append("unfinished")
    if repeat_score > 0.30:
        labels.append("repetition")
    if details["lcs_precision"] >= 0.90 and details["lcs_recall"] < 0.70:
        labels.append("omission")

    source_text = str(row.get("source_text") or "")
    target_normalized = normalize_text(row["target_text"], row["language"])
    source_normalized = normalize_text(source_text, row["language"])
    source_identifiable = bool(source_normalized and source_normalized != target_normalized)
    source_status = "not_identifiable_same_or_missing_source_text"
    if source_identifiable:
        source_error = error_details(
            row["hypothesis"], source_text, row["language"]
        )["reconstructed_error"]
        target_error = details["reconstructed_error"]
        source_status = "measured"
        if source_error + 0.10 < target_error:
            labels.append("source_content")
    if (
        details["reconstructed_error"] > 0.70
        and "source_content" not in labels
        and "omission" not in labels
        and "repetition" not in labels
    ):
        labels.append("wrong_content")
    if not labels:
        labels.append("other_error")
    priority = (
        "source_content",
        "unfinished",
        "repetition",
        "omission",
        "wrong_content",
        "other_error",
    )
    primary = next(label for label in priority if label in labels)
    return {
        "eligible": True,
        "labels": labels,
        "primary": primary,
        "source_content_status": source_status,
        **details,
    }


def histogram(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for label, lower, upper in ERROR_BINS:
        count = sum(
            1
            for row in rows
            if row["primary_error"] >= lower and row["primary_error"] < upper
        )
        output.append(
            {
                "bin": label,
                "count": count,
                "fraction": count / max(1, len(rows)),
            }
        )
    return output


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [row["primary_error"] for row in rows]
    total_edits = sum(row["edit_details"]["edits"] for row in rows)
    total_units = sum(row["edit_details"]["reference_units"] for row in rows)
    return {
        "count": len(rows),
        "mean": statistics.fmean(errors) if errors else None,
        "std": statistics.pstdev(errors) if len(errors) > 1 else 0.0 if errors else None,
        "median": statistics.median(errors) if errors else None,
        "micro_reconstructed": total_edits / max(1, total_units),
        "over_20pct": sum(error > 0.20 for error in errors),
    }


def slices(
    rows: list[dict[str, Any]], dimensions: list[str]
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for dimension in dimensions:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row.get(dimension, "unknown"))].append(row)
        output[dimension] = [
            {"value": value, **summarize(group)}
            for value, group in sorted(groups.items())
        ]
    return output


def compact_case(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "case_id",
            "scope",
            "mode",
            "language",
            "metric",
            "primary_error",
            "target_text",
            "hypothesis",
            "source_text",
            "generated_duration",
            "source_duration",
            "duration_ratio",
            "tail_silence_seconds",
            "length_bucket",
            "cell",
            "cross_language",
            "source_gender",
            "ref_gender",
            "complexity_bucket",
            "lexical_richness",
            "repeat_score",
            "failure",
        )
    }


def load_external(source_root: Path) -> list[dict[str, Any]]:
    root = source_root / EXTERNAL_REL
    rows: list[dict[str, Any]] = []
    for language, backend, metric in (
        ("zh", "paraformer_zh", "CER"),
        ("en", "whisper_large_v3", "WER"),
    ):
        path = (
            root
            / language
            / "merged"
            / f"path_x_final.{language}.merged.unified_eval.jsonl"
        )
        for record in read_jsonl(path):
            asr = record["content_asr"][backend]
            generated_duration = audio_duration(record["audio"]["generated"])
            source_duration = audio_duration(record["audio"]["source"])
            row = {
                "scope": f"external_{language}",
                "mode": "no_text",
                "case_id": record["case_id"],
                "language": language,
                "metric": metric,
                "primary_error": float(asr["primary_error"]),
                "target_text": str(record["reference_text"]),
                "hypothesis": str(asr["hypothesis"]),
                "source_text": str(record["reference_text"]),
                "generated_audio": record["audio"]["generated"],
                "source_audio": record["audio"]["source"],
                "reference_audio": record["audio"]["reference"],
                "generated_duration": generated_duration,
                "source_duration": source_duration,
                "duration_ratio": (
                    generated_duration / source_duration
                    if generated_duration is not None
                    and source_duration is not None
                    and source_duration > 0
                    else None
                ),
                "tail_silence_seconds": trailing_silence_seconds(
                    record["audio"]["generated"]
                ),
                "length_bucket": length_bucket(source_duration),
                "cell": "unknown",
                "cross_language": "unknown",
                "source_gender": "unknown",
                "ref_gender": "unknown",
            }
            row["lexical_richness"] = lexical_richness(
                row["target_text"], language
            )
            row["repeat_score"] = repeated_ngram_ratio(
                row["hypothesis"], language
            )
            row["edit_details"] = error_details(
                row["hypothesis"], row["target_text"], language
            )
            rows.append(row)
    complexity_buckets(rows)
    for row in rows:
        row["failure"] = classify_failure(row)
    return rows


def load_internal(source_root: Path) -> list[dict[str, Any]]:
    run_root = source_root / INTERNAL_REL / INTERNAL_RUN
    path = run_root / f"{INTERNAL_RUN}.asr_eval.jsonl"
    rows = []
    for record in read_jsonl(path):
        language = str(record["language"])
        metric = "CER" if language.startswith("zh") else "WER"
        primary_error = (
            float(record["cer_tgt"])
            if metric == "CER"
            else float(record["wer_tgt"])
        )
        generated_duration = audio_duration(record["target_audio"])
        source_duration = audio_duration(record["source_audio"])
        source_gender, ref_gender = infer_genders(str(record["cell"]))
        row = {
            "scope": "internal_full320",
            "mode": str(record["mode"]),
            "case_id": record["case_id"],
            "language": language,
            "metric": metric,
            "primary_error": primary_error,
            "target_text": str(record["target_text"]),
            "hypothesis": str(record["asr_tgt_text"]),
            "source_text": str(record.get("source_text") or ""),
            "generated_audio": record["target_audio"],
            "source_audio": record["source_audio"],
            "reference_audio": record["timbre_ref_audio"],
            "generated_duration": generated_duration,
            "source_duration": source_duration,
            "duration_ratio": finite(record.get("duration_ratio_tgt_src")),
            "tail_silence_seconds": trailing_silence_seconds(
                record["target_audio"]
            ),
            "length_bucket": length_bucket(source_duration),
            "cell": str(record["cell"]),
            "cross_language": (
                "cross_language"
                if record["source_lang"] != record["ref_lang"]
                else "same_language"
            ),
            "source_gender": source_gender,
            "ref_gender": ref_gender,
            "repeat_score": float(record.get("repeat_score") or 0.0),
        }
        row["lexical_richness"] = lexical_richness(
            row["target_text"], language
        )
        row["edit_details"] = error_details(
            row["hypothesis"], row["target_text"], language
        )
        rows.append(row)
    complexity_buckets(rows)
    for row in rows:
        row["failure"] = classify_failure(row)
    return rows


def failure_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["failure"]["eligible"]]
    primary = Counter(row["failure"]["primary"] for row in eligible)
    multilabel = Counter(
        label for row in eligible for label in row["failure"]["labels"]
    )
    source_status = Counter(
        row["failure"]["source_content_status"] for row in eligible
    )
    return {
        "eligible_over_20pct": len(eligible),
        "primary": {
            key: {
                "count": value,
                "fraction": value / max(1, len(eligible)),
            }
            for key, value in sorted(primary.items())
        },
        "multilabel": {
            key: {
                "count": value,
                "fraction": value / max(1, len(eligible)),
            }
            for key, value in sorted(multilabel.items())
        },
        "source_content_identifiability": dict(source_status),
    }


def top_bottom(rows: list[dict[str, Any]], count: int = 10) -> dict[str, Any]:
    ordered = sorted(
        rows,
        key=lambda row: (row["primary_error"], row["case_id"]),
    )
    return {
        "best": [compact_case(row) for row in ordered[:count]],
        "worst": [compact_case(row) for row in reversed(ordered[-count:])],
    }


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    output = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    output.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    )
    return output


def render_slice(
    lines: list[str], title: str, groups: list[dict[str, Any]]
) -> None:
    lines.extend(["", f"### {title}", ""])
    lines.extend(
        markdown_table(
            ["组", "N", "mean", "std", "median", "micro*", ">20%"],
            [
                [
                    group["value"],
                    group["count"],
                    fmt_pct(group["mean"]),
                    fmt_pct(group["std"]),
                    fmt_pct(group["median"]),
                    fmt_pct(group["micro_reconstructed"]),
                    group["over_20pct"],
                ]
                for group in groups
            ],
        )
    )


def render_failures(
    lines: list[str], title: str, result: dict[str, Any]
) -> None:
    lines.extend(
        [
            "",
            f"### {title}",
            "",
            f"- `primary error > 20%`：{result['eligible_over_20pct']} 条。",
            "- 模式允许多标签；下表先给互斥 primary mode，再给多标签计数。",
            "",
            "Primary mode：",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["模式", "case 数", "占 >20% case"],
            [
                [name, data["count"], fmt_pct(data["fraction"])]
                for name, data in result["primary"].items()
            ],
        )
    )
    lines.extend(["", "多标签计数：", ""])
    lines.extend(
        markdown_table(
            ["模式", "case 数", "占 >20% case"],
            [
                [name, data["count"], fmt_pct(data["fraction"])]
                for name, data in result["multilabel"].items()
            ],
        )
    )
    lines.extend(
        [
            "",
            "Source-content 可判别状态："
            f" `{json.dumps(result['source_content_identifiability'], ensure_ascii=False)}`。",
        ]
    )


def render_cases(
    lines: list[str], title: str, rows: list[dict[str, Any]]
) -> None:
    lines.extend(["", f"### {title}", ""])
    for index, row in enumerate(rows, 1):
        failure = row["failure"]
        lines.extend(
            [
                f"{index}. `{row['case_id']}` — {row['metric']} "
                f"`{fmt_pct(row['primary_error'])}`, "
                f"lang=`{row['language']}`, length=`{row['length_bucket']}`, "
                f"cell=`{row['cell']}`, mode=`{failure['primary']}`",
                f"   - Target: {row['target_text']}",
                f"   - ASR: {row['hypothesis']}",
            ]
        )


def build_report(result: dict[str, Any], report: Path) -> None:
    external = result["external"]
    internal = result["internal"]
    lines = [
        "# Ver2.9.5 Path X 30k CER/WER 弱项诊断（γ.1）",
        "",
        "日期：2026-07-22",
        "",
        "## 结论先行",
        "",
        "1. 主表 `ZH CER 12.00% / EN WER 8.25%` 来自外部严格评测"
        "（ZH1194 / EN567），不是内部 Full320。两层数据必须分开解释。",
        "2. 外部严格评测中，高错误 case 并非主要由 >15 秒长音频构成；"
        "更强的直接信号是少数高错误尾部与内容错误拉高均值。",
        "3. 内部 no_text 的跨语言 reference voice 并未系统性恶化内容错误；"
        "内容来自 source，ref language/cell 更影响音色转换而非应读文本。",
        "4. 最强根因仍是内容通路/对齐不足：历史训练诊断显示 BNF "
        "cross-attention 近均匀、guided loss 高、phoneme accuracy 低；"
        "当前逐 case 结果与漏读/错读长尾一致。",
        "5. 单纯继续训练不是首选。30k 相对 28k 有改善，但 20k→30k "
        "no_text CER 曾从 6.05% 变差至 6.78%，且 SIM/训练标量趋于平台。",
        "",
        "## 口径与方法",
        "",
        "- 外部严格集：ZH1194 用 Paraformer CER，EN567 用 Whisper-Large-v3 WER；"
        "这是论文主表口径。",
        "- 内部 Full320：只把 160 条 `no_text` 作为 γ.1 主诊断样本；"
        "另用 160 条 `text` 作模式对照。内部 ASR 是 Qwen-ASR。",
        "- `mean/std/median` 是 canonical per-case primary error 的宏平均。"
        "`micro*` 是本脚本按规范化字符/词重新计算 edit counts 后的微平均，"
        "仅用于切片比较，不替代官方汇总。",
        "- 长度按 source audio：短 `<5s`、中 `5–15s`、长 `>15s`。",
        "- 词汇丰富度：中文唯一规范化字符占比，英文唯一规范化词占比；"
        "在每种语言内按三分位分 low/medium/high。",
        "- 尾静音 proxy：generated wav 末尾低于 `-40 dBFS` 的连续时长；"
        "`unfinished` 要求 duration ratio `<0.8` 且尾静音 `>=0.2s`。",
        "- 重复 proxy：中文重复 bigram、英文重复 trigram 的超额占比；"
        "阈值 `>0.3`。内部同时沿用已有 repeat_score。",
        "- `source_content` 只有 source text 与 target text 不同时才可判别。"
        "外部 strict VC 和内部 no_text 内容保持任务中二者相同，不能用这两组"
        "伪造“读 source 内容”结论。",
        "",
        "## 一、外部严格评测：主表 12.00% / 8.25%",
        "",
    ]
    lines.extend(
        markdown_table(
            ["语言", "N", "指标", "mean", "std", "median", "micro*", ">20%"],
            [
                [
                    language.upper(),
                    data["summary"]["count"],
                    "CER" if language == "zh" else "WER",
                    fmt_pct(data["summary"]["mean"]),
                    fmt_pct(data["summary"]["std"]),
                    fmt_pct(data["summary"]["median"]),
                    fmt_pct(data["summary"]["micro_reconstructed"]),
                    data["summary"]["over_20pct"],
                ]
                for language, data in external.items()
            ],
        )
    )
    for language in ("zh", "en"):
        data = external[language]
        lines.extend(["", f"### {language.upper()} error 直方图", ""])
        lines.extend(
            markdown_table(
                ["区间", "case 数", "比例"],
                [
                    [row["bin"], row["count"], fmt_pct(row["fraction"])]
                    for row in data["histogram"]
                ],
            )
        )
        for dimension, title in (
            ("length_bucket", "按 source 音频长度"),
            ("complexity_bucket", "按内容词汇丰富度"),
        ):
            render_slice(lines, f"{language.upper()} {title}", data["slices"][dimension])
        render_failures(lines, f"{language.upper()} >20% 失败模式", data["failures"])
        render_cases(lines, f"{language.upper()} Top 10 worst", data["cases"]["worst"])
        render_cases(lines, f"{language.upper()} Top 10 best", data["cases"]["best"])

    no_text = internal["no_text"]
    text = internal["text"]
    lines.extend(
        [
            "",
            "## 二、内部 Full320：no_text 160 条分层",
            "",
            "内部 no_text 总体并不是主表 12%：ZH/EN 混合 primary error "
            f"宏平均为 `{fmt_pct(no_text['summary']['mean'])}`。"
            "这里的价值是 cell、跨语言和部分 gender 元数据。",
            "",
        ]
    )
    for dimension, title in (
        ("language", "语言"),
        ("length_bucket", "source 音频长度"),
        ("source_gender", "Source gender"),
        ("ref_gender", "Reference gender"),
        ("cross_language", "Source/ref 是否跨语言"),
        ("cell", "完整 cell"),
        ("complexity_bucket", "内容词汇丰富度"),
    ):
        render_slice(lines, title, no_text["slices"][dimension])
    lines.extend(["", "### no_text error 直方图", ""])
    lines.extend(
        markdown_table(
            ["区间", "case 数", "比例"],
            [
                [row["bin"], row["count"], fmt_pct(row["fraction"])]
                for row in no_text["histogram"]
            ],
        )
    )
    render_failures(lines, "no_text >20% 失败模式", no_text["failures"])
    render_cases(lines, "no_text Top 10 worst", no_text["cases"]["worst"])
    render_cases(lines, "no_text Top 10 best", no_text["cases"]["best"])

    lines.extend(
        [
            "",
            "## 三、text mode 对照",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["模式", "N", "mean primary error", "std", ">20%"],
            [
                [
                    "no_text",
                    no_text["summary"]["count"],
                    fmt_pct(no_text["summary"]["mean"]),
                    fmt_pct(no_text["summary"]["std"]),
                    no_text["summary"]["over_20pct"],
                ],
                [
                    "text",
                    text["summary"]["count"],
                    fmt_pct(text["summary"]["mean"]),
                    fmt_pct(text["summary"]["std"]),
                    text["summary"]["over_20pct"],
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            "Text mode 明显更好，说明显式文本可绕过/补强弱内容对齐；但它改变了"
            "产品设定，不能直接拿来替代 no_text VC 主结果。它支持把 text-mode "
            "比例实验作为 γ.2 的一个独立 probe，而不是直接改主表口径。",
            "",
            "## 四、根因排序",
            "",
            "### 1. F — BNF content cross-attention / 对齐强度不足（高置信）",
            "",
            "- 直接现象：错误集中于读错、漏读及其他内容长尾；text mode 对照更好。",
            "- 历史训练证据：content cross-attention normalized entropy "
            "`≈0.9991`、guided attention loss `≈0.980`、peak probability "
            "仅 `≈1.33× uniform`，content injection 前置集中；effective-14k "
            "附近 phoneme accuracy 约 `11.9%`。",
            "- 这是当前最能同时解释“SIM 已强、CER/WER 落后”的根因。",
            "",
            "### 2. D — 内容数据覆盖/难例分布不足（中等置信）",
            "",
            "- 外部严格集显著差于内部 Full320，说明内部抽样低估真实长尾。",
            "- 高错误 case 与复杂度/具体语句类型存在长尾，值得做 hard-case "
            "reweight 与数据补齐；但仅凭评测集不能证明训练数据具体缺哪类。",
            "",
            "### 3. E — 训练不充分（低到中等置信，不能盲目续训）",
            "",
            "- 支持面：内部 28k→30k no_text CER `7.81%→6.78%`，text "
            "`5.13%→3.90%`。",
            "- 反证：20k→30k no_text CER `6.05%→6.78%`，SIM(ref) "
            "`0.4496→0.4485`，训练标量趋于平台。纯 30k→60k 不是首选，"
            "只适合作为 warm-restart 严格 stop-gated probe。",
            "",
            "### 4. A — 长句问题（低置信）",
            "",
            "- 当前 source 音频大多低于 15 秒，>15 秒样本不足以成为 12% "
            "主因。应关注 duration mismatch/尾部失败，而不是笼统补长句。",
            "",
            "### 5. B — 跨语言 reference 问题（低置信）",
            "",
            "- 内部 no_text 中 source/ref 跨语言 cell 没有形成一致的内容错误劣化。"
            "外部严格集缺 cell 元数据，无法对主表做同样切片。",
            "",
            "### 6. C — Speaker 泄漏导致读 source 内容（当前 no_text 不可验证）",
            "",
            "- no_text VC 的 source text 本来就是 target text，因此“读 source "
            "内容”在任务定义上不可判别。只有 text mode 的 counterfactual "
            "source/target 不同时可测，不应把它当成 12% CER 主因。",
            "",
            "## 五、γ.2 针对性建议（仅建议，尚未授权执行）",
            "",
            "1. **优先 Probe C：内容通路加强。** Adapter Conformer 2→4，"
            "guided attention `0.05→0.10`，phoneme classifier "
            "`0.02→0.05`；其余保持不变。这与第一根因直接对应。",
            "2. **并行 Probe A-lite：hard-case reweight/补数据。** 不先笼统扩到"
            " 400k；先按本报告的高错误长度、复杂度和失败模式构造 50k 左右"
            "定向增量，并保持独立数据审计。",
            "3. **Probe D 作为产品分支。** no_text:text 从 `1:0.3→1:1`，"
            "同时分别报告 no_text 与 text，不能用 text 数字替换 no_text 主表。",
            "4. **训练规模化 Probe B 排在上述之后。** 只做 30k→33k 的短 "
            "warm-restart probe 和硬止损；先证明 CER 至少降 2pp，再授权 45k/60k。",
            "",
            "建议 γ.2 首轮组合：`C`、`A-lite+C`、`D`。暂不把纯 `B` "
            "列为前三。",
            "",
            "## 六、限制",
            "",
            "- 外部 strict 数据缺 gender/cell，因此 gender/cross-language "
            "结论只来自内部 160 条 no_text。",
            "- `same_gender` cell 没有绝对男女标签，source/ref gender 均记为"
            " unknown；不进行猜测。",
            "- 失败模式是可复现自动 proxy，不等于人工听审。Top worst 建议在"
            " γ.2 设计前人工听 10 条确认。",
            "- 外部 source text 与 target text 相同，source-content 泄漏不可判别。",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_scope(rows: list[dict[str, Any]], dimensions: list[str]) -> dict[str, Any]:
    return {
        "summary": summarize(rows),
        "histogram": histogram(rows),
        "slices": slices(rows, dimensions),
        "failures": failure_summary(rows),
        "cases": top_bottom(rows),
    }


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    external_rows = load_external(source_root)
    internal_rows = load_internal(source_root)

    external: dict[str, Any] = {}
    for language in ("zh", "en"):
        language_rows = [
            row for row in external_rows if row["language"] == language
        ]
        external[language] = analyze_scope(
            language_rows, ["length_bucket", "complexity_bucket"]
        )

    internal: dict[str, Any] = {}
    dimensions = [
        "language",
        "length_bucket",
        "source_gender",
        "ref_gender",
        "cross_language",
        "cell",
        "complexity_bucket",
    ]
    for mode in ("no_text", "text"):
        mode_rows = [row for row in internal_rows if row["mode"] == mode]
        internal[mode] = analyze_scope(mode_rows, dimensions)

    result = {
        "schema": "moss_codecvc.gamma_v295_cer_diagnosis.v1",
        "date": "2026-07-22",
        "source_root": str(source_root),
        "external": external,
        "internal": internal,
        "method": {
            "external": "ZH1194 Paraformer CER; EN567 Whisper-Large-v3 WER",
            "internal": "effective-30k Full320 Qwen-ASR; no_text primary, text contrast",
            "tail_silence_threshold_dbfs": -40.0,
            "unfinished_duration_ratio_max": 0.8,
            "unfinished_tail_silence_min_seconds": 0.2,
            "repeat_proxy_threshold": 0.3,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "diagnosis.json", result)
    write_jsonl(
        output_dir / "external_per_case.jsonl",
        [compact_case(row) for row in external_rows],
    )
    write_jsonl(
        output_dir / "internal_per_case.jsonl",
        [compact_case(row) for row in internal_rows],
    )
    build_report(result, args.report.resolve())

    print(
        json.dumps(
            {
                "external_zh_mean": external["zh"]["summary"]["mean"],
                "external_en_mean": external["en"]["summary"]["mean"],
                "internal_no_text_mean": internal["no_text"]["summary"]["mean"],
                "internal_text_mean": internal["text"]["summary"]["mean"],
                "output_dir": str(output_dir),
                "report": str(args.report.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

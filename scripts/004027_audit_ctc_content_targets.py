#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moss_codecvc.io_utils import load_torch_file


DEFAULT_MANIFESTS = [
    ROOT / "testset/validation/ver2_3_debug/moss_codecvc_ver2_3_loss_valid_160.no_text.jsonl",
    ROOT
    / "trainset/zh45w_en22w_no_text/sft/"
    "moss_codecvc_sft.zh45w_en22w_no_text.with_light_ecapa_spk.with_prosody."
    "with_asr_filter.with_hubert.with_spm_content_tokens.ctc_clean.jsonl",
    ROOT
    / "trainset/zh3w_en3w_text_prosody_independent_timbre/sft/"
    "moss_codecvc_sft.zh3w_en3w_text_prosody_independent_timbre.with_light_ecapa_spk."
    "with_prosody.with_target_asr.with_content_tokens.with_target_hubert."
    "with_spm_content_tokens.ctc_clean.jsonl",
]
DEFAULT_TEXT_KEYS = (
    "content_ref_text",
    "asr_src_text",
    "source_asr_text",
    "source_text",
    "asr_tgt_text",
    "target_text",
    "text",
)
TOKEN_PATH_KEYS = (
    "content_token_ids_path",
    "content_tokens_path",
    "content_ref_token_ids_path",
)
TOKEN_VALUE_KEYS = ("content_token_ids", "content_ref_token_ids")
TOKEN_PAYLOAD_KEYS = (
    "content_token_ids",
    "content_ref_token_ids",
    "content_ids",
    "semantic_tokens",
    "semantic_ids",
    "unit_ids",
    "units",
)
NO_TEXT_VALUES = {"", "<NO_TEXT>", "None", "none", "null"}
PUNCT_OR_SYMBOL_RE = re.compile(r"^[\W_]+$", re.UNICODE)
CONTENT_CHAR_RE = re.compile(r"[\w\u4e00-\u9fff]", re.UNICODE)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit Ver2.3 TextCTC content_token_ids targets before CTC training.")
    ap.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="JSONL manifest to audit. Can be repeated. Defaults to the current Ver2.3 valid/no-text/text CTC manifests.",
    )
    ap.add_argument("--sample-size", type=int, default=1000, help="Reservoir sample size per manifest; <=0 keeps all scanned rows.")
    ap.add_argument("--max-rows", type=int, default=5000, help="Max rows to scan per manifest; <=0 scans the full file.")
    ap.add_argument("--seed", type=int, default=20260630)
    ap.add_argument("--output-report", default=str(ROOT / "outputs/debug_ctc/audit_report.json"))
    ap.add_argument("--output-examples", default=str(ROOT / "outputs/debug_ctc/audit_examples.jsonl"))
    ap.add_argument("--content-tokenizer", default="", help="Optional tokenizer path overriding per-record content_vocab_path.")
    ap.add_argument("--blank-id", type=int, default=None, help="Override content_ctc_blank_id.")
    ap.add_argument("--token-offset", type=int, default=None, help="Override content_ctc_token_offset.")
    ap.add_argument("--min-target-len", type=int, default=3)
    ap.add_argument("--high-punct-ratio", type=float, default=0.5)
    ap.add_argument("--examples-per-category", type=int, default=20)
    return ap.parse_args()


def nested_get(row: dict[str, Any], key: str) -> Any | None:
    value = row.get(key)
    if value not in (None, ""):
        return value
    meta = row.get("moss_codecvc_meta")
    if isinstance(meta, dict):
        value = meta.get(key)
        if value not in (None, ""):
            return value
    return None


def normalize_manifest_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    return p


def normalize_text(text: Any) -> str:
    value = str(text or "").replace("\t", " ").replace("\n", " ").strip()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def pick_text(row: dict[str, Any]) -> tuple[str, str]:
    for key in DEFAULT_TEXT_KEYS:
        value = nested_get(row, key)
        if value in (None, ""):
            continue
        text = normalize_text(value)
        if text not in NO_TEXT_VALUES:
            return text, key
    return "", ""


def parse_optional_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def path_from_record(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = nested_get(row, key)
        if value:
            return str(value)
    return ""


def make_abs_record_path(path: str, manifest_path: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (manifest_path.parent / p).resolve(strict=False)


def tensor_to_list(value: Any) -> list[int]:
    try:
        import torch

        if torch.is_tensor(value):
            return [int(item) for item in value.detach().cpu().long().flatten().tolist()]
    except Exception:
        pass
    if value is None or isinstance(value, str):
        return []
    if isinstance(value, dict):
        for key in TOKEN_PAYLOAD_KEYS:
            if key in value:
                return tensor_to_list(value[key])
        return []
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        stack: list[Any] = list(value)
        while stack:
            item = stack.pop(0)
            if isinstance(item, (list, tuple)):
                stack = list(item) + stack
            else:
                try:
                    out.append(int(item))
                except (TypeError, ValueError):
                    pass
        return out
    return []


def load_token_ids_from_path(path: str, manifest_path: Path) -> list[int]:
    if not path:
        return []
    p = make_abs_record_path(path, manifest_path)
    if not p.exists():
        return []
    try:
        if p.suffix.lower() in {".json", ".jsonl"}:
            payload = json.loads(p.read_text(encoding="utf-8"))
        elif p.suffix.lower() == ".npy":
            import numpy as np

            payload = np.load(str(p), allow_pickle=False)
        else:
            payload = load_torch_file(p, map_location="cpu")
    except Exception:
        return []
    return tensor_to_list(payload)


def extract_token_ids(row: dict[str, Any], manifest_path: Path) -> tuple[list[int], str]:
    path = path_from_record(row, TOKEN_PATH_KEYS)
    if path:
        ids = load_token_ids_from_path(path, manifest_path)
        if ids:
            return ids, path
    for key in TOKEN_VALUE_KEYS:
        ids = tensor_to_list(nested_get(row, key))
        if ids:
            return ids, key
    return [], path or ""


def reservoir_sample_jsonl(path: Path, *, sample_size: int, max_rows: int, seed: int) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    rng = random.Random(seed)
    rows: list[tuple[int, dict[str, Any]]] = []
    scanned = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if max_rows > 0 and scanned >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            scanned += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
            item = (line_no, row)
            if sample_size <= 0:
                rows.append(item)
            elif len(rows) < sample_size:
                rows.append(item)
            else:
                idx = rng.randint(0, scanned - 1)
                if idx < sample_size:
                    rows[idx] = item
    return rows, scanned


class DecoderCache:
    def __init__(self, override_path: str = "") -> None:
        self.override_path = override_path
        self._cache: dict[tuple[str, str], Any] = {}

    def resolve_path(self, row: dict[str, Any], manifest_path: Path) -> str:
        path = self.override_path or str(nested_get(row, "content_vocab_path") or "")
        if not path:
            return ""
        return str(make_abs_record_path(path, manifest_path))

    def get(self, row: dict[str, Any], manifest_path: Path) -> Any | None:
        path = self.resolve_path(row, manifest_path)
        tokenizer = str(nested_get(row, "content_tokenizer") or "").lower()
        if path.endswith(".model") or tokenizer == "sentencepiece":
            kind = "sentencepiece"
        elif path.endswith(".json") or tokenizer in {"char", "byte"}:
            kind = "json_vocab"
        else:
            return None
        key = (kind, path)
        if key in self._cache:
            return self._cache[key]
        try:
            if kind == "sentencepiece":
                import sentencepiece as spm

                decoder = spm.SentencePieceProcessor(model_file=path)
            else:
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
                symbols = payload.get("symbols") if isinstance(payload, dict) else None
                if not isinstance(symbols, dict):
                    return None
                inverse = {int(v): str(k) for k, v in symbols.items()}
                decoder = inverse
        except Exception as exc:
            decoder = {"__decode_error__": f"{type(exc).__name__}: {exc}"}
        self._cache[key] = decoder
        return decoder

    def decode(self, row: dict[str, Any], manifest_path: Path, ids: list[int], offset: int) -> tuple[str, str]:
        decoder = self.get(row, manifest_path)
        if decoder is None:
            return "", "missing_decoder"
        if isinstance(decoder, dict) and "__decode_error__" in decoder:
            return "", str(decoder["__decode_error__"])
        raw_ids = [int(item) - int(offset) for item in ids if int(item) - int(offset) >= 0]
        if not raw_ids:
            return "", ""
        try:
            if hasattr(decoder, "decode"):
                return str(decoder.decode(raw_ids)), ""
            pieces = []
            for item in ids:
                symbol = decoder.get(int(item))
                if symbol is None:
                    continue
                if symbol.startswith("c:"):
                    pieces.append(symbol[2:])
                elif symbol.startswith("b:"):
                    try:
                        pieces.append(bytes([int(symbol[2:])]).decode("utf-8", errors="ignore"))
                    except ValueError:
                        pass
                else:
                    pieces.append(symbol)
            return "".join(pieces), ""
        except Exception as exc:
            return "", f"{type(exc).__name__}: {exc}"

    def token_is_punctuation(self, row: dict[str, Any], manifest_path: Path, token_id: int, offset: int) -> bool:
        text, error = self.decode(row, manifest_path, [int(token_id)], offset)
        if error:
            return False
        compact = text.replace(" ", "").replace("\u2581", "")
        if not compact:
            return False
        return bool(PUNCT_OR_SYMBOL_RE.match(compact)) and not bool(CONTENT_CHAR_RE.search(compact))


def infer_input_length(row: dict[str, Any]) -> int | None:
    for key in (
        "target_codec_frames",
        "source_codec_frames",
        "audio_codec_frames",
        "codec_frames",
        "target_frames",
        "source_frames",
    ):
        value = parse_optional_int(nested_get(row, key))
        if value is not None and value > 0:
            return value
    audio_codes = nested_get(row, "audio_codes")
    if isinstance(audio_codes, list) and audio_codes:
        first = audio_codes[0]
        if isinstance(first, list):
            return len(audio_codes)
    return None


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p05": None, "p50": None, "p95": None, "max": None, "mean": None}
    ordered = sorted(float(v) for v in values)

    def pick(frac: float) -> float:
        idx = int(round((len(ordered) - 1) * frac))
        return ordered[max(0, min(len(ordered) - 1, idx))]

    return {
        "min": ordered[0],
        "p05": pick(0.05),
        "p50": pick(0.50),
        "p95": pick(0.95),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def add_example(
    examples: dict[str, list[dict[str, Any]]],
    category: str,
    row: dict[str, Any],
    *,
    manifest: Path,
    line_no: int,
    text_key: str,
    raw_text: str,
    decoded_text: str,
    ids: list[int],
    input_len: int | None,
    punct_ratio: float,
    limit: int,
) -> None:
    bucket = examples[category]
    if len(bucket) >= limit:
        return
    bucket.append(
        {
            "category": category,
            "manifest": str(manifest),
            "line_no": int(line_no),
            "sample_id": row.get("sample_id") or row.get("case_id") or row.get("eval_id"),
            "mode": row.get("moss_codecvc_mode") or row.get("mode"),
            "language": row.get("language") or row.get("source_lang"),
            "text_key": text_key,
            "content_ref_text": raw_text,
            "decoded_text": decoded_text,
            "token_len": len(ids),
            "input_len": input_len,
            "punctuation_token_ratio": punct_ratio,
            "content_token_ids_head": ids[:64],
        }
    )


def audit_manifest(
    manifest_path: Path,
    *,
    args: argparse.Namespace,
    seed: int,
    decoder_cache: DecoderCache,
    examples: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    sampled, scanned = reservoir_sample_jsonl(
        manifest_path,
        sample_size=int(args.sample_size),
        max_rows=int(args.max_rows),
        seed=seed,
    )
    counters: Counter[str] = Counter()
    token_lengths: list[float] = []
    input_lengths: list[float] = []
    input_target_ratios: list[float] = []
    punct_ratios: list[float] = []
    tokenizer_meta: dict[str, Counter[str]] = defaultdict(Counter)

    for line_no, row in sampled:
        counters["sampled"] += 1
        raw_text, text_key = pick_text(row)
        normalized_text = normalize_text(raw_text)
        ids_raw, ids_source = extract_token_ids(row, manifest_path)
        blank_id = int(args.blank_id if args.blank_id is not None else parse_optional_int(nested_get(row, "content_ctc_blank_id"), 0))
        offset = int(
            args.token_offset
            if args.token_offset is not None
            else parse_optional_int(nested_get(row, "content_ctc_token_offset"), 1)
        )
        valid_ids = [int(item) for item in ids_raw if int(item) >= 0 and int(item) != blank_id]
        input_len = infer_input_length(row)
        decoded_text, decode_error = decoder_cache.decode(row, manifest_path, valid_ids, offset)
        decoded_compact = decoded_text.replace(" ", "").replace("\u2581", "")
        punct_tokens = sum(
            1 for item in valid_ids if decoder_cache.token_is_punctuation(row, manifest_path, int(item), offset)
        )
        punct_ratio = float(punct_tokens) / float(len(valid_ids)) if valid_ids else 0.0

        token_lengths.append(float(len(valid_ids)))
        punct_ratios.append(float(punct_ratio))
        if input_len is not None:
            input_lengths.append(float(input_len))
            if valid_ids:
                input_target_ratios.append(float(input_len) / float(len(valid_ids)))
            if valid_ids and input_len <= len(valid_ids):
                counters["input_len_le_target_len"] += 1

        for key in ("content_ctc_vocab_size", "content_ctc_blank_id", "content_ctc_token_offset", "content_tokenizer", "content_tokenizer_id", "content_vocab_path"):
            value = nested_get(row, key)
            if value not in (None, ""):
                tokenizer_meta[key][str(value)] += 1
        if ids_source:
            tokenizer_meta["content_token_source"][str(ids_source)] += 1

        if not normalized_text:
            counters["empty_text"] += 1
        if not ids_raw:
            counters["missing_or_empty_token_ids"] += 1
        if any(int(item) == blank_id for item in ids_raw):
            counters["contains_blank_id"] += 1
        if any(int(item) < 0 for item in ids_raw):
            counters["contains_negative_id"] += 1
        vocab_size = parse_optional_int(nested_get(row, "content_ctc_vocab_size"))
        if vocab_size is not None and any(int(item) >= vocab_size for item in ids_raw):
            counters["contains_oov_id"] += 1
        if decode_error:
            counters["decode_error"] += 1
        if not decoded_compact and valid_ids:
            counters["decoded_empty"] += 1
        decoded_only_punctuation = bool(decoded_compact) and bool(PUNCT_OR_SYMBOL_RE.match(decoded_compact)) and not bool(
            CONTENT_CHAR_RE.search(decoded_compact)
        )
        if decoded_only_punctuation:
            counters["decoded_only_punctuation"] += 1
        if decoded_only_punctuation and len(decoded_compact) == 1:
            counters["single_punctuation_decoded"] += 1
        if len(valid_ids) < int(args.min_target_len):
            counters["target_too_short"] += 1
        if punct_ratio >= float(args.high_punct_ratio):
            counters["punctuation_ratio_high"] += 1

        common = {
            "manifest": manifest_path,
            "line_no": line_no,
            "text_key": text_key,
            "raw_text": raw_text,
            "decoded_text": decoded_text if not decode_error else f"<decode_error:{decode_error}>",
            "ids": valid_ids,
            "input_len": input_len,
            "punct_ratio": punct_ratio,
            "limit": int(args.examples_per_category),
        }
        if decoded_only_punctuation:
            add_example(examples, "decoded_only_punctuation", row, **common)
        if not decoded_compact and valid_ids:
            add_example(examples, "decoded_empty", row, **common)
        if any(int(item) == blank_id for item in ids_raw):
            add_example(examples, "target_contains_blank", row, **common)
        if any(int(item) < 0 for item in ids_raw):
            add_example(examples, "target_contains_negative", row, **common)
        if len(valid_ids) < int(args.min_target_len):
            add_example(examples, "target_too_short", row, **common)
        if punct_ratio >= float(args.high_punct_ratio):
            add_example(examples, "punctuation_ratio_high", row, **common)
        if input_len is not None and valid_ids and input_len <= len(valid_ids):
            add_example(examples, "input_len_le_target_len", row, **common)
        if decode_error:
            add_example(examples, "decode_error", row, **common)

    rows = int(counters["sampled"])
    meta_summary: dict[str, Any] = {}
    metadata_consistent = True
    for key, values in sorted(tokenizer_meta.items()):
        top_values = values.most_common(10)
        meta_summary[key] = {
            "unique": len(values),
            "top": [{"value": value, "count": count} for value, count in top_values],
        }
        if key in {"content_ctc_vocab_size", "content_ctc_blank_id", "content_ctc_token_offset", "content_tokenizer", "content_tokenizer_id", "content_vocab_path"} and len(values) > 1:
            metadata_consistent = False

    return {
        "manifest": str(manifest_path),
        "rows_scanned": scanned,
        "rows_sampled": rows,
        "sample_size_requested": int(args.sample_size),
        "max_rows": int(args.max_rows),
        "metadata_consistent_within_manifest": metadata_consistent,
        "tokenizer_metadata": meta_summary,
        "counts": dict(counters),
        "rates": {
            "empty_text": ratio(counters["empty_text"], rows),
            "missing_or_empty_token_ids": ratio(counters["missing_or_empty_token_ids"], rows),
            "contains_blank_id": ratio(counters["contains_blank_id"], rows),
            "contains_negative_id": ratio(counters["contains_negative_id"], rows),
            "contains_oov_id": ratio(counters["contains_oov_id"], rows),
            "decoded_empty": ratio(counters["decoded_empty"], rows),
            "decoded_only_punctuation": ratio(counters["decoded_only_punctuation"], rows),
            "single_punctuation_decoded": ratio(counters["single_punctuation_decoded"], rows),
            "target_too_short": ratio(counters["target_too_short"], rows),
            "punctuation_ratio_high": ratio(counters["punctuation_ratio_high"], rows),
            "input_len_le_target_len": ratio(counters["input_len_le_target_len"], rows),
            "decode_error": ratio(counters["decode_error"], rows),
        },
        "target_length": quantiles(token_lengths),
        "input_length": quantiles(input_lengths),
        "input_length_over_target_length": quantiles(input_target_ratios),
        "punctuation_token_ratio": quantiles(punct_ratios),
    }


def merge_global(manifest_reports: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    rows = 0
    for report in manifest_reports:
        rows += int(report.get("rows_sampled") or 0)
        counts.update(report.get("counts") or {})
    return {
        "rows_sampled": rows,
        "counts": dict(counts),
        "rates": {
            "empty_text": ratio(counts["empty_text"], rows),
            "missing_or_empty_token_ids": ratio(counts["missing_or_empty_token_ids"], rows),
            "contains_blank_id": ratio(counts["contains_blank_id"], rows),
            "contains_negative_id": ratio(counts["contains_negative_id"], rows),
            "contains_oov_id": ratio(counts["contains_oov_id"], rows),
            "decoded_empty": ratio(counts["decoded_empty"], rows),
            "decoded_only_punctuation": ratio(counts["decoded_only_punctuation"], rows),
            "single_punctuation_decoded": ratio(counts["single_punctuation_decoded"], rows),
            "target_too_short": ratio(counts["target_too_short"], rows),
            "punctuation_ratio_high": ratio(counts["punctuation_ratio_high"], rows),
            "input_len_le_target_len": ratio(counts["input_len_le_target_len"], rows),
            "decode_error": ratio(counts["decode_error"], rows),
        },
    }


def write_examples(path: Path, examples: dict[str, list[dict[str, Any]]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for category in sorted(examples):
            for row in examples[category]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    return count


def main() -> int:
    args = parse_args()
    manifests = [normalize_manifest_path(item) for item in args.manifest] if args.manifest else list(DEFAULT_MANIFESTS)
    decoder_cache = DecoderCache(args.content_tokenizer)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reports = []
    for idx, manifest in enumerate(manifests):
        reports.append(
            audit_manifest(
                manifest,
                args=args,
                seed=int(args.seed) + idx,
                decoder_cache=decoder_cache,
                examples=examples,
            )
        )
    report = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve(strict=False)),
        "args": {
            "sample_size": int(args.sample_size),
            "max_rows": int(args.max_rows),
            "seed": int(args.seed),
            "blank_id_override": args.blank_id,
            "token_offset_override": args.token_offset,
            "content_tokenizer_override": args.content_tokenizer,
            "min_target_len": int(args.min_target_len),
            "high_punct_ratio": float(args.high_punct_ratio),
        },
        "global": merge_global(reports),
        "manifests": reports,
    }
    report_path = Path(args.output_report).expanduser()
    examples_path = Path(args.output_examples).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    examples_count = write_examples(examples_path, examples)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "examples": str(examples_path),
                "examples_count": examples_count,
                "global": report["global"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

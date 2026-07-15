#!/usr/bin/env python3
"""Prepare and run Path X on the strict Batch-42 VC manifests.

This adapter keeps the paper-facing Batch-42 input/output contract separate
from the historical SeedTTS-320 driver.  The official five-column VC rows are
mapped as follows:

* field 5 (``source_audio``) -> Path X content/source waveform;
* field 3 (``prompt_audio``) -> Path X timbre reference waveform;
* field 4 (``target_text``) -> scorer reference text only;
* every inference row -> ``mode=no_text`` so the WavLM-BNF side path runs.

The actual model forward remains in ``004044``.  This script prepares its
canonical no-text JSONL, invokes one persistent engine per shard, and converts
the historical inference ledger into the same schema consumed by 004089 and
004091 for the external baselines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "moss_codecvc.baseline_vc_infer.v1"
SYSTEM_ID = "path_x_3k"

PROJECT_ROOT = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC"
)
REGISTERED_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs/lora_runs/"
    "ver23_content_side_3k_olddata_textrep10_"
    "ver23_content_side_text_bypass_3k_20260710/step-3000"
)
REGISTERED_CODE_ROOT = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
    "MOSS-CodecVC_snapshots/ver23_batch3436_eval_20260711_1092820"
)
REGISTERED_ENGINE_SCRIPT = (
    REGISTERED_CODE_ROOT / "scripts/004044_run_seedtts_validation_infer_persistent.py"
)
REGISTERED_BASE_MODEL_PATH = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
    "vcdata_construction/MOSS-TTS"
)
REGISTERED_BASE_MODEL_CONFIG_SHA256 = (
    "214fc997d98f51ab57925a5939afc6280e76044198b664221622e70d098ed06e"
)
REGISTERED_SOURCE_SEMANTIC_MODEL = Path(
    "/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/huggingface/"
    "models--microsoft--wavlm-base-plus/snapshots/"
    "4c66d4806a428f2e922ccfa1a962776e232d487b"
)
REGISTERED_SOURCE_SEMANTIC_CACHE = Path(
    "/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/huggingface"
)
REGISTERED_SOURCE_SEMANTIC_FILES = {
    "config.json": {
        "size": 2_232,
        "sha256": "fea6df1c2700a3954fc07e70588aecc9055eeb28db2ff57151a2db0d19180ed4",
    },
    "preprocessor_config.json": {
        "size": 215,
        "sha256": "99272fe8ccfab114b68b478681ea47ee3a1ce62bb788cb92dd6e4f69fb1f1da2",
    },
    "pytorch_model.bin": {
        "size": 377_617_425,
        "sha256": "3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51",
    },
}

REGISTERED_MODEL_FILES = {
    "README.md": {
        "size": 179,
        "sha256": "4d45f7d68a88a39671cc0cbc86f1acdfbee5351401eee2a97df253f0d077717f",
    },
    "adapter_config.json": {
        "size": 1_179,
        "sha256": "06530eac22376a6befd9e81c95c333e4bb1c889de96e9059c2d5498cd90a7aee",
    },
    "adapter_model.safetensors": {
        "size": 87_366_096,
        "sha256": "3a51162fc7ccf1b9e1aa477ad7c44fa64390d109b8b63765a9cd636f090f4b25",
    },
    "timbre_memory_config.json": {
        "size": 5_026,
        "sha256": "5c8842d87327c2cf1af2697725a19bf2b53ba654fa0a6b3f68b6a42fd50e9970",
    },
    "timbre_memory_adapter.pt": {
        "size": 1_697_093_491,
        "sha256": "020a16ad4bba5a812b2f62e29cb68dcec9d4055344e02de01555be8afd9d6895",
    },
}
REGISTERED_CODE_FILES = {
    "scripts/004044_run_seedtts_validation_infer_persistent.py": (
        "c9dec31f4155d39cdbd02069dd8b91677ff5dee03e98d441377d949135a8e709"
    ),
    "scripts/003001_infer_moss_codecvc.py": (
        "d9a3426a3668a4bdd95a81fdf86b02e32d774b4893ba0428e5b1c6fba4f5ce73"
    ),
    "moss_codecvc/models/moss_codecvc_wrapper.py": (
        "5815c8ab5e0aab69d19328fd01782620064327eaf5f39cc4923df8ce3ae9ca42"
    ),
    "moss_codecvc/models/content_cross_attn.py": (
        "a8e4cd12d279cfff7c38e3e2d8b21b55d70c403cec654edf7ef77de58acba66a"
    ),
}

# 004044 accepts many defaults from environment variables.  A QZ runner may
# inherit variables from an earlier experiment, so clear every behavior knob
# that can silently change Batch-33 inference before adding the registered
# values back through explicit CLI arguments.
INFERENCE_ENV_KEYS = {
    "AUDIO_REPETITION_PENALTY",
    "AUDIO_SEGMENT_POLICY",
    "AUDIO_TEMPERATURE",
    "AUDIO_TOP_K",
    "AUDIO_TOP_P",
    "BASE_MODEL_PATH",
    "CONFIG",
    "DEBUG_GENERATION_STRUCTURE",
    "DEVICE",
    "DISABLE_MODE_TOKEN",
    "DISABLE_SOURCE_SEMANTIC_MEMORY",
    "DISABLE_SOURCE_SEMANTIC_MONOTONIC_BIAS",
    "DISABLE_TIMBRE_MEMORY",
    "FILTER_V2_REAL_NO_TEXT_REF_CONTENT_LEAK",
    "INSTRUCTION",
    "LANGUAGE",
    "MAX_NEW_TOKENS",
    "MIN_AUDIO_TOKENS",
    "MIN_NEW_TOKENS",
    "MOSS_TTS_ATTN_IMPLEMENTATION",
    "N_VQ",
    "NO_TEXT_AUDIO_REPETITION_PENALTY",
    "NO_TEXT_AUDIO_TEMPERATURE",
    "NO_TEXT_AUDIO_TOP_K",
    "NO_TEXT_AUDIO_TOP_P",
    "NO_TEXT_DURATION_BUDGET_RATIO",
    "NO_TEXT_MAX_TOKEN_MARGIN",
    "NO_TEXT_PLACEHOLDER",
    "NO_TEXT_SOFT_DURATION_BUDGET",
    "NO_TEXT_SOFT_EXTRA_TOKEN_MARGIN",
    "NO_TEXT_SOFT_MIN_AUDIO_RATIO",
    "NO_TEXT_SOURCE_GATE_FLOOR",
    "REF_AUDIO_CFG_SCALE",
    "REF_PROMPT_CODEC_PERMUTATION",
    "REF_PROMPT_CODEC_PERMUTATION_BLOCK_SECONDS",
    "REF_PROMPT_CODEC_PERMUTATION_BOOTSTRAP",
    "REF_PROMPT_CODEC_PERMUTATION_FRAME_RATE",
    "REF_PROMPT_CODEC_PERMUTATION_MAX_SECONDS",
    "REF_PROMPT_CODEC_PERMUTATION_MIN_SECONDS",
    "REF_PROMPT_CODEC_PERMUTATION_MODE",
    "REF_PROMPT_CODEC_PERMUTATION_SEED",
    "REF_SPEAKER_PROMPT_ATTENTION_CAPTURE_FRAMES",
    "REF_SPEAKER_PROMPT_ATTENTION_LAYERS",
    "REF_SPEAKER_PROMPT_SLOT",
    "REF_SPEAKER_PROMPT_TOKENS",
    "SAVE_GENERATION_IDS_DIR",
    "SEED",
    "SOURCE_CONTENT_SPM_MODEL",
    "SOURCE_CONTENT_TOKEN_IDS",
    "SOURCE_CONTENT_TOKEN_IDS_PATH",
    "SOURCE_SEMANTIC_CACHE_DIR",
    "SOURCE_SEMANTIC_DEVICE",
    "SOURCE_SEMANTIC_DOWNSAMPLE_STRIDE",
    "SOURCE_SEMANTIC_DTYPE",
    "SOURCE_SEMANTIC_FEATURE_PATH",
    "SOURCE_SEMANTIC_LAYER",
    "SOURCE_SEMANTIC_LOCAL_FILES_ONLY",
    "SOURCE_SEMANTIC_MODEL_NAME_OR_PATH",
    "SOURCE_SEMANTIC_MONOTONIC_BIAS_STRENGTH",
    "SOURCE_SEMANTIC_MONOTONIC_BIAS_WIDTH",
    "SOURCE_SEMANTIC_POSITION_SCALE",
    "SOURCE_SEMANTIC_PROGRESS_CLOCK",
    "SOURCE_SEMANTIC_RELEASE_AFTER_PROGRESS",
    "SOURCE_SEMANTIC_RELEASE_START",
    "SPEAKER_EMBEDDING_DIM",
    "SPEAKER_ENCODER_PATH",
    "SPEAKER_ENCODER_TYPE",
    "SPEAKER_SEQ_PATH",
    "SPEAKER_VEC_PATH",
    "TEMPERATURE",
    "TEXT_AUTO_MAX_NEW_TOKENS",
    "TEXT_CJK_CHARS_PER_SECOND",
    "TEXT_DURATION_MARGIN",
    "TEXT_EXTRA_NEW_TOKENS",
    "TEXT_LATIN_WORDS_PER_SECOND",
    "TEXT_MIN_NEW_TOKENS_FLOOR",
    "TIMBRE_CFG_SCALE",
    "TIMBRE_REF_SPEAKER_EMBEDDING_PATH",
    "TIMBRE_SIDE_ONLY",
    "TOP_K",
    "TOP_P",
}


@dataclass(frozen=True)
class StrictCase:
    input_index: int
    input_line: int
    case_id: str
    case_uid: str
    prompt_text: str
    target_text: str
    source_audio: Path
    reference_audio: Path
    language: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def require_within(path: Path, root: Path, *, role: str) -> None:
    try:
        resolved(path).relative_to(resolved(root))
    except ValueError as exc:
        raise ValueError(f"{role} escapes registered input root: {path} not under {root}") from exc


def validate_registered_assets(
    args: argparse.Namespace,
    *,
    verify_large_hashes: bool = False,
) -> dict[str, Any]:
    model_path = resolved(args.model_path)
    code_root = resolved(args.code_root)
    engine_script = resolved(args.engine_script)
    base_model_path = resolved(args.base_model_path)
    source_semantic_model = resolved(args.source_semantic_model)
    source_semantic_cache = resolved(args.source_semantic_cache)
    expected = {
        "model_path": resolved(REGISTERED_MODEL_PATH),
        "code_root": resolved(REGISTERED_CODE_ROOT),
        "engine_script": resolved(REGISTERED_ENGINE_SCRIPT),
        "base_model_path": resolved(REGISTERED_BASE_MODEL_PATH),
        "source_semantic_model": resolved(REGISTERED_SOURCE_SEMANTIC_MODEL),
        "source_semantic_cache": resolved(REGISTERED_SOURCE_SEMANTIC_CACHE),
    }
    observed = {
        "model_path": model_path,
        "code_root": code_root,
        "engine_script": engine_script,
        "base_model_path": base_model_path,
        "source_semantic_model": source_semantic_model,
        "source_semantic_cache": source_semantic_cache,
    }
    for name, expected_path in expected.items():
        if observed[name] != expected_path:
            raise ValueError(
                f"registered Path X identity mismatch for {name}: "
                f"expected {expected_path}, got {observed[name]}"
            )

    for role, directory in (
        ("checkpoint", model_path),
        ("frozen code", code_root),
        ("base model", base_model_path),
        ("WavLM snapshot", source_semantic_model),
        ("Hugging Face cache", source_semantic_cache),
    ):
        if not directory.is_dir():
            raise FileNotFoundError(f"registered {role} directory is missing: {directory}")

    model_files: dict[str, Any] = {}
    for name, registration in REGISTERED_MODEL_FILES.items():
        path = model_path / name
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        expected_size = int(registration["size"])
        if size != expected_size:
            raise ValueError(
                f"registered checkpoint size mismatch for {name}: "
                f"expected {expected_size}, got {size}"
            )
        should_hash = verify_large_hashes or size < 10_000_000
        actual_sha256 = sha256_file(path) if should_hash else "size_only_in_worker"
        if should_hash and actual_sha256 != registration["sha256"]:
            raise ValueError(
                f"registered checkpoint SHA256 mismatch for {name}: "
                f"expected {registration['sha256']}, got {actual_sha256}"
            )
        model_files[name] = {
            "path": str(path),
            "size": size,
            "sha256": actual_sha256,
            "registered_sha256": registration["sha256"],
        }

    code_files: dict[str, Any] = {}
    for relative, expected_sha256 in REGISTERED_CODE_FILES.items():
        path = code_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"registered frozen-code SHA256 mismatch for {relative}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        code_files[relative] = {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": actual_sha256,
        }

    adapter_config = json.loads(
        (model_path / "adapter_config.json").read_text(encoding="utf-8")
    )
    adapter_base_value = str(adapter_config.get("base_model_name_or_path") or "").strip()
    if not adapter_base_value:
        raise ValueError("registered adapter_config.json has no base_model_name_or_path")
    adapter_base_path = resolved(Path(adapter_base_value))
    if adapter_base_path != base_model_path:
        raise ValueError(
            "registered adapter base_model_name_or_path mismatch: "
            f"expected {base_model_path}, got {adapter_base_path}"
        )

    base_config = base_model_path / "config.json"
    if not base_config.is_file():
        raise FileNotFoundError(base_config)
    base_config_sha256 = sha256_file(base_config)
    if base_config_sha256 != REGISTERED_BASE_MODEL_CONFIG_SHA256:
        raise ValueError(
            "registered base-model config SHA256 mismatch: "
            f"expected {REGISTERED_BASE_MODEL_CONFIG_SHA256}, got {base_config_sha256}"
        )

    source_semantic_files: dict[str, Any] = {}
    for name, registration in REGISTERED_SOURCE_SEMANTIC_FILES.items():
        path = source_semantic_model / name
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        expected_size = int(registration["size"])
        if size != expected_size:
            raise ValueError(
                f"registered WavLM snapshot size mismatch for {name}: "
                f"expected {expected_size}, got {size}"
            )
        should_hash = verify_large_hashes or size < 10_000_000
        actual_sha256 = sha256_file(path) if should_hash else "size_only_in_worker"
        if should_hash and actual_sha256 != registration["sha256"]:
            raise ValueError(
                f"registered WavLM snapshot SHA256 mismatch for {name}: "
                f"expected {registration['sha256']}, got {actual_sha256}"
            )
        source_semantic_files[name] = {
            "path": str(path),
            "size": size,
            "sha256": actual_sha256,
            "registered_sha256": registration["sha256"],
        }

    timbre_config = json.loads(
        (model_path / "timbre_memory_config.json").read_text(encoding="utf-8")
    )
    required_config = {
        "content_cross_attn_enabled": True,
        "content_cross_attn_layers": "all",
        "content_cross_attn_feature_dim": 768,
        "content_cross_attn_gate_init": -0.5,
        "content_cross_attn_output_scale": 0.3,
        "content_encoder_layers": 2,
        "guided_attn_loss_weight": 0.05,
        "phoneme_classifier_loss_weight": 0.02,
        "progress_loss_weight": 0.1,
        "stop_loss_weight": 0.2,
        "source_semantic_memory_enabled": False,
        "speaker_side_pathway_enabled": False,
        "speaker_cross_attn_enabled": False,
        "source_content_memory_type": "wavlm_bnf_continuous",
    }
    mismatches = {
        key: {"expected": value, "actual": timbre_config.get(key)}
        for key, value in required_config.items()
        if timbre_config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"registered Batch-33 timbre config mismatch: {mismatches}")
    return {
        "schema_version": "moss_codecvc.batch42_pathx_registered_identity.v1",
        "model_path": str(model_path),
        "code_root": str(code_root),
        "engine_script": str(engine_script),
        "base_model_path": str(base_model_path),
        "base_model_config": {
            "path": str(base_config),
            "size": base_config.stat().st_size,
            "sha256": base_config_sha256,
        },
        "adapter_base_model_name_or_path": str(adapter_base_path),
        "source_semantic_model": str(source_semantic_model),
        "source_semantic_cache": str(source_semantic_cache),
        "large_checkpoint_hashes_verified": bool(verify_large_hashes),
        "model_files": model_files,
        "code_files": code_files,
        "source_semantic_files": source_semantic_files,
        "registered_config": required_config,
        "actual_base_language_layers": 36,
    }


def stable_case_uid(case_id: str, source_audio: Path, reference_audio: Path) -> str:
    payload = "\0".join(
        (
            case_id,
            str(source_audio.resolve(strict=False)),
            str(reference_audio.resolve(strict=False)),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return stem[:180] or "case"


def require_unique_safe_stems(case_ids: Sequence[str], *, context: str) -> None:
    by_stem: dict[str, str] = {}
    for case_id in case_ids:
        stem = safe_stem(case_id)
        previous = by_stem.get(stem)
        if previous is not None and previous != case_id:
            raise ValueError(
                f"{context} has safe_stem collision {stem!r}: "
                f"{previous!r} and {case_id!r} would share one output WAV"
            )
        by_stem[stem] = case_id


def resolve_audio(value: str, input_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = input_root / path
    return path.resolve(strict=False)


def parse_strict_lst(
    path: Path,
    *,
    input_root: Path,
    language: str,
    expected_cases: int,
    expected_sha256: str,
) -> list[StrictCase]:
    path = path.expanduser().resolve()
    input_root = input_root.expanduser().resolve()
    if language not in {"en", "zh"}:
        raise ValueError(f"unsupported language: {language!r}")
    if not path.is_file():
        raise FileNotFoundError(path)
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    observed_hash = sha256_file(path)
    if observed_hash != expected_sha256:
        raise ValueError(
            f"strict manifest SHA256 mismatch: expected {expected_sha256}, got {observed_hash}"
        )

    cases: list[StrictCase] = []
    with path.open(encoding="utf-8") as handle:
        for input_index, raw in enumerate(handle):
            input_line = input_index + 1
            raw = raw.rstrip("\r\n")
            if not raw.strip():
                raise ValueError(f"{path}:{input_line}: blank rows are forbidden")
            fields = [item.strip() for item in raw.split("|")]
            if len(fields) != 5:
                raise ValueError(
                    f"{path}:{input_line}: expected five fields "
                    "id|prompt_text|prompt_audio|target_text|source_audio, "
                    f"got {len(fields)}"
                )
            case_id, prompt_text, prompt_audio, target_text, source_audio = fields
            if not all((case_id, prompt_text, prompt_audio, target_text, source_audio)):
                raise ValueError(f"{path}:{input_line}: required field is empty")
            source_path = resolve_audio(source_audio, input_root)
            reference_path = resolve_audio(prompt_audio, input_root)
            require_within(source_path, input_root, role="source audio")
            require_within(reference_path, input_root, role="reference audio")
            if source_path == reference_path:
                raise ValueError(
                    f"{path}:{input_line}: source and reference resolve to the same audio"
                )
            missing = [
                str(audio_path)
                for audio_path in (source_path, reference_path)
                if not audio_path.is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"{path}:{input_line}: missing source/reference audio: {missing}"
                )
            cases.append(
                StrictCase(
                    input_index=input_index,
                    input_line=input_line,
                    case_id=case_id,
                    case_uid=stable_case_uid(case_id, source_path, reference_path),
                    prompt_text=prompt_text,
                    target_text=target_text,
                    source_audio=source_path,
                    reference_audio=reference_path,
                    language=language,
                )
            )

    if len(cases) != expected_cases:
        raise ValueError(f"expected {expected_cases} strict cases, got {len(cases)}")
    case_ids = [case.case_id for case in cases]
    case_uids = [case.case_uid for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("strict manifest has duplicate case_id values")
    if len(set(case_uids)) != len(case_uids):
        raise ValueError("strict manifest has duplicate case_uid values")
    require_unique_safe_stems(case_ids, context="strict manifest")
    return cases


def canonical_record(case: StrictCase, *, test_set_id: str) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "case_uid": case.case_uid,
        "input_index": case.input_index,
        "input_line": case.input_line,
        "mode": "no_text",
        "moss_codecvc_mode": "no_text",
        "cell": f"batch42_strict_{case.language}",
        "language": case.language,
        "source_lang": case.language,
        "ref_lang": case.language,
        "source_audio": str(case.source_audio),
        "timbre_ref_audio": str(case.reference_audio),
        "reference_audio": str(case.reference_audio),
        "text": "<NO_TEXT>",
        "source_text": case.target_text,
        "content_ref_text": case.target_text,
        "timbre_ref_text": case.prompt_text,
        "target_text": case.target_text,
        "reference_text": case.target_text,
        "prompt_text": case.prompt_text,
        "test_set_id": test_set_id,
        "batch42_field_mapping": {
            "source_audio": "field_5/source_audio",
            "timbre_ref_audio": "field_3/prompt_audio",
            "target_text": "field_4/target_text; scorer-only in no_text mode",
        },
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def audit_canonical_rows(
    rows: list[dict[str, Any]],
    *,
    canonical_path: Path,
    expected_cases: int,
    expected_sha256: str,
    test_set_id: str,
    language: str,
    input_root: Path,
) -> dict[str, Any]:
    canonical_path = resolved(canonical_path)
    input_root = resolved(input_root)
    if len(rows) != expected_cases:
        raise ValueError(
            f"canonical denominator mismatch: expected {expected_cases}, got {len(rows)}"
        )
    digest = sha256_file(canonical_path)
    if digest != expected_sha256:
        raise ValueError(
            f"canonical JSONL SHA256 mismatch: expected {expected_sha256}, got {digest}"
        )
    case_ids = [str(row.get("case_id") or "") for row in rows]
    case_uids = [str(row.get("case_uid") or "") for row in rows]
    if any(not value for value in case_ids) or len(set(case_ids)) != len(rows):
        raise ValueError("canonical JSONL has missing/duplicate case_id values")
    if any(not value for value in case_uids) or len(set(case_uids)) != len(rows):
        raise ValueError("canonical JSONL has missing/duplicate case_uid values")
    require_unique_safe_stems(case_ids, context="canonical JSONL")
    indices = [int(row.get("input_index", -1)) for row in rows]
    if indices != list(range(expected_cases)):
        raise ValueError("canonical input_index must be ordered and cover 0..N-1")

    for index, row in enumerate(rows):
        case_id = case_ids[index]
        if row.get("mode") != "no_text" or row.get("moss_codecvc_mode") != "no_text":
            raise ValueError(f"canonical row is not no_text: {case_id}")
        if row.get("test_set_id") != test_set_id:
            raise ValueError(
                f"test_set_id mismatch for {case_id}: {row.get('test_set_id')!r}"
            )
        if str(row.get("language") or "") != language:
            raise ValueError(f"language mismatch for {case_id}: {row.get('language')!r}")
        if row.get("cell") != f"batch42_strict_{language}":
            raise ValueError(f"cell mismatch for {case_id}: {row.get('cell')!r}")
        for language_key in ("source_lang", "ref_lang"):
            if row.get(language_key) != language:
                raise ValueError(
                    f"{language_key} mismatch for {case_id}: {row.get(language_key)!r}"
                )
        if str(row.get("text") or "") != "<NO_TEXT>":
            raise ValueError(f"no_text placeholder mismatch for {case_id}")
        target_text = str(row.get("target_text") or "").strip()
        if not target_text:
            raise ValueError(f"missing target text for {case_id}")
        for text_key in ("source_text", "content_ref_text", "reference_text"):
            if str(row.get(text_key) or "").strip() != target_text:
                raise ValueError(
                    f"{text_key} must preserve field 4 target_text for {case_id}"
                )
        expected_mapping = {
            "source_audio": "field_5/source_audio",
            "timbre_ref_audio": "field_3/prompt_audio",
            "target_text": "field_4/target_text; scorer-only in no_text mode",
        }
        if row.get("batch42_field_mapping") != expected_mapping:
            raise ValueError(f"field mapping metadata mismatch for {case_id}")
        source = resolved(Path(str(row.get("source_audio") or "")))
        reference = resolved(Path(str(row.get("timbre_ref_audio") or "")))
        reference_alias = resolved(Path(str(row.get("reference_audio") or "")))
        if reference_alias != reference:
            raise ValueError(
                f"reference_audio must equal field 3 timbre_ref_audio for {case_id}"
            )
        for role, path in (("source", source), ("reference", reference)):
            require_within(path, input_root, role=f"{case_id} {role}")
            if not path.is_file():
                raise FileNotFoundError(f"{case_id}: missing {role} audio: {path}")
        if source == reference:
            raise ValueError(f"{case_id}: source and reference audio are identical")
        expected_uid = stable_case_uid(case_id, source, reference)
        if case_uids[index] != expected_uid:
            raise ValueError(
                f"{case_id}: case_uid mismatch: expected {expected_uid}, got {case_uids[index]}"
            )
        if int(row.get("input_line", -1)) != index + 1:
            raise ValueError(f"{case_id}: input_line must equal input_index + 1")
    return {
        "schema_version": "moss_codecvc.batch42_pathx_canonical_audit.v1",
        "path": str(canonical_path),
        "sha256": digest,
        "rows": len(rows),
        "unique_case_ids": len(set(case_ids)),
        "unique_case_uids": len(set(case_uids)),
        "language": language,
        "test_set_id": test_set_id,
        "input_root": str(input_root),
        "audio_files_checked": len(rows) * 2,
    }


def prepare_command(args: argparse.Namespace) -> int:
    cases = parse_strict_lst(
        args.input,
        input_root=args.input_root,
        language=args.language,
        expected_cases=args.expected_cases,
        expected_sha256=args.expected_sha256,
    )
    rows = [canonical_record(case, test_set_id=args.test_set_id) for case in cases]
    atomic_jsonl(args.output_jsonl, rows)
    summary = {
        "schema_version": "moss_codecvc.batch42_pathx_strict_input.v1",
        "status": "ready",
        "input": str(args.input.expanduser().resolve()),
        "input_sha256": args.expected_sha256,
        "input_root": str(args.input_root.expanduser().resolve()),
        "output_jsonl": str(args.output_jsonl.expanduser().resolve()),
        "output_sha256": sha256_file(args.output_jsonl),
        "language": args.language,
        "test_set_id": args.test_set_id,
        "cases": len(rows),
        "mode_counts": {"no_text": len(rows)},
        "field_mapping": {
            "content_source": "field_5/source_audio",
            "timbre_reference": "field_3/prompt_audio",
            "scorer_reference_text": "field_4/target_text",
        },
    }
    atomic_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def output_is_valid(path: Path, minimum_bytes: int) -> bool:
    if not path.is_file() or path.stat().st_size < minimum_bytes:
        return False
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.frames > 0 and info.samplerate > 0
    except Exception:
        return path.stat().st_size >= max(44, minimum_bytes)


def selected_rows(
    canonical_rows: list[dict[str, Any]],
    *,
    num_shards: int,
    shard_index: int,
    max_cases: int,
) -> list[dict[str, Any]]:
    rows = canonical_rows[: max_cases or None]
    return [
        row
        for index, row in enumerate(rows)
        if index % num_shards == shard_index
    ]


def validate_inference_contract(args: argparse.Namespace) -> None:
    expected = {
        "system_id": SYSTEM_ID,
        "seed": 1234,
        "temperature": 0.7,
        "audio_temperature": 1.1,
        "audio_top_p": 0.7,
        "audio_top_k": 20,
        "audio_repetition_penalty": 1.0,
        "no_text_duration_budget_ratio": 1.0,
        "no_text_max_token_margin": 0,
        "timbre_cfg_scale": 1.0,
        "source_semantic_layer": 9,
        "source_semantic_downsample_stride": 1,
    }
    mismatches = {
        name: {"expected": value, "actual": getattr(args, name)}
        for name, value in expected.items()
        if getattr(args, name) != value
    }
    if mismatches:
        raise ValueError(f"registered Path X inference contract mismatch: {mismatches}")
    if args.num_shards not in {1, 8}:
        raise ValueError("registered Path X allows one-shard smoke or eight-shard full inference only")


def inference_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "engine": "004044_run_seedtts_validation_infer_persistent.py",
        "engine_sha256": sha256_file(resolved(args.engine_script)),
        "mode": "no_text",
        "content_memory": "online WavLM-base-plus layer 9 -> BNF cross-attention",
        "source_semantic_model": str(args.source_semantic_model.expanduser().resolve()),
        "source_semantic_layer": args.source_semantic_layer,
        "source_semantic_downsample_stride": args.source_semantic_downsample_stride,
        "source_semantic_device": "same",
        "source_semantic_dtype": "auto",
        "source_semantic_local_files_only": True,
        "temperature": args.temperature,
        "audio_temperature": args.audio_temperature,
        "audio_top_p": args.audio_top_p,
        "audio_top_k": args.audio_top_k,
        "audio_repetition_penalty": args.audio_repetition_penalty,
        "no_text_duration_budget_ratio": args.no_text_duration_budget_ratio,
        "no_text_max_token_margin": args.no_text_max_token_margin,
        "no_text_soft_duration_budget": False,
        "no_text_soft_min_audio_ratio": 0.5,
        "no_text_soft_extra_token_margin": None,
        "filter_ref_content_leak": False,
        "timbre_cfg_scale": args.timbre_cfg_scale,
        "ref_audio_cfg_scale": 1.0,
        "ref_audio_cfg_implementation": "not present in registered pre-C2 engine; identity scale is implicit",
        "ref_prompt_codec_permutation": False,
        "ref_speaker_prompt_slot": False,
        "timbre_side_only": False,
        "audio_segment_policy": "all",
        "source_semantic_monotonic_bias_strength": 0.0,
        "source_semantic_progress_clock": "decode_step",
        "source_semantic_release_after_progress": False,
        "source_semantic_release_start": 1.0,
        "seed": args.seed,
        "seed_scope": "one deterministic RNG stream per persistent shard",
        "model_path": str(args.model_path.expanduser().resolve()),
        "base_model_path": str(args.base_model_path.expanduser().resolve()),
        "code_root": str(args.code_root.expanduser().resolve()),
        "model_config_sha256": REGISTERED_MODEL_FILES["timbre_memory_config.json"]["sha256"],
        "environment_knobs_cleared": sorted(INFERENCE_ENV_KEYS),
    }


def build_engine_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.python),
        str(resolved(args.engine_script)),
        "--validation-jsonl",
        str(resolved(args.canonical_jsonl)),
        "--model-path",
        str(resolved(args.model_path)),
        "--base-model-path",
        str(resolved(args.base_model_path)),
        "--output-dir",
        str(resolved(args.output_dir)),
        "--manifest-jsonl",
        str(resolved(args.raw_manifest)),
        "--mode",
        "no_text",
        "--num-shards",
        str(args.num_shards),
        "--shard-index",
        str(args.shard_index),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--speaker-encoder-type",
        "embedding_loader",
        "--source-semantic-model-name-or-path",
        str(resolved(args.source_semantic_model)),
        "--source-semantic-cache-dir",
        str(resolved(args.source_semantic_cache)),
        "--source-semantic-local-files-only",
        "--source-semantic-layer",
        str(args.source_semantic_layer),
        "--source-semantic-device",
        "same",
        "--source-semantic-dtype",
        "auto",
        "--source-semantic-downsample-stride",
        str(args.source_semantic_downsample_stride),
        "--source-semantic-monotonic-bias-strength",
        "0.0",
        "--source-semantic-progress-clock",
        "decode_step",
        "--source-semantic-release-start",
        "1.0",
        "--temperature",
        str(args.temperature),
        "--audio-temperature",
        str(args.audio_temperature),
        "--audio-top-p",
        str(args.audio_top_p),
        "--audio-top-k",
        str(args.audio_top_k),
        "--audio-repetition-penalty",
        str(args.audio_repetition_penalty),
        "--no-text-audio-temperature",
        str(args.audio_temperature),
        "--no-text-audio-top-p",
        str(args.audio_top_p),
        "--no-text-audio-top-k",
        str(args.audio_top_k),
        "--no-text-audio-repetition-penalty",
        str(args.audio_repetition_penalty),
        "--no-text-duration-budget-ratio",
        str(args.no_text_duration_budget_ratio),
        "--no-text-max-token-margin",
        str(args.no_text_max_token_margin),
        "--audio-segment-policy",
        "all",
        "--timbre-cfg-scale",
        str(args.timbre_cfg_scale),
        "--no-filter-v2-real-no-text-ref-content-leak",
        "--no-ref-prompt-codec-permutation",
        "--no-ref-speaker-prompt-slot",
        "--ref-speaker-prompt-attention-capture-frames",
        "0",
        "--no-timbre-side-only",
    ]
    if args.max_cases:
        command.extend(("--max-cases", str(args.max_cases)))
    if not args.resume:
        command.append("--overwrite")
    if args.engine_dry_run:
        command.append("--dry-run")
    return command


def sanitized_engine_environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    for key in INFERENCE_ENV_KEYS:
        env.pop(key, None)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env.update(
        {
            "PYTHONPATH": str(resolved(args.code_root))
            + (os.pathsep + existing_pythonpath if existing_pythonpath else ""),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": str(args.omp_num_threads),
            # The frozen 004044 predates ref-audio CFG.  Keeping the inherited
            # variable absent is the explicit identity-scale setting.
            "NO_TEXT_SOFT_DURATION_BUDGET": "0",
            "DISABLE_MODE_TOKEN": "0",
            "DISABLE_SOURCE_SEMANTIC_MEMORY": "0",
            "SOURCE_SEMANTIC_RELEASE_AFTER_PROGRESS": "0",
        }
    )
    return env


def convert_raw_manifest(
    args: argparse.Namespace,
    canonical_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    expected = selected_rows(
        canonical_rows,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        max_cases=args.max_cases,
    )
    expected_by_id = {str(row["case_id"]): row for row in expected}
    raw_by_id: dict[str, dict[str, Any]] = {}
    raw_duplicate_rows = 0
    for row in raw_rows:
        case_id = str(row.get("case_id") or "")
        if not case_id:
            raise ValueError("raw manifest has a row with missing case_id")
        if case_id in raw_by_id:
            raw_duplicate_rows += 1
        # Last-row-wins is deliberate.  It makes conversion robust to a
        # previously appended 004044 ledger while the run path still rebuilds
        # the raw manifest before every invocation.
        raw_by_id[case_id] = row
    unexpected = sorted(set(raw_by_id) - set(expected_by_id))
    if unexpected:
        raise ValueError(f"raw manifest contains unexpected case ids: {unexpected[:10]}")

    records: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    config = inference_config(args)
    for canonical in expected:
        case_id = str(canonical["case_id"])
        raw = raw_by_id.get(case_id)
        generated = Path(
            str((raw or {}).get("output_wav") or args.output_dir / f"{safe_stem(case_id)}.wav")
        ).expanduser().resolve(strict=False)
        raw_status = str((raw or {}).get("status") or "missing")
        if raw_status in {"ok", "skipped_exists"} and output_is_valid(
            generated, args.min_output_bytes
        ):
            status = "ok" if raw_status == "ok" else "skipped_existing"
        else:
            status = "error"
        counts[status] = counts.get(status, 0) + 1
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "baseline_vc_inference",
            "system_id": args.system_id,
            "test_set_id": args.test_set_id,
            "case_id": case_id,
            "case_uid": canonical["case_uid"],
            "input_index": canonical["input_index"],
            "input_line": canonical["input_line"],
            "language": canonical["language"],
            "source_audio": canonical["source_audio"],
            "reference_audio": canonical["reference_audio"],
            "generated_audio": str(generated),
            "target_text": canonical["target_text"],
            "reference_text": canonical["reference_text"],
            "prompt_text": canonical["prompt_text"],
            "status": status,
            "metadata": {
                "mode": "no_text",
                "moss_codecvc_mode": "no_text",
                "field_mapping": canonical["batch42_field_mapping"],
            },
            "provenance": {
                "input": str(args.canonical_jsonl.expanduser().resolve()),
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "seed": args.seed,
                "inference_config": config,
                "raw_manifest": str(args.raw_manifest.expanduser().resolve()),
                "raw_status": raw_status,
            },
        }
        if raw:
            record["runtime_seconds"] = raw.get("elapsed_sec")
            record["backend_details"] = {
                key: raw[key]
                for key in (
                    "generation_max_new_tokens",
                    "generation_min_new_tokens",
                    "generation_min_audio_tokens",
                    "generation_stop_head_budget",
                    "generation_structure",
                    "progress_stop_infer_stats",
                    "ref_prompt_codec_permutation",
                    "ref_prompt_codec_permutation_applied",
                )
                if key in raw
            }
            if raw.get("error"):
                record["error"] = {
                    "type": "PathXInferenceError",
                    "message": str(raw["error"]),
                }
        else:
            record["error"] = {
                "type": "MissingRawManifestRow",
                "message": "004044 did not emit a row for this selected case",
            }
        records.append(record)
    return records, counts, {
        "raw_rows": len(raw_rows),
        "raw_unique_case_ids": len(raw_by_id),
        "raw_duplicate_rows": raw_duplicate_rows,
        "expected_rows": len(expected),
    }


def run_command(args: argparse.Namespace) -> int:
    validate_inference_contract(args)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    if args.max_cases < 0:
        raise ValueError("--max-cases must be >= 0")
    if args.max_cases and args.num_shards != 1:
        raise ValueError("--max-cases is restricted to single-shard smoke runs")
    identity = validate_registered_assets(
        args, verify_large_hashes=bool(args.verify_large_checkpoint_hashes)
    )
    for path in (args.canonical_jsonl, args.input_root):
        if not path.expanduser().exists():
            raise FileNotFoundError(path)

    canonical_rows = read_jsonl(args.canonical_jsonl)
    if not canonical_rows:
        raise ValueError("canonical JSONL is empty")
    canonical_audit = audit_canonical_rows(
        canonical_rows,
        canonical_path=args.canonical_jsonl,
        expected_cases=args.expected_cases,
        expected_sha256=args.expected_canonical_sha256,
        test_set_id=args.test_set_id,
        language=args.language,
        input_root=args.input_root,
    )
    expected = selected_rows(
        canonical_rows,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        max_cases=args.max_cases,
    )
    if not expected:
        raise ValueError("selected shard is empty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.raw_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    # Always rebuild the append-only 004044 ledger.  In resume mode the WAVs
    # are reused, so 004044 emits one fresh skipped_exists row per case rather
    # than appending duplicates to an old ledger.
    if args.raw_manifest.exists():
        args.raw_manifest.unlink()
    command = build_engine_command(args)
    env = sanitized_engine_environment(args)
    started = time.monotonic()
    completed = subprocess.run(
        command, cwd=resolved(args.code_root), env=env, check=False
    )
    elapsed = time.monotonic() - started

    if args.engine_dry_run:
        summary = {
            "schema_version": "moss_codecvc.batch42_pathx_strict_shard.v1",
            "status": "engine_dry_run_complete" if completed.returncode == 0 else "engine_dry_run_failed",
            "command": command,
            "engine_returncode": completed.returncode,
            "selected_cases": len(expected),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "inference_config": inference_config(args),
            "registered_identity": identity,
            "canonical_audit": canonical_audit,
        }
        atomic_json(args.summary_json, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return completed.returncode

    raw_rows = read_jsonl(args.raw_manifest) if args.raw_manifest.is_file() else []
    records, counts, raw_audit = convert_raw_manifest(args, canonical_rows, raw_rows)
    atomic_jsonl(args.manifest_jsonl, records)
    all_ok = len(records) == len(expected) and set(counts) <= {"ok", "skipped_existing"}
    summary = {
        "schema_version": "moss_codecvc.batch42_pathx_strict_shard.v1",
        "status": "complete" if all_ok and completed.returncode == 0 else "failed",
        "system_id": args.system_id,
        "test_set_id": args.test_set_id,
        "language": args.language,
        "canonical_jsonl": str(args.canonical_jsonl.expanduser().resolve()),
        "raw_manifest": str(args.raw_manifest.expanduser().resolve()),
        "manifest_jsonl": str(args.manifest_jsonl.expanduser().resolve()),
        "engine_returncode": completed.returncode,
        "runtime_seconds": elapsed,
        "selected_cases": len(expected),
        "manifest_rows": len(records),
        "status_counts": counts,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "inference_config": inference_config(args),
        "registered_identity": identity,
        "canonical_audit": canonical_audit,
        "raw_manifest_audit": raw_audit,
    }
    atomic_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all_ok and completed.returncode == 0 else 1


def common_prepare_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--language", choices=("en", "zh"), required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--test-set-id", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Audit a strict LST and write canonical no-text JSONL.")
    common_prepare_parser(prepare)
    prepare.set_defaults(func=prepare_command)

    run = subparsers.add_parser("run", help="Run one Path X shard and emit the Batch-42 baseline schema.")
    run.add_argument("--python", type=Path, default=Path(sys.executable))
    run.add_argument("--engine-script", type=Path, default=REGISTERED_ENGINE_SCRIPT)
    run.add_argument("--code-root", type=Path, default=REGISTERED_CODE_ROOT)
    run.add_argument("--model-path", type=Path, default=REGISTERED_MODEL_PATH)
    run.add_argument("--base-model-path", type=Path, default=REGISTERED_BASE_MODEL_PATH)
    run.add_argument("--canonical-jsonl", type=Path, required=True)
    run.add_argument("--expected-canonical-sha256", required=True)
    run.add_argument("--expected-cases", type=int, required=True)
    run.add_argument("--input-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--raw-manifest", type=Path, required=True)
    run.add_argument("--manifest-jsonl", type=Path, required=True)
    run.add_argument("--summary-json", type=Path, required=True)
    run.add_argument("--system-id", default=SYSTEM_ID)
    run.add_argument("--test-set-id", required=True)
    run.add_argument("--language", choices=("en", "zh"), required=True)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--max-cases", type=int, default=0)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--seed", type=int, default=1234)
    run.add_argument("--temperature", type=float, default=0.7)
    run.add_argument("--audio-temperature", type=float, default=1.1)
    run.add_argument("--audio-top-p", type=float, default=0.7)
    run.add_argument("--audio-top-k", type=int, default=20)
    run.add_argument("--audio-repetition-penalty", type=float, default=1.0)
    run.add_argument("--no-text-duration-budget-ratio", type=float, default=1.0)
    run.add_argument("--no-text-max-token-margin", type=int, default=0)
    run.add_argument("--timbre-cfg-scale", type=float, default=1.0)
    run.add_argument(
        "--source-semantic-model", type=Path, default=REGISTERED_SOURCE_SEMANTIC_MODEL
    )
    run.add_argument(
        "--source-semantic-cache", type=Path, default=REGISTERED_SOURCE_SEMANTIC_CACHE
    )
    run.add_argument("--source-semantic-layer", type=int, default=9)
    run.add_argument("--source-semantic-downsample-stride", type=int, default=1)
    run.add_argument("--min-output-bytes", type=int, default=1024)
    run.add_argument("--omp-num-threads", type=int, default=8)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--engine-dry-run", action="store_true")
    run.add_argument("--verify-large-checkpoint-hashes", action="store_true")
    run.set_defaults(func=run_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

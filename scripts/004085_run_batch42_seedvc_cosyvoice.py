#!/usr/bin/env python3
"""Run Batch-42 Seed-VC v2.0 and CosyVoice2 direct-VC baselines.

The input is either a Seed-TTS five-column VC manifest::

    id|prompt_text|prompt_audio|target_text|source_audio

or canonical JSONL with explicit source/reference audio fields.  For ``.lst``
input the role mapping is intentionally strict: field 3 is the target-speaker
reference and field 5 is the source/content waveform.  JSONL ``target_audio``
is never guessed as source audio because that name is ambiguous in historical
MOSS-CodecVC artifacts.

The runner is resumable and shard-safe.  Every shard writes deterministic WAV
names, an atomic manifest JSONL, a runtime audit, and a summary JSON.  Heavy
model imports are lazy, so parsing, dry-runs, and unit tests do not require a
GPU or downloaded checkpoints.

The real backends deliberately enforce the registered Batch-42 versions:

* Seed-VC v2.0 code plus ``v2/ar_base.pth`` and ``v2/cfm_small.pth``.  The
  upstream CLI default is timbre-only CFM (``convert_style=False``); use the
  explicit ``--seed-convert-style`` sensitivity to also run the AR
  style/emotion/accent stage.
* CosyVoice tag ``v2.0`` and ``CosyVoice2.inference_vc`` (direct audio VC).

Do not disable the code-revision check for paper-facing runs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_ROOT = Path(
    "/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/batch42"
)
DEFAULT_REPOS = {
    "seed_vc_v2": DOWNLOAD_ROOT / "repos/seed-vc",
    "cosyvoice2_vc": DOWNLOAD_ROOT / "repos/CosyVoice-v2.0",
}
DEFAULT_MODELS = {
    "seed_vc_v2": DOWNLOAD_ROOT / "models/seed-vc-v2",
    "cosyvoice2_vc": DOWNLOAD_ROOT / "models/cosyvoice2-0.5b",
}
EXPECTED_CODE_REVISIONS = {
    "seed_vc_v2": "51383efd921027683c89e5348211d93ff12ac2a8",
    "cosyvoice2_vc": "8555549e882236e6541748b1042d95693caa82ba",
}
REGISTERED_MODEL_REVISIONS = {
    "seed_vc_v2": "257283f9f41585055e8f858fba4fd044e5caed6e",
    "cosyvoice2_vc": "eec1ae6c79877dbd9379285cf8789c9e0879293d",
}
SCHEMA_VERSION = "moss_codecvc.baseline_vc_infer.v1"
SYSTEMS = tuple(DEFAULT_REPOS)


@dataclass(frozen=True)
class VCCase:
    input_index: int
    input_line: int
    case_id: str
    case_uid: str
    source_audio: Path
    reference_audio: Path
    target_text: str = ""
    prompt_text: str = ""
    language: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    field_mapping: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class InputIssue:
    input_index: int
    input_line: int
    case_id: str
    message: str
    raw_excerpt: str


def first_value(row: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def first_named_value(
    row: dict[str, Any], keys: Sequence[str]
) -> tuple[str, Any]:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return key, value
    return "", None


def normalize_language(value: Any, text: str) -> str:
    language = str(value or "").strip().lower().replace("_", "-")
    if language.startswith(("zh", "cmn", "zho", "chi")) or language in {
        "cn",
        "chinese",
        "mandarin",
    }:
        return "zh"
    if language.startswith(("en", "eng")) or language == "english":
        return "en"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return language or "unknown"


def safe_component(
    value: str, *, fallback: str = "case", max_length: int = 80
) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^0-9A-Za-z._-]+", "-", text).strip("-._")
    return (text or fallback)[:max_length]


def resolve_audio_path(value: Any, root: Path) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


def stable_case_uid(
    case_id: str, source_audio: Path, reference_audio: Path
) -> str:
    payload = "\0".join(
        (
            str(case_id),
            str(source_audio.resolve(strict=False)),
            str(reference_audio.resolve(strict=False)),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def deterministic_wav_name(case: VCCase) -> str:
    return f"{safe_component(case.case_id)}__{case.case_uid[:12]}.wav"


def metadata_subset(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "mode",
        "cell",
        "split",
        "source_id",
        "ref_id",
        "source_lang",
        "ref_lang",
        "source_gender",
        "ref_gender",
    )
    return {key: row[key] for key in keys if row.get(key) not in (None, "")}


def parse_lst(
    path: Path, input_root: Path
) -> tuple[list[VCCase], list[InputIssue]]:
    cases: list[VCCase] = []
    issues: list[InputIssue] = []
    with path.open("r", encoding="utf-8") as handle:
        for input_index, raw in enumerate(handle):
            line_number = input_index + 1
            raw = raw.rstrip("\r\n")
            if not raw.strip():
                continue
            fields = raw.split("|")
            case_id = fields[0].strip() if fields else f"line-{line_number}"
            if len(fields) != 5:
                issues.append(
                    InputIssue(
                        input_index=input_index,
                        input_line=line_number,
                        case_id=case_id or f"line-{line_number}",
                        message=(
                            "Seed-TTS VC .lst must have exactly 5 fields: "
                            "id|prompt_text|prompt_audio|target_text|source_audio; "
                            f"got {len(fields)}. Field 5 is required as source/content audio."
                        ),
                        raw_excerpt=raw[:500],
                    )
                )
                continue
            case_id, prompt_text, prompt_audio, target_text, source_audio = (
                item.strip() for item in fields
            )
            source_path = resolve_audio_path(source_audio, input_root)
            reference_path = resolve_audio_path(prompt_audio, input_root)
            uid = stable_case_uid(case_id, source_path, reference_path)
            cases.append(
                VCCase(
                    input_index=input_index,
                    input_line=line_number,
                    case_id=case_id or f"line-{line_number}",
                    case_uid=uid,
                    source_audio=source_path,
                    reference_audio=reference_path,
                    target_text=target_text,
                    prompt_text=prompt_text,
                    language=normalize_language("", target_text),
                    metadata={"input_format": "seedtts_vc_lst"},
                    field_mapping={
                        "source_audio": "field_5/source_audio",
                        "reference_audio": "field_3/prompt_audio",
                        "target_text": "field_4/target_text",
                        "prompt_text": "field_2/prompt_text",
                    },
                )
            )
    return cases, issues


def parse_jsonl(
    path: Path, input_root: Path
) -> tuple[list[VCCase], list[InputIssue]]:
    cases: list[VCCase] = []
    issues: list[InputIssue] = []
    with path.open("r", encoding="utf-8") as handle:
        for input_index, raw in enumerate(handle):
            line_number = input_index + 1
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise TypeError("JSONL record must be an object")
            except Exception as exc:
                issues.append(
                    InputIssue(
                        input_index=input_index,
                        input_line=line_number,
                        case_id=f"line-{line_number}",
                        message=f"invalid JSONL record: {type(exc).__name__}: {exc}",
                        raw_excerpt=raw[:500],
                    )
                )
                continue

            case_id = str(first_value(row, ("case_id", "id")) or f"line-{line_number}")
            audio = row.get("audio") if isinstance(row.get("audio"), dict) else {}

            source_field, source_value = first_named_value(row, ("source_audio",))
            if source_value in (None, "") and audio.get("source") not in (None, ""):
                source_field, source_value = "audio.source", audio["source"]

            reference_field, reference_value = first_named_value(
                row, ("reference_audio", "timbre_ref_audio", "prompt_audio")
            )
            if reference_value in (None, "") and audio.get("reference") not in (
                None,
                "",
            ):
                reference_field, reference_value = "audio.reference", audio["reference"]

            text_field, target_text = first_named_value(
                row,
                ("target_text", "reference_text", "text", "content_ref_text"),
            )
            prompt_field, prompt_text = first_named_value(
                row, ("prompt_text", "timbre_ref_text", "reference_prompt_text")
            )

            missing: list[str] = []
            if source_value in (None, ""):
                missing.append("source_audio (or canonical audio.source)")
            if reference_value in (None, ""):
                missing.append(
                    "reference_audio/timbre_ref_audio/prompt_audio "
                    "(or canonical audio.reference)"
                )
            if missing:
                message = "missing required field(s): " + ", ".join(missing)
                if row.get("target_audio") not in (None, "") and source_value in (
                    None,
                    "",
                ):
                    message += "; target_audio is intentionally not guessed as source_audio"
                issues.append(
                    InputIssue(
                        input_index=input_index,
                        input_line=line_number,
                        case_id=case_id,
                        message=message,
                        raw_excerpt=raw[:500],
                    )
                )
                continue

            source_path = resolve_audio_path(source_value, input_root)
            reference_path = resolve_audio_path(reference_value, input_root)
            uid = stable_case_uid(case_id, source_path, reference_path)
            target_text_str = str(target_text or "").strip()
            cases.append(
                VCCase(
                    input_index=input_index,
                    input_line=line_number,
                    case_id=case_id,
                    case_uid=uid,
                    source_audio=source_path,
                    reference_audio=reference_path,
                    target_text=target_text_str,
                    prompt_text=str(prompt_text or "").strip(),
                    language=normalize_language(
                        first_value(row, ("language", "target_lang", "source_lang")),
                        target_text_str,
                    ),
                    metadata={"input_format": "canonical_jsonl", **metadata_subset(row)},
                    field_mapping={
                        "source_audio": source_field,
                        "reference_audio": reference_field,
                        "target_text": text_field or "missing",
                        "prompt_text": prompt_field or "missing",
                    },
                )
            )
    return cases, issues


def read_input(
    path: Path, *, input_format: str = "auto", input_root: Path | None = None
) -> tuple[list[VCCase], list[InputIssue]]:
    path = path.expanduser().resolve()
    root = (input_root or path.parent).expanduser().resolve()
    fmt = input_format
    if fmt == "auto":
        fmt = "jsonl" if path.suffix.lower() in {".jsonl", ".json"} else "lst"
    if fmt == "lst":
        return parse_lst(path, root)
    if fmt == "jsonl":
        return parse_jsonl(path, root)
    raise ValueError(f"unsupported input format: {fmt}")


def shard_selected(input_index: int, num_shards: int, shard_index: int) -> bool:
    return input_index % num_shards == shard_index


def shard_suffix(num_shards: int, shard_index: int) -> str:
    return "" if num_shards == 1 else f".shard-{shard_index:05d}-of-{num_shards:05d}"


def file_state(path: Path, *, min_bytes: int = 1) -> dict[str, Any]:
    state: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        state.update({"size": 0, "ready": False, "reason": "missing"})
        return state
    size = path.stat().st_size
    state["size"] = size
    try:
        with path.open("rb") as handle:
            prefix = handle.read(200)
    except OSError as exc:
        state.update({"ready": False, "reason": f"unreadable: {exc}"})
        return state
    if b"version https://git-lfs.github.com/spec" in prefix:
        state.update({"ready": False, "reason": "git_lfs_pointer"})
    elif size < min_bytes:
        state.update({"ready": False, "reason": f"too_small_lt_{min_bytes}"})
    else:
        state.update({"ready": True, "reason": "ok"})
    return state


def directory_state(path: Path, *, required_child: str = "") -> dict[str, Any]:
    child = path / required_child if required_child else path
    return {
        "path": str(path),
        "required_child": required_child,
        "exists": path.is_dir(),
        "ready": path.is_dir() and child.exists(),
        "reason": "ok" if path.is_dir() and child.exists() else "missing",
    }


def module_state(module_name: str) -> dict[str, Any]:
    try:
        found = importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        return {"module": module_name, "available": False, "error": str(exc)}
    return {"module": module_name, "available": found}


def git_revision(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def resolve_asset_paths(args: argparse.Namespace) -> dict[str, Path]:
    repo_root = (args.repo_root or DEFAULT_REPOS[args.system]).expanduser().resolve()
    model_root = (args.model_root or DEFAULT_MODELS[args.system]).expanduser().resolve()
    paths: dict[str, Path] = {"repo_root": repo_root, "model_root": model_root}
    if args.system == "seed_vc_v2":
        paths.update(
            {
                "config": (args.seed_config or repo_root / "configs/v2/vc_wrapper.yaml")
                .expanduser()
                .resolve(),
                "ar_checkpoint": (
                    args.seed_ar_checkpoint or model_root / "v2/ar_base.pth"
                )
                .expanduser()
                .resolve(),
                "cfm_checkpoint": (
                    args.seed_cfm_checkpoint or model_root / "v2/cfm_small.pth"
                )
                .expanduser()
                .resolve(),
                "runtime_cache": (
                    args.seed_runtime_cache or model_root / "runtime_cache"
                )
                .expanduser()
                .resolve(),
                "hf_home": (args.hf_home or DOWNLOAD_ROOT / "models/huggingface")
                .expanduser()
                .resolve(),
            }
        )
    else:
        paths.update(
            {
                "model_config": (model_root / "cosyvoice2.yaml").resolve(),
                "campplus": (model_root / "campplus.onnx").resolve(),
                "speech_tokenizer": (model_root / "speech_tokenizer_v2.onnx").resolve(),
                "speaker_info": (model_root / "spk2info.pt").resolve(),
                "llm": (model_root / "llm.pt").resolve(),
                "flow": (model_root / "flow.pt").resolve(),
                "hift": (model_root / "hift.pt").resolve(),
                "blank_en": (model_root / "CosyVoice-BlankEN").resolve(),
                "blank_en_model": (
                    model_root / "CosyVoice-BlankEN/model.safetensors"
                ).resolve(),
                "wetext_root": (
                    args.cosy_wetext_root
                    or DOWNLOAD_ROOT
                    / "models/modelscope_runtime/hub/pengzhendong/wetext"
                )
                .expanduser()
                .resolve(),
                "matcha": (repo_root / "third_party/Matcha-TTS").resolve(),
            }
        )
    return paths


def runtime_audit(args: argparse.Namespace) -> dict[str, Any]:
    paths = resolve_asset_paths(args)
    expected_revision = EXPECTED_CODE_REVISIONS[args.system]
    actual_revision = git_revision(paths["repo_root"])
    revision_match = actual_revision == expected_revision

    if args.system == "seed_vc_v2":
        dependencies = (
            "torch",
            "torchaudio",
            "numpy",
            "librosa",
            "soundfile",
            "yaml",
            "hydra",
            "omegaconf",
            "transformers",
            "huggingface_hub",
            "pydub",
            "einops",
            "scipy",
        )
        files = {
            "repo_v2_wrapper": file_state(
                paths["repo_root"] / "modules/v2/vc_wrapper.py", min_bytes=1000
            ),
            "repo_v2_config": file_state(paths["config"], min_bytes=1000),
            "repo_v2_cli": file_state(
                paths["repo_root"] / "inference_v2.py", min_bytes=1000
            ),
            "ar_base": file_state(paths["ar_checkpoint"], min_bytes=1_000_000),
            "cfm_small": file_state(paths["cfm_checkpoint"], min_bytes=1_000_000),
        }
        optional_files: dict[str, dict[str, Any]] = {}
        directories: dict[str, dict[str, Any]] = {}
        notes = [
            "Seed-VC v2 initialization also resolves ASTRAL bsq32/bsq2048, "
            "CAMPPlus, HuBERT/Whisper tokenizer assets, and BigVGAN. They must "
            "already exist in cache when --offline is used."
        ]
    else:
        dependencies = (
            "torch",
            "torchaudio",
            "numpy",
            "yaml",
            "hyperpyyaml",
            "modelscope",
            "onnxruntime",
            "whisper",
            "tiktoken",
            "inflect",
            "wetext",
            "conformer",
            "diffusers",
            "einops",
            "lightning",
            "pyarrow",
            "pyworld",
            "gdown",
            "wget",
        )
        files = {
            "repo_api": file_state(
                paths["repo_root"] / "cosyvoice/cli/cosyvoice.py", min_bytes=1000
            ),
            "repo_frontend": file_state(
                paths["repo_root"] / "cosyvoice/cli/frontend.py", min_bytes=1000
            ),
            "model_config": file_state(paths["model_config"], min_bytes=1000),
            "campplus": file_state(paths["campplus"], min_bytes=1_000_000),
            "speech_tokenizer_v2": file_state(
                paths["speech_tokenizer"], min_bytes=1_000_000
            ),
            "llm": file_state(paths["llm"], min_bytes=1_000_000),
            "flow": file_state(paths["flow"], min_bytes=1_000_000),
            "hift": file_state(paths["hift"], min_bytes=1_000_000),
            "blank_en_model": file_state(
                paths["blank_en_model"], min_bytes=1_000_000
            ),
        }
        optional_files = {
            "spk2info": file_state(paths["speaker_info"], min_bytes=100),
        }
        directories = {
            "matcha_submodule": directory_state(paths["matcha"], required_child="matcha"),
            "cosyvoice_blank_en": directory_state(
                paths["blank_en"], required_child="config.json"
            ),
            "wetext_fst": directory_state(
                paths["wetext_root"], required_child="zh/tn/tagger.fst"
            ),
        }
        notes = [
            "CosyVoice2 direct VC uses CosyVoice2.inference_vc inherited from "
            "CosyVoice at tag v2.0; it is not zero-shot TTS.",
            "Upstream tag-v2.0 selects the first visible CUDA device internally. "
            "Use CUDA_VISIBLE_DEVICES per process; --device cuda:0 then refers to "
            "that process-local first GPU.",
            "speech_tokenizer_v2.onnx uses the explicitly registered ONNX provider "
            f"policy {args.cosy_speech_tokenizer_provider!r}; the paper-facing "
            "default is CUDAExecutionProvider. CPU is retained only as an explicit "
            "post-failure fallback; the PyTorch LLM/flow/HiFT main model also "
            "remains on CUDA.",
            "The registered CosyVoice2-0.5B revision does not ship spk2info.pt. "
            "It is optional for direct inference_vc; the frontend initializes an "
            "empty registered-speaker map when the file is absent.",
        ]

    modules = {name: module_state(name) for name in dependencies}
    files_ready = all(item["ready"] for item in files.values())
    directories_ready = all(item["ready"] for item in directories.values())
    dependencies_ready = all(item["available"] for item in modules.values())
    revision_ready = revision_match or not args.enforce_code_revision
    ready = files_ready and directories_ready and dependencies_ready and revision_ready
    return {
        "schema_version": SCHEMA_VERSION,
        "system": args.system,
        "ready": ready,
        "expected_code_revision": expected_revision,
        "repo_revision": actual_revision,
        "revision_match": revision_match,
        "enforce_code_revision": args.enforce_code_revision,
        "registered_model_revision": REGISTERED_MODEL_REVISIONS[args.system],
        "inference_config": registered_inference_config(args, paths),
        "paths": {key: str(value) for key, value in paths.items()},
        "files": files,
        "optional_files": optional_files,
        "directories": directories,
        "dependencies": modules,
        "ffmpeg": shutil.which("ffmpeg"),
        "runtime_environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "pythonpath": os.environ.get("PYTHONPATH", ""),
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "hf_home": os.environ.get("HF_HOME", ""),
            "modelscope_cache": os.environ.get("MODELSCOPE_CACHE", ""),
        },
        "notes": notes,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def configure_seed_cudnn(torch_module: Any, *, disable: bool) -> bool | None:
    """Apply the registered Seed-VC cuDNN policy and report the runtime state.

    The Batch-42 H200 image currently exposes a CUDA/cuDNN combination for
    which Seed-VC's HuBERT Conv1d path can fail with
    ``CUDNN_STATUS_NOT_INITIALIZED``.  Disabling cuDNN keeps the model on CUDA
    while routing those convolutions through PyTorch's native CUDA kernels.
    """

    backends = getattr(torch_module, "backends", None)
    cudnn = getattr(backends, "cudnn", None)
    if cudnn is None:
        return None
    if disable:
        cudnn.enabled = False
    return bool(cudnn.enabled)


def case_seed(base_seed: int, case_uid: str) -> int:
    return (base_seed + int(case_uid[:8], 16)) % (2**31 - 1)


def output_is_valid(path: Path, min_output_bytes: int) -> bool:
    if not path.is_file() or path.stat().st_size < min_output_bytes:
        return False
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.frames > 0 and info.samplerate > 0
    except Exception:
        return path.stat().st_size >= max(44, min_output_bytes)


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


def load_prior_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            uid = str(row.get("case_uid") or "")
            if uid:
                records[uid] = row
    return records


@contextmanager
def pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def ensure_expected_revision(args: argparse.Namespace, repo_root: Path) -> str:
    revision = git_revision(repo_root)
    expected = EXPECTED_CODE_REVISIONS[args.system]
    if args.enforce_code_revision and revision != expected:
        raise RuntimeError(
            f"{args.system} code revision mismatch: expected {expected}, got {revision}. "
            "Use the registered checkout; do not bypass this for paper-facing runs."
        )
    return revision


class SeedVCV2Backend:
    """Official Seed-VC V2 wrapper with explicit AR and CFM checkpoints."""

    def __init__(self, args: argparse.Namespace, paths: dict[str, Path], output_dir: Path):
        del output_dir
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        if args.disable_hf_xet:
            os.environ["HF_HUB_DISABLE_XET"] = "1"
        if args.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        hf_home = paths["hf_home"]
        hf_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))

        revision = ensure_expected_revision(args, paths["repo_root"])
        repo = str(paths["repo_root"])
        if repo not in sys.path:
            sys.path.insert(0, repo)

        import soundfile as sf
        import torch
        import yaml
        from hydra.utils import instantiate
        from omegaconf import DictConfig

        cudnn_enabled = configure_seed_cudnn(
            torch, disable=args.seed_disable_cudnn
        )
        self.sf = sf
        self.torch = torch
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {args.device}")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.args = args
        self.paths = paths
        self.revision = revision
        self.cudnn_enabled = cudnn_enabled

        with paths["config"].open("r", encoding="utf-8") as handle:
            config = DictConfig(yaml.safe_load(handle))
        with pushd(paths["runtime_cache"]):
            model = instantiate(config)
            model.load_checkpoints(
                ar_checkpoint_path=str(paths["ar_checkpoint"]),
                cfm_checkpoint_path=str(paths["cfm_checkpoint"]),
            )
        model.to(self.device)
        model.eval()
        model.setup_ar_caches(
            max_batch_size=1,
            max_seq_len=args.seed_ar_max_seq_len,
            dtype=self.dtype,
            device=self.device,
        )
        if args.seed_compile_ar:
            model.compile_ar()
        self.model = model

    def convert(self, case: VCCase, output_path: Path) -> dict[str, Any]:
        generator = self.model.convert_voice_with_streaming(
            source_audio_path=str(case.source_audio),
            target_audio_path=str(case.reference_audio),
            diffusion_steps=self.args.seed_diffusion_steps,
            length_adjust=self.args.seed_length_adjust,
            intelligebility_cfg_rate=self.args.seed_intelligibility_cfg_rate,
            similarity_cfg_rate=self.args.seed_similarity_cfg_rate,
            top_p=self.args.seed_top_p,
            temperature=self.args.seed_temperature,
            repetition_penalty=self.args.seed_repetition_penalty,
            convert_style=self.args.seed_convert_style,
            anonymization_only=False,
            device=self.device,
            dtype=self.dtype,
            stream_output=True,
        )
        full_audio = None
        for _stream_chunk, candidate in generator:
            if candidate is not None:
                full_audio = candidate
        if full_audio is None:
            raise RuntimeError("Seed-VC v2 generator produced no full_audio result")
        sample_rate, waveform = full_audio
        temporary = output_path.with_name(
            f".{output_path.name}.partial-{os.getpid()}.wav"
        )
        temporary.unlink(missing_ok=True)
        try:
            self.sf.write(str(temporary), waveform, int(sample_rate))
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
        return {
            "backend": "Seed-VC v2 VoiceConversionWrapper.convert_voice_with_streaming",
            "architecture_path": "AR+CFM" if self.args.seed_convert_style else "CFM timbre-only",
            "code_revision": self.revision,
            "model_revision": REGISTERED_MODEL_REVISIONS["seed_vc_v2"],
            "ar_checkpoint": str(self.paths["ar_checkpoint"]),
            "cfm_checkpoint": str(self.paths["cfm_checkpoint"]),
            "convert_style": self.args.seed_convert_style,
            "diffusion_steps": self.args.seed_diffusion_steps,
            "length_adjust": self.args.seed_length_adjust,
            "intelligibility_cfg_rate": self.args.seed_intelligibility_cfg_rate,
            "similarity_cfg_rate": self.args.seed_similarity_cfg_rate,
            "top_p": self.args.seed_top_p,
            "temperature": self.args.seed_temperature,
            "repetition_penalty": self.args.seed_repetition_penalty,
            "disable_hf_xet": self.args.disable_hf_xet,
            "disable_cudnn": self.args.seed_disable_cudnn,
            "cudnn_enabled": self.cudnn_enabled,
            "sample_rate": int(sample_rate),
        }


def cosy_onnx_inference_session_factory(
    original_factory: Callable[..., Any],
    *,
    speech_tokenizer_path: Path,
    provider_mode: str,
    events: list[dict[str, Any]],
) -> Callable[..., Any]:
    """Override only speech-tokenizer ONNX providers; leave all other sessions intact."""
    target = speech_tokenizer_path.expanduser().resolve(strict=False)

    def create_session(model_path: Any, *args: Any, **kwargs: Any) -> Any:
        current = Path(str(model_path)).expanduser().resolve(strict=False)
        if current != target:
            return original_factory(model_path, *args, **kwargs)

        if provider_mode == "cpu":
            requested = ["CPUExecutionProvider"]
        else:
            requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        configured = dict(kwargs)
        configured["providers"] = requested
        try:
            session = original_factory(model_path, *args, **configured)
        except Exception as exc:
            if provider_mode != "auto":
                raise
            events.append(
                {
                    "event": "speech_tokenizer_cuda_provider_init_failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "fallback": "CPUExecutionProvider",
                }
            )
            configured["providers"] = ["CPUExecutionProvider"]
            session = original_factory(model_path, *args, **configured)
        events.append(
            {
                "event": "speech_tokenizer_session_created",
                "provider_mode": provider_mode,
                "providers": list(session.get_providers()),
            }
        )
        return session

    return create_session


class CosyVoice2VCBackend:
    """CosyVoice tag-v2.0 direct audio-to-audio VC backend."""

    def __init__(self, args: argparse.Namespace, paths: dict[str, Path], output_dir: Path):
        del output_dir
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        revision = ensure_expected_revision(args, paths["repo_root"])
        repo = str(paths["repo_root"])
        matcha = str(paths["matcha"])
        for entry in (repo, matcha):
            if entry not in sys.path:
                sys.path.insert(0, entry)

        import torch
        import torchaudio
        import onnxruntime
        from cosyvoice.cli.cosyvoice import CosyVoice2
        from cosyvoice.utils.file_utils import load_wav
        import wetext.wetext as wetext_impl

        self.torch = torch
        self.torchaudio = torchaudio
        self.load_wav = load_wav
        self.args = args
        self.paths = paths
        self.revision = revision
        self.device = torch.device(args.device)
        self.onnx_available_providers = list(onnxruntime.get_available_providers())
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {args.device}")
        if self.device.type == "cuda" and self.device.index not in (None, 0):
            raise RuntimeError(
                "CosyVoice tag v2.0 ignores nonzero torch device indices and uses "
                "the first visible GPU. Set CUDA_VISIBLE_DEVICES to the desired "
                "physical GPU and pass --device cuda:0."
            )
        if self.device.type == "cpu" and torch.cuda.is_available():
            raise RuntimeError(
                "CosyVoice tag v2.0 auto-selects CUDA when it is visible. Hide GPUs "
                "with CUDA_VISIBLE_DEVICES='' for a real CPU run."
            )
        if args.cosy_speech_tokenizer_provider in {"cuda", "auto"} and (
            "CUDAExecutionProvider" not in self.onnx_available_providers
        ):
            raise RuntimeError(
                "CosyVoice speech tokenizer requested CUDAExecutionProvider, but "
                "onnxruntime exposes only "
                f"{self.onnx_available_providers}. Check Cosy ORT capi and the "
                "vc-benchmark pip NVIDIA library prefix."
            )

        wetext_root = paths["wetext_root"]
        if wetext_root.is_dir():
            def local_wetext_snapshot(repo_id: str, *unused_args: Any, **unused_kwargs: Any) -> str:
                if repo_id != "pengzhendong/wetext":
                    raise ValueError(f"unexpected ModelScope repo requested: {repo_id}")
                return str(wetext_root)

            wetext_impl.snapshot_download = local_wetext_snapshot

        provider_events: list[dict[str, Any]] = []
        original_inference_session = onnxruntime.InferenceSession
        onnxruntime.InferenceSession = cosy_onnx_inference_session_factory(
            original_inference_session,
            speech_tokenizer_path=paths["speech_tokenizer"],
            provider_mode=args.cosy_speech_tokenizer_provider,
            events=provider_events,
        )
        try:
            self.model = CosyVoice2(
                str(paths["model_root"]),
                load_jit=False,
                load_trt=False,
                load_vllm=False,
                fp16=args.cosy_fp16,
            )
        finally:
            onnxruntime.InferenceSession = original_inference_session

        self.speech_tokenizer_providers = list(
            self.model.frontend.speech_tokenizer_session.get_providers()
        )
        self.campplus_providers = list(
            self.model.frontend.campplus_session.get_providers()
        )
        self.onnx_provider_events = provider_events
        self.main_model_device = str(self.model.model.device)
        if args.cosy_speech_tokenizer_provider == "cpu" and (
            not self.speech_tokenizer_providers
            or self.speech_tokenizer_providers[0] != "CPUExecutionProvider"
        ):
            raise RuntimeError(
                "CosyVoice speech tokenizer requested CPUExecutionProvider but got "
                f"{self.speech_tokenizer_providers}"
            )
        if args.cosy_speech_tokenizer_provider == "cuda" and (
            not self.speech_tokenizer_providers
            or self.speech_tokenizer_providers[0] != "CUDAExecutionProvider"
        ):
            raise RuntimeError(
                "CosyVoice speech tokenizer requested CUDAExecutionProvider but got "
                f"{self.speech_tokenizer_providers}"
            )
        if self.device.type == "cuda" and self.main_model_device != "cuda":
            raise RuntimeError(
                "CosyVoice main model must remain on CUDA when --device cuda:0; "
                f"got {self.main_model_device}"
            )

    def convert(self, case: VCCase, output_path: Path) -> dict[str, Any]:
        source_speech_16k = self.load_wav(str(case.source_audio), 16000)
        prompt_speech_16k = self.load_wav(str(case.reference_audio), 16000)
        chunks = []
        for result in self.model.inference_vc(
            source_speech_16k,
            prompt_speech_16k,
            stream=False,
            speed=self.args.cosy_speed,
        ):
            speech = result.get("tts_speech")
            if speech is None:
                raise RuntimeError("CosyVoice2 inference_vc result lacks tts_speech")
            chunks.append(speech.detach().cpu())
        if not chunks:
            raise RuntimeError("CosyVoice2 inference_vc produced no audio chunks")
        waveform = self.torch.cat(chunks, dim=1)
        temporary = output_path.with_name(
            f".{output_path.name}.partial-{os.getpid()}.wav"
        )
        temporary.unlink(missing_ok=True)
        try:
            self.torchaudio.save(
                str(temporary), waveform, int(self.model.sample_rate)
            )
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
        return {
            "backend": "CosyVoice2.inference_vc",
            "task": "direct_audio_voice_conversion",
            "code_revision": self.revision,
            "model_revision": REGISTERED_MODEL_REVISIONS["cosyvoice2_vc"],
            "speed": self.args.cosy_speed,
            "fp16": self.args.cosy_fp16,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "wetext_root": str(self.paths["wetext_root"]),
            "speech_tokenizer_provider_requested": (
                self.args.cosy_speech_tokenizer_provider
            ),
            "speech_tokenizer_providers_actual": self.speech_tokenizer_providers,
            "onnx_available_providers": self.onnx_available_providers,
            "campplus_providers_actual": self.campplus_providers,
            "onnx_provider_events": self.onnx_provider_events,
            "main_model_device": self.main_model_device,
            "sample_rate": int(self.model.sample_rate),
        }


def default_backend_factory(
    args: argparse.Namespace, paths: dict[str, Path], output_dir: Path
) -> Any:
    if args.system == "seed_vc_v2":
        return SeedVCV2Backend(args, paths, output_dir)
    return CosyVoice2VCBackend(args, paths, output_dir)


def resolve_outputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    test_set = safe_component(args.test_set_id, fallback="test-set")
    output_dir = (
        args.output_dir
        or ROOT / "testset/outputs" / f"baseline_{args.system}_{test_set}"
    ).expanduser().resolve()
    suffix = shard_suffix(args.num_shards, args.shard_index)
    manifest = (
        args.manifest_jsonl or output_dir / f"manifest{suffix}.jsonl"
    ).expanduser().resolve()
    summary = (
        args.summary_json or output_dir / f"summary{suffix}.json"
    ).expanduser().resolve()
    return output_dir, manifest, summary


def registered_inference_config(
    args: argparse.Namespace, paths: dict[str, Path]
) -> dict[str, Any]:
    common = {
        "expected_code_revision": EXPECTED_CODE_REVISIONS[args.system],
        "registered_model_revision": REGISTERED_MODEL_REVISIONS[args.system],
        "repo_root": str(paths["repo_root"]),
        "model_root": str(paths["model_root"]),
    }
    if args.system == "seed_vc_v2":
        return {
            **common,
            "api": "VoiceConversionWrapper.convert_voice_with_streaming",
            "ar_checkpoint": str(paths["ar_checkpoint"]),
            "cfm_checkpoint": str(paths["cfm_checkpoint"]),
            "convert_style": args.seed_convert_style,
            "diffusion_steps": args.seed_diffusion_steps,
            "length_adjust": args.seed_length_adjust,
            "intelligibility_cfg_rate": args.seed_intelligibility_cfg_rate,
            "similarity_cfg_rate": args.seed_similarity_cfg_rate,
            "top_p": args.seed_top_p,
            "temperature": args.seed_temperature,
            "repetition_penalty": args.seed_repetition_penalty,
            "disable_hf_xet": args.disable_hf_xet,
            "disable_cudnn": args.seed_disable_cudnn,
        }
    return {
        **common,
        "api": "CosyVoice2.inference_vc",
        "direct_vc": True,
        "speed": args.cosy_speed,
        "fp16": args.cosy_fp16,
        "speech_tokenizer_onnx_provider": args.cosy_speech_tokenizer_provider,
        "speech_tokenizer_cpu_fallback": (
            args.cosy_speech_tokenizer_provider == "auto"
        ),
        "main_model_device": args.device,
        "wetext_root": str(paths["wetext_root"]),
        "load_jit": False,
        "load_trt": False,
        "load_vllm": False,
    }


def base_record(
    args: argparse.Namespace,
    case: VCCase,
    output_path: Path,
    paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "baseline_vc_inference",
        "system_id": args.system,
        "test_set_id": args.test_set_id,
        "case_id": case.case_id,
        "case_uid": case.case_uid,
        "input_index": case.input_index,
        "input_line": case.input_line,
        "language": case.language,
        "source_audio": str(case.source_audio),
        "reference_audio": str(case.reference_audio),
        "generated_audio": str(output_path),
        "target_text": case.target_text,
        "reference_text": case.target_text,
        "prompt_text": case.prompt_text,
        "metadata": case.metadata,
        "provenance": {
            "input": str(args.input.expanduser().resolve()),
            "field_mapping": case.field_mapping,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "seed": case_seed(args.seed, case.case_uid),
            "inference_config": registered_inference_config(args, paths),
        },
    }


def issue_record(args: argparse.Namespace, issue: InputIssue) -> dict[str, Any]:
    uid = hashlib.sha256(
        f"input-error\0{issue.input_index}\0{issue.case_id}".encode("utf-8")
    ).hexdigest()[:20]
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "baseline_vc_inference",
        "system_id": args.system,
        "test_set_id": args.test_set_id,
        "case_id": issue.case_id,
        "case_uid": uid,
        "input_index": issue.input_index,
        "input_line": issue.input_line,
        "status": "input_error",
        "error": {
            "type": "InputError",
            "message": issue.message,
            "raw_excerpt": issue.raw_excerpt,
        },
        "provenance": {
            "input": str(args.input.expanduser().resolve()),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        },
    }


def ordered_records(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records.values(),
        key=lambda row: (
            int(row.get("input_index", 10**18)),
            str(row.get("case_uid", "")),
        ),
    )


def manifest_status_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def execute(
    args: argparse.Namespace,
    *,
    backend_factory: Callable[[argparse.Namespace, dict[str, Path], Path], Any]
    | None = None,
) -> int:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < num_shards")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")

    output_dir, manifest_path, summary_path = resolve_outputs(args)
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    cases, issues = read_input(
        args.input, input_format=args.input_format, input_root=args.input_root
    )
    cases = [
        case
        for case in cases
        if shard_selected(case.input_index, args.num_shards, args.shard_index)
    ]
    issues = [
        issue
        for issue in issues
        if shard_selected(issue.input_index, args.num_shards, args.shard_index)
    ]
    if args.limit is not None:
        cases = cases[: args.limit]

    paths = resolve_asset_paths(args)
    audit = runtime_audit(args)
    suffix = shard_suffix(args.num_shards, args.shard_index)
    audit_path = output_dir / f"runtime_audit{suffix}.json"
    atomic_json(audit_path, audit)

    records = load_prior_manifest(manifest_path) if args.resume else {}
    for issue in issues:
        record = issue_record(args, issue)
        records[record["case_uid"]] = record
    atomic_jsonl(manifest_path, ordered_records(records))

    if issues and not args.continue_on_error:
        rows = ordered_records(records)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "system_id": args.system,
            "test_set_id": args.test_set_id,
            "status": "stopped_on_input_error",
            "manifest_jsonl": str(manifest_path),
            "runtime_audit": str(audit_path),
            "selected_valid_cases": len(cases),
            "selected_input_errors": len(issues),
            "manifest_status_counts": manifest_status_counts(rows),
        }
        atomic_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1

    def needs_backend(case: VCCase) -> bool:
        output_path = wav_dir / deterministic_wav_name(case)
        prior = records.get(case.case_uid)
        if args.resume and prior and prior.get("status") in {"ok", "skipped_existing"}:
            prior_output = Path(str(prior.get("generated_audio") or output_path))
            if output_is_valid(prior_output, args.min_output_bytes):
                return False
        if args.skip_existing and output_is_valid(output_path, args.min_output_bytes):
            return False
        if not case.source_audio.is_file() or not case.reference_audio.is_file():
            return False
        return True

    runtime_needed = any(needs_backend(case) for case in cases)
    if (
        not args.dry_run
        and backend_factory is None
        and runtime_needed
        and not audit["ready"]
    ):
        for case in cases:
            output_path = wav_dir / deterministic_wav_name(case)
            prior = records.get(case.case_uid)
            if args.resume and prior and prior.get("status") in {"ok", "skipped_existing"}:
                prior_output = Path(str(prior.get("generated_audio") or output_path))
                if output_is_valid(prior_output, args.min_output_bytes):
                    continue
            record = base_record(args, case, output_path, paths)
            record.update(
                {
                    "status": "blocked_runtime_not_ready",
                    "runtime_audit": str(audit_path),
                }
            )
            records[case.case_uid] = record
        rows = ordered_records(records)
        atomic_jsonl(manifest_path, rows)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "system_id": args.system,
            "test_set_id": args.test_set_id,
            "status": "blocked_runtime_not_ready",
            "input": str(args.input.expanduser().resolve()),
            "output_dir": str(output_dir),
            "manifest_jsonl": str(manifest_path),
            "runtime_audit": str(audit_path),
            "selected_valid_cases": len(cases),
            "selected_input_errors": len(issues),
            "manifest_status_counts": manifest_status_counts(rows),
        }
        atomic_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2

    backend = None
    counters: dict[str, int] = {}
    stopped_on_error = False

    def count(status: str) -> None:
        counters[status] = counters.get(status, 0) + 1

    for position, case in enumerate(cases, start=1):
        output_path = wav_dir / deterministic_wav_name(case)
        prior = records.get(case.case_uid)
        if args.resume and prior and prior.get("status") in {"ok", "skipped_existing"}:
            prior_output = Path(str(prior.get("generated_audio") or output_path))
            if output_is_valid(prior_output, args.min_output_bytes):
                prior["resume_action"] = "kept_prior_success"
                records[case.case_uid] = prior
                atomic_jsonl(manifest_path, ordered_records(records))
                count("resumed")
                print(
                    f"[{position}/{len(cases)}] resume {case.case_id}: {prior_output}",
                    flush=True,
                )
                continue

        record = base_record(args, case, output_path, paths)
        missing_audio = []
        if not case.source_audio.is_file():
            missing_audio.append(f"source_audio={case.source_audio}")
        if not case.reference_audio.is_file():
            missing_audio.append(f"reference_audio={case.reference_audio}")
        if missing_audio:
            record.update(
                {
                    "status": "input_error",
                    "error": {
                        "type": "FileNotFoundError",
                        "message": "missing input audio: " + "; ".join(missing_audio),
                    },
                }
            )
            count("input_error")
            if not args.continue_on_error:
                stopped_on_error = True
        elif args.dry_run:
            record.update(
                {
                    "status": "dry_run",
                    "runtime_ready": bool(audit["ready"]),
                    "output_exists": output_is_valid(output_path, args.min_output_bytes),
                }
            )
            count("dry_run")
        elif args.skip_existing and output_is_valid(output_path, args.min_output_bytes):
            record.update(
                {
                    "status": "skipped_existing",
                    "output_bytes": output_path.stat().st_size,
                }
            )
            count("skipped_existing")
        else:
            started = time.monotonic()
            try:
                if backend is None:
                    factory = backend_factory or default_backend_factory
                    backend = factory(args, paths, output_dir)
                seed_everything(case_seed(args.seed, case.case_uid))
                details = backend.convert(case, output_path)
                if not output_is_valid(output_path, args.min_output_bytes):
                    raise RuntimeError(
                        f"backend returned without a valid WAV >= {args.min_output_bytes} bytes"
                    )
                record.update(
                    {
                        "status": "ok",
                        "runtime_seconds": time.monotonic() - started,
                        "output_bytes": output_path.stat().st_size,
                        "backend_details": details,
                    }
                )
                count("ok")
            except Exception as exc:
                record.update(
                    {
                        "status": "error",
                        "runtime_seconds": time.monotonic() - started,
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(limit=20),
                        },
                    }
                )
                count("error")
                print(
                    f"[{position}/{len(cases)}] ERROR {case.case_id}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if not args.continue_on_error:
                    stopped_on_error = True
            else:
                print(
                    f"[{position}/{len(cases)}] ok {case.case_id}: {output_path}",
                    flush=True,
                )

        records[case.case_uid] = record
        atomic_jsonl(manifest_path, ordered_records(records))
        if stopped_on_error:
            break

    rows = ordered_records(records)
    counts = manifest_status_counts(rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "system_id": args.system,
        "test_set_id": args.test_set_id,
        "status": (
            "stopped_on_error"
            if stopped_on_error
            else "dry_run_complete"
            if args.dry_run
            else "complete"
        ),
        "input": str(args.input.expanduser().resolve()),
        "output_dir": str(output_dir),
        "manifest_jsonl": str(manifest_path),
        "runtime_audit": str(audit_path),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "limit": args.limit,
        "continue_on_error": args.continue_on_error,
        "selected_valid_cases": len(cases),
        "selected_input_errors": len(issues),
        "run_action_counts": counters,
        "manifest_status_counts": counts,
        "runtime_ready": bool(audit["ready"]),
        "inference_config": registered_inference_config(args, paths),
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if stopped_on_error:
        return 1
    if args.fail_if_any_error and any(
        status in counts for status in ("error", "input_error")
    ):
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-format", choices=("auto", "lst", "jsonl"), default="auto")
    parser.add_argument(
        "--input-root",
        type=Path,
        help="Root for relative audio paths; defaults to the input manifest directory.",
    )
    parser.add_argument("--test-set-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest-jsonl", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--limit",
        "--max-cases",
        dest="limit",
        type=int,
        help="Limit valid cases after modulo sharding (aliases: --limit/--max-cases).",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record a bad case and continue; use --no-continue-on-error to stop immediately.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-if-any-error", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-output-bytes", type=int, default=1024)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument(
        "--enforce-code-revision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the registered Batch-42 code commit (recommended and default).",
    )

    parser.add_argument("--seed-config", type=Path)
    parser.add_argument("--seed-ar-checkpoint", type=Path)
    parser.add_argument("--seed-cfm-checkpoint", type=Path)
    parser.add_argument("--seed-runtime-cache", type=Path)
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--disable-hf-xet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use regular HTTP downloads for Seed-VC auxiliary HF assets.",
    )
    parser.add_argument("--seed-diffusion-steps", type=int, default=30)
    parser.add_argument("--seed-length-adjust", type=float, default=1.0)
    parser.add_argument("--seed-intelligibility-cfg-rate", type=float, default=0.7)
    parser.add_argument("--seed-similarity-cfg-rate", type=float, default=0.7)
    parser.add_argument("--seed-top-p", type=float, default=0.9)
    parser.add_argument("--seed-temperature", type=float, default=1.0)
    parser.add_argument("--seed-repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--seed-convert-style",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Upstream default false runs timbre-only CFM; enable explicitly to "
            "run the AR style/emotion/accent stage before CFM."
        ),
    )
    parser.add_argument("--seed-ar-max-seq-len", type=int, default=4096)
    parser.add_argument("--seed-compile-ar", action="store_true")
    parser.add_argument(
        "--seed-disable-cudnn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Disable cuDNN only for Seed-VC inference while keeping CUDA enabled; "
            "use this H200 compatibility fallback for HuBERT Conv1d initialization."
        ),
    )

    parser.add_argument("--cosy-speed", type=float, default=1.0)
    parser.add_argument("--cosy-wetext-root", type=Path)
    parser.add_argument(
        "--cosy-fp16", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--cosy-speech-tokenizer-provider",
        choices=("cpu", "auto", "cuda"),
        default="cuda",
        help=(
            "ONNX EP for speech_tokenizer_v2.onnx only. cuda is the registered "
            "paper-facing path; auto tries CUDA then explicitly falls back to CPU, "
            "and cpu is a post-failure fallback. The CosyVoice LLM/flow/HiFT main "
            "model remains on the requested torch GPU."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return execute(args)
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

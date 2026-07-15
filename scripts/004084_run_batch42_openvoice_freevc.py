#!/usr/bin/env python3
"""Run the Batch-42 OpenVoice V2 and FreeVC v1 VC baselines.

The runner intentionally keeps heavy model imports lazy.  Input can be either
the official Seed-TTS five-column VC manifest

    id|prompt_text|prompt_audio|target_text|source_audio

or a canonical JSONL.  For the official manifest the roles are deliberately
strict: column 5 is the content/source waveform and column 3 is the target
speaker reference.  ``target_audio`` is never guessed as a source in JSONL,
because that name is ambiguous in existing evaluation artifacts.

Each shard writes deterministic WAV names, a resumable manifest JSONL, and a
summary JSON under ``testset/outputs/baseline_<system>_<test-set>/`` by
default.  Per-case failures are recorded and do not stop later cases.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import logging
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_ROOT = Path(
    "/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/batch42"
)
DEFAULT_REPOS = {
    "openvoice_v2": DOWNLOAD_ROOT / "repos/OpenVoice",
    "freevc_v1": DOWNLOAD_ROOT / "repos/FreeVC",
}
DEFAULT_MODELS = {
    "openvoice_v2": DOWNLOAD_ROOT / "models/openvoice-v2",
    "freevc_v1": DOWNLOAD_ROOT / "models/freevc-v1",
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


def first_named_value(row: dict[str, Any], keys: Sequence[str]) -> tuple[str, Any]:
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


def safe_component(value: str, *, fallback: str = "case", max_length: int = 80) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^0-9A-Za-z._-]+", "-", text).strip("-._")
    return (text or fallback)[:max_length]


def resolve_audio_path(value: Any, root: Path) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


def stable_case_uid(case_id: str, source_audio: Path, reference_audio: Path) -> str:
    payload = "\0".join(
        (str(case_id), str(source_audio.resolve(strict=False)), str(reference_audio.resolve(strict=False)))
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


def parse_lst(path: Path, input_root: Path) -> tuple[list[VCCase], list[InputIssue]]:
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
                            f"got {len(fields)}. Column 5 is required as source audio."
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
                    },
                )
            )
    return cases, issues


def parse_jsonl(path: Path, input_root: Path) -> tuple[list[VCCase], list[InputIssue]]:
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
            if reference_value in (None, "") and audio.get("reference") not in (None, ""):
                reference_field, reference_value = "audio.reference", audio["reference"]
            text_field, target_text = first_named_value(
                row, ("target_text", "reference_text", "text", "content_ref_text")
            )
            prompt_field, prompt_text = first_named_value(
                row, ("prompt_text", "timbre_ref_text", "reference_prompt_text")
            )
            missing = []
            if source_value in (None, ""):
                missing.append("source_audio (or canonical audio.source)")
            if reference_value in (None, ""):
                missing.append(
                    "reference_audio/timbre_ref_audio/prompt_audio (or canonical audio.reference)"
                )
            if missing:
                issues.append(
                    InputIssue(
                        input_index=input_index,
                        input_line=line_number,
                        case_id=case_id,
                        message="missing required field(s): " + ", ".join(missing),
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


def module_state(module_name: str) -> dict[str, Any]:
    try:
        found = importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        return {"module": module_name, "available": False, "error": str(exc)}
    state: dict[str, Any] = {"module": module_name, "available": found}
    if found:
        distribution_candidates = {
            "faster_whisper": ("faster-whisper",),
            "whisper_timestamped": ("whisper-timestamped",),
            "eng_to_ipa": ("eng-to-ipa", "eng_to_ipa"),
            "unidecode": ("Unidecode",),
        }.get(module_name, (module_name, module_name.replace("_", "-")))
        for distribution in distribution_candidates:
            try:
                state["version"] = importlib.metadata.version(distribution)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
    return state


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
    paths = {"repo_root": repo_root, "model_root": model_root}
    if args.system == "openvoice_v2":
        paths.update(
            {
                "config": (args.openvoice_config or model_root / "converter/config.json")
                .expanduser()
                .resolve(),
                "checkpoint": (
                    args.openvoice_checkpoint or model_root / "converter/checkpoint.pth"
                )
                .expanduser()
                .resolve(),
            }
        )
    else:
        paths.update(
            {
                "config": (args.freevc_config or model_root / "configs/freevc.json")
                .expanduser()
                .resolve(),
                "checkpoint": (
                    args.freevc_checkpoint or model_root / "checkpoints/freevc.pth"
                )
                .expanduser()
                .resolve(),
                "speaker_encoder": (
                    args.freevc_speaker_encoder
                    or model_root / "speaker_encoder/ckpt/pretrained_bak_5805000.pt"
                )
                .expanduser()
                .resolve(),
                "wavlm": (args.freevc_wavlm or model_root / "wavlm/WavLM-Large.pt")
                .expanduser()
                .resolve(),
            }
        )
    return paths


def effective_openvoice_segmentation(args: argparse.Namespace) -> str:
    """Resolve the legacy VAD flag to the registered OpenVoice segmentation."""
    legacy_vad = getattr(args, "openvoice_vad", None)
    if legacy_vad is True:
        return "upstream_silero_vad"
    if legacy_vad is False:
        return "full_audio"
    return str(args.openvoice_segmentation)


def registered_inference_config(
    args: argparse.Namespace, paths: dict[str, Path]
) -> dict[str, Any]:
    common = {
        "repo_root": str(paths["repo_root"]),
        "model_root": str(paths["model_root"]),
        "device": args.device,
    }
    if args.system == "openvoice_v2":
        segmentation = effective_openvoice_segmentation(args)
        torch_home = Path(os.environ.get("TORCH_HOME", "~/.cache/torch")).expanduser()
        upstream_silero = segmentation == "upstream_silero_vad"
        return {
            **common,
            "api": "ToneColorConverter.convert",
            "speaker_embedding_api": (
                "openvoice.se_extractor.get_se"
                if upstream_silero
                else "ToneColorConverter.extract_se"
            ),
            "speaker_embedding_segmentation": segmentation,
            "speaker_embedding_vad_implementation": (
                "whisper_timestamped.get_vad_segments(method=silero)"
                if upstream_silero
                else (
                    "librosa.effects.split"
                    if segmentation == "local_energy_vad"
                    else "disabled"
                )
            ),
            "speaker_embedding_vad_top_db": args.openvoice_vad_top_db,
            "speaker_embedding_max_segment_seconds": (
                args.openvoice_short_audio_split_seconds
            ),
            "upstream_silero_short_audio_retry_split_seconds": (
                args.openvoice_silero_short_retry_split_seconds
            ),
            "upstream_silero_vad": upstream_silero,
            "torch_home": str(torch_home),
            "silero_hub_cache": str(
                torch_home / "hub/snakers4_silero-vad_master"
            ),
            "network_access": "disabled",
            "tau": args.openvoice_tau,
            "watermark_enabled": args.openvoice_enable_watermark,
        }
    if args.system == "freevc_v1":
        return {
            **common,
            "api": "FreeVC SynthesizerTrn.voice_conversion",
            "speaker_encoder": str(paths["speaker_encoder"]),
            "wavlm": str(paths["wavlm"]),
        }
    # 004087 deliberately reuses this execution loop with Vevo-specific
    # asset/runtime hooks. Its backend details and runtime audit carry the full
    # registered Vevo config; keep this shared provenance entry generic.
    return {**common, "api": str(args.system)}


def runtime_audit(args: argparse.Namespace) -> dict[str, Any]:
    paths = resolve_asset_paths(args)
    if args.system == "openvoice_v2":
        dependencies = [
            "torch",
            "numpy",
            "librosa",
            "soundfile",
            "inflect",
            "eng_to_ipa",
            "unidecode",
            "pypinyin",
            "jieba",
            "cn2an",
        ]
        segmentation = effective_openvoice_segmentation(args)
        if segmentation == "upstream_silero_vad":
            dependencies.extend(
                ["faster_whisper", "whisper_timestamped", "onnxruntime"]
            )
        torch_home = Path(os.environ.get("TORCH_HOME", "~/.cache/torch")).expanduser()
        if args.openvoice_enable_watermark:
            dependencies.append("wavmark")
        files = {
            "repo_api": file_state(paths["repo_root"] / "openvoice/api.py", min_bytes=100),
            "repo_mel_processing": file_state(
                paths["repo_root"] / "openvoice/mel_processing.py", min_bytes=100
            ),
            "config": file_state(paths["config"], min_bytes=100),
            "checkpoint": file_state(paths["checkpoint"], min_bytes=1_000_000),
        }
        if segmentation == "upstream_silero_vad":
            files.update(
                {
                    "repo_se_extractor": file_state(
                        paths["repo_root"] / "openvoice/se_extractor.py",
                        min_bytes=100,
                    ),
                    "silero_hubconf": file_state(
                        torch_home
                        / "hub/snakers4_silero-vad_master/hubconf.py",
                        min_bytes=100,
                    ),
                    "silero_local_model": file_state(
                        torch_home
                        / "hub/snakers4_silero-vad_master/src/silero_vad/data/silero_vad.jit",
                        min_bytes=1_000_000,
                    ),
                }
            )
    else:
        dependencies = (
            "torch",
            "torchvision",
            "numpy",
            "librosa",
            "scipy",
            "webrtcvad",
        )
        files = {
            "repo_models": file_state(paths["repo_root"] / "models.py", min_bytes=100),
            "repo_wavlm": file_state(paths["repo_root"] / "wavlm/WavLM.py", min_bytes=100),
            "config": file_state(paths["config"], min_bytes=100),
            "checkpoint": file_state(paths["checkpoint"], min_bytes=1_000_000),
            "speaker_encoder": file_state(paths["speaker_encoder"], min_bytes=1_000_000),
            "wavlm": file_state(paths["wavlm"], min_bytes=1_000_000),
        }
    modules = {name: module_state(name) for name in dependencies}
    ready = all(item["ready"] for item in files.values()) and all(
        item["available"] for item in modules.values()
    )
    return {
        "system": args.system,
        "ready": ready,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "repo_revision": git_revision(paths["repo_root"]),
        "paths": {key: str(value) for key, value in paths.items()},
        "files": files,
        "dependencies": modules,
        "ffmpeg": shutil.which("ffmpeg"),
        "inference_config": registered_inference_config(args, paths),
        "notes": (
            [
                "The paper-facing OpenVoice path uses upstream "
                "openvoice.se_extractor.get_se with Silero VAD.",
                "TORCH_HOME must point at the pre-populated local "
                "snakers4_silero-vad torch.hub cache, so upstream behavior is "
                "preserved without a network download.",
            ]
            if args.system == "openvoice_v2"
            else []
        ),
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


class OpenVoiceBackend:
    def __init__(self, args: argparse.Namespace, paths: dict[str, Path], output_dir: Path):
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        repo = str(paths["repo_root"])
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import librosa
        import soundfile as sf
        import torch
        from openvoice.api import OpenVoiceBaseClass, ToneColorConverter
        from openvoice import se_extractor

        self.librosa = librosa
        self.sf = sf
        self.torch = torch
        self.se_extractor = se_extractor
        self.device = args.device
        if args.openvoice_enable_watermark:
            self.converter = ToneColorConverter(str(paths["config"]), device=args.device)
        else:
            # The pinned upstream constructor reads ``enable_watermark`` from
            # kwargs but forwards the same unknown kwarg to its base class,
            # which raises TypeError.  Initialize the exact converter class
            # through its base and reproduce the two remaining assignments.
            # This avoids an evaluation-only watermark and the wavmark model.
            self.converter = ToneColorConverter.__new__(ToneColorConverter)
            OpenVoiceBaseClass.__init__(
                self.converter, str(paths["config"]), device=args.device
            )
            self.converter.watermark_model = None
            self.converter.version = getattr(self.converter.hps, "_version_", "v1")
        self.converter.load_ckpt(str(paths["checkpoint"]))
        self.tau = args.openvoice_tau
        self.message = args.openvoice_watermark_message
        self.enable_watermark = args.openvoice_enable_watermark
        self.segmentation = effective_openvoice_segmentation(args)
        self.max_segment_seconds = args.openvoice_short_audio_split_seconds
        self.silero_short_retry_split_seconds = (
            args.openvoice_silero_short_retry_split_seconds
        )
        self.vad_top_db = args.openvoice_vad_top_db
        self.energy_vad_fallbacks = 0
        self.offline_segments_written = 0
        self.upstream_silero_calls = 0
        self.upstream_silero_short_audio_retries = 0
        suffix = shard_suffix(args.num_shards, args.shard_index).lstrip(".") or "single"
        self.work_dir = output_dir / ".openvoice_se_work" / suffix
        self.cache_dir = output_dir / ".openvoice_se_cache" / suffix
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, audio_path: Path) -> Path:
        stat = audio_path.stat()
        payload = "\0".join(
            (
                "offline-se-v1",
                str(audio_path.resolve()),
                str(stat.st_size),
                str(stat.st_mtime_ns),
                self.segmentation,
                str(self.vad_top_db),
                str(self.max_segment_seconds),
                str(self.silero_short_retry_split_seconds),
            )
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{digest}.pth"

    def _offline_segments(self, audio_path: Path, cache_path: Path) -> list[Path]:
        """Create deterministic local SE segments without upstream Silero/Whisper."""
        waveform, sample_rate = self.librosa.load(
            str(audio_path), sr=None, mono=True
        )
        if sample_rate is None or int(sample_rate) <= 0 or waveform.size == 0:
            raise RuntimeError(f"OpenVoice speaker reference is empty: {audio_path}")

        if self.segmentation == "local_energy_vad":
            raw_intervals = self.librosa.effects.split(
                waveform,
                top_db=self.vad_top_db,
                frame_length=2048,
                hop_length=512,
            )
            intervals = [
                (max(0, int(start)), min(int(end), int(waveform.size)))
                for start, end in raw_intervals
                if int(end) > int(start)
            ]
            if not intervals:
                intervals = [(0, int(waveform.size))]
                self.energy_vad_fallbacks += 1
        else:
            intervals = [(0, int(waveform.size))]

        max_frames = max(1, int(round(float(sample_rate) * self.max_segment_seconds)))
        wavs_dir = self.work_dir / cache_path.stem / "wavs"
        if wavs_dir.is_dir():
            for stale in wavs_dir.glob("segment-*.wav"):
                stale.unlink()
        wavs_dir.mkdir(parents=True, exist_ok=True)

        segments: list[Path] = []
        for start, end in intervals:
            for chunk_start in range(start, end, max_frames):
                chunk_end = min(end, chunk_start + max_frames)
                if chunk_end <= chunk_start:
                    continue
                segment_path = wavs_dir / f"segment-{len(segments):05d}.wav"
                self.sf.write(
                    str(segment_path),
                    waveform[chunk_start:chunk_end],
                    int(sample_rate),
                    subtype="PCM_16",
                )
                segments.append(segment_path)
        if not segments:
            raise RuntimeError(
                f"offline OpenVoice segmentation produced no audio: {audio_path}"
            )
        self.offline_segments_written += len(segments)
        return segments

    def _speaker_embedding(self, audio_path: Path):
        cache_path = self._cache_path(audio_path)
        if cache_path.is_file():
            return self.torch.load(cache_path, map_location=self.device)
        if self.segmentation == "upstream_silero_vad":
            try:
                self.upstream_silero_calls += 1
                embedding, _audio_name = self.se_extractor.get_se(
                    str(audio_path),
                    self.converter,
                    target_dir=str(self.work_dir),
                    vad=True,
                )
            except AssertionError as exc:
                if "input audio is too short" not in str(exc):
                    raise
                # Upstream computes round(active_duration / 10) and rejects many
                # ordinary 2--5 s Seed-TTS prompts. Keep upstream get_se and
                # Silero VAD; only retry its split helper with the registered
                # shorter split interval.
                original_split = self.se_extractor.split_audio_vad

                def short_audio_split(
                    retry_audio_path: str,
                    audio_name: str,
                    target_dir: str,
                    split_seconds: float = 10.0,
                ):
                    del split_seconds
                    return original_split(
                        retry_audio_path,
                        audio_name,
                        target_dir,
                        split_seconds=self.silero_short_retry_split_seconds,
                    )

                self.se_extractor.split_audio_vad = short_audio_split
                try:
                    self.upstream_silero_calls += 1
                    embedding, _audio_name = self.se_extractor.get_se(
                        str(audio_path),
                        self.converter,
                        target_dir=str(self.work_dir),
                        vad=True,
                    )
                    self.upstream_silero_short_audio_retries += 1
                finally:
                    self.se_extractor.split_audio_vad = original_split
        else:
            segments = self._offline_segments(audio_path, cache_path)
            embedding = self.converter.extract_se([str(path) for path in segments])
        temporary = cache_path.with_name(f".{cache_path.name}.tmp-{os.getpid()}")
        self.torch.save(embedding.detach().cpu(), temporary)
        os.replace(temporary, cache_path)
        return embedding.to(self.device)

    def convert(self, case: VCCase, output_path: Path) -> dict[str, Any]:
        source_se = self._speaker_embedding(case.source_audio)
        target_se = self._speaker_embedding(case.reference_audio)
        temporary = output_path.with_name(f".{output_path.name}.partial-{os.getpid()}.wav")
        temporary.unlink(missing_ok=True)
        try:
            self.converter.convert(
                audio_src_path=str(case.source_audio),
                src_se=source_se,
                tgt_se=target_se,
                output_path=str(temporary),
                tau=self.tau,
                message=self.message,
            )
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
        return {
            "backend": "OpenVoice ToneColorConverter",
            "se_extractor": (
                "openvoice.se_extractor.get_se"
                if self.segmentation == "upstream_silero_vad"
                else "ToneColorConverter.extract_se"
            ),
            "se_segmentation": self.segmentation,
            "se_vad_implementation": (
                "whisper_timestamped.get_vad_segments(method=silero)"
                if self.segmentation == "upstream_silero_vad"
                else (
                    "librosa.effects.split"
                    if self.segmentation == "local_energy_vad"
                    else "disabled"
                )
            ),
            "se_vad_top_db": self.vad_top_db,
            "se_max_segment_seconds": self.max_segment_seconds,
            "upstream_silero_vad": self.segmentation == "upstream_silero_vad",
            "upstream_silero_calls": self.upstream_silero_calls,
            "upstream_silero_short_audio_retries": (
                self.upstream_silero_short_audio_retries
            ),
            "upstream_silero_short_audio_retry_split_seconds": (
                self.silero_short_retry_split_seconds
            ),
            "torch_home": os.environ.get("TORCH_HOME", ""),
            "network_access": "disabled",
            "tau": self.tau,
            "watermark_message": self.message,
            "watermark_enabled": self.enable_watermark,
            "energy_vad_fallbacks": self.energy_vad_fallbacks,
            "offline_segments_written": self.offline_segments_written,
        }


class FreeVCBackend:
    def __init__(self, args: argparse.Namespace, paths: dict[str, Path], output_dir: Path):
        del output_dir
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        repo = str(paths["repo_root"])
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import librosa
        import soundfile as sf
        import torch
        import utils as freevc_utils
        from models import SynthesizerTrn
        from speaker_encoder.voice_encoder import SpeakerEncoder
        from wavlm import WavLM, WavLMConfig

        # FreeVC's utils.py configures the process-wide root logger at DEBUG,
        # which otherwise dumps thousands of numba compiler lines on the first
        # librosa call.  Keep model information while suppressing compiler IR.
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("numba").setLevel(logging.WARNING)
        self.librosa = librosa
        self.sf = sf
        self.torch = torch
        self.device = torch.device(args.device)
        self.hps = freevc_utils.get_hparams_from_file(str(paths["config"]))
        self.net_g = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model,
        ).to(self.device)
        checkpoint = torch.load(
            str(paths["checkpoint"]), map_location="cpu", weights_only=False
        )
        self.net_g.load_state_dict(checkpoint["model"], strict=True)
        self.net_g.eval()

        wavlm_checkpoint = torch.load(
            str(paths["wavlm"]), map_location="cpu", weights_only=False
        )
        wavlm_config = WavLMConfig(wavlm_checkpoint["cfg"])
        self.content_model = WavLM(wavlm_config).to(self.device)
        self.content_model.load_state_dict(wavlm_checkpoint["model"], strict=True)
        self.content_model.eval()
        self.speaker_model = SpeakerEncoder(
            str(paths["speaker_encoder"]), device=self.device
        )
        self.speaker_model.eval()
        self._speaker_cache: dict[Path, Any] = {}

    def _speaker_embedding(self, reference_audio: Path):
        key = reference_audio.resolve()
        if key not in self._speaker_cache:
            wav, _ = self.librosa.load(
                str(reference_audio), sr=self.hps.data.sampling_rate
            )
            wav, _ = self.librosa.effects.trim(wav, top_db=20)
            if wav.size == 0:
                raise ValueError(f"empty reference after trim: {reference_audio}")
            embedding = self.speaker_model.embed_utterance(wav)
            self._speaker_cache[key] = self.torch.from_numpy(embedding).unsqueeze(0)
        return self._speaker_cache[key].to(self.device)

    def convert(self, case: VCCase, output_path: Path) -> dict[str, Any]:
        target_embedding = self._speaker_embedding(case.reference_audio)
        wav_source, _ = self.librosa.load(
            str(case.source_audio), sr=self.hps.data.sampling_rate
        )
        if wav_source.size == 0:
            raise ValueError(f"empty source audio: {case.source_audio}")
        source = self.torch.from_numpy(wav_source).float().unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            content = self.content_model.extract_features(source)[0].transpose(1, 2)
            audio = self.net_g.infer(content, g=target_embedding)
            audio = audio[0][0].detach().cpu().float().numpy()
        temporary = output_path.with_name(f".{output_path.name}.partial-{os.getpid()}.wav")
        temporary.unlink(missing_ok=True)
        try:
            self.sf.write(str(temporary), audio, self.hps.data.sampling_rate)
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
        return {
            "backend": "FreeVC v1 original",
            "content_encoder": "WavLM-Large",
            "speaker_encoder": "FreeVC pretrained_bak_5805000.pt",
            "sampling_rate": int(self.hps.data.sampling_rate),
        }


def default_backend_factory(
    args: argparse.Namespace, paths: dict[str, Path], output_dir: Path
):
    if args.system == "openvoice_v2":
        return OpenVoiceBackend(args, paths, output_dir)
    return FreeVCBackend(args, paths, output_dir)


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
        key=lambda row: (int(row.get("input_index", 10**18)), str(row.get("case_uid", ""))),
    )


def execute(
    args: argparse.Namespace,
    *,
    backend_factory: Callable[[argparse.Namespace, dict[str, Path], Path], Any] | None = None,
) -> int:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < num_shards")
    if args.max_cases is not None and args.max_cases < 1:
        raise ValueError("--max-cases must be >= 1")
    if args.system == "openvoice_v2":
        if args.openvoice_short_audio_split_seconds <= 0:
            raise ValueError("--openvoice-short-audio-split-seconds must be positive")
        if args.openvoice_silero_short_retry_split_seconds <= 0:
            raise ValueError(
                "--openvoice-silero-short-retry-split-seconds must be positive"
            )
        if not 0 < args.openvoice_vad_top_db <= 120:
            raise ValueError("--openvoice-vad-top-db must satisfy 0 < value <= 120")

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
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    audit = runtime_audit(args)
    audit_path = output_dir / f"runtime_audit{shard_suffix(args.num_shards, args.shard_index)}.json"
    atomic_json(audit_path, audit)

    records = load_prior_manifest(manifest_path) if args.resume else {}
    for issue in issues:
        record = issue_record(args, issue)
        records[record["case_uid"]] = record
    atomic_jsonl(manifest_path, ordered_records(records))

    backend = None
    paths = resolve_asset_paths(args)
    if not args.dry_run:
        if backend_factory is None and not audit["ready"]:
            summary = {
                "schema_version": SCHEMA_VERSION,
                "system_id": args.system,
                "test_set_id": args.test_set_id,
                "status": "blocked_runtime_not_ready",
                "runtime_audit": str(audit_path),
                "manifest_jsonl": str(manifest_path),
                "selected_cases": len(cases),
                "input_errors": len(issues),
            }
            atomic_json(summary_path, summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 2
        factory = backend_factory or default_backend_factory
        try:
            backend = factory(args, paths, output_dir)
        except Exception as exc:
            summary = {
                "schema_version": SCHEMA_VERSION,
                "system_id": args.system,
                "test_set_id": args.test_set_id,
                "status": "backend_load_error",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "runtime_audit": str(audit_path),
                "manifest_jsonl": str(manifest_path),
            }
            atomic_json(summary_path, summary)
            traceback.print_exc()
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 2

    counters: dict[str, int] = {}

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
            else:
                print(
                    f"[{position}/{len(cases)}] ok {case.case_id}: {output_path}",
                    flush=True,
                )

        records[case.case_uid] = record
        atomic_jsonl(manifest_path, ordered_records(records))
        if record.get("status") in {"error", "input_error"} and not args.continue_on_error:
            break

    manifest_rows = ordered_records(records)
    manifest_counts: dict[str, int] = {}
    for row in manifest_rows:
        status = str(row.get("status") or "unknown")
        manifest_counts[status] = manifest_counts.get(status, 0) + 1
    summary = {
        "schema_version": SCHEMA_VERSION,
        "system_id": args.system,
        "test_set_id": args.test_set_id,
        "status": "dry_run_complete" if args.dry_run else "complete",
        "input": str(args.input.expanduser().resolve()),
        "output_dir": str(output_dir),
        "manifest_jsonl": str(manifest_path),
        "runtime_audit": str(audit_path),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "selected_valid_cases": len(cases),
        "selected_input_errors": len(issues),
        "run_action_counts": counters,
        "manifest_status_counts": manifest_counts,
        "runtime_ready": bool(audit["ready"]),
        "inference_config": registered_inference_config(args, paths),
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.fail_if_any_error and any(
        status in manifest_counts for status in ("error", "input_error")
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
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--continue-on-error", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--fail-if-any-error", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-output-bytes", type=int, default=1024)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--model-root", type=Path)

    parser.add_argument("--openvoice-config", type=Path)
    parser.add_argument("--openvoice-checkpoint", type=Path)
    parser.add_argument("--openvoice-tau", type=float, default=0.3)
    parser.add_argument(
        "--openvoice-segmentation",
        choices=("upstream_silero_vad", "local_energy_vad", "full_audio"),
        default="upstream_silero_vad",
        help=(
            "Speaker-embedding segmentation. upstream_silero_vad is the official "
            "OpenVoice path and requires a pre-populated TORCH_HOME torch.hub cache; "
            "local_energy_vad is an explicit fallback; full_audio disables VAD."
        ),
    )
    parser.add_argument(
        "--openvoice-vad",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Compatibility alias: --openvoice-vad selects upstream_silero_vad and "
            "--no-openvoice-vad selects full_audio. Prefer --openvoice-segmentation."
        ),
    )
    parser.add_argument(
        "--openvoice-vad-top-db",
        type=float,
        default=40.0,
        help="Top-dB threshold for deterministic local_energy_vad segmentation.",
    )
    parser.add_argument(
        "--openvoice-short-audio-split-seconds",
        type=float,
        default=20.0,
        help=(
            "Maximum length of each local speaker-embedding segment. The legacy "
            "option name is retained for compatibility."
        ),
    )
    parser.add_argument(
        "--openvoice-silero-short-retry-split-seconds",
        type=float,
        default=2.0,
        help=(
            "When upstream Silero get_se raises only 'input audio is too short', "
            "retry the same upstream splitter with this interval. This preserves "
            "Silero and does not select the local energy-VAD fallback."
        ),
    )
    parser.add_argument(
        "--openvoice-enable-watermark",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disabled for metric fairness; enabling requires the wavmark package/model.",
    )
    parser.add_argument(
        "--openvoice-watermark-message",
        default="",
        help="Message used only when --openvoice-enable-watermark is set.",
    )

    parser.add_argument("--freevc-config", type=Path)
    parser.add_argument("--freevc-checkpoint", type=Path)
    parser.add_argument("--freevc-speaker-encoder", type=Path)
    parser.add_argument("--freevc-wavlm", type=Path)
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

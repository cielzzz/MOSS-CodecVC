#!/usr/bin/env python3
"""Run the Batch-42 Vevo-Timbre baseline with the official Amphion pipeline.

This runner deliberately imports the common Batch-42 input and execution
machinery from ``004084_run_batch42_openvoice_freevc.py``.  Consequently the
Seed-TTS five-column contract, canonical JSONL contract, deterministic output
names, modulo sharding, resume rules, per-case errors, and manifest schema stay
identical across baseline systems.

The official Seed-TTS VC manifest contract is strict:

    id|prompt_text|prompt_audio|target_text|source_audio

Column 5 is always the content/source waveform and column 3 is always the
target-speaker reference.  A four-column row is an input error; ``target_audio``
is never guessed as source audio in canonical JSONL.

Heavy Amphion imports are lazy.  ``--dry-run`` always emits the selected-case
manifest and a detailed runtime audit even while checkpoints are incomplete.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence


ROOT = Path(__file__).resolve().parents[1]
COMMON_RUNNER_PATH = ROOT / "scripts/004084_run_batch42_openvoice_freevc.py"
DOWNLOAD_ROOT = Path(
    "/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/batch42"
)
DEFAULT_REPO_ROOT = DOWNLOAD_ROOT / "repos/Amphion"
DEFAULT_MODEL_ROOT = DOWNLOAD_ROOT / "models/vevo-timbre"
DEFAULT_TORCH_HOME = DOWNLOAD_ROOT / "models/torch"
DEFAULT_MODEL_REVISION = "7edf4640c400c20542aa39c45b63f60e6c7baba0"
SYSTEM_ID = "vevo_timbre"

EXPECTED_MODEL_FILES = {
    "tokenizer_checkpoint": {
        "size": 177_183_712,
        "sha256": "660bd48b023e637a786a9c78f404cb979ef9a5d1c93ce24837e0bec942352c4d",
    },
    "fmt_checkpoint": {
        "size": 1_350_803_704,
        "sha256": "750f013ac1485855bfbe992ffec8ed5f625e6070b8bc52ce71a6f9ae0229c5c4",
    },
    "vocoder_checkpoint": {
        "size": 1_020_206_416,
        "sha256": "7670d180569fdae986fbf94ede07d6fc4ce8bfcf406cd1aadbe33e08581b5f6a",
    },
    "hubert_checkpoint": {
        "size": 1_261_897_861,
        "sha256": "c95371600b87881db5ce1576cf41c884bfa7e1e81e9032d73215c364d551ea2e",
    },
}


def _load_common_runner():
    module_name = "moss_codecvc_batch42_common_004084"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, COMMON_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Batch-42 common runner from {COMMON_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


COMMON = _load_common_runner()

# Public aliases intentionally point to 004084 rather than copied code.  Tests
# and downstream wrappers can rely on one protocol implementation.
VCCase = COMMON.VCCase
InputIssue = COMMON.InputIssue
read_input = COMMON.read_input
deterministic_wav_name = COMMON.deterministic_wav_name
shard_selected = COMMON.shard_selected
shard_suffix = COMMON.shard_suffix
case_seed = COMMON.case_seed
output_is_valid = COMMON.output_is_valid


def resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def resolve_asset_paths(args: argparse.Namespace) -> dict[str, Path]:
    repo_root = resolved(args.repo_root or DEFAULT_REPO_ROOT)
    model_root = resolved(args.model_root or DEFAULT_MODEL_ROOT)
    torch_home = resolved(args.torch_home or DEFAULT_TORCH_HOME)
    tokenizer_dir = resolved(
        args.tokenizer_checkpoint_dir or model_root / "tokenizer/vq8192"
    )
    fmt_checkpoint_dir = resolved(
        args.fmt_checkpoint_dir or model_root / "acoustic_modeling/Vq8192ToMels"
    )
    vocoder_checkpoint_dir = resolved(
        args.vocoder_checkpoint_dir or model_root / "acoustic_modeling/Vocoder"
    )
    return {
        "repo_root": repo_root,
        "model_root": model_root,
        "torch_home": torch_home,
        "official_entry": repo_root / "models/vc/vevo/infer_vevotimbre.py",
        "vevo_utils": repo_root / "models/vc/vevo/vevo_utils.py",
        "fmt_config": resolved(
            args.fmt_config
            or repo_root / "models/vc/vevo/config/Vq8192ToMels.json"
        ),
        "vocoder_config": resolved(
            args.vocoder_config or repo_root / "models/vc/vevo/config/Vocoder.json"
        ),
        "hubert_stats": resolved(
            args.hubert_stats
            or repo_root / "models/vc/vevo/config/hubert_large_l18_mean_std.npz"
        ),
        "tokenizer_checkpoint_dir": tokenizer_dir,
        "tokenizer_checkpoint": tokenizer_dir / "model.safetensors",
        "fmt_checkpoint_dir": fmt_checkpoint_dir,
        "fmt_checkpoint": fmt_checkpoint_dir / "model.safetensors",
        "vocoder_checkpoint_dir": vocoder_checkpoint_dir,
        "vocoder_checkpoint": vocoder_checkpoint_dir / "model.safetensors",
        "hubert_checkpoint": resolved(
            args.hubert_checkpoint
            or torch_home / "hub/checkpoints/hubert_fairseq_large_ll60k.pth"
        ),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def expected_file_state(
    path: Path,
    *,
    min_bytes: int,
    expected_size: int | None = None,
    expected_sha256: str = "",
    verify_sha256: bool = False,
) -> dict[str, Any]:
    state = COMMON.file_state(path, min_bytes=min_bytes)
    if expected_size is not None:
        state["expected_size"] = expected_size
        state["size_matches"] = state.get("size") == expected_size
        if state.get("ready") and not state["size_matches"]:
            state.update(
                ready=False,
                reason=f"size_mismatch_expected_{expected_size}",
            )
    if expected_sha256:
        state["expected_sha256"] = expected_sha256
        if verify_sha256 and state.get("ready"):
            actual = sha256_file(path)
            state["sha256"] = actual
            state["sha256_matches"] = actual == expected_sha256
            if not state["sha256_matches"]:
                state.update(ready=False, reason="sha256_mismatch")
    return state


def safetensors_header_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"ready": False, "reason": "missing"}
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            metadata = handle.metadata()
        return {
            "ready": bool(keys),
            "reason": "ok" if keys else "no_tensors",
            "tensor_count": len(keys),
            "first_tensor_keys": keys[:10],
            "metadata": metadata or {},
        }
    except Exception as exc:
        return {
            "ready": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def device_state(device: str) -> dict[str, Any]:
    state: dict[str, Any] = {"requested": device, "ready": True}
    try:
        import torch

        state.update(
            torch_version=str(torch.__version__),
            torch_cuda_version=str(torch.version.cuda),
            cuda_available=bool(torch.cuda.is_available()),
            cuda_device_count=int(torch.cuda.device_count()),
        )
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            state.update(ready=False, reason="cuda_requested_but_unavailable")
        elif str(device).startswith("cuda"):
            try:
                index = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
                state["resolved_index"] = index
                state["name"] = torch.cuda.get_device_name(index)
            except Exception as exc:
                state.update(
                    ready=False,
                    reason=f"invalid_cuda_device: {type(exc).__name__}: {exc}",
                )
    except Exception as exc:
        state.update(
            ready=False,
            reason=f"torch_probe_failed: {type(exc).__name__}: {exc}",
        )
    return state


def runtime_audit(args: argparse.Namespace) -> dict[str, Any]:
    paths = resolve_asset_paths(args)
    dependencies = (
        "torch",
        "torchaudio",
        "numpy",
        "librosa",
        "accelerate",
        "safetensors",
        "yaml",
        "IPython",
        "json5",
        "ruamel.yaml",
        "einops",
        "scipy",
        "soundfile",
        "tqdm",
        "transformers",
        "pyworld",
    )
    modules = {name: COMMON.module_state(name) for name in dependencies}
    files = {
        "common_runner": expected_file_state(
            COMMON_RUNNER_PATH, min_bytes=10_000
        ),
        "official_entry": expected_file_state(
            paths["official_entry"], min_bytes=1_000
        ),
        "vevo_utils": expected_file_state(paths["vevo_utils"], min_bytes=10_000),
        "fmt_config": expected_file_state(paths["fmt_config"], min_bytes=1_000),
        "vocoder_config": expected_file_state(
            paths["vocoder_config"], min_bytes=1_000
        ),
        "hubert_stats": expected_file_state(paths["hubert_stats"], min_bytes=1_000),
    }
    for key in (
        "tokenizer_checkpoint",
        "fmt_checkpoint",
        "vocoder_checkpoint",
        "hubert_checkpoint",
    ):
        expected = EXPECTED_MODEL_FILES[key]
        files[key] = expected_file_state(
            paths[key],
            min_bytes=1_000_000,
            expected_size=int(expected["size"]),
            expected_sha256=str(expected["sha256"]),
            verify_sha256=bool(args.verify_checkpoint_sha256),
        )
    for key in (
        "tokenizer_checkpoint",
        "fmt_checkpoint",
        "vocoder_checkpoint",
    ):
        header = safetensors_header_state(paths[key])
        files[key]["safetensors_header"] = header
        if files[key].get("ready") and not header.get("ready"):
            files[key].update(
                ready=False,
                reason=f"invalid_safetensors:{header.get('reason', 'unknown')}",
            )

    incomplete_downloads = []
    cache_root = paths["model_root"] / ".cache/huggingface/download"
    if cache_root.is_dir():
        for item in sorted(cache_root.rglob("*.incomplete")):
            incomplete_downloads.append(
                {"path": str(item), "size": item.stat().st_size}
            )

    device = device_state(args.device)
    blocking_reasons = [
        f"file:{name}:{state.get('reason', 'not_ready')}"
        for name, state in files.items()
        if not state.get("ready")
    ]
    blocking_reasons.extend(
        f"module:{name}:unavailable"
        for name, state in modules.items()
        if not state.get("available")
    )
    if not device.get("ready"):
        blocking_reasons.append(f"device:{device.get('reason', 'not_ready')}")
    return {
        "system": SYSTEM_ID,
        "ready": not blocking_reasons,
        "official_pipeline": "models/vc/vevo/infer_vevotimbre.py::VevoInferencePipeline.inference_fm",
        "model_repo": "amphion/Vevo",
        "model_revision": args.model_revision,
        "repo_revision": COMMON.git_revision(paths["repo_root"]),
        "common_runner": str(COMMON_RUNNER_PATH),
        "paths": {key: str(value) for key, value in paths.items()},
        "files": files,
        "dependencies": modules,
        "device": device,
        "flow_matching_steps": args.flow_matching_steps,
        "target_db": args.target_db,
        "incomplete_downloads": incomplete_downloads,
        "blocking_reasons": blocking_reasons,
    }


@contextlib.contextmanager
def pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class VevoTimbreBackend:
    """Thin adapter around Amphion's official Vevo-Timbre inference path."""

    def __init__(self, args: argparse.Namespace, paths: dict[str, Path], output_dir: Path):
        del output_dir
        os.environ["TORCH_HOME"] = str(paths["torch_home"])
        repo = str(paths["repo_root"])
        if repo not in sys.path:
            sys.path.insert(0, repo)

        import torch

        # Keep the official repo as cwd while loading JSON configs because its
        # Vq8192ToMels config contains a repo-relative HuBERT statistics path.
        with pushd(paths["repo_root"]):
            from models.vc.vevo.vevo_utils import VevoInferencePipeline, save_audio

            self.pipeline = VevoInferencePipeline(
                content_style_tokenizer_ckpt_path=str(
                    paths["tokenizer_checkpoint_dir"]
                ),
                fmt_cfg_path=str(paths["fmt_config"]),
                fmt_ckpt_path=str(paths["fmt_checkpoint_dir"]),
                vocoder_cfg_path=str(paths["vocoder_config"]),
                vocoder_ckpt_path=str(paths["vocoder_checkpoint_dir"]),
                device=torch.device(args.device),
            )
        self.torch = torch
        self.save_audio = save_audio
        self.flow_matching_steps = int(args.flow_matching_steps)
        self.target_db = float(args.target_db)

    def convert(self, case: VCCase, output_path: Path) -> dict[str, Any]:
        temporary = output_path.with_name(
            f".{output_path.name}.partial-{os.getpid()}.wav"
        )
        temporary.unlink(missing_ok=True)
        try:
            with self.torch.inference_mode():
                generated = self.pipeline.inference_fm(
                    src_wav_path=str(case.source_audio),
                    timbre_ref_wav_path=str(case.reference_audio),
                    flow_matching_steps=self.flow_matching_steps,
                )
            self.save_audio(
                generated,
                output_path=str(temporary),
                target_db=self.target_db,
            )
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
        return {
            "backend": "Amphion Vevo-Timbre",
            "official_entry": "models/vc/vevo/infer_vevotimbre.py",
            "pipeline_method": "VevoInferencePipeline.inference_fm",
            "flow_matching_steps": self.flow_matching_steps,
            "target_db": self.target_db,
            "sampling_rate": 24_000,
        }


def default_backend_factory(
    args: argparse.Namespace, paths: dict[str, Path], output_dir: Path
) -> VevoTimbreBackend:
    return VevoTimbreBackend(args, paths, output_dir)


@contextlib.contextmanager
def common_hooks() -> Iterator[None]:
    # 004084 intentionally exposes a generic execute loop but its asset audit
    # and default backend factory are system-specific.  Replacing only these
    # three hooks keeps all protocol-sensitive execution behavior shared.  The
    # restoration is important for unittest discovery or notebooks that load
    # multiple baseline runners in the same Python process.
    original = (
        COMMON.resolve_asset_paths,
        COMMON.runtime_audit,
        COMMON.default_backend_factory,
    )
    try:
        COMMON.resolve_asset_paths = resolve_asset_paths
        COMMON.runtime_audit = runtime_audit
        COMMON.default_backend_factory = default_backend_factory
        yield
    finally:
        (
            COMMON.resolve_asset_paths,
            COMMON.runtime_audit,
            COMMON.default_backend_factory,
        ) = original


def execute(
    args: argparse.Namespace,
    *,
    backend_factory: Callable[[argparse.Namespace, dict[str, Path], Path], Any]
    | None = None,
) -> int:
    args.system = SYSTEM_ID
    # Retained only because the shared execute loop validates this common
    # parser field.  Vevo never reads or uses it.
    args.openvoice_short_audio_split_seconds = 1.0
    with common_hooks():
        return COMMON.execute(args, backend_factory=backend_factory)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--input-format", choices=("auto", "lst", "jsonl"), default="auto"
    )
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
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--torch-home", type=Path)
    parser.add_argument("--hubert-checkpoint", type=Path)
    parser.add_argument("--hubert-stats", type=Path)
    parser.add_argument("--tokenizer-checkpoint-dir", type=Path)
    parser.add_argument("--fmt-config", type=Path)
    parser.add_argument("--fmt-checkpoint-dir", type=Path)
    parser.add_argument("--vocoder-config", type=Path)
    parser.add_argument("--vocoder-checkpoint-dir", type=Path)
    parser.add_argument("--flow-matching-steps", type=int, default=32)
    parser.add_argument("--target-db", type=float, default=-25.0)
    parser.add_argument(
        "--verify-checkpoint-sha256",
        action="store_true",
        help="Hash the three Hugging Face Vevo checkpoints during runtime audit.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.flow_matching_steps < 1:
        raise ValueError("--flow-matching-steps must be >= 1")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
        return execute(args)
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

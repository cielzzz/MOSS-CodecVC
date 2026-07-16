#!/usr/bin/env python
"""Batch-local validation inference for the ver3.1 DDLFM CFM probe.

The regular :mod:`infer_ddlfm_cfm` entry point is intentionally a small,
auditable one-example command.  Calling it 320 times would reload the 184M
parameter DiT and the MOSS tokenizer for every case.  This runner keeps those
models (and the WavLM/adapter/ECAPA sidecars) resident and writes a normal
SeedTTS-style manifest that can be consumed by the existing ``004050`` and
``004042`` scorers.

Important contracts
-------------------
* no_text semantic = source audio -> WavLM base-plus layer 9 -> the frozen
  Step-3 ContentAdapterV31.  Target-BNF is never used as a fallback.
* text semantic = manifest text -> SentencePiece (+1 offset) -> the trained
  ``SourceTokenMemoryEncoder``.  Source audio is used only for duration and
  the reference ECAPA speaker embedding.
* speaker sidecar = SpeechBrain ECAPA 192-D, matching the Step-4 training
  index (it is not CAM++).
* target frame count = floor(source samples after 24-kHz resampling / 1920),
  matching the tokenizer's encoder length contract.
* inference predicts decoder-domain zq directly and calls ``decode_latents``;
  no quantizer is invoked at inference time.

This script is evaluation-only.  It does not submit QZ jobs and never mutates
an existing output root unless ``--overwrite`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moss_codecvc.audio import (
    decode_latents,
    denormalize_zq,
    load_zq_channel_stats,
    sha256_file,
)
from moss_codecvc.moss_codec import MossCodec
from moss_codecvc.third_party import add_download_python_deps, default_speechbrain_ecapa_dir
from scripts.ver3_1.extract_semantic_v3_1 import (
    extract_source_feature,
    load_adapter,
    load_wavlm,
)
from scripts.ver3_1.infer_ddlfm_cfm import (
    combine_cfg_velocity,
    combine_dual_cfg_velocity,
    encode_text,
    load_checkpoint,
)


DEFAULT_VALIDATION = ROOT / "testset/validation/seedtts_vc_ver2_3_validation.jsonl"
DEFAULT_CHECKPOINT = ROOT / "outputs/ver3_1_ddlfm_cfm_probe_ddpfix_20260715/last.pt"
DEFAULT_ADAPTER = ROOT / "outputs/ver3_1_content_adapter_probe_20260715/step-003000"
DEFAULT_CODEC = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
    "vcdata_construction/MOSS-Audio-Tokenizer"
)
DEFAULT_MOSS_ROOT = Path(
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-TTS"
)
DEFAULT_HF_CACHE = Path(
    "/inspire/ssd/project/embodied-multimodality/public/xyzhang/download/huggingface"
)
DEFAULT_SPM = ROOT / "trainset/shared_content_tokenizers/spm_multilingual_byte_fallback_v1.model"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validation-jsonl", default=str(DEFAULT_VALIDATION))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--adapter-checkpoint", default=str(DEFAULT_ADAPTER))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mode", choices=("all", "no_text", "text"), default="no_text")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--per-mode", type=int, default=0)
    ap.add_argument("--case-id", action="append", default=[])
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--sampling-steps", type=int, default=20)
    ap.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="Default: 1.5 for CFG-trained checkpoints, otherwise 1.0.",
    )
    ap.add_argument("--speaker-cfg-scale", type=float, default=None)
    ap.add_argument("--semantic-cfg-scale", type=float, default=None)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument(
        "--zq-channel-stats",
        default="",
        help="Override channel_stats.pt; default comes from the checkpoint config.",
    )
    ap.add_argument("--seed", type=int, default=20260715)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--codec-path", default=str(DEFAULT_CODEC))
    ap.add_argument("--moss-root", default=str(DEFAULT_MOSS_ROOT))
    ap.add_argument("--codec-dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    ap.add_argument("--wavlm-model", default="microsoft/wavlm-base-plus")
    ap.add_argument("--wavlm-cache-dir", default=str(DEFAULT_HF_CACHE))
    ap.add_argument("--wavlm-precision", choices=("float32", "bf16", "fp16"), default="bf16")
    ap.add_argument("--wavlm-local-files-only", action="store_true")
    ap.add_argument("--spm-model", default=str(DEFAULT_SPM))
    ap.add_argument("--ecapa-model", default=str(default_speechbrain_ecapa_dir()))
    ap.add_argument("--ecapa-device", default="cuda:1")
    ap.add_argument("--semantic-cache-dir", default="")
    ap.add_argument("--speaker-cache-dir", default="")
    ap.add_argument(
        "--semantic-manifest",
        default="",
        help="Optional pre-extracted no_text semantic manifest. Rows are keyed by case_id/sample_id.",
    )
    ap.add_argument(
        "--ecapa-embedding-dir",
        default="",
        help="Optional directory of pre-extracted 192-D ECAPA .pt files named <case_id>.pt.",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-ecapa", action="store_true", help="Write zero speaker vectors; only for wiring tests.")
    return ap.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            yield row


def safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return stem[:180] or "case"


def file_key(path: str | Path) -> str:
    resolved = str(Path(path).expanduser().resolve())
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:20]


def case_seed(base_seed: int, case_id: str) -> int:
    """Return a stable per-case seed independent of batching/sharding/order."""

    digest = hashlib.sha256(f"{int(base_seed)}\0{case_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def required_text(row: dict[str, Any]) -> str:
    value = str(row.get("text") or "").strip()
    if value and value != "<NO_TEXT>":
        return value
    return str(row.get("content_ref_text") or row.get("source_text") or "").strip()


def read_audio_length(path: str | Path, sample_rate: int = 24000) -> tuple[int, int]:
    import soundfile as sf

    info = sf.info(str(path))
    samples = int(info.frames)
    sr = int(info.samplerate)
    if sr != int(sample_rate):
        # This is only a frame-count estimate.  The MOSS codec resamples then
        # uses floor(input_samples / 1920); scale before flooring to match it.
        samples = int(round(samples * float(sample_rate) / float(sr)))
        sr = int(sample_rate)
    return samples, sr


def target_frames(path: str | Path) -> int:
    samples, _ = read_audio_length(path, 24000)
    return max(1, samples // 1920)


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    wanted = {str(x) for x in args.case_id}
    mode_counts: dict[str, int] = {"no_text": 0, "text": 0}
    out: list[dict[str, Any]] = []
    for row in rows:
        case_id = str(row.get("case_id") or "")
        mode = str(row.get("mode") or "")
        if mode not in {"no_text", "text"}:
            continue
        if wanted and case_id not in wanted:
            continue
        if args.mode != "all" and mode != args.mode:
            continue
        if args.per_mode > 0 and mode_counts[mode] >= int(args.per_mode):
            continue
        out.append(row)
        mode_counts[mode] += 1
        if args.max_cases > 0 and len(out) >= int(args.max_cases):
            break
    if int(args.num_shards) <= 0 or not 0 <= int(args.shard_index) < int(args.num_shards):
        raise ValueError("invalid --shard-index/--num-shards")
    if int(args.num_shards) > 1:
        out = [row for i, row in enumerate(out) if i % int(args.num_shards) == int(args.shard_index)]
    return out


def atomic_jsonl_append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_condition_manifest(path: str) -> dict[str, dict[str, Any]]:
    """Load a compact condition manifest keyed by case/sample/utterance id."""

    if not path:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(Path(path).expanduser().resolve()):
        for key in ("case_id", "sample_id", "utt_id", "utterance_id"):
            value = row.get(key)
            if value not in (None, ""):
                result[str(value)] = row
    return result


def load_condition_embedding_dir(path: str) -> dict[str, Path]:
    """Map exact ``<case_id>.pt`` files; no fuzzy matching is intentional."""

    if not path:
        return {}
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ECAPA embedding directory does not exist: {root}")
    return {item.stem: item for item in root.glob("*.pt") if item.is_file()}


def load_ecapa(model_source: str, device: str):
    add_download_python_deps()
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier
    source = str(Path(model_source).expanduser().resolve())
    kwargs: dict[str, Any] = {"source": source, "savedir": source, "run_opts": {"device": str(device)}}
    try:
        encoder = EncoderClassifier.from_hparams(**kwargs)
    except TypeError:
        kwargs.pop("run_opts", None)
        encoder = EncoderClassifier.from_hparams(**kwargs).to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder


@torch.inference_mode()
def ecapa_embedding(encoder: Any, audio_path: str, device: str) -> torch.Tensor:
    if hasattr(encoder, "encode_file"):
        value = encoder.encode_file(str(audio_path)).squeeze()
    else:
        wav = encoder.load_audio(str(audio_path)).to(device)
        value = encoder.encode_batch(wav.unsqueeze(0)).squeeze()
    value = torch.as_tensor(value, dtype=torch.float32).flatten()
    if value.numel() != 192:
        raise ValueError(f"ECAPA embedding must be 192-D, got {value.numel()} for {audio_path}")
    return torch.nn.functional.normalize(value, dim=0).cpu()


def load_cached_embedding(encoder: Any, path: str, cache_dir: Path | None, device: str, overwrite: bool) -> torch.Tensor:
    destination = cache_dir / f"{file_key(path)}.pt" if cache_dir else None
    if destination is not None and destination.is_file() and not overwrite:
        payload = torch.load(str(destination), map_location="cpu", weights_only=False)
        value = payload.get("embedding", payload) if isinstance(payload, dict) else payload
        value = torch.as_tensor(value, dtype=torch.float32).flatten()
        if value.numel() == 192:
            return torch.nn.functional.normalize(value, dim=0)
    value = ecapa_embedding(encoder, path, device)
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"embedding": value, "backend": "speechbrain_ecapa_192d", "audio": str(path)}, destination)
    return value


@dataclass
class Prepared:
    row: dict[str, Any]
    mode: str
    semantic: torch.Tensor
    speaker: torch.Tensor
    target_frames: int
    source_audio: str
    ref_audio: str


class ResourceBundle:
    """Resident model bundle shared by all batches."""

    def __init__(
        self,
        args: argparse.Namespace,
        need_no_text: bool,
        need_ecapa: bool,
        semantic_conditions: dict[str, dict[str, Any]] | None = None,
        ecapa_conditions: dict[str, Path] | None = None,
    ) -> None:
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
        self.module, self.cfg = load_checkpoint(
            Path(args.checkpoint).expanduser().resolve(),
            self.device,
            use_ema=not bool(args.no_ema),
        )
        self.module.eval()
        self.using_ema = bool(self.cfg.get("_checkpoint_using_ema", False))
        self.zq_normalization_enabled = bool(self.cfg.get("zq_normalization_enabled", False))
        self.cfg_scale = (
            float(args.cfg_scale)
            if args.cfg_scale is not None
            else (1.5 if float(self.cfg.get("speaker_dropout", 0.0)) > 0.0 else 1.0)
        )
        self.speaker_cfg_scale = (
            float(args.speaker_cfg_scale)
            if args.speaker_cfg_scale is not None
            else float(self.cfg.get("speaker_cfg_scale", self.cfg_scale))
        )
        self.semantic_cfg_scale = (
            float(args.semantic_cfg_scale)
            if args.semantic_cfg_scale is not None
            else float(self.cfg.get("semantic_cfg_scale", 0.0))
        )
        self.zq_stats_path: Path | None = None
        self.zq_stats_sha256 = ""
        self.zq_stats: dict[str, Any] | None = None
        if self.zq_normalization_enabled:
            stats_value = args.zq_channel_stats or str(self.cfg.get("zq_channel_stats") or "")
            if not stats_value:
                raise ValueError("normalized DDLFM evaluation requires zq channel stats")
            self.zq_stats_path = Path(stats_value).expanduser().resolve()
            self.zq_stats_sha256 = sha256_file(self.zq_stats_path)
            expected_stats_sha = str(self.cfg.get("zq_channel_stats_sha256") or "")
            if expected_stats_sha and self.zq_stats_sha256 != expected_stats_sha:
                raise ValueError(
                    "zq channel stats SHA256 mismatch: "
                    f"{self.zq_stats_sha256} != {expected_stats_sha}"
                )
            self.zq_stats = load_zq_channel_stats(self.zq_stats_path)
            if str(self.zq_stats.get("status")) != "completed" or bool(self.zq_stats.get("partial", False)):
                raise ValueError(f"refusing incomplete zq channel stats: {self.zq_stats_path}")
        self.codec = MossCodec(
            args.codec_path,
            moss_root=args.moss_root or None,
            device=str(self.device),
            dtype=args.codec_dtype,
        )
        self.spm_model = str(Path(args.spm_model).expanduser().resolve())
        self.semantic_cache = Path(args.semantic_cache_dir).expanduser().resolve() if args.semantic_cache_dir else None
        self.speaker_cache = Path(args.speaker_cache_dir).expanduser().resolve() if args.speaker_cache_dir else None
        self.semantic_conditions = dict(semantic_conditions or {})
        self.ecapa_conditions = dict(ecapa_conditions or {})
        self.adapter = None
        self.wavlm_extractor = None
        self.wavlm_model = None
        self.wavlm_device = None
        if need_no_text:
            self.adapter, _ = load_adapter(Path(args.adapter_checkpoint).expanduser().resolve(), self.device)
            wavlm_args = SimpleNamespace(
                wavlm_model=args.wavlm_model,
                cache_dir=args.wavlm_cache_dir,
                local_files_only=bool(args.wavlm_local_files_only),
                device=str(self.device),
                precision=args.wavlm_precision,
            )
            self.wavlm_extractor, self.wavlm_model, self.wavlm_device = load_wavlm(wavlm_args)
        self.ecapa = None
        if need_ecapa and not args.skip_ecapa:
            self.ecapa = load_ecapa(args.ecapa_model, args.ecapa_device)

    def semantic_no_text(self, source_audio: str, case_id: str = "") -> torch.Tensor:
        condition = self.semantic_conditions.get(str(case_id))
        if condition is not None:
            semantic_path = condition.get("semantic_v3_1_path") or condition.get("semantic_path")
            if semantic_path and Path(str(semantic_path)).is_file():
                value = np.load(str(semantic_path)).astype("float32", copy=False)
                if value.ndim != 2 or value.shape[1] != 512 or value.shape[0] <= 0:
                    raise ValueError(f"invalid pre-extracted semantic for {case_id}: {value.shape}")
                return torch.from_numpy(value)
            raise FileNotFoundError(f"semantic condition row for {case_id} has no readable path")
        cache_path = self.semantic_cache / f"{file_key(source_audio)}.npy" if self.semantic_cache else None
        if cache_path is not None and cache_path.is_file() and not self.args.overwrite:
            value = np.load(cache_path).astype("float32", copy=False)
            return torch.from_numpy(value)
        if self.wavlm_model is None or self.adapter is None:
            raise RuntimeError("no_text requested but WavLM/adapter are not loaded")
        feature = extract_source_feature(source_audio, self.wavlm_extractor, self.wavlm_model, self.wavlm_device)
        output = self.adapter(feature.unsqueeze(0).to(self.device), torch.ones((1, feature.shape[0]), dtype=torch.bool, device=self.device))
        value = output.semantic[0, output.semantic_mask[0]].detach().float().cpu()
        if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] != 512:
            raise ValueError(f"invalid adapter semantic for {source_audio}: {tuple(value.shape)}")
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_name(f".{cache_path.name}.tmp-{os.getpid()}")
            with tmp.open("wb") as handle:
                np.save(handle, value.numpy().astype(np.float16, copy=False))
            tmp.replace(cache_path)
        return value

    def semantic_text(self, text: str) -> torch.Tensor:
        ids = encode_text(text, self.spm_model).unsqueeze(0).to(self.device)
        mask = torch.ones_like(ids, dtype=torch.bool, device=self.device)
        state = self.module.text_encoder(ids, mask)
        return state.memory[0, state.mask[0]].detach().float().cpu()

    def speaker_embedding(self, ref_audio: str, case_id: str = "") -> torch.Tensor:
        if self.args.skip_ecapa:
            return torch.zeros(192, dtype=torch.float32)
        condition_path = self.ecapa_conditions.get(str(case_id))
        if condition_path is not None:
            payload = torch.load(str(condition_path), map_location="cpu", weights_only=False)
            value = payload.get("embedding", payload) if isinstance(payload, dict) else payload
            value = torch.as_tensor(value, dtype=torch.float32).flatten()
            if value.numel() != 192:
                raise ValueError(f"ECAPA condition for {case_id} is not 192-D: {condition_path}")
            return torch.nn.functional.normalize(value, dim=0)
        if self.ecapa is None:
            raise RuntimeError("ECAPA encoder is not loaded")
        return load_cached_embedding(self.ecapa, ref_audio, self.speaker_cache, self.args.ecapa_device, self.args.overwrite)


@torch.inference_mode()
def sample_velocity_batch(
    module: Any,
    target_lengths: list[int],
    semantics: list[torch.Tensor],
    speakers: torch.Tensor,
    modalities: list[int],
    steps: int,
    cfg_scale: float,
    semantic_cfg_scale: float,
    device: torch.device,
    seeds: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not target_lengths or int(steps) <= 0:
        raise ValueError("target_lengths must be non-empty and steps must be positive")
    batch = len(target_lengths)
    max_t = max(int(x) for x in target_lengths)
    max_s = max(int(x.shape[0]) for x in semantics)
    x = torch.zeros((batch, max_t, 768), dtype=torch.float32, device=device)
    target_mask = torch.zeros((batch, max_t), dtype=torch.bool, device=device)
    semantic = torch.zeros((batch, max_s, 512), dtype=torch.float32, device=device)
    semantic_mask = torch.zeros((batch, max_s), dtype=torch.bool, device=device)
    if len(seeds) != batch:
        raise ValueError(f"expected {batch} per-case seeds, got {len(seeds)}")
    for i, (length, value) in enumerate(zip(target_lengths, semantics)):
        length = int(length)
        generator = torch.Generator(device=device).manual_seed(int(seeds[i]))
        x[i, :length] = torch.randn((length, 768), generator=generator, device=device)
        target_mask[i, :length] = True
        semantic[i, : value.shape[0]] = value.to(device=device, dtype=torch.float32)
        semantic_mask[i, : value.shape[0]] = True
    speaker = speakers.to(device=device, dtype=torch.float32)
    if not math.isfinite(float(cfg_scale)) or float(cfg_scale) < 0.0:
        raise ValueError("cfg_scale must be finite and non-negative")
    zero_speaker = torch.zeros_like(speaker)
    zero_semantic = torch.zeros_like(semantic)
    zero_semantic_mask = torch.zeros_like(semantic_mask, dtype=torch.bool)
    modality = torch.as_tensor(modalities, dtype=torch.long, device=device)
    if not math.isfinite(float(semantic_cfg_scale)) or float(semantic_cfg_scale) < 0.0:
        raise ValueError("semantic_cfg_scale must be finite and non-negative")
    for index in range(int(steps)):
        t = torch.full((batch,), float(index) / float(steps), device=device)
        if float(semantic_cfg_scale) > 0.0:
            velocity_uncond = module.decoder(
                x, t, zero_semantic, zero_speaker,
                target_mask=target_mask,
                semantic_mask=zero_semantic_mask,
                semantic_modality=modality,
            ).velocity
            velocity_speaker = module.decoder(
                x, t, zero_semantic, speaker,
                target_mask=target_mask,
                semantic_mask=zero_semantic_mask,
                semantic_modality=modality,
            ).velocity
            velocity_semantic = module.decoder(
                x, t, semantic, zero_speaker,
                target_mask=target_mask,
                semantic_mask=semantic_mask,
                semantic_modality=modality,
            ).velocity
            velocity = combine_dual_cfg_velocity(
                velocity_uncond,
                velocity_speaker,
                velocity_semantic,
                float(cfg_scale),
                float(semantic_cfg_scale),
            )
        else:
            velocity_cond = module.decoder(
                x,
                t,
                semantic,
                speaker,
                target_mask=target_mask,
                semantic_mask=semantic_mask,
                semantic_modality=modality,
            ).velocity
            if float(cfg_scale) == 1.0:
                velocity = velocity_cond
            else:
                velocity_uncond = module.decoder(
                    x,
                    t,
                    semantic,
                    zero_speaker,
                    target_mask=target_mask,
                    semantic_mask=semantic_mask,
                    semantic_modality=modality,
                ).velocity
                velocity = combine_cfg_velocity(velocity_cond, velocity_uncond, float(cfg_scale))
        x = x + velocity / float(steps)
    return x, torch.as_tensor(target_lengths, dtype=torch.long, device=device)


def write_wavs(bundle: ResourceBundle, prepared: list[Prepared], output_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    import soundfile as sf

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to append to an existing generated manifest: {manifest_path}; "
            "choose a fresh --output-dir or pass --overwrite"
        )
    if manifest_path.exists() and args.overwrite:
        manifest_path.unlink()
    rows_out: list[dict[str, Any]] = []
    for start in range(0, len(prepared), max(1, int(args.batch_size))):
        group = prepared[start : start + max(1, int(args.batch_size))]
        paths = [output_dir / f"{safe_stem(p.row.get('case_id', f'case-{start+i}'))}.wav" for i, p in enumerate(group)]
        todo = [(i, p) for i, (p, path) in enumerate(zip(group, paths)) if args.overwrite or not path.exists()]
        if todo:
            todo_group = [item[1] for item in todo]
            z_pred, lengths = sample_velocity_batch(
                bundle.module,
                [item.target_frames for item in todo_group],
                [item.semantic for item in todo_group],
                torch.stack([item.speaker for item in todo_group]),
                [0 if item.mode == "no_text" else 1 for item in todo_group],
                int(args.sampling_steps),
                float(bundle.speaker_cfg_scale),
                float(bundle.semantic_cfg_scale),
                bundle.device,
                [case_seed(int(args.seed), str(item.row.get("case_id") or "")) for item in todo_group],
            )
            if bundle.zq_normalization_enabled:
                if bundle.zq_stats is None:
                    raise RuntimeError("normalization is enabled but zq stats are unavailable")
                z_pred = denormalize_zq(z_pred, bundle.zq_stats, channel_dim=-1)
            latent_lengths = lengths.to(device=bundle.device)
            waveform, waveform_lengths = decode_latents(
                bundle.codec.model,
                z_pred.transpose(1, 2).contiguous(),
                latent_lengths,
            )
            for local_index, item in enumerate(todo_group):
                orig_index = todo[local_index][0]
                path = paths[orig_index]
                wav = waveform[local_index, 0, : int(waveform_lengths[local_index].item())].detach().float().cpu().numpy()
                sf.write(str(path), wav, bundle.codec.sample_rate)
        for item, path in zip(group, paths):
            status = "ok" if path.is_file() else "failed"
            manifest_row = {
                "case_id": item.row.get("case_id"),
                "mode": item.mode,
                "cell": item.row.get("cell"),
                "source_audio": item.source_audio,
                "timbre_ref_audio": item.ref_audio,
                "source_text": item.row.get("source_text"),
                "content_ref_text": item.row.get("content_ref_text"),
                "text": item.row.get("text"),
                "target_audio": str(path),
                "output_wav": str(path),
                "target_frames": int(item.target_frames),
                "sampling_steps": int(args.sampling_steps),
                "cfg_scale": float(bundle.speaker_cfg_scale),
                "speaker_cfg_scale": float(bundle.speaker_cfg_scale),
                "semantic_cfg_scale": float(bundle.semantic_cfg_scale),
                "cfg_formula": "four_state_additive_v1" if bundle.semantic_cfg_scale > 0.0 else "single_condition_v1",
                "using_ema": bundle.using_ema,
                "zq_normalization_enabled": bundle.zq_normalization_enabled,
                "zq_channel_stats": str(bundle.zq_stats_path) if bundle.zq_stats_path is not None else None,
                "zq_channel_stats_sha256": bundle.zq_stats_sha256 or None,
                "seed": int(args.seed),
                "case_seed": case_seed(int(args.seed), str(item.row.get("case_id") or "")),
                "speaker_embedding_backend": "speechbrain_ecapa_192d_sidecar",
                "status": status,
            }
            atomic_jsonl_append(manifest_path, manifest_row)
            rows_out.append(manifest_row)
        print(f"[ddlfm-eval] generated {min(start + len(group), len(prepared))}/{len(prepared)}", flush=True)
    return rows_out


def main() -> int:
    args = parse_args()
    validation = Path(args.validation_jsonl).expanduser().resolve()
    rows = select_rows(list(iter_jsonl(validation)), args)
    if not rows:
        raise ValueError("no validation rows selected")
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run",
            "validation_jsonl": str(validation),
            "rows": len(rows),
            "by_mode": {mode: sum(str(r.get("mode")) == mode for r in rows) for mode in ("no_text", "text")},
            "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        }, ensure_ascii=False, indent=2))
        return 0
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest = output_dir / "manifest.jsonl"
    if existing_manifest.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to reuse an existing generated manifest: {existing_manifest}; "
            "choose a fresh --output-dir or pass --overwrite"
        )
    semantic_conditions = load_condition_manifest(args.semantic_manifest)
    ecapa_conditions = load_condition_embedding_dir(args.ecapa_embedding_dir)
    no_text_rows = [row for row in rows if str(row.get("mode")) == "no_text"]
    missing_semantic = [
        str(row.get("case_id") or "")
        for row in no_text_rows
        if str(row.get("case_id") or "") not in semantic_conditions
    ]
    missing_ecapa = [
        str(row.get("case_id") or "")
        for row in rows
        if str(row.get("case_id") or "") not in ecapa_conditions
    ]
    need_no_text = bool(missing_semantic)
    need_ecapa = bool(missing_ecapa) and not bool(args.skip_ecapa)
    bundle = ResourceBundle(
        args,
        need_no_text=need_no_text,
        need_ecapa=need_ecapa,
        semantic_conditions=semantic_conditions,
        ecapa_conditions=ecapa_conditions,
    )
    if missing_semantic and not args.semantic_manifest:
        print("[ddlfm-eval] no semantic manifest supplied; extracting no_text conditions with resident WavLM+adapter", flush=True)
    elif missing_semantic:
        print(f"[ddlfm-eval] semantic manifest missing {len(missing_semantic)} rows; extracting those rows online", flush=True)
    if missing_ecapa and not args.skip_ecapa and not args.ecapa_embedding_dir:
        print("[ddlfm-eval] no ECAPA condition dir supplied; extracting ref embeddings with resident SpeechBrain ECAPA", flush=True)
    prepared: list[Prepared] = []
    started = time.time()
    for index, row in enumerate(rows):
        mode = str(row.get("mode") or "")
        source_audio = str(row.get("source_audio") or "")
        ref_audio = str(row.get("timbre_ref_audio") or "")
        if not Path(source_audio).is_file() or not Path(ref_audio).is_file():
            raise FileNotFoundError(f"missing source/ref audio for {row.get('case_id')}: {source_audio} / {ref_audio}")
        case_id = str(row.get("case_id") or "")
        semantic = bundle.semantic_no_text(source_audio, case_id) if mode == "no_text" else bundle.semantic_text(required_text(row))
        speaker = bundle.speaker_embedding(ref_audio, case_id)
        prepared.append(Prepared(row, mode, semantic, speaker, target_frames(source_audio), source_audio, ref_audio))
        if (index + 1) % 10 == 0 or index + 1 == len(rows):
            print(f"[ddlfm-eval] prepared {index + 1}/{len(rows)}", flush=True)
    output_rows = write_wavs(bundle, prepared, output_dir, args)
    case_map_path = output_dir / "case_id_to_wav.json"
    case_map_path.write_text(
        json.dumps(
            {str(row["case_id"]): str(row["output_wav"]) for row in output_rows},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    completion = {
        "schema": "ver3_1_ddlfm_validation_completion_v1",
        "status": "completed",
        "validation_jsonl": str(validation),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "adapter_checkpoint": str(Path(args.adapter_checkpoint).expanduser().resolve()),
        "output_dir": str(output_dir),
        "generated_manifest": str(output_dir / "manifest.jsonl"),
        "case_id_to_wav": str(case_map_path),
        "rows": len(output_rows),
        "by_mode": {mode: sum(str(r.get("mode")) == mode for r in output_rows) for mode in ("no_text", "text")},
        "sampling_steps": int(args.sampling_steps),
        "cfg_scale": float(bundle.cfg_scale),
        "speaker_cfg_scale": float(bundle.speaker_cfg_scale),
        "semantic_cfg_scale": float(bundle.semantic_cfg_scale),
        "using_ema": bundle.using_ema,
        "zq_normalization_enabled": bundle.zq_normalization_enabled,
        "zq_channel_stats": str(bundle.zq_stats_path) if bundle.zq_stats_path is not None else None,
        "zq_channel_stats_sha256": bundle.zq_stats_sha256 or None,
        "seed": int(args.seed),
        "target_frame_rate_hz": 12.5,
        "speaker_backend": "speechbrain_ecapa_192d",
        "elapsed_sec": round(time.time() - started, 3),
    }
    (output_dir / "COMPLETED.json").write_text(json.dumps(completion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(completion, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

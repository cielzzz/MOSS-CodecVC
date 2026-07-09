from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

from moss_codecvc.io_utils import load_torch_file
from moss_codecvc.third_party import add_download_python_deps, default_speechbrain_ecapa_dir


DEFAULT_SEED_TTS_EVAL_MODEL_ROOT = (
    "/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/"
    "vcdata_construction/models"
)


def _extract_embedding(payload: Any, path: Path) -> torch.Tensor:
    if torch.is_tensor(payload):
        emb = payload
    elif isinstance(payload, dict):
        emb = None
        for key in ("embedding", "speaker_embedding", "emb", "xvector", "vector"):
            value = payload.get(key)
            if value is not None:
                emb = value
                break
        if emb is None:
            raise ValueError(f"No speaker embedding tensor found in {path}")
    else:
        emb = payload
    emb = torch.as_tensor(emb, dtype=torch.float32)
    if emb.dim() > 1:
        emb = emb.reshape(-1, emb.shape[-1]).mean(dim=0)
    if emb.dim() != 1:
        raise ValueError(f"Speaker embedding from {path} must flatten to [D], got {tuple(emb.shape)}")
    return emb


class FrozenSpeakerEmbeddingLoader(nn.Module):
    """Frozen speaker-encoder interface backed by precomputed embeddings.

    This keeps the Ver1.6 training contract compatible with a real frozen
    speaker encoder later: callers ask for speaker embeddings, and this module
    supplies normalized vectors without adding trainable parameters.
    """

    def __init__(self, embedding_dim: int | None = None) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim) if embedding_dim else None
        self.expects_audio_paths = False
        self._cache: dict[str, torch.Tensor] = {}
        for param in self.parameters():
            param.requires_grad = False

    def _load_one(self, path: str | Path) -> torch.Tensor:
        key = str(path)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        path_obj = Path(path)
        suffix = path_obj.suffix.lower()
        if suffix == ".npy":
            import numpy as np

            payload = np.load(path_obj)
        elif suffix == ".npz":
            import numpy as np

            npz = np.load(path_obj)
            if "embedding" in npz:
                payload = npz["embedding"]
            else:
                first_key = next(iter(npz.files))
                payload = npz[first_key]
        else:
            payload = load_torch_file(path_obj)
        emb = _extract_embedding(payload, path_obj)
        if self.embedding_dim is not None and int(emb.numel()) != self.embedding_dim:
            raise ValueError(
                f"Speaker embedding dim mismatch for {path_obj}: got {int(emb.numel())}, expected {self.embedding_dim}"
            )
        emb = torch.nn.functional.normalize(emb.float(), dim=0)
        self._cache[key] = emb
        return emb

    def forward(
        self,
        paths: list[str | None] | tuple[str | None, ...] | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if paths is None:
            return None, None
        loaded: list[torch.Tensor | None] = []
        valid: list[bool] = []
        inferred_dim = self.embedding_dim
        for path in paths:
            if not path:
                loaded.append(None)
                valid.append(False)
                continue
            emb = self._load_one(path)
            inferred_dim = inferred_dim or int(emb.numel())
            loaded.append(emb)
            valid.append(True)
        if inferred_dim is None:
            return None, None
        rows = []
        for emb in loaded:
            if emb is None:
                rows.append(torch.zeros(inferred_dim, dtype=torch.float32))
            else:
                rows.append(emb)
        batch = torch.stack(rows, dim=0).to(device=device, dtype=dtype)
        mask = torch.tensor(valid, device=device, dtype=torch.bool)
        return batch, mask


class FrozenSpeechBrainECAPAEncoder(nn.Module):
    """Optional frozen online ECAPA-TDNN wrapper.

    This path is intentionally not the default for training: Ver1.6 prefers
    precomputed E_src/E_ref/E_tgt embeddings in the manifest to avoid running a
    wav speaker encoder every step.
    """

    def __init__(self, source: str | None = None, savedir: str | None = None) -> None:
        super().__init__()
        self.expects_audio_paths = True
        add_download_python_deps()
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:
            raise ImportError(
                "speaker_encoder_type='speechbrain_ecapa' requires the `speechbrain` package. "
                "Use speaker_encoder_type='embedding_loader' with precomputed ECAPA embeddings otherwise."
            ) from exc
        resolved_source = source
        if not resolved_source:
            local_source = default_speechbrain_ecapa_dir()
            resolved_source = str(local_source) if local_source.exists() else "speechbrain/spkrec-ecapa-voxceleb"
        self.classifier = EncoderClassifier.from_hparams(
            source=resolved_source,
            savedir=savedir,
            run_opts={"device": "cpu"},
        )
        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad = False

    def _sync_classifier_device(self, device: torch.device) -> torch.device:
        normalized = torch.device(device)
        device_str = str(normalized)
        if getattr(self.classifier, "device", None) != device_str:
            self.classifier.device = device_str
            self.classifier.device_type = normalized.type
            self.classifier.to(normalized)
            if hasattr(self.classifier, "mods"):
                self.classifier.mods.to(normalized)
        return normalized

    @torch.inference_mode()
    def _load_one(self, path: str | Path, *, device: torch.device) -> torch.Tensor:
        device = self._sync_classifier_device(device)
        if hasattr(self.classifier, "encode_file"):
            emb = self.classifier.encode_file(str(path)).squeeze()
        else:
            signal = self.classifier.load_audio(str(path))
            signal = signal.to(device)
            emb = self.classifier.encode_batch(signal.unsqueeze(0)).squeeze()
        return torch.nn.functional.normalize(emb.float(), dim=0).to(device=device)

    def forward(
        self,
        paths: list[str | None] | tuple[str | None, ...] | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if paths is None:
            return None, None
        rows: list[torch.Tensor | None] = []
        valid: list[bool] = []
        inferred_dim = None
        for path in paths:
            if not path:
                rows.append(None)
                valid.append(False)
                continue
            emb = self._load_one(path, device=device)
            inferred_dim = int(emb.numel())
            rows.append(emb)
            valid.append(True)
        if inferred_dim is None:
            return None, None
        batch = torch.stack(
            [emb if emb is not None else torch.zeros(inferred_dim, device=device) for emb in rows],
            dim=0,
        ).to(dtype=dtype)
        mask = torch.tensor(valid, device=device, dtype=torch.bool)
        return batch, mask


def _resolve_seed_tts_eval_paths(model_root_or_checkpoint: str | None) -> tuple[Path, Path, Path]:
    root = Path(model_root_or_checkpoint or DEFAULT_SEED_TTS_EVAL_MODEL_ROOT).expanduser()
    if root.is_file():
        checkpoint = root
        model_root = root.parent
    else:
        model_root = root
        checkpoint = model_root / "wavlm_large_finetune.pth"
    seed_tts_eval_root = model_root / "seed-tts-eval"
    wavlm_dir = model_root / "wavlm-large"
    missing = [str(path) for path in (checkpoint, seed_tts_eval_root, wavlm_dir) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "SeedTTSEval ECAPA backend expects a model root containing "
            "wavlm_large_finetune.pth, seed-tts-eval/ and wavlm-large/. "
            f"Missing: {missing}. Got speaker_encoder_path={model_root_or_checkpoint!r}"
        )
    return checkpoint, seed_tts_eval_root, wavlm_dir


class FrozenSeedTTSEvalECAPAEncoder(nn.Module):
    """Frozen online WavLM-Large + ECAPA-TDNN speaker encoder.

    This wraps the local `vcdata_construction/speaker_similarity.py` implementation.
    It is useful for smoke experiments that should not precompute embeddings, but
    full training is usually faster and more reproducible with offline embeddings.
    """

    def __init__(
        self,
        model_root_or_checkpoint: str | None = None,
        *,
        embedding_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.expects_audio_paths = True
        self.embedding_dim = int(embedding_dim) if embedding_dim else None
        self.checkpoint, self.seed_tts_eval_root, self.wavlm_dir = _resolve_seed_tts_eval_paths(
            model_root_or_checkpoint
        )
        self.vcdata_root = self.seed_tts_eval_root.parent.parent
        self._encoder = None
        self._encoder_device: torch.device | None = None
        self._cache: dict[str, torch.Tensor] = {}

    def _get_encoder(self, device: torch.device):
        normalized_device = torch.device(device)
        if self._encoder is not None and self._encoder_device == normalized_device:
            return self._encoder
        if str(self.vcdata_root) not in sys.path:
            sys.path.insert(0, str(self.vcdata_root))
        from speaker_similarity import SpeakerSimilarity

        encoder = SpeakerSimilarity(
            device=str(normalized_device),
            checkpoint=str(self.checkpoint),
            seed_tts_eval_root=str(self.seed_tts_eval_root),
            wavlm_dir=str(self.wavlm_dir),
        )
        if hasattr(encoder, "model"):
            encoder.model.eval()
            for param in encoder.model.parameters():
                param.requires_grad = False
        self._encoder = encoder
        self._encoder_device = normalized_device
        return encoder

    @torch.inference_mode()
    def _load_one(self, path: str | Path, *, device: torch.device) -> torch.Tensor:
        key = str(path)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"speaker audio path does not exist: {path_obj}")
        encoder = self._get_encoder(device)
        emb = torch.as_tensor(encoder.embed_from_file(str(path_obj)), dtype=torch.float32).flatten()
        if self.embedding_dim is not None and int(emb.numel()) != self.embedding_dim:
            raise ValueError(
                f"SeedTTSEval ECAPA embedding dim mismatch for {path_obj}: "
                f"got {int(emb.numel())}, expected {self.embedding_dim}. "
                "Use --speaker-embedding-dim 256 for the local WavLM-Large + ECAPA model."
            )
        emb = torch.nn.functional.normalize(emb.float(), dim=0).cpu()
        self._cache[key] = emb
        return emb

    def forward(
        self,
        paths: list[str | None] | tuple[str | None, ...] | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if paths is None:
            return None, None
        loaded: list[torch.Tensor | None] = []
        valid: list[bool] = []
        inferred_dim = self.embedding_dim
        for path in paths:
            if not path:
                loaded.append(None)
                valid.append(False)
                continue
            emb = self._load_one(path, device=device)
            inferred_dim = inferred_dim or int(emb.numel())
            loaded.append(emb)
            valid.append(True)
        if inferred_dim is None:
            return None, None
        batch = torch.stack(
            [emb if emb is not None else torch.zeros(inferred_dim, dtype=torch.float32) for emb in loaded],
            dim=0,
        ).to(device=device, dtype=dtype)
        mask = torch.tensor(valid, device=device, dtype=torch.bool)
        return batch, mask


class FrozenWavLMSVEncoder(nn.Module):
    """Frozen Hugging Face WavLM speaker-vector encoder.

    The preferred full-training path is still offline `speaker_vec_path`
    precomputation. This online backend keeps smoke tests and precompute scripts
    using the same encoder contract as training.
    """

    def __init__(
        self,
        model_name_or_path: str | None = None,
        *,
        embedding_dim: int | None = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.expects_audio_paths = True
        self.model_name_or_path = model_name_or_path or "microsoft/wavlm-base-plus-sv"
        self.embedding_dim = int(embedding_dim) if embedding_dim else None
        self.local_files_only = bool(local_files_only)
        self._processor = None
        self._model = None
        self._model_device: torch.device | None = None
        self._cache: dict[str, torch.Tensor] = {}

    def _get_model(self, device: torch.device):
        normalized_device = torch.device(device)
        if self._model is not None and self._model_device == normalized_device:
            return self._processor, self._model
        from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor

        try:
            processor = AutoFeatureExtractor.from_pretrained(
                self.model_name_or_path,
                local_files_only=self.local_files_only,
            )
        except Exception:
            processor = AutoProcessor.from_pretrained(
                self.model_name_or_path,
                local_files_only=self.local_files_only,
            )
        try:
            from transformers import AutoModelForAudioXVector

            model = AutoModelForAudioXVector.from_pretrained(
                self.model_name_or_path,
                local_files_only=self.local_files_only,
            )
        except Exception:
            model = AutoModel.from_pretrained(
                self.model_name_or_path,
                local_files_only=self.local_files_only,
            )
        model.eval().to(normalized_device)
        for param in model.parameters():
            param.requires_grad = False
        self._processor = processor
        self._model = model
        self._model_device = normalized_device
        return processor, model

    @staticmethod
    def _read_audio(path_obj: Path) -> tuple[torch.Tensor, int]:
        try:
            import soundfile as sf

            data, sample_rate = sf.read(str(path_obj), dtype="float32", always_2d=True)
            wav = torch.from_numpy(data).float()
            if wav.dim() == 2:
                wav = wav.mean(dim=1)
            return wav.contiguous(), int(sample_rate)
        except Exception as sf_exc:
            try:
                import librosa

                data, sample_rate = librosa.load(str(path_obj), sr=None, mono=True)
                wav = torch.as_tensor(data, dtype=torch.float32)
                return wav.contiguous(), int(sample_rate)
            except Exception as librosa_exc:
                try:
                    import torchaudio

                    wav, sample_rate = torchaudio.load(str(path_obj))
                    wav = wav.float().mean(dim=0)
                    return wav.contiguous(), int(sample_rate)
                except Exception as torchaudio_exc:
                    raise RuntimeError(
                        f"failed to load speaker audio {path_obj}; "
                        f"soundfile={sf_exc!r}; librosa={librosa_exc!r}; torchaudio={torchaudio_exc!r}"
                    ) from torchaudio_exc

    @staticmethod
    def _resample_audio(wav: torch.Tensor, sample_rate: int, target_rate: int) -> torch.Tensor:
        if int(sample_rate) == int(target_rate):
            return wav
        try:
            import torchaudio.functional as AF

            return AF.resample(wav, int(sample_rate), int(target_rate)).contiguous()
        except Exception:
            import math

            import numpy as np
            from scipy.signal import resample_poly

            gcd = math.gcd(int(sample_rate), int(target_rate))
            up = int(target_rate) // gcd
            down = int(sample_rate) // gcd
            data = resample_poly(wav.detach().cpu().numpy(), up, down).astype(np.float32)
            return torch.from_numpy(data).contiguous()

    @staticmethod
    def _extract_model_embeddings(outputs, *, context: str) -> torch.Tensor:
        emb = getattr(outputs, "embeddings", None)
        if emb is None:
            emb = getattr(outputs, "last_hidden_state", None)
            if emb is None and isinstance(outputs, (tuple, list)) and outputs:
                emb = outputs[0]
            if emb is None:
                raise RuntimeError(f"WavLM-SV model returned no embedding/hidden state for {context}")
            if emb.dim() >= 3:
                emb = emb.mean(dim=1)
        emb = torch.as_tensor(emb, dtype=torch.float32)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        elif emb.dim() > 2:
            emb = emb.reshape(emb.shape[0], -1, emb.shape[-1]).mean(dim=1)
        return emb

    @torch.inference_mode()
    def _load_many(self, paths: list[str | Path], *, device: torch.device) -> list[torch.Tensor]:
        results: list[torch.Tensor | None] = [None] * len(paths)
        pending: list[tuple[int, str, Path]] = []
        for idx, path in enumerate(paths):
            key = str(path)
            cached = self._cache.get(key)
            if cached is not None:
                results[idx] = cached
                continue
            path_obj = Path(path)
            if not path_obj.exists():
                raise FileNotFoundError(f"speaker audio path does not exist: {path_obj}")
            pending.append((idx, key, path_obj))
        if not pending:
            return [emb for emb in results if emb is not None]
        processor, model = self._get_model(device)
        target_rate = int(getattr(processor, "sampling_rate", 16000) or 16000)
        wavs = []
        for _idx, _key, path_obj in pending:
            wav, sample_rate = self._read_audio(path_obj)
            if int(sample_rate) != target_rate:
                wav = self._resample_audio(wav, int(sample_rate), target_rate)
            wavs.append(wav.detach().cpu().numpy())
        inputs = processor(
            wavs,
            sampling_rate=target_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        outputs = model(**inputs)
        embs = self._extract_model_embeddings(
            outputs,
            context=", ".join(str(item[2]) for item in pending[:3]),
        )
        if int(embs.shape[0]) != len(pending):
            raise RuntimeError(f"WavLM-SV batch size mismatch: got {tuple(embs.shape)}, expected {len(pending)}")
        for row_idx, (_out_idx, key, path_obj) in enumerate(pending):
            emb = embs[row_idx].reshape(-1)
            if self.embedding_dim is not None and int(emb.numel()) != self.embedding_dim:
                raise ValueError(
                    f"WavLM-SV embedding dim mismatch for {path_obj}: "
                    f"got {int(emb.numel())}, expected {self.embedding_dim}"
                )
            emb = torch.nn.functional.normalize(emb.float(), dim=0).cpu()
            self._cache[key] = emb
            results[_out_idx] = emb
        return [emb for emb in results if emb is not None]

    @torch.inference_mode()
    def _load_one(self, path: str | Path, *, device: torch.device) -> torch.Tensor:
        return self._load_many([path], device=device)[0]

    def forward(
        self,
        paths: list[str | None] | tuple[str | None, ...] | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if paths is None:
            return None, None
        loaded: list[torch.Tensor | None] = []
        valid: list[bool] = []
        inferred_dim = self.embedding_dim
        valid_paths: list[str | Path] = []
        valid_indices: list[int] = []
        for path in paths:
            if not path:
                loaded.append(None)
                valid.append(False)
                continue
            valid_indices.append(len(loaded))
            valid_paths.append(path)
            loaded.append(None)
            valid.append(True)
        if valid_paths:
            embs = self._load_many(valid_paths, device=device)
            for idx, emb in zip(valid_indices, embs, strict=True):
                inferred_dim = inferred_dim or int(emb.numel())
                loaded[idx] = emb
        if inferred_dim is None:
            return None, None
        batch = torch.stack(
            [emb if emb is not None else torch.zeros(inferred_dim, dtype=torch.float32) for emb in loaded],
            dim=0,
        ).to(device=device, dtype=dtype)
        mask = torch.tensor(valid, device=device, dtype=torch.bool)
        return batch, mask


def build_frozen_speaker_encoder(
    encoder_type: str = "embedding_loader",
    *,
    encoder_path: str | None = None,
    embedding_dim: int | None = None,
) -> nn.Module:
    normalized = str(encoder_type or "embedding_loader").strip().lower()
    if normalized in {"embedding_loader", "precomputed", "precomputed_ecapa", "ecapa_embedding_loader"}:
        return FrozenSpeakerEmbeddingLoader(embedding_dim=embedding_dim)
    if normalized in {"speechbrain_ecapa", "ecapa_tdnn", "ecapa"}:
        return FrozenSpeechBrainECAPAEncoder(source=encoder_path)
    if normalized in {"seed_tts_eval_ecapa", "seedttseval_ecapa", "wavlm_ecapa", "wavlm_large_ecapa"}:
        return FrozenSeedTTSEvalECAPAEncoder(encoder_path, embedding_dim=embedding_dim)
    if normalized in {"wavlm_sv", "wavlm_base_plus_sv", "wavlm-base-plus-sv", "wavlm_xvector"}:
        return FrozenWavLMSVEncoder(encoder_path, embedding_dim=embedding_dim)
    raise ValueError(
        f"unsupported speaker_encoder_type={encoder_type!r}; "
        "expected embedding_loader/precomputed_ecapa, speechbrain_ecapa, seed_tts_eval_ecapa, or wavlm_sv"
    )

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch


def ensure_moss_on_path(moss_root: str | Path) -> None:
    root = str(Path(moss_root).expanduser().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


class MossCodec:
    """Thin wrapper around MOSS-Audio-Tokenizer.

    The public tokenizer API uses `(NQ, B, T)` internally. This wrapper keeps the
    MOSS-TTS processor convention: codes are always `(T, NQ)` on CPU.
    """

    def __init__(
        self,
        codec_path: str | Path,
        *,
        moss_root: str | Path | None = None,
        device: str = "cuda:0",
        dtype: str = "float32",
        trust_remote_code: bool = True,
    ) -> None:
        if moss_root is not None:
            ensure_moss_on_path(moss_root)

        from transformers import AutoModel

        self.codec_path = str(codec_path)
        self.device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype, torch.float32)
        self.torch_dtype = torch_dtype if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            self.codec_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=self.torch_dtype,
        ).eval()
        self.model.to(self.device)
        self.sample_rate = int(getattr(self.model, "sampling_rate", 24000))

    def _load_audio(self, audio_path: str | Path) -> torch.Tensor:
        import torchaudio

        try:
            wav, sr = torchaudio.load(str(audio_path))
        except Exception:
            try:
                import soundfile as sf

                data, sr = sf.read(str(audio_path), always_2d=True, dtype="float32")
                wav = torch.from_numpy(data.T)
            except Exception:
                import librosa

                data, sr = librosa.load(str(audio_path), sr=None, mono=False)
                wav = torch.as_tensor(data, dtype=torch.float32)
                if wav.ndim == 1:
                    wav = wav.unsqueeze(0)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if int(sr) != self.sample_rate:
            wav = torchaudio.functional.resample(wav, int(sr), self.sample_rate)
        return wav.to(torch.float32)

    @torch.inference_mode()
    def encode_path(self, audio_path: str | Path, *, n_vq: int | None = None) -> dict[str, Any]:
        wav = self._load_audio(audio_path)
        wav_1d = wav.squeeze(0).to(device=self.device, dtype=self.torch_dtype)
        if hasattr(self.model, "batch_encode"):
            enc = self.model.batch_encode([wav_1d], num_quantizers=n_vq)
            audio_codes = enc.audio_codes
            lengths = enc.audio_codes_lengths
        else:
            x = wav.unsqueeze(0).to(device=self.device, dtype=self.torch_dtype)
            padding_mask = torch.ones(1, x.shape[-1], dtype=torch.bool, device=self.device)
            enc = self.model.encode(
                x,
                padding_mask=padding_mask,
                num_quantizers=n_vq,
                return_dict=True,
            )
            audio_codes = enc.audio_codes
            lengths = enc.audio_codes_lengths
        if audio_codes is None or lengths is None:
            raise RuntimeError("MOSS codec encode returned empty audio_codes.")
        length = int(lengths[0].item())
        codes = audio_codes[:, 0, :length].transpose(0, 1).contiguous().to(torch.long).cpu()
        return {
            "codes": codes,
            "num_frames": int(codes.shape[0]),
            "n_vq": int(codes.shape[1]),
            "sample_rate": self.sample_rate,
            "duration_sec": float(wav.shape[-1] / self.sample_rate),
        }

    @torch.inference_mode()
    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        codes = torch.as_tensor(codes, dtype=torch.long)
        if codes.ndim != 2:
            raise ValueError(f"codes must be (T, NQ), got {tuple(codes.shape)}")
        audio_codes = codes.transpose(0, 1).unsqueeze(1).contiguous().to(self.device)
        padding_mask = torch.ones(1, audio_codes.shape[-1], dtype=torch.bool, device=self.device)
        dec = self.model.decode(
            audio_codes,
            padding_mask=padding_mask,
            return_dict=True,
            chunk_duration=8,
        )
        if dec.audio is None or dec.audio_lengths is None:
            raise RuntimeError("MOSS codec decode returned empty audio.")
        length = int(dec.audio_lengths[0].item())
        return dec.audio[0, 0, :length].detach().to(torch.float32).cpu()

    def save_wav(self, path: str | Path, wav: torch.Tensor) -> None:
        import torchaudio

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(p), wav.view(1, -1), self.sample_rate)

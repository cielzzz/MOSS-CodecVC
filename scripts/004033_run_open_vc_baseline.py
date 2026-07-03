#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_JSONL = ROOT / "testset/validation/seedtts_vc_ver2_3_validation.jsonl"
DEFAULT_OUTPUT_ROOT = ROOT / "testset/outputs/open_vc_baselines"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run official open-source VC baselines on the fixed SeedTTS VC validation set."
    )
    ap.add_argument("--provider", required=True, choices=("seedvc_v1", "meanvc", "xvc"))
    ap.add_argument("--validation-jsonl", default=str(DEFAULT_VALIDATION_JSONL))
    ap.add_argument("--output-dir", default="", help="Defaults to testset/outputs/open_vc_baselines/<provider>.")
    ap.add_argument("--manifest-jsonl", default="", help="Defaults to <output-dir>/manifest.jsonl.")
    ap.add_argument("--mode", choices=("no_text", "all"), default="all")
    ap.add_argument("--max-cases", type=int, default=0, help="Maximum supported rows to run. 0 means no limit.")
    ap.add_argument("--case-id", action="append", default=[])
    ap.add_argument("--device", default="cuda:0", help="cuda:0, cuda:1, cpu, or an integer GPU id.")
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--overwrite-manifest", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--seed-vc-dir", default=str(ROOT / "external/seed-vc"))
    ap.add_argument("--seedvc-checkpoint", default="")
    ap.add_argument("--seedvc-config", default="")
    ap.add_argument("--seedvc-diffusion-steps", type=int, default=25)
    ap.add_argument("--seedvc-length-adjust", type=float, default=1.0)
    ap.add_argument("--seedvc-inference-cfg-rate", type=float, default=0.7)
    ap.add_argument("--seedvc-fp16", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--meanvc-dir", default=str(ROOT / "external/MeanVC"))
    ap.add_argument("--meanvc-model-config", default="")
    ap.add_argument("--meanvc-ckpt-path", default="")
    ap.add_argument("--meanvc-asr-ckpt-path", default="")
    ap.add_argument("--meanvc-sv-ckpt-path", default="")
    ap.add_argument("--meanvc-vocoder-ckpt-path", default="")
    ap.add_argument("--meanvc-chunk-size", type=int, default=20)
    ap.add_argument("--meanvc-steps", type=int, default=2)

    ap.add_argument("--xvc-dir", default=str(ROOT / "external/X-VC"))
    ap.add_argument("--xvc-config", default="")
    ap.add_argument("--xvc-ckpt", default="")
    ap.add_argument("--xvc-ema-load", action="store_true")
    ap.add_argument("--xvc-mask-target-condition", action="store_true")
    ap.add_argument("--xvc-latent-hop-length", type=int, default=1280)
    return ap.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return stem[:180] or "case"


def normalize_device(device: str) -> str:
    text = str(device or "").strip()
    if text.isdigit():
        return f"cuda:{text}"
    return text or "cuda:0"


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


@contextlib.contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    wanted = set(args.case_id or [])
    selected: list[dict[str, Any]] = []
    supported_count = 0
    for row in rows:
        case_id = str(row.get("case_id") or "")
        mode = str(row.get("mode") or "")
        if wanted and case_id not in wanted:
            continue
        if args.mode == "no_text" and mode != "no_text":
            continue
        if mode == "no_text":
            if args.max_cases > 0 and supported_count >= args.max_cases:
                continue
            supported_count += 1
        selected.append(row)
    return selected


class SeedVCV1Backend:
    name = "seedvc_v1"

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.seed_vc_dir = Path(args.seed_vc_dir).expanduser().resolve()
        if str(self.seed_vc_dir) not in sys.path:
            sys.path.insert(0, str(self.seed_vc_dir))
        with pushd(self.seed_vc_dir):
            import inference  # type: ignore

            self.inference = inference
            model_args = SimpleNamespace(
                fp16=bool(args.seedvc_fp16),
                f0_condition=False,
                auto_f0_adjust=False,
                semi_tone_shift=0,
                checkpoint=args.seedvc_checkpoint or None,
                config=args.seedvc_config or None,
            )
            self.loaded = inference.load_models(model_args)

    @staticmethod
    def _crossfade(chunk1, chunk2, overlap):
        import numpy as np

        fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
        fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
        if len(chunk2) < overlap:
            chunk2[:overlap] = chunk2[:overlap] * fade_in[: len(chunk2)] + (chunk1[-overlap:] * fade_out)[: len(chunk2)]
        else:
            chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
        return chunk2

    def convert(self, source_audio: str, timbre_ref_audio: str, output_wav: Path) -> dict[str, Any]:
        import librosa
        import numpy as np
        import torch
        import torchaudio

        model, semantic_fn, _f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = self.loaded
        device = self.inference.device
        sr = int(mel_fn_args["sampling_rate"])
        hop_length = 256
        max_context_window = sr // hop_length * 30
        overlap_frame_len = 16
        overlap_wave_len = overlap_frame_len * hop_length

        source = librosa.load(source_audio, sr=sr)[0]
        ref = librosa.load(timbre_ref_audio, sr=sr)[0]
        if source.size == 0:
            raise ValueError(f"empty source audio: {source_audio}")
        if ref.size == 0:
            raise ValueError(f"empty timbre reference audio: {timbre_ref_audio}")
        source_tensor = torch.tensor(source).unsqueeze(0).float().to(device)
        ref_tensor = torch.tensor(ref[: sr * 25]).unsqueeze(0).float().to(device)

        source_16k = torchaudio.functional.resample(source_tensor, sr, 16000)
        if source_16k.size(-1) <= 16000 * 30:
            source_semantic = semantic_fn(source_16k)
        else:
            semantic_chunks = []
            buffer = None
            traversed_time = 0
            overlapping_time = 5
            while traversed_time < source_16k.size(-1):
                if buffer is None:
                    chunk = source_16k[:, traversed_time : traversed_time + 16000 * 30]
                else:
                    chunk = torch.cat(
                        [buffer, source_16k[:, traversed_time : traversed_time + 16000 * (30 - overlapping_time)]],
                        dim=-1,
                    )
                chunk_semantic = semantic_fn(chunk)
                semantic_chunks.append(chunk_semantic if traversed_time == 0 else chunk_semantic[:, 50 * overlapping_time :])
                buffer = chunk[:, -16000 * overlapping_time :]
                traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
            source_semantic = torch.cat(semantic_chunks, dim=1)

        ref_16k = torchaudio.functional.resample(ref_tensor, sr, 16000)
        ref_semantic = semantic_fn(ref_16k)
        source_mel = mel_fn(source_tensor.float())
        ref_mel = mel_fn(ref_tensor.float())
        target_lengths = torch.LongTensor([int(source_mel.size(2) * self.args.seedvc_length_adjust)]).to(source_mel.device)
        ref_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
        feat = torchaudio.compliance.kaldi.fbank(ref_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        style = campplus_model(feat.unsqueeze(0))

        with torch.no_grad():
            cond, _, _, _, _ = model.length_regulator(source_semantic, ylens=target_lengths, n_quantizers=3, f0=None)
            prompt_condition, _, _, _, _ = model.length_regulator(ref_semantic, ylens=ref_lengths, n_quantizers=3, f0=None)
        max_source_window = max_context_window - ref_mel.size(2)
        if max_source_window <= overlap_frame_len + 1:
            raise ValueError("timbre reference is too long for Seed-VC context window")

        processed_frames = 0
        chunks = []
        previous_chunk = None
        start = time.time()
        while processed_frames < cond.size(1):
            chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
            is_last = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16 if self.inference.fp16 else torch.float32):
                vc_target = model.cfm.inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(ref_mel.device),
                    ref_mel,
                    style,
                    None,
                    int(self.args.seedvc_diffusion_steps),
                    inference_cfg_rate=float(self.args.seedvc_inference_cfg_rate),
                )
                vc_target = vc_target[:, :, ref_mel.size(-1) :]
                vc_wave = vocoder_fn(vc_target.float()).squeeze()[None, :]
            if processed_frames == 0:
                if is_last:
                    chunks.append(vc_wave[0].cpu().numpy())
                    break
                chunks.append(vc_wave[0, :-overlap_wave_len].cpu().numpy())
                previous_chunk = vc_wave[0, -overlap_wave_len:]
                processed_frames += vc_target.size(2) - overlap_frame_len
            elif is_last:
                chunks.append(self._crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len))
                break
            else:
                chunks.append(self._crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len))
                previous_chunk = vc_wave[0, -overlap_wave_len:]
                processed_frames += vc_target.size(2) - overlap_frame_len

        output = torch.tensor(np.concatenate(chunks))[None, :].float().cpu()
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_wav), output, sr)
        elapsed = time.time() - start
        return {
            "sample_rate": sr,
            "num_frames": int(output.size(-1)),
            "duration_sec": float(output.size(-1) / sr),
            "elapsed_sec": float(elapsed),
            "rtf": float(elapsed / max(1e-6, output.size(-1) / sr)),
        }


class MeanVCBackend:
    name = "meanvc"

    def __init__(self, args: argparse.Namespace):
        import torch

        self.args = args
        self.meanvc_dir = Path(args.meanvc_dir).expanduser().resolve()
        for import_path in (self.meanvc_dir, self.meanvc_dir / "src/infer"):
            if str(import_path) not in sys.path:
                sys.path.insert(0, str(import_path))
        with pushd(self.meanvc_dir):
            from src.infer.dit_kvcache import DiT  # type: ignore
            from src.infer.infer_ref import MelSpectrogramFeatures, extract_features_from_audio, inference  # type: ignore
            from src.model.utils import load_checkpoint  # type: ignore
            from src.runtime.speaker_verification.verification import init_model as init_sv_model  # type: ignore

            self.extract_features_from_audio = extract_features_from_audio
            self.infer_fn = inference
            self.device = normalize_device(args.device)
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                self.device = "cpu"

            model_config = Path(args.meanvc_model_config or self.meanvc_dir / "src/config/config_200ms.json")
            ckpt_path = Path(args.meanvc_ckpt_path or self.meanvc_dir / "src/ckpt/model_200ms.safetensors")
            asr_ckpt = Path(args.meanvc_asr_ckpt_path or self.meanvc_dir / "src/ckpt/fastu2++.pt")
            sv_ckpt = Path(args.meanvc_sv_ckpt_path or self.meanvc_dir / "src/runtime/speaker_verification/ckpt/wavlm_large_finetune.pth")
            vocoder_ckpt = Path(args.meanvc_vocoder_ckpt_path or self.meanvc_dir / "src/ckpt/vocos.pt")

            cfg = json.loads(model_config.read_text(encoding="utf-8"))
            self.model = DiT(**cfg["model"]).to(self.device)
            self.model = load_checkpoint(self.model, str(ckpt_path), device=self.device, use_ema=False)
            self.model = self.model.float().eval()
            self.vocoder_ckpt_path = str(vocoder_ckpt)
            self.asr_model = torch.jit.load(str(asr_ckpt)).to(self.device)
            self.sv_model = init_sv_model("wavlm_large", str(sv_ckpt)).to(self.device).eval()
            self.mel_extractor = MelSpectrogramFeatures(
                sample_rate=16000,
                n_fft=1024,
                win_size=640,
                hop_length=160,
                n_mels=80,
                fmin=0,
                fmax=8000,
                center=True,
            ).to(self.device)

    def convert(self, source_audio: str, timbre_ref_audio: str, output_wav: Path) -> dict[str, Any]:
        import torch
        import torchaudio

        start = time.time()
        with pushd(self.meanvc_dir):
            bn, spk_emb, prompt_mel = self.extract_features_from_audio(
                source_audio,
                timbre_ref_audio,
                self.asr_model,
                self.sv_model,
                self.mel_extractor,
                self.device,
            )
            # The official TorchScript Vocos object can keep complex-valued state
            # after decode under torch 2.5, so reload the small vocoder per item.
            vocos = torch.jit.load(self.vocoder_ckpt_path).to(self.device)
            mel, wav, infer_elapsed = self.infer_fn(
                self.model,
                vocos,
                bn,
                spk_emb,
                prompt_mel,
                int(self.args.meanvc_chunk_size),
                int(self.args.meanvc_steps),
                self.device,
            )
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_wav), wav.cpu(), 16000)
        elapsed = time.time() - start
        return {
            "sample_rate": 16000,
            "num_frames": int(wav.shape[-1]),
            "duration_sec": float(wav.shape[-1] / 16000),
            "elapsed_sec": float(elapsed),
            "infer_elapsed_sec": float(infer_elapsed),
            "rtf": float(infer_elapsed / max(1e-6, wav.shape[-1] / 16000)),
        }


class XVCBackend:
    name = "xvc"

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.xvc_dir = Path(args.xvc_dir).expanduser().resolve()
        if str(self.xvc_dir) not in sys.path:
            sys.path.insert(0, str(self.xvc_dir))
        with pushd(self.xvc_dir):
            from bins.infer_utils import load_pair_as_tensors, load_xvc, run_offline, to_numpy_audio  # type: ignore

            self.load_pair_as_tensors = load_pair_as_tensors
            self.run_offline = run_offline
            self.to_numpy_audio = to_numpy_audio
            config = str(Path(args.xvc_config or self.xvc_dir / "configs/xvc.yaml"))
            ckpt = str(Path(args.xvc_ckpt or self.xvc_dir / "ckpts/xvc.pt"))
            device = normalize_device(args.device)
            device_id = 0
            if device.startswith("cuda:"):
                device_id = int(device.split(":", 1)[1])
            self.cfg, self.model, self.device = load_xvc(config, ckpt, device_id, bool(args.xvc_ema_load))

    def convert(self, source_audio: str, timbre_ref_audio: str, output_wav: Path) -> dict[str, Any]:
        import soundfile as sf

        start = time.time()
        with pushd(self.xvc_dir):
            source_wav, target_wav, target_wav_cond = self.load_pair_as_tensors(
                source_wav_path=source_audio,
                target_wav_path=timbre_ref_audio,
                cfg=self.cfg,
                device=self.device,
                latent_hop_length=int(self.args.xvc_latent_hop_length),
                mask_target_condition=bool(self.args.xvc_mask_target_condition),
            )
            recon = self.run_offline(self.model, source_wav, target_wav, target_wav_cond)
            recon_np = self.to_numpy_audio(recon)
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav), recon_np, samplerate=int(self.cfg["sample_rate"]))
        elapsed = time.time() - start
        duration = float(recon_np.shape[-1]) / float(self.cfg["sample_rate"])
        return {
            "sample_rate": int(self.cfg["sample_rate"]),
            "num_frames": int(recon_np.shape[-1]),
            "duration_sec": duration,
            "elapsed_sec": float(elapsed),
            "rtf": float(elapsed / max(1e-6, duration)),
        }


def make_backend(args: argparse.Namespace):
    if args.provider == "seedvc_v1":
        return SeedVCV1Backend(args)
    if args.provider == "meanvc":
        return MeanVCBackend(args)
    if args.provider == "xvc":
        return XVCBackend(args)
    raise ValueError(args.provider)


def output_status_if_existing(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        import torchaudio

        info = torchaudio.info(str(path))
        return {
            "sample_rate": int(info.sample_rate),
            "num_frames": int(info.num_frames),
            "duration_sec": float(info.num_frames / max(1, info.sample_rate)),
            "reused": True,
        }
    except Exception:
        return {"reused": True}


def manifest_base(row: dict[str, Any], output_wav: Path, provider: str) -> dict[str, Any]:
    return {
        "case_id": row.get("case_id"),
        "mode": row.get("mode"),
        "cell": row.get("cell"),
        "source_audio": row.get("source_audio"),
        "timbre_ref_audio": row.get("timbre_ref_audio"),
        "text": row.get("text"),
        "content_ref_text": row.get("content_ref_text"),
        "output_wav": str(output_wav),
        "provider": provider,
    }


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))

    validation_jsonl = Path(args.validation_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (DEFAULT_OUTPUT_ROOT / args.provider).resolve()
    manifest = Path(args.manifest_jsonl).expanduser().resolve() if args.manifest_jsonl else output_dir / "manifest.jsonl"
    rows = select_rows(list(iter_jsonl(validation_jsonl)), args)
    if not rows:
        print("[open-vc] no rows selected", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists() and args.overwrite_manifest:
        manifest.unlink()

    supported_rows = [r for r in rows if str(r.get("mode") or "") == "no_text"]
    unsupported_rows = [r for r in rows if str(r.get("mode") or "") != "no_text"]
    print(
        f"[open-vc] provider={args.provider} selected={len(rows)} "
        f"supported_no_text={len(supported_rows)} unsupported={len(unsupported_rows)} "
        f"output_dir={output_dir} manifest={manifest}",
        flush=True,
    )
    if args.dry_run:
        return 0

    backend = make_backend(args)
    failures = 0
    ok = 0
    unsupported = 0
    skipped = 0
    for row in rows:
        case_id = str(row.get("case_id") or "")
        output_wav = output_dir / f"{safe_stem(case_id)}.wav"
        base = manifest_base(row, output_wav, args.provider)
        if str(row.get("mode") or "") != "no_text":
            base.update({"status": "unsupported_text_mode", "output_exists": False})
            append_jsonl(manifest, base)
            unsupported += 1
            continue
        existing = output_status_if_existing(output_wav) if args.skip_existing else None
        if existing:
            base.update(existing)
            base.update({"status": "skipped_exists", "output_exists": True})
            append_jsonl(manifest, base)
            skipped += 1
            print(f"[open-vc] skip existing {case_id}", flush=True)
            continue
        start = time.time()
        try:
            print(f"[open-vc] run {case_id}", flush=True)
            metrics = backend.convert(str(row.get("source_audio") or ""), str(row.get("timbre_ref_audio") or ""), output_wav)
            base.update(metrics)
            base.update({"status": "ok", "output_exists": output_wav.exists(), "elapsed_total_sec": round(time.time() - start, 3)})
            append_jsonl(manifest, base)
            ok += 1
        except Exception as exc:
            failures += 1
            base.update(
                {
                    "status": "failed",
                    "output_exists": output_wav.exists(),
                    "elapsed_total_sec": round(time.time() - start, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            append_jsonl(manifest, base)
            print(f"[open-vc] failed {case_id}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if args.fail_fast:
                break

    print(
        f"[open-vc] complete provider={args.provider} ok={ok} skipped={skipped} "
        f"unsupported={unsupported} failures={failures}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from moss_codecvc.audio import decode_latents
from moss_codecvc.config import deep_get, load_config
from moss_codecvc.moss_codec import MossCodec


@dataclass(frozen=True)
class SanityCase:
    case_id: str
    mode: str
    language: str
    relative_path: str
    expected_frames: int


DEFAULT_CASES = (
    SanityCase(
        case_id="no_text_zh_70",
        mode="no_text",
        language="zh",
        relative_path=(
            "trainset/zh45w_en22w_no_text/seedvc_targets/large_mixed_mode_full/audio/"
            "seedvc_v1/zh_slim_0173_734d0ce679/00049152.wav"
        ),
        expected_frames=70,
    ),
    SanityCase(
        case_id="no_text_en_72",
        mode="no_text",
        language="en",
        relative_path=(
            "trainset/zh45w_en22w_no_text/seedvc_targets/large_mixed_mode_full/audio/"
            "seedvc_v1/en_slim_0074_22da7b0f20/00426178.wav"
        ),
        expected_frames=72,
    ),
    SanityCase(
        case_id="text_en_73",
        mode="text",
        language="en",
        relative_path=(
            "trainset/zh3w_en3w_text_prosody_independent_timbre/seedvc_targets/"
            "en/en_slim_0001/000000_92190e3e3871.wav"
        ),
        expected_frames=73,
    ),
    SanityCase(
        case_id="text_zh_73",
        mode="text",
        language="zh",
        relative_path=(
            "trainset/zh3w_en3w_text_prosody_independent_timbre/seedvc_targets/"
            "zh/zh_slim_0004/007710_2a3ecd8b4f8d.wav"
        ),
        expected_frames=73,
    ),
    SanityCase(
        case_id="no_text_en_95",
        mode="no_text",
        language="en",
        relative_path=(
            "trainset/zh45w_en22w_no_text/seedvc_targets/large_mixed_mode_full/audio/"
            "seedvc_v1/en_slim_0026_71f80b7150/00252098.wav"
        ),
        expected_frames=95,
    ),
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["cases"]
    lines = [
        "# ver3.1 Step 1 decode_latents consistency",
        "",
        f"- Status: **{payload['status']}**",
        f"- Generated: `{payload['generated_at_utc']}`",
        f"- Checkpoint: `{payload['checkpoint']['path']}`",
        f"- Latent contract: `(B, {payload['latent_dim']}, T)` at `{payload['frame_rate_hz']}` Hz",
        f"- Decoder upsample: `{payload['decoder_upsample']}` samples per latent frame",
        "",
        "| Case | Mode | Lang | Frames | Samples | Exact | Max abs diff |",
        "|---|---|---|---:|---:|:---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['mode']} | {row['language']} | "
            f"{row['latent_frames']} | {row['waveform_samples']} | "
            f"{'Y' if row['exact_equal'] else 'N'} | {row['max_abs_diff']:.9g} |"
        )
    batch = payload["variable_length_batch"]
    lines.extend(
        [
            "",
            "## Variable-length batch check",
            "",
            f"- Cases: `{', '.join(batch['case_ids'])}`",
            f"- Exact full padded tensor equality: `{batch['exact_equal']}`",
            f"- Max absolute difference: `{batch['max_abs_diff']:.9g}`",
            f"- Output lengths: `{batch['waveform_lengths']}`",
            "",
            "## Decision",
            "",
            "The public codes decoder and direct decoder-domain zq path are bit-exact "
            "for all five fixed v1 targets and the mixed-length batch. The thin "
            "`decode_latents()` API is accepted for DDLFM Step 1.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    tmp.replace(path)


def save_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Write a mono WAV without depending on torchaudio's TorchCodec backend."""

    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    samples = waveform.detach().to(torch.float32).cpu().numpy()
    sf.write(str(path), samples, sample_rate, subtype="FLOAT")


def checkpoint_provenance(codec_path: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for name in (
        "config.json",
        "configuration_moss_audio_tokenizer.py",
        "modeling_moss_audio_tokenizer.py",
        "model.safetensors.index.json",
    ):
        path = codec_path / name
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint provenance file missing: {path}")
        files[name] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    shards = []
    for path in sorted(codec_path.glob("model-*-of-*.safetensors")):
        shards.append({"name": path.name, "size_bytes": path.stat().st_size})
    if not shards:
        raise FileNotFoundError(f"no checkpoint weight shards found under {codec_path}")
    return {"path": str(codec_path.resolve()), "files": files, "weight_shards": shards}


@torch.inference_mode()
def encode_case(codec: MossCodec, case: SanityCase) -> dict[str, Any]:
    audio_path = ROOT / case.relative_path
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    wav = codec._load_audio(audio_path)
    input_values = wav.unsqueeze(0).to(device=codec.device, dtype=codec.torch_dtype)
    padding_mask = torch.ones(
        1,
        input_values.shape[-1],
        dtype=torch.bool,
        device=codec.device,
    )
    encoded = codec.model.encode(
        input_values,
        padding_mask=padding_mask,
        num_quantizers=32,
        return_dict=True,
    )
    if encoded.audio_codes is None or encoded.audio_codes_lengths is None:
        raise RuntimeError(f"empty encode output for {case.case_id}")

    latent_frames = int(encoded.audio_codes_lengths[0].item())
    if latent_frames != case.expected_frames:
        raise AssertionError(
            f"{case.case_id}: expected {case.expected_frames} frames, got {latent_frames}"
        )
    codes = encoded.audio_codes[:, :, :latent_frames].contiguous()
    frame_mask = torch.ones(1, latent_frames, dtype=torch.bool, device=codec.device)
    standard = codec.model.decode(
        codes,
        padding_mask=frame_mask,
        return_dict=True,
        chunk_duration=None,
    )
    if standard.audio is None or standard.audio_lengths is None:
        raise RuntimeError(f"empty standard decode output for {case.case_id}")

    zq = codec.model.quantizer.decode_codes(codes)
    zq_lengths = torch.tensor([latent_frames], dtype=torch.long, device=zq.device)
    direct_audio, direct_lengths = decode_latents(codec.model, zq, zq_lengths)

    max_abs_diff = float((standard.audio - direct_audio).abs().max().item())
    exact_equal = bool(torch.equal(standard.audio, direct_audio))
    lengths_equal = bool(torch.equal(standard.audio_lengths, direct_lengths))
    expected_samples = latent_frames * int(codec.model.downsample_rate)
    output_samples = int(direct_lengths[0].item())
    finite_zq = bool(torch.all(torch.isfinite(zq)))
    finite_audio = bool(torch.all(torch.isfinite(direct_audio)))
    if not exact_equal or max_abs_diff != 0.0:
        raise AssertionError(f"{case.case_id}: direct zq waveform is not bit-exact")
    if not lengths_equal or output_samples != expected_samples:
        raise AssertionError(
            f"{case.case_id}: decoder length mismatch, standard={standard.audio_lengths.tolist()} "
            f"direct={direct_lengths.tolist()} expected={expected_samples}"
        )
    if not finite_zq or not finite_audio:
        raise AssertionError(f"{case.case_id}: non-finite zq or decoded waveform")

    return {
        "case": case,
        "audio_path": audio_path,
        "audio_sha256": sha256_file(audio_path),
        "input_samples_24k": int(wav.shape[-1]),
        "codes": codes,
        "zq": zq,
        "standard_audio": standard.audio,
        "direct_audio": direct_audio,
        "direct_lengths": direct_lengths,
        "metrics": {
            "case_id": case.case_id,
            "mode": case.mode,
            "language": case.language,
            "audio_path": str(audio_path.resolve()),
            "audio_sha256": sha256_file(audio_path),
            "input_samples_24k": int(wav.shape[-1]),
            "latent_frames": latent_frames,
            "latent_dim": int(zq.shape[1]),
            "waveform_samples": output_samples,
            "exact_equal": exact_equal,
            "lengths_equal": lengths_equal,
            "max_abs_diff": max_abs_diff,
            "finite_zq": finite_zq,
            "finite_waveform": finite_audio,
        },
    }


@torch.inference_mode()
def check_variable_length_batch(codec: MossCodec, encoded_cases: list[dict[str, Any]]) -> dict[str, Any]:
    selected = encoded_cases[:2]
    lengths = torch.tensor(
        [int(item["codes"].shape[-1]) for item in selected],
        dtype=torch.long,
        device=codec.device,
    )
    max_frames = int(lengths.max().item())
    n_quantizers = int(selected[0]["codes"].shape[0])
    codes = torch.zeros(
        n_quantizers,
        len(selected),
        max_frames,
        dtype=torch.long,
        device=codec.device,
    )
    for index, item in enumerate(selected):
        item_codes = item["codes"][:, 0]
        codes[:, index, : item_codes.shape[-1]] = item_codes
    frame_mask = torch.arange(max_frames, device=codec.device).unsqueeze(0) < lengths.unsqueeze(1)
    standard = codec.model.decode(
        codes,
        padding_mask=frame_mask,
        return_dict=True,
        chunk_duration=None,
    )
    if standard.audio is None or standard.audio_lengths is None:
        raise RuntimeError("empty variable-length standard decode output")
    zq = codec.model.quantizer.decode_codes(codes)
    direct_audio, direct_lengths = decode_latents(codec.model, zq, lengths)
    exact_equal = bool(torch.equal(standard.audio, direct_audio))
    max_abs_diff = float((standard.audio - direct_audio).abs().max().item())
    if not exact_equal or max_abs_diff != 0.0 or not torch.equal(standard.audio_lengths, direct_lengths):
        raise AssertionError("variable-length batch direct zq decode is not bit-exact")
    expected_lengths = lengths * int(codec.model.downsample_rate)
    if not torch.equal(direct_lengths, expected_lengths):
        raise AssertionError(
            f"variable-length batch lengths mismatch: got={direct_lengths.tolist()} "
            f"expected={expected_lengths.tolist()}"
        )
    return {
        "case_ids": [item["case"].case_id for item in selected],
        "latent_lengths": lengths.tolist(),
        "waveform_lengths": direct_lengths.tolist(),
        "exact_equal": exact_equal,
        "max_abs_diff": max_abs_diff,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bit-exact zq -> MOSS decoder consistency smoke")
    parser.add_argument("--config", default=str(ROOT / "configs/remote_full.yaml"))
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "testset/outputs/ver3_1_step1_decode_latents_sanity_20260715"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float32", choices=("float32",))
    parser.add_argument("--no-save-wavs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    codec_path = Path(str(deep_get(config, "moss.codec_path"))).expanduser().resolve()
    moss_root = deep_get(config, "moss.root")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    codec = MossCodec(
        codec_path,
        moss_root=moss_root,
        device=args.device,
        dtype=args.dtype,
    )
    codec.model.eval()
    codec.model.requires_grad_(False)

    encoded_cases: list[dict[str, Any]] = []
    for case in DEFAULT_CASES:
        print(f"[decode-latents-sanity] encoding {case.case_id}", flush=True)
        result = encode_case(codec, case)
        encoded_cases.append(result)
        if not args.no_save_wavs:
            length = int(result["direct_lengths"][0].item())
            save_wav(
                output_dir / f"{case.case_id}.direct_zq.wav",
                result["direct_audio"][0, 0, :length].detach().cpu(),
                codec.sample_rate,
            )
        print(
            f"[decode-latents-sanity] PASS {case.case_id} "
            f"frames={result['metrics']['latent_frames']} "
            f"samples={result['metrics']['waveform_samples']} max_abs=0",
            flush=True,
        )

    variable_batch = check_variable_length_batch(codec, encoded_cases)
    frame_rate = float(codec.model.sampling_rate) / float(codec.model.downsample_rate)
    latent_dims = {int(item["metrics"]["latent_dim"]) for item in encoded_cases}
    if latent_dims != {768} or frame_rate != 12.5:
        raise AssertionError(f"unexpected latent contract: dims={latent_dims}, frame_rate={frame_rate}")

    payload = {
        "schema_version": 1,
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(args.config).expanduser().resolve()),
        "checkpoint": checkpoint_provenance(codec_path),
        "sample_rate": int(codec.model.sampling_rate),
        "decoder_upsample": int(codec.model.downsample_rate),
        "frame_rate_hz": frame_rate,
        "latent_dim": 768,
        "num_quantizers": 32,
        "cases": [item["metrics"] for item in encoded_cases],
        "variable_length_batch": variable_batch,
    }
    write_json(output_dir / "metrics.json", payload)
    write_report(output_dir / "REPORT.md", payload)
    write_json(output_dir / "COMPLETED.json", {"status": "PASS", "metrics": "metrics.json"})
    print(f"[decode-latents-sanity] COMPLETE output={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

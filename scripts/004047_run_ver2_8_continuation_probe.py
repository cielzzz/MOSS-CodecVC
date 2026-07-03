#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
import torchaudio
from transformers import GenerationConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_infer_module():
    path = ROOT / "scripts" / "003001_infer_moss_codecvc.py"
    spec = importlib.util.spec_from_file_location("moss_codecvc_infer_003001", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Ver2.8 D5 continuation probe for no-text badcases.")
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--base-model-path", default=None)
    ap.add_argument("--source-audio", required=True)
    ap.add_argument("--timbre-ref-audio", required=True)
    ap.add_argument("--text", default="<NO_TEXT>")
    ap.add_argument("--language", default=None)
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--disable-mode-token", action="store_true")
    ap.add_argument("--timbre-side-only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--prefix-ratio", type=float, default=0.50)
    ap.add_argument("--output-wav", required=True)
    ap.add_argument("--output-generated-codec", default="")
    ap.add_argument("--audio-segment-policy", choices=("all", "first", "longest"), default="all")
    ap.add_argument("--n-vq", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--min-new-tokens", type=int, default=0)
    ap.add_argument("--min-audio-tokens", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--audio-temperature", type=float, default=None)
    ap.add_argument("--audio-top-p", type=float, default=None)
    ap.add_argument("--audio-top-k", type=int, default=None)
    ap.add_argument("--audio-repetition-penalty", type=float, default=None)
    ap.add_argument("--disable-source-semantic-memory", action="store_true")
    ap.add_argument("--source-semantic-feature-path", default="")
    ap.add_argument("--source-semantic-model-name-or-path", default="")
    ap.add_argument("--source-semantic-cache-dir", default="")
    ap.add_argument("--source-semantic-local-files-only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--source-semantic-layer", type=int, default=9)
    ap.add_argument("--source-semantic-device", default="same")
    ap.add_argument("--source-semantic-dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    ap.add_argument("--source-semantic-downsample-stride", type=int, default=1)
    ap.add_argument("--source-semantic-monotonic-bias-strength", type=float, default=None)
    ap.add_argument("--source-semantic-monotonic-bias-width", type=float, default=None)
    ap.add_argument("--source-semantic-position-scale", type=float, default=None)
    ap.add_argument("--source-semantic-attention-debug-dir", default="")
    ap.add_argument("--source-semantic-attention-debug-max-tokens", type=int, default=2048)
    ap.add_argument("--debug-generation-structure", action="store_true")
    return ap.parse_args()


def main() -> int:
    mod = load_infer_module()
    args = parse_args()
    cfg = mod.load_config(args.config)
    moss_root = mod.deep_get(cfg, "moss.root")
    codec_path = mod.deep_get(cfg, "moss.codec_path")
    n_vq = args.n_vq or int(mod.deep_get(cfg, "moss.default_n_vq", 32))
    mod.ensure_moss_on_path(moss_root)
    mod.patch_torchaudio_load_with_soundfile_fallback()

    from moss_codecvc.models import MossCodecVCTimbreMemoryWrapper
    from moss_codecvc.models.moss_codecvc_wrapper import normalize_source_content_memory_type
    from moss_codecvc.modes import VC_MODE_NO_TEXT, apply_vc_mode_token
    from moss_codecvc.roles import count_roles, infer_prompt_role_ids_from_audio_spans
    from moss_tts_delay.modeling_moss_tts import MossTTSDelayModel
    from moss_tts_delay.processing_moss_tts import MossTTSDelayProcessor
    from peft import PeftConfig, PeftModel

    device_arg = mod.normalize_device_arg(args.device)
    device = torch.device(device_arg if torch.cuda.is_available() or not device_arg.startswith("cuda") else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model_path = Path(args.model_path)
    is_lora_adapter = (model_path / "adapter_config.json").exists()
    peft_model_path = mod.prepare_peft_adapter_dir_for_inference(model_path) if is_lora_adapter else model_path
    if is_lora_adapter:
        peft_cfg = PeftConfig.from_pretrained(str(peft_model_path))
        base_model_path = args.base_model_path or peft_cfg.base_model_name_or_path
        if not base_model_path:
            raise ValueError("--base-model-path is required for adapter-only checkpoints.")
        processor_path = base_model_path
        model_load_path = base_model_path
    else:
        processor_path = args.model_path
        model_load_path = args.model_path

    processor = MossTTSDelayProcessor.from_pretrained(
        processor_path,
        codec_path=codec_path,
        trust_remote_code=True,
    )
    processor.audio_tokenizer.to(device)
    model = MossTTSDelayModel.from_pretrained(
        model_load_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if not hasattr(model, "prepare_inputs_for_generation"):
        model.prepare_inputs_for_generation = lambda *a, **kw: kw
    if not hasattr(model, "generation_config"):
        model.generation_config = GenerationConfig.from_model_config(model.config)
    if is_lora_adapter:
        print(f"[d5] loading PEFT adapter from {peft_model_path}")
        model = PeftModel.from_pretrained(model, str(peft_model_path))
    use_timbre_memory = (
        is_lora_adapter
        and (model_path / "timbre_memory_config.json").exists()
        and (model_path / "timbre_memory_adapter.pt").exists()
    )
    if args.timbre_side_only and not use_timbre_memory:
        raise ValueError("--timbre-side-only requires timbre memory.")
    if use_timbre_memory:
        model = MossCodecVCTimbreMemoryWrapper.from_pretrained_timbre_memory(
            model,
            model_path,
            map_location="cpu",
        )
    model = model.to(device).eval()
    mod.apply_source_semantic_debug_overrides(
        model,
        position_scale=args.source_semantic_position_scale,
        monotonic_bias_strength=args.source_semantic_monotonic_bias_strength,
        monotonic_bias_width=args.source_semantic_monotonic_bias_width,
    )

    source_codes, timbre_codes = processor.encode_audios_from_path(
        [args.source_audio, args.timbre_ref_audio],
        n_vq=n_vq,
    )
    prefix_len = max(1, min(int(source_codes.shape[0]) - 1, int(math.ceil(float(source_codes.shape[0]) * args.prefix_ratio))))
    prefix_codes = source_codes[:prefix_len].contiguous()
    remaining_source_frames = max(1, int(source_codes.shape[0]) - prefix_len)

    instruction = args.instruction or (
        mod.deep_get(cfg, "instruction.no_text")
        or mod.deep_get(cfg, "instruction.prosody_no_timbre")
        or mod.deep_get(cfg, "instruction.default")
    )
    instruction = apply_vc_mode_token(instruction, VC_MODE_NO_TEXT, enabled=not args.disable_mode_token)
    prompt_references = [source_codes] if args.timbre_side_only else [source_codes, timbre_codes]
    user_message = processor.build_user_message(
        text=args.text,
        reference=prompt_references,
        instruction=instruction,
        tokens=int(source_codes.shape[0]),
        language=args.language,
        quality="high",
    )
    assistant_prefix = processor.build_assistant_message(audio_codes_list=[prefix_codes])
    inputs = processor([[user_message, assistant_prefix]], mode="continuation", n_vq=n_vq)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    config_max_new_tokens = int(mod.deep_get(cfg, "inference.max_new_tokens", 2048))
    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = min(config_max_new_tokens, remaining_source_frames + int(n_vq) + 16)
    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "text_temperature": args.temperature if args.temperature is not None else float(mod.deep_get(cfg, "inference.temperature", 0.8)),
        "text_top_p": args.top_p if args.top_p is not None else float(mod.deep_get(cfg, "inference.top_p", 0.9)),
        "text_top_k": args.top_k if args.top_k is not None else int(mod.deep_get(cfg, "inference.top_k", 50)),
        "audio_temperature": args.audio_temperature if args.audio_temperature is not None else float(mod.deep_get(cfg, "inference.audio_temperature", 1.7)),
        "audio_top_p": args.audio_top_p if args.audio_top_p is not None else float(mod.deep_get(cfg, "inference.audio_top_p", 0.8)),
        "audio_top_k": args.audio_top_k if args.audio_top_k is not None else int(mod.deep_get(cfg, "inference.audio_top_k", 25)),
        "audio_repetition_penalty": args.audio_repetition_penalty if args.audio_repetition_penalty is not None else float(mod.deep_get(cfg, "inference.audio_repetition_penalty", 1.0)),
    }
    if int(args.min_new_tokens) > 0:
        gen_kwargs["min_new_tokens"] = int(args.min_new_tokens)
    if int(args.min_audio_tokens) > 0:
        gen_kwargs["min_audio_tokens"] = int(args.min_audio_tokens)

    print(
        "[d5] encoded codec shapes: "
        f"source={tuple(source_codes.shape)} prefix={tuple(prefix_codes.shape)} "
        f"remaining={remaining_source_frames} timbre_ref={tuple(timbre_codes.shape)} "
        f"prompt={tuple(inputs['input_ids'].shape)}"
    )
    print(
        "[d5] generation budget "
        f"max_new_tokens={gen_kwargs['max_new_tokens']} min_new_tokens={gen_kwargs.get('min_new_tokens', 0)} "
        f"min_audio_tokens={gen_kwargs.get('min_audio_tokens', 0)}"
    )

    if use_timbre_memory:
        mode_id = int(getattr(model, "MODE_TO_ID", {}).get(VC_MODE_NO_TEXT, 0))
        if mode_id > 0:
            gen_kwargs["vc_mode_id"] = torch.tensor([mode_id], dtype=torch.long, device=device)
        timbre_codes_for_memory = torch.as_tensor(timbre_codes, dtype=torch.long, device=device).unsqueeze(0)
        gen_kwargs["timbre_ref_codes"] = timbre_codes_for_memory
        gen_kwargs["timbre_ref_mask"] = torch.ones(timbre_codes_for_memory.shape[:2], dtype=torch.bool, device=device)
        gen_kwargs["timbre_ref_speaker_audio_path"] = [args.timbre_ref_audio]
        if getattr(model.timbre_memory_config, "use_role_routing", False):
            prompt_role_ids = infer_prompt_role_ids_from_audio_spans(
                inputs["input_ids"],
                audio_pad_code=int(model.config.audio_pad_code),
            )
            print(f"[d5] inferred prompt role counts={count_roles(prompt_role_ids).as_dict()}")

        source_content_memory_type = normalize_source_content_memory_type(
            str(getattr(model.timbre_memory_config, "source_content_memory_type", "hubert_continuous") or "hubert_continuous")
        )
        if getattr(model, "source_semantic_memory_encoder", None) is not None and not args.disable_source_semantic_memory:
            if mod.is_continuous_source_memory_type(source_content_memory_type):
                if args.source_semantic_feature_path:
                    source_semantic_features = mod.load_source_semantic_feature_tensor(args.source_semantic_feature_path)
                    origin = args.source_semantic_feature_path
                else:
                    semantic_device = str(device) if args.source_semantic_device == "same" else mod.normalize_device_arg(args.source_semantic_device)
                    semantic_model_name = mod.resolve_source_semantic_model_name_for_memory(
                        source_content_memory_type,
                        args.source_semantic_model_name_or_path or mod.DEFAULT_HUBERT_MODEL,
                    )
                    source_semantic_features = mod.extract_source_semantic_features_online(
                        audio_path=args.source_audio,
                        model_name_or_path=semantic_model_name,
                        cache_dir=args.source_semantic_cache_dir or str(mod.DEFAULT_HUBERT_CACHE_DIR),
                        local_files_only=bool(args.source_semantic_local_files_only),
                        layer=int(args.source_semantic_layer),
                        device=semantic_device,
                        dtype_name=args.source_semantic_dtype,
                        downsample_stride=int(args.source_semantic_downsample_stride),
                    )
                    origin = f"online:{semantic_model_name}:layer{int(args.source_semantic_layer)}"
                source_semantic_features = source_semantic_features.unsqueeze(0).to(device=device)
                gen_kwargs["source_semantic_features"] = source_semantic_features
                gen_kwargs["source_semantic_features_mask"] = torch.ones(
                    source_semantic_features.shape[:2],
                    dtype=torch.bool,
                    device=device,
                )
                if mod.source_codec_residual_memory_enabled(model):
                    mod.add_source_ref_codes_for_memory(gen_kwargs, source_codes=source_codes, device=device)
                print(
                    "[d5] source semantic memory enabled: "
                    f"type={source_content_memory_type} features={tuple(source_semantic_features.shape)} origin={origin}"
                )
            else:
                raise ValueError(f"D5 probe only supports continuous source memory, got {source_content_memory_type}")

    if args.source_semantic_attention_debug_dir and hasattr(model, "clear_source_semantic_attention_capture"):
        model.clear_source_semantic_attention_capture()
        model.capture_source_semantic_attention = True
        model.source_semantic_attention_capture_max_tokens = int(args.source_semantic_attention_debug_max_tokens)

    with torch.inference_mode():
        output = model.generate(**inputs, **gen_kwargs)

    if args.source_semantic_attention_debug_dir:
        if hasattr(model, "capture_source_semantic_attention"):
            model.capture_source_semantic_attention = False
        mod.save_source_semantic_attention_debug(model, args.source_semantic_attention_debug_dir)
    if args.debug_generation_structure:
        mod.print_generation_structure(output, processor=processor, config=model.config, n_vq=n_vq)

    generated_codec = None
    if args.output_generated_codec:
        segments = mod.extract_dedelayed_audio_segments(output, processor=processor, config=model.config)
        generated_codec = mod.select_codec_segments(segments, policy=args.audio_segment_policy)
        if generated_codec is not None:
            out_codec = Path(args.output_generated_codec)
            out_codec.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "generated_codec": generated_codec.cpu(),
                    "source_codec": torch.as_tensor(source_codes, dtype=torch.long).cpu(),
                    "prefix_codec": torch.as_tensor(prefix_codes, dtype=torch.long).cpu(),
                    "ref_codec": torch.as_tensor(timbre_codes, dtype=torch.long).cpu(),
                    "n_vq": int(n_vq),
                    "prefix_ratio": float(args.prefix_ratio),
                    "prefix_len": int(prefix_len),
                    "remaining_source_frames": int(remaining_source_frames),
                    "audio_segment_policy": args.audio_segment_policy,
                },
                out_codec,
            )
            print(f"[d5] wrote generated codec -> {out_codec}")

    messages = processor.decode(output)
    wavs = []
    for message in messages:
        if message is None:
            continue
        for cur_wav in message.to_dict().get("audio_codes_list", []):
            if torch.is_tensor(cur_wav):
                wavs.append(cur_wav)
    if not wavs:
        raise RuntimeError("No audio waveform decoded from continuation output.")
    if len(wavs) > 1:
        lengths = [int(item.reshape(-1).numel()) for item in wavs]
        print(f"[d5] decoded_audio_segments={len(wavs)} sample_lengths={lengths} policy={args.audio_segment_policy}")
        if args.audio_segment_policy == "first":
            wavs = [wavs[0]]
        elif args.audio_segment_policy == "longest":
            wavs = [wavs[max(range(len(wavs)), key=lambda idx: lengths[idx])]]
    wav = torch.cat([w.reshape(-1).cpu() for w in wavs], dim=0)
    out_path = Path(args.output_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), wav.view(1, -1), int(processor.model_config.sampling_rate))
    print(f"[d5] wrote {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

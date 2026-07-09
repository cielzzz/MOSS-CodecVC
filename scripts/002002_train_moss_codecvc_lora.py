#!/usr/bin/env python
from __future__ import annotations

import argparse
from array import array
from bisect import bisect_right
from dataclasses import asdict, dataclass
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, get_scheduler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.config import deep_get, load_config
from moss_codecvc.roles import REF_CODEC, SOURCE_CODEC, TARGET_CODEC, count_roles
from moss_codecvc.models import (
    MossCodecVCTimbreMemoryWrapper,
    MossCodecVCTimbreSFTDataset,
    TimbreMemoryConfig,
)


LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LoRA SFT for MOSS-CodecVC on MossTTSDelay.")
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--codec-path", default=None)
    ap.add_argument("--train-jsonl", default=None)
    ap.add_argument(
        "--train-jsonl-spec",
        default=None,
        help=(
            "Optional comma/newline separated training sources. Each item is "
            "PATH[::repeat=N][::max_rows=N], e.g. no_text.jsonl::repeat=1,text.jsonl::repeat=3."
        ),
    )
    ap.add_argument(
        "--eval-jsonl",
        default=None,
        help="Optional train-ready JSONL used for teacher-forced validation loss.",
    )
    ap.add_argument(
        "--eval-jsonl-spec",
        default=None,
        help=(
            "Optional comma/newline separated eval sources using the same syntax as "
            "--train-jsonl-spec. Repeats are honored only when explicitly specified."
        ),
    )
    ap.add_argument("--eval-seen-jsonl", default=None, help="Optional seen-valid JSONL for teacher-forced eval.")
    ap.add_argument(
        "--eval-seen-jsonl-spec",
        default=None,
        help="Optional seen-valid eval spec using the same syntax as --train-jsonl-spec.",
    )
    ap.add_argument("--eval-unseen-jsonl", default=None, help="Optional unseen-valid JSONL for teacher-forced eval.")
    ap.add_argument(
        "--eval-unseen-jsonl-spec",
        default=None,
        help="Optional unseen-valid eval spec using the same syntax as --train-jsonl-spec.",
    )
    ap.add_argument(
        "--eval-steps",
        type=int,
        default=0,
        help="Run eval every N optimizer steps. 0 means run eval at checkpoint save steps and final.",
    )
    ap.add_argument(
        "--eval-max-batches",
        type=int,
        default=0,
        help="Maximum eval batches per eval call. 0 means the full eval loader.",
    )
    ap.add_argument("--eval-num-workers", type=int, default=0)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument(
        "--lr-scheduler-type",
        default=os.environ.get("LR_SCHEDULER_TYPE", "cosine"),
        help="Transformers scheduler type, e.g. cosine, constant, or constant_with_warmup.",
    )
    ap.add_argument("--num-epochs", type=int, default=1)
    ap.add_argument("--max-train-steps", type=int, default=0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--logging-steps", type=int, default=1)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    ap.add_argument("--attn-implementation", default="auto")
    ap.add_argument("--n-vq", type=int, default=None)
    ap.add_argument("--channelwise-loss-weight", default="1,32")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--jsonl-index-path", default=None)
    ap.add_argument("--rebuild-jsonl-index", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--resume-adapter-path", default="")
    ap.add_argument("--trainable-lora-modules", choices=("all", "mlp", "mlp_plus_o"), default="all")
    ap.add_argument("--lm-heads-mode", choices=("none", "audio", "all"), default="none")
    ap.add_argument("--version", choices=("ver1.6", "ver2"), default=None)
    ap.add_argument("--use-timbre-memory", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--timbre-memory-tokens", type=int, default=None)
    ap.add_argument("--timbre-adapter-layers", default=None)
    ap.add_argument("--timbre-adapter-init-gate", type=float, default=None)
    ap.add_argument("--timbre-encoder-type", choices=("perceiver", "transformer", "conformer", "moe_conformer"), default=None)
    ap.add_argument("--timbre-encoder-layers", type=int, default=None)
    ap.add_argument("--timbre-conformer-kernel-size", type=int, default=None)
    ap.add_argument("--timbre-speaker-conditioning", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument(
        "--speaker-encoder-type",
        choices=("embedding_loader", "precomputed_ecapa", "speechbrain_ecapa", "seed_tts_eval_ecapa", "wavlm_sv"),
        default=None,
    )
    ap.add_argument("--speaker-encoder-path", default=None)
    ap.add_argument("--speaker-embedding-dim", type=int, default=None)
    ap.add_argument("--target-speaker-similarity-weight", type=float, default=None)
    ap.add_argument("--source-speaker-suppression-weight", type=float, default=None)
    ap.add_argument("--speaker-loss-margin", type=float, default=None)
    ap.add_argument("--speaker-loss-warmup-steps", type=int, default=None)
    ap.add_argument("--speaker-loss-warmup-weight", type=float, default=None)
    ap.add_argument("--speaker-loss-schedule", choices=("step", "cosine"), default=None)
    ap.add_argument("--ref-speaker-prompt-tokens", type=int, default=None)
    ap.add_argument("--ref-speaker-prompt-dropout", type=float, default=None)
    ap.add_argument("--ref-speaker-prompt-mode", choices=("memory", "slot", "both"), default=None)
    ap.add_argument(
        "--ref-speaker-prompt-token-source",
        choices=("speaker_mlp", "timbre_memory"),
        default=None,
        help="Soft-token source for ref speaker prompt slot. speaker_mlp is A1; timbre_memory reuses T_ref tokens for A3.",
    )
    ap.add_argument("--ref-speaker-prompt-slot", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--ref-speaker-prompt-slot-code", type=int, default=None)
    ap.add_argument("--ref-speaker-prompt-slot-pack-mode", choices=("pad", "audio_like"), default=None)
    ap.add_argument("--ref-speaker-prompt-output-norm", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--ref-speaker-prompt-output-scale", type=float, default=None)
    ap.add_argument("--ref-prompt-codec-permutation", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--ref-prompt-codec-permutation-min-seconds", type=float, default=None)
    ap.add_argument("--ref-prompt-codec-permutation-max-seconds", type=float, default=None)
    ap.add_argument("--ref-prompt-codec-permutation-frame-rate", type=float, default=None)
    ap.add_argument("--ref-prompt-codec-permutation-seed", type=int, default=None)
    ap.add_argument("--ref-prompt-codec-permutation-mode", choices=("shuffle", "contiguous", "block_shuffle"), default=None)
    ap.add_argument("--ref-prompt-codec-permutation-block-seconds", type=float, default=None)
    ap.add_argument(
        "--ref-speaker-prompt-lr-multiplier",
        type=float,
        default=1.0,
        help="LR multiplier for ref_speaker_prompt MLP parameters. 1.0 keeps them in the base group.",
    )
    ap.add_argument(
        "--target-front-ce-weight",
        type=float,
        default=None,
        help="CE multiplier for the first target audio frames. 1 disables A4 weighting.",
    )
    ap.add_argument(
        "--target-front-ce-seconds",
        type=float,
        default=None,
        help="Duration in seconds for A4 front-target CE upweight. 0 disables it.",
    )
    ap.add_argument(
        "--target-front-ce-frame-rate",
        type=float,
        default=None,
        help="Codec frame rate used to convert A4 seconds to target frames.",
    )
    ap.add_argument("--ref-speaker-adaln-weight", type=float, default=None)
    ap.add_argument("--speaker-infonce-weight", type=float, default=None)
    ap.add_argument("--speaker-infonce-temperature", type=float, default=None)
    ap.add_argument("--speaker-infonce-negative-pool-size", type=int, default=None)
    ap.add_argument("--speaker-infonce-negative-pool-seed", type=int, default=None)
    ap.add_argument("--speaker-condition-dropout", type=float, default=None)
    ap.add_argument("--enable-speaker-side-pathway", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--speaker-side-pathway-layers", default=None)
    ap.add_argument("--speaker-side-pathway-kv-bias", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--speaker-side-pathway-gate-init", type=float, default=None)
    ap.add_argument("--speaker-side-pathway-dropout", type=float, default=None)
    ap.add_argument("--enable-speaker-cross-attn", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--speaker-cross-attn-layers", default=None)
    ap.add_argument("--speaker-cross-attn-tokens", type=int, default=None)
    ap.add_argument("--speaker-cross-attn-gate-init", type=float, default=None)
    ap.add_argument("--speaker-cross-attn-dropout", type=float, default=None)
    ap.add_argument("--speaker-cross-attn-output-scale", type=float, default=None)
    ap.add_argument("--speaker-cross-attn-token-init-std", type=float, default=None)
    ap.add_argument("--speaker-cross-attn-alpha-warmup-steps", type=int, default=None)
    ap.add_argument("--speaker-cross-attn-source", choices=("vector", "sequence"), default=None)
    ap.add_argument("--speaker-cross-attn-seq-dim", type=int, default=None)
    ap.add_argument("--use-perturbed-source-prompt", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--enable-role-routing", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--enable-target-head-routing", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument(
        "--routing-gate-lr-multiplier",
        type=float,
        default=1.0,
        help="LR multiplier for learnable Ver2 routing gate logits. Gate logits are kept in fp32.",
    )
    ap.add_argument(
        "--content-ctc-head-lr-multiplier",
        type=float,
        default=1.0,
        help="LR multiplier for the auxiliary content_ctc_head parameters. 1.0 keeps them in the base optimizer group.",
    )
    ap.add_argument(
        "--timbre-adapter-gate-lr-multiplier",
        type=float,
        default=1.0,
        help="LR multiplier for TargetOnlyTimbreAdapter scalar gates. Gate logits are kept in fp32.",
    )
    ap.add_argument("--lambda-route", type=float, default=None)
    ap.add_argument("--lambda-prosody", type=float, default=None)
    ap.add_argument("--lambda-content", type=float, default=None)
    ap.add_argument("--prosody-f0-weight", type=float, default=None)
    ap.add_argument("--prosody-voiced-weight", type=float, default=None)
    ap.add_argument("--prosody-energy-weight", type=float, default=None)
    ap.add_argument("--prosody-pause-weight", type=float, default=None)
    ap.add_argument("--prosody-duration-weight", type=float, default=None)
    ap.add_argument("--prosody-normalize-f0", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--prosody-normalize-energy", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--content-embedding-dim", type=int, default=None)
    ap.add_argument(
        "--content-positive",
        choices=("source", "target", "source_or_target", "source_and_target", "average"),
        default=None,
        help="Positive semantic embedding target for content proxy loss.",
    )
    ap.add_argument("--content-embedding-weight", type=float, default=None)
    ap.add_argument("--content-ctc-weight", type=float, default=None)
    ap.add_argument("--content-ctc-vocab-size", type=int, default=None)
    ap.add_argument("--content-ctc-blank-id", type=int, default=None)
    ap.add_argument("--content-ctc-token-offset", type=int, default=None)
    ap.add_argument(
        "--allow-mixed-content-tokenizers",
        action="store_true",
        help=(
            "Allow mixed content CTC tokenizer/vocab metadata across train JSONL sources. "
            "This is unsafe for text/no_text mixed training and should only be used for debugging."
        ),
    )
    ap.add_argument("--content-token-vocab-size", type=int, default=None)
    ap.add_argument(
        "--content-token-weight",
        type=float,
        default=None,
        help="Internal weight for optional semantic/content token CE under lambda-content.",
    )
    ap.add_argument(
        "--content-source-codec-weight",
        type=float,
        default=None,
        help="Internal weight for optional source-codec content proxy under lambda-content.",
    )
    ap.add_argument(
        "--content-source-codec-codebooks",
        default=None,
        help="Comma-separated source RVQ codebooks used by source-codec content proxy, e.g. 0,1,2,3 or first_4.",
    )
    ap.add_argument("--semantic-loss-weight", type=float, default=None)
    ap.add_argument("--semantic-mode", choices=("discrete", "continuous"), default=None)
    ap.add_argument("--semantic-source", choices=("source", "target", "mode_aware", "auto"), default=None)
    ap.add_argument("--semantic-vocab-size", type=int, default=None)
    ap.add_argument("--semantic-feature-dim", type=int, default=None)
    ap.add_argument("--semantic-feature-loss-type", choices=("cosine", "mse"), default=None)
    ap.add_argument(
        "--progress-loss-weight",
        type=float,
        default=None,
        help="Weight for target-position monotonic progress-bin auxiliary loss.",
    )
    ap.add_argument(
        "--stop-loss-weight",
        type=float,
        default=None,
        help="Weight for target-position final-frame stop auxiliary loss.",
    )
    ap.add_argument("--progress-num-bins", type=int, default=None)
    ap.add_argument("--enable-source-semantic-memory", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--source-semantic-feature-dim", type=int, default=None)
    ap.add_argument("--source-semantic-adapter-layers", default=None)
    ap.add_argument("--source-semantic-no-text-gate", type=float, default=None)
    ap.add_argument("--source-semantic-text-gate", type=float, default=None)
    ap.add_argument("--source-semantic-learned-text-gate", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--source-semantic-progress-weight", type=float, default=None)
    ap.add_argument("--source-semantic-dropout", type=float, default=None)
    ap.add_argument("--source-semantic-init-gate", type=float, default=None)
    ap.add_argument(
        "--source-semantic-position-scale",
        type=float,
        default=None,
        help="Scale for fixed sinusoidal time position encoding added to SourceSemanticMemory. 0 disables it.",
    )
    ap.add_argument(
        "--source-semantic-monotonic-bias-strength",
        type=float,
        default=None,
        help="Strength of target-progress to source-progress attention logit bias. 0 disables it.",
    )
    ap.add_argument(
        "--source-semantic-monotonic-bias-width",
        type=float,
        default=None,
        help="Normalized width for SourceSemanticMemory monotonic attention bias.",
    )
    ap.add_argument(
        "--source-content-memory-type",
        choices=(
            "none",
            "hubert",
            "hubert_continuous",
            "continuous",
            "ssl_continuous",
            "asr_bnf",
            "asr_bnf_continuous",
            "bnf",
            "wavlm",
            "wavlm_bnf",
            "wavlm_bnf_continuous",
            "wavlm_continuous",
            "text_tokens",
            "semantic_units",
            "codec_bottleneck",
        ),
        default=None,
        help=(
            "Ver2.5 memory source for the existing SourceSemanticAdapter stack. "
            "hubert_continuous keeps the old path; wavlm_bnf_continuous/asr_bnf_continuous "
            "use continuous WavLM/ASR-BNF features; text_tokens uses content_token_ids; "
            "semantic_units uses source_semantic_units; codec_bottleneck uses source codec embeddings."
        ),
    )
    ap.add_argument("--source-content-vocab-size", type=int, default=None)
    ap.add_argument("--source-content-padding-id", type=int, default=None)
    ap.add_argument("--source-content-codec-bottleneck-dim", type=int, default=None)
    ap.add_argument(
        "--source-content-codec-codebooks",
        default=None,
        help="RVQ codebooks for codec_bottleneck memory, e.g. first_4 or 0,1,2,3.",
    )
    ap.add_argument("--source-content-dedup-units", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument(
        "--source-codec-residual-memory-weight",
        type=float,
        default=None,
        help="Append source codec bottleneck memory after continuous WavLM/ASR-BNF memory with this scale.",
    )
    ap.add_argument(
        "--source-codec-residual-memory-detach",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Detach source codec embeddings before the residual bottleneck memory encoder.",
    )
    ap.add_argument(
        "--timbre-side-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep S2/timbre reference out of the AR prompt and use it only through timbre memory/speaker conditioning.",
    )
    ap.add_argument("--ref-content-suppression-weight", type=float, default=None)
    ap.add_argument("--ref-content-suppression-margin", type=float, default=None)
    ap.add_argument(
        "--ref-content-suppression-source",
        choices=("auto", "prompt_hidden", "codec_embedding"),
        default=None,
        help="Negative representation for timbre-ref content suppression.",
    )
    ap.add_argument("--ref-content-suppression-detach-ref", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument(
        "--source-semantic-gate-lr-multiplier",
        type=float,
        default=10.0,
        help="LR multiplier for SourceSemanticAdapter scalar gates. Gate logits are kept in fp32.",
    )
    ap.add_argument(
        "--source-semantic-lr-multiplier",
        type=float,
        default=1.0,
        help=(
            "LR multiplier for non-gate SourceSemanticMemory parameters "
            "(source_semantic_memory_encoder, source_semantic_codec_residual_encoder, "
            "and source_semantic_layer_adapters)."
        ),
    )
    ap.add_argument(
        "--train-source-semantic-only",
        action="store_true",
        help=(
            "Train only SourceSemanticMemoryEncoder and SourceSemanticAdapter. "
            "Used for Ver2.5 adapter warmup before joint LoRA training."
        ),
    )
    ap.add_argument("--freeze-lora", action="store_true", help="Freeze LoRA parameters after loading/resume.")
    ap.add_argument("--freeze-role-routing", action="store_true", help="Freeze Ver2 role/head routing parameters.")
    ap.add_argument("--freeze-timbre-adapter", action="store_true", help="Freeze TTE/TTA and related timbre/prosody/content heads.")
    ap.add_argument("--prosody-memory-tokens", type=int, default=None)
    ap.add_argument("--source-prosody-encoder-type", choices=("perceiver", "transformer", "conformer", "moe_conformer"), default=None)
    ap.add_argument("--source-prosody-encoder-layers", type=int, default=None)
    ap.add_argument("--source-prosody-conv-kernel-size", type=int, default=None)
    ap.add_argument(
        "--source-prosody-no-text-gate",
        type=float,
        default=None,
        help="Batch gate for source-codec prosody memory in no-text mode.",
    )
    ap.add_argument(
        "--source-prosody-text-gate",
        type=float,
        default=None,
        help="Batch gate for source-codec prosody memory in text mode. Set near 0 to prevent source lexical leakage.",
    )
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--pack-only", action="store_true", help="Validate dataset packing and exit before loading model weights.")
    ap.add_argument("--eval-only", action="store_true", help="Load model/checkpoint, run eval loaders once, and exit.")
    return ap.parse_args()


def resolve_dtype(mixed_precision: str) -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def resolve_attn_implementation(requested: str, dtype: torch.dtype) -> str:
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "eager"
    if importlib.util.find_spec("flash_attn") is not None and dtype in {torch.float16, torch.bfloat16}:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    return "sdpa"


def parse_channelwise_loss_weight(spec: str | None, n_heads: int) -> list[float] | None:
    if spec is None:
        return None
    values = [float(item.strip()) for item in spec.split(",") if item.strip()]
    if not values:
        return None
    if len(values) == n_heads:
        return values
    if len(values) == 2 and n_heads > 1:
        text_weight, total_audio_weight = values
        return [text_weight] + [total_audio_weight / (n_heads - 1)] * (n_heads - 1)
    raise ValueError(f"channelwise loss expects 2 or {n_heads} values, got {len(values)}")


def _distributed_rank() -> int:
    for name in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        value = os.environ.get(name)
        if value not in (None, ""):
            try:
                return int(value)
            except ValueError:
                continue
    return 0


class JsonlIndexedRecords:
    """Map-style JSONL reader that keeps only byte offsets in memory.

    The 68w Ver2 train file is tens of GB. Loading every JSON object into a
    Python list can get the QZ worker SIGKILLed before training starts, so this
    reader scans line offsets once and parses only the requested row.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_rows: int = 0,
        index_path: str | Path | None = None,
        rebuild_index: bool = False,
        wait_timeout_secs: int = 7200,
    ) -> None:
        self.path = Path(path).expanduser()
        self.max_rows = int(max_rows)
        self.persist_index = self.max_rows <= 0
        default_index = Path(str(self.path) + ".offsets.u64")
        self.index_path = Path(index_path).expanduser() if index_path else default_index
        self.meta_path = Path(str(self.index_path) + ".json")
        self.rebuild_index = bool(rebuild_index)
        self.wait_timeout_secs = int(wait_timeout_secs)
        self._fh = None
        self._fh_pid = None
        self.offsets = self._load_or_build_offsets()
        if len(self.offsets) == 0:
            raise ValueError(f"No records found in {self.path}")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self.offsets)
        if index < 0 or index >= len(self.offsets):
            raise IndexError(index)
        handle = self._handle()
        handle.seek(int(self.offsets[index]))
        line = handle.readline()
        if not line:
            raise IndexError(f"Offset {int(self.offsets[index])} in {self.path} produced an empty line")
        return json.loads(line.decode("utf-8"))

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_fh"] = None
        state["_fh_pid"] = None
        return state

    def _handle(self):
        pid = os.getpid()
        if self._fh is None or self._fh.closed or self._fh_pid != pid:
            if self._fh is not None and not self._fh.closed:
                self._fh.close()
            self._fh = self.path.open("rb")
            self._fh_pid = pid
        return self._fh

    def _source_meta(self) -> dict[str, Any]:
        stat = self.path.stat()
        return {
            "source_path": str(self.path.resolve()),
            "source_size": int(stat.st_size),
            "source_mtime_ns": int(stat.st_mtime_ns),
        }

    def _load_valid_persisted_offsets(self) -> array | None:
        if not self.persist_index or self.rebuild_index:
            return None
        if not self.index_path.exists() or not self.meta_path.exists():
            return None
        try:
            with self.meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
            source_meta = self._source_meta()
            for key, value in source_meta.items():
                if meta.get(key) != value:
                    return None
            rows = int(meta.get("rows") or 0)
            if rows <= 0:
                return None
            offsets = array("Q")
            with self.index_path.open("rb") as handle:
                offsets.fromfile(handle, rows)
            if len(offsets) != rows:
                return None
            return offsets
        except Exception:
            return None

    def _wait_for_persisted_offsets(self) -> array | None:
        if not self.persist_index:
            return None
        started = time.time()
        while time.time() - started < self.wait_timeout_secs:
            offsets = self._load_valid_persisted_offsets()
            if offsets is not None:
                return offsets
            time.sleep(5.0)
        return None

    def _build_offsets(self) -> array:
        offsets = array("Q")
        print(f"[jsonl-index] scanning {self.path}", flush=True)
        with self.path.open("rb") as handle:
            while True:
                pos = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(pos)
                    if self.max_rows > 0 and len(offsets) >= self.max_rows:
                        break
                    if self.persist_index and len(offsets) % 100000 == 0:
                        print(f"[jsonl-index] rows={len(offsets)} offset={handle.tell()}", flush=True)
        print(f"[jsonl-index] rows={len(offsets)}", flush=True)
        return offsets

    def _persist_offsets(self, offsets: array) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_index = Path(str(self.index_path) + f".tmp.{os.getpid()}")
        tmp_meta = Path(str(self.meta_path) + f".tmp.{os.getpid()}")
        with tmp_index.open("wb") as handle:
            offsets.tofile(handle)
        meta = self._source_meta()
        meta.update(
            {
                "rows": int(len(offsets)),
                "index_path": str(self.index_path.resolve()),
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        with tmp_meta.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)
        os.replace(tmp_index, self.index_path)
        os.replace(tmp_meta, self.meta_path)
        print(f"[jsonl-index] wrote {self.index_path} rows={len(offsets)}", flush=True)

    def _load_or_build_offsets(self) -> array:
        offsets = self._load_valid_persisted_offsets()
        if offsets is not None:
            print(f"[jsonl-index] loaded {self.index_path} rows={len(offsets)}", flush=True)
            return offsets
        rank = _distributed_rank()
        if self.persist_index and rank != 0:
            print(f"[jsonl-index] rank={rank} waiting for rank0 index {self.index_path}", flush=True)
            offsets = self._wait_for_persisted_offsets()
            if offsets is not None:
                print(f"[jsonl-index] rank={rank} loaded {self.index_path} rows={len(offsets)}", flush=True)
                return offsets
            print(f"[jsonl-index] rank={rank} timed out waiting; building local index", flush=True)
        offsets = self._build_offsets()
        if self.persist_index:
            try:
                self._persist_offsets(offsets)
            except Exception as exc:
                print(f"[jsonl-index] WARNING: failed to persist index: {exc}", flush=True)
        return offsets


def load_records(
    path: str,
    max_rows: int,
    *,
    jsonl_index_path: str | None = None,
    rebuild_jsonl_index: bool = False,
) -> JsonlIndexedRecords:
    return JsonlIndexedRecords(
        path,
        max_rows=max_rows,
        index_path=jsonl_index_path,
        rebuild_index=rebuild_jsonl_index,
    )


class RepeatedRecords:
    def __init__(self, records, repeat: int = 1) -> None:
        self.records = records
        self.repeat = max(1, int(repeat))

    def __len__(self) -> int:
        return len(self.records) * self.repeat

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.records[index % len(self.records)]


class ConcatRecords:
    def __init__(self, sources: list[RepeatedRecords]) -> None:
        if not sources:
            raise ValueError("ConcatRecords requires at least one source")
        self.sources = sources
        self.cumulative: list[int] = []
        total = 0
        for source in sources:
            total += len(source)
            self.cumulative.append(total)
        if total <= 0:
            raise ValueError("ConcatRecords has no records")

    def __len__(self) -> int:
        return self.cumulative[-1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source_idx = bisect_right(self.cumulative, index)
        previous = 0 if source_idx == 0 else self.cumulative[source_idx - 1]
        return self.sources[source_idx][index - previous]


@dataclass(frozen=True)
class TrainJsonlSourceSpec:
    path: str
    repeat: int = 1
    max_rows: int = 0


def parse_train_jsonl_spec(spec: str | None) -> list[TrainJsonlSourceSpec]:
    if not spec:
        return []
    chunks: list[str] = []
    for line in str(spec).splitlines():
        chunks.extend(item.strip() for item in line.split(","))
    parsed: list[TrainJsonlSourceSpec] = []
    for chunk in chunks:
        if not chunk:
            continue
        parts = chunk.split("::")
        path = parts[0].strip()
        repeat = 1
        max_rows = 0
        for part in parts[1:]:
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"Invalid --train-jsonl-spec part {part!r} in {chunk!r}")
            key, value = part.split("=", 1)
            key = key.strip().lower().replace("-", "_")
            value = value.strip()
            if key == "repeat":
                repeat = int(value)
            elif key == "max_rows":
                max_rows = int(value)
            else:
                raise ValueError(f"Unknown --train-jsonl-spec key {key!r} in {chunk!r}")
        if not path:
            raise ValueError(f"Missing path in --train-jsonl-spec item {chunk!r}")
        if repeat <= 0:
            raise ValueError(f"repeat must be positive in --train-jsonl-spec item {chunk!r}")
        if max_rows < 0:
            raise ValueError(f"max_rows must be >= 0 in --train-jsonl-spec item {chunk!r}")
        parsed.append(TrainJsonlSourceSpec(path=path, repeat=repeat, max_rows=max_rows))
    return parsed


def load_training_records(args: argparse.Namespace):
    specs = parse_train_jsonl_spec(args.train_jsonl_spec)
    if specs:
        if args.jsonl_index_path:
            raise ValueError("--jsonl-index-path is only supported with a single --train-jsonl source")
        sources: list[RepeatedRecords] = []
        for idx, spec in enumerate(specs):
            source_max_rows = int(spec.max_rows) if int(spec.max_rows) > 0 else int(args.max_rows)
            records = load_records(
                spec.path,
                source_max_rows,
                jsonl_index_path=None,
                rebuild_jsonl_index=args.rebuild_jsonl_index,
            )
            repeated = RepeatedRecords(records, repeat=spec.repeat)
            sources.append(repeated)
            print(
                "[train-jsonl-spec] "
                f"source={idx} path={spec.path} rows={len(records)} repeat={spec.repeat} "
                f"effective_rows={len(repeated)} max_rows={source_max_rows}",
                flush=True,
            )
        combined = ConcatRecords(sources)
        print(f"[train-jsonl-spec] effective_total_rows={len(combined)} sources={len(sources)}", flush=True)
        return combined
    if not args.train_jsonl:
        raise ValueError("Either --train-jsonl or --train-jsonl-spec is required")
    return load_records(
        args.train_jsonl,
        args.max_rows,
        jsonl_index_path=args.jsonl_index_path,
        rebuild_jsonl_index=args.rebuild_jsonl_index,
    )


def load_eval_records_from_sources(
    *,
    label: str,
    jsonl_path: str | None,
    jsonl_spec: str | None,
    args: argparse.Namespace,
):
    if jsonl_path and jsonl_spec:
        if label == "eval":
            raise ValueError("Use only one of --eval-jsonl or --eval-jsonl-spec")
        raise ValueError(f"Use only one of --eval-{label}-jsonl or --eval-{label}-jsonl-spec")
    specs = parse_train_jsonl_spec(jsonl_spec)
    if specs:
        sources: list[RepeatedRecords] = []
        for idx, spec in enumerate(specs):
            records = load_records(
                spec.path,
                int(spec.max_rows),
                jsonl_index_path=None,
                rebuild_jsonl_index=args.rebuild_jsonl_index,
            )
            repeated = RepeatedRecords(records, repeat=spec.repeat)
            sources.append(repeated)
            print(
                f"[eval-{label}-jsonl-spec] "
                f"source={idx} path={spec.path} rows={len(records)} repeat={spec.repeat} "
                f"effective_rows={len(repeated)} max_rows={int(spec.max_rows)}",
                flush=True,
            )
        combined = ConcatRecords(sources)
        print(f"[eval-{label}-jsonl-spec] effective_total_rows={len(combined)} sources={len(sources)}", flush=True)
        return combined
    if jsonl_path:
        print(f"[eval-{label}-jsonl] path={jsonl_path}", flush=True)
        return load_records(
            jsonl_path,
            0,
            jsonl_index_path=None,
            rebuild_jsonl_index=args.rebuild_jsonl_index,
        )
    return None


def load_eval_records(args: argparse.Namespace):
    return load_eval_records_from_sources(
        label="eval",
        jsonl_path=args.eval_jsonl,
        jsonl_spec=args.eval_jsonl_spec,
        args=args,
    )


def load_named_eval_records(args: argparse.Namespace) -> list[tuple[str, Any]]:
    named: list[tuple[str, Any]] = []
    legacy = load_eval_records(args)
    if legacy is not None:
        named.append(("eval", legacy))
    for label, jsonl_path, jsonl_spec in (
        ("seen", args.eval_seen_jsonl, args.eval_seen_jsonl_spec),
        ("unseen", args.eval_unseen_jsonl, args.eval_unseen_jsonl_spec),
    ):
        records = load_eval_records_from_sources(
            label=label,
            jsonl_path=jsonl_path,
            jsonl_spec=jsonl_spec,
            args=args,
        )
        if records is not None:
            named.append((label, records))
    return named


def _record_value(record: dict[str, Any], key: str) -> Any | None:
    if key in record and record[key] not in (None, ""):
        return record[key]
    meta = record.get("moss_codecvc_meta")
    if isinstance(meta, dict) and meta.get(key) not in (None, ""):
        return meta[key]
    return None


def sample_timbre_ref_speaker_embedding_paths(
    records,
    *,
    pool_size: int,
    seed: int,
    max_attempt_multiplier: int = 64,
) -> list[str]:
    requested = int(pool_size)
    if requested <= 0:
        return []
    total = len(records)
    if total <= 0:
        raise ValueError("Cannot sample speaker InfoNCE negatives from an empty training set")
    rng = random.Random(int(seed))
    selected: list[str] = []
    seen: set[str] = set()
    max_attempts = max(requested * int(max_attempt_multiplier), requested + 100)
    attempts = 0
    while len(selected) < requested and attempts < max_attempts:
        attempts += 1
        record = records[rng.randrange(total)]
        value = _record_value(record, "timbre_ref_speaker_embedding_path")
        if value in (None, ""):
            value = _record_value(record, "target_speaker_embedding_path")
        if value in (None, ""):
            continue
        path = str(value)
        if path in seen:
            continue
        seen.add(path)
        selected.append(path)
    if len(selected) < requested:
        raise ValueError(
            "Could not sample enough unique timbre_ref_speaker_embedding_path values for "
            f"speaker InfoNCE negative pool: requested={requested} got={len(selected)} "
            f"attempts={attempts} total_rows={total}"
        )
    return selected


def _int_record_value(record: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = _record_value(record, key)
        if value in (None, ""):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 1:
            return parsed
    return 0


def iter_manifest_probe_records(records, *, max_probe_rows: int = 32):
    """Yield a small probe set from each logical JSONL source.

    Mixed Ver2.1 training can concatenate no_text and text manifests with
    logical repeats. Looking only at the first N effective rows may miss vocab
    metadata from later sources, so probe each source independently when the
    lazy wrappers expose that structure.
    """
    limit = max(1, int(max_probe_rows))
    if isinstance(records, ConcatRecords):
        for source in records.sources:
            source_records = getattr(source, "records", source)
            for idx in range(min(len(source_records), limit)):
                yield source_records[idx]
        return
    if isinstance(records, RepeatedRecords):
        source_records = records.records
        for idx in range(min(len(source_records), limit)):
            yield source_records[idx]
        return
    for idx in range(min(len(records), limit)):
        yield records[idx]


def infer_content_ctc_vocab_size_from_manifest(records, *, max_probe_rows: int = 32) -> int:
    """Prefer explicit manifest vocab metadata from offline content-token extraction."""
    best_explicit = 0
    best_inline_max = 0
    for record in iter_manifest_probe_records(records, max_probe_rows=max_probe_rows):
        explicit = _int_record_value(
            record,
            "content_ctc_vocab_size",
            "content_vocab_size",
            "content_token_vocab_size",
            "vocab_size_with_blank",
        )
        if explicit > 1:
            best_explicit = max(best_explicit, explicit)
        ids = _record_value(record, "content_token_ids") or _record_value(record, "content_ref_token_ids")
        if isinstance(ids, (list, tuple)) and ids:
            try:
                best_inline_max = max(best_inline_max, max(int(item) for item in ids))
            except (TypeError, ValueError):
                pass
    if best_explicit > 1:
        return best_explicit
    return best_inline_max + 1 if best_inline_max > 0 else 0


def _normalise_manifest_meta_value(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if "/" in text or text.endswith((".json", ".model", ".vocab")):
        try:
            return str(Path(text).expanduser().resolve(strict=False))
        except Exception:
            return text
    return text


def collect_content_tokenizer_metadata(records, *, max_probe_rows: int = 32) -> list[dict[str, Any]]:
    """Collect CTC tokenizer metadata from each logical source.

    Mixed text/no_text training previously allowed different char vocabs to be
    concatenated. That makes the same content_token_id mean different symbols
    across sources, so CTC becomes contradictory. Probe each JSONL source and
    fail early when metadata is inconsistent.
    """

    grouped: dict[str, dict[str, Any]] = {}
    for source_idx, record in enumerate(iter_manifest_probe_records(records, max_probe_rows=max_probe_rows)):
        source_path = ""
        if isinstance(records, ConcatRecords):
            # iter_manifest_probe_records yields source rows in order, but does not
            # expose source index. Use per-record vocab path when available.
            source_path = _normalise_manifest_meta_value(_record_value(record, "content_vocab_path"))
        key_values = {
            "content_ctc_vocab_size": _int_record_value(
                record,
                "content_ctc_vocab_size",
                "content_vocab_size",
                "content_token_vocab_size",
                "vocab_size_with_blank",
            ),
            "content_tokenizer": _normalise_manifest_meta_value(_record_value(record, "content_tokenizer")),
            "content_tokenizer_id": _normalise_manifest_meta_value(_record_value(record, "content_tokenizer_id")),
            "content_vocab_path": _normalise_manifest_meta_value(_record_value(record, "content_vocab_path")),
        }
        if not any(key_values.values()):
            continue
        group_key = source_path or f"probe_record_{source_idx}"
        if group_key not in grouped:
            grouped[group_key] = {"source": group_key, "count": 0, **key_values}
        grouped[group_key]["count"] += 1
        for key, value in key_values.items():
            if not value:
                continue
            old = grouped[group_key].get(key)
            if old in (None, "", 0):
                grouped[group_key][key] = value
    return list(grouped.values())


def infer_sentencepiece_content_tokenizer_path(records, *, max_probe_rows: int = 32) -> str:
    paths: set[str] = set()
    for meta in collect_content_tokenizer_metadata(records, max_probe_rows=max_probe_rows):
        tokenizer = str(meta.get("content_tokenizer") or "").strip().lower()
        vocab_path = str(meta.get("content_vocab_path") or "").strip()
        if vocab_path and (tokenizer == "sentencepiece" or vocab_path.endswith(".model")):
            paths.add(vocab_path)
    if len(paths) == 1:
        return next(iter(paths))
    return ""


def load_sentencepiece_content_tokenizer(model_path: str):
    path = Path(model_path).expanduser()
    if not path.exists():
        return None
    try:
        import sentencepiece as spm
    except ImportError:
        print(
            f"[content-tokenizer] sentencepiece is not installed; cannot load fallback tokenizer {path}",
            flush=True,
        )
        return None
    processor = spm.SentencePieceProcessor()
    if not processor.Load(str(path)):
        raise ValueError(f"failed to load SentencePiece content tokenizer: {path}")
    print(f"[content-tokenizer] loaded sentencepiece fallback tokenizer={path}", flush=True)
    return processor


def content_tokenizer_vocab_size(tokenizer: Any | None) -> int:
    if tokenizer is None:
        return 0
    getter = getattr(tokenizer, "get_piece_size", None)
    if getter is not None:
        return int(getter())
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if callable(vocab_size):
        return int(vocab_size())
    if vocab_size is not None:
        return int(vocab_size)
    try:
        return int(len(tokenizer))
    except TypeError:
        return 0


def validate_content_tokenizer_consistency(
    records,
    *,
    enabled: bool,
    allow_mixed: bool,
    max_probe_rows: int = 32,
) -> None:
    if not enabled:
        return
    metas = collect_content_tokenizer_metadata(records, max_probe_rows=max_probe_rows)
    if len(metas) <= 1:
        return
    vocab_sizes = {int(item.get("content_ctc_vocab_size") or 0) for item in metas if int(item.get("content_ctc_vocab_size") or 0) > 1}
    tokenizer_ids = {str(item.get("content_tokenizer_id") or "") for item in metas if item.get("content_tokenizer_id")}
    vocab_paths = {str(item.get("content_vocab_path") or "") for item in metas if item.get("content_vocab_path")}
    tokenizers = {str(item.get("content_tokenizer") or "") for item in metas if item.get("content_tokenizer")}
    inconsistent = False
    reasons: list[str] = []
    if len(vocab_sizes) > 1:
        inconsistent = True
        reasons.append(f"content_ctc_vocab_size={sorted(vocab_sizes)}")
    if len(tokenizer_ids) > 1:
        inconsistent = True
        reasons.append(f"content_tokenizer_id={sorted(tokenizer_ids)}")
    if not tokenizer_ids and len(vocab_paths) > 1:
        inconsistent = True
        reasons.append(f"content_vocab_path={sorted(vocab_paths)}")
    if len(tokenizers) > 1:
        inconsistent = True
        reasons.append(f"content_tokenizer={sorted(tokenizers)}")
    if not inconsistent:
        print(f"[content_ctc] tokenizer metadata consistent sources={len(metas)} vocab_sizes={sorted(vocab_sizes)}", flush=True)
        return
    message = (
        "[content_ctc] Mixed content tokenizer/vocab metadata detected. "
        "This makes content_token_ids contradictory across train JSONL sources. "
        f"Reasons: {'; '.join(reasons)}. "
        "Regenerate no_text/text content_token_ids with one shared tokenizer, or pass "
        "--allow-mixed-content-tokenizers only for debugging."
    )
    print(message, flush=True)
    print(json.dumps(metas, ensure_ascii=False, indent=2), flush=True)
    if not allow_mixed:
        raise ValueError(message)


def infer_semantic_feature_dim_from_probe(
    probe: dict[str, Any],
    *,
    semantic_source: str,
) -> int:
    source = str(semantic_source or "source").strip().lower()
    preferred_key = "target_semantic_features" if source == "target" else "source_semantic_features"
    if source in {"mode_aware", "auto"}:
        preferred_key = "source_semantic_features"
    for key in (preferred_key, "source_semantic_features", "target_semantic_features"):
        value = probe.get(key)
        if torch.is_tensor(value) and value.dim() == 3 and int(value.shape[-1]) > 0:
            return int(value.shape[-1])
    return 0


def build_processor(model_path: str, codec_path: str | None, moss_root: str | None):
    if moss_root and str(moss_root) not in sys.path:
        sys.path.insert(0, str(moss_root))

    from moss_tts_delay.processing_moss_tts import MossTTSDelayProcessor

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return MossTTSDelayProcessor(
        tokenizer=tokenizer,
        audio_tokenizer=None,
        model_config=config,
    )


def build_lora_target_modules(lm_heads_mode: str, n_heads: int) -> list[str]:
    targets = list(LORA_TARGET_MODULES)
    if lm_heads_mode == "audio":
        targets.extend([f"lm_heads.{idx}" for idx in range(1, n_heads)])
    elif lm_heads_mode == "all":
        targets.extend([f"lm_heads.{idx}" for idx in range(n_heads)])
    return targets


def _map_saved_lora_key_to_active_adapter(key: str, adapter_name: str = "default") -> str:
    if f".lora_A.{adapter_name}.weight" in key or f".lora_B.{adapter_name}.weight" in key:
        return key
    return key.replace(".lora_A.weight", f".lora_A.{adapter_name}.weight").replace(
        ".lora_B.weight", f".lora_B.{adapter_name}.weight"
    )


def load_lora_adapter_direct(model: torch.nn.Module, adapter_path: str | Path, adapter_name: str = "default") -> None:
    try:
        from safetensors.torch import load_file as load_safetensors_file
    except ImportError as exc:
        raise ImportError("Resuming LoRA adapters requires `safetensors`.") from exc

    adapter_file = Path(adapter_path) / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(f"Missing LoRA adapter weights: {adapter_file}")
    adapter_state_raw = load_safetensors_file(str(adapter_file), device="cpu")
    adapter_state = {
        _map_saved_lora_key_to_active_adapter(key, adapter_name): value for key, value in adapter_state_raw.items()
    }
    load_result = model.load_state_dict(adapter_state, strict=False)
    missing = list(getattr(load_result, "missing_keys", []) or [])
    unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
    missing_lora = [key for key in missing if "lora_" in key]
    unexpected_lora = [key for key in unexpected if "lora_" in key]
    if missing_lora or unexpected_lora:
        raise RuntimeError(
            "LoRA resume key mismatch: "
            f"missing_lora={len(missing_lora)} unexpected_lora={len(unexpected_lora)} "
            f"missing_sample={missing_lora[:3]} unexpected_sample={unexpected_lora[:3]} "
            f"from={adapter_file}"
        )
    print(
        "[resume] loaded LoRA adapter "
        f"keys={len(adapter_state)} ignored_missing_non_lora={len(missing)} "
        f"ignored_unexpected_non_lora={len(unexpected)} from={adapter_file}",
        flush=True,
    )


def apply_lora(model: torch.nn.Module, args: argparse.Namespace) -> tuple[torch.nn.Module, dict[str, int]]:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("LoRA training requires `peft`. Install it in the moss-tts environment first.") from exc

    for param in model.parameters():
        param.requires_grad = False

    n_heads = int(model.config.n_vq) + 1
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=build_lora_target_modules(args.lm_heads_mode, n_heads),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    original_get_input_embeddings = type(model).get_input_embeddings
    type(model).get_input_embeddings = lambda self, input_ids=None: (
        original_get_input_embeddings(self, input_ids)
        if input_ids is not None
        else self.language_model.get_input_embeddings()
    )
    if not hasattr(type(model), "prepare_inputs_for_generation"):
        type(model).prepare_inputs_for_generation = lambda self, *args, **kwargs: kwargs

    model = get_peft_model(model, lora_config)
    if args.resume_adapter_path:
        load_lora_adapter_direct(model, args.resume_adapter_path)

    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    original_forward = type(base_model).forward

    def patched_forward(self, *forward_args, output_hidden_states=None, return_dict=None, **kwargs):
        return original_forward(self, *forward_args, **kwargs)

    type(base_model).forward = patched_forward

    allowed_fragments = ["language_model.layers."]
    if args.lm_heads_mode in {"audio", "all"}:
        allowed_fragments.append("lm_heads.")
    module_substrings = {
        "all": LORA_TARGET_MODULES,
        "mlp": ("gate_proj", "up_proj", "down_proj"),
        "mlp_plus_o": ("gate_proj", "up_proj", "down_proj", "o_proj"),
    }
    allowed_modules = module_substrings[args.trainable_lora_modules]

    for name, param in model.named_parameters():
        param.requires_grad = (
            "lora_" in name
            and any(fragment in name for fragment in allowed_fragments)
            and any(module_name in name for module_name in allowed_modules)
        )
        if args.lm_heads_mode == "audio" and "lm_heads.0." in name:
            param.requires_grad = False
        if "emb_ext" in name:
            param.requires_grad = False

    trainable = {name: param.numel() for name, param in model.named_parameters() if param.requires_grad}
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters found.")
    if args.lm_heads_mode == "none" and any("lm_heads." in name for name in trainable):
        raise RuntimeError("lm_heads LoRA params are trainable despite --lm-heads-mode none")
    if args.lm_heads_mode in {"audio", "all"} and not any("lm_heads." in name for name in trainable):
        raise RuntimeError("No lm_heads LoRA params found for requested lm_heads mode")
    return model, trainable


def collect_trainable_parameters(model: torch.nn.Module) -> dict[str, int]:
    return {name: param.numel() for name, param in model.named_parameters() if param.requires_grad}


def apply_ver25_freeze_controls(model: torch.nn.Module, args: argparse.Namespace) -> dict[str, Any]:
    """Apply Ver2.5 ablation freeze switches after LoRA/wrapper construction."""

    source_semantic_markers = (
        "source_semantic_memory_encoder",
        "source_semantic_codec_residual_encoder",
        "source_semantic_layer_adapters",
    )
    timbre_markers = (
        "timbre_memory",
        "layer_adapters",
        "speaker_projection",
        "prosody_head",
        "content_head",
        "content_ctc_head",
        "content_token_head",
        "content_codec_head",
        "semantic_token_head",
        "semantic_feature_head",
        "progress_stop_head",
        "source_prosody_encoder",
        "speaker_side_adaln",
        "speaker_side_kv_bias",
        "speaker_side_gate_logits",
        "speaker_cross_attn_tokens",
        "speaker_cross_attn_seq_projector",
        "speaker_cross_attn_layers",
        "null_speaker_embedding",
    )
    route_markers = (
        "role_router",
        "target_head_router",
    )
    changed = 0
    trainable_before = sum(param.numel() for param in model.parameters() if param.requires_grad)
    for name, param in model.named_parameters():
        should_train = bool(param.requires_grad)
        is_source_semantic = any(marker in name for marker in source_semantic_markers)
        if args.train_source_semantic_only:
            should_train = is_source_semantic
        else:
            if args.freeze_lora and "lora_" in name:
                should_train = False
            if args.freeze_role_routing and any(marker in name for marker in route_markers):
                should_train = False
            if args.freeze_timbre_adapter and any(marker in name for marker in timbre_markers):
                should_train = False
        if bool(param.requires_grad) != bool(should_train):
            param.requires_grad = bool(should_train)
            changed += int(param.numel())
    trainable_after = {
        name: param.numel()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    source_semantic_trainable = {
        name: count
        for name, count in trainable_after.items()
        if any(marker in name for marker in source_semantic_markers)
    }
    if args.train_source_semantic_only and not source_semantic_trainable:
        raise RuntimeError(
            "--train-source-semantic-only was requested, but no SourceSemanticMemory parameters are trainable. "
            "Check --enable-source-semantic-memory and source semantic config."
        )
    return {
        "trainable_before": int(trainable_before),
        "trainable_after": int(sum(trainable_after.values())),
        "changed_param_count": int(changed),
        "trainable_tensors_after": len(trainable_after),
        "source_semantic_trainable_tensors": len(source_semantic_trainable),
        "source_semantic_trainable_params": int(sum(source_semantic_trainable.values())),
        "train_source_semantic_only": bool(args.train_source_semantic_only),
        "freeze_lora": bool(args.freeze_lora),
        "freeze_role_routing": bool(args.freeze_role_routing),
        "freeze_timbre_adapter": bool(args.freeze_timbre_adapter),
    }


ROUTING_GATE_PARAM_SUFFIXES = (
    "role_router.gate_logits",
    "target_head_router.prosody_gate_logits",
    "target_head_router.timbre_gate_logits",
)

ROUTING_GATE_BUFFER_SUFFIXES = (
    "role_router.gate_prior",
    "target_head_router.prosody_gate_prior",
    "target_head_router.timbre_gate_prior",
)


def is_routing_gate_param_name(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in ROUTING_GATE_PARAM_SUFFIXES)


def is_source_semantic_gate_param_name(name: str) -> bool:
    return (
        "source_semantic_layer_adapters" in name
        and (name.endswith(".gate_logit") or name.endswith(".text_gate_logit"))
    )


def is_timbre_adapter_gate_param_name(name: str) -> bool:
    return (
        (
            "source_semantic_layer_adapters" not in name
            and "layer_adapters" in name
            and name.endswith(".gate")
        )
        or "speaker_side_gate_logits" in name
        or ("speaker_cross_attn_layers" in name and name.endswith(".gate_logit"))
    )


def is_routing_gate_buffer_name(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in ROUTING_GATE_BUFFER_SUFFIXES)


def routing_gate_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {name: param for name, param in model.named_parameters() if param.requires_grad and is_routing_gate_param_name(name)}


def source_semantic_gate_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {
        name: param
        for name, param in model.named_parameters()
        if param.requires_grad and is_source_semantic_gate_param_name(name)
    }


def timbre_adapter_gate_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {
        name: param
        for name, param in model.named_parameters()
        if param.requires_grad and is_timbre_adapter_gate_param_name(name)
    }


def is_source_semantic_trainable_param_name(name: str) -> bool:
    return (
        "source_semantic_memory_encoder" in name
        or "source_semantic_codec_residual_encoder" in name
        or "source_semantic_layer_adapters" in name
    )


def source_semantic_trainable_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {
        name: param
        for name, param in model.named_parameters()
        if param.requires_grad and is_source_semantic_trainable_param_name(name)
    }


def collect_lora_fsdp_ignored_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Keep trainable LoRA A/B modules replicated so rank0 can save them without FSDP gather."""
    ignored: list[torch.nn.Module] = []
    lora_markers = (
        ".lora_A",
        ".lora_B",
        ".lora_embedding_A",
        ".lora_embedding_B",
        ".lora_magnitude_vector",
    )
    for name, module in model.named_modules():
        if not any(marker in name for marker in lora_markers):
            continue
        if any(param.requires_grad for param in module.parameters(recurse=True)):
            ignored.append(module)
    return ignored


def sync_ignored_trainable_grads(modules: list[torch.nn.Module]) -> None:
    """Average gradients for trainable modules excluded from FSDP."""
    if not modules:
        return
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    world_size = torch.distributed.get_world_size()
    if world_size <= 1:
        return
    seen: set[int] = set()
    for module in modules:
        for param in module.parameters(recurse=True):
            param_id = id(param)
            if param_id in seen or not param.requires_grad:
                continue
            seen.add(param_id)
            # Every rank must execute the exact same collective sequence. Some
            # Ver2.1 auxiliary heads are conditionally active per batch/mode, so
            # a parameter can have grad=None on one rank and a real grad on
            # another. Reducing an explicit zero gradient keeps DDP/FSDP
            # collectives aligned and preserves the correct averaged update.
            if param.grad is None:
                param.grad = torch.zeros_like(param.data)
            torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
            param.grad.div_(world_size)


def cast_floating_state(module: torch.nn.Module, dtype: torch.dtype) -> None:
    """Keep FSDP-wrapped modules dtype-uniform after adding LoRA/adapters."""
    for name, param in module.named_parameters():
        if (
            is_routing_gate_param_name(name)
            or is_source_semantic_gate_param_name(name)
            or is_timbre_adapter_gate_param_name(name)
        ):
            if param.dtype != torch.float32:
                param.data = param.data.float()
                if param.grad is not None:
                    param.grad.data = param.grad.data.float()
            continue
        if param.is_floating_point() and param.dtype != dtype:
            param.data = param.data.to(dtype=dtype)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(dtype=dtype)
    for name, buffer in module.named_buffers():
        if is_routing_gate_buffer_name(name):
            if buffer.dtype != torch.float32:
                buffer.data = buffer.data.float()
            continue
        if buffer.is_floating_point() and buffer.dtype != dtype:
            buffer.data = buffer.data.to(dtype=dtype)


def build_optimizer_param_groups(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    routing_gate_lr_multiplier: float,
    source_semantic_lr_multiplier: float = 1.0,
    source_semantic_gate_lr_multiplier: float = 10.0,
    content_ctc_head_lr_multiplier: float = 1.0,
    timbre_adapter_gate_lr_multiplier: float = 1.0,
    ref_speaker_prompt_lr_multiplier: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gate_named_params = routing_gate_parameters(model)
    gate_param_ids = {id(param) for param in gate_named_params.values()}
    source_semantic_gate_named_params = source_semantic_gate_parameters(model)
    source_semantic_gate_param_ids = {id(param) for param in source_semantic_gate_named_params.values()}
    timbre_adapter_gate_named_params = timbre_adapter_gate_parameters(model)
    timbre_adapter_gate_param_ids = {id(param) for param in timbre_adapter_gate_named_params.values()}
    source_semantic_named_params = source_semantic_trainable_parameters(model)
    source_semantic_named_params = {
        name: param
        for name, param in source_semantic_named_params.items()
        if id(param) not in source_semantic_gate_param_ids
    }
    use_source_semantic_group = bool(source_semantic_named_params) and float(source_semantic_lr_multiplier) != 1.0
    source_semantic_param_ids = (
        {id(param) for param in source_semantic_named_params.values()} if use_source_semantic_group else set()
    )
    ctc_head_named_params = {
        name: param
        for name, param in model.named_parameters()
        if "content_ctc_head" in name and param.requires_grad
    }
    use_ctc_head_group = bool(ctc_head_named_params) and float(content_ctc_head_lr_multiplier) != 1.0
    ctc_head_param_ids = {id(param) for param in ctc_head_named_params.values()} if use_ctc_head_group else set()
    ref_speaker_prompt_named_params = {
        name: param
        for name, param in model.named_parameters()
        if "ref_speaker_prompt" in name and param.requires_grad
    }
    use_ref_speaker_prompt_group = bool(ref_speaker_prompt_named_params) and float(ref_speaker_prompt_lr_multiplier) != 1.0
    ref_speaker_prompt_param_ids = (
        {id(param) for param in ref_speaker_prompt_named_params.values()} if use_ref_speaker_prompt_group else set()
    )
    base_params = [
        param
        for param in model.parameters()
        if (
            param.requires_grad
            and id(param) not in gate_param_ids
            and id(param) not in source_semantic_gate_param_ids
            and id(param) not in timbre_adapter_gate_param_ids
            and id(param) not in source_semantic_param_ids
            and id(param) not in ctc_head_param_ids
            and id(param) not in ref_speaker_prompt_param_ids
        )
    ]
    param_groups: list[dict[str, Any]] = [{"params": base_params, "lr": float(learning_rate), "weight_decay": float(weight_decay)}]
    summary = {
        "base_tensors": len(base_params),
        "base_params": sum(param.numel() for param in base_params),
        "routing_gate_tensors": len(gate_named_params),
        "routing_gate_params": sum(param.numel() for param in gate_named_params.values()),
        "source_semantic_gate_tensors": len(source_semantic_gate_named_params),
        "source_semantic_gate_params": sum(param.numel() for param in source_semantic_gate_named_params.values()),
        "source_semantic_gate_lr_multiplier": (
            float(source_semantic_gate_lr_multiplier) if source_semantic_gate_named_params else 1.0
        ),
        "timbre_adapter_gate_tensors": len(timbre_adapter_gate_named_params),
        "timbre_adapter_gate_params": sum(param.numel() for param in timbre_adapter_gate_named_params.values()),
        "timbre_adapter_gate_lr_multiplier": (
            float(timbre_adapter_gate_lr_multiplier) if timbre_adapter_gate_named_params else 1.0
        ),
        "source_semantic_tensors": len(source_semantic_named_params),
        "source_semantic_params": sum(param.numel() for param in source_semantic_named_params.values()),
        "source_semantic_lr_multiplier": (
            float(source_semantic_lr_multiplier) if source_semantic_named_params else 1.0
        ),
        "content_ctc_head_tensors": len(ctc_head_named_params),
        "content_ctc_head_params": sum(param.numel() for param in ctc_head_named_params.values()),
        "content_ctc_head_lr_multiplier": float(content_ctc_head_lr_multiplier) if ctc_head_named_params else 1.0,
        "ref_speaker_prompt_tensors": len(ref_speaker_prompt_named_params),
        "ref_speaker_prompt_params": sum(param.numel() for param in ref_speaker_prompt_named_params.values()),
        "ref_speaker_prompt_lr_multiplier": (
            float(ref_speaker_prompt_lr_multiplier) if ref_speaker_prompt_named_params else 1.0
        ),
    }
    if gate_named_params:
        for param in gate_named_params.values():
            if param.dtype != torch.float32:
                param.data = param.data.float()
        param_groups.append(
            {
                "params": list(gate_named_params.values()),
                "lr": float(learning_rate) * float(routing_gate_lr_multiplier),
                "weight_decay": 0.0,
            }
        )
    if source_semantic_gate_named_params:
        for param in source_semantic_gate_named_params.values():
            if param.dtype != torch.float32:
                param.data = param.data.float()
        param_groups.append(
            {
                "params": list(source_semantic_gate_named_params.values()),
                "lr": float(learning_rate) * float(source_semantic_gate_lr_multiplier),
                "weight_decay": 0.0,
            }
        )
    if timbre_adapter_gate_named_params:
        for param in timbre_adapter_gate_named_params.values():
            if param.dtype != torch.float32:
                param.data = param.data.float()
        param_groups.append(
            {
                "params": list(timbre_adapter_gate_named_params.values()),
                "lr": float(learning_rate) * float(timbre_adapter_gate_lr_multiplier),
                "weight_decay": 0.0,
            }
        )
    if use_source_semantic_group:
        param_groups.append(
            {
                "params": list(source_semantic_named_params.values()),
                "lr": float(learning_rate) * float(source_semantic_lr_multiplier),
                "weight_decay": float(weight_decay),
            }
        )
    if use_ctc_head_group:
        param_groups.append(
            {
                "params": list(ctc_head_named_params.values()),
                "lr": float(learning_rate) * float(content_ctc_head_lr_multiplier),
                "weight_decay": float(weight_decay),
            }
        )
    if use_ref_speaker_prompt_group:
        param_groups.append(
            {
                "params": list(ref_speaker_prompt_named_params.values()),
                "lr": float(learning_rate) * float(ref_speaker_prompt_lr_multiplier),
                "weight_decay": float(weight_decay),
            }
        )
    return param_groups, summary


def resolve_timbre_memory_config(cfg: dict[str, Any], args: argparse.Namespace) -> TimbreMemoryConfig:
    enabled = deep_get(cfg, "model.use_timbre_memory", False)
    if args.version == "ver2":
        enabled = True
    if args.use_timbre_memory is not None:
        enabled = args.use_timbre_memory
    tokens = args.timbre_memory_tokens
    if tokens is None:
        tokens = int(deep_get(cfg, "model.timbre_memory_tokens", 16))
    layers = args.timbre_adapter_layers
    if layers is None:
        layers = deep_get(cfg, "model.timbre_adapter_layers", "last_4")
    encoder_type = args.timbre_encoder_type or str(deep_get(cfg, "model.timbre_encoder_type", "conformer"))
    encoder_layers = (
        int(args.timbre_encoder_layers)
        if args.timbre_encoder_layers is not None
        else int(deep_get(cfg, "model.timbre_encoder_layers", 2))
    )
    conv_kernel_size = (
        int(args.timbre_conformer_kernel_size)
        if args.timbre_conformer_kernel_size is not None
        else int(deep_get(cfg, "model.timbre_conformer_kernel_size", 7))
    )
    speaker_conditioning = (
        bool(args.timbre_speaker_conditioning)
        if args.timbre_speaker_conditioning is not None
        else bool(deep_get(cfg, "model.timbre_speaker_conditioning", True))
    )
    target_speaker_similarity_weight = (
        float(args.target_speaker_similarity_weight)
        if args.target_speaker_similarity_weight is not None
        else float(deep_get(cfg, "loss.target_speaker_similarity_weight", 0.0))
    )
    source_speaker_suppression_weight = (
        float(args.source_speaker_suppression_weight)
        if args.source_speaker_suppression_weight is not None
        else float(deep_get(cfg, "loss.source_speaker_suppression_weight", 0.0))
    )
    speaker_embedding_dim = (
        int(args.speaker_embedding_dim)
        if args.speaker_embedding_dim is not None
        else int(deep_get(cfg, "model.speaker_embedding_dim", 0))
    )
    speaker_loss_margin = (
        float(args.speaker_loss_margin)
        if args.speaker_loss_margin is not None
        else float(deep_get(cfg, "loss.speaker_loss_margin", 0.0))
    )
    ref_speaker_prompt_tokens = (
        int(args.ref_speaker_prompt_tokens)
        if args.ref_speaker_prompt_tokens is not None
        else int(deep_get(cfg, "model.ref_speaker_prompt_tokens", 0))
    )
    ref_speaker_prompt_dropout = (
        float(args.ref_speaker_prompt_dropout)
        if args.ref_speaker_prompt_dropout is not None
        else float(deep_get(cfg, "model.ref_speaker_prompt_dropout", 0.0))
    )
    ref_speaker_prompt_mode = args.ref_speaker_prompt_mode or str(deep_get(cfg, "model.ref_speaker_prompt_mode", "memory"))
    ref_speaker_prompt_token_source = args.ref_speaker_prompt_token_source or str(
        deep_get(cfg, "model.ref_speaker_prompt_token_source", "speaker_mlp")
    )
    ref_speaker_prompt_slot = (
        bool(args.ref_speaker_prompt_slot)
        if args.ref_speaker_prompt_slot is not None
        else bool(deep_get(cfg, "model.ref_speaker_prompt_slot", False))
    )
    ref_speaker_prompt_slot_code = (
        int(args.ref_speaker_prompt_slot_code)
        if args.ref_speaker_prompt_slot_code is not None
        else int(deep_get(cfg, "model.ref_speaker_prompt_slot_code", -1))
    )
    ref_speaker_prompt_slot_pack_mode = args.ref_speaker_prompt_slot_pack_mode or str(
        deep_get(cfg, "model.ref_speaker_prompt_slot_pack_mode", "pad")
    )
    ref_speaker_prompt_output_norm = (
        bool(args.ref_speaker_prompt_output_norm)
        if args.ref_speaker_prompt_output_norm is not None
        else bool(deep_get(cfg, "model.ref_speaker_prompt_output_norm", False))
    )
    ref_speaker_prompt_output_scale = (
        float(args.ref_speaker_prompt_output_scale)
        if args.ref_speaker_prompt_output_scale is not None
        else float(deep_get(cfg, "model.ref_speaker_prompt_output_scale", 1.0))
    )
    ref_prompt_codec_permutation_enabled = (
        bool(args.ref_prompt_codec_permutation)
        if args.ref_prompt_codec_permutation is not None
        else bool(deep_get(cfg, "data.ref_prompt_codec_permutation_enabled", False))
    )
    ref_prompt_codec_permutation_min_seconds = (
        float(args.ref_prompt_codec_permutation_min_seconds)
        if args.ref_prompt_codec_permutation_min_seconds is not None
        else float(deep_get(cfg, "data.ref_prompt_codec_permutation_min_seconds", 2.0))
    )
    ref_prompt_codec_permutation_max_seconds = (
        float(args.ref_prompt_codec_permutation_max_seconds)
        if args.ref_prompt_codec_permutation_max_seconds is not None
        else float(deep_get(cfg, "data.ref_prompt_codec_permutation_max_seconds", 4.0))
    )
    ref_prompt_codec_permutation_frame_rate = (
        float(args.ref_prompt_codec_permutation_frame_rate)
        if args.ref_prompt_codec_permutation_frame_rate is not None
        else float(deep_get(cfg, "data.ref_prompt_codec_permutation_frame_rate", 12.5))
    )
    ref_prompt_codec_permutation_seed = (
        int(args.ref_prompt_codec_permutation_seed)
        if args.ref_prompt_codec_permutation_seed is not None
        else int(deep_get(cfg, "data.ref_prompt_codec_permutation_seed", 1234))
    )
    ref_prompt_codec_permutation_mode = args.ref_prompt_codec_permutation_mode or str(
        deep_get(cfg, "data.ref_prompt_codec_permutation_mode", "shuffle")
    )
    ref_prompt_codec_permutation_block_seconds = (
        float(args.ref_prompt_codec_permutation_block_seconds)
        if args.ref_prompt_codec_permutation_block_seconds is not None
        else float(deep_get(cfg, "data.ref_prompt_codec_permutation_block_seconds", 0.4))
    )
    target_front_ce_weight = (
        float(args.target_front_ce_weight)
        if args.target_front_ce_weight is not None
        else float(deep_get(cfg, "loss.target_front_ce_weight", 1.0))
    )
    target_front_ce_seconds = (
        float(args.target_front_ce_seconds)
        if args.target_front_ce_seconds is not None
        else float(deep_get(cfg, "loss.target_front_ce_seconds", 0.0))
    )
    target_front_ce_frame_rate = (
        float(args.target_front_ce_frame_rate)
        if args.target_front_ce_frame_rate is not None
        else float(deep_get(cfg, "loss.target_front_ce_frame_rate", 12.5))
    )
    ref_speaker_adaln_weight = (
        float(args.ref_speaker_adaln_weight)
        if args.ref_speaker_adaln_weight is not None
        else float(deep_get(cfg, "model.ref_speaker_adaln_weight", 0.0))
    )
    speaker_infonce_weight = (
        float(args.speaker_infonce_weight)
        if args.speaker_infonce_weight is not None
        else float(deep_get(cfg, "loss.speaker_infonce_weight", 0.0))
    )
    speaker_infonce_temperature = (
        float(args.speaker_infonce_temperature)
        if args.speaker_infonce_temperature is not None
        else float(deep_get(cfg, "loss.speaker_infonce_temperature", 0.07))
    )
    speaker_infonce_negative_pool_size = (
        int(args.speaker_infonce_negative_pool_size)
        if args.speaker_infonce_negative_pool_size is not None
        else int(deep_get(cfg, "loss.speaker_infonce_negative_pool_size", 0))
    )
    speaker_infonce_negative_pool_seed = (
        int(args.speaker_infonce_negative_pool_seed)
        if args.speaker_infonce_negative_pool_seed is not None
        else int(deep_get(cfg, "loss.speaker_infonce_negative_pool_seed", 1234))
    )
    speaker_condition_dropout = (
        float(args.speaker_condition_dropout)
        if args.speaker_condition_dropout is not None
        else float(deep_get(cfg, "model.speaker_condition_dropout", 0.0))
    )
    speaker_side_pathway_enabled = (
        bool(args.enable_speaker_side_pathway)
        if args.enable_speaker_side_pathway is not None
        else bool(deep_get(cfg, "model.speaker_side_pathway.enabled", False))
    )
    speaker_side_pathway_layers = args.speaker_side_pathway_layers or deep_get(
        cfg,
        "model.speaker_side_pathway.layers",
        "all",
    )
    speaker_side_pathway_kv_bias = (
        bool(args.speaker_side_pathway_kv_bias)
        if args.speaker_side_pathway_kv_bias is not None
        else bool(deep_get(cfg, "model.speaker_side_pathway.kv_bias", True))
    )
    speaker_side_pathway_gate_init = (
        float(args.speaker_side_pathway_gate_init)
        if args.speaker_side_pathway_gate_init is not None
        else float(deep_get(cfg, "model.speaker_side_pathway.gate_init", 0.0))
    )
    speaker_side_pathway_dropout = (
        float(args.speaker_side_pathway_dropout)
        if args.speaker_side_pathway_dropout is not None
        else float(deep_get(cfg, "model.speaker_side_pathway.dropout", 0.15))
    )
    speaker_cross_attn_enabled = (
        bool(args.enable_speaker_cross_attn)
        if args.enable_speaker_cross_attn is not None
        else bool(deep_get(cfg, "model.speaker_cross_attn.enabled", False))
    )
    speaker_cross_attn_layers = args.speaker_cross_attn_layers or deep_get(
        cfg,
        "model.speaker_cross_attn.layers",
        "all",
    )
    speaker_cross_attn_tokens = (
        int(args.speaker_cross_attn_tokens)
        if args.speaker_cross_attn_tokens is not None
        else int(deep_get(cfg, "model.speaker_cross_attn.tokens", 8 if speaker_cross_attn_enabled else 0))
    )
    speaker_cross_attn_gate_init = (
        float(args.speaker_cross_attn_gate_init)
        if args.speaker_cross_attn_gate_init is not None
        else float(deep_get(cfg, "model.speaker_cross_attn.gate_init", 0.0))
    )
    speaker_cross_attn_dropout = (
        float(args.speaker_cross_attn_dropout)
        if args.speaker_cross_attn_dropout is not None
        else float(deep_get(cfg, "model.speaker_cross_attn.dropout", 0.0))
    )
    speaker_cross_attn_output_scale = (
        float(args.speaker_cross_attn_output_scale)
        if args.speaker_cross_attn_output_scale is not None
        else float(deep_get(cfg, "model.speaker_cross_attn.output_scale", 1.0))
    )
    speaker_cross_attn_token_init_std = (
        float(args.speaker_cross_attn_token_init_std)
        if args.speaker_cross_attn_token_init_std is not None
        else deep_get(cfg, "model.speaker_cross_attn.token_init_std", None)
    )
    if speaker_cross_attn_token_init_std is not None:
        speaker_cross_attn_token_init_std = float(speaker_cross_attn_token_init_std)
    speaker_cross_attn_alpha_warmup_steps = (
        int(args.speaker_cross_attn_alpha_warmup_steps)
        if args.speaker_cross_attn_alpha_warmup_steps is not None
        else int(deep_get(cfg, "model.speaker_cross_attn.alpha_warmup_steps", 0))
    )
    speaker_cross_attn_source = args.speaker_cross_attn_source or str(
        deep_get(cfg, "model.speaker_cross_attn.source", "vector")
    )
    speaker_cross_attn_seq_dim = (
        int(args.speaker_cross_attn_seq_dim)
        if args.speaker_cross_attn_seq_dim is not None
        else int(deep_get(cfg, "model.speaker_cross_attn.seq_dim", 0))
    )
    if speaker_side_pathway_enabled or speaker_cross_attn_enabled:
        enabled = True
    legacy_timbre_memory_enabled = bool(args.use_timbre_memory) and int(tokens) > 0 and bool(str(layers).strip())
    if not legacy_timbre_memory_enabled:
        tokens = 0
        layers = ""
    use_perturbed_source_prompt = (
        bool(args.use_perturbed_source_prompt)
        if args.use_perturbed_source_prompt is not None
        else bool(deep_get(cfg, "data.use_perturbed_source_prompt", False))
    )
    use_role_routing = bool(deep_get(cfg, "model.use_role_routing", False))
    if args.version == "ver2":
        use_role_routing = True
    if args.enable_role_routing is not None:
        use_role_routing = bool(args.enable_role_routing)
    target_head_routing = bool(deep_get(cfg, "model.target_head_routing", use_role_routing))
    if args.version == "ver2":
        target_head_routing = True
    if args.enable_target_head_routing is not None:
        target_head_routing = bool(args.enable_target_head_routing)
    if target_head_routing:
        use_role_routing = True
    if use_role_routing:
        enabled = True
    route_loss_weight = (
        float(args.lambda_route)
        if args.lambda_route is not None
        else float(deep_get(cfg, "loss.route_loss_weight", deep_get(cfg, "loss.lambda_route", 0.01)))
    )
    prosody_loss_weight = (
        float(args.lambda_prosody)
        if args.lambda_prosody is not None
        else float(deep_get(cfg, "loss.prosody_loss_weight", deep_get(cfg, "loss.lambda_prosody", 0.0)))
    )
    content_loss_weight = (
        float(args.lambda_content)
        if args.lambda_content is not None
        else float(deep_get(cfg, "loss.content_loss_weight", deep_get(cfg, "loss.lambda_content", 0.0)))
    )
    prosody_f0_weight = (
        float(args.prosody_f0_weight)
        if args.prosody_f0_weight is not None
        else float(deep_get(cfg, "loss.prosody_f0_weight", 1.0))
    )
    prosody_voiced_weight = (
        float(args.prosody_voiced_weight)
        if args.prosody_voiced_weight is not None
        else float(deep_get(cfg, "loss.prosody_voiced_weight", 0.5))
    )
    prosody_energy_weight = (
        float(args.prosody_energy_weight)
        if args.prosody_energy_weight is not None
        else float(deep_get(cfg, "loss.prosody_energy_weight", 0.5))
    )
    prosody_pause_weight = (
        float(args.prosody_pause_weight)
        if args.prosody_pause_weight is not None
        else float(deep_get(cfg, "loss.prosody_pause_weight", 1.0))
    )
    prosody_duration_weight = (
        float(args.prosody_duration_weight)
        if args.prosody_duration_weight is not None
        else float(deep_get(cfg, "loss.prosody_duration_weight", 0.5))
    )
    prosody_normalize_f0 = (
        bool(args.prosody_normalize_f0)
        if args.prosody_normalize_f0 is not None
        else bool(deep_get(cfg, "loss.prosody_normalize_f0", True))
    )
    prosody_normalize_energy = (
        bool(args.prosody_normalize_energy)
        if args.prosody_normalize_energy is not None
        else bool(deep_get(cfg, "loss.prosody_normalize_energy", True))
    )
    content_embedding_dim = (
        int(args.content_embedding_dim)
        if args.content_embedding_dim is not None
        else int(deep_get(cfg, "model.content_embedding_dim", 0))
    )
    content_positive = args.content_positive or str(deep_get(cfg, "loss.content_positive", "source"))
    content_embedding_weight = (
        float(args.content_embedding_weight)
        if args.content_embedding_weight is not None
        else float(deep_get(cfg, "loss.content_embedding_weight", 1.0))
    )
    content_ctc_weight = (
        float(args.content_ctc_weight)
        if args.content_ctc_weight is not None
        else float(deep_get(cfg, "loss.content_ctc_weight", 0.0))
    )
    content_ctc_vocab_size = (
        int(args.content_ctc_vocab_size)
        if args.content_ctc_vocab_size is not None
        else int(deep_get(cfg, "model.content_ctc_vocab_size", 0))
    )
    content_ctc_blank_id = (
        int(args.content_ctc_blank_id)
        if args.content_ctc_blank_id is not None
        else int(deep_get(cfg, "loss.content_ctc_blank_id", 0))
    )
    content_ctc_token_offset = (
        int(args.content_ctc_token_offset)
        if args.content_ctc_token_offset is not None
        else int(deep_get(cfg, "loss.content_ctc_token_offset", 1))
    )
    content_token_vocab_size = (
        int(args.content_token_vocab_size)
        if args.content_token_vocab_size is not None
        else int(deep_get(cfg, "model.content_token_vocab_size", 0))
    )
    content_token_weight = (
        float(args.content_token_weight)
        if args.content_token_weight is not None
        else float(deep_get(cfg, "loss.content_token_weight", 0.0))
    )
    content_source_codec_weight = (
        float(args.content_source_codec_weight)
        if args.content_source_codec_weight is not None
        else float(deep_get(cfg, "loss.content_source_codec_weight", 0.0))
    )
    content_source_codec_codebooks = args.content_source_codec_codebooks or str(
        deep_get(cfg, "loss.content_source_codec_codebooks", "0,1,2,3")
    )
    semantic_loss_weight = (
        float(args.semantic_loss_weight)
        if args.semantic_loss_weight is not None
        else float(deep_get(cfg, "loss.semantic_loss_weight", 0.0))
    )
    semantic_mode = args.semantic_mode or str(deep_get(cfg, "semantic_loss.mode", "discrete"))
    semantic_source = args.semantic_source or str(deep_get(cfg, "semantic_loss.source", "source"))
    semantic_vocab_size = (
        int(args.semantic_vocab_size)
        if args.semantic_vocab_size is not None
        else int(deep_get(cfg, "semantic_loss.vocab_size", 0))
    )
    semantic_feature_dim = (
        int(args.semantic_feature_dim)
        if args.semantic_feature_dim is not None
        else int(deep_get(cfg, "semantic_loss.feature_dim", 0))
    )
    semantic_feature_loss_type = args.semantic_feature_loss_type or str(
        deep_get(cfg, "semantic_loss.feature_loss_type", "cosine")
    )
    progress_loss_weight = (
        float(args.progress_loss_weight)
        if args.progress_loss_weight is not None
        else float(deep_get(cfg, "loss.progress_loss_weight", 0.0))
    )
    stop_loss_weight = (
        float(args.stop_loss_weight)
        if args.stop_loss_weight is not None
        else float(deep_get(cfg, "loss.stop_loss_weight", 0.0))
    )
    progress_num_bins = (
        int(args.progress_num_bins)
        if args.progress_num_bins is not None
        else int(deep_get(cfg, "model.progress_num_bins", deep_get(cfg, "loss.progress_num_bins", 32)))
    )
    source_semantic_memory_enabled = bool(deep_get(cfg, "source_semantic_memory.enabled", False))
    if args.enable_source_semantic_memory is not None:
        source_semantic_memory_enabled = bool(args.enable_source_semantic_memory)
    if source_semantic_memory_enabled:
        enabled = True
    source_semantic_feature_dim = (
        int(args.source_semantic_feature_dim)
        if args.source_semantic_feature_dim is not None
        else int(deep_get(cfg, "source_semantic_memory.input_dim", 768))
    )
    if source_semantic_memory_enabled:
        source_semantic_adapter_layers = args.source_semantic_adapter_layers or deep_get(
            cfg,
            "source_semantic_memory.selected_layers",
            "28,30,32,34,35",
        )
    else:
        source_semantic_adapter_layers = ""
    source_semantic_no_text_gate = (
        float(args.source_semantic_no_text_gate)
        if args.source_semantic_no_text_gate is not None
        else float(deep_get(cfg, "source_semantic_memory.no_text_gate", 1.0))
    )
    source_semantic_text_gate = (
        float(args.source_semantic_text_gate)
        if args.source_semantic_text_gate is not None
        else float(deep_get(cfg, "source_semantic_memory.text_gate", 0.0))
    )
    source_semantic_allow_learned_text_gate = (
        bool(args.source_semantic_learned_text_gate)
        if args.source_semantic_learned_text_gate is not None
        else bool(deep_get(cfg, "source_semantic_memory.allow_learned_text_gate", False))
    )
    source_semantic_progress_weight = (
        float(args.source_semantic_progress_weight)
        if args.source_semantic_progress_weight is not None
        else float(deep_get(cfg, "loss.source_semantic_progress_weight", 0.0))
    )
    source_semantic_dropout = (
        float(args.source_semantic_dropout)
        if args.source_semantic_dropout is not None
        else float(deep_get(cfg, "source_semantic_memory.dropout", 0.1))
    )
    source_semantic_init_gate = (
        float(args.source_semantic_init_gate)
        if args.source_semantic_init_gate is not None
        else float(deep_get(cfg, "source_semantic_memory.init_gate", -2.0))
    )
    source_semantic_position_scale = (
        float(args.source_semantic_position_scale)
        if args.source_semantic_position_scale is not None
        else float(deep_get(cfg, "source_semantic_memory.position_scale", 0.0))
    )
    source_semantic_monotonic_bias_strength = (
        float(args.source_semantic_monotonic_bias_strength)
        if args.source_semantic_monotonic_bias_strength is not None
        else float(deep_get(cfg, "source_semantic_memory.monotonic_bias_strength", 0.0))
    )
    source_semantic_monotonic_bias_width = (
        float(args.source_semantic_monotonic_bias_width)
        if args.source_semantic_monotonic_bias_width is not None
        else float(deep_get(cfg, "source_semantic_memory.monotonic_bias_width", 0.25))
    )
    source_content_memory_type = args.source_content_memory_type or str(
        deep_get(cfg, "source_semantic_memory.memory_type", deep_get(cfg, "source_content_memory.type", "hubert_continuous"))
    )
    source_content_vocab_size = (
        int(args.source_content_vocab_size)
        if args.source_content_vocab_size is not None
        else int(deep_get(cfg, "source_content_memory.vocab_size", 0))
    )
    source_content_padding_id = (
        int(args.source_content_padding_id)
        if args.source_content_padding_id is not None
        else int(deep_get(cfg, "source_content_memory.padding_id", 0))
    )
    source_content_codec_bottleneck_dim = (
        int(args.source_content_codec_bottleneck_dim)
        if args.source_content_codec_bottleneck_dim is not None
        else int(deep_get(cfg, "source_content_memory.codec_bottleneck_dim", 256))
    )
    source_content_codec_codebooks = args.source_content_codec_codebooks or str(
        deep_get(cfg, "source_content_memory.codec_codebooks", "first_4")
    )
    source_content_dedup_units = (
        bool(args.source_content_dedup_units)
        if args.source_content_dedup_units is not None
        else bool(deep_get(cfg, "source_content_memory.dedup_units", False))
    )
    source_codec_residual_memory_weight = (
        float(args.source_codec_residual_memory_weight)
        if args.source_codec_residual_memory_weight is not None
        else float(
            deep_get(
                cfg,
                "source_semantic_memory.codec_residual_memory_weight",
                deep_get(cfg, "source_content_memory.codec_residual_memory_weight", 0.0),
            )
        )
    )
    source_codec_residual_memory_detach = (
        bool(args.source_codec_residual_memory_detach)
        if args.source_codec_residual_memory_detach is not None
        else bool(
            deep_get(
                cfg,
                "source_semantic_memory.codec_residual_memory_detach",
                deep_get(cfg, "source_content_memory.codec_residual_memory_detach", False),
            )
        )
    )
    ref_content_suppression_weight = (
        float(args.ref_content_suppression_weight)
        if args.ref_content_suppression_weight is not None
        else float(deep_get(cfg, "loss.ref_content_suppression_weight", 0.0))
    )
    ref_content_suppression_margin = (
        float(args.ref_content_suppression_margin)
        if args.ref_content_suppression_margin is not None
        else float(deep_get(cfg, "loss.ref_content_suppression_margin", 0.0))
    )
    ref_content_suppression_source = args.ref_content_suppression_source or str(
        deep_get(cfg, "loss.ref_content_suppression_source", "auto")
    )
    ref_content_suppression_detach_ref = (
        bool(args.ref_content_suppression_detach_ref)
        if args.ref_content_suppression_detach_ref is not None
        else bool(deep_get(cfg, "loss.ref_content_suppression_detach_ref", True))
    )
    prosody_memory_tokens = (
        int(args.prosody_memory_tokens)
        if args.prosody_memory_tokens is not None
        else int(deep_get(cfg, "model.prosody_memory_tokens", 8))
    )
    source_prosody_encoder_type = args.source_prosody_encoder_type or str(
        deep_get(cfg, "model.source_prosody_encoder_type", encoder_type)
    )
    source_prosody_encoder_layers = (
        int(args.source_prosody_encoder_layers)
        if args.source_prosody_encoder_layers is not None
        else int(deep_get(cfg, "model.source_prosody_encoder_layers", encoder_layers))
    )
    source_prosody_conv_kernel_size = (
        int(args.source_prosody_conv_kernel_size)
        if args.source_prosody_conv_kernel_size is not None
        else int(deep_get(cfg, "model.source_prosody_conv_kernel_size", conv_kernel_size))
    )
    source_prosody_no_text_gate = (
        float(args.source_prosody_no_text_gate)
        if args.source_prosody_no_text_gate is not None
        else float(deep_get(cfg, "model.source_prosody_no_text_gate", 1.0))
    )
    source_prosody_text_gate = (
        float(args.source_prosody_text_gate)
        if args.source_prosody_text_gate is not None
        else float(deep_get(cfg, "model.source_prosody_text_gate", 1.0))
    )
    return TimbreMemoryConfig(
        enabled=bool(enabled),
        timbre_side_only=bool(args.timbre_side_only),
        num_memory_tokens=int(tokens),
        adapter_layers=layers,
        num_heads=int(deep_get(cfg, "model.timbre_adapter_heads", 8)),
        adapter_dim=int(deep_get(cfg, "model.timbre_adapter_dim", 256)),
        dropout=float(deep_get(cfg, "model.timbre_adapter_dropout", 0.0)),
        init_gate=(
            float(args.timbre_adapter_init_gate)
            if args.timbre_adapter_init_gate is not None
            else float(deep_get(cfg, "model.timbre_adapter_init_gate", -4.0))
        ),
        encoder_type=encoder_type,
        encoder_layers=encoder_layers,
        conv_kernel_size=conv_kernel_size,
        speaker_conditioning=speaker_conditioning,
        target_speaker_similarity_weight=target_speaker_similarity_weight,
        source_speaker_suppression_weight=source_speaker_suppression_weight,
        speaker_embedding_dim=speaker_embedding_dim,
        speaker_loss_margin=speaker_loss_margin,
        speaker_encoder_type=args.speaker_encoder_type
        or str(deep_get(cfg, "model.speaker_encoder_type", "embedding_loader")),
        speaker_encoder_path=args.speaker_encoder_path or deep_get(cfg, "model.speaker_encoder_path", None),
        freeze_speaker_encoder=bool(deep_get(cfg, "model.freeze_speaker_encoder", True)),
        ref_speaker_prompt_tokens=ref_speaker_prompt_tokens,
        ref_speaker_prompt_dropout=ref_speaker_prompt_dropout,
        ref_speaker_prompt_mode=ref_speaker_prompt_mode,
        ref_speaker_prompt_token_source=ref_speaker_prompt_token_source,
        ref_speaker_prompt_slot=ref_speaker_prompt_slot,
        ref_speaker_prompt_slot_code=ref_speaker_prompt_slot_code,
        ref_speaker_prompt_slot_pack_mode=ref_speaker_prompt_slot_pack_mode,
        ref_speaker_prompt_output_norm=ref_speaker_prompt_output_norm,
        ref_speaker_prompt_output_scale=ref_speaker_prompt_output_scale,
        ref_prompt_codec_permutation_enabled=ref_prompt_codec_permutation_enabled,
        ref_prompt_codec_permutation_min_seconds=ref_prompt_codec_permutation_min_seconds,
        ref_prompt_codec_permutation_max_seconds=ref_prompt_codec_permutation_max_seconds,
        ref_prompt_codec_permutation_frame_rate=ref_prompt_codec_permutation_frame_rate,
        ref_prompt_codec_permutation_seed=ref_prompt_codec_permutation_seed,
        ref_prompt_codec_permutation_mode=ref_prompt_codec_permutation_mode,
        ref_prompt_codec_permutation_block_seconds=ref_prompt_codec_permutation_block_seconds,
        target_front_ce_weight=target_front_ce_weight,
        target_front_ce_seconds=target_front_ce_seconds,
        target_front_ce_frame_rate=target_front_ce_frame_rate,
        ref_speaker_adaln_weight=ref_speaker_adaln_weight,
        speaker_infonce_weight=speaker_infonce_weight,
        speaker_infonce_temperature=speaker_infonce_temperature,
        speaker_infonce_negative_pool_size=speaker_infonce_negative_pool_size,
        speaker_infonce_negative_pool_seed=speaker_infonce_negative_pool_seed,
        speaker_condition_dropout=speaker_condition_dropout,
        speaker_side_pathway_enabled=speaker_side_pathway_enabled,
        speaker_side_pathway_layers=speaker_side_pathway_layers,
        speaker_side_pathway_kv_bias=speaker_side_pathway_kv_bias,
        speaker_side_pathway_gate_init=speaker_side_pathway_gate_init,
        speaker_side_pathway_dropout=speaker_side_pathway_dropout,
        speaker_cross_attn_enabled=speaker_cross_attn_enabled,
        speaker_cross_attn_layers=speaker_cross_attn_layers,
        speaker_cross_attn_tokens=speaker_cross_attn_tokens,
        speaker_cross_attn_gate_init=speaker_cross_attn_gate_init,
        speaker_cross_attn_dropout=speaker_cross_attn_dropout,
        speaker_cross_attn_output_scale=speaker_cross_attn_output_scale,
        speaker_cross_attn_token_init_std=speaker_cross_attn_token_init_std,
        speaker_cross_attn_alpha_warmup_steps=speaker_cross_attn_alpha_warmup_steps,
        speaker_cross_attn_source=speaker_cross_attn_source,
        speaker_cross_attn_seq_dim=speaker_cross_attn_seq_dim,
        use_perturbed_source_prompt=use_perturbed_source_prompt,
        use_role_routing=use_role_routing,
        route_loss_weight=route_loss_weight,
        prosody_memory_tokens=prosody_memory_tokens,
        source_prosody_encoder_type=source_prosody_encoder_type,
        source_prosody_encoder_layers=source_prosody_encoder_layers,
        source_prosody_conv_kernel_size=source_prosody_conv_kernel_size,
        source_prosody_no_text_gate=source_prosody_no_text_gate,
        source_prosody_text_gate=source_prosody_text_gate,
        target_head_routing=target_head_routing,
        prosody_loss_weight=prosody_loss_weight,
        prosody_f0_weight=prosody_f0_weight,
        prosody_voiced_weight=prosody_voiced_weight,
        prosody_energy_weight=prosody_energy_weight,
        prosody_pause_weight=prosody_pause_weight,
        prosody_duration_weight=prosody_duration_weight,
        prosody_normalize_f0=prosody_normalize_f0,
        prosody_normalize_energy=prosody_normalize_energy,
        content_loss_weight=content_loss_weight,
        content_embedding_dim=content_embedding_dim,
        content_positive=content_positive,
        content_embedding_weight=content_embedding_weight,
        content_ctc_weight=content_ctc_weight,
        content_ctc_vocab_size=content_ctc_vocab_size,
        content_ctc_blank_id=content_ctc_blank_id,
        content_ctc_token_offset=content_ctc_token_offset,
        content_token_vocab_size=content_token_vocab_size,
        content_token_weight=content_token_weight,
        content_source_codec_weight=content_source_codec_weight,
        content_source_codec_codebooks=content_source_codec_codebooks,
        semantic_loss_weight=semantic_loss_weight,
        semantic_mode=semantic_mode,
        semantic_source=semantic_source,
        semantic_vocab_size=semantic_vocab_size,
        semantic_feature_dim=semantic_feature_dim,
        semantic_feature_loss_type=semantic_feature_loss_type,
        progress_loss_weight=progress_loss_weight,
        stop_loss_weight=stop_loss_weight,
        progress_num_bins=progress_num_bins,
        source_semantic_memory_enabled=source_semantic_memory_enabled,
        source_semantic_feature_dim=source_semantic_feature_dim,
        source_semantic_adapter_layers=source_semantic_adapter_layers,
        source_semantic_no_text_gate=source_semantic_no_text_gate,
        source_semantic_text_gate=source_semantic_text_gate,
        source_semantic_allow_learned_text_gate=source_semantic_allow_learned_text_gate,
        source_semantic_progress_weight=source_semantic_progress_weight,
        source_semantic_dropout=source_semantic_dropout,
        source_semantic_init_gate=source_semantic_init_gate,
        source_semantic_position_scale=source_semantic_position_scale,
        source_semantic_monotonic_bias_strength=source_semantic_monotonic_bias_strength,
        source_semantic_monotonic_bias_width=source_semantic_monotonic_bias_width,
        source_content_memory_type=source_content_memory_type,
        source_content_vocab_size=source_content_vocab_size,
        source_content_padding_id=source_content_padding_id,
        source_content_codec_bottleneck_dim=source_content_codec_bottleneck_dim,
        source_content_codec_codebooks=source_content_codec_codebooks,
        source_content_dedup_units=source_content_dedup_units,
        source_codec_residual_memory_weight=source_codec_residual_memory_weight,
        source_codec_residual_memory_detach=source_codec_residual_memory_detach,
        ref_content_suppression_weight=ref_content_suppression_weight,
        ref_content_suppression_margin=ref_content_suppression_margin,
        ref_content_suppression_source=ref_content_suppression_source,
        ref_content_suppression_detach_ref=ref_content_suppression_detach_ref,
    )


def timbre_memory_resume_overrides(config: TimbreMemoryConfig) -> dict[str, Any]:
    """Keep resumed auxiliary modules aligned with this run's CLI/config switches."""

    overrides = asdict(config)
    overrides["enabled"] = True
    return overrides


def enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    language_model = getattr(base_model, "language_model", None)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()
    if language_model is not None and hasattr(language_model, "gradient_checkpointing_enable"):
        language_model.gradient_checkpointing_enable()
    elif hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
    for cfg in (getattr(model, "config", None), getattr(base_model, "config", None), getattr(language_model, "config", None)):
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = False


def lora_grad_stats(model: torch.nn.Module) -> tuple[float, bool]:
    total_norm_sq = 0.0
    has_nonzero = False
    for name, param in model.named_parameters():
        if "lora_" not in name or param.grad is None:
            continue
        grad = param.grad.detach().float()
        norm = grad.norm(2).item()
        total_norm_sq += norm * norm
        has_nonzero = has_nonzero or torch.count_nonzero(grad).item() > 0
    return math.sqrt(total_norm_sq), has_nonzero


def timbre_adapter_grad_stats(model: torch.nn.Module) -> tuple[float, bool]:
    total_norm_sq = 0.0
    has_nonzero = False
    for name, param in model.named_parameters():
        if is_source_semantic_trainable_param_name(name):
            continue
        if (
            "timbre_memory" not in name
            and "layer_adapters" not in name
            and "speaker_projection" not in name
            and "prosody_head" not in name
            and "content_head" not in name
            and "content_ctc_head" not in name
            and "content_token_head" not in name
            and "content_codec_head" not in name
            and "semantic_token_head" not in name
            and "semantic_feature_head" not in name
            and "progress_stop_head" not in name
            and "role_router" not in name
            and "source_prosody_encoder" not in name
            and "target_head_router" not in name
            and "speaker_side_adaln" not in name
            and "speaker_side_kv_bias" not in name
            and "speaker_side_gate_logits" not in name
            and "speaker_cross_attn_tokens" not in name
            and "speaker_cross_attn_seq_projector" not in name
            and "speaker_cross_attn_layers" not in name
            and "null_speaker_embedding" not in name
        ) or param.grad is None:
            continue
        grad = param.grad.detach().float()
        norm = grad.norm(2).item()
        total_norm_sq += norm * norm
        has_nonzero = has_nonzero or torch.count_nonzero(grad).item() > 0
    return math.sqrt(total_norm_sq), has_nonzero


def named_fragment_grad_stats(model: torch.nn.Module, fragments: tuple[str, ...]) -> tuple[float, bool]:
    total_norm_sq = 0.0
    has_nonzero = False
    for name, param in model.named_parameters():
        if param.grad is None or not any(fragment in name for fragment in fragments):
            continue
        grad = param.grad.detach().float()
        norm = grad.norm(2).item()
        total_norm_sq += norm * norm
        has_nonzero = has_nonzero or torch.count_nonzero(grad).item() > 0
    return math.sqrt(total_norm_sq), has_nonzero


def _extract_layer_after_fragment(name: str, fragment: str) -> str | None:
    if fragment not in name:
        return None
    tail = name.split(fragment, 1)[1].lstrip("._")
    for token in tail.split("."):
        if token.isdigit():
            return token
    return None


def named_fragment_layer_grad_norms(model: torch.nn.Module, fragment: str) -> dict[str, float]:
    per_layer: dict[str, float] = {}
    for name, param in model.named_parameters():
        if fragment not in name or param.grad is None:
            continue
        layer = _extract_layer_after_fragment(name, fragment)
        if layer is None:
            continue
        grad = param.grad.detach().float()
        norm = float(torch.linalg.vector_norm(grad).item())
        per_layer[layer] = per_layer.get(layer, 0.0) + norm * norm
    return {layer: math.sqrt(total_sq) for layer, total_sq in per_layer.items()}


def merge_max_norms(dst: dict[str, float], src: dict[str, float]) -> None:
    for layer, value in src.items():
        dst[layer] = max(float(dst.get(layer, 0.0)), float(value))


def ref_speaker_prompt_grad_stats(model: torch.nn.Module) -> tuple[float, bool]:
    total_norm_sq = 0.0
    has_nonzero = False
    for name, param in model.named_parameters():
        if "ref_speaker_prompt" not in name or param.grad is None:
            continue
        grad = param.grad.detach().float()
        norm = grad.norm(2).item()
        total_norm_sq += norm * norm
        has_nonzero = has_nonzero or torch.count_nonzero(grad).item() > 0
    return math.sqrt(total_norm_sq), has_nonzero


def routing_gate_grad_stats(model: torch.nn.Module) -> tuple[float, bool]:
    total_norm_sq = 0.0
    has_nonzero = False
    for name, param in model.named_parameters():
        if not is_routing_gate_param_name(name) or param.grad is None:
            continue
        grad = param.grad.detach().float()
        norm = grad.norm(2).item()
        total_norm_sq += norm * norm
        has_nonzero = has_nonzero or torch.count_nonzero(grad).item() > 0
    return math.sqrt(total_norm_sq), has_nonzero


def source_semantic_gate_grad_stats(model: torch.nn.Module) -> tuple[float, bool]:
    total_norm_sq = 0.0
    has_nonzero = False
    for name, param in model.named_parameters():
        if not is_source_semantic_gate_param_name(name) or param.grad is None:
            continue
        grad = param.grad.detach().float()
        norm = grad.norm(2).item()
        total_norm_sq += norm * norm
        has_nonzero = has_nonzero or torch.count_nonzero(grad).item() > 0
    return math.sqrt(total_norm_sq), has_nonzero


def speaker_cross_attn_gate_grad_stats(model: torch.nn.Module) -> tuple[float, bool]:
    total_norm_sq = 0.0
    has_nonzero = False
    for name, param in model.named_parameters():
        if "speaker_cross_attn_layers" not in name or not name.endswith(".gate_logit") or param.grad is None:
            continue
        grad = param.grad.detach().float()
        norm = grad.norm(2).item()
        total_norm_sq += norm * norm
        has_nonzero = has_nonzero or torch.count_nonzero(grad).item() > 0
    return math.sqrt(total_norm_sq), has_nonzero


def speaker_cross_attn_gate_layer_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    per_layer: dict[str, float] = {}
    for name, param in model.named_parameters():
        if "speaker_cross_attn_layers" not in name or not name.endswith(".gate_logit") or param.grad is None:
            continue
        layer = _extract_layer_after_fragment(name, "speaker_cross_attn_layers")
        if layer is None:
            continue
        grad = param.grad.detach().float()
        per_layer[layer] = float(torch.linalg.vector_norm(grad).item())
    return per_layer


def source_semantic_trainable_grad_stats(model: torch.nn.Module) -> tuple[float, bool]:
    total_norm_sq = 0.0
    has_nonzero = False
    for name, param in model.named_parameters():
        if not is_source_semantic_trainable_param_name(name) or param.grad is None:
            continue
        grad = param.grad.detach().float()
        norm = grad.norm(2).item()
        total_norm_sq += norm * norm
        has_nonzero = has_nonzero or torch.count_nonzero(grad).item() > 0
    return math.sqrt(total_norm_sq), has_nonzero


def content_ctc_head_grad_stats(model: torch.nn.Module) -> tuple[float, bool]:
    total_norm_sq = 0.0
    has_nonzero = False
    for name, param in model.named_parameters():
        if "content_ctc_head" not in name or param.grad is None:
            continue
        grad = param.grad.detach().float()
        norm = grad.norm(2).item()
        total_norm_sq += norm * norm
        has_nonzero = has_nonzero or torch.count_nonzero(grad).item() > 0
    return math.sqrt(total_norm_sq), has_nonzero


def clip_mixed_dtype_trainable_grad_norm_(model: torch.nn.Module, max_norm: float, norm_type: float = 2.0) -> float:
    """Clip trainable gradients without requiring uniform grad dtype.

    FSDP's built-in clip path requires all gradients to share a dtype. Ver2 keeps
    routing gate logits in fp32 while most LoRA/adapter tensors are bf16, so we
    compute a global norm in fp32 and scale each gradient in-place in its native
    dtype.
    """
    grads = [
        param.grad
        for param in model.parameters()
        if param.requires_grad and param.grad is not None
    ]
    if not grads:
        return 0.0
    if norm_type != 2.0:
        torch.nn.utils.clip_grad_norm_(
            [param for param in model.parameters() if param.requires_grad and param.grad is not None],
            max_norm=max_norm,
            norm_type=norm_type,
        )
        return 0.0
    device = grads[0].device
    total_sq = torch.zeros((), device=device, dtype=torch.float32)
    for grad in grads:
        total_sq = total_sq + grad.detach().float().pow(2).sum().to(device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(total_sq, op=torch.distributed.ReduceOp.SUM)
    total_norm = torch.sqrt(total_sq)
    clip_coef = float(max_norm) / (float(total_norm.item()) + 1.0e-6)
    if clip_coef < 1.0:
        for grad in grads:
            grad.detach().mul_(clip_coef)
    return float(total_norm.item())


def capture_routing_gate_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    unwrapped = model
    for name, param in unwrapped.named_parameters():
        if is_routing_gate_param_name(name):
            snapshot[name] = param.detach().float().cpu().clone()
    return snapshot


def capture_source_semantic_gate_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if is_source_semantic_gate_param_name(name):
            snapshot[name] = param.detach().float().cpu().clone()
    return snapshot


def capture_speaker_side_initial_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if (
            "speaker_side_adaln" in name
            or "speaker_side_kv_bias" in name
            or "speaker_side_gate_logits" in name
            or "speaker_cross_attn_tokens" in name
            or "speaker_cross_attn_seq_projector" in name
            or "speaker_cross_attn_layers" in name
            or "null_speaker_embedding" in name
        ):
            snapshot[name] = param.detach().float().cpu().clone()
    return snapshot


def speaker_side_gate_values(model: torch.nn.Module) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, param in model.named_parameters():
        layer = _extract_layer_after_fragment(name, "speaker_side_gate_logits")
        if layer is None:
            continue
        values[layer] = float(torch.sigmoid(param.detach().float()).mean().cpu().item())
    return values


def speaker_cross_attn_gate_values(model: torch.nn.Module) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, param in model.named_parameters():
        if "speaker_cross_attn_layers" not in name or not name.endswith(".gate_logit"):
            continue
        layer = _extract_layer_after_fragment(name, "speaker_cross_attn_layers")
        if layer is None:
            continue
        values[layer] = float(torch.sigmoid(param.detach().float()).mean().cpu().item())
    return values


def summarize_gate_values(values: dict[str, float]) -> dict[str, Any]:
    gate_tensor = torch.tensor(list(values.values()), dtype=torch.float32) if values else torch.empty(0)
    if gate_tensor.numel() <= 0:
        return {"num_layers": 0}
    hist = torch.histc(gate_tensor, bins=10, min=0.0, max=1.0).to(dtype=torch.int64).tolist()
    return {
        "num_layers": int(gate_tensor.numel()),
        "mean": float(gate_tensor.mean().item()),
        "std": float(gate_tensor.std(unbiased=False).item()) if gate_tensor.numel() > 1 else 0.0,
        "min": float(gate_tensor.min().item()),
        "max": float(gate_tensor.max().item()),
        "histogram_bins_0_1_count10": [int(x) for x in hist],
    }


def speaker_side_adaln_drift_stats(
    model: torch.nn.Module,
    initial_snapshot: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    per_layer: dict[str, dict[str, float]] = {}
    for name, param in model.named_parameters():
        if "speaker_side_adaln" not in name or name not in initial_snapshot:
            continue
        layer = _extract_layer_after_fragment(name, "speaker_side_adaln")
        if layer is None:
            continue
        current = param.detach().float().cpu()
        initial = initial_snapshot[name].float()
        diff = current - initial
        row = per_layer.setdefault(
            layer,
            {
                "param_l2_drift_sq": 0.0,
                "shift_head_l2_drift_sq": 0.0,
                "scale_head_l2_drift_sq": 0.0,
                "param_tensors": 0.0,
            },
        )
        row["param_l2_drift_sq"] += float(diff.pow(2).sum().item())
        row["param_tensors"] += 1.0
        tail = name.split("speaker_side_adaln", 1)[1]
        if ".3." in tail and diff.shape and int(diff.shape[0]) % 2 == 0:
            half = int(diff.shape[0]) // 2
            row["shift_head_l2_drift_sq"] += float(diff[:half].pow(2).sum().item())
            row["scale_head_l2_drift_sq"] += float(diff[half:].pow(2).sum().item())
    out: dict[str, dict[str, float]] = {}
    for layer, row in per_layer.items():
        out[layer] = {
            "param_l2_drift": math.sqrt(float(row["param_l2_drift_sq"])),
            "shift_head_l2_drift": math.sqrt(float(row["shift_head_l2_drift_sq"])),
            "scale_head_l2_drift": math.sqrt(float(row["scale_head_l2_drift_sq"])),
            "param_tensors": int(row["param_tensors"]),
        }
    return out


def write_ver29_smoke_train_diagnostics(
    output_dir: Path,
    model: torch.nn.Module,
    *,
    initial_snapshot: dict[str, torch.Tensor],
    max_grad_norms: dict[str, dict[str, float]],
    global_step: int,
) -> None:
    gate_values = speaker_side_gate_values(model)
    gate_summary = summarize_gate_values(gate_values)
    cross_gate_values = speaker_cross_attn_gate_values(model)
    cross_gate_summary = summarize_gate_values(cross_gate_values)
    payload = {
        "global_step": int(global_step),
        "speaker_side_gate_by_layer": {k: gate_values[k] for k in sorted(gate_values, key=lambda x: int(x))},
        "speaker_side_gate_summary": gate_summary,
        "speaker_cross_attn_gate_by_layer": {
            k: cross_gate_values[k] for k in sorted(cross_gate_values, key=lambda x: int(x))
        },
        "speaker_cross_attn_gate_summary": cross_gate_summary,
        "speaker_side_max_grad_norm_by_layer": {
            group: {k: float(values[k]) for k in sorted(values, key=lambda x: int(x))}
            for group, values in max_grad_norms.items()
        },
        "speaker_side_adaln_drift_by_layer": {
            k: v for k, v in sorted(
                speaker_side_adaln_drift_stats(model, initial_snapshot).items(),
                key=lambda item: int(item[0]),
            )
        },
        "null_speaker_embedding_l2": None,
        "progress_stop_aux_stats_last_batch": getattr(model, "last_progress_stop_aux_stats", {}) or {},
    }
    for name, param in model.named_parameters():
        if name.endswith("null_speaker_embedding"):
            payload["null_speaker_embedding_l2"] = float(torch.linalg.vector_norm(param.detach().float()).cpu().item())
            break
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ver2_9_smoke_train_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)


def source_semantic_gate_delta_stats(
    model: torch.nn.Module,
    initial_snapshot: dict[str, torch.Tensor],
) -> dict[str, float]:
    deltas = []
    logit_deltas = []
    for name, param in model.named_parameters():
        if name not in initial_snapshot:
            continue
        current = param.detach().float().cpu()
        initial = initial_snapshot[name].float()
        deltas.append((torch.sigmoid(current) - torch.sigmoid(initial)).abs().reshape(-1))
        logit_deltas.append((current - initial).abs().reshape(-1))
    if not deltas:
        return {}
    delta = torch.cat(deltas)
    logit_delta = torch.cat(logit_deltas)
    return {
        "source_semantic_gate_delta_mean": float(delta.mean().item()),
        "source_semantic_gate_delta_max": float(delta.max().item()),
        "source_semantic_gate_logit_delta_max": float(logit_delta.max().item()),
    }


def routing_gate_delta_stats(model: torch.nn.Module, initial_snapshot: dict[str, torch.Tensor]) -> dict[str, float]:
    stats: dict[str, float] = {}
    role_logits = None
    role_initial = None
    prosody_logits = None
    prosody_initial = None
    timbre_logits = None
    timbre_initial = None
    for name, param in model.named_parameters():
        if name not in initial_snapshot:
            continue
        current = param.detach().float().cpu()
        initial = initial_snapshot[name].float()
        if name.endswith("role_router.gate_logits"):
            role_logits = current
            role_initial = initial
        elif name.endswith("target_head_router.prosody_gate_logits"):
            prosody_logits = current
            prosody_initial = initial
        elif name.endswith("target_head_router.timbre_gate_logits"):
            timbre_logits = current
            timbre_initial = initial

    if role_logits is not None and role_initial is not None:
        delta = (torch.sigmoid(role_logits) - torch.sigmoid(role_initial)).abs()
        stats["role_gate_delta_max"] = float(delta.max().item())
        if delta.shape[0] > TARGET_CODEC:
            stats["source_gate_delta_mean"] = float(delta[SOURCE_CODEC].mean().item())
            stats["ref_gate_delta_mean"] = float(delta[REF_CODEC].mean().item())
            stats["target_gate_delta_mean"] = float(delta[TARGET_CODEC].mean().item())
        stats["role_gate_logit_delta_max"] = float((role_logits - role_initial).abs().max().item())
    if prosody_logits is not None and prosody_initial is not None:
        delta = (torch.sigmoid(prosody_logits) - torch.sigmoid(prosody_initial)).abs()
        stats["prosody_head_gate_delta_mean"] = float(delta.mean().item())
        stats["prosody_head_gate_logit_delta_max"] = float((prosody_logits - prosody_initial).abs().max().item())
    if timbre_logits is not None and timbre_initial is not None:
        delta = (torch.sigmoid(timbre_logits) - torch.sigmoid(timbre_initial)).abs()
        stats["timbre_head_gate_delta_mean"] = float(delta.mean().item())
        stats["timbre_head_gate_logit_delta_max"] = float((timbre_logits - timbre_initial).abs().max().item())
    return stats


def set_speaker_aux_weights(model: torch.nn.Module, speaker_weight: float, source_suppression_weight: float) -> None:
    config = getattr(model, "timbre_memory_config", None)
    if config is None:
        return
    config.target_speaker_similarity_weight = float(speaker_weight)
    config.source_speaker_suppression_weight = float(source_suppression_weight)


def scheduled_speaker_aux_weights(
    *,
    step: int,
    final_speaker_weight: float,
    final_source_suppression_weight: float,
    warmup_steps: int,
    warmup_weight: float,
    schedule: str = "step",
) -> tuple[float, float, bool]:
    if int(warmup_steps) <= 0 or int(step) >= int(warmup_steps):
        return float(final_speaker_weight), float(final_source_suppression_weight), False
    start_weight = max(0.0, float(warmup_weight))
    mode = str(schedule or "step").strip().lower()
    if mode == "cosine":
        progress = max(0.0, min(1.0, float(step) / max(1.0, float(warmup_steps))))
        alpha = 0.5 - 0.5 * math.cos(math.pi * progress)

        def ramp(final_weight: float) -> float:
            if final_weight <= 0:
                return 0.0
            start = min(start_weight, float(final_weight))
            return start + (float(final_weight) - start) * alpha

        speaker_weight = ramp(float(final_speaker_weight))
        source_suppression_weight = ramp(float(final_source_suppression_weight))
    else:
        speaker_weight = min(start_weight, float(final_speaker_weight)) if final_speaker_weight > 0 else 0.0
        source_suppression_weight = (
            min(start_weight, float(final_source_suppression_weight))
            if final_source_suppression_weight > 0
            else 0.0
        )
    return speaker_weight, source_suppression_weight, True


TIMBRE_OPTIONAL_BATCH_KEYS = (
    "source_logf0",
    "source_logf0_mask",
    "source_voiced_mask",
    "source_voiced_mask_mask",
    "source_energy",
    "source_energy_mask",
    "source_pause_mask",
    "source_pause_mask_mask",
    "source_duration",
    "source_duration_mask",
    "target_logf0",
    "target_logf0_mask",
    "target_voiced_mask",
    "target_voiced_mask_mask",
    "target_energy",
    "target_energy_mask",
    "target_pause_mask",
    "target_pause_mask_mask",
    "target_duration",
    "target_duration_mask",
    "source_content_embedding",
    "source_content_embedding_mask",
    "target_content_embedding",
    "target_content_embedding_mask",
    "source_content_ids",
    "source_content_ids_mask",
    "target_content_ids",
    "target_content_ids_mask",
    "content_token_ids",
    "content_token_ids_mask",
    "source_semantic_units",
    "source_semantic_units_mask",
    "target_semantic_units",
    "target_semantic_units_mask",
    "source_semantic_features",
    "source_semantic_features_mask",
    "target_semantic_features",
    "target_semantic_features_mask",
)

AUX_LOSS_ATTRS = {
    "speaker_aux_loss": "last_speaker_aux_loss",
    "prosody_aux_loss": "last_prosody_aux_loss",
    "content_aux_loss": "last_content_aux_loss",
    "content_ctc_aux_loss": "last_content_ctc_aux_loss",
    "semantic_aux_loss": "last_semantic_aux_loss",
    "source_semantic_aux_loss": "last_source_semantic_aux_loss",
    "ref_content_suppression_loss": "last_ref_content_suppression_loss",
    "progress_stop_aux_loss": "last_progress_stop_aux_loss",
    "route_loss": "last_route_loss",
}


def build_forward_kwargs_from_batch(
    batch: dict[str, Any],
    *,
    timbre_memory_enabled: bool,
    channelwise_loss_weight: list[float] | None,
) -> dict[str, Any]:
    forward_kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"],
        "channelwise_loss_weight": channelwise_loss_weight,
    }
    if not timbre_memory_enabled:
        return forward_kwargs

    forward_kwargs.update(
        {
            "source_ref_codes": batch.get("source_ref_codes"),
            "source_ref_mask": batch.get("source_ref_mask"),
            "timbre_ref_codes": batch["timbre_ref_codes"],
            "timbre_ref_mask": batch["timbre_ref_mask"],
            "target_position_mask": batch["target_assistant_positions"],
            "source_prompt_positions": batch["source_prompt_positions"],
            "timbre_ref_prompt_positions": batch["timbre_ref_prompt_positions"],
            "ref_speaker_prompt_slot_positions": batch.get("ref_speaker_prompt_slot_positions"),
            "role_ids": batch.get("role_ids"),
            "vc_mode_id": batch["vc_mode_id"],
            "source_speaker_embedding_path": batch.get("source_speaker_embedding_path"),
            "timbre_ref_speaker_embedding_path": batch.get("timbre_ref_speaker_embedding_path"),
            "target_speaker_embedding_path": batch.get("target_speaker_embedding_path"),
            "speaker_vec_path": batch.get("speaker_vec_path"),
            "speaker_seq_path": batch.get("speaker_seq_path"),
            "speaker_seq_features": batch.get("speaker_seq_features"),
            "speaker_seq_features_mask": batch.get("speaker_seq_features_mask"),
            "source_speaker_audio_path": batch.get("source_speaker_audio_path"),
            "timbre_ref_speaker_audio_path": batch.get("timbre_ref_speaker_audio_path"),
            "target_speaker_audio_path": batch.get("target_speaker_audio_path"),
        }
    )
    for optional_key in TIMBRE_OPTIONAL_BATCH_KEYS:
        if optional_key in batch:
            forward_kwargs[optional_key] = batch[optional_key]
    return forward_kwargs


def scalar_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            return float(value.detach().float().mean().item())
        return float(value)
    except (TypeError, ValueError, RuntimeError):
        return None


def collect_aux_loss_scalars(model: torch.nn.Module) -> dict[str, float | None]:
    values = {name: scalar_float(getattr(model, attr, None)) for name, attr in AUX_LOSS_ATTRS.items()}
    for attr in (
        "last_speaker_aux_stats",
        "last_prosody_aux_stats",
        "last_content_aux_stats",
        "last_content_ctc_aux_stats",
        "last_semantic_aux_stats",
        "last_source_semantic_aux_stats",
        "last_ref_content_suppression_stats",
        "last_progress_stop_aux_stats",
        "last_route_stats",
        "last_speaker_side_stats",
    ):
        stats = getattr(model, attr, None) or {}
        if not isinstance(stats, dict):
            continue
        for key, value in stats.items():
            values[str(key)] = scalar_float(value)
    return values


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    moss_root = deep_get(cfg, "moss.root")
    model_path = args.model_path or deep_get(cfg, "moss.model_path")
    codec_path = args.codec_path or deep_get(cfg, "moss.codec_path")
    n_vq = args.n_vq or int(deep_get(cfg, "training.n_vq", deep_get(cfg, "moss.default_n_vq", 32)))
    timbre_memory_config = resolve_timbre_memory_config(cfg, args)
    final_target_speaker_similarity_weight = float(timbre_memory_config.target_speaker_similarity_weight)
    final_source_speaker_suppression_weight = float(timbre_memory_config.source_speaker_suppression_weight)
    speaker_loss_warmup_steps = (
        int(args.speaker_loss_warmup_steps)
        if args.speaker_loss_warmup_steps is not None
        else int(deep_get(cfg, "loss.speaker_loss_warmup_steps", 1000))
    )
    speaker_loss_warmup_weight = (
        float(args.speaker_loss_warmup_weight)
        if args.speaker_loss_warmup_weight is not None
        else float(deep_get(cfg, "loss.speaker_loss_warmup_weight", 0.02))
    )
    speaker_loss_schedule = str(
        args.speaker_loss_schedule
        if args.speaker_loss_schedule is not None
        else deep_get(cfg, "loss.speaker_loss_schedule", "step")
    ).strip().lower()
    if speaker_loss_schedule not in {"step", "cosine"}:
        raise ValueError(f"unsupported speaker_loss_schedule: {speaker_loss_schedule!r}")
    if final_target_speaker_similarity_weight <= 0 and final_source_speaker_suppression_weight <= 0:
        speaker_loss_warmup_steps = 0
    if args.gradient_checkpointing and timbre_memory_config.enabled:
        print(
            "[gradient_checkpointing] disabled for timbre-memory Ver2 because TTA layer hooks "
            "are not compatible with checkpoint recomputation.",
            flush=True,
        )
        args.gradient_checkpointing = False
    args.use_timbre_memory = bool(
        int(timbre_memory_config.num_memory_tokens) > 0 and str(timbre_memory_config.adapter_layers).strip()
    )
    args.timbre_memory_tokens = timbre_memory_config.num_memory_tokens
    args.timbre_adapter_layers = timbre_memory_config.adapter_layers
    args.timbre_adapter_heads = timbre_memory_config.num_heads
    args.timbre_adapter_dim = timbre_memory_config.adapter_dim
    args.timbre_adapter_dropout = timbre_memory_config.dropout
    args.timbre_adapter_init_gate = timbre_memory_config.init_gate
    args.timbre_encoder_type = timbre_memory_config.encoder_type
    args.timbre_encoder_layers = timbre_memory_config.encoder_layers
    args.timbre_conformer_kernel_size = timbre_memory_config.conv_kernel_size
    args.timbre_speaker_conditioning = timbre_memory_config.speaker_conditioning
    args.speaker_embedding_dim = timbre_memory_config.speaker_embedding_dim
    args.speaker_loss_margin = timbre_memory_config.speaker_loss_margin
    args.speaker_encoder_type = timbre_memory_config.speaker_encoder_type
    args.speaker_encoder_path = timbre_memory_config.speaker_encoder_path
    args.freeze_speaker_encoder = timbre_memory_config.freeze_speaker_encoder
    args.target_speaker_similarity_weight = timbre_memory_config.target_speaker_similarity_weight
    args.source_speaker_suppression_weight = timbre_memory_config.source_speaker_suppression_weight
    args.ref_speaker_prompt_tokens = timbre_memory_config.ref_speaker_prompt_tokens
    args.ref_speaker_prompt_dropout = timbre_memory_config.ref_speaker_prompt_dropout
    args.ref_speaker_prompt_mode = timbre_memory_config.ref_speaker_prompt_mode
    args.ref_speaker_prompt_token_source = timbre_memory_config.ref_speaker_prompt_token_source
    args.ref_speaker_prompt_slot = timbre_memory_config.ref_speaker_prompt_slot
    args.ref_speaker_prompt_slot_code = timbre_memory_config.ref_speaker_prompt_slot_code
    args.ref_speaker_prompt_slot_pack_mode = timbre_memory_config.ref_speaker_prompt_slot_pack_mode
    args.ref_speaker_prompt_output_norm = timbre_memory_config.ref_speaker_prompt_output_norm
    args.ref_speaker_prompt_output_scale = timbre_memory_config.ref_speaker_prompt_output_scale
    args.ref_prompt_codec_permutation = timbre_memory_config.ref_prompt_codec_permutation_enabled
    args.ref_prompt_codec_permutation_min_seconds = timbre_memory_config.ref_prompt_codec_permutation_min_seconds
    args.ref_prompt_codec_permutation_max_seconds = timbre_memory_config.ref_prompt_codec_permutation_max_seconds
    args.ref_prompt_codec_permutation_frame_rate = timbre_memory_config.ref_prompt_codec_permutation_frame_rate
    args.ref_prompt_codec_permutation_seed = timbre_memory_config.ref_prompt_codec_permutation_seed
    args.ref_prompt_codec_permutation_mode = timbre_memory_config.ref_prompt_codec_permutation_mode
    args.ref_prompt_codec_permutation_block_seconds = timbre_memory_config.ref_prompt_codec_permutation_block_seconds
    args.target_front_ce_weight = timbre_memory_config.target_front_ce_weight
    args.target_front_ce_seconds = timbre_memory_config.target_front_ce_seconds
    args.target_front_ce_frame_rate = timbre_memory_config.target_front_ce_frame_rate
    args.ref_speaker_adaln_weight = timbre_memory_config.ref_speaker_adaln_weight
    args.speaker_infonce_weight = timbre_memory_config.speaker_infonce_weight
    args.speaker_infonce_temperature = timbre_memory_config.speaker_infonce_temperature
    args.speaker_infonce_negative_pool_size = timbre_memory_config.speaker_infonce_negative_pool_size
    args.speaker_infonce_negative_pool_seed = timbre_memory_config.speaker_infonce_negative_pool_seed
    args.speaker_condition_dropout = timbre_memory_config.speaker_condition_dropout
    args.enable_speaker_side_pathway = timbre_memory_config.speaker_side_pathway_enabled
    args.speaker_side_pathway_layers = timbre_memory_config.speaker_side_pathway_layers
    args.speaker_side_pathway_kv_bias = timbre_memory_config.speaker_side_pathway_kv_bias
    args.speaker_side_pathway_gate_init = timbre_memory_config.speaker_side_pathway_gate_init
    args.speaker_side_pathway_dropout = timbre_memory_config.speaker_side_pathway_dropout
    args.enable_speaker_cross_attn = timbre_memory_config.speaker_cross_attn_enabled
    args.speaker_cross_attn_layers = timbre_memory_config.speaker_cross_attn_layers
    args.speaker_cross_attn_tokens = timbre_memory_config.speaker_cross_attn_tokens
    args.speaker_cross_attn_gate_init = timbre_memory_config.speaker_cross_attn_gate_init
    args.speaker_cross_attn_dropout = timbre_memory_config.speaker_cross_attn_dropout
    args.speaker_cross_attn_output_scale = timbre_memory_config.speaker_cross_attn_output_scale
    args.speaker_cross_attn_token_init_std = timbre_memory_config.speaker_cross_attn_token_init_std
    args.speaker_cross_attn_alpha_warmup_steps = timbre_memory_config.speaker_cross_attn_alpha_warmup_steps
    args.speaker_cross_attn_source = timbre_memory_config.speaker_cross_attn_source
    args.speaker_cross_attn_seq_dim = timbre_memory_config.speaker_cross_attn_seq_dim
    args.use_perturbed_source_prompt = timbre_memory_config.use_perturbed_source_prompt
    args.speaker_loss_warmup_steps = speaker_loss_warmup_steps
    args.speaker_loss_warmup_weight = speaker_loss_warmup_weight
    args.enable_role_routing = timbre_memory_config.use_role_routing
    args.enable_target_head_routing = timbre_memory_config.target_head_routing
    args.lambda_route = timbre_memory_config.route_loss_weight
    args.lambda_prosody = timbre_memory_config.prosody_loss_weight
    args.lambda_content = timbre_memory_config.content_loss_weight
    args.prosody_f0_weight = timbre_memory_config.prosody_f0_weight
    args.prosody_voiced_weight = timbre_memory_config.prosody_voiced_weight
    args.prosody_energy_weight = timbre_memory_config.prosody_energy_weight
    args.prosody_pause_weight = timbre_memory_config.prosody_pause_weight
    args.prosody_duration_weight = timbre_memory_config.prosody_duration_weight
    args.prosody_normalize_f0 = timbre_memory_config.prosody_normalize_f0
    args.prosody_normalize_energy = timbre_memory_config.prosody_normalize_energy
    args.content_embedding_dim = timbre_memory_config.content_embedding_dim
    args.content_positive = timbre_memory_config.content_positive
    args.content_embedding_weight = timbre_memory_config.content_embedding_weight
    args.content_ctc_weight = timbre_memory_config.content_ctc_weight
    args.content_ctc_vocab_size = timbre_memory_config.content_ctc_vocab_size
    args.content_ctc_blank_id = timbre_memory_config.content_ctc_blank_id
    args.content_ctc_token_offset = timbre_memory_config.content_ctc_token_offset
    args.content_token_vocab_size = timbre_memory_config.content_token_vocab_size
    args.content_token_weight = timbre_memory_config.content_token_weight
    args.content_source_codec_weight = timbre_memory_config.content_source_codec_weight
    args.content_source_codec_codebooks = timbre_memory_config.content_source_codec_codebooks
    args.semantic_loss_weight = timbre_memory_config.semantic_loss_weight
    args.semantic_mode = timbre_memory_config.semantic_mode
    args.semantic_source = timbre_memory_config.semantic_source
    args.semantic_vocab_size = timbre_memory_config.semantic_vocab_size
    args.semantic_feature_dim = timbre_memory_config.semantic_feature_dim
    args.semantic_feature_loss_type = timbre_memory_config.semantic_feature_loss_type
    args.progress_loss_weight = timbre_memory_config.progress_loss_weight
    args.stop_loss_weight = timbre_memory_config.stop_loss_weight
    args.progress_num_bins = timbre_memory_config.progress_num_bins
    args.enable_source_semantic_memory = timbre_memory_config.source_semantic_memory_enabled
    args.source_semantic_feature_dim = timbre_memory_config.source_semantic_feature_dim
    args.source_semantic_adapter_layers = timbre_memory_config.source_semantic_adapter_layers
    args.source_semantic_no_text_gate = timbre_memory_config.source_semantic_no_text_gate
    args.source_semantic_text_gate = timbre_memory_config.source_semantic_text_gate
    args.source_semantic_learned_text_gate = timbre_memory_config.source_semantic_allow_learned_text_gate
    args.source_semantic_progress_weight = timbre_memory_config.source_semantic_progress_weight
    args.source_semantic_dropout = timbre_memory_config.source_semantic_dropout
    args.source_semantic_init_gate = timbre_memory_config.source_semantic_init_gate
    args.source_semantic_position_scale = timbre_memory_config.source_semantic_position_scale
    args.source_semantic_monotonic_bias_strength = timbre_memory_config.source_semantic_monotonic_bias_strength
    args.source_semantic_monotonic_bias_width = timbre_memory_config.source_semantic_monotonic_bias_width
    args.source_prosody_no_text_gate = timbre_memory_config.source_prosody_no_text_gate
    args.source_prosody_text_gate = timbre_memory_config.source_prosody_text_gate
    args.source_content_memory_type = timbre_memory_config.source_content_memory_type
    args.source_content_vocab_size = timbre_memory_config.source_content_vocab_size
    args.source_content_padding_id = timbre_memory_config.source_content_padding_id
    args.source_content_codec_bottleneck_dim = timbre_memory_config.source_content_codec_bottleneck_dim
    args.source_content_codec_codebooks = timbre_memory_config.source_content_codec_codebooks
    args.source_content_dedup_units = timbre_memory_config.source_content_dedup_units
    args.source_codec_residual_memory_weight = timbre_memory_config.source_codec_residual_memory_weight
    args.source_codec_residual_memory_detach = timbre_memory_config.source_codec_residual_memory_detach
    args.ref_content_suppression_weight = timbre_memory_config.ref_content_suppression_weight
    args.ref_content_suppression_margin = timbre_memory_config.ref_content_suppression_margin
    args.ref_content_suppression_source = timbre_memory_config.ref_content_suppression_source
    args.ref_content_suppression_detach_ref = timbre_memory_config.ref_content_suppression_detach_ref
    args.prosody_memory_tokens = timbre_memory_config.prosody_memory_tokens
    args.source_prosody_encoder_type = timbre_memory_config.source_prosody_encoder_type
    args.source_prosody_encoder_layers = timbre_memory_config.source_prosody_encoder_layers
    args.source_prosody_conv_kernel_size = timbre_memory_config.source_prosody_conv_kernel_size
    if moss_root and str(moss_root) not in sys.path:
        sys.path.insert(0, str(moss_root))

    from moss_tts_delay.finetuning.dataset import MossTTSSFTDataset
    from moss_tts_delay.modeling_moss_tts import MossTTSDelayModel

    class LazyMossTTSSFTDataset(MossTTSSFTDataset):
        def __init__(self, records, processor, n_vq=None) -> None:
            self.records = records
            self.processor = processor
            self.n_vq = n_vq
            self._audio_cache = {}

    torch.manual_seed(args.seed)
    processor = build_processor(model_path, codec_path, moss_root)
    if int(processor.model_config.n_vq) != int(n_vq):
        raise ValueError(f"n_vq mismatch: requested {n_vq}, model config has {processor.model_config.n_vq}")

    records = load_training_records(args)
    source_content_memory_type = str(timbre_memory_config.source_content_memory_type or "hubert_continuous").strip().lower()
    source_content_uses_content_tokens = source_content_memory_type == "text_tokens"
    validate_content_tokenizer_consistency(
        records,
        enabled=timbre_memory_config.content_ctc_weight > 0 or source_content_uses_content_tokens,
        allow_mixed=bool(args.allow_mixed_content_tokenizers),
    )
    manifest_content_vocab_size = (
        infer_content_ctc_vocab_size_from_manifest(records)
        if timbre_memory_config.content_ctc_weight > 0 or source_content_uses_content_tokens
        else 0
    )
    content_ctc_uses_manifest_tokens = manifest_content_vocab_size > 1
    content_tokenizer = None
    content_tokenizer_path = (
        infer_sentencepiece_content_tokenizer_path(records)
        if timbre_memory_config.content_ctc_weight > 0 or source_content_uses_content_tokens
        else ""
    )
    if content_tokenizer_path:
        content_tokenizer = load_sentencepiece_content_tokenizer(content_tokenizer_path)
    if source_content_uses_content_tokens and timbre_memory_config.source_content_vocab_size <= 1:
        if manifest_content_vocab_size <= 1:
            tokenizer_vocab_size = content_tokenizer_vocab_size(content_tokenizer)
            if tokenizer_vocab_size <= 1:
                raise ValueError(
                    "source_content_memory_type=text_tokens requires source_content_vocab_size > 1, "
                    "manifest content_ctc_vocab_size/content_token_ids metadata, or a valid sentencepiece content_vocab_path."
                )
            timbre_memory_config.source_content_vocab_size = int(tokenizer_vocab_size) + int(
                timbre_memory_config.content_ctc_token_offset
            )
        else:
            timbre_memory_config.source_content_vocab_size = int(manifest_content_vocab_size)
        args.source_content_vocab_size = int(timbre_memory_config.source_content_vocab_size)
        print(
            "[source_content_memory] auto_vocab_size="
            f"{timbre_memory_config.source_content_vocab_size} padding_id={timbre_memory_config.source_content_padding_id}",
            flush=True,
        )
    if source_content_memory_type == "semantic_units" and timbre_memory_config.source_content_vocab_size <= 1:
        raise ValueError(
            "source_content_memory_type=semantic_units requires --source-content-vocab-size. "
            "The current manifests do not expose a reliable unit vocab size."
        )
    if timbre_memory_config.content_ctc_weight > 0 and timbre_memory_config.content_ctc_vocab_size <= 1:
        if manifest_content_vocab_size > 1:
            timbre_memory_config.content_ctc_vocab_size = int(manifest_content_vocab_size)
            print(
                "[content_ctc] auto_vocab_size_from_manifest="
                f"{timbre_memory_config.content_ctc_vocab_size} blank_id={timbre_memory_config.content_ctc_blank_id}",
                flush=True,
            )
        else:
            tokenizer_vocab_size = len(processor.tokenizer)
            timbre_memory_config.content_ctc_vocab_size = int(tokenizer_vocab_size) + int(
                timbre_memory_config.content_ctc_token_offset
            )
            print(
                "[content_ctc] auto_vocab_size_from_moss_tokenizer="
                f"{timbre_memory_config.content_ctc_vocab_size} tokenizer_vocab_size={tokenizer_vocab_size} "
                f"blank_id={timbre_memory_config.content_ctc_blank_id} "
                f"token_offset={timbre_memory_config.content_ctc_token_offset}",
                flush=True,
            )
        if timbre_memory_config.content_ctc_vocab_size <= int(timbre_memory_config.content_ctc_blank_id):
            raise ValueError(
                    "content_ctc_vocab_size must be greater than content_ctc_blank_id; "
                    f"got vocab={timbre_memory_config.content_ctc_vocab_size} blank={timbre_memory_config.content_ctc_blank_id}"
                )
    if content_tokenizer is None and timbre_memory_config.content_ctc_weight > 0 and not content_ctc_uses_manifest_tokens:
        content_tokenizer = processor.tokenizer
    def build_sft_dataset(records_obj):
        base = LazyMossTTSSFTDataset(records=records_obj, processor=processor, n_vq=n_vq)
        if timbre_memory_config.enabled:
            return MossCodecVCTimbreSFTDataset(
                records=records_obj,
                base_dataset=base,
                n_vq=n_vq,
                audio_pad_code=int(processor.model_config.audio_pad_code),
                content_tokenizer=content_tokenizer,
                content_ctc_token_offset=timbre_memory_config.content_ctc_token_offset,
                timbre_side_only=bool(args.timbre_side_only),
                use_perturbed_source_prompt=bool(timbre_memory_config.use_perturbed_source_prompt),
                ref_speaker_prompt_slot=bool(timbre_memory_config.ref_speaker_prompt_slot),
                ref_speaker_prompt_tokens=int(timbre_memory_config.ref_speaker_prompt_tokens),
                ref_speaker_prompt_slot_code=int(timbre_memory_config.ref_speaker_prompt_slot_code),
                ref_speaker_prompt_slot_pack_mode=str(timbre_memory_config.ref_speaker_prompt_slot_pack_mode),
                ref_prompt_codec_permutation_enabled=bool(timbre_memory_config.ref_prompt_codec_permutation_enabled),
                ref_prompt_codec_permutation_min_seconds=float(
                    timbre_memory_config.ref_prompt_codec_permutation_min_seconds
                ),
                ref_prompt_codec_permutation_max_seconds=float(
                    timbre_memory_config.ref_prompt_codec_permutation_max_seconds
                ),
                ref_prompt_codec_permutation_frame_rate=float(
                    timbre_memory_config.ref_prompt_codec_permutation_frame_rate
                ),
                ref_prompt_codec_permutation_mode=str(timbre_memory_config.ref_prompt_codec_permutation_mode),
                ref_prompt_codec_permutation_block_seconds=float(
                    timbre_memory_config.ref_prompt_codec_permutation_block_seconds
                ),
                speaker_side_pathway_enabled=bool(timbre_memory_config.speaker_side_pathway_enabled),
            )
        return base

    dataset = build_sft_dataset(records)
    eval_record_sets = load_named_eval_records(args)
    eval_datasets: list[tuple[str, Any]] = []
    for eval_label, eval_records in eval_record_sets:
        validate_content_tokenizer_consistency(
            eval_records,
            enabled=timbre_memory_config.content_ctc_weight > 0,
            allow_mixed=bool(args.allow_mixed_content_tokenizers),
        )
        eval_datasets.append((eval_label, build_sft_dataset(eval_records)))
    probe = dataset.collate_fn([dataset[0]])
    if (
        timbre_memory_config.enabled
        and timbre_memory_config.semantic_loss_weight > 0
        and str(timbre_memory_config.semantic_mode).strip().lower() == "continuous"
        and int(timbre_memory_config.semantic_feature_dim) <= 0
    ):
        inferred_dim = infer_semantic_feature_dim_from_probe(
            probe,
            semantic_source=timbre_memory_config.semantic_source,
        )
        if inferred_dim <= 0:
            raise ValueError(
                "semantic_mode=continuous with semantic_loss_weight > 0 requires precomputed "
                "source/target semantic features in the training JSONL or an explicit --semantic-feature-dim. "
                "Run scripts/001020_extract_hubert_semantic_features.py first."
            )
        timbre_memory_config.semantic_feature_dim = int(inferred_dim)
        args.semantic_feature_dim = int(inferred_dim)
        print(
            f"[semantic_loss] auto_feature_dim_from_batch={inferred_dim} "
            f"source={timbre_memory_config.semantic_source}",
            flush=True,
        )
    print(
        f"[pack] records={len(records)} input_ids={tuple(probe['input_ids'].shape)} "
        f"labels={tuple(probe['labels'].shape)} valid_labels={int((probe['labels'] != -100).sum().item())}"
    )
    if timbre_memory_config.enabled:
        print(
            f"[pack] timbre_ref_codes={tuple(probe['timbre_ref_codes'].shape)} "
            f"target_positions={int(probe['target_assistant_positions'].sum().item())} "
            f"slot_positions={int(probe.get('ref_speaker_prompt_slot_positions', torch.zeros(1)).sum().item())} "
            f"mode_ids={probe['vc_mode_id'].tolist()}"
        )
        print(
            "[pack] speaker_prompt="
            f"tokens={timbre_memory_config.ref_speaker_prompt_tokens} "
            f"mode={timbre_memory_config.ref_speaker_prompt_mode} "
            f"source={timbre_memory_config.ref_speaker_prompt_token_source} "
            f"slot={timbre_memory_config.ref_speaker_prompt_slot} "
            f"slot_code={timbre_memory_config.ref_speaker_prompt_slot_code} "
            f"slot_pack={timbre_memory_config.ref_speaker_prompt_slot_pack_mode} "
            f"output_norm={timbre_memory_config.ref_speaker_prompt_output_norm} "
            f"output_scale={timbre_memory_config.ref_speaker_prompt_output_scale}",
            flush=True,
        )
        print(
            "[pack] source_content_memory="
            f"type={timbre_memory_config.source_content_memory_type} "
            f"vocab={timbre_memory_config.source_content_vocab_size} "
            f"padding_id={timbre_memory_config.source_content_padding_id} "
            f"codec_bottleneck_dim={timbre_memory_config.source_content_codec_bottleneck_dim} "
            f"codec_codebooks={timbre_memory_config.source_content_codec_codebooks} "
            f"dedup_units={timbre_memory_config.source_content_dedup_units} "
            f"codec_residual_weight={timbre_memory_config.source_codec_residual_memory_weight} "
            f"codec_residual_detach={timbre_memory_config.source_codec_residual_memory_detach}",
            flush=True,
        )
        print(
            "[pack] timbre_side_only="
            f"{bool(args.timbre_side_only)} "
            f"ref_content_suppression_weight={timbre_memory_config.ref_content_suppression_weight} "
            f"margin={timbre_memory_config.ref_content_suppression_margin} "
            f"source={timbre_memory_config.ref_content_suppression_source}",
            flush=True,
        )
        if bool(timbre_memory_config.ref_prompt_codec_permutation_enabled):
            prompt_perm = probe.get("timbre_ref_prompt_permutation")
            prompt_perm_values = prompt_perm.tolist() if torch.is_tensor(prompt_perm) else None
            print(
                "[pack] ref_prompt_codec_permutation="
                f"enabled={timbre_memory_config.ref_prompt_codec_permutation_enabled} "
                f"seconds={timbre_memory_config.ref_prompt_codec_permutation_min_seconds:.3f}-"
                f"{timbre_memory_config.ref_prompt_codec_permutation_max_seconds:.3f} "
                f"frame_rate={timbre_memory_config.ref_prompt_codec_permutation_frame_rate:.3f} "
                f"mode={timbre_memory_config.ref_prompt_codec_permutation_mode} "
                f"block_seconds={timbre_memory_config.ref_prompt_codec_permutation_block_seconds:.3f} "
                f"probe={prompt_perm_values}",
                flush=True,
            )
        print(
            "[pack] source_prosody_gates="
            f"no_text={timbre_memory_config.source_prosody_no_text_gate} "
            f"text={timbre_memory_config.source_prosody_text_gate}",
            flush=True,
        )
        feature_shapes = {}
        if "role_ids" in probe:
            role_counts = count_roles(probe["role_ids"]).as_dict()
            prompt_has_labels = bool((probe["labels"][probe["role_ids"] != TARGET_CODEC] != -100).any().item())
            target_valid_labels = int((probe["labels"][probe["role_ids"] == TARGET_CODEC] != -100).sum().item())
            print(
                f"[pack] role_ids={tuple(probe['role_ids'].shape)} role_counts={role_counts} "
                f"prompt_has_labels={prompt_has_labels} target_valid_labels={target_valid_labels}"
            )
            feature_shapes = {
                key: tuple(value.shape)
                for key, value in probe.items()
                if key.startswith(
                    (
                        "source_logf0",
                        "source_voiced",
                        "source_energy",
                        "source_pause",
                        "source_duration",
                        "target_logf0",
                        "target_voiced",
                        "target_energy",
                        "target_pause",
                        "target_duration",
                        "source_content",
                        "content_token",
                        "source_semantic",
                        "target_semantic",
                    )
                )
                and torch.is_tensor(value)
            }
        if feature_shapes:
            print(f"[pack] auxiliary_features={feature_shapes}")
    for eval_label, eval_dataset in eval_datasets:
        eval_probe = eval_dataset.collate_fn([eval_dataset[0]])
        print(
            f"[eval-pack:{eval_label}] records={len(eval_dataset)} input_ids={tuple(eval_probe['input_ids'].shape)} "
            f"labels={tuple(eval_probe['labels'].shape)} valid_labels={int((eval_probe['labels'] != -100).sum().item())}"
        )
        if timbre_memory_config.enabled:
            print(
                f"[eval-pack:{eval_label}] timbre_ref_codes={tuple(eval_probe['timbre_ref_codes'].shape)} "
                f"target_positions={int(eval_probe['target_assistant_positions'].sum().item())} "
                f"slot_positions={int(eval_probe.get('ref_speaker_prompt_slot_positions', torch.zeros(1)).sum().item())} "
                f"mode_ids={eval_probe['vc_mode_id'].tolist()}"
            )
            print(
                f"[eval-pack:{eval_label}] speaker_prompt="
                f"tokens={timbre_memory_config.ref_speaker_prompt_tokens} "
                f"mode={timbre_memory_config.ref_speaker_prompt_mode} "
                f"source={timbre_memory_config.ref_speaker_prompt_token_source} "
                f"slot={timbre_memory_config.ref_speaker_prompt_slot} "
                f"slot_code={timbre_memory_config.ref_speaker_prompt_slot_code} "
                f"slot_pack={timbre_memory_config.ref_speaker_prompt_slot_pack_mode} "
                f"output_norm={timbre_memory_config.ref_speaker_prompt_output_norm} "
                f"output_scale={timbre_memory_config.ref_speaker_prompt_output_scale}",
                flush=True,
            )

    if args.pack_only:
        print("[pack] ok")
        return 0

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    model_dtype = resolve_dtype(args.mixed_precision)
    attn_implementation = resolve_attn_implementation(args.attn_implementation, model_dtype)
    model = MossTTSDelayModel.from_pretrained(
        model_path,
        torch_dtype=model_dtype,
        attn_implementation=attn_implementation,
    )
    model, trainable = apply_lora(model, args)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)
    if timbre_memory_config.enabled:
        resume_adapter_path = Path(args.resume_adapter_path).expanduser() if args.resume_adapter_path else None
        if resume_adapter_path is not None and (resume_adapter_path / "timbre_memory_adapter.pt").exists():
            model = MossCodecVCTimbreMemoryWrapper.from_pretrained_timbre_memory(
                model,
                resume_adapter_path,
                map_location="cpu",
                config_overrides=timbre_memory_resume_overrides(timbre_memory_config),
            )
            if accelerator.is_main_process:
                active_cfg = model.timbre_memory_config
                print(
                    "[resume] timbre_memory_config_overrides "
                    f"content_ctc={active_cfg.content_ctc_weight:.4f} "
                    f"semantic={active_cfg.semantic_loss_weight:.4f} "
                    f"prosody={active_cfg.prosody_loss_weight:.4f} "
                    f"progress={active_cfg.progress_loss_weight:.4f} "
                    f"stop={active_cfg.stop_loss_weight:.4f} "
                    f"target_spk={active_cfg.target_speaker_similarity_weight:.4f} "
                    f"source_suppress={active_cfg.source_speaker_suppression_weight:.4f}",
                    flush=True,
                )
        else:
            model = MossCodecVCTimbreMemoryWrapper(model, timbre_memory_config)
        if resume_adapter_path is not None and isinstance(model, MossCodecVCTimbreMemoryWrapper):
            model.peft_adapter_fallback_directory = resume_adapter_path
        if (
            isinstance(model, MossCodecVCTimbreMemoryWrapper)
            and float(timbre_memory_config.speaker_infonce_weight) > 0.0
            and int(timbre_memory_config.speaker_infonce_negative_pool_size) > 0
        ):
            negative_paths = sample_timbre_ref_speaker_embedding_paths(
                records,
                pool_size=int(timbre_memory_config.speaker_infonce_negative_pool_size),
                seed=int(timbre_memory_config.speaker_infonce_negative_pool_seed),
            )
            pool_stats = model.build_speaker_infonce_negative_pool(
                negative_paths,
                pool_size=int(timbre_memory_config.speaker_infonce_negative_pool_size),
                seed=int(timbre_memory_config.speaker_infonce_negative_pool_seed),
            )
            if accelerator.is_main_process:
                print(
                    "[speaker_infonce_negative_pool] "
                    f"sampled_paths={len(negative_paths)} "
                    f"pool_size={int(pool_stats.get('speaker_infonce_negative_pool_size', 0.0))} "
                    f"dim={int(pool_stats.get('speaker_infonce_negative_pool_dim', 0.0))} "
                    f"seed={int(timbre_memory_config.speaker_infonce_negative_pool_seed)}",
                    flush=True,
                )
        trainable = collect_trainable_parameters(model)
    if model_dtype != torch.float32:
        cast_floating_state(model, model_dtype)
        trainable = collect_trainable_parameters(model)
    freeze_summary = apply_ver25_freeze_controls(model, args)
    trainable = collect_trainable_parameters(model)
    if not trainable:
        raise RuntimeError(f"No trainable parameters remain after Ver2.5 freeze controls: {freeze_summary}")

    fsdp_plugin = getattr(getattr(accelerator, "state", None), "fsdp_plugin", None)
    if fsdp_plugin is not None and getattr(fsdp_plugin, "sync_module_states", False):
        # FSDP with sync_module_states=True requires every parameter/buffer to
        # live on GPU before wrapping. This is especially important for custom
        # modules that are excluded from FSDP wrapping via ignored_modules.
        model.to(accelerator.device)
        if accelerator.is_main_process:
            print(f"[fsdp] moved_model_to_device_for_sync_module_states device={accelerator.device}")

    ignored_modules_for_manual_grad_sync: list[torch.nn.Module] = []
    if fsdp_plugin is not None:
        ignored_modules: list[torch.nn.Module] = []
        if timbre_memory_config.enabled and hasattr(model, "get_fsdp_ignored_modules"):
            ignored_modules.extend(list(model.get_fsdp_ignored_modules()))
        lora_ignored_modules = collect_lora_fsdp_ignored_modules(model)
        ignored_modules.extend(lora_ignored_modules)
        if ignored_modules:
            deduped_ignored_modules: list[torch.nn.Module] = []
            seen_module_ids: set[int] = set()
            for module in ignored_modules:
                module_id = id(module)
                if module_id in seen_module_ids:
                    continue
                seen_module_ids.add(module_id)
                deduped_ignored_modules.append(module)
            fsdp_plugin.ignored_modules = deduped_ignored_modules
            ignored_modules_for_manual_grad_sync = deduped_ignored_modules
            if accelerator.is_main_process:
                ignored_trainable_params = sum(
                    param.numel()
                    for module in deduped_ignored_modules
                    for param in module.parameters(recurse=True)
                    if param.requires_grad
                )
                print(
                    "[fsdp] "
                    f"ignored_modules={len(deduped_ignored_modules)} "
                    f"lora_ignored_modules={len(lora_ignored_modules)} "
                    f"ignored_trainable_params={ignored_trainable_params}"
                )

    train_loader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )
    eval_loaders: list[tuple[str, Any]] = []
    for eval_label, eval_dataset in eval_datasets:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_batch_size,
            shuffle=False,
            num_workers=args.eval_num_workers,
            collate_fn=eval_dataset.collate_fn,
            drop_last=False,
        )
        eval_loaders.append((eval_label, eval_loader))
    optimizer_param_groups, optimizer_group_summary = build_optimizer_param_groups(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        routing_gate_lr_multiplier=args.routing_gate_lr_multiplier,
        source_semantic_lr_multiplier=args.source_semantic_lr_multiplier,
        source_semantic_gate_lr_multiplier=args.source_semantic_gate_lr_multiplier,
        content_ctc_head_lr_multiplier=args.content_ctc_head_lr_multiplier,
        timbre_adapter_gate_lr_multiplier=args.timbre_adapter_gate_lr_multiplier,
        ref_speaker_prompt_lr_multiplier=args.ref_speaker_prompt_lr_multiplier,
    )
    optimizer = AdamW(optimizer_param_groups)

    estimated_global_batch_size = (
        int(args.per_device_batch_size)
        * int(args.gradient_accumulation_steps)
        * max(1, int(accelerator.num_processes))
    )
    update_steps_per_epoch = math.ceil(len(dataset) / max(1, estimated_global_batch_size))
    max_train_steps = args.max_train_steps or args.num_epochs * update_steps_per_epoch
    if args.smoke_test and int(args.max_train_steps or 0) <= 0:
        max_train_steps = min(max_train_steps, 2)
    effective_num_epochs = args.num_epochs
    if max_train_steps > 0 and update_steps_per_epoch > 0:
        effective_num_epochs = max(args.num_epochs, math.ceil(max_train_steps / update_steps_per_epoch))
    warmup_steps = math.ceil(max_train_steps * args.warmup_ratio)
    # Accelerate's prepared scheduler advances once per process for each real
    # optimizer update when split_batches=False. Keep user-facing
    # max_train_steps/warmup_ratio in global optimizer-update units.
    scheduler_step_multiplier = 1 if getattr(accelerator, "split_batches", False) else max(1, int(accelerator.num_processes))
    scheduler_warmup_steps = warmup_steps * scheduler_step_multiplier
    scheduler_training_steps = max_train_steps * scheduler_step_multiplier
    lr_scheduler_type = str(args.lr_scheduler_type or "cosine").strip().lower()
    lr_scheduler = get_scheduler(
        lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=scheduler_warmup_steps,
        num_training_steps=scheduler_training_steps,
    )
    channelwise_loss_weight = parse_channelwise_loss_weight(args.channelwise_loss_weight, n_vq + 1)

    if eval_loaders:
        prepared = accelerator.prepare(
            model,
            optimizer,
            train_loader,
            lr_scheduler,
            *(loader for _, loader in eval_loaders),
        )
        model, optimizer, train_loader, lr_scheduler = prepared[:4]
        eval_loaders = [
            (label, loader)
            for (label, _), loader in zip(eval_loaders, prepared[4:])
        ]
    else:
        model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            model,
            optimizer,
            train_loader,
            lr_scheduler,
        )
    routing_gate_initial_snapshot: dict[str, torch.Tensor] = {}
    source_semantic_gate_initial_snapshot: dict[str, torch.Tensor] = {}
    speaker_side_initial_snapshot: dict[str, torch.Tensor] = {}
    if timbre_memory_config.enabled:
        unwrapped_after_prepare = accelerator.unwrap_model(model)
        routing_gate_initial_snapshot = capture_routing_gate_snapshot(unwrapped_after_prepare)
        source_semantic_gate_initial_snapshot = capture_source_semantic_gate_snapshot(unwrapped_after_prepare)
        if bool(timbre_memory_config.speaker_side_pathway_enabled):
            speaker_side_initial_snapshot = capture_speaker_side_initial_snapshot(unwrapped_after_prepare)

    if accelerator.is_main_process:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_log_handle = (output_dir / "train.log").open("a", encoding="utf-8")
        tb_dir = output_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
        total_trainable = sum(trainable.values())
        print(f"[trainable] trainable_tensors={len(trainable)} trainable_params={total_trainable}")
        print(
            "[freeze] "
            f"train_source_semantic_only={freeze_summary['train_source_semantic_only']} "
            f"freeze_lora={freeze_summary['freeze_lora']} "
            f"freeze_role_routing={freeze_summary['freeze_role_routing']} "
            f"freeze_timbre_adapter={freeze_summary['freeze_timbre_adapter']} "
            f"trainable_before={freeze_summary['trainable_before']} "
            f"trainable_after={freeze_summary['trainable_after']} "
            f"changed_param_count={freeze_summary['changed_param_count']} "
            f"source_semantic_trainable_tensors={freeze_summary['source_semantic_trainable_tensors']} "
            f"source_semantic_trainable_params={freeze_summary['source_semantic_trainable_params']}"
        )
        for name in list(trainable)[:8]:
            print(f"[trainable] {name}")
        print(
            "[optimizer] "
            f"base_tensors={optimizer_group_summary['base_tensors']} "
            f"base_params={optimizer_group_summary['base_params']} "
            f"routing_gate_tensors={optimizer_group_summary['routing_gate_tensors']} "
            f"routing_gate_params={optimizer_group_summary['routing_gate_params']} "
            f"source_semantic_gate_tensors={optimizer_group_summary['source_semantic_gate_tensors']} "
            f"source_semantic_gate_params={optimizer_group_summary['source_semantic_gate_params']} "
            f"timbre_adapter_gate_tensors={optimizer_group_summary['timbre_adapter_gate_tensors']} "
            f"timbre_adapter_gate_params={optimizer_group_summary['timbre_adapter_gate_params']} "
            f"source_semantic_tensors={optimizer_group_summary['source_semantic_tensors']} "
            f"source_semantic_params={optimizer_group_summary['source_semantic_params']} "
            f"content_ctc_head_tensors={optimizer_group_summary['content_ctc_head_tensors']} "
            f"content_ctc_head_params={optimizer_group_summary['content_ctc_head_params']} "
            f"content_ctc_head_lr_multiplier={optimizer_group_summary['content_ctc_head_lr_multiplier']:.4g} "
            f"ref_speaker_prompt_tensors={optimizer_group_summary['ref_speaker_prompt_tensors']} "
            f"ref_speaker_prompt_params={optimizer_group_summary['ref_speaker_prompt_params']} "
            f"ref_speaker_prompt_lr_multiplier={optimizer_group_summary['ref_speaker_prompt_lr_multiplier']:.4g} "
            f"ref_speaker_prompt_lr={args.learning_rate * optimizer_group_summary['ref_speaker_prompt_lr_multiplier']:.4g} "
            f"routing_gate_lr_multiplier={args.routing_gate_lr_multiplier:.4g} "
            f"routing_gate_lr={args.learning_rate * args.routing_gate_lr_multiplier:.4g} "
            f"source_semantic_lr_multiplier={optimizer_group_summary['source_semantic_lr_multiplier']:.4g} "
            f"source_semantic_lr={args.learning_rate * optimizer_group_summary['source_semantic_lr_multiplier']:.4g} "
            f"source_semantic_gate_lr_multiplier={optimizer_group_summary['source_semantic_gate_lr_multiplier']:.4g} "
            f"source_semantic_gate_lr={args.learning_rate * optimizer_group_summary['source_semantic_gate_lr_multiplier']:.4g} "
            f"timbre_adapter_gate_lr_multiplier={optimizer_group_summary['timbre_adapter_gate_lr_multiplier']:.4g} "
            f"timbre_adapter_gate_lr={args.learning_rate * optimizer_group_summary['timbre_adapter_gate_lr_multiplier']:.4g}"
        )
        if routing_gate_initial_snapshot:
            for gate_name, gate_param in routing_gate_parameters(accelerator.unwrap_model(model)).items():
                print(f"[routing_gate] {gate_name} dtype={gate_param.dtype} shape={tuple(gate_param.shape)}")
        if source_semantic_gate_initial_snapshot:
            for gate_name, gate_param in source_semantic_gate_parameters(accelerator.unwrap_model(model)).items():
                print(f"[source_semantic_gate] {gate_name} dtype={gate_param.dtype} shape={tuple(gate_param.shape)}")
        for gate_name, gate_param in timbre_adapter_gate_parameters(accelerator.unwrap_model(model)).items():
            print(f"[timbre_adapter_gate] {gate_name} dtype={gate_param.dtype} shape={tuple(gate_param.shape)}")
        print(f"[trainable] tensorboard_dir={tb_dir}")
        print(
            "[schedule] "
            f"records={len(dataset)} "
            f"num_processes={accelerator.num_processes} "
            f"global_batch_size={estimated_global_batch_size} "
            f"steps_per_epoch={update_steps_per_epoch} "
            f"max_train_steps={max_train_steps} "
            f"effective_num_epochs={effective_num_epochs}"
        )
        print(
            "[lr_schedule] "
            f"type={lr_scheduler_type} "
            f"warmup_update_steps={warmup_steps} "
            f"scheduler_step_multiplier={scheduler_step_multiplier} "
            f"scheduler_warmup_steps={scheduler_warmup_steps} "
            f"scheduler_training_steps={scheduler_training_steps}"
        )
        if timbre_memory_config.enabled:
            print(
                "[speaker_loss_schedule] "
                f"type={speaker_loss_schedule} "
                f"warmup_steps={speaker_loss_warmup_steps} "
                f"warmup_weight={speaker_loss_warmup_weight:.4f} "
                f"final_ref_weight={final_target_speaker_similarity_weight:.4f} "
                f"final_srcsup_weight={final_source_speaker_suppression_weight:.4f}"
            )
            print(
                "[speaker_timbre_repair] "
                f"ref_prompt_tokens={timbre_memory_config.ref_speaker_prompt_tokens} "
                f"ref_prompt_dropout={timbre_memory_config.ref_speaker_prompt_dropout:.4f} "
                f"ref_prompt_mode={timbre_memory_config.ref_speaker_prompt_mode} "
                f"ref_prompt_source={timbre_memory_config.ref_speaker_prompt_token_source} "
                f"ref_prompt_slot={timbre_memory_config.ref_speaker_prompt_slot} "
                f"ref_prompt_slot_code={timbre_memory_config.ref_speaker_prompt_slot_code} "
                f"ref_prompt_slot_pack={timbre_memory_config.ref_speaker_prompt_slot_pack_mode} "
                f"ref_prompt_output_norm={timbre_memory_config.ref_speaker_prompt_output_norm} "
                f"ref_prompt_output_scale={timbre_memory_config.ref_speaker_prompt_output_scale:.4f} "
                f"target_front_ce_weight={timbre_memory_config.target_front_ce_weight:.4f} "
                f"target_front_ce_seconds={timbre_memory_config.target_front_ce_seconds:.4f} "
                f"ref_adaln_weight={timbre_memory_config.ref_speaker_adaln_weight:.4f} "
                f"infonce_weight={timbre_memory_config.speaker_infonce_weight:.4f} "
                f"infonce_temp={timbre_memory_config.speaker_infonce_temperature:.4f} "
                f"infonce_neg_pool={timbre_memory_config.speaker_infonce_negative_pool_size} "
                f"infonce_neg_pool_seed={timbre_memory_config.speaker_infonce_negative_pool_seed} "
                f"condition_dropout={timbre_memory_config.speaker_condition_dropout:.4f} "
                f"speaker_side={timbre_memory_config.speaker_side_pathway_enabled} "
                f"speaker_side_layers={timbre_memory_config.speaker_side_pathway_layers} "
                f"speaker_side_kv={timbre_memory_config.speaker_side_pathway_kv_bias} "
                f"speaker_side_gate_init={timbre_memory_config.speaker_side_pathway_gate_init:.4f} "
                f"speaker_side_dropout={timbre_memory_config.speaker_side_pathway_dropout:.4f} "
                f"speaker_cross_attn={timbre_memory_config.speaker_cross_attn_enabled} "
                f"speaker_cross_attn_source={timbre_memory_config.speaker_cross_attn_source} "
                f"speaker_cross_attn_seq_dim={timbre_memory_config.speaker_cross_attn_seq_dim} "
                f"speaker_cross_attn_layers={timbre_memory_config.speaker_cross_attn_layers} "
                f"speaker_cross_attn_tokens={timbre_memory_config.speaker_cross_attn_tokens} "
                f"speaker_cross_attn_gate_init={timbre_memory_config.speaker_cross_attn_gate_init:.4f} "
                f"speaker_cross_attn_dropout={timbre_memory_config.speaker_cross_attn_dropout:.4f} "
                f"speaker_cross_attn_output_scale={timbre_memory_config.speaker_cross_attn_output_scale:.4f} "
                f"speaker_cross_attn_token_init_std={timbre_memory_config.speaker_cross_attn_token_init_std} "
                f"speaker_cross_attn_alpha_warmup_steps={timbre_memory_config.speaker_cross_attn_alpha_warmup_steps} "
                f"use_perturbed_source_prompt={timbre_memory_config.use_perturbed_source_prompt}"
            )
    else:
        train_log_handle = None
        tb_writer = None

    def write_train_log(line: str) -> None:
        if train_log_handle is None:
            return
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        train_log_handle.write(f"{timestamp} {line}\n")
        train_log_handle.flush()

    def save_accelerated_checkpoint(save_dir: Path) -> None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            tmp_dir = save_dir.with_name(f".{save_dir.name}.tmp-{os.getpid()}")
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            try:
                accelerator.unwrap_model(model).save_pretrained(tmp_dir)
                if save_dir.exists():
                    shutil.rmtree(save_dir)
                tmp_dir.rename(save_dir)
            except Exception:
                save_dir.mkdir(parents=True, exist_ok=True)
                with (save_dir / "checkpoint_save_error.txt").open("w", encoding="utf-8") as handle:
                    handle.write(traceback.format_exc())
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
                raise
        accelerator.wait_for_everyone()

    def gather_mean_scalar(value: float) -> float:
        tensor = torch.tensor([float(value)], device=accelerator.device, dtype=torch.float32)
        gathered = accelerator.gather(tensor)
        return float(gathered.mean().item())

    def gather_sum_scalar(value: float) -> float:
        tensor = torch.tensor([float(value)], device=accelerator.device, dtype=torch.float32)
        gathered = accelerator.gather(tensor)
        return float(gathered.sum().item())

    def gather_optional_mean_scalar(value: float | None) -> float | None:
        raw = float("nan") if value is None else float(value)
        tensor = torch.tensor([raw], device=accelerator.device, dtype=torch.float32)
        gathered = accelerator.gather(tensor)
        finite = torch.isfinite(gathered)
        if not bool(finite.any().item()):
            return None
        return float(gathered[finite].mean().item())

    last_eval_steps: dict[str, int] = {}

    def run_eval_one(step: int, *, reason: str, split: str, loader: Any) -> None:
        if not eval_loaders:
            return
        if int(step) == int(last_eval_steps.get(split, -1)):
            return
        if timbre_memory_config.enabled:
            set_speaker_aux_weights(
                accelerator.unwrap_model(model),
                final_target_speaker_similarity_weight,
                final_source_speaker_suppression_weight,
            )
        was_training = bool(model.training)
        model.eval()
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        eval_batches = 0
        eval_samples = 0
        eval_started = time.perf_counter()

        def add_metric(name: str, value: float | None) -> None:
            if value is None or not math.isfinite(float(value)):
                return
            metric_sums[name] = metric_sums.get(name, 0.0) + float(value)
            metric_counts[name] = metric_counts.get(name, 0) + 1

        with torch.no_grad():
            for eval_batch_idx, eval_batch in enumerate(loader):
                if int(args.eval_max_batches) > 0 and eval_batch_idx >= int(args.eval_max_batches):
                    break
                forward_kwargs = build_forward_kwargs_from_batch(
                    eval_batch,
                    timbre_memory_enabled=timbre_memory_config.enabled,
                    channelwise_loss_weight=channelwise_loss_weight,
                )
                outputs = model(**forward_kwargs)
                loss = outputs.loss
                if loss is None:
                    debug_model = accelerator.unwrap_model(model)
                    debug = getattr(debug_model, "last_forward_debug", {})
                    raise RuntimeError(f"Eval returned None loss; forward_debug={debug}")
                add_metric("loss", gather_mean_scalar(float(loss.detach().float().mean().item())))
                if timbre_memory_config.enabled:
                    aux_losses = collect_aux_loss_scalars(accelerator.unwrap_model(model))
                    for aux_name in AUX_LOSS_ATTRS:
                        add_metric(aux_name, gather_optional_mean_scalar(aux_losses.get(aux_name)))
                eval_batches += 1
                eval_samples += int(round(gather_sum_scalar(float(eval_batch["input_ids"].shape[0]))))

        if was_training:
            model.train()
        last_eval_steps[split] = int(step)
        metrics = {name: metric_sums[name] / max(1, metric_counts[name]) for name in sorted(metric_sums)}
        elapsed = time.perf_counter() - eval_started
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            tb_prefix = "eval" if split == "eval" else f"eval_{split}"
            row = {
                "step": int(step),
                "split": split,
                "reason": reason,
                "batches": int(eval_batches),
                "samples": int(eval_samples),
                "elapsed_sec": round(float(elapsed), 3),
                **metrics,
            }
            eval_log_path = Path(args.output_dir) / "eval_loss.jsonl"
            with eval_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            if tb_writer is not None:
                for metric_name, metric_value in metrics.items():
                    tb_writer.add_scalar(f"{tb_prefix}/{metric_name}", float(metric_value), int(step))
                tb_writer.add_scalar(f"{tb_prefix}/batches", int(eval_batches), int(step))
                tb_writer.add_scalar(f"{tb_prefix}/samples", int(eval_samples), int(step))
                tb_writer.flush()
            loss_text = f"{metrics.get('loss', math.nan):.4f}" if "loss" in metrics else "nan"
            msg = (
                f"eval split={split} step={step} reason={reason} loss={loss_text} "
                f"batches={eval_batches} samples={eval_samples} elapsed={elapsed:.1f}s"
            )
            if "content_ctc_aux_loss" in metrics:
                msg += f" content_ctc_aux_loss={metrics['content_ctc_aux_loss']:.4f}"
            if "content_ctc_loss" in metrics:
                msg += f" content_ctc_loss_raw={metrics['content_ctc_loss']:.4f}"
            if "content_ctc_loss_weighted" in metrics:
                msg += f" content_ctc_loss_weighted={metrics['content_ctc_loss_weighted']:.4f}"
            if "content_ctc_nonblank_posterior_mean" in metrics:
                msg += f" content_ctc_nonblank_post={metrics['content_ctc_nonblank_posterior_mean']:.4f}"
            if "semantic_aux_loss" in metrics:
                msg += f" semantic_aux_loss={metrics['semantic_aux_loss']:.4f}"
            if "source_semantic_aux_loss" in metrics:
                msg += f" source_semantic_aux_loss={metrics['source_semantic_aux_loss']:.4f}"
            if "source_semantic_gate_mean" in metrics:
                msg += f" source_semantic_gate_mean={metrics['source_semantic_gate_mean']:.4f}"
            if "source_semantic_progress_loss" in metrics:
                msg += f" source_semantic_progress_loss={metrics['source_semantic_progress_loss']:.4f}"
            if "progress_stop_aux_loss" in metrics:
                msg += f" progress_stop_aux_loss={metrics['progress_stop_aux_loss']:.4f}"
            if "speaker_side_gate_mean" in metrics:
                msg += f" speaker_side_gate_mean={metrics['speaker_side_gate_mean']:.4f}"
            if "speaker_cross_attn_gate_mean" in metrics:
                msg += f" speaker_cross_attn_gate_mean={metrics['speaker_cross_attn_gate_mean']:.4f}"
            write_train_log(msg)
            accelerator.print(msg)

    def run_eval(step: int, *, reason: str) -> None:
        for split, loader in eval_loaders:
            run_eval_one(step, reason=reason, split=split, loader=loader)

    if args.eval_only:
        if not eval_loaders:
            raise ValueError("--eval-only requires at least one eval loader")
        eval_only_started = time.perf_counter()
        accelerator.wait_for_everyone()
        run_eval(0, reason="eval_only")
        if accelerator.is_main_process:
            if tb_writer is not None:
                tb_writer.flush()
                tb_writer.close()
            write_train_log(f"eval_only finished elapsed={time.perf_counter() - eval_only_started:.1f}s")
            if train_log_handle is not None:
                train_log_handle.close()
        return 0

    model.train()
    global_step = 0
    saw_nonzero_lora_grad = False
    saw_nonzero_timbre_grad = False
    saw_nonzero_ref_speaker_prompt_grad = False
    saw_nonzero_routing_gate_grad = False
    saw_nonzero_source_semantic_grad = False
    saw_nonzero_source_semantic_gate_grad = False
    expect_source_semantic_gate_grad = False
    saw_nonzero_content_ctc_head_grad = False
    saw_nonzero_speaker_side_adaln_grad = False
    saw_nonzero_speaker_side_kv_bias_grad = False
    saw_nonzero_speaker_side_gate_grad = False
    saw_nonzero_speaker_cross_attn_grad = False
    saw_nonzero_speaker_cross_attn_gate_grad = False
    last_lora_grad_norm = 0.0
    last_timbre_grad_norm = 0.0
    last_ref_speaker_prompt_grad_norm = 0.0
    last_routing_gate_grad_norm = 0.0
    last_source_semantic_grad_norm = 0.0
    last_source_semantic_gate_grad_norm = 0.0
    last_content_ctc_head_grad_norm = 0.0
    last_speaker_side_adaln_grad_norm = 0.0
    last_speaker_side_kv_bias_grad_norm = 0.0
    last_speaker_side_gate_grad_norm = 0.0
    last_speaker_cross_attn_grad_norm = 0.0
    last_speaker_cross_attn_gate_grad_norm = 0.0
    speaker_side_max_grad_norms: dict[str, dict[str, float]] = {
        "adaln": {},
        "kv_bias": {},
        "gate": {},
        "cross_attn": {},
        "cross_attn_gate": {},
    }
    started = time.perf_counter()
    for epoch in range(effective_num_epochs):
        for batch in train_loader:
            with accelerator.accumulate(model):
                current_speaker_weight = final_target_speaker_similarity_weight
                current_source_suppression_weight = final_source_speaker_suppression_weight
                speaker_warmup_active = False
                if timbre_memory_config.enabled:
                    (
                        current_speaker_weight,
                        current_source_suppression_weight,
                        speaker_warmup_active,
                    ) = scheduled_speaker_aux_weights(
                        step=global_step,
                        final_speaker_weight=final_target_speaker_similarity_weight,
                        final_source_suppression_weight=final_source_speaker_suppression_weight,
                        warmup_steps=speaker_loss_warmup_steps,
                        warmup_weight=speaker_loss_warmup_weight,
                        schedule=speaker_loss_schedule,
                    )
                    set_speaker_aux_weights(
                        accelerator.unwrap_model(model),
                        current_speaker_weight,
                        current_source_suppression_weight,
                    )
                forward_kwargs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "labels": batch["labels"],
                    "channelwise_loss_weight": channelwise_loss_weight,
                }
                if timbre_memory_config.enabled:
                    if (
                        timbre_memory_config.speaker_cross_attn_enabled
                        and int(timbre_memory_config.speaker_cross_attn_alpha_warmup_steps) > 0
                    ):
                        alpha = min(
                            1.0,
                            float(global_step + 1)
                            / float(max(1, int(timbre_memory_config.speaker_cross_attn_alpha_warmup_steps))),
                        )
                        set_runtime_scale = getattr(
                            accelerator.unwrap_model(model),
                            "set_speaker_cross_attn_runtime_scale_multiplier",
                            None,
                        )
                        if callable(set_runtime_scale):
                            set_runtime_scale(alpha)
                        active_config = getattr(accelerator.unwrap_model(model), "timbre_memory_config", None)
                        if active_config is not None and hasattr(
                            active_config,
                            "speaker_cross_attn_runtime_scale_multiplier",
                        ):
                            active_config.speaker_cross_attn_runtime_scale_multiplier = float(alpha)
                    forward_kwargs.update(
                        {
                            "source_ref_codes": batch.get("source_ref_codes"),
                            "source_ref_mask": batch.get("source_ref_mask"),
                            "timbre_ref_codes": batch["timbre_ref_codes"],
                            "timbre_ref_mask": batch["timbre_ref_mask"],
                            "target_position_mask": batch["target_assistant_positions"],
                            "source_prompt_positions": batch["source_prompt_positions"],
                            "timbre_ref_prompt_positions": batch["timbre_ref_prompt_positions"],
                            "ref_speaker_prompt_slot_positions": batch.get("ref_speaker_prompt_slot_positions"),
                            "role_ids": batch.get("role_ids"),
                            "vc_mode_id": batch["vc_mode_id"],
                            "source_speaker_embedding_path": batch.get("source_speaker_embedding_path"),
                            "timbre_ref_speaker_embedding_path": batch.get("timbre_ref_speaker_embedding_path"),
                            "target_speaker_embedding_path": batch.get("target_speaker_embedding_path"),
                            "speaker_vec_path": batch.get("speaker_vec_path"),
                            "speaker_seq_path": batch.get("speaker_seq_path"),
                            "speaker_seq_features": batch.get("speaker_seq_features"),
                            "speaker_seq_features_mask": batch.get("speaker_seq_features_mask"),
                            "source_speaker_audio_path": batch.get("source_speaker_audio_path"),
                            "timbre_ref_speaker_audio_path": batch.get("timbre_ref_speaker_audio_path"),
                            "target_speaker_audio_path": batch.get("target_speaker_audio_path"),
                        }
                    )
                if timbre_memory_config.enabled:
                    for optional_key in TIMBRE_OPTIONAL_BATCH_KEYS:
                        if optional_key in batch:
                            forward_kwargs[optional_key] = batch[optional_key]
                outputs = model(**forward_kwargs)
                loss = outputs.loss
                if loss is None:
                    debug_model = accelerator.unwrap_model(model)
                    debug = getattr(debug_model, "last_forward_debug", {})
                    raise RuntimeError(f"Model returned None loss; forward_debug={debug}")
                speaker_aux_loss_value = None
                speaker_aux_stats: dict[str, float] = {}
                prosody_aux_loss_value = None
                prosody_aux_stats: dict[str, float] = {}
                content_aux_loss_value = None
                content_aux_stats: dict[str, float] = {}
                content_ctc_aux_loss_value = None
                content_ctc_aux_stats: dict[str, float] = {}
                semantic_aux_loss_value = None
                semantic_aux_stats: dict[str, float] = {}
                source_semantic_aux_loss_value = None
                source_semantic_aux_stats: dict[str, float] = {}
                ref_content_suppression_loss_value = None
                ref_content_suppression_stats: dict[str, float] = {}
                progress_stop_aux_loss_value = None
                progress_stop_aux_stats: dict[str, float] = {}
                route_loss_value = None
                route_stats: dict[str, float] = {}
                speaker_side_stats: dict[str, float] = {}
                target_front_ce_stats: dict[str, float] = {}
                if timbre_memory_config.enabled:
                    unwrapped_for_stats = accelerator.unwrap_model(model)
                    speaker_aux_loss_value = getattr(unwrapped_for_stats, "last_speaker_aux_loss", None)
                    speaker_aux_stats = getattr(unwrapped_for_stats, "last_speaker_aux_stats", {}) or {}
                    prosody_aux_loss_value = getattr(unwrapped_for_stats, "last_prosody_aux_loss", None)
                    prosody_aux_stats = getattr(unwrapped_for_stats, "last_prosody_aux_stats", {}) or {}
                    content_aux_loss_value = getattr(unwrapped_for_stats, "last_content_aux_loss", None)
                    content_aux_stats = getattr(unwrapped_for_stats, "last_content_aux_stats", {}) or {}
                    content_ctc_aux_loss_value = getattr(unwrapped_for_stats, "last_content_ctc_aux_loss", None)
                    content_ctc_aux_stats = getattr(unwrapped_for_stats, "last_content_ctc_aux_stats", {}) or {}
                    semantic_aux_loss_value = getattr(unwrapped_for_stats, "last_semantic_aux_loss", None)
                    semantic_aux_stats = getattr(unwrapped_for_stats, "last_semantic_aux_stats", {}) or {}
                    source_semantic_aux_loss_value = getattr(
                        unwrapped_for_stats,
                        "last_source_semantic_aux_loss",
                        None,
                    )
                    source_semantic_aux_stats = getattr(
                        unwrapped_for_stats,
                        "last_source_semantic_aux_stats",
                        {},
                    ) or {}
                    ref_content_suppression_loss_value = getattr(
                        unwrapped_for_stats,
                        "last_ref_content_suppression_loss",
                        None,
                    )
                    ref_content_suppression_stats = getattr(
                        unwrapped_for_stats,
                        "last_ref_content_suppression_stats",
                        {},
                    ) or {}
                    progress_stop_aux_loss_value = getattr(unwrapped_for_stats, "last_progress_stop_aux_loss", None)
                    progress_stop_aux_stats = getattr(unwrapped_for_stats, "last_progress_stop_aux_stats", {}) or {}
                    route_loss_value = getattr(unwrapped_for_stats, "last_route_loss", None)
                    route_stats = getattr(unwrapped_for_stats, "last_route_stats", {}) or {}
                    speaker_side_stats = getattr(unwrapped_for_stats, "last_speaker_side_stats", {}) or {}
                    target_front_ce_stats = getattr(unwrapped_for_stats, "last_target_front_ce_stats", {}) or {}
                if (
                    timbre_memory_config.enabled
                    and timbre_memory_config.source_semantic_memory_enabled
                    and source_semantic_gate_initial_snapshot
                    and float(source_semantic_aux_stats.get("source_semantic_gate_mean", 0.0)) > 0.0
                ):
                    expect_source_semantic_gate_grad = True
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    sync_ignored_trainable_grads(ignored_modules_for_manual_grad_sync)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    clip_mixed_dtype_trainable_grad_norm_(model, args.max_grad_norm)
                if accelerator.sync_gradients:
                    last_lora_grad_norm, has_nonzero = lora_grad_stats(model)
                    saw_nonzero_lora_grad = saw_nonzero_lora_grad or has_nonzero
                    if timbre_memory_config.enabled:
                        last_timbre_grad_norm, has_nonzero_timbre = timbre_adapter_grad_stats(model)
                        saw_nonzero_timbre_grad = saw_nonzero_timbre_grad or has_nonzero_timbre
                        (
                            last_speaker_side_adaln_grad_norm,
                            has_nonzero_speaker_side_adaln,
                        ) = named_fragment_grad_stats(model, ("speaker_side_adaln",))
                        saw_nonzero_speaker_side_adaln_grad = (
                            saw_nonzero_speaker_side_adaln_grad or has_nonzero_speaker_side_adaln
                        )
                        merge_max_norms(
                            speaker_side_max_grad_norms["adaln"],
                            named_fragment_layer_grad_norms(model, "speaker_side_adaln"),
                        )
                        (
                            last_speaker_side_kv_bias_grad_norm,
                            has_nonzero_speaker_side_kv_bias,
                        ) = named_fragment_grad_stats(model, ("speaker_side_kv_bias",))
                        saw_nonzero_speaker_side_kv_bias_grad = (
                            saw_nonzero_speaker_side_kv_bias_grad or has_nonzero_speaker_side_kv_bias
                        )
                        merge_max_norms(
                            speaker_side_max_grad_norms["kv_bias"],
                            named_fragment_layer_grad_norms(model, "speaker_side_kv_bias"),
                        )
                        (
                            last_speaker_side_gate_grad_norm,
                            has_nonzero_speaker_side_gate,
                        ) = named_fragment_grad_stats(model, ("speaker_side_gate_logits",))
                        saw_nonzero_speaker_side_gate_grad = (
                            saw_nonzero_speaker_side_gate_grad or has_nonzero_speaker_side_gate
                        )
                        merge_max_norms(
                            speaker_side_max_grad_norms["gate"],
                            named_fragment_layer_grad_norms(model, "speaker_side_gate_logits"),
                        )
                        (
                            last_speaker_cross_attn_grad_norm,
                            has_nonzero_speaker_cross_attn,
                        ) = named_fragment_grad_stats(
                            model,
                            ("speaker_cross_attn_tokens", "speaker_cross_attn_seq_projector", "speaker_cross_attn_layers"),
                        )
                        saw_nonzero_speaker_cross_attn_grad = (
                            saw_nonzero_speaker_cross_attn_grad or has_nonzero_speaker_cross_attn
                        )
                        merge_max_norms(
                            speaker_side_max_grad_norms["cross_attn"],
                            named_fragment_layer_grad_norms(model, "speaker_cross_attn_layers"),
                        )
                        (
                            last_speaker_cross_attn_gate_grad_norm,
                            has_nonzero_speaker_cross_attn_gate,
                        ) = speaker_cross_attn_gate_grad_stats(model)
                        saw_nonzero_speaker_cross_attn_gate_grad = (
                            saw_nonzero_speaker_cross_attn_gate_grad or has_nonzero_speaker_cross_attn_gate
                        )
                        merge_max_norms(
                            speaker_side_max_grad_norms["cross_attn_gate"],
                            speaker_cross_attn_gate_layer_grad_norms(model),
                        )
                        (
                            last_ref_speaker_prompt_grad_norm,
                            has_nonzero_ref_speaker_prompt,
                        ) = ref_speaker_prompt_grad_stats(model)
                        saw_nonzero_ref_speaker_prompt_grad = (
                            saw_nonzero_ref_speaker_prompt_grad or has_nonzero_ref_speaker_prompt
                        )
                        (
                            last_source_semantic_grad_norm,
                            has_nonzero_source_semantic,
                        ) = source_semantic_trainable_grad_stats(model)
                        saw_nonzero_source_semantic_grad = (
                            saw_nonzero_source_semantic_grad or has_nonzero_source_semantic
                        )
                        last_routing_gate_grad_norm, has_nonzero_routing_gate = routing_gate_grad_stats(model)
                        saw_nonzero_routing_gate_grad = saw_nonzero_routing_gate_grad or has_nonzero_routing_gate
                        (
                            last_source_semantic_gate_grad_norm,
                            has_nonzero_source_semantic_gate,
                        ) = source_semantic_gate_grad_stats(model)
                        saw_nonzero_source_semantic_gate_grad = (
                            saw_nonzero_source_semantic_gate_grad or has_nonzero_source_semantic_gate
                        )
                        last_content_ctc_head_grad_norm, has_nonzero_content_ctc_head = content_ctc_head_grad_stats(model)
                        saw_nonzero_content_ctc_head_grad = (
                            saw_nonzero_content_ctc_head_grad or has_nonzero_content_ctc_head
                        )
                if accelerator.sync_gradients:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.logging_steps == 0:
                    logged_loss = accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    routing_gate_delta_values: dict[str, float] = {}
                    if timbre_memory_config.enabled and routing_gate_initial_snapshot:
                        routing_gate_delta_values = routing_gate_delta_stats(
                            accelerator.unwrap_model(model),
                            routing_gate_initial_snapshot,
                        )
                    source_semantic_gate_delta_values: dict[str, float] = {}
                    if timbre_memory_config.enabled and source_semantic_gate_initial_snapshot:
                        source_semantic_gate_delta_values = source_semantic_gate_delta_stats(
                            accelerator.unwrap_model(model),
                            source_semantic_gate_initial_snapshot,
                        )
                    if tb_writer is not None:
                        samples_seen = global_step * estimated_global_batch_size
                        epoch_fraction = samples_seen / max(1, len(dataset))
                        tb_writer.add_scalar("train/loss", logged_loss, global_step)
                        tb_writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step)
                        tb_writer.add_scalar("train/lora_grad_norm", last_lora_grad_norm, global_step)
                        tb_writer.add_scalar("train/global_batch_size_estimate", estimated_global_batch_size, global_step)
                        tb_writer.add_scalar("train/samples_seen_estimate", samples_seen, global_step)
                        tb_writer.add_scalar("train/epoch_fraction_estimate", epoch_fraction, global_step)
                        if timbre_memory_config.enabled:
                            tb_writer.add_scalar("train/timbre_adapter_grad_norm", last_timbre_grad_norm, global_step)
                            tb_writer.add_scalar(
                                "train/ref_speaker_prompt_grad_norm",
                                last_ref_speaker_prompt_grad_norm,
                                global_step,
                            )
                            tb_writer.add_scalar(
                                "train/source_semantic_grad_norm",
                                last_source_semantic_grad_norm,
                                global_step,
                            )
                            tb_writer.add_scalar("train/routing_gate_grad_norm", last_routing_gate_grad_norm, global_step)
                            tb_writer.add_scalar(
                                "train/source_semantic_gate_grad_norm",
                                last_source_semantic_gate_grad_norm,
                                global_step,
                            )
                            tb_writer.add_scalar(
                                "train/content_ctc_head_grad_norm",
                                last_content_ctc_head_grad_norm,
                                global_step,
                            )
                            if timbre_memory_config.speaker_side_pathway_enabled:
                                tb_writer.add_scalar(
                                    "train/speaker_side_adaln_grad_norm",
                                    last_speaker_side_adaln_grad_norm,
                                    global_step,
                                )
                                tb_writer.add_scalar(
                                    "train/speaker_side_kv_bias_grad_norm",
                                    last_speaker_side_kv_bias_grad_norm,
                                    global_step,
                                )
                                tb_writer.add_scalar(
                                    "train/speaker_side_gate_grad_norm",
                                    last_speaker_side_gate_grad_norm,
                                    global_step,
                                )
                            if timbre_memory_config.speaker_cross_attn_enabled:
                                tb_writer.add_scalar(
                                    "train/speaker_cross_attn_grad_norm",
                                    last_speaker_cross_attn_grad_norm,
                                    global_step,
                                )
                                tb_writer.add_scalar(
                                    "train/speaker_cross_attn_gate_grad_norm",
                                    last_speaker_cross_attn_gate_grad_norm,
                                    global_step,
                                )
                            if len(lr_scheduler.get_last_lr()) > 1:
                                tb_writer.add_scalar("train/routing_gate_lr", lr_scheduler.get_last_lr()[1], global_step)
                            if source_semantic_gate_initial_snapshot:
                                tb_writer.add_scalar(
                                    "train/source_semantic_gate_lr",
                                    args.learning_rate * optimizer_group_summary["source_semantic_gate_lr_multiplier"],
                                    global_step,
                                )
                            if speaker_aux_loss_value is not None:
                                tb_writer.add_scalar("train/speaker_aux_loss", float(speaker_aux_loss_value), global_step)
                            tb_writer.add_scalar("train/current_speaker_loss_weight", current_speaker_weight, global_step)
                            tb_writer.add_scalar(
                                "train/current_source_suppression_weight",
                                current_source_suppression_weight,
                                global_step,
                            )
                            tb_writer.add_scalar("train/speaker_loss_warmup_active", float(speaker_warmup_active), global_step)
                            for stat_name, stat_value in speaker_aux_stats.items():
                                tb_writer.add_scalar(f"train/{stat_name}", float(stat_value), global_step)
                            if prosody_aux_loss_value is not None:
                                tb_writer.add_scalar("train/prosody_aux_loss", float(prosody_aux_loss_value), global_step)
                            for stat_name, stat_value in prosody_aux_stats.items():
                                tb_writer.add_scalar(f"train/{stat_name}", float(stat_value), global_step)
                            if content_aux_loss_value is not None:
                                tb_writer.add_scalar("train/content_aux_loss", float(content_aux_loss_value), global_step)
                            for stat_name, stat_value in content_aux_stats.items():
                                tb_writer.add_scalar(f"train/{stat_name}", float(stat_value), global_step)
                            if content_ctc_aux_loss_value is not None:
                                tb_writer.add_scalar(
                                    "train/content_ctc_aux_loss",
                                    float(content_ctc_aux_loss_value),
                                    global_step,
                                )
                            for stat_name, stat_value in content_ctc_aux_stats.items():
                                tb_writer.add_scalar(f"train/{stat_name}", float(stat_value), global_step)
                            if semantic_aux_loss_value is not None:
                                tb_writer.add_scalar("train/semantic_aux_loss", float(semantic_aux_loss_value), global_step)
                            for stat_name, stat_value in semantic_aux_stats.items():
                                tb_writer.add_scalar(f"train/{stat_name}", float(stat_value), global_step)
                            if source_semantic_aux_loss_value is not None:
                                tb_writer.add_scalar(
                                    "train/source_semantic_aux_loss",
                                    float(source_semantic_aux_loss_value),
                                    global_step,
                                )
                            for stat_name, stat_value in source_semantic_aux_stats.items():
                                tb_writer.add_scalar(f"train/{stat_name}", float(stat_value), global_step)
                            for stat_name, stat_value in source_semantic_gate_delta_values.items():
                                tb_writer.add_scalar(f"train/{stat_name}", float(stat_value), global_step)
                            if ref_content_suppression_loss_value is not None:
                                tb_writer.add_scalar(
                                    "train/ref_content_suppression_loss",
                                    float(ref_content_suppression_loss_value),
                                    global_step,
                                )
                            for stat_name, stat_value in ref_content_suppression_stats.items():
                                tb_writer.add_scalar(f"train/{stat_name}", float(stat_value), global_step)
                            if progress_stop_aux_loss_value is not None:
                                tb_writer.add_scalar(
                                    "train/progress_stop_aux_loss",
                                    float(progress_stop_aux_loss_value),
                                    global_step,
                                )
                            for stat_name, stat_value in progress_stop_aux_stats.items():
                                tb_writer.add_scalar(f"train/{stat_name}", float(stat_value), global_step)
                            if route_loss_value is not None:
                                tb_writer.add_scalar("train/route_loss", float(route_loss_value), global_step)
                            for stat_name, stat_value in route_stats.items():
                                tb_writer.add_scalar(f"train/{stat_name}", float(stat_value), global_step)
                            for stat_name, stat_value in speaker_side_stats.items():
                                tb_writer.add_scalar(f"train/{stat_name}", float(stat_value), global_step)
                            for stat_name, stat_value in target_front_ce_stats.items():
                                tb_writer.add_scalar(f"train/{stat_name}", float(stat_value), global_step)
                            for stat_name, stat_value in routing_gate_delta_values.items():
                                tb_writer.add_scalar(f"train/{stat_name}", float(stat_value), global_step)
                        tb_writer.add_scalar("train/epoch", epoch, global_step)
                    msg = (
                        f"step={global_step}/{max_train_steps} epoch={epoch} "
                        f"loss={logged_loss:.4f} lr={lr_scheduler.get_last_lr()[0]:.2e} "
                        f"lora_grad_norm={last_lora_grad_norm:.4f}"
                    )
                    if timbre_memory_config.enabled:
                        msg += f" timbre_adapter_grad_norm={last_timbre_grad_norm:.4f}"
                        msg += f" ref_speaker_prompt_grad_norm={last_ref_speaker_prompt_grad_norm:.6f}"
                        msg += f" source_semantic_grad_norm={last_source_semantic_grad_norm:.4f}"
                        msg += f" routing_gate_grad_norm={last_routing_gate_grad_norm:.6f}"
                        msg += f" source_semantic_gate_grad_norm={last_source_semantic_gate_grad_norm:.6f}"
                        msg += f" content_ctc_head_grad_norm={last_content_ctc_head_grad_norm:.6f}"
                        if timbre_memory_config.speaker_side_pathway_enabled:
                            msg += f" speaker_side_adaln_grad_norm={last_speaker_side_adaln_grad_norm:.6f}"
                            msg += f" speaker_side_kv_bias_grad_norm={last_speaker_side_kv_bias_grad_norm:.6f}"
                            msg += f" speaker_side_gate_grad_norm={last_speaker_side_gate_grad_norm:.6f}"
                        if timbre_memory_config.speaker_cross_attn_enabled:
                            msg += f" speaker_cross_attn_grad_norm={last_speaker_cross_attn_grad_norm:.6f}"
                            msg += f" speaker_cross_attn_gate_grad_norm={last_speaker_cross_attn_gate_grad_norm:.6f}"
                        if speaker_aux_loss_value is not None:
                            msg += f" speaker_aux_loss={float(speaker_aux_loss_value):.4f}"
                        msg += (
                            f" spk_w={current_speaker_weight:.3f}"
                            f" srcsup_w={current_source_suppression_weight:.3f}"
                        )
                        if speaker_warmup_active:
                            msg += " spk_warmup=1"
                        if prosody_aux_loss_value is not None:
                            msg += f" prosody_aux_loss={float(prosody_aux_loss_value):.4f}"
                        if content_aux_loss_value is not None:
                            msg += f" content_aux_loss={float(content_aux_loss_value):.4f}"
                        if content_ctc_aux_loss_value is not None:
                            msg += f" content_ctc_aux_loss={float(content_ctc_aux_loss_value):.4f}"
                        if "content_ctc_loss" in content_ctc_aux_stats:
                            msg += f" content_ctc_loss_raw={content_ctc_aux_stats['content_ctc_loss']:.4f}"
                        if "content_ctc_loss_weighted" in content_ctc_aux_stats:
                            msg += f" content_ctc_loss_weighted={content_ctc_aux_stats['content_ctc_loss_weighted']:.4f}"
                        if "content_ctc_nonblank_posterior_mean" in content_ctc_aux_stats:
                            msg += (
                                f" content_ctc_nonblank_post="
                                f"{content_ctc_aux_stats['content_ctc_nonblank_posterior_mean']:.4f}"
                            )
                        if semantic_aux_loss_value is not None:
                            msg += f" semantic_aux_loss={float(semantic_aux_loss_value):.4f}"
                        if source_semantic_aux_loss_value is not None:
                            msg += f" source_semantic_aux_loss={float(source_semantic_aux_loss_value):.4f}"
                        if "source_semantic_gate_mean" in source_semantic_aux_stats:
                            msg += f" source_semantic_gate_mean={source_semantic_aux_stats['source_semantic_gate_mean']:.4f}"
                        if "source_semantic_delta_ratio" in source_semantic_aux_stats:
                            msg += f" source_semantic_delta_ratio={source_semantic_aux_stats['source_semantic_delta_ratio']:.6f}"
                        if "source_semantic_prompt_delta_norm" in source_semantic_aux_stats:
                            msg += (
                                f" source_semantic_prompt_delta_norm="
                                f"{source_semantic_aux_stats['source_semantic_prompt_delta_norm']:.6f}"
                            )
                        if "source_semantic_attn_coverage" in source_semantic_aux_stats:
                            msg += f" source_semantic_attn_coverage={source_semantic_aux_stats['source_semantic_attn_coverage']:.4f}"
                        if "source_semantic_attn_expected_pos_slope" in source_semantic_aux_stats:
                            msg += (
                                f" source_semantic_attn_slope="
                                f"{source_semantic_aux_stats['source_semantic_attn_expected_pos_slope']:.4f}"
                            )
                        if "source_semantic_progress_loss" in source_semantic_aux_stats:
                            msg += (
                                f" source_semantic_progress_loss="
                                f"{source_semantic_aux_stats['source_semantic_progress_loss']:.4f}"
                            )
                        if ref_content_suppression_loss_value is not None:
                            msg += f" ref_content_suppression_loss={float(ref_content_suppression_loss_value):.4f}"
                        if "ref_content_cos" in ref_content_suppression_stats:
                            msg += f" ref_content_cos={ref_content_suppression_stats['ref_content_cos']:.4f}"
                        if "ref_content_suppression_loss_weighted" in ref_content_suppression_stats:
                            msg += (
                                f" ref_content_suppression_w="
                                f"{ref_content_suppression_stats['ref_content_suppression_loss_weighted']:.4f}"
                            )
                        if "source_semantic_gate_delta_max" in source_semantic_gate_delta_values:
                            msg += (
                                f" source_semantic_gate_delta_max="
                                f"{source_semantic_gate_delta_values['source_semantic_gate_delta_max']:.6g}"
                            )
                        if progress_stop_aux_loss_value is not None:
                            msg += f" progress_stop_aux_loss={float(progress_stop_aux_loss_value):.4f}"
                        if "ref_speaker_cos" in speaker_aux_stats:
                            msg += f" ref_speaker_cos={speaker_aux_stats['ref_speaker_cos']:.4f}"
                        if "source_minus_ref_cos" in speaker_aux_stats:
                            msg += f" source_minus_ref_cos={speaker_aux_stats['source_minus_ref_cos']:.4f}"
                        if route_loss_value is not None:
                            msg += f" route_loss={float(route_loss_value):.4f}"
                        if "speaker_side_gate_mean" in speaker_side_stats:
                            msg += f" speaker_side_gate_mean={speaker_side_stats['speaker_side_gate_mean']:.4f}"
                        if "speaker_side_dropout_rate" in speaker_side_stats:
                            msg += f" speaker_side_dropout={speaker_side_stats['speaker_side_dropout_rate']:.4f}"
                        if "speaker_cross_attn_gate_mean" in speaker_side_stats:
                            msg += f" speaker_cross_attn_gate_mean={speaker_side_stats['speaker_cross_attn_gate_mean']:.4f}"
                        if "speaker_cross_attn_token_norm_mean" in speaker_side_stats:
                            msg += (
                                f" speaker_cross_attn_token_norm="
                                f"{speaker_side_stats['speaker_cross_attn_token_norm_mean']:.4f}"
                            )
                        if "speaker_cross_attn_delta_ratio" in speaker_side_stats:
                            msg += (
                                f" speaker_cross_attn_delta_ratio="
                                f"{speaker_side_stats['speaker_cross_attn_delta_ratio']:.6f}"
                            )
                        if "speaker_cross_attn_runtime_scale_multiplier" in speaker_side_stats:
                            msg += (
                                f" speaker_cross_attn_alpha="
                                f"{speaker_side_stats['speaker_cross_attn_runtime_scale_multiplier']:.4f}"
                            )
                        if float(target_front_ce_stats.get("target_front_ce_weighted_tokens", 0.0)) > 0.0:
                            msg += (
                                f" target_front_ce_weight={target_front_ce_stats.get('target_front_ce_weight', 1.0):.3f}"
                                f" target_front_ce_frames={target_front_ce_stats.get('target_front_ce_frames', 0.0):.1f}"
                                f" target_front_ce_tokens="
                                f"{target_front_ce_stats.get('target_front_ce_weighted_tokens', 0.0):.0f}"
                            )
                        if "role_gate_mean" in route_stats:
                            msg += f" role_gate_mean={route_stats['role_gate_mean']:.4f}"
                        if "prosody_head_gate_mean" in route_stats:
                            msg += f" prosody_head_gate_mean={route_stats['prosody_head_gate_mean']:.4f}"
                        if "timbre_head_gate_mean" in route_stats:
                            msg += f" timbre_head_gate_mean={route_stats['timbre_head_gate_mean']:.4f}"
                        if "source_prosody_batch_gate_mean" in route_stats:
                            msg += f" source_prosody_gate_mean={route_stats['source_prosody_batch_gate_mean']:.4f}"
                        if "role_gate_delta_max" in routing_gate_delta_values:
                            msg += f" role_gate_delta_max={routing_gate_delta_values['role_gate_delta_max']:.6g}"
                        if "prosody_head_gate_delta_mean" in routing_gate_delta_values:
                            msg += (
                                f" prosody_head_gate_delta_mean="
                                f"{routing_gate_delta_values['prosody_head_gate_delta_mean']:.6g}"
                            )
                        if "timbre_head_gate_delta_mean" in routing_gate_delta_values:
                            msg += (
                                f" timbre_head_gate_delta_mean="
                                f"{routing_gate_delta_values['timbre_head_gate_delta_mean']:.6g}"
                            )
                    write_train_log(msg)
                    if tb_writer is not None:
                        tb_writer.flush()
                    accelerator.print(msg)
                saved_this_step = False
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_dir = Path(args.output_dir) / f"step-{global_step}"
                    if timbre_memory_config.enabled:
                        set_speaker_aux_weights(
                            accelerator.unwrap_model(model),
                            final_target_speaker_similarity_weight,
                            final_source_speaker_suppression_weight,
                        )
                    save_accelerated_checkpoint(save_dir)
                    saved_this_step = True
                should_eval = bool(eval_loaders) and (
                    (args.eval_steps > 0 and global_step % args.eval_steps == 0)
                    or (args.eval_steps <= 0 and saved_this_step)
                )
                if should_eval:
                    run_eval(global_step, reason="checkpoint" if saved_this_step else "interval")
                if global_step >= max_train_steps:
                    break
        if eval_loaders and global_step > 0:
            run_eval(global_step, reason="epoch_end")
        if global_step >= max_train_steps:
            break

    if (
        accelerator.is_main_process
        and args.smoke_test
        and timbre_memory_config.enabled
        and (timbre_memory_config.speaker_side_pathway_enabled or timbre_memory_config.speaker_cross_attn_enabled)
    ):
        write_ver29_smoke_train_diagnostics(
            Path(args.output_dir),
            accelerator.unwrap_model(model),
            initial_snapshot=speaker_side_initial_snapshot,
            max_grad_norms=speaker_side_max_grad_norms,
            global_step=global_step,
        )

    if args.smoke_test and not saw_nonzero_lora_grad:
        raise RuntimeError("Smoke test failed: LoRA gradients are all zero")
    if args.smoke_test and timbre_memory_config.enabled and not saw_nonzero_timbre_grad:
        raise RuntimeError("Smoke test failed: timbre memory adapter gradients are all zero")
    if (
        args.smoke_test
        and timbre_memory_config.source_semantic_memory_enabled
        and source_semantic_gate_initial_snapshot
        and expect_source_semantic_gate_grad
        and not saw_nonzero_source_semantic_gate_grad
    ):
        raise RuntimeError("Smoke test failed: SourceSemanticAdapter gate gradients are all zero")
    if args.smoke_test and timbre_memory_config.speaker_side_pathway_enabled:
        if not saw_nonzero_speaker_side_adaln_grad:
            raise RuntimeError("Smoke test failed: speaker-side AdaLN gradients are all zero")
        if timbre_memory_config.speaker_side_pathway_kv_bias and not saw_nonzero_speaker_side_kv_bias_grad:
            raise RuntimeError("Smoke test failed: speaker-side K/V bias gradients are all zero")
        if not saw_nonzero_speaker_side_gate_grad:
            raise RuntimeError("Smoke test failed: speaker-side gate gradients are all zero")
    if args.smoke_test and timbre_memory_config.speaker_cross_attn_enabled:
        if not saw_nonzero_speaker_cross_attn_grad:
            raise RuntimeError("Smoke test failed: speaker cross-attn gradients are all zero")
        if not saw_nonzero_speaker_cross_attn_gate_grad:
            raise RuntimeError("Smoke test failed: speaker cross-attn gate gradients are all zero")

    accelerator.wait_for_everyone()
    if eval_loaders:
        run_eval(global_step, reason="final")
    final_dir = Path(args.output_dir) / "final"
    if timbre_memory_config.enabled:
        set_speaker_aux_weights(
            accelerator.unwrap_model(model),
            final_target_speaker_similarity_weight,
            final_source_speaker_suppression_weight,
        )
    save_accelerated_checkpoint(final_dir)
    if accelerator.is_main_process:
        final_dir = Path(args.output_dir) / "final"
        if timbre_memory_config.enabled:
            set_speaker_aux_weights(
                accelerator.unwrap_model(model),
                final_target_speaker_similarity_weight,
                final_source_speaker_suppression_weight,
            )
        with (Path(args.output_dir) / "lora_train_args.json").open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()
        final_msg = f"finished global_step={global_step} elapsed={time.perf_counter() - started:.1f}s output={final_dir}"
        write_train_log(final_msg)
        if train_log_handle is not None:
            train_log_handle.close()
        print(final_msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

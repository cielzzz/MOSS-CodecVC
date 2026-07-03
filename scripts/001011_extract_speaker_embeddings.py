#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moss_codecvc.io_utils import iter_jsonl
from moss_codecvc.third_party import add_download_python_deps, default_speechbrain_ecapa_dir


def resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def load_speechbrain_ecapa(source: str, savedir: str | None, device: torch.device):
    add_download_python_deps()
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        try:
            from speechbrain.pretrained import EncoderClassifier
        except ImportError as exc:
            raise ImportError(
                "ECAPA extraction requires `speechbrain`. Install it in the data-prep environment, "
                "or generate embeddings with another speaker encoder and keep the same plan paths."
            ) from exc
    kwargs: dict[str, Any] = {"source": source}
    if Path(source).expanduser().exists():
        local_source = str(Path(source).expanduser().resolve())
        kwargs["source"] = local_source
        kwargs["overrides"] = {"pretrained_path": local_source}
        if savedir is None:
            savedir = local_source
    if savedir:
        kwargs["savedir"] = savedir
    try:
        kwargs["run_opts"] = {"device": str(device)}
        classifier = EncoderClassifier.from_hparams(**kwargs)
    except TypeError:
        kwargs.pop("run_opts", None)
        classifier = EncoderClassifier.from_hparams(**kwargs)
        classifier = classifier.to(device)
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False
    return classifier


@torch.inference_mode()
def encode_speechbrain_ecapa(classifier, audio_path: str, device: torch.device) -> torch.Tensor:
    if hasattr(classifier, "encode_file"):
        embedding = classifier.encode_file(str(audio_path)).squeeze()
    else:
        signal = classifier.load_audio(str(audio_path)).to(device)
        embedding = classifier.encode_batch(signal.unsqueeze(0)).squeeze()
    embedding = torch.as_tensor(embedding, dtype=torch.float32, device=device).flatten()
    return torch.nn.functional.normalize(embedding, dim=0).cpu()


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline frozen speaker embedding extraction for MOSS-CodecVC Ver1.6.")
    ap.add_argument("--embedding-plan-jsonl", required=True)
    ap.add_argument("--backend", choices=("speechbrain_ecapa",), default="speechbrain_ecapa")
    ap.add_argument(
        "--model-source",
        default=str(default_speechbrain_ecapa_dir()),
        help="Local SpeechBrain ECAPA directory or HF source id.",
    )
    ap.add_argument("--savedir", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-missing-audio", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--summary-json", default=None)
    args = ap.parse_args()

    rows = []
    for row in iter_jsonl(args.embedding_plan_jsonl):
        rows.append(row)
        if args.max_rows > 0 and len(rows) >= args.max_rows:
            break
    if not rows:
        raise ValueError(f"No rows found in {args.embedding_plan_jsonl}")

    missing_audio = 0
    existing = 0
    todo = 0
    for row in rows:
        audio = row.get("audio")
        out_path = row.get("speaker_embedding_path")
        if not audio or not Path(str(audio)).exists():
            missing_audio += 1
            continue
        if out_path and Path(str(out_path)).exists() and not args.overwrite:
            existing += 1
            continue
        todo += 1

    if args.dry_run:
        print(
            f"dry_run rows={len(rows)} todo={todo} existing={existing} "
            f"missing_audio={missing_audio} backend={args.backend}"
        )
        return 0

    if missing_audio and not args.allow_missing_audio:
        raise FileNotFoundError(
            f"{missing_audio} plan rows have missing audio. Re-run with --allow-missing-audio to skip them."
        )

    device = resolve_device(args.device)
    if args.backend == "speechbrain_ecapa":
        encoder = load_speechbrain_ecapa(args.model_source, args.savedir, device)
        encode_one = lambda audio: encode_speechbrain_ecapa(encoder, audio, device)
    else:
        raise ValueError(f"unsupported backend: {args.backend}")

    written = 0
    skipped_existing = 0
    skipped_missing = 0
    failed = 0
    for idx, row in enumerate(rows):
        audio = row.get("audio")
        out_path = row.get("speaker_embedding_path")
        if not audio or not out_path:
            skipped_missing += 1
            continue
        audio_path = Path(str(audio))
        if not audio_path.exists():
            if args.allow_missing_audio:
                skipped_missing += 1
                continue
            raise FileNotFoundError(str(audio_path))
        out = Path(str(out_path))
        if out.exists() and not args.overwrite:
            skipped_existing += 1
            continue
        try:
            embedding = encode_one(str(audio_path))
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "embedding": embedding,
                    "speaker_embedding": embedding,
                    "embedding_dim": int(embedding.numel()),
                    "backend": args.backend,
                    "model_source": args.model_source,
                    "audio": str(audio_path),
                    "role": row.get("role"),
                    "sample_id": row.get("sample_id"),
                    "model_name": row.get("model_name"),
                },
                out,
            )
            written += 1
        except Exception as exc:
            failed += 1
            print(f"extract error row={idx} audio={audio_path} err={type(exc).__name__}: {exc}", flush=True)
            if not args.allow_missing_audio:
                raise
        if args.log_every > 0 and (idx + 1) % args.log_every == 0:
            print(
                f"processed={idx + 1}/{len(rows)} written={written} "
                f"skipped_existing={skipped_existing} skipped_missing={skipped_missing} failed={failed}",
                flush=True,
            )

    summary = {
        "rows": len(rows),
        "written": written,
        "skipped_existing": skipped_existing,
        "skipped_missing": skipped_missing,
        "failed": failed,
        "backend": args.backend,
        "model_source": args.model_source,
    }
    if args.summary_json:
        out_summary = Path(args.summary_json)
        out_summary.parent.mkdir(parents=True, exist_ok=True)
        out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

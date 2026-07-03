# MOSS-CodecVC

[English](README.md) | [简体中文](README_zh.md)

MOSS-CodecVC is a codec-native voice conversion framework built on top of
MOSS-TTS and MOSS-Audio-Tokenizer.

The target task is not standard text-to-speech. It is:

```text
content / pauses / duration / rhythm from a source wav
+ timbre from a reference wav
-> converted target wav
```

MOSS-CodecVC reuses the codec-token generation and SFT path of MOSS-TTS, while
adding data preparation, role routing, timbre reference conditioning, and
voice-conversion-oriented training and inference wrappers.

## Architecture Notes

The public repository keeps the runnable source code, configs, small examples,
and experiment entry scripts. Longer design notes, diagrams, and internal
experiment writeups live under the local `docs/` directory and are intentionally
excluded from the GitHub release to keep the repository lightweight.

The first version reuses the existing
`MOSS-TTS/moss_tts_delay/finetuning/sft.py` teacher-forcing SFT pipeline:

```text
VC manifest
  -> MOSS audio codec token extraction
  -> MossTTSDelay SFT JSONL
  -> MOSS-TTS SFT
  -> VC inference + MOSS codec decode
```

## Path Conventions

The default remote paths are assumed to be:

```text
/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-TTS
/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC
```

This project does not modify `MOSS-TTS`. It only calls its tokenizer,
processor, model, and fine-tuning scripts.

The default base model is `OpenMOSS-Team/MOSS-TTS` / `MossTTSDelayModel`, whose
audio codebook count is `n_vq=32`. Data preparation, training, and inference
should therefore keep `N_VQ=32`. If you switch to another MOSS-TTS variant,
verify the checkpoint's `config.n_vq` first.

## Script Numbering

Scripts under `scripts/` are grouped by numeric prefixes:

```text
001xxx_* = data construction / codec extraction / speaker, prosody, content,
           semantic offline features / data-processing job submission
002xxx_* = training / training job submission / training monitors
003xxx_* = inference / codec decode
004xxx_* = utilities / smoke tests / visualization / TensorBoard / frontend helpers
```

Old two-digit script names are deprecated. For example,
`05_infer_moss_codecvc.py` is now `scripts/003001_infer_moss_codecvc.py`, and
`run_moss_codecvc_infer.sh` is now `scripts/003003_run_moss_codecvc_infer.sh`.

## Data Schema

`scripts/001001_build_vc_manifest.py` writes a unified JSONL schema:

```json
{
  "sample_id": "case_000000",
  "pair_type": "MOSSCodecVC",
  "language": "zh",
  "source_audio": "/path/source.wav",
  "source_text": "source audio transcript",
  "timbre_ref_audio": "/path/ref.wav",
  "timbre_ref_text": "reference timbre transcript",
  "target_audio": "/path/teacher_target.wav",
  "target_text": "target text, usually the same as source_text",
  "instruction": "Preserve S1 content/prosody/timing and convert it to S2 timbre.",
  "meta": {}
}
```

Supervised training requires `target_audio`. It can come from:

- Seed-VC teacher targets
- MOSI Studio / MOSS-Speech teacher targets
- Manually constructed same-speaker parallel targets
- Identity / reconstruction sanity checks

## Quickstart

### 0. Environment Check

```bash
cd /inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC
PY=/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-tts/bin/python
$PY scripts/004001_check_env.py --config configs/default.yaml
```

### 1. Build a VC Manifest

Build from an existing pair JSONL:

```bash
$PY scripts/001001_build_vc_manifest.py \
  --input-jsonl /path/to/pairs.jsonl \
  --output-jsonl data/manifests/vc_manifest.jsonl \
  --mode from_pairs
```

Build from a source-data JSONL by selecting another sample in the batch as the
timbre reference:

```bash
$PY scripts/001001_build_vc_manifest.py \
  --input-jsonl /path/to/source.jsonl \
  --output-jsonl data/manifests/vc_manifest.jsonl \
  --mode from_sources \
  --timbre-offset 1
```

### 2. Extract Codec Tokens

```bash
$PY scripts/001002_encode_codec_tokens.py \
  --config configs/default.yaml \
  --input-jsonl data/manifests/vc_manifest.jsonl \
  --output-jsonl data/encoded/vc_manifest.encoded.jsonl \
  --codes-dir data/codes \
  --n-vq 32 \
  --device cuda:0
```

### 3. Build the MOSS SFT JSONL

```bash
$PY scripts/001003_build_moss_sft_jsonl.py \
  --input-jsonl data/encoded/vc_manifest.encoded.jsonl \
  --output-jsonl data/sft/moss_codecvc_sft.jsonl
```

### 4. Train

Run a smoke test without loading the full 8B weights first. This checks the
processor, `n_vq`, and label-mask alignment:

```bash
$PY scripts/004002_smoke_sft_batch.py --config configs/default.yaml
```

If a complete local model directory is available on the remote machine:

```bash
$PY scripts/004002_smoke_sft_batch.py --config configs/remote_full.yaml
```

#### H200 Full-SFT Path

For a single 8-GPU H200 80G node, Ver1 can use full-SFT/FSDP:

```bash
TRAIN_JSONL=data/sft/moss_codecvc_sft.jsonl \
OUTPUT_DIR=outputs/sft/moss_codecvc_ver1_full_sft \
MODEL_PATH=/path/to/MOSS-TTS \
CODEC_PATH=/path/to/MOSS-Audio-Tokenizer \
N_VQ=32 \
ACCELERATE_CONFIG_FILE=configs/accelerate_fsdp_h200_8gpu.yaml \
PER_DEVICE_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=2 \
bash scripts/002001_train_moss_codecvc_sft.sh
```

For two H200 nodes with 16 GPUs total, use
`configs/accelerate_fsdp_h200_16gpu_2node.yaml`, and set
`machine_rank`, `main_process_ip`, and `main_process_port` on each node. A job
submission script can copy this YAML into the run directory and patch those
fields before launch.

#### LoRA Sanity / Ablation

LoRA is used for fast data and task-adaptation checks. It saves an adapter
instead of a full base model.

The environment must have `peft` installed. The full-SFT/FSDP path does not
depend on `peft`.

```bash
TRAIN_JSONL=data/sft/moss_codecvc_sft.jsonl \
OUTPUT_DIR=outputs/lora/moss_codecvc_ver1_lora \
MODEL_PATH=/path/to/MOSS-TTS \
CONFIG=configs/remote_full.yaml \
N_VQ=32 \
MAX_TRAIN_STEPS=20 \
SMOKE_TEST=1 \
bash scripts/002003_train_moss_codecvc_lora.sh
```

### 5. Inference

```bash
$PY scripts/003001_infer_moss_codecvc.py \
  --config configs/default.yaml \
  --model-path outputs/sft/moss_codecvc_pilot \
  --source-audio /path/source.wav \
  --timbre-ref-audio /path/ref.wav \
  --text "text to convert, or the source ASR transcript" \
  --output-wav outputs/infer/case.wav \
  --n-vq 32
```

## Current Implementation

- Unified data cleaning and pair-manifest construction
- MOSS-Audio-Tokenizer codec-token extraction and caching
- Conversion into the existing MOSS-TTS SFT training format
- Training shell wrapper
- Inference wrapper
- Design hooks for codebook routing, length lock, and source/timbre disentanglement

## Key Next Experiments

1. Use Seed-VC 500/500 outputs as teacher targets and train a small LoRA /
   short-step sanity run.
2. Compare `reference=[source, timbre]` against
   `reference=[timbre] + source_codes in instruction`.
3. Add length-locked decoding to constrain the output codec-frame count near the
   source frame count.
4. Add speaker/prosody evaluation: source-sim drop, target-sim rise, duration
   ratio, pause Jaccard, and F0/energy correlation.

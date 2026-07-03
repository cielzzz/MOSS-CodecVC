# Formal LoRA Experiment: zh1000 + en1000

## Goal

Train the first formal bilingual LoRA baseline for MOSS-CodecVC with balanced Chinese and English data.

## Status

- `target_formal_dataset`: `zh1000 + en1000 = 2000` samples total
- `current_available_dataset`: `zh500 + en500 = 1000` samples total
- `current_blocker`: the repo does not yet contain a real `zh1000 + en1000` SFT JSONL

This experiment is therefore split into two stages:

1. `warmup_available_1000`: start immediately with the currently available balanced bilingual set.
2. `formal_target_2000`: switch to the true `zh1000 + en1000` set once the missing data is prepared.

## Dataset Paths

- Current warmup set:
  `experiments/2026-06-16_formal_lora_zh1000_en1000/data/train_current_balanced_zh500_en500.jsonl`
- Target formal set:
  `experiments/2026-06-16_formal_lora_zh1000_en1000/data/train_target_zh1000_en1000.jsonl`

## Training Recommendation

### Stage A: warmup on current available 1000 rows

- Data: `zh500 + en500`
- Optimizer steps per epoch:
  `1000 / gradient_accumulation_steps(8) ~= 125`
- Recommended steps: `600`
- Effective epochs: about `4.8`
- Purpose:
  - validate bilingual fit beyond smoke test
  - observe whether target timbre improves without collapsing prosody
  - decide whether to scale the same hyperparameters to the full 2000-row run

### Stage B: formal run on target 2000 rows

- Data: `zh1000 + en1000`
- Optimizer steps per epoch:
  `2000 / gradient_accumulation_steps(8) ~= 250`
- Recommended steps: `1000`
- Effective epochs: about `4.0`
- Purpose:
  - establish the first real Ver1 bilingual LoRA baseline
  - produce checkpoints for manual inference comparison
  - create a stable baseline before Ver2/Ver3 feature work

## Save / Logging

- `logging_steps=1`
- `save_steps=100`
- TensorBoard:
  `outputs/lora_runs/formal_lora_zh1000_en1000_{stage}/tensorboard`

`save_steps=100` is deliberate:
- warmup run gets checkpoints near `step-100`, `200`, `300`, `400`, `500`, `600`
- formal run gets checkpoints every ~`0.4` epoch

## Primary Metrics

### Online train metrics

- `train/loss`
- `train/lr`
- `train/lora_grad_norm`

### Offline VC quality checks

At each saved checkpoint, run a small fixed bilingual probe set and check:

- target speaker similarity rises
- source speaker similarity drops, but not catastrophically
- duration ratio stays near `1.0`
- pause pattern remains close to source
- Mandarin and English both remain intelligible

## Decision Rules

- Continue the warmup run if `loss` drops smoothly and sample quality improves by `step 200-300`.
- Stop or lower LR if loss becomes unstable or outputs collapse into short/noisy audio.
- Promote to the 2000-row formal run only after the warmup run shows clear bilingual signal.

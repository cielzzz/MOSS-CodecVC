# v1sec_seq_pure Fix 6 Result Archive

Date: 2026-07-09

## Run

- Arm: `v1sec_seq_pure`
- Job ID: `job-421181e5-024c-4b13-abdb-6de9f5e4e879`
- Output dir: `/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/outputs/lora_runs/ver2_9_v1sec_seq_pure_olddata_steps3000`
- Eval root: `/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC/testset/outputs/ver2_9_v1sec_seq_pure_olddata_quick_eval`
- Eval set: old SeedTTS quick20, `testset/validation/ver2_9_seq_ver2_8_timbre_quick20_seedtts_no_text_20260704.jsonl`
- Data/control: old data, same repeat as V1 second, `no_text.train.jsonl::repeat=1,text.train.jsonl::repeat=10`
- Architecture delta vs V1 second: add WavLM layer-9 speaker sequence cross-attn; keep AdaLN + K/V bias; no old timbre memory; no source semantic memory.

The QZ job was stopped on 2026-07-09 after the negative 2500-step readout. The final local `train.log` reached step 2780 before the stop completed, but there is no step-3000 checkpoint or step-3000 quick20. This archive uses the evaluated checkpoints through step 2500.

## Quick20 Metrics

| step | primary | CER | keep | fail | repeat | duration | sim(ref) | sim(src) | ref-bound | ref-content F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.1090 | 0.0953 | 17/20 | 0.15 | 0.0528 | 0.9919 | 0.1444 | 0.6136 | 0/20 | 0.0625 |
| 1000 | 0.1412 | 0.1335 | 16/20 | 0.20 | 0.0692 | 0.9919 | 0.1522 | 0.4970 | 1/20 | 0.0641 |
| 1500 | 0.0669 | 0.0533 | 20/20 | 0.00 | 0.0071 | 0.9865 | 0.1849 | 0.5145 | 0/20 | 0.0643 |
| 2000 | 0.1352 | 0.1096 | 17/20 | 0.15 | 0.0464 | 0.9927 | 0.1696 | 0.4807 | 0/20 | 0.0596 |
| 2500 | 0.1046 | 0.0937 | 17/20 | 0.15 | 0.0303 | 0.9927 | 0.1808 | 0.4343 | 3/20 | 0.0624 |

Baseline V1 second old-data 3k: CER `0.057`, fail `0.00`, keep `20/20`, sim(ref) `0.201`, sim(src) `0.463`.

## Training Signals

| step | AdaLN gate | seq cross-attn gate | seq token norm | seq delta ratio | note |
|---:|---:|---:|---:|---:|---|
| 20 | 0.5000 | 0.3770 | 63.9991 | 0.0398 | early training reference |
| 500 | 0.5112 | 0.3666 | 64.0012 | 0.0375 | checkpoint |
| 1000 | 0.5213 | 0.3580 | 64.0015 | 0.0371 | token/delta from step 980 because step 1000 had speaker dropout |
| 1500 | 0.5296 | 0.3504 | 64.0028 | 0.0367 | checkpoint |
| 2000 | 0.5372 | 0.3436 | 64.0043 | 0.0355 | token/delta from step 1980 because step 2000 had speaker dropout |
| 2500 | 0.5440 | 0.3373 | 64.0044 | 0.0350 | checkpoint |

Key trajectories:

- sim(src) decreases from `0.6136` at step 500 to `0.4343` at step 2500, so the sequence pathway is active and pushes away from source.
- sim(ref) peaks at `0.1849` at step 1500, below the V1 second baseline `0.201`.
- Sequence gate is actively down-weighted from `0.3770` early to `0.3373` at step 2500, while AdaLN gate rises from `0.5000` to `0.5440`.
- Content is not the blocker at the best point: step 1500 has `CER 0.0533`, `keep 20/20`.

## Interpretation

This run is a negative result for the third-layer diagnosis. WavLM layer-9 speaker sequence features `[T_ref, 768]` do not outperform the single frozen speaker-vector broadcast on the old-data quick20 benchmark.

The sequence representation is not dead: it reduces source similarity substantially. The failure mode is that the model does not anchor to the target reference speaker; it instead lowers the sequence gate and relies more on the AdaLN broadcast path. This supports the next diagnosis that the AR LM speaker-conditioning path has an architectural ceiling around sim(ref) `0.20` for these side-channel variants.

Suggested paper notes:

- "Speaker sequence representation (WavLM layer 9 `[T_ref, 768]`) does not surpass a single frozen speaker vector broadcast on the old-data quick20 benchmark."
- "The model actively down-weights the sequence pathway while up-weighting the AdaLN broadcast pathway."
- "AR LM speaker conditioning appears to have a ceiling around sim(ref)=0.20 that neither vector nor sequence side-channel representations bypass."

## Files

- `quick20_metrics.csv`: compact step-level quick20 metrics.
- `train_gate_metrics.csv`: gate and sequence-path amplitude trajectory.
- `raw_summaries/`: copied raw summary JSON/CSV for steps 500, 1000, 1500, 2000, and 2500.

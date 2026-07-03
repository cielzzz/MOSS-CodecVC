# MOSS-CodecVC Data Processing Status

Updated UTC: 2026-06-30 10:45

## Current Task

Dataset: `zh11w_en11w_0005_0015_vcdata_first`

Stage 1 is complete: raw `zh/en` JSONL shards `0005` through `0015` have been converted by `vcdata_construction` into cloned MOSS-TTS vcdata manifests.

Current Qizhi jobs:

- `codec-vc-data-process-20260629-121104-2475841-g00`
  - Job ID: `job-727c3214-76ad-4a9f-98db-765f15d8d3fb`
  - Priority: `10`
  - Compute group: `MTTS-3-2-0715`
  - Resource: `8xH200`
  - Status: `job_succeeded`
  - Finished UTC: `2026-06-30 08:06:02`
- `codec-vc-data-process-20260629-121104-2475841-g01`
  - Job ID: `job-f6c39cba-9712-44b2-a9eb-1dc30915fa53`
  - Priority: `10`
  - Compute group: `MTTS-3-2-0715`
  - Resource: `8xH200`
  - Status: `job_succeeded`

Latest local progress sample:

- `en`: `110000 / 110000`, complete
- `zh`: `110000 / 110000`, complete
- total: `220000 / 220000`

The final vcdata input list has `22` entries:

- `trainset/zh11w_en11w_0005_0015_vcdata_first/vcdata_jsonls.txt`

## Branch Task Status

The first branch submission failed because Seed-VC was forced offline but inherited a generic `HF_HOME=$DOWNLOAD_ROOT/huggingface`, so it could not find cached `openai/whisper-small` / BigVGAN files. The needed cache exists under:

- `/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/pair_construction_prosody_routes/third_party/seed-vc/checkpoints/hf_cache`

Failed branch jobs:

- text: `job-5f98dfb1-e933-4ffd-b714-8640cbf76863`
- no_text: `job-695adea8-0229-4816-9a29-6871276ed169`

Fix applied:

- `scripts/001035_run_text_prosody_mosstts_seedvc_pipeline.sh` now forces Seed-VC `HF_HOME/HUGGINGFACE_HUB_CACHE/TRANSFORMERS_CACHE` to `seed-vc/checkpoints/hf_cache`.
- `scripts/001050_run_vcdata_text_no_text_pipeline.sh` does the same for the no_text Seed-VC stage.

Resubmitted branch jobs, high priority `10`, batch `20260630-104452-hfcachefix`:

- text: `job-72ec9280-7bb8-4c6e-9eb4-2d8d3bdac765`
- no_text: `job-d58ca762-2599-44ca-9d54-47c39d959770`

Current status at update time: both are `job_running`.

- text node: `qb-prod-gpu933`
- no_text node: `qb-prod-gpu830`

## Watcher

Local watcher has already submitted the original branch jobs. Those failed and have been superseded by the manual `hfcachefix` resubmission above.

- Script: `scripts/001052_watch_vcdata_then_submit_branches.sh`
- PID file: `trainset/zh11w_en11w_0005_0015_vcdata_first/qz_jobs/vcdata_branch_watch/watch.pid`
- Log: `trainset/zh11w_en11w_0005_0015_vcdata_first/qz_jobs/vcdata_branch_watch/nohup.log`
- Progress: `trainset/zh11w_en11w_0005_0015_vcdata_first/qz_jobs/vcdata_branch_watch/progress.tsv`
- Submit marker after success: `trainset/zh11w_en11w_0005_0015_vcdata_first/qz_jobs/vcdata_branch_watch/submitted.done`

Watcher submit settings:

- `PRIORITY=10`
- `BRANCHES=text,no_text`
- `ENABLE_TRAIN_READY_GPU_KEEPALIVE=1`
- `ENABLE_SEMANTIC_GPU_KEEPALIVE=1`
- `TEXT_SEMANTIC_GPU_KEEPALIVE=1`
- `NO_TEXT_SEMANTIC_GPU_KEEPALIVE=1`

## Next Automatic Stage

When Stage 1 completes, the watcher will merge every split with:

- `vcdata/<split>/manifest_merged.jsonl`
- symlink: `vcdata/<split>/merged.stepaudio_input.all.jsonl`
- merged list: `trainset/zh11w_en11w_0005_0015_vcdata_first/vcdata_jsonls.txt`

Then it will submit two high-priority Qizhi branch jobs through:

- `scripts/001051_submit_vcdata_branch_pipelines_qz.sh`

Expected branch datasets:

- text branch: `trainset/zh11w_en11w_0005_0015_vcdata_first_text_prosody`
- no_text branch: `trainset/zh11w_en11w_0005_0015_vcdata_first_no_text`

The branch runner is:

- `scripts/001050_run_vcdata_text_no_text_pipeline.sh`

Branch semantics:

- `text`: independent-timbre `text_prosody`; builds text-guided triples, then runs Seed-VC target generation, codec encode, SFT, ECAPA, prosody, CTC content tokens and target HuBERT.
- `no_text`: random original different-row timbre reference; builds no_text triples, runs Seed-VC target generation, codec encode, SFT, ECAPA, prosody, ASR cleaning, CTC content tokens and HuBERT.

## Script Updates In This Pass

- Added `SEMANTIC_GPU_KEEPALIVE` support to `scripts/001018_prepare_ver2_1_content_semantic_68w.sh` for no_text semantic processing.
- Wired `NO_TEXT_SEMANTIC_GPU_KEEPALIVE` through `scripts/001050_run_vcdata_text_no_text_pipeline.sh`.
- Updated `scripts/001051_submit_vcdata_branch_pipelines_qz.sh` and `scripts/001052_watch_vcdata_then_submit_branches.sh` so semantic keepalive defaults to enabled for both text and no_text branch jobs.
- Verified with `bash -n` on all changed shell scripts.

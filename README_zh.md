# MOSS-CodecVC

MOSS-CodecVC 是一个基于 MOSS-TTS / MOSS-Audio-Tokenizer 的 codec-native voice conversion 框架骨架。

目标任务不是普通 TTS，而是：

```text
source wav 的内容/停顿/时长/节奏 + timbre reference wav 的音色 -> target wav
```

## 模型图

- [MOSS-VC / MOSS-CodecVC 模型结构图](docs/moss_codecvc_model_diagram.md)
- [SVG 原图](docs/assets/moss_codecvc_model_diagram.svg)
- [Ver1/Ver2/Ver3/Ver4 版本规划与消融设计](docs/versions_and_ablation.md)
- [Ver2 大规模 LoRA 训练策略](docs/ver2_large_scale_training_strategy.md)

第一版先复用现有 `MOSS-TTS/moss_tts_delay/finetuning/sft.py` 做 teacher-forcing SFT：

```text
VC manifest
  -> MOSS audio codec token extraction
  -> MossTTSDelay SFT JSONL
  -> MOSS-TTS SFT
  -> VC inference + MOSS codec decode
```

## 路径约定

默认假设远端路径：

```text
/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-TTS
/inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC
```

本项目不修改 `MOSS-TTS`，只调用其中的 tokenizer、processor、model 和 finetuning 脚本。

默认基座是 `OpenMOSS-Team/MOSS-TTS` / `MossTTSDelayModel`，其 audio codebook 数为 `n_vq=32`。
因此准备数据、训练、推理应保持 `N_VQ=32`；如果换成其他 MOSS-TTS 变体，需要先确认对应 checkpoint 的 `config.n_vq`。

## 脚本编号约定

`scripts/` 目录已按用途统一重命名，后续文档和任务脚本都应使用新编号：

```text
001xxx_* = 数据构造 / codec 提取 / speaker、prosody、content、semantic 离线特征 / 数据处理任务提交
002xxx_* = 训练 / 训练任务提交 / 训练任务巡检
003xxx_* = 推理 / codec decode
004xxx_* = 工具 / smoke test / 可视化 / TensorBoard / 前端辅助
```

旧的两位数脚本名已经废弃，例如 `05_infer_moss_codecvc.py` 对应现在的 `scripts/003001_infer_moss_codecvc.py`，`run_moss_codecvc_infer.sh` 对应现在的 `scripts/003003_run_moss_codecvc_infer.sh`。

## 数据 schema

`scripts/001001_build_vc_manifest.py` 输出统一 JSONL：

```json
{
  "sample_id": "case_000000",
  "pair_type": "MOSSCodecVC",
  "language": "zh",
  "source_audio": "/path/source.wav",
  "source_text": "源音频文本",
  "timbre_ref_audio": "/path/ref.wav",
  "timbre_ref_text": "参考音色文本",
  "target_audio": "/path/teacher_target.wav",
  "target_text": "目标文本，默认等于 source_text",
  "instruction": "Preserve S1 content/prosody/timing and convert it to S2 timbre.",
  "meta": {}
}
```

监督训练需要 `target_audio`。可以来自：

- Seed-VC teacher target
- MOSI Studio / MOSS-Speech teacher target
- 人工构造的同说话人 parallel target
- identity/reconstruction sanity check

## 快速流程

### 0. 环境检查

```bash
cd /inspire/qb-ilm2/project/embodied-multimodality/public/xyzhang/projects/MOSS-CodecVC
PY=/inspire/ssd/project/embodied-multimodality/public/xyzhang/anaconda3/envs/moss-tts/bin/python
$PY scripts/004001_check_env.py --config configs/default.yaml
```

### 1. 构造 VC manifest

从已有 pair JSONL 构造：

```bash
$PY scripts/001001_build_vc_manifest.py \
  --input-jsonl /path/to/pairs.jsonl \
  --output-jsonl data/manifests/vc_manifest.jsonl \
  --mode from_pairs
```

从普通源数据 JSONL 中 batch 内挑另一个样本做音色参考：

```bash
$PY scripts/001001_build_vc_manifest.py \
  --input-jsonl /path/to/source.jsonl \
  --output-jsonl data/manifests/vc_manifest.jsonl \
  --mode from_sources \
  --timbre-offset 1
```

### 2. 提取 codec tokens

```bash
$PY scripts/001002_encode_codec_tokens.py \
  --config configs/default.yaml \
  --input-jsonl data/manifests/vc_manifest.jsonl \
  --output-jsonl data/encoded/vc_manifest.encoded.jsonl \
  --codes-dir data/codes \
  --n-vq 32 \
  --device cuda:0
```

### 3. 构建 MOSS SFT JSONL

```bash
$PY scripts/001003_build_moss_sft_jsonl.py \
  --input-jsonl data/encoded/vc_manifest.encoded.jsonl \
  --output-jsonl data/sft/moss_codecvc_sft.jsonl
```

### 4. 训练

先做一次不加载 8B 权重的 SFT batch smoke，确认 processor、`n_vq` 和 label mask 对齐：

```bash
$PY scripts/004002_smoke_sft_batch.py --config configs/default.yaml
```

如果使用远端已有完整本地权重目录：

```bash
$PY scripts/004002_smoke_sft_batch.py --config configs/remote_full.yaml
```

#### H200 full-SFT 主线

如果有单机 8 卡 H200 80G，Ver1 可以直接走 full-SFT/FSDP：

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

双机 16 卡 H200 时使用 `configs/accelerate_fsdp_h200_16gpu_2node.yaml`，并在两台机器分别设置 `machine_rank / main_process_ip / main_process_port`。实际提交脚本里可以复制该 YAML 到 run 目录后替换这些字段。

#### LoRA sanity / ablation

LoRA 用于快速验证数据和任务适配，输出是 adapter，不会保存完整 base 模型：

需要环境中安装 `peft`；full-SFT/FSDP 路线不依赖 `peft`。

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

### 5. 推理

```bash
$PY scripts/003001_infer_moss_codecvc.py \
  --config configs/default.yaml \
  --model-path outputs/sft/moss_codecvc_pilot \
  --source-audio /path/source.wav \
  --timbre-ref-audio /path/ref.wav \
  --text "需要转换的文本或 source ASR 文本" \
  --output-wav outputs/infer/case.wav \
  --n-vq 32
```

## 现在这版实现了什么

- 数据清洗和 pair manifest 统一
- MOSS-Audio-Tokenizer codec token 提取与缓存
- 转成现有 MOSS-TTS SFT 可训练格式
- 训练 shell wrapper
- 推理 wrapper
- 预留 codebook routing / length lock / source-timbre disentangle 的模型设计文档

## 后续关键实验

1. 用 Seed-VC 500/500 结果作为 teacher target，先训一个小 LoRA/小步数 sanity。
2. 比较 `reference=[source, timbre]` vs `reference=[timbre] + source_codes in instruction`。
3. 加 length-locked decoding：限制输出 codec frame 数接近 source frame 数。
4. 加 speaker/prosody 评估：source sim drop、target sim rise、duration ratio、pause jaccard、F0/energy corr。

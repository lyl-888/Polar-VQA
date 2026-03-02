# Polar LLaVA 训练命令参考

## 🖥️ 硬件配置

**当前配置：双卡 RTX 4090 24GB**

- **GPU 数量**: 2 × RTX 4090 (24GB 显存/卡)
- **训练方式**: 分布式数据并行（DDP）
- **启动方式**: 使用 `torchrun --nproc_per_node=2` 启动双卡训练
- **Batch Size 配置**:
  - `per_device_train_batch_size=1`: 每张卡每次处理 1 个样本
  - `gradient_accumulation_steps=16`: 梯度累积步数（双卡时减半）
  - **总 batch size** = 1 × 2 GPUs × 16 = 32（与单卡配置保持一致）

**⚠️ 注意事项**：
- 确保两张 GPU 都可用：`nvidia-smi` 检查
- 如果只有单卡，将 `--nproc_per_node=2` 改为 `--nproc_per_node=1`，并将 `gradient_accumulation_steps` 改回 32
- `--master_port` 用于多进程通信，如果端口被占用，可以改为其他端口（如 29502, 29503 等）

## 📁 模型路径

- **LLaVA 模型**: `/openbayes/input/input0/models/llava-v1.5-13b`
- **CLIP 模型**: `/openbayes/input/input0/models/clip-vit-large-patch14-336`
- **VAE 模型**: `/openbayes/input/input0/models/sd-vae-ft-mse`

## 📁 数据路径说明

### Stage 1 训练（Polar Projector Alignment）

**数据格式**：使用 **crop 数据**（512x512 裁剪后的图像）

- **训练集 JSON**: `train_stage2_data.json`
  - 包含 `polar_crop_paths` 字段
  - 图像路径格式：`rgb_crop/23/0000_rgb.png`
- **验证集 JSON**: `val_stage2_data.json`
- **数据根目录**: `/openbayes/input/input0`
- **RGB 图像目录**: `/openbayes/input/input0/rgb_crop`（**带 crop**）
- **Polar 图像目录**: `/openbayes/input/input0/polar_crop`（**带 crop**）

### Stage 2 训练（Visual Instruction Tuning）

**数据格式**：使用 **非裁剪数据**（原始尺寸图像，会在数据加载时 resize）

- **训练集 JSON**: `merged_stage3_qwen.json`
  - 包含 `input_path` 字段，格式：`rgb/04/0002_rgb.png`
  - 偏振路径从 RGB 路径推断：`polar/04/0002_000.png` 等
- **验证集 JSON**: `val_stage3_qwen.json`
- **数据根目录**: `/openbayes/input/input0`
- **RGB 图像目录**: `/openbayes/input/input0/rgb`（**不带 crop**）
- **Polar 图像目录**: `/openbayes/input/input0/polar`（**不带 crop**）

## 🚀 Stage 1: Polar Projector Alignment

**⚠️ 显存优化（双卡 RTX 4090 24GB）**：
- 使用 4-bit 量化（QLoRA）降低 LLM 显存占用
- **双卡分布式训练**：使用 `torchrun --nproc_per_node=2` 启动
- `per_device_train_batch_size=1`：减小单次前向传播的显存占用（**最有效**）
- `gradient_accumulation_steps=16`：双卡时减少一半，总 batch size = 1 × 2 GPUs × 16 = 32（与单卡保持一致）
- `gradient_checkpointing=False`：已禁用（Stage 1 不需要）
- Polar Projector 保持 BF16 精度（不会被量化）

**实时输出配置**：
- 代码已自动配置无缓冲输出，确保在 `nohup` 下也能实时看到 loss
- 使用 `python -u` 运行，确保实时输出

```bash
## Stage 1（Polar Projector 对齐）- 单机双卡 4090 训练命令

### 前台运行（推荐调试时使用）

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m llava.train.train \
  --model_name_or_path /openbayes/input/input0/models/llava-v1.5-13b \
  --version v1 \
  --vision_tower /openbayes/input/input0/models/clip-vit-large-patch14-336 \
  --polar_vae_model_path /openbayes/input/input0/models/sd-vae-ft-mse \
  --training_stage stage1 \
  --freeze_polar_encoder True \
  --freeze_rgb_tower True \
  --freeze_rgb_projector True \
  --data_path train_stage2_data.json \
  --val_json val_stage2_data.json \
  --image_folder /openbayes/input/input0/rgb_crop \
  --polar_folder /openbayes/input/input0/polar_crop \
  --data_root /openbayes/input/input0 \
  --use_polar True \
  --output_dir /openbayes/input/input0/llava_train_stage1_new \
  --bits 4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 5 \
  --learning_rate 5e-4 \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --logging_steps 10 \
  --save_steps 100 \
  --eval_steps 50 \
  --save_total_limit 2 \
  --load_best_model_at_end True \
  --bf16 True \
  --gradient_checkpointing False \
  --dataloader_num_workers 4 \
  --remove_unused_columns False \
  --report_to tensorboard \
  --save_strategy steps \
  --eval_strategy steps
```

### 后台运行（nohup，适合长时间训练）

```bash
CUDA_VISIBLE_DEVICES=0,1 nohup torchrun --nproc_per_node=2 -m llava.train.train \
  --model_name_or_path /openbayes/input/input0/models/llava-v1.5-13b \
  --version v1 \
  --vision_tower /openbayes/input/input0/models/clip-vit-large-patch14-336 \
  --polar_vae_model_path /openbayes/input/input0/models/sd-vae-ft-mse \
  --training_stage stage1 \
  --freeze_polar_encoder True \
  --freeze_rgb_tower True \
  --freeze_rgb_projector True \
  --data_path train_stage2_data.json \
  --val_json val_stage2_data.json \
  --image_folder /openbayes/input/input0/rgb_crop \
  --polar_folder /openbayes/input/input0/polar_crop \
  --data_root /openbayes/input/input0 \
  --use_polar True \
  --output_dir /openbayes/input/input0/llava_train_stage1 \
  --bits 4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --num_train_epochs 5 \
  --learning_rate 5e-4 \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --logging_steps 10 \
  --save_steps 100 \
  --eval_steps 50 \
  --save_total_limit 2 \
  --load_best_model_at_end True \
  --bf16 True \
  --gradient_checkpointing False \
  --dataloader_num_workers 4 \
  --remove_unused_columns False \
  --report_to tensorboard \
  --save_strategy steps \
  --eval_strategy steps \
  > llava_train_stage1.log 2>&1 &
```

```bash
torchrun --nproc_per_node=2 --master_port=29500 -m llava.train.train \
  --model_name_or_path /openbayes/input/input0/models/llava-v1.5-13b \
  --version v1 \
  --vision_tower /openbayes/input/input0/models/clip-vit-large-patch14-336 \
  --polar_vae_model_path /openbayes/input/input0/models/sd-vae-ft-mse \
  --training_stage stage1 \
  --freeze_polar_encoder True \
  --freeze_rgb_tower True \
  --freeze_rgb_projector True \
  --data_path train_stage2_data.json \
  --val_json val_stage2_data.json \
  --image_folder /openbayes/input/input0/rgb_crop \
  --polar_folder /openbayes/input/input0/polar_crop \
  --data_root /openbayes/input/input0 \
  --use_polar True \
  --output_dir /openbayes/input/input0/1.25/stage1 \
  --bits 4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 5 \
  --learning_rate 5e-4 \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --logging_steps 10 \
  --save_steps 100 \
  --eval_steps 50 \
  --save_total_limit 2 \
  --load_best_model_at_end True \
  --bf16 True \
  --gradient_checkpointing False \
  --dataloader_num_workers 4 \
  --remove_unused_columns False \
  --report_to tensorboard \
  --save_strategy steps \
  --eval_strategy steps \
  --ddp_find_unused_parameters False 
```

**⚠️ 重要说明**：
- `--gradient_checkpointing False`: **已关闭梯度检查点**，解决 `grad_norm=0.0` 问题
  - 原因：梯度检查点与冻结的 VAE 配合时，会导致梯度流断开
  - 代价：显存占用会增加，但训练可以正常进行
  - 如果显存不足，可以减小 `per_device_train_batch_size` 或增加 `gradient_accumulation_steps`

**关键参数说明**：
- `--training_stage stage1`: 只训练 Polar Projector
- `--image_folder`: 使用 **rgb_crop** 目录（crop 数据）
- `--polar_folder`: 使用 **polar_crop** 目录（crop 数据）
- `--data_root`: 用于解析 JSON 中的 `rgb_crop/` 和 `polar_crop/` 路径

**输出**：
- `mm_projector.bin`: 包含 `polar_projector` 权重（用于 Stage 2）

## 🚀 Stage 2: Visual Instruction Tuning

**⚠️ 显存优化（双卡 RTX 4090 24GB）**：
- 使用 4-bit 量化（QLoRA）降低 LLM 显存占用
- **双卡分布式训练**：使用 `torchrun --nproc_per_node=2` 启动
- Batch size 配置：`per_device_train_batch_size=1`，`gradient_accumulation_steps=16`
  - 总 batch size = 1 × 2 GPUs × 16 = 32（与单卡配置保持一致）
- Stage 2 需要训练 LoRA + Projector，显存压力更大，保持保守配置
- Polar Projector 保持 BF16 精度（不会被量化）

**实时输出配置**：
- 代码已自动配置无缓冲输出，确保在 `nohup` 下也能实时看到 loss
- 使用 `python -u` 运行，确保实时输出

```bash
# ========== Stage 2: Visual Instruction Tuning (单卡 4090 24GB) ==========
# 注意：如果 Stage 1 使用了量化并冻结了 LoRA，会生成 non_lora_trainables.bin 而不是 mm_projector.bin
# train.py 会自动从 .bin 文件中提取包含 'polar_projector' 的权重，所以直接指向 non_lora_trainables.bin 即可
#
# ⚠️ 重要：关于 --pretrain_polar_projector 路径的选择
# 1. 如果 Stage 1 使用了 --load_best_model_at_end True，训练结束后会加载最佳模型并保存到根目录
#    此时根目录下的 non_lora_trainables.bin 就是最佳模型的权重（推荐使用）
# 2. 如果想使用特定 checkpoint 的权重，可以使用：
#    --pretrain_polar_projector /openbayes/input/input0/llava_train_stage1/checkpoint-600/non_lora_trainables.bin
# 3. 从你的训练日志看，最佳模型是 checkpoint-600，根目录下的 non_lora_trainables.bin 应该就是它的权重
## Stage 2（指令微调）- 单机双卡 4090 训练命令

### 前台运行（推荐调试时使用）

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m llava.train.train \
  --model_name_or_path /openbayes/input/input0/models/llava-v1.5-13b \
  --version v1 \
  --vision_tower /openbayes/input/input0/models/clip-vit-large-patch14-336 \
  --polar_vae_model_path /openbayes/input/input0/models/sd-vae-ft-mse \
  --pretrain_polar_projector /openbayes/input/input0/llava_train_stage1_new/non_lora_trainables.bin \
  --training_stage stage2 \
  --freeze_polar_encoder True \
  --freeze_rgb_tower True \
  --freeze_rgb_projector True \
  --lora_enable True \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --lora_bias none \
  --data_path train_stage3_qwen.json \
  --val_json val_stage3_qwen.json \
  --image_folder /openbayes/input/input0/rgb \
  --polar_folder /openbayes/input/input0/polar \
  --data_root /openbayes/input/input0 \
  --use_polar True \
  --output_dir /openbayes/input/input0/llava_train_stage2_new1 \
  --bits 4 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 5 \
  --learning_rate 2e-4 \
  --mm_projector_lr 2e-5 \
  --warmup_ratio 0.03 \
  --weight_decay 0.05 \
  --logging_steps 4 \
  --save_steps 100 \
  --eval_steps 50 \
  --save_total_limit 2 \
  --load_best_model_at_end True \
  --bf16 True \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --remove_unused_columns False \
  --report_to tensorboard \
  --save_strategy steps \
  --eval_strategy steps
```

### 后台运行（nohup，适合长时间训练）

```bash
CUDA_VISIBLE_DEVICES=0,1 nohup torchrun --nproc_per_node=2 -m llava.train.train \
  --model_name_or_path /openbayes/input/input0/models/llava-v1.5-13b \
  --version v1 \
  --vision_tower /openbayes/input/input0/models/clip-vit-large-patch14-336 \
  --polar_vae_model_path /openbayes/input/input0/models/sd-vae-ft-mse \
  --pretrain_polar_projector /openbayes/input/input0/llava_train_stage1/non_lora_trainables.bin \
  --training_stage stage2 \
  --freeze_polar_encoder True \
  --freeze_rgb_tower True \
  --freeze_rgb_projector True \
  --lora_enable True \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --lora_bias none \
  --data_path train_stage3_qwen.json \
  --val_json val_stage3_qwen.json \
  --image_folder /openbayes/input/input0/rgb \
  --polar_folder /openbayes/input/input0/polar \
  --data_root /openbayes/input/input0 \
  --use_polar True \
  --output_dir /openbayes/input/input0/llava_train_stage2_new \
  --bits 4 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 5 \
  --learning_rate 2e-4 \
  --mm_projector_lr 2e-5 \
  --warmup_ratio 0.03 \
  --weight_decay 0.05 \
  --logging_steps 4 \
  --save_steps 100 \
  --eval_steps 50 \
  --save_total_limit 2 \
  --load_best_model_at_end True \
  --bf16 True \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --remove_unused_columns False \
  --report_to tensorboard \
  --save_strategy steps \
  --eval_strategy steps \
  > llava_train_stage2_new.log 2>&1 &
```
```

## ✅ 参数兼容性检查

**所有参数均已验证支持**：

| 参数 | 类型 | 支持状态 | 说明 |
|------|------|---------|------|
| `--model_name_or_path` | ModelArguments | ✅ | 基础模型路径 |
| `--version` | ModelArguments | ✅ | 对话模板版本 |
| `--vision_tower` | ModelArguments | ✅ | CLIP 视觉编码器路径 |
| `--polar_vae_model_path` | ModelArguments | ✅ | Polar VAE 编码器路径 |
| `--pretrain_polar_projector` | ModelArguments | ✅ | Stage 1 权重路径 |
| `--training_stage` | ModelArguments | ✅ | 训练阶段（stage2） |
| `--freeze_polar_encoder` | ModelArguments | ✅ | 冻结 Polar 编码器 |
| `--freeze_rgb_tower` | ModelArguments | ✅ | 冻结 RGB 视觉塔 |
| `--freeze_rgb_projector` | ModelArguments | ✅ | 冻结 RGB 投影器 |
| `--lora_enable` | TrainingArguments | ✅ | 启用 LoRA |
| `--lora_r` | TrainingArguments | ✅ | LoRA rank |
| `--lora_alpha` | TrainingArguments | ✅ | LoRA alpha |
| `--lora_dropout` | TrainingArguments | ✅ | LoRA dropout |
| `--lora_bias` | TrainingArguments | ✅ | LoRA bias 设置 |
| `--data_path` | DataArguments | ✅ | 训练数据 JSON |
| `--val_json` | DataArguments | ✅ | 验证数据 JSON |
| `--image_folder` | DataArguments | ✅ | RGB 图像目录 |
| `--polar_folder` | DataArguments | ✅ | Polar 图像目录 |
| `--data_root` | DataArguments | ✅ | 数据根目录 |
| `--use_polar` | DataArguments | ✅ | 启用 Polar 模态 |
| `--output_dir` | TrainingArguments | ✅ | 输出目录 |
| `--bits` | TrainingArguments | ✅ | 量化位数（4-bit） |
| `--per_device_train_batch_size` | TrainingArguments | ✅ | 每设备批次大小 |
| `--gradient_accumulation_steps` | TrainingArguments | ✅ | 梯度累积步数 |
| `--num_train_epochs` | TrainingArguments | ✅ | 训练轮数 |
| `--learning_rate` | TrainingArguments | ✅ | LoRA 学习率 |
| `--mm_projector_lr` | TrainingArguments | ✅ | Projector 学习率（独立） |
| `--warmup_ratio` | TrainingArguments | ✅ | Warmup 比例 |
| `--weight_decay` | TrainingArguments | ✅ | 权重衰减 |
| `--logging_steps` | TrainingArguments | ✅ | 日志记录步数 |
| `--save_steps` | TrainingArguments | ✅ | 保存步数 |
| `--eval_steps` | TrainingArguments | ✅ | 验证步数 |
| `--save_total_limit` | TrainingArguments | ✅ | 保存检查点数量限制 |
| `--load_best_model_at_end` | TrainingArguments | ✅ | 训练结束加载最佳模型 |
| `--bf16` | TrainingArguments | ✅ | BF16 混合精度 |
| `--gradient_checkpointing` | TrainingArguments | ✅ | 梯度检查点 |
| `--dataloader_num_workers` | TrainingArguments | ✅ | 数据加载器工作进程数 |
| `--remove_unused_columns` | TrainingArguments | ✅ | 移除未使用列 |
| `--report_to` | TrainingArguments | ✅ | 报告后端（tensorboard） |
| `--save_strategy` | TrainingArguments | ✅ | 保存策略 |
| `--eval_strategy` | TrainingArguments | ✅ | 验证策略 |

**关键功能验证**：
- ✅ `mm_projector_lr` 支持：`llava_trainer.py` 第178-207行实现了独立学习率设置
- ✅ Polar Projector 加载：`train.py` 第1512-1624行支持从 `non_lora_trainables.bin` 加载
- ✅ Stage 2 训练逻辑：`train.py` 第1654-1713行完整实现了 Stage 2 训练流程
- ✅ LoRA + Projector 联合训练：代码已正确配置梯度流

**⚠️ 重要说明**：

1. **关于 `--pretrain_polar_projector` 路径**：
   - 如果 Stage 1 使用了量化（`--bits 4`）并启用了 LoRA（即使冻结），训练完成后会生成 `non_lora_trainables.bin` 而不是 `mm_projector.bin`
   - `train.py` 会自动从 `.bin` 文件中提取包含 `polar_projector` 的权重（见代码第1522-1531行）
   - **如果 Stage 1 使用了 `--load_best_model_at_end True`**：
     - 训练结束后会加载最佳模型（如 `checkpoint-600`）并保存到根目录
     - **根目录下的 `non_lora_trainables.bin` 就是最佳模型的权重（推荐使用）**
     - 路径：`/openbayes/input/input0/llava_train_stage1/non_lora_trainables.bin`
   - **如果想使用特定 checkpoint 的权重**（不推荐，除非有特殊需求）：
     - 路径：`/openbayes/input/input0/llava_train_stage1/checkpoint-600/non_lora_trainables.bin`
     - 注意：需要确认该 checkpoint 目录下是否有 `non_lora_trainables.bin` 文件
   - 如果 Stage 1 没有使用 LoRA，则生成 `mm_projector.bin`，指向该文件即可

2. **单卡 4090 24GB 配置**：
   - `--per_device_train_batch_size 2`: 单卡显存允许，相比多卡配置提升了并行度
   - `--gradient_accumulation_steps 8`: 有效 batch size = 2 × 8 = 16
   - `--gradient_checkpointing True`: **已启用**，节省显存（单卡建议开启）
   - 如果显存不足（OOM），可以：
     - 降低 `per_device_train_batch_size` 到 1
     - 增加 `gradient_accumulation_steps` 到 16（保持有效 batch size）

3. **学习率设置（参考官方 LLaVA 1.5 LoRA 配置）**：
   - `--learning_rate 2e-4`: **LoRA 学习率**，参考官方 665k 数据集配置（`v1_5/finetune_lora.sh`）
     - 官方使用 `2e-4` 用于 665k 数据集，7500 数据量虽然更小，但训练 10 个 epochs，所以使用相同学习率是合理的
     - 如果训练不稳定，可以降低到 `1e-4` 或 `1.5e-4`
   - `--mm_projector_lr 2e-5`: **Projector 学习率**（包括 `polar_projector` 和 `mm_projector`）
     - 官方使用 `2e-5` 用于 projector，通常比 LoRA 学习率低 10 倍
     - 这样可以避免 projector 更新过快导致训练不稳定
   - `--warmup_ratio 0.03`: 参考官方设置（从 0.1 降低到 0.03）
     - 官方使用 0.03，对于 7500 数据量更合适（warmup 步数更少）

4. **关键参数说明**：
   - `--pretrain_polar_projector`: **必须提供** Stage 1 训练的权重路径（`non_lora_trainables.bin` 或 `mm_projector.bin`）
   - `--training_stage stage2`: 训练 Polar Projector + LLM (LoRA)
   - `--image_folder`: 使用 **rgb** 目录（非裁剪数据）
   - `--polar_folder`: 使用 **polar** 目录（非裁剪数据）
   - `--data_root`: 用于解析可能的 crop 路径（如果 JSON 中有）

**输出**：
- `non_lora_trainables.bin`: 包含 `polar_projector` 权重（更新后的，如果使用 LoRA）
- LoRA 适配器：`adapter_model.safetensors`、`adapter_config.json`

## ⏱️ 训练时间估算（7500 数据量）

**配置参数**：
- 数据量：7500 条
- Batch size：`per_device_train_batch_size=2`, `gradient_accumulation_steps=8`
- 有效 batch size：2 × 8 = 16
- 训练轮数：10 epochs
- 硬件：单卡 RTX 4090 24GB

**计算**：
- 每轮步数 = 7500 ÷ 16 ≈ **469 步**
- 总训练步数 = 469 × 10 = **4690 步**
- 单步时间估算（4090，4-bit 量化，gradient_checkpointing=True）：
  - 前向传播：~0.5-0.8 秒
  - 反向传播：~0.8-1.2 秒
  - 优化器更新：~0.1-0.2 秒
  - **单步总时间：~1.5-2.2 秒/步**（取平均 1.8 秒/步）

**总训练时间估算**：
- 纯训练时间：4690 × 1.8 秒 ≈ **2.4 小时**
- 加上验证、保存、数据加载开销：**约 3-4 小时**
- 如果启用验证集（每 100 步验证一次）：**约 4-5 小时**

**优化建议**：
- 如果训练时间过长，可以减少 `num_train_epochs` 到 5-7
- 如果显存充足，可以增加 `per_device_train_batch_size` 到 3-4，减少总步数

## ⚠️ 重要提示

### 显存优化（RTX 4090 24GB）

1. **4-bit 量化（QLoRA）**：
   - 必须使用 `--bits 4` 参数，否则 13B 模型无法加载到 24GB 显存
   - LLM 以 4-bit 加载（约 7-8GB 显存）
   - CLIP、VAE、Projector 保持 16-bit（BF16）
   - **Polar Projector 不会被量化**：代码中已确保 `polar_projector` 保持 BF16 精度

2. **Batch Size 调整（单卡 4090 24GB）**：
   - Stage 1: `per_device_train_batch_size=2`, `gradient_accumulation_steps=16`（总 BS=32）
   - Stage 2: `per_device_train_batch_size=2`, `gradient_accumulation_steps=8`（总 BS=16）
   - 原理：用时间换空间，减小单次并行量，多累积梯度再更新
   - 如果显存不足，可以降低到 `per_device_train_batch_size=1`, `gradient_accumulation_steps=16`（总 BS=16）

3. **验证集确认**：
   - 请确认 `val_stage2_data.json` 和 `val_stage3_qwen.json` 文件存在
   - 如果验证集不存在，请移除 `--val_json` 和 `--eval_steps`、`--load_best_model_at_end` 参数

### 学习率配置说明（参考官方 LLaVA 1.5）

**官方 LLaVA 1.5 LoRA 训练脚本** (`v1_5/finetune_lora.sh`)：
- **数据集**: 665k 样本
- **LoRA 学习率**: `2e-4` (0.0002)
- **Projector 学习率**: `2e-5` (0.00002) - 通过 `--mm_projector_lr` 设置
- **Warmup 比例**: `0.03`
- **训练轮数**: `1` epoch

**当前配置（7500 数据量）**：
- **LoRA 学习率**: `2e-4` - 与官方保持一致（虽然数据量更小，但训练 10 个 epochs）
- **Projector 学习率**: `2e-5` - 与官方保持一致
- **Warmup 比例**: `0.03` - 与官方保持一致
- **训练轮数**: `10` epochs - 因为数据量小，需要更多轮次

**学习率调整建议**：
- 如果训练不稳定（loss 波动大或 NaN）：
  - 降低 LoRA 学习率到 `1e-4` 或 `1.5e-4`
  - 保持 Projector 学习率 `2e-5` 不变
- 如果收敛太慢：
  - 可以尝试提高 LoRA 学习率到 `3e-4`（但需谨慎监控）
  - 不建议提高 Projector 学习率（容易导致训练不稳定）

### 其他重要提示

1. **RGB 图像尺寸**：自动设置为 336x336（LLaVA v1.5 要求）
2. **VAE 归一化**：自动转换为 `[-1, 1]` 范围
3. **Stage 1 → Stage 2**：必须提供 `--pretrain_polar_projector` 参数
4. **数据路径**：Stage 1 使用 crop 数据，Stage 2 使用非裁剪数据
5. **序列长度**：RGB+Polar 拼接后 1152 tokens，13B 模型 context window 4096，完全足够

### 验证 Stage 1 权重文件

如果 Stage 1 生成了 `non_lora_trainables.bin`，可以用以下命令验证是否包含 `polar_projector` 权重：

```python
import torch

# 加载权重文件
weights = torch.load('/path/to/stage1/non_lora_trainables.bin', map_location='cpu')

# 查找包含 'polar_projector' 的键
polar_keys = [k for k in weights.keys() if 'polar_projector' in k]

if polar_keys:
    print(f"✓ 找到 {len(polar_keys)} 个 polar_projector 相关的权重键")
    print(f"  示例键名: {polar_keys[:3]}")
else:
    print("✗ 未找到 polar_projector 权重！")
    print(f"  文件中的键示例: {list(weights.keys())[:5]}")
```

如果输出显示找到了 `polar_projector` 相关的键，说明权重文件正确，可以直接使用。

## 🔍 训练后验证

### ⚠️ 重要说明：验证时应该加载哪个模型？

**Stage 2 训练后的输出**：
- `adapter_model.safetensors`：LoRA 适配器权重（LLM 的 LoRA 参数）
- `non_lora_trainables.bin`：非 LoRA 可训练参数（包含**更新后的** `polar_projector` 权重）
- `adapter_config.json`：LoRA 配置

**验证方式选择**：

#### 方式 1：直接使用 Stage 2 checkpoint（推荐，最简单）

**优点**：
- ✅ 无需合并，直接使用训练输出
- ✅ `inference.py` 会自动加载 LoRA 适配器 + `non_lora_trainables.bin`
- ✅ 包含 Stage 2 训练后的所有权重（LoRA + 更新后的 polar_projector）

**命令**：
```bash
python LLaVA/inference.py \
    --checkpoint_dir /openbayes/input/input0/1.25/stage2 \
    --llm_model_name /openbayes/input/input0/models/llava-v1.5-13b \
    --clip_model_name /openbayes/input/input0/models/clip-vit-large-patch14-336 \
    --vae_model_path /openbayes/input/input0/models/sd-vae-ft-mse \
    --rgb_path /path/to/test/rgb/image.png \
    --polar_paths /path/to/I_0.png /path/to/I_45.png /path/to/I_90.png /path/to/I_135.png \
    --question "Describe this image in detail."
```

#### 方式 2：使用合并后的模型（可选，用于部署）

**优点**：
- ✅ 模型文件更简洁（LoRA 已合并到 base model）
- ✅ 部署时不需要分别加载 base + LoRA

**步骤**：
```bash
# 1. 合并 LoRA 权重（将 LoRA 合并到 base model）
python LLaVA/scripts/merge_lora_weights.py \
    --model-path /openbayes/input/input0/1.25/stage2 \
    --model-base /openbayes/input/input0/models/llava-v1.5-13b \
    --save-model-path /openbayes/input/input0/1.25/stage2_merged \
    --polar-vae-model-path /openbayes/input/input0/models/sd-vae-ft-mse

# 2. 使用合并后的模型验证
python LLaVA/inference.py \
    --checkpoint_dir /openbayes/input/input0/1.25/stage2_merged \
    --llm_model_name /openbayes/input/input0/1.25/stage2_merged \
    --clip_model_name /openbayes/input/input0/models/clip-vit-large-patch14-336 \
    --vae_model_path /openbayes/input/input0/models/sd-vae-ft-mse \
    --rgb_path /path/to/test/rgb/image.png \
    --polar_paths /path/to/I_0.png /path/to/I_45.png /path/to/I_90.png /path/to/I_135.png \
    --question "Describe this image in detail."
```

**注意**：
- 合并后，`inference.py` 会自动从 `polar_projector.bin` 加载 polar_projector 权重（`merge_lora_weights.py` 会生成此文件）
- 如果合并后的目录中没有 `polar_projector.bin`，会尝试从 `non_lora_trainables.bin` 加载

### 方法 1：使用推理脚本验证（推荐）

**使用 Stage 2 checkpoint（未合并）**：
```bash
python LLaVA/inference.py \
    --checkpoint_dir /openbayes/input/input0/1.25/stage2 \
    --llm_model_name /openbayes/input/input0/models/llava-v1.5-13b \
    --clip_model_name /openbayes/input/input0/models/clip-vit-large-patch14-336 \
    --vae_model_path /openbayes/input/input0/models/sd-vae-ft-mse \
    --rgb_path /openbayes/input/input0/rgb/30/0020_rgb.png \
    --polar_paths \
        /openbayes/input/input0/polar/30/0020_000.png \
        /openbayes/input/input0/polar/30/0020_045.png \
        /openbayes/input/input0/polar/30/0020_090.png \
        /openbayes/input/input0/polar/30/0020_135.png \
    --question "Focus on region [0.2, 0.2, 0.6, 0.6]. Please describe the visual content within this region."
```

**使用合并后的模型**：
```bash
# 先合并（如果还没合并）
python LLaVA/scripts/merge_lora_weights.py \
    --model-path /openbayes/input/input0/1.25/stage2 \
    --model-base /openbayes/input/input0/models/llava-v1.5-13b \
    --save-model-path /openbayes/input/input0/1.25/stage2_merged \
    --polar-vae-model-path /openbayes/input/input0/models/sd-vae-ft-mse

# 然后验证
python LLaVA/inference.py \
    --checkpoint_dir /openbayes/input/input0/1.25/stage2_merged \
    --llm_model_name /openbayes/input/input0/1.25/stage2_merged \
    --clip_model_name /openbayes/input/input0/models/clip-vit-large-patch14-336 \
    --vae_model_path /openbayes/input/input0/models/sd-vae-ft-mse \
    --rgb_path /openbayes/input/input0/rgb/30/0020_rgb.png \
    --polar_paths \
        /openbayes/input/input0/polar/30/0020_000.png \
        /openbayes/input/input0/polar/30/0020_045.png \
        /openbayes/input/input0/polar/30/0020_090.png \
        /openbayes/input/input0/polar/30/0020_135.png \
    --question "Focus on region [0.2, 0.2, 0.6, 0.6]. Please describe the visual content within this region."
```

### 方法 2：使用 Streamlit 可视化界面验证（推荐用于交互式验证）

使用 `inference_app_streamlit.py` 进行可视化验证：

```bash
streamlit run LLaVA/inference_app_streamlit.py
```

**功能**：
- ✅ 交互式上传图片并画框选择区域
- ✅ 支持多种问题模板（content, detail, behind, layer 等）
- ✅ 自动对比 RGB-only 基线（无偏振信号）
- ✅ 支持自动翻译为中文
- ✅ 实时显示推理结果

**界面配置**：
- 在侧边栏设置模型路径（默认已配置好）
- 上传 RGB + 4 张偏振图像
- 在 RGB 图像上画框选择区域
- 选择问题类型或自定义问题
- 点击"运行验证"查看结果

### 方法 3：批量验证（使用训练数据 JSON）

使用训练数据 JSON 进行批量验证：

```bash
python LLaVA/inference.py \
    --checkpoint_dir /openbayes/input/input0/1.25/stage2 \
    --llm_model_name /openbayes/input/input0/models/llava-v1.5-13b \
    --clip_model_name /openbayes/input/input0/models/clip-vit-large-patch14-336 \
    --vae_model_path /openbayes/input/input0/models/sd-vae-ft-mse \
    --dataset_json merged_stage3_qwen.json \
    --polar_root /openbayes/input/input0/polar \
    --data_root /openbayes/input/input0 \
    --qa_types content detail \
    --max_images_per_scene 10
```

**输出**：
- 结果保存到 `merged_stage3_qwen.results.json`
- 包含每个样本的问题、回答、bbox_norm 等信息

### 验证指标

**成功的验证应该显示**：
1. ✅ 模型能够加载（无错误）
2. ✅ 能够生成文本输出（不是空字符串）
3. ✅ 生成的文本与图像内容相关（至少部分相关）
4. ✅ 对于训练集中的图像，输出应该更准确
5. ✅ Polar Projector 权重已成功加载（检查日志中的 "✓ Polar Projector 权重已加载"）

**如果验证失败**：
- 检查模型路径是否正确
- 检查 VAE 模型路径是否正确
- 检查图像路径和格式是否正确
- 检查 `non_lora_trainables.bin` 或 `polar_projector.bin` 是否存在
- 查看日志中的错误信息

### 📋 验证时加载的权重总结

**Stage 2 checkpoint（未合并）包含**：
- ✅ `adapter_model.safetensors`：LoRA 权重（LLM 的 LoRA 参数）
- ✅ `non_lora_trainables.bin`：**更新后的** `polar_projector` 权重（Stage 2 训练后）
- ✅ `adapter_config.json`：LoRA 配置

**合并后的模型包含**：
- ✅ 合并后的 LLM 权重（base + LoRA）
- ✅ `polar_projector.bin`：**更新后的** `polar_projector` 权重（从 Stage 2 checkpoint 复制）

**关键点**：
- ✅ **验证 Stage 2 时，必须加载 Stage 2 的 `polar_projector` 权重**（不是 Stage 1 的）
- ✅ Stage 2 训练时，`polar_projector` 会继续更新，所以应该使用 Stage 2 输出的权重
- ✅ `inference.py` 会自动从 `non_lora_trainables.bin` 或 `polar_projector.bin` 加载 `polar_projector` 权重
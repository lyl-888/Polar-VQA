# PolarVLM Stage 3 训练指南

## 概述

Stage 3 是 PolarVLM 的**视觉指令微调（Visual Instruction Tuning）**阶段，使用高质量的 VQA（视觉问答）数据训练模型，使其能够根据 RGB 和偏振图像回答用户问题。

### 训练目标
- 在 Stage 2 的语义对齐基础上，进一步微调模型，使其能够理解并回答关于图像内容的自然语言问题
- 利用 RGB 和偏振双重信息，提高模型在反光场景下的视觉理解能力
- 通过问答对训练，增强模型的视觉-语言交互能力

---

## 数据格式

### JSON 文件结构

训练数据使用 `merged_stage3_data.json` 格式，每个样本包含：

```json
{
  "image": "GT/04/0000_rgb.png",           // GT 图像路径（相对路径）
  "input_path": "rgb/04/0002_rgb.png",     // RGB 输入图像路径（相对路径）
  "gt_path": "GT/04/0000_rgb.png",         // GT 图像路径（与 image 相同）
  "conversations": [
    {"from": "human", "value": "请简要描述框选区域内的视觉内容。"},
    {"from": "gpt", "value": "这一区域展示了中国山水画的中景部分..."}
  ],
  "scene_id": "04",                         // 场景 ID
  "bbox": [125, 329, 186, 24],             // 边界框（像素坐标，可选）
  "bbox_norm": [0.1, 0.2, 0.3, 0.4],       // 归一化边界框（0-1）
  "type": "typeB_visual",
  "subtype": "content"                      // "content", "detail", 或 "spatial"
}
```

### 关键字段说明

- **`input_path`**: RGB 输入图像路径（优先使用，指向 `rgb/` 目录下的非裁剪图像）
- **`image`**: GT 图像路径（指向 `GT/` 目录）
- **`conversations`**: 问答对列表，包含 `human` 和 `gpt` 两个角色的对话
- **`scene_id`**: 场景 ID，用于推导偏振图像路径
- **`bbox_norm`**: 归一化边界框坐标（0-1），用于定位感兴趣区域

### 图像路径结构

```
data_root/
├── rgb/              # RGB 输入图像（非裁剪，Stage 3 格式）
│   ├── 04/
│   │   ├── 0002_rgb.png
│   │   └── ...
│   └── ...
├── polar/            # 偏振图像（4 角度：000, 045, 090, 135）
│   ├── 04/
│   │   ├── 0002_000.png
│   │   ├── 0002_045.png
│   │   ├── 0002_090.png
│   │   ├── 0002_135.png
│   │   └── ...
│   └── ...
└── GT/               # GT 图像（清晰参考图像）
    ├── 04/
    │   ├── 0000_rgb.png
    │   └── ...
    └── ...
```

---

## 核心要点

### 1. 模型权重加载

Stage 3 训练需要加载 Stage 1 和 Stage 2 的预训练权重：

- **Stage 1 检查点**: 提供偏振编码器的 MAE 预训练权重
- **Stage 2 检查点**: 提供投影器（Projector）的语义对齐权重，包含 `rgb_scale` 和 `polar_scale` 动态平衡参数

### 2. 数据增强约束（关键）

⚠️ **绝对禁止使用改变几何形状的数据增强**（如 `RandomResizedCrop`、`RandomCrop`、`Rotation`）。

**原因**: Stage 3 数据包含归一化坐标（`bbox_norm`），如果图像被裁剪或旋转，坐标与图像内容将不匹配，导致模型无法学习正确的视觉定位。

**允许的增强**:
- RGB 图像：仅使用 `ColorJitter`（颜色抖动）+ `Resize`
- 偏振图像：仅使用 `Resize`（保持物理特性一致）

### 3. 图像处理流程

#### RGB 图像
- 使用 `CLIPImageProcessor` 处理（包含 resize、归一化等）
- 输入尺寸：224x224（CLIP ViT-L/14 标准尺寸）

#### 偏振图像
- 从 4 个角度（I_0, I_45, I_90, I_135）计算物理参数
- 输出 4 通道：`[Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]`
- **关键**: 不使用 ImageNet 归一化，值范围保持在 `[0, 1]`（与 Stage 2 保持一致）
- 输入尺寸：224x224（ViT 标准尺寸）

### 4. Prompt 格式

使用 **LLaMA-3 Chat 格式**:

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

<image>
{Question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{Answer}<|eot_id|>
```

### 5. 损失计算

- 使用 **Causal Language Modeling Loss**
- 仅对答案部分计算损失，问题部分（包括 `<image>` token）的标签设为 `-100`（忽略）

### 6. 训练策略

- **RGB Tower**: 冻结（已预训练，不参与训练）
- **Polar Tower**: 可训练（默认不冻结，可根据数据量调整）
- **Projector**: 可训练（必须参与训练，Stage 3 的核心）
- **LLM**: 通过 LoRA 微调（参数高效，只训练 adapter 参数）

---

## 完整训练命令

### 基础命令

```bash
python train.py \
  --train_json merged_stage3_data.json \
  --val_json val_stage3_data.json \
  --rgb_root /openbayes/input/input0/rgb \
  --polar_root /openbayes/input/input0/polar \
  --data_root /openbayes/input/input0 \
  --llm_model_name /openbayes/input/input0/models/Meta-Llama-3-8B-Instruct \
  --clip_model_name /openbayes/input/input0/models/clip-vit-large-patch14 \
  --polar_backbone /openbayes/input/input0/models/vit-base-patch16-224-in21k \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --load_in_4bit \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --freeze_rgb_tower \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 10 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.1 \
  --weight_decay 0.05 \
  --logging_steps 10 \
  --save_steps 230 \
  --save_strategy steps \
  --eval_steps 230 \
  --eval_strategy steps \
  --save_total_limit 3 \
  --load_best_model_at_end \
  --output_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --dataloader_num_workers 4
```

### 参数说明

#### 数据路径
- `--train_json`: 训练集 JSON 文件路径（默认: `merged_stage3_data.json`）
- `--val_json`: 验证集 JSON 文件路径（可选）
- `--rgb_root`: RGB 图像根目录（默认: `/openbayes/input/input0/rgb`）
- `--polar_root`: 偏振图像根目录（默认: `/openbayes/input/input0/polar`）
- `--data_root`: 数据根目录（用于解析 crop 路径，默认使用 `rgb_root` 的父目录）

#### 模型配置
- `--llm_model_name`: LLaMA-3 模型路径（默认: `/openbayes/input/input0/models/Meta-Llama-3-8B-Instruct`）
- `--clip_model_name`: CLIP 模型路径（默认: `openai/clip-vit-large-patch14`）
- `--polar_backbone`: 偏振流 backbone（默认: `google/vit-base-patch16-224-in21k`）
- `--stage1_checkpoint`: Stage 1 检查点路径（可选，用于加载 MAE 编码器权重）
- `--stage2_checkpoint`: Stage 2 检查点路径（可选，用于加载投影器权重）
- `--load_in_4bit`: 使用 4-bit 量化（推荐，节省显存）

#### LoRA 配置
- `--lora_r`: LoRA rank（默认: 64）
- `--lora_alpha`: LoRA alpha（默认: 128）
- `--lora_dropout`: LoRA dropout（默认: 0.05）

#### 训练配置
- `--per_device_train_batch_size`: 每设备批次大小（默认: 8）
- `--gradient_accumulation_steps`: 梯度累积步数（默认: 2，有效批次 = 8 * 2 = 16）
- `--num_train_epochs`: 训练轮数（默认: 10）
- `--learning_rate`: 学习率（默认: 1e-4）
- `--warmup_ratio`: 预热比例（默认: 0.1）
- `--weight_decay`: 权重衰减（默认: 0.05）

#### 保存/验证配置
- `--save_steps`: 保存步数（默认: 230，约每半个 epoch）
- `--save_strategy`: 保存策略（`steps` 或 `epoch`）
- `--eval_steps`: 验证步数（默认: 230）
- `--eval_strategy`: 验证策略（`steps`、`epoch` 或 `no`）
- `--save_total_limit`: 保存 checkpoint 数量限制（默认: 3）
- `--load_best_model_at_end`: 训练结束后加载最佳模型（需要验证集）

#### 冻结策略
- `--freeze_rgb_tower`: 冻结 RGB Tower（默认: True）
- `--freeze_polar_tower`: 冻结偏振 Tower（默认: False）
- `--freeze_polar_layers`: 冻结偏振流的前 N 层（默认: 0）

---

## 输出文件

训练完成后，检查点目录包含以下文件：

```
checkpoints/polarvlm_stage3/
├── adapter_config.json          # LoRA 配置
├── adapter_model.safetensors    # LoRA adapter 权重
├── projector.pth                # 多模态投影器权重
├── polar_tower.pth              # 偏振流权重（如果未冻结）
├── model_config.json            # 模型配置信息
├── tokenizer_config.json        # Tokenizer 配置
├── tokenizer.json               # Tokenizer 文件
└── special_tokens_map.json      # 特殊 token 映射
```

---

## 重要注意事项

### 1. 数据增强约束

⚠️ **绝对不能使用 `RandomResizedCrop`**，这会导致 bbox 坐标失效。只允许：
- `Resize`（保持宽高比）
- `ColorJitter`（仅 RGB，不改变几何形状）

### 2. 偏振图像归一化

⚠️ **偏振图像不使用 ImageNet 归一化**，值范围保持在 `[0, 1]`，与 Stage 2 保持一致。

### 3. 图像尺寸

- RGB 图像：224x224（CLIP ViT-L/14 标准尺寸）
- 偏振图像：224x224（ViT 标准尺寸）

### 4. 显存要求

- **推荐配置**: A5880 48GB（或类似显存）
- **批次大小**: Batch Size = 8，Gradient Accumulation = 2（有效批次 = 16）
- 如果显存不足，可以降低 `--per_device_train_batch_size` 或增加 `--gradient_accumulation_steps`

### 5. 训练监控

- 观察训练损失和验证损失，如果验证损失持续上升（过拟合），考虑：
  - 减少训练轮数（`--num_train_epochs`）
  - 增加权重衰减（`--weight_decay`）
  - 冻结更多层（`--freeze_polar_layers`）

### 6. 最佳模型选择

如果启用了 `--load_best_model_at_end`，训练结束后会自动加载验证损失最低的模型。建议在训练早期（如 epoch 0.5 附近）就停止训练，避免过拟合。

---

## 常见问题

### Q: 训练时出现 "bbox 坐标与图像不匹配" 的错误？

A: 检查数据增强配置，确保没有使用 `RandomResizedCrop` 等改变几何形状的增强。只允许 `Resize` 和 `ColorJitter`。

### Q: 偏振图像处理失败？

A: 检查偏振图像路径，确保 4 个角度（000, 045, 090, 135）的文件都存在。路径格式应为：`{polar_root}/{scene_id}/{base_name}_000.png`。

### Q: 显存不足（OOM）？

A: 尝试：
1. 降低 `--per_device_train_batch_size`（如改为 4）
2. 增加 `--gradient_accumulation_steps`（保持有效批次大小不变）
3. 确保使用 `--load_in_4bit`（4-bit 量化）

### Q: 验证损失上升（过拟合）？

A: 尝试：
1. 减少训练轮数
2. 增加权重衰减（`--weight_decay`）
3. 冻结更多偏振层（`--freeze_polar_layers`）
4. 使用更多数据增强（但注意不能改变几何形状）

---

## 参考资料

- Stage 1 训练指南: 参见 `STAGE1_STAGE2_README.md`
- Stage 2 训练指南: 参见 `STAGE1_STAGE2_README.md`
- 模型架构: 参见 `model.py`
- 数据集实现: 参见 `dataset.py`


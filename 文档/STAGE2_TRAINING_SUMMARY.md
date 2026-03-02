# PolarVLM Stage 2 训练总结

## 概述

Stage 2 是**语义对齐（Projector 预训练）**阶段，目标是将 Stage 1 预训练的偏振编码器特征与 LLM 的文本空间对齐。

---

## 核心目标

1. **训练 Projector 模块**：对齐偏振特征和 LLM 的文本表示空间
2. **学习动态平衡参数**：自动学习 `rgb_scale` 和 `polar_scale`，平衡 RGB 和偏振特征的重要性
3. **冻结所有其他模块**：只训练 Projector（RGB Tower、Polar Tower、LLM 全部冻结）

---

## 关键处理

### 1. 数据格式

- **数据源**：使用 `subtype == "content"` 的图像描述数据
- **输入格式**：
  - RGB 图像（裁剪后的 224×224）
  - 偏振图像（4 角度 → 转换为 4 通道物理参数：`[Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]`）
  - 文本对话（Question + Answer）
- **输出格式**：`input_ids`, `attention_mask`, `labels`

### 2. 空间对齐（关键修复）

⚠️ **核心约束**：RGB 和 Polar 图像必须保持严格的空间对齐

- ❌ **禁用** `RandomResizedCrop`：避免 RGB 和 Polar 图像空间错位
- ✅ **使用** `Resize` + `ColorJitter`（仅 RGB）：保持空间对齐，RGB 可以做颜色增强
- **原因**：Projector 需要学习精确的空间对应关系，任何空间错位都会影响对齐效果

### 3. 特征处理

- **RGB 特征**：使用 CLIP 处理器（`CLIPImageProcessor`），保持与 Stage 3 一致
- **Polar 特征**：
  - 使用 `process_polar_images` 计算 Stokes 参数
  - 转换为 4 通道物理参数：`[Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]`
  - ⚠️ **不使用 ImageNet 归一化**：与 Stage 1 训练保持一致（值范围 [0, 1]）
  - 输入到 Stage 1 预训练的编码器中提取特征

### 4. 损失计算

- **只对 Answer 部分计算损失**：Question 部分的 `labels` 设为 `-100`（忽略）
- **原因**：Stage 2 的目标是学习视觉特征到文本的对齐，只需要学习 Answer（caption）部分
- **格式**：`[BOS] + Question + "\n" + Answer`
  - Question 部分（包括 BOS 和 "\n"）的 labels 设为 -100
  - Answer 部分的 labels 与 input_ids 相同

### 5. 动态平衡参数

Projector 中包含两个可学习的缩放参数：
- `rgb_scale`：RGB 特征的缩放权重（初始值：1.0）
- `polar_scale`：偏振特征的缩放权重（初始值：0.1）

这两个参数在训练过程中自动学习，用于平衡 RGB 和偏振特征的重要性。

---

## 训练结果

### 动态平衡参数变化

- **RGB Scale**: `1.0` → `1.100`（微涨 10%）
  - 说明 RGB 特征本身质量很高，模型稍微放大了它的权重

- **Polar Scale**: `0.1` → `0.030`（显著下降 70%）
  - 说明偏振特征的强度相对较弱，模型降低了它的权重
  - 这可能是因为：
    1. 偏振特征分布与 RGB 特征分布存在差异
    2. 模型需要降低偏振特征的权重来平衡两者

### 训练损失

- **训练过程**：Loss 从初始较高值逐渐下降到 `1.2-1.3` 左右
- **最终训练损失**：`1.55`（在整个训练集上的平均损失）
- **梯度范数**：稳定在 `0.39-0.45` 之间，说明训练稳定

### 训练配置

- **训练轮数**：5 epochs
- **有效批次大小**：32（per_device_batch_size=16 × gradient_accumulation_steps=2）
- **学习率**：5e-4（使用 cosine 调度）
- **优化器**：AdamW
- **精度**：BF16（4-bit 量化的 LLM）

---

## 完整训练指令

```bash
python train_stage2.py \
    --train_json stage2_gt_captions_all.json \
    --rgb_root /openbayes/input/input0/rgb \
    --polar_root /openbayes/input/input0/polar \
    --data_root /openbayes/input/input0 \
    --clip_model_name /openbayes/input/input0/models/clip-vit-large-patch14 \
    --polar_backbone /openbayes/input/input0/models/vit-base-patch16-224-in21k \
    --llm_model_name /openbayes/input/input0/models/Meta-Llama-3-8B-Instruct \
    --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
    --load_in_4bit \
    --output_dir /openbayes/input/input0/checkpoints/stage2_projector \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 5 \
    --learning_rate 5e-4 \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --logging_steps 10 \
    --save_steps 200 \
    --save_total_limit 2 \
    --dataloader_num_workers 8 \
    --bf16 \
    --freeze_rgb_tower \
    --freeze_polar_tower \
    --freeze_llm
```

### 参数说明

- `--train_json`：Stage 2 训练数据 JSON 文件（包含 `conversations` 字段）
- `--data_root`：数据根目录（用于解析 `rgb_crop/` 和 `polar_crop/` 路径）
- `--stage1_checkpoint`：Stage 1 检查点路径（用于加载偏振编码器权重）
- `--load_in_4bit`：使用 4-bit 量化加载 LLM（节省显存）
- `--per_device_train_batch_size`：每设备批次大小（根据显存调整，A5880 48GB 可设置为 16）
- `--gradient_accumulation_steps`：梯度累积步数（有效批次大小 = batch_size × gradient_accumulation_steps）
- `--num_train_epochs`：训练轮数（建议 5 轮）
- `--learning_rate`：学习率（建议 5e-4，较保守，提高稳定性）
- `--weight_decay`：权重衰减（设置为 0.0，Projector 参数少，不需要强正则化）

---

## 输出文件

训练完成后，会生成以下文件：

- `{output_dir}/projector.pth`：Projector 权重（用于 Stage 3）
- `{output_dir}/model.safetensors`：完整模型权重（可选）
- `{output_dir}/logs/`：TensorBoard 日志文件

---

## 注意事项

1. **空间对齐**：确保 RGB 和 Polar 图像使用相同的变换（只使用 `Resize`，禁用 `RandomResizedCrop`）
2. **特征归一化**：Polar 特征不使用 ImageNet 归一化（与 Stage 1 保持一致）
3. **损失计算**：只对 Answer 部分计算损失，Question 部分设为 -100
4. **显存要求**：使用 4-bit 量化时，A5880 48GB 可以运行 batch_size=16
5. **动态平衡参数**：`rgb_scale` 和 `polar_scale` 会在训练日志中显示，用于监控训练过程

---

## 下一步

完成 Stage 2 训练后，可以：
1. 检查 `projector.pth` 是否成功保存
2. 验证 `rgb_scale` 和 `polar_scale` 的最终值（应该在训练日志中）
3. 进入 Stage 3 训练（视觉指令微调）


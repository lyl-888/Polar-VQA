# Stage 1 训练命令优化分析

## 🔍 您的命令分析

### 原始命令问题

1. **批次大小 512**：对于 24GB 显存，这个批次大小**太大**，可能导致 OOM
2. **学习率 1e-3**：对于 MAE 预训练，这个学习率**太高**，可能导致训练不稳定
3. **训练轮数 2000**：对于 7000 条数据，2000 epochs **太多**，可能导致过拟合
4. **梯度累积 4**：有效批次大小 = 512 * 4 = 2048，**过大**
5. **bf16**：`train_stage1.py` 不支持 `bf16`，只支持 `fp16`
6. **缺少 gradient_accumulation_steps 参数**：代码中没有这个参数

---

## ✅ 优化后的训练命令

### 方案1：保守配置（推荐，确保不 OOM）

```bash
nohup python train_stage1.py \
    --polar_root /openbayes/home/data/polar_pt \
    --use_pt_data \
    --model_name /openbayes/input/input0/models/vit-mae-base \
    --image_size 224 \
    --patch_size 16 \
    --num_channels 4 \
    --norm_pix_loss False \
    --output_dir /openbayes/home/checkpoints/stage1_encoder_new2 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --dataloader_num_workers 8 \
    --num_train_epochs 800 \
    --learning_rate 1.5e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --logging_steps 50 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --load_best_model_at_end \
    --fp16 \
    --hf_token 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 \
    > train_stage1.log 2>&1 &
```

### 方案2：激进配置（如果方案1显存充足，可以尝试）

```bash
nohup python train_stage1.py \
    --polar_root /openbayes/home/data/polar_pt \
    --use_pt_data \
    --model_name /openbayes/input/input0/models/vit-mae-base \
    --image_size 224 \
    --patch_size 16 \
    --num_channels 4 \
    --norm_pix_loss False \
    --output_dir /openbayes/home/checkpoints/stage1_encoder_new2 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --dataloader_num_workers 8 \
    --num_train_epochs 800 \
    --learning_rate 1.5e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --logging_steps 50 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --load_best_model_at_end \
    --fp16 \
    --hf_token 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 \
    > train_stage1.log 2>&1 &
```

---

## 📊 参数对比表

| 参数 | 您的命令 | 方案1（保守） | 方案2（激进） | 说明 |
|------|---------|-------------|-------------|------|
| **批次大小** | 512 | 128 | 256 | 512 太大，可能导致 OOM |
| **梯度累积** | 4 | - | - | 代码不支持，已移除 |
| **有效批次** | 2048 | 128 | 256 | 2048 对于 7000 条数据太大 |
| **学习率** | 1e-3 | 1.5e-4 | 1.5e-4 | 1e-3 太高，可能导致训练不稳定 |
| **训练轮数** | 2000 | 800 | 800 | 2000 太多，800 足够 |
| **精度** | bf16 | fp16 | fp16 | 代码只支持 fp16 |
| **评估策略** | steps | steps | steps | ✅ 正确 |
| **评估步数** | 500 | 500 | 500 | ✅ 合理 |

---

## 🎯 关键修改说明

### 1. 批次大小：512 → 128/256

**原因**：
- MAE 模型本身较大（~112M 参数）
- 4 通道输入（224x224x4）
- 24GB 显存限制
- 批次大小 512 很可能导致 OOM

**建议**：
- 先尝试 128，如果显存充足再增加到 256
- 如果 128 仍然 OOM，可以降低到 64

### 2. 学习率：1e-3 → 1.5e-4

**原因**：
- MAE 预训练通常使用 1.5e-4 左右的学习率
- 1e-3 太高，可能导致：
  - 训练不稳定
  - Loss 震荡
  - 难以收敛

**参考**：
- MAE 论文：1.5e-4
- 您的调试训练：1e-3（但那是单样本过拟合，可以容忍高学习率）

### 3. 训练轮数：2000 → 800

**原因**：
- 7000 条数据，800 epochs 已经足够
- 2000 epochs 可能导致：
  - 过拟合
  - 训练时间过长
  - 资源浪费

**计算**：
- 7000 条数据，批次 128，每 epoch ≈ 55 步
- 800 epochs ≈ 44,000 步（足够收敛）

### 4. 移除梯度累积

**原因**：
- `train_stage1.py` 代码中没有 `gradient_accumulation_steps` 参数
- 如果需要更大的有效批次，直接增加 `per_device_train_batch_size`

### 5. bf16 → fp16

**原因**：
- `train_stage1.py` 只支持 `fp16`，不支持 `bf16`
- 如果需要 bf16，需要修改代码

---

## 💡 显存估算

### 批次大小 128（方案1）

```
模型参数：~112M * 4 bytes (fp32) ≈ 448 MB
梯度：~112M * 4 bytes ≈ 448 MB
优化器状态：~112M * 8 bytes (AdamW) ≈ 896 MB
输入数据：128 * 4 * 224 * 224 * 4 bytes ≈ 103 MB
激活值：~2-4 GB（取决于模型深度）
总计：~6-7 GB（fp16 模式下更少）
```

**结论**：24GB 显存完全足够，还有很大余量

### 批次大小 256（方案2）

```
输入数据：256 * 4 * 224 * 224 * 4 bytes ≈ 206 MB
激活值：~4-6 GB
总计：~8-10 GB
```

**结论**：24GB 显存仍然足够

### 批次大小 512（您的原始命令）

```
输入数据：512 * 4 * 224 * 224 * 4 bytes ≈ 412 MB
激活值：~8-12 GB
总计：~12-15 GB
```

**结论**：可能接近显存上限，有 OOM 风险

---

## 🚀 推荐执行步骤

### 步骤1：先用方案1（保守）测试

```bash
# 运行方案1，观察显存使用情况
nvidia-smi  # 监控显存
```

### 步骤2：如果显存充足，可以尝试方案2

```bash
# 如果方案1显存使用 < 15GB，可以尝试方案2
```

### 步骤3：如果方案1仍然 OOM

```bash
# 降低批次大小到 64
--per_device_train_batch_size 64
```

---

## 📝 训练监控建议

1. **监控显存使用**：
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **监控训练日志**：
   ```bash
   tail -f train_stage1.log
   ```

3. **关键指标**：
   - Training Loss 应该持续下降
   - Validation Loss 应该稳定（不过度过拟合）
   - 显存使用应该稳定（不持续增长）

---

## ✅ 最终推荐命令

**推荐使用方案1（保守配置）**：

```bash
nohup python train_stage1.py \
    --polar_root /openbayes/home/data/polar_pt \
    --use_pt_data \
    --model_name /openbayes/input/input0/models/vit-mae-base \
    --image_size 224 \
    --patch_size 16 \
    --num_channels 4 \
    --norm_pix_loss False \
    --output_dir /openbayes/home/checkpoints/stage1_encoder_new2 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --dataloader_num_workers 8 \
    --num_train_epochs 800 \
    --learning_rate 1.5e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --logging_steps 50 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --load_best_model_at_end \
    --fp16 \
    --hf_token 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 \
    > train_stage1.log 2>&1 &
```

**关键优化**：
- ✅ 批次大小：512 → 128（避免 OOM）
- ✅ 学习率：1e-3 → 1.5e-4（MAE 标准学习率）
- ✅ 训练轮数：2000 → 800（足够收敛）
- ✅ 移除梯度累积（代码不支持）
- ✅ bf16 → fp16（代码只支持 fp16）

---

## 🎯 预期训练时间

**估算**（基于 7000 条数据，批次 128）：
- 每 epoch：~55 步
- 800 epochs：~44,000 步
- 每步时间：~0.5-1 秒（取决于 GPU）
- **总时间**：约 6-12 小时

---

## 💡 总结

**您的原始命令存在以下问题**：
1. ❌ 批次大小太大（512）→ 可能导致 OOM
2. ❌ 学习率太高（1e-3）→ 可能导致训练不稳定
3. ❌ 训练轮数太多（2000）→ 可能导致过拟合
4. ❌ 使用了不支持的参数（bf16, gradient_accumulation_steps）

**推荐使用方案1（保守配置）**，确保：
- ✅ 显存安全（批次 128）
- ✅ 训练稳定（学习率 1.5e-4）
- ✅ 充分训练（800 epochs）
- ✅ 所有参数都支持

祝训练顺利！🚀


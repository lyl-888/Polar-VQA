# verify_stage1.py 修复总结

## 修复内容

### 1. 统一 unpatchify 逻辑
- **之前**：使用 `permute(0, 5, 1, 3, 2, 4)` 进行维度重排
- **现在**：使用与 `verify_debug_stage1.py` 完全相同的 `einsum('nhwpqc->nchpwq', x)` 方式
- **原因**：单样本验证（`verify_debug_stage1.py`）比较成功，说明其 unpatchify 逻辑是正确的

### 2. 简化 logits 排列逻辑
- **之前**：尝试三种不同的 logits 排列方式（method1, method2, method3），选择与模型 Loss 最接近的方式
- **现在**：直接使用 `ids_restore` 重新排列到 Raster Order（与 `verify_debug_stage1.py` 一致）
- **原因**：简化逻辑，避免复杂的验证流程，直接使用已验证正确的方式

### 3. 移除相关性自动切换逻辑
- **之前**：如果相关性极低，自动尝试不同的 unpatchify 方式并切换
- **现在**：只计算相关性统计并给出警告，不自动切换
- **原因**：使用与 `verify_debug_stage1.py` 相同的 unpatchify 方式后，不应该再需要切换

## 训练脚本差异分析

### train_stage1.py vs debug_stage1.py

#### 1. 模型冻结策略
- **train_stage1.py**：
  - 使用层级冻结策略（Partial Fine-Tuning）
  - 冻结所有参数，只解冻：
    - Embeddings（学习4通道输入投影）
    - 最后3层 Encoder（layer 9-11）
    - Decoder（学习偏振数据重建）
  - 可训练参数占比：约 20-30%

- **debug_stage1.py**：
  - 也使用 `create_mae_model`，所以也会应用相同的冻结策略
  - 但单样本训练成功，说明冻结策略不是问题

#### 2. 训练参数
- **train_stage1.py**：
  - `learning_rate`: 1e-4（默认）
  - `num_train_epochs`: 100（默认）
  - `warmup_ratio`: 0.1
  - `weight_decay`: 0.05
  - `fp16`: True

- **debug_stage1.py**：
  - `learning_rate`: 1e-3（更高，便于快速过拟合）
  - `num_train_epochs`: 100
  - `warmup_ratio`: 0.0（不需要预热）
  - `weight_decay`: 0.0（关闭正则化）
  - `fp16`: False（调试模式建议关闭）

#### 3. 数据增强
- **train_stage1.py**：
  - 训练集：`is_train=True`（应用数据增强）
  - 验证集：`is_train=False`（不应用数据增强）

- **debug_stage1.py**：
  - `is_train=False`（关闭数据增强，使用固定样本）

#### 4. 数据集大小
- **train_stage1.py**：
  - 使用完整数据集（6312 张图）

- **debug_stage1.py**：
  - 使用单个样本，重复 batch_size 次

## 可能的问题

### 1. 训练不充分
- Masked PSNR 只有 12.88 dB，说明重建质量较差
- 可能原因：
  - 训练轮数不足（100 epochs 可能不够）
  - 学习率不合适
  - 数据增强可能影响训练

### 2. 数据预处理不一致
- 训练时和验证时的数据范围可能不同
- 建议检查：
  - 训练数据是否归一化到 [0, 1]
  - 验证数据是否使用相同的预处理

### 3. 层级冻结策略
- 虽然单样本训练成功，但完整数据集可能需要更多可训练参数
- 建议尝试：
  - 解冻更多 Encoder 层（例如最后6层）
  - 或者完全解冻 Encoder（只冻结底层）

## 建议

1. **验证修复后的代码**：
   ```bash
   python verify_stage1.py \
       --checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
       --polar_root /openbayes/home/data/polar_pt \
       --use_pt_data \
       --scene_id 17 \
       --base_name 0004
   ```

2. **如果相关性仍然很低**：
   - 检查训练数据预处理是否与验证一致
   - 检查模型是否训练充分（Loss 是否还在下降）
   - 考虑增加训练轮数或调整学习率

3. **如果 PSNR 仍然很低**：
   - 考虑解冻更多 Encoder 层
   - 检查数据质量（是否有异常值）
   - 考虑使用更大的学习率或更长的训练时间


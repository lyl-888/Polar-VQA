# Stage 1 验证脚本深度诊断与修复

## 📋 问题总结

### 核心矛盾
- **模型 Loss**: 0.0040（极低，说明数值预测很准）
- **Masked MSE**: 0.0503（是 Loss 的 12 倍）
- **DoLP 相关性**: -0.0672（接近0，负相关）
- **Masked PSNR**: 12.98 dB（较差）

### 结论
只要相关性是0或负数，无论 MSE 多少，都意味着重建的图像是一团乱码。这绝对不是模型没学好（如果是模型没学好，相关性应该是正的，只是比较低，比如 0.3）。这说明像素的位置或者通道的顺序依然是乱的。

## 🔍 深度排查方案

### 1. Unpatchify 逻辑自测（最高优先级）

**目的**: 验证 unpatchify 代码逻辑是否正确

**方法**: 将原图手动 patchify，再用 unpatchify 拼回去，检查是否完全一致

**实现位置**: `verify_stage1.py` 的 `visualize_mae_reconstruction` 函数开头

**检查标准**:
- ✅ 如果差异 < 1e-6：Unpatchify 逻辑正确，问题在模型输出或数据对齐
- ⚠️ 如果差异 < 1e-3：有小误差，可能是数值精度问题
- ❌ 如果差异 > 1e-3：Unpatchify 逻辑错误，需要修复 einsum 公式

**代码逻辑**:
```python
# 1. 手动模拟 patchify
input_reshaped = pixel_values.reshape(B, C, H_patches, P, W_patches, P)
patches_test = input_reshaped.permute(0, 2, 4, 3, 5, 1)  # (B, H_p, W_p, P, P, C)
patches_flatten = patches_test.reshape(B, H_patches * W_patches, P * P * C)

# 2. 使用 unpatchify 拼回去
x_test = patches_flatten.reshape(B, H_patches, W_patches, P, P, C)
x_test = torch.einsum('nhwpqc->nchpwq', x_test)
reconstructed_test = x_test.reshape(B, C, H, W)

# 3. 检查差异
diff = (pixel_values - reconstructed_test).abs()
```

### 2. 通道顺序检查

**怀疑点**: 如果 Dataset 加载时把通道顺序搞错了，模型虽然能强行学（Loss会降），但在验证时，你会拿真实的DoLP去和预测的sin做对比，导致极低的相关性。

**检查位置**: `verify_stage1.py` 的 `run_one_example` 函数

**检查内容**:
- ✅ `.pt` 文件保存时的通道顺序：`[Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]`
- ✅ `dataset_stage1.py` 加载时：`pixel_values[1:4, :, :]` → `[DoLP, sin, cos]`
- ✅ `verify_stage1.py` 加载时：`pixel_values_tensor[1:4, :, :]` → `[DoLP, sin, cos]`

**诊断输出**:
- 打印每个通道的统计信息（mean, std, min, max）
- 确认通道顺序一致性

### 3. ids_restore 处理检查

**怀疑点**: HuggingFace MAE 的 `ids_restore` 含义可能理解错误

**检查位置**: `verify_stage1.py` 的 `visualize_mae_reconstruction` 函数

**检查内容**:
- ✅ `ids_restore` 是否包含所有索引（0 到 num_total_patches-1）
- ✅ `ids_restore` 索引是否在有效范围内
- ✅ `ids_restore` 恢复逻辑是否正确（使用测试向量验证）

**恢复逻辑**:
```python
# HuggingFace MAE 的 ids_restore 含义：
# ids_restore[i] 表示 logits 中第 i 个元素应该放到原始位置的哪个位置
# 所以：restored_logits[ids_restore[i]] = logits[i]
restored_logits[0, ids_restore_tensor] = logits[0]
```

### 4. 数据增强检查

**怀疑点**: 如果训练时做了数据增强（随机翻转、随机裁剪），而验证时没有，会导致模型输出和 Ground Truth 对不上。

**检查位置**: `dataset_stage1.py` 的 `__getitem__` 方法

**检查内容**:
- ✅ `is_train=False` 时，验证模式是否关闭所有随机增强
- ✅ 验证脚本是否直接加载 `.pt` 文件（不经过 Dataset，无数据增强）

**确认**:
- ✅ `dataset_stage1.py` 在 `is_train=False` 时，验证模式直接 `pass`，不做任何处理
- ✅ `verify_stage1.py` 直接加载 `.pt` 文件，不经过 Dataset，无数据增强

### 5. 数据预处理一致性检查

**怀疑点**: 训练时的数据预处理和验证时完全不一致（比如训练时归一化了，验证时没归一化，或者反之）。

**检查位置**: `verify_stage1.py` 的 `run_one_example` 函数

**检查内容**:
- ✅ 输入数据范围：`[0, 1]`（与训练时一致）
- ✅ 输入数据均值、标准差（打印统计信息）
- ✅ 通道统计信息（每个通道的 mean, std, min, max）

## 🛠️ 修改总结

### 1. 添加 Unpatchify 逻辑自测代码
- **位置**: `verify_stage1.py` 第 500-550 行
- **功能**: 验证 unpatchify 代码逻辑是否正确
- **输出**: 打印差异统计，判断问题是否在 unpatchify 逻辑

### 2. 添加通道顺序检查
- **位置**: `verify_stage1.py` 第 1280-1300 行
- **功能**: 检查通道顺序一致性
- **输出**: 打印每个通道的统计信息，确认通道顺序

### 3. 添加 ids_restore 诊断
- **位置**: `verify_stage1.py` 第 600-620 行和第 580-600 行
- **功能**: 检查 ids_restore 的有效性和恢复逻辑
- **输出**: 打印 ids_restore 的统计信息和恢复逻辑测试结果

### 4. 添加数据预处理一致性检查
- **位置**: `verify_stage1.py` 第 1320-1340 行
- **功能**: 检查数据预处理一致性
- **输出**: 打印输入数据的统计信息，确认与训练时一致

## 📊 下一步行动

### 运行验证脚本后，检查以下输出：

1. **Unpatchify 逻辑自测结果**
   - ✅ 如果差异 < 1e-6：继续检查其他问题
   - ❌ 如果差异 > 1e-3：修复 unpatchify 逻辑（最高优先级）

2. **通道顺序检查结果**
   - ✅ 如果通道顺序一致：继续检查其他问题
   - ❌ 如果通道顺序不一致：修复通道顺序问题

3. **ids_restore 检查结果**
   - ✅ 如果 ids_restore 有效且恢复逻辑正确：继续检查其他问题
   - ❌ 如果 ids_restore 有问题：修复 ids_restore 处理逻辑

4. **相关性检查结果**
   - ✅ 如果相关性 > 0.8：问题已解决
   - ⚠️ 如果相关性 0.3-0.8：模型训练不充分，但逻辑正确
   - ❌ 如果相关性 < 0.3 或为负：继续排查其他问题

## 🎯 预期结果

修复后，应该看到：
- ✅ Unpatchify 逻辑自测：差异 < 1e-6
- ✅ 通道顺序：`[DoLP, sin(2*AoLP), cos(2*AoLP)]` 一致
- ✅ ids_restore：有效且恢复逻辑正确
- ✅ 相关性：> 0.8（而不是接近0或负数）
- ✅ Loss vs Masked MSE：差异在 1.0x - 1.2x 之间（正常范围）

## 📝 注意事项

1. **不要急着重新训练**：如果"自测代码"通过了，说明问题不在 unpatchify 逻辑，而在模型输出或数据对齐上。

2. **重点关注相关性**：相关性是最关键的指标，如果相关性为负或接近0，说明图像重组错了。

3. **逐步排查**：按照优先级逐个检查，不要同时修改多个地方。

4. **保存诊断输出**：运行验证脚本后，保存所有诊断输出，用于分析问题。



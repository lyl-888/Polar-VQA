# 三个方案分析

## 方案A：检查数据分布 ✅ **强烈推荐**

### 优点：
- ✅ **最科学**：先诊断问题，再决定解决方案
- ✅ **无副作用**：只是检查数据，不改变训练流程
- ✅ **快速**：几分钟就能知道问题所在

### 已创建脚本：
`check_data_distribution.py` - 可以检查数据分布

### 使用方法：
```bash
# 检查多个样本（推荐）
python check_data_distribution.py \
    --pt_root /openbayes/home/data/polar_pt \
    --num_samples 100

# 或检查单个文件（快速测试）
python check_data_distribution.py \
    --single_file /openbayes/home/data/polar_pt/train/17/0003.pt
```

### 预期结果：
- **如果 DoLP 均值 < 0.2**：说明数据值域小，Loss 低是正常的
- **如果 sin/cos 范围是 [-1, 1]**：需要映射到 [0, 1]
- **如果最大值远小于 1**：检查预处理是否有问题

---

## 方案B：调整 Loss 权重（Re-scaling）❌ **不推荐**

### 问题分析：
1. **破坏物理意义**：
   - 在 dataset 里乘以系数（如10）会破坏数据的物理意义
   - DoLP=0.1 乘以10变成1.0，模型学到的不是真实的物理值
   - 这违背了禁用 `norm_pix_loss` 的初衷（保留物理量的绝对意义）

2. **训练和推理不一致**：
   - 训练时数据乘以10，推理时数据不乘以10
   - 会导致模型在推理时表现异常

3. **治标不治本**：
   - 只是让 Loss 数值变大，但模型学到的特征可能不正确
   - 如果数据值域确实小，Loss 低是正常的，不需要强行放大

### 结论：
❌ **不推荐使用**，除非方案A检查后发现数据确实有问题

---

## 方案C：学习率过大 ❌ **不推荐**

### 问题分析：
1. **Loss 在下降**：
   - 从 0.0091 降到 0.0081，虽然慢但是**持续下降**
   - 如果学习率太大，Loss 应该会**上升或震荡**，而不是下降

2. **Grad Norm 偶尔飙升是正常的**：
   - Grad norm 偶尔到 0.19 是正常的波动
   - 只要没有持续飙升（>1.0），就不是问题
   - 你的 grad_norm 大部分在 0.01-0.08 之间，这是正常的

3. **验证 Loss 略高是正常的**：
   - 训练 Loss: 0.008-0.009
   - 验证 Loss: 0.012
   - 这是**轻微的过拟合**，是正常的（训练集和验证集有差异）

4. **当前学习率合理**：
   - 4e-4 对于部分微调（43% 参数）是合理的
   - 如果降低到 1e-4，Loss 下降会更慢

### 结论：
❌ **不推荐降低学习率**，当前学习率是合理的

---

## 🎯 最终建议

### 第一步：运行方案A（必须）
```bash
python check_data_distribution.py \
    --pt_root /openbayes/home/data/polar_pt \
    --num_samples 100
```

### 第二步：根据方案A的结果决定

#### 情况1：数据值域小（DoLP均值 < 0.2）
- ✅ **继续训练**，Loss 低是正常的
- ✅ **运行验证脚本**检查重建质量：
  ```bash
  python verify_stage1.py \
      --checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_3_bs256/checkpoint-250 \
      --polar_root /openbayes/home/data/polar_pt \
      --use_pt_data \
      --scene_id 17 \
      --base_name 0003 \
      --num_examples 5
  ```
- ✅ **如果 PSNR > 15 dB**：说明训练正常，Loss 低是合理的

#### 情况2：sin/cos 范围是 [-1, 1]
- ⚠️ **需要修复数据预处理**：确保 sin/cos 映射到 [0, 1]
- 修改 `dataset_common.py` 中的 `process_polar_images` 函数

#### 情况3：数据最大值远小于 1
- ⚠️ **检查预处理**：可能数据被过度压缩了
- 检查 `.pt` 文件的生成过程

### 第三步：如果重建质量差（PSNR < 10 dB）
- 考虑提高学习率（如 1e-3）
- 或解冻更多层

---

## 📊 总结

| 方案 | 推荐度 | 原因 |
|------|--------|------|
| **方案A** | ✅✅✅ **强烈推荐** | 科学诊断，无副作用 |
| **方案B** | ❌ **不推荐** | 破坏物理意义，治标不治本 |
| **方案C** | ❌ **不推荐** | Loss在下降，学习率合理 |

**建议行动顺序**：
1. ✅ 运行 `check_data_distribution.py`（方案A）
2. ✅ 根据结果运行 `verify_stage1.py` 检查重建质量
3. ✅ 如果重建质量好，继续训练（Loss低是正常的）
4. ❌ 不要使用方案B（破坏物理意义）
5. ❌ 不要降低学习率（Loss在下降，学习率合理）



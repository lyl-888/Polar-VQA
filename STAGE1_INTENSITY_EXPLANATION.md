# Stage 1 Intensity（总光强）说明

## 核心问题解答

### ❓ 问题1：为什么训练指令中只有 polar 图像的路径，没有 RGB 的路径？

**答案**：因为 **Intensity（总光强）不是 RGB，而是从偏振图像计算出来的物理量**。

### ❓ 问题2：Intensity 参与 Loss 计算了吗？

**答案**：**是的，Intensity 完全参与 Loss 计算**。MAE 的重建损失是对所有4个通道计算的。

---

## Intensity 的计算方式

### 物理原理

**Intensity（总光强）是从4张偏振角度图像计算出来的，不是从RGB图像来的**。

### 计算公式

```python
# 从4张偏振角度图像计算 Stokes 参数
S0 = (I_0 + I_90) / 2.0  # 总光强（Stokes 参数 S0）

# Intensity 归一化到 [0, 1]
Intensity = S0 / 255.0
```

### 数据流程

```
输入：4张偏振角度图像
├── I_0.png   (0度偏振角)
├── I_45.png  (45度偏振角)
├── I_90.png  (90度偏振角)
└── I_135.png (135度偏振角)
         ↓
    计算 Stokes 参数
         ↓
    S0 = (I_0 + I_90) / 2.0
    S1 = I_0 - I_90
    S2 = I_45 - I_135
         ↓
    计算4通道物理参数
         ↓
    ┌─────────────────────────────────────┐
    │ 通道0: Intensity = S0 / 255.0       │ ← 从偏振图像计算
    │ 通道1: DoLP = sqrt(S1²+S2²) / S0    │ ← 从偏振图像计算
    │ 通道2: sin(2*AoLP) = sin(2*atan2...)│ ← 从偏振图像计算
    │ 通道3: cos(2*AoLP) = cos(2*atan2...)│ ← 从偏振图像计算
    └─────────────────────────────────────┘
         ↓
    输入到 MAE 模型（4通道）
         ↓
    MAE 重建损失（对所有4个通道计算）
```

---

## 为什么 Intensity 不是 RGB？

### 1. **物理定义不同**

- **RGB**：红、绿、蓝三个颜色通道，表示可见光的颜色信息
- **Intensity（总光强）**：Stokes 参数 S0，表示**总的光强**（不考虑偏振方向）

### 2. **计算来源不同**

- **RGB**：直接从相机传感器读取（如果是RGB相机）
- **Intensity**：从偏振角度图像计算：`S0 = (I_0 + I_90) / 2.0`

### 3. **数值关系**

虽然 Intensity 在数值上可能接近 RGB 的亮度（灰度值），但它们是**不同的物理量**：
- Intensity 是偏振成像系统测量的总光强
- RGB 的亮度是颜色相机测量的光强

**注意**：如果您的数据集同时有 RGB 和偏振图像，它们可能来自不同的相机或不同的拍摄时间，数值上可能不完全一致。

---

## MAE Loss 计算

### Loss 计算方式

MAE 的重建损失是对**所有4个通道**计算的：

```python
# MAE 前向传播
outputs = model(pixel_values=pixel_values)  # pixel_values: (B, 4, H, W)

# Loss 计算（对所有4个通道）
loss = outputs.loss  # 这是对所有通道的 MSE 损失
```

### 通道参与情况

**所有4个通道都参与 Loss 计算**：

1. ✅ **Intensity（通道0）**：参与 Loss 计算
2. ✅ **DoLP（通道1）**：参与 Loss 计算
3. ✅ **sin(2*AoLP)（通道2）**：参与 Loss 计算
4. ✅ **cos(2*AoLP)（通道3）**：参与 Loss 计算

### Loss 计算公式

MAE 的重建损失通常是：

```
Loss = MSE(predicted_pixels, target_pixels)
     = mean((predicted - target)²)
```

其中 `predicted` 和 `target` 都是 `(B, 4, H, W)` 的张量，包含所有4个通道。

---

## 数据流程总结

### Stage 1 训练流程

```
1. 输入：偏振图像目录
   └── polar_root/
       ├── scene_00/
       │   ├── 0000_000.png  (I_0)
       │   ├── 0000_045.png  (I_45)
       │   ├── 0000_090.png  (I_90)
       │   └── 0000_135.png  (I_135)
       └── ...

2. 计算物理参数（process_polar_images）
   └── 从4张角度图像计算：
       - Intensity = (I_0 + I_90) / 2.0 / 255.0
       - DoLP = sqrt((I_0-I_90)² + (I_45-I_135)²) / S0
       - sin(2*AoLP) = sin(2 * atan2(S2, S1))
       - cos(2*AoLP) = cos(2 * atan2(S2, S1))

3. 输入到 MAE 模型
   └── (B, 4, H, W) 张量
       - 通道0: Intensity
       - 通道1: DoLP
       - 通道2: sin(2*AoLP)
       - 通道3: cos(2*AoLP)

4. MAE 重建损失
   └── 对所有4个通道计算 MSE 损失
```

---

## 常见误解澄清

### ❌ 误解1：Intensity 就是 RGB

**正确理解**：
- Intensity 是从偏振图像计算出来的物理量
- 虽然数值上可能接近 RGB 的亮度，但它们是不同的物理量
- Intensity 是偏振成像系统特有的测量值

### ❌ 误解2：需要 RGB 图像路径

**正确理解**：
- Stage 1 训练偏振编码器，只需要偏振图像
- 所有4个通道都是从偏振图像计算出来的
- 不需要 RGB 图像路径

### ❌ 误解3：Intensity 不参与 Loss 计算

**正确理解**：
- Intensity **完全参与** Loss 计算
- MAE 的重建损失是对所有4个通道计算的
- 所有通道都同等重要

---

## 验证方法

如果想验证 Intensity 的计算和参与情况，可以：

1. **检查数据范围**：
   ```python
   # 在 dataset_stage1.py 的 __getitem__ 中添加
   print(f"Intensity 范围: [{pixel_values[0].min():.4f}, {pixel_values[0].max():.4f}]")
   ```

2. **检查 Loss 计算**：
   ```python
   # 在 train_stage1.py 的 compute_loss 中添加
   print(f"Loss 形状: {loss.shape if hasattr(loss, 'shape') else 'scalar'}")
   print(f"Loss 值: {loss.item() if hasattr(loss, 'item') else loss}")
   ```

3. **可视化重建结果**：
   - 使用 `verify_stage1.py` 可视化重建图像
   - 检查 Intensity 通道的重建质量

---

## 总结

1. ✅ **Intensity 是从偏振图像计算出来的**，不是 RGB
2. ✅ **不需要 RGB 图像路径**，只需要偏振图像路径
3. ✅ **Intensity 完全参与 Loss 计算**，所有4个通道都参与
4. ✅ **数据流程正确**：4张角度图像 → Stokes 参数 → 4通道物理参数 → MAE 模型

您的训练指令是正确的，不需要添加 RGB 路径！



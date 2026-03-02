# Stage 1 训练中的特征结合方式

## 关键澄清

**Stage 1 训练中，输入的不是 RGB，而是偏振图像的物理参数！**

## 输入数据流程

### 1. 原始输入：4 张偏振角度图像

```
I_0.png   (0° 偏振角度)
I_45.png  (45° 偏振角度)
I_90.png  (90° 偏振角度)
I_135.png (135° 偏振角度)
```

### 2. 计算 Stokes 参数

从 4 张角度图像计算 Stokes 参数（Stokes Parameters）：

```python
# 计算 Stokes 参数
S0 = (I_0 + I_90) / 2.0      # 总光强
S1 = I_0 - I_90              # 线性偏振的 x 分量
S2 = I_45 - I_135            # 线性偏振的 y 分量
```

### 3. 计算物理参数

从 Stokes 参数计算物理量：

```python
# Intensity（总光强）
Intensity = S0 / 255.0  # 归一化到 [0, 1]

# DoLP（偏振度，Degree of Linear Polarization）
DoLP = sqrt(S1² + S2²) / S0  # 归一化到 [0, 1]

# AoLP（偏振角，Angle of Linear Polarization）
AoLP = 0.5 * arctan2(S2, S1)  # 范围 [-π/2, π/2]
```

### 4. 转换为 4 通道表示

**关键改进**：使用 `sin(2*AoLP)` 和 `cos(2*AoLP)` 代替原始 `AoLP`

**原因**：
- AoLP 具有周期性：0° 和 180° 在物理上等价（偏振方向相同）
- 直接使用 AoLP 会导致边界不连续问题（0° 和 180° 数值相差很大）
- `sin(2*AoLP)` 和 `cos(2*AoLP)` 能够唯一表示偏振角，避免周期性边界问题

```python
# 计算 sin(2*AoLP) 和 cos(2*AoLP)
sin_2aolp = sin(2 * AoLP)  # 范围 [-1, 1]
cos_2aolp = cos(2 * AoLP)  # 范围 [-1, 1]

# 归一化到 [0, 1]
sin_2aolp_normalized = (sin_2aolp + 1.0) / 2.0
cos_2aolp_normalized = (cos_2aolp + 1.0) / 2.0
```

### 5. 最终输入：4 通道图像

**结合方式：通道拼接（Channel Concatenation）**

将 4 个物理参数在**通道维度**上拼接成一个 4 通道图像：

```
输入形状: (H, W, 4)
通道 0: Intensity      (总光强)
通道 1: DoLP           (偏振度)
通道 2: sin(2*AoLP)    (偏振角的正弦表示)
通道 3: cos(2*AoLP)    (偏振角的余弦表示)
```

**注意**：
- 这不是 RGB 图像！
- 这是**像素级别的通道拼接**，每个像素位置都有 4 个值
- 类似于 RGB 图像的 3 通道，这里是 4 通道物理参数

## 数据流图

```
4 张角度图像
    ↓
[I_0, I_45, I_90, I_135]  (每个都是 H×W 的灰度图)
    ↓
计算 Stokes 参数
    ↓
[S0, S1, S2]  (每个都是 H×W)
    ↓
计算物理参数
    ↓
[Intensity, DoLP, AoLP]  (每个都是 H×W)
    ↓
转换为 4 通道表示
    ↓
[Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]  (H×W×4)
    ↓
输入到 MAE 模型
```

## 在 MAE 模型中的处理

MAE 模型接收 4 通道输入：

```python
# 模型配置
config.num_channels = 4  # 4 通道输入

# 输入形状
pixel_values: (B, 4, H, W)  # Batch, Channels, Height, Width
```

**关键点**：
- MAE 的 `patch_embeddings.projection` 层会处理这 4 个通道
- 从 3 通道预训练权重扩展到 4 通道（前 3 个通道复制 RGB 权重，第 4 个通道初始化为前 3 个通道的平均值）
- 模型学习如何从这 4 个物理参数中提取特征

## 与 Stage 2/3 的区别

### Stage 1（MAE 预训练）
- **输入**：4 通道物理参数 `[Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]`
- **处理**：自监督学习（重建被 mask 的 patch）
- **输出**：编码器特征（用于 Stage 2）

### Stage 2/3（多模态训练）
- **RGB 流**：RGB 图像 → CLIP 编码器 → RGB 特征
- **Polar 流**：4 通道物理参数 → Stage 1 编码器 → Polar 特征
- **结合方式**：在特征级别拼接（不是通道拼接）
  ```python
  # Stage 2/3 中的结合
  combined_features = concat([rgb_features, polar_features], dim=-1)
  # 形状: (B, N, rgb_dim + polar_dim) = (B, 256, 1024 + 768)
  ```

## 总结

1. **Stage 1 输入不是 RGB**，而是偏振图像的 4 个物理参数
2. **结合方式是通道拼接**：4 个物理参数在通道维度上拼接成 4 通道图像
3. **每个像素位置**都有 4 个值：`[Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]`
4. **类似于 RGB 图像的 3 通道**，这里是 4 通道物理参数图像
5. **AoLP 使用 sin/cos 表示**，避免周期性边界问题


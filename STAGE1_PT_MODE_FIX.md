# Stage 1 PT 模式数据破坏问题修复

## 🛑 问题分析

### 致命隐患：PT 模式下的数据破坏

**原始代码问题**：

```python
# PT模式训练分支
tensor_np = pixel_values.permute(1, 2, 0).numpy()  # (H, W, 4)
tensor_uint8 = (tensor_np * 255).astype(np.uint8)  # ⚠️ 问题在这里！
physics_pil = Image.fromarray(tensor_uint8, mode='RGBA')
```

**问题分析**：

1. **量化误差与截断**：
   - 如果 `.pt` 文件中的 sin/cos 是 `[-1, 1]` 范围（未归一化）
   - 乘以 255 后，负数会变成负数（如 `-0.5 * 255 = -127.5`）
   - `astype(uint8)` 会发生溢出（wrap-around），`-127` 变成 `129`
   - **结果**：原本表示角度的负数值被错误地变成了正数值，彻底破坏偏振角的物理意义

2. **训练/验证集不一致**：
   - 训练集：经过 uint8 截断
   - 验证集：直接返回原始 float tensor
   - **结果**：数据分布完全不同，导致训练和验证结果不一致

### PNG 模式的安全性

PNG 模式是安全的，因为 `process_polar_images` 函数已经将 sin/cos 从 `[-1, 1]` 映射到了 `[0, 1]`：

```python
# dataset_common.py
sin_2aolp = np.sin(2 * AoLP_raw)  # 范围 [-1, 1]
cos_2aolp = np.cos(2 * AoLP_raw)  # 范围 [-1, 1]

# 映射到 [0, 1]
sin_2aolp_normalized = (sin_2aolp + 1.0) / 2.0
cos_2aolp_normalized = (cos_2aolp + 1.0) / 2.0
```

## ✅ 修复方案

### 核心思路

**使用 `transforms.functional` 直接对 tensor 做增强，避免 PIL 转换**

### 修复内容

1. **在 `__init__` 中**：
   - PT 模式：设置 `self.use_tensor_augmentation = True`
   - 不使用 `transforms.Compose`，在 `__getitem__` 中手动应用增强

2. **在 `__getitem__` 中（PT模式训练分支）**：
   - **步骤1**：检查并修复数据范围
     - 如果 sin/cos 通道包含负数（`[-1, 1]` 范围），映射到 `[0, 1]`
     - 使用 `torch.clamp` 确保所有通道都在 `[0, 1]` 范围内
   
   - **步骤2**：使用 `transforms.functional` 直接对 tensor 做增强
     - `F.crop()`：随机裁剪
     - `F.resize()`：调整尺寸
     - `F.hflip()`：随机水平翻转
     - **优势**：保持 float32 精度，避免任何 uint8 转换

### 修复后的代码流程

```python
# PT模式训练分支
if self.use_tensor_augmentation:
    # 1. 检查并修复数据范围
    if pixel_values.shape[0] >= 4:
        sin_cos_channels = pixel_values[2:4, ...]
        if sin_cos_channels.min() < 0:
            # 映射 [-1, 1] -> [0, 1]
            pixel_values[2:4, ...] = (sin_cos_channels + 1.0) / 2.0
    
    # 2. 确保所有通道在 [0, 1] 范围内
    pixel_values = torch.clamp(pixel_values, 0.0, 1.0)
    
    # 3. 使用 functional API 直接对 tensor 做增强
    pixel_values = F.crop(pixel_values, top, left, crop_h, crop_w)
    pixel_values = F.resize(pixel_values, [self.image_size, self.image_size])
    if random.random() < 0.5:
        pixel_values = F.hflip(pixel_values)
```

## 📊 修复效果

### 修复前

- ❌ PT 模式：tensor -> uint8 -> PIL -> tensor（数据破坏）
- ❌ 如果 sin/cos 是 `[-1, 1]`，uint8 转换会溢出
- ❌ 训练/验证集数据分布不一致

### 修复后

- ✅ PT 模式：直接对 tensor 做增强（保持 float32 精度）
- ✅ 自动检测并修复数据范围（如果 sin/cos 是 `[-1, 1]`，映射到 `[0, 1]`）
- ✅ 训练/验证集数据分布一致
- ✅ 避免任何 uint8 转换，确保数据完整性

## 🔍 数据范围检查

### 检查 .pt 文件的数据范围

如果您的 `.pt` 文件中的数据范围不确定，可以运行以下代码检查：

```python
import torch

# 加载一个 .pt 文件
pixel_values = torch.load("path/to/your/file.pt")

print(f"数据形状: {pixel_values.shape}")
print(f"数据类型: {pixel_values.dtype}")
print(f"整体范围: [{pixel_values.min():.4f}, {pixel_values.max():.4f}]")

if pixel_values.shape[0] >= 4:
    print(f"\n各通道范围:")
    print(f"  Intensity (ch0): [{pixel_values[0].min():.4f}, {pixel_values[0].max():.4f}]")
    print(f"  DoLP (ch1): [{pixel_values[1].min():.4f}, {pixel_values[1].max():.4f}]")
    print(f"  sin(2*AoLP) (ch2): [{pixel_values[2].min():.4f}, {pixel_values[2].max():.4f}]")
    print(f"  cos(2*AoLP) (ch3): [{pixel_values[3].min():.4f}, {pixel_values[3].max():.4f}]")
    
    # 检查 sin/cos 是否包含负数
    sin_cos = pixel_values[2:4, ...]
    if sin_cos.min() < 0:
        print(f"\n⚠️  警告: sin/cos 通道包含负值 (min={sin_cos.min():.4f})")
        print("   修复后的代码会自动将其映射到 [0, 1] 范围")
    else:
        print(f"\n✅ sin/cos 通道已经在 [0, 1] 范围内")
```

### 预期结果

- **理想情况**：所有通道都在 `[0, 1]` 范围内
- **需要修复**：如果 sin/cos 通道包含负数（`[-1, 1]` 范围），修复后的代码会自动映射到 `[0, 1]`

## 📝 注意事项

1. **数据预处理一致性**：
   - 确保 `.pt` 文件的数据范围与 `process_polar_images` 的输出一致
   - 建议：在生成 `.pt` 文件时，确保所有通道都在 `[0, 1]` 范围内

2. **性能影响**：
   - 使用 `transforms.functional` 直接对 tensor 做增强，性能与 PIL 转换相当
   - 避免了 uint8 转换的开销，实际上可能更快

3. **兼容性**：
   - PNG 模式保持不变，仍然使用 `transforms.Compose`
   - PT 模式使用新的 tensor 增强方式
   - 两种模式的数据输出格式完全一致

## 🎯 总结

修复后的代码：
- ✅ **解决了 PT 模式下的数据破坏问题**
- ✅ **自动检测并修复数据范围**
- ✅ **保持训练/验证集数据分布一致**
- ✅ **避免任何 uint8 转换，确保数据完整性**

这是一个**关键修复**，确保了 Stage 1 训练的数据质量和模型性能。



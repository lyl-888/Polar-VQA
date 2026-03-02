# Stage 1 3通道训练迁移指南

## 概述

已将 Stage 1 训练和验证脚本从4通道（Intensity, DoLP, sin, cos）迁移到3通道（DoLP, sin, cos），移除了Intensity通道。

## 已完成的修改

### 1. `train_stage1.py`
- ✅ 将 `num_channels` 默认值从4改为3
- ✅ 移除了权重扩展策略，直接使用3通道预训练权重
- ✅ 更新了所有相关注释和文档

### 2. `verify_stage1.py`
- ✅ 将 `num_channels` 默认值从4改为3
- ✅ 修改了数据加载逻辑：自动从4通道数据中提取后3个通道
- ✅ 更新了可视化代码：只显示3个通道
- ✅ 更新了指标计算：支持3通道模式

## ⚠️ 需要额外修改：数据集代码

**重要**：`dataset_stage1.py` 也需要修改，否则训练时会出错（数据集返回4通道，但模型期望3通道）。

### 需要修改的地方：

1. **`PolarMAEDataset.__getitem__` 方法**：
   - 在返回 `pixel_values` 之前，如果数据是4通道，需要去掉Intensity通道
   - 修改位置：`dataset_stage1.py` 第406-560行

2. **建议的修改方式**：

```python
# 在 __getitem__ 方法中，加载数据后添加：
if self.use_pt_data:
    pixel_values = torch.load(pt_path, map_location='cpu')
    # ... 现有的数据增强代码 ...
    
    # ⚠️ 新增：如果数据是4通道，去掉Intensity通道（第0个通道）
    if pixel_values.shape[0] == 4:
        pixel_values = pixel_values[1:4, :, :]  # 只保留 DoLP, sin, cos
else:
    # PNG模式：process_polar_images 返回4通道
    physics_img = process_polar_images(polar_paths=polar_paths)  # (H, W, 4)
    # ... 现有的转换代码 ...
    
    # ⚠️ 新增：去掉Intensity通道
    physics_img = physics_img[:, :, 1:4]  # 只保留 DoLP, sin, cos
```

## 使用说明

### 训练命令（无需修改）

```bash
python train_stage1.py \
    --polar_root /path/to/polar \
    --use_pt_data \
    --pt_root /path/to/pt \
    --num_channels 3 \
    --output_dir /path/to/output
```

### 验证命令（无需修改）

```bash
python verify_stage1.py \
    --checkpoint /path/to/checkpoint \
    --polar_root /path/to/polar \
    --use_pt_data \
    --scene_id 17 \
    --base_name 0003
```

## 优势

1. **更聚焦**：只学习偏振特有的信息（DoLP和AoLP）
2. **权重加载更简单**：直接使用3通道预训练权重，无需扩展策略
3. **训练更稳定**：3个通道的分布更一致（DoLP: [0,1], sin/cos: [0,1]）
4. **与Stage 2架构匹配**：Stage 2使用CLIP提取RGB特征（包含光强信息），Stage 1专注偏振信息

## 注意事项

1. **数据兼容性**：现有的4通道 `.pt` 文件仍然可以使用，代码会自动去掉Intensity通道
2. **向后兼容**：验证脚本支持3通道和4通道模式（自动检测）
3. **训练数据**：确保数据集代码已修改，否则训练时会报维度不匹配错误


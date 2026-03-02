# PolarVLM Stage 3 批量验证完整命令集

本文档提供了完整的批量验证命令，包括不同类型的问题和基于坐标的问题。

---

## 基础命令结构

```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id <场景ID> \
  --max_images <图片数量> \
  --question "<问题文本>" \
  --output_json <输出文件> \
  [其他参数]
```

---

## 1. Content 类型问题（内容描述）

### 1.1 基础内容描述（全图）

```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 10 \
  --max_images 10 \
  --question "Describe the visual content in this image." \
  --output_json results_scene10_content.json
```

### 1.2 详细内容描述

```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 10 \
  --max_images 10 \
  --question "Describe the main objects and their spatial arrangement in this image." \
  --output_json results_scene10_content_detailed.json
```

### 1.3 简洁内容描述

```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 10 \
  --max_images 10 \
  --question "What objects are visible in this image?" \
  --output_json results_scene10_content_simple.json
```

---

## 2. Detail 类型问题（颜色/材质/属性）

### 2.1 颜色询问

```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 10 \
  --max_images 10 \
  --question "What are the colors of the main objects in this image?" \
  --output_json results_scene10_detail_color.json
```

### 2.2 材质询问

```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 10 \
  --max_images 10 \
  --question "What are the materials and textures of the objects in this image?" \
  --output_json results_scene10_detail_material.json
```

### 2.3 颜色和材质组合

```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 10 \
  --max_images 10 \
  --question "What are the color and material of the main object in this image?" \
  --output_json results_scene10_detail_color_material.json
```

---

## 3. Spatial 类型问题（空间关系）

### 3.1 位置询问

```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 10 \
  --max_images 10 \
  --question "Where is the main object located in this image?" \
  --output_json results_scene10_spatial_location.json
```

### 3.2 相对位置询问

```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 10 \
  --max_images 10 \
  --question "What objects are located to the left and right of the main object?" \
  --output_json results_scene10_spatial_relative.json
```

### 3.3 空间排列询问

```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 10 \
  --max_images 10 \
  --question "Describe the spatial arrangement of objects in this image." \
  --output_json results_scene10_spatial_arrangement.json
```

---

## 4. 基于坐标的问题（区域描述）

### 4.1 指定区域内容描述（手动坐标）

坐标格式：`[xmin, ymin, xmax, ymax]`，归一化到 0-1 范围。

```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 10 \
  --max_images 10 \
  --question "Describe the object in region [0.100, 0.200, 0.300, 0.400]." \
  --output_json results_scene10_region_content.json
```

### 4.2 指定区域颜色/材质询问

```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 10 \
  --max_images 10 \
  --question "What are the color and material of the object in region [0.100, 0.200, 0.300, 0.400]?" \
  --output_json results_scene10_region_detail.json
```

### 4.3 指定区域物体识别

```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 10 \
  --max_images 10 \
  --question "What object is located at region [0.100, 0.200, 0.300, 0.400]?" \
  --output_json results_scene10_region_object.json
```

### 4.4 常用坐标区域示例

**左上角区域**（例如：`[0.0, 0.0, 0.3, 0.3]`）：
```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 10 \
  --max_images 10 \
  --question "Describe the visual content in region [0.0, 0.0, 0.3, 0.3]." \
  --output_json results_scene10_region_top_left.json
```

**中心区域**（例如：`[0.35, 0.35, 0.65, 0.65]`）：
```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 10 \
  --max_images 10 \
  --question "Describe the visual content in region [0.35, 0.35, 0.65, 0.65]." \
  --output_json results_scene10_region_center.json
```

**右下角区域**（例如：`[0.7, 0.7, 1.0, 1.0]`）：
```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 10 \
  --max_images 10 \
  --question "Describe the visual content in region [0.7, 0.7, 1.0, 1.0]." \
  --output_json results_scene10_region_bottom_right.json
```

---

## 5. 不同场景的验证命令

### 5.1 场景 04（验证 5 张图片）

```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 04 \
  --max_images 5 \
  --question "Describe the visual content in this image." \
  --output_json results_scene04_content.json
```

### 5.2 场景 17（验证 10 张图片，不同类型问题）

**Content 类型**：
```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 17 \
  --max_images 10 \
  --question "Describe the visual content in this image." \
  --output_json results_scene17_content.json
```

**Detail 类型**：
```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 17 \
  --max_images 10 \
  --question "What are the color and material of the main objects in this image?" \
  --output_json results_scene17_detail.json
```

**Spatial 类型**：
```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 17 \
  --max_images 10 \
  --question "Describe the spatial arrangement of objects in this image." \
  --output_json results_scene17_spatial.json
```

### 5.3 场景 23（验证 15 张图片）

```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 23 \
  --max_images 15 \
  --question "Describe the visual content in this image." \
  --output_json results_scene23_content.json
```

---

## 6. 不同生成参数的命令

### 6.1 确定性生成（低温度，更确定）

```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 10 \
  --max_images 10 \
  --question "Describe the visual content in this image." \
  --temperature 0.3 \
  --output_json results_scene10_deterministic.json
```

### 6.2 多样性生成（高温度，更多样）

```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 10 \
  --max_images 10 \
  --question "Describe the visual content in this image." \
  --temperature 0.9 \
  --output_json results_scene10_diverse.json
```

### 6.3 长答案生成（更多 token）

```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 10 \
  --max_images 10 \
  --question "Describe the visual content in this image in detail." \
  --max_new_tokens 1024 \
  --output_json results_scene10_long.json
```

### 6.4 短答案生成（更少 token）

```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 10 \
  --max_images 10 \
  --question "What is the main object in this image?" \
  --max_new_tokens 128 \
  --output_json results_scene10_short.json
```

---

## 7. 完整参数命令示例

### 7.1 使用所有推荐参数

```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/home/checkpoints/polarvlm_stage3_final \
  --llm_model_name /openbayes/input/input0/models/Meta-Llama-3-8B-Instruct \
  --clip_model_name /openbayes/input/input0/models/clip-vit-large-patch14 \
  --polar_backbone /openbayes/input/input0/models/vit-base-patch16-224-in21k \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --rgb_root /openbayes/input/input0/rgb \
  --polar_root /openbayes/input/input0/polar \
  --scene_id 23 \
  --max_images 5 \
  --question "Describe the visual content in this image." \
  --max_new_tokens 512 \
  --temperature 0.2 \
  --top_p 0.9 \
  --do_sample \
  --output_json results_scene23_complete.json \
  --device cuda
```

---

## 8. 坐标说明

### 8.1 坐标格式

- **格式**：`[xmin, ymin, xmax, ymax]`
- **范围**：归一化到 0.0 - 1.0
- **示例**：`[0.1, 0.2, 0.3, 0.4]` 表示左上角 (0.1, 0.2) 到右下角 (0.3, 0.4) 的矩形区域

### 8.2 如何获取坐标

1. **从训练数据中获取**：
   - 查看 `merged_stage3_data.json` 文件中的 `bbox_norm` 字段
   - 格式：`[xmin, ymin, xmax, ymax]`（已归一化）

2. **手动指定**：
   - 根据图像尺寸计算归一化坐标
   - 公式：`x_norm = x_pixel / image_width`

3. **常用区域坐标**：
   - 全图：`[0.0, 0.0, 1.0, 1.0]`
   - 左上角：`[0.0, 0.0, 0.3, 0.3]`
   - 中心：`[0.35, 0.35, 0.65, 0.65]`
   - 右下角：`[0.7, 0.7, 1.0, 1.0]`

---

## 9. 问题模板库

### 9.1 Content 类型问题模板

- `"Describe the visual content in this image."`
- `"What objects are visible in this image?"`
- `"Describe the main objects and their spatial arrangement in this image."`
- `"What is the main subject of this image?"`
- `"Describe the scene in this image."`

### 9.2 Detail 类型问题模板

- `"What are the colors of the main objects in this image?"`
- `"What are the materials and textures of the objects in this image?"`
- `"What are the color and material of the main object in this image?"`
- `"Describe the visual properties of the main object."`
- `"What is the surface texture of the main object?"`

### 9.3 Spatial 类型问题模板

- `"Where is the main object located in this image?"`
- `"What objects are located to the left and right of the main object?"`
- `"Describe the spatial arrangement of objects in this image."`
- `"What is the relative position of objects in this image?"`
- `"What objects are in the foreground and background?"`

### 9.4 区域问题模板（带坐标）

- `"Describe the object in region [xmin, ymin, xmax, ymax]."`
- `"What are the color and material of the object in region [xmin, ymin, xmax, ymax]?"`
- `"What object is located at region [xmin, ymin, xmax, ymax]?"`
- `"Describe the visual content in region [xmin, ymin, xmax, ymax]."`
- `"Focus on region [xmin, ymin, xmax, ymax]. What is this?"`

---

## 10. 批量验证脚本示例

如果需要批量验证多个场景，可以创建一个脚本：

```bash
#!/bin/bash

# 批量验证多个场景
scenes=(04 10 17 23)

for scene in "${scenes[@]}"; do
    echo "验证场景 $scene..."
    python batch_inference.py \
      --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
      --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
      --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
      --scene_id $scene \
      --max_images 10 \
      --question "Describe the visual content in this image." \
      --output_json results_scene${scene}_content.json
done
```

---

## 注意事项

1. **坐标格式**：确保坐标格式正确，使用方括号和逗号分隔
2. **坐标范围**：坐标值必须在 0.0 - 1.0 之间
3. **文件路径**：确保 `--output_json` 的目录存在，或脚本会自动创建
4. **模型加载**：模型只加载一次，然后批量处理所有图片（效率高）
5. **错误处理**：如果某张图片处理失败，会跳过并继续处理下一张
6. **问题语言**：建议使用英文，与训练数据一致

---

## 输出格式

结果会保存为 JSON 格式，包含以下字段：

```json
[
  {
    "scene_id": "10",
    "base_name": "0000",
    "rgb_path": "/path/to/rgb/10/0000_rgb.png",
    "question": "Describe the visual content in this image.",
    "response": "A clean, well-lit room with..."
  },
  ...
]
```


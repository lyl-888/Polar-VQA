# Stage 2 数据生成说明文档

## 概述

`generate_stage2_captions_llava.py` 是一个用于生成 Stage 2 语义对齐描述数据的脚本。该脚本使用 LLaVA v1.6 Vicuna-7B 模型，通过对比 RGB（带眩光）和 GT（干净）图像来检测眩光区域，并生成用于 PolarVLM 训练的描述文本。

## 核心思路

**差分检测 + GT 图直接描述**：

1. **检测方法**：使用差分方法（GT vs RGB）检测眩光区域，这是反光位置的 ground truth
2. **描述生成策略**：
   - **有反光时**：直接看 GT 图像（干净），描述 bbox 区域的真实物体内容
   - **无反光时**：看 RGB 图像，描述普通场景
3. **输出格式**：兼容 LLaVA 训练格式，每条数据包含图像相对路径 + conversations（human / gpt）

## 整体流程

```
1. 加载 RGB 和 GT 图像
   ↓
2. 使用差分方法（GT vs RGB）检测眩光 bbox
   ↓
3. 根据检测结果处理图像：
   ├─ 有眩光：
   │  ├─ 裁剪 RGB 图像（bbox 区域）→ resize 到 224x224 → 保存到 rgb_crop/
   │  ├─ 裁剪 GT 图像（bbox 区域）→ resize 到 224x224 → 保存到 GT_crop/
   │  ├─ 裁剪 Polar 图像（4个角度，bbox 区域）→ resize 到 224x224 → 保存到 polar_crop/
   │  └─ 使用 GT crop 生成描述（看干净的图像）
   │
   └─ 无眩光：
      ├─ Resize RGB 图像到 224x224 → 保存到 rgb_crop/
      ├─ Resize GT 图像到 224x224 → 保存到 GT_crop/
      ├─ Resize Polar 图像（4个角度）到 224x224 → 保存到 polar_crop/
      └─ 使用 RGB 图像生成描述
   ↓
4. 使用 LLaVA 模型生成描述文本
   ↓
5. 清理描述文本（移除冗余短语、区域提及等）
   ↓
6. 保存为 JSON 格式（LLaVA 训练格式）
```

## 核心要点

### 1. 眩光检测方法

- **主要方法**：差分方法（GT vs RGB）
  - 计算 GT 和 RGB 图像的灰度差分
  - 应用阈值创建二值掩码
  - 查找最大轮廓作为眩光区域
  - 返回归一化 bbox 坐标 `[ymin, xmin, ymax, xmax]`
- **检测参数**：
  - `GLARE_THRESHOLD = 100`：差分阈值
  - `MIN_GLARE_AREA = 500`：最小眩光区域面积（像素）
  - `MAX_BBOX_RATIO = 0.4`：最大边界框面积比例

### 2. 图像处理策略

- **有眩光时**：
  - 裁剪 bbox 区域（添加 10% padding）
  - 如果裁剪尺寸 < 64x64，降级为 resize 原图到 224x224
  - 使用 **GT crop** 生成描述（确保描述的是真实物体，而不是眩光）
- **无眩光时**：
  - 直接 resize 原图到 224x224
  - 使用 **RGB 图像** 生成描述

### 3. 描述生成

- **Prompt**：`"Describe what you see in this image, such as objects, colors, materials, shapes, or any text. Be concise and factual. Aim for 40-50 words."`
- **生成参数**：
  - `MAX_NEW_TOKENS = 256`
  - `TEMPERATURE = 0.2`
  - `TOP_P = 0.9`
- **文本清理**：
  - 移除冗余开头短语（"The image depicts", "In this picture" 等）
  - 移除区域提及（"in this region", "this area" 等）
  - 移除坐标信息
  - 限制长度为 40-60 字

### 4. 输出格式

每条数据包含以下字段：

```json
{
  "id": "00_0000_rgb.png_crop",
  "image": "rgb_crop/00/0000_rgb.png",          // RGB crop 路径（训练输入）
  "gt_image": "GT_crop/00/0000_rgb.png",        // GT crop 路径（参考）
  "scene_id": "00",
  "bbox_norm": [0.1, 0.2, 0.3, 0.4] or null,   // 有眩光时不为 None
  "polar_crop_paths": {                         // Polar crop 路径（4个角度）
    "I_0": "polar_crop/00/0000_000.png",
    "I_45": "polar_crop/00/0000_045.png",
    "I_90": "polar_crop/00/0000_090.png",
    "I_135": "polar_crop/00/0000_135.png"
  },
  "conversations": [
    {
      "from": "human",
      "value": "Describe the image"
    },
    {
      "from": "gpt",
      "value": "A dark wooden cabinet with glass doors."
    }
  ]
}
```

### 5. 裁剪图像保存

所有处理后的图像（裁剪或 resize）都会保存到对应的 crop 目录：

- `rgb_crop/{scene_id}/`：RGB 裁剪图像
- `GT_crop/{scene_id}/`：GT 裁剪图像
- `polar_crop/{scene_id}/`：Polar 裁剪图像（4个角度）

**注意**：所有图像都会 resize 到 **224x224** 像素（LLM 视觉编码器支持的尺寸）。

## 使用示例

### 基本用法（生成全部场景）

```bash
python generate_stage2_captions_llava.py \
    --model_name /path/to/llava-v1.6-vicuna-7b-hf \
    --rgb_root /openbayes/input/input0/rgb \
    --polar_root /openbayes/input/input0/polar \
    --gt_root /openbayes/input/input0/GT \
    --output_json stage2_gt_captions_all.json
```

### 限制场景范围

```bash
python generate_stage2_captions_llava.py \
    --model_name /path/to/llava-v1.6-vicuna-7b-hf \
    --rgb_root /openbayes/input/input0/rgb \
    --polar_root /openbayes/input/input0/polar \
    --gt_root /openbayes/input/input0/GT \
    --scene_id_min 0 \
    --scene_id_max 10 \
    --max_images_per_scene 20 \
    --output_json stage2_gt_captions_test.json
```

### 主要参数说明

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `--model_name` | LLaVA 模型路径 | `llava-hf/llava-v1.6-vicuna-7b-hf` |
| `--rgb_root` | RGB 图像根目录 | `/openbayes/input/input0/rgb` |
| `--polar_root` | 偏振图像根目录 | `/openbayes/input/input0/polar` |
| `--gt_root` | GT 图像根目录 | 自动查找 |
| `--detection_method` | 检测方法 | `diff`（差分方法） |
| `--scene_id_min` | 最小场景ID（包含） | None（不限制） |
| `--scene_id_max` | 最大场景ID（包含） | None（不限制） |
| `--max_images_per_scene` | 每个场景最多处理的图像数 | None（不限制） |
| `--max_total_images` | 总共最多处理的图像数 | None（不限制） |
| `--batch_size` | 批处理大小（当前未使用） | 4 |
| `--output_json` | 输出 JSON 文件路径 | 自动生成（带时间戳） |

## 生成结果示例

根据实际运行结果：

```
================ 生成完成 ================
共成功生成 5722 条 Stage 2 GT Captions
输出文件：/output/train/stage2_gt_captions_all.json

  - 有眩光区域（bbox_norm 不为 None）：1897 条
  - 无眩光区域（bbox_norm 为 None）：3825 条
  - 裁剪图像保存目录：
    * RGB crop: /openbayes/input/input0/rgb_crop
    * GT crop: /openbayes/input/input0/GT_crop
    * Polar crop: /openbayes/input/input0/polar_crop
```

### 统计信息解读

- **总数据量**：5722 条
- **有眩光数据**：1897 条（33.2%）
  - 这些数据使用 GT crop 生成描述，确保描述的是真实物体
- **无眩光数据**：3825 条（66.8%）
  - 这些数据使用 RGB 图像生成描述
- **所有图像**：都已 resize 到 224x224 并保存到对应的 crop 目录

## 训练时的语义对齐

在 Stage 2 训练中：

- **输入**：
  - RGB crop（带反光）或 RGB resize（无反光）
  - Polar crop（4个角度）→ 处理为 4 通道物理参数（Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)）
- **目标**：GT 图中的真实物体描述（无反光）
- **模型学习**：偏振特征（反光区域）→ 背后真实物体的语义

通过这种方式，模型学习到如何使用偏振信息来"透视"眩光，描述背后的真实物体。

## 注意事项

1. **模型要求**：需要 4bit 量化的 LLaVA 模型（使用 BitsAndBytesConfig）
2. **显存需求**：LLaVA-7B 4bit 量化需要约 6-8GB 显存
3. **图像尺寸**：所有输出图像统一为 224x224 像素
4. **描述长度**：生成的描述会被限制在 40-60 字之间
5. **文件路径**：输出的 JSON 中使用相对路径（相对于数据根目录）
6. **批处理**：当前代码中的批处理功能未完全实现，实际为逐个处理

## 依赖要求

- `transformers >= 4.37`（支持 LlavaNextForConditionalGeneration）
- `bitsandbytes`（4bit 量化）
- `accelerate`, `safetensors`
- `pillow`, `numpy`, `opencv-python`, `tqdm`
- `dataset_common`（项目自定义模块，包含 `process_polar_images` 函数）

## 与 Stage 3 的区别

| 特性 | Stage 2 | Stage 3 |
|-----|---------|---------|
| **任务** | 语义对齐（描述生成） | 视觉问答（QA 生成） |
| **输出** | 单一描述文本 | 3 种类型的 QA 对（Content, Detail, Spatial） |
| **模型** | LLaVA-7B | Qwen2-VL-72B-AWQ |
| **描述对象** | 裁剪后的图像（224x224） | 原图 + bbox 坐标 |
| **数据格式** | 单轮对话（human + gpt） | 单轮对话（human + gpt），但类型更丰富 |


# Stage 3 VQA 数据格式转换与合并指南

## 功能说明

本脚本用于：
1. **转换第一种格式**：将 `stage3_vqa_visual_direct_merged.json` 的 bbox 从像素坐标转换为归一化坐标，并在 prompt 中添加坐标
2. **转换第二种格式**：将 `stage3_qa_pairs_all.json` 转换为第一种格式（拆分成 3 条独立样本）
3. **合并数据**：将两种格式的数据合并为一个统一格式的文件

## 使用方法

### 基本用法

```bash
python convert_and_merge_stage3_data.py \
    --old_format stage3_vqa_visual_direct_merged.json \
    --new_format stage3_qa_pairs_all.json \
    --data_root /openbayes/input/input0 \
    --output merged_stage3_data.json
```

### 参数说明

- `--old_format`: 第一种格式的 JSON 文件路径（必需）
- `--new_format`: 第二种格式的 JSON 文件路径（必需）
- `--data_root`: 数据根目录，用于查找图像文件以获取尺寸（默认：`/openbayes/input/input0`）
- `--output`: 输出合并后的 JSON 文件路径（默认：`merged_stage3_data.json`）

### 完整示例

```bash
# 转换并合并数据
python convert_and_merge_stage3_data.py \
    --old_format stage3_vqa_visual_direct_merged.json \
    --new_format 1/stage3_qa_pairs_all.json \
    --data_root /openbayes/input/input0 \
    --output merged_stage3_data_final.json
```

## 数据格式说明

### 输入格式 1（旧格式）

```json
{
  "image": "GT/04/0000_rgb.png",
  "input_path": "rgb/04/0002_rgb.png",
  "gt_path": "GT/04/0000_rgb.png",
  "conversations": [
    {
      "from": "human",
      "value": "请简要描述框选区域内的视觉内容。"
    },
    {
      "from": "gpt",
      "value": "这一区域展示了中国山水画的中景部分..."
    }
  ],
  "scene_id": "04",
  "bbox": [125, 329, 186, 24],  // [x, y, width, height] 像素坐标
  "type": "typeB_visual",
  "subtype": "content"
}
```

**转换后**：
- `bbox` 字段保留（兼容性）
- 添加 `bbox_norm` 字段：归一化坐标 `[xmin, ymin, xmax, ymax]` (0-1)
- `human` 的 `value` 中添加坐标信息：`"聚焦区域 [0.122, 0.321, 0.304, 0.345]。请简要描述框选区域内的视觉内容。"`

### 输入格式 2（新格式）

```json
{
  "id": "00_0000_rgb.png",
  "image": "rgb/00/0000_rgb.png",
  "gt_image": "GT/00/0000_rgb.png",
  "scene_id": "00",
  "bbox_norm": [0.702, 0.000, 0.797, 0.075],
  "qa_content": {
    "q": "Focus on region [0.702, 0.000, 0.797, 0.075]. What is the object?",
    "a": "A potted plant with green leaves"
  },
  "qa_detail": {
    "q": "What is the color of the leaves?",
    "a": "The leaves are a vibrant green color"
  },
  "qa_spatial": {
    "q": "Where is the potted plant located?",
    "a": "The potted plant is located on a table indoors"
  }
}
```

**转换后**（拆分成 3 条独立样本）：

```json
// 样本 1: Content
{
  "image": "GT/00/0000_rgb.png",
  "input_path": "rgb/00/0000_rgb.png",
  "gt_path": "GT/00/0000_rgb.png",
  "scene_id": "00",
  "bbox_norm": [0.702, 0.000, 0.797, 0.075],
  "type": "typeB_visual",
  "subtype": "content",
  "conversations": [
    {
      "from": "human",
      "value": "Focus on region [0.702, 0.000, 0.797, 0.075]. What is the object?"
    },
    {
      "from": "gpt",
      "value": "A potted plant with green leaves"
    }
  ]
}

// 样本 2: Detail（类似结构，subtype: "detail"）
// 样本 3: Spatial（类似结构，subtype: "spatial"）
```

### 输出格式（合并后）

合并后的数据使用统一的第一种格式，所有样本都包含：
- `bbox_norm`: 归一化坐标 `[xmin, ymin, xmax, ymax]`
- `conversations`: 包含 human 和 gpt 的对话
- `subtype`: `"content"`, `"detail"`, 或 `"spatial"`

## 注意事项

1. **图像尺寸获取**：脚本会尝试从 GT 图像文件读取尺寸。如果无法读取，会使用默认尺寸 1024x1024，并输出警告。

2. **坐标转换**：第一种格式的 `bbox` 是 `[x, y, width, height]` 格式（像素坐标），会转换为归一化坐标 `[xmin, ymin, xmax, ymax]`。

3. **Prompt 坐标添加**：
   - 如果 prompt 中已包含坐标格式（如 `[0.123, 0.456, 0.789, 0.012]`），则不再重复添加
   - 中文 prompt：添加 `"聚焦区域 [xmin, ymin, xmax, ymax]。"`
   - 英文 prompt：添加 `"Focus on region [xmin, ymin, xmax, ymax]."`

4. **数据拆分**：第二种格式的一条数据会拆分成 3 条独立样本（content, detail, spatial），即使某些 QA 类型为 `null` 也会处理。

5. **字段兼容性**：
   - 第一种格式转换后，保留原始的 `bbox` 字段（像素坐标），同时添加 `bbox_norm`（归一化坐标）
   - 第二种格式转换后，只包含 `bbox_norm`，不包含 `bbox`（因为需要图像尺寸才能反推）

## 统计信息

脚本运行完成后会输出：
- 第一种格式转换后的样本数量
- 第二种格式转换后的样本数量
- 合并后的总样本数量
- 按 subtype 分布统计

## 错误处理

- 如果无法读取图像尺寸，会使用默认值并输出警告
- 如果 bbox 转换失败，会跳过该样本并输出警告
- 如果缺少必要字段，会跳过该样本并输出警告


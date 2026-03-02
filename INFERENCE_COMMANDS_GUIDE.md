# PolarVLM Stage 3 推理脚本完整命令指南

本文档提供了 `inference.py` 所有推理模式的完整命令示例，包括所有参数和详细说明。
python detect_glare_bboxes.py \
    --rgb_root /openbayes/input/input0/rgb_test \
    --gt_root /openbayes/input/input0/GT_test \
    --polar_root /openbayes/input/input0/polar_test \
    --scene_id 05 \
    --output glare_bboxes_scene5.json
---

## 📋 目录

1. [模式1：从反光框检测 JSON 文件读取（批量推理）](#模式1从反光框检测-json-文件读取批量推理)
2. [模式2：任意图片路径推理（单张图片）](#模式2任意图片路径推理单张图片)
3. [模式3：传统模式（scene_id + base_name）](#模式3传统模式scene_id--base_name)

---

## 模式1：从反光框检测 JSON 文件读取（批量推理）

**用途**：从反光框检测脚本生成的 JSON 文件中读取多张图片及其反光框坐标，自动生成三类问题（Content、Detail、Spatial）并进行批量推理。

**适用场景**：
- 已经使用 `detect_glare_bboxes.py` 检测了某个场景的所有反光框
- 需要对多张图片进行批量推理
- 需要自动生成三类问题

### 完整命令示例（所有参数）

```bash
python inference.py \
    --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
    --glare_bbox_json glare_bboxes_scene5.json \
    --data_root /openbayes/input/input0 \
    --llm_model_name /openbayes/input/input0/models/Meta-Llama-3-8B-Instruct \
    --clip_model_name /openbayes/input/input0/models/clip-vit-large-patch14 \
    --polar_backbone /openbayes/input/input0/models/vit-base-patch16-224-in21k \
    --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
    --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
    --qa_types all \
    --max_new_tokens 512 \
    --temperature 0.2 \
    --top_p 0.9 \
    --do_sample \
    --device cuda
```

### 参数说明

| 参数 | 是否必需 | 默认值 | 说明 |
|------|---------|--------|------|
| `--checkpoint_dir` | ✅ **必需** | - | Stage 3 检查点目录（包含 `adapter_model.safetensors`、`projector.pth`、`polar_tower.pth`、`model_config.json` 等） |
| `--glare_bbox_json` | ✅ **必需**（模式1） | - | 反光框检测 JSON 文件路径（由 `detect_glare_bboxes.py` 生成） |
| `--data_root` | ❌ | `/openbayes/input/input0` | 数据根目录（用于解析 JSON 文件中的相对路径） |
| `--llm_model_name` | ❌ | `/openbayes/input/input0/models/Meta-Llama-3-8B-Instruct` | 基础 LLM 模型路径（Instruct 版本） |
| `--clip_model_name` | ❌ | `openai/clip-vit-large-patch14` | CLIP 模型路径（支持本地路径或 Hugging Face 模型名） |
| `--polar_backbone` | ❌ | `google/vit-base-patch16-224-in21k` | 偏振流 backbone 路径（支持本地路径或 Hugging Face 模型名） |
| `--stage1_checkpoint` | ❌ | `None` | Stage 1 检查点路径（可选，用于加载 MAE 编码器权重） |
| `--stage2_checkpoint` | ❌ | `None` | Stage 2 检查点路径（可选，用于加载投影器权重） |
| `--hf_token` | ❌ | `None` | Hugging Face token（如果模型需要认证） |
| `--qa_types` | ❌ | `["all"]` | 要生成的问题类型：`content`、`detail`、`spatial`、`all`（可指定多个，如 `--qa_types content detail`） |
| `--max_new_tokens` | ❌ | `512` | 最大生成 token 数（控制回答长度） |
| `--temperature` | ❌ | `0.7` | 温度参数（0.0-1.0，越高越随机） |
| `--top_p` | ❌ | `0.9` | nucleus sampling 参数（0.0-1.0） |
| `--do_sample` | ❌ | `True` | 使用采样（默认开启，添加此参数启用） |
| `--device` | ❌ | `cuda` | 设备（`cuda` 或 `cpu`） |

### 简化命令示例（使用默认值）

```bash
python inference.py \
    --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
    --glare_bbox_json glare_bboxes_scene10.json \
    --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
    --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
    --qa_types all
```

### 输出

- 控制台输出：每个图片的三类问题和回答
- 自动保存结果到：`glare_bboxes_scene10.results.json`

---

## 模式2：任意图片路径推理（单张图片）

**用途**：对任意一张图片进行推理，可以指定 RGB 图像路径、偏振图像路径和反光框坐标。

**适用场景**：
- 单张图片推理
- 图片不在标准目录结构中
- 需要自定义反光框坐标
- 需要自定义问题

### 完整命令示例（所有参数 - 带反光框坐标）

```bash
python inference.py \
    --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
    --rgb_path /openbayes/input/input0/rgb_est/10/0002_rgb.png \
    --polar_paths \
        /openbayes/input/input0/polar/10/0002_000.png \
        /openbayes/input/input0/polar/10/0002_045.png \
        /openbayes/input/input0/polar/10/0002_090.png \
        /openbayes/input/input0/polar/10/0002_135.png \
    --bbox_norm 0.1021 0.3213 0.2541 0.3447 \
    --llm_model_name /openbayes/input/input0/models/Meta-Llama-3-8B-Instruct \
    --clip_model_name /openbayes/input/input0/models/clip-vit-large-patch14 \
    --polar_backbone /openbayes/input/input0/models/vit-base-patch16-224-in21k \
    --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
    --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
    --hf_token YOUR_HF_TOKEN \
    --qa_types all \
    --max_new_tokens 512 \
    --temperature 0.2 \
    --top_p 0.9 \
    --do_sample \
    --device cuda
```

### 完整命令示例（所有参数 - 自定义问题）

**示例1：自定义问题（不带坐标）**
```bash
python inference.py \
    --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
    --rgb_path /openbayes/input/input0/rgb_est/10/0002_rgb.png \
    --polar_paths \
        /openbayes/input/input0/polar/10/0002_000.png \
        /openbayes/input/input0/polar/10/0002_045.png \
        /openbayes/input/input0/polar/10/0002_090.png \
        /openbayes/input/input0/polar/10/0002_135.png \
    --question "What is the color of the object in the center of the image?" \
    --llm_model_name /openbayes/input/input0/models/Meta-Llama-3-8B-Instruct \
    --clip_model_name /openbayes/input/input0/models/clip-vit-large-patch14 \
    --polar_backbone /openbayes/input/input0/models/vit-base-patch16-224-in21k \
    --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
    --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
    --hf_token YOUR_HF_TOKEN \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --top_p 0.9 \
    --do_sample \
    --device cuda
```

**示例2：自定义问题（在问题中直接写坐标）**
```bash
python inference.py \
    --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
    --rgb_path /openbayes/input/input0/rgb_est/10/0002_rgb.png \
    --polar_paths \
        /openbayes/input/input0/polar/10/0002_000.png \
        /openbayes/input/input0/polar/10/0002_045.png \
        /openbayes/input/input0/polar/10/0002_090.png \
        /openbayes/input/input0/polar/10/0002_135.png \
    --question "Focus on region [0.102, 0.321, 0.254, 0.345]. What is the color and material of the object in this region?" \
    --llm_model_name /openbayes/input/input0/models/Meta-Llama-3-8B-Instruct \
    --clip_model_name /openbayes/input/input0/models/clip-vit-large-patch14 \
    --polar_backbone /openbayes/input/input0/models/vit-base-patch16-224-in21k \
    --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
    --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
    --hf_token YOUR_HF_TOKEN \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --top_p 0.9 \
    --do_sample \
    --device cuda
```

**注意**：在 `--question` 中可以**直接写入反光框坐标**，格式为 `[xmin, ymin, xmax, ymax]`（归一化坐标，0.0-1.0 范围）。例如：
- ✅ `"Focus on region [0.102, 0.321, 0.254, 0.345]. What is the color of the object?"`
- ✅ `"What are the color and material of the object in region [0.102, 0.321, 0.254, 0.345]?"`

### 参数说明

| 参数 | 是否必需 | 默认值 | 说明 |
|------|---------|--------|------|
| `--checkpoint_dir` | ✅ **必需** | - | Stage 3 检查点目录 |
| `--rgb_path` | ✅ **必需**（模式2） | - | RGB 图像完整路径（绝对路径或相对于 `--data_root` 的相对路径） |
| `--polar_paths` | ✅ **必需**（模式2） | - | 偏振图像路径（4个角度，按顺序：I_0, I_45, I_90, I_135） |
| `--bbox_norm` | ❌ | `None` | 归一化反光框坐标 `[xmin, ymin, xmax, ymax]`（0.0-1.0 范围）<br>如果指定，将自动生成三类问题；如果不指定，将生成全图问题<br>**注意**：如果同时指定 `--question`，此参数将被忽略 |
| `--question` | ❌ | `None` | 自定义问题文本（如果指定，将忽略 `--bbox_norm` 和 `--qa_types`）<br>**可以在问题中直接写入坐标**，例如：`"Focus on region [0.102, 0.321, 0.254, 0.345]. What is the color?"` |
| `--qa_types` | ❌ | `["all"]` | 要生成的问题类型（仅在未指定 `--question` 时生效） |
| 其他参数 | 同模式1 | - | 与模式1相同 |

### 简化命令示例

**示例1：带反光框坐标，生成三类问题**
```bash
python inference.py \
    --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
    --rgb_path /openbayes/input/input0/rgb_est/10/0002_rgb.png \
    --polar_paths \
        /openbayes/input/input0/polar/10/0002_000.png \
        /openbayes/input/input0/polar/10/0002_045.png \
        /openbayes/input/input0/polar/10/0002_090.png \
        /openbayes/input/input0/polar/10/0002_135.png \
    --bbox_norm 0.1021 0.3213 0.2541 0.3447 \
    --qa_types content detail spatial \
    --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
    --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector
```

**示例2：不带反光框坐标，生成全图三类问题**
```bash
python inference.py \
    --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
    --rgb_path /openbayes/input/input0/rgb_est/10/0002_rgb.png \
    --polar_paths \
        /openbayes/input/input0/polar/10/0002_000.png \
        /openbayes/input/input0/polar/10/0002_045.png \
        /openbayes/input/input0/polar/10/0002_090.png \
        /openbayes/input/input0/polar/10/0002_135.png \
    --qa_types all \
    --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
    --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector
```

**示例3：自定义问题（不带坐标）**
```bash
python inference.py \
    --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
    --rgb_path /openbayes/input/input0/rgb_est/10/0002_rgb.png \
    --polar_paths \
        /openbayes/input/input0/polar/10/0002_000.png \
        /openbayes/input/input0/polar/10/0002_045.png \
        /openbayes/input/input0/polar/10/0002_090.png \
        /openbayes/input/input0/polar/10/0002_135.png \
    --question "What is the color of the object in the center of the image?" \
    --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
    --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector
```

**示例4：自定义问题（在问题中直接写坐标）**
```bash
python inference.py \
    --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
    --rgb_path /openbayes/input/input0/rgb_est/10/0002_rgb.png \
    --polar_paths \
        /openbayes/input/input0/polar/10/0002_000.png \
        /openbayes/input/input0/polar/10/0002_045.png \
        /openbayes/input/input0/polar/10/0002_090.png \
        /openbayes/input/input0/polar/10/0002_135.png \
    --question "Focus on region [0.102, 0.321, 0.254, 0.345]. What are the color and material of the object in this region?" \
    --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
    --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector
```

---

## 模式3：传统模式（scene_id + base_name）

**用途**：使用场景 ID 和基础文件名进行推理（向后兼容旧版本）。

**适用场景**：
- 图片在标准目录结构中（`rgb_root/scene_id/base_name_rgb.png`）
- 快速测试单个场景的图片
- 向后兼容旧版本的使用方式

### 完整命令示例（所有参数）

```bash
python inference.py \
    --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
    --scene_id 10 \
    --base_name 0002 \
    --rgb_root /openbayes/input/input0/rgb \
    --polar_root /openbayes/input/input0/polar \
    --question "Describe the visual content in this image." \
    --llm_model_name /openbayes/input/input0/models/Meta-Llama-3-8B-Instruct \
    --clip_model_name /openbayes/input/input0/models/clip-vit-large-patch14 \
    --polar_backbone /openbayes/input/input0/models/vit-base-patch16-224-in21k \
    --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
    --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
    --hf_token YOUR_HF_TOKEN \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --top_p 0.9 \
    --do_sample \
    --device cuda
```

### 参数说明

| 参数 | 是否必需 | 默认值 | 说明 |
|------|---------|--------|------|
| `--checkpoint_dir` | ✅ **必需** | - | Stage 3 检查点目录 |
| `--scene_id` | ✅ **必需**（模式3） | - | 场景 ID（如 `"10"`） |
| `--base_name` | ✅ **必需**（模式3） | - | 基础文件名（不含扩展名，如 `"0002"`，对应文件名为 `0002_rgb.png`） |
| `--rgb_root` | ❌ | `/openbayes/input/input0/rgb` | RGB 图像根目录 |
| `--polar_root` | ❌ | `/openbayes/input/input0/polar` | 偏振图像根目录 |
| `--question` | ❌ | `"Describe the visual content in this image."` | 问题文本（建议使用英文，与训练一致） |
| 其他参数 | 同模式1 | - | 与模式1相同 |

### 简化命令示例

```bash
python inference.py \
    --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
    --scene_id 10 \
    --base_name 0002 \
    --question "Describe the visual content in this image." \
    --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
    --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector
```

---

## 📝 三类问题模板说明

当使用 `--qa_types all` 或指定 `content`、`detail`、`spatial` 时，会自动生成以下问题：

### Content（内容描述）
```
Focus on region [xmin, ymin, xmax, ymax]. Please briefly describe the visual content within this region.
```

### Detail（细节属性）
```
Focus on region [xmin, ymin, xmax, ymax]. What are the color and material of the object in this region?
```

### Spatial（空间关系）
```
Focus on region [xmin, ymin, xmax, ymax]. What object is located near this region?
```

**注意**：如果 `bbox_norm` 为 `None`（未指定反光框坐标），将使用全图坐标 `[0.000, 0.000, 1.000, 1.000]`。

---

## 🔄 完整工作流程示例

### 步骤1：检测反光框

```bash
python detect_glare_bboxes.py \
    --rgb_root /openbayes/input/input0/rgb_est \
    --gt_root /openbayes/input/input0/GT \
    --polar_root /openbayes/input/input0/polar \
    --scene_id 10 \
    --output glare_bboxes_scene10.json \
    --threshold 100 \
    --min_area 500 \
    --max_bbox_ratio 0.4
```

### 步骤2：批量推理（模式1）

```bash
python inference.py \
    --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
    --glare_bbox_json glare_bboxes_scene10.json \
    --data_root /openbayes/input/input0 \
    --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
    --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
    --qa_types all
```

### 步骤3：查看结果

结果会自动保存到 `glare_bboxes_scene10.results.json`，包含每个图片的三类问题和回答。

---

## ⚠️ 注意事项

1. **三种模式互斥**：`--glare_bbox_json`、`--rgb_path`、`--scene_id + --base_name` 三种模式只能选择一种。

2. **路径格式**：
   - 支持绝对路径和相对路径
   - 相对路径相对于 `--data_root` 解析

3. **问题生成**：
   - 如果指定 `--question`，将使用自定义问题，忽略 `--bbox_norm` 和 `--qa_types`
   - **可以在 `--question` 中直接写入坐标**，例如：`"Focus on region [0.102, 0.321, 0.254, 0.345]. What is the color?"`
   - 如果未指定 `--question` 但指定了 `--bbox_norm`，将根据 `--qa_types` 生成三类问题
   - 如果都未指定，将生成全图的三类问题

4. **模型加载**：
   - `--stage1_checkpoint` 和 `--stage2_checkpoint` 是可选的，但如果提供了，建议都提供以确保模型完整性

5. **生成参数**：
   - `--temperature`：控制随机性（0.0-1.0），越高越随机
   - `--top_p`：nucleus sampling（0.0-1.0）
   - `--max_new_tokens`：控制回答长度

---

## 📚 相关文件

- `detect_glare_bboxes.py`：反光框检测脚本
- `inference.py`：推理脚本（本文档）
- `batch_inference.py`：批量推理脚本（旧版本）


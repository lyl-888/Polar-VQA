# PolarVLM Stage 3 推理验证指南

## 概述

本指南用于验证 Stage 3 训练后的模型效果。推理脚本 `inference.py` 支持加载训练后的 Stage 3 模型并进行单样本推理测试。

---

## 代码检查结果

✅ **代码基本正确**，关键点已验证：

1. **模型加载**：正确加载 LoRA 适配器、projector.pth、polar_tower.pth
2. **图像预处理**：与训练阶段一致（RGB 使用 CLIPImageProcessor，偏振不使用 ImageNet 归一化）
3. **Prompt 格式**：使用 LLaMA-3 Chat 格式，与训练一致
4. **生成方法**：正确调用 `model.generate`

---

## 完整验证指令

### 基础命令

```bash
python inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --llm_model_name /openbayes/input/input0/models/Meta-Llama-3-8B-Instruct \
  --clip_model_name /openbayes/input/input0/models/clip-vit-large-patch14 \
  --polar_backbone /openbayes/input/input0/models/vit-base-patch16-224-in21k \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --rgb_root /openbayes/input/input0/rgb \
  --polar_root /openbayes/input/input0/polar \
  --scene_id 17 \
  --base_name 0002 \
  --question "Describe the visual content in this image."
```

### 参数说明

#### 必需参数
- `--checkpoint_dir`: Stage 3 检查点目录（包含 `adapter_model.safetensors`、`projector.pth`、`polar_tower.pth`、`model_config.json` 等）
- `--scene_id`: 场景 ID（如 `"10"`、`"04"`）
- `--base_name`: 基础文件名（不含扩展名，如 `"0002"`，对应文件名为 `0002_rgb.png`）

#### 模型路径（推荐指定）
- `--llm_model_name`: LLaMA-3 模型路径（默认: `/openbayes/input/input0/models/Meta-Llama-3-8B-Instruct`）
- `--clip_model_name`: CLIP 模型路径（默认: `openai/clip-vit-large-patch14`，支持本地路径）
- `--polar_backbone`: 偏振流 backbone 路径（默认: `google/vit-base-patch16-224-in21k`）

#### Stage 检查点
- `--stage1_checkpoint`: Stage 1 检查点路径（**推荐提供**，用于加载 MAE 编码器权重）
- `--stage2_checkpoint`: Stage 2 检查点路径（**推荐提供**，用于在初始化时加载投影器权重；即使不提供，Stage 3 checkpoint 中的 `projector.pth` 也会覆盖它，但提供它可以确保模型初始化的正确性）

#### 数据路径
- `--rgb_root`: RGB 图像根目录（默认: `/openbayes/input/input0/rgb`）
- `--polar_root`: 偏振图像根目录（默认: `/openbayes/input/input0/polar`）

#### 问题文本
- `--question`: 问题文本（默认: `"Describe the visual content in this image."`）
  - 建议使用英文，与训练数据一致
  - 可以使用训练数据中的问题，例如：
    - `"Describe the object in region [0.100, 0.200, 0.300, 0.400]."`
    - `"What are the color and material of the object in this region?"`
    - `"What object is located to the left of the lamp?"`

#### 生成参数（可选）
- `--max_new_tokens`: 最大生成 token 数（默认: `512`）
- `--temperature`: 温度参数（默认: `0.7`，降低温度可使输出更确定）
- `--top_p`: nucleus sampling 参数（默认: `0.9`）
- `--do_sample`: 使用采样（默认: `True`，设为 `False` 使用贪心解码）

#### 其他
- `--device`: 设备（默认: `cuda`）
- `--hf_token`: Hugging Face token（如果模型需要认证）

---

## 实际使用示例

### 示例 1: 基础图像描述

```bash
python inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 10 \
  --base_name 0002 \
  --question "Describe the visual content in this image."
```

### 示例 2: 使用训练数据中的问题

```bash
python inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 04 \
  --base_name 0002 \
  --question "Describe the object in region [0.221, 0.000, 0.315, 0.113]."
```

### 示例 3: 详细属性问题

```bash
python inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 17 \
  --base_name 0005 \
  --question "What are the color and material of the object in this region?" \
  --temperature 0.5 \
  --max_new_tokens 256
```

### 示例 4: 空间推理问题

```bash
python inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 23 \
  --base_name 0001 \
  --question "What object is located to the left of the lamp?" \
  --temperature 0.3 \
  --max_new_tokens 128
```

---

## 批量验证（验证一个场景下的多张图片）

如果需要验证一个场景下的多张图片（例如10张），可以使用 `batch_inference.py` 脚本：

### 批量验证命令

```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --llm_model_name /openbayes/input/input0/models/Meta-Llama-3-8B-Instruct \
  --clip_model_name /openbayes/input/input0/models/clip-vit-large-patch14 \
  --polar_backbone /openbayes/input/input0/models/vit-base-patch16-224-in21k \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --rgb_root /openbayes/input/input0/rgb \
  --polar_root /openbayes/input/input0/polar \
  --scene_id 10 \
  --max_images 10 \
  --question "Describe the visual content in this image." \
  --output_json results_scene10.json
```

### 批量验证参数说明

- `--scene_id`: 场景 ID（**必需**，如 `"10"`）
- `--max_images`: 最大图片数量（默认: `10`，会按文件名排序取前N张）
- `--question`: 问题文本（所有图片使用相同的问题）
- `--output_json`: 输出JSON文件路径（可选，如 `"results_scene10.json"`，会保存所有结果）

### 批量验证示例

#### 示例 1: 验证场景10的前10张图片

```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 10 \
  --max_images 10 \
  --question "Describe the visual content in this image." \
  --output_json results_scene10.json
```

#### 示例 2: 验证场景17的前5张图片（使用不同问题）

```bash
python batch_inference.py \
  --checkpoint_dir /openbayes/input/input0/checkpoints/polarvlm_stage3 \
  --stage1_checkpoint /openbayes/input/input0/checkpoints/stage1_encoder_new \
  --stage2_checkpoint /openbayes/input/input0/checkpoints/stage2_projector \
  --scene_id 17 \
  --max_images 5 \
  --question "What are the main objects in this image?" \
  --output_json results_scene17.json
```

**注意**：
- 脚本会自动扫描场景目录下的所有 `*_rgb.png` 文件
- 按文件名排序，取前 `max_images` 张
- 模型只加载一次，然后批量处理所有图片（效率更高）
- 如果某张图片处理失败，会跳过并继续处理下一张
- 最终会输出统计信息（成功/失败数量）和所有结果

### 批量验证完整命令集

详细的批量验证命令（包括不同类型的问题、基于坐标的问题等）请参考 `BATCH_INFERENCE_COMMANDS.md` 文档。

---

## 验证要点

### 1. 检查模型加载

运行命令后，应该看到：
- ✓ 读取模型配置
- ✓ Tokenizer 加载完成
- ✓ LoRA 适配器加载完成
- ✓ 多模态投影器已加载（包含 `rgb_scale` 和 `polar_scale` 的值）
- ✓ 偏振流已加载（如果未冻结）
- ✓ 模型加载完成

**关键检查点**：
- `RGB scale` 和 `Polar scale` 的值应该与训练结束时的值接近（例如：rgb_scale ≈ 1.1, polar_scale ≈ 0.03）

### 2. 检查图像预处理

- RGB 图像路径正确
- 偏振图像 4 个角度文件都存在（000, 045, 090, 135）
- ✓ 图像预处理完成

### 3. 检查生成结果

- 回答应该是完整的句子（不是单个词或短语）
- 回答应该与问题相关
- 如果问题包含坐标（bbox），回答应该关注该区域
- 如果问题询问颜色/材质，回答应该包含这些信息

---

## 常见问题

### Q: 模型加载失败？

**A:** 检查：
1. `checkpoint_dir` 路径是否正确
2. 是否包含 `adapter_model.safetensors`、`projector.pth`、`model_config.json`
3. 如果使用 Stage 1 checkpoint，路径是否正确

### Q: 偏振图像加载失败？

**A:** 检查：
1. `polar_root` 路径是否正确
2. 文件命名格式是否为 `{base_name}_000.png`、`{base_name}_045.png` 等（三位数角度格式）
3. 如果使用两位数格式（`{base_name}_0.png`），代码会自动尝试

### Q: 生成结果不合理？

**A:** 尝试：
1. 降低 `temperature`（如 `0.3` 或 `0.5`）使输出更确定
2. 增加 `max_new_tokens`（如 `512` 或 `1024`）确保答案完整
3. 使用训练数据中的问题格式（包含坐标等）

### Q: 显存不足（OOM）？

**A:** 
- Stage 3 推理通常需要 ~20-25GB 显存（4-bit 量化模型）
- A5880 48GB 显存足够
- 如果 OOM，确保使用 `--device cuda` 且模型以 4-bit 加载

### Q: 生成速度慢？

**A:** 
- 首次运行需要加载模型，可能较慢（1-2 分钟）
- 后续推理速度应该在 5-10 秒/样本（A5880）
- 如果很慢，检查是否使用了 CPU（`--device cpu`）

---

## 批量验证脚本（可选）

如果需要批量验证多个样本，可以创建脚本：

```python
# batch_inference.py
import subprocess
import sys

test_cases = [
    {"scene_id": "10", "base_name": "0002", "question": "Describe the visual content in this image."},
    {"scene_id": "04", "base_name": "0002", "question": "Describe the object in region [0.221, 0.000, 0.315, 0.113]."},
    {"scene_id": "17", "base_name": "0005", "question": "What are the color and material of the object in this region?"},
]

checkpoint_dir = "/openbayes/input/input0/checkpoints/polarvlm_stage3"
stage1_checkpoint = "/openbayes/input/input0/checkpoints/stage1_encoder_new"

for i, case in enumerate(test_cases):
    print(f"\n{'='*80}")
    print(f"Test Case {i+1}/{len(test_cases)}: Scene {case['scene_id']}, {case['base_name']}")
    print(f"{'='*80}")
    
    cmd = [
        "python", "inference.py",
        "--checkpoint_dir", checkpoint_dir,
        "--stage1_checkpoint", stage1_checkpoint,
        "--scene_id", case["scene_id"],
        "--base_name", case["base_name"],
        "--question", case["question"],
    ]
    
    subprocess.run(cmd)
```

---

## 验证清单

- [ ] 模型加载成功（所有组件都显示 ✓）
- [ ] RGB scale 和 Polar scale 值合理（与训练一致）
- [ ] 图像预处理成功（RGB 和偏振图像都加载）
- [ ] 生成结果完整（不是截断的）
- [ ] 回答与问题相关
- [ ] 回答格式合理（完整句子，不是单个词）

---

## 注意事项

1. **Prompt 格式**：推理时使用的 prompt 格式与训练时完全一致（LLaMA-3 Chat 格式）
2. **图像预处理**：RGB 使用 CLIPImageProcessor，偏振不使用 ImageNet 归一化（与训练一致）
3. **坐标格式**：如果问题中包含坐标，使用 `[xmin, ymin, xmax, ymax]` 格式（归一化，0-1）
4. **问题语言**：建议使用英文，与训练数据一致
5. **生成参数**：默认参数（temperature=0.7, top_p=0.9）通常效果较好，可根据需要调整


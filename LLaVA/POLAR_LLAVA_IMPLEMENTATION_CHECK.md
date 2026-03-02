# Polar LLaVA 实现检查清单

## ✅ 1. 模型定义模块 (Model Architecture)

### ✅ `llava/model/llava_arch.py`
- [x] **PolarLlavaMetaModel 类**：已实现
  - [x] `__init__` 中初始化 VAE 编码器和 Polar Projector
  - [x] `_initialize_polar_modules` 方法：加载 VAE，创建 `polar_encoder`、`polar_quant_conv`、`vae_latent_to_feature`、`polar_projector`
  - [x] `encode_polar_images` 方法：处理偏振图像（3通道，512x512）→ VAE编码 → 下采样到24x24 → 投影到LLM hidden_size
  - [x] `encode_images` 方法：支持 `polar_images` 参数，拼接 RGB 和 Polar 特征
  - **作用阶段**：Stage 1 和 Stage 2 都使用

### ✅ `llava/model/language_model/llava_llama.py`
- [x] **PolarLlavaLlamaModel 类**：已实现（继承自 PolarLlavaMetaModel 和 LlamaModel）
- [x] **PolarLlavaLlamaForCausalLM 类**：已实现
  - [x] `forward` 方法：支持 `polar_images` 参数
  - [x] `generate` 方法：支持 `polar_images` 参数
  - [x] `prepare_inputs_labels_for_multimodal` 方法：正确处理双流输入（RGB + Polar）
  - [x] `prepare_inputs_for_generation` 方法：支持 `polar_images`
  - [x] **PolarLlavaConfig 类**：继承自 `LlavaConfig`，添加 `polar_vae_model_path` 和 `freeze_polar_encoder` 字段
  - **作用阶段**：Stage 1 和 Stage 2 都使用

### ✅ `llava/model/builder.py`
- [x] **load_pretrained_model 函数**：已修改
  - [x] 添加 `polar_vae_model_path` 参数
  - [x] 检测是否使用 Polar 模型（通过 `polar_vae_model_path` 判断）
  - [x] 在 LoRA、base model、full model 三种加载模式下都支持 Polar 模型
  - [x] 创建 `PolarLlavaConfig` 并设置 `polar_vae_model_path`
  - **作用阶段**：Stage 1 和 Stage 2 都使用

## ✅ 2. 数据处理模块 (Data Processing)

### ✅ `llava/train/train.py - LazySupervisedDataset`
- [x] **__getitem__ 方法**：已修改
  - [x] 检查 `use_polar` 标志
  - [x] 调用 `_load_polar_image` 加载偏振图像
  - [x] 返回字典中包含 `polar_image` 字段
  - **作用阶段**：Stage 1 和 Stage 2 都使用

- [x] **_load_polar_image 方法**：已实现
  - [x] 支持 `polar_crop_paths` 字段（Stage 1 格式）
  - [x] 支持 `rgb_crop/` 和 `polar_crop/` 路径格式（需要 `data_root`）
  - [x] 支持 `rgb/` 和 `polar/` 路径格式（Stage 2 格式）
  - [x] 从 RGB 路径推断偏振路径（回退方案）
  - [x] 使用 `process_polar_images` 处理 4 张角度图像
  - [x] 提取 3 通道：`[DoLP, sin(2*AoLP), cos(2*AoLP)]`
  - [x] Resize 到 512x512（VAE 输入尺寸）
  - [x] 返回形状为 `(3, 512, 512)` 的 tensor，值范围 `[0, 1]`
  - **作用阶段**：Stage 1 和 Stage 2 都使用

### ✅ `llava/train/train.py - DataCollatorForSupervisedDataset`
- [x] **__call__ 方法**：已修改
  - [x] 检查 `polar_image` 字段
  - [x] 堆叠 `polar_images` 为 batch tensor
  - [x] 处理 None 值和形状不一致的情况
  - [x] 返回 batch 字典中包含 `polar_images` 字段
  - **作用阶段**：Stage 1 和 Stage 2 都使用

### ✅ `llava/train/train.py - DataArguments`
- [x] 添加 `polar_folder` 字段：偏振图像根目录
- [x] 添加 `use_polar` 字段：是否启用双流模式
- [x] 添加 `data_root` 字段：数据根目录（用于解析 crop 路径）
- **作用阶段**：Stage 1 和 Stage 2 都使用

## ✅ 3. 训练控制模块 (Training & Optimization)

### ✅ `llava/train/train.py - train 函数`
- [x] **模型初始化**：已修改
  - [x] 检测 `polar_vae_model_path` 是否存在
  - [x] 如果存在，使用 `PolarLlavaLlamaForCausalLM` 而不是 `LlavaLlamaForCausalLM`
  - [x] 创建 `PolarLlavaConfig` 并设置 `polar_vae_model_path`
  - [x] **关键修复 1**：确保 RGB 图像尺寸为 336x336（LLaVA v1.5 要求）
  - [x] **关键修复 3**：加载 Stage 1 训练的 Polar Projector 权重（如果提供 `pretrain_polar_projector`）
  - **作用阶段**：Stage 1 和 Stage 2 都使用

- [x] **冻结/解冻逻辑**：已实现
  - [x] **Stage 1 (Polar Projector Alignment)**：
    - [x] 冻结：RGB Tower、RGB Projector、Polar Encoder、LLM
    - [x] 解冻：Polar Projector
  - [x] **Stage 2 (Visual Instruction Tuning)**：
    - [x] 冻结：RGB Tower、RGB Projector（可选）
    - [x] 解冻：Polar Projector、LLM (LoRA)
    - [x] 可选：Polar Encoder（根据 `freeze_polar_encoder` 设置）
  - **作用阶段**：根据 `training_stage` 参数分别作用于 Stage 1 或 Stage 2

- [x] **参数统计**：已实现
  - [x] 打印可训练参数数量
  - [x] 打印各模块的冻结状态
  - **作用阶段**：Stage 1 和 Stage 2 都使用

- [x] **find_all_linear_names 函数**：已修改
  - [x] 排除 `polar_projector` 从 LoRA target_modules（确保 Polar Projector 全量微调）
  - **作用阶段**：Stage 2 使用（LoRA 配置）

### ✅ `llava/train/llava_trainer.py`
- [x] **_save_checkpoint 方法**：已修改
  - [x] 在 `keys_to_match` 中添加 `polar_projector`
  - [x] 确保 Polar Projector 权重被保存
  - **作用阶段**：Stage 1 和 Stage 2 都使用

## ✅ 4. 配置与参数模块 (Arguments & Config)

### ✅ `llava/train/train.py - ModelArguments`
- [x] 添加 `polar_vae_model_path` 参数
- [x] 添加 `freeze_polar_encoder` 参数
- [x] 添加 `freeze_rgb_tower` 参数
- [x] 添加 `freeze_rgb_projector` 参数
- [x] 添加 `training_stage` 参数（"stage1" 或 "stage2"）
- [x] 添加 `pretrain_polar_projector` 参数（用于 Stage 2 加载 Stage 1 权重）
- **作用阶段**：Stage 1 和 Stage 2 都使用

### ✅ `llava/train/train.py - DataArguments`
- [x] 添加 `polar_folder` 参数
- [x] 添加 `use_polar` 参数
- [x] 添加 `data_root` 参数
- **作用阶段**：Stage 1 和 Stage 2 都使用

## 📋 数据格式兼容性

### ✅ 数据路径说明

#### Stage 1 训练（Polar Projector Alignment）
- **训练集 JSON**: `train_stage2_data.json`
- **验证集 JSON**: `val_stage2_data.json`
- **数据根目录**: `--data_root /openbayes/input/input0`
- **RGB 图像目录**: `--image_folder /openbayes/input/input0/rgb_crop`（**带 crop**，512x512 裁剪图像）
- **Polar 图像目录**: `--polar_folder /openbayes/input/input0/polar_crop`（**带 crop**，512x512 裁剪图像）
- **JSON 格式**: 包含 `polar_crop_paths` 字段，路径如 `polar_crop/23/0000_000.png`
- **图像处理流程**：
  1. 从磁盘加载 512x512 crop 图像
  2. RGB 图像通过 `image_processor.preprocess` 自动 resize 到 336x336（CLIP 输入）
  3. Polar 图像 resize 到 512x512（VAE 输入）

#### Stage 2 训练（Visual Instruction Tuning）
- **训练集 JSON**: `merged_stage3_qwen.json`
- **验证集 JSON**: `val_stage3_qwen.json`
- **数据根目录**: `--data_root /openbayes/input/input0`
- **RGB 图像目录**: `--image_folder /openbayes/input/input0/rgb`（**不带 crop**，原始尺寸图像）
- **Polar 图像目录**: `--polar_folder /openbayes/input/input0/polar`（**不带 crop**，原始尺寸图像）
- **JSON 格式**: 包含 `input_path` 字段，路径如 `rgb/04/0002_rgb.png`（偏振路径从 RGB 路径推断）
- **图像处理流程**：
  1. 从磁盘加载原始尺寸图像
  2. RGB 图像通过 `image_processor.preprocess` 自动 resize 到 336x336（CLIP 输入）
  3. Polar 图像 resize 到 512x512（VAE 输入）

### ✅ 支持的 JSON 数据格式

1. **Stage 1 格式（crop 数据）**：
```json
{
  "image": "rgb_crop/23/0000_rgb.png",
  "scene_id": "23",
  "polar_crop_paths": {
    "I_0": "polar_crop/23/0000_000.png",
    "I_45": "polar_crop/23/0000_045.png",
    "I_90": "polar_crop/23/0000_090.png",
    "I_135": "polar_crop/23/0000_135.png"
  },
  "conversations": [...]
}
```

2. **Stage 2 格式（非裁剪数据）**：
```json
{
  "image": "rgb/04/0002_rgb.png",
  "scene_id": "04",
  "conversations": [...]
}
```
（偏振路径从 RGB 路径推断：`polar/04/0002_000.png` 等）

## ✅ 关键实现细节

### 图像尺寸（已修复）
- **RGB 图像**：336x336（LLaVA v1.5 要求，CLIP-ViT-L-336 输入）
  - ⚠️ **关键修复**：从 224x224 改为 336x336，确保与 LLaVA v1.5 预训练权重匹配
  - RGB 输出：24x24 = 576 tokens（336 / 14 = 24）
  - **RGB 编码器**：`clip-vit-large-patch14-336`（确认）
- **Polar 图像**：512x512（VAE 输入）
  - Polar 输出：24x24 = 576 tokens（下采样后与 RGB 对齐）

### 通道格式
- **Polar 输入**：3 通道 `[DoLP, sin(2*AoLP), cos(2*AoLP)]`
- **数据加载时值范围**：`[0, 1]`（PIL Image → ToTensor）
- **VAE 编码时值范围**：`[-1, 1]`（在 `encode_polar_images` 中转换：`polar_input = polar_images * 2.0 - 1.0`）
  - ⚠️ **关键修复**：确保 VAE 输入归一化到 `[-1, 1]` 范围（SD VAE 标准）
  - **数据一致性**：
    - Sin/Cos：原始 `[-1, 1]` → 存图 `[0, 255]` → 读取 `[0, 1]` → 归一化 `[-1, 1]` ✅
    - DoLP：原始 `[0, 1]` → 存图 `[0, 255]` → 读取 `[0, 1]` → 归一化 `[-1, 1]` ✅（物理上无负值，但 VAE 可接受）

### RGB 特征维度流程（详细）

**输入阶段**：
1. **原始图像**：`(B, 3, H, W)` - 任意尺寸 RGB 图像
2. **图像预处理**（`CLIPImageProcessor`）：
   - Resize 到 `(B, 3, 336, 336)` - LLaVA v1.5 标准输入尺寸
   - 归一化：ImageNet 均值和标准差

**CLIP 编码阶段**：
3. **CLIP-ViT-L-336 Encoder**：
   - 输入：`(B, 3, 336, 336)`
   - Patch 分割：patch_size=14，得到 `(336/14)² = 24² = 576` 个 patches
   - 输出：`(B, 576, 1024)` - 576 个 token，每个 token 1024 维（ViT-L 的 hidden_size）
   - **模块**：`vision_tower`（冻结）

**RGB Projector 投影阶段**：
4. **mm_projector**（LLaVA 原带的 MLP，冻结）：
   - 输入：`(B, 576, 1024)` - CLIP 输出特征
   - 结构：`nn.Linear(1024, hidden_size)` 或 `MLP(1024 → hidden_size)`
   - 输出：`(B, 576, hidden_size)` - 576 个 token，每个 token `hidden_size` 维
   - **hidden_size**：对于 Vicuna-13B 是 **5120**，对于 LLaMA-3-8B 是 **4096**
   - **模块**：`model.mm_projector`（冻结）

**最终 RGB 特征**：
- **形状**：`(B, 576, hidden_size)`
- **含义**：576 个空间位置的视觉 token，每个 token 是 `hidden_size` 维向量

---

### Polar 特征维度流程（详细）

**输入阶段**：
1. **原始偏振图像**：`(B, 3, H, W)` - 3通道 `[DoLP, sin(2*AoLP), cos(2*AoLP)]`，值范围 `[0, 1]`
2. **Resize**（如果需要）：
   - 使用 `F.interpolate`，mode='bilinear'
   - 输出：`(B, 3, 512, 512)` - VAE 标准输入尺寸

**VAE 编码阶段**：
3. **值范围转换**：
   - `polar_input = polar_images * 2.0 - 1.0`
   - 输出：`(B, 3, 512, 512)` - 值范围 `[-1, 1]`（SD VAE 标准）

4. **VAE Encoder**（`polar_encoder`，冻结）：
   - 输入：`(B, 3, 512, 512)` - 归一化后的偏振图像
   - 结构：Stable Diffusion VAE Encoder（多层 ResNet + Attention）
   - 输出：`(B, 512, 64, 64)` - 下采样 8 倍（512/8 = 64）
   - **模块**：`model.polar_encoder`（冻结）

5. **Quant Conv**（`polar_quant_conv`，冻结）：
   - 输入：`(B, 512, 64, 64)`
   - 输出：`(B, 8, 64, 64)` - 8 通道（mean + logvar）
   - **模块**：`model.polar_quant_conv`（冻结）

6. **提取 Latent**：
   - `latent, _ = torch.chunk(moments, 2, dim=1)` - 提取 mean
   - 输出：`(B, 4, 64, 64)` - VAE latent 空间

7. **SD VAE 标准缩放**：
   - `latent = latent * 0.18215`
   - 输出：`(B, 4, 64, 64)` - 缩放后的 latent

**特征投影与下采样阶段**：
8. **VAE Latent → Feature 投影**（`vae_latent_to_feature`，Conv2d）：
   - 输入：`(B, 4, 64, 64)` - VAE latent
   - 结构：`nn.Conv2d(4, 256, kernel_size=1)` - 1x1 卷积
   - 输出：`(B, 256, 64, 64)` - 投影到特征维度 256
   - **模块**：`model.vae_latent_to_feature`

9. **空间下采样**（与 RGB 对齐）：
   - 使用 `F.interpolate`，mode='bilinear'，align_corners=False
   - 输入：`(B, 256, 64, 64)`
   - 输出：`(B, 256, 24, 24)` - 下采样到 24x24（与 RGB 的 24x24 对齐）
   - ✅ **关键**：使用双线性插值，确保空间对齐且不丢失边缘信息

10. **展平为序列**：
    - `polar_features_flat = polar_features_spatial.permute(0, 2, 3, 1).reshape(B, 24*24, 256)`
    - 输出：`(B, 576, 256)` - 576 个 token，每个 token 256 维

**Polar Projector 投影阶段**：
11. **polar_projector**（新建的 MLP，需训练）：
    - 输入：`(B, 576, 256)` - 下采样后的特征
    - 结构：`nn.Sequential(
        nn.Linear(256, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size)
      )`
    - 输出：`(B, 576, hidden_size)` - 576 个 token，每个 token `hidden_size` 维
    - **hidden_size**：对于 Vicuna-13B 是 **5120**，对于 LLaMA-3-8B 是 **4096**
    - **模块**：`model.polar_projector`（**可训练**）

**最终 Polar 特征**：
- **形状**：`(B, 576, hidden_size)`
- **含义**：576 个空间位置的偏振 token，每个 token 是 `hidden_size` 维向量

---

### 特征拼接与对齐

**RGB 和 Polar 特征拼接**：
- **RGB 特征**：`(B, 576, hidden_size)` - 576 tokens
- **Polar 特征**：`(B, 576, hidden_size)` - 576 tokens
- **拼接**：`torch.cat([rgb_features, polar_features], dim=1)`
- **最终输出**：`(B, 1152, hidden_size)` - 1152 个视觉 token（576 RGB + 576 Polar）

**空间对齐确认**：
- ✅ RGB：336x336 输入 → CLIP 输出 24x24 = 576 tokens
- ✅ Polar：512x512 输入 → VAE 输出 64x64 → 下采样到 24x24 = 576 tokens
- ✅ **完美对齐**：两者都是 576 tokens，空间分辨率都是 24x24

**维度一致性**：
- ✅ RGB 和 Polar 都投影到相同的 `hidden_size`（5120 或 4096）
- ✅ 拼接后的 1152 tokens 可以直接输入 LLM
- ✅ LLM context window（4096）足够容纳 1152 tokens + 文本 tokens

---

### ⚠️ 特征值范围分析（拼接前）

**RGB 特征值范围**：
1. **CLIP 输出**：`(B, 576, 1024)`
   - CLIP-ViT-L 的输出经过 LayerNorm，值范围通常在 `[-10, 10]` 左右
   - 标准差通常在 `[0.5, 2.0]` 范围内
   - 均值接近 0（LayerNorm 后）

2. **mm_projector 输出**：`(B, 576, hidden_size)`
   - 经过 Linear/MLP 投影
   - 值范围取决于权重初始化（通常 Xavier/Kaiming 初始化）
   - 典型范围：`[-5, 5]` 到 `[-20, 20]`（取决于权重尺度）
   - **无额外归一化**：直接输出到拼接

**Polar 特征值范围**：
1. **VAE Latent**：`(B, 4, 64, 64)`
   - 经过 `× 0.18215` 缩放（SD VAE 标准）
   - 值范围通常在 `[-1, 1]` 左右（缩放后的 latent 空间）

2. **vae_latent_to_feature 输出**：`(B, 256, 64, 64)`
   - Conv2d(4, 256, kernel_size=1) 投影
   - 值范围取决于权重初始化
   - 典型范围：`[-2, 2]` 到 `[-5, 5]`

3. **下采样后**：`(B, 256, 24, 24)`
   - 双线性插值不改变值范围
   - 值范围保持：`[-2, 2]` 到 `[-5, 5]`

4. **polar_projector 输出**：`(B, 576, hidden_size)`
   - 经过 MLP：`Linear(256 → hidden_size) → GELU → Linear(hidden_size → hidden_size)`
   - GELU 激活函数会将负值压缩，正值放大
   - 值范围：`[-10, 20]` 左右（取决于权重初始化）
   - **无额外归一化**：直接输出到拼接

**值范围对比**：
- **RGB 特征**：`[-5, 5]` 到 `[-20, 20]`（取决于 mm_projector 权重）
- **Polar 特征**：`[-10, 20]` 左右（经过 GELU 激活）
- **差异**：两者值范围可能不同，但都在合理范围内

**是否可以直接拼接？**
- ✅ **可以**：虽然值范围可能略有差异，但：
  1. **两者都经过 MLP 投影**：都投影到相同的 `hidden_size`，权重初始化会控制输出尺度
  2. **训练过程中自适应**：模型会在训练过程中学习适应这种差异
  3. **LLM 的鲁棒性**：Transformer 的 LayerNorm 和 Attention 机制对输入尺度有一定鲁棒性
  4. **实际效果**：在 LLaVA 等模型中，直接拼接多模态特征是常见做法

**潜在优化（可选）**：
- 如果训练不稳定，可以考虑：
  1. **LayerNorm**：在拼接前对 RGB 和 Polar 特征分别进行 LayerNorm
  2. **缩放平衡**：添加可学习的缩放参数（类似之前的 `rgb_scale` 和 `polar_scale`）
  3. **统一归一化**：在拼接后对整个特征进行 LayerNorm

**当前实现**：
- ⚠️ **直接拼接**：代码中直接使用 `torch.cat([rgb_features, polar_features], dim=1)`
- **原因**：LLaVA 原架构也是直接拼接，且训练过程中模型会自适应
- **建议**：先使用直接拼接，如果训练不稳定再考虑添加归一化

## 🔍 关键检查点确认

### ✅ 1. VAE 特征下采样的数学逻辑
- **实现**：使用 `torch.nn.functional.interpolate`，mode='bilinear'，align_corners=False
- **位置**：`llava/model/llava_arch.py` 第 509-514 行
- **验证**：64x64 → 24x24 使用双线性插值，安全且不会丢失边缘信息 ✅

### ✅ 2. LoRA 模块的覆盖范围
- **实现**：`find_all_linear_names` 函数排除 `polar_projector` 从 LoRA target_modules
- **位置**：`llava/train/train.py` 第 201-214 行
- **验证**：Polar Projector 不会被 LoRA 包裹，全量微调 ✅
- **确认**：LoRA 只作用于 LLM 的 `q_proj`、`v_proj` 等线性层 ✅

### ✅ 3. 偏振图像数据的预处理逻辑
- **数据一致性**：已确认 ✅
  - Sin/Cos：`[-1, 1]` → PNG `[0, 255]` → 读取 `[0, 1]` → 归一化 `[-1, 1]` ✅
  - DoLP：`[0, 1]` → PNG `[0, 255]` → 读取 `[0, 1]` → 归一化 `[-1, 1]` ✅

### ✅ 4. RGB 编码器确认
- **编码器**：`clip-vit-large-patch14-336` ✅
- **输入尺寸**：336x336（自动设置）✅
- **Stage 1 图像处理**：512x512 crop → resize 到 336x336 ✅

### ✅ 5. 显存优化与量化确认
- **4-bit 量化支持**：代码已支持 `--bits 4` 参数 ✅
- **Polar Projector 精度保护**：
  - `polar_projector` 不在 LLM 内部，而是在 `PolarLlavaMetaModel` 中，不会被 BitsAndBytesConfig 量化
  - 代码中明确将 `polar_projector` 转换为 `compute_dtype`（BF16），确保保持 BF16 精度（第 1343-1344 行）
  - **验证**：Polar Projector 不会被量化，保持 BF16 精度 ✅
- **序列长度**：RGB+Polar 拼接后 1152 tokens，13B 模型 context window 4096，完全足够 ✅
- **验证集确认**：请确认 `val_stage2_data.json` 和 `val_stage3_qwen.json` 文件存在，否则请移除相关参数 ✅

## 🎯 完整训练命令

### ⚠️ 显存优化说明（RTX 4090 24GB）

**必须使用 4-bit 量化（QLoRA）**：
- 13B 模型以 BF16/FP16 加载需要约 26-28GB 显存，超过 24GB 限制
- 使用 `--bits 4` 后，LLM 以 4-bit 加载（约 7-8GB 显存）
- CLIP、VAE、Projector 保持 16-bit（BF16）
- **Polar Projector 不会被量化**：代码中已确保 `polar_projector` 保持 BF16 精度（第 1343-1344 行）

**Batch Size 调整**：
- Stage 1: `per_device_train_batch_size=2`, `gradient_accumulation_steps=16`（总 BS=32）
- Stage 2: `per_device_train_batch_size=1`, `gradient_accumulation_steps=32`（总 BS=32）
- 原理：用时间换空间，减小单次并行量，多累积梯度再更新

### Stage 1: Polar Projector Alignment

```bash
python -m llava.train.train \
  --model_name_or_path /openbayes/input/input0/models/llava-v1.5-13b \
  --vision_tower /openbayes/input/input0/models/clip-vit-large-patch14-336 \
  --polar_vae_model_path /openbayes/input/input0/models/sd-vae-ft-mse \
  --training_stage stage1 \
  --freeze_polar_encoder True \
  --freeze_rgb_tower True \
  --freeze_rgb_projector True \
  --data_path train_stage2_data.json \
  --val_json val_stage2_data.json \
  --image_folder /openbayes/input/input0/rgb_crop \
  --polar_folder /openbayes/input/input0/polar_crop \
  --data_root /openbayes/input/input0 \
  --use_polar True \
  --output_dir /openbayes/input/input0/1.24/stage1 \
  --bits 4 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 5 \
  --learning_rate 5e-4 \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --logging_steps 10 \
  --save_steps 200 \
  --eval_steps 50 \
  --save_total_limit 2 \
  --load_best_model_at_end True \
  --bf16 True \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --remove_unused_columns False \
  --report_to tensorboard \
  --save_strategy steps \
  --eval_strategy steps
```

**关键参数说明**：
- `--training_stage stage1`: 只训练 Polar Projector
- `--image_folder`: 使用 **rgb_crop** 目录（crop 数据，512x512）
- `--polar_folder`: 使用 **polar_crop** 目录（crop 数据，512x512）
- `--data_root`: 用于解析 JSON 中的 `rgb_crop/` 和 `polar_crop/` 路径
- `--output_dir`: `/openbayes/input/input0/1.24/stage1`

**输出**：
- `mm_projector.bin`: 包含 `polar_projector` 权重（用于 Stage 2）

### Stage 2: Visual Instruction Tuning

```bash
python -m llava.train.train \
  --model_name_or_path /openbayes/input/input0/models/llava-v1.5-13b \
  --vision_tower /openbayes/input/input0/models/clip-vit-large-patch14-336 \
  --polar_vae_model_path /openbayes/input/input0/models/sd-vae-ft-mse \
  --pretrain_polar_projector /openbayes/input/input0/1.24/stage1/mm_projector.bin \
  --training_stage stage2 \
  --freeze_polar_encoder True \
  --freeze_rgb_tower True \
  --freeze_rgb_projector True \
  --lora_enable True \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --lora_bias none \
  --data_path merged_stage3_qwen.json \
  --val_json val_stage3_qwen.json \
  --image_folder /openbayes/input/input0/rgb \
  --polar_folder /openbayes/input/input0/polar \
  --data_root /openbayes/input/input0 \
  --use_polar True \
  --output_dir /openbayes/input/input0/1.24/stage2 \
  --bits 4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --num_train_epochs 10 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.1 \
  --weight_decay 0.05 \
  --logging_steps 5 \
  --save_steps 100 \
  --eval_steps 100 \
  --save_total_limit 2 \
  --load_best_model_at_end True \
  --bf16 True \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --remove_unused_columns False \
  --report_to tensorboard \
  --save_strategy steps \
  --eval_strategy steps
```

**关键参数说明**：
- `--pretrain_polar_projector`: **必须提供** Stage 1 训练的权重路径
- `--training_stage stage2`: 训练 Polar Projector + LLM (LoRA)
- `--image_folder`: 使用 **rgb** 目录（非裁剪数据，原始尺寸）
- `--polar_folder`: 使用 **polar** 目录（非裁剪数据，原始尺寸）
- `--data_root`: 用于解析可能的 crop 路径（如果 JSON 中有）
- `--output_dir`: `/openbayes/input/input0/1.24/stage2`
- `--lora_enable True`: 启用 LoRA（只作用于 LLM，不作用于 Polar Projector）

**输出**：
- `mm_projector.bin`: 包含 `polar_projector` 权重（更新后的）
- LoRA 适配器：`adapter_model.safetensors`、`adapter_config.json`

## 📝 修改过的代码文件清单

### 核心模型文件

1. **`llava/model/llava_arch.py`**
   - **修改内容**：
     - 添加 `PolarLlavaMetaModel` 类（继承自 `LlavaMetaModel`）
     - 实现 `_initialize_polar_modules`：加载 VAE 编码器，创建 Polar Projector
     - 实现 `encode_polar_images`：VAE 编码 → 下采样（64x64 → 24x24）→ 投影
     - 修改 `encode_images`：支持 `polar_images` 参数，拼接 RGB 和 Polar 特征
   - **作用阶段**：Stage 1 和 Stage 2 都使用

2. **`llava/model/language_model/llava_llama.py`**
   - **修改内容**：
     - 添加 `PolarLlavaConfig` 类（继承自 `LlavaConfig`）
     - 添加 `PolarLlavaLlamaModel` 类（继承自 `PolarLlavaMetaModel` 和 `LlamaModel`）
     - 添加 `PolarLlavaLlamaForCausalLM` 类（继承自 `LlamaForCausalLM` 和 `LlavaMetaForCausalLM`）
     - 修改 `forward`、`generate`、`prepare_inputs_labels_for_multimodal`：支持 `polar_images` 参数
     - 注册 `PolarLlavaConfig` 和 `PolarLlavaLlamaForCausalLM` 到 AutoConfig/AutoModel
   - **作用阶段**：Stage 1 和 Stage 2 都使用

3. **`llava/model/builder.py`**
   - **修改内容**：
     - 修改 `load_pretrained_model`：检测 `polar_vae_model_path`，加载 `PolarLlavaLlamaForCausalLM`
   - **作用阶段**：Stage 1 和 Stage 2 都使用

### 训练相关文件

4. **`llava/train/train.py`**
   - **修改内容**：
     - **ModelArguments**：添加 `polar_vae_model_path`、`freeze_polar_encoder`、`freeze_rgb_tower`、`freeze_rgb_projector`、`training_stage`、`pretrain_polar_projector`
     - **DataArguments**：添加 `polar_folder`、`use_polar`、`data_root`
     - **LazySupervisedDataset.__getitem__**：加载偏振图像
     - **LazySupervisedDataset._load_polar_image**：处理 4 张角度图像 → 3 通道 → 512x512
     - **DataCollatorForSupervisedDataset.__call__**：堆叠 `polar_images`
     - **train 函数**：
       - 模型初始化：检测 `polar_vae_model_path`，创建 `PolarLlavaLlamaForCausalLM`
       - RGB 图像尺寸修复：自动设置为 336x336
       - Stage 1 权重加载：加载 `pretrain_polar_projector`
       - 冻结/解冻逻辑：根据 `training_stage` 设置参数状态
     - **find_all_linear_names**：排除 `polar_projector` 从 LoRA target_modules
   - **作用阶段**：
     - 数据加载：Stage 1 和 Stage 2 都使用
     - 模型初始化：Stage 1 和 Stage 2 都使用
     - 冻结/解冻逻辑：根据 `training_stage` 分别作用于 Stage 1 或 Stage 2
     - LoRA 配置：Stage 2 使用

5. **`llava/train/llava_trainer.py`**
   - **修改内容**：
     - 修改 `_save_checkpoint`：在 `keys_to_match` 中添加 `polar_projector`
   - **作用阶段**：Stage 1 和 Stage 2 都使用

## ✅ 关键修复总结

### 🔴 已修复的关键隐患

1. **✅ RGB 分辨率不匹配（Critical）**
   - **问题**：原设计使用 224x224，但 LLaVA v1.5 需要 336x336
   - **修复**：在 `train.py` 中自动检测并设置 `image_processor` 为 336x336
   - **验证**：RGB 输出 576 tokens (24x24)，与 Polar 的 576 tokens 完美对齐

2. **✅ VAE 输入归一化范围（Normalization）**
   - **问题**：SD VAE 需要 `[-1, 1]` 范围输入
   - **修复**：在 `encode_polar_images` 中转换：`polar_input = polar_images * 2.0 - 1.0`
   - **验证**：数据加载返回 `[0, 1]`，编码时转换为 `[-1, 1]`

3. **✅ Stage 2 加载 Stage 1 权重**
   - **问题**：Stage 2 需要加载 Stage 1 训练的 Polar Projector 权重
   - **修复**：添加 `--pretrain_polar_projector` 参数，支持加载 `.bin`、`.pth` 或目录路径
   - **验证**：自动提取 `polar_projector` 权重并加载

### 🟡 已确认的建议检查点

4. **✅ Prompt 模板与 Token 拼接**
   - **逻辑**：采用逻辑 A（拼接成一个大 Tensor）
   - **实现**：`encode_images` 返回 `torch.cat([rgb_features, polar_features], dim=1)` → 1152 tokens
   - **验证**：数据集只需要一个 `<image>` token，会被替换为拼接后的 1152 tokens

5. **✅ 显存优化（Gradient Checkpointing）**
   - **状态**：已在 `TrainingArguments` 中默认开启 `gradient_checkpointing=True`
   - **验证**：LLaVA 的 `PolarLlavaMetaModel` 继承自 `LlavaMetaModel`，自动支持梯度检查点

6. **✅ VAE 特征下采样（关键检查点 1）**
   - **实现**：使用 `F.interpolate`，mode='bilinear'，align_corners=False
   - **验证**：64x64 → 24x24 使用双线性插值，安全且不会丢失边缘信息 ✅

7. **✅ LoRA 模块覆盖范围（关键检查点 2）**
   - **实现**：`find_all_linear_names` 排除 `polar_projector`
   - **验证**：Polar Projector 不会被 LoRA 包裹，全量微调 ✅

8. **✅ 偏振图像预处理逻辑（关键检查点 3）**
   - **验证**：数据一致性已确认，Sin/Cos 和 DoLP 的映射关系正确 ✅

## ✅ 总结

所有关键模块已实现并修复：
1. ✅ **模型架构**：PolarLlavaMetaModel、PolarLlavaLlamaForCausalLM
2. ✅ **数据加载**：支持 Stage 1（crop）和 Stage 2（非裁剪）数据格式
3. ✅ **训练控制**：两阶段训练逻辑、参数冻结/解冻、Stage 1 权重加载
4. ✅ **权重保存**：Polar Projector 权重会被正确保存
5. ✅ **关键修复**：RGB 336x336、VAE 归一化 `[-1, 1]`、Stage 1 权重加载、VAE 下采样、LoRA 配置
6. ✅ **显存优化**：4-bit 量化（QLoRA）、Batch Size 调整、Polar Projector 精度保护

### ⚠️ 训练前最后检查清单

- [ ] 确认验证集文件存在（`val_stage2_data.json`、`val_stage3_qwen.json`），否则移除相关参数
- [ ] 确认使用 `--bits 4` 参数（RTX 4090 24GB 必须）
- [ ] 确认 Batch Size 已调整（Stage 1: BS=2, GA=16; Stage 2: BS=1, GA=32）
- [ ] 确认 Stage 1 输出路径存在，Stage 2 的 `--pretrain_polar_projector` 路径正确

代码已准备好进行训练测试！

---

## 📝 数据路径总结

### Stage 1 训练数据路径
- **JSON 文件**: `train_stage2_data.json`, `val_stage2_data.json`
- **数据根目录**: `/openbayes/input/input0`
- **RGB 图像**: `/openbayes/input/input0/rgb_crop`（**带 crop**，512x512 裁剪图像）
- **Polar 图像**: `/openbayes/input/input0/polar_crop`（**带 crop**，512x512 裁剪图像）
- **JSON 格式**: 包含 `polar_crop_paths` 字段，如 `polar_crop/23/0000_000.png`

### Stage 2 训练数据路径
- **JSON 文件**: `merged_stage3_qwen.json`, `val_stage3_qwen.json`
- **数据根目录**: `/openbayes/input/input0`
- **RGB 图像**: `/openbayes/input/input0/rgb`（**不带 crop**，原始尺寸图像）
- **Polar 图像**: `/openbayes/input/input0/polar`（**不带 crop**，原始尺寸图像）
- **JSON 格式**: 包含 `input_path` 字段，如 `rgb/04/0002_rgb.png`（偏振路径从 RGB 路径推断）

### 模型路径
- **LLaVA**: `/openbayes/input/input0/models/llava-v1.5-13b`
- **CLIP**: `/openbayes/input/input0/models/clip-vit-large-patch14-336`
- **VAE**: `/openbayes/input/input0/models/sd-vae-ft-mse`

### 输出路径
- **Stage 1**: `/openbayes/input/input0/1.24/stage1`
- **Stage 2**: `/openbayes/input/input0/1.24/stage2`

# Stage 2 训练代码检查报告

## 📋 检查项

### ✅ 1. 加载第一阶段结果

**路径处理**：
- ✅ 支持完整路径到 `.bin` 文件：`/openbayes/input/input0/llava_train_stage1_ultra6/checkpoint-1500/non_lora_trainables.bin`
- ✅ 代码在 `train.py:2762` 和 `train.py:2852` 会检查 `endswith('.bin')`，然后直接加载
- ✅ 如果是目录路径，会查找 `non_lora_trainables.bin` 或 `mm_projector.bin`

**加载内容**：
- ✅ `polar_projector` 权重（`train.py:2777-2838`）
- ✅ `vae_latent_to_feature` 权重（`train.py:2843-2895`）
- ⚠️ **Polar token embedding**：由于 Stage 1 不添加任何 token（词表大小 32000），所以不会从 Stage 1 加载 token embedding
  - 这是**正确的行为**，因为 Stage 1 没有训练过这些 token
  - Stage 2 会从头训练所有4个分隔 token

### ✅ 2. 训练什么

**Stage 2 训练内容**（`train.py:2955-3024`）：
- ✅ **Polar Projector**：可训练（从 Stage 1 加载权重后继续训练）
- ✅ **vae_latent_to_feature**：可训练（从 Stage 1 加载权重后继续训练）
- ✅ **LLM (LoRA)**：可训练（通过 LoRA 适配器）
- ✅ **4个分隔 token embedding**：可训练（`<RGB_START>`, `<RGB_END>`, `<POL_START>`, `<POL_END>`）
  - 在 `train.py:3105` 和 `train.py:3112` 会启用 `input_embeddings` 和 `output_embeddings` 的梯度

**冻结内容**：
- ✅ **RGB Vision Tower**：冻结（`train.py:2964-2966`）
- ✅ **RGB Projector**：冻结（`train.py:2967-2969`）
- ✅ **Polar Encoder**：默认冻结（可通过 `--freeze_polar_encoder False` 解冻）

### ✅ 3. 保存什么

**最终保存内容**（`train.py:3308-3447`）：
- ✅ **LoRA 权重**：保存到 `adapter_model.bin` 和 `adapter_config.json`
- ✅ **non_lora_trainables.bin**：包含
  - `polar_projector` 权重（`train.py:3323-3333`）
  - `vae_latent_to_feature` 权重（`train.py:3323-3333`）
  - `embed_tokens` 权重（如果词表被 resize，`train.py:3382-3395`）
  - `lm_head` 权重（如果词表被 resize，`train.py:3382-3395`）

**Checkpoint 保存**（`llava_trainer.py:534-574`）：
- ✅ 每个 checkpoint 都会保存 `non_lora_trainables.bin`（包含上述所有内容）

### ✅ 4. 四个分隔 token

**添加 token**（`train.py:2175-2182`）：
- ✅ Stage 2 会添加所有4个 token：`["<RGB_START>", "<RGB_END>", "<POL_START>", "<POL_END>"]`
- ✅ Token ID 会保存到 `model.config`（`train.py:2484-2492`）：
  - `rgb_start_id`, `rgb_end_id`
  - `pol_start_id`, `pol_end_id`

**训练 token embedding**（`train.py:3096-3114`）：
- ✅ Stage 2 会启用 `input_embeddings` 和 `output_embeddings` 的梯度
- ✅ 所有4个 token 的 embedding 都会被训练

**使用分隔 token**（`llava_arch.py:998-1048`）：
- ✅ Stage 2 会使用分隔 token 包装图像特征：
  - `[RGB_START] + [RGB Features] + [RGB_END] + [POL_START] + [Pol Features] + [POL_END]`
- ✅ 分隔 token 的 embedding 会被插入到输入序列中（`llava_arch.py:1006-1017`）
- ✅ 分隔 token 对应的 labels 会被设置为 `IGNORE_INDEX`（`llava_arch.py:1030-1035`）

### ✅ 5. RGB 和 Polar 一起输入

**编码逻辑**（`llava_arch.py:683-822`）：
- ✅ Stage 2 会同时编码 RGB 和 Polar 图像
- ✅ RGB 特征：通过 CLIP Vision Tower + RGB Projector（`llava_arch.py:757-758`）
- ✅ Polar 特征：通过 VAE Encoder + Polar Projector（`llava_arch.py:770-771`）
- ✅ 拼接：`torch.cat([rgb_features, polar_features], dim=1)`（`llava_arch.py:783`）
- ✅ 输出形状：`(B, 1152, hidden_size)` = 576 (RGB) + 576 (Polar)

**输入格式**（`llava_arch.py:998-1048`）：
- ✅ 如果 `image_features` 长度是 1152，会拆分为：
  - RGB features: `cur_image_features[:576]`
  - Polar features: `cur_image_features[576:]`
- ✅ 然后用分隔 token 包装：
  ```
  [RGB_START] + [RGB Features] + [RGB_END] + [POL_START] + [Pol Features] + [POL_END]
  ```

## 🔧 已修复的问题

### 1. Stage 1 Polar token embedding 加载逻辑

**问题**：代码假设 Stage 1 的词表大小是 32002（32000 + 2个Polar token），但实际上 Stage 1 不添加任何 token，所以词表大小应该是 32000。

**修复**：
- 在 `train.py:2550-2556` 添加了对 `stage1_vocab_size == 32000` 的检查
- 如果 Stage 1 词表大小是 32000，会跳过加载 Polar token embedding（因为 Stage 1 没有训练过）
- 如果 Stage 1 词表大小是 32002（旧版本可能的情况），仍然会尝试加载

## 📝 使用建议

### Stage 2 训练命令示例

```bash
torchrun --nproc_per_node=2 \
    LLaVA/llava/train/train.py \
    --model_name_or_path /autodl-fs/data/models/llava-v1.5-13b \
    --version v1 \
    --vision_tower /autodl-fs/data/models/clip-vit-large-patch14-336 \
    --polar_vae_model_path /autodl-fs/data/models/sd-vae-ft-mse \
    --pretrain_polar_projector /openbayes/input/input0/llava_train_stage1_ultra6/checkpoint-1500/non_lora_trainables.bin \
    --training_stage stage2 \
    --freeze_polar_encoder True \
    --freeze_rgb_tower True \
    --freeze_rgb_projector True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --bits 4 \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules_to_save "embed_tokens,lm_head" \
    --mm_projector_lr 2e-3 \
    --data_path 数据json/train_stage2_qwen.json \
    --val_json 数据json/val_stage2_qwen.json \
    --image_folder /openbayes/input/input0/rgb \
    --polar_folder /openbayes/input/input0/polar \
    --data_root /openbayes/input/input0 \
    --use_polar True \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --ddp_find_unused_parameters False \
    --dataloader_pin_memory False \
    --output_dir /openbayes/input/input0/llava_train_stage2_ultra1 \
    --num_train_epochs 3
```

### 关键参数说明

- `--pretrain_polar_projector`：Stage 1 的 `non_lora_trainables.bin` 路径（完整路径或目录路径）
- `--training_stage stage2`：指定为 Stage 2 训练
- `--lora_modules_to_save "embed_tokens,lm_head"`：确保4个分隔 token 的 embedding 可训练
- `--use_polar True`：启用 Polar 图像输入
- `--model_max_length 2048`：足够容纳 1152 个图像 tokens + 文本

## ✅ 总结

代码检查结果：**所有功能都正确实现**

1. ✅ **加载 Stage 1 结果**：会正确加载 `polar_projector` 和 `vae_latent_to_feature` 权重
2. ✅ **训练内容**：Polar Projector + LLM LoRA + 4个分隔 token embedding
3. ✅ **保存内容**：所有训练的参数都会被保存
4. ✅ **四个分隔 token**：会被添加、训练和使用
5. ✅ **RGB + Polar 输入**：会同时编码并拼接（1152 tokens）

**唯一需要注意的点**：
- Stage 1 没有训练过任何分隔 token，所以 Stage 2 会从头训练所有4个 token
- 这是**正确的行为**，因为 Stage 1 的目标是训练 Polar Projector，而不是 token embedding

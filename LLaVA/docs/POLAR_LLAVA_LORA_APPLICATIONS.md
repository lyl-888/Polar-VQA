# LLaVA 1.5 官方 LoRA 文档对 Polar LLaVA 项目的应用分析

## 📋 概述

本文档分析 LLaVA 1.5 官方 LoRA 文档（`docs/LoRA.md`）中哪些内容可以应用到 Polar LLaVA 项目中。

## ✅ 已实现的功能

### 1. LoRA 训练支持
- **状态**: ✅ 已实现
- **位置**: `LLaVA/llava/train/train.py` (Stage 2 训练)
- **配置**: 
  - `lora_r=64`, `lora_alpha=128`, `lora_dropout=0.05`
  - 与官方推荐配置一致
- **说明**: Polar LLaVA 在 Stage 2 使用 LoRA 训练 LLM，同时训练 `polar_projector`

### 2. LoRA 权重合并脚本
- **状态**: ✅ 已存在，但需要验证 Polar 支持
- **位置**: `LLaVA/scripts/merge_lora_weights.py`
- **当前实现**: 
  - 使用 `load_pretrained_model` 加载模型
  - `load_pretrained_model` 已支持 `polar_vae_model_path` 参数
- **潜在问题**: 
  - 合并后需要确保 `polar_projector` 权重也被保存
  - 需要验证 `model.save_pretrained()` 是否包含 `polar_projector`

### 3. DeepSpeed 配置文件
- **状态**: ✅ 已存在
- **位置**: 
  - `LLaVA/scripts/zero2.json`
  - `LLaVA/scripts/zero3.json`
  - `LLaVA/scripts/zero3_offload.json`
- **当前使用**: ❌ 训练命令中未使用
- **建议**: 
  - 单卡 4090 (24GB) 通常不需要 DeepSpeed
  - 如果显存紧张，可以考虑 `zero3_offload.json`（CPU offload）

## 🔧 可以改进的地方

### 1. LoRA 合并脚本增强

**问题**: 当前的 `merge_lora_weights.py` 可能不会保存 `polar_projector` 权重

**建议修改**:
```python
def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, 
        args.model_base, 
        model_name, 
        device_map='cpu',
        polar_vae_model_path=args.polar_vae_model_path  # 添加 Polar 支持
    )

    # 保存合并后的模型
    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)
    
    # [新增] 保存 polar_projector 权重（如果存在）
    if hasattr(model, 'get_model') and hasattr(model.get_model(), 'polar_projector'):
        polar_projector = model.get_model().polar_projector
        polar_projector_path = os.path.join(args.save_model_path, 'polar_projector.bin')
        torch.save(polar_projector.state_dict(), polar_projector_path)
        print(f"✓ Polar Projector 权重已保存: {polar_projector_path}")
    
    # [新增] 保存 mm_projector 权重（包含 polar_projector）
    if hasattr(model, 'get_model') and hasattr(model.get_model(), 'mm_projector'):
        mm_projector = model.get_model().mm_projector
        mm_projector_path = os.path.join(args.save_model_path, 'mm_projector.bin')
        torch.save(mm_projector.state_dict(), mm_projector_path)
        print(f"✓ MM Projector 权重已保存: {mm_projector_path}")
```

### 2. DeepSpeed 配置使用（可选）

**适用场景**: 
- 显存不足时（虽然单卡 4090 通常不需要）
- 多卡训练时（如果未来扩展到多卡）

**使用方法**:
```bash
deepspeed llava/train/train.py \
    --deepspeed ./scripts/zero3_offload.json \
    # ... 其他参数
```

**注意事项**:
- `zero3_offload.json`: 最省显存，但速度较慢（CPU offload）
- `zero3.json`: 较快，但需要更多显存
- `zero2.json`: 介于两者之间

### 3. 训练参数优化参考

**官方 QLoRA 脚本参数** (`finetune_qlora.sh`):
- `--lora_r 128 --lora_alpha 256`: 官方使用更大的 rank
- `--mm_projector_lr 2e-5`: 投影器单独的学习率
- `--model_max_length 2048`: 序列长度
- `--gradient_checkpointing True`: 梯度检查点（但 Polar LLaVA 已禁用）

**当前 Polar LLaVA 配置**:
- `--lora_r 64 --lora_alpha 128`: 较小的 rank（适合小数据集）
- `--model_max_length 2048`: 已设置（推荐 4096 用于 Polar）
- `--gradient_checkpointing False`: 已禁用（解决 grad_norm=0 问题）

**建议**: 当前配置适合 Polar LLaVA，无需修改

## 📝 官方文档中的其他要点

### 1. Demo/推理支持
- **官方**: 使用 `--model-base` 参数指定基础模型
- **Polar LLaVA**: 推理代码已支持 LoRA 模型加载
- **位置**: `LLaVA/llava/model/builder.py` 中的 `load_pretrained_model` 函数

### 2. 训练脚本参考
- **官方脚本**: `scripts/finetune_lora.sh`, `scripts/finetune_qlora.sh`
- **Polar LLaVA**: 使用 `train.py`，参数更灵活
- **建议**: 可以参考官方脚本的参数组合，但当前实现已足够

## 🎯 总结

### 可以直接使用的功能
1. ✅ LoRA 训练（已实现）
2. ✅ LoRA 合并脚本（已存在，可能需要增强）
3. ✅ DeepSpeed 配置（已存在，但未使用）

### 建议改进
1. 🔧 增强 `merge_lora_weights.py` 以支持 `polar_projector` 权重保存
2. 📚 添加 DeepSpeed 使用说明（如果需要）
3. ✅ 验证合并后的模型是否包含所有必要权重

### 不需要修改的部分
1. ✅ LoRA 参数配置（当前配置合理）
2. ✅ 训练流程（已适配 Polar LLaVA）
3. ✅ 模型加载逻辑（已支持 Polar）

## 🔍 验证清单

- [ ] 测试 `merge_lora_weights.py` 是否能正确保存 `polar_projector` 权重
- [ ] 验证合并后的模型能否正常加载和推理
- [ ] （可选）测试 DeepSpeed 配置在显存不足时的效果

## 📚 相关文件

- 官方文档: `LLaVA/docs/LoRA.md`
- 合并脚本: `LLaVA/scripts/merge_lora_weights.py`
- 模型加载: `LLaVA/llava/model/builder.py`
- 训练脚本: `LLaVA/llava/train/train.py`
- DeepSpeed 配置: `LLaVA/scripts/zero*.json`

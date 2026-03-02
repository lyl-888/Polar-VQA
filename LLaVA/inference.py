"""
PolarVLM Stage 2 推理脚本（适配 LLaVA 1.5 架构）
用于测试训练后的 Stage 2 模型

[LLaVA 1.5 架构适配]：
- 使用 PolarLlavaLlamaForCausalLM（基于 LLaVA 1.5）
- 偏振编码器：使用 VAE 编码器（通过 vae_model_path 参数）
- 偏振输入：3 通道（DoLP, sin(2*AoLP), cos(2*AoLP)），去掉 Intensity
- 图像尺寸：
  - RGB: 336x336（LLaVA 1.5 CLIP 需要）
  - Polar: 512x512（VAE 编码器需要）
"""

import os
import sys
import torch
import argparse
import json
import warnings
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
from peft import PeftModel
import torchvision.transforms as transforms
import numpy as np
from tqdm.auto import tqdm

# 添加 LLaVA 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入 LLaVA 模型加载函数
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava import conversation as conversation_lib

# 导入共享的处理函数
try:
    from dataset_common import process_polar_images
except ImportError:
    # 如果在 LLaVA 目录下运行，需要从父目录导入
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from dataset_common import process_polar_images

# [修复] 抑制 PEFT 和 torch.load 的警告信息
warnings.filterwarnings("ignore", category=UserWarning, module="peft")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only.*")

# [修复] 禁用 transformers 的 torch.load 安全检查（与训练代码一致）
# 新版本 transformers 要求 torch >= 2.6 才能使用 torch.load，但我们使用的是本地模型文件，可以安全禁用
try:
    import transformers

    def _noop_safety_check():
        pass

    # 在模块级别禁用检查
    transformers.utils.import_utils.check_torch_load_is_safe = _noop_safety_check
    # 如果 modeling_utils 中有直接引用，也 patch 它
    if hasattr(transformers.modeling_utils, 'check_torch_load_is_safe'):
        transformers.modeling_utils.check_torch_load_is_safe = _noop_safety_check
    # 检查 modeling_utils 的命名空间
    import inspect
    if 'check_torch_load_is_safe' in transformers.modeling_utils.__dict__:
        transformers.modeling_utils.__dict__['check_torch_load_is_safe'] = _noop_safety_check
    # 也 patch load_state_dict 函数，确保在调用时检查被禁用
    _original_load = transformers.modeling_utils.load_state_dict

    def _patched_load(checkpoint_file, *args, **kwargs):
        # 确保检查函数被禁用
        transformers.utils.import_utils.check_torch_load_is_safe = _noop_safety_check
        # 如果 modeling_utils 中有引用，也禁用
        if hasattr(transformers.modeling_utils, 'check_torch_load_is_safe'):
            transformers.modeling_utils.check_torch_load_is_safe = _noop_safety_check
        return _original_load(checkpoint_file, *args, **kwargs)

    transformers.modeling_utils.load_state_dict = _patched_load
    print("✓ Disabled torch.load safety check for model loading")

    # 🟢 额外：降低 transformers 日志等级，屏蔽权重未使用等噪音提示
    try:
        transformers.utils.logging.set_verbosity_error()
        # 进一步屏蔽 modeling_utils 的 INFO/WARNING（包含 “Some weights ... were not used”）
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        logging.getLogger("transformers.modeling_utils").propagate = False
    except Exception:
        pass
except Exception as e:
    print(f"Warning: Failed to disable safety check: {e}")


# 🟢 运行时精度转换包装器：
# 确保 VAE(FP32) 输出在进入 FP16 的 vae_latent_to_feature 之前安全地转换为 FP16，
# 避免 FP32 -> FP16 隐式截断带来的数值崩塌。
class FP32toFP16Wrapper(torch.nn.Module):
    def __init__(self, original_module: torch.nn.Module):
        super().__init__()
        self.original_module = original_module
        # 确保内部权重本身就是 FP16（与 LLM 对齐）
        self.original_module.to(dtype=torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 此时来自 VAE，是 FP32；这里显式转换为 FP16，作为“气闸层”
        x = x.to(dtype=torch.float16)
        return self.original_module(x)


def detect_training_stage(checkpoint_dir: str) -> str:
    """
    自动检测训练阶段（Stage 1 或 Stage 2）
    
    检测逻辑：
    1. 从 checkpoint 目录名中检测（包含 "stage1" 或 "stage2"）
    2. 从 model_config.json 中读取（如果存在）
    3. 从 non_lora_trainables.bin 的键名中检测（Stage 2 通常包含 LoRA 相关键）
    4. 默认返回 "stage2"（向后兼容）
    
    Args:
        checkpoint_dir: 检查点目录路径
    
    Returns:
        "stage1" 或 "stage2"
    """
    checkpoint_path = Path(checkpoint_dir)
    
    # 方法1: 从目录名检测
    dir_name_lower = checkpoint_path.name.lower()
    if "stage1" in dir_name_lower or "stage_1" in dir_name_lower:
        print(f"  🔍 从目录名检测到 Stage 1: {checkpoint_path.name}")
        return "stage1"
    elif "stage2" in dir_name_lower or "stage_2" in dir_name_lower:
        print(f"  🔍 从目录名检测到 Stage 2: {checkpoint_path.name}")
        return "stage2"
    
    # 方法2: 从 model_config.json 检测
    config_path = checkpoint_path / "model_config.json"
    if config_path.exists():
        try:
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                if 'training_stage' in config:
                    detected_stage = config['training_stage']
                    print(f"  🔍 从 model_config.json 检测到训练阶段: {detected_stage}")
                    return detected_stage
        except Exception as e:
            print(f"  ⚠️  读取 model_config.json 失败: {e}")
    
    # 方法3: 从 non_lora_trainables.bin 检测（Stage 2 通常包含 LoRA 相关键）
    non_lora_path = checkpoint_path / "non_lora_trainables.bin"
    if non_lora_path.exists():
        try:
            weights = torch.load(non_lora_path, map_location="cpu", weights_only=False)
            # Stage 2 通常包含 LoRA 相关的键（如 base_model.model.model.layers...）
            # Stage 1 通常只包含 polar_projector 相关键
            has_lora_keys = any("base_model" in k or "lora" in k.lower() for k in weights.keys())
            if has_lora_keys:
                print(f"  🔍 从权重文件检测到 Stage 2（包含 LoRA 相关键）")
                return "stage2"
            else:
                print(f"  🔍 从权重文件检测到 Stage 1（仅包含 Projector 键）")
                return "stage1"
        except Exception as e:
            print(f"  ⚠️  读取权重文件失败: {e}")
    
    # 默认返回 stage2（向后兼容）
    print(f"  ⚠️  无法自动检测训练阶段，默认使用 Stage 2")
    return "stage2"


def load_trained_model(
    checkpoint_dir: str,
    llm_model_name: str = "/openbayes/input/input0/models/llava-v1.5-13b",
    clip_model_name: str = "/openbayes/input/input0/models/clip-vit-large-patch14-336",
    vae_model_path: str = "/openbayes/input/input0/models/sd-vae-ft-mse",
    hf_token: Optional[str] = None,
    device: str = "cuda",
    load_4bit: bool = False,  # 🟢 修改：默认关闭 4-bit，使用 FP16 全精度（推荐双卡 4090）
    use_multi_gpu: bool = False,
    training_stage: Optional[str] = None,  # 🟢 新增：手动指定训练阶段（"stage1" 或 "stage2"），如果为 None 则自动检测
):
    """
    加载训练后的模型（支持 Stage 1 和 Stage 2，LLaVA 1.5 架构）
    
    Args:
        checkpoint_dir: 检查点目录（包含 adapter_model.safetensors, non_lora_trainables.bin 等）
        llm_model_name: 基础 LLM 模型路径（LLaVA 1.5 完整模型路径）
        clip_model_name: CLIP 模型路径
        vae_model_path: VAE 模型路径（用于加载 VAE 编码器）
        hf_token: Hugging Face token
        device: 设备（cuda 或 cpu）
        load_4bit: 是否使用 4-bit 量化加载（默认 False，推荐双卡 4090 使用 FP16）
        use_multi_gpu: 是否使用多GPU模式（默认 False，检测到多卡时建议开启）
        training_stage: 手动指定训练阶段（"stage1" 或 "stage2"），如果为 None 则自动检测
    
    Returns:
        model: 加载后的模型
        tokenizer: 分词器
        image_processor: 图像处理器
        training_stage: 检测到的训练阶段（"stage1" 或 "stage2"）
    """
    # 🟢 自动检测或使用指定的训练阶段
    if training_stage is None:
        print("🔍 正在自动检测训练阶段...")
        detected_stage = detect_training_stage(checkpoint_dir)
        training_stage = detected_stage
    else:
        training_stage = training_stage.lower()
        if training_stage not in ["stage1", "stage2"]:
            raise ValueError(f"training_stage 必须是 'stage1' 或 'stage2'，但得到: {training_stage}")
        print(f"🔍 使用指定的训练阶段: {training_stage}")
    
    print("=" * 80)
    if training_stage == "stage1":
        print(f"加载训练后的 PolarVLM Stage 1 模型（特征对齐阶段）")
    else:
        print(f"加载训练后的 PolarVLM Stage 2 模型（对话微调阶段）")
    if load_4bit:
        print("⚠️  使用 4-bit 量化模式（精度较低，速度较慢）")
    else:
        print("🚀 使用 FP16 全精度模式（推荐双卡 4090，精度更高，速度更快）")
    print("=" * 80)
    
    # 导入必要的模块
    from transformers import BitsAndBytesConfig, AutoTokenizer, CLIPImageProcessor
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from llava.model.language_model.llava_llama import PolarLlavaConfig, PolarLlavaLlamaForCausalLM
    from peft import PeftModel
    from diffusers import AutoencoderKL  # 需要导入这个来加载 VAE 权重
    
    # 路径验证
    if not checkpoint_dir or not checkpoint_dir.strip():
        raise ValueError("checkpoint_dir 不能为空")
    if not llm_model_name or not llm_model_name.strip():
        raise ValueError("llm_model_name 不能为空")
    if not vae_model_path or not vae_model_path.strip():
        vae_model_path = "/openbayes/input/input0/models/sd-vae-ft-mse"
    
    # 清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 检测GPU数量
    num_gpus = torch.cuda.device_count() if device == "cuda" and torch.cuda.is_available() else 0
    print(f"检测到 {num_gpus} 个 GPU")
    
    # 1. 准备配置
    print(f"正在加载配置: {llm_model_name}")
    config = PolarLlavaConfig.from_pretrained(llm_model_name)
    config.polar_vae_model_path = vae_model_path
    config.mm_vision_tower = clip_model_name
    # 🟢 与当前训练架构对齐：默认 residual 融合（token 数保持 576）
    if not hasattr(config, "polar_fusion_mode") or not config.polar_fusion_mode:
        config.polar_fusion_mode = "residual"
    if not hasattr(config, "polar_only"):
        config.polar_only = False
    if not hasattr(config, "polar_rgb_dropout_p"):
        config.polar_rgb_dropout_p = 0.0
    
    # 量化配置
    quantization_config = None
    if load_4bit:
        print("  ⚡️ 启用 4-bit 量化配置 (NF4)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["mm_projector", "polar_projector", "polar_encoder", "polar_quant_conv"]
        )
        config.quantization_config = quantization_config
    else:
        print("  🚀 使用 FP16 全精度模式")
    
    # 2. Meta 初始化
    print("正在初始化 Meta 模型结构...")
    with init_empty_weights():
        model = PolarLlavaLlamaForCausalLM(config)
    
    # 3. 手动实体化 Polar 层 (解决 Meta Tensor 问题)
    print("正在实体化 Polar 分支...")
    base_model = model.get_model() if hasattr(model, 'get_model') else model
    
    # 🟢 关键修复：加入 polar_quant_conv 和 polar_projector_norm
    modules_to_materialize = [
        'polar_encoder', 
        'polar_quant_conv',   # <--- 之前漏了这一层！
        'polar_projector', 
        'polar_projector_norm',  # 🟢 新增：LayerNorm 也需要实体化
        'vae_latent_to_feature', 
        'mm_projector'
    ]
    
    for module_name in modules_to_materialize:
        if hasattr(base_model, module_name):
            print(f"  - Materializing {module_name}...")
            getattr(base_model, module_name).to_empty(device='cpu')
    
    # 关键修复：polar_alpha 是独立 Parameter，不属于任何子模块。
    # 在 init_empty_weights 下它会是 meta tensor；若不先实体化，后续 dispatch_model 会在 model.to(device) 时报错。
    if hasattr(base_model, "polar_alpha"):
        try:
            if hasattr(base_model.polar_alpha, "is_meta") and base_model.polar_alpha.is_meta:
                alpha_init = float(getattr(config, "polar_alpha_init", 0.5) or 0.5)
                base_model.polar_alpha = torch.nn.Parameter(
                    torch.tensor(alpha_init, dtype=torch.float32, device="cpu")
                )
                print(f"  - Materialized polar_alpha on CPU (init={alpha_init:.4f})")
        except Exception as e:
            print(f"  ⚠️ 警告: polar_alpha 实体化失败: {e}")
    
    # 🟢 关键修复：完全移除 polar_range_scale（推理时不需要，且未训练）
    # polar_range_scale 是 register_buffer，device_map="auto" 无法自动分配设备
    # 由于未训练，直接删除即可
    if hasattr(base_model, 'polar_range_scale'):
        if 'polar_range_scale' in base_model._buffers:
            del base_model._buffers['polar_range_scale']
        if hasattr(base_model, 'polar_range_scale'):
            delattr(base_model, 'polar_range_scale')
        print(f"  - 已移除 polar_range_scale（推理时不需要，且未训练）")
    
    # 4. 重载 VAE 权重 (因为 init_empty_weights 跳过了加载)
    print(f"正在重载 VAE 权重 (强制 FP32 以防溢出)...")
    try:
        # 🟢 关键修复：加载 VAE 并保持 FP32 (不要 .to(torch.float16))
        # Stable Diffusion VAE 在 FP16 下极易产生 NaN，导致 CUDA 设备端断言错误
        vae = AutoencoderKL.from_pretrained(vae_model_path, local_files_only=True)
        # 保持 FP32，不转换为 FP16
        
        if hasattr(base_model, 'polar_encoder'):
            # 直接加载权重，保持 FP32
            encoder_state = vae.encoder.state_dict()
            encoder_values = torch.cat([v.flatten() for v in encoder_state.values()])
            encoder_min, encoder_max = encoder_values.min().item(), encoder_values.max().item()
            encoder_nan = torch.isnan(encoder_values).sum().item()
            encoder_inf = torch.isinf(encoder_values).sum().item()
            
            base_model.polar_encoder.load_state_dict(encoder_state)
            # 保持 FP32，不转换为 FP16
            print("  ✓ Polar Encoder 权重已恢复 (FP32 - 稳定模式)")
            
        if hasattr(base_model, 'polar_quant_conv'):
            # 直接加载权重，保持 FP32
            quant_conv_state = vae.quant_conv.state_dict()
            quant_conv_values = torch.cat([v.flatten() for v in quant_conv_state.values()])
            quant_conv_min, quant_conv_max = quant_conv_values.min().item(), quant_conv_values.max().item()
            quant_conv_nan = torch.isnan(quant_conv_values).sum().item()
            quant_conv_inf = torch.isinf(quant_conv_values).sum().item()
            
            if abs(quant_conv_min) > 1e10 or abs(quant_conv_max) > 1e10:
                print(f"  ❌ 严重警告: Quant Conv 权重值异常巨大！这会导致 VAE 输出爆炸！")
                print(f"     建议：重新下载 VAE 模型权重文件")
            
            base_model.polar_quant_conv.load_state_dict(quant_conv_state)
            # 保持 FP32，不转换为 FP16
            print("  ✓ Polar Quant Conv 权重已恢复 (FP32 - 稳定模式)")
        
        # 🟢 关键修复：vae_latent_to_feature 设为 FP16，与 LLM 对齐
        # VAE 输出后会立即转换为 FP16，所以这个层也应该是 FP16
        if hasattr(base_model, 'vae_latent_to_feature'):
            base_model.vae_latent_to_feature.to(dtype=torch.float16)
            print("  ✓ VAE Latent to Feature 设为 FP16 (与 LLM 对齐)")
            
        del vae
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        print(f"  ⚠️ 警告: VAE 权重加载失败: {e}")
    
    # 5. 加载基础模型权重 (LLaVA)
    device_map_strategy = "auto"
    if load_4bit:
        device_map_strategy = {"": 0}
    
    # 多GPU模式：设置 max_memory
    max_memory_config = None
    if device == "cuda" and num_gpus > 1 and (use_multi_gpu or not load_4bit):
        max_memory_config = {i: "23GB" for i in range(num_gpus)}
        print(f"  ✓ 多GPU模式：使用 device_map='auto' 自动分布到 {num_gpus} 个 GPU")
        print(f"  ✓ 显存限制: 每个GPU最多23GB（双卡 4090 显存充裕）")
    
    print(f"正在加载基础模型权重 (Device Map: {device_map_strategy})...")
    model = load_checkpoint_and_dispatch(
        model,
        llm_model_name,
        device_map=device_map_strategy,
        no_split_module_classes=["LlamaDecoderLayer", "PolarEncoder"],
        dtype=torch.float16,
        max_memory=max_memory_config,
    )
    
    # 打印显存分配情况
    if torch.cuda.is_available():
        print("  显存分配情况:")
        for i in range(num_gpus):
            print(f"    GPU {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
    
    # 🟢 polar_range_scale 已在实体化阶段移除，无需重新注册

    def _cast_module_to_fp16_inplace(module):
        for p in module.parameters(recurse=True):
            if hasattr(p, "is_meta") and p.is_meta:
                continue
            if p.dtype.is_floating_point and p.dtype != torch.float16:
                p.data = p.data.to(dtype=torch.float16)
        for b in module.buffers(recurse=True):
            if hasattr(b, "is_meta") and b.is_meta:
                continue
            if b.dtype.is_floating_point and b.dtype != torch.float16:
                b.data = b.data.to(dtype=torch.float16)

    def _force_non_vae_fp16(model_for_cast):
        """强制非 VAE 模块为 FP16，避免 hidden_states=float -> weight=half 的错误。"""
        try:
            skip_keywords = ("polar_encoder", "polar_quant_conv")
            for name, p in model_for_cast.named_parameters():
                if hasattr(p, "is_meta") and p.is_meta:
                    continue
                if any(k in name for k in skip_keywords):
                    continue
                if p.dtype.is_floating_point and p.dtype != torch.float16:
                    p.data = p.data.to(dtype=torch.float16)
            for name, b in model_for_cast.named_buffers():
                if hasattr(b, "is_meta") and b.is_meta:
                    continue
                if any(k in name for k in skip_keywords):
                    continue
                if b.dtype.is_floating_point and b.dtype != torch.float16:
                    b.data = b.data.to(dtype=torch.float16)
        except Exception as e:
            print(f"  ⚠️  警告: 强制非 VAE 模块为 FP16 失败: {e}")
    
    # 🟢 关键修复：确保所有手动实体化的模块都是正确的 dtype
    # load_checkpoint_and_dispatch 可能不会自动转换这些模块
    if not load_4bit:
        if hasattr(base_model, 'mm_projector'):
            base_model.mm_projector.to(dtype=torch.float16)
        # 🟢 vae_latent_to_feature 保持 FP32（已在 Step 4 中设置，这里不再修改）
        # 注意：vae_latent_to_feature 在 Step 4 中已设为 FP32，与 VAE 保持一致
        _force_non_vae_fp16(model)
    
    # 6. 加载 Tokenizer 和 Vision Tower (🟢 修复 Vision Tower 未初始化问题)
    print("加载 Tokenizer 和 Vision Tower...")
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name, use_fast=False)
    
    # 记录原始词表大小（用于后续添加新 token 与对齐 Stage 2 训练结果）
    orig_vocab_size = len(tokenizer)
    
    # 🟢 关键修复：添加 LLaVA 特有 Token（防止 Token ID 越界）
    # Stage 1 和 Stage 2 都不使用分隔 token，RGB 和 Polar 特征直接拼接
    vocab_size_before_tokens = len(tokenizer)
    
    mm_use_im_patch_token = getattr(config, "mm_use_im_patch_token", True)
    mm_use_im_start_end = getattr(config, "mm_use_im_start_end", False)
    
    tokens_to_add = []
    if mm_use_im_patch_token and DEFAULT_IMAGE_PATCH_TOKEN not in tokenizer.get_vocab():
        tokens_to_add.append(DEFAULT_IMAGE_PATCH_TOKEN)
        print(f"  ✓ 添加 {DEFAULT_IMAGE_PATCH_TOKEN} token")
    
    if mm_use_im_start_end:
        if DEFAULT_IM_START_TOKEN not in tokenizer.get_vocab():
            tokens_to_add.append(DEFAULT_IM_START_TOKEN)
        if DEFAULT_IM_END_TOKEN not in tokenizer.get_vocab():
            tokens_to_add.append(DEFAULT_IM_END_TOKEN)
        if tokens_to_add:
            print(f"  ✓ 添加 IM_START/END tokens")
    
    # 先添加LLaVA基础token（如果有）
    if tokens_to_add:
        num_added_base = tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
    
    # 🟢 Stage 1 和 Stage 2: 都不添加分隔 token，使用残差融合（与训练时一致）
    if training_stage == "stage1":
        print(f"  ℹ️  Stage 1: 不添加分隔 token，使用残差融合（特征对齐阶段）")
    else:
        print(f"  ℹ️  Stage 2: 不添加分隔 token，使用残差融合（避免过拟合）")
    
    vocab_size_after_tokens = len(tokenizer)
    
    # 重新计算词表大小并调整 Embedding
    # ⚠️ 重要：禁用 mean_resizing（covariance 计算会访问 meta tensor，导致 Tensor.item() on meta 报错）
    if len(tokenizer) != orig_vocab_size:
        try:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        except TypeError:
            # 兼容旧版 transformers 没有 mean_resizing 参数的情况
            model.resize_token_embeddings(len(tokenizer))
        print(f"  ✓ 模型 Embedding 已调整到词汇表大小: {len(tokenizer)} (原始 {orig_vocab_size})")
    
    # 🟢 关键修复：确保 pad_token 和 eos_token 正确设置（防止 CUDA 设备端断言错误）
    if tokenizer.pad_token is None:
        # LLaVA 1.5 通常使用 unk_token 作为 pad_token
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
            print(f"  ✓ 设置 pad_token = unk_token (ID: {tokenizer.pad_token_id})")
        else:
            # 如果 unk_token 也不存在，使用 eos_token
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"  ✓ 设置 pad_token = eos_token (ID: {tokenizer.pad_token_id})")
    else:
        print(f"  ✓ pad_token 已设置 (ID: {tokenizer.pad_token_id})")
    
    # 确保 eos_token 存在
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer 缺少 eos_token，这是必需的！")
    else:
        print(f"  ✓ eos_token 已设置 (ID: {tokenizer.eos_token_id})")
    
    # 验证 token IDs 是否在有效范围内
    vocab_size = len(tokenizer)
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id >= vocab_size:
        raise ValueError(f"pad_token_id ({tokenizer.pad_token_id}) 超出词汇表大小 ({vocab_size})")
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id >= vocab_size:
        raise ValueError(f"eos_token_id ({tokenizer.eos_token_id}) 超出词汇表大小 ({vocab_size})")
    
    print(f"  ✓ Tokenizer 词汇表大小: {vocab_size}")

    # 🟢 显式获取并加载 Vision Tower
    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        print(f"  正在加载 Vision Tower: {clip_model_name}")
        # 确保 vision_tower 配置了正确的路径
        vision_tower.vision_tower_name = clip_model_name
        
        # 显式加载 CLIP 权重
        if not vision_tower.is_loaded:
            # 使用与主模型相同的 device_map 策略
            # 对于多GPU模式，vision_tower 通常放在 GPU 0
            if device_map_strategy == "auto" and num_gpus > 1:
                vision_tower.load_model(device_map="cuda:0")
            else:
                vision_tower.load_model(device_map=device_map_strategy)
        
        # 确保数据类型正确 (FP16)
        # 注意：需要在 load_model 之后，因为此时 vision_tower.vision_tower 才存在
        if not load_4bit and vision_tower.is_loaded:
            # 通过内部的 vision_tower 设置 dtype
            if hasattr(vision_tower, 'vision_tower') and vision_tower.vision_tower is not None:
                vision_tower.vision_tower = vision_tower.vision_tower.to(dtype=torch.float16)
            
        print("  ✓ Vision Tower 加载完成")
        image_processor = vision_tower.image_processor
    else:
        # 备用方案：如果没有 vision_tower，手动加载 processor
        print("  ⚠️ 警告: 未找到 Vision Tower，尝试手动加载 Processor")
        image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)

    # 修复 image_processor 尺寸为 336 (LLaVA 1.5 标准)
    if hasattr(image_processor, 'crop_size'):
        image_processor.crop_size['height'] = 336
        image_processor.crop_size['width'] = 336
        if hasattr(image_processor, 'size'):
            image_processor.size['shortest_edge'] = 336
    
    # 7. 加载训练好的 Projector 权重
    print("加载训练后的适配器权重...")
    non_lora_path = os.path.join(checkpoint_dir, "non_lora_trainables.bin")
    # 预留变量，用于在加载 non_lora_trainables.bin 时捕获 Stage 2 训练好的 embedding 权重
    embed_tokens_weight = None
    lm_head_weight = None
    
    if os.path.exists(non_lora_path):
        print(f"  - 加载 Polar Projector: {non_lora_path}")
        non_lora_weights = torch.load(non_lora_path, map_location="cpu")
        
        clean_weights = {}
        vae_latent_weights = {}  # 🟢 新增：单独处理 vae_latent_to_feature 权重
        norm_weights = {}  # 🟢 新增：单独处理 polar_projector_norm 权重
        llm_weights = {}  # 🟢 Stage 1: 可选加载 LLM 权重（如果训练时未冻结）
        polar_alpha_weight = None  # 🟢 保存 polar_alpha
        
        for k, v in non_lora_weights.items():
            new_k = k
            # 🟢 关键修复：处理多层嵌套的前缀（训练时保存的键名可能是 base_model.model.model.polar_projector.*）
            for prefix in ["module.base_model.model.model.", "base_model.model.model.", "base_model.model.", "model."]:
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
                    break
            
            # 🟢 关键修复：同时处理 polar_projector、polar_projector_norm 和 vae_latent_to_feature
            # 🚨 统一精度修复：所有组件统一为 FP16，与 LLM 对齐
            if new_k.startswith("polar_projector.") and "polar_projector_norm" not in new_k:
                new_k = new_k[len("polar_projector."):]
                # 🟢 关键修改：统一为 FP16（与 LLM 对齐）
                clean_weights[new_k] = v.to(dtype=torch.float16)
            elif new_k.startswith("polar_projector_norm."):
                # 🟢 新增：处理 polar_projector_norm 权重（LayerNorm），统一为 FP16
                norm_key = new_k[len("polar_projector_norm."):]
                norm_weights[norm_key] = v.to(dtype=torch.float16)
                print(f"  ✓ 找到 polar_projector_norm 权重: {norm_key}")
            elif new_k.startswith("vae_latent_to_feature."):
                # 🟢 新增：处理 vae_latent_to_feature 权重，统一为 FP16
                vae_latent_key = new_k[len("vae_latent_to_feature."):]
                vae_latent_weights[vae_latent_key] = v.to(dtype=torch.float16)
                print(f"  ✓ 找到 vae_latent_to_feature 权重: {vae_latent_key}")
            elif new_k == "polar_alpha" or new_k.endswith(".polar_alpha"):
                # 🟢 保存 polar_alpha（残差融合的可学习系数）
                polar_alpha_weight = v.detach().float().cpu()
                print(f"  ✓ 找到 polar_alpha 权重: {polar_alpha_weight.item():.6f}")
            # 捕获 Stage 2 训练好的 embedding 权重
            elif new_k.endswith("embed_tokens.weight") or new_k == "embed_tokens.weight":
                embed_tokens_weight = v
                print(f"  ✓ 捕获到 embed_tokens.weight (shape={tuple(v.shape)})")
            elif new_k.endswith("lm_head.weight") or new_k == "lm_head.weight":
                lm_head_weight = v
                print(f"  ✓ 捕获到 lm_head.weight (shape={tuple(v.shape)})")
            elif training_stage == "stage1":
                # 🟢 Stage 1: 如果训练时未冻结 LLM，这里需要加载 LLM 权重以匹配训练分布
                llm_key = new_k
                if llm_key.startswith("model."):
                    llm_key = llm_key[len("model."):]
                if llm_key.startswith(("layers.", "embed_tokens.", "norm.")):
                    llm_key = f"model.{llm_key}"
                    llm_key = llm_key.replace(".base_layer.", ".")
                    llm_weights[llm_key] = v.to(dtype=torch.float16)
        
        # 🟢 关键修复：检查权重健康度（防止加载损坏的权重）
        if clean_weights:
            print("  正在检查 Projector 权重数值健康度...")
            is_broken = False
            for k, v in clean_weights.items():
                if torch.isnan(v).any() or torch.isinf(v).any():
                    nan_count = torch.isnan(v).sum().item() if torch.isnan(v).any() else 0
                    inf_count = torch.isinf(v).sum().item() if torch.isinf(v).any() else 0
                    print(f"  ❌ 致命错误: 权重 {k} 包含 NaN ({nan_count} 个) 或 Inf ({inf_count} 个)！")
                    print(f"     训练可能已失败，建议回退到更早的 checkpoint 或重新训练。")
                    is_broken = True
                    break
            
            if not is_broken:
                # 检查权重范围是否合理
                all_values = torch.cat([v.flatten() for v in clean_weights.values()])
                weight_min, weight_max = all_values.min().item(), all_values.max().item()
                weight_mean, weight_std = all_values.mean().item(), all_values.std().item()
                print(f"  ✓ 权重数值正常 (范围: [{weight_min:.4f}, {weight_max:.4f}], 均值: {weight_mean:.4f}, 标准差: {weight_std:.4f})")
        
        if hasattr(base_model, 'polar_projector') and clean_weights:
            try:
                # 自动处理设备
                target_device = next(base_model.polar_projector.parameters()).device
                # 🟢 关键修改：统一为 FP16（与 LLM 对齐）
                clean_weights = {k: v.to(device=target_device, dtype=torch.float16) for k, v in clean_weights.items()}
                base_model.polar_projector.load_state_dict(clean_weights, strict=False)
                # 🟢 关键修改：确保模块本身是 FP16（与 LLM 对齐）
                base_model.polar_projector.to(dtype=torch.float16)
                print(f"    ✓ Polar Projector 加载成功 (FP16 - 与 LLM 对齐)")
            except Exception as e:
                print(f"    ❌ Polar Projector 加载失败: {e}")

        # 🟢 加载 polar_alpha（残差融合系数）
        if polar_alpha_weight is not None and hasattr(base_model, "polar_alpha"):
            try:
                # 获取一个非 meta 的设备作为落点
                target_device = None
                for p in base_model.parameters():
                    if p is not None and hasattr(p, "is_meta") and not p.is_meta:
                        target_device = p.device
                        break
                if target_device is None:
                    target_device = torch.device("cpu")

                if hasattr(base_model.polar_alpha, "is_meta") and base_model.polar_alpha.is_meta:
                    base_model.polar_alpha = torch.nn.Parameter(
                        polar_alpha_weight.to(device=target_device, dtype=torch.float32)
                    )
                else:
                    base_model.polar_alpha.data = polar_alpha_weight.to(
                        device=base_model.polar_alpha.data.device,
                        dtype=base_model.polar_alpha.data.dtype,
                    )
                print(f"    ✓ polar_alpha 已加载: {base_model.polar_alpha.data.item():.6f}")
            except Exception as e:
                print(f"    ❌ polar_alpha 加载失败: {e}")

        # 🟢 Stage 1: 如果存在 LLM 权重，加载以匹配训练时分布
        if training_stage == "stage1" and llm_weights:
            print(f"  - 加载 Stage 1 LLM 权重（检测到 {len(llm_weights)} 个参数）...")
            try:
                base_model.load_state_dict(llm_weights, strict=False)
                print("    ✓ Stage 1 LLM 权重加载完成")
                _force_non_vae_fp16(model)
            except Exception as e:
                print(f"    ❌ Stage 1 LLM 权重加载失败: {e}")
        
        # 🟢 关键修复：加载 polar_projector_norm 权重（LayerNorm，Stage 2 训练后必须加载）
        if norm_weights and hasattr(base_model, 'polar_projector_norm'):
            print(f"  - 加载 polar_projector_norm 权重（LayerNorm）...")
            try:
                # 检查权重健康度
                for k, v in norm_weights.items():
                    if torch.isnan(v).any() or torch.isinf(v).any():
                        print(f"  ❌ 警告: polar_projector_norm.{k} 包含 NaN/Inf，跳过加载")
                        norm_weights = {}
                        break
                
                if norm_weights:
                    target_device = next(base_model.polar_projector_norm.parameters()).device
                    if target_device == torch.device("meta"):
                        target_device = torch.device("cpu")
                    # 🟢 统一为 FP16（与 LLM 对齐）
                    norm_weights = {k: v.to(device=target_device, dtype=torch.float16) for k, v in norm_weights.items()}
                    base_model.polar_projector_norm.load_state_dict(norm_weights, strict=False)
                    base_model.polar_projector_norm.to(dtype=torch.float16)
                    
                    # 检查权重范围（特别是 gamma 参数应该接近 1.0）
                    all_values = torch.cat([v.flatten() for v in norm_weights.values()]).to("cpu")
                    weight_min, weight_max = all_values.min().item(), all_values.max().item()
                    weight_mean, weight_std = all_values.mean().item(), all_values.std().item()
                    
                    # 检查 gamma 参数（weight）
                    if 'weight' in norm_weights:
                        gamma_mean = norm_weights['weight'].mean().item()
                        gamma_std = norm_weights['weight'].std().item()
                        print(f"    ✓ polar_projector_norm 权重已加载 (范围: [{weight_min:.4f}, {weight_max:.4f}])")
                        print(f"    📊 LayerNorm gamma 参数: mean={gamma_mean:.6f}, std={gamma_std:.6f}")
                        if abs(gamma_mean - 1.0) < 0.1:
                            print(f"      ✅ gamma 参数正常（接近 1.0），LayerNorm 已正确训练")
                        else:
                            print(f"      ⚠️  警告: gamma 参数偏离 1.0（mean={gamma_mean:.6f}），可能影响输出分布")
                    else:
                        print(f"    ✓ polar_projector_norm 权重已加载 (范围: [{weight_min:.4f}, {weight_max:.4f}])")
                else:
                    print(f"    ⚠️ polar_projector_norm 权重未找到或损坏，将使用随机初始化（可能导致输出分布异常）")
            except Exception as e:
                print(f"    ❌ polar_projector_norm 权重加载失败: {e}")
                print(f"    ⚠️ 警告: 将使用随机初始化的权重（可能导致输出分布异常）")
        elif hasattr(base_model, 'polar_projector_norm'):
            print(f"  ⚠️ 警告: checkpoint 中未找到 polar_projector_norm 权重，将使用随机初始化（LayerNorm 会从头训练）")
        
        # 🟢 关键修复：加载 vae_latent_to_feature 权重（如果存在）
        if vae_latent_weights and hasattr(base_model, 'vae_latent_to_feature'):
            print(f"  - 加载 vae_latent_to_feature 权重...")
            try:
                # 检查权重健康度（在 CPU 上完成，避免 meta tensor）
                for k, v in vae_latent_weights.items():
                    if torch.isnan(v).any() or torch.isinf(v).any():
                        print(f"  ❌ 警告: vae_latent_to_feature.{k} 包含 NaN/Inf，跳过加载")
                        vae_latent_weights = {}
                        break
                
                if vae_latent_weights:
                    target_device = next(base_model.vae_latent_to_feature.parameters()).device
                    # 如果还是 meta 设备，则先落到 CPU
                    if target_device == torch.device("meta"):
                        target_device = torch.device("cpu")
                    # 🟢 关键修改：统一为 FP16（与 LLM 对齐）
                    vae_latent_weights = {k: v.to(device=target_device, dtype=torch.float16) for k, v in vae_latent_weights.items()}
                    base_model.vae_latent_to_feature.load_state_dict(vae_latent_weights, strict=False)
                    base_model.vae_latent_to_feature.to(dtype=torch.float16)
                    
                    # 检查权重范围（此时不应再是 meta）
                    all_values = torch.cat([v.flatten() for v in vae_latent_weights.values()]).to("cpu")
                    weight_min, weight_max = all_values.min().item(), all_values.max().item()
                    print(f"    ✓ vae_latent_to_feature 权重已加载 (范围: [{weight_min:.4f}, {weight_max:.4f}])")
                else:
                    print(f"    ⚠️ vae_latent_to_feature 权重未找到或损坏，将使用随机初始化（可能导致输出异常）")
            except Exception as e:
                print(f"    ❌ vae_latent_to_feature 权重加载失败: {e}")
                print(f"    ⚠️ 警告: 将使用随机初始化的权重（可能导致输出异常）")
        
        # 🟢 关键修复：如果 vae_latent_to_feature 权重不存在，尝试从 checkpoint 目录中查找
        if not vae_latent_weights and hasattr(base_model, 'vae_latent_to_feature'):
            print(f"  ⚠️ 警告: checkpoint 中未找到 vae_latent_to_feature 权重")
            
            # 尝试从 checkpoint 子目录中查找（如 checkpoint-400）
            checkpoint_found = False
            checkpoint_dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-') and os.path.isdir(os.path.join(checkpoint_dir, d))]
            checkpoint_dirs.sort(key=lambda x: int(x.split('-')[1]) if x.split('-')[1].isdigit() else 0, reverse=True)  # 按数字降序排序
            
            for ckpt_dir in checkpoint_dirs:
                ckpt_path = os.path.join(checkpoint_dir, ckpt_dir)
                # 检查是否有 non_lora_trainables.bin 或完整模型文件
                ckpt_non_lora = os.path.join(ckpt_path, 'non_lora_trainables.bin')
                if os.path.exists(ckpt_non_lora):
                    print(f"  正在检查 {ckpt_dir}...")
                    try:
                        ckpt_weights = torch.load(ckpt_non_lora, map_location="cpu")
                        # 查找 vae_latent_to_feature 权重
                        for k, v in ckpt_weights.items():
                            new_k = k
                            for prefix in ["base_model.model.", "model."]:
                                if new_k.startswith(prefix):
                                    new_k = new_k[len(prefix):]
                            if new_k.startswith("vae_latent_to_feature."):
                                vae_latent_key = new_k[len("vae_latent_to_feature."):]
                                vae_latent_weights[vae_latent_key] = v.to(dtype=torch.float16)
                                print(f"  ✓ 在 {ckpt_dir} 中找到 vae_latent_to_feature 权重: {vae_latent_key}")
                                checkpoint_found = True
                                break
                        if checkpoint_found:
                            break
                    except Exception as e:
                        print(f"  ⚠️ 检查 {ckpt_dir} 时出错: {e}")
                        continue
            
            # 如果从 checkpoint 中找到了权重，加载它
            if vae_latent_weights and hasattr(base_model, 'vae_latent_to_feature'):
                print(f"  - 加载 vae_latent_to_feature 权重（从 checkpoint）...")
                try:
                    target_device = next(base_model.vae_latent_to_feature.parameters()).device
                    if target_device == torch.device("meta"):
                        target_device = torch.device("cpu")
                    vae_latent_weights = {k: v.to(device=target_device, dtype=torch.float16) for k, v in vae_latent_weights.items()}
                    base_model.vae_latent_to_feature.load_state_dict(vae_latent_weights, strict=False)
                    base_model.vae_latent_to_feature.to(dtype=torch.float16)
                    all_values = torch.cat([v.flatten() for v in vae_latent_weights.values()]).to("cpu")
                    weight_min, weight_max = all_values.min().item(), all_values.max().item()
                    print(f"    ✓ vae_latent_to_feature 权重已加载 (范围: [{weight_min:.4f}, {weight_max:.4f}])")
                except Exception as e:
                    print(f"    ❌ vae_latent_to_feature 权重加载失败: {e}")
                    vae_latent_weights = {}  # 清空，使用初始化
            
            # 如果仍然没有找到，使用合理的初始化
            if not vae_latent_weights:
                print(f"     将使用 Xavier 初始化重新初始化该层...")
                try:
                    import torch.nn.init as init
                    # 使用 Xavier 初始化（适合 Conv2d）
                    if hasattr(base_model.vae_latent_to_feature, 'weight'):
                        init.xavier_uniform_(base_model.vae_latent_to_feature.weight, gain=0.1)  # 使用较小的 gain
                    if hasattr(base_model.vae_latent_to_feature, 'bias') and base_model.vae_latent_to_feature.bias is not None:
                        init.zeros_(base_model.vae_latent_to_feature.bias)
                    base_model.vae_latent_to_feature.to(dtype=torch.float16)
                    print(f"    ✓ vae_latent_to_feature 已重新初始化 (Xavier, gain=0.1, FP16)")
                except Exception as e:
                    print(f"    ❌ 重新初始化失败: {e}")
    
    # 在成功解析 non_lora_trainables 之后，再尝试覆盖 Stage 2 训练好的 embedding 权重
    try:
        if embed_tokens_weight is not None:
            input_embeddings = model.get_input_embeddings()
            if hasattr(input_embeddings, "weight"):
                if embed_tokens_weight.shape == input_embeddings.weight.shape:
                    input_embeddings.weight.data.copy_(embed_tokens_weight.to(
                        device=input_embeddings.weight.device,
                        dtype=input_embeddings.weight.dtype,
                    ))
                    print(f"  ✓ 已从 non_lora_trainables.bin 加载完整 embed_tokens 权重")
                elif embed_tokens_weight.shape[0] == input_embeddings.weight.shape[0]:
                    # 只形状 dtype 不同，按行复制
                    input_embeddings.weight.data.copy_(embed_tokens_weight.to(
                        device=input_embeddings.weight.device,
                        dtype=input_embeddings.weight.dtype,
                    ))
                    print(f"  ✓ 已从 non_lora_trainables.bin 对齐复制 embed_tokens 权重")
                else:
                    print(f"  ⚠️ embed_tokens 权重形状不匹配，跳过加载: "
                          f"checkpoint={tuple(embed_tokens_weight.shape)}, "
                          f"model={tuple(input_embeddings.weight.shape)}")
            # 🟢 确保 embed_tokens 为 FP16，避免 hidden_states=float
            try:
                input_embeddings.to(dtype=torch.float16)
                print("  ✓ embed_tokens 已强制转换为 FP16")
            except Exception as e:
                print(f"  ⚠️ embed_tokens 转 FP16 失败: {e}")
        if lm_head_weight is not None and hasattr(model, "get_output_embeddings"):
            output_embeddings = model.get_output_embeddings()
            if output_embeddings is not None and hasattr(output_embeddings, "weight"):
                if lm_head_weight.shape == output_embeddings.weight.shape:
                    output_embeddings.weight.data.copy_(lm_head_weight.to(
                        device=output_embeddings.weight.device,
                        dtype=output_embeddings.weight.dtype,
                    ))
                    print(f"  ✓ 已从 non_lora_trainables.bin 加载完整 lm_head 权重")
                elif lm_head_weight.shape[0] == output_embeddings.weight.shape[0]:
                    output_embeddings.weight.data.copy_(lm_head_weight.to(
                        device=output_embeddings.weight.device,
                        dtype=output_embeddings.weight.dtype,
                    ))
                    print(f"  ✓ 已从 non_lora_trainables.bin 对齐复制 lm_head 权重")
                else:
                    print(f"  ⚠️ lm_head 权重形状不匹配，跳过加载: "
                          f"checkpoint={tuple(lm_head_weight.shape)}, "
                          f"model={tuple(output_embeddings.weight.shape)}")
            # 🟢 确保 lm_head 为 FP16（若存在）
            try:
                if output_embeddings is not None:
                    output_embeddings.to(dtype=torch.float16)
                    print("  ✓ lm_head 已强制转换为 FP16")
            except Exception as e:
                print(f"  ⚠️ lm_head 转 FP16 失败: {e}")
    except Exception as e:
        print(f"  ⚠️ 加载 Stage 2 embedding 权重时出错，已跳过（不会影响基本推理）: {e}")
    
    # 8. 加载 LoRA（仅 Stage 2 需要；Stage 1 必须跳过）
    adapter_config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    adapter_model_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    if training_stage == "stage2" and os.path.exists(adapter_config_path) and os.path.exists(adapter_model_path):
        print(f"  - 加载 LoRA 适配器（Stage 2 冻结 embedding 层，跳过 embedding 权重）...")
        try:
            from peft import PeftConfig
            from safetensors.torch import load_file

            # 1）构建 PEFT 模型包装
            peft_config = PeftConfig.from_pretrained(checkpoint_dir)
            model = PeftModel(model, peft_config)

            # 🟢 关键修复：LoRA 包装后，重新设置 training_stage 到包装后的 model.config
            # 因为 PeftModel 可能使用 base_model.config 或自己的 config
            if hasattr(model, 'config'):
                model.config.training_stage = training_stage
                print(f"    ✓ 重新设置 training_stage = '{training_stage}' 到 LoRA 包装后的 model.config")
            
            # 2）加载 adapter state_dict，并过滤掉 embedding 权重（Stage 2 冻结了 embedding 层）
            adapter_state = load_file(adapter_model_path, device="cpu")
            filtered_state = {}
            skipped_keys = []
            for k, v in adapter_state.items():
                # 跳过 embedding / lm_head 权重（Stage 2 冻结了 embedding 层，使用基础模型的 embedding）
                if (
                    "embed_tokens.original_module.weight" in k
                    or "lm_head.original_module.weight" in k
                ):
                    skipped_keys.append(f"{k} (shape={tuple(v.shape)})")
                    continue
                filtered_state[k] = v

            if skipped_keys:
                print("    ℹ️  跳过以下 LoRA embedding 权重（Stage 2 冻结 embedding 层，使用基础模型 embedding）：")
                for name in skipped_keys:
                    print(f"       - {name}")

            missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
            print(f"    ✓ LoRA 权重已加载（filtered）: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")
        except torch.cuda.OutOfMemoryError as e:
            print(f"  ❌ 加载 LoRA 时显存不足: {e}")
            print("  💡 建议：")
            print("    1) 在有两张 4090 的机器上，使用 CUDA_VISIBLE_DEVICES=0,1 并保持 load_4bit=False，多卡 FP16 推理；")
            print("    2) 或在当前 24GB 单卡环境中，将 eval_polar_stage2.py 改为传入 load_4bit=True 进行 4-bit 推理。")
            raise
        except Exception as e:
            print(f"  ❌ 加载 LoRA 适配器失败: {e}")
            raise
    elif training_stage == "stage1":
        print("  ℹ️ Stage 1：跳过 LoRA 加载（即使存在 adapter 文件）")
    elif os.path.exists(adapter_config_path) or os.path.exists(adapter_model_path):
        print("  ⚠️ 检测到 LoRA 文件但当前不是 Stage 2，已跳过加载")
    
    # 🟢 关键修复：根据训练阶段设置对话模板（必须与训练时完全一致）
    # Stage 1: 使用 plain 模板（无 System Prompt）
    # Stage 2: 使用 v1 模板（有 System Prompt）
    if hasattr(conversation_lib, 'default_conversation'):
        if training_stage == "stage1":
            # Stage 1: 使用 plain 模板（无 System Prompt）
            if "plain" in conversation_lib.conv_templates:
                conversation_lib.default_conversation = conversation_lib.conv_templates["plain"]
            else:
                print(f"⚠️ 警告: 'plain' 模板不存在，使用默认模板")
                # 手动创建 plain 模板
                from llava.conversation import Conversation, SeparatorStyle
                conversation_lib.default_conversation = Conversation(
                    system="",
                    roles=("", ""),
                    messages=(),
                    offset=0,
                    sep_style=SeparatorStyle.PLAIN,
                    sep="\n",
                    version="plain",
                )
                print(f"✓ 已手动创建 plain 模板")
        else:
            # Stage 2: 使用 v1 模板（有 System Prompt）
            if "v1" in conversation_lib.conv_templates:
                conversation_lib.default_conversation = conversation_lib.conv_templates["v1"]
            else:
                print(f"⚠️ 警告: 'v1' 模板不存在，使用 'vicuna_v1'")
                conversation_lib.default_conversation = conversation_lib.conv_templates.get("vicuna_v1", conversation_lib.conv_templates[list(conversation_lib.conv_templates.keys())[0]])
        
    
    # 🟢 关键修复：设置 training_stage，确保融合方式正确
    # Stage 1/2: 不使用分隔 token，默认 residual 融合（token 数保持 576）
    # 需要同时设置到 model.config 和 base_model.config（如果存在）
    if hasattr(model, 'config'):
        model.config.training_stage = training_stage
        print(f"✓ 设置 training_stage = '{training_stage}'（残差融合，不使用分隔 token）")
        
        # 🟢 如果模型被 LoRA 包装，也需要设置到 base_model.config
        if hasattr(model, 'get_base_model'):
            try:
                base_model = model.get_base_model()
                if hasattr(base_model, 'config'):
                    base_model.config.training_stage = training_stage
                    print(f"  ✓ 同时设置到 base_model.config（LoRA 包装后）")
            except Exception as e:
                print(f"  ⚠️ 无法访问 base_model.config: {e}")
                import traceback
                traceback.print_exc()
    
    # 🟢 [终极精度修复]：为 vae_latent_to_feature 加上 FP32->FP16 运行时转换包装器
    # 数据流：VAE Encoder(FP32) -> Quant Conv(FP32) -> latent(FP32)
    #       -> vae_latent_to_feature(Wrapper: FP32 输入显式转 FP16) -> Projector/LLM(FP16)
    try:
        # 再次获取 base_model（此时模型可能已经被 LoRA 包装）
        if hasattr(model, "get_model"):
            base_model_for_wrapper = model.get_model()
        else:
            base_model_for_wrapper = model

        if hasattr(base_model_for_wrapper, "vae_latent_to_feature"):
            print("  🛠️ 应用 FP32->FP16 运行时转换补丁 (Wrapping vae_latent_to_feature)...")
            base_module = base_model_for_wrapper.vae_latent_to_feature
            # 避免重复包裹
            if not isinstance(base_module, FP32toFP16Wrapper):
                base_model_for_wrapper.vae_latent_to_feature = FP32toFP16Wrapper(base_module)
                print("  ✓ 补丁应用成功：VAE(FP32) -> [Cast FP16] -> vae_latent_to_feature(FP16) 数据流已打通")
            else:
                print("  ℹ️  已检测到 vae_latent_to_feature 已被 FP32toFP16Wrapper 包裹，跳过重复应用")
        else:
            print("  ⚠️ 警告：未找到 vae_latent_to_feature 层，无法应用 FP32->FP16 补丁")
    except Exception as e:
        print(f"  ⚠️ 警告：应用 FP32->FP16 运行时补丁失败: {e}")
    
    model.eval()
    print("✓ 模型加载完成！")
    print(f"✓ 训练阶段: {training_stage}")
    return model, tokenizer, image_processor, training_stage


def get_polar_paths(polar_root: str, scene_id: str, base_name: str) -> Dict[str, Path]:
    """
    根据 scene_id 和 base_name 推导偏振图像的四个通道路径
    
    Args:
        polar_root: 偏振图像根目录
        scene_id: 场景ID
        base_name: 基础文件名（不含扩展名，如 "0002"）
    
    Returns:
        包含四个偏振通道路径的字典
    """
    polar_root_path = Path(polar_root)
    polar_scene_dir = polar_root_path / scene_id
    
    # 尝试三位数角度格式（推荐）
    polar_paths = {
        'I_0': polar_scene_dir / f"{base_name}_000.png",
        'I_45': polar_scene_dir / f"{base_name}_045.png",
        'I_90': polar_scene_dir / f"{base_name}_090.png",
        'I_135': polar_scene_dir / f"{base_name}_135.png",
    }
    
    if all(p.exists() for p in polar_paths.values()):
        return polar_paths
    
    # 尝试两位数角度格式
    polar_paths = {
        'I_0': polar_scene_dir / f"{base_name}_0.png",
        'I_45': polar_scene_dir / f"{base_name}_45.png",
        'I_90': polar_scene_dir / f"{base_name}_90.png",
        'I_135': polar_scene_dir / f"{base_name}_135.png",
    }
    
    if all(p.exists() for p in polar_paths.values()):
        return polar_paths
    
    # 返回最后尝试的路径（会在后续步骤中报错）
    return polar_paths


def preprocess_images(
    rgb_path: str,
    polar_paths: Dict[str, Path],
    image_processor: CLIPImageProcessor,
    device: str = "cuda",
    model=None,
) -> tuple:
    """
    预处理 RGB 和偏振图像（与训练阶段一致，LLaVA 1.5 架构）
    
    ⚠️ 关键修改：
    - RGB 图像：使用 image_processor 处理（LLaVA 1.5 使用 336x336）
    - Polar 图像：保持 512x512（VAE 编码器需要）
    - Polar 输入：3 通道（DoLP, sin(2*AoLP), cos(2*AoLP)），去掉 Intensity
    
    Args:
        rgb_path: RGB 图像路径（字符串或 Path）
        polar_paths: 偏振图像路径字典（值可以是字符串或 Path 对象）
        image_processor: CLIP 图像处理器（从模型加载）
        device: 设备
    
    Returns:
        (pixel_values_rgb, pixel_values_polar) 元组
        - pixel_values_rgb: (1, 3, 336, 336) - LLaVA 1.5 CLIP 需要的尺寸
        - pixel_values_polar: (1, 3, 512, 512) - VAE 需要的尺寸，3通道
    """
    # 1. 处理 RGB 图像（使用 image_processor，LLaVA 1.5 会自动处理为 336x336）
    # ⚠️ 关键修复：不要手动 resize，完全交给 image_processor 处理
    # image_processor 已经在 load_trained_model 中配置为 336x336
    rgb_image = Image.open(rgb_path).convert("RGB")
    
    pixel_values_rgb = image_processor(
        rgb_image,
        return_tensors="pt"
    )["pixel_values"]  # (1, 3, 336, 336) for LLaVA 1.5
    
    # 验证输出尺寸（确保是 336x336，避免维度不匹配错误）
    actual_size = pixel_values_rgb.shape[-2:]
    if actual_size != (336, 336):
        print(f"⚠️  错误: RGB 图像尺寸为 {actual_size}，但 LLaVA v1.5 需要 336x336")
        print(f"  这会导致 RuntimeError: shapes cannot be multiplied")
        print(f"  请检查 image_processor 是否正确配置为 336x336")
        raise ValueError(
            f"RGB 图像尺寸不匹配: 期望 (336, 336)，实际 {actual_size}。"
            f"请确保 image_processor 已正确配置为 336x336（在 load_trained_model 中已自动设置）。"
        )
    
    # ⚠️ 多GPU适配：确定正确的设备（避免 meta device）
    def _get_real_device_from_model(fallback_device: str):
        if model is None:
            return fallback_device
        # 优先从实际参数上取非 meta 的设备
        candidates = []
        if hasattr(model, "get_model"):
            try:
                base_model = model.get_model()
                candidates.extend(list(base_model.parameters()))
            except Exception:
                pass
        candidates.extend(list(model.parameters()))
        for p in candidates:
            if p is not None and hasattr(p, "is_meta") and not p.is_meta:
                return p.device
        # 回退：如果 model.device 存在且不是 meta
        if hasattr(model, "device"):
            return model.device
        return fallback_device

    target_device = _get_real_device_from_model(device)
    
    # 如果 target_device 是字符串，确保格式正确
    if isinstance(target_device, str):
        if target_device == "cuda":
            target_device = "cuda:0"
    elif hasattr(target_device, 'index'):
        # torch.device 对象
        target_device = str(target_device)
    
    pixel_values_rgb = pixel_values_rgb.to(target_device)
    
    # 2. 处理偏振图像（与训练阶段保持一致）
    # [修复] 确保 polar_paths 中的值是 Path 对象（process_polar_images 需要 Path 对象）
    polar_paths_path = {}
    for name, path in polar_paths.items():
        if isinstance(path, str):
            polar_paths_path[name] = Path(path)
        else:
            polar_paths_path[name] = path
    
    # 使用 process_polar_images 转换为 4 通道物理参数
    physics_img = process_polar_images(polar_paths=polar_paths_path)  # (H, W, 4)，期望值范围[0, 1]
    # 通道顺序：[Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]
    
    # 🟢 关键修复：检查并自动归一化输入数据（防止 NaN 溢出）
    img_min, img_max = physics_img.min(), physics_img.max()
    
    # 如果最大值超过 10，说明可能是 0-255 范围，需要归一化
    if img_max > 10.0:
        print(f"  ⚠️ 检测到输入未归一化 (Max={img_max:.2f})，正在执行 / 255.0 归一化...")
        physics_img = physics_img / 255.0
        img_min, img_max = physics_img.min(), physics_img.max()
        print(f"  ✓ 归一化后范围 - Min: {img_min:.4f}, Max: {img_max:.4f}")
    
    # 额外的安全检查：裁剪异常值到 [0, 1] 范围
    if img_min < -0.1 or img_max > 1.1:
        print(f"  ⚠️ 检测到异常值范围 [{img_min:.4f}, {img_max:.4f}]，正在裁剪到 [0, 1]...")
        physics_img = np.clip(physics_img, 0.0, 1.0)
        print(f"  ✓ 裁剪后范围 - Min: {physics_img.min():.4f}, Max: {physics_img.max():.4f}")
    
    # ⚠️ 关键修改：只取后3个通道（去掉 Intensity）
    # 与训练阶段保持一致：dataset.py 中的 _load_polar_images 只取 [DoLP, sin(2*AoLP), cos(2*AoLP)]
    physics_img_3ch = physics_img[:, :, 1:4]  # (H, W, 3)，值范围[0, 1]
    
    # 转换为 PIL Image（RGB 模式，3通道）
    physics_img_uint8 = (physics_img_3ch * 255).astype(np.uint8)
    physics_pil = Image.fromarray(physics_img_uint8, mode='RGB')
    
    # ⚠️ 关键修改：Polar 图像需要保持 512x512（VAE 编码器要求）
    # 🟢 [关键修复] 在预处理阶段显式归一化到 [-1, 1]，确保与 SD-VAE 的预训练分布一致
    # SD-VAE 在 [-1, 1] 范围内训练：-1.0=黑色, 0.0=灰色, 1.0=白色
    # 虽然 llava_arch.py 的 encode_polar_images 中也有转换，但为了保险起见，在预处理阶段就做好
    # 这样可以确保无论模型内部是否转换，数据都是正确的范围
    # 注意：如果模型内部也有转换，会导致重复归一化，所以需要检查并移除模型内部的转换
    polar_transform = transforms.Compose([
        transforms.Resize((512, 512)),  # VAE 需要的尺寸：512x512
        transforms.ToTensor(),  # 转换为 [0, 1] 范围的张量
        # 🟢 [关键修复] 强制归一化到 [-1, 1] 以匹配 SD-VAE 的预训练分布
        # (x - 0.5) / 0.5 = 2x - 1 -> [0, 1] -> [-1, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    pixel_values_polar = polar_transform(physics_pil)  # (3, 512, 512)，值范围[-1, 1]
    pixel_values_polar = pixel_values_polar.unsqueeze(0)  # (1, 3, 512, 512)
    pixel_values_polar = pixel_values_polar.to(target_device)
    
    # 🟢 验证归一化后的范围（应该是 [-1, 1]）
    polar_min, polar_max = pixel_values_polar.min().item(), pixel_values_polar.max().item()
    if polar_min < -1.1 or polar_max > 1.1:
        print(f"  ⚠️ 警告: Polar 归一化后范围异常！预期 [-1, 1]，实际 [{polar_min:.4f}, {polar_max:.4f}]")
    
    # 确保 dtype 匹配（VAE 需要 FP32）
    # 注意：polar_values_polar 已经是 [-1, 1] 范围，直接传递给 VAE
    pixel_values_polar = pixel_values_polar.to(dtype=torch.float32)
    
    return pixel_values_rgb, pixel_values_polar


def format_prompt(question: str, conversation=None, training_stage: str = "stage2") -> str:
    """
    格式化提示，根据训练阶段自动选择格式
    
    🟢 关键修复：根据训练阶段选择正确的格式
    - Stage 1: 使用 plain 格式（无 System Prompt，极简格式：<image>Question\nAnswer\n）
    - Stage 2: 使用 v1 格式（有 System Prompt，完整对话格式）
    
    这是 LLaVA 标准做法：
    - Stage 1: 特征对齐，只需要极简格式，System Prompt 会干扰 Projector 学习视觉特征
    - Stage 2: 对话微调，需要完整的对话格式（包括 System Prompt）
    
    Args:
        question: 用户问题（不包含 <image> token）
        conversation: 对话对象（如果为 None，使用默认对话）
        training_stage: 训练阶段（"stage1" 或 "stage2"），默认 "stage2"
    
    Returns:
        格式化后的提示字符串
    """
    if training_stage == "stage1":
        # 🟢 Stage 1: 使用 plain 格式（无 System Prompt）
        # 🟢 关键修复：与训练时完全一致！
        # preprocess_plain 会强制把 human 内容替换为 DEFAULT_IMAGE_TOKEN，
        # 训练时输入实际为：<image> + assistant_answer + sep
        # 因此推理时只应提供 <image>，不要附加额外文本（如问题描述）
        prompt = DEFAULT_IMAGE_TOKEN
        return prompt
    else:
        # 🟢 Stage 2: 使用 v1 格式（有 System Prompt）
        # 使用 conversation template 生成 prompt，与训练时完全一致
        if conversation is None:
            conversation = conversation_lib.default_conversation.copy()
        
        # 重置对话
        conversation.messages = []
        
        # 🟢 关键：确保格式与训练时完全一致
        # 训练时（preprocess_multimodal）的格式：DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
        # 所以这里也使用相同的格式：<image>\n{question}
        user_message = DEFAULT_IMAGE_TOKEN + "\n" + question.strip()
        
        # 添加用户消息（包含图像 token）
        conversation.append_message(conversation.roles[0], user_message)
        conversation.append_message(conversation.roles[1], None)
        
        # 🟢 使用 get_prompt() 生成完整 prompt（包含 System Prompt，与训练时一致）
        prompt = conversation.get_prompt()
        
        # 🟢 验证：检查 System Prompt 是否正确添加
        system_prompt_in_prompt = conversation.system in prompt[:200] if conversation.system else False
        if not system_prompt_in_prompt:
            print(f"  ⚠️  警告: Stage 2 应该包含 System Prompt，但未检测到")
        return prompt


def format_bbox_string(bbox_norm: List[float]) -> str:
    """格式化归一化 bbox 为字符串，用于 prompt"""
    return f"[{bbox_norm[0]:.3f}, {bbox_norm[1]:.3f}, {bbox_norm[2]:.3f}, {bbox_norm[3]:.3f}]"


def generate_qa_questions(bbox_norm: Optional[List[float]] = None) -> Dict[str, str]:
    """
    生成多种类型的问题模板，自动带入当前选定的 bbox 坐标。

    Args:
        bbox_norm: 归一化坐标 [xmin, ymin, xmax, ymax]，如果为 None，则使用整图 [0, 0, 1, 1]

    Returns:
        包含多种类型问题的字典：
        - content: 描述该区域的可见内容
        - detail: 描述该区域物体的颜色和材质
        - spatial: 描述该区域附近有什么物体
        - behind: 透视分析：忽略表面反射，透过眩光识别背后的物理结构
        - contour: 描述被反光遮挡物体的轮廓和物理特征
        - layer: 分析视觉层次，区分反射/眩光（前景）与真实物体（背景）
        - layer_analysis: 分层分析：Layer 1 反射场景 + Layer 2 实际物体
    """
    if bbox_norm is None:
        bbox_str = "[0.000, 0.000, 1.000, 1.000]"
    else:
        bbox_str = format_bbox_string(bbox_norm)

    # 为了避免回答过长，这里统一要求用 1-2 句简短英文回答
    brevity_suffix = " Answer in no more than two short, concise sentences."

    questions = {
        "content": (
            f"Focus on region {bbox_str}. Please briefly describe the visual content within this region."
            f"{brevity_suffix}"
        ),
        "detail": (
            f"Focus on region {bbox_str}. What are the color and material of the object in this region?"
            f"{brevity_suffix}"
        ),
        "spatial": (
            f"Focus on region {bbox_str}. What object is located near this region?"
            f"{brevity_suffix}"
        ),
        "behind": (
            f"Focus on region {bbox_str}. Ignore the surface reflection. Look through the glare to identify the physical structure underneath. What object is located behind this region?"
            f"{brevity_suffix}"
        ),
        "contour": (
            f"Focus on region {bbox_str}. Describe the visible contours and physical characteristics of the object that is obscured by the reflection in this region."
            f"{brevity_suffix}"
        ),
        "layer": (
            f"Focus on region {bbox_str}. Analyze the visual layers in this region. Distinctly describe what constitutes the reflection/glare (foreground) versus the actual physical object (background). Do not mix them up."
            f"{brevity_suffix}"
        ),
        "layer_analysis": (
            f"Focus on region {bbox_str}. Analyze the visual layers. Layer 1: What scene is being reflected on the surface? Layer 2: What is the actual object underneath the reflection? Describe them separately. Provide a clear and complete description for each layer."
        ),
    }

    return questions


def verify_model_config(model, tokenizer):
    """
    验证模型配置是否正确设置（支持 Stage 1 和 Stage 2）
    返回 (is_valid, error_messages, warnings)
    """
    errors = []
    warnings = []
    
    # 检查 training_stage
    training_stage = None
    if hasattr(model, 'config'):
        training_stage = getattr(model.config, 'training_stage', None)
        if training_stage not in ["stage1", "stage2"]:
            errors.append(f"training_stage 应为 'stage1' 或 'stage2'，但实际为 '{training_stage}'")
        else:
            pass
    else:
        errors.append("model.config 不存在")
    
    # 🟢 Stage 1/2：不使用分隔 token，默认 residual 融合
    fusion_mode = None
    if hasattr(model, "config"):
        fusion_mode = getattr(model.config, "polar_fusion_mode", None)
    fusion_desc = "residual" if fusion_mode != "concat" else "concat"
    # 保留验证逻辑，但不输出常规日志
    
    # 检查对话模板（根据训练阶段）
    if hasattr(conversation_lib, 'default_conversation'):
        conv = conversation_lib.default_conversation
        if training_stage == "stage1":
            # Stage 1 应该使用 plain 模板
            if conv.version != "plain":
                warnings.append(f"Stage 1 对话模板版本应为 'plain'，但实际为 '{conv.version}'")
            else:
                pass
        elif training_stage == "stage2":
            # Stage 2 应该使用 v1 模板
            if conv.version != "v1":
                warnings.append(f"Stage 2 对话模板版本应为 'v1'，但实际为 '{conv.version}'")
            else:
                pass
    
    return len(errors) == 0, errors, warnings


def generate_response(
    model,
    tokenizer: AutoTokenizer,
    pixel_values_rgb: torch.Tensor,
    pixel_values_polar: torch.Tensor,
    question: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    # 🟢 测试参数：用于诊断 Polar 分支问题
    test_mode: str = "normal",  # "normal", "no_polar", "zero_polar", "swap_channels"
) -> str:
    """
    生成回答（LLaVA 1.5 架构）
    
    Args:
        model: PolarLlavaLlamaForCausalLM 模型
        tokenizer: 分词器
        pixel_values_rgb: RGB 图像张量 (1, 3, 336, 336) - LLaVA 1.5
        pixel_values_polar: 偏振图像张量 (1, 3, 512, 512) - 3通道：[DoLP, sin(2*AoLP), cos(2*AoLP)]
        question: 问题文本
        max_new_tokens: 最大生成 token 数
        temperature: 温度参数
        top_p: nucleus sampling 参数
        do_sample: 是否使用采样
        test_mode: 测试模式
            - "normal": 正常推理（默认）
            - "no_polar": 屏蔽 Polar 分支，只用 RGB（测试 A）
            - "zero_polar": Polar 输入全零（测试 B）
            - "swap_channels": 交换通道顺序（测试 C，需要修改 preprocess_images）
    
    Returns:
        生成的回答文本
    """
    # 🟢 设置调试环境变量（如果 test_mode 不是 normal，自动启用 Polar 分支调试）
    import os
    if test_mode != "normal":
        os.environ['DEBUG_POLAR_BRANCH'] = '1'
    
    # 🔍 关键验证：在生成前验证模型配置
    # 从模型配置中获取训练阶段
    is_valid, errors, warnings = verify_model_config(model, tokenizer)
    if errors:
        print(f"  ❌ 发现 {len(errors)} 个错误:")
        for err in errors:
            print(f"     - {err}")
        print(f"  ⚠️  这些错误可能导致特征错位、乱码或空回答！")
    if warnings:
        print(f"  ⚠️  发现 {len(warnings)} 个警告:")
        for warn in warnings:
            print(f"     - {warn}")
    
    # 格式化提示（根据训练阶段自动选择格式）
    # 🟢 关键：确保格式与训练时完全一致
    # 优先使用模型配置中的 training_stage，如果不存在则使用传入的参数
    training_stage_for_prompt = "stage2"  # 默认 Stage 2
    if hasattr(model, 'config'):
        training_stage_for_prompt = getattr(model.config, 'training_stage', training_stage_for_prompt)
    
    prompt = format_prompt(question, training_stage=training_stage_for_prompt)
    
    # 仅在系统提示词缺失时输出警告
    if len(prompt) > 0 and hasattr(conversation_lib, 'default_conversation'):
        conv = conversation_lib.default_conversation
        expected_system = conv.system
        if expected_system and expected_system not in prompt:
            print(f"  ❌ 严重错误: System Prompt 不匹配！这会导致模型输出异常！")
            print(f"     预期 System Prompt: {expected_system}")
            print(f"     实际 prompt 开头: {prompt[:100]}...")
            print(f"     ⚠️  如果训练时没有 System Prompt，推理时也不应该有！")
    
    # 确定模型所在的设备
    if hasattr(model, 'device'):
        model_device = model.device
    elif hasattr(model, 'get_model'):
        base_model = model.get_model()
        first_param = next(base_model.parameters(), None)
        model_device = first_param.device if first_param is not None else pixel_values_rgb.device
    else:
        first_param = next(model.parameters(), None)
        model_device = first_param.device if first_param is not None else pixel_values_rgb.device
    
    # 使用 tokenizer_image_token 编码
    input_ids = tokenizer_image_token(
        prompt, tokenizer, return_tensors='pt'
    ).unsqueeze(0).to(model_device)

    # 🟢 关键修复：混合精度策略
    # RGB 图像转 FP16 (匹配 LLM 和 mm_projector)
    target_dtype_rgb = torch.float16
    if pixel_values_rgb is not None:
        pixel_values_rgb = pixel_values_rgb.to(device=model_device, dtype=target_dtype_rgb)
    
    # 🟢 测试模式：用于诊断 Polar 分支问题
    # 仅在 debug 模式下输出一次提示，便于确认当前评测模式
    if os.getenv('DEBUG_POLAR_BRANCH', '0') == '1':
        if '_test_mode_notice' not in globals():
            globals()['_test_mode_notice'] = False
        if not globals()['_test_mode_notice']:
            if test_mode == "no_polar":
                print("🧪 test_mode=no_polar: polar_images=None (RGB-only)")
            elif test_mode == "zero_polar":
                print("🧪 test_mode=zero_polar: polar_images=all zeros (Polar branch on)")
            elif test_mode == "swap_channels":
                print("🧪 test_mode=swap_channels: polar channel order swapped")
            else:
                print("🧪 test_mode=normal: polar_images provided (default)")
            globals()['_test_mode_notice'] = True

    if test_mode == "no_polar":
        # 测试 A：屏蔽 Polar 分支，只用 RGB
        pixel_values_polar = None
    elif test_mode == "zero_polar":
        # 测试 B：Polar 输入全零（验证 Projector 噪声）
        if pixel_values_polar is not None:
            pixel_values_polar = torch.zeros_like(pixel_values_polar).to(device=model_device, dtype=torch.float32)
    elif test_mode == "swap_channels":
        # 测试 C：交换通道顺序（测试通道顺序是否匹配）
        pass
    
    # 🟢 Polar 图像转 FP32 (匹配 VAE，防止数值溢出)
    # 🟢 [关键修复] Polar 图像已经在 preprocess_images 中归一化到 [-1, 1]
    # 直接传递给 VAE，不需要在 encode_polar_images 中再次转换
    # 这样确保数据在进入模型前就已经是正确的范围，不依赖模型内部的转换逻辑
    # 注意：llava_arch.py 的 encode_polar_images 会检查输入范围，如果已经是 [-1, 1] 则跳过转换
    if pixel_values_polar is not None:
        # 验证范围是否正确（仅保留必要的警告）
        polar_min, polar_max = pixel_values_polar.min().item(), pixel_values_polar.max().item()
        if polar_min < -1.1 or polar_max > 1.1:
            print(f"  ⚠️ 警告: Polar 输入范围异常！预期 [-1, 1]，实际 [{polar_min:.4f}, {polar_max:.4f}]")
            print(f"     这可能导致 VAE 编码异常，因为 SD-VAE 期望 [-1, 1] 范围的输入")
        
        pixel_values_polar = pixel_values_polar.to(device=model_device, dtype=torch.float32)
        
        # 🟢 NaN/Inf 清洗 (防止输入本身有问题)
        if torch.isnan(pixel_values_polar).any() or torch.isinf(pixel_values_polar).any():
            nan_count = torch.isnan(pixel_values_polar).sum().item()
            inf_count = torch.isinf(pixel_values_polar).sum().item()
            print(f"  ⚠️ 警告: Polar 输入包含 NaN ({nan_count} 个) / Inf ({inf_count} 个)，已自动替换为安全值")
            pixel_values_polar = torch.nan_to_num(pixel_values_polar, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 生成回答
    # 🟢 关键修复：确保 pad_token_id 和 eos_token_id 有效（防止 CUDA 设备端断言错误）
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    
    # 验证 token IDs
    if pad_token_id is None:
        print("  ⚠️ 警告: pad_token_id 为 None，使用 eos_token_id 作为替代")
        pad_token_id = eos_token_id
    
    if eos_token_id is None:
        raise ValueError("eos_token_id 不能为 None！")
    
    # 验证 token IDs 是否在有效范围内（防止 CUDA 设备端断言错误）
    vocab_size = len(tokenizer)
    if pad_token_id >= vocab_size:
        print(f"  ⚠️ 警告: pad_token_id ({pad_token_id}) 超出词汇表大小 ({vocab_size})，使用 eos_token_id")
        pad_token_id = eos_token_id
    if eos_token_id >= vocab_size:
        raise ValueError(f"eos_token_id ({eos_token_id}) 超出词汇表大小 ({vocab_size})！")
    
    
    # 🟢 生成回答 (带自动降级和错误处理)
    # 在生成前再次检查输入是否有 NaN/Inf（双重保险）
    if pixel_values_rgb is not None:
        if torch.isnan(pixel_values_rgb).any() or torch.isinf(pixel_values_rgb).any():
            print("  ⚠️ 警告: RGB 输入包含 NaN/Inf，已自动清理")
            pixel_values_rgb = torch.nan_to_num(pixel_values_rgb, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 🟢 计算最小新token数：确保至少生成一定数量的新 token（解决空回答问题）
    input_length = input_ids.shape[1]
    # 🔍 调试：根据融合方式估算视觉 token 数
    fusion_mode = None
    if hasattr(model, "config"):
        fusion_mode = getattr(model.config, "polar_fusion_mode", None)
    if fusion_mode == "concat":
        estimated_vision_tokens = 1152  # RGB 576 + Polar 576
    else:
        # residual 或未提供 polar 都是 576
        estimated_vision_tokens = 576
    min_new_tokens_required = 10  # 🟢 至少生成10个新token，防止立即结束（提高阈值）
    
    try:
        # 🟢 关键修复：确保模型参数 dtype 统一为 FP16（避免 float vs half）
        # 只在首次调用时执行，避免每次生成都遍历参数
        if not getattr(model, "_force_fp16_done", False):
            converted = 0
            for p in model.parameters():
                if p is not None and p.dtype.is_floating_point and p.dtype != torch.float16:
                    p.data = p.data.to(dtype=torch.float16)
                    converted += 1
            for b in model.buffers():
                if b is not None and b.dtype.is_floating_point and b.dtype != torch.float16:
                    b.data = b.data.to(dtype=torch.float16)
            model._force_fp16_done = True
            if converted > 0:
                pass

        # 🟢 统一处理采样策略：
        # - 当 temperature <= 0 时，自动退化为 Greedy（do_sample=False, temperature=None）
        effective_do_sample = do_sample
        effective_temperature = temperature
        if effective_temperature is None or effective_temperature <= 0.0:
            effective_do_sample = False
            effective_temperature = 0.0

        with torch.no_grad():
            # 🟢 测试模式：根据 test_mode 决定是否传递 polar_images
            polar_images_for_generate = pixel_values_polar if test_mode != "no_polar" else None
            
            # 🟢 关键修复：开启 autocast，确保输入/中间态与 LoRA 权重 dtype 对齐（避免 float vs half）
            model_dtype = None
            try:
                model_dtype = next(model.parameters()).dtype
            except Exception:
                model_dtype = torch.float16
            use_autocast = (model_device.type == "cuda") and (model_dtype in (torch.float16, torch.bfloat16))
            autocast_ctx = torch.autocast(device_type="cuda", dtype=model_dtype) if use_autocast else nullcontext()

            with autocast_ctx:
                generated_ids = model.generate(
                    inputs=input_ids,
                    images=pixel_values_rgb,
                    polar_images=polar_images_for_generate,  # 🟢 测试模式：可能为 None
                    max_new_tokens=max_new_tokens,
                    temperature=effective_temperature,
                    top_p=top_p if effective_do_sample else None,
                    do_sample=effective_do_sample,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )
            
    except RuntimeError as e:
        error_str = str(e)
        if "device-side assert" in error_str or "probability" in error_str or "multinomial" in error_str or "inf" in error_str.lower() or "nan" in error_str.lower():
            print("\n  ❌ 捕获到数值不稳定错误 (NaN/Inf Logits)")
            print("  🔄 尝试使用 Greedy Search (do_sample=False) 重试...")
            try:
                # 🟢 强制关闭采样，只取最大概率，通常能绕过 NaN
                with torch.no_grad():
                    # 🟢 测试模式：根据 test_mode 决定是否传递 polar_images
                    polar_images_for_generate = pixel_values_polar if test_mode != "no_polar" else None
                    
                    generated_ids = model.generate(
                        inputs=input_ids,
                        images=pixel_values_rgb,
                        polar_images=polar_images_for_generate,  # 🟢 测试模式：可能为 None
                        max_new_tokens=max_new_tokens,
                        do_sample=False,  # 强制 Greedy Search
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                    )
            except RuntimeError as e2:
                # 如果 Greedy Search 也失败，返回错误信息
                print(f"  ❌ Greedy Search 也失败: {e2}")
                print("  💡 建议：检查 VAE 输出是否有异常值，或尝试使用不同的图像")
                raise RuntimeError(f"数值不稳定错误，即使使用 Greedy Search 也无法恢复。原始错误: {e}, 重试错误: {e2}")
        else:
            # 其他错误直接抛出
            raise e
    
    # 提取回答（更稳健：先 decode 全序列，再按对话分隔符/角色切分）
    # 说明：不同模型/自定义 generate 实现可能返回“仅新 token”或“prompt+新 token”，
    # 直接用 input_length 切片会在某些情况下把回答前半段截断，造成空串/残句。
    try:
        decoded_full = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    except Exception:
        decoded_full = ""

    response_text = (decoded_full or "").strip()

    # 优先用角色标记切分（v1 模板通常包含 "ASSISTANT:"）
    if response_text:
        if "ASSISTANT:" in response_text:
            response_text = response_text.split("ASSISTANT:")[-1].strip()
        elif "Assistant:" in response_text:
            response_text = response_text.split("Assistant:")[-1].strip()

    # 再按 conversation 的分隔符切分（与 eval_rgb_baseline.py 的做法一致）
    if response_text and hasattr(conversation_lib, "default_conversation"):
        try:
            conv = conversation_lib.default_conversation
            sep = getattr(conv, "sep", None)
            sep2 = getattr(conv, "sep2", None)
            sep_style = getattr(conv, "sep_style", None)
            # SeparatorStyle.TWO 时优先按 sep2 切
            if sep_style is not None and str(sep_style).endswith("TWO") and sep2:
                response_text = response_text.split(sep2)[-1].strip()
            elif sep:
                response_text = response_text.split(sep)[-1].strip()
        except Exception:
            pass

    # 兜底：如果仍为空，尝试按 input_length 切片 decode “新 token”
    if not response_text:
        try:
            generated_length = generated_ids.shape[1]
            input_length = input_ids.shape[1]
            new_token_ids = generated_ids[0][input_length:] if generated_length >= input_length else generated_ids[0]
            response_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        except Exception:
            response_text = ""
    
    # 🟢 关键修复：更积极的文本清理和格式符号检测
    if response_text:
        import re
        # 🟢 新增：检测并截断格式符号（在清理之前）
        # 检测常见的格式符号模式（Markdown、代码块等）
        format_patterns = [
            r'\n\n\n>',  # 引用块开始
            r'\n```',    # 代码块开始
            r'\n- ',     # 列表项开始
            r'\n\* ',    # 列表项开始（星号）
            r'\n\\_\\_', # 分隔线
        ]
        
        # 找到第一个格式符号的位置
        first_format_pos = len(response_text)
        for pattern in format_patterns:
            match = re.search(pattern, response_text)
            if match and match.start() < first_format_pos:
                first_format_pos = match.start()
        
        # 如果找到格式符号，截断到该位置之前
        if first_format_pos < len(response_text):
            response_text = response_text[:first_format_pos].strip()
        
        # 移除可能的特殊 token
        special_tokens = ["<|end_of_text|>", "<|endoftext|>", "<|eot_id|>", "</s>", "<s>", "</s>"]
        for token in special_tokens:
            response_text = response_text.replace(token, "")
        
        # 🟢 新增：移除HTML/XML标签和格式标记
        # 移除所有HTML/XML标签（如 <span>, <div> 等）
        response_text = re.sub(r'<[^>]+>', '', response_text)
        
        # 🟢 新增：移除Markdown格式标记（如 `> ` 开头的引用块）
        response_text = re.sub(r'^>\s*', '', response_text, flags=re.MULTILINE)
        
        # 🟢 新增：移除代码块标记
        response_text = re.sub(r'```[^`]*```', '', response_text)  # 移除完整的代码块
        response_text = re.sub(r'`[^`]*`', '', response_text)  # 移除行内代码
        
        # 🟢 新增：移除列表标记
        response_text = re.sub(r'^\s*[-*+]\s+', '', response_text, flags=re.MULTILINE)
        
        # 🟢 新增：移除分隔线
        response_text = re.sub(r'_{3,}', '', response_text)  # 移除多个下划线
        response_text = re.sub(r'-{3,}', '', response_text)  # 移除多个连字符
        
        # 🟢 新增：移除开头的异常符号（如 `)`, `>`, `'` 等）
        response_text = re.sub(r'^[)\'>\s\n]+', '', response_text)
        
        # 🟢 新增：移除结尾的异常符号
        response_text = re.sub(r'[)\'>\s\n]+$', '', response_text)
        
        # 🟢 新增：清理多余的换行和空白
        response_text = re.sub(r'\n{3,}', '\n\n', response_text)  # 多个换行合并为两个
        response_text = re.sub(r'[ \t]+', ' ', response_text)  # 多个空格合并为一个
        
        # 🟢 检测并清理重复模式（防止模型陷入循环）
        # 对任何长度的回答都进行检测，降低阈值以适应温度采样
        words = response_text.split()
        if len(words) >= 6:  # 至少6个词才进行检测
            from collections import Counter
            # 检查 2-gram 和 3-gram 的重复
            ngrams_2 = [tuple(words[i:i+2]) for i in range(len(words)-1)]
            ngrams_3 = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            ngram_counts_2 = Counter(ngrams_2)
            ngram_counts_3 = Counter(ngrams_3)
            max_repeat_2 = max(ngram_counts_2.values()) if ngram_counts_2 else 0
            max_repeat_3 = max(ngram_counts_3.values()) if ngram_counts_3 else 0
            
            # 🟢 降低阈值：温度采样时，重复可能更频繁，但超过3次仍可能是循环
            threshold = 3 if do_sample else 5  # 采样时阈值更低
            
            if max_repeat_2 > threshold or max_repeat_3 > threshold:
                # 找到第一个重复的 n-gram 的位置
                for i, ngram in enumerate(ngrams_3 if max_repeat_3 > threshold else ngrams_2):
                    ngram_counts = ngram_counts_3 if max_repeat_3 > threshold else ngram_counts_2
                    if ngram_counts[ngram] > threshold:
                        # 截取到第一个重复位置
                        ngram_len = 3 if max_repeat_3 > threshold else 2
                        response_text = " ".join(words[:i+ngram_len])
                        break
        
        response_text = response_text.strip()
        
        # 🟢 新增：如果清理后的文本仍然很奇怪（主要是符号和格式），尝试提取有意义的文本
        if len(response_text) > 0:
            # 计算字母和数字的比例
            alnum_count = sum(1 for c in response_text if c.isalnum())
            total_chars = len(response_text.replace(' ', '').replace('\n', ''))
            if total_chars > 0:
                alnum_ratio = alnum_count / total_chars
                if alnum_ratio < 0.3:  # 如果字母和数字占比小于30%，说明主要是符号
                    # 尝试提取包含字母的单词
                    words = response_text.split()
                    words_meaningful = [w for w in words if any(c.isalnum() for c in w)]
                    if len(words_meaningful) > 0:
                        response_text = " ".join(words_meaningful)
                    else:
                        response_text = ""  # 如果完全没有有意义的文本，返回空
    
    # 🟢 如果回答为空或只有特殊字符，返回提示信息而不是空字符串
    if not response_text or response_text in ["</s>", "<|end_of_text|>", "<|endoftext|>"]:
        print("  ⚠️ 警告: 模型生成空回答或只有结束符")
        return ""  # 保持返回空字符串，但记录警告
    
    return response_text


def main():
    parser = argparse.ArgumentParser(description="PolarVLM Stage 2 推理脚本（LLaVA 1.5 架构）")
    
    # 模型配置
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="检查点目录（包含 adapter_model.safetensors, non_lora_trainables.bin 等，支持 Stage 1 和 Stage 2）")
    parser.add_argument("--llm_model_name", type=str,
                        default="/openbayes/input/input0/models/llava-v1.5-13b",
                        help="基础 LLM 模型路径（LLaVA 1.5 完整模型路径）")
    parser.add_argument("--clip_model_name", type=str, default="/openbayes/input/input0/models/clip-vit-large-patch14-336",
                        help="CLIP 模型路径（LLaVA 1.5 使用 336 版本）")
    parser.add_argument("--vae_model_path", type=str, default="/openbayes/input/input0/models/sd-vae-ft-mse",
                        help="VAE 模型路径（用于加载 VAE 编码器）。如果提供此路径，--polar_backbone 将被忽略。")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face token")
    parser.add_argument("--training_stage", type=str, default=None,
                        choices=["stage1", "stage2"],
                        help="手动指定训练阶段（'stage1' 或 'stage2'）。如果未指定，将自动从 checkpoint 目录名或配置文件中检测。")
    
    # 输入模式选择（三种模式互斥）
    # ⚠️ 注意：不能使用 required=True，因为还支持第三种模式（scene_id + base_name）
    # 这里不强制要求，在后续代码中手动检查三种模式必须至少选择一种
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--glare_bbox_json", type=str, default=None,
                             help="反光框检测 JSON 文件路径（包含 rgb_path, polar_paths, bbox_norm 等）")
    input_group.add_argument("--rgb_path", type=str, default=None,
                             help="RGB 图像完整路径（用于任意图片推理）")
    
    # Stage 2 训练集 JSON 批量验证模式
    parser.add_argument("--dataset_json", type=str, default=None,
                        help="Stage 2 训练数据 JSON（如 merged_stage3_qwen.json），用于批量验证")
    parser.add_argument("--max_images_per_scene", type=int, default=None,
                        help="每个场景最多验证的图片数量（仅在 --dataset_json 模式下生效）")
    
    # 传统模式参数（当使用 --scene_id 和 --base_name 时）
    parser.add_argument("--rgb_root", type=str, default="/openbayes/input/input0/rgb",
                        help="RGB 图像根目录（传统模式）")
    parser.add_argument("--polar_root", type=str, default="/openbayes/input/input0/polar",
                        help="偏振图像根目录（传统模式或任意图片模式）")
    parser.add_argument("--scene_id", type=str, default=None,
                        help="场景 ID（如 '10'，传统模式）")
    parser.add_argument("--base_name", type=str, default=None,
                        help="基础文件名（不含扩展名，如 '0002'，传统模式）")
    
    # 任意图片模式参数
    parser.add_argument("--polar_paths", type=str, nargs=4, default=None,
                        metavar=("I_0", "I_45", "I_90", "I_135"),
                        help="偏振图像路径（4个角度，任意图片模式）")
    parser.add_argument("--bbox_norm", type=float, nargs=4, default=None,
                        metavar=("xmin", "ymin", "xmax", "ymax"),
                        help="归一化反光框坐标 [xmin, ymin, xmax, ymax]（任意图片模式）")
    
    # 问题设置
    parser.add_argument("--question", type=str, default=None,
                        help="问题文本（如果未指定，将根据 bbox_norm 生成三类问题）")
    parser.add_argument("--qa_types", type=str, nargs="+", 
                        choices=["content", "detail", "spatial", "behind", "contour", "layer", "all"],
                        default=["all"],
                        help="要生成的问题类型（默认：all，生成所有问题类型）")
    
    # 数据根目录（用于解析相对路径）
    parser.add_argument("--data_root", type=str, default="/openbayes/input/input0",
                        help="数据根目录（用于解析相对路径）")
    
    # 生成参数
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="温度参数")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="nucleus sampling 参数")
    parser.add_argument("--do_sample", action="store_true", default=True,
                        help="使用采样（默认开启）")
    
    # 设备
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备（cuda 或 cpu）")
    
    args = parser.parse_args()
    
    # ========== 关键修复 3: VAE 路径的默认值逻辑 ==========
    # 确保 vae_model_path 不为空字符串或 None
    vae_path = args.vae_model_path if (args.vae_model_path and args.vae_model_path.strip()) else '/openbayes/input/input0/models/sd-vae-ft-mse'
    
    # 加载模型
    # 🟢 默认使用 FP16 全精度（双卡 4090 推荐）
    model, tokenizer, image_processor, detected_training_stage = load_trained_model(
        checkpoint_dir=args.checkpoint_dir,
        llm_model_name=args.llm_model_name,
        clip_model_name=args.clip_model_name,
        vae_model_path=vae_path,
        hf_token=args.hf_token,
        device=args.device,
        load_4bit=False,  # 🟢 默认关闭 4-bit，使用 FP16 全精度
        use_multi_gpu=True,  # 🟢 默认启用多GPU（检测到多卡时）
        training_stage=args.training_stage,  # 🟢 支持手动指定或自动检测
    )
    
    # 🟢 使用检测到的训练阶段（用于后续的 prompt 格式化）
    training_stage_for_inference = detected_training_stage
    
    data_root = Path(args.data_root)
    
    # ========== 解析输入模式 ==========
    rgb_path = None
    polar_paths_dict = None
    bbox_norm = None
    
    # 模式0：直接从 Stage 2 训练集 JSON 批量验证
    if args.dataset_json:
        print("\n" + "=" * 80)
        print("模式：Stage 2 训练集 JSON 批量验证")
        print("=" * 80)

        dataset_path = Path(args.dataset_json)
        if not dataset_path.exists():
            raise FileNotFoundError(f"dataset_json 不存在: {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset_items = json.load(f)

        if not isinstance(dataset_items, list):
            raise ValueError(f"dataset_json 应该是列表格式，但得到: {type(dataset_items)}")

        total_items = len(dataset_items)
        print(f"检测到 {total_items} 条训练样本，将按配置进行批量验证（终端不再逐条打印回答，仅显示进度条）...")

        # 每个场景最多图片数量统计
        scene_image_count: Dict[str, int] = {}
        results: List[Dict] = []

        for idx, item in enumerate(tqdm(dataset_items, desc="Dataset JSON 验证", total=total_items)):
            scene_id = str(item.get("scene_id", "")).zfill(2)
            # 按场景过滤（如果提供了 scene_id）
            if args.scene_id and scene_id != args.scene_id:
                continue

            # 从 input_path 或 image 中获取 RGB 路径（Stage 2 训练数据格式）
            rgb_rel = item.get("input_path") or item.get("image")
            if not rgb_rel:
                continue

            # 解析 RGB 绝对路径（通常是 rgb/{scene}/{base_name}_rgb.png）
            rgb_path_abs = Path(rgb_rel)
            if not rgb_path_abs.is_absolute():
                rgb_path_abs = data_root / rgb_rel

            base_name = rgb_path_abs.stem
            if base_name.endswith("_rgb"):
                base_name = base_name[:-4]

            # 按 base_name 过滤（如果提供了 base_name）
            if args.base_name and base_name != args.base_name:
                continue

            # 每个场景的最大图片数控制
            if args.max_images_per_scene is not None:
                cur_cnt = scene_image_count.get(scene_id, 0)
                if cur_cnt >= args.max_images_per_scene:
                    continue
                scene_image_count[scene_id] = cur_cnt + 1

            if not rgb_path_abs.exists():
                print(f"Warning: RGB 图像不存在，跳过: {rgb_path_abs}")
                continue

            # 根据 scene_id 和 base_name 推导偏振图像路径
            polar_paths_dict = get_polar_paths(args.polar_root, scene_id, base_name)
            missing_polar = [name for name, path in polar_paths_dict.items() if not path.exists()]
            if missing_polar:
                print(f"Warning: 偏振图像缺失，跳过: {missing_polar}")
                continue

            bbox_norm_item = item.get("bbox_norm")

            # 生成问题：复用 generate_qa_questions 模板
            qa_questions = generate_qa_questions(bbox_norm_item)
            qa_types_to_process = args.qa_types
            if "all" in qa_types_to_process:
                qa_types_to_process = ["content", "detail", "spatial", "behind", "contour", "layer"]

            for qa_type in qa_types_to_process:
                if qa_type not in qa_questions:
                    continue

                question = qa_questions[qa_type]
                # 预处理图像
                pixel_values_rgb, pixel_values_polar = preprocess_images(
                    rgb_path=str(rgb_path_abs),
                    polar_paths=polar_paths_dict,
                    image_processor=image_processor,
                    device=args.device,
                    model=model,
                )

                # 生成回答
                response = generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    pixel_values_rgb=pixel_values_rgb,
                    pixel_values_polar=pixel_values_polar,
                    question=question,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=args.do_sample,
                )

                results.append(
                    {
                        "scene_id": scene_id,
                        "base_name": base_name,
                        "qa_type": qa_type,
                        "question": question,
                        "answer": response,
                        "bbox_norm": bbox_norm_item,
                        "rgb_path": str(rgb_path_abs),
                    }
                )

        # 保存批量结果
        output_file = dataset_path.with_suffix(".results.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Stage 2 批量验证结果已保存到: {output_file}")
        return

    # 模式1：从反光框检测 JSON 文件读取
    if args.glare_bbox_json:
        # 模式1：从反光框检测 JSON 文件读取
        print("\n" + "=" * 80)
        print("模式：从反光框检测 JSON 文件读取")
        print("=" * 80)
        
        with open(args.glare_bbox_json, 'r', encoding='utf-8') as f:
            glare_data = json.load(f)
        
        if isinstance(glare_data, list):
            # 如果是列表，处理所有条目
            print(f"检测到 {len(glare_data)} 条数据，将逐一处理（终端不再逐条打印回答，仅显示进度条）...")
            results = []
            
            for idx, item in enumerate(tqdm(glare_data, desc="Glare JSON 验证", total=len(glare_data))):
                # 解析路径
                rgb_path_rel = item.get("rgb_path", "")
                polar_paths_rel = item.get("polar_paths", {})
                bbox_norm_item = item.get("bbox_norm")
                
                # 转换为绝对路径
                if not Path(rgb_path_rel).is_absolute():
                    rgb_path = data_root / rgb_path_rel
                else:
                    rgb_path = Path(rgb_path_rel)
                
                polar_paths_dict = {}
                for name, path_rel in polar_paths_rel.items():
                    if not Path(path_rel).is_absolute():
                        polar_paths_dict[name] = data_root / path_rel
                    else:
                        polar_paths_dict[name] = Path(path_rel)
                
                # 检查路径
                if not rgb_path.exists():
                    print(f"Warning: RGB 图像不存在，跳过: {rgb_path}")
                    continue
                
                missing_polar = [name for name, path in polar_paths_dict.items() if not path.exists()]
                if missing_polar:
                    print(f"Warning: 偏振图像缺失，跳过: {missing_polar}")
                    continue
                
                # 生成问题
                if bbox_norm_item:
                    qa_questions = generate_qa_questions(bbox_norm_item)
                else:
                    qa_questions = generate_qa_questions(None)  # 全图问题
                
                # 确定要生成的问题类型
                qa_types_to_process = args.qa_types
                if "all" in qa_types_to_process:
                    qa_types_to_process = ["content", "detail", "spatial"]
                
                # 处理每个问题类型
                for qa_type in qa_types_to_process:
                    if qa_type not in qa_questions:
                        continue
                    
                    question = qa_questions[qa_type]
                    
                    # 预处理图像
                    # [修复] 保持 polar_paths 为 Path 对象（preprocess_images 内部会处理）
                    pixel_values_rgb, pixel_values_polar = preprocess_images(
                        rgb_path=str(rgb_path),
                        polar_paths=polar_paths_dict,  # 保持 Path 对象，不要转换为字符串
                        image_processor=image_processor,
                        device=args.device,
                        model=model,
                    )
                    
                    # 生成回答
                    response = generate_response(
                        model=model,
                        tokenizer=tokenizer,
                        pixel_values_rgb=pixel_values_rgb,
                        pixel_values_polar=pixel_values_polar,
                        question=question,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=args.do_sample,
                    )
                    
                    results.append({
                        "rgb_path": str(rgb_path),
                        "bbox_norm": bbox_norm_item,
                        "qa_type": qa_type,
                        "question": question,
                        "answer": response,
                    })
            
            # 保存结果
            output_file = Path(args.glare_bbox_json).with_suffix('.results.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n✓ 所有结果已保存到: {output_file}")
            
        else:
            # 单个条目
            rgb_path_rel = glare_data.get("rgb_path", "")
            polar_paths_rel = glare_data.get("polar_paths", {})
            bbox_norm = glare_data.get("bbox_norm")
            
            # 转换为绝对路径
            if not Path(rgb_path_rel).is_absolute():
                rgb_path = data_root / rgb_path_rel
            else:
                rgb_path = Path(rgb_path_rel)
            
            polar_paths_dict = {}
            for name, path_rel in polar_paths_rel.items():
                if not Path(path_rel).is_absolute():
                    polar_paths_dict[name] = data_root / path_rel
                else:
                    polar_paths_dict[name] = Path(path_rel)
    
    elif args.rgb_path:
        # 模式2：任意图片路径
        print("\n" + "=" * 80)
        print("模式：任意图片路径推理")
        print("=" * 80)
        
        rgb_path = Path(args.rgb_path)
        if not rgb_path.is_absolute():
            rgb_path = data_root / args.rgb_path
        
        if args.polar_paths:
            # 使用指定的偏振图像路径
            polar_paths_dict = {
                'I_0': Path(args.polar_paths[0]),
                'I_45': Path(args.polar_paths[1]),
                'I_90': Path(args.polar_paths[2]),
                'I_135': Path(args.polar_paths[3]),
            }
            # 转换为绝对路径
            for name, path in polar_paths_dict.items():
                if not path.is_absolute():
                    polar_paths_dict[name] = data_root / path
        else:
            raise ValueError("使用 --rgb_path 时必须指定 --polar_paths")
        
        bbox_norm = args.bbox_norm
    
    else:
        # 模式3：传统模式（scene_id + base_name）
        if not args.scene_id or not args.base_name:
            raise ValueError("必须指定 --glare_bbox_json、--rgb_path 或 --scene_id + --base_name")
        
        print("\n" + "=" * 80)
        print("模式：传统模式（scene_id + base_name）")
        print("=" * 80)
        
        rgb_path = Path(args.rgb_root) / args.scene_id / f"{args.base_name}_rgb.png"
        polar_paths_dict = get_polar_paths(Path(args.polar_root), args.scene_id, args.base_name)
    
    # ========== 检查路径 ==========
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB 图像不存在: {rgb_path}")
    
    missing_polar = [name for name, path in polar_paths_dict.items() if not path.exists()]
    if missing_polar:
        raise FileNotFoundError(
            f"偏振图像缺失: {missing_polar}\n"
            f"路径: {[str(polar_paths_dict[name]) for name in missing_polar]}"
        )
    
    # ========== 生成问题 ==========
    if args.question:
        # 使用指定的问题
        questions_to_process = [("custom", args.question)]
    else:
        # 根据 bbox_norm 生成三类问题
        qa_questions = generate_qa_questions(bbox_norm)
        qa_types_to_process = args.qa_types
        if "all" in qa_types_to_process:
            qa_types_to_process = ["content", "detail", "spatial"]
        
        questions_to_process = [(qa_type, qa_questions[qa_type]) 
                               for qa_type in qa_types_to_process 
                               if qa_type in qa_questions]
    
    # ========== 预处理图像 ==========
    print("\n正在预处理图像...")
    # [修复] 保持 polar_paths 为 Path 对象（preprocess_images 内部会处理）
    pixel_values_rgb, pixel_values_polar = preprocess_images(
        rgb_path=str(rgb_path),
        polar_paths=polar_paths_dict,  # 保持 Path 对象，不要转换为字符串
        image_processor=image_processor,
        device=args.device,
        model=model,
    )
    print("✓ 图像预处理完成")
    
    # ========== 生成回答 ==========
    print("\n" + "=" * 80)
    print("推理结果")
    print("=" * 80)
    print(f"RGB 图像: {rgb_path}")
    if bbox_norm:
        print(f"反光框坐标: {bbox_norm}")
    
    results = []
    for qa_type, question in questions_to_process:
        print(f"\n问题类型: {qa_type.upper()}")
        print(f"问题: {question}")
        
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            pixel_values_rgb=pixel_values_rgb,
            pixel_values_polar=pixel_values_polar,
            question=question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
        )
        
        print(f"回答: {response}")
        print("-" * 80)
        
        results.append({
            "qa_type": qa_type,
            "question": question,
            "answer": response,
            "bbox_norm": bbox_norm,
        })
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

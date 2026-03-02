"""
PolarLlava: 双流多模态大语言模型
结合RGB语义流和偏振物理流进行视觉语言理解
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import (
    CLIPVisionModel,
    CLIPVisionConfig,
    LlamaForCausalLM,
    LlamaConfig,
    ViTModel,
    ViTConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from diffusers import AutoencoderKL


def adapt_first_layer_for_4channels(
    model: nn.Module,
    layer_name: str = "conv1"  # ResNet的默认第一层名称，ViT可能是"patch_embed.proj"
) -> None:
    """
    适配第一层卷积/线性层以处理4通道输入（偏振图像）
    
    策略：
    1. 复制预训练权重（前3个通道，对应RGB）
    2. 第4个通道初始化为前3个通道的均值（保持激活尺度一致）
    
    这种方法可以充分利用预训练权重，在小数据集上实现快速收敛。
    
    Args:
        model: 预训练的视觉模型（ResNet或ViT）
        layer_name: 第一层的名称（根据模型架构不同而变化）
    """
    # 获取第一层
    if hasattr(model, layer_name):
        first_layer = getattr(model, layer_name)
    elif hasattr(model, "patch_embed") and hasattr(model.patch_embed, "proj"):
        # ViT的情况
        first_layer = model.patch_embed.proj
        layer_name = "patch_embed.proj"
    elif hasattr(model, "conv1"):
        # ResNet的情况
        first_layer = model.conv1
        layer_name = "conv1"
    else:
        # 尝试查找第一个卷积层
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and "conv" in name.lower():
                first_layer = module
                layer_name = name
                break
        else:
            raise ValueError(f"无法找到第一层卷积/线性层。请手动指定layer_name。")
    
    # 检查第一层的类型
    if isinstance(first_layer, nn.Conv2d):
        # 卷积层的情况
        out_channels, in_channels, kernel_h, kernel_w = first_layer.weight.shape
        
        if in_channels == 3:
            # 扩展权重：从3通道扩展到4通道
            new_weight = torch.zeros(
                out_channels, 4, kernel_h, kernel_w,
                device=first_layer.weight.device,
                dtype=first_layer.weight.dtype
            )
            
            # 复制前3个通道的权重
            new_weight[:, :3, :, :] = first_layer.weight.data.clone()
            
            # 第4个通道初始化为前3个通道的均值
            new_weight[:, 3, :, :] = first_layer.weight.data.mean(dim=1)
            
            # 创建新的卷积层
            new_conv = nn.Conv2d(
                in_channels=4,
                out_channels=out_channels,
                kernel_size=(kernel_h, kernel_w),
                stride=first_layer.stride,
                padding=first_layer.padding,
                bias=first_layer.bias is not None
            )
            
            # 复制偏置（如果有）
            if first_layer.bias is not None:
                new_conv.bias.data = first_layer.bias.data.clone()
            
            # 设置新权重
            new_conv.weight.data = new_weight
            
            # 替换原层
            if "." in layer_name:
                # 嵌套属性，需要逐层访问
                parts = layer_name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], new_conv)
            else:
                setattr(model, layer_name, new_conv)
            
            print(f"✓ 成功适配第一层 {layer_name}: 3通道 -> 4通道")
        else:
            print(f"⚠ 第一层 {layer_name} 的输入通道数不是3，跳过适配")
    
    elif isinstance(first_layer, nn.Linear):
        # 线性层的情况（较少见，但某些架构可能使用）
        out_features, in_features = first_layer.weight.shape
        
        # 假设输入是展平后的图像，需要根据实际情况调整
        # 这里提供一个通用框架
        if in_features % 3 == 0:
            # 假设输入是展平的RGB图像
            pixels_per_channel = in_features // 3
            new_in_features = pixels_per_channel * 4  # 扩展到4通道
            
            new_weight = torch.zeros(
                out_features, new_in_features,
                device=first_layer.weight.device,
                dtype=first_layer.weight.dtype
            )
            
            # 复制前3个通道的权重
            new_weight[:, :pixels_per_channel * 3] = first_layer.weight.data.clone()
            
            # 第4个通道初始化为前3个通道的均值
            for i in range(pixels_per_channel):
                new_weight[:, pixels_per_channel * 3 + i] = (
                    first_layer.weight.data[:, i] +
                    first_layer.weight.data[:, pixels_per_channel + i] +
                    first_layer.weight.data[:, pixels_per_channel * 2 + i]
                ) / 3.0
            
            # 创建新的线性层
            new_linear = nn.Linear(
                in_features=new_in_features,
                out_features=out_features,
                bias=first_layer.bias is not None
            )
            
            if first_layer.bias is not None:
                new_linear.bias.data = first_layer.bias.data.clone()
            
            new_linear.weight.data = new_weight
            
            # 替换原层
            if "." in layer_name:
                parts = layer_name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], new_linear)
            else:
                setattr(model, layer_name, new_linear)
            
            print(f"✓ 成功适配第一层 {layer_name}: 3通道 -> 4通道（线性层）")
        else:
            print(f"⚠ 第一层 {layer_name} 的输入特征数不是3的倍数，跳过适配")
    else:
        raise TypeError(f"不支持的第一层类型: {type(first_layer)}")


class MultiModalProjector(nn.Module):
    """
    多模态投影器：在特征维度融合 RGB 和 Polar 特征（空间对齐后）
    
    架构：
    - 输入：RGB 特征 (B, N, rgb_dim) 和 Polar 特征 (B, N, polar_dim)
    - 动态平衡参数：可学习的缩放因子，用于平衡 RGB 和 Polar 特征的重要性
    - 投影器：2层MLP (Linear -> GELU -> Linear)
    - LayerNorm：输出层归一化，提高训练稳定性
    - 输出：投影后的特征 (B, N, llm_hidden_dim)
    
    关键改进：
    - RGB 和 Polar 特征在空间对齐后（都变为 16x16），在特征维度拼接
    - 每个 token 同时包含对应位置的 RGB 和 Polar 信息
    - 降低学习难度，LLM 无需跨 token 学习对应关系
    - 可学习的缩放参数自动平衡两种特征的重要性
    """
    
    def __init__(
        self,
        rgb_feature_dim: int = 1024,  # CLIP ViT-L/14的特征维度
        polar_feature_dim: int = 768,  # ViT-Base的特征维度（固定，因为现在只支持 ViT）
        llm_hidden_dim: int = 4096,  # LLaMA-3-8B的隐藏维度（动态获取）
    ):
        super().__init__()
        
        # 输入维度是 RGB + Polar 的拼接
        self.input_dim = rgb_feature_dim + polar_feature_dim
        
        # [新增] 动态平衡参数：可学习的缩放因子，用于平衡 RGB 和 Polar 特征的重要性
        # 
        # 注意：由于在拼接后添加了 LayerNorm，数值差异问题已通过标准化解决。
        # 因此 rgb_scale 和 polar_scale 主要用于控制特征的重要性权重，而非解决数值差异。
        # 
        # 方案：使用可学习的缩放参数，让模型自动学习如何平衡两种特征
        # - rgb_scale: 控制 RGB 特征的缩放（初始化为 1.0）
        # - polar_scale: 控制 Polar 特征的缩放（初始化为 1.0）
        #   理由：Polar 标准差为 0.08，RGB 为 0.94。差距缩小到 10 倍左右，
        #         这是一个 LLM 可以接受的范围（Softmax 对 10 倍的差异还是有反应的）。
        #         且 LayerNorm 会进一步标准化，所以不需要过小的初始值。
        # 
        # 优势：
        # 1. 可学习：模型可以根据任务自动调整两种特征的重要性
        # 2. 简单：只有 2 个参数，不会增加太多计算开销
        # 3. 灵活：可以适应不同的数据分布和任务需求
        # 4. 自动保存：包含在 projector.state_dict() 中，保存和加载无需额外处理
        self.rgb_scale = nn.Parameter(torch.ones(1))  # 初始化为 1.0（RGB 特征保持原样）
        self.polar_scale = nn.Parameter(torch.ones(1) * 1.0)  # 初始化为 1.0（保持原样，LayerNorm 会处理数值差异）
        
        # [关键修改] 在拼接之前添加 LayerNorm，确保输入到 Linear 的特征已归一化
        # 这样可以抹平 RGB 和 Polar 特征的数值差异，即使 rgb_scale 和 polar_scale 不平衡也没关系
        self.input_layernorm = nn.LayerNorm(self.input_dim)
        
        # 统一的投影器（处理拼接后的特征）
        self.projector = nn.Sequential(
            nn.Linear(self.input_dim, llm_hidden_dim),
            nn.GELU(),
            nn.Linear(llm_hidden_dim, llm_hidden_dim),
        )
        
        # LayerNorm：归一化输出，提高训练稳定性
        self.output_layernorm = nn.LayerNorm(llm_hidden_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化投影器权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用Xavier初始化（对于GELU激活函数是合理的）
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm 使用默认初始化即可
                pass
    
    def forward(
        self,
        rgb_features: torch.Tensor,
        polar_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播：动态平衡并投影 RGB + Polar 特征
        
        Args:
            rgb_features: RGB 特征，形状为 (B, N, rgb_feature_dim)
                - 例如：(B, 256, 1024)
                - N 是统一的 patch 数量（16x16 = 256），RGB 和 Polar 已空间对齐
            polar_features: Polar 特征，形状为 (B, N, polar_feature_dim)
                - 例如：(B, 256, 768)
                - N 是统一的 patch 数量（16x16 = 256），RGB 和 Polar 已空间对齐
        
        Returns:
            投影后的特征，形状为 (B, N, llm_hidden_dim)
                - 例如：(B, 256, 4096)
        """
        # 1. 动态平衡：使用可学习的缩放参数来平衡 RGB 和 Polar 特征
        # 这一步非常关键，它解决了 "MAE 特征值太大，压制 CLIP 特征" 的问题
        # CLIP (RGB流) 输出特征：数值较小且集中（-1 到 1 之间），语义密度极高
        # MAE (Polar流) 输出特征：数值较大且发散（std=2.45，范围 -16 到 21）
        # 
        # 方案：使用可学习的缩放参数，让模型自动学习如何平衡两种特征
        # - rgb_scale: 控制 RGB 特征的缩放（初始化为 1.0）
        # - polar_scale: 控制 Polar 特征的缩放（初始化为 0.1，以平衡分布差异）
        rgb_features_scaled = rgb_features * self.rgb_scale  # (B, 256, 1024)
        polar_features_scaled = polar_features * self.polar_scale  # (B, 256, 768)
        
        # 2. 特征维度拼接（空间对齐后，使用缩放后的特征）
        # 现在的形状都是 (B, 256, D)
        # 拼接后: (B, 256, 1024+768) = (B, 256, 1792)
        combined_features = torch.cat([rgb_features_scaled, polar_features_scaled], dim=-1)
        
        # 3. [关键修改] 在输入到 Linear 之前应用 LayerNorm，抹平 RGB 和 Polar 特征的数值差异
        # 这样即使 rgb_scale 和 polar_scale 不平衡，LayerNorm 也会将输入标准化为均值 0、方差 1
        # 这是用户建议的改进：如果第一层是 LayerNorm，数值差异问题不大
        combined_features = self.input_layernorm(combined_features)  # (B, 256, 1792)
        
        # 4. 投影拼接后的特征
        projected_features = self.projector(combined_features)  # (B, 256, 4096)
        
        # 5. LayerNorm：归一化输出，提高训练稳定性
        projected_features = self.output_layernorm(projected_features)
        
        return projected_features


class PolarLlava(nn.Module):
    """
    PolarLlava: 双流多模态大语言模型
    
    架构：
    1. RGB流：冻结的CLIP ViT-L/14（提取语义特征）
    2. 偏振流：可训练的 transformers ViTModel（提取物理特征，输入为3通道：Intensity, DoLP, AoLP）
    3. 多模态投影器：将双流特征投影到LLM空间
    4. LLM：LLaMA-3-8B（量化 + LoRA）
    
    注意：
        - 偏振流使用 transformers.ViTModel，与 Stage 1 MAE 预训练兼容
        - 输入为3通道物理参数（从4个角度图像计算Stokes参数得到）
        - 可以直接使用标准ViT预训练权重（无需修改第一层）
        - 支持从 Stage 1 MAE 检查点加载编码器权重
    """
    
    def __init__(
        self,
        # RGB流配置
        clip_model_name: str = "openai/clip-vit-large-patch14",
        freeze_rgb_tower: bool = True,
        
        # 偏振流配置
        polar_backbone: str = "google/vit-base-patch16-224-in21k",  # 使用 transformers ViT（与 Stage 1 MAE 兼容）
        freeze_polar_tower: bool = False,
        freeze_polar_layers: int = 0,  # 冻结前N层（用于防止过拟合）
        
        # Stage 1 权重加载（可选，已废弃，改用 vae_model_path）
        stage1_checkpoint: Optional[str] = None,  # Stage 1 检查点路径（用于加载 MAE 编码器权重，已废弃）
        
        # VAE 编码器配置（新增）
        vae_model_path: Optional[str] = None,  # VAE 模型路径（用于加载 VAE 编码器，替代 Stage 1）
        
        # Stage 2 权重加载（可选）
        stage2_checkpoint: Optional[str] = None,  # Stage 2 检查点路径（用于加载投影器权重）
        
        # LLM配置（默认使用本地 Instruct 模型）
        llm_model_name: str = "/openbayes/input/input0/models/Meta-Llama-3-8B-Instruct",
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        
        # 投影器配置
        rgb_feature_dim: int = 1024,
        polar_feature_dim: int = 768,  # ViT-Base: 768（固定，因为现在只支持 ViT）
        llm_hidden_dim: Optional[int] = None,  # 如果为 None，将从 LLM 配置动态获取
        
        # LoRA配置
        use_lora: bool = True,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        
        # Hugging Face token（用于下载模型）
        hf_token: Optional[str] = None,
        
        # Tokenizer配置（用于resize embeddings）
        new_vocab_size: Optional[int] = None,  # 如果提供，在应用LoRA之前resize
    ):
        super().__init__()
        
        # ========== 1. RGB流：CLIP视觉编码器（冻结） ==========
        print("正在加载CLIP视觉编码器...")
        # 检查是否为本地路径（绝对路径且存在，或者相对路径且存在）
        is_local_path = (os.path.isabs(clip_model_name) and os.path.exists(clip_model_name)) or \
                       (not os.path.isabs(clip_model_name) and os.path.exists(clip_model_name))
        
        # 使用 token 进行认证（优先使用参数，其次环境变量）
        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        load_kwargs = {}
        
        if is_local_path:
            # 本地路径：使用 local_files_only=True 强制从本地加载
            load_kwargs["local_files_only"] = True
            print(f"✓ 检测到本地路径，从本地加载 CLIP 模型: {clip_model_name}")
        else:
            # Hugging Face Hub 路径：可能需要 token
            if token:
                load_kwargs["token"] = token
                print(f"✓ 使用 Hugging Face token 从 Hub 加载 CLIP 模型: {clip_model_name}")
            else:
                print(f"⚠ 警告: 未找到 Hugging Face token，某些模型可能需要认证")
                print(f"  尝试从 Hub 加载: {clip_model_name}")
        
        self.vision_tower_rgb = CLIPVisionModel.from_pretrained(clip_model_name, **load_kwargs)
        
        if freeze_rgb_tower:
            # 严格冻结RGB流
            for param in self.vision_tower_rgb.parameters():
                param.requires_grad = False
            self.vision_tower_rgb.eval()  # 设置为评估模式
            print("✓ RGB流已冻结")
        else:
            print("⚠ RGB流未冻结（可训练）")
        
        # ========== 2. 偏振流：使用 VAE 编码器或 ViTModel ==========
        # 优先使用 VAE 编码器（如果提供了 vae_model_path）
        if vae_model_path and os.path.exists(vae_model_path):
            print(f"正在加载 VAE 编码器: {vae_model_path}...")
            print("  使用 VAE 编码器（3通道输入：DoLP, sin(2*AoLP), cos(2*AoLP)）")
            
            # 加载 VAE 模型
            vae = AutoencoderKL.from_pretrained(vae_model_path, local_files_only=True)
            
            # 1. 提取 Encoder（输出高维特征，通常是 512 通道）
            self.vision_tower_polar = vae.encoder
            
            # 2. 提取 Quant Conv（关键修复！必须有这一层才能得到 Latent）
            # quant_conv 将 encoder 的 512 通道压缩到 8 通道（均值+方差）
            self.vision_tower_polar_quant_conv = vae.quant_conv
            
            # 3. 删除 Decoder 以节省显存
            del vae.decoder
            del vae
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # ⚠️ 关键修复：动态获取实际的 latent 通道数
            # 注意：encoder 输出是 Tensor，不是 EncoderOutput 对象
            # 需要经过 quant_conv 才能得到 moments（8通道：前4个是均值，后4个是方差）
            print("  ⚠ 正在验证 VAE latent 通道数...")
            
            # 获取 VAE 所在的设备和数据类型（避免设备不匹配错误）
            vae_device = next(self.vision_tower_polar.parameters()).device
            vae_dtype = next(self.vision_tower_polar.parameters()).dtype
            
            with torch.no_grad():
                # 创建虚拟输入（在正确的设备和 dtype 上）
                dummy_input = torch.zeros(1, 3, 512, 512, device=vae_device, dtype=vae_dtype)
                # 转换为 [-1, 1] 范围
                dummy_input = dummy_input * 2.0 - 1.0
                
                # 正确的前向传播流程
                # 1. Encoder 提取高维特征
                h = self.vision_tower_polar(dummy_input)  # (B, 512, 64, 64) 或类似
                # 2. Quant Conv 压缩到 Latent 空间
                moments = self.vision_tower_polar_quant_conv(h)  # (B, 8, 64, 64)
                # 3. 提取均值 (Mean) 作为特征，丢弃方差 (Logvar)
                # moments 的前4个通道是均值，后4个通道是方差
                mean, _ = torch.chunk(moments, 2, dim=1)  # (B, 4, 64, 64)
                
                actual_latent_dim = mean.shape[1]  # 应该是 4（对于 sd-vae-ft-mse）
            
            # 创建适配层（将 latent 投影到特征维度）
            # 输入维度必须是实际 latent 通道数（4），而不是配置中的值（8）
            self.vae_latent_to_feature = nn.Linear(actual_latent_dim, 768)
            # 初始化权重
            nn.init.xavier_uniform_(self.vae_latent_to_feature.weight)
            nn.init.zeros_(self.vae_latent_to_feature.bias)
            
            polar_feature_dim = 768  # 保持与 ViT 一致的特征维度
            self.use_vae_encoder = True
            print(f"  ✓ VAE 编码器及 QuantConv 加载完成")
            print(f"  - 输入通道数: 3 (DoLP, sin(2*AoLP), cos(2*AoLP))")
            print(f"  - 输入值范围: [0, 1]（将在 forward 中转换为 [-1, 1]）")
            print(f"  - 实际 Latent 通道数: {actual_latent_dim} (从 moments 中提取的均值通道数)")
            print(f"  - 输出特征维度: {polar_feature_dim}")
            print(f"  - 适配层: Linear({actual_latent_dim} -> 768)")
        else:
            # 使用 ViTModel（兼容旧代码）
            print(f"正在加载偏振流backbone: {polar_backbone}...")
            print("  使用 transformers.ViTModel")
            self.use_vae_encoder = False
            
            # 检查是否为本地路径
            is_local_path = (os.path.isabs(polar_backbone) and os.path.exists(polar_backbone)) or \
                           (not os.path.isabs(polar_backbone) and os.path.exists(polar_backbone))
            
            # 使用 token 进行认证
            token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            load_kwargs = {}
            
            if is_local_path:
                load_kwargs["local_files_only"] = True
                print(f"  ✓ 检测到本地路径，从本地加载 ViT 模型: {polar_backbone}")
            else:
                if token:
                    load_kwargs["token"] = token
                    print(f"  ✓ 使用 Hugging Face token 从 Hub 加载 ViT 模型: {polar_backbone}")
                else:
                    print(f"  ⚠ 警告: 未找到 Hugging Face token")
            
            # 加载配置
            if is_local_path:
                config = ViTConfig.from_pretrained(polar_backbone, local_files_only=True)
            else:
                config_load_kwargs = {}
                if token:
                    config_load_kwargs["token"] = token
                config = ViTConfig.from_pretrained(polar_backbone, **config_load_kwargs)
            
            # 修改配置以支持 3 通道输入（DoLP, sin(2*AoLP), cos(2*AoLP)）
            config.num_channels = 3
            print(f"  ✓ 配置为 3 通道输入（DoLP, sin(2*AoLP), cos(2*AoLP)）")
            
            # 初始化模型
            if stage1_checkpoint and os.path.exists(stage1_checkpoint):
                print("  ✓ 将使用 Stage 1 权重，跳过预训练权重加载")
                self.vision_tower_polar = ViTModel(config)
            else:
                print("  ⚠ 未提供 Stage 1 权重，将从预训练权重加载（需要适配第一层）")
                # 先临时用 3 通道配置加载预训练权重
                temp_config = ViTConfig.from_pretrained(
                    polar_backbone,
                    **({} if not is_local_path else {"local_files_only": True})
                )
                temp_model = ViTModel.from_pretrained(
                    polar_backbone,
                    config=temp_config,
                    **load_kwargs
                )
                # 提取除了第一层之外的所有权重
                state_dict = temp_model.state_dict()
                keys_to_remove = [
                    "embeddings.patch_embeddings.projection.weight",
                    "embeddings.patch_embeddings.projection.bias"
                ]
                for key in keys_to_remove:
                    if key in state_dict:
                        del state_dict[key]
                
                # 使用 3 通道配置初始化新模型
                self.vision_tower_polar = ViTModel(config)
                missing_keys, unexpected_keys = self.vision_tower_polar.load_state_dict(
                    state_dict, strict=False
                )
                if missing_keys:
                    print(f"  ⚠ 警告: {len(missing_keys)} 个键未加载（这是正常的，因为第一层需要适配）")
                # 适配第一层（从 3 通道适配，但现在是 3 通道，所以只需要初始化）
                self._adapt_patch_embedding_for_3channels_from_pretrained(temp_model)
                del temp_model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # ViT-Base 的特征维度固定为 768
            polar_feature_dim = 768
            
            # 如果提供了 Stage 1 检查点路径，尝试加载 MAE 编码器权重
            if stage1_checkpoint and os.path.exists(stage1_checkpoint):
                print(f"\n正在尝试从 Stage 1 检查点加载编码器权重: {stage1_checkpoint}")
                self._load_stage1_encoder_weights(stage1_checkpoint)
            elif stage1_checkpoint:
                print(f"  ⚠ 警告: Stage 1 检查点路径不存在: {stage1_checkpoint}")
                print("    已使用预训练的 ViT 权重并适配到 3 通道输入")
            
            print(f"✓ 偏振流backbone加载完成，特征维度: {polar_feature_dim}")
        
        # 冻结偏振流的指定层数（用于防止过拟合）
        if freeze_polar_tower:
            # 1. 冻结 Encoder（VAE 编码器或 ViT）
            for param in self.vision_tower_polar.parameters():
                param.requires_grad = False
            self.vision_tower_polar.eval()  # 设置为评估模式
            
            # 2. [新增] 冻结 Quant Conv（如果是 VAE）
            if hasattr(self, 'vision_tower_polar_quant_conv'):
                for param in self.vision_tower_polar_quant_conv.parameters():
                    param.requires_grad = False
                self.vision_tower_polar_quant_conv.eval()
            
            # 3. 处理适配层 (Adapter)
            if hasattr(self, 'use_vae_encoder') and self.use_vae_encoder:
                if hasattr(self, 'vae_latent_to_feature'):
                    # 适配层保持可训练（这是连接 VAE 和 Projector 的桥梁）
                    for param in self.vae_latent_to_feature.parameters():
                        param.requires_grad = True
                    print("✓ 偏振流（VAE Encoder + QuantConv）已冻结，但 Linear 适配层保持可训练")
                else:
                    print("✓ 偏振流（VAE Encoder + QuantConv）已完全冻结")
            else:
                print("✓ 偏振流已完全冻结")
        elif freeze_polar_layers > 0:
            # 冻结前N层（针对 ViT 的 encoder 层）
            # ViT 的 encoder 结构：encoder.layer.0, encoder.layer.1, ..., encoder.layer.11
            layer_count = 0
            for name, param in self.vision_tower_polar.named_parameters():
                # 只冻结 encoder.layer 中的层（不包括 embeddings）
                if "encoder.layer" in name:
                    layer_idx = int(name.split("encoder.layer.")[1].split(".")[0])
                    if layer_idx < freeze_polar_layers:
                        param.requires_grad = False
                        layer_count += 1
            print(f"✓ 偏振流前{freeze_polar_layers}层已冻结（共{layer_count}个参数）")
        
        # 更新polar_feature_dim（使用实际计算的值）
        self.polar_feature_dim = polar_feature_dim
        
        # ========== 3. 多模态投影器 ==========
        # ⚠️ 注意：投影器需要在 LLM 加载之前初始化，但需要知道 LLM 的 hidden_size
        # 因此我们先尝试从 LLM 配置获取 hidden_size，如果失败则使用默认值
        print("正在初始化多模态投影器...")
        
        # 动态获取 LLM 的 hidden_size（如果未指定）
        # 这允许代码适配不同尺寸的 LLM（如 Llama-3-70B）
        if llm_hidden_dim is None:
            # 先尝试从 LLM 配置获取 hidden_size
            # 注意：这里先创建一个临时配置来获取 hidden_size，实际 LLM 会在后面加载
            try:
                from transformers import LlamaConfig
                # 检查是否为本地路径
                is_local_llm = (os.path.isabs(llm_model_name) and os.path.exists(llm_model_name)) or \
                              (not os.path.isabs(llm_model_name) and os.path.exists(llm_model_name))
                
                config_load_kwargs = {}
                if is_local_llm:
                    config_load_kwargs["local_files_only"] = True
                else:
                    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                    if token:
                        config_load_kwargs["token"] = token
                
                temp_config = LlamaConfig.from_pretrained(llm_model_name, **config_load_kwargs)
                llm_hidden_dim = temp_config.hidden_size
                print(f"  ✓ 从 LLM 配置动态获取 hidden_size: {llm_hidden_dim}")
            except Exception as e:
                print(f"  ⚠ 警告: 无法从 LLM 配置获取 hidden_size，使用默认值 4096: {e}")
                llm_hidden_dim = 4096  # LLaMA-3-8B 的默认值
        else:
            print(f"  ✓ 使用指定的 llm_hidden_dim: {llm_hidden_dim}")
        
        self.multi_modal_projector = MultiModalProjector(
            rgb_feature_dim=rgb_feature_dim,
            polar_feature_dim=self.polar_feature_dim,
            llm_hidden_dim=llm_hidden_dim,
        )
        print("✓ 多模态投影器初始化完成（包含动态平衡参数：RGB scale: 1.0, Polar scale: 0.1）")
        print("  注意：这是初始值，如果提供 Stage 2 检查点，将从检查点加载实际训练后的值")
        
        # 如果提供了 Stage 2 检查点路径，尝试加载投影器权重
        if stage2_checkpoint and os.path.exists(stage2_checkpoint):
            print(f"\n正在尝试从 Stage 2 检查点加载投影器权重: {stage2_checkpoint}")
            self._load_stage2_projector_weights(stage2_checkpoint)
        elif stage2_checkpoint:
            print(f"  ⚠ 警告: Stage 2 检查点路径不存在: {stage2_checkpoint}")
            print("    将使用随机初始化的投影器权重（RGB scale: 1.0, Polar scale: 0.1）")
        
        # ========== 4. LLM：LLaMA-3-8B ==========
        print("正在加载LLM...")
        from transformers import BitsAndBytesConfig
        
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,  # 使用 float16 而不是 bfloat16（避免 PyTorch 2.0.0 的 triu 错误）
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # 使用 token 进行认证（优先使用参数，其次环境变量）
        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        
        # 尝试使用 Flash Attention 2（如果可用），否则使用默认实现
        try:
            import flash_attn
            attn_implementation = "flash_attention_2"
            print("✓ 检测到 Flash Attention 2，将使用它来加速训练")
        except ImportError:
            attn_implementation = None  # 使用默认实现（通常是 SDPA）
            print("⚠ 未安装 Flash Attention 2，将使用默认注意力实现（速度可能稍慢）")
        
        llm_load_kwargs = {
            "quantization_config": quantization_config,
            # 不设置 torch_dtype，让 Trainer 的 fp16 参数管理精度转换
            # 对于量化模型，模型参数会保持 fp32，forward 时自动转换为 fp16
            "device_map": "auto",
        }
        if attn_implementation:
            llm_load_kwargs["attn_implementation"] = attn_implementation
        
        if token:
            llm_load_kwargs["token"] = token
            print(f"✓ 使用 Hugging Face token 加载 LLM 模型")
        else:
            print("⚠ 警告: 未找到 Hugging Face token，某些模型可能需要认证")
        
        self.language_model = LlamaForCausalLM.from_pretrained(
            llm_model_name,
            **llm_load_kwargs
        )
        print("✓ LLM加载完成")
        
        # 验证投影器的 hidden_dim 与 LLM 的 hidden_size 是否匹配
        actual_llm_hidden_size = self.language_model.config.hidden_size
        if llm_hidden_dim != actual_llm_hidden_size:
            print(f"  ⚠ 警告: 投影器的 llm_hidden_dim ({llm_hidden_dim}) 与 LLM 的 hidden_size ({actual_llm_hidden_size}) 不匹配")
            print(f"    这可能导致维度不匹配错误。建议重新初始化投影器。")
        
        # ========== 5. Resize token embeddings（必须在应用LoRA之前）==========
        # 如果 tokenizer 添加了新 token（如<image>），需要先 resize embedding 层
        # 这必须在应用 LoRA 之前完成，因为 LoRA 会包装 embed_tokens
        if new_vocab_size is not None and new_vocab_size > self.language_model.config.vocab_size:
            print(f"正在调整词汇表大小: {self.language_model.config.vocab_size} -> {new_vocab_size}")
            self.language_model.resize_token_embeddings(new_vocab_size)
            print(f"✓ 词汇表大小已更新: {new_vocab_size}")
        
        # ========== 6. LoRA适配器 ==========
        if use_lora:
            print("正在配置LoRA适配器...")
            
            # [关键优化] 对于小数据集（4k样本），不保存 embed_tokens 和 lm_head
            # 这样可以大幅减少可训练参数（从 21% 降到 1-3%），防止过拟合
            # 注意：即使添加了新 token（如<image>），新 token 的 embedding 也会通过 LoRA 学习
            # 如果确实需要保存新 token 的 embedding，可以在训练后手动保存
            modules_to_save = []  # 保持为空，只训练 LoRA 适配器
            
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                modules_to_save=modules_to_save,  # 空列表，不保存完整权重
            )
            
            self.language_model = get_peft_model(self.language_model, lora_config)
            print("✓ LoRA适配器配置完成 (modules_to_save=[]，仅训练 LoRA 参数)")
        
        # ========== 7. 图像占位符token（如果需要） ==========
        # LLaMA-3的tokenizer需要特殊处理<image>标记
        # 这里假设tokenizer已经添加了<image> token
        # 在实际使用中，需要在tokenizer中添加这个特殊token
        
        # 存储image_token_id（将在训练时设置）
        self.image_token_id = None
    
    def _adapt_patch_embedding_for_4channels(self):
        """
        适配 patch embedding 层以支持 4 通道输入（从 3 通道预训练权重扩展）
        
        策略：
        1. 如果当前权重是 3 通道，复制前 3 个通道的权重
        2. 第 4 个通道初始化为前 3 个通道的均值
        
        这个方法用于在没有 Stage 1 权重时，从标准 ViT 预训练权重适配到 4 通道输入。
        """
        try:
            patch_embed = self.vision_tower_polar.embeddings.patch_embeddings
            if hasattr(patch_embed, 'projection'):
                proj = patch_embed.projection
                if isinstance(proj, nn.Conv2d):
                    # 检查当前通道数
                    if proj.in_channels == 3:
                        print("  ⚠ 检测到 3 通道 patch embedding，正在适配到 4 通道...")
                        # 创建新的 4 通道权重
                        old_weight = proj.weight.data  # (out_channels, 3, kernel_h, kernel_w)
                        new_weight = torch.zeros(
                            old_weight.shape[0], 4, old_weight.shape[2], old_weight.shape[3],
                            dtype=old_weight.dtype,
                            device=old_weight.device
                        )
                        # 复制前 3 个通道
                        new_weight[:, :3, :, :] = old_weight
                        # 第 4 个通道初始化为前 3 个通道的均值
                        new_weight[:, 3, :, :] = old_weight.mean(dim=1)
                        # 替换权重
                        proj.weight = nn.Parameter(new_weight)
                        proj.in_channels = 4
                        print("  ✓ Patch embedding 已适配到 4 通道")
                    elif proj.in_channels == 4:
                        print("  ✓ Patch embedding 已经是 4 通道，无需适配")
                    else:
                        print(f"  ⚠ 警告: Patch embedding 通道数异常: {proj.in_channels}")
        except Exception as e:
            print(f"  ⚠ 适配 patch embedding 时出现错误: {e}")
            import traceback
            traceback.print_exc()
    
    def _adapt_patch_embedding_for_4channels_from_pretrained(self, pretrained_model: ViTModel):
        """
        从预训练模型适配 patch embedding 层以支持 4 通道输入
        
        这个方法用于从已加载的 3 通道预训练模型中提取第一层权重，并适配到 4 通道。
        
        Args:
            pretrained_model: 已加载的 3 通道预训练 ViTModel
        """
        try:
            # 获取预训练模型的第一层权重
            pretrained_proj = pretrained_model.embeddings.patch_embeddings.projection
            if not isinstance(pretrained_proj, nn.Conv2d) or pretrained_proj.in_channels != 3:
                print(f"  ⚠ 警告: 预训练模型的第一层不是 3 通道 Conv2d，跳过适配")
                return
            
            # 获取当前模型的第一层
            current_proj = self.vision_tower_polar.embeddings.patch_embeddings.projection
            if not isinstance(current_proj, nn.Conv2d) or current_proj.in_channels != 4:
                print(f"  ⚠ 警告: 当前模型的第一层不是 4 通道 Conv2d，跳过适配")
                return
            
            print("  ⚠ 正在从预训练权重适配第一层（3通道 -> 4通道）...")
            
            # 提取预训练权重
            old_weight = pretrained_proj.weight.data  # (out_channels, 3, kernel_h, kernel_w)
            old_bias = pretrained_proj.bias.data if pretrained_proj.bias is not None else None
            
            # 创建新的 4 通道权重
            new_weight = torch.zeros(
                old_weight.shape[0], 4, old_weight.shape[2], old_weight.shape[3],
                dtype=old_weight.dtype,
                device=old_weight.device
            )
            
            # 复制前 3 个通道
            new_weight[:, :3, :, :] = old_weight
            
            # 第 4 个通道初始化为前 3 个通道的均值
            new_weight[:, 3, :, :] = old_weight.mean(dim=1)
            
            # 替换权重
            current_proj.weight = nn.Parameter(new_weight)
            
            # 复制偏置（如果有）
            if old_bias is not None:
                current_proj.bias = nn.Parameter(old_bias.clone())
            
            print("  ✓ Patch embedding 已从预训练权重适配到 4 通道")
        except Exception as e:
            print(f"  ⚠ 从预训练模型适配 patch embedding 时出现错误: {e}")
            import traceback
            traceback.print_exc()
    
    def _adapt_patch_embedding_for_3channels_from_pretrained(self, pretrained_model: ViTModel):
        """
        从预训练模型适配 patch embedding 层以支持 3 通道输入
        
        如果预训练模型已经是 3 通道，直接复制权重即可。
        
        Args:
            pretrained_model: 已加载的 3 通道预训练 ViTModel
        """
        try:
            # 获取预训练模型的第一层权重
            pretrained_proj = pretrained_model.embeddings.patch_embeddings.projection
            if not isinstance(pretrained_proj, nn.Conv2d) or pretrained_proj.in_channels != 3:
                print(f"  ⚠ 警告: 预训练模型的第一层不是 3 通道 Conv2d，跳过适配")
                return
            
            # 获取当前模型的第一层
            current_proj = self.vision_tower_polar.embeddings.patch_embeddings.projection
            if not isinstance(current_proj, nn.Conv2d) or current_proj.in_channels != 3:
                print(f"  ⚠ 警告: 当前模型的第一层不是 3 通道 Conv2d，跳过适配")
                return
            
            print("  ⚠ 正在从预训练权重复制第一层（3通道 -> 3通道）...")
            
            # 直接复制权重（都是 3 通道，形状应该匹配）
            current_proj.weight = nn.Parameter(pretrained_proj.weight.data.clone())
            if pretrained_proj.bias is not None:
                if current_proj.bias is not None:
                    current_proj.bias = nn.Parameter(pretrained_proj.bias.data.clone())
            
            print("  ✓ Patch embedding 已从预训练权重复制（3通道）")
        except Exception as e:
            print(f"  ⚠ 适配 patch embedding 时出错: {e}")
            print("    将使用随机初始化的权重")
    
    def _load_vae_encoder_weights(self, vae_model_path: str):
        """
        从 VAE 模型加载编码器权重（已废弃，VAE 编码器在 __init__ 中直接加载）
        
        这个方法保留以兼容旧代码，实际不再使用。
        """
        print(f"  ⚠ 注意: VAE 编码器已在 __init__ 中直接加载，无需调用此方法")
    
    def _load_stage1_encoder_weights(self, checkpoint_path: str):
        """
        从 Stage 1 MAE 检查点加载编码器权重
        
        Stage 1 使用 ViTMAEForPreTraining，其状态字典包含：
        - vit.embeddings.*: 嵌入层权重
        - vit.encoder.*: 编码器权重
        - decoder.*: 解码器权重（不需要）
        
        我们需要提取 vit.embeddings 和 vit.encoder 的权重，并加载到 ViTModel 中。
        ViTModel 的键名是：
        - embeddings.*
        - encoder.*
        
        因此需要将 "vit." 前缀去掉。
        
        Args:
            checkpoint_path: Stage 1 检查点路径（可以是目录或文件路径）
        """
        try:
            # 尝试加载检查点（可能是目录或文件）
            if os.path.isdir(checkpoint_path):
                # 如果是目录，尝试加载 pytorch_model.bin 或 model.safetensors
                possible_files = [
                    os.path.join(checkpoint_path, "pytorch_model.bin"),
                    os.path.join(checkpoint_path, "model.safetensors"),
                ]
                checkpoint_file = None
                for f in possible_files:
                    if os.path.exists(f):
                        checkpoint_file = f
                        break
                
                if checkpoint_file is None:
                    raise FileNotFoundError(f"在 {checkpoint_path} 中未找到模型文件")
            else:
                checkpoint_file = checkpoint_path
            
            print(f"  正在加载检查点文件: {checkpoint_file}")
            
            # 加载状态字典
            if checkpoint_file.endswith(".safetensors"):
                import safetensors.torch
                state_dict = safetensors.torch.load_file(checkpoint_file)
            else:
                state_dict = torch.load(checkpoint_file, map_location="cpu")
            
            # 如果是完整的模型状态字典（包含 "model." 前缀），需要去掉
            if any(k.startswith("model.") for k in state_dict.keys()):
                state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            
            # 提取编码器权重（vit.embeddings.* 和 vit.encoder.*）
            encoder_state_dict = {}
            decoder_keys = []
            
            for key, value in state_dict.items():
                # 提取 vit.embeddings 和 vit.encoder 的权重
                if key.startswith("vit.embeddings."):
                    # 去掉 "vit." 前缀，直接使用 embeddings.*
                    new_key = key.replace("vit.embeddings.", "embeddings.")
                    encoder_state_dict[new_key] = value
                elif key.startswith("vit.encoder."):
                    # 去掉 "vit." 前缀，直接使用 encoder.*
                    new_key = key.replace("vit.encoder.", "encoder.")
                    encoder_state_dict[new_key] = value
                elif key.startswith("decoder.") or key.startswith("mask_token"):
                    # 解码器权重，不需要
                    decoder_keys.append(key)
            
            if not encoder_state_dict:
                raise ValueError(
                    f"在检查点中未找到编码器权重（期望 vit.embeddings.* 或 vit.encoder.*）\n"
                    f"找到的键示例: {list(state_dict.keys())[:10]}"
                )
            
            print(f"  ✓ 提取到 {len(encoder_state_dict)} 个编码器权重")
            print(f"  ⚠ 跳过 {len(decoder_keys)} 个解码器权重")
            
            # 加载权重（使用 strict=False，因为可能有部分不匹配的键）
            # 注意：如果 patch_embeddings.projection.weight 的形状不匹配，需要特殊处理
            patch_proj_key = "embeddings.patch_embeddings.projection.weight"
            if patch_proj_key in encoder_state_dict:
                checkpoint_weight = encoder_state_dict[patch_proj_key]
                current_weight = self.vision_tower_polar.embeddings.patch_embeddings.projection.weight
                
                # 检查形状是否匹配
                if checkpoint_weight.shape != current_weight.shape:
                    print(f"  ⚠ 检测到 patch embedding 形状不匹配:")
                    print(f"    检查点: {checkpoint_weight.shape}")
                    print(f"    当前模型: {current_weight.shape}")
                    
                    # 如果检查点是 4 通道，当前模型也是 4 通道，应该匹配
                    # 如果不匹配，可能是其他问题，跳过这个键
                    if checkpoint_weight.shape[1] == 4 and current_weight.shape[1] == 4:
                        # 形状应该匹配，直接使用
                        encoder_state_dict[patch_proj_key] = checkpoint_weight
                    else:
                        print(f"  ⚠ 跳过 patch embedding 权重（形状不匹配）")
                        del encoder_state_dict[patch_proj_key]
            
            missing_keys, unexpected_keys = self.vision_tower_polar.load_state_dict(
                encoder_state_dict, strict=False
            )
            
            if missing_keys:
                print(f"  ⚠ 警告: {len(missing_keys)} 个键未加载: {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"  ⚠ 警告: {len(unexpected_keys)} 个意外的键: {unexpected_keys[:5]}...")
            
            print("  ✓ Stage 1 编码器权重加载完成")
            
        except Exception as e:
            print(f"  ⚠ 加载 Stage 1 权重时出现错误: {e}")
            print("    将使用预训练的 ViT 权重")
            import traceback
            traceback.print_exc()
    
    def _load_stage2_projector_weights(self, checkpoint_path: str):
        """
        从 Stage 2 检查点加载投影器权重
        
        Stage 2 保存的投影器权重文件：projector.pth
        
        Args:
            checkpoint_path: Stage 2 检查点路径（可以是目录或文件路径）
        """
        try:
            # 尝试加载投影器权重
            if os.path.isdir(checkpoint_path):
                # 如果是目录，尝试加载 projector.pth
                projector_path = os.path.join(checkpoint_path, "projector.pth")
                if not os.path.exists(projector_path):
                    # 尝试其他可能的路径
                    possible_files = [
                        os.path.join(checkpoint_path, "projector.pth"),
                        os.path.join(checkpoint_path, "pytorch_model.bin"),
                        os.path.join(checkpoint_path, "model.safetensors"),
                    ]
                    projector_path = None
                    for f in possible_files:
                        if os.path.exists(f):
                            projector_path = f
                            break
                    
                    if projector_path is None:
                        raise FileNotFoundError(f"在 {checkpoint_path} 中未找到投影器权重文件")
            else:
                projector_path = checkpoint_path
            
            print(f"  正在加载投影器权重文件: {projector_path}")
            
            # 加载状态字典
            if projector_path.endswith(".safetensors"):
                import safetensors.torch
                projector_state = safetensors.torch.load_file(projector_path)
            else:
                projector_state = torch.load(projector_path, map_location="cpu")
            
            # 如果状态字典包含 "model." 前缀，需要去掉
            if any(k.startswith("model.") for k in projector_state.keys()):
                projector_state = {k.replace("model.", ""): v for k, v in projector_state.items()}
            
            # 如果状态字典包含 "multi_modal_projector." 前缀，需要去掉
            if any(k.startswith("multi_modal_projector.") for k in projector_state.keys()):
                projector_state = {
                    k.replace("multi_modal_projector.", ""): v
                    for k, v in projector_state.items()
                }
            
            # 加载投影器权重
            missing_keys, unexpected_keys = self.multi_modal_projector.load_state_dict(
                projector_state, strict=False
            )
            
            if missing_keys:
                print(f"  ⚠ 警告: {len(missing_keys)} 个键未加载: {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"  ⚠ 警告: {len(unexpected_keys)} 个意外的键: {unexpected_keys[:5]}...")
            
            # [新增] 打印加载后的 rgb_scale 和 polar_scale 值，验证是否正确加载
            if hasattr(self.multi_modal_projector, 'rgb_scale'):
                rgb_scale_value = self.multi_modal_projector.rgb_scale.item()
                print(f"  ✓ RGB scale (从 Stage 2 加载): {rgb_scale_value:.6f}")
            if hasattr(self.multi_modal_projector, 'polar_scale'):
                polar_scale_value = self.multi_modal_projector.polar_scale.item()
                print(f"  ✓ Polar scale (从 Stage 2 加载): {polar_scale_value:.6f}")
            
            print("  ✓ Stage 2 投影器权重加载完成")
            
        except Exception as e:
            print(f"  ⚠ 加载 Stage 2 权重时出现错误: {e}")
            print("    将使用随机初始化的投影器权重")
            import traceback
            traceback.print_exc()
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """启用梯度检查点（将调用传递给 language_model）"""
        if hasattr(self.language_model, 'gradient_checkpointing_enable'):
            if gradient_checkpointing_kwargs is not None:
                self.language_model.gradient_checkpointing_enable(**gradient_checkpointing_kwargs)
            else:
                self.language_model.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        """禁用梯度检查点（将调用传递给 language_model）"""
        if hasattr(self.language_model, 'gradient_checkpointing_disable'):
            self.language_model.gradient_checkpointing_disable()
    
    def get_vision_features(
        self,
        pixel_values_rgb: torch.Tensor,
        pixel_values_polar: torch.Tensor,
    ) -> torch.Tensor:
        """
        提取RGB和偏振视觉特征（空间对齐后在特征维度融合）
        
        关键改进：
        - 将 Polar 特征从 14x14 插值到 16x16，与 RGB 空间对齐
        - 在特征维度拼接（而非序列维度），每个 token 同时包含 RGB 和 Polar 信息
        - 降低学习难度，LLM 无需跨 token 学习对应关系
        
        Args:
            pixel_values_rgb: RGB图像，形状为 (B, C, H, W)，C=3
            pixel_values_polar: 偏振图像，形状为 (B, 3, H, W) 或 (B, 4, H, W)
                - 如果使用 VAE 编码器：3通道 [DoLP, sin(2*AoLP), cos(2*AoLP)]
                - 如果使用 ViT：可能是 3 或 4 通道
            
        Returns:
            投影后的多模态特征，形状为 (B, N, llm_hidden_dim)
                - 例如：(B, 256, 4096) - 256个空间对齐的 patch tokens
                - 每个 token 同时包含对应位置的 RGB 和 Polar 信息
        """
        # 1. RGB 特征提取（冻结，不计算梯度）
        with torch.no_grad() if not self.vision_tower_rgb.training else torch.enable_grad():
            rgb_outputs = self.vision_tower_rgb(pixel_values_rgb)
            # CLIP ViT-L/14 输出: (B, 257, 1024) - 1个CLS token + 256个patch
            # 去掉 CLS token，保留 Patch tokens: (B, 256, 1024)
            rgb_features = rgb_outputs.last_hidden_state[:, 1:, :]  # (B, 256, 1024)
            B, N_rgb, D_rgb = rgb_features.shape
            H_rgb = W_rgb = int(N_rgb ** 0.5)  # 16
        
        # 2. Polar 特征提取（使用 VAE 编码器或 ViTModel）
        # Stage 2 中 Polar tower 被冻结，使用 no_grad 节省显存和计算
        with torch.no_grad() if not self.vision_tower_polar.training else torch.enable_grad():
            if hasattr(self, 'use_vae_encoder') and self.use_vae_encoder:
                # === VAE 编码器分支 ===
                # VAE 输入需要是 [0, 1] 范围，然后转换为 [-1, 1]
                # pixel_values_polar 已经是 [0, 1] 范围
                polar_input = pixel_values_polar  # (B, 3, H, W)，值范围 [0, 1]
                
                # Resize 到 512x512（VAE 的输入尺寸）
                if polar_input.shape[2] != 512 or polar_input.shape[3] != 512:
                    polar_input = torch.nn.functional.interpolate(
                        polar_input,
                        size=(512, 512),
                        mode='bilinear',
                        align_corners=False
                    )
                
                # 转换范围：[0, 1] -> [-1, 1]
                polar_input = polar_input * 2.0 - 1.0
                
                # ⚠️ 修复后的前向传播逻辑
                # 1. Encoder 提取高维特征（输出是 Tensor，不是 EncoderOutput 对象）
                h = self.vision_tower_polar(polar_input)  # (B, 512, 64, 64) 或类似
                
                # 2. Quant Conv 压缩到 Latent 空间
                moments = self.vision_tower_polar_quant_conv(h)  # (B, 8, 64, 64)
                # moments 的前4个通道是均值（mean），后4个通道是方差（logvar）
                
                # 3. 提取均值 (Mean) 作为特征，丢弃方差 (Logvar)
                # 对于特征提取任务，确定性的均值比随机采样的 latent 更好
                latent, _ = torch.chunk(moments, 2, dim=1)  # (B, 4, 64, 64)
                
                # 4. [可选] 缩放系数 (Scaling Factor)
                # SD VAE 的 latent 数值很小，乘以系数可以对齐到标准分布
                # 虽然 Projector 可以学习，但乘上这个系数通常收敛更快
                # 注意：0.18215 是 SD VAE 的标准缩放系数
                latent = latent * 0.18215
                
                # ⚠️ 关键修复：直接从 64x64 插值到 16x16，跳过中间的 14x14 步骤
                # 这样可以保留更多信息，避免不必要的下采样损失
                
                # 维度变换: (B, 4, 64, 64) -> (B, 64, 64, 4)
                latent_permuted = latent.permute(0, 2, 3, 1)
                
                # 投影通道: (B, 64, 64, 4) -> (B, 64, 64, 768)
                # 确保适配层在正确的设备上
                if self.vae_latent_to_feature.weight.device != latent.device:
                    self.vae_latent_to_feature = self.vae_latent_to_feature.to(latent.device)
                
                polar_features_64x64 = self.vae_latent_to_feature(latent_permuted)  # (B, 64, 64, 768)
                
                # 空间对齐: 直接从 64x64 下采样到 16x16（RGB 的分辨率）
                # 先变回 (B, 768, 64, 64) 用于 interpolate
                polar_features_spatial = polar_features_64x64.permute(0, 3, 1, 2)  # (B, 768, 64, 64)
                
                # 直接插值到 RGB 分辨率 (16x16)
                dtype_original = polar_features_spatial.dtype
                polar_features_aligned = torch.nn.functional.interpolate(
                    polar_features_spatial.to(torch.float32),  # 临时转 float32 确保精度
                    size=(H_rgb, W_rgb),  # (16, 16)
                    mode='bilinear',  # VAE 特征是平滑的，bilinear 足够
                    align_corners=False
                ).to(dtype_original)  # 转回原始 dtype
                # 输出: (B, 768, 16, 16)
                
                # 展平为序列形式: (B, 768, 16, 16) -> (B, 768, 256) -> (B, 256, 768)
                polar_features_aligned = polar_features_aligned.flatten(2).permute(0, 2, 1)  # (B, 256, 768)
                
                # 设置维度信息（用于后续处理，虽然这里已经对齐了）
                B, N_polar, D_polar = polar_features_aligned.shape
                H_polar = W_polar = int(N_polar ** 0.5)  # 16（已经对齐到 RGB 分辨率）
            else:
                # === ViTModel 分支（兼容旧代码）===
                polar_outputs = self.vision_tower_polar(pixel_values_polar)
                # ViT-Base 输出: (B, 197, 768) - 1个CLS token + 196个patch
                # 去掉 CLS token，保留 Patch tokens: (B, 196, 768)
                polar_features = polar_outputs.last_hidden_state[:, 1:, :]  # (B, 196, 768)
                B, N_polar, D_polar = polar_features.shape
                H_polar = W_polar = int(N_polar ** 0.5)  # 14
                
                # 空间对齐：将 Polar 特征从 14x14 插值到 RGB 的分辨率 (16x16)
                # [Fix]: 使用 reshape 代替 view 以处理非连续内存（permute 后 tensor 可能不连续）
                polar_features_spatial = polar_features.permute(0, 2, 1).reshape(B, D_polar, H_polar, W_polar)  # (B, 768, 14, 14)
                
                # [Fix]: 临时使用 float32 进行 bicubic 插值以防 NaN（FP16/BF16 下 bicubic 可能数值不稳定）
                # 插值到 16x16（使用 bicubic 插值，保持特征平滑）
                dtype_original = polar_features_spatial.dtype
                polar_features_aligned = torch.nn.functional.interpolate(
                    polar_features_spatial.to(torch.float32),  # 临时转 float32 确保精度
                    size=(H_rgb, W_rgb),  # (16, 16)
                    mode='bicubic',  # ViT 特征建议用 bicubic
                    align_corners=False
                ).to(dtype_original)  # 转回原始 dtype (fp16/bf16)
                # 输出: (B, 768, 16, 16)
                
                # Reshape back to (B, N_rgb, D_polar)
                # (B, 768, 16, 16) -> (B, 768, 256) -> (B, 256, 768)
                polar_features_aligned = polar_features_aligned.flatten(2).permute(0, 2, 1)  # (B, 256, 768)
        
        # [新增] 一次性验证：输出 RGB 和 Polar 特征的值范围（用于判断平衡参数是否合理）
        if not hasattr(self, '_feature_range_printed'):
            print("\n" + "=" * 80)
            print("🔍 Stage 2 特征值范围验证（仅输出一次）")
            print("=" * 80)
            
            # 1. RGB 特征值范围（原始，未缩放）
            rgb_min = rgb_features.min().item()
            rgb_max = rgb_features.max().item()
            rgb_mean = rgb_features.mean().item()
            rgb_std = rgb_features.std().item()
            print(f"\n📊 RGB 特征（原始，未缩放）:")
            print(f"  - 形状: {rgb_features.shape}")
            print(f"  - 最小值: {rgb_min:.6f}")
            print(f"  - 最大值: {rgb_max:.6f}")
            print(f"  - 均值: {rgb_mean:.6f}")
            print(f"  - 标准差: {rgb_std:.6f}")
            
            # 2. Polar 特征值范围（原始，未缩放）
            polar_min = polar_features_aligned.min().item()
            polar_max = polar_features_aligned.max().item()
            polar_mean = polar_features_aligned.mean().item()
            polar_std = polar_features_aligned.std().item()
            print(f"\n📊 Polar 特征（原始，未缩放）:")
            print(f"  - 形状: {polar_features_aligned.shape}")
            print(f"  - 最小值: {polar_min:.6f}")
            print(f"  - 最大值: {polar_max:.6f}")
            print(f"  - 均值: {polar_mean:.6f}")
            print(f"  - 标准差: {polar_std:.6f}")
            
            # 3. 获取当前的平衡参数值
            if hasattr(self.multi_modal_projector, 'rgb_scale'):
                rgb_scale_val = self.multi_modal_projector.rgb_scale.item()
            else:
                rgb_scale_val = 1.0
            if hasattr(self.multi_modal_projector, 'polar_scale'):
                polar_scale_val = self.multi_modal_projector.polar_scale.item()
            else:
                polar_scale_val = 1.0
            
            print(f"\n⚖️  动态平衡参数（当前值）:")
            print(f"  - rgb_scale: {rgb_scale_val:.6f}")
            print(f"  - polar_scale: {polar_scale_val:.6f}")
            
            # 4. 缩放后的特征值范围
            rgb_scaled = rgb_features * rgb_scale_val
            polar_scaled = polar_features_aligned * polar_scale_val
            
            rgb_scaled_min = rgb_scaled.min().item()
            rgb_scaled_max = rgb_scaled.max().item()
            rgb_scaled_mean = rgb_scaled.mean().item()
            rgb_scaled_std = rgb_scaled.std().item()
            
            polar_scaled_min = polar_scaled.min().item()
            polar_scaled_max = polar_scaled.max().item()
            polar_scaled_mean = polar_scaled.mean().item()
            polar_scaled_std = polar_scaled.std().item()
            
            print(f"\n📊 RGB 特征（缩放后，rgb_scale={rgb_scale_val:.6f}）:")
            print(f"  - 最小值: {rgb_scaled_min:.6f}")
            print(f"  - 最大值: {rgb_scaled_max:.6f}")
            print(f"  - 均值: {rgb_scaled_mean:.6f}")
            print(f"  - 标准差: {rgb_scaled_std:.6f}")
            
            print(f"\n📊 Polar 特征（缩放后，polar_scale={polar_scale_val:.6f}）:")
            print(f"  - 最小值: {polar_scaled_min:.6f}")
            print(f"  - 最大值: {polar_scaled_max:.6f}")
            print(f"  - 均值: {polar_scaled_mean:.6f}")
            print(f"  - 标准差: {polar_scaled_std:.6f}")
            
            # 5. 拼接后的特征值范围（在 LayerNorm 之前）
            combined_features = torch.cat([rgb_scaled, polar_scaled], dim=-1)
            combined_min = combined_features.min().item()
            combined_max = combined_features.max().item()
            combined_mean = combined_features.mean().item()
            combined_std = combined_features.std().item()
            
            print(f"\n📊 拼接后的特征（RGB + Polar，LayerNorm 前）:")
            print(f"  - 形状: {combined_features.shape}")
            print(f"  - 最小值: {combined_min:.6f}")
            print(f"  - 最大值: {combined_max:.6f}")
            print(f"  - 均值: {combined_mean:.6f}")
            print(f"  - 标准差: {combined_std:.6f}")
            
            # 6. [新增] LayerNorm 归一化后的特征值范围（这是输入到 Linear 的实际值）
            # 这是关键：LayerNorm 会将输入标准化为均值 0、方差 1，抹平数值差异
            if hasattr(self.multi_modal_projector, 'input_layernorm'):
                combined_features_norm = self.multi_modal_projector.input_layernorm(combined_features)
                norm_min = combined_features_norm.min().item()
                norm_max = combined_features_norm.max().item()
                norm_mean = combined_features_norm.mean().item()
                norm_std = combined_features_norm.std().item()
                
                print(f"\n📊 LayerNorm 归一化后的特征（输入到 Linear 的实际值）:")
                print(f"  - 形状: {combined_features_norm.shape}")
                print(f"  - 最小值: {norm_min:.6f}")
                print(f"  - 最大值: {norm_max:.6f}")
                print(f"  - 均值: {norm_mean:.6f} (应该接近 0)")
                print(f"  - 标准差: {norm_std:.6f} (应该接近 1)")
                
                # 验证 LayerNorm 的效果
                if abs(norm_mean) < 1e-5 and abs(norm_std - 1.0) < 0.1:
                    print(f"  ✓ LayerNorm 工作正常：均值接近 0，标准差接近 1")
                    print(f"  ✓ 数值差异问题已通过 LayerNorm 解决，rgb_scale 和 polar_scale 的影响被标准化抹平")
                else:
                    print(f"  ⚠️  警告: LayerNorm 可能未正常工作（均值: {norm_mean:.6f}, 标准差: {norm_std:.6f}）")
            else:
                print(f"\n⚠️  注意: MultiModalProjector 没有 input_layernorm，数值差异问题可能仍然存在")
            
            # 7. 分析建议（基于 LayerNorm 前的特征）
            print(f"\n💡 分析建议（LayerNorm 前）:")
            std_ratio = polar_scaled_std / rgb_scaled_std if rgb_scaled_std > 0 else float('inf')
            mean_ratio = abs(polar_scaled_mean) / abs(rgb_scaled_mean) if rgb_scaled_mean != 0 else float('inf')
            
            print(f"  - 缩放后标准差比例 (polar/rgb): {std_ratio:.4f}")
            print(f"  - 缩放后均值比例 (polar/rgb): {mean_ratio:.4f}")
            
            if std_ratio > 2.0:
                print(f"  ⚠️  警告: Polar 特征的标准差远大于 RGB，可能需要降低 polar_scale")
            elif std_ratio < 0.5:
                print(f"  ⚠️  警告: Polar 特征的标准差远小于 RGB，可能需要提高 polar_scale")
            else:
                print(f"  ✓ 标准差比例合理（0.5 - 2.0 之间）")
            
            if abs(polar_scaled_mean) > abs(rgb_scaled_mean) * 2:
                print(f"  ⚠️  警告: Polar 特征的均值绝对值远大于 RGB，可能需要调整 polar_scale")
            else:
                print(f"  ✓ 均值比例合理")
            
            # 8. [新增] LayerNorm 后的总结
            if hasattr(self.multi_modal_projector, 'input_layernorm'):
                print(f"\n💡 LayerNorm 效果总结:")
                print(f"  ✓ LayerNorm 在拼接后、Linear 前应用，会将输入标准化为均值 0、方差 1")
                print(f"  ✓ 即使 RGB 和 Polar 特征的数值范围差异很大，LayerNorm 也会将其抹平")
                print(f"  ✓ rgb_scale 和 polar_scale 仍然有用（控制特征的重要性），但数值差异问题已解决")
                print(f"  ✓ 训练应该更加稳定，不会因为数值差异导致梯度问题")
            
            print("=" * 80 + "\n")
            
            # 标记已输出，避免重复打印
            self._feature_range_printed = True
        
        # 4. 投影到 LLM 空间（在 MultiModalProjector 内部进行动态平衡和拼接）
        # 注意：缩放和拼接现在在 MultiModalProjector.forward 中完成
        # 这样做的好处：
        # - rgb_scale 和 polar_scale 参数会自动包含在 projector.state_dict() 中
        # - 保存和加载时无需额外处理，避免参数丢失
        # - 更符合模块化设计
        projected_features = self.multi_modal_projector(
            rgb_features,  # (B, 256, 1024)
            polar_features_aligned  # (B, 256, 768)
        )  # (B, 256, 4096)
        
        return projected_features
    
    def forward(
        self,
        pixel_values_rgb: torch.Tensor,
        pixel_values_polar: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        采用"插入并扩展"方案处理视觉特征：
        - 如果知道 image_token_id，精准替换 <image> token
        - 否则，在序列开头插入视觉特征，同时扩展 attention_mask 和 labels
        
        Args:
            pixel_values_rgb: RGB图像，形状为 (B, C, H, W)
            pixel_values_polar: 偏振图像，形状为 (B, 4, H, W)，4通道：[Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]
            input_ids: 输入token IDs，形状为 (B, L)
            attention_mask: 注意力掩码，形状为 (B, L)
            labels: 标签，形状为 (B, L)，用于计算损失
            
        Returns:
            模型输出（包含loss和logits）
        """
        # 1. 提取视觉特征 (B, N_vision, D) - 例如 (B, 256, 4096)
        # 注意：vision_features 现在包含 256 个空间对齐的 patch tokens，每个 token 同时包含 RGB 和 Polar 信息
        vision_features = self.get_vision_features(
            pixel_values_rgb,
            pixel_values_polar
        )
        
        # 2. 获取文本嵌入 (B, L, D)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # 确保 vision_features 和 input_embeds 的 dtype 一致
        # 在 4-bit 量化模型中，input_embeds 可能是 float16（因为 bnb_4bit_compute_dtype=float16）
        # 需要确保 vision_features 也是相同的 dtype
        if vision_features.dtype != input_embeds.dtype:
            vision_features = vision_features.to(dtype=input_embeds.dtype)
        
        # 3. 插入视觉特征（关键修复：正确处理 BOS token 位置）
        # 
        # ⚠️ 重要说明：
        # - dataset_stage2.py 的 collate_fn_stage2 已经手动添加了 BOS token，并设置 add_special_tokens=False
        # - 正确的序列结构应该是：[BOS] + [Vision] + [Text]
        # - LLM 依赖 BOS token 来重置注意力状态，必须放在最前面
        # 
        # 方案选择：
        # - 如果 self.image_token_id 已设置且 input_ids 包含 <image> token：使用替换方案（更符合 LLaVA 标准）
        # - 否则：检查 BOS token，将视觉特征插在 BOS 后面（推荐方案）
        
        # 检查是否使用替换方案
        use_replace_mode = False
        if self.image_token_id is not None:
            # 检查 input_ids 中是否包含 <image> token
            has_image_token = (input_ids == self.image_token_id).any().item()
            if has_image_token:
                use_replace_mode = True
        
        if use_replace_mode:
            # 替换方案：在 <image> token 位置插入视觉特征（更符合 LLaVA 标准）
            # ⚠️ 注意：vision_features 是多 token (452个)，插入后会导致序列长度变化
            # 这需要重新 pad 所有样本到相同长度，但为了简化，这里假设所有样本的 <image> token 位置相同
            # 如果位置不同，需要在 collate_fn 中处理
            
            inputs_embeds_list = []
            attention_mask_list = []
            labels_list = []
            
            for b in range(input_ids.shape[0]):
                # 找到该样本中 <image> token 的位置索引
                img_indices = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
                if len(img_indices) > 0:
                    idx = img_indices[0].item()
                    # 在 <image> 位置插入所有视觉 tokens，并删除 <image> token
                    before = input_embeds[b, :idx]
                    after = input_embeds[b, idx+1:]
                    combined = torch.cat([before, vision_features[b], after], dim=0)
                    inputs_embeds_list.append(combined)
                    
                    # 同步更新 attention_mask 和 labels
                    if attention_mask is not None:
                        before_mask = attention_mask[b, :idx]
                        after_mask = attention_mask[b, idx+1:]
                        vision_mask = torch.ones(
                            vision_features.shape[1],
                            device=attention_mask.device,
                            dtype=attention_mask.dtype
                        )
                        combined_mask = torch.cat([before_mask, vision_mask, after_mask], dim=0)
                        attention_mask_list.append(combined_mask)
                    
                    if labels is not None:
                        before_labels = labels[b, :idx]
                        after_labels = labels[b, idx+1:]
                        vision_labels = torch.full(
                            (vision_features.shape[1],),
                            -100,
                            device=labels.device,
                            dtype=labels.dtype
                        )
                        combined_labels = torch.cat([before_labels, vision_labels, after_labels], dim=0)
                        labels_list.append(combined_labels)
                else:
                    # 如果没有 <image> token，回退到插入方案
                    inputs_embeds_list.append(torch.cat([vision_features[b], input_embeds[b]], dim=0))
                    if attention_mask is not None:
                        vision_mask = torch.ones(
                            vision_features.shape[1],
                            device=attention_mask.device,
                            dtype=attention_mask.dtype
                        )
                        attention_mask_list.append(torch.cat([vision_mask, attention_mask[b]], dim=0))
                    if labels is not None:
                        vision_labels = torch.full(
                            (vision_features.shape[1],),
                            -100,
                            device=labels.device,
                            dtype=labels.dtype
                        )
                        labels_list.append(torch.cat([vision_labels, labels[b]], dim=0))
            
            # 重新堆叠（需要 pad 到相同长度）
            max_len = max(emb.shape[0] for emb in inputs_embeds_list)
            batch_size = len(inputs_embeds_list)
            embed_dim = inputs_embeds_list[0].shape[1]
            
            inputs_embeds = torch.zeros(
                (batch_size, max_len, embed_dim),
                device=inputs_embeds_list[0].device,
                dtype=inputs_embeds_list[0].dtype
            )
            
            for b, emb in enumerate(inputs_embeds_list):
                inputs_embeds[b, :emb.shape[0]] = emb
            
            if attention_mask is not None:
                attention_mask = torch.zeros(
                    (batch_size, max_len),
                    device=attention_mask_list[0].device,
                    dtype=attention_mask_list[0].dtype
                )
                for b, mask in enumerate(attention_mask_list):
                    attention_mask[b, :mask.shape[0]] = mask
            
            if labels is not None:
                labels = torch.full(
                    (batch_size, max_len),
                    -100,
                    device=labels_list[0].device,
                    dtype=labels_list[0].dtype
                )
                for b, lbl in enumerate(labels_list):
                    labels[b, :lbl.shape[0]] = lbl
        else:
            # 默认方案：检查 BOS token 位置，将视觉特征插在 BOS 后面
            # 正确结构：[BOS] + [Vision] + [Text]
            # 这与 dataset_stage2.py 的 collate_fn_stage2 逻辑一致（手动添加 BOS，add_special_tokens=False）
            
            # 获取 BOS token ID
            bos_token_id = None
            if hasattr(self.language_model, 'config') and hasattr(self.language_model.config, 'bos_token_id'):
                bos_token_id = self.language_model.config.bos_token_id
            
            # 检查 input_ids 是否以 BOS 开头（batch 中所有样本的第一个 token 都应该是 BOS）
            has_bos = False
            if bos_token_id is not None and input_ids.shape[0] > 0:
                # 检查第一个样本的第一个 token 是否是 BOS（假设 batch 中所有样本都以 BOS 开头）
                has_bos = (input_ids[:, 0] == bos_token_id).all().item() if input_ids.numel() > 0 else False
            
            if has_bos:
                # 如果有 BOS，把视觉特征插在 BOS 后面：[BOS] + [Vision] + [Text]
                bos_embeds = input_embeds[:, :1, :]  # (B, 1, D) - BOS token 的 embedding
                text_embeds = input_embeds[:, 1:, :]  # (B, L-1, D) - 剩余文本的 embedding
                
                inputs_embeds = torch.cat([bos_embeds, vision_features, text_embeds], dim=1)
                
                # 同步处理 attention_mask
                if attention_mask is not None:
                    bos_mask = attention_mask[:, :1]  # (B, 1)
                    text_mask = attention_mask[:, 1:]  # (B, L-1)
                    vision_mask = torch.ones(
                        (attention_mask.shape[0], vision_features.shape[1]),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype
                    )
                    attention_mask = torch.cat([bos_mask, vision_mask, text_mask], dim=1)
                
                # 同步处理 labels
                if labels is not None:
                    bos_label = labels[:, :1]  # (B, 1) - BOS token 的 label（通常也是 BOS token ID）
                    text_label = labels[:, 1:]  # (B, L-1) - 剩余文本的 label
                    vision_label = torch.full(
                        (labels.shape[0], vision_features.shape[1]),
                        -100,  # Ignore Index，视觉特征不计算损失
                        device=labels.device,
                        dtype=labels.dtype
                    )
                    labels = torch.cat([bos_label, vision_label, text_label], dim=1)
            else:
                # 如果没有 BOS（不推荐，但为了兼容性保留），则直接插在最前面
                # 这种情况不应该发生，因为 collate_fn_stage2 已经手动添加了 BOS
                inputs_embeds = torch.cat([vision_features, input_embeds], dim=1)
                
                # 同时，必须同步更新 attention_mask 和 labels
                if attention_mask is not None:
                    # 在前面补 1（视觉特征需要参与注意力计算）
                    ones = torch.ones(
                        (attention_mask.shape[0], vision_features.shape[1]),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype
                    )
                    attention_mask = torch.cat([ones, attention_mask], dim=1)
                
                if labels is not None:
                    # 在前面补 -100 (Ignore Index，视觉特征不计算损失)
                    ignore = torch.full(
                        (labels.shape[0], vision_features.shape[1]),
                        -100,
                        device=labels.device,
                        dtype=labels.dtype
                    )
                    labels = torch.cat([ignore, labels], dim=1)
        
        # 4. 前向传播LLM
        # 确保 inputs_embeds 的 dtype 与 LLM 的 compute dtype 匹配
        # 在 4-bit 量化模型中，如果使用 bnb_4bit_compute_dtype=float16，
        # 需要确保 inputs_embeds 也是 float16
        if hasattr(self.language_model, 'config') and hasattr(self.language_model.config, 'quantization_config'):
            # 如果是量化模型，检查 compute dtype
            quant_config = self.language_model.config.quantization_config
            if hasattr(quant_config, 'bnb_4bit_compute_dtype') and quant_config.bnb_4bit_compute_dtype is not None:
                target_dtype = quant_config.bnb_4bit_compute_dtype
                if inputs_embeds.dtype != target_dtype:
                    inputs_embeds = inputs_embeds.to(dtype=target_dtype)
        
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return outputs
    
    def generate(
        self,
        pixel_values_rgb: torch.Tensor,
        pixel_values_polar: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        """
        生成文本
        
        使用与 forward 相同的逻辑处理视觉特征
        
        Args:
            pixel_values_rgb: RGB图像，形状为 (B, C, H, W)
            pixel_values_polar: 偏振图像，形状为 (B, 4, H, W)，4通道：[Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]
            input_ids: 输入token IDs
            attention_mask: 注意力掩码（可选）
            **generation_kwargs: 生成参数（max_length, temperature等）
            
        Returns:
            生成的token IDs
        """
        # 1. 提取视觉特征 (B, N_vision, D) - 例如 (B, 256, 4096)
        vision_features = self.get_vision_features(
            pixel_values_rgb,
            pixel_values_polar
        )
        
        # 2. 获取文本嵌入 (B, L, D)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # 3. 插入视觉特征（正确处理 BOS token 位置）
        # 正确结构：[BOS] + [Vision] + [Text]
        # 注意：vision_features 现在是 (B, 256, 4096)，每个 token 同时包含 RGB 和 Polar 信息
        
        # 获取 BOS token ID
        bos_token_id = None
        if hasattr(self.language_model, 'config') and hasattr(self.language_model.config, 'bos_token_id'):
            bos_token_id = self.language_model.config.bos_token_id
        
        # 检查 input_ids 是否以 BOS 开头
        has_bos = False
        if bos_token_id is not None and input_ids.shape[0] > 0:
            has_bos = (input_ids[:, 0] == bos_token_id).all().item() if input_ids.numel() > 0 else False
        
        if has_bos:
            # 如果有 BOS，把视觉特征插在 BOS 后面：[BOS] + [Vision] + [Text]
            bos_embeds = input_embeds[:, :1, :]  # (B, 1, D)
            text_embeds = input_embeds[:, 1:, :]  # (B, L-1, D)
            inputs_embeds = torch.cat([bos_embeds, vision_features, text_embeds], dim=1)
            
            # 同步更新 attention_mask
            if attention_mask is None:
                # 创建新的 attention_mask
                attention_mask = torch.ones(
                    (input_ids.shape[0], inputs_embeds.shape[1]),
                    dtype=torch.long,
                    device=input_ids.device
                )
            else:
                bos_mask = attention_mask[:, :1]  # (B, 1)
                text_mask = attention_mask[:, 1:]  # (B, L-1)
                vision_mask = torch.ones(
                    (attention_mask.shape[0], vision_features.shape[1]),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                attention_mask = torch.cat([bos_mask, vision_mask, text_mask], dim=1)
        else:
            # 如果没有 BOS（不推荐），则直接插在最前面
            inputs_embeds = torch.cat([vision_features, input_embeds], dim=1)
            
            # 同步更新 attention_mask
            if attention_mask is None:
                # 创建新的 attention_mask
                attention_mask = torch.ones(
                    (input_ids.shape[0], inputs_embeds.shape[1]),
                    dtype=torch.long,
                    device=input_ids.device
                )
            else:
                ones = torch.ones(
                    (attention_mask.shape[0], vision_features.shape[1]),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                attention_mask = torch.cat([ones, attention_mask], dim=1)
        
        # 4. 确保 inputs_embeds 的 dtype 与 LLM 的计算 dtype 一致（避免 float vs half 冲突）
        target_dtype = None
        if hasattr(self.language_model, "config") and hasattr(self.language_model.config, "quantization_config"):
            quant_config = self.language_model.config.quantization_config
            if hasattr(quant_config, "bnb_4bit_compute_dtype") and quant_config.bnb_4bit_compute_dtype is not None:
                target_dtype = quant_config.bnb_4bit_compute_dtype
        
        # 如果不是量化模型，或上面没拿到 dtype，则回退到模型参数的 dtype
        if target_dtype is None:
            try:
                target_dtype = next(self.language_model.parameters()).dtype
            except StopIteration:
                target_dtype = inputs_embeds.dtype
        
        if inputs_embeds.dtype != target_dtype:
            inputs_embeds = inputs_embeds.to(dtype=target_dtype)
        
        # 5. 生成文本
        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
        
        return outputs


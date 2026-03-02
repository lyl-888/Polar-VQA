"""
Stage 1 验证脚本：可视化 MAE 重建效果 + 输出量化指标

功能：
1. 加载训练好的 Stage 1 MAE 模型
2. 使用测试样本进行前向传播
3. 可视化原始图像、掩码图像和重建图像
4. 分别显示 DoLP、sin(2*AoLP)、cos(2*AoLP) 三个通道的重建效果
5. 计算并输出重建的量化指标（MSE / PSNR），支持多样本统计

输出：
- stage1_vis.png: 默认包含单个样本的原始图像和重建图像对比图
- 当指定多样本时，会生成多个可视化文件，并在终端打印每个样本的指标以及整体平均指标
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import ViTMAEConfig, ViTMAEForPreTraining

# Matplotlib 配置：只关闭负号乱码，不强制使用不存在的中文字体，避免 findfont 警告
plt.rcParams["axes.unicode_minus"] = False
# 导入共享的数据处理函数
from dataset_common import process_polar_images
from dataset_stage1 import PolarMAEDataset


def load_mae_model(checkpoint_path: str, hf_token: Optional[str] = None):
    """
    加载训练好的 Stage 1 MAE 模型或预训练模型（Zero-shot Baseline）
    
    Args:
        checkpoint_path: 
            - Stage 1 检查点路径（目录），例如 "/path/to/checkpoint"
            - 或 Hugging Face 模型名称，例如 "facebook/vit-mae-base"（用于 Zero-shot Baseline）
        hf_token: Hugging Face token（可选）
    
    Returns:
        加载的 MAE 模型
    """
    print("=" * 80)
    print("加载 MAE 模型")
    print("=" * 80)
    
    # 获取 token
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    load_kwargs = {}
    if token:
        load_kwargs["token"] = token
        print(f"✓ 使用 Hugging Face token")
    
    # 判断是本地路径还是 Hugging Face 模型名称
    # 如果路径不存在，且看起来像是 Hugging Face 模型名称（包含 "/" 且不是绝对路径），则尝试加载预训练模型
    is_hf_model = (
        not os.path.exists(checkpoint_path) and 
        "/" in checkpoint_path and 
        not os.path.isabs(checkpoint_path) and
        not checkpoint_path.startswith(".")  # 排除相对路径如 "./checkpoint"
    )
    
    # 如果路径不存在，且看起来像是 Hugging Face 模型名称，则尝试加载预训练模型
    if is_hf_model:
        print(f"检测到 Hugging Face 模型名称: {checkpoint_path}")
        print("正在加载预训练模型（Zero-shot Baseline）...")
        
        # 加载预训练配置
        config = ViTMAEConfig.from_pretrained(checkpoint_path, **load_kwargs)
        
        # 修改配置以适配3通道输入（DoLP, sin(2*AoLP), cos(2*AoLP)）
        config.num_channels = 3
        config.image_size = 224
        config.patch_size = 16
        config.norm_pix_loss = False  # 与训练时保持一致
        
        print(f"✓ 预训练模型配置:")
        print(f"  - 模型名称: {checkpoint_path}")
        print(f"  - 图像尺寸: {config.image_size}")
        print(f"  - Patch 尺寸: {config.patch_size}")
        print(f"  - 输入通道数: {config.num_channels} (DoLP, sin(2*AoLP), cos(2*AoLP))")
        print(f"  - 归一化像素损失: {config.norm_pix_loss}")
        print(f"  ⚠ 注意: 这是 Zero-shot Baseline，使用 ImageNet 预训练权重")
        
        # 加载预训练模型（会自动适配3通道输入）
        model = ViTMAEForPreTraining.from_pretrained(
            checkpoint_path,
            config=config,
            ignore_mismatched_sizes=True,  # 允许通道数不匹配
            **load_kwargs
        )
        print(f"✓ 已加载预训练模型（Zero-shot Baseline）")
        model.eval()
        return model
    
    # 原有逻辑：从本地检查点加载
    # 检查检查点路径
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点路径不存在: {checkpoint_path}")
    
    # 尝试加载模型文件
    model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if not os.path.exists(model_path):
        # 尝试 safetensors 格式
        model_path = os.path.join(checkpoint_path, "model.safetensors")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"在 {checkpoint_path} 中未找到模型文件")
    
    # 加载配置
    config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(config_path):
        config = ViTMAEConfig.from_pretrained(checkpoint_path, **load_kwargs)
        print(f"✓ 从检查点加载配置:")
        print(f"  - image_size: {config.image_size}")
        print(f"  - patch_size: {config.patch_size}")
        print(f"  - num_channels: {getattr(config, 'num_channels', 'N/A')}")
    else:
        # 如果没有配置文件，使用默认配置
        print("⚠ 警告: 未找到 config.json，使用默认配置")
        config = ViTMAEConfig()
        config.num_channels = 3  # 3通道输入（DoLP, sin(2*AoLP), cos(2*AoLP)），移除Intensity
        config.image_size = 224
        config.patch_size = 16
    
    # 确保配置正确（3通道：DoLP, sin(2*AoLP), cos(2*AoLP)）
    if not hasattr(config, 'num_channels') or config.num_channels != 3:
        config.num_channels = 3  # 3通道输入（与训练时一致）
    
    # ⚠️ 关键：不要强制覆盖 image_size 和 patch_size！
    # 如果检查点中的配置不同，说明模型是基于不同配置训练的
    # 强制覆盖会导致维度不匹配
    print(f"✓ 最终使用的配置:")
    print(f"  - 图像尺寸: {config.image_size}")
    print(f"  - Patch 尺寸: {config.patch_size}")
    if config.num_channels == 3:
        print(f"  - 输入通道数: {config.num_channels} (DoLP, sin(2*AoLP), cos(2*AoLP))")
    else:
        print(f"  - 输入通道数: {config.num_channels}")
    print(f"  - 预期 patch 数量: {(config.image_size // config.patch_size) ** 2}")
    
    # 加载模型
    if model_path.endswith(".safetensors"):
        # 如果是 safetensors，需要先创建模型再加载权重
        model = ViTMAEForPreTraining(config)
        
        import safetensors.torch
        state_dict = safetensors.torch.load_file(model_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"✓ 已从 safetensors 加载模型权重")
        
    else:
        # 直接加载
        model = ViTMAEForPreTraining.from_pretrained(
            checkpoint_path,
            config=config,
            ignore_mismatched_sizes=True,
            **load_kwargs
        )
        print(f"✓ 已加载模型权重")
    
    model.eval()  # 设置为评估模式
    return model


def preprocess_image(physics_img: np.ndarray, image_size: int = 224, num_channels: int = 3) -> torch.Tensor:
    """
    预处理图像：转换为模型输入格式
    
    Args:
        physics_img: 4通道物理参数图像 (H, W, 4)，值范围 [0, 1]
                    通道顺序：[Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]
                    如果 num_channels=3，只使用后3个通道（DoLP, sin, cos）
        image_size: 目标图像尺寸
        num_channels: 输出通道数（默认3，只使用DoLP, sin, cos）
    
    Returns:
        预处理后的图像张量 (1, num_channels, H, W)
    """
    # 如果只需要3通道，只取后3个通道（DoLP, sin, cos）
    if num_channels == 3 and physics_img.shape[2] == 4:
        physics_img = physics_img[:, :, 1:4]  # 去掉Intensity通道
    
    # 转换为 PIL Image（需要 uint8 格式）
    physics_img_uint8 = (physics_img * 255).astype(np.uint8)
    # 根据通道数选择模式
    if num_channels == 3:
        # 3通道：使用RGB模式（虽然数据不是RGB，但PIL支持3通道）
        physics_pil = Image.fromarray(physics_img_uint8, mode='RGB')
    else:
        # 4通道：使用RGBA模式
        physics_pil = Image.fromarray(physics_img_uint8, mode='RGBA')
    
    # Resize 到目标尺寸
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # 注意：MAE 模型期望的归一化可能不同，这里先不归一化
        # 因为 process_polar_images 已经将值归一化到 [0, 1]
    ])
    
    img_tensor = transform(physics_pil)  # (num_channels, H, W)
    img_tensor = img_tensor.unsqueeze(0)  # (1, num_channels, H, W)
    
    return img_tensor


def compute_metrics(original_img: np.ndarray, reconstructed_img: np.ndarray, 
                    mask: Optional[np.ndarray] = None, patch_size: int = 16, num_channels: int = 3) -> Dict[str, float]:
    """
    计算重建的量化指标（逐通道 + 整体）：
    - MSE: Mean Squared Error
    - PSNR: Peak Signal-to-Noise Ratio（峰值信噪比，单位 dB，值越大越好）
    
    ⚠️ 关键修复：支持只计算 Masked 区域的指标，与 Training Loss 对齐

    Args:
        original_img: 原始图像 (num_channels, H, W)，值范围 [0, 1]
                      如果 num_channels=3：通道顺序：[DoLP, sin(2*AoLP), cos(2*AoLP)]
                      如果 num_channels=4：通道顺序：[Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]
        reconstructed_img: 重建图像 (num_channels, H, W)，值范围 [0, 1]
        mask: 掩码 (num_patches,)，1 表示被掩码（需要预测），0 表示可见
              如果提供，会同时计算全图和 Masked 区域的指标
        patch_size: Patch 尺寸，用于将 mask 扩展到图像尺寸
        num_channels: 通道数（默认3）

    Returns:
        包含逐通道和整体指标的字典（如果提供了 mask，会包含 masked 区域的指标）
    """
    # 避免数值问题，先裁剪
    original = np.clip(original_img, 0, 1)
    recon = np.clip(reconstructed_img, 0, 1)

    # 计算全图的指标
    mse_per_channel_full = ((original - recon) ** 2).mean(axis=(1, 2))  # (num_channels,)
    mse_overall_full = mse_per_channel_full.mean()

    # PSNR 计算：PSNR = 10 * log10(MAX^2 / MSE)，这里 MAX=1
    eps = 1e-8
    psnr_per_channel_full = 10.0 * np.log10(1.0 / (mse_per_channel_full + eps))
    psnr_overall_full = 10.0 * np.log10(1.0 / (mse_overall_full + eps))

    # 根据通道数构建指标字典
    if num_channels == 3:
        metrics = {
            # 全图指标（3通道）
            "mse_dolp": float(mse_per_channel_full[0]),
            "mse_sin_2aolp": float(mse_per_channel_full[1]),
            "mse_cos_2aolp": float(mse_per_channel_full[2]),
            "mse_mean": float(mse_overall_full),
            "psnr_dolp": float(psnr_per_channel_full[0]),
            "psnr_sin_2aolp": float(psnr_per_channel_full[1]),
            "psnr_cos_2aolp": float(psnr_per_channel_full[2]),
            "psnr_mean": float(psnr_overall_full),
        }
    else:
        # 4通道模式（向后兼容）
        metrics = {
            "mse_intensity": float(mse_per_channel_full[0]),
            "mse_dolp": float(mse_per_channel_full[1]),
            "mse_sin_2aolp": float(mse_per_channel_full[2]),
            "mse_cos_2aolp": float(mse_per_channel_full[3]),
            "mse_mean": float(mse_overall_full),
            "psnr_intensity": float(psnr_per_channel_full[0]),
            "psnr_dolp": float(psnr_per_channel_full[1]),
            "psnr_sin_2aolp": float(psnr_per_channel_full[2]),
            "psnr_cos_2aolp": float(psnr_per_channel_full[3]),
            "psnr_mean": float(psnr_overall_full),
        }
    
    # 如果提供了 mask，计算 Masked 区域的指标（与 Training Loss 对齐）
    if mask is not None:
        # 将 mask 从 patch 级别扩展到像素级别
        # mask: (num_patches,) -> (h_patches, w_patches) -> (H, W)
        h_patches = w_patches = int(np.sqrt(len(mask)))
        mask_2d = mask.reshape(h_patches, w_patches)
        
        # 扩展到图像尺寸
        mask_expanded = np.repeat(np.repeat(mask_2d, patch_size, axis=0), patch_size, axis=1)
        # mask_expanded: (H, W)，1 表示被掩码的区域
        
        # 只计算被掩码区域的 MSE
        masked_mse_per_channel = []
        for c in range(num_channels):
            diff_sq = (original[c] - recon[c]) ** 2
            masked_mse = (diff_sq * mask_expanded).sum() / (mask_expanded.sum() + eps)
            masked_mse_per_channel.append(float(masked_mse))
        
        masked_mse_per_channel = np.array(masked_mse_per_channel)
        masked_mse_overall = masked_mse_per_channel.mean()
        
        # 计算 Masked 区域的 PSNR
        masked_psnr_per_channel = 10.0 * np.log10(1.0 / (masked_mse_per_channel + eps))
        masked_psnr_overall = 10.0 * np.log10(1.0 / (masked_mse_overall + eps))
        
        # 根据通道数添加到 metrics
        if num_channels == 3:
            metrics.update({
                # Masked 区域指标（与 Training Loss 对齐，3通道）
                "mse_masked_dolp": float(masked_mse_per_channel[0]),
                "mse_masked_sin_2aolp": float(masked_mse_per_channel[1]),
                "mse_masked_cos_2aolp": float(masked_mse_per_channel[2]),
                "mse_masked_mean": float(masked_mse_overall),
                "psnr_masked_dolp": float(masked_psnr_per_channel[0]),
                "psnr_masked_sin_2aolp": float(masked_psnr_per_channel[1]),
                "psnr_masked_cos_2aolp": float(masked_psnr_per_channel[2]),
                "psnr_masked_mean": float(masked_psnr_overall),
            })
        else:
            # 4通道模式（向后兼容）
            metrics.update({
                "mse_masked_intensity": float(masked_mse_per_channel[0]),
                "mse_masked_dolp": float(masked_mse_per_channel[1]),
                "mse_masked_sin_2aolp": float(masked_mse_per_channel[2]),
                "mse_masked_cos_2aolp": float(masked_mse_per_channel[3]),
                "mse_masked_mean": float(masked_mse_overall),
                "psnr_masked_intensity": float(masked_psnr_per_channel[0]),
                "psnr_masked_dolp": float(masked_psnr_per_channel[1]),
                "psnr_masked_sin_2aolp": float(masked_psnr_per_channel[2]),
                "psnr_masked_cos_2aolp": float(masked_psnr_per_channel[3]),
                "psnr_masked_mean": float(masked_psnr_overall),
            })
    
    return metrics


def evaluate_quality(metrics: Dict[str, float]) -> tuple:
    """
    评估训练质量，提供明确的"好/中/差"判断
    
    ⚠️ 关键修复：优先使用 Masked PSNR 进行评估（与 Loss 对齐）
    
    Args:
        metrics: 包含 MSE 和 PSNR 指标的字典
    
    Returns:
        (quality_level, recommendation): 质量等级和建议
    """
    # 优先使用 Masked 区域的 PSNR（与 Loss 对齐）
    if 'psnr_masked_mean' in metrics:
        psnr_mean = metrics['psnr_masked_mean']  # 使用 Masked 区域的 PSNR
        mse_mean = metrics['mse_masked_mean']
    else:
        psnr_mean = metrics['psnr_mean']  # 使用全图的 PSNR（不推荐）
        mse_mean = metrics['mse_mean']
    
    # 质量判断标准（基于 PSNR）
    if psnr_mean >= 25.0:
        quality = "优秀"
        recommendation = "✓ 编码器质量很好，可以直接用于 Stage 2"
    elif psnr_mean >= 20.0:
        quality = "良好"
        recommendation = "✓ 编码器质量良好，可以用于 Stage 2"
    elif psnr_mean >= 15.0:
        quality = "中等"
        recommendation = "⚠ 编码器质量中等，但 Loss 很低，可以用于 Stage 2（偏暗不影响特征提取）"
    else:
        quality = "较差"
        recommendation = "❌ 编码器质量较差，建议重新训练 Stage 1（检查数据、配置、训练参数）"
    
    return quality, recommendation


def extract_encoder_features(model: ViTMAEForPreTraining, pixel_values: torch.Tensor) -> Dict:
    """
    提取编码器特征（用于评估 Stage 2 适用性）
    
    ⚠️ 关键修复：MAE 模型的 `vit` 编码器只处理可见的（unmasked）patches
    要获取所有 patches 的特征，需要直接访问 ViT 的 embeddings 和 encoder，绕过 MAE 的 mask 逻辑
    
    Args:
        model: MAE 模型
        pixel_values: 输入图像 (1, 4, 224, 224)
    
    Returns:
        包含编码器特征统计信息的字典
    """
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(device)
    
    with torch.no_grad():
        # ⚠️ 关键修复：手动构建完整的 embeddings，绕过 MAE 的 mask 逻辑
        # MAE 模型的 `model.vit.embeddings()` 会在 forward 时应用 mask，只处理可见的 patches
        # 但 Stage 2 需要所有 patches 的特征，所以需要手动构建完整的 embeddings
        
        # 方法：手动构建完整的 embeddings（所有 patches）
        # 1. 获取 patch embeddings（所有 patches，不应用 mask）
        patch_emb = model.vit.embeddings.patch_embeddings(pixel_values)  # (B, 196, 768)
        
        # 2. 添加 CLS token
        cls_token = model.vit.embeddings.cls_token.expand(pixel_values.shape[0], -1, -1)  # (B, 1, 768)
        embeddings = torch.cat([cls_token, patch_emb], dim=1)  # (B, 197, 768)
        
        # 3. 添加位置编码
        embeddings = embeddings + model.vit.embeddings.position_embeddings  # (B, 197, 768)
        
        # embeddings 形状: (B, 197, 768) - 包含所有 patches + CLS token
        
        # 2. 通过 encoder（所有 patches）
        encoder_outputs = model.vit.encoder(embeddings)
        # encoder() 可能返回 tuple，需要处理
        if isinstance(encoder_outputs, tuple):
            features = encoder_outputs[0]  # 取第一个元素（last_hidden_state）
        else:
            features = encoder_outputs.last_hidden_state if hasattr(encoder_outputs, 'last_hidden_state') else encoder_outputs
        # features 形状: (B, 197, 768) - 包含所有 patches
        
        # 去掉 CLS token，只保留 patch tokens（与 Stage 2 一致）
        patch_features = features[:, 1:, :]  # (B, 196, 768)
        
        # 统计信息
        mean = patch_features.mean().item()
        std = patch_features.std().item()
        min_val = patch_features.min().item()
        max_val = patch_features.max().item()
        
        # 评估特征分布是否正常
        # 注意：ViT 编码器输出的特征分布特点：
        # - mean 应该接近 0（LayerNorm 的效果）
        # - std 通常在 1-4 之间（经过多层 transformer 后，方差可能累积）
        # - 对于 768 维的特征，std 在 2-3 之间是正常的
        if abs(mean) < 0.5 and 0.5 < std < 5.0:
            if abs(mean) < 0.1 and 1.0 < std < 3.0:
                distribution_status = "✓ 特征分布正常，适合 Stage 2"
            else:
                distribution_status = f"✓ 特征分布可接受 (mean={mean:.3f}, std={std:.3f})，适合 Stage 2"
        else:
            distribution_status = f"⚠ 特征分布异常 (mean={mean:.3f}, std={std:.3f})，可能影响 Stage 2 训练"
        
        return {
            'features': patch_features,
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
            'distribution_status': distribution_status,
        }


def visualize_mae_reconstruction(
    model: ViTMAEForPreTraining,
    pixel_values: torch.Tensor,
    output_path: str = "stage1_vis.png",
) -> Dict[str, float]:
    """
    可视化 MAE 重建效果，并计算量化指标（MSE / PSNR）。
    
    Args:
        model: MAE 模型
        pixel_values: 输入图像 (1, num_channels, 224, 224)
                      如果 num_channels=3：通道顺序：[DoLP, sin(2*AoLP), cos(2*AoLP)]
                      如果 num_channels=4：通道顺序：[Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]
        output_path: 输出图像路径

    Returns:
        metrics: dict，包含本次样本的 MSE / PSNR 指标
    """
    print("\n" + "=" * 80)
    print("进行 MAE 前向传播...")
    print("=" * 80)
    
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(device)
    
    # 获取模型配置（需要在 Unpatchify 自测之前）
    config = model.config
    
    # ========== 关键诊断：Unpatchify 逻辑自测 ==========
    # 目的：验证 unpatchify 逻辑是否正确
    # 方法：将原图手动 patchify，再用 unpatchify 拼回去，检查是否完全一致
    print("\n" + "=" * 80)
    print("🔍 Unpatchify 逻辑自测（关键诊断）")
    print("=" * 80)
    
    B, C, H, W = pixel_values.shape
    P = config.patch_size
    H_patches = H // P
    W_patches = W // P
    
    print(f"  - 输入图像形状: {pixel_values.shape} (B={B}, C={C}, H={H}, W={W})")
    print(f"  - Patch 尺寸: {P}x{P}")
    print(f"  - Patch 网格: {H_patches}x{W_patches} = {H_patches * W_patches} patches")
    
    # 1. 手动模拟 patchify（使用与 HuggingFace ViT 相同的方式）
    # HuggingFace 的 Conv2d patch embedding: kernel_size=patch_size, stride=patch_size
    # 这相当于将图像分成 (H//P) x (W//P) 个 patches，每个 patch 是 PxP 的块
    # 然后 flatten 每个 patch 为 P*P*C 的向量
    
    # 方法：使用 reshape + permute 模拟 patchify
    # (B, C, H, W) -> (B, C, H_patches, P, W_patches, P) -> (B, H_patches, W_patches, P, P, C) -> (B, N, P*P*C)
    input_reshaped = pixel_values.reshape(B, C, H_patches, P, W_patches, P)
    # 重排维度：(B, C, H_p, P, W_p, P) -> (B, H_p, W_p, P, P, C)
    patches_test = input_reshaped.permute(0, 2, 4, 3, 5, 1)  # (B, H_p, W_p, P, P, C)
    # Flatten 为 (B, N, D)
    patches_flatten = patches_test.reshape(B, H_patches * W_patches, P * P * C)
    
    print(f"  ✓ 手动 patchify 完成: {patches_flatten.shape} (B={B}, N={H_patches*W_patches}, D={P*P*C})")
    
    # 2. 使用你的 unpatchify 逻辑拼回去
    x_test = patches_flatten.reshape(B, H_patches, W_patches, -1)  # (B, H_p, W_p, P*P*C)
    x_test = x_test.reshape(B, H_patches, W_patches, P, P, C)  # (B, H_p, W_p, P, P, C)
    
    # 使用 einsum 进行维度重排（与主逻辑一致）
    x_test = torch.einsum('nhwpqc->nchpwq', x_test)  # (B, C, H_p, P, W_p, P)
    reconstructed_test = x_test.reshape(B, C, H_patches * P, W_patches * P)  # (B, C, H, W)
    
    print(f"  ✓ Unpatchify 完成: {reconstructed_test.shape}")
    
    # 3. 检查是否完全一致
    diff = (pixel_values - reconstructed_test).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    sum_diff = diff.sum().item()
    
    print(f"\n  🔍 自测结果:")
    print(f"    - 最大差异: {max_diff:.8f}")
    print(f"    - 平均差异: {mean_diff:.8f}")
    print(f"    - 总差异: {sum_diff:.8f}")
    
    if max_diff < 1e-6 and mean_diff < 1e-8:
        print(f"    ✅ Unpatchify 逻辑正确！原图切开再拼回去完全一致（差异 < 1e-6）")
        print(f"    → 问题不在 unpatchify 代码，可能在模型输出或数据对齐上")
    elif max_diff < 1e-3:
        print(f"    ⚠ 警告：Unpatchify 逻辑有小误差（差异 < 1e-3），可能是数值精度问题")
        print(f"    → 继续检查其他问题")
    else:
        print(f"    ❌ Unpatchify 逻辑错误！原图切开再拼回去差异很大（{max_diff:.6f}）")
        print(f"    → 问题一定出在 unpatchify 代码逻辑上，需要修复 einsum 公式")
        print(f"    → 建议：检查 patchify 和 unpatchify 的维度变换是否匹配")
    
    print("=" * 80 + "\n")
    
    # 前向传播
    with torch.no_grad():
        # 检查输入图像的实际尺寸
        print(f"  - 输入图像形状: {pixel_values.shape}")
        
        # 前向传播
        outputs = model(pixel_values)
    
    # 获取输出
    loss = outputs.loss.item()
    logits = outputs.logits  # 重建的 patch tokens
    ids_restore = outputs.ids_restore  # 恢复的 patch 顺序
    mask = outputs.mask  # 掩码（哪些 patch 被掩码）
    
    # ⚠️ 关键诊断：检查HuggingFace的loss计算方式
    # HuggingFace的MAE loss可能使用了特殊的计算方式，我们需要理解它
    # 尝试直接使用模型内部的loss计算逻辑来验证
    print(f"\n  🔍 关键诊断：检查模型内部的loss计算方式")
    print(f"    - outputs.loss: {loss:.6f}")
    
    # 检查outputs中是否有其他loss相关信息
    if hasattr(outputs, 'loss_detail'):
        print(f"    - outputs.loss_detail: {outputs.loss_detail}")
    
    # 尝试手动重新计算loss（使用与HuggingFace相同的方式）
    # HuggingFace的MAE loss计算方式：对每个被掩码的patch计算MSE，然后平均
    # 但可能使用了某种归一化或缩放
    
    # 检查 mask 的形状和数量
    print(f"  - mask 形状: {mask.shape}, 掩码的 patch 数量: {mask.sum().item()}")
    print(f"  - ids_restore 形状: {ids_restore.shape}")
    print(f"  - logits 形状: {logits.shape}")
    
    print(f"✓ 重建损失: {loss:.4f}")
    print(f"✓ 掩码比例: {mask.sum().item() / mask.numel():.2%}")
    print(f"✓ norm_pix_loss: {config.norm_pix_loss} (如果为True，重建值需要反归一化)")
    
    # 获取模型配置参数
    image_size = config.image_size
    patch_size = config.patch_size
    num_patches = (image_size // patch_size) ** 2
    h = w = image_size // patch_size
    num_channels = config.num_channels
    
    # ========== 关键修复：正确组合可见 patches 和预测 patches ==========
    # MAE 的 outputs.logits 可能只包含被掩码的 patches 的预测，也可能是全图的预测
    # 需要根据实际情况处理
    
    B = pixel_values.shape[0]
    num_total_patches = num_patches
    patch_dim = patch_size ** 2 * num_channels
    
    # 检查 logits 的形状
    # HuggingFace 的 ViTMAEForPreTraining 的 logits 通常已经是全图的预测
    # 但为了安全，我们检查一下
    print(f"  DEBUG: logits shape: {logits.shape}")
    print(f"  DEBUG: num_total_patches: {num_total_patches}")
    print(f"  DEBUG: mask shape: {mask.shape}")
    print(f"  DEBUG: ids_restore shape: {ids_restore.shape}")
    
    # ⚠️ 关键诊断：检查 ids_restore 的有效性
    ids_restore_np = ids_restore[0].cpu().numpy()
    mask_np = mask[0].cpu().numpy()
    print(f"  DEBUG: ids_restore 范围: [{ids_restore_np.min()}, {ids_restore_np.max()}]")
    print(f"  DEBUG: ids_restore 唯一值数量: {len(np.unique(ids_restore_np))} (期望: {num_total_patches})")
    print(f"  DEBUG: mask 中被掩码的 patch 数量: {mask_np.sum()} (期望: ~{num_total_patches * 0.5})")
    
    # 检查 ids_restore 是否包含所有索引（0 到 num_total_patches-1）
    if len(np.unique(ids_restore_np)) != num_total_patches:
        print(f"  ⚠ 警告: ids_restore 包含重复或缺失的索引！")
    if ids_restore_np.min() < 0 or ids_restore_np.max() >= num_total_patches:
        print(f"  ⚠ 警告: ids_restore 索引超出范围！")
    
    # 检查 ids_restore 的排列模式（用于诊断）
    # 如果 ids_restore 是 [0, 1, 2, ..., N-1]，说明已经是 Raster Order
    # 如果 ids_restore 是乱序的，说明需要重新排列
    is_raster_order = np.allclose(ids_restore_np, np.arange(num_total_patches))
    print(f"  DEBUG: ids_restore 是否已经是 Raster Order: {is_raster_order}")
    if not is_raster_order:
        print(f"  DEBUG: ids_restore 前10个值: {ids_restore_np[:10]}")
        print(f"  DEBUG: ids_restore 后10个值: {ids_restore_np[-10:]}")
    
    if logits.shape[1] == num_total_patches:
        # HuggingFace 的 logits 已经是全图的预测（已经组合了可见和预测的 patches）
        # 实践验证表明：此时 logits 已按原始 Raster Order 排列，
        # 再用 ids_restore 重排反而会打乱顺序，导致相关性接近 0。
        # 因此这里 **直接使用 logits 作为 pred_patches**，不再使用 ids_restore。
        print("  ✓ logits 已是全图预测，假定已按 Raster Order 排列，直接使用 logits 作为 pred_patches")
        pred_patches = logits  # (B, num_total_patches, patch_dim)
    else:
        # logits 只包含被掩码的 patches，需要手动组合
        print(f"  ⚠ logits 只包含 {logits.shape[1]} 个 patches（被掩码的部分），需要手动组合")
        
        # 1. 将输入图像 patchify（获取所有 patches 的像素值）
        patches_list = []
        for i in range(h):
            for j in range(w):
                i_start = i * patch_size
                i_end = (i + 1) * patch_size
                j_start = j * patch_size
                j_end = (j + 1) * patch_size
                patch = pixel_values[0, :, i_start:i_end, j_start:j_end]  # (4, patch_size, patch_size)
                patch_flat = patch.reshape(-1)  # (patch_size^2 * 4)
                patches_list.append(patch_flat)
        
        original_patches = torch.stack(patches_list, dim=0).unsqueeze(0)  # (1, num_total_patches, patch_dim)
        
        # 2. 创建一个全图的 tensor，初始化为原图的 patches
        pred_patches = original_patches.clone()  # (B, num_total_patches, patch_dim)
        
        # 3. 将预测的 patches（logits）放到被掩码的位置
        # ids_restore 告诉我们每个位置应该放哪个 patch
        # mask 告诉我们哪些位置是被掩码的（需要预测）
        # logits 只包含被掩码的 patches 的预测，顺序与 ids_restore 中被掩码的位置对应
        
        # 找到被掩码的 patches 在 ids_restore 中的位置
        mask_np = mask[0].cpu().numpy()  # (num_total_patches,)
        ids_restore_np = ids_restore[0].cpu().numpy()  # (num_total_patches,)
        
        # 找到所有被掩码的位置
        masked_positions = np.where(mask_np == 1)[0]  # 被掩码的原始位置索引
        
        # 将 logits 中的预测放到对应的位置
        # logits 的顺序应该与 ids_restore 中被掩码的位置顺序一致
        for i, masked_pos in enumerate(masked_positions):
            if i < logits.shape[1]:
                # 找到这个 masked_pos 在 ids_restore 中的位置
                restore_idx = np.where(ids_restore_np == masked_pos)[0]
                if len(restore_idx) > 0:
                    pred_patches[0, restore_idx[0]] = logits[0, i]
        
        # 4. 根据 ids_restore 恢复原始顺序
        restored_logits = torch.zeros_like(pred_patches)
        for i, orig_idx in enumerate(ids_restore_np):
            restored_logits[0, orig_idx] = pred_patches[0, i]
        pred_patches = restored_logits
    
    # 5. Unpatchify：将 patches 还原为图像
    # ⚠️ 关键修复：使用经过验证的 Unpatchify 逻辑
    # pred_patches: (B, num_total_patches, patch_dim)
    # patch_dim = patch_size * patch_size * num_channels
    # 
    # 关键假设：
    # 1. pred_patches 已经按 Raster Order 排列（从左到右、从上到下）
    # 2. patch 内部的像素按 (patch_size, patch_size, channels) 排列
    # 3. 需要正确地将 patch 网格和 patch 内部像素组合成完整图像
    
    # 定义形状变量
    p = patch_size
    c = config.num_channels
    h_patches = h
    w_patches = w
    
    # ⚠️ 关键修复：使用用户建议的经过验证的 unpatchify 方式
    # 步骤1：将 (B, N, D) reshape 为 (B, h_patches, w_patches, p, p, c)
    # 假设 patches 是按 Raster Order 排列的：pred_patches[0, i*w_patches + j] 对应位置 (i, j) 的patch
    x = pred_patches.reshape(B, h_patches, w_patches, -1)  # (B, h, w, patch_dim)
    x = x.reshape(B, h_patches, w_patches, p, p, c)  # (B, h, w, p, p, c)
    
    # 步骤2：使用 einsum 进行维度重排
    # 从 (B, h, w, p, p, c) 到 (B, c, h*p, w*p)
    # 'nhwpqc->nchpwq' 的含义：
    #   n: batch
    #   h: patch row index (height patches)
    #   w: patch col index (width patches)  
    #   p: pixel row index (inside patch)
    #   q: pixel col index (inside patch)
    #   c: channel
    # 变换: nhwpqc -> nchpwq，然后 reshape 到 (B, c, h*p, w*p)
    x = torch.einsum('nhwpqc->nchpwq', x)
    reconstructed_patches = x.reshape(B, c, h_patches * p, w_patches * p)
    
    # ⚠️ 诊断：验证unpatchify是否正确
    # 检查重建图像的形状和范围
    print(f"  DEBUG: unpatchify后形状: {reconstructed_patches.shape}")
    print(f"  DEBUG: unpatchify后范围: [{reconstructed_patches.min():.4f}, {reconstructed_patches.max():.4f}]")
    
    # ========== 关键修复：处理 norm_pix_loss 的情况 ==========
    # 当 norm_pix_loss=True 时，MAE 预测的是归一化后的像素值
    # 需要从原始图像中提取每个 patch 的均值和标准差，然后反归一化
    original_img_tensor = pixel_values[0]  # (num_channels, 224, 224)
    
    if config.norm_pix_loss:
        # 需要反归一化：denormalized = normalized * std + mean
        # 对每个 patch 计算均值和标准差
        reconstructed_denorm = torch.zeros_like(reconstructed_patches[0])
        
        for i in range(h):
            for j in range(w):
                # 原始 patch 的位置
                i_start = i * patch_size
                i_end = (i + 1) * patch_size
                j_start = j * patch_size
                j_end = (j + 1) * patch_size
                
                # 提取原始 patch
                original_patch = original_img_tensor[:, i_start:i_end, j_start:j_end]  # (num_channels, patch_size, patch_size)
                # 计算均值和标准差（对每个通道分别计算）
                patch_mean = original_patch.mean(dim=(1, 2), keepdim=True)  # (num_channels, 1, 1)
                patch_std = original_patch.std(dim=(1, 2), keepdim=True) + 1e-6  # (num_channels, 1, 1)，避免除零
                
                # 提取重建的 patch（归一化后的值）
                reconstructed_patch_norm = reconstructed_patches[0, :, i_start:i_end, j_start:j_end]  # (num_channels, patch_size, patch_size)
                
                # 反归一化：denormalized = normalized * std + mean
                reconstructed_patch_denorm = reconstructed_patch_norm * patch_std + patch_mean
                
                # 写回
                reconstructed_denorm[:, i_start:i_end, j_start:j_end] = reconstructed_patch_denorm
        
        reconstructed_patches_denorm = reconstructed_denorm.unsqueeze(0)  # (1, num_channels, 224, 224)
    else:
        # 如果 norm_pix_loss=False，直接使用重建值
        reconstructed_patches_denorm = reconstructed_patches
    
    # 转换为 numpy 用于可视化
    original_img = original_img_tensor.cpu().numpy()  # (num_channels, 224, 224)
    reconstructed_img = reconstructed_patches_denorm[0].cpu().numpy()  # (num_channels, 224, 224)
    mask_np = mask[0].cpu().numpy()  # (num_patches,)
    
    # ⚠️ 关键诊断：验证重建图像是否合理
    print(f"\n  🔍 重建图像验证：")
    print(f"    - 原始图像范围: [{original_img.min():.4f}, {original_img.max():.4f}]")
    print(f"    - 重建图像范围: [{reconstructed_img.min():.4f}, {reconstructed_img.max():.4f}]")
    print(f"    - 原始图像均值: {original_img.mean():.4f}")
    print(f"    - 重建图像均值: {reconstructed_img.mean():.4f}")
    
    # 检查重建图像是否全是0或全是1（说明unpatchify有问题）
    if np.allclose(reconstructed_img, 0) or np.allclose(reconstructed_img, 1):
        print(f"    ⚠ 警告：重建图像值异常（全0或全1），说明unpatchify逻辑有问题")
    elif reconstructed_img.min() < -0.5 or reconstructed_img.max() > 1.5:
        print(f"    ⚠ 警告：重建图像值超出合理范围，可能需要clip或检查数据预处理")
    
    # 检查重建图像和原始图像的相关性（如果重建正确，应该有较高的相关性）
    # 只检查被掩码的区域
    mask_2d = mask_np.reshape(h, w)
    mask_expanded = np.repeat(np.repeat(mask_2d, patch_size, axis=0), patch_size, axis=1)
    masked_original = original_img * mask_expanded[np.newaxis, :, :]
    masked_reconstructed = reconstructed_img * mask_expanded[np.newaxis, :, :]
    
    # ⚠️ 关键诊断：计算被掩码区域的相关性
    # 如果重建正确，相关性应该 >0.5
    # 如果相关性很低或为负，说明unpatchify逻辑有问题
    correlations = []
    if num_channels == 3:
        channel_names = ['DoLP', 'sin(2*AoLP)', 'cos(2*AoLP)']
    else:
        channel_names = ['Intensity', 'DoLP', 'sin(2*AoLP)', 'cos(2*AoLP)']
    
    for c_idx in range(num_channels):
        orig_flat = masked_original[c_idx].flatten()
        recon_flat = masked_reconstructed[c_idx].flatten()
        # 只计算非零区域（被掩码的区域）
        non_zero_mask = (orig_flat != 0) | (recon_flat != 0)
        if non_zero_mask.sum() > 0:
            orig_masked = orig_flat[non_zero_mask]
            recon_masked = recon_flat[non_zero_mask]
            if orig_masked.std() > 1e-6 and recon_masked.std() > 1e-6:
                correlation = np.corrcoef(orig_masked, recon_masked)[0, 1]
                correlations.append(correlation)
                print(f"    - {channel_names[c_idx]} 被掩码区域相关性: {correlation:.4f}")
                if correlation < 0.3:
                    print(f"      ⚠ 相关性很低，说明重建质量差或unpatchify有问题")
    
    # ⚠️ 关键诊断：如果所有通道的相关性都很低或为负，尝试不同的unpatchify方式
    if correlations:
        avg_correlation = np.mean(correlations)
        min_correlation = np.min(correlations)
        print(f"\n    🔍 相关性统计：平均={avg_correlation:.4f}, 最小={min_correlation:.4f}")
        
        # 如果相关性极低，尝试使用不同的unpatchify方式
        if avg_correlation < 0.1 or min_correlation < -0.1:
            print(f"    ⚠ 严重警告：平均相关性极低或为负（{avg_correlation:.4f}），尝试不同的unpatchify方式")
            
            # ⚠️ 关键修复：重新执行 unpatchify（使用备选方案）
            # 方式1：使用 permute 方式（可能更符合某些实现）
            # 从 (B, h, w, p, p, c) 到 (B, c, h*p, w*p)
            # 注意：需要重新获取 x，因为之前的 x 可能已经被修改
            x_retry1 = pred_patches.reshape(B, h_patches, w_patches, -1)
            x_retry1 = x_retry1.reshape(B, h_patches, w_patches, p, p, c)
            x_retry1 = x_retry1.permute(0, 5, 1, 3, 2, 4)  # (B, c, h, p, w, p)
            reconstructed_patches_retry1 = x_retry1.reshape(B, num_channels, h_patches * p, w_patches * p)
            
            # 处理norm_pix_loss（如果启用）
            if config.norm_pix_loss:
                reconstructed_denorm_retry1 = torch.zeros_like(reconstructed_patches_retry1[0])
                for i in range(h):
                    for j in range(w):
                        i_start = i * patch_size
                        i_end = (i + 1) * patch_size
                        j_start = j * patch_size
                        j_end = (j + 1) * patch_size
                        original_patch = original_img_tensor[:, i_start:i_end, j_start:j_end]
                        patch_mean = original_patch.mean(dim=(1, 2), keepdim=True)
                        patch_std = original_patch.std(dim=(1, 2), keepdim=True) + 1e-6
                        reconstructed_patch_norm = reconstructed_patches_retry1[0, :, i_start:i_end, j_start:j_end]
                        reconstructed_patch_denorm = reconstructed_patch_norm * patch_std + patch_mean
                        reconstructed_denorm_retry1[:, i_start:i_end, j_start:j_end] = reconstructed_patch_denorm
                reconstructed_patches_denorm_retry1 = reconstructed_denorm_retry1.unsqueeze(0)
                reconstructed_img_retry1 = np.clip(reconstructed_patches_denorm_retry1[0].cpu().numpy(), 0, 1)
            else:
                reconstructed_img_retry1 = np.clip(reconstructed_patches_retry1[0].cpu().numpy(), 0, 1)
            
            # 计算重试版本的相关性
            masked_reconstructed_retry1 = reconstructed_img_retry1 * mask_expanded[np.newaxis, :, :]
            
            correlations_retry1 = []
            for c_idx in range(num_channels):
                orig_flat = masked_original[c_idx].flatten()
                recon_flat_retry1 = masked_reconstructed_retry1[c_idx].flatten()
                non_zero_mask = (orig_flat != 0) | (recon_flat_retry1 != 0)
                if non_zero_mask.sum() > 0:
                    orig_masked = orig_flat[non_zero_mask]
                    recon_masked_retry1 = recon_flat_retry1[non_zero_mask]
                    if orig_masked.std() > 1e-6 and recon_masked_retry1.std() > 1e-6:
                        correlation_retry1 = np.corrcoef(orig_masked, recon_masked_retry1)[0, 1]
                        correlations_retry1.append(correlation_retry1)
            
            if correlations_retry1:
                avg_correlation_retry1 = np.mean(correlations_retry1)
                print(f"    - 方式1（permute(0,5,1,2,3,4)）的平均相关性: {avg_correlation_retry1:.4f}")
                
                # 选择相关性更高的版本
                if avg_correlation_retry1 > avg_correlation:
                    print(f"    ✓ 方式1相关性更高（{avg_correlation_retry1:.4f} vs {avg_correlation:.4f}），切换到该版本")
                    reconstructed_patches = reconstructed_patches_retry1
                    reconstructed_patches_denorm = reconstructed_patches_denorm_retry1 if config.norm_pix_loss else reconstructed_patches_retry1
                    reconstructed_img = reconstructed_img_retry1
                    masked_reconstructed = masked_reconstructed_retry1
                else:
                    print(f"    - 原方式相关性更高（{avg_correlation:.4f} vs {avg_correlation_retry1:.4f}），保持原版本")
    
    # 创建掩码图像（将掩码的 patch 设为黑色）
    masked_img = original_img.copy()
    mask_2d = mask_np.reshape(h, w)
    for i in range(h):
        for j in range(w):
            if mask_2d[i, j]:
                # 掩码这个 patch
                i_start = i * patch_size
                i_end = (i + 1) * patch_size
                j_start = j * patch_size
                j_end = (j + 1) * patch_size
                masked_img[:, i_start:i_end, j_start:j_end] = 0
    
    # ⚠️ 关键修复：归一化到 [0, 1]（如果还没有）
    # 重建图像可能超出 [0, 1] 范围，需要clip
    original_img = np.clip(original_img, 0, 1)
    reconstructed_img_before_clip = reconstructed_img.copy()
    reconstructed_img = np.clip(reconstructed_img, 0, 1)
    masked_img = np.clip(masked_img, 0, 1)
    
    # ⚠️ 诊断：检查clip前后的差异
    if reconstructed_img_before_clip.min() < 0 or reconstructed_img_before_clip.max() > 1:
        print(f"    ⚠ 重建图像值超出[0,1]范围，已clip")
        print(f"      clip前范围: [{reconstructed_img_before_clip.min():.4f}, {reconstructed_img_before_clip.max():.4f}]")
        print(f"      clip后范围: [{reconstructed_img.min():.4f}, {reconstructed_img.max():.4f}]")
        print(f"      可能原因：")
        print(f"        1. unpatchify逻辑有误，导致值域错误")
        print(f"        2. 模型预测值本身超出[0,1]范围（训练数据可能未归一化）")
        print(f"        3. 数据预处理不一致（训练和验证的数据范围不同）")

    # 计算量化指标（传递 mask 以计算 Masked 区域指标）
    metrics = compute_metrics(original_img, reconstructed_img, mask=mask_np, patch_size=patch_size, num_channels=num_channels)
    
    # ⚠️ 关键诊断：Loss vs Masked MSE 对比
    # 注意：模型Loss是patch级别的MSE（只计算被掩码的patches），而Masked MSE是像素级别的
    # 两者可能有差异，但应该在同一数量级
    if 'mse_masked_mean' in metrics:
        print(f"\n  🔍 诊断：Loss vs Masked MSE 对比")
        print(f"    - 模型 Loss (patch级别，只计算被掩码patches): {loss:.6f}")
        print(f"    - Masked MSE (像素级别，只计算被掩码区域): {metrics['mse_masked_mean']:.6f}")
        print(f"    - 差异: {abs(loss - metrics['mse_masked_mean']):.6f}")
        if loss > 0:
            print(f"    - 差异倍数: {metrics['mse_masked_mean'] / loss:.2f}x")
        
        # ⚠️ 注意：已使用与 verify_debug_stage1.py 相同的 logits 排列方式
        # （直接使用 ids_restore 重新排列到 Raster Order）
        
        # ⚠️ 关键说明：Loss和Masked MSE的差异分析
        loss_mse_diff = abs(loss - metrics['mse_masked_mean'])
        loss_mse_ratio = metrics['mse_masked_mean'] / loss if loss > 0 else float('inf')
        
        if loss_mse_diff < 0.01:
            print(f"\n    ✓ Loss 和 Masked MSE 匹配良好（差异 < 0.01），验证代码正确")
        elif loss_mse_diff < 0.1:
            print(f"\n    ⚠ 注意：Loss 和 Masked MSE 有差异（但可接受），可能原因：")
            print(f"       - patch级别 vs 像素级别的计算差异")
            print(f"       - 重建质量中等（部分patches重建较好，部分较差）")
        else:
            print(f"\n    ⚠ 警告：Loss 和 Masked MSE 差异较大（{loss_mse_ratio:.2f}x），可能原因：")
            print(f"       1. HuggingFace的loss计算使用了特殊的归一化或缩放（即使norm_pix_loss=False）")
            print(f"       2. 重建逻辑有问题（logits排列或unpatchify错误）")
            print(f"       3. 模型训练不充分（Loss低但重建质量差）")
            print(f"       4. 数据预处理不一致（训练和验证的数据范围不同）")
            print(f"    💡 建议：")
            print(f"       - 虽然Loss差异大，但Masked PSNR仍然是有意义的指标（与训练目标相关）")
            print(f"       - 检查可视化图像，确认重建是否合理")
            print(f"       - 如果重建图像明显错误，说明unpatchify逻辑有问题")
            print(f"       - 如果重建图像看起来合理但PSNR低，可能是模型训练不充分")
            print(f"       - 重点关注Masked PSNR（12-13 dB）和重建图像的可视化效果")

    print("\n" + "=" * 80)
    print("重建质量量化指标（详细）")
    print("=" * 80)
    
    # 全图指标
    print("\n📊 MSE (Mean Squared Error) - 全图:")
    if num_channels == 3:
        print(f"  - DoLP:        {metrics['mse_dolp']:.6f}")
        print(f"  - sin(2*AoLP): {metrics['mse_sin_2aolp']:.6f}")
        print(f"  - cos(2*AoLP): {metrics['mse_cos_2aolp']:.6f}")
    else:
        print(f"  - Intensity:   {metrics['mse_intensity']:.6f}")
        print(f"  - DoLP:        {metrics['mse_dolp']:.6f}")
        print(f"  - sin(2*AoLP): {metrics['mse_sin_2aolp']:.6f}")
        print(f"  - cos(2*AoLP): {metrics['mse_cos_2aolp']:.6f}")
    print(f"  - 平均 (Mean):  {metrics['mse_mean']:.6f}")
    
    print("\n📈 PSNR (Peak Signal-to-Noise Ratio, 单位: dB, 值越大越好) - 全图:")
    if num_channels == 3:
        print(f"  - DoLP:        {metrics['psnr_dolp']:.2f} dB")
        print(f"  - sin(2*AoLP): {metrics['psnr_sin_2aolp']:.2f} dB")
        print(f"  - cos(2*AoLP): {metrics['psnr_cos_2aolp']:.2f} dB")
    else:
        print(f"  - Intensity:   {metrics['psnr_intensity']:.2f} dB")
        print(f"  - DoLP:        {metrics['psnr_dolp']:.2f} dB")
        print(f"  - sin(2*AoLP): {metrics['psnr_sin_2aolp']:.2f} dB")
        print(f"  - cos(2*AoLP): {metrics['psnr_cos_2aolp']:.2f} dB")
    print(f"  - 平均 (Mean):  {metrics['psnr_mean']:.2f} dB")
    
    # Masked 区域指标（与 Training Loss 对齐）
    if 'mse_masked_mean' in metrics:
        print("\n" + "=" * 80)
        print("⚠️  关键对比：Masked 区域指标（与 Training Loss 对齐）")
        print("=" * 80)
        print("\n📊 MSE (Mean Squared Error) - 仅 Masked 区域:")
        if num_channels == 3:
            print(f"  - DoLP:        {metrics['mse_masked_dolp']:.6f}")
            print(f"  - sin(2*AoLP): {metrics['mse_masked_sin_2aolp']:.6f}")
            print(f"  - cos(2*AoLP): {metrics['mse_masked_cos_2aolp']:.6f}")
        else:
            print(f"  - Intensity:   {metrics['mse_masked_intensity']:.6f}")
            print(f"  - DoLP:        {metrics['mse_masked_dolp']:.6f}")
            print(f"  - sin(2*AoLP): {metrics['mse_masked_sin_2aolp']:.6f}")
            print(f"  - cos(2*AoLP): {metrics['mse_masked_cos_2aolp']:.6f}")
        print(f"  - 平均 (Mean):  {metrics['mse_masked_mean']:.6f}")
        print(f"  ⚠️  模型 Loss: {loss:.6f} (应该与上面的 MSE 接近)")
        
        print("\n📈 PSNR (Peak Signal-to-Noise Ratio, 单位: dB) - 仅 Masked 区域:")
        if num_channels == 3:
            print(f"  - DoLP:        {metrics['psnr_masked_dolp']:.2f} dB")
            print(f"  - sin(2*AoLP): {metrics['psnr_masked_sin_2aolp']:.2f} dB")
            print(f"  - cos(2*AoLP): {metrics['psnr_masked_cos_2aolp']:.2f} dB")
        else:
            print(f"  - Intensity:   {metrics['psnr_masked_intensity']:.2f} dB")
            print(f"  - DoLP:        {metrics['psnr_masked_dolp']:.2f} dB")
            print(f"  - sin(2*AoLP): {metrics['psnr_masked_sin_2aolp']:.2f} dB")
            print(f"  - cos(2*AoLP): {metrics['psnr_masked_cos_2aolp']:.2f} dB")
        print(f"  - 平均 (Mean):  {metrics['psnr_masked_mean']:.2f} dB")
        
        # 对比分析
        print("\n" + "=" * 80)
        print("🔍 对比分析")
        print("=" * 80)
        print(f"  - 全图 MSE: {metrics['mse_mean']:.6f}")
        print(f"  - Masked 区域 MSE: {metrics['mse_masked_mean']:.6f}")
        print(f"  - 模型 Loss: {loss:.6f}")
        if metrics['mse_masked_mean'] > 0:
            print(f"  - 差异: 全图 MSE 是 Masked MSE 的 {metrics['mse_mean'] / metrics['mse_masked_mean']:.2f} 倍")
        print(f"\n  💡 解释：")
        print(f"     - 如果 Masked MSE ≈ Loss，说明验证代码正确")
        print(f"     - 如果全图 MSE >> Masked MSE，说明 Visible 区域存在数值偏移（如'偏暗'）")
        print(f"     - 全图 PSNR 被 Visible 区域偏移影响，不可信")
        print(f"     - Masked PSNR 才是真实的重建质量指标（与 Loss 对齐）")
    
    # ========== 质量评估（新增，优先使用 Masked PSNR）==========
    quality_level, recommendation = evaluate_quality(metrics)
    print("\n" + "=" * 80)
    print("训练质量评估（基于 Masked PSNR，与 Loss 对齐）")
    print("=" * 80)
    if 'psnr_masked_mean' in metrics:
        print(f"  ⚠️  注意：使用 Masked 区域的 PSNR 进行评估（与 Loss 对齐）")
        print(f"  - Masked PSNR: {metrics['psnr_masked_mean']:.2f} dB (可信)")
        print(f"  - 全图 PSNR: {metrics['psnr_mean']:.2f} dB (仅供参考，包含 Visible 区域的数值偏移)")
    else:
        print(f"  - 全图 PSNR: {metrics['psnr_mean']:.2f} dB")
    print(f"  质量等级: {quality_level}")
    print(f"  建议: {recommendation}")
    print("=" * 80)
    
    # 创建可视化图像（根据通道数调整布局）
    if num_channels == 3:
        fig, axes = plt.subplots(3, 2, figsize=(12, 18))
        fig.suptitle('Stage 1 MAE Reconstruction Comparison (3 Channels)', fontsize=16, fontweight='bold')
        channel_names = [
            'DoLP (Degree of Linear Polarization)',
            'sin(2*AoLP)',
            'cos(2*AoLP)'
        ]
    else:
        fig, axes = plt.subplots(4, 2, figsize=(12, 24))
        fig.suptitle('Stage 1 MAE Reconstruction Comparison (4 Channels)', fontsize=16, fontweight='bold')
        channel_names = [
            'Intensity',
            'DoLP (Degree of Linear Polarization)',
            'sin(2*AoLP)',
            'cos(2*AoLP)'
        ]
    
    # 确定通道索引（根据通道数）
    if num_channels == 3:
        intensity_idx = None
        dolp_idx = 0
        sin_idx = 1
        cos_idx = 2
    else:
        intensity_idx = 0
        dolp_idx = 1
        sin_idx = 2
        cos_idx = 3
    
    # 计算 AoLP 角度（用于 HSV 可视化）
    # 从 sin(2*AoLP) 和 cos(2*AoLP) 计算 AoLP
    # 注意：sin(2*AoLP) 和 cos(2*AoLP) 的周期是 π，所以 AoLP 的范围是 [0, π]
    def compute_aolp_hsv(sin_ch, cos_ch):
        """从 sin(2*AoLP) 和 cos(2*AoLP) 计算 AoLP 的 HSV 表示"""
        # 计算 AoLP 角度
        # 因为 sin(2*AoLP) 和 cos(2*AoLP) 的周期是 π，所以：
        # AoLP = 0.5 * arctan2(sin(2*AoLP), cos(2*AoLP))
        # arctan2 返回 [-π, π]，所以 AoLP = [-π/2, π/2]
        # 但我们需要映射到 [0, π]
        aolp_raw = 0.5 * np.arctan2(sin_ch, cos_ch)  # [-π/2, π/2]
        
        # 将负角度转换到 [0, π] 范围
        # 如果 aolp_raw < 0，则 AoLP = aolp_raw + π
        aolp_angle = np.where(aolp_raw < 0, aolp_raw + np.pi, aolp_raw)  # [0, π]
        
        # 将角度归一化到 [0, 1] 用于 HSV 的 H 通道
        h = aolp_angle / np.pi  # H: 色调（角度），范围 [0, 1]
        s = np.ones_like(h)  # S: 饱和度（设为1）
        v = np.ones_like(h)  # V: 明度（设为1）
        
        # 转换为 RGB 用于显示
        from matplotlib.colors import hsv_to_rgb
        hsv_img = np.stack([h, s, v], axis=-1)
        rgb_img = hsv_to_rgb(hsv_img)
        return rgb_img
    
    for ch_idx in range(num_channels):
        # 原始图像
        if ch_idx == dolp_idx:
            # DoLP: 使用 jet 色图
            im_orig = axes[ch_idx, 0].imshow(original_img[ch_idx], cmap='jet', vmin=0, vmax=1)
            axes[ch_idx, 0].set_title(f'Original - {channel_names[ch_idx]}', fontsize=12)
            axes[ch_idx, 0].axis('off')
            plt.colorbar(im_orig, ax=axes[ch_idx, 0], fraction=0.046)
        elif ch_idx == sin_idx or ch_idx == cos_idx:
            # AoLP (sin/cos): 转换为 AoLP 角度并使用 HSV 可视化
            # 需要同时使用 sin 和 cos 通道来计算 AoLP
            sin_ch_orig = original_img[sin_idx]
            cos_ch_orig = original_img[cos_idx]
            rgb_orig = compute_aolp_hsv(sin_ch_orig, cos_ch_orig)
            im_orig = axes[ch_idx, 0].imshow(rgb_orig)
            if ch_idx == sin_idx:
                axes[ch_idx, 0].set_title(f'Original - AoLP (from sin/cos, HSV)', fontsize=12)
            else:
                axes[ch_idx, 0].set_title(f'Original - AoLP (from sin/cos, HSV)', fontsize=12)
            axes[ch_idx, 0].axis('off')
        else:
            # Intensity 或其他通道：使用灰度图
            im_orig = axes[ch_idx, 0].imshow(original_img[ch_idx], cmap='gray', vmin=0, vmax=1)
            axes[ch_idx, 0].set_title(f'Original - {channel_names[ch_idx]}', fontsize=12)
            axes[ch_idx, 0].axis('off')
            plt.colorbar(im_orig, ax=axes[ch_idx, 0], fraction=0.046)
        
        # 重建图像
        if ch_idx == dolp_idx:
            # DoLP: 使用 jet 色图
            im_recon = axes[ch_idx, 1].imshow(reconstructed_img[ch_idx], cmap='jet', vmin=0, vmax=1)
            axes[ch_idx, 1].set_title(f'Reconstructed - {channel_names[ch_idx]}', fontsize=12)
            axes[ch_idx, 1].axis('off')
            plt.colorbar(im_recon, ax=axes[ch_idx, 1], fraction=0.046)
        elif ch_idx == sin_idx or ch_idx == cos_idx:
            # AoLP (sin/cos): 转换为 AoLP 角度并使用 HSV 可视化
            # 需要同时使用 sin 和 cos 通道来计算 AoLP
            sin_ch_recon = reconstructed_img[sin_idx]
            cos_ch_recon = reconstructed_img[cos_idx]
            rgb_recon = compute_aolp_hsv(sin_ch_recon, cos_ch_recon)
            im_recon = axes[ch_idx, 1].imshow(rgb_recon)
            axes[ch_idx, 1].set_title(f'Reconstructed - AoLP (from sin/cos, HSV)', fontsize=12)
            axes[ch_idx, 1].axis('off')
        else:
            # Intensity 或其他通道：使用灰度图
            im_recon = axes[ch_idx, 1].imshow(reconstructed_img[ch_idx], cmap='gray', vmin=0, vmax=1)
            axes[ch_idx, 1].set_title(f'Reconstructed - {channel_names[ch_idx]}', fontsize=12)
            axes[ch_idx, 1].axis('off')
            plt.colorbar(im_recon, ax=axes[ch_idx, 1], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ 可视化结果已保存到: {output_path}")
    plt.close()

    return metrics


def list_base_names(scene_dir: Path) -> List[str]:
    """
    列出某个场景下可用的 base_name 列表（根据 *_000.png 自动推断）。

    Args:
        scene_dir: 某个 scene_id 目录，例如 polar_root/01

    Returns:
        base_names: 例如 ["0006", "0007", ...]
    """
    if not scene_dir.exists():
        return []

    candidates = sorted(scene_dir.glob("*_000.png"))
    base_names = []
    for p in candidates:
        stem = p.stem  # 例如 "0006_000"
        if stem.endswith("_000"):
            base = stem[:-4]  # 去掉 "_000"
            base_names.append(base)
    return base_names


def list_base_names_pt(pt_root: Path, scene_id: str) -> List[str]:
    """
    列出某个场景下可用的 base_name 列表（根据 *.pt 文件自动推断）。
    支持两种目录结构：
    1. pt_root/train/scene_id/xxxxx.pt（优先）
    2. pt_root/scene_id/xxxxx.pt（回退）

    Args:
        pt_root: .pt 文件根目录，例如 /openbayes/home/data/polar_pt
        scene_id: 场景ID，例如 "17"

    Returns:
        base_names: 例如 ["0006", "0007", ...]
    """
    # 优先尝试 train 目录
    train_scene_dir = pt_root / "train" / scene_id
    if train_scene_dir.exists():
        candidates = sorted(train_scene_dir.glob("*.pt"))
        if candidates:
            base_names = []
            for p in candidates:
                base_name = p.stem  # 例如 "0006"（去除 .pt 后缀）
                base_names.append(base_name)
            return base_names
    
    # 回退：尝试 val 目录
    val_scene_dir = pt_root / "val" / scene_id
    if val_scene_dir.exists():
        candidates = sorted(val_scene_dir.glob("*.pt"))
        if candidates:
            base_names = []
            for p in candidates:
                base_name = p.stem
                base_names.append(base_name)
            return base_names
    
    # 最后回退：直接使用 pt_root/scene_id
    scene_dir = pt_root / scene_id
    if scene_dir.exists():
        candidates = sorted(scene_dir.glob("*.pt"))
        if candidates:
            base_names = []
            for p in candidates:
                base_name = p.stem
                base_names.append(base_name)
            return base_names
    
    return []


def run_one_example(
    model: ViTMAEForPreTraining,
    device: torch.device,
    polar_root: Path,
    scene_id: str,
    base_name: str,
    output_path: str,
    use_pt_data: bool = False,
) -> Dict[str, float]:
    """
    运行单个样本的重建与可视化，并返回量化指标。
    
    Args:
        use_pt_data: 是否使用 .pt 文件（如果 True，polar_root 应该是 pt_root）
    """
    print("\n" + "=" * 80)
    print(f"准备测试数据...  scene_id={scene_id}, base_name={base_name}")
    print("=" * 80)

    if use_pt_data:
        # 使用 .pt 文件，支持 train/val 子目录结构
        # 优先尝试 train 目录
        pt_path = polar_root / "train" / scene_id / f"{base_name}.pt"
        if not pt_path.exists():
            # 回退：尝试 val 目录
            pt_path = polar_root / "val" / scene_id / f"{base_name}.pt"
            if not pt_path.exists():
                # 最后回退：直接使用 pt_root/scene_id
                pt_path = polar_root / scene_id / f"{base_name}.pt"
        
        if not pt_path.exists():
            # 尝试所有可能的路径
            possible_paths = [
                polar_root / "train" / scene_id / f"{base_name}.pt",
                polar_root / "val" / scene_id / f"{base_name}.pt",
                polar_root / scene_id / f"{base_name}.pt",
            ]
            raise FileNotFoundError(
                f".pt 文件不存在，无法进行验证。已尝试以下路径：\n" +
                "\n".join([f"  - {p}" for p in possible_paths])
            )
        
        # 加载 .pt 文件（已经是 (4, H, W) 张量，值范围 [0, 1]）
        # 注意：weights_only=False 因为这是数据文件，不是模型权重
        pixel_values_tensor = torch.load(pt_path, map_location='cpu', weights_only=False)
        
        # 确保数据类型和形状正确
        if pixel_values_tensor.dtype != torch.float32:
            pixel_values_tensor = pixel_values_tensor.float()
        
        # 确保形状是 (4, H, W) 或 (3, H, W)
        if len(pixel_values_tensor.shape) != 3:
            raise ValueError(
                f"加载的 .pt 文件形状不正确: {pixel_values_tensor.shape}, "
                f"期望: (3或4, H, W)"
            )
        
        # ⚠️ 关键诊断：检查通道顺序
        print(f"  - 加载的 .pt 文件形状: {pixel_values_tensor.shape}")
        print(f"  - 通道顺序检查:")
        if pixel_values_tensor.shape[0] == 4:
            print(f"    - 原始4通道: [Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]")
            print(f"    - 通道统计:")
            for i, name in enumerate(['Intensity', 'DoLP', 'sin(2*AoLP)', 'cos(2*AoLP)']):
                ch_data = pixel_values_tensor[i]
                print(f"      [{i}] {name}: mean={ch_data.mean():.4f}, std={ch_data.std():.4f}, "
                      f"min={ch_data.min():.4f}, max={ch_data.max():.4f}")
            
            # 如果加载的是4通道，只取后3个通道（DoLP, sin, cos）
            pixel_values_tensor = pixel_values_tensor[1:4, :, :]  # 去掉Intensity通道
            print(f"    ⚠ 注意: 已移除Intensity通道，保留 [DoLP, sin(2*AoLP), cos(2*AoLP)]")
        elif pixel_values_tensor.shape[0] == 3:
            print(f"    - 已经是3通道: [?, ?, ?]")
            print(f"    - 通道统计:")
            for i in range(3):
                ch_data = pixel_values_tensor[i]
                print(f"      [{i}]: mean={ch_data.mean():.4f}, std={ch_data.std():.4f}, "
                      f"min={ch_data.min():.4f}, max={ch_data.max():.4f}")
            print(f"    ⚠ 注意: 假设通道顺序为 [DoLP, sin(2*AoLP), cos(2*AoLP)]")
            print(f"    ⚠ 警告: 请确认 .pt 文件的通道顺序与训练时一致！")
        
        if pixel_values_tensor.shape[0] != 3:
            raise ValueError(
                f"加载的 .pt 文件通道数不正确: {pixel_values_tensor.shape[0]}, "
                f"期望: 3 (DoLP, sin, cos)"
            )
        
        # 最终确认通道顺序
        print(f"  ✓ 最终输入通道顺序: [DoLP, sin(2*AoLP), cos(2*AoLP)]")
        print(f"  ✓ 与训练时一致（dataset_stage1.py 使用 pixel_values[1:4, :, :]）")
        
        # 添加 batch 维度: (3, H, W) -> (1, 3, H, W)
        pixel_values = pixel_values_tensor.unsqueeze(0)
        print(f"✓ 已加载 .pt 文件，形状: {pixel_values.shape}")
    else:
        # 使用 PNG 图像
        scene_dir = polar_root / scene_id
        # 构建4个角度图像路径
        polar_paths = {
            "I_0": scene_dir / f"{base_name}_000.png",
            "I_45": scene_dir / f"{base_name}_045.png",
            "I_90": scene_dir / f"{base_name}_090.png",
            "I_135": scene_dir / f"{base_name}_135.png",
        }

        # 检查文件是否存在
        missing = [name for name, path in polar_paths.items() if not path.exists()]
        if missing:
            print(f"⚠ 警告: 下列角度图像不存在（尝试使用无前导零的备选命名）: {missing}")
            polar_paths = {
                "I_0": scene_dir / f"{base_name}_0.png",
                "I_45": scene_dir / f"{base_name}_45.png",
                "I_90": scene_dir / f"{base_name}_90.png",
                "I_135": scene_dir / f"{base_name}_135.png",
            }

        # 最终再检查一遍
        for name, path in polar_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"{name} 图像不存在，无法进行验证: {path}")

        # 使用共享函数处理偏振图像（返回4通道：[Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]）
        physics_img = process_polar_images(polar_paths=polar_paths)  # (H, W, 4)
        print(f"✓ 已处理偏振图像，形状: {physics_img.shape}")

        # 预处理图像（只使用后3个通道）
        # 获取模型配置以确定通道数
        config = model.config
        num_channels = getattr(config, 'num_channels', 3)
        pixel_values = preprocess_image(physics_img, image_size=224, num_channels=num_channels)  # (1, 3, 224, 224)
        print(f"✓ 已预处理图像，形状: {pixel_values.shape}")

    # ⚠️ 关键诊断：检查数据预处理一致性
    print("\n" + "=" * 80)
    print("🔍 数据预处理一致性检查")
    print("=" * 80)
    print(f"  - 输入数据形状: {pixel_values.shape}")
    print(f"  - 输入数据范围: [{pixel_values.min():.4f}, {pixel_values.max():.4f}]")
    print(f"  - 输入数据均值: {pixel_values.mean():.4f}")
    print(f"  - 输入数据标准差: {pixel_values.std():.4f}")
    print(f"  - 通道统计:")
    channel_names = ['DoLP', 'sin(2*AoLP)', 'cos(2*AoLP)']
    for i, name in enumerate(channel_names):
        ch_data = pixel_values[0, i]
        print(f"    [{i}] {name}: mean={ch_data.mean():.4f}, std={ch_data.std():.4f}, "
              f"min={ch_data.min():.4f}, max={ch_data.max():.4f}")
    print(f"\n  ⚠️ 关键检查点:")
    print(f"    1. 通道顺序: [DoLP, sin(2*AoLP), cos(2*AoLP)] ✓")
    print(f"    2. 数据范围: [0, 1] (期望) vs [{pixel_values.min():.4f}, {pixel_values.max():.4f}] (实际)")
    print(f"    3. 数据增强: 验证脚本直接加载 .pt 文件，无数据增强 ✓")
    print(f"    4. 与训练时一致: dataset_stage1.py 在 is_train=False 时关闭所有随机增强 ✓")
    print("=" * 80 + "\n")
    
    # 可视化 + 指标计算
    metrics = visualize_mae_reconstruction(model, pixel_values, output_path=output_path)
    return metrics


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 1 MAE 验证脚本")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/stage1_encoder",
        help="Stage 1 检查点路径"
    )
    parser.add_argument(
        "--polar_root",
        type=str,
        default="/openbayes/input/input0/polar",
        help="偏振图像根目录"
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default="01",
        help="测试场景ID"
    )
    parser.add_argument(
        "--base_name",
        type=str,
        default="0006",
        help="测试图像基础名称（不含后缀，当 num_examples>1 时会以此为起点或被忽略）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stage1_vis.png",
        help="输出图像路径（单个文件）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（多样本时使用，如果提供，会在此目录中保存多个文件）"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token（可选）"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1,
        help="验证样本数量（默认1）。当 >1 时，会自动在指定 scene_id 下遍历多个 base_name，并为每个样本生成一张可视化图像。",
    )
    # 添加 .pt 文件支持
    parser.add_argument(
        "--use_pt_data",
        action="store_true",
        default=False,
        help="使用预处理的 .pt 文件（如果提供，polar_root 应该是 pt_root）"
    )
    parser.add_argument(
        "--pt_root",
        type=str,
        default=None,
        help="[暂不支持] .pt 文件根目录"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="[可选] 图像尺寸（从模型配置自动读取，此参数仅供参考）"
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help="[可选] Patch 尺寸（从模型配置自动读取，此参数仅供参考）"
    )
    parser.add_argument(
        "--num_channels",
        type=int,
        default=3,
        help="[可选] 输入通道数（从模型配置自动读取，此参数仅供参考，默认3：DoLP, sin, cos）"
    )
    parser.add_argument(
        "--norm_pix_loss",
        type=lambda x: (str(x).lower() == 'true'),
        default=False,
        nargs='?',
        const=False,
        help="[可选] 归一化像素损失（从模型配置自动读取，此参数仅供参考）"
    )
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 加载模型
    model = load_mae_model(args.checkpoint, hf_token=args.hf_token)
    model = model.to(device)

    # 2. 准备测试数据 & 多样本验证
    polar_root = Path(args.polar_root)

    # 列出可用的 base_name 列表
    if args.use_pt_data:
        # 使用 .pt 文件，支持 train/val 子目录结构
        all_base_names = list_base_names_pt(polar_root, args.scene_id)
        if not all_base_names:
            # 尝试列出所有可能的目录
            possible_dirs = [
                polar_root / "train" / args.scene_id,
                polar_root / "val" / args.scene_id,
                polar_root / args.scene_id,
            ]
            print(f"⚠ 警告: 未找到 *.pt 文件，已尝试以下目录：")
            for d in possible_dirs:
                print(f"  - {d} ({'存在' if d.exists() else '不存在'})")
            print(f"  尝试使用单个 base_name={args.base_name}")
            all_base_names = [args.base_name]
    else:
        # 使用 PNG 图像
        scene_dir = polar_root / args.scene_id
        
        # ⚠️ 关键检查：确保 scene_id 目录存在
        if not scene_dir.exists():
            raise FileNotFoundError(
                f"错误：指定的 scene_id={args.scene_id} 对应的目录不存在：{scene_dir}\n"
                f"请检查：\n"
                f"  1. polar_root 是否正确：{polar_root}\n"
                f"  2. scene_id 是否正确：{args.scene_id}\n"
                f"  3. 目录结构是否为：{polar_root}/<scene_id>/*_000.png"
            )
        
        all_base_names = list_base_names(scene_dir)
        if not all_base_names:
            print(f"⚠ 警告: 在目录 {scene_dir} 中未找到 *_000.png 文件")
            print(f"  已尝试查找模式: {scene_dir}/*_000.png")
            print(f"  尝试使用指定的 base_name={args.base_name}")
            all_base_names = [args.base_name]
        else:
            print(f"✓ 在 {scene_dir} 中找到 {len(all_base_names)} 个可用样本")

    # 确定要使用的样本数量
    num_examples = max(1, args.num_examples)
    if num_examples > len(all_base_names):
        print(f"⚠ 请求的样本数 num_examples={num_examples} 大于可用样本数 {len(all_base_names)}，将使用所有可用样本。")
        num_examples = len(all_base_names)

    selected_base_names = all_base_names[:num_examples]

    print("\n" + "=" * 80)
    print(f"准备测试数据... 共选取 {num_examples} 个样本")
    print(f"✓ 确认 scene_id={args.scene_id} (polar_root={polar_root})")
    print(f"✓ base_names={selected_base_names}")
    print("=" * 80)

    # 3. 逐样本运行，并统计平均指标
    all_metrics: List[Dict[str, float]] = []

    # 处理输出路径（支持 --output_dir）
    if args.output_dir is not None:
        # 如果提供了 output_dir，使用目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_base = output_dir / "stage1_vis.png"  # 默认文件名
    else:
        # 如果只提供了 output，使用文件路径
        output_base = Path(args.output)
        if num_examples > 1:
            # 多样本时，自动使用输出文件的父目录
            output_dir = output_base.parent
            output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, base_name in enumerate(selected_base_names):
        if num_examples == 1:
            output_path = str(output_base)
        else:
            # 多个样本时，在文件名中加入索引和 base_name，避免覆盖
            if args.output_dir is not None:
                # 使用指定的输出目录
                output_name = f"stage1_vis_scene{args.scene_id}_idx{idx}_{base_name}.png"
                output_path = str(output_dir / output_name)
            else:
                # 使用输出文件的父目录
                stem = output_base.stem
                suffix = output_base.suffix or ".png"
                output_name = f"{stem}_scene{args.scene_id}_idx{idx}_{base_name}{suffix}"
                output_path = str(output_base.parent / output_name)

        metrics = run_one_example(
            model=model,
            device=device,
            polar_root=polar_root,
            scene_id=args.scene_id,
            base_name=base_name,
            output_path=output_path,
            use_pt_data=args.use_pt_data,
        )
        all_metrics.append(metrics)

    # 4. 汇总多样本平均指标
    if len(all_metrics) > 1:
        print("\n" + "=" * 80)
        print("多样本平均指标统计：")
        print("=" * 80)
        metric_keys = list(all_metrics[0].keys())
        avg_metrics = {}
        for k in metric_keys:
            avg_metrics[k] = float(np.mean([m[k] for m in all_metrics]))

        print("\n📊 MSE (Mean Squared Error) - 全图:")
        if 'mse_intensity' in avg_metrics:
            print(f"  [MSE]  Intensity:   {avg_metrics['mse_intensity']:.6f}")
        print(f"  [MSE]  DoLP:        {avg_metrics['mse_dolp']:.6f}")
        print(f"  [MSE]  sin(2*AoLP): {avg_metrics['mse_sin_2aolp']:.6f}")
        print(f"  [MSE]  cos(2*AoLP): {avg_metrics['mse_cos_2aolp']:.6f}")
        print(f"  [MSE]  Mean:        {avg_metrics['mse_mean']:.6f}")
        print(f"\n📈 PSNR (Peak Signal-to-Noise Ratio, 单位: dB) - 全图:")
        if 'psnr_intensity' in avg_metrics:
            print(f"  [PSNR] Intensity:   {avg_metrics['psnr_intensity']:.2f} dB")
        print(f"  [PSNR] DoLP:        {avg_metrics['psnr_dolp']:.2f} dB")
        print(f"  [PSNR] sin(2*AoLP): {avg_metrics['psnr_sin_2aolp']:.2f} dB")
        print(f"  [PSNR] cos(2*AoLP): {avg_metrics['psnr_cos_2aolp']:.2f} dB")
        print(f"  [PSNR] Mean:        {avg_metrics['psnr_mean']:.2f} dB")
        
        # Masked 区域指标（如果存在）
        if 'mse_masked_mean' in avg_metrics:
            print("\n" + "=" * 80)
            print("⚠️  关键对比：Masked 区域指标（与 Training Loss 对齐）")
            print("=" * 80)
            print("\n📊 MSE (Mean Squared Error) - 仅 Masked 区域:")
            if 'mse_masked_intensity' in avg_metrics:
                print(f"  [MSE]  Intensity:   {avg_metrics['mse_masked_intensity']:.6f}")
            print(f"  [MSE]  DoLP:        {avg_metrics['mse_masked_dolp']:.6f}")
            print(f"  [MSE]  sin(2*AoLP): {avg_metrics['mse_masked_sin_2aolp']:.6f}")
            print(f"  [MSE]  cos(2*AoLP): {avg_metrics['mse_masked_cos_2aolp']:.6f}")
            print(f"  [MSE]  Mean:        {avg_metrics['mse_masked_mean']:.6f}")
            print(f"\n📈 PSNR (Peak Signal-to-Noise Ratio, 单位: dB) - 仅 Masked 区域:")
            if 'psnr_masked_intensity' in avg_metrics:
                print(f"  [PSNR] Intensity:   {avg_metrics['psnr_masked_intensity']:.2f} dB")
            print(f"  [PSNR] DoLP:        {avg_metrics['psnr_masked_dolp']:.2f} dB")
            print(f"  [PSNR] sin(2*AoLP): {avg_metrics['psnr_masked_sin_2aolp']:.2f} dB")
            print(f"  [PSNR] cos(2*AoLP): {avg_metrics['psnr_masked_cos_2aolp']:.2f} dB")
            print(f"  [PSNR] Mean:        {avg_metrics['psnr_masked_mean']:.2f} dB")
            
            # 对比分析
            print("\n" + "=" * 80)
            print("🔍 对比分析")
            print("=" * 80)
            print(f"  - 全图 MSE: {avg_metrics['mse_mean']:.6f}")
            print(f"  - Masked 区域 MSE: {avg_metrics['mse_masked_mean']:.6f}")
            if avg_metrics['mse_masked_mean'] > 0:
                print(f"  - 差异: 全图 MSE 是 Masked MSE 的 {avg_metrics['mse_mean'] / avg_metrics['mse_masked_mean']:.2f} 倍")
            print(f"\n  💡 解释：")
            print(f"     - 全图 MSE 远大于 Masked MSE，说明 Visible 区域存在数值偏移")
            print(f"     - 全图 PSNR 被 Visible 区域偏移影响，不可信")
            print(f"     - Masked PSNR 才是真实的重建质量指标（与 Loss 对齐）")
        
        # ========== 平均质量评估（新增，优先使用 Masked PSNR）==========
        avg_quality_level, avg_recommendation = evaluate_quality(avg_metrics)
        print("\n" + "=" * 80)
        print("平均训练质量评估（基于 Masked PSNR，与 Loss 对齐）")
        print("=" * 80)
        if 'psnr_masked_mean' in avg_metrics:
            print(f"  ⚠️  注意：使用 Masked 区域的 PSNR 进行评估（与 Loss 对齐）")
            print(f"  - Masked PSNR: {avg_metrics['psnr_masked_mean']:.2f} dB (可信)")
            print(f"  - 全图 PSNR: {avg_metrics['psnr_mean']:.2f} dB (仅供参考，包含 Visible 区域的数值偏移)")
        else:
            print(f"  - 全图 PSNR: {avg_metrics['psnr_mean']:.2f} dB")
        print(f"  质量等级: {avg_quality_level}")
        print(f"  建议: {avg_recommendation}")
        print("=" * 80)

    print("\n" + "=" * 80)
    print("验证完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()


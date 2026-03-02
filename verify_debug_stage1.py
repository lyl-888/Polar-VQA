"""
验证调试模式训练结果：可视化重建效果

功能：
1. 加载调试模式训练的模型
2. 使用训练时的同一个样本进行重建
3. 可视化原始图像和重建图像的对比
4. 计算重建质量指标（MSE, PSNR）

使用方法：
    python verify_debug_stage1.py \
        --checkpoint /openbayes/input/input0/checkpoints/stage1_debug \
        --polar_root /openbayes/home/data/polar_pt \
        --use_pt_data \
        --output_dir ./debug_verification
"""

import os
import argparse
from pathlib import Path
from typing import Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from train_stage1 import create_mae_model
from dataset_stage1 import PolarMAEDataset, collate_fn_stage1

# Matplotlib 配置
plt.rcParams["axes.unicode_minus"] = False


def load_trained_model(checkpoint_dir: str, model_name: str, image_size: int, 
                      patch_size: int, num_channels: int, norm_pix_loss: bool,
                      hf_token: Optional[str] = None):
    """加载训练好的模型"""
    print("=" * 80)
    print("加载训练好的模型")
    print("=" * 80)
    
    # 创建模型
    model = create_mae_model(
        model_name=model_name,
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,
        norm_pix_loss=norm_pix_loss,
        hf_token=hf_token,
    )
    
    # 加载训练好的权重
    checkpoint_path = Path(checkpoint_dir)
    
    # 尝试加载 pytorch_model.bin 或 model.safetensors
    model_file = None
    if (checkpoint_path / "pytorch_model.bin").exists():
        model_file = checkpoint_path / "pytorch_model.bin"
    elif (checkpoint_path / "model.safetensors").exists():
        model_file = checkpoint_path / "model.safetensors"
    else:
        # 尝试加载 checkpoint-* 子目录
        checkpoint_dirs = sorted([d for d in checkpoint_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])
        if checkpoint_dirs:
            latest_checkpoint = checkpoint_dirs[-1]
            if (latest_checkpoint / "pytorch_model.bin").exists():
                model_file = latest_checkpoint / "pytorch_model.bin"
            elif (latest_checkpoint / "model.safetensors").exists():
                model_file = latest_checkpoint / "model.safetensors"
    
    if model_file:
        print(f"✓ 找到模型文件: {model_file}")
        try:
            # 优先使用 safetensors（如果可用）
            if model_file.suffix == '.safetensors':
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(str(model_file))
                    print("✓ 使用 safetensors 格式加载")
                except ImportError:
                    print("⚠ 警告: safetensors 库未安装，尝试使用 torch.load")
                    state_dict = torch.load(model_file, map_location="cpu", weights_only=False)
            else:
                # 使用 torch.load，添加 weights_only=False 以避免安全限制
                state_dict = torch.load(model_file, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict, strict=False)
            print("✓ 模型权重加载成功")
        except Exception as e:
            print(f"⚠ 警告: 加载模型权重时出错: {e}")
            print("  将使用当前模型状态（可能是随机初始化）")
    else:
        print("⚠ 警告: 未找到模型权重文件，将使用当前模型状态")
    
    model.eval()
    return model


def compute_metrics(original_img: np.ndarray, reconstructed_img: np.ndarray, 
                    mask: Optional[np.ndarray] = None, patch_size: int = 16) -> dict:
    """
    计算重建的量化指标（逐通道 + 整体）：
    - MSE: Mean Squared Error
    - PSNR: Peak Signal-to-Noise Ratio（峰值信噪比，单位 dB，值越大越好）
    
    ⚠️ 关键修复：支持只计算 Masked 区域的指标，与 Training Loss 对齐

    Args:
        original_img: 原始图像 (4, H, W)，值范围 [0, 1]
                      通道顺序：[Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]
        reconstructed_img: 重建图像 (4, H, W)，值范围 [0, 1]
        mask: 掩码 (num_patches,)，1 表示被掩码（需要预测），0 表示可见
              如果提供，会同时计算全图和 Masked 区域的指标
        patch_size: Patch 尺寸，用于将 mask 扩展到图像尺寸

    Returns:
        包含逐通道和整体指标的字典（如果提供了 mask，会包含 masked 区域的指标）
    """
    # 避免数值问题，先裁剪
    original = np.clip(original_img, 0, 1)
    recon = np.clip(reconstructed_img, 0, 1)

    # 计算全图的指标
    mse_per_channel_full = ((original - recon) ** 2).mean(axis=(1, 2))  # (4,)
    mse_overall_full = mse_per_channel_full.mean()

    # PSNR 计算：PSNR = 10 * log10(MAX^2 / MSE)，这里 MAX=1
    eps = 1e-8
    psnr_per_channel_full = 10.0 * np.log10(1.0 / (mse_per_channel_full + eps))
    psnr_overall_full = 10.0 * np.log10(1.0 / (mse_overall_full + eps))

    metrics = {
        # 全图指标
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
        for c in range(4):
            diff_sq = (original[c] - recon[c]) ** 2
            masked_mse = (diff_sq * mask_expanded).sum() / (mask_expanded.sum() + eps)
            masked_mse_per_channel.append(float(masked_mse))
        
        masked_mse_per_channel = np.array(masked_mse_per_channel)
        masked_mse_overall = masked_mse_per_channel.mean()
        
        # 计算 Masked 区域的 PSNR
        masked_psnr_per_channel = 10.0 * np.log10(1.0 / (masked_mse_per_channel + eps))
        masked_psnr_overall = 10.0 * np.log10(1.0 / (masked_mse_overall + eps))
        
        # 添加到 metrics
        metrics.update({
            # Masked 区域指标（与 Training Loss 对齐）
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


def visualize_reconstruction(model, sample_data, output_dir: Path, device: torch.device):
    """
    可视化重建结果并计算详细的 PSNR 指标
    
    Args:
        model: MAE 模型
        sample_data: 单个样本数据（包含 pixel_values）
        output_dir: 输出目录
        device: 设备
    """
    print("\n" + "=" * 80)
    print("可视化重建结果")
    print("=" * 80)
    
    # 准备输入数据
    pixel_values = sample_data["pixel_values"].unsqueeze(0).to(device)  # (1, 4, H, W)
    
    print(f"✓ 输入形状: {pixel_values.shape}")
    print(f"✓ 输入数据范围: [{pixel_values.min():.4f}, {pixel_values.max():.4f}]")
    
    # 获取模型配置
    config = model.config
    image_size = config.image_size
    patch_size = config.patch_size
    num_patches = (image_size // patch_size) ** 2
    h = w = image_size // patch_size
    
    # 前向传播（获取重建结果）
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        
        # 获取输出
        loss = outputs.loss.item()
        logits = outputs.logits  # 重建的 patch tokens (B, num_patches, patch_size^2 * num_channels)
        ids_restore = outputs.ids_restore  # 恢复的 patch 顺序 (B, num_patches)
        mask = outputs.mask  # 掩码 (B, num_patches)
    
    print(f"✓ 重建损失: {loss:.6f}")
    print(f"✓ 掩码比例: {mask.sum().item() / mask.numel():.2%}")
    print(f"✓ norm_pix_loss: {config.norm_pix_loss}")
    
    # ========== 关键修复：正确组合可见 patches 和预测 patches ==========
    # MAE 的 outputs.logits 只包含被掩码的 patches 的预测
    # 需要将可见的 patches（来自原图）和预测的 patches（来自模型）正确组合
    
    B = pixel_values.shape[0]
    num_total_patches = num_patches
    patch_dim = patch_size ** 2 * config.num_channels
    
    print(f"  DEBUG: logits shape: {logits.shape}")
    print(f"  DEBUG: num_total_patches: {num_total_patches}")
    print(f"  DEBUG: mask shape: {mask.shape}")
    print(f"  DEBUG: ids_restore shape: {ids_restore.shape}")
    
    # 检查 logits 的形状
    # HuggingFace 的 ViTMAEForPreTraining 的 logits 通常已经是全图的预测
    # 但为了安全，我们检查一下
    if logits.shape[1] == num_total_patches:
        # HuggingFace 的 logits 已经是全图的预测（已经组合了可见和预测的 patches）
        # ⚠️ 关键修复：logits是按ids_restore的顺序排列的，即logits[i]对应原始位置ids_restore[i]
        # 要恢复到Raster Order，我们需要：restored_logits[ids_restore[i]] = logits[i]
        print("  ✓ logits 已经是全图的预测，按 ids_restore 顺序排列，需要恢复到 Raster Order")
        
        # 根据 ids_restore 重新排列 logits 到 Raster Order
        # ids_restore[i] 表示 logits 中第 i 个元素应该放到原始位置的哪个位置
        # 所以：restored_logits[ids_restore[i]] = logits[i]
        ids_restore_np = ids_restore[0].cpu().numpy()  # (num_total_patches,)
        restored_logits = torch.zeros_like(logits)
        for i in range(len(ids_restore_np)):
            orig_idx = ids_restore_np[i]  # logits[i] 应该放到原始位置 orig_idx
            restored_logits[0, orig_idx] = logits[0, i]
        pred_patches = restored_logits  # (B, num_total_patches, patch_dim)
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
    # ⚠️ 关键修复：使用更稳健的 Unpatchify 逻辑
    # pred_patches: (B, num_total_patches, patch_dim)
    # patch_dim = patch_size * patch_size * num_channels
    
    # 定义形状变量
    p = patch_size
    c = config.num_channels
    h_patches = h
    w_patches = w
    
    # 将 (B, N, D) reshape 为 (B, h_patches, w_patches, D)
    # 假设 patches 是按 Raster Order 排列的（从左到右、从上到下）
    x = pred_patches.reshape(B, h_patches, w_patches, -1)
    
    # 将最后一维 reshape 为 (patch_size, patch_size, num_channels)
    # 这是最常见的 ViT/MAE 排列方式：(p, p, c)
    x = x.reshape(B, h_patches, w_patches, p, p, c)
    
    # 使用 einsum 进行维度重排：从 (B, h, w, p, p, c) 到 (B, c, h*p, w*p)
    # 'nhwpqc->nchpwq' 表示：
    # n: batch, h: height_patches, w: width_patches, p: patch_size, q: patch_size, c: channels
    # 目标：nchpwq -> n c (h*p) (w*q)
    reconstructed_patches = torch.einsum('nhwpqc->nchpwq', x)
    reconstructed_patches = reconstructed_patches.reshape(B, c, h_patches * p, w_patches * p)
    
    # 5. 处理 norm_pix_loss 的情况
    original_img_tensor = pixel_values[0]  # (4, 224, 224)
    
    if config.norm_pix_loss:
        # 需要反归一化：denormalized = normalized * std + mean
        reconstructed_denorm = torch.zeros_like(reconstructed_patches[0])
        
        for i in range(h):
            for j in range(w):
                i_start = i * patch_size
                i_end = (i + 1) * patch_size
                j_start = j * patch_size
                j_end = (j + 1) * patch_size
                
                original_patch = original_img_tensor[:, i_start:i_end, j_start:j_end]  # (4, patch_size, patch_size)
                patch_mean = original_patch.mean(dim=(1, 2), keepdim=True)  # (4, 1, 1)
                patch_std = original_patch.std(dim=(1, 2), keepdim=True) + 1e-6  # (4, 1, 1)
                
                reconstructed_patch_norm = reconstructed_patches[0, :, i_start:i_end, j_start:j_end]
                reconstructed_patch_denorm = reconstructed_patch_norm * patch_std + patch_mean
                
                reconstructed_denorm[:, i_start:i_end, j_start:j_end] = reconstructed_patch_denorm
        
        reconstructed_patches_denorm = reconstructed_denorm.unsqueeze(0)  # (1, 4, 224, 224)
    else:
        reconstructed_patches_denorm = reconstructed_patches
    
    # 转换为 numpy 用于可视化和计算指标
    original = original_img_tensor.cpu().numpy()  # (4, H, W)
    reconstructed = reconstructed_patches_denorm[0].cpu().numpy()  # (4, H, W)
    mask_np = mask[0].cpu().numpy()  # (num_patches,)
    
    # 归一化到 [0, 1]
    original = np.clip(original, 0, 1)
    reconstructed = np.clip(reconstructed, 0, 1)
    
    print(f"\n✓ 原始图像形状: {original.shape}")
    print(f"✓ 重建图像形状: {reconstructed.shape}")
    print(f"✓ 各通道范围（原始）:")
    print(f"  - Intensity (ch0): [{original[0].min():.4f}, {original[0].max():.4f}], 均值: {original[0].mean():.4f}")
    print(f"  - DoLP (ch1): [{original[1].min():.4f}, {original[1].max():.4f}], 均值: {original[1].mean():.4f}")
    print(f"  - sin(2*AoLP) (ch2): [{original[2].min():.4f}, {original[2].max():.4f}], 均值: {original[2].mean():.4f}")
    print(f"  - cos(2*AoLP) (ch3): [{original[3].min():.4f}, {original[3].max():.4f}], 均值: {original[3].mean():.4f}")
    print(f"✓ 各通道范围（重建）:")
    print(f"  - Intensity (ch0): [{reconstructed[0].min():.4f}, {reconstructed[0].max():.4f}], 均值: {reconstructed[0].mean():.4f}")
    print(f"  - DoLP (ch1): [{reconstructed[1].min():.4f}, {reconstructed[1].max():.4f}], 均值: {reconstructed[1].mean():.4f}")
    print(f"  - sin(2*AoLP) (ch2): [{reconstructed[2].min():.4f}, {reconstructed[2].max():.4f}], 均值: {reconstructed[2].mean():.4f}")
    print(f"  - cos(2*AoLP) (ch3): [{reconstructed[3].min():.4f}, {reconstructed[3].max():.4f}], 均值: {reconstructed[3].mean():.4f}")
    
    # 计算均值差异（用于诊断"偏暗"问题）
    mean_diff = reconstructed.mean(axis=(1, 2)) - original.mean(axis=(1, 2))
    print(f"✓ 均值差异（重建 - 原始，负值表示偏暗）:")
    print(f"  - Intensity: {mean_diff[0]:.4f}")
    print(f"  - DoLP: {mean_diff[1]:.4f}")
    print(f"  - sin(2*AoLP): {mean_diff[2]:.4f}")
    print(f"  - cos(2*AoLP): {mean_diff[3]:.4f}")
    
    # 计算详细的 PSNR 指标（同时计算全图和 Masked 区域）
    metrics = compute_metrics(original, reconstructed, mask=mask_np, patch_size=patch_size)
    
    # ⚠️ 关键诊断：打印Loss和Masked MSE的对比，验证计算是否正确
    if 'mse_masked_mean' in metrics:
        print(f"\n  🔍 诊断：Loss vs Masked MSE 对比")
        print(f"    - 模型 Loss: {loss:.6f}")
        print(f"    - Masked MSE (计算值): {metrics['mse_masked_mean']:.6f}")
        print(f"    - 差异: {abs(loss - metrics['mse_masked_mean']):.6f}")
        if abs(loss - metrics['mse_masked_mean']) < 0.001:
            print(f"    ✓ Loss 和 Masked MSE 匹配良好，验证代码正确")
        else:
            print(f"    ⚠ 警告：Loss 和 Masked MSE 差异较大，可能存在问题")
            print(f"      可能原因：")
            print(f"        1. norm_pix_loss 处理不一致")
            print(f"        2. 损失计算方式不同（如通道加权）")
            print(f"        3. 数值精度问题")
    
    print("\n" + "=" * 80)
    print("重建质量量化指标（详细）")
    print("=" * 80)
    
    # 全图指标
    print("\n📊 MSE (Mean Squared Error) - 全图:")
    print(f"  - Intensity:   {metrics['mse_intensity']:.6f}")
    print(f"  - DoLP:        {metrics['mse_dolp']:.6f}")
    print(f"  - sin(2*AoLP): {metrics['mse_sin_2aolp']:.6f}")
    print(f"  - cos(2*AoLP): {metrics['mse_cos_2aolp']:.6f}")
    print(f"  - 平均 (Mean):  {metrics['mse_mean']:.6f}")
    
    print("\n📈 PSNR (Peak Signal-to-Noise Ratio, 单位: dB, 值越大越好) - 全图:")
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
        print(f"  - Intensity:   {metrics['mse_masked_intensity']:.6f}")
        print(f"  - DoLP:        {metrics['mse_masked_dolp']:.6f}")
        print(f"  - sin(2*AoLP): {metrics['mse_masked_sin_2aolp']:.6f}")
        print(f"  - cos(2*AoLP): {metrics['mse_masked_cos_2aolp']:.6f}")
        print(f"  - 平均 (Mean):  {metrics['mse_masked_mean']:.6f}")
        print(f"  ⚠️  模型 Loss: {loss:.6f} (应该与上面的 MSE 接近)")
        
        print("\n📈 PSNR (Peak Signal-to-Noise Ratio, 单位: dB) - 仅 Masked 区域:")
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
        print(f"  - 差异: 全图 MSE 是 Masked MSE 的 {metrics['mse_mean'] / metrics['mse_masked_mean']:.2f} 倍")
        print(f"\n  💡 解释：")
        print(f"     - 如果 Masked MSE ≈ Loss，说明验证代码正确")
        print(f"     - 如果全图 MSE >> Masked MSE，说明 Visible 区域存在数值偏移（如'偏暗'）")
        print(f"     - 这是正常的，不影响特征提取能力")
    
    # 质量评估（优先使用 Masked 区域的 PSNR，因为它与 Loss 对齐）
    if 'psnr_masked_mean' in metrics:
        psnr_mean = metrics['psnr_masked_mean']  # 使用 Masked 区域的 PSNR
        print(f"\n  ⚠️  注意：使用 Masked 区域的 PSNR 进行评估（与 Loss 对齐）")
    else:
        psnr_mean = metrics['psnr_mean']  # 使用全图的 PSNR
    
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
        recommendation = "❌ 编码器质量较差，建议重新训练 Stage 1"
    
    print("\n" + "=" * 80)
    print("训练质量评估")
    print("=" * 80)
    print(f"  质量等级: {quality}")
    print(f"  建议: {recommendation}")
    print("=" * 80)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 使用 Masked PSNR 作为标题（如果可用）
    if 'psnr_masked_mean' in metrics:
        title_psnr = metrics['psnr_masked_mean']
        title_text = f"MAE 重建结果对比（调试模式）\nLoss: {loss:.6f} | Masked PSNR: {title_psnr:.2f} dB (可信) | 全图 PSNR: {metrics['psnr_mean']:.2f} dB (仅供参考)"
    else:
        title_psnr = metrics['psnr_mean']
        title_text = f"MAE 重建结果对比（调试模式）\nLoss: {loss:.6f} | PSNR: {title_psnr:.2f} dB"
    
    fig.suptitle(title_text, fontsize=16, fontweight='bold')
    
    channel_names = ["Intensity", "DoLP", "sin(2*AoLP)", "cos(2*AoLP)"]
    
    for i in range(4):
        # 原始图像
        ax = axes[0, i]
        im = ax.imshow(original[i], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"原始 - {channel_names[i]}", fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 重建图像
        ax = axes[1, i]
        im = ax.imshow(reconstructed[i], cmap='gray', vmin=0, vmax=1)
        # 优先显示 Masked PSNR（如果可用）
        if 'psnr_masked_intensity' in metrics:
            psnr_vals = [metrics['psnr_masked_intensity'], metrics['psnr_masked_dolp'], 
                        metrics['psnr_masked_sin_2aolp'], metrics['psnr_masked_cos_2aolp']]
            psnr_val = psnr_vals[i]
            ax.set_title(f"重建 - {channel_names[i]}\nMasked PSNR: {psnr_val:.2f} dB", fontsize=12)
        else:
            psnr_val = [metrics['psnr_intensity'], metrics['psnr_dolp'], 
                        metrics['psnr_sin_2aolp'], metrics['psnr_cos_2aolp']][i]
            ax.set_title(f"重建 - {channel_names[i]}\nPSNR: {psnr_val:.2f} dB", fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 保存图像
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "debug_reconstruction.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ 可视化结果已保存到: {output_file}")
    
    print(f"\n✓ 当前 Loss: {loss:.6f}")
    print("=" * 80)
    
    return loss, metrics


def main(
    checkpoint_dir: str,
    polar_root: str,
    use_pt_data: bool = False,
    pt_root: Optional[str] = None,
    model_name: str = "facebook/vit-mae-base",
    image_size: int = 224,
    patch_size: int = 16,
    num_channels: int = 4,
    norm_pix_loss: bool = False,
    output_dir: str = "./debug_verification",
    hf_token: Optional[str] = None,
):
    """主函数"""
    print("=" * 80)
    print("验证调试模式训练结果")
    print("=" * 80)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 加载模型
    model = load_trained_model(
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,
        norm_pix_loss=norm_pix_loss,
        hf_token=hf_token,
    )
    model = model.to(device)
    
    # 加载数据集（获取第一个样本）
    if use_pt_data:
        base_root = Path(pt_root) if pt_root else Path(polar_root)
        train_pt_root = str(base_root / "train")
        train_polar_root = train_pt_root
    else:
        train_polar_root = polar_root
        train_pt_root = None
    
    dataset = PolarMAEDataset(
        polar_root=train_polar_root,
        image_size=image_size,
        is_train=False,  # 验证模式：不应用数据增强
        use_pt_data=use_pt_data,
        pt_root=train_pt_root,
    )
    
    print(f"\n✓ 数据集大小: {len(dataset)}")
    
    # 获取第一个样本（与调试模式使用的样本相同）
    sample = dataset[0]
    print(f"✓ 使用第一个样本进行验证")
    
    # 可视化重建结果
    output_path = Path(output_dir)
    loss, metrics = visualize_reconstruction(model, sample, output_path, device)
    
    # 打印验证结果总结
    print("\n" + "=" * 80)
    print("验证结果总结")
    print("=" * 80)
    print(f"✓ 模型 Loss: {loss:.6f}")
    
    # 优先使用 Masked 区域的 PSNR（与 Loss 对齐）
    if 'psnr_masked_mean' in metrics:
        psnr_for_summary = metrics['psnr_masked_mean']
        mse_for_summary = metrics['mse_masked_mean']
        print(f"✓ Masked 区域 MSE: {mse_for_summary:.6f} (与 Loss 对齐)")
        print(f"✓ Masked 区域 PSNR: {psnr_for_summary:.2f} dB (与 Loss 对齐)")
        print(f"✓ 全图 PSNR: {metrics['psnr_mean']:.2f} dB (仅供参考，包含 Visible 区域的数值偏移)")
    else:
        psnr_for_summary = metrics['psnr_mean']
        mse_for_summary = metrics['mse_mean']
        print(f"✓ 平均 MSE: {mse_for_summary:.6f}")
        print(f"✓ 平均 PSNR: {psnr_for_summary:.2f} dB")
    
    # 使用与 Loss 对齐的指标进行评估
    if psnr_for_summary >= 25.0:
        print("\n✅ 优秀！重建质量很好，编码器可以直接用于 Stage 2")
        print("   - Masked 区域重建质量优秀（PSNR > 25 dB）")
        print("   - Loss 极低，说明训练成功")
    elif psnr_for_summary >= 20.0:
        print("\n✅ 良好！重建质量不错，编码器可以用于 Stage 2")
        print("   - Masked 区域重建质量良好（PSNR > 20 dB）")
        print("   - Loss 较低，说明训练成功")
    elif psnr_for_summary >= 15.0:
        print("\n⚠️  中等。但 Loss 很低，可以用于 Stage 2")
        print("   - Masked 区域 PSNR 中等，但 Loss 极低")
        print("   - 如果全图 MSE >> Masked MSE，说明存在数值偏移（如'偏暗'）")
        print("   - 这是正常的，不影响特征提取能力")
    else:
        print("\n❌ 较差。建议检查训练配置或增加训练轮数")
        print("   - Masked 区域 PSNR 较低，Loss 可能也较高")
        print("   - 建议重新训练 Stage 1")
    
    print("\n💡 建议：")
    print("  1. 查看可视化结果：debug_reconstruction.png")
    print("  2. 查看详细的 PSNR 指标（见上方输出）")
    print("  3. 如果 PSNR >= 20 dB，可以放心使用完整数据集训练")
    print("  4. 如果 PSNR < 20 dB，可以：")
    print("     - 增加训练 epoch（例如 500-1000）")
    print("     - 调整学习率（例如 2e-3 或 5e-3）")
    print("     - 检查模型是否完全解冻了需要训练的部分")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证调试模式训练结果")
    
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="调试模式训练的检查点目录")
    parser.add_argument("--polar_root", type=str, required=True,
                        help="偏振图像根目录")
    parser.add_argument("--use_pt_data", action="store_true", default=False,
                        help="使用预处理的 .pt 文件")
    parser.add_argument("--pt_root", type=str, default=None,
                        help=".pt 文件根目录")
    parser.add_argument("--model_name", type=str, default="facebook/vit-mae-base",
                        help="MAE 模型名称")
    parser.add_argument("--image_size", type=int, default=224,
                        help="图像尺寸")
    parser.add_argument("--patch_size", type=int, default=16,
                        help="Patch 尺寸")
    parser.add_argument("--num_channels", type=int, default=4,
                        help="输入通道数")
    parser.add_argument("--norm_pix_loss", type=lambda x: (str(x).lower() == 'true'), default=False,
                        nargs='?', const=False,
                        help="归一化像素损失")
    parser.add_argument("--output_dir", type=str, default="./debug_verification",
                        help="输出目录")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face token")
    
    args = parser.parse_args()
    
    main(
        checkpoint_dir=args.checkpoint_dir,
        polar_root=args.polar_root,
        use_pt_data=args.use_pt_data,
        pt_root=args.pt_root,
        model_name=args.model_name,
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_channels=args.num_channels,
        norm_pix_loss=args.norm_pix_loss,
        output_dir=args.output_dir,
        hf_token=args.hf_token,
    )


"""
Stage 1 批量验证脚本：遍历所有场景，统计每个场景的平均指标，并找到 DoLP PSNR 最高的样本

功能：
1. 遍历 polar_root 下的所有场景目录
2. 对每个场景的所有样本进行验证（支持批处理加速）
3. 统计每个场景的平均 Masked MSE 和 Masked PSNR（按通道）
4. 找到所有样本中 DoLP 的 Masked PSNR 最高的单个样本
5. 只输出最高样本的重建对比图
6. 将所有统计结果保存到 JSON 文件
"""

import json
import os
import sys
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import ViTMAEForPreTraining

from verify_stage1 import (
    load_mae_model,
    run_one_example,
    list_base_names,
    visualize_mae_reconstruction,
    preprocess_image,
    compute_metrics,
)
from dataset_common import process_polar_images


def find_all_scenes(polar_root: Path) -> List[str]:
    """
    查找 polar_root 下的所有场景目录
    
    Args:
        polar_root: 偏振图像根目录
        
    Returns:
        scene_ids: 场景ID列表，例如 ["00", "01", "02", ...]
    """
    if not polar_root.exists():
        raise FileNotFoundError(f"目录不存在: {polar_root}")
    
    scene_ids = []
    all_dirs = []
    for item in sorted(polar_root.iterdir()):
        if item.is_dir():
            all_dirs.append(item.name)
            # 检查是否是场景目录（通常场景ID是数字字符串）
            scene_id = item.name
            # 检查目录下是否有 *_000.png 文件（PNG模式）或 *.pt 文件（PT模式）
            has_png = any(item.glob("*_000.png"))
            has_pt = any(item.glob("*.pt"))
            if has_png or has_pt:
                scene_ids.append(scene_id)
            else:
                # 调试信息：显示跳过的目录
                print(f"  ⚠ 跳过目录 {scene_id}（未找到 .pt 或 *_000.png 文件）")
    
    # 调试信息
    if all_dirs:
        print(f"  📁 扫描到的所有目录: {sorted(all_dirs)}")
    if scene_ids:
        print(f"  ✓ 找到的有效场景: {sorted(scene_ids)}")
    else:
        print(f"  ⚠ 警告: 未找到任何有效场景目录")
        print(f"     请检查 {polar_root} 下的目录是否包含 .pt 文件或 *_000.png 文件")
    
    return sorted(scene_ids)


@contextmanager
def suppress_stdout():
    """临时抑制标准输出"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def convert_data_format(pixel_values_tensor: torch.Tensor, mode: str) -> torch.Tensor:
    """
    根据模式转换数据格式
    
    Args:
        pixel_values_tensor: 原始数据 (3, H, W) 或 (4, H, W) - [Intensity, DoLP, sin, cos] 或 [DoLP, sin, cos]
        mode: "dolp" 或 "aolp"
        
    Returns:
        转换后的数据 (3, H, W)
    """
    if len(pixel_values_tensor.shape) != 3:
        raise ValueError(f"输入形状不正确: {pixel_values_tensor.shape}")
    
    # 如果是4通道，去掉Intensity
    if pixel_values_tensor.shape[0] == 4:
        pixel_values_tensor = pixel_values_tensor[1:4, :, :]  # [DoLP, sin, cos]
    elif pixel_values_tensor.shape[0] != 3:
        raise ValueError(f"通道数不正确: {pixel_values_tensor.shape[0]}")
    
    # 现在 pixel_values_tensor 是 (3, H, W) - [DoLP, sin, cos]
    if mode == "dolp":
        # DoLP 模式：复制 DoLP 通道3次 -> [DoLP, DoLP, DoLP]
        dolp_channel = pixel_values_tensor[0:1, :, :]  # (1, H, W)
        pixel_values_converted = dolp_channel.repeat(3, 1, 1)  # (3, H, W)
    elif mode == "aolp":
        # AoLP 模式：提取 sin 和 cos，加上全0通道 -> [sin, cos, 0]
        sin_channel = pixel_values_tensor[1:2, :, :]  # (1, H, W)
        cos_channel = pixel_values_tensor[2:3, :, :]  # (1, H, W)
        _, h, w = sin_channel.shape
        zero_channel = torch.zeros(1, h, w, dtype=sin_channel.dtype)  # (1, H, W)
        pixel_values_converted = torch.cat([sin_channel, cos_channel, zero_channel], dim=0)  # (3, H, W)
    else:
        # 标准模式：保持原样 [DoLP, sin, cos]
        pixel_values_converted = pixel_values_tensor
    
    return pixel_values_converted


def verify_sample_silent(
    model: ViTMAEForPreTraining,
    device: torch.device,
    polar_root: Path,
    scene_id: str,
    base_name: str,
    use_pt_data: bool = False,
    mode: str = "standard",  # "dolp", "aolp", 或 "standard"
) -> Dict[str, float]:
    """
    静默验证单个样本（不输出详细信息，只返回指标）
    
    Args:
        model: MAE 模型
        device: 设备
        polar_root: 偏振图像根目录
        scene_id: 场景ID
        base_name: 样本基础名称
        use_pt_data: 是否使用 .pt 文件
        mode: 数据格式模式 - "dolp" (DoLP专用), "aolp" (AoLP专用), 或 "standard" (标准3通道)
        
    Returns:
        指标字典
    """
    # 加载数据
    if use_pt_data:
        # 修复路径查找顺序：先尝试直接路径，再尝试 train/val 子目录
        # 因为 polar_root 可能已经是 train 目录，不应该再加一层 train
        pt_path = polar_root / scene_id / f"{base_name}.pt"
        if not pt_path.exists():
            pt_path = polar_root / "train" / scene_id / f"{base_name}.pt"
        if not pt_path.exists():
            pt_path = polar_root / "val" / scene_id / f"{base_name}.pt"
        
        if not pt_path.exists():
            raise FileNotFoundError(
                f".pt 文件不存在: {base_name}\n"
                f"已尝试以下路径：\n"
                f"  - {polar_root / scene_id / f'{base_name}.pt'}\n"
                f"  - {polar_root / 'train' / scene_id / f'{base_name}.pt'}\n"
                f"  - {polar_root / 'val' / scene_id / f'{base_name}.pt'}"
            )
        
        pixel_values_tensor = torch.load(pt_path, map_location='cpu', weights_only=False)
        if pixel_values_tensor.dtype != torch.float32:
            pixel_values_tensor = pixel_values_tensor.float()
        
        # 根据模式转换数据格式
        pixel_values_tensor = convert_data_format(pixel_values_tensor, mode)
        pixel_values = pixel_values_tensor.unsqueeze(0)
    else:
        scene_dir = polar_root / scene_id
        polar_paths = {
            "I_0": scene_dir / f"{base_name}_000.png",
            "I_45": scene_dir / f"{base_name}_045.png",
            "I_90": scene_dir / f"{base_name}_090.png",
            "I_135": scene_dir / f"{base_name}_135.png",
        }
        
        missing = [name for name, path in polar_paths.items() if not path.exists()]
        if missing:
            polar_paths = {
                "I_0": scene_dir / f"{base_name}_0.png",
                "I_45": scene_dir / f"{base_name}_45.png",
                "I_90": scene_dir / f"{base_name}_90.png",
                "I_135": scene_dir / f"{base_name}_135.png",
            }
        
        for name, path in polar_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"{name} 图像不存在: {path}")
        
        physics_img = process_polar_images(polar_paths=polar_paths)
        config = model.config
        num_channels = getattr(config, 'num_channels', 3)
        pixel_values = preprocess_image(physics_img, image_size=224, num_channels=num_channels)
        
        # 根据模式转换数据格式
        pixel_values_tensor = pixel_values.squeeze(0)  # (3, H, W)
        pixel_values_tensor = convert_data_format(pixel_values_tensor, mode)
        pixel_values = pixel_values_tensor.unsqueeze(0)  # (1, 3, H, W)
    
    # 静默运行验证
    with suppress_stdout():
        metrics = visualize_mae_reconstruction(
            model, 
            pixel_values.to(device), 
            output_path=os.devnull  # 不保存可视化
        )
    
    return metrics


def verify_batch_samples(
    model: ViTMAEForPreTraining,
    device: torch.device,
    pixel_values_batch: torch.Tensor,
    config,
    original_data_batch: Optional[List[torch.Tensor]] = None,  # 原始3通道数据 [DoLP, sin, cos]
    mode: str = "standard",
) -> List[Dict[str, float]]:
    """
    批处理验证多个样本（利用GPU显存加速）
    
    Args:
        model: MAE 模型
        device: 设备
        pixel_values_batch: 批处理输入 (B, 3, 224, 224)
        config: 模型配置
        
    Returns:
        指标列表
    """
    model.eval()
    num_channels = config.num_channels
    patch_size = config.patch_size
    image_size = config.image_size
    num_patches = (image_size // patch_size) ** 2
    h = w = image_size // patch_size
    p = patch_size
    c = num_channels
    h_patches = h
    w_patches = w
    B = pixel_values_batch.shape[0]
    
    with torch.no_grad():
        # 批处理前向传播（关键：一次性处理整个batch，充分利用GPU）
        outputs = model(pixel_values_batch)
    
    # 处理每个样本
    metrics_list = []
    for b in range(B):
        logits = outputs.logits[b:b+1]  # (1, N, D)
        mask = outputs.mask[b:b+1]  # (1, N)
        pixel_values = pixel_values_batch[b:b+1]  # (1, 3, 224, 224)
        
        # 使用与 visualize_mae_reconstruction 相同的逻辑
        if logits.shape[1] == num_patches:
            # HuggingFace 的 logits 已经是全图预测，直接使用
            pred_patches = logits
        else:
            # 需要手动组合（这种情况较少见）
            pred_patches = logits
        
        # Unpatchify
        x = pred_patches.reshape(1, h_patches, w_patches, -1)
        x = x.reshape(1, h_patches, w_patches, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        reconstructed_patches = x.reshape(1, c, h_patches * p, w_patches * p)
        
        # 处理 norm_pix_loss（如果需要）
        if config.norm_pix_loss:
            # 对每个 patch 进行反归一化
            original_img_tensor = pixel_values[0]  # (3, 224, 224)
            reconstructed_denorm = torch.zeros_like(reconstructed_patches[0])
            
            for i in range(h):
                for j in range(w):
                    i_start = i * patch_size
                    i_end = (i + 1) * patch_size
                    j_start = j * patch_size
                    j_end = (j + 1) * patch_size
                    
                    original_patch = original_img_tensor[:, i_start:i_end, j_start:j_end]
                    patch_mean = original_patch.mean(dim=(1, 2), keepdim=True)
                    patch_std = original_patch.std(dim=(1, 2), keepdim=True) + 1e-6
                    
                    reconstructed_patch_norm = reconstructed_patches[0, :, i_start:i_end, j_start:j_end]
                    reconstructed_patch_denorm = reconstructed_patch_norm * patch_std + patch_mean
                    
                    reconstructed_denorm[:, i_start:i_end, j_start:j_end] = reconstructed_patch_denorm
            
            reconstructed_patches_denorm = reconstructed_denorm.unsqueeze(0)
        else:
            reconstructed_patches_denorm = reconstructed_patches
        
        # 转换为 numpy
        original_img = pixel_values[0].cpu().numpy()
        reconstructed_img = reconstructed_patches_denorm[0].cpu().numpy()
        mask_np = mask[0].cpu().numpy()
        
        # 计算指标
        original_img = np.clip(original_img, 0, 1)
        reconstructed_img = np.clip(reconstructed_img, 0, 1)
        
        metrics = compute_metrics(
            original_img, 
            reconstructed_img, 
            mask=mask_np, 
            patch_size=patch_size, 
            num_channels=num_channels
        )
        
        # 如果提供了原始数据，计算其他通道的重建质量
        if original_data_batch is not None and b < len(original_data_batch):
            original_full = original_data_batch[b].cpu().numpy()  # (3, H, W) - [DoLP, sin, cos]
            original_full = np.clip(original_full, 0, 1)
            
            if mode == "dolp":
                # DoLP模式：用重建的DoLP（所有3个通道都是DoLP）与原始数据的sin和cos比较
                # 注意：这不太合理，因为模型只见过DoLP，但用户想看效果
                # 我们使用重建的第0通道（DoLP）作为"预测"，与原始sin和cos比较
                reconstructed_dolp = reconstructed_img[0:1, :, :]  # (1, H, W) - DoLP
                
                # 计算sin和cos的重建质量（用DoLP重建结果去"预测"sin和cos，虽然不合理）
                # 这里我们计算：原始sin/cos vs 重建DoLP（作为"预测"）
                # 实际上，这更像是看DoLP编码器能否"学习到"sin和cos的信息
                # 但更合理的做法是：用重建的DoLP去"预测"sin和cos，看看相关性
                # 为了简化，我们计算重建DoLP与原始sin/cos的MSE和PSNR
                original_sin = original_full[1:2, :, :]  # (1, H, W)
                original_cos = original_full[2:3, :, :]  # (1, H, W)
                
                # 计算sin的重建质量（用DoLP重建结果）
                sin_mse = ((original_sin - reconstructed_dolp) ** 2)
                sin_mse_masked = sin_mse.copy()
                mask_2d = mask_np.reshape(h, w)
                for i in range(h):
                    for j in range(w):
                        if not mask_2d[i, j]:
                            i_start = i * patch_size
                            i_end = (i + 1) * patch_size
                            j_start = j * patch_size
                            j_end = (j + 1) * patch_size
                            sin_mse_masked[:, i_start:i_end, j_start:j_end] = 0
                sin_mse_masked_val = sin_mse_masked.sum() / (mask_np.sum() * patch_size * patch_size + 1e-8)
                sin_psnr = 10.0 * np.log10(1.0 / (sin_mse_masked_val + 1e-8))
                
                # 计算cos的重建质量（用DoLP重建结果）
                cos_mse = ((original_cos - reconstructed_dolp) ** 2)
                cos_mse_masked = cos_mse.copy()
                for i in range(h):
                    for j in range(w):
                        if not mask_2d[i, j]:
                            i_start = i * patch_size
                            i_end = (i + 1) * patch_size
                            j_start = j * patch_size
                            j_end = (j + 1) * patch_size
                            cos_mse_masked[:, i_start:i_end, j_start:j_end] = 0
                cos_mse_masked_val = cos_mse_masked.sum() / (mask_np.sum() * patch_size * patch_size + 1e-8)
                cos_psnr = 10.0 * np.log10(1.0 / (cos_mse_masked_val + 1e-8))
                
                metrics["mse_masked_sin_2aolp_cross"] = float(sin_mse_masked_val)
                metrics["mse_masked_cos_2aolp_cross"] = float(cos_mse_masked_val)
                metrics["psnr_masked_sin_2aolp_cross"] = float(sin_psnr)
                metrics["psnr_masked_cos_2aolp_cross"] = float(cos_psnr)
                metrics["psnr_masked_aolp_cross"] = float((sin_psnr + cos_psnr) / 2)
                
            elif mode == "aolp":
                # AoLP模式：用重建的sin和cos（第0、1通道）与原始数据的DoLP比较
                reconstructed_sin = reconstructed_img[0:1, :, :]  # (1, H, W) - sin
                reconstructed_cos = reconstructed_img[1:2, :, :]  # (1, H, W) - cos
                original_dolp = original_full[0:1, :, :]  # (1, H, W) - DoLP
                
                # 计算DoLP的重建质量（用sin和cos重建结果的平均值作为"预测"）
                # 或者用sin和cos的某种组合
                # 这里我们用sin和cos的平均值作为DoLP的"预测"
                reconstructed_dolp_pred = (reconstructed_sin + reconstructed_cos) / 2
                
                dolp_mse = ((original_dolp - reconstructed_dolp_pred) ** 2)
                dolp_mse_masked = dolp_mse.copy()
                mask_2d = mask_np.reshape(h, w)
                for i in range(h):
                    for j in range(w):
                        if not mask_2d[i, j]:
                            i_start = i * patch_size
                            i_end = (i + 1) * patch_size
                            j_start = j * patch_size
                            j_end = (j + 1) * patch_size
                            dolp_mse_masked[:, i_start:i_end, j_start:j_end] = 0
                dolp_mse_masked_val = dolp_mse_masked.sum() / (mask_np.sum() * patch_size * patch_size + 1e-8)
                dolp_psnr = 10.0 * np.log10(1.0 / (dolp_mse_masked_val + 1e-8))
                
                metrics["mse_masked_dolp_cross"] = float(dolp_mse_masked_val)
                metrics["psnr_masked_dolp_cross"] = float(dolp_psnr)
        
        metrics_list.append(metrics)
    
    return metrics_list


def verify_single_image_detailed(
    checkpoint_path: str,
    polar_root: str,
    scene_id: str,
    base_name: str,
    output_dir: str = "./stage1_single_validation",
    use_pt_data: bool = False,
    hf_token: Optional[str] = None,
    mode: str = "standard",
    num_eval_runs: int = 10,  # 多次验证取平均，减少随机mask的影响
) -> Dict:
    """
    详细验证单张图片（用于验证 overfit 训练的图片）
    
    Args:
        checkpoint_path: Stage 1 检查点路径
        polar_root: 偏振图像根目录
        scene_id: 场景ID
        base_name: 图片基础名称
        output_dir: 输出目录
        use_pt_data: 是否使用 .pt 文件
        hf_token: Hugging Face token
        mode: 数据格式模式
        
    Returns:
        包含详细指标的字典
    """
    mode_display = {
        "dolp": "DoLP 专用编码器",
        "aolp": "AoLP 专用编码器",
        "standard": "标准3通道编码器"
    }
    print("=" * 80)
    print(f"Stage 1 单张图片详细验证 ({mode_display.get(mode, mode)})")
    print("=" * 80)
    print(f"场景ID: {scene_id}")
    print(f"图片名称: {base_name}")
    print("=" * 80)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载模型
    print("\n正在加载模型...", end="", flush=True)
    with suppress_stdout():
        model = load_mae_model(checkpoint_path, hf_token=hf_token)
    model = model.to(device)
    print(" ✓")
    
    # 创建输出目录
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 验证单张图片
    print(f"\n正在验证图片: scene_id={scene_id}, base_name={base_name}")
    print(f"⚠️  重要提示: MAE每次前向传播都会随机生成mask（mask_ratio=0.75）")
    print(f"   训练loss降到0只表示对训练时见过的mask pattern过拟合")
    print(f"   验证时将进行{num_eval_runs}次验证取平均，以减少随机mask的影响")
    
    # 加载数据
    polar_root_path = Path(polar_root)
    if use_pt_data:
        pt_path = polar_root_path / scene_id / f"{base_name}.pt"
        if not pt_path.exists():
            pt_path = polar_root_path / "train" / scene_id / f"{base_name}.pt"
        if not pt_path.exists():
            pt_path = polar_root_path / "val" / scene_id / f"{base_name}.pt"
        
        if not pt_path.exists():
            raise FileNotFoundError(f".pt 文件不存在: {pt_path}")
        
        pixel_values_tensor = torch.load(pt_path, map_location='cpu', weights_only=False)
        if pixel_values_tensor.dtype != torch.float32:
            pixel_values_tensor = pixel_values_tensor.float()
        
        # 保存原始数据（用于交叉重建）
        if pixel_values_tensor.shape[0] == 4:
            original_full = pixel_values_tensor[1:4, :, :]  # [DoLP, sin, cos]
        else:
            original_full = pixel_values_tensor.clone()  # [DoLP, sin, cos]
        
        # 根据模式转换数据格式
        pixel_values_tensor = convert_data_format(pixel_values_tensor, mode)
        pixel_values = pixel_values_tensor.unsqueeze(0).to(device)
    else:
        scene_dir = polar_root_path / scene_id
        polar_paths = {
            "I_0": scene_dir / f"{base_name}_000.png",
            "I_45": scene_dir / f"{base_name}_045.png",
            "I_90": scene_dir / f"{base_name}_090.png",
            "I_135": scene_dir / f"{base_name}_135.png",
        }
        
        missing = [name for name, path in polar_paths.items() if not path.exists()]
        if missing:
            polar_paths = {
                "I_0": scene_dir / f"{base_name}_0.png",
                "I_45": scene_dir / f"{base_name}_45.png",
                "I_90": scene_dir / f"{base_name}_90.png",
                "I_135": scene_dir / f"{base_name}_135.png",
            }
        
        if not all(p.exists() for p in polar_paths.values()):
            raise FileNotFoundError(f"图片文件不存在: {polar_paths}")
        
        physics_img = process_polar_images(polar_paths=polar_paths)
        config = model.config
        num_channels = getattr(config, 'num_channels', 3)
        pixel_values = preprocess_image(physics_img, image_size=224, num_channels=num_channels)
        
        # 保存原始数据（用于交叉重建）
        original_full = pixel_values.squeeze(0).clone()  # (3, H, W) - [DoLP, sin, cos]
        
        # 根据模式转换数据格式
        pixel_values_tensor = pixel_values.squeeze(0)  # (3, H, W)
        pixel_values_tensor = convert_data_format(pixel_values_tensor, mode)
        pixel_values = pixel_values_tensor.unsqueeze(0).to(device)
    
    # 多次验证取平均（因为MAE每次前向传播都会随机生成mask）
    all_metrics_list = []
    all_model_losses = []  # 收集每次的模型内部loss
    
    print(f"\n正在进行 {num_eval_runs} 次验证（每次mask都不同）...")
    
    for run_idx in range(num_eval_runs):
        original_data_batch = [original_full] if use_pt_data else [original_full]
        
        # 先获取模型内部loss（与训练时一致）
        model.eval()
        with torch.no_grad():
            outputs = model(pixel_values)
            model_loss = outputs.loss.item()
            all_model_losses.append(model_loss)
        
        batch_metrics = verify_batch_samples(
            model=model,
            device=device,
            pixel_values_batch=pixel_values,
            config=model.config,
            original_data_batch=original_data_batch,
            mode=mode,
        )
        
        all_metrics_list.append(batch_metrics[0])
        if (run_idx + 1) % 5 == 0:
            print(f"  已完成 {run_idx + 1}/{num_eval_runs} 次验证...")
    
    # 诊断：对比训练loss和验证MSE
    avg_model_loss = np.mean(all_model_losses)
    std_model_loss = np.std(all_model_losses)
    print(f"\n" + "=" * 80)
    print("诊断：对比训练loss和验证MSE")
    print("=" * 80)
    print(f"  - 模型内部loss (outputs.loss, {num_eval_runs}次平均): {avg_model_loss:.6f} ± {std_model_loss:.6f}")
    print(f"  - 这是训练时使用的loss值")
    
    # 计算平均指标和标准差
    print(f"\n计算 {num_eval_runs} 次验证的平均指标...")
    metrics = {}
    metrics_std = {}
    if all_metrics_list:
        # 对所有指标取平均
        for key in all_metrics_list[0].keys():
            values = [m[key] for m in all_metrics_list if key in m]
            if values:
                metrics[key] = float(np.mean(values))
                metrics_std[key] = float(np.std(values))
        
        print(f"✓ 平均指标计算完成（标准差显示不确定性）")
        
        # 对比训练loss和验证MSE
        masked_mse = metrics.get('mse_masked_mean', 0)
        masked_mse_std = metrics_std.get('mse_masked_mean', 0)
        print(f"\n  - 验证MSE (masked区域, {num_eval_runs}次平均): {masked_mse:.6f} ± {masked_mse_std:.6f}")
        print(f"  - 差异倍数: {masked_mse / (avg_model_loss + 1e-8):.2f}x")
        if abs(masked_mse - avg_model_loss) > 0.001:
            print(f"  ⚠️  警告: 训练loss和验证MSE差异较大！")
            print(f"     可能原因:")
            print(f"     1. MAE的loss计算可能使用了归一化或缩放")
            print(f"     2. 训练时看到的mask pattern和验证时不同（这是正常的）")
            print(f"     3. 模型过拟合了特定的mask pattern（如果差异很大）")
            if masked_mse / (avg_model_loss + 1e-8) > 10:
                print(f"     4. ⚠️  差异超过10倍，可能是计算方式不同或模型严重过拟合")
        else:
            print(f"  ✓ 训练loss和验证MSE基本一致")
    
    # 保存可视化
    output_path = output_dir_path / f"reconstruction_{scene_id}_{base_name}.png"
    print(f"\n正在保存可视化结果...")
    with suppress_stdout():
        visualize_mae_reconstruction(
            model,
            pixel_values,
            output_path=str(output_path)
        )
    print(f"✓ 可视化结果已保存到: {output_path}")
    
    # 打印详细指标（只显示Mask区域的指标）
    print("\n" + "=" * 80)
    print(f"详细指标（Mask区域，{num_eval_runs}次验证平均）")
    print("=" * 80)
    
    if mode == "dolp":
        dolp_mse = metrics.get('mse_masked_dolp', 0)
        dolp_mse_std = metrics_std.get('mse_masked_dolp', 0)
        dolp_psnr = metrics.get('psnr_masked_dolp', 0)
        dolp_psnr_std = metrics_std.get('psnr_masked_dolp', 0)
        print(f"\nDoLP 专用编码器重建结果:")
        print(f"  - Masked DoLP MSE: {dolp_mse:.6f} ± {dolp_mse_std:.6f}")
        print(f"  - Masked DoLP PSNR: {dolp_psnr:.2f} ± {dolp_psnr_std:.2f} dB")
        sin_cross_mse = metrics.get('mse_masked_sin_2aolp_cross', 0)
        sin_cross_mse_std = metrics_std.get('mse_masked_sin_2aolp_cross', 0)
        cos_cross_mse = metrics.get('mse_masked_cos_2aolp_cross', 0)
        cos_cross_mse_std = metrics_std.get('mse_masked_cos_2aolp_cross', 0)
        sin_cross_psnr = metrics.get('psnr_masked_sin_2aolp_cross', 0)
        sin_cross_psnr_std = metrics_std.get('psnr_masked_sin_2aolp_cross', 0)
        cos_cross_psnr = metrics.get('psnr_masked_cos_2aolp_cross', 0)
        cos_cross_psnr_std = metrics_std.get('psnr_masked_cos_2aolp_cross', 0)
        aolp_cross_psnr = metrics.get('psnr_masked_aolp_cross', 0)
        aolp_cross_psnr_std = metrics_std.get('psnr_masked_aolp_cross', 0)
        print(f"\n交叉重建（用DoLP编码器重建AoLP）:")
        print(f"  - Masked sin(2*AoLP) MSE (交叉): {sin_cross_mse:.6f} ± {sin_cross_mse_std:.6f}")
        print(f"  - Masked cos(2*AoLP) MSE (交叉): {cos_cross_mse:.6f} ± {cos_cross_mse_std:.6f}")
        print(f"  - Masked sin(2*AoLP) PSNR (交叉): {sin_cross_psnr:.2f} ± {sin_cross_psnr_std:.2f} dB")
        print(f"  - Masked cos(2*AoLP) PSNR (交叉): {cos_cross_psnr:.2f} ± {cos_cross_psnr_std:.2f} dB")
        print(f"  - Masked AoLP PSNR (交叉，平均): {aolp_cross_psnr:.2f} ± {aolp_cross_psnr_std:.2f} dB")
    elif mode == "aolp":
        sin_psnr = metrics.get("psnr_masked_dolp", 0)  # 实际是 sin
        cos_psnr = metrics.get("psnr_masked_sin_2aolp", 0)  # 实际是 cos
        aolp_psnr = (sin_psnr + cos_psnr) / 2
        print(f"\nAoLP 专用编码器重建结果:")
        print(f"  - Masked sin(2*AoLP) MSE: {metrics.get('mse_masked_dolp', 0):.6f}")
        print(f"  - Masked cos(2*AoLP) MSE: {metrics.get('mse_masked_sin_2aolp', 0):.6f}")
        print(f"  - Masked sin(2*AoLP) PSNR: {sin_psnr:.2f} dB")
        print(f"  - Masked cos(2*AoLP) PSNR: {cos_psnr:.2f} dB")
        print(f"  - Masked AoLP PSNR (平均): {aolp_psnr:.2f} dB")
        print(f"\n交叉重建（用AoLP编码器重建DoLP）:")
        print(f"  - Masked DoLP MSE (交叉): {metrics.get('mse_masked_dolp_cross', 0):.6f}")
        print(f"  - Masked DoLP PSNR (交叉): {metrics.get('psnr_masked_dolp_cross', 0):.2f} dB")
    else:
        print(f"\n标准3通道编码器重建结果:")
        print(f"  - Masked DoLP MSE: {metrics.get('mse_masked_dolp', 0):.6f}")
        print(f"  - Masked sin(2*AoLP) MSE: {metrics.get('mse_masked_sin_2aolp', 0):.6f}")
        print(f"  - Masked cos(2*AoLP) MSE: {metrics.get('mse_masked_cos_2aolp', 0):.6f}")
        print(f"  - Masked DoLP PSNR: {metrics.get('psnr_masked_dolp', 0):.2f} dB")
        print(f"  - Masked sin(2*AoLP) PSNR: {metrics.get('psnr_masked_sin_2aolp', 0):.2f} dB")
        print(f"  - Masked cos(2*AoLP) PSNR: {metrics.get('psnr_masked_cos_2aolp', 0):.2f} dB")
        sin_psnr = metrics.get("psnr_masked_sin_2aolp", 0)
        cos_psnr = metrics.get("psnr_masked_cos_2aolp", 0)
        aolp_psnr = (sin_psnr + cos_psnr) / 2
        print(f"  - Masked AoLP PSNR (平均): {aolp_psnr:.2f} dB")
    
    mean_mse = metrics.get('mse_masked_mean', 0)
    mean_mse_std = metrics_std.get('mse_masked_mean', 0)
    mean_psnr = metrics.get('psnr_masked_mean', 0)
    mean_psnr_std = metrics_std.get('psnr_masked_mean', 0)
    print(f"\n  - Masked Mean MSE: {mean_mse:.6f} ± {mean_mse_std:.6f}")
    print(f"  - Masked Mean PSNR: {mean_psnr:.2f} ± {mean_psnr_std:.2f} dB")
    
    # 诊断信息
    print(f"\n" + "=" * 80)
    print("诊断信息")
    print("=" * 80)
    print(f"  - 验证次数: {num_eval_runs}")
    print(f"  - 如果PSNR的标准差很大（>5 dB），说明模型对不同mask的重建质量差异很大")
    print(f"  - 如果PSNR的标准差很小（<1 dB），说明模型对不同mask都能稳定重建")
    if mean_psnr_std > 5:
        print(f"  ⚠️  警告: PSNR标准差较大（{mean_psnr_std:.2f} dB），模型可能对不同mask的重建质量不稳定")
        print(f"     这可能是因为模型只对训练时见过的mask pattern过拟合了")
    elif mean_psnr_std < 1:
        print(f"  ✓ PSNR标准差较小（{mean_psnr_std:.2f} dB），模型对不同mask的重建质量稳定")
    
    # 保存结果到JSON
    result = {
        "checkpoint": checkpoint_path,
        "polar_root": polar_root,
        "scene_id": scene_id,
        "base_name": base_name,
        "mode": mode,
        "num_eval_runs": num_eval_runs,
        "output_path": str(output_path),
        "metrics": metrics,
        "metrics_std": metrics_std,  # 标准差
        "all_runs_metrics": all_metrics_list,  # 所有运行的详细指标
    }
    
    json_path = output_dir_path / "single_image_validation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 详细结果已保存到: {json_path}")
    
    return result


def verify_all_scenes(
    checkpoint_path: str,
    polar_root: str,
    output_dir: str = "./stage1_batch_validation",
    use_pt_data: bool = False,
    hf_token: Optional[str] = None,
    batch_size: int = 8,
    mode: str = "standard",  # "dolp", "aolp", 或 "standard"
) -> Dict:
    """
    批量验证所有场景
    
    Args:
        checkpoint_path: Stage 1 检查点路径
        polar_root: 偏振图像根目录
        output_dir: 输出目录
        use_pt_data: 是否使用 .pt 文件
        hf_token: Hugging Face token
        batch_size: 批处理大小
        mode: 数据格式模式 - "dolp" (DoLP专用), "aolp" (AoLP专用), 或 "standard" (标准3通道)
        
    Returns:
        包含所有统计结果的字典
    """
    mode_display = {
        "dolp": "DoLP 专用编码器",
        "aolp": "AoLP 专用编码器",
        "standard": "标准3通道编码器"
    }
    print("=" * 80)
    print(f"Stage 1 批量验证：遍历所有场景 ({mode_display.get(mode, mode)})")
    print("=" * 80)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"批处理大小: {batch_size} (利用GPU显存加速)")
        # 显示初始显存使用
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"初始显存占用: {initial_memory:.2f} GB\n")
    else:
        batch_size = 1  # CPU模式使用单样本处理
        print("CPU模式，使用单样本处理\n")
    
    # 加载模型（静默模式）
    print("正在加载模型...", end="", flush=True)
    with suppress_stdout():
        model = load_mae_model(checkpoint_path, hf_token=hf_token)
    model = model.to(device)
    print(" ✓")
    
    # 查找所有场景
    polar_root_path = Path(polar_root)
    scene_ids = find_all_scenes(polar_root_path)
    print(f"找到 {len(scene_ids)} 个场景\n")
    
    if not scene_ids:
        raise ValueError(f"在 {polar_root} 中未找到任何场景目录")
    
    # 创建输出目录
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 存储所有结果
    all_results = {
        "checkpoint": checkpoint_path,
        "polar_root": polar_root,
        "total_scenes": len(scene_ids),
        "scenes": {},
        "high_quality_samples": [],  # 所有 DoLP PSNR >= 30 dB 的样本
        "dolp_psnr_threshold": 30.0,  # DoLP PSNR 阈值
        "best_sample": None,  # 兼容旧格式：全局最高的 DoLP PSNR 样本
    }
    
    # 所有 DoLP PSNR >= 30 dB 的样本信息
    high_quality_samples = []  # 存储 (psnr, scene_id, base_name, metrics) 的列表
    dolp_psnr_threshold = 30.0  # PSNR 阈值
    
    # 收集所有样本用于找后20名
    all_samples_global = []  # 存储 (psnr, scene_id, base_name, metrics) 的列表
    
    # 统计总样本数（用于进度条）
    total_samples = 0
    scene_sample_map = {}  # {scene_id: [base_names]}
    for scene_id in scene_ids:
        scene_dir = polar_root_path / scene_id
        if use_pt_data:
            pt_files = sorted(scene_dir.glob("*.pt"))
            base_names = [f.stem for f in pt_files]
        else:
            base_names = list_base_names(scene_dir)
        if base_names:
            scene_sample_map[scene_id] = base_names
            total_samples += len(base_names)
    
    # 创建总体进度条
    pbar = tqdm(total=total_samples, desc="验证进度", unit="样本", ncols=100)
    
    # 遍历每个场景
    for scene_idx, scene_id in enumerate(scene_ids, 1):
        if scene_id not in scene_sample_map:
            all_results["scenes"][scene_id] = {
                "num_samples": 0,
                "skipped": True,
            }
            continue
        
        base_names = scene_sample_map[scene_id]
        scene_dir = polar_root_path / scene_id
        
        # 存储该场景的所有样本指标
        scene_metrics_list = []
        
        # 批处理验证
        if device.type == "cuda" and batch_size > 1:
            # GPU批处理模式
            for batch_start in range(0, len(base_names), batch_size):
                batch_end = min(batch_start + batch_size, len(base_names))
                batch_names = base_names[batch_start:batch_end]
                
                # 加载批处理数据
                pixel_values_batch = []
                original_data_batch = []  # 保存原始3通道数据 [DoLP, sin, cos]
                valid_indices = []
                valid_names = []
                
                for idx, base_name in enumerate(batch_names):
                    try:
                        if use_pt_data:
                            # 修复路径查找顺序：先尝试直接路径，再尝试 train/val 子目录
                            # 因为 polar_root 可能已经是 train 目录，不应该再加一层 train
                            pt_path = polar_root_path / scene_id / f"{base_name}.pt"
                            if not pt_path.exists():
                                pt_path = polar_root_path / "train" / scene_id / f"{base_name}.pt"
                            if not pt_path.exists():
                                pt_path = polar_root_path / "val" / scene_id / f"{base_name}.pt"
                            
                            if not pt_path.exists():
                                continue
                            
                            pixel_values_tensor = torch.load(pt_path, map_location='cpu', weights_only=False)
                            if pixel_values_tensor.dtype != torch.float32:
                                pixel_values_tensor = pixel_values_tensor.float()
                            
                            if len(pixel_values_tensor.shape) != 3:
                                continue
                            
                            # 保存原始数据（在转换格式之前）
                            if pixel_values_tensor.shape[0] == 4:
                                original_full = pixel_values_tensor[1:4, :, :]  # [DoLP, sin, cos]
                            else:
                                original_full = pixel_values_tensor.clone()  # [DoLP, sin, cos]
                            original_data_batch.append(original_full)
                            
                            # 根据模式转换数据格式
                            pixel_values_tensor = convert_data_format(pixel_values_tensor, mode)
                            pixel_values = pixel_values_tensor.unsqueeze(0)
                        else:
                            polar_paths = {
                                "I_0": scene_dir / f"{base_name}_000.png",
                                "I_45": scene_dir / f"{base_name}_045.png",
                                "I_90": scene_dir / f"{base_name}_090.png",
                                "I_135": scene_dir / f"{base_name}_135.png",
                            }
                            
                            missing = [name for name, path in polar_paths.items() if not path.exists()]
                            if missing:
                                polar_paths = {
                                    "I_0": scene_dir / f"{base_name}_0.png",
                                    "I_45": scene_dir / f"{base_name}_45.png",
                                    "I_90": scene_dir / f"{base_name}_90.png",
                                    "I_135": scene_dir / f"{base_name}_135.png",
                                }
                            
                            if not all(p.exists() for p in polar_paths.values()):
                                continue
                            
                            physics_img = process_polar_images(polar_paths=polar_paths)
                            config = model.config
                            num_channels = getattr(config, 'num_channels', 3)
                            pixel_values = preprocess_image(physics_img, image_size=224, num_channels=num_channels)
                            
                            # 保存原始数据（在转换格式之前）
                            original_full = pixel_values.squeeze(0).clone()  # (3, H, W) - [DoLP, sin, cos]
                            original_data_batch.append(original_full)
                            
                            # 根据模式转换数据格式
                            pixel_values_tensor = pixel_values.squeeze(0)  # (3, H, W)
                            pixel_values_tensor = convert_data_format(pixel_values_tensor, mode)
                            pixel_values = pixel_values_tensor.unsqueeze(0)  # (1, 3, H, W)
                        
                        pixel_values_batch.append(pixel_values)
                        valid_indices.append(batch_start + idx)
                        valid_names.append(base_name)
                    except Exception:
                        continue
                
                if not pixel_values_batch:
                    pbar.update(len(batch_names))
                    continue
                
                # 批处理前向传播
                pixel_values_batch_tensor = torch.cat(pixel_values_batch, dim=0).to(device)
                actual_batch_size = pixel_values_batch_tensor.shape[0]
                
                try:
                    # 真正的批处理验证（一次性处理多个样本，充分利用GPU）
                    # 注意：MAE模型支持批处理，每个样本的mask是独立的
                    batch_metrics = verify_batch_samples(
                        model=model,
                        device=device,
                        pixel_values_batch=pixel_values_batch_tensor,
                        config=model.config,
                        original_data_batch=original_data_batch,
                        mode=mode,
                    )
                    
                    # 释放显存（避免OOM）
                    del pixel_values_batch_tensor
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    
                    # 处理每个样本的结果
                    for i, (base_name, metrics) in enumerate(zip(valid_names, batch_metrics)):
                        pbar.update(1)
                        
                        # 根据模式检查 PSNR 是否 >= 30 dB
                        if mode == "dolp":
                            # DoLP 模式：检查 DoLP PSNR
                            if "psnr_masked_dolp" in metrics:
                                dolp_psnr = metrics["psnr_masked_dolp"]
                                if dolp_psnr >= dolp_psnr_threshold:
                                    high_quality_samples.append((dolp_psnr, scene_id, base_name, metrics.copy()))
                        elif mode == "aolp":
                            # AoLP 模式：检查 AoLP PSNR（sin 和 cos 的平均值）
                            # 注意：compute_metrics 的命名是 [dolp, sin, cos]，但实际数据是 [sin, cos, 0]
                            sin_psnr = metrics.get("psnr_masked_dolp", 0)  # 实际是 sin
                            cos_psnr = metrics.get("psnr_masked_sin_2aolp", 0)  # 实际是 cos
                            aolp_psnr = (sin_psnr + cos_psnr) / 2
                            if aolp_psnr >= dolp_psnr_threshold:
                                high_quality_samples.append((aolp_psnr, scene_id, base_name, metrics.copy()))
                        else:
                            # 标准模式：检查 DoLP PSNR
                            if "psnr_masked_dolp" in metrics:
                                dolp_psnr = metrics["psnr_masked_dolp"]
                                if dolp_psnr >= dolp_psnr_threshold:
                                    high_quality_samples.append((dolp_psnr, scene_id, base_name, metrics.copy()))
                        
                        scene_metrics_list.append({
                            "base_name": base_name,
                            "metrics": metrics,
                        })
                        
                        # 收集到全局列表（用于找后20名）
                        if mode == "dolp":
                            psnr_val = metrics.get("psnr_masked_dolp", 0)
                        elif mode == "aolp":
                            sin_psnr = metrics.get("psnr_masked_dolp", 0)  # 实际是 sin
                            cos_psnr = metrics.get("psnr_masked_sin_2aolp", 0)  # 实际是 cos
                            psnr_val = (sin_psnr + cos_psnr) / 2
                        else:
                            psnr_val = metrics.get("psnr_masked_dolp", 0)
                        all_samples_global.append((psnr_val, scene_id, base_name, metrics.copy()))
                except Exception as e:
                    # 如果批处理失败，回退到单样本模式
                    for base_name in valid_names:
                        try:
                            metrics = verify_sample_silent(
                                model, device, polar_root_path, scene_id, base_name, use_pt_data, mode=mode
                            )
                            pbar.update(1)
                            
                            # 根据模式检查 PSNR 是否 >= 30 dB
                            if mode == "dolp":
                                if "psnr_masked_dolp" in metrics:
                                    dolp_psnr = metrics["psnr_masked_dolp"]
                                    if dolp_psnr >= dolp_psnr_threshold:
                                        high_quality_samples.append((dolp_psnr, scene_id, base_name, metrics.copy()))
                            elif mode == "aolp":
                                sin_psnr = metrics.get("psnr_masked_dolp", 0)  # 实际是 sin
                                cos_psnr = metrics.get("psnr_masked_sin_2aolp", 0)  # 实际是 cos
                                aolp_psnr = (sin_psnr + cos_psnr) / 2
                                if aolp_psnr >= dolp_psnr_threshold:
                                    high_quality_samples.append((aolp_psnr, scene_id, base_name, metrics.copy()))
                            else:
                                if "psnr_masked_dolp" in metrics:
                                    dolp_psnr = metrics["psnr_masked_dolp"]
                                    if dolp_psnr >= dolp_psnr_threshold:
                                        high_quality_samples.append((dolp_psnr, scene_id, base_name, metrics.copy()))
                            
                            scene_metrics_list.append({
                                "base_name": base_name,
                                "metrics": metrics,
                            })
                            
                            # 收集到全局列表（用于找后20名）
                            if mode == "dolp":
                                psnr_val = metrics.get("psnr_masked_dolp", 0)
                            elif mode == "aolp":
                                sin_psnr = metrics.get("psnr_masked_dolp", 0)  # 实际是 sin
                                cos_psnr = metrics.get("psnr_masked_sin_2aolp", 0)  # 实际是 cos
                                psnr_val = (sin_psnr + cos_psnr) / 2
                            else:
                                psnr_val = metrics.get("psnr_masked_dolp", 0)
                            all_samples_global.append((psnr_val, scene_id, base_name, metrics.copy()))
                        except Exception:
                            pbar.update(1)
                    continue
                
                # 更新进度条（处理跳过的样本）
                skipped = len(batch_names) - len(valid_names)
                if skipped > 0:
                    pbar.update(skipped)
        else:
            # 单样本模式（CPU或batch_size=1）
            for base_name in base_names:
                try:
                    metrics = verify_sample_silent(
                        model, device, polar_root_path, scene_id, base_name, use_pt_data, mode=mode
                    )
                    pbar.update(1)
                    
                    # 根据模式检查 PSNR 是否 >= 30 dB
                    if mode == "dolp":
                        if "psnr_masked_dolp" in metrics:
                            dolp_psnr = metrics["psnr_masked_dolp"]
                            if dolp_psnr >= dolp_psnr_threshold:
                                high_quality_samples.append((dolp_psnr, scene_id, base_name, metrics.copy()))
                    elif mode == "aolp":
                        sin_psnr = metrics.get("psnr_masked_dolp", 0)  # 实际是 sin
                        cos_psnr = metrics.get("psnr_masked_sin_2aolp", 0)  # 实际是 cos
                        aolp_psnr = (sin_psnr + cos_psnr) / 2
                        if aolp_psnr >= dolp_psnr_threshold:
                            high_quality_samples.append((aolp_psnr, scene_id, base_name, metrics.copy()))
                    else:
                        if "psnr_masked_dolp" in metrics:
                            dolp_psnr = metrics["psnr_masked_dolp"]
                            if dolp_psnr >= dolp_psnr_threshold:
                                high_quality_samples.append((dolp_psnr, scene_id, base_name, metrics.copy()))
                    
                    scene_metrics_list.append({
                        "base_name": base_name,
                        "metrics": metrics,
                    })
                    
                    # 收集到全局列表（用于找后20名）
                    if mode == "dolp":
                        psnr_val = metrics.get("psnr_masked_dolp", 0)
                    elif mode == "aolp":
                        sin_psnr = metrics.get("psnr_masked_dolp", 0)  # 实际是 sin
                        cos_psnr = metrics.get("psnr_masked_sin_2aolp", 0)  # 实际是 cos
                        psnr_val = (sin_psnr + cos_psnr) / 2
                    else:
                        psnr_val = metrics.get("psnr_masked_dolp", 0)
                    all_samples_global.append((psnr_val, scene_id, base_name, metrics.copy()))
                except Exception as e:
                    pbar.update(1)
                    continue
        
        # 计算该场景的平均指标
        if scene_metrics_list:
            num_samples = len(scene_metrics_list)
            
            # 根据模式收集相关指标
            if mode == "dolp":
                # DoLP 模式：所有3个通道都是 DoLP，只统计 DoLP 指标（使用第0通道）
                all_mse_dolp = [m["metrics"].get("mse_masked_dolp", 0) for m in scene_metrics_list]
                all_psnr_dolp = [m["metrics"].get("psnr_masked_dolp", 0) for m in scene_metrics_list]
                
                scene_stats = {
                    "num_samples": num_samples,
                    "mse_masked_dolp": float(np.mean(all_mse_dolp)),
                    "psnr_masked_dolp": float(np.mean(all_psnr_dolp)),
                }
            elif mode == "aolp":
                # AoLP 模式：第0通道是 sin，第1通道是 cos，第2通道是全0
                # 注意：compute_metrics 的命名是 [dolp, sin, cos]，但实际数据是 [sin, cos, 0]
                # 所以：dolp 指标对应 sin，sin 指标对应 cos，cos 指标对应全0通道（忽略）
                all_mse_sin = [m["metrics"].get("mse_masked_dolp", 0) for m in scene_metrics_list]  # 实际是 sin
                all_mse_cos = [m["metrics"].get("mse_masked_sin_2aolp", 0) for m in scene_metrics_list]  # 实际是 cos
                all_psnr_sin = [m["metrics"].get("psnr_masked_dolp", 0) for m in scene_metrics_list]  # 实际是 sin
                all_psnr_cos = [m["metrics"].get("psnr_masked_sin_2aolp", 0) for m in scene_metrics_list]  # 实际是 cos
                
                scene_stats = {
                    "num_samples": num_samples,
                    "mse_masked_sin_2aolp": float(np.mean(all_mse_sin)),
                    "mse_masked_cos_2aolp": float(np.mean(all_mse_cos)),
                    "psnr_masked_sin_2aolp": float(np.mean(all_psnr_sin)),
                    "psnr_masked_cos_2aolp": float(np.mean(all_psnr_cos)),
                }
            else:
                # 标准模式：标准3通道 [DoLP, sin, cos]
                all_mse_dolp = [m["metrics"].get("mse_masked_dolp", 0) for m in scene_metrics_list]
                all_mse_sin = [m["metrics"].get("mse_masked_sin_2aolp", 0) for m in scene_metrics_list]
                all_mse_cos = [m["metrics"].get("mse_masked_cos_2aolp", 0) for m in scene_metrics_list]
                all_psnr_dolp = [m["metrics"].get("psnr_masked_dolp", 0) for m in scene_metrics_list]
                all_psnr_sin = [m["metrics"].get("psnr_masked_sin_2aolp", 0) for m in scene_metrics_list]
                all_psnr_cos = [m["metrics"].get("psnr_masked_cos_2aolp", 0) for m in scene_metrics_list]
                
                scene_stats = {
                    "num_samples": num_samples,
                    "mse_masked_dolp": float(np.mean(all_mse_dolp)),
                    "mse_masked_sin_2aolp": float(np.mean(all_mse_sin)),
                    "mse_masked_cos_2aolp": float(np.mean(all_mse_cos)),
                    "psnr_masked_dolp": float(np.mean(all_psnr_dolp)),
                    "psnr_masked_sin_2aolp": float(np.mean(all_psnr_sin)),
                    "psnr_masked_cos_2aolp": float(np.mean(all_psnr_cos)),
                }
            
            # 更新进度条描述
            best_dolp_psnr = max([s[0] for s in high_quality_samples], default=0.0)
            pbar.set_postfix({
                "场景": scene_id,
                "样本": num_samples,
                "最佳DoLP": f"{best_dolp_psnr:.1f}dB" if best_dolp_psnr > 0 else "N/A",
                f">={dolp_psnr_threshold}dB": len(high_quality_samples)
            })
            
            all_results["scenes"][scene_id] = scene_stats
        else:
            all_results["scenes"][scene_id] = {
                "num_samples": 0,
                "skipped": True,
            }
    
    pbar.close()
    
    # 显示最终显存使用
    if device.type == "cuda" and torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        current_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"\n显存使用: 峰值={peak_memory:.2f} GB, 当前={current_memory:.2f} GB")
    
    # 按场景分组，每个场景只保留 PSNR 最高和最低的样本
    scene_best_samples = {}  # {scene_id: (psnr, base_name, metrics)}
    scene_worst_samples = {}  # {scene_id: (psnr, base_name, metrics)} - 后20名
    
    for psnr, scene_id, base_name, metrics in high_quality_samples:
        if scene_id not in scene_best_samples:
            scene_best_samples[scene_id] = (psnr, base_name, metrics)
        else:
            # 如果当前样本的 PSNR 更高，则替换
            current_best_psnr = scene_best_samples[scene_id][0]
            if psnr > current_best_psnr:
                scene_best_samples[scene_id] = (psnr, base_name, metrics)
    
    # 使用全局收集的样本列表找后20名
    all_samples_with_psnr = all_samples_global
    
    # 按PSNR排序，找后20名（每个场景只保留一个最差的）
    all_samples_with_psnr_sorted = sorted(all_samples_with_psnr, key=lambda x: x[0])
    
    # 收集后20名，每个场景只保留一个最差的
    worst_count = 0
    worst_scenes_used = set()
    for psnr, scene_id, base_name, metrics in all_samples_with_psnr_sorted:
        if worst_count >= 20:
            break
        if scene_id not in worst_scenes_used:
            scene_worst_samples[scene_id] = (psnr, base_name, metrics)
            worst_scenes_used.add(scene_id)
            worst_count += 1
    
    # 按 PSNR 降序排列（用于显示）
    best_samples_sorted = sorted(
        [(psnr, scene_id, base_name, metrics) for scene_id, (psnr, base_name, metrics) in scene_best_samples.items()],
        key=lambda x: x[0],
        reverse=True
    )
    
    worst_samples_sorted = sorted(
        [(psnr, scene_id, base_name, metrics) for scene_id, (psnr, base_name, metrics) in scene_worst_samples.items()],
        key=lambda x: x[0],
        reverse=False  # 升序，最差的在前
    )
    
    # 保存每个场景的最佳样本的可视化（延迟保存，避免影响批处理速度）
    if best_samples_sorted:
        threshold_label = "DoLP PSNR" if mode == "dolp" else ("AoLP PSNR" if mode == "aolp" else "PSNR")
        print(f"\n正在保存 {len(best_samples_sorted)} 个场景的最佳样本可视化 ({threshold_label} >= {dolp_psnr_threshold} dB)...", end="", flush=True)
        saved_samples_list = []
        
        for rank, (dolp_psnr, scene_id, base_name, metrics) in enumerate(best_samples_sorted, 1):
            output_path = output_dir_path / f"best_scene{scene_id}_{base_name}_dolp{dolp_psnr:.1f}dB.png"
            
            # 手动加载数据并转换格式，然后保存可视化
            try:
                if use_pt_data:
                    pt_path = polar_root_path / scene_id / f"{base_name}.pt"
                    if not pt_path.exists():
                        pt_path = polar_root_path / "train" / scene_id / f"{base_name}.pt"
                    if not pt_path.exists():
                        pt_path = polar_root_path / "val" / scene_id / f"{base_name}.pt"
                    
                    if pt_path.exists():
                        pixel_values_tensor = torch.load(pt_path, map_location='cpu', weights_only=False)
                        if pixel_values_tensor.dtype != torch.float32:
                            pixel_values_tensor = pixel_values_tensor.float()
                        pixel_values_tensor = convert_data_format(pixel_values_tensor, mode)
                        pixel_values = pixel_values_tensor.unsqueeze(0).to(device)
                    else:
                        continue
                else:
                    scene_dir = polar_root_path / scene_id
                    polar_paths = {
                        "I_0": scene_dir / f"{base_name}_000.png",
                        "I_45": scene_dir / f"{base_name}_045.png",
                        "I_90": scene_dir / f"{base_name}_090.png",
                        "I_135": scene_dir / f"{base_name}_135.png",
                    }
                    
                    missing = [name for name, path in polar_paths.items() if not path.exists()]
                    if missing:
                        polar_paths = {
                            "I_0": scene_dir / f"{base_name}_0.png",
                            "I_45": scene_dir / f"{base_name}_45.png",
                            "I_90": scene_dir / f"{base_name}_90.png",
                            "I_135": scene_dir / f"{base_name}_135.png",
                        }
                    
                    if not all(p.exists() for p in polar_paths.values()):
                        continue
                    
                    physics_img = process_polar_images(polar_paths=polar_paths)
                    config = model.config
                    num_channels = getattr(config, 'num_channels', 3)
                    pixel_values = preprocess_image(physics_img, image_size=224, num_channels=num_channels)
                    pixel_values_tensor = pixel_values.squeeze(0)
                    pixel_values_tensor = convert_data_format(pixel_values_tensor, mode)
                    pixel_values = pixel_values_tensor.unsqueeze(0).to(device)
                
                # 保存可视化
                with suppress_stdout():
                    visualize_mae_reconstruction(
                        model,
                        pixel_values,
                        output_path=str(output_path)
                    )
            except Exception as e:
                print(f"  ⚠ 保存可视化失败 (scene_id={scene_id}, base_name={base_name}): {e}")
                continue
            
            # 计算 AoLP 的 PSNR（根据模式）
            if mode == "dolp":
                # DoLP 模式：所有通道都是 DoLP，AoLP PSNR 无意义，设为 DoLP PSNR
                aolp_psnr_approx = dolp_psnr
                aolp_psnr_sin = dolp_psnr
                aolp_psnr_cos = dolp_psnr
            elif mode == "aolp":
                # AoLP 模式：第0通道是 sin，第1通道是 cos
                # compute_metrics 的命名是 [dolp, sin, cos]，但实际数据是 [sin, cos, 0]
                aolp_psnr_sin = metrics.get("psnr_masked_dolp", 0)  # 实际是 sin
                aolp_psnr_cos = metrics.get("psnr_masked_sin_2aolp", 0)  # 实际是 cos
                aolp_psnr_approx = (aolp_psnr_sin + aolp_psnr_cos) / 2
            else:
                # 标准模式：标准3通道 [DoLP, sin, cos]
                aolp_psnr_sin = metrics.get("psnr_masked_sin_2aolp", 0)
                aolp_psnr_cos = metrics.get("psnr_masked_cos_2aolp", 0)
                aolp_psnr_approx = (aolp_psnr_sin + aolp_psnr_cos) / 2
            
            saved_samples_list.append({
                "rank": rank,
                "scene_id": scene_id,
                "base_name": base_name,
                "dolp_psnr": dolp_psnr,
                "output_path": output_path.name,
                # MSE 指标（Masked 区域）
                # MSE 指标（Masked 区域，根据模式）
                "mse_masked_dolp": metrics.get("mse_masked_dolp", 0) if mode != "aolp" else 0,
                "mse_masked_sin_2aolp": metrics.get("mse_masked_sin_2aolp", 0) if mode != "dolp" else 0,
                "mse_masked_cos_2aolp": metrics.get("mse_masked_cos_2aolp", 0) if mode != "dolp" else 0,
                "mse_masked_mean": metrics.get("mse_masked_mean", 0),
                # PSNR 指标（Masked 区域，根据模式）
                "psnr_masked_dolp": dolp_psnr if mode != "aolp" else 0,
                "psnr_masked_sin_2aolp": aolp_psnr_sin if mode != "dolp" else 0,
                "psnr_masked_cos_2aolp": aolp_psnr_cos if mode != "dolp" else 0,
                "psnr_masked_aolp_approx": aolp_psnr_approx,  # AoLP 的近似 PSNR
                "psnr_masked_mean": metrics.get("psnr_masked_mean", 0),
                # 交叉重建的PSNR（专用编码器重建其他通道的效果）
                "psnr_masked_dolp_cross": metrics.get("psnr_masked_dolp_cross", 0) if mode == "aolp" else 0,
                "psnr_masked_aolp_cross": metrics.get("psnr_masked_aolp_cross", 0) if mode == "dolp" else 0,
                "psnr_masked_sin_2aolp_cross": metrics.get("psnr_masked_sin_2aolp_cross", 0) if mode == "dolp" else 0,
                "psnr_masked_cos_2aolp_cross": metrics.get("psnr_masked_cos_2aolp_cross", 0) if mode == "dolp" else 0,
                # 全图指标（可选，用于对比）
                "mse_dolp": metrics.get("mse_dolp", 0),
                "mse_sin_2aolp": metrics.get("mse_sin_2aolp", 0),
                "mse_cos_2aolp": metrics.get("mse_cos_2aolp", 0),
                "mse_mean": metrics.get("mse_mean", 0),
                "psnr_dolp": metrics.get("psnr_dolp", 0),
                "psnr_sin_2aolp": metrics.get("psnr_sin_2aolp", 0),
                "psnr_cos_2aolp": metrics.get("psnr_cos_2aolp", 0),
                "psnr_mean": metrics.get("psnr_mean", 0),
            })
        
        print(f" ✓ (已保存 {len(best_samples_sorted)} 个场景的最佳样本)")
        
        # 保存到主结果中（用于兼容）
        all_results["high_quality_samples"] = saved_samples_list
        
        # 兼容旧格式：best_sample 指向最高分样本
        if saved_samples_list:
            all_results["best_sample"] = {
                "scene_id": saved_samples_list[0]["scene_id"],
                "base_name": saved_samples_list[0]["base_name"],
                "dolp_psnr": saved_samples_list[0]["dolp_psnr"],
                "output_path": saved_samples_list[0]["output_path"],
            }
        
        # 创建专门的 JSON 文件记录这些样本的详细指标
        detailed_json_path = output_dir_path / "high_quality_samples_details.json"
        detailed_data = {
            "dolp_psnr_threshold": dolp_psnr_threshold,
            "total_scenes": len(best_samples_sorted),
            "description": "每个场景 DoLP PSNR >= 30 dB 的最佳样本的详细指标",
            "samples": saved_samples_list,
        }
        
        with open(detailed_json_path, "w", encoding="utf-8") as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 详细指标已保存: {detailed_json_path}")
        
        # 打印统计信息（根据模式）
        if mode == "dolp":
            print(f"\n✓ 高质量样本统计 (DoLP PSNR >= {dolp_psnr_threshold} dB):")
            print(f"  - 符合条件的场景数: {len(best_samples_sorted)}")
            print(f"  - 原始高质量样本数: {len(high_quality_samples)}")
            if saved_samples_list:
                max_psnr = saved_samples_list[0]["dolp_psnr"]
                min_psnr = saved_samples_list[-1]["dolp_psnr"]
                avg_psnr = np.mean([s["dolp_psnr"] for s in saved_samples_list])
                avg_aolp_cross = np.mean([s.get("psnr_masked_aolp_cross", 0) for s in saved_samples_list])
                avg_sin_cross = np.mean([s.get("psnr_masked_sin_2aolp_cross", 0) for s in saved_samples_list])
                avg_cos_cross = np.mean([s.get("psnr_masked_cos_2aolp_cross", 0) for s in saved_samples_list])
                print(f"  - 最高 DoLP PSNR: {max_psnr:.2f} dB")
                print(f"  - 最低 DoLP PSNR: {min_psnr:.2f} dB")
                print(f"  - 平均 DoLP PSNR: {avg_psnr:.2f} dB")
                print(f"  - 平均 AoLP 交叉重建 PSNR: {avg_aolp_cross:.2f} dB (用DoLP编码器重建AoLP)")
                print(f"  - 平均 sin(2*AoLP) 交叉重建 PSNR: {avg_sin_cross:.2f} dB")
                print(f"  - 平均 cos(2*AoLP) 交叉重建 PSNR: {avg_cos_cross:.2f} dB")
                print(f"\n  所有高质量样本详情:")
                for item in saved_samples_list:
                    aolp_cross = item.get("psnr_masked_aolp_cross", 0)
                    sin_cross = item.get("psnr_masked_sin_2aolp_cross", 0)
                    cos_cross = item.get("psnr_masked_cos_2aolp_cross", 0)
                    print(f"    #{item['rank']:2d}: scene_id={item['scene_id']}, base_name={item['base_name']}, "
                          f"DoLP PSNR={item['dolp_psnr']:.2f} dB | "
                          f"AoLP交叉={aolp_cross:.2f} dB (sin={sin_cross:.2f}, cos={cos_cross:.2f})")
        elif mode == "aolp":
            print(f"\n✓ 高质量样本统计 (AoLP PSNR >= {dolp_psnr_threshold} dB):")
            print(f"  - 符合条件的场景数: {len(best_samples_sorted)}")
            print(f"  - 原始高质量样本数: {len(high_quality_samples)}")
            if saved_samples_list:
                max_psnr = saved_samples_list[0]["psnr_masked_aolp_approx"]
                min_psnr = saved_samples_list[-1]["psnr_masked_aolp_approx"]
                avg_psnr = np.mean([s["psnr_masked_aolp_approx"] for s in saved_samples_list])
                avg_sin_psnr = np.mean([s["psnr_masked_sin_2aolp"] for s in saved_samples_list])
                avg_cos_psnr = np.mean([s["psnr_masked_cos_2aolp"] for s in saved_samples_list])
                avg_dolp_cross = np.mean([s.get("psnr_masked_dolp_cross", 0) for s in saved_samples_list])
                print(f"  - 最高 AoLP PSNR (近似): {max_psnr:.2f} dB")
                print(f"  - 最低 AoLP PSNR (近似): {min_psnr:.2f} dB")
                print(f"  - 平均 AoLP PSNR (近似): {avg_psnr:.2f} dB")
                print(f"  - 平均 sin(2*AoLP) PSNR: {avg_sin_psnr:.2f} dB")
                print(f"  - 平均 cos(2*AoLP) PSNR: {avg_cos_psnr:.2f} dB")
                print(f"  - 平均 DoLP 交叉重建 PSNR: {avg_dolp_cross:.2f} dB (用AoLP编码器重建DoLP)")
                print(f"\n  所有高质量样本详情:")
                for item in saved_samples_list:
                    dolp_cross = item.get("psnr_masked_dolp_cross", 0)
                    print(f"    #{item['rank']:2d}: scene_id={item['scene_id']}, base_name={item['base_name']}, "
                          f"AoLP PSNR≈{item['psnr_masked_aolp_approx']:.2f} dB "
                          f"(sin={item['psnr_masked_sin_2aolp']:.2f} dB, cos={item['psnr_masked_cos_2aolp']:.2f} dB) | "
                          f"DoLP交叉={dolp_cross:.2f} dB")
        else:
            print(f"\n✓ 高质量样本统计 (DoLP PSNR >= {dolp_psnr_threshold} dB):")
            print(f"  - 符合条件的场景数: {len(best_samples_sorted)}")
            print(f"  - 原始高质量样本数: {len(high_quality_samples)}")
            if saved_samples_list:
                max_psnr = saved_samples_list[0]["dolp_psnr"]
                min_psnr = saved_samples_list[-1]["dolp_psnr"]
                avg_psnr = np.mean([s["dolp_psnr"] for s in saved_samples_list])
                avg_aolp_psnr = np.mean([s["psnr_masked_aolp_approx"] for s in saved_samples_list])
                print(f"  - 最高 DoLP PSNR: {max_psnr:.2f} dB")
                print(f"  - 最低 DoLP PSNR: {min_psnr:.2f} dB")
                print(f"  - 平均 DoLP PSNR: {avg_psnr:.2f} dB")
                print(f"  - 平均 AoLP PSNR (近似): {avg_aolp_psnr:.2f} dB")
                print(f"\n  所有高质量样本详情:")
                for item in saved_samples_list:
                    print(f"    #{item['rank']:2d}: scene_id={item['scene_id']}, base_name={item['base_name']}, "
                          f"DoLP PSNR={item['dolp_psnr']:.2f} dB, AoLP PSNR≈{item['psnr_masked_aolp_approx']:.2f} dB")
    else:
        threshold_label = "DoLP PSNR" if mode == "dolp" else ("AoLP PSNR" if mode == "aolp" else "PSNR")
        print(f"\n⚠ 未找到 {threshold_label} >= {dolp_psnr_threshold} dB 的样本")
    
    # 保存后20名样本的可视化
    if worst_samples_sorted:
        print(f"\n正在保存 {len(worst_samples_sorted)} 个场景的最差样本可视化 (后20名)...", end="", flush=True)
        saved_worst_samples_list = []
        
        for rank, (psnr, scene_id, base_name, metrics) in enumerate(worst_samples_sorted, 1):
            output_path = output_dir_path / f"worst_scene{scene_id}_{base_name}_psnr{psnr:.1f}dB.png"
            
            # 手动加载数据并转换格式，然后保存可视化
            try:
                if use_pt_data:
                    pt_path = polar_root_path / scene_id / f"{base_name}.pt"
                    if not pt_path.exists():
                        pt_path = polar_root_path / "train" / scene_id / f"{base_name}.pt"
                    if not pt_path.exists():
                        pt_path = polar_root_path / "val" / scene_id / f"{base_name}.pt"
                    
                    if pt_path.exists():
                        pixel_values_tensor = torch.load(pt_path, map_location='cpu', weights_only=False)
                        if pixel_values_tensor.dtype != torch.float32:
                            pixel_values_tensor = pixel_values_tensor.float()
                        pixel_values_tensor = convert_data_format(pixel_values_tensor, mode)
                        pixel_values = pixel_values_tensor.unsqueeze(0).to(device)
                    else:
                        continue
                else:
                    scene_dir = polar_root_path / scene_id
                    polar_paths = {
                        "I_0": scene_dir / f"{base_name}_000.png",
                        "I_45": scene_dir / f"{base_name}_045.png",
                        "I_90": scene_dir / f"{base_name}_090.png",
                        "I_135": scene_dir / f"{base_name}_135.png",
                    }
                    
                    missing = [name for name, path in polar_paths.items() if not path.exists()]
                    if missing:
                        polar_paths = {
                            "I_0": scene_dir / f"{base_name}_0.png",
                            "I_45": scene_dir / f"{base_name}_45.png",
                            "I_90": scene_dir / f"{base_name}_90.png",
                            "I_135": scene_dir / f"{base_name}_135.png",
                        }
                    
                    if not all(p.exists() for p in polar_paths.values()):
                        continue
                    
                    physics_img = process_polar_images(polar_paths=polar_paths)
                    config = model.config
                    num_channels = getattr(config, 'num_channels', 3)
                    pixel_values = preprocess_image(physics_img, image_size=224, num_channels=num_channels)
                    pixel_values_tensor = pixel_values.squeeze(0)
                    pixel_values_tensor = convert_data_format(pixel_values_tensor, mode)
                    pixel_values = pixel_values_tensor.unsqueeze(0).to(device)
                
                # 保存可视化
                with suppress_stdout():
                    visualize_mae_reconstruction(
                        model,
                        pixel_values,
                        output_path=str(output_path)
                    )
            except Exception as e:
                print(f"  ⚠ 保存可视化失败 (scene_id={scene_id}, base_name={base_name}): {e}")
                continue
            
            # 计算交叉重建的PSNR（根据模式）
            if mode == "dolp":
                aolp_psnr_cross = metrics.get("psnr_masked_aolp_cross", 0)
                sin_psnr_cross = metrics.get("psnr_masked_sin_2aolp_cross", 0)
                cos_psnr_cross = metrics.get("psnr_masked_cos_2aolp_cross", 0)
                saved_worst_samples_list.append({
                    "rank": rank,
                    "scene_id": scene_id,
                    "base_name": base_name,
                    "psnr": psnr,
                    "psnr_masked_dolp": metrics.get("psnr_masked_dolp", 0),
                    "psnr_masked_aolp_cross": aolp_psnr_cross,
                    "psnr_masked_sin_2aolp_cross": sin_psnr_cross,
                    "psnr_masked_cos_2aolp_cross": cos_psnr_cross,
                    "output_path": output_path.name,
                })
            elif mode == "aolp":
                dolp_psnr_cross = metrics.get("psnr_masked_dolp_cross", 0)
                sin_psnr = metrics.get("psnr_masked_dolp", 0)  # 实际是 sin
                cos_psnr = metrics.get("psnr_masked_sin_2aolp", 0)  # 实际是 cos
                aolp_psnr = (sin_psnr + cos_psnr) / 2
                saved_worst_samples_list.append({
                    "rank": rank,
                    "scene_id": scene_id,
                    "base_name": base_name,
                    "psnr": psnr,
                    "psnr_masked_aolp": aolp_psnr,
                    "psnr_masked_sin_2aolp": sin_psnr,
                    "psnr_masked_cos_2aolp": cos_psnr,
                    "psnr_masked_dolp_cross": dolp_psnr_cross,
                    "output_path": output_path.name,
                })
            else:
                saved_worst_samples_list.append({
                    "rank": rank,
                    "scene_id": scene_id,
                    "base_name": base_name,
                    "psnr": psnr,
                    "psnr_masked_dolp": metrics.get("psnr_masked_dolp", 0),
                    "output_path": output_path.name,
                })
        
        print(f" ✓ (已保存 {len(worst_samples_sorted)} 个场景的最差样本)")
        
        # 保存后20名样本的详细信息到JSON
        worst_json_path = output_dir_path / "worst_samples_details.json"
        worst_data = {
            "description": "每个场景 PSNR 后20名的最差样本的详细指标",
            "total_scenes": len(worst_samples_sorted),
            "samples": saved_worst_samples_list,
        }
        
        with open(worst_json_path, "w", encoding="utf-8") as f:
            json.dump(worst_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 后20名样本详细指标已保存: {worst_json_path}")
        
        # 打印后20名样本的详细信息（根据模式）
        if mode == "dolp":
            print(f"\n✓ 后20名样本统计 (DoLP PSNR 最低):")
            if saved_worst_samples_list:
                min_psnr = saved_worst_samples_list[0]["psnr"]
                max_psnr = saved_worst_samples_list[-1]["psnr"]
                avg_psnr = np.mean([s["psnr"] for s in saved_worst_samples_list])
                avg_aolp_cross = np.mean([s.get("psnr_masked_aolp_cross", 0) for s in saved_worst_samples_list])
                print(f"  - 最低 DoLP PSNR: {min_psnr:.2f} dB")
                print(f"  - 最高 DoLP PSNR (后20名中): {max_psnr:.2f} dB")
                print(f"  - 平均 DoLP PSNR: {avg_psnr:.2f} dB")
                print(f"  - 平均 AoLP 交叉重建 PSNR: {avg_aolp_cross:.2f} dB (用DoLP编码器重建AoLP)")
                print(f"\n  所有后20名样本详情:")
                for item in saved_worst_samples_list:
                    aolp_cross = item.get("psnr_masked_aolp_cross", 0)
                    sin_cross = item.get("psnr_masked_sin_2aolp_cross", 0)
                    cos_cross = item.get("psnr_masked_cos_2aolp_cross", 0)
                    print(f"    #{item['rank']:2d}: scene_id={item['scene_id']}, base_name={item['base_name']}, "
                          f"DoLP PSNR={item['psnr']:.2f} dB | "
                          f"AoLP交叉={aolp_cross:.2f} dB (sin={sin_cross:.2f}, cos={cos_cross:.2f})")
        elif mode == "aolp":
            print(f"\n✓ 后20名样本统计 (AoLP PSNR 最低):")
            if saved_worst_samples_list:
                min_psnr = saved_worst_samples_list[0]["psnr"]
                max_psnr = saved_worst_samples_list[-1]["psnr"]
                avg_psnr = np.mean([s["psnr"] for s in saved_worst_samples_list])
                avg_dolp_cross = np.mean([s.get("psnr_masked_dolp_cross", 0) for s in saved_worst_samples_list])
                print(f"  - 最低 AoLP PSNR: {min_psnr:.2f} dB")
                print(f"  - 最高 AoLP PSNR (后20名中): {max_psnr:.2f} dB")
                print(f"  - 平均 AoLP PSNR: {avg_psnr:.2f} dB")
                print(f"  - 平均 DoLP 交叉重建 PSNR: {avg_dolp_cross:.2f} dB (用AoLP编码器重建DoLP)")
                print(f"\n  所有后20名样本详情:")
                for item in saved_worst_samples_list:
                    dolp_cross = item.get("psnr_masked_dolp_cross", 0)
                    aolp_psnr = item.get("psnr_masked_aolp", 0)
                    print(f"    #{item['rank']:2d}: scene_id={item['scene_id']}, base_name={item['base_name']}, "
                          f"AoLP PSNR={aolp_psnr:.2f} dB | "
                          f"DoLP交叉={dolp_cross:.2f} dB")
        else:
            print(f"\n✓ 后20名样本统计 (DoLP PSNR 最低):")
            if saved_worst_samples_list:
                min_psnr = saved_worst_samples_list[0]["psnr"]
                max_psnr = saved_worst_samples_list[-1]["psnr"]
                avg_psnr = np.mean([s["psnr"] for s in saved_worst_samples_list])
                print(f"  - 最低 DoLP PSNR: {min_psnr:.2f} dB")
                print(f"  - 最高 DoLP PSNR (后20名中): {max_psnr:.2f} dB")
                print(f"  - 平均 DoLP PSNR: {avg_psnr:.2f} dB")
                print(f"\n  所有后20名样本详情:")
                for item in saved_worst_samples_list:
                    print(f"    #{item['rank']:2d}: scene_id={item['scene_id']}, base_name={item['base_name']}, "
                          f"DoLP PSNR={item['psnr']:.2f} dB")
    
    # 保存 JSON 结果
    json_path = output_dir_path / "batch_validation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 统计结果已保存: {json_path}")
    
    # 打印总体统计（根据模式显示相关指标）
    valid_scenes = [s for s in all_results["scenes"].values() if s.get("num_samples", 0) > 0]
    if valid_scenes:
        total_samples = sum(s["num_samples"] for s in valid_scenes)
        print(f"\n总体统计:")
        print(f"  - 有效场景数: {len(valid_scenes)}")
        print(f"  - 总样本数: {total_samples}")
        
        if mode == "dolp":
            avg_psnr_dolp = np.mean([s.get("psnr_masked_dolp", 0) for s in valid_scenes])
            print(f"  - 平均 DoLP PSNR: {avg_psnr_dolp:.2f} dB")
        elif mode == "aolp":
            avg_psnr_sin = np.mean([s.get("psnr_masked_sin_2aolp", 0) for s in valid_scenes])
            avg_psnr_cos = np.mean([s.get("psnr_masked_cos_2aolp", 0) for s in valid_scenes])
            print(f"  - 平均 sin(2*AoLP) PSNR: {avg_psnr_sin:.2f} dB")
            print(f"  - 平均 cos(2*AoLP) PSNR: {avg_psnr_cos:.2f} dB")
        else:
            avg_psnr_dolp = np.mean([s.get("psnr_masked_dolp", 0) for s in valid_scenes])
            avg_psnr_sin = np.mean([s.get("psnr_masked_sin_2aolp", 0) for s in valid_scenes])
            avg_psnr_cos = np.mean([s.get("psnr_masked_cos_2aolp", 0) for s in valid_scenes])
            print(f"  - 平均 DoLP PSNR: {avg_psnr_dolp:.2f} dB")
            print(f"  - 平均 sin(2*AoLP) PSNR: {avg_psnr_sin:.2f} dB")
            print(f"  - 平均 cos(2*AoLP) PSNR: {avg_psnr_cos:.2f} dB")
    
    return all_results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 1 批量验证脚本")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Stage 1 检查点路径"
    )
    parser.add_argument(
        "--polar_root",
        type=str,
        default="/openbayes/input/input0/polar",
        help="偏振图像根目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./stage1_batch_validation",
        help="输出目录"
    )
    parser.add_argument(
        "--use_pt_data",
        action="store_true",
        default=False,
        help="使用预处理的 .pt 文件"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token（可选）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="批处理大小（GPU模式，默认8，CPU模式自动设为1）"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=["dolp", "aolp", "standard"],
        help="数据格式模式：'dolp' (DoLP专用编码器), 'aolp' (AoLP专用编码器), 'standard' (标准3通道)"
    )
    parser.add_argument(
        "--verify_single_image",
        action="store_true",
        default=False,
        help="验证单张图片模式（用于验证 overfit 训练的图片）"
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="单张图片验证：场景ID（如果提供，将只验证这一张图片）"
    )
    parser.add_argument(
        "--base_name",
        type=str,
        default=None,
        help="单张图片验证：图片基础名称（如果提供，将只验证这一张图片）"
    )
    parser.add_argument(
        "--checkpoint_info",
        type=str,
        default=None,
        help="从checkpoint目录读取overfit_image_info.json，自动获取scene_id和base_name"
    )
    parser.add_argument(
        "--num_eval_runs",
        type=int,
        default=10,
        help="单张图片验证时，进行多次验证取平均的次数（默认10次，减少随机mask的影响）"
    )
    
    args = parser.parse_args()
    
    # 如果指定了checkpoint_info，尝试读取overfit图片信息
    if args.checkpoint_info or args.verify_single_image:
        if args.checkpoint_info:
            info_path = Path(args.checkpoint_info) / "overfit_image_info.json"
            if info_path.exists():
                with open(info_path, 'r', encoding='utf-8') as f:
                    overfit_info = json.load(f)
                if not args.scene_id:
                    args.scene_id = overfit_info.get("scene_id")
                if not args.base_name:
                    args.base_name = overfit_info.get("base_name")
                print(f"✓ 从 {info_path} 读取 overfit 图片信息:")
                print(f"  - scene_id: {args.scene_id}")
                print(f"  - base_name: {args.base_name}")
            else:
                print(f"⚠ 警告: 未找到 {info_path}，将使用命令行参数")
        
        # 如果提供了scene_id和base_name，验证单张图片
        if args.scene_id and args.base_name:
            verify_single_image_detailed(
                checkpoint_path=args.checkpoint,
                polar_root=args.polar_root,
                scene_id=args.scene_id,
                base_name=args.base_name,
                output_dir=args.output_dir,
                use_pt_data=args.use_pt_data,
                hf_token=args.hf_token,
                mode=args.mode,
                num_eval_runs=args.num_eval_runs,
            )
        else:
            print("⚠ 警告: 单张图片验证模式需要提供 scene_id 和 base_name")
            print("  使用批量验证模式...")
            verify_all_scenes(
                checkpoint_path=args.checkpoint,
                polar_root=args.polar_root,
                output_dir=args.output_dir,
                use_pt_data=args.use_pt_data,
                hf_token=args.hf_token,
                batch_size=args.batch_size,
                mode=args.mode,
            )
    else:
        verify_all_scenes(
            checkpoint_path=args.checkpoint,
            polar_root=args.polar_root,
            output_dir=args.output_dir,
            use_pt_data=args.use_pt_data,
            hf_token=args.hf_token,
            batch_size=args.batch_size,
            mode=args.mode,
        )


if __name__ == "__main__":
    main()


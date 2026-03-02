"""
Zero-shot VAE 验证脚本：评估预训练的 Stable Diffusion VAE 在偏振图像上的重建能力

功能：
1. 加载预训练的 VAE (stabilityai/sd-vae-ft-mse)
2. 对偏振图像进行编码-解码（不进行任何微调）
3. 计算重建质量指标（PSNR/MSE）
4. 可视化原始图像与重建图像的对比

使用方法：
    python verify_vae_polar.py \
        --polar_root /path/to/polar/data \
        --scene_id 01 \
        --output_dir ./vae_validation \
        --use_pt_data
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import AutoencoderKL
from matplotlib.colors import hsv_to_rgb
from PIL import Image
from tqdm import tqdm

from dataset_common import process_polar_images
from verify_stage1 import compute_metrics, list_base_names

# Matplotlib 配置
plt.rcParams["axes.unicode_minus"] = False


def load_vae_model(model_id: str = "stabilityai/sd-vae-ft-mse", device: torch.device = None, cache_dir: Optional[str] = None):
    """
    加载预训练的 VAE 模型
    
    Args:
        model_id: Hugging Face 模型 ID 或本地路径
        device: 设备（如果为 None，自动选择）
        cache_dir: 模型缓存目录（如果提供，模型会下载/加载到此目录）
    
    Returns:
        加载的 VAE 模型
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("加载 VAE 模型")
    print("=" * 80)
    print(f"模型 ID: {model_id}")
    if cache_dir:
        print(f"模型缓存目录: {cache_dir}")
    print(f"使用设备: {device}")
    
    # 加载 VAE
    load_kwargs = {}
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir
    
    vae = AutoencoderKL.from_pretrained(model_id, **load_kwargs)
    vae = vae.to(device)
    vae.eval()
    
    print(f"✓ VAE 模型加载成功")
    print(f"  - 输入尺寸: 512x512 (VAE 会自动 resize)")
    print(f"  - 输入范围: [-1, 1]")
    print(f"  - 输出范围: [-1, 1]")
    
    return vae


def preprocess_for_vae(image: np.ndarray, target_size: int = 512) -> torch.Tensor:
    """
    预处理图像以供 VAE 使用
    
    Args:
        image: 输入图像 (C, H, W)，值范围 [0, 1]
        target_size: 目标尺寸（VAE 通常需要 512x512）
    
    Returns:
        预处理后的张量 (1, C, H, W)，值范围 [-1, 1]
    """
    # 转换为 torch tensor
    if isinstance(image, np.ndarray):
        image_tensor = torch.from_numpy(image).float()
    else:
        image_tensor = image.float()
    
    # 添加 batch 维度
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W)
    
    # Resize 到目标尺寸（使用双线性插值）
    if image_tensor.shape[2] != target_size or image_tensor.shape[3] != target_size:
        image_tensor = torch.nn.functional.interpolate(
            image_tensor,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )
    
    # 转换范围：[0, 1] -> [-1, 1]
    image_tensor = image_tensor * 2.0 - 1.0
    
    return image_tensor


def postprocess_from_vae(image_tensor: torch.Tensor) -> np.ndarray:
    """
    后处理 VAE 输出
    
    Args:
        image_tensor: VAE 输出 (B, C, H, W)，值范围 [-1, 1]
    
    Returns:
        后处理后的图像 (C, H, W)，值范围 [0, 1]
    """
    # 转换范围：[-1, 1] -> [0, 1]
    image = (image_tensor + 1.0) / 2.0
    
    # Clip 到 [0, 1]
    image = torch.clamp(image, 0.0, 1.0)
    
    # 转换为 numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # 移除 batch 维度
    if len(image.shape) == 4:
        image = image[0]  # (C, H, W)
    
    return image


def compute_aolp_hsv(sin_ch: np.ndarray, cos_ch: np.ndarray) -> np.ndarray:
    """
    从 sin(2*AoLP) 和 cos(2*AoLP) 计算 AoLP 的 HSV 表示
    
    Args:
        sin_ch: sin(2*AoLP) 通道
        cos_ch: cos(2*AoLP) 通道
    
    Returns:
        RGB 图像（HSV 转换后）
    """
    # 计算 AoLP 角度
    aolp_raw = 0.5 * np.arctan2(sin_ch, cos_ch)  # [-π/2, π/2]
    
    # 将负角度转换到 [0, π] 范围
    aolp_angle = np.where(aolp_raw < 0, aolp_raw + np.pi, aolp_raw)  # [0, π]
    
    # 将角度归一化到 [0, 1] 用于 HSV 的 H 通道
    h = aolp_angle / np.pi  # H: 色调（角度），范围 [0, 1]
    s = np.ones_like(h)  # S: 饱和度（设为1）
    v = np.ones_like(h)  # V: 明度（设为1）
    
    # 转换为 RGB 用于显示
    hsv_img = np.stack([h, s, v], axis=-1)
    rgb_img = hsv_to_rgb(hsv_img)
    return rgb_img


def visualize_reconstruction(
    original_img: np.ndarray,
    reconstructed_img: np.ndarray,
    metrics: Dict[str, float],
    output_path: str,
    num_channels: int = 3,
):
    """
    可视化原始图像与重建图像的对比
    
    Args:
        original_img: 原始图像 (C, H, W)，值范围 [0, 1]
        reconstructed_img: 重建图像 (C, H, W)，值范围 [0, 1]
        metrics: 指标字典
        output_path: 输出路径
        num_channels: 通道数
    """
    # 确定通道索引
    if num_channels == 3:
        dolp_idx = 0
        sin_idx = 1
        cos_idx = 2
        channel_names = ['DoLP', 'sin(2*AoLP)', 'cos(2*AoLP)']
    else:
        intensity_idx = 0
        dolp_idx = 1
        sin_idx = 2
        cos_idx = 3
        channel_names = ['Intensity', 'DoLP', 'sin(2*AoLP)', 'cos(2*AoLP)']
    
    # 创建图像
    fig, axes = plt.subplots(num_channels, 2, figsize=(12, 6 * num_channels))
    if num_channels == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(
        f'VAE Zero-shot Reconstruction Comparison\n'
        f'PSNR: DoLP={metrics.get("psnr_dolp", 0):.2f} dB, '
        f'sin={metrics.get("psnr_sin_2aolp", 0):.2f} dB, '
        f'cos={metrics.get("psnr_cos_2aolp", 0):.2f} dB, '
        f'Mean={metrics.get("psnr_mean", 0):.2f} dB',
        fontsize=14,
        fontweight='bold'
    )
    
    for ch_idx in range(num_channels):
        # 原始图像
        if ch_idx == dolp_idx:
            # DoLP: 使用 jet 色图
            im_orig = axes[ch_idx, 0].imshow(original_img[ch_idx], cmap='jet', vmin=0, vmax=1)
            axes[ch_idx, 0].set_title(f'Original - {channel_names[ch_idx]}', fontsize=12)
            axes[ch_idx, 0].axis('off')
            plt.colorbar(im_orig, ax=axes[ch_idx, 0], fraction=0.046)
        elif ch_idx == sin_idx or ch_idx == cos_idx:
            # AoLP: 使用 HSV 可视化
            sin_ch_orig = original_img[sin_idx]
            cos_ch_orig = original_img[cos_idx]
            rgb_orig = compute_aolp_hsv(sin_ch_orig, cos_ch_orig)
            im_orig = axes[ch_idx, 0].imshow(rgb_orig)
            axes[ch_idx, 0].set_title(f'Original - AoLP (from sin/cos, HSV)', fontsize=12)
            axes[ch_idx, 0].axis('off')
        else:
            # Intensity: 使用灰度图
            im_orig = axes[ch_idx, 0].imshow(original_img[ch_idx], cmap='gray', vmin=0, vmax=1)
            axes[ch_idx, 0].set_title(f'Original - {channel_names[ch_idx]}', fontsize=12)
            axes[ch_idx, 0].axis('off')
            plt.colorbar(im_orig, ax=axes[ch_idx, 0], fraction=0.046)
        
        # 重建图像
        if ch_idx == dolp_idx:
            # DoLP: 使用 jet 色图
            im_recon = axes[ch_idx, 1].imshow(reconstructed_img[ch_idx], cmap='jet', vmin=0, vmax=1)
            psnr_val = metrics.get("psnr_dolp", 0)
            axes[ch_idx, 1].set_title(f'Reconstructed - {channel_names[ch_idx]}\nPSNR: {psnr_val:.2f} dB', fontsize=12)
            axes[ch_idx, 1].axis('off')
            plt.colorbar(im_recon, ax=axes[ch_idx, 1], fraction=0.046)
        elif ch_idx == sin_idx or ch_idx == cos_idx:
            # AoLP: 使用 HSV 可视化
            sin_ch_recon = reconstructed_img[sin_idx]
            cos_ch_recon = reconstructed_img[cos_idx]
            rgb_recon = compute_aolp_hsv(sin_ch_recon, cos_ch_recon)
            im_recon = axes[ch_idx, 1].imshow(rgb_recon)
            if ch_idx == sin_idx:
                psnr_val = metrics.get("psnr_sin_2aolp", 0)
            else:
                psnr_val = metrics.get("psnr_cos_2aolp", 0)
            axes[ch_idx, 1].set_title(f'Reconstructed - AoLP (from sin/cos, HSV)\nPSNR: {psnr_val:.2f} dB', fontsize=12)
            axes[ch_idx, 1].axis('off')
        else:
            # Intensity: 使用灰度图
            im_recon = axes[ch_idx, 1].imshow(reconstructed_img[ch_idx], cmap='gray', vmin=0, vmax=1)
            psnr_val = metrics.get("psnr_intensity", 0) if num_channels == 4 else 0
            axes[ch_idx, 1].set_title(f'Reconstructed - {channel_names[ch_idx]}\nPSNR: {psnr_val:.2f} dB', fontsize=12)
            axes[ch_idx, 1].axis('off')
            plt.colorbar(im_recon, ax=axes[ch_idx, 1], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 可视化结果已保存到: {output_path}")


def process_single_image(
    vae: AutoencoderKL,
    device: torch.device,
    image_data: np.ndarray,
    original_size: Optional[tuple] = None,
) -> tuple:
    """
    处理单张图像：编码-解码
    
    Args:
        vae: VAE 模型
        device: 设备
        image_data: 输入图像 (C, H, W)，值范围 [0, 1]
        original_size: 原始尺寸 (H, W)，如果提供，会将输出 resize 回原始尺寸
    
    Returns:
        (reconstructed_image, metrics): 重建图像和指标
    """
    # 保存原始尺寸
    if original_size is None:
        original_size = (image_data.shape[1], image_data.shape[2])
    
    # 预处理：转换为 VAE 输入格式
    input_tensor = preprocess_for_vae(image_data, target_size=512)  # (1, C, 512, 512), [-1, 1]
    input_tensor = input_tensor.to(device)
    
    # VAE 编码-解码
    with torch.no_grad():
        # 编码
        latent = vae.encode(input_tensor).latent_dist.sample()
        
        # 解码
        output = vae.decode(latent).sample  # (1, C, 512, 512), [-1, 1]
    
    # 后处理：转换回 [0, 1] 范围
    reconstructed = postprocess_from_vae(output)  # (C, 512, 512), [0, 1]
    
    # 如果原始尺寸不是 512x512，需要 resize 回原始尺寸
    if original_size != (512, 512):
        reconstructed_tensor = torch.from_numpy(reconstructed).unsqueeze(0)  # (1, C, 512, 512)
        reconstructed_tensor = torch.nn.functional.interpolate(
            reconstructed_tensor,
            size=original_size,
            mode='bilinear',
            align_corners=False
        )
        reconstructed = reconstructed_tensor[0].cpu().numpy()  # (C, H, W)
    
    # 确保原始图像和重建图像尺寸一致
    if image_data.shape[1:] != reconstructed.shape[1:]:
        # Resize 原始图像到重建图像的尺寸
        image_data_tensor = torch.from_numpy(image_data).unsqueeze(0)
        image_data_tensor = torch.nn.functional.interpolate(
            image_data_tensor,
            size=reconstructed.shape[1:],
            mode='bilinear',
            align_corners=False
        )
        image_data = image_data_tensor[0].cpu().numpy()
    
    # 计算指标
    metrics = compute_metrics(
        image_data,
        reconstructed,
        mask=None,  # VAE 是全图重建，不需要 mask
        patch_size=16,  # 不使用，但需要提供
        num_channels=image_data.shape[0]
    )
    
    return reconstructed, metrics


def verify_vae_on_scene(
    vae: AutoencoderKL,
    device: torch.device,
    polar_root: Path,
    scene_id: str,
    output_dir: Path,
    use_pt_data: bool = False,
    batch_size: int = 1,
    save_all_samples: bool = False,
    all_samples_vis_dir: Optional[Path] = None,
):
    """
    在单个场景上验证 VAE
    
    Args:
        vae: VAE 模型
        device: 设备
        polar_root: 偏振图像根目录（可能是 train 子目录）
        scene_id: 场景ID
        output_dir: 输出目录
        use_pt_data: 是否使用 .pt 文件
        batch_size: 批处理大小（VAE 显存较大，建议使用 1）
        save_all_samples: 是否保存所有样本的可视化
        all_samples_vis_dir: 所有样本可视化的输出目录
    
    Returns:
        (avg_metrics, all_sample_metrics): 平均指标和所有样本的指标列表
    """
    scene_dir = polar_root / scene_id
    
    # 获取所有样本
    if use_pt_data:
        pt_files = sorted(scene_dir.glob("*.pt"))
        base_names = [f.stem for f in pt_files]
    else:
        base_names = list_base_names(scene_dir)
    
    if not base_names:
        print(f"⚠ 场景 {scene_id} 没有可用样本，跳过")
        return None, []
    
    print(f"\n处理场景 {scene_id}，共 {len(base_names)} 个样本")
    
    all_metrics = []
    all_sample_metrics = []  # 保存所有样本的详细指标
    
    # 处理每个样本
    for base_name in tqdm(base_names, desc=f"场景 {scene_id}"):
        try:
            # 加载数据
            if use_pt_data:
                pt_path = scene_dir / f"{base_name}.pt"
                if not pt_path.exists():
                    # 尝试 train/val 子目录
                    pt_path = polar_root / "train" / scene_id / f"{base_name}.pt"
                if not pt_path.exists():
                    pt_path = polar_root / "val" / scene_id / f"{base_name}.pt"
                
                if not pt_path.exists():
                    continue
                
                pixel_values_tensor = torch.load(pt_path, map_location='cpu', weights_only=False)
                if pixel_values_tensor.dtype != torch.float32:
                    pixel_values_tensor = pixel_values_tensor.float()
                
                # 如果是4通道，只取后3个通道
                if pixel_values_tensor.shape[0] == 4:
                    image_data = pixel_values_tensor[1:4, :, :].numpy()  # [DoLP, sin, cos]
                else:
                    image_data = pixel_values_tensor.numpy()  # [DoLP, sin, cos]
            else:
                # 使用 PNG 图像
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
                
                physics_img = process_polar_images(polar_paths=polar_paths)  # (H, W, 4)
                # 转换为 (C, H, W) 格式，只取后3个通道
                image_data = physics_img[:, :, 1:4].transpose(2, 0, 1)  # [DoLP, sin, cos]
            
            # 处理图像
            reconstructed, metrics = process_single_image(
                vae=vae,
                device=device,
                image_data=image_data,
                original_size=(image_data.shape[1], image_data.shape[2])
            )
            
            sample_metric = {
                "scene_id": scene_id,
                "base_name": base_name,
                **metrics
            }
            all_metrics.append(sample_metric)
            all_sample_metrics.append(sample_metric)
            
            # 保存第一个样本的可视化（到原始输出目录）
            if len(all_metrics) == 1:
                output_path = output_dir / f"vae_vis_scene{scene_id}_{base_name}.png"
                visualize_reconstruction(
                    original_img=image_data,
                    reconstructed_img=reconstructed,
                    metrics=metrics,
                    output_path=str(output_path),
                    num_channels=3
                )
            
            # 如果启用保存所有样本，保存到指定目录
            if save_all_samples and all_samples_vis_dir:
                vis_output_path = all_samples_vis_dir / f"vae_vis_scene{scene_id}_{base_name}.png"
                visualize_reconstruction(
                    original_img=image_data,
                    reconstructed_img=reconstructed,
                    metrics=metrics,
                    output_path=str(vis_output_path),
                    num_channels=3
                )
        
        except Exception as e:
            print(f"⚠ 处理样本 {base_name} 时出错: {e}")
            continue
    
    # 计算平均指标
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if key not in ["scene_id", "base_name"]:
                values = [m[key] for m in all_metrics if key in m]
                if values:
                    avg_metrics[key] = float(np.mean(values))
        
        print(f"\n场景 {scene_id} 平均指标:")
        print(f"  - DoLP PSNR: {avg_metrics.get('psnr_dolp', 0):.2f} dB")
        print(f"  - sin(2*AoLP) PSNR: {avg_metrics.get('psnr_sin_2aolp', 0):.2f} dB")
        print(f"  - cos(2*AoLP) PSNR: {avg_metrics.get('psnr_cos_2aolp', 0):.2f} dB")
        print(f"  - 平均 PSNR: {avg_metrics.get('psnr_mean', 0):.2f} dB")
        
        return avg_metrics, all_sample_metrics
    
    return None, []


def main():
    parser = argparse.ArgumentParser(description="Zero-shot VAE 验证脚本")
    parser.add_argument(
        "--polar_root",
        type=str,
        required=True,
        help="偏振图像根目录"
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="测试场景ID（如果提供，只验证该场景；否则验证所有场景）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./vae_validation",
        help="输出目录"
    )
    parser.add_argument(
        "--use_pt_data",
        action="store_true",
        default=False,
        help="使用预处理的 .pt 文件"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
        help="VAE 模型 ID（默认: stabilityai/sd-vae-ft-mse）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="批处理大小（VAE 显存较大，建议使用 1）"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="模型缓存目录（如果提供，模型会下载/加载到此目录，例如: /openbayes/input/input0/models/stage1_encoder_dolp_new）"
    )
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载 VAE 模型
    vae = load_vae_model(model_id=args.model_id, device=device, cache_dir=args.cache_dir)
    
    # 查找场景
    polar_root = Path(args.polar_root)
    
    # 支持 train/val 子目录结构
    # 如果 polar_root 是 /path/to/polar_pt/train，直接使用
    # 如果 polar_root 是 /path/to/polar_pt，尝试 train 子目录
    actual_data_root = polar_root
    if args.use_pt_data:
        # 检查是否有 train 子目录
        train_dir = polar_root / "train"
        if train_dir.exists() and any(train_dir.iterdir()):
            actual_data_root = train_dir
            print(f"✓ 检测到 train 子目录，使用: {actual_data_root}")
        else:
            # 直接使用 polar_root
            actual_data_root = polar_root
            print(f"✓ 使用数据根目录: {actual_data_root}")
    
    if args.scene_id:
        scene_ids = [args.scene_id]
    else:
        # 查找所有场景
        scene_ids = []
        for item in sorted(actual_data_root.iterdir()):
            if item.is_dir():
                if args.use_pt_data:
                    if any(item.glob("*.pt")):
                        scene_ids.append(item.name)
                else:
                    if any(item.glob("*_000.png")):
                        scene_ids.append(item.name)
        scene_ids = sorted(scene_ids)
        print(f"\n找到 {len(scene_ids)} 个场景")
    
    # 创建所有样本可视化目录
    all_samples_vis_dir = Path("/openbayes/input/input0/vae_vis")
    all_samples_vis_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ 所有样本可视化将保存到: {all_samples_vis_dir}")
    
    # 验证每个场景
    all_results = {}
    all_samples_list = []  # 收集所有样本的指标
    
    for scene_id in scene_ids:
        avg_metrics, sample_metrics = verify_vae_on_scene(
            vae=vae,
            device=device,
            polar_root=actual_data_root,  # 使用实际数据根目录
            scene_id=scene_id,
            output_dir=output_dir,
            use_pt_data=args.use_pt_data,
            batch_size=args.batch_size,
            save_all_samples=True,  # 保存所有样本的可视化
            all_samples_vis_dir=all_samples_vis_dir,
        )
        if avg_metrics:
            all_results[scene_id] = avg_metrics
            all_samples_list.extend(sample_metrics)  # 添加该场景的所有样本
    
    # 保存结果
    if all_results:
        json_path = output_dir / "vae_validation_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 结果已保存到: {json_path}")
        
        # 计算总体平均
        all_psnr_dolp = [r.get("psnr_dolp", 0) for r in all_results.values()]
        all_psnr_sin = [r.get("psnr_sin_2aolp", 0) for r in all_results.values()]
        all_psnr_cos = [r.get("psnr_cos_2aolp", 0) for r in all_results.values()]
        all_psnr_mean = [r.get("psnr_mean", 0) for r in all_results.values()]
        
        print(f"\n" + "=" * 80)
        print(f"总体平均指标（{len(all_results)} 个场景）")
        print("=" * 80)
        print(f"  - DoLP PSNR: {np.mean(all_psnr_dolp):.2f} dB")
        print(f"  - sin(2*AoLP) PSNR: {np.mean(all_psnr_sin):.2f} dB")
        print(f"  - cos(2*AoLP) PSNR: {np.mean(all_psnr_cos):.2f} dB")
        print(f"  - 平均 PSNR: {np.mean(all_psnr_mean):.2f} dB")
        
        # 保存总体统计到 JSON
        summary = {
            "total_scenes": len(all_results),
            "overall_average": {
                "psnr_dolp": float(np.mean(all_psnr_dolp)),
                "psnr_sin_2aolp": float(np.mean(all_psnr_sin)),
                "psnr_cos_2aolp": float(np.mean(all_psnr_cos)),
                "psnr_mean": float(np.mean(all_psnr_mean)),
            },
            "per_scene": all_results
        }
        
        summary_json_path = output_dir / "vae_validation_summary.json"
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 总体统计已保存到: {summary_json_path}")
        
        # 按平均PSNR对所有场景排序
        print(f"\n" + "=" * 80)
        print(f"所有场景按平均PSNR排序（从高到低）")
        print("=" * 80)
        
        # 创建场景列表，包含PSNR信息
        scene_psnr_list = []
        for scene_id, metrics in all_results.items():
            psnr_mean = metrics.get("psnr_mean", 0)
            psnr_dolp = metrics.get("psnr_dolp", 0)
            psnr_sin = metrics.get("psnr_sin_2aolp", 0)
            psnr_cos = metrics.get("psnr_cos_2aolp", 0)
            scene_psnr_list.append({
                "scene_id": scene_id,
                "psnr_mean": psnr_mean,
                "psnr_dolp": psnr_dolp,
                "psnr_sin": psnr_sin,
                "psnr_cos": psnr_cos,
            })
        
        # 按平均PSNR降序排序
        scene_psnr_list.sort(key=lambda x: x["psnr_mean"], reverse=True)
        
        # 输出排序结果
        print(f"\n排名 | 场景ID | 平均PSNR | DoLP PSNR | sin(2*AoLP) PSNR | cos(2*AoLP) PSNR")
        print("-" * 80)
        for rank, scene_info in enumerate(scene_psnr_list, 1):
            print(f"  {rank:2d}  |   {scene_info['scene_id']:4s}  | "
                  f"{scene_info['psnr_mean']:7.2f} dB | "
                  f"{scene_info['psnr_dolp']:9.2f} dB | "
                  f"{scene_info['psnr_sin']:15.2f} dB | "
                  f"{scene_info['psnr_cos']:15.2f} dB")
        
        # 输出统计信息
        print(f"\n" + "-" * 80)
        print(f"统计信息:")
        print(f"  - 最高PSNR: {scene_psnr_list[0]['psnr_mean']:.2f} dB (场景 {scene_psnr_list[0]['scene_id']})")
        print(f"  - 最低PSNR: {scene_psnr_list[-1]['psnr_mean']:.2f} dB (场景 {scene_psnr_list[-1]['scene_id']})")
        print(f"  - 中位数PSNR: {np.median([s['psnr_mean'] for s in scene_psnr_list]):.2f} dB")
        print(f"  - PSNR标准差: {np.std([s['psnr_mean'] for s in scene_psnr_list]):.2f} dB")
        
        # 保存排序结果到JSON
        sorted_summary = {
            "total_scenes": len(all_results),
            "overall_average": summary["overall_average"],
            "sorted_scenes": [
                {
                    "rank": rank,
                    "scene_id": scene_info["scene_id"],
                    "psnr_mean": scene_info["psnr_mean"],
                    "psnr_dolp": scene_info["psnr_dolp"],
                    "psnr_sin_2aolp": scene_info["psnr_sin"],
                    "psnr_cos_2aolp": scene_info["psnr_cos"],
                }
                for rank, scene_info in enumerate(scene_psnr_list, 1)
            ],
            "statistics": {
                "max_psnr": float(scene_psnr_list[0]['psnr_mean']),
                "max_psnr_scene": scene_psnr_list[0]['scene_id'],
                "min_psnr": float(scene_psnr_list[-1]['psnr_mean']),
                "min_psnr_scene": scene_psnr_list[-1]['scene_id'],
                "median_psnr": float(np.median([s['psnr_mean'] for s in scene_psnr_list])),
                "std_psnr": float(np.std([s['psnr_mean'] for s in scene_psnr_list])),
            },
            "per_scene": all_results  # 保留原始数据
        }
        
        sorted_json_path = output_dir / "vae_validation_sorted.json"
        with open(sorted_json_path, "w", encoding="utf-8") as f:
            json.dump(sorted_summary, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 排序结果已保存到: {sorted_json_path}")
        
        # 对所有样本按平均PSNR排序并保存
        if all_samples_list:
            print(f"\n" + "=" * 80)
            print(f"所有样本按平均PSNR排序（从高到低）")
            print("=" * 80)
            
            # 按平均PSNR排序
            all_samples_list.sort(key=lambda x: x.get("psnr_mean", 0), reverse=True)
            
            # 输出前20名和后20名
            print(f"\n前20名样本:")
            print(f"排名 | 场景ID | base_name | 平均PSNR | DoLP PSNR | sin PSNR | cos PSNR")
            print("-" * 80)
            for rank, sample in enumerate(all_samples_list[:20], 1):
                print(f"  {rank:2d}  |   {sample['scene_id']:4s}  | "
                      f"{sample['base_name']:8s} | "
                      f"{sample.get('psnr_mean', 0):7.2f} dB | "
                      f"{sample.get('psnr_dolp', 0):9.2f} dB | "
                      f"{sample.get('psnr_sin_2aolp', 0):8.2f} dB | "
                      f"{sample.get('psnr_cos_2aolp', 0):8.2f} dB")
            
            print(f"\n后20名样本:")
            print(f"排名 | 场景ID | base_name | 平均PSNR | DoLP PSNR | sin PSNR | cos PSNR")
            print("-" * 80)
            for rank, sample in enumerate(all_samples_list[-20:], len(all_samples_list) - 19):
                print(f"  {rank:2d}  |   {sample['scene_id']:4s}  | "
                      f"{sample['base_name']:8s} | "
                      f"{sample.get('psnr_mean', 0):7.2f} dB | "
                      f"{sample.get('psnr_dolp', 0):9.2f} dB | "
                      f"{sample.get('psnr_sin_2aolp', 0):8.2f} dB | "
                      f"{sample.get('psnr_cos_2aolp', 0):8.2f} dB")
            
            # 保存所有样本排序结果到JSON
            all_samples_sorted = {
                "total_samples": len(all_samples_list),
                "statistics": {
                    "max_psnr": float(all_samples_list[0].get("psnr_mean", 0)),
                    "max_psnr_scene": all_samples_list[0].get("scene_id", ""),
                    "max_psnr_base_name": all_samples_list[0].get("base_name", ""),
                    "min_psnr": float(all_samples_list[-1].get("psnr_mean", 0)),
                    "min_psnr_scene": all_samples_list[-1].get("scene_id", ""),
                    "min_psnr_base_name": all_samples_list[-1].get("base_name", ""),
                    "median_psnr": float(np.median([s.get("psnr_mean", 0) for s in all_samples_list])),
                    "std_psnr": float(np.std([s.get("psnr_mean", 0) for s in all_samples_list])),
                    "mean_psnr": float(np.mean([s.get("psnr_mean", 0) for s in all_samples_list])),
                },
                "sorted_samples": [
                    {
                        "rank": rank,
                        "scene_id": sample.get("scene_id", ""),
                        "base_name": sample.get("base_name", ""),
                        "psnr_mean": sample.get("psnr_mean", 0),
                        "psnr_dolp": sample.get("psnr_dolp", 0),
                        "psnr_sin_2aolp": sample.get("psnr_sin_2aolp", 0),
                        "psnr_cos_2aolp": sample.get("psnr_cos_2aolp", 0),
                        "mse_dolp": sample.get("mse_dolp", 0),
                        "mse_sin_2aolp": sample.get("mse_sin_2aolp", 0),
                        "mse_cos_2aolp": sample.get("mse_cos_2aolp", 0),
                        "mse_mean": sample.get("mse_mean", 0),
                    }
                    for rank, sample in enumerate(all_samples_list, 1)
                ]
            }
            
            all_samples_json_path = output_dir / "vae_all_samples_sorted.json"
            with open(all_samples_json_path, "w", encoding="utf-8") as f:
                json.dump(all_samples_sorted, f, indent=2, ensure_ascii=False)
            print(f"\n✓ 所有样本排序结果已保存到: {all_samples_json_path}")
            print(f"  - 总样本数: {len(all_samples_list)}")
            print(f"  - 最高PSNR: {all_samples_sorted['statistics']['max_psnr']:.2f} dB (场景 {all_samples_sorted['statistics']['max_psnr_scene']}, {all_samples_sorted['statistics']['max_psnr_base_name']})")
            print(f"  - 最低PSNR: {all_samples_sorted['statistics']['min_psnr']:.2f} dB (场景 {all_samples_sorted['statistics']['min_psnr_scene']}, {all_samples_sorted['statistics']['min_psnr_base_name']})")
            print(f"  - 平均PSNR: {all_samples_sorted['statistics']['mean_psnr']:.2f} dB")
            print(f"  - 中位数PSNR: {all_samples_sorted['statistics']['median_psnr']:.2f} dB")
            print(f"  - PSNR标准差: {all_samples_sorted['statistics']['std_psnr']:.2f} dB")


if __name__ == "__main__":
    main()

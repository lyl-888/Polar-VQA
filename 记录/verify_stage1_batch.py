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
    for item in sorted(polar_root.iterdir()):
        if item.is_dir():
            # 检查是否是场景目录（通常场景ID是数字字符串）
            scene_id = item.name
            # 检查目录下是否有 *_000.png 文件（PNG模式）或 *.pt 文件（PT模式）
            has_png = any(item.glob("*_000.png"))
            has_pt = any(item.glob("*.pt"))
            if has_png or has_pt:
                scene_ids.append(scene_id)
    
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


def verify_sample_silent(
    model: ViTMAEForPreTraining,
    device: torch.device,
    polar_root: Path,
    scene_id: str,
    base_name: str,
    use_pt_data: bool = False,
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
        
    Returns:
        指标字典
    """
    # 加载数据
    if use_pt_data:
        pt_path = polar_root / "train" / scene_id / f"{base_name}.pt"
        if not pt_path.exists():
            pt_path = polar_root / "val" / scene_id / f"{base_name}.pt"
        if not pt_path.exists():
            pt_path = polar_root / scene_id / f"{base_name}.pt"
        
        if not pt_path.exists():
            raise FileNotFoundError(f".pt 文件不存在: {base_name}")
        
        pixel_values_tensor = torch.load(pt_path, map_location='cpu', weights_only=False)
        if pixel_values_tensor.dtype != torch.float32:
            pixel_values_tensor = pixel_values_tensor.float()
        
        if len(pixel_values_tensor.shape) != 3:
            raise ValueError(f"加载的 .pt 文件形状不正确: {pixel_values_tensor.shape}")
        
        if pixel_values_tensor.shape[0] == 4:
            pixel_values_tensor = pixel_values_tensor[1:4, :, :]
        elif pixel_values_tensor.shape[0] != 3:
            raise ValueError(f"通道数不正确: {pixel_values_tensor.shape[0]}")
        
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
        
        metrics_list.append(metrics)
    
    return metrics_list


def verify_all_scenes(
    checkpoint_path: str,
    polar_root: str,
    output_dir: str = "./stage1_batch_validation",
    use_pt_data: bool = False,
    hf_token: Optional[str] = None,
    batch_size: int = 8,
) -> Dict:
    """
    批量验证所有场景
    
    Args:
        checkpoint_path: Stage 1 检查点路径
        polar_root: 偏振图像根目录
        output_dir: 输出目录
        use_pt_data: 是否使用 .pt 文件
        hf_token: Hugging Face token
        
    Returns:
        包含所有统计结果的字典
    """
    print("=" * 80)
    print("Stage 1 批量验证：遍历所有场景")
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
    high_quality_samples = []  # 存储 (dolp_psnr, scene_id, base_name, metrics) 的列表
    dolp_psnr_threshold = 30.0  # DoLP PSNR 阈值
    
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
                valid_indices = []
                valid_names = []
                
                for idx, base_name in enumerate(batch_names):
                    try:
                        if use_pt_data:
                            pt_path = polar_root_path / "train" / scene_id / f"{base_name}.pt"
                            if not pt_path.exists():
                                pt_path = polar_root_path / "val" / scene_id / f"{base_name}.pt"
                            if not pt_path.exists():
                                pt_path = polar_root_path / scene_id / f"{base_name}.pt"
                            
                            if not pt_path.exists():
                                continue
                            
                            pixel_values_tensor = torch.load(pt_path, map_location='cpu', weights_only=False)
                            if pixel_values_tensor.dtype != torch.float32:
                                pixel_values_tensor = pixel_values_tensor.float()
                            
                            if len(pixel_values_tensor.shape) != 3:
                                continue
                            
                            if pixel_values_tensor.shape[0] == 4:
                                pixel_values_tensor = pixel_values_tensor[1:4, :, :]
                            elif pixel_values_tensor.shape[0] != 3:
                                continue
                            
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
                    )
                    
                    # 释放显存（避免OOM）
                    del pixel_values_batch_tensor
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    
                    # 处理每个样本的结果
                    for i, (base_name, metrics) in enumerate(zip(valid_names, batch_metrics)):
                        pbar.update(1)
                        
                        # 检查 DoLP PSNR 是否 >= 30 dB
                        if "psnr_masked_dolp" in metrics:
                            dolp_psnr = metrics["psnr_masked_dolp"]
                            
                            # 如果 PSNR >= 30，添加到高质量样本列表
                            if dolp_psnr >= dolp_psnr_threshold:
                                high_quality_samples.append((dolp_psnr, scene_id, base_name, metrics.copy()))
                        
                        scene_metrics_list.append({
                            "base_name": base_name,
                            "metrics": metrics,
                        })
                except Exception as e:
                    # 如果批处理失败，回退到单样本模式
                    for base_name in valid_names:
                        try:
                            metrics = verify_sample_silent(
                                model, device, polar_root_path, scene_id, base_name, use_pt_data
                            )
                            pbar.update(1)
                            
                            if "psnr_masked_dolp" in metrics:
                                dolp_psnr = metrics["psnr_masked_dolp"]
                                
                                # 如果 PSNR >= 30，添加到高质量样本列表
                                if dolp_psnr >= dolp_psnr_threshold:
                                    high_quality_samples.append((dolp_psnr, scene_id, base_name, metrics.copy()))
                            
                            scene_metrics_list.append({
                                "base_name": base_name,
                                "metrics": metrics,
                            })
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
                        model, device, polar_root_path, scene_id, base_name, use_pt_data
                    )
                    pbar.update(1)
                    
                    # 检查 DoLP PSNR 是否 >= 30 dB
                    if "psnr_masked_dolp" in metrics:
                        dolp_psnr = metrics["psnr_masked_dolp"]
                        
                        # 如果 PSNR >= 30，添加到高质量样本列表
                        if dolp_psnr >= dolp_psnr_threshold:
                            high_quality_samples.append((dolp_psnr, scene_id, base_name, metrics.copy()))
                    
                    scene_metrics_list.append({
                        "base_name": base_name,
                        "metrics": metrics,
                    })
                except Exception as e:
                    pbar.update(1)
                    continue
        
        # 计算该场景的平均指标
        if scene_metrics_list:
            num_samples = len(scene_metrics_list)
            
            # 收集所有指标
            all_mse_dolp = [m["metrics"].get("mse_masked_dolp", 0) for m in scene_metrics_list]
            all_mse_sin = [m["metrics"].get("mse_masked_sin_2aolp", 0) for m in scene_metrics_list]
            all_mse_cos = [m["metrics"].get("mse_masked_cos_2aolp", 0) for m in scene_metrics_list]
            all_psnr_dolp = [m["metrics"].get("psnr_masked_dolp", 0) for m in scene_metrics_list]
            all_psnr_sin = [m["metrics"].get("psnr_masked_sin_2aolp", 0) for m in scene_metrics_list]
            all_psnr_cos = [m["metrics"].get("psnr_masked_cos_2aolp", 0) for m in scene_metrics_list]
            
            # 计算平均值
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
    
    # 按场景分组，每个场景只保留 DoLP PSNR 最高的样本
    scene_best_samples = {}  # {scene_id: (dolp_psnr, base_name, metrics)}
    
    for dolp_psnr, scene_id, base_name, metrics in high_quality_samples:
        if scene_id not in scene_best_samples:
            scene_best_samples[scene_id] = (dolp_psnr, base_name, metrics)
        else:
            # 如果当前样本的 DoLP PSNR 更高，则替换
            current_best_psnr = scene_best_samples[scene_id][0]
            if dolp_psnr > current_best_psnr:
                scene_best_samples[scene_id] = (dolp_psnr, base_name, metrics)
    
    # 按 DoLP PSNR 降序排列（用于显示）
    best_samples_sorted = sorted(
        [(psnr, scene_id, base_name, metrics) for scene_id, (psnr, base_name, metrics) in scene_best_samples.items()],
        key=lambda x: x[0],
        reverse=True
    )
    
    # 保存每个场景的最佳样本的可视化（延迟保存，避免影响批处理速度）
    if best_samples_sorted:
        print(f"\n正在保存 {len(best_samples_sorted)} 个场景的最佳样本可视化 (DoLP PSNR >= {dolp_psnr_threshold} dB)...", end="", flush=True)
        saved_samples_list = []
        
        for rank, (dolp_psnr, scene_id, base_name, metrics) in enumerate(best_samples_sorted, 1):
            output_path = output_dir_path / f"best_scene{scene_id}_{base_name}_dolp{dolp_psnr:.1f}dB.png"
            
            with suppress_stdout():
                run_one_example(
                    model=model,
                    device=device,
                    polar_root=polar_root_path,
                    scene_id=scene_id,
                    base_name=base_name,
                    output_path=str(output_path),
                    use_pt_data=use_pt_data,
                )
            
            # 计算 AoLP 的 PSNR（从 sin 和 cos 的 PSNR 计算平均值，或者单独计算）
            # 注意：AoLP 本身不是直接存储的，通常用 sin(2*AoLP) 和 cos(2*AoLP) 表示
            # 这里我们使用 sin 和 cos 的平均 PSNR 作为 AoLP 的近似
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
                "mse_masked_dolp": metrics.get("mse_masked_dolp", 0),
                "mse_masked_sin_2aolp": metrics.get("mse_masked_sin_2aolp", 0),
                "mse_masked_cos_2aolp": metrics.get("mse_masked_cos_2aolp", 0),
                "mse_masked_mean": metrics.get("mse_masked_mean", 0),
                # PSNR 指标（Masked 区域）
                "psnr_masked_dolp": dolp_psnr,
                "psnr_masked_sin_2aolp": aolp_psnr_sin,
                "psnr_masked_cos_2aolp": aolp_psnr_cos,
                "psnr_masked_aolp_approx": aolp_psnr_approx,  # AoLP 的近似 PSNR
                "psnr_masked_mean": metrics.get("psnr_masked_mean", 0),
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
        
        # 打印统计信息
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
            print(f"\n  前5名场景:")
            for item in saved_samples_list[:5]:
                print(f"    #{item['rank']:2d}: scene_id={item['scene_id']}, base_name={item['base_name']}, "
                      f"DoLP PSNR={item['dolp_psnr']:.2f} dB, AoLP PSNR≈{item['psnr_masked_aolp_approx']:.2f} dB")
            if len(best_samples_sorted) > 5:
                print(f"    ... (共 {len(best_samples_sorted)} 个场景，全部已保存)")
    else:
        print(f"\n⚠ 未找到 DoLP PSNR >= {dolp_psnr_threshold} dB 的样本")
    
    # 保存 JSON 结果
    json_path = output_dir_path / "batch_validation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 统计结果已保存: {json_path}")
    
    # 打印总体统计
    valid_scenes = [s for s in all_results["scenes"].values() if s.get("num_samples", 0) > 0]
    if valid_scenes:
        total_samples = sum(s["num_samples"] for s in valid_scenes)
        avg_psnr_dolp = np.mean([s["psnr_masked_dolp"] for s in valid_scenes])
        avg_psnr_sin = np.mean([s["psnr_masked_sin_2aolp"] for s in valid_scenes])
        avg_psnr_cos = np.mean([s["psnr_masked_cos_2aolp"] for s in valid_scenes])
        print(f"\n总体统计:")
        print(f"  - 有效场景数: {len(valid_scenes)}")
        print(f"  - 总样本数: {total_samples}")
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
    
    args = parser.parse_args()
    
    verify_all_scenes(
        checkpoint_path=args.checkpoint,
        polar_root=args.polar_root,
        output_dir=args.output_dir,
        use_pt_data=args.use_pt_data,
        hf_token=args.hf_token,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()


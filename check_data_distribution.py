"""
检查 Stage 1 训练数据的分布情况
用于诊断 loss 低的问题

用法:
python check_data_distribution.py \
    --pt_root /openbayes/home/data/polar_pt \
    --num_samples 100
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict


def check_pt_file_distribution(pt_path: Path) -> Dict[str, float]:
    """
    检查单个 .pt 文件的数据分布
    
    Args:
        pt_path: .pt 文件路径
    
    Returns:
        包含统计信息的字典
    """
    data = torch.load(pt_path, map_location='cpu')
    
    # 确保是float32
    if data.dtype != torch.float32:
        data = data.float()
    
    # 检查形状
    if len(data.shape) != 3:
        raise ValueError(f"数据形状不正确: {data.shape}, 期望: (C, H, W)")
    
    # 如果是4通道，去掉Intensity通道（与训练时一致）
    if data.shape[0] == 4:
        data = data[1:4, :, :]  # 只保留 DoLP, sin, cos
        print(f"  ⚠ 注意: 文件包含4通道，已自动移除Intensity通道")
    
    # 确保是3通道
    if data.shape[0] != 3:
        raise ValueError(f"数据通道数不正确: {data.shape[0]}, 期望: 3")
    
    # 计算统计信息
    stats = {
        'dolp_min': data[0].min().item(),
        'dolp_max': data[0].max().item(),
        'dolp_mean': data[0].mean().item(),
        'dolp_std': data[0].std().item(),
        'sin_min': data[1].min().item(),
        'sin_max': data[1].max().item(),
        'sin_mean': data[1].mean().item(),
        'sin_std': data[1].std().item(),
        'cos_min': data[2].min().item(),
        'cos_max': data[2].max().item(),
        'cos_mean': data[2].mean().item(),
        'cos_std': data[2].std().item(),
    }
    
    return stats


def check_dataset_distribution(pt_root: Path, num_samples: int = 100) -> Dict[str, List[float]]:
    """
    检查整个数据集的数据分布
    
    Args:
        pt_root: .pt 文件根目录（应该包含 train/ 和 val/ 子目录）
        num_samples: 检查的样本数量
    
    Returns:
        包含所有样本统计信息的字典
    """
    # 收集所有 .pt 文件路径
    pt_files = []
    
    # 优先检查 train 目录
    train_dir = pt_root / "train"
    if train_dir.exists():
        for scene_dir in sorted(train_dir.iterdir()):
            if scene_dir.is_dir():
                for pt_file in sorted(scene_dir.glob("*.pt")):
                    pt_files.append(pt_file)
                    if len(pt_files) >= num_samples:
                        break
            if len(pt_files) >= num_samples:
                break
    
    # 如果 train 目录样本不够，从 val 目录补充
    if len(pt_files) < num_samples:
        val_dir = pt_root / "val"
        if val_dir.exists():
            for scene_dir in sorted(val_dir.iterdir()):
                if scene_dir.is_dir():
                    for pt_file in sorted(scene_dir.glob("*.pt")):
                        pt_files.append(pt_file)
                        if len(pt_files) >= num_samples:
                            break
                if len(pt_files) >= num_samples:
                    break
    
    if len(pt_files) == 0:
        raise ValueError(f"未找到任何 .pt 文件！请检查目录: {pt_root}")
    
    print(f"✓ 找到 {len(pt_files)} 个样本，开始检查数据分布...")
    
    # 收集所有样本的统计信息
    all_stats = {
        'dolp_min': [],
        'dolp_max': [],
        'dolp_mean': [],
        'dolp_std': [],
        'sin_min': [],
        'sin_max': [],
        'sin_mean': [],
        'sin_std': [],
        'cos_min': [],
        'cos_max': [],
        'cos_mean': [],
        'cos_std': [],
    }
    
    # 检查每个样本
    for i, pt_file in enumerate(pt_files):
        try:
            stats = check_pt_file_distribution(pt_file)
            for key in all_stats.keys():
                all_stats[key].append(stats[key])
            
            if (i + 1) % 20 == 0:
                print(f"  已处理 {i + 1}/{len(pt_files)} 个样本...")
        except Exception as e:
            print(f"  ⚠ 警告: 处理文件 {pt_file} 时出错: {e}")
            continue
    
    return all_stats


def print_statistics(all_stats: Dict[str, List[float]], num_samples: int):
    """
    打印统计信息
    
    Args:
        all_stats: 包含所有样本统计信息的字典
        num_samples: 样本数量
    """
    print("\n" + "=" * 80)
    print("数据分布统计（基于多个样本）")
    print("=" * 80)
    
    # DoLP 通道
    dolp_min_mean = np.mean(all_stats['dolp_min'])
    dolp_min_min = np.min(all_stats['dolp_min'])
    dolp_max_mean = np.mean(all_stats['dolp_max'])
    dolp_max_max = np.max(all_stats['dolp_max'])
    dolp_mean_mean = np.mean(all_stats['dolp_mean'])
    dolp_mean_std = np.std(all_stats['dolp_mean'])
    dolp_std_mean = np.mean(all_stats['dolp_std'])
    
    print("\n📊 DoLP 通道统计:")
    print(f"  - 最小值范围: [{dolp_min_min:.4f}, {dolp_min_mean:.4f}] (平均)")
    print(f"  - 最大值范围: [{dolp_max_mean:.4f}, {dolp_max_max:.4f}] (平均)")
    print(f"  - 均值: {dolp_mean_mean:.4f} ± {dolp_mean_std:.4f} (跨样本)")
    print(f"  - 标准差: {dolp_std_mean:.4f} (平均)")
    
    # sin 通道
    sin_min_mean = np.mean(all_stats['sin_min'])
    sin_min_min = np.min(all_stats['sin_min'])
    sin_max_mean = np.mean(all_stats['sin_max'])
    sin_max_max = np.max(all_stats['sin_max'])
    sin_mean_mean = np.mean(all_stats['sin_mean'])
    sin_mean_std = np.std(all_stats['sin_mean'])
    sin_std_mean = np.mean(all_stats['sin_std'])
    
    print("\n📊 sin(2*AoLP) 通道统计:")
    print(f"  - 最小值范围: [{sin_min_min:.4f}, {sin_min_mean:.4f}] (平均)")
    print(f"  - 最大值范围: [{sin_max_mean:.4f}, {sin_max_max:.4f}] (平均)")
    print(f"  - 均值: {sin_mean_mean:.4f} ± {sin_mean_std:.4f} (跨样本)")
    print(f"  - 标准差: {sin_std_mean:.4f} (平均)")
    
    # cos 通道
    cos_min_mean = np.mean(all_stats['cos_min'])
    cos_min_min = np.min(all_stats['cos_min'])
    cos_max_mean = np.mean(all_stats['cos_max'])
    cos_max_max = np.max(all_stats['cos_max'])
    cos_mean_mean = np.mean(all_stats['cos_mean'])
    cos_mean_std = np.std(all_stats['cos_mean'])
    cos_std_mean = np.mean(all_stats['cos_std'])
    
    print("\n📊 cos(2*AoLP) 通道统计:")
    print(f"  - 最小值范围: [{cos_min_min:.4f}, {cos_min_mean:.4f}] (平均)")
    print(f"  - 最大值范围: [{cos_max_mean:.4f}, {cos_max_max:.4f}] (平均)")
    print(f"  - 均值: {cos_mean_mean:.4f} ± {cos_mean_std:.4f} (跨样本)")
    print(f"  - 标准差: {cos_std_mean:.4f} (平均)")
    
    # 诊断建议
    print("\n" + "=" * 80)
    print("诊断建议")
    print("=" * 80)
    
    # 检查 DoLP 值域
    if dolp_max_max < 0.3:
        print("⚠️  警告: DoLP 最大值很小（<0.3），说明数据值域很小")
        print("   → Loss 低（0.008-0.009）是正常的，因为数据值域小")
        print("   → 这是物理特性，不是数据问题")
    elif dolp_max_max < 0.5:
        print("✓ DoLP 值域中等（0-0.5），Loss 低但可接受")
    else:
        print("✓ DoLP 值域正常（0-1），Loss 应该更高")
    
    # 检查 sin/cos 范围
    if sin_min_min < -0.1 or cos_min_min < -0.1:
        print("⚠️  警告: sin/cos 通道包含负值（可能未归一化到[0,1]）")
        print("   → 建议检查数据预处理，确保 sin/cos 映射到 [0, 1]")
    elif sin_max_max > 1.1 or cos_max_max > 1.1:
        print("⚠️  警告: sin/cos 通道值超出 [0, 1] 范围")
        print("   → 建议检查数据预处理")
    else:
        print("✓ sin/cos 通道值域正常（[0, 1]）")
    
    # 检查 DoLP 均值
    if dolp_mean_mean < 0.15:
        print(f"\n💡 关键发现: DoLP 均值很小（{dolp_mean_mean:.4f}）")
        print("   → 说明大部分区域的偏振度很低")
        print("   → Loss 低（0.008-0.009）是正常的，因为：")
        print("     * 数据值域小 → MSE 自然小")
        print("     * 模型预测接近真实值（即使绝对值误差小，相对误差可能不小）")
        print("   → 建议：运行 verify_stage1.py 检查重建质量（PSNR）")
        print("   → 如果 PSNR > 15 dB，说明训练正常，Loss 低是合理的")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="检查 Stage 1 训练数据的分布情况")
    parser.add_argument(
        "--pt_root",
        type=str,
        default="/openbayes/home/data/polar_pt",
        help=".pt 文件根目录（应该包含 train/ 和 val/ 子目录）"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="检查的样本数量（默认100）"
    )
    parser.add_argument(
        "--single_file",
        type=str,
        default=None,
        help="如果提供，只检查单个文件（用于快速测试）"
    )
    
    args = parser.parse_args()
    
    pt_root = Path(args.pt_root)
    
    if not pt_root.exists():
        raise FileNotFoundError(f"目录不存在: {pt_root}")
    
    print("=" * 80)
    print("检查 Stage 1 训练数据分布")
    print("=" * 80)
    print(f"数据目录: {pt_root}")
    
    if args.single_file:
        # 检查单个文件
        pt_file = Path(args.single_file)
        if not pt_file.exists():
            raise FileNotFoundError(f"文件不存在: {pt_file}")
        
        print(f"\n检查单个文件: {pt_file}")
        stats = check_pt_file_distribution(pt_file)
        
        print("\n📊 数据统计:")
        print(f"  - DoLP:   范围=[{stats['dolp_min']:.4f}, {stats['dolp_max']:.4f}], "
              f"均值={stats['dolp_mean']:.4f}, 标准差={stats['dolp_std']:.4f}")
        print(f"  - sin:    范围=[{stats['sin_min']:.4f}, {stats['sin_max']:.4f}], "
              f"均值={stats['sin_mean']:.4f}, 标准差={stats['sin_std']:.4f}")
        print(f"  - cos:    范围=[{stats['cos_min']:.4f}, {stats['cos_max']:.4f}], "
              f"均值={stats['cos_mean']:.4f}, 标准差={stats['cos_std']:.4f}")
    else:
        # 检查多个样本
        print(f"\n检查样本数量: {args.num_samples}")
        all_stats = check_dataset_distribution(pt_root, args.num_samples)
        print_statistics(all_stats, len(all_stats['dolp_mean']))
    
    print("\n✓ 检查完成！")


if __name__ == "__main__":
    main()


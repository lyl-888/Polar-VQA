"""
对比 DoLP 专用编码器和 VAE 的重建结果

功能：
1. 对 DoLP 编码器结果按 psnr_masked_dolp 排序
2. 对比两种方法的重建质量
3. 分析相关性、分布特征和一致性
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def load_dolp_encoder_results(json_path: Path) -> Dict:
    """加载 DoLP 编码器结果"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_vae_results(json_path: Path) -> Dict:
    """加载 VAE 结果"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def sort_dolp_encoder_results(dolp_data: Dict) -> List[Dict]:
    """对 DoLP 编码器结果按 psnr_masked_dolp 排序"""
    scenes = dolp_data.get('scenes', {})
    
    scene_list = []
    for scene_id, metrics in scenes.items():
        scene_list.append({
            'scene_id': scene_id,
            'psnr_masked_dolp': metrics.get('psnr_masked_dolp', 0),
            'mse_masked_dolp': metrics.get('mse_masked_dolp', 0),
            'num_samples': metrics.get('num_samples', 0),
        })
    
    # 按 psnr_masked_dolp 降序排序
    scene_list.sort(key=lambda x: x['psnr_masked_dolp'], reverse=True)
    
    return scene_list


def sort_joint_encoder_results(joint_data: Dict) -> List[Dict]:
    """对联合编码器结果按平均PSNR排序"""
    scenes = joint_data.get('scenes', {})
    
    scene_list = []
    for scene_id, metrics in scenes.items():
        psnr_dolp = metrics.get('psnr_masked_dolp', 0)
        psnr_sin = metrics.get('psnr_masked_sin_2aolp', 0)
        psnr_cos = metrics.get('psnr_masked_cos_2aolp', 0)
        psnr_mean = (psnr_dolp + psnr_sin + psnr_cos) / 3.0
        
        scene_list.append({
            'scene_id': scene_id,
            'psnr_masked_dolp': psnr_dolp,
            'psnr_masked_sin_2aolp': psnr_sin,
            'psnr_masked_cos_2aolp': psnr_cos,
            'psnr_mean': psnr_mean,
            'mse_masked_dolp': metrics.get('mse_masked_dolp', 0),
            'mse_masked_sin_2aolp': metrics.get('mse_masked_sin_2aolp', 0),
            'mse_masked_cos_2aolp': metrics.get('mse_masked_cos_2aolp', 0),
            'num_samples': metrics.get('num_samples', 0),
        })
    
    # 按平均PSNR降序排序
    scene_list.sort(key=lambda x: x['psnr_mean'], reverse=True)
    
    return scene_list


def extract_vae_dolp_psnr(vae_data: Dict) -> Dict[str, float]:
    """从 VAE 结果中提取每个场景的 DoLP PSNR"""
    vae_psnr = {}
    
    # 从 sorted_scenes 中提取
    sorted_scenes = vae_data.get('sorted_scenes', [])
    for scene in sorted_scenes:
        scene_id = scene.get('scene_id')
        psnr_dolp = scene.get('psnr_dolp', 0)
        if scene_id:
            vae_psnr[scene_id] = psnr_dolp
    
    # 如果 sorted_scenes 中没有，从 per_scene 中提取
    if not vae_psnr:
        per_scene = vae_data.get('per_scene', {})
        for scene_id, metrics in per_scene.items():
            vae_psnr[scene_id] = metrics.get('psnr_dolp', 0)
    
    return vae_psnr


def compare_results(
    encoder_sorted: List[Dict],
    vae_psnr: Dict[str, float],
    output_path: Path,
    encoder_type: str = "dolp"  # "dolp" or "joint"
):
    """对比两种方法的结果
    
    Args:
        encoder_sorted: 编码器结果（已排序）
        vae_psnr: VAE的DoLP PSNR字典
        output_path: 输出路径
        encoder_type: 编码器类型，"dolp" 或 "joint"
    """
    
    # 提取两个方法中每个场景的 PSNR
    encoder_psnr_list = []
    vae_psnr_list = []
    scene_ids = []
    
    for scene_info in encoder_sorted:
        scene_id = scene_info['scene_id']
        
        if encoder_type == "dolp":
            encoder_psnr = scene_info['psnr_masked_dolp']
        else:  # joint
            encoder_psnr = scene_info.get('psnr_mean', 
                (scene_info.get('psnr_masked_dolp', 0) + 
                 scene_info.get('psnr_masked_sin_2aolp', 0) + 
                 scene_info.get('psnr_masked_cos_2aolp', 0)) / 3.0)
        
        if scene_id in vae_psnr:
            encoder_psnr_list.append(encoder_psnr)
            vae_psnr_list.append(vae_psnr[scene_id])
            scene_ids.append(scene_id)
    
    encoder_psnr_array = np.array(encoder_psnr_list)
    vae_psnr_array = np.array(vae_psnr_list)
    
    # 对于联合编码器，还需要单独计算每个通道的统计量（与验证脚本保持一致）
    if encoder_type == "joint":
        # 提取每个通道的PSNR（用于与验证脚本的统计方式保持一致）
        encoder_dolp_list = [s.get('psnr_masked_dolp', 0) for s in encoder_sorted if s['scene_id'] in scene_ids]
        encoder_sin_list = [s.get('psnr_masked_sin_2aolp', 0) for s in encoder_sorted if s['scene_id'] in scene_ids]
        encoder_cos_list = [s.get('psnr_masked_cos_2aolp', 0) for s in encoder_sorted if s['scene_id'] in scene_ids]
        
        encoder_dolp_mean = np.mean(encoder_dolp_list)
        encoder_sin_mean = np.mean(encoder_sin_list)
        encoder_cos_mean = np.mean(encoder_cos_list)
        
        # 验证：平均PSNR应该等于三个通道平均值的平均
        # (encoder_dolp_mean + encoder_sin_mean + encoder_cos_mean) / 3.0 应该约等于 encoder_mean
    
    # 计算统计量（基于平均PSNR）
    encoder_mean = np.mean(encoder_psnr_array)
    encoder_std = np.std(encoder_psnr_array)
    encoder_min = np.min(encoder_psnr_array)
    encoder_max = np.max(encoder_psnr_array)
    encoder_median = np.median(encoder_psnr_array)
    
    vae_mean = np.mean(vae_psnr_array)
    vae_std = np.std(vae_psnr_array)
    vae_min = np.min(vae_psnr_array)
    vae_max = np.max(vae_psnr_array)
    vae_median = np.median(vae_psnr_array)
    
    # 计算相关性
    correlation = np.corrcoef(encoder_psnr_array, vae_psnr_array)[0, 1]
    
    # 计算差异
    diff_array = encoder_psnr_array - vae_psnr_array
    mean_diff = np.mean(diff_array)
    std_diff = np.std(diff_array)
    
    # 创建所有场景的差异列表（用于完整分析）
    all_scenes_diff = []
    for i, scene_id in enumerate(scene_ids):
        all_scenes_diff.append({
            'scene_id': scene_id,
            'encoder_psnr': encoder_psnr_array[i],
            'vae_psnr': vae_psnr_array[i],
            'diff': diff_array[i],
            'abs_diff': abs(diff_array[i])
        })
    
    # 找出两种方法下都高/都低/不一致的场景
    # 定义"高"为高于中位数，"低"为低于中位数
    encoder_high_threshold = encoder_median
    vae_high_threshold = vae_median
    
    both_high = []
    both_low = []
    encoder_high_vae_low = []
    encoder_low_vae_high = []
    
    encoder_name = "DoLP编码器" if encoder_type == "dolp" else "联合编码器"
    
    for i, scene_id in enumerate(scene_ids):
        encoder_val = encoder_psnr_array[i]
        vae_val = vae_psnr_array[i]
        
        encoder_is_high = encoder_val >= encoder_high_threshold
        vae_is_high = vae_val >= vae_high_threshold
        
        if encoder_is_high and vae_is_high:
            both_high.append({
                'scene_id': scene_id,
                'encoder_psnr': encoder_val,
                'vae_psnr': vae_val,
                'diff': encoder_val - vae_val
            })
        elif not encoder_is_high and not vae_is_high:
            both_low.append({
                'scene_id': scene_id,
                'encoder_psnr': encoder_val,
                'vae_psnr': vae_val,
                'diff': encoder_val - vae_val
            })
        elif encoder_is_high and not vae_is_high:
            encoder_high_vae_low.append({
                'scene_id': scene_id,
                'encoder_psnr': encoder_val,
                'vae_psnr': vae_val,
                'diff': encoder_val - vae_val
            })
        else:  # encoder_low and vae_high
            encoder_low_vae_high.append({
                'scene_id': scene_id,
                'encoder_psnr': encoder_val,
                'vae_psnr': vae_val,
                'diff': encoder_val - vae_val
            })
    
    # 输出结果
    print("=" * 100)
    if encoder_type == "dolp":
        print("DoLP 专用编码器结果排序（按 psnr_masked_dolp 从高到低）")
        print("=" * 100)
        print(f"\n排名 | 场景ID | DoLP PSNR | MSE | 样本数")
        print("-" * 100)
        for rank, scene_info in enumerate(encoder_sorted, 1):
            print(f"  {rank:2d}  |   {scene_info['scene_id']:4s}  | "
                  f"{scene_info['psnr_masked_dolp']:9.2f} dB | "
                  f"{scene_info['mse_masked_dolp']:.6f} | "
                  f"{scene_info['num_samples']:4d}")
    else:  # joint
        print("联合编码器结果排序（按平均PSNR从高到低）")
        print("=" * 100)
        print(f"\n排名 | 场景ID | 平均PSNR | DoLP PSNR | sin PSNR | cos PSNR | 样本数")
        print("-" * 100)
        for rank, scene_info in enumerate(encoder_sorted, 1):
            print(f"  {rank:2d}  |   {scene_info['scene_id']:4s}  | "
                  f"{scene_info['psnr_mean']:9.2f} dB | "
                  f"{scene_info['psnr_masked_dolp']:9.2f} dB | "
                  f"{scene_info['psnr_masked_sin_2aolp']:9.2f} dB | "
                  f"{scene_info['psnr_masked_cos_2aolp']:9.2f} dB | "
                  f"{scene_info['num_samples']:4d}")
    
    print("\n" + "=" * 100)
    print("统计对比分析")
    print("=" * 100)
    
    encoder_label = "DoLP专用编码器" if encoder_type == "dolp" else "联合编码器"
    print(f"\n【{encoder_label}统计】")
    if encoder_type == "joint":
        # 对于联合编码器，显示每个通道的平均值（与验证脚本保持一致）
        # 这是按照 verify_stage1_batch.py 的方式：先对每个场景的每个通道求平均，再对所有场景求平均
        print(f"  - 平均 DoLP PSNR: {encoder_dolp_mean:.2f} dB")
        print(f"  - 平均 sin(2*AoLP) PSNR: {encoder_sin_mean:.2f} dB")
        print(f"  - 平均 cos(2*AoLP) PSNR: {encoder_cos_mean:.2f} dB")
        print(f"  - 平均 PSNR (三通道平均): {encoder_mean:.2f} dB")
        print(f"    注: 三通道平均 = (DoLP + sin + cos) / 3 = ({encoder_dolp_mean:.2f} + {encoder_sin_mean:.2f} + {encoder_cos_mean:.2f}) / 3")
    else:
        print(f"  - 平均 PSNR: {encoder_mean:.2f} dB")
    print(f"  - 标准差: {encoder_std:.2f} dB")
    print(f"  - 最小值: {encoder_min:.2f} dB")
    print(f"  - 最大值: {encoder_max:.2f} dB")
    print(f"  - 中位数: {encoder_median:.2f} dB")
    print(f"  - 范围: {encoder_max - encoder_min:.2f} dB")
    
    print(f"\n【VAE Zero-shot 统计】")
    print(f"  - 平均 PSNR: {vae_mean:.2f} dB")
    print(f"  - 标准差: {vae_std:.2f} dB")
    print(f"  - 最小值: {vae_min:.2f} dB")
    print(f"  - 最大值: {vae_max:.2f} dB")
    print(f"  - 中位数: {vae_median:.2f} dB")
    print(f"  - 范围: {vae_max - vae_min:.2f} dB")
    
    print(f"\n【对比分析】")
    print(f"  - 相关性系数: {correlation:.4f}")
    print(f"  - 平均差异 ({encoder_label} - VAE): {mean_diff:.2f} dB")
    print(f"  - 差异标准差: {std_diff:.2f} dB")
    print(f"  - {encoder_label} 标准差 / VAE 标准差: {encoder_std / vae_std:.2f}x")
    print(f"  - {encoder_label} 范围 / VAE 范围: {(encoder_max - encoder_min) / (vae_max - vae_min):.2f}x")
    
    print(f"\n【一致性分析】（以各自中位数为阈值）")
    print(f"  - 两种方法都高: {len(both_high)} 个场景 ({len(both_high)/len(scene_ids)*100:.1f}%)")
    print(f"  - 两种方法都低: {len(both_low)} 个场景 ({len(both_low)/len(scene_ids)*100:.1f}%)")
    print(f"  - {encoder_label}高但VAE低: {len(encoder_high_vae_low)} 个场景 ({len(encoder_high_vae_low)/len(scene_ids)*100:.1f}%)")
    print(f"  - {encoder_label}低但VAE高: {len(encoder_low_vae_high)} 个场景 ({len(encoder_low_vae_high)/len(scene_ids)*100:.1f}%)")
    print(f"  - 一致性: {(len(both_high) + len(both_low))/len(scene_ids)*100:.1f}%")
    
    # 显示不一致的场景（完整输出所有场景）
    if encoder_high_vae_low:
        print(f"\n【{encoder_label}高但VAE低的场景】（{encoder_name}表现更好，共 {len(encoder_high_vae_low)} 个）")
        encoder_high_vae_low.sort(key=lambda x: x['diff'], reverse=True)
        print(f"排名 | 场景ID | {encoder_label} PSNR | VAE PSNR | 差异 ({encoder_label}-VAE)")
        print("-" * 100)
        for rank, item in enumerate(encoder_high_vae_low, 1):
            print(f"  {rank:2d}  |   {item['scene_id']:4s}  | "
                  f"{item['encoder_psnr']:9.2f} dB | "
                  f"{item['vae_psnr']:9.2f} dB | "
                  f"{item['diff']:9.2f} dB")
    
    if encoder_low_vae_high:
        print(f"\n【{encoder_label}低但VAE高的场景】（VAE表现更好，共 {len(encoder_low_vae_high)} 个）")
        encoder_low_vae_high.sort(key=lambda x: x['diff'], reverse=False)  # 差异从最负到最正
        print(f"排名 | 场景ID | {encoder_label} PSNR | VAE PSNR | 差异 ({encoder_label}-VAE)")
        print("-" * 100)
        for rank, item in enumerate(encoder_low_vae_high, 1):
            print(f"  {rank:2d}  |   {item['scene_id']:4s}  | "
                  f"{item['encoder_psnr']:9.2f} dB | "
                  f"{item['vae_psnr']:9.2f} dB | "
                  f"{item['diff']:9.2f} dB")
    
    # 显示所有场景的差异分析（按绝对差异排序）
    print(f"\n" + "=" * 100)
    print("所有场景的PSNR差异分析（按绝对差异从大到小排序）")
    print("=" * 100)
    
    # 按绝对差异排序
    all_scenes_diff_sorted = sorted(all_scenes_diff, key=lambda x: x['abs_diff'], reverse=True)
    
    # 定义显著差异阈值（比如差异超过2dB）
    significant_diff_threshold = 2.0
    significant_diff_scenes = [s for s in all_scenes_diff_sorted if s['abs_diff'] >= significant_diff_threshold]
    
    print(f"\n【显著差异场景】（绝对差异 >= {significant_diff_threshold} dB，共 {len(significant_diff_scenes)} 个）")
    print(f"排名 | 场景ID | {encoder_label} PSNR | VAE PSNR | 差异 ({encoder_label}-VAE) | 绝对差异")
    print("-" * 100)
    for rank, item in enumerate(significant_diff_scenes, 1):
        diff_sign = "+" if item['diff'] >= 0 else "-"
        print(f"  {rank:2d}  |   {item['scene_id']:4s}  | "
              f"{item['encoder_psnr']:9.2f} dB | "
              f"{item['vae_psnr']:9.2f} dB | "
              f"{diff_sign}{abs(item['diff']):8.2f} dB | "
              f"{item['abs_diff']:9.2f} dB")
    
    # 统计差异分布
    positive_diff = [s for s in all_scenes_diff_sorted if s['diff'] > 0]  # encoder > VAE
    negative_diff = [s for s in all_scenes_diff_sorted if s['diff'] < 0]  # encoder < VAE
    zero_diff = [s for s in all_scenes_diff_sorted if abs(s['diff']) < 0.01]  # 几乎相等
    
    print(f"\n【差异分布统计】")
    print(f"  - {encoder_label} > VAE（{encoder_name}表现更好）: {len(positive_diff)} 个场景 ({len(positive_diff)/len(scene_ids)*100:.1f}%)")
    if positive_diff:
        avg_pos_diff = np.mean([s['diff'] for s in positive_diff])
        max_pos_diff = max([s['diff'] for s in positive_diff])
        print(f"    平均差异: {avg_pos_diff:.2f} dB, 最大差异: {max_pos_diff:.2f} dB")
    
    print(f"  - {encoder_label} < VAE（VAE表现更好）: {len(negative_diff)} 个场景 ({len(negative_diff)/len(scene_ids)*100:.1f}%)")
    if negative_diff:
        avg_neg_diff = np.mean([s['diff'] for s in negative_diff])
        min_neg_diff = min([s['diff'] for s in negative_diff])
        print(f"    平均差异: {avg_neg_diff:.2f} dB, 最大差异: {min_neg_diff:.2f} dB")
    
    print(f"  - {encoder_label} ≈ VAE（几乎相等）: {len(zero_diff)} 个场景 ({len(zero_diff)/len(scene_ids)*100:.1f}%)")
    
    # 显示差异最大的前20个场景
    print(f"\n【差异最大的前20个场景】（按绝对差异排序）")
    print(f"排名 | 场景ID | {encoder_label} PSNR | VAE PSNR | 差异 ({encoder_label}-VAE) | 绝对差异")
    print("-" * 100)
    for rank, item in enumerate(all_scenes_diff_sorted[:20], 1):
        diff_sign = "+" if item['diff'] >= 0 else "-"
        print(f"  {rank:2d}  |   {item['scene_id']:4s}  | "
              f"{item['encoder_psnr']:9.2f} dB | "
              f"{item['vae_psnr']:9.2f} dB | "
              f"{diff_sign}{abs(item['diff']):8.2f} dB | "
              f"{item['abs_diff']:9.2f} dB")
    
    # 保存结果到JSON
    result = {
        'encoder_type': encoder_type,
        'encoder_sorted': [
            {
                'rank': rank,
                **scene_info
            }
            for rank, scene_info in enumerate(encoder_sorted, 1)
        ],
        'statistics': {
            'encoder': {
                'mean': float(encoder_mean),
                'std': float(encoder_std),
                'min': float(encoder_min),
                'max': float(encoder_max),
                'median': float(encoder_median),
                'range': float(encoder_max - encoder_min),
            },
            'vae': {
                'mean': float(vae_mean),
                'std': float(vae_std),
                'min': float(vae_min),
                'max': float(vae_max),
                'median': float(vae_median),
                'range': float(vae_max - vae_min),
            },
            'comparison': {
                'correlation': float(correlation),
                'mean_diff': float(mean_diff),
                'std_diff': float(std_diff),
                'std_ratio': float(encoder_std / vae_std),
                'range_ratio': float((encoder_max - encoder_min) / (vae_max - vae_min)),
            },
            'consistency': {
                'both_high': len(both_high),
                'both_low': len(both_low),
                'encoder_high_vae_low': len(encoder_high_vae_low),
                'encoder_low_vae_high': len(encoder_low_vae_high),
                'consistency_rate': float((len(both_high) + len(both_low)) / len(scene_ids) * 100),
            }
        },
        'inconsistent_scenes': {
            'encoder_high_vae_low': encoder_high_vae_low,
            'encoder_low_vae_high': encoder_low_vae_high,
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] 对比结果已保存到: {output_path}")
    
    # 输出结论
    print("\n" + "=" * 100)
    print("结论")
    print("=" * 100)
    
    if encoder_std > vae_std * 1.2:
        print(f"[结论1] {encoder_label} 的 PSNR 分布方差更大（标准差更高），说明存在特别高和特别低的情况")
    elif vae_std > encoder_std * 1.2:
        print("[结论1] VAE 的 PSNR 分布方差更大")
    else:
        print("[结论1] 两种方法的 PSNR 分布方差相近")
    
    if (encoder_max - encoder_min) > (vae_max - vae_min) * 1.2:
        print(f"[结论2] {encoder_label} 的 PSNR 范围更大，说明会产生特别高和特别低的值")
    elif (vae_max - vae_min) > (encoder_max - encoder_min) * 1.2:
        print("[结论2] VAE 的 PSNR 范围更大")
    else:
        print("[结论2] 两种方法的 PSNR 范围相近")
    
    if correlation > 0.7:
        print("[结论3] 两种方法的相关性较高，说明好重建的场景在两种方法下都表现较好")
    elif correlation > 0.5:
        print("[结论3] 两种方法的相关性中等，存在一定的一致性但也有差异")
    else:
        print("[结论3] 两种方法的相关性较低，说明场景的难易程度在两种方法下可能不同")
    
    consistency_rate = (len(both_high) + len(both_low)) / len(scene_ids) * 100
    if consistency_rate > 70:
        print("[结论4] 一致性较高，说明好重建的场景在两种方法下都高PSNR，不好重建的都低PSNR")
    elif consistency_rate > 50:
        print("[结论4] 一致性中等，存在一些不一致的场景")
    else:
        print("[结论4] 一致性较低，说明两种方法对场景难易程度的判断存在较大差异")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="对比编码器和VAE的重建结果")
    parser.add_argument(
        "--encoder_type",
        type=str,
        choices=["dolp", "joint"],
        default="dolp",
        help="编码器类型: 'dolp' (DoLP专用编码器) 或 'joint' (联合编码器)"
    )
    parser.add_argument(
        "--encoder_json",
        type=str,
        default=None,
        help="编码器结果JSON文件路径（如果不提供，根据encoder_type自动选择）"
    )
    parser.add_argument(
        "--vae_json",
        type=str,
        default="重要/vae_validation_sorted.json",
        help="VAE结果JSON文件路径"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="输出JSON文件路径（如果不提供，根据encoder_type自动生成）"
    )
    
    args = parser.parse_args()
    
    # 根据encoder_type自动选择文件
    if args.encoder_json is None:
        if args.encoder_type == "dolp":
            encoder_json = Path("重要/batch_validation_results.json")
        else:  # joint
            encoder_json = Path("重要/quan_batch_validation_results.json")
    else:
        encoder_json = Path(args.encoder_json)
    
    if args.output_json is None:
        if args.encoder_type == "dolp":
            output_json = Path("重要/dolp_vae_comparison.json")
        else:  # joint
            output_json = Path("重要/joint_vae_comparison.json")
    else:
        output_json = Path(args.output_json)
    
    vae_json = Path(args.vae_json)
    
    encoder_name = "DoLP专用编码器" if args.encoder_type == "dolp" else "联合编码器"
    print("=" * 100)
    print(f"{encoder_name} vs VAE Zero-shot 对比分析")
    print("=" * 100)
    
    # 加载数据
    print(f"\n正在加载数据...")
    print(f"  编码器文件: {encoder_json}")
    print(f"  VAE文件: {vae_json}")
    
    encoder_data = load_dolp_encoder_results(encoder_json)  # 使用同一个加载函数
    vae_data = load_vae_results(vae_json)
    
    # 排序编码器结果
    print(f"正在排序 {encoder_name} 结果...")
    if args.encoder_type == "dolp":
        encoder_sorted = sort_dolp_encoder_results(encoder_data)
    else:  # joint
        encoder_sorted = sort_joint_encoder_results(encoder_data)
    
    # 提取 VAE 的 DoLP PSNR
    print("正在提取 VAE 结果...")
    vae_psnr = extract_vae_dolp_psnr(vae_data)
    
    # 对比分析
    print("正在进行对比分析...\n")
    compare_results(encoder_sorted, vae_psnr, output_json, encoder_type=args.encoder_type)


if __name__ == "__main__":
    main()

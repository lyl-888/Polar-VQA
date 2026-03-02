#!/usr/bin/env python3
"""
Stage 3 数据集拆分脚本：按场景分割训练集、验证集和测试集

功能：
1. 读取 Stage 3 数据（支持 positive/physics/negative 或 content/detail/spatial）
2. 按场景ID分组，提取测试集（可指定场景）
3. 剩余场景按比例分成训练集和验证集
4. 自动检测并保持数据分布（subtype 的比例）
5. 保存训练集、验证集和测试集到不同的文件

使用方法：
    python split_train_val.py \
        --input stage3_all_scenes_full_new.json \
        --train_output train_stage3_qwen.json \
        --val_output val_stage3_qwen.json \
        --test_output test_stage3_qwen.json \
        --val_ratio 0.1 \
        --test_scenes 05 06 08 18 26 12 35
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple


def load_data(input_file: str) -> List[Dict]:
    """加载 JSON 数据"""
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    print(f"📖 读取数据文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"输入文件格式错误：期望 list，得到 {type(data)}")
    
    print(f"   ✓ 读取 {len(data)} 条样本")
    return data


def analyze_data(data: List[Dict]) -> Dict:
    """分析数据分布"""
    stats = {
        'total': len(data),
        'by_scene': defaultdict(int),
        'by_subtype': defaultdict(int),
        'scenes': set(),
    }
    
    for item in data:
        scene_id = item.get('scene_id', 'unknown')
        subtype = item.get('subtype', 'unknown')
        stats['by_scene'][scene_id] += 1
        stats['by_subtype'][subtype] += 1
        stats['scenes'].add(scene_id)
    
    return stats


def split_by_scene_with_test(
    data: List[Dict], 
    val_ratio: float,
    test_scenes: List[str] = ['05', '23', '37', '39']
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    按场景ID分割数据（保持场景完整性），包含测试集
    
    功能：
    1. 先提取测试集场景（05, 23, 37, 39）
    2. 剩余场景按比例分成训练集和验证集
    
    Args:
        data: 数据列表
        val_ratio: 验证集比例（应用于排除测试集后的剩余场景）
        test_scenes: 测试集场景ID列表
    
    Returns:
        (train_data, val_data, test_data) 元组
    """
    print(f"\n🔄 按场景ID分割数据（包含测试集）...")
    print(f"   - 验证集比例: {val_ratio:.1%}（应用于排除测试集后的剩余场景）")
    
    # 按场景ID分组
    scene_groups = defaultdict(list)
    for item in data:
        scene_id = item.get('scene_id', 'unknown')
        # 统一转换为字符串格式，处理可能的数字格式
        scene_id_str = str(scene_id).zfill(2) if scene_id != 'unknown' else 'unknown'
        scene_groups[scene_id_str].append(item)
    
    # 统一测试场景格式
    test_scenes_set = set([str(s).zfill(2) for s in test_scenes])
    
    print(f"   - 总场景数: {len(scene_groups)}")
    print(f"   - 测试集场景: {sorted(test_scenes_set)}")
    
    # 分离测试集场景
    remaining_scenes = [s for s in scene_groups.keys() if s not in test_scenes_set and s != 'unknown']
    test_scenes_found = [s for s in test_scenes_set if s in scene_groups]
    test_scenes_missing = [s for s in test_scenes_set if s not in scene_groups]
    
    if test_scenes_missing:
        print(f"   ⚠️  警告: 以下测试场景在数据中未找到: {test_scenes_missing}")
    
    print(f"   - 剩余场景数: {len(remaining_scenes)}")
    
    # 计算测试集样本数（用于平衡）
    test_sample_count = sum(len(scene_groups[s]) for s in test_scenes_found)
    print(f"   - 测试集样本数: {test_sample_count}")
    
    # 显示测试集各场景的样本数
    if test_scenes_found:
        print(f"\n   📊 测试集各场景样本数:")
        for scene_id in sorted(test_scenes_found):
            count = len(scene_groups[scene_id])
            print(f"     场景 {scene_id}: {count} 个样本")
    
    # 从剩余场景中按比例拆分训练集和验证集
    val_scene_count = max(1, int(len(remaining_scenes) * val_ratio))
    
    # 随机选择验证集场景
    random.shuffle(remaining_scenes)
    val_scene_ids = set(remaining_scenes[:val_scene_count])
    train_scene_ids = set(remaining_scenes[val_scene_count:])
    
    # 计算验证集样本数
    val_sample_count = sum(len(scene_groups[s]) for s in val_scene_ids)
    
    print(f"   - 验证集场景数: {len(val_scene_ids)}")
    print(f"   - 验证集样本数: {val_sample_count}")
    print(f"   - 训练集场景数: {len(train_scene_ids)}")
    
    # 显示验证集各场景的样本数
    if val_scene_ids:
        print(f"\n   📊 验证集各场景样本数:")
        for scene_id in sorted(val_scene_ids):
            count = len(scene_groups[scene_id])
            print(f"     场景 {scene_id}: {count} 个样本")
    
    # 分割数据
    train_data = []
    val_data = []
    test_data = []
    
    # 测试集
    for scene_id in test_scenes_found:
        test_data.extend(scene_groups[scene_id])
    
    # 训练集
    for scene_id in train_scene_ids:
        train_data.extend(scene_groups[scene_id])
    
    # 验证集
    for scene_id in val_scene_ids:
        val_data.extend(scene_groups[scene_id])
    
    # 处理无场景ID的数据（默认分配到训练集）
    if 'unknown' in scene_groups:
        train_data.extend(scene_groups['unknown'])
        print(f"   ⚠️  警告: {len(scene_groups['unknown'])} 个无场景ID的样本已分配到训练集")
    
    return train_data, val_data, test_data


def split_random(data: List[Dict], val_ratio: float) -> Tuple[List[Dict], List[Dict]]:
    """
    随机分割数据
    
    优点：数据分布均匀
    缺点：同一场景的数据可能同时出现在训练集和验证集
    """
    print(f"\n🔄 随机分割数据（验证集比例: {val_ratio:.1%}）...")
    
    # 随机打乱
    random.shuffle(data)
    
    # 计算分割点
    val_count = int(len(data) * val_ratio)
    val_count = max(1, val_count)  # 至少1条
    
    val_data = data[:val_count]
    train_data = data[val_count:]
    
    return train_data, val_data


def split_by_count(data: List[Dict], val_count: int) -> Tuple[List[Dict], List[Dict]]:
    """
    按数量分割数据（随机）
    """
    print(f"\n🔄 按数量分割数据（验证集数量: {val_count}）...")
    
    # 随机打乱
    random.shuffle(data)
    
    val_count = min(val_count, len(data) - 1)  # 确保至少保留1条训练数据
    val_data = data[:val_count]
    train_data = data[val_count:]
    
    return train_data, val_data


def print_statistics(
    train_data: List[Dict], 
    val_data: List[Dict], 
    test_data: List[Dict],
    original_stats: Dict
):
    """打印详细统计信息"""
    print("\n" + "=" * 80)
    print("📊 分割后统计信息")
    print("=" * 80)
    
    total_split = len(train_data) + len(val_data) + len(test_data)
    total_original = original_stats['total']
    
    print(f"\n📦 样本数量统计:")
    print(f"   {'数据集':<12} {'样本数':<12} {'占比':<12} {'场景数':<12}")
    print(f"   {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    train_pct = (len(train_data) / total_original * 100) if total_original > 0 else 0
    val_pct = (len(val_data) / total_original * 100) if total_original > 0 else 0
    test_pct = (len(test_data) / total_original * 100) if total_original > 0 else 0
    
    train_scenes = set(str(item.get('scene_id', 'unknown')).zfill(2) if item.get('scene_id') != 'unknown' else 'unknown' for item in train_data)
    val_scenes = set(str(item.get('scene_id', 'unknown')).zfill(2) if item.get('scene_id') != 'unknown' else 'unknown' for item in val_data)
    test_scenes = set(str(item.get('scene_id', 'unknown')).zfill(2) if item.get('scene_id') != 'unknown' else 'unknown' for item in test_data)
    
    print(f"   {'训练集':<12} {len(train_data):<12} {train_pct:>8.2f}%   {len(train_scenes):<12}")
    print(f"   {'验证集':<12} {len(val_data):<12} {val_pct:>8.2f}%   {len(val_scenes):<12}")
    print(f"   {'测试集':<12} {len(test_data):<12} {test_pct:>8.2f}%   {len(test_scenes):<12}")
    print(f"   {'总计':<12} {total_split:<12} {100.0:>8.2f}%   {'-':<12}")
    
    if total_original != total_split:
        print(f"\n   ⚠️  数量不匹配，差异: {abs(total_original - total_split)} 个样本")
    
    # 按 subtype 统计（自动检测所有 subtype）
    print(f"\n📊 按 subtype 分布统计:")
    print(f"   {'subtype':<12} {'训练集':<12} {'验证集':<12} {'测试集':<12} {'总计':<12}")
    print(f"   {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    # 自动检测所有 subtype（从原始数据中）
    all_subtypes = sorted(set(item.get('subtype', 'unknown') for item in train_data + val_data + test_data))
    for subtype in all_subtypes:
        train_count = sum(1 for item in train_data if item.get('subtype') == subtype)
        val_count = sum(1 for item in val_data if item.get('subtype') == subtype)
        test_count = sum(1 for item in test_data if item.get('subtype') == subtype)
        total_count = train_count + val_count + test_count
        print(f"   {subtype:<12} {train_count:<12} {val_count:<12} {test_count:<12} {total_count:<12}")
    
    # 按场景统计（详细）
    print(f"\n📋 场景分布详情:")
    print(f"   - 训练集场景数: {len(train_scenes)}")
    if len(train_scenes) <= 20:
        print(f"     场景列表: {sorted([s for s in train_scenes if s != 'unknown'])}")
    else:
        print(f"     场景列表（前10个）: {sorted([s for s in train_scenes if s != 'unknown'])[:10]} ...")
    
    print(f"   - 验证集场景数: {len(val_scenes)}")
    print(f"     场景列表: {sorted([s for s in val_scenes if s != 'unknown'])}")
    
    print(f"   - 测试集场景数: {len(test_scenes)}")
    print(f"     场景列表: {sorted([s for s in test_scenes if s != 'unknown'])}")
    
    # 检查重叠
    train_val_overlap = train_scenes & val_scenes
    train_test_overlap = train_scenes & test_scenes
    val_test_overlap = val_scenes & test_scenes
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print(f"\n   ⚠️  场景重叠警告:")
        if train_val_overlap:
            print(f"     训练集与验证集重叠: {sorted(train_val_overlap)}")
        if train_test_overlap:
            print(f"     训练集与测试集重叠: {sorted(train_test_overlap)}")
        if val_test_overlap:
            print(f"     验证集与测试集重叠: {sorted(val_test_overlap)}")
    else:
        print(f"\n   ✅ 场景无重叠，拆分正确")


def save_data(data: List[Dict], output_file: str):
    """保存数据到 JSON 文件"""
    output_path = Path(output_file)
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 保存数据到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"   ✓ 保存成功，共 {len(data)} 条样本")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3 数据集拆分脚本：按场景分割训练集、验证集和测试集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python split_train_val.py \\
      --input stage3_all_scenes_full_new.json \\
      --train_output train_stage3_qwen.json \\
      --val_output val_stage3_qwen.json \\
      --test_output test_stage3_qwen.json \\
      --val_ratio 0.1 \\
      --test_scenes 05 06 08 18 26 12 35
        """
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入文件路径（必需）"
    )
    parser.add_argument(
        "--train_output",
        type=str,
        default="train_stage3_qwen.json",
        help="训练集输出文件路径（默认: train_stage3_qwen.json）"
    )
    parser.add_argument(
        "--val_output",
        type=str,
        default="val_stage3_qwen.json",
        help="验证集输出文件路径（默认: val_stage3_qwen.json）"
    )
    parser.add_argument(
        "--test_output",
        type=str,
        default="test_stage3_qwen.json",
        help="测试集输出文件路径（默认: test_stage3_qwen.json）"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="验证集比例（0.0-1.0），默认 0.1（10%%）。此比例应用于排除测试集后的剩余场景"
    )
    parser.add_argument(
        "--test_scenes",
        type=str,
        nargs='+',
        default=['05', '23', '37', '39'],
        help="测试集场景ID列表，默认: 05 23 37 39"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42）"
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    print("=" * 80)
    print("Stage 3 数据集拆分脚本（包含测试集）")
    print("=" * 80)
    print(f"✓ 随机种子: {args.seed}")
    
    # 加载数据
    data = load_data(args.input)
    
    # 分析原始数据
    print("\n📊 原始数据统计:")
    original_stats = analyze_data(data)
    print(f"   - 总样本数: {original_stats['total']}")
    print(f"   - 总场景数: {len(original_stats['scenes'])}")
    print(f"   - 按 subtype 分布:")
    for subtype, count in sorted(original_stats['by_subtype'].items()):
        pct = (count / original_stats['total'] * 100) if original_stats['total'] > 0 else 0
        print(f"     {subtype}: {count} 条 ({pct:.1f}%)")
    
    # 分割数据（按场景，包含测试集）
    train_data, val_data, test_data = split_by_scene_with_test(
        data=data,
        val_ratio=args.val_ratio,
        test_scenes=args.test_scenes
    )
    
    # 打印统计信息
    print_statistics(train_data, val_data, test_data, original_stats)
    
    # 保存数据
    print(f"\n💾 正在保存结果...")
    save_data(train_data, args.train_output)
    save_data(val_data, args.val_output)
    save_data(test_data, args.test_output)
    
    print("\n" + "=" * 80)
    print("✅ 数据分割完成！")
    print("=" * 80)
    print(f"   - 训练集: {args.train_output} ({len(train_data)} 个样本)")
    print(f"   - 验证集: {args.val_output} ({len(val_data)} 个样本)")
    print(f"   - 测试集: {args.test_output} ({len(test_data)} 个样本)")


if __name__ == "__main__":
    main()


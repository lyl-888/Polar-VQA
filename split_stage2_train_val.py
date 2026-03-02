"""
Stage 1 数据集拆分脚本：按场景拆分训练集、验证集和测试集

功能：
1. 读取 stage1_captions_qwen.json
2. 按 scene_id 分组
3. 将场景 05, 23, 37 单独提取为测试集
4. 剩余场景按 9:1 分成训练集和验证集
5. 保存为 train_stage1_data.json, val_stage1_data.json, test_stage1_data.json

拆分策略：
- 测试集：场景 05, 23, 37（固定）
- 训练集：剩余场景的 90%
- 验证集：剩余场景的 10%
"""

import json
from pathlib import Path
from collections import defaultdict
import argparse
import random


def load_json(file_path: str) -> list:
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: list, file_path: str):
    """保存 JSON 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✓ 已保存: {file_path} ({len(data)} 个样本)")


def split_by_scene_with_test(
    data: list,
    test_scenes: list = ['05', '23', '37'],
    train_ratio: float = 0.9,
    random_seed: int = 42,
) -> tuple:
    """
    按场景拆分数据，包含测试集
    
    Args:
        data: 数据列表
        test_scenes: 测试集场景ID列表（默认：['05', '23', '37']）
        train_ratio: 训练集比例（剩余场景中，默认 0.9，即 90% 训练，10% 验证）
        random_seed: 随机种子（用于可复现性）
    
    Returns:
        (train_data, val_data, test_data) 元组
    """
    # 按 scene_id 分组
    scenes_dict = defaultdict(list)
    for item in data:
        scene_id = item.get('scene_id')
        if scene_id is not None:
            # 统一转换为字符串格式，处理可能的数字格式
            scene_id_str = str(scene_id).zfill(2)  # 确保是两位数字符串，如 '05', '23'
            scenes_dict[scene_id_str].append(item)
        else:
            # 如果没有 scene_id，放到一个特殊组
            scenes_dict['_no_scene'].append(item)
    
    # 获取所有场景 ID（排除无场景ID的）
    all_scenes = sorted([s for s in scenes_dict.keys() if s != '_no_scene'])
    
    # 统一测试场景格式
    test_scenes_set = set([str(s).zfill(2) for s in test_scenes])
    
    print("=" * 80)
    print("📊 数据统计（拆分前）")
    print("=" * 80)
    print(f"  - 总样本数: {len(data)}")
    print(f"  - 场景数量: {len(all_scenes)}")
    print(f"  - 无场景ID的样本: {len(scenes_dict.get('_no_scene', []))}")
    
    # 打印每个场景的样本数（详细统计）
    print(f"\n📋 各场景样本数（按场景ID排序）:")
    scene_counts = []
    for scene_id in sorted(all_scenes):
        count = len(scenes_dict[scene_id])
        scene_counts.append((scene_id, count))
        status = " [测试集]" if scene_id in test_scenes_set else ""
        print(f"  - scene_{scene_id}: {count:4d} 个样本{status}")
    
    # 统计测试场景
    test_scenes_found = [s for s in test_scenes_set if s in all_scenes]
    test_scenes_missing = [s for s in test_scenes_set if s not in all_scenes]
    
    if test_scenes_missing:
        print(f"\n⚠️  警告: 以下测试场景在数据中未找到: {test_scenes_missing}")
    
    # 分离测试集场景
    remaining_scenes = [s for s in all_scenes if s not in test_scenes_set]
    
    print(f"\n🔀 场景分组:")
    print(f"  - 测试集场景: {sorted(test_scenes_found)} ({len(test_scenes_found)} 个场景)")
    print(f"  - 剩余场景: {len(remaining_scenes)} 个场景")
    
    # 从剩余场景中按比例拆分训练集和验证集
    random.seed(random_seed)
    shuffled_remaining = remaining_scenes.copy()
    random.shuffle(shuffled_remaining)
    
    num_val_scenes = max(1, int(len(remaining_scenes) * (1 - train_ratio)))
    val_scene_set = set(shuffled_remaining[:num_val_scenes])
    train_scene_set = set(shuffled_remaining[num_val_scenes:])
    
    print(f"\n🎲 剩余场景拆分 (train_ratio={train_ratio}, seed={random_seed}):")
    print(f"  - 训练场景数: {len(train_scene_set)}")
    print(f"  - 验证场景数: {len(val_scene_set)}")
    print(f"  - 训练场景: {sorted(train_scene_set)}")
    print(f"  - 验证场景: {sorted(val_scene_set)}")
    
    # 分配数据
    train_data = []
    val_data = []
    test_data = []
    
    # 测试集
    for scene_id in test_scenes_found:
        test_data.extend(scenes_dict[scene_id])
    
    # 训练集
    for scene_id in train_scene_set:
        train_data.extend(scenes_dict[scene_id])
    
    # 验证集
    for scene_id in val_scene_set:
        val_data.extend(scenes_dict[scene_id])
    
    # 处理无场景ID的数据（默认分配到训练集）
    if '_no_scene' in scenes_dict:
        train_data.extend(scenes_dict['_no_scene'])
        print(f"\n⚠️  警告: {len(scenes_dict['_no_scene'])} 个无场景ID的样本已分配到训练集")
    
    # 详细统计输出
    print("\n" + "=" * 80)
    print("✅ 拆分结果统计")
    print("=" * 80)
    
    total_with_test = len(train_data) + len(val_data) + len(test_data)
    total_original = len(data)
    
    print(f"\n📊 样本数量统计:")
    print(f"  - 原始数据总数: {total_original:6d} 个样本")
    print(f"  - 拆分后总数:   {total_with_test:6d} 个样本")
    if total_original != total_with_test:
        print(f"  ⚠️  数量不匹配，差异: {abs(total_original - total_with_test)} 个样本")
    
    print(f"\n📦 各数据集详细统计:")
    print(f"  {'数据集':<12} {'样本数':<10} {'占比':<10} {'场景数':<10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    
    train_pct = (len(train_data) / total_original * 100) if total_original > 0 else 0
    val_pct = (len(val_data) / total_original * 100) if total_original > 0 else 0
    test_pct = (len(test_data) / total_original * 100) if total_original > 0 else 0
    
    print(f"  {'训练集':<12} {len(train_data):<10} {train_pct:>6.2f}%   {len(train_scene_set):<10}")
    print(f"  {'验证集':<12} {len(val_data):<10} {val_pct:>6.2f}%   {len(val_scene_set):<10}")
    print(f"  {'测试集':<12} {len(test_data):<10} {test_pct:>6.2f}%   {len(test_scenes_found):<10}")
    
    print(f"\n📋 测试集场景详情:")
    for scene_id in sorted(test_scenes_found):
        count = len(scenes_dict[scene_id])
        print(f"  - scene_{scene_id}: {count:4d} 个样本")
    
    print(f"\n📋 训练集场景详情（前10个）:")
    for scene_id in sorted(train_scene_set)[:10]:
        count = len(scenes_dict[scene_id])
        print(f"  - scene_{scene_id}: {count:4d} 个样本")
    if len(train_scene_set) > 10:
        print(f"  ... (还有 {len(train_scene_set) - 10} 个训练场景)")
    
    print(f"\n📋 验证集场景详情:")
    for scene_id in sorted(val_scene_set):
        count = len(scenes_dict[scene_id])
        print(f"  - scene_{scene_id}: {count:4d} 个样本")
    
    return train_data, val_data, test_data


def main():
    parser = argparse.ArgumentParser(description="Stage 1 数据集拆分脚本（包含测试集）")
    parser.add_argument(
        "--input_file",
        type=str,
        default="/openbayes/home/train/stage1_captions_qwen.json",
        help="输入 JSON 文件路径"
    )
    parser.add_argument(
        "--train_output",
        type=str,
        default="train_stage1_data.json",
        help="训练集输出文件路径"
    )
    parser.add_argument(
        "--val_output",
        type=str,
        default="val_stage1_data.json",
        help="验证集输出文件路径"
    )
    parser.add_argument(
        "--test_output",
        type=str,
        default="test_stage1_data.json",
        help="测试集输出文件路径"
    )
    parser.add_argument(
        "--test_scenes",
        type=str,
        nargs='+',
        default=['05', '23', '37'],
        help="测试集场景ID列表，默认: 05 23 37"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="训练集比例（0.0-1.0），默认 0.9（90%% 训练，10%% 验证）。此比例应用于排除测试集后的剩余场景"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="随机种子（用于可复现性），默认 42"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Stage 1 数据集拆分脚本（包含测试集）")
    print("=" * 80)
    
    # 1. 加载数据
    print(f"\n📂 正在加载数据: {args.input_file}")
    if not Path(args.input_file).exists():
        raise FileNotFoundError(f"输入文件不存在: {args.input_file}")
    
    data = load_json(args.input_file)
    print(f"✓ 加载完成: {len(data)} 个样本")
    
    # 2. 拆分数据
    print(f"\n🔀 正在拆分数据...")
    train_data, val_data, test_data = split_by_scene_with_test(
        data=data,
        test_scenes=args.test_scenes,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
    )
    
    # 3. 保存结果
    print(f"\n💾 正在保存结果...")
    save_json(train_data, args.train_output)
    save_json(val_data, args.val_output)
    save_json(test_data, args.test_output)
    
    print("\n" + "=" * 80)
    print("✅ 拆分完成！")
    print("=" * 80)
    print(f"训练集: {args.train_output} ({len(train_data)} 个样本)")
    print(f"验证集: {args.val_output} ({len(val_data)} 个样本)")
    print(f"测试集: {args.test_output} ({len(test_data)} 个样本)")


if __name__ == "__main__":
    main()

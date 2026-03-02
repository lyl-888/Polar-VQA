"""
对验证结果 JSON 文件进行分类和排序
- 根据 rgb_path 的目录路径进行分组（如 /openbayes/input/input0/rgb/30/）
- 在每个组内，按照文件名中的序号从小到大排序
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def extract_dir_and_index(rgb_path: str) -> Tuple[str, int]:
    """
    从 rgb_path 中提取目录路径和文件名序号
    
    Args:
        rgb_path: 例如 "/openbayes/input/input0/rgb/30/0010_rgb.png"
    
    Returns:
        (目录路径, 序号) 例如 ("/openbayes/input/input0/rgb/30/", 10)
    """
    path_obj = Path(rgb_path)
    # 获取目录路径（包含末尾的 /）
    dir_path = str(path_obj.parent) + "/"
    
    # 从文件名中提取序号（如 "0010_rgb.png" -> 10）
    stem = path_obj.stem  # "0010_rgb"
    if "_rgb" in stem:
        base_name = stem.replace("_rgb", "")  # "0010"
    else:
        base_name = stem
    
    try:
        index = int(base_name)
    except ValueError:
        # 如果无法转换为整数，使用 0 作为默认值
        index = 0
    
    return dir_path, index


def sort_results_by_path(input_file: str, output_file: str = None):
    """
    对验证结果进行分类和排序
    
    Args:
        input_file: 输入的 JSON 文件路径
        output_file: 输出的 JSON 文件路径（如果为 None，则覆盖原文件）
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    print(f"正在读取文件: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"JSON 文件应该是列表格式，但得到: {type(data)}")
    
    print(f"原始条目数: {len(data)}")
    
    # 按目录路径分组
    grouped: Dict[str, List[Tuple[int, Dict]]] = defaultdict(list)
    
    for item in data:
        rgb_path = item.get("rgb_path", "")
        if not rgb_path:
            print(f"警告: 条目缺少 rgb_path，跳过: {item.get('scene_id', 'N/A')}/{item.get('base_name', 'N/A')}")
            continue
        
        dir_path, index = extract_dir_and_index(rgb_path)
        grouped[dir_path].append((index, item))
    
    print(f"检测到 {len(grouped)} 个不同的目录")
    
    # 对每个组内的条目按序号排序
    sorted_data: List[Dict] = []
    
    # 先对目录路径进行排序（按场景ID）
    sorted_dirs = sorted(grouped.keys(), key=lambda x: Path(x).parts[-2] if len(Path(x).parts) >= 2 else x)
    
    for dir_path in sorted_dirs:
        items = grouped[dir_path]
        # 按序号排序
        items.sort(key=lambda x: x[0])
        # 只保留 item 部分
        for _, item in items:
            sorted_data.append(item)
    
    print(f"排序后条目数: {len(sorted_data)}")
    
    # 保存结果
    if output_file is None:
        output_file = str(input_path)
        # 备份原文件
        backup_file = str(input_path.with_suffix('.json.backup2'))
        print(f"\n正在备份原文件到: {backup_file}")
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    output_path = Path(output_file)
    print(f"\n正在保存排序后的结果到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_data, f, ensure_ascii=False, indent=2)
    
    print("排序完成！")
    print(f"  - 原始条目: {len(data)}")
    print(f"  - 排序后条目: {len(sorted_data)}")
    print(f"  - 分组目录数: {len(grouped)}")
    
    # 打印每个目录的统计信息
    print("\n各目录统计:")
    for dir_path in sorted_dirs[:10]:  # 只显示前10个
        count = len(grouped[dir_path])
        print(f"  {dir_path}: {count} 条")
    if len(sorted_dirs) > 10:
        print(f"  ... 还有 {len(sorted_dirs) - 10} 个目录")
    
    return sorted_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="对验证结果 JSON 文件进行分类和排序")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入的 JSON 文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出的 JSON 文件路径（如果未指定，则覆盖原文件并创建备份）"
    )
    
    args = parser.parse_args()
    
    sort_results_by_path(args.input, args.output)

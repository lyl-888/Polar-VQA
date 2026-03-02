"""
去重验证结果 JSON 文件
根据 scene_id, base_name, qa_type 三个字段去重，只保留第一个出现的条目
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


def deduplicate_results(input_file: str, output_file: str = None):
    """
    去重验证结果 JSON 文件
    
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
    
    original_count = len(data)
    print(f"原始条目数: {original_count}")
    
    # 使用 (scene_id, base_name, qa_type) 作为唯一标识
    seen_keys: Dict[Tuple[str, str, str], bool] = {}
    deduplicated: List[Dict] = []
    duplicates_count = 0
    
    for item in data:
        scene_id = str(item.get("scene_id", ""))
        base_name = str(item.get("base_name", ""))
        qa_type = str(item.get("qa_type", ""))
        
        key = (scene_id, base_name, qa_type)
        
        if key not in seen_keys:
            seen_keys[key] = True
            deduplicated.append(item)
        else:
            duplicates_count += 1
    
    print(f"去重后条目数: {len(deduplicated)}")
    print(f"删除重复条目数: {duplicates_count}")
    print(f"去重率: {duplicates_count / original_count * 100:.2f}%")
    
    # 保存结果
    if output_file is None:
        output_file = str(input_path)
        # 备份原文件
        backup_file = str(input_path.with_suffix('.json.backup'))
        print(f"\n正在备份原文件到: {backup_file}")
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    output_path = Path(output_file)
    print(f"\n正在保存去重后的结果到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(deduplicated, f, ensure_ascii=False, indent=2)
    
    print("去重完成！")
    print(f"  - 原始条目: {original_count}")
    print(f"  - 去重后条目: {len(deduplicated)}")
    print(f"  - 删除重复: {duplicates_count}")
    
    return deduplicated


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="去重验证结果 JSON 文件")
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
    
    deduplicate_results(args.input, args.output)

#!/usr/bin/env python3
"""
Stage 3 VQA 数据格式转换与合并脚本

功能：
1. 将第一种格式（stage3_vqa_visual_direct_merged.json）的 bbox 从像素坐标转换为归一化坐标
2. 在第一种格式的 prompt 中添加坐标信息
3. 将第二种格式（stage3_qa_pairs_all.json）转换为第一种格式（拆分成3条独立样本）
4. 合并两种格式的数据

使用方法：
    python convert_and_merge_stage3_data.py \
        --old_format stage3_vqa_visual_direct_merged_en.json \
        --new_format stage3_qa_pairs_qwen_1.json \
        --data_root /openbayes/input/input0 \
        --output merged_stage2_new_format.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image

def get_image_dimensions(image_path: Path) -> Optional[tuple]:
    """获取图像尺寸 (width, height)"""
    try:
        if not image_path.exists():
            # 尝试相对于 data_root 查找
            return None
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f"Warning: 无法读取图像尺寸 {image_path}: {e}")
        return None

def convert_bbox_to_normalized(bbox_pixel: List[int], img_width: int, img_height: int) -> List[float]:
    """
    将像素坐标的 bbox 转换为归一化坐标 [xmin, ymin, xmax, ymax]
    
    Args:
        bbox_pixel: [x, y, width, height]（像素坐标，这是第一种格式的标准格式）
        img_width: 图像宽度
        img_height: 图像高度
    
    Returns:
        [xmin, ymin, xmax, ymax] 归一化坐标 (0-1)
    """
    if len(bbox_pixel) == 4:
        # 第一种格式的 bbox 是 [x, y, width, height] 格式
        # 例如：[125, 329, 186, 24] 表示 x=125, y=329, width=186, height=24
        x, y, w, h = bbox_pixel
        
        # 转换为 [xmin, ymin, xmax, ymax] 归一化坐标
        xmin = x / img_width
        ymin = y / img_height
        xmax = (x + w) / img_width
        ymax = (y + h) / img_height
        
        # 确保在 [0, 1] 范围内
        xmin = max(0.0, min(1.0, xmin))
        ymin = max(0.0, min(1.0, ymin))
        xmax = max(0.0, min(1.0, xmax))
        ymax = max(0.0, min(1.0, ymax))
        
        return [round(xmin, 4), round(ymin, 4), round(xmax, 4), round(ymax, 4)]
    else:
        raise ValueError(f"Invalid bbox format: {bbox_pixel}, expected 4 values [x, y, width, height]")

def format_bbox_string_normalized(bbox_norm: List[float]) -> str:
    """格式化归一化 bbox 为字符串，用于 prompt"""
    return f"[{bbox_norm[0]:.3f}, {bbox_norm[1]:.3f}, {bbox_norm[2]:.3f}, {bbox_norm[3]:.3f}]"

def update_old_format_prompt(prompt: str, bbox_norm: List[float]) -> str:
    """在 prompt 中添加坐标信息"""
    bbox_str = format_bbox_string_normalized(bbox_norm)
    
    # 如果 prompt 中已经包含坐标格式（如 [0.123, 0.456, 0.789, 0.012]），则不重复添加
    import re
    if re.search(r'\[\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*\]', prompt):
        return prompt
    
    # 在 prompt 前面添加坐标信息
    # 根据不同的 prompt 类型，采用不同的添加方式
    if any(char >= '\u4e00' and char <= '\u9fff' for char in prompt):
        # 中文 prompt：在开头添加坐标
        return f"聚焦区域 {bbox_str}。{prompt}"
    else:
        # 英文 prompt：在开头添加坐标
        return f"Focus on region {bbox_str}. {prompt}"

def convert_old_format(old_data: List[Dict], data_root: Path) -> List[Dict]:
    """
    转换第一种格式：将 bbox 转换为归一化坐标，并在 prompt 中添加坐标
    
    Args:
        old_data: 第一种格式的数据列表
        data_root: 数据根目录
    
    Returns:
        转换后的数据列表
    """
    converted_data = []
    
    for item in old_data:
        new_item = {}
        
        # 保留必要字段
        new_item["image"] = item.get("image") or item.get("gt_path")  # GT 图像路径
        new_item["input_path"] = item.get("input_path", "")  # RGB 图像路径
        new_item["gt_path"] = item.get("gt_path") or item.get("image")  # GT 图像路径
        new_item["scene_id"] = item.get("scene_id", "")
        new_item["type"] = item.get("type", "typeB_visual")
        new_item["subtype"] = item.get("subtype", "")
        
        # 获取图像路径用于计算尺寸
        gt_path_str = new_item["gt_path"]
        if not gt_path_str:
            print(f"Warning: 跳过缺少图像路径的样本 (scene_id: {new_item.get('scene_id', 'unknown')})")
            continue
        
        # 构建完整路径（处理相对路径）
        if not Path(gt_path_str).is_absolute():
            gt_path = data_root / gt_path_str
        else:
            gt_path = Path(gt_path_str)
        
        # 获取图像尺寸
        img_dims = get_image_dimensions(gt_path)
        if img_dims is None:
            # 如果无法获取尺寸，尝试使用默认值或跳过
            print(f"Warning: 无法获取图像尺寸 {gt_path_str}，使用默认尺寸 1024x1024")
            img_width, img_height = 1024, 1024
        else:
            img_width, img_height = img_dims
        
        # 转换 bbox
        bbox_pixel = item.get("bbox")
        if not bbox_pixel:
            print(f"Warning: 跳过缺少 bbox 的样本: scene_id={new_item.get('scene_id', 'unknown')}")
            continue
        
        try:
            bbox_norm = convert_bbox_to_normalized(bbox_pixel, img_width, img_height)
            new_item["bbox_norm"] = bbox_norm
            # 显式确保不包含 bbox 字段（像素坐标），只保留 bbox_norm（归一化坐标）
            if "bbox" in new_item:
                del new_item["bbox"]
        except Exception as e:
            print(f"Warning: 转换 bbox 失败 {bbox_pixel} (图像尺寸: {img_width}x{img_height}): {e}")
            continue
        
        # 更新 prompt（在 human 的 value 中添加坐标）
        conversations = item.get("conversations", [])
        if conversations and len(conversations) > 0:
            human_value = conversations[0].get("value", "")
            if human_value:
                updated_prompt = update_old_format_prompt(human_value, bbox_norm)
                new_conversations = conversations.copy()
                new_conversations[0] = {
                    "from": "human",
                    "value": updated_prompt
                }
                new_item["conversations"] = new_conversations
        else:
            print(f"Warning: 跳过缺少 conversations 的样本: scene_id={new_item.get('scene_id', 'unknown')}")
            continue
        
        converted_data.append(new_item)
    
    return converted_data

def convert_new_format_to_old(new_data: List[Dict], data_root: Path, add_coords_to_all: bool = True) -> List[Dict]:
    """
    将第二种格式转换为第一种格式（拆分成3条独立样本）
    
    Args:
        new_data: 第二种格式的数据列表
        data_root: 数据根目录
        add_coords_to_all: 是否在所有问题类型中都添加坐标（默认 True，所有问题都包含坐标）
                          - True: 所有问题都包含坐标（默认，确保 Prompt 必须包含 [bbox]）
                          - False: 只有 Content 问题包含坐标，Detail 和 Spatial 不包含
    
    Returns:
        转换后的第一种格式数据列表
    """
    converted_data = []
    
    for item in new_data:
        scene_id = item.get("scene_id", "")
        image_path = item.get("image", "")  # RGB 路径
        gt_image_path = item.get("gt_image", "")  # GT 路径
        bbox_norm = item.get("bbox_norm", [0.0, 0.0, 1.0, 1.0])
        
        # 统一 bbox_norm 精度为 4 位小数（与第一种格式保持一致）
        bbox_norm_rounded = [round(coord, 4) for coord in bbox_norm]
        bbox_str = format_bbox_string_normalized(bbox_norm_rounded)
        
        base_sample = {
            "image": gt_image_path,  # 第一种格式使用 GT 作为 image 字段
            "input_path": image_path,  # RGB 路径（训练时使用）
            "gt_path": gt_image_path,  # GT 路径
            "scene_id": scene_id,
            "bbox_norm": bbox_norm_rounded,  # 归一化坐标 [xmin, ymin, xmax, ymax]，统一精度为4位小数
            "type": "typeB_visual",
        }
        
        # 辅助函数：根据配置决定是否添加坐标前缀
        def maybe_add_coords(question: str, should_add: bool) -> str:
            """如果 should_add=True，在问题前添加坐标前缀"""
            if not should_add:
                return question
            # 检查问题中是否已经包含坐标格式
            import re
            if re.search(r'\[\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*\]', question):
                return question  # 已经包含坐标，不重复添加
            # 添加坐标前缀
            if any(char >= '\u4e00' and char <= '\u9fff' for char in question):
                return f"聚焦区域 {bbox_str}。{question}"
            else:
                return f"Focus on region {bbox_str}. {question}"
        
        # 1. Content 类型（始终包含坐标）
        qa_content = item.get("qa_content")
        if qa_content and isinstance(qa_content, dict):
            content_sample = base_sample.copy()
            content_sample["subtype"] = "content"
            content_question = qa_content.get("q", "")
            # Content 问题始终包含坐标（用于建立区域理解）
            content_question_with_coords = maybe_add_coords(content_question, True)
            content_sample["conversations"] = [
                {
                    "from": "human",
                    "value": content_question_with_coords
                },
                {
                    "from": "gpt",
                    "value": qa_content.get("a", "")
                }
            ]
            converted_data.append(content_sample)
        
        # 2. Detail 类型（根据配置决定是否添加坐标）
        qa_detail = item.get("qa_detail")
        if qa_detail and isinstance(qa_detail, dict):
            detail_sample = base_sample.copy()
            detail_sample["subtype"] = "detail"
            detail_question = qa_detail.get("q", "")
            # Detail 问题根据配置决定是否包含坐标
            detail_question_with_coords = maybe_add_coords(detail_question, add_coords_to_all)
            detail_sample["conversations"] = [
                {
                    "from": "human",
                    "value": detail_question_with_coords
                },
                {
                    "from": "gpt",
                    "value": qa_detail.get("a", "")
                }
            ]
            converted_data.append(detail_sample)
        
        # 3. Spatial 类型（根据配置决定是否添加坐标）
        qa_spatial = item.get("qa_spatial")
        if qa_spatial and isinstance(qa_spatial, dict):
            spatial_sample = base_sample.copy()
            spatial_sample["subtype"] = "spatial"
            spatial_question = qa_spatial.get("q", "")
            # Spatial 问题根据配置决定是否包含坐标
            spatial_question_with_coords = maybe_add_coords(spatial_question, add_coords_to_all)
            spatial_sample["conversations"] = [
                {
                    "from": "human",
                    "value": spatial_question_with_coords
                },
                {
                    "from": "gpt",
                    "value": qa_spatial.get("a", "")
                }
            ]
            converted_data.append(spatial_sample)
    
    return converted_data

def main():
    parser = argparse.ArgumentParser(description="转换和合并 Stage 3 VQA 数据")
    parser.add_argument(
        "--old_format",
        type=str,
        required=True,
        help="第一种格式的 JSON 文件路径（stage3_vqa_visual_direct_merged.json）"
    )
    parser.add_argument(
        "--new_format",
        type=str,
        required=True,
        help="第二种格式的 JSON 文件路径（stage3_qa_pairs_all.json）"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/openbayes/input/input0",
        help="数据根目录（用于查找图像文件以获取尺寸）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="merged_stage3_data.json",
        help="输出合并后的 JSON 文件路径"
    )
    parser.add_argument(
        "--add_coords_to_all",
        action="store_true",
        default=True,
        help="是否在所有问题类型（Content、Detail、Spatial）中都添加坐标（默认 True，所有问题都包含坐标）"
    )
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    
    # 1. 读取第一种格式的数据
    print(f"📖 读取第一种格式数据: {args.old_format}")
    with open(args.old_format, "r", encoding="utf-8") as f:
        old_data = json.load(f)
    print(f"   ✓ 读取 {len(old_data)} 条样本")
    
    # 2. 转换第一种格式（bbox 归一化 + prompt 添加坐标）
    print(f"\n🔄 转换第一种格式（bbox 归一化 + prompt 添加坐标）...")
    converted_old_data = convert_old_format(old_data, data_root)
    print(f"   ✓ 转换完成，共 {len(converted_old_data)} 条样本")
    
    # 3. 读取第二种格式的数据
    print(f"\n📖 读取第二种格式数据: {args.new_format}")
    with open(args.new_format, "r", encoding="utf-8") as f:
        new_data = json.load(f)
    print(f"   ✓ 读取 {len(new_data)} 条样本")
    
    # 4. 转换第二种格式为第一种格式（拆分）
    print(f"\n🔄 转换第二种格式为第一种格式（拆分成独立样本）...")
    if args.add_coords_to_all:
        print(f"   ✓ 将在所有问题类型中都添加坐标（确保 Prompt 必须包含 [bbox]）")
    else:
        print(f"   ⚠ 注意: 只有 Content 问题包含坐标，Detail 和 Spatial 不包含")
    converted_new_data = convert_new_format_to_old(new_data, data_root, add_coords_to_all=args.add_coords_to_all)
    print(f"   ✓ 转换完成，共 {len(converted_new_data)} 条样本")
    
    # 5. 合并两种格式的数据（不去重）
    print(f"\n🔀 合并数据（不去重）...")
    merged_data = converted_old_data + converted_new_data
    print(f"   ✓ 合并完成，总计 {len(merged_data)} 条样本")
    
    # 6. 保存结果
    print(f"\n💾 保存合并后的数据到: {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    print(f"   ✓ 保存成功")
    
    # 6. 统计信息
    print(f"\n📊 统计信息:")
    print(f"   - 第一种格式（转换后）: {len(converted_old_data)} 条")
    print(f"   - 第二种格式（转换后）: {len(converted_new_data)} 条")
    print(f"   - 合并后总计: {len(merged_data)} 条")
    
    # 统计 subtype 分布
    subtype_counts = {}
    for item in merged_data:
        subtype = item.get("subtype", "unknown")
        subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
    
    print(f"\n   按 subtype 分布:")
    for subtype, count in sorted(subtype_counts.items()):
        print(f"     - {subtype}: {count} 条")

if __name__ == "__main__":
    main()


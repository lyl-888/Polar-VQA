#!/usr/bin/env python3
"""
自动检测 RGB 图像中的反光区域（眩光框）
输出归一化坐标 [xmin, ymin, xmax, ymax] 和对应的 RGB、Polar 图像路径

使用方法：
    python detect_glare_bboxes.py \
        --rgb_root /openbayes/input/input0/rgb_est \
        --gt_root /openbayes/input/input0/GT \
        --polar_root /openbayes/input/input0/polar \
        --scene_id 10 \
        --output glare_bboxes_scene10.json
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm


def detect_glare_bbox_diff(
    input_image_path: Path,
    gt_image_path: Path,
    threshold: int = 100,
    min_area: int = 500,
    max_bbox_ratio: float = 0.4,
) -> Optional[List[float]]:
    """
    使用计算机视觉方法检测眩光区域并返回边界框（差分方法）
    
    Args:
        input_image_path: 输入眩光图像路径（RGB，带眩光）
        gt_image_path: GT干净图像路径
        threshold: 差分阈值
        min_area: 最小眩光区域面积（像素）
        max_bbox_ratio: 最大边界框面积比例
        
    Returns:
        bbox_norm: [xmin, ymin, xmax, ymax]，归一化坐标（标准格式）；如果检测失败，返回 None
    """
    try:
        input_img = cv2.imread(str(input_image_path))
        gt_img = cv2.imread(str(gt_image_path))
        
        if input_img is None or gt_img is None:
            return None
        
        input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
        
        if input_gray.shape != gt_gray.shape:
            gt_gray = cv2.resize(gt_gray, (input_gray.shape[1], input_gray.shape[0]))
        
        # 计算绝对差分
        diff = cv2.absdiff(input_gray, gt_gray)
        
        # 应用阈值创建二值掩码
        _, binary_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        
        if contour_area < min_area:
            return None
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        rect_area = w * h
        
        h_img, w_img = input_gray.shape[:2]
        img_area = h_img * w_img
        
        if rect_area < min_area:
            return None
        
        if rect_area > (img_area * max_bbox_ratio):
            return None
        
        # 转换为归一化坐标 [xmin, ymin, xmax, ymax]
        xmin_n = float(x / w_img)
        xmax_n = float((x + w) / w_img)
        ymin_n = float(y / h_img)
        ymax_n = float((y + h) / h_img)
        
        xmin_n = float(np.clip(xmin_n, 0.0, 1.0))
        xmax_n = float(np.clip(xmax_n, 0.0, 1.0))
        ymin_n = float(np.clip(ymin_n, 0.0, 1.0))
        ymax_n = float(np.clip(ymax_n, 0.0, 1.0))
        
        return [round(xmin_n, 4), round(ymin_n, 4), round(xmax_n, 4), round(ymax_n, 4)]
        
    except Exception as e:
        print(f"Warning: 检测反光框失败 {input_image_path}: {e}")
        return None


def find_gt_image_path(
    scene_id: str,
    rgb_filename: str,
    rgb_root: Path,
    gt_root: Optional[Path] = None,
) -> Optional[Path]:
    """查找对应的 GT 图像路径"""
    prefix = rgb_filename.replace("_rgb.png", "")
    
    if gt_root is None:
        possible_gt_roots = [
            rgb_root.parent / "GT",
            rgb_root / ".." / "GT",
            Path("/openbayes/input/input0/GT"),
        ]
    else:
        possible_gt_roots = [Path(gt_root)]
    
    for gt_dir in possible_gt_roots:
        gt_scene_dir = gt_dir / scene_id
        possible_gt_names = [
            rgb_filename,
            f"{prefix}_rgb.png",
            f"0000_rgb.png",
        ]
        for gt_name in possible_gt_names:
            candidate_path = gt_scene_dir / gt_name
            if candidate_path.exists():
                return candidate_path
    
    return None


def get_polar_paths(polar_root: Path, scene_id: str, base_name: str) -> Dict[str, Path]:
    """
    根据 scene_id 和 base_name 推导偏振图像的四个通道路径
    
    Args:
        polar_root: 偏振图像根目录
        scene_id: 场景ID
        base_name: 基础文件名（不含扩展名，如 "0002"）
    
    Returns:
        包含四个偏振通道路径的字典
    """
    polar_scene_dir = polar_root / scene_id
    
    # 尝试三位数角度格式（推荐）
    polar_paths = {
        'I_0': polar_scene_dir / f"{base_name}_000.png",
        'I_45': polar_scene_dir / f"{base_name}_045.png",
        'I_90': polar_scene_dir / f"{base_name}_090.png",
        'I_135': polar_scene_dir / f"{base_name}_135.png",
    }
    
    if all(p.exists() for p in polar_paths.values()):
        return polar_paths
    
    # 尝试两位数角度格式
    polar_paths = {
        'I_0': polar_scene_dir / f"{base_name}_0.png",
        'I_45': polar_scene_dir / f"{base_name}_45.png",
        'I_90': polar_scene_dir / f"{base_name}_90.png",
        'I_135': polar_scene_dir / f"{base_name}_135.png",
    }
    
    return polar_paths


def detect_scene_glare_bboxes(
    rgb_root: str,
    gt_root: str,
    polar_root: str,
    scene_id: str,
    threshold: int = 100,
    min_area: int = 500,
    max_bbox_ratio: float = 0.4,
) -> List[Dict]:
    """
    检测某个场景中所有 RGB 图像的反光框
    
    Args:
        rgb_root: RGB 图像根目录
        gt_root: GT 图像根目录
        polar_root: 偏振图像根目录
        scene_id: 场景ID
        threshold: 差分阈值
        min_area: 最小眩光区域面积
        max_bbox_ratio: 最大边界框面积比例
    
    Returns:
        检测结果列表，每个元素包含：
        - rgb_path: RGB 图像路径（相对路径）
        - polar_paths: 偏振图像路径字典（相对路径）
        - bbox_norm: 归一化坐标 [xmin, ymin, xmax, ymax]
        - scene_id: 场景ID
        - base_name: 基础文件名
    """
    rgb_root_path = Path(rgb_root)
    gt_root_path = Path(gt_root)
    polar_root_path = Path(polar_root)
    
    scene_rgb_dir = rgb_root_path / scene_id
    if not scene_rgb_dir.exists():
        raise FileNotFoundError(f"RGB 场景目录不存在: {scene_rgb_dir}")
    
    # 获取所有 RGB 图像
    rgb_images = sorted(scene_rgb_dir.glob("*_rgb.png"))
    
    if not rgb_images:
        print(f"Warning: 场景 {scene_id} 中没有找到 RGB 图像")
        return []
    
    results = []
    
    for rgb_path in tqdm(rgb_images, desc=f"检测场景 {scene_id} 的反光框"):
        rgb_filename = rgb_path.name
        base_name = rgb_filename.replace("_rgb.png", "")
        
        # 查找对应的 GT 图像
        gt_path = find_gt_image_path(scene_id, rgb_filename, rgb_root_path, gt_root_path)
        if gt_path is None:
            print(f"Warning: 未找到 GT 图像，跳过: {rgb_filename}")
            continue
        
        # 检测反光框
        bbox_norm = detect_glare_bbox_diff(
            rgb_path,
            gt_path,
            threshold=threshold,
            min_area=min_area,
            max_bbox_ratio=max_bbox_ratio,
        )
        
        # 获取偏振图像路径
        polar_paths_dict = get_polar_paths(polar_root_path, scene_id, base_name)
        
        # 检查偏振图像是否存在
        missing_polar = [name for name, path in polar_paths_dict.items() if not path.exists()]
        if missing_polar:
            print(f"Warning: 偏振图像缺失 {rgb_filename}: {missing_polar}")
            continue
        
        # 转换为相对路径（相对于数据根目录）
        data_root = rgb_root_path.parent
        rgb_path_rel = str(rgb_path.relative_to(data_root))
        polar_paths_rel = {
            name: str(path.relative_to(data_root))
            for name, path in polar_paths_dict.items()
        }
        
        result = {
            "rgb_path": rgb_path_rel,
            "polar_paths": polar_paths_rel,
            "bbox_norm": bbox_norm,  # 如果检测失败，为 None
            "scene_id": scene_id,
            "base_name": base_name,
        }
        
        results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="自动检测 RGB 图像中的反光区域")
    parser.add_argument(
        "--rgb_root",
        type=str,
        required=True,
        help="RGB 图像根目录（如 /openbayes/input/input0/rgb_est）"
    )
    parser.add_argument(
        "--gt_root",
        type=str,
        default="/openbayes/input/input0/GT",
        help="GT 图像根目录"
    )
    parser.add_argument(
        "--polar_root",
        type=str,
        default="/openbayes/input/input0/polar",
        help="偏振图像根目录"
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        required=True,
        help="场景 ID（如 '10'）"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出 JSON 文件路径"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=100,
        help="差分阈值（默认 100）"
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=500,
        help="最小眩光区域面积（像素，默认 500）"
    )
    parser.add_argument(
        "--max_bbox_ratio",
        type=float,
        default=0.4,
        help="最大边界框面积比例（默认 0.4）"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("反光框检测")
    print("=" * 80)
    print(f"RGB 根目录: {args.rgb_root}")
    print(f"GT 根目录: {args.gt_root}")
    print(f"偏振根目录: {args.polar_root}")
    print(f"场景 ID: {args.scene_id}")
    print(f"输出文件: {args.output}")
    print()
    
    # 检测反光框
    results = detect_scene_glare_bboxes(
        rgb_root=args.rgb_root,
        gt_root=args.gt_root,
        polar_root=args.polar_root,
        scene_id=args.scene_id,
        threshold=args.threshold,
        min_area=args.min_area,
        max_bbox_ratio=args.max_bbox_ratio,
    )
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 统计信息
    total = len(results)
    with_glare = sum(1 for r in results if r.get("bbox_norm") is not None)
    without_glare = total - with_glare
    
    print("\n" + "=" * 80)
    print("检测完成")
    print("=" * 80)
    print(f"总图像数: {total}")
    print(f"检测到反光: {with_glare}")
    print(f"未检测到反光: {without_glare}")
    print(f"结果已保存到: {output_path.resolve()}")


if __name__ == "__main__":
    main()


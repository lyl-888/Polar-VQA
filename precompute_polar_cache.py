import argparse
import json
import os
from pathlib import Path
import multiprocessing as mp

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

# 从项目根目录导入 process_polar_images
import sys
sys.path.insert(0, os.path.dirname(__file__))
try:
    from dataset_common import process_polar_images
except ImportError as exc:
    raise ImportError(
        "未找到 dataset_common.process_polar_images。"
        "请确认 dataset_common.py 位于项目根目录。"
    ) from exc


def resolve_polar_paths(sample, data_root, polar_folder, image_folder):
    """解析偏振图像路径（优先使用 polar/ 与 rgb/，不使用 crop）。"""
    # 1) JSON 中直接给出 polar_paths
    polar_paths_json = sample.get("polar_paths", {})
    if polar_paths_json:
        polar_paths = {}
        for angle_name in ["I_0", "I_45", "I_90", "I_135"]:
            polar_path_str = polar_paths_json.get(angle_name)
            if not polar_path_str:
                break
            if os.path.isabs(polar_path_str):
                polar_paths[angle_name] = Path(polar_path_str)
            else:
                if data_root and polar_path_str.startswith("polar/"):
                    polar_paths[angle_name] = Path(data_root) / polar_path_str
                elif polar_folder:
                    polar_paths[angle_name] = Path(polar_folder) / polar_path_str
                elif data_root:
                    polar_paths[angle_name] = Path(data_root) / polar_path_str
                else:
                    polar_paths[angle_name] = Path(polar_path_str)
        if len(polar_paths) == 4 and all(p.exists() for p in polar_paths.values()):
            return polar_paths

    # 2) JSON 中给出 polar_paths 但未包含 polar/ 前缀时，使用 polar_folder/data_root 兜底
    if polar_paths_json:
        polar_paths = {}
        for angle_name in ["I_0", "I_45", "I_90", "I_135"]:
            polar_path_str = polar_paths_json.get(angle_name)
            if not polar_path_str:
                break
            if os.path.isabs(polar_path_str):
                polar_paths[angle_name] = Path(polar_path_str)
            else:
                if polar_folder:
                    polar_paths[angle_name] = Path(polar_folder) / polar_path_str
                elif data_root:
                    polar_paths[angle_name] = Path(data_root) / polar_path_str
                else:
                    polar_paths[angle_name] = Path(polar_path_str)
        if len(polar_paths) == 4 and all(p.exists() for p in polar_paths.values()):
            return polar_paths

    # 3) 从 image/input_path 推断（只支持 rgb/）
    image_file = sample.get("input_path") or sample.get("image", "")
    if not image_file:
        return None

    if not data_root and image_folder:
        data_root = str(Path(image_folder).parent)

    if image_file.startswith("rgb/"):
        data_root_path = Path(data_root) if data_root else Path(image_folder).parent
        image_path = data_root_path / image_file
        base_name = image_path.stem.replace("_rgb", "")
        scene_id = image_path.parent.name
        polar_base = Path(polar_folder) if polar_folder else data_root_path / "polar"
        return {
            "I_0": polar_base / scene_id / f"{base_name}_000.png",
            "I_45": polar_base / scene_id / f"{base_name}_045.png",
            "I_90": polar_base / scene_id / f"{base_name}_090.png",
            "I_135": polar_base / scene_id / f"{base_name}_135.png",
        }

    return None


def compute_polar_tensor(polar_paths):
    physics_img = process_polar_images(polar_paths=polar_paths)  # (H, W, 4)
    polar_3ch = physics_img[:, :, 1:4]  # (H, W, 3)
    polar_3ch_uint8 = (polar_3ch * 255).astype(np.uint8)
    polar_pil = Image.fromarray(polar_3ch_uint8, mode="RGB")
    polar_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),  # [0, 1]
    ])
    polar_tensor = polar_transform(polar_pil)  # (3, 512, 512)
    return polar_tensor


def _build_cache_name(sample, idx):
    sample_id = sample.get("id")
    if not sample_id:
        image_file = sample.get("image") or sample.get("input_path") or f"idx_{idx}"
        sample_id = str(image_file).replace("/", "_")
    return f"{Path(sample_id).stem}.pt"


def _get_scene_id(sample, image_file):
    scene_id = sample.get("scene_id")
    if scene_id:
        return str(scene_id)
    if isinstance(image_file, str) and image_file:
        # rgb/05/0001_rgb.png -> 05
        return Path(image_file).parent.name
    return "unknown"


def _process_one(args_tuple):
    idx, sample, data_root, polar_folder, image_folder, output_dir, overwrite = args_tuple
    polar_paths = resolve_polar_paths(sample, data_root, polar_folder, image_folder)
    if not polar_paths or not all(Path(p).exists() for p in polar_paths.values()):
        return idx, None, f"第 {idx} 条样本缺少偏振图像: {polar_paths}"

    image_file = sample.get("image") or sample.get("input_path") or ""
    scene_id = _get_scene_id(sample, image_file)
    cache_name = _build_cache_name(sample, idx)
    cache_dir = Path(output_dir) / scene_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_name

    if cache_path.exists() and not overwrite:
        return idx, cache_path, None

    polar_tensor = compute_polar_tensor(polar_paths)
    torch.save(polar_tensor, cache_path)
    return idx, cache_path, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="输入 JSON 路径")
    parser.add_argument("--output_dir", required=True, help="缓存输出目录")
    parser.add_argument("--output_json", required=True, help="输出 JSON（包含 polar_tensor_path）")
    parser.add_argument("--data_root", default=None, help="数据根目录（用于拼接 rgb/ 与 polar/）")
    parser.add_argument("--polar_folder", default=None, help="偏振图像目录（可选）")
    parser.add_argument("--image_folder", default=None, help="RGB 图像目录（可选）")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有缓存文件")
    parser.add_argument("--limit", type=int, default=0, help="限制处理条数（0 表示全部）")
    parser.add_argument("--num_workers", type=int, default=4, help="并行进程数（建议 4~16）")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("data_path 必须是 JSON 数组（list of samples）。")

    total = len(data) if args.limit <= 0 else min(len(data), args.limit)

    indices = list(range(total))
    tasks = [
        (idx, data[idx], args.data_root, args.polar_folder, args.image_folder, str(output_dir), args.overwrite)
        for idx in indices
    ]

    if args.num_workers <= 1:
        for idx in tqdm(indices, desc="预计算 polar 缓存"):
            _, cache_path, err = _process_one(tasks[idx])
            if err:
                raise FileNotFoundError(err)
            if cache_path is None:
                raise FileNotFoundError(f"第 {idx} 条样本处理失败（无缓存路径）")
            if args.data_root and str(cache_path).startswith(str(Path(args.data_root))):
                rel_path = str(Path(cache_path).relative_to(Path(args.data_root)))
                data[idx]["polar_tensor_path"] = rel_path
            else:
                data[idx]["polar_tensor_path"] = str(cache_path)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.num_workers) as pool:
            for idx, cache_path, err in tqdm(
                pool.imap_unordered(_process_one, tasks),
                total=total,
                desc=f"预计算 polar 缓存 (workers={args.num_workers})",
            ):
                if err:
                    pool.terminate()
                    raise FileNotFoundError(err)
                if cache_path is None:
                    pool.terminate()
                    raise FileNotFoundError(f"第 {idx} 条样本处理失败（无缓存路径）")
                if args.data_root and str(cache_path).startswith(str(Path(args.data_root))):
                    rel_path = str(Path(cache_path).relative_to(Path(args.data_root)))
                    data[idx]["polar_tensor_path"] = rel_path
                else:
                    data[idx]["polar_tensor_path"] = str(cache_path)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"完成。缓存写入目录: {output_dir}")
    print(f"更新后的 JSON: {args.output_json}")


if __name__ == "__main__":
    main()

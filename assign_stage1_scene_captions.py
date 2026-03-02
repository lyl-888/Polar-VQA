import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resolve_path(path_str: str) -> Path:
    """
    解析输入路径：
    1) 优先直接按给定路径解析；
    2) 若不存在，则按 basename 在当前目录递归搜索（兼容 Windows 终端中文编码问题）。
    """
    p = Path(path_str)
    if p.exists():
        return p

    basename = p.name
    matches = list(Path(".").rglob(basename))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"路径不存在：{path_str}；同时发现多个同名文件，请显式指定路径："
            + ", ".join(str(x) for x in matches[:5])
        )
    raise FileNotFoundError(f"路径不存在，且未找到同名文件：{path_str}")


def build_scene_caption_map(scene_caption_data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    scene_to_caps: Dict[str, List[str]] = {}
    for item in scene_caption_data:
        scene_id = str(item.get("scene_id", "")).strip()
        caps = item.get("captions", [])
        if not scene_id:
            continue
        if not isinstance(caps, list):
            continue
        cleaned = [str(c).strip() for c in caps if str(c).strip()]
        if cleaned:
            scene_to_caps[scene_id] = cleaned
    return scene_to_caps


def assign_captions(
    image_samples: List[Dict[str, Any]],
    scene_to_caps: Dict[str, List[str]],
    seed: int,
) -> Dict[str, int]:
    """
    按 scene_id 为每个样本随机分配 caption。
    采用“每场景打乱后循环”策略：
    - 保持随机性
    - 避免某个 caption 过度重复
    """
    rng = random.Random(seed)

    scene_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, sample in enumerate(image_samples):
        scene_id = str(sample.get("scene_id", "")).strip()
        scene_to_indices[scene_id].append(idx)

    missing_scene_count = 0
    assigned_count = 0

    for scene_id, indices in scene_to_indices.items():
        caps = scene_to_caps.get(scene_id)
        if not caps:
            missing_scene_count += len(indices)
            continue

        cap_pool = caps[:]
        rng.shuffle(cap_pool)
        pointer = 0

        shuffled_indices = indices[:]
        rng.shuffle(shuffled_indices)

        for idx in shuffled_indices:
            image_samples[idx]["caption"] = cap_pool[pointer]
            assigned_count += 1
            pointer += 1
            if pointer >= len(cap_pool):
                # 用完一轮后再打乱，继续随机循环
                rng.shuffle(cap_pool)
                pointer = 0

    return {
        "assigned_count": assigned_count,
        "missing_scene_count": missing_scene_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "将 by_scene_simple.json 中每个 scene 的多种 caption 随机分配到 "
            "stage1_captions_qwen_cleaned.json 的每张图片样本上。"
        )
    )
    parser.add_argument(
        "--scene-caption-file",
        type=str,
        default="stage1_captions_by_scene_simple.json",
        help="按 scene 存放 captions 列表的 JSON 文件路径",
    )
    parser.add_argument(
        "--sample-file",
        type=str,
        default="stage1_captions_qwen_cleaned.json",
        help="逐图片样本 JSON 文件路径（含 scene_id 字段）",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="stage1_captions_qwen_cleaned_randomized.json",
        help="输出文件路径（默认不覆盖原文件）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（用于可复现）",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="直接覆盖 sample-file（谨慎使用）",
    )
    args = parser.parse_args()

    scene_caption_path = resolve_path(args.scene_caption_file)
    sample_path = resolve_path(args.sample_file)
    output_path = sample_path if args.inplace else sample_path.with_name(args.output_file)

    scene_caption_data = load_json(scene_caption_path)
    sample_data = load_json(sample_path)

    if not isinstance(scene_caption_data, list):
        raise ValueError("scene-caption-file 顶层必须是 list。")
    if not isinstance(sample_data, list):
        raise ValueError("sample-file 顶层必须是 list。")

    scene_to_caps = build_scene_caption_map(scene_caption_data)
    if not scene_to_caps:
        raise ValueError("scene-caption-file 中没有可用 captions。")

    stats = assign_captions(sample_data, scene_to_caps, seed=args.seed)
    dump_json(output_path, sample_data)

    # 打印一些统计信息，便于确认结果
    scene_counter = Counter(str(x.get("scene_id", "")).strip() for x in sample_data)
    print("Done.")
    print(f"Output: {output_path}")
    print(f"Total samples: {len(sample_data)}")
    print(f"Scenes in sample file: {len(scene_counter)}")
    print(f"Scenes with caption pool: {len(scene_to_caps)}")
    print(f"Assigned samples: {stats['assigned_count']}")
    print(f"Samples with missing scene captions: {stats['missing_scene_count']}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()

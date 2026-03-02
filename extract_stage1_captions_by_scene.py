import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def load_json(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of caption records.")
    return data


def group_captions_by_scene(data: List[Dict], mode: str = "unique") -> List[Dict]:
    """
    mode:
      - first: each scene outputs one caption (the first seen)
      - unique: each scene outputs deduplicated captions
    """
    scene_to_captions = defaultdict(list)
    for item in data:
        scene_id = str(item.get("scene_id", "unknown")).zfill(2)
        caption = str(item.get("caption", "")).strip()
        if caption:
            scene_to_captions[scene_id].append(caption)

    results: List[Dict] = []
    for scene_id in sorted(scene_to_captions.keys()):
        captions = scene_to_captions[scene_id]
        if mode == "first":
            results.append({
                "scene_id": scene_id,
                "caption": captions[0],
            })
        else:
            # unique mode: keep insertion order while deduplicating
            uniq = list(dict.fromkeys(captions))
            results.append({
                "scene_id": scene_id,
                "captions": uniq,
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract stage1 captions grouped by scene_id."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=r"C:\Users\LY\Desktop\train\没招\stage1_captions_qwen_cleaned.json",
        help="Input caption JSON path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"C:\Users\LY\Desktop\train\没招\stage1_captions_by_scene_simple.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="unique",
        choices=["first", "unique"],
        help="Output one caption per scene (first) or all unique captions per scene (unique).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_json(input_path)
    grouped = group_captions_by_scene(data, mode=args.mode)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)

    print(f"✓ Done. Scenes: {len(grouped)}")
    print(f"✓ Saved to: {output_path}")


if __name__ == "__main__":
    main()


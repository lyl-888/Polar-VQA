#!/usr/bin/env python3
"""
Stage 1 Caption 生成脚本（Polar 语义对齐）
使用 Qwen2-VL-7B-Instruct 基于 GT 图像生成“仅描述物体”的简短字幕。

设计目标：
- 避免颜色/材质/光照/纹理等描述（Polar 看不到颜色）
- 只描述图像中可见的主要物体（简洁、客观）
- 每张 RGB 图像对应一条 GT Caption（不需要 bbox）

依赖：
- transformers >= 4.40（支持 Qwen2-VL）
- qwen-vl-utils（可选）
- pillow, tqdm
"""

import argparse
import json
import os
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

# 过滤警告
warnings.filterwarnings("ignore")

from PIL import Image
from tqdm import tqdm

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


# ==================== 配置参数 ====================

# Qwen2-VL-7B-Instruct (local)
QWEN2VL_MODEL_NAME = "/openbayes/input/input0/models/Qwen/Qwen2-VL-7B-Instruct"

# 数据根目录
DATA_ROOT = Path("/openbayes/input/input0")
RGB_ROOT = DATA_ROOT / "rgb"  # rgb/{scene_id}/{filename}_rgb.png
GT_ROOT = DATA_ROOT / "GT"    # GT/{scene_id}/{filename}_rgb.png
POLAR_ROOT = DATA_ROOT / "polar"  # polar/{scene_id}/{filename}_{angle}.png

# 输出 JSON 文件名
OUTPUT_JSON_PREFIX = "stage1_captions_qwen"

# 生成参数
MAX_NEW_TOKENS = 48
TEMPERATURE = 0.2
TOP_P = 0.01

# 显存/性能限制
BATCH_SIZE = 1
MAX_IMAGE_SIZE = 1024

# 生成后清洗：禁止出现颜色/材质/光照/反射等词
FORBIDDEN_TOKEN_RE = re.compile(
    r"\b(?:"
    r"red|blue|green|yellow|black|white|brown|gray|grey|orange|purple|pink|gold|silver|"
    r"wood(?:en)?|metal(?:lic)?|plastic|glass|concrete|brick|paper|silk|"
    r"material|texture(?:d)?|lighting|light|shadow(?:s)?|style|"
    r"reflect(?:ion|ive|ed|ing|s)?"
    r")\b",
    flags=re.IGNORECASE,
)

SAFE_FALLBACK_CAPTION = "Several objects are arranged in the scene."


def build_caption_prompt(strict: bool = False) -> str:
    """
    构建用于“仅描述物体”的提示词，避免颜色/材质/光照/纹理等细节。
    """
    base = (
        "You are given a single image. "
        "Write a short caption that focuses on object identity and coarse layout. "
        "Emphasize shape, outline, relative position, and grouping. "
        "Do NOT mention color, material, texture, lighting, shadow, reflection, or style. "
        "Avoid fine visual details; keep it generic and concise. "
        "Use simple English. "
        "Use one plain sentence. Output ONLY the sentence with no extra text."
    )
    if strict:
        base += (
            " Forbidden vocabulary includes common color/material/lighting words. "
            "If such words would appear, rewrite the sentence with neutral wording."
        )
    return base


def contains_forbidden_tokens(text: str) -> bool:
    if not text:
        return False
    return FORBIDDEN_TOKEN_RE.search(text) is not None


def sanitize_caption(text: str) -> str:
    """
    生成后清洗：剔除禁用词，并尽量保持句子可读。
    """
    if not text:
        return SAFE_FALLBACK_CAPTION

    cleaned = text.strip()
    cleaned = FORBIDDEN_TOKEN_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\b(and|or|with|of|in|on|at|to)\s*([,.;:!?])", r"\2", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" ,;:.-")

    if not cleaned:
        return SAFE_FALLBACK_CAPTION

    # 保持单句
    cleaned = cleaned.split("\n")[0].strip()
    if len(cleaned.split()) < 3:
        return SAFE_FALLBACK_CAPTION

    # 统一句号
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def resize_keep_ratio(img: Image.Image, max_s: int) -> Image.Image:
    """保持宽高比，限制最大边长"""
    w, h = img.size
    if w > max_s or h > max_s:
        ratio = max_s / max(w, h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img


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
            "0000_rgb.png",
        ]
        for gt_name in possible_gt_names:
            candidate_path = gt_scene_dir / gt_name
            if candidate_path.exists():
                return candidate_path

    return None


def get_polar_paths(polar_root: Path, scene_id: str, base_name: str) -> Dict[str, Path]:
    """
    根据 scene_id 和 base_name 推导偏振图像的四个通道路径（非 crop）
    """
    polar_scene_dir = polar_root / scene_id

    # 优先三位数角度格式
    polar_paths = {
        "I_0": polar_scene_dir / f"{base_name}_000.png",
        "I_45": polar_scene_dir / f"{base_name}_045.png",
        "I_90": polar_scene_dir / f"{base_name}_090.png",
        "I_135": polar_scene_dir / f"{base_name}_135.png",
    }
    if all(p.exists() for p in polar_paths.values()):
        return polar_paths

    # 退回两位数角度格式
    polar_paths = {
        "I_0": polar_scene_dir / f"{base_name}_0.png",
        "I_45": polar_scene_dir / f"{base_name}_45.png",
        "I_90": polar_scene_dir / f"{base_name}_90.png",
        "I_135": polar_scene_dir / f"{base_name}_135.png",
    }
    return polar_paths


def load_qwen2vl_model(model_name: str = QWEN2VL_MODEL_NAME):
    """加载本地 Qwen2-VL-7B-Instruct"""
    import torch

    resolved_model_path = os.path.abspath(os.path.expanduser(model_name))
    print(f"Loading Qwen2-VL model: {resolved_model_path}...")
    if not os.path.exists(resolved_model_path):
        raise FileNotFoundError(
            f"Local model path not found: {resolved_model_path}. "
            "Please set --model_name to the local directory that contains the model files."
        )

    processor = AutoProcessor.from_pretrained(
        resolved_model_path, trust_remote_code=True
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        resolved_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    device = list(model.parameters())[0].device
    print(f"✓ Model loaded. Primary device: {device}")
    return model, processor, device


def _build_qwen2vl_input(processor, image: Image.Image, user_prompt: str, device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
                images=[image],
            return_tensors="pt",
    )
    return {k: v.to(device) for k, v in inputs.items()}


def generate_caption(
    model,
    processor,
    device,
    gt_image: Image.Image,
    strict_retries: int = 2,
) -> str:
    """使用 Qwen2-VL 为单张 GT 图像生成简短对象描述"""
    import torch

    max_attempts = max(1, strict_retries + 1)
    for attempt in range(max_attempts):
        user_prompt = build_caption_prompt(strict=(attempt > 0))
        inputs = _build_qwen2vl_input(processor, gt_image, user_prompt, device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.2,
                temperature=None,
                top_p=None,
            )

        generated_ids = generated_ids[:, inputs["input_ids"].shape[-1]:]
        response_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        cleaned = sanitize_caption(response_text)
        if not contains_forbidden_tokens(cleaned):
            return cleaned

    # 最后兜底
    return SAFE_FALLBACK_CAPTION


def clean_existing_caption_json(input_json: str, output_json: Optional[str] = None) -> None:
    """
    对已有 caption JSON 做离线清洗，无需重跑模型。
    """
    in_path = Path(input_json)
    out_path = Path(output_json) if output_json else in_path.with_name(in_path.stem + "_cleaned.json")
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list.")

    changed = 0
    had_forbidden = 0
    for item in data:
        caption = str(item.get("caption", "")).strip()
        if contains_forbidden_tokens(caption):
            had_forbidden += 1
        new_caption = sanitize_caption(caption)
        if new_caption != caption:
            item["caption"] = new_caption
            changed += 1

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✓ 清洗完成: {in_path}")
    print(f"  - 原始含禁用词条目: {had_forbidden}")
    print(f"  - 被修改条目: {changed}")
    print(f"  - 输出文件: {out_path}")


def iter_rgb_images(
    rgb_root: Optional[Path] = None,
    scene_id_min: Optional[int] = None,
    scene_id_max: Optional[int] = None,
    max_images_per_scene: Optional[int] = None,
    max_total_images: Optional[int] = None,
):
    """遍历 RGB 目录下的所有 RGB 图像"""
    rgb_dir = rgb_root or RGB_ROOT
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB root directory does not exist: {rgb_dir}")

    scene_dirs = sorted([p for p in rgb_dir.iterdir() if p.is_dir()])
    total_count = 0

    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        try:
            scene_id_int = int(scene_id)
            if scene_id_min is not None and scene_id_int < scene_id_min:
                continue
            if scene_id_max is not None and scene_id_int > scene_id_max:
                continue
        except ValueError:
            pass

        img_paths = sorted(scene_dir.glob("*_rgb.png"))
        if max_images_per_scene is not None:
            img_paths = img_paths[:max_images_per_scene]

        for img_path in img_paths:
            if max_total_images is not None and total_count >= max_total_images:
                return
            yield scene_id, img_path, img_path.name
            total_count += 1


def generate_stage1_captions(
    rgb_root: Optional[str] = None,
    gt_root: Optional[str] = None,
    polar_root: Optional[str] = None,
    scene_id_min: Optional[int] = None,
    scene_id_max: Optional[int] = None,
    max_images_per_scene: Optional[int] = None,
    max_total_images: Optional[int] = None,
    model_name: Optional[str] = None,
    output_json: Optional[str] = None,
    strict_retries: int = 2,
):
    model_path = model_name or QWEN2VL_MODEL_NAME
    rgb_dir = Path(rgb_root) if rgb_root else RGB_ROOT
    gt_dir = Path(gt_root) if gt_root else GT_ROOT
    polar_dir = Path(polar_root) if polar_root else POLAR_ROOT

    model, processor, device = load_qwen2vl_model(model_path)
    model.eval()

    if output_json is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{OUTPUT_JSON_PREFIX}_{timestamp}.json"
    else:
        output_path = output_json

    results: List[Dict] = []

    rgb_items = list(
        iter_rgb_images(
            rgb_root=rgb_dir,
            scene_id_min=scene_id_min,
            scene_id_max=scene_id_max,
            max_images_per_scene=max_images_per_scene,
            max_total_images=max_total_images,
        )
    )

    print(f"Found {len(rgb_items)} RGB images, generating captions...")
    for scene_id, rgb_path, rgb_filename in tqdm(rgb_items, desc="Generating Captions"):
        gt_image_path = find_gt_image_path(scene_id, rgb_filename, rgb_dir, gt_dir)
        if gt_image_path is None:
            continue

        try:
            gt_image = Image.open(gt_image_path).convert("RGB")
        except Exception:
            continue

        caption = generate_caption(
            model,
            processor,
            device,
            gt_image,
            strict_retries=strict_retries,
        )
        print(f"[{rgb_path}] {caption}")
        base_name = rgb_filename.replace("_rgb.png", "")
        polar_paths = get_polar_paths(polar_dir, scene_id, base_name)
        results.append(
            {
                "id": f"{scene_id}_{rgb_filename}",
                "image": str(rgb_path.relative_to(DATA_ROOT)),
                "gt_image": str(gt_image_path.relative_to(DATA_ROOT)),
                "scene_id": scene_id,
                "polar_paths": {k: str(v.relative_to(DATA_ROOT)) for k, v in polar_paths.items()},
                "caption": caption,
            }
        )

    output_file = Path(output_path)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n================ Generation Complete ================")
    print(f"Successfully generated {len(results)} captions")
    print(f"Output file: {output_file.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Stage 1 captions using Qwen2-VL (GT images only)"
    )
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--rgb_root", type=str, default=None)
    parser.add_argument("--gt_root", type=str, default=None)
    parser.add_argument("--polar_root", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--scene_id_min", type=int, default=None)
    parser.add_argument("--scene_id_max", type=int, default=None)
    parser.add_argument("--max_images_per_scene", type=int, default=None)
    parser.add_argument("--max_total_images", type=int, default=None)
    parser.add_argument("--strict_retries", type=int, default=2)
    parser.add_argument(
        "--clean_existing_json",
        type=str,
        default=None,
        help="仅清洗已有 caption JSON（不跑模型）",
    )
    parser.add_argument(
        "--clean_output_json",
        type=str,
        default=None,
        help="clean_existing_json 的输出路径（默认自动加 _cleaned）",
    )

    args = parser.parse_args()

    if args.clean_existing_json:
        clean_existing_caption_json(args.clean_existing_json, args.clean_output_json)
        raise SystemExit(0)

    generate_stage1_captions(
        rgb_root=args.rgb_root,
        gt_root=args.gt_root,
        polar_root=args.polar_root,
        scene_id_min=args.scene_id_min,
        scene_id_max=args.scene_id_max,
        max_images_per_scene=args.max_images_per_scene,
        max_total_images=args.max_total_images,
        model_name=args.model_name,
        output_json=args.output_json,
        strict_retries=args.strict_retries,
    )

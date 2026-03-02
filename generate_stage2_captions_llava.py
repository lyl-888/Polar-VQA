"""
Stage 2 语义对齐描述生成脚本
使用本地 LLaVA v1.6 Vicuna-7B HF 模型，对 RGB + GT 数据集生成用于 PolarVLM 训练的阶段二描述。

核心思路（差分检测 + GT 图直接描述）：
- **检测方法**：使用差分方法（GT vs RGB）检测眩光区域，这是反光位置的 ground truth
- **描述生成策略**：
  - **有反光时**：直接看 GT 图像（干净），描述 bbox 区域的真实物体内容（极简短，不超过30字）
  - **无反光时**：看 RGB 图像，描述普通场景
- **输出格式**：兼容 LLaVA 训练格式，每条数据包含 image 相对路径 + conversations（human / gpt）

工作流程：
1. 加载 RGB 和 GT 图像 → 2. 使用差分方法检测眩光 bbox → 3. 直接看 GT 图生成描述（有反光时）
   → 4. 保存为训练数据

训练时的语义对齐：
- 输入：RGB（带反光） + 偏振特征（检测到反光区域）
- 目标：GT 图中的真实物体描述（无反光）
- 模型学习：偏振特征（反光区域）→ 背后真实物体的语义

依赖：
- transformers >= 4.37（支持 LlavaNextForConditionalGeneration / LlavaNextProcessor）
- bitsandbytes（4bit 量化）
- accelerate, safetensors
- pillow, numpy, opencv-python, tqdm

作者：AI Assistant
日期：2025
"""

import argparse
import json
import math
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    BitsAndBytesConfig,
)

# 这里假设你已经有一个公共工具文件 dataset_common.py，
# 内部包含 process_polar_images 函数，用于读取 4 张偏振图并计算 DoLP 等。
# 你可以按照自己项目中的实际实现进行替换。
#
# 示例（仅说明接口，不在本脚本中实现）：
#   polar_info = process_polar_images(polar_paths)
#   dolp = polar_info["dolp"]  # H x W 的 DoLP 数组，范围 [0,1]
from dataset_common import process_polar_images  # noqa: F401  # type: ignore


# ==================== 配置参数 ====================

# 模型名称（Hugging Face Hub 路径）
LLAVA_MODEL_NAME = "llava-hf/llava-v1.6-vicuna-7b-hf"

# 数据根目录（已通过前面脚本整理好的结构）
# 默认路径：OpenBayes 服务器路径
DATA_ROOT = Path("/openbayes/input/input0")
RGB_ROOT = DATA_ROOT / "rgb"      # 结构：rgb/{scene_id}/{filename}_rgb.png
POLAR_ROOT = DATA_ROOT / "polar"  # 结构：polar/{scene_id}/{filename}_{angle}.png

# 输出 JSON 文件（默认会加上时间戳）
OUTPUT_JSON_PREFIX = "stage2_gt_captions"

# 裁剪图像输出目录（相对于数据根目录）
RGB_CROP_DIR = "rgb_crop"      # 将保存为 {DATA_ROOT}/rgb_crop/{scene_id}/
GT_CROP_DIR = "GT_crop"         # 将保存为 {DATA_ROOT}/GT_crop/{scene_id}/
POLAR_CROP_DIR = "polar_crop"   # 将保存为 {DATA_ROOT}/polar_crop/{scene_id}/

# 裁剪图像最小尺寸（像素），小于此尺寸的裁剪将被跳过
MIN_CROP_SIZE = 64  # 最小 64x64 像素

# ⚠️ 关键修改：统一使用 512x512 保存所有 crop 图像（RGB 和 Polar）
# 原因：避免视野错位问题
# - 如果 RGB 裁剪 224x224，Polar 裁剪 512x512，会导致视野不匹配
# - 解决方案：都保存为 512x512，训练时 CLIP 再从 512 resize 到 224
CROP_IMAGE_SIZE = 512  # 512x512 像素（用于所有 crop 图像：RGB、GT、Polar）

# LLM 视觉编码器支持的图像尺寸（训练时使用）
LLM_IMAGE_SIZE = 224  # 224x224 像素（训练时 CLIP 会将 512x512 的 RGB resize 到 224x224）

# 反光检测参数（DoLP方法）
DOLP_THRESHOLD = 0.15          # DoLP 阈值，大于该值视为高偏振/可能反光区域
MIN_REFLECTION_AREA_RATIO = 0.01   # 反光区域最小占比（相对于整图），过小则视为无效
MAX_REFLECTION_AREA_RATIO = 0.6    # 反光区域最大占比，过大可能是噪声或整图

# 反光检测参数（差分方法 - 备用）
GLARE_THRESHOLD = 100          # 眩光检测阈值（提高阈值，只抓最亮的核心反光区域）
MIN_GLARE_AREA = 500           # 最小眩光区域面积（像素），小于此值则跳过
MAX_BBOX_RATIO = 0.4           # 最大边界框面积比例（相对于整张图像），超过此比例则跳过

# 生成参数
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
TOP_P = 0.9

# 批处理参数（用于加速处理）
BATCH_SIZE = 4  # 批处理大小，根据GPU显存调整（4bit量化模型可以设置更大，如8-16）

# 输出清理：需要移除的冗余短语（正则表达式模式）
CLEANUP_PATTERNS = [
    r"^(The image depicts|In this picture|In the image|This image shows|The picture shows|This picture shows|The image shows)",
    r"^(The image|This image|The picture|This picture)",
    r"\.$",  # 移除末尾的句号（如果需要更简洁）
    # 移除"无其他物体/颜色/材质/形状"等冗余表述
    r"\s*[Tt]here are no other objects?[^.]*\.",  # "There are no other objects..."
    r"\s*[Tt]here are no other colors?[^.]*\.",  # "There are no other colors..."
    r"\s*[Tt]here are no other materials?[^.]*\.",  # "There are no other materials..."
    r"\s*[Tt]here are no other shapes?[^.]*\.",  # "There are no other shapes..."
    r"\s*[Nn]o other objects?[^.]*\.",  # "No other objects..."
    r"\s*[Nn]o other colors?[^.]*\.",  # "No other colors..."
    r"\s*[Nn]o other materials?[^.]*\.",  # "No other materials..."
    r"\s*[Nn]o other shapes?[^.]*\.",  # "No other shapes..."
    r"\s*[Nn]o visible text[^.]*\.",  # "No visible text..."
    r"\s*[Tt]here is no visible text[^.]*\.",  # "There is no visible text..."
    r"\s*[Nn]o texts? or patterns?[^.]*\.",  # "No text or patterns..."
    r"\s*[Tt]here are no texts? or patterns?[^.]*\.",  # "There are no texts or patterns..."
    r"\s*[Nn]o additional[^.]*\.",  # "No additional..."
    r"\s*[Tt]here are no additional[^.]*\.",  # "There are no additional..."
    # 移除"这个区域"、"该区域"、"这个部分"等提及区域的词（重要：避免答案提到区域）
    r"\s*[Ii]n this (region|area|section|part|portion)[^.]*\.",  # "In this region/area/section/part..."
    r"\s*[Tt]his (region|area|section|part|portion)[^.]*\.",  # "This region/area/section/part..."
    r"\s*[Tt]he (region|area|section|part|portion)[^.]*\.",  # "The region/area/section/part..."
    r"\s*[Ww]ithin this (region|area|section|part|portion)[^.]*\.",  # "Within this region/area..."
    r"\s*[Ii]n the (region|area|section|part|portion)[^.]*\.",  # "In the region/area..."
    r"\s*[Oo]f this (region|area|section|part|portion)[^.]*\.",  # "Of this region/area..."
    r"\s*[Aa]t this (region|area|section|part|portion)[^.]*\.",  # "At this region/area..."
    r"\s*[Ii]nside this (region|area|section|part|portion)[^.]*\.",  # "Inside this region/area..."
]


# ==================== 偏振 & 反光检测相关函数 ====================

def get_reflection_bbox_from_dolp(
    dolp: np.ndarray,
    threshold: float = DOLP_THRESHOLD,
    min_area_ratio: float = MIN_REFLECTION_AREA_RATIO,
    max_area_ratio: float = MAX_REFLECTION_AREA_RATIO,
):
    """
    根据 DoLP 图像检测主要高偏振区域，并返回归一化的 bbox（[ymin, xmin, ymax, xmax], 0~1）。

    Args:
        dolp: DoLP 数组，形状 (H, W)，值域 [0,1]
        threshold: 像素级阈值，大于该值视为候选高偏振区域
        min_area_ratio: 高偏振区域在整图中的最小占比（剔除小噪声）
        max_area_ratio: 高偏振区域在整图中的最大占比（防止整图被选中）

    Returns:
        bbox_norm: [ymin, xmin, ymax, xmax]，归一化坐标；如果无有效区域则返回 None
    """
    if dolp is None:
        return None

    h, w = dolp.shape[:2]
    if h == 0 or w == 0:
        return None

    # 阈值分割：高 DoLP 区域
    mask = (dolp >= threshold).astype(np.uint8) * 255

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 找面积最大的轮廓
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    img_area = float(h * w)
    area_ratio = area / img_area

    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        # 面积太小或太大都不认为是有效反光区域
        return None

    x, y, bw, bh = cv2.boundingRect(largest)
    xmin, ymin, xmax, ymax = x, y, x + bw, y + bh

    # 归一化到 [0,1]
    xmin_n = xmin / w
    xmax_n = xmax / w
    ymin_n = ymin / h
    ymax_n = ymax / h

    # 限制在 [0,1] 范围
    xmin_n = float(np.clip(xmin_n, 0.0, 1.0))
    xmax_n = float(np.clip(xmax_n, 0.0, 1.0))
    ymin_n = float(np.clip(ymin_n, 0.0, 1.0))
    ymax_n = float(np.clip(ymax_n, 0.0, 1.0))

    return [ymin_n, xmin_n, ymax_n, xmax_n]


def find_gt_image_path(
    scene_id: str,
    rgb_filename: str,
    rgb_root: Path,
    gt_root: Optional[Path] = None,
) -> Optional[Path]:
    """
    根据场景 ID 和 RGB 文件名，查找对应的 GT 图像路径。
    
    Args:
        scene_id: 场景 ID（字符串，例如 "04", "59"）
        rgb_filename: RGB 文件名，例如 "0000_rgb.png"
        rgb_root: RGB 图像根目录
        gt_root: GT 图像根目录（None 表示自动查找）
        
    Returns:
        GT 图像路径；如果找不到，返回 None
    """
    prefix = rgb_filename.replace("_rgb.png", "")
    
    # 尝试找到GT图像路径
    if gt_root is None:
        # 尝试多个可能的GT路径
        possible_gt_roots = [
            rgb_root.parent / "GT",  # data/GT
            rgb_root / ".." / "GT",  # 相对路径
            Path("/openbayes/input/input0/GT"),  # OpenBayes默认路径
        ]
    else:
        possible_gt_roots = [Path(gt_root)]
    
    for gt_dir in possible_gt_roots:
        gt_scene_dir = gt_dir / scene_id
        # GT图像文件名可能与RGB相同，也可能不同（根据数据集结构）
        possible_gt_names = [
            rgb_filename,  # 相同文件名
            f"{prefix}_rgb.png",  # 去掉_rgb后缀再加回来（如果prefix不同）
            f"0000_rgb.png",  # 默认GT文件名（根据README中的示例）
        ]
        for gt_name in possible_gt_names:
            candidate_path = gt_scene_dir / gt_name
            if candidate_path.exists():
                return candidate_path
    
    return None


def detect_glare_bbox_diff(
    input_image_path: Path,
    gt_image_path: Path,
    threshold: int = GLARE_THRESHOLD,
    min_area: int = MIN_GLARE_AREA,
    max_bbox_ratio: float = MAX_BBOX_RATIO,
):
    """
    使用计算机视觉方法检测眩光区域并返回边界框（差分方法）。
    
    通过对比输入RGB图像（带眩光）和GT干净图像的差异来检测眩光区域。
    这是反光位置的 ground truth。
    
    Args:
        input_image_path: 输入眩光图像路径（RGB，带眩光）
        gt_image_path: GT干净图像路径
        threshold: 差分阈值（提高阈值，只抓最亮的核心反光区域）
        min_area: 最小眩光区域面积（像素），小于此值则跳过
        max_bbox_ratio: 最大边界框面积比例（相对于整张图像），超过此比例则跳过
        
    Returns:
        bbox_norm: [ymin, xmin, ymax, xmax]，归一化坐标；如果检测失败，返回 None
    """
    try:
        # 读取输入图像和GT图像
        input_img = cv2.imread(str(input_image_path))
        gt_img = cv2.imread(str(gt_image_path))
        
        if input_img is None or gt_img is None:
            return None
        
        # 转换为灰度图
        input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
        
        # 确保两张图像尺寸相同（如果不一致，需要调整）
        if input_gray.shape != gt_gray.shape:
            # 将GT图像调整为输入图像尺寸
            gt_gray = cv2.resize(gt_gray, (input_gray.shape[1], input_gray.shape[0]))
        
        # 计算绝对差分
        diff = cv2.absdiff(input_gray, gt_gray)
        
        # 应用阈值创建二值掩码（提高阈值，只抓最亮的核心反光区域）
        _, binary_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 找到面积最大的轮廓（假设这是眩光区域）
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        
        # 如果轮廓面积太小（可能是噪音），跳过
        if contour_area < min_area:
            return None
        
        # 计算边界框 [x, y, w, h]
        x, y, w, h = cv2.boundingRect(largest_contour)
        rect_area = w * h  # 边界框的面积
        
        # 计算整张图像的面积
        h_img, w_img = input_gray.shape[:2]
        img_area = h_img * w_img
        
        # 过滤逻辑：边界框面积太大（超过图像面积的一定比例），跳过
        if rect_area < min_area:
            return None
        
        if rect_area > (img_area * max_bbox_ratio):
            # 边界框太大，可能框住了分散的反光点或太多背景，跳过
            return None
        
        # 转换为归一化坐标 [ymin, xmin, ymax, xmax]
        xmin_n = float(x / w_img)
        xmax_n = float((x + w) / w_img)
        ymin_n = float(y / h_img)
        ymax_n = float((y + h) / h_img)
        
        # 限制在 [0,1] 范围
        xmin_n = float(np.clip(xmin_n, 0.0, 1.0))
        xmax_n = float(np.clip(xmax_n, 0.0, 1.0))
        ymin_n = float(np.clip(ymin_n, 0.0, 1.0))
        ymax_n = float(np.clip(ymax_n, 0.0, 1.0))
        
        return [ymin_n, xmin_n, ymax_n, xmax_n]
        
    except Exception as e:
        print(f"眩光检测出错（差分方法）: {e}")
        traceback.print_exc()
        return None


def get_reflection_bbox(
    scene_id: str,
    rgb_filename: str,
    polar_root: Optional[Path] = None,
    rgb_root: Optional[Path] = None,
    gt_root: Optional[Path] = None,
    method: str = "dolp",
):
    """
    给定场景 ID 和 RGB 文件名，检测反光区域并返回 bbox。
    
    检测策略（强制使用 DoLP 作为主要方法）：
    - 优先使用 DoLP 方法（偏振图像的物理真值）
    - 仅在 DoLP 完全失败（偏振图像不存在或处理异常）时，才尝试差分方法
    - 如果都失败，返回 None

    Args:
        scene_id: 场景 ID（字符串，例如 "04", "59"）
        rgb_filename: RGB 文件名，例如 "0000_rgb.png"
        polar_root: 偏振图像根目录（None 表示使用默认 POLAR_ROOT）
        rgb_root: RGB 图像根目录（用于差分方法，None 表示使用默认 RGB_ROOT）
        gt_root: GT 图像根目录（用于差分方法，None 表示尝试 data/GT 或 rgb_root/../GT）
        method: 检测方法，"dolp"（仅DoLP，默认）、"diff"（仅差分）、"auto"（DoLP失败则用差分）

    Returns:
        bbox_norm: [ymin, xmin, ymax, xmax]，归一化坐标；若失败则返回 None
    """
    prefix = rgb_filename.replace("_rgb.png", "")
    
    # ========== 方法1：DoLP检测（主要方法，物理真值） ==========
    if method in ("auto", "dolp"):
        polar_dir = polar_root or POLAR_ROOT
        polar_scene_dir = polar_dir / scene_id
        
        polar_paths_dict = {
            "I_0": polar_scene_dir / f"{prefix}_000.png",
            "I_45": polar_scene_dir / f"{prefix}_045.png",
            "I_90": polar_scene_dir / f"{prefix}_090.png",
            "I_135": polar_scene_dir / f"{prefix}_135.png",
        }
        
        # 检查偏振图像是否存在
        all_polar_exist = all(p.exists() for p in polar_paths_dict.values())
        
        if all_polar_exist:
            try:
                # 复用当前项目中的偏振处理函数
                physics_img = process_polar_images(polar_paths=polar_paths_dict)
                if physics_img is not None:
                    # 提取 DoLP（第 1 通道），形状 (H, W)，值域 [0,1]
                    dolp_np = np.asarray(physics_img[:, :, 1], dtype=np.float32)
                    bbox_dolp = get_reflection_bbox_from_dolp(dolp_np)
                    if bbox_dolp is not None:
                        return bbox_dolp  # DoLP方法成功，直接返回
            except Exception as e:
                # DoLP方法失败，记录错误但仅在 method="auto" 时尝试差分
                if method == "dolp":
                    # 如果强制使用 DoLP，直接返回 None
                    return None
                # method="auto" 时继续尝试差分方法
    
    # ========== 方法2：差分检测（仅在 DoLP 完全失败时使用） ==========
    if method in ("auto", "diff"):
        rgb_dir = rgb_root or RGB_ROOT
        rgb_scene_dir = rgb_dir / scene_id
        input_rgb_path = rgb_scene_dir / rgb_filename
        
        # 尝试找到GT图像路径
        if gt_root is None:
            # 尝试多个可能的GT路径
            possible_gt_roots = [
                rgb_dir.parent / "GT",  # data/GT
                rgb_dir / ".." / "GT",  # 相对路径
                Path("/openbayes/input/input0/GT"),  # OpenBayes默认路径
            ]
        else:
            possible_gt_roots = [Path(gt_root)]
        
        gt_image_path = None
        for gt_dir in possible_gt_roots:
            gt_scene_dir = gt_dir / scene_id
            # GT图像文件名可能与RGB相同，也可能不同（根据数据集结构）
            possible_gt_names = [
                rgb_filename,  # 相同文件名
                f"{prefix}_rgb.png",  # 去掉_rgb后缀再加回来（如果prefix不同）
                f"0000_rgb.png",  # 默认GT文件名（根据README中的示例）
            ]
            for gt_name in possible_gt_names:
                candidate_path = gt_scene_dir / gt_name
                if candidate_path.exists():
                    gt_image_path = candidate_path
                    break
            if gt_image_path is not None:
                break
        
        if gt_image_path is not None and input_rgb_path.exists():
            try:
                bbox_diff = detect_glare_bbox_diff(input_rgb_path, gt_image_path)
                if bbox_diff is not None:
                    return bbox_diff  # 差分方法成功
            except Exception:
                traceback.print_exc()
                return None
    
    # 所有方法都失败
    return None


# ==================== LLaVA 模型加载 ====================

def load_llava_model(model_name: str = LLAVA_MODEL_NAME):
    """
    加载 LLaVA-Next 模型（4bit 量化，自动映射到 GPU）。

    根据官方文档，有两种方式：
    1. 使用 BitsAndBytesConfig（更灵活，可配置 double quantization）
    2. 直接使用 load_in_4bit=True（更简单，transformers>=4.48）

    这里使用 BitsAndBytesConfig 方式，以支持更细粒度的量化配置。

    返回：
        model: LlavaNextForConditionalGeneration
        processor: LlavaNextProcessor
        device: torch.device 或 str
    """
    import torch
    
    # 4bit 量化配置（需要安装 bitsandbytes）
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # 使用 torch.bfloat16 而不是字符串
    )

    print(f"加载 LLaVA 模型：{model_name}（4bit 量化）...")
    processor = LlavaNextProcessor.from_pretrained(model_name)
    
    # 根据官方文档，也可以直接使用 load_in_4bit=True（更简单）：
    # model = LlavaNextForConditionalGeneration.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.float16,
    #     low_cpu_mem_usage=True,
    #     load_in_4bit=True,
    #     device_map="auto",
    # )
    # 但这里使用 BitsAndBytesConfig 以支持 double quantization
    
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
    )

    # 推断 device：取第一个参数所在设备
    device = list(model.parameters())[0].device
    print(f"模型已加载到设备：{device}")

    return model, processor, device


# ==================== Prompt 构造 ====================

def build_prompt_gt_content_cropped():
    """
    构造描述裁剪后 GT 图像的 Prompt（简洁但信息丰富，适合语义对齐）。
    
    注意：裁剪后的图像已经聚焦在物体上，不需要坐标信息。
    
    参考提示词思路：
    - 描述确信看到的视觉内容（物体、文字、形状、颜色、材质）
    - 实事求是，不要脑补
    - 简洁但信息丰富，包含足够的语义信息（40-50字）
    - 使用"或"而不是"及"，避免模型强制列出所有元素
    """
    prompt = (
        "Describe what you see in this image, such as objects, colors, materials, shapes, or any text. "
        "Be concise and factual. Aim for 40-50 words."
    )
    return prompt


def build_prompt_no_glare():
    """
    构造无反光时的描述 Prompt（与有反光时使用相同的 prompt）。
    使用"或"而不是"及"，避免模型强制列出所有元素。
    """
    # 使用与有反光时相同的 prompt
    return build_prompt_gt_content_cropped()


def crop_image_by_bbox(
    image: Image.Image,
    bbox_norm: list,
    min_size: int = MIN_CROP_SIZE,
    padding_ratio: float = 0.1,
    resize_to: Optional[int] = None,
) -> Optional[Image.Image]:
    """
    根据归一化 bbox 裁剪图像，并可选择性地调整大小。

    Args:
        image: PIL Image 对象
        bbox_norm: [ymin, xmin, ymax, xmax]（归一化坐标，0-1）
        min_size: 最小裁剪尺寸（像素），小于此尺寸返回 None
        padding_ratio: 裁剪时添加的边距比例（相对于 bbox 尺寸）
        resize_to: 如果提供，将裁剪后的图像调整到此尺寸（例如 512，用于统一所有 crop 图像尺寸）
    
    Returns:
        裁剪后的 PIL Image（如果指定了 resize_to，则已调整大小），如果尺寸太小则返回 None
    """
    if bbox_norm is None:
        return None
    
    ymin, xmin, ymax, xmax = bbox_norm
    img_width, img_height = image.size
    
    # 转换为像素坐标
    x1 = int(xmin * img_width)
    y1 = int(ymin * img_height)
    x2 = int(xmax * img_width)
    y2 = int(ymax * img_height)
    
    # 计算 bbox 尺寸
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    # 添加边距（padding）
    padding_x = int(bbox_width * padding_ratio)
    padding_y = int(bbox_height * padding_ratio)
    
    # 扩展裁剪区域（确保不超出图像边界）
    x1 = max(0, x1 - padding_x)
    y1 = max(0, y1 - padding_y)
    x2 = min(img_width, x2 + padding_x)
    y2 = min(img_height, y2 + padding_y)
    
    # 检查最小尺寸
    crop_width = x2 - x1
    crop_height = y2 - y1
    
    if crop_width < min_size or crop_height < min_size:
        return None
    
    # 裁剪图像
    cropped = image.crop((x1, y1, x2, y2))
    
    # 如果指定了 resize_to，调整大小
    if resize_to is not None:
        cropped = cropped.resize((resize_to, resize_to), Image.Resampling.LANCZOS)
    
    return cropped


def crop_image_centered_on_bbox(
    image: Image.Image,
    bbox_norm: list,
    target_size: int = CROP_IMAGE_SIZE,  # 默认使用 512x512，确保视野一致
) -> Optional[Image.Image]:
    """
    以 bbox 的中心点为圆心，向外扩展，切出一个标准的 target_size x target_size 区域。
    
    如果 bbox 太小导致无法直接裁剪，使用此函数以 bbox 中心为圆心切出固定大小的区域。
    
    Args:
        image: PIL Image 对象
        bbox_norm: [ymin, xmin, ymax, xmax]（归一化坐标，0-1）
        target_size: 目标裁剪尺寸（像素），默认 512x512（CROP_IMAGE_SIZE，确保与 RGB/Polar 视野一致）
    
    Returns:
        裁剪后的 PIL Image（target_size x target_size），如果失败则返回 None
    """
    if bbox_norm is None:
        return None
    
    ymin, xmin, ymax, xmax = bbox_norm
    img_width, img_height = image.size
    
    # 转换为像素坐标
    x1 = int(xmin * img_width)
    y1 = int(ymin * img_height)
    x2 = int(xmax * img_width)
    y2 = int(ymax * img_height)
    
    # 计算 bbox 的中心点
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    
    # 计算以中心点为圆心的 target_size x target_size 区域的边界
    half_size = target_size / 2.0
    crop_x1 = center_x - half_size
    crop_y1 = center_y - half_size
    crop_x2 = center_x + half_size
    crop_y2 = center_y + half_size
    
    # 处理边界情况：如果超出图像边界，需要调整位置
    # 优先保持中心点不变，如果无法保持，则尽量保持 target_size x target_size 的尺寸
    
    # 检查并调整 X 方向
    if crop_x1 < 0:
        # 左边界超出，向右移动
        offset = -crop_x1
        crop_x1 = 0
        crop_x2 = min(img_width, crop_x2 + offset)
    if crop_x2 > img_width:
        # 右边界超出，向左移动
        offset = crop_x2 - img_width
        crop_x2 = img_width
        crop_x1 = max(0, crop_x1 - offset)
    
    # 检查并调整 Y 方向
    if crop_y1 < 0:
        # 上边界超出，向下移动
        offset = -crop_y1
        crop_y1 = 0
        crop_y2 = min(img_height, crop_y2 + offset)
    if crop_y2 > img_height:
        # 下边界超出，向上移动
        offset = crop_y2 - img_height
        crop_y2 = img_height
        crop_y1 = max(0, crop_y1 - offset)
    
    # 转换为整数坐标
    crop_x1 = int(crop_x1)
    crop_y1 = int(crop_y1)
    crop_x2 = int(crop_x2)
    crop_y2 = int(crop_y2)
    
    # 确保裁剪区域至少是 target_size x target_size
    # 如果图像本身小于 target_size，则先裁剪再 resize
    actual_width = crop_x2 - crop_x1
    actual_height = crop_y2 - crop_y1
    
    if actual_width >= target_size and actual_height >= target_size:
        # 图像足够大，直接裁剪 target_size x target_size
        # 从中心点开始裁剪
        center_x_int = crop_x1 + actual_width // 2
        center_y_int = crop_y1 + actual_height // 2
        final_x1 = max(0, center_x_int - target_size // 2)
        final_y1 = max(0, center_y_int - target_size // 2)
        final_x2 = min(img_width, final_x1 + target_size)
        final_y2 = min(img_height, final_y1 + target_size)
        
        # 如果调整后尺寸不足，从另一边扩展
        if final_x2 - final_x1 < target_size:
            final_x1 = max(0, final_x2 - target_size)
        if final_y2 - final_y1 < target_size:
            final_y1 = max(0, final_y2 - target_size)
        
        cropped = image.crop((final_x1, final_y1, final_x1 + target_size, final_y1 + target_size))
    else:
        # 图像太小，先裁剪再 resize
        cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        cropped = cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    return cropped


def crop_polar_images_by_bbox(
    polar_paths: dict,
    bbox_norm: list,
    min_size: int = MIN_CROP_SIZE,
    padding_ratio: float = 0.1,
) -> Optional[dict]:
    """
    根据归一化 bbox 裁剪偏振图像（4个角度）。
    
    Args:
        polar_paths: 包含4个角度图像路径的字典 {"I_0": Path, ...}
        bbox_norm: [ymin, xmin, ymax, xmax]（归一化坐标，0-1）
        min_size: 最小裁剪尺寸（像素），小于此尺寸返回 None
        padding_ratio: 裁剪时添加的边距比例（相对于 bbox 尺寸）
    
    Returns:
        包含裁剪后图像路径的字典，如果失败则返回 None
    """
    if bbox_norm is None:
        return None
    
    # 加载第一张图像以获取尺寸
    try:
        first_img = Image.open(polar_paths["I_0"]).convert("RGB")
    except Exception:
        return None
    
    img_width, img_height = first_img.size
    
    # 转换为像素坐标
    ymin, xmin, ymax, xmax = bbox_norm
    x1 = int(xmin * img_width)
    y1 = int(ymin * img_height)
    x2 = int(xmax * img_width)
    y2 = int(ymax * img_height)
    
    # 计算 bbox 尺寸
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    # 添加边距
    padding_x = int(bbox_width * padding_ratio)
    padding_y = int(bbox_height * padding_ratio)
    
    # 扩展裁剪区域
    x1 = max(0, x1 - padding_x)
    y1 = max(0, y1 - padding_y)
    x2 = min(img_width, x2 + padding_x)
    y2 = min(img_height, y2 + padding_y)
    
    # 检查最小尺寸
    crop_width = x2 - x1
    crop_height = y2 - y1
    
    if crop_width < min_size or crop_height < min_size:
        return None
    
    # 裁剪所有4个角度的图像
    cropped_paths = {}
    for angle_name, path in polar_paths.items():
        try:
            img = Image.open(path).convert("RGB")
            cropped = img.crop((x1, y1, x2, y2))
            # 保存裁剪后的图像（临时，稍后会在主流程中统一保存）
            cropped_paths[angle_name] = cropped
        except Exception:
            return None
    
    return cropped_paths


def clean_caption(caption: str) -> str:
    """
    清理生成的描述，移除冗余短语，使其更简洁、密集。
    
    特别注意：
    - 移除所有提及"区域"、"部分"等词，避免答案提到"这个区域"
    - 移除坐标信息（虽然 prompt 已经避免，但以防万一）
    
    Args:
        caption: 原始生成的描述文本
        
    Returns:
        清理后的描述文本
    """
    if not caption:
        return caption
    
    # 移除常见的冗余开头短语
    for pattern in CLEANUP_PATTERNS:
        caption = re.sub(pattern, "", caption, flags=re.IGNORECASE)
    
    # 额外清理：移除包含坐标信息的句子（以防万一）
    # 匹配类似 "[x=0.123, y=0.456]" 或 "region [x=..." 的模式
    caption = re.sub(r'\[x\s*=\s*[\d.]+\s*,\s*y\s*=\s*[\d.]+\s*[^\]]*\]', '', caption, flags=re.IGNORECASE)
    caption = re.sub(r'region\s*\[x\s*=\s*[\d.]+\s*,\s*y\s*=\s*[\d.]+\s*[^\]]*\]', '', caption, flags=re.IGNORECASE)
    caption = re.sub(r'\[x\s*=\s*[\d.]+\s*,\s*y\s*=\s*[\d.]+\s*,\s*width\s*=\s*[\d.]+\s*,\s*height\s*=\s*[\d.]+\]', '', caption, flags=re.IGNORECASE)
    
    # 清理多余的空格和换行
    caption = re.sub(r"\s+", " ", caption)  # 多个空格合并为一个
    caption = caption.strip()
    
    # 确保首字母大写（如果还有内容）
    if caption:
        caption = caption[0].upper() + caption[1:] if len(caption) > 1 else caption.upper()

    return caption


def build_llava_input(processor, image: Image.Image, user_prompt: str, device):
    """
    构造符合 LLaVA-Next chat 模板的输入（单图像版本）。

    使用官方推荐的 chat 模板格式：
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "..."},
                    {"type": "image", "image": <PIL.Image>},
                ],
            }
        ]

    注意：根据 transformers 文档，可以直接在 conversation 中传入 PIL Image 对象，
    而不需要使用占位符 + images 参数的方式，这样可以避免参数冲突。

    返回：
        inputs: processor(...) 的结果（包含 input_ids, pixel_values 等），已移动到指定设备
    """
    # 方式1：直接在 conversation 中传入 PIL Image 对象（推荐，避免参数冲突）
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image", "image": image},  # 直接传入 PIL Image 对象
            ],
        }
    ]

    # 使用 apply_chat_template，不需要 images 参数（因为图像已经在 conversation 中）
    try:
        inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception as e:
        # 如果方式1失败，尝试方式2：使用占位符 + images 参数
        print(f"警告：方式1失败，尝试方式2：{e}")
        conversation_alt = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image"},  # 占位符
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            conversation_alt,
            images=[image],  # 通过 images 参数传入
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        return_tensors="pt",
    )
    
    # 将所有张量移动到指定设备
    inputs = {k: v.to(device) for k, v in inputs.items()}

    return inputs


def build_llava_input_batch(processor, images: list, user_prompt: str, device):
    """
    构造符合 LLaVA-Next chat 模板的批量输入（批处理版本，用于加速）。

    Args:
        processor: LLaVA processor
        images: PIL Image 对象列表
        user_prompt: 用户提示词（所有图像使用相同的提示词）
        device: 设备

    Returns:
        inputs: processor(...) 的结果（包含 input_ids, pixel_values 等），已移动到指定设备
    """
    # 为每个图像构造 conversation
    conversations = []
    for image in images:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image", "image": image},
                ],
            }
        ]
        conversations.append(conversation)
    
    # 批量处理
    try:
        inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        return_tensors="pt",
    )
    except Exception as e:
        # 如果批量处理失败，回退到逐个处理
        print(f"警告：批量处理失败，回退到逐个处理：{e}")
        # 这里可以回退到逐个处理，但为了简化，我们直接抛出异常
        raise e
    
    # 将所有张量移动到指定设备
    inputs = {k: v.to(device) for k, v in inputs.items()}

    return inputs


# ==================== 主处理流程 ====================

def iter_rgb_images(
    rgb_root: Optional[Path] = None,
    scene_id_min: Optional[int] = None,
    scene_id_max: Optional[int] = None,
    max_images_per_scene: Optional[int] = None,
    max_total_images: Optional[int] = None,
):
    """
    遍历 RGB 目录下的所有 RGB 图像，支持场景和数量过滤。

    结构假设：
        {rgb_root}/{scene_id}/{filename}_rgb.png

    Args:
        rgb_root: RGB 图像根目录（None 表示使用默认 RGB_ROOT）
        scene_id_min: 最小场景ID（包含），例如 0 表示从场景 "00" 开始
        scene_id_max: 最大场景ID（包含），例如 58 表示到场景 "58" 结束
        max_images_per_scene: 每个场景最多处理的图像数量（None 表示不限制）
        max_total_images: 总共最多处理的图像数量（None 表示不限制）

    产出：
        (scene_id, rgb_path: Path, rgb_filename: str)
    """
    rgb_dir = rgb_root or RGB_ROOT
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB 根目录不存在：{rgb_dir}")

    scene_dirs = sorted([p for p in rgb_dir.iterdir() if p.is_dir()])
    
    total_count = 0
    
    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        
        # 场景ID过滤：尝试将 scene_id 转换为整数进行范围检查
        try:
            scene_id_int = int(scene_id)
            if scene_id_min is not None and scene_id_int < scene_id_min:
                continue
            if scene_id_max is not None and scene_id_int > scene_id_max:
                continue
        except ValueError:
            # 如果 scene_id 不是纯数字，跳过过滤（保留所有非数字场景）
            pass
        
        # 获取该场景下的所有图像
        img_paths = sorted(scene_dir.glob("*_rgb.png"))
        
        # 限制每个场景的图像数量
        if max_images_per_scene is not None:
            img_paths = img_paths[:max_images_per_scene]
        
        # 遍历该场景的图像
        for img_path in img_paths:
            # 检查总数量限制
            if max_total_images is not None and total_count >= max_total_images:
                return  # 达到总数限制，停止生成
            
            yield scene_id, img_path, img_path.name
            total_count += 1
        
        # 如果已达到总数限制，提前退出
        if max_total_images is not None and total_count >= max_total_images:
            break


def generate_stage2_captions(
    rgb_root: Optional[str] = None,
    polar_root: Optional[str] = None,
    gt_root: Optional[str] = None,
    detection_method: str = "auto",
    scene_id_min: Optional[int] = None,
    scene_id_max: Optional[int] = None,
    max_images_per_scene: Optional[int] = None,
    max_total_images: Optional[int] = None,
    model_name: Optional[str] = None,
    output_json: Optional[str] = None,
    batch_size: int = 4,
):
    """
    生成 Stage 2 Physics-Aware Captions 的主函数。

    主要步骤：
    1. 加载本地 LLaVA 模型（4bit 量化）。
    2. 遍历所有 RGB 图像，并尝试通过 polar 图计算反光 bbox。
    3. 动态构造 Prompt，调用 LLaVA 生成描述。
    4. 保存为 LLaVA 兼容 JSON 格式。

    Args:
        rgb_root: RGB 图像根目录（None 表示使用默认 RGB_ROOT）
        polar_root: 偏振图像根目录（None 表示使用默认 POLAR_ROOT）
        scene_id_min: 最小场景ID（包含），例如 0 表示从场景 "00" 开始
        scene_id_max: 最大场景ID（包含），例如 58 表示到场景 "58" 结束
        max_images_per_scene: 每个场景最多处理的图像数量（None 表示不限制）
        max_total_images: 总共最多处理的图像数量（None 表示不限制）
        model_name: 模型路径（None 表示使用默认 LLAVA_MODEL_NAME）
        output_json: 输出 JSON 文件路径（None 表示使用默认 OUTPUT_JSON）
    """
    model_path = model_name or LLAVA_MODEL_NAME
    
    # 处理输出文件路径：如果未指定，自动添加时间戳
    if output_json is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{OUTPUT_JSON_PREFIX}_{timestamp}.json"
    else:
        output_path = output_json
    
    # 处理路径参数
    rgb_dir = Path(rgb_root) if rgb_root else RGB_ROOT
    polar_dir = Path(polar_root) if polar_root else POLAR_ROOT
    gt_dir = Path(gt_root) if gt_root else None
    data_root = rgb_dir.parent  # 假设 rgb_root 的父目录是数据根目录
    
    # 创建裁剪图像输出目录（分别保存 RGB、GT、Polar）
    rgb_crop_dir = data_root / RGB_CROP_DIR
    gt_crop_dir = data_root / GT_CROP_DIR
    polar_crop_dir = data_root / POLAR_CROP_DIR
    rgb_crop_dir.mkdir(parents=True, exist_ok=True)
    gt_crop_dir.mkdir(parents=True, exist_ok=True)
    polar_crop_dir.mkdir(parents=True, exist_ok=True)
    print(f"裁剪图像将保存到：")
    print(f"  - RGB crop: {rgb_crop_dir}")
    print(f"  - GT crop: {gt_crop_dir}")
    print(f"  - Polar crop: {polar_crop_dir}")
    
    # 验证检测方法（使用差分方法，GT vs RGB）
    if detection_method != "diff":
        print(f"警告：检测方法 '{detection_method}' 不支持，使用 'diff'（差分方法）")
        detection_method = "diff"
    
    print("使用差分方法进行眩光检测（GT vs RGB）")
    print("新方案：根据 bbox 裁剪图像，用 GT crop 生成描述（不包含坐标信息）")
    
    model, processor, device = load_llava_model(model_path)
    model.eval()
    
    # 使用 torch.compile 加速推理（PyTorch 2.0+）
    try:
        import torch
        if hasattr(torch, 'compile'):
            print("使用 torch.compile 加速模型推理...")
            model = torch.compile(model, mode="reduce-overhead")
            print("✓ torch.compile 已启用（预计加速 20-30%）")
    except Exception as e:
        print(f"警告：torch.compile 不可用或失败，继续使用原始模型：{e}")

    results = []

    rgb_items = list(iter_rgb_images(
        rgb_root=rgb_dir,
        scene_id_min=scene_id_min,
        scene_id_max=scene_id_max,
        max_images_per_scene=max_images_per_scene,
        max_total_images=max_total_images,
    ))
    
    print(f"\n共找到 RGB 图像 {len(rgb_items)} 张，将逐一生成 Stage 2 描述...")
    if scene_id_min is not None or scene_id_max is not None:
        print(f"  场景范围：{scene_id_min or '无限制'} - {scene_id_max or '无限制'}")
    if max_images_per_scene is not None:
        print(f"  每个场景最多：{max_images_per_scene} 张")
    if max_total_images is not None:
        print(f"  总数量限制：{max_total_images} 张")
    print(f"  批处理大小：{batch_size}（GPU显存充足时可增大以加速处理）")
    
    # ⚠️ 批处理优化：收集一批样本，批量调用 LLaVA 生成描述
    batch_items = []  # 存储一批待处理的样本信息
    
    def process_batch(batch_items, model, processor, device):
        """处理一批样本的 LLaVA 描述生成"""
        if not batch_items:
            return []
        
        batch_results = []
        
        # 准备批量输入
        batch_images = []
        batch_metadata = []  # 存储每个样本的元数据
        
        for item in batch_items:
            batch_images.append(item['image_for_llava'])
            batch_metadata.append({
                'scene_id': item['scene_id'],
                'rgb_filename': item['rgb_filename'],
                'base_name': item['base_name'],
                'rgb_crop_path': item['rgb_crop_path'],
                'gt_crop_path': item['gt_crop_path'],
                'bbox_norm': item['bbox_norm'],
                'polar_crop_paths': item['polar_crop_paths'],
                'data_root': item['data_root'],
            })
        
        # 批量生成描述
        try:
            llava_prompt = build_prompt_gt_content_cropped()
            inputs = build_llava_input_batch(processor, batch_images, llava_prompt, device)
            
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=True,
                )

            # 解码所有生成的描述
            input_length = inputs["input_ids"].shape[-1]
            generated_ids = output_ids[:, input_length:]
            captions_raw = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            
            # 处理每个样本的结果
            for i, (caption_raw, metadata) in enumerate(zip(captions_raw, batch_metadata)):
                final_caption = clean_caption(caption_raw.strip())
                
                # 确保描述长度合理（40-60字）
                words = final_caption.split()
                if len(words) > 60:
                    truncated = " ".join(words[:60])
                    last_period = truncated.rfind('.')
                    if last_period > 30:
                        final_caption = truncated[:last_period + 1]
                    else:
                        final_caption = truncated
                
                # 构建样本数据
                rel_rgb_crop_path = str(metadata['rgb_crop_path'].relative_to(metadata['data_root']))
                rel_gt_crop_path = str(metadata['gt_crop_path'].relative_to(metadata['data_root']))
                
                if metadata['bbox_norm'] is not None:
                    sample = {
                        "id": f"{metadata['scene_id']}_{metadata['rgb_filename']}_crop",
                        "image": rel_rgb_crop_path,
                        "gt_image": rel_gt_crop_path,
                        "scene_id": metadata['scene_id'],
                        "bbox_norm": metadata['bbox_norm'],
                        "polar_crop_paths": metadata['polar_crop_paths'],
                        "conversations": [
                            {
                                "from": "human",
                                "value": "Describe the image",
                            },
                            {
                                "from": "gpt",
                                "value": final_caption,
                            },
                        ],
                    }
                else:
                    sample = {
                        "id": f"{metadata['scene_id']}_{metadata['rgb_filename']}",
                        "image": rel_rgb_crop_path,
                        "gt_image": rel_gt_crop_path,
                        "scene_id": metadata['scene_id'],
                        "bbox_norm": None,
                        "polar_crop_paths": metadata['polar_crop_paths'],
                        "conversations": [
                            {
                                "from": "human",
                                "value": "Describe the image",
                            },
                            {
                                "from": "gpt",
                                "value": final_caption,
                            },
                        ],
                    }
                
                batch_results.append(sample)
        
        except Exception as e:
            # 如果批量处理失败，回退到逐个处理
            print(f"\n警告：批量处理失败，回退到逐个处理：{e}")
            traceback.print_exc()
            # 逐个处理这批样本
            for item in batch_items:
                try:
                    llava_prompt = build_prompt_gt_content_cropped()
                    inputs = build_llava_input(processor, item['image_for_llava'], llava_prompt, device)
                    
                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=MAX_NEW_TOKENS,
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            do_sample=True,
                        )
                    generated_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
                    caption_raw = processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0].strip()
                    final_caption = clean_caption(caption_raw)
                    
                    # 构建样本数据（与上面相同的逻辑）
                    words = final_caption.split()
                    if len(words) > 60:
                        truncated = " ".join(words[:60])
                        last_period = truncated.rfind('.')
                        if last_period > 30:
                            final_caption = truncated[:last_period + 1]
                        else:
                            final_caption = truncated
                    
                    rel_rgb_crop_path = str(item['rgb_crop_path'].relative_to(item['data_root']))
                    rel_gt_crop_path = str(item['gt_crop_path'].relative_to(item['data_root']))
                    
                    if item['bbox_norm'] is not None:
                        sample = {
                            "id": f"{item['scene_id']}_{item['rgb_filename']}_crop",
                            "image": rel_rgb_crop_path,
                            "gt_image": rel_gt_crop_path,
                            "scene_id": item['scene_id'],
                            "bbox_norm": item['bbox_norm'],
                            "polar_crop_paths": item['polar_crop_paths'],
                            "conversations": [
                                {"from": "human", "value": "Describe the image"},
                                {"from": "gpt", "value": final_caption},
                            ],
                        }
                    else:
                        sample = {
                            "id": f"{item['scene_id']}_{item['rgb_filename']}",
                            "image": rel_rgb_crop_path,
                            "gt_image": rel_gt_crop_path,
                            "scene_id": item['scene_id'],
                            "bbox_norm": None,
                            "polar_crop_paths": item['polar_crop_paths'],
                            "conversations": [
                                {"from": "human", "value": "Describe the image"},
                                {"from": "gpt", "value": final_caption},
                            ],
                        }
                    
                    batch_results.append(sample)
                except Exception as e2:
                    print(f"\n警告：处理样本失败，跳过：{item['rgb_path']}，错误：{e2}")
                    continue
        
        return batch_results

    for idx, (scene_id, rgb_path, rgb_filename) in enumerate(
        tqdm(rgb_items, desc="生成 Stage 2 GT Captions")
    ):
        try:
            # 1. 加载 RGB 图像（带眩光）
            rgb_image = Image.open(rgb_path).convert("RGB")
        except Exception as e:
            print(f"\n警告：加载 RGB 图像失败，跳过：{rgb_path}，错误：{e}")
            traceback.print_exc()
            continue

        # 2. 查找并加载 GT 图像（干净）
        gt_image_path = find_gt_image_path(scene_id, rgb_filename, rgb_dir, gt_dir)
        if gt_image_path is None:
            print(f"\n警告：找不到 GT 图像（scene {scene_id}, {rgb_filename}），跳过。")
            continue
        
        try:
            gt_image = Image.open(gt_image_path).convert("RGB")
        except Exception as e:
            print(f"\n警告：加载 GT 图像失败，跳过：{gt_image_path}，错误：{e}")
            traceback.print_exc()
            continue

        # 3. 使用差分方法检测眩光 bbox（GT vs RGB）
        bbox_norm = None
        try:
            bbox_norm = detect_glare_bbox_diff(
                rgb_path,
                gt_image_path,
                threshold=GLARE_THRESHOLD,
                min_area=MIN_GLARE_AREA,
                max_bbox_ratio=MAX_BBOX_RATIO,
            )
        except Exception:
            # 检测失败时，按无反光处理（不输出错误信息，这是正常情况）
            bbox_norm = None

        # 4. 根据 bbox 裁剪图像（新方案）
        rgb_crop = None
        gt_crop = None
        polar_crop_paths = {}  # 初始化为空字典，确保字段始终存在
        crop_saved = False
        rgb_crop_path = None
        gt_crop_path = None
        
        # 获取 base_name（用于文件名）
        base_name = rgb_path.stem
        if base_name.endswith('_rgb'):
            base_name = base_name[:-4]
        
        # 创建场景裁剪目录（统一创建，无论是否有反光）
        scene_rgb_crop_dir = rgb_crop_dir / scene_id
        scene_gt_crop_dir = gt_crop_dir / scene_id
        scene_polar_crop_dir = polar_crop_dir / scene_id
        scene_rgb_crop_dir.mkdir(parents=True, exist_ok=True)
        scene_gt_crop_dir.mkdir(parents=True, exist_ok=True)
        scene_polar_crop_dir.mkdir(parents=True, exist_ok=True)
        
        if bbox_norm is not None:
            # 有眩光：尝试裁剪 RGB、GT、Polar 图像
            # ⚠️ 关键修复：统一使用 512x512 保存所有 crop 图像，避免视野错位
            # RGB、GT、Polar 都使用相同的裁剪区域和尺寸，确保视野完全一致
            try:
                # 裁剪 RGB 图像并 resize 到 512x512（与 Polar 保持一致）
                rgb_crop = crop_image_by_bbox(rgb_image, bbox_norm, resize_to=CROP_IMAGE_SIZE)

                # 裁剪 GT 图像并 resize 到 512x512（与 Polar 保持一致）
                gt_crop = crop_image_by_bbox(gt_image, bbox_norm, resize_to=CROP_IMAGE_SIZE)

                # 裁剪 Polar 图像（需要先加载4个角度图像）
                polar_paths_dict = {
                    "I_0": polar_dir / scene_id / f"{base_name}_000.png",
                    "I_45": polar_dir / scene_id / f"{base_name}_045.png",
                    "I_90": polar_dir / scene_id / f"{base_name}_090.png",
                    "I_135": polar_dir / scene_id / f"{base_name}_135.png",
                }
                
                # 检查偏振图像是否存在
                all_polar_exist = all(p.exists() for p in polar_paths_dict.values())
                
                # 如果裁剪成功，保存裁剪后的图像
                if rgb_crop is not None and gt_crop is not None:
                    # 使用原始文件名（不加 crop 后缀）
                    rgb_crop_filename = f"{base_name}_rgb.png"
                    gt_crop_filename = rgb_filename  # GT 使用与 RGB 相同的文件名
                    
                    # 保存 RGB crop 到 rgb_crop/{scene_id}/
                    rgb_crop_path = scene_rgb_crop_dir / rgb_crop_filename
                    rgb_crop.save(rgb_crop_path)
                    
                    # 保存 GT crop 到 GT_crop/{scene_id}/
                    gt_crop_path = scene_gt_crop_dir / gt_crop_filename
                    gt_crop.save(gt_crop_path)
                    
                    # 裁剪并保存 Polar 图像（4个角度）到 polar_crop/{scene_id}/
                    polar_crop_paths = {}
                    if all_polar_exist:
                        for angle_name, polar_path in polar_paths_dict.items():
                            try:
                                polar_img = Image.open(polar_path).convert("RGB")
                                # ⚠️ 关键修复：Polar 图像 resize 到 512x512（与 RGB 保持一致，避免视野错位）
                                polar_crop = crop_image_by_bbox(polar_img, bbox_norm, resize_to=CROP_IMAGE_SIZE)
                                if polar_crop is not None:
                                    # 使用原始 Polar 文件名（例如：0000_000.png）
                                    polar_crop_filename = polar_path.name
                                    polar_crop_path = scene_polar_crop_dir / polar_crop_filename
                                    polar_crop.save(polar_crop_path)
                                    polar_crop_paths[angle_name] = str(polar_crop_path.relative_to(data_root))
                            except Exception:
                                # Polar 图像处理失败时，跳过该角度（不输出错误信息）
                                pass
                    
                    crop_saved = True
                else:
                    # 裁剪失败（bbox太小等）：以 bbox 中心为圆心，切出 512x512 区域（与 Polar 保持一致）
                    rgb_crop = crop_image_centered_on_bbox(rgb_image, bbox_norm, target_size=CROP_IMAGE_SIZE)
                    gt_crop = crop_image_centered_on_bbox(gt_image, bbox_norm, target_size=CROP_IMAGE_SIZE)
                    
                    if rgb_crop is not None and gt_crop is not None:
                        rgb_crop_filename = f"{base_name}_rgb.png"
                        gt_crop_filename = rgb_filename
                        
                        rgb_crop_path = scene_rgb_crop_dir / rgb_crop_filename
                        gt_crop_path = scene_gt_crop_dir / gt_crop_filename
                        
                        rgb_crop.save(rgb_crop_path)
                        gt_crop.save(gt_crop_path)
                        
                        # 处理 Polar 图像（以 bbox 中心为圆心切出 512x512，与 RGB 保持一致）
                        polar_paths_dict = {
                            "I_0": polar_dir / scene_id / f"{base_name}_000.png",
                            "I_45": polar_dir / scene_id / f"{base_name}_045.png",
                            "I_90": polar_dir / scene_id / f"{base_name}_090.png",
                            "I_135": polar_dir / scene_id / f"{base_name}_135.png",
                        }
                        
                        all_polar_exist = all(p.exists() for p in polar_paths_dict.values())
                        polar_crop_paths = {}
                        if all_polar_exist:
                            for angle_name, polar_path in polar_paths_dict.items():
                                try:
                                    polar_img = Image.open(polar_path).convert("RGB")
                                    # ⚠️ 关键修复：Polar 图像 resize 到 512x512（与 RGB 保持一致，避免视野错位）
                                    polar_crop = crop_image_centered_on_bbox(polar_img, bbox_norm, target_size=CROP_IMAGE_SIZE)
                                    if polar_crop is not None:
                                        polar_crop_filename = polar_path.name
                                        polar_crop_path = scene_polar_crop_dir / polar_crop_filename
                                        polar_crop.save(polar_crop_path)
                                        polar_crop_paths[angle_name] = str(polar_crop_path.relative_to(data_root))
                                except Exception:
                                    pass
                        
                        crop_saved = True
                    else:
                        # 如果连中心裁剪都失败，降级为 resize 原图到 512x512（与 Polar 保持一致）
                        rgb_resized = rgb_image.resize((CROP_IMAGE_SIZE, CROP_IMAGE_SIZE), Image.Resampling.LANCZOS)
                        gt_resized = gt_image.resize((CROP_IMAGE_SIZE, CROP_IMAGE_SIZE), Image.Resampling.LANCZOS)
                        
                        rgb_crop_filename = f"{base_name}_rgb.png"
                        gt_crop_filename = rgb_filename
                        
                        rgb_crop_path = scene_rgb_crop_dir / rgb_crop_filename
                        gt_crop_path = scene_gt_crop_dir / gt_crop_filename
                        
                        rgb_resized.save(rgb_crop_path)
                        gt_resized.save(gt_crop_path)
                        
                        rgb_crop = rgb_resized
                        gt_crop = gt_resized
                        crop_saved = True
                    
            except Exception as e:
                # 如果裁剪过程出现异常，尝试以 bbox 中心为圆心切出 512x512（与 Polar 保持一致）
                try:
                    if bbox_norm is not None:
                        rgb_crop = crop_image_centered_on_bbox(rgb_image, bbox_norm, target_size=CROP_IMAGE_SIZE)
                        gt_crop = crop_image_centered_on_bbox(gt_image, bbox_norm, target_size=CROP_IMAGE_SIZE)
                    else:
                        rgb_crop = None
                        gt_crop = None
                    
                    if rgb_crop is not None and gt_crop is not None:
                        rgb_crop_filename = f"{base_name}_rgb.png"
                        gt_crop_filename = rgb_filename
                        
                        rgb_crop_path = scene_rgb_crop_dir / rgb_crop_filename
                        gt_crop_path = scene_gt_crop_dir / gt_crop_filename
                        
                        rgb_crop.save(rgb_crop_path)
                        gt_crop.save(gt_crop_path)
                        
                        crop_saved = True
                    else:
                        # 如果连中心裁剪都失败，降级为 resize 原图到 512x512（与 Polar 保持一致）
                        rgb_resized = rgb_image.resize((CROP_IMAGE_SIZE, CROP_IMAGE_SIZE), Image.Resampling.LANCZOS)
                        gt_resized = gt_image.resize((CROP_IMAGE_SIZE, CROP_IMAGE_SIZE), Image.Resampling.LANCZOS)
                        
                        rgb_crop_filename = f"{base_name}_rgb.png"
                        gt_crop_filename = rgb_filename
                        
                        rgb_crop_path = scene_rgb_crop_dir / rgb_crop_filename
                        gt_crop_path = scene_gt_crop_dir / gt_crop_filename
                        
                        rgb_resized.save(rgb_crop_path)
                        gt_resized.save(gt_crop_path)
                        
                        rgb_crop = rgb_resized
                        gt_crop = gt_resized
                        crop_saved = True
                except Exception:
                    # 如果所有方法都失败，跳过这个样本
                    continue

        # 5. 处理无反光情况：将 RGB、GT、Polar 图像 resize 并保存到 crop 文件夹
        if bbox_norm is None:
            # ⚠️ 关键修复：无反光时，RGB/GT/Polar 都 resize 到 512x512，确保视野一致
            try:
                # Resize RGB 图像到 512x512 并保存（与 Polar 保持一致）
                rgb_resized = rgb_image.resize((CROP_IMAGE_SIZE, CROP_IMAGE_SIZE), Image.Resampling.LANCZOS)
                rgb_crop_filename = f"{base_name}_rgb.png"
                rgb_crop_path = scene_rgb_crop_dir / rgb_crop_filename
                rgb_resized.save(rgb_crop_path)
                
                # Resize GT 图像到 512x512 并保存（与 Polar 保持一致）
                gt_resized = gt_image.resize((CROP_IMAGE_SIZE, CROP_IMAGE_SIZE), Image.Resampling.LANCZOS)
                gt_crop_filename = rgb_filename  # GT 使用与 RGB 相同的文件名
                gt_crop_path = scene_gt_crop_dir / gt_crop_filename
                gt_resized.save(gt_crop_path)
                
                # Resize Polar 图像（4个角度）到 512x512 并保存（与 RGB 保持一致）
                polar_paths_dict = {
                    "I_0": polar_dir / scene_id / f"{base_name}_000.png",
                    "I_45": polar_dir / scene_id / f"{base_name}_045.png",
                    "I_90": polar_dir / scene_id / f"{base_name}_090.png",
                    "I_135": polar_dir / scene_id / f"{base_name}_135.png",
                }
                
                # 检查偏振图像是否存在
                all_polar_exist = all(p.exists() for p in polar_paths_dict.values())
                
                polar_crop_paths = {}
                if all_polar_exist:
                    for angle_name, polar_path in polar_paths_dict.items():
                        try:
                            polar_img = Image.open(polar_path).convert("RGB")
                            # ⚠️ 关键修复：Polar 图像 resize 到 512x512（与 RGB 保持一致，避免视野错位）
                            polar_resized = polar_img.resize((CROP_IMAGE_SIZE, CROP_IMAGE_SIZE), Image.Resampling.LANCZOS)
                            polar_crop_filename = polar_path.name  # 使用原始文件名
                            polar_crop_path = scene_polar_crop_dir / polar_crop_filename
                            polar_resized.save(polar_crop_path)
                            polar_crop_paths[angle_name] = str(polar_crop_path.relative_to(data_root))
                        except Exception:
                            # Polar 图像处理失败时，跳过该角度（不输出错误信息）
                            pass
                
                # 使用 resize 后的 RGB 图像生成描述（512x512，LLaVA 会自动处理）
                rgb_image_for_llava = rgb_resized
                crop_saved = True
                
            except Exception as e:
                print(f"\n警告：处理无反光图像失败，跳过：{rgb_path}，错误：{e}")
                traceback.print_exc()
                continue
        else:
            # 有眩光：使用 GT crop（已经 resize 到 512x512）
            # 如果裁剪失败，gt_crop 可能是 resize 后的原图，仍然可以使用
            if gt_crop is not None:
                rgb_image_for_llava = gt_crop
            else:
                # 如果 gt_crop 仍然为 None（不应该发生，但以防万一），使用 resize 后的原图到 512x512
                rgb_image_for_llava = rgb_image.resize((CROP_IMAGE_SIZE, CROP_IMAGE_SIZE), Image.Resampling.LANCZOS)

        # 6. 收集样本信息，准备批处理
        # 确保 rgb_crop_path 和 gt_crop_path 都存在（无论是裁剪还是resize）
        if not crop_saved or rgb_crop_path is None or gt_crop_path is None:
            # 如果处理失败，跳过这个样本
            continue
        
        # 将当前样本添加到批处理队列
        batch_items.append({
            'scene_id': scene_id,
            'rgb_filename': rgb_filename,
            'base_name': base_name,
            'rgb_path': rgb_path,
            'rgb_crop_path': rgb_crop_path,
            'gt_crop_path': gt_crop_path,
            'bbox_norm': bbox_norm,
            'polar_crop_paths': polar_crop_paths,
            'image_for_llava': rgb_image_for_llava,
            'data_root': data_root,
        })
        
        # 当批处理队列达到 batch_size 时，批量处理
        if len(batch_items) >= batch_size:
            batch_results = process_batch(batch_items, model, processor, device)
            results.extend(batch_results)
            batch_items = []  # 清空队列
    
    # 处理剩余的样本（不足一个 batch 的）
    if batch_items:
        batch_results = process_batch(batch_items, model, processor, device)
        results.extend(batch_results)

    # 7. 保存 JSON
    output_file = Path(output_path)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n================ 生成完成 ================")
    print(f"共成功生成 {len(results)} 条 Stage 2 GT Captions")
    print(f"输出文件：{output_file.resolve()}")
    
    # 统计信息
    # 通过 bbox_norm 字段判断是否有反光：bbox_norm 不为 None 表示有反光
    with_glare = sum(1 for r in results if r.get("bbox_norm") is not None)
    without_glare = len(results) - with_glare
    print(f"  - 有眩光区域（bbox_norm 不为 None）：{with_glare} 条")
    print(f"  - 无眩光区域（bbox_norm 为 None）：{without_glare} 条")
    print(f"  - 裁剪图像保存目录：")
    print(f"    * RGB crop: {rgb_crop_dir}")
    print(f"    * GT crop: {gt_crop_dir}")
    print(f"    * Polar crop: {polar_crop_dir}")


if __name__ == "__main__":
    """

    4. 运行脚本：
       - 完整生成：python generate_stage2_captions_llava.py
       - 测试生成（场景 00-10，每个场景 1 张，总共 10 张）：
             python generate_stage2_captions_llava.py --scene_id_min 0 --scene_id_max 10 --max_images_per_scene 1 --max_total_images 10
       - 结果将保存到 stage2_physics_captions.json 中。
    """

    parser = argparse.ArgumentParser(
        description="生成 Stage 2 GT Captions（差分检测 + GT 图直接描述，用于语义对齐）"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help=f"LLaVA 模型路径（默认：{LLAVA_MODEL_NAME}）",
    )
    parser.add_argument(
        "--rgb_root",
        type=str,
        default=None,
        help=f"RGB 图像根目录（默认：{RGB_ROOT}）",
    )
    parser.add_argument(
        "--polar_root",
        type=str,
        default=None,
        help=f"偏振图像根目录（默认：{POLAR_ROOT}）",
    )
    parser.add_argument(
        "--gt_root",
        type=str,
        default=None,
        help="GT图像根目录（用于差分检测方法，默认：自动查找 data/GT 或 rgb_root/../GT）",
    )
    parser.add_argument(
        "--detection_method",
        type=str,
        default="diff",
        choices=["diff"],
        help="反光检测方法：'diff'（差分方法，GT vs RGB，默认）",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help=f"输出 JSON 文件路径（默认：自动生成带时间戳的文件名，格式：{OUTPUT_JSON_PREFIX}_YYYYMMDD_HHMMSS.json）",
    )
    parser.add_argument(
        "--scene_id_min",
        type=int,
        default=None,
        help="最小场景ID（包含），例如 0 表示从场景 '00' 开始",
    )
    parser.add_argument(
        "--scene_id_max",
        type=int,
        default=None,
        help="最大场景ID（包含），例如 58 表示到场景 '58' 结束",
    )
    parser.add_argument(
        "--max_images_per_scene",
        type=int,
        default=None,
        help="每个场景最多处理的图像数量（None 表示不限制）",
    )
    parser.add_argument(
        "--max_total_images",
        type=int,
        default=None,
        help="总共最多处理的图像数量（None 表示不限制）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help=f"批处理大小，用于加速处理（默认：4，GPU显存充足时可增大到8-16）",
    )

    args = parser.parse_args()

    # 为了在不破坏类型检查的前提下使用 torch.no_grad，这里延迟导入 torch
    import torch  # noqa: E402

    generate_stage2_captions(
        rgb_root=args.rgb_root,
        polar_root=args.polar_root,
        gt_root=args.gt_root,
        detection_method=args.detection_method,
        scene_id_min=args.scene_id_min,
        scene_id_max=args.scene_id_max,
        max_images_per_scene=args.max_images_per_scene,
        max_total_images=args.max_total_images,
        model_name=args.model_name,
        output_json=args.output_json,
        batch_size=args.batch_size,
    )



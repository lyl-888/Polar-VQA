"""
Stage 3 QA 对生成脚本
使用本地 Qwen2-VL-72B-Instruct-AWQ 模型，对 RGB + GT 数据集生成用于 PolarVLM Stage 3 训练的问答对。

核心思路：
- 使用差分方法（GT vs RGB）检测眩光差异，辅助数据筛选
- 生成 3 类物理感知 QA 对（A/B/C）：
  1. Task A: Positive (Transmission Description)
  2. Task B: Physics (Reflection Identification)
  3. Task C: Hard Negative (Counterfactual / Anti-Deception)
- 使用非裁剪图像（原始 RGB 和 GT 图像）
- 所有提示词和生成内容使用英文

工作流程：
1. 加载 RGB 和 GT 图像 → 2. 使用差分方法检测眩光 bbox → 3. 调用 Qwen2-VL 生成 3 类 QA 对 → 4. 保存为训练数据

依赖：
- transformers >= 4.40（支持 Qwen2VLForConditionalGeneration / AutoProcessor）
- qwen-vl-utils（用于 process_vision_info）
- auto-gptq（AWQ 量化支持）
- pillow, numpy, opencv-python, tqdm

作者：AI Assistant
日期：2025
"""

import argparse
import json
import random
import re
import traceback
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

# [新增] AutoAWQ 兼容性检查
# transformers 4.40+ 内置了 AWQ 支持，不需要 AutoAWQ 库
# 但 transformers 在检测到 AWQ 配置时可能会尝试使用 AutoAWQ（如果已安装）
# AutoAWQ 与 transformers 4.57.3 不兼容（缺少 PytorchGELUTanh）
# 解决方法：卸载 AutoAWQ 库，让 transformers 使用内置的 AWQ 支持
try:
    import awq
    print("⚠ Warning: AutoAWQ is installed. Ensure its version is compatible with your transformers.")
    print("  → If loading fails with AWQ-related errors, try: pip install -U autoawq")
except ImportError:
    pass  # AutoAWQ 未安装，这是正常的

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
)
from qwen_vl_utils import process_vision_info

# 全局开关：首次生成异常时打印完整 traceback，便于定位环境兼容问题
_PRINTED_GEN_TRACEBACK = False

# ==================== 配置参数 ====================

# [修改] 模型路径（Qwen2-VL-72B-Instruct-AWQ）
QWEN_MODEL_NAME = "/openbayes/input/input0/models/Qwen2-72B-Instruct-AWQ"

# 数据根目录
DATA_ROOT = Path("/openbayes/input/input0")
RGB_ROOT = DATA_ROOT / "rgb"      # 结构：rgb/{scene_id}/{filename}_rgb.png
GT_ROOT = DATA_ROOT / "GT"        # 结构：GT/{scene_id}/{filename}_rgb.png
POLAR_ROOT = DATA_ROOT / "polar"  # 结构：polar/{scene_id}/{filename}_{angle}.png

# 输出 JSON 文件（默认会加上时间戳）
OUTPUT_JSON_PREFIX = "stage3_qa_pairs"

# 反光检测参数（差分方法）
GLARE_THRESHOLD = 100          # 眩光检测阈值（提高阈值，只抓最亮的核心反光区域）
MIN_GLARE_AREA = 500           # 最小眩光区域面积（像素）
MAX_BBOX_RATIO = 0.6           # 最大边界框面积比例（相对于整张图像）
RELAX_GLARE_THRESHOLD = 15     # 反光检测兜底阈值（更敏感）
RELAX_MIN_GLARE_AREA = 200     # 反光检测兜底最小面积
RELAX_MAX_BBOX_RATIO = 0.98    # 兜底允许更大的区域（超大反光）
PERCENTILE_LEVELS = [99.5, 99, 98, 97, 95, 93, 90]  # 热点兜底的百分位
PERCENTILE_MAX_BBOX_RATIO = 0.8                      # 百分位兜底允许的最大框比例
FORCE_BBOX_RATIO = 0.25                               # 仍失败时，基于热点强制框的面积比例
BORDER_IGNORE_RATIO = 0.02                            # 忽略边缘差分，避免边缘噪声成框
MIN_BBOX_SIDE_RATIO = 0.03                            # bbox 最小边长比例，避免过细条纹
USE_POSITIVE_DIFF = True                              # 仅保留 RGB 比 GT 更亮的差分（更贴近反光）
DIFF_BLUR_KERNEL = 3                                  # 差分轻度平滑，减少噪声（0 表示关闭）
FOCUS_GLARE_OBJECT = True                             # 让模型自行寻找被反光遮挡的核心物体

# [修改] 生成参数（Qwen2-VL 优化）
MAX_NEW_TOKENS = 256           # 限制输出长度，避免长文本拖慢推理
TEMPERATURE = 0.2              # Qwen2-VL 比较聪明，0.1-0.3 就能生成很好的多样性，0.5 可能会有点野
TOP_P = 0.01                   # Qwen 官方推荐低 Top_P 或 Top_K
TEMPERATURE_RETRY = 0.4        # 重试时的温度（稍微提高随机性）
REPETITION_PENALTY = 1.05      # 抑制重复 token，缓解“复读机”现象

# [关键修改] 加速参数
BATCH_SIZE = 1                 # [关键] 72B AWQ 占用 ~42GB 显存，推理时 KV Cache 还需要空间，48GB 显卡只能跑 Batch Size = 1
USE_TORCH_COMPILE = False      # Qwen2-VL 不需要 torch.compile 也能跑得很快，且 compile 对动态分辨率支持一般
DEDUP_NO_GLARE_BY_SCENE = False  # 是否对无眩光样本按 scene 去重（默认关闭，保证每图都有输出）

# [新增] 图像分辨率限制（Qwen2-VL 支持动态分辨率，但为了省显存需要限制最大尺寸）
MAX_IMAGE_SIZE = 1024          # 限制最大边长为 1024（如果显存 OOM，可以改为 768 或 512）
MIN_IMAGE_SIDE = 28            # Qwen2-VL 视觉分块因子相关，短边过小可能触发 height/width 错误

# Debug: bbox 可视化
DEBUG_BBOX = False             # 是否保存 bbox 可视化图
DEBUG_BBOX_DIR = "debug_bboxes"  # 输出目录（相对 output_json 所在目录）

# Task A Positive Prompt 模板（英文）
POSITIVE_PROMPT_TEMPLATES = [
    "Describe the {object} obscured by the glare.",
    "What does the {object} look like beneath the glare?",
    "Describe the true {object} in the scene and ignore reflections.",
    "Identify the {object} hidden by glare on the surface.",
    "Briefly describe the real {object} visible in the clear view.",
]

# Task B Physics Prompt 模板（英文）
PHYSICS_PROMPT_TEMPLATES = [
    "Identify the reflection of the {object} based on the polarization map.",
    "Where is the reflected {object} visible according to the high polarization signal?",
    "Describe the glare or reflection of the {object} indicated by the polarization data.",
    "Which {object} appears only as a reflection on the surface?",
    "What reflected {object} is suggested by the polarization map?",
]

# Task C Hard Negative Prompt 模板（英文）
NEGATIVE_PROMPT_TEMPLATES = [
    # “陷阱题”应该尽量像普通问句，避免在问题里直接提示 glare/reflection/polarization
    "Is there a {object} in this scene?",
    "Is there a {object} present here?",
    "Do you see a {object} here?",
    "Is there actually a {object} there?",
    # 少量保留“surface”相关问法（覆盖强高光/白斑场景），但不作为主力
    "Is there really a {object} on the surface?",
]

# ==================== 物体名抽取与模板填充 ====================

_GENERIC_OBJECT_NAMES = {
    "object", "reflection", "reflections", "glare", "surface", "area", "region",
    "spot", "bright area", "bright spot", "light", "highlight", "reflected object",
    "glare spot", "reflection spot",
}


def _normalize_object_name(name: str) -> Optional[str]:
    if not isinstance(name, str):
        return None
    cleaned = re.sub(r'^\s*(a|an|the)\s+', '', name.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'[\s\.,;:!]+$', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if not cleaned:
        return None
    words = cleaned.split()
    if len(words) > 5:
        cleaned = " ".join(words[:5])
    if cleaned.lower() in _GENERIC_OBJECT_NAMES:
        return None
    return cleaned


def _extract_object_from_text(text: str) -> Optional[str]:
    """尽量从答案里抽取具体物体名（person/car/sign 等）。"""
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = re.sub(r'^\s*No\.\s*', '', text.strip(), flags=re.IGNORECASE)
    patterns = [
        r'Although\s+(?:a|an|the)\s+([a-zA-Z0-9\- ]{2,80}?)\s+(?:is|was|appears|appearing|seems)\b',
        r'(?:reflection|glare)\s+(?:of|from)\s+(?:a|an|the)?\s*([a-zA-Z0-9\- ]{2,80}?)\b',
        r'(?:appearing|appears)\s+(?:as|like)\s+(?:a|an|the)?\s*([a-zA-Z0-9\- ]{2,80}?)\b',
        r'(?:looks|looking)\s+like\s+(?:a|an|the)?\s*([a-zA-Z0-9\- ]{2,80}?)\b',
        r'^(?:a|an|the)\s+([a-zA-Z0-9\- ]{2,80}?)(?:,|\.|\bwith\b|\bon\b|\bin\b|\bat\b|\bis\b|\bare\b)',
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            candidate = _normalize_object_name(match.group(1))
            if candidate:
                return candidate
    # fallback: 取前 3-4 个词
    fallback = " ".join(cleaned.split()[:4])
    return _normalize_object_name(fallback)


def _format_object_question(templates: List[str], object_name: Optional[str], fallback_object: str) -> str:
    template = random.choice(templates)
    obj = object_name or fallback_object
    if "{object}" in template:
        return template.format(object=obj)
    return template.replace("[OBJECT]", obj).replace("<OBJECT>", obj)


def _question_mentions_object(question: str, object_name: Optional[str]) -> bool:
    if not isinstance(question, str) or not question.strip() or not object_name:
        return False
    q_lower = question.lower()
    obj_lower = object_name.lower()
    if obj_lower in q_lower:
        return True
    last_token = obj_lower.split()[-1] if obj_lower.split() else ""
    return bool(last_token and last_token in q_lower)


def _ensure_object_bound_question(
    qa_item: Dict,
    templates: List[str],
    fallback_object: str,
) -> None:
    if not isinstance(qa_item, dict):
        return
    question = qa_item.get("q", "")
    answer = qa_item.get("a", "")
    object_name = _extract_object_from_text(answer)
    if not question or "{object}" in question or "[OBJECT]" in question or "<OBJECT>" in question:
        qa_item["q"] = _format_object_question(templates, object_name, fallback_object)
        return
    # 若问题过于泛化且未包含具体物体名，则替换为占位模板
    generic_hits = re.search(r"\b(object|reflection|glare|bright area|surface)\b", question, flags=re.IGNORECASE)
    if generic_hits and not _question_mentions_object(question, object_name):
        qa_item["q"] = _format_object_question(templates, object_name, fallback_object)

# System Prompt（英文，Physics-Aware A/B/C）
def build_system_prompt() -> str:
    """
    Constructs the System Prompt for Physics-Aware Instruction Tuning (A/B/C Tasks)
    No bounding boxes. Pure semantic comparison and counterfactual reasoning.
    Adapted for general surface glare and reflections (not limited to glass).
    """
    return f"""
I will present two images of the same scene:
- **Image 1 (RGB)**: The actual observation, which may contain strong glare, specular reflections, or visual disturbances on the surface.
- **Image 2 (GT)**: The clear Ground Truth image showing ONLY the true object without any glare or reflections.

**Task:**
You are an expert AI training data creator. Your job is to generate a specialized Visual Question Answering (VQA) dataset to train a "Polarization-Aware Vision-Language Model".
The target model will receive an RGB image AND a polarization map. In physical terms, high polarization typically indicates glare or surface reflection, while low polarization indicates the true underlying object.
You must use Image 1 and Image 2 to simulate the answers this target model should learn to produce.

**Step 1: Image Comparison (Mental Step)**
1. Look at Image 2 (GT) to understand the TRUE object that is actually there.
2. Compare Image 1 to Image 2 to strictly identify the **GLARE or REFLECTION** (visible in Image 1 but missing in Image 2), such as reflected trees, buildings, or bright light spots obscuring the surface.

**Step 2: Generate 3 Specific QA Pairs based on the A/B/C Strategy**

* **Task A: Positive (True Object Description)**
  * **Goal**: Teach the model to describe the true object and ignore the glare/reflections.
  * **Focus**: Compare Image 1 and Image 2, and locate the object that is obscured by glare in Image 1 but appears clearly in Image 2.
  * **Q**: Use instruction-style questions (vary phrasing slightly). You MUST name the specific object in the question (e.g., person, car, sign). Do NOT mention Image 1/2 or GT/RGB.
  * **A**: A concise one-sentence description of the object that is visible clearly in **Image 2 (GT)** but is partially/fully obscured by glare in **Image 1 (RGB)**. Do NOT mention reflections or polarization here.

* **Task B: Physics Awareness (Glare/Reflection Identification)**
  * **Goal**: Teach the model to explicitly locate and describe the glare/reflection using polarization logic.
  * **Q**: Ask to identify the reflection or glare based on the polarization map (vary phrasing slightly). You MUST include the specific reflected object in the question (e.g., person, car, sign).
  * **A**: Describe what the glare/reflection looks like (based on differences between Image 1 and Image 2) and roughly where it appears. You **MUST** end the answer by stating the physics reason.
  * *Example ending*: "This is indicated by the high polarization signal in that area."

* **Task C: Hard Negative (Anti-Deception)**
  * **Goal**: Trick the model into hallucinating the reflected object or glare spot as a real object, and teach it to say "No".
  * **Q**: Ask a natural existence question like "Is there a [Reflected Object] here?" Replace [Reflected Object] with a specific thing you found in the reflection of Image 1, and make sure the object name appears in the question. Do NOT mention glare, reflection, surface, or polarization in the question.
  * **A**: You **MUST** start with "No." and explain the physics.
  * *Example format*: "No. Although a [Reflected Object] is visible, the high polarization indicates it is merely a reflection or glare on the surface, not a real physical object."

**IMPORTANT:**
- All questions and answers must be in English.
- Do NOT include numeric coordinates.
- Do NOT mention Image 1/2 or GT/RGB in the final Q/A text.

**Output Format (Strict Constraint):**
You must return ONLY a valid JSON object. Do not output any other text, explanations, or Markdown tags.

{{
  "qa_positive": {{"q": "...", "a": "..."}},
  "qa_physics": {{"q": "...", "a": "..."}},
  "qa_negative": {{"q": "...", "a": "..."}}
}}

**IMPORTANT: Output JSON directly, no other content.**

"""


# ==================== 反光检测相关函数 ====================

def _apply_border_ignore(diff: np.ndarray, ratio: float) -> np.ndarray:
    """忽略图像边缘差分，减少边缘噪声导致的细条框。"""
    if ratio is None or ratio <= 0:
        return diff
    h, w = diff.shape[:2]
    margin_x = int(w * ratio)
    margin_y = int(h * ratio)
    if margin_x > 0:
        diff[:, :margin_x] = 0
        diff[:, w - margin_x :] = 0
    if margin_y > 0:
        diff[:margin_y, :] = 0
        diff[h - margin_y :, :] = 0
    return diff


def _bbox_too_thin(w: int, h: int, w_img: int, h_img: int) -> bool:
    """过滤过细/过矮的 bbox（常见于边缘噪声条纹）。"""
    if w_img <= 0 or h_img <= 0:
        return True
    return (w / w_img) < MIN_BBOX_SIDE_RATIO or (h / h_img) < MIN_BBOX_SIDE_RATIO


def _expand_bbox_to_min_size(
    x: int, y: int, w: int, h: int, w_img: int, h_img: int
) -> tuple[int, int, int, int]:
    """将 bbox 扩展到最小边长，避免极细条纹框。"""
    min_w = max(1, int(w_img * MIN_BBOX_SIDE_RATIO))
    min_h = max(1, int(h_img * MIN_BBOX_SIDE_RATIO))
    if w >= min_w and h >= min_h:
        return x, y, w, h
    cx = x + w / 2.0
    cy = y + h / 2.0
    w = max(w, min_w)
    h = max(h, min_h)
    x = int(round(cx - w / 2.0))
    y = int(round(cy - h / 2.0))
    x = max(0, min(w_img - 1, x))
    y = max(0, min(h_img - 1, y))
    x2 = max(x + 1, min(w_img, x + w))
    y2 = max(y + 1, min(h_img, y + h))
    return x, y, x2 - x, y2 - y


def _contour_mean_diff(diff: np.ndarray, contour: np.ndarray) -> float:
    """计算 contour 区域内的平均差分强度。"""
    mask = np.zeros(diff.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    values = diff[mask > 0]
    if values.size == 0:
        return 0.0
    return float(values.mean())


def _pick_best_bbox_from_contours(
    diff: np.ndarray,
    contours: List[np.ndarray],
    min_area: int,
    max_bbox_ratio: float,
    w_img: int,
    h_img: int,
    allow_expand: bool,
) -> Optional[tuple[int, int, int, int]]:
    """从轮廓中选择“最强反光”的 bbox（按强度优先，其次面积）。"""
    img_area = w_img * h_img
    best = None
    best_score = -1.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        if rect_area <= 0:
            continue
        if rect_area > (img_area * max_bbox_ratio):
            continue
        if _bbox_too_thin(w, h, w_img, h_img):
            if allow_expand:
                x, y, w, h = _expand_bbox_to_min_size(x, y, w, h, w_img, h_img)
                rect_area = w * h
                if rect_area > (img_area * max_bbox_ratio):
                    continue
            else:
                continue
        mean_diff = _contour_mean_diff(diff, contour)
        score = mean_diff * (area ** 0.5)
        if score > best_score:
            best_score = score
            best = (x, y, w, h)
    return best

def detect_glare_bbox_diff(
    input_image_path: Path,
    gt_image_path: Path,
    threshold: int = GLARE_THRESHOLD,
    min_area: int = MIN_GLARE_AREA,
    max_bbox_ratio: float = MAX_BBOX_RATIO,
):
    """
    使用计算机视觉方法检测眩光区域并返回边界框（差分方法）。
    与旧代码保持一致，不使用形态学操作，确保检测结果与之前调 API 生成的数据框一致。
    
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
        
        # 差分：优先保留 RGB 比 GT 更亮的区域（更贴近反光）
        diff = cv2.subtract(input_gray, gt_gray) if USE_POSITIVE_DIFF else cv2.absdiff(input_gray, gt_gray)
        if DIFF_BLUR_KERNEL and DIFF_BLUR_KERNEL >= 3:
            k = int(DIFF_BLUR_KERNEL) | 1
            diff = cv2.GaussianBlur(diff, (k, k), 0)
        diff = _apply_border_ignore(diff, BORDER_IGNORE_RATIO)
        
        # 应用阈值创建二值掩码
        _, binary_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # [移除形态学操作] 为了与旧代码保持一致，移除 MORPH_OPEN 和 MORPH_CLOSE
        # 旧代码没有形态学操作，直接使用原始二值掩码进行轮廓检测
        # 这样可以确保检测结果与之前调 API 生成的数据框一致
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        h_img, w_img = input_gray.shape[:2]
        best_bbox = _pick_best_bbox_from_contours(
            diff,
            contours,
            min_area=min_area,
            max_bbox_ratio=max_bbox_ratio,
            w_img=w_img,
            h_img=h_img,
            allow_expand=False,
        )
        if best_bbox is None:
            return None
        x, y, w, h = best_bbox
        
        # [修改] 使用标准格式 [xmin, ymin, xmax, ymax]（与 LLaVA 格式兼容）
        xmin_n = float(x / w_img)
        xmax_n = float((x + w) / w_img)
        ymin_n = float(y / h_img)
        ymax_n = float((y + h) / h_img)
        
        xmin_n = float(np.clip(xmin_n, 0.0, 1.0))
        xmax_n = float(np.clip(xmax_n, 0.0, 1.0))
        ymin_n = float(np.clip(ymin_n, 0.0, 1.0))
        ymax_n = float(np.clip(ymax_n, 0.0, 1.0))
        
        # 返回标准格式 [xmin, ymin, xmax, ymax]
        return [xmin_n, ymin_n, xmax_n, ymax_n]
        
    except Exception:
        return None


def detect_glare_bbox_percentile(
    input_image_path: Path,
    gt_image_path: Path,
    percentiles: List[float] = PERCENTILE_LEVELS,
    min_area: int = MIN_GLARE_AREA,
    max_bbox_ratio: float = PERCENTILE_MAX_BBOX_RATIO,
    force_bbox_ratio: float = FORCE_BBOX_RATIO,
) -> Optional[List[float]]:
    """
    兜底方案：对差分图使用高分位阈值，提取“最强反光热点”区域的 bbox。
    当常规阈值检测失败或框过大时，仍然尽量输出非全图 bbox。
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

        diff = cv2.subtract(input_gray, gt_gray) if USE_POSITIVE_DIFF else cv2.absdiff(input_gray, gt_gray)
        if DIFF_BLUR_KERNEL and DIFF_BLUR_KERNEL >= 3:
            k = int(DIFF_BLUR_KERNEL) | 1
            diff = cv2.GaussianBlur(diff, (k, k), 0)
        diff = _apply_border_ignore(diff, BORDER_IGNORE_RATIO)
        h_img, w_img = diff.shape[:2]
        img_area = h_img * w_img
        if img_area <= 0:
            return None

        diff_max = float(diff.max())
        if diff_max <= 0:
            return None

        diff_flat = diff.reshape(-1)

        for p in percentiles:
            thr = np.percentile(diff_flat, p)
            if thr <= 0:
                continue
            mask = (diff >= thr).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            best_bbox = _pick_best_bbox_from_contours(
                diff,
                contours,
                min_area=min_area,
                max_bbox_ratio=max_bbox_ratio,
                w_img=w_img,
                h_img=h_img,
                allow_expand=True,
            )
            if best_bbox is None:
                continue
            x, y, w, h = best_bbox

            xmin_n = float(x / w_img)
            xmax_n = float((x + w) / w_img)
            ymin_n = float(y / h_img)
            ymax_n = float((y + h) / h_img)
            xmin_n = float(np.clip(xmin_n, 0.0, 1.0))
            xmax_n = float(np.clip(xmax_n, 0.0, 1.0))
            ymin_n = float(np.clip(ymin_n, 0.0, 1.0))
            ymax_n = float(np.clip(ymax_n, 0.0, 1.0))
            return [xmin_n, ymin_n, xmax_n, ymax_n]

        # 若分位阈值仍失败，使用热点像素强制构造一个中等大小的 bbox
        max_idx = int(np.argmax(diff_flat))
        cy, cx = divmod(max_idx, w_img)
        target_area = max(1, int(img_area * force_bbox_ratio))
        side = int(np.sqrt(target_area))
        min_w = max(1, int(w_img * MIN_BBOX_SIDE_RATIO))
        min_h = max(1, int(h_img * MIN_BBOX_SIDE_RATIO))
        side = max(side, min_w, min_h)
        half = max(1, side // 2)
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w_img - 1, cx + half)
        y2 = min(h_img - 1, cy + half)
        if x2 <= x1 or y2 <= y1:
            return None
        xmin_n = float(x1 / w_img)
        xmax_n = float(x2 / w_img)
        ymin_n = float(y1 / h_img)
        ymax_n = float(y2 / h_img)
        return [xmin_n, ymin_n, xmax_n, ymax_n]

    except Exception:
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
        if not gt_scene_dir.exists():
            continue

        possible_gt_names = [
            rgb_filename,
            f"{prefix}_rgb.png",
        ]
        for gt_name in possible_gt_names:
            candidate_path = gt_scene_dir / gt_name
            if candidate_path.exists():
                return candidate_path

        # 兜底：如果该场景只有一张 GT 图，则使用它并给出提示
        try:
            gt_candidates = sorted(gt_scene_dir.glob("*_rgb.png"))
        except Exception:
            gt_candidates = []
        if len(gt_candidates) == 1:
            # 数据集场景可能只有一张 GT，对应多张 RGB（反光位置不同）
            return gt_candidates[0]
    
    return None


def format_bbox_int(bbox_norm: list) -> str:
    """
    将归一化坐标 [0.0-1.0] 转换为 Qwen2-VL 原生偏好的 [0-1000] 整数格式
    
    用于构建给 Qwen 模型看的 prompt，提高视觉定位准确性
    
    Args:
        bbox_norm: [xmin, ymin, xmax, ymax]（归一化坐标，0.0-1.0）
    
    Returns:
        格式化的字符串，例如 "[100, 200, 300, 400]"
    """
    xmin, ymin, xmax, ymax = bbox_norm
    
    # 转换为 0-1000 的整数
    x1 = int(xmin * 1000)
    y1 = int(ymin * 1000)
    x2 = int(xmax * 1000)
    y2 = int(ymax * 1000)
    
    # 边界保护：确保不超出 0-1000 范围
    x1 = max(0, min(1000, x1))
    y1 = max(0, min(1000, y1))
    x2 = max(0, min(1000, x2))
    y2 = max(0, min(1000, y2))
    
    # Qwen2-VL 偏好的格式是纯数字列表字符串
    return f"[{x1}, {y1}, {x2}, {y2}]"


def format_bbox_string(bbox_norm: list) -> str:
    """
    将归一化 bbox 格式化为字符串（标准格式，用于保存到 JSON）
    
    保留 0-1.0 小数格式，与 LLaVA 训练格式兼容
    
    Args:
        bbox_norm: [xmin, ymin, xmax, ymax]（标准格式，0.0-1.0）
    
    Returns:
        格式化的字符串，例如 "[0.100, 0.200, 0.300, 0.400]"
    """
    xmin, ymin, xmax, ymax = bbox_norm
    # 使用标准格式 [xmin, ymin, xmax, ymax]，小数格式，与 LLaVA 兼容
    return f"[{xmin:.3f}, {ymin:.3f}, {xmax:.3f}, {ymax:.3f}]"


def build_stage2_style_records(
    rgb_path: Path,
    gt_image_path: Path,
    scene_id: str,
    bbox_norm: list,
    qa_pairs: Dict,
    sample_type: str = "typeB_visual",
) -> List[Dict]:
    """
    将内部 qa_pairs 结构转换为 Stage2 风格的扁平 JSON 记录列表。

    输出格式对齐 `merged_stage2_new_format_rewrite.json`：
    - 一条 QA 对对应一条样本记录
    - 字段包含 image / input_path / gt_path / scene_id / type / subtype / bbox_norm / conversations
    """
    records: List[Dict] = []

    mapping = [
        ("positive", qa_pairs.get("qa_positive")),
        ("physics", qa_pairs.get("qa_physics")),
        ("negative", qa_pairs.get("qa_negative")),
    ]

    image_rel = str(gt_image_path.relative_to(DATA_ROOT))
    input_rel = str(rgb_path.relative_to(DATA_ROOT))
    gt_rel = image_rel

    for subtype, qa_item in mapping:
        if not isinstance(qa_item, dict):
            continue
        q = qa_item.get("q")
        a = qa_item.get("a")
        if not isinstance(q, str) or not q.strip():
            continue
        if not isinstance(a, str) or not a.strip():
            continue

        records.append(
            {
                "image": image_rel,
                "input_path": input_rel,
                "gt_path": gt_rel,
                "scene_id": str(scene_id),
                "type": sample_type,
                "subtype": subtype,
                "bbox_norm": bbox_norm,
                "conversations": [
                    {"from": "human", "value": q.strip()},
                    {"from": "gpt", "value": a.strip()},
                ],
            }
        )

    return records


def draw_bbox_on_image(image: Image.Image, bbox_norm: list, color="red", width=8) -> Image.Image:
    """
    在图像上绘制显眼的红色边界框（不修改原图）
    
    用于给 Qwen2-VL 模型提供视觉提示，提高定位准确性
    
    Args:
        image: 原始图像（PIL Image）
        bbox_norm: 归一化边界框 [xmin, ymin, xmax, ymax]（0.0-1.0）
        color: 框的颜色（默认红色，最显眼）
        width: 线宽（默认 8，较粗以便模型容易识别）
    
    Returns:
        绘制了边界框的图像副本（不修改原图）
    """
    # 复制图像，不修改原图
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    w, h = img_draw.size
    xmin, ymin, xmax, ymax = bbox_norm
    
    # 转换为绝对坐标
    x1 = xmin * w
    y1 = ymin * h
    x2 = xmax * w
    y2 = ymax * h
    
    # 绘制矩形（outline=颜色，width=线宽）
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    
    return img_draw


def save_debug_bbox_image(
    image: Image.Image,
    bbox_norm: Optional[list],
    output_path: Path,
) -> None:
    """保存 bbox 可视化图像（bbox=None 时保存原图）。"""
    try:
        img = image.copy()
        if bbox_norm is not None:
            img = draw_bbox_on_image(img, bbox_norm, color="red", width=6)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
    except Exception:
        pass


def safe_resize(img: Image.Image, min_size: int = 224, max_size: int = 1024) -> Image.Image:
    """
    安全缩放图像：保持宽高比，无黑边填充，并强制对齐到 28 的倍数。
    完美适配 Qwen2-VL 的动态分辨率要求。
    """
    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size: {img.size}")

    # 限制最大边（防止显存 OOM）
    if w > max_size or h > max_size:
        ratio = max_size / max(w, h)
        w = int(w * ratio)
        h = int(h * ratio)

    # 限制最小边（防止触发 height/width must be > 0 的报错）
    if w < min_size or h < min_size:
        ratio = min_size / min(w, h)
        w = int(w * ratio)
        h = int(h * ratio)

    # Qwen2-VL 的 Patch Size 是 28，强制长宽必须是 28 的整数倍
    w = max(28, round(w / 28) * 28)
    h = max(28, round(h / 28) * 28)

    return img.resize((w, h), Image.Resampling.LANCZOS)


def normalize_qwen_batch_inputs(inputs, device, model_dtype):
    """
    规范 Qwen2-VL 输入 dtype，避免 generate 阶段出现
    'expected scalar type Int but found Half'。
    """
    import torch

    int_key_hints = (
        "input_ids",
        "attention_mask",
        "position_ids",
        "token_type_ids",
        "grid_thw",
        "cross_attention_mask",
        "image_attention_mask",
        "vision_attention_mask",
    )
    # 仅对明确的像素张量做 half，避免误把应为整型的辅助张量转成 half
    float_key_exact = {"pixel_values", "pixel_values_videos"}
    for k in list(inputs.keys()):
        v = inputs[k]
        if not torch.is_tensor(v):
            continue
        if any(hint in k for hint in int_key_hints):
            inputs[k] = v.to(device=device, dtype=torch.long)
        elif k in float_key_exact and v.is_floating_point():
            inputs[k] = v.to(device=device, dtype=model_dtype)
        else:
            inputs[k] = v.to(device=device)
    return inputs


def build_processor_inputs_from_messages(processor, messages, device, model_dtype):
    """
    统一构建 processor 输入，兼容不同版本 qwen_vl_utils/transformers 的差异。
    关键点：
    - process_vision_info 可能返回 []，需要转成 None，避免部分版本把空视频当成非法输入。
    - 若视觉解析失败，回退到直接从 messages 中提取 PIL 图像。
    """
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 强制稳定模式：
    # 在当前环境（transformers 4.57.x + qwen_vl_utils）下，
    # process_vision_info 返回结果在部分机型会触发 "height and width must be > 0"。
    # 这里直接提取 PIL 图像，不再依赖 process_vision_info。
    # 注意 images 需要按样本分组（与 text batch 对齐），例如 [[img1, img2]]。
    batched_image_inputs = []
    for msg in messages:
        msg_images = []
        for item in msg.get("content", []):
            if item.get("type") == "image" and item.get("image") is not None:
                msg_images.append(item["image"])
        if msg_images:
            batched_image_inputs.append(msg_images)

    if not batched_image_inputs:
        raise ValueError("No image inputs found in messages.")

    # 再做一层保护，防止 0 尺寸图进入 processor
    safe_batched_images = []
    for msg_images in batched_image_inputs:
        safe_images = []
        for img in msg_images:
            if hasattr(img, "size"):
                w, h = img.size
                if w <= 0 or h <= 0:
                    continue
            safe_images.append(img)
        if safe_images:
            safe_batched_images.append(safe_images)
    batched_image_inputs = safe_batched_images
    if not batched_image_inputs:
        raise ValueError("All image inputs are invalid (width/height <= 0).")

    # 不传 videos，避免空视频分支在低兼容环境触发异常
    try:
        inputs = processor(
            text=[text],
            images=batched_image_inputs,
            padding=True,
            return_tensors="pt",
        )
    except Exception as e:
        image_sizes = [
            [getattr(img, "size", None) for img in msg_images]
            for msg_images in batched_image_inputs
        ]
        raise ValueError(
            f"Processor input build failed: {e}; image_sizes={image_sizes}"
        ) from e
    inputs = normalize_qwen_batch_inputs(inputs, device, model_dtype)
    return inputs


# ==================== Qwen2-VL 模型加载 ====================

def load_qwen_model(model_name: str = QWEN_MODEL_NAME):
    """
    加载 Qwen2-VL 模型（AWQ 安全加载）
    """
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoConfig

    print(f"Loading Qwen2-VL model: {model_name}...")

    # 保护 lm_head 不被错误量化（避免生成头被替换成 qweight/qzeros）
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(cfg, "quantization_config") and isinstance(cfg.quantization_config, dict):
        mnc = cfg.quantization_config.get("modules_to_not_convert", [])
        if "lm_head" not in mnc:
            mnc.append("lm_head")
        cfg.quantization_config["modules_to_not_convert"] = mnc

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        config=cfg,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    # 修复 AWQ 量化参数 dtype，避免 Triton 算子出现类型冲突
    fixed_awq = 0
    for _, module in model.named_modules():
        if hasattr(module, "qweight") and torch.is_tensor(module.qweight):
            if module.qweight.dtype != torch.int32:
                module.qweight.data = module.qweight.data.to(dtype=torch.int32)
                fixed_awq += 1
        if hasattr(module, "qzeros") and torch.is_tensor(module.qzeros):
            if module.qzeros.dtype != torch.int32:
                module.qzeros.data = module.qzeros.data.to(dtype=torch.int32)
                fixed_awq += 1
        if hasattr(module, "scales") and torch.is_tensor(module.scales):
            if module.scales.dtype != torch.float16:
                module.scales.data = module.scales.data.to(dtype=torch.float16)
                fixed_awq += 1
        if hasattr(module, "g_idx") and torch.is_tensor(module.g_idx):
            if module.g_idx.dtype != torch.int32:
                module.g_idx.data = module.g_idx.data.to(dtype=torch.int32)
                fixed_awq += 1
    if fixed_awq > 0:
        print(f"✓ Fixed AWQ dtypes for Triton kernel: {fixed_awq} tensors corrected")

    def _build_size_override_from_preprocessor_cfg(path_like: str) -> Dict[str, int]:
        """
        读取 preprocessor_config.json 并构造兼容 Qwen2VLImageProcessor 的 size 覆盖参数。
        若读取失败，返回安全默认值。
        """
        default_size = {"shortest_edge": 28 * 32, "longest_edge": 28 * 32}  # 896
        try:
            model_path = Path(path_like)
            cfg_path = model_path / "preprocessor_config.json"
            if not cfg_path.exists():
                return default_size
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            size_cfg = cfg.get("size", {})
            if isinstance(size_cfg, dict):
                if "shortest_edge" in size_cfg and "longest_edge" in size_cfg:
                    return {
                        "shortest_edge": int(size_cfg["shortest_edge"]),
                        "longest_edge": int(size_cfg["longest_edge"]),
                    }
                h = int(size_cfg.get("height", size_cfg.get("shortest_edge", default_size["shortest_edge"])))
                w = int(size_cfg.get("width", size_cfg.get("longest_edge", default_size["longest_edge"])))
                shortest = min(h, w)
                longest = max(h, w)
                return {"shortest_edge": shortest, "longest_edge": longest}
            return default_size
        except Exception:
            return default_size

    try:
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
        )
    except ValueError as e:
        if "shortest_edge" in str(e) and "longest_edge" in str(e):
            size_override = {"shortest_edge": 28, "longest_edge": 4096}
            print(f"⚠ Processor size error, retrying with size override: {size_override}")
            processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                size=size_override,
                use_fast=True,
            )
        else:
            raise
    except Exception as e:
        print(f"⚠ Fast processor load failed, fallback to slow processor: {e}")
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )

    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    print(f"✓ Model loaded successfully on device: {device} | dtype: {model.dtype}")

    return model, processor, device


# ==================== Qwen2-VL 输入构造 ====================

def build_qwen_input(processor, rgb_image: Image.Image, gt_image: Image.Image, user_prompt: str, device, model_dtype):
    """
    构造符合 Qwen2-VL chat 模板的输入（双图像版本）。
    
    Qwen2-VL 使用 messages 格式，包含两张图像（RGB 和 GT）。
    注意：图像顺序很重要，第一张是 RGB（Img1），第二张是 GT（Img2），与 System Prompt 中的描述一致。
    
    Args:
        processor: Qwen2-VL processor
        rgb_image: RGB 图像（Img1: Glare/RGB）
        gt_image: GT 图像（Img2: Clear/GT）
        user_prompt: 用户提示词
        device: 设备
    
    Returns:
        inputs: processor(...) 的结果，已移动到指定设备
    """
    # Qwen2-VL 的 messages 格式
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": rgb_image},  # Image 1: RGB (Glare)
                {"type": "image", "image": gt_image},   # Image 2: GT (Clear)
                {"type": "text", "text": user_prompt},
            ],
        }
    ]
    
    return build_processor_inputs_from_messages(processor, messages, device, model_dtype)


def parse_qa_json(response_text: str) -> Optional[Dict]:
    """
    解析 LLaVA 返回的 JSON 响应（改进版，支持多种格式修复）
    
    Args:
        response_text: LLaVA 生成的文本
    
    Returns:
        解析后的 QA 字典，如果解析失败返回 None
    """
    # 移除可能的 Markdown 标记
    json_str = re.sub(r'```json\s*', '', response_text)
    json_str = re.sub(r'```\s*', '', json_str)
    json_str = json_str.strip()
    
    # 移除可能的注释（JSON 不支持注释，但 LLM 可能会添加）
    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)  # 单行注释
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)  # 多行注释
    # [新增] 移除 HTML 风格的注释（如 `<-- EXAMPLE ONLY! DO NOT USE! -->` 或 `<-- EXAMPLE ONLY! DO NOT USE!`）
    # 注意：这些注释可能在字符串值内部，需要小心处理
    json_str = re.sub(r'<\s*--.*?--\s*>', '', json_str, flags=re.DOTALL | re.IGNORECASE)  # 完整的 HTML 注释
    json_str = re.sub(r'<\s*--[^>]*$', '', json_str, flags=re.MULTILINE | re.IGNORECASE)  # 未闭合的 HTML 注释（行尾）
    # 特别处理：在字符串值内部的注释（在引号内）
    # 使用更精确的正则，匹配 `"value"  <-- comment` 这种模式
    json_str = re.sub(r'"\s*<\s*--.*?--\s*>', '"', json_str, flags=re.DOTALL | re.IGNORECASE)  # 引号后跟注释
    json_str = re.sub(r'"\s*<\s*--[^>]*$', '"', json_str, flags=re.MULTILINE | re.IGNORECASE)  # 引号后跟未闭合注释
    
    # 尝试找到 JSON 对象的开始和结束
    start_idx = json_str.find('{')
    end_idx = json_str.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = json_str[start_idx:end_idx + 1]
    else:
        json_str = json_str.strip()
    
    # 尝试1：直接解析
    try:
        qa_data = json.loads(json_str)
        return qa_data
    except json.JSONDecodeError:
        pass
    
    # 尝试2：修复单引号
    try:
        json_str_fixed = re.sub(r"'([^']*)':", r'"\1":', json_str)  # 键
        json_str_fixed = re.sub(r":\s*'([^']*)'", r': "\1"', json_str_fixed)  # 值
        qa_data = json.loads(json_str_fixed)
        return qa_data
    except:
        pass
    
    # 尝试3：修复未闭合的括号（简单修复）
    try:
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str_fixed = json_str + '}' * (open_braces - close_braces)
            qa_data = json.loads(json_str_fixed)
            return qa_data
    except:
        pass
    
    # 尝试4：提取部分 JSON（如果完整解析失败）
    try:
        # [改进] 在提取之前，先清理匹配到的文本中的注释
        def clean_match_text(text):
            """清理匹配到的文本，去除注释"""
            if not text:
                return text
            # 移除 HTML 风格注释
            text = re.sub(r'<\s*--.*?--\s*>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<\s*--.*?$', '', text, flags=re.MULTILINE | re.IGNORECASE)
            return text.strip()
        
        # 尝试提取新版 A/B/C 任务字段（更宽松的正则，允许嵌套对象）
        # 使用非贪婪匹配，但需要处理嵌套的大括号
        positive_pattern = r'"qa_positive"\s*:\s*(\{(?:[^{}]|(?:\{[^{}]*\}))*\}|null)'
        physics_pattern = r'"qa_physics"\s*:\s*(\{(?:[^{}]|(?:\{[^{}]*\}))*\}|null)'
        negative_pattern = r'"qa_negative"\s*:\s*(\{(?:[^{}]|(?:\{[^{}]*\}))*\}|null)'
        
        positive_match = re.search(positive_pattern, json_str, re.DOTALL)
        physics_match = re.search(physics_pattern, json_str, re.DOTALL)
        negative_match = re.search(negative_pattern, json_str, re.DOTALL)
        
        if positive_match:
            # 手动构建 JSON
            qa_data = {}
            try:
                positive_str = clean_match_text(positive_match.group(1))
                if positive_str and positive_str.strip() != 'null':
                    positive_json = json.loads('{"qa_positive": ' + positive_str + '}')
                    qa_data['qa_positive'] = positive_json.get('qa_positive')
            except Exception as e:
                # 如果解析失败，尝试更简单的方法：直接提取 q 和 a 字段
                try:
                    # [改进] 提取 q 和 a 字段，处理可能包含注释的情况
                    # 使用非贪婪匹配，并处理引号内可能包含的转义字符
                    q_match = re.search(r'"q"\s*:\s*"((?:[^"\\]|\\.)*)"', positive_match.group(1), re.DOTALL)
                    a_match = re.search(r'"a"\s*:\s*"((?:[^"\\]|\\.)*)"', positive_match.group(1), re.DOTALL)
                    if q_match and a_match:
                        q_text = clean_match_text(q_match.group(1))
                        a_text = clean_match_text(a_match.group(1))
                        # 再次清理答案中的注释（可能在字符串中间，如 "A yellow banana."  <-- EXAMPLE ONLY!）
                        a_text = re.sub(r'<\s*--.*?--\s*>', '', a_text, flags=re.DOTALL | re.IGNORECASE).strip()
                        a_text = re.sub(r'<\s*--.*$', '', a_text, flags=re.MULTILINE | re.IGNORECASE).strip()
                        qa_data['qa_positive'] = {
                            'q': q_text,
                            'a': a_text
                        }
                except:
                    pass
            
            if physics_match:
                physics_str = clean_match_text(physics_match.group(1))
                if physics_str and physics_str.strip() != 'null':
                    try:
                        physics_json = json.loads('{"qa_physics": ' + physics_str + '}')
                        qa_data['qa_physics'] = physics_json.get('qa_physics')
                    except:
                        # 尝试简单提取
                        try:
                            q_match = re.search(r'"q"\s*:\s*"([^"]*)"', physics_match.group(1))
                            a_match = re.search(r'"a"\s*:\s*"([^"]*)"', physics_match.group(1))
                            if q_match and a_match:
                                qa_data['qa_physics'] = {
                                    'q': clean_match_text(q_match.group(1)),
                                    'a': clean_match_text(a_match.group(1))
                                }
                            else:
                                qa_data['qa_physics'] = None
                        except:
                            qa_data['qa_physics'] = None
                else:
                    qa_data['qa_physics'] = None
            
            if negative_match:
                negative_str = clean_match_text(negative_match.group(1))
                if negative_str and negative_str.strip() != 'null':
                    try:
                        negative_json = json.loads('{"qa_negative": ' + negative_str + '}')
                        qa_data['qa_negative'] = negative_json.get('qa_negative')
                    except:
                        # 尝试简单提取
                        try:
                            q_match = re.search(r'"q"\s*:\s*"([^"]*)"', negative_match.group(1))
                            a_match = re.search(r'"a"\s*:\s*"([^"]*)"', negative_match.group(1))
                            if q_match and a_match:
                                qa_data['qa_negative'] = {
                                    'q': clean_match_text(q_match.group(1)),
                                    'a': clean_match_text(a_match.group(1))
                                }
                            else:
                                qa_data['qa_negative'] = None
                        except:
                            qa_data['qa_negative'] = None
                else:
                    qa_data['qa_negative'] = None
            
            if qa_data:
                return qa_data
    except:
        pass
    
    return None


def generate_qa_pairs_batch(
    model,
    processor,
    device,
    rgb_images: List[Image.Image],
    gt_images: List[Image.Image],
    bbox_norms: List[list],
    max_retries: int = 1,
) -> List[Dict]:
    """
    Qwen2-VL 版本的批量生成（伪批处理）
    
    注意：由于 72B 模型显存限制，实际在内部使用 for 循环逐个处理（Batch Size = 1）
    真正的 Batch 可能会导致 padding 过长而 OOM，所以这里采用伪批处理方式
    
    Args:
        model: Qwen2-VL 模型
        processor: Qwen2-VL processor
        device: 设备
        rgb_images: RGB 图像列表
        gt_images: GT 图像列表
        bbox_norms: 归一化 bbox 列表 [xmin, ymin, xmax, ymax]
        max_retries: 最大重试次数（如果 JSON 解析失败）
    
    Returns:
        包含 3 类 QA 对的字典列表
    """
    import torch
    
    # [修改] Qwen2-VL 支持动态分辨率，不需要强制 resize 到 336
    # 但为了防止 4K 图爆显存，可以做一个较宽松的限制 (比如长边 1024 或 1280)
    # 如果显存 OOM，请调小 MAX_IMAGE_SIZE 全局变量
    max_s = MAX_IMAGE_SIZE
    
    results = []
    
    # [关键] Qwen2-VL 72B 模型在 48GB 显存上只能逐个处理
    # 即使是"批处理"函数，内部也是 for 循环，避免显存 OOM
    for rgb_img, gt_img, bbox_norm in zip(rgb_images, gt_images, bbox_norms):
        # 1. 预处理图片（稳定模式：统一成 896x896）
        # [修改] 不再画红框，直接使用原图
        rgb_small = safe_resize(rgb_img)
        gt_small = safe_resize(gt_img)
        
        # 3. 构建 prompt
        # [关键修改] 使用整数坐标（0-1000）给 Qwen 看（作为参考）
        bbox_str_prompt = format_bbox_int(bbox_norm)  # 整数坐标，给 Qwen 看（参考用）
        bbox_str_save = format_bbox_string(bbox_norm)  # 小数坐标，用于保存和覆盖问题
        # [优化] 新的 Prompt (v5.5 - GT-Based) 强调对比GT和RGB，答案基于GT
        user_prompt = build_system_prompt()
        
        # 4. 构建 Qwen 格式的 messages
        # 注意：图像顺序很重要，第一张是 RGB（Img1），第二张是 GT（Img2），与 System Prompt 描述一致
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_small},  # Image 1: RGB (Glare)
                    {"type": "image", "image": gt_small},   # Image 2: GT (Clear)
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        
        # 5. 准备输入（兼容模式）
        inputs = build_processor_inputs_from_messages(processor, messages, device, model.dtype)
        
        # 6. 生成（带重试机制）
        result = None
        for attempt in range(max_retries + 1):
            try:
                # 重试时使用更高的 temperature，增加随机性
                current_temp = TEMPERATURE if attempt == 0 else TEMPERATURE_RETRY
                
                # 检查输入序列长度
                input_length = inputs["input_ids"].shape[1]
                max_model_length = getattr(model.config, "max_position_embeddings", 32768)  # Qwen2-VL 支持更长序列
                
                if input_length > max_model_length:
                    print(f"  ⚠ Warning: Input sequence length ({input_length}) exceeds model max length ({max_model_length})")
                    print(f"     Skipping this sample to avoid indexing errors.")
                    result = {
                        "qa_positive": None,
                        "qa_physics": None,
                        "qa_negative": None,
                    }
                    break
                
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=current_temp,
                        top_p=TOP_P,
                        repetition_penalty=REPETITION_PENALTY,
                        do_sample=True,
                    )
                
                # 解码响应
                # Qwen2-VL 的 decode 需要去除 input_ids 部分
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0].strip()
                
                # 解析 JSON
                qa_data = parse_qa_json(response_text)
                
                if qa_data is not None:
                    # [复用原有的格式修复和清洗逻辑]
                    # 修复 Content
                    if isinstance(qa_data.get('qa_positive'), str):
                        answer_text = qa_data['qa_positive']
                        object_name = _extract_object_from_text(answer_text)
                        qa_data['qa_positive'] = {
                            "q": _format_object_question(POSITIVE_PROMPT_TEMPLATES, object_name, "object"),
                            "a": answer_text
                        }
                    elif isinstance(qa_data.get('qa_positive'), list):
                        answer_text = str(qa_data['qa_positive'][0])
                        object_name = _extract_object_from_text(answer_text)
                        qa_data['qa_positive'] = {
                            "q": _format_object_question(POSITIVE_PROMPT_TEMPLATES, object_name, "object"),
                            "a": answer_text
                        }
                    
                    # 修复 Physics
                    if isinstance(qa_data.get('qa_physics'), str):
                        answer_text = qa_data['qa_physics']
                        object_name = _extract_object_from_text(answer_text)
                        qa_data['qa_physics'] = {
                            "q": _format_object_question(PHYSICS_PROMPT_TEMPLATES, object_name, "reflected object"),
                            "a": answer_text
                        }
                    
                    # 修复 Negative
                    if isinstance(qa_data.get('qa_negative'), str):
                        answer_text = qa_data['qa_negative']
                        object_name = _extract_object_from_text(answer_text)
                        qa_data['qa_negative'] = {
                            "q": _format_object_question(NEGATIVE_PROMPT_TEMPLATES, object_name, "reflected object"),
                            "a": answer_text
                        }
                    
                    # [增强版] 强力清洗：去除 Prompt 泄露的指令词
                    def clean_text(text, is_question=False):
                        """清洗文本，去除 Prompt 泄露的指令词"""
                        if not isinstance(text, str):
                            return text
                        
                        patterns = [
                            r"based on Img\d.*",
                            r"based ONLY on Img\d.*",
                            r"\(GT\).*",
                            r"in the clear image.*",
                            r"based strictly on.*",
                            r"\bImage\s*1\b",
                            r"\bImage\s*2\b",
                            r"\bGT\b",
                            r"\bRGB\b",
                        ]
                        for p in patterns:
                            text = re.sub(p, "", text, flags=re.IGNORECASE)
                        
                        text = text.strip().rstrip(".,;:").strip()
                        
                        if is_question and text and not text.endswith("?"):
                            text += "?"
                        
                        return text
                    
                    # 清洗所有问题和答案
                    if qa_data.get('qa_positive') and isinstance(qa_data['qa_positive'], dict):
                        qa_data['qa_positive']['q'] = clean_text(qa_data['qa_positive'].get('q', ''), is_question=True)
                        qa_data['qa_positive']['a'] = clean_text(qa_data['qa_positive'].get('a', ''), is_question=False)
                    
                    if qa_data.get('qa_physics') and isinstance(qa_data['qa_physics'], dict):
                        qa_data['qa_physics']['q'] = clean_text(qa_data['qa_physics'].get('q', ''), is_question=True)
                        qa_data['qa_physics']['a'] = clean_text(qa_data['qa_physics'].get('a', ''), is_question=False)
                    
                    if qa_data.get('qa_negative') and isinstance(qa_data['qa_negative'], dict):
                        qa_data['qa_negative']['q'] = clean_text(qa_data['qa_negative'].get('q', ''), is_question=True)
                        qa_data['qa_negative']['a'] = clean_text(qa_data['qa_negative'].get('a', ''), is_question=False)
                    
                    # 补齐/强化问题（避免空问题或过于泛化）
                    if qa_data.get('qa_positive') and isinstance(qa_data['qa_positive'], dict):
                        _ensure_object_bound_question(qa_data['qa_positive'], POSITIVE_PROMPT_TEMPLATES, "object")
                    if qa_data.get('qa_physics') and isinstance(qa_data['qa_physics'], dict):
                        _ensure_object_bound_question(qa_data['qa_physics'], PHYSICS_PROMPT_TEMPLATES, "reflected object")
                    if qa_data.get('qa_negative') and isinstance(qa_data['qa_negative'], dict):
                        _ensure_object_bound_question(qa_data['qa_negative'], NEGATIVE_PROMPT_TEMPLATES, "reflected object")
                    
                    result = {
                        "qa_positive": qa_data.get("qa_positive"),
                        "qa_physics": qa_data.get("qa_physics"),
                        "qa_negative": qa_data.get("qa_negative"),
                    }
                    break  # 成功则跳出重试循环
                    
            except Exception as e:
                if attempt < max_retries:
                    print(f"  ⚠ Generation error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    continue
                else:
                    global _PRINTED_GEN_TRACEBACK
                    if not _PRINTED_GEN_TRACEBACK:
                        print("  🔍 Full traceback (first generation error):")
                        print(traceback.format_exc())
                        _PRINTED_GEN_TRACEBACK = True
                    print(f"  ⚠ Failed to generate QA after {max_retries + 1} attempts: {e}")
                    result = {
                        "qa_positive": None,
                        "qa_physics": None,
                        "qa_negative": None,
                    }
        
        # 如果重试都失败
        if result is None:
            result = {
                "qa_positive": None,
                "qa_physics": None,
                "qa_negative": None,
            }
        
        results.append(result)
    
    return results


def generate_qa_pairs_no_glare(
    model,
    processor,
    device,
    gt_image: Image.Image,
    max_retries: int = 0,
) -> Dict:
    """
    为无反光图像生成简化的 QA 对（仅 Task A Positive）
    
    Args:
        model: Qwen2-VL 模型
        processor: Qwen2-VL processor
        device: 设备
        gt_image: GT 图像（原始，未裁剪）
        max_retries: 最大重试次数（通常不需要重试，设为0）
    
    Returns:
        包含 qa_positive 的字典（qa_physics 和 qa_negative 为 null）
    """
    import torch
    
    # 全图 bbox
    bbox_norm = [0.0, 0.0, 1.0, 1.0]
    bbox_str = "[0.000, 0.000, 1.000, 1.000]"
    
    # [修改] 使用动态分辨率，限制最大边长
    gt_image_small = safe_resize(gt_image)
    
    # [优化] 简化 Prompt：要求一句话简要描述，避免过长描述
    user_prompt = f"""
Describe the main real object in this image concisely in one sentence.
Output strictly in JSON format:

{{
  "qa_positive": {{ "q": "Describe the main real object in the scene.", "a": "..." }},
  "qa_physics": null,
  "qa_negative": null
}}
"""
    
    # [修改] 构建 Qwen 格式的 messages（单图像版本）
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": gt_image_small},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]
    
    try:
        # 准备输入（兼容模式）
        inputs = build_processor_inputs_from_messages(processor, messages, device, model.dtype)
    except Exception as e:
        print(f"Warning: Failed to build input for no-glare image: {e}")
        return {
            "qa_positive": None,
            "qa_physics": None,
            "qa_negative": None,
        }
    
    # 检查输入序列长度
    input_length = inputs["input_ids"].shape[1]
    max_model_length = getattr(model.config, "max_position_embeddings", 32768)
    
    if input_length > max_model_length:
        print(f"  ⚠ Warning: Input sequence length ({input_length}) exceeds model max length ({max_model_length}) for no-glare image")
        return {
            "qa_positive": None,
            "qa_physics": None,
            "qa_negative": None,
        }
    
    # 生成描述
    for attempt in range(max_retries + 1):
        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    repetition_penalty=REPETITION_PENALTY,
                    do_sample=True,
                )
            
            # 解码响应（Qwen2-VL 格式）
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            # [优化] 尝试解析 JSON 响应（新的 Prompt 要求输出 JSON）
            qa_data = parse_qa_json(response_text)
            
            if qa_data is not None and qa_data.get('qa_positive') is not None:
                # 成功解析 JSON，使用解析后的内容
                if isinstance(qa_data['qa_positive'], dict):
                    result = {
                        "qa_positive": qa_data['qa_positive'],
                        "qa_physics": None,
                        "qa_negative": None,
                    }
                else:
                    # 如果解析失败，使用原始文本作为答案
                    def clean_text(text):
                        if not isinstance(text, str):
                            return text
                        patterns = [
                            r"based on Img\d.*",
                            r"based ONLY on Img\d.*",
                            r"\(GT\).*",
                            r"in the clear image.*",
                            r"based strictly on.*",
                            r"\bImage\s*1\b",
                            r"\bImage\s*2\b",
                            r"\bGT\b",
                            r"\bRGB\b",
                        ]
                        for p in patterns:
                            text = re.sub(p, "", text, flags=re.IGNORECASE)
                        text = text.strip().rstrip(".,;:").strip()
                        return text
                    
                    cleaned_answer = clean_text(response_text)
                    result = {
                        "qa_positive": {
                            "q": "Describe the main real object in the scene.",
                            "a": cleaned_answer,
                        },
                        "qa_physics": None,
                        "qa_negative": None,
                    }
                return result
            else:
                # JSON 解析失败，使用原始响应文本（兜底方案）
                def clean_text(text):
                    if not isinstance(text, str):
                        return text
                    patterns = [
                        r"based on Img\d.*",
                        r"based ONLY on Img\d.*",
                        r"\(GT\).*",
                        r"in the clear image.*",
                        r"based strictly on.*",
                        r"\bImage\s*1\b",
                        r"\bImage\s*2\b",
                        r"\bGT\b",
                        r"\bRGB\b",
                    ]
                    for p in patterns:
                        text = re.sub(p, "", text, flags=re.IGNORECASE)
                    text = text.strip().rstrip(".,;:").strip()
                    return text
                
                cleaned_answer = clean_text(response_text)
                
                # 构造结果
                result = {
                    "qa_positive": {
                        "q": "Describe the main real object in the scene.",
                        "a": cleaned_answer,
                    },
                    "qa_physics": None,
                    "qa_negative": None,
                }
                return result
            
        except Exception as e:
            if attempt < max_retries:
                print(f"  ⚠ Generation failed for no-glare image, retrying ({attempt + 1}/{max_retries})...")
                continue
            else:
                global _PRINTED_GEN_TRACEBACK
                if not _PRINTED_GEN_TRACEBACK:
                    print("  🔍 Full traceback (first no-glare generation error):")
                    print(traceback.format_exc())
                    _PRINTED_GEN_TRACEBACK = True
                print(f"  ⚠ Failed to generate QA for no-glare image after {max_retries + 1} attempts: {e}")
                return {
                    "qa_positive": None,
                    "qa_physics": None,
                    "qa_negative": None,
                }
    
    return {
        "qa_positive": None,
        "qa_physics": None,
        "qa_negative": None,
    }


def generate_qa_pairs(
    model,
    processor,
    device,
    rgb_image: Image.Image,
    gt_image: Image.Image,
    bbox_norm: list,
    max_retries: int = 1,
) -> Dict:
    """
    为给定的 bbox 生成 3 类 QA 对（Qwen2-VL 版本，带重试机制）
    
    Args:
        model: Qwen2-VL 模型
        processor: Qwen2-VL processor
        device: 设备
        rgb_image: RGB 图像（原始，未裁剪）
        gt_image: GT 图像（原始，未裁剪）
        bbox_norm: 归一化 bbox [xmin, ymin, xmax, ymax]（标准格式）
        max_retries: 最大重试次数（如果 JSON 解析失败）
    
    Returns:
        包含 3 类 QA 对的字典
    """
    import torch
    
    # [修改] Qwen2-VL 支持动态分辨率，不需要强制 resize 到 336
    # 但为了防止 4K 图爆显存，可以做一个较宽松的限制
    # [修改] 不再画红框，直接使用原图
    # 安全缩放：保持比例、无黑边填充，并对齐到 28 倍数
    rgb_image_small = safe_resize(rgb_image)
    gt_image_small = safe_resize(gt_image)
    
    # [关键修改] 使用整数坐标（0-1000）给 Qwen 看（作为参考）
    bbox_str_prompt = format_bbox_int(bbox_norm)  # 整数坐标，给 Qwen 看（参考用）
    bbox_str_save = format_bbox_string(bbox_norm)  # 小数坐标，用于保存和覆盖问题
    # [优化] 新的 Prompt (v5.5 - GT-Based) 强调对比GT和RGB，答案基于GT
    user_prompt = build_system_prompt()
    
    # 重试机制：如果 JSON 解析失败，最多重试 max_retries 次
    # [改进] 重试时提高 temperature，增加随机性，避免重复相同的错误
    for attempt in range(max_retries + 1):
        # [修改] 使用 Qwen2-VL 的输入构造方式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_image_small},  # Image 1: RGB (Glare)
                    {"type": "image", "image": gt_image_small},   # Image 2: GT (Clear)
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        
        try:
            # 准备输入（兼容模式）
            inputs = build_processor_inputs_from_messages(processor, messages, device, model.dtype)
        except Exception as e:
            print(f"  ⚠ Warning: Failed to build input: {e}")
            if attempt < max_retries:
                continue
            else:
                return {
                    "qa_positive": None,
                    "qa_physics": None,
                    "qa_negative": None,
                }
        
        # [关键修复] 检查输入序列长度，如果超过模型限制则跳过
        input_length = inputs["input_ids"].shape[1]
        max_model_length = getattr(model.config, "max_position_embeddings", 32768)  # Qwen2-VL 支持更长序列
        
        if input_length > max_model_length:
            print(f"  ⚠ Warning: Input sequence length ({input_length}) exceeds model max length ({max_model_length})")
            print(f"     Skipping this sample to avoid indexing errors.")
            return {
                "qa_positive": None,
                "qa_physics": None,
                "qa_negative": None,
            }
        
        # [改进] 重试时使用更高的 temperature，强制模型换一种说法
        current_temperature = TEMPERATURE if attempt == 0 else TEMPERATURE_RETRY
        
        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=current_temperature,
                    top_p=TOP_P,
                    repetition_penalty=REPETITION_PENALTY,
                    do_sample=True,
                )
            
            # [修改] Qwen2-VL 的解码方式
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()
            
        except Exception as e:
            if attempt < max_retries:
                print(f"  ⚠ Generation error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                continue
            else:
                global _PRINTED_GEN_TRACEBACK
                if not _PRINTED_GEN_TRACEBACK:
                    print("  🔍 Full traceback (first generation error):")
                    print(traceback.format_exc())
                    _PRINTED_GEN_TRACEBACK = True
                print(f"  ⚠ Failed to generate after {max_retries + 1} attempts: {e}")
                return {
                    "qa_positive": None,
                    "qa_physics": None,
                    "qa_negative": None,
                }
        
        # 解析 JSON 响应
        qa_data = parse_qa_json(response_text)
        
        if qa_data is not None:
            # [复用原有的格式修复和清洗逻辑]
            # 修复 Content
            if isinstance(qa_data.get('qa_positive'), str):
                answer_text = qa_data['qa_positive']
                object_name = _extract_object_from_text(answer_text)
                qa_data['qa_positive'] = {
                    "q": _format_object_question(POSITIVE_PROMPT_TEMPLATES, object_name, "object"),
                    "a": answer_text
                }
            elif isinstance(qa_data.get('qa_positive'), list):
                answer_text = str(qa_data['qa_positive'][0])
                object_name = _extract_object_from_text(answer_text)
                qa_data['qa_positive'] = {
                    "q": _format_object_question(POSITIVE_PROMPT_TEMPLATES, object_name, "object"),
                    "a": answer_text
                }
            
            # 修复 Physics
            if isinstance(qa_data.get('qa_physics'), str):
                answer_text = qa_data['qa_physics']
                object_name = _extract_object_from_text(answer_text)
                qa_data['qa_physics'] = {
                    "q": _format_object_question(PHYSICS_PROMPT_TEMPLATES, object_name, "reflected object"),
                    "a": answer_text
                }
            
            # 修复 Negative
            if isinstance(qa_data.get('qa_negative'), str):
                answer_text = qa_data['qa_negative']
                object_name = _extract_object_from_text(answer_text)
                qa_data['qa_negative'] = {
                    "q": _format_object_question(NEGATIVE_PROMPT_TEMPLATES, object_name, "reflected object"),
                    "a": answer_text
                }
            
            # [增强版] 强力清洗：去除 Prompt 泄露的指令词
            def clean_text(text, is_question=False):
                """清洗文本，去除 Prompt 泄露的指令词"""
                if not isinstance(text, str):
                    return text
                
                patterns = [
                    r"based on Img\d.*",
                    r"based ONLY on Img\d.*",
                    r"\(GT\).*",
                    r"in the clear image.*",
                    r"based strictly on.*",
                    r"\bImage\s*1\b",
                    r"\bImage\s*2\b",
                    r"\bGT\b",
                    r"\bRGB\b",
                ]
                for p in patterns:
                    text = re.sub(p, "", text, flags=re.IGNORECASE)
                
                text = text.strip().rstrip(".,;:").strip()
                
                if is_question and text and not text.endswith("?"):
                    text += "?"
                
                return text
            
            # 清洗所有问题和答案
            if qa_data.get('qa_positive') and isinstance(qa_data['qa_positive'], dict):
                qa_data['qa_positive']['q'] = clean_text(qa_data['qa_positive'].get('q', ''), is_question=True)
                qa_data['qa_positive']['a'] = clean_text(qa_data['qa_positive'].get('a', ''), is_question=False)
            
            if qa_data.get('qa_physics') and isinstance(qa_data['qa_physics'], dict):
                qa_data['qa_physics']['q'] = clean_text(qa_data['qa_physics'].get('q', ''), is_question=True)
                qa_data['qa_physics']['a'] = clean_text(qa_data['qa_physics'].get('a', ''), is_question=False)
            
            if qa_data.get('qa_negative') and isinstance(qa_data['qa_negative'], dict):
                qa_data['qa_negative']['q'] = clean_text(qa_data['qa_negative'].get('q', ''), is_question=True)
                qa_data['qa_negative']['a'] = clean_text(qa_data['qa_negative'].get('a', ''), is_question=False)
            
            # 补齐/强化问题（避免空问题或过于泛化）
            if qa_data.get('qa_positive') and isinstance(qa_data['qa_positive'], dict):
                _ensure_object_bound_question(qa_data['qa_positive'], POSITIVE_PROMPT_TEMPLATES, "object")
            if qa_data.get('qa_physics') and isinstance(qa_data['qa_physics'], dict):
                _ensure_object_bound_question(qa_data['qa_physics'], PHYSICS_PROMPT_TEMPLATES, "reflected object")
            if qa_data.get('qa_negative') and isinstance(qa_data['qa_negative'], dict):
                _ensure_object_bound_question(qa_data['qa_negative'], NEGATIVE_PROMPT_TEMPLATES, "reflected object")
            
            # 解析成功，提取 QA 对
            result = {
                "qa_positive": qa_data.get("qa_positive"),
                "qa_physics": qa_data.get("qa_physics"),
                "qa_negative": qa_data.get("qa_negative"),
            }
            return result
        
        # 解析失败，如果还有重试机会，继续（使用更高的 temperature）
        if attempt < max_retries:
            print(f"  ⚠ JSON parsing failed, retrying with higher temperature ({attempt + 1}/{max_retries})...")
            print(f"     Response preview: {response_text[:200]}...")
            continue
    
    # 所有重试都失败，返回空结构
    print(f"  ⚠ Failed to parse JSON after {max_retries + 1} attempts")
    print(f"     Final response preview: {response_text[:200]}...")
    return {
        "qa_positive": None,
        "qa_physics": None,
        "qa_negative": None,
    }


# ==================== 主处理流程 ====================

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
        
        if max_total_images is not None and total_count >= max_total_images:
            break


def generate_stage3_qa_pairs(
    rgb_root: Optional[str] = None,
    gt_root: Optional[str] = None,
    scene_id_min: Optional[int] = None,
    scene_id_max: Optional[int] = None,
    max_images_per_scene: Optional[int] = None,
    max_total_images: Optional[int] = None,
    model_name: Optional[str] = None,
    output_json: Optional[str] = None,
    batch_size: Optional[int] = None,
    debug_bbox: Optional[bool] = None,
    debug_bbox_dir: Optional[str] = None,
):
    """
    生成 Stage 3 QA 对的主函数
    
    Args:
        rgb_root: RGB 图像根目录
        gt_root: GT 图像根目录
        scene_id_min: 最小场景ID
        scene_id_max: 最大场景ID
        max_images_per_scene: 每个场景最多处理的图像数量
        max_total_images: 总共最多处理的图像数量
        model_name: 模型路径
        output_json: 输出 JSON 文件路径
        batch_size: 批处理大小（如果为 None，使用全局 BATCH_SIZE）
    """
    # 使用传入的 batch_size 或全局默认值
    current_batch_size = batch_size if batch_size is not None else BATCH_SIZE
    enable_debug_bbox = DEBUG_BBOX if debug_bbox is None else debug_bbox
    
    model_path = model_name or QWEN_MODEL_NAME
    
    if output_json is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{OUTPUT_JSON_PREFIX}_{timestamp}.json"
    else:
        output_path = output_json
    debug_root = None
    if enable_debug_bbox:
        base_dir = Path(output_path).resolve().parent
        debug_root = Path(debug_bbox_dir) if debug_bbox_dir else (base_dir / DEBUG_BBOX_DIR)
    
    rgb_dir = Path(rgb_root) if rgb_root else RGB_ROOT
    gt_dir = Path(gt_root) if gt_root else GT_ROOT
    
    print("Using differential method for glare detection (GT vs RGB)")
    print("Generating A/B/C physics-aware Q&A pairs for each detected glare sample")
    
    # [修改] 使用 Qwen2-VL 模型加载函数
    model, processor, device = load_qwen_model(model_path)
    model.eval()
    
    # [修改] Qwen2-VL 不需要 torch.compile，且对动态分辨率支持一般
    # 如果 USE_TORCH_COMPILE 为 True，跳过（已经设置为 False）
    if USE_TORCH_COMPILE:
        print("⚠ Warning: torch.compile is not recommended for Qwen2-VL (dynamic resolution support issues)")
        print("  → Continuing without torch.compile (Qwen2-VL runs fast without it)")
    
    # [修改] 打印生成参数
    print(f"Generation parameters: max_new_tokens={MAX_NEW_TOKENS}, temperature={TEMPERATURE}, top_p={TOP_P}")
    print(f"Image max size: {MAX_IMAGE_SIZE} (Qwen2-VL supports dynamic resolution)")
    print(f"Batch size: {current_batch_size} (72B model requires Batch Size = 1 on 48GB GPU)")
    
    results = []
    
    rgb_items = list(iter_rgb_images(
        rgb_root=rgb_dir,
        scene_id_min=scene_id_min,
        scene_id_max=scene_id_max,
        max_images_per_scene=max_images_per_scene,
        max_total_images=max_total_images,
    ))
    
    print(f"\nFound {len(rgb_items)} RGB images, generating Stage 3 Q&A pairs...")
    if scene_id_min is not None or scene_id_max is not None:
        print(f"  Scene range: {scene_id_min or 'unlimited'} - {scene_id_max or 'unlimited'}")
    if max_images_per_scene is not None:
        print(f"  Max per scene: {max_images_per_scene}")
    if max_total_images is not None:
        print(f"  Total limit: {max_total_images}")
    print(f"  Batch size: {current_batch_size} (批处理加速)")
    
    # [可选] 去重策略：记录每个 scene_id 是否已经处理过全图描述（无反光图像）
    scene_id_no_glare_processed = set()
    
    # [批处理改进] 初始化批次收集列表
    batch_rgb_images = []
    batch_gt_images = []
    batch_bbox_norms = []
    batch_scene_ids = []
    batch_rgb_filenames = []
    batch_rgb_paths = []
    batch_gt_image_paths = []
    
    for idx, (scene_id, rgb_path, rgb_filename) in enumerate(
        tqdm(rgb_items, desc="Generating Stage 3 Q&A Pairs")
    ):
        try:
            rgb_image = Image.open(rgb_path).convert("RGB")
        except Exception as e:
            print(f"\nWarning: Failed to load RGB image, skipping: {rgb_path}, error: {e}")
            continue
        
        gt_image_path = find_gt_image_path(scene_id, rgb_filename, rgb_dir, gt_dir)
        if gt_image_path is None:
            print(f"\nWarning: GT image not found (scene {scene_id}, {rgb_filename}), skipping.")
            continue
        
        try:
            gt_image = Image.open(gt_image_path).convert("RGB")
        except Exception as e:
            print(f"\nWarning: Failed to load GT image, skipping: {gt_image_path}, error: {e}")
            continue
        
        # 检测眩光 bbox（使用与 Stage 2 相同的算法）
        bbox_norm = None
        try:
            bbox_norm = detect_glare_bbox_diff(
                rgb_path,
                gt_image_path,
                threshold=GLARE_THRESHOLD,
                min_area=MIN_GLARE_AREA,
                max_bbox_ratio=MAX_BBOX_RATIO,
            )
            # 兜底：如果没有检测到反光，放宽阈值再试一次
            if bbox_norm is None:
                bbox_norm = detect_glare_bbox_diff(
                    rgb_path,
                    gt_image_path,
                    threshold=RELAX_GLARE_THRESHOLD,
                    min_area=RELAX_MIN_GLARE_AREA,
                    max_bbox_ratio=RELAX_MAX_BBOX_RATIO,
                )
            # 兜底：若仍失败，用高分位热点提取非全图 bbox
            if bbox_norm is None:
                bbox_norm = detect_glare_bbox_percentile(
                    rgb_path,
                    gt_image_path,
                    percentiles=PERCENTILE_LEVELS,
                    min_area=RELAX_MIN_GLARE_AREA,
                    max_bbox_ratio=PERCENTILE_MAX_BBOX_RATIO,
                    force_bbox_ratio=FORCE_BBOX_RATIO,
                )
        except Exception:
            bbox_norm = None

        # 保存 bbox 可视化（调试用）
        if enable_debug_bbox and debug_root is not None:
            dbg_name = rgb_filename.replace("_rgb.png", "_bbox.png")
            dbg_path = debug_root / scene_id / dbg_name
            save_debug_bbox_image(rgb_image, bbox_norm, dbg_path)
        
        # [处理逻辑] 如果没有检测到反光，生成简化版 QA（仅 Type 1 Content，全图 bbox）
        if bbox_norm is None:
            # 去重可配：默认关闭，确保每张图都有至少 1 条 QA 输出
            if DEDUP_NO_GLARE_BY_SCENE and scene_id in scene_id_no_glare_processed:
                # 跳过，不生成 QA（已经为该场景生成过一个全图描述）
                continue
            
            # 对于无反光图像，使用简化版生成函数
            try:
                qa_pairs = generate_qa_pairs_no_glare(
                    model, processor, device, gt_image, max_retries=0
                )
                
                # 构造 Stage2 风格样本（无反光图仅 content，若模型返回 detail/spatial 也会自动收录）
                no_glare_bbox = [0.0, 0.0, 1.0, 1.0]
                records = build_stage2_style_records(
                    rgb_path=rgb_path,
                    gt_image_path=gt_image_path,
                    scene_id=scene_id,
                    bbox_norm=no_glare_bbox,
                    qa_pairs=qa_pairs,
                )
                results.extend(records)
                
                if DEDUP_NO_GLARE_BY_SCENE:
                    scene_id_no_glare_processed.add(scene_id)
            except Exception as e:
                print(f"\nWarning: Failed to generate no-glare QA for {scene_id}/{rgb_filename}: {e}")
                # 如果生成失败，跳过该图像（不标记为已处理，允许重试其他图片）
                continue
            
            # 无反光图像不进入批处理流程，直接处理完就继续下一个
            continue
        
        # [批处理改进] 收集到批次中（仅针对有反光的图像）
        batch_rgb_images.append(rgb_image)
        batch_gt_images.append(gt_image)
        batch_bbox_norms.append(bbox_norm)
        batch_scene_ids.append(scene_id)
        batch_rgb_filenames.append(rgb_filename)
        batch_rgb_paths.append(rgb_path)
        batch_gt_image_paths.append(gt_image_path)
        
        # 当批次满了，或者到达最后一个样本时，批量生成
        if len(batch_rgb_images) >= current_batch_size or idx == len(rgb_items) - 1:
            if len(batch_rgb_images) > 0:
                # 批量生成 QA 对
                try:
                    batch_qa_pairs = generate_qa_pairs_batch(
                        model, processor, device,
                        batch_rgb_images, batch_gt_images, batch_bbox_norms,
                        max_retries=0  # 批处理模式下不重试，避免复杂度
                    )
                except Exception as e:
                    print(f"\nWarning: Batch generation failed, falling back to individual processing: {e}")
                    # 回退到逐个处理
                    batch_qa_pairs = []
                    for rgb_img, gt_img, bbox in zip(batch_rgb_images, batch_gt_images, batch_bbox_norms):
                        try:
                            qa_pairs = generate_qa_pairs(
                                model, processor, device, rgb_img, gt_img, bbox, max_retries=1
                            )
                            batch_qa_pairs.append(qa_pairs)
                        except Exception as e2:
                            print(f"  ⚠ Individual generation also failed: {e2}")
                            batch_qa_pairs.append({
                                "qa_positive": None,
                                "qa_physics": None,
                                "qa_negative": None,
                            })
                
                # 组织数据并添加到结果中（Stage2 风格扁平结构）
                for i, qa_pairs in enumerate(batch_qa_pairs):
                    scene_id_item = batch_scene_ids[i]
                    rgb_filename_item = batch_rgb_filenames[i]
                    rgb_path_item = batch_rgb_paths[i]
                    gt_image_path_item = batch_gt_image_paths[i]
                    bbox_norm_item = batch_bbox_norms[i]
                    records = build_stage2_style_records(
                        rgb_path=rgb_path_item,
                        gt_image_path=gt_image_path_item,
                        scene_id=scene_id_item,
                        bbox_norm=bbox_norm_item,
                        qa_pairs=qa_pairs,
                    )
                    results.extend(records)
                
                # 清空批次
                batch_rgb_images.clear()
                batch_gt_images.clear()
                batch_bbox_norms.clear()
                batch_scene_ids.clear()
                batch_rgb_filenames.clear()
                batch_rgb_paths.clear()
                batch_gt_image_paths.clear()
    
    # 保存 JSON
    output_file = Path(output_path)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n================ Generation Complete ================")
    print(f"Successfully generated {len(results)} Stage 3 QA records")
    print(f"Output file: {output_file.resolve()}")
    
    # 统计信息
    total_processed = len(rgb_items)
    with_glare_count = sum(1 for r in results if r.get("bbox_norm") != [0.0, 0.0, 1.0, 1.0])
    no_glare_count = len(results) - with_glare_count
    with_content = sum(1 for r in results if r.get("subtype") == "positive")
    with_detail = sum(1 for r in results if r.get("subtype") == "physics")
    with_spatial = sum(1 for r in results if r.get("subtype") == "negative")
    
    print(f"\nStatistics:")
    print(f"  - Total images processed: {total_processed}")
    print(f"  - QA records with glare bbox: {with_glare_count}")
    print(f"  - QA records without glare bbox (full-image): {no_glare_count}")
    print(f"  - Total generated records: {len(results)}")
    print(f"  - Positive records: {with_content}")
    print(f"  - Physics records: {with_detail}")
    print(f"  - Negative records: {with_spatial}")
    if len(scene_id_no_glare_processed) > 0:
        print(f"  - Scenes with no-glare images (deduplicated): {len(scene_id_no_glare_processed)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Stage 3 Q&A pairs (using non-cropped images)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help=f"Qwen2-VL model path (default: {QWEN_MODEL_NAME})",
    )
    parser.add_argument(
        "--rgb_root",
        type=str,
        default=None,
        help=f"RGB image root directory (default: {RGB_ROOT})",
    )
    parser.add_argument(
        "--gt_root",
        type=str,
        default=None,
        help=f"GT image root directory (default: {GT_ROOT})",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help=f"Output JSON file path (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--scene_id_min",
        type=int,
        default=None,
        help="Minimum scene ID (inclusive), e.g., 0 means starting from scene '00'",
    )
    parser.add_argument(
        "--scene_id_max",
        type=int,
        default=None,
        help="Maximum scene ID (inclusive), e.g., 58 means ending at scene '58'",
    )
    parser.add_argument(
        "--max_images_per_scene",
        type=int,
        default=None,
        help="Maximum number of images to process per scene (None means unlimited)",
    )
    parser.add_argument(
        "--max_total_images",
        type=int,
        default=None,
        help="Maximum total number of images to process (None means unlimited)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for processing (default: {BATCH_SIZE}, increase for faster processing if GPU memory allows, decrease if OOM occurs)",
    )
    parser.add_argument(
        "--debug_bbox",
        action="store_true",
        help="Save bbox overlay images for debugging",
    )
    parser.add_argument(
        "--debug_bbox_dir",
        type=str,
        default=None,
        help=f"Output directory for debug bbox images (default: {DEBUG_BBOX_DIR} under output_json dir)",
    )
    
    args = parser.parse_args()
    
    import torch
    
    generate_stage3_qa_pairs(
        rgb_root=args.rgb_root,
        gt_root=args.gt_root,
        scene_id_min=args.scene_id_min,
        scene_id_max=args.scene_id_max,
        max_images_per_scene=args.max_images_per_scene,
        max_total_images=args.max_total_images,
        model_name=args.model_name,
        output_json=args.output_json,
        batch_size=args.batch_size,
        debug_bbox=args.debug_bbox,
        debug_bbox_dir=args.debug_bbox_dir,
    )


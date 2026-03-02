"""
Stage 3 QA 对生成脚本
使用本地 LLaVA v1.6 Vicuna-7B HF 模型，对 RGB + GT 数据集生成用于 PolarVLM Stage 3 训练的问答对。

核心思路：
- 使用差分方法（GT vs RGB）检测眩光区域，获取 bbox_norm
- 对每个检测到的反光区域，生成 3 类 QA 对：
  1. Type 1: Content (区域描述)
  2. Type 2: Detail (视觉属性恢复 - 颜色和材质)
  3. Type 3: Spatial (空间推理 - 视觉指代)
- 使用非裁剪图像（原始 RGB 和 GT 图像）
- 所有提示词和生成内容使用英文

工作流程：
1. 加载 RGB 和 GT 图像 → 2. 使用差分方法检测眩光 bbox → 3. 调用 LLaVA 生成 3 类 QA 对 → 4. 保存为训练数据

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
import random
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    BitsAndBytesConfig,
)

# ==================== 配置参数 ====================

# 模型名称（Hugging Face Hub 路径）
LLAVA_MODEL_NAME = "llava-hf/llava-v1.6-vicuna-7b-hf"

# 数据根目录
DATA_ROOT = Path("/openbayes/input/input0")
RGB_ROOT = DATA_ROOT / "rgb"      # 结构：rgb/{scene_id}/{filename}_rgb.png
GT_ROOT = DATA_ROOT / "GT"        # 结构：GT/{scene_id}/{filename}_rgb.png
POLAR_ROOT = DATA_ROOT / "polar"  # 结构：polar/{scene_id}/{filename}_{angle}.png

# 输出 JSON 文件（默认会加上时间戳）
OUTPUT_JSON_PREFIX = "stage3_qa_pairs"

# 反光检测参数（差分方法）
GLARE_THRESHOLD = 100          # 眩光检测阈值
MIN_GLARE_AREA = 500           # 最小眩光区域面积（像素）
MAX_BBOX_RATIO = 0.4           # 最大边界框面积比例

# 生成参数
MAX_NEW_TOKENS = 2400           # [修改] 降低到 800，生成 JSON 不需要那么长，节省显存给图片
TEMPERATURE = 0.3              # [修改] 降低温度以提高生成的一致性和准确性
TOP_P = 0.9
TEMPERATURE_RETRY = 0.5        # 重试时的温度（提高随机性，避免重复错误）

# 加速参数
BATCH_SIZE = 4                 # [改进] 批处理大小（根据 GPU 显存调整，4bit 量化模型可以设置 4-8）
USE_TORCH_COMPILE = True       # 是否使用 torch.compile 加速

# Type 1 Content Prompt 模板（英文）
CONTENT_PROMPT_TEMPLATES = [
    "Please focus on the region {bbox}. Please briefly describe the visual content within this region.",
    "Do you see the area at {bbox}? Please tell me what is here.",
    "What does the area selected by {bbox} show?",
    "Describe the image details in the {bbox} region.",
    "What can you see in the region {bbox}?",
    "Please describe what is displayed in the {bbox} area.",
]

# Type 2 Detail Prompt 模板（英文）
DETAIL_PROMPT_TEMPLATES = [
    "What are the color and material of the object in the region {bbox}?",
    "Please describe the surface material and color of the object in {bbox}.",
    "What color and material does the object in {bbox} appear to be?",
    "Describe the color and texture of the object located at {bbox}.",
]

# Type 3 Spatial Prompt 模板（英文）
SPATIAL_PROMPT_TEMPLATES = [
    "What object is located to the [direction] of the [anchor] in the region {bbox}?",
    "What is the object at the [direction] of the [anchor] in {bbox}?",
    "What can be found [direction] of the [anchor] in the {bbox} area?",
]

# System Prompt（英文，v3.2 - Anti-Copying One-Shot）
def build_system_prompt(bbox_str: str) -> str:
    """
    构建系统提示词（英文，v3.2 - Anti-Copying One-Shot）
    
    关键改进：
    1. Anti-Copying One-Shot：使用完全无关的场景（水果）作为示例，避免内容抄袭
    2. 严厉禁止抄袭：明确要求只抄格式不抄内容，示例内容标记为 "EXAMPLE ONLY! DO NOT USE!"
    3. 强化 Anti-Glare：明确禁止描述反光
    4. 强调基于实际图像：要求基于 YOUR image 生成，不是示例场景
    
    Args:
        bbox_str: 归一化坐标字符串，格式 "[0.1000, 0.2000, 0.3000, 0.4000]"
    
    Returns:
        系统提示词字符串（v3.2 - Anti-Copying One-Shot）
    """
    return f"""
You are a data generator.

**Input**: Two images (Img1: Glare/RGB, Img2: Clear/GT).

**Goal**: Describe the object in region {bbox_str} based ONLY on Img2 (GT).

**CRITICAL RULES:**

1. **NO COPYING**: The example below is about FRUIT. Your image is NOT about fruit. DO NOT COPY the content of the example. Use the ACTUAL object in your image.

2. **IGNORE GLARE**: Never mention glare/reflection.

3. **FORMAT**: Follow the JSON structure exactly.

**Generate 3 Types (Strict JSON):**

**1. Content (Identity ONLY):**

* Q: "What object is in this region?" / "Identify the object." / "Describe the object in this region." (Do NOT ask about color/material/texture here. Only ask "what is it".)

* A: The object name/category. (e.g., "A wooden chair." / "A glass window." / "A metal door handle.")

**2. Detail (Attributes - Color & Material):**

* Q: "What are the color and material of the [Object]?" (Use the object name from Type 1, e.g., "chair", "window", "door handle".)

* A: Color + Material + Texture. (e.g., "It is dark brown wood with a smooth finish." / "It is transparent glass with a clear surface.")

**3. Spatial (Reasoning - Visual Reference):**

* Find an **Anchor Object** visible in **BOTH** Image 1 (RGB) and Image 2 (GT).

* Q: "What is located [direction] of the [Anchor]?" (e.g., "What is located to the left of the lamp?")

* A: The target object name only. (e.g., "A small table." / "A red book.")

* If no clear anchor exists, return null.

**JSON FORMAT EXAMPLE (Do NOT copy the content, only the format!):**

{{
  "qa_content": {{
    "q": "What object is in this region?",
    "a": "A yellow banana."  <-- EXAMPLE ONLY! DO NOT USE!
  }},
  "qa_detail": {{
    "q": "What are the color and material of the banana?",
    "a": "It is bright yellow fruit skin with a smooth texture." <-- EXAMPLE ONLY!
  }},
  "qa_spatial": {{
    "q": "What is located to the right of the banana?",
    "a": "A red apple." <-- EXAMPLE ONLY!
  }}
}}

**NOW, generate the JSON for YOUR image (Scene: Indoor/Street/etc.) in region {bbox_str}.**

"""


# ==================== 反光检测相关函数 ====================

def detect_glare_bbox_diff(
    input_image_path: Path,
    gt_image_path: Path,
    threshold: int = GLARE_THRESHOLD,
    min_area: int = MIN_GLARE_AREA,
    max_bbox_ratio: float = MAX_BBOX_RATIO,
):
    """
    使用计算机视觉方法检测眩光区域并返回边界框（差分方法）。
    与 Stage 2 使用相同的算法，添加形态学操作提高鲁棒性。
    
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
        
        # [改进] 添加形态学操作，去除噪点和填补空洞，提高鲁棒性
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)  # 去除噪点
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)  # 填补空洞
        
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


def format_bbox_string(bbox_norm: list) -> str:
    """
    将归一化 bbox 格式化为字符串（标准格式）
    
    Args:
        bbox_norm: [xmin, ymin, xmax, ymax]（标准格式）
    
    Returns:
        格式化的字符串，例如 "[0.1000, 0.2000, 0.3000, 0.4000]"
    """
    xmin, ymin, xmax, ymax = bbox_norm
    # 使用标准格式 [xmin, ymin, xmax, ymax]，纯数字格式，与 LLaVA 兼容
    return f"[{xmin:.4f}, {ymin:.4f}, {xmax:.4f}, {ymax:.4f}]"


# ==================== LLaVA 模型加载 ====================

def load_llava_model(model_name: str = LLAVA_MODEL_NAME):
    """加载 LLaVA-Next 模型（4bit 量化）"""
    import torch
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading LLaVA model: {model_name} (4bit quantization)...")
    processor = LlavaNextProcessor.from_pretrained(model_name)
    
    # [关键修正] 批量生成必须使用左侧填充，否则短序列会生成乱码
    # LLaMA 等 Decoder-only 模型在批处理生成时，padding_side 必须为 "left"
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        # 确保有 pad_token（LLaMA 默认没有 pad_token）
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        print(f"✓ Tokenizer padding_side set to 'left' for batch generation")
        print(f"✓ Pad token: {processor.tokenizer.pad_token}")
    
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
    )

    device = list(model.parameters())[0].device
    print(f"Model loaded on device: {device}")

    return model, processor, device


# ==================== Prompt 构造 ====================

def build_llava_input(processor, rgb_image: Image.Image, gt_image: Image.Image, user_prompt: str, device):
    """
    构造符合 LLaVA-Next chat 模板的输入（双图像版本）。
    
    使用 conversation 格式，包含两张图像（RGB 和 GT）。
    """
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image", "image": rgb_image},  # Image 1: RGB
                {"type": "image", "image": gt_image},   # Image 2: GT
            ],
        }
    ]

    try:
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception as e:
        print(f"Warning: Method 1 failed, trying method 2: {e}")
        conversation_alt = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image"},
                    {"type": "image"},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            conversation_alt,
            images=[rgb_image, gt_image],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def build_llava_input_batch(processor, rgb_images: List[Image.Image], gt_images: List[Image.Image], user_prompts: List[str], device):
    """
    构造符合 LLaVA-Next chat 模板的批量输入（批处理版本，用于加速）。
    
    Args:
        processor: LLaVA processor
        rgb_images: RGB 图像列表
        gt_images: GT 图像列表
        user_prompts: 用户提示词列表（每个图像一个提示词）
        device: 设备
    
    Returns:
        inputs: processor(...) 的结果（包含 input_ids, pixel_values 等），已移动到指定设备
    """
    # 为每个图像构造 conversation
    conversations = []
    for rgb_img, gt_img, prompt in zip(rgb_images, gt_images, user_prompts):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": rgb_img},  # Image 1: RGB
                    {"type": "image", "image": gt_img},   # Image 2: GT
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
            padding=True,  # [建议] 显式添加，确保 batch 维度对齐
        )
    except Exception as e:
        # 如果批量处理失败，抛出异常，让调用者处理
        raise e
    
    # 将所有张量移动到指定设备
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


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
        
        # 尝试提取 qa_content 字段（更宽松的正则，允许嵌套对象）
        # 使用非贪婪匹配，但需要处理嵌套的大括号
        content_pattern = r'"qa_content"\s*:\s*(\{(?:[^{}]|(?:\{[^{}]*\}))*\}|null)'
        detail_pattern = r'"qa_detail"\s*:\s*(\{(?:[^{}]|(?:\{[^{}]*\}))*\}|null)'
        spatial_pattern = r'"qa_spatial"\s*:\s*(\{(?:[^{}]|(?:\{[^{}]*\}))*\}|null)'
        
        content_match = re.search(content_pattern, json_str, re.DOTALL)
        detail_match = re.search(detail_pattern, json_str, re.DOTALL)
        spatial_match = re.search(spatial_pattern, json_str, re.DOTALL)
        
        if content_match:
            # 手动构建 JSON
            qa_data = {}
            try:
                content_str = clean_match_text(content_match.group(1))
                if content_str and content_str.strip() != 'null':
                    content_json = json.loads('{"qa_content": ' + content_str + '}')
                    qa_data['qa_content'] = content_json.get('qa_content')
            except Exception as e:
                # 如果解析失败，尝试更简单的方法：直接提取 q 和 a 字段
                try:
                    # [改进] 提取 q 和 a 字段，处理可能包含注释的情况
                    # 使用非贪婪匹配，并处理引号内可能包含的转义字符
                    q_match = re.search(r'"q"\s*:\s*"((?:[^"\\]|\\.)*)"', content_match.group(1), re.DOTALL)
                    a_match = re.search(r'"a"\s*:\s*"((?:[^"\\]|\\.)*)"', content_match.group(1), re.DOTALL)
                    if q_match and a_match:
                        q_text = clean_match_text(q_match.group(1))
                        a_text = clean_match_text(a_match.group(1))
                        # 再次清理答案中的注释（可能在字符串中间，如 "A yellow banana."  <-- EXAMPLE ONLY!）
                        a_text = re.sub(r'<\s*--.*?--\s*>', '', a_text, flags=re.DOTALL | re.IGNORECASE).strip()
                        a_text = re.sub(r'<\s*--.*$', '', a_text, flags=re.MULTILINE | re.IGNORECASE).strip()
                        qa_data['qa_content'] = {
                            'q': q_text,
                            'a': a_text
                        }
                except:
                    pass
            
            if detail_match:
                detail_str = clean_match_text(detail_match.group(1))
                if detail_str and detail_str.strip() != 'null':
                    try:
                        detail_json = json.loads('{"qa_detail": ' + detail_str + '}')
                        qa_data['qa_detail'] = detail_json.get('qa_detail')
                    except:
                        # 尝试简单提取
                        try:
                            q_match = re.search(r'"q"\s*:\s*"([^"]*)"', detail_match.group(1))
                            a_match = re.search(r'"a"\s*:\s*"([^"]*)"', detail_match.group(1))
                            if q_match and a_match:
                                qa_data['qa_detail'] = {
                                    'q': clean_match_text(q_match.group(1)),
                                    'a': clean_match_text(a_match.group(1))
                                }
                            else:
                                qa_data['qa_detail'] = None
                        except:
                            qa_data['qa_detail'] = None
                else:
                    qa_data['qa_detail'] = None
            
            if spatial_match:
                spatial_str = clean_match_text(spatial_match.group(1))
                if spatial_str and spatial_str.strip() != 'null':
                    try:
                        spatial_json = json.loads('{"qa_spatial": ' + spatial_str + '}')
                        qa_data['qa_spatial'] = spatial_json.get('qa_spatial')
                    except:
                        # 尝试简单提取
                        try:
                            q_match = re.search(r'"q"\s*:\s*"([^"]*)"', spatial_match.group(1))
                            a_match = re.search(r'"a"\s*:\s*"([^"]*)"', spatial_match.group(1))
                            if q_match and a_match:
                                qa_data['qa_spatial'] = {
                                    'q': clean_match_text(q_match.group(1)),
                                    'a': clean_match_text(a_match.group(1))
                                }
                            else:
                                qa_data['qa_spatial'] = None
                        except:
                            qa_data['qa_spatial'] = None
                else:
                    qa_data['qa_spatial'] = None
            
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
    批量生成 3 类 QA 对（批处理版本，用于加速）
    
    Args:
        model: LLaVA 模型
        processor: LLaVA processor
        device: 设备
        rgb_images: RGB 图像列表
        gt_images: GT 图像列表
        bbox_norms: 归一化 bbox 列表 [xmin, ymin, xmax, ymax]
        max_retries: 最大重试次数（如果 JSON 解析失败）
    
    Returns:
        包含 3 类 QA 对的字典列表
    """
    import torch
    
    # [关键修复] 必须将尺寸限制在 336 以内，强迫 LLaVA 使用单 Patch 模式
    # LLaVA-Next v1.6 有动态切图机制 (AnyRes)：如果图片 > 336，会切分成多个 336x336 的块
    # 336 是单 Patch 的极限，设置为 336 让模型能看清物体细节，避免因分辨率太低导致幻觉
    # 设置为 336 确保每张图只占用 1 个 Patch ≈ 576 tokens，两张图 ≈ 1152 tokens
    # 加上精简的 System Prompt（~300 tokens）和对话格式（~500 tokens）≈ 1950 tokens（安全）
    max_size = 336  # 限制最大边长为 336，强制单 Patch 模式，提高分辨率以看清物体细节
    
    def resize_if_too_large(img, max_s):
        w, h = img.size
        if w > max_s or h > max_s:
            ratio = max_s / max(w, h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return img
    
    # 缩小所有图片
    rgb_images_small = [resize_if_too_large(img, max_size) for img in rgb_images]
    gt_images_small = [resize_if_too_large(img, max_size) for img in gt_images]
    
    # 为每个 bbox 构建 prompt
    user_prompts = []
    for bbox_norm in bbox_norms:
        bbox_str = format_bbox_string(bbox_norm)
        system_prompt = build_system_prompt(bbox_str)
        user_prompt = f"{system_prompt}\n\nNow, please generate the 3 types of Q&A pairs for the region {bbox_str}."
        user_prompts.append(user_prompt)
    
    # 批量处理（使用缩小后的图片）
    try:
        inputs = build_llava_input_batch(processor, rgb_images_small, gt_images_small, user_prompts, device)
        
        # [关键修复] 检查输入序列长度，如果超过模型限制则回退到逐个处理
        input_length = inputs["input_ids"].shape[1]
        max_model_length = getattr(model.config, "max_position_embeddings", 4096)
        
        if input_length > max_model_length:
            print(f"  ⚠ Warning: Batch input sequence length ({input_length}) exceeds model max length ({max_model_length})")
            print(f"     Falling back to individual processing...")
            # 回退到逐个处理
            raise ValueError(f"Input sequence too long: {input_length} > {max_model_length}")
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
            )
        
        # 解码所有响应
        batch_size = len(rgb_images)
        input_length = inputs["input_ids"].shape[1]
        results = []
        
        for i in range(batch_size):
            generated_ids = output_ids[i, input_length:]
            response_text = processor.batch_decode([generated_ids], skip_special_tokens=True)[0].strip()
            
            # 解析 JSON 响应
            qa_data = parse_qa_json(response_text)
            
            if qa_data is not None:
                # [新增] 格式自动修复逻辑（批处理版本）
                # 如果模型偷懒只返回了字符串，我们手动补上 Question
                
                # 修复 Content
                if isinstance(qa_data.get('qa_content'), str):
                    qa_data['qa_content'] = {
                        "q": "Describe the visual content in this region.",
                        "a": qa_data['qa_content']
                    }
                elif isinstance(qa_data.get('qa_content'), list):  # 修复列表情况
                    qa_data['qa_content'] = {
                        "q": "Describe the visual content in this region.",
                        "a": str(qa_data['qa_content'][0])
                    }
                
                # 修复 Detail
                if isinstance(qa_data.get('qa_detail'), str):
                    qa_data['qa_detail'] = {
                        "q": "What are the color and material of this object?",
                        "a": qa_data['qa_detail']
                    }
                
                # 修复 Spatial
                if isinstance(qa_data.get('qa_spatial'), str):
                    # Spatial 比较特殊，如果只返回答案，我们很难猜出方向，建议直接丢弃或泛化
                    qa_data['qa_spatial'] = None
                
                # [增强版] 强力清洗：去除 Prompt 泄露的指令词（批处理版本）
                def clean_text(text, is_question=False):
                    """清洗文本，去除 Prompt 泄露的指令词"""
                    if not isinstance(text, str):
                        return text
                    
                    # 1. 去掉常见的指令泄露后缀（大小写不敏感，匹配后面所有内容）
                    patterns = [
                        r"based on Img\d.*",          # 匹配 based on Img2 及其后面所有内容
                        r"based ONLY on Img\d.*",
                        r"\(GT\).*",
                        r"in the clear image.*",
                        r"based strictly on.*",
                    ]
                    for p in patterns:
                        # 替换为空字符串
                        text = re.sub(p, "", text, flags=re.IGNORECASE)
                    
                    # 2. 去掉首尾的标点和空格（防止切完后剩下一个句号）
                    text = text.strip().rstrip(".,;:").strip()
                    
                    # 3. 兜底修复：如果清洗后问题没有问号了，补一个（仅对问题字段）
                    if is_question and text and not text.endswith("?"):
                        text += "?"
                    
                    return text
                
                # 清洗所有问题和答案
                if qa_data.get('qa_content'):
                    if isinstance(qa_data['qa_content'], dict):
                        qa_data['qa_content']['q'] = clean_text(qa_data['qa_content'].get('q', ''), is_question=True)
                        qa_data['qa_content']['a'] = clean_text(qa_data['qa_content'].get('a', ''), is_question=False)
                
                if qa_data.get('qa_detail'):
                    if isinstance(qa_data['qa_detail'], dict):
                        qa_data['qa_detail']['q'] = clean_text(qa_data['qa_detail'].get('q', ''), is_question=True)
                        qa_data['qa_detail']['a'] = clean_text(qa_data['qa_detail'].get('a', ''), is_question=False)
                
                if qa_data.get('qa_spatial'):
                    if isinstance(qa_data['qa_spatial'], dict):
                        qa_data['qa_spatial']['q'] = clean_text(qa_data['qa_spatial'].get('q', ''), is_question=True)
                        qa_data['qa_spatial']['a'] = clean_text(qa_data['qa_spatial'].get('a', ''), is_question=False)
                
                # [新增] 强制注入坐标逻辑 + 强制覆盖 Content Question（批处理版本）
                # 注意：批处理模式下，每个样本的 bbox_norm 可能不同，需要在循环内处理
                current_bbox_norm = bbox_norms[i]  # 获取当前样本的 bbox
                bbox_str = f"[{current_bbox_norm[0]:.3f}, {current_bbox_norm[1]:.3f}, {current_bbox_norm[2]:.3f}, {current_bbox_norm[3]:.3f}]"
                
                # 1. 处理 Content Question：强制覆盖为标准格式（解决"问颜色"等问题）
                # 无论模型生成什么 Q，我们都把它替换为标准格式，既解决了"指令泄露"，也解决了"问颜色"的问题
                if qa_data.get('qa_content') and isinstance(qa_data['qa_content'], dict):
                    # 随机选择一个标准 Content 模板（只问"是什么"，不问颜色/材质）
                    content_templates = [
                        f"Describe the object in region {bbox_str}.",
                        f"What is the object located at {bbox_str}?",
                        f"Focus on region {bbox_str}. What is this?",
                        f"Identify the object in the bounding box {bbox_str}.",
                        f"What object is in region {bbox_str}?",
                    ]
                    qa_data['qa_content']['q'] = random.choice(content_templates)
                
                # 2. 处理 Detail Question：保持原样（不强制加坐标，保持对话的自然流畅性）
                # 3. 处理 Spatial Question：保持原样（不强制加坐标）
                
                result = {
                    "qa_content": qa_data.get("qa_content"),
                    "qa_detail": qa_data.get("qa_detail"),
                    "qa_spatial": qa_data.get("qa_spatial"),
                }
            else:
                # 如果解析失败，返回空结构（批处理模式下不重试，避免复杂度）
                print(f"  ⚠ Warning: Failed to parse JSON for batch item {i+1}/{batch_size}")
                print(f"     Response preview: {response_text[:200]}...")  # 打印前 200 个字符用于调试
                result = {
                    "qa_content": None,
                    "qa_detail": None,
                    "qa_spatial": None,
                }
            
            results.append(result)
        
        return results
        
    except Exception as e:
        # 如果批处理失败，回退到逐个处理
        print(f"Warning: Batch processing failed, falling back to individual processing: {e}")
        results = []
        for rgb_img, gt_img, bbox_norm in zip(rgb_images, gt_images, bbox_norms):
            result = generate_qa_pairs(model, processor, device, rgb_img, gt_img, bbox_norm, max_retries=0)
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
    为无反光图像生成简化的 QA 对（仅 Type 1 Content，全图 bbox）
    
    Args:
        model: LLaVA 模型
        processor: LLaVA processor
        device: 设备
        gt_image: GT 图像（原始，未裁剪）
        max_retries: 最大重试次数（通常不需要重试，设为0）
    
    Returns:
        包含 qa_content 的字典（qa_detail 和 qa_spatial 为 null）
    """
    import torch
    
    # 全图 bbox
    bbox_norm = [0.0, 0.0, 1.0, 1.0]
    bbox_str = "[0.000, 0.000, 1.000, 1.000]"
    
    # 限制图像尺寸（与正常流程一致）
    max_size = 336
    def resize_if_too_large(img, max_s):
        w, h = img.size
        if w > max_s or h > max_s:
            ratio = max_s / max(w, h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return img
    
    gt_image_small = resize_if_too_large(gt_image, max_size)
    
    # 简单的描述指令（只需GT图像，不需要RGB对比）
    user_prompt = f"Describe the visual content in the region {bbox_str}."
    
    # 构建输入（只需要GT图像，不需要RGB）
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image", "image": gt_image_small},
            ],
        }
    ]
    
    try:
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception as e:
        print(f"Warning: Failed to build input for no-glare image: {e}")
        return {
            "qa_content": None,
            "qa_detail": None,
            "qa_spatial": None,
        }
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 检查输入序列长度
    input_length = inputs["input_ids"].shape[1]
    max_model_length = getattr(model.config, "max_position_embeddings", 4096)
    
    if input_length > max_model_length:
        print(f"  ⚠ Warning: Input sequence length ({input_length}) exceeds model max length ({max_model_length}) for no-glare image")
        return {
            "qa_content": None,
            "qa_detail": None,
            "qa_spatial": None,
        }
    
    # 生成描述
    for attempt in range(max_retries + 1):
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=True,
                )
            
            generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
            response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            # 清洗生成的文本（去除指令泄露等）
            def clean_text(text):
                if not isinstance(text, str):
                    return text
                patterns = [
                    r"based on Img\d.*",
                    r"based ONLY on Img\d.*",
                    r"\(GT\).*",
                    r"in the clear image.*",
                    r"based strictly on.*",
                ]
                for p in patterns:
                    text = re.sub(p, "", text, flags=re.IGNORECASE)
                text = text.strip().rstrip(".,;:").strip()
                return text
            
            cleaned_answer = clean_text(response_text)
            
            # 构造结果
            result = {
                "qa_content": {
                    "q": f"Describe the visual content in the region {bbox_str}.",
                    "a": cleaned_answer,
                },
                "qa_detail": None,
                "qa_spatial": None,
            }
            return result
            
        except Exception as e:
            if attempt < max_retries:
                print(f"  ⚠ Generation failed for no-glare image, retrying ({attempt + 1}/{max_retries})...")
                continue
            else:
                print(f"  ⚠ Failed to generate QA for no-glare image after {max_retries + 1} attempts: {e}")
                return {
                    "qa_content": None,
                    "qa_detail": None,
                    "qa_spatial": None,
                }
    
    return {
        "qa_content": None,
        "qa_detail": None,
        "qa_spatial": None,
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
    为给定的 bbox 生成 3 类 QA 对（带重试机制，重试时提高温度增加随机性）
    
    Args:
        model: LLaVA 模型
        processor: LLaVA processor
        device: 设备
        rgb_image: RGB 图像（原始，未裁剪）
        gt_image: GT 图像（原始，未裁剪）
        bbox_norm: 归一化 bbox [xmin, ymin, xmax, ymax]（标准格式）
        max_retries: 最大重试次数（如果 JSON 解析失败）
    
    Returns:
        包含 3 类 QA 对的字典
    """
    import torch
    
    # [关键修复] 必须将尺寸限制在 336 以内，强迫 LLaVA 使用单 Patch 模式
    # LLaVA-Next v1.6 有动态切图机制 (AnyRes)：如果图片 > 336，会切分成多个 336x336 的块
    # 336 是单 Patch 的极限，设置为 336 让模型能看清物体细节，避免因分辨率太低导致幻觉
    # 设置为 336 确保每张图只占用 1 个 Patch ≈ 576 tokens，两张图 ≈ 1152 tokens
    # 加上精简的 System Prompt（~300 tokens）和对话格式（~500 tokens）≈ 1950 tokens（安全）
    max_size = 336  # 限制最大边长为 336，强制单 Patch 模式，提高分辨率以看清物体细节
    
    def resize_if_too_large(img, max_s):
        w, h = img.size
        if w > max_s or h > max_s:
            ratio = max_s / max(w, h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return img
    
    rgb_image_small = resize_if_too_large(rgb_image, max_size)
    gt_image_small = resize_if_too_large(gt_image, max_size)
    
    bbox_str = format_bbox_string(bbox_norm)
    system_prompt = build_system_prompt(bbox_str)
    
    # 构建完整的 user prompt（包含 system prompt）
    user_prompt = f"{system_prompt}\n\nNow, please generate the 3 types of Q&A pairs for the region {bbox_str}."
    
    # 重试机制：如果 JSON 解析失败，最多重试 max_retries 次
    # [改进] 重试时提高 temperature，增加随机性，避免重复相同的错误
    for attempt in range(max_retries + 1):
        # 使用缩小后的图片构建输入
        inputs = build_llava_input(processor, rgb_image_small, gt_image_small, user_prompt, device)
        
        # [关键修复] 检查输入序列长度，如果超过模型限制则跳过
        input_length = inputs["input_ids"].shape[1]
        max_model_length = getattr(model.config, "max_position_embeddings", 4096)
        
        if input_length > max_model_length:
            print(f"  ⚠ Warning: Input sequence length ({input_length}) exceeds model max length ({max_model_length})")
            print(f"     Skipping this sample to avoid indexing errors.")
            # 返回空结构，跳过这个样本
            return {
                "qa_content": None,
                "qa_detail": None,
                "qa_spatial": None,
            }
        
        # [改进] 重试时使用更高的 temperature，强制模型换一种说法
        current_temperature = TEMPERATURE if attempt == 0 else TEMPERATURE_RETRY
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,  # 使用 2048，确保完整输出
                temperature=current_temperature,
                top_p=TOP_P,
                do_sample=True,
            )
        
        generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
        response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # 解析 JSON 响应
        qa_data = parse_qa_json(response_text)
        
        if qa_data is not None:
            # [新增] 格式自动修复逻辑
            # 如果模型偷懒只返回了字符串，我们手动补上 Question
            
            # 修复 Content
            if isinstance(qa_data.get('qa_content'), str):
                qa_data['qa_content'] = {
                    "q": "Describe the visual content in this region.",
                    "a": qa_data['qa_content']
                }
            elif isinstance(qa_data.get('qa_content'), list):  # 修复列表情况
                qa_data['qa_content'] = {
                    "q": "Describe the visual content in this region.",
                    "a": str(qa_data['qa_content'][0])
                }
            
            # 修复 Detail
            if isinstance(qa_data.get('qa_detail'), str):
                qa_data['qa_detail'] = {
                    "q": "What are the color and material of this object?",
                    "a": qa_data['qa_detail']
                }
            
            # 修复 Spatial
            if isinstance(qa_data.get('qa_spatial'), str):
                # Spatial 比较特殊，如果只返回答案，我们很难猜出方向，建议直接丢弃或泛化
                qa_data['qa_spatial'] = None
            
            # [增强版] 强力清洗：去除 Prompt 泄露的指令词
            def clean_text(text, is_question=False):
                """清洗文本，去除 Prompt 泄露的指令词"""
                if not isinstance(text, str):
                    return text
                
                # 1. 去掉常见的指令泄露后缀（大小写不敏感，匹配后面所有内容）
                patterns = [
                    r"based on Img\d.*",          # 匹配 based on Img2 及其后面所有内容
                    r"based ONLY on Img\d.*",
                    r"\(GT\).*",
                    r"in the clear image.*",
                    r"based strictly on.*",
                ]
                for p in patterns:
                    # 替换为空字符串
                    text = re.sub(p, "", text, flags=re.IGNORECASE)
                
                # 2. 去掉首尾的标点和空格（防止切完后剩下一个句号）
                text = text.strip().rstrip(".,;:").strip()
                
                # 3. 兜底修复：如果清洗后问题没有问号了，补一个（仅对问题字段）
                if is_question and text and not text.endswith("?"):
                    text += "?"
                
                return text
            
            # 清洗所有问题和答案
            if qa_data.get('qa_content'):
                if isinstance(qa_data['qa_content'], dict):
                    qa_data['qa_content']['q'] = clean_text(qa_data['qa_content'].get('q', ''), is_question=True)
                    qa_data['qa_content']['a'] = clean_text(qa_data['qa_content'].get('a', ''), is_question=False)
            
            if qa_data.get('qa_detail'):
                if isinstance(qa_data['qa_detail'], dict):
                    qa_data['qa_detail']['q'] = clean_text(qa_data['qa_detail'].get('q', ''), is_question=True)
                    qa_data['qa_detail']['a'] = clean_text(qa_data['qa_detail'].get('a', ''), is_question=False)
            
            if qa_data.get('qa_spatial'):
                if isinstance(qa_data['qa_spatial'], dict):
                    qa_data['qa_spatial']['q'] = clean_text(qa_data['qa_spatial'].get('q', ''), is_question=True)
                    qa_data['qa_spatial']['a'] = clean_text(qa_data['qa_spatial'].get('a', ''), is_question=False)
            
            # [新增] 强制注入坐标逻辑 + 强制覆盖 Content Question
            # 获取 bbox 字符串（保留3位小数）
            # 注意：这里的 bbox_norm 是 [xmin, ymin, xmax, ymax]
            bbox_str = f"[{bbox_norm[0]:.3f}, {bbox_norm[1]:.3f}, {bbox_norm[2]:.3f}, {bbox_norm[3]:.3f}]"
            
            # 1. 处理 Content Question：强制覆盖为标准格式（解决"问颜色"等问题）
            # 无论模型生成什么 Q，我们都把它替换为标准格式，既解决了"指令泄露"，也解决了"问颜色"的问题
            if qa_data.get('qa_content') and isinstance(qa_data['qa_content'], dict):
                # 随机选择一个标准 Content 模板（只问"是什么"，不问颜色/材质）
                content_templates = [
                    f"Describe the object in region {bbox_str}.",
                    f"What is the object located at {bbox_str}?",
                    f"Focus on region {bbox_str}. What is this?",
                    f"Identify the object in the bounding box {bbox_str}.",
                    f"What object is in region {bbox_str}?",
                ]
                qa_data['qa_content']['q'] = random.choice(content_templates)
            
            # 2. 处理 Detail Question：保持原样（不强制加坐标，保持对话的自然流畅性）
            # Detail 通常不需要坐标，因为上下文里已经有了，或者指代了 object name
            
            # 3. 处理 Spatial Question：保持原样（不强制加坐标）
            # Spatial 问题通常不需要 Target 的坐标（因为是要找它），但需要 Anchor 的位置
            
            # 解析成功，提取 QA 对
            result = {
                "qa_content": qa_data.get("qa_content"),
                "qa_detail": qa_data.get("qa_detail"),
                "qa_spatial": qa_data.get("qa_spatial"),
            }
            return result
        
        # 解析失败，如果还有重试机会，继续（使用更高的 temperature）
        if attempt < max_retries:
            print(f"  ⚠ JSON parsing failed, retrying with higher temperature ({attempt + 1}/{max_retries})...")
            print(f"     Response preview: {response_text[:200]}...")  # 打印前 200 个字符用于调试
            continue
    
    # 所有重试都失败，返回空结构
    print(f"  ⚠ Failed to parse JSON after {max_retries + 1} attempts")
    print(f"     Final response preview: {response_text[:200]}...")  # 打印前 200 个字符用于调试
    return {
        "qa_content": None,
        "qa_detail": None,
        "qa_spatial": None,
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
    
    model_path = model_name or LLAVA_MODEL_NAME
    
    if output_json is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{OUTPUT_JSON_PREFIX}_{timestamp}.json"
    else:
        output_path = output_json
    
    rgb_dir = Path(rgb_root) if rgb_root else RGB_ROOT
    gt_dir = Path(gt_root) if gt_root else GT_ROOT
    
    print("Using differential method for glare detection (GT vs RGB)")
    print("Generating 3 types of Q&A pairs for each detected glare region")
    
    model, processor, device = load_llava_model(model_path)
    model.eval()
    
    # [加速方法 1] 使用 torch.compile 加速推理（PyTorch 2.0+）
    # 注意：torch.compile 与 bitsandbytes (4bit) 有时会有兼容性问题
    if USE_TORCH_COMPILE:
        try:
            import torch
            if hasattr(torch, 'compile'):
                print("Using torch.compile to accelerate model inference...")
                # 尝试编译模型，如果失败则继续使用原始模型
                model = torch.compile(model, mode="reduce-overhead")
                print("✓ torch.compile enabled (expected 20-30% speedup)")
            else:
                print("⚠ torch.compile not available (requires PyTorch 2.0+)")
        except (AttributeError, RuntimeError, TypeError) as e:
            # 捕获常见的兼容性错误（AttributeError, RuntimeError, TypeError）
            print(f"⚠ Warning: torch.compile failed (possibly incompatible with 4bit quantization), continuing with original model: {e}")
            print("  → This is normal if using 4bit quantization. Model will work without torch.compile.")
        except Exception as e:
            # 捕获其他未知错误
            print(f"⚠ Warning: torch.compile failed with unexpected error, continuing with original model: {e}")
    
    # [加速方法 2] 降低生成参数（可选，会略微降低质量但提高速度）
    # 如果速度是瓶颈，可以降低 MAX_NEW_TOKENS 或使用 do_sample=False
    print(f"Generation parameters: max_new_tokens={MAX_NEW_TOKENS}, temperature={TEMPERATURE}, top_p={TOP_P}")
    
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
        except Exception:
            bbox_norm = None
        
        # [处理逻辑] 如果没有检测到反光，生成简化版 QA（仅 Type 1 Content，全图 bbox）
        if bbox_norm is None:
            # 对于无反光图像，使用简化版生成函数
            try:
                qa_pairs = generate_qa_pairs_no_glare(
                    model, processor, device, gt_image, max_retries=0
                )
                
                # 构造样本（全图 bbox）
                sample = {
                    "id": f"{scene_id}_{rgb_filename}",
                    "image": str(rgb_path.relative_to(DATA_ROOT)),
                    "gt_image": str(gt_image_path.relative_to(DATA_ROOT)),
                    "scene_id": scene_id,
                    "bbox_norm": [0.0, 0.0, 1.0, 1.0],  # 全图坐标
                    "qa_content": qa_pairs.get("qa_content"),
                    "qa_detail": qa_pairs.get("qa_detail"),
                    "qa_spatial": qa_pairs.get("qa_spatial"),
                }
                results.append(sample)
            except Exception as e:
                print(f"\nWarning: Failed to generate no-glare QA for {scene_id}/{rgb_filename}: {e}")
                # 如果生成失败，跳过该图像
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
                                "qa_content": None,
                                "qa_detail": None,
                                "qa_spatial": None,
                            })
                
                # 组织数据并添加到结果中
                for i, qa_pairs in enumerate(batch_qa_pairs):
                    scene_id_item = batch_scene_ids[i]
                    rgb_filename_item = batch_rgb_filenames[i]
                    rgb_path_item = batch_rgb_paths[i]
                    gt_image_path_item = batch_gt_image_paths[i]
                    bbox_norm_item = batch_bbox_norms[i]
                    
                    sample = {
                        "id": f"{scene_id_item}_{rgb_filename_item}",
                        "image": str(rgb_path_item.relative_to(DATA_ROOT)),  # 原始 RGB 路径
                        "gt_image": str(gt_image_path_item.relative_to(DATA_ROOT)),  # 原始 GT 路径
                        "scene_id": scene_id_item,
                        "bbox_norm": bbox_norm_item,
                        "qa_content": qa_pairs.get("qa_content"),
                        "qa_detail": qa_pairs.get("qa_detail"),
                        "qa_spatial": qa_pairs.get("qa_spatial"),
                    }
                    
                    results.append(sample)
                
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
    print(f"Successfully generated {len(results)} Stage 3 Q&A pairs")
    print(f"Output file: {output_file.resolve()}")
    
    # 统计信息
    total_processed = len(rgb_items)
    skipped_no_glare = total_processed - len(results)
    with_content = sum(1 for r in results if r.get("qa_content") is not None)
    with_detail = sum(1 for r in results if r.get("qa_detail") is not None)
    with_spatial = sum(1 for r in results if r.get("qa_spatial") is not None)
    
    print(f"\nStatistics:")
    print(f"  - Total images processed: {total_processed}")
    print(f"  - Images with glare (generated Q&A): {len(results)}")
    print(f"  - Images without glare (skipped): {skipped_no_glare}")
    print(f"  - With Content Q&A: {with_content}")
    print(f"  - With Detail Q&A: {with_detail}")
    print(f"  - With Spatial Q&A: {with_spatial}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Stage 3 Q&A pairs (using non-cropped images)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help=f"LLaVA model path (default: {LLAVA_MODEL_NAME})",
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
    )


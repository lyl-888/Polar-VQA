"""
Stage 3 QA 对生成脚本
使用本地 Qwen2-VL-72B-Instruct-AWQ 模型，对 RGB + GT 数据集生成用于 PolarVLM Stage 3 训练的问答对。

核心思路：
- 基于 RGB + GT 直接生成 3 类物理感知 QA 对（A/B/C）：
  1. Task A: Positive (True Object Description)
  2. Task B: Physics (Reflection Identification)
  3. Task C: Hard Negative (Counterfactual / Anti-Deception)
- 使用非裁剪图像（原始 RGB 和 GT 图像），不使用反光 bbox
- 所有提示词和生成内容使用英文

工作流程：
1. 加载 RGB 和 GT 图像 → 2. 调用 Qwen2-VL 生成 3 类 QA 对 → 3. 保存为训练数据

依赖：
- transformers >= 4.40（支持 Qwen2VLForConditionalGeneration / AutoProcessor）
- qwen-vl-utils（用于 process_vision_info）
- auto-gptq（AWQ 量化支持）
- pillow, tqdm

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
from collections import Counter

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

from PIL import Image
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

# [修改] 生成参数（Qwen2-VL 优化）
MAX_NEW_TOKENS = 256           # 限制输出长度，避免长文本拖慢推理
TEMPERATURE = 0.2              # Qwen2-VL 比较聪明，0.1-0.3 就能生成很好的多样性，0.5 可能会有点野
TOP_P = 0.01                   # Qwen 官方推荐低 Top_P 或 Top_K
TEMPERATURE_RETRY = 0.4        # 重试时的温度（稍微提高随机性）
REPETITION_PENALTY = 1.05      # 抑制重复 token，缓解“复读机”现象

# [关键修改] 加速参数
BATCH_SIZE = 1                 # [关键] 72B AWQ 占用 ~42GB 显存，推理时 KV Cache 还需要空间，48GB 显卡只能跑 Batch Size = 1
USE_TORCH_COMPILE = False      # Qwen2-VL 不需要 torch.compile 也能跑得很快，且 compile 对动态分辨率支持一般

# [新增] 图像分辨率限制（Qwen2-VL 支持动态分辨率，但为了省显存需要限制最大尺寸）
MAX_IMAGE_SIZE = 1024          # 限制最大边长为 1024（如果显存 OOM，可以改为 768 或 512）
MIN_IMAGE_SIDE = 28            # Qwen2-VL 视觉分块因子相关，短边过小可能触发 height/width 错误

# 反光 bbox 检测与可视化已移除（不再使用坐标）

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
    # B 任务：你希望“问反射内容是什么”，不在问题里给出具体物体名（避免泄露答案）
    "Identify the reflection in the image.",
    "What is the reflection showing in the image?",
    "Describe what content appears in the reflection.",
]

# Task C Hard Negative Prompt 模板（英文）
NEGATIVE_PROMPT_TEMPLATES = [
    # C 任务：问“是否真实存在”，不在问题里提示 glare/reflection/polarization/surface
    # 同时避免 "in this scene/here" 这类上下文词，让问题更像“客观存在性判断”
    "Is there actually a {object} in this scene?",
]

# ==================== 物体名抽取与模板填充 ====================

_GENERIC_OBJECT_NAMES = {
    "object", "reflection", "reflections", "glare", "surface", "area", "region",
    "spot", "bright area", "bright spot", "light", "highlight", "reflected object",
    "glare spot", "reflection spot",
}

# 用于“是否提到了 object”的严格词匹配：避免把 in/on/at 这类介词误判为命中
_OBJECT_STOPWORDS = {
    "a", "an", "the",
    "there", "this", "that", "these", "those", "it", "its", "it's",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "in", "on", "at", "of", "for", "from", "with", "to", "into", "onto", "by", "as",
    "and", "or", "but",
    "visible", "shown", "seen", "appears", "appear", "appearing", "seems", "seem", "looks", "look", "looking",
    "area", "region", "spot", "surface", "corner", "side", "part", "background", "foreground", "center",
    "left", "right", "upper", "lower", "top", "bottom",
    # physics / reflection generic words (avoid treating them as object mentions)
    "reflection", "reflections", "glare", "highlight", "specular",
    "polarization", "signal", "dop", "aolp", "dolp", "map", "mask",
    # 防止 Physics 抽取误抓 "reflected off ..."
    "off",
}

# A/C 问句 object 简化：避免把颜色/材质等细节直接喂进问题里
_LEADING_ADJECTIVES = {
    # colors
    "red","green","blue","yellow","white","black","gray","grey","brown","orange","purple","pink",
    "gold","silver",
    # materials / textures
    "metal","metallic","wood","wooden","stone","brick","concrete","glass","plastic","fabric","leather",
    # sizes / shapes (常见，避免泄露)
    "big","small","large","tiny","tall","short","long","round","square",
}

_B_LOCATION_WORDS = {"top","bottom","left","right","center","corner"}
_B_ALLOWED_LOCATION_PHRASES = {
    "top-left corner", "top-right corner", "bottom-left corner", "bottom-right corner",
    "top-left", "top-right", "bottom-left", "bottom-right",
    "top side", "bottom side", "left side", "right side",
    "top", "bottom", "left", "right", "center", "corner",
}

def _simplify_object_for_question(obj: Optional[str]) -> Optional[str]:
    """把 'green metal bench' -> 'bench'；'tall tree trunk' -> 'tree trunk'（只去掉前缀形容词）。"""
    if not isinstance(obj, str) or not obj.strip():
        return obj
    words = obj.strip().split()
    # 去掉前缀话语词/冠词/连词
    while words and words[0].lower() in {"a","an","the","and","or","but"}:
        words = words[1:]
    # 去掉前缀形容词（颜色/材质/大小等）
    while words and words[0].lower() in _LEADING_ADJECTIVES:
        words = words[1:]
        while words and words[0].lower() in {"and","or"}:
            words = words[1:]
    if not words:
        return obj
    return " ".join(words)

def _physics_answer_has_location(a: str) -> bool:
    if not isinstance(a, str) or not a.strip():
        return False
    tokens = set(re.findall(r"[a-zA-Z]+", a.lower()))
    return any(w in tokens for w in _B_LOCATION_WORDS)

def _negative_answer_has_location(a: str) -> bool:
    if not isinstance(a, str) or not a.strip():
        return False
    tokens = set(re.findall(r"[a-zA-Z]+", a.lower()))
    return any(w in tokens for w in _B_LOCATION_WORDS)

def _extract_object_from_negative_question(q: str) -> Optional[str]:
    """从 Task C 的 existence question 中抽取 object（优先用这个，稳定且不会被答案的 Although 结构污染）。"""
    if not isinstance(q, str) or not q.strip():
        return None
    qq = q.strip()
    # 常见模板：
    # - Is there a {object} in this scene?
    # - Do you see a {object} here?
    patterns = [
        r"\b(?:is\s+there|do\s+you\s+see)\s+(?:actually\s+)?(?:a|an|the)\s+(.+?)\s+(?:in\s+this\s+scene|present\s+here|here|there)\s*\?\s*$",
        r"\b(?:is\s+there|do\s+you\s+see)\s+(?:actually\s+)?(?:a|an|the)\s+(.+?)\s*\?\s*$",
    ]
    for p in patterns:
        m = re.search(p, qq, flags=re.IGNORECASE)
        if m:
            cand_raw = m.group(1)
            cand_raw = _strip_location_suffix(cand_raw)
            cand = _normalize_object_name(cand_raw)
            if cand:
                return cand
    return None

def _extract_coarse_location_phrase(text: str) -> Optional[str]:
    """从 B 答案里抽一个粗粒度方位短语（top-left corner / left side / center 等）。"""
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.lower()
    # top-left / top left / top-left corner
    m = re.search(r"\b(top|bottom)\s*[- ]\s*(left|right)\b(?:\s+corner)?", t, flags=re.IGNORECASE)
    if m:
        # 标准化为 "top-left corner" 这种
        loc = f"{m.group(1).lower()}-{m.group(2).lower()} corner"
        return loc
    # "top-left corner" 直接匹配
    m = re.search(r"\b(top|bottom)\s*[- ]\s*(left|right)\s+corner\b", t, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1).lower()}-{m.group(2).lower()} corner"
    # left/right/top/bottom + side/corner/center
    m = re.search(r"\b(left|right|top|bottom)\s+(side|corner)\b", t, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1).lower()} {m.group(2).lower()}"
    if re.search(r"\bcenter\b", t):
        return "center"
    if re.search(r"\bcorner\b", t):
        return "corner"
    return None

def _valid_location_phrase(loc: Optional[str]) -> bool:
    if not isinstance(loc, str) or not loc.strip():
        return False
    l = loc.strip().lower()
    # normalize: "top left corner" -> "top-left corner"
    l = re.sub(r"\s+", " ", l)
    l = l.replace("top left", "top-left").replace("top right", "top-right").replace("bottom left", "bottom-left").replace("bottom right", "bottom-right")
    l = l.replace("top-left corner", "top-left corner").replace("top-right corner", "top-right corner")
    return l in _B_ALLOWED_LOCATION_PHRASES

def _sanitize_object_desc(desc: Optional[str], max_words: int = 12) -> Optional[str]:
    """用于 C 问句：允许带少量修饰词，但禁止带 reflection/glare/surface/polarization 等提示词。"""
    if not isinstance(desc, str) or not desc.strip():
        return None
    d = desc.strip()
    # remove banned words
    d = re.sub(r"\b(reflection|reflections|glare|surface|polarization|signal)\b", "", d, flags=re.IGNORECASE)
    d = re.sub(r"\s+", " ", d).strip()
    # truncate to at most max_words words
    words = d.split()
    if max_words and len(words) > max_words:
        d = " ".join(words[:max_words])
    # normalize object name style (still allow adjectives)
    d = d.strip().strip(".,;:").strip()
    if not d:
        return None
    return d

def _strip_location_suffix(text: str) -> str:
    if not isinstance(text, str):
        return text
    return re.sub(
        r"\s+(?:near|at|in|on)\s+the\s+(?:top|bottom|left|right|center)(?:[-\s]*(?:left|right))?(?:\s+corner|\s+side)?\s*$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    ).strip()

def _extract_desc_and_location_from_negative_answer(a: str) -> (Optional[str], Optional[str]):
    if not isinstance(a, str) or not a.strip():
        return None, None
    # 去掉前缀 "No." / "No," 等，避免污染描述
    t = re.sub(r'^(?:\s*no[.,]?\s*)+', '', a.strip(), flags=re.IGNORECASE).lstrip(" ,;:")
    # 尝试抓取 "Although <desc> is/are visible ..."
    m = re.search(r'Although\s+(.+?)\s+(?:is|are)\s+visible', t, flags=re.IGNORECASE)
    if not m:
        m = re.search(r'Although\s+(.+?)\s+(?:appears?|seems?)\b', t, flags=re.IGNORECASE)
    # 次级：句首直接描述 "The <desc> is/appears ..."
    if not m:
        m = re.search(r'^(?:the|a|an)\s+(.+?)\s+(?:is|are|was|were|appears?|seems?|looks?)\b', t, flags=re.IGNORECASE)
    desc = m.group(1).strip() if m else None
    # 抽取方位（near/at/in/on + the + location）
    loc = None
    mloc = re.search(
        r"\b(?:near|at|in|on)\s+the\s+((?:top|bottom|left|right|center)(?:[-\s]*(?:left|right))?(?:\s+corner|\s+side)?)\b",
        t,
        flags=re.IGNORECASE,
    )
    if mloc:
        loc = mloc.group(1).strip().lower()
    if desc:
        desc = _strip_location_suffix(desc)
        desc = re.sub(r'\s+(?:due to|because of|because|from|caused by|which|that|indicating)\b.*$', '', desc, flags=re.IGNORECASE).strip()
    return desc, loc

def _rewrite_physics_answer_strict(a: str, reflected_object: Optional[str], location_override: Optional[str] = None) -> str:
    """
    把 B 答案重写为你想要的结构：
    - 先给方位（从原答案抽取）
    - 再说呈现为什么内容（使用 C 抽到的 object；若没有则回退从 B 自己抽）
    - 物理 ending 交给 _normalize_physics_answer 补齐
    """
    if not isinstance(a, str):
        return ""
    loc = None
    if _valid_location_phrase(location_override):
        loc = location_override.strip().lower()
        loc = re.sub(r"\s+", " ", loc)
        loc = loc.replace("top left", "top-left").replace("top right", "top-right").replace("bottom left", "bottom-left").replace("bottom right", "bottom-right")
        if loc in {"top-left","top-right","bottom-left","bottom-right"}:
            loc = loc + " corner"
    loc = loc or (_extract_coarse_location_phrase(a) or "top-left corner")
    obj = reflected_object
    if not obj:
        obj = _extract_reflection_object_from_text(a)
    obj = _normalize_object_name(obj) if obj else None
    obj = obj or "reflected object"
    # 只保留物体名，不泄露形容词
    obj = _simplify_object_for_question(obj) or obj
    # 严格格式（英文）+ 物理 ending（避免模型偷懒）
    return f"The reflection is mainly in the {loc}, appearing as {obj}. This is indicated by the high polarization signal in that area."

def _format_question_from_templates(templates: List[str]) -> str:
    """用于不需要 {object} 的模板（Task B）。"""
    if not templates:
        return ""
    return random.choice(templates)

def _strip_leading_copula(text: str) -> str:
    """
    去掉答案开头的模板句式，防止抽取到 'There is ...' 这种整句当 object。
    """
    if not isinstance(text, str):
        return ""
    t = text.strip()
    # 常见开头：There is/are, This is, It is, These are, Those are ...
    t = re.sub(
        r'^\s*(?:there\s+(?:is|are)|this\s+is|it\s+is|these\s+are|those\s+are)\s+',
        '',
        t,
        flags=re.IGNORECASE,
    )
    return t.strip()

def _ensure_period(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if not t:
        return t
    if t.endswith((".", "!", "?")):
        return t
    return t + "."

def _normalize_positive_answer(a: str) -> str:
    """
    Task A 答案规范化（不改语义，只让格式更像 caption）：
    - 去掉 There is/This is/It is... 开头
    - 弱化 "is placed/located/visible" 等啰嗦动词结构
    """
    if not isinstance(a, str):
        return ""
    t = a.strip()
    if not t:
        return t
    t = _strip_leading_copula(t)
    # "X is placed/located/sitting ..." -> "X ..."
    t = re.sub(r"\b(is|are)\s+(placed|located|sitting|positioned)\s+\b", " ", t, flags=re.IGNORECASE)
    # "X is visible ..." -> "X ..."
    t = re.sub(r"\b(is|are)\s+visible\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _normalize_physics_answer(a: str) -> str:
    """
    Task B 答案规范化：必须有标准的物理结尾句式，避免训练分布漂移。
    """
    if not isinstance(a, str):
        return ""
    t = a.strip()
    if not t:
        return t
    ending = "This is indicated by the high polarization signal in that area."
    t_lower = t.lower()
    if ("high polarization" in t_lower) and ("indicat" in t_lower) and ("that area" in t_lower):
        return _ensure_period(t)
    t = _ensure_period(t)
    return (t + " " + ending).strip()

def _normalize_negative_answer(a: str) -> str:
    """
    Task C 答案规范化：必须以 No. 开头，且包含物理解释关键字。
    """
    if not isinstance(a, str):
        return ""
    t = a.strip()
    if not t:
        return t
    # 移除重复的 "No/No," 前缀，统一成单个 "No."
    t = re.sub(r'^(?:\s*no[.,]?\s*)+', '', t, flags=re.IGNORECASE).lstrip(" ,;:")
    t = "No. " + t if t else "No."
    t_lower = t.lower()
    if "high polarization" not in t_lower:
        t = t.rstrip() + " The high polarization indicates it is a reflection or glare on the surface, not a real physical object."
    return _ensure_period(t)


def _normalize_object_name(name: str) -> Optional[str]:
    if not isinstance(name, str):
        return None
    cleaned = _strip_leading_copula(name)
    # 去掉常见话语标记开头，防止 "Although a tree" 被当作 object
    cleaned = re.sub(r'^\s*(?:although|though|however|but|no)\s+', '', cleaned.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'^\s*(a|an|the)\s+', '', cleaned.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'[\s\.,;:!]+$', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if not cleaned:
        return None
    # 去掉尾部介词/方位等无信息 token（避免 "trees are visible in" 抽到 "in"）
    words = cleaned.split()
    while words and words[-1].lower() in _OBJECT_STOPWORDS:
        words = words[:-1]
    if not words:
        return None
    # 截断到前 4 个词，避免把完整句子当 object
    if len(words) > 4:
        words = words[:4]
    cleaned = " ".join(words).strip()
    if cleaned.lower() in _GENERIC_OBJECT_NAMES:
        return None
    # 防止“reflection/glare”类泛词作为 object
    lowered_tokens = set(re.findall(r"[a-zA-Z0-9\-]+", cleaned.lower()))
    if cleaned.lower() in {"reflection", "reflections", "glare"}:
        return None
    # 进一步：包含这些泛词的短语也不作为 object（例如 "reflection on the tree trunk"）
    if lowered_tokens & {"reflection", "reflections", "glare", "surface"}:
        return None
    return cleaned


def _extract_object_from_text(text: str) -> Optional[str]:
    """尽量从答案里抽取具体物体名（person/car/sign 等）。"""
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = re.sub(r'^\s*No[.,]?\s*', '', text.strip(), flags=re.IGNORECASE)
    cleaned = _strip_leading_copula(cleaned)
    patterns = [
        # 0) 优先吃掉让步结构，避免被“句首主语 + 系动词”误抓成 "Although a tree"
        r'Although\s+(?:a|an|the)\s+([a-zA-Z0-9\- ]{2,80}?)\s+(?:is|are|was|were|appears|appear|appearing|seems|seem|might\s+appear|may\s+appear|could\s+appear|can\s+appear|might\s+be|may\s+be|could\s+be|can\s+be)\b',
        # 1) 句首主语 + 系动词： "Trees are visible ..." / "A chair is ..." -> 抽 Trees / chair
        r'^([a-zA-Z0-9\- ]{2,80}?)\s+(?:is|are|was|were|appears|appear|seems|seem|can\s+be|could\s+be|may\s+be|might\s+be|might\s+appear|may\s+appear|could\s+appear|can\s+appear)\b',
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
    # fallback: 取前 2-3 个词（更保守，减少把整句塞回模板）
    fallback = " ".join(cleaned.split()[:3])
    return _normalize_object_name(fallback)

def _extract_reflection_object_from_text(text: str) -> Optional[str]:
    """
    Physics(B) 专用：优先抽取“反射里看到的具体物体”，而不是抽到 reflection/glare 本身。
    """
    if not isinstance(text, str) or not text.strip():
            return None
    cleaned = re.sub(r'^\s*No[.,]?\s*', '', text.strip(), flags=re.IGNORECASE)
    cleaned = _strip_leading_copula(cleaned)

    patterns = [
        # "The reflection shows a blurred image of trees and sky, ..."
        r'\breflection\s+(?:shows?|depicts?|contains?)\s+(?:a|an|the)?\s*([a-zA-Z0-9\- ]{2,80}?)\b(?:,|\.|\bindicated\b|\bindicating\b|\bwhich\b|\bthat\b)',
        # "reflection of trees/buildings ..."
        r'\breflection\s+(?:of|from)\s+(?:a|an|the)?\s*([a-zA-Z0-9\- ]{2,80}?)\b',
        # "reflected trees/buildings ..."（排除 "reflected off ..." 这种介词短语）
        r'\breflected\s+(?!off\b)([a-zA-Z0-9\- ]{2,80}?)\b',
        # "shows trees in the reflection"
        r'\bshows?\s+(?:a|an|the)?\s*([a-zA-Z0-9\- ]{2,80}?)\s+\b(?:in|within)\s+the\s+reflection\b',
        # "The reflection on the tree trunk appears to be sunlight ..."
        r'\breflection\b.*?\bappears?\s+to\s+be\s+(?:a|an|the)?\s*([a-zA-Z0-9\- ]{2,80}?)\b(?:,|\.|\bindicated\b|\bindicating\b|\bwhich\b|\bthat\b|\bby\b)',
    ]
    for p in patterns:
        m = re.search(p, cleaned, flags=re.IGNORECASE)
        if not m:
            continue
        cand = m.group(1).strip()
        # 清理常见修饰短语："blurred image of" / "image of"
        cand = re.sub(r'^\s*(?:a|an|the)?\s*(?:blurred\s+)?(?:image|view|picture)\s+of\s+', '', cand, flags=re.IGNORECASE)
        cand = re.sub(r'^\s*(?:a|an|the)\s+', '', cand, flags=re.IGNORECASE).strip()
        cand_norm = _normalize_object_name(cand)
        if cand_norm:
            return cand_norm

    # fallback：回退到通用抽取
    return _extract_object_from_text(text)


def _format_object_question(templates: List[str], object_name: Optional[str], fallback_object: str) -> str:
    template = random.choice(templates)
    obj = object_name or fallback_object
    # 轻量规范化：常见名词短语尽量用小写开头（避免 "Trees and foliage" 出现在句中间）
    if isinstance(obj, str) and obj[:1].isupper() and obj.lower() not in {"i"}:
        obj = obj[:1].lower() + obj[1:]
    # 如果模板本身已有冠词（a/an/the），则避免 obj 再带冠词导致 "a a ..."
    if isinstance(obj, str) and re.search(r"\b(a|an|the)\s+\{object\}", template, flags=re.IGNORECASE):
        obj = re.sub(r'^\s*(a|an|the)\s+', '', obj.strip(), flags=re.IGNORECASE)
    if "{object}" in template:
        return template.format(object=obj)
    return template.replace("[OBJECT]", obj).replace("<OBJECT>", obj)


def _question_mentions_object(question: str, object_name: Optional[str]) -> bool:
    if not isinstance(question, str) or not question.strip() or not object_name:
        return False
    # 用 token-level 精确匹配，避免把 "in" 这种介词当成命中
    q_tokens = re.findall(r"[a-zA-Z0-9\-]+", question.lower())
    obj_tokens = [t for t in re.findall(r"[a-zA-Z0-9\-]+", object_name.lower()) if t not in _OBJECT_STOPWORDS]
    obj_tokens = [t for t in obj_tokens if len(t) >= 3]
    if not obj_tokens:
        return False
    q_set = set(q_tokens)
    return any(t in q_set for t in obj_tokens)


def _ensure_object_bound_question(
    qa_item: Dict,
    templates: List[str],
    fallback_object: str,
    force_template_if_missing_object: bool = False,
    always_use_template: bool = False,
    object_extractor=None,
    simplify_object: bool = False,
) -> None:
    if not isinstance(qa_item, dict):
        return
    question = qa_item.get("q", "")
    answer = qa_item.get("a", "")
    extractor = object_extractor or _extract_object_from_text
    try:
        object_name = extractor(answer)
    except Exception:
        object_name = _extract_object_from_text(answer)
    if simplify_object:
        object_name = _simplify_object_for_question(object_name)

    # 硬规则：直接用模板重写（A 任务用）
    if always_use_template:
        qa_item["q"] = _format_object_question(templates, object_name, fallback_object)
        return

    if not question or "{object}" in question or "[OBJECT]" in question or "<OBJECT>" in question:
        qa_item["q"] = _format_object_question(templates, object_name, fallback_object)
        return
    # 若问题过于泛化且未包含具体物体名，则替换为占位模板
    generic_hits = re.search(r"\b(object|reflection|glare|bright area|surface)\b", question, flags=re.IGNORECASE)
    if (generic_hits or force_template_if_missing_object) and not _question_mentions_object(question, object_name):
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
  * **A**: A concise one-sentence description of the object that is visible clearly in **Image 2 (GT)** but is partially/fully obscured by glare in **Image 1 (RGB)**. Include 1-2 concrete attributes if possible (e.g., color/material/shape/texture). Do NOT mention reflections or polarization here. Do NOT start the answer with \"There is\" / \"There are\".

* **Task B: Physics Awareness (Glare/Reflection Identification)**
  * **Goal**: Teach the model to explicitly locate and describe the glare/reflection using polarization logic.
  * **Q**: Ask what the reflection content is (vary phrasing slightly). Do NOT reveal the reflected object in the question.
  * **A**: Describe what the glare/reflection looks like (based on differences between Image 1 and Image 2), and where it appears. You MUST include at least one coarse location word: **top / bottom / left / right / center / corner**. Choose the location based on the visual evidence; do NOT default to the same location across samples. You **MUST** end the answer by stating the physics reason.
  * *Example ending*: "This is indicated by the high polarization signal in that area."

* **Task C: Hard Negative (Anti-Deception)**
  * **Goal**: Trick the model into hallucinating the reflected object or glare spot as a real object, and teach it to say "No".
  * **Q**: Ask a natural existence question like "Is there a [Reflected Object] here?" Replace [Reflected Object] with a specific thing you found in the reflection of Image 1, and make sure the object name appears in the question. Do NOT mention glare, reflection, surface, or polarization in the question.
  * **A**: You **MUST** start with "No." and explain the physics. You must also mention **where** the reflected object appears in the image (coarse location words like top/bottom/left/right/center/corner) and give a short descriptive phrase (1-3 modifiers) for the reflected object.

* **Extra metadata (for internal formatting, do NOT mention in Q/A text unless asked)**
  * Also output a `meta` object with:
    * `reflection_location`: the coarse location of the **same reflected object mentioned in Task C's answer**. Choose ONE phrase from: "top-left corner", "top-right corner", "bottom-left corner", "bottom-right corner", "left side", "right side", "top side", "bottom side", "center".
    * `negative_object_desc`: a short noun phrase describing that same reflected object (the one used in Task C's answer) with 1-3 modifiers (e.g., "a tall leafy tree", "a distant building silhouette"). Do NOT include the words reflection/glare/surface/polarization in this phrase.

**IMPORTANT:**
- All questions and answers must be in English.
- Do NOT include numeric coordinates.
- Do NOT mention Image 1/2 or GT/RGB in the final Q/A text.

**Output Format (Strict Constraint):**
You must return ONLY a valid JSON object. Do not output any other text, explanations, or Markdown tags.

{{
  "qa_positive": {{"q": "...", "a": "..."}},
  "qa_physics": {{"q": "...", "a": "..."}},
  "qa_negative": {{"q": "...", "a": "..."}},
  "meta": {{"reflection_location": "...", "negative_object_desc": "..."}}
}}

**IMPORTANT: Output JSON directly, no other content.**

"""


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


def build_stage2_style_records(
    rgb_path: Path,
    gt_image_path: Path,
    scene_id: str,
    qa_pairs: Dict,
    sample_type: str = "typeB_visual",
) -> List[Dict]:
    """
    将内部 qa_pairs 结构转换为 Stage2 风格的扁平 JSON 记录列表。

    输出格式对齐 `merged_stage2_new_format_rewrite.json`：
    - 一条 QA 对对应一条样本记录
    - 字段包含 image / input_path / gt_path / scene_id / type / subtype / conversations
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
                "conversations": [
                    {"from": "human", "value": q.strip()},
                    {"from": "gpt", "value": a.strip()},
                ],
            }
        )

    return records


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
    for rgb_img, gt_img in zip(rgb_images, gt_images):
        # 1. 预处理图片（稳定模式：统一成 896x896）
        # [修改] 不再画红框，直接使用原图
        rgb_small = safe_resize(rgb_img)
        gt_small = safe_resize(gt_img)
        
        # 3. 构建 prompt
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
                        object_name = _simplify_object_for_question(_extract_object_from_text(answer_text))
                        qa_data['qa_positive'] = {
                            "q": _format_object_question(POSITIVE_PROMPT_TEMPLATES, object_name, "object"),
                            "a": answer_text
                        }
                    elif isinstance(qa_data.get('qa_positive'), list):
                        answer_text = str(qa_data['qa_positive'][0])
                        object_name = _simplify_object_for_question(_extract_object_from_text(answer_text))
                        qa_data['qa_positive'] = {
                            "q": _format_object_question(POSITIVE_PROMPT_TEMPLATES, object_name, "object"),
                            "a": answer_text
                        }
                    
                    # 修复 Physics
                    if isinstance(qa_data.get('qa_physics'), str):
                        answer_text = qa_data['qa_physics']
                        qa_data['qa_physics'] = {
                            "q": _format_question_from_templates(PHYSICS_PROMPT_TEMPLATES),
                            "a": answer_text
                        }
                    
                    # 修复 Negative
                    if isinstance(qa_data.get('qa_negative'), str):
                        answer_text = qa_data['qa_negative']
                        object_name = _simplify_object_for_question(_extract_object_from_text(answer_text))
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
                    
                    # ✅ 答案硬约束后处理（让 A/B/C 的输出稳定可训练）
                    if qa_data.get('qa_positive') and isinstance(qa_data['qa_positive'], dict):
                        qa_data['qa_positive']['a'] = _normalize_positive_answer(qa_data['qa_positive'].get('a', ''))
                    if qa_data.get('qa_physics') and isinstance(qa_data['qa_physics'], dict):
                        qa_data['qa_physics']['a'] = _normalize_physics_answer(qa_data['qa_physics'].get('a', ''))
                    if qa_data.get('qa_negative') and isinstance(qa_data['qa_negative'], dict):
                        qa_data['qa_negative']['a'] = _normalize_negative_answer(qa_data['qa_negative'].get('a', ''))
                    
                    # 补齐/强化问题（避免空问题或过于泛化）
                    if qa_data.get('qa_positive') and isinstance(qa_data['qa_positive'], dict):
                        # A 任务：硬约束——问题永远使用模板，并填入抽到的 object
                        _ensure_object_bound_question(
                            qa_data['qa_positive'],
                            POSITIVE_PROMPT_TEMPLATES,
                            "object",
                            always_use_template=True,
                            simplify_object=True,
                        )
                    if qa_data.get('qa_physics') and isinstance(qa_data['qa_physics'], dict):
                        # B 任务：硬约束——问题永远使用 B 模板（不包含具体物体名）
                        qa_data['qa_physics']["q"] = _format_question_from_templates(PHYSICS_PROMPT_TEMPLATES)
                    if qa_data.get('qa_negative') and isinstance(qa_data['qa_negative'], dict):
                        # C 任务也强制模板：避免出现 "in the image/photo" 等漏网表述，并严格遵守“不提 surface/reflect/glare/polarization”
                        _ensure_object_bound_question(
                            qa_data['qa_negative'],
                            NEGATIVE_PROMPT_TEMPLATES,
                            "object",
                            always_use_template=True,
                            simplify_object=True,
                        )

                    # ✅ 强制 B 答案包含方位词；否则触发自动重试
                    if qa_data.get('qa_physics') and isinstance(qa_data['qa_physics'], dict):
                        if not _physics_answer_has_location(qa_data['qa_physics'].get("a", "")):
                            raise ValueError("Physics(B) answer missing coarse location word (top/bottom/left/right/center/corner)")
                        # ✅ 强制 B 答案包含“反射内容”（用 C 的 object），并规整为固定结构
                        neg_q = (qa_data.get("qa_negative") or {}).get("q", "") if isinstance(qa_data.get("qa_negative"), dict) else ""
                        neg_a = (qa_data.get("qa_negative") or {}).get("a", "") if isinstance(qa_data.get("qa_negative"), dict) else ""
                        reflected_obj = _extract_object_from_negative_question(neg_q) or _extract_object_from_text(neg_a)
                        meta = qa_data.get("meta") if isinstance(qa_data.get("meta"), dict) else {}
                        meta_loc = meta.get("reflection_location") if isinstance(meta, dict) else None
                        neg_loc = _extract_coarse_location_phrase(neg_a)
                        qa_data['qa_physics']["a"] = _rewrite_physics_answer_strict(
                            qa_data['qa_physics'].get("a", ""),
                            reflected_obj,
                            location_override=(neg_loc or meta_loc),
                        )

                    # ✅ C：用“答案里的完整描述+方位”重写 C 问句（不改 C 答案）
                    if qa_data.get('qa_negative') and isinstance(qa_data['qa_negative'], dict):
                        neg_a = qa_data['qa_negative'].get("a", "")
                        neg_q = qa_data['qa_negative'].get("q", "")
                        desc, loc = _extract_desc_and_location_from_negative_answer(neg_a)
                        if not desc:
                            desc = _extract_object_from_text(neg_a) or _extract_object_from_negative_question(neg_q)
                        desc = _sanitize_object_desc(desc, max_words=14) if desc else None
                        loc = loc or _extract_coarse_location_phrase(neg_a)
                        if desc:
                            desc_core = _strip_location_suffix(desc)
                            # 若描述没有冠词，加一个 "a"
                            if not re.match(r'^(a|an|the)\b', desc_core, flags=re.IGNORECASE):
                                desc_core = f"a {desc_core}"
                            if loc and not re.search(r"\b(near|at|in|on)\b", desc_core, flags=re.IGNORECASE):
                                desc_full = f"{desc_core} near the {loc}"
                            else:
                                desc_full = desc_core
                            qa_data['qa_negative']["q"] = f"Is there actually {desc_full}?"
                        # 硬约束：C 答案必须包含方位词，否则重试
                        if not _negative_answer_has_location(qa_data['qa_negative'].get("a", "")):
                            raise ValueError("Negative(C) answer missing coarse location word (top/bottom/left/right/center/corner)")
                    
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


def generate_qa_pairs(
    model,
    processor,
    device,
    rgb_image: Image.Image,
    gt_image: Image.Image,
    max_retries: int = 1,
) -> Dict:
    """
    为给定图像生成 3 类 QA 对（Qwen2-VL 版本，带重试机制）
    
    Args:
        model: Qwen2-VL 模型
        processor: Qwen2-VL processor
        device: 设备
        rgb_image: RGB 图像（原始，未裁剪）
        gt_image: GT 图像（原始，未裁剪）
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
                object_name = _simplify_object_for_question(_extract_object_from_text(answer_text))
                qa_data['qa_positive'] = {
                    "q": _format_object_question(POSITIVE_PROMPT_TEMPLATES, object_name, "object"),
                    "a": answer_text
                }
            elif isinstance(qa_data.get('qa_positive'), list):
                answer_text = str(qa_data['qa_positive'][0])
                object_name = _simplify_object_for_question(_extract_object_from_text(answer_text))
                qa_data['qa_positive'] = {
                    "q": _format_object_question(POSITIVE_PROMPT_TEMPLATES, object_name, "object"),
                    "a": answer_text
                }
            
            # 修复 Physics
            if isinstance(qa_data.get('qa_physics'), str):
                answer_text = qa_data['qa_physics']
                qa_data['qa_physics'] = {
                    "q": _format_question_from_templates(PHYSICS_PROMPT_TEMPLATES),
                    "a": answer_text
                }
            
            # 修复 Negative
            if isinstance(qa_data.get('qa_negative'), str):
                answer_text = qa_data['qa_negative']
                object_name = _simplify_object_for_question(_extract_object_from_text(answer_text))
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
            
            # ✅ 答案硬约束后处理（让 A/B/C 的输出稳定可训练）
            if qa_data.get('qa_positive') and isinstance(qa_data['qa_positive'], dict):
                qa_data['qa_positive']['a'] = _normalize_positive_answer(qa_data['qa_positive'].get('a', ''))
            if qa_data.get('qa_physics') and isinstance(qa_data['qa_physics'], dict):
                qa_data['qa_physics']['a'] = _normalize_physics_answer(qa_data['qa_physics'].get('a', ''))
            if qa_data.get('qa_negative') and isinstance(qa_data['qa_negative'], dict):
                qa_data['qa_negative']['a'] = _normalize_negative_answer(qa_data['qa_negative'].get('a', ''))
            
            # 补齐/强化问题（避免空问题或过于泛化）
            if qa_data.get('qa_positive') and isinstance(qa_data['qa_positive'], dict):
                _ensure_object_bound_question(
                    qa_data['qa_positive'],
                    POSITIVE_PROMPT_TEMPLATES,
                    "object",
                    always_use_template=True,
                    simplify_object=True,
                )
            if qa_data.get('qa_physics') and isinstance(qa_data['qa_physics'], dict):
                qa_data['qa_physics']["q"] = _format_question_from_templates(PHYSICS_PROMPT_TEMPLATES)
            if qa_data.get('qa_negative') and isinstance(qa_data['qa_negative'], dict):
                # C 任务也强制模板：避免出现 "in the image/photo" 等漏网表述，并严格遵守“不提 surface/reflect/glare/polarization”
                _ensure_object_bound_question(
                    qa_data['qa_negative'],
                    NEGATIVE_PROMPT_TEMPLATES,
                    "object",
                    always_use_template=True,
                    simplify_object=True,
                )

            # ✅ 强制 B 答案包含方位词；否则触发自动重试（在 generate_qa_pairs 的重试循环中捕获）
            if qa_data.get('qa_physics') and isinstance(qa_data['qa_physics'], dict):
                if not _physics_answer_has_location(qa_data['qa_physics'].get("a", "")):
                    raise ValueError("Physics(B) answer missing coarse location word (top/bottom/left/right/center/corner)")
                # ✅ 强制 B 答案包含“反射内容”（用 C 的 object），并规整为固定结构
                neg_q = (qa_data.get("qa_negative") or {}).get("q", "") if isinstance(qa_data.get("qa_negative"), dict) else ""
                neg_a = (qa_data.get("qa_negative") or {}).get("a", "") if isinstance(qa_data.get("qa_negative"), dict) else ""
                reflected_obj = _extract_object_from_negative_question(neg_q) or _extract_object_from_text(neg_a)
                meta = qa_data.get("meta") if isinstance(qa_data.get("meta"), dict) else {}
                meta_loc = meta.get("reflection_location") if isinstance(meta, dict) else None
                neg_loc = _extract_coarse_location_phrase(neg_a)
                qa_data['qa_physics']["a"] = _rewrite_physics_answer_strict(
                    qa_data['qa_physics'].get("a", ""),
                    reflected_obj,
                    location_override=(neg_loc or meta_loc),
                )

            # ✅ C：用“答案里的完整描述+方位”重写 C 问句（不改 C 答案）
            if qa_data.get('qa_negative') and isinstance(qa_data['qa_negative'], dict):
                neg_a = qa_data['qa_negative'].get("a", "")
                neg_q = qa_data['qa_negative'].get("q", "")
                desc, loc = _extract_desc_and_location_from_negative_answer(neg_a)
                if not desc:
                    desc = _extract_object_from_text(neg_a) or _extract_object_from_negative_question(neg_q)
                desc = _sanitize_object_desc(desc, max_words=14) if desc else None
                loc = loc or _extract_coarse_location_phrase(neg_a)
                if desc:
                    desc_core = _strip_location_suffix(desc)
                    if not re.match(r'^(a|an|the)\b', desc_core, flags=re.IGNORECASE):
                        desc_core = f"a {desc_core}"
                    if loc and not re.search(r"\b(near|at|in|on)\b", desc_core, flags=re.IGNORECASE):
                        desc_full = f"{desc_core} near the {loc}"
                    else:
                        desc_full = desc_core
                    qa_data['qa_negative']["q"] = f"Is there actually {desc_full}?"
                if not _negative_answer_has_location(qa_data['qa_negative'].get("a", "")):
                    raise ValueError("Negative(C) answer missing coarse location word (top/bottom/left/right/center/corner)")
            
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
    
    model_path = model_name or QWEN_MODEL_NAME
    
    if output_json is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{OUTPUT_JSON_PREFIX}_{timestamp}.json"
    else:
        output_path = output_json
    rgb_dir = Path(rgb_root) if rgb_root else RGB_ROOT
    gt_dir = Path(gt_root) if gt_root else GT_ROOT
    
    print("Generating A/B/C physics-aware Q&A pairs (no glare bbox detection)")
    
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
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_every = 50

    def _save_results_snapshot():
        """周期性保存，避免中途终止丢数据（原子写入）。"""
        tmp_path = output_file.with_suffix(output_file.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        tmp_path.replace(output_file)
    
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
    batch_scene_ids = []
    batch_rgb_filenames = []
    batch_rgb_paths = []
    batch_gt_image_paths = []
    
    processed_images = 0
    save_every_images = 50
    next_save_images = save_every_images

    try:
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
        
                # [批处理改进] 收集到批次中
            batch_rgb_images.append(rgb_image)
            batch_gt_images.append(gt_image)
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
                                model,
                                processor,
                                device,
                                batch_rgb_images,
                                batch_gt_images,
                                max_retries=0,  # 批处理模式下不重试，避免复杂度
                        )
                    except Exception as e:
                        print(f"\nWarning: Batch generation failed, falling back to individual processing: {e}")
                        # 回退到逐个处理
                        batch_qa_pairs = []
                        for rgb_img, gt_img in zip(batch_rgb_images, batch_gt_images):
                            try:
                                qa_pairs = generate_qa_pairs(
                                    model, processor, device, rgb_img, gt_img, max_retries=1
                                )
                                batch_qa_pairs.append(qa_pairs)
                            except Exception as e2:
                                print(f"  ⚠ Individual generation also failed: {e2}")
                                batch_qa_pairs.append(
                                    {
                                        "qa_positive": None,
                                        "qa_physics": None,
                                        "qa_negative": None,
                                    }
                                )
                
                    # 组织数据并添加到结果中（Stage2 风格扁平结构）
                    for i, qa_pairs in enumerate(batch_qa_pairs):
                        scene_id_item = batch_scene_ids[i]
                        rgb_path_item = batch_rgb_paths[i]
                        gt_image_path_item = batch_gt_image_paths[i]
                        records = build_stage2_style_records(
                            rgb_path=rgb_path_item,
                            gt_image_path=gt_image_path_item,
                            scene_id=scene_id_item,
                            qa_pairs=qa_pairs,
                        )
                        results.extend(records)
                
                        processed_images += len(batch_qa_pairs)

                    # 清空批次
                    batch_rgb_images.clear()
                    batch_gt_images.clear()
                    batch_scene_ids.clear()
                    batch_rgb_filenames.clear()
                    batch_rgb_paths.clear()
                    batch_gt_image_paths.clear()

                    # ✅ 周期性保存（按“处理的图片数”计）
                    if processed_images >= next_save_images:
                        _save_results_snapshot()
                        while processed_images >= next_save_images:
                            next_save_images += save_every_images
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user, saving partial results...")
        if results:
            _save_results_snapshot()
        raise
    
    # 保存 JSON
    _save_results_snapshot()
    
    print("\n================ Generation Complete ================")
    print(f"Successfully generated {len(results)} Stage 3 QA records")
    print(f"Output file: {output_file.resolve()}")
    
    # 统计信息
    total_processed = len(rgb_items)
    with_content = sum(1 for r in results if r.get("subtype") == "positive")
    with_detail = sum(1 for r in results if r.get("subtype") == "physics")
    with_spatial = sum(1 for r in results if r.get("subtype") == "negative")
    
    print(f"\nStatistics:")
    print(f"  - Total images processed: {total_processed}")
    print(f"  - Total generated records: {len(results)}")
    print(f"  - Positive records: {with_content}")
    print(f"  - Physics records: {with_detail}")
    print(f"  - Negative records: {with_spatial}")


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
        help="Output JSON file path (default: auto-generated with timestamp)",
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
        help=(
            "Batch size for processing "
            f"(default: {BATCH_SIZE}, increase for faster processing if GPU memory allows, "
            "decrease if OOM occurs)"
        ),
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

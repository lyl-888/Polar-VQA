"""
Streamlit 可视化验证界面：PolarVLM Stage 3

功能概览：
- 统一加载 Stage 3 训练好的模型（双流：RGB + Polar + VAE 编码器）
- 支持多种验证模式：
  1）单张图片（通过路径）验证
  2）上传 RGB + Polar 图片验证
  3）上传 RGB + Polar 后，在 RGB 上画红圈（circle），自动转换为 bbox_norm 进行验证
- 支持设置主要参数和模型路径（提供合理默认值）
- 调用 `inference.py` 中已经实现的核心函数：
  - load_trained_model
  - preprocess_images
  - generate_qa_questions
  - generate_response
"""

import os
import re
import tempfile
import hashlib
import random
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import streamlit as st
import torch
from PIL import Image

# 尝试导入 streamlit-drawable-canvas，如果失败则使用备用方案
HAS_CANVAS = False
try:
    from streamlit_drawable_canvas import st_canvas
    # 测试是否可以正常使用（检查是否有 image_to_url 属性错误）
    HAS_CANVAS = True
except (ImportError, AttributeError):
    HAS_CANVAS = False
    # 不显示警告，因为会在使用时显示

from inference import (
    load_trained_model,
    preprocess_images,
    generate_qa_questions,
    generate_response,
    format_bbox_string,
)

# ======================= 基本配置 ======================= #
DEFAULT_STAGE3_CKPT = "/openbayes/input/input0/checkpoints/polarvlm_stage3_vae"
DEFAULT_LLM = "/openbayes/input/input0/models/Meta-Llama-3-8B-Instruct"
DEFAULT_CLIP = "/openbayes/input/input0/models/clip-vit-large-patch14"
DEFAULT_VAE = "/openbayes/input/input0/models/sd-vae-ft-mse"
DEFAULT_POLAR_ROOT = "/openbayes/input/input0/polar"

# 百度翻译 API 配置
BAIDU_TRANSLATE_APPID = "20260110002536869"
BAIDU_TRANSLATE_SECRET_KEY = "X_ZctW2qIGk4d2JY1I1g"


# ======================= 工具函数 ======================= #
def translate_baidu(text: str, appid: str = BAIDU_TRANSLATE_APPID, secret_key: str = BAIDU_TRANSLATE_SECRET_KEY, from_lang: str = "en", to_lang: str = "zh") -> str:
    """
    使用百度翻译 API 翻译文本（简化版，用于 Streamlit 界面）
    
    Args:
        text: 待翻译的文本
        appid: 百度翻译 API AppID
        secret_key: 百度翻译 API Secret Key
        from_lang: 源语言（默认：en 英文）
        to_lang: 目标语言（默认：zh 中文）
    
    Returns:
        翻译后的文本，如果翻译失败则返回原文
    """
    if not text or not text.strip():
        return text
    
    # 如果文本已经是中文，直接返回
    if to_lang == "zh" and any('\u4e00' <= char <= '\u9fff' for char in text):
        return text
    
    # 如果文本已经是英文且目标语言是英文，直接返回
    if to_lang == "en" and not any('\u4e00' <= char <= '\u9fff' for char in text):
        return text
    
    try:
        # 百度翻译 API 端点
        url = "https://fanyi-api.baidu.com/api/trans/vip/translate"
        
        # 生成签名
        salt = str(random.randint(32768, 65536))
        sign_str = appid + text + salt + secret_key
        sign = hashlib.md5(sign_str.encode('utf-8')).hexdigest()
        
        # 构建请求参数
        params = {
            "q": text,
            "from": from_lang,
            "to": to_lang,
            "appid": appid,
            "salt": salt,
            "sign": sign
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        # 检查错误
        if "error_code" in result:
            error_code = result.get("error_code")
            error_msg = result.get("error_msg", "Unknown error")
            print(f"Warning: 百度翻译 API 错误 {error_code}: {error_msg}")
            return text  # 翻译失败，返回原文
        
        # 提取翻译结果（合并所有分段，保持格式）
        if "trans_result" in result and len(result["trans_result"]) > 0:
            # 百度翻译 API 可能会将文本分段处理（特别是包含换行符时）
            # 需要合并所有翻译结果
            translated_parts = []
            for trans_item in result["trans_result"]:
                dst = trans_item.get("dst", "")
                if dst:
                    translated_parts.append(dst)
            
            if translated_parts:
                # 合并所有翻译结果，保持原始文本的换行符等格式
                # 策略：如果原始文本包含换行符，尝试在合并时保持换行符
                # 如果原始文本包含 "Layer 1:" 和 "Layer 2:"，确保它们之间有换行
                if "\n" in text or ("Layer 1:" in text and "Layer 2:" in text):
                    # 检查翻译结果中是否包含 Layer 标记
                    has_layer1 = any("Layer 1" in part or "层 1" in part or "第一层" in part for part in translated_parts)
                    has_layer2 = any("Layer 2" in part or "层 2" in part or "第二层" in part for part in translated_parts)
                    
                    # 合并翻译结果
                    if has_layer1 or has_layer2:
                        # 如果有 Layer 标记，尝试用换行符连接，但先检查每个部分
                        translated_text = ""
                        for i, part in enumerate(translated_parts):
                            if i > 0:
                                # 检查前一个部分是否以句号、问号、感叹号或冒号结尾
                                # 如果是，可能需要在后面添加换行符
                                prev_part = translated_parts[i-1]
                                if prev_part.rstrip().endswith(('.', '?', '!', ':', '。', '？', '！', '：')):
                                    translated_text += "\n"
                                else:
                                    translated_text += " "
                            translated_text += part
                        
                        # 确保 Layer 标记前有换行符（如果翻译结果中包含）
                        translated_text = translated_text.replace("Layer 1", "\nLayer 1")
                        translated_text = translated_text.replace("层 1", "\n层 1")
                        translated_text = translated_text.replace("第一层", "\n第一层")
                        translated_text = translated_text.replace("Layer 2", "\nLayer 2")
                        translated_text = translated_text.replace("层 2", "\n层 2")
                        translated_text = translated_text.replace("第二层", "\n第二层")
                        # 清理开头的换行符
                        translated_text = translated_text.lstrip("\n")
                    else:
                        # 没有 Layer 标记，用空格连接
                        translated_text = " ".join(translated_parts)
                else:
                    # 没有换行符，直接用空格连接
                    translated_text = " ".join(translated_parts)
                
                return translated_text
            else:
                return text
        else:
            return text
    
    except Exception as e:
        print(f"Warning: 翻译失败: {e}")
        return text  # 翻译失败，返回原文


@st.cache_resource(show_spinner=True)
def load_model_cached(
    checkpoint_dir: str,
    llm_model_name: str,
    clip_model_name: str,
    vae_model_path: str,
    device: str = "cuda",
):
    """带缓存的多模态 PolarVLM（Stage 3）模型加载，避免重复初始化。"""
    model, tokenizer, config = load_trained_model(
        checkpoint_dir=checkpoint_dir,
        llm_model_name=llm_model_name,
        clip_model_name=clip_model_name,
        polar_backbone="google/vit-base-patch16-224-in21k",
        stage1_checkpoint=None,
        stage2_checkpoint=None,
        vae_model_path=vae_model_path,
        hf_token=None,
        device=device,
    )
    return model, tokenizer, config




def save_uploaded_file_to_temp(uploaded_file, suffix: str = ".png") -> str:
    """将上传的文件保存到临时目录，并返回路径。"""
    if uploaded_file is None:
        return ""
    # 重置文件指针到开头
    uploaded_file.seek(0)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    # 验证文件是否成功保存
    if not Path(tmp.name).exists():
        raise IOError(f"无法保存临时文件: {tmp.name}")
    return tmp.name


def find_polar_images_from_rgb(rgb_file_path: str) -> Dict[str, Optional[BytesIO]]:
    """
    根据 RGB 图像路径，自动查找同文件夹下的对应偏振图像
    
    Args:
        rgb_file_path: RGB 图像路径（临时文件路径或原始路径）
    
    Returns:
        包含 4 个偏振图像文件内容的字典，如果找不到则对应值为 None
        {
            "I_0": BytesIO or None,
            "I_45": BytesIO or None,
            "I_90": BytesIO or None,
            "I_135": BytesIO or None
        }
    """
    rgb_path = Path(rgb_file_path)
    
    # 验证 RGB 文件是否存在
    if not rgb_path.exists():
        print(f"Warning: RGB 文件不存在: {rgb_path}")
        return {"I_0": None, "I_45": None, "I_90": None, "I_135": None}
    
    # 提取基础名称（如 "0012_rgb.png" -> "0012"）
    stem = rgb_path.stem  # "0012_rgb"
    if "_rgb" in stem:
        base_name = stem.replace("_rgb", "")
    else:
        # 如果没有 _rgb 后缀，尝试其他方式
        base_name = stem
    
    # 获取 RGB 图像所在目录
    rgb_dir = rgb_path.parent
    
    # 查找 4 张偏振图像
    polar_files = {}
    polar_suffixes = {
        "I_0": ["_000.png", "_0.png", "_000.jpg", "_0.jpg"],
        "I_45": ["_045.png", "_45.png", "_045.jpg", "_45.jpg"],
        "I_90": ["_090.png", "_90.png", "_090.jpg", "_90.jpg"],
        "I_135": ["_135.png", "_135.jpg"],
    }
    
    for polar_key, suffixes in polar_suffixes.items():
        found = False
        for suffix in suffixes:
            polar_path = rgb_dir / f"{base_name}{suffix}"
            if polar_path.exists():
                try:
                    with open(polar_path, "rb") as f:
                        content = f.read()
                        if len(content) > 0:  # 验证文件不为空
                            polar_files[polar_key] = BytesIO(content)
                            found = True
                            break
                        else:
                            print(f"Warning: 文件为空: {polar_path}")
                except Exception as e:
                    print(f"Warning: 无法读取文件 {polar_path}: {e}")
        
        if not found:
            polar_files[polar_key] = None
            # 调试信息：显示尝试查找的路径
            attempted_paths = [str(rgb_dir / f"{base_name}{s}") for s in suffixes]
            print(f"Debug: 未找到 {polar_key}，尝试的路径: {attempted_paths}")
    
    return polar_files


def circle_to_bbox_norm(
    center_x: float,
    center_y: float,
    radius: float,
    img_width: int,
    img_height: int,
) -> List[float]:
    """将画布上的圆（红圈）转换为归一化 bbox [xmin, ymin, xmax, ymax]。"""
    x1 = max(0.0, center_x - radius)
    y1 = max(0.0, center_y - radius)
    x2 = min(float(img_width), center_x + radius)
    y2 = min(float(img_height), center_y + radius)
    # 归一化到 [0, 1]
    return [
        round(x1 / img_width, 4),
        round(y1 / img_height, 4),
        round(x2 / img_width, 4),
        round(y2 / img_height, 4),
    ]


def render_bbox_from_canvas(canvas_result) -> Optional[List[float]]:
    """从 st_canvas 的结果中解析最后一个 circle，返回 bbox_norm。"""
    if not canvas_result or not canvas_result.json_data:
        return None

    objects = canvas_result.json_data.get("objects", [])
    if not objects:
        return None

    # 取最后一个对象作为当前选中区域
    obj = objects[-1]
    if obj.get("type") != "circle":
        return None

    # streamlit-drawable-canvas 中，circle 的 x,y 是左上角，radius 在 "radius"
    # 实际 center = (x + radius, y + radius)
    x = float(obj.get("left", 0.0))
    y = float(obj.get("top", 0.0))
    radius = float(obj.get("radius", 0.0))

    img_data = canvas_result.image_data
    if img_data is None:
        return None
    h, w = img_data.shape[0], img_data.shape[1]

    center_x = x + radius
    center_y = y + radius
    bbox_norm = circle_to_bbox_norm(center_x, center_y, radius, w, h)
    return bbox_norm


# ======================= 页面布局 ======================= #
st.set_page_config(
    page_title="PolarVLM Stage 3 可视化验证",
    layout="wide",
)

st.title("PolarVLM Stage 3 可视化验证界面")
st.markdown(
    "双流架构（RGB + Polar + VAE 编码器），用于交互式验证 Stage 3 训练效果。"
)


# ---------- 侧边栏：模型与参数配置 ---------- #
st.sidebar.header("模型与参数配置")

checkpoint_dir = st.sidebar.text_input(
    "Stage 3 checkpoint 路径 (--checkpoint_dir)",
    value=DEFAULT_STAGE3_CKPT,
)
llm_model_name = st.sidebar.text_input(
    "LLM 模型路径 (--llm_model_name)",
    value=DEFAULT_LLM,
)
clip_model_name = st.sidebar.text_input(
    "CLIP 模型路径 (--clip_model_name)",
    value=DEFAULT_CLIP,
)
vae_model_path = st.sidebar.text_input(
    "VAE 模型路径 (--vae_model_path)",
    value=DEFAULT_VAE,
)
device = st.sidebar.selectbox("设备 (--device)", ["cuda", "cpu"], index=0)

max_new_tokens = st.sidebar.slider(
    "max_new_tokens", min_value=32, max_value=512, value=256, step=32
)
temperature = st.sidebar.slider(
    "temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.05
)
top_p = st.sidebar.slider(
    "top_p", min_value=0.1, max_value=1.0, value=0.9, step=0.05
)
do_sample = st.sidebar.checkbox("do_sample", value=True)

with st.sidebar:
    if st.button("加载 / 重新加载模型", type="primary"):
        # 清掉缓存强制重载
        load_model_cached.clear()

model, tokenizer, _ = load_model_cached(
    checkpoint_dir=checkpoint_dir,
    llm_model_name=llm_model_name,
    clip_model_name=clip_model_name,
    vae_model_path=vae_model_path,
    device=device,
)


# ---------- 主区域：模式选择 ---------- #
mode = st.selectbox(
    "选择验证模式",
    [
        "上传图片并画红圈（推荐）",
        "单张图片（本地路径）",
    ],
)


# ======================= 模式 1：上传图片并画红圈 ======================= #
if mode == "上传图片并画红圈（推荐）":
    st.subheader("模式 1：上传 RGB + Polar 并在 RGB 上画红圈")

    col_rgb, col_polar = st.columns(2)

    with col_rgb:
        rgb_file = st.file_uploader("上传 RGB 图像", type=["png", "jpg", "jpeg"])
        
        # 自动查找偏振图像功能
        if rgb_file is not None:
            # 初始化 session_state
            if "found_polar_paths" not in st.session_state:
                st.session_state.found_polar_paths = {}
            
            st.markdown("**自动查找偏振图像**")
            st.caption("💡 提示：如果上传的文件来自本地路径，请输入原始文件路径以便查找同文件夹下的偏振图像。")
            
            # 允许用户输入原始文件路径
            original_rgb_path = st.text_input(
                "原始 RGB 文件路径（可选，用于查找同文件夹下的偏振图像）",
                value="",
                key="original_rgb_path",
                help="例如：E:\\datasets\\Mingde--PolaRGB\\train\\hard\\input\\23\\0000_rgb.png"
            )
            
            if st.button("🔍 自动查找同文件夹下的偏振图像", key="auto_find_polar"):
                with st.spinner("正在查找偏振图像..."):
                    # 优先使用用户输入的原始路径，否则使用临时文件路径
                    if original_rgb_path and Path(original_rgb_path).exists():
                        search_path = original_rgb_path
                        st.info(f"使用原始路径查找: {original_rgb_path}")
                    else:
                        # 保存 RGB 文件到临时目录以便查找
                        rgb_temp_path = save_uploaded_file_to_temp(rgb_file, suffix=Path(rgb_file.name).suffix)
                        search_path = rgb_temp_path
                        if original_rgb_path:
                            st.warning(f"原始路径不存在: {original_rgb_path}，改用临时文件路径查找")
                    
                    found_files = find_polar_images_from_rgb(search_path)
                    
                    # 将找到的文件保存到临时目录
                    found_paths = {}
                    for key, file_content in found_files.items():
                        if file_content is not None:
                            # 保存 BytesIO 内容到临时文件
                            file_content.seek(0)
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                            tmp.write(file_content.read())
                            tmp.flush()
                            tmp.close()
                            found_paths[key] = tmp.name
                        else:
                            found_paths[key] = None
                    
                    st.session_state.found_polar_paths = found_paths
                    
                    # 统计找到的文件
                    found_count = sum(1 for v in found_paths.values() if v is not None)
                    if found_count == 4:
                        st.success(f"✓ 成功找到所有 4 张偏振图像！")
                    elif found_count > 0:
                        st.warning(f"⚠ 只找到 {found_count}/4 张偏振图像，请手动上传缺失的文件。")
                        # 显示未找到的文件
                        missing = [k for k, v in found_paths.items() if v is None]
                        st.caption(f"缺失的文件: {', '.join(missing)}")
                    else:
                        st.error("❌ 未找到任何偏振图像，请手动上传。")
                        st.caption("💡 提示：请检查原始文件路径是否正确，或确认偏振图像文件是否存在。")
            
            # 显示找到的文件信息
            if "found_polar_paths" in st.session_state and st.session_state.found_polar_paths:
                found_count = sum(1 for v in st.session_state.found_polar_paths.values() if v is not None)
                if found_count > 0:
                    st.info(f"已找到 {found_count}/4 张偏振图像，将在下方自动使用。")
    
    with col_polar:
        st.markdown("上传对应的 4 张偏振图像（I_0, I_45, I_90, I_135）")
        
        # 检查是否找到了文件
        found_paths = st.session_state.get("found_polar_paths", {})
        
        # 显示找到的文件信息
        if found_paths:
            found_count = sum(1 for v in found_paths.values() if v is not None)
            if found_count > 0:
                st.success(f"✓ 已自动找到 {found_count}/4 张偏振图像")
        
        # 显示上传框，如果找到了文件则显示提示
        if found_paths.get("I_0"):
            st.caption("✓ I_0 (000) - 已自动找到")
        polar_0 = st.file_uploader("I_0 (000)", type=["png", "jpg", "jpeg"], key="polar0")
        
        if found_paths.get("I_45"):
            st.caption("✓ I_45 (045) - 已自动找到")
        polar_45 = st.file_uploader("I_45 (045)", type=["png", "jpg", "jpeg"], key="polar45")
        
        if found_paths.get("I_90"):
            st.caption("✓ I_90 (090) - 已自动找到")
        polar_90 = st.file_uploader("I_90 (090)", type=["png", "jpg", "jpeg"], key="polar90")
        
        if found_paths.get("I_135"):
            st.caption("✓ I_135 (135) - 已自动找到")
        polar_135 = st.file_uploader("I_135 (135)", type=["png", "jpg", "jpeg"], key="polar135")

    # 检查是否有足够的文件（RGB + 4 张偏振图像）
    # 偏振图像可以是找到的文件（在 found_polar_paths 中）或用户上传的文件
    found_paths = st.session_state.get("found_polar_paths", {})
    
    # 检查是否所有文件都已准备好
    has_all_polar = (
        (found_paths.get("I_0") or polar_0) and
        (found_paths.get("I_45") or polar_45) and
        (found_paths.get("I_90") or polar_90) and
        (found_paths.get("I_135") or polar_135)
    )
    
    if rgb_file and has_all_polar:
        # 保存到临时文件
        rgb_path = save_uploaded_file_to_temp(rgb_file)
        
        # 处理偏振图像：优先使用用户上传的文件，否则使用找到的文件
        def get_polar_path(polar_key: str, uploaded_file, found_path: Optional[str] = None):
            """获取偏振图像的临时文件路径，优先使用用户上传的文件"""
            if uploaded_file:
                # 用户上传的文件优先
                return save_uploaded_file_to_temp(uploaded_file)
            elif found_path and Path(found_path).exists():
                # 使用找到的文件路径（已经是临时文件）
                return found_path
            else:
                return None
        
        polar_paths: Dict[str, str] = {
            "I_0": get_polar_path("I_0", polar_0, found_paths.get("I_0")),
            "I_45": get_polar_path("I_45", polar_45, found_paths.get("I_45")),
            "I_90": get_polar_path("I_90", polar_90, found_paths.get("I_90")),
            "I_135": get_polar_path("I_135", polar_135, found_paths.get("I_135")),
        }
        
        # 确保所有路径都存在
        if not all(p and Path(p).exists() for p in polar_paths.values()):
            st.error("❌ 部分偏振图像文件缺失，请检查文件路径或重新上传。")
            st.stop()

        # 验证 RGB 文件是否存在且可读
        if not Path(rgb_path).exists():
            st.error(f"❌ RGB 图像文件不存在: {rgb_path}")
            st.stop()
        
        try:
            rgb_pil = Image.open(rgb_path).convert("RGB")
            # 验证图像是否有效
            rgb_pil.verify()
            rgb_pil = Image.open(rgb_path).convert("RGB")  # 重新打开，因为 verify() 会关闭文件
        except Exception as e:
            st.error(f"❌ 无法读取 RGB 图像文件: {e}\n文件路径: {rgb_path}")
            st.stop()
        w, h = rgb_pil.size

        st.markdown("### 在 RGB 图像上选择区域")
        
        # 初始化 canvas_result（用于后续检查）
        canvas_result = None
        
        # 使用 canvas 画框（如果可用），否则使用 slider
        canvas_available = HAS_CANVAS
        bbox_norm = None  # 初始化 bbox_norm
        
        if HAS_CANVAS:
            try:
                st.markdown("**💡 提示：在图像上直接画矩形框选择区域（推荐）**")
                st.caption("使用鼠标在图像上拖拽绘制矩形框，系统会自动识别并转换为归一化坐标。")
                
                # 使用 canvas 画框（捕获可能的兼容性错误）
                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",  # 红色半透明填充
                    stroke_width=2,
                    stroke_color="red",
                    background_image=rgb_pil,
                    height=h,
                    width=w,
                    drawing_mode="rect",  # 矩形模式
                    point_display_radius=0,
                    key="canvas",
                )
                
                # 从 canvas 结果中解析 bbox
                if canvas_result and canvas_result.json_data:
                    objects = canvas_result.json_data.get("objects", [])
                    if objects:
                        # 取最后一个矩形
                        rects = [obj for obj in objects if obj.get("type") == "rect"]
                        if rects:
                            rect = rects[-1]
                            left = float(rect.get("left", 0))
                            top = float(rect.get("top", 0))
                            width = float(rect.get("width", 0))
                            height = float(rect.get("height", 0))
                            
                            # 确保宽度和高度大于 0
                            if width > 0 and height > 0:
                                # 转换为归一化坐标 [xmin, ymin, xmax, ymax]
                                x_min = max(0.0, left / w)
                                y_min = max(0.0, top / h)
                                x_max = min(1.0, (left + width) / w)
                                y_max = min(1.0, (top + height) / h)
                                bbox_norm = [round(x_min, 4), round(y_min, 4), round(x_max, 4), round(y_max, 4)]
                
                # 如果 canvas 没有选择区域，显示提示
                if bbox_norm is None:
                    st.warning("⚠️ 请先在图像上拖拽绘制矩形框选择区域，然后再点击「运行验证」按钮。")
                    # 使用默认值作为占位符（但会在验证时提示）
                    bbox_norm = [0.1, 0.1, 0.5, 0.5]
                else:
                    st.success(f"✓ 已选择区域: {bbox_norm}")
            except (AttributeError, Exception) as e:
                # streamlit-drawable-canvas 兼容性问题，回退到滑块
                canvas_available = False
                bbox_norm = None  # 重置，让滑块模式重新定义
                st.warning(f"⚠️ streamlit-drawable-canvas 存在兼容性问题，已自动切换到滑块模式。")
                st.caption(f"错误详情: {str(e)}")
        
        if not canvas_available:
            # 备用方案：使用 slider
            st.markdown("**使用滑块选择区域（备用方案）**")
            st.caption("💡 提示：如果安装了 streamlit-drawable-canvas，可以直接在图像上画框。")
            col1, col2 = st.columns(2)
            with col1:
                x1 = st.slider("xmin (归一化)", 0.0, 0.95, 0.1, 0.01)
                x2 = st.slider("xmax (归一化)", 0.05, 1.0, 0.5, 0.01)
            with col2:
                y1 = st.slider("ymin (归一化)", 0.0, 0.95, 0.1, 0.01)
                y2 = st.slider("ymax (归一化)", 0.05, 1.0, 0.5, 0.01)

            # 保证坐标有序并计算归一化 bbox
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            bbox_norm = [round(x_min, 4), round(y_min, 4), round(x_max, 4), round(y_max, 4)]

            # 在图像上画出红框预览（像素坐标也确保单调）
            from PIL import ImageDraw

            preview = rgb_pil.copy()
            draw = ImageDraw.Draw(preview)
            x0_px = int(x_min * w)
            x1_px = int(x_max * w)
            y0_px = int(y_min * h)
            y1_px = int(y_max * h)
            draw.rectangle([x0_px, y0_px, x1_px, y1_px], outline="red", width=3)
            st.image(preview, caption=f"选定区域: {bbox_norm}", use_container_width=True)

        st.markdown("---")

        # 问题设置
        st.markdown("### 问题设置")
        qa_mode = st.radio(
            "选择提问方式",
            ["使用预设模板问题（多种类型）", "自定义问题"],
            index=0,
        )

        template_options = {
            "content": "描述该区域的可见内容（content）",
            "detail": "描述该区域物体的颜色和材质（detail）",
            "behind": "透视分析：忽略表面反射，透过眩光识别背后的物理结构（behind）",
            "contour": "描述被反光遮挡物体的轮廓和物理特征（contour）",
            "layer": "分析视觉层次：区分反射/眩光（前景）与真实物体（背景）（layer）",
            "layer_analysis": "分层分析：Layer 1 反射场景 + Layer 2 实际物体（layer_analysis）",
        }

        if qa_mode == "使用预设模板问题（多种类型）":
            qa_types = st.multiselect(
                "选择要提问的模板类型",
                options=list(template_options.keys()),
                default=["content", "detail"],
            )
            custom_question = None

            # 实时预览当前区域下各模板对应的完整英文问题（自动带入坐标）
            if qa_types:
                current_questions = generate_qa_questions(bbox_norm)
                st.markdown("**当前选定区域下，各模板对应的实际问题：**")
                for t in qa_types:
                    if t in current_questions:
                        st.markdown(f"- **{template_options[t]}**：`{current_questions[t]}`")
        else:
            qa_types = ["custom"]
            custom_question = st.text_area(
                "自定义问题（英文，建议包含 `Focus on region [...]` 描述）",
                value="Focus on region [0.1, 0.1, 0.5, 0.5]. Please briefly describe the visual content within this region.",
                height=100,
            )

        enable_baseline = st.checkbox(
            "同时使用 RGB-only 基线（无偏振信号）生成对比答案", value=True
        )
        st.caption("💡 RGB-only 基线：使用相同的 PolarVLM 模型，但将偏振输入设为全零，用于公平对比偏振信息的作用。")
        
        enable_translation = st.checkbox(
            "自动将英文回答翻译为中文", value=False
        )
        st.caption("🌐 使用百度翻译 API 自动翻译回答（需要网络连接）。")

        if st.button("运行验证", type="primary"):
            # 检查是否选择了有效区域（仅对 canvas 模式）
            if HAS_CANVAS and bbox_norm == [0.1, 0.1, 0.5, 0.5] and qa_mode == "使用预设模板问题（多种类型）":
                st.warning("⚠️ 请先在图像上拖拽绘制矩形框选择区域，然后再点击「运行验证」按钮。")
                st.stop()
            else:
                # 预处理图像
                with st.spinner("正在预处理图像并生成回答..."):
                    # 验证 RGB 图像尺寸（预处理前）
                    rgb_pil_check = Image.open(rgb_path).convert("RGB")
                    original_size = rgb_pil_check.size
                    st.info(f"📊 RGB 图像原始尺寸: {original_size[0]}x{original_size[1]} (预处理后会 resize 到 224x224)")
                    
                    pixel_values_rgb, pixel_values_polar = preprocess_images(
                        rgb_path=rgb_path,
                        polar_paths={k: Path(v) for k, v in polar_paths.items()},
                        clip_model_name=clip_model_name,
                        device=device,
                    )
                    
                    # 验证预处理后的尺寸
                    rgb_shape = pixel_values_rgb.shape
                    polar_shape = pixel_values_polar.shape
                    st.success(f"✓ RGB 特征尺寸: {rgb_shape} (应为 (1, 3, 224, 224))")
                    st.success(f"✓ Polar 特征尺寸: {polar_shape} (应为 (1, 3, 512, 512))")
                    
                    # 验证偏振图像文件名匹配
                    polar_paths_info = []
                    for key, path in polar_paths.items():
                        if path and Path(path).exists():
                            polar_paths_info.append(f"✓ {key}: {Path(path).name}")
                        else:
                            polar_paths_info.append(f"❌ {key}: 文件不存在")
                    st.info("📁 偏振图像文件匹配情况:\n" + "\n".join(polar_paths_info))

                    results = []

                    # RGB-only 基线：使用相同的模型，但将 Polar 输入设为全零（模拟无偏振信号）
                    base_results = []

                    if qa_mode == "使用预设模板问题（多种类型）":
                        # 使用模板生成问题（根据当前区域生成完整问题字典）
                        questions = generate_qa_questions(bbox_norm)
                        for qa_type in qa_types:
                            # 保险：如果模板字典中没有该类型，直接跳过，避免 KeyError
                            if qa_type not in questions:
                                continue
                            question = questions[qa_type]
                            # 为 layer_analysis 类型的回答使用更大的 max_new_tokens（需要描述两层内容）
                            effective_max_tokens = max_new_tokens * 2 if qa_type == "layer_analysis" else max_new_tokens
                            answer = generate_response(
                                model=model,
                                tokenizer=tokenizer,
                                pixel_values_rgb=pixel_values_rgb,
                                pixel_values_polar=pixel_values_polar,
                                question=question,
                                max_new_tokens=effective_max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                do_sample=do_sample,
                            )
                            results.append((qa_type, question, answer))

                            if enable_baseline:
                                # RGB-only 基线：使用相同的模型和 RGB 输入，但将 Polar 输入设为全零
                                # 这样可以公平对比"有偏振 vs 无偏振"的差异
                                pixel_values_polar_zero = torch.zeros_like(pixel_values_polar)
                                
                                # 为 layer_analysis 类型的回答使用更大的 max_new_tokens
                                effective_max_tokens_baseline = max_new_tokens * 2 if qa_type == "layer_analysis" else max_new_tokens
                                base_answer = generate_response(
                                    model=model,
                                    tokenizer=tokenizer,
                                    pixel_values_rgb=pixel_values_rgb,
                                    pixel_values_polar=pixel_values_polar_zero,  # 全零，模拟无偏振信号
                                    question=question,
                                    max_new_tokens=effective_max_tokens_baseline,
                                    temperature=temperature,
                                    top_p=top_p,
                                    do_sample=do_sample,
                                )
                                
                                base_results.append((qa_type, question, base_answer))
                    else:
                        # 自定义问题
                        if not custom_question:
                            st.error("请填写自定义问题。")
                        else:
                            question = custom_question
                            answer = generate_response(
                                model=model,
                                tokenizer=tokenizer,
                                pixel_values_rgb=pixel_values_rgb,
                                pixel_values_polar=pixel_values_polar,
                                question=question,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                do_sample=do_sample,
                            )
                            results.append(("custom", question, answer))
                            
                            if enable_baseline:
                                # RGB-only 基线：使用相同的模型和 RGB 输入，但将 Polar 输入设为全零
                                # 这样可以公平对比"有偏振 vs 无偏振"的差异
                                pixel_values_polar_zero = torch.zeros_like(pixel_values_polar)
                                
                                base_answer = generate_response(
                                    model=model,
                                    tokenizer=tokenizer,
                                    pixel_values_rgb=pixel_values_rgb,
                                    pixel_values_polar=pixel_values_polar_zero,  # 全零，模拟无偏振信号
                                    question=question,
                                    max_new_tokens=max_new_tokens,
                                    temperature=temperature,
                                    top_p=top_p,
                                    do_sample=do_sample,
                                )
                                
                                base_results.append(("custom", question, base_answer))

                # 展示结果
                st.markdown("### 结果")
                if bbox_norm is not None:
                    st.write(f"选择的归一化 bbox: `{bbox_norm}`")
                    st.write(
                        f"bbox 字符串: `{format_bbox_string(bbox_norm)}`"
                    )

                for qa_type, question, answer in results:
                    st.markdown(f"**类型：{qa_type.upper()}**")
                    st.markdown(f"**问题：** {question}")
                    st.markdown(f"**PolarVLM (RGB+Polar) 回答：** {answer}")
                    
                    # 如果启用了翻译，显示中文翻译
                    if enable_translation:
                        with st.spinner("正在翻译为中文..."):
                            answer_zh = translate_baidu(answer, from_lang="en", to_lang="zh")
                            st.markdown(f"**PolarVLM (RGB+Polar) 回答（中文）：** {answer_zh}")
                            time.sleep(0.5)  # 避免 API 频率限制

                    if enable_baseline:
                        # 找到对应的基线答案
                        base_match = next(
                            (b for b in base_results if b[0] == qa_type and b[1] == question),
                            None,
                        )
                        if base_match is not None:
                            base_answer = base_match[2]
                            st.markdown(f"**RGB-only 基线（无偏振信号）回答：** {base_answer}")
                            
                            # 如果启用了翻译，显示基线答案的中文翻译
                            if enable_translation:
                                with st.spinner("正在翻译基线答案为中文..."):
                                    base_answer_zh = translate_baidu(base_answer, from_lang="en", to_lang="zh")
                                    st.markdown(f"**RGB-only 基线（无偏振信号）回答（中文）：** {base_answer_zh}")
                                    time.sleep(0.5)  # 避免 API 频率限制
                            
                            st.caption("💡 对比说明：PolarVLM 使用了 RGB + Polar 双流输入，而 RGB-only 基线只使用 RGB（Polar 输入为全零），两者的差异体现了偏振信息的作用。")
                    st.markdown("---")


# ======================= 模式 2：单张图片（路径） ======================= #
elif mode == "单张图片（本地路径）":
    st.subheader("模式 2：单张图片（本地路径）")

    rgb_path_str = st.text_input(
        "RGB 图像路径（绝对路径或相对当前工作目录）",
        value="/openbayes/input/input0/rgb/30/0020_rgb.png",
    )
    st.markdown("偏振图像路径（4 个角度）")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        polar_0 = st.text_input(
            "I_0", value="/openbayes/input/input0/polar/30/0020_000.png"
        )
    with col2:
        polar_45 = st.text_input(
            "I_45", value="/openbayes/input/input0/polar/30/0020_045.png"
        )
    with col3:
        polar_90 = st.text_input(
            "I_90", value="/openbayes/input/input0/polar/30/0020_090.png"
        )
    with col4:
        polar_135 = st.text_input(
            "I_135", value="/openbayes/input/input0/polar/30/0020_135.png"
        )

    use_template = st.checkbox(
        "使用模板问题（根据 bbox_norm 自动生成 content/detail）", value=True
    )
    
    enable_baseline_mode2 = st.checkbox(
        "同时使用 RGB-only 基线（无偏振信号）生成对比答案", value=True, key="baseline_mode2"
    )
    st.caption("💡 RGB-only 基线：使用相同的 PolarVLM 模型，但将偏振输入设为全零，用于公平对比偏振信息的作用。")
    
    enable_translation_mode2 = st.checkbox(
        "自动将英文回答翻译为中文", value=False, key="translation_mode2"
    )
    st.caption("🌐 使用百度翻译 API 自动翻译回答（需要网络连接）。")

    if use_template:
        bbox_norm_default = [0.15, 0.05, 0.55, 0.45]
        bbox_vals = st.text_input(
            "bbox_norm（格式：[x1, y1, x2, y2]）",
            value=str(bbox_norm_default),
        )
        try:
            bbox_norm = [float(x) for x in bbox_vals.strip("[] ").split(",")]
            assert len(bbox_norm) == 4
        except Exception:
            st.error("bbox_norm 格式错误，请输入类似 `[0.15, 0.05, 0.55, 0.45]` 的四个浮点数。")
            bbox_norm = None
        qa_types = ["content", "detail"]
        custom_question = None
    else:
        bbox_norm = None
        qa_types = ["custom"]
        custom_question = st.text_area(
            "自定义问题",
            value="Focus on region [0.15, 0.05, 0.55, 0.45]. Please briefly describe the visual content within this region.",
            height=100,
        )

    if st.button("运行验证（单张图片）", type="primary"):
        rgb_path = Path(rgb_path_str)
        polar_paths = {
            "I_0": Path(polar_0),
            "I_45": Path(polar_45),
            "I_90": Path(polar_90),
            "I_135": Path(polar_135),
        }
        missing = [str(p) for p in [rgb_path, *polar_paths.values()] if not p.exists()]
        if missing:
            st.error(f"以下路径不存在，请检查：\n" + "\n".join(missing))
        else:
            with st.spinner("正在预处理图像并生成回答..."):
                # 验证 RGB 图像尺寸（预处理前）
                rgb_pil_check = Image.open(rgb_path).convert("RGB")
                original_size = rgb_pil_check.size
                st.info(f"📊 RGB 图像原始尺寸: {original_size[0]}x{original_size[1]} (预处理后会 resize 到 224x224)")
                
                pixel_values_rgb, pixel_values_polar = preprocess_images(
                    rgb_path=str(rgb_path),
                    polar_paths=polar_paths,
                    clip_model_name=clip_model_name,
                    device=device,
                )
                
                # 验证预处理后的尺寸
                rgb_shape = pixel_values_rgb.shape
                polar_shape = pixel_values_polar.shape
                st.success(f"✓ RGB 特征尺寸: {rgb_shape} (应为 (1, 3, 224, 224))")
                st.success(f"✓ Polar 特征尺寸: {polar_shape} (应为 (1, 3, 512, 512))")
                
                # 验证偏振图像文件名匹配
                polar_paths_info = []
                for key, path in polar_paths.items():
                    if path and path.exists():
                        polar_paths_info.append(f"✓ {key}: {path.name}")
                    else:
                        polar_paths_info.append(f"❌ {key}: 文件不存在")
                st.info("📁 偏振图像文件匹配情况:\n" + "\n".join(polar_paths_info))

                results = []
                base_results = []  # RGB-only 基线结果
                
                if use_template and bbox_norm is not None:
                    questions = generate_qa_questions(bbox_norm)
                    for qa_type in qa_types:
                        question = questions[qa_type]
                        # 为 layer_analysis 类型的回答使用更大的 max_new_tokens（需要描述两层内容）
                        effective_max_tokens = max_new_tokens * 2 if qa_type == "layer_analysis" else max_new_tokens
                        answer = generate_response(
                            model=model,
                            tokenizer=tokenizer,
                            pixel_values_rgb=pixel_values_rgb,
                            pixel_values_polar=pixel_values_polar,
                            question=question,
                            max_new_tokens=effective_max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=do_sample,
                        )
                        results.append((qa_type, question, answer))
                        
                        # RGB-only 基线：使用相同的模型和 RGB 输入，但将 Polar 输入设为全零
                        if enable_baseline_mode2:
                            pixel_values_polar_zero = torch.zeros_like(pixel_values_polar)
                            # 为 layer_analysis 类型的回答使用更大的 max_new_tokens
                            effective_max_tokens_baseline = max_new_tokens * 2 if qa_type == "layer_analysis" else max_new_tokens
                            base_answer = generate_response(
                                model=model,
                                tokenizer=tokenizer,
                                pixel_values_rgb=pixel_values_rgb,
                                pixel_values_polar=pixel_values_polar_zero,  # 全零，模拟无偏振信号
                                question=question,
                                max_new_tokens=effective_max_tokens_baseline,
                                temperature=temperature,
                                top_p=top_p,
                                do_sample=do_sample,
                            )
                            base_results.append((qa_type, question, base_answer))
                else:
                    if not custom_question:
                        st.error("请填写自定义问题。")
                    else:
                        question = custom_question
                        answer = generate_response(
                            model=model,
                            tokenizer=tokenizer,
                            pixel_values_rgb=pixel_values_rgb,
                            pixel_values_polar=pixel_values_polar,
                            question=question,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=do_sample,
                        )
                        results.append(("custom", question, answer))
                        
                        # RGB-only 基线：使用相同的模型和 RGB 输入，但将 Polar 输入设为全零
                        if enable_baseline_mode2:
                            pixel_values_polar_zero = torch.zeros_like(pixel_values_polar)
                            base_answer = generate_response(
                                model=model,
                                tokenizer=tokenizer,
                                pixel_values_rgb=pixel_values_rgb,
                                pixel_values_polar=pixel_values_polar_zero,  # 全零，模拟无偏振信号
                                question=question,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                do_sample=do_sample,
                            )
                            base_results.append(("custom", question, base_answer))

            st.markdown("### 结果")
            if bbox_norm is not None:
                st.write(f"bbox_norm: `{bbox_norm}`")
                st.write(f"bbox 字符串: `{format_bbox_string(bbox_norm)}`")

            for qa_type, question, answer in results:
                st.markdown(f"**类型：{qa_type.upper()}**")
                st.markdown(f"**问题：** {question}")
                st.markdown(f"**PolarVLM (RGB+Polar) 回答：** {answer}")
                
                # 如果启用了翻译，显示中文翻译
                if enable_translation_mode2:
                    with st.spinner("正在翻译为中文..."):
                        answer_zh = translate_baidu(answer, from_lang="en", to_lang="zh")
                        st.markdown(f"**PolarVLM (RGB+Polar) 回答（中文）：** {answer_zh}")
                        time.sleep(0.5)  # 避免 API 频率限制
                
                # 如果启用了基线对比，显示基线答案
                if enable_baseline_mode2:
                    base_match = next(
                        (b for b in base_results if b[0] == qa_type and b[1] == question),
                        None,
                    )
                    if base_match is not None:
                        base_answer = base_match[2]
                        st.markdown(f"**RGB-only 基线（无偏振信号）回答：** {base_answer}")
                        
                        # 如果启用了翻译，显示基线答案的中文翻译
                        if enable_translation_mode2:
                            with st.spinner("正在翻译基线答案为中文..."):
                                base_answer_zh = translate_baidu(base_answer, from_lang="en", to_lang="zh")
                                st.markdown(f"**RGB-only 基线（无偏振信号）回答（中文）：** {base_answer_zh}")
                                time.sleep(0.5)  # 避免 API 频率限制
                        
                        st.caption("💡 对比说明：PolarVLM 使用了 RGB + Polar 双流输入，而 RGB-only 基线只使用 RGB（Polar 输入为全零），两者的差异体现了偏振信息的作用。")
                
                st.markdown("---")


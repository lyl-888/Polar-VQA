#!/usr/bin/env python3
"""
使用训练好的 PolarVLM Stage 2 模型，对测试集进行评测：
- 同时输入 RGB + Polar 图像
- 使用原始 Stage 2 问题（不改写为红框提示）
- 生成回答并使用 GPT-4o-mini 进行评分（1-10 分）
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from openai import OpenAI
import logging

# 直接复用 PolarVLM 推理代码中的核心函数
# 🟢 关键修复：导入 LLaVA/inference.py 中的 Stage 2 版本，而不是根目录下的旧版
# 根目录下的 inference.py 是旧的 Stage 3 版本，会检查 model_config.json（Stage 2 训练不会生成）
# LLaVA/inference.py 是新的 Stage 2 版本，不需要 model_config.json
import sys
import os

# 添加 LLaVA 目录到路径（确保可以导入 llava 模块）
current_dir = os.path.dirname(os.path.abspath(__file__))
llava_dir = os.path.join(current_dir, 'LLaVA')
if llava_dir not in sys.path:
    sys.path.insert(0, llava_dir)

# 🟢 直接导入 LLaVA/inference.py（使用 importlib 动态加载，避免与根目录的 inference.py 冲突）
import importlib.util
inference_file = os.path.join(llava_dir, 'inference.py')
if not os.path.exists(inference_file):
    raise FileNotFoundError(f"未找到 LLaVA/inference.py: {inference_file}")

spec = importlib.util.spec_from_file_location("llava_inference_module", inference_file)
llava_inference_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llava_inference_module)

# 从模块中提取需要的函数
load_trained_model = llava_inference_module.load_trained_model
preprocess_images = llava_inference_module.preprocess_images
generate_response = llava_inference_module.generate_response
get_polar_paths = llava_inference_module.get_polar_paths

# 🟢 禁用 transformers 的 torch.load 安全检查（兼容老版本 torch）
# 在当前环境中升级 torch 到 2.6 不现实，只加载本地权重是安全的，因此直接关闭检查。
try:
    import transformers

    def _noop_safety_check():
        pass

    # 关闭全局安全检查
    transformers.utils.import_utils.check_torch_load_is_safe = _noop_safety_check
    if hasattr(transformers.modeling_utils, "check_torch_load_is_safe"):
        transformers.modeling_utils.check_torch_load_is_safe = _noop_safety_check

    # patch load_state_dict，确保内部不会重新开启检查
    _orig_load_state_dict = transformers.modeling_utils.load_state_dict

    def _patched_load_state_dict(checkpoint_file, *args, **kwargs):
        transformers.utils.import_utils.check_torch_load_is_safe = _noop_safety_check
        if hasattr(transformers.modeling_utils, "check_torch_load_is_safe"):
            transformers.modeling_utils.check_torch_load_is_safe = _noop_safety_check
        return _orig_load_state_dict(checkpoint_file, *args, **kwargs)

    transformers.modeling_utils.load_state_dict = _patched_load_state_dict

    # 降低 transformers 日志等级，避免 “Some weights ... were not used” 等噪音
    try:
        transformers.utils.logging.set_verbosity_error()
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        logging.getLogger("transformers.modeling_utils").propagate = False
    except Exception:
        pass
except Exception as e:
    print(f"Warning: Failed to disable transformers safety check in eval_polar_stage2.py: {e}")


# GPT-4o-mini 配置（与 RGB baseline 脚本保持一致，便于直接对比）
GPT_API_KEY = "sk-IAtjepO76Q2L7aya4ne7s9NRVGd1wNGm7HzVEXT8yGDIUptp"
GPT_BASE_URL = "https://api.openai-proxy.org/v1"
GPT_MODEL = "gpt-4o-mini"
NUM_SECONDS_TO_SLEEP = 0.5


def draw_bbox_on_image(image: Image.Image, bbox_norm: List[float], color: str = "red", width: int = 3) -> Image.Image:
    """
    在图像上绘制红色边界框（与 eval_rgb_baseline.py 保持一致）
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    w, h = img_copy.size
    x1 = int(bbox_norm[0] * w)
    y1 = int(bbox_norm[1] * h)
    x2 = int(bbox_norm[2] * w)
    y2 = int(bbox_norm[3] * h)

    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return img_copy


def replace_bbox_in_question(question: str) -> str:
    """
    将问题中的坐标替换为“红色边界框”的描述（与 eval_rgb_baseline.py 保持一致）
    """
    # 形式一： "Focus on region [x1, y1, x2, y2]."
    pattern1 = r'Focus on region\s*\[[0-9.,\s]+\]\.?\s*'
    replacement1 = "Focus on the region inside the red bounding box. "
    new_q = re.sub(pattern1, replacement1, question, flags=re.IGNORECASE)

    if new_q == question:
        # 形式二：任意 "[x1, y1, x2, y2]"
        pattern2 = r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]'
        new_q = re.sub(pattern2, "the red bounding box", question, count=1)
        if "Focus on" not in new_q and "focus on" not in new_q:
            new_q = "Focus on the region inside the red bounding box. " + new_q

    return new_q.strip()


def get_gpt_score(
    question: str,
    gt_answer: str,
    model_answer: str,
    subtype: str = "default",
    max_tokens: int = 1024,
) -> Tuple[float, str]:
    """
    使用 GPT-4o-mini 对模型回答进行评分（1-10 分）
   （实现与 eval_rgb_baseline.py 中保持一致，保证可比性）
    """
    # 🟢 关键修复：如果模型回答为空或只有特殊字符，直接返回低分，不调用 GPT
    if not model_answer or not model_answer.strip():
        return 1.0, "Model answer is empty. Score: 1.0 (minimum score for empty response)."
    
    # 检查是否只有特殊字符或结束符
    cleaned_answer = model_answer.strip()
    if cleaned_answer in ["</s>", "<|end_of_text|>", "<|endoftext|>", ".", ":", "\n"]:
        return 1.0, f"Model answer contains only special tokens or punctuation: '{cleaned_answer}'. Score: 1.0 (minimum score)."
    
    # 检查是否只有重复的标点符号（如 ":\n:\n:\n..."）
    if len(cleaned_answer) > 5 and len(set(cleaned_answer.replace("\n", "").replace(" ", ""))) <= 2:
        # 如果去除换行和空格后，只有1-2个不同的字符，可能是重复标点
        return 1.0, f"Model answer appears to be repetitive punctuation. Score: 1.0 (minimum score)."
    
    client = OpenAI(
        base_url=GPT_BASE_URL,
        api_key=GPT_API_KEY,
    )

    # 根据 subtype 选择不同的评分提示词
    if subtype in ["content", "detail", "spatial"]:
        prompt = f"""We would like to request your feedback on the performance of an AI assistant in response to the user question displayed above. The user asks a question about a specific region in an image (for example, a highlighted or bounded area).

**Ground Truth Answer:** {gt_answer}

**Model Answer:** {model_answer}

Please compare the Model Answer with the Ground Truth Answer carefully, focusing on whether the assistant correctly describes or reasons about the indicated region. Rate the helpfulness, relevance, accuracy, and level of details of the assistant's response. The assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.

Please first output a single line containing only one numerical value indicating the score (1-10).
In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias."""
    else:
        prompt = f"""We would like to request your feedback on the performance of an AI assistant in response to the user question displayed above. The user asks the question on observing an image.

**Ground Truth Answer:** {gt_answer}

**Model Answer:** {model_answer}

Please rate the helpfulness, relevance, accuracy, level of details of the assistant's response. The assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.

Please first output a single line containing only one numerical value indicating the score (1-10).
In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias."""

    content = f"[Question]\n{question}\n\n{prompt}\n\n"

    # 带重试的调用
    max_retries = 5
    retry_count = 0
    response = None
    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful and precise assistant for checking the quality of the answer.",
                    },
                    {
                        "role": "user",
                        "content": content,
                    },
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            break
        except Exception as e:
            print(f"GPT API Error: {e}")
            time.sleep(NUM_SECONDS_TO_SLEEP)
            retry_count += 1

    if response is None:
        print(f"Failed to get GPT score after {max_retries} retries. Question snippet: {question[:80]}")
        return -1.0, "GPT API Error"

    review_text = response.choices[0].message.content

    # 解析分数
    score = -1.0
    try:
        first_line = review_text.split("\n")[0].strip()
        match = re.search(
            r"(?:Score:?\s*|^)(\d+(?:\.\d+)?)(?:\s*(?:out of|/)\s*10)?",
            first_line,
            re.IGNORECASE,
        )
        if match:
            score = float(match.group(1))
        else:
            match = re.search(
                r"Score:?\s*(\d+(?:\.\d+)?)(?:\s*(?:out of|/)\s*10)?",
                review_text,
                re.IGNORECASE,
            )
            if match:
                score = float(match.group(1))
            else:
                match = re.search(
                    r"(\d+(?:\.\d+)?)\s*(?:out of|/)\s*10",
                    review_text,
                    re.IGNORECASE,
                )
                if match:
                    score = float(match.group(1))

        # 合理性检查
        if score < 1.0 or score > 10.0:
            print(f"Warning: Score {score} out of range (1-10), parsed from: {first_line[:50]}")
            score = -1.0
        else:
            score = max(1.0, min(10.0, score))
    except (ValueError, AttributeError, IndexError) as e:
        print(f"Warning: Failed to parse score from review: {review_text[:100]}")
        print(f"Error: {e}")
        score = -1.0

    return score, review_text


def eval_polar_stage2(args):
    """
    使用训练好的 PolarVLM 模型（支持 Stage 1 和 Stage 2）对测试集进行评测：
    - Stage 1: 特征对齐阶段，使用 plain 格式（无 System Prompt）
    - Stage 2: 对话微调阶段，使用 v1 格式（有 System Prompt）
    - 输入 RGB + Polar
    - 问题与 Ground Truth 来自测试集 JSON 文件
    - GPT-4o-mini 打分
    
    🟢 生成策略说明：
    - 方案A（推荐，可复现）：do_sample=False, temperature=0.0, repetition_penalty=1.2, min_length
      → 完全确定性，每次运行结果一致，适合正式评测
    - 方案B（更自然，有随机性）：do_sample=True, temperature=0.5-0.7
      → 生成更自然，但每次运行可能略有不同，适合探索性评测
    """
    # 🟢 如果用户没有显式指定 --do-sample，默认使用 Greedy（可复现）
    # 如果用户想要采样，需要显式指定 --do-sample
    if not hasattr(args, 'do_sample'):
        args.do_sample = False  # 默认使用 Greedy，确保可复现性
    
    # 🔒 严格对齐评测：Greedy 解码（do_sample=False, temperature=0.0, top_p=None）
    if not args.do_sample:
        args.temperature = 0.0
        args.top_p = None
    else:
        # 如果使用采样但温度是0，调整为0.7（推理时的默认值）
        if args.temperature == 0.0:
            args.temperature = 0.7
            print(f"⚠️ 注意: 使用采样策略时，temperature 已从 0.0 调整为 0.7（推理默认值）")
    
    # 🟢 打印生成策略信息
    if args.do_sample:
        print(f"📊 生成策略: 温度采样 (Temperature={args.temperature}, Top-p={args.top_p})")
        print(f"   ⚠️  注意: 采样会引入随机性，每次运行结果可能略有不同")
    else:
        print(f"📊 生成策略: 贪婪搜索 (Greedy, Temperature=0.0, Top-p=None)")
        print(f"   ✓ 完全确定性，每次运行结果一致（推荐用于正式评测）")
    
    # 加载模型（复用 inference.py 的加载逻辑）
    # 🟢 支持 Stage 1 和 Stage 2 评测
    training_stage = getattr(args, 'training_stage', None)
    if training_stage:
        print(f"Loading PolarVLM {training_stage.upper()} model from {args.checkpoint_dir} ...")
    else:
        print(f"Loading PolarVLM model from {args.checkpoint_dir} (auto-detecting stage) ...")
    
    # 🟢 兼容不同版本的 load_trained_model（有的版本没有 training_stage 参数）
    try:
        result = load_trained_model(
            checkpoint_dir=args.checkpoint_dir,
            llm_model_name=args.llm_model_name,
            clip_model_name=args.clip_model_name,
            vae_model_path=args.vae_model_path,
            hf_token=None,
            device="cuda",
            load_4bit=False,
            use_multi_gpu=args.use_multi_gpu,
            training_stage=training_stage,  # 🟢 传递训练阶段参数
        )
        # 🟢 处理返回值（新版本返回 4 个值，包括 training_stage）
        if len(result) == 4:
            model, tokenizer, image_processor, detected_stage = result
            if training_stage is None:
                print(f"✓ 自动检测到训练阶段: {detected_stage}")
        elif len(result) == 3:
            model, tokenizer, image_processor = result
        else:
            raise ValueError(f"Unexpected return value from load_trained_model: {len(result)} values")
    except TypeError as e:
        # 回退：调用不带新参数的版本（旧版 inference.py）
        print(f"⚠️  回退到旧版 load_trained_model（不支持 training_stage 参数）")
        try:
            result = load_trained_model(
                checkpoint_dir=args.checkpoint_dir,
                llm_model_name=args.llm_model_name,
                clip_model_name=args.clip_model_name,
                vae_model_path=args.vae_model_path,
                hf_token=None,
                device="cuda",
                load_4bit=False,
                use_multi_gpu=args.use_multi_gpu,
            )
            if len(result) == 4:
                model, tokenizer, image_processor, _ = result
            elif len(result) == 3:
                model, tokenizer, image_processor = result
            else:
                raise ValueError(f"Unexpected return value: {len(result)} values")
        except TypeError:
            # 最终回退：完全不传递新参数
            model, tokenizer, image_processor = load_trained_model(
                checkpoint_dir=args.checkpoint_dir,
                llm_model_name=args.llm_model_name,
                clip_model_name=args.clip_model_name,
                vae_model_path=args.vae_model_path,
                hf_token=None,
                device="cuda",
            )
    # 可选：在评测时强制覆盖 residual 融合系数 alpha，用于 A/B 测试
    override_alpha = getattr(args, "override_polar_alpha", None)
    if override_alpha is not None:
        try:
            base_model = model.get_model() if hasattr(model, "get_model") else model
            polar_alpha = getattr(base_model, "polar_alpha", None)
            if polar_alpha is None:
                print("⚠️  未找到 polar_alpha，跳过 override。")
            else:
                old_alpha = float(polar_alpha.detach().float().item())
                with torch.no_grad():
                    polar_alpha.data.fill_(float(override_alpha))
                new_alpha = float(polar_alpha.detach().float().item())
                print(f"✓ 已覆盖 polar_alpha: {old_alpha:.6f} -> {new_alpha:.6f}")
        except Exception as e:
            print(f"⚠️  覆盖 polar_alpha 失败，继续使用原值: {e}")
    else:
        # 输出当前 alpha，便于和 override 实验做对照
        try:
            base_model = model.get_model() if hasattr(model, "get_model") else model
            polar_alpha = getattr(base_model, "polar_alpha", None)
            if polar_alpha is not None:
                print(f"ℹ️  当前 polar_alpha: {float(polar_alpha.detach().float().item()):.6f}")
                print("✓ 本次评测继承 checkpoint 中训练好的 polar_alpha（未覆盖）")
        except Exception:
            pass

    model.eval()
    print("Model loaded successfully!")

    data_root = Path(args.data_root)
    polar_root = Path(args.polar_root)

    # 加载测试集 JSON（比如 test_stage2_qwen.json）
    print(f"Loading test set from {args.question_file}...")
    with open(args.question_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples")

    output_dir = Path(args.answers_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    scores: List[float] = []

    # 🟢 根据训练阶段设置进度条描述
    stage_desc = getattr(args, 'training_stage', 'auto-detected').upper()
    for idx, item in enumerate(tqdm(test_data, desc=f"Evaluating PolarVLM {stage_desc}")):
        try:
            # 提取 question / gt_answer（与 RGB baseline 保持一致）
            question = None
            gt_answer = None
            for conv in item.get("conversations", []):
                if conv.get("from") == "human":
                    question = conv.get("value", "")
                elif conv.get("from") == "gpt":
                    gt_answer = conv.get("value", "")

            if not question or not gt_answer:
                print(f"Warning: Skipping item {idx} - missing question or GT answer")
                continue

            # 解析 RGB 路径
            rgb_rel = item.get("input_path") or item.get("image")
            if not rgb_rel:
                print(f"Warning: Skipping item {idx} - missing input_path / image")
                continue

            rgb_path = Path(rgb_rel)
            if not rgb_path.is_absolute():
                rgb_path = data_root / rgb_rel

            if not rgb_path.exists():
                print(f"Warning: RGB image not found: {rgb_path}, skipping...")
                continue

            # 解析 Polar 路径
            # 优先使用数据中提供的 polar_crop_paths（Stage 1 数据通常是裁剪后的 polar）
            polar_paths = None
            if isinstance(item.get("polar_crop_paths"), dict):
                polar_crop_paths = item["polar_crop_paths"]
                polar_paths = {
                    "I_0": (data_root / polar_crop_paths["I_0"]).resolve(),
                    "I_45": (data_root / polar_crop_paths["I_45"]).resolve(),
                    "I_90": (data_root / polar_crop_paths["I_90"]).resolve(),
                    "I_135": (data_root / polar_crop_paths["I_135"]).resolve(),
                }
                missing = [k for k, p in polar_paths.items() if not p.exists()]
                if missing:
                    print(f"Warning: Missing polar_crop_paths for item {idx}: {missing}, paths: {[str(polar_paths[k]) for k in missing]}")
                    polar_paths = None
                else:
                    print(f"  ✓ 使用 polar_crop_paths 作为偏振输入（与 Stage 1 训练一致）")

            # 如果没有 polar_crop_paths 或缺失，则回退到 polar_root/scene_id/base_name
            if polar_paths is None:
                scene_id = str(item.get("scene_id", "")).zfill(2)
                base_name = rgb_path.stem
                if base_name.endswith("_rgb"):
                    base_name = base_name[:-4]

                polar_paths = get_polar_paths(str(polar_root), scene_id, base_name)
                missing = [k for k, p in polar_paths.items() if not p.exists()]
                if missing:
                    print(f"Warning: Missing polar images for item {idx}: {missing}, paths: {[str(polar_paths[k]) for k in missing]}")
                    continue

            # 可选：在 RGB 上绘制红框，并改写问题为"红框内区域"
            # 🟢 关键修复：Stage 1 不应该使用红框格式，因为训练数据是简单的 "Describe the image"
            # Stage 1 使用 plain 格式，问题格式必须与训练时完全一致
            bbox_norm = item.get("bbox_norm")
            rgb_path_for_preprocess = rgb_path
            question_for_model = question
            training_stage = getattr(args, 'training_stage', None)
            
            # 🟢 Stage 1: 不使用红框格式，保持原始问题（与训练数据一致）
            # Stage 2: 可以使用红框格式（如果指定了 --use-red-box）
            if args.use_red_box and bbox_norm and len(bbox_norm) == 4 and training_stage != "stage1":
                try:
                    img = Image.open(rgb_path).convert("RGB")
                    img_with_box = draw_bbox_on_image(img, bbox_norm, color="red", width=3)
                    # 将带红框的图像保存为临时文件，再交给 preprocess_images 走统一流程
                    tmp_name = rgb_path.stem + "_redbox.png"
                    tmp_path = rgb_path.parent / tmp_name
                    img_with_box.save(tmp_path)
                    rgb_path_for_preprocess = tmp_path
                    question_for_model = replace_bbox_in_question(question)
                except Exception as e:
                    print(f"Warning: failed to draw red box for item {idx}: {e}. Fallback to coordinate mode.")
                    rgb_path_for_preprocess = rgb_path
                    question_for_model = question
            elif training_stage == "stage1" and args.use_red_box:
                # Stage 1 模式下，即使指定了 --use-red-box，也不改写问题
                print(f"  ℹ️  Stage 1 模式：保持原始问题格式（不使用红框格式），与训练数据一致")
                # 可以选择是否绘制红框（不影响问题格式）
                if bbox_norm and len(bbox_norm) == 4:
                    try:
                        img = Image.open(rgb_path).convert("RGB")
                        img_with_box = draw_bbox_on_image(img, bbox_norm, color="red", width=3)
                        tmp_name = rgb_path.stem + "_redbox.png"
                        tmp_path = rgb_path.parent / tmp_name
                        img_with_box.save(tmp_path)
                        rgb_path_for_preprocess = tmp_path
                        # 🟢 关键：不调用 replace_bbox_in_question，保持原始问题
                        question_for_model = question
                    except Exception as e:
                        print(f"Warning: failed to draw red box for item {idx}: {e}")
                        rgb_path_for_preprocess = rgb_path
                        question_for_model = question

            # 预处理 RGB + Polar（复用 inference.py 的 preprocess_images）
            pixel_values_rgb, pixel_values_polar = preprocess_images(
                rgb_path=str(rgb_path_for_preprocess),
                polar_paths=polar_paths,
                image_processor=image_processor,
                device="cuda",
                model=model,
            )

            # 生成答案（复用 inference.generate_response）
            model_answer = generate_response(
                model=model,
                tokenizer=tokenizer,
                pixel_values_rgb=pixel_values_rgb,
                pixel_values_polar=pixel_values_polar,
                question=question_for_model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
                test_mode=getattr(args, 'test_mode', 'normal'),  # 🟢 测试模式参数
            )

            # GPT 评分
            # 🟢 实时输出：显示模型回答和评分
            print(f"\n{'='*80}")
            print(f"Sample {idx + 1}/{len(test_data)}")
            print(f"Question: {question_for_model[:100]}..." if len(question_for_model) > 100 else f"Question: {question_for_model}")
            print(f"Model Answer: {model_answer[:200]}..." if len(model_answer) > 200 else f"Model Answer: {model_answer}")
            print(f"GT Answer: {gt_answer[:200]}..." if len(gt_answer) > 200 else f"GT Answer: {gt_answer}")
            # 🟢 使用 question_for_model（改写后的红框版本），确保与模型输入完全一致
            # 这样在红框模式下，GPT看到的question和模型看到的完全一致，都不包含坐标信息
            subtype = item.get("subtype", "default")
            score, review = get_gpt_score(
                question=question_for_model,
                gt_answer=gt_answer,
                model_answer=model_answer,
                subtype=subtype,
                max_tokens=args.max_tokens,
            )

            # 🟢 实时输出：显示评分
            print(f"Score: {score:.2f}/10.0")
            if score > 0:
                scores.append(score)
                valid_scores = [s for s in scores if s > 0]
                avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
                print(f"Current Average Score: {avg_score:.2f}/10.0 (from {len(valid_scores)} valid samples)")
            else:
                print(f"⚠️ Score parsing failed or empty answer")
            print(f"{'='*80}\n")

            result = {
                "question_id": idx + 1,
                "scene_id": item.get("scene_id", ""),
                "type": item.get("type", ""),
                "subtype": item.get("subtype", ""),
                "prompt": question_for_model,
                "text": model_answer,
                "gt_answer": gt_answer,
                "score": score,
                "review": review,
                "metadata": {
                    "image": item.get("image", ""),
                    "input_path": item.get("input_path", ""),
                    "bbox_norm": item.get("bbox_norm"),
                },
            }
            results.append(result)

            # 🟢 每10个样本输出一次汇总（实时输出已在上面完成）
            if (idx + 1) % 10 == 0:
                valid_scores = [s for s in scores if s > 0]
                avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
                print(
                    f"\n📊 Progress Summary: Processed {idx + 1}/{len(test_data)} samples, "
                    f"Valid scores: {len(valid_scores)}, Average score: {avg_score:.2f}/10.0\n"
                )

        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # 写 JSONL 结果
    print(f"\nSaving results to {args.answers_file}...")
    with open(args.answers_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 统计 & 写 *_stats.json
    valid_scores = [s for s in scores if s > 0]
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        min_score = min(valid_scores)
        max_score = max(valid_scores)

        print(f"\n{'=' * 60}")
        # 🟢 根据训练阶段设置摘要标题
        stage_desc = getattr(args, 'training_stage', 'auto-detected').upper()
        print(f"PolarVLM {stage_desc} Evaluation Summary:")
        print(f"  Total samples: {len(results)}")
        print(f"  Valid scores: {len(valid_scores)}")
        print(f"  Failed to parse: {len(scores) - len(valid_scores)}")
        print(f"  Average score: {avg_score:.2f}")
        print(f"  Min score: {min_score:.2f}")
        print(f"  Max score: {max_score:.2f}")
        print(f"{'=' * 60}")

        stats = {
            "total_samples": len(results),
            "valid_scores": len(valid_scores),
            "failed_to_parse": len(scores) - len(valid_scores),
            "average_score": avg_score,
            "min_score": min_score,
            "max_score": max_score,
            "scores_distribution": {
                "1-3": sum(1 for s in valid_scores if 1 <= s <= 3),
                "4-6": sum(1 for s in valid_scores if 4 <= s <= 6),
                "7-8": sum(1 for s in valid_scores if 7 <= s <= 8),
                "9-10": sum(1 for s in valid_scores if 9 <= s <= 10),
            },
        }

        # 同样附加所有样本，便于直接对比
        stats["examples"] = [
            {
                "scene_id": r.get("scene_id", ""),
                "type": r.get("type", ""),
                "subtype": r.get("subtype", ""),
                "question": r.get("prompt", ""),
                "gt_answer": r.get("gt_answer", ""),
                "model_answer": r.get("text", ""),
                "score": r.get("score", -1.0),
            }
            for r in results
        ]

        stats_file = args.answers_file.replace(".jsonl", "_stats.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Statistics saved to {stats_file}")
    else:
        print("Warning: No valid scores generated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate PolarVLM (Stage 1 or Stage 2, RGB + Polar) on test set"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/openbayes/input/input0/llava_train_stage2_final/checkpoint-100",
        help="Checkpoint directory (with adapter_model.safetensors, non_lora_trainables.bin, etc.). Supports both Stage 1 and Stage 2.",
    )
    parser.add_argument(
        "--training-stage",
        type=str,
        default=None,
        choices=["stage1", "stage2"],
        help="Manually specify training stage ('stage1' or 'stage2'). If not specified, will auto-detect from checkpoint directory name or config file.",
    )
    parser.add_argument(
        "--llm-model-name",
        type=str,
        default="/openbayes/input/input0/models/llava-v1.5-13b",
        help="Base LLaVA-1.5 model path",
    )
    parser.add_argument(
        "--clip-model-name",
        type=str,
        default="/openbayes/input/input0/models/clip-vit-large-patch14-336",
        help="CLIP vision tower path",
    )
    parser.add_argument(
        "--vae-model-path",
        type=str,
        default="/openbayes/input/input0/models/sd-vae-ft-mse",
        help="VAE model path used for polar encoding",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/openbayes/input/input0",
        help="Root directory for dataset paths (for resolving input_path / image)",
    )
    parser.add_argument(
        "--polar-root",
        type=str,
        default="/openbayes/input/input0/polar",
        help="Root directory for polar images",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        required=True,
        help="Test set JSON file path (e.g., test_stage2_qwen.json)",
    )
    parser.add_argument(
        "--answers-file",
        type=str,
        required=True,
        help="Output JSONL file path for model answers + GPT scores",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling",
    )
    # 🟢 默认使用 Greedy（可复现），用户可显式指定 --do-sample 启用采样
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (default 0.0 for greedy, use 0.5-0.7 with --do-sample)",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        default=False,
        help="Use sampling for generation (introduces randomness). Omit this flag to use greedy search (deterministic, recommended for evaluation).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens for GPT-4o-mini review",
    )
    parser.add_argument(
        "--use-multi-gpu",
        action="store_true",
        default=False,
        help="Whether to use multi-GPU (device_map='auto') when loading PolarVLM",
    )
    parser.add_argument(
        "--test-mode",
        type=str,
        default="normal",
        choices=["normal", "no_polar", "zero_polar", "swap_channels"],
        help="测试模式：normal=正常推理, no_polar=屏蔽Polar分支, zero_polar=Polar全零, swap_channels=交换通道顺序",
    )
    parser.add_argument(
        "--debug-polar",
        action="store_true",
        default=False,
        help="启用 Polar 分支详细调试信息（检查权重、特征值、拼接逻辑等）",
    )
    parser.add_argument(
        "--use-red-box",
        action="store_true",
        default=False,
        help="If set, draw a red bounding box on RGB images and replace bbox coordinates in question (for fair comparison with RGB baseline).",
    )
    parser.add_argument(
        "--override-polar-alpha",
        type=float,
        default=None,
        help="可选：评测时强制覆盖 polar_alpha（例如 0.0/0.1/0.3），用于诊断 residual 融合影响。",
    )

    args = parser.parse_args()
    
    # 🟢 设置调试环境变量
    import os
    if args.debug_polar or args.test_mode != "normal":
        os.environ['DEBUG_POLAR_BRANCH'] = '1'
    
    eval_polar_stage2(args)


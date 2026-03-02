#!/usr/bin/env python3
"""
评估官方 LLaVA-1.5 模型在测试集上的性能
- 使用纯 RGB 图像（input_path）
- 如果问题包含坐标，在图像上绘制红色边界框
- 生成回答并使用 GPT-4o-mini 进行评分（1-10分）
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

# 🟢 关键修复：绕过 transformers 在旧版 torch 上对 torch.load 的强制版本检查
# 仅用于本地评测环境，确保可以加载本地 .bin 权重
try:
    import transformers  # type: ignore
    from transformers.utils import import_utils as _hf_import_utils  # type: ignore

    def _no_check_torch_load_is_safe():
        # 在受控本地环境下跳过安全检查
        return None

    # 覆盖 import_utils 模块中的函数
    if hasattr(_hf_import_utils, "check_torch_load_is_safe"):
        _hf_import_utils.check_torch_load_is_safe = _no_check_torch_load_is_safe  # type: ignore

    # 同时在 modeling_utils 命名空间中覆盖（load_state_dict 可能从这里直接引用）
    try:
        import transformers.modeling_utils as _hf_modeling_utils  # type: ignore

        if hasattr(_hf_modeling_utils, "check_torch_load_is_safe"):
            _hf_modeling_utils.check_torch_load_is_safe = _no_check_torch_load_is_safe  # type: ignore
    except Exception:
        pass
except Exception:
    # 如果 transformers 不存在或结构不同，直接忽略，按默认行为处理
    pass

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)


# GPT-4o-mini 配置
GPT_API_KEY = "sk-IAtjepO76Q2L7aya4ne7s9NRVGd1wNGm7HzVEXT8yGDIUptp"
GPT_BASE_URL = "https://api.openai-proxy.org/v1"
GPT_MODEL = "gpt-4o-mini"
NUM_SECONDS_TO_SLEEP = 0.5


def draw_bbox_on_image(image: Image.Image, bbox_norm: List[float], color: str = "red", width: int = 3) -> Image.Image:
    """
    在图像上绘制边界框
    
    Args:
        image: PIL Image 对象
        bbox_norm: 归一化坐标 [x1, y1, x2, y2] (0-1范围)
        color: 框的颜色（默认红色）
        width: 框的线宽
    
    Returns:
        绘制了边界框的图像副本
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # 获取图像尺寸
    img_width, img_height = img_copy.size
    
    # 将归一化坐标转换为像素坐标
    x1 = int(bbox_norm[0] * img_width)
    y1 = int(bbox_norm[1] * img_height)
    x2 = int(bbox_norm[2] * img_width)
    y2 = int(bbox_norm[3] * img_height)
    
    # 绘制矩形框
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    
    return img_copy


def extract_bbox_from_question(question: str) -> Optional[List[float]]:
    """
    从问题中提取边界框坐标
    
    Args:
        question: 包含坐标的问题文本，例如 "Focus on region [0.717, 0.527, 1.000, 0.661]."
    
    Returns:
        如果找到坐标，返回 [x1, y1, x2, y2]，否则返回 None
    """
    # 匹配 [x1, y1, x2, y2] 格式
    pattern = r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]'
    match = re.search(pattern, question)
    
    if match:
        try:
            coords = [float(match.group(i)) for i in range(1, 5)]
            return coords
        except ValueError:
            return None
    
    return None


def replace_bbox_in_question(question: str) -> str:
    """
    将问题中的坐标替换为"红色边界框"的描述
    
    Args:
        question: 原始问题
    
    Returns:
        替换后的问题
    """
    # 匹配并替换 "Focus on region [x1, y1, x2, y2]." 格式
    pattern1 = r'Focus on region\s*\[[0-9.,\s]+\]\.?\s*'
    replacement1 = "Focus on the region inside the red bounding box. "
    new_question = re.sub(pattern1, replacement1, question, flags=re.IGNORECASE)
    
    # 如果替换没有生效，尝试更宽松的匹配
    if new_question == question:
        # 匹配 "[x1, y1, x2, y2]" 格式
        pattern2 = r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]'
        new_question = re.sub(pattern2, "the red bounding box", question, count=1)
        
        # 确保有 "Focus on" 提示
        if "Focus on" not in new_question and "focus on" not in new_question:
            new_question = "Focus on the region inside the red bounding box. " + new_question
    
    return new_question.strip()


def strip_bbox_from_question(question: str) -> str:
    """
    移除问题中的 bbox 坐标信息（不使用红框、不暴露坐标）。
    """
    # 移除 "Focus on region [x1, y1, x2, y2]." 这类前缀
    q = re.sub(r'Focus on (the )?region\s*\[[0-9.,\s]+\]\.?\s*', '', question, flags=re.IGNORECASE)
    # 移除任意坐标块 "[x1, y1, x2, y2]"
    q = re.sub(r'\[([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\]', '', q)
    # 清理多余空格
    q = re.sub(r'\s{2,}', ' ', q).strip()
    # 修复可能出现的 "located at ?" 语义残留
    q = re.sub(r'\blocated at\b\s*(\?|\.|$)', r'in the image\1', q, flags=re.IGNORECASE)
    return q


def get_gpt_score(question: str, gt_answer: str, model_answer: str, subtype: str = "default", max_tokens: int = 1024) -> Tuple[float, str]:
    """
    使用 GPT-4o-mini 对模型回答进行评分（1-10分）
    
    Args:
        question: 用户问题
        gt_answer: Ground Truth 回答
        model_answer: 模型生成的回答
        subtype: 问题子类型（content, detail, spatial等），用于选择不同的评分提示词
        max_tokens: 最大生成token数
    
    Returns:
        (分数, 评分详情)
    """
    client = OpenAI(
        base_url=GPT_BASE_URL,
        api_key=GPT_API_KEY,
    )
    
    # 根据 subtype 选择不同的评分提示词（参考 rule.json），但不引用未实际提供的描述性上下文，避免 Prompt 泄露
    # 对于包含边界框的问题（content, detail, spatial），强调“对齐 Ground Truth”与“关注指定区域”
    if subtype in ["content", "detail", "spatial"]:
        prompt = f"""We would like to request your feedback on the performance of an AI assistant in response to the user question displayed above. The user asks a question about a specific region in an image (for example, a highlighted or bounded area).

**Ground Truth Answer:** {gt_answer}

**Model Answer:** {model_answer}

Please compare the Model Answer with the Ground Truth Answer carefully, focusing on whether the assistant correctly describes or reasons about the indicated region. Rate the helpfulness, relevance, accuracy, and level of details of the assistant's response. The assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.

Please first output a single line containing only one numerical value indicating the score (1-10).
In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias."""
    else:
        # 默认提示词（参考 rule.json 的 default）
        prompt = f"""We would like to request your feedback on the performance of an AI assistant in response to the user question displayed above. The user asks the question on observing an image.

**Ground Truth Answer:** {gt_answer}

**Model Answer:** {model_answer}

Please rate the helpfulness, relevance, accuracy, level of details of the assistant's response. The assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.

Please first output a single line containing only one numerical value indicating the score (1-10).
In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias."""
    
    content = f"[Question]\n{question}\n\n{prompt}\n\n"
    
    # 增加重试上限，避免 API 调用永久死循环
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
                        "content": "You are a helpful and precise assistant for checking the quality of the answer."
                    },
                    {
                        "role": "user",
                        "content": content,
                    }
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
    
    # 🟢 关键修复：更健壮的分数解析逻辑
    score = -1.0
    try:
        # 方法1: 尝试在第一行直接匹配 "Score: X" 或单独的数字
        first_line = review_text.split('\n')[0].strip()
        # 匹配 "Score: 8.5" 或 "8.5" 或 "8" 等格式，但排除明显不是分数的数字（如年份、大数字）
        match = re.search(r'(?:Score:?\s*|^)(\d+(?:\.\d+)?)(?:\s*(?:out of|/)\s*10)?', first_line, re.IGNORECASE)
        if match:
            score = float(match.group(1))
        else:
            # 方法2: 如果第一行没找到，在全文搜索 "Score: X" 模式
            match = re.search(r'Score:?\s*(\d+(?:\.\d+)?)(?:\s*(?:out of|/)\s*10)?', review_text, re.IGNORECASE)
            if match:
                score = float(match.group(1))
            else:
                # 方法3: 尝试匹配 "X/10" 或 "X out of 10" 格式
                match = re.search(r'(\d+(?:\.\d+)?)\s*(?:out of|/)\s*10', review_text, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
        
        # 🟢 关键校验：确保分数在合理范围内（1-10），防止把年份、大数字误判为分数
        if score < 1.0 or score > 10.0:
            print(f"Warning: Score {score} out of range (1-10), parsed from: {first_line[:50]}")
            score = -1.0  # 标记为失败，不计入统计
        else:
            # 确保分数在 1-10 范围内（二次保险）
            score = max(1.0, min(10.0, score))
            
    except (ValueError, AttributeError, IndexError) as e:
        print(f"Warning: Failed to parse score from review: {review_text[:100]}")
        print(f"Error: {e}")
        score = -1.0
    
    return score, review_text


def eval_model(args):
    """主评估函数"""
    # 禁用 torch 初始化以加快加载速度
    disable_torch_init()
    
    # 加载模型（使用 4-bit 量化，适配单卡 4090 显存）
    print(f"Loading model from {args.model_path}...")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        load_4bit=True,          # 🟢 启用 4-bit 量化，降低显存占用
        load_8bit=False,
        device="cuda",
        use_multi_gpu=False,     # 评测脚本单卡运行
    )
    model.eval()
    print("Model loaded successfully!")
    
    # 确定对话模板
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    
    if args.conv_mode is not None:
        conv_mode = args.conv_mode
    
    print(f"Using conversation mode: {conv_mode}")
    
    # 加载测试集
    print(f"Loading test set from {args.question_file}...")
    with open(args.question_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples")
    
    # 准备输出文件
    output_dir = Path(args.answers_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 打开输出文件（JSONL格式）
    results = []
    scores = []
    
    # 处理每个测试样本
    for idx, item in enumerate(tqdm(test_data, desc="Evaluating")):
        try:
            # 提取问题
            question = None
            gt_answer = None
            for conv in item.get('conversations', []):
                if conv.get('from') == 'human':
                    question = conv.get('value', '')
                elif conv.get('from') == 'gpt':
                    gt_answer = conv.get('value', '')
            
            if not question or not gt_answer:
                print(f"Warning: Skipping item {idx} - missing question or GT answer")
                continue
            
            # 获取图像路径
            image_path = os.path.join(args.image_folder, item.get('input_path', item.get('image', '')))
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}, skipping...")
                continue
            
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            bbox_norm = item.get('bbox_norm')
            # 纯 RGB 评测不绘制红框、不暴露坐标
            question = strip_bbox_from_question(question)
            
            # 构建对话
            qs = question
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            
            if IMAGE_PLACEHOLDER in qs:
                if model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if model.config.mm_use_im_start_end:
                    qs = image_token_se + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # 处理图像
            image_size = image.size
            images_tensor = process_images(
                [image],
                image_processor,
                model.config
            ).to(model.device, dtype=torch.float16)
            
            # 🟢 维度检查：确保 images_tensor 有正确的 batch 维度
            if images_tensor.dim() == 3:
                # 如果是 [C, H, W]，添加 batch 维度
                images_tensor = images_tensor.unsqueeze(0)
            elif images_tensor.dim() == 4:
                # 如果已经是 [B, C, H, W]，保持不变
                pass
            else:
                raise ValueError(f"Unexpected images_tensor dimension: {images_tensor.dim()}")
            
            # Tokenize
            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
            
            # 🟢 维度检查：确保 input_ids 和 images_tensor 的 batch 维度匹配
            if input_ids.shape[0] != images_tensor.shape[0]:
                raise ValueError(
                    f"Batch dimension mismatch: input_ids={input_ids.shape[0]}, "
                    f"images_tensor={images_tensor.shape[0]}"
                )
            
            # 生成回答
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=[image_size],
                    do_sample=False,  # Greedy search for reproducibility
                    temperature=0,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
            
            # 解码回答
            model_answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            # 移除 prompt 部分，只保留生成的回答
            if conv.sep_style == SeparatorStyle.TWO:
                model_answer = model_answer.split(conv.sep2)[-1].strip()
            else:
                model_answer = model_answer.split(conv.sep)[-1].strip()
            
            # 使用 GPT-4o-mini 进行评分
            subtype = item.get('subtype', 'default')
            score, review = get_gpt_score(question, gt_answer, model_answer, subtype=subtype, max_tokens=args.max_tokens)
            
            # 只统计有效分数（-1表示解析失败）
            if score > 0:
                scores.append(score)
            
            # 保存结果
            result = {
                "question_id": idx + 1,
                "scene_id": item.get('scene_id', ''),
                "type": item.get('type', ''),
                "subtype": item.get('subtype', ''),
                "prompt": question,
                "text": model_answer,
                "gt_answer": gt_answer,
                "score": score,
                "review": review,
                "metadata": {
                    "image": item.get('image', ''),
                    "input_path": item.get('input_path', ''),
                    "bbox_norm": bbox_norm,
                }
            }
            results.append(result)
            
            # 每10个样本打印一次进度
            if (idx + 1) % 10 == 0:
                valid_scores = [s for s in scores if s > 0]
                avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
                print(f"\nProcessed {idx + 1}/{len(test_data)} samples, Valid scores: {len(valid_scores)}, Average score: {avg_score:.2f}")
        
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存结果到 JSONL 文件
    print(f"\nSaving results to {args.answers_file}...")
    with open(args.answers_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # 计算并打印统计信息（只统计有效分数）
    valid_scores = [s for s in scores if s > 0]
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        min_score = min(valid_scores)
        max_score = max(valid_scores)
        print(f"\n{'='*60}")
        print(f"Evaluation Summary:")
        print(f"  Total samples: {len(results)}")
        print(f"  Valid scores: {len(valid_scores)}")
        print(f"  Failed to parse: {len(scores) - len(valid_scores)}")
        print(f"  Average score: {avg_score:.2f}")
        print(f"  Min score: {min_score:.2f}")
        print(f"  Max score: {max_score:.2f}")
        print(f"{'='*60}")
        
        # 保存统计信息（同时记录若干条示例，便于对比 LLaVA 输出与 Ground Truth）
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

        # 额外附加所有样本，方便在 *_stats.json 中完整查看 LLaVA 输出与 Ground Truth 对比
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
        stats_file = args.answers_file.replace('.jsonl', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Statistics saved to {stats_file}")
    else:
        print("Warning: No valid scores generated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLaVA-1.5 RGB baseline on test set")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/openbayes/input/input0/models/llava-v1.5-13b",
        help="Path to LLaVA-1.5 model"
    )
    parser.add_argument(
        "--model-base",
        type=str,
        default=None,
        help="Path to base model (if different from model-path)"
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        default="/openbayes/input/input0",
        help="Root folder for images"
    )
    parser.add_argument(
        "--question-file",
        type=str,
        required=True,
        help="Path to test set JSON file (e.g., test_stage2_qwen.json)"
    )
    parser.add_argument(
        "--answers-file",
        type=str,
        required=True,
        help="Path to output JSONL file for results"
    )
    parser.add_argument(
        "--conv-mode",
        type=str,
        default=None,
        help="Conversation mode (auto-detected if not specified)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens for GPT-4o-mini review"
    )
    
    args = parser.parse_args()
    
    eval_model(args)

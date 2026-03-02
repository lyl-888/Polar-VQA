"""
PolarVLM Stage 3 批量推理脚本
用于批量验证一个场景下的多张图片
"""

import os
import torch
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, List
from PIL import Image
from transformers import AutoTokenizer
from inference import (
    load_trained_model,
    get_polar_paths,
    preprocess_images,
    format_prompt,
    generate_response,
)


def list_scene_images(rgb_root: str, scene_id: str, max_images: int = 10) -> List[str]:
    """
    列出场景目录下的所有RGB图片的基础文件名
    
    Args:
        rgb_root: RGB图像根目录
        scene_id: 场景ID
        max_images: 最大图片数量
    
    Returns:
        基础文件名列表（不含扩展名，如 ['0000', '0001', '0002']）
    """
    scene_dir = Path(rgb_root) / scene_id
    if not scene_dir.exists():
        raise FileNotFoundError(f"场景目录不存在: {scene_dir}")
    
    # 查找所有 _rgb.png 文件
    rgb_files = sorted(scene_dir.glob("*_rgb.png"))
    
    # 提取基础文件名（去掉 _rgb.png 后缀）
    base_names = []
    for rgb_file in rgb_files[:max_images]:
        base_name = rgb_file.stem.replace("_rgb", "")  # 去掉 _rgb 后缀
        base_names.append(base_name)
    
    return base_names


def batch_inference(
    checkpoint_dir: str,
    rgb_root: str,
    polar_root: str,
    scene_id: str,
    max_images: int = 10,
    question: str = "Describe the visual content in this image.",
    llm_model_name: str = "/openbayes/input/input0/models/Meta-Llama-3-8B-Instruct",
    clip_model_name: str = "openai/clip-vit-large-patch14",
    polar_backbone: str = "google/vit-base-patch16-224-in21k",
    stage1_checkpoint: Optional[str] = None,
    stage2_checkpoint: Optional[str] = None,
    hf_token: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    device: str = "cuda",
    output_json: Optional[str] = None,
):
    """
    批量推理：验证一个场景下的多张图片
    
    Args:
        checkpoint_dir: Stage 3 检查点目录
        rgb_root: RGB图像根目录
        polar_root: 偏振图像根目录
        scene_id: 场景ID
        max_images: 最大图片数量
        question: 问题文本
        output_json: 输出JSON文件路径（可选）
        其他参数：与 inference.py 相同
    """
    print("=" * 80)
    print("PolarVLM Stage 3 批量推理")
    print("=" * 80)
    print(f"场景 ID: {scene_id}")
    print(f"最大图片数量: {max_images}")
    print(f"问题: {question}")
    
    # 列出场景下的所有图片
    print(f"\n正在扫描场景目录: {rgb_root}/{scene_id}/")
    base_names = list_scene_images(rgb_root, scene_id, max_images)
    print(f"✓ 找到 {len(base_names)} 张图片: {base_names}")
    
    if len(base_names) == 0:
        print("⚠ 警告: 未找到任何图片，退出")
        return
    
    # 加载模型（只加载一次）
    print("\n" + "=" * 80)
    print("加载模型")
    print("=" * 80)
    model, tokenizer, config = load_trained_model(
        checkpoint_dir=checkpoint_dir,
        llm_model_name=llm_model_name,
        clip_model_name=clip_model_name,
        polar_backbone=polar_backbone,
        stage1_checkpoint=stage1_checkpoint,
        stage2_checkpoint=stage2_checkpoint,
        hf_token=hf_token,
        device=device,
    )
    
    # 批量推理
    print("\n" + "=" * 80)
    print("开始批量推理")
    print("=" * 80)
    
    results = []
    successful = 0
    failed = 0
    
    for idx, base_name in enumerate(base_names, 1):
        print(f"\n[{idx}/{len(base_names)}] 处理: {base_name}")
        print("-" * 80)
        
        try:
            # 构建图像路径
            rgb_path = os.path.join(rgb_root, scene_id, f"{base_name}_rgb.png")
            polar_paths = get_polar_paths(polar_root, scene_id, base_name)
            
            # 检查路径
            if not os.path.exists(rgb_path):
                print(f"⚠ 警告: RGB 图像不存在: {rgb_path}")
                failed += 1
                continue
            
            missing_polar = [name for name, path in polar_paths.items() if not path.exists()]
            if missing_polar:
                print(f"⚠ 警告: 偏振图像缺失: {missing_polar}")
                failed += 1
                continue
            
            # 预处理图像
            pixel_values_rgb, pixel_values_polar = preprocess_images(
                rgb_path=rgb_path,
                polar_paths=polar_paths,
                clip_model_name=clip_model_name,
                device=device,
            )
            
            # 生成回答
            response = generate_response(
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
            
            # 保存结果
            result = {
                "scene_id": scene_id,
                "base_name": base_name,
                "rgb_path": rgb_path,
                "question": question,
                "response": response,
            }
            results.append(result)
            successful += 1
            
            # 打印结果
            print(f"✓ 成功")
            print(f"问题: {question}")
            print(f"回答: {response}")
            
        except Exception as e:
            print(f"❌ 失败: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    # 输出统计信息
    print("\n" + "=" * 80)
    print("批量推理完成")
    print("=" * 80)
    print(f"总图片数: {len(base_names)}")
    print(f"成功: {successful}")
    print(f"失败: {failed}")
    
    # 保存结果到JSON文件
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 结果已保存到: {output_path}")
    
    # 打印所有结果
    print("\n" + "=" * 80)
    print("所有结果")
    print("=" * 80)
    for idx, result in enumerate(results, 1):
        print(f"\n[{idx}] {result['base_name']}:")
        print(f"  问题: {result['question']}")
        print(f"  回答: {result['response']}")


def main():
    parser = argparse.ArgumentParser(description="PolarVLM Stage 3 批量推理脚本")
    
    # 模型配置（与 inference.py 相同）
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Stage 3 检查点目录")
    parser.add_argument("--llm_model_name", type=str,
                        default="/openbayes/input/input0/models/Meta-Llama-3-8B-Instruct",
                        help="基础 LLM 模型路径")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-large-patch14",
                        help="CLIP 模型路径")
    parser.add_argument("--polar_backbone", type=str, default="google/vit-base-patch16-224-in21k",
                        help="偏振流backbone路径")
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Stage 1 检查点路径")
    parser.add_argument("--stage2_checkpoint", type=str, default=None,
                        help="Stage 2 检查点路径")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face token")
    
    # 数据配置
    parser.add_argument("--rgb_root", type=str, default="/openbayes/input/input0/rgb",
                        help="RGB 图像根目录")
    parser.add_argument("--polar_root", type=str, default="/openbayes/input/input0/polar",
                        help="偏振图像根目录")
    parser.add_argument("--scene_id", type=str, required=True,
                        help="场景 ID（如 '10'）")
    parser.add_argument("--max_images", type=int, default=10,
                        help="最大图片数量（默认: 10）")
    parser.add_argument("--question", type=str, default="Describe the visual content in this image.",
                        help="问题文本（所有图片使用相同的问题）")
    
    # 生成参数
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="温度参数")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="nucleus sampling 参数")
    parser.add_argument("--do_sample", action="store_true", default=True,
                        help="使用采样（默认开启）")
    
    # 输出配置
    parser.add_argument("--output_json", type=str, default=None,
                        help="输出JSON文件路径（可选，如 'results_scene10.json'）")
    
    # 设备
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备（cuda 或 cpu）")
    
    args = parser.parse_args()
    
    batch_inference(
        checkpoint_dir=args.checkpoint_dir,
        rgb_root=args.rgb_root,
        polar_root=args.polar_root,
        scene_id=args.scene_id,
        max_images=args.max_images,
        question=args.question,
        llm_model_name=args.llm_model_name,
        clip_model_name=args.clip_model_name,
        polar_backbone=args.polar_backbone,
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_checkpoint=args.stage2_checkpoint,
        hf_token=args.hf_token,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        device=args.device,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()


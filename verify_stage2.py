"""
Stage 2 验证脚本：测试训练好的 PolarVLM 模型推理能力

功能：
1. 加载训练好的 Stage 2 模型（包括 VAE 编码器和 Stage 2 投影器）
2. 使用提示进行推理（建议使用英文，与 Stage 2 训练一致）
3. 验证模型是否能正确理解偏振图像并生成描述

关键修改（适配 Stage 2 训练代码）：
- 偏振流：完全使用 VAE 编码器（不再使用 ViT backbone）
- Polar 输入：3 通道（DoLP, sin(2*AoLP), cos(2*AoLP)），去掉 Intensity
- RGB 图像：从 512x512 crop 图像 resize 到 224x224（CLIP 需要）
- Polar 图像：保持 512x512（VAE 编码器需要）
- 数据路径：使用 data_root 解析 rgb_crop/ 和 polar_crop/ 路径

关键约束：
- 必须正确加载 VAE 编码器权重（通过 vae_model_path）和 Stage 2 投影器权重
- 推理提示建议使用英文（与 Stage 2 训练数据格式一致）
"""

import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
import torchvision.transforms as transforms

# 导入模型和数据处理函数
from model import PolarLlava
from dataset_common import process_polar_images


def load_stage2_model(
    stage2_checkpoint: str,
    clip_model_name: str = "/openbayes/input/input0/models/clip-vit-large-patch14",
    polar_backbone: str = "/openbayes/input/input0/models/vit-base-patch16-224-in21k",  # ⚠️ 已废弃：仅作为备用选项
    llm_model_name: str = "/openbayes/input/input0/models/Meta-Llama-3-8B-Instruct",
    vae_model_path: str = "/openbayes/input/input0/models/sd-vae-ft-mse",  # VAE 模型路径（用于加载 VAE 编码器）
    stage1_checkpoint: str = None,  # ⚠️ 已废弃：改用 vae_model_path
    hf_token: str = None,
    device: torch.device = None,
    new_vocab_size: int = None,
):
    """
    加载 Stage 2 训练好的模型
    
    Args:
        stage2_checkpoint: Stage 2 检查点路径（包含投影器权重）
        clip_model_name: CLIP 模型路径
        polar_backbone: ⚠️ 已废弃：偏振流 backbone 路径（仅作为备用选项，当 vae_model_path 不存在时使用）
        llm_model_name: LLM 模型路径
        vae_model_path: VAE 模型路径（用于加载 VAE 编码器）。如果提供此路径，polar_backbone 将被忽略。
        stage1_checkpoint: ⚠️ 已废弃：Stage 1 检查点路径（改用 vae_model_path）
        hf_token: Hugging Face token
        device: 设备
        new_vocab_size: 新的词表大小（用于调整 LLM embedding）
    
    Returns:
        加载的 PolarLlava 模型
    """
    print("=" * 80)
    print("加载 Stage 2 模型")
    print("=" * 80)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 获取 token
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    # 提示信息：如果提供了 vae_model_path，polar_backbone 将被忽略
    if vae_model_path and os.path.exists(vae_model_path):
        print(f"✓ 检测到 VAE 模型路径: {vae_model_path}")
        print(f"  ⚠️  注意: polar_backbone 参数将被忽略（仅作为备用选项）")
    elif vae_model_path:
        print(f"⚠️  警告: VAE 模型路径不存在: {vae_model_path}")
        print(f"  → 将回退使用 polar_backbone: {polar_backbone}")
    
    # 创建模型（不加载 Stage 2 权重，稍后手动加载）
    model = PolarLlava(
        clip_model_name=clip_model_name,
        polar_backbone=polar_backbone,  # 备用选项（如果 vae_model_path 不存在）
        llm_model_name=llm_model_name,
        load_in_4bit=True,  # 使用 4-bit 量化节省显存
        hf_token=token,
        freeze_rgb_tower=True,
        freeze_polar_tower=True,  # Stage 2 中偏振流是冻结的
        stage1_checkpoint=stage1_checkpoint,  # 已废弃，保留以兼容旧代码
        vae_model_path=vae_model_path,  # 使用 VAE 编码器
        use_lora=False,  # Stage 2 不使用 LoRA
        # 关键：传入新的词表大小，确保 LLM 的 embedding 层已为 <image> 等新 token 扩容
        new_vocab_size=new_vocab_size,
    )
    model = model.to(device)
    model.eval()
    
    # 加载 Stage 2 投影器权重
    print("\n" + "=" * 80)
    print("加载 Stage 2 投影器权重...")
    print("=" * 80)
    
    projector_path = os.path.join(stage2_checkpoint, "projector.pth")
    if not os.path.exists(projector_path):
        # 尝试其他可能的路径
        possible_paths = [
            os.path.join(stage2_checkpoint, "projector.pth"),
            os.path.join(stage2_checkpoint, "pytorch_model.bin"),
            os.path.join(stage2_checkpoint, "model.safetensors"),
        ]
        projector_path = None
        for p in possible_paths:
            if os.path.exists(p):
                projector_path = p
                break
        
        if projector_path is None:
            raise FileNotFoundError(
                f"在 {stage2_checkpoint} 中未找到投影器权重文件\n"
                f"尝试的路径: {possible_paths}"
            )
    
    print(f"  正在加载: {projector_path}")
    
    if projector_path.endswith(".safetensors"):
        import safetensors.torch
        projector_state = safetensors.torch.load_file(projector_path)
    else:
        projector_state = torch.load(projector_path, map_location=device)
    
    # 如果状态字典包含 "model." 前缀，需要去掉
    if any(k.startswith("model.") for k in projector_state.keys()):
        projector_state = {k.replace("model.", ""): v for k, v in projector_state.items()}
    
    # 如果状态字典包含 "multi_modal_projector." 前缀，需要去掉
    if any(k.startswith("multi_modal_projector.") for k in projector_state.keys()):
        projector_state = {
            k.replace("multi_modal_projector.", ""): v
            for k, v in projector_state.items()
        }
    
    # 加载投影器权重
    missing_keys, unexpected_keys = model.multi_modal_projector.load_state_dict(
        projector_state, strict=False
    )
    
    if missing_keys:
        print(f"  ⚠ 警告: {len(missing_keys)} 个键未加载: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"  ⚠ 警告: {len(unexpected_keys)} 个意外的键: {unexpected_keys[:5]}...")
    
    print(f"  ✓ Stage 2 投影器权重加载完成")
    
    # 打印 Scale 参数，确认是否加载成功（这些参数在 MultiModalProjector 内部）
    if hasattr(model.multi_modal_projector, 'rgb_scale'):
        rgb_scale_val = model.multi_modal_projector.rgb_scale.item()
        print(f"  ✓ Projector RGB Scale: {rgb_scale_val:.4f}")
    if hasattr(model.multi_modal_projector, 'polar_scale'):
        polar_scale_val = model.multi_modal_projector.polar_scale.item()
        print(f"  ✓ Projector Polar Scale: {polar_scale_val:.4f}")
    
    return model, device


def setup_tokenizer(llm_model_name: str, hf_token: str = None):
    """
    设置 tokenizer
    
    Args:
        llm_model_name: LLM 模型名称或路径
        hf_token: Hugging Face token
    
    Returns:
        配置好的 tokenizer
    """
    print("\n" + "=" * 80)
    print("设置 Tokenizer...")
    print("=" * 80)
    
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    load_kwargs = {}
    if token:
        load_kwargs["token"] = token
    
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name, **load_kwargs)
    
    # 检查并添加 <image> token
    if "<image>" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["<image>"], special_tokens=True)
        print("  ✓ 已添加 <image> token")
    else:
        print("  ✓ <image> token 已存在")
    
    # 设置 padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    tokenizer.padding_side = "right"
    
    print(f"  ✓ Tokenizer 配置完成，词汇表大小: {len(tokenizer)}")
    return tokenizer


def preprocess_images(
    rgb_path: Path,
    polar_paths: dict,
    device: torch.device,
    clip_model_name: str = "/openbayes/input/input0/models/clip-vit-large-patch14",
):
    """
    预处理 RGB 和偏振图像（与训练阶段保持一致）
    
    ⚠️ 关键修改：
    - RGB 图像：从 512x512 crop 图像 resize 到 224x224（CLIP 需要）
    - Polar 图像：保持 512x512（VAE 编码器需要）
    - Polar 输入：3 通道（DoLP, sin(2*AoLP), cos(2*AoLP)），去掉 Intensity
    
    Args:
        rgb_path: RGB 图像路径（应该指向 rgb_crop/ 目录，512x512）
        polar_paths: 偏振图像路径字典（应该指向 polar_crop/ 目录）
        device: 设备
        clip_model_name: CLIP 模型路径
    
    Returns:
        (pixel_values_rgb, pixel_values_polar) 元组
        - pixel_values_rgb: (1, 3, 224, 224) - CLIP 需要的尺寸
        - pixel_values_polar: (1, 3, 512, 512) - VAE 需要的尺寸，3通道
    """
    # 1. 处理 RGB 图像（使用 CLIPImageProcessor，与训练阶段一致）
    rgb_image = Image.open(rgb_path).convert("RGB")
    
    # ⚠️ 关键：RGB 图像从 512x512 resize 到 224x224（CLIP 需要）
    # 与训练阶段保持一致：dataset_stage2.py 中的 rgb_transform 会 resize 到 224x224
    rgb_image = rgb_image.resize((224, 224), Image.BILINEAR)
    
    # 尝试使用本地路径加载 CLIP 处理器
    try:
        if os.path.exists(clip_model_name):
            clip_processor = CLIPImageProcessor.from_pretrained(
                clip_model_name,
                local_files_only=True
            )
        else:
            clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    except Exception:
        clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    pixel_values_rgb = clip_processor(
        rgb_image,
        return_tensors="pt"
    )["pixel_values"]  # (1, 3, 224, 224)
    pixel_values_rgb = pixel_values_rgb.to(device)
    
    # 2. 处理偏振图像（与训练阶段保持一致）
    # 使用 process_polar_images 转换为 4 通道物理参数
    physics_img = process_polar_images(polar_paths=polar_paths)  # (H, W, 4)，值范围[0, 1]
    # 通道顺序：[Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]
    
    # ⚠️ 关键修改：只取后3个通道（去掉 Intensity）
    # 与训练阶段保持一致：dataset_stage2.py 中的 _load_polar_images 只取 [DoLP, sin(2*AoLP), cos(2*AoLP)]
    physics_img_3ch = physics_img[:, :, 1:4]  # (H, W, 3)，值范围[0, 1]
    
    # 转换为 PIL Image（RGB 模式，3通道）
    physics_img_uint8 = (physics_img_3ch * 255).astype(np.uint8)
    physics_pil = Image.fromarray(physics_img_uint8, mode='RGB')
    
    # ⚠️ 关键修改：Polar 图像需要保持 512x512（VAE 编码器要求）
    # 与训练阶段保持一致：dataset_stage2.py 中的 polar_transform 会 resize 到 512x512
    polar_transform = transforms.Compose([
        transforms.Resize((512, 512)),  # VAE 需要的尺寸：512x512
        transforms.ToTensor(),  # 转换为 [0, 1] 范围的张量
        # ⚠️ 关键：不使用 ImageNet 归一化，与 Stage 2 训练保持一致
    ])
    
    pixel_values_polar = polar_transform(physics_pil)  # (3, 512, 512)，值范围[0, 1]
    pixel_values_polar = pixel_values_polar.unsqueeze(0)  # (1, 3, 512, 512)
    pixel_values_polar = pixel_values_polar.to(device)
    
    # 确保 dtype 匹配
    if pixel_values_polar.dtype != pixel_values_rgb.dtype:
        pixel_values_polar = pixel_values_polar.to(dtype=pixel_values_rgb.dtype)
    
    return pixel_values_rgb, pixel_values_polar


def clean_generated_text(text: str) -> str:
    """
    清理生成的文本，移除乱码、多语言、重复内容
    
    处理的问题：
    1. 末尾的乱码符号（如 ・━・━、udál、***、?\"等）
    2. 多语言文本（检测到非英文时截断）
    3. 重复的句子或短语
    4. 异常的长句子（可能是模型开始"编造"）
    
    Args:
        text: 原始生成的文本
    
    Returns:
        清理后的文本
    """
    import re
    
    if not text:
        return text
    
    # 1. 检测并移除常见的乱码符号模式
    # 乱码符号：・━・━、udál、***、?\"、等
    garbage_patterns = [
        r'・━・━+',  # 重复的日文符号（常见乱码）
        r'udál[^\s]*',  # 捷克语乱码
        r'\*\*\*+',  # 多个星号（3个或更多）
        r'\?\\?"\s*$',  # 末尾的转义问号和引号
        r'[^\x20-\x7E\u00A0-\u00FF]+',  # 非标准ASCII/拉丁字符（但保留常见标点）
    ]
    
    # 先尝试找到第一个乱码符号的位置
    first_garbage_pos = len(text)
    for pattern in garbage_patterns:
        match = re.search(pattern, text)
        if match:
            # 对于末尾模式，直接截断；对于中间模式，截断到匹配位置
            if pattern.endswith('$'):
                # 末尾模式：找到最后一个匹配的位置
                matches = list(re.finditer(pattern, text))
                if matches:
                    first_garbage_pos = min(first_garbage_pos, matches[-1].start())
            else:
                first_garbage_pos = min(first_garbage_pos, match.start())
    
    # 如果找到乱码，截断到乱码之前
    if first_garbage_pos < len(text):
        text = text[:first_garbage_pos].strip()
        print(f"  ✓ 检测到乱码符号，截断到第 {first_garbage_pos} 个字符")
    
    # 2. 检测多语言文本（如果出现明显的非英文句子，截断）
    # 检测法语、中文、日文等常见模式
    non_english_patterns = [
        r'\b(forme|montre|situé|étagère|livre|tailles|couleurs|trouée|fond|mur|trous|percés|luminosité|image|ton|atténué)',  # 法语关键词
        r'[\u4e00-\u9fff]+',  # 中文字符
        r'[\u3040-\u309F\u30A0-\u30FF]+',  # 日文假名
    ]
    
    for pattern in non_english_patterns:
        match = re.search(pattern, text)
        if match:
            # 如果非英文出现在文本的后半部分（可能是模型开始"编造"），截断
            if match.start() > len(text) * 0.5:
                text = text[:match.start()].strip()
                print(f"  ✓ 检测到多语言文本，截断到第 {match.start()} 个字符")
                break
    
    # 3. 移除末尾的异常标点或重复模式
    # 如果文本以奇怪的标点结尾（如 ?\"、***），移除
    text = re.sub(r'[?\\"]+\s*$', '', text)
    text = re.sub(r'\*+\s*$', '', text)
    
    # 4. 移除重复的句子（简单的启发式方法）
    sentences = re.split(r'[.!?]\s+', text)
    if len(sentences) > 1:
        # 检查是否有完全重复的句子
        seen = set()
        unique_sentences = []
        for sent in sentences:
            sent_lower = sent.lower().strip()
            if sent_lower and sent_lower not in seen:
                seen.add(sent_lower)
                unique_sentences.append(sent)
            elif sent_lower in seen:
                print(f"  ✓ 检测到重复句子，已移除: {sent[:50]}...")
        
        if len(unique_sentences) < len(sentences):
            # 重新组合（保留最后一个句子的标点）
            text = '. '.join(unique_sentences)
            if text and not text.endswith(('.', '!', '?')):
                text += '.'
    
    # 5. 最终清理：移除多余空格和换行
    text = re.sub(r'\s+', ' ', text)  # 多个空格合并为一个
    text = text.strip()
    
    return text


def format_prompt(question: str) -> str:
    """
    格式化提示（Stage 2 版本，与训练阶段保持一致）

    非常重要的一点：
    - 在 Stage 2 训练时，我们并没有使用 LLaMA-3 的 chat template、<image> 等特殊标记，
      只是把 Question 和 Answer 简单拼成一段纯文本：

          prompt_text = f"{question}\\n{answer}"

    - 因此，在 Stage 2 验证时，如果继续手工拼接
      `<|begin_of_text|><|start_header_id|>user...<image>...`
      会和训练分布严重不一致，导致 LLM 不知道如何停止、容易复读输入。

    这里为了评估"在有图像特征注入时，LLM 是否学会往下接 Answer 风格的文字"，
    我们只保留 Question 本身，作为前缀，让模型生成后续内容。
    
    注意：Stage 2 训练数据使用英文（"Describe the image"），建议验证时也使用英文提示。
    """
    # 添加换行符，与训练格式保持一致（训练时是 "question\nanswer"）
    # 这样模型知道问题结束了，要开始生成答案了
    return f"{question.strip()}\n"


def inference(
    model: PolarLlava,
    tokenizer: AutoTokenizer,
    pixel_values_rgb: torch.Tensor,
    pixel_values_polar: torch.Tensor,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """
    执行推理
    
    Args:
        model: PolarLlava 模型
        tokenizer: Tokenizer
        pixel_values_rgb: RGB 图像张量
        pixel_values_polar: 偏振图像张量
        prompt: 提示文本（中文）
        max_new_tokens: 最大生成 token 数
        temperature: 温度参数
        top_p: Top-p 采样参数
    
    Returns:
        生成的文本
    """
    print("\n" + "=" * 80)
    print("执行推理...")
    print("=" * 80)

    # [关键修复] 确保 prompt 包含 BOS token，与训练时保持一致
    # 训练时（collate_fn_stage2）手动添加了 BOS token，并设置 add_special_tokens=False
    # 正确的结构应该是：[BOS] + [Vision] + [Text]
    # LLM 依赖 BOS token 来重置注意力状态，必须放在最前面
    if tokenizer.bos_token:
        # 手动添加 BOS，与训练时的 collate_fn_stage2 保持一致
        prompt_with_bos = tokenizer.bos_token + prompt
        print(f"  ✓ 已添加 BOS token: {tokenizer.bos_token}")
    else:
        prompt_with_bos = prompt
        print(f"  ⚠ 警告: Tokenizer 没有 BOS token，这可能导致生成失败")
    
    # Tokenize 提示（与 Stage 2 训练保持一致：手动添加 BOS，然后设置 add_special_tokens=False）
    # 这样确保 tokenizer 不会乱加，而我们可以控制结构为：[BOS] + [Text]
    # 模型 forward 中会将视觉特征插在 BOS 后面：[BOS] + [Vision] + [Text]
    input_ids = tokenizer.encode(prompt_with_bos, return_tensors="pt", add_special_tokens=False)
    
    # 确保 input_ids 在正确的设备上（显式指定设备，避免类型不匹配）
    device = pixel_values_rgb.device
    input_ids = input_ids.to(device=device)
    
    # [诊断] 检查第一个 token 是否是 BOS（用于调试）
    # 获取 BOS token ID（可能来自 tokenizer 或 model config）
    bos_token_id = None
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        bos_token_id = tokenizer.bos_token_id
    elif hasattr(tokenizer, 'convert_tokens_to_ids') and tokenizer.bos_token:
        bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
    
    if bos_token_id is not None:
        first_token_id = input_ids[0, 0].item()
        if first_token_id == bos_token_id:
            print(f"  ✓ 确认第一个 token 是 BOS (ID: {bos_token_id})")
        else:
            print(f"  ⚠ 警告: 第一个 token ID ({first_token_id}) 不是 BOS token ID ({bos_token_id})")
            print(f"    这可能导致模型无法正确生成")
            print(f"    尝试手动修复...")
            # 如果第一个 token 不是 BOS，手动添加
            if input_ids[0, 0].item() != bos_token_id:
                bos_tensor = torch.tensor([[bos_token_id]], device=device, dtype=input_ids.dtype)
                input_ids = torch.cat([bos_tensor, input_ids], dim=1)
                print(f"  ✓ 已手动添加 BOS token")
    else:
        print(f"  ⚠ 警告: 无法确定 BOS token ID")
        print(f"    如果生成失败，可能需要检查 tokenizer 配置")
    
    # 创建 attention_mask（全1，因为所有token都是有效的）
    attention_mask = torch.ones(
        input_ids.shape,
        dtype=torch.long,
        device=device  # 使用相同的设备
    )
    
    print(f"  ✓ 提示已 tokenize，长度: {input_ids.shape[1]}")
    print(f"  提示内容: {prompt[:100]}...")
    
    # 生成（添加重复惩罚和更严格的停止条件）
    with torch.no_grad():
        try:
            generated_ids = model.generate(
                pixel_values_rgb=pixel_values_rgb,
                pixel_values_polar=pixel_values_polar,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 使用 greedy decoding，更稳定
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.3,  # [改进] 提高重复惩罚，从1.2到1.3，减少重复生成
                no_repeat_ngram_size=4,  # [改进] 从3-gram改为4-gram，更严格防止重复
                length_penalty=1.1,  # [新增] 长度惩罚，鼓励生成更简洁的文本
            )
        except Exception as e:
            print(f"  ⚠ Greedy decoding 失败，尝试 sampling: {e}")
            import traceback
            traceback.print_exc()
            # 如果失败，尝试 sampling
            generated_ids = model.generate(
                pixel_values_rgb=pixel_values_rgb,
                pixel_values_polar=pixel_values_polar,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.6,  # [改进] 降低温度，从0.7到0.6，减少随机性
                top_p=0.85,  # [改进] 降低top_p，从0.9到0.85，更聚焦高概率token
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.3,  # [改进] 提高重复惩罚，从1.2到1.3
                no_repeat_ngram_size=4,  # [改进] 从3-gram改为4-gram
                length_penalty=1.1,  # [新增] 长度惩罚
            )
    
    # 解码生成的文本
    # 只解码新生成的部分（去掉输入部分）
    new_token_ids = generated_ids[0][input_ids.shape[1]:]
    
    # 调试：打印生成的 token IDs（前20个）
    print(f"  ✓ 生成完成，新生成 {len(new_token_ids)} 个 token")
    if len(new_token_ids) > 0:
        print(f"  前20个 token IDs: {new_token_ids[:20].tolist()}")
        
        # 检查 token IDs 是否在有效范围内
        vocab_size = len(tokenizer)
        invalid_tokens = (new_token_ids >= vocab_size).sum().item()
        if invalid_tokens > 0:
            print(f"  ⚠ 警告: 发现 {invalid_tokens} 个超出词表范围的 token ID")
            # 过滤掉无效的 token IDs
            valid_mask = new_token_ids < vocab_size
            new_token_ids = new_token_ids[valid_mask]
            print(f"  过滤后剩余 {len(new_token_ids)} 个有效 token")
    
    # 如果生成了 token，尝试解码
    if len(new_token_ids) == 0:
        return "[未生成任何内容]"
    
    # 找到 EOS token 的位置，只解码到 EOS 之前
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        eos_positions = (new_token_ids == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            new_token_ids = new_token_ids[:eos_positions[0]]
            print(f"  ✓ 检测到 EOS token，截断到第 {len(new_token_ids)} 个 token")
    
    # 检查是否有异常重复的 token（如果前10个token都是同一个，说明可能出问题了）
    if len(new_token_ids) >= 10:
        first_10_tokens = new_token_ids[:10].tolist()
        if len(set(first_10_tokens)) == 1:
            print(f"  ⚠ 警告: 检测到前10个token都是同一个 ({first_10_tokens[0]})，可能是生成失败")
            print(f"    尝试截断到第一个不同token之前")
            # 找到第一个不同的token
            first_token = new_token_ids[0].item()
            for i in range(1, len(new_token_ids)):
                if new_token_ids[i].item() != first_token:
                    new_token_ids = new_token_ids[:i]
                    break
    
    # 解码（跳过特殊token）
    generated_text = tokenizer.decode(
        new_token_ids,
        skip_special_tokens=True
    )
    
    # 清理可能的残留特殊token文本
    generated_text = generated_text.strip()
    
    # [改进] 后处理清理：移除乱码、多语言、重复内容
    generated_text = clean_generated_text(generated_text)
    
    return generated_text


def verify_single_image(
    model: PolarLlava,
    tokenizer: AutoTokenizer,
    device: torch.device,
    data_root: Path,
    rgb_root: Path,
    polar_root: Path,
    scene_id: str,
    base_name: str,
    prompt: str,
    use_crop: bool,
    max_new_tokens: int,
    clip_model_name: str,
) -> dict:
    """
    验证单张图像
    
    Returns:
        包含验证结果的字典
    """
    result = {
        "scene_id": scene_id,
        "base_name": base_name,
        "success": False,
        "error": None,
        "generated_text": None,
        "rgb_path": None,
        "polar_paths": {},
    }
    
    try:
        # 准备图像路径
        if use_crop:
            # RGB 图像路径（从 rgb_crop/ 目录读取）
            rgb_crop_path = data_root / "rgb_crop" / scene_id / f"{base_name}_rgb.png"
            if rgb_crop_path.exists():
                rgb_path = rgb_crop_path
            else:
                # 回退到原始路径
                rgb_path = rgb_root / scene_id / f"{base_name}_rgb.png"
                if not rgb_path.exists():
                    rgb_path = rgb_root / scene_id / f"{base_name}.png"
                if not rgb_path.exists():
                    raise FileNotFoundError(f"RGB 图像不存在: {rgb_crop_path} 或 {rgb_path}")
            
            # 偏振图像路径（从 polar_crop/ 目录读取）
            polar_crop_dir = data_root / "polar_crop" / scene_id
            polar_paths = {
                'I_0': polar_crop_dir / f"{base_name}_000.png",
                'I_45': polar_crop_dir / f"{base_name}_045.png",
                'I_90': polar_crop_dir / f"{base_name}_090.png",
                'I_135': polar_crop_dir / f"{base_name}_135.png",
            }
            
            # 检查是否存在，如果不存在则回退到原始路径
            all_exist = all(p.exists() for p in polar_paths.values())
            if not all_exist:
                scene_dir = polar_root / scene_id
                polar_paths = {
                    'I_0': scene_dir / f"{base_name}_000.png",
                    'I_45': scene_dir / f"{base_name}_045.png",
                    'I_90': scene_dir / f"{base_name}_090.png",
                    'I_135': scene_dir / f"{base_name}_135.png",
                }
                # 尝试其他格式
                if not all(p.exists() for p in polar_paths.values()):
                    polar_paths = {
                        'I_0': scene_dir / f"{base_name}_0.png",
                        'I_45': scene_dir / f"{base_name}_45.png",
                        'I_90': scene_dir / f"{base_name}_90.png",
                        'I_135': scene_dir / f"{base_name}_135.png",
                    }
        else:
            # 使用原始路径
            rgb_path = rgb_root / scene_id / f"{base_name}_rgb.png"
            if not rgb_path.exists():
                rgb_path = rgb_root / scene_id / f"{base_name}.png"
            if not rgb_path.exists():
                raise FileNotFoundError(f"RGB 图像不存在: {rgb_path}")
            
            scene_dir = polar_root / scene_id
            polar_paths = {
                'I_0': scene_dir / f"{base_name}_000.png",
                'I_45': scene_dir / f"{base_name}_045.png",
                'I_90': scene_dir / f"{base_name}_090.png",
                'I_135': scene_dir / f"{base_name}_135.png",
            }
            # 尝试其他格式
            if not all(p.exists() for p in polar_paths.values()):
                polar_paths = {
                    'I_0': scene_dir / f"{base_name}_0.png",
                    'I_45': scene_dir / f"{base_name}_45.png",
                    'I_90': scene_dir / f"{base_name}_90.png",
                    'I_135': scene_dir / f"{base_name}_135.png",
                }
        
        # 最终检查
        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB 图像不存在: {rgb_path}")
        for name, path in polar_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"偏振图像不存在: {name}: {path}")
        
        result["rgb_path"] = str(rgb_path)
        result["polar_paths"] = {k: str(v) for k, v in polar_paths.items()}
        
        # 预处理图像
        pixel_values_rgb, pixel_values_polar = preprocess_images(
            rgb_path=rgb_path,
            polar_paths=polar_paths,
            device=device,
            clip_model_name=clip_model_name,
        )
        
        # 格式化提示
        formatted_prompt = format_prompt(prompt)
        
        # 执行推理
        generated_text = inference(
            model=model,
            tokenizer=tokenizer,
            pixel_values_rgb=pixel_values_rgb,
            pixel_values_polar=pixel_values_polar,
            prompt=formatted_prompt,
            max_new_tokens=max_new_tokens,
        )
        
        result["success"] = True
        result["generated_text"] = generated_text
        
    except Exception as e:
        result["error"] = str(e)
        result["success"] = False
    
    return result


def main():
    """主函数"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Stage 2 验证脚本")
    parser.add_argument(
        "--stage1_checkpoint",
        type=str,
        default=None,
        help="⚠️ 已废弃：Stage 1 检查点路径（改用 --vae_model_path）"
    )
    parser.add_argument(
        "--stage2_checkpoint",
        type=str,
        default="/openbayes/input/input0/checkpoints/stage2_projector",
        help="Stage 2 检查点路径（包含投影器权重）"
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="/openbayes/input/input0/models/clip-vit-large-patch14",
        help="CLIP 模型路径"
    )
    parser.add_argument(
        "--polar_backbone",
        type=str,
        default="/openbayes/input/input0/models/vit-base-patch16-224-in21k",
        help="⚠️ 已废弃：偏振流 backbone 路径（仅作为备用选项，当 --vae_model_path 不存在时使用）。如果提供了 --vae_model_path，此参数将被完全忽略。"
    )
    parser.add_argument(
        "--vae_model_path",
        type=str,
        default="/openbayes/input/input0/models/sd-vae-ft-mse",
        help="VAE 模型路径（用于加载 VAE 编码器）。如果提供此路径，--polar_backbone 将被忽略。"
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="/openbayes/input/input0/models/Meta-Llama-3-8B-Instruct",
        help="LLM 模型路径（Instruct 版本）"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/openbayes/input/input0",
        help="数据根目录（用于解析 crop 路径，如 rgb_crop/, polar_crop/）"
    )
    parser.add_argument(
        "--rgb_root",
        type=str,
        default="/openbayes/input/input0/rgb",
        help="RGB 图像根目录（备用，如果 crop 路径不存在）"
    )
    parser.add_argument(
        "--polar_root",
        type=str,
        default="/openbayes/input/input0/polar",
        help="偏振图像根目录（备用，如果 crop 路径不存在）"
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default="01",
        help="测试场景ID"
    )
    parser.add_argument(
        "--base_name",
        type=str,
        default=None,
        help="测试图像基础名称（不含后缀）。如果为 None，将验证整个场景的所有图像"
    )
    parser.add_argument(
        "--use_crop",
        action="store_true",
        default=True,
        help="使用裁剪后的图像（从 rgb_crop/ 和 polar_crop/ 目录读取，默认启用，与训练一致）"
    )
    parser.add_argument(
        "--batch_verify",
        action="store_true",
        default=False,
        help="批量验证模式：验证指定场景下的所有图像（当 base_name 为 None 时自动启用）"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="批量验证结果输出文件（JSON格式）。如果为 None，只打印结果"
    )
    parser.add_argument(
        "--no_use_crop",
        action="store_false",
        dest="use_crop",
        help="禁用裁剪图像，使用原始图像路径"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the image",
        help="提示文本（建议使用英文，与 Stage 2 训练一致）"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token（可选）"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="最大生成 token 数"
    )
    parser.add_argument(
        "--overfit_mode",
        action="store_true",
        default=False,
        help="过拟合验证模式：从 stage2_checkpoint/overfit_image_info.json 读取过拟合图片信息并自动验证"
    )
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 处理过拟合模式
    if args.overfit_mode:
        print("\n" + "=" * 80)
        print("⚠️  过拟合验证模式")
        print("=" * 80)
        
        # 从 stage2_checkpoint 目录读取过拟合图片信息
        overfit_info_path = os.path.join(args.stage2_checkpoint, "overfit_image_info.json")
        if not os.path.exists(overfit_info_path):
            raise FileNotFoundError(
                f"过拟合信息文件不存在: {overfit_info_path}\n"
                f"请确保 Stage 2 训练时使用了 --overfit_single_image 参数"
            )
        
        with open(overfit_info_path, 'r', encoding='utf-8') as f:
            overfit_info = json.load(f)
        
        # 自动设置 scene_id 和 base_name
        if 'scene_id' in overfit_info and overfit_info['scene_id']:
            args.scene_id = overfit_info['scene_id']
        if 'base_name' in overfit_info and overfit_info['base_name']:
            args.base_name = overfit_info['base_name']
        
        print(f"  ✓ 从 {overfit_info_path} 读取过拟合图片信息:")
        print(f"    - scene_id: {args.scene_id}")
        print(f"    - base_name: {args.base_name}")
        print(f"    - image_idx: {overfit_info.get('image_idx', 'N/A')}")
        print(f"\n  ⚠️  注意: 将使用过拟合图片进行验证")
        print(f"    如果模型能够准确描述这张图片，说明过拟合训练成功！")
        # 过拟合模式下强制使用 crop 图像（与训练一致）
        args.use_crop = True
        print(f"  ✓ 已自动启用 --use_crop（与过拟合训练保持一致）")
        print("=" * 80)
    
    # 1. 先设置 tokenizer（确保包含 <image>，并获取最终词表大小）
    tokenizer = setup_tokenizer(args.llm_model, hf_token=args.hf_token)
    
    # 2. 加载模型，并将 new_vocab_size 传入，以便调整 LLM 的 embedding 大小
    model, device = load_stage2_model(
        stage2_checkpoint=args.stage2_checkpoint,
        clip_model_name=args.clip_model,
        polar_backbone=args.polar_backbone,
        llm_model_name=args.llm_model,
        vae_model_path=args.vae_model_path,
        stage1_checkpoint=args.stage1_checkpoint,  # 已废弃，保留以兼容旧代码
        hf_token=args.hf_token,
        device=device,
        new_vocab_size=len(tokenizer),
    )
    
    # 3. 准备测试数据
    data_root = Path(args.data_root)
    rgb_root = Path(args.rgb_root)
    polar_root = Path(args.polar_root)
    
    # 判断是单张验证还是批量验证
    if args.base_name is None or args.batch_verify:
        # 批量验证模式：验证整个场景的所有图像
        print("\n" + "=" * 80)
        print(f"批量验证模式：场景 {args.scene_id}")
        print("=" * 80)
        
        # 获取场景下的所有RGB图像
        if args.use_crop:
            scene_rgb_dir = data_root / "rgb_crop" / args.scene_id
        else:
            scene_rgb_dir = rgb_root / args.scene_id
        
        if not scene_rgb_dir.exists():
            raise FileNotFoundError(f"场景目录不存在: {scene_rgb_dir}")
        
        # 查找所有 RGB 图像
        rgb_files = sorted(scene_rgb_dir.glob("*_rgb.png"))
        if not rgb_files:
            # 尝试其他格式
            rgb_files = sorted(scene_rgb_dir.glob("*.png"))
        
        if not rgb_files:
            raise FileNotFoundError(f"场景 {args.scene_id} 下没有找到图像文件")
        
        # 提取 base_name（去掉 _rgb.png 后缀）
        base_names = []
        for rgb_file in rgb_files:
            base_name = rgb_file.stem
            if base_name.endswith('_rgb'):
                base_name = base_name[:-4]
            base_names.append(base_name)
        
        print(f"  找到 {len(base_names)} 张图像，开始批量验证...")
        print(f"  图像列表: {base_names[:10]}{'...' if len(base_names) > 10 else ''}")
        
        # 批量验证
        results = []
        for idx, base_name in enumerate(base_names, 1):
            print(f"\n[{idx}/{len(base_names)}] 验证图像: {base_name}")
            print("-" * 80)
            
            result = verify_single_image(
                model=model,
                tokenizer=tokenizer,
                device=device,
                data_root=data_root,
                rgb_root=rgb_root,
                polar_root=polar_root,
                scene_id=args.scene_id,
                base_name=base_name,
                prompt=args.prompt,
                use_crop=args.use_crop,
                max_new_tokens=args.max_new_tokens,
                clip_model_name=args.clip_model,
            )
            
            results.append(result)
            
            if result["success"]:
                print(f"  ✓ 成功: {result['generated_text'][:100]}...")
            else:
                print(f"  ✗ 失败: {result['error']}")
        
        # 统计结果
        success_count = sum(1 for r in results if r["success"])
        print("\n" + "=" * 80)
        print("批量验证结果统计")
        print("=" * 80)
        print(f"  总图像数: {len(results)}")
        print(f"  成功: {success_count}")
        print(f"  失败: {len(results) - success_count}")
        print(f"  成功率: {success_count / len(results) * 100:.1f}%")
        
        # 保存结果
        if args.output_file:
            output_path = Path(args.output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "scene_id": args.scene_id,
                    "total": len(results),
                    "success": success_count,
                    "failed": len(results) - success_count,
                    "results": results
                }, f, ensure_ascii=False, indent=2)
            print(f"\n  ✓ 结果已保存到: {output_path}")
        else:
            print("\n  ⚠️ 提示: 使用 --output_file 参数可以保存详细结果到 JSON 文件")
        
        # 显示部分成功结果
        print("\n" + "=" * 80)
        print("部分验证结果示例")
        print("=" * 80)
        success_results = [r for r in results if r["success"]]
        for i, result in enumerate(success_results[:5], 1):
            print(f"\n[{i}] {result['base_name']}:")
            print(f"    {result['generated_text'][:150]}...")
        
    else:
        # 单张验证模式
        print("\n" + "=" * 80)
        print("准备测试数据...")
        print("=" * 80)
        
        # 优先使用裁剪后的图像（与训练阶段一致）
        if args.use_crop:
            # RGB 图像路径（从 rgb_crop/ 目录读取）
            rgb_crop_path = data_root / "rgb_crop" / args.scene_id / f"{args.base_name}_rgb.png"
            if rgb_crop_path.exists():
                rgb_path = rgb_crop_path
                print(f"  ✓ RGB 图像（crop）: {rgb_path}")
            else:
                # 回退到原始路径
                rgb_path = rgb_root / args.scene_id / f"{args.base_name}_rgb.png"
                if not rgb_path.exists():
                    rgb_path = rgb_root / args.scene_id / f"{args.base_name}.png"
                if not rgb_path.exists():
                    raise FileNotFoundError(f"RGB 图像不存在: {rgb_crop_path} 或 {rgb_path}")
                print(f"  ⚠ RGB 图像（原始，非 crop）: {rgb_path}")
            
            # 偏振图像路径（从 polar_crop/ 目录读取）
            polar_crop_dir = data_root / "polar_crop" / args.scene_id
            polar_paths = {
                'I_0': polar_crop_dir / f"{args.base_name}_000.png",
                'I_45': polar_crop_dir / f"{args.base_name}_045.png",
                'I_90': polar_crop_dir / f"{args.base_name}_090.png",
                'I_135': polar_crop_dir / f"{args.base_name}_135.png",
            }
            
            # 检查是否存在，如果不存在则回退到原始路径
            all_exist = all(p.exists() for p in polar_paths.values())
            if not all_exist:
                print("  ⚠ 警告: 部分 polar_crop 图像不存在，尝试原始路径...")
                scene_dir = polar_root / args.scene_id
                polar_paths = {
                    'I_0': scene_dir / f"{args.base_name}_000.png",
                    'I_45': scene_dir / f"{args.base_name}_045.png",
                    'I_90': scene_dir / f"{args.base_name}_090.png",
                    'I_135': scene_dir / f"{args.base_name}_135.png",
                }
                # 尝试其他格式
                if not all(p.exists() for p in polar_paths.values()):
                    polar_paths = {
                        'I_0': scene_dir / f"{args.base_name}_0.png",
                        'I_45': scene_dir / f"{args.base_name}_45.png",
                        'I_90': scene_dir / f"{args.base_name}_90.png",
                        'I_135': scene_dir / f"{args.base_name}_135.png",
                    }
        else:
            # 使用原始路径
            rgb_path = rgb_root / args.scene_id / f"{args.base_name}_rgb.png"
            if not rgb_path.exists():
                rgb_path = rgb_root / args.scene_id / f"{args.base_name}.png"
            if not rgb_path.exists():
                raise FileNotFoundError(f"RGB 图像不存在: {rgb_path}")
            print(f"  ✓ RGB 图像: {rgb_path}")
            
            scene_dir = polar_root / args.scene_id
            polar_paths = {
                'I_0': scene_dir / f"{args.base_name}_000.png",
                'I_45': scene_dir / f"{args.base_name}_045.png",
                'I_90': scene_dir / f"{args.base_name}_090.png",
                'I_135': scene_dir / f"{args.base_name}_135.png",
            }
            # 尝试其他格式
            if not all(p.exists() for p in polar_paths.values()):
                polar_paths = {
                    'I_0': scene_dir / f"{args.base_name}_0.png",
                    'I_45': scene_dir / f"{args.base_name}_45.png",
                    'I_90': scene_dir / f"{args.base_name}_90.png",
                    'I_135': scene_dir / f"{args.base_name}_135.png",
                }
        
        # 最终检查
        for name, path in polar_paths.items():
            if path.exists():
                print(f"  ✓ {name}: {path}")
            else:
                raise FileNotFoundError(f"偏振图像不存在: {path}")
        
        result = verify_single_image(
            model=model,
            tokenizer=tokenizer,
            device=device,
            data_root=data_root,
            rgb_root=rgb_root,
            polar_root=polar_root,
            scene_id=args.scene_id,
            base_name=args.base_name,
            prompt=args.prompt,
            use_crop=args.use_crop,
            max_new_tokens=args.max_new_tokens,
            clip_model_name=args.clip_model,
        )
        
        if result["success"]:
            print(f"  ✓ RGB 图像: {result['rgb_path']}")
            for name, path in result['polar_paths'].items():
                print(f"  ✓ {name}: {path}")
            
            print("\n" + "=" * 80)
            if args.overfit_mode:
                print("过拟合验证结果")
            else:
                print("推理结果")
            print("=" * 80)
            print(f"\n原始生成文本:\n{result['generated_text']}\n")
            print("=" * 80)
            if args.overfit_mode:
                print("\n⚠️ 过拟合验证提示:")
                print("这是过拟合训练后的验证结果。")
                print("如果上面的输出能够准确描述图像内容（使用英文），")
                print("说明模型已经成功过拟合到这张图片，训练流程正常！")
                print("接下来可以使用完整数据集进行正常训练。")
            else:
                print("\n⚠️ 重要提示:")
                print("如果上面的输出能够准确描述图像内容（使用英文），")
                print("说明 Stage 2 语义对齐训练成功！")
            print("=" * 80)
        else:
            print(f"\n✗ 验证失败: {result['error']}")
            raise RuntimeError(f"验证失败: {result['error']}")


if __name__ == "__main__":
    main()


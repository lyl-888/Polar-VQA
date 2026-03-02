"""
训练脚本：PolarVLM Stage 3 - Visual Instruction Tuning
针对小数据集（~800场景，~2400 QA对）优化的训练配置

[Stage 3 更新] 适配 Stage 2 的改动：
- 偏振编码器：使用 VAE 编码器（通过 vae_model_path 参数），而不是 Stage 1 的 MAE 编码器
- 偏振输入：3 通道（DoLP, sin(2*AoLP), cos(2*AoLP)），去掉 Intensity
- 图像尺寸：
  - RGB: 224x224（CLIP 需要）
  - Polar: 512x512（VAE 编码器需要）
- polar_backbone 参数：已废弃，仅作为备用选项（当 vae_model_path 不存在时使用）
"""

import os
import json
import torch
from pathlib import Path
from typing import Optional
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, TaskType
from dataset import PolarGlareDataset, collate_fn
from model import PolarLlava
import warnings
warnings.filterwarnings("ignore")

# [新增] 设置 tokenizers 并行化环境变量，消除 fork 后的警告
# 这需要在导入 transformers 之前设置，避免 DataLoader worker fork 后的警告
# 设置为 "false" 可以避免 "The current process just got forked" 警告
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 设置 Hugging Face token（如果提供）
def setup_hf_token(token: Optional[str] = None):
    """
    设置 Hugging Face token 用于下载模型
    
    Args:
        token: Hugging Face token，如果为 None，尝试从环境变量或文件读取
    """
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        print("✓ 已设置 Hugging Face token")
    elif os.environ.get("HF_TOKEN"):
        print("✓ 使用环境变量中的 Hugging Face token")
    elif os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("✓ 使用环境变量中的 Hugging Face token")
    else:
        # 尝试从 ~/.huggingface/token 读取
        token_file = os.path.expanduser("~/.huggingface/token")
        if os.path.exists(token_file):
            with open(token_file, 'r') as f:
                token = f.read().strip()
                os.environ["HF_TOKEN"] = token
                os.environ["HUGGING_FACE_HUB_TOKEN"] = token
                print("✓ 从 ~/.huggingface/token 读取 Hugging Face token")
        else:
            print("⚠ 警告: 未找到 Hugging Face token，某些模型可能需要认证")


def setup_tokenizer(model_name: str = "/openbayes/input/input0/models/Meta-Llama-3-8B-Instruct", hf_token: Optional[str] = None) -> AutoTokenizer:
    """
    设置并配置tokenizer，添加<image>特殊token
    
    Args:
        model_name: LLM模型名称
        hf_token: Hugging Face token（用于下载模型）
        
    Returns:
        配置好的tokenizer
    """
    print("正在加载tokenizer...")
    # 优先使用参数，其次环境变量
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    load_kwargs = {}
    if token:
        load_kwargs["token"] = token
    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    
    # 检查 LLaMA-3 的关键特殊 token 是否存在
    # 这些 token 是 LLaMA-3 格式对话所必需的
    # 注意：这些 token 必须与 dataset.py 中 _format_prompt 使用的格式完全一致
    required_llama3_tokens = [
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
    ]
    missing_tokens = []
    for token in required_llama3_tokens:
        if token not in tokenizer.get_vocab():
            missing_tokens.append(token)
    
    if missing_tokens:
        raise ValueError(
            f"Tokenizer 缺少 LLaMA-3 必需的特殊标记: {missing_tokens}\n"
            f"请确保使用的是正确的 LLaMA-3 tokenizer (与 /openbayes/input/input0/models/Meta-Llama-3-8B-Instruct 对应)\n"
            f"这些 token 必须与 dataset.py 中 _format_prompt 使用的格式完全一致"
        )
    else:
        print("✓ LLaMA-3 特殊标记检查通过")
        print("  - 确认所有必需 token 存在，与 dataset.py 中的 _format_prompt 格式一致")
    
    # 添加<image>特殊token（如果还没有）
    if "<image>" not in tokenizer.get_vocab():
        # 添加特殊token
        tokenizer.add_tokens(["<image>"], special_tokens=True)
        print("✓ 已添加<image>特殊token")
    else:
        print("✓ <image>特殊token已存在")
    
    # 设置pad token（LLaMA-3默认没有pad token）
    if tokenizer.pad_token is None:
        # 优先尝试使用保留的special token作为pad token
        # 这样可以避免使用eos_token，保持语义清晰
        fallback_pad_tokens = [
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
        ]
        found_pad_token = False
        for pad_candidate in fallback_pad_tokens:
            if pad_candidate in tokenizer.get_vocab():
                tokenizer.pad_token = pad_candidate
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(pad_candidate)
                print(f"✓ 使用保留token作为pad_token: {pad_candidate}")
                found_pad_token = True
                break
        
        # 如果没有找到保留token，使用eos_token（兼容方案）
        if not found_pad_token:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print("✓ 使用eos_token作为pad_token（兼容方案）")
    
    # 设置padding方向
    tokenizer.padding_side = "right"
    
    # 关键检查：确保 <image> token 不会被拆分成多个 token
    # 这对于 Model 和 Dataset 的一致性至关重要
    test_ids = tokenizer.encode("<image>", add_special_tokens=False)
    if len(test_ids) != 1:
        raise ValueError(
            f"错误：Tokenizer 把 <image> 拆成了多个 token: {test_ids}\n"
            f"这会导致 Model 和 Dataset 之间的不一致。\n"
            f"请检查 tokenizer.add_tokens() 是否成功执行，或 tokenizer 是否正确加载。"
        )
    else:
        print(f"✓ <image> token 完整性检查通过 (token_id: {test_ids[0]})")
    
    print(f"✓ Tokenizer配置完成，词汇表大小: {len(tokenizer)}")
    return tokenizer


def create_custom_collate_fn(tokenizer):
    """
    创建自定义的collate函数（包装dataset.py中的collate_fn）
    
    Args:
        tokenizer: 分词器
        
    Returns:
        collate函数
    """
    def custom_collate_fn(batch):
        return collate_fn(batch, tokenizer)
    return custom_collate_fn


def main(
    # 数据路径
    train_json: str = "merged_stage3_data.json",  # [Stage 3 更新] 默认使用合并后的 Stage 3 数据
    val_json: Optional[str] = None,  # [Stage 3 更新] 验证集可选
    rgb_root: str = "/openbayes/input/input0/rgb",  # [Stage 3 更新] 默认路径（用于非裁剪图像）
    polar_root: str = "/openbayes/input/input0/polar",  # [Stage 3 更新] 默认路径（用于非裁剪图像）
    data_root: Optional[str] = None,  # [Stage 3 新增] 数据根目录（用于解析 crop 路径和 GT 路径，默认使用 rgb_root 的父目录）
    
    # 模型配置
    clip_model_name: str = "openai/clip-vit-large-patch14",
    polar_backbone: str = "google/vit-base-patch16-224-in21k",  # ⚠️ 已废弃：仅作为备用选项（当 vae_model_path 不存在时使用）。如果提供了 vae_model_path，此参数将被完全忽略。
    llm_model_name: str = "/openbayes/input/input0/models/Meta-Llama-3-8B-Instruct",
    stage1_checkpoint: Optional[str] = None,  # ⚠️ 已废弃：Stage 1 检查点路径（改用 vae_model_path）
    stage2_checkpoint: Optional[str] = None,  # Stage 2 检查点路径（可选，用于加载投影器权重）
    vae_model_path: str = "/openbayes/input/input0/models/sd-vae-ft-mse",  # VAE 模型路径（用于加载 VAE 编码器）。如果提供此路径，polar_backbone 将被忽略。
    load_in_4bit: bool = True,
    hf_token: Optional[str] = None,  # Hugging Face token
    
    # LoRA配置
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    
    # 训练配置（针对中等数据集优化，默认适用于~8000条数据）
    output_dir: str = "../autodl-tmp/checkpoints/polarvlm",  # 保存到 autodl-tmp 目录（与 train 并列）
    per_device_train_batch_size: int = 8,  # [A5880 优化] A5880 48G 显存充足，Batch=8 提升GPU利用率
    gradient_accumulation_steps: int = 2,  # 梯度累积（有效批次大小 = 8 * 2 = 16，保持不变）
    num_train_epochs: int = 10,  # [优化] 减少到10个epoch防止过拟合（约4700步，足够收敛）
    learning_rate: float = 1e-4,  # 降低学习率，防止梯度波动过大
    warmup_ratio: float = 0.1,  # 增加预热时间，让权重平稳对齐
    weight_decay: float = 0.05,  # 增强正则化，防止过拟合
    max_grad_norm: float = 1.0,
    
    # 其他配置（针对中等数据集优化监控）
    logging_steps: int = 10,  # 日志记录步数（约每半个epoch记录一次）
    save_steps: Optional[int] = None,  # 如果为 None，将在训练途中保存一次（在中间位置），建议设置为230
    save_strategy: str = "steps",  # 保存策略："steps" 或 "epoch"
    eval_steps: int = 230,  # [优化] 验证步数（约每半个epoch验证一次，与save_steps对齐）
    eval_strategy: str = "steps",  # 验证策略："steps"、"epoch" 或 "no"
    save_total_limit: int = 3,  # [优化] 保留3个checkpoint（最佳+最新+中间）
    load_best_model_at_end: bool = True,  # 训练结束后加载最佳模型（需要验证集）
    bf16: bool = True,  # [关键修改] 开启 BF16，A5880 支持且推荐（PyTorch >= 1.13）
    fp16: bool = False,  # 4-bit 量化模型不支持 fp16 混合精度训练（会导致 "Attempting to unscale FP16 gradients" 错误）
    dataloader_num_workers: int = 4,  # [A5880 优化] A5880 48G 显存充足，workers=4 提升数据加载效率
    remove_unused_columns: bool = False,  # 保留所有列（包括图像）
    
    # 冻结策略
    freeze_rgb_tower: bool = True,
    freeze_polar_tower: bool = True,  # [Stage 3 更新] 默认冻结 Polar Tower，只训练 Projector + LoRA
    freeze_polar_layers: int = 0,  # 冻结偏振流的前N层（防止过拟合，当 freeze_polar_tower=False 时生效）
):
    """
    主训练函数
    """
    print("=" * 80)
    print("PolarVLM Stage 3: Visual Instruction Tuning")
    print("=" * 80)
    
    # ========== 0. 设置 Hugging Face Token ==========
    setup_hf_token(hf_token)
    
    # ========== 0.5. 确保输出目录存在 ==========
    # 提前创建输出目录，避免训练结束时才发现路径不存在
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ 输出目录已创建/确认存在: {output_dir}")
    
    # ========== 1. 设置设备 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    
    # ========== 2. 加载tokenizer ==========
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tokenizer = setup_tokenizer(llm_model_name, hf_token=token)
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    print(f"<image> token ID: {image_token_id}")
    
    # ========== 3. 创建数据集 ==========
    print("\n正在创建训练集...")
    # [Stage 3 更新] 自动推断 data_root（如果未提供）
    if data_root is None:
        rgb_root_path = Path(rgb_root)
        if rgb_root_path.is_absolute():
            data_root = str(rgb_root_path.parent)
        else:
            data_root = str(Path(rgb_root).resolve().parent)
        print(f"✓ 自动推断 data_root: {data_root}")
    
    train_dataset = PolarGlareDataset(
        rgb_root=rgb_root,
        polar_root=polar_root,
        json_file=train_json,
        is_train=True,
        image_size=224,  # ViT 模型期望的输入尺寸
        clip_model_name=clip_model_name,  # 传递 CLIP 模型路径，使用本地模型避免网络下载
        data_root=data_root,  # [Stage 3 新增] 传递 data_root 参数
    )
    print(f"✓ 训练集大小: {len(train_dataset)}")
    
    val_dataset = None
    if val_json and os.path.exists(val_json):
        print("\n正在创建验证集...")
        val_dataset = PolarGlareDataset(
            rgb_root=rgb_root,
            polar_root=polar_root,
            json_file=val_json,
            is_train=False,
            image_size=224,  # ViT 模型期望的输入尺寸
            clip_model_name=clip_model_name,  # 传递 CLIP 模型路径，使用本地模型避免网络下载
            data_root=data_root,  # [Stage 3 新增] 传递 data_root 参数
        )
        print(f"✓ 验证集大小: {len(val_dataset)}")
    else:
        print("⚠ 未找到验证集，将跳过验证")
    
    # ========== 4. 创建模型 ==========
    print("\n正在初始化模型...")
    # 获取 token（用于模型加载）
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    # 提示信息：如果提供了 vae_model_path，polar_backbone 将被忽略
    if vae_model_path and os.path.exists(vae_model_path):
        print(f"✓ 检测到 VAE 模型路径: {vae_model_path}")
        print(f"  ⚠️  注意: polar_backbone 参数将被忽略（仅作为备用选项）")
    elif vae_model_path:
        print(f"⚠️  警告: VAE 模型路径不存在: {vae_model_path}")
        print(f"  → 将回退使用 polar_backbone: {polar_backbone}")
    
    # 在模型初始化时就传递 vocab_size，这样可以在应用LoRA之前resize
    # LLaMA-3-8B 的默认 vocab_size 是 128256，如果 tokenizer 添加了新 token（如<image>），
    # 需要 resize。我们传递 tokenizer 的大小给模型，模型会在应用 LoRA 之前检查并 resize
    model = PolarLlava(
        clip_model_name=clip_model_name,
        freeze_rgb_tower=freeze_rgb_tower,
        polar_backbone=polar_backbone,  # 备用选项（如果 vae_model_path 不存在）
        freeze_polar_tower=freeze_polar_tower,
        freeze_polar_layers=freeze_polar_layers,
        stage1_checkpoint=stage1_checkpoint,  # 已废弃，保留以兼容旧代码
        stage2_checkpoint=stage2_checkpoint,  # 传递 Stage 2 检查点路径（如果提供）
        vae_model_path=vae_model_path,  # 使用 VAE 编码器
        llm_model_name=llm_model_name,
        load_in_4bit=load_in_4bit,
        use_lora=True,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        hf_token=token,  # 传递 token 给模型
        new_vocab_size=len(tokenizer),  # 传递 tokenizer 的词汇表大小，模型会自动判断是否需要 resize
    )
    
    # 设置image_token_id（用于在forward中识别<image> token）
    if "<image>" in tokenizer.get_vocab():
        model.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        print(f"✓ 已设置image_token_id: {model.image_token_id}")
    else:
        # 尝试使用保留的special token作为替代
        # LLaMA-3通常有 <|reserved_special_token_0|> 到 <|reserved_special_token_9|>
        fallback_tokens = [
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
        ]
        found_fallback = False
        for fallback_token in fallback_tokens:
            if fallback_token in tokenizer.get_vocab():
                model.image_token_id = tokenizer.convert_tokens_to_ids(fallback_token)
                print(f"⚠ 警告: <image> token未找到，使用备用token: {fallback_token} (ID: {model.image_token_id})")
                print(f"   提示: 请确保dataset.py中的<image>也被替换为{fallback_token}")
                found_fallback = True
                break
        
        if not found_fallback:
            print("⚠ 警告: <image> token未找到，且无可用备用token")
            print("   将使用'插入并扩展'方案（在序列开头插入视觉特征）")
            model.image_token_id = None
    
    # 将<image> token的嵌入初始化为视觉特征的均值（可选优化）
    # 这里简化处理，实际可以更精细地初始化
    
    print("✓ 模型初始化完成")
    
    # [关键优化] ========== 强制冻结 Embedding 和 LM Head ==========
    # 对于小数据集（4k样本），强制冻结 Embedding 和 LM Head 以防止过拟合
    # [Stage 3 更新] 只训练 LoRA + Projector（Polar Tower 默认冻结）
    print("\n[关键优化] 正在强制冻结 Embedding 和 LM Head 以防止小数据集过拟合...")
    print(f"  - Polar Tower 冻结状态: {freeze_polar_tower} (默认 True，只训练 Projector + LoRA)")
    
    # 1. 冻结输入 Embedding 层
    if hasattr(model.language_model, "get_input_embeddings"):
        input_embeddings = model.language_model.get_input_embeddings()
        if input_embeddings is not None:
            input_embeddings.requires_grad_(False)
            print("  ✓ Embedding 层 (get_input_embeddings) 已冻结")
    
    # 2. 冻结输出 Head 层
    if hasattr(model.language_model, "get_output_embeddings"):
        output_embeddings = model.language_model.get_output_embeddings()
        if output_embeddings is not None:
            output_embeddings.requires_grad_(False)
            print("  ✓ LM Head 层 (get_output_embeddings) 已冻结")
    
    # 3. 双重保险：直接通过 named_parameters 冻结（处理 PEFT 包装的情况）
    # PEFT 可能会包装模型，需要检查实际的模型结构
    from peft import PeftModel
    if isinstance(model.language_model, PeftModel):
        # 如果是 PEFT 模型，需要访问 base_model
        base_model = model.language_model.get_base_model()
        if hasattr(base_model, "embed_tokens"):
            base_model.embed_tokens.requires_grad_(False)
            print("  ✓ embed_tokens (通过 base_model) 已冻结")
        if hasattr(base_model, "lm_head"):
            base_model.lm_head.requires_grad_(False)
            print("  ✓ lm_head (通过 base_model) 已冻结")
    else:
        # 如果不是 PEFT 模型（通常不会发生）
        if hasattr(model.language_model, "embed_tokens"):
            model.language_model.embed_tokens.requires_grad_(False)
            print("  ✓ embed_tokens 已冻结")
        if hasattr(model.language_model, "lm_head"):
            model.language_model.lm_head.requires_grad_(False)
            print("  ✓ lm_head 已冻结")
    
    # 4. 确保 Projector 和 Polar Tower 的梯度状态正确
    # 解冻 Projector（必须训练）
    for n, p in model.multi_modal_projector.named_parameters():
        p.requires_grad = True
    print("  ✓ Projector 梯度已开启（必须训练）")
    
    # Polar Tower 的梯度状态（根据配置）
    # [Stage 3 更新] 默认冻结 Polar Tower，只训练 Projector + LoRA
    if not freeze_polar_tower:
        for n, p in model.vision_tower_polar.named_parameters():
            p.requires_grad = True
        print("  ✓ Polar Tower 梯度已开启（使用 --no_freeze_polar_tower 启用）")
    else:
        # 确保 Polar Tower 完全冻结
        for n, p in model.vision_tower_polar.named_parameters():
            p.requires_grad = False
        print("  ✓ Polar Tower 已冻结（默认配置，只训练 Projector + LoRA）")
    
    # 5. 重新计算并打印参数量，确认比例是否下降
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_ratio = trainable_params / total_params * 100
    print(f"\n[最终确认] 优化后可训练参数: {trainable_params:,} / {total_params:,}")
    print(f"  -> 参数占比: {trainable_ratio:.2f}% (预期应在 1% - 3% 之间)")
    
    # 6. 打印各模块的可训练参数统计
    print("\n[可训练模块统计]")
    projector_trainable = sum(p.numel() for p in model.multi_modal_projector.parameters() if p.requires_grad)
    print(f"  - Projector: {projector_trainable:,} 参数（必须训练）")
    
    if not freeze_polar_tower:
        polar_trainable = sum(p.numel() for p in model.vision_tower_polar.parameters() if p.requires_grad)
        print(f"  - Polar Tower: {polar_trainable:,} 参数（使用 --no_freeze_polar_tower 启用）")
    else:
        print(f"  - Polar Tower: 0 参数（已冻结，默认配置）")
    
    llm_trainable = sum(p.numel() for p in model.language_model.parameters() if p.requires_grad)
    print(f"  - LLM (LoRA): {llm_trainable:,} 参数（必须训练）")
    print(f"\n[训练配置总结]")
    print(f"  - 可训练模块: Projector + LoRA")
    print(f"  - 冻结模块: RGB Tower + Polar Tower + Embedding + LM Head")
    
    print("=" * 80 + "\n")
    
    # ========== 5. 创建自定义collate函数 ==========
    data_collator = create_custom_collate_fn(tokenizer)
    
    # ========== 6. 配置训练参数 ==========
    print("\n正在配置训练参数（针对中等数据集优化，A5880 48G显存）...")
    
    # 估算总步数（用于调整保存策略）
    train_size = len(train_dataset)
    total_steps = (train_size * num_train_epochs) // (per_device_train_batch_size * gradient_accumulation_steps)
    print(f"  - 训练集大小: {train_size}")
    print(f"  - 估算总步数: ~{total_steps}")
    
    # 保存策略：根据 save_strategy 决定
    if save_strategy == "epoch":
        # 按 epoch 保存，不需要 save_steps
        adjusted_save_steps = None
        print(f"  - 保存策略：每个 epoch 保存一次")
    elif save_steps is None:
        # 在训练中途保存一次（总步数的一半左右）
        adjusted_save_steps = max(1, total_steps // 2)  # 在中间位置保存一次
        print(f"  - 保存策略：训练途中保存一次（第 {adjusted_save_steps} 步），训练结束后保存一次")
    else:
        adjusted_save_steps = save_steps
        print(f"  - 保存策略：每 {adjusted_save_steps} 步保存一次")
    
    # 验证策略
    if val_dataset:
        if eval_strategy == "no":
            adjusted_eval_strategy = "no"
            adjusted_eval_steps = None
        elif eval_strategy == "epoch":
            adjusted_eval_strategy = "epoch"
            adjusted_eval_steps = None
        else:  # "steps"
            adjusted_eval_strategy = "steps"
            adjusted_eval_steps = eval_steps
    else:
        adjusted_eval_strategy = "no"
        adjusted_eval_steps = None
    
    # 检查 load_best_model_at_end 与策略匹配性
    # 如果启用了 load_best_model_at_end，保存策略和验证策略必须匹配
    if load_best_model_at_end and val_dataset:
        if save_strategy != adjusted_eval_strategy:
            print(f"\n⚠ 警告: --load_best_model_at_end 要求保存策略和验证策略必须匹配")
            print(f"  当前设置: save_strategy={save_strategy}, eval_strategy={adjusted_eval_strategy}")
            print(f"  自动调整: 将 eval_strategy 调整为 {save_strategy} 以匹配 save_strategy")
            adjusted_eval_strategy = save_strategy
            if save_strategy == "epoch":
                adjusted_eval_steps = None
            elif save_strategy == "steps":
                # 如果 save_strategy 是 steps，需要设置 eval_steps
                if adjusted_eval_steps is None:
                    adjusted_eval_steps = eval_steps if eval_steps else 230
            print(f"  ✓ 已调整: eval_strategy={adjusted_eval_strategy}, eval_steps={adjusted_eval_steps}")
    
    adjusted_logging_steps = logging_steps
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # 批次配置（针对小数据集优化）
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        
        # 训练轮数和学习率（小数据集特调）
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,  # 增强正则化，防止过拟合
        max_grad_norm=max_grad_norm,
        
        # 优化器配置
        # 使用 8-bit 优化器可以显著减少显存占用（优化器状态从 fp32 压缩到 8-bit）
        # 对于 4-bit 量化模型，8-bit 优化器是很好的选择
        optim="adamw_bnb_8bit",  # 使用 8-bit AdamW 优化器（节省显存）
        # optim="adamw_torch",  # 标准 AdamW 优化器（如果 8-bit 有问题，可以改回这个）
        lr_scheduler_type="cosine",  # 余弦学习率调度
        
        # 精度配置
        # [关键修改] 开启 BF16，A5880 支持且推荐（PyTorch >= 1.13）
        # 注意：BF16 与 4-bit 量化兼容，可以同时使用以进一步节省显存和加速训练
        bf16=bf16,  # 开启 BF16（A5880 硬件支持，加速训练）
        fp16=fp16,  # 禁用 fp16（4-bit 量化模型不支持 fp16 混合精度训练）
        
        # 日志和保存配置（高频监控，针对小数据集）
        logging_dir=f"{output_dir}/logs",
        logging_steps=adjusted_logging_steps,
        save_steps=adjusted_save_steps,
        eval_steps=adjusted_eval_steps,
        save_total_limit=save_total_limit,
        save_strategy=save_strategy,  # 支持 "steps" 或 "epoch"
        eval_strategy=adjusted_eval_strategy,  # 支持 "steps"、"epoch" 或 "no"
        
        # 其他配置
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=remove_unused_columns,
        report_to="tensorboard",  # 使用TensorBoard记录
        load_best_model_at_end=load_best_model_at_end if val_dataset else False,  # 只有在有验证集时才生效
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False,
        
        # 梯度检查点（节省显存）
        gradient_checkpointing=True,
        
        # [新增] 针对 DDP 的优化（如果以后用多卡）
        ddp_find_unused_parameters=False,
        
        # 数据加载配置
        dataloader_pin_memory=True,
        
        # [A5880 优化] 提高数据加载效率
        # persistent_workers: 持久化workers，减少重复初始化开销（需要 dataloader_num_workers > 0）
        dataloader_persistent_workers=True if dataloader_num_workers > 0 else False,
    )
    
    print("✓ 训练参数配置完成")
    print(f"  - 有效批次大小: {per_device_train_batch_size * gradient_accumulation_steps}")
    print(f"  - 训练轮数: {num_train_epochs} (小数据集需要更多epoch)")
    print(f"  - 学习率: {learning_rate} (降低LR，防止梯度波动)")
    print(f"  - 预热比例: {warmup_ratio} (增加预热时间)")
    print(f"  - 权重衰减: {weight_decay} (增强正则化)")
    print(f"  - BF16: {bf16} ({'已开启' if bf16 else '已关闭'})")
    print(f"  - 日志步数: {adjusted_logging_steps}")
    print(f"  - 保存步数: {adjusted_save_steps}")
    if val_dataset:
        print(f"  - 验证步数: {adjusted_eval_steps}")
        print(f"  - 加载最佳模型: {load_best_model_at_end}")
    
    # ========== 7. 创建Trainer ==========
    print("\n正在创建Trainer...")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("✓ Trainer创建完成")
    
    # ========== 8. 开始训练 ==========
    print("\n" + "=" * 80)
    print("开始训练...")
    print("=" * 80)
    
    # 训练前检查
    print("\n训练前检查:")
    print(f"  - 可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  - 总参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 计算可训练参数占比
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_ratio = trainable_params / total_params * 100
    print(f"  - 可训练参数占比: {trainable_ratio:.2f}%")
    
    # [关键模块梯度检查]
    print("\n[关键模块梯度检查]")
    
    # 检查 Projector（Stage 3 必须参与训练）
    if hasattr(model, 'multi_modal_projector'):
        projector_trainable = any(p.requires_grad for p in model.multi_modal_projector.parameters())
        projector_param_count = sum(p.numel() for p in model.multi_modal_projector.parameters())
        projector_trainable_count = sum(p.numel() for p in model.multi_modal_projector.parameters() if p.requires_grad)
        print(f"  - Projector 参与训练: {projector_trainable} (Stage 3 必须为 True)")
        print(f"    Projector 总参数: {projector_param_count:,}")
        print(f"    Projector 可训练参数: {projector_trainable_count:,}")
        
        # [新增] 打印当前的 rgb_scale 和 polar_scale 值（应该是从 Stage 2 加载的值）
        if hasattr(model.multi_modal_projector, 'rgb_scale'):
            rgb_scale_current = model.multi_modal_projector.rgb_scale.item()
            print(f"    RGB scale (当前值): {rgb_scale_current:.6f}")
        if hasattr(model.multi_modal_projector, 'polar_scale'):
            polar_scale_current = model.multi_modal_projector.polar_scale.item()
            print(f"    Polar scale (当前值): {polar_scale_current:.6f}")
        
        if not projector_trainable:
            print("    ⚠ 警告: Projector 未参与训练！这可能导致模型无法学习视觉-语言对齐。")
    else:
        print("  - ⚠ 警告: 未找到 multi_modal_projector，无法检查")
    
    # 检查 RGB Tower（应该被冻结）
    if hasattr(model, 'vision_tower_rgb'):
        rgb_tower_trainable = any(p.requires_grad for p in model.vision_tower_rgb.parameters())
        print(f"  - RGB Tower 参与训练: {rgb_tower_trainable} (应该为 False，已冻结)")
        if rgb_tower_trainable and freeze_rgb_tower:
            print("    ⚠ 警告: RGB Tower 应该被冻结但未冻结！")
    else:
        print("  - RGB Tower: 未找到（可能是通过其他方式加载的）")
    
    # 检查 Polar Tower（根据 freeze_polar_tower 设置）
    if hasattr(model, 'vision_tower_polar'):
        polar_tower_trainable = any(p.requires_grad for p in model.vision_tower_polar.parameters())
        print(f"  - Polar Tower 参与训练: {polar_tower_trainable} (freeze_polar_tower={freeze_polar_tower})")
    else:
        print("  - Polar Tower: 未找到")
    
    # 检查 LLM（应该通过 LoRA 参与训练）
    if hasattr(model, 'language_model'):
        llm_trainable = sum(p.numel() for p in model.language_model.parameters() if p.requires_grad)
        llm_total = sum(p.numel() for p in model.language_model.parameters())
        llm_ratio = (llm_trainable / llm_total * 100) if llm_total > 0 else 0
        print(f"  - LLM 可训练参数: {llm_trainable:,} / {llm_total:,} ({llm_ratio:.2f}%)")
        print(f"    (通过 LoRA 微调，只有 LoRA adapter 参数可训练)")
    else:
        print("  - LLM: 未找到 language_model 属性")
    
    # 开始训练
    try:
        train_result = trainer.train()
        
        print("\n" + "=" * 80)
        print("训练完成！")
        print("=" * 80)
        print(f"训练损失: {train_result.training_loss:.4f}")
        
        # 保存最终模型
        print("\n正在保存最终模型...")
        # [关键修复] 对于自定义模型（PolarLlava），trainer.save_model() 不会保存内部的 PEFT 适配器
        # 需要显式保存 LoRA 适配器
        from peft import PeftModel
        if isinstance(model.language_model, PeftModel):
            print("正在保存 LoRA 适配器...")
            model.language_model.save_pretrained(output_dir)
            print(f"  ✓ LoRA 适配器已保存到: {output_dir}")
            print(f"    - adapter_model.safetensors (或 adapter_model.bin)")
            print(f"    - adapter_config.json")
        else:
            # 如果不是 PEFT 模型，使用 trainer.save_model()（通常不会发生）
            trainer.save_model()
        
        tokenizer.save_pretrained(output_dir)
        print(f"✓ Tokenizer已保存到: {output_dir}")
        
        # 额外保险：手动保存视觉模块的权重
        # 对于PolarLlava外层的multi_modal_projector和vision_tower_polar，
        # 需要手动保存以确保完整保存
        print("\n正在额外保存视觉模块权重...")
        try:
            # 保存多模态投影器
            projector_path = os.path.join(output_dir, "projector.pth")
            torch.save(model.multi_modal_projector.state_dict(), projector_path)
            print(f"  ✓ 多模态投影器已保存: {projector_path}")
            
            # 保存偏振流（如果可训练）
            if not freeze_polar_tower:
                polar_tower_path = os.path.join(output_dir, "polar_tower.pth")
                torch.save(model.vision_tower_polar.state_dict(), polar_tower_path)
                print(f"  ✓ 偏振流已保存: {polar_tower_path}")
            else:
                print("  ⚠ 偏振流已冻结，跳过保存（使用预训练权重）")
            
            # 保存模型配置信息（用于加载时参考）
            config_info = {
                "clip_model_name": clip_model_name,
                "polar_backbone": polar_backbone,
                "llm_model_name": llm_model_name,
                "image_token_id": model.image_token_id,
                "freeze_rgb_tower": freeze_rgb_tower,
                "freeze_polar_tower": freeze_polar_tower,
                "freeze_polar_layers": freeze_polar_layers,
            }
            import json
            config_path = os.path.join(output_dir, "model_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_info, f, indent=2, ensure_ascii=False)
            print(f"  ✓ 模型配置已保存: {config_path}")
            
        except Exception as e:
            print(f"  ⚠ 保存视觉模块时出现警告: {e}")
            print("  （这不会影响LLM和LoRA权重的保存）")
        
        print(f"\n✓ 所有模型组件已保存到: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        print("正在保存检查点...")
        # [关键修复] 显式保存 LoRA 适配器
        from peft import PeftModel
        if isinstance(model.language_model, PeftModel):
            print("正在保存 LoRA 适配器...")
            model.language_model.save_pretrained(output_dir)
            print(f"  ✓ LoRA 适配器已保存到: {output_dir}")
        else:
            trainer.save_model()
        
        tokenizer.save_pretrained(output_dir)
        print(f"✓ Tokenizer检查点已保存到: {output_dir}")
        
        # 也保存视觉模块（如果可能）
        try:
            projector_path = os.path.join(output_dir, "projector.pth")
            torch.save(model.multi_modal_projector.state_dict(), projector_path)
            if not freeze_polar_tower:
                polar_tower_path = os.path.join(output_dir, "polar_tower.pth")
                torch.save(model.vision_tower_polar.state_dict(), polar_tower_path)
            print(f"✓ 视觉模块检查点已保存")
        except Exception as e:
            print(f"⚠ 保存视觉模块时出现警告: {e}")
        
        print(f"✓ 检查点已保存到: {output_dir}")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n训练脚本执行完成！")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PolarVLM训练脚本")
    
    # 数据路径
    parser.add_argument("--train_json", type=str, default="merged_stage3_data.json",
                        help="训练集JSON文件路径（Stage 3 默认使用合并后的 merged_stage3_data.json）")
    parser.add_argument("--val_json", type=str, default=None,
                        help="验证集JSON文件路径（可选）")
    parser.add_argument("--rgb_root", type=str, default="/openbayes/input/input0/rgb",
                        help="RGB图像根目录（用于非裁剪图像，如 rgb/04/0002_rgb.png）")
    parser.add_argument("--polar_root", type=str, default="/openbayes/input/input0/polar",
                        help="偏振图像根目录（用于非裁剪图像，如 polar/04/0002_000.png）")
    parser.add_argument("--data_root", type=str, default=None,
                        help="数据根目录（用于解析 crop 路径和 GT 路径，默认使用 rgb_root 的父目录）")
    
    # 模型配置
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-large-patch14",
                        help="CLIP模型名称")
    parser.add_argument("--polar_backbone", type=str, default="google/vit-base-patch16-224-in21k",
                        help="⚠️ 已废弃：偏振流backbone（仅作为备用选项，当 --vae_model_path 不存在时使用）。如果提供了 --vae_model_path，此参数将被完全忽略。")
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="⚠️ 已废弃：Stage 1 检查点路径（改用 --vae_model_path）")
    parser.add_argument("--stage2_checkpoint", type=str, default=None,
                        help="Stage 2 检查点路径（可选，用于加载投影器权重）")
    parser.add_argument("--vae_model_path", type=str, default="/openbayes/input/input0/models/sd-vae-ft-mse",
                        help="VAE 模型路径（用于加载 VAE 编码器）。如果提供此路径，--polar_backbone 将被忽略。")
    parser.add_argument("--llm_model_name", type=str, default="/openbayes/input/input0/models/Meta-Llama-3-8B-Instruct",
                        help="LLM 模型路径（Instruct 版本）")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="使用4-bit量化")
    parser.add_argument("--load_in_8bit", action="store_true", default=False,
                        help="使用8-bit量化")
    
    # LoRA配置
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    
    # 训练配置
    parser.add_argument("--output_dir", type=str, default="../autodl-tmp/checkpoints/polarvlm",
                        help="输出目录（默认保存到 autodl-tmp，与 train 目录并列）")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="每设备训练批次大小（A5880 优化：Batch=8，充分利用48G显存）")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="梯度累积步数（有效批次 = 8 * 2 = 16，保持不变）")
    parser.add_argument("--num_train_epochs", type=int, default=10,
                        help="训练轮数（建议10个epoch，防止过拟合，约4700步足够收敛）")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="学习率（小数据集建议1e-4，防止梯度波动）")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="预热比例（小数据集建议0.1，增加预热时间）")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="权重衰减（小数据集建议0.05，增强正则化）")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志记录步数（约每半个epoch记录一次）")
    parser.add_argument("--save_steps", type=int, default=None,
                        help="保存步数（默认 None，将在训练中途保存一次；也可手动指定步数，建议230）")
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["steps", "epoch"],
                        help="保存策略：'steps'（按步数）或 'epoch'（按轮数）")
    parser.add_argument("--eval_steps", type=int, default=230,
                        help="验证步数（约每半个epoch验证一次，与save_steps对齐）")
    parser.add_argument("--eval_strategy", type=str, default="steps", choices=["steps", "epoch", "no"],
                        help="验证策略：'steps'（按步数）、'epoch'（按轮数）或 'no'（不验证）")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="保存checkpoint数量限制（默认3，保留最佳+最新+中间共3个checkpoint）")
    parser.add_argument("--load_best_model_at_end", action="store_true", default=False,
                        help="训练结束后加载最佳模型（需要验证集，使用 --load_best_model_at_end 启用，不需要加 True）")
    
    # 精度配置
    # BF16 默认开启，使用 --no_bf16 关闭
    parser.add_argument("--no_bf16", action="store_true", default=False,
                        help="禁用 BF16 混合精度训练（默认开启，使用 --no_bf16 关闭）")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="使用 FP16 混合精度训练（4-bit 量化模型不支持，默认关闭）")
    
    # 数据加载配置
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="数据加载器工作进程数（A5880 优化：默认4，充分利用多核CPU）")
    parser.add_argument("--remove_unused_columns", action="store_true", default=False,
                        help="移除未使用的列（默认False，保留所有列包括图像）")
    
    # 冻结策略
    parser.add_argument("--freeze_rgb_tower", action="store_true", default=True,
                        help="冻结RGB流")
    parser.add_argument("--freeze_polar_tower", action="store_true", default=True,
                        help="冻结偏振流（默认启用，只训练 Projector + LoRA）")
    parser.add_argument("--no_freeze_polar_tower", action="store_false", dest="freeze_polar_tower",
                        help="解冻偏振流（使用此参数可以训练 Polar Tower）")
    parser.add_argument("--freeze_polar_layers", type=int, default=0,
                        help="冻结偏振流的前N层")
    
    # Hugging Face token
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face token（用于下载模型，也可通过环境变量HF_TOKEN设置）")
    
    args = parser.parse_args()
    
    # 运行主函数
    main(
        train_json=args.train_json,
        val_json=args.val_json,
        rgb_root=args.rgb_root,
        polar_root=args.polar_root,
        data_root=getattr(args, 'data_root', None),  # [Stage 3 新增] 传递 data_root 参数
        clip_model_name=args.clip_model_name,
        polar_backbone=args.polar_backbone,
        llm_model_name=args.llm_model_name,
        load_in_4bit=args.load_in_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy=getattr(args, 'save_strategy', 'steps'),
        eval_steps=args.eval_steps,
        eval_strategy=getattr(args, 'eval_strategy', 'steps'),
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=getattr(args, 'load_best_model_at_end', False),
        bf16=not args.no_bf16,  # [关键修改] 默认 True（如果未指定 --no_bf16），开启 BF16（A5880 支持且推荐，PyTorch >= 1.13）
        fp16=getattr(args, 'fp16', False),  # 默认 False（4-bit 量化模型不支持 fp16 混合精度训练）
        dataloader_num_workers=args.dataloader_num_workers,  # [新增] 数据加载器工作进程数
        remove_unused_columns=getattr(args, 'remove_unused_columns', False),  # [新增] 移除未使用的列
        freeze_rgb_tower=args.freeze_rgb_tower,
        freeze_polar_tower=args.freeze_polar_tower,
        freeze_polar_layers=args.freeze_polar_layers,
        stage1_checkpoint=getattr(args, 'stage1_checkpoint', None),  # 已废弃，保留以兼容旧代码
        stage2_checkpoint=getattr(args, 'stage2_checkpoint', None),  # Stage 2 检查点路径
        vae_model_path=getattr(args, 'vae_model_path', '/openbayes/input/input0/models/sd-vae-ft-mse'),  # VAE 模型路径
        hf_token=args.hf_token,
    )


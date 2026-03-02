"""
Stage 1 训练脚本：偏振编码器预训练（MAE）

目标：
- 使用 Masked Autoencoder (MAE) 预训练偏振编码器
- 学习偏振图像的表示，理解偏振物理特性
- 输出：训练好的编码器权重，用于 Stage 2

模型架构：
- 使用 ViTMAE（Vision Transformer Masked Autoencoder）
- 输入：3通道物理参数（DoLP, sin(2*AoLP), cos(2*AoLP)）
- 直接使用3通道预训练权重（无需扩展策略）

损失函数：
- 重建损失（MSE）：预测被mask的patch

输出：
- 编码器权重保存到 checkpoints/stage1_encoder/pytorch_model.bin

注意：
- 使用 process_polar_images 函数将4张角度图转换为4通道物理参数，然后只使用后3个通道（DoLP, sin(2*AoLP), cos(2*AoLP)）
- 3通道输入直接使用3通道预训练权重（无需扩展策略）
- Intensity通道被移除，因为Stage 2会使用CLIP提取RGB特征（包含光强信息）
"""

import os
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader
from transformers import (
    ViTMAEConfig,
    ViTMAEForPreTraining,
    TrainingArguments,
    Trainer,
)
from dataset_stage1 import PolarMAEDataset, collate_fn_stage1
import warnings
import json
from datetime import datetime
warnings.filterwarnings("ignore")


def create_mae_model(
    model_name: str = "facebook/vit-mae-base",
    image_size: int = 224,
    patch_size: int = 16,
    num_channels: int = 3,  # 偏振图像：3通道（DoLP, sin(2*AoLP), cos(2*AoLP)），移除Intensity
    norm_pix_loss: bool = False,  # 归一化像素损失（关键修复：对偏振数据应设为 False）
    hf_token: Optional[str] = None,
) -> ViTMAEForPreTraining:
    """
    创建 MAE 模型（3通道输入，针对偏振物理数据优化）
    
    Args:
        model_name: 预训练 MAE 模型名称（默认使用 facebook/vit-mae-base）
        image_size: 图像尺寸（默认224）
        patch_size: Patch 尺寸（默认16）
        num_channels: 输入通道数（默认3，物理参数：DoLP, sin(2*AoLP), cos(2*AoLP)）
                     注意：Intensity通道已移除，因为Stage 2会使用CLIP提取RGB特征
        norm_pix_loss: 是否启用归一化像素损失（默认False，关键优化）
                       设置为 True 时，MAE 会在每个 patch 内计算局部统计量进行归一化，
                       不依赖全局统计量，适合统计特性未知的物理数据
        hf_token: Hugging Face token（用于下载模型）
    
    Returns:
        MAE 模型（直接使用3通道预训练权重，无需扩展策略）
    """
    print("=" * 80)
    print("创建 MAE 模型（Stage 1：偏振编码器预训练）")
    print("=" * 80)
    
    # 读取模型配置
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    load_kwargs = {}
    if token:
        load_kwargs["token"] = token
        print(f"✓ 使用 Hugging Face token 加载模型")
    
    try:
        # 加载预训练配置
        config = ViTMAEConfig.from_pretrained(model_name, **load_kwargs)
        
        # 修改配置以适配3通道输入（DoLP, sin, cos）
        config.num_channels = num_channels
        config.image_size = image_size
        config.patch_size = patch_size
        
        # ========== 关键修复：禁用归一化像素损失（对偏振数据）==========
        # 
        # norm_pix_loss=True 的工作原理：
        # 1. MAE 在计算重建损失时，会对目标像素进行归一化：
        #    normalized_pixel = (pixel - mean) / std
        #    其中 mean 和 std 是在每个 patch 内计算的
        # 2. 这样做的目的（对于 ImageNet RGB）：
        #    - 忽略光照变化，专注形状特征
        #    - 适应不同场景的光照条件
        # 
        # ⚠️ 为什么对偏振数据应该设为 False：
        # 
        # 1. **DoLP（偏振度）的绝对物理意义**：
        #    - DoLP 取值严格在 [0, 1]，具有明确的物理含义
        #    - 低偏振区域（例如 0.05）表示"几乎没有偏振"
        #    - 高偏振区域（例如 0.8）表示"高度偏振"
        #    - 如果使用 norm_pix_loss=True：
        #      * 低偏振 patch（均值 0.05，标准差 0.01）归一化后变成标准正态分布
        #      * 高偏振 patch（均值 0.8，标准差 0.1）归一化后也变成标准正态分布
        #      * 模型会丢失"偏振强度"这个绝对物理量的概念
        #      * 无法区分"低偏振"和"高偏振"的绝对差异
        # 
        # 2. **Intensity（光强）的绝对意义**：
        #    - 虽然 Intensity 类似 RGB，归一化可能有助于忽略光照变化
        #    - 但偏振数据的 Intensity 分布相对稳定，不需要归一化
        # 
        # 3. **sin(2*AoLP) 和 cos(2*AoLP)**：
        #    - 已经是归一化的表示（范围 [0, 1]）
        #    - 归一化可能会影响角度信息的准确性
        # 
        # ✅ 解决方案：norm_pix_loss=False
        # - 让模型直接预测物理数值，保留物理量的绝对意义
        # - 模型学习的是"DoLP=0.8 表示高偏振"，而不是"这个 patch 的偏振度相对较高"
        # - 这对于下游任务（如 Stage 2/3）提取有意义的特征至关重要
        # 
        config.norm_pix_loss = norm_pix_loss
        
        # ========== 关键优化：调整掩码率（Mask Ratio）==========
        # 
        # 默认 mask_ratio = 0.75（遮住 75% 的 patch）是 ImageNet（120万张图）的标准
        # 对于垂直领域数据集（6312 张图），0.75 的掩码率太高，模型可能"摆烂"学不到东西
        # 
        # 建议：降低到 0.5 或 0.6
        # - 相当于降低了"完形填空"的难度
        # - 让模型先学会简单的特征，再逐步提升难度
        # - 对于小数据集，更低的掩码率有助于稳定训练
        # 
        # 注意：ViTMAEConfig 的默认 mask_ratio 是 0.75
        # 我们将其调整为 0.6，在难度和训练效率之间取得平衡
        if hasattr(config, 'mask_ratio'):
            config.mask_ratio = 0.5  # 从默认 0.75 降低到 0.6
            print(f"  - 掩码率: {config.mask_ratio} ✓ (关键优化：从 0.75 降低到 0.5，适应小数据集)")
        else:
            print("  ⚠ 警告: ViTMAEConfig 不支持 mask_ratio 参数，将使用默认值 0.75")
        
        print(f"✓ 模型配置:")
        print(f"  - 图像尺寸: {config.image_size}")
        print(f"  - Patch 尺寸: {config.patch_size}")
        if config.num_channels == 3:
            print(f"  - 输入通道数: {config.num_channels} (DoLP, sin(2*AoLP), cos(2*AoLP))")
        else:
            print(f"  - 输入通道数: {config.num_channels}")
        print(f"  - 隐藏维度: {config.hidden_size}")
        print(f"  - 注意力头数: {config.num_attention_heads}")
        print(f"  - Transformer 层数: {config.num_hidden_layers}")
        if config.norm_pix_loss:
            print(f"  - 归一化像素损失: {config.norm_pix_loss} ⚠ (警告：可能丢失 DoLP 的绝对物理意义)")
        else:
            print(f"  - 归一化像素损失: {config.norm_pix_loss} ✓ (关键修复：保留偏振物理量的绝对意义)")
        
        # ========== 关键：直接加载3通道预训练权重 ==========
        # 
        # 优势：
        # - facebook/vit-mae-base 的预训练权重是3通道（RGB）
        # - 我们的输入是3通道（DoLP, sin(2*AoLP), cos(2*AoLP)）
        # - 可以直接使用预训练权重，无需扩展策略
        # - 权重加载更简单、更稳定
        # 
        # 注意：
        # - Intensity通道已移除，因为Stage 2会使用CLIP提取RGB特征（包含光强信息）
        # - 3通道聚焦于偏振特有的信息（DoLP和AoLP），训练目标更一致
        # 
        print("\n正在加载预训练权重（3通道，直接使用）...")
        
        # 直接加载3通道预训练权重
        model = ViTMAEForPreTraining.from_pretrained(
            model_name,
            config=config,
            **load_kwargs
        )
        
        print("✓ 已加载3通道预训练权重（无需扩展策略）")
        
        # ========== 层级冻结策略（Partial Fine-Tuning）==========
        # 
        # 策略说明：
        # - 小数据集（6312张图）不适合全量微调，容易过拟合
        # - 利用 ImageNet 预训练权重的通用视觉特征（底层：边缘、纹理）
        # - 只微调适应偏振数据的部分：
        #   1. Embeddings：输入保持3通道，但需要适应偏振数据特征
        #   2. 高层 Encoder 层（9-11）：适应高层次的偏振语义特征
        #   3. Decoder：重建目标是偏振数据（非RGB），必须重新学习
        # 
        print("\n" + "=" * 80)
        print("应用层级冻结策略（Partial Fine-Tuning）")
        print("=" * 80)
        
        # Step 1: 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False
        print("✓ Step 1: 已冻结所有参数")
        
        # Step 2: 解冻 Embeddings（输入维度保持3通道，但需要适应偏振数据）
        for param in model.vit.embeddings.parameters():
            param.requires_grad = True
        print("✓ Step 2: 已解冻 Embeddings（学习偏振数据特征）")
        
        # Step 3: 解冻最后3层 Encoder（layer 9, 10, 11）
        # 底层（0-8）保持冻结：提取通用视觉特征（边缘、纹理）
        # 高层（9-11）解冻：适应高层次的偏振语义特征
        num_encoder_layers = len(model.vit.encoder.layer)
        if num_encoder_layers >= 12:
            # ViT-Base 有12层，解冻最后3层（9, 10, 11）
            for i in range(9, 12):
                for param in model.vit.encoder.layer[i].parameters():
                    param.requires_grad = True
            print(f"✓ Step 3: 已解冻最后3层 Encoder（layer 9-11，共{num_encoder_layers}层）")
        else:
            # 如果层数不足12层，解冻最后3层（或所有层，如果少于3层）
            start_layer = max(0, num_encoder_layers - 3)
            for i in range(start_layer, num_encoder_layers):
                for param in model.vit.encoder.layer[i].parameters():
                    param.requires_grad = True
            print(f"✓ Step 3: 已解冻最后{num_encoder_layers - start_layer}层 Encoder（layer {start_layer}-{num_encoder_layers-1}，共{num_encoder_layers}层）")
        
        # Step 4: 解冻整个 Decoder（重建目标是偏振数据，非RGB，必须重新学习）
        for param in model.decoder.parameters():
            param.requires_grad = True
        print("✓ Step 4: 已解冻 Decoder（学习偏振数据重建）")
        
        # 计算并显示参数统计
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_percentage = 100 * trainable_params / total_params
        
        print("\n" + "=" * 80)
        print("参数统计（层级冻结后）")
        print("=" * 80)
        print(f"  - 可训练参数: {trainable_params:,}")
        print(f"  - 总参数: {total_params:,}")
        print(f"  - 可训练参数占比: {trainable_percentage:.2f}%")
        print("=" * 80)
        
        # 超参数建议（注释）
        # 注意：由于只训练部分参数，学习率可以适当提高（例如 2e-4 或 5e-4）
        # 但建议先使用默认的 1e-4，根据训练效果再调整
        
    except Exception as e:
        print(f"❌ 创建模型时出现错误: {e}")
        raise
    
    return model


def main(
    # 数据路径
    polar_root: str = "data/polar",
    use_pt_data: bool = False,  # 是否使用预处理的 .pt 文件（快速模式）
    pt_root: Optional[str] = None,  # .pt 文件根目录（如果 use_pt_data=True）
    
    # 模型配置
    model_name: str = "facebook/vit-mae-base",
    image_size: int = 224,
    patch_size: int = 16,
    num_channels: int = 3,  # 偏振图像：3通道（DoLP, sin(2*AoLP), cos(2*AoLP)），移除Intensity
    norm_pix_loss: bool = False,  # 归一化像素损失（关键修复：对偏振数据应设为 False，保留物理量的绝对意义）
    
    # 训练配置
    output_dir: str = "../autodl-tmp/checkpoints/stage1_encoder",
    per_device_train_batch_size: int = 16,
    num_train_epochs: int = 100,
    learning_rate: float = 5e-4,  # [优化] 从 1e-4 提高到 5e-4，因为部分冻结模型需要更大学习率
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.05,
    max_grad_norm: float = 1.0,
    
    # 其他配置
    logging_steps: int = 50,
    save_steps: Optional[int] = 1000,
    save_strategy: str = "steps",  # 保存策略："steps"、"epoch" 或 "no"
    save_total_limit: int = 3,
    dataloader_num_workers: int = 4,
    fp16: bool = True,  # MAE 可以使用 fp16
    
    # 验证/评估配置
    evaluation_strategy: str = "no",  # 评估策略："no"、"steps" 或 "epoch"
    eval_steps: int = None,  # 评估步数（仅在 evaluation_strategy="steps" 时有效，默认与 save_steps 相同）
    per_device_eval_batch_size: int = None,  # 验证批次大小（默认与训练批次大小相同）
    load_best_model_at_end: bool = False,  # 训练结束时加载最佳模型
    metric_for_best_model: str = "loss",  # 用于选择最佳模型的指标
    
    # Hugging Face token
    hf_token: Optional[str] = None,
):
    """
    主训练函数
    """
    print("=" * 80)
    print("PolarVLM Stage 1: 偏振编码器预训练（MAE）")
    print("=" * 80)
    
    # ========== 1. 设置设备 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    
    # ========== 2. 创建数据集 ==========
    print("\n" + "=" * 80)
    print("正在创建数据集...")
    print("=" * 80)
    
    # ========== 2.1 创建训练集 ==========
    # 
    # 路径逻辑：
    # - 如果 use_pt_data=True：
    #   - 如果提供了 pt_root：使用 {pt_root}/train
    #   - 否则：使用 {polar_root}/train
    # - 如果 use_pt_data=False：
    #   - 训练集路径：polar_root（传统模式）
    # 
    if use_pt_data:
        # PT模式：优先使用 pt_root（如果提供），否则使用 polar_root
        base_root = Path(pt_root) if pt_root else Path(polar_root)
        train_pt_root = str(base_root / "train")
        train_polar_root = train_pt_root  # PT模式下，polar_root 和 pt_root 指向同一路径
    else:
        # PNG模式：训练集直接在 polar_root
        train_polar_root = polar_root
        train_pt_root = None
    
    train_dataset = PolarMAEDataset(
        polar_root=train_polar_root,
        image_size=image_size,
        is_train=True,
        use_pt_data=use_pt_data,  # 是否使用预处理的 .pt 文件
        pt_root=train_pt_root,  # .pt 文件根目录（训练集）
    )
    
    print(f"✓ 训练集大小: {len(train_dataset)}")
    
    # ========== 2.2 创建验证集（如果启用评估）==========
    eval_dataset = None
    if evaluation_strategy != "no":
        if use_pt_data:
            # PT模式：验证集在 {polar_root}/val 或 {pt_root}/val
            # 优先使用 pt_root（如果提供），否则使用 polar_root
            base_root = Path(pt_root) if pt_root else Path(polar_root)
            val_pt_root = str(base_root / "val")
            val_polar_root = val_pt_root  # PT模式下，polar_root 和 pt_root 指向同一路径
            
            # 检查验证集目录是否存在
            val_path = Path(val_pt_root)
            if val_path.exists():
                eval_dataset = PolarMAEDataset(
                    polar_root=val_polar_root,
                    image_size=image_size,
                    is_train=False,  # 验证集不应用数据增强
                    use_pt_data=use_pt_data,
                    pt_root=val_pt_root,
                )
                print(f"✓ 验证集大小: {len(eval_dataset)}")
            else:
                print(f"⚠ 警告: 验证集目录不存在: {val_path}")
                print("  将跳过验证（设置 evaluation_strategy='no'）")
                evaluation_strategy = "no"
        else:
            # PNG模式：暂时不支持验证集（需要用户手动准备）
            print("⚠ 警告: PNG模式暂不支持自动验证集加载")
            print("  将跳过验证（设置 evaluation_strategy='no'）")
            evaluation_strategy = "no"
    
    # ========== 3. 创建模型 ==========
    print("\n" + "=" * 80)
    print("正在创建模型...")
    print("=" * 80)
    
    model = create_mae_model(
        model_name=model_name,
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,  # 偏振图像：3通道（DoLP, sin(2*AoLP), cos(2*AoLP)），移除Intensity
        norm_pix_loss=norm_pix_loss,  # 归一化像素损失（关键优化）
        hf_token=hf_token,
    )
    
    # 移动到设备
    model = model.to(device)
    
    # ========== 4. 创建 DataLoader ==========
    # 注意：如果使用 Trainer，DataLoader 会自动创建，这里可以删除
    # 但为了计算总步数，我们保留 train_loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_train_batch_size,
        shuffle=True,
        num_workers=dataloader_num_workers,
        collate_fn=collate_fn_stage1,
        pin_memory=True,
    )
    
    # ========== 5. 配置训练参数 ==========
    print("\n" + "=" * 80)
    print("正在配置训练参数...")
    print("=" * 80)
    
    # 计算总步数
    total_steps = len(train_loader) * num_train_epochs
    print(f"  - 训练集大小: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"  - 验证集大小: {len(eval_dataset)}")
    print(f"  - 训练批次大小: {per_device_train_batch_size}")
    eval_batch_size = per_device_eval_batch_size if per_device_eval_batch_size else per_device_train_batch_size
    print(f"  - 验证批次大小: {eval_batch_size}")
    print(f"  - 训练轮数: {num_train_epochs}")
    print(f"  - 总步数: ~{total_steps}")
    print(f"  - 学习率: {learning_rate}")
    print(f"  - 预热比例: {warmup_ratio}")
    print(f"  - 权重衰减: {weight_decay}")
    if evaluation_strategy != "no":
        print(f"  - 评估策略: {evaluation_strategy}")
        if load_best_model_at_end:
            print(f"  - 自动加载最佳模型: ✓ (基于 {metric_for_best_model})")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # 批次配置
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size if per_device_eval_batch_size else per_device_train_batch_size,
        
        # 训练轮数和学习率
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        
        # 优化器配置
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        
        # 精度配置
        fp16=fp16,
        
        # 日志和保存配置
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,
        save_steps=save_steps if save_strategy == "steps" else None,  # 如果 save_strategy 不是 "steps"，则 save_steps 设为 None
        save_strategy=save_strategy,  # 使用参数传入的保存策略
        save_total_limit=save_total_limit,
        
        # 评估配置
        # 注意：TrainingArguments 使用 eval_strategy 而不是 evaluation_strategy
        eval_strategy=evaluation_strategy,  # 评估策略："no"、"steps" 或 "epoch"
        eval_steps=eval_steps if eval_steps is not None else (save_steps if evaluation_strategy == "steps" else None),  # 如果 evaluation_strategy="steps"，使用 eval_steps 或 save_steps
        
        # 如果 load_best_model_at_end=True，需要确保 eval_strategy 和 save_strategy 匹配
        load_best_model_at_end=load_best_model_at_end,  # 训练结束时加载最佳模型
        # 注意：metric_for_best_model 应该使用 "loss" 而不是 "eval_loss"
        # 但 Trainer 内部可能会将 "loss" 转换为 "eval_loss"，所以我们需要确保 metrics 中同时有这两个键
        # 我们在 evaluation_loop 中会同时注入 "loss" 和 "eval_loss"
        metric_for_best_model="loss" if load_best_model_at_end else metric_for_best_model,  # 用于选择最佳模型的指标
        greater_is_better=False,  # loss 越小越好
        
        # 其他配置
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=False,
        report_to="tensorboard",
        
        # 梯度检查点（节省显存）
        gradient_checkpointing=True,
        
        # 数据加载配置
        dataloader_pin_memory=True,
    )
    
    # ========== 6. 定义训练函数 ==========
    # 创建验证loss日志文件路径
    eval_loss_log_file = os.path.join(output_dir, "eval_loss_log.json")
    eval_loss_log_txt = os.path.join(output_dir, "eval_loss_log.txt")
    
    class MAETrainer(Trainer):
        """自定义 Trainer，处理 MAE 的输入格式"""
        
        def __init__(self, *args, eval_loss_log_file=None, eval_loss_log_txt=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.eval_loss_log_file = eval_loss_log_file
            self.eval_loss_log_txt = eval_loss_log_txt
            self.eval_loss_history = []  # 存储所有epoch的验证loss
        
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """
            计算 MAE 损失
            
            Args:
                model: MAE 模型
                inputs: 输入数据（包含 pixel_values）
                return_outputs: 是否返回模型输出
                num_items_in_batch: 批次中的项目数（新版本 transformers 传递的参数，可选）
            
            Returns:
                损失值（如果 return_outputs=True，还返回输出）
            """
            pixel_values = inputs["pixel_values"]  # (B, 3, H, W) - [DoLP, sin(2*AoLP), cos(2*AoLP)]
            
            # MAE 前向传播
            outputs = model(pixel_values=pixel_values)
            
            loss = outputs.loss
            
            return (loss, outputs) if return_outputs else loss
        
        def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
            """
            健壮的自定义评估循环，手动计算 loss 以避免 Trainer 的问题
            
            Args:
                dataloader: 数据加载器
                description: 描述
                prediction_loss_only: 是否只预测损失（此参数在此实现中不使用）
                ignore_keys: 忽略的键
                metric_key_prefix: 指标键前缀（默认 "eval"）
            
            Returns:
                EvalLoopOutput 对象
            
            关键修复：
            - 完全绕过父类的 loss 聚合逻辑（因为它失败了）
            - 手动遍历 dataloader 计算平均 loss
            - 确保 metrics 中始终包含 "loss" 和 "{metric_key_prefix}_loss"
            """
            # ========== 1. 初始化变量 ==========
            model = self._wrap_model(self.model, training=False)
            model.eval()
            
            total_loss = 0.0
            num_samples = 0
            
            # ========== 2. 手动遍历 dataloader ==========
            print(f"\n开始手动计算验证 loss...")
            for step, inputs in enumerate(dataloader):
                # 将 inputs 移动到设备
                inputs = self._prepare_inputs(inputs)
                
                # 前向传播（无梯度）
                with torch.no_grad():
                    # 调用 compute_loss 方法（它会调用 model 并返回 loss）
                    loss = self.compute_loss(model, inputs)
                    
                    # 累积 loss
                    # 处理 loss 是标量（mean）还是向量（reduction='none'）的情况
                    if isinstance(loss, torch.Tensor):
                        if loss.ndim == 0:
                            # 标量 loss（已经是对 batch 的平均值）
                            batch_size = inputs["pixel_values"].shape[0]
                            total_loss += loss.item() * batch_size
                            num_samples += batch_size
                        else:
                            # 向量 loss（每个样本一个 loss 值）
                            total_loss += loss.sum().item()
                            num_samples += loss.numel()
                    else:
                        # loss 是 Python float
                        batch_size = inputs["pixel_values"].shape[0]
                        total_loss += float(loss) * batch_size
                        num_samples += batch_size
                
                # 每100步打印一次进度
                if (step + 1) % 100 == 0:
                    print(f"  已处理 {step + 1} 个批次...")
            
            # ========== 3. 计算平均 loss ==========
            if num_samples > 0:
                avg_loss = total_loss / num_samples
            else:
                avg_loss = 0.0
                print("⚠ 警告: 没有样本，loss 设为 0.0")
            
            print(f"✓ 验证 loss 计算完成: {avg_loss:.6f} (基于 {num_samples} 个样本)")
            
            # ========== 4. 手动构造 metrics ==========
            metrics = {
                f"{metric_key_prefix}_loss": avg_loss,  # 例如 "eval_loss"
                "loss": avg_loss,  # metric_for_best_model 需要的键
            }
            
            # 如果 metric_key_prefix 是 "eval"，确保同时有 "eval_loss" 键
            if metric_key_prefix == "eval":
                metrics['eval_loss'] = avg_loss
            
            # 添加其他必要的键（以满足 Trainer 的期望）
            # 注意：这些是虚拟值，因为我们不关心速度指标
            metrics[f"{metric_key_prefix}_samples_per_second"] = 0.0
            metrics[f"{metric_key_prefix}_steps_per_second"] = 0.0
            
            # 添加 epoch（如果 state 中有）
            if hasattr(self.state, 'epoch'):
                metrics["epoch"] = self.state.epoch
            
            # ========== 5. 记录验证loss到文件 ==========
            loss_value = avg_loss  # 使用计算出的平均 loss
            if metric_key_prefix == "eval" and self.eval_loss_log_file and loss_value is not None:
                # 获取当前epoch（从state中获取）
                current_epoch = self.state.epoch if hasattr(self.state, 'epoch') else None
                current_step = self.state.global_step if hasattr(self.state, 'global_step') else None
                
                # 记录验证loss（使用 loss_value，而不是 eval_output.loss）
                log_entry = {
                    "epoch": current_epoch,
                    "step": current_step,
                    "eval_loss": float(loss_value),  # 使用 loss_value，而不是 eval_output.loss
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # 添加到历史记录
                self.eval_loss_history.append(log_entry)
                
                # 保存到JSON文件（追加模式）
                try:
                    # 读取现有日志（如果存在）
                    if os.path.exists(self.eval_loss_log_file):
                        with open(self.eval_loss_log_file, 'r', encoding='utf-8') as f:
                            log_data = json.load(f)
                            # 检查是否是旧格式（直接是列表）还是新格式（有metadata）
                            if isinstance(log_data, list):
                                # 旧格式：直接是列表，转换为新格式
                                log_data = {
                                    "description": "Stage 1 MAE训练验证loss记录",
                                    "logs": log_data
                                }
                            existing_logs = log_data.get("logs", [])
                    else:
                        log_data = {
                            "description": "Stage 1 MAE训练验证loss记录",
                            "logs": []
                        }
                        existing_logs = []
                    
                    # 添加新条目
                    existing_logs.append(log_entry)
                    log_data["logs"] = existing_logs
                    
                    # 写回文件
                    with open(self.eval_loss_log_file, 'w', encoding='utf-8') as f:
                        json.dump(log_data, f, indent=2, ensure_ascii=False)
                    
                    # 同时保存到文本文件（追加模式，便于查看）
                    with open(self.eval_loss_log_txt, 'a', encoding='utf-8') as f:
                        if current_epoch is not None:
                            f.write(f"Epoch {current_epoch:.2f} | Step {current_step} | Eval Loss: {loss_value:.6f} | {log_entry['timestamp']}\n")
                        else:
                            f.write(f"Step {current_step} | Eval Loss: {loss_value:.6f} | {log_entry['timestamp']}\n")
                    
                    print(f"\n✓ 验证loss已记录到: {self.eval_loss_log_file}")
                    if current_epoch is not None:
                        print(f"  Epoch {current_epoch:.2f} | Eval Loss: {loss_value:.6f}")
                    else:
                        print(f"  Step {current_step} | Eval Loss: {loss_value:.6f}")
                except Exception as e:
                    print(f"⚠ 警告: 记录验证loss时出错: {e}")
            
            # ========== 6. 返回标准的 EvalLoopOutput ==========
            from transformers.trainer_utils import EvalLoopOutput
            return EvalLoopOutput(
                predictions=None,  # 我们不需要 predictions
                label_ids=None,    # 我们不需要 label_ids
                metrics=metrics,   # 手动构造的 metrics
                num_samples=num_samples  # 样本数量
            )
    
    # ========== 7. 创建 Trainer ==========
    print("\n" + "=" * 80)
    print("正在创建 Trainer...")
    print("=" * 80)
    
    trainer = MAETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # 传入验证集（如果存在）
        data_collator=collate_fn_stage1,
        eval_loss_log_file=eval_loss_log_file if eval_dataset is not None else None,  # 只在有验证集时记录
        eval_loss_log_txt=eval_loss_log_txt if eval_dataset is not None else None,
    )
    
    # 初始化日志文件（如果启用验证）
    if eval_dataset is not None:
        # 创建日志文件头
        log_header = {
            "description": "Stage 1 MAE训练验证loss记录",
            "train_dataset_size": len(train_dataset),
            "eval_dataset_size": len(eval_dataset),
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "logs": []
        }
        
        # 初始化JSON日志文件
        with open(eval_loss_log_file, 'w', encoding='utf-8') as f:
            json.dump(log_header, f, indent=2, ensure_ascii=False)
        
        # 初始化文本日志文件
        with open(eval_loss_log_txt, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Stage 1 MAE训练验证loss记录\n")
            f.write("=" * 80 + "\n")
            f.write(f"训练集大小: {len(train_dataset)}\n")
            f.write(f"验证集大小: {len(eval_dataset)}\n")
            f.write(f"开始时间: {log_header['start_time']}\n")
            f.write("=" * 80 + "\n\n")
        
        print(f"✓ 验证loss日志文件已初始化:")
        print(f"  - JSON格式: {eval_loss_log_file}")
        print(f"  - 文本格式: {eval_loss_log_txt}")
    
    # ========== 8. 开始训练 ==========
    print("\n" + "=" * 80)
    print("开始训练...")
    print("=" * 80)
    
    try:
        train_result = trainer.train()
        
        print("\n" + "=" * 80)
        print("训练完成！")
        print("=" * 80)
        print(f"训练损失: {train_result.training_loss:.4f}")
        
        # 如果有验证集，打印验证损失
        if eval_dataset is not None:
            # Trainer 会在训练过程中记录验证损失
            # 如果 load_best_model_at_end=True，最终模型就是最佳模型
            if hasattr(train_result, 'eval_loss'):
                print(f"最终验证损失: {train_result.eval_loss:.4f}")
            if load_best_model_at_end:
                print("✓ 已自动加载验证集上表现最佳的模型")
        
        # 保存最终模型
        # 注意：如果 load_best_model_at_end=True，Trainer 会自动加载最佳模型
        # 这里保存的就是最佳模型（如果启用了验证）或最终模型（如果未启用验证）
        print("\n正在保存最终模型...")
        trainer.save_model()
        print(f"✓ 模型已保存到: {output_dir}")
        
        # 额外保存编码器权重（用于 Stage 2）
        encoder_path = os.path.join(output_dir, "encoder.pth")
        torch.save(model.vit.state_dict(), encoder_path)
        print(f"✓ 编码器权重已保存到: {encoder_path}")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        print("正在保存检查点...")
        trainer.save_model()
        print(f"✓ 检查点已保存到: {output_dir}")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n训练脚本执行完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PolarVLM Stage 1 训练脚本")
    
    # 数据路径
    parser.add_argument("--polar_root", type=str, default="data/polar",
                        help="偏振图像根目录（如果 use_pt_data=False）或 .pt 文件根目录（如果 use_pt_data=True）")
    parser.add_argument("--use_pt_data", action="store_true", default=False,
                        help="使用预处理的 .pt 文件（快速模式，大幅提升数据加载速度）")
    parser.add_argument("--pt_root", type=str, default=None,
                        help=".pt 文件根目录（如果 use_pt_data=True，必须提供）")
    
    # 模型配置
    parser.add_argument("--model_name", type=str, default="facebook/vit-mae-base",
                        help="MAE 模型名称")
    parser.add_argument("--image_size", type=int, default=224,
                        help="图像尺寸")
    parser.add_argument("--patch_size", type=int, default=16,
                        help="Patch 尺寸")
    parser.add_argument("--num_channels", type=int, default=3,
                        help="输入通道数（默认3：DoLP, sin(2*AoLP), cos(2*AoLP)，移除Intensity）")
    parser.add_argument("--norm_pix_loss", type=lambda x: (str(x).lower() == 'true'), default=False,
                        nargs='?', const=False,
                        help="启用归一化像素损失（默认False，关键修复：对偏振数据应设为False以保留DoLP等物理量的绝对意义）")
    
    # 训练配置
    parser.add_argument("--output_dir", type=str, default="../autodl-tmp/checkpoints/stage1_encoder",
                        help="输出目录")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16,
                        help="每设备训练批次大小")
    parser.add_argument("--num_train_epochs", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="预热比例")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="权重衰减")
    
    # 其他配置
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="保存步数（仅在 save_strategy='steps' 时有效）")
    parser.add_argument("--save_strategy", type=str, default="steps",
                        choices=["steps", "epoch", "no"],
                        help="保存策略：'steps'（按步数）、'epoch'（按轮数）或 'no'（不保存）")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="保存checkpoint数量限制（只保留最近 N 个，自动删除旧的）")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="DataLoader worker 数量")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="使用 fp16 混合精度")
    
    # 验证/评估配置
    parser.add_argument("--evaluation_strategy", type=str, default="no",
                        choices=["no", "steps", "epoch"],
                        help="评估策略：'no'（不评估）、'steps'（按步数）或 'epoch'（按轮数，推荐）")
    parser.add_argument("--eval_steps", type=int, default=None,
                        help="评估步数（仅在 evaluation_strategy='steps' 时有效，默认与 save_steps 相同）")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None,
                        help="每设备验证批次大小（默认与训练批次大小相同）")
    parser.add_argument("--load_best_model_at_end", action="store_true", default=False,
                        help="训练结束时加载最佳模型（基于验证集loss）")
    parser.add_argument("--metric_for_best_model", type=str, default="loss",
                        help="用于选择最佳模型的指标（默认'loss'）")
    
    # Hugging Face token
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face token")
    
    args = parser.parse_args()
    
    main(
        polar_root=args.polar_root,
        use_pt_data=args.use_pt_data,
        pt_root=args.pt_root if args.use_pt_data else None,
        model_name=args.model_name,
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_channels=args.num_channels,
        norm_pix_loss=args.norm_pix_loss,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        fp16=args.fp16,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        hf_token=args.hf_token,
    )


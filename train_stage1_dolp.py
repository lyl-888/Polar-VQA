"""
Stage 1 DoLP 专用训练脚本：偏振度编码器预训练（MAE）

目标：
- 使用 Masked Autoencoder (MAE) 预训练 DoLP（偏振度）编码器
- 只训练 DoLP 通道，将单通道复制3次以复用 ImageNet 3通道预训练权重
- 冻结策略：冻结前9层，只微调后3层+Head

数据处理：
- 从 PolarMAEDataset 获取 3 通道数据 [DoLP, sin(2*AoLP), cos(2*AoLP)]
- 提取第0个通道（DoLP）
- 将单通道复制3次，构建成 (B, 3, H, W) 的伪彩色图像

模型设置：
- 加载 facebook/vit-mae-base (标准 3 通道)
- 冻结策略：冻结 Encoder 前 9 层 (0-8)，只微调后 3 层 + Decoder
- 参数：norm_pix_loss=False，mask_ratio=0.75
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from typing import Optional, Dict, List
from torch.utils.data import Dataset, DataLoader
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

# ========== 关键：禁用输出缓冲，确保 nohup 时能实时看到输出 ==========
if sys.stdout.isatty():
    pass
else:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

os.environ['PYTHONUNBUFFERED'] = '1'


class DoLPWrapperDataset(Dataset):
    """
    DoLP 数据包装器：从 PolarMAEDataset 提取 DoLP 通道并复制3次
    
    输入：PolarMAEDataset 返回的 (3, H, W) 数据 [DoLP, sin, cos]
    输出：(3, H, W) 数据 [DoLP, DoLP, DoLP] - 3个通道都是 DoLP
    """
    def __init__(self, base_dataset: PolarMAEDataset):
        self.base_dataset = base_dataset
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        pixel_values = sample["pixel_values"]  # (3, H, W) - [DoLP, sin, cos]
        
        # 提取 DoLP 通道（第0个通道）
        dolp_channel = pixel_values[0:1, :, :]  # (1, H, W)
        
        # 复制3次，构建伪彩色图像
        dolp_3ch = dolp_channel.repeat(3, 1, 1)  # (3, H, W) - [DoLP, DoLP, DoLP]
        
        return {
            "pixel_values": dolp_3ch,  # (3, H, W)
        }


class OverfitSingleImageDataset(Dataset):
    """
    Overfit 单张图片的数据集：始终返回同一张图片
    用于测试模型是否能够学习（overfit）
    """
    def __init__(self, base_dataset: PolarMAEDataset, image_idx: int = 0):
        """
        Args:
            base_dataset: 基础数据集
            image_idx: 要 overfit 的图片索引（默认第0张）
        """
        self.base_dataset = base_dataset
        self.image_idx = image_idx % len(base_dataset)
        
        # 预先加载并处理单张图片
        sample = base_dataset[self.image_idx]
        pixel_values = sample["pixel_values"]  # (3, H, W) - [DoLP, sin, cos]
        
        # 提取 DoLP 通道（第0个通道）
        dolp_channel = pixel_values[0:1, :, :]  # (1, H, W)
        
        # 复制3次，构建伪彩色图像
        self.single_image = dolp_channel.repeat(3, 1, 1)  # (3, H, W) - [DoLP, DoLP, DoLP]
        
        # 保存图片信息（scene_id 和 base_name）
        # PolarMAEDataset 有 samples 属性，每个样本是包含 scene_id 和 base_name 的字典
        self.scene_id = None
        self.base_name = None
        if hasattr(base_dataset, 'samples') and len(base_dataset.samples) > self.image_idx:
            sample_info = base_dataset.samples[self.image_idx]
            if isinstance(sample_info, dict):
                self.scene_id = sample_info.get('scene_id')
                self.base_name = sample_info.get('base_name')
        
        print(f"✓ Overfit 模式：使用第 {self.image_idx} 张图片（共 {len(base_dataset)} 张）")
        if self.scene_id and self.base_name:
            print(f"  - scene_id: {self.scene_id}, base_name: {self.base_name}")
        else:
            print(f"  ⚠ 警告: 无法获取 scene_id 和 base_name，请在验证时手动指定")
    
    def __len__(self):
        # 返回一个较大的数字，让训练可以持续进行
        return 10000  # 可以设置任意大的数字
    
    def __getitem__(self, idx):
        # 始终返回同一张图片
        return {
            "pixel_values": self.single_image.clone(),  # (3, H, W)
        }
    
    def get_overfit_info(self):
        """返回 overfit 图片的信息"""
        return {
            "image_idx": self.image_idx,
            "scene_id": self.scene_id,
            "base_name": self.base_name,
        }


def collate_fn_dolp(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """DoLP 专用的 collate 函数"""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    return {
        "pixel_values": pixel_values,  # (B, 3, H, W) - [DoLP, DoLP, DoLP]
    }


def create_mae_model_dolp(
    model_name: str = "facebook/vit-mae-base",
    image_size: int = 224,
    patch_size: int = 16,
    mask_ratio: float = 0.75,
    norm_pix_loss: bool = False,
    hf_token: Optional[str] = None,
) -> ViTMAEForPreTraining:
    """
    创建 DoLP 专用的 MAE 模型
    
    Args:
        model_name: 预训练 MAE 模型名称
        image_size: 图像尺寸
        patch_size: Patch 尺寸
        mask_ratio: 掩码率（默认 0.75）
        norm_pix_loss: 归一化像素损失（默认 False）
        hf_token: Hugging Face token
    
    Returns:
        MAE 模型（3通道输入，冻结前9层）
    """
    print("=" * 80)
    print("创建 DoLP 专用 MAE 模型")
    print("=" * 80)
    
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    load_kwargs = {}
    if token:
        load_kwargs["token"] = token
        print(f"✓ 使用 Hugging Face token 加载模型")
    
    try:
        # 加载预训练配置
        config = ViTMAEConfig.from_pretrained(model_name, **load_kwargs)
        
        # 修改配置
        config.num_channels = 3  # 3通道输入（DoLP 复制3次）
        config.image_size = image_size
        config.patch_size = patch_size
        config.norm_pix_loss = norm_pix_loss
        
        # 设置掩码率
        if hasattr(config, 'mask_ratio'):
            config.mask_ratio = mask_ratio
            print(f"  - 掩码率: {config.mask_ratio}")
        else:
            print("  ⚠ 警告: ViTMAEConfig 不支持 mask_ratio 参数，将使用默认值")
        
        print(f"✓ 模型配置:")
        print(f"  - 图像尺寸: {config.image_size}")
        print(f"  - Patch 尺寸: {config.patch_size}")
        print(f"  - 输入通道数: {config.num_channels} (DoLP 复制3次)")
        print(f"  - 归一化像素损失: {config.norm_pix_loss}")
        
        # 加载预训练权重
        print("\n正在加载预训练权重（3通道，ImageNet）...")
        model = ViTMAEForPreTraining.from_pretrained(
            model_name,
            config=config,
            **load_kwargs
        )
        print("✓ 已加载3通道预训练权重")
        
        # ========== 冻结策略：冻结前9层，只微调后3层+Head ==========
        print("\n" + "=" * 80)
        print("应用层级冻结策略（DoLP：冻结前9层）")
        print("=" * 80)
        
        # Step 1: 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False
        print("✓ Step 1: 已冻结所有参数")
        
        # Step 2: 解冻 Embeddings（需要适应 DoLP 数据）
        for param in model.vit.embeddings.parameters():
            param.requires_grad = True
        print("✓ Step 2: 已解冻 Embeddings")
        
        # Step 3: 解冻最后3层 Encoder（layer 9, 10, 11）
        num_encoder_layers = len(model.vit.encoder.layer)
        if num_encoder_layers >= 12:
            for i in range(9, 12):
                for param in model.vit.encoder.layer[i].parameters():
                    param.requires_grad = True
            print(f"✓ Step 3: 已解冻最后3层 Encoder（layer 9-11，共{num_encoder_layers}层）")
        else:
            start_layer = max(0, num_encoder_layers - 3)
            for i in range(start_layer, num_encoder_layers):
                for param in model.vit.encoder.layer[i].parameters():
                    param.requires_grad = True
            print(f"✓ Step 3: 已解冻最后{num_encoder_layers - start_layer}层 Encoder")
        
        # Step 4: 解冻整个 Decoder
        for param in model.decoder.parameters():
            param.requires_grad = True
        print("✓ Step 4: 已解冻 Decoder")
        
        # 计算参数统计
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
        
    except Exception as e:
        print(f"❌ 创建模型时出现错误: {e}")
        raise
    
    return model


def main(
    # 数据路径
    polar_root: str = "data/polar",
    use_pt_data: bool = False,
    pt_root: Optional[str] = None,
    
    # 模型配置
    model_name: str = "facebook/vit-mae-base",
    image_size: int = 224,
    patch_size: int = 16,
    mask_ratio: float = 0.75,
    norm_pix_loss: bool = False,
    
    # 训练配置
    output_dir: str = "../autodl-tmp/checkpoints/stage1_encoder_dolp",
    per_device_train_batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    num_train_epochs: int = 100,
    learning_rate: float = 3e-4,  # 部分冻结模型，可以使用较大学习率
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.05,
    max_grad_norm: float = 1.0,
    
    # 其他配置
    logging_steps: int = 50,
    save_steps: Optional[int] = 1000,
    save_strategy: str = "steps",
    save_total_limit: int = 3,
    dataloader_num_workers: int = 4,
    fp16: bool = True,
    
    # 验证/评估配置
    evaluation_strategy: str = "no",
    eval_steps: int = None,
    per_device_eval_batch_size: int = None,
    load_best_model_at_end: bool = False,
    metric_for_best_model: str = "loss",
    
    # Overfit 模式配置
    overfit_single_image: bool = False,  # 是否启用 overfit 单张图片模式
    overfit_image_idx: int = 0,  # Overfit 的图片索引
    
    # Hugging Face token
    hf_token: Optional[str] = None,
):
    """主训练函数"""
    print("=" * 80)
    print("PolarVLM Stage 1 DoLP: 偏振度编码器预训练（MAE）")
    print("=" * 80)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    
    # 创建数据集
    print("\n" + "=" * 80)
    print("正在创建数据集...")
    print("=" * 80)
    
    if use_pt_data:
        base_root = Path(pt_root) if pt_root else Path(polar_root)
        train_pt_root = str(base_root / "train")
        train_polar_root = train_pt_root
    else:
        train_polar_root = polar_root
        train_pt_root = None
    
    base_dataset = PolarMAEDataset(
        polar_root=train_polar_root,
        image_size=image_size,
        is_train=True,
        use_pt_data=use_pt_data,
        pt_root=train_pt_root,
    )
    
    # 根据模式选择数据集
    if overfit_single_image:
        print("\n" + "=" * 80)
        print("⚠️  Overfit 模式：只使用单张图片进行训练")
        print("=" * 80)
        train_dataset = OverfitSingleImageDataset(base_dataset, image_idx=overfit_image_idx)
        print(f"✓ Overfit 数据集大小: {len(train_dataset)} (虚拟大小，实际只有1张图片)")
        print(f"  - 数据格式: DoLP 通道复制3次 (B, 3, H, W)")
        print(f"  - 注意: 所有批次都会使用同一张图片")
    else:
        # 包装为 DoLP 专用数据集
        train_dataset = DoLPWrapperDataset(base_dataset)
        print(f"✓ 训练集大小: {len(train_dataset)}")
        print(f"  - 数据格式: DoLP 通道复制3次 (B, 3, H, W)")
    
    # 创建验证集（如果启用）
    eval_dataset = None
    if evaluation_strategy != "no":
        if use_pt_data:
            base_root = Path(pt_root) if pt_root else Path(polar_root)
            val_pt_root = str(base_root / "val")
            val_polar_root = val_pt_root
            
            val_path = Path(val_pt_root)
            if val_path.exists():
                base_eval_dataset = PolarMAEDataset(
                    polar_root=val_polar_root,
                    image_size=image_size,
                    is_train=False,
                    use_pt_data=use_pt_data,
                    pt_root=val_pt_root,
                )
                eval_dataset = DoLPWrapperDataset(base_eval_dataset)
                print(f"✓ 验证集大小: {len(eval_dataset)}")
            else:
                print(f"⚠ 警告: 验证集目录不存在: {val_path}")
                evaluation_strategy = "no"
        else:
            print("⚠ 警告: PNG模式暂不支持自动验证集加载")
            evaluation_strategy = "no"
    
    # 创建模型
    print("\n" + "=" * 80)
    print("正在创建模型...")
    print("=" * 80)
    
    model = create_mae_model_dolp(
        model_name=model_name,
        image_size=image_size,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        norm_pix_loss=norm_pix_loss,
        hf_token=hf_token,
    )
    
    model = model.to(device)
    
    # 创建 DataLoader（用于计算步数）
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_train_batch_size,
        shuffle=True,
        num_workers=dataloader_num_workers,
        collate_fn=collate_fn_dolp,
        pin_memory=True,
    )
    
    # 配置训练参数
    print("\n" + "=" * 80)
    print("正在配置训练参数...")
    print("=" * 80)
    
    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    total_steps = steps_per_epoch * num_train_epochs
    
    print(f"  - 训练集大小: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"  - 验证集大小: {len(eval_dataset)}")
    print(f"  - 每设备训练批次大小: {per_device_train_batch_size}")
    print(f"  - 梯度累积步数: {gradient_accumulation_steps}")
    print(f"  - 有效批次大小: {effective_batch_size}")
    print(f"  - 训练轮数: {num_train_epochs}")
    print(f"  - 每轮步数: ~{steps_per_epoch}")
    print(f"  - 总步数: ~{total_steps}")
    print(f"  - 学习率: {learning_rate}")
    print(f"  - 预热比例: {warmup_ratio}")
    print(f"  - 权重衰减: {weight_decay}")
    
    if overfit_single_image:
        print(f"\n  ⚠️  Overfit 模式提示:")
        print(f"     - 如果 loss 能降到接近 0，说明模型能够学习")
        print(f"     - 如果 loss 无法下降，可能是模型容量或学习率问题")
        print(f"     - 建议观察前几个 epoch 的 loss 变化")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=per_device_eval_batch_size if per_device_eval_batch_size else per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        fp16=fp16,
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,
        logging_first_step=True,
        save_steps=save_steps if save_strategy == "steps" else None,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        eval_strategy=evaluation_strategy,
        eval_steps=eval_steps if eval_steps is not None else (save_steps if evaluation_strategy == "steps" else None),
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="eval_loss" if load_best_model_at_end else metric_for_best_model,
        greater_is_better=False,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=False,
        report_to="tensorboard",
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
    )
    
    # 自定义 Trainer
    eval_loss_log_file = os.path.join(output_dir, "eval_loss_log.json")
    eval_loss_log_txt = os.path.join(output_dir, "eval_loss_log.txt")
    
    class MAETrainer(Trainer):
        def __init__(self, *args, eval_loss_log_file=None, eval_loss_log_txt=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.eval_loss_log_file = eval_loss_log_file
            self.eval_loss_log_txt = eval_loss_log_txt
            self.eval_loss_history = []
        
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            pixel_values = inputs["pixel_values"]
            outputs = model(pixel_values=pixel_values)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss
        
        def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
            model = self._wrap_model(self.model, training=False)
            model.eval()
            
            total_loss = 0.0
            num_samples = 0
            
            print(f"\n开始手动计算验证 loss...")
            for step, inputs in enumerate(dataloader):
                inputs = self._prepare_inputs(inputs)
                with torch.no_grad():
                    loss = self.compute_loss(model, inputs)
                    if isinstance(loss, torch.Tensor):
                        if loss.ndim == 0:
                            batch_size = inputs["pixel_values"].shape[0]
                            total_loss += loss.item() * batch_size
                            num_samples += batch_size
                        else:
                            total_loss += loss.sum().item()
                            num_samples += loss.numel()
                    else:
                        batch_size = inputs["pixel_values"].shape[0]
                        total_loss += float(loss) * batch_size
                        num_samples += batch_size
                
                if (step + 1) % 100 == 0:
                    print(f"  已处理 {step + 1} 个批次...")
            
            if num_samples > 0:
                avg_loss = total_loss / num_samples
            else:
                avg_loss = 0.0
            
            print(f"✓ 验证 loss 计算完成: {avg_loss:.6f} (基于 {num_samples} 个样本)")
            
            metrics = {
                f"{metric_key_prefix}_loss": avg_loss,
            }
            if metric_key_prefix == "eval":
                metrics['eval_loss'] = avg_loss
            
            metrics[f"{metric_key_prefix}_samples_per_second"] = 0.0
            metrics[f"{metric_key_prefix}_steps_per_second"] = 0.0
            
            if hasattr(self.state, 'epoch'):
                metrics["epoch"] = self.state.epoch
            
            # 记录验证loss到文件
            if metric_key_prefix == "eval" and self.eval_loss_log_file and avg_loss is not None:
                current_epoch = self.state.epoch if hasattr(self.state, 'epoch') else None
                current_step = self.state.global_step if hasattr(self.state, 'global_step') else None
                
                log_entry = {
                    "epoch": current_epoch,
                    "step": current_step,
                    "eval_loss": float(avg_loss),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                self.eval_loss_history.append(log_entry)
                
                try:
                    if os.path.exists(self.eval_loss_log_file):
                        with open(self.eval_loss_log_file, 'r', encoding='utf-8') as f:
                            log_data = json.load(f)
                            if isinstance(log_data, list):
                                log_data = {"description": "Stage 1 DoLP MAE训练验证loss记录", "logs": log_data}
                            existing_logs = log_data.get("logs", [])
                    else:
                        log_data = {"description": "Stage 1 DoLP MAE训练验证loss记录", "logs": []}
                        existing_logs = []
                    
                    existing_logs.append(log_entry)
                    log_data["logs"] = existing_logs
                    
                    with open(self.eval_loss_log_file, 'w', encoding='utf-8') as f:
                        json.dump(log_data, f, indent=2, ensure_ascii=False)
                    
                    with open(self.eval_loss_log_txt, 'a', encoding='utf-8') as f:
                        if current_epoch is not None:
                            f.write(f"Epoch {current_epoch:.2f} | Step {current_step} | Eval Loss: {avg_loss:.6f} | {log_entry['timestamp']}\n")
                        else:
                            f.write(f"Step {current_step} | Eval Loss: {avg_loss:.6f} | {log_entry['timestamp']}\n")
                except Exception as e:
                    print(f"⚠ 警告: 记录验证loss时出错: {e}")
            
            from transformers.trainer_utils import EvalLoopOutput
            return EvalLoopOutput(
                predictions=None,
                label_ids=None,
                metrics=metrics,
                num_samples=num_samples
            )
    
    # 创建 Trainer
    print("\n" + "=" * 80)
    print("正在创建 Trainer...")
    print("=" * 80)
    
    trainer = MAETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn_dolp,
        eval_loss_log_file=eval_loss_log_file if eval_dataset is not None else None,
        eval_loss_log_txt=eval_loss_log_txt if eval_dataset is not None else None,
    )
    
    # 初始化日志文件
    if eval_dataset is not None:
        log_header = {
            "description": "Stage 1 DoLP MAE训练验证loss记录",
            "train_dataset_size": len(train_dataset),
            "eval_dataset_size": len(eval_dataset),
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "logs": []
        }
        
        with open(eval_loss_log_file, 'w', encoding='utf-8') as f:
            json.dump(log_header, f, indent=2, ensure_ascii=False)
        
        with open(eval_loss_log_txt, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Stage 1 DoLP MAE训练验证loss记录\n")
            f.write("=" * 80 + "\n")
            f.write(f"训练集大小: {len(train_dataset)}\n")
            f.write(f"验证集大小: {len(eval_dataset)}\n")
            f.write(f"开始时间: {log_header['start_time']}\n")
            f.write("=" * 80 + "\n\n")
    
    # 开始训练
    print("\n" + "=" * 80)
    print("开始训练...")
    print("=" * 80)
    
    try:
        train_result = trainer.train()
        
        print("\n" + "=" * 80)
        print("训练完成！")
        print("=" * 80)
        print(f"训练损失: {train_result.training_loss:.4f}")
        
        if eval_dataset is not None:
            if hasattr(train_result, 'eval_loss'):
                print(f"最终验证损失: {train_result.eval_loss:.4f}")
            if load_best_model_at_end:
                print("✓ 已自动加载验证集上表现最佳的模型")
        
        print("\n正在保存最终模型...")
        trainer.save_model()
        print(f"✓ 模型已保存到: {output_dir}")
        
        encoder_path = os.path.join(output_dir, "encoder.pth")
        torch.save(model.vit.state_dict(), encoder_path)
        print(f"✓ 编码器权重已保存到: {encoder_path}")
        
        # 如果使用 overfit 模式，保存图片信息
        if overfit_single_image and hasattr(train_dataset, 'get_overfit_info'):
            overfit_info = train_dataset.get_overfit_info()
            overfit_info_path = os.path.join(output_dir, "overfit_image_info.json")
            with open(overfit_info_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "overfit_mode": True,
                    "image_idx": overfit_info.get("image_idx"),
                    "scene_id": overfit_info.get("scene_id"),
                    "base_name": overfit_info.get("base_name"),
                    "description": "Overfit训练使用的单张图片信息，可用于验证脚本"
                }, f, indent=2, ensure_ascii=False)
            print(f"✓ Overfit 图片信息已保存到: {overfit_info_path}")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        trainer.save_model()
        print(f"✓ 检查点已保存到: {output_dir}")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n训练脚本执行完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PolarVLM Stage 1 DoLP 训练脚本")
    
    parser.add_argument("--polar_root", type=str, default="data/polar")
    parser.add_argument("--use_pt_data", action="store_true", default=False)
    parser.add_argument("--pt_root", type=str, default=None)
    
    parser.add_argument("--model_name", type=str, default="facebook/vit-mae-base")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--norm_pix_loss", type=lambda x: (str(x).lower() == 'true'), default=False, nargs='?', const=False)
    
    parser.add_argument("--output_dir", type=str, default="../autodl-tmp/checkpoints/stage1_encoder_dolp")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["steps", "epoch", "no"])
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--fp16", action="store_true", default=True)
    
    parser.add_argument("--evaluation_strategy", type=str, default="no", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None)
    parser.add_argument("--load_best_model_at_end", action="store_true", default=False)
    parser.add_argument("--metric_for_best_model", type=str, default="loss")
    
    parser.add_argument("--overfit_single_image", action="store_true", default=False,
                        help="启用 overfit 单张图片模式（用于测试模型学习能力）")
    parser.add_argument("--overfit_image_idx", type=int, default=0,
                        help="Overfit 的图片索引（默认第0张）")
    
    parser.add_argument("--hf_token", type=str, default=None)
    
    args = parser.parse_args()
    
    main(
        polar_root=args.polar_root,
        use_pt_data=args.use_pt_data,
        pt_root=args.pt_root if args.use_pt_data else None,
        model_name=args.model_name,
        image_size=args.image_size,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        norm_pix_loss=args.norm_pix_loss,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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
        overfit_single_image=args.overfit_single_image,
        overfit_image_idx=args.overfit_image_idx,
        hf_token=args.hf_token,
    )


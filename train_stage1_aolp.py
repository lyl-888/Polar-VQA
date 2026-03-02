"""
Stage 1 AoLP 专用训练脚本：偏振角编码器预训练（MAE）

目标：
- 使用 Masked Autoencoder (MAE) 预训练 AoLP（偏振角）编码器
- 只训练 sin(2*AoLP) 和 cos(2*AoLP) 通道，加上全0通道以复用 ImageNet 3通道预训练权重
- 冻结策略：全量解冻（Full Unfreeze），因为 AoLP 纹理特征与 RGB 差异巨大

数据处理：
- 从 PolarMAEDataset 获取 3 通道数据 [DoLP, sin(2*AoLP), cos(2*AoLP)]
- 提取第1、2个通道（sin, cos）
- 加上全0通道，构建成 (B, 3, H, W)，顺序为 [sin, cos, 0]

模型设置：
- 加载 facebook/vit-mae-base (标准 3 通道)
- 冻结策略：全量解冻（所有层都训练）
- 参数：norm_pix_loss=False，mask_ratio=0.6，learning_rate=1e-4（保守）
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


class AoLPWrapperDataset(Dataset):
    """
    AoLP 数据包装器：从 PolarMAEDataset 提取 sin 和 cos 通道并加上全0通道
    
    输入：PolarMAEDataset 返回的 (3, H, W) 数据 [DoLP, sin, cos]
    输出：(3, H, W) 数据 [sin, cos, 0] - sin和cos通道加上全0通道
    """
    def __init__(self, base_dataset: PolarMAEDataset):
        self.base_dataset = base_dataset
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        pixel_values = sample["pixel_values"]  # (3, H, W) - [DoLP, sin, cos]
        
        # 提取 sin 和 cos 通道（第1、2个通道，索引1和2）
        sin_channel = pixel_values[1:2, :, :]  # (1, H, W)
        cos_channel = pixel_values[2:3, :, :]  # (1, H, W)
        
        # 创建全0通道
        _, h, w = sin_channel.shape
        zero_channel = torch.zeros(1, h, w, dtype=sin_channel.dtype)  # (1, H, W)
        
        # 拼接成3通道：[sin, cos, 0]
        aolp_3ch = torch.cat([sin_channel, cos_channel, zero_channel], dim=0)  # (3, H, W)
        
        return {
            "pixel_values": aolp_3ch,  # (3, H, W) - [sin, cos, 0]
        }


def collate_fn_aolp(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """AoLP 专用的 collate 函数"""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    return {
        "pixel_values": pixel_values,  # (B, 3, H, W) - [sin, cos, 0]
    }


def create_mae_model_aolp(
    model_name: str = "facebook/vit-mae-base",
    image_size: int = 224,
    patch_size: int = 16,
    mask_ratio: float = 0.6,
    norm_pix_loss: bool = False,
    hf_token: Optional[str] = None,
) -> ViTMAEForPreTraining:
    """
    创建 AoLP 专用的 MAE 模型（全量解冻）
    
    Args:
        model_name: 预训练 MAE 模型名称
        image_size: 图像尺寸
        patch_size: Patch 尺寸
        mask_ratio: 掩码率（默认 0.6，稍微降低难度）
        norm_pix_loss: 归一化像素损失（默认 False）
        hf_token: Hugging Face token
    
    Returns:
        MAE 模型（3通道输入，全量解冻）
    """
    print("=" * 80)
    print("创建 AoLP 专用 MAE 模型（全量解冻）")
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
        config.num_channels = 3  # 3通道输入（sin, cos, 0）
        config.image_size = image_size
        config.patch_size = patch_size
        config.norm_pix_loss = norm_pix_loss
        
        # 设置掩码率（降低到0.6）
        if hasattr(config, 'mask_ratio'):
            config.mask_ratio = mask_ratio
            print(f"  - 掩码率: {config.mask_ratio} (降低难度，适应 AoLP 特征)")
        else:
            print("  ⚠ 警告: ViTMAEConfig 不支持 mask_ratio 参数，将使用默认值")
        
        print(f"✓ 模型配置:")
        print(f"  - 图像尺寸: {config.image_size}")
        print(f"  - Patch 尺寸: {config.patch_size}")
        print(f"  - 输入通道数: {config.num_channels} (sin, cos, 0)")
        print(f"  - 归一化像素损失: {config.norm_pix_loss}")
        
        # 加载预训练权重
        print("\n正在加载预训练权重（3通道，ImageNet）...")
        model = ViTMAEForPreTraining.from_pretrained(
            model_name,
            config=config,
            **load_kwargs
        )
        print("✓ 已加载3通道预训练权重")
        
        # ========== 冻结策略：全量解冻（所有层都训练）==========
        print("\n" + "=" * 80)
        print("应用层级冻结策略（AoLP：全量解冻）")
        print("=" * 80)
        print("✓ 所有参数都已解冻（全量微调）")
        print("  - 原因：AoLP 的纹理特征与 RGB 差异巨大，底层特征需要重新学习")
        
        # 所有参数默认都是 requires_grad=True，无需额外操作
        # 但为了明确，我们确保所有参数都是可训练的
        for param in model.parameters():
            param.requires_grad = True
        
        # 计算参数统计
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_percentage = 100 * trainable_params / total_params
        
        print("\n" + "=" * 80)
        print("参数统计（全量解冻）")
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
    mask_ratio: float = 0.6,
    norm_pix_loss: bool = False,
    
    # 训练配置
    output_dir: str = "../autodl-tmp/checkpoints/stage1_encoder_aolp",
    per_device_train_batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    num_train_epochs: int = 100,
    learning_rate: float = 1e-4,  # 全量微调，使用保守的学习率
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
    
    # Hugging Face token
    hf_token: Optional[str] = None,
):
    """主训练函数"""
    print("=" * 80)
    print("PolarVLM Stage 1 AoLP: 偏振角编码器预训练（MAE）")
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
    
    # 包装为 AoLP 专用数据集
    train_dataset = AoLPWrapperDataset(base_dataset)
    print(f"✓ 训练集大小: {len(train_dataset)}")
    print(f"  - 数据格式: sin, cos 通道 + 全0通道 (B, 3, H, W)")
    
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
                eval_dataset = AoLPWrapperDataset(base_eval_dataset)
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
    
    model = create_mae_model_aolp(
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
        collate_fn=collate_fn_aolp,
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
    print(f"  - 学习率: {learning_rate} (全量微调，保守设置)")
    print(f"  - 预热比例: {warmup_ratio}")
    print(f"  - 权重衰减: {weight_decay}")
    
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
                                log_data = {"description": "Stage 1 AoLP MAE训练验证loss记录", "logs": log_data}
                            existing_logs = log_data.get("logs", [])
                    else:
                        log_data = {"description": "Stage 1 AoLP MAE训练验证loss记录", "logs": []}
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
        data_collator=collate_fn_aolp,
        eval_loss_log_file=eval_loss_log_file if eval_dataset is not None else None,
        eval_loss_log_txt=eval_loss_log_txt if eval_dataset is not None else None,
    )
    
    # 初始化日志文件
    if eval_dataset is not None:
        log_header = {
            "description": "Stage 1 AoLP MAE训练验证loss记录",
            "train_dataset_size": len(train_dataset),
            "eval_dataset_size": len(eval_dataset),
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "logs": []
        }
        
        with open(eval_loss_log_file, 'w', encoding='utf-8') as f:
            json.dump(log_header, f, indent=2, ensure_ascii=False)
        
        with open(eval_loss_log_txt, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Stage 1 AoLP MAE训练验证loss记录\n")
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
    parser = argparse.ArgumentParser(description="PolarVLM Stage 1 AoLP 训练脚本")
    
    parser.add_argument("--polar_root", type=str, default="data/polar")
    parser.add_argument("--use_pt_data", action="store_true", default=False)
    parser.add_argument("--pt_root", type=str, default=None)
    
    parser.add_argument("--model_name", type=str, default="facebook/vit-mae-base")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--mask_ratio", type=float, default=0.6)
    parser.add_argument("--norm_pix_loss", type=lambda x: (str(x).lower() == 'true'), default=False, nargs='?', const=False)
    
    parser.add_argument("--output_dir", type=str, default="../autodl-tmp/checkpoints/stage1_encoder_aolp")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
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
        hf_token=args.hf_token,
    )


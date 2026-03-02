"""
Stage 1 调试脚本：过拟合单个批次（Sanity Check）

功能：
- 只使用一张图片（或一个批次）进行训练
- 验证代码逻辑是否正确（梯度、损失、数据流）
- 如果连一张图都拟合不了，说明代码有 Bug

预期结果：
- Training Loss 应该迅速下降，最终接近 0.0000
- 重建图像应该和原图几乎一模一样

使用方法：
    python debug_stage1.py \
        --polar_root /openbayes/home/data/polar_pt \
        --use_pt_data \
        --model_name /openbayes/input/input0/models/vit-mae-base \
        --output_dir /openbayes/input/input0/checkpoints/stage1_debug \
        --num_train_epochs 100 \
        --learning_rate 1e-3
"""

import os
import torch
import argparse
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from transformers import (
    ViTMAEConfig,
    ViTMAEForPreTraining,
    TrainingArguments,
    Trainer,
)
from dataset_stage1 import PolarMAEDataset, collate_fn_stage1
import warnings
warnings.filterwarnings("ignore")

# 导入训练脚本中的模型创建函数
from train_stage1 import create_mae_model


class SingleSampleDataset(Dataset):
    """
    调试用数据集：只返回第一个样本，重复 batch_size 次
    这样可以测试模型是否能过拟合单个样本
    """
    def __init__(self, base_dataset: Dataset, batch_size: int = 8):
        self.base_dataset = base_dataset
        self.batch_size = batch_size
        # 获取第一个样本
        self.sample = base_dataset[0]
        print(f"✓ 调试模式：使用单个样本，重复 {batch_size} 次组成一个批次")
        print(f"  样本形状: {self.sample['pixel_values'].shape}")
        print(f"  数据范围: [{self.sample['pixel_values'].min():.4f}, {self.sample['pixel_values'].max():.4f}]")
    
    def __len__(self):
        return self.batch_size  # 返回 batch_size，这样 DataLoader 会创建一个批次
    
    def __getitem__(self, idx):
        # 无论 idx 是什么，都返回同一个样本
        return self.sample


def main(
    # 数据路径
    polar_root: str = "data/polar",
    use_pt_data: bool = False,
    pt_root: Optional[str] = None,
    
    # 模型配置
    model_name: str = "facebook/vit-mae-base",
    image_size: int = 224,
    patch_size: int = 16,
    num_channels: int = 4,
    norm_pix_loss: bool = False,
    
    # 训练配置
    output_dir: str = "../autodl-tmp/checkpoints/stage1_debug",
    per_device_train_batch_size: int = 8,  # 调试模式：小批次
    num_train_epochs: int = 100,  # 调试模式：足够多的 epoch 来过拟合
    learning_rate: float = 1e-3,  # 调试模式：较高的学习率
    warmup_ratio: float = 0.0,  # 调试模式：不需要 warmup
    weight_decay: float = 0.0,  # 调试模式：关闭正则化
    max_grad_norm: float = 1.0,
    
    # 其他配置
    logging_steps: int = 1,  # 调试模式：每步都打印
    save_strategy: str = "no",  # 调试模式：不保存
    save_total_limit: int = 1,
    dataloader_num_workers: int = 0,  # 调试模式：单进程，避免多进程问题
    fp16: bool = False,  # 调试模式：关闭 fp16，避免精度问题
    
    # Hugging Face token
    hf_token: Optional[str] = None,
):
    """
    调试主函数：过拟合单个批次
    """
    print("=" * 80)
    print("Stage 1 调试模式：过拟合单个批次（Sanity Check）")
    print("=" * 80)
    print("\n🎯 目标：验证代码逻辑是否正确")
    print("   - 如果连一张图都拟合不了，说明代码有 Bug")
    print("   - 如果 Loss 能降到接近 0，说明代码逻辑正确")
    print("=" * 80)
    
    # ========== 1. 设置设备 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    
    # ========== 2. 创建完整数据集（用于获取第一个样本）==========
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
    
    # 创建完整数据集（但只用于获取第一个样本）
    full_dataset = PolarMAEDataset(
        polar_root=train_polar_root,
        image_size=image_size,
        is_train=False,  # 调试模式：关闭数据增强，使用固定样本
        use_pt_data=use_pt_data,
        pt_root=train_pt_root,
    )
    
    print(f"✓ 完整数据集大小: {len(full_dataset)}")
    
    # ========== 3. 创建单样本数据集 ==========
    debug_dataset = SingleSampleDataset(
        base_dataset=full_dataset,
        batch_size=per_device_train_batch_size
    )
    
    # ========== 4. 创建模型 ==========
    print("\n" + "=" * 80)
    print("正在创建模型...")
    print("=" * 80)
    
    model = create_mae_model(
        model_name=model_name,
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,
        norm_pix_loss=norm_pix_loss,
        hf_token=hf_token,
    )
    
    # 移动到设备
    model = model.to(device)
    
    # ========== 5. 配置训练参数 ==========
    print("\n" + "=" * 80)
    print("调试模式训练参数:")
    print("=" * 80)
    print(f"  - 训练样本数: {len(debug_dataset)} (单个样本重复 {per_device_train_batch_size} 次)")
    print(f"  - 批次大小: {per_device_train_batch_size}")
    print(f"  - 训练轮数: {num_train_epochs}")
    print(f"  - 学习率: {learning_rate} (较高，便于快速过拟合)")
    print(f"  - 权重衰减: {weight_decay} (关闭正则化)")
    print(f"  - 数据增强: 关闭 (使用固定样本)")
    print(f"  - FP16: {fp16} (调试模式建议关闭)")
    print("=" * 80)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # 批次配置
        per_device_train_batch_size=per_device_train_batch_size,
        
        # 训练轮数和学习率
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,  # 调试模式：关闭正则化
        max_grad_norm=max_grad_norm,
        
        # 优化器配置
        optim="adamw_torch",
        lr_scheduler_type="constant",  # 调试模式：固定学习率
        
        # 精度配置
        fp16=fp16,  # 调试模式：建议关闭 fp16
        
        # 日志和保存配置
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,  # 调试模式：每步都打印
        save_strategy=save_strategy,  # 调试模式：不保存
        save_total_limit=save_total_limit,
        
        # 评估配置（调试模式：不评估）
        eval_strategy="no",
        
        # 其他配置
        dataloader_num_workers=dataloader_num_workers,  # 调试模式：单进程
        remove_unused_columns=False,
        report_to="tensorboard",
        
        # 梯度检查点（调试模式：可以关闭以加快速度）
        gradient_checkpointing=False,
        
        # 数据加载配置
        dataloader_pin_memory=True,
    )
    
    # ========== 6. 定义训练函数 ==========
    class MAETrainer(Trainer):
        """自定义 Trainer，处理 MAE 的输入格式"""
        
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """计算 MAE 损失"""
            pixel_values = inputs["pixel_values"]
            outputs = model(pixel_values=pixel_values)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss
    
    # ========== 7. 创建 Trainer ==========
    print("\n" + "=" * 80)
    print("正在创建 Trainer...")
    print("=" * 80)
    
    trainer = MAETrainer(
        model=model,
        args=training_args,
        train_dataset=debug_dataset,
        data_collator=collate_fn_stage1,
    )
    
    # ========== 8. 打印初始状态 ==========
    print("\n" + "=" * 80)
    print("初始状态检查")
    print("=" * 80)
    
    # 获取一个批次的数据
    sample_batch = collate_fn_stage1([debug_dataset[0] for _ in range(per_device_train_batch_size)])
    pixel_values = sample_batch["pixel_values"].to(device)
    
    print(f"✓ 批次形状: {pixel_values.shape}")
    print(f"✓ 数据范围: [{pixel_values.min():.4f}, {pixel_values.max():.4f}]")
    print(f"✓ 数据类型: {pixel_values.dtype}")
    
    # 计算初始损失
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        initial_loss = outputs.loss.item()
    
    print(f"✓ 初始 Loss: {initial_loss:.6f}")
    print("=" * 80)
    
    # ========== 9. 开始训练 ==========
    print("\n" + "=" * 80)
    print("开始训练（调试模式）...")
    print("=" * 80)
    print("\n📊 预期结果：")
    print("  - Loss 应该迅速下降，最终接近 0.0000")
    print("  - 如果 Loss 降不下去，说明代码有 Bug")
    print("=" * 80 + "\n")
    
    try:
        train_result = trainer.train()
        
        print("\n" + "=" * 80)
        print("训练完成！")
        print("=" * 80)
        print(f"最终训练损失: {train_result.training_loss:.6f}")
        
        # 计算最终损失
        model.eval()
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            final_loss = outputs.loss.item()
        
        print(f"最终验证损失: {final_loss:.6f}")
        print(f"Loss 下降: {initial_loss:.6f} -> {final_loss:.6f}")
        print(f"Loss 下降比例: {(initial_loss - final_loss) / initial_loss * 100:.2f}%")
        
        # 判断结果
        if final_loss < 0.001:
            print("\n✅ 成功！Loss 降到了接近 0，说明代码逻辑正确！")
            print("   可以放心地使用完整数据集进行训练了。")
        elif final_loss < initial_loss * 0.1:
            print("\n✅ 基本成功！Loss 显著下降，代码逻辑基本正确。")
            print("   可能需要调整学习率或训练更多 epoch。")
        else:
            print("\n❌ 失败！Loss 没有显著下降，说明代码可能有 Bug。")
            print("   请检查：")
            print("   1. 数据是否正确加载（范围、形状）")
            print("   2. 模型梯度是否正常（是否有 requires_grad=False）")
            print("   3. 损失函数是否正确")
            print("   4. 学习率是否合适")
        
        print("=" * 80)
        
        # ========== 10. 保存模型权重（调试模式也需要保存以便验证）==========
        print("\n" + "=" * 80)
        print("正在保存模型权重...")
        print("=" * 80)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型权重
        model_save_path = output_path / "pytorch_model.bin"
        torch.save(model.state_dict(), model_save_path)
        print(f"✓ 模型权重已保存到: {model_save_path}")
        
        # 保存模型配置（如果需要）
        if hasattr(model, 'config'):
            config_save_path = output_path / "config.json"
            model.config.to_json_file(config_save_path)
            print(f"✓ 模型配置已保存到: {config_save_path}")
        
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n调试脚本执行完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 调试脚本：过拟合单个批次")
    
    # 数据路径
    parser.add_argument("--polar_root", type=str, default="data/polar",
                        help="偏振图像根目录（如果 use_pt_data=False）或 .pt 文件根目录（如果 use_pt_data=True）")
    parser.add_argument("--use_pt_data", action="store_true", default=False,
                        help="使用预处理的 .pt 文件")
    parser.add_argument("--pt_root", type=str, default=None,
                        help=".pt 文件根目录（如果 use_pt_data=True）")
    
    # 模型配置
    parser.add_argument("--model_name", type=str, default="facebook/vit-mae-base",
                        help="MAE 模型名称")
    parser.add_argument("--image_size", type=int, default=224,
                        help="图像尺寸")
    parser.add_argument("--patch_size", type=int, default=16,
                        help="Patch 尺寸")
    parser.add_argument("--num_channels", type=int, default=4,
                        help="输入通道数")
    parser.add_argument("--norm_pix_loss", type=lambda x: (str(x).lower() == 'true'), default=False,
                        nargs='?', const=False,
                        help="启用归一化像素损失（默认False）")
    
    # 训练配置
    parser.add_argument("--output_dir", type=str, default="../autodl-tmp/checkpoints/stage1_debug",
                        help="输出目录")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="每设备训练批次大小（调试模式：小批次）")
    parser.add_argument("--num_train_epochs", type=int, default=100,
                        help="训练轮数（调试模式：足够多的 epoch）")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="学习率（调试模式：较高学习率）")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="权重衰减（调试模式：关闭正则化）")
    
    # 其他配置
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="日志记录步数（调试模式：每步都打印）")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help="DataLoader worker 数量（调试模式：单进程）")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="使用 fp16 混合精度（调试模式：建议关闭）")
    
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
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        fp16=args.fp16,
        hf_token=args.hf_token,
    )


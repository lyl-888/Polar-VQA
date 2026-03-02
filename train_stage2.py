"""
Stage 2 训练脚本：语义对齐（Projector 预训练）

目标：
- 训练 Projector 对齐偏振特征和 LLM
- 使用 subtype == "content" 的数据作为图像描述任务
- Question 作为 prompt，Answer 作为 caption

模型架构：
- 使用 PolarLlava 模型
- RGB Tower: 冻结（使用预训练 CLIP）
- Polar Tower: 冻结（使用 Stage 1 训练的编码器）
- LLM: 冻结（使用预训练 LLaMA）
- Projector: 可训练（对齐偏振特征和 LLM）

损失函数：
- 因果语言建模损失（Causal Language Modeling Loss）
- 只对 Answer 部分计算损失

输出：
- Projector 权重保存到 checkpoints/stage2_projector/projector.pth
"""

import os
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from typing import Optional
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from dataset_stage2 import PolarAlignmentDataset, collate_fn_stage2
from model import PolarLlava
from torch.utils.data import Dataset
import warnings
import json
warnings.filterwarnings("ignore")


def setup_tokenizer(model_name: str = "/openbayes/input/input0/models/Meta-Llama-3-8B-Instruct", hf_token: Optional[str] = None) -> AutoTokenizer:
    """
    设置并配置tokenizer（从 train.py 复用）
    
    Args:
        model_name: LLM模型名称
        hf_token: Hugging Face token
    
    Returns:
        配置好的tokenizer
    """
    print("正在加载tokenizer...")
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    load_kwargs = {}
    if token:
        load_kwargs["token"] = token
    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    
    # 设置pad token
    if tokenizer.pad_token is None:
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
                found_pad_token = True
                break
        if not found_pad_token:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
    tokenizer.padding_side = "right"
    print(f"✓ Tokenizer配置完成，词汇表大小: {len(tokenizer)}")
    return tokenizer


# 注意：load_polar_encoder_from_stage1 函数已删除
# Stage 1 权重加载现在完全由 PolarLlava.__init__ 中的 _load_stage1_encoder_weights 方法处理
# 这样可以避免重复加载，提高初始化效率


class OverfitSingleImageDataset(Dataset):
    """
    Overfit 单张图片的数据集：始终返回同一张图片
    用于测试模型是否能够学习（overfit）
    """
    def __init__(self, base_dataset: PolarAlignmentDataset, image_idx: int = 0):
        """
        Args:
            base_dataset: 基础数据集（PolarAlignmentDataset）
            image_idx: 要 overfit 的图片索引（默认第0张）
        """
        self.base_dataset = base_dataset
        self.image_idx = image_idx % len(base_dataset)
        
        # 预先加载并处理单张图片
        sample = base_dataset[self.image_idx]
        
        # 保存单张图片的所有数据
        self.single_sample = {
            "pixel_values_rgb": sample["pixel_values_rgb"].clone(),
            "pixel_values_polar": sample["pixel_values_polar"].clone(),
            "question": sample["question"],
            "answer": sample["answer"],
            "prompt_text": sample["prompt_text"],
        }
        
        # 保存图片信息（从 JSON 数据中提取）
        self.scene_id = None
        self.base_name = None
        if hasattr(base_dataset, 'data') and len(base_dataset.data) > self.image_idx:
            item = base_dataset.data[self.image_idx]
            self.scene_id = item.get('scene_id')
            # 从 image 路径提取 base_name
            image_path = item.get('image', '')
            if image_path:
                # 例如：rgb_crop/23/0000_rgb.png -> 0000
                parts = image_path.split('/')
                if len(parts) > 0:
                    filename = parts[-1]
                    if filename.endswith('_rgb.png'):
                        self.base_name = filename[:-8]  # 去掉 '_rgb.png'
        
        print(f"✓ Overfit 模式：使用第 {self.image_idx} 张图片（共 {len(base_dataset)} 张）")
        if self.scene_id and self.base_name:
            print(f"  - scene_id: {self.scene_id}, base_name: {self.base_name}")
        else:
            print(f"  ⚠ 警告: 无法获取 scene_id 和 base_name")
    
    def __len__(self):
        # 返回一个较大的数字，让训练可以持续进行
        return 1000  # 可以设置任意大的数字
    
    def __getitem__(self, idx):
        # 始终返回同一张图片（深拷贝以避免数据污染）
        return {
            "pixel_values_rgb": self.single_sample["pixel_values_rgb"].clone(),
            "pixel_values_polar": self.single_sample["pixel_values_polar"].clone(),
            "question": self.single_sample["question"],
            "answer": self.single_sample["answer"],
            "prompt_text": self.single_sample["prompt_text"],
        }
    
    def get_overfit_info(self):
        """返回 overfit 图片的信息"""
        return {
            "image_idx": self.image_idx,
            "scene_id": self.scene_id,
            "base_name": self.base_name,
        }


def main(
    # 数据路径
    train_json: str = "stage2_gt_captions_all.json",
    val_json: Optional[str] = None,  # 验证集 JSON 文件路径（可选）
    rgb_root: str = "/openbayes/input/input0/rgb",
    polar_root: str = "/openbayes/input/input0/polar",
    data_root: Optional[str] = None,  # 数据根目录（用于解析 crop 路径，默认使用 rgb_root 的父目录）
    
    # 模型配置
    clip_model_name: str = "openai/clip-vit-large-patch14",
    polar_backbone: str = "google/vit-base-patch16-224-in21k",  # ⚠️ 已废弃：仅作为备用选项（当 vae_model_path 不存在时使用）。如果提供了 vae_model_path，此参数将被完全忽略。
    llm_model_name: str = "/openbayes/input/input0/models/Meta-Llama-3-8B-Instruct",
    stage1_checkpoint: Optional[str] = None,  # Stage 1 检查点路径（已废弃，改用 vae_model_path）
    vae_model_path: str = "/openbayes/input/input0/models/sd-vae-ft-mse",  # VAE 模型路径（用于加载 VAE 编码器）。如果提供此路径，polar_backbone 将被忽略。
    load_in_4bit: bool = True,
    hf_token: Optional[str] = None,
    
    # 训练配置（针对空间对齐后的特征融合架构优化）
    output_dir: str = "/openbayes/home/train/checkpoints/stage2_projector_v2",
    per_device_train_batch_size: int = 16,  # 可以尝试增加到 32（如果显存足够）
    gradient_accumulation_steps: int = 2,  # 有效 batch size = 16 * 2 = 32
    num_train_epochs: int = 5,  # 增加 epoch（空间对齐后任务变简单，Loss 应该快速下降）
    learning_rate: float = 5e-4,  # 降低到 5e-4（更保守，提高训练稳定性）
    warmup_ratio: float = 0.03,  # 减少预热时间
    weight_decay: float = 0.0,  # 不使用权重衰减（Projector 参数少，不需要强正则化）
    max_grad_norm: float = 1.0,
    
    # 其他配置
    logging_steps: int = 10,
    save_steps: int = 200,  # 更频繁的保存
    eval_steps: Optional[int] = None,  # 验证步数（如果为 None，将使用 save_steps 或自动计算）
    save_total_limit: int = 2,  # 只保留 2 个 checkpoint
    dataloader_num_workers: int = 8,  # 增加 worker 数量
    bf16: bool = True,  # 启用 bf16（如果支持）
    fp16: bool = False,  # 4-bit 量化模型不支持混合精度
    
    # 冻结策略
    freeze_rgb_tower: bool = True,
    freeze_polar_tower: bool = True,  # Stage 2 冻结偏振编码器
    freeze_llm: bool = True,  # Stage 2 冻结 LLM
    
    # 过拟合测试
    overfit_single_image: bool = False,  # 是否启用 overfit 单张图片模式
    overfit_image_idx: int = 0,  # Overfit 的图片索引
):
    """
    主训练函数
    """
    print("=" * 80)
    print("PolarVLM Stage 2: 语义对齐（Projector 预训练）")
    print("=" * 80)
    
    # ========== 1. 设置设备 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    
    # ========== 2. 加载tokenizer ==========
    print("\n" + "=" * 80)
    print("正在加载tokenizer...")
    print("=" * 80)
    
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tokenizer = setup_tokenizer(llm_model_name, hf_token=token)
    
    # ========== 3. 创建数据集 ==========
    print("\n" + "=" * 80)
    print("正在创建数据集...")
    print("=" * 80)
    
    # 确定数据根目录（用于解析 crop 路径）
    if data_root is None:
        # 默认使用 rgb_root 的父目录（假设结构为 data_root/rgb, data_root/rgb_crop）
        data_root = str(Path(rgb_root).parent)
    
    print(f"数据根目录: {data_root}")
    print(f"RGB 根目录: {rgb_root}")
    print(f"Polar 根目录: {polar_root}")
    
    # 创建基础数据集
    base_dataset = PolarAlignmentDataset(
        rgb_root=rgb_root,
        polar_root=polar_root,
        json_file=train_json,
        tokenizer=tokenizer,
        is_train=True,
        data_root=data_root,  # 传递数据根目录
    )
    
    # 根据模式选择数据集
    if overfit_single_image:
        print("\n" + "=" * 80)
        print("⚠️  Overfit 模式：只使用单张图片进行训练")
        print("=" * 80)
        train_dataset = OverfitSingleImageDataset(base_dataset, image_idx=overfit_image_idx)
        print(f"✓ Overfit 数据集大小: {len(train_dataset)} (虚拟大小，实际只有1张图片)")
        print(f"  - 数据格式: RGB (224x224) + Polar (512x512, 3通道)")
        print(f"  - 注意: 所有批次都会使用同一张图片")
        val_dataset = None  # Overfit 模式下不使用验证集
    else:
        train_dataset = base_dataset
        print(f"✓ 训练集大小: {len(train_dataset)}")
        
        # 创建验证集（如果提供了 val_json）
        val_dataset = None
        if val_json and os.path.exists(val_json):
            print("\n正在创建验证集...")
            val_dataset = PolarAlignmentDataset(
                rgb_root=rgb_root,
                polar_root=polar_root,
                json_file=val_json,
                tokenizer=tokenizer,
                is_train=False,  # 验证集不使用数据增强
                data_root=data_root,
            )
            print(f"✓ 验证集大小: {len(val_dataset)}")
        elif val_json:
            print(f"⚠️  警告: 验证集文件不存在: {val_json}，将跳过验证")
        else:
            print("⚠️  未提供验证集，将跳过验证")
    
    # ========== 4. 创建模型 ==========
    print("\n" + "=" * 80)
    print("正在创建模型...")
    print("=" * 80)
    
    # 提示信息：如果提供了 vae_model_path，polar_backbone 将被忽略
    if vae_model_path and os.path.exists(vae_model_path):
        print(f"✓ 检测到 VAE 模型路径: {vae_model_path}")
        print(f"  ⚠️  注意: polar_backbone 参数将被忽略（仅作为备用选项）")
    elif vae_model_path:
        print(f"⚠️  警告: VAE 模型路径不存在: {vae_model_path}")
        print(f"  → 将回退使用 polar_backbone: {polar_backbone}")
    
    model = PolarLlava(
        clip_model_name=clip_model_name,
        polar_backbone=polar_backbone,
        llm_model_name=llm_model_name,
        load_in_4bit=load_in_4bit,
        hf_token=token,
        freeze_rgb_tower=freeze_rgb_tower,
        freeze_polar_tower=freeze_polar_tower,
        freeze_polar_layers=0,
        use_lora=False,  # Stage 2 不使用 LoRA，只训练 Projector
        stage1_checkpoint=stage1_checkpoint,  # 已废弃，保留以兼容旧代码
        vae_model_path=vae_model_path,  # 使用 VAE 编码器
    )
    
    # 注意：VAE 编码器权重已在 PolarLlava.__init__ 中自动加载
    
    # 冻结 LLM（如果指定）
    if freeze_llm:
        for param in model.language_model.parameters():
            param.requires_grad = False
        print("✓ LLM 已冻结")
    
    # 确保只有 Projector 可训练
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n训练参数统计:")
    print(f"  - 可训练参数: {trainable_params:,}")
    print(f"  - 总参数: {total_params:,}")
    print(f"  - 可训练参数占比: {100 * trainable_params / total_params:.2f}%")
    
    # [新增] 详细检查可训练参数，确保 rgb_scale 和 polar_scale 可训练
    print(f"\n可训练参数详情:")
    trainable_param_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_param_names.append(name)
            if 'rgb_scale' in name or 'polar_scale' in name:
                print(f"  ✓ {name}: {param.numel()} 参数, 值={param.item():.4f}")
    
    # 检查动态平衡参数
    has_rgb_scale = any('rgb_scale' in name for name in trainable_param_names)
    has_polar_scale = any('polar_scale' in name for name in trainable_param_names)
    
    if not has_rgb_scale:
        print("  ⚠️ 警告: rgb_scale 不可训练！这可能导致 RGB 和 Polar 特征不平衡")
    if not has_polar_scale:
        print("  ⚠️ 警告: polar_scale 不可训练！这可能导致 RGB 和 Polar 特征不平衡")
    
    if has_rgb_scale and has_polar_scale:
        print("  ✓ 动态平衡参数（rgb_scale, polar_scale）已确认可训练")
    
    # 打印前10个可训练参数（用于调试）
    print(f"\n前10个可训练参数:")
    for name in trainable_param_names[:10]:
        param = dict(model.named_parameters())[name]
        print(f"  - {name}: {param.numel():,} 参数")
    
    # ========== 5. 创建自定义collate函数 ==========
    def create_collate_fn(tokenizer):
        def custom_collate_fn(batch):
            return collate_fn_stage2(batch, tokenizer)
        return custom_collate_fn
    
    data_collator = create_collate_fn(tokenizer)
    
    # ========== 6. 配置训练参数 ==========
    print("\n" + "=" * 80)
    print("正在配置训练参数...")
    print("=" * 80)
    
    train_size = len(train_dataset)
    total_steps = (train_size * num_train_epochs) // (per_device_train_batch_size * gradient_accumulation_steps)
    
    print(f"  - 训练集大小: {train_size}")
    if overfit_single_image:
        print(f"    ⚠️  Overfit 模式：虚拟大小 {train_size}，实际只有 1 张图片")
    if val_dataset is not None:
        print(f"  - 验证集大小: {len(val_dataset)}")
    print(f"  - 总步数: ~{total_steps}")
    print(f"  - 训练轮数: {num_train_epochs}")
    print(f"  - 学习率: {learning_rate}")
    print(f"  - 预热比例: {warmup_ratio}")
    print(f"  - 权重衰减: {weight_decay}")
    
    if overfit_single_image:
        print(f"\n  ⚠️  Overfit 模式提示:")
        print(f"    - 如果 loss 能降到接近 0，说明模型能够学习")
        print(f"    - 如果 loss 无法下降，可能是模型容量或学习率问题")
        print(f"    - 建议观察前几个 epoch 的 loss 变化")
    
    # 配置验证相关参数（如果有验证集）
    eval_steps_value = None
    eval_strategy = "no"
    load_best_model_at_end = False
    metric_for_best_model = "eval_loss"
    greater_is_better = False
    
    if val_dataset is not None:
        # 计算验证步数（更频繁的验证，以便更准确地观察验证曲线）
        if eval_steps is not None:
            # 如果用户指定了 eval_steps，使用用户指定的值
            eval_steps_value = eval_steps
        else:
            # 默认：每 50 步验证一次（比保存更频繁，以便更准确地观察验证曲线）
            # 或者每 save_steps/4 步验证一次（确保验证频率是保存频率的 4 倍）
            eval_steps_value = min(50, max(10, save_steps // 4))
        
        eval_strategy = "steps"
        load_best_model_at_end = True
        print(f"  - 验证策略: {eval_strategy}")
        print(f"  - 验证步数: {eval_steps_value} (每 {eval_steps_value} 步验证一次，比保存更频繁)")
        print(f"  - 保存步数: {save_steps} (每 {save_steps} 步保存一次)")
        print(f"  - 训练结束后加载最佳模型: {load_best_model_at_end}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # 批次配置
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        
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
        bf16=bf16,
        fp16=fp16,
        
        # 日志和保存配置
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        save_strategy="steps",
        
        # 验证配置（如果有验证集）
        eval_strategy=eval_strategy,
        eval_steps=eval_steps_value,
        per_device_eval_batch_size=per_device_train_batch_size,  # 验证批次大小与训练相同
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        
        # 其他配置
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=False,
        report_to="tensorboard",
        
        # 梯度检查点
        # [改回 True] 牺牲约 30% 速度，换取大量显存空间
        gradient_checkpointing=True,
        
        # 数据加载配置
        dataloader_pin_memory=True,
    )
    
    # ========== 7. 创建Trainer（带监控） ==========
    print("\n" + "=" * 80)
    print("正在创建Trainer...")
    print("=" * 80)
    
    # 自定义 Trainer 类，用于监控动态平衡参数
    class Stage2Trainer(Trainer):
        def log(self, logs: dict, start_time: Optional[float] = None) -> None:
            """重写 log 方法，添加动态平衡参数的监控"""
            # 获取动态平衡参数的值
            try:
                rgb_scale_val = model.multi_modal_projector.rgb_scale.item()
                polar_scale_val = model.multi_modal_projector.polar_scale.item()
                logs['rgb_scale'] = rgb_scale_val
                logs['polar_scale'] = polar_scale_val
            except Exception as e:
                # 如果获取失败，不影响训练
                pass
            
            # 调用父类的 log 方法（传递所有参数）
            if start_time is not None:
                super().log(logs, start_time)
            else:
                super().log(logs)
    
    trainer = Stage2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # 添加验证集
        data_collator=data_collator,
    )
    
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
        
        # 如果有验证集，显示验证损失
        if val_dataset is not None:
            eval_result = trainer.evaluate()
            eval_loss = eval_result.get('eval_loss', None)
            if eval_loss is not None:
                print(f"验证损失: {eval_loss:.4f}")
            print(f"最佳模型已自动加载（基于验证损失）")
        
        # 保存最终模型
        print("\n正在保存最终模型...")
        trainer.save_model()
        print(f"✓ 模型已保存到: {output_dir}")
        
        # 额外保存 Projector 权重（用于 Stage 3）
        projector_path = os.path.join(output_dir, "projector.pth")
        torch.save(model.multi_modal_projector.state_dict(), projector_path)
        print(f"✓ Projector 权重已保存到: {projector_path}")
        
        # 如果是过拟合模式，保存过拟合图片信息
        if overfit_single_image and hasattr(train_dataset, 'get_overfit_info'):
            overfit_info = train_dataset.get_overfit_info()
            overfit_info_path = os.path.join(output_dir, "overfit_image_info.json")
            with open(overfit_info_path, 'w', encoding='utf-8') as f:
                json.dump(overfit_info, f, ensure_ascii=False, indent=2)
            print(f"✓ 过拟合图片信息已保存到: {overfit_info_path}")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        print("正在保存检查点...")
        trainer.save_model()
        projector_path = os.path.join(output_dir, "projector.pth")
        torch.save(model.multi_modal_projector.state_dict(), projector_path)
        print(f"✓ 检查点已保存到: {output_dir}")
        
        # 如果是过拟合模式，保存过拟合图片信息
        if overfit_single_image and hasattr(train_dataset, 'get_overfit_info'):
            overfit_info = train_dataset.get_overfit_info()
            overfit_info_path = os.path.join(output_dir, "overfit_image_info.json")
            with open(overfit_info_path, 'w', encoding='utf-8') as f:
                json.dump(overfit_info, f, ensure_ascii=False, indent=2)
            print(f"✓ 过拟合图片信息已保存到: {overfit_info_path}")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n训练脚本执行完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PolarVLM Stage 2 训练脚本")
    
    # 数据路径
    parser.add_argument("--train_json", type=str, default="stage2_gt_captions_all.json",
                        help="训练集JSON文件路径")
    parser.add_argument("--val_json", type=str, default=None,
                        help="验证集JSON文件路径（可选，如果提供将启用验证）")
    parser.add_argument("--rgb_root", type=str, default="/openbayes/input/input0/rgb",
                        help="RGB图像根目录")
    parser.add_argument("--polar_root", type=str, default="/openbayes/input/input0/polar",
                        help="偏振图像根目录")
    parser.add_argument("--data_root", type=str, default=None,
                        help="数据根目录（用于解析 crop 路径，默认使用 rgb_root 的父目录）")
    
    # 模型配置
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-large-patch14",
                        help="CLIP模型名称")
    parser.add_argument("--polar_backbone", type=str, default="google/vit-base-patch16-224-in21k",
                        help="⚠️ 已废弃：偏振流backbone（仅作为备用选项，当 --vae_model_path 不存在时使用）。如果提供了 --vae_model_path，此参数将被完全忽略。")
    parser.add_argument("--llm_model_name", type=str, default="/openbayes/input/input0/models/Meta-Llama-3-8B-Instruct",
                        help="LLM 模型路径（Instruct 版本）")
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Stage 1 检查点路径（已废弃，改用 --vae_model_path）")
    parser.add_argument("--vae_model_path", type=str, default="/openbayes/input/input0/models/sd-vae-ft-mse",
                        help="VAE 模型路径（用于加载 VAE 编码器）")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="使用4-bit量化")
    
    # 训练配置
    parser.add_argument("--output_dir", type=str, default="/openbayes/home/train/checkpoints/stage2_projector_v2",
                        help="输出目录")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16,
                        help="每设备训练批次大小（如果显存不够，可以降低到 8）")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="梯度累积步数（有效 batch size = per_device * gradient_accumulation）")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                        help="训练轮数（空间对齐后任务变简单，可以适当增加 epoch）")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="学习率（降低到 5e-4，提高训练稳定性）")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="预热比例（减少预热时间）")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="权重衰减（不使用权重衰减，Projector 参数少）")
    
    # 其他配置
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=200,
                        help="保存步数（更频繁的保存）")
    parser.add_argument("--eval_steps", type=int, default=None,
                        help="验证步数（每 N 步验证一次）。如果为 None，将自动计算为 min(50, save_steps//4)，确保验证比保存更频繁")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="保存checkpoint数量限制（只保留 2 个 checkpoint）")
    parser.add_argument("--dataloader_num_workers", type=int, default=8,
                        help="DataLoader worker 数量（增加 worker 数量）")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="使用 bf16（如果支持）")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="使用 fp16（4-bit 量化模型不支持）")
    
    # 冻结策略
    parser.add_argument("--freeze_rgb_tower", action="store_true", default=True,
                        help="冻结RGB流")
    parser.add_argument("--freeze_polar_tower", action="store_true", default=True,
                        help="冻结偏振流")
    parser.add_argument("--freeze_llm", action="store_true", default=True,
                        help="冻结LLM")
    
    # 过拟合测试
    parser.add_argument("--overfit_single_image", action="store_true", default=False,
                        help="启用单样本过拟合模式（用于测试）")
    parser.add_argument("--overfit_image_idx", type=int, default=0,
                        help="过拟合的图片索引（默认第0张）")
    
    # Hugging Face token
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face token")
    
    args = parser.parse_args()
    
    main(
        train_json=args.train_json,
        val_json=args.val_json,
        rgb_root=args.rgb_root,
        polar_root=args.polar_root,
        data_root=args.data_root,
        clip_model_name=args.clip_model_name,
        polar_backbone=args.polar_backbone,
        llm_model_name=args.llm_model_name,
        stage1_checkpoint=args.stage1_checkpoint,
        vae_model_path=args.vae_model_path,
        load_in_4bit=args.load_in_4bit,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=args.bf16,
        fp16=args.fp16,
        freeze_rgb_tower=args.freeze_rgb_tower,
        freeze_polar_tower=args.freeze_polar_tower,
        freeze_llm=args.freeze_llm,
        overfit_single_image=args.overfit_single_image,
        overfit_image_idx=args.overfit_image_idx,
        hf_token=args.hf_token,
    )


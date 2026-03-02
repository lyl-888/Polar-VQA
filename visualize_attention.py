"""
Stage 1 注意力可视化脚本：可视化 Polar-ViT (MAE) 的自注意力图

功能：
1. 加载训练好的 Stage 1 MAE 模型
2. 提取最后一层 Transformer Encoder 的注意力权重
3. 关注 [CLS] token 对各个图像 patch 的注意力分布
4. 可视化原始 Intensity、DoLP 和注意力热图叠加
5. 验证模型是否关注物理上重要的区域（如反射、边缘）

目标：
- 验证模型是否学习到了偏振物理特性
- 检查注意力是否集中在高 DoLP 区域（反射、边缘等）
- 分析模型的视觉注意力机制
"""

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from transformers import ViTMAEConfig, ViTMAEForPreTraining, ViTConfig, ViTModel

# Matplotlib 配置：关闭负号乱码
plt.rcParams["axes.unicode_minus"] = False

# 导入共享的数据处理函数
from dataset_common import process_polar_images


def load_mae_model(checkpoint_path: str, hf_token: Optional[str] = None):
    """
    加载训练好的 Stage 1 MAE 模型
    
    Args:
        checkpoint_path: Stage 1 检查点路径（目录）
        hf_token: Hugging Face token（可选）
    
    Returns:
        加载的 MAE 模型
    """
    print("=" * 80)
    print("加载 Stage 1 MAE 模型")
    print("=" * 80)
    
    # 获取 token
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    load_kwargs = {}
    if token:
        load_kwargs["token"] = token
        print(f"✓ 使用 Hugging Face token")
    
    # 检查检查点路径
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点路径不存在: {checkpoint_path}")
    
    # 尝试加载模型文件
    model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if not os.path.exists(model_path):
        # 尝试 safetensors 格式
        model_path = os.path.join(checkpoint_path, "model.safetensors")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"在 {checkpoint_path} 中未找到模型文件")
    
    # 加载配置
    config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(config_path):
        config = ViTMAEConfig.from_pretrained(checkpoint_path, **load_kwargs)
    else:
        # 如果没有配置文件，使用默认配置
        print("⚠ 警告: 未找到 config.json，使用默认配置")
        config = ViTMAEConfig()
        config.num_channels = 4  # 4通道输入（Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)）
        config.image_size = 224
        config.patch_size = 16
    
    # 确保配置正确（4通道：Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)）
    if not hasattr(config, 'num_channels') or config.num_channels != 4:
        config.num_channels = 4
    config.image_size = 224
    config.patch_size = 16
    
    print(f"✓ 模型配置:")
    print(f"  - 图像尺寸: {config.image_size}")
    print(f"  - Patch 尺寸: {config.patch_size}")
    print(f"  - 输入通道数: {config.num_channels} (Intensity, DoLP, sin(2*AoLP), cos(2*AoLP))")
    
    # 加载模型
    if model_path.endswith(".safetensors"):
        # 如果是 safetensors，需要先创建模型再加载权重
        model = ViTMAEForPreTraining(config)
        import safetensors.torch
        state_dict = safetensors.torch.load_file(model_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"✓ 已从 safetensors 加载模型权重")
    else:
        # 直接加载
        model = ViTMAEForPreTraining.from_pretrained(
            checkpoint_path,
            config=config,
            ignore_mismatched_sizes=True,
            **load_kwargs
        )
        print(f"✓ 已加载模型权重")
    
    model.eval()  # 设置为评估模式
    return model


def preprocess_image(physics_img: np.ndarray, image_size: int = 224) -> torch.Tensor:
    """
    预处理图像：转换为模型输入格式
    
    Args:
        physics_img: 4通道物理参数图像 (H, W, 4)，值范围 [0, 1]
                    通道顺序：[Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]
        image_size: 目标图像尺寸
    
    Returns:
        预处理后的图像张量 (1, 4, H, W)
    """
    # 转换为 PIL Image（需要 uint8 格式，使用 RGBA 模式处理4通道）
    physics_img_uint8 = (physics_img * 255).astype(np.uint8)
    physics_pil = Image.fromarray(physics_img_uint8, mode='RGBA')
    
    # Resize 到目标尺寸
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(physics_pil)  # (4, H, W)
    img_tensor = img_tensor.unsqueeze(0)  # (1, 4, H, W)
    
    return img_tensor


def extract_attention_maps(model: ViTMAEForPreTraining, pixel_values: torch.Tensor):
    """
    提取 Transformer Encoder 的注意力权重
    
    Args:
        model: MAE 模型
        pixel_values: 输入图像 (1, 4, 224, 224)
    
    Returns:
        attention_maps: 注意力图字典，包含：
            - 'cls_attention': [CLS] token 对各个 patch 的注意力 (14, 14)
            - 'patch_size': patch 尺寸
            - 'image_size': 图像尺寸
    """
    print("\n" + "=" * 80)
    print("提取注意力权重...")
    print("=" * 80)
    
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(device)
    
    # 获取模型配置
    config = model.config
    image_size = config.image_size
    patch_size = config.patch_size
    num_patches = (image_size // patch_size) ** 2  # 14 * 14 = 196
    
    # 关键：构造一个单独的 ViTModel，并从 MAE 的 encoder 拷贝权重，用它来输出注意力
    # 原因：当前 transformers 版本的 ViTMAEEncoder / ViTMAELayer 不支持 output_attentions 参数
    #       但 ViTModel 支持，因此我们用相同配置和权重构建一个 ViTModel 来做注意力可视化。
    #
    # 步骤：
    # 1. 从 MAE 的 config 创建一个 ViTConfig
    # 2. 用该配置构造 ViTModel（支持 output_attentions）
    # 3. 从 model.vit.state_dict() 加载权重到 ViTModel 中
    # 4. 使用 vit_encoder(pixel_values, output_attentions=True) 获取 attentions
    vit_cfg_dict = config.to_dict() if hasattr(config, "to_dict") else {}
    vit_config = ViTConfig(**vit_cfg_dict)
    # 确保关键配置一致
    vit_config.image_size = image_size
    vit_config.patch_size = patch_size
    vit_config.num_channels = config.num_channels if hasattr(config, "num_channels") else 4
    vit_config.output_attentions = True

    # 构建 ViT 编码器并加载 MAE 的 encoder 权重
    vit_encoder = ViTModel(vit_config)
    vit_encoder.to(device)
    vit_encoder.eval()

    # 从 MAE 的 vit 部分拷贝权重
    mae_vit_state = model.vit.state_dict()
    missing, unexpected = vit_encoder.load_state_dict(mae_vit_state, strict=False)
    if missing or unexpected:
        print("⚠ 加载 ViT 编码器权重时存在不完全匹配：")
        if missing:
            print(f"  - 缺失权重: {missing}")
        if unexpected:
            print(f"  - 多余权重: {unexpected}")

    # 使用 ViT 编码器前向传播，获取注意力
    with torch.no_grad():
        vit_outputs = vit_encoder(
            pixel_values=pixel_values,
            output_attentions=True,
            output_hidden_states=False,
            return_dict=True,
        )

    attentions = getattr(vit_outputs, "attentions", None)
    if attentions is None or len(attentions) == 0:
        print("⚠ 未能从 ViT 编码器输出中获取 attentions。")
        print(f"  - vit_outputs 类型: {type(vit_outputs)}")
        try:
            if hasattr(vit_outputs, '__dict__'):
                print(f"  - vit_outputs.__dict__.keys(): {list(vit_outputs.__dict__.keys())}")
            else:
                print(f"  - vit_outputs keys: {list(vit_outputs.keys())}")
        except Exception as e:
            print(f"  - 无法获取 keys: {e}")
        raise ValueError("未能提取到注意力权重，请检查 ViTModel 输出。")
    
    print(f"✓ 成功提取 {len(attentions)} 层的注意力权重")
    
    # 获取最后一层的注意力权重（最深层，包含最丰富的语义信息）
    last_layer_attention = attentions[-1]  # (batch_size, num_heads, seq_len, seq_len)
    
    print(f"  - 最后一层注意力形状: {last_layer_attention.shape}")
    print(f"  - 注意力头数: {last_layer_attention.shape[1]}")
    print(f"  - 序列长度: {last_layer_attention.shape[2]} (1个 [CLS] token + {num_patches} 个 patch tokens)")
    
    # ========== 注意力处理：关注 [CLS] token ==========
    # 
    # [CLS] token 是第一个 token（索引 0），它聚合了全局信息
    # 我们关注 [CLS] token 对各个 patch 的注意力分布
    # 
    # 注意力矩阵形状：(batch_size, num_heads, seq_len, seq_len)
    # 其中 seq_len = 1 ([CLS]) + num_patches (图像 patches)
    # 
    # 我们提取 [CLS] token（第0行）对各个 patch 的注意力（第1列到最后一列）
    # 然后对所有注意力头求平均，得到综合的注意力分布
    # 
    
    # 选择 batch 0
    batch_attention = last_layer_attention[0]  # (num_heads, seq_len, seq_len)
    
    # 提取 [CLS] token 对各个 patch 的注意力
    # [CLS] token 是第0个 token，我们取第0行，第1列到最后一列（跳过 [CLS] 自己）
    cls_attention_to_patches = batch_attention[:, 0, 1:]  # (num_heads, num_patches)
    
    # 对所有注意力头求平均（得到综合的注意力分布）
    cls_attention_mean = cls_attention_to_patches.mean(dim=0)  # (num_patches,)
    
    print(f"✓ [CLS] token 注意力提取完成")
    print(f"  - 注意力值范围: [{cls_attention_mean.min().item():.4f}, {cls_attention_mean.max().item():.4f}]")
    
    # 将 1D 的 patch 序列 reshape 回 2D 空间网格
    h = w = image_size // patch_size  # 14
    cls_attention_2d = cls_attention_mean.reshape(h, w).cpu().numpy()  # (14, 14)
    
    print(f"✓ 注意力图 reshape 为空间网格: {cls_attention_2d.shape}")
    
    return {
        'cls_attention': cls_attention_2d,
        'patch_size': patch_size,
        'image_size': image_size,
    }


def upsample_attention_map(attention_map: np.ndarray, target_size: int = 224):
    """
    将低分辨率的注意力图（14x14）上采样到原始图像尺寸（224x224）
    
    Args:
        attention_map: 注意力图 (14, 14)
        target_size: 目标尺寸（默认224）
    
    Returns:
        上采样后的注意力图 (224, 224)，值范围 [0, 1]
    """
    # 转换为 torch tensor 以便使用 F.interpolate
    attn_tensor = torch.from_numpy(attention_map).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 14, 14)
    
    # 使用双三次插值上采样到目标尺寸
    upsampled = F.interpolate(
        attn_tensor,
        size=(target_size, target_size),
        mode='bicubic',
        align_corners=False
    )
    
    # 转换为 numpy 并归一化到 [0, 1]
    upsampled_np = upsampled.squeeze().numpy()
    
    # 归一化到 [0, 1]
    attn_min = upsampled_np.min()
    attn_max = upsampled_np.max()
    if attn_max > attn_min:
        upsampled_np = (upsampled_np - attn_min) / (attn_max - attn_min)
    else:
        upsampled_np = np.zeros_like(upsampled_np)
    
    return upsampled_np


def visualize_attention(
    model: ViTMAEForPreTraining,
    pixel_values: torch.Tensor,
    physics_img: np.ndarray,
    output_path: str = "attention_vis.png",
):
    """
    可视化注意力图：显示原始 Intensity、DoLP 和注意力热图叠加
    
    Args:
        model: MAE 模型
        pixel_values: 输入图像 (1, 4, 224, 224)
        physics_img: 原始物理参数图像 (H, W, 4)，用于提取 Intensity 和 DoLP
        output_path: 输出图像路径
    """
    print("\n" + "=" * 80)
    print("生成注意力可视化...")
    print("=" * 80)
    
    # 提取注意力图
    attention_info = extract_attention_maps(model, pixel_values)
    cls_attention_2d = attention_info['cls_attention']
    image_size = attention_info['image_size']
    
    # 上采样注意力图到原始图像尺寸
    attention_map_upsampled = upsample_attention_map(cls_attention_2d, target_size=image_size)
    
    print(f"✓ 注意力图已上采样到 {image_size}x{image_size}")
    
    # 从原始物理参数图像中提取 Intensity 和 DoLP
    # physics_img 形状：(H, W, 4)，通道顺序：[Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]
    # 需要 resize 到 224x224（如果还不是这个尺寸）
    if physics_img.shape[:2] != (image_size, image_size):
        from PIL import Image
        intensity_pil = Image.fromarray((physics_img[:, :, 0] * 255).astype(np.uint8), mode='L')
        dolp_pil = Image.fromarray((physics_img[:, :, 1] * 255).astype(np.uint8), mode='L')
        intensity_pil = intensity_pil.resize((image_size, image_size), Image.BICUBIC)
        dolp_pil = dolp_pil.resize((image_size, image_size), Image.BICUBIC)
        intensity_img = np.array(intensity_pil) / 255.0
        dolp_img = np.array(dolp_pil) / 255.0
    else:
        intensity_img = physics_img[:, :, 0]  # Intensity 通道
        dolp_img = physics_img[:, :, 1]  # DoLP 通道
    
    # 创建可视化图像
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Stage 1 MAE Self-Attention Visualization', fontsize=16, fontweight='bold')
    
    # 第1列：原始 Intensity（灰度图）
    axes[0].imshow(intensity_img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original Intensity', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 第2列：DoLP（偏振度，显示反射和边缘区域）
    axes[1].imshow(dolp_img, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('DoLP (Ground Truth)\nHigh values = Reflections/Edges', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 第3列：注意力热图叠加在 Intensity 上
    # 使用 jet 或 magma colormap 显示注意力，alpha 混合叠加在原始图像上
    axes[2].imshow(intensity_img, cmap='gray', vmin=0, vmax=1)
    im = axes[2].imshow(
        attention_map_upsampled,
        cmap='jet',
        alpha=0.6,  # 透明度，让原始图像可见
        vmin=0,
        vmax=1
    )
    axes[2].set_title('Attention Heatmap\n(Overlay on Intensity)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # 添加颜色条
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ 注意力可视化已保存到: {output_path}")
    plt.close()
    
    # 打印统计信息
    print("\n注意力统计信息：")
    print(f"  - 注意力值范围: [{attention_map_upsampled.min():.4f}, {attention_map_upsampled.max():.4f}]")
    print(f"  - 注意力均值: {attention_map_upsampled.mean():.4f}")
    print(f"  - 注意力标准差: {attention_map_upsampled.std():.4f}")
    
    # 计算注意力与 DoLP 的相关性（验证是否关注高偏振度区域）
    # 将 DoLP 也 resize 到相同尺寸（如果还没有）
    if dolp_img.shape != attention_map_upsampled.shape:
        # 使用 PIL 或 torch 进行 resize（避免依赖 scipy）
        dolp_pil = Image.fromarray((dolp_img * 255).astype(np.uint8), mode='L')
        dolp_pil = dolp_pil.resize((attention_map_upsampled.shape[1], attention_map_upsampled.shape[0]), Image.BICUBIC)
        dolp_resized = np.array(dolp_pil) / 255.0
    else:
        dolp_resized = dolp_img
    
    # 计算 Pearson 相关系数
    correlation = np.corrcoef(
        attention_map_upsampled.flatten(),
        dolp_resized.flatten()
    )[0, 1]
    
    print(f"\n注意力与 DoLP 的相关性分析：")
    print(f"  - Pearson 相关系数: {correlation:.4f}")
    if correlation > 0.3:
        print(f"  ✓ 强正相关：模型注意力与高偏振度区域（反射/边缘）高度一致")
    elif correlation > 0.1:
        print(f"  ⚠ 中等相关：模型注意力与偏振度区域有一定关联")
    else:
        print(f"  ⚠ 弱相关：模型注意力与偏振度区域关联较弱")


def list_base_names(scene_dir: Path) -> list:
    """
    列出某个场景下可用的 base_name 列表（根据 *_000.png 自动推断）
    
    Args:
        scene_dir: 某个 scene_id 目录，例如 polar_root/01
    
    Returns:
        base_names: 例如 ["0006", "0007", ...]
    """
    if not scene_dir.exists():
        return []
    
    candidates = sorted(scene_dir.glob("*_000.png"))
    base_names = []
    for p in candidates:
        stem = p.stem  # 例如 "0006_000"
        if stem.endswith("_000"):
            base = stem[:-4]  # 去掉 "_000"
            base_names.append(base)
    return base_names


def scan_all_scenes(polar_root: Path) -> list:
    """
    扫描所有场景，返回可用的场景ID列表
    
    Args:
        polar_root: 偏振图像根目录
    
    Returns:
        scene_ids: 可用的场景ID列表，例如 ["01", "02", "16", ...]
    """
    if not polar_root.exists():
        return []
    
    # 扫描所有子目录（场景目录）
    scene_ids = []
    for item in sorted(polar_root.iterdir()):
        if item.is_dir():
            # 检查该目录下是否有可用的图像
            base_names = list_base_names(item)
            if len(base_names) > 0:
                scene_ids.append(item.name)
    
    return scene_ids


def process_single_sample(
    model: ViTMAEForPreTraining,
    device: torch.device,
    polar_root: Path,
    scene_id: str,
    base_name: str,
    output_path: str,
) -> dict:
    """
    处理单个样本：加载图像、提取注意力、生成可视化
    
    Args:
        model: MAE 模型
        device: 设备
        polar_root: 偏振图像根目录
        scene_id: 场景ID
        base_name: 图像基础名称
        output_path: 输出图像路径
    
    Returns:
        metrics: 包含相关性等指标的字典
    """
    scene_dir = polar_root / scene_id
    
    # 构建4个角度图像路径
    polar_paths = {
        "I_0": scene_dir / f"{base_name}_000.png",
        "I_45": scene_dir / f"{base_name}_045.png",
        "I_90": scene_dir / f"{base_name}_090.png",
        "I_135": scene_dir / f"{base_name}_135.png",
    }
    
    # 检查文件是否存在（尝试带前导零和不带前导零的命名）
    missing = [name for name, path in polar_paths.items() if not path.exists()]
    if missing:
        polar_paths = {
            "I_0": scene_dir / f"{base_name}_0.png",
            "I_45": scene_dir / f"{base_name}_45.png",
            "I_90": scene_dir / f"{base_name}_90.png",
            "I_135": scene_dir / f"{base_name}_135.png",
        }
    
    # 最终再检查一遍
    for name, path in polar_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} 图像不存在: {path}")
    
    # 使用共享函数处理偏振图像（返回4通道：[Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]）
    physics_img = process_polar_images(polar_paths=polar_paths)  # (H, W, 4)
    
    # 预处理图像
    pixel_values = preprocess_image(physics_img, image_size=224)  # (1, 4, 224, 224)
    
    # 提取注意力图（不生成可视化，只返回相关性）
    attention_info = extract_attention_maps(model, pixel_values)
    cls_attention_2d = attention_info['cls_attention']
    image_size = attention_info['image_size']
    
    # 上采样注意力图
    attention_map_upsampled = upsample_attention_map(cls_attention_2d, target_size=image_size)
    
    # 提取 Intensity 和 DoLP
    if physics_img.shape[:2] != (image_size, image_size):
        intensity_pil = Image.fromarray((physics_img[:, :, 0] * 255).astype(np.uint8), mode='L')
        dolp_pil = Image.fromarray((physics_img[:, :, 1] * 255).astype(np.uint8), mode='L')
        intensity_pil = intensity_pil.resize((image_size, image_size), Image.BICUBIC)
        dolp_pil = dolp_pil.resize((image_size, image_size), Image.BICUBIC)
        intensity_img = np.array(intensity_pil) / 255.0
        dolp_img = np.array(dolp_pil) / 255.0
    else:
        intensity_img = physics_img[:, :, 0]
        dolp_img = physics_img[:, :, 1]
    
    # 计算相关性
    if dolp_img.shape != attention_map_upsampled.shape:
        dolp_pil = Image.fromarray((dolp_img * 255).astype(np.uint8), mode='L')
        dolp_pil = dolp_pil.resize((attention_map_upsampled.shape[1], attention_map_upsampled.shape[0]), Image.BICUBIC)
        dolp_resized = np.array(dolp_pil) / 255.0
    else:
        dolp_resized = dolp_img
    
    correlation = np.corrcoef(
        attention_map_upsampled.flatten(),
        dolp_resized.flatten()
    )[0, 1]
    
    # 生成可视化
    visualize_attention(
        model=model,
        pixel_values=pixel_values,
        physics_img=physics_img,
        output_path=output_path,
    )
    
    return {
        'scene_id': scene_id,
        'base_name': base_name,
        'correlation': float(correlation),
        'attention_mean': float(attention_map_upsampled.mean()),
        'attention_std': float(attention_map_upsampled.std()),
    }


def main():
    """主函数"""
    import argparse
    import random
    
    parser = argparse.ArgumentParser(description="Stage 1 MAE 注意力可视化脚本")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/stage1_encoder",
        help="Stage 1 检查点路径"
    )
    parser.add_argument(
        "--polar_root",
        type=str,
        default="/openbayes/input/input0/polar",
        help="偏振图像根目录"
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="测试场景ID（如果指定，则只处理该场景；否则自动扫描所有场景）"
    )
    parser.add_argument(
        "--base_name",
        type=str,
        default=None,
        help="测试图像基础名称（如果指定，则使用该图像；否则使用场景下的第一个可用图像）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/openbayes/home/train/vis/attention_vis.png",
        help="输出图像路径（当处理多个场景时，会自动添加场景ID和索引）。默认输出到 /openbayes/home/train/vis/"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/openbayes/home/train/vis",
        help="输出目录（所有图像都会保存到此目录）"
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=10,
        help="要处理的场景数量（默认10，当 scene_id 未指定时有效）"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="随机种子（用于可复现的场景选择）"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token（可选）"
    )
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 加载模型
    model = load_mae_model(args.checkpoint, hf_token=args.hf_token)
    model = model.to(device)
    
    polar_root = Path(args.polar_root)
    
    # 2. 确定要处理的场景和图像
    if args.scene_id is not None:
        # 单场景模式：使用指定的场景
        scene_ids = [args.scene_id]
        print(f"\n单场景模式: scene_id={args.scene_id}")
    else:
        # 多场景模式：扫描所有场景并随机选择（限制在 00-58 之间）
        print("\n" + "=" * 80)
        print("扫描所有场景...")
        print("=" * 80)
        
        all_scene_ids = scan_all_scenes(polar_root)
        if len(all_scene_ids) == 0:
            raise ValueError(f"在 {polar_root} 中未找到任何场景目录")
        
        print(f"✓ 找到 {len(all_scene_ids)} 个场景（全部）")
        
        # 仅保留 ID 在 00-58 范围内的场景
        filtered_scene_ids = []
        for sid in all_scene_ids:
            try:
                sid_int = int(sid)
            except ValueError:
                # 非数字 ID 直接跳过
                continue
            if 0 <= sid_int <= 58:
                filtered_scene_ids.append(sid)
        
        if len(filtered_scene_ids) == 0:
            raise ValueError(f"在 {polar_root} 中未找到 ID 在 [00, 58] 范围内的场景，请检查数据命名。")
        
        print(f"✓ 过滤后（ID 在 00-58 之间）的场景数: {len(filtered_scene_ids)}")
        
        # 随机选择场景（在过滤后的列表中）
        random.seed(args.random_seed)
        num_scenes = min(args.num_scenes, len(filtered_scene_ids))
        selected_scene_ids = random.sample(filtered_scene_ids, num_scenes)
        selected_scene_ids = sorted(selected_scene_ids)  # 排序以便输出有序
        
        scene_ids = selected_scene_ids
        print(f"✓ 从 00-58 中随机选择了 {num_scenes} 个场景: {scene_ids}")
    
    # 3. 为每个场景选择一张图像
    samples_to_process = []
    for scene_id in scene_ids:
        scene_dir = polar_root / scene_id
        base_names = list_base_names(scene_dir)
        
        if len(base_names) == 0:
            print(f"⚠ 警告: 场景 {scene_id} 下没有可用图像，跳过")
            continue
        
        # 选择图像：如果指定了 base_name 且该场景有该图像，使用它；否则使用第一个
        if args.base_name is not None and args.base_name in base_names:
            selected_base_name = args.base_name
        else:
            selected_base_name = base_names[0]  # 使用第一个可用图像
        
        samples_to_process.append((scene_id, selected_base_name))
        print(f"  - 场景 {scene_id}: 选择图像 {selected_base_name}")
    
    if len(samples_to_process) == 0:
        raise ValueError("没有找到任何可处理的样本")
    
    print(f"\n✓ 共准备处理 {len(samples_to_process)} 个样本")
    
    # 4. 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ 输出目录: {output_dir}")
    
    # 5. 处理每个样本
    print("\n" + "=" * 80)
    print("开始处理样本...")
    print("=" * 80)
    
    all_metrics = []
    output_base = Path(args.output)
    
    for idx, (scene_id, base_name) in enumerate(samples_to_process):
        print(f"\n[{idx+1}/{len(samples_to_process)}] 处理场景 {scene_id}, 图像 {base_name}...")
        
        # 生成输出路径：所有图像都保存到 output_dir
        if len(samples_to_process) == 1:
            # 单样本：使用用户指定的文件名，但保存到 output_dir
            output_filename = output_base.name
            output_path = str(output_dir / output_filename)
        else:
            # 多样本：自动生成文件名，包含场景ID和base_name
            stem = output_base.stem
            suffix = output_base.suffix or ".png"
            output_filename = f"{stem}_scene{scene_id}_{base_name}{suffix}"
            output_path = str(output_dir / output_filename)
        
        try:
            metrics = process_single_sample(
                model=model,
                device=device,
                polar_root=polar_root,
                scene_id=scene_id,
                base_name=base_name,
                output_path=output_path,
            )
            all_metrics.append(metrics)
            print(f"  ✓ 完成，相关性: {metrics['correlation']:.4f}")
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 6. 汇总统计
    if len(all_metrics) > 1:
        print("\n" + "=" * 80)
        print("多样本统计汇总：")
        print("=" * 80)
        
        correlations = [m['correlation'] for m in all_metrics]
        attention_means = [m['attention_mean'] for m in all_metrics]
        attention_stds = [m['attention_std'] for m in all_metrics]
        
        print(f"  - 处理样本数: {len(all_metrics)}")
        print(f"  - 平均相关性: {np.mean(correlations):.4f} ± {np.std(correlations):.4f}")
        print(f"  - 相关性范围: [{np.min(correlations):.4f}, {np.max(correlations):.4f}]")
        print(f"  - 平均注意力均值: {np.mean(attention_means):.4f}")
        print(f"  - 平均注意力标准差: {np.mean(attention_stds):.4f}")
        
        # 统计强相关、中等相关、弱相关的样本数
        strong_corr = sum(1 for c in correlations if c > 0.3)
        medium_corr = sum(1 for c in correlations if 0.1 < c <= 0.3)
        weak_corr = sum(1 for c in correlations if c <= 0.1)
        
        print(f"\n相关性分布：")
        print(f"  - 强相关 (>0.3): {strong_corr} 个样本 ({strong_corr/len(correlations)*100:.1f}%)")
        print(f"  - 中等相关 (0.1-0.3): {medium_corr} 个样本 ({medium_corr/len(correlations)*100:.1f}%)")
        print(f"  - 弱相关 (≤0.1): {weak_corr} 个样本 ({weak_corr/len(correlations)*100:.1f}%)")
        
        # 打印每个样本的详细信息
        print(f"\n各样本详细信息：")
        for m in all_metrics:
            corr_level = "强" if m['correlation'] > 0.3 else ("中" if m['correlation'] > 0.1 else "弱")
            print(f"  - 场景 {m['scene_id']}, 图像 {m['base_name']}: "
                  f"相关性={m['correlation']:.4f} ({corr_level})")
    
    print("\n" + "=" * 80)
    print("注意力可视化完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()


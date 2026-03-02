"""
数据集通用工具函数
用于处理偏振图像，支持两种数据格式：
1. Glare数据集：4张角度图像 (I_0, I_45, I_90, I_135) → 计算Stokes参数 → 4通道物理参数
2. Glass/Water数据集：已有3通道物理参数 (Intensity, DoLP, AoLP) → 转换为4通道

设计目标：
- 统一所有阶段（Stage 1, 2, 3）的数据处理逻辑
- 支持未来扩展（直接输入3通道数据）
- 确保归一化一致性（所有通道值范围 [0, 1]）

关键改进（v2）：
- 从3通道 [I, DoLP, AoLP] 升级到4通道 [I, DoLP, sin(2*AoLP), cos(2*AoLP)]
- 解决 AoLP 的周期性边界问题（0° 和 180° 在物理上等价，但数值上相差很大）
- sin(2*AoLP) 和 cos(2*AoLP) 能够唯一表示偏振角，避免边界不连续问题
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional
from PIL import Image


def process_polar_images(
    polar_paths: Optional[Dict[str, Path]] = None,
    intensity_path: Optional[Path] = None,
    dolp_path: Optional[Path] = None,
    aolp_path: Optional[Path] = None,
) -> np.ndarray:
    """
    通用的偏振图像处理函数，将输入转换为4通道物理参数
    
    支持的两种输入格式：
    1. Glare数据集：4张角度图像 (I_0, I_45, I_90, I_135) → 计算Stokes参数 → [Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]
    2. Glass/Water数据集：已有3通道物理参数 (Intensity, DoLP, AoLP) → 转换为4通道
    
    物理参数计算（Glare数据集）：
    - 计算Stokes参数：S0 = (I_0 + I_90) / 2.0, S1 = I_0 - I_90, S2 = I_45 - I_135
    - Intensity = S0 / 255.0（归一化到 [0, 1]）
    - DoLP = sqrt(S1^2 + S2^2) / S0（偏振度，归一化到 [0, 1]）
    - AoLP = 0.5 * arctan2(S2, S1)（偏振角，范围 [-pi/2, pi/2]）
    - sin_2aolp = sin(2 * AoLP)，范围 [-1, 1]，映射到 [0, 1]
    - cos_2aolp = cos(2 * AoLP)，范围 [-1, 1]，映射到 [0, 1]
    
    为什么使用 sin(2*AoLP) 和 cos(2*AoLP)？
    - AoLP 具有周期性：0° 和 180° 在物理上等价（偏振方向相同）
    - 直接使用 AoLP 会导致边界不连续问题（0° 和 180° 数值相差很大）
    - sin(2*AoLP) 和 cos(2*AoLP) 能够唯一表示偏振角，避免周期性边界问题
    - 这种表示方法在深度学习中更稳定，梯度更平滑
    
    Args:
        polar_paths: 包含4个角度图像路径的字典
            {'I_0': Path, 'I_45': Path, 'I_90': Path, 'I_135': Path}
            用于Glare数据集（从4张角度图计算）
        intensity_path: Intensity图像路径（用于Glass/Water数据集，已有物理参数）
        dolp_path: DoLP图像路径（用于Glass/Water数据集，已有物理参数）
        aolp_path: AoLP图像路径（用于Glass/Water数据集，已有物理参数）
    
    Returns:
        4通道物理参数图像 (H, W, 4)，numpy数组，dtype=float32
        - 值范围：[0, 1]（所有通道都已归一化）
        - 通道顺序：[Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]
        - 通道0：Intensity（总光强）
        - 通道1：DoLP（偏振度，Degree of Linear Polarization）
        - 通道2：sin(2*AoLP)（偏振角的正弦表示，映射到 [0, 1]）
        - 通道3：cos(2*AoLP)（偏振角的余弦表示，映射到 [0, 1]）
    
    Raises:
        FileNotFoundError: 如果输入图像文件不存在
        ValueError: 如果提供的参数不足（需要polar_paths或intensity+dolp+aolp）
    """
    
    if polar_paths is not None:
        # ========== 情况1：Glare数据集 - 从4张角度图像计算Stokes参数 ==========
        
        # 提取4个角度图像路径
        i0_path = polar_paths['I_0']
        i45_path = polar_paths['I_45']
        i90_path = polar_paths['I_90']
        i135_path = polar_paths['I_135']
        
        # 检查文件是否存在
        for channel_name, polar_path in [('I_0', i0_path), ('I_45', i45_path), 
                                         ('I_90', i90_path), ('I_135', i135_path)]:
            if not polar_path.exists():
                raise FileNotFoundError(
                    f"偏振图像不存在: {polar_path} (通道: {channel_name})"
                )
        
        # 加载为灰度图像并转换为numpy数组（float32精度）
        i0 = np.array(Image.open(i0_path).convert('L')).astype(np.float32)
        i45 = np.array(Image.open(i45_path).convert('L')).astype(np.float32)
        i90 = np.array(Image.open(i90_path).convert('L')).astype(np.float32)
        i135 = np.array(Image.open(i135_path).convert('L')).astype(np.float32)
        
        # 检查图像尺寸是否一致
        shapes = [i0.shape, i45.shape, i90.shape, i135.shape]
        if not all(s == shapes[0] for s in shapes):
            raise ValueError(
                f"4个角度图像的尺寸不一致: I_0={i0.shape}, I_45={i45.shape}, "
                f"I_90={i90.shape}, I_135={i135.shape}"
            )
        
        # 计算Stokes参数（Stokes Parameters）
        eps = 1e-6  # 防止除以零的极小值
        
        # S0：总光强（使用简化公式，更准确的是 S0 = (I_0 + I_45 + I_90 + I_135) / 2.0）
        S0 = (i0 + i90) / 2.0
        
        # S1, S2：用于计算偏振度和偏振角
        S1 = i0 - i90
        S2 = i45 - i135
        
        # 计算物理通道
        
        # Intensity（总光强）：归一化到 [0, 1]
        # S0的范围是 [0, 255]，除以255归一化
        Intensity = np.clip(S0 / 255.0, 0, 1)
        
        # DoLP（偏振度，Degree of Linear Polarization）：归一化到 [0, 1]
        # DoLP = sqrt(S1^2 + S2^2) / S0
        DoLP = np.sqrt(S1**2 + S2**2) / (S0 + eps)
        DoLP = np.clip(DoLP, 0, 1)  # 截断异常值（理论上DoLP <= 1）
        
        # AoLP（偏振角，Angle of Linear Polarization）：计算原始角度值
        # AoLP = 0.5 * arctan2(S2, S1)，范围 [-pi/2, pi/2]
        AoLP_raw = 0.5 * np.arctan2(S2, S1 + eps)
        
        # ========== 关键改进：使用 sin(2*AoLP) 和 cos(2*AoLP) 表示偏振角 ==========
        # 
        # 为什么这样做？
        # 1. AoLP 具有周期性：0° 和 180° 在物理上等价（偏振方向相同）
        # 2. 直接使用 AoLP 会导致边界不连续问题：
        #    - 0° 和 180° 在数值上相差很大（0.0 vs 1.0）
        #    - 但它们在物理上表示相同的偏振方向
        #    - 这会导致模型在边界处学习困难，梯度不稳定
        # 3. sin(2*AoLP) 和 cos(2*AoLP) 的优势：
        #    - 能够唯一表示偏振角（在 [-pi/2, pi/2] 范围内）
        #    - 避免了周期性边界问题
        #    - 在深度学习中更稳定，梯度更平滑
        #    - 符合偏振物理学的标准表示方法
        # 
        # 计算 sin(2*AoLP) 和 cos(2*AoLP)
        # 注意：AoLP_raw 的范围是 [-pi/2, pi/2]，所以 2*AoLP_raw 的范围是 [-pi, pi]
        sin_2aolp = np.sin(2 * AoLP_raw)  # 范围 [-1, 1]
        cos_2aolp = np.cos(2 * AoLP_raw)  # 范围 [-1, 1]
        
        # 将 sin 和 cos 从 [-1, 1] 映射到 [0, 1]
        # 公式：(x + 1) / 2，这样 -1 -> 0, 0 -> 0.5, 1 -> 1
        sin_2aolp_normalized = (sin_2aolp + 1.0) / 2.0
        cos_2aolp_normalized = (cos_2aolp + 1.0) / 2.0
        
        # 确保值在 [0, 1] 范围内（理论上已经在范围内，但为了安全）
        sin_2aolp_normalized = np.clip(sin_2aolp_normalized, 0, 1)
        cos_2aolp_normalized = np.clip(cos_2aolp_normalized, 0, 1)
        
    elif intensity_path is not None and dolp_path is not None and aolp_path is not None:
        # ========== 情况2：Glass/Water数据集 - 直接加载已有物理参数 ==========
        
        # 检查文件是否存在
        for name, path in [('Intensity', intensity_path), ('DoLP', dolp_path), ('AoLP', aolp_path)]:
            if not path.exists():
                raise FileNotFoundError(f"物理参数图像不存在: {path} ({name})")
        
        # 加载为灰度图像并归一化到 [0, 1]
        # 注意：这里假设输入图像的值范围是 [0, 255]
        Intensity = np.array(Image.open(intensity_path).convert('L')).astype(np.float32) / 255.0
        Intensity = np.clip(Intensity, 0, 1)
        
        DoLP = np.array(Image.open(dolp_path).convert('L')).astype(np.float32) / 255.0
        DoLP = np.clip(DoLP, 0, 1)
        
        # 加载 AoLP（值范围 [0, 1]），需要转换回角度值 [-pi/2, pi/2]
        AoLP_normalized = np.array(Image.open(aolp_path).convert('L')).astype(np.float32) / 255.0
        AoLP_normalized = np.clip(AoLP_normalized, 0, 1)
        
        # 将归一化的 AoLP 转换回角度值
        # 原始映射：AoLP = (AoLP_raw + pi/2) / pi，所以 AoLP_raw = AoLP * pi - pi/2
        AoLP_raw = AoLP_normalized * np.pi - (np.pi / 2.0)
        
        # 计算 sin(2*AoLP) 和 cos(2*AoLP)
        sin_2aolp = np.sin(2 * AoLP_raw)  # 范围 [-1, 1]
        cos_2aolp = np.cos(2 * AoLP_raw)  # 范围 [-1, 1]
        
        # 将 sin 和 cos 从 [-1, 1] 映射到 [0, 1]
        sin_2aolp_normalized = (sin_2aolp + 1.0) / 2.0
        cos_2aolp_normalized = (cos_2aolp + 1.0) / 2.0
        
        # 确保值在 [0, 1] 范围内
        sin_2aolp_normalized = np.clip(sin_2aolp_normalized, 0, 1)
        cos_2aolp_normalized = np.clip(cos_2aolp_normalized, 0, 1)
        
        # 检查图像尺寸是否一致
        shapes = [Intensity.shape, DoLP.shape, sin_2aolp_normalized.shape, cos_2aolp_normalized.shape]
        if not all(s == shapes[0] for s in shapes):
            raise ValueError(
                f"4个物理参数图像的尺寸不一致: Intensity={Intensity.shape}, "
                f"DoLP={DoLP.shape}, sin(2*AoLP)={sin_2aolp_normalized.shape}, "
                f"cos(2*AoLP)={cos_2aolp_normalized.shape}"
            )
        
    else:
        raise ValueError(
            "必须提供 polar_paths (Glare数据集：4张角度图) "
            "或 intensity_path+dolp_path+aolp_path (Glass/Water数据集：已有3通道物理参数)"
        )
    
    # 堆叠成4通道图像 (H, W, 4)
    # 通道顺序：[Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]
    physics_img = np.stack([
        Intensity, 
        DoLP, 
        sin_2aolp_normalized, 
        cos_2aolp_normalized
    ], axis=2)
    
    # 确保数据类型为float32
    physics_img = physics_img.astype(np.float32)
    
    return physics_img


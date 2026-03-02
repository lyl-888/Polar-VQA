"""
Stage 1 数据集：无标签偏振图像预训练
用于 MAE (Masked Autoencoder) 预训练偏振编码器

数据格式：
- 输入：偏振图像的4个角度（I_0, I_45, I_90, I_135）→ 计算Stokes参数 → 4通道（I, DoLP, sin(2*AoLP), cos(2*AoLP)）
- 输出：仅返回 pixel_values（无标签，3通道：DoLP, sin(2*AoLP), cos(2*AoLP)）
- 数据增强：随机裁剪、随机水平翻转

注意：
- 使用共享的 process_polar_images 函数将4张角度图转换为4通道物理参数，然后去掉Intensity通道
- 3通道输入直接使用3通道预训练权重（无需扩展策略）
- Intensity通道被移除，因为Stage 2会使用CLIP提取RGB特征（包含光强信息）
- 使用 sin(2*AoLP) 和 cos(2*AoLP) 解决 AoLP 的周期性边界问题
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

# 导入共享的处理函数
from dataset_common import process_polar_images


class PolarMAEDataset(Dataset):
    """
    Stage 1 数据集：用于 MAE 预训练的偏振图像数据集
    
    功能：
    1. 扫描偏振图像目录，找到所有4个角度图像对
    2. 使用 process_polar_images 函数将4张角度图转换为4通道物理参数（I, DoLP, sin(2*AoLP), cos(2*AoLP)）
    3. 去掉Intensity通道，只保留后3个通道（DoLP, sin(2*AoLP), cos(2*AoLP)）
    4. 应用数据增强（随机裁剪、随机翻转）
    5. 返回像素值（无标签，3通道，用于自监督学习）
    
    Args:
        polar_root: 偏振图像根目录
        image_size: 输入图像尺寸（默认224，MAE通常使用224）
        is_train: 是否为训练集（决定是否应用数据增强）
    
    注意：
        - 输出3通道物理参数（DoLP, sin(2*AoLP), cos(2*AoLP)）
        - Intensity通道被移除，因为Stage 2会使用CLIP提取RGB特征（包含光强信息）
        - 使用 sin(2*AoLP) 和 cos(2*AoLP) 解决 AoLP 的周期性边界问题
        - 3通道输入直接使用3通道预训练权重（无需扩展策略）
    """
    
    def __init__(
        self,
        polar_root: str,
        image_size: int = 224,
        is_train: bool = True,
        use_pt_data: bool = False,  # 是否使用预处理的 .pt 文件
        pt_root: Optional[str] = None,  # .pt 文件根目录（如果 use_pt_data=True）
    ):
        self.image_size = image_size
        self.is_train = is_train
        self.use_pt_data = use_pt_data
        
        # ========== 数据加载模式选择 ==========
        if use_pt_data:
            # 模式1：使用预处理的 .pt 文件（快速模式）
            if pt_root is None:
                raise ValueError("use_pt_data=True 时必须提供 pt_root 参数")
            
            pt_path = Path(pt_root)
            if not pt_path.is_absolute():
                pt_path_resolved = pt_path.resolve()
                if not pt_path_resolved.exists():
                    # 尝试在上级目录查找
                    cwd = Path.cwd()
                    parent_dir = cwd.parent
                    alt_path = parent_dir / pt_path
                    if alt_path.exists():
                        pt_path = alt_path.resolve()
                        print(f"✓ 在上级目录找到 PT 数据路径: {pt_path}")
                    else:
                        pt_path = pt_path_resolved
                else:
                    pt_path = pt_path_resolved
            
            self.pt_root = pt_path
            self.samples = self._scan_pt_files()
            print(f"✓ Stage 1 数据集加载完成（PT模式）: {len(self.samples)} 个样本")
            print(f"  ⚡ 使用预处理的 .pt 文件，数据加载速度大幅提升！")
        else:
            # 模式2：从原始PNG图像加载（传统模式）
            # 转换为绝对路径（避免在DataLoader worker进程中路径问题）
            polar_path = Path(polar_root)
            if not polar_path.is_absolute():
                polar_path_resolved = polar_path.resolve()
                if not polar_path_resolved.exists():
                    # 尝试在上级目录查找
                    cwd = Path.cwd()
                    parent_dir = cwd.parent
                    alt_path = parent_dir / polar_path
                    if alt_path.exists():
                        polar_path = alt_path.resolve()
                        print(f"✓ 在上级目录找到 Polar 路径: {polar_path}")
                    else:
                        polar_path = polar_path_resolved
                else:
                    polar_path = polar_path_resolved
            
            self.polar_root = polar_path
            self.samples = self._scan_polar_images()
            print(f"✓ Stage 1 数据集加载完成（PNG模式）: {len(self.samples)} 个样本")
        
        if len(self.samples) == 0:
            if use_pt_data:
                raise ValueError(f"未找到任何 .pt 文件！请检查目录: {self.pt_root}")
            else:
                raise ValueError(f"未找到任何偏振图像对！请检查目录: {self.polar_root}")
        
        # ========== 数据增强配置（关键优化）==========
        # 
        # ⚠️ 重要：偏振物理数据的归一化策略
        # 
        # 数据范围分析：
        # 1. process_polar_images 返回的4通道数据，所有通道都在 [0, 1] 范围内：
        #    - Intensity: [0, 1]（S0 / 255.0，归一化后的光强）
        #    - DoLP: [0, 1]（偏振度，理论最大值1）
        #    - sin(2*AoLP): [-1, 1] -> 映射到 [0, 1]（(sin + 1) / 2）
        #    - cos(2*AoLP): [-1, 1] -> 映射到 [0, 1]（(cos + 1) / 2）
        # 
        # 2. 数值量级一致性：
        #    ✅ 所有通道都在 [0, 1] 范围内，没有量级差异问题
        #    ✅ ToTensor() 会将 PIL Image 的 [0, 255] uint8 转换为 [0, 1] float32
        #    ✅ 最终输入到模型的数据范围是 [0, 1]，与预训练权重的期望范围一致
        # 
        # 3. 分布差异（不影响训练）：
        #    - Intensity: 分布取决于场景光照，可能集中在某个范围（例如 0.3-0.7）
        #    - DoLP: 通常集中在较低值（例如 0.05-0.3），但这是物理特性，应该保留
        #    - sin(2*AoLP), cos(2*AoLP): 分布相对均匀
        #    - 这些分布差异是正常的，反映了物理量的真实特性
        # 
        # 4. 为什么不使用 ImageNet 归一化：
        #    - ImageNet 归一化（mean=[0.485, ...], std=[0.229, ...]）是为 RGB 设计的
        #    - 偏振物理数据的统计特性完全不同
        #    - 使用 ImageNet 归一化会：
        #      * 错误地假设数据分布，导致梯度方向错误
        #      * 破坏物理量的原始语义（DoLP 和 AoLP 的物理意义）
        #      * 使训练损失难以收敛
        # 
        # ✅ 当前方案（已优化）：
        # - 保持原始值范围 [0, 1]，所有通道统一
        # - 配合 norm_pix_loss=False，让模型直接学习物理数值
        # - 这样既保证了数值量级一致性，又保留了物理量的绝对意义
        # 
        # 数据增强调整：
        # - RandomResizedCrop scale 从 (0.2, 1.0) 调整为 (0.5, 1.0)
        #   原因：0.2 的下界过于激进，对于 6312 个样本的数据集，
        #   结合 MAE 的 masking（通常 mask 75%），会导致模型看到的信息太少，
        #   训练不稳定。0.5 的下界更保守，保证每个 patch 都有足够的上下文信息。
        # 
        # ⚠️ 注意：现在输入是3通道（DoLP, sin(2*AoLP), cos(2*AoLP)），Intensity通道已移除
        # torchvision 的 transforms 支持任意通道数，所以 RandomResizedCrop 和 ToTensor 都能正常工作
        # 
        # ========== 数据增强配置 ==========
        # 
        # 注意：如果 use_pt_data=True，我们需要对张量应用数据增强
        # torchvision.transforms 主要针对 PIL Image，但我们可以使用 functional API
        # 或者先转换为PIL Image再应用transform
        # 
        if self.is_train:
            # 训练模式：应用数据增强
            if use_pt_data:
                # PT模式：使用 functional API 直接对 tensor 做增强，避免 PIL 转换
                # 
                # ⚠️ 关键修复：避免 tensor -> uint8 -> PIL -> tensor 的转换
                # 
                # 问题分析：
                # 1. 如果 .pt 文件中的 sin/cos 是 [-1, 1] 范围（未归一化）
                # 2. 乘以 255 后，负数会变成负数（如 -0.5 * 255 = -127.5）
                # 3. astype(uint8) 会发生溢出（wrap-around），-127 变成 129
                # 4. 结果：原本表示角度的负数值被错误地变成了正数值，彻底破坏偏振角的物理意义
                # 
                # 解决方案：
                # - 使用 transforms.functional 直接操作 tensor，保持 float32 精度
                # - 在增强前检查并修复数据范围（如果 sin/cos 是 [-1, 1]，映射到 [0, 1]）
                # - 避免任何 uint8 转换，确保数据完整性
                # 
                self.transform = None  # 不使用 Compose，在 __getitem__ 中手动应用增强
                self.use_tensor_augmentation = True  # 标记使用 tensor 增强
            else:
                # PNG模式：从PIL Image开始
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(0.5, 1.0),
                        ratio=(0.75, 1.33),
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),  # 转换为张量，自动将 [0, 255] -> [0.0, 1.0]，dtype=float32
                ])
                self.use_tensor_augmentation = False
        else:
            # 验证模式：只resize，不应用数据增强
            if use_pt_data:
                # PT模式：数据已经是正确尺寸的张量，不需要transform
                self.transform = None
                self.use_tensor_augmentation = False
            else:
                # PNG模式：resize到目标尺寸
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),  # 转换为张量，[0, 255] -> [0.0, 1.0]，dtype=float32
                ])
                self.use_tensor_augmentation = False
        
        # ⚠️ 已移除 ImageNet 归一化：不使用 transforms.Normalize
        # 
        # 原因：
        # 1. 偏振物理数据的统计特性与 ImageNet RGB 图像完全不同
        # 2. 所有通道已经在 [0, 1] 范围内，数值量级一致，无需额外归一化
        # 3. 配合 norm_pix_loss=False，让模型直接学习物理数值，保留绝对意义
        # 
        # 数据流程确认：
        # - process_polar_images: 返回 [0, 1] 范围的 numpy 数组
        # - 转换为 PIL Image: 乘以 255，变成 [0, 255] uint8
        # - ToTensor(): 除以 255，变回 [0, 1] float32
        # - 最终输入: (B, 3, H, W)，值范围 [0, 1]，所有通道统一（已移除Intensity通道）
    
    def _scan_polar_images(self) -> List[Dict[str, Path]]:
        """
        扫描偏振图像目录，找到所有完整的4通道图像对
        
        支持的命名格式：
        1. {scene_id}/{base_name}_000.png, {base_name}_045.png, {base_name}_090.png, {base_name}_135.png
        2. {scene_id}/{base_name}_0.png, {base_name}_45.png, {base_name}_90.png, {base_name}_135.png
        
        Returns:
            样本列表，每个元素包含4个通道的路径
        """
        samples = []
        
        # 扫描所有场景目录
        if not self.polar_root.exists():
            return samples
        
        # 遍历所有子目录（场景目录）
        for scene_dir in sorted(self.polar_root.iterdir()):
            if not scene_dir.is_dir():
                continue
            
            scene_id = scene_dir.name
            
            # 获取该场景下的所有图像文件
            image_files = sorted([f for f in scene_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
            
            # 按基础文件名分组（去除角度后缀）
            base_names = {}
            for img_file in image_files:
                # 提取基础文件名（去除角度后缀）
                base_name = self._extract_base_name(img_file.stem)
                if base_name not in base_names:
                    base_names[base_name] = {}
                
                # 识别角度
                angle = self._extract_angle(img_file.stem)
                if angle is not None:
                    base_names[base_name][angle] = img_file
            
            # 检查每个基础文件名是否有完整的4个角度
            for base_name, angle_files in base_names.items():
                required_angles = [0, 45, 90, 135]
                if all(angle in angle_files for angle in required_angles):
                    samples.append({
                        'scene_id': scene_id,
                        'base_name': base_name,
                        'I_0': angle_files[0],
                        'I_45': angle_files[45],
                        'I_90': angle_files[90],
                        'I_135': angle_files[135],
                    })
        
        return samples
    
    def _extract_base_name(self, filename: str) -> str:
        """
        从文件名中提取基础名称（去除角度后缀）
        
        例如：
        - "0002_000" -> "0002"
        - "0002_045" -> "0002"
        - "0002_0" -> "0002"
        - "0002_45" -> "0002"
        """
        # 尝试匹配角度模式：_000, _045, _090, _135, _0, _45, _90, _135
        import re
        patterns = [
            r'_000$', r'_045$', r'_090$', r'_135$',
            r'_0$', r'_45$', r'_90$', r'_135$',
        ]
        
        base_name = filename
        for pattern in patterns:
            base_name = re.sub(pattern, '', base_name)
            if base_name != filename:
                break
        
        return base_name
    
    def _extract_angle(self, filename: str) -> int:
        """
        从文件名中提取角度
        
        例如：
        - "0002_000" -> 0
        - "0002_045" -> 45
        - "0002_0" -> 0
        - "0002_45" -> 45
        """
        import re
        # 尝试匹配三位数角度
        match = re.search(r'_(\d{3})$', filename)
        if match:
            angle = int(match.group(1))
            if angle in [0, 45, 90, 135]:
                return angle
        
        # 尝试匹配一位或两位数角度
        match = re.search(r'_(\d{1,2})$', filename)
        if match:
            angle = int(match.group(1))
            if angle in [0, 45, 90, 135]:
                return angle
        
        return None
    
    def _scan_pt_files(self) -> List[Dict[str, Path]]:
        """
        扫描 .pt 文件目录，找到所有预处理好的张量文件
        
        Returns:
            样本列表，每个元素包含场景ID、基础名称和.pt文件路径
        """
        samples = []
        
        if not self.pt_root.exists():
            return samples
        
        # 遍历所有子目录（场景目录）
        for scene_dir in sorted(self.pt_root.iterdir()):
            if not scene_dir.is_dir():
                continue
            
            scene_id = scene_dir.name
            
            # 获取该场景下的所有 .pt 文件
            pt_files = sorted([
                f for f in scene_dir.iterdir() 
                if f.suffix == '.pt'
            ])
            
            # 为每个 .pt 文件创建一个样本
            for pt_file in pt_files:
                base_name = pt_file.stem  # 去除 .pt 后缀
                samples.append({
                    'scene_id': scene_id,
                    'base_name': base_name,
                    'pt_path': pt_file,
                })
        
        return samples
    
    def _load_polar_images(self, sample: Dict[str, Path]) -> Image.Image:
        """
        加载4个偏振角度图像并转换为4通道物理参数（I, DoLP, sin(2*AoLP), cos(2*AoLP)）
        
        使用共享的 process_polar_images 函数计算Stokes参数
        
        Args:
            sample: 包含4个角度图像路径的字典 {'I_0': Path, 'I_45': Path, 'I_90': Path, 'I_135': Path}
        
        Returns:
            PIL Image对象（4通道RGBA格式，值范围[0, 255]），通道顺序：[Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]
            注意：
            - 虽然使用RGBA格式，但第4个通道是cos(2*AoLP)，不是透明度
            - 此方法返回4通道，但在 __getitem__ 中会去掉Intensity通道，最终返回3通道数据
        """
        # 构建polar_paths字典
        polar_paths = {
            'I_0': sample['I_0'],
            'I_45': sample['I_45'],
            'I_90': sample['I_90'],
            'I_135': sample['I_135'],
        }
        
        # 使用共享的process_polar_images函数计算Stokes参数
        physics_img = process_polar_images(polar_paths=polar_paths)  # (H, W, 4)，值范围[0, 1]
        
        # 转换为PIL Image（需要uint8格式）
        # 注意：PIL Image 不支持4通道RGB，需要使用RGBA模式
        # 虽然第4个通道是cos(2*AoLP)而不是透明度，但RGBA格式可以存储4通道数据
        physics_img_uint8 = (physics_img * 255).astype(np.uint8)
        physics_pil = Image.fromarray(physics_img_uint8, mode='RGBA')
        
        return physics_pil
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            包含 pixel_values 的字典
            - pixel_values: (3, H, W) 的偏振图像张量，3通道：[DoLP, sin(2*AoLP), cos(2*AoLP)]
              注意：Intensity通道已移除
        """
        sample = self.samples[idx]
        
        if self.use_pt_data:
            # ========== 模式1：从预处理的 .pt 文件加载（快速模式）==========
            # 
            # 优势：
            # - 跳过 IO 和 Stokes 计算，大幅提升数据加载速度
            # - GPU 利用率从 ~30% 提升到 90%+
            # - 训练速度从 1.0 it/s 提升到 4.0-5.0 it/s
            # 
            pt_path = sample['pt_path']
            
            # 加载张量（可能是 (4, H, W) 或 (3, H, W)，值范围 [0, 1]，dtype=float32）
            pixel_values = torch.load(pt_path, map_location='cpu')
            
            # 确保数据类型和形状正确
            if pixel_values.dtype != torch.float32:
                pixel_values = pixel_values.float()
            
            # 确保形状是 (3, H, W) 或 (4, H, W)
            if len(pixel_values.shape) != 3:
                raise ValueError(
                    f"加载的 .pt 文件形状不正确: {pixel_values.shape}, "
                    f"期望: (3或4, H, W)"
                )
            
            # ⚠️ 关键修改：如果数据是4通道，去掉Intensity通道（第0个通道）
            if pixel_values.shape[0] == 4:
                pixel_values = pixel_values[1:4, :, :]  # 只保留 DoLP, sin, cos
                # print(f"  ⚠ 注意: .pt 文件包含4通道，已自动移除Intensity通道")
            
            # 确保最终是3通道
            if pixel_values.shape[0] != 3:
                raise ValueError(
                    f"加载的 .pt 文件通道数不正确: {pixel_values.shape[0]}, "
                    f"期望: 3 (DoLP, sin, cos) 或 4 (会自动移除Intensity)"
                )
            
            # 应用数据增强（训练模式）或验证模式处理
            if self.is_train:
                # 训练模式：应用数据增强
                if self.use_tensor_augmentation:
                    # ⚠️ 关键修复：使用 functional API 直接对 tensor 做增强
                    # 避免 tensor -> uint8 -> PIL -> tensor 的转换，防止数据破坏
                    
                    # 1. 确保数据范围在 [0, 1]（如果 sin/cos 是 [-1, 1]，需要映射）
                    # 检查第2和第3通道（sin/cos）是否包含负数（现在pixel_values已经是3通道）
                    if pixel_values.shape[0] == 3:
                        sin_cos_channels = pixel_values[1:3, ...]  # (2, H, W) - DoLP在第0通道，sin/cos在第1-2通道
                        if sin_cos_channels.min() < 0:
                            # 如果 sin/cos 是 [-1, 1] 范围，映射到 [0, 1]
                            print(f"⚠ 警告: PT 数据中的 sin/cos 通道包含负值 (min={sin_cos_channels.min():.4f})，"
                                  f"正在映射到 [0, 1] 范围")
                            pixel_values[1:3, ...] = (sin_cos_channels + 1.0) / 2.0
                    
                    # 2. 确保所有通道都在 [0, 1] 范围内
                    pixel_values = torch.clamp(pixel_values, 0.0, 1.0)
                    
                    # 3. 使用 functional API 直接对 tensor 做增强
                    import random
                    
                    # 获取原始尺寸
                    _, h, w = pixel_values.shape
                    
                    # RandomResizedCrop: 随机裁剪并resize到目标尺寸
                    scale = random.uniform(0.5, 1.0)
                    ratio = random.uniform(0.75, 1.33)
                    
                    # 计算裁剪尺寸
                    crop_h = int(h * scale)
                    crop_w = int(w * scale * ratio)
                    crop_h = min(crop_h, h)
                    crop_w = min(crop_w, w)
                    
                    # 随机选择裁剪位置
                    top = random.randint(0, max(0, h - crop_h))
                    left = random.randint(0, max(0, w - crop_w))
                    
                    # 裁剪
                    pixel_values = F.crop(pixel_values, top, left, crop_h, crop_w)
                    
                    # Resize 到目标尺寸
                    pixel_values = F.resize(pixel_values, [self.image_size, self.image_size], 
                                          interpolation=F.InterpolationMode.BILINEAR)
                    
                    # RandomHorizontalFlip
                    if random.random() < 0.5:
                        pixel_values = F.hflip(pixel_values)
                    
                elif self.transform is not None:
                    # PNG模式：使用 Compose transform（从 PIL Image 开始）
                    # 将张量转换为 PIL Image（仅用于兼容性，PT模式不应该走这里）
                    tensor_np = pixel_values.permute(1, 2, 0).numpy()  # (H, W, 3)
                    # 确保数据在 [0, 1] 范围内
                    tensor_np = np.clip(tensor_np, 0, 1)
                    tensor_uint8 = (tensor_np * 255).astype(np.uint8)
                    physics_pil = Image.fromarray(tensor_uint8, mode='RGB')  # 3通道使用RGB模式
                    
                    # 应用数据增强
                    pixel_values = self.transform(physics_pil)  # (3, H, W)
            else:
                # 验证模式：数据已经是正确尺寸，直接使用
                # 如果需要resize，可以在这里添加（但预处理时应该已经是224x224）
                pass
            
            # 确保数据类型为 float32
            if pixel_values.dtype != torch.float32:
                pixel_values = pixel_values.float()
            
        else:
            # ========== 模式2：从原始PNG图像加载（传统模式）==========
            # 加载4通道物理参数图像（使用process_polar_images函数）
            # 注意：_load_polar_images 返回4通道，但后续会去掉Intensity通道
            physics_pil = self._load_polar_images(sample)  # PIL Image (RGBA格式，4通道)
            
            # 应用数据增强（transform会处理resize、crop等）
            # 注意：transform 只包含 ToTensor，不包含 Normalize
            # 输出值范围：[0.0, 1.0]，dtype=float32
            pixel_values = self.transform(physics_pil)  # (4, H, W)，值范围 [0, 1]，dtype=float32
            
            # ⚠️ 关键修改：去掉Intensity通道（第0个通道），只保留后3个通道
            # 这样最终返回的是3通道数据（DoLP, sin, cos），与模型期望的输入一致
            pixel_values = pixel_values[1:4, :, :]  # (3, H, W) - 只保留 DoLP, sin, cos
            
            # 确保数据类型为 float32（避免精度问题）
            if pixel_values.dtype != torch.float32:
                pixel_values = pixel_values.float()
        
        # [数据验证] 确保数据范围正确
        # 所有通道应该在 [0, 1] 范围内，数值量级一致
        if pixel_values.min() < 0 or pixel_values.max() > 1:
            print(f"⚠ 警告: pixel_values 超出 [0, 1] 范围: min={pixel_values.min():.4f}, max={pixel_values.max():.4f}")
            # 强制裁剪到 [0, 1]（防止异常值）
            pixel_values = torch.clamp(pixel_values, 0.0, 1.0)
        
        # [诊断] 打印第一个样本的数据统计（用于诊断loss低的问题）
        if idx == 0 and self.is_train:
            print(f"\n📊 数据统计（第一个训练样本，用于诊断loss）:")
            dolp_min, dolp_max = pixel_values[0].min().item(), pixel_values[0].max().item()
            dolp_mean, dolp_std = pixel_values[0].mean().item(), pixel_values[0].std().item()
            sin_min, sin_max = pixel_values[1].min().item(), pixel_values[1].max().item()
            sin_mean, sin_std = pixel_values[1].mean().item(), pixel_values[1].std().item()
            cos_min, cos_max = pixel_values[2].min().item(), pixel_values[2].max().item()
            cos_mean, cos_std = pixel_values[2].mean().item(), pixel_values[2].std().item()
            print(f"  - DoLP通道:  范围=[{dolp_min:.4f}, {dolp_max:.4f}], "
                  f"均值={dolp_mean:.4f}, 标准差={dolp_std:.4f}")
            print(f"  - sin通道:   范围=[{sin_min:.4f}, {sin_max:.4f}], "
                  f"均值={sin_mean:.4f}, 标准差={sin_std:.4f}")
            print(f"  - cos通道:   范围=[{cos_min:.4f}, {cos_max:.4f}], "
                  f"均值={cos_mean:.4f}, 标准差={cos_std:.4f}")
            print(f"  💡 说明: 如果DoLP均值很小（<0.2），loss低是正常的（数据值域小）")
        
        return {
            "pixel_values": pixel_values,  # (3, H, W) - [DoLP, sin(2*AoLP), cos(2*AoLP)]，值范围 [0, 1]，dtype=float32
        }


def collate_fn_stage1(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Stage 1 的 collate 函数
    
    功能：
    - 将多个样本堆叠成批次
    - 只处理 pixel_values（无标签）
    
    Args:
        batch: 批次数据列表
    
    Returns:
        批处理后的字典
        - pixel_values: (B, 3, H, W) - 3通道：[DoLP, sin(2*AoLP), cos(2*AoLP)]
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    
    return {
        "pixel_values": pixel_values,  # (B, 3, H, W) - [DoLP, sin(2*AoLP), cos(2*AoLP)]
    }


"""
PolarGlareDataset: 自定义数据集实现
用于加载RGB图像、偏振图像和对应的视觉问答对
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from transformers import CLIPImageProcessor
import torchvision.transforms as transforms

# 导入共享的处理函数
from dataset_common import process_polar_images


class PolarGlareDataset(Dataset):
    """
    自定义数据集类，用于加载RGB图像、偏振图像和QA对
    
    Args:
        rgb_root: RGB图像的根目录路径
        polar_root: 偏振图像的根目录路径（包含I_0, I_45, I_90, I_135四个角度图像）
        json_file: 包含QA对的JSON文件路径
        is_train: 是否为训练集（决定是否应用数据增强）
        image_size: RGB输入图像尺寸（默认224，CLIP需要）
    
    注意：
        - 偏振图像会从4个角度（I_0, I_45, I_90, I_135）计算Stokes参数
        - [Stage 3 更新] 输出3通道物理参数：DoLP（偏振度）、sin(2*AoLP)、cos(2*AoLP)，去掉 Intensity
        - [Stage 3 更新] 与 Stage 2 保持一致，使用 VAE 编码器，输入为 3 通道，尺寸为 512x512
    """
    
    def __init__(
        self,
        rgb_root: str,
        polar_root: str,
        json_file: str,
        is_train: bool = True,
        image_size: int = 224,  # ViT 模型期望的输入尺寸
        clip_model_name: str = "openai/clip-vit-large-patch14",  # CLIP 模型路径（支持本地路径）
        data_root: Optional[str] = None,  # [Stage 3 新增] 数据根目录（用于解析 crop 路径，默认使用 rgb_root 的父目录）
    ):
        # 转换为绝对路径（避免在DataLoader worker进程中路径问题）
        # 如果路径是相对路径且不存在，尝试在上级目录查找（适用于 data 和 train 并列的情况）
        rgb_path = Path(rgb_root)
        if not rgb_path.is_absolute():
            # 先尝试当前工作目录下的路径
            rgb_path_resolved = rgb_path.resolve()
            if not rgb_path_resolved.exists():
                # 如果不存在，尝试在上级目录查找（适用于 train 和 data 并列的情况）
                # 例如：当前在 /root/train/，data/rgb 会解析为 /root/train/data/rgb
                # 如果不存在，尝试 /root/data/rgb
                cwd = Path.cwd()
                parent_dir = cwd.parent
                alt_path = parent_dir / rgb_path
                if alt_path.exists():
                    rgb_path = alt_path.resolve()
                    print(f"✓ 在上级目录找到 RGB 路径: {rgb_path}")
                else:
                    rgb_path = rgb_path_resolved
            else:
                rgb_path = rgb_path_resolved
        
        polar_path = Path(polar_root)
        if not polar_path.is_absolute():
            # 先尝试当前工作目录下的路径
            polar_path_resolved = polar_path.resolve()
            if not polar_path_resolved.exists():
                # 如果不存在，尝试在上级目录查找
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
        
        self.rgb_root = rgb_path
        self.polar_root = polar_path
        self.is_train = is_train
        self.image_size = image_size
        
        # [Stage 3 新增] 设置 data_root（用于解析 crop 路径）
        if data_root is None:
            # 默认使用 rgb_root 的父目录
            self.data_root = rgb_path.parent
        else:
            data_root_path = Path(data_root)
            if not data_root_path.is_absolute():
                data_root_path = data_root_path.resolve()
            self.data_root = data_root_path
        print(f"✓ Data root: {self.data_root}")
        
        # 加载JSON数据
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 初始化CLIP图像处理器（用于RGB图像）
        # CLIP ViT-L/14 期望的输入尺寸
        # 支持使用本地模型路径，避免网络下载
        self.clip_processor = CLIPImageProcessor.from_pretrained(
            clip_model_name
        )
        
        # 数据增强策略（Stage 3 关键修复：禁止改变几何形状的增强）
        # ⚠️ 致命问题修复：Stage 3 数据包含归一化坐标 (bbox_norm)，如果使用 RandomResizedCrop，
        # 图像被裁剪后坐标就失效了（坐标指向的位置与图像内容不匹配），会彻底摧毁模型的定位能力。
        # 
        # 解决方案：仅使用颜色增强，保持几何形状不变（Resize），确保坐标与图像内容一致。
        if self.is_train:
            # RGB图像增强：仅使用颜色抖动，保持几何形状不变
            # 注意：绝对禁止使用 RandomResizedCrop、RandomCrop、Rotation 等改变几何形状的增强
            # 因为 QA 对中包含归一化坐标 (bbox_norm)，必须与图像内容精确对应
            # ⚠️ 关键修改：RGB 图像从 512x512（crop 数据）resize 到 224x224（CLIP 需要）
            self.rgb_transform = transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                # [关键修复] 使用 Resize 而非 RandomResizedCrop，保持坐标与图像内容一致
                # Resize 可能会轻微改变宽高比，但保留了所有内容和坐标的相对位置
                transforms.Resize((image_size, image_size)),  # 从 512x512 resize 到 224x224（CLIP 需要）
            ])
            
            # ⚠️ 关键修改：Polar 图像需要保持 512x512（VAE 编码器要求）
            # VAE 在 512x512 上训练，如果输入是 224x224，会导致特征模糊
            # 不进行颜色抖动（偏振数据代表物理量），不进行裁剪（保持坐标一致）
            self.polar_transform = transforms.Compose([
                transforms.Resize((512, 512)),  # VAE 需要的尺寸：512x512
            ])
        else:
            # 验证/测试集：仅进行 Resize，保持与训练集一致
            # 注意：也使用 Resize 而非 CenterCrop，保持坐标与图像内容一致
            # RGB 从 512x512 resize 到 224x224（CLIP 需要）
            self.rgb_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),  # 从 512x512 resize 到 224x224
            ])
            # ⚠️ 关键修改：Polar 图像需要保持 512x512（VAE 编码器要求）
            self.polar_transform = transforms.Compose([
                transforms.Resize((512, 512)),  # VAE 需要的尺寸：512x512
            ])
        
        # 将图像转换为张量的转换（RGB和偏振都需要）
        self.to_tensor = transforms.ToTensor()
        
        # [Stage 3 更新] 偏振图像归一化（3通道：DoLP, sin(2*AoLP), cos(2*AoLP)）
        # ⚠️ 注意：与 Stage 2 保持一致，不使用 ImageNet 归一化
        # Stage 2 训练时只使用 ToTensor()，值范围保持在 [0, 1]
        # 这里保留定义但不使用（为了兼容性）
        # 注意：现在只有 3 通道，不再需要 4 通道的归一化
    
    def _normalize_path(self, path: str, scene_id: Optional[str] = None, path_type: str = "rgb") -> Path:
        """
        规范化路径：处理相对路径或绝对路径
        
        适配新的JSON格式：
        - 相对路径：rgb/04/0002_rgb.png -> {rgb_root}/04/0002_rgb.png
        - 相对路径：GT/04/0000_rgb.png -> {gt_root}/04/0000_rgb.png（如果指定gt_root）
        - 绝对路径：E:\\... -> 转换为相对路径
        
        Args:
            path: 原始路径字符串
            scene_id: 场景ID（如果JSON中提供了）
            path_type: 路径类型，'rgb' 或 'gt'，用于选择根目录
            
        Returns:
            规范化后的Path对象
        """
        path_obj = Path(path)
        
        # 如果是绝对路径（Windows格式：E:\... 或 Linux格式：/...）
        if os.path.isabs(path):
            # 处理绝对路径（兼容旧格式）
            if scene_id:
                filename = path_obj.name
                root = self.rgb_root if path_type == "rgb" else getattr(self, 'gt_root', self.rgb_root)
                scene_path = root / scene_id / filename
                if scene_path.exists():
                    return scene_path
                return root / filename
            else:
                # 尝试从路径中提取scene_id
                parts = path_obj.parts
                for i, part in enumerate(parts):
                    if part.isdigit() and i < len(parts) - 1:
                        scene_id_candidate = part
                        filename = path_obj.name
                        root = self.rgb_root if path_type == "rgb" else getattr(self, 'gt_root', self.rgb_root)
                        scene_path = root / scene_id_candidate / filename
                        if scene_path.exists():
                            return scene_path
                filename = path_obj.name
                root = self.rgb_root if path_type == "rgb" else getattr(self, 'gt_root', self.rgb_root)
                return root / filename
        else:
            # 处理相对路径（新格式）
            # 路径格式：rgb/04/0002_rgb.png 或 GT/04/0000_rgb.png
            parts = path_obj.parts
            
            # 如果路径以 'rgb/' 或 'GT/' 开头，去掉前缀
            if len(parts) > 0:
                if parts[0].lower() in ['rgb', 'gt']:
                    # 去掉第一个部分（rgb或GT），剩余部分直接拼接
                    remaining_parts = parts[1:]
                    root = self.rgb_root if path_type == "rgb" else getattr(self, 'gt_root', self.rgb_root)
                    return root / Path(*remaining_parts)
                else:
                    # 没有前缀，直接使用
                    root = self.rgb_root if path_type == "rgb" else getattr(self, 'gt_root', self.rgb_root)
                    return root / path
            else:
                # 空路径，直接返回
                root = self.rgb_root if path_type == "rgb" else getattr(self, 'gt_root', self.rgb_root)
                return root / path
    
    def _get_polar_paths(self, rgb_path: Path, scene_id: Optional[str] = None, base_name: Optional[str] = None) -> Dict[str, Path]:
        """
        根据RGB图像路径和scene_id推导偏振图像的四个通道路径
        
        适配新的目录结构：
        - {polar_root}/{scene_id}/{base_name}_000.png, {base_name}_045.png, {base_name}_090.png, {base_name}_135.png
        
        注意：角度使用三位数格式（000, 045, 090, 135）
        
        Args:
            rgb_path: RGB图像的Path对象
            scene_id: 场景ID（从JSON中提取）
            base_name: 基础文件名（不含扩展名，如 "0002"）
            
        Returns:
            包含四个偏振通道路径的字典
        """
        # 获取基础文件名（不含扩展名和_rgb后缀）
        if base_name is None:
            base_name = rgb_path.stem
            # 移除可能的_rgb后缀
            if base_name.endswith('_rgb'):
                base_name = base_name[:-4]
        
        # 尝试多种可能的命名模式
        polar_paths = {}
        
        # 模式1（推荐）: {polar_root}/{scene_id}/{base_name}_000.png, {base_name}_045.png, ...
        # 使用三位数角度格式：000, 045, 090, 135
        if scene_id:
            polar_scene_dir = self.polar_root / scene_id
            if polar_scene_dir.exists():
                polar_paths = {
                    'I_0': polar_scene_dir / f"{base_name}_000.png",
                    'I_45': polar_scene_dir / f"{base_name}_045.png",
                    'I_90': polar_scene_dir / f"{base_name}_090.png",
                    'I_135': polar_scene_dir / f"{base_name}_135.png",
                }
                # 如果模式1存在，直接返回
                if all(p.exists() for p in polar_paths.values()):
                    return polar_paths
                
                # 如果模式1不存在，尝试模式2：使用一位数/两位数角度格式
                polar_paths = {
                    'I_0': polar_scene_dir / f"{base_name}_0.png",
                    'I_45': polar_scene_dir / f"{base_name}_45.png",
                    'I_90': polar_scene_dir / f"{base_name}_90.png",
                    'I_135': polar_scene_dir / f"{base_name}_135.png",
                }
                if all(p.exists() for p in polar_paths.values()):
                    return polar_paths
                
                # 如果模式2不存在，尝试模式3：{polar_root}/{scene_id}/0.png, 45.png, ...
                polar_paths = {
                    'I_0': polar_scene_dir / "0.png",
                    'I_45': polar_scene_dir / "45.png",
                    'I_90': polar_scene_dir / "90.png",
                    'I_135': polar_scene_dir / "135.png",
                }
                if all(p.exists() for p in polar_paths.values()):
                    return polar_paths
        
        # 模式4: {polar_root}/{base_name}_000.png, {base_name}_045.png, ...（直接在polar_root下）
        polar_paths = {
            'I_0': self.polar_root / f"{base_name}_000.png",
            'I_45': self.polar_root / f"{base_name}_045.png",
            'I_90': self.polar_root / f"{base_name}_090.png",
            'I_135': self.polar_root / f"{base_name}_135.png",
        }
        if all(p.exists() for p in polar_paths.values()):
            return polar_paths
        
        # 模式5: 从RGB路径的父目录推断scene_id
        if rgb_path.parent != self.rgb_root:
            scene_id_from_path = rgb_path.parent.name
            polar_scene_dir = self.polar_root / scene_id_from_path
            if polar_scene_dir.exists():
                polar_paths = {
                    'I_0': polar_scene_dir / f"{base_name}_000.png",
                    'I_45': polar_scene_dir / f"{base_name}_045.png",
                    'I_90': polar_scene_dir / f"{base_name}_090.png",
                    'I_135': polar_scene_dir / f"{base_name}_135.png",
                }
                if all(p.exists() for p in polar_paths.values()):
                    return polar_paths
        
        # 如果所有模式都不存在，返回最后尝试的路径（会在_load_polar_images中报错）
        return polar_paths
    
    def _load_rgb_image(self, rgb_path: Path) -> Image.Image:
        """
        加载RGB图像并应用数据增强
        
        Args:
            rgb_path: RGB图像路径
            
        Returns:
            PIL Image对象
        """
        # 检查文件是否存在
        if not rgb_path.exists():
            # [新增] 详细报错，方便调试
            print(f"❌ 路径错误: 无法找到文件 {rgb_path}")
            print(f"   self.rgb_root: {self.rgb_root}")
            print(f"   self.data_root: {self.data_root}")
            print(f"   路径是否为绝对路径: {rgb_path.is_absolute()}")
            print(f"   路径的父目录是否存在: {rgb_path.parent.exists() if rgb_path.parent else 'N/A'}")
            if rgb_path.parent.exists():
                print(f"   父目录内容: {list(rgb_path.parent.iterdir())[:10]}")  # 只显示前10个文件
            raise FileNotFoundError(f"RGB图像不存在: {rgb_path}")
        
        # 加载图像
        image = Image.open(rgb_path).convert('RGB')
        
        # 应用数据增强（训练集和验证集使用不同的transform，已在__init__中设置）
        # ⚠️ 注意：这里的transform只包含颜色增强和Resize，不会改变几何形状
        # CLIP处理器会在后续步骤中进行最终的resize和归一化
        image = self.rgb_transform(image)
        
        return image
    
    def _load_polar_images(self, polar_paths: Dict[str, Path]) -> torch.Tensor:
        """
        加载四个偏振通道图像并转换为物理参数（3通道：DoLP, sin(2*AoLP), cos(2*AoLP)）
        
        [Stage 3 更新] 使用与 Stage 2 一致的 3 通道格式，确保模型兼容性
        - 去掉 Intensity 通道，只保留 [DoLP, sin(2*AoLP), cos(2*AoLP)]
        - 输出尺寸为 512x512（VAE 编码器要求）
        
        物理计算：
        1. 从4个角度的光强图（I_0, I_45, I_90, I_135）计算Stokes参数
        2. 计算Intensity (总光强)、DoLP (偏振度)、sin(2*AoLP)、cos(2*AoLP)
        3. 只取后3个通道，去掉 Intensity
        4. 使用共享的 process_polar_images 函数（与 Stage 2 一致）
        
        Args:
            polar_paths: 包含四个偏振通道路径的字典
        
        Returns:
            形状为(3, 512, 512)的归一化张量，通道顺序：[DoLP, sin(2*AoLP), cos(2*AoLP)]
        """
        # [新增] 在调用 process_polar_images 之前检查路径，提供更详细的错误信息
        missing_paths = []
        for channel_name, polar_path in polar_paths.items():
            if not polar_path.exists():
                missing_paths.append((channel_name, polar_path))
        
        if missing_paths:
            print(f"❌ 偏振图像路径错误: 无法找到以下文件")
            print(f"   self.polar_root: {self.polar_root}")
            print(f"   self.data_root: {self.data_root}")
            for channel_name, polar_path in missing_paths:
                print(f"   - {channel_name}: {polar_path}")
                print(f"     路径是否为绝对路径: {polar_path.is_absolute()}")
                if polar_path.parent.exists():
                    print(f"     父目录内容: {list(polar_path.parent.iterdir())[:10]}")  # 只显示前10个文件
        
        # [Stage 3 更新] 使用共享的 process_polar_images 函数（与 Stage 2 一致）
        # 返回 (H, W, 4)，值范围[0, 1]，通道：[Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]
        # 注意：process_polar_images 内部也会检查文件是否存在，但这里的提前检查可以提供更详细的调试信息
        physics_img = process_polar_images(polar_paths=polar_paths)  # (H, W, 4)，值范围[0, 1]
        
        # ⚠️ 关键修改：只取后3个通道（去掉 Intensity）
        # 与训练阶段保持一致：dataset_stage2.py 中的 _load_polar_images 只取 [DoLP, sin(2*AoLP), cos(2*AoLP)]
        physics_img_3ch = physics_img[:, :, 1:4]  # (H, W, 3)，值范围[0, 1]
        
        # 转换为PIL Image以便应用transform（需要uint8格式）
        # 使用RGB模式来处理3通道
        physics_img_uint8 = (physics_img_3ch * 255).astype(np.uint8)
        physics_pil = Image.fromarray(physics_img_uint8, mode='RGB')
        
        # 应用数据增强（如果是训练集）
        # ⚠️ 关键修改：Polar 图像需要保持 512x512（VAE 编码器要求）
        # 与训练阶段保持一致：dataset_stage2.py 中的 polar_transform 会 resize 到 512x512
        physics_pil = self.polar_transform(physics_pil)
        
        # 转换为张量（保持 [0, 1] 范围，与 Stage 2 一致）
        # ⚠️ 关键修复：Stage 2 训练时不使用 ImageNet 归一化，只使用 ToTensor()
        # Stage 3 也必须保持一致，否则编码器无法正确提取特征
        to_tensor = transforms.ToTensor()
        physics_tensor = to_tensor(physics_pil)  # (3, 512, 512)，值范围[0, 1]
        
        # ❌ 已移除 ImageNet 归一化（与 Stage 2 保持一致）
        # normalize = transforms.Normalize(
        #     mean=[0.485, 0.485, 0.485], 
        #     std=[0.229, 0.229, 0.229]
        # )
        # physics_tensor = normalize(physics_tensor)
        
        return physics_tensor  # 直接返回 [0, 1] 范围的张量，形状为 (3, 512, 512)
    
    def _format_conversation(self, conversations: List[Dict]) -> Tuple[str, str]:
        """
        从JSON中的conversations列表提取问题和答案
        
        Args:
            conversations: 对话列表，格式为 [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
            
        Returns:
            (question, answer) 元组
        """
        question = ""
        answer = ""
        
        for conv in conversations:
            if conv.get("from") == "human":
                question = conv.get("value", "")
            elif conv.get("from") == "gpt":
                answer = conv.get("value", "")
        
        return question, answer
    
    def _format_prompt(self, question: str, answer: str) -> str:
        """
        格式化提示为LLaMA-3标准格式
        
        LLaMA-3格式（训练时）：
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        
        <image>
        {Question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        
        {Answer}<|eot_id|>
        
        ⚠️ 重要提示：
        - 训练时：包含完整的 question 和 answer
        - 推理时：prompt 应该截止到 ...assistant<|end_header_id|>\n\n，不包含 answer
          推理格式：<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<image>\n{Question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
        
        Args:
            question: 用户问题
            answer: 助手回答（训练时使用，推理时不需要）
            
        Returns:
            格式化后的提示字符串（包含 question 和 answer，用于训练）
        """
        prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "<image>\n"
            f"{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{answer}<|eot_id|>"
        )
        return prompt
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个数据样本
        
        适配新的JSON格式：
        - 使用 input_path 字段获取RGB图像（而不是 image 字段）
        - image 字段指向GT图像，input_path 指向输入RGB图像
        - 根据 scene_id 和 input_path 中的文件名推导偏振图像路径
        - 偏振图像使用三位数角度格式：000, 045, 090, 135
        
        Args:
            idx: 样本索引
            
        Returns:
            包含以下键的字典：
            - pixel_values_rgb: RGB图像的像素值（经过CLIP处理器，224x224）
            - pixel_values_polar: 偏振图像的像素值（3通道张量：[DoLP, sin(2*AoLP), cos(2*AoLP)]，512x512）
            - question: 问题文本
            - answer: 答案文本
            - prompt_text: 格式化后的完整提示文本
        """
        item = self.data[idx]
        
        # 1. 提取scene_id和RGB图像路径
        # [Stage 3 适配] 支持多种格式：
        # - Stage 3 合并格式：input_path 字段指向 rgb/，如 "rgb/04/0002_rgb.png"（优先使用）
        # - Stage 2 格式：image 字段指向 rgb_crop/，如 "rgb_crop/10/0000_rgb.png"（向后兼容）
        # 注意：合并后的数据中，image 字段指向 GT 图像，input_path 指向 RGB 图像
        scene_id = item.get("scene_id", None)  # 从JSON中提取scene_id
        
        # 优先使用 input_path（RGB 路径），如果没有则尝试 image（可能是 rgb_crop 格式）
        rgb_path_str = item.get("input_path") or item.get("image", "")
        
        if not rgb_path_str:
            raise ValueError(f"样本 {idx} (scene_id: {scene_id}) 缺少 RGB 图像路径（需要 input_path 或 image 字段）")
        
        # 2. 处理RGB图像路径（优先支持 Stage 2 的 crop 格式，然后是 Stage 3 格式）
        if rgb_path_str.startswith("rgb_crop/"):
            # Stage 2 格式：从 data_root 解析 crop 路径
            # 例如：rgb_crop/10/0000_rgb.png -> data_root/rgb_crop/10/0000_rgb.png
            relative_path = rgb_path_str.replace("rgb_crop/", "")
            rgb_path = self.data_root / "rgb_crop" / relative_path
        elif rgb_path_str.startswith("rgb/"):
            # Stage 3 格式：rgb/04/0002_rgb.png（非裁剪图像）
            # 使用 _normalize_path 处理相对路径
            rgb_path = self._normalize_path(rgb_path_str, scene_id=scene_id, path_type="rgb")
        else:
            # 兼容其他格式（绝对路径等）
            rgb_path = self._normalize_path(rgb_path_str, scene_id=scene_id, path_type="rgb")
        
        rgb_image = self._load_rgb_image(rgb_path)
        
        # 使用CLIP处理器处理RGB图像
        # CLIP处理器会进行resize、归一化等操作
        pixel_values_rgb = self.clip_processor(
            rgb_image,
            return_tensors="pt"
        )["pixel_values"].squeeze(0)  # 移除batch维度: (C, H, W)
        
        # 3. 处理偏振图像
        # [Stage 3 适配] 支持多种格式：
        # - Stage 2 格式：polar_crop_paths 字段（向后兼容）
        # - Stage 3 格式：根据 input_path 和 scene_id 推导原始偏振图像路径（主要格式）
        polar_crop_paths = item.get("polar_crop_paths", {})
        
        if polar_crop_paths and len(polar_crop_paths) > 0:
            # Stage 2 格式：从 polar_crop_paths 读取裁剪后的偏振图像路径
            polar_paths = {}
            for angle_name in ["I_0", "I_45", "I_90", "I_135"]:
                polar_path_str = polar_crop_paths.get(angle_name)
                if polar_path_str:
                    # 处理相对路径：polar_crop/10/0000_000.png -> data_root/polar_crop/10/0000_000.png
                    if polar_path_str.startswith("polar_crop/"):
                        relative_path = polar_path_str.replace("polar_crop/", "")
                        polar_paths[angle_name] = self.data_root / "polar_crop" / relative_path
                    elif polar_path_str.startswith("polar/"):
                        # 兼容旧格式
                        relative_path = polar_path_str.replace("polar/", "")
                        polar_paths[angle_name] = self.polar_root / relative_path
                    else:
                        # 绝对路径或其他格式
                        polar_paths[angle_name] = Path(polar_path_str)
                else:
                    # 如果某个角度缺失，回退到推导方法
                    break
            
            # 检查是否所有路径都存在
            if len(polar_paths) == 4 and all(p.exists() for p in polar_paths.values()):
                # 使用 Stage 2 格式的路径（裁剪后的偏振图像）
                pass
            else:
                # 回退到推导方法：根据 scene_id 和 base_name 从原始偏振图像目录推导
                base_name = rgb_path.stem
                if base_name.endswith('_rgb'):
                    base_name = base_name[:-4]
                polar_paths = self._get_polar_paths(rgb_path, scene_id=scene_id, base_name=base_name)
        else:
            # Stage 3 格式或旧格式：根据 scene_id 和 input_path 中的文件名推导原始偏振图像路径
            # 注意：Stage 3 使用的是非裁剪图像，所以从原始 polar/ 目录读取
            base_name = rgb_path.stem
            if base_name.endswith('_rgb'):
                base_name = base_name[:-4]
            polar_paths = self._get_polar_paths(rgb_path, scene_id=scene_id, base_name=base_name)
        
        pixel_values_polar = self._load_polar_images(polar_paths)  # (3, 512, 512) - [DoLP, sin(2*AoLP), cos(2*AoLP)]
        
        # 4. 处理对话
        conversations = item.get("conversations", [])
        question, answer = self._format_conversation(conversations)
        
        # 格式化提示（注意：这里我们返回原始文本，tokenization在collate_fn中完成）
        # 为了简化，我们返回问题和答案，让tokenizer在collate_fn中处理
        prompt_text = self._format_prompt(question, answer)
        
        return {
            "pixel_values_rgb": pixel_values_rgb,
            "pixel_values_polar": pixel_values_polar,
            "question": question,
            "answer": answer,
            "prompt_text": prompt_text,
        }


def collate_fn(batch: List[Dict], tokenizer) -> Dict[str, torch.Tensor]:
    """
    自定义collate函数，用于批处理数据（Token级对齐版本）
    
    关键改进：使用更稳健的 Masking 策略，修复 tokenizer 长度计算的潜在 bug
    
    Args:
        batch: 批次数据列表
        tokenizer: 用于分词的分词器
        
        Returns:
            批处理后的字典，包含：
        - pixel_values_rgb: (B, 3, 224, 224) - RGB图像，CLIP输入尺寸
        - pixel_values_polar: (B, 3, 512, 512) - 3通道：[DoLP, sin(2*AoLP), cos(2*AoLP)]，VAE输入尺寸
        - input_ids: (B, L)
        - attention_mask: (B, L)
        - labels: (B, L) - 用于计算损失，问题部分为-100（忽略）
    """
    # 分离RGB和偏振图像
    pixel_values_rgb = torch.stack([item["pixel_values_rgb"] for item in batch])
    pixel_values_polar = torch.stack([item["pixel_values_polar"] for item in batch])
    
    # 提取文本
    prompt_texts = [item["prompt_text"] for item in batch]
    
    # 1. Tokenize 完整文本
    # 必须加上 <|begin_of_text|>，因为这是 LLaMA-3 的入口
    # 如果 prompt_text 里已经有了，add_special_tokens 设为 False
    encoded = tokenizer(
        prompt_texts,
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt",
        add_special_tokens=False,  # prompt_text 中已经包含了所有特殊 token
    )
    
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = input_ids.clone()
    
    # 2. 改进的 Masking 逻辑（更精准）
    # 重构 prompt 的前半部分（直到 assistant 回答开始之前）
    # 必须与 _format_prompt 中的格式完全一致
    for i in range(len(batch)):
        # 获取纯问题文本
        question_part = batch[i]["question"]
        
        # 重构 prompt 的前半部分（直到 assistant 回答开始之前）
        # 必须与 _format_prompt 中的格式完全一致
        prefix = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "<image>\n"
            f"{question_part}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        # Tokenize 前缀（不加 special tokens，因为已经在 prefix 里了）
        prefix_encoded = tokenizer(
            prefix,
            add_special_tokens=False,
            truncation=True,
            max_length=2048
        )
        prefix_tokens = prefix_encoded.input_ids
        
        # 处理返回格式：tokenizer 可能返回 list 或 tensor
        # tokenizer 返回的 input_ids 通常是 list，需要处理嵌套情况
        if isinstance(prefix_tokens, list):
            # 如果返回的是列表的列表（batch），取第一个
            if len(prefix_tokens) > 0 and isinstance(prefix_tokens[0], list):
                prefix_len = len(prefix_tokens[0])
            else:
                # 如果返回的是单个列表
                prefix_len = len(prefix_tokens)
        elif hasattr(prefix_tokens, 'shape'):
            # 如果是 tensor（通常不会出现，但为了安全）
            if len(prefix_tokens.shape) > 1:
                prefix_len = prefix_tokens.shape[1]
            else:
                prefix_len = prefix_tokens.shape[0]
        else:
            # 其他情况（不太可能）
            prefix_len = len(prefix_tokens) if hasattr(prefix_tokens, '__len__') else 0
        
        # 设置 Label Mask
        # 将"回答之前"的所有 token 设为 -100
        # 还要确保不越界
        valid_len = min(prefix_len, labels.shape[1])
        labels[i, :valid_len] = -100
        
        # 处理 Padding（attention_mask 为 0 的位置）
        labels[i][attention_mask[i] == 0] = -100
    
    return {
        "pixel_values_rgb": pixel_values_rgb,
        "pixel_values_polar": pixel_values_polar,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


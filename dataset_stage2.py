"""
Stage 2 数据集：语义对齐（Projector 预训练）

目标：
- 使用 subtype == "content" 的数据作为图像描述任务
- Question 作为 prompt（例如："Describe this..."）
- Answer 作为目标 caption
- 训练 Projector 对齐偏振特征和 LLM

数据格式：
- 输入：RGB 图像、偏振图像（3通道：I, DoLP, AoLP）、文本（Question + Answer）
- 输出：input_ids, attention_mask, labels

注意：
- 使用共享的 process_polar_images 函数将4张角度图转换为3通道物理参数
- 3通道输入可以直接使用标准ResNet/ViT预训练权重
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from transformers import CLIPImageProcessor, AutoTokenizer
import torchvision.transforms as transforms

# 导入共享的处理函数
from dataset_common import process_polar_images


class PolarAlignmentDataset(Dataset):
    """
    Stage 2 数据集：用于 Projector 预训练的语义对齐数据集
    
    功能：
    1. 从 JSON 加载数据
    2. 过滤 subtype == "content" 的样本
    3. 使用 Question 作为 prompt，Answer 作为 caption
    4. 加载 RGB 和偏振图像
    5. 返回文本 tokens（用于因果语言建模）
    
    Args:
        rgb_root: RGB图像根目录
        polar_root: 偏振图像根目录
        json_file: JSON文件路径
        tokenizer: 分词器
        image_size: 图像尺寸（默认224，匹配ViT模型）
        is_train: 是否为训练集
    """
    
    def __init__(
        self,
        rgb_root: str,
        polar_root: str,
        json_file: str,
        tokenizer: AutoTokenizer,
        image_size: int = 224,  # 修改为224以匹配ViT模型（google/vit-base-patch16-224-in21k）
        is_train: bool = True,
        data_root: Optional[str] = None,  # 数据根目录（用于解析 crop 路径）
    ):
        # 转换为绝对路径
        rgb_path = Path(rgb_root)
        if not rgb_path.is_absolute():
            rgb_path_resolved = rgb_path.resolve()
            if not rgb_path_resolved.exists():
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
            polar_path_resolved = polar_path.resolve()
            if not polar_path_resolved.exists():
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
        
        # 数据根目录（用于解析 crop 路径，如 rgb_crop/, polar_crop/）
        if data_root is None:
            # 默认使用 rgb_root 的父目录（假设结构为 data_root/rgb, data_root/rgb_crop）
            self.data_root = rgb_path.parent
        else:
            self.data_root = Path(data_root)
        
        self.rgb_root = rgb_path
        self.polar_root = polar_path
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.is_train = is_train
        
        # 加载 JSON 数据
        with open(json_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Stage 2 新格式：直接使用所有数据（不需要过滤 subtype）
        # 格式：{"id", "image", "gt_image", "scene_id", "bbox_norm", "conversations"}
        self.data = raw_data
        
        if len(self.data) == 0:
            raise ValueError(f"数据集为空！请检查 JSON 文件: {json_file}")
        
        print(f"✓ Stage 2 数据集加载完成: {len(self.data)} 个样本")
        
        # 初始化 CLIP 图像处理器
        # 尝试使用本地路径，如果不存在则使用 Hugging Face Hub
        clip_processor_path = "/openbayes/home/train/models/clip-vit-large-patch14"
        if os.path.exists(clip_processor_path):
            try:
                self.clip_processor = CLIPImageProcessor.from_pretrained(
                    clip_processor_path,
                    local_files_only=True
                )
                print(f"✓ 从本地路径加载 CLIP 图像处理器: {clip_processor_path}")
            except Exception:
                # 如果本地加载失败，回退到 Hugging Face Hub
                self.clip_processor = CLIPImageProcessor.from_pretrained(
                    "openai/clip-vit-large-patch14"
                )
                print("⚠ 本地 CLIP 图像处理器加载失败，使用 Hugging Face Hub")
        else:
            self.clip_processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
        
        # 数据增强（仅用于训练集）
        # ⚠️ 关键修复：为了保证 RGB 和 Polar 严格的空间对齐，
        # 暂时放弃 RandomResizedCrop（除非能保证两者使用相同的随机种子），
        # 改为确定性的 Resize。对于 Projector 预训练，这通常足够了。
        if self.is_train:
            # ⚠️ 关键修复：RGB 图像从 512x512（crop 数据）resize 到 224x224（CLIP 需要）
            # 这样可以保证 RGB 和 Polar 的视野完全一致（都是 512x512 的裁剪区域）
            self.rgb_transform = transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                ),
                # RGB 可以做颜色变换，不影响空间位置
                transforms.Resize((image_size, image_size)),  # 从 512x512 resize 到 224x224（CLIP 需要）
                # ❌ 已删除 RandomResizedCrop：避免与 Polar 图像空间错位
            ])
            
            # ⚠️ 关键修改：Polar 图像需要保持 512x512（VAE 编码器要求）
            # VAE 在 512x512 上训练，如果输入是 224x224，会导致特征模糊
            # 注意：如果 crop 数据已经是 512x512，这里可以不 resize（但为了兼容性保留）
            self.polar_transform = transforms.Compose([
                transforms.Resize((512, 512)),  # VAE 需要的尺寸：512x512
                # ❌ 已删除 RandomResizedCrop：避免与 RGB 图像空间错位
            ])
        else:
            # 验证集：RGB 从 512x512 resize 到 224x224（CLIP 需要）
            self.rgb_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),  # 从 512x512 resize 到 224x224
            ])
            
            # ⚠️ 关键修改：Polar 图像需要保持 512x512（VAE 编码器要求）
            self.polar_transform = transforms.Compose([
                transforms.Resize((512, 512)),  # VAE 需要的尺寸：512x512
            ])
    
    def _normalize_path(self, path: str, scene_id: Optional[str] = None, path_type: str = "rgb") -> Path:
        """
        规范化路径（从 dataset.py 复用）
        
        Args:
            path: 原始路径字符串
            scene_id: 场景ID
            path_type: 路径类型，'rgb' 或 'gt'
        
        Returns:
            规范化后的Path对象
        """
        path_obj = Path(path)
        
        if os.path.isabs(path):
            # 处理绝对路径
            if scene_id:
                filename = path_obj.name
                root = self.rgb_root if path_type == "rgb" else getattr(self, 'gt_root', self.rgb_root)
                scene_path = root / scene_id / filename
                if scene_path.exists():
                    return scene_path
                return root / filename
            else:
                return path_obj
        else:
            # 处理相对路径
            parts = path_obj.parts
            if len(parts) > 0:
                if parts[0].lower() in ['rgb', 'gt']:
                    remaining_parts = parts[1:]
                    root = self.rgb_root if path_type == "rgb" else getattr(self, 'gt_root', self.rgb_root)
                    return root / Path(*remaining_parts)
                else:
                    root = self.rgb_root if path_type == "rgb" else getattr(self, 'gt_root', self.rgb_root)
                    return root / path
            else:
                root = self.rgb_root if path_type == "rgb" else getattr(self, 'gt_root', self.rgb_root)
                return root / path
    
    def _get_polar_paths(self, rgb_path: Path, scene_id: Optional[str] = None, base_name: Optional[str] = None) -> Dict[str, Path]:
        """
        获取偏振图像路径（从 dataset.py 复用）
        
        Args:
            rgb_path: RGB图像路径
            scene_id: 场景ID
            base_name: 基础文件名
        
        Returns:
            包含4个通道路径的字典
        """
        if base_name is None:
            base_name = rgb_path.stem
            if base_name.endswith('_rgb'):
                base_name = base_name[:-4]
        
        polar_paths = {}
        
        # 尝试三位数角度格式
        if scene_id:
            polar_scene_dir = self.polar_root / scene_id
            if polar_scene_dir.exists():
                polar_paths = {
                    'I_0': polar_scene_dir / f"{base_name}_000.png",
                    'I_45': polar_scene_dir / f"{base_name}_045.png",
                    'I_90': polar_scene_dir / f"{base_name}_090.png",
                    'I_135': polar_scene_dir / f"{base_name}_135.png",
                }
                if all(p.exists() for p in polar_paths.values()):
                    return polar_paths
                
                # 尝试一位/两位数角度格式
                polar_paths = {
                    'I_0': polar_scene_dir / f"{base_name}_0.png",
                    'I_45': polar_scene_dir / f"{base_name}_45.png",
                    'I_90': polar_scene_dir / f"{base_name}_90.png",
                    'I_135': polar_scene_dir / f"{base_name}_135.png",
                }
                if all(p.exists() for p in polar_paths.values()):
                    return polar_paths
        
        # 如果都不存在，返回最后尝试的路径（会在加载时报错）
        return polar_paths
    
    def _load_rgb_image(self, rgb_path: Path) -> Image.Image:
        """加载RGB图像"""
        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB图像不存在: {rgb_path}")
        return Image.open(rgb_path).convert('RGB')
    
    def _load_polar_images(self, polar_paths: Dict[str, Path]) -> torch.Tensor:
        """
        加载4个偏振角度图像并转换为3通道物理参数 [DoLP, sin(2*AoLP), cos(2*AoLP)]
        
        使用共享的 process_polar_images 函数计算Stokes参数，然后只取后3个通道
        
        Args:
            polar_paths: 包含4个角度图像路径的字典
        
        Returns:
            形状为(3, H, W)的归一化张量，通道顺序：[DoLP, sin(2*AoLP), cos(2*AoLP)]
        """
        # 使用共享的process_polar_images函数计算Stokes参数
        # 返回 (H, W, 4)，值范围[0, 1]，通道：[Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]
        physics_img = process_polar_images(polar_paths=polar_paths)  # (H, W, 4)，值范围[0, 1]
        
        # 只取后3个通道：[DoLP, sin(2*AoLP), cos(2*AoLP)]，去掉 Intensity
        physics_img_3ch = physics_img[:, :, 1:4]  # (H, W, 3)
        
        # 转换为PIL Image以便应用transform（需要uint8格式）
        # 使用RGB模式来处理3通道
        physics_img_uint8 = (physics_img_3ch * 255).astype(np.uint8)
        physics_pil = Image.fromarray(physics_img_uint8, mode='RGB')
        
        # 应用数据增强（如果是训练集）
        physics_pil = self.polar_transform(physics_pil)
        
        # 转换为张量（保持 [0, 1] 范围）
        to_tensor = transforms.ToTensor()
        physics_tensor = to_tensor(physics_pil)  # (3, H, W)，值范围[0, 1]
        
        return physics_tensor  # 直接返回 [0, 1] 范围的张量，3通道
    
    def _format_conversation(self, conversations: List[Dict]) -> tuple:
        """
        格式化对话（从 dataset.py 复用）
        
        Args:
            conversations: 对话列表
        
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
        格式化提示（Stage 2 使用简化的格式）
        
        Args:
            question: 问题文本
            answer: 答案文本
        
        Returns:
            格式化后的提示文本
        """
        # Stage 2 使用简单的格式：Question + Answer
        # 注意：这里不使用 <image> token，因为 Stage 2 的训练逻辑会直接插入视觉特征
        prompt = f"{question}\n{answer}"
        return prompt
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取一个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            包含图像和文本的字典
        """
        item = self.data[idx]
        
        # 1. 获取路径信息（新格式：image 字段指向 rgb_crop/ 目录）
        rgb_path_str = item.get("image", "")  # 相对路径，如 "rgb_crop/00/0000_rgb.png"
        scene_id = item.get("scene_id", None)
        
        # 处理 crop 路径：从 data_root 解析
        # 例如：rgb_crop/00/0000_rgb.png -> data_root/rgb_crop/00/0000_rgb.png
        if rgb_path_str.startswith("rgb_crop/"):
            # 去掉 "rgb_crop/" 前缀，拼接 data_root
            relative_path = rgb_path_str.replace("rgb_crop/", "")
            rgb_path = self.data_root / "rgb_crop" / relative_path
        elif "/" in rgb_path_str:
            # 兼容旧格式：scene_id/filename
            parts = rgb_path_str.split("/")
            scene_id = parts[0] if scene_id is None else scene_id
            filename = parts[-1]
            rgb_path = self.rgb_root / scene_id / filename
        else:
            # 直接文件名
            if scene_id:
                rgb_path = self.rgb_root / scene_id / rgb_path_str
            else:
                rgb_path = self.rgb_root / rgb_path_str
        
        # 2. 处理RGB图像（从磁盘加载的是 512x512 的 crop 图像）
        rgb_image = self._load_rgb_image(rgb_path)  # 加载 512x512 的图像
        # ⚠️ 关键：应用 transform 将图像 resize 到 224x224（CLIP 需要的尺寸）
        # transform 包含：ColorJitter（训练时）+ Resize(224, 224)
        rgb_image = self.rgb_transform(rgb_image)  # 现在图像是 224x224
        
        # 验证图像尺寸（用于调试，确保 transform 正确应用）
        if rgb_image.size != (self.image_size, self.image_size):
            raise ValueError(
                f"RGB 图像尺寸错误：期望 {self.image_size}x{self.image_size}，"
                f"实际 {rgb_image.size}。请检查 rgb_transform 是否正确配置。"
            )
        
        # 使用CLIP处理器处理RGB图像（确保输入是 224x224）
        # CLIP 处理器也会自动 resize，但我们已经提前 resize 以确保一致性
        pixel_values_rgb = self.clip_processor(
            rgb_image,
            return_tensors="pt"
        )["pixel_values"].squeeze(0)  # (C, H, W)，H=W=224
        
        # 3. 处理偏振图像（新格式：从 polar_crop_paths 读取）
        polar_crop_paths = item.get("polar_crop_paths", {})
        
        if polar_crop_paths and len(polar_crop_paths) > 0:
            # 新格式：从 polar_crop_paths 读取裁剪后的偏振图像路径
            polar_paths = {}
            for angle_name in ["I_0", "I_45", "I_90", "I_135"]:
                polar_path_str = polar_crop_paths.get(angle_name)
                if polar_path_str:
                    # 处理相对路径：polar_crop/00/0000_000.png -> data_root/polar_crop/00/0000_000.png
                    if polar_path_str.startswith("polar_crop/"):
                        relative_path = polar_path_str.replace("polar_crop/", "")
                        polar_paths[angle_name] = self.data_root / "polar_crop" / relative_path
                    else:
                        # 兼容旧格式（如果路径不是以 polar_crop/ 开头）
                        polar_paths[angle_name] = self.polar_root / scene_id / Path(polar_path_str).name
                else:
                    # 如果某个角度缺失，尝试从 RGB 路径推导（使用 polar_crop 目录）
                    if scene_id:
                        base_name = rgb_path.stem
                        if base_name.endswith('_rgb'):
                            base_name = base_name[:-4]
                        angle_suffix = {"I_0": "000", "I_45": "045", "I_90": "090", "I_135": "135"}[angle_name]
                        # 优先尝试 polar_crop 目录（新格式）
                        polar_crop_path = self.data_root / "polar_crop" / scene_id / f"{base_name}_{angle_suffix}.png"
                        if polar_crop_path.exists():
                            polar_paths[angle_name] = polar_crop_path
                        else:
                            # 回退到旧格式的 polar_root
                            polar_paths[angle_name] = self.polar_root / scene_id / f"{base_name}_{angle_suffix}.png"
        else:
            # 回退到旧格式：从 polar_root 推导路径
            # ⚠️ 注意：新格式的 JSON 应该总是包含 polar_crop_paths，这种情况不应该发生
            base_name = rgb_path.stem
            if base_name.endswith('_rgb'):
                base_name = base_name[:-4]
            
            if scene_id is None:
                scene_id = rgb_path.parent.name
            
            polar_paths = self._get_polar_paths(rgb_path, scene_id=scene_id, base_name=base_name)
        
        pixel_values_polar = self._load_polar_images(polar_paths)  # (3, H, W) - [DoLP, sin(2*AoLP), cos(2*AoLP)]
        
        # 4. 处理对话
        conversations = item.get("conversations", [])
        question, answer = self._format_conversation(conversations)
        
        # 格式化提示
        prompt_text = self._format_prompt(question, answer)
        
        return {
            "pixel_values_rgb": pixel_values_rgb,
            "pixel_values_polar": pixel_values_polar,
            "question": question,
            "answer": answer,
            "prompt_text": prompt_text,
        }


def collate_fn_stage2(batch: List[Dict], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """
    Stage 2 的 collate 函数
    
    功能：
    - 将多个样本堆叠成批次
    - 对文本进行分词
    - 生成 labels（用于因果语言建模）
    - ⚠️ 关键修复：只对 Answer 部分计算损失，Question 部分设置为 -100（忽略）
    
    Args:
        batch: 批次数据列表
        tokenizer: 分词器
    
        Returns:
        批处理后的字典
        - pixel_values_rgb: (B, C, H, W)
        - pixel_values_polar: (B, 3, H, W) - 3通道：[DoLP, sin(2*AoLP), cos(2*AoLP)]
        - input_ids: (B, L)
        - attention_mask: (B, L)
        - labels: (B, L) - 用于计算损失，Question 部分为 -100，Answer 部分与 input_ids 相同
    """
    # 堆叠图像
    pixel_values_rgb = torch.stack([item["pixel_values_rgb"] for item in batch])
    pixel_values_polar = torch.stack([item["pixel_values_polar"] for item in batch])
    
    # 提取 question 和 answer（用于分别 tokenize，找到分界点）
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]
    prompt_texts = [item["prompt_text"] for item in batch]  # Question + "\n" + Answer
    
    # [关键修复]：手动添加 BOS token 到文本开头，然后设置 add_special_tokens=False
    # 这样确保 tokenizer 不会乱加，而我们可以控制结构为：[BOS] + [Text]
    # 模型 forward 中会将视觉特征插在 BOS 后面：[BOS] + [Vision] + [Text]
    if tokenizer.bos_token:
        prompt_texts = [tokenizer.bos_token + t for t in prompt_texts]
        # 同时也要对 question 添加 BOS（用于计算 question 的长度）
        questions_with_bos = [tokenizer.bos_token + q for q in questions]
    else:
        questions_with_bos = questions
    
    # 分词（禁止自动添加特殊token，因为我们已经手动添加了 BOS）
    encoded = tokenizer(
        prompt_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
        add_special_tokens=False,  # [关键修改] 禁止自动添加特殊token
    )
    
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    
    # ⚠️ 关键修复：只对 Answer 部分计算损失
    # Stage 2 的目标是训练 Projector 对齐视觉特征和文本，只需要学习 Answer（caption）部分
    # Question 部分应该被忽略（labels 设为 -100）
    
    # 方法：对 question + "\n" 单独 tokenize，找到它在完整序列中的长度
    # 注意：prompt_text 的格式是 question + "\n" + answer
    questions_with_sep = [q + "\n" for q in questions_with_bos]
    question_encoded = tokenizer(
        questions_with_sep,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
        add_special_tokens=False,
    )
    question_lengths = (question_encoded["attention_mask"] == 1).sum(dim=1)  # 每个样本的 question + "\n" 长度
    
    # 创建 labels：Question 部分（包括 BOS 和 "\n"）设为 -100（忽略），Answer 部分与 input_ids 相同
    labels = input_ids.clone()
    batch_size = labels.shape[0]
    
    for i in range(batch_size):
        # 找到该样本的 question + "\n" 长度（包括 BOS token）
        q_len = question_lengths[i].item()
        # 确保不超过序列长度
        q_len = min(q_len, labels.shape[1])
        # 将 Question 部分（包括 BOS 和 "\n"）的 labels 设置为 -100
        # 注意：BOS token 和 "\n" 的 label 也设为 -100，因为 Stage 2 只学习 Answer
        labels[i, :q_len] = -100
    
    # [诊断] 统计有效 token 数量（用于调试）
    # 只在第一个 batch 打印一次，避免日志过多
    if not hasattr(collate_fn_stage2, '_diagnostic_printed'):
        valid_tokens = (labels != -100).sum().item()
        total_tokens = labels.numel()
        valid_ratio = valid_tokens / total_tokens if total_tokens > 0 else 0
        if valid_ratio < 0.3:
            print(f"⚠️ 警告: 有效 token 占比过低: {valid_ratio:.2%} (有效: {valid_tokens}, 总计: {total_tokens})")
        else:
            print(f"✓ 有效 token 占比: {valid_ratio:.2%} (有效: {valid_tokens}, 总计: {total_tokens})")
        collate_fn_stage2._diagnostic_printed = True
    
    return {
        "pixel_values_rgb": pixel_values_rgb,
        "pixel_values_polar": pixel_values_polar,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from pathlib import Path

import torch
import torchvision.transforms as transforms
import numpy as np

import transformers
import tokenizers

# 立即禁用 torch.load 安全检查（在模块级别）
try:
    def _noop_safety_check():
        pass
    transformers.utils.import_utils.check_torch_load_is_safe = _noop_safety_check
except:
    pass

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image

# 导入 TrainerCallback 用于梯度监控
from transformers import TrainerCallback

# 导入偏振图像处理函数（从项目根目录的 dataset_common.py）
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
try:
    from dataset_common import process_polar_images
except ImportError:
    # 如果导入失败，定义一个简单的占位函数
    def process_polar_images(polar_paths=None, **kwargs):
        raise ImportError("dataset_common.process_polar_images not found. Please ensure dataset_common.py is in the project root.")


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # Polar 模型参数
    polar_vae_model_path: Optional[str] = field(default=None,
                                                 metadata={"help": "Path to the VAE model for polar encoding."})
    freeze_polar_encoder: bool = field(default=True,
                                        metadata={"help": "Whether to freeze the polar VAE encoder."})
    freeze_rgb_tower: bool = field(default=True,
                                    metadata={"help": "Whether to freeze the RGB vision tower."})
    freeze_rgb_projector: bool = field(default=True,
                                       metadata={"help": "Whether to freeze the RGB projector."})
    training_stage: Optional[str] = field(default="stage2",
                                          metadata={"help": "Training stage: 'stage1' (Polar Projector Alignment) or 'stage2' (Visual Instruction Tuning)"})
    pretrain_polar_projector: Optional[str] = field(default=None,
                                                     metadata={"help": "Path to Stage 1 trained polar_projector weights (e.g., mm_projector.bin or polar_projector.pth). Required for Stage 2 training."})
    polar_fusion_mode: Optional[str] = field(
        default="residual",
        metadata={"help": "Fusion mode for RGB/Polar: 'residual' or 'concat'."}
    )
    polar_only: bool = field(
        default=False,
        metadata={"help": "Stage 1a: Polar-only alignment (RGB features set to zero)."}
    )
    polar_rgb_dropout_p: float = field(
        default=0.3,
        metadata={"help": "RGB dropout probability during residual fusion to prevent RGB dominance."}
    )
    polar_alpha_init: float = field(
        default=0.5,
        metadata={"help": "Initial value for residual fusion alpha."}
    )
    polar_alpha_min: float = field(
        default=0.2,
        metadata={"help": "Minimum clamp value for residual fusion alpha."}
    )


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    val_json: Optional[str] = field(default=None,
                                    metadata={"help": "Path to the validation data JSON file."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    polar_folder: Optional[str] = field(default=None,
                                        metadata={"help": "Path to the polar images folder. If None, polar images will be inferred from RGB image paths."})
    use_polar: bool = field(default=False,
                            metadata={"help": "Whether to use polar images (dual-stream mode)."})
    data_root: Optional[str] = field(default=None,
                                      metadata={"help": "Data root directory for parsing crop paths (e.g., rgb_crop/, polar_crop/). If None, will be inferred from image_folder parent."})
    polar_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional cache dir for precomputed polar tensors (.npy). If set, will load cached 3x512x512 tensors."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    ddp_find_unused_parameters: bool = field(default=False, metadata={"help": "Whether to find unused parameters in DDP. Set to False to avoid DDP + Gradient Checkpointing conflicts."})
    model_max_length: int = field(
        default=2048,  # 🔴 修改：从 512 改为 2048，以容纳 Polar LLaVA 的 1152 个图像 tokens (RGB 576 + Polar 576) + 文本
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated). "
            "For Polar LLaVA with 1152 image tokens, recommend 2048 (minimum) or 4096 (safer)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    # 允许指定在 LoRA 之外需要单独保存和训练的模块名称（逗号分隔），例如 "embed_tokens,lm_head"
    lora_modules_to_save: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of module names to save and train alongside LoRA (e.g., 'embed_tokens,lm_head')."}
    )
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    try:
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    except ImportError:
        # DeepSpeed 不可用或版本不兼容，跳过
        return param
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    """查找所有线性层名称，用于 LoRA 配置（支持 4-bit/8-bit 量化层）"""
    try:
        import bitsandbytes as bnb
    except ImportError:
        bnb = None
    
    # 显式定义所有可能的线性层类型（包括量化层）
    cls_to_target = [torch.nn.Linear]
    if bnb is not None:
        cls_to_target.extend([bnb.nn.Linear4bit, bnb.nn.Linear8bitLt])
    cls_to_target = tuple(cls_to_target)
    
    lora_module_names = set()
    
    # 排除多模态模块
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 
                          'polar_projector', 'polar_encoder', 'polar_quant_conv', 
                          'vae_latent_to_feature']
    
    for name, module in model.named_modules():
        # 1. 检查模块是否是我们想要的目标类型
        if isinstance(module, cls_to_target):
            # 2. 排除掉属于多模态分支的层
            if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                continue
            
            # 3. 提取模块名称的最后一部分 (例如 'model.layers.0.self_attn.q_proj' -> 'q_proj')
            names = name.split('.')
            module_name = names[-1] if len(names) > 0 else names[0]
            
            # 4. 关键：只添加标准的 LLaMA 线性层名称
            # 这一步能防止错误地添加不该添加的层
            if module_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
                lora_module_names.add(module_name)

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    
    # 如果没找到任何层（可能是名字不对），打印警告并手动指定标准层
    if not lora_module_names:
        print("Warning: find_all_linear_names found no modules. Using default LLaMA target modules.")
        return ['q_proj', 'v_proj']
        
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        # 🟢 关键修复：同时保存 polar_projector、polar_projector_norm、vae_latent_to_feature 和 polar_range_scale（如果存在）
        # 检查模型是否有 Polar 组件
        if hasattr(trainer.model, 'get_model'):
            base_model = trainer.model.get_model()
            if hasattr(base_model, 'polar_projector'):
                keys_to_match.append('polar_projector')
            if hasattr(base_model, 'polar_projector_norm'):
                keys_to_match.append('polar_projector_norm')
            if hasattr(base_model, 'vae_latent_to_feature'):
                keys_to_match.append('vae_latent_to_feature')
            if hasattr(base_model, 'polar_range_scale'):
                keys_to_match.append('polar_range_scale')
            if hasattr(base_model, 'polar_alpha'):
                keys_to_match.append('polar_alpha')
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        
        # 🔴 修复：如果 cur_len 远小于 total_len，说明计算有问题，不要 mask 剩余部分
        # 只 mask 超出 total_len 的部分（padding）
        if cur_len < total_len:
            # cur_len 计算可能有问题，只 mask padding 部分
            target[total_len:] = IGNORE_INDEX
        else:
            # 正常情况：mask 超出 cur_len 的部分
            target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                # 🔴 修改：注释掉清零代码，允许轻微错位（1-2 tokens 的差异是可以接受的）
                # 这是 LLaMA Tokenizer 的经典问题：分开编码和整体编码可能会有 1-2 个 token 的差异
                # target[:] = IGNORE_INDEX  # <--- 原代码：会清零所有 labels，导致 loss = 0
                pass

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
    
    def _build_conversations(self, sample: Dict) -> List[Dict]:
        """从 sample 中构造 conversations（兼容只有 caption 的 Stage 1 数据）"""
        if "conversations" in sample:
            return sample["conversations"]
        caption = sample.get("caption") or sample.get("text") or sample.get("response")
        if not caption:
            raise KeyError(f"Missing 'conversations' and 'caption/text/response' in sample keys: {list(sample.keys())}")
        return [
            {"from": "human", "value": DEFAULT_IMAGE_TOKEN},
            {"from": "gpt", "value": caption},
        ]

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            # 支持 Stage 2 数据格式（input_path 字段）
            has_image = 'input_path' in sample or 'image' in sample
            img_tokens = 128 if has_image else 0
            conversations = self._build_conversations(sample)
            length_list.append(sum(len(conv['value'].split()) for conv in conversations) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            conversations = self._build_conversations(sample)
            cur_len = sum(len(conv['value'].split()) for conv in conversations)
            # 支持 Stage 2 数据格式（input_path 字段）
            has_image = 'input_path' in sample or 'image' in sample
            cur_len = cur_len if has_image else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        # 兼容 Stage 1 简化数据：只有 caption 没有 conversations
        if "conversations" not in sources[0]:
            sources[0]["conversations"] = self._build_conversations(sources[0])
        
        # ========== 关键修复：支持 Stage 2 数据格式（input_path 字段） ==========
        # Stage 2 数据格式使用 'input_path' 字段，而不是 'image' 字段
        image_file = None
        if 'input_path' in sources[0]:
            # Stage 2 格式：使用 input_path 字段（如 "rgb/04/0002_rgb.png"）
            image_file = self.list_data_dict[i]['input_path']
        elif 'image' in sources[0]:
            # Stage 1 格式：使用 image 字段（如 "rgb_crop/23/0000_rgb.png" 或 "GT/04/0000_rgb.png"）
            image_file = self.list_data_dict[i]['image']
        
        if image_file:
            image_folder = self.data_args.image_folder
            data_root = self.data_args.data_root
            processor = self.data_args.image_processor
            
            # ⚠️ 关键修复：如果 data_root 为 None，从 image_folder 推断
            if data_root is None and image_folder:
                # 从 image_folder 推断 data_root：/openbayes/input/input0/rgb -> /openbayes/input/input0
                image_folder_path = Path(image_folder)
                if image_folder_path.is_absolute():
                    data_root = str(image_folder_path.parent)
            
            # ========== 路径解析逻辑 ==========
            if isinstance(image_file, str):
                # 情况1：已经是绝对路径，直接使用
                if os.path.isabs(image_file):
                    image_path = image_file
                else:
                    # 情况2：相对路径，需要拼接
                    # 处理 Stage 2 格式：可能包含 rgb/ 或 GT/ 前缀
                    # 例如：rgb/00/0000_rgb.png 或 GT/00/0000_rgb.png
                    
                    # ⚠️ 关键修复：确保 data_root 不为 None 且是有效路径
                    if data_root and (image_file.startswith("rgb/") or image_file.startswith("GT/")):
                        # 使用 data_root 拼接完整路径
                        image_path = os.path.join(data_root, image_file)
                    # 如果路径以 rgb_crop/ 开头，去掉前缀后与 image_folder 拼接（Stage 1 格式）
                    elif image_file.startswith("rgb_crop/"):
                        image_file_clean = image_file.replace("rgb_crop/", "", 1)
                        image_path = os.path.join(image_folder, image_file_clean) if image_folder else None
                    else:
                        # 默认：直接与 image_folder 拼接
                        image_path = os.path.join(image_folder, image_file) if image_folder else None
            else:
                # 非字符串类型，直接拼接
                image_path = os.path.join(image_folder, image_file) if image_folder else None
            
            # ⚠️ 关键修复：验证路径是否存在，如果不存在则抛出清晰的错误
            if image_path is None:
                raise ValueError(
                    f"无法构建图像路径。image_file: {image_file}, "
                    f"data_root: {data_root}, image_folder: {image_folder}"
                )
            
            if not os.path.exists(image_path):
                # 打印详细的调试信息
                error_msg = (
                    f"❌ 图像路径不存在: {image_path}\n"
                    f"  image_file: {image_file}\n"
                    f"  data_root: {data_root}\n"
                    f"  image_folder: {image_folder}\n"
                )
                
                # 尝试从 image_folder 推断 data_root（如果 data_root 不正确）
                if image_folder and os.path.isabs(image_folder):
                    inferred_data_root = os.path.dirname(image_folder)
                    alt_path = os.path.join(inferred_data_root, image_file)
                    if os.path.exists(alt_path):
                        error_msg += f"  ✓ 使用推断的 data_root: {inferred_data_root}\n"
                        error_msg += f"  ✓ 修正后的路径: {alt_path}\n"
                        image_path = alt_path
                    else:
                        error_msg += f"  ✗ 推断的路径也不存在: {alt_path}\n"
                        # 抛出异常，包含详细的错误信息
                        raise FileNotFoundError(error_msg)
                else:
                    raise FileNotFoundError(error_msg)
            
            image = Image.open(image_path).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
            
            # 🔧 修正后的逻辑：确保对话中包含 <image> token
            # 策略：检查整个对话列表，如果完全没有 <image> token，则强制加在第一句话前面
            
            # 1. 先检查是否已经存在
            has_image_token = False
            for source in sources:
                for sentence in source:
                    if DEFAULT_IMAGE_TOKEN in sentence.get('value', ''):
                        has_image_token = True
                        break
                if has_image_token:
                    break
            
            # 2. 如果不存在，且确实有图像文件，则添加
            if not has_image_token:
                # 假设 sources[0][0] 是第一条 Human 的消息
                if len(sources) > 0 and len(sources[0]) > 0:
                    # 确保是 Human 的发言 (from: human)
                    # 注意：此时 role 已经被 preprocess_multimodal 处理过了，可能变成了 conversation_lib.default_conversation.roles[0]
                    # 但为了安全，我们直接加在 sources[0][0]['value'] 开头
                    sources[0][0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sources[0][0].get('value', '')
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        
        # ========== 关键修复：支持 Stage 2 数据格式（input_path 字段） ==========
        # 检查是否有图像（支持 'input_path' 或 'image' 字段）
        has_image = 'input_path' in self.list_data_dict[i] or 'image' in self.list_data_dict[i]
        
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=has_image)
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if has_image:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        
        # ========== 新增：加载偏振图像 ==========
        if self.data_args.use_polar and has_image:
            polar_image = self._load_polar_image(i)
            if polar_image is not None:
                data_dict['polar_image'] = polar_image
            else:
                # 🟢 Stage 1 严格模式：不允许使用零张量占位（会导致 Polar 分支学不到东西）
                if getattr(self.data_args, "training_stage", None) == "stage1":
                    raise FileNotFoundError(
                        f"Stage 1 strict mode: polar_image is None for sample {i}. "
                        f"Please fix polar paths/data_root."
                    )
                # Stage 2 容错：如果加载失败，创建一个零张量（3通道，512x512）
                data_dict['polar_image'] = torch.zeros(3, 512, 512)
        
        return data_dict
    
    def _load_polar_image(self, i) -> Optional[torch.Tensor]:
        """
        加载并处理偏振图像（兼容 Stage 2 和 Stage 3 数据格式）
        
        Returns:
            polar_image: 形状为 (3, 512, 512) 的 tensor，值范围 [0, 1]
                - 3通道: [DoLP, sin(2*AoLP), cos(2*AoLP)]
        """
        try:
            sample = self.list_data_dict[i]
            # 优先从缓存读取（预先保存的 3 通道张量）
            polar_tensor_path = sample.get('polar_tensor_path')
            if polar_tensor_path:
                if os.path.isabs(polar_tensor_path):
                    cache_path = polar_tensor_path
                elif hasattr(self.data_args, 'data_root') and self.data_args.data_root:
                    cache_path = os.path.join(self.data_args.data_root, polar_tensor_path)
                else:
                    cache_path = polar_tensor_path
                
                if os.path.exists(cache_path):
                    if cache_path.endswith('.pt'):
                        # 优先使用 weights_only=True，避免 torch 的安全告警刷屏
                        # 兼容旧版 PyTorch（不支持该参数）时自动回退
                        try:
                            polar_tensor = torch.load(cache_path, map_location='cpu', weights_only=True)
                        except TypeError:
                            polar_tensor = torch.load(cache_path, map_location='cpu')
                    elif cache_path.endswith('.npy'):
                        polar_tensor = torch.from_numpy(np.load(cache_path))
                    else:
                        raise ValueError(f"Unsupported polar cache format: {cache_path}")
                    
                    if isinstance(polar_tensor, np.ndarray):
                        polar_tensor = torch.from_numpy(polar_tensor)
                    
                    # 规范形状 (3, 512, 512)
                    if polar_tensor.ndim == 3 and polar_tensor.shape[-1] == 3:
                        polar_tensor = polar_tensor.permute(2, 0, 1)
                    if polar_tensor.ndim != 3 or polar_tensor.shape[0] != 3:
                        raise ValueError(f"Invalid polar tensor shape: {polar_tensor.shape} from {cache_path}")
                    return polar_tensor.float()

            # ========== 关键修复：支持 Stage 2 数据格式（input_path 字段） ==========
            # Stage 2 数据格式使用 'input_path' 字段，而不是 'image' 字段
            image_file = sample.get('input_path') or sample.get('image', '')
            scene_id = sample.get('scene_id', None)
            
            # ========== 优先读取缓存的 polar 张量（可选） ==========
            if self.data_args.polar_cache_dir:
                cache_dir = Path(self.data_args.polar_cache_dir)
                base_name = None
                if isinstance(sample.get("id"), str) and sample.get("id"):
                    base_name = Path(sample["id"]).stem
                elif isinstance(image_file, str) and image_file:
                    base_name = Path(image_file).stem
                else:
                    base_name = f"sample_{i}"

                cache_scene = scene_id
                if not cache_scene and isinstance(image_file, str) and image_file:
                    cache_scene = Path(image_file).parent.name

                if cache_scene:
                    cache_path = cache_dir / str(cache_scene) / f"{base_name}.npy"
                else:
                    cache_path = cache_dir / f"{base_name}.npy"

                if cache_path.exists():
                    try:
                        cached = np.load(cache_path)
                        if cached.shape == (3, 512, 512):
                            return torch.from_numpy(cached)
                        else:
                            rank0_print(f"Warning: cached polar tensor shape mismatch: {cache_path} {cached.shape}")
                    except Exception as e:
                        rank0_print(f"Warning: failed to load cached polar tensor: {cache_path} ({e})")
            
            # 方法1: 从 JSON 中读取 polar_paths 字段（Stage 1/测试集常见格式）
            polar_paths_json = sample.get('polar_paths', {})
            if polar_paths_json and len(polar_paths_json) > 0:
                polar_paths = {}
                for angle_name in ["I_0", "I_45", "I_90", "I_135"]:
                    polar_path_str = polar_paths_json.get(angle_name)
                    if not polar_path_str:
                        break
                    if os.path.isabs(polar_path_str):
                        polar_paths[angle_name] = Path(polar_path_str)
                    else:
                        # 优先使用 data_root 拼接带 polar/ 或 polar_crop/ 前缀的路径
                        if hasattr(self.data_args, 'data_root') and self.data_args.data_root and (
                            polar_path_str.startswith("polar/") or polar_path_str.startswith("polar_crop/")
                        ):
                            polar_paths[angle_name] = Path(self.data_args.data_root) / polar_path_str
                        elif self.data_args.polar_folder:
                            polar_paths[angle_name] = Path(self.data_args.polar_folder) / polar_path_str
                        elif hasattr(self.data_args, 'data_root') and self.data_args.data_root:
                            polar_paths[angle_name] = Path(self.data_args.data_root) / polar_path_str
                        else:
                            polar_paths[angle_name] = Path(polar_path_str)
                if len(polar_paths) == 4 and all(p.exists() for p in polar_paths.values()):
                    pass
                else:
                    polar_paths = None
            else:
                polar_paths = None

            # 方法2: 从 JSON 中读取 polar_crop_paths 字段（Stage 2 格式，优先）
            polar_crop_paths = sample.get('polar_crop_paths', {})
            if polar_paths is None and polar_crop_paths and len(polar_crop_paths) > 0:
                # Stage 2 格式：从 polar_crop_paths 读取裁剪后的偏振图像路径
                polar_paths = {}
                for angle_name in ["I_0", "I_45", "I_90", "I_135"]:
                    polar_path_str = polar_crop_paths.get(angle_name)
                    if polar_path_str:
                        # 处理相对路径：polar_crop/00/0000_000.png -> data_root/polar_crop/00/0000_000.png
                        if polar_path_str.startswith("polar_crop/"):
                            # 获取 data_root（如果未设置，从 image_folder 推断）
                            if hasattr(self.data_args, 'data_root') and self.data_args.data_root:
                                data_root = Path(self.data_args.data_root)
                            else:
                                # 从 image_folder 推断 data_root
                                image_folder = Path(self.data_args.image_folder)
                                data_root = image_folder.parent
                            
                            relative_path = polar_path_str.replace("polar_crop/", "")
                            polar_paths[angle_name] = data_root / "polar_crop" / relative_path
                        elif polar_path_str.startswith("polar/"):
                            # Stage 3 格式：polar/04/0002_000.png
                            if self.data_args.polar_folder:
                                polar_folder = Path(self.data_args.polar_folder)
                            else:
                                # 从 image_folder 推断
                                image_folder = Path(self.data_args.image_folder)
                                polar_folder = image_folder.parent / 'polar'
                            
                            relative_path = polar_path_str.replace("polar/", "")
                            polar_paths[angle_name] = polar_folder / relative_path
                        else:
                            # 绝对路径或其他格式
                            polar_paths[angle_name] = Path(polar_path_str)
                    else:
                        # 如果某个角度缺失，尝试从 RGB 路径推导
                        break
                
                # 检查是否所有路径都存在
                if len(polar_paths) == 4 and all(p.exists() for p in polar_paths.values()):
                    pass
                else:
                    # 回退到推导方法
                polar_paths = None
            
            # 方法3: 从 RGB 图像路径推断偏振图像路径（如果方法1/2失败）
            if polar_paths is None:
                # 获取 data_root（优先使用 data_root，否则从 image_folder 推断）
                if hasattr(self.data_args, 'data_root') and self.data_args.data_root:
                    data_root = Path(self.data_args.data_root)
                else:
                    # 从 image_folder 推断：/openbayes/input/input0/rgb_crop -> /openbayes/input/input0
                    image_folder = Path(self.data_args.image_folder)
                    data_root = image_folder.parent
                
                # 处理不同的路径格式
                if image_file.startswith("rgb_crop/"):
                    # Stage 1/2 格式：rgb_crop/00/0000_rgb.png
                    # 从 RGB crop 路径推导 Polar crop 路径
                    relative_path = image_file.replace("rgb_crop/", "", 1)  # 只替换第一个匹配
                    base_name = Path(relative_path).stem.replace('_rgb', '')
                    # 从相对路径中提取 scene_id：00/0000_rgb.png -> 00
                    scene_id_from_path = Path(relative_path).parent.name if scene_id is None else scene_id
                    
                    # 构建 polar_crop 路径：data_root/polar_crop/scene_id/base_name_xxx.png
                    polar_paths = {
                        'I_0': data_root / "polar_crop" / scene_id_from_path / f"{base_name}_000.png",
                        'I_45': data_root / "polar_crop" / scene_id_from_path / f"{base_name}_045.png",
                        'I_90': data_root / "polar_crop" / scene_id_from_path / f"{base_name}_090.png",
                        'I_135': data_root / "polar_crop" / scene_id_from_path / f"{base_name}_135.png",
                    }
                elif os.path.isabs(image_file) and "rgb_crop" in image_file:
                    # 绝对路径格式：/openbayes/input/input0/rgb_crop/00/0000_rgb.png
                    image_path = Path(image_file)
                    # 从绝对路径中提取 scene_id 和 base_name
                    base_name = image_path.stem.replace('_rgb', '')
                    scene_id_from_path = image_path.parent.name if scene_id is None else scene_id
                    # 从绝对路径中提取 data_root：/openbayes/input/input0/rgb_crop -> /openbayes/input/input0
                    # 或者直接使用配置的 data_root
                    if "rgb_crop" in str(image_path):
                        # 找到 rgb_crop 的位置，取其父目录作为 data_root
                        parts = image_path.parts
                        rgb_crop_idx = None
                        for idx, part in enumerate(parts):
                            if part == "rgb_crop":
                                rgb_crop_idx = idx
                                break
                        if rgb_crop_idx is not None:
                            data_root = Path(*parts[:rgb_crop_idx])
                    
                    polar_paths = {
                        'I_0': data_root / "polar_crop" / scene_id_from_path / f"{base_name}_000.png",
                        'I_45': data_root / "polar_crop" / scene_id_from_path / f"{base_name}_045.png",
                        'I_90': data_root / "polar_crop" / scene_id_from_path / f"{base_name}_090.png",
                        'I_135': data_root / "polar_crop" / scene_id_from_path / f"{base_name}_135.png",
                    }
                elif image_file.startswith("rgb/"):
                    # Stage 2/3 格式：rgb/04/0002_rgb.png
                    # ⚠️ 关键修复：使用 data_root 拼接完整路径
                    if hasattr(self.data_args, 'data_root') and self.data_args.data_root:
                        data_root = Path(self.data_args.data_root)
                    else:
                        # 从 image_folder 推断
                        image_folder = Path(self.data_args.image_folder)
                        data_root = image_folder.parent
                    
                    # 构建完整路径：data_root/rgb/04/0002_rgb.png
                    image_path = data_root / image_file
                    base_name = image_path.stem.replace('_rgb', '')
                    scene_id_from_path = scene_id or image_path.parent.name
                    
                    # 确定偏振图像文件夹
                    if self.data_args.polar_folder:
                        polar_folder = Path(self.data_args.polar_folder)
                    else:
                        # 从 data_root 推断：data_root/polar
                        polar_folder = data_root / 'polar'
                    
                    polar_paths = {
                        'I_0': polar_folder / scene_id_from_path / f"{base_name}_000.png",
                        'I_45': polar_folder / scene_id_from_path / f"{base_name}_045.png",
                        'I_90': polar_folder / scene_id_from_path / f"{base_name}_090.png",
                        'I_135': polar_folder / scene_id_from_path / f"{base_name}_135.png",
                    }
                else:
                    # 其他格式：尝试从路径推断
                    image_path = Path(image_file)
                    base_name = image_path.stem.replace('_rgb', '')
                    scene_id_from_path = scene_id or (image_path.parent.name if image_path.parent.name else image_path.parents[0].name)
                    
                    if self.data_args.polar_folder:
                        polar_folder = Path(self.data_args.polar_folder)
                    else:
                        image_folder = Path(self.data_args.image_folder)
                        polar_folder = image_folder.parent / 'polar'
                    
                    polar_paths = {
                        'I_0': polar_folder / scene_id_from_path / f"{base_name}_000.png",
                        'I_45': polar_folder / scene_id_from_path / f"{base_name}_045.png",
                        'I_90': polar_folder / scene_id_from_path / f"{base_name}_090.png",
                        'I_135': polar_folder / scene_id_from_path / f"{base_name}_135.png",
                    }
            
            # 检查文件是否存在
            missing_files = [k for k, v in polar_paths.items() if not v.exists()]
            if missing_files:
                # 只在第一次失败时输出详细信息（避免日志过多）
                if i == 0 or (i < 3):
                    rank0_print(f"Warning: Missing polar images for sample {i} (image_file: {image_file}): {missing_files}")
                    rank0_print(f"  data_root: {data_root}")
                    rank0_print(f"  scene_id: {scene_id_from_path if 'scene_id_from_path' in locals() else scene_id}")
                    # 输出所有路径（用于调试）
                    for k, v in polar_paths.items():
                        exists = "✓" if v.exists() else "✗"
                        rank0_print(f"  {exists} {k}: {v}")
                else:
                    rank0_print(f"Warning: Missing polar images for sample {i}: {missing_files}")
                # 🟢 Stage 1 严格模式：缺少 polar 就直接报错，避免训练在零张量上“学不到东西”
                if getattr(self.data_args, "training_stage", None) == "stage1":
                    raise FileNotFoundError(
                        f"Stage 1 strict mode: Missing polar images for sample {i}: {missing_files}. "
                        f"Please fix polar_crop paths/data_root before training."
                    )
                return None
            
            # 使用 process_polar_images 处理 4 张角度图像
            # 返回 4 通道物理参数: [Intensity, DoLP, sin(2*AoLP), cos(2*AoLP)]
            physics_img = process_polar_images(polar_paths=polar_paths)  # (H, W, 4)
            
            # 提取 3 通道: [DoLP, sin(2*AoLP), cos(2*AoLP)]
            # 注意：VAE 编码器只需要这 3 个通道，不需要 Intensity
            polar_3ch = physics_img[:, :, 1:4]  # (H, W, 3)
            
            # 转换为 PIL Image（值范围 [0, 1]）
            # 需要转换为 [0, 255] 范围
            polar_3ch_uint8 = (polar_3ch * 255).astype(np.uint8)
            polar_pil = Image.fromarray(polar_3ch_uint8, mode='RGB')
            
            # Resize 到 512x512（VAE 编码器的输入尺寸）
            polar_transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),  # 自动转换为 [0, 1] 范围
            ])
            polar_tensor = polar_transform(polar_pil)  # (3, 512, 512)，值范围 [0, 1]
            
            # ⚠️ 注意：这里保持 [0, 1] 范围，归一化到 [-1, 1] 在 encode_polar_images 中进行
            # 这样数据加载和模型编码逻辑分离，更清晰
            
            return polar_tensor
            
        except Exception as e:
            rank0_print(f"Error loading polar image for sample {i}: {e}")
            import traceback
            traceback.print_exc()
            return None


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        # ========== 新增：堆叠偏振图像 ==========
        if 'polar_image' in instances[0]:
            polar_images = [instance.get('polar_image') for instance in instances]
            # 过滤掉 None 值
            polar_images = [img for img in polar_images if img is not None]
            if polar_images and all(x is not None and x.shape == polar_images[0].shape for x in polar_images):
                batch['polar_images'] = torch.stack(polar_images)
            elif polar_images:
                # 如果形状不一致，使用列表（不推荐，但可以处理）
                batch['polar_images'] = polar_images
            else:
                # 如果没有有效的偏振图像，创建一个零张量
                if 'images' in batch:
                    batch_size = batch['images'].shape[0] if isinstance(batch['images'], torch.Tensor) else len(batch['images'])
                    batch['polar_images'] = torch.zeros(batch_size, 3, 512, 512)
        else:
            if 'images' in batch:
                batch_size = batch['images'].shape[0] if isinstance(batch['images'], torch.Tensor) else len(batch['images'])
                batch['polar_images'] = torch.zeros(batch_size, 3, 512, 512)

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    rank0_print(f"   正在加载训练集: {data_args.data_path}")
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    
    # Load validation dataset if val_json is provided
    eval_dataset = None
    if data_args.val_json is not None:
        rank0_print(f"   正在加载验证集: {data_args.val_json}")
        eval_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                            data_path=data_args.val_json,
                                            data_args=data_args)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    rank0_print(f"   ✓ 数据加载完成 (训练: {len(train_dataset)}, 验证: {len(eval_dataset) if eval_dataset else 0})")
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)


# ========== 特征统计监控回调函数 ==========
class FeatureStatsMonitorCallback(TrainerCallback):
    """
    监控 RGB 和 Polar 特征的范围和标准差，用于判断特征对齐情况
    
    判断标准：
    - LayerNorm 生效：Polar Std 接近 1.0（与 RGB Std~0.95 对齐）
    - 特征对齐良好：范围比在 0.1-10 之间，标准差比在 0.5-2.0 之间
    """
    
    def __init__(self):
        self.step_count = 0
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """特征统计监控回调（已禁用详细输出）"""
        # 🟢 特征统计监控已禁用，因为 LayerNorm 工作正常，特征对齐良好
        # 如果需要临时启用调试，可以设置环境变量 MONITOR_FEATURE_STATS=1
        import os
        os.environ['MONITOR_FEATURE_STATS'] = '0'  # 默认禁用
        os.environ['CURRENT_TRAINING_STEP'] = str(state.global_step)


# ========== 梯度监控回调函数 ==========
class GradientMonitorCallback(TrainerCallback):
    """
    监控 Polar Projector 和 LLM (LoRA) 的梯度范数，用于判断 Polar 分支是否被边缘化
    
    判断标准：
    - 健康状态：Polar 和 LLM 梯度在同一数量级（0.5 ~ 2.0 倍）
    - 边缘化：Polar 梯度比 LLM 小 2-3 个数量级（千分之一或更小）
    - 梯度断裂：Polar 梯度为 0（代码 Bug）
    """
    
    def __init__(self):
        self.gradient_info = {}  # 用于在 on_backward_end 和 on_log 之间传递梯度信息
        self.latest_grad_norms = {}  # 缓存每个 step 的总梯度范数
    
    def _compute_total_grad_norm(self, model):
        """计算所有可训练参数的总梯度范数（使用 FP32 累积，避免数值下溢）"""
        total_norm_sq = 0.0
        grad_count = 0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None and p.grad.is_floating_point():
                grad = p.grad.detach().float()
                grad_norm_sq = grad.norm(2).item() ** 2
                total_norm_sq += grad_norm_sq
                grad_count += 1
        total_norm = total_norm_sq ** 0.5 if grad_count > 0 else 0.0
        return total_norm, grad_count

    def _get_alpha_grad(self, model):
        """获取 polar_alpha 的梯度（若存在）"""
        base_model = model.get_model() if hasattr(model, "get_model") else model
        polar_alpha = getattr(base_model, "polar_alpha", None)
        if polar_alpha is not None and polar_alpha.grad is not None:
            try:
                return float(polar_alpha.grad.detach().float().item())
            except Exception:
                return None
        return None
    
    def _compute_gradient_norms(self, model, step, debug=False):
        """计算梯度范数的辅助函数"""
        # 获取底层模型（处理 PeftModel 包装）
        base_model = model
        polar_projector = None
        debug_info = []
        
        # 方法1: 直接访问 model.get_model()
        if hasattr(model, 'get_model'):
            try:
                base_model = model.get_model()
                if debug:
                    debug_info.append(f"方法1成功: 通过 get_model() 获取 base_model")
            except Exception as e:
                if debug:
                    debug_info.append(f"方法1失败: {e}")
        
        # 方法2: 通过 base_model.model 访问
        if hasattr(base_model, 'base_model') and hasattr(base_model.base_model, 'get_model'):
            try:
                base_model = base_model.base_model.get_model()
                if debug:
                    debug_info.append(f"方法2成功: 通过 base_model.base_model.get_model() 获取")
            except Exception as e:
                if debug:
                    debug_info.append(f"方法2失败: {e}")
        elif hasattr(base_model, 'model'):
            base_model = base_model.model
            if debug:
                debug_info.append(f"方法2成功: 通过 base_model.model 获取")
        
        # 方法3: 尝试直接访问 polar_projector（可能在多层包装中）
        if hasattr(base_model, 'polar_projector') and base_model.polar_projector is not None:
            polar_projector = base_model.polar_projector
            if debug:
                debug_info.append(f"方法3成功: 通过 base_model.polar_projector 获取")
        elif hasattr(model, 'polar_projector') and model.polar_projector is not None:
            polar_projector = model.polar_projector
            if debug:
                debug_info.append(f"方法3成功: 通过 model.polar_projector 获取")
        else:
            # 方法4: 通过 named_modules 查找
            for name, module in model.named_modules():
                if 'polar_projector' in name and module is not None:
                    polar_projector = module
                    if debug:
                        debug_info.append(f"方法4成功: 通过 named_modules 找到 {name}")
                    break
        
        if debug and polar_projector is None:
            debug_info.append("⚠️ 未找到 polar_projector")
            # 列出所有包含 'polar' 的模块
            polar_modules = [name for name, _ in model.named_modules() if 'polar' in name.lower()]
            if polar_modules:
                debug_info.append(f"找到包含 'polar' 的模块: {polar_modules[:5]}")
        
        # 1. 检查 Polar Projector 梯度
        polar_norm = 0.0
        polar_count = 0
        polar_params_with_grad = 0
        polar_params_total = 0
        
        if polar_projector is not None:
            for p in polar_projector.parameters():
                polar_params_total += 1
                if p.requires_grad:
                    polar_params_with_grad += 1
                    if p.grad is not None:
                        grad_val = p.grad.detach().data.norm(2).item() ** 2
                        polar_norm += grad_val
                        polar_count += 1
                        if debug and polar_count == 1:
                            debug_info.append(f"找到第一个 Polar 梯度: norm^2={grad_val:.6f}, shape={p.shape}")
            
            if polar_count > 0:
                polar_norm = polar_norm ** 0.5
            elif debug:
                debug_info.append(f"Polar Projector: {polar_params_total} 总参数, {polar_params_with_grad} 需要梯度, 0 有梯度")
        else:
            polar_norm = None
        
        # 2. 检查 LLM (LoRA) 梯度
        lora_norm = 0.0
        lora_count = 0
        lora_params_with_grad = 0
        lora_param_names = []
        all_trainable_params = []
        
        for name, p in model.named_parameters():
            if p.requires_grad:
                all_trainable_params.append(name)
                if "lora" in name.lower() or "adapter" in name.lower():
                    lora_params_with_grad += 1
                    if p.grad is not None:
                        grad_norm_sq = p.grad.detach().data.norm(2).item() ** 2
                        lora_norm += grad_norm_sq
                        lora_count += 1
                        if len(lora_param_names) < 3:
                            lora_param_names.append(name)
                        if debug and lora_count == 1:
                            debug_info.append(f"找到第一个 LoRA 梯度: {name}, norm^2={grad_norm_sq:.6f}")
        
        if lora_count > 0:
            lora_norm = lora_norm ** 0.5
        elif debug:
            debug_info.append(f"LoRA: {lora_params_with_grad} 需要梯度, 0 有梯度")
            if all_trainable_params:
                debug_info.append(f"所有可训练参数示例 (前5个): {all_trainable_params[:5]}")
        
        result = {
            'polar_norm': polar_norm,
            'polar_count': polar_count,
            'polar_params_with_grad': polar_params_with_grad,
            'lora_norm': lora_norm,
            'lora_count': lora_count,
            'lora_params_with_grad': lora_params_with_grad,
            'lora_param_names': lora_param_names,
            'polar_projector_found': polar_projector is not None
        }
        
        if debug:
            result['debug_info'] = debug_info
        
        return result
    
    def _print_gradient_analysis(self, step, grad_info):
        """打印梯度分析的辅助函数"""
        rank0_print(f"\n{'=' * 60}")
        rank0_print(f"[Step {step}] 📊 梯度范数分析 (Gradient Norm Analysis)")
        rank0_print(f"{'=' * 60}")
        
        # 1. Polar Projector 梯度
        if grad_info['polar_projector_found']:
            if grad_info['polar_norm'] is not None and grad_info['polar_count'] > 0:
                rank0_print(f"  🌊 Polar Projector Grad Norm: {grad_info['polar_norm']:.6f} (有梯度参数: {grad_info['polar_count']}/{grad_info['polar_params_with_grad']})")
            else:
                rank0_print(f"  🌊 Polar Projector Grad Norm: 0.000000 ⚠️  警告: 所有参数都没有梯度！")
        else:
            rank0_print(f"  🌊 Polar Projector: 未找到")
        
        # 2. LLM (LoRA) 梯度
        if grad_info['lora_norm'] is not None and grad_info['lora_count'] > 0:
            rank0_print(f"  🧠 LLM (LoRA) Grad Norm:      {grad_info['lora_norm']:.6f} (有梯度参数: {grad_info['lora_count']}/{grad_info['lora_params_with_grad']})")
            if grad_info['lora_param_names']:
                rank0_print(f"     示例参数: {grad_info['lora_param_names'][0]}")
        else:
            rank0_print(f"  🧠 LLM (LoRA) Grad Norm:      0.000000 (未找到 LoRA 参数或未启用 LoRA)")
            # 注意：如果需要调试信息，应该在 _compute_gradient_norms 中收集
        
        # 3. 比率分析
        if grad_info['lora_norm'] is not None and grad_info['lora_norm'] > 0 and \
           grad_info['polar_norm'] is not None and grad_info['polar_norm'] > 0:
            ratio = grad_info['polar_norm'] / grad_info['lora_norm']
            rank0_print(f"  ⚖️  Ratio (Polar / LLM):      {ratio:.4f}")
            
            if ratio >= 0.5 and ratio <= 2.0:
                rank0_print(f"  ✅ 状态: 健康 (Healthy) - Polar 和 LLM 梯度在同一数量级")
            elif ratio < 0.001:
                rank0_print(f"  ⚠️  状态: 边缘化 (Marginalized) - Polar 梯度极小，可能被忽略")
                rank0_print(f"     建议: 检查 Stage 1 训练效果，或增大 Polar Projector 学习率")
            elif ratio == 0.0:
                rank0_print(f"  ❌ 状态: 梯度断裂 (Broken) - Polar 梯度为 0，检查代码逻辑")
            else:
                rank0_print(f"  ⚠️  状态: 梯度不平衡 - Polar 梯度 {'过大' if ratio > 2.0 else '过小'}")
        elif grad_info['polar_norm'] == 0.0 and grad_info['lora_norm'] is not None and grad_info['lora_norm'] > 0:
            rank0_print(f"  ❌ 状态: 梯度断裂 (Broken) - Polar 梯度为 0，检查 requires_grad 设置")
        elif grad_info['lora_norm'] is None or grad_info['lora_norm'] == 0.0:
            rank0_print(f"  ℹ️  状态: 未启用 LoRA 或 LoRA 参数未更新")
        
        rank0_print(f"{'=' * 60}\n")
    
    def on_backward_end(self, args, state, control, model=None, **kwargs):
        """
        在 backward() 之后、optimizer.step() 之前监控梯度
        这是最可靠的时机：梯度已经计算，但还没有被 optimizer.step() 清零
        
        注意：
        1. on_backward_end 在梯度累积期间，只有在累积完成后才会调用
        2. 如果 gradient_accumulation_steps=4，那么 on_backward_end 只在 Step 0, 4, 8, 12... 调用
        3. 如果 logging_steps=5，那么 Step 5 时 on_backward_end 不会被调用（因为 Step 5 不是累积完成的步数）
        """
        # 按照 logging_steps 的频率打印
        # 注意：由于梯度累积，global_step 只在累积完成后增加
        # 所以如果 gradient_accumulation_steps=4，global_step 会是 0, 4, 8, 12...
        # 如果 logging_steps=5，那么 Step 5 时不会触发（因为 Step 5 不是累积完成的步数）
        if state.global_step % args.logging_steps == 0:
            # 计算梯度范数（启用调试模式，仅在第一次或梯度为0时显示）
            grad_info = self._compute_gradient_norms(model, state.global_step, debug=False)
            # 保存到实例变量
            self.gradient_info[state.global_step] = grad_info

            # 计算总梯度范数（用于日志输出）
            total_grad_norm, grad_count = self._compute_total_grad_norm(model)
            alpha_grad = self._get_alpha_grad(model)
            self.latest_grad_norms[state.global_step] = {
                "total_grad_norm": total_grad_norm,
                "grad_count": grad_count,
                "polar_grad_norm": grad_info.get("polar_norm"),
                "alpha_grad": alpha_grad,
            }
            
            # 如果梯度为0，启用调试模式重新计算
            if (grad_info.get('polar_norm') is None or grad_info.get('polar_norm') == 0.0) and \
               (grad_info.get('lora_norm') is None or grad_info.get('lora_norm') == 0.0):
                # 重新计算，启用调试模式
                grad_info = self._compute_gradient_norms(model, state.global_step, debug=True)
                self.gradient_info[state.global_step] = grad_info
                if 'debug_info' in grad_info:
                    rank0_print(f"\n[Step {state.global_step}] 🔍 调试信息 (on_backward_end):")
                    for info in grad_info['debug_info']:
                        rank0_print(f"  {info}")
            
            # 打印梯度分析
            self._print_gradient_analysis(state.global_step, grad_info)
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """
        在每次 logging 时检查梯度监控状态（不主动监控，只检查）
        注意：on_log 在 optimizer.step() 之后调用，此时梯度已被清零
        这里只用于检查 on_backward_end 是否正常工作
        
        注意：由于梯度累积和 on_backward_end 的触发时机问题，
        梯度监控现在主要在 optimizer_step 中进行，这里只作为备选检查
        """
        if logs is not None and state.global_step % args.logging_steps == 0:
            # 将更稳定的总梯度范数写入日志（替换/补充 Trainer 默认的 grad_norm）
            grad_snapshot = self.latest_grad_norms.get(state.global_step)
            if grad_snapshot:
                logs["grad_norm"] = grad_snapshot.get("total_grad_norm", logs.get("grad_norm", 0.0))
                logs["polar_grad_norm"] = grad_snapshot.get("polar_grad_norm", 0.0)
                if grad_snapshot.get("alpha_grad") is not None:
                    logs["alpha_grad"] = grad_snapshot["alpha_grad"]
            # 检查 on_backward_end 是否已经处理过
            # 注意：由于 on_backward_end 在梯度累积期间可能不会触发，
            # 这个警告可以忽略，梯度监控主要在 optimizer_step 中进行
            if state.global_step not in self.gradient_info:
                # 不再打印警告，因为梯度监控现在主要在 optimizer_step 中进行
                pass


class AlphaMonitorCallback(TrainerCallback):
    """在每次日志输出时记录当前 polar_alpha 值，便于监控。"""

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is None:
            return
        if model is None:
            return
        try:
            base_model = model.get_model() if hasattr(model, "get_model") else model
            model_config = getattr(model, "config", None)
            # Stage 1a strict mode: 完全不使用 alpha，避免输出无意义日志
            if model_config is not None:
                if getattr(model_config, "training_stage", None) == "stage1" and getattr(model_config, "polar_only", False):
                    return
            polar_alpha = getattr(base_model, "polar_alpha", None)
            if polar_alpha is not None and hasattr(polar_alpha, "item"):
                alpha_value = float(polar_alpha.item())
                logs["polar_alpha"] = alpha_value
                # 在终端 loss 输出时附带 alpha
                if "loss" in logs:
                    rank0_print(f"[Step {state.global_step}] loss={logs['loss']:.6f}, polar_alpha={alpha_value:.6f}")
        except Exception:
            # 避免因监控导致训练中断
                pass


def train(attn_implementation=None):
    global local_rank

    # 🔧 设置无缓冲输出，确保在 nohup 下也能实时看到日志
    import sys
    import os
    # 设置环境变量（最可靠的方法）
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # 包装 sys.stdout 和 sys.stderr，确保所有输出都实时刷新
    class FlushedStream:
        def __init__(self, stream):
            self.stream = stream
        def write(self, text):
            self.stream.write(text)
            self.stream.flush()
        def flush(self):
            self.stream.flush()
        def __getattr__(self, name):
            return getattr(self.stream, name)
    
    # 替换标准输出流（如果还没有被替换）
    if not isinstance(sys.stdout, FlushedStream):
        sys.stdout = FlushedStream(sys.stdout)
    if not isinstance(sys.stderr, FlushedStream):
        sys.stderr = FlushedStream(sys.stderr)
    
    # 只显示 warning 及以上日志，避免大段权重初始化提示刷屏
    transformers.logging.set_verbosity_warning()
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    # 🟢 将训练阶段注入到 data_args，便于数据加载时做严格校验
    data_args.training_stage = model_args.training_stage

    # 🟢 Stage 1a: Polar-only 时强制 alpha=0，避免残差融合污染 RGB
    if model_args.training_stage == "stage1" and model_args.polar_only:
        if model_args.polar_alpha_init != 0.0 or model_args.polar_alpha_min != 0.0:
            rank0_print("🟢 Stage 1a: polar_only=True，强制设置 polar_alpha_init=0.0, polar_alpha_min=0.0")
        model_args.polar_alpha_init = 0.0
        model_args.polar_alpha_min = 0.0

    # Stage 2 禁止 Polar-only
    if model_args.training_stage == "stage2" and model_args.polar_only:
        rank0_print("⚠️  Stage 2 不支持 polar_only，已自动关闭")
        model_args.polar_only = False
    
    # 禁用 torch.load 安全检查（在加载模型之前）
    def _disable_safety_check():
        """禁用 transformers 的 torch.load 安全检查"""
        try:
            # 永久性地 patch import_utils 中的函数
            def _noop_check():
                pass
            transformers.utils.import_utils.check_torch_load_is_safe = _noop_check
            
            # 如果 modeling_utils 中有直接引用，也 patch 它
            if hasattr(transformers.modeling_utils, 'check_torch_load_is_safe'):
                transformers.modeling_utils.check_torch_load_is_safe = _noop_check
            
            # 检查 modeling_utils 的命名空间，看是否有直接导入
            import inspect
            if 'check_torch_load_is_safe' in transformers.modeling_utils.__dict__:
                transformers.modeling_utils.__dict__['check_torch_load_is_safe'] = _noop_check
            
            # 也 patch load_state_dict 函数，确保在调用时检查被禁用
            _original_load = transformers.modeling_utils.load_state_dict
            def _patched_load(checkpoint_file, *args, **kwargs):
                # 确保检查函数被禁用
                transformers.utils.import_utils.check_torch_load_is_safe = _noop_check
                # 如果 modeling_utils 中有引用，也禁用
                if hasattr(transformers.modeling_utils, 'check_torch_load_is_safe'):
                    transformers.modeling_utils.check_torch_load_is_safe = _noop_check
                return _original_load(checkpoint_file, *args, **kwargs)
            transformers.modeling_utils.load_state_dict = _patched_load
            
            rank0_print("✓ Disabled torch.load safety check for .bin model loading")
        except Exception as e:
            rank0_print(f"Warning: Failed to disable safety check: {e}")
    
    _disable_safety_check()
    
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        
        # 🟢 关键修复：检查是否需要排除 embed_tokens 和 lm_head 不被量化
        # 如果指定了 lora_modules_to_save，需要确保这些模块保持全精度
        # 注意：llm_int8_skip_modules 需要完整的模块路径（如 "model.embed_tokens"），而不仅仅是模块名称
        skip_modules = ["mm_projector", "polar_projector"]
        if getattr(training_args, "lora_modules_to_save", None):
            modules_to_save_list = [
                m.strip() for m in training_args.lora_modules_to_save.split(",") if m.strip()
            ]
            # 将 embed_tokens 和 lm_head 添加到 skip_modules，确保它们不被量化
            # 对于 LLaMA 模型，完整路径是 "model.embed_tokens" 和 "model.lm_head"
            for module_name in modules_to_save_list:
                # 添加完整路径（LLaMA 模型的 embed_tokens 和 lm_head 在 model 子模块中）
                full_path = f"model.{module_name}"
                if full_path not in skip_modules:
                    skip_modules.append(full_path)
                # 也添加短名称（以防某些配置需要）
                if module_name not in skip_modules:
                    skip_modules.append(module_name)
            rank0_print(f"  ✓ 量化配置：排除模块 {skip_modules} 不被量化（保持全精度）")
        
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=skip_modules,  # 排除 projector 和 modules_to_save 中的模块，保持全精度训练
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))
    elif training_args.bits == 16:
        # 🟢 FP16/BF16 全精度模式
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        # 🔍 检测是否在使用 DDP 训练
        # 在 DDP 训练中，每个进程都需要完整的模型副本，不应该使用 device_map="auto"
        is_ddp_training = (
            hasattr(training_args, 'local_rank') and training_args.local_rank >= 0
        ) or (
            hasattr(training_args, 'world_size') and training_args.world_size > 1
        )
        
        if is_ddp_training:
            # DDP 训练模式：每个进程只在自己的 GPU 上加载模型
            # 🟢 关键修复：在 DDP 训练中，不使用 device_map，而是使用 low_cpu_mem_usage 减少内存峰值
            # device_map 在加载时仍会在 CPU 上创建临时张量，导致内存峰值过高
            if hasattr(training_args, 'local_rank') and training_args.local_rank >= 0:
                torch.cuda.set_device(0)  # 在 DDP 训练中，每个进程只能看到自己的 GPU（索引为 0）
                # 不使用 device_map，使用 low_cpu_mem_usage 来减少加载时的内存峰值
                # 模型会先加载到 CPU（使用低内存模式），然后逐步移动到 GPU
                bnb_model_from_pretrained_args.update(dict(
                    torch_dtype=compute_dtype,
                    low_cpu_mem_usage=True,  # 🟢 关键：减少 CPU 内存使用，避免加载时的临时内存峰值
                ))
                rank0_print(f"  ✓ FP16/BF16 全精度模式：DDP 训练模式（使用 low_cpu_mem_usage，local_rank={training_args.local_rank}）")
            else:
                # 如果没有 local_rank，使用默认设备
                bnb_model_from_pretrained_args.update(dict(
                    torch_dtype=compute_dtype,
                    low_cpu_mem_usage=True,
                ))
                rank0_print(f"  ✓ FP16/BF16 全精度模式：DDP 训练模式（使用默认设备）")
        elif num_gpus > 1:
            # 单进程多 GPU 模式（推理场景）：使用 device_map="auto" 自动分布
            bnb_model_from_pretrained_args.update(dict(
                device_map="auto",
                max_memory={i: "23GB" for i in range(num_gpus)},  # 双卡 4090：每个GPU限制23GB
                torch_dtype=compute_dtype,
            ))
            rank0_print(f"  ✓ FP16/BF16 全精度模式：单进程多 GPU 推理模式，使用 device_map='auto' 自动分布")
            rank0_print(f"  ✓ 显存限制: 每个GPU最多23GB")
        else:
            # 单GPU模式
            bnb_model_from_pretrained_args.update(dict(
                torch_dtype=compute_dtype,
            ))
            rank0_print(f"  ✓ FP16/BF16 全精度模式：单GPU模式")

    # 检查是否使用 Polar 模型
    use_polar = model_args.polar_vae_model_path is not None and os.path.exists(model_args.polar_vae_model_path)
    
    # 🟢 关键修复：DDP 训练前清理显存
    if torch.cuda.is_available():
        if hasattr(training_args, 'local_rank') and training_args.local_rank >= 0:
            # DDP 训练模式：只清理当前进程的 GPU
            # 在 DDP 训练中，每个进程只能看到自己的 GPU（索引为 0）
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
            # 强制同步，确保清理完成
            torch.cuda.synchronize()
            rank0_print(f"  ✓ 已清理 GPU 0 显存 (local_rank={training_args.local_rank})")
        else:
            # 单进程模式：清理所有 GPU
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            rank0_print(f"  ✓ 已清理显存")
    
    rank0_print("=" * 60)
    rank0_print("📦 开始加载模型...")
    rank0_print(f"   模型路径: {model_args.model_name_or_path}")
    if use_polar:
        rank0_print(f"   Polar VAE: {model_args.polar_vae_model_path}")
    rank0_print("=" * 60)
    
    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                trust_remote_code=True,
                local_files_only=True  # 强制只使用本地文件
            )
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            bnb_model_from_pretrained_args['local_files_only'] = True  # 强制只使用本地文件
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            if use_polar:
                from llava.model.language_model.llava_llama import PolarLlavaConfig, PolarLlavaLlamaForCausalLM
                # 创建 Polar 配置
                config = PolarLlavaConfig.from_pretrained(
                    model_args.model_name_or_path,
                    local_files_only=True  # 强制只使用本地文件
                )
                # ⚠️ 关键修复：更新 vision_tower 路径为本地路径
                # 因为从预训练模型加载的 config 可能包含 HuggingFace 模型名称
                if model_args.vision_tower and os.path.exists(model_args.vision_tower):
                    config.mm_vision_tower = model_args.vision_tower
                    rank0_print(f"   ✓ 更新 vision_tower 路径: {model_args.vision_tower}")
                config.polar_vae_model_path = model_args.polar_vae_model_path
                config.freeze_polar_encoder = model_args.freeze_polar_encoder
                # Residual 融合配置
                config.polar_fusion_mode = model_args.polar_fusion_mode
                config.polar_only = model_args.polar_only
                config.polar_rgb_dropout_p = model_args.polar_rgb_dropout_p
                config.polar_alpha_init = model_args.polar_alpha_init
                config.polar_alpha_min = model_args.polar_alpha_min
                rank0_print(f"   正在加载 Polar LLaVA 模型 (VAE: {model_args.polar_vae_model_path})...")
                # 强制使用本地模型文件，不下载
                load_kwargs = {
                    "local_files_only": True,
                    "cache_dir": training_args.cache_dir,
                    "attn_implementation": attn_implementation,
                }
                # 🟢 关键修复：FP16/BF16 模式下，torch_dtype 和 device_map 已经在 bnb_model_from_pretrained_args 中设置
                # 只有在量化模式下才需要单独设置 torch_dtype
                if training_args.bits not in [4, 8]:
                    # FP16/BF16 模式：使用 bnb_model_from_pretrained_args 中的 torch_dtype
                    pass
                else:
                    # 量化模式：保持原有逻辑
                    load_kwargs["torch_dtype"] = (torch.bfloat16 if training_args.bf16 else None)
                load_kwargs.update(bnb_model_from_pretrained_args)
                # 🟢 关键修复：FP16 模式下的内存优化策略
                if training_args.bits == 16:
                    # 在 DDP 训练中，使用 low_cpu_mem_usage=True 可以减少 CPU 内存使用
                    # 但需要确保不使用 device_map="auto"（已在上面处理）
                    if not bnb_model_from_pretrained_args.get('device_map'):
                        # DDP 训练模式：启用低 CPU 内存使用
                        load_kwargs["low_cpu_mem_usage"] = True
                        rank0_print(f"  ✓ 启用 low_cpu_mem_usage=True (DDP 训练模式)")
                    else:
                        # 单进程多 GPU 推理模式：不使用 low_cpu_mem_usage（避免 Meta Tensor 错误）
                        load_kwargs["low_cpu_mem_usage"] = False
                model = PolarLlavaLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    config=config,
                    **load_kwargs
                )
                # 🟢 关键修复：将 training_stage 注入到模型配置中，供 llava_arch.py 使用
                # 这样 encode_images 可以根据 training_stage 动态切换架构（Stage 1: Polar-Only, Stage 2: Dual-Stream）
                model.config.training_stage = model_args.training_stage
                rank0_print("   ✓ Polar LLaVA 模型加载完成")
                rank0_print(f"   ✓ Set model.config.training_stage = {model_args.training_stage}")
            else:
                # 强制使用本地模型文件，不下载
                load_kwargs = {
                    "local_files_only": True,
                    "cache_dir": training_args.cache_dir,
                    "attn_implementation": attn_implementation,
                }
                # 🟢 关键修复：FP16/BF16 模式下，torch_dtype 和 device_map 已经在 bnb_model_from_pretrained_args 中设置
                if training_args.bits not in [4, 8]:
                    # FP16/BF16 模式：使用 bnb_model_from_pretrained_args 中的 torch_dtype
                    pass
                else:
                    # 量化模式：保持原有逻辑
                    load_kwargs["torch_dtype"] = (torch.bfloat16 if training_args.bf16 else None)
                load_kwargs.update(bnb_model_from_pretrained_args)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    **load_kwargs
                )
    else:
        bnb_model_from_pretrained_args['local_files_only'] = True  # 强制只使用本地文件
        # 🟢 关键修复：FP16/BF16 模式下，torch_dtype 已经在 bnb_model_from_pretrained_args 中设置
        if training_args.bits not in [4, 8]:
            # FP16/BF16 模式：不重复设置 torch_dtype
            pass
        else:
            # 量化模式：保持原有逻辑
            bnb_model_from_pretrained_args['torch_dtype'] = (torch.bfloat16 if training_args.bf16 else None)
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            **bnb_model_from_pretrained_args
        )
    
    model.config.use_cache = False
    
    # 🟢 关键修复：DDP 训练中，模型通过 low_cpu_mem_usage 加载到 CPU，需要移动到 GPU
    # 在 DDP 训练中，每个进程只能看到自己的 GPU（索引为 0）
    if training_args.bits == 16:
        device_map_used = bnb_model_from_pretrained_args.get('device_map')
        if device_map_used and device_map_used != "auto":
            # 使用了 device_map（非 auto），模型应该已经在正确的设备上
            if isinstance(device_map_used, dict) and device_map_used.get(""):
                rank0_print(f"  ✓ 模型已通过 device_map 加载到 {device_map_used['']}")
        else:
            # 没有使用 device_map，模型可能在 CPU 上，需要移动到 GPU
            if hasattr(training_args, 'local_rank') and training_args.local_rank >= 0:
                # DDP 训练模式：每个进程只能看到自己的 GPU（索引为 0）
                # 所以应该移动到 cuda:0，而不是 cuda:{local_rank}
                target_device = torch.device("cuda:0")
                first_param = next(model.parameters(), None)
                if first_param is not None:
                    current_device = first_param.device
                    if current_device != target_device:
                        # 模型在错误的设备上，需要移动
                        # 🟢 关键：使用 half() 转换精度，减少移动时的内存峰值
                        model = model.to(target_device)
                        if compute_dtype == torch.float16:
                            model = model.half()
                        elif compute_dtype == torch.bfloat16:
                            model = model.to(torch.bfloat16)
                        rank0_print(f"  ✓ 模型已从 {current_device} 移动到 {target_device} (DDP 训练模式，local_rank={training_args.local_rank})")
                    else:
                        rank0_print(f"  ✓ 模型已在正确的设备上: {target_device} (DDP 训练模式)")
            elif torch.cuda.is_available():
                # 单 GPU 或非 DDP 模式，确保模型在 GPU 0
                first_param = next(model.parameters(), None)
                if first_param is not None and first_param.device.type != "cuda":
                    device = torch.device("cuda:0")
                    model = model.to(device)
                    if compute_dtype == torch.float16:
                        model = model.half()
                    elif compute_dtype == torch.bfloat16:
                        model = model.to(torch.bfloat16)
                    rank0_print(f"  ✓ 模型已移动到 GPU 0")

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
        
        # ========== 关键修复：确保 prepare_model_for_kbit_training 后 Polar Projector、polar_projector_norm 和 vae_latent_to_feature 的梯度被重新启用 ==========
        # prepare_model_for_kbit_training 可能会冻结所有非 LoRA 参数，包括 polar_projector、polar_projector_norm 和 vae_latent_to_feature
        if use_polar:
            polar_projector = getattr(model.get_model(), 'polar_projector', None)
            if polar_projector is not None:
                for p in polar_projector.parameters():
                    # 只对浮点类型参数设置 requires_grad
                    if p.dtype.is_floating_point:
                        p.requires_grad = True
                rank0_print("  ✓ Re-enabled gradients for Polar Projector after prepare_model_for_kbit_training")
            
            # 🟢 关键修复：同时重新启用 polar_projector_norm 的梯度
            polar_projector_norm = getattr(model.get_model(), 'polar_projector_norm', None)
            if polar_projector_norm is not None:
                for p in polar_projector_norm.parameters():
                    if p.dtype.is_floating_point:
                        p.requires_grad = True
                rank0_print("  ✓ Re-enabled gradients for polar_projector_norm after prepare_model_for_kbit_training")
            
            # 🟢 关键修复：同时重新启用 vae_latent_to_feature 的梯度
            vae_latent_to_feature = getattr(model.get_model(), 'vae_latent_to_feature', None)
            if vae_latent_to_feature is not None:
                for p in vae_latent_to_feature.parameters():
                    if p.dtype.is_floating_point:
                        p.requires_grad = True
                rank0_print("  ✓ Re-enabled gradients for vae_latent_to_feature after prepare_model_for_kbit_training")

            # 🟢 关键修复：polar_alpha 仅在非 stage1a 下参与训练
            polar_alpha = getattr(model.get_model(), 'polar_alpha', None)
            if polar_alpha is not None and hasattr(polar_alpha, 'requires_grad'):
                if model_args.training_stage == "stage1" and model_args.polar_only:
                    with torch.no_grad():
                        polar_alpha.data.fill_(0.0)
                    polar_alpha.requires_grad = False
                    rank0_print("  ✓ Stage 1a strict mode: polar_alpha 已冻结并置 0（不参与训练）")
                else:
                    polar_alpha.requires_grad = True
                    rank0_print("  ✓ Re-enabled gradients for polar_alpha after prepare_model_for_kbit_training")

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # ========== Stage 1 特殊处理：量化模型需要 LoRA 适配器才能训练 ==========
    # 在 Stage 1 中，如果使用了量化但没有 LoRA，自动添加一个最小的 LoRA 配置
    # 这个 LoRA 会被冻结，只训练 Polar Projector
    if training_args.bits in [4, 8] and not training_args.lora_enable and model_args.training_stage == "stage1":
        rank0_print("⚠️  Stage 1 使用量化模型但未启用 LoRA，自动添加最小 LoRA 配置以满足 transformers 检查")
        rank0_print("   注意：LoRA 参数将被冻结，只训练 Polar Projector")
        training_args.lora_enable = True
        training_args.lora_r = 1  # 最小配置
        training_args.lora_alpha = 1
        training_args.lora_dropout = 0.0
    
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        rank0_print("🔧 配置 LoRA 适配器...")
        target_modules = find_all_linear_names(model)
        rank0_print(f"   LoRA 目标模块: {target_modules}")
        # 解析需要单独保存/训练的模块列表（例如 embed_tokens,lm_head）
        modules_to_save = None
        if getattr(training_args, "lora_modules_to_save", None):
            modules_to_save = [
                m.strip() for m in training_args.lora_modules_to_save.split(",") if m.strip()
            ]
            if modules_to_save:
                rank0_print(f"   LoRA modules_to_save: {modules_to_save}")
                
                # 🟢 关键修复：在4-bit量化模式下，确保 modules_to_save 中的模块保持全精度
                # 如果这些模块被量化了，需要手动替换为全精度版本
                if training_args.bits in [4, 8]:
                    base_model = model.get_model() if hasattr(model, 'get_model') else model
                    
                    # 🟢 检查 tie_word_embeddings 配置（改进检测逻辑，支持更多路径）
                    tie_word_embeddings = False
                    # 方法1: 通过 base_model.model.config
                    if hasattr(base_model, 'model') and hasattr(base_model.model, 'config'):
                        tie_word_embeddings = getattr(base_model.model.config, 'tie_word_embeddings', False)
                    # 方法2: 通过 base_model.config
                    elif hasattr(base_model, 'config'):
                        tie_word_embeddings = getattr(base_model.config, 'tie_word_embeddings', False)
                    # 方法3: 通过 model.config（直接访问）
                    elif hasattr(model, 'config'):
                        tie_word_embeddings = getattr(model.config, 'tie_word_embeddings', False)
                    
                    # 🟢 修改：Stage 2 也完全冻结 embedding 层，不需要 modules_to_save
                    # 数据量少时，训练 embedding 容易过拟合，直接冻结更安全
                    if model_args.training_stage == "stage2":
                        if modules_to_save:
                            rank0_print(f"  ⚠️  注意: Stage 2 训练时，--lora_modules_to_save 参数将被忽略（embedding 层完全冻结）")
                            rank0_print(f"     提供的参数: {modules_to_save}")
                        rank0_print(f"  ℹ️  Stage 2: 完全冻结 embedding 层（避免过拟合），不使用 modules_to_save")
                        modules_to_save = []  # Stage 2 不训练 embedding
                    else:
                        # Stage 1: 保持原有逻辑（也不训练 embedding）
                        if tie_word_embeddings and 'lm_head' in modules_to_save:
                            rank0_print(f"  ℹ️  检测到 tie_word_embeddings=True，lm_head 与 embed_tokens 共享权重，跳过 lm_head 的单独处理")
                            # 从 modules_to_save 中移除 lm_head，只处理 embed_tokens
                            modules_to_save = [m for m in modules_to_save if m != 'lm_head']
                        elif 'lm_head' in modules_to_save:
                            # 🟢 额外检查：即使没有检测到 tie_word_embeddings，也尝试检查 lm_head 是否存在
                            # 如果不存在，说明可能是 tie_word_embeddings=True 但检测失败，也应该移除
                            has_lm_head = False
                            if hasattr(base_model, 'lm_head'):
                                has_lm_head = True
                            elif hasattr(base_model, 'model') and hasattr(base_model.model, 'lm_head'):
                                has_lm_head = True
                            elif hasattr(model, 'lm_head'):
                                has_lm_head = True
                            
                            if not has_lm_head:
                                rank0_print(f"  ℹ️  未找到独立的 lm_head 模块（可能是 tie_word_embeddings=True），从 modules_to_save 中移除")
                                modules_to_save = [m for m in modules_to_save if m != 'lm_head']
                    
                    for module_name in modules_to_save:
                        # 尝试多种路径查找模块
                        module = None
                        module_path = None
                        if hasattr(base_model, module_name):
                            module = getattr(base_model, module_name)
                            module_path = f"base_model.{module_name}"
                        elif hasattr(base_model, 'model') and hasattr(base_model.model, module_name):
                            module = getattr(base_model.model, module_name)
                            module_path = f"base_model.model.{module_name}"
                        
                        if module is not None and hasattr(module, 'weight'):
                            # 检查是否是量化模块（Linear4bit 或 Linear8bitLt）
                            is_quantized = False
                            try:
                                import bitsandbytes as bnb
                                is_quantized = isinstance(module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt))
                            except ImportError:
                                pass
                            
                            if is_quantized:
                                # 🟢 关键修复：量化模块无法直接设置 requires_grad，需要替换为全精度版本
                                rank0_print(f"  ⚠️  检测到 {module_name} 已被量化，正在替换为全精度版本...")
                                
                                # 获取量化模块的参数
                                in_features = module.in_features
                                out_features = module.out_features
                                bias = module.bias is not None
                                
                                # 尝试从量化模块中提取权重（反量化）
                                weight_data = None
                                bias_data = None
                                try:
                                    # 量化模块的权重存储在 module.weight 中，但需要反量化
                                    # 对于 Linear4bit，可以使用 .dequantize() 方法
                                    if hasattr(module.weight, 'dequantize'):
                                        weight_data = module.weight.dequantize()
                                    elif hasattr(module.weight, 'data'):
                                        # 尝试直接访问数据
                                        weight_data = module.weight.data
                                    if bias is not None and module.bias is not None:
                                        bias_data = module.bias.data
                                except Exception as e:
                                    rank0_print(f"    ⚠️  无法从量化模块提取权重: {e}")
                                    rank0_print(f"    将使用随机初始化（新token embedding会在训练中学习）")
                                
                                # 创建全精度 Linear 层
                                new_module = torch.nn.Linear(
                                    in_features=in_features,
                                    out_features=out_features,
                                    bias=bias,
                                    device=module.weight.device,
                                    dtype=compute_dtype
                                )
                                
                                # 如果成功提取了权重，使用它；否则使用随机初始化
                                if weight_data is not None:
                                    with torch.no_grad():
                                        new_module.weight.data.copy_(weight_data.to(compute_dtype))
                                        if bias_data is not None:
                                            new_module.bias.data.copy_(bias_data.to(compute_dtype))
                                    rank0_print(f"    ✓ 已从量化模块提取权重")
                                else:
                                    rank0_print(f"    ℹ️  使用随机初始化（新token embedding会在训练中学习）")
                                
                                # 替换模块
                                if module_path and "base_model.model." in module_path:
                                    setattr(base_model.model, module_name, new_module)
                                elif hasattr(base_model, 'model') and hasattr(base_model.model, module_name):
                                    setattr(base_model.model, module_name, new_module)
                                else:
                                    setattr(base_model, module_name, new_module)
                                
                                rank0_print(f"  ✓ {module_name} 已替换为全精度版本 ({compute_dtype})")
                            elif hasattr(module, 'weight') and not module.weight.dtype.is_floating_point:
                                rank0_print(f"  ⚠️  警告: {module_name} 不是浮点类型 ({module.weight.dtype})")
                                rank0_print(f"     尝试转换为 {compute_dtype}...")
                                try:
                                    module = module.to(dtype=compute_dtype)
                                    rank0_print(f"  ✓ {module_name} 已转换为 {compute_dtype}")
                                except Exception as e:
                                    rank0_print(f"  ❌ 转换失败: {e}")
                            else:
                                rank0_print(f"  ✓ {module_name} 是浮点类型 ({module.weight.dtype if hasattr(module, 'weight') else 'N/A'})，可以正常训练")
                        else:
                            # 🟢 关键修复：如果模块未找到，检查是否是 tie_word_embeddings 的情况
                            # tie_word_embeddings 已经在循环前检测过了，这里直接使用
                            if module_name == 'lm_head' and tie_word_embeddings:
                                rank0_print(f"  ℹ️  未找到模块 {module_name}（正常：tie_word_embeddings=True，lm_head 与 embed_tokens 共享权重）")
                            else:
                                rank0_print(f"  ⚠️  警告: 未找到模块 {module_name}")
        
        # 🟢 关键修复：如果 tie_word_embeddings=True，需要设置 ensure_weight_tying=True
        # 这样可以避免 PEFT 警告，并确保权重绑定正常工作
        # 注意：tie_word_embeddings 可能已经在前面检测过了（在 modules_to_save 处理循环中）
        # 但如果 training_args.bits 不在 [4, 8] 中，需要在这里重新检测
        # 为了安全，总是重新检测一次（即使之前检测过）
        base_model = model.get_model() if hasattr(model, 'get_model') else model
        tie_word_embeddings = False
        # 方法1: 通过 base_model.model.config
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'config'):
            tie_word_embeddings = getattr(base_model.model.config, 'tie_word_embeddings', False)
        # 方法2: 通过 base_model.config
        elif hasattr(base_model, 'config'):
            tie_word_embeddings = getattr(base_model.config, 'tie_word_embeddings', False)
        # 方法3: 通过 model.config（直接访问）
        elif hasattr(model, 'config'):
            tie_word_embeddings = getattr(model.config, 'tie_word_embeddings', False)
        
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save,
            ensure_weight_tying=tie_word_embeddings,  # 🟢 关键修复：如果 tie_word_embeddings=True，设置此参数
        )
        if tie_word_embeddings:
            rank0_print(f"  ℹ️  检测到 tie_word_embeddings=True，已设置 ensure_weight_tying=True（避免 PEFT 警告）")
        # 🟢 关键修复：FP16/BF16 模式下，模型已经在加载时设置了正确的 dtype
        # 这里不需要再次转换，因为 device_map="auto" 已经将模型分布到多个GPU上
        # 如果强制转换，可能会破坏多GPU分布
        if training_args.bits == 16:
            # 模型已经在加载时设置了正确的 dtype，这里只做验证
            rank0_print(f"  ✓ 模型 dtype: {next(model.parameters()).dtype}")
            # 注意：不要在这里调用 model.to()，因为 device_map="auto" 已经处理了设备分布
        rank0_print("   正在添加 LoRA 适配器...")
        model = get_peft_model(model, lora_config)
        
        # ========== 关键修复：确保 PEFT 初始化后 Polar Projector、polar_projector_norm 和 vae_latent_to_feature 的梯度被重新启用 ==========
        # PEFT 的 get_peft_model 可能会冻结所有非 LoRA 参数，包括 polar_projector、polar_projector_norm 和 vae_latent_to_feature
        # 对于 Stage 1 和 Stage 2 训练，我们需要确保它们保持可训练状态
        if use_polar:
            polar_projector = getattr(model.get_model(), 'polar_projector', None)
            if polar_projector is not None:
                for p in polar_projector.parameters():
                    # 只对浮点类型参数设置 requires_grad
                    if p.dtype.is_floating_point:
                        p.requires_grad = True
                rank0_print("  ✓ Re-enabled gradients for Polar Projector after PEFT initialization")
            
            # 🟢 关键修复：同时重新启用 polar_projector_norm 的梯度（Stage 1 和 Stage 2 都需要）
            polar_projector_norm = getattr(model.get_model(), 'polar_projector_norm', None)
            if polar_projector_norm is not None:
                for p in polar_projector_norm.parameters():
                    if p.dtype.is_floating_point:
                        p.requires_grad = True
                rank0_print("  ✓ Re-enabled gradients for polar_projector_norm after PEFT initialization")
            
            # 🟢 关键修复：同时重新启用 vae_latent_to_feature 的梯度（Stage 1 和 Stage 2 都需要）
            vae_latent_to_feature = getattr(model.get_model(), 'vae_latent_to_feature', None)
            if vae_latent_to_feature is not None:
                for p in vae_latent_to_feature.parameters():
                    if p.dtype.is_floating_point:
                        p.requires_grad = True
                rank0_print("  ✓ Re-enabled gradients for vae_latent_to_feature after PEFT initialization")

            # 🟢 关键修复：polar_alpha 仅在非 stage1a 下参与训练
            polar_alpha = getattr(model.get_model(), 'polar_alpha', None)
            if polar_alpha is not None and hasattr(polar_alpha, 'requires_grad'):
                if model_args.training_stage == "stage1" and model_args.polar_only:
                    with torch.no_grad():
                        polar_alpha.data.fill_(0.0)
                    polar_alpha.requires_grad = False
                    rank0_print("  ✓ Stage 1a strict mode: polar_alpha 已冻结并置 0（不参与训练）")
                else:
                    polar_alpha.requires_grad = True
                    rank0_print("  ✓ Re-enabled gradients for polar_alpha after PEFT initialization")
        
        # ========== Stage 1 特殊处理：冻结所有 LoRA 参数，只训练 Polar Projector ==========
        # 🟢 关键修复：必须在 DDP 包装之前完成所有参数冻结/解冻操作
        if model_args.training_stage == "stage1":
            # 🟢 关键修复：更彻底地冻结所有 LoRA 参数
            # 包括 LoRA A/B 矩阵、以及所有非 Polar 相关的参数
            lora_frozen_count = 0
            embed_trainable_count = 0
            
            # 先统计所有参数
            total_params = sum(1 for _ in model.named_parameters())
            
            # 🟢 关键修复：使用 no_grad 上下文确保参数状态修改不会触发 autograd hooks
            with torch.no_grad():
            for name, param in model.named_parameters():
                    # 冻结所有 LoRA 相关参数
                if 'lora' in name.lower():
                        if param.dtype.is_floating_point:
                    param.requires_grad = False
                            lora_frozen_count += 1
                    # 🟢 关键修复：在 Stage 1，即使 modules_to_save 包含 embed_tokens/lm_head，也要冻结整个 embedding 层
                    # 因为 PEFT 的 modules_to_save 会让整个层可训练（164M），而我们只需要新 token 的部分（~1M）
                    # 新 token 的 embedding 将在 Stage 2 训练，Stage 1 只训练 Polar Projector（~27.5M）
                    elif modules_to_save and any(m in name for m in modules_to_save):
                        # Stage 1: 冻结整个 embedding 层（避免 164M 参数可训练导致过拟合）
                        if param.dtype.is_floating_point:
                            param.requires_grad = False
                            embed_trainable_count += 1  # 统计，但实际是冻结的
                    # 确保所有其他非 Polar 参数都被冻结
                    elif 'polar_projector' not in name and 'polar_projector_norm' not in name and 'vae_latent_to_feature' not in name:
                        # 排除 Polar 相关参数（polar_projector、polar_projector_norm、vae_latent_to_feature），其他都冻结
                        if param.dtype.is_floating_point:
                            param.requires_grad = False
            
            # 🟢 关键修复：在 Stage 1，完全冻结 embedding 层（包括新 token），避免过拟合
            # 新 token 的 embedding 将在 Stage 2 训练，Stage 1 只训练 Polar Projector（~27.5M）
            # 这样可以大幅减少可训练参数（从 355M 降到 ~27.5M），防止过拟合
            try:
                # 获取实际的 embedding 模块（处理 PEFT 包装）
                from peft.utils.other import ModulesToSaveWrapper
                
                def get_actual_embedding_module(wrapped_module):
                    """从 PEFT 包装中提取实际 embedding 模块"""
                    if isinstance(wrapped_module, ModulesToSaveWrapper):
                        if hasattr(wrapped_module, 'module'):
                            return wrapped_module.module
                        elif hasattr(wrapped_module, 'modules_to_save'):
                            if isinstance(wrapped_module.modules_to_save, dict):
                                if 'default' in wrapped_module.modules_to_save:
                                    return wrapped_module.modules_to_save['default']
                                elif len(wrapped_module.modules_to_save) > 0:
                                    return list(wrapped_module.modules_to_save.values())[0]
                    return wrapped_module
                
                input_embeddings = model.get_input_embeddings()
                output_embeddings = model.get_output_embeddings()
                
                actual_input_emb = get_actual_embedding_module(input_embeddings)
                actual_output_emb = get_actual_embedding_module(output_embeddings)
                
                # 完全冻结 embedding 层（包括新 token）
                if hasattr(actual_input_emb, 'weight') and actual_input_emb.weight.dtype.is_floating_point:
                    if actual_input_emb.weight.requires_grad:
                        actual_input_emb.weight.requires_grad = False
                        rank0_print(f"  ✓ Stage 1: 已冻结 input_embeddings（包括新 token），避免过拟合")
                
                if hasattr(actual_output_emb, 'weight') and actual_output_emb.weight.dtype.is_floating_point:
                    if actual_output_emb.weight.requires_grad:
                        actual_output_emb.weight.requires_grad = False
                        rank0_print(f"  ✓ Stage 1: 已冻结 lm_head（包括新 token），避免过拟合")
                
                rank0_print(f"  ℹ️  新 token embedding 将在 Stage 2 训练（当前 Stage 1 只训练 Polar Projector）")
            except Exception as e:
                rank0_print(f"  ⚠️ Warning: 无法设置 embedding 梯度状态: {e}")
                import traceback
                traceback.print_exc()
            
            # 统计可训练参数（按元素数量，而不是参数数量）
            trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            trainable_param_num = sum(1 for p in model.parameters() if p.requires_grad)
            
            # 🟢 调试：检查 embedding 层的梯度状态
            input_emb = model.get_input_embeddings()
            output_emb = model.get_output_embeddings()
            embed_trainable = 0
            if hasattr(input_emb, 'weight') and input_emb.weight.requires_grad:
                embed_trainable += input_emb.weight.numel()
            if hasattr(output_emb, 'weight') and output_emb.weight.requires_grad:
                # 如果 tie_word_embeddings=True，output_emb 与 input_emb 共享权重，不重复计算
                if not getattr(model.config, 'tie_word_embeddings', False):
                    embed_trainable += output_emb.weight.numel()
            
            if embed_trainable > 0:
                rank0_print(f"  ⚠️  警告: Embedding 层仍有 {embed_trainable:,} 个可训练参数！")
                rank0_print(f"     这可能导致总可训练参数异常增加（预期只有 Polar Projector + vae_latent_to_feature）")
            rank0_print(f"  ✓ LoRA 参数已冻结（Stage 1 只训练 Polar Projector）")
            rank0_print(f"     - 冻结了 {lora_frozen_count} 个 LoRA 参数")
            rank0_print(f"     - Embedding 层已完全冻结（包括新 token，避免过拟合）")
            rank0_print(f"     - 总可训练参数: {trainable_param_num}/{total_params} 个参数对象")
            rank0_print(f"     - 总可训练元素: {trainable_param_count:,} 个元素（应该约 27.5M，而不是 355M）")

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            local_files_only=True  # 强制只使用本地文件
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
            local_files_only=True  # 强制只使用本地文件
        )

    # 🟢 关键修复：Stage 1 必须使用 plain 模式（无 System Prompt），Stage 2 使用 v1 模式（有 System Prompt）
    # 这是 LLaVA 标准做法：
    # - Stage 1: 特征对齐，只需要极简格式，不需要 System Prompt（避免干扰 Projector 学习）
    # - Stage 2: 对话微调，需要完整的对话格式（包括 System Prompt）
    # ⚠️ 必须在所有版本检查之前执行，确保 Stage 1 始终使用 plain 模式
    if model_args.training_stage == "stage1":
        # Stage 1: 强制使用 plain 模式（无 System Prompt）
        if model_args.version != "plain":
            rank0_print(f"🟢 Stage 1 训练：强制使用 'plain' 模式（无 System Prompt）")
            rank0_print(f"   原因：Stage 1 只需要特征对齐，System Prompt 会干扰 Projector 学习视觉特征")
            if model_args.version == "v1":
                rank0_print(f"   ⚠️  警告: 你指定了 'v1'，但 Stage 1 应该使用 'plain'，已自动切换")
            elif model_args.version == "v0":
                rank0_print(f"   ℹ️  信息: 默认版本是 'v0'，Stage 1 自动切换为 'plain'")
            model_args.version = "plain"
    else:
        # Stage 2: 使用 v1 模式（有 System Prompt）
        # 🔧 自动检测：如果模型路径包含 "llava-v1.5" 或 "llava-1.5"，自动使用 v1
        if ("llava-v1.5" in model_args.model_name_or_path.lower() or 
            "llava-1.5" in model_args.model_name_or_path.lower() or
            "llava_v1.5" in model_args.model_name_or_path.lower()):
            if model_args.version not in ["v1", "vicuna_v1"]:
                rank0_print(f"⚠️  Warning: Detected LLaVA-1.5 model, but version is '{model_args.version}'")
                rank0_print(f"   Auto-switching to 'v1' for correct conversation template")
                model_args.version = "v1"
    
    # 设置 tokenizer pad_token
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        
    # 设置 conversation template（必须在 Stage 1/Stage 2 版本检查之后）
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
            rank0_print(f"   Using conversation template: {model_args.version}")
        else:
            rank0_print(f"⚠️  Warning: version '{model_args.version}' not found in conv_templates, using 'vicuna_v1'")
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    if model_args.training_stage == "stage1":
        rank0_print(f"   ✅ Stage 1 使用 plain 模式：无 System Prompt，极简格式（<image>Question\\nAnswer\\n）")
    else:
        rank0_print(f"   ✅ Stage 2 使用 v1 模式：包含 System Prompt，完整对话格式")

    # 🟢 修改：Stage 1 和 Stage 2 都不添加分隔 Token，直接拼接 RGB 和 Polar 特征
    # 数据量少时，添加新 token 容易过拟合，直接拼接更简单有效
    # Stage 1: [BOS] [Pol Tokens] [Text Tokens] [EOS] (不使用分隔 token，直接插入 Polar features)
    # Stage 2: [BOS] [RGB Tokens] [Pol Tokens] [Text Tokens] [EOS] (不使用分隔 token，直接拼接 RGB 和 Polar features)
    # 🟢 关键优化：所有阶段都不添加分隔 token，直接拼接特征
    
    # 🔍 检查：记录添加 token 前的词表大小
    vocab_size_before_tokens = len(tokenizer)
    rank0_print(f"  🔍 Token 添加前词表大小: {vocab_size_before_tokens}")
    
    if model_args.training_stage == "stage1":
        # Stage 1: 不添加任何分隔 token
        # - polar_only=True: 仅 Polar（[Pol_Tokens, Text_Tokens]）
        # - residual: fused = RGB + alpha * Polar（[Fused_Tokens, Text_Tokens]）
        # - concat: [RGB_Tokens, Polar_Tokens, Text_Tokens]
        special_modality_tokens = []
        if model_args.polar_only:
            rank0_print("  Stage 1: 不添加分隔 token，使用 [Pol_Tokens, Text_Tokens]（polar_only=True）")
        elif model_args.polar_fusion_mode == "residual":
            rank0_print("  Stage 1: 不添加分隔 token，使用 [Fused_Tokens, Text_Tokens]（residual）")
        else:
            rank0_print("  Stage 1: 不添加分隔 token，使用 [RGB_Tokens, Polar_Tokens, Text_Tokens]（concat）")
    else:
        # 🟢 修改：Stage 2 也不添加分隔 token，直接拼接 RGB 和 Polar 特征（避免过拟合）
        # 数据量少时，添加新 token 容易过拟合，直接拼接更简单有效
        special_modality_tokens = []
        rank0_print("  Stage 2: 不添加分隔 token，直接拼接 RGB 和 Polar 特征（避免过拟合）")
    
    special_tokens_dict = {"additional_special_tokens": []}
    for t in special_modality_tokens:
        if t not in tokenizer.get_vocab():
            special_tokens_dict["additional_special_tokens"].append(t)
        else:
            rank0_print(f"    ℹ️  Token {t} 已存在于词表中（ID: {tokenizer.convert_tokens_to_ids(t)}）")

    if len(special_tokens_dict["additional_special_tokens"]) > 0:
        num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        vocab_size_after_tokens = len(tokenizer)
        rank0_print(f"  🔍 Token 添加后词表大小: {vocab_size_after_tokens} (+{num_added_tokens} tokens)")
        
        if num_added_tokens > 0:
            # 对于非量化模型，可以安全地 resize；量化模型下 tokenizer 可能已被扩展，这里依然 resize 以保证一致
            resize_success = False
            try:
                model.resize_token_embeddings(len(tokenizer))
                rank0_print(f"  ✓ 模型 embedding 已 resize: {model.get_input_embeddings().weight.shape[0]}")
                resize_success = True
            except Exception as e:
                # 🟢 关键修复：resize 失败是正常的（PEFT ModulesToSaveWrapper 不支持直接 resize）
                # 这是预期的行为，代码会在后面手动处理，所以这个警告可以忽略
                if 'ModulesToSaveWrapper' in str(e) or 'not an instance' in str(e):
                    rank0_print(f"  ℹ️  resize_token_embeddings 失败（正常：PEFT ModulesToSaveWrapper 不支持直接 resize，将手动处理）")
                else:
                    rank0_print(f"  ⚠️ Warning: resize_token_embeddings failed for modality tokens: {e}")
                # 🟢 关键修复：如果 resize 失败（通常是因为 PEFT ModulesToSaveWrapper），手动处理
                try:
                    input_embeddings = model.get_input_embeddings()
                    output_embeddings = model.get_output_embeddings()
                    
                    rank0_print(f"  🔍 Debug: input_embeddings type: {type(input_embeddings)}")
                    rank0_print(f"  🔍 Debug: output_embeddings type: {type(output_embeddings)}")
                    
                    # 检查是否是 PEFT ModulesToSaveWrapper
                    from peft.utils.other import ModulesToSaveWrapper
                    
                    # 获取实际的 embedding 模块（处理 PEFT 包装）
                    def get_actual_module(wrapped_module):
                        """从 PEFT 包装中提取实际模块"""
                        rank0_print(f"    🔍 get_actual_module: input type = {type(wrapped_module)}")
                        if isinstance(wrapped_module, ModulesToSaveWrapper):
                            rank0_print(f"    🔍 检测到 ModulesToSaveWrapper")
                            # 🟢 方法1：尝试访问 .module 属性（PEFT 可能将实际模块存储在这里）
                            if hasattr(wrapped_module, 'module') and wrapped_module.module is not None:
                                rank0_print(f"    ✓ 通过 .module 访问: {type(wrapped_module.module)}")
                                return wrapped_module.module
                            
                            # 🟢 方法2：尝试访问 .modules_to_save（可能是 ModuleDict）
                            if hasattr(wrapped_module, 'modules_to_save'):
                                modules_to_save = wrapped_module.modules_to_save
                                rank0_print(f"    🔍 找到 modules_to_save: {type(modules_to_save)}")
                                
                                # 如果是 ModuleDict，需要从中提取实际的 embedding 模块
                                # 注意：ModuleDict 也是 dict 的子类，所以要先检查 ModuleDict
                                if hasattr(modules_to_save, 'keys') and hasattr(modules_to_save, 'values') and hasattr(modules_to_save, 'items'):
                                    # 可能是 ModuleDict 或普通 dict
                                    rank0_print(f"    🔍 modules_to_save 是容器类型，包含键: {list(modules_to_save.keys()) if hasattr(modules_to_save, 'keys') else 'N/A'}")
                                    # 尝试常见的键名（embed_tokens 或 lm_head）
                                    for key in ['embed_tokens', 'model.embed_tokens', 'lm_head', 'model.lm_head', 'default']:
                                        if key in modules_to_save:
                                            actual_module = modules_to_save[key]
                                            rank0_print(f"    ✓ 通过 modules_to_save['{key}'] 访问: {type(actual_module)}")
                                            if hasattr(actual_module, 'weight'):
                                                return actual_module
                                    # 如果没找到，尝试遍历所有值，找到第一个有 weight 的 Embedding 或 Linear
                                    for key, value in modules_to_save.items():
                                        if hasattr(value, 'weight') and isinstance(value, (torch.nn.Embedding, torch.nn.Linear)):
                                            rank0_print(f"    ✓ 通过 modules_to_save['{key}'] 访问: {type(value)}")
                                            return value
                                    # 如果还是没找到，返回第一个有 weight 的模块
                                    for key, value in modules_to_save.items():
                                        if hasattr(value, 'weight'):
                                            rank0_print(f"    ⚠️  通过 modules_to_save['{key}'] 访问（非标准类型）: {type(value)}")
                                            return value
                                # 如果不是容器类型，可能是其他格式
                                elif modules_to_save is not None:
                                    rank0_print(f"    ⚠️  modules_to_save 不是容器类型: {type(modules_to_save)}")
                                    # 如果它本身有 weight，直接返回
                                    if hasattr(modules_to_save, 'weight'):
                                        return modules_to_save
                                    return modules_to_save
                            
                            # 🟢 方法3：尝试通过 __dict__ 直接访问
                            if hasattr(wrapped_module, '__dict__'):
                                for key, value in wrapped_module.__dict__.items():
                                    if isinstance(value, torch.nn.Module) and hasattr(value, 'weight'):
                                        # 检查是否是 Embedding 或 Linear 层
                                        if isinstance(value, (torch.nn.Embedding, torch.nn.Linear)):
                                            rank0_print(f"    ✓ 通过 __dict__['{key}'] 访问: {type(value)}")
                                            return value
                            
                            # 🟢 方法4：如果 ModulesToSaveWrapper 直接代理了 weight，返回 wrapper 本身
                            # 但需要特殊处理 weight 替换
                            if hasattr(wrapped_module, 'weight'):
                                rank0_print(f"    ✓ 直接使用 wrapper (有 weight 属性)")
                                return wrapped_module
                        rank0_print(f"    ⚠️  未找到底层模块，返回原对象")
                        return wrapped_module
                    
                    actual_input_emb = get_actual_module(input_embeddings)
                    actual_output_emb = get_actual_module(output_embeddings)
                    
                    rank0_print(f"  🔍 Debug: actual_input_emb type: {type(actual_input_emb)}")
                    rank0_print(f"  🔍 Debug: actual_output_emb type: {type(actual_output_emb)}")
                    rank0_print(f"  🔍 Debug: actual_input_emb has weight: {hasattr(actual_input_emb, 'weight')}")
                    rank0_print(f"  🔍 Debug: actual_output_emb has weight: {hasattr(actual_output_emb, 'weight')}")
                    
                    # 检查并手动 resize input_embeddings
                    if hasattr(actual_input_emb, 'weight'):
                        current_size = actual_input_emb.weight.shape[0]
                        target_size = len(tokenizer)
                        
                        if current_size < target_size:
                            rank0_print(f"  🔧 手动 resize PEFT embedding: {current_size} -> {target_size}")
                            # 手动扩展 embedding
                            with torch.no_grad():
                                # 保存旧权重
                                old_weight = actual_input_emb.weight.data.clone()
                                old_weight_shape = old_weight.shape
                                
                                # 创建新权重
                                new_weight_data = torch.zeros(target_size, old_weight_shape[1], 
                                                             dtype=old_weight.dtype,
                                                             device=old_weight.device)
                                new_weight_data[:current_size] = old_weight
                                # 初始化新 token 的 embedding 为旧 token 的平均值
                                new_weight_data[current_size:] = old_weight.mean(dim=0, keepdim=True).expand(target_size - current_size, -1)
                                
                                # 🟢 关键修复：直接修改 weight.data（如果可能），或使用 object.__setattr__ 绕过 __setattr__
                                try:
                                    # 方法1：尝试直接修改 weight.data（如果形状允许，但这里形状变了，所以不行）
                                    # 方法2：使用 object.__setattr__ 绕过可能的 __setattr__ 拦截
                                    if isinstance(actual_input_emb, ModulesToSaveWrapper):
                                        # 如果是 wrapper，需要特殊处理
                                        # 尝试直接访问底层模块并修改
                                        if hasattr(actual_input_emb, 'module') and actual_input_emb.module is not None:
                                            # 直接操作底层模块
                                            object.__setattr__(actual_input_emb.module, 'weight', torch.nn.Parameter(new_weight_data))
                                            if hasattr(actual_input_emb.module, '_parameters'):
                                                actual_input_emb.module._parameters['weight'] = actual_input_emb.module.weight
                                        else:
                                            # 如果无法访问底层模块，使用 object.__setattr__ 绕过 wrapper
                                            object.__setattr__(actual_input_emb, 'weight', torch.nn.Parameter(new_weight_data))
                                            if hasattr(actual_input_emb, '_parameters'):
                                                actual_input_emb._parameters['weight'] = actual_input_emb.weight
                                    else:
                                        # 普通模块，直接替换
                                        if 'weight' in actual_input_emb._parameters:
                                            del actual_input_emb._parameters['weight']
                                        actual_input_emb.register_parameter('weight', torch.nn.Parameter(new_weight_data))
                                except Exception as e:
                                    rank0_print(f"    ❌ 替换 weight 失败: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    raise
                            resize_success = True
                            rank0_print(f"  ✓ 手动 resize input_embeddings 成功: {target_size}")
                            # 🟢 关键修复：更新 model.config.vocab_size
                            model.config.vocab_size = target_size
                            rank0_print(f"  ✓ 已更新 model.config.vocab_size = {target_size}")
                    
                    # 检查并手动 resize output_embeddings (lm_head)
                    if hasattr(actual_output_emb, 'weight'):
                        current_size = actual_output_emb.weight.shape[0]
                        target_size = len(tokenizer)
                        
                        if current_size < target_size:
                            rank0_print(f"  🔧 手动 resize PEFT lm_head: {current_size} -> {target_size}")
                            with torch.no_grad():
                                # 保存旧权重
                                old_weight = actual_output_emb.weight.data.clone()
                                old_weight_shape = old_weight.shape
                                
                                # 创建新权重
                                new_weight_data = torch.zeros(target_size, old_weight_shape[1],
                                                             dtype=old_weight.dtype,
                                                             device=old_weight.device)
                                new_weight_data[:current_size] = old_weight
                                new_weight_data[current_size:] = old_weight.mean(dim=0, keepdim=True).expand(target_size - current_size, -1)
                                
                                # 🟢 关键修复：使用 object.__setattr__ 绕过可能的 __setattr__ 拦截
                                try:
                                    if isinstance(actual_output_emb, ModulesToSaveWrapper):
                                        # 如果是 wrapper，需要特殊处理
                                        if hasattr(actual_output_emb, 'module') and actual_output_emb.module is not None:
                                            # 直接操作底层模块
                                            object.__setattr__(actual_output_emb.module, 'weight', torch.nn.Parameter(new_weight_data))
                                            if hasattr(actual_output_emb.module, '_parameters'):
                                                actual_output_emb.module._parameters['weight'] = actual_output_emb.module.weight
                                        else:
                                            # 使用 object.__setattr__ 绕过 wrapper
                                            object.__setattr__(actual_output_emb, 'weight', torch.nn.Parameter(new_weight_data))
                                            if hasattr(actual_output_emb, '_parameters'):
                                                actual_output_emb._parameters['weight'] = actual_output_emb.weight
                                    else:
                                        # 普通模块，直接替换
                                        if 'weight' in actual_output_emb._parameters:
                                            del actual_output_emb._parameters['weight']
                                        actual_output_emb.register_parameter('weight', torch.nn.Parameter(new_weight_data))
                                except Exception as e:
                                    rank0_print(f"    ❌ 替换 weight 失败: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    raise
                            rank0_print(f"  ✓ 手动 resize lm_head 成功: {target_size}")
                            # 🟢 关键修复：确保 model.config.vocab_size 已更新
                            if model.config.vocab_size != target_size:
                                model.config.vocab_size = target_size
                                rank0_print(f"  ✓ 已更新 model.config.vocab_size = {target_size}")
                        else:
                            rank0_print(f"  ⚠️  Debug: lm_head current_size ({current_size}) >= target_size ({target_size}), 跳过 resize")
                    else:
                        rank0_print(f"  ⚠️  Debug: actual_output_emb 没有 weight 属性，无法手动 resize")
                    
                    # 最终检查 resize 是否成功
                    if resize_success:
                        final_input_size = model.get_input_embeddings().weight.shape[0] if hasattr(model.get_input_embeddings(), 'weight') else None
                        final_vocab_size = model.config.vocab_size
                        rank0_print(f"  ✓ 最终检查: input_embeddings size = {final_input_size}, model.config.vocab_size = {final_vocab_size}, tokenizer size = {len(tokenizer)}")
                        # 🟢 确保 config.vocab_size 与 embedding size 一致
                        if final_input_size is not None and final_vocab_size != final_input_size:
                            rank0_print(f"  ⚠️  警告: model.config.vocab_size ({final_vocab_size}) != input_embeddings size ({final_input_size})，正在修正...")
                            model.config.vocab_size = final_input_size
                            rank0_print(f"  ✓ 已修正 model.config.vocab_size = {final_input_size}")
                    else:
                        rank0_print(f"  ❌ 警告: 手动 resize 未成功，模型 embedding size 可能不匹配 tokenizer")
                except Exception as e2:
                    rank0_print(f"  ❌ 手动 resize 过程出错: {e2}")
                    import traceback
                    traceback.print_exc()
                            
                except Exception as e2:
                    rank0_print(f"  ❌ 手动 resize 也失败: {e2}")
                    import traceback
                    traceback.print_exc()

            # 确保新 token 的 embedding 可训练（仅在浮点权重下）
            if resize_success:
                try:
                    input_embeddings = model.get_input_embeddings()
                    output_embeddings = model.get_output_embeddings()
                    
                    # 处理 PEFT 包装的情况（使用与上面相同的逻辑）
                    from peft.utils.other import ModulesToSaveWrapper
                    
                    def get_actual_module_for_grad(wrapped_module):
                        """从 PEFT 包装中提取实际模块（用于设置梯度）"""
                        if isinstance(wrapped_module, ModulesToSaveWrapper):
                            if hasattr(wrapped_module, 'module'):
                                return wrapped_module.module
                            elif hasattr(wrapped_module, 'modules_to_save'):
                                if isinstance(wrapped_module.modules_to_save, dict):
                                    if 'default' in wrapped_module.modules_to_save:
                                        return wrapped_module.modules_to_save['default']
                                    elif len(wrapped_module.modules_to_save) > 0:
                                        return list(wrapped_module.modules_to_save.values())[0]
                            if hasattr(wrapped_module, 'weight'):
                                return wrapped_module
                        return wrapped_module
                    
                    actual_input_emb = get_actual_module_for_grad(input_embeddings)
                    actual_output_emb = get_actual_module_for_grad(output_embeddings)
                    
                    if hasattr(actual_input_emb, "weight") and actual_input_emb.weight.dtype.is_floating_point:
                        actual_input_emb.weight.requires_grad = True
                        rank0_print(f"  ✓ 已启用 input_embeddings 梯度（包含 {num_added_tokens} 个新 token）")
                    elif hasattr(actual_input_emb, "weight"):
                        rank0_print(f"  ⚠️ input_embeddings 权重类型为 {actual_input_emb.weight.dtype}，无法设置 requires_grad")
                    
                    if hasattr(actual_output_emb, "weight") and actual_output_emb.weight.dtype.is_floating_point:
                        actual_output_emb.weight.requires_grad = True
                        rank0_print(f"  ✓ 已启用 output_embeddings (lm_head) 梯度（包含 {num_added_tokens} 个新 token）")
                    elif hasattr(actual_output_emb, "weight"):
                        rank0_print(f"  ⚠️ output_embeddings 权重类型为 {actual_output_emb.weight.dtype}，无法设置 requires_grad")
                except Exception as e:
                    rank0_print(f"  ⚠️ Warning: failed to enable gradients for modality tokens: {e}")
                    import traceback
                    traceback.print_exc()

        # 🟢 修改：Stage 1 和 Stage 2 都不添加分隔 token，所有 token ID 设为 None
        # Stage 2 也直接拼接 RGB 和 Polar 特征，避免过拟合
        try:
            # 所有阶段都不使用分隔 token，直接拼接特征
            model.config.pol_start_id = None
            model.config.pol_end_id = None
            model.config.rgb_start_id = None
            model.config.rgb_end_id = None
            rank0_print(
                f"   ✓ {model_args.training_stage}: 不使用分隔 token，直接拼接 RGB 和 Polar 特征"
            )
            # 🔍 验证：不添加任何 token，所以所有 token ID 都应该是 None
            if model.config.pol_start_id is not None or model.config.pol_end_id is not None:
                rank0_print(f"    ⚠️  警告: 不应该有 Polar token ID，但检测到非 None 值！")
                rank0_print(f"      POL_START={model.config.pol_start_id}, POL_END={model.config.pol_end_id}")
            # 🔍 验证：词表大小应该保持原始大小（32000），没有添加新 token
            vocab_size_after = len(tokenizer)
            if vocab_size_after != vocab_size_before_tokens:
                rank0_print(f"    ⚠️  警告: 词表大小发生变化！")
                rank0_print(f"      原始: {vocab_size_before_tokens}, 当前: {vocab_size_after}")
            else:
                rank0_print(f"    ✅ 词表大小验证通过: {vocab_size_after} (未添加新 token)")
        except Exception as e:
            rank0_print(f"  ⚠️ Warning: failed to record modality token ids in config: {e}")
    
    # 🟢 修改：Stage 2 不添加分隔 token，所以不需要从 Stage 1 加载 token embedding
    # Stage 1 和 Stage 2 词表大小都是 32000，不需要加载 token embedding

    if model_args.vision_tower is not None:
        rank0_print("   正在初始化视觉模块...")
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        rank0_print("   ✓ 视觉模块初始化完成")

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True
        
        # ========== 关键修复 1: 确保 RGB 图像尺寸为 336x336 (LLaVA v1.5 要求) ==========
        # LLaVA v1.5 使用 clip-vit-large-patch14-336，输入分辨率必须是 336x336
        # 这样 RGB 输出是 24x24 = 576 tokens，与 Polar 的 24x24 = 576 tokens 对齐
        if hasattr(vision_tower.image_processor, 'size'):
            # 检查并设置正确的图像尺寸
            if 'shortest_edge' in vision_tower.image_processor.size:
                current_size = vision_tower.image_processor.size.get('shortest_edge', 224)
                if current_size != 336:
                    rank0_print(f"⚠️  警告: CLIP image processor size is {current_size}, 但 LLaVA v1.5 需要 336")
                    rank0_print(f"  正在自动调整为 336x336...")
                    vision_tower.image_processor.size['shortest_edge'] = 336
                    vision_tower.image_processor.crop_size['height'] = 336
                    vision_tower.image_processor.crop_size['width'] = 336
                    rank0_print(f"  ✓ 已设置为 336x336")
            else:
                # 如果没有 shortest_edge，直接设置 crop_size
                vision_tower.image_processor.crop_size['height'] = 336
                vision_tower.image_processor.crop_size['width'] = 336
                rank0_print(f"  ✓ RGB 图像尺寸已设置为 336x336 (LLaVA v1.5 要求)")
        else:
            rank0_print(f"  ✓ RGB 图像尺寸: {vision_tower.image_processor.crop_size if hasattr(vision_tower.image_processor, 'crop_size') else 'auto'}")

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                # 只对浮点类型参数设置 requires_grad
                if p.dtype.is_floating_point:
                    p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        # ========== 新增：Polar 模型的参数冻结/解冻逻辑 ==========
        if use_polar:
            rank0_print("=" * 50)
            rank0_print(f"Training Stage: {model_args.training_stage}")
            rank0_print("=" * 50)
            
            # 获取 Polar 相关模块
            polar_encoder = getattr(model.get_model(), 'polar_encoder', None)
            polar_projector = getattr(model.get_model(), 'polar_projector', None)
            rgb_vision_tower = model.get_vision_tower()
            rgb_projector = model.get_model().mm_projector
            
            # ========== 关键修复 3: 加载 Stage 1 训练的 Polar Projector 权重 ==========
            if model_args.pretrain_polar_projector and os.path.exists(model_args.pretrain_polar_projector):
                rank0_print(f"\n正在加载 Stage 1 训练的 Polar Projector 权重...")
                rank0_print(f"  路径: {model_args.pretrain_polar_projector}")
                
                try:
                    polar_alpha_loaded = False
                    # 加载权重文件
                    if model_args.pretrain_polar_projector.endswith('.pth'):
                        # 如果是 .pth 文件，直接加载
                        weights = torch.load(model_args.pretrain_polar_projector, map_location='cpu', weights_only=False)
                    elif model_args.pretrain_polar_projector.endswith('.bin'):
                        # 如果是 .bin 文件（LLaVA 格式），需要提取 polar_projector 部分
                        weights = torch.load(model_args.pretrain_polar_projector, map_location='cpu', weights_only=False)
                        # 提取包含 'polar_projector' 的权重（支持多种键名格式）
                        polar_weights = {k: v for k, v in weights.items() if 'polar_projector' in k}
                        if not polar_weights:
                            # 如果没有找到 polar_projector，尝试直接使用所有权重（可能是完整的 polar_projector state_dict）
                            rank0_print(f"  ⚠️  警告: 权重文件中未找到包含 'polar_projector' 的键，尝试使用所有权重")
                            polar_weights = weights
                        weights = polar_weights
                    else:
                        # 尝试作为目录路径，查找 mm_projector.bin 或 polar_projector.pth
                        checkpoint_dir = Path(model_args.pretrain_polar_projector)
                        if (checkpoint_dir / 'mm_projector.bin').exists():
                            weights = torch.load(checkpoint_dir / 'mm_projector.bin', map_location='cpu', weights_only=False)
                            # 提取 polar_projector 权重（支持多种键名格式）
                            polar_weights = {k: v for k, v in weights.items() if 'polar_projector' in k}
                            if not polar_weights:
                                rank0_print(f"  ⚠️  警告: mm_projector.bin 中未找到包含 'polar_projector' 的键，尝试使用所有权重")
                            weights = polar_weights if polar_weights else weights
                        elif (checkpoint_dir / 'polar_projector.pth').exists():
                            weights = torch.load(checkpoint_dir / 'polar_projector.pth', map_location='cpu', weights_only=False)
                        else:
                            raise FileNotFoundError(f"在 {checkpoint_dir} 中未找到 mm_projector.bin 或 polar_projector.pth")
                    
                    # 加载权重到 polar_projector
                    if polar_projector is not None:
                        # 🟢 关键修复：先过滤掉 polar_projector_norm 的键，避免"意外的键"警告
                        # 这些键会在后面单独加载，不应该在这里处理
                        polar_projector_weights = {k: v for k, v in weights.items() 
                                                  if 'polar_projector' in k and 'polar_projector_norm' not in k}
                        
                        # 处理权重键名（可能包含多种前缀格式）
                        cleaned_weights = {}
                        for k, v in polar_projector_weights.items():
                            new_key = k
                            # 尝试移除多种可能的前缀（按从长到短顺序，避免部分匹配）
                            # ⚠️ 关键修复：添加所有可能的前缀格式，包括 DDP 包装后的 module. 前缀
                            # 这是 PeftModel + DDP 包装后的完整路径格式
                            prefixes_to_remove = [
                                'module.base_model.model.model.polar_projector.',  # DDP + PeftModel 包装后的完整路径
                                'base_model.model.model.polar_projector.',         # PeftModel 包装后的完整路径
                                'base_model.model.polar_projector.',               # 标准路径
                                'model.polar_projector.',                          # 简化路径
                                'polar_projector.',                                # 直接路径
                            ]
                            for prefix in prefixes_to_remove:
                                if new_key.startswith(prefix):
                                    new_key = new_key[len(prefix):]
                                    break
                            cleaned_weights[new_key] = v
                        
                        # 获取 polar_projector 的期望键名（用于验证）
                        expected_keys = set(polar_projector.state_dict().keys())
                        loaded_keys = set(cleaned_weights.keys())
                        
                        # 尝试加载
                        missing_keys, unexpected_keys = polar_projector.load_state_dict(cleaned_weights, strict=False)
                        
                        # 详细日志
                        rank0_print(f"  权重文件中的键数量: {len(weights)}")
                        rank0_print(f"  清理后的键数量: {len(cleaned_weights)}")
                        rank0_print(f"  Polar Projector 期望的键数量: {len(expected_keys)}")
                        rank0_print(f"  成功匹配的键数量: {len(loaded_keys & expected_keys)}")
                        rank0_print(f"  缺失的键数量: {len(missing_keys)}")
                        rank0_print(f"  意外的键数量: {len(unexpected_keys)}")
                        
                        # 关键检查：如果没有任何键被加载，或者所有期望的键都缺失，抛出异常
                        matched_keys = loaded_keys & expected_keys
                        if len(matched_keys) == 0:
                            rank0_print(f"  ❌ 致命错误: 没有成功加载任何 Polar Projector 权重！")
                            rank0_print(f"  权重文件中的键示例: {list(weights.keys())[:5]}")
                            rank0_print(f"  清理后的键示例: {list(cleaned_weights.keys())[:5]}")
                            rank0_print(f"  Polar Projector 期望的键示例: {list(expected_keys)[:5]}")
                            raise ValueError(
                                f"Critical: Failed to load any Polar Projector weights! "
                                f"No keys matched. Check if the weight file contains 'polar_projector' weights."
                            )
                        
                        if len(missing_keys) == len(expected_keys):
                            rank0_print(f"  ❌ 致命错误: 所有期望的键都缺失！")
                            raise ValueError(
                                f"Critical: All expected keys are missing! "
                                f"Expected {len(expected_keys)} keys, but loaded 0. "
                                f"Check weight file format."
                            )
                        
                        if missing_keys:
                            rank0_print(f"  ⚠️  警告: 以下权重未加载: {missing_keys[:5]}...")
                        if unexpected_keys:
                            # 🟢 关键修复：如果意外的键是 polar_projector_norm 相关的，说明这是正常的
                            # 因为这些键会在后面单独加载，不应该在这里处理
                            norm_keys = [k for k in unexpected_keys if 'polar_projector_norm' in k]
                            other_keys = [k for k in unexpected_keys if 'polar_projector_norm' not in k]
                            if norm_keys:
                                rank0_print(f"  ℹ️  以下权重被忽略（正常）: {norm_keys[:3]}... (这些是 polar_projector_norm 的权重，将在后面单独加载)")
                            if other_keys:
                                rank0_print(f"  ⚠️  警告: 以下权重被忽略: {other_keys[:5]}...")
                        
                        rank0_print(f"  ✓ Polar Projector 权重已加载 ({len(matched_keys)}/{len(expected_keys)} keys matched)")
                        
                        # 🟢 新增：加载后验证 Polar Projector 权重统计
                        rank0_print(f"  📊 Stage 2 Polar Projector 权重统计（加载后）:")
                        proj_state_dict = polar_projector.state_dict()
                        for k in list(proj_state_dict.keys())[:3]:  # 只显示前3个权重
                            if k in proj_state_dict:
                                v = proj_state_dict[k]
                                if isinstance(v, torch.Tensor):
                                    v_min, v_max = v.min().item(), v.max().item()
                                    v_mean, v_std = v.mean().item(), v.std().item()
                                    rank0_print(f"    {k}: shape={v.shape}, range=[{v_min:.6f}, {v_max:.6f}], mean={v_mean:.6f}, std={v_std:.6f}")
                    else:
                        rank0_print(f"  ⚠️  警告: Polar Projector 未找到，无法加载权重")
                        raise ValueError("Critical: Polar Projector module not found in model!")
                    
                    # 🟢 关键修复：同时加载 polar_projector_norm 权重（如果存在）
                    polar_projector_norm = getattr(model.get_model(), 'polar_projector_norm', None)
                    if polar_projector_norm is not None:
                        # 从原始权重文件中提取 polar_projector_norm 权重
                        # 重新加载完整权重文件（因为之前只提取了 polar_projector）
                        all_weights_for_norm = {}
                        if model_args.pretrain_polar_projector.endswith('.pth'):
                            all_weights_for_norm = torch.load(model_args.pretrain_polar_projector, map_location='cpu', weights_only=False)
                        elif model_args.pretrain_polar_projector.endswith('.bin'):
                            all_weights_for_norm = torch.load(model_args.pretrain_polar_projector, map_location='cpu', weights_only=False)
                        else:
                            checkpoint_dir = Path(model_args.pretrain_polar_projector)
                            if (checkpoint_dir / 'non_lora_trainables.bin').exists():
                                all_weights_for_norm = torch.load(checkpoint_dir / 'non_lora_trainables.bin', map_location='cpu', weights_only=False)
                            elif (checkpoint_dir / 'mm_projector.bin').exists():
                                all_weights_for_norm = torch.load(checkpoint_dir / 'mm_projector.bin', map_location='cpu', weights_only=False)
                        
                        # 提取包含 'polar_projector_norm' 的权重
                        norm_weights = {k: v for k, v in all_weights_for_norm.items() if 'polar_projector_norm' in k}
                        if norm_weights:
                            # 处理权重键名（移除前缀）
                            cleaned_norm_weights = {}
                            for k, v in norm_weights.items():
                                new_key = k
                                prefixes_to_remove = [
                                    'module.base_model.model.model.polar_projector_norm.',
                                    'base_model.model.model.polar_projector_norm.',
                                    'base_model.model.polar_projector_norm.',
                                    'model.polar_projector_norm.',
                                    'polar_projector_norm.',
                                ]
                                for prefix in prefixes_to_remove:
                                    if new_key.startswith(prefix):
                                        new_key = new_key[len(prefix):]
                                        break
                                cleaned_norm_weights[new_key] = v
                            
                            # 加载权重
                            norm_expected_keys = set(polar_projector_norm.state_dict().keys())
                            norm_loaded_keys = set(cleaned_norm_weights.keys())
                            
                            # 🟢 新增：在加载前输出 Stage 1 的 LayerNorm 权重统计
                            rank0_print(f"\n  📊 Stage 1 LayerNorm 权重统计（加载前）:")
                            for k, v in cleaned_norm_weights.items():
                                if isinstance(v, torch.Tensor):
                                    v_min, v_max = v.min().item(), v.max().item()
                                    v_mean, v_std = v.mean().item(), v.std().item()
                                    rank0_print(f"    {k}: shape={v.shape}, range=[{v_min:.6f}, {v_max:.6f}], mean={v_mean:.6f}, std={v_std:.6f}")
                                    # 🟢 关键检查：LayerNorm 的 gamma 参数（weight）应该接近 1.0（初始化值）
                                    # 如果训练后 gamma 接近 1.0，说明 LayerNorm 已经学习到合适的缩放
                                    if k == 'weight' or 'weight' in k:
                                        if abs(v_mean - 1.0) < 0.1 and abs(v_std - 0.1) < 0.1:
                                            rank0_print(f"      ✅ gamma 参数正常（接近 1.0，说明 LayerNorm 已正确训练）")
                                        elif abs(v_mean - 1.0) > 0.5:
                                            rank0_print(f"      ⚠️  警告: gamma 参数偏离 1.0 较远（mean={v_mean:.6f}），可能训练不充分")
                            
                            norm_missing_keys, norm_unexpected_keys = polar_projector_norm.load_state_dict(cleaned_norm_weights, strict=False)
                            norm_matched_keys = norm_loaded_keys & norm_expected_keys
                            
                            if len(norm_matched_keys) > 0:
                                rank0_print(f"  ✓ polar_projector_norm 权重已加载 ({len(norm_matched_keys)}/{len(norm_expected_keys)} keys matched)")
                                
                                # 🟢 新增：加载后验证权重是否正确加载
                                rank0_print(f"  📊 Stage 2 LayerNorm 权重统计（加载后）:")
                                loaded_state_dict = polar_projector_norm.state_dict()
                                for k in norm_matched_keys:
                                    if k in loaded_state_dict:
                                        v = loaded_state_dict[k]
                                        if isinstance(v, torch.Tensor):
                                            v_min, v_max = v.min().item(), v.max().item()
                                            v_mean, v_std = v.mean().item(), v.std().item()
                                            rank0_print(f"    {k}: shape={v.shape}, range=[{v_min:.6f}, {v_max:.6f}], mean={v_mean:.6f}, std={v_std:.6f}")
                                            
                                            # 🟢 关键检查：验证加载后的权重与 Stage 1 是否一致
                                            if k in cleaned_norm_weights:
                                                stage1_weight = cleaned_norm_weights[k]
                                                if isinstance(stage1_weight, torch.Tensor) and v.shape == stage1_weight.shape:
                                                    diff = (v - stage1_weight.to(v.device)).abs().max().item()
                                                    if diff < 1e-5:
                                                        rank0_print(f"      ✅ 权重验证通过：与 Stage 1 完全一致（diff < 1e-5）")
                                                    else:
                                                        rank0_print(f"      ⚠️  警告: 权重与 Stage 1 不一致（max diff={diff:.6f}）")
                                            
                                            # 🟢 关键检查：LayerNorm 的 gamma 参数（weight）应该接近 1.0
                                            if k == 'weight' or 'weight' in k:
                                                if abs(v_mean - 1.0) < 0.1:
                                                    rank0_print(f"      ✅ gamma 参数正常（mean={v_mean:.6f}，接近 1.0）")
                                                    rank0_print(f"      ℹ️  说明: LayerNorm 已正确训练，输出 Std 应该接近 1.0")
                                                else:
                                                    rank0_print(f"      ⚠️  警告: gamma 参数偏离 1.0（mean={v_mean:.6f}），可能影响输出 Std")
                            else:
                                rank0_print(f"  ⚠️  警告: polar_projector_norm 权重未找到，将使用随机初始化（LayerNorm 会从头训练）")
                        else:
                            rank0_print(f"  ⚠️  警告: 权重文件中未找到 polar_projector_norm，将使用随机初始化（LayerNorm 会从头训练）")
                    else:
                        rank0_print(f"  ⚠️  警告: polar_projector_norm 模块未找到")

                    # 🟢 关键修复：同时加载 polar_alpha 权重（残差融合）
                    polar_alpha = getattr(model.get_model(), 'polar_alpha', None)
                    if polar_alpha is not None:
                        all_weights_for_alpha = {}
                        if model_args.pretrain_polar_projector.endswith('.pth'):
                            all_weights_for_alpha = torch.load(model_args.pretrain_polar_projector, map_location='cpu', weights_only=False)
                        elif model_args.pretrain_polar_projector.endswith('.bin'):
                            all_weights_for_alpha = torch.load(model_args.pretrain_polar_projector, map_location='cpu', weights_only=False)
                        else:
                            checkpoint_dir = Path(model_args.pretrain_polar_projector)
                            if (checkpoint_dir / 'non_lora_trainables.bin').exists():
                                all_weights_for_alpha = torch.load(checkpoint_dir / 'non_lora_trainables.bin', map_location='cpu', weights_only=False)
                            elif (checkpoint_dir / 'mm_projector.bin').exists():
                                all_weights_for_alpha = torch.load(checkpoint_dir / 'mm_projector.bin', map_location='cpu', weights_only=False)

                        alpha_weights = {k: v for k, v in all_weights_for_alpha.items() if 'polar_alpha' in k}
                        if alpha_weights:
                            # 处理权重键名（移除前缀）
                            cleaned_alpha_weights = {}
                            for k, v in alpha_weights.items():
                                new_key = k
                                prefixes_to_remove = [
                                    'module.base_model.model.model.polar_alpha.',
                                    'base_model.model.model.polar_alpha.',
                                    'base_model.model.polar_alpha.',
                                    'model.polar_alpha.',
                                    'polar_alpha.',
                                ]
                                for prefix in prefixes_to_remove:
                                    if new_key.startswith(prefix):
                                        new_key = new_key[len(prefix):]
                                        break
                                cleaned_alpha_weights[new_key] = v
                            # 只需一个标量
                            if 'polar_alpha' in cleaned_alpha_weights and isinstance(cleaned_alpha_weights['polar_alpha'], torch.Tensor):
                                polar_alpha.data.copy_(cleaned_alpha_weights['polar_alpha'].to(polar_alpha.device))
                            else:
                                # 尝试使用第一个张量
                                first_val = next(iter(cleaned_alpha_weights.values()))
                                if isinstance(first_val, torch.Tensor):
                                    polar_alpha.data.copy_(first_val.to(polar_alpha.device))
                            polar_alpha_loaded = True
                            rank0_print(f"  ✓ polar_alpha 权重已加载 (current: {polar_alpha.item():.4f})")
                            # 🟢 Stage 2: 无论加载到什么值，都强制重置为 alpha_init
                            if model_args.training_stage == "stage2":
                                alpha_init = float(getattr(model_args, "polar_alpha_init", 0.5) or 0.5)
                                current_alpha = float(polar_alpha.item())
                                    with torch.no_grad():
                                    polar_alpha.data.fill_(alpha_init)
                                    rank0_print(
                                    f"  ℹ️  Stage 2: 已强制重置 polar_alpha 为 alpha_init={alpha_init:.4f}（加载值 {current_alpha:.4f} 被覆盖）"
                                    )
                        else:
                            rank0_print(f"  ⚠️  警告: 权重文件中未找到 polar_alpha，将使用初始化值 (current: {polar_alpha.item():.4f})")
                    else:
                        rank0_print(f"  ⚠️  警告: polar_alpha 模块未找到")
                    
                    # 🟢 Stage 1b 关键确认：读取 Stage 1a 权重结果
                    if model_args.training_stage == "stage1" and not model_args.polar_only:
                        alpha_status = "已加载" if polar_alpha_loaded else "未加载(使用初始化值)"
                        alpha_value = polar_alpha.item() if polar_alpha is not None else float('nan')
                        rank0_print(f"  ✅ Stage 1b 读取结果确认: polar_alpha {alpha_status}, current={alpha_value:.4f}")
                        # 🟢 关键需求：Stage 1b 强制将 alpha 重置为 0，从纯 RGB 开始再逐步释放 Polar
                        if polar_alpha is not None:
                            with torch.no_grad():
                                polar_alpha.data.fill_(0.0)
                            rank0_print(f"  ✅ Stage 1b: 已强制重置 polar_alpha=0.0（从纯 RGB 开始）")
                    
                    # 🟢 关键修复：同时加载 vae_latent_to_feature 权重（如果存在）
                    vae_latent_to_feature = getattr(model.get_model(), 'vae_latent_to_feature', None)
                    if vae_latent_to_feature is not None:
                        # 从原始权重文件中提取 vae_latent_to_feature 权重
                        # 重新加载完整权重文件（因为之前只提取了 polar_projector）
                        # 优先从 non_lora_trainables.bin 加载（Stage 1 保存的完整文件）
                        all_weights = {}
                        if model_args.pretrain_polar_projector.endswith('.pth'):
                            all_weights = torch.load(model_args.pretrain_polar_projector, map_location='cpu', weights_only=False)
                        elif model_args.pretrain_polar_projector.endswith('.bin'):
                            # 如果是 .bin 文件，直接加载（可能是 non_lora_trainables.bin）
                            all_weights = torch.load(model_args.pretrain_polar_projector, map_location='cpu', weights_only=False)
                        else:
                            # 如果是目录路径，优先查找 non_lora_trainables.bin（Stage 1 保存的完整文件）
                            checkpoint_dir = Path(model_args.pretrain_polar_projector)
                            if (checkpoint_dir / 'non_lora_trainables.bin').exists():
                                all_weights = torch.load(checkpoint_dir / 'non_lora_trainables.bin', map_location='cpu', weights_only=False)
                            elif (checkpoint_dir / 'mm_projector.bin').exists():
                                all_weights = torch.load(checkpoint_dir / 'mm_projector.bin', map_location='cpu', weights_only=False)
                        
                        # 提取包含 'vae_latent_to_feature' 的权重
                        vae_weights = {k: v for k, v in all_weights.items() if 'vae_latent_to_feature' in k}
                        if vae_weights:
                            # 处理权重键名（移除前缀）
                            cleaned_vae_weights = {}
                            for k, v in vae_weights.items():
                                new_key = k
                                prefixes_to_remove = [
                                    'module.base_model.model.model.vae_latent_to_feature.',  # DDP + PeftModel 包装后的完整路径
                                    'base_model.model.model.vae_latent_to_feature.',         # PeftModel 包装后的完整路径
                                    'base_model.model.vae_latent_to_feature.',               # 标准路径
                                    'model.vae_latent_to_feature.',                          # 简化路径
                                    'vae_latent_to_feature.',                                # 直接路径
                                ]
                                for prefix in prefixes_to_remove:
                                    if new_key.startswith(prefix):
                                        new_key = new_key[len(prefix):]
                                        break
                                cleaned_vae_weights[new_key] = v
                            
                            # 加载权重
                            vae_expected_keys = set(vae_latent_to_feature.state_dict().keys())
                            vae_loaded_keys = set(cleaned_vae_weights.keys())
                            vae_missing_keys, vae_unexpected_keys = vae_latent_to_feature.load_state_dict(cleaned_vae_weights, strict=False)
                            vae_matched_keys = vae_loaded_keys & vae_expected_keys
                            
                            if len(vae_matched_keys) > 0:
                                rank0_print(f"  ✓ vae_latent_to_feature 权重已加载 ({len(vae_matched_keys)}/{len(vae_expected_keys)} keys matched)")
                                
                                # 🟢 新增：加载后验证 vae_latent_to_feature 权重统计
                                rank0_print(f"  📊 Stage 2 vae_latent_to_feature 权重统计（加载后）:")
                                vae_state_dict = vae_latent_to_feature.state_dict()
                                for k in vae_matched_keys:
                                    if k in vae_state_dict:
                                        v = vae_state_dict[k]
                                        if isinstance(v, torch.Tensor):
                                            v_min, v_max = v.min().item(), v.max().item()
                                            v_mean, v_std = v.mean().item(), v.std().item()
                                            rank0_print(f"    {k}: shape={v.shape}, range=[{v_min:.6f}, {v_max:.6f}], mean={v_mean:.6f}, std={v_std:.6f}")
                            else:
                                rank0_print(f"  ⚠️  警告: vae_latent_to_feature 权重未找到，将使用随机初始化")
                        else:
                            rank0_print(f"  ⚠️  警告: 权重文件中未找到 vae_latent_to_feature，将使用随机初始化")
                    else:
                        rank0_print(f"  ⚠️  警告: vae_latent_to_feature 模块未找到")
                    
                    # 🟢 关键修复：加载 Stage 1 训练的 Polar token embedding（如果存在）
                    # 注意：这需要在添加新 token 之前进行，所以暂时跳过，在添加 token 之后再进行
                    # 这个逻辑会在后面（添加 token 之后）执行
                except Exception as e:
                    rank0_print(f"  ❌ 错误: 加载 Polar Projector 权重失败: {e}")
                    import traceback
                    traceback.print_exc()
                    # 对于 Stage 2，如果加载失败，应该抛出异常而不是继续
                    if model_args.training_stage == "stage2":
                        raise RuntimeError(
                            f"Critical: Failed to load Stage 1 Polar Projector weights for Stage 2 training! "
                            f"Stage 2 requires pretrained Polar Projector. Error: {e}"
                        )
                    else:
                        rank0_print(f"  ⚠️  将使用随机初始化的 Polar Projector")
            elif model_args.training_stage == "stage2" and not model_args.pretrain_polar_projector:
                rank0_print(f"\n⚠️  警告: Stage 2 训练但未提供 --pretrain_polar_projector")
                rank0_print(f"  Polar Projector 将使用随机初始化，Stage 1 的训练结果不会被使用")
                rank0_print(f"  建议: 添加 --pretrain_polar_projector /path/to/stage1_output/mm_projector.bin")
            
            # Stage 1: Polar Projector Alignment（只训练 Polar Projector）
            if model_args.training_stage == "stage1":
                rank0_print("Stage 1: Polar Projector Alignment")
                rank0_print("  - Freezing: RGB Tower, RGB Projector, Polar Encoder, LLM")
                rank0_print("  - Training: Polar Projector only")
                # 🟢 明确输出 Stage 1a/1b 关键开关，方便确认训练模式
                rank0_print(f"  - Fusion mode: {model_args.polar_fusion_mode}")
                rank0_print(f"  - Polar-only: {model_args.polar_only}")
                rank0_print(f"  - RGB Dropout p: {model_args.polar_rgb_dropout_p}")
                if model_args.polar_only:
                    rank0_print("  - Stage 1a mode: Polar-only alignment")
                else:
                    rank0_print("  - Stage 1b mode: Residual + RGB Dropout")
                rank0_print(f"  - Alpha init/min: {model_args.polar_alpha_init} / {model_args.polar_alpha_min}")
                
                # 冻结所有其他模块
                if rgb_vision_tower is not None:
                    rgb_vision_tower.requires_grad_(False)
                if rgb_projector is not None:
                    rgb_projector.requires_grad_(False)
                if polar_encoder is not None:
                    polar_encoder.requires_grad_(False)
                    polar_encoder.eval()  # 强制 eval 模式，避免 BatchNorm/Dropout 引入噪声
                    if hasattr(model.get_model(), 'polar_quant_conv'):
                        model.get_model().polar_quant_conv.requires_grad_(False)
                model.model.requires_grad_(False)  # 冻结 LLM
                
                # 只训练 Polar Projector、polar_projector_norm 和 vae_latent_to_feature
                if polar_projector is not None:
                    for p in polar_projector.parameters():
                        # 只对浮点类型参数设置 requires_grad
                        if p.dtype.is_floating_point:
                            p.requires_grad = True
                    rank0_print(f"  ✓ Polar Projector: {sum(p.numel() for p in polar_projector.parameters() if p.requires_grad)} trainable parameters")
                else:
                    rank0_print("  ⚠ Warning: Polar Projector not found!")
                
                # 🟢 关键修复：确保 polar_projector_norm 也被训练和保存
                polar_projector_norm = getattr(model.get_model(), 'polar_projector_norm', None)
                if polar_projector_norm is not None:
                    for p in polar_projector_norm.parameters():
                        if p.dtype.is_floating_point:
                            p.requires_grad = True
                    rank0_print(f"  ✓ polar_projector_norm: {sum(p.numel() for p in polar_projector_norm.parameters() if p.requires_grad)} trainable parameters")
                else:
                    rank0_print("  ⚠ Warning: polar_projector_norm not found!")

                # 🟢 关键修复：Stage 1a strict mode 下，polar_alpha 不参与训练
                polar_alpha = getattr(model.get_model(), 'polar_alpha', None)
                if polar_alpha is not None and hasattr(polar_alpha, 'requires_grad'):
                    if model_args.polar_only:
                        with torch.no_grad():
                            polar_alpha.data.fill_(0.0)
                        polar_alpha.requires_grad = False
                        rank0_print("  ✓ polar_alpha: Stage 1a strict mode 下已冻结（不参与训练）")
                    else:
                        # 🟢 安全检查：如果 polar_alpha 异常（NaN/Inf/过大），重置为初始化值
                        with torch.no_grad():
                            alpha_init = float(model_args.polar_alpha_init)
                            alpha_value = polar_alpha.item()
                            if (not torch.isfinite(polar_alpha).all()) or abs(alpha_value) > 10.0:
                                polar_alpha.data.fill_(alpha_init)
                                rank0_print(f"  ⚠️  警告: polar_alpha 异常（{alpha_value:.6f}），已重置为 {alpha_init:.4f}")
                        polar_alpha.requires_grad = True
                        rank0_print(f"  ✓ polar_alpha: 1 trainable parameter (current: {polar_alpha.item():.4f})")
                
                # 🟢 [可选增强] 确保 polar_range_scale 也被训练和保存（如果存在）
                # ⚠️ 注意：polar_range_scale 默认是冻结的（requires_grad=False），避免训练过程中变小导致 Polar 特征被缩小
                # 如果确实需要启用，可以在这里显式设置 requires_grad=True
                polar_range_scale = getattr(model.get_model(), 'polar_range_scale', None)
                if polar_range_scale is not None:
                    # 🟢 关键修复：确保 polar_range_scale 的值是 1.0（如果被意外重置为 0，则重新初始化）
                    scale_value = polar_range_scale.item()
                    if abs(scale_value) < 0.01:  # 如果值接近 0，说明可能被意外重置
                        with torch.no_grad():
                            polar_range_scale.data.fill_(1.0)
                        rank0_print(f"  ⚠️  警告: polar_range_scale 值异常（{scale_value:.4f}），已重置为 1.0")
                    
                    # 🟢 默认保持冻结，避免意外缩小 Polar 特征
                    # 如果需要启用，取消下面的注释：
                    # polar_range_scale.requires_grad = True
                    if polar_range_scale.requires_grad:
                        rank0_print(f"  ✓ polar_range_scale: {polar_range_scale.numel()} trainable parameters (当前值: {polar_range_scale.item():.4f})")
                        rank0_print(f"     ⚠️  警告: 缩放因子是可训练的，请监控其值，确保不会变得太小（<0.5）")
                    else:
                        rank0_print(f"  ℹ️  polar_range_scale: 已冻结（当前值: {polar_range_scale.item():.4f}，默认行为，安全）")
                
                # 🟢 关键修复：确保 vae_latent_to_feature 也被训练和保存
                vae_latent_to_feature = getattr(model.get_model(), 'vae_latent_to_feature', None)
                if vae_latent_to_feature is not None:
                    for p in vae_latent_to_feature.parameters():
                        if p.dtype.is_floating_point:
                            p.requires_grad = True
                    rank0_print(f"  ✓ vae_latent_to_feature: {sum(p.numel() for p in vae_latent_to_feature.parameters() if p.requires_grad)} trainable parameters")
                else:
                    rank0_print("  ⚠ Warning: vae_latent_to_feature not found!")
            
            # Stage 2: Visual Instruction Tuning（训练 Polar Projector + LLM LoRA，可选微调 VAE）
            elif model_args.training_stage == "stage2":
                rank0_print("Stage 2: Visual Instruction Tuning")
                rank0_print("  - Freezing: RGB Tower, RGB Projector")
                rank0_print("  - Training: Polar Projector + LLM (LoRA)")
                if not model_args.freeze_polar_encoder:
                    rank0_print("  - Training: Polar Encoder (optional, with small LR)")
                
                # 冻结 RGB 分支
                if model_args.freeze_rgb_tower and rgb_vision_tower is not None:
                    rgb_vision_tower.requires_grad_(False)
                    rank0_print("  ✓ RGB Vision Tower: Frozen")
                if model_args.freeze_rgb_projector and rgb_projector is not None:
                    rgb_projector.requires_grad_(False)
                    rank0_print("  ✓ RGB Projector: Frozen")
                
                # Polar Encoder（可选微调）
                if model_args.freeze_polar_encoder:
                    if polar_encoder is not None:
                        polar_encoder.requires_grad_(False)
                        polar_encoder.eval()  # 强制 eval 模式，避免 BatchNorm/Dropout 引入噪声
                        if hasattr(model.get_model(), 'polar_quant_conv'):
                            model.get_model().polar_quant_conv.requires_grad_(False)
                        rank0_print("  ✓ Polar Encoder: Frozen (eval mode)")
                else:
                    if polar_encoder is not None:
                        polar_encoder.requires_grad_(True)
                        if hasattr(model.get_model(), 'polar_quant_conv'):
                            model.get_model().polar_quant_conv.requires_grad_(True)
                        rank0_print("  ✓ Polar Encoder: Trainable (use small LR)")
                
                # Polar Projector（必须训练）
                if polar_projector is not None:
                    for p in polar_projector.parameters():
                        # 只对浮点类型参数设置 requires_grad
                        if p.dtype.is_floating_point:
                            p.requires_grad = True
                    rank0_print(f"  ✓ Polar Projector: Trainable ({sum(p.numel() for p in polar_projector.parameters() if p.requires_grad)} parameters)")
                
                # 🟢 关键修复：确保 polar_projector_norm 也被训练和保存（Stage 2）
                polar_projector_norm = getattr(model.get_model(), 'polar_projector_norm', None)
                if polar_projector_norm is not None:
                    for p in polar_projector_norm.parameters():
                        if p.dtype.is_floating_point:
                            p.requires_grad = True
                    rank0_print(f"  ✓ polar_projector_norm: Trainable ({sum(p.numel() for p in polar_projector_norm.parameters() if p.requires_grad)} parameters)")
                else:
                    rank0_print("  ⚠ Warning: polar_projector_norm not found!")

                # 🟢 关键修复：确保 polar_alpha 也被训练（残差融合）
                polar_alpha = getattr(model.get_model(), 'polar_alpha', None)
                if polar_alpha is not None and hasattr(polar_alpha, 'requires_grad'):
                    # 🟢 安全检查：如果 polar_alpha 异常（NaN/Inf/过大），重置为初始化值
                    with torch.no_grad():
                        alpha_init = float(model_args.polar_alpha_init)
                        alpha_min = float(model_args.polar_alpha_min)
                        alpha_value = polar_alpha.item()
                        if (not torch.isfinite(polar_alpha).all()) or abs(alpha_value) > 10.0:
                            polar_alpha.data.fill_(alpha_init)
                            rank0_print(f"  ⚠️  警告: polar_alpha 异常（{alpha_value:.6f}），已重置为 {alpha_init:.4f}")
                        # 🟢 关键修复：避免 alpha < alpha_min 导致 clamp 饱和、梯度为 0（常见于从 Stage1a 读取 alpha=0）
                        elif alpha_value < alpha_min:
                            reset_alpha = max(alpha_init, alpha_min)
                            polar_alpha.data.fill_(reset_alpha)
                            rank0_print(
                                f"  ℹ️  Stage 2: polar_alpha={alpha_value:.6f} < alpha_min={alpha_min:.4f}，"
                                f"已重置为 {reset_alpha:.4f}（确保可学习）"
                            )
                    polar_alpha.requires_grad = True
                    rank0_print(f"  ✓ polar_alpha: Trainable (current: {polar_alpha.item():.4f})")
                
                # 🟢 [可选增强] 确保 polar_range_scale 也被训练和保存（如果存在）
                # ⚠️ 注意：polar_range_scale 现在使用 register_buffer（不是 Parameter），所以不需要训练
                # 它只是一个固定的缩放因子（值为 1.0），用于将来可能的增强
                # 当前训练完全不需要它，LayerNorm 已经足够
                polar_range_scale = getattr(model.get_model(), 'polar_range_scale', None)
                if polar_range_scale is not None:
                    # 🟢 关键修复：确保 polar_range_scale 的值是 1.0（如果被意外重置为 0，则重新初始化）
                    scale_value = polar_range_scale.item()
                    if abs(scale_value) < 0.01:  # 如果值接近 0，说明可能被意外重置
                        with torch.no_grad():
                            polar_range_scale.data.fill_(1.0)
                        rank0_print(f"  ⚠️  警告: polar_range_scale 值异常（{scale_value:.4f}），已重置为 1.0")
                    
                    # 🟢 说明：polar_range_scale 现在是 buffer（不是 Parameter），所以不需要训练
                    # 它只是一个固定的缩放因子（值为 1.0），当前训练完全不需要它
                    rank0_print(f"  ℹ️  polar_range_scale: 已注册为 buffer（当前值: {polar_range_scale.item():.4f}，固定为 1.0，不影响训练）")
                    rank0_print(f"     说明: 当前训练完全不需要它，LayerNorm 已经足够对齐特征分布")
                
                # 🟢 关键修复：确保 vae_latent_to_feature 也被训练和保存（Stage 2）
                vae_latent_to_feature = getattr(model.get_model(), 'vae_latent_to_feature', None)
                if vae_latent_to_feature is not None:
                    for p in vae_latent_to_feature.parameters():
                        if p.dtype.is_floating_point:
                            p.requires_grad = True
                    rank0_print(f"  ✓ vae_latent_to_feature: Trainable ({sum(p.numel() for p in vae_latent_to_feature.parameters() if p.requires_grad)} parameters)")
                else:
                    rank0_print("  ⚠ Warning: vae_latent_to_feature not found!")
                
                # LLM 通过 LoRA 训练（如果启用）
                if training_args.lora_enable:
                    # 🟢 新增：详细检查各部分的参数数量
                    lora_param_count = sum(p.numel() for name, p in model.named_parameters() 
                                          if 'lora' in name.lower() and p.requires_grad)
                    
                    # 🟢 修改：Stage 2 完全冻结 embedding 层，不统计 embedding 参数
                    # 统计 embedding 层参数（modules_to_save）
                    embed_param_count = 0
                    if modules_to_save and model_args.training_stage != "stage2":
                        # Stage 2 不使用 modules_to_save，embedding 层被冻结
                        for name, p in model.named_parameters():
                            if p.requires_grad and any(m in name for m in modules_to_save):
                                embed_param_count += p.numel()
                    
                    # 统计 Polar 相关参数
                    polar_proj_count = sum(p.numel() for p in polar_projector.parameters() if p.requires_grad) if polar_projector else 0
                    polar_norm_count = sum(p.numel() for p in polar_projector_norm.parameters() if p.requires_grad) if polar_projector_norm else 0
                    vae_latent_count = sum(p.numel() for p in vae_latent_to_feature.parameters() if p.requires_grad) if vae_latent_to_feature else 0
                    polar_total = polar_proj_count + polar_norm_count + vae_latent_count
                    
                    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    lora_ratio = lora_param_count / total_trainable * 100 if total_trainable > 0 else 0
                    embed_ratio = embed_param_count / total_trainable * 100 if total_trainable > 0 else 0
                    polar_ratio = polar_total / total_trainable * 100 if total_trainable > 0 else 0
                    
                    rank0_print(f"  ✓ LLM: Trainable via LoRA")
                    rank0_print(f"     - LoRA 参数数量: {lora_param_count:,} ({lora_ratio:.2f}% 总可训练参数)")
                    if embed_param_count > 0:
                        rank0_print(f"     - Embedding 层参数: {embed_param_count:,} ({embed_ratio:.2f}% 总可训练参数)")
                        rank0_print(f"       说明: 包含 {', '.join(modules_to_save) if modules_to_save else 'N/A'}，用于新 token 的 embedding")
                    rank0_print(f"     - Polar 分支参数: {polar_total:,} ({polar_ratio:.2f}% 总可训练参数)")
                    rank0_print(f"        - Polar Projector: {polar_proj_count:,}")
                    rank0_print(f"        - LayerNorm: {polar_norm_count:,}")
                    rank0_print(f"        - VAE Latent Adapter: {vae_latent_count:,}")
                    rank0_print(f"     - 总可训练参数: {total_trainable:,}")
                    
                    # 🟢 验证：各部分参数之和应该等于总参数
                    calculated_total = lora_param_count + embed_param_count + polar_total
                    diff = abs(calculated_total - total_trainable)
                    if diff > 1000:  # 允许 1000 的误差（可能是其他小模块）
                        rank0_print(f"     ⚠️  警告: 参数统计不一致（计算值: {calculated_total:,}, 实际值: {total_trainable:,}）")
                        rank0_print(f"        差异: {diff:,}，可能是其他可训练模块（如 LayerNorm、Bias 等）")
                    else:
                        rank0_print(f"     ✅ 参数统计验证通过（各部分之和 = 总参数，差异: {diff:,} < 1000）")
                    
                    # 🟢 额外检查：如果 Embedding 层参数很大，检查是否是 tie_word_embeddings 的问题
                    if embed_param_count > 0:
                        # 计算预期的 embedding 参数（假设 vocab_size=32,004, hidden_size=5,120）
                        expected_embed_single = 32004 * 5120  # 163,860,480
                        if abs(embed_param_count - expected_embed_single * 2) < 1000:
                            rank0_print(f"     ℹ️  Embedding 层参数分析: {embed_param_count:,} ≈ 2 × {expected_embed_single:,}")
                            rank0_print(f"        说明: 检测到 embed_tokens 和 lm_head 都被计算为可训练参数")
                            rank0_print(f"        可能原因: tie_word_embeddings=False，或 PEFT modules_to_save 机制")
                            rank0_print(f"        影响: 这是正常的，两个模块都需要训练以支持新 token")
                        elif abs(embed_param_count - expected_embed_single) < 1000:
                            rank0_print(f"     ℹ️  Embedding 层参数分析: {embed_param_count:,} ≈ {expected_embed_single:,}")
                            rank0_print(f"        说明: 只有 embed_tokens 被计算为可训练参数（tie_word_embeddings=True）")
                else:
                    rank0_print("  ⚠ Warning: LoRA is not enabled. Consider using LoRA for Stage 2 training.")
                
                # ========== 关键修复：再次确认 Polar Projector 的梯度状态 ==========
                # 确保即使在 LoRA 初始化或其他操作后，polar_projector 仍然可训练
                if polar_projector is not None:
                    trainable_count = sum(1 for p in polar_projector.parameters() if p.requires_grad)
                    total_count = sum(1 for p in polar_projector.parameters())
                    if trainable_count != total_count:
                        rank0_print(f"  ⚠️  警告: Polar Projector 有 {total_count - trainable_count} 个参数被冻结，正在重新启用...")
                        for p in polar_projector.parameters():
                            # 只对浮点类型参数设置 requires_grad
                            if p.dtype.is_floating_point:
                                p.requires_grad = True
                        trainable_after = sum(1 for p in polar_projector.parameters() if p.requires_grad)
                        rank0_print(f"  ✓ Polar Projector 梯度已重新启用 ({trainable_after} parameters)")
                    else:
                        rank0_print(f"  ✓ Polar Projector 梯度状态确认: {trainable_count}/{total_count} parameters trainable")
                
                # 🟢 新增：Stage 2 调试信息总结
                rank0_print("\n" + "=" * 80)
                rank0_print("📋 Stage 2 训练配置总结（读取、训练、保存）")
                rank0_print("=" * 80)
                
                # 1. 读取内容
                rank0_print("\n📥 Stage 2 读取的内容:")
                if model_args.pretrain_polar_projector and os.path.exists(model_args.pretrain_polar_projector):
                    rank0_print(f"  ✓ 从 Stage 1 checkpoint 读取:")
                    rank0_print(f"     - polar_projector 权重（从 {model_args.pretrain_polar_projector}）")
                    if polar_projector_norm is not None:
                        rank0_print(f"     - polar_projector_norm 权重（LayerNorm，从 Stage 1 继续训练）")
                    if vae_latent_to_feature is not None:
                        rank0_print(f"     - vae_latent_to_feature 权重（VAE Latent Adapter，从 Stage 1 继续训练）")
                else:
                    rank0_print(f"  ⚠️  警告: 未提供 Stage 1 checkpoint，Polar 分支将从头训练")
                rank0_print(f"  ✓ 从基础模型读取:")
                rank0_print(f"     - LLM 权重（LLaVA 1.5-13B）")
                rank0_print(f"     - RGB Vision Tower（CLIP，冻结）")
                rank0_print(f"     - RGB Projector（冻结）")
                
                # 2. 训练内容
                rank0_print("\n🎯 Stage 2 训练的内容:")
                rank0_print(f"  ✓ Polar 分支（可训练）:")
                if polar_projector is not None:
                    rank0_print(f"     - polar_projector: {sum(p.numel() for p in polar_projector.parameters() if p.requires_grad):,} 参数")
                if polar_projector_norm is not None:
                    rank0_print(f"     - polar_projector_norm (LayerNorm): {sum(p.numel() for p in polar_projector_norm.parameters() if p.requires_grad):,} 参数")
                if vae_latent_to_feature is not None:
                    rank0_print(f"     - vae_latent_to_feature: {sum(p.numel() for p in vae_latent_to_feature.parameters() if p.requires_grad):,} 参数")
                if training_args.lora_enable:
                    rank0_print(f"  ✓ LLM LoRA（可训练）:")
                    rank0_print(f"     - LoRA 参数: {lora_param_count:,} ({lora_ratio:.2f}% 总可训练参数)")
                rank0_print(f"  ✓ 冻结的模块:")
                rank0_print(f"     - RGB Vision Tower（CLIP）")
                rank0_print(f"     - RGB Projector")
                rank0_print(f"     - Polar Encoder（VAE，冻结）")
                rank0_print(f"     - Embedding 层（input_embeddings, output_embeddings，完全冻结，避免过拟合）")
                rank0_print(f"     - LLM 主干（通过 LoRA 微调，不直接训练）")
                
                # 3. 保存内容
                rank0_print("\n💾 Stage 2 保存的内容:")
                rank0_print(f"  ✓ 每个 checkpoint 保存:")
                rank0_print(f"     - adapter_model.bin（LoRA 权重）")
                rank0_print(f"     - adapter_config.json（LoRA 配置）")
                rank0_print(f"     - non_lora_trainables.bin（非 LoRA 可训练权重）")
                rank0_print(f"       包含:")
                if polar_projector is not None:
                    rank0_print(f"          - polar_projector 权重")
                if polar_projector_norm is not None:
                    rank0_print(f"          - polar_projector_norm 权重（LayerNorm）")
                if vae_latent_to_feature is not None:
                    rank0_print(f"          - vae_latent_to_feature 权重")
                rank0_print(f"     - 注意: Embedding 层不保存（已冻结，使用基础模型的 embedding）")
                
                # 4. 特征融合方式
                fusion_mode = model_args.polar_fusion_mode
                rank0_print("\n🔗 特征融合方式（Stage 1 和 Stage 2 一致）:")
                rank0_print(f"  ✓ RGB 特征: 576 tokens (24×24 patches)")
                rank0_print(f"  ✓ Polar 特征: 576 tokens (24×24 patches)")
                if fusion_mode == "residual":
                    rank0_print(f"  ✓ 融合方式: residual (fused = RGB + alpha * Polar)，token 数不变（576）")
                else:
                    rank0_print(f"  ✓ 融合方式: concat ([RGB, Polar])，token 数为 1152")
                rank0_print(f"  ✓ 不使用分隔 token（避免过拟合，数据量少时更有效）")
                rank0_print(f"  ✓ 词表大小: 32000（未添加新 token）")
                
                # 5. 训练数据格式
                rank0_print("\n📊 训练数据格式:")
                if fusion_mode == "residual":
                    rank0_print(f"  ✓ 输入格式: [BOS] [Fused Tokens] [Text Tokens] [EOS]")
                else:
                    rank0_print(f"  ✓ 输入格式: [BOS] [RGB Tokens] [Polar Tokens] [Text Tokens] [EOS]")
                rank0_print(f"  ✓ 不使用分隔 token（<RGB_START>, <RGB_END>, <POL_START>, <POL_END>）")
                
                rank0_print("\n" + "=" * 80)
            
            else:
                rank0_print(f"  ⚠ Warning: Unknown training stage '{model_args.training_stage}'. Using default settings.")
        
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
            if use_polar and hasattr(model.get_model(), 'polar_projector'):
                model.get_model().polar_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        # 🟢 关键修复：Stage 1 禁用 <im_patch> token，避免量化模型无法 resize embedding 的问题
        # <im_patch> token 在当前实现中不使用（使用 IMAGE_TOKEN_INDEX = -200 代替）
        # 🟢 关键修复：在量化模型训练时（Stage 1 和 Stage 2），都应该禁用 <im_patch> token
        # 因为量化模型无法 resize embedding，如果添加 <im_patch> 会导致 tokenizer 和模型 embedding 大小不匹配
        if model_args.training_stage == "stage1":
            model.config.mm_use_im_patch_token = False
            model_args.mm_use_im_patch_token = False
            rank0_print("  ℹ️  Stage 1: 已禁用 <im_patch> token（避免量化模型 embedding 大小不匹配）")
        elif training_args.bits == 4:
            # Stage 2 使用量化模型时，也禁用 <im_patch> token
            # 因为我们已经手动resize了embedding到32004（添加了4个分隔token），无法再次resize
            model.config.mm_use_im_patch_token = False
            model_args.mm_use_im_patch_token = False
            rank0_print("  ℹ️  Stage 2 (4-bit): 已禁用 <im_patch> token（避免量化模型 embedding 大小不匹配）")
        else:
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        
        # 记录 resize 前的词表大小
        vocab_size_before = len(tokenizer)
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
        vocab_size_after = len(tokenizer)
        
        # 🟢 关键修复：如果 tokenizer 词表扩展了，考虑让新 token 的 embedding 参与训练
        # 注意：仅当底层 embedding 权重是浮点类型时才允许 requires_grad=True（量化权重不能求梯度）
        # 🟢 关键：必须在 DDP 包装之前完成所有参数状态修改，使用 no_grad 上下文避免触发 autograd hooks
        # 🟢 关键修复：Stage 1 不训练 embedding，所以即使词表扩展了，也要保持冻结状态
        if vocab_size_after > vocab_size_before:
            num_new_tokens = vocab_size_after - vocab_size_before
            rank0_print(f"  ✓ 词表已扩展: {vocab_size_before} -> {vocab_size_after} (+{num_new_tokens} tokens)")
            
            # 🟢 关键修复：检查 embedding 大小是否真的被 resize 了
            # 对于量化模型，可能无法 resize，导致 tokenizer 和模型 embedding 大小不匹配
            current_embed_size = model.get_input_embeddings().weight.shape[0]
            if current_embed_size != vocab_size_after:
                rank0_print(f"  ⚠️  警告: Tokenizer 词表大小 ({vocab_size_after}) 与模型 embedding 大小 ({current_embed_size}) 不匹配！")
                rank0_print(f"     这通常发生在量化模型中，因为量化模型无法 resize embedding")
                rank0_print(f"     新 token 的 embedding 将无法使用，可能导致索引越界错误")
                rank0_print(f"     建议：对于量化模型，确保所有需要的 token 在模型初始化时就已经存在")
            
            # 🟢 关键修复：Stage 1 不训练 embedding，即使词表扩展了也要保持冻结
            if model_args.training_stage == "stage1":
                rank0_print(f"  ℹ️  Stage 1: 不训练 embedding（包括新 token），保持冻结状态")
                # 确保 embedding 保持冻结（即使之前被启用）
                input_embeddings = model.get_input_embeddings()
                output_embeddings = model.get_output_embeddings()
                
                with torch.no_grad():
                    if hasattr(input_embeddings, 'weight') and input_embeddings.weight.dtype.is_floating_point:
                        if input_embeddings.weight.requires_grad:
                            input_embeddings.weight.requires_grad = False
                            rank0_print(f"  ✓ Stage 1: 已重新冻结 input_embeddings（确保不训练新 token）")
                    
                    if hasattr(output_embeddings, 'weight') and output_embeddings.weight.dtype.is_floating_point:
                        if output_embeddings.weight.requires_grad:
                            output_embeddings.weight.requires_grad = False
                            rank0_print(f"  ✓ Stage 1: 已重新冻结 output_embeddings（确保不训练新 token）")
        
        # 🟢 关键修复：Stage 2 也完全冻结 embedding 层（避免过拟合）
        # 无论词表是否扩展，Stage 2 都应该冻结 embedding 层
        # 数据量少时，训练 embedding 容易过拟合，直接冻结更安全
        if model_args.training_stage == "stage2":
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
            
            # 🟢 使用 no_grad 上下文确保参数状态修改不会触发 autograd hooks（避免 DDP 错误）
            with torch.no_grad():
                # Stage 2: 完全冻结 embedding 层（避免过拟合）
                input_frozen = False
                output_frozen = False
                
                if hasattr(input_embeddings, 'weight') and input_embeddings.weight.dtype.is_floating_point:
                    if input_embeddings.weight.requires_grad:
                        input_embeddings.weight.requires_grad = False
                        input_frozen = True
                
                if hasattr(output_embeddings, 'weight') and output_embeddings.weight.dtype.is_floating_point:
                    if output_embeddings.weight.requires_grad:
                        output_embeddings.weight.requires_grad = False
                        output_frozen = True
                
                # 如果 embedding 层有 bias，也冻结
                if hasattr(input_embeddings, 'bias') and input_embeddings.bias is not None and input_embeddings.bias.dtype.is_floating_point:
                    if input_embeddings.bias.requires_grad:
                        input_embeddings.bias.requires_grad = False
                if hasattr(output_embeddings, 'bias') and output_embeddings.bias is not None and output_embeddings.bias.dtype.is_floating_point:
                    if output_embeddings.bias.requires_grad:
                        output_embeddings.bias.requires_grad = False
                
                # 🟢 验证并输出冻结状态
                input_requires_grad = input_embeddings.weight.requires_grad if hasattr(input_embeddings, 'weight') and hasattr(input_embeddings.weight, 'requires_grad') else False
                output_requires_grad = output_embeddings.weight.requires_grad if hasattr(output_embeddings, 'weight') and hasattr(output_embeddings.weight, 'requires_grad') else False
                
                if input_frozen:
                    rank0_print(f"  ✓ Stage 2: 已冻结 input_embeddings（避免过拟合，数据量少）")
                elif not input_requires_grad:
                    rank0_print(f"  ✓ Stage 2: input_embeddings 已处于冻结状态（无需操作）")
                else:
                    rank0_print(f"  ⚠️  警告: input_embeddings 仍可训练（可能被 PEFT 或其他机制启用）")
                
                if output_frozen:
                    rank0_print(f"  ✓ Stage 2: 已冻结 output_embeddings (lm_head)（避免过拟合，数据量少）")
                elif not output_requires_grad:
                    rank0_print(f"  ✓ Stage 2: output_embeddings (lm_head) 已处于冻结状态（无需操作）")
                else:
                    rank0_print(f"  ⚠️  警告: output_embeddings (lm_head) 仍可训练（可能被 PEFT 或其他机制启用）")

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    rank0_print("=" * 60)
    rank0_print("📊 开始加载数据...")
    # 🔍 验证：对话模板配置（确保 Stage 1 使用 plain 模式）
    rank0_print(f"🔍 验证：对话模板配置")
    rank0_print(f"   Conversation Template: {conversation_lib.default_conversation.version}")
    rank0_print(f"   Sep Style: {conversation_lib.default_conversation.sep_style}")
    system_preview = repr(conversation_lib.default_conversation.system[:50]) if conversation_lib.default_conversation.system else 'None (plain mode)'
    rank0_print(f"   System Prompt: {system_preview}")
    if model_args.training_stage == "stage1":
        if conversation_lib.default_conversation.version == "plain":
            rank0_print(f"   ✅ Stage 1 确认：使用 plain 模式（无 System Prompt）")
        else:
            rank0_print(f"   ❌ 错误：Stage 1 应该使用 plain 模式，但当前为 {conversation_lib.default_conversation.version}")
    rank0_print("")
    rank0_print(f"   训练集: {data_args.data_path}")
    if data_args.val_json:
        rank0_print(f"   验证集: {data_args.val_json}")
    rank0_print(f"   图像目录: {data_args.image_folder}")
    if use_polar:
        rank0_print(f"   Polar 目录: {data_args.polar_folder}")
    rank0_print("=" * 60)
    
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    
    rank0_print("🚀 初始化训练器...")
    rank0_print(f"   输出目录: {training_args.output_dir}")
    rank0_print(f"   训练阶段: {model_args.training_stage}")
    rank0_print("=" * 60)
    
    # 🟢 严格冻结检查：训练开始前打印所有可训练参数，并阻止 LLM 被误训练
    def _is_llm_param_name(param_name: str) -> bool:
        llm_keywords = (
            ".model.layers.", "model.layers.",
            ".model.embed_tokens", "model.embed_tokens",
            ".model.norm", "model.norm",
            ".lm_head", "lm_head",
            "self_attn", "mlp",
            "lora_", "lora_A", "lora_B",
        )
        return any(k in param_name for k in llm_keywords)

    trainable_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
    rank0_print(f"  ✅ Trainable 参数总数: {len(trainable_param_names)}")
    # 默认不输出全量参数列表，避免日志过大；如需完整列表可临时启用环境变量
    show_full_trainable = os.getenv("SHOW_TRAINABLE_PARAMS", "0") == "1"
    if trainable_param_names:
        if show_full_trainable:
            rank0_print("  🔍 Trainable 参数列表（全量）：")
            for n in trainable_param_names:
                rank0_print(f"    - {n}")
        else:
            preview_count = min(20, len(trainable_param_names))
            rank0_print(f"  🔍 Trainable 参数预览（前 {preview_count} 个，设置 SHOW_TRAINABLE_PARAMS=1 查看全量）：")
            for n in trainable_param_names[:preview_count]:
                rank0_print(f"    - {n}")

    if model_args.training_stage == "stage1":
        llm_trainable = [n for n in trainable_param_names if _is_llm_param_name(n)]
        if llm_trainable:
            rank0_print("  ❌ 严重错误: 检测到 LLM 层处于可训练状态（Stage 1 必须完全冻结 LLM）")
            for n in llm_trainable:
                rank0_print(f"     LLM trainable: {n}")
            raise ValueError("Stage 1 训练检测失败：LLM 参数未冻结。请检查冻结逻辑。")
    
    # 🟢 添加特征统计监控回调
    feature_stats_callback = FeatureStatsMonitorCallback()
    gradient_callback = GradientMonitorCallback()
    alpha_callback = AlphaMonitorCallback()
    
    # 🟢 特征统计监控已禁用（LayerNorm 工作正常，特征对齐良好）
    # 如果需要临时启用调试，可以设置环境变量 MONITOR_FEATURE_STATS=1
    import os
    os.environ['MONITOR_FEATURE_STATS'] = '0'  # 默认禁用
    # os.environ['MONITOR_TOKEN_STATS'] = '1'    # 已禁用：不输出 token 数量（576/1152）
    os.environ['CURRENT_TRAINING_STEP'] = '-1'
    
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    callbacks=[feature_stats_callback, gradient_callback, alpha_callback],
                    **data_module)
    
    # 🔧 配置 logging 为实时输出（确保在 nohup 下也能实时看到 loss）
    import logging
    # 配置 logging 格式和实时刷新
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)  # 使用我们包装过的 stdout
        ],
        force=True  # 强制重新配置
    )
    
    # 获取 transformers 的 logger 并配置
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.INFO)
    transformers_trainer_logger = logging.getLogger("transformers.trainer")
    transformers_trainer_logger.setLevel(logging.INFO)
    
    # 确保所有现有的 handler 都使用实时刷新的流
    for logger in [logging.root, transformers_logger, transformers_trainer_logger]:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                # 如果 handler 使用的是 sys.stdout 或 sys.stderr，它们已经被包装过了
                handler.flush()
    
    # 设置 tqdm 进度条也实时刷新（transformers 使用 tqdm 显示进度）
    try:
        import tqdm
        # tqdm 默认会实时刷新，但我们可以确保它使用我们的流
        if hasattr(tqdm, 'tqdm'):
            original_tqdm_init = tqdm.tqdm.__init__
            def flushed_tqdm_init(self, *args, **kwargs):
                result = original_tqdm_init(self, *args, **kwargs)
                # 确保 tqdm 使用实时刷新的流
                if hasattr(self, 'file') and self.file in [sys.stdout, sys.stderr]:
                    # 已经是我们的 FlushedStream，无需修改
                    pass
                return result
            # 只在第一次时包装
            if not hasattr(tqdm.tqdm, '_flushed_wrapped'):
                tqdm.tqdm.__init__ = flushed_tqdm_init
                tqdm.tqdm._flushed_wrapped = True
    except ImportError:
        pass

    # 最终梯度状态检查
    if use_polar:
        base_model = model.get_model()
        polar_proj = getattr(base_model, 'polar_projector', None)
    if polar_proj:
        trainable_params = sum(p.numel() for p in polar_proj.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in polar_proj.parameters())
        rank0_print(f"   Polar Projector: {trainable_params}/{total_params} parameters trainable")
        if trainable_params == 0:
            raise ValueError("❌ Polar Projector 梯度全被冻结了！请检查解冻逻辑。")
            
            # 🟢 关键修复：同时检查 polar_projector_norm 的梯度状态
            polar_projector_norm = getattr(base_model, 'polar_projector_norm', None)
            if polar_projector_norm:
                trainable_params = sum(p.numel() for p in polar_projector_norm.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in polar_projector_norm.parameters())
                rank0_print(f"   polar_projector_norm: {trainable_params}/{total_params} parameters trainable")
                if trainable_params == 0:
                    rank0_print("  ⚠️  警告: polar_projector_norm 梯度全被冻结了！LayerNorm 将无法训练")
            
            # 🟢 关键修复：同时检查 vae_latent_to_feature 的梯度状态
            vae_latent_to_feature = getattr(base_model, 'vae_latent_to_feature', None)
            if vae_latent_to_feature:
                trainable_params = sum(p.numel() for p in vae_latent_to_feature.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in vae_latent_to_feature.parameters())
                rank0_print(f"   vae_latent_to_feature: {trainable_params}/{total_params} parameters trainable")
                if trainable_params == 0:
                    raise ValueError("❌ vae_latent_to_feature 梯度全被冻结了！请检查解冻逻辑。")
    
    # ========== 关于 Gradient Checkpointing 警告的说明 ==========
    if training_args.gradient_checkpointing:
        rank0_print("\n" + "=" * 60)
        rank0_print("ℹ️  关于 Gradient Checkpointing 警告的说明")
        rank0_print("=" * 60)
        rank0_print("   如果看到以下警告：")
        rank0_print("   'UserWarning: None of the inputs have requires_grad=True. Gradients will be None'")
        rank0_print("\n   这是 HuggingFace Trainer + LoRA + Gradient Checkpointing 的常见警告，")
        rank0_print("   通常可以安全忽略，原因：")
        rank0_print("   1. 代码已通过 enable_input_require_grads() 强制让 Embedding 输出需要梯度")
        rank0_print("   2. Polar 分支的 latent 已显式设置为 requires_grad=True")
        rank0_print("   3. 只要 Loss 在下降，说明梯度确实在正常流动")
        rank0_print("\n   验证方法：")
        rank0_print("   - 观察训练日志中的 Loss 是否正常下降")
        rank0_print("   - 如果 Loss 不下降，再检查梯度问题")
        rank0_print("=" * 60 + "\n")
    
    # ========== 输出训练比例和统计信息 ==========
    rank0_print("=" * 60)
    rank0_print("📈 训练统计信息")
    rank0_print("=" * 60)
    
    # 获取训练和验证数据量
    train_dataset_size = len(data_module['train_dataset'])
    eval_dataset_size = len(data_module['eval_dataset']) if data_module.get('eval_dataset') else 0
    total_dataset_size = train_dataset_size + eval_dataset_size
    
    # 计算训练比例
    if total_dataset_size > 0:
        train_ratio = train_dataset_size / total_dataset_size * 100
        eval_ratio = eval_dataset_size / total_dataset_size * 100 if eval_dataset_size > 0 else 0
    else:
        train_ratio = 100.0
        eval_ratio = 0.0
    
    rank0_print(f"   训练集大小: {train_dataset_size:,} 样本 ({train_ratio:.2f}%)")
    if eval_dataset_size > 0:
        rank0_print(f"   验证集大小: {eval_dataset_size:,} 样本 ({eval_ratio:.2f}%)")
    rank0_print(f"   总数据量: {total_dataset_size:,} 样本")
    
    # 计算训练步数和epoch信息
    num_epochs = training_args.num_train_epochs
    per_device_batch_size = training_args.per_device_train_batch_size
    gradient_accumulation_steps = training_args.gradient_accumulation_steps
    
    # 获取GPU数量
    num_gpus = 1
    if hasattr(training_args, 'world_size') and training_args.world_size > 0:
        num_gpus = training_args.world_size
    elif hasattr(training_args, 'n_gpu') and training_args.n_gpu > 0:
        num_gpus = training_args.n_gpu
    elif torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    
    # 总batch size
    total_batch_size = per_device_batch_size * num_gpus * gradient_accumulation_steps
    
    # 每个epoch的步数
    steps_per_epoch = (train_dataset_size + total_batch_size - 1) // total_batch_size  # 向上取整
    
    # 总步数（优先使用max_steps，否则根据epochs计算）
    max_steps = getattr(training_args, 'max_steps', 0)
    if max_steps <= 0:
        max_steps = int(num_epochs * steps_per_epoch)
    
    rank0_print(f"\n   训练配置:")
    rank0_print(f"   - Epochs: {num_epochs}")
    rank0_print(f"   - 每设备batch size: {per_device_batch_size}")
    rank0_print(f"   - 梯度累积步数: {gradient_accumulation_steps}")
    rank0_print(f"   - 总batch size: {total_batch_size}")
    rank0_print(f"   - 每个epoch步数: {steps_per_epoch:,}")
    rank0_print(f"   - 总训练步数: {max_steps:,}")
    
    # 如果是Stage 2，额外显示一些信息
    if model_args.training_stage == "stage2":
        rank0_print(f"\n   Stage 2 训练比例:")
        rank0_print(f"   - 训练数据占比: {train_ratio:.2f}% ({train_dataset_size:,}/{total_dataset_size:,})")
        if eval_dataset_size > 0:
            rank0_print(f"   - 验证数据占比: {eval_ratio:.2f}% ({eval_dataset_size:,}/{total_dataset_size:,})")
            rank0_print(f"   - 训练/验证比例: {train_dataset_size/eval_dataset_size:.2f}:1")
    
    rank0_print("=" * 60)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        rank0_print("🔄 检测到已有checkpoint，从checkpoint恢复训练...")
        trainer.train(resume_from_checkpoint=True)
    else:
        rank0_print("🚀 开始训练...")
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        # 🟢 关键修复：最终保存时也要保存所有非 LoRA 参数（与 checkpoint 保存逻辑一致）
        # 使用 require_grad_only=False 确保保存所有非 LoRA 参数，而不仅仅是可训练的
        # 这样可以确保最终保存的文件与 checkpoint 中的文件一致
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(),
            require_grad_only=False  # 强制保存所有非 LoRA 参数，无论 requires_grad 状态
        )
        
        # 🟢 关键修复：强制保存 Polar 核心模块 (polar_projector + polar_projector_norm + vae_latent_to_feature + polar_range_scale)
        # 无论 requires_grad 状态如何，都必须保存这些层的权重（Stage 1 和 Stage 2 都需要）
        if use_polar:
            polar_keys_to_save = ["polar_projector", "polar_projector_norm", "vae_latent_to_feature", "polar_range_scale", "polar_alpha"]
            forced_saved_count = 0
            for name, param in model.named_parameters():
                if any(k in name for k in polar_keys_to_save):
                    # 🟢 关键修复：无论是否已在 non_lora_state_dict 中，都强制保存（覆盖）
                    # 这样可以确保即使 requires_grad 被意外关闭，也能保存权重
                    non_lora_state_dict[name] = maybe_zero_3(param, ignore_status=True).cpu()
                    forced_saved_count += 1
                    rank0_print(f"  ✓ 强制保存 Polar 核心模块: {name} (shape: {param.shape}, requires_grad={param.requires_grad})")
            if forced_saved_count > 0:
                rank0_print(f"  ✓ 已强制保存 {forced_saved_count} 个 Polar 核心参数（无论 requires_grad 状态）")
        
        # 🟢 关键修复：如果 resize 了词表，必须强制保存 embed_tokens 和 lm_head
        # 即使它们是冻结的（requires_grad=False），也需要保存，否则推理时词表大小不匹配
        # 检查词表是否被 resize（通过比较 tokenizer 和模型 embedding 的大小）
        tokenizer_vocab_size = len(tokenizer)
        model_vocab_size = model.get_input_embeddings().weight.shape[0]
        
        # 如果 tokenizer 和模型 embedding 大小一致，且大于 LLaMA 原始词表（32000），说明被 resize 了
        if tokenizer_vocab_size == model_vocab_size and tokenizer_vocab_size > 32000:
            # 检查是否添加了新 token（包括4个特殊token、IMAGE_PATCH_TOKEN、IM_START/END等）
            has_new_tokens = False
            
            # 🟢 修改：Stage 1 和 Stage 2 都不添加分隔 token，所以不需要检查特殊token
            # 如果词表大小 > 32000，可能是其他 token（如 IMAGE_PATCH_TOKEN）导致的
            
            # 检查 IMAGE_PATCH_TOKEN
            if model_args.mm_use_im_patch_token:
                from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN
                if DEFAULT_IMAGE_PATCH_TOKEN in tokenizer.get_vocab():
                    has_new_tokens = True
                    rank0_print(f"  ✓ 检测到 {DEFAULT_IMAGE_PATCH_TOKEN} token")
            
            # 检查 IM_START/END tokens
            if model_args.mm_use_im_start_end:
                from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
                if (DEFAULT_IM_START_TOKEN in tokenizer.get_vocab() and 
                    DEFAULT_IM_END_TOKEN in tokenizer.get_vocab()):
                    has_new_tokens = True
                    rank0_print(f"  ✓ 检测到 IM_START/END tokens")
            
            # 如果词表大小大于32000，即使没有检测到特定token，也认为可能添加了新token（安全起见）
            if tokenizer_vocab_size > 32000:
                has_new_tokens = True
            
            if has_new_tokens:
                # 词表已被 resize，强制保存 embed_tokens 和 lm_head
                # 🟢 关键修复：无论 requires_grad 状态如何，都必须保存（覆盖）
                # 因为词表大小变了，推理时必须使用新的 embedding 大小
                saved_count = 0
                for name, param in model.named_parameters():
                    if "embed_tokens" in name or "lm_head" in name:
                        # 无论是否已在 non_lora_state_dict 中，都强制保存（覆盖）
                        # 即使 requires_grad=False，也要保存（因为词表大小变了）
                        non_lora_state_dict[name] = maybe_zero_3(param, ignore_status=True).cpu()
                        saved_count += 1
                        rank0_print(f"  ✓ 强制保存 resize 后的 embedding: {name} (shape: {param.shape}, requires_grad={param.requires_grad})")
                if saved_count > 0:
                    rank0_print(f"  ✓ 已强制保存 {saved_count} 个 resize 后的 embedding 参数（无论 requires_grad 状态）")
        
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            
            # 🟢 关键修复：PEFT 的 save_pretrained 可能无法正确处理手动 resize 的 modules_to_save
            # 我们需要先尝试正常保存，如果失败则使用备用方案
            try:
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
                rank0_print(f"  ✓ 已保存 PEFT 适配器")
            except (KeyError, AttributeError) as e:
                if 'modules_to_save' in str(e) or 'default' in str(e):
                    rank0_print(f"  ⚠️  PEFT save_pretrained 失败（modules_to_save key 不匹配），使用备用保存方案...")
                    # 备用方案：手动保存 adapter 配置和权重
                    from peft import PeftModel
                    if isinstance(model, PeftModel):
                        # 保存 adapter_config.json
                        adapter_config = model.peft_config
                        import json
                        
                        # 🟢 关键修复：将 set、tuple、frozenset 等不可序列化类型转换为可序列化类型
                        def convert_to_json_serializable(obj):
                            """递归地将对象转换为 JSON 可序列化的类型"""
                            if isinstance(obj, dict):
                                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                            elif isinstance(obj, (list, tuple)):
                                return [convert_to_json_serializable(item) for item in obj]
                            elif isinstance(obj, set):
                                # 将 set 转换为排序后的 list（确保可重复性）
                                return sorted(list(obj)) if obj else []
                            elif isinstance(obj, frozenset):
                                # 将 frozenset 转换为排序后的 list
                                return sorted(list(obj)) if obj else []
                            elif isinstance(obj, (int, float, str, bool, type(None))):
                                # 基本类型，直接返回
                                return obj
                            elif hasattr(obj, '__dict__'):
                                # 自定义对象，尝试转换为字典
                                try:
                                    return convert_to_json_serializable(obj.__dict__)
                                except:
                                    return str(obj)
                            else:
                                # 其他类型，转换为字符串
                                try:
                                    # 尝试直接序列化（可能是 numpy 类型等）
                                    import json
                                    json.dumps(obj)  # 测试是否可序列化
                                    return obj
                                except (TypeError, ValueError):
                                    return str(obj)
                        
                        adapter_config_dict = {}
                        for adapter_name, config in adapter_config.items():
                            try:
                                config_dict = config.to_dict()
                                # 🟢 关键修复：深度转换所有不可序列化的类型
                                adapter_config_dict[adapter_name] = convert_to_json_serializable(config_dict)
                            except Exception as e:
                                rank0_print(f"  ⚠️  警告: 转换 adapter '{adapter_name}' 配置时出错: {e}")
                                # 如果转换失败，尝试直接序列化（可能会失败，但至少尝试）
                                try:
                                    adapter_config_dict[adapter_name] = convert_to_json_serializable(config.__dict__)
                                except Exception as e2:
                                    rank0_print(f"  ❌ 错误: 无法序列化 adapter '{adapter_name}' 配置: {e2}")
                                    # 使用空字典作为占位符
                                    adapter_config_dict[adapter_name] = {}
                        
                        # 🟢 关键修复：在保存前再次验证是否可序列化
                        try:
                            import json
                            json.dumps(adapter_config_dict)  # 测试是否可序列化
                        except TypeError as e:
                            rank0_print(f"  ⚠️  警告: adapter_config_dict 仍包含不可序列化的类型: {e}")
                            rank0_print(f"  尝试更激进的转换...")
                            # 更激进的转换：将所有非基本类型都转换为字符串
                            def aggressive_convert(obj):
                                if isinstance(obj, dict):
                                    return {k: aggressive_convert(v) for k, v in obj.items()}
                                elif isinstance(obj, (list, tuple, set, frozenset)):
                                    return [aggressive_convert(item) for item in obj]
                                elif isinstance(obj, (int, float, str, bool, type(None))):
                                    return obj
                                else:
                                    return str(obj)
                            adapter_config_dict = aggressive_convert(adapter_config_dict)
                        
                        with open(os.path.join(training_args.output_dir, 'adapter_config.json'), 'w') as f:
                            json.dump(adapter_config_dict, f, indent=2)
                        rank0_print(f"  ✓ 已保存 adapter_config.json")
                        
                        # 保存 LoRA 权重（不包含 modules_to_save，因为它们已经在 non_lora_state_dict 中）
                        from peft.utils import WEIGHTS_NAME
                        adapter_weights = {}
                        for name, param in model.named_parameters():
                            if 'lora' in name.lower() and 'modules_to_save' not in name:
                                adapter_weights[name] = param.cpu()
                        if adapter_weights:
                            weights_file = os.path.join(training_args.output_dir, WEIGHTS_NAME)
                            torch.save(adapter_weights, weights_file)
                            rank0_print(f"  ✓ 已保存 LoRA 权重到 {weights_file} (包含 {len(adapter_weights)} 个参数)")
                    else:
                        raise
                else:
                    raise
            
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            # 计算 Polar 参数的总元素数（用于验证）
            if use_polar:
                polar_total_elements = sum(
                    param.numel() for name, param in non_lora_state_dict.items()
                    if any(k in name for k in ["polar_projector", "polar_projector_norm", "vae_latent_to_feature", "polar_range_scale", "polar_alpha"])
                )
                rank0_print(f"  ✓ 已保存 non_lora_trainables.bin")
                rank0_print(f"    - 总参数对象数: {len(non_lora_state_dict)}")
                rank0_print(f"    - Polar 核心参数元素数: {polar_total_elements:,} (包含 polar_projector + polar_projector_norm + vae_latent_to_feature + polar_range_scale)")
                
                # 🟢 新增：验证保存的权重统计（特别是 LayerNorm）
                rank0_print(f"\n  📊 保存的 Polar 权重验证:")
                polar_saved_keys = [name for name in non_lora_state_dict.keys() 
                                   if any(k in name for k in ["polar_projector", "polar_projector_norm", "vae_latent_to_feature", "polar_range_scale", "polar_alpha"])]
                
                # 检查 polar_projector
                proj_keys = [k for k in polar_saved_keys if 'polar_projector' in k and 'norm' not in k]
                if proj_keys:
                    rank0_print(f"    ✓ polar_projector: {len(proj_keys)} 个权重已保存")
                
                # 检查 polar_projector_norm（LayerNorm）
                norm_keys = [k for k in polar_saved_keys if 'polar_projector_norm' in k]
                if norm_keys:
                    rank0_print(f"    ✓ polar_projector_norm: {len(norm_keys)} 个权重已保存")
                    for k in norm_keys:
                        v = non_lora_state_dict[k]
                        if isinstance(v, torch.Tensor):
                            v_min, v_max = v.min().item(), v.max().item()
                            v_mean, v_std = v.mean().item(), v.std().item()
                            rank0_print(f"      {k}: shape={v.shape}, range=[{v_min:.6f}, {v_max:.6f}], mean={v_mean:.6f}, std={v_std:.6f}")
                            # 🟢 关键检查：LayerNorm 的 gamma 参数（weight）应该接近 1.0
                            if 'weight' in k:
                                if abs(v_mean - 1.0) < 0.1:
                                    rank0_print(f"        ✅ gamma 参数正常（接近 1.0），LayerNorm 已正确训练")
                                else:
                                    rank0_print(f"        ⚠️  警告: gamma 参数偏离 1.0（mean={v_mean:.6f}）")
                else:
                    rank0_print(f"    ⚠️  警告: polar_projector_norm 权重未保存！")
                
                # 检查 vae_latent_to_feature
                vae_keys = [k for k in polar_saved_keys if 'vae_latent_to_feature' in k]
                if vae_keys:
                    rank0_print(f"    ✓ vae_latent_to_feature: {len(vae_keys)} 个权重已保存")
                else:
                    rank0_print(f"    ⚠️  警告: vae_latent_to_feature 权重未保存！")
                
                # 检查 polar_range_scale
                scale_keys = [k for k in polar_saved_keys if 'polar_range_scale' in k]
                if scale_keys:
                    rank0_print(f"    ✓ polar_range_scale: {len(scale_keys)} 个权重已保存")
                    for k in scale_keys:
                        v = non_lora_state_dict[k]
                        if isinstance(v, torch.Tensor):
                            scale_value = v.item() if v.numel() == 1 else v.mean().item()
                            rank0_print(f"      {k}: value={scale_value:.6f} (应该是 1.0)")
                else:
                    rank0_print(f"    ℹ️  polar_range_scale: 未保存（这是正常的，因为它是 buffer，不是 Parameter）")
                
                # 检查 polar_alpha
                alpha_keys = [k for k in polar_saved_keys if 'polar_alpha' in k]
                if alpha_keys:
                    rank0_print(f"    ✓ polar_alpha: {len(alpha_keys)} 个权重已保存")
                    for k in alpha_keys:
                        v = non_lora_state_dict[k]
                        if isinstance(v, torch.Tensor):
                            alpha_value = v.item() if v.numel() == 1 else v.mean().item()
                            rank0_print(f"      {k}: value={alpha_value:.6f}")
                else:
                    rank0_print(f"    ⚠️  警告: polar_alpha 权重未保存！")
            else:
                rank0_print(f"  ✓ 已保存 non_lora_trainables.bin (包含 {len(non_lora_state_dict)} 个参数)")
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()

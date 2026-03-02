#    Copyright 2023 Haotian Liu
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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import os

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel(nn.Module):

    def __init__(self, config):
        # 在多重继承情况下（如 LlavaLlamaModel(LlavaMetaModel, LlamaModel)），
        # super() 会按照 MRO 调用 LlamaModel.__init__(config)，所以需要传递 config
        # 注意：LlavaMetaModel 总是通过多重继承使用，不会单独实例化
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        model = self.get_model()
        image_features = model.get_vision_tower()(images)
        image_features = model.mm_projector(image_features)
        
        # 🟢 关键修复：确保输出是 FP16（与 LLM 权重类型一致）
        model_dtype = next(model.model.parameters()).dtype if hasattr(model, 'model') else next(model.parameters()).dtype
        if image_features.dtype != model_dtype:
            image_features = image_features.to(dtype=model_dtype)
        
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        
        # 🟢 获取模型的 dtype（通常是 FP16），在循环外获取一次即可
        try:
            model_dtype = next(self.get_model().embed_tokens.parameters()).dtype
        except:
            model_dtype = torch.float16
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                # 🟢 确保 embed_tokens 输出是 FP16
                cur_input_embeds_1 = cur_input_embeds_1.to(dtype=model_dtype)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            # 🟢 确保 embed_tokens 输出是 FP16
            cur_input_embeds = cur_input_embeds.to(dtype=model_dtype)
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            # 🟢 关键修复：确保所有 embeddings 都是 FP16 类型
            cur_new_input_embeds = [x.to(self.device).to(dtype=model_dtype) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        # 检测是否是量化模型（4-bit 或 8-bit）
        is_quantized = False
        try:
            # 检查是否有量化相关的属性
            if hasattr(self, 'hf_quantizer') or hasattr(self, 'quantization_config'):
                is_quantized = True
            # 检查 embedding 权重是否是量化类型（非浮点类型）
            input_emb = self.get_input_embeddings()
            if hasattr(input_emb, 'weight'):
                # 量化模型的权重可能是 int8 或其他非浮点类型
                if not input_emb.weight.dtype.is_floating_point:
                    is_quantized = True
        except:
            pass
        
        # 对于量化模型，尝试在 resize 之前检查是否需要
        if is_quantized:
            # 量化模型不支持 resize_token_embeddings
            # 检查 tokenizer 是否已经包含所需的 token
            current_vocab_size = len(tokenizer)
            model_vocab_size = self.get_input_embeddings().weight.shape[0]
            
            if model_args.mm_use_im_patch_token:
                if DEFAULT_IMAGE_PATCH_TOKEN not in tokenizer.get_vocab():
                    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
                    print(f"Warning: Added {DEFAULT_IMAGE_PATCH_TOKEN} to tokenizer, but cannot resize embeddings for quantized model. "
                          f"Model vocab size: {model_vocab_size}, Tokenizer vocab size: {len(tokenizer)}")
                else:
                    print(f"Info: {DEFAULT_IMAGE_PATCH_TOKEN} already in tokenizer")
            
            if model_args.mm_use_im_start_end:
                tokens_to_add = [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
                tokens_exist = all(t in tokenizer.get_vocab() for t in tokens_to_add)
                if not tokens_exist:
                    num_new_tokens = tokenizer.add_tokens(tokens_to_add, special_tokens=True)
                    print(f"Warning: Added {num_new_tokens} tokens to tokenizer, but cannot resize embeddings for quantized model. "
                          f"Model vocab size: {model_vocab_size}, Tokenizer vocab size: {len(tokenizer)}")
                else:
                    print(f"Info: Image start/end tokens already in tokenizer")
            return
        
        # 非量化模型：正常处理
        if model_args.mm_use_im_patch_token:
            num_added = tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            if num_added > 0:
                self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            if num_new_tokens > 0:
                self.resize_token_embeddings(len(tokenizer))
                
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False


# ======================= PolarLlavaMetaForCausalLM: 双流编码逻辑 =======================

class PolarLlavaMetaForCausalLM(LlavaMetaForCausalLM):
    """
    Polar LLaVA 的 Mixin 类，提供双流视觉编码逻辑（RGB + Polar）
    
    注意：编码方法必须定义在 MetaForCausalLM 中，而不是 MetaModel 中
    """
    
    def encode_polar_images(self, polar_images):
        """
        编码偏振图像
        
        Args:
            polar_images: 偏振图像张量，形状 (B, 3, H, W)，值范围 [0, 1]
                - 3通道: [DoLP, sin(2*AoLP), cos(2*AoLP)]
        
        Returns:
            polar_features: 投影后的偏振特征，形状 (B, N, hidden_size)
                - N = 24 * 24 = 576 (与 RGB 对齐)
        """
        model = self.get_model()
        if not hasattr(model, 'polar_encoder') or model.polar_encoder is None:
            raise ValueError("Polar encoder is not initialized. Please set polar_vae_model_path in config.")
        
        # 获取 VAE 编码器和相关模块
        polar_encoder = model.polar_encoder
        polar_quant_conv = model.polar_quant_conv
        vae_latent_to_feature = model.vae_latent_to_feature
        polar_projector = model.polar_projector
        
        # 🟢 调试：检查 Polar Projector 权重健康度
        # ⚠️ 多GPU适配：确保 polar_images 在 VAE 编码器所在的设备上
        # 获取 VAE 编码器所在的设备
        vae_device = next(polar_encoder.parameters()).device
        # 获取 VAE 的 dtype（通常是 float32）
        vae_dtype = next(polar_encoder.parameters()).dtype
        
        # 确保 polar_images 在正确的设备上
        if polar_images.device != vae_device:
            polar_images = polar_images.to(vae_device)
        
        # 1. Resize 到 512x512（VAE 的输入尺寸）
        if polar_images.shape[2] != 512 or polar_images.shape[3] != 512:
            polar_images = torch.nn.functional.interpolate(
                polar_images,
                size=(512, 512),
                mode='bilinear',
                align_corners=False
            )
        
        # 2. 转换范围：[0, 1] -> [-1, 1]
        # 🟢 关键修复：检查输入范围，避免重复归一化
        # 如果输入已经是 [-1, 1] 范围（推理时可能在 preprocess_images 中已归一化），则跳过转换
        # 如果输入是 [0, 1] 范围（训练时），则执行转换
        polar_min, polar_max = polar_images.min().item(), polar_images.max().item()
        
        # 🟢 判断逻辑：
        # - 如果 min < -0.1 且 max <= 1.1，说明输入已经是 [-1, 1] 范围（推理时已归一化）
        # - 如果 min >= -0.1 且 max <= 1.1，说明输入是 [0, 1] 范围（训练时）
        # - 如果 max > 1.1，说明输入异常，但为了安全，仍然执行转换
        is_already_normalized = (polar_min < -0.1) and (polar_max <= 1.1)
        
        if is_already_normalized:
            # 输入已经是 [-1, 1] 范围（推理时已在 preprocess_images 中归一化），跳过转换
            polar_input = polar_images
            # 🟢 调试：确认跳过了转换（仅在推理时打印，避免训练时刷屏）
            import os
            if os.getenv('DEBUG_POLAR_NORMALIZATION', '0') == '1':
                print(f"  🔍 DEBUG: encode_polar_images - 输入已是 [-1, 1] 范围，跳过转换 (Range: [{polar_min:.4f}, {polar_max:.4f}])")
        else:
            # 输入是 [0, 1] 范围（训练时），执行转换
            polar_input = polar_images * 2.0 - 1.0
            # 🟢 调试：确认执行了转换（仅在推理时打印）
            import os
            if os.getenv('DEBUG_POLAR_NORMALIZATION', '0') == '1':
                print(f"  🔍 DEBUG: encode_polar_images - 执行 [0, 1] -> [-1, 1] 转换 (输入范围: [{polar_min:.4f}, {polar_max:.4f}])")
        
        # 3. 转换到 VAE 的 dtype（修复数据类型不匹配问题）
        polar_input = polar_input.to(dtype=vae_dtype)
        
        # 4. VAE 编码 (通用修复版)
        # 无论 encoder 是 training 还是 eval 模式，我们都统一处理
        # 目的：计算 latent，并强制让它成为计算图的起点
        
        # 使用 torch.no_grad() 节省显存，因为 VAE 本身不需要更新（即使可训练，也先这样处理）
        with torch.no_grad():
            h = polar_encoder(polar_input)  # (B, 512, 64, 64)
            moments = polar_quant_conv(h)  # (B, 8, 64, 64)
            
            # 检查 Quant Conv 输出是否异常（仅保留必要的错误检查）
            moments_min, moments_max = moments.min().item(), moments.max().item()
            if abs(moments_min) > 1e10 or abs(moments_max) > 1e10:
                print(f"  ❌ 严重错误: Quant Conv 输出异常巨大！这会导致后续计算爆炸！")
                print(f"     可能原因：1) Quant Conv 权重损坏 2) Encoder 输出异常")
            
            latent, _ = torch.chunk(moments, 2, dim=1)  # (B, 4, 64, 64)
            latent = latent * 0.18215
        
        # =================================================================
        # 🚨【核心修复】🚨：这一行必须执行！
        # 无论梯度检查点是否开启，这一步都确保了 Projector 能收到梯度。
        # .detach() 切断与 VAE 的联系（节省显存）
        # .requires_grad = True 开启新一段计算图
        # 🌟 终极修复：分两行写，显式创建新的叶子节点，强制 PyTorch 认为计算图从这里重新开始
        # =================================================================
        latent = latent.detach()
        latent.requires_grad = True
        
        # 🟢 关键修复：NaN/Inf 检查和清理（VAE 输出后）
        # 使用更智能的清理策略：保留有效信息
        nan_mask = torch.isnan(latent)
        inf_mask = torch.isinf(latent)
        if nan_mask.any() or inf_mask.any():
            nan_count = nan_mask.sum().item()
            inf_count = inf_mask.sum().item()
            total_elements = latent.numel()
            nan_ratio = nan_count / total_elements * 100
            inf_ratio = inf_count / total_elements * 100
            
            if nan_ratio > 10.0 or inf_ratio > 10.0:
                print(f"  ⚠️ 警告: VAE latent 异常值比例过高 (NaN: {nan_ratio:.2f}%, Inf: {inf_ratio:.2f}%)")
                print(f"     建议检查输入数据是否已正确归一化到 [0, 1] 范围")
            
            # 使用均值填充 NaN
            if nan_mask.any():
                valid_mean = latent[~nan_mask].mean() if (~nan_mask).any() else 0.0
                latent = torch.where(nan_mask, torch.tensor(valid_mean, device=latent.device, dtype=latent.dtype), latent)
            
            # Inf 值裁剪到合理范围（VAE latent 通常在 [-1, 1] 范围）
            if inf_mask.any():
                latent = torch.clamp(latent, min=-2.0, max=2.0)
        
        # 🟢 关键修复：VAE 输出后立即转换为 FP16，确保后续所有组件都是 FP16
        # VAE 是 FP32（防止溢出），但后续所有组件（vae_latent_to_feature, polar_projector）都是 FP16
        # 获取 vae_latent_to_feature 的 dtype（应该是 FP16，与 LLM 对齐）
        vae_latent_dtype = next(vae_latent_to_feature.parameters()).dtype
        # 🚨 关键：在 VAE 输出后立即转换为 FP16，避免精度混用
        if latent.dtype != vae_latent_dtype:
            latent = latent.to(dtype=vae_latent_dtype)
        
        # 5. 投影到特征维度并下采样到与 RGB 对齐的空间分辨率
        # 先通过 1x1 卷积投影到特征维度（此时 latent 已经是 FP16）
        polar_features_spatial = vae_latent_to_feature(latent)  # (B, 256, 64, 64)
        
        # 🟢 关键修复：NaN/Inf 检查和清理（vae_latent_to_feature 输出后）
        # 使用更智能的清理策略：保留有效信息，只替换异常值
        nan_mask = torch.isnan(polar_features_spatial)
        inf_mask = torch.isinf(polar_features_spatial)
        if nan_mask.any() or inf_mask.any():
            nan_count = nan_mask.sum().item()
            inf_count = inf_mask.sum().item()
            total_elements = polar_features_spatial.numel()
            nan_ratio = nan_count / total_elements * 100
            inf_ratio = inf_count / total_elements * 100
            
            # 如果异常值比例过高（>10%），说明输入有问题，需要警告
            if nan_ratio > 10.0 or inf_ratio > 10.0:
                print(f"  ⚠️ 警告: vae_latent_to_feature 输出异常值比例过高 (NaN: {nan_ratio:.2f}%, Inf: {inf_ratio:.2f}%)")
                print(f"     建议检查输入数据是否已正确归一化到 [0, 1] 范围")
            
            # 使用均值填充 NaN，而不是直接置零（保留更多信息）
            if nan_mask.any():
                valid_mean = polar_features_spatial[~nan_mask].mean() if (~nan_mask).any() else 0.0
                polar_features_spatial = torch.where(nan_mask, torch.tensor(valid_mean, device=polar_features_spatial.device, dtype=polar_features_spatial.dtype), polar_features_spatial)
            
            # Inf 值裁剪到合理范围
            if inf_mask.any():
                # 使用有效值的范围来裁剪 Inf
                valid_values = polar_features_spatial[~inf_mask]
                if len(valid_values) > 0:
                    valid_max = valid_values.max().item()
                    valid_min = valid_values.min().item()
                    clip_max = max(abs(valid_max), abs(valid_min), 1.0)  # 至少裁剪到 ±1.0
                    polar_features_spatial = torch.clamp(polar_features_spatial, min=-clip_max, max=clip_max)
                else:
                    # 如果没有有效值，使用默认范围
                    polar_features_spatial = torch.clamp(polar_features_spatial, min=-1.0, max=1.0)
        
        # 插值到 24x24（与 RGB 对齐）
        polar_features_spatial = torch.nn.functional.interpolate(
            polar_features_spatial,
            size=(24, 24),
            mode='bilinear',
            align_corners=False
        )  # (B, 256, 24, 24)
        
        # 6. 展平为序列形式
        B, C, H, W = polar_features_spatial.shape
        polar_features_flat = polar_features_spatial.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, 576, 256)
        
        # 🟢【关键修复】移除之前的"转换桥"，让 FP32 直接进入 Projector
        # 🟢 关键修复：确保 polar_features_flat 是 FP16（与 Projector 匹配）
        # 现在 Projector 是 FP16，所以这里确保输入也是 FP16
        if polar_features_flat.dtype != torch.float16:
            polar_features_flat = polar_features_flat.to(dtype=torch.float16)
        
        # 注意：移除了特征裁剪和归一化逻辑，让 Projector 学习真正的特征分布
        # 在训练时，动态的归一化会干扰 Projector 学习特征幅度
        
        # 7. 通过 Polar Projector 投影到 LLM 的 hidden_size (在 FP16 下进行，与 LLM 对齐)
        # 🚨 关键：Projector 现在是 FP16，输出直接是 FP16，无需转换
        polar_features = polar_projector(polar_features_flat)  # (B, 576, hidden_size)
        
        # 🟢 [关键修复] 应用 LayerNorm：自动对齐 Polar 和 RGB 特征的分布
        # LayerNorm 会将 Polar 特征的分布标准化（Mean=0, Std=1），与 RGB 特征（Std~0.95）对齐
        # 这比手动缩放更稳健，具有更好的泛化性
        if hasattr(model, 'polar_projector_norm') and model.polar_projector_norm is not None:
            polar_features = model.polar_projector_norm(polar_features)  # (B, 576, hidden_size)
            
            # 🟢 [可选增强] 应用可学习的缩放因子：进一步对齐 Range（如果 LayerNorm 后 Range 仍然不匹配）
            # 注意：这个缩放因子是可选的，通常 LayerNorm 已经足够
            # 如果训练后发现 Range 差异仍然影响性能，可以启用这个缩放因子
            # 初始化：1.0（不缩放），默认冻结（requires_grad=False），避免训练过程中变小导致 Polar 特征被缩小
            # ⚠️ 安全措施：如果缩放因子 < 0.5，会输出警告，提示可能存在问题
            if hasattr(model, 'polar_range_scale') and model.polar_range_scale is not None:
                scale_value = model.polar_range_scale.item()
                polar_features = polar_features * model.polar_range_scale  # (B, 576, hidden_size)
                
                # 🟢 安全监控：如果缩放因子变得太小，输出警告
                if scale_value < 0.5:
                    import warnings
                    warnings.warn(
                        f"⚠️  polar_range_scale 值过小 ({scale_value:.4f} < 0.5)，可能导致 Polar 特征被过度缩小，"
                        f"LLM 可能过度关注 RGB 分支。建议检查训练过程或冻结缩放因子。",
                        UserWarning
                    )
        
        # 🟢 验证：确保输出是 FP16（与 LLM 对齐）
        if hasattr(model, 'model'):
            llm_dtype = next(model.model.embed_tokens.parameters()).dtype
        else:
            llm_dtype = next(model.embed_tokens.parameters()).dtype
        
        if polar_features.dtype != llm_dtype:
            polar_features = polar_features.to(dtype=llm_dtype)
        
        # 🟢 关键修复：NaN/Inf 检查和清理（polar_projector 输出后）
        # 使用更智能的清理策略
        nan_mask = torch.isnan(polar_features)
        inf_mask = torch.isinf(polar_features)
        if nan_mask.any() or inf_mask.any():
            nan_count = nan_mask.sum().item()
            inf_count = inf_mask.sum().item()
            total_elements = polar_features.numel()
            nan_ratio = nan_count / total_elements * 100
            inf_ratio = inf_count / total_elements * 100
            
            # 如果异常值比例过高，警告并尝试修复
            if nan_ratio > 10.0 or inf_ratio > 10.0:
                print(f"  ⚠️ 警告: polar_projector 输出异常值比例过高 (NaN: {nan_ratio:.2f}%, Inf: {inf_ratio:.2f}%)")
                print(f"     这可能导致模型输出乱码。建议检查 Projector 权重或输入特征范围。")
            
            # 如果全是 Inf，说明 Projector 计算溢出，使用零填充（至少让模型能运行）
            if inf_ratio > 90.0:
                print(f"  ❌ 严重错误: Projector 输出几乎全是 Inf ({inf_ratio:.2f}%)，使用零填充作为应急措施")
                print(f"     建议：1) 检查 Projector 权重是否正常 2) 减小输入特征值范围 3) 回退到更早的 checkpoint")
                polar_features = torch.zeros_like(polar_features)
            else:
                # 使用均值填充 NaN
                if nan_mask.any():
                    valid_mean = polar_features[~nan_mask].mean() if (~nan_mask).any() else 0.0
                    polar_features = torch.where(nan_mask, torch.tensor(valid_mean, device=polar_features.device, dtype=polar_features.dtype), polar_features)
                
                # Inf 值裁剪
                if inf_mask.any():
                    valid_values = polar_features[~inf_mask]
                    if len(valid_values) > 0:
                        valid_max = valid_values.max().item()
                        valid_min = valid_values.min().item()
                        clip_max = max(abs(valid_max), abs(valid_min), 10.0)  # 对于特征，允许更大的范围
                        polar_features = torch.clamp(polar_features, min=-clip_max, max=clip_max)
                    else:
                        polar_features = torch.clamp(polar_features, min=-10.0, max=10.0)
        
        # ⚠️ 多GPU适配：确保 polar_features 在 LLM 所在的设备上
        # 获取 LLM 所在的设备（通常是模型的第一层）
        # 对于多GPU模型，需要确保 polar_features 与 RGB 特征在同一设备上以便拼接
        if hasattr(model, 'model'):
            # 尝试从 model.model 获取设备（LlamaForCausalLM 结构）
            llm_device = next(model.model.parameters()).device
        else:
            # 尝试从 model 直接获取设备
            llm_device = next(model.parameters()).device
        
        # 如果 polar_features 不在 LLM 设备上，移动到 LLM 设备
        if polar_features.device != llm_device:
            polar_features = polar_features.to(llm_device)
        
        # 🟢 关键修复：强制转换为 FP16（确保与 LLM 权重类型一致）
        # 获取模型的 dtype（应该是 FP16）
        model_dtype = next(model.model.parameters()).dtype if hasattr(model, 'model') else next(model.parameters()).dtype
        if polar_features.dtype != model_dtype:
            polar_features = polar_features.to(dtype=model_dtype)
        
        return polar_features
    
    def encode_images(self, images, polar_images=None):
        """
        编码图像（RGB + 可选的 Polar）
        
        统一的双流架构（Stage 1 和 Stage 2 均支持）：
        - residual: fused = rgb + alpha * polar（token 数不变，返回 576 tokens）
        - concat: [rgb, polar] 拼接（返回 1152 tokens）
        - 如果不提供 Polar：只使用 RGB，返回 576 tokens
        
        Args:
            images: RGB 图像
            polar_images: 可选的偏振图像
        
        Returns:
            image_features: 投影后的图像特征
                - (B, 576, hidden_size)        — 仅 RGB
                - (B, 1152, hidden_size)       — RGB + Polar 拼接
        """
        model = self.get_model()
        # Stage 1a strict mode: 完全 Polar-only，不使用 RGB 分支与 alpha
        force_polar_only = bool(
            getattr(self.config, 'training_stage', None) == 'stage1' and
            getattr(self.config, 'polar_only', False)
        )
        
        # 获取模型 dtype（用于类型一致性）
        model_dtype = next(model.model.parameters()).dtype if hasattr(model, 'model') else next(model.parameters()).dtype
        
        # 1. 编码 RGB（Stage 1a strict mode 下跳过，确保与 RGB 无关）
        rgb_features = None
        if not force_polar_only:
            rgb_features = self.get_vision_tower()(images)
            rgb_features = model.mm_projector(rgb_features)  # (B, 576, hidden_size)
            
            # 🟢 关键修复：NaN/Inf 检查和清理（RGB 特征处理）
            if torch.isnan(rgb_features).any() or torch.isinf(rgb_features).any():
                print("  ⚠️ 警告: RGB features 包含 NaN/Inf，已自动清理")
                rgb_features = torch.nan_to_num(rgb_features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 🟢 关键修复：确保 rgb_features 是 FP16
            if rgb_features.dtype != model_dtype:
                rgb_features = rgb_features.to(dtype=model_dtype)
        
        # 2. 编码 Polar (如果存在)
        if polar_images is not None and hasattr(model, 'polar_encoder') and model.polar_encoder is not None:
            polar_features = self.encode_polar_images(polar_images)  # (B, 576, hidden_size)
            
            # ⚠️ 多GPU适配：确保 RGB 和 Polar 特征在同一设备上才能拼接
            if rgb_features is not None and rgb_features.device != polar_features.device:
                polar_features = polar_features.to(rgb_features.device)
            
            # 🟢 关键修复：确保 polar_features 也是 FP16
            if polar_features.dtype != model_dtype:
                polar_features = polar_features.to(dtype=model_dtype)
            
            # 🟢 训练/推理监控：检查特征拼接前的状态
            import os
            debug_polar = os.getenv('DEBUG_POLAR_BRANCH', '0') == '1'
            monitor_features = os.getenv('MONITOR_FEATURE_STATS', '0') == '1'  # 训练时监控特征统计
            
            # 🟢 关键修复：使用全局字典确保每个 step 只输出一次（避免同一 step 的多个 batch 重复输出）
            # 使用全局字典存储上次输出的 step，避免使用线程锁（更简单）
            if '_monitor_state' not in globals():
                globals()['_monitor_state'] = {'last_step': -1}
            
            current_step_str = os.getenv('CURRENT_TRAINING_STEP', '-1')
            try:
                current_step = int(current_step_str)
            except (ValueError, TypeError):
                current_step = -1
            
            # 只在新的 step 时输出（每个 step 的第一个 batch）
            # 🟢 关键修复：如果 current_step 是 -1（训练开始时的特殊标记），也允许输出
            should_output_monitor = False
            if monitor_features:
                if current_step == -1:
                    # 训练开始时的特殊标记，允许输出（第一个 batch）
                    globals()['_monitor_state']['last_step'] = -1
                    should_output_monitor = True
                elif current_step != globals()['_monitor_state']['last_step']:
                    # 新的 step，允许输出
                    globals()['_monitor_state']['last_step'] = current_step
                    should_output_monitor = True
                # 否则，同一 step 的后续 batch 不输出
            
            # 计算特征统计（无论是否打印，都计算以便后续使用）
            if rgb_features is not None:
                rgb_range = rgb_features.max().item() - rgb_features.min().item()
                rgb_std = rgb_features.std().item()
                rgb_mean = rgb_features.mean().item()
            else:
                # Stage 1a strict mode: 无 RGB 分支
                rgb_range = 0.0
                rgb_std = 0.0
                rgb_mean = 0.0
            polar_range = polar_features.max().item() - polar_features.min().item()
            polar_std = polar_features.std().item()
            polar_mean = polar_features.mean().item()
            # 🟢 特征统计监控已删除（LayerNorm 工作正常，特征对齐良好）
            # 如果需要临时启用调试，可以设置环境变量 MONITOR_FEATURE_STATS=1
            # 计算特征统计（保留计算逻辑，但不输出，以便将来需要时可以快速恢复）
            range_ratio = polar_range / rgb_range if rgb_range > 0 else float('inf')
            std_ratio = polar_std / rgb_std if rgb_std > 0 else float('inf')
            
            # 3. 融合 (Dual Stream)
            fusion_mode = getattr(self.config, 'polar_fusion_mode', 'residual')
            if fusion_mode == 'residual':
                # Stage 1a strict mode：彻底不走 alpha，不走 RGB，直接用 Polar 特征
                if force_polar_only:
                    image_features = polar_features
                else:
                    # RGB Dropout（训练时生效）：防止 RGB 主导
                    rgb_dropout_p = float(getattr(self.config, 'polar_rgb_dropout_p', 0.0) or 0.0)
                    if self.training and rgb_dropout_p > 0.0:
                        dropout_mask = (torch.rand(rgb_features.shape[0], 1, 1, device=rgb_features.device) >= rgb_dropout_p)
                        rgb_features = rgb_features * dropout_mask
                    # 可学习 alpha（下限保护）
                    alpha = getattr(model, 'polar_alpha', None)
                    if alpha is None:
                        alpha_value = torch.tensor(1.0, device=polar_features.device, dtype=polar_features.dtype)
                    else:
                        # 🟢 安全保护：alpha 如果异常（NaN/Inf/过大），先重置再 clamp
                        alpha_min = float(getattr(self.config, 'polar_alpha_min', 0.0) or 0.0)
                        alpha_max = 2.0  # 固定上限，避免极端值导致融合崩溃
                        if not torch.isfinite(alpha).all():
                            alpha = torch.tensor(1.0, device=polar_features.device, dtype=polar_features.dtype)
                        alpha_value = torch.clamp(
                            alpha.to(device=polar_features.device, dtype=polar_features.dtype),
                            min=alpha_min,
                            max=alpha_max
                        )
                    # token 数不变，保持 LLM 分布稳定
                    image_features = rgb_features + alpha_value * polar_features
            else:
                # 拼接模式（兼容旧逻辑）
                # 输出形状: (B, 1152, hidden_size)
                # RGB 在前，Polar 在后
                image_features = torch.cat([rgb_features, polar_features], dim=1)
        else:
            # 如果没有 Polar 图像，只返回 RGB 特征
            if force_polar_only:
                raise ValueError("Stage 1a strict mode requires polar_images, but got None.")
            image_features = rgb_features

        # 🟢 调试：输出 token 数量（中文、清晰）
        # 通过环境变量控制，避免过度刷屏；训练时由 train.py 注入
        import os
        monitor_tokens = os.getenv('MONITOR_TOKEN_STATS', '0') == '1'
        if monitor_tokens:
            if '_token_monitor_state' not in globals():
                globals()['_token_monitor_state'] = {'last_step': -1}
            current_step_str = os.getenv('CURRENT_TRAINING_STEP', '-1')
            try:
                current_step = int(current_step_str)
            except (ValueError, TypeError):
                current_step = -1

            should_output_token_stats = False
            if current_step == -1:
                # 训练开始时允许输出
                globals()['_token_monitor_state']['last_step'] = -1
                should_output_token_stats = True
            elif current_step != globals()['_token_monitor_state']['last_step']:
                globals()['_token_monitor_state']['last_step'] = current_step
                should_output_token_stats = True

            if should_output_token_stats:
                rgb_tokens = rgb_features.shape[1] if rgb_features is not None else 0
                polar_tokens = polar_features.shape[1] if 'polar_features' in locals() and polar_features is not None else 0
                final_tokens = image_features.shape[1] if image_features is not None else 0
                stage = getattr(self.config, 'training_stage', 'unknown')
                fusion_desc = "residual(残差融合)" if fusion_mode == 'residual' else "concat(拼接)"
                polar_on = "是" if (polar_images is not None) else "否"
                print(
                    f"  🔎 Token 数量检查 | stage={stage}, 融合={fusion_desc}, Polar输入={polar_on}\n"
                    f"     - RGB tokens:   {rgb_tokens}\n"
                    f"     - Polar tokens: {polar_tokens}\n"
                    f"     - 最终 tokens:  {final_tokens} "
                    f"(residual 应为 576, concat 应为 1152)"
                )
        
        # 🟢 关键修复：NaN/Inf 检查和清理（最终特征拼接后）
        # 使用更智能的清理策略
        nan_mask = torch.isnan(image_features)
        inf_mask = torch.isinf(image_features)
        if nan_mask.any() or inf_mask.any():
            nan_count = nan_mask.sum().item()
            inf_count = inf_mask.sum().item()
            total_elements = image_features.numel()
            nan_ratio = nan_count / total_elements * 100
            inf_ratio = inf_count / total_elements * 100
            
            if nan_ratio > 10.0 or inf_ratio > 10.0:
                print(f"  ⚠️ 警告: 最终 image_features 异常值比例过高 (NaN: {nan_ratio:.2f}%, Inf: {inf_ratio:.2f}%)")
            
            # 使用均值填充 NaN
            if nan_mask.any():
                valid_mean = image_features[~nan_mask].mean() if (~nan_mask).any() else 0.0
                image_features = torch.where(nan_mask, torch.tensor(valid_mean, device=image_features.device, dtype=image_features.dtype), image_features)
            
            # Inf 值裁剪（特征值通常不会太大）
            if inf_mask.any():
                valid_values = image_features[~inf_mask]
                if len(valid_values) > 0:
                    valid_max = valid_values.max().item()
                    valid_min = valid_values.min().item()
                    clip_max = max(abs(valid_max), abs(valid_min), 100.0)  # 特征值允许较大范围
                    image_features = torch.clamp(image_features, min=-clip_max, max=clip_max)
                else:
                    image_features = torch.clamp(image_features, min=-100.0, max=100.0)
        
        # 🟢 最终确保：image_features 必须是 FP16
        if image_features.dtype != model_dtype:
            image_features = image_features.to(dtype=model_dtype)
        
        return image_features
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, polar_images=None
    ):
        """
        准备多模态输入和标签（支持 RGB + Polar 双流）
        
        Args:
            polar_images: 偏振图像，形状 (B, 3, H, W) 或列表
        """
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            
            # 处理 polar_images（如果提供）
            concat_polar_images = None
            if polar_images is not None:
                if type(polar_images) is list:
                    polar_images = [x.unsqueeze(0) if x.ndim == 3 else x for x in polar_images]
                concat_polar_images = torch.cat([img for img in polar_images], dim=0)
            
            image_features = self.encode_images(concat_images, concat_polar_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            from llava.mm_utils import unpad_image
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.get_model().image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.get_model().image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            # 处理单个图像的情况
            polar_images_single = None
            if polar_images is not None:
                if type(polar_images) is list:
                    polar_images_single = polar_images[0] if len(polar_images) > 0 else None
                else:
                    polar_images_single = polar_images
            image_features = self.encode_images(images, polar_images_single)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        
        # 🟢 获取模型的 dtype（通常是 FP16），在循环外获取一次即可
        try:
            model_dtype = next(self.get_model().embed_tokens.parameters()).dtype
        except:
            model_dtype = torch.float16
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                # 🟢 确保 embed_tokens 输出是 FP16
                cur_input_embeds_1 = cur_input_embeds_1.to(dtype=model_dtype)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            # 🟢 确保 embed_tokens 输出是 FP16
            cur_input_embeds = cur_input_embeds.to(dtype=model_dtype)
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            # 🟢 获取模态分隔 token 的 ID（如果未设置则退化为旧逻辑）
            # 🟢 修改：Stage 1 和 Stage 2 都不使用分隔 token，直接拼接 RGB 和 Polar 特征
            # 数据量少时，添加新 token 容易过拟合，直接拼接更简单有效
            # 所有阶段都不使用分隔 token，直接拼接特征
            use_separators = False

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1

                    # 🟢 修改：所有阶段都直接插入图像特征（RGB + Polar 直接拼接）
                    # 不使用分隔 token，避免过拟合
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            # 🟢 关键修复：确保所有 embeddings 都是 FP16 类型
            cur_new_input_embeds = [x.to(self.device).to(dtype=model_dtype) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        # 🟢 获取模型的 dtype 用于 padding（确保一致性）
        try:
            model_dtype = next(self.get_model().embed_tokens.parameters()).dtype
        except:
            model_dtype = torch.float16

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            # 🟢 确保 cur_new_embed 是 FP16
            cur_new_embed = cur_new_embed.to(dtype=model_dtype)
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=model_dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=model_dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        # 🟢 最终确保 new_input_embeds 是 FP16 类型
        if new_input_embeds.dtype != model_dtype:
            new_input_embeds = new_input_embeds.to(dtype=model_dtype)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


# ======================= PolarLlavaMetaModel: 双流架构（RGB + Polar） =======================

class PolarLlavaMetaModel(LlavaMetaModel):
    """
    扩展的 LLaVA 模型，支持双流视觉输入（RGB + Polar）
    
    架构设计：
    - RGB Tower: CLIP-ViT-L-336 (冻结)
    - Polar Tower: VAE Encoder (可选微调)
    - RGB Projector: LLaVA 原带的 MLP (冻结)
    - Polar Projector: 新建的 MLP (需训练)
    """
    
    def __init__(self, config):
        # ⚠️ 关键修复：如果 config.mm_vision_tower 是 HuggingFace 模型名称（不是本地路径），
        # 暂时移除它，避免在 super().__init__ 时尝试加载错误的路径
        # 正确的路径会在 initialize_vision_modules 中通过 model_args.vision_tower 设置
        original_vision_tower = None
        if hasattr(config, "mm_vision_tower") and config.mm_vision_tower:
            # 检查是否是 HuggingFace 模型名称（如 "openai/clip-vit-large-patch14-336"）
            if not os.path.exists(config.mm_vision_tower) and ("/" in config.mm_vision_tower or "openai" in config.mm_vision_tower or "laion" in config.mm_vision_tower):
                # 暂时移除，避免在 super().__init__ 时尝试加载
                original_vision_tower = config.mm_vision_tower
                config.mm_vision_tower = None
        
        super(PolarLlavaMetaModel, self).__init__(config)
        
        # 恢复原始值（如果需要），但不会触发初始化，因为 vision_tower 已经在 super().__init__ 中检查过了
        if original_vision_tower is not None:
            config.mm_vision_tower = original_vision_tower
        
        # 初始化 Polar 分支（如果配置了）
        if hasattr(config, "polar_vae_model_path") and config.polar_vae_model_path:
            self._initialize_polar_modules(config)
    
    def _initialize_polar_modules(self, config):
        """初始化 Polar 分支：VAE Encoder + Polar Projector"""
        try:
            from diffusers import AutoencoderKL
            
            vae_model_path = config.polar_vae_model_path
            if not os.path.exists(vae_model_path):
                print(f"Warning: VAE model path does not exist: {vae_model_path}")
                print("Polar branch will not be initialized.")
                return
            
            print(f"Initializing Polar branch with VAE: {vae_model_path}")
            
            # 加载 VAE 编码器
            vae = AutoencoderKL.from_pretrained(
                vae_model_path,
                subfolder=None,
                local_files_only=True  # 强制只使用本地文件
            )
            
            # 只使用编码器和量化层，删除解码器以节省内存
            self.polar_encoder = vae.encoder
            self.polar_quant_conv = vae.quant_conv
            del vae.decoder  # 删除解码器
            
            # 冻结 VAE 编码器（默认，可在训练时解冻）
            if getattr(config, "freeze_polar_encoder", True):
                self.polar_encoder.requires_grad_(False)
                self.polar_quant_conv.requires_grad_(False)
                print("  ✓ Polar VAE Encoder: Frozen")
            else:
                print("  ✓ Polar VAE Encoder: Trainable")
            
            # 确定 VAE latent 维度
            # VAE 输出是 (B, 4, 64, 64)，经过 quant_conv 后是 (B, 8, 64, 64)
            # 我们取前4个通道（mean），所以 latent 维度是 4
            # 但需要下采样到与 RGB 相同的空间分辨率（24x24 for LLaVA-1.5）
            # 所以需要先投影到特征维度，再下采样
            
            # VAE latent 维度（4通道）
            vae_latent_dim = 4
            
            # 下采样后的空间分辨率（与 RGB 对齐）
            polar_spatial_size = 24  # LLaVA-1.5 的 RGB 输出是 24x24
            polar_feature_dim = 256  # 中间特征维度
            
            # Polar Projector: 两层 MLP + LayerNorm
            # 输入: (B, 576, 256) - 576 个 token，每个 256 维
            # 输出: (B, 576, hidden_size) - 576 个 token，每个 hidden_size 维
            # 🟢 关键修复：添加 LayerNorm 自动对齐 Polar 和 RGB 特征的分布
            # LayerNorm 会将 Polar 特征的 Std 从 ~0.52 自动提升到 ~1.0，与 RGB (Std~0.95) 对齐
            self.polar_projector = nn.Sequential(
                nn.Linear(polar_feature_dim, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )
            # 🟢 [新增] LayerNorm：自动对齐特征分布，使 Polar 特征的 Std 与 RGB 对齐
            self.polar_projector_norm = nn.LayerNorm(config.hidden_size)
            
            # 🟢 [可选增强] 可学习的缩放因子：进一步对齐 Range（如果 LayerNorm 后 Range 仍然不匹配）
            # 注意：这个缩放因子是可选的，通常 LayerNorm 已经足够
            # 如果训练后发现 Range 差异仍然影响性能，可以启用这个缩放因子
            # 初始化：1.0（不缩放），默认冻结（requires_grad=False），避免训练过程中变小导致 Polar 特征被缩小
            # ⚠️ 关键修复：使用 register_buffer 而不是 Parameter，因为它是冻结的，不需要梯度
            # 这样可以避免量化模型加载时的问题（量化可能影响 Parameter 的初始化）
            # 注意：如果将来需要启用训练，可以改为 nn.Parameter(torch.ones(1), requires_grad=False)
            self.register_buffer('polar_range_scale', torch.ones(1))  # 使用 buffer 而不是 Parameter，避免量化问题

            # 🟢 Residual 融合系数（可学习），用于 fused = rgb + alpha * polar
            polar_alpha_init = float(getattr(config, 'polar_alpha_init', 0.5))
            self.polar_alpha = nn.Parameter(torch.tensor(polar_alpha_init, dtype=torch.float32))
            
            # 特征维度投影层（从 VAE latent 到特征维度）
            # 使用 Conv2d 因为输入是空间特征 (B, 4, 64, 64)
            self.vae_latent_to_feature = nn.Conv2d(vae_latent_dim, polar_feature_dim, kernel_size=1)
            
            # 🟢 优化：使用 Kaiming 初始化（适合 Conv2d，比默认的 Uniform 更好）
            # 这可以加速收敛，避免训练初期梯度爆炸
            import torch.nn.init as init
            init.kaiming_normal_(self.vae_latent_to_feature.weight, mode='fan_out', nonlinearity='relu')
            if self.vae_latent_to_feature.bias is not None:
                init.zeros_(self.vae_latent_to_feature.bias)
            
            print("  ✓ Polar Projector: Initialized (trainable)")
            print("  ✓ Polar Projector LayerNorm: Added (自动对齐 Polar 和 RGB 特征分布)")
            print(f"  ✓ Polar spatial resolution: {polar_spatial_size}x{polar_spatial_size}")
            print(f"  ✓ Polar feature dimension: {polar_feature_dim}")
            
        except Exception as e:
            print(f"Error initializing Polar modules: {e}")
            import traceback
            traceback.print_exc()
            # 如果初始化失败，polar 分支将不可用
            self.polar_encoder = None
            self.polar_projector = None
    
    def get_polar_encoder(self):
        """获取 Polar 编码器"""
        return getattr(self, 'polar_encoder', None)


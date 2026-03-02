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


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, polar_vae_model_path=None, max_memory=None, use_multi_gpu=False, **kwargs):
    # 双卡支持：检测GPU数量并配置 device_map
    num_gpus = 0
    if device == "cuda" and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            print(f"  ✓ 检测到 {num_gpus} 个 GPU")
    
    # 🟢 优化：双卡 4090 优先使用 FP16 全精度 + 多GPU自动分布
    if device == "cuda" and num_gpus > 1:
        # 检测到多GPU时，优先使用多GPU模式（除非强制使用 4-bit 量化）
        if use_multi_gpu or not load_4bit:
            # 多GPU模式或FP16模式：使用 "auto" 自动分布
            if device_map == "auto" or (isinstance(device_map, dict) and device_map.get("") == 0):
                device_map = "auto"
                print(f"  ✓ 多GPU模式：使用 device_map='auto' 自动分布到 {num_gpus} 个 GPU")
                # 🟢 关键：只有在未传入 max_memory 时才设置默认值
                # 如果 inference.py 已经传入了 max_memory，则使用传入的值（优先级更高）
                if max_memory is None and num_gpus > 1:
                    # 双卡 4090：每个GPU限制为23GB（留出1GB余量，显存充裕）
                    max_memory = {i: "23GB" for i in range(num_gpus)}
                    print(f"  ✓ 显存限制（默认值）: 每个GPU最多23GB（双卡 4090 显存充裕）")
                elif max_memory is not None:
                    print(f"  ✓ 显存限制（已传入）: {max_memory}（使用 inference.py 传入的值）")
        elif load_4bit:
            # 4-bit 量化 + 多卡：bitsandbytes 对多卡支持有限，使用单卡
            device_map = {"": 0}
            print(f"  ⚠️  4-bit 量化模式：使用单GPU (device_map={{'': 0}})")
            print(f"  ⚠️  建议：关闭 load_4bit 以充分利用双卡显存")
    elif load_4bit and device == "cuda" and device_map == "auto":
        # 单GPU模式：4-bit 量化时使用 {"": 0} 避免显存峰值
        device_map = {"": 0}
        print(f"  ⚠️  4-bit 量化模式（单GPU）：将 device_map 从 'auto' 改为 {{'': 0}} 以避免显存峰值")
    
    kwargs = {"device_map": device_map, **kwargs}
    
    # 🟢 关键：如果提供了 max_memory，添加到 kwargs 中
    # 这确保了 HuggingFace 的 from_pretrained 能够正确使用显存限制
    if max_memory is not None:
        kwargs['max_memory'] = max_memory
        print(f"  ✓ 显存限制已添加到 kwargs: {max_memory}")
        print(f"  ✓ 这将传递给 from_pretrained，确保双卡平衡分配")
    else:
        print(f"  ℹ️  未设置 max_memory，将使用 accelerate 的默认分配策略")
    
    # 多GPU模式：确保 vision_tower 也分布在合适的GPU上
    if use_multi_gpu and num_gpus > 1:
        print(f"  ✓ 多GPU模式已启用，模型将自动分布到 {num_gpus} 个 GPU")

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        # ⚠️ 修复：新版本 transformers 不允许同时传递 load_in_4bit 和 quantization_config
        # 只使用 quantization_config，不要同时设置 load_in_4bit
        # ⚠️ 关键：确保 device_map 是字典格式（{"": 0}），这样 bitsandbytes 才能正确流式量化
        if isinstance(kwargs.get('device_map'), dict):
            print(f"  ✓ 4-bit 量化配置：device_map={kwargs.get('device_map')}，将使用流式量化")
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    # 检查是否使用 Polar 模型
    use_polar = polar_vae_model_path is not None and os.path.exists(polar_vae_model_path)
    if use_polar:
        print(f"Detected Polar VAE model path: {polar_vae_model_path}")
        print("Will load PolarLlavaLlamaForCausalLM instead of LlavaLlamaForCausalLM")
    
    if 'llava' in model_name.lower() or use_polar:
        # Load LLaVA model (or Polar LLaVA model)
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            if use_polar:
                from llava.model.language_model.llava_llama import PolarLlavaConfig
                lora_cfg_pretrained = PolarLlavaConfig.from_pretrained(model_path, local_files_only=True)
                lora_cfg_pretrained.polar_vae_model_path = polar_vae_model_path
            else:
                from llava.model.language_model.llava_llama import LlavaConfig
                lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, local_files_only=True)
            print('Loading LLaVA from base model...')
            if use_polar:
                from llava.model.language_model.llava_llama import PolarLlavaLlamaForCausalLM
                kwargs['local_files_only'] = True  # 强制只使用本地文件
                # 🔴 关键修复：对于 Polar 模型，FP16 模式下必须设为 False
                # 这样模型会先在 CPU RAM 中完整初始化，避开 Meta Tensor 错误
                print(f"  ⚡️ Polar 模型加载策略：low_cpu_mem_usage=False (避免 Meta Tensor 错误)")
                model = PolarLlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=False, config=lora_cfg_pretrained, **kwargs)
            else:
                kwargs['local_files_only'] = True  # 强制只使用本地文件
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path, local_files_only=True)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, local_files_only=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
                kwargs['local_files_only'] = True  # 强制只使用本地文件
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, local_files_only=True)
                if use_polar:
                    from llava.model.language_model.llava_llama import PolarLlavaConfig, PolarLlavaLlamaForCausalLM
                    cfg_pretrained = PolarLlavaConfig.from_pretrained(model_path, local_files_only=True)
                    cfg_pretrained.polar_vae_model_path = polar_vae_model_path
                    kwargs['local_files_only'] = True  # 强制只使用本地文件
                    # 🔴 关键修复：对于 Polar 模型，FP16 模式下必须设为 False
                    # 这样模型会先在 CPU RAM 中完整初始化，避开 Meta Tensor 错误
                    print(f"  ⚡️ Polar 模型加载策略：low_cpu_mem_usage=False (避免 Meta Tensor 错误)")
                    model = PolarLlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=False, config=cfg_pretrained, **kwargs)
                else:
                    cfg_pretrained = AutoConfig.from_pretrained(model_path, local_files_only=True)
                    kwargs['local_files_only'] = True  # 强制只使用本地文件
                    model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
                kwargs['local_files_only'] = True  # 强制只使用本地文件
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                kwargs['local_files_only'] = True  # 强制只使用本地文件
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
                if use_polar:
                    from llava.model.language_model.llava_llama import PolarLlavaConfig, PolarLlavaLlamaForCausalLM
                    # 尝试加载配置
                    try:
                        cfg_pretrained = PolarLlavaConfig.from_pretrained(model_path, local_files_only=True)
                        cfg_pretrained.polar_vae_model_path = polar_vae_model_path
                        kwargs['local_files_only'] = True  # 强制只使用本地文件
                        # 🔴 关键修复：对于 Polar 模型，FP16 模式下必须设为 False
                        # 这样模型会先在 CPU RAM 中完整初始化，避开 Meta Tensor 错误
                        print(f"  ⚡️ Polar 模型加载策略：low_cpu_mem_usage=False (避免 Meta Tensor 错误)")
                        model = PolarLlavaLlamaForCausalLM.from_pretrained(
                            model_path,
                            low_cpu_mem_usage=False,
                            config=cfg_pretrained,
                            **kwargs
                        )
                    except:
                        # 如果配置加载失败，使用默认配置
                        cfg_pretrained = PolarLlavaConfig.from_pretrained(
                            model_base if model_base else model_path,
                            local_files_only=True
                        )
                        cfg_pretrained.polar_vae_model_path = polar_vae_model_path
                        kwargs['local_files_only'] = True  # 强制只使用本地文件
                        # 🔴 关键修复：对于 Polar 模型，FP16 模式下必须设为 False
                        # 这样模型会先在 CPU RAM 中完整初始化，避开 Meta Tensor 错误
                        print(f"  ⚡️ Polar 模型加载策略：low_cpu_mem_usage=False (避免 Meta Tensor 错误)")
                        model = PolarLlavaLlamaForCausalLM.from_pretrained(
                            model_base if model_base else model_path,
                            low_cpu_mem_usage=False,
                            config=cfg_pretrained,
                            **kwargs
                        )
                else:
                    kwargs['local_files_only'] = True  # 强制只使用本地文件
                    model = LlavaLlamaForCausalLM.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        **kwargs
                    )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, local_files_only=True)
            kwargs['local_files_only'] = True  # 强制只使用本地文件
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path, local_files_only=True)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
                kwargs['local_files_only'] = True  # 强制只使用本地文件
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
                kwargs['local_files_only'] = True  # 强制只使用本地文件
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower() or use_polar:
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            # 多GPU模式：vision_tower 也使用 "auto" 自动分布，或放在 GPU 0
            if use_multi_gpu and num_gpus > 1 and device_map == "auto":
                # 多GPU模式：vision_tower 放在 GPU 0（与模型的第一层在同一GPU）
                vision_tower.load_model(device_map="cuda:0")
            else:
                vision_tower.load_model(device_map=device_map)
        
        # ⚠️ 修复：如果 device_map 是字典（如 {"": 0}），需要正确提取设备
        # vision_tower.to() 不接受字典，需要提取实际的设备字符串或整数
        # 多GPU模式下，vision_tower 放在 GPU 0（与模型的第一层在同一GPU）
        if device_map == 'auto' and use_multi_gpu and num_gpus > 1:
            # 多GPU模式：vision_tower 放在 GPU 0
            vision_tower.to(device="cuda:0", dtype=torch.float16)
            print(f"  ✓ Vision Tower 已放置在 GPU 0（多GPU模式）")
        elif device_map != 'auto':
            if isinstance(device_map, dict):
                # 从 device_map 字典中提取设备（通常是 {"": 0} 或 {"": "cuda:0"}）
                device_for_vision = list(device_map.values())[0] if device_map else device
                if isinstance(device_for_vision, int):
                    device_for_vision = f"cuda:{device_for_vision}"
                elif device_for_vision == "cuda":
                    device_for_vision = "cuda:0"
                vision_tower.to(device=device_for_vision, dtype=torch.float16)
            else:
                vision_tower.to(device=device_map, dtype=torch.float16)
        else:
            # 单GPU模式：vision_tower 放在 GPU 0
            vision_tower.to(device="cuda:0", dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len





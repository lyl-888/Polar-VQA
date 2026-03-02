import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    logger,
)

# 梯度监控辅助函数（直接从 train.py 复制逻辑，避免循环导入）
def _compute_gradient_norms_inline(model, step, debug=False):
    """计算梯度范数的辅助函数（内联版本，避免导入问题）"""
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
    
    # 方法3: 尝试直接访问 polar_projector
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
    
    for name, p in model.named_parameters():
        if p.requires_grad:
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

def _print_gradient_analysis_inline(step, grad_info):
    """打印梯度分析的辅助函数（内联版本）"""
    print(f"\n{'=' * 60}")
    print(f"[Step {step}] 📊 梯度范数分析 (Gradient Norm Analysis)")
    print(f"{'=' * 60}")
    
    # 1. Polar Projector 梯度
    if grad_info['polar_projector_found']:
        if grad_info['polar_norm'] is not None and grad_info['polar_count'] > 0:
            print(f"  🌊 Polar Projector Grad Norm: {grad_info['polar_norm']:.6f} (有梯度参数: {grad_info['polar_count']}/{grad_info['polar_params_with_grad']})")
        else:
            print(f"  🌊 Polar Projector Grad Norm: 0.000000 ⚠️  警告: 所有参数都没有梯度！")
    else:
        print(f"  🌊 Polar Projector: 未找到")
    
    # 2. LLM (LoRA) 梯度
    if grad_info['lora_norm'] is not None and grad_info['lora_count'] > 0:
        print(f"  🧠 LLM (LoRA) Grad Norm:      {grad_info['lora_norm']:.6f} (有梯度参数: {grad_info['lora_count']}/{grad_info['lora_params_with_grad']})")
        if grad_info['lora_param_names']:
            print(f"     示例参数: {grad_info['lora_param_names'][0]}")
    else:
        print(f"  🧠 LLM (LoRA) Grad Norm:      0.000000 (未找到 LoRA 参数或未启用 LoRA)")
    
    # 3. 比率分析
    if grad_info['lora_norm'] is not None and grad_info['lora_norm'] > 0 and \
       grad_info['polar_norm'] is not None and grad_info['polar_norm'] > 0:
        ratio = grad_info['polar_norm'] / grad_info['lora_norm']
        print(f"  ⚖️  Ratio (Polar / LLM):      {ratio:.4f}")
        
        if ratio >= 0.5 and ratio <= 2.0:
            print(f"  ✅ 状态: 健康 (Healthy) - Polar 和 LLM 梯度在同一数量级")
        elif ratio < 0.001:
            print(f"  ⚠️  状态: 边缘化 (Marginalized) - Polar 梯度极小，可能被忽略")
            print(f"     建议: 检查 Stage 1 训练效果，或增大 Polar Projector 学习率")
        elif ratio == 0.0:
            print(f"  ❌ 状态: 梯度断裂 (Broken) - Polar 梯度为 0，检查代码逻辑")
        else:
            print(f"  ⚠️  状态: 梯度不平衡 - Polar 梯度 {'过大' if ratio > 2.0 else '过小'}")
    elif grad_info['polar_norm'] == 0.0 and grad_info['lora_norm'] is not None and grad_info['lora_norm'] > 0:
        print(f"  ❌ 状态: 梯度断裂 (Broken) - Polar 梯度为 0，检查 requires_grad 设置")
    elif grad_info['lora_norm'] is None or grad_info['lora_norm'] == 0.0:
        print(f"  ℹ️  状态: 未启用 LoRA 或 LoRA 参数未更新")
    
    print(f"{'=' * 60}\n")

# ALL_LAYERNORM_LAYERS 在新版 transformers 中已移除，使用兼容方式
try:
    from transformers.trainer import ALL_LAYERNORM_LAYERS
except ImportError:
    # transformers 4.40+ 中已移除，使用 nn.LayerNorm 作为替代
    ALL_LAYERNORM_LAYERS = (nn.LayerNorm,)
from typing import List, Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=False):
    """
    获取非 LoRA 参数的状态字典（用于保存 non_lora_trainables.bin）
    
    Args:
        named_params: 模型的所有命名参数
        require_grad_only: 是否只保存 requires_grad=True 的参数（默认 False，强制保存所有）
    
    Returns:
        非 LoRA 参数的状态字典
    """
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 用于跟踪梯度累积的计数器
        self._grad_accumulation_counter = 0
        # 标记静态图是否已设置
        self._static_graph_set = False
    
    def _wrap_model(self, model, training=True, dataloader=None):
        """
        重写 _wrap_model 方法，在 DDP 包装后设置静态图（用于解决 DDP + Gradient Checkpointing 冲突）
        """
        # 先调用父类方法进行 DDP 包装
        wrapped_model = super()._wrap_model(model, training=training, dataloader=dataloader)
        
        # 如果使用了 Gradient Checkpointing 和 DDP，按需设置静态图
        if (self.args.gradient_checkpointing and 
            hasattr(self.args, 'world_size') and self.args.world_size > 1):
            enable_static_graph = os.getenv("LLAVA_DDP_STATIC_GRAPH", "0") == "1"
            if not enable_static_graph:
                if self.args.local_rank == 0 or self.args.local_rank == -1:
                    logger.info("ℹ️  已禁用 DDP 静态图（避免 DDP + Gradient Checkpointing 内部断言错误）。如需启用，设置环境变量 LLAVA_DDP_STATIC_GRAPH=1")
                return wrapped_model
            
            # 递归查找并设置 DDP 静态图
            def find_and_set_ddp_static_graph(model_obj, depth=0, path="", visited=None):
                """递归查找 DDP 包装并设置静态图"""
                if visited is None:
                    visited = set()
                
                if depth > 15:  # 增加深度限制
                    return False
                
                # 防止循环引用
                obj_id = id(model_obj)
                if obj_id in visited:
                    return False
                visited.add(obj_id)
                
                import torch.nn.parallel
                
                # 方法1: 直接检查是否是 DistributedDataParallel
                if isinstance(model_obj, torch.nn.parallel.DistributedDataParallel):
                    if hasattr(model_obj, '_set_static_graph'):
                        model_obj._set_static_graph()
                        if self.args.local_rank == 0 or self.args.local_rank == -1:
                            logger.info(f"✓ [_wrap_model] 已设置静态图模式（DDP + Gradient Checkpointing 兼容，路径: {path}）")
                        return True
                
                # 方法2: 检查是否有 _set_static_graph 方法（可能是 Accelerate 包装）
                if hasattr(model_obj, '_set_static_graph'):
                    try:
                        model_obj._set_static_graph()
                        if self.args.local_rank == 0 or self.args.local_rank == -1:
                            logger.info(f"✓ [_wrap_model] 已设置静态图模式（Accelerate/DDP + Gradient Checkpointing 兼容，路径: {path}）")
                        return True
                    except Exception as e:
                        if self.args.local_rank == 0 or self.args.local_rank == -1:
                            logger.debug(f"[_wrap_model] 尝试设置静态图失败（路径: {path}）: {e}")
                
                # 方法3: 递归检查所有可能的属性（更彻底的搜索）
                attrs_to_check = ['module', 'base_model', 'model', '_orig_mod', 'wrapped_model']
                for attr_name in attrs_to_check:
                    if hasattr(model_obj, attr_name):
                        attr_obj = getattr(model_obj, attr_name)
                        if find_and_set_ddp_static_graph(attr_obj, depth + 1, f"{path}.{attr_name}", visited):
                            return True
                
                # 方法4: 检查 __dict__ 中的所有属性（更彻底的搜索）
                if hasattr(model_obj, '__dict__'):
                    for attr_name, attr_obj in model_obj.__dict__.items():
                        if isinstance(attr_obj, (torch.nn.Module, torch.nn.parallel.DistributedDataParallel)):
                            if find_and_set_ddp_static_graph(attr_obj, depth + 1, f"{path}.{attr_name}", visited):
                                return True
                
                return False
            
            # 尝试递归查找并设置
            if find_and_set_ddp_static_graph(wrapped_model, path="model"):
                self._static_graph_set = True
            else:
                if self.args.local_rank == 0 or self.args.local_rank == -1:
                    logger.warning("⚠️  无法找到 DDP 包装，静态图设置可能失败。如果遇到 DDP 错误，请考虑禁用 gradient_checkpointing")
        
        return wrapped_model

    def _get_train_sampler(self, dataset=None) -> Optional[torch.utils.data.Sampler]:
        # 兼容新版本 transformers：如果传入了 dataset 参数，使用它；否则使用 self.train_dataset
        train_dataset = dataset if dataset is not None else self.train_dataset
        
        if train_dataset is None or not has_length(train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            # 调用父类方法时也传入 dataset 参数（如果存在）
            if dataset is not None:
                return super()._get_train_sampler(dataset)
            else:
                return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                # 同时包含 mm_projector、polar_projector 和 vae_latent_to_feature，使用相同的学习率
                projector_parameters = [name for name, _ in opt_model.named_parameters() 
                                        if "mm_projector" in name or "polar_projector" in name or "vae_latent_to_feature" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        # 🔧 注意：新版本 transformers 的父类 _save_checkpoint 可能不接受 metrics 参数
        # 我们保留 metrics 参数以兼容旧版本，但在调用父类时根据实际情况处理
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            # 🟢 关键修复：同时保存 polar_projector 和 vae_latent_to_feature（如果存在）
            base_model = self.model.get_model()
            if hasattr(base_model, 'polar_projector'):
                keys_to_match.append('polar_projector')
            if hasattr(base_model, 'vae_latent_to_feature'):
                keys_to_match.append('vae_latent_to_feature')
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            # 🔧 修复：新版本 transformers 的 _save_checkpoint 只接受 (model, trial) 两个参数
            # metrics 参数已被移除，直接调用父类方法
            super(LLaVATrainer, self)._save_checkpoint(model, trial)
            
            # 🟢 关键修复：在每个 checkpoint 也保存 non_lora_trainables.bin（包含 polar_projector 和 vae_latent_to_feature）
            # 这样可以确保即使训练中断，也能恢复这些关键权重
            if getattr(self.args, 'lora_enable', False):
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                run_dir = self._get_output_dir(trial=trial)
                checkpoint_dir = os.path.join(run_dir, checkpoint_folder)
                
                # 获取非 LoRA 参数（包含 polar_projector 和 vae_latent_to_feature）
                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                    model.named_parameters(),
                    require_grad_only=False  # 强制保存所有非 LoRA 参数，无论 requires_grad 状态
                )
                
                # 🟢 强制保存 Polar 核心模块（无论 requires_grad 状态）
                base_model = model.get_model() if hasattr(model, 'get_model') else model
                polar_keys_to_save = ["polar_projector", "vae_latent_to_feature"]
                forced_saved_count = 0
                polar_params_info = []  # 用于调试
                for name, param in model.named_parameters():
                    if any(k in name for k in polar_keys_to_save):
                        # 无论是否已在 non_lora_state_dict 中，都强制保存（覆盖）
                        non_lora_state_dict[name] = maybe_zero_3(param, ignore_status=True).cpu()
                        forced_saved_count += 1
                        # 记录参数信息用于调试
                        param_size = param.numel()
                        polar_params_info.append(f"{name}: {param_size:,} elements (shape: {tuple(param.shape)})")
                
                # 🟢 强制保存 resize 后的 embedding（如果词表被 resize）
                if hasattr(model, 'get_input_embeddings') and hasattr(model, 'config'):
                    try:
                        tokenizer_vocab_size = getattr(model.config, 'vocab_size', None)
                        if tokenizer_vocab_size and tokenizer_vocab_size > 32000:
                            # 词表被 resize，强制保存 embed_tokens 和 lm_head
                            for name, param in model.named_parameters():
                                if ("embed_tokens" in name or "lm_head" in name):
                                    non_lora_state_dict[name] = maybe_zero_3(param, ignore_status=True).cpu()
                    except:
                        pass  # 如果获取失败，跳过（不影响主要功能）
                
                # 保存到 checkpoint 目录
                if self.args.local_rank == 0 or self.args.local_rank == -1:
                    # 确保 checkpoint 目录存在（父类 _save_checkpoint 应该已经创建，但为了安全起见再检查一次）
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    non_lora_path = os.path.join(checkpoint_dir, 'non_lora_trainables.bin')
                    torch.save(non_lora_state_dict, non_lora_path)
                    if forced_saved_count > 0:
                        # 计算 Polar 参数的总元素数
                        polar_total_elements = sum(
                            param.numel() for name, param in non_lora_state_dict.items()
                            if any(k in name for k in polar_keys_to_save)
                        )
                        logger.info(f"  ✓ Checkpoint {checkpoint_folder}: 已保存 non_lora_trainables.bin")
                        logger.info(f"    - 总参数对象数: {len(non_lora_state_dict)}")
                        logger.info(f"    - Polar 核心参数: {forced_saved_count} 个对象, {polar_total_elements:,} 个元素")
                        if polar_params_info and (self.args.local_rank == 0 or self.args.local_rank == -1):
                            # 只在 rank 0 输出详细信息，避免日志过多
                            for info in polar_params_info:
                                logger.debug(f"      {info}")
                    else:
                        logger.info(f"  ✓ Checkpoint {checkpoint_folder}: 已保存 non_lora_trainables.bin (包含 {len(non_lora_state_dict)} 个参数)")

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        重写 training_step，在第一次调用时尝试设置静态图（备用方案）
        """
        # 如果还没设置静态图，且使用了 Gradient Checkpointing 和 DDP，尝试设置
        enable_static_graph = os.getenv("LLAVA_DDP_STATIC_GRAPH", "0") == "1"
        if (enable_static_graph and not self._static_graph_set and 
            self.args.gradient_checkpointing and 
            hasattr(self.args, 'world_size') and self.args.world_size > 1):
            
            def find_and_set_ddp_static_graph(model_obj, depth=0, path="", visited=None):
                """递归查找 DDP 包装并设置静态图"""
                if visited is None:
                    visited = set()
                
                if depth > 15:
                    return False
                
                # 防止循环引用
                obj_id = id(model_obj)
                if obj_id in visited:
                    return False
                visited.add(obj_id)
                
                import torch.nn.parallel
                
                # 检查是否是 DistributedDataParallel
                if isinstance(model_obj, torch.nn.parallel.DistributedDataParallel):
                    if hasattr(model_obj, '_set_static_graph'):
                        model_obj._set_static_graph()
                        if self.args.local_rank == 0 or self.args.local_rank == -1:
                            logger.info(f"✓ [training_step] 已设置静态图模式（DDP + Gradient Checkpointing 兼容，路径: {path}）")
                        return True
                
                # 检查是否有 _set_static_graph 方法
                if hasattr(model_obj, '_set_static_graph'):
                    try:
                        model_obj._set_static_graph()
                        if self.args.local_rank == 0 or self.args.local_rank == -1:
                            logger.info(f"✓ [training_step] 已设置静态图模式（Accelerate/DDP + Gradient Checkpointing 兼容，路径: {path}）")
                        return True
                    except Exception as e:
                        if self.args.local_rank == 0 or self.args.local_rank == -1:
                            logger.debug(f"[training_step] 尝试设置静态图失败（路径: {path}）: {e}")
                
                # 递归检查所有可能的属性
                attrs_to_check = ['module', 'base_model', 'model', '_orig_mod', 'wrapped_model']
                for attr_name in attrs_to_check:
                    if hasattr(model_obj, attr_name):
                        attr_obj = getattr(model_obj, attr_name)
                        if find_and_set_ddp_static_graph(attr_obj, depth + 1, f"{path}.{attr_name}", visited):
                            return True
                
                # 检查 __dict__ 中的所有属性
                if hasattr(model_obj, '__dict__'):
                    for attr_name, attr_obj in model_obj.__dict__.items():
                        if isinstance(attr_obj, (torch.nn.Module, torch.nn.parallel.DistributedDataParallel)):
                            if find_and_set_ddp_static_graph(attr_obj, depth + 1, f"{path}.{attr_name}", visited):
                                return True
                
                return False
            
            # 尝试在传入的 model 上设置
            if find_and_set_ddp_static_graph(model, path="model"):
                self._static_graph_set = True
        
        # 调用父类的 training_step
        if num_items_in_batch is not None:
            return super().training_step(model, inputs, num_items_in_batch)
        return super().training_step(model, inputs)
        
    def optimizer_step(self, *args, **kwargs):
        """禁用额外的梯度监控，直接使用父类的 optimizer_step。"""
        return super().optimizer_step(*args, **kwargs)
    
    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        """
        重写 train 方法，在训练开始前确保静态图已设置（用于解决 DDP + Gradient Checkpointing 冲突）
        并处理 checkpoint 加载时的 embedding 大小不匹配问题
        """
        enable_static_graph = os.getenv("LLAVA_DDP_STATIC_GRAPH", "0") == "1"
        # 🟢 关键修复：在训练开始前，强制查找并设置 DDP 静态图
        if (enable_static_graph and self.args.gradient_checkpointing and 
            hasattr(self.args, 'world_size') and self.args.world_size > 1 and
            not self._static_graph_set):
            
            def find_and_set_ddp_static_graph_force(model_obj, depth=0, path=""):
                """强制递归查找 DDP 包装并设置静态图"""
                if depth > 15:  # 增加深度限制
                    return False
                
                import torch.nn.parallel
                
                # 方法1: 直接检查是否是 DistributedDataParallel
                if isinstance(model_obj, torch.nn.parallel.DistributedDataParallel):
                    if hasattr(model_obj, '_set_static_graph'):
                        model_obj._set_static_graph()
                        if self.args.local_rank == 0 or self.args.local_rank == -1:
                            logger.info(f"✓ [train()] 已设置静态图模式（DDP + Gradient Checkpointing 兼容，路径: {path}）")
                        return True
                
                # 方法2: 检查是否有 _set_static_graph 方法（可能是 Accelerate 包装）
                if hasattr(model_obj, '_set_static_graph'):
                    try:
                        model_obj._set_static_graph()
                        if self.args.local_rank == 0 or self.args.local_rank == -1:
                            logger.info(f"✓ [train()] 已设置静态图模式（Accelerate/DDP + Gradient Checkpointing 兼容，路径: {path}）")
                        return True
                    except Exception as e:
                        if self.args.local_rank == 0 or self.args.local_rank == -1:
                            logger.debug(f"[train()] 尝试设置静态图失败（路径: {path}）: {e}")
                
                # 方法3: 递归检查所有可能的属性
                attrs_to_check = ['module', 'base_model', 'model', '_orig_mod', 'wrapped_model']
                for attr_name in attrs_to_check:
                    if hasattr(model_obj, attr_name):
                        attr_obj = getattr(model_obj, attr_name)
                        if find_and_set_ddp_static_graph_force(attr_obj, depth + 1, f"{path}.{attr_name}"):
                            return True
                
                # 方法4: 检查 __dict__ 中的所有属性（更彻底的搜索）
                if hasattr(model_obj, '__dict__'):
                    for attr_name, attr_obj in model_obj.__dict__.items():
                        if isinstance(attr_obj, (torch.nn.Module, torch.nn.parallel.DistributedDataParallel)):
                            if find_and_set_ddp_static_graph_force(attr_obj, depth + 1, f"{path}.{attr_name}"):
                                return True
                
                return False
            
            # 尝试在 self.model 上设置
            if find_and_set_ddp_static_graph_force(self.model, path="self.model"):
                self._static_graph_set = True
            else:
                if self.args.local_rank == 0 or self.args.local_rank == -1:
                    logger.warning("⚠️  [train()] 无法找到 DDP 包装，静态图设置可能失败。如果遇到 DDP 错误，请考虑禁用 gradient_checkpointing")
        
        # 🟢 关键修复：在加载 checkpoint 之前，处理 embedding 大小不匹配问题
        # 如果 checkpoint 中的 embedding 大小与当前模型不匹配（例如 checkpoint 有 32002 个 token，但当前模型只有 32000），
        # 需要先调整 checkpoint 中的权重，只加载匹配的部分
        if resume_from_checkpoint is not None:
            from pathlib import Path
            
            checkpoint_path = Path(resume_from_checkpoint) if isinstance(resume_from_checkpoint, (str, Path)) else None
            if checkpoint_path and checkpoint_path.exists():
                # 检查 checkpoint 中是否有 adapter_model.safetensors 或 adapter_model.bin
                adapter_file = None
                if (checkpoint_path / "adapter_model.safetensors").exists():
                    adapter_file = checkpoint_path / "adapter_model.safetensors"
                elif (checkpoint_path / "adapter_model.bin").exists():
                    adapter_file = checkpoint_path / "adapter_model.bin"
                
                if adapter_file:
                    try:
                        # 获取当前模型的 embedding 大小
                        current_embed_size = None
                        current_lm_head_size = None
                        tie_word_embeddings = False
                        
                        # 检查是否 tie_word_embeddings
                        if hasattr(self.model, 'config'):
                            tie_word_embeddings = getattr(self.model.config, 'tie_word_embeddings', False)
                        elif hasattr(self.model, 'get_model') and hasattr(self.model.get_model(), 'config'):
                            tie_word_embeddings = getattr(self.model.get_model().config, 'tie_word_embeddings', False)
                        
                        if hasattr(self.model, 'get_input_embeddings'):
                            current_embed = self.model.get_input_embeddings()
                            if hasattr(current_embed, 'weight'):
                                current_embed_size = current_embed.weight.shape[0]
                        
                        if hasattr(self.model, 'get_output_embeddings'):
                            current_lm_head = self.model.get_output_embeddings()
                            if hasattr(current_lm_head, 'weight'):
                                current_lm_head_size = current_lm_head.weight.shape[0]
                        
                        # 如果 tie_word_embeddings=True，lm_head 与 embed_tokens 共享权重，不需要单独处理
                        if tie_word_embeddings:
                            current_lm_head_size = current_embed_size
                        
                        # 加载 checkpoint 的 adapter 权重
                        if adapter_file.suffix == '.safetensors':
                            from safetensors import safe_open
                            checkpoint_state_dict = {}
                            with safe_open(adapter_file, framework="pt", device="cpu") as f:
                                for key in f.keys():
                                    checkpoint_state_dict[key] = f.get_tensor(key)
                        else:
                            checkpoint_state_dict = torch.load(adapter_file, map_location="cpu")
                        
                        # 检查 embedding 大小是否不匹配
                        embed_key = None
                        lm_head_key = None
                        keys_to_remove = []
                        
                        for key in list(checkpoint_state_dict.keys()):
                            if 'embed_tokens.weight' in key:
                                embed_key = key
                                # 检查大小是否匹配
                                if current_embed_size and checkpoint_state_dict[key].shape[0] != current_embed_size:
                                    if self.args.local_rank == 0 or self.args.local_rank == -1:
                                        logger.warning(
                                            f"⚠️  Checkpoint embedding 大小 ({checkpoint_state_dict[key].shape[0]}) 与当前模型 ({current_embed_size}) 不匹配！"
                                        )
                                        logger.warning(
                                            f"   Stage 1 不训练 embedding，将跳过此权重"
                                        )
                                    keys_to_remove.append(key)
                            elif 'lm_head.weight' in key:
                                lm_head_key = key
                                # 检查大小是否匹配
                                if current_lm_head_size and checkpoint_state_dict[key].shape[0] != current_lm_head_size:
                                    if self.args.local_rank == 0 or self.args.local_rank == -1:
                                        logger.warning(
                                            f"⚠️  Checkpoint lm_head 大小 ({checkpoint_state_dict[key].shape[0]}) 与当前模型 ({current_lm_head_size}) 不匹配！"
                                        )
                                        logger.warning(
                                            f"   Stage 1 不训练 lm_head，将跳过此权重"
                                        )
                                    keys_to_remove.append(key)
                            
                        # 移除不匹配的键
                        for key in keys_to_remove:
                            del checkpoint_state_dict[key]
                        
                        # 如果移除了键，需要保存调整后的 adapter 文件和配置
                        if keys_to_remove:
                            # 创建临时 checkpoint 目录
                            import tempfile
                            import shutil
                            import json
                            temp_checkpoint_dir = Path(tempfile.mkdtemp())
                            
                            # 复制所有文件到临时目录
                            for file in checkpoint_path.iterdir():
                                if file.is_file() and file.name != adapter_file.name:
                                    shutil.copy2(file, temp_checkpoint_dir / file.name)
                            
                            # 处理 adapter_config.json（如果存在）
                            adapter_config_file = checkpoint_path / "adapter_config.json"
                            if adapter_config_file.exists():
                                with open(adapter_config_file, 'r') as f:
                                    adapter_config = json.load(f)
                                
                                # 如果 modules_to_save 包含被移除的模块，从配置中移除
                                if 'modules_to_save' in adapter_config:
                                    modules_to_save = adapter_config['modules_to_save']
                                    if isinstance(modules_to_save, list):
                                        # 移除 embed_tokens 和 lm_head（如果它们的大小不匹配）
                                        if embed_key and embed_key in keys_to_remove:
                                            modules_to_save = [m for m in modules_to_save if m != 'embed_tokens']
                                        if lm_head_key and lm_head_key in keys_to_remove:
                                            modules_to_save = [m for m in modules_to_save if m != 'lm_head']
                                        adapter_config['modules_to_save'] = modules_to_save
                                
                                # 保存调整后的配置
                                with open(temp_checkpoint_dir / "adapter_config.json", 'w') as f:
                                    json.dump(adapter_config, f, indent=2)
                            
                            # 保存调整后的 adapter 权重
                            if adapter_file.suffix == '.safetensors':
                                from safetensors.torch import save_file
                                save_file(checkpoint_state_dict, temp_checkpoint_dir / "adapter_model.safetensors")
                            else:
                                torch.save(checkpoint_state_dict, temp_checkpoint_dir / "adapter_model.bin")
                            
                            # 使用临时 checkpoint 路径
                            resume_from_checkpoint = str(temp_checkpoint_dir)
                            
                            if self.args.local_rank == 0 or self.args.local_rank == -1:
                                logger.info(f"✓ 已创建临时 checkpoint（已移除不匹配的 embedding 权重）: {resume_from_checkpoint}")
                                logger.info(f"   移除了 {len(keys_to_remove)} 个不匹配的权重键")
                    except Exception as e:
                        if self.args.local_rank == 0 or self.args.local_rank == -1:
                            logger.warning(f"⚠️  处理 checkpoint embedding 大小不匹配时出错: {e}")
                            logger.warning("   将尝试直接加载 checkpoint（可能会失败）")
        
        # 在训练开始前，再次尝试设置静态图（确保在 DDP 包装完成后）
        if (enable_static_graph and not self._static_graph_set and 
            self.args.gradient_checkpointing and 
            hasattr(self.args, 'world_size') and self.args.world_size > 1):
            
            def find_and_set_ddp_static_graph(model_obj, depth=0):
                """递归查找 DDP 包装并设置静态图"""
                if depth > 10:
                    return False
                
                import torch.nn.parallel
                
                # 检查是否是 DistributedDataParallel
                if isinstance(model_obj, torch.nn.parallel.DistributedDataParallel):
                    if hasattr(model_obj, '_set_static_graph'):
                        model_obj._set_static_graph()
                        if self.args.local_rank == 0 or self.args.local_rank == -1:
                            logger.info("✓ 已在 train() 中设置静态图模式（DDP + Gradient Checkpointing 兼容）")
                        return True
                
                # 检查是否有 _set_static_graph 方法
                if hasattr(model_obj, '_set_static_graph'):
                    try:
                        model_obj._set_static_graph()
                        if self.args.local_rank == 0 or self.args.local_rank == -1:
                            logger.info("✓ 已在 train() 中设置静态图模式（Accelerate/DDP + Gradient Checkpointing 兼容）")
                        return True
                    except Exception:
                        pass
                
                # 递归检查子模块
                for attr_name in ['module', 'base_model', 'model']:
                    if hasattr(model_obj, attr_name):
                        if find_and_set_ddp_static_graph(getattr(model_obj, attr_name), depth + 1):
                            return True
                
                return False
            
            # 尝试在 trainer 的模型上设置
            if hasattr(self, 'model') and self.model is not None:
                if find_and_set_ddp_static_graph(self.model):
                    self._static_graph_set = True
        
        # 调用父类的 train 方法
        # 🟢 关键修复：在调用 super().train() 之前，再次尝试设置静态图（以防 DDP 包装发生在 _wrap_model 之后）
        if (enable_static_graph and self.args.gradient_checkpointing and 
            hasattr(self.args, 'world_size') and self.args.world_size > 1 and
            not self._static_graph_set):
            
            def find_and_set_ddp_static_graph_final(model_obj, depth=0, path="", visited=None):
                """最终尝试：递归查找 DDP 包装并设置静态图"""
                if visited is None:
                    visited = set()
                
                if depth > 15:
                    return False
                
                obj_id = id(model_obj)
                if obj_id in visited:
                    return False
                visited.add(obj_id)
                
                import torch.nn.parallel
                
                if isinstance(model_obj, torch.nn.parallel.DistributedDataParallel):
                    if hasattr(model_obj, '_set_static_graph'):
                        model_obj._set_static_graph()
                        if self.args.local_rank == 0 or self.args.local_rank == -1:
                            logger.info(f"✓ [train() before super] 已设置静态图模式（路径: {path}）")
                        return True
                
                if hasattr(model_obj, '_set_static_graph'):
                    try:
                        model_obj._set_static_graph()
                        if self.args.local_rank == 0 or self.args.local_rank == -1:
                            logger.info(f"✓ [train() before super] 已设置静态图模式（路径: {path}）")
                        return True
                    except Exception:
                        pass
                
                attrs_to_check = ['module', 'base_model', 'model', '_orig_mod', 'wrapped_model']
                for attr_name in attrs_to_check:
                    if hasattr(model_obj, attr_name):
                        attr_obj = getattr(model_obj, attr_name)
                        if find_and_set_ddp_static_graph_final(attr_obj, depth + 1, f"{path}.{attr_name}", visited):
                            return True
                
                if hasattr(model_obj, '__dict__'):
                    for attr_name, attr_obj in model_obj.__dict__.items():
                        if isinstance(attr_obj, (torch.nn.Module, torch.nn.parallel.DistributedDataParallel)):
                            if find_and_set_ddp_static_graph_final(attr_obj, depth + 1, f"{path}.{attr_name}", visited):
                                return True
                
                return False
            
            if find_and_set_ddp_static_graph_final(self.model, path="self.model"):
                self._static_graph_set = True
        
        # 调用父类的 train 方法
        result = super().train(resume_from_checkpoint=resume_from_checkpoint, trial=trial, ignore_keys_for_eval=ignore_keys_for_eval, **kwargs)
                    
        # 🟢 关键修复：在 super().train() 之后，再次尝试设置静态图（以防 DDP 包装发生在 super().train() 内部）
        if (self.args.gradient_checkpointing and 
            hasattr(self.args, 'world_size') and self.args.world_size > 1 and
            not self._static_graph_set):
            
            def find_and_set_ddp_static_graph_after(model_obj, depth=0, path="", visited=None):
                """在 super().train() 之后再次尝试设置静态图"""
                if visited is None:
                    visited = set()
                
                if depth > 15:
                    return False
                
                obj_id = id(model_obj)
                if obj_id in visited:
                    return False
                visited.add(obj_id)
                
                import torch.nn.parallel
                
                if isinstance(model_obj, torch.nn.parallel.DistributedDataParallel):
                    if hasattr(model_obj, '_set_static_graph'):
                        model_obj._set_static_graph()
                        if self.args.local_rank == 0 or self.args.local_rank == -1:
                            logger.info(f"✓ [train() after super] 已设置静态图模式（路径: {path}）")
                        return True
                
                if hasattr(model_obj, '_set_static_graph'):
                    try:
                        model_obj._set_static_graph()
                        if self.args.local_rank == 0 or self.args.local_rank == -1:
                            logger.info(f"✓ [train() after super] 已设置静态图模式（路径: {path}）")
                        return True
                    except Exception:
                        pass
                
                attrs_to_check = ['module', 'base_model', 'model', '_orig_mod', 'wrapped_model']
                for attr_name in attrs_to_check:
                    if hasattr(model_obj, attr_name):
                        attr_obj = getattr(model_obj, attr_name)
                        if find_and_set_ddp_static_graph_after(attr_obj, depth + 1, f"{path}.{attr_name}", visited):
                            return True
                
                if hasattr(model_obj, '__dict__'):
                    for attr_name, attr_obj in model_obj.__dict__.items():
                        if isinstance(attr_obj, (torch.nn.Module, torch.nn.parallel.DistributedDataParallel)):
                            if find_and_set_ddp_static_graph_after(attr_obj, depth + 1, f"{path}.{attr_name}", visited):
                                return True
                
                return False
            
            if find_and_set_ddp_static_graph_after(self.model, path="self.model"):
                self._static_graph_set = True
        
        return result
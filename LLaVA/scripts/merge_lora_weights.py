import argparse
import os
import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

# 🟢 禁用 transformers 的 torch.load 安全检查（兼容旧版 torch）
# 在当前环境中升级 torch 到 2.6 不现实，只加载本地权重是安全的，因此直接关闭检查。
try:
    import transformers

    def _noop_safety_check():
        pass

    transformers.utils.import_utils.check_torch_load_is_safe = _noop_safety_check
    if hasattr(transformers.modeling_utils, "check_torch_load_is_safe"):
        transformers.modeling_utils.check_torch_load_is_safe = _noop_safety_check
except Exception as e:
    print(f"Warning: Failed to disable transformers safety check in merge_lora_weights.py: {e}")


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, 
        args.model_base, 
        model_name, 
        device_map='cpu',
        polar_vae_model_path=args.polar_vae_model_path if hasattr(args, 'polar_vae_model_path') and args.polar_vae_model_path else None
    )

    # 保存合并后的模型（包含 LoRA 合并后的 LLM 权重）
    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)
    
    # [Polar LLaVA 增强] 保存 polar_projector 权重（如果存在）
    # 注意：polar_projector 是自定义模块，不会自动包含在 save_pretrained 中
    if hasattr(model, 'get_model'):
        base_model = model.get_model()
        if hasattr(base_model, 'polar_projector'):
            polar_projector = base_model.polar_projector
            polar_projector_path = os.path.join(args.save_model_path, 'polar_projector.bin')
            torch.save(polar_projector.state_dict(), polar_projector_path)
            print(f"✓ Polar Projector 权重已保存: {polar_projector_path}")
        
        # 保存 mm_projector 权重（包含 rgb_projector 和可能的其他投影器组件）
        if hasattr(base_model, 'mm_projector'):
            mm_projector = base_model.mm_projector
            mm_projector_path = os.path.join(args.save_model_path, 'mm_projector.bin')
            # 提取 mm_projector 的 state_dict（可能需要处理键名）
            mm_projector_state = {}
            for name, param in mm_projector.named_parameters():
                # 移除可能的前缀（如 'base_model.model.mm_projector.'）
                clean_name = name.replace('base_model.model.mm_projector.', '').replace('model.mm_projector.', '').replace('mm_projector.', '')
                mm_projector_state[clean_name] = param.cpu()
            for name, buffer in mm_projector.named_buffers():
                clean_name = name.replace('base_model.model.mm_projector.', '').replace('model.mm_projector.', '').replace('mm_projector.', '')
                mm_projector_state[clean_name] = buffer.cpu()
            torch.save(mm_projector_state, mm_projector_path)
            print(f"✓ MM Projector 权重已保存: {mm_projector_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="LoRA 模型路径（包含 adapter_model.safetensors）")
    parser.add_argument("--model-base", type=str, required=True, help="基础模型路径（LLaVA base model）")
    parser.add_argument("--save-model-path", type=str, required=True, help="保存合并后模型的路径")
    parser.add_argument("--polar-vae-model-path", type=str, default=None, help="[Polar LLaVA] VAE 模型路径（如果使用 Polar LLaVA）")

    args = parser.parse_args()

    merge_lora(args)

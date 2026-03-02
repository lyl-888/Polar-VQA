try:
    from .model import LlavaLlamaForCausalLM
    from .model.language_model.llava_llama import PolarLlavaLlamaForCausalLM, PolarLlavaConfig
except ImportError:
    # 如果导入失败，可能是某些依赖未安装
    pass

# Polar LLaVA 安装指南

## 📋 前置要求

- Python 3.10（推荐）
- CUDA（用于 GPU 训练）
- 已激活的虚拟环境（如 `/openbayes/home/env/polarvqa`）

## 🔧 安装步骤

### 1. 激活虚拟环境

```bash
source /openbayes/home/env/polarvqa/bin/activate
# 或
conda activate polarvqa
```

### 2. 进入 LLaVA 目录

```bash
cd /openbayes/input/input0/train/LLaVA
# 或根据你的实际路径
cd /path/to/train/LLaVA
```

### 3. 升级 pip（启用 PEP 660 支持）

```bash
pip install --upgrade pip
```

### 4. 安装 LLaVA 包（可编辑模式）

```bash
pip install -e .
```

这将安装基础依赖，包括：
- torch==2.1.2, torchvision==0.16.2
- transformers==4.37.2
- accelerate==0.21.0
- peft, bitsandbytes
- 其他基础依赖

### 5. 安装训练相关依赖

```bash
pip install -e ".[train]"
```

这将安装：
- deepspeed==0.12.6
- ninja
- wandb

### 6. 安装 Flash Attention（可选，但推荐）

Flash Attention 可以加速训练并减少显存占用：

```bash
pip install flash-attn --no-build-isolation
```

**注意**：如果安装失败，可以尝试：
```bash
pip install flash-attn --no-build-isolation --no-cache-dir
```

### 7. 安装 Polar 相关依赖

由于我们使用了 VAE 编码器，需要确保安装了 `diffusers`：

```bash
pip install diffusers
```

### 8. 验证安装

```bash
python -c "import llava; print('LLaVA installed successfully')"
python -c "from llava.model.language_model.llava_llama import PolarLlavaLlamaForCausalLM; print('Polar LLaVA imported successfully')"
```

## 📦 完整依赖列表

根据 `pyproject.toml`，主要依赖包括：

### 基础依赖
- torch==2.1.2
- torchvision==0.16.2
- transformers==4.37.2
- tokenizers==0.15.1
- sentencepiece==0.1.99
- accelerate==0.21.0
- peft
- bitsandbytes
- pydantic
- numpy
- scikit-learn==1.2.2
- einops==0.6.1
- timm==0.6.13

### 训练依赖（通过 `[train]` 安装）
- deepspeed==0.12.6
- ninja
- wandb

### 额外依赖（Polar 项目需要）
- diffusers（VAE 编码器）
- Pillow（图像处理）
- tqdm（进度条）

## ⚠️ 常见问题

### 问题 1: `ModuleNotFoundError: No module named 'llava'`

**原因**：LLaVA 包未安装或未以可编辑模式安装

**解决方案**：
```bash
cd /path/to/train/LLaVA
pip install -e .
```

### 问题 2: Flash Attention 安装失败

**原因**：编译环境问题或 CUDA 版本不匹配

**解决方案**：
- 如果不需要 Flash Attention，可以跳过此步骤
- 训练时使用 `--attn_implementation sdpa` 或 `--attn_implementation eager` 替代

### 问题 3: `ImportError: cannot import name 'PolarLlavaLlamaForCausalLM'`

**原因**：代码修改后未重新安装包

**解决方案**：
```bash
cd /path/to/train/LLaVA
pip install -e . --force-reinstall --no-deps
```

### 问题 4: DeepSpeed 相关错误

**原因**：DeepSpeed 未正确安装

**解决方案**：
```bash
pip install deepspeed==0.12.6
```

## 🚀 快速安装命令（一键安装）

```bash
# 激活环境
source /openbayes/home/env/polarvqa/bin/activate

# 进入 LLaVA 目录
cd /openbayes/input/input0/train/LLaVA

# 升级 pip
pip install --upgrade pip

# 安装基础包
pip install -e .

# 安装训练依赖
pip install -e ".[train]"

# 安装额外依赖
pip install diffusers tqdm

# 尝试安装 Flash Attention（可选）
pip install flash-attn --no-build-isolation || echo "Flash Attention installation failed, continuing without it"
```

## ✅ 安装验证

运行以下命令验证安装：

```bash
python -c "
import torch
import transformers
from llava.model import LlavaLlamaForCausalLM
from llava.model.language_model.llava_llama import PolarLlavaLlamaForCausalLM
print('✓ All imports successful')
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ Transformers version: {transformers.__version__}')
"
```

如果所有导入成功，说明安装完成！

## 📝 训练前检查清单

- [ ] LLaVA 包已安装（`pip install -e .`）
- [ ] 训练依赖已安装（`pip install -e ".[train]"`）
- [ ] diffusers 已安装（用于 VAE）
- [ ] 验证导入成功（`import llava` 无错误）
- [ ] 验证 Polar 模型导入成功（`from llava.model.language_model.llava_llama import PolarLlavaLlamaForCausalLM`）

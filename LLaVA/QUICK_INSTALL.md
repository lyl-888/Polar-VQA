# Polar LLaVA 快速安装指南

## 🚀 一键安装（推荐）

### 方法 1: 使用安装脚本

```bash
cd /openbayes/home/train/LLaVA
bash setup_new_environment.sh
```

### 方法 2: 手动安装

```bash
# 1. 创建新环境
conda create -p /openbayes/home/env/polar_llava python=3.10 -y

# 2. 激活环境
source activate /openbayes/home/env/polar_llava
# 或
conda activate /openbayes/home/env/polar_llava

# 3. 升级 pip
pip install --upgrade pip

# 4. 进入 LLaVA 目录
cd /openbayes/home/train/LLaVA

# 5. 安装 LLaVA 基础包
pip install -e .

# 6. 安装训练依赖
pip install -e ".[train]"

# 7. 安装额外依赖
pip install diffusers tqdm

# 8. 安装 Flash Attention（可选）
pip install flash-attn --no-build-isolation || echo "Flash Attention 安装失败，可跳过"
```

## ✅ 验证安装

```bash
# 激活环境
source activate /openbayes/home/env/polar_llava

# 验证导入
python -c "
import torch
import transformers
from llava.model import LlavaLlamaForCausalLM
from llava.model.language_model.llava_llama import PolarLlavaLlamaForCausalLM
print('✓ PyTorch:', torch.__version__)
print('✓ Transformers:', transformers.__version__)
print('✓ LLaVA 导入成功')
print('✓ Polar LLaVA 导入成功')
"
```

## 📝 训练时使用新环境

在训练脚本开头添加环境激活：

```bash
#!/bin/bash
source activate /openbayes/home/env/polar_llava
cd /openbayes/home/train/LLaVA

python -m llava.train.train \
  --model_name_or_path /openbayes/input/input0/models/llava-v1.5-13b \
  ...
```

## 🔧 环境管理

### 查看环境列表
```bash
conda env list
```

### 删除环境（如果需要）
```bash
conda env remove -p /openbayes/home/env/polar_llava
```

### 重新安装（如果出现问题）
```bash
# 删除旧环境
conda env remove -p /openbayes/home/env/polar_llava

# 重新运行安装脚本
bash setup_new_environment.sh
```

# Stage 3 训练命令（OpenBayes 服务器）

## 服务器路径信息

- 训练脚本：`/openbayes/home/train/train.py`
- RGB图像：`/openbayes/input/input0/rgb`
- 偏振图像：`/openbayes/input/input0/polar`
- GT图像：`/openbayes/input/input0/GT`
- CLIP模型：`/openbayes/home/train/models/clip-vit-large-patch14`
- LLaMA模型：`/openbayes/input/input0/models/Meta-Llama-3-8B-Instruct`
- 输出目录：`/openbayes/home/train/checkpoints/polarvlm`（与train.py同目录）

```bash
python train.py \
    --train_json stage3_vqa_visual_direct_relpath.json \
    --rgb_root /openbayes/input/input0/rgb \
    --polar_root /openbayes/input/input0/polar \
    --output_dir ./checkpoints/polarvlm \
    --clip_model_name /openbayes/home/train/models/clip-vit-large-patch14 \
    --llm_model_name /openbayes/input/input0/models/Meta-Llama-3-8B-Instruct \
    --hf_token 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 \
    --polar_backbone resnet50 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 15 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.1 \
    --weight_decay 0.05 \
    --logging_steps 5 \
    --save_total_limit 1
```

## 参数说明

- `--train_json`: JSON文件路径（相对于train.py所在目录）
- `--rgb_root`: RGB图像根目录（绝对路径）
- `--polar_root`: 偏振图像根目录（绝对路径）
- `--output_dir`: 输出目录（相对路径`./checkpoints/polarvlm`，与train.py同目录）
- `--clip_model_name`: CLIP模型路径（绝对路径）
- `--llm_model_name`: LLaMA模型路径（绝对路径）
- `--hf_token`: Hugging Face token（用于下载其他模型）
- `--polar_backbone`: 偏振流backbone（resnet50）
- `--save_total_limit`: 只保留1个checkpoint（节省空间）

## 保存策略

- **训练途中**：在第 `total_steps // 2` 步左右保存一次
- **训练结束后**：自动保存最终模型
- **保存位置**：`/openbayes/home/train/checkpoints/polarvlm/`
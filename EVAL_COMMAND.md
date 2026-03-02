# PolarVLM Stage 2 评测命令

## 完整命令（红框模式 + 贪婪策略）

```bash
python eval_polar_stage2.py \
  --checkpoint-dir "/openbayes/input/input0/llava_train_stage2_final/checkpoint-100" \
  --llm-model-name "/openbayes/input/input0/models/llava-v1.5-13b" \
  --clip-model-name "/openbayes/input/input0/models/clip-vit-large-patch14-336" \
  --vae-model-path "/openbayes/input/input0/models/sd-vae-ft-mse" \
  --data-root "/openbayes/input/input0" \
  --polar-root "/openbayes/input/input0/polar" \
  --question-file "/openbayes/home/train/数据json/test_stage2_qwen.json" \
  --answers-file "/openbayes/home/train/数据json/polar_stage2_results_redbox_greedy.jsonl" \
  --use-red-box \
  --use-multi-gpu \
  --max-new-tokens 256 \
  --temperature 0.0 \
  --max-tokens 1024
```

## 参数说明

- `--use-red-box`: **启用红框模式**，在RGB图像上绘制红色边界框，并将问题中的坐标替换为"红色边界框"描述（与RGB baseline保持一致）
- `--temperature 0.0`: **贪婪策略**（默认值，可不写）
- **不设置 `--do-sample`**: 使用贪婪搜索（确定性，可复现）
- `--use-multi-gpu`: 使用多GPU（如果有多张卡）
- `--max-new-tokens 256`: 最大生成token数
- `--max-tokens 1024`: GPT-4o-mini评分时的最大token数

## 使用 nohup 后台运行（推荐）

```bash
nohup python -u eval_polar_stage2.py \
  --checkpoint-dir "/openbayes/input/input0/llava_train_stage2_final/checkpoint-100" \
  --llm-model-name "/openbayes/input/input0/models/llava-v1.5-13b" \
  --clip-model-name "/openbayes/input/input0/models/clip-vit-large-patch14-336" \
  --vae-model-path "/openbayes/input/input0/models/sd-vae-ft-mse" \
  --data-root "/openbayes/input/input0" \
  --polar-root "/openbayes/input/input0/polar" \
  --question-file "/openbayes/home/train/数据json/test_stage2_qwen.json" \
  --answers-file "/openbayes/home/train/数据json/polar_stage2_results_redbox_greedy.jsonl" \
  --use-red-box \
  --use-multi-gpu \
  --max-new-tokens 256 \
  --temperature 0.0 \
  --max-tokens 1024 \
  > eval_polar_redbox_greedy.log 2>&1 &
```

## 输出文件

- **JSONL结果文件**: `polar_stage2_results_redbox_greedy.jsonl`
  - 每行一个JSON对象，包含 `question_id`, `prompt`, `text` (模型回答), `gt_answer`, `score`, `review` 等
  
- **统计文件**: `polar_stage2_results_redbox_greedy_stats.json`
  - 包含平均分、分数分布、所有样本的详细结果

- **日志文件**: `eval_polar_redbox_greedy.log` (如果使用nohup)

## 注意事项

1. **路径确认**: 请根据实际情况修改以下路径：
   - `--question-file`: 测试集JSON文件路径
   - `--answers-file`: 输出文件路径
   - `--checkpoint-dir`: Stage 2 checkpoint路径

2. **GPU使用**: 
   - 如果只有单卡，移除 `--use-multi-gpu`
   - 如果显存不足，可以考虑在 `inference.py` 中设置 `load_4bit=True`

3. **实时输出**: 脚本会在终端实时输出每个样本的问题、模型回答、GT回答和评分，方便监控进度

4. **生成策略**: 
   - 贪婪策略（`temperature=0.0`, 不设置 `--do-sample`）是完全确定性的，每次运行结果一致
   - 适合正式评测和结果复现

"""
分析 Stage 3 训练数据中的答案长度分布
帮助确定评测时应该使用的 max-new-tokens 参数
"""

import json
from pathlib import Path
from collections import Counter
import re

def count_tokens_approx(text: str) -> int:
    """粗略估算 token 数（英文：~4字符/token，中文：~1.5字符/token）"""
    if not text:
        return 0
    # 简单估算：英文单词数 + 中文字符数
    english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    # 英文：平均每个单词约1.3个token，中文：每个字符约1.5个token
    return int(english_words * 1.3 + chinese_chars * 1.5)

def analyze_answer_lengths(json_path: str):
    """分析答案长度分布"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总样本数: {len(data)}")
    
    # 按 subtype 分类统计
    subtypes = ['positive', 'physics', 'negative']
    all_lengths = []
    by_subtype = {st: [] for st in subtypes}
    
    for item in data:
        subtype = item.get('subtype', '')
        conversations = item.get('conversations', [])
        
        if len(conversations) < 2:
            continue
        
        answer = conversations[1].get('value', '')
        if not answer:
            continue
        
        # 估算 token 数
        token_count = count_tokens_approx(answer)
        all_lengths.append(token_count)
        
        if subtype in by_subtype:
            by_subtype[subtype].append(token_count)
    
    # 统计信息
    def print_stats(name: str, lengths: list):
        if not lengths:
            print(f"\n{name}: 无数据")
            return
        
        lengths.sort()
        total = len(lengths)
        mean = sum(lengths) / total
        median = lengths[total // 2]
        p75 = lengths[int(total * 0.75)]
        p90 = lengths[int(total * 0.90)]
        p95 = lengths[int(total * 0.95)]
        p99 = lengths[int(total * 0.99)]
        max_len = max(lengths)
        
        # 统计超过不同阈值的比例
        over_64 = sum(1 for l in lengths if l > 64)
        over_128 = sum(1 for l in lengths if l > 128)
        over_256 = sum(1 for l in lengths if l > 256)
        
        print(f"\n{name} ({total} 个样本):")
        print(f"  平均长度: {mean:.1f} tokens")
        print(f"  中位数: {median} tokens")
        print(f"  P75: {p75} tokens")
        print(f"  P90: {p90} tokens")
        print(f"  P95: {p95} tokens")
        print(f"  P99: {p99} tokens")
        print(f"  最大长度: {max_len} tokens")
        print(f"\n  超过 64 tokens: {over_64} ({over_64/total*100:.1f}%)")
        print(f"  超过 128 tokens: {over_128} ({over_128/total*100:.1f}%)")
        print(f"  超过 256 tokens: {over_256} ({over_256/total*100:.1f}%)")
    
    print("=" * 80)
    print("答案长度分布分析")
    print("=" * 80)
    
    print_stats("总体", all_lengths)
    
    for subtype in subtypes:
        if by_subtype[subtype]:
            print_stats(f"Task {subtype.upper()}", by_subtype[subtype])
    
    # 建议
    print("\n" + "=" * 80)
    print("建议")
    print("=" * 80)
    
    if all_lengths:
        p95 = sorted(all_lengths)[int(len(all_lengths) * 0.95)]
        p99 = sorted(all_lengths)[int(len(all_lengths) * 0.99)]
        max_len = max(all_lengths)
        
        print(f"\n基于分析结果，建议的 max-new-tokens 设置：")
        print(f"  - 保守（覆盖95%的答案）: {p95} tokens")
        print(f"  - 安全（覆盖99%的答案）: {p99} tokens")
        print(f"  - 完整（覆盖所有答案）: {max_len} tokens")
        
        over_64 = sum(1 for l in all_lengths if l > 64)
        if over_64 > 0:
            pct = over_64 / len(all_lengths) * 100
            print(f"\n⚠️  警告: {over_64} 个答案（{pct:.1f}%）超过 64 tokens")
            print(f"   如果评测时使用 --max-new-tokens 64，这些答案会被截断！")
            print(f"   建议至少使用 --max-new-tokens {p95} 以避免截断问题")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python analyze_answer_length.py <训练数据JSON文件>")
        print("示例: python analyze_answer_length.py train_stage3_qwen_full.json")
        sys.exit(1)
    
    json_path = sys.argv[1]
    if not Path(json_path).exists():
        print(f"错误: 文件不存在: {json_path}")
        sys.exit(1)
    
    analyze_answer_lengths(json_path)

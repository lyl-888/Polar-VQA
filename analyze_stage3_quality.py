"""
分析 Stage 3 QA 数据集质量
检查格式、语法、一致性等问题
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict

def analyze_qa_quality(json_path: str):
    """分析 QA 数据集质量"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总样本数: {len(data)}")
    
    # 按 subtype 分类统计
    subtype_counts = Counter(item.get('subtype', 'unknown') for item in data)
    print(f"\n按 subtype 统计:")
    for subtype, count in subtype_counts.items():
        print(f"  {subtype}: {count}")
    
    # 问题分析
    issues = {
        'task_c_question_has_location': [],  # C 问题包含位置（不应该）
        'task_c_answer_missing_period': [],  # C 答案缺少句号
        'task_b_answer_format_issues': [],  # B 答案格式问题
        'task_a_answer_starts_with_there': [],  # A 答案以 "There is" 开头
        'grammar_issues': [],  # 语法错误
        'missing_physics_ending': [],  # 缺少物理解释结尾
    }
    
    location_words = {'top', 'bottom', 'left', 'right', 'center', 'corner', 'side', 'near', 'at', 'in', 'on'}
    
    for idx, item in enumerate(data):
        subtype = item.get('subtype', '')
        conversations = item.get('conversations', [])
        
        if len(conversations) < 2:
            continue
        
        question = conversations[0].get('value', '')
        answer = conversations[1].get('value', '')
        
        # Task C 问题检查：不应该包含位置信息
        if subtype == 'negative':
            q_lower = question.lower()
            has_location = any(word in q_lower for word in location_words)
            if has_location:
                issues['task_c_question_has_location'].append({
                    'idx': idx,
                    'question': question,
                    'answer': answer[:100] + '...' if len(answer) > 100 else answer
                })
        
        # Task C 答案检查：缺少句号
        if subtype == 'negative':
            if answer and not answer.rstrip().endswith(('.', '!', '?')):
                issues['task_c_answer_missing_period'].append({
                    'idx': idx,
                    'answer': answer
                })
        
        # Task A 答案检查：不应该以 "There is" 开头
        if subtype == 'positive':
            if answer.strip().lower().startswith(('there is', 'there are')):
                issues['task_a_answer_starts_with_there'].append({
                    'idx': idx,
                    'answer': answer
                })
        
        # Task B 答案检查：格式和物理解释
        if subtype == 'physics':
            if 'high polarization' not in answer.lower():
                issues['missing_physics_ending'].append({
                    'idx': idx,
                    'answer': answer
                })
            # 检查格式是否符合 "The reflection is mainly in the {loc}, appearing as {obj}..."
            if not re.search(r'reflection is (mainly )?in the', answer, re.IGNORECASE):
                issues['task_b_answer_format_issues'].append({
                    'idx': idx,
                    'answer': answer
                })
        
        # 语法检查：常见错误
        if 'a appearance' in answer.lower() or 'a appearance' in question.lower():
            issues['grammar_issues'].append({
                'idx': idx,
                'type': 'a/an error',
                'question': question,
                'answer': answer[:100] + '...' if len(answer) > 100 else answer
            })
    
    # 打印问题统计
    print(f"\n=== 质量问题统计 ===")
    for issue_type, items in issues.items():
        if items:
            print(f"\n{issue_type}: {len(items)} 个问题")
            # 显示前 5 个示例
            for item in items[:5]:
                if 'idx' in item:
                    print(f"  索引 {item['idx']}: {item.get('question', item.get('answer', ''))[:80]}")
    
    # 计算质量分数
    total_negative = subtype_counts.get('negative', 0)
    total_positive = subtype_counts.get('positive', 0)
    total_physics = subtype_counts.get('physics', 0)
    
    c_location_issue_rate = len(issues['task_c_question_has_location']) / total_negative * 100 if total_negative > 0 else 0
    c_period_issue_rate = len(issues['task_c_answer_missing_period']) / total_negative * 100 if total_negative > 0 else 0
    a_there_issue_rate = len(issues['task_a_answer_starts_with_there']) / total_positive * 100 if total_positive > 0 else 0
    b_physics_issue_rate = len(issues['missing_physics_ending']) / total_physics * 100 if total_physics > 0 else 0
    
    print(f"\n=== 质量分数 ===")
    print(f"Task C 问题包含位置: {c_location_issue_rate:.1f}% ({len(issues['task_c_question_has_location'])}/{total_negative})")
    print(f"Task C 答案缺少句号: {c_period_issue_rate:.1f}% ({len(issues['task_c_answer_missing_period'])}/{total_negative})")
    print(f"Task A 答案以 'There is' 开头: {a_there_issue_rate:.1f}% ({len(issues['task_a_answer_starts_with_there'])}/{total_positive})")
    print(f"Task B 答案缺少物理解释: {b_physics_issue_rate:.1f}% ({len(issues['missing_physics_ending'])}/{total_physics})")
    print(f"语法错误: {len(issues['grammar_issues'])} 个")
    
    # 整体质量评估
    total_issues = sum(len(v) for v in issues.values())
    total_samples = len(data)
    quality_score = (1 - total_issues / total_samples) * 100
    
    print(f"\n=== 整体质量评估 ===")
    print(f"总问题数: {total_issues}")
    print(f"总样本数: {total_samples}")
    print(f"质量分数: {quality_score:.1f}%")
    
    if quality_score >= 95:
        print("[OK] 质量优秀，可以直接用于训练")
    elif quality_score >= 90:
        print("[WARN] 质量良好，建议修复主要问题后再训练")
    else:
        print("[ERROR] 质量需要改进，建议修复问题后再训练")
    
    return issues

if __name__ == "__main__":
    json_path = Path("评测/stage3_all_scenes_full_new.json")
    if not json_path.exists():
        print(f"文件不存在: {json_path}")
        exit(1)
    
    issues = analyze_qa_quality(str(json_path))
    
    # 保存详细问题报告
    report_path = json_path.parent / "stage3_quality_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(issues, f, ensure_ascii=False, indent=2)
    print(f"\n详细报告已保存到: {report_path}")

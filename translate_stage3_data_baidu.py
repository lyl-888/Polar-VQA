#!/usr/bin/env python3
"""
Stage 3 VQA 数据中文翻译为英文脚本
使用百度翻译 API 将 conversations 中的中文文本翻译为英文

使用方法：
    python translate_stage3_data_baidu.py `
        --input stage3_vqa_visual_direct_merged.json `
        --output stage3_vqa_visual_direct_merged_en.json `
        --appid 20260110002536869 `
        --secret_key X_ZctW2qIGk4d2JY1I1g
"""

import argparse
import json
import hashlib
import random
import time
from pathlib import Path
from typing import List, Dict, Optional
import requests
from tqdm import tqdm

def translate_baidu(text: str, appid: str, secret_key: str, from_lang: str = "zh", to_lang: str = "en", max_retries: int = 3) -> str:
    """
    使用百度翻译 API 翻译文本（带重试机制）
    
    Args:
        text: 待翻译的文本
        appid: 百度翻译 API AppID
        secret_key: 百度翻译 API Secret Key
        from_lang: 源语言（默认：zh 中文）
        to_lang: 目标语言（默认：en 英文）
        max_retries: 最大重试次数
    
    Returns:
        翻译后的文本
    """
    if not text or not text.strip():
        return text
    
    # 检查是否包含中文（如果不包含，直接返回）
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
    if not has_chinese:
        return text
    
    # 检查文本长度（百度翻译 API 单次最多 6000 字节，约 2000 个汉字）
    # 如果超过限制，需要分段翻译
    if len(text.encode('utf-8')) > 6000:
        print(f"Warning: 文本过长（{len(text.encode('utf-8'))} 字节），将分段翻译...")
        # 简单分段：按句子分割（遇到句号、问号、感叹号分割）
        import re
        sentences = re.split(r'([。！？\n])', text)
        translated_parts = []
        current_chunk = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")
            if len((current_chunk + sentence).encode('utf-8')) > 6000:
                if current_chunk:
                    translated_chunk = translate_baidu(current_chunk, appid, secret_key, from_lang, to_lang, max_retries=1)
                    translated_parts.append(translated_chunk)
                    time.sleep(1.1)
                current_chunk = sentence
            else:
                current_chunk += sentence
        
        if current_chunk:
            translated_chunk = translate_baidu(current_chunk, appid, secret_key, from_lang, to_lang, max_retries=1)
            translated_parts.append(translated_chunk)
        
        return " ".join(translated_parts)
    
    # 百度翻译 API 端点
    url = "https://fanyi-api.baidu.com/api/trans/vip/translate"
    
    # 重试机制
    for attempt in range(max_retries):
        try:
            # 生成签名
            salt = str(random.randint(32768, 65536))
            sign_str = appid + text + salt + secret_key
            sign = hashlib.md5(sign_str.encode('utf-8')).hexdigest()
            
            # 构建请求参数
            params = {
                "q": text,
                "from": from_lang,
                "to": to_lang,
                "appid": appid,
                "salt": salt,
                "sign": sign
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            # 检查错误
            if "error_code" in result:
                error_code = result.get("error_code")
                error_msg = result.get("error_msg", "Unknown error")
                
                # 特殊处理 IP 白名单错误
                if error_code == "58000":
                    # 尝试多个服务获取当前公网 IP
                    current_ip = "无法获取"
                    ip_services = [
                        ("https://api.ipify.org?format=json", lambda r: r.json().get("ip")),
                        ("https://ipapi.co/json/", lambda r: r.json().get("ip")),
                        ("http://httpbin.org/ip", lambda r: r.json().get("origin", "").split(",")[0].strip()),
                        ("https://api.myip.com", lambda r: r.json().get("ip")),
                    ]
                    
                    for service_url, extractor in ip_services:
                        try:
                            ip_response = requests.get(service_url, timeout=5)
                            ip = extractor(ip_response)
                            if ip:
                                current_ip = ip
                                break
                        except:
                            continue
                    
                    error_detail = (
                        f"\n"
                        f"❌ 错误: 当前 IP 地址不在百度翻译 API 白名单中\n"
                        f"   错误代码: {error_code}\n"
                        f"   错误信息: {error_msg}\n"
                        f"   当前公网 IP: {current_ip}\n"
                        f"\n"
                        f"📝 解决方案（重要）：\n"
                        f"   1. 访问 https://fanyi-api.baidu.com/manage\n"
                        f"   2. 登录并找到你的应用（AppID: {appid}）\n"
                        f"   3. 点击进入应用详情页\n"
                        f"   4. 找到 '访问控制' 或 'IP白名单' 设置（不是'服务器地址'）\n"
                        f"   5. 如果找不到，可能需要在应用设置的高级选项中\n"
                        f"   6. 添加以下 IP 地址到白名单:\n"
                        f"      - {current_ip} (如果检测到)\n"
                        f"      - 59.64.129.17 (你已设置的本地IP)\n"
                        f"      - 或者添加 0.0.0.0/0 允许所有IP（临时测试用）\n"
                        f"   7. 保存后等待 3-5 分钟生效\n"
                        f"\n"
                        f"⚠️  注意事项：\n"
                        f"   - '服务器地址'字段（你已设置59.64.129.17）不是IP白名单\n"
                        f"   - 需要找到专门的'IP白名单'或'访问控制'设置\n"
                        f"   - 某些账户类型可能没有IP白名单功能，需要升级账户\n"
                        f"   - 如果仍然找不到，建议联系百度翻译API客服\n"
                    )
                    raise Exception(error_detail)
                
                # 某些错误可以重试（但排除 58000，因为它需要人工处理）
                if error_code in ["54000", "54001", "54003", "90107"] and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 指数退避：2秒, 4秒, 6秒
                    print(f"Warning: API 错误 {error_code}，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"百度翻译 API 错误: {error_code} - {error_msg}")
            
            # 提取翻译结果
            if "trans_result" in result and len(result["trans_result"]) > 0:
                translated_text = result["trans_result"][0].get("dst", text)
                return translated_text
            else:
                print(f"Warning: 翻译结果为空，返回原文: {text[:50]}...")
                return text
        
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"Warning: 网络错误，等待 {wait_time} 秒后重试: {e}")
                time.sleep(wait_time)
                continue
            else:
                print(f"Warning: 翻译失败（网络错误） '{text[:50]}...': {e}")
                return text
        
        except Exception as e:
            # 如果是 IP 白名单错误，直接抛出（不返回原文，让用户知道问题）
            if "58000" in str(e) or "IP" in str(e) or "白名单" in str(e):
                print(f"\n{e}")
                raise
            print(f"Warning: 翻译失败 '{text[:50]}...': {e}")
            return text
    
    # 所有重试都失败，返回原文
    print(f"Warning: 翻译失败，已重试 {max_retries} 次，返回原文: {text[:50]}...")
    return text


def translate_json_data(
    data: List[Dict],
    appid: str,
    secret_key: str,
    from_lang: str = "zh",
    to_lang: str = "en",
    cache_file: Optional[str] = None,
) -> List[Dict]:
    """
    翻译 JSON 数据中的中文文本（支持翻译缓存），用于 VQA conversations 格式。
    
    Args:
        data: JSON 数据列表（包含 conversations 字段）
        appid: 百度翻译 API AppID
        secret_key: 百度翻译 API Secret Key
        from_lang: 源语言（默认：zh）
        to_lang: 目标语言（默认：en）
        cache_file: 翻译缓存文件路径（可选，用于保存/加载已翻译的内容）
    
    Returns:
        翻译后的数据列表
    """
    # 加载翻译缓存（如果存在）
    translation_cache = {}
    if cache_file and Path(cache_file).exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                translation_cache = json.load(f)
            print(f"   ✓ 加载翻译缓存: {len(translation_cache)} 条记录")
        except Exception as e:
            print(f"   ⚠ 无法加载翻译缓存: {e}")
    
    translated_data = []
    cache_updated = False
    
    print(f"📝 开始翻译 {len(data)} 条样本...")
    
    for idx, item in enumerate(tqdm(data, desc="处理样本")):
        new_item = item.copy()
        
        # 翻译 conversations 中的文本
        conversations = item.get("conversations", [])
        if conversations:
            new_conversations = []
            for conv in conversations:
                new_conv = conv.copy()
                value = conv.get("value", "")
                
                # 当 from_lang 为中文时，只在包含中文时翻译；其他语言时，只要非空就翻译
                has_chinese = any('\u4e00' <= char <= '\u9fff' for char in value)
                need_translate = False
                if from_lang == "zh":
                    need_translate = has_chinese and bool(value.strip())
                else:
                    need_translate = bool(value.strip())
                
                if need_translate:
                    # 检查缓存
                    if value in translation_cache:
                        translated_value = translation_cache[value]
                    else:
                        # 翻译文本
                        translated_value = translate_baidu(
                            value, appid, secret_key, from_lang=from_lang, to_lang=to_lang
                        )
                        # 保存到缓存
                        translation_cache[value] = translated_value
                        cache_updated = True
                        # 添加小延迟避免 API 频率限制（百度免费版：1次/秒）
                        time.sleep(1.1)  # 1.1秒延迟，确保不超过频率限制
                    
                    new_conv["value"] = translated_value
                else:
                    # 已经是英文或其他语言，不翻译
                    new_conv["value"] = value
                
                new_conversations.append(new_conv)
            
            new_item["conversations"] = new_conversations
        
        translated_data.append(new_item)
        
        # 每100条样本保存一次缓存（防止中途中断丢失进度）
        if cache_file and cache_updated and (idx + 1) % 100 == 0:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(translation_cache, f, ensure_ascii=False, indent=2)
                cache_updated = False
            except Exception as e:
                print(f"   ⚠ 保存缓存失败: {e}")
    
    # 最终保存缓存
    if cache_file and cache_updated:
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(translation_cache, f, ensure_ascii=False, indent=2)
            print(f"   ✓ 翻译缓存已保存: {cache_file}")
        except Exception as e:
            print(f"   ⚠ 保存缓存失败: {e}")
    
    return translated_data


def translate_results_data(
    data: List[Dict],
    appid: str,
    secret_key: str,
    from_lang: str = "en",
    to_lang: str = "zh",
    cache_file: Optional[str] = None,
) -> List[Dict]:
    """
    翻译 Stage 3 结果文件（如 merged_stage3_qwen.results.json）中的 question / answer 字段。

    典型数据格式：
    {
        "scene_id": "30",
        "base_name": "0010",
        "qa_type": "content",
        "question": "...",
        "answer": "...",
        "bbox_norm": [...],
        "rgb_path": "..."
    }
    """
    translation_cache = {}
    if cache_file and Path(cache_file).exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                translation_cache = json.load(f)
            print(f"   ✓ 加载翻译缓存: {len(translation_cache)} 条记录")
        except Exception as e:
            print(f"   ⚠ 无法加载翻译缓存: {e}")

    translated_data: List[Dict] = []
    cache_updated = False

    print(f"📝 开始翻译结果文件中的 {len(data)} 条样本（question/answer）...")

    fields_to_translate = ["question", "answer"]

    for idx, item in enumerate(tqdm(data, desc="处理结果样本")):
        new_item = item.copy()

        for field in fields_to_translate:
            text = str(item.get(field, ""))
            if not text.strip():
                continue

            # from_lang 为中文时，只翻译包含中文的；否则只要非空就翻译
            has_chinese = any("\u4e00" <= ch <= "\u9fff" for ch in text)
            if from_lang == "zh":
                need_translate = has_chinese
            else:
                need_translate = True

            if not need_translate:
                continue

            if text in translation_cache:
                translated = translation_cache[text]
            else:
                translated = translate_baidu(
                    text, appid, secret_key, from_lang=from_lang, to_lang=to_lang
                )
                translation_cache[text] = translated
                cache_updated = True
                time.sleep(1.1)

            new_item[field] = translated

        translated_data.append(new_item)

        if cache_file and cache_updated and (idx + 1) % 100 == 0:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(translation_cache, f, ensure_ascii=False, indent=2)
                cache_updated = False
            except Exception as e:
                print(f"   ⚠ 保存缓存失败: {e}")

    if cache_file and cache_updated:
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(translation_cache, f, ensure_ascii=False, indent=2)
            print(f"   ✓ 翻译缓存已保存: {cache_file}")
        except Exception as e:
            print(f"   ⚠ 保存缓存失败: {e}")

    return translated_data


def main():
    parser = argparse.ArgumentParser(description="使用百度翻译 API 翻译 Stage 3 数据（支持 VQA 对话和结果文件）")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入的 JSON 文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出的 JSON 文件路径（默认：在输入文件名后加 _en 或 _zh）"
    )
    parser.add_argument(
        "--appid",
        type=str,
        required=True,
        help="百度翻译 API AppID"
    )
    parser.add_argument(
        "--secret_key",
        type=str,
        required=True,
        help="百度翻译 API Secret Key"
    )
    parser.add_argument(
        "--from_lang",
        type=str,
        default="zh",
        help="源语言（默认：zh 中文，可改为 en 等）"
    )
    parser.add_argument(
        "--to_lang",
        type=str,
        default="en",
        help="目标语言（默认：en 英文，可改为 zh 等）"
    )
    parser.add_argument(
        "--cache_file",
        type=str,
        default=None,
        help="翻译缓存文件路径（可选，用于保存/加载已翻译的内容，避免重复翻译）"
    )
    
    # 解析参数
    try:
        args = parser.parse_args()
    except SystemExit:
        # 如果是 Windows PowerShell，提供更友好的错误提示
        print("\n⚠️  Windows PowerShell 使用提示：")
        print("   在 Windows 中，请使用以下格式（不要使用反斜杠换行）：")
        print("\n   python translate_stage3_data_baidu.py --input stage3_vqa_visual_direct_merged.json --output stage3_vqa_visual_direct_merged_en.json --appid YOUR_APPID --secret_key YOUR_SECRET_KEY")
        print("\n   或者使用多行命令（使用反引号 ` 作为行继续符）：")
        print("   python translate_stage3_data_baidu.py `")
        print("       --input stage3_vqa_visual_direct_merged.json `")
        print("       --output stage3_vqa_visual_direct_merged_en.json `")
        print("       --appid YOUR_APPID `")
        print("       --secret_key YOUR_SECRET_KEY")
        raise
    
    # 确定输出文件路径
    if args.output is None:
        input_path = Path(args.input)
        output_path = input_path.parent / f"{input_path.stem}_en{input_path.suffix}"
    else:
        output_path = Path(args.output)
    
    # 读取输入 JSON
    print(f"📖 读取输入文件: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"   ✓ 读取 {len(data)} 条样本")
    
    # 确定缓存文件路径
    if args.cache_file is None:
        cache_file = str(Path(args.input).parent / "translation_cache.json")
    else:
        cache_file = args.cache_file
    
    # 翻译数据
    print(f"\n🌐 开始翻译（从 {args.from_lang} 到 {args.to_lang}）...")
    if args.cache_file:
        print(f"   缓存文件: {cache_file}")

    # 根据数据结构自动选择翻译模式
    first_item = data[0] if isinstance(data, list) and data else {}
    has_conversations = isinstance(first_item, dict) and "conversations" in first_item
    has_qa_fields = isinstance(first_item, dict) and "question" in first_item and "answer" in first_item

    if has_conversations:
        print("检测到 conversations 字段，按 VQA 对话数据格式翻译...")
        translated_data = translate_json_data(
            data,
            args.appid,
            args.secret_key,
            from_lang=args.from_lang,
            to_lang=args.to_lang,
            cache_file=cache_file,
        )
    elif has_qa_fields:
        print("检测到 question/answer 字段，按结果文件格式翻译（如 merged_stage3_qwen.results.json）...")
        translated_data = translate_results_data(
            data,
            args.appid,
            args.secret_key,
            from_lang=args.from_lang,
            to_lang=args.to_lang,
            cache_file=cache_file,
        )
    else:
        raise ValueError("无法识别输入 JSON 结构：既没有 conversations 字段，也没有 question/answer 字段。")
    
    # 保存结果
    print(f"\n💾 保存翻译后的数据到: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)
    print(f"   ✓ 保存成功")
    
    # 统计信息
    print(f"\n📊 统计信息:")
    print(f"   - 原始样本数: {len(data)}")
    print(f"   - 翻译后样本数: {len(translated_data)}")
    
    # 统计翻译的对话数量
    translated_count = 0
    for item in translated_data:
        conversations = item.get("conversations", [])
        for conv in conversations:
            value = conv.get("value", "")
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in value)
            if not has_chinese and value.strip():
                translated_count += 1
    
    print(f"   - 翻译的对话数: {translated_count}")

if __name__ == "__main__":
    main()


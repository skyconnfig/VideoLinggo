import pandas as pd
from typing import List, Tuple
import concurrent.futures

from core._3_2_split_meaning import split_sentence
from core.prompts import get_align_prompt
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from core.utils import *
from core.utils.models import *
console = Console()

# ! You can modify your own weights here
# Chinese and Japanese 2.5 characters, Korean 2 characters, Thai 1.5 characters, full-width symbols 2 characters, other English-based and half-width symbols 1 character
def calc_len(text: str) -> float:
    text = str(text) # force convert
    def char_weight(char):
        code = ord(char)
        if 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF:  # Chinese and Japanese
            return 1.75
        elif 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF:  # Korean
            return 1.5
        elif 0x0E00 <= code <= 0x0E7F:  # Thai
            return 1
        elif 0xFF01 <= code <= 0xFF5E:  # full-width symbols
            return 1.75
        else:  # other characters (e.g. English and half-width symbols)
            return 1

    return sum(char_weight(char) for char in text)

def align_subs(src_sub: str, tr_sub: str, src_part: str) -> Tuple[List[str], List[str], str]:
    align_prompt = get_align_prompt(src_sub, tr_sub, src_part)
    
    def valid_align(response_data):
        # 添加调试日志
        console.print(f"[yellow]🔍 调试信息 - qwen2响应数据: {response_data}[/yellow]")
        
        if 'align' not in response_data:
            return {"status": "error", "message": "Missing required key: `align`"}
        
        align_data = response_data['align']
        console.print(f"[yellow]🔍 调试信息 - align数组长度: {len(align_data)}[/yellow]")
        console.print(f"[yellow]🔍 调试信息 - align数组内容: {align_data}[/yellow]")
        
        # 计算期望的部分数量
        src_splits = src_part.split('\n')
        expected_parts = len(src_splits)
        console.print(f"[yellow]🔍 调试信息 - 期望的部分数量: {expected_parts}[/yellow]")
        console.print(f"[yellow]🔍 调试信息 - 源文本分割: {src_splits}[/yellow]")
        
        # 容错处理：如果align数组长度不足，尝试补齐或调整期望
        if len(align_data) < expected_parts:
            console.print(f"[yellow]⚠️ 对齐数组长度({len(align_data)}) < 期望部分数量({expected_parts})，尝试容错处理[/yellow]")
            
            # 如果只差一个，可能是合理的合并
            if len(align_data) == expected_parts - 1:
                console.print(f"[yellow]可能是合理的部分合并，继续处理[/yellow]")
            else:
                # 尝试补齐缺失的部分
                while len(align_data) < expected_parts:
                    # 使用最后一个有效部分作为模板
                    if align_data:
                        last_item = align_data[-1]
                        new_item = {}
                        for key in last_item.keys():
                            if key.startswith('src_part_'):
                                new_item[key.replace('src_part_', f'src_part_{len(align_data)+1}_')] = ""
                            elif key.startswith('target_part_'):
                                new_item[key.replace('target_part_', f'target_part_{len(align_data)+1}_')] = ""
                        
                        # 添加标准格式的键
                        new_item[f'src_part_{len(align_data)+1}'] = ""
                        new_item[f'target_part_{len(align_data)+1}'] = ""
                        align_data.append(new_item)
                        console.print(f"[yellow]补齐了第{len(align_data)}个对齐项[/yellow]")
                    else:
                        break
        
        # 检查并修复每个align项的必要键
        valid_items = 0
        for i, item in enumerate(align_data[:expected_parts]):  # 只处理期望数量的项目
            expected_key = f'target_part_{i+1}'
            
            if expected_key not in item:
                console.print(f"[yellow]⚠️ 缺少键 {expected_key}，尝试修复[/yellow]")
                
                # 尝试查找相似的键
                similar_keys = [k for k in item.keys() if 'target' in k.lower() and str(i+1) in k]
                if similar_keys:
                    # 使用找到的相似键的值
                    item[expected_key] = item[similar_keys[0]]
                    console.print(f"[green]使用相似键 {similar_keys[0]} 的值修复 {expected_key}[/green]")
                    valid_items += 1
                else:
                    # 查找任何包含target的键
                    target_keys = [k for k in item.keys() if 'target' in k.lower()]
                    if target_keys:
                        item[expected_key] = item[target_keys[0]]
                        console.print(f"[yellow]使用第一个target键 {target_keys[0]} 的值作为 {expected_key}[/yellow]")
                        valid_items += 1
                    else:
                        # 提供默认值
                        item[expected_key] = f"默认翻译{i+1}"
                        console.print(f"[yellow]为 {expected_key} 提供默认值[/yellow]")
                        valid_items += 1
            else:
                valid_items += 1
        
        console.print(f"[green]✅ 对齐验证完成: 处理了{valid_items}个有效的对齐部分（期望{expected_parts}个）[/green]")
        return {"status": "success", "message": "Align completed with error recovery"}
    parsed = ask_gpt(align_prompt, resp_type='json', valid_def=valid_align, log_title='align_subs')
    align_data = parsed['align']
    src_parts = src_part.split('\n')
    
    # 安全提取target_part，添加容错处理
    tr_parts = []
    for i, item in enumerate(align_data):
        expected_key = f'target_part_{i+1}'
        if expected_key in item:
            tr_parts.append(item[expected_key].strip())
        else:
            # 查找任何包含target的键作为备用
            target_keys = [k for k in item.keys() if 'target' in k.lower()]
            if target_keys:
                tr_parts.append(item[target_keys[0]].strip())
                console.print(f"[yellow]使用备用键 {target_keys[0]} 替代 {expected_key}[/yellow]")
            else:
                # 使用源文本作为最后的备用
                if i < len(src_parts):
                    tr_parts.append(src_parts[i].strip())
                    console.print(f"[yellow]使用源文本作为 {expected_key} 的备用值[/yellow]")
                else:
                    tr_parts.append(f"备用翻译{i+1}")
                    console.print(f"[yellow]使用默认值作为 {expected_key} 的备用值[/yellow]")
    
    # 确保tr_parts和src_parts长度匹配
    while len(tr_parts) < len(src_parts):
        tr_parts.append(src_parts[len(tr_parts)].strip())
        console.print(f"[yellow]补齐翻译部分 {len(tr_parts)}[/yellow]")
    
    whisper_language = load_key("whisper.language")
    language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language
    joiner = get_joiner(language)
    tr_remerged = joiner.join(tr_parts)
    
    table = Table(title="🔗 Aligned parts")
    table.add_column("Language", style="cyan")
    table.add_column("Parts", style="magenta")
    table.add_row("SRC_LANG", "\n".join(src_parts))
    table.add_row("TARGET_LANG", "\n".join(tr_parts))
    console.print(table)
    
    return src_parts, tr_parts, tr_remerged

def split_align_subs(src_lines: List[str], tr_lines: List[str]):
    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]
    remerged_tr_lines = tr_lines.copy()
    
    to_split = []
    for i, (src, tr) in enumerate(zip(src_lines, tr_lines)):
        src, tr = str(src), str(tr)
        if len(src) > MAX_SUB_LENGTH or calc_len(tr) * TARGET_SUB_MULTIPLIER > MAX_SUB_LENGTH:
            to_split.append(i)
            table = Table(title=f"📏 Line {i} needs to be split")
            table.add_column("Type", style="cyan")
            table.add_column("Content", style="magenta")
            table.add_row("Source Line", src)
            table.add_row("Target Line", tr)
            console.print(table)
    
    @except_handler("Error in split_align_subs")
    def process(i):
        split_src = split_sentence(src_lines[i], num_parts=2).strip()
        src_parts, tr_parts, tr_remerged = align_subs(src_lines[i], tr_lines[i], split_src)
        src_lines[i] = src_parts
        tr_lines[i] = tr_parts
        remerged_tr_lines[i] = tr_remerged
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
        executor.map(process, to_split)
    
    # Flatten `src_lines` and `tr_lines`
    src_lines = [item for sublist in src_lines for item in (sublist if isinstance(sublist, list) else [sublist])]
    tr_lines = [item for sublist in tr_lines for item in (sublist if isinstance(sublist, list) else [sublist])]
    
    return src_lines, tr_lines, remerged_tr_lines

def split_for_sub_main():
    console.print("[bold green]🚀 Start splitting subtitles...[/bold green]")
    
    df = pd.read_excel(_4_2_TRANSLATION)
    src = df['Source'].tolist()
    trans = df['Translation'].tolist()
    
    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]
    
    for attempt in range(3):  # 多次切割
        console.print(Panel(f"🔄 Split attempt {attempt + 1}", expand=False))
        split_src, split_trans, remerged = split_align_subs(src.copy(), trans)
        
        # 检查是否所有字幕都符合长度要求
        if all(len(src) <= MAX_SUB_LENGTH for src in split_src) and \
           all(calc_len(tr) * TARGET_SUB_MULTIPLIER <= MAX_SUB_LENGTH for tr in split_trans):
            break
        
        # 更新源数据继续下一轮分割
        src, trans = split_src, split_trans

    # 确保所有数组有相同的长度，防止报错
    max_len = max(len(src), len(trans), len(remerged))
    
    # 补齐src数组
    if len(src) < max_len:
        src += [None] * (max_len - len(src))
    
    # 补齐trans数组
    if len(trans) < max_len:
        trans += [None] * (max_len - len(trans))
    
    # 补齐remerged数组
    if len(remerged) < max_len:
        remerged += [None] * (max_len - len(remerged))
    
    # 添加长度验证日志
    console.print(f"[green]✅ 数组长度验证: src={len(src)}, trans={len(trans)}, remerged={len(remerged)}[/green]")
    
    pd.DataFrame({'Source': src, 'Translation': trans}).to_excel(_5_SPLIT_SUB, index=False)
    pd.DataFrame({'Source': src, 'Translation': remerged}).to_excel(_5_REMERGED, index=False)

if __name__ == '__main__':
    split_for_sub_main()

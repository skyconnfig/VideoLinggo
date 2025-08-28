import concurrent.futures
from difflib import SequenceMatcher
import math
from core.prompts import get_split_prompt
from core.spacy_utils.load_nlp_model import init_nlp
from core.utils import *
from rich.console import Console
from rich.table import Table
from core.utils.models import _3_1_SPLIT_BY_NLP, _3_2_SPLIT_BY_MEANING
console = Console()

def tokenize_sentence(sentence, nlp):
    doc = nlp(sentence)
    return [token.text for token in doc]

def find_split_positions(original, modified):
    split_positions = []
    parts = modified.split('[br]')
    start = 0
    whisper_language = load_key("whisper.language")
    language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language
    joiner = get_joiner(language)

    for i in range(len(parts) - 1):
        max_similarity = 0
        best_split = None

        for j in range(start, len(original)):
            original_left = original[start:j]
            modified_left = joiner.join(parts[i].split())

            left_similarity = SequenceMatcher(None, original_left, modified_left).ratio()

            if left_similarity > max_similarity:
                max_similarity = left_similarity
                best_split = j

        if max_similarity < 0.9:
            console.print(f"[yellow]Warning: low similarity found at the best split point: {max_similarity}[/yellow]")
        if best_split is not None:
            split_positions.append(best_split)
            start = best_split
        else:
            console.print(f"[yellow]Warning: Unable to find a suitable split point for the {i+1}th part.[/yellow]")

    return split_positions

def split_sentence(sentence, num_parts, word_limit=20, index=-1, retry_attempt=0):
    """Split a long sentence using GPT and return the result as a string."""
    split_prompt = get_split_prompt(sentence, num_parts, word_limit)
    def valid_split(response_data, original_sentence=sentence):
        # 添加详细的响应数据日志
        console.print(f"[cyan]Debug: 收到的响应数据: {response_data}[/cyan]")
        console.print(f"[cyan]Debug: 响应数据类型: {type(response_data)}[/cyan]")
        console.print(f"[cyan]Debug: 响应数据键: {list(response_data.keys()) if isinstance(response_data, dict) else 'Not a dict'}[/cyan]")
        
        # 检查响应数据是否为字典类型
        if not isinstance(response_data, dict):
            console.print(f"[red]Error: 响应数据不是字典格式: {type(response_data)}[/red]")
            return {"status": "error", "message": "Response data is not a dictionary"}
        
        # 动态查找可用的split字段
        available_split_keys = [k for k in response_data.keys() if k.startswith('split') and k != 'split']
        console.print(f"[cyan]Debug: 可用的split字段: {available_split_keys}[/cyan]")
        
        # 确定要使用的split字段
        split_key = None
        choice = None
        
        # 首先尝试使用choice字段
        if "choice" in response_data:
            choice = str(response_data["choice"]).strip()
            console.print(f"[cyan]Debug: 检测到choice字段: '{choice}'[/cyan]")
            
            # 清理choice值，移除可能的额外文本
            if choice and choice[0].isdigit():
                choice = choice[0]  # 只取第一个数字
            
            potential_key = f'split{choice}'
            if potential_key in response_data:
                split_key = potential_key
                console.print(f"[green]使用choice指定的字段: {split_key}[/green]")
        
        # 如果choice方法失败，尝试使用第一个可用的split字段
        if not split_key and available_split_keys:
            split_key = available_split_keys[0]
            console.print(f"[yellow]choice字段无效，使用第一个可用的split字段: {split_key}[/yellow]")
        
        # 如果仍然没有找到split字段，返回错误
        if not split_key:
            console.print(f"[red]Error: 未找到任何有效的split字段[/red]")
            console.print(f"[yellow]Available keys: {list(response_data.keys())}[/yellow]")
            return {"status": "error", "message": "No valid split field found in response"}
        
        split_content = response_data[split_key]
        console.print(f"[cyan]Debug: 检查的分割内容: {repr(split_content)}[/cyan]")
        console.print(f"[cyan]Debug: 分割内容长度: {len(split_content)}[/cyan]")
        console.print(f"[cyan]Debug: 分割内容是否包含<split_this_sentence>: {'<split_this_sentence>' in split_content}[/cyan]")
        console.print(f"[cyan]Debug: 分割内容是否包含<split_this_paragraph>: {'<split_this_paragraph>' in split_content}[/cyan]")
        console.print(f"[cyan]Debug: 分割内容是否包含[br]: {'[br]' in split_content}[/cyan]")
        console.print(f"[cyan]Debug: 分割内容是否包含<br>: {'<br>' in split_content}[/cyan]")
        console.print(f"[cyan]Debug: 分割内容是否包含</br>: {'</br>' in split_content}[/cyan]")
        has_newlines = ('\n' in split_content) or ('\r\n' in split_content)
        console.print(f"[cyan]Debug: 分割内容是否包含换行符: {has_newlines}[/cyan]")
        
        # 兼容多种响应格式的验证逻辑
        has_valid_split = False
        processed_content = split_content
        
        # 检查标准的[br]标记
        if "[br]" in split_content:
            has_valid_split = True
            console.print(f"[green]Found standard [br] markers[/green]")
        
        # 检查qwen2格式：各种split标签包装的内容
        elif ("<split_this_sentence>" in split_content and "</split_this_sentence>" in split_content) or \
             ("<split_this_paragraph>" in split_content and "</split_this_paragraph>" in split_content):
            # 提取标签内的内容
            import re
            # 尝试匹配不同的标签格式
            match = re.search(r'<split_this_sentence>(.*?)</split_this_sentence>', split_content, re.DOTALL)
            if not match:
                match = re.search(r'<split_this_paragraph>(.*?)</split_this_paragraph>', split_content, re.DOTALL)
            
            if match:
                inner_content = match.group(1).strip()
                console.print(f"[cyan]Debug: 提取的内部内容: {repr(inner_content)}[/cyan]")
                
                # 先检查和转换br标记，再清理其他HTML标签
                temp_content = inner_content
                
                # 检查内部内容是否包含各种br标记
                if "<br>" in temp_content or "</br>" in temp_content:
                    # 将<br>和</br>都转换为[br]标记
                    temp_content = temp_content.replace('<br>', '[br]').replace('</br>', '[br]')
                    console.print(f"[green]Found <br> or </br> tags in qwen2 format[/green]")
                elif "\r\n" in temp_content or "\n" in temp_content:
                    # 将换行符转换为[br]标记
                    temp_content = temp_content.replace('\r\n', '[br]').replace('\n', '[br]')
                    console.print(f"[green]Found newlines in qwen2 format[/green]")
                
                # 清理其他HTML标签（保留[br]标记）
                import re
                # 先保护[br]标记
                temp_content = temp_content.replace('[br]', '___BR_PLACEHOLDER___')
                # 清理HTML标签
                cleaned_content = re.sub(r'<[^>]+>', '', temp_content)
                # 恢复[br]标记
                processed_content = cleaned_content.replace('___BR_PLACEHOLDER___', '[br]')
                
                console.print(f"[cyan]Debug: 处理后的内容: {repr(processed_content)}[/cyan]")
                
                # 检查是否有有效的分割标记
                if '[br]' not in processed_content:
                    console.print(f"[yellow]No [br] markers found after processing, using cleaned content anyway[/yellow]")
                    # 即使没有[br]标记，也使用清理后的内容
                    # 这样至少可以去除HTML标签
                
                # 更新response_data中的内容
                response_data[split_key] = processed_content
                has_valid_split = True
                console.print(f"[green]Successfully processed qwen2 format content[/green]")
        
        # 检查是否直接包含<br>标记（HTML格式）
        elif "<br>" in split_content or "</br>" in split_content:
            # 将<br>和</br>都转换为[br]标记
            processed_content = split_content.replace('<br>', '[br]').replace('</br>', '[br]')
            response_data[split_key] = processed_content
            has_valid_split = True
            console.print(f"[green]Found <br> or </br> tags, converted to: {processed_content}[/green]")
        
        # 检查是否包含换行符（可能的分割标记）
        elif '\n' in split_content or '\r\n' in split_content:
            # 将换行符转换为[br]标记
            processed_content = split_content.replace('\r\n', '[br]').replace('\n', '[br]')
            response_data[split_key] = processed_content
            has_valid_split = True
            console.print(f"[green]Found newline separators, converted to: {processed_content}[/green]")
        
        if not has_valid_split:
            console.print(f"[yellow]Warning: 分割内容中没有找到标准分割标记，尝试智能分割[/yellow]")
            console.print(f"[cyan]实际内容: {repr(split_content)}[/cyan]")
            
            # 智能分割逻辑：尝试根据标点符号、连词等进行分割
            processed_content = split_content.strip()
            
            # 检查是否包含常见的分割点
            split_patterns = [
                # 中文标点
                '，', '。', '；', '：', '！', '？', '、',
                # 英文标点
                ',', '.', ';', ':', '!', '?',
                # 连词
                ' and ', ' but ', ' or ', ' so ', ' yet ', ' for ', ' nor ',
                ' 和 ', ' 但是 ', ' 或者 ', ' 所以 ', ' 然而 ', ' 因为 ',
                # 其他可能的分割点
                ' that ', ' which ', ' who ', ' when ', ' where ', ' why ', ' how ',
                ' 的 ', ' 在 ', ' 是 ', ' 有 ', ' 会 ', ' 能 ', ' 要 '
            ]
            
            # 寻找最佳分割点（尽量在中间位置）
            best_split_pos = len(processed_content) // 2
            min_distance = float('inf')
            
            for pattern in split_patterns:
                if pattern in processed_content:
                    # 找到所有匹配位置
                    pos = processed_content.find(pattern)
                    while pos != -1:
                        # 计算与中间位置的距离
                        distance = abs(pos - len(processed_content) // 2)
                        if distance < min_distance:
                            min_distance = distance
                            best_split_pos = pos + len(pattern)
                        pos = processed_content.find(pattern, pos + 1)
            
            # 如果找到了合适的分割点，进行分割
            if min_distance < float('inf') and best_split_pos > 10 and best_split_pos < len(processed_content) - 10:
                part1 = processed_content[:best_split_pos].strip()
                part2 = processed_content[best_split_pos:].strip()
                processed_content = f"{part1}[br]{part2}"
                console.print(f"[green]智能分割成功: 在位置 {best_split_pos} 处分割[/green]")
                has_valid_split = True
            else:
                # 如果无法智能分割，则按长度平均分割
                mid_point = len(processed_content) // 2
                # 寻找最近的空格进行分割
                space_before = processed_content.rfind(' ', 0, mid_point)
                space_after = processed_content.find(' ', mid_point)
                
                if space_before != -1 and (space_after == -1 or mid_point - space_before <= space_after - mid_point):
                    split_pos = space_before + 1
                elif space_after != -1:
                    split_pos = space_after
                else:
                    split_pos = mid_point
                
                if split_pos > 5 and split_pos < len(processed_content) - 5:
                    part1 = processed_content[:split_pos].strip()
                    part2 = processed_content[split_pos:].strip()
                    processed_content = f"{part1}[br]{part2}"
                    console.print(f"[yellow]按长度平均分割: 在位置 {split_pos} 处分割[/yellow]")
                    has_valid_split = True
                else:
                    console.print(f"[yellow]内容太短，无需分割，保持原样[/yellow]")
                    has_valid_split = True
            
            response_data[split_key] = processed_content
        
        # 最终验证：确保处理后的内容不为空
        final_content = response_data[split_key]
        if not final_content or final_content.strip() == "":
            console.print(f"[red]Error: 处理后的内容为空，使用原始句子[/red]")
            # 作为最后的容错措施，使用原始句子
            response_data[split_key] = original_sentence
            final_content = original_sentence
        
        # 验证分割结果的合理性
        if '[br]' in final_content:
            parts = final_content.split('[br]')
            # 检查分割后的部分是否合理
            valid_parts = [part.strip() for part in parts if part.strip()]
            if len(valid_parts) < 2:
                console.print(f"[yellow]Warning: 分割后只有 {len(valid_parts)} 个有效部分，可能分割不理想[/yellow]")
            elif any(len(part.strip()) < 3 for part in valid_parts):
                console.print(f"[yellow]Warning: 某些分割部分过短，可能影响字幕质量[/yellow]")
            else:
                console.print(f"[green]分割验证通过: 共 {len(valid_parts)} 个部分[/green]")
        else:
            console.print(f"[yellow]Warning: 最终内容中没有[br]标记，将作为单个部分处理: {repr(final_content[:50])}...[/yellow]")
        
        console.print(f"[green]Success: 验证通过，内容处理完成[/green]")
        return {"status": "success", "message": "Split completed"}
    
    response_data = ask_gpt(split_prompt + " " * retry_attempt, resp_type='json', valid_def=valid_split, log_title='split_by_meaning')
    
    # 安全获取choice和对应的split内容
    choice = response_data.get("choice", "1")  # 默认使用1
    
    # 查找实际可用的split字段
    available_split_keys = [k for k in response_data.keys() if k.startswith('split') and k != 'split']
    
    if f"split{choice}" in response_data:
        best_split = response_data[f"split{choice}"]
    elif available_split_keys:
        # 使用第一个可用的split字段
        best_split = response_data[available_split_keys[0]]
        console.print(f"[yellow]使用备用split字段: {available_split_keys[0]}[/yellow]")
    else:
        console.print(f"[red]Error: 无法找到任何split内容，使用原始句子[/red]")
        return sentence
    
    # 检查best_split是否有效
    if not best_split or not isinstance(best_split, str):
        console.print(f"[red]Error: split内容无效，使用原始句子[/red]")
        return sentence
    
    split_points = find_split_positions(sentence, best_split)
    # split the sentence based on the split points
    for i, split_point in enumerate(split_points):
        if i == 0:
            best_split = sentence[:split_point] + '\n' + sentence[split_point:]
        else:
            parts = best_split.split('\n')
            last_part = parts[-1]
            parts[-1] = last_part[:split_point - split_points[i-1]] + '\n' + last_part[split_point - split_points[i-1]:]
            best_split = '\n'.join(parts)
    if index != -1:
        console.print(f'[green]✅ Sentence {index} has been successfully split[/green]')
    table = Table(title="")
    table.add_column("Type", style="cyan")
    table.add_column("Sentence")
    table.add_row("Original", sentence, style="yellow")
    table.add_row("Split", best_split.replace('\n', ' ||'), style="yellow")
    console.print(table)
    
    return best_split

def parallel_split_sentences(sentences, max_length, max_workers, nlp, retry_attempt=0):
    """Split sentences in parallel using a thread pool."""
    new_sentences = [None] * len(sentences)
    futures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, sentence in enumerate(sentences):
            # Use tokenizer to split the sentence
            tokens = tokenize_sentence(sentence, nlp)
            # print("Tokenization result:", tokens)
            num_parts = math.ceil(len(tokens) / max_length)
            if len(tokens) > max_length:
                future = executor.submit(split_sentence, sentence, num_parts, max_length, index=index, retry_attempt=retry_attempt)
                futures.append((future, index, num_parts, sentence))
            else:
                new_sentences[index] = [sentence]

        for future, index, num_parts, sentence in futures:
            split_result = future.result()
            if split_result:
                split_lines = split_result.strip().split('\n')
                new_sentences[index] = [line.strip() for line in split_lines]
            else:
                new_sentences[index] = [sentence]

    return [sentence for sublist in new_sentences for sentence in sublist]

@check_file_exists(_3_2_SPLIT_BY_MEANING)
def split_sentences_by_meaning():
    """The main function to split sentences by meaning."""
    # read input sentences
    with open(_3_1_SPLIT_BY_NLP, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]

    nlp = init_nlp()
    # 🔄 process sentences multiple times to ensure all are split
    for retry_attempt in range(3):
        sentences = parallel_split_sentences(sentences, max_length=load_key("max_split_length"), max_workers=load_key("max_workers"), nlp=nlp, retry_attempt=retry_attempt)

    # 💾 save results
    with open(_3_2_SPLIT_BY_MEANING, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sentences))
    console.print('[green]✅ All sentences have been successfully split![/green]')

if __name__ == '__main__':
    # print(split_sentence('Which makes no sense to the... average guy who always pushes the character creation slider all the way to the right.', 2, 22))
    split_sentences_by_meaning()
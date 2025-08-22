import pandas as pd
import os
import re
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
import autocorrect_py as autocorrect
from core.utils import *
from core.utils.models import *
console = Console()

SUBTITLE_OUTPUT_CONFIGS = [ 
    ('src.srt', ['Source']),
    ('trans.srt', ['Translation']),
    ('src_trans.srt', ['Source', 'Translation']),
    ('trans_src.srt', ['Translation', 'Source'])
]

AUDIO_SUBTITLE_OUTPUT_CONFIGS = [
    ('src_subs_for_audio.srt', ['Source']),
    ('trans_subs_for_audio.srt', ['Translation'])
]

def convert_to_srt_format(start_time, end_time):
    """Convert time (in seconds) to the format: hours:minutes:seconds,milliseconds"""
    def seconds_to_hmsm(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int(seconds * 1000) % 1000
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

    start_srt = seconds_to_hmsm(start_time)
    end_srt = seconds_to_hmsm(end_time)
    return f"{start_srt} --> {end_srt}"

def remove_punctuation(text):
    """Enhanced text cleaning function that handles HTML tags and punctuation"""
    # First, remove HTML tags (like <span>, <br/>, etc.)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove HTML entities
    text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation but keep alphanumeric characters and spaces
    text = re.sub(r'[^\w\s]', '', text)
    
    return text.strip()

def normalize_sentence_for_matching(text):
    """Advanced sentence normalization for better matching accuracy"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove HTML tags and entities
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)
    
    # Remove common subtitle artifacts
    text = re.sub(r'\[.*?\]', '', text)  # Remove [music], [applause], etc.
    text = re.sub(r'\(.*?\)', '', text)  # Remove (background noise), etc.
    
    # Normalize quotes and apostrophes
    text = re.sub(r'[""''`]', '', text)
    
    # Remove all punctuation except spaces
    text = re.sub(r'[^\w\s]', '', text)
    
    # Normalize whitespace and remove extra spaces
    text = re.sub(r'\s+', '', text)  # Remove all spaces for matching
    
    return text.strip()

def show_difference(str1, str2):
    """Show the difference positions between two strings"""
    min_len = min(len(str1), len(str2))
    diff_positions = []
    
    for i in range(min_len):
        if str1[i] != str2[i]:
            diff_positions.append(i)
    
    if len(str1) != len(str2):
        diff_positions.extend(range(min_len, max(len(str1), len(str2))))
    
    print("Difference positions:")
    print(f"Expected sentence: {str1}")
    print(f"Actual match: {str2}")
    print("Position markers: " + "".join("^" if i in diff_positions else " " for i in range(max(len(str1), len(str2)))))
    print(f"Difference indices: {diff_positions}")

def get_sentence_timestamps(df_words, df_sentences):
    time_stamp_list = []
    
    # Build complete string and position mapping
    full_words_str = ''
    position_to_word_idx = {}
    
    for idx, word in enumerate(df_words['text']):
        clean_word = normalize_sentence_for_matching(word)
        start_pos = len(full_words_str)
        full_words_str += clean_word
        for pos in range(start_pos, len(full_words_str)):
            position_to_word_idx[pos] = idx
    
    current_pos = 0
    for idx, sentence in df_sentences['Source'].items():
        # 检查并处理NaN值
        if pd.isna(sentence):
            print(f"⚠️ Warning: Skipping NaN sentence at index {idx}")
            # 为NaN值添加默认时间戳（使用前一个有效时间戳或0）
            if time_stamp_list:
                last_timestamp = time_stamp_list[-1]
                time_stamp_list.append(last_timestamp)
            else:
                time_stamp_list.append((0.0, 0.1))  # 默认很短的时间戳
            continue
        
        # 使用增强的句子标准化函数
        clean_sentence = normalize_sentence_for_matching(sentence)
        sentence_len = len(clean_sentence)
        
        # 添加调试信息
        print(f"🔍 Processing sentence {idx}: '{sentence[:50]}...'")
        print(f"📝 Normalized: '{clean_sentence[:50]}...'" if len(clean_sentence) > 50 else f"📝 Normalized: '{clean_sentence}'")
        print(f"📏 Length: {sentence_len}")
        
        match_found = False
        best_match_pos = current_pos
        best_match_score = 0
        
        # Try exact match first
        search_start = max(0, current_pos - 50)  # Allow some backward search
        search_end = min(len(full_words_str), current_pos + len(clean_sentence) + 100)
        
        for pos in range(search_start, search_end - sentence_len + 1):
            if full_words_str[pos:pos+sentence_len] == clean_sentence:
                start_word_idx = position_to_word_idx[pos]
                end_word_idx = position_to_word_idx[pos + sentence_len - 1]
                
                time_stamp_list.append((
                    float(df_words['start'][start_word_idx]),
                    float(df_words['end'][end_word_idx])
                ))
                
                current_pos = pos + sentence_len
                match_found = True
                break
        
        # If exact match not found, try fuzzy matching
        if not match_found:
            print(f"\n⚠️ Warning: No exact match found for sentence: {sentence}")
            print(f"🔍 Attempting fuzzy matching...")
            
            # Try different window sizes for better matching
            for window_adjustment in [0, -5, 5, -10, 10]:
                adjusted_len = max(1, sentence_len + window_adjustment)
                
                for pos in range(search_start, search_end - adjusted_len + 1):
                    candidate = full_words_str[pos:pos+adjusted_len]
                    
                    # Use Levenshtein-like similarity calculation
                    min_len = min(len(clean_sentence), len(candidate))
                    max_len = max(len(clean_sentence), len(candidate))
                    
                    # Character-level matching
                    matches = sum(1 for a, b in zip(clean_sentence, candidate) if a == b)
                    
                    # Penalize length differences
                    length_penalty = abs(len(clean_sentence) - len(candidate)) / max_len
                    score = (matches / max_len) * (1 - length_penalty * 0.3)
                    
                    if score > best_match_score:
                        best_match_score = score
                        best_match_pos = pos
                        sentence_len = adjusted_len  # Update for position calculation
            
            # Accept match if similarity > 60% (lowered threshold for better coverage)
            if best_match_score > 0.6:
                print(f"✅ Found fuzzy match with {best_match_score:.2%} similarity")
                start_word_idx = position_to_word_idx[best_match_pos]
                end_word_idx = position_to_word_idx[min(best_match_pos + sentence_len - 1, len(full_words_str) - 1)]
                
                time_stamp_list.append((
                    float(df_words['start'][start_word_idx]),
                    float(df_words['end'][end_word_idx])
                ))
                
                current_pos = best_match_pos + sentence_len
                match_found = True
            else:
                # If still no good match, use approximate positioning
                print(f"⚠️ Using approximate positioning (best score: {best_match_score:.2%})")
                show_difference(clean_sentence, 
                              full_words_str[current_pos:current_pos+len(clean_sentence)])
                print("\nOriginal sentence:", df_sentences['Source'][idx])
                
                # Use current position as fallback
                if current_pos < len(position_to_word_idx):
                    start_word_idx = position_to_word_idx[current_pos]
                    end_pos = min(current_pos + sentence_len - 1, len(full_words_str) - 1)
                    end_word_idx = position_to_word_idx[end_pos]
                    
                    time_stamp_list.append((
                        float(df_words['start'][start_word_idx]),
                        float(df_words['end'][end_word_idx])
                    ))
                    
                    current_pos += sentence_len
                else:
                    # Last resort: use the last available timestamp
                    last_idx = len(df_words) - 1
                    time_stamp_list.append((
                        float(df_words['start'][last_idx]),
                        float(df_words['end'][last_idx])
                    ))
    
    return time_stamp_list

def align_timestamp(df_text, df_translate, subtitle_output_configs: list, output_dir: str, for_display: bool = True):
    """Align timestamps and add a new timestamp column to df_translate"""
    df_trans_time = df_translate.copy()

    # Assign an ID to each word in df_text['text'] and create a new DataFrame
    words = df_text['text'].str.split(expand=True).stack().reset_index(level=1, drop=True).reset_index()
    words.columns = ['id', 'word']
    words['id'] = words['id'].astype(int)

    # Process timestamps ⏰
    time_stamp_list = get_sentence_timestamps(df_text, df_translate)
    df_trans_time['timestamp'] = time_stamp_list
    df_trans_time['duration'] = df_trans_time['timestamp'].apply(lambda x: x[1] - x[0])

    # Remove gaps 🕳️
    for i in range(len(df_trans_time)-1):
        delta_time = df_trans_time.loc[i+1, 'timestamp'][0] - df_trans_time.loc[i, 'timestamp'][1]
        if 0 < delta_time < 1:
            df_trans_time.at[i, 'timestamp'] = (df_trans_time.loc[i, 'timestamp'][0], df_trans_time.loc[i+1, 'timestamp'][0])

    # Convert start and end timestamps to SRT format
    df_trans_time['timestamp'] = df_trans_time['timestamp'].apply(lambda x: convert_to_srt_format(x[0], x[1]))

    # Polish subtitles: replace punctuation in Translation if for_display
    if for_display:
        def safe_polish_translation(x):
            """安全地处理翻译文本，避免NaN值导致的错误"""
            if pd.isna(x):
                return ''
            return re.sub(r'[，。]', ' ', str(x)).strip()
        
        df_trans_time['Translation'] = df_trans_time['Translation'].apply(safe_polish_translation)

    # Output subtitles 📜
    def generate_subtitle_string(df, columns):
        def safe_strip(value):
            """安全地处理可能为NaN或非字符串的值"""
            if pd.isna(value):
                return ''
            return str(value).strip()
        
        subtitle_lines = []
        for i, row in df.iterrows():
            # 安全处理第一列（通常是源语言）
            first_col = safe_strip(row[columns[0]])
            
            # 安全处理第二列（通常是翻译）
            second_col = safe_strip(row[columns[1]]) if len(columns) > 1 else ''
            
            # 构建字幕条目
            subtitle_entry = f"{i+1}\n{row['timestamp']}\n{first_col}\n{second_col}\n\n"
            subtitle_lines.append(subtitle_entry)
        
        return ''.join(subtitle_lines).strip()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for filename, columns in subtitle_output_configs:
            subtitle_str = generate_subtitle_string(df_trans_time, columns)
            with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                f.write(subtitle_str)
    
    return df_trans_time

# ✨ Beautify the translation
def clean_translation(x):
    if pd.isna(x):
        return ''
    cleaned = str(x).strip('。').strip('，')
    return autocorrect.format(cleaned)

def align_timestamp_main():
    df_text = pd.read_excel(_2_CLEANED_CHUNKS)
    df_text['text'] = df_text['text'].str.strip('"').str.strip()
    df_translate = pd.read_excel(_5_SPLIT_SUB)
    df_translate['Translation'] = df_translate['Translation'].apply(clean_translation)
    
    align_timestamp(df_text, df_translate, SUBTITLE_OUTPUT_CONFIGS, _OUTPUT_DIR)
    console.print(Panel("[bold green]🎉📝 Subtitles generation completed! Please check in the `output` folder 👀[/bold green]"))

    # for audio
    df_translate_for_audio = pd.read_excel(_5_REMERGED) # use remerged file to avoid unmatched lines when dubbing
    df_translate_for_audio['Translation'] = df_translate_for_audio['Translation'].apply(clean_translation)
    
    align_timestamp(df_text, df_translate_for_audio, AUDIO_SUBTITLE_OUTPUT_CONFIGS, _AUDIO_DIR)
    console.print(Panel(f"[bold green]🎉📝 Audio subtitles generation completed! Please check in the `{_AUDIO_DIR}` folder 👀[/bold green]"))
    

if __name__ == '__main__':
    align_timestamp_main()
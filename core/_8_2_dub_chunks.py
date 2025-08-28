import datetime
import re
import pandas as pd
from difflib import SequenceMatcher
from core._8_1_audio_task import time_diff_seconds
from core.asr_backend.audio_preprocess import get_audio_duration
from core.tts_backend.estimate_duration import init_estimator, estimate_duration
from core.utils import *
from core.utils.models import *

SRC_SRT = "output/src.srt"
TRANS_SRT = "output/trans.srt"
MAX_MERGE_COUNT = 5
ESTIMATOR = None

def calc_if_too_fast(est_dur, tol_dur, duration, tolerance):
    accept = load_key("speed_factor.accept") # Maximum acceptable speed factor
    if est_dur / accept > tol_dur:  # Even max speed factor cannot adapt
        return 2
    elif est_dur > tol_dur:  # Speed adjustment needed within acceptable range
        return 1
    elif est_dur < duration - tolerance:  # Speaking speed too slow
        return -1
    else:  # Normal speaking speed
        return 0

def merge_rows(df, start_idx, merge_count):
    """Merge multiple rows and calculate cumulative values"""
    merged = {
        'est_dur': df.iloc[start_idx]['est_dur'],
        'tol_dur': df.iloc[start_idx]['tol_dur'],
        'duration': df.iloc[start_idx]['duration']
    }
    
    while merge_count < MAX_MERGE_COUNT and (start_idx + merge_count) < len(df):
        next_row = df.iloc[start_idx + merge_count]
        merged['est_dur'] += next_row['est_dur']
        merged['tol_dur'] += next_row['tol_dur']
        merged['duration'] += next_row['duration']
        
        speed_flag = calc_if_too_fast(
            merged['est_dur'],
            merged['tol_dur'],
            merged['duration'],
            df.iloc[start_idx + merge_count]['tolerance']
        )
        
        if speed_flag <= 0 or merge_count == 2:
            df.at[start_idx + merge_count, 'cut_off'] = 1
            return merge_count + 1
        
        merge_count += 1
    
    # If no suitable merge point is found
    if merge_count >= MAX_MERGE_COUNT or (start_idx + merge_count) >= len(df):
        df.at[start_idx + merge_count - 1, 'cut_off'] = 1
    return merge_count

def analyze_subtitle_timing_and_speed(df):
    rprint("[🔍 Analyzing] Calculating subtitle timing and speed...")
    global ESTIMATOR
    if ESTIMATOR is None:
        ESTIMATOR = init_estimator()
    TOLERANCE = load_key("tolerance")
    whole_dur = get_audio_duration(_RAW_AUDIO_FILE)
    df['gap'] = 0.0  # Initialize gap column
    for i in range(len(df) - 1):
        current_end = datetime.datetime.strptime(df.loc[i, 'end_time'], '%H:%M:%S.%f').time()
        next_start = datetime.datetime.strptime(df.loc[i + 1, 'start_time'], '%H:%M:%S.%f').time()
        df.loc[i, 'gap'] = time_diff_seconds(current_end, next_start, datetime.date.today())
    
    # Set the gap for the last line
    last_end = datetime.datetime.strptime(df.iloc[-1]['end_time'], '%H:%M:%S.%f').time()
    last_end_seconds = (last_end.hour * 3600 + last_end.minute * 60 + 
                       last_end.second + last_end.microsecond / 1000000)
    df.iloc[-1, df.columns.get_loc('gap')] = whole_dur - last_end_seconds
    
    df['tolerance'] = df['gap'].apply(lambda x: TOLERANCE if x > TOLERANCE else x)
    df['tol_dur'] = df['duration'] + df['tolerance']
    df['est_dur'] = df.apply(lambda x: estimate_duration(x['text'], ESTIMATOR), axis=1)

    ## Calculate speed indicators
    accept = load_key("speed_factor.accept") # Maximum acceptable speed factor
    def calc_if_too_fast(row):
        est_dur = row['est_dur']
        tol_dur = row['tol_dur']
        duration = row['duration']
        tolerance = row['tolerance']
        
        if est_dur / accept > tol_dur:  # Even max speed factor cannot adapt
            return 2
        elif est_dur > tol_dur:  # Speed adjustment needed within acceptable range
            return 1
        elif est_dur < duration - tolerance:  # Speaking speed too slow
            return -1
        else:  # Normal speaking speed
            return 0
    
    df['if_too_fast'] = df.apply(calc_if_too_fast, axis=1)
    return df

def process_cutoffs(df):
    rprint("[✂️ Processing] Generating cutoff points...")
    df['cut_off'] = 0  # Initialize cut_off column
    df.loc[df['gap'] >= load_key("tolerance"), 'cut_off'] = 1  # Set to 1 when gap is greater than TOLERANCE
    idx = 0
    while idx < len(df):
        # Process marked split points
        if df.iloc[idx]['cut_off'] == 1:
            if df.iloc[idx]['if_too_fast'] == 2:
                rprint(f"[⚠️ Warning] Line {idx} is too fast and cannot be fixed by speed adjustment")
            idx += 1
            continue

        # Process the last line
        if idx + 1 >= len(df):
            df.at[idx, 'cut_off'] = 1
            break

        # Process normal or slow lines
        if df.iloc[idx]['if_too_fast'] <= 0:
            if df.iloc[idx + 1]['if_too_fast'] <= 0:
                df.at[idx, 'cut_off'] = 1
                idx += 1
            else:
                idx += merge_rows(df, idx, 1)
        # Process fast lines
        else:
            idx += merge_rows(df, idx, 1)
    
    return df

def gen_dub_chunks():
    rprint("[🎬 Starting] Generating dubbing chunks...")
    df = pd.read_excel(_8_1_AUDIO_TASK)
    
    rprint("[📊 Processing] Analyzing timing and speed...")
    df = analyze_subtitle_timing_and_speed(df)
    
    rprint("[✂️ Processing] Processing cutoffs...")
    df = process_cutoffs(df)

    rprint("[📝 Reading] Loading transcript files...")
    content = open(TRANS_SRT, "r", encoding="utf-8").read()
    ori_content = open(SRC_SRT, "r", encoding="utf-8").read()
    
    # Process subtitle content
    content_lines = []
    ori_content_lines = []
    
    # Process translated subtitles
    for block in content.strip().split('\n\n'):
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) >= 3:
            text = ' '.join(lines[2:])
            text = re.sub(r'\([^)]*\)|（[^）]*）', '', text).strip().replace('-', '')
            content_lines.append(text)
            
    # Process source subtitles (same structure)
    for block in ori_content.strip().split('\n\n'):
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) >= 3:
            text = ' '.join(lines[2:])
            text = re.sub(r'\([^)]*\)|（[^）]*）', '', text).strip().replace('-', '')
            ori_content_lines.append(text)

    # Match processing
    df['lines'] = None
    df['src_lines'] = None
    last_idx = 0

    def clean_text(text, level='strict'):
        """Clean text with different levels of strictness"""
        if not text or not isinstance(text, str):
            return ''
        
        if level == 'strict':
            # 移除所有标点和空格
            return re.sub(r'[^\w\s]|[\s]', '', text)
        elif level == 'moderate':
            # 只移除标点，保留空格
            return re.sub(r'[^\w\s]', '', text).strip()
        elif level == 'loose':
            # 只移除多余空格
            return re.sub(r'\s+', ' ', text).strip()
        else:
            return text.strip()
    
    def calculate_similarity(text1, text2):
        """Calculate similarity between two texts"""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def fuzzy_match(target, candidates, threshold=0.8):
        """Find best fuzzy match from candidates"""
        best_match = None
        best_score = 0
        best_index = -1
        
        for i, candidate in enumerate(candidates):
            score = calculate_similarity(target, candidate)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = candidate
                best_index = i
        
        return best_match, best_score, best_index
    
    def try_multiple_match_strategies(target_text, content_lines, ori_content_lines, start_idx):
        """Try multiple matching strategies with increasing tolerance"""
        strategies = [
            ('strict', 1.0),      # 完全匹配
            ('moderate', 0.95),   # 高相似度匹配
            ('loose', 0.85),      # 中等相似度匹配
            ('fuzzy', 0.7)        # 模糊匹配
        ]
        
        for strategy, threshold in strategies:
            rprint(f"[🔍 Trying] {strategy} matching with threshold {threshold}")
            
            matches = []
            current = ''
            match_indices = []
            
            for i in range(start_idx, len(content_lines)):
                line = content_lines[i]
                cleaned_line = clean_text(line, 'moderate' if strategy == 'fuzzy' else strategy)
                current += cleaned_line
                matches.append(line)
                match_indices.append(i)
                
                # 计算相似度
                similarity = calculate_similarity(clean_text(target_text, 'moderate' if strategy == 'fuzzy' else strategy), current)
                
                if similarity >= threshold:
                    rprint(f"[✅ Success] {strategy} match found with similarity {similarity:.3f}")
                    return matches, [ori_content_lines[j] for j in match_indices], i + 1, similarity
                
                # 如果累积了太多行还没匹配，尝试下一个策略
                if len(matches) > 10:
                    break
        
        return None, None, start_idx, 0.0

    # 添加匹配统计
    total_matches = 0
    successful_matches = 0
    match_strategies_used = {'strict': 0, 'moderate': 0, 'loose': 0, 'fuzzy': 0}
    
    for idx, row in df.iterrows():
        target_text = row['text']
        total_matches += 1
        
        rprint(f"[🎯 Matching] Processing line {idx}: '{target_text[:50]}...'") 
        
        # 尝试多种匹配策略
        matches, src_matches, new_last_idx, similarity = try_multiple_match_strategies(
            target_text, content_lines, ori_content_lines, last_idx
        )
        
        if matches is not None:
            df.at[idx, 'lines'] = matches
            df.at[idx, 'src_lines'] = src_matches
            last_idx = new_last_idx
            successful_matches += 1
            
            # 记录使用的策略
            if similarity == 1.0:
                match_strategies_used['strict'] += 1
            elif similarity >= 0.95:
                match_strategies_used['moderate'] += 1
            elif similarity >= 0.85:
                match_strategies_used['loose'] += 1
            else:
                match_strategies_used['fuzzy'] += 1
                
            rprint(f"[✅ Success] Line {idx} matched with similarity {similarity:.3f}")
        else:
            # 最后的容错处理：跳过这一行并使用空匹配
            rprint(f"[⚠️ Warning] No suitable match found for line {idx}, using fallback")
            rprint(f"Target text: '{target_text}'")
            
            # 提供回退选项
            fallback_matches = [target_text]  # 使用原文本作为回退
            fallback_src = [target_text]      # 源文本也使用原文本
            
            df.at[idx, 'lines'] = fallback_matches
            df.at[idx, 'src_lines'] = fallback_src
            
            # 不更新last_idx，继续从当前位置匹配
            rprint(f"[🔄 Fallback] Using original text as fallback for line {idx}")
    
    # 输出匹配统计
    success_rate = (successful_matches / total_matches) * 100 if total_matches > 0 else 0
    rprint(f"[📊 Statistics] Matching completed: {successful_matches}/{total_matches} ({success_rate:.1f}% success rate)")
    rprint(f"[📈 Strategies] Used: Strict={match_strategies_used['strict']}, Moderate={match_strategies_used['moderate']}, Loose={match_strategies_used['loose']}, Fuzzy={match_strategies_used['fuzzy']}")

    # Save results
    df.to_excel(_8_1_AUDIO_TASK, index=False)
    rprint("[✅ Complete] Matching completed successfully!")

if __name__ == "__main__":
    gen_dub_chunks()
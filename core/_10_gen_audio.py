import os
import time
import shutil
import subprocess
from typing import Tuple

import pandas as pd
from pydub import AudioSegment
from rich.console import Console
from rich.progress import Progress
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.utils import *
from core.utils.models import *
from core.asr_backend.audio_preprocess import get_audio_duration
from core.tts_backend.tts_main import tts_main

console = Console()

TEMP_FILE_TEMPLATE = f"{_AUDIO_TMP_DIR}/{{}}_temp.wav"
OUTPUT_FILE_TEMPLATE = f"{_AUDIO_SEGS_DIR}/{{}}.wav"
WARMUP_SIZE = 5

def parse_df_srt_time(time_str: str) -> float:
    """Convert SRT time format to seconds"""
    hours, minutes, seconds = time_str.strip().split(':')
    seconds, milliseconds = seconds.split('.')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000

def adjust_audio_speed(input_file: str, output_file: str, speed_factor: float) -> None:
    """Adjust audio speed using ffmpeg with proper atempo range handling"""
    if abs(speed_factor - 1.0) < 0.001:
        shutil.copy2(input_file, output_file)
        return
    
    # FFmpeg atempo filter supports range 0.5-100.0
    # For values outside this range, we need to chain multiple atempo filters or clamp the value
    def build_atempo_filter(factor: float) -> str:
        """Build atempo filter chain for speed factors outside normal range"""
        if 0.5 <= factor <= 100.0:
            return f'atempo={factor}'
        
        # For extreme values, clamp to safe range and warn user
        if factor < 0.5:
            rprint(f"[yellow]⚠️ Speed factor {factor} too low, clamping to 0.5[/yellow]")
            return 'atempo=0.5'
        elif factor > 100.0:
            rprint(f"[yellow]⚠️ Speed factor {factor} too high, clamping to 100.0[/yellow]")
            return 'atempo=100.0'
        
        return f'atempo={factor}'
    
    atempo_filter = build_atempo_filter(speed_factor)
    cmd = ['ffmpeg', '-i', input_file, '-filter:a', atempo_filter, '-y', output_file]
    input_duration = get_audio_duration(input_file)
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            result = subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            output_duration = get_audio_duration(output_file)
            expected_duration = input_duration / speed_factor
            diff = output_duration - expected_duration
            
            # If the output duration exceeds the expected duration, but the input audio is less than 3 seconds, and the error is within 0.1 seconds, truncate to the expected length
            if output_duration >= expected_duration * 1.02 and input_duration < 3 and diff <= 0.1:
                audio = AudioSegment.from_wav(output_file)
                trimmed_audio = audio[:(expected_duration * 1000)]  # pydub uses milliseconds
                trimmed_audio.export(output_file, format="wav")
                print(f"✂️ Trimmed to expected duration: {expected_duration:.2f} seconds")
                return
            elif output_duration >= expected_duration * 1.02:
                raise Exception(f"Audio duration abnormal: input file={input_file}, output file={output_file}, speed factor={speed_factor}, input duration={input_duration:.2f}s, output duration={output_duration:.2f}s")
            return
            
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode('utf-8') if e.stderr else 'Unknown error'
            if attempt < max_retries - 1:
                rprint(f"[yellow]⚠️ Audio speed adjustment failed, retrying in 1s ({attempt + 1}/{max_retries})[/yellow]")
                rprint(f"[yellow]Error details: {error_msg}[/yellow]")
                time.sleep(1)
            else:
                rprint(f"[red]❌ Audio speed adjustment failed, max retries reached ({max_retries})[/red]")
                rprint(f"[red]Final error: {error_msg}[/red]")
                rprint(f"[red]Command: {' '.join(cmd)}[/red]")
                raise e
        except Exception as e:
            rprint(f"[red]❌ Unexpected error in audio speed adjustment: {str(e)}[/red]")
            raise e

def process_row(row: pd.Series, tasks_df: pd.DataFrame) -> Tuple[int, float]:
    """Helper function for processing single row data"""
    number = row['number']
    lines = eval(row['lines']) if isinstance(row['lines'], str) else row['lines']
    real_dur = 0
    for line_index, line in enumerate(lines):
        temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
        tts_main(line, temp_file, number, tasks_df)
        real_dur += get_audio_duration(temp_file)
    return number, real_dur

def generate_tts_audio(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Generate TTS audio sequentially and calculate actual duration"""
    tasks_df['real_dur'] = 0
    rprint("[bold green]🎯 Starting TTS audio generation...[/bold green]")
    
    with Progress() as progress:
        task = progress.add_task("[cyan]🔄 Generating TTS audio...", total=len(tasks_df))
        
        # warm up for first 5 rows
        warmup_size = min(WARMUP_SIZE, len(tasks_df))
        for _, row in tasks_df.head(warmup_size).iterrows():
            try:
                number, real_dur = process_row(row, tasks_df)
                tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = real_dur
                progress.advance(task)
            except Exception as e:
                rprint(f"[red]❌ Error in warmup: {str(e)}[/red]")
                raise e
        
        # for gpt_sovits, do not use parallel to avoid mistakes
        max_workers = load_key("max_workers") if load_key("tts_method") != "gpt_sovits" else 1
        # parallel processing for remaining tasks
        if len(tasks_df) > warmup_size:
            remaining_tasks = tasks_df.iloc[warmup_size:].copy()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_row, row, tasks_df.copy())
                    for _, row in remaining_tasks.iterrows()
                ]
                
                for future in as_completed(futures):
                    try:
                        number, real_dur = future.result()
                        tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = real_dur
                        progress.advance(task)
                    except Exception as e:
                        rprint(f"[red]❌ Error: {str(e)}[/red]")
                        raise e

    rprint("[bold green]✨ TTS audio generation completed![/bold green]")
    return tasks_df

def process_chunk(chunk_df: pd.DataFrame, accept: float, min_speed: float) -> tuple[float, bool]:
    """Process audio chunk and calculate speed factor with safety limits"""
    chunk_durs = chunk_df['real_dur'].sum()
    tol_durs = chunk_df['tol_dur'].sum()
    durations = tol_durs - chunk_df.iloc[-1]['tolerance']
    all_gaps = chunk_df['gap'].sum() - chunk_df.iloc[-1]['gap']
    
    # 防止除零错误
    if durations <= 0.1:
        rprint(f"[yellow]⚠️ Warning: Very small duration value ({durations:.3f}s), using minimum safe value[/yellow]")
        durations = 0.1
    if tol_durs <= 0.1:
        rprint(f"[yellow]⚠️ Warning: Very small tolerance duration value ({tol_durs:.3f}s), using minimum safe value[/yellow]")
        tol_durs = 0.1
    
    keep_gaps = True
    speed_var_error = 0.1
    
    # 计算速度因子前记录原始值用于日志
    raw_speed_factor = 0
    
    if (chunk_durs + all_gaps) / accept < durations:
        raw_speed_factor = (chunk_durs + all_gaps) / (durations-speed_var_error)
        speed_factor = max(min_speed, raw_speed_factor)
    elif chunk_durs / accept < durations:
        raw_speed_factor = chunk_durs / (durations-speed_var_error)
        speed_factor = max(min_speed, raw_speed_factor)
        keep_gaps = False
    elif (chunk_durs + all_gaps) / accept < tol_durs:
        raw_speed_factor = (chunk_durs + all_gaps) / (tol_durs-speed_var_error)
        speed_factor = max(min_speed, raw_speed_factor)
    else:
        raw_speed_factor = chunk_durs / (tol_durs-speed_var_error)
        speed_factor = max(min_speed, raw_speed_factor)
        keep_gaps = False
    
    # 限制最大速度因子，防止FFmpeg atempo滤镜错误
    MAX_SPEED_FACTOR = 100.0
    if speed_factor > MAX_SPEED_FACTOR:
        rprint(f"[yellow]⚠️ Calculated speed factor {speed_factor:.3f} exceeds maximum allowed value, clamping to {MAX_SPEED_FACTOR}[/yellow]")
        rprint(f"[yellow]Original values: chunk_durs={chunk_durs:.3f}s, durations={durations:.3f}s, raw_factor={raw_speed_factor:.3f}[/yellow]")
        speed_factor = MAX_SPEED_FACTOR
    
    # 记录异常高的速度因子
    if speed_factor > 10.0:
        rprint(f"[yellow]⚠️ High speed factor detected: {speed_factor:.3f}[/yellow]")
        rprint(f"[yellow]Debug info: chunk_durs={chunk_durs:.3f}s, durations={durations:.3f}s, tol_durs={tol_durs:.3f}s, gaps={all_gaps:.3f}s[/yellow]")
        
    return round(speed_factor, 3), keep_gaps

def merge_chunks(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Merge audio chunks and adjust timeline"""
    rprint("[bold blue]🔄 Starting audio chunks processing...[/bold blue]")
    accept = load_key("speed_factor.accept")
    min_speed = load_key("speed_factor.min")
    chunk_start = 0
    
    tasks_df['new_sub_times'] = None
    
    for index, row in tasks_df.iterrows():
        if row['cut_off'] == 1:
            chunk_df = tasks_df.iloc[chunk_start:index+1].reset_index(drop=True)
            speed_factor, keep_gaps = process_chunk(chunk_df, accept, min_speed)
            
            # 🎯 Step1: Start processing new timeline
            chunk_start_time = parse_df_srt_time(chunk_df.iloc[0]['start_time'])
            chunk_end_time = parse_df_srt_time(chunk_df.iloc[-1]['end_time']) + chunk_df.iloc[-1]['tolerance'] # 加上tolerance才是这一块的结束
            cur_time = chunk_start_time
            for i, row in chunk_df.iterrows():
                # If i is not 0, which is not the first row of the chunk, cur_time needs to be added with the gap of the previous row, remember to divide by speed_factor
                if i != 0 and keep_gaps:
                    cur_time += chunk_df.iloc[i-1]['gap']/speed_factor
                new_sub_times = []
                number = row['number']
                lines = eval(row['lines']) if isinstance(row['lines'], str) else row['lines']
                for line_index, line in enumerate(lines):
                    # 🔄 Step2: Start speed change and save as OUTPUT_FILE_TEMPLATE
                    temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
                    output_file = OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}")
                    adjust_audio_speed(temp_file, output_file, speed_factor)
                    ad_dur = get_audio_duration(output_file)
                    new_sub_times.append([cur_time, cur_time+ad_dur])
                    cur_time += ad_dur
                # 🔄 Step3: Find corresponding main DataFrame index and update new_sub_times
                main_df_idx = tasks_df[tasks_df['number'] == row['number']].index[0]
                tasks_df.at[main_df_idx, 'new_sub_times'] = new_sub_times
                # 🎯 Step4: Choose emoji based on speed_factor and accept comparison
                emoji = "⚡" if speed_factor <= accept else "⚠️"
                rprint(f"[cyan]{emoji} Processed chunk {chunk_start} to {index} with speed factor {speed_factor}[/cyan]")
            # 🔄 Step5: Check if the last row exceeds the range
            if cur_time > chunk_end_time:
                time_diff = cur_time - chunk_end_time
                
                # 从配置文件读取时间容忍度设置
                try:
                    max_tolerance = load_key("audio_chunk.time_tolerance")
                except (KeyError, Exception):
                    max_tolerance = 10.0  # 默认值
                
                try:
                    min_tolerance = load_key("audio_chunk.min_tolerance")
                except (KeyError, Exception):
                    min_tolerance = 0.6  # 默认值
                
                rprint(f"[yellow]⚠️ Chunk {chunk_start} to {index} exceeds by {time_diff:.3f}s (max tolerance: {max_tolerance}s)[/yellow]")
                
                if time_diff <= max_tolerance:  # 使用配置的最大容忍度
                    rprint(f"[cyan]🔧 Time difference {time_diff:.3f}s is within tolerance, attempting audio truncation...[/cyan]")
                    
                    # 智能音频截断：从最后一个音频开始，逐步截断直到时间符合要求
                    remaining_time_diff = time_diff
                    truncated_files = []
                    
                    # 获取当前chunk的所有音频文件（从后往前处理）
                    chunk_audio_files = []
                    for chunk_idx in range(chunk_start, index + 1):
                        chunk_row = tasks_df.iloc[chunk_idx]
                        chunk_number = chunk_row['number']
                        chunk_lines = eval(chunk_row['lines']) if isinstance(chunk_row['lines'], str) else chunk_row['lines']
                        for line_idx in range(len(chunk_lines)):
                            audio_file = OUTPUT_FILE_TEMPLATE.format(f"{chunk_number}_{line_idx}")
                            chunk_audio_files.append((chunk_idx, chunk_number, line_idx, audio_file))
                    
                    # 从最后一个音频文件开始截断
                    for chunk_idx, chunk_number, line_idx, audio_file in reversed(chunk_audio_files):
                        if remaining_time_diff <= 0.01:  # 精度阈值
                            break
                            
                        try:
                            audio = AudioSegment.from_wav(audio_file)
                            original_duration = len(audio) / 1000  # Convert to seconds
                            
                            if original_duration > remaining_time_diff:
                                # 截断这个音频文件
                                new_duration = original_duration - remaining_time_diff
                                if new_duration > 0.1:  # 确保截断后的音频不会太短
                                    trimmed_audio = audio[:(new_duration * 1000)]  # pydub uses milliseconds
                                    trimmed_audio.export(audio_file, format="wav")
                                    truncated_files.append((audio_file, remaining_time_diff))
                                    
                                    # 更新对应的时间戳
                                    if chunk_idx == index:  # 如果是最后一行
                                        last_times = tasks_df.at[chunk_idx, 'new_sub_times']
                                        if last_times and line_idx < len(last_times):
                                            last_times[line_idx][1] -= remaining_time_diff
                                            tasks_df.at[chunk_idx, 'new_sub_times'] = last_times
                                    
                                    rprint(f"[green]✅ Truncated {audio_file} by {remaining_time_diff:.3f}s[/green]")
                                    remaining_time_diff = 0
                                    break
                                else:
                                    # 如果截断后太短，删除整个音频文件
                                    remaining_time_diff -= original_duration
                                    truncated_files.append((audio_file, original_duration))
                                    # 创建一个很短的静音文件替代
                                    silent_audio = AudioSegment.silent(duration=100)  # 0.1秒静音
                                    silent_audio.export(audio_file, format="wav")
                                    rprint(f"[yellow]⚠️ Replaced {audio_file} with silence (original: {original_duration:.3f}s)[/yellow]")
                            else:
                                # 整个音频文件都需要被移除的时间覆盖
                                remaining_time_diff -= original_duration
                                truncated_files.append((audio_file, original_duration))
                                # 创建一个很短的静音文件替代
                                silent_audio = AudioSegment.silent(duration=100)  # 0.1秒静音
                                silent_audio.export(audio_file, format="wav")
                                rprint(f"[yellow]⚠️ Replaced {audio_file} with silence (removed: {original_duration:.3f}s)[/yellow]")
                                
                        except Exception as audio_error:
                            rprint(f"[red]❌ Error processing {audio_file}: {str(audio_error)}[/red]")
                            continue
                    
                    if truncated_files:
                        total_truncated = sum(duration for _, duration in truncated_files)
                        rprint(f"[green]✅ Successfully truncated {len(truncated_files)} audio files, total time reduced: {total_truncated:.3f}s[/green]")
                    
                    # 最终验证：重新计算当前时间
                    final_cur_time = chunk_start_time
                    for i, row in chunk_df.iterrows():
                        if i != 0 and keep_gaps:
                            final_cur_time += chunk_df.iloc[i-1]['gap']/speed_factor
                        number = row['number']
                        lines = eval(row['lines']) if isinstance(row['lines'], str) else row['lines']
                        for line_index, line in enumerate(lines):
                            output_file = OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}")
                            try:
                                ad_dur = get_audio_duration(output_file)
                                final_cur_time += ad_dur
                            except:
                                rprint(f"[yellow]⚠️ Could not get duration for {output_file}, using estimated duration[/yellow]")
                                final_cur_time += 1.0  # 默认1秒
                    
                    if final_cur_time <= chunk_end_time + 0.1:  # 允许0.1秒的精度误差
                        rprint(f"[green]✅ Time adjustment successful: {final_cur_time:.3f}s <= {chunk_end_time:.3f}s[/green]")
                    else:
                        rprint(f"[yellow]⚠️ Time still exceeds after truncation: {final_cur_time:.3f}s > {chunk_end_time:.3f}s[/yellow]")
                        rprint(f"[yellow]   Remaining difference: {final_cur_time - chunk_end_time:.3f}s (acceptable if < {max_tolerance}s)[/yellow]")
                        
                else:
                    # 超出最大容忍度，提供详细的错误信息和建议
                    rprint(f"[red]❌ Chunk {chunk_start} to {index} exceeds maximum time tolerance[/red]")
                    rprint(f"[red]   Expected end time: {chunk_end_time:.2f}s[/red]")
                    rprint(f"[red]   Actual end time: {cur_time:.2f}s[/red]")
                    rprint(f"[red]   Time difference: {time_diff:.2f}s[/red]")
                    rprint(f"[red]   Maximum tolerance: {max_tolerance:.2f}s[/red]")
                    rprint(f"[yellow]💡 Suggestions:[/yellow]")
                    rprint(f"[yellow]   1. Increase 'audio_chunk.time_tolerance' in config.yaml[/yellow]")
                    rprint(f"[yellow]   2. Adjust speed_factor settings to reduce audio duration[/yellow]")
                    rprint(f"[yellow]   3. Check if TTS is generating audio longer than expected[/yellow]")
                    raise Exception(f"Chunk {chunk_start} to {index} exceeds the maximum time tolerance {max_tolerance:.2f}s with time difference {time_diff:.2f}s")
            chunk_start = index+1
    
    rprint("[bold green]✅ Audio chunks processing completed![/bold green]")
    return tasks_df

def gen_audio() -> None:
    """Main function: Generate audio and process timeline"""
    rprint("[bold magenta]🚀 Starting audio generation process...[/bold magenta]")
    
    # 🎯 Step1: Create necessary directories
    os.makedirs(_AUDIO_TMP_DIR, exist_ok=True)
    os.makedirs(_AUDIO_SEGS_DIR, exist_ok=True)
    
    # 📝 Step2: Load task file
    tasks_df = pd.read_excel(_8_1_AUDIO_TASK)
    rprint("[green]📊 Loaded task file successfully[/green]")
    
    # 🔊 Step3: Generate TTS audio
    tasks_df = generate_tts_audio(tasks_df)
    
    # 🔄 Step4: Merge audio chunks
    tasks_df = merge_chunks(tasks_df)
    
    # 💾 Step5: Save results
    tasks_df.to_excel(_8_1_AUDIO_TASK, index=False)
    rprint("[bold green]🎉 Audio generation completed successfully![/bold green]")

if __name__ == "__main__":
    gen_audio()

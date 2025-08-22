import os
import pandas as pd
import subprocess
from pydub import AudioSegment
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console
from core.utils import *
from core.utils.models import *
console = Console()

DUB_VOCAL_FILE = 'output/dub.mp3'

DUB_SUB_FILE = 'output/dub.srt'
OUTPUT_FILE_TEMPLATE = f"{_AUDIO_SEGS_DIR}/{{}}.wav"

def load_and_flatten_data(excel_file):
    """Load and flatten Excel data"""
    df = pd.read_excel(excel_file)
    lines = [eval(line) if isinstance(line, str) else line for line in df['lines'].tolist()]
    lines = [item for sublist in lines for item in sublist]
    
    new_sub_times = [eval(time) if isinstance(time, str) else time for time in df['new_sub_times'].tolist()]
    new_sub_times = [item for sublist in new_sub_times for item in sublist]
    
    return df, lines, new_sub_times

def get_audio_files(df):
    """Generate a list of audio file paths"""
    audios = []
    for index, row in df.iterrows():
        number = row['number']
        line_count = len(eval(row['lines']) if isinstance(row['lines'], str) else row['lines'])
        for line_index in range(line_count):
            temp_file = OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}")
            audios.append(temp_file)
    return audios

def process_audio_segment(audio_file):
    """Process a single audio segment with MP3 compression and error handling"""
    # 检查原始音频文件是否存在和有效
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    # 检查文件大小
    file_size = os.path.getsize(audio_file)
    if file_size < 1000:  # 小于1KB的文件可能损坏
        raise ValueError(f"Audio file too small (possibly corrupted): {audio_file} ({file_size} bytes)")
    
    temp_file = f"{audio_file}_temp.mp3"
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # 尝试直接加载WAV文件作为备用方案
            if attempt == 0:
                try:
                    # 先尝试直接读取WAV文件
                    audio_segment = AudioSegment.from_wav(audio_file)
                    # 转换为目标格式
                    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                    return audio_segment
                except Exception:
                    # 如果直接读取失败，继续使用FFmpeg转换
                    pass
            
            # 使用FFmpeg转换
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', audio_file,
                '-ar', '16000',
                '-ac', '1',
                '-b:a', '64k',
                temp_file
            ]
            
            result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # 检查FFmpeg是否成功执行
            if result.returncode != 0:
                error_msg = result.stderr
                if attempt < max_retries - 1:
                    console.print(f"[yellow]⚠️ FFmpeg conversion failed (attempt {attempt + 1}/{max_retries}): {error_msg[:200]}[/yellow]")
                    continue
                else:
                    raise subprocess.CalledProcessError(result.returncode, ffmpeg_cmd, result.stderr)
            
            # 检查生成的临时文件
            if not os.path.exists(temp_file) or os.path.getsize(temp_file) < 100:
                if attempt < max_retries - 1:
                    console.print(f"[yellow]⚠️ Generated temp file invalid (attempt {attempt + 1}/{max_retries})[/yellow]")
                    continue
                else:
                    raise ValueError(f"Failed to generate valid temp file: {temp_file}")
            
            # 尝试加载MP3文件
            audio_segment = AudioSegment.from_mp3(temp_file)
            
            # 清理临时文件
            try:
                os.remove(temp_file)
            except Exception as e:
                console.print(f"[yellow]⚠️ Failed to remove temp file {temp_file}: {e}[/yellow]")
            
            return audio_segment
            
        except Exception as e:
            # 清理可能存在的临时文件
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
            
            if attempt < max_retries - 1:
                console.print(f"[yellow]⚠️ Audio processing failed (attempt {attempt + 1}/{max_retries}): {str(e)[:200]}[/yellow]")
                continue
            else:
                # 最后一次尝试：生成静音音频作为备用
                console.print(f"[red]❌ All attempts failed for {audio_file}, generating silent audio as fallback[/red]")
                silent_audio = AudioSegment.silent(duration=1000, frame_rate=16000)  # 1秒静音
                return silent_audio.set_channels(1)
    
    # 理论上不会到达这里，但作为最后的保险
    return AudioSegment.silent(duration=1000, frame_rate=16000).set_channels(1)

def merge_audio_segments(audios, new_sub_times, sample_rate):
    """Merge audio segments with proper timing and enhanced error handling"""
    merged_audio = AudioSegment.empty()
    
    if not audios or not new_sub_times:
        console.print("[yellow]⚠️ No audio files or timing data provided[/yellow]")
        return AudioSegment.silent(duration=1000, frame_rate=sample_rate)
    
    if len(audios) != len(new_sub_times):
        console.print(f"[red]❌ Mismatch: {len(audios)} audio files vs {len(new_sub_times)} timing entries[/red]")
        return AudioSegment.silent(duration=1000, frame_rate=sample_rate)
    
    successful_segments = 0
    
    for i, (audio_file, (start_time, end_time)) in enumerate(zip(audios, new_sub_times)):
        try:
            console.print(f"[cyan]Processing audio segment {i+1}/{len(audios)}: {audio_file}[/cyan]")
            
            # 验证时间参数
            if start_time < 0 or end_time < 0 or start_time >= end_time:
                console.print(f"[yellow]⚠️ Invalid timing for segment {i+1}: start={start_time}, end={end_time}[/yellow]")
                # 使用默认时长的静音
                audio_segment = AudioSegment.silent(duration=1000, frame_rate=sample_rate)
            else:
                # Process the audio segment
                audio_segment = process_audio_segment(audio_file)
                
                # 验证音频段是否有效
                if len(audio_segment) == 0:
                    console.print(f"[yellow]⚠️ Empty audio segment for {audio_file}, using silence[/yellow]")
                    audio_segment = AudioSegment.silent(duration=1000, frame_rate=sample_rate)
            
            # Calculate timing
            current_length = len(merged_audio)
            target_start = int(start_time * 1000)  # Convert to milliseconds
            
            # Add silence if needed
            if target_start > current_length:
                silence_duration = target_start - current_length
                if silence_duration > 0:
                    silence = AudioSegment.silent(duration=silence_duration, frame_rate=sample_rate)
                    merged_audio += silence
            
            # Add the audio segment
            merged_audio += audio_segment
            successful_segments += 1
            
        except Exception as e:
            console.print(f"[red]❌ Failed to process segment {i+1} ({audio_file}): {str(e)[:200]}[/red]")
            # 添加静音作为占位符
            try:
                silence_duration = int((end_time - start_time) * 1000) if end_time > start_time else 1000
                silence = AudioSegment.silent(duration=silence_duration, frame_rate=sample_rate)
                merged_audio += silence
            except Exception:
                # 如果连静音都无法生成，添加默认静音
                silence = AudioSegment.silent(duration=1000, frame_rate=sample_rate)
                merged_audio += silence
    
    console.print(f"[green]✅ Successfully processed {successful_segments}/{len(audios)} audio segments[/green]")
    
    # 确保返回的音频不为空
    if len(merged_audio) == 0:
        console.print("[yellow]⚠️ Final merged audio is empty, returning default silence[/yellow]")
        return AudioSegment.silent(duration=5000, frame_rate=sample_rate)  # 5秒静音
    
    return merged_audio

def create_srt_subtitle():
    df, lines, new_sub_times = load_and_flatten_data(_8_1_AUDIO_TASK)
    
    with open(DUB_SUB_FILE, 'w', encoding='utf-8') as f:
        for i, ((start_time, end_time), line) in enumerate(zip(new_sub_times, lines), 1):
            start_str = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d},{int((start_time*1000)%1000):03d}"
            end_str = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d},{int((end_time*1000)%1000):03d}"
            
            f.write(f"{i}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"{line}\n\n")
    
    rprint(f"[bold green]✅ Subtitle file created: {DUB_SUB_FILE}[/bold green]")

def merge_full_audio():
    """Main function: Process the complete audio merging process"""
    console.print("\n[bold cyan]🎬 Starting audio merging process...[/bold cyan]")
    
    with console.status("[bold cyan]📊 Loading data from Excel...[/bold cyan]"):
        df, lines, new_sub_times = load_and_flatten_data(_8_1_AUDIO_TASK)
    console.print("[bold green]✅ Data loaded successfully[/bold green]")
    
    with console.status("[bold cyan]🔍 Getting audio file list...[/bold cyan]"):
        audios = get_audio_files(df)
    console.print(f"[bold green]✅ Found {len(audios)} audio segments[/bold green]")
    
    with console.status("[bold cyan]📝 Generating subtitle file...[/bold cyan]"):
        create_srt_subtitle()
    
    if not os.path.exists(audios[0]):
        console.print(f"[bold red]❌ Error: First audio file {audios[0]} does not exist![/bold red]")
        return
    
    sample_rate = 16000
    console.print(f"[bold green]✅ Sample rate: {sample_rate}Hz[/bold green]")

    console.print("[bold cyan]🔄 Starting audio merge process...[/bold cyan]")
    merged_audio = merge_audio_segments(audios, new_sub_times, sample_rate)
    
    with console.status("[bold cyan]💾 Exporting final audio file...[/bold cyan]"):
        merged_audio = merged_audio.set_frame_rate(16000).set_channels(1)
        merged_audio.export(DUB_VOCAL_FILE, format="mp3", parameters=["-b:a", "64k"])
    console.print(f"[bold green]✅ Audio file successfully merged![/bold green]")
    console.print(f"[bold green]📁 Output file: {DUB_VOCAL_FILE}[/bold green]")

if __name__ == "__main__":
    merge_full_audio()
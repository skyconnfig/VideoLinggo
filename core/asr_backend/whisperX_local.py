import os
import warnings
import time
import subprocess
import torch
import whisperx
import librosa
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rich import print as rprint
from core.utils import *

warnings.filterwarnings("ignore")
MODEL_DIR = load_key("model_dir")

@except_handler("failed to check hf mirror", default_return=None)
def check_hf_mirror():
    # 优先使用国内镜像源以提高下载速度和稳定性
    mirror_url = "https://hf-mirror.com"
    rprint("[cyan]🔍 Using HuggingFace China mirror for better download speed...[/cyan]")
    rprint(f"[cyan]🚀 Selected mirror:[/cyan] {mirror_url}")
    return mirror_url

def setup_robust_session():
    """设置带有重试机制的HTTP会话"""
    session = requests.Session()
    retry_strategy = Retry(
        total=5,  # 总重试次数
        backoff_factor=2,  # 退避因子
        status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的HTTP状态码
        allowed_methods=["HEAD", "GET", "OPTIONS"]  # 允许重试的HTTP方法
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def download_with_retry(url, local_path, max_retries=3):
    """带重试机制的文件下载"""
    session = setup_robust_session()
    
    for attempt in range(max_retries):
        try:
            rprint(f"[cyan]📥 Downloading attempt {attempt + 1}/{max_retries}...[/cyan]")
            response = session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            rprint(f"[green]✅ Download completed successfully![/green]")
            return True
            
        except (requests.exceptions.ChunkedEncodingError, 
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            rprint(f"[yellow]⚠️ Download attempt {attempt + 1} failed: {str(e)}[/yellow]")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                rprint(f"[cyan]⏳ Waiting {wait_time}s before retry...[/cyan]")
                time.sleep(wait_time)
            else:
                rprint(f"[red]❌ All download attempts failed![/red]")
                return False
        except Exception as e:
            rprint(f"[red]❌ Unexpected error during download: {str(e)}[/red]")
            return False
    
    return False

@except_handler("WhisperX processing error:")
def transcribe_audio(raw_audio_file, vocal_audio_file, start, end):
    os.environ['HF_ENDPOINT'] = check_hf_mirror()
    WHISPER_LANGUAGE = load_key("whisper.language")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rprint(f"🚀 Starting WhisperX using device: {device} ...")
    
    if device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        batch_size = 16 if gpu_mem > 8 else 2
        compute_type = "float16" if torch.cuda.is_bf16_supported() else "int8"
        rprint(f"[cyan]🎮 GPU memory:[/cyan] {gpu_mem:.2f} GB, [cyan]📦 Batch size:[/cyan] {batch_size}, [cyan]⚙️ Compute type:[/cyan] {compute_type}")
    else:
        batch_size = 1
        compute_type = "int8"
        rprint(f"[cyan]📦 Batch size:[/cyan] {batch_size}, [cyan]⚙️ Compute type:[/cyan] {compute_type}")
    rprint(f"[green]▶️ Starting WhisperX for segment {start:.2f}s to {end:.2f}s...[/green]")
    
    if WHISPER_LANGUAGE == 'zh':
        model_name = "Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper"
        local_model = os.path.join(MODEL_DIR, "Belle-whisper-large-v3-zh-punct-fasterwhisper")
    else:
        model_name = load_key("whisper.model")
        local_model = os.path.join(MODEL_DIR, model_name)
        
    if os.path.exists(local_model):
        rprint(f"[green]📥 Loading local WHISPER model:[/green] {local_model} ...")
        model_name = local_model
    else:
        rprint(f"[green]📥 Using WHISPER model from HuggingFace:[/green] {model_name} ...")

    vad_options = {"vad_onset": 0.500,"vad_offset": 0.363}
    asr_options = {"temperatures": [0],"initial_prompt": "",}
    whisper_language = None if 'auto' in WHISPER_LANGUAGE else WHISPER_LANGUAGE
    rprint("[bold yellow] You can ignore warning of `Model was trained with torch 1.10.0+cu102, yours is 2.0.0+cu118...`[/bold yellow]")
    
    # 增强模型加载的重试机制
    model = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            rprint(f"[cyan]🔄 Model loading attempt {attempt + 1}/{max_retries}...[/cyan]")
            model = whisperx.load_model(model_name, device, compute_type=compute_type, language=whisper_language, vad_options=vad_options, asr_options=asr_options, download_root=MODEL_DIR)
            rprint(f"[green]✅ Model loaded successfully![/green]")
            break
        except (Exception) as e:
            rprint(f"[yellow]⚠️ Model loading attempt {attempt + 1} failed: {str(e)}[/yellow]")
            if "ChunkedEncodingError" in str(e) or "IncompleteRead" in str(e) or "Connection broken" in str(e):
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)  # 递增等待时间
                    rprint(f"[cyan]⏳ Network error detected, waiting {wait_time}s before retry...[/cyan]")
                    time.sleep(wait_time)
                    # 清理可能损坏的缓存文件
                    try:
                        import glob
                        incomplete_files = glob.glob(os.path.join(MODEL_DIR, "**", "*.incomplete"), recursive=True)
                        for file in incomplete_files:
                            os.remove(file)
                            rprint(f"[cyan]🗑️ Removed incomplete file: {file}[/cyan]")
                    except:
                        pass
                else:
                    rprint(f"[red]❌ All model loading attempts failed due to network issues![/red]")
                    raise e
            else:
                rprint(f"[red]❌ Model loading failed with non-network error: {str(e)}[/red]")
                raise e
    
    if model is None:
        raise Exception("Failed to load WhisperX model after multiple attempts")

    def load_audio_segment(audio_file, start, end):
        audio, _ = librosa.load(audio_file, sr=16000, offset=start, duration=end - start, mono=True)
        return audio
    raw_audio_segment = load_audio_segment(raw_audio_file, start, end)
    vocal_audio_segment = load_audio_segment(vocal_audio_file, start, end)
    
    # -------------------------
    # 1. transcribe raw audio
    # -------------------------
    transcribe_start_time = time.time()
    rprint("[bold green]Note: You will see Progress if working correctly ↓[/bold green]")
    result = model.transcribe(raw_audio_segment, batch_size=batch_size, print_progress=True)
    transcribe_time = time.time() - transcribe_start_time
    rprint(f"[cyan]⏱️ time transcribe:[/cyan] {transcribe_time:.2f}s")

    # Free GPU resources
    del model
    torch.cuda.empty_cache()

    # Save language
    update_key("whisper.language", result['language'])
    if result['language'] == 'zh' and WHISPER_LANGUAGE != 'zh':
        raise ValueError("Please specify the transcription language as zh and try again!")

    # -------------------------
    # 2. align by vocal audio
    # -------------------------
    align_start_time = time.time()
    # Align timestamps using vocal audio
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, vocal_audio_segment, device, return_char_alignments=False)
    align_time = time.time() - align_start_time
    rprint(f"[cyan]⏱️ time align:[/cyan] {align_time:.2f}s")

    # Free GPU resources again
    torch.cuda.empty_cache()
    del model_a

    # Adjust timestamps
    for segment in result['segments']:
        segment['start'] += start
        segment['end'] += start
        for word in segment['words']:
            if 'start' in word:
                word['start'] += start
            if 'end' in word:
                word['end'] += start
    return result
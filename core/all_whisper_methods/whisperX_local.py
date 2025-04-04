import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import whisperx
import torch
import time
import subprocess
from typing import Dict
from rich import print as rprint
import librosa
from core.config_utils import load_key
from core.all_whisper_methods.audio_preprocess import save_language
import numpy as np

MODEL_DIR = load_key("model_dir")

def check_hf_mirror() -> str:
    """Check and return the fastest HF mirror"""
    mirrors = {
        'Official': 'huggingface.co',
        'Mirror': 'hf-mirror.com'
    }
    fastest_url = f"https://{mirrors['Official']}"
    best_time = float('inf')
    rprint("[cyan]🔍 Checking HuggingFace mirrors...[/cyan]")
    for name, domain in mirrors.items():
        try:
            if os.name == 'nt':
                cmd = ['ping', '-n', '1', '-w', '3000', domain]
            else:
                cmd = ['ping', '-c', '1', '-W', '3', domain]
            start = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            response_time = time.time() - start
            if result.returncode == 0:
                if response_time < best_time:
                    best_time = response_time
                    fastest_url = f"https://{domain}"
                rprint(f"[green]✓ {name}:[/green] {response_time:.2f}s")
        except:
            rprint(f"[red]✗ {name}:[/red] Failed to connect")
    if best_time == float('inf'):
        rprint("[yellow]⚠️ All mirrors failed, using default[/yellow]")
    rprint(f"[cyan]🚀 Selected mirror:[/cyan] {fastest_url} ({best_time:.2f}s)")
    return fastest_url

def load_audio_segment(audio_file: str, start: float, end: float) -> np.ndarray:
    """load audio segment from audio file"""
    duration = end - start
    audio, _ = librosa.load(
        audio_file,
        sr=16000,
        offset=start,
        duration=duration,
        mono=True
    )
    return audio

def transcribe_audio(raw_audio_file: str, vocal_audio_file: str, start: float, end: float) -> Dict:
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
    
    try:
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
        model = whisperx.load_model(model_name, device, compute_type=compute_type, language=whisper_language, vad_options=vad_options, asr_options=asr_options, download_root=MODEL_DIR)

        raw_audio_segment = load_audio_segment(raw_audio_file, start, end)
        vocal_audio_segment = load_audio_segment(vocal_audio_file, start, end)
        
        rprint("[bold green]Note: You will see Progress if working correctly ↓[/bold green]")
        result = model.transcribe(raw_audio_segment, batch_size=batch_size, print_progress=True)

        # Free GPU resources
        del model
        torch.cuda.empty_cache()

        # Save language
        save_language(result['language'])
        if result['language'] == 'zh' and WHISPER_LANGUAGE != 'zh':
            raise ValueError("Please specify the transcription language as zh and try again!")

        # Align timestamps using vocal audio
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, vocal_audio_segment, device, return_char_alignments=False)

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
    except Exception as e:
        rprint(f"[red]WhisperX processing error:[/red] {e}")
        raise
import os
import sys
import replicate
import pandas as pd
import json
from typing import Dict
import subprocess
import base64
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def convert_video_to_audio(input_file: str) -> str:
    os.makedirs('output/audio', exist_ok=True)
    audio_file = 'output/audio/raw_full_audio.wav'
    
    if not os.path.exists(audio_file):
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', input_file,
            '-vn',
            '-acodec', 'libmp3lame',
            '-ar', '16000',
            '-b:a', '64k',
            audio_file
        ]
        print(f"🎬➡️🎵 正在转换为音频......")
        subprocess.run(ffmpeg_cmd, check=True, stderr=subprocess.PIPE)
        print(f"🎬➡️🎵 已将 <{input_file}> 转换为 <{audio_file}>\n")
    
    return audio_file

def encode_file_to_base64(file_path: str) -> str:
    print("🔄 正在将音频文件编码为base64...")
    with open(file_path, 'rb') as file:
        encoded = base64.b64encode(file.read()).decode('utf-8')
        print("✅ 文件已成功编码为base64")
        return encoded

def transcribe_audio(audio_base64: str) -> Dict:
    from config import WHISPER_LANGUAGE
    from config import REPLICATE_API_TOKEN
    # 设置 API 令牌
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
    print(f"🚀 正在启动WhisperX API... 有时需要等待官方启动服务器，请耐心等待... 实际处理速度 2min 音频 10s，一次花费约¥0.1")
    try:
        input_params = {
            "debug": False,
            "vad_onset": 0.5,
            "audio_file": f"data:audio/wav;base64,{audio_base64}",
            "batch_size": 64,
            "vad_offset": 0.363,
            "diarization": False,
            "temperature": 0,
            "align_output": True,
            "language_detection_min_prob": 0,
            "language_detection_max_tries": 5
        }
        
        if WHISPER_LANGUAGE != 'auto':
            input_params["language"] = WHISPER_LANGUAGE
        
        output = replicate.run(
            "victor-upmeet/whisperx:84d2ad2d6194fe98a17d2b60bef1c7f910c46b2f6fd38996ca457afd9c8abfcb",
            input=input_params
        )
        return output
    except Exception as e:
        raise Exception(f"访问whisperX API错误: {e} \n Cuda错误是官方API启动的服务器实例出错导致的，请等候五分钟等待官方切换服务器再重试。")

def process_transcription(result: Dict) -> pd.DataFrame:
    all_words = []
    for segment in result['segments']:
        for i, word in enumerate(segment['words']):
            if 'start' not in word and i > 0:
                # 如果当前单词没有start，将其添加到上一个单词中，通常出现在特殊符号时
                all_words[-1]['text'] = f'{all_words[-1]["text"][:-1]}{word["word"]}"'
            else:
                word_dict = {
                    'text': f'"{word["word"]}"',
                    'start': word.get('start', all_words[-1]['end'] if all_words else 0),
                    'end': word['end'],
                    'score': word.get('score', 0)
                }
                all_words.append(word_dict)
    
    return pd.DataFrame(all_words)

def save_results(df: pd.DataFrame):
    os.makedirs('output/log', exist_ok=True)
    excel_path = os.path.join('output/log', "cleaned_chunks.xlsx")
    df['text'] = df['text'].apply(lambda x: f'"{x}"')
    df.to_excel(excel_path, index=False)
    print(f"📊 Excel文件已保存到 {excel_path}")

def save_language(language: str):
    os.makedirs('output/log', exist_ok=True)
    with open('output/log/transcript_language.json', 'w', encoding='utf-8') as f:
        json.dump({"language": language}, f, ensure_ascii=False, indent=4)
    
def transcribe(video_file: str):
    if not os.path.exists("output/log/cleaned_chunks.xlsx"):
        audio_file = convert_video_to_audio(video_file)
        
        if os.path.getsize(audio_file) > 25 * 1024 * 1024:
            print("⚠️ 文件大小超过25MB。请使用更小的文件。")
            return
        
        audio_base64 = encode_file_to_base64(audio_file)
        result = transcribe_audio(audio_base64)
        
        save_language(result['detected_language'])
        
        df = process_transcription(result)
        save_results(df)
    else:
        print("📊 转录结果已存在，跳过转录步骤。")

if __name__ == "__main__":
    from core.step1_ytdlp import find_video_files
    video_file = find_video_files()
    print(f"🎬 找到的视频文件: {video_file}, 开始转录...")
    transcribe(video_file)

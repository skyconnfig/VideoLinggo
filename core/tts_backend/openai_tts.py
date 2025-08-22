from pathlib import Path
import requests
import json
from core.utils import load_key, except_handler

BASE_URL = "https://api.302.ai/v1/audio/speech"
VOICE_LIST = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
# voice options: alloy, echo, fable, onyx, nova, and shimmer
# refer to: https://platform.openai.com/docs/guides/text-to-speech/quickstart
@except_handler("Failed to generate audio using OpenAI TTS", retry=3, delay=1)
def openai_tts(text, save_path):
    API_KEY = load_key("openai_tts.api_key")
    voice = load_key("openai_tts.voice")
    payload = json.dumps({
        "model": "tts-1",
        "input": text,
        "voice": voice,
        "response_format": "wav"
    })
    
    if voice not in VOICE_LIST:
        raise ValueError(f"Invalid voice: {voice}. Please choose from {VOICE_LIST}")
    headers = {'Authorization': f"Bearer {API_KEY}", 'Content-Type': 'application/json'}
    
    speech_file_path = Path(save_path)
    speech_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.post(BASE_URL, headers=headers, data=payload)
    
    if response.status_code == 200:
        # 检查响应内容类型和大小
        content_type = response.headers.get('content-type', '')
        if 'audio' not in content_type.lower() and len(response.content) < 1000:
            raise Exception(f"Invalid audio response: {response.text[:200]}")
            
        with open(speech_file_path, 'wb') as f:
            f.write(response.content)
            
        # 验证文件大小
        if speech_file_path.stat().st_size < 1000:
            raise Exception(f"Generated audio file too small: {speech_file_path.stat().st_size} bytes")
            
        print(f"Audio saved to {speech_file_path}")
    else:
        raise Exception(f"OpenAI TTS API error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    openai_tts("Hi! Welcome to VideoLingo!", "test.wav")
import requests
from pathlib import Path
from core.utils import *
import json

@except_handler("Failed to generate audio using 302.ai Fish TTS", retry=3, delay=1)
def fish_tts(text: str, save_as: str) -> bool:
    """302.ai Fish TTS conversion"""
    API_KEY = load_key("fish_tts.api_key")
    character = load_key("fish_tts.character")
    refer_id = load_key("fish_tts.character_id_dict")[character]
    
    url = "https://api.302.ai/fish-audio/v1/tts"
    payload = json.dumps({
        "text": text,
        "reference_id": refer_id,
        "chunk_length": 200,
        "normalize": True,
        "format": "wav",
        "latency": "normal"
    })
    
    headers = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
    
    response = requests.post(url, headers=headers, data=payload)
    response.raise_for_status()
    response_data = response.json()
    
    if "url" in response_data:
        audio_response = requests.get(response_data["url"])
        audio_response.raise_for_status()
        
        # 检查音频内容大小
        if len(audio_response.content) < 1000:
            raise Exception(f"Audio content too small: {len(audio_response.content)} bytes")
        
        # 确保目录存在
        Path(save_as).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_as, "wb") as f:
            f.write(audio_response.content)
            
        # 验证文件大小
        if Path(save_as).stat().st_size < 1000:
            raise Exception(f"Generated audio file too small: {Path(save_as).stat().st_size} bytes")
            
        return True
    else:
        raise Exception(f"Fish TTS API error: {response_data}")

if __name__ == '__main__':
    fish_tts("Hi! Welcome to VideoLingo!", "test.wav")

import requests
from pathlib import Path
from core.utils import load_key, except_handler

@except_handler("Failed to generate audio using Azure TTS", retry=3, delay=1)
def azure_tts(text: str, save_path: str) -> None:
    url = "https://api.302.ai/cognitiveservices/v1"
    
    API_KEY = load_key("azure_tts.api_key")
    voice = load_key("azure_tts.voice")
    
    payload = f"""<speak version='1.0' xml:lang='zh-CN'><voice name='{voice}'>{text}</voice></speak>"""
    headers = {
       'Authorization': f'Bearer {API_KEY}',
       'X-Microsoft-OutputFormat': 'riff-16khz-16bit-mono-pcm',
       'Content-Type': 'application/ssml+xml'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    
    # 检查响应状态码
    if response.status_code != 200:
        raise Exception(f"Azure TTS API error: {response.status_code} - {response.text}")
    
    # 检查响应内容类型和大小
    content_type = response.headers.get('content-type', '')
    if 'audio' not in content_type.lower() and len(response.content) < 1000:
        # 如果不是音频类型且内容很小，可能是错误响应
        raise Exception(f"Invalid audio response: {response.text[:200]}")
    
    # 确保目录存在
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 写入音频文件
    with open(save_path, 'wb') as f:
        f.write(response.content)
    
    # 验证文件大小
    if Path(save_path).stat().st_size < 1000:
        raise Exception(f"Generated audio file too small: {Path(save_path).stat().st_size} bytes")
        
    print(f"Audio saved to {save_path}")

if __name__ == "__main__":
    azure_tts("Hi! Welcome to VideoLingo!", "test.wav")
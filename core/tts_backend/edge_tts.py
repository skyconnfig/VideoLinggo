from pathlib import Path
import edge_tts
from core.utils import *
import subprocess

# Available voices can be listed using edge-tts --list-voices command
# Common English voices:
# en-US-JennyNeural - Female
# en-US-GuyNeural - Male  
# en-GB-SoniaNeural - Female British
# Common Chinese voices:
# zh-CN-XiaoxiaoNeural - Female
# zh-CN-YunxiNeural - Male
# zh-CN-XiaoyiNeural - Female
@except_handler("Failed to generate audio using Edge TTS", retry=3, delay=1)
def edge_tts(text, save_path):
    # Load settings from config file
    edge_set = load_key("edge_tts")
    voice = edge_set.get("voice", "en-US-JennyNeural")
    
    # Create output directory if it doesn't exist
    speech_file_path = Path(save_path)
    speech_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        cmd = ["edge-tts", "--voice", voice, "--text", text, "--write-media", str(speech_file_path)]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # 检查文件是否成功生成
        if not speech_file_path.exists():
            raise Exception(f"Edge TTS failed to create audio file: {speech_file_path}")
            
        # 验证文件大小
        if speech_file_path.stat().st_size < 1000:
            raise Exception(f"Generated audio file too small: {speech_file_path.stat().st_size} bytes")
            
        print(f"Audio saved to {speech_file_path}")
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Edge TTS command failed: {e.stderr if e.stderr else str(e)}"
        raise Exception(error_msg)

if __name__ == "__main__":
    edge_tts("Today is a good day!", "edge_tts.wav")

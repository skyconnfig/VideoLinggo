import streamlit as st
import os, sys
from core.st_utils.imports_and_utils import *
from core import *

# SET PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['PATH'] += os.pathsep + current_dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="VideoLingo", page_icon="docs/logo.svg")

SUB_VIDEO = "output/output_sub.mp4"
DUB_VIDEO = "output/output_dub.mp4"

def main():
    # 移除了logo显示，界面更简洁
    st.title("VideoLingo - AI视频翻译和配音工具")
    st.markdown("### 🤖 本地AI版本 - 支持Ollama大模型和本地Whisper")
    
    # 显示本地AI状态
    with st.sidebar:
        st.header("🔧 本地AI服务状态")
        
        # 检查Ollama服务状态
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                st.success("✅ Ollama服务运行中")
            else:
                st.error("❌ Ollama服务异常")
        except:
            st.warning("⚠️ Ollama服务未启动")
            st.info("请运行: ollama serve")
        
        # 显示配置信息
        st.info("📋 当前配置:\n- 模型: Qwen2\n- Whisper: 本地模式\n- 完全离线运行")
    
    # 主界面内容
    sidebar_setting()
    
    # 视频下载部分
    download_video_section()
    
    # 文本处理部分
    if text_processing_section():
        # 音频处理部分
        audio_processing_section()

if __name__ == "__main__":
    main()
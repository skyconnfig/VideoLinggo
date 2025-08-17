<div align="center">

<img src="/docs/logo.png" alt="VideoLingo Logo" height="140">

# Connect the World, Frame by Frame

<a href="https://trendshift.io/repositories/12200" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12200" alt="Huanshere%2FVideoLingo | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

[**English**](/README.md)｜[**简体中文**](/translations/README.zh.md)｜[**繁體中文**](/translations/README.zh-TW.md)｜[**日本語**](/translations/README.ja.md)｜[**Español**](/translations/README.es.md)｜[**Русский**](/translations/README.ru.md)｜[**Français**](/translations/README.fr.md)

</div>

## 🌟 Overview ([Try VL Now!](https://videolingo.io))

VideoLingo is an all-in-one video translation, localization, and dubbing tool aimed at generating Netflix-quality subtitles. It eliminates stiff machine translations and multi-line subtitles while adding high-quality dubbing, enabling global knowledge sharing across language barriers.

Key features:
- 🎥 YouTube video download via yt-dlp

- **🎙️ Word-level and Low-illusion subtitle recognition with WhisperX**

- **📝 NLP and AI-powered subtitle segmentation**

- **📚 Custom + AI-generated terminology for coherent translation**

- **🔄 3-step Translate-Reflect-Adaptation for cinematic quality**

- **✅ Netflix-standard, Single-line subtitles Only**

- **🗣️ Dubbing with GPT-SoVITS, Azure, OpenAI, and more**

- 🚀 One-click startup and processing in Streamlit

- 🌍 Multi-language support in Streamlit UI

- 📝 Detailed logging with progress resumption

Difference from similar projects: **Single-line subtitles only, superior translation quality, seamless dubbing experience**

## 🎥 Demo

<table>
<tr>
<td width="33%">

### Dual Subtitles
---
https://github.com/user-attachments/assets/a5c3d8d1-2b29-4ba9-b0d0-25896829d951

</td>
</tr>
</table>

## 🚀 本地AI部署版本

本项目已配置支持完全本地化的AI服务：

### 🤖 本地大模型服务
- **Ollama + Qwen2**: 本地运行的大语言模型，支持视频翻译和字幕优化
- **完全离线**: 无需API密钥，保护数据隐私
- **高性能**: 支持GPU加速，快速响应

### 🎙️ 本地语音识别
- **WhisperX Local**: 本地部署的语音识别服务
- **多语言支持**: 支持中英文等多种语言识别
- **高精度**: 词级别对齐，低幻觉率

### 📋 配置说明
项目已预配置本地AI服务，详见：
- `本地AI部署指南.md` - 完整的本地部署教程
- `环境配置指南.md` - 环境配置和使用说明
- `config.yaml` - 已优化的本地服务配置

### 🔧 快速开始
1. 安装Ollama并下载Qwen2模型
2. 激活Python虚拟环境
3. 运行 `streamlit run st.py`
4. 享受完全本地化的AI视频翻译体验！

---

*本版本专为隐私保护和离线使用而优化，适合企业和个人用户的本地部署需求。*
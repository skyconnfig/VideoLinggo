<div align="center">

# VideoLingo: Connecting the World, Frame by Frame
<p align="center">
  <a href="https://www.python.org" target="_blank"><img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python"></a>
  <a href="https://github.com/Huanshere/VideoLingo/blob/main/LICENSE" target="_blank"><img src="https://img.shields.io/github/license/Huanshere/VideoLingo.svg" alt="License"></a>
  <a href="https://github.com/Huanshere/VideoLingo/stargazers" target="_blank"><img src="https://img.shields.io/github/stars/Huanshere/VideoLingo.svg" alt="GitHub stars"></a>
</p>

[**中文**](README.md) | [**English**](README.en.md)

**QQ Group: 875297969**

</div>

## 🌟 Project Introduction

VideoLingo is an all-in-one video translation and localization dubbing tool, aimed at generating Netflix-quality subtitles, eliminating stiff machine translations and multi-line subtitles, while also adding high-quality dubbing. It enables knowledge sharing across language barriers worldwide. Through an intuitive Streamlit web interface, you can complete the entire process from video link to embedded high-quality bilingual subtitles and even dubbing with just a few clicks, easily creating Netflix-quality localized videos.

Key features and functionalities:
- 🎥 Uses yt-dlp to download videos from YouTube links

- 🎙️ Uses WhisperX for word-level timeline subtitle recognition

- 📝 Uses NLP and GPT for subtitle segmentation based on sentence meaning

- 📚 GPT summarizes intelligent terminology knowledge base for context-aware translation

- 🔄 Three-step direct translation, reflection, and paraphrasing to eliminate awkward machine translations

- ✅ Netflix-standard single-line subtitle length and translation quality checks

- 🗣️ Uses GPT-SoVITS for high-quality aligned dubbing

- 🚀 One-click integrated package launch, one-click video production in Streamlit

## 🎥 Demo

<table>
<tr>
<td width="33%">

### Russian Translation

https://github.com/user-attachments/assets/25264b5b-6931-4d39-948c-5a1e4ce42fa7

</td>
<td width="33%">

### GPT-SoVITS

https://github.com/user-attachments/assets/47d965b2-b4ab-4a0b-9d08-b49a7bf3508c

</td>
<td width="33%">

### Azure TTS

https://github.com/user-attachments/assets/a5384bd1-0dc8-431a-9aa7-bbe2ea4831b8

</td>
</tr>
</table>

Currently supported input languages and examples:

| Input Language | Support Level | Translation Demo | Dubbing Demo |
|----------------|---------------|-------------------|--------------|
| 🇬🇧🇺🇸 English | 🤩 | [English to Chinese](https://github.com/user-attachments/assets/127373bb-c152-4b7a-8d9d-e586b2c62b4b) | TODO |
| 🇷🇺 Russian | 😊 | [Russian to Chinese](https://github.com/user-attachments/assets/25264b5b-6931-4d39-948c-5a1e4ce42fa7) | TODO |
| 🇫🇷 French | 🤩 | [French to Japanese](https://github.com/user-attachments/assets/3ce068c7-9854-4c72-ae77-f2484c7c6630) | TODO |
| 🇩🇪 German | 🤩 | [German to Chinese](https://github.com/user-attachments/assets/07cb9d21-069e-4725-871d-c4d9701287a3) | TODO |
| 🇮🇹 Italian | 🤩 | [Italian to Chinese](https://github.com/user-attachments/assets/f1f893eb-dad3-4460-aaf6-10cac999195e) | TODO |
| 🇪🇸 Spanish | 🤩 | [Spanish to Chinese](https://github.com/user-attachments/assets/c1d28f1c-83d2-4f13-a1a1-859bd6cc3553) | TODO |
| 🇯🇵 Japanese | 😐 | [Japanese to Chinese](https://github.com/user-attachments/assets/856c3398-2da3-4e25-9c36-27ca2d1f68c2) | TODO |
| 🇨🇳 Chinese | 😖 | ❌ | TODO |

Translation languages support all languages that the large language model can handle, while dubbing languages depend on the chosen TTS method.

## 🚀 One-Click Integrated Package for Windows

### Important Notes:

1. The integrated package uses the CPU version of torch, with a size of about **2.5G**.
2. When using UVR5 for noise reduction in the dubbing step, the CPU version will be significantly slower than GPU-accelerated torch.
3. The integrated package **only supports calling whisperX ☁️ via API**, and does not support running whisperX locally 💻.
4. Due to technical reasons, the integrated package **cannot use edge-tts for dubbing**, but all other functions are complete.

If you need the following features, please install from source code (requires an Nvidia GPU and at least **20G** of disk space):
- Run whisperX locally 💻
- Use GPU-accelerated UVR5 for noise reduction

### Download and Usage Instructions

1. Download the `v1.1.0` one-click package (750M): [CPU Version Download](https://vip.123pan.cn/1817874751/8147218) | [Baidu Backup](https://pan.baidu.com/s/1H_3PthZ3R3NsjS0vrymimg?pwd=ra64)

2. After extracting, double-click `OneKeyStart.bat` in the folder

3. In the opened browser window, configure the necessary settings in the sidebar, then create your video with one click!
  ![settings](https://github.com/user-attachments/assets/3d99cf63-ab89-404c-ae61-5a8a3b27d840)

> 💡 Note: This project requires configuration of large language models, WhisperX, and TTS. Please carefully read the **API Preparation** section below

## 📋 API Preparation
This project requires the use of large language models, WhisperX, and TTS. Multiple options are provided for each component. **Please read the configuration guide carefully 😊**
### 1. **Obtain API_KEY for Large Language Models**:

| Recommended Model | Recommended Provider | base_url | Price | Effect |
|:-----|:---------|:---------|:-----|:---------|
| claude-3-5-sonnet-20240620 | [Yunwu API](https://yunwu.zeabur.app/register?aff=TXMB) | https://yunwu.zeabur.app | ¥15 / 1M | 🤩 |
| Qwen/Qwen2.5-72B-Instruct | [Silicon Flow](https://cloud.siliconflow.cn/i/ttKDEsxE) | https://api.siliconflow.cn | ¥4 / 1M | 😲 |
> Note: Yunwu API also supports OpenAI's tts-1 interface, which can be used in the dubbing step

#### Common Questions

<details>
<summary>How to choose a model?</summary>

- 🚀 Qwen2.5 is used by default, costing about ¥3 for a 1-hour video translation.
- 🌟 Claude 3.5 has better results, with very good translation coherence and no AI flavor, but it's more expensive.
</details>

<details>
<summary>How to get an API key?</summary>

1. Click the link for the recommended provider above
2. Register an account and recharge
3. Create a new API key on the API key page
</details>

<details>
<summary>Can I use other models?</summary>

- ✅ Supports OAI-Like API interfaces, but you need to change it yourself in the Streamlit sidebar.
- ⚠️ However, other models have weak ability to follow instructions and are very likely to report errors during translation, which is strongly discouraged.
</details>

### 2. **Prepare Replicate Token** (Only when using Replicate's whisperX ☁️)

VideoLingo uses WhisperX for speech recognition, supporting both local deployment and cloud API.
#### Comparison of options:
| Option | Disadvantages |
|:-----|:-----|
| **whisperX 🖥️** | • Install CUDA 🛠️<br>• Download model 📥<br>• High VRAM requirement 💾 |
| **whisperX ☁️ (Recommended)** | • Requires VPN 🕵️‍♂️<br>• Visa card 💳 |

#### Obtaining the token
   - Register at [Replicate](https://replicate.com/account/api-tokens), bind a Visa card payment method, and obtain the token
   - **Or join the QQ group to get a free test token from the group announcement**

### 3. **TTS API**
VideoLingo provides multiple TTS integration methods. Here's a comparison (skip this if you're only translating without dubbing):

| TTS Option | Advantages | Disadvantages | Chinese Effect | Non-Chinese Effect |
|:---------|:-----|:-----|:---------|:-----------|
| 🎙️ OpenAI TTS | High quality, realistic emotion | Chinese sounds like a foreigner | 😕 | 🤩 |
| 🎤 Edge TTS | Free | Overused | 😊 | 😊 |
| 🔊 Azure TTS (Recommended) | Natural Chinese effect | Inconvenient recharge | 🤩 | 😃 |
| 🗣️ GPT-SoVITS (beta) | Local, cloning, unbeatable in Chinese | Currently only supports English input Chinese output, requires GPU for model training, best for single-person videos without obvious BGM, and the base model should be close to the original voice | 😱 | 🚫 |

For OpenAI TTS, we recommend using [Yunwu API](https://yunwu.zeabur.app/register?aff=TXMB).
Edge TTS requires no configuration, **Azure TTS free keys can be obtained in the QQ group** or you can register and recharge yourself. Configure these later in the sidebar of the VideoLingo running webpage.

<details>
<summary>GPT-SoVITS Usage Tutorial (Only supports v2 new version)</summary>

1. Go to the [official Yuque document](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e/dkxgpiy9zb96hob4#KTvnO) to check the configuration requirements and download the integrated package.

2. Place `GPT-SoVITS-v2-xxx` in the same directory level as `VideoLingo`. **Note that they should be parallel, not nested.**

3. Choose one of the following methods to configure the model:

   a. Self-trained model:
   - After training the model, `tts_infer.yaml` under `GPT-SoVITS-v2-xxx\GPT_SoVITS\configs` will automatically be filled with your model address. Copy and rename it to `your_preferred_character_name.yaml`
   - In the same directory as the `yaml` file, place the reference audio you'll use later, named `your_preferred_character_name_text_content_of_reference_audio.wav` or `.mp3`, for example `Huanyuv2_Hello, this is a test audio.wav`
   - In the sidebar of the VideoLingo webpage, set `GPT-SoVITS Character` to `your_preferred_character_name`.

   b. Use pre-trained model:
   - Download my model from [here](https://vip.123pan.cn/1817874751/8137723), extract and overwrite to `GPT-SoVITS-v2-xxx`.
   - Set `GPT-SoVITS Character` to `Huanyuv2`.

   c. Use other trained models:
   - Place model files in `GPT_weights_v2` and `SoVITS_weights_v2` respectively.
   - Refer to method a, rename and modify the paths in `tts_infer.yaml` to point to your two models.
   - Refer to method a, place the reference audio you'll use later in the same directory as the `yaml` file, named `your_preferred_character_name_text_content_of_reference_audio.wav` or `.mp3`

   ```
   # Directory structure example
   .
   ├── VideoLingo
   │   └── ...
   └── GPT-SoVITS-v2-xxx
       ├── GPT_SoVITS
       │   └── configs
       │       ├── tts_infer.yaml
       │       ├── your_preferred_character_name.yaml
       │       └── your_preferred_character_name_text_content_of_reference_audio.wav
       ├── GPT_weights_v2
       │   └── [Your GPT model file]
       └── SoVITS_weights_v2
           └── [Your SoVITS model file]
   ```
        
After configuration, VideoLingo will automatically open the inference API port of GPT-SoVITS in the pop-up command line during the dubbing step. You can manually close it after dubbing is complete. Note that this method is still not very stable and may result in missing words or sentences, so please use it with caution.</details>

## 🛠️ Source Code Installation Process

### Windows Prerequisites

Before starting the installation of VideoLingo, please ensure you have **20G** of free disk space and complete the following steps:

| Dependency | whisperX 🖥️ | whisperX ☁️ |
|:-----|:-------------------|:----------------|
| Miniconda 🐍 | [Download](https://docs.conda.io/en/latest/miniconda.html) | [Download](https://docs.conda.io/en/latest/miniconda.html) |
| Git 🌿 | [Download](https://git-scm.com/download/win) | [Download](https://git-scm.com/download/win) |
| Cuda Toolkit 12.6 🚀 | [Download](https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.76_windows.exe) | - |
| Cudnn 9.3.0 🧠 | [Download](https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn_9.3.0_windows.exe) | - |

> Note: When installing Miniconda, check "Add to system Path", and restart your computer after installation 🔄

### Installation Steps

Supports Win, Mac, Linux. If you encounter any issues, you can ask GPT about the entire process~
1. Open Anaconda Powershell Prompt and switch to the desktop directory:
   ```bash
   cd desktop
   ```

2. Clone the project:
   ```bash
   git clone https://github.com/Huanshere/VideoLingo.git
   cd VideoLingo
   ```

3. Configure virtual environment (must be 3.10.0):
   ```bash
   conda create -n videolingo python=3.10.0
   conda activate videolingo
   ```

4. Run the installation script:
   ```bash
   python install.py
   ```
   Follow the prompts to select the desired Whisper project, the script will automatically install the corresponding torch and whisper versions

   Note: Mac users need to manually install ffmpeg according to the prompts

5. 🎉 Enter the command or click `OneKeyStart.bat` to launch the Streamlit application:
   ```bash
   streamlit run st.py
   ```

6. Set the key in the sidebar of the pop-up webpage, and be sure to select the whisper method

   ![settings](https://github.com/user-attachments/assets/3d99cf63-ab89-404c-ae61-5a8a3b27d840)

This project uses structured module development. You can run `core\step__.py` files in sequence. Technical documentation: [Chinese](./docs/README_guide_zh.md) | [English](./docs/README_guide_en.md) (To be updated)

## 📄 License

This project is licensed under the MIT License. When using this project, please follow these rules:

1. Credit VideoLingo for subtitle generation when publishing works.
2. Follow the terms of the large language models and TTS used for proper attribution.

We sincerely thank the following open-source projects for their contributions, which provided important support for the development of VideoLingo:

- [whisperX](https://github.com/m-bain/whisperX)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [json_repair](https://github.com/mangiucugna/json_repair)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

## 📬 Contact Us

- Join our QQ Group: 875297969
- Submit [Issues](https://github.com/Huanshere/VideoLingo/issues) or [Pull Requests](https://github.com/Huanshere/VideoLingo/pulls) on GitHub

---

<p align="center">If you find VideoLingo helpful, please give us a ⭐️!</p>
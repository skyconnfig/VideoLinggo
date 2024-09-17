# 🏠 VideoLingo 安装指南

VideoLingo 语音识别文本步骤提供多种 Whisper 方案的选择，根据个人配置和需求选择其一即可。

| 方案 | 优势 | 劣势 |
|:-----|:-----|:-----|
| **whisper_timestamped** | • 本地运行<br>• 安装简便<br>• 使用原生 Whisper 模型 | • 仅英文效果理想<br>• 需要8G以上显存的显卡 |
| **whisperX**  | • 本地运行<br>• 基于 faster-whisper，性能卓越<br>• 多语言支持好 | • 需安装 CUDA 和 cuDNN<br>• 各语言需单独下载 wav2vec 模型<br>• 需要8G以上显存的显卡 |
| **whisperX_api** (🌟推荐) | • 利用 Replicate 云算力，无需本地算力 | • 需绑定 Visa 卡支付（一次转录约¥0.1） |

## 📋 API 准备

1. 获取大模型的 API_KEY：

| 模型 | 推荐渠道 | 价格 | 效果 |
|:-----|:---------|:-----|:---------|
| claude-3-5-sonnet | [Deepbricks](https://deepbricks.ai/api-key) | ￥50 / 1M (官方的1/2) | 🤩 |
| TA/Qwen/Qwen1.5-72B-Chat | [OHMYGPT](https://www.ohmygpt.com?aff=u20olROA) | ￥3 / 1M | 😲 |
| deepseek-coder | [OHMYGPT](https://www.ohmygpt.com?aff=u20olROA) | ￥2 / 1M | 😲 |

   注：默认使用 3.5sonnet，10分钟视频翻译约花费￥3。兼容任何 OpenAI-Like 模型，但只建议这三种，其余容易出错。

2. 若选用 `whisperX_api`，需准备 Replicate 的 Token：
   - 在 [Replicate](https://replicate.com/account/api-tokens) 注册并绑定 Visa 卡支付方式，获取令牌
   - 或加入 QQ 群联系作者免费获取测试令牌

## 💾 一键整合包教程

1. 下载 `v0.8.0` 一键整合包(700M): [直达链接](https://vip.123pan.cn/1817874751/8050534) | [度盘备用](https://pan.baidu.com/s/1H_3PthZ3R3NsjS0vrymimg?pwd=ra64)

2. 解压后双击运行文件夹中的 `一键启动.bat`

3. 在打开的浏览器窗口中，在侧边栏进行必要配置，然后一键出片！

> 提示: 侧边栏配置 key 的说明可以参考最下方图片

## 🛠️ 源码安装流程

### Windows 前置依赖

在开始安装本地 Whisper 版的 VideoLingo 之前，注意预留至少 **20G** 硬盘空间，并请确保完成以下步骤：

| 依赖 | whisperX | whisper_timestamped | whisperX_api |
|:-----|:--------------|:-------------------------|:-------------------|
| [Anaconda](https://www.anaconda.com/download/success)<br>*勾选"添加到环境变量"* | ✅ | ✅ | ✅ |
| [Git](https://git-scm.com/download/win) | ✅ | ✅ | ✅ |
| [Cuda Toolkit 12.6](https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.76_windows.exe) | ✅ | | |
| [Cudnn 9.3.0](https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn_9.3.0_windows.exe) | ✅ | | |
| [Visual Studio 2022](https://visualstudio.microsoft.com/zh-hans/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSLandingPage&cid=2030&passive=false)<br>*勾选"使用 C++ 的桌面开发"* | | ✅ | |
| [CMake](https://github.com/Kitware/CMake/releases/download/v3.30.2/cmake-3.30.2-windows-x86_64.msi) | | ✅ | |

> 注意：
> - 安装后需要重启计算机
> - 如果在运行 whisperX 时报错 CUDA Memory，请自行在 `core/all_whisper_methods/whisperX.py` 中调小 batch size 重试

### 安装步骤
支持Win, Mac, Linux。遇到问题可以把整个步骤丢给 GPT 问问~
1. 打开 Anaconda Powershell Prompt 并切换到桌面目录：
   ```bash
   cd desktop
   ```

2. 克隆项目：
   ```bash
   git clone https://github.com/Huanshere/VideoLingo.git
   cd VideoLingo
   ```

3. 配置虚拟环境：
   ```bash
   conda create -n videolingo python=3.10.0
   conda activate videolingo
   ```

4. 运行安装脚本：
   ```bash
   python install.py
   ```
   根据提示选择所需的 Whisper 项目，脚本将自动安装相应的 torch 和 whisper 版本

   注意：Mac 用户需根据提示手动安装 ffmpeg

5. 🎉 启动 Streamlit 应用：
   ```bash
   streamlit run st.py
   ```

6. 在弹出网页的侧边栏中设置key，并注意选择whisper方法

   ![settings](https://github.com/user-attachments/assets/3d99cf63-ab89-404c-ae61-5a8a3b27d840)
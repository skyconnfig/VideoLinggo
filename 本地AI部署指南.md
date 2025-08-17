# VideoLingo 本地AI部署指南

## 概述
本指南将帮助您在VideoLingo项目中部署本地AI服务，包括Ollama大模型和本地Whisper语音识别，实现完全离线的视频翻译和配音功能。

## 🚀 部署架构

```
VideoLingo 项目
├── 本地Ollama服务 (端口: 11434)
│   └── Qwen2模型 (用于翻译和总结)
├── 本地Whisper服务
│   └── WhisperX本地模式 (语音识别)
└── Streamlit界面 (端口: 8501)
```

## 📋 系统要求

### 硬件要求
- **内存**: 建议16GB以上
- **存储**: 至少20GB可用空间（用于模型存储）
- **GPU**: 可选，有NVIDIA GPU可显著提升性能
- **CPU**: 支持AVX指令集的现代处理器

### 软件要求
- Windows 10/11
- Python 3.9+
- Git

## 🛠️ 安装步骤

### 1. 安装Ollama

#### 方法一：官方安装包（推荐）
1. 访问 [Ollama官网](https://ollama.com/)
2. 下载Windows安装包 `OllamaSetup.exe`
3. 双击安装包，按向导完成安装
4. 安装完成后，打开命令行验证：
   ```bash
   ollama --version
   ```

#### 方法二：使用Chocolatey
```bash
# 如果已安装Chocolatey
choco install ollama
```

### 2. 配置Ollama模型存储路径

为避免C盘空间不足，建议修改模型存储路径：

```bash
# 创建模型存储目录
mkdir D:\ollama_models

# 设置环境变量
setx OLLAMA_MODELS "D:\ollama_models"

# 允许外部访问（可选）
setx OLLAMA_HOST "0.0.0.0:11434"
```

**注意**: 设置环境变量后需要重启命令行窗口。

### 3. 下载和配置Qwen2模型

```bash
# 下载Qwen2模型（约4GB）
ollama pull qwen2

# 验证模型安装
ollama list

# 测试模型运行
ollama run qwen2
```

### 4. 启动Ollama服务

```bash
# 启动Ollama服务
ollama serve
```

服务启动后，可通过 `http://localhost:11434` 访问API。

### 5. 配置VideoLingo项目

项目的 `config.yaml` 已经配置为使用本地服务：

```yaml
# API settings - 本地Ollama配置
api:
  key: 'ollama'  # 本地服务占位符
  base_url: 'http://localhost:11434/v1'  # Ollama本地API地址
  model: 'qwen2'  # 使用qwen2模型
  llm_support_json: false

# 本地LLM优化设置
max_workers: 1  # 单线程访问
summary_length: 2000  # 较小的摘要长度

# Whisper本地配置
whisper:
  runtime: 'local'  # 使用本地模式
  model: 'large-v3'
  language: 'en'
```

## 🎯 使用说明

### 启动服务

1. **启动Ollama服务**:
   ```bash
   ollama serve
   ```

2. **启动VideoLingo**:
   ```bash
   # 激活虚拟环境
   venv\Scripts\Activate.ps1
   
   # 启动Streamlit界面
   streamlit run st.py
   ```

3. **访问界面**: 打开浏览器访问 `http://localhost:8501`

### 功能验证

#### 测试Ollama连接
```bash
# 测试API连接
curl http://localhost:11434/v1/models

# 测试对话功能
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

#### 测试Whisper功能
在VideoLingo界面中上传一个音频文件，验证语音识别功能是否正常。

## 🔧 性能优化

### GPU加速

如果有NVIDIA GPU，可以启用GPU加速：

1. **安装CUDA工具包** (如果尚未安装)
2. **验证GPU支持**:
   ```bash
   ollama run qwen2 --verbose
   ```

### 内存优化

- **调整模型大小**: 如果内存不足，可以使用更小的模型：
  ```bash
  ollama pull qwen2:0.5b  # 更小的模型
  ```

- **调整批处理大小**: 在 `whisperX_local.py` 中会自动根据GPU内存调整批处理大小。

## 🚨 故障排除

### 常见问题

#### 1. Ollama服务无法启动
- 检查端口11434是否被占用
- 确认环境变量设置正确
- 重启计算机后重试

#### 2. 模型下载失败
- 检查网络连接
- 尝试使用代理或镜像源
- 手动下载模型文件

#### 3. Whisper识别错误
- 确认音频格式支持
- 检查音频质量和清晰度
- 尝试不同的Whisper模型

#### 4. 翻译质量不佳
- 调整prompt设置
- 尝试不同的模型参数
- 检查源语言设置

### 日志查看

- **Ollama日志**: 查看命令行输出
- **VideoLingo日志**: 查看Streamlit界面的错误信息
- **系统日志**: 检查Windows事件查看器

## 📊 性能监控

### 资源使用监控

```bash
# 查看Ollama进程
tasklist | findstr ollama

# 监控GPU使用（如果有GPU）
nvidia-smi
```

### 性能基准

- **翻译速度**: 约100-200字符/秒（取决于硬件）
- **语音识别**: 约1-3倍实时速度
- **内存使用**: 4-8GB（取决于模型大小）

## 🔄 更新和维护

### 更新Ollama
```bash
# 检查更新
ollama --version

# 重新安装最新版本
# 下载新的安装包并重新安装
```

### 更新模型
```bash
# 更新模型到最新版本
ollama pull qwen2:latest

# 清理旧模型
ollama rm qwen2:old-version
```

### 清理缓存
```bash
# 清理模型缓存
ollama prune

# 清理Whisper缓存
# 删除 _model_cache 目录中的旧文件
```

## 🎉 完成

恭喜！您已经成功部署了VideoLingo的本地AI服务。现在可以享受完全离线的视频翻译和配音功能，无需依赖外部API服务，保护您的数据隐私。

## 📞 技术支持

如果遇到问题，请：
1. 查看本指南的故障排除部分
2. 检查项目的GitHub Issues
3. 参考Ollama官方文档
4. 联系项目维护者

---

**注意**: 本地部署需要较好的硬件配置，建议在正式使用前进行充分测试。
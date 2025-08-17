#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VideoLingo 本地AI服务测试脚本
用于验证Ollama和Whisper本地服务是否正常工作
"""

import requests
import json
import os
import sys
from pathlib import Path

def test_ollama_connection():
    """测试Ollama服务连接"""
    print("🔍 测试Ollama服务连接...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama服务正常运行")
            print(f"📦 已安装模型数量: {len(models)}")
            
            if models:
                print("📋 可用模型列表:")
                for model in models:
                    print(f"   - {model.get('name', 'Unknown')}")
                return True, models
            else:
                print("⚠️  没有检测到已安装的模型")
                print("💡 请运行: ollama pull qwen2")
                return True, []
        else:
            print(f"❌ Ollama服务响应异常: {response.status_code}")
            return False, []
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到Ollama服务")
        print("💡 请确保Ollama服务正在运行: ollama serve")
        return False, []
    except Exception as e:
        print(f"❌ 连接测试失败: {str(e)}")
        return False, []

def test_ollama_chat(model_name="qwen2"):
    """测试Ollama对话功能"""
    print(f"\n💬 测试Ollama对话功能 (模型: {model_name})...")
    try:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": "你好，请用中文回答：1+1等于多少？"}
            ],
            "stream": False
        }
        
        response = requests.post(
            "http://localhost:11434/v1/chat/completions",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            message = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            print(f"✅ 对话测试成功")
            print(f"🤖 模型回答: {message[:100]}..." if len(message) > 100 else f"🤖 模型回答: {message}")
            return True
        else:
            print(f"❌ 对话测试失败: {response.status_code}")
            print(f"📄 响应内容: {response.text[:200]}...")
            return False
    except Exception as e:
        print(f"❌ 对话测试异常: {str(e)}")
        return False

def test_config_file():
    """测试配置文件设置"""
    print("\n📋 检查配置文件...")
    config_path = Path("config.yaml")
    
    if not config_path.exists():
        print("❌ 找不到config.yaml文件")
        return False
    
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        api_config = config.get('api', {})
        whisper_config = config.get('whisper', {})
        
        print("✅ 配置文件读取成功")
        print(f"🔗 API地址: {api_config.get('base_url')}")
        print(f"🤖 API模型: {api_config.get('model')}")
        print(f"🎤 Whisper模式: {whisper_config.get('runtime')}")
        print(f"⚙️  最大工作线程: {config.get('max_workers')}")
        print(f"📝 摘要长度: {config.get('summary_length')}")
        
        # 检查关键配置
        checks = [
            (api_config.get('base_url') == 'http://localhost:11434/v1', "API地址配置"),
            (api_config.get('model') == 'qwen2', "模型配置"),
            (whisper_config.get('runtime') == 'local', "Whisper本地模式"),
            (config.get('max_workers') == 1, "工作线程数配置"),
            (config.get('summary_length') == 2000, "摘要长度配置")
        ]
        
        all_good = True
        for check, desc in checks:
            if check:
                print(f"✅ {desc}: 正确")
            else:
                print(f"⚠️  {desc}: 可能需要调整")
                all_good = False
        
        return all_good
        
    except ImportError:
        print("❌ 缺少PyYAML库，无法解析配置文件")
        return False
    except Exception as e:
        print(f"❌ 配置文件检查失败: {str(e)}")
        return False

def test_whisper_local():
    """测试Whisper本地配置"""
    print("\n🎤 检查Whisper本地配置...")
    
    whisper_file = Path("core/asr_backend/whisperX_local.py")
    if whisper_file.exists():
        print("✅ WhisperX本地文件存在")
        
        # 检查必要的依赖
        try:
            import whisperx
            print("✅ WhisperX库已安装")
        except ImportError:
            print("❌ WhisperX库未安装")
            return False
        
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"✅ PyTorch已安装，设备: {device}")
            if device == "cuda":
                gpu_count = torch.cuda.device_count()
                print(f"🎮 检测到 {gpu_count} 个GPU")
        except ImportError:
            print("❌ PyTorch库未安装")
            return False
        
        return True
    else:
        print("❌ WhisperX本地文件不存在")
        return False

def main():
    """主测试函数"""
    print("🚀 VideoLingo 本地AI服务测试")
    print("=" * 50)
    
    # 测试结果统计
    results = {
        'ollama_connection': False,
        'ollama_chat': False,
        'config_file': False,
        'whisper_local': False
    }
    
    # 1. 测试Ollama连接
    ollama_ok, models = test_ollama_connection()
    results['ollama_connection'] = ollama_ok
    
    # 2. 如果有模型，测试对话功能
    if ollama_ok and models:
        model_name = models[0].get('name', 'qwen2')
        results['ollama_chat'] = test_ollama_chat(model_name)
    elif ollama_ok:
        print("\n⏳ 模型还在下载中，跳过对话测试")
    
    # 3. 测试配置文件
    results['config_file'] = test_config_file()
    
    # 4. 测试Whisper配置
    results['whisper_local'] = test_whisper_local()
    
    # 输出总结
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 总体状态: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 恭喜！本地AI服务配置完成，可以开始使用VideoLingo了！")
    elif passed >= total - 1:
        print("⚠️  基本配置完成，可能需要等待模型下载完成")
    else:
        print("❌ 配置存在问题，请检查上述失败项目")
    
    print("\n💡 使用提示:")
    print("   - 启动服务: streamlit run st.py")
    print("   - 访问界面: http://localhost:8501")
    print("   - 查看指南: 本地AI部署指南.md")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
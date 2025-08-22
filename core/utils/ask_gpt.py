import os
import json
import time
import requests
import functools
from threading import Lock
import json_repair
from openai import OpenAI
from core.utils.config_utils import load_key
from rich import print as rprint
from core.utils.decorator import except_handler

# ------------
# cache gpt response
# ------------

LOCK = Lock()
GPT_LOG_FOLDER = 'output/gpt_log'

def _save_cache(model, prompt, resp_content, resp_type, resp, message=None, log_title="default"):
    with LOCK:
        logs = []
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        logs.append({"model": model, "prompt": prompt, "resp_content": resp_content, "resp_type": resp_type, "resp": resp, "message": message})
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)

def _load_cache(prompt, resp_type, log_title):
    with LOCK:
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    if item["prompt"] == prompt and item["resp_type"] == resp_type:
                        return item["resp"]
        return False

# ------------
# ask gpt once
# ------------

def _check_ollama_service(base_url):
    """检查Ollama服务是否可用"""
    try:
        # 移除/v1后缀进行健康检查
        health_url = base_url.replace('/v1', '').rstrip('/')
        response = requests.get(f"{health_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception as e:
        rprint(f"[yellow]⚠️ Ollama service check failed: {e}[/yellow]")
        return False

def _enhanced_error_handler(error_msg, retry=5, delay=1):
    """增强的错误处理装饰器，专门针对502错误优化"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            base_url = load_key("api.base_url")
            
            for i in range(retry + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_str = str(e)
                    
                    # 详细的错误分类和日志
                    if "502" in error_str or "Bad Gateway" in error_str:
                        rprint(f"[red]🔄 502 Bad Gateway error (attempt {i+1}/{retry+1}): Ollama服务暂时不可用[/red]")
                        # 502错误时检查服务状态
                        if not _check_ollama_service(base_url):
                            rprint(f"[yellow]⚠️ Ollama服务连接失败，等待服务恢复...[/yellow]")
                    elif "timeout" in error_str.lower():
                        rprint(f"[red]⏱️ Timeout error (attempt {i+1}/{retry+1}): 请求超时[/red]")
                    elif "connection" in error_str.lower():
                        rprint(f"[red]🔌 Connection error (attempt {i+1}/{retry+1}): 连接失败[/red]")
                    else:
                        rprint(f"[red]{error_msg}: {e}, retry: {i+1}/{retry+1}[/red]")
                    
                    if i == retry:
                        rprint(f"[red]❌ 最终失败: {error_msg} - {e}[/red]")
                        raise last_exception
                    
                    # 指数退避，但对502错误使用更长的等待时间
                    if "502" in error_str:
                        wait_time = delay * (3 ** i)  # 502错误使用3的指数
                    else:
                        wait_time = delay * (2 ** i)  # 其他错误使用2的指数
                    
                    rprint(f"[cyan]⏳ 等待 {wait_time} 秒后重试...[/cyan]")
                    time.sleep(wait_time)
        
        return wrapper
    return decorator

def _direct_api_request(base_url, model, messages, response_format=None, timeout=60):
    """直接使用requests调用Ollama API，绕过OpenAI客户端库兼容性问题"""
    api_url = f"{base_url.rstrip('/')}/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {load_key('api.key')}"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    
    if response_format:
        payload["response_format"] = response_format
    
    rprint(f"[cyan]🔧 使用直接API请求: {api_url}[/cyan]")
    
    response = requests.post(
        api_url,
        headers=headers,
        json=payload,
        timeout=timeout
    )
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")
    
    return response.json()

@_enhanced_error_handler("GPT request failed", retry=5, delay=2)
def ask_gpt(prompt, resp_type=None, valid_def=None, log_title="default"):
    if not load_key("api.key"):
        raise ValueError("API key is not set")
    # check cache
    cached = _load_cache(prompt, resp_type, log_title)
    if cached:
        rprint("use cache response")
        return cached

    model = load_key("api.model")
    base_url = load_key("api.base_url")
    if 'ark' in base_url:
        base_url = "https://ark.cn-beijing.volces.com/api/v3" # huoshan base url
    elif 'v1' not in base_url:
        base_url = base_url.strip('/') + '/v1'
    
    # 在请求前检查服务状态
    if "localhost" in base_url and not _check_ollama_service(base_url):
        rprint(f"[yellow]⚠️ Ollama服务似乎未响应，但继续尝试连接...[/yellow]")
    
    response_format = {"type": "json_object"} if resp_type == "json" and load_key("api.llm_support_json") else None
    messages = [{"role": "user", "content": prompt}]
    
    # 优化超时设置：本地服务使用较短超时，云服务使用较长超时
    timeout = 60 if "localhost" in base_url else 120
    
    rprint(f"[cyan]🤖 正在请求 {model} 模型...[/cyan]")
    
    # 对于本地Ollama服务，使用直接API请求避免OpenAI客户端库兼容性问题
    if "localhost" in base_url:
        try:
            resp_raw = _direct_api_request(base_url, model, messages, response_format, timeout)
            resp_content = resp_raw['choices'][0]['message']['content']
        except Exception as e:
            rprint(f"[yellow]⚠️ 直接API请求失败，尝试使用OpenAI客户端库: {e}[/yellow]")
            # 回退到OpenAI客户端库
            client = OpenAI(api_key=load_key("api.key"), base_url=base_url)
            params = dict(
                model=model,
                messages=messages,
                response_format=response_format,
                timeout=timeout
            )
            resp_raw = client.chat.completions.create(**params)
            resp_content = resp_raw.choices[0].message.content
    else:
        # 对于云服务，继续使用OpenAI客户端库
        client = OpenAI(api_key=load_key("api.key"), base_url=base_url)
        params = dict(
            model=model,
            messages=messages,
            response_format=response_format,
            timeout=timeout
        )
        resp_raw = client.chat.completions.create(**params)
        resp_content = resp_raw.choices[0].message.content

    # process and return full result
    if resp_type == "json":
        resp = json_repair.loads(resp_content)
    else:
        resp = resp_content
    
    # check if the response format is valid
    if valid_def:
        valid_resp = valid_def(resp)
        if valid_resp['status'] != 'success':
            _save_cache(model, prompt, resp_content, resp_type, resp, log_title="error", message=valid_resp['message'])
            raise ValueError(f"❎ API response error: {valid_resp['message']}")

    _save_cache(model, prompt, resp_content, resp_type, resp, log_title=log_title)
    rprint(f"[green]✅ GPT请求成功完成[/green]")
    return resp


if __name__ == '__main__':
    from rich import print as rprint
    
    result = ask_gpt("""test respond ```json\n{\"code\": 200, \"message\": \"success\"}\n```""", resp_type="json")
    rprint(f"Test json output result: {result}")

import json
import logging
import os
import time
from typing import Dict, Any, Generator
import requests
from flask import Flask, request, jsonify, Response, stream_with_context

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_proxy.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('LLMProxy')

class OpenAIClient:
    """OpenAI客户端封装"""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })

    def make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """向OpenAI API发送请求"""
        url = f"{self.base_url}/{endpoint}"

        try:
            # 记录发送的请求数据
            logger.info(f"发送请求到: {url}")
            logger.info(f"请求数据: {json.dumps(data, ensure_ascii=False)}")
            
            response = self.session.post(url, json=data, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI API请求失败: {e}")
            logger.error(f"请求URL: {url}")
            logger.error(f"请求数据: {json.dumps(data, ensure_ascii=False)}")
            raise

    def stream_request(self, endpoint: str, data: Dict[str, Any]) -> Generator[bytes, None, None]:
        """向OpenAI API发送流式请求"""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            # 记录发送的请求数据
            logger.info(f"发送流式请求到: {url}")
            logger.info(f"请求数据: {json.dumps(data, ensure_ascii=False)}")
            
            response = self.session.post(url, json=data, timeout=60, stream=True)
            response.raise_for_status()
            
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    yield chunk
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI API流式请求失败: {e}")
            logger.error(f"请求URL: {url}")
            logger.error(f"请求数据: {json.dumps(data, ensure_ascii=False)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n".encode('utf-8')

class LLMProxy:
    """LLM代理服务"""

    def __init__(self, openai_api_key: str, openai_base_url: str = None):
        openai_base_url = openai_base_url or "https://api.openai.com/v1"
        self.openai_client = OpenAIClient(openai_api_key, openai_base_url)

    def process_chat_completion(self, request_data: Dict[str, Any]) -> Dict[str, Any] | Generator[bytes, None, None]:
        """处理聊天补全请求"""
        # 打印输入参数
        self._log_request(request_data)

        # 记录开始时间
        start_time = time.time()

        try:
            # 检查是否需要流式输出
            if request_data.get("stream", False):
                # 流式输出
                def generate() -> Generator[bytes, None, None]:
                    full_response = ""
                    try:
                        for chunk in self.openai_client.stream_request("chat/completions", request_data):
                            # 累积完整响应用于日志记录
                            if chunk.startswith(b"data: "):
                                try:
                                    data_str = chunk.decode('utf-8')[6:]  # 移除 "data: " 前缀
                                    if data_str.strip() != "[DONE]":
                                        data = json.loads(data_str)
                                        if "choices" in data and len(data["choices"]) > 0:
                                            delta = data["choices"][0].get("delta", {})
                                            if "content" in delta:
                                                content = delta["content"]
                                                if content is not None:
                                                    full_response += content
                                                    
                                            # 检查是否有工具调用
                                            if "tool_calls" in delta:
                                                self._log_tool_calls(delta["tool_calls"])
                                       
                                except (json.JSONDecodeError, KeyError):
                                    pass
                            yield chunk
                        
                        # 计算耗时
                        duration = time.time() - start_time
                        # 打印完整响应
                        self._log_stream_response(full_response, duration)
                    except Exception as e:
                        logger.error(f"流式传输过程中发生错误: {e}")
                        yield f"data: {json.dumps({'error': str(e)})}\n\n".encode('utf-8')
                
                return generate()
            else:
                # 非流式输出
                response = self.openai_client.make_request("chat/completions", request_data)
                # 计算耗时
                duration = time.time() - start_time
                # 打印响应
                self._log_response(response, duration)
                
                # 检查是否有工具调用
                if "choices" in response and len(response["choices"]) > 0:
                    choice = response["choices"][0]
                    if "message" in choice and "tool_calls" in choice["message"]:
                        self._log_tool_calls(choice["message"]["tool_calls"])
                
                return response

        except Exception as e:
            logger.error(f"处理请求时发生错误: {e}")
            return {
                "error": str(e),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request_data.get("model", "unknown")
            }

    def process_completion(self, request_data: Dict[str, Any]) -> Dict[str, Any] | Generator[bytes, None, None]:
        """处理文本补全请求"""
        self._log_request(request_data)

        start_time = time.time()

        try:
            if request_data.get("stream", False):
                # 流式输出
                def generate() -> Generator[bytes, None, None]:
                    full_response = ""
                    try:
                        for chunk in self.openai_client.stream_request("completions", request_data):
                            # 累积完整响应用于日志记录
                            if chunk.startswith(b"data: "):
                                try:
                                    data_str = chunk.decode('utf-8')[6:]  # 移除 "data: " 前缀
                                    if data_str.strip() != "[DONE]":
                                        data = json.loads(data_str)
                                        if "choices" in data and len(data["choices"]) > 0:
                                            text = data["choices"][0].get("text", "")
                                            if text is not None:
                                                full_response += text
                                except (json.JSONDecodeError, KeyError):
                                    pass
                            yield chunk
                        
                        # 计算耗时
                        duration = time.time() - start_time
                        # 打印完整响应
                        self._log_stream_response(full_response, duration)
                    except Exception as e:
                        logger.error(f"流式传输过程中发生错误: {e}")
                        yield f"data: {json.dumps({'error': str(e)})}\n\n".encode('utf-8')
                
                return generate()
            else:
                # 非流式输出
                response = self.openai_client.make_request("completions", request_data)
                duration = time.time() - start_time
                self._log_response(response, duration)
                return response

        except Exception as e:
            logger.error(f"处理请求时发生错误: {e}")
            return {
                "error": str(e),
                "object": "text_completion",
                "created": int(time.time()),
                "model": request_data.get("model", "unknown")
            }

    def _log_request(self, request_data: Dict[str, Any]):
        """记录请求日志"""
        logger.info("=" * 80)
        logger.info("📥 收到LLM请求")
        logger.info("=" * 80)

        # 基本信息
        model = request_data.get("model", "unknown")
        logger.info(f"🤖 模型: {model}")

        # 消息内容（聊天补全）
        if "messages" in request_data:
            messages = request_data.get("messages", [])
            logger.info("💬 消息内容:")
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                logger.info(f"  {i + 1}. [{role.upper()}] {content}")

        # 提示词（文本补全）
        if "prompt" in request_data:
            prompt = request_data.get("prompt", "")
            if isinstance(prompt, str):
                logger.info(f"📝 提示词: {prompt}")
            else:
                logger.info(f"📝 提示词: {json.dumps(prompt, ensure_ascii=False)}")

        # 参数配置
        logger.info("⚙️ 参数配置:")
        params = [
            ("temperature", request_data.get("temperature")),
            ("max_tokens", request_data.get("max_tokens")),
            ("top_p", request_data.get("top_p")),
            ("frequency_penalty", request_data.get("frequency_penalty")),
            ("presence_penalty", request_data.get("presence_penalty")),
            ("stream", request_data.get("stream", False))
        ]

        for param_name, param_value in params:
            if param_value is not None:
                logger.info(f"  {param_name}: {param_value}")

    def _log_response(self, response: Dict[str, Any], duration: float):
        """记录响应日志"""
        logger.info("=" * 80)
        logger.info("📤 返回LLM响应")
        logger.info("=" * 80)

        logger.info(f"⏱️ 响应时间: {duration:.2f}秒")

        if "error" in response:
            logger.error(f"❌ 错误: {response['error']}")
            return

        # 聊天补全响应
        if "choices" in response and response["choices"]:
            choice = response["choices"][0]

            if "message" in choice:
                # 聊天格式
                message = choice.get("message", {})
                role = message.get("role", "assistant")
                content = message.get("content", "")
                logger.info(f"💭 助手回复 [{role}]: {content}")

                # 如果有函数调用
                if "function_call" in message:
                    func_call = message["function_call"]
                    logger.info(f"🔧 函数调用: {func_call.get('name', 'unknown')}")
                    logger.info(f"📋 函数参数: {func_call.get('arguments', '')}")

            elif "text" in choice:
                # 文本补全格式
                text = choice.get("text", "")
                logger.info(f"📄 生成文本: {text}")

        # 使用情况统计
        if "usage" in response:
            usage = response["usage"]
            logger.info("📊 使用统计:")
            logger.info(f"  提示token: {usage.get('prompt_tokens', 0)}")
            logger.info(f"  完成token: {usage.get('completion_tokens', 0)}")
            logger.info(f"  总token: {usage.get('total_tokens', 0)}")

    def _log_stream_response(self, full_response: str, duration: float):
        """记录流式响应日志"""
        logger.info("=" * 80)
        logger.info("📤 返回LLM流式响应")
        logger.info("=" * 80)

        logger.info(f"⏱️ 响应时间: {duration:.2f}秒")
        logger.info(f"💭 助手完整回复: {full_response}")
        
    def _log_tool_calls(self, tool_calls):
        """记录工具调用日志"""
        logger.info("🔧 检测到工具调用:")
        for i, tool_call in enumerate(tool_calls):
            if "function" in tool_call:
                function = tool_call["function"]
                logger.info(f"  {i+1}. 工具名称: {function.get('name', 'unknown')}")
                logger.info(f"     工具参数: {function.get('arguments', '')}")
                if "id" in tool_call:
                    logger.info(f"     调用ID: {tool_call['id']}")


# 创建Flask应用
app = Flask(__name__)

# 初始化代理（从环境变量获取API密钥）
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.warning("未设置OPENAI_API_KEY环境变量，使用虚拟密钥")
    openai_api_key = "sk-dummy-key"

openai_base_url = os.getenv("OPENAI_BASE_URL")  # 可选的自定义基础URL

llm_proxy = LLMProxy(openai_api_key, openai_base_url)


@app.route('/v1/chat/completions', methods=['POST'])
@app.route('/chat/completions', methods=['POST'])
def chat_completions():
    """处理聊天补全端点"""
    try:
        request_data = request.get_json()
        response_data = llm_proxy.process_chat_completion(request_data)
        
        # 检查是否是流式响应
        if isinstance(response_data, Generator):
            return Response(stream_with_context(response_data), content_type='text/event-stream')
        else:
            return jsonify(response_data)
    except Exception as e:
        logger.error(f"聊天补全处理失败: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/v1/completions', methods=['POST'])
def completions():
    """处理文本补全端点"""
    try:
        request_data = request.get_json()
        response_data = llm_proxy.process_completion(request_data)
        
        # 检查是否是流式响应
        if isinstance(response_data, Generator):
            return Response(stream_with_context(response_data), content_type='text/event-stream')
        else:
            return jsonify(response_data)
    except Exception as e:
        logger.error(f"文本补全处理失败: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/v1/models', methods=['GET'])
def models():
    """返回模型列表（模拟OpenAI格式）"""
    # 这里可以返回模拟的模型列表，或者代理到真实API
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": 1687882410,
                "owned_by": "openai"
            },
            {
                "id": "gpt-4",
                "object": "model",
                "created": 1687882411,
                "owned_by": "openai"
            }
        ]
    })


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({"status": "healthy", "service": "LLM Proxy"})


if __name__ == '__main__':
    # 启动服务
    host = os.getenv('PROXY_HOST', '0.0.0.0')
    port = int(os.getenv('PROXY_PORT', 8000))

    logger.info(f"🚀 启动LLM代理服务在 {host}:{port}")
    logger.info(f"🔑 使用的OpenAI密钥: {openai_api_key[:10]}...")

    app.run(host=host, port=port, debug=False)
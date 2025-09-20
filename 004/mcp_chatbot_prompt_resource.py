import traceback

from dotenv import load_dotenv
import openai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
import json
import asyncio
import os

load_dotenv()


BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"
api_key = os.getenv("DASHSCOPE_API_KEY")
class MCP_ChatBot:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=BASE_URL
        )
        # Tools list required for OpenAI API
        self.available_tools = []
        # Prompts list for quick display
        self.available_prompts = []
        # Sessions dict maps tool/prompt names or resource URIs to MCP client sessions
        self.sessions = {}

    async def connect_to_server(self, server_name, server_config):
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()

            try:
                # List available tools
                response = await session.list_tools()
                for tool in response.tools:
                    self.sessions[tool.name] = session
                    self.available_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    })

                # List available prompts
                    # 尝试使用 list_prompts() 方法（某些MCP实现）
                try:
                    prompts_response = await session.list_prompts()
                    if prompts_response and prompts_response.prompts:
                        for prompt in prompts_response.prompts:
                            self.sessions[prompt.name] = session
                            self.available_prompts.append({
                                "name": prompt.name,
                                "description": prompt.description,
                                "arguments": prompt.arguments
                            })
                except Exception as e:
                    # Handle the case where list_prompts method is not implemented by the server
                    if "Method not found" in str(e):
                        print(f"Warning: list_prompts method not implemented by the server {tool.name}")
                    else:
                        print(f"Error listing prompts: {e}")
                    # Continue execution even if prompts are not available
                # List available resources
                try:
                    resources_response = await session.list_resources()
                    if resources_response and resources_response.resources:
                        for resource in resources_response.resources:
                            resource_uri = str(resource.uri)
                            self.sessions[resource_uri] = session
                except Exception as e:
                    # Handle the case where list_prompts method is not implemented by the server
                    if "Method not found" in str(e):
                        print(f"Warning: list_resources method not implemented by the server {tool.name}")
                    else:
                        print(f"Error listing resources: {e}")
                    # Continue execution even if prompts are not available

            except Exception as e:
                print(f"Error {e}")
                stack_str = traceback.format_exc()  # 返回字符串
                print(f"Error occurred:{stack_str}")

        except Exception as e:
            print(f"Error connecting to {server_name}: {e}")

    async def connect_to_servers(self):
        try:
            with open("server_config_prompt_resource.json", "r") as file:
                data = json.load(file)
            servers = data.get("mcpServers", {})
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server config: {e}")
            raise

    async def process_query(self, query):
        messages = [{"role": "user", "content": query}]

        while True:
            response = self.client.chat.completions.create(
                model=MODEL,
                tools=self.available_tools,
                messages=messages
            )

            message = response.choices[0].message

            # 检查是否有普通文本内容
            if message.content:
                print(message.content)
                messages.append({"role": "assistant", "content": message.content})
                break

            # 检查是否有工具调用
            elif message.tool_calls:
                # 添加助手消息到历史
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": message.tool_calls
                })

                # 处理每个工具调用
                for tool_call in message.tool_calls:
                    tool_id = tool_call.id
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    print(f"Calling tool {tool_name} with args {tool_args}")

                    # 获取session并调用工具
                    session = self.sessions.get(tool_name)
                    if not session:
                        print(f"Tool '{tool_name}' not found.")
                        break

                    result = await session.call_tool(tool_name, arguments=tool_args)

                    # 添加工具结果到消息历史
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result.content
                    })
            else:
                break

    async def get_resource(self, resource_uri):
        session = self.sessions.get(resource_uri)

        # Fallback for papers URIs - try any papers resource session
        if not session and resource_uri.startswith("papers://"):
            for uri, sess in self.sessions.items():
                if uri.startswith("papers://"):
                    session = sess
                    break

        if not session:
            print(f"Resource '{resource_uri}' not found.")
            return

        try:
            result = await session.read_resource(uri=resource_uri)
            if result and result.contents:
                print(f"\nResource: {resource_uri}")
                print("Content:")
                print(result.contents[0].text)
            else:
                print("No content available.")
        except Exception as e:
            print(f"Error: {e}")

    async def list_prompts(self):
        """List all available prompts."""
        if not self.available_prompts:
            print("No prompts available.")
            return

        print("\nAvailable prompts:")
        for prompt in self.available_prompts:
            print(f"- {prompt['name']}: {prompt['description']}")
            if prompt['arguments']:
                print(f"  Arguments:")
                for arg in prompt['arguments']:
                    arg_name = arg.name if hasattr(arg, 'name') else arg.get('name', '')
                    print(f"    - {arg_name}")

    async def execute_prompt(self, prompt_name, args):
        """Execute a prompt with the given arguments."""
        session = self.sessions.get(prompt_name)
        if not session:
            print(f"Prompt '{prompt_name}' not found.")
            return

        try:
            result = await session.get_prompt(prompt_name, arguments=args)
            if result and result.messages:
                prompt_content = result.messages[0].content

                # Extract text from content (handles different formats)
                if isinstance(prompt_content, str):
                    text = prompt_content
                elif hasattr(prompt_content, 'text'):
                    text = prompt_content.text
                else:
                    # Handle list of content items
                    text = " ".join(item.text if hasattr(item, 'text') else str(item)
                                    for item in prompt_content)

                print(f"\nExecuting prompt '{prompt_name}'...")
                await self.process_query(text)
        except Exception as e:
            print(f"Error: {e}")

    async def chat_loop(self):
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        print("Use @folders to see available topics")
        print("Use @<topic> to search papers in that topic")
        print("Use /prompts to list available prompts")
        print("Use /prompt <name> <arg1=value1> to execute a prompt")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if not query:
                    continue

                if query.lower() == 'quit':
                    break

                # Check for @resource syntax first
                if query.startswith('@'):
                    # Remove @ sign
                    topic = query[1:]
                    if topic == "folders":
                        resource_uri = "papers://folders"
                    else:
                        resource_uri = f"papers://{topic}"
                    await self.get_resource(resource_uri)
                    continue

                # Check for /command syntax
                if query.startswith('/'):
                    parts = query.split()
                    command = parts[0].lower()

                    if command == '/prompts':
                        await self.list_prompts()
                    elif command == '/prompt':
                        if len(parts) < 2:
                            print("Usage: /prompt <name> <arg1=value1> <arg2=value2>")
                            continue

                        prompt_name = parts[1]
                        args = {}

                        # Parse arguments
                        for arg in parts[2:]:
                            if '=' in arg:
                                key, value = arg.split('=', 1)
                                args[key] = value

                        await self.execute_prompt(prompt_name, args)
                    else:
                        print(f"Unknown command: {command}")
                    continue

                await self.process_query(query)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    chatbot = MCP_ChatBot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

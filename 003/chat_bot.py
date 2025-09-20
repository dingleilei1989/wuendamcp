from dotenv import load_dotenv
import openai
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List
import asyncio
import json
import os

load_dotenv()
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"
api_key = os.getenv("DASHSCOPE_API_KEY")



class MCP_ChatBot:

    def __init__(self):
        # Initialize session and client objects
        self.session: ClientSession = None
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=BASE_URL
        )
        self.available_tools: List[dict] = []

    async def process_query(self, query):
        messages = [{'role': 'user', 'content': query}]
        response = self.client.chat.completions.create(
            model=MODEL,
            max_tokens=2024,
            tools=self.available_tools,
            messages=messages
        )

        process_query = True
        while process_query:
            # 获取助手的回复
            message = response.choices[0].message

            # 检查是否有普通文本内容
            if message.content:
                print(message.content)
                process_query = False

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

                    # 执行工具调用
                    result = await self.session.call_tool(tool_name, arguments=tool_args)

                    # 添加工具结果到消息历史
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result.content
                    })

                # 获取下一个回复
                response = self.client.chat.completions.create(
                    model=MODEL,
                    max_tokens=2024,
                    tools=self.available_tools,
                    messages=messages
                )

                # 如果只有文本回复，则结束处理
                if response.choices[0].message.content and not response.choices[0].message.tool_calls:
                    print(response.choices[0].message.content)
                    process_query = False

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break

                await self.process_query(query)
                print("\n")
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def connect_to_server_and_run(self):
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="uv",  # Executable
            args=["run", "research_server.py"],  # Optional command line arguments
            env=None,  # Optional environment variables
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                # Initialize the connection
                await session.initialize()

                # List available tools
                response = await session.list_tools()

                tools = response.tools
                print("\nConnected to server with tools:", [tool.name for tool in tools])

                self.available_tools = [{
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                } for tool in response.tools]

                await self.chat_loop()


async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()


if __name__ == "__main__":
    asyncio.run(main())

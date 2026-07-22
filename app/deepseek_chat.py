import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from app.tools import calculate, read_file

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算两个数字的加减乘除。用户出现明确计算需求时必须调用此工具，不要自己心算。",
            "parameters": {
                "type": "object",
                "properties": {
                    "left": {
                        "type": "number",
                        "description": "左侧数字",
                    },
                    "right": {
                        "type": "number",
                        "description": "右侧数字",
                    },
                    "operator": {
                        "type": "string",
                        "enum": ["+", "-", "*", "/"],
                        "description": "运算符",
                    },
                },
                "required": ["left", "right", "operator"],
                "additionalProperties": False,
            },

        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取 data 文件夹内指定的 UTF-8 文本文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "文件名，例如 energy_note.txt，不包含路径。",
                    }
                },
                "required": ["filename"],
                "additionalProperties": False,
            },
        },
    }
]


def execute_tool(name: str, arguments: str) -> str:


    try:
        data = json.loads(arguments)

        if name == "calculate":
            left = data["left"]
            right = data["right"]
            operator = data["operator"]

            result = calculate(left, right, operator)
            print(f"[Tool] calculate({left}, {right}, '{operator}') -> {result}")
            return str(result)

        if name == "read_file":
            filename = data["filename"]
            result = read_file(filename)
            print(f"[Tool] read_file('{filename}')")
            return result

        return f"未知工具：{name}"

    except (
            json.JSONDecodeError,
            FileNotFoundError,
            KeyError,
            TypeError,
            ValueError,
    ) as error:
        return f"工具执行失败：{error}"


def main() -> None:
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("没有读取到 DEEPSEEK_API_KEY，请检查 .env 文件。")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    messages = [
        {
            "role": "system",
            "content": "你是一个简洁、友好的大模型学习助手。",
        }
    ]

    print("DeepSeek Agent 已启动，输入 exit 退出。")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "exit":
            print("已退出 Agent。")
            break

        if not user_input:
            continue

        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        for _ in range(4):
            response = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                extra_body={"thinking": {"type": "disabled"}},
            )

            message = response.choices[0].message
            messages.append(message)

            if not message.tool_calls:
                print(f"Assistant: {message.content}")
                break

            for tool_call in message.tool_calls:
                result = execute_tool(
                    tool_call.function.name,
                    tool_call.function.arguments,
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
        else:
            print("Assistant: 工具调用次数达到上限，任务已停止。")


if __name__ == "__main__":
    main()
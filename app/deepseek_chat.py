import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from app.tools import calculate, read_file
from app.retrieval import search_notes

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
            "name": "search_notes",
            "description": "从 data 文件夹的学习资料中检索与用户问题相关的内容。涉及天然气、四川盆地、能源资料时优先使用该工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用于检索资料的关键词或问题。",
                    }
                },
                "required": ["query"],
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

        if name == "search_notes":
            query = data["query"]
            result = search_notes(query)
            print(f"[Tool] search_notes('{query}') -> 找到 {len(result)} 份资料")
            return json.dumps(result, ensure_ascii=False)

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
            "content": (
            "你是一个能源学习助手。回答天然气、四川盆地、能源资料等问题时，"
            "必须先调用 search_notes 检索 data 文件夹中的资料，再仅依据检索结果回答。"
            "若检索结果为空，请明确说明资料中没有相关内容，不要自行编造。"
            "不得补充检索资料中没有明确出现的地名、数据或事实。"
            "回答末尾必须用“资料来源：文件名”的格式列出实际使用过的资料文件。"
        ),
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
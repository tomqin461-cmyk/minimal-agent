import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from app.tools import calculate, read_file
from app.retrieval import search_notes
from app.reranked_retrieval import RerankedRetriever

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
    },
    {
        "type": "function",
        "function": {
            "name": "search_semantic_notes",
            "description": "使用本地向量模型，从 data 文件夹中按语义检索资料。回答天然气、四川盆地、能源等自然语言问题时优先使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户的完整自然语言问题。",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
]
semantic_retriever: RerankedRetriever | None = None


def get_semantic_retriever() -> RerankedRetriever:
    global semantic_retriever

    if semantic_retriever is None:
        print("[System] 正在加载向量检索模型、Reranker 和本地数据库...")
        semantic_retriever = RerankedRetriever()
    return semantic_retriever

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

        if name == "search_semantic_notes":
            query = data["query"]
            result = get_semantic_retriever().search(query)
            print(
                f"[Tool] search_semantic_notes('{query}') "
                f"-> 找到 {len(result)} 个文本块"
            )
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

def ask_agent(
    question: str,
    history: list[dict] | None = None,
) -> dict:
    """回答一个独立问题，供网页/API 调用。"""
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
                "必须先调用 search_semantic_notes 检索 data 文件夹中的资料，"
                "再仅依据检索结果回答。"
                "若检索结果为空，请明确说明资料中没有相关内容，不要自行编造。"
                "不得补充检索资料中没有明确出现的地名、数据或事实。"
                "回答末尾必须用“资料来源：文件名”的格式列出实际使用过的资料文件。"
            ),
        },
        *(history or []),
        {
            "role": "user",
            "content": question,
        },
    ]
    sources: set[str] = set()
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
            return {
                "answer": message.content or "模型没有返回内容。",
                "sources": sorted(sources),
            }

        for tool_call in message.tool_calls:
            result = execute_tool(
                tool_call.function.name,
                tool_call.function.arguments,
            )
            if tool_call.function.name == "search_semantic_notes":
                search_results = json.loads(result)

                for item in search_results:
                    source = item.get("source")
                    if source:
                        sources.add(source)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    return {
        "answer": "工具调用次数达到上限，任务已停止。",
        "sources": sorted(sources),
    }

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
            "必须先调用 search_semantic_notes检索 data 文件夹中的资料，再仅依据检索结果回答。"
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
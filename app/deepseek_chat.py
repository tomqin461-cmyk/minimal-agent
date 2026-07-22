import os

from dotenv import load_dotenv
from openai import OpenAI


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

    print("DeepSeek 聊天已启动，输入 exit 退出。")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "exit":
            print("已退出聊天。")
            break

        if not user_input:
            continue

        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            extra_body={"thinking": {"type": "disabled"}},
        )

        answer = response.choices[0].message.content
        print(f"Assistant: {answer}")

        messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )


if __name__ == "__main__":
    main()
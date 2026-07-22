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

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {
                "role": "system",
                "content": "你是一个简洁、友好的大模型学习助手。",
            },
            {
                "role": "user",
                "content": "请用三句话解释什么是 AI Agent。",
            },
        ],
        extra_body={"thinking": {"type": "disabled"}},
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
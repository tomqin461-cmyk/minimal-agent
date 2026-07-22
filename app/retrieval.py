from pathlib import Path
import jieba

DATA_DIRECTORY = Path(__file__).parent.parent / "data"


def search_notes(query: str) -> list[str]:
    keywords = {
        word.lower()
        for word in jieba.lcut(query)
        if len(word.strip()) > 1
    }
    matches = []

    for file_path in DATA_DIRECTORY.glob("*.txt"):
        content = file_path.read_text(encoding="utf-8")

        if any(keyword in content.lower() for keyword in keywords):
            matches.append(f"文件：{file_path.name}\n内容：{content}")

    return matches
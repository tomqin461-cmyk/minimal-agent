from pathlib import Path

import jieba

DATA_DIRECTORY = Path(__file__).parent.parent / "data"


def search_notes(query: str, top_k: int = 3) -> list[dict]:
    keywords = {
        word.lower()
        for word in jieba.lcut(query)
        if len(word.strip()) > 1
    }

    matches = []

    for file_path in DATA_DIRECTORY.glob("*.txt"):
        content = file_path.read_text(encoding="utf-8")
        content_lower = content.lower()

        score = sum(keyword in content_lower for keyword in keywords)
        if score > 0:
            matches.append(
                {
                    "source": file_path.name,
                    "content": content,
                    "score": score,
                }
            )

    matches.sort(key=lambda item: item["score"], reverse=True)
    return matches[:top_k]
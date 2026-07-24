from pathlib import Path

from sentence_transformers import SentenceTransformer


DATA_DIRECTORY = Path(__file__).parent.parent / "data"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def split_text(
    text: str,
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[str]:
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap 必须大于等于 0 且小于 chunk_size。")

    text = text.strip()
    chunks = []
    start = 0

    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap

    return chunks


def load_chunks() -> list[dict]:
    chunks = []

    for file_path in DATA_DIRECTORY.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        for chunk_id, content in enumerate(split_text(text), start=1):
            chunks.append(
                {
                    "source": file_path.name,
                    "chunk_id": chunk_id,
                    "content": content,
                }
            )

    return chunks


class SemanticRetriever:
    def __init__(self) -> None:
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")
        self.chunks = load_chunks()

        contents = [chunk["content"] for chunk in self.chunks]
        self.embeddings = self.model.encode(
            contents,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_embedding = self.model.encode(
            [QUERY_INSTRUCTION + query],
            normalize_embeddings=True,
        )

        scores = (query_embedding @ self.embeddings.T)[0]
        indices = scores.argsort()[::-1][:top_k]

        return [
            {
                **self.chunks[index],
                "score": float(scores[index]),
            }
            for index in indices
        ]


def main() -> None:
    retriever = SemanticRetriever()

    query = "居民做饭和取暖可以使用什么能源？"
    results = retriever.search(query)

    for result in results:
        print(
            f"{result['score']:.4f} | "
            f"{result['source']} | "
            f"chunk {result['chunk_id']}"
        )
        print(result["content"])
        print()


if __name__ == "__main__":
    main()
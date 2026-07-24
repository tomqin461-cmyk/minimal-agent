from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


DATABASE_DIRECTORY = Path(__file__).parent.parent / "data" / "chroma"
COLLECTION_NAME = "energy_notes"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class ChromaSemanticRetriever:
    def __init__(self) -> None:
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")

        client = chromadb.PersistentClient(path=str(DATABASE_DIRECTORY))
        self.collection = client.get_collection(COLLECTION_NAME)

    def search(self, query: str, top_k: int = 2) -> list[dict]:
        if self.collection.count() == 0:
            return []

        query_embedding = self.model.encode(
            [QUERY_INSTRUCTION + query],
            normalize_embeddings=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        return [
            {
                "source": metadata["source"],
                "chunk_id": metadata["chunk_id"],
                "content": document,
                "distance": distance,
            }
            for document, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                strict=True,
            )
        ]


def main() -> None:
    retriever = ChromaSemanticRetriever()

    results = retriever.search(
        "居民做饭和取暖可以使用什么能源？"
    )

    for result in results:
        print(
            f"{result['distance']:.4f} | "
            f"{result['source']} | "
            f"chunk {result['chunk_id']}"
        )
        print(result["content"])
        print()


if __name__ == "__main__":
    main()
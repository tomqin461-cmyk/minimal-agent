from sentence_transformers import CrossEncoder


MODEL_NAME = "BAAI/bge-reranker-base"


class LocalReranker:
    def __init__(self) -> None:
        self.model = CrossEncoder(
            MODEL_NAME,
            device="cpu",
        )

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 1,
    ) -> list[dict]:
        if not candidates:
            return []

        pairs = [
            (query, candidate["content"])
            for candidate in candidates
        ]

        scores = self.model.predict(pairs)

        ranked_results = [
            {
                **candidate,
                "rerank_score": float(score),
            }
            for candidate, score in zip(
                candidates,
                scores,
                strict=True,
            )
        ]

        ranked_results.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return ranked_results[:top_k]

if __name__ == "__main__":
    from app.chroma_retrieval import ChromaSemanticRetriever

    query = "居民做饭和取暖可以使用什么能源？"

    retriever = ChromaSemanticRetriever()
    candidates = retriever.search(query, top_k=3)

    print("===== 初始候选资料 =====")
    for candidate in candidates:
        print(
            f"{candidate['distance']:.4f} | "
            f"{candidate['source']}"
        )
        print(candidate["content"])
        print()

    reranker = LocalReranker()
    results = reranker.rerank(
        query,
        candidates,
        top_k=3,
    )

    print("===== Reranker 重排结果 =====")
    for result in results:
        print(
            f"{result['rerank_score']:.4f} | "
            f"{result['source']}"
        )
        print(result["content"])
        print()
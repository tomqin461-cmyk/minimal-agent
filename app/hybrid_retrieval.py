from app.chroma_retrieval import ChromaSemanticRetriever
from app.retrieval import search_notes


RRF_K = 60


class HybridRetriever:
    def __init__(
            self,
            semantic_retriever: ChromaSemanticRetriever | None = None,
    ) -> None:
        self.semantic_retriever = (
                semantic_retriever
                or ChromaSemanticRetriever()
        )

    def search(self, query: str, top_k: int = 2) -> list[dict]:
        keyword_results = search_notes(query, top_k=3)
        semantic_results = self.semantic_retriever.search(
            query,
            top_k=3,
        )

        merged_results = {}

        for method, results in [
            ("keyword", keyword_results),
            ("semantic", semantic_results),
        ]:
            for rank, result in enumerate(results, start=1):
                chunk_id = result.get("chunk_id", 1)
                key = f"{result['source']}:{chunk_id}"

                if key not in merged_results:
                    merged_results[key] = {
                        **result,
                        "chunk_id": chunk_id,
                        "rrf_score": 0.0,
                        "matched_by": [],
                    }

                merged_results[key]["rrf_score"] += 1 / (RRF_K + rank)
                merged_results[key]["matched_by"].append(method)

        ranked_results = sorted(
            merged_results.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        return ranked_results[:top_k]


def main() -> None:
    retriever = HybridRetriever()

    results = retriever.search(
        "居民做饭和取暖可以使用什么能源？"
    )

    for result in results:
        print(
            f"{result['rrf_score']:.4f} | "
            f"{result['source']} | "
            f"来自：{', '.join(result['matched_by'])}"
        )
        print(result["content"])
        print()


if __name__ == "__main__":
    main()
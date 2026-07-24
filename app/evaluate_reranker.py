from app.chroma_retrieval import ChromaSemanticRetriever
from app.evaluate_retrieval import (
    get_rank,
    get_sources,
    load_cases,
    show_rank,
)
from app.reranker import LocalReranker


CANDIDATE_K = 3


def evaluate(top_k: int) -> None:
    cases = load_cases()
    retriever = ChromaSemanticRetriever()
    reranker = LocalReranker()

    retrieval_hits = 0
    rerank_hits = 0
    retrieval_mrr = 0.0
    rerank_mrr = 0.0

    for case in cases:
        question = case["question"]
        expected_source = case["expected_source"]

        candidates = retriever.search(
            question,
            top_k=CANDIDATE_K,
        )
        retrieval_results = candidates[:top_k]
        rerank_results = reranker.rerank(
            question,
            candidates,
            top_k=top_k,
        )

        retrieval_sources = get_sources(retrieval_results)
        rerank_sources = get_sources(rerank_results)

        retrieval_rank = get_rank(
            retrieval_sources,
            expected_source,
        )
        rerank_rank = get_rank(
            rerank_sources,
            expected_source,
        )

        retrieval_hits += retrieval_rank is not None
        rerank_hits += rerank_rank is not None
        retrieval_mrr += 1 / retrieval_rank if retrieval_rank else 0
        rerank_mrr += 1 / rerank_rank if rerank_rank else 0

        print(f"\n问题：{question}")
        print(f"期望来源：{expected_source}")
        print(
            f"向量检索：{retrieval_sources} | "
            f"{show_rank(retrieval_rank)}"
        )
        print(
            f"Reranker：{rerank_sources} | "
            f"{show_rank(rerank_rank)}"
        )

    total = len(cases)

    print(f"\n===== Reranker Evaluation @ {top_k} =====")
    print(
        f"向量检索：Recall {retrieval_hits}/{total} | "
        f"MRR {retrieval_mrr / total:.3f}"
    )
    print(
        f"Reranker：Recall {rerank_hits}/{total} | "
        f"MRR {rerank_mrr / total:.3f}"
    )


def main() -> None:
    for top_k in [1, 2]:
        evaluate(top_k)


if __name__ == "__main__":
    main()
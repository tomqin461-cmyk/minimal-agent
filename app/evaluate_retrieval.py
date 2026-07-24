import json
from pathlib import Path

from app.chroma_retrieval import ChromaSemanticRetriever
from app.hybrid_retrieval import HybridRetriever
from app.retrieval import search_notes


EVALUATION_FILE = (
    Path(__file__).parent.parent
    / "tests"
    / "evaluation_cases.json"
)


def load_cases() -> list[dict]:
    return json.loads(EVALUATION_FILE.read_text(encoding="utf-8"))


def get_sources(results: list[dict]) -> list[str]:
    return [result["source"] for result in results]


def get_rank(
    sources: list[str],
    expected_source: str,
) -> int | None:
    try:
        return sources.index(expected_source) + 1
    except ValueError:
        return None


def show_rank(rank: int | None) -> str:
    if rank is None:
        return "未命中"

    return f"第 {rank} 名"


def evaluate(
    cases: list[dict],
    semantic_retriever: ChromaSemanticRetriever,
    hybrid_retriever: HybridRetriever,
    top_k: int,
) -> None:
    keyword_hits = 0
    semantic_hits = 0
    hybrid_hits = 0

    keyword_mrr = 0.0
    semantic_mrr = 0.0
    hybrid_mrr = 0.0

    for case in cases:
        question = case["question"]
        expected_source = case["expected_source"]

        keyword_sources = get_sources(
            search_notes(question, top_k=top_k)
        )
        semantic_sources = get_sources(
            semantic_retriever.search(question, top_k=top_k)
        )
        hybrid_sources = get_sources(
            hybrid_retriever.search(question, top_k=top_k)
        )

        keyword_rank = get_rank(keyword_sources, expected_source)
        semantic_rank = get_rank(semantic_sources, expected_source)
        hybrid_rank = get_rank(hybrid_sources, expected_source)

        keyword_hits += keyword_rank is not None
        semantic_hits += semantic_rank is not None
        hybrid_hits += hybrid_rank is not None

        keyword_mrr += 1 / keyword_rank if keyword_rank else 0
        semantic_mrr += 1 / semantic_rank if semantic_rank else 0
        hybrid_mrr += 1 / hybrid_rank if hybrid_rank else 0

        print(f"\n问题：{question}")
        print(f"期望来源：{expected_source}")
        print(
            f"关键词检索：{keyword_sources} | "
            f"{show_rank(keyword_rank)}"
        )
        print(
            f"向量检索：{semantic_sources} | "
            f"{show_rank(semantic_rank)}"
        )
        print(
            f"混合检索：{hybrid_sources} | "
            f"{show_rank(hybrid_rank)}"
        )

    total = len(cases)

    print(f"\n===== Recall@{top_k} / MRR@{top_k} =====")
    print(
        f"关键词检索：Recall {keyword_hits}/{total} | "
        f"MRR {keyword_mrr / total:.3f}"
    )
    print(
        f"向量检索：Recall {semantic_hits}/{total} | "
        f"MRR {semantic_mrr / total:.3f}"
    )
    print(
        f"混合检索：Recall {hybrid_hits}/{total} | "
        f"MRR {hybrid_mrr / total:.3f}"
    )


def main() -> None:
    cases = load_cases()
    semantic_retriever = ChromaSemanticRetriever()
    hybrid_retriever = HybridRetriever(semantic_retriever)

    for top_k in [1, 2]:
        evaluate(
            cases,
            semantic_retriever,
            hybrid_retriever,
            top_k,
        )


if __name__ == "__main__":
    main()
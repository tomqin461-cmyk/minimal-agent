import json
from pathlib import Path

from app.chroma_retrieval import ChromaSemanticRetriever
from app.retrieval import search_notes


EVALUATION_FILE = (
    Path(__file__).parent.parent
    / "tests"
    / "evaluation_cases.json"
)


def load_cases() -> list[dict]:
    return json.loads(EVALUATION_FILE.read_text(encoding="utf-8"))


def get_sources(results: list[dict]) -> set[str]:
    return {result["source"] for result in results}


def main() -> None:
    cases = load_cases()
    semantic_retriever = ChromaSemanticRetriever()

    keyword_hits = 0
    semantic_hits = 0

    for case in cases:
        question = case["question"]
        expected_source = case["expected_source"]

        keyword_sources = get_sources(
            search_notes(question, top_k=2)
        )
        semantic_sources = get_sources(
            semantic_retriever.search(question, top_k=2)
        )

        keyword_hit = expected_source in keyword_sources
        semantic_hit = expected_source in semantic_sources

        keyword_hits += keyword_hit
        semantic_hits += semantic_hit

        print(f"\n问题：{question}")
        print(f"期望来源：{expected_source}")
        print(f"关键词检索：{keyword_sources} | 命中：{keyword_hit}")
        print(f"向量检索：{semantic_sources} | 命中：{semantic_hit}")

    total = len(cases)

    print("\n===== Recall@2 =====")
    print(f"关键词检索：{keyword_hits}/{total}")
    print(f"向量检索：{semantic_hits}/{total}")


if __name__ == "__main__":
    main()
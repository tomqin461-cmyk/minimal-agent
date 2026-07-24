from app.chroma_retrieval import ChromaSemanticRetriever
from app.reranker import LocalReranker


class RerankedRetriever:
    def __init__(self) -> None:
        self.semantic_retriever = ChromaSemanticRetriever()
        self.reranker = LocalReranker()

    def search(
        self,
        query: str,
        candidate_k: int = 3,
        top_k: int = 2,
    ) -> list[dict]:
        candidates = self.semantic_retriever.search(
            query,
            top_k=candidate_k,
        )

        return self.reranker.rerank(
            query,
            candidates,
            top_k=top_k,
        )
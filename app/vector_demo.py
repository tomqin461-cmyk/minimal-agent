from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-zh-v1.5"

DOCUMENTS = [
    "天然气可用于发电、城市燃气、工业燃料和化工原料。",
    "四川盆地的重要天然气开发类型包括页岩气和常规天然气。",
    "太阳能是一种可再生能源。",
]


def main() -> None:
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    query = "居民做饭和取暖可以使用什么能源？"
    query_text = f"为这个句子生成表示以用于检索相关文章：{query}"

    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True,
    )
    document_embeddings = model.encode(
        DOCUMENTS,
        normalize_embeddings=True,
    )

    scores = (query_embedding @ document_embeddings.T)[0]

    ranked_results = sorted(
        zip(DOCUMENTS, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )

    for document, score in ranked_results:
        print(f"{score:.4f} | {document}")


if __name__ == "__main__":
    main()

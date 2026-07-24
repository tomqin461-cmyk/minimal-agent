from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from app.vector_retrieval import MODEL_NAME, load_chunks


DATABASE_DIRECTORY = Path(__file__).parent.parent / "data" / "chroma"
COLLECTION_NAME = "energy_notes"


def main() -> None:
    print("[System] 正在读取资料并生成向量...")

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    chunks = load_chunks()

    contents = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(
        contents,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    client = chromadb.PersistentClient(path=str(DATABASE_DIRECTORY))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    collection.upsert(
        ids=[
            f"{chunk['source']}-{chunk['chunk_id']}"
            for chunk in chunks
        ],
        documents=contents,
        metadatas=[
            {
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
            }
            for chunk in chunks
        ],
        embeddings=embeddings,
    )

    print(f"[System] 已保存 {collection.count()} 个文本块到：")
    print(DATABASE_DIRECTORY)


if __name__ == "__main__":
    main()
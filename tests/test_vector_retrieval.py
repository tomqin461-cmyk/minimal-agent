import pytest

from app.vector_retrieval import load_chunks, split_text


def test_split_text_keeps_overlap():
    chunks = split_text(
        "abcdefghij",
        chunk_size=4,
        overlap=1,
    )

    assert chunks[:3] == ["abcd", "defg", "ghij"]


def test_split_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        split_text(
            "test",
            chunk_size=4,
            overlap=4,
        )


def test_load_chunks_keeps_source_and_chunk_id():
    chunks = load_chunks()

    sources = {chunk["source"] for chunk in chunks}

    assert "natural_gas.txt" in sources
    assert "sichuan_basin.txt" in sources
    assert all(chunk["chunk_id"] >= 1 for chunk in chunks)
    assert all(chunk["content"] for chunk in chunks)
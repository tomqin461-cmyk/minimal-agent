from app.retrieval import search_notes


def test_search_notes_finds_natural_gas_document():
    results = search_notes("天然气有哪些用途？")

    sources = {item["source"] for item in results}

    assert "natural_gas.txt" in sources


def test_search_notes_sorts_by_score():
    results = search_notes("天然气有哪些用途？")

    scores = [item["score"] for item in results]

    assert scores == sorted(scores, reverse=True)


def test_search_notes_returns_empty_list_for_unknown_topic():
    results = search_notes("xyzunmatchedtopic")

    assert results == []


def test_search_notes_respects_top_k():
    results = search_notes("天然气", top_k=1)

    assert len(results) == 1
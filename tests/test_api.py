from fastapi.testclient import TestClient

import app.api as api


client = TestClient(api.app)


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_agent_answer(monkeypatch) -> None:
    def fake_ask_agent(
            question: str,
            history: list[dict],
    ) -> dict:
        assert history == []
        assert question == "四川盆地重点开发哪些天然气资源？"

        return {
            "answer": "页岩气和常规天然气。",
            "sources": ["sichuan_basin.txt"],
        }

    monkeypatch.setattr(api, "ask_agent", fake_ask_agent)

    response = client.post(
        "/chat",
        json={
            "question": "四川盆地重点开发哪些天然气资源？",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "页岩气和常规天然气。",
        "sources": ["sichuan_basin.txt"],
        "status": "success",
    }


def test_chat_rejects_empty_question() -> None:
    response = client.post(
        "/chat",
        json={
            "question": "",
        },
    )

    assert response.status_code == 422

def test_chat_passes_history_to_agent(monkeypatch) -> None:
    def fake_ask_agent(
        question: str,
        history: list[dict],
    ) -> dict:
        assert question == "那里哪一种属于非常规天然气？"
        assert history == [
            {
                "role": "user",
                "content": "四川盆地重点开发哪些天然气资源？",
            },
            {
                "role": "assistant",
                "content": "页岩气和常规天然气。",
            },
        ]

        return {
            "answer": "页岩气。",
            "sources": ["sichuan_basin.txt"],
        }

    monkeypatch.setattr(api, "ask_agent", fake_ask_agent)

    response = client.post(
        "/chat",
        json={
            "question": "那里哪一种属于非常规天然气？",
            "history": [
                {
                    "role": "user",
                    "content": "四川盆地重点开发哪些天然气资源？",
                },
                {
                    "role": "assistant",
                    "content": "页岩气和常规天然气。",
                },
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "页岩气。"
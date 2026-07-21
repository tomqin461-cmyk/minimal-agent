from app.greeting import greet


def test_greet() -> None:
    assert greet("Alice") == "Hello, Alice! Welcome to EnergyMind."
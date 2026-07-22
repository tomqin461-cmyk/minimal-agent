from app.tools import calculate


def test_add() -> None:
    assert calculate(2, 3, "+") == 5


def test_multiply() -> None:
    assert calculate(1250, 0.08, "*") == 100.0


def test_divide_by_zero() -> None:
    try:
        calculate(1, 0, "/")
    except ValueError as error:
        assert str(error) == "除数不能为 0。"
    else:
        raise AssertionError("应该抛出 ValueError")
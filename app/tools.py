def calculate(left: float, right: float, operator: str) -> float:
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if operator == "/":
        if right == 0:
            raise ValueError("除数不能为 0。")
        return left / right

    raise ValueError(f"不支持的运算符：{operator}")

from pathlib import Path


DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data"


def read_file(filename: str) -> str:
    requested_path = (DATA_DIRECTORY / filename).resolve()

    if requested_path.parent != DATA_DIRECTORY.resolve():
        raise ValueError("只能读取 data 文件夹内的直接文件。")

    if requested_path.suffix != ".txt":
        raise ValueError("只允许读取 .txt 文件。")

    if not requested_path.exists():
        raise FileNotFoundError(f"文件不存在：{filename}")

    return requested_path.read_text(encoding="utf-8")
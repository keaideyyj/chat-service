import pytest

from app import tools


# ── calculate：正常路径 ────────────────────────────────────

@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2", "3"),
        ("2 - 1", "1"),
        ("3 * 4", "12"),
        ("8 / 2", "4"),
        ("7 / 2", "3.5"),
        ("2 ** 3", "8"),
        ("10 % 3", "1"),
        ("-2", "-2"),
        ("(12+8)*3", "60"),
    ],
)
def test_calculate_ok(expression, expected):
    assert tools.calculate(expression) == expected


# ── calculate：安全边界（本文件最值钱的部分）────────────────

@pytest.mark.parametrize(
    "expression",
    [
        '__import__("os").system("dir")',
        "open('x.txt', 'w')",
        "[1, 2][0]",
        "1 if 1 else 2",
        "lambda: 1",
        "2 ** 999999",
        "9 / 0",
    ],
)
def test_calculate_rejects_unsafe(expression):
    with pytest.raises(Exception):
        tools.calculate(expression)


def test_calculate_rejects_statement():
    with pytest.raises(SyntaxError):
        tools.calculate("import os")


# ── search_kb ────────────────────────────────────────────

def test_search_kb_hit():
    assert "年假" in tools.search_kb("年假")


def test_search_kb_miss():
    assert "没有找到" in tools.search_kb("量子计算机")


# ── run_tool：分发 ────────────────────────────────────────

def test_run_tool_passes_arguments():
    assert tools.run_tool("calculate", {"expression": "6*7"}) == "42"


def test_run_tool_unknown_name_returns_text():
    assert "没有名为" in tools.run_tool("no_such_tool", {})


def test_run_tool_wraps_exception_as_text():
    result = tools.run_tool("calculate", {"expression": "1 +"})
    assert result.startswith("错误：")


# ── 注册表一致性 ──────────────────────────────────────────

def test_every_spec_name_matches_registry_key():
    for key, item in tools.TOOLS.items():
        assert item["spec"]["function"]["name"] == key


def test_every_description_is_long_enough():
    for key, item in tools.TOOLS.items():
        desc = item["spec"]["function"]["description"]
        assert len(desc) >= 10, f"{key} 的描述太短，模型可能不会调用它"
import ast
import operator
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from typing import Any

# ── 工具实现 ────────────────────────────────────────────────

def get_current_time(timezone_offset: int = 8) -> str:
    tz = dt_timezone(timedelta(hours=timezone_offset))
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %z")


_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op = _ALLOWED_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("指数过大")
        return op(_eval_node(node.left), right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("表达式里只允许数字和四则运算")


def calculate(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    value = _eval_node(tree.body)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


_KB = [
    {"id": "kb-1", "title": "入职流程", "content": "新员工需在入职当天提交身份证复印件与学历证明。"},
    {"id": "kb-2", "title": "年假规则", "content": "工作满一年享有 5 天年假，满三年 10 天。"},
    {"id": "kb-3", "title": "报销时限", "content": "费用发生后 30 天内提交报销，超期需部门负责人审批。"},
]


MIN_SCORE = 0.60  # 正例最低 0.687（年假），负例最高 0.508（股票代码），取中点让两侧容错均衡


def search_kb(query: str, top_k: int = 2) -> str:
    from .rag import search

    hits = search(query, top_k=top_k)
    relevant = [h for h in hits if h["score"] >= MIN_SCORE]
    if not relevant:
        return "知识库中没有找到相关内容。"
    return "\n".join(
        f"[{h['doc']} / {h['title']}] {h['text']}\n（相关度 {h['score']:.3f}）"
        for h in relevant
    )

TOOLS: dict[str, dict[str, Any]] = {
    "get_current_time": {
        "func": get_current_time,
        "spec": {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "获取当前日期与时间。当用户问'现在几点''今天几号''星期几'时调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone_offset": {
                            "type": "integer",
                            "description": "与 UTC 的小时差。中国用户填 8，不填默认 8。",
                        }
                    },
                    "required": [],
                },
            },
        },
    },
    "calculate": {
        "func": calculate,
        "spec": {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "计算数学表达式。用户需要精确算术时调用，不要自己心算。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "只含数字与 + - * / % ** 的表达式，例如 (12+8)*3。",
                        }
                    },
                    "required": ["expression"],
                },
            },
        },
    },
    "search_kb": {
        "func": search_kb,
        "spec": {
            "type": "function",
            "function": {
                "name": "search_kb",
                "description": "在内部知识库检索公司制度（入职、年假、报销等）。涉及公司规定时必须调用，不要凭常识回答。每次对话最多调用一次；如果返回结果为空，直接告知用户未查到，不要更换关键词重复调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "检索关键词，例如 年假 或 报销"},
                        "top_k": {"type": "integer", "description": "返回条数，默认 2"},
                    },
                    "required": ["query"],
                },
            },
        },
    },
}


def tool_specs() -> list[dict]:
    return [item["spec"] for item in TOOLS.values()]


def run_tool(name: str, arguments: dict) -> str:
    item = TOOLS.get(name)
    if item is None:
        return f"错误：没有名为 {name} 的工具。"
    # 宽捕获是设计而非疏忽：工具由模型（外部输入）触发，
    # 任何异常都必须转成文本回灌给模型，否则一个参数错误就能让整个请求 500。
    try:
        return item["func"](**arguments)
    except Exception as exc:  # noqa: BLE001
        return f"工具执行失败（内部错误：{type(exc).__name__}）。请用自然语言向用户说明情况，不要复述这段技术细节。"
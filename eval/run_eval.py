"""跑评估集：固定任务 → 真实调用 agent → 统计成功率。

用法：
    python -m eval.run_eval              全跑 10 条
    python -m eval.run_eval --limit 3    只跑前 3 条（省钱）
    python -m eval.run_eval --case c05   只跑一条
"""
import argparse
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

from app.agent import run_agent
from app.main import _build_messages
from app.prompts import PROMPT_VERSION, get_variant
from app.schemas import ChatRequest
from app.config import get_settings

HERE = Path(__file__).resolve().parent
CASES_FILE = HERE / "cases.jsonl"

def assert_treatment(cases: list[dict]) -> None:
    """处理完整性检查：确认"当前变体"与"实际注入"一致。

    on  → 必须恰好 1 条非空 system 消息
    off → 必须没有 system 消息
    """
    first = cases[0]
    messages = _build_messages(ChatRequest(message=first["input"]))
    variant = get_variant()
    system_msgs = [m for m in messages if m.role == "system"]

    print(f"[treatment] variant={variant} prompt_version={PROMPT_VERSION}")
    print(f"[treatment] roles={[m.role for m in messages]}")
    print(
        f"[treatment] system 消息数={len(system_msgs)} "
        f"长度={len(system_msgs[0].content) if system_msgs else 0}"
    )

    if variant == "on":
        assert len(system_msgs) == 1, f"变体 on，但 system 消息数={len(system_msgs)}，处理未生效"
        assert len(system_msgs[0].content) > 100, "变体 on，但系统提示过短，模板可能没注入"
    else:
        assert not system_msgs, f"变体 {variant}，但仍有 {len(system_msgs)} 条 system 消息，处理未生效"

def load_cases() -> list[dict]:
    cases = []
    text = CASES_FILE.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"cases.jsonl 第 {lineno} 行不是合法 JSON（{exc}）\n内容开头：{line[:100]}"
            ) from exc
    return cases


def check(case: dict, result: dict) -> tuple[bool, str]:
    tools_called = [s.tool for s in result["steps"] if s.type == "tool"]

    for expected in case.get("expect_tools", []):
        if expected not in tools_called:
            return False, f"缺少工具 {expected}（实际调用 {tools_called}）"

    for forbidden in case.get("forbid_tools", []):
        if forbidden in tools_called:
            return False, f"不该调用 {forbidden}"

    for pattern in case.get("forbid_patterns", []):
        if re.search(pattern, result["content"]):
            return False, f"回答命中禁止模式：{pattern}"

    for needle in case.get("expect_contains", []):
        if needle not in result["content"]:
            return False, f"答案缺少 {needle!r}"

    if "expect_contains_any" in case:
        needles = case["expect_contains_any"]
        if not any(n in result["content"] for n in needles):
            return False, f"回答里没有出现任意一种期望表述：{needles}"

    if "max_tool_calls" in case and len(tools_called) > case["max_tool_calls"]:
        return False, f"工具调用 {len(tools_called)} 次，超过上限 {case['max_tool_calls']}"

    if "max_cost" in case and result["cost_yuan"] > case["max_cost"]:
        return False, f"成本 {result['cost_yuan']:.6f} 超过上限 {case['max_cost']}"        

    return True, "ok"

def run_round(cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        started = time.perf_counter()
        result = run_agent([m.model_dump() for m in _build_messages(ChatRequest(message=case["input"]))])
        wall_ms = int((time.perf_counter() - started) * 1000)

        manual = case.get("judge") == "manual"
        passed, reason = check(case, result) if not manual else (None, "需人工判断")

        trace = []
        for s in result["steps"]:
            if s.type == "tool":
                args_text = json.dumps(s.arguments, ensure_ascii=False)
                result_text = (s.result or "").replace("\n", " ")
                if len(result_text) > 300:
                    result_text = result_text[:300] + "…"
                trace.append(f"`{s.tool}({args_text})` → {result_text}")
            else:
                trace.append(f"最终回答（{len(s.content or '')} 字）")

        rows.append(
            {
                "id": case["id"],
                "input": case["input"],
                "manual": manual,
                "passed": passed,
                "reason": reason,
                "tools": [s.tool for s in result["steps"] if s.type == "tool"],
                "steps": len(result["steps"]),
                "cost": result["cost_yuan"],
                "latency": wall_ms,
                "stop": result["stop_reason"],
                "content": result["content"],
                "content_len": len(result["content"] or ""),
                "trace": trace,
            }
        )

        flag = "MANUAL" if manual else ("PASS" if passed else "FAIL")
        print(
            f"[{flag}] {case['id']} {case['input'][:24]} "
            f"stop={result['stop_reason']} cost={result['cost_yuan']:.6f} "
            f"字={len(result['content'] or '')} tools={len(trace) - 1 if trace else 0}"
        )
        if not manual and not passed:
            print(f"        理由：{reason}")

    return rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--case", default="")
    parser.add_argument("--repeat", type=int, default=1, help="整轮重复次数，用于估计波动")
    args = parser.parse_args()

    cases = load_cases()
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("没有匹配的用例，检查 --case / --limit 参数")
        return
    assert_treatment(cases)    
    rounds: list[list[dict]] = []
    for rep in range(1, args.repeat + 1):
        if args.repeat > 1:
            print(f"\n===== 第 {rep}/{args.repeat} 轮 =====")
        rounds.append(run_round(cases))

    rows = rounds[-1]  # 报告用最后一轮

    print("\n== 各轮汇总 ==")
    for index, round_rows in enumerate(rounds, start=1):
        auto = [r for r in round_rows if not r["manual"]]
        ok = sum(1 for r in auto if r["passed"])
        print(
            f"第 {index} 轮：{ok}/{len(auto)} 通过　"
            f"成本 {sum(r['cost'] for r in round_rows):.6f} 元　"
            f"平均字数 {statistics.mean(r['content_len'] for r in round_rows):.0f}　"
            f"平均延迟 {statistics.mean(r['latency'] for r in round_rows):.0f}ms"
        )

    if args.repeat > 1:
        print("\n== 逐条稳定性 ==")
        per_case: dict[str, list[dict]] = {}
        for round_rows in rounds:
            for row in round_rows:
                per_case.setdefault(row["id"], []).append(row)

        for cid, case_rows in per_case.items():
            if case_rows[0]["manual"]:
                print(f"{cid}  人工待判")
                continue
            ok = sum(1 for r in case_rows if r["passed"])
            tag = "稳定" if ok == len(case_rows) else ("全挂" if ok == 0 else "抖动")
            print(f"{cid}  {ok}/{len(case_rows)}  {tag}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report = HERE / f"report-{stamp}.md"
    lines = [
        f"# 评估报告 {stamp}",
        "",
        f"- 变体：`{get_variant()}`（prompt {PROMPT_VERSION}）",
        f"- 温度：`{get_settings().temperature}`　模型：`{get_settings().model_name}`",
        f"- 用例：{len(rows)} 条　重复：{args.repeat} 轮",
        "",
        "| # | 用例 | 工具 | 步数 | 成本 | 延迟 | 结果 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        flag = "人工" if r["manual"] else ("✅" if r["passed"] else "❌")
        lines.append(
            f"| {r['id']} | {r['input'][:20]} | {','.join(r['tools']) or '-'} "
            f"| {r['steps']} | {r['cost']:.6f} | {r['latency']}ms | {flag} |"
        )

    lines.append("")
    lines.append("## 详情")
    for r in rows:
        flag = "人工待判" if r["manual"] else ("通过" if r["passed"] else "失败")
        lines.append("")
        lines.append(f"### {r['id']}｜{r['input']}｜{flag}")
        lines.append("")
        lines.append(
            f"- 结束原因：`{r['stop']}`　工具调用 {len(r['tools'])} 次　"
            f"步数 {r['steps']}　成本 {r['cost']:.6f} 元　延迟 {r['latency']}ms"
        )
        if r["reason"] and r["reason"] != "ok":
            lines.append(f"- 判定理由：{r['reason']}")
        lines.append("")
        lines.append("**执行轨迹**")
        lines.append("")
        for item in r["trace"]:
            lines.append(f"1. {item}")
        lines.append("")
        lines.append("**模型完整回答**")
        lines.append("")
        content = (r["content"] or "").strip() or "（空）"
        for para in content.splitlines():
            lines.append(f"> {para}")
        lines.append("")    
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入 {report}")

if __name__ == "__main__":
    main()
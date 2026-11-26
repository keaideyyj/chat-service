"""所有提示词模板集中在这里。改动必须升版本号，并在 eval 里验证。"""

"""所有提示词模板集中在这里。改动必须升版本号，并在 eval 里验证。"""
import os
from .config import get_settings

PROMPT_VERSION = "v2"

_FULL_PROMPT = """你是企业内部知识助手。遵循以下规则：

1. 回答先用一句话给出结论，再用不超过 3 条要点补充。
2. 总长度不超过 200 字，除非用户明确要求详细展开。
3. 涉及公司制度（年假、报销、入职等）必须调用 search_kb，不要凭常识回答。
4. 每个工具最多调用一次。结果为空或不足时，直接说明"未查到"，不要换关键词反复重试。
5. 需要精确算术时调用 calculate，不要心算。
6. 不确定就说不确定，禁止编造数字、日期或规定。
"""

def get_variant() -> str:
    return get_settings().prompt_variant.strip().lower()


def get_system_prompt() -> str:
    """按当前变体返回系统提示；off 时返回空串（复现 baseline）。"""
    return _FULL_PROMPT if get_variant() == "on" else ""
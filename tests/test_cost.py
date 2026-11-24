"""成本计算测试。

estimate_cost 是纯数学函数，最适合做单元测试：
输入确定、没有副作用、结果可精确验证。

为什么这类函数必须测：它算错了你不会报错，只会悄悄给出错的数字，
然后你基于错的数字做"成本优化"决策。
"""

import pytest

from app import llm


class _FakeSettings:
    """假的配置对象，只提供成本计算需要的那两个字段。

    为什么要造假：真实 Settings 依赖 .env 文件。测试如果依赖 .env，
    换台机器（或 CI）就红 —— 那测的就不是代码，而是环境。
    """
    input_price_per_1m = 2.0
    output_price_per_1m = 8.0


def test_estimate_cost(monkeypatch):
    """43 个输入 token + 42 个输出 token 应该算出 0.000422 元。

    手工验算：43*2 = 86，42*8 = 336，合计 422，再除以一百万。
    测试里保留这个算式，等于把"这个数字从哪来"永久记录下来。
    """
    # 把 llm 模块里的 get_settings 临时换成"返回假配置"的函数
    monkeypatch.setattr(llm, "get_settings", lambda: _FakeSettings())

    # 浮点数不能用 == 精确比较，approx 允许极小误差
    assert llm.estimate_cost(43, 42) == pytest.approx(0.000422)


def test_estimate_cost_is_zero_without_tokens(monkeypatch):
    """没有 token 时必须返回 0，不能是 None 或报错。

    为什么重要：流式场景下拿不到 usage 时，token 计数会停留在 0，
    这个函数必须能安然度过，而不是抛异常把整个响应搞挂。
    """
    monkeypatch.setattr(llm, "get_settings", lambda: _FakeSettings())
    assert llm.estimate_cost(0, 0) == 0.0
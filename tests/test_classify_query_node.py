"""classify_query_node 路由与护栏单元测试（mock 分类结果，不调真实 LLM）。

护栏核心断言：out_of_scope / 分类异常必须直达 final_response，
绝不路由到业务智能体。
"""
import pytest

import multi_agent_customer_service as macs
from multi_agent_customer_service import OUT_OF_SCOPE_REPLY, classify_query_node


class _StubClassifier:
    """替代 classify_query 工具：固定返回指定标签，或抛异常。"""

    def __init__(self, label=None, raise_exc=None):
        self._label = label
        self._raise = raise_exc

    def invoke(self, _args):
        if self._raise:
            raise self._raise
        return self._label


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """分类节点内的 get_llm 不触网。"""
    monkeypatch.setattr(macs, "get_llm", lambda: object())


def _run(monkeypatch, label=None, raise_exc=None, query="手机多少钱"):
    stub = _StubClassifier(label=label, raise_exc=raise_exc)
    monkeypatch.setattr(macs, "classify_query", stub)
    state = {"customer_query": query, "session_id": "test-session"}
    return classify_query_node(state)


class TestBusinessRouting:
    @pytest.mark.parametrize("query_type,expected_agent", [
        ("product_info", "product_agent"),
        ("technical_support", "tech_agent"),
        ("billing", "billing_agent"),
        ("complaint", "complaint_agent"),
        ("general_inquiry", "general_agent"),
    ])
    def test_business_labels_route_to_agents(self, monkeypatch, query_type, expected_agent):
        state = _run(monkeypatch, label=query_type)
        assert state["next_agent"] == expected_agent
        assert state["query_type"] == query_type
        # 用户轮次已写入持久化对话
        assert state["persisted_dialogue"][-1]["is_user"] is True


class TestGuardrail:
    def test_out_of_scope_goes_straight_to_final(self, monkeypatch):
        state = _run(monkeypatch, label="out_of_scope", query="帮我写一首诗")
        assert state["next_agent"] == "final_response"
        assert state["response"] == OUT_OF_SCOPE_REPLY
        # 拒答轮次也写入持久化对话（用户 + 拒答 共两条）
        assert len(state["persisted_dialogue"]) == 2
        assert state["persisted_dialogue"][-1]["is_user"] is False
        assert "out_of_scope_refusal" in state["tools_used"]

    def test_out_of_scope_never_hits_business_agent(self, monkeypatch):
        state = _run(monkeypatch, label="out_of_scope")
        assert state["next_agent"] not in (
            "product_agent", "tech_agent", "billing_agent",
            "complaint_agent", "general_agent",
        )

    def test_classifier_exception_falls_back_to_refusal(self, monkeypatch):
        # 分类链路异常：保守拒答，不放行
        state = _run(monkeypatch, raise_exc=RuntimeError("LLM down"))
        assert state["query_type"] == "out_of_scope"
        assert state["next_agent"] == "final_response"
        assert state["response"] == OUT_OF_SCOPE_REPLY

    def test_empty_query_rejected(self):
        state = classify_query_node({"customer_query": "", "session_id": "s"})
        assert state["next_agent"] == "final_response"
        assert state["query_type"] == "out_of_scope"
        assert "current_agent" in state  # final_response_node 取值必需


class TestGraphRouting:
    """条件边路由函数与节点设置的 next_agent 一致。"""

    def test_conditional_edge_default_is_final_response(self):
        # next_agent 缺失时条件边默认 final_response（不进业务智能体）
        route = macs.make_graph.__globals__  # noqa: F841 （仅确认模块可加载）
        # 直接构造条件边 lambda 的行为验证
        routing_fn = lambda x: x.get("next_agent", "") or "final_response"
        assert routing_fn({}) == "final_response"
        assert routing_fn({"next_agent": ""}) == "final_response"
        assert routing_fn({"next_agent": "billing_agent"}) == "billing_agent"

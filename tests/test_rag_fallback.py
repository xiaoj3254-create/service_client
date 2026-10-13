"""RAG 检索降级单元测试：RAG 不可用时必须无缝回退到硬编码 _match_data。"""
import pytest

from multi_agents.base_agent import BaseAgent
from multi_agents.product_agent import ProductAgent


class _MinimalAgent(BaseAgent):
    """用于测试的具体子类：固定 _match_data 返回值。"""
    domain = "test"

    def _get_system_prompt(self):
        return "test"

    def _match_data(self, query):
        return f"HARDCODED:{query}"


class TestRetrieveKnowledgeFallback:
    def test_falls_back_when_rag_unavailable(self, monkeypatch):
        # rag_tools 初始化失败（如未构建知识库）→ retrieve 返回空串 → 回退硬编码
        import tools.rag_tools as rt
        monkeypatch.setattr(rt, "_get_collection", lambda: None)
        agent = _MinimalAgent(name="t", role="t", expertise=[])
        assert agent._retrieve_knowledge("手机多少钱") == "HARDCODED:手机多少钱"

    def test_falls_back_when_rag_empty_collection(self, monkeypatch):
        import tools.rag_tools as rt

        class _EmptyColl:
            def count(self):
                return 0

        monkeypatch.setattr(rt, "_get_collection", lambda: _EmptyColl())
        agent = _MinimalAgent(name="t", role="t", expertise=[])
        assert agent._retrieve_knowledge("q") == "HARDCODED:q"

    def test_falls_back_on_rag_exception(self, monkeypatch):
        import tools.rag_tools as rt

        def _boom(*_a, **_kw):
            raise RuntimeError("chroma down")

        monkeypatch.setattr(rt, "retrieve", _boom)
        agent = _MinimalAgent(name="t", role="t", expertise=[])
        assert agent._retrieve_knowledge("q") == "HARDCODED:q"

    def test_uses_rag_when_available(self, monkeypatch):
        import tools.rag_tools as rt
        monkeypatch.setattr(rt, "retrieve", lambda q, d, k=3: f"RAG:{d}:{q}")
        agent = _MinimalAgent(name="t", role="t", expertise=[])
        assert agent._retrieve_knowledge("q") == "RAG:test:q"

    def test_product_agent_domain(self):
        assert ProductAgent.domain == "product"

    def test_retrieve_skips_empty_query(self, monkeypatch):
        # 空查询不触发检索，直接回退
        called = {"n": 0}
        import tools.rag_tools as rt

        def _spy(q, d, k=3):
            called["n"] += 1
            return ""

        monkeypatch.setattr(rt, "retrieve", _spy)
        agent = _MinimalAgent(name="t", role="t", expertise=[])
        assert agent._retrieve_knowledge("  ") == "HARDCODED:  "

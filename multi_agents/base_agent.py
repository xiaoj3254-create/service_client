"""
基础智能体类
所有专门智能体的基类
"""

import logging
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    # RAG 领域标识（子类覆盖），对应 knowledge/<domain>/ 目录与 Chroma 元数据过滤
    domain: str = ""

    def __init__(self, name: str, role: str, expertise: List[str]):
        self.name = name
        self.role = role
        self.expertise = expertise
        self.llm = None  # 将在运行时注入

    def set_llm(self, llm):
        """设置LLM客户端"""
        self.llm = llm

    def _build_human_message(self, text: str, state: Optional[Dict[str, Any]] = None) -> HumanMessage:
        """
        构造用户消息：若 state 中含客户上传的图片列表（customer_images，data URL），
        则构造 OpenAI 兼容的多模态消息（text + 多个 image_url），否则为纯文本消息。
        """
        images = (state or {}).get("customer_images")
        if images:
            content_parts = [{"type": "text", "text": str(text)}]
            for img in images:
                content_parts.append({"type": "image_url", "image_url": {"url": img}})
            return HumanMessage(content=content_parts)
        return HumanMessage(content=text)

    # ------------------------------------------------------------------
    # 子类必须实现的抽象方法
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """返回该智能体的基础系统提示词（不含对话上下文增强）。"""

    @abstractmethod
    def _match_data(self, query: str) -> str:
        """根据用户查询匹配领域知识库（硬编码兜底），返回格式化的上下文文本；无匹配时返回空串。"""

    # ------------------------------------------------------------------
    # 可选覆盖的方法
    # ------------------------------------------------------------------

    def _get_error_fallback(self) -> str:
        """LLM 调用失败时的兜底回复，子类可覆盖以提供更贴切的文案。"""
        return "抱歉，处理您的请求时遇到系统错误，请稍后重试。"

    # ------------------------------------------------------------------
    # 公共流程模板（所有领域 agent 共用）
    # ------------------------------------------------------------------

    def _retrieve_knowledge(self, query: str) -> str:
        """获取领域知识：RAG 向量检索优先，不可用/无命中时回退子类硬编码匹配。"""
        try:
            from tools.rag_tools import retrieve
            rag_result = retrieve(query, self.domain)
            if rag_result:
                return rag_result
        except Exception as e:
            # rag_tools 导入失败等异常同样降级，不阻断对话
            logger.debug("%s RAG 检索异常，回退硬编码匹配: %s", self.name, e)
        return self._match_data(query)

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理客户查询的模板方法。
        公共流程：取查询 -> 取对话上下文 -> 获取领域知识（RAG 优先）-> 构建消息 -> 调 LLM -> 写 state。
        子类只需实现 _get_system_prompt / _match_data（兜底），可覆盖 _get_error_fallback。
        """
        customer_query = state["customer_query"]
        session_id = state.get("session_id", "default")

        # 对话轮次由 classify / 外层节点写入 persisted_dialogue，此处只读 state
        conversation_context = self._get_conversation_context(session_id, state)

        # 从领域知识库匹配相关信息（RAG 优先，硬编码兜底）
        matched_info = self._retrieve_knowledge(customer_query)

        # 构建系统提示并增强对话上下文说明
        base_system_prompt = self._get_system_prompt()
        system_prompt = self._enhance_system_prompt_with_context(base_system_prompt)

        # 构建消息列表
        messages = []

        # 添加对话历史上下文（如果有的话）
        if conversation_context:
            context_message = f"""对话历史上下文：
{conversation_context}

请基于以上对话历史和当前查询，提供连贯的回答。"""
            messages.append(SystemMessage(content=context_message))

        # 添加系统提示
        messages.append(SystemMessage(content=system_prompt))

        # 如果有匹配的领域信息，添加到上下文中（含客户上传图片时构造多模态消息）
        if matched_info:
            data_context = f"""领域信息：
{matched_info}

当前查询：{customer_query}"""
            messages.append(self._build_human_message(data_context, state))
        else:
            messages.append(self._build_human_message(customer_query, state))

        # 调用LLM（流式优先，不可用/失败自动回退非流式）
        response_content = self._call_llm(messages)

        state["response"] = response_content
        state["current_agent"] = self.name
        state["tools_used"].append(f"{self.name}_processing")

        return state

    def _call_llm(self, messages) -> str:
        """
        调用 LLM 获取回复文本，流式优先。

        处于 LangGraph 流式上下文（以 stream_mode="custom" 运行）时，用 get_stream_writer()
        拿到的 writer 逐个 token 推给前端，实现打字机效果；同时累积完整文本返回，
        保证 state["response"] / persisted_dialogue / 侧栏预览与非流式路径完全一致。

        以下任一情况都自动回退到原有阻塞 invoke()，行为与改动前一致：
        - 非流式上下文（独立脚本运行、直接调用 process）
        - 流式调用在产出任何内容之前就失败（已产出部分内容时绝不重试，防止重复文本）
        """
        writer = None
        try:
            # 放在函数内导入：老版本 langgraph 无此接口时不影响模块导入
            from langgraph.config import get_stream_writer
            writer = get_stream_writer()
        except Exception:
            writer = None

        if writer is not None and self.llm is not None:
            collected: List[str] = []
            try:
                # ChatOpenAI 原生 stream()：逐 chunk 产出 AIMessageChunk
                for chunk in self.llm.stream(messages):
                    piece = getattr(chunk, "content", None)
                    if not piece:
                        continue
                    collected.append(piece)
                    try:
                        writer({"type": "token", "content": piece})
                    except Exception:
                        # 推流失败不影响正文生成，静默降级为纯累积
                        logger.debug("%s 推流失败，降级为纯累积模式", self.name)
                if collected:
                    return "".join(collected)
                logger.warning("%s 流式调用未产出任何内容，回退非流式", self.name)
            except Exception as e:
                if collected:
                    # 已产出部分内容：绝不回退重试，否则前端会看到重复文本
                    logger.warning(
                        "%s 流式中途失败（已产出 %d 块），返回已累积内容: %s",
                        self.name, len(collected), e,
                    )
                    return "".join(collected)
                logger.warning("%s 流式调用失败，回退非流式: %s", self.name, e)

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.warning("%s 调用LLM时出错: %s", self.name, e)
            return self._get_error_fallback()

    def _get_conversation_context(
        self,
        session_id: str,
        state: Optional[Dict[str, Any]] = None,
        max_messages: int = 12,
    ) -> str:
        """从图状态 persisted_dialogue 读取对话上下文（由 checkpointer 持久化，跨进程有效）。"""
        records = (state or {}).get("persisted_dialogue")
        if not records:
            return ""
        trimmed = list(records)
        # classify 节点在路由前已把当前用户轮次写入 persisted_dialogue，
        # 而当前查询随后会作为独立问题发送给 LLM；剔除末尾的当前轮次，
        # 避免同一问题在 prompt 中出现两遍
        if trimmed and trimmed[-1].get("is_user", True):
            trimmed = trimmed[:-1]
        if not trimmed:
            return ""
        tail = trimmed[-max_messages:]
        context_lines = []
        for msg in tail:
            role = "用户" if msg.get("is_user", True) else "AI"
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
            context_lines.append(f"[{timestamp}] {role}: {content}")
        return "\n".join(context_lines)

    def _enhance_system_prompt_with_context(self, base_prompt: str) -> str:
        """增强系统提示，添加对话上下文说明"""
        context_instruction = """

重要：请结合对话历史上下文，理解客户之前的问题和需求，提供连贯、个性化的回答。
如果这是多轮对话，请参考之前的对话内容，避免重复信息，并基于客户的新问题提供补充信息。
保持对话的连贯性和自然性，让客户感受到你理解他们的完整需求。"""

        return base_prompt + context_instruction

    def get_info(self) -> Dict[str, Any]:
        """获取智能体信息"""
        return {
            "name": self.name,
            "role": self.role,
            "expertise": self.expertise
        }

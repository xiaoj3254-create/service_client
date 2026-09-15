"""
基础智能体类
所有专门智能体的基类
"""

import logging
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from langchain_core.messages import HumanMessage, SystemMessage
from session_manager import LangChainSessionManager

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    def __init__(self, name: str, role: str, expertise: List[str], session_manager: LangChainSessionManager = None):
        self.name = name
        self.role = role
        self.expertise = expertise
        self.llm = None  # 将在运行时注入
        self.session_manager = session_manager or LangChainSessionManager()

    def set_llm(self, llm):
        """设置LLM客户端"""
        self.llm = llm

    def set_session_manager(self, session_manager: LangChainSessionManager):
        """设置会话管理器"""
        self.session_manager = session_manager

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
        """根据用户查询匹配领域知识库，返回格式化的上下文文本；无匹配时返回空串。"""

    # ------------------------------------------------------------------
    # 可选覆盖的方法
    # ------------------------------------------------------------------

    def _get_error_fallback(self) -> str:
        """LLM 调用失败时的兜底回复，子类可覆盖以提供更贴切的文案。"""
        return "抱歉，处理您的请求时遇到系统错误，请稍后重试。"

    # ------------------------------------------------------------------
    # 公共流程模板（所有领域 agent 共用）
    # ------------------------------------------------------------------

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理客户查询的模板方法。
        公共流程：取查询 -> 取对话上下文 -> 匹配领域数据 -> 构建消息 -> 调 LLM -> 写 state。
        子类只需实现 _get_system_prompt / _match_data，可覆盖 _get_error_fallback。
        """
        customer_query = state["customer_query"]
        session_id = state.get("session_id", "default")

        # 对话轮次由 classify / 外层节点写入 persisted_dialogue，此处只读 state
        conversation_context = self._get_conversation_context(session_id, state)

        # 从领域知识库匹配相关信息
        matched_info = self._match_data(customer_query)

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

        # 调用LLM
        try:
            response = self.llm.invoke(messages)
            response_content = response.content
        except Exception as e:
            logger.warning("%s 调用LLM时出错: %s", self.name, e)
            response_content = self._get_error_fallback()

        state["response"] = response_content
        state["current_agent"] = self.name
        state["tools_used"].append(f"{self.name}_processing")

        return state

    def _get_conversation_context(
        self,
        session_id: str,
        state: Optional[Dict[str, Any]] = None,
        max_messages: int = 12,
    ) -> str:
        """优先使用图状态中持久化的对话（跨 LangGraph 进程/工作有效），否则回退到 session_manager。"""
        if state is not None:
            records = state.get("persisted_dialogue")
            if records:
                tail = records[-max_messages:]
                context_lines = []
                for msg in tail:
                    role = "用户" if msg.get("is_user", True) else "AI"
                    content = msg.get("content", "")
                    timestamp = msg.get("timestamp", "")
                    context_lines.append(f"[{timestamp}] {role}: {content}")
                return "\n".join(context_lines)
        try:
            conversation_context = self.session_manager.get_conversation_context(session_id, max_messages)

            if not conversation_context:
                return ""

            # 格式化对话历史
            context_lines = []
            for msg in conversation_context:
                role = "用户" if msg.get("is_user", True) else "AI"
                content = msg.get("content", "")
                timestamp = msg.get("timestamp", "")
                context_lines.append(f"[{timestamp}] {role}: {content}")

            return "\n".join(context_lines)
        except Exception as e:
            logger.warning("获取对话上下文时出错: %s", e)
            return ""

    def _add_message_to_session(self, session_id: str, message: str, is_user: bool = True):
        """添加消息到会话历史"""
        try:
            self.session_manager.add_message(session_id, message, is_user)
        except Exception as e:
            logger.warning("添加消息到会话时出错: %s", e)

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

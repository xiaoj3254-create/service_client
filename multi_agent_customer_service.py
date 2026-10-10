"""
多智能体客服系统
使用LangGraph构建，包含多个专门的智能体来处理不同类型的客户查询
基于OpenAI兼容API提供LLM能力
支持多轮对话和会话管理
"""

import logging
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.config import get_config

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)

# 导入配置
from config import *

# 导入智能体和工具
from multi_agents import (
    ProductAgent, TechAgent, BillingAgent,
    ComplaintAgent, GeneralAgent
)
from tools import classify_query

# 超出客服范围时的固定回复（护栏：不调用业务智能体）
OUT_OF_SCOPE_REPLY = (
    "抱歉，这里是智能客服，仅处理与产品、技术、账单、投诉及相关售后政策类问题；"
    "请用一句话说明您的具体业务诉求，我很乐意协助。"
)

# 定义状态类型
class AgentState(TypedDict):
    session_id: str
    messages: List[Any]
    current_agent: str
    customer_query: str
    query_type: str
    response: str
    tools_used: List[str]
    next_agent: str
    conversation_history: List[Any]
    # 由图 checkpointer 持久化，跨 LangGraph 工作进程仍可续聊
    persisted_dialogue: List[Any]
    # 客户上传的图片列表（data URL，如 "data:image/jpeg;base64,..."）；无图时为 None
    customer_images: Optional[List[str]]

# 延迟初始化LLM
_llm_instance = None

def get_llm():
    """获取LLM实例（langchain-openai ChatOpenAI），延迟初始化。

    使用标准客户端替代此前的手写 HTTP 实现，原生获得：
    - system 角色正确透传（此前被降级为 user，削弱指令遵从与护栏强度）
    - 流式 / 重试 / 多模态（image_url 消息为 OpenAI 原生格式）
    - 与 LangChain 生态（with_structured_output 等）的完整兼容
    """
    global _llm_instance
    if _llm_instance is None:
        if not OPENAI_API_KEY:
            logger.error("API密钥未设置，无法初始化LLM")
            return None
        try:
            _llm_instance = ChatOpenAI(
                model=OPENAI_MODEL,
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                timeout=HTTP_TIMEOUT,
                max_retries=HTTP_MAX_RETRIES,
            )
            logger.info("成功初始化 ChatOpenAI 客户端: model=%s base_url=%s", OPENAI_MODEL, OPENAI_BASE_URL)
        except Exception:
            logger.exception("初始化 ChatOpenAI 客户端失败")
            _llm_instance = None
    return _llm_instance

# 智能体类映射（按需实例化：一次路由只创建实际需要的那一个智能体）
_AGENT_CLASSES = {
    "product_agent": ProductAgent,
    "tech_agent": TechAgent,
    "billing_agent": BillingAgent,
    "complaint_agent": ComplaintAgent,
    "general_agent": GeneralAgent,
}

# 已实例化的智能体缓存（同进程多次对话复用，避免重复创建）
_agent_instances: Dict[str, Any] = {}

def get_agent(agent_name: str):
    """按需获取智能体实例（首次调用时创建并缓存，后续复用）。

    LLM 未就绪（如 API 密钥缺失导致 get_llm() 返回 None）时不缓存实例，
    待密钥配置好后下次调用会重新创建，避免 None-LLM 实例被永久固化。
    """
    if agent_name in _agent_instances:
        return _agent_instances[agent_name]
    cls = _AGENT_CLASSES.get(agent_name)
    if cls is None:
        return None
    agent = cls()
    agent.set_llm(get_llm())
    if agent.llm is None:
        logger.warning("智能体 %s 的 LLM 未就绪（密钥缺失或初始化失败），本次不缓存实例", agent_name)
        return agent
    _agent_instances[agent_name] = agent
    return agent

def _push_stream_status(text: str) -> None:
    """
    向 LangGraph custom 流通道推送一条阶段性状态，供前端在等待期展示进度。

    仅在流式上下文（平台/CLI 以 stream_mode="custom" 运行）中生效；
    非流式运行（独立脚本、直接调用节点）会取不到 writer，静默忽略。
    """
    try:
        from langgraph.config import get_stream_writer
        get_stream_writer()({"type": "status", "content": text})
    except Exception:
        pass


# 定义查询分类节点
def classify_query_node(state: AgentState) -> AgentState:
    """Classify customer query"""
    # 分类必须先阻塞调用一次 LLM 才能确定路由，用户侧会有十几秒空白，先给个进度反馈
    _push_stream_status("正在识别您的问题类型…")

    try:
        cfg = get_config()  # 当前 graph 运行时的 `RunnableConfig` 对象
        tid = (cfg.get("configurable") or {}).get("thread_id")
        if tid:
            state["session_id"] = str(tid)
    except RuntimeError:
        pass

    # 初始化状态对象
    if "session_id" not in state or not state.get("session_id"):
        import uuid
        state["session_id"] = str(uuid.uuid4())

    if "tools_used" not in state:
        state["tools_used"] = []

    if "conversation_history" not in state:
        state["conversation_history"] = []

    if "persisted_dialogue" not in state or state.get("persisted_dialogue") is None:
        state["persisted_dialogue"] = []

    if "next_agent" not in state:
        state["next_agent"] = ""

    if "messages" not in state:
        state["messages"] = []

    # 客户图片列表（data URL）；由 run input 传入，无图时为 None
    if "customer_images" not in state:
        state["customer_images"] = None

    # 获取必需字段
    customer_query = state.get("customer_query", "")
    session_id = state["session_id"]

    if not customer_query:
        state["response"] = "Error: No customer query provided"
        state["query_type"] = "out_of_scope"
        # 必须显式设置 next_agent：否则条件边会取默认值 general_agent，绕过护栏
        state["next_agent"] = "final_response"
        # 补齐 current_agent，否则 final_response_node 取值会 KeyError 导致整个 run 失败
        state["current_agent"] = "智能客服"
        return state

    # 使用分类工具
    # 兜底一律取 out_of_scope（保守拒答），绝不默认路由到业务智能体：
    # 分类链路异常时无法判断是否越界，放行存在护栏绕过风险。
    query_type = "out_of_scope"
    try:
        llm_instance = get_llm()
        # 使用正确的工具调用方式
        try:
            result = classify_query.invoke({"query": customer_query, "llm": llm_instance})
            query_type = result
        except Exception as e:
            logger.warning("分类工具调用失败，回退到保守兜底: %s", e)
            # 回退到保守兜底
            query_type = "out_of_scope"
    except Exception as e:
        logger.exception("查询分类出错，回退到保守兜底")
        query_type = "out_of_scope"

    # 更新状态
    state["query_type"] = query_type
    state["tools_used"].append("query_classification")

    # 根据 query_type 设置 next_agent（条件边的路由依据）
    QUERY_TYPE_TO_AGENT = {
        "product_info": "product_agent",
        "technical_support": "tech_agent",
        "billing": "billing_agent",
        "complaint": "complaint_agent",
        "general_inquiry": "general_agent",
    }
    state["next_agent"] = QUERY_TYPE_TO_AGENT.get(query_type, "")

    # 写入由 checkpointer 持久化的对话（用户轮次；含图片则一并存入，供历史回显）
    pd = list(state.get("persisted_dialogue") or [])
    user_turn = {
        "content": str(customer_query),
        "is_user": True,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    images = state.get("customer_images")
    if images:
        user_turn["images"] = images
    pd.append(user_turn)
    state["persisted_dialogue"] = pd

    # 护栏：超出范围直接固定回复，不进入业务智能体
    if state["query_type"] == "out_of_scope":
        state["response"] = OUT_OF_SCOPE_REPLY
        state["current_agent"] = "智能客服"
        state["next_agent"] = "final_response"
        pd_oos = list(state.get("persisted_dialogue") or [])
        pd_oos.append({
            "content": OUT_OF_SCOPE_REPLY,
            "is_user": False,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        state["persisted_dialogue"] = pd_oos
        state["tools_used"].append("out_of_scope_refusal")
        return state

    return state

# 定义智能体处理节点
def create_agent_node(agent_name: str):
    """创建智能体处理节点"""
    def agent_node(state: AgentState) -> AgentState:
        agent = get_agent(agent_name)
        if agent:
            # 获取会话上下文
            state["conversation_history"] = list(state.get("persisted_dialogue") or [])

            # 处理查询
            result = agent.process(state)

            if not isinstance(result, dict):
                logger.error("Agent %s 返回非dict结果: %s", agent_name, type(result))
                result = {"response": "Error: Agent processing failed", "current_agent": agent_name}

            if "response" not in result:
                logger.error("Agent %s 结果缺少response字段", agent_name)
                result["response"] = "Error: No response from agent"

            # 助手轮次写入 checkpointer 状态
            pd = list(result.get("persisted_dialogue") or [])
            pd.append({
                "content": str(result["response"]),
                "is_user": False,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            result["persisted_dialogue"] = pd

            return result
        else:
            state["response"] = f"Error: Agent {agent_name} not found"
            # 补齐 current_agent，否则 final_response_node 取值会 KeyError 导致整个 run 失败
            state["current_agent"] = agent_name
            return state
    return agent_node

# 定义最终响应节点
def final_response_node(state: AgentState) -> AgentState:
    """Generate final response"""
    current_agent = state["current_agent"]
    response = state["response"]

    state["response"] = f"【{current_agent}'s Response】\n{response}"
    # 路由已完成，清空 next_agent 避免残留上一个智能体名称
    state["next_agent"] = ""
    return state

# 默认 checkpointer：进程内持久化，无外部依赖。
# 显式挂载它，使 persisted_dialogue 的持久化语义在「独立运行」与「平台托管」两种
# 模式下都成立（此前 compile() 未传 checkpointer，独立运行时多轮对话无记忆）。
_default_checkpointer = None


def get_default_checkpointer():
    """惰性创建并复用进程级默认 checkpointer。"""
    global _default_checkpointer
    if _default_checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver
        _default_checkpointer = InMemorySaver()
        logger.info("已启用默认 InMemorySaver checkpointer（进程内持久化）")
    return _default_checkpointer


# 图表入口点
# 使用方式：在langgraph.json文件中增加以下配置，声明构建图的方式，硬编码方式实现。
# "graphs": {
#     "customer_service": "./multi_agent_customer_service.py:make_graph"
# },
# 也可以在langgraph.json文件中使用workflow配置化的方式定义图的结构，但功能相对简单，无法实现复杂的逻辑
def make_graph(checkpointer=None):
    """
    构建LangGraph工作流图。

    Args:
        checkpointer: 可选的 checkpointer 实例。
            - 由 LangGraph Platform / CLI 托管时，平台会自动注入服务端 checkpointer，
              此时无需传入，图会交由平台管理持久化。
            - 独立运行（如 `python multi_agent_customer_service.py`、自写脚本）时
              传 None，将自动使用进程内 InMemorySaver，保证多轮对话记忆可用。
            - 需要跨进程/持久化到磁盘时，可传入 SqliteSaver / PostgresSaver 等实例。
    """
    # 创建工作流图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("classify_query", classify_query_node)
    workflow.add_node("product_agent", create_agent_node("product_agent"))
    workflow.add_node("tech_agent", create_agent_node("tech_agent"))
    workflow.add_node("billing_agent", create_agent_node("billing_agent"))
    workflow.add_node("complaint_agent", create_agent_node("complaint_agent"))
    workflow.add_node("general_agent", create_agent_node("general_agent"))
    workflow.add_node("final_response", final_response_node)

    # 设置入口点
    workflow.set_entry_point("classify_query")

    # classify_query → 各智能体：根据 next_agent 路由（next_agent 在分类节点内设置）
    workflow.add_conditional_edges(
        "classify_query",
        # 路由异常时保守拒答（与护栏哲学一致：路由不了就拒答，不放行业务智能体）
        lambda x: x.get("next_agent", "") or "final_response",
        {
            "product_agent": "product_agent",
            "tech_agent": "tech_agent",
            "billing_agent": "billing_agent",
            "complaint_agent": "complaint_agent",
            "general_agent": "general_agent",
            "final_response": "final_response",
        }
    )

    # 添加直接边（所有智能体都连接到最终响应）
    workflow.add_edge("product_agent", "final_response")
    workflow.add_edge("tech_agent", "final_response")
    workflow.add_edge("billing_agent", "final_response")
    workflow.add_edge("complaint_agent", "final_response")
    workflow.add_edge("general_agent", "final_response")

    # 设置结束点
    workflow.set_finish_point("final_response")

    # 编译工作流：显式挂载 checkpointer，保证 persisted_dialogue 真正被持久化
    # - 独立运行（python multi_agent_customer_service.py）：checkpointer=None，使用进程内 InMemorySaver
    # - 平台托管（langgraph dev/CLI）：平台会注入 dict 形式的配置描述，
    #   此时不能传给 compile()，应交由平台自行管理 checkpointer
    if checkpointer is None:
        checkpointer = get_default_checkpointer()
    elif isinstance(checkpointer, dict):
        # 平台注入的配置 dict，非合法 saver 实例，传 None 让平台托管
        logger.info("检测到平台注入的 checkpointer 配置，交由 LangGraph 平台托管持久化")
        checkpointer = None
    app = workflow.compile(checkpointer=checkpointer)

    logger.info("LangGraph工作流图构建完成（已挂载 checkpointer）")
    return app

# 创建默认工作流实例
if __name__ == "__main__":
    app = make_graph()
    logger.info("多智能体客服系统启动成功！")

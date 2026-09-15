"""
投诉处理专家智能体
专门负责客户投诉和建议处理
"""

from .base_agent import BaseAgent

class ComplaintAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="投诉处理专家",
            role="客户投诉和建议处理",
            expertise=["问题记录", "解决方案", "补偿措施", "服务改进"]
        )

        # TODO: 投诉处理信息应该从客服系统获取，这里只是模拟数据
        # 实际应用中应该连接客服数据库或调用客服API服务
        self.complaint_database = {
            "服务问题": {
                "响应速度慢": "承诺24小时内响应，超时提供补偿",
                "服务态度差": "记录问题，安排专人跟进，提供道歉补偿",
                "专业能力不足": "安排专业培训，提供专家支持",
                "处理流程": "记录问题 → 分析原因 → 制定方案 → 执行解决 → 回访确认"
            },
            "产品质量": {
                "功能缺陷": "提供免费维修或更换，延长保修期",
                "外观瑕疵": "提供更换或折扣补偿",
                "性能不达标": "技术检测确认后，提供升级或退款",
                "补偿标准": "根据问题严重程度，提供10%-100%的补偿"
            },
            "物流配送": {
                "配送延迟": "超时提供运费补偿，加急配送",
                "包装破损": "拍照记录，提供更换或补偿",
                "配送错误": "免费重新配送，提供额外补偿",
                "紧急处理": "24小时内响应，48小时内解决"
            }
        }

    def _get_system_prompt(self) -> str:
        return f"""你是{self.name}，专门负责{self.role}。
        你的专业领域包括：{', '.join(self.expertise)}

        请以专业、耐心的态度处理客户投诉：
        1. 认真倾听客户的问题和不满
        2. 表达理解和歉意
        3. 提供具体的解决方案和时间承诺
        4. 如果问题复杂，说明后续处理流程

        回答要真诚、专业，体现对客户的重视。如果投诉超出你的处理权限，请说明并承诺转交给相关部门处理。"""

    def _get_error_fallback(self) -> str:
        return "抱歉，处理您的投诉时遇到系统错误，请稍后重试。"

    def _match_data(self, query: str) -> str:
        """匹配查询中的投诉信息。

        匹配策略（按优先级）：
        1. 类别名精确命中：query 含 "服务问题"/"产品质量"/"物流配送" 时，返回该类别全部条目。
        2. 具体条目名命中：query 含 "配送延迟"/"功能缺陷" 等条目名时，返回所属类别中命中的条目。
        3. 模糊兜底：出现投诉类关键词时，**只返回最相关的一个类别**的前 2 条，
           避免把整个知识库塞进 prompt。
        """
        query_lower = query.lower()
        matched_info = []
        matched_categories = set()

        # 1. 类别名精确命中
        for category, solutions in self.complaint_database.items():
            if any(keyword in query_lower for keyword in category.lower().split()):
                info_text = f"【{category}】\n"
                for issue, solution in solutions.items():
                    info_text += f"• {issue}：{solution}\n"
                matched_info.append(info_text)
                matched_categories.add(category)

        # 2. 具体条目名命中（补足类别名未命中的场景，如只说了"配送延迟"）
        if not matched_info:
            for category, solutions in self.complaint_database.items():
                hit_items = [
                    (issue, solution)
                    for issue, solution in solutions.items()
                    if issue in query_lower
                ]
                if hit_items:
                    info_text = f"【{category}】\n"
                    for issue, solution in hit_items:
                        info_text += f"• {issue}：{solution}\n"
                    matched_info.append(info_text)
                    matched_categories.add(category)

        # 3. 模糊兜底：关键词判断在循环外，且只取一个类别
        if not matched_info and any(
            keyword in query_lower
            for keyword in ["投诉", "问题", "建议", "不满", "改进", "反馈"]
        ):
            for category, solutions in self.complaint_database.items():
                if category in matched_categories:
                    continue
                info_text = f"相关处理：{category}\n"
                # 只显示前2项解决方案
                for i, (issue, solution) in enumerate(solutions.items()):
                    if i < 2:
                        info_text += f"• {issue}：{solution}\n"
                info_text += "..."
                matched_info.append(info_text)
                matched_categories.add(category)
                # 兜底只取第一个命中的类别即停止，避免全量注入
                break

        return "\n".join(matched_info) if matched_info else ""

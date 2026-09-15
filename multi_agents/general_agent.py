"""
综合客服智能体
专门负责一般咨询处理
"""

from .base_agent import BaseAgent

class GeneralAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="综合客服",
            role="一般咨询处理",
            expertise=["信息查询", "基础服务", "问题转接"]
        )

        # TODO: 服务信息应该从客服系统获取，这里只是模拟数据
        # 实际应用中应该连接客服数据库或调用客服API服务
        self.service_database = {
            "营业时间": {
                "在线客服": "7×24小时在线服务",
                "电话客服": "周一至周日 9:00-21:00",
                "门店服务": "周一至周日 10:00-22:00",
                "节假日安排": "节假日期间服务时间可能调整，请关注公告"
            },
            "联系方式": {
                "客服热线": "400-123-4567",
                "在线客服": "官网右下角在线聊天",
                "邮箱支持": "support@company.com",
                "微信客服": "关注公众号，点击在线客服"
            },
            "常见服务": {
                "订单查询": "提供订单号或手机号即可查询",
                "物流跟踪": "支持实时物流信息查询",
                "会员服务": "积分查询、等级升级、专属优惠",
                "售后服务": "7天无理由退货，30天质量问题换货"
            }
        }

    def _get_system_prompt(self) -> str:
        return f"""你是{self.name}，专门负责{self.role}。
        你的专业领域包括：{', '.join(self.expertise)}

        请以友好、专业的态度处理客户的一般咨询：
        1. 耐心倾听客户的问题
        2. 提供准确、有用的信息
        3. 如果问题超出你的专业范围，建议转接给相关专家
        4. 确保客户得到满意的答复

        回答要友好、专业，体现良好的服务态度。如果问题复杂或需要专业知识，请说明并建议转接给相应的专业智能体。"""

    def _get_error_fallback(self) -> str:
        return "抱歉，处理您的咨询时遇到系统错误，请稍后重试。"

    def _match_data(self, query: str) -> str:
        """匹配查询中的服务信息"""
        query_lower = query.lower()
        matched_info = []
        matched_categories = set()

        # 精确匹配服务类型
        for category, services in self.service_database.items():
            if any(keyword in query_lower for keyword in category.lower().split()):
                # 格式化服务信息
                info_text = f"""【{category}】\n"""
                for service, description in services.items():
                    info_text += f"• {service}：{description}\n"
                matched_info.append(info_text)
                matched_categories.add(category)

        # 如果没有精确匹配，尝试关键词匹配
        if not matched_info:
            for category, services in self.service_database.items():
                if category in matched_categories:
                    continue
                if any(keyword in query_lower for keyword in ["时间", "联系", "服务", "营业", "电话", "邮箱"]):
                    info_text = f"""相关信息：{category}\n"""
                    # 只显示前2项服务
                    for i, (service, description) in enumerate(services.items()):
                        if i < 2:
                            info_text += f"• {service}：{description}\n"
                    info_text += "..."
                    matched_info.append(info_text)
                    matched_categories.add(category)

        return "\n".join(matched_info) if matched_info else ""

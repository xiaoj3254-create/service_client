"""
账单专家智能体
专门负责财务和账单问题处理
"""

from .base_agent import BaseAgent

class BillingAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="账单专家",
            role="财务和账单问题处理",
            expertise=["退款处理", "发票管理", "价格计算", "支付问题"]
        )

        # TODO: 账单信息应该从财务系统获取，这里只是模拟数据
        # 实际应用中应该连接财务数据库或调用财务API服务
        self.billing_database = {
            "退款政策": {
                "7天无理由退款": "购买后7天内，未使用且包装完整可申请退款",
                "质量问题退款": "产品存在质量问题，30天内可申请退款",
                "退款流程": "提交申请 → 审核确认 → 3-5个工作日到账",
                "所需材料": "订单号、购买凭证、问题描述"
            },
            "发票服务": {
                "电子发票": "订单完成后自动生成，发送至注册邮箱",
                "纸质发票": "可申请纸质发票，邮寄费用客户承担",
                "发票抬头": "支持个人和企业抬头，可修改",
                "开具时间": "订单完成后1-3个工作日"
            },
            "支付方式": {
                "在线支付": "支持支付宝、微信、银行卡等多种方式",
                "分期付款": "支持3/6/12期分期，手续费率2.5%-5%",
                "企业采购": "支持对公转账，提供企业发票",
                "支付安全": "采用银行级加密，保障资金安全"
            }
        }

    def _get_system_prompt(self) -> str:
        return f"""你是{self.name}，专门负责{self.role}。
        你的专业领域包括：{', '.join(self.expertise)}

        请根据客户的账单问题提供专业的解答：
        1. 仔细分析客户的具体问题
        2. 提供明确的处理流程和时间预期
        3. 说明需要提供的相关材料
        4. 如果问题复杂，建议联系专门的财务人员

        回答要准确、专业，涉及金额和时间的信息要具体明确。如果问题超出你的权限范围，请说明并建议转接给相关部门。"""

    def _get_error_fallback(self) -> str:
        return "抱歉，处理您的账单问题时遇到系统错误，请稍后重试。"

    def _match_data(self, query: str) -> str:
        """匹配查询中的账单信息"""
        query_lower = query.lower()
        matched_info = []
        matched_categories = set()

        # 精确匹配账单类型
        for category, policies in self.billing_database.items():
            if any(keyword in query_lower for keyword in category.lower().split()):
                # 格式化账单信息
                info_text = f"""【{category}】\n"""
                for policy, description in policies.items():
                    info_text += f"• {policy}：{description}\n"
                matched_info.append(info_text)
                matched_categories.add(category)

        # 如果没有精确匹配，尝试关键词匹配
        if not matched_info:
            for category, policies in self.billing_database.items():
                if category in matched_categories:
                    continue
                if any(keyword in query_lower for keyword in ["退款", "发票", "支付", "账单", "价格"]):
                    info_text = f"""相关服务：{category}\n"""
                    # 只显示前2项政策
                    for i, (policy, description) in enumerate(policies.items()):
                        if i < 2:
                            info_text += f"• {policy}：{description}\n"
                    info_text += "..."
                    matched_info.append(info_text)
                    matched_categories.add(category)

        return "\n".join(matched_info) if matched_info else ""

"""normalize_classifier_label 单元测试：标签归一化与保守兜底（护栏核心）。"""
from tools.query_tools import normalize_classifier_label


class TestExactLabels:
    """六个合法标签的精确命中。"""

    def test_all_exact_labels(self):
        for label in (
            "product_info", "technical_support", "billing",
            "complaint", "general_inquiry", "out_of_scope",
        ):
            assert normalize_classifier_label(label) == label

    def test_strip_whitespace(self):
        assert normalize_classifier_label("  billing\n") == "billing"


class TestVariants:
    """大小写 / 分隔符 / 前缀说明等变体。"""

    def test_case_insensitive(self):
        assert normalize_classifier_label("Product_Info") == "product_info"
        assert normalize_classifier_label("BILLING") == "billing"

    def test_hyphen_to_underscore(self):
        assert normalize_classifier_label("product-info") == "product_info"

    def test_spaced_variant(self):
        # "general inquiry"（空格分隔）应还原为 general_inquiry 而非误判越界
        assert normalize_classifier_label("general inquiry") == "general_inquiry"

    def test_multiline_with_explanation(self):
        # 模型夹带解释：取首个词命中
        assert normalize_classifier_label(
            "complaint\n因为客户表达了对售后服务的不满"
        ) == "complaint"

    def test_leading_punctuation(self):
        assert normalize_classifier_label('"billing"') == "billing"


class TestFallback:
    """无法识别时必须保守兜底为 out_of_scope（不放行）。"""

    def test_empty_string(self):
        assert normalize_classifier_label("") == "out_of_scope"

    def test_garbage_text(self):
        assert normalize_classifier_label("今天天气不错") == "out_of_scope"

    def test_refusal_in_chinese(self):
        # 模型以自然语言拒答：仍是 out_of_scope，绝不能改判为业务咨询
        assert normalize_classifier_label("抱歉，我无法完成该请求") == "out_of_scope"

    def test_refusal_in_english(self):
        assert normalize_classifier_label("I'm sorry, I cannot help with that") == "out_of_scope"

    def test_refusal_not_misrouted_to_business(self):
        result = normalize_classifier_label("对不起，我不能回答这个问题")
        assert result == "out_of_scope"
        assert result not in (
            "product_info", "technical_support", "billing",
            "complaint", "general_inquiry",
        )

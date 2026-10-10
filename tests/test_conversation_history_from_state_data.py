"""conversation_history_from_state_data / append_turn_from_state 单元测试：
线程 state JSON 解析优先级、response 去重、图片透传（防止双气泡回归）。"""
from chat_web_service import (
    append_turn_from_state,
    conversation_history_from_state_data,
    last_user_question_from_history,
)


class TestSourcePriority:
    """persisted_dialogue > conversation_history > messages。"""

    def test_prefers_persisted_dialogue(self):
        state = {
            "values": {
                "persisted_dialogue": [
                    {"content": "手机多少钱", "is_user": True, "timestamp": "2026-01-01 10:00:00"},
                    {"content": "1999元起", "is_user": False, "timestamp": "2026-01-01 10:00:05"},
                ],
                "conversation_history": [
                    {"content": "旧数据不应出现", "is_user": True, "timestamp": "t"},
                ],
                "messages": [{"role": "user", "content": "更旧数据"}],
            }
        }
        history = conversation_history_from_state_data(state)
        assert [m["content"] for m in history] == ["手机多少钱", "1999元起"]

    def test_falls_back_to_conversation_history(self):
        state = {
            "values": {
                "persisted_dialogue": [],
                "conversation_history": [
                    {"content": "退款进度", "is_user": True, "timestamp": "t1"},
                ],
            }
        }
        history = conversation_history_from_state_data(state)
        assert len(history) == 1
        assert history[0]["content"] == "退款进度"

    def test_falls_back_to_messages(self):
        state = {
            "values": {
                "messages": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "您好，请问有什么可以帮您？"},
                ],
            }
        }
        history = conversation_history_from_state_data(state)
        assert len(history) == 2
        assert history[0]["is_user"] is True
        assert history[1]["is_user"] is False

    def test_non_dict_input(self):
        assert conversation_history_from_state_data(None) == []
        assert conversation_history_from_state_data("garbage") == []


class TestResponseDedup:
    """有轮次列表时不再追加 values.response，防止双气泡。"""

    def test_response_not_duplicated_when_turns_exist(self):
        state = {
            "values": {
                "persisted_dialogue": [
                    {"content": "问", "is_user": True, "timestamp": "t"},
                    {"content": "答", "is_user": False, "timestamp": "t"},
                ],
                "response": "【产品专家's Response】\n答",
            }
        }
        history = conversation_history_from_state_data(state)
        # 不应出现第二个包含 "答" 的助手气泡
        assistant_contents = [m["content"] for m in history if not m["is_user"]]
        assert assistant_contents == ["答"]

    def test_response_appended_when_only_messages(self):
        state = {
            "values": {
                "messages": [{"role": "user", "content": "问"}],
                "response": "答",
            }
        }
        history = conversation_history_from_state_data(state)
        assert history[-1]["content"] == "答"
        assert history[-1]["is_user"] is False

    def test_response_dedup_by_content(self):
        # messages 已含相同助手内容时不重复追加
        state = {
            "values": {
                "messages": [
                    {"role": "user", "content": "问"},
                    {"role": "assistant", "content": "答"},
                ],
                "response": "答",
            }
        }
        history = conversation_history_from_state_data(state)
        assert len([m for m in history if not m["is_user"]]) == 1


class TestAppendTurn:
    def test_images_passthrough(self):
        history = []
        append_turn_from_state(history, {
            "content": "看看这张图",
            "is_user": True,
            "timestamp": "t",
            "images": ["data:image/png;base64,aGk="],
        })
        assert history[0]["images"] == ["data:image/png;base64,aGk="]

    def test_empty_content_skipped(self):
        history = []
        append_turn_from_state(history, {"content": "", "is_user": True})
        assert history == []

    def test_timestamp_only_when_present(self):
        history = []
        append_turn_from_state(history, {"content": "hi", "is_user": True})
        assert "timestamp" not in history[0]


class TestLastUserQuestion:
    def test_takes_last_user_message(self):
        history = [
            {"is_user": True, "content": "第一问"},
            {"is_user": False, "content": "第一答"},
            {"is_user": True, "content": "  第二问  "},
        ]
        assert last_user_question_from_history(history) == "第二问"

    def test_empty_history(self):
        assert last_user_question_from_history([]) == ""

"""validate_image_data / _validate_single_image 单元测试：图片校验的边界控制。"""
import base64

import pytest

from app import validation as web_app


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _data_url(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{_b64(data)}"


VALID_PNG_1B = _data_url(b"\x89PNG\r\n\x1a\n")  # 8 字节合法 payload


class TestSingleImage:
    def test_valid_data_url(self):
        ok, err = web_app._validate_single_image(VALID_PNG_1B)
        assert err is None
        assert ok == VALID_PNG_1B

    def test_rejects_non_string(self):
        ok, err = web_app._validate_single_image(123)
        assert ok is None
        assert err is not None

    def test_rejects_wrong_prefix(self):
        ok, err = web_app._validate_single_image("data:text/plain;base64,aGk=")
        assert ok is None
        assert "data URL" in err

    def test_rejects_missing_comma(self):
        ok, err = web_app._validate_single_image("data:image/png;base64")
        assert ok is None

    def test_rejects_non_base64_header(self):
        ok, err = web_app._validate_single_image("data:image/png,xxxx")
        assert ok is None
        assert "base64" in err

    def test_rejects_invalid_base64_body(self):
        ok, err = web_app._validate_single_image("data:image/png;base64,!!!not-base64!!!")
        assert ok is None

    def test_rejects_empty_content(self):
        ok, err = web_app._validate_single_image("data:image/png;base64,")
        assert ok is None
        assert "空" in err

    def test_rejects_oversize(self, monkeypatch):
        # 缩小上限避免测试构造 5MB 字符串
        monkeypatch.setattr(web_app, "MAX_IMAGE_BYTES", 4)
        ok, err = web_app._validate_single_image(_data_url(b"12345"))
        assert ok is None
        assert "过大" in err


class TestBatch:
    def test_none_input(self):
        imgs, err = web_app.validate_image_data(None)
        assert imgs is None and err is None

    def test_empty_string_input(self):
        imgs, err = web_app.validate_image_data("")
        assert imgs is None and err is None

    def test_single_string_becomes_list(self):
        imgs, err = web_app.validate_image_data(VALID_PNG_1B)
        assert err is None
        assert imgs == [VALID_PNG_1B]

    def test_valid_list(self):
        imgs, err = web_app.validate_image_data([VALID_PNG_1B, VALID_PNG_1B])
        assert err is None
        assert len(imgs) == 2

    def test_rejects_too_many_images(self, monkeypatch):
        monkeypatch.setattr(web_app, "MAX_IMAGE_COUNT", 2)
        imgs, err = web_app.validate_image_data([VALID_PNG_1B] * 3)
        assert imgs is None
        assert "数量" in err

    def test_rejects_invalid_type(self):
        imgs, err = web_app.validate_image_data(42)
        assert imgs is None
        assert err is not None

    def test_rejects_if_any_invalid(self):
        imgs, err = web_app.validate_image_data([VALID_PNG_1B, "not-a-data-url"])
        assert imgs is None
        assert err is not None

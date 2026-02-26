"""Tests for pii.redactor (the legacy standalone PII redactor module)."""
from __future__ import annotations

import pytest

from agent_observability.pii.redactor import PiiRedactor, RedactionConfig


class TestRedactionConfig:
    def test_defaults(self) -> None:
        config = RedactionConfig()
        assert config.redact_emails is True
        assert config.redact_phone is True
        assert config.redact_ssn is True
        assert config.redact_credit_card is True
        assert config.redact_ip is True
        assert config.redact_custom_patterns == []
        assert config.replacement == "[REDACTED]"

    def test_custom_replacement(self) -> None:
        config = RedactionConfig(replacement="***")
        assert config.replacement == "***"


class TestPiiRedactorLegacy:
    def test_redact_email(self) -> None:
        redactor = PiiRedactor()
        result = redactor.redact("contact user@example.com today")
        assert "user@example.com" not in result
        assert "[REDACTED]" in result

    def test_redact_ssn(self) -> None:
        redactor = PiiRedactor()
        result = redactor.redact("ssn: 123-45-6789")
        assert "123-45-6789" not in result

    def test_redact_credit_card_visa(self) -> None:
        redactor = PiiRedactor()
        result = redactor.redact("card 4111111111111111 charged")
        assert "4111111111111111" not in result

    def test_redact_ipv4(self) -> None:
        redactor = PiiRedactor()
        result = redactor.redact("server 192.168.0.1 pinged")
        assert "192.168.0.1" not in result

    def test_no_redaction_for_clean_text(self) -> None:
        redactor = PiiRedactor()
        text = "nothing sensitive at all"
        assert redactor.redact(text) == text

    def test_redact_disabled_fields(self) -> None:
        config = RedactionConfig(
            redact_emails=False,
            redact_phone=False,
            redact_ssn=False,
            redact_credit_card=False,
            redact_ip=False,
        )
        redactor = PiiRedactor(config=config)
        text = "user@example.com 123-45-6789 192.168.1.1"
        # No patterns active — text should pass through
        assert redactor.redact(text) == text

    def test_redact_span_attributes(self) -> None:
        redactor = PiiRedactor()
        attributes = {
            "user_email": "user@example.com",
            "count": 42,
            "safe": "nothing here",
        }
        result = redactor.redact_span_attributes(attributes)
        assert "user@example.com" not in result["user_email"]
        assert result["count"] == 42
        assert result["safe"] == "nothing here"

    def test_redact_span_attributes_non_string_pass_through(self) -> None:
        redactor = PiiRedactor()
        attributes = {"number": 123, "flag": True, "items": [1, 2, 3]}
        result = redactor.redact_span_attributes(attributes)
        assert result["number"] == 123
        assert result["flag"] is True

    def test_redact_dict_shallow(self) -> None:
        redactor = PiiRedactor()
        data = {"email": "user@example.com"}
        result = redactor.redact_dict(data)
        assert "user@example.com" not in result["email"]

    def test_redact_dict_nested(self) -> None:
        redactor = PiiRedactor()
        data = {"outer": {"inner": "user@example.com"}}
        result = redactor.redact_dict(data)
        inner = result["outer"]
        assert isinstance(inner, dict)
        assert "user@example.com" not in inner["inner"]

    def test_redact_dict_with_list(self) -> None:
        redactor = PiiRedactor()
        data = {"emails": ["user@a.com", "user@b.com"]}
        result = redactor.redact_dict(data)
        for email in result["emails"]:  # type: ignore[union-attr]
            assert "user@" not in email

    def test_redact_dict_non_string_leaf_passthrough(self) -> None:
        redactor = PiiRedactor()
        data = {"value": 42}
        result = redactor.redact_dict(data)
        assert result["value"] == 42

    def test_custom_patterns_applied(self) -> None:
        config = RedactionConfig(redact_custom_patterns=[r"SECRET-\d+"])
        redactor = PiiRedactor(config=config)
        result = redactor.redact("token SECRET-9876 used")
        assert "SECRET-9876" not in result

    def test_invalid_custom_pattern_skipped(self) -> None:
        config = RedactionConfig(redact_custom_patterns=[r"[invalid("])
        # Should not raise during init
        redactor = PiiRedactor(config=config)
        result = redactor.redact("some text")
        assert isinstance(result, str)

    def test_custom_replacement_token(self) -> None:
        config = RedactionConfig(replacement="***")
        redactor = PiiRedactor(config=config)
        result = redactor.redact("email user@example.com here")
        assert "***" in result

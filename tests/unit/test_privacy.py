"""Comprehensive tests for the privacy subsystem.

Covers:
- privacy.patterns  (PiiPattern, COMMON_PATTERNS, US_PATTERNS, EU_PATTERNS,
                     get_patterns_for_jurisdiction)
- privacy.config    (CustomPatternConfig, PrivacyConfig)
- privacy.detector  (PiiMatch, DetectionResult, PiiDetector)
- privacy.redactor  (PiiRedactor, PiiRedactingSpanProcessor)
"""
from __future__ import annotations

import pytest

from agent_observability.privacy.patterns import (
    PiiPattern,
    COMMON_PATTERNS,
    EU_PATTERNS,
    US_PATTERNS,
    ALL_PATTERNS,
    PATTERNS_BY_JURISDICTION,
    get_patterns_for_jurisdiction,
)
from agent_observability.privacy.config import CustomPatternConfig, PrivacyConfig
from agent_observability.privacy.detector import (
    DetectionResult,
    PiiDetector,
    PiiMatch,
)
from agent_observability.privacy.redactor import PiiRedactor


# ── PiiPattern ─────────────────────────────────────────────────────────────────


class TestPiiPattern:
    def test_compiled_returns_pattern(self) -> None:
        pattern = PiiPattern(
            name="test",
            pattern=r"\d{4}",
            jurisdiction="common",
            replacement_token="[REDACTED_TEST]",
        )
        compiled = pattern.compiled()
        assert compiled.search("1234") is not None
        assert compiled.search("abc") is None

    def test_compiled_is_case_insensitive(self) -> None:
        pattern = PiiPattern(
            name="test",
            pattern=r"hello",
            jurisdiction="common",
            replacement_token="[REDACTED]",
        )
        assert pattern.compiled().search("HELLO") is not None


# ── Pattern collections ────────────────────────────────────────────────────────


class TestPatternCollections:
    def test_common_patterns_not_empty(self) -> None:
        assert len(COMMON_PATTERNS) > 0

    def test_us_patterns_not_empty(self) -> None:
        assert len(US_PATTERNS) > 0

    def test_eu_patterns_not_empty(self) -> None:
        assert len(EU_PATTERNS) > 0

    def test_all_patterns_combines_all(self) -> None:
        assert len(ALL_PATTERNS) == len(COMMON_PATTERNS) + len(US_PATTERNS) + len(EU_PATTERNS)

    def test_patterns_by_jurisdiction_keys(self) -> None:
        assert set(PATTERNS_BY_JURISDICTION.keys()) == {"common", "us", "eu", "all"}


class TestGetPatternsForJurisdiction:
    def test_common_jurisdiction(self) -> None:
        patterns = get_patterns_for_jurisdiction("common")
        assert patterns == COMMON_PATTERNS

    def test_us_jurisdiction_includes_common(self) -> None:
        patterns = get_patterns_for_jurisdiction("us")
        for p in COMMON_PATTERNS:
            assert p in patterns

    def test_eu_jurisdiction_includes_common(self) -> None:
        patterns = get_patterns_for_jurisdiction("eu")
        for p in COMMON_PATTERNS:
            assert p in patterns

    def test_all_jurisdiction(self) -> None:
        patterns = get_patterns_for_jurisdiction("all")
        assert patterns == ALL_PATTERNS

    def test_unknown_jurisdiction_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown jurisdiction"):
            get_patterns_for_jurisdiction("mars")


# ── Pattern detection accuracy ─────────────────────────────────────────────────


class TestCommonPatternDetection:
    def test_email_detected(self) -> None:
        detector = PiiDetector(jurisdiction="common")
        assert detector.has_pii("email me at user@example.com please")

    def test_ipv4_detected(self) -> None:
        detector = PiiDetector(jurisdiction="common")
        assert detector.has_pii("server at 192.168.1.100")

    def test_bearer_token_detected(self) -> None:
        detector = PiiDetector(jurisdiction="common")
        assert detector.has_pii("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9abcdefghijklmno")

    def test_clean_text_not_detected(self) -> None:
        detector = PiiDetector(jurisdiction="common")
        assert not detector.has_pii("the quick brown fox jumps over the lazy dog")


class TestUsPatternDetection:
    def test_ssn_detected(self) -> None:
        detector = PiiDetector(jurisdiction="us")
        assert detector.has_pii("ssn: 123-45-6789")

    def test_credit_card_detected(self) -> None:
        detector = PiiDetector(jurisdiction="us")
        # Valid Visa test card
        assert detector.has_pii("card: 4111111111111111")


# ── PrivacyConfig ──────────────────────────────────────────────────────────────


class TestPrivacyConfig:
    def test_defaults(self) -> None:
        config = PrivacyConfig()
        assert config.enabled is True
        assert config.jurisdiction == "all"
        assert config.redact_span_attributes is True
        assert config.redact_log_messages is False
        assert config.custom_patterns == []
        assert "agent.id" in config.exclude_attribute_keys
        assert config.full_logging_opt_in is False
        assert "{NAME}" in config.redaction_token_template

    def test_effective_redaction_enabled_when_enabled(self) -> None:
        config = PrivacyConfig(enabled=True, full_logging_opt_in=False)
        assert config.effective_redaction_enabled() is True

    def test_effective_redaction_disabled_when_disabled(self) -> None:
        config = PrivacyConfig(enabled=False)
        assert config.effective_redaction_enabled() is False

    def test_effective_redaction_disabled_by_opt_in(self) -> None:
        config = PrivacyConfig(enabled=True, full_logging_opt_in=True)
        assert config.effective_redaction_enabled() is False

    def test_invalid_token_template_raises(self) -> None:
        with pytest.raises(Exception):
            PrivacyConfig(redaction_token_template="NO_PLACEHOLDER_HERE")

    def test_custom_pattern_config(self) -> None:
        cp = CustomPatternConfig(
            name="employee_id",
            pattern=r"EMP\d{6}",
            replacement_token="[REDACTED_EMP_ID]",
            jurisdiction="custom",
        )
        assert cp.name == "employee_id"
        assert cp.replacement_token == "[REDACTED_EMP_ID]"

    def test_jurisdiction_options(self) -> None:
        for jur in ("common", "us", "eu", "all"):
            config = PrivacyConfig(jurisdiction=jur)
            assert config.jurisdiction == jur


# ── PiiMatch / DetectionResult ─────────────────────────────────────────────────


class TestDetectionResult:
    def test_has_pii_false_when_no_matches(self) -> None:
        result = DetectionResult(original="clean text", matches=[])
        assert result.has_pii is False

    def test_has_pii_true_when_matches(self) -> None:
        match = PiiMatch(
            pattern_name="email",
            replacement_token="[REDACTED_EMAIL]",
            start=0,
            end=5,
            matched_text="test@x.com",
        )
        result = DetectionResult(original="test@x.com", matches=[match])
        assert result.has_pii is True

    def test_pattern_names(self) -> None:
        matches = [
            PiiMatch("email", "[REDACTED_EMAIL]", 0, 10, "test@x.com"),
            PiiMatch("ssn", "[REDACTED_SSN]", 20, 31, "123-45-6789"),
        ]
        result = DetectionResult(original="...", matches=matches)
        assert result.pattern_names == ["email", "ssn"]


# ── PiiDetector ────────────────────────────────────────────────────────────────


class TestPiiDetector:
    def test_detect_email(self) -> None:
        detector = PiiDetector(jurisdiction="common")
        result = detector.detect("contact user@example.com for info")
        assert result.has_pii
        assert any(m.pattern_name == "email" for m in result.matches)

    def test_detect_no_pii(self) -> None:
        detector = PiiDetector(jurisdiction="common")
        result = detector.detect("nothing to see here")
        assert not result.has_pii

    def test_detect_sorts_matches_by_position(self) -> None:
        detector = PiiDetector(jurisdiction="common")
        text = "email a@b.com and ip 192.168.1.1"
        result = detector.detect(text)
        if len(result.matches) >= 2:
            for i in range(len(result.matches) - 1):
                assert result.matches[i].start <= result.matches[i + 1].start

    def test_has_pii_fast_path(self) -> None:
        detector = PiiDetector(jurisdiction="common")
        assert detector.has_pii("a@b.com")
        assert not detector.has_pii("nothing sensitive")

    def test_scan_dict_returns_pii_keys_only(self) -> None:
        detector = PiiDetector(jurisdiction="common")
        data = {
            "user_email": "user@example.com",
            "safe_field": "nothing here",
            "count": "42",
        }
        results = detector.scan_dict(data)
        assert "user_email" in results
        assert "safe_field" not in results

    def test_scan_dict_excludes_keys(self) -> None:
        detector = PiiDetector(jurisdiction="common")
        data = {"email": "user@example.com", "trace_id": "user@example.com"}
        results = detector.scan_dict(data, exclude_keys=["trace_id"])
        assert "email" in results
        assert "trace_id" not in results

    def test_scan_dict_ignores_non_strings(self) -> None:
        detector = PiiDetector(jurisdiction="common")
        data = {"count": 42, "items": [1, 2, 3]}  # type: ignore[dict-item]
        results = detector.scan_dict(data)
        assert results == {}

    def test_extra_patterns_applied(self) -> None:
        custom = PiiPattern(
            name="employee_id",
            pattern=r"EMP\d{6}",
            jurisdiction="custom",
            replacement_token="[REDACTED_EMP]",
        )
        detector = PiiDetector(jurisdiction="common", extra_patterns=[custom])
        assert detector.has_pii("employee EMP123456 handled this")

    def test_dedup_by_name_last_writer_wins(self) -> None:
        p1 = PiiPattern("dupe", r"aaa", "common", "[A]")
        p2 = PiiPattern("dupe", r"bbb", "custom", "[B]")
        detector = PiiDetector(jurisdiction="common", extra_patterns=[p1, p2])
        # Only one pattern named "dupe" should remain
        pattern_names = [p.name for p, _ in detector._patterns]
        assert pattern_names.count("dupe") == 1


# ── PiiRedactor ────────────────────────────────────────────────────────────────


class TestPiiRedactor:
    def test_redact_email(self) -> None:
        redactor = PiiRedactor()
        redacted, count = redactor.redact("Email me at user@example.com please")
        assert "user@example.com" not in redacted
        assert count == 1

    def test_redact_no_pii(self) -> None:
        redactor = PiiRedactor()
        redacted, count = redactor.redact("nothing sensitive here")
        assert redacted == "nothing sensitive here"
        assert count == 0

    def test_redact_with_custom_template(self) -> None:
        config = PrivacyConfig(redaction_token_template="<<{NAME}>>")
        redactor = PiiRedactor(config=config)
        redacted, count = redactor.redact("contact user@example.com")
        assert "<<EMAIL>>" in redacted

    def test_redact_disabled_by_config(self) -> None:
        config = PrivacyConfig(enabled=False)
        redactor = PiiRedactor(config=config)
        text = "user@example.com"
        redacted, count = redactor.redact(text)
        assert redacted == text
        assert count == 0

    def test_redact_disabled_by_opt_in(self) -> None:
        config = PrivacyConfig(full_logging_opt_in=True)
        redactor = PiiRedactor(config=config)
        text = "user@example.com"
        redacted, count = redactor.redact(text)
        assert redacted == text

    def test_redact_dict_strings_only(self) -> None:
        redactor = PiiRedactor()
        data = {
            "email": "user@example.com",
            "count": 42,
            "nested": {"inner": "safe"},
        }
        redacted_dict, total = redactor.redact_dict(data)
        assert "user@example.com" not in str(redacted_dict["email"])
        assert redacted_dict["count"] == 42

    def test_redact_dict_excludes_config_keys(self) -> None:
        config = PrivacyConfig()
        redactor = PiiRedactor(config=config)
        data = {
            "agent.id": "user@example.com",  # should be excluded
            "message": "contact user@example.com",
        }
        redacted_dict, _ = redactor.redact_dict(data)
        assert redacted_dict["agent.id"] == "user@example.com"

    def test_redact_dict_extra_exclude_keys(self) -> None:
        redactor = PiiRedactor()
        data = {"email": "user@example.com", "skip_me": "user@example.com"}
        redacted_dict, _ = redactor.redact_dict(data, exclude_keys=["skip_me"])
        assert redacted_dict["skip_me"] == "user@example.com"

    def test_redact_dict_disabled(self) -> None:
        config = PrivacyConfig(enabled=False)
        redactor = PiiRedactor(config=config)
        data = {"email": "user@example.com"}
        redacted_dict, count = redactor.redact_dict(data)
        assert redacted_dict["email"] == "user@example.com"
        assert count == 0

    def test_custom_pattern_applied(self) -> None:
        custom_pattern = CustomPatternConfig(
            name="ticket_id",
            pattern=r"TICKET-[A-Z]{6}",
            replacement_token="[REDACTED_TICKET]",
        )
        config = PrivacyConfig(custom_patterns=[custom_pattern])
        redactor = PiiRedactor(config=config)
        redacted, count = redactor.redact("Issue TICKET-ABCDEF opened")
        assert "TICKET-ABCDEF" not in redacted
        assert count >= 1

    def test_multiple_pii_items_in_one_string(self) -> None:
        redactor = PiiRedactor()
        text = "from user@example.com and also user2@example.org"
        redacted, count = redactor.redact(text)
        assert count >= 2
        assert "user@example.com" not in redacted


# ── PiiRedactingSpanProcessor ─────────────────────────────────────────────────


class TestPiiRedactingSpanProcessor:
    def test_on_start_is_noop(self) -> None:
        from agent_observability.privacy.redactor import PiiRedactingSpanProcessor

        processor = PiiRedactingSpanProcessor()
        # Should not raise
        processor.on_start(object())

    def test_on_end_without_otel_sdk(self) -> None:
        from agent_observability.privacy.redactor import PiiRedactingSpanProcessor

        processor = PiiRedactingSpanProcessor()
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "agent_observability.privacy.redactor._OTEL_SDK_AVAILABLE", False
            )
            # Should not raise
            processor.on_end(MagicMock())

    def test_shutdown_is_noop(self) -> None:
        from agent_observability.privacy.redactor import PiiRedactingSpanProcessor

        processor = PiiRedactingSpanProcessor()
        processor.shutdown()  # should not raise

    def test_force_flush_returns_true(self) -> None:
        from agent_observability.privacy.redactor import PiiRedactingSpanProcessor

        processor = PiiRedactingSpanProcessor()
        assert processor.force_flush() is True

    def test_on_end_with_mock_span_and_pii(self) -> None:
        from unittest.mock import MagicMock, patch
        from agent_observability.privacy.redactor import PiiRedactingSpanProcessor

        processor = PiiRedactingSpanProcessor()
        mock_span = MagicMock()
        mock_span.attributes = {"message": "contact user@example.com"}

        with patch("agent_observability.privacy.redactor._OTEL_SDK_AVAILABLE", True):
            processor.on_end(mock_span)
            mock_span.set_attribute.assert_called()


try:
    from unittest.mock import MagicMock
except ImportError:
    pass

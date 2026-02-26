"""Unit tests for spans.enricher — SpanEnricher."""
from __future__ import annotations

from unittest.mock import MagicMock, call

from agent_observability.spans.enricher import SpanEnricher
from agent_observability.spans.conventions import (
    AGENT_ENVIRONMENT,
    AGENT_FRAMEWORK,
    AGENT_ID,
    AGENT_NAME,
    AGENT_SESSION_ID,
    AGENT_VERSION,
)


def _mock_span() -> MagicMock:
    span = MagicMock()
    span.set_attribute = MagicMock()
    return span


class TestSpanEnricher:
    def test_constructor_builds_static_attributes_from_non_empty_fields(self) -> None:
        enricher = SpanEnricher(
            agent_id="a1",
            session_id="s1",
            framework="langchain",
            agent_name="MyBot",
            agent_version="1.2.3",
            environment="production",
        )
        assert AGENT_ID in enricher._static_attributes
        assert AGENT_SESSION_ID in enricher._static_attributes
        assert AGENT_FRAMEWORK in enricher._static_attributes
        assert AGENT_NAME in enricher._static_attributes
        assert AGENT_VERSION in enricher._static_attributes
        assert AGENT_ENVIRONMENT in enricher._static_attributes

    def test_empty_fields_excluded_from_static_attributes(self) -> None:
        enricher = SpanEnricher(agent_id="a1")
        assert AGENT_ID in enricher._static_attributes
        assert AGENT_SESSION_ID not in enricher._static_attributes
        assert AGENT_FRAMEWORK not in enricher._static_attributes

    def test_on_start_stamps_attributes_when_otel_available(self) -> None:
        enricher = SpanEnricher(agent_id="a1", session_id="s1", framework="crewai")
        # Patch _OTEL_SDK_AVAILABLE so the on_start path is exercised
        import agent_observability.spans.enricher as enricher_mod
        original = enricher_mod._OTEL_SDK_AVAILABLE
        enricher_mod._OTEL_SDK_AVAILABLE = True
        try:
            span = _mock_span()
            enricher.on_start(span)
            called_keys = {c.args[0] for c in span.set_attribute.call_args_list}
            assert AGENT_ID in called_keys
            assert AGENT_SESSION_ID in called_keys
            assert AGENT_FRAMEWORK in called_keys
        finally:
            enricher_mod._OTEL_SDK_AVAILABLE = original

    def test_on_start_is_noop_when_otel_not_available(self) -> None:
        enricher = SpanEnricher(agent_id="a1")
        import agent_observability.spans.enricher as enricher_mod
        original = enricher_mod._OTEL_SDK_AVAILABLE
        enricher_mod._OTEL_SDK_AVAILABLE = False
        try:
            span = _mock_span()
            enricher.on_start(span)
            span.set_attribute.assert_not_called()
        finally:
            enricher_mod._OTEL_SDK_AVAILABLE = original

    def test_on_end_is_noop(self) -> None:
        enricher = SpanEnricher()
        enricher.on_end(MagicMock())  # should not raise

    def test_force_flush_returns_true(self) -> None:
        enricher = SpanEnricher()
        assert enricher.force_flush() is True

    def test_shutdown_does_not_raise(self) -> None:
        enricher = SpanEnricher()
        enricher.shutdown()

    def test_update_session_changes_session_id(self) -> None:
        enricher = SpanEnricher(session_id="old-session")
        enricher.update_session("new-session")
        assert enricher._session_id == "new-session"
        assert enricher._static_attributes[AGENT_SESSION_ID] == "new-session"

    def test_update_agent_id_changes_agent_id(self) -> None:
        enricher = SpanEnricher(agent_id="old-agent")
        enricher.update_agent_id("new-agent")
        assert enricher._agent_id == "new-agent"
        assert enricher._static_attributes[AGENT_ID] == "new-agent"

    def test_on_start_silently_ignores_span_attribute_errors(self) -> None:
        enricher = SpanEnricher(agent_id="a1")
        import agent_observability.spans.enricher as enricher_mod
        original = enricher_mod._OTEL_SDK_AVAILABLE
        enricher_mod._OTEL_SDK_AVAILABLE = True
        try:
            span = MagicMock()
            span.set_attribute.side_effect = RuntimeError("span is read-only")
            enricher.on_start(span)  # must not raise
        finally:
            enricher_mod._OTEL_SDK_AVAILABLE = original

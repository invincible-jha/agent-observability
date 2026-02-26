"""PiiRedactor — redact PII from span attributes and text before export.

Can be used standalone or installed as an OTel SpanProcessor.
"""
from __future__ import annotations

import logging
import re
from typing import Optional, Sequence

from agent_observability.privacy.config import PrivacyConfig
from agent_observability.privacy.detector import PiiDetector
from agent_observability.privacy.patterns import PiiPattern, get_patterns_for_jurisdiction
from agent_observability.spans.conventions import PRIVACY_REDACTED, PRIVACY_REDACTION_COUNT

logger = logging.getLogger(__name__)

try:
    from opentelemetry.sdk.trace import ReadableSpan, Span
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    _OTEL_SDK_AVAILABLE = True
except ImportError:
    _OTEL_SDK_AVAILABLE = False
    ReadableSpan = None  # type: ignore[assignment,misc]
    Span = None  # type: ignore[assignment,misc]
    SpanExporter = None  # type: ignore[assignment]
    SpanExportResult = None  # type: ignore[assignment]

try:
    from opentelemetry.sdk.trace import SpanProcessor

    _PROCESSOR_AVAILABLE = True
except ImportError:
    _PROCESSOR_AVAILABLE = False
    SpanProcessor = object  # type: ignore[assignment,misc]


class PiiRedactor:
    """Apply regex-based PII redaction to arbitrary strings and OTel spans.

    Parameters
    ----------
    config:
        A :class:`PrivacyConfig` controlling which patterns and jurisdictions
        to apply.  Defaults to redacting everything.
    """

    def __init__(self, config: Optional[PrivacyConfig] = None) -> None:
        self._config = config or PrivacyConfig()
        extra = [
            PiiPattern(
                name=cp.name,
                pattern=cp.pattern,
                jurisdiction=cp.jurisdiction,
                replacement_token=cp.replacement_token,
            )
            for cp in self._config.custom_patterns
        ]
        self._detector = PiiDetector(
            jurisdiction=self._config.jurisdiction,
            extra_patterns=extra,
        )

    # ── Text redaction ────────────────────────────────────────────────────────

    def redact(self, text: str) -> tuple[str, int]:
        """Return *(redacted_text, redaction_count)*.

        Parameters
        ----------
        text:
            Input string to scan.

        Returns
        -------
        A tuple of ``(redacted_text, number_of_redactions_applied)``.
        """
        if not self._config.effective_redaction_enabled():
            return text, 0

        result = self._detector.detect(text)
        if not result.has_pii:
            return text, 0

        # Apply substitutions right-to-left to preserve offsets
        redacted = text
        count = 0
        for match in sorted(result.matches, key=lambda m: m.start, reverse=True):
            token = self._config.redaction_token_template.replace(
                "{NAME}", match.pattern_name.upper()
            )
            redacted = redacted[: match.start] + token + redacted[match.end :]
            count += 1

        return redacted, count

    def redact_dict(
        self,
        data: dict[str, object],
        exclude_keys: Optional[Sequence[str]] = None,
    ) -> tuple[dict[str, object], int]:
        """Return a copy of *data* with PII values redacted.

        Parameters
        ----------
        data:
            A flat attribute dict (e.g. OTel span attributes).
        exclude_keys:
            Keys to skip (in addition to ``config.exclude_attribute_keys``).

        Returns
        -------
        ``(redacted_dict, total_redaction_count)``
        """
        if not self._config.effective_redaction_enabled():
            return dict(data), 0

        excluded = set(self._config.exclude_attribute_keys) | set(exclude_keys or [])
        redacted: dict[str, object] = {}
        total_count = 0

        for key, value in data.items():
            if key in excluded or not isinstance(value, str):
                redacted[key] = value
                continue
            new_value, count = self.redact(value)
            redacted[key] = new_value
            total_count += count

        return redacted, total_count


class PiiRedactingSpanProcessor(SpanProcessor):  # type: ignore[misc]
    """OTel SpanProcessor that redacts PII from span attribute values on end.

    Parameters
    ----------
    redactor:
        A configured :class:`PiiRedactor` instance.
    """

    def __init__(self, redactor: Optional[PiiRedactor] = None) -> None:
        self._redactor = redactor or PiiRedactor()

    def on_start(self, span: object, parent_context: Optional[object] = None) -> None:
        """No-op — redaction happens at span end."""

    def on_end(self, span: object) -> None:
        """Redact PII from span attributes in place."""
        if not _OTEL_SDK_AVAILABLE:
            return
        try:
            raw_attrs = getattr(span, "attributes", None) or {}
            attr_dict = dict(raw_attrs)
            redacted_attrs, count = self._redactor.redact_dict(attr_dict)

            if count > 0:
                # Stamp the redaction metadata on the span
                if hasattr(span, "set_attribute"):
                    span.set_attribute(PRIVACY_REDACTED, True)  # type: ignore[union-attr]
                    span.set_attribute(PRIVACY_REDACTION_COUNT, count)  # type: ignore[union-attr]
                    for key, value in redacted_attrs.items():
                        if isinstance(value, (str, int, float, bool)):
                            span.set_attribute(key, value)  # type: ignore[union-attr]
        except Exception:
            logger.debug("PiiRedactingSpanProcessor: error during attribute redaction")

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

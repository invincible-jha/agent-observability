"""PiiRedactor — scrub PII from span attributes before export.

All regex patterns are pre-compiled at class instantiation for efficiency.
Custom patterns are supported via :attr:`RedactionConfig.redact_custom_patterns`.

Supported built-in detectors
-----------------------------
* Email addresses
* US phone numbers (various formats)
* US Social Security Numbers (SSN)
* Credit card numbers (major card formats)
* IPv4 and IPv6 addresses
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Union

# ── Compiled regex patterns ────────────────────────────────────────────────────

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

_PHONE_RE = re.compile(
    r"(?:\+?1[\s.\-]?)?(?:\(?\d{3}\)?[\s.\-]?)?\d{3}[\s.\-]?\d{4}",
)

_SSN_RE = re.compile(
    r"\b(?!000|666|9\d\d)\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b",
)

_CREDIT_CARD_RE = re.compile(
    r"\b(?:4[0-9]{12}(?:[0-9]{3})?|"     # Visa
    r"5[1-5][0-9]{14}|"                   # MasterCard
    r"3[47][0-9]{13}|"                    # Amex
    r"3(?:0[0-5]|[68][0-9])[0-9]{11}|"   # Diners
    r"6(?:011|5[0-9]{2})[0-9]{12}|"      # Discover
    r"(?:2131|1800|35\d{3})\d{11})\b",    # JCB
)

_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
)

_IPV6_RE = re.compile(
    r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
    r"|(?:[0-9a-fA-F]{1,4}:)*::[0-9a-fA-F:]*",
    re.IGNORECASE,
)


@dataclass
class RedactionConfig:
    """Configuration controlling which PII categories are redacted.

    All fields default to ``True`` so the safest mode is the default.
    """

    redact_emails: bool = True
    redact_phone: bool = True
    redact_ssn: bool = True
    redact_credit_card: bool = True
    redact_ip: bool = True
    redact_custom_patterns: list[str] = field(default_factory=list)
    replacement: str = "[REDACTED]"


class PiiRedactor:
    """Apply PII redaction to strings and span attribute dicts.

    Parameters
    ----------
    config:
        A :class:`RedactionConfig` controlling which patterns are active.
        Defaults to redacting everything.

    Example
    -------
    >>> redactor = PiiRedactor()
    >>> redactor.redact("Email me at user@example.com please")
    'Email me at [REDACTED] please'
    """

    def __init__(self, config: RedactionConfig | None = None) -> None:
        self.config: RedactionConfig = config if config is not None else RedactionConfig()
        self._patterns: list[re.Pattern[str]] = self._compile_patterns()

    # ── Pattern compilation ───────────────────────────────────────────────────

    def _compile_patterns(self) -> list[re.Pattern[str]]:
        """Build the list of active compiled patterns from the config."""
        patterns: list[re.Pattern[str]] = []

        if self.config.redact_emails:
            patterns.append(_EMAIL_RE)
        if self.config.redact_ssn:
            patterns.append(_SSN_RE)
        if self.config.redact_credit_card:
            patterns.append(_CREDIT_CARD_RE)
        if self.config.redact_phone:
            patterns.append(_PHONE_RE)
        if self.config.redact_ip:
            # IPv6 before IPv4 to avoid partial matches
            patterns.append(_IPV6_RE)
            patterns.append(_IPV4_RE)

        for raw_pattern in self.config.redact_custom_patterns:
            try:
                patterns.append(re.compile(raw_pattern))
            except re.error:
                pass  # Silently skip invalid custom patterns

        return patterns

    # ── Public API ────────────────────────────────────────────────────────────

    def redact(self, text: str) -> str:
        """Apply all enabled redaction patterns to *text*.

        Parameters
        ----------
        text:
            The string to scan and redact.

        Returns
        -------
        A new string with all detected PII replaced by
        :attr:`RedactionConfig.replacement`.
        """
        result = text
        for pattern in self._patterns:
            result = pattern.sub(self.config.replacement, result)
        return result

    def redact_span_attributes(
        self,
        attributes: dict[str, object],
    ) -> dict[str, object]:
        """Deep-redact all string values in a span attributes dict.

        Non-string values are passed through unchanged.

        Parameters
        ----------
        attributes:
            A flat ``{key: value}`` dict of span attributes.

        Returns
        -------
        A new dict with string values redacted.
        """
        return {
            key: self.redact(value) if isinstance(value, str) else value
            for key, value in attributes.items()
        }

    def redact_dict(
        self,
        data: dict[str, object],
    ) -> dict[str, object]:
        """Recursively redact all string values in an arbitrarily nested dict.

        Parameters
        ----------
        data:
            Dict to redact.  May contain nested dicts and lists.

        Returns
        -------
        A new dict with all string leaf values redacted.
        """
        result: dict[str, object] = {}
        for key, value in data.items():
            result[key] = self._redact_value(value)
        return result

    def _redact_value(
        self,
        value: object,
    ) -> object:
        """Recursively redact a single value."""
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, dict):
            return self.redact_dict(value)  # type: ignore[arg-type]
        if isinstance(value, list):
            return [self._redact_value(item) for item in value]
        return value

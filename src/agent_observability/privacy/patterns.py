"""PII regex patterns organised by jurisdiction and category.

All patterns are raw regex strings.  They are intentionally conservative
(may have false positives) to maximise privacy protection.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Pattern


@dataclass(frozen=True)
class PiiPattern:
    """A single PII detection rule."""

    name: str
    pattern: str
    jurisdiction: str  # "common", "us", "eu", "uk", etc.
    replacement_token: str  # e.g. "[REDACTED_EMAIL]"
    description: str = ""

    def compiled(self) -> Pattern[str]:
        return re.compile(self.pattern, re.IGNORECASE)


# ── Common patterns (all jurisdictions) ────────────────────────────────────────

COMMON_PATTERNS: list[PiiPattern] = [
    PiiPattern(
        name="email",
        pattern=r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        jurisdiction="common",
        replacement_token="[REDACTED_EMAIL]",
        description="Email address",
    ),
    PiiPattern(
        name="url_with_credentials",
        pattern=r"https?://[A-Za-z0-9._~:/?#@!$&'()*+,;=%-]+:[A-Za-z0-9._~:/?#@!$&'()*+,;=%-]+@[A-Za-z0-9.-]+",
        jurisdiction="common",
        replacement_token="[REDACTED_URL_CREDENTIALS]",
        description="URL containing embedded credentials",
    ),
    PiiPattern(
        name="ipv4",
        pattern=r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
        jurisdiction="common",
        replacement_token="[REDACTED_IP]",
        description="IPv4 address",
    ),
    PiiPattern(
        name="api_key_generic",
        pattern=r"(?i)\b(?:api[_\-]?key|access[_\-]?token|secret[_\-]?key)\s*[:=]\s*['\"]?([A-Za-z0-9_\-]{20,})['\"]?",
        jurisdiction="common",
        replacement_token="[REDACTED_API_KEY]",
        description="Generic API key or secret",
    ),
    PiiPattern(
        name="bearer_token",
        pattern=r"(?i)bearer\s+[A-Za-z0-9\-_=+/]{20,}",
        jurisdiction="common",
        replacement_token="[REDACTED_BEARER_TOKEN]",
        description="HTTP Bearer token",
    ),
]

# ── US patterns ────────────────────────────────────────────────────────────────

US_PATTERNS: list[PiiPattern] = [
    PiiPattern(
        name="ssn",
        pattern=r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",
        jurisdiction="us",
        replacement_token="[REDACTED_SSN]",
        description="US Social Security Number",
    ),
    PiiPattern(
        name="phone_us",
        pattern=r"\b(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b",
        jurisdiction="us",
        replacement_token="[REDACTED_PHONE]",
        description="US phone number",
    ),
    PiiPattern(
        name="credit_card",
        pattern=(
            r"\b(?:4[0-9]{12}(?:[0-9]{3})?|"          # Visa
            r"5[1-5][0-9]{14}|"                         # MasterCard
            r"3[47][0-9]{13}|"                           # AmEx
            r"3(?:0[0-5]|[68][0-9])[0-9]{11}|"         # Diners
            r"6(?:011|5[0-9]{2})[0-9]{12})"             # Discover
            r"\b"
        ),
        jurisdiction="us",
        replacement_token="[REDACTED_CREDIT_CARD]",
        description="Credit card number",
    ),
    PiiPattern(
        name="zip_code_us",
        pattern=r"\b\d{5}(?:-\d{4})?\b",
        jurisdiction="us",
        replacement_token="[REDACTED_ZIP]",
        description="US ZIP code",
    ),
    PiiPattern(
        name="ein",
        pattern=r"\b\d{2}-\d{7}\b",
        jurisdiction="us",
        replacement_token="[REDACTED_EIN]",
        description="US Employer Identification Number",
    ),
]

# ── EU patterns ────────────────────────────────────────────────────────────────

EU_PATTERNS: list[PiiPattern] = [
    PiiPattern(
        name="iban",
        pattern=r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]{0,16})\b",
        jurisdiction="eu",
        replacement_token="[REDACTED_IBAN]",
        description="IBAN bank account number",
    ),
    PiiPattern(
        name="vat_eu",
        pattern=r"\b(?:ATU|BE|BG|CY|CZ|DE|DK|EE|EL|ES|FI|FR|GB|HR|HU|IE|IT|LT|LU|LV|MT|NL|PL|PT|RO|SE|SI|SK)\s*\d{6,12}\b",
        jurisdiction="eu",
        replacement_token="[REDACTED_VAT]",
        description="EU VAT number",
    ),
    PiiPattern(
        name="phone_eu",
        pattern=r"\b\+(?:3[0-6]|4[0-9]|5[0-9]|7[0-4])[\s\-]?[\d\s\-]{6,14}\b",
        jurisdiction="eu",
        replacement_token="[REDACTED_PHONE]",
        description="EU phone number",
    ),
]

# ── All patterns combined ──────────────────────────────────────────────────────

ALL_PATTERNS: list[PiiPattern] = COMMON_PATTERNS + US_PATTERNS + EU_PATTERNS

PATTERNS_BY_JURISDICTION: dict[str, list[PiiPattern]] = {
    "common": COMMON_PATTERNS,
    "us": COMMON_PATTERNS + US_PATTERNS,
    "eu": COMMON_PATTERNS + EU_PATTERNS,
    "all": ALL_PATTERNS,
}


def get_patterns_for_jurisdiction(jurisdiction: str) -> list[PiiPattern]:
    """Return the PII patterns for the given *jurisdiction*.

    Parameters
    ----------
    jurisdiction:
        One of ``"common"``, ``"us"``, ``"eu"``, or ``"all"``.

    Raises
    ------
    ValueError
        If *jurisdiction* is not recognised.
    """
    if jurisdiction not in PATTERNS_BY_JURISDICTION:
        raise ValueError(
            f"Unknown jurisdiction: {jurisdiction!r}. "
            f"Choose from: {sorted(PATTERNS_BY_JURISDICTION.keys())}"
        )
    return PATTERNS_BY_JURISDICTION[jurisdiction]

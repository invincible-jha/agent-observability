"""PII-safe telemetry — detection, redaction, and privacy configuration."""
from __future__ import annotations

from agent_observability.privacy.config import CustomPatternConfig, PrivacyConfig
from agent_observability.privacy.detector import DetectionResult, PiiDetector, PiiMatch
from agent_observability.privacy.patterns import (
    ALL_PATTERNS,
    COMMON_PATTERNS,
    EU_PATTERNS,
    US_PATTERNS,
    PiiPattern,
    get_patterns_for_jurisdiction,
)
from agent_observability.privacy.redactor import PiiRedactor, PiiRedactingSpanProcessor

__all__ = [
    "PrivacyConfig",
    "CustomPatternConfig",
    "PiiDetector",
    "DetectionResult",
    "PiiMatch",
    "PiiRedactor",
    "PiiRedactingSpanProcessor",
    "PiiPattern",
    "ALL_PATTERNS",
    "COMMON_PATTERNS",
    "US_PATTERNS",
    "EU_PATTERNS",
    "get_patterns_for_jurisdiction",
]

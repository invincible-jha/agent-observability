"""PrivacyConfig — Pydantic v2 config schema for PII redaction settings."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


class CustomPatternConfig(BaseModel):
    """A user-defined regex PII pattern."""

    name: str = Field(..., description="Unique identifier for this pattern")
    pattern: str = Field(..., description="Python regex pattern string")
    replacement_token: str = Field(
        default="[REDACTED]",
        description="Token substituted for matched text",
    )
    jurisdiction: str = Field(
        default="custom",
        description="Jurisdiction label for grouping",
    )


class PrivacyConfig(BaseModel):
    """Configuration for PII detection and redaction.

    Attributes
    ----------
    enabled:
        Master switch — when ``False``, no redaction is applied.
    jurisdiction:
        Which predefined pattern set to apply.  One of ``"common"``,
        ``"us"``, ``"eu"``, ``"all"``.
    redact_span_attributes:
        When ``True`` (default), attribute values in OTel spans are scanned.
    redact_log_messages:
        When ``True``, log messages are scanned before emission.
    custom_patterns:
        Additional user-defined regex patterns to apply in addition to the
        jurisdiction patterns.
    exclude_attribute_keys:
        OTel attribute keys whose values should never be redacted.  Useful
        for trace IDs and other safe identifiers.
    full_logging_opt_in:
        Set to ``True`` to explicitly opt into full logging with NO redaction.
        Intended only for development environments with synthetic data.
    redaction_token_template:
        Override the ``[REDACTED_{NAME}]`` token template.  ``{NAME}`` is
        replaced with the upper-cased pattern name.
    """

    enabled: bool = True
    jurisdiction: Literal["common", "us", "eu", "all"] = "all"
    redact_span_attributes: bool = True
    redact_log_messages: bool = False
    custom_patterns: list[CustomPatternConfig] = Field(default_factory=list)
    exclude_attribute_keys: list[str] = Field(
        default_factory=lambda: [
            "agent.id",
            "agent.session_id",
            "agent.run_id",
            "trace_id",
            "span_id",
        ]
    )
    full_logging_opt_in: bool = False
    redaction_token_template: str = "[REDACTED_{NAME}]"

    @field_validator("redaction_token_template")
    @classmethod
    def _validate_token_template(cls, value: str) -> str:
        if "{NAME}" not in value:
            raise ValueError("redaction_token_template must contain the placeholder {NAME}")
        return value

    def effective_redaction_enabled(self) -> bool:
        """Return whether redaction is actually active (disabled by opt-in)."""
        return self.enabled and not self.full_logging_opt_in

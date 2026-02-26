"""CostAnnotator — auto-annotate OTel spans with cost data.

Intercepts LLM response objects from Anthropic and OpenAI SDKs and
stamps the active span with token counts and computed costs.
"""
from __future__ import annotations

import logging
from typing import Optional

from agent_observability.cost.pricing import estimate_cost
from agent_observability.spans.conventions import (
    LLM_COST_USD,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TOKENS_INPUT,
    LLM_TOKENS_OUTPUT,
    LLM_TOKENS_TOTAL,
)

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace as otel_trace

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    otel_trace = None  # type: ignore[assignment]


def _get_current_span() -> Optional[object]:
    """Return the currently-active OTel span, or ``None``."""
    if not _OTEL_AVAILABLE or otel_trace is None:
        return None
    span = otel_trace.get_current_span()
    # OTel returns a NonRecordingSpan when nothing is active
    if hasattr(span, "is_recording") and not span.is_recording():
        return None
    return span


class CostAnnotator:
    """Stamps the active OTel span with cost data extracted from an LLM response.

    Parameters
    ----------
    provider:
        Default provider name (e.g. ``"anthropic"``).  Can be overridden
        per call.
    default_model:
        Fallback model name when none can be extracted from the response.
    tracker:
        Optional :class:`~agent_observability.cost.tracker.CostTracker` to
        also record costs to.
    """

    def __init__(
        self,
        provider: str = "",
        default_model: str = "",
        tracker: Optional[object] = None,
    ) -> None:
        self._provider = provider
        self._default_model = default_model
        self._tracker = tracker

    # ── Anthropic SDK ─────────────────────────────────────────────────────────

    def annotate_anthropic(self, response: object, provider: str = "anthropic") -> float:
        """Extract usage from an Anthropic ``Message`` and stamp the active span.

        Parameters
        ----------
        response:
            An ``anthropic.types.Message`` object.
        provider:
            Provider label (defaults to ``"anthropic"``).

        Returns
        -------
        Computed cost in USD.
        """
        input_tokens = 0
        output_tokens = 0
        model = self._default_model

        usage = getattr(response, "usage", None)
        if usage is not None:
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0

        model = getattr(response, "model", None) or model

        return self._stamp(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    # ── OpenAI SDK ────────────────────────────────────────────────────────────

    def annotate_openai(self, response: object, provider: str = "openai") -> float:
        """Extract usage from an OpenAI ``ChatCompletion`` and stamp the active span.

        Parameters
        ----------
        response:
            An ``openai.types.chat.ChatCompletion`` object.
        provider:
            Provider label (defaults to ``"openai"``).

        Returns
        -------
        Computed cost in USD.
        """
        input_tokens = 0
        output_tokens = 0
        model = self._default_model

        usage = getattr(response, "usage", None)
        if usage is not None:
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0

        model = getattr(response, "model", None) or model

        return self._stamp(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    # ── Generic ───────────────────────────────────────────────────────────────

    def annotate(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "",
        provider: str = "",
        cached_input_tokens: int = 0,
    ) -> float:
        """Directly provide token counts to stamp on the active span.

        Returns
        -------
        Computed cost in USD.
        """
        return self._stamp(
            provider=provider or self._provider,
            model=model or self._default_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _stamp(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> float:
        cost = estimate_cost(provider, model, input_tokens, output_tokens, cached_input_tokens)
        span = _get_current_span()

        if span is not None:
            try:
                span.set_attribute(LLM_PROVIDER, provider)  # type: ignore[union-attr]
                span.set_attribute(LLM_MODEL, model)  # type: ignore[union-attr]
                span.set_attribute(LLM_TOKENS_INPUT, input_tokens)  # type: ignore[union-attr]
                span.set_attribute(LLM_TOKENS_OUTPUT, output_tokens)  # type: ignore[union-attr]
                span.set_attribute(LLM_TOKENS_TOTAL, input_tokens + output_tokens)  # type: ignore[union-attr]
                span.set_attribute(LLM_COST_USD, cost)  # type: ignore[union-attr]
            except Exception:
                logger.debug("CostAnnotator: failed to set span attributes")

        if self._tracker is not None and hasattr(self._tracker, "record"):
            try:
                self._tracker.record(  # type: ignore[union-attr]
                    provider=provider,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_input_tokens=cached_input_tokens,
                    cost_usd=cost,
                )
            except Exception:
                logger.debug("CostAnnotator: tracker.record() failed")

        return cost

"""Provider pricing data for LLM cost attribution.

All prices are in USD per **million** tokens.  Prices reflect publicly
announced rates as of early 2026 and should be treated as approximate.
Override via :func:`register_pricing` for private deployments.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """Per-million-token pricing for a single model."""

    provider: str
    model: str
    input_per_million: float
    output_per_million: float
    # Cached/prompt-cached input price if the provider offers one; else same as input
    cached_input_per_million: float = 0.0

    def compute_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> float:
        """Return total USD cost for the given token counts."""
        input_cost = (input_tokens / 1_000_000) * self.input_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_per_million
        cached_cost = (cached_input_tokens / 1_000_000) * (
            self.cached_input_per_million or self.input_per_million
        )
        return round(input_cost + output_cost + cached_cost, 8)


# ── Canonical registry ─────────────────────────────────────────────────────────
# Key: "<provider>/<model>"   Value: ModelPricing
PROVIDER_PRICING: dict[str, ModelPricing] = {
    # ── Anthropic Claude ──────────────────────────────────────────────────────
    "anthropic/claude-3-5-sonnet-20241022": ModelPricing(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        input_per_million=3.00,
        output_per_million=15.00,
        cached_input_per_million=0.30,
    ),
    "anthropic/claude-3-5-haiku-20241022": ModelPricing(
        provider="anthropic",
        model="claude-3-5-haiku-20241022",
        input_per_million=0.80,
        output_per_million=4.00,
        cached_input_per_million=0.08,
    ),
    "anthropic/claude-3-opus-20240229": ModelPricing(
        provider="anthropic",
        model="claude-3-opus-20240229",
        input_per_million=15.00,
        output_per_million=75.00,
        cached_input_per_million=1.50,
    ),
    "anthropic/claude-opus-4": ModelPricing(
        provider="anthropic",
        model="claude-opus-4",
        input_per_million=15.00,
        output_per_million=75.00,
        cached_input_per_million=1.50,
    ),
    "anthropic/claude-sonnet-4": ModelPricing(
        provider="anthropic",
        model="claude-sonnet-4",
        input_per_million=3.00,
        output_per_million=15.00,
        cached_input_per_million=0.30,
    ),
    # ── OpenAI GPT ────────────────────────────────────────────────────────────
    "openai/gpt-4o": ModelPricing(
        provider="openai",
        model="gpt-4o",
        input_per_million=2.50,
        output_per_million=10.00,
        cached_input_per_million=1.25,
    ),
    "openai/gpt-4o-mini": ModelPricing(
        provider="openai",
        model="gpt-4o-mini",
        input_per_million=0.15,
        output_per_million=0.60,
        cached_input_per_million=0.075,
    ),
    "openai/o1": ModelPricing(
        provider="openai",
        model="o1",
        input_per_million=15.00,
        output_per_million=60.00,
        cached_input_per_million=7.50,
    ),
    "openai/o3-mini": ModelPricing(
        provider="openai",
        model="o3-mini",
        input_per_million=1.10,
        output_per_million=4.40,
        cached_input_per_million=0.55,
    ),
    # ── Google Gemini ─────────────────────────────────────────────────────────
    "google/gemini-1.5-pro": ModelPricing(
        provider="google",
        model="gemini-1.5-pro",
        input_per_million=1.25,
        output_per_million=5.00,
    ),
    "google/gemini-1.5-flash": ModelPricing(
        provider="google",
        model="gemini-1.5-flash",
        input_per_million=0.075,
        output_per_million=0.30,
    ),
    "google/gemini-2.0-flash": ModelPricing(
        provider="google",
        model="gemini-2.0-flash",
        input_per_million=0.10,
        output_per_million=0.40,
    ),
    # ── Mistral ───────────────────────────────────────────────────────────────
    "mistral/mistral-large-2407": ModelPricing(
        provider="mistral",
        model="mistral-large-2407",
        input_per_million=3.00,
        output_per_million=9.00,
    ),
    "mistral/mistral-small-2409": ModelPricing(
        provider="mistral",
        model="mistral-small-2409",
        input_per_million=0.20,
        output_per_million=0.60,
    ),
    # ── Meta Llama (via various providers, estimated) ─────────────────────────
    "meta/llama-3.3-70b-instruct": ModelPricing(
        provider="meta",
        model="llama-3.3-70b-instruct",
        input_per_million=0.59,
        output_per_million=0.79,
    ),
    "meta/llama-3.1-8b-instruct": ModelPricing(
        provider="meta",
        model="llama-3.1-8b-instruct",
        input_per_million=0.06,
        output_per_million=0.06,
    ),
    # ── DeepSeek ─────────────────────────────────────────────────────────────
    "deepseek/deepseek-chat": ModelPricing(
        provider="deepseek",
        model="deepseek-chat",
        input_per_million=0.27,
        output_per_million=1.10,
    ),
    "deepseek/deepseek-reasoner": ModelPricing(
        provider="deepseek",
        model="deepseek-reasoner",
        input_per_million=0.55,
        output_per_million=2.19,
    ),
}


# ── Registry helpers ───────────────────────────────────────────────────────────

def register_pricing(pricing: ModelPricing) -> None:
    """Register or override pricing for a model.

    Parameters
    ----------
    pricing:
        A :class:`ModelPricing` dataclass.  The key is derived from
        ``{pricing.provider}/{pricing.model}``.
    """
    key = f"{pricing.provider}/{pricing.model}"
    PROVIDER_PRICING[key] = pricing


def get_pricing(provider: str, model: str) -> ModelPricing | None:
    """Return pricing for *provider*/*model*, or ``None`` if unknown.

    Falls back to a prefix match so ``"gpt-4o"`` resolves even if the caller
    does not know the exact key.
    """
    exact_key = f"{provider}/{model}"
    if exact_key in PROVIDER_PRICING:
        return PROVIDER_PRICING[exact_key]

    # Attempt prefix match (first entry whose model string starts with *model*)
    for key, pricing in PROVIDER_PRICING.items():
        if pricing.provider == provider and (
            pricing.model.startswith(model) or model.startswith(pricing.model)
        ):
            return pricing

    # Last resort: search by model name only (ignores provider)
    for pricing in PROVIDER_PRICING.values():
        if pricing.model == model or pricing.model.startswith(model):
            return pricing

    return None


def estimate_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    """Convenience: compute cost or return ``0.0`` if model is unknown."""
    pricing = get_pricing(provider, model)
    if pricing is None:
        return 0.0
    return pricing.compute_cost(input_tokens, output_tokens, cached_input_tokens)

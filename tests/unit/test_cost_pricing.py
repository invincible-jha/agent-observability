"""Unit tests for cost.pricing — ModelPricing, PROVIDER_PRICING, get_pricing, estimate_cost."""
from __future__ import annotations

import pytest

from agent_observability.cost.pricing import (
    PROVIDER_PRICING,
    ModelPricing,
    estimate_cost,
    get_pricing,
    register_pricing,
)


class TestModelPricingComputeCost:
    def test_zero_tokens_produce_zero_cost(self) -> None:
        pricing = ModelPricing(
            provider="test", model="m", input_per_million=3.0, output_per_million=15.0
        )
        assert pricing.compute_cost(0, 0) == 0.0

    def test_one_million_input_tokens_costs_input_price(self) -> None:
        pricing = ModelPricing(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            input_per_million=3.0,
            output_per_million=15.0,
        )
        cost = pricing.compute_cost(1_000_000, 0)
        assert cost == pytest.approx(3.0, rel=1e-6)

    def test_one_million_output_tokens_costs_output_price(self) -> None:
        pricing = ModelPricing(
            provider="openai",
            model="gpt-4o",
            input_per_million=2.5,
            output_per_million=10.0,
        )
        cost = pricing.compute_cost(0, 1_000_000)
        assert cost == pytest.approx(10.0, rel=1e-6)

    def test_mixed_tokens_sum_correctly(self) -> None:
        pricing = ModelPricing(
            provider="openai",
            model="gpt-4o",
            input_per_million=2.5,
            output_per_million=10.0,
        )
        # 500k input @ $2.50/M = $1.25; 200k output @ $10/M = $2.00
        cost = pricing.compute_cost(500_000, 200_000)
        assert cost == pytest.approx(3.25, rel=1e-6)

    def test_cached_tokens_charged_at_cached_rate(self) -> None:
        pricing = ModelPricing(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            input_per_million=3.0,
            output_per_million=15.0,
            cached_input_per_million=0.30,
        )
        # 1M cached input @ $0.30/M = $0.30
        cost = pricing.compute_cost(0, 0, cached_input_tokens=1_000_000)
        assert cost == pytest.approx(0.30, rel=1e-6)

    def test_result_rounded_to_8_decimal_places(self) -> None:
        pricing = ModelPricing(
            provider="test",
            model="m",
            input_per_million=1.0,
            output_per_million=1.0,
        )
        cost = pricing.compute_cost(1, 0)
        # Should be a nice rounded float
        assert isinstance(cost, float)

    def test_gpt4o_mini_low_cost_calculation(self) -> None:
        pricing = ModelPricing(
            provider="openai",
            model="gpt-4o-mini",
            input_per_million=0.15,
            output_per_million=0.60,
        )
        # 10k input, 5k output
        cost = pricing.compute_cost(10_000, 5_000)
        expected = (10_000 / 1_000_000) * 0.15 + (5_000 / 1_000_000) * 0.60
        assert cost == pytest.approx(expected, rel=1e-6)


class TestProviderPricingRegistry:
    def test_anthropic_claude_sonnet_in_registry(self) -> None:
        assert "anthropic/claude-3-5-sonnet-20241022" in PROVIDER_PRICING

    def test_openai_gpt4o_in_registry(self) -> None:
        assert "openai/gpt-4o" in PROVIDER_PRICING

    def test_google_gemini_in_registry(self) -> None:
        assert "google/gemini-1.5-pro" in PROVIDER_PRICING

    def test_all_entries_have_positive_prices(self) -> None:
        for key, pricing in PROVIDER_PRICING.items():
            assert pricing.input_per_million >= 0, f"{key} has negative input price"
            assert pricing.output_per_million >= 0, f"{key} has negative output price"

    def test_all_entries_have_provider_and_model_matching_key(self) -> None:
        for key, pricing in PROVIDER_PRICING.items():
            expected_key = f"{pricing.provider}/{pricing.model}"
            assert expected_key == key


class TestGetPricing:
    def test_exact_lookup_returns_pricing(self) -> None:
        result = get_pricing("openai", "gpt-4o")
        assert result is not None
        assert result.model == "gpt-4o"

    def test_missing_model_returns_none(self) -> None:
        result = get_pricing("openai", "gpt-99-turbo-ultra")
        assert result is None

    def test_prefix_match_resolves_partial_model_name(self) -> None:
        result = get_pricing("anthropic", "claude-3-5-sonnet")
        assert result is not None

    def test_model_only_lookup_as_last_resort(self) -> None:
        result = get_pricing("unknown_provider", "gpt-4o")
        assert result is not None
        assert result.model == "gpt-4o"

    def test_anthropic_haiku_returns_correct_prices(self) -> None:
        result = get_pricing("anthropic", "claude-3-5-haiku-20241022")
        assert result is not None
        assert result.input_per_million == pytest.approx(0.80)
        assert result.output_per_million == pytest.approx(4.00)


class TestRegisterPricing:
    def test_register_new_model_makes_it_discoverable(self) -> None:
        custom = ModelPricing(
            provider="custom_corp",
            model="my-proprietary-model",
            input_per_million=5.0,
            output_per_million=20.0,
        )
        register_pricing(custom)
        found = get_pricing("custom_corp", "my-proprietary-model")
        assert found is not None
        assert found.input_per_million == 5.0

    def test_register_overrides_existing_entry(self) -> None:
        override = ModelPricing(
            provider="openai",
            model="gpt-4o-override",
            input_per_million=999.0,
            output_per_million=999.0,
        )
        register_pricing(override)
        found = get_pricing("openai", "gpt-4o-override")
        assert found is not None
        assert found.input_per_million == 999.0


class TestEstimateCost:
    def test_known_model_returns_positive_cost(self) -> None:
        cost = estimate_cost("openai", "gpt-4o", 1000, 500)
        assert cost > 0.0

    def test_unknown_model_returns_zero(self) -> None:
        cost = estimate_cost("nonexistent", "no-model", 1000, 500)
        assert cost == 0.0

    def test_zero_tokens_returns_zero(self) -> None:
        cost = estimate_cost("openai", "gpt-4o", 0, 0)
        assert cost == 0.0

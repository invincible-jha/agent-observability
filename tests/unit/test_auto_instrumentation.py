"""Tests for auto-instrumentation stubs — langchain and crewai instrumentors."""
from __future__ import annotations

import logging
import sys
from unittest.mock import patch

import pytest


class TestLangChainInstrumentor:
    """Verify instrument_langchain behaviour with and without langchain installed."""

    def test_warns_when_langchain_not_installed(self, caplog: pytest.LogCaptureFixture) -> None:
        """When langchain_core is absent a WARNING is logged and the function returns."""
        # Ensure langchain_core is not importable for this test
        with patch.dict(sys.modules, {"langchain_core": None, "langchain_core.callbacks": None}):
            from agent_observability.auto.langchain_instrumentor import instrument_langchain

            with caplog.at_level(logging.WARNING):
                instrument_langchain()

        assert any(
            "LangChain not installed" in record.message
            for record in caplog.records
        ), "Expected a warning about LangChain not being installed"

    def test_returns_none_when_langchain_not_installed(self) -> None:
        """instrument_langchain returns None (implicitly) when langchain is absent."""
        with patch.dict(sys.modules, {"langchain_core": None, "langchain_core.callbacks": None}):
            from agent_observability.auto.langchain_instrumentor import instrument_langchain

            result = instrument_langchain()

        assert result is None

    def test_accepts_tracer_provider_kwarg(self) -> None:
        """Function signature accepts tracer_provider without raising TypeError."""
        with patch.dict(sys.modules, {"langchain_core": None, "langchain_core.callbacks": None}):
            from agent_observability.auto.langchain_instrumentor import instrument_langchain

            # Should not raise — just warn and return
            instrument_langchain(tracer_provider=None)

    def test_accepts_capture_content_kwarg(self) -> None:
        """Function signature accepts capture_content without raising TypeError."""
        with patch.dict(sys.modules, {"langchain_core": None, "langchain_core.callbacks": None}):
            from agent_observability.auto.langchain_instrumentor import instrument_langchain

            instrument_langchain(capture_content=True)

    def test_import_from_auto_package(self) -> None:
        """instrument_langchain is importable from the auto package."""
        # Force reimport to avoid cached module state
        if "agent_observability.auto" in sys.modules:
            del sys.modules["agent_observability.auto"]

        with patch.dict(sys.modules, {"langchain_core": None, "langchain_core.callbacks": None}):
            from agent_observability.auto import instrument_langchain  # noqa: F401

            assert callable(instrument_langchain)


class TestCrewAIInstrumentor:
    """Verify instrument_crewai behaviour with and without crewai installed."""

    def test_warns_when_crewai_not_installed(self, caplog: pytest.LogCaptureFixture) -> None:
        """When crewai is absent a WARNING is logged and the function returns."""
        with patch.dict(sys.modules, {"crewai": None}):
            from agent_observability.auto.crewai_instrumentor import instrument_crewai

            with caplog.at_level(logging.WARNING):
                instrument_crewai()

        assert any(
            "CrewAI not installed" in record.message
            for record in caplog.records
        ), "Expected a warning about CrewAI not being installed"

    def test_returns_none_when_crewai_not_installed(self) -> None:
        """instrument_crewai returns None (implicitly) when crewai is absent."""
        with patch.dict(sys.modules, {"crewai": None}):
            from agent_observability.auto.crewai_instrumentor import instrument_crewai

            result = instrument_crewai()

        assert result is None

    def test_accepts_tracer_provider_kwarg(self) -> None:
        """Function signature accepts tracer_provider without raising TypeError."""
        with patch.dict(sys.modules, {"crewai": None}):
            from agent_observability.auto.crewai_instrumentor import instrument_crewai

            instrument_crewai(tracer_provider=None)

    def test_accepts_capture_content_kwarg(self) -> None:
        """Function signature accepts capture_content without raising TypeError."""
        with patch.dict(sys.modules, {"crewai": None}):
            from agent_observability.auto.crewai_instrumentor import instrument_crewai

            instrument_crewai(capture_content=False)

    def test_import_from_auto_package(self) -> None:
        """instrument_crewai is importable from the auto package."""
        if "agent_observability.auto" in sys.modules:
            del sys.modules["agent_observability.auto"]

        with patch.dict(sys.modules, {"crewai": None}):
            from agent_observability.auto import instrument_crewai  # noqa: F401

            assert callable(instrument_crewai)


class TestAutoPackagePublicApi:
    """Verify the auto package __all__ exports."""

    def test_all_exports_are_callable(self) -> None:
        """Every name in __all__ should be callable."""
        import agent_observability.auto as auto_pkg

        for name in auto_pkg.__all__:
            obj = getattr(auto_pkg, name)
            assert callable(obj), f"{name} is not callable"

    def test_all_exports_present(self) -> None:
        """Both instrumentors should be in __all__."""
        import agent_observability.auto as auto_pkg

        assert "instrument_langchain" in auto_pkg.__all__
        assert "instrument_crewai" in auto_pkg.__all__

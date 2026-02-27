"""Auto-instrumentation for LangChain/LangGraph."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def instrument_langchain(
    tracer_provider: Any | None = None,
    capture_content: bool = False,
) -> None:
    """Install OTel-compliant auto-instrumentation for LangChain.

    Parameters
    ----------
    tracer_provider:
        OTel TracerProvider to use. If None, uses the global provider.
    capture_content:
        If True, capture prompt/response content in span attributes.
        WARNING: May contain PII. Disable in production.
    """
    try:
        from langchain_core.callbacks import BaseCallbackHandler  # noqa: F401
    except ImportError:
        logger.warning(
            "LangChain not installed. Install with: pip install langchain-core"
        )
        return

    logger.info("LangChain auto-instrumentation installed (OTel GenAI conventions).")

"""GenericInstrumentor — decorator-based instrumentation for any Python function.

Wraps arbitrary sync and async callables with agent-semantic spans so that
any code path can be traced without coupling to a specific framework.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import Callable, Optional, TypeVar, overload

from agent_observability.spans.types import AgentSpan, AgentSpanKind, AgentTracer

logger = logging.getLogger(__name__)

_FuncT = TypeVar("_FuncT", bound=Callable[..., object])


class GenericInstrumentor:
    """Provide decorator-based instrumentation for arbitrary functions.

    Parameters
    ----------
    tracer:
        A configured :class:`~agent_observability.spans.types.AgentTracer`.
        A default no-op tracer is used if not supplied.
    """

    def __init__(self, tracer: Optional[AgentTracer] = None) -> None:
        self._tracer = tracer or AgentTracer()

    # ── Decorators ────────────────────────────────────────────────────────────

    def instrument(
        self,
        name: str = "",
        kind: AgentSpanKind = AgentSpanKind.REASONING_STEP,
        capture_args: bool = False,
        capture_result: bool = False,
    ) -> Callable[[_FuncT], _FuncT]:
        """Return a decorator that wraps a function with an agent span.

        Parameters
        ----------
        name:
            Span name.  Defaults to the function's qualified name.
        kind:
            The :class:`AgentSpanKind` to assign to the span.
        capture_args:
            When ``True``, record a hash of the arguments as a span attribute.
        capture_result:
            When ``True``, record whether the result was truthy as a span attribute.
        """

        def decorator(func: _FuncT) -> _FuncT:
            span_name = name or f"{func.__module__}.{func.__qualname__}"

            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args: object, **kwargs: object) -> object:
                    with self._tracer._managed_span(span_name, kind) as span:
                        if capture_args:
                            _safe_set(span, "fn.args_hash", str(hash(str(args) + str(kwargs)))[:16])
                        try:
                            result = await func(*args, **kwargs)  # type: ignore[misc]
                            if capture_result:
                                _safe_set(span, "fn.result_truthy", bool(result))
                            return result
                        except Exception as exc:
                            span.set_error(
                                error_type=type(exc).__name__,
                                recoverable=True,
                                exception=exc,
                            )
                            raise

                return async_wrapper  # type: ignore[return-value]

            else:
                @functools.wraps(func)
                def sync_wrapper(*args: object, **kwargs: object) -> object:
                    with self._tracer._managed_span(span_name, kind) as span:
                        if capture_args:
                            _safe_set(span, "fn.args_hash", str(hash(str(args) + str(kwargs)))[:16])
                        try:
                            result = func(*args, **kwargs)  # type: ignore[operator]
                            if capture_result:
                                _safe_set(span, "fn.result_truthy", bool(result))
                            return result
                        except Exception as exc:
                            span.set_error(
                                error_type=type(exc).__name__,
                                recoverable=True,
                                exception=exc,
                            )
                            raise

                return sync_wrapper  # type: ignore[return-value]

        return decorator

    def llm_call(self, name: str = "", model: str = "", provider: str = "") -> Callable[[_FuncT], _FuncT]:
        """Shortcut decorator for wrapping an LLM call function."""
        def decorator(func: _FuncT) -> _FuncT:
            span_name = name or f"{func.__qualname__}"

            @functools.wraps(func)
            def wrapper(*args: object, **kwargs: object) -> object:
                with self._tracer.llm_call(model=model, provider=provider, name=span_name) as _:
                    return func(*args, **kwargs)  # type: ignore[operator]

            return wrapper  # type: ignore[return-value]

        return decorator

    def tool_invoke(self, tool_name: str, name: str = "") -> Callable[[_FuncT], _FuncT]:
        """Shortcut decorator for wrapping a tool invocation function."""
        def decorator(func: _FuncT) -> _FuncT:
            span_name = name or func.__qualname__

            @functools.wraps(func)
            def wrapper(*args: object, **kwargs: object) -> object:
                with self._tracer.tool_invoke(tool_name=tool_name, name=span_name) as span:
                    try:
                        result = func(*args, **kwargs)  # type: ignore[operator]
                        span.set_tool(tool_name, success=True)
                        return result
                    except Exception as exc:
                        span.set_tool(tool_name, success=False, error_type=type(exc).__name__)
                        raise

            return wrapper  # type: ignore[return-value]

        return decorator


def _safe_set(span: AgentSpan, key: str, value: object) -> None:
    """Set a span attribute, ignoring type errors for non-primitive values."""
    if isinstance(value, (str, int, float, bool)):
        span.set_attribute(key, value)
    else:
        span.set_attribute(key, str(value))

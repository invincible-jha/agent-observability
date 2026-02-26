"""agent-observability — OpenTelemetry-native agent tracing, cost attribution, and drift detection.

Public API
----------
The stable public surface is everything exported from this module.
Anything inside submodules not re-exported here is considered private
and may change without notice.

Example
-------
>>> import agent_observability
>>> agent_observability.__version__
'0.1.0'
"""
from __future__ import annotations

__version__: str = "0.1.0"
__all__ = ["__version__"]

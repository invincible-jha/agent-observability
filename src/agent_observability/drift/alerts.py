"""DriftAlertManager — fire alerts when agent drift is detected.

Supports three output channels:
- Console (Rich/logging)
- Application log (Python ``logging``)
- Webhook (HTTP POST)
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Callable, Optional, Sequence

from agent_observability.drift.detector import DriftResult

logger = logging.getLogger(__name__)


AlertHandler = Callable[["DriftAlert"], None]


@dataclass
class DriftAlert:
    """An alert fired when drift is detected."""

    agent_id: str
    severity: str  # "low", "medium", "high"
    max_z_score: float
    threshold: float
    drifted_features: dict[str, float]
    baseline_age_seconds: float
    window_span_count: int
    timestamp: float
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "severity": self.severity,
            "max_z_score": self.max_z_score,
            "threshold": self.threshold,
            "drifted_features": self.drifted_features,
            "baseline_age_seconds": self.baseline_age_seconds,
            "window_span_count": self.window_span_count,
            "timestamp": self.timestamp,
            "message": self.message,
        }


def _result_to_alert(result: DriftResult) -> DriftAlert:
    top_features = sorted(
        result.drifted_features.items(), key=lambda x: abs(x[1]), reverse=True
    )[:3]
    feature_summary = ", ".join(
        f"{k}={v:.2f}σ" for k, v in top_features
    )
    message = (
        f"Agent '{result.agent_id}' drifted ({result.severity.upper()}): "
        f"max_z={result.max_z_score:.2f}σ. "
        f"Top features: {feature_summary or 'none'}."
    )
    return DriftAlert(
        agent_id=result.agent_id,
        severity=result.severity,
        max_z_score=result.max_z_score,
        threshold=result.threshold,
        drifted_features=result.drifted_features,
        baseline_age_seconds=result.baseline_age_seconds,
        window_span_count=result.window_span_count,
        timestamp=result.timestamp,
        message=message,
    )


class ConsoleAlertHandler:
    """Write alerts to the Python logger (or Rich console)."""

    def __call__(self, alert: DriftAlert) -> None:
        log_fn = {
            "low": logger.info,
            "medium": logger.warning,
            "high": logger.error,
        }.get(alert.severity, logger.warning)
        log_fn("[DRIFT ALERT] %s", alert.message)


class WebhookAlertHandler:
    """HTTP POST the alert payload to a webhook URL.

    Parameters
    ----------
    url:
        The webhook endpoint to POST to.
    timeout_seconds:
        Request timeout in seconds.
    extra_headers:
        Additional HTTP headers to include (e.g. authorization).
    """

    def __init__(
        self,
        url: str,
        timeout_seconds: float = 5.0,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> None:
        self._url = url
        self._timeout = timeout_seconds
        self._extra_headers = extra_headers or {}

    def __call__(self, alert: DriftAlert) -> None:
        payload = json.dumps(alert.to_dict()).encode("utf-8")
        headers = {"Content-Type": "application/json", **self._extra_headers}
        req = urllib.request.Request(self._url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as response:
                status = response.status
                if status >= 400:
                    logger.warning(
                        "WebhookAlertHandler: webhook returned HTTP %d", status
                    )
                else:
                    logger.debug("WebhookAlertHandler: alert sent (HTTP %d)", status)
        except Exception:
            logger.exception(
                "WebhookAlertHandler: failed to send alert to %s", self._url
            )


class DriftAlertManager:
    """Evaluate :class:`DriftResult` objects and dispatch alerts to handlers.

    Parameters
    ----------
    handlers:
        A sequence of callables accepting a :class:`DriftAlert`.  Defaults to
        a single :class:`ConsoleAlertHandler`.
    min_severity:
        Minimum severity level to fire an alert for.  Options: ``"low"``,
        ``"medium"``, ``"high"``.
    cooldown_seconds:
        Suppress repeated alerts for the same agent within this window.
    """

    _SEVERITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}

    def __init__(
        self,
        handlers: Optional[Sequence[AlertHandler]] = None,
        min_severity: str = "low",
        cooldown_seconds: float = 300.0,
    ) -> None:
        self._handlers: list[AlertHandler] = list(handlers or [ConsoleAlertHandler()])
        self._min_severity = min_severity
        self._cooldown = cooldown_seconds
        # {agent_id: last_alert_timestamp}
        self._last_alert: dict[str, float] = {}

    def add_handler(self, handler: AlertHandler) -> None:
        """Register an additional alert handler."""
        self._handlers.append(handler)

    def process(self, result: DriftResult) -> Optional[DriftAlert]:
        """Evaluate *result* and fire an alert if warranted.

        Returns
        -------
        The :class:`DriftAlert` that was fired, or ``None`` if no alert was sent.
        """
        if not result.drifted:
            return None

        min_rank = self._SEVERITY_ORDER.get(self._min_severity, 1)
        result_rank = self._SEVERITY_ORDER.get(result.severity, 0)
        if result_rank < min_rank:
            return None

        now = time.time()
        last = self._last_alert.get(result.agent_id, 0.0)
        if now - last < self._cooldown:
            logger.debug(
                "DriftAlertManager: suppressing alert for %s (cooldown)",
                result.agent_id,
            )
            return None

        alert = _result_to_alert(result)
        self._last_alert[result.agent_id] = now

        for handler in self._handlers:
            try:
                handler(alert)
            except Exception:
                logger.exception("DriftAlertManager: handler %s raised an exception", handler)

        return alert

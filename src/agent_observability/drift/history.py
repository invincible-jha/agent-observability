"""BaselineHistory — store and retrieve historical baselines.

Provides an in-memory store with optional JSON persistence.  Baselines are
keyed by ``(agent_id, version)`` where version increments automatically.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from agent_observability.drift.baseline import AgentBaseline, BaselineStats

logger = logging.getLogger(__name__)


def _baseline_to_dict(baseline: AgentBaseline) -> dict[str, object]:
    """Serialise an AgentBaseline to a JSON-compatible dict."""
    return {
        "agent_id": baseline.agent_id,
        "created_at": baseline.created_at,
        "span_count": baseline.span_count,
        "metadata": baseline.metadata,
        "feature_stats": {
            key: {
                "mean": stats.mean,
                "std_dev": stats.std_dev,
                "sample_count": stats.sample_count,
            }
            for key, stats in baseline.feature_stats.items()
        },
    }


def _baseline_from_dict(d: dict[str, object]) -> AgentBaseline:
    """Deserialise an AgentBaseline from a JSON-compatible dict."""
    feature_stats_raw = d.get("feature_stats", {})
    feature_stats: dict[str, BaselineStats] = {}
    if isinstance(feature_stats_raw, dict):
        for key, raw in feature_stats_raw.items():
            if isinstance(raw, dict):
                feature_stats[key] = BaselineStats(
                    mean=float(raw.get("mean", 0.0)),
                    std_dev=float(raw.get("std_dev", 0.0)),
                    sample_count=int(raw.get("sample_count", 0)),
                )

    metadata = d.get("metadata", {})
    return AgentBaseline(
        agent_id=str(d.get("agent_id", "")),
        created_at=float(d.get("created_at", 0.0)),
        span_count=int(d.get("span_count", 0)),
        feature_stats=feature_stats,
        metadata=metadata if isinstance(metadata, dict) else {},
    )


class BaselineHistory:
    """Thread-safe store for versioned agent baselines.

    Parameters
    ----------
    persist_path:
        Optional path to a JSON file for persistence across process restarts.
        Pass ``None`` (default) for a pure in-memory store.
    max_per_agent:
        Maximum number of historical baselines to keep per agent (oldest pruned).
    """

    def __init__(
        self,
        persist_path: Optional[str] = None,
        max_per_agent: int = 10,
    ) -> None:
        # {agent_id: [baseline, ...]} ordered oldest → newest
        self._store: dict[str, list[AgentBaseline]] = {}
        self._lock = threading.Lock()
        self._max_per_agent = max_per_agent
        self._persist_path = persist_path

        if persist_path and Path(persist_path).exists():
            self._load(persist_path)

    # ── Write ─────────────────────────────────────────────────────────────────

    def save(self, baseline: AgentBaseline) -> None:
        """Save *baseline* and optionally persist to disk."""
        with self._lock:
            history = self._store.setdefault(baseline.agent_id, [])
            history.append(baseline)
            if len(history) > self._max_per_agent:
                history.pop(0)

        if self._persist_path:
            self._flush()

        logger.debug(
            "BaselineHistory: saved baseline for agent=%s (total=%d)",
            baseline.agent_id,
            len(self._store.get(baseline.agent_id, [])),
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def latest(self, agent_id: str) -> Optional[AgentBaseline]:
        """Return the most recently saved baseline for *agent_id*."""
        with self._lock:
            history = self._store.get(agent_id, [])
            return history[-1] if history else None

    def all_for_agent(self, agent_id: str) -> list[AgentBaseline]:
        """Return all stored baselines for *agent_id*, oldest first."""
        with self._lock:
            return list(self._store.get(agent_id, []))

    def all_agent_ids(self) -> list[str]:
        """Return a sorted list of all agent IDs with stored baselines."""
        with self._lock:
            return sorted(self._store.keys())

    def delete_agent(self, agent_id: str) -> None:
        """Remove all baselines for *agent_id*."""
        with self._lock:
            self._store.pop(agent_id, None)
        if self._persist_path:
            self._flush()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _flush(self) -> None:
        assert self._persist_path is not None
        with self._lock:
            payload = {
                agent_id: [_baseline_to_dict(b) for b in baselines]
                for agent_id, baselines in self._store.items()
            }
        try:
            path = Path(self._persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            logger.exception("BaselineHistory: failed to persist to %s", self._persist_path)

    def _load(self, path: str) -> None:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            with self._lock:
                for agent_id, history_raw in raw.items():
                    if isinstance(history_raw, list):
                        self._store[agent_id] = [
                            _baseline_from_dict(b)
                            for b in history_raw
                            if isinstance(b, dict)
                        ]
        except Exception:
            logger.exception("BaselineHistory: failed to load from %s", path)

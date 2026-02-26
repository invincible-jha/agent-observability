"""PiiDetector — regex-based PII detection in span attributes and strings."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern, Sequence

from agent_observability.privacy.patterns import (
    PiiPattern,
    get_patterns_for_jurisdiction,
)


@dataclass
class PiiMatch:
    """A single PII finding within a text value."""

    pattern_name: str
    replacement_token: str
    start: int
    end: int
    matched_text: str


@dataclass
class DetectionResult:
    """Result of scanning a single text value for PII."""

    original: str
    matches: list[PiiMatch]

    @property
    def has_pii(self) -> bool:
        return bool(self.matches)

    @property
    def pattern_names(self) -> list[str]:
        return [m.pattern_name for m in self.matches]


class PiiDetector:
    """Scan text values for PII using compiled regex patterns.

    Parameters
    ----------
    jurisdiction:
        Which predefined pattern set to use (``"common"``, ``"us"``,
        ``"eu"``, ``"all"``).
    extra_patterns:
        Additional :class:`~agent_observability.privacy.patterns.PiiPattern`
        objects to include beyond the jurisdiction defaults.
    """

    def __init__(
        self,
        jurisdiction: str = "all",
        extra_patterns: Sequence[PiiPattern] | None = None,
    ) -> None:
        base = get_patterns_for_jurisdiction(jurisdiction)
        all_patterns = list(base) + list(extra_patterns or [])
        # De-duplicate by name (last writer wins)
        seen: dict[str, PiiPattern] = {}
        for p in all_patterns:
            seen[p.name] = p
        self._patterns: list[tuple[PiiPattern, Pattern[str]]] = [
            (p, p.compiled()) for p in seen.values()
        ]

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self, text: str) -> DetectionResult:
        """Scan *text* for PII and return all matches.

        Parameters
        ----------
        text:
            The string to scan.

        Returns
        -------
        A :class:`DetectionResult` containing all :class:`PiiMatch` objects.
        """
        matches: list[PiiMatch] = []
        for pii_pattern, compiled in self._patterns:
            for m in compiled.finditer(text):
                matches.append(
                    PiiMatch(
                        pattern_name=pii_pattern.name,
                        replacement_token=pii_pattern.replacement_token,
                        start=m.start(),
                        end=m.end(),
                        matched_text=m.group(0),
                    )
                )
        # Sort by position for deterministic output
        matches.sort(key=lambda x: x.start)
        return DetectionResult(original=text, matches=matches)

    def has_pii(self, text: str) -> bool:
        """Return ``True`` if *text* contains any recognised PII."""
        for _, compiled in self._patterns:
            if compiled.search(text):
                return True
        return False

    def scan_dict(
        self,
        data: dict[str, object],
        exclude_keys: Sequence[str] | None = None,
    ) -> dict[str, DetectionResult]:
        """Scan all string values in a flat dict.

        Parameters
        ----------
        data:
            A flat ``{key: value}`` dict (e.g. span attributes).
        exclude_keys:
            Keys to skip during scanning.

        Returns
        -------
        ``{key: DetectionResult}`` for every string value that was scanned.
        """
        excluded = set(exclude_keys or [])
        results: dict[str, DetectionResult] = {}
        for key, value in data.items():
            if key in excluded or not isinstance(value, str):
                continue
            result = self.detect(value)
            if result.has_pii:
                results[key] = result
        return results

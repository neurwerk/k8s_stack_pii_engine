"""Typed analysis failures that cross service-layer boundaries."""

from __future__ import annotations

from pii_engine.models.contracts import LimitDetail


class InvalidAnalysisRequestError(ValueError):
    """Raise when a validated protocol request cannot be analyzed safely."""


class AnalysisRequestTooLargeError(ValueError):
    """Raise when an analysis request exceeds a configured size limit."""

    def __init__(self, message: str = "", *, limit: LimitDetail | None = None) -> None:
        """Preserve actual measurements without deriving them from exception text."""
        super().__init__(message)
        self.limit = limit

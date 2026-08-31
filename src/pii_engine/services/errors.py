"""Typed analysis failures that cross service-layer boundaries."""

from __future__ import annotations


class InvalidAnalysisRequestError(ValueError):
    """Raise when a validated protocol request cannot be analyzed safely."""


class AnalysisRequestTooLargeError(ValueError):
    """Raise when an analysis request exceeds a configured size limit."""

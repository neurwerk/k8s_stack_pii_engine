"""Presidio-backed ordinary anonymization operators."""

from __future__ import annotations

from typing import Protocol

from pii_engine.services.analyzer import EntityMatch


class Anonymizer(Protocol):
    """Apply one ordinary transformation to one independently positioned span."""

    def apply(self, text: str, match: EntityMatch, action: str, params: dict[str, object]) -> str:
        """Return the transformed complete text."""


class PresidioAnonymizer:
    """Use upstream Presidio operators for mask, replace, redact, and encrypt."""

    def __init__(self, encryption_key: str) -> None:
        """Create one reusable in-process engine with its runtime encryption key."""
        from presidio_anonymizer import AnonymizerEngine

        self._engine = AnonymizerEngine()
        self._encryption_key = encryption_key

    def apply(self, text: str, match: EntityMatch, action: str, params: dict[str, object]) -> str:
        """Apply one Presidio operator while preserving all unaffected offsets."""
        from presidio_anonymizer.entities.engine import OperatorConfig, RecognizerResult

        operator_params = dict(params)
        if action == "encrypt":
            operator_params = {"key": self._encryption_key}
        result = self._engine.anonymize(
            text=text,
            analyzer_results=[
                RecognizerResult(
                    entity_type=match.entity_type,
                    start=match.start,
                    end=match.end,
                    score=match.score,
                )
            ],
            operators={match.entity_type: OperatorConfig(action, operator_params)},
        )
        return result.text


class TestAnonymizer:
    """Dependency-free operator equivalent used only by explicit unit-test runtime."""

    def apply(self, text: str, match: EntityMatch, action: str, params: dict[str, object]) -> str:
        """Transform one span using the same externally visible semantics."""
        original = text[match.start : match.end]
        if action == "mask":
            character = str(params.get("masking_char", "*"))
            configured_count = params.get("chars_to_mask", len(original))
            if not isinstance(configured_count, int):
                raise TypeError("chars_to_mask must be an integer")
            count = min(configured_count, len(original))
            from_end = bool(params.get("from_end", True))
            masked = character * count
            replacement = original[:-count] + masked if from_end else masked + original[count:]
        elif action == "replace":
            replacement = str(params.get("new_value", f"<{match.entity_type}>"))
        elif action == "redact":
            replacement = ""
        elif action == "encrypt":
            replacement = original.encode().hex()
        else:
            raise ValueError(f"unsupported anonymizer action: {action}")
        return text[: match.start] + replacement + text[match.end :]

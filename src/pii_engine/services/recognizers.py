"""Normalized deterministic and customer-defined Presidio recognizers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pii_engine.lib.catalog import STEUERNUMMER_PATTERNS

if TYPE_CHECKING:
    from pii_engine.config.policy import CustomRecognizer


def normalized_transformers_recognizer(entities: list[str], language: str) -> Any:  # noqa: ANN401
    """Expose normalized transformer outputs for one configured language."""
    from presidio_analyzer.predefined_recognizers.nlp_engine_recognizers import (
        transformers_recognizer,
    )

    return transformers_recognizer.TransformersRecognizer(
        supported_entities=entities,
        supported_language=language,
    )


def normalized_recognizers(languages: tuple[str, ...]) -> list[Any]:
    """Create stable DE/NL/EN recognizers for every configured language."""
    from presidio_analyzer import Pattern, PatternRecognizer

    recognizers: list[Any] = []
    for language in languages:
        recognizers.extend(
            [
                _bsn_recognizer(language),
                PatternRecognizer(
                    supported_entity="POSTAL_CODE",
                    patterns=[
                        Pattern("NL or DE postal code", r"\b(?:\d{4}\s?[A-Z]{2}|\d{5})\b", 0.55)
                    ],
                    supported_language=language,
                ),
                PatternRecognizer(
                    supported_entity="VAT_NUMBER",
                    patterns=[
                        Pattern("Dutch VAT", r"\bNL\d{9}B\d{2}\b", 0.85),
                        Pattern("German VAT", r"\bDE\d{9}\b", 0.85),
                    ],
                    supported_language=language,
                ),
                PatternRecognizer(
                    supported_entity="STEUERNUMMER",
                    patterns=[
                        Pattern(name, pattern, 0.95) for name, pattern in STEUERNUMMER_PATTERNS
                    ],
                    supported_language=language,
                ),
                PatternRecognizer(
                    supported_entity="NATIONAL_ID_NUMBER",
                    patterns=[Pattern("German ID", r"\b[A-Z0-9]{9}\b", 0.35)],
                    context=["personalausweis", "ausweis", "id nummer"],
                    supported_language=language,
                ),
                PatternRecognizer(
                    supported_entity="PASSWORD_OR_SECRET",
                    patterns=[
                        Pattern(
                            "Assigned secret",
                            r"(?i)\b(?:password|passwort|api[ _-]?key|secret)\s*[:=]\s*\S+",
                            0.9,
                        )
                    ],
                    supported_language=language,
                ),
            ]
        )
    return recognizers


def custom_recognizers(definitions: list[CustomRecognizer]) -> list[Any]:
    """Build customer recognizers already validated by policy models."""
    from presidio_analyzer import Pattern, PatternRecognizer

    recognizers: list[Any] = []
    for definition in definitions:
        for language in definition.supported_languages:
            recognizers.append(
                PatternRecognizer(
                    supported_entity=definition.entity,
                    patterns=[Pattern(definition.name, definition.regex, definition.score)],
                    supported_language=language,
                )
            )
    return recognizers


def _bsn_recognizer(language: str) -> Any:  # noqa: ANN401
    """Create a Presidio recognizer applying the Dutch eleven-test."""
    from presidio_analyzer import Pattern, PatternRecognizer

    class BsnRecognizer(PatternRecognizer):
        """Reject nine-digit candidates that fail the BSN checksum."""

        def validate_result(self, pattern_text: str) -> bool:
            weights = (9, 8, 7, 6, 5, 4, 3, 2, -1)
            return (
                sum(
                    int(digit) * weight for digit, weight in zip(pattern_text, weights, strict=True)
                )
                % 11
                == 0
            )

    return cast(
        Any,
        BsnRecognizer(
            supported_entity="BSN",
            patterns=[Pattern("Dutch BSN", r"\b\d{9}\b", 0.85)],
            context=["bsn", "burgerservicenummer"],
            supported_language=language,
        ),
    )

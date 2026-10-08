"""Production Presidio rules without an NLP engine or trained pipeline.

Context uses casefolded Unicode words, five preceding non-stop words per active
language and no following words, with Presidio's +0.35 boost and 0.4 floor.
Unlike learned lemmatization, only explicit inflections and small fixed stop-word
lists are recognized; phrases must be contiguous content words. This deliberately
avoids substring boosts ("telephone" inside an unrelated word) while supporting
compound context phrases such as "id nummer". It is not morphological equivalence
for arbitrary inflections, compounds or language-model-specific stop words.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_left
from typing import TYPE_CHECKING, Any

from pii_engine.lib.catalog import ENTITY_CATALOG
from pii_engine.services.analyzer import EntityMatch
from pii_engine.services.recognizers import custom_recognizers, normalized_recognizers

if TYPE_CHECKING:
    from pii_engine.config.policy import PolicySettings

_NORMALIZED = {
    "CREDIT_CARD": "CREDIT_CARD_NUMBER",
    "IBAN_CODE": "IBAN",
    "US_BANK_NUMBER": "BANK_ACCOUNT",
    "US_DRIVER_LICENSE": "DRIVERS_LICENSE_NUMBER",
    "US_ITIN": "TAX_ID",
    "US_PASSPORT": "PASSPORT_NUMBER",
    "US_SSN": "NATIONAL_ID_NUMBER",
    "UK_NHS": "HEALTH_INSURANCE_ID",
}
# DATE_TIME, URL, CRYPTO and MEDICAL_LICENSE have no exact normalized policy
# entity. In particular, an arbitrary date is not evidence of a date of birth.
_CONTEXT = {
    "PHONE_NUMBER": {
        "en": ("phone", "telephone", "mobile", "call"),
        "de": ("telefon", "telefonnummer", "handy", "rufnummer"),
        "nl": ("telefoon", "telefoonnummer", "mobiel", "nummer"),
    },
    "NATIONAL_ID_NUMBER": {
        "en": ("national id", "id number", "identity card"),
        "de": ("personalausweis", "personalausweisnummer", "ausweis", "ausweisnummer", "id nummer"),
        "nl": ("identiteitskaart", "identiteitsnummer", "id nummer"),
    },
    "BANK_ACCOUNT": {
        "de": ("konto", "kontonummer", "bankkonto"),
        "nl": ("rekening", "rekeningnummer", "bankrekening"),
    },
    "PASSPORT_NUMBER": {"de": ("reisepass", "passnummer"), "nl": ("paspoort",)},
    "DRIVERS_LICENSE_NUMBER": {"de": ("führerschein",), "nl": ("rijbewijs",)},
    "TAX_ID": {"de": ("steuer", "steuerid", "steueridentifikationsnummer"), "nl": ("belasting",)},
    "CREDIT_CARD_NUMBER": {"de": ("kreditkarte",), "nl": ("creditcard",)},
}
_STOP_WORDS = {
    "en": frozenset(
        [
            "a",
            "an",
            "the",
            "my",
            "your",
            "his",
            "her",
            "our",
            "their",
            "is",
            "are",
            "was",
            "were",
            "of",
            "for",
            "to",
            "and",
            "in",
            "on",
        ]
    ),
    "de": frozenset(
        [
            "der",
            "die",
            "das",
            "des",
            "dem",
            "den",
            "ein",
            "eine",
            "einer",
            "eines",
            "mein",
            "meine",
            "ist",
            "sind",
            "von",
            "für",
            "und",
            "im",
        ]
    ),
    "nl": frozenset(
        [
            "de",
            "het",
            "een",
            "mijn",
            "jouw",
            "zijn",
            "haar",
            "ons",
            "is",
            "was",
            "van",
            "voor",
            "en",
            "in",
            "op",
        ]
    ),
}
_INFLECTIONS = {
    "phones": "phone",
    "numbers": "number",
    "telephones": "telephone",
    "telephoned": "telephone",
    "cards": "card",
    "accounts": "account",
    "passports": "passport",
    "licenses": "license",
    "identities": "identity",
    "called": "call",
    "calling": "call",
    "ausweise": "ausweis",
    "ausweises": "ausweis",
    "ausweisen": "ausweis",
    "personalausweise": "personalausweis",
    "personalausweises": "personalausweis",
    "personalausweisen": "personalausweis",
    "telefonnummern": "telefonnummer",
    "rufnummern": "rufnummer",
    "konten": "konto",
    "kreditkarten": "kreditkarte",
    "reisepässe": "reisepass",
    "reisepasses": "reisepass",
    "telefoonnummers": "telefoonnummer",
    "nummers": "nummer",
    "identiteitskaarten": "identiteitskaart",
    "rekeningen": "rekening",
    "rekeningnummers": "rekeningnummer",
    "paspoorten": "paspoort",
}
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


class RulesAnalyzer:
    """Run validated deterministic recognizers once over the complete text."""

    def __init__(self, policy: PolicySettings) -> None:
        """Build only CPU pattern/checksum recognizers, never an AnalyzerEngine."""
        from presidio_analyzer import PatternRecognizer, RecognizerRegistry
        from presidio_analyzer.predefined_recognizers import PhoneRecognizer

        self.policy = policy
        self._validate_policy(policy)
        registry = RecognizerRegistry(supported_languages=["en", "de", "nl"])
        # This constructs recognizers, not an NLP engine; discard NLP recognizers
        # before any load/analyze call. Include English country-specific rules
        # even when the configured text NER language is German or Dutch.
        registry.load_predefined_recognizers(languages=["en", "de", "nl"])
        self._rules: list[Any] = []
        seen: set[tuple[type, str]] = set()
        for recognizer in registry.recognizers:
            key = (type(recognizer), recognizer.name)
            if (
                isinstance(recognizer, (PatternRecognizer, PhoneRecognizer))
                and recognizer.name != "EmailRecognizer"
                and key not in seen
                and any(
                    _normalize(entity) in ENTITY_CATALOG for entity in recognizer.supported_entities
                )
            ):
                seen.add(key)
                if isinstance(recognizer, PhoneRecognizer):
                    recognizer.supported_regions = (*recognizer.supported_regions, "NL")
                self._rules.append(recognizer)
        self._rules.extend(normalized_recognizers(("en",)))
        self._custom = [
            (definition, custom_recognizers([definition])[0])
            for definition in policy.pii.custom_recognizers
        ]
        self._custom_ids = {id(recognizer) for _, recognizer in self._custom}

    def analyze(self, text: str, policy: PolicySettings | None = None) -> list[EntityMatch]:
        """Preserve original offsets, validation, context boosts and selection."""
        active = policy or self.policy
        self._validate_policy(active)
        if active.pii.custom_recognizers != self.policy.pii.custom_recognizers or not set(
            active.pii.analyzer_languages
        ).issubset(self.policy.pii.supported_languages):
            raise ValueError("request policy changes loaded rule configuration")
        selected = set(active.pii.analyzer_entities or ENTITY_CATALOG)
        if not active.pii.analyzer_entities:
            selected.update(definition.entity for definition in active.pii.custom_recognizers)
        rules = [
            *self._rules,
            *(
                recognizer
                for definition, recognizer in self._custom
                if set(definition.supported_languages).intersection(active.pii.analyzer_languages)
            ),
        ]
        context = _Context(text, tuple(active.pii.analyzer_languages))
        matches: dict[tuple[str, int, int], EntityMatch] = {}
        for recognizer in rules:
            custom = id(recognizer) in self._custom_ids
            entities = [
                entity
                for entity in recognizer.supported_entities
                if (entity if custom else _normalize(entity)) in selected
            ]
            if entities:
                self._analyze_rule(
                    recognizer, entities, text, active, context, matches, custom=custom
                )
        return sorted(matches.values(), key=lambda item: (item.start, item.end, item.entity_type))

    @staticmethod
    def _analyze_rule(
        recognizer: Any,  # noqa: ANN401
        entities: list[str],
        text: str,
        policy: PolicySettings,
        context: _Context,
        matches: dict[tuple[str, int, int], EntityMatch],
        *,
        custom: bool,
    ) -> None:
        for result in recognizer.analyze(text=text, entities=entities, nlp_artifacts=None):
            score = _validated_score(result, len(text), entities)
            entity = result.entity_type if custom else _normalize(result.entity_type)
            # Presidio removes invalid checksums itself. Never revive zero-score
            # candidates, including at policy threshold zero.
            if score <= 0:
                continue
            if not custom and context.supports(result.start, recognizer.context, entity):
                score = min(1.0, max(0.4, score + 0.35))
            if score >= policy.pii.score_threshold:
                key = (entity, result.start, result.end)
                match = EntityMatch(entity, result.start, result.end, score, "deterministic")
                if key not in matches or matches[key].score < score:
                    matches[key] = match

    @staticmethod
    def _validate_policy(policy: PolicySettings) -> None:
        custom = {definition.entity for definition in policy.pii.custom_recognizers}
        if set(policy.pii.analyzer_entities) - set(ENTITY_CATALOG) - custom:
            raise ValueError("policy selects an unknown entity")
        if not set(policy.pii.supported_languages).issubset({"en", "de", "nl"}):
            raise ValueError("configured language has no rule context support")


def _validated_score(result: Any, text_length: int, entities: list[str]) -> float:  # noqa: ANN401
    """Reject malformed evidence before offset use or numeric coercion."""
    if (
        type(result.start) is not int
        or type(result.end) is not int
        or not 0 <= result.start < result.end <= text_length
        or type(result.score) not in (int, float)
        or not 0 <= result.score <= 1
        or not math.isfinite(result.score)
        or result.entity_type not in entities
    ):
        raise ValueError("rule recognizer returned an invalid result")
    return float(result.score)


def _normalize(entity: str) -> str:
    return _NORMALIZED.get(entity, entity)


def _words(text: str) -> tuple[str, ...]:
    return tuple(_INFLECTIONS.get(word.casefold(), word.casefold()) for word in _WORD.findall(text))


class _Context:
    """Index context once without retaining text after the analyze call."""

    def __init__(self, text: str, languages: tuple[str, ...]) -> None:
        self.languages = languages
        self.starts: dict[str, list[int]] = {language: [] for language in languages}
        self.words: dict[str, list[str]] = {language: [] for language in languages}
        for match in _WORD.finditer(text):
            word = match.group().casefold()
            word = _INFLECTIONS.get(word, word)
            start = match.start()
            for language in languages:
                if word not in _STOP_WORDS[language]:
                    self.starts[language].append(start)
                    self.words[language].append(word)

    def supports(self, start: int, contexts: list[str], entity: str) -> bool:
        for language in self.languages:
            index = bisect_left(self.starts[language], start)
            window = tuple(self.words[language][max(0, index - 5) : index])
            candidates = [*(contexts or []), *_CONTEXT.get(entity, {}).get(language, ())]
            if any(_contains_phrase(window, _words(candidate)) for candidate in candidates):
                return True
        return False


def _contains_phrase(window: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    return bool(phrase) and any(
        window[index : index + len(phrase)] == phrase
        for index in range(len(window) - len(phrase) + 1)
    )

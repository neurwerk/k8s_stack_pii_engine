"""Opt-in differential checks for the pinned real tokenizer/model bundle."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pii_engine.config.policy import PolicySettings
from pii_engine.config.settings import Settings
from pii_engine.services.analyzer import PresidioAnalyzer, _chunks

_CACHE = os.getenv("PII_ENGINE_REAL_MODEL_CACHE")
_VERSION = os.getenv("PII_ENGINE_REAL_MODEL_VERSION")
_DIGEST = os.getenv("PII_ENGINE_REAL_MODEL_MANIFEST_SHA256")

pytestmark = pytest.mark.skipif(
    not all((_CACHE, _VERSION, _DIGEST)),
    reason="set PII_ENGINE_REAL_MODEL_CACHE, VERSION, and MANIFEST_SHA256",
)


@pytest.fixture(scope="module")
def real_analyzer() -> PresidioAnalyzer:
    """Load every pinned model once for the opt-in differential matrix."""
    policy = PolicySettings.model_validate(
        {
            "pii": {
                "analyzerLanguages": ["en", "de", "nl"],
                "supportedLanguages": ["en", "de", "nl"],
                "scoreThreshold": 0.45,
                "defaultAction": "block",
                "analyzerEntities": ["PERSON_NAME"],
                "ner": {
                    "strategy": "perLanguage",
                    "perLanguage": {
                        "en": "english-pii",
                        "de": "german-pii",
                        "nl": "dutch-pii",
                    },
                },
            },
            "safety": {"enabled": []},
            "classifier": {"defaultClass": "general", "classes": []},
            "session": {"enabled": False},
            "notice": {
                "rerouted": "rerouted",
                "masked": "masked",
                "showWhenNoPiiDetected": False,
            },
            "routing": {"defaultTarget": "local", "targets": []},
        }
    )
    settings = Settings(
        model_cache_path=Path(_CACHE or ""),
        model_bundle_reference=Path(_CACHE or "") / "desired-bundle.json",
        model_bundle_version=_VERSION,
        model_manifest_sha256=_DIGEST,
        policy_config=Path("unused"),
        hash_key="h" * 32,
        encryption_key="e" * 32,
    )
    return PresidioAnalyzer(settings, policy)


@pytest.mark.parametrize(
    ("language", "names"),
    [
        ("en", ("Alice Johnson", "Maria Garcia", "Emma Williams")),
        ("de", ("Erika Mustermann", "Hans Schmidt", "Anna Schneider")),
        ("nl", ("Jan de Vries", "Sophie Jansen", "Daan Bakker")),
    ],
)
def test_native_stride_matches_retained_long_text_path(
    real_analyzer: PresidioAnalyzer, language: str, names: tuple[str, str, str]
) -> None:
    """Detect names before, near, and after tokenizer windows in both paths."""
    tokenizer = real_analyzer._tokenizers[language]
    max_tokens = min(int(tokenizer.model_max_length) - 32, 480)
    text = (
        f"{names[0]} lives here. "
        + "context " * max_tokens
        + f"{names[1]} works here. "
        + "context " * max_tokens
        + f"{names[2]} finishes here."
    )
    assert len(_chunks(text, tokenizer)) >= 3

    active = real_analyzer.policy.model_copy(deep=True)
    active.pii.analyzer_languages = [language]
    retained = {
        (match.start, match.end)
        for match in real_analyzer.analyze(text, active)
        if match.entity_type == "PERSON_NAME"
    }
    native = {
        (result.start, result.end)
        for result in real_analyzer._engine.analyze(
            text=text,
            language=language,
            entities=["PERSON_NAME"],
            score_threshold=active.pii.score_threshold,
        )
        if result.entity_type == "PERSON_NAME"
    }
    expected = {(text.index(name), text.index(name) + len(name)) for name in names}
    assert expected.issubset(native)
    assert expected.issubset(retained)
    assert native == retained

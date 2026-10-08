"""Protect deployment selection, independent layers and fail-closed NER."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from pii_engine.config.ner import PROFILES, ModelProfile, NerConfig, load_ner
from pii_engine.config.policy import test_policy as make_policy
from pii_engine.config.settings import Settings
from pii_engine.services.analyzer import EntityMatch, PresidioAnalyzer, _chunks
from pii_engine.services.layers import LayeredAnalyzer
from pii_engine.services.remote_http import RemoteAnalysisError


def remote_config():
    return {
        "mode": "remote",
        "languageModels": {"en": "multilingual-pii", "de": "multilingual-pii"},
        "models": {
            "multilingual-pii": {
                "profile": "gliner-multilingual-pii-v1",
                "endpoint": "https://ner.example.com/extract",
                "inferenceThreshold": 0.4,
            }
        },
    }


def test_profile_resolution_and_revision_are_not_attestation(tmp_path):
    path = tmp_path / "ner.yaml"
    path.write_text(json.dumps(remote_config()))
    config = load_ner(path)
    model = config.remote_model("multilingual-pii")
    assert model.upstream == "urchade/gliner_multi_pii-v1"
    assert model.revision is None
    assert config.models[model.name].revision_state == "unattested"
    assert model.profile is not None
    assert PROFILES[model.profile].prompt_overhead == "server-managed"
    assert config.capacity.max_calls == 2048


@pytest.mark.parametrize(
    "change",
    [
        {"languageModels": {"nl": "multilingual-pii"}},
        {"languageModels": {"en": "missing"}},
        {"unknown": True},
        {"mode": "local"},
    ],
)
def test_invalid_deployment_selection_is_rejected(change):
    with pytest.raises(ValueError):
        NerConfig.model_validate(remote_config() | change)


def test_canonical_settings_reject_explicit_legacy_defaults(tmp_path):
    with pytest.raises(ValueError, match="legacy"):
        Settings(ner_config=tmp_path / "ner.yaml", analyzer_backend="local")


def test_existing_adapter_profiles_can_be_added_as_data(monkeypatch):
    profile = ModelProfile(
        kind="kserve",
        languages=["en", "de"],
        upstream="example/model",
        revision="immutable-revision",
        tokenizer_sha256={"config.json": "0" * 64},
        tokenizer="offline-fast",
        label_mapping={"PERSON": "PERSON_NAME"},
        max_tokens=1024,
        window_tokens=900,
        overlap_tokens=100,
        prompt_overhead="tokenizer-special-tokens",
    )
    monkeypatch.setitem(PROFILES, "example-multilingual-v1", profile)
    config = remote_config()
    config["models"]["multilingual-pii"] = {
        "profile": "example-multilingual-v1",
        "modelName": "example-model",
        "endpoint": "https://ner.example.com/v1/models/example-model:predict",
        "tokenizerPath": "/remote-tokenizers/example",
    }
    model = NerConfig.model_validate(config).remote_model("multilingual-pii")
    assert model.languages == ["en", "de"]
    assert model.revision == "immutable-revision"


def test_layers_share_original_text_and_remote_failure_never_returns_rules(monkeypatch, tmp_path):
    seen = []

    class Rules:
        def __init__(self, _policy):
            pass

        def analyze(self, text, _policy):
            seen.append(("rules", text))
            return [EntityMatch("EMAIL_ADDRESS", 0, 3, 0.9, "deterministic")]

    class Ner:
        def analyze(self, text, _policy):
            seen.append(("ner", text))
            raise RemoteAnalysisError("remote inference failed")

    monkeypatch.setitem(
        sys.modules, "pii_engine.services.rules", SimpleNamespace(RulesAnalyzer=Rules)
    )
    monkeypatch.setattr("pii_engine.services.layers.create_ner_analyzer", lambda *_args: Ner())
    path = tmp_path / "ner.yaml"
    path.write_text(json.dumps(remote_config()))
    analyzer = LayeredAnalyzer(Settings(ner_config=path), make_policy(), "remote")
    with pytest.raises(RemoteAnalysisError):
        analyzer.analyze("Original 😀")
    assert seen == [("rules", "Original 😀"), ("ner", "Original 😀")]
    candidate = make_policy()
    candidate.pii.analyzer_languages = ["nl"]
    with pytest.raises(ValueError, match="not loaded"):
        analyzer.analyze("", candidate)
    assert len(seen) == 2


def test_canonical_local_loads_only_deployment_languages(monkeypatch, tmp_path):
    class Rules:
        def __init__(self, _policy):
            pass

    captured = {}

    def local(policy, **kwargs):
        captured["languages"] = policy.pii.supported_languages
        captured["profiles"] = kwargs["profiles"]
        return SimpleNamespace(analyze=lambda *_args: [])

    monkeypatch.setitem(
        sys.modules, "pii_engine.services.rules", SimpleNamespace(RulesAnalyzer=Rules)
    )
    monkeypatch.setattr("pii_engine.services.analyzer.PresidioSpacyAnalyzer", local)
    path = tmp_path / "ner.yaml"
    path.write_text(
        json.dumps(
            {
                "mode": "local",
                "languageModels": {"en": "english"},
                "models": {"english": {"profile": "spacy-en-sm-v1"}},
            }
        )
    )
    analyzer = LayeredAnalyzer(Settings(ner_config=path), make_policy(), "local")
    assert captured["languages"] == ["en"]
    assert set(captured["profiles"]) == {"en"}
    candidate = make_policy()
    candidate.pii.analyzer_languages = ["de"]
    with pytest.raises(ValueError, match="not loaded"):
        analyzer.validate_policy(candidate)


def test_legacy_multilingual_transformer_calls_once_and_retains_whitespace(monkeypatch):
    class Recognizer:
        @staticmethod
        def remove_duplicates(results):
            return results

    monkeypatch.setitem(
        sys.modules,
        "presidio_analyzer",
        SimpleNamespace(
            EntityRecognizer=Recognizer,
            RecognizerResult=object,
        ),
    )
    policy = make_policy()
    policy.pii.analyzer_languages = ["en", "de"]
    policy.pii.ner.strategy = "multilingual"
    analyzer = object.__new__(PresidioAnalyzer)
    analyzer.policy = policy
    analyzer._loaded_aliases = {"en": "multilingual-pii", "de": "multilingual-pii"}

    class Tokenizer:
        model_max_length = 512

        def __call__(self, *_args, **_kwargs):
            return {"offset_mapping": [(2, 6)]}

    tokenizer = Tokenizer()
    analyzer._tokenizers = {"en": tokenizer, "de": tokenizer}
    seen = []
    analyzer._engine = SimpleNamespace(analyze=lambda **kwargs: seen.append(kwargs["text"]) or [])
    assert analyzer.analyze("  Anna  ") == []
    assert seen == ["  Anna  "]


def test_legacy_transformer_chunks_keep_boundary_whitespace():
    class Tokenizer:
        model_max_length = 40

        def __call__(self, *_args, **_kwargs):
            return {"offset_mapping": [(index, index + 1) for index in range(2, 52, 2)]}

    text = " " * 54
    chunks = _chunks(text, Tokenizer())
    covered = set()
    for offset, chunk in chunks:
        assert chunk == text[offset : offset + len(chunk)]
        covered.update(range(offset, offset + len(chunk)))
    assert covered == set(range(len(text)))

"""Protect remote NER coverage, alignment, admission and fail-closed behavior."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import pii_engine.services.remote_analyzer as remote_module
from pii_engine.config.policy import test_policy as make_test_policy
from pii_engine.config.remote import RemoteModel
from pii_engine.config.settings import Settings
from pii_engine.runtime import EngineRuntime
from pii_engine.services.analyzer import resolve_analyzer_mode
from pii_engine.services.remote_analyzer import RemoteAnalyzer, gliner_matches, kserve_matches
from pii_engine.services.remote_http import RemoteAnalysisError, RemoteTransport, analysis_budget
from pii_engine.services.remote_tokens import (
    _verify_files,
    decode_predictions,
    load_tokenizer,
    token_chunks,
)


def gliner(**overrides):
    return RemoteModel.model_validate(
        {
            "name": "multilingual",
            "kind": "gliner",
            "model_name": "ner-multilingual",
            "url": "https://ner.example.test/extract",
            "languages": ["en", "de"],
            "label_mapping": {"person": "PERSON_NAME"},
            "inference_threshold": 0.3,
            **overrides,
        }
    )


def kserve(**overrides):
    return RemoteModel.model_validate(
        {
            "name": "english",
            "kind": "kserve",
            "model_name": "ner-english",
            "url": "https://ner.example.test/v1/models/ner-english:predict",
            "languages": ["en"],
            "label_mapping": {"PRIVATE": "PRIVATE"},
            "tokenizer_path": "/tokenizers/en",
            "tokenizer_sha256": {"config.json": "0" * 64},
            **overrides,
        }
    )


def transport(handler, **settings):
    return RemoteTransport(
        Settings(**settings),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )


class CharacterTokenizer:
    is_fast = True

    def __call__(self, text, *, add_special_tokens=True, **_kwargs):
        offsets = [(index, index + 1) for index in range(len(text))]
        if add_special_tokens:
            offsets = [(0, 0), *offsets, (0, 0)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


def test_multilingual_gliner_sends_short_mixed_text_once():
    calls = []
    text = "  Grüße 😀 Anna meets Alex  "

    def handler(request):
        calls.append(json.loads(request.content)["text"])
        start = text.index("Anna")
        return httpx.Response(
            200,
            json={
                "model": "ner-multilingual",
                "entities": [
                    {"start": start, "end": start + 4, "label": "person", "score": 0.9},
                ],
            },
        )

    remote = transport(handler)
    with analysis_budget(5, 2):
        matches = gliner_matches(text, gliner(), remote)
    assert calls == [text]
    assert text[matches[0].start : matches[0].end] == "Anna"
    remote.close()


def test_gliner_size_rejections_split_with_complete_character_coverage():
    text = "  \u03b1😀 " * 280
    accepted = []

    def handler(request):
        chunk = json.loads(request.content)["text"]
        if len(chunk) > 160:
            return httpx.Response(413)
        accepted.append(chunk)
        return httpx.Response(200, json={"model": "ner-multilingual", "entities": []})

    remote = transport(handler)
    with analysis_budget(5, 100):
        assert gliner_matches(text, gliner(), remote) == []
    # Distinct characters also verify coverage independently of repeated words.
    unique = "".join(chr(0x1000 + index) for index in range(1500))
    accepted.clear()
    with analysis_budget(5, 100):
        gliner_matches(unique, gliner(), remote)
    assert set(unique) == set("".join(accepted))
    remote.close()


@pytest.mark.parametrize(
    "reply",
    [
        {"model": "wrong", "entities": []},
        {"model": "ner-multilingual"},
        {
            "model": "ner-multilingual",
            "entities": [
                {"start": True, "end": 3, "label": "person", "score": 0.8},
            ],
        },
        {
            "model": "ner-multilingual",
            "entities": [
                {"start": 0, "end": 50, "label": "person", "score": 0.8},
            ],
        },
        {
            "model": "ner-multilingual",
            "entities": [
                {"start": 0, "end": 3, "label": "unknown", "score": 0.8},
            ],
        },
        {
            "model": "ner-multilingual",
            "entities": [
                {"start": 0, "end": 3, "label": "person", "score": True},
            ],
        },
    ],
)
def test_invalid_gliner_reply_is_not_a_clean_scan(reply):
    remote = transport(lambda _request: httpx.Response(200, json=reply))
    with analysis_budget(5, 1), pytest.raises(RemoteAnalysisError):
        gliner_matches("Anna", gliner(), remote)
    remote.close()


@pytest.mark.parametrize("status", [401, 429, 500, 302])
def test_http_errors_do_not_return_partial_results(status):
    remote = transport(lambda _request: httpx.Response(status, text="private response"))
    with analysis_budget(5, 1), pytest.raises(RemoteAnalysisError) as error:
        gliner_matches("private request", gliner(), remote)
    assert "private" not in str(error.value)
    remote.close()


def test_response_and_request_budgets_are_enforced():
    remote = transport(
        lambda _request: httpx.Response(200, content=b"x" * 1025), remote_max_response_bytes=1024
    )
    with analysis_budget(5, 1), pytest.raises(RemoteAnalysisError, match="size"):
        gliner_matches("Anna", gliner(), remote)
    remote.close()
    remote = transport(lambda _request: httpx.Response(413))
    with analysis_budget(5, 1), pytest.raises(RemoteAnalysisError, match="budget"):
        gliner_matches("x" * 200, gliner(), remote)
    remote.close()


def test_timeout_failure_is_content_free():
    def handler(request):
        raise httpx.ReadTimeout("private response", request=request)

    remote = transport(handler)
    with analysis_budget(5, 1), pytest.raises(RemoteAnalysisError) as error:
        gliner_matches("Anna", gliner(), remote)
    assert "private" not in str(error.value)
    remote.close()


def test_token_chunks_keep_unicode_whitespace_and_special_tokens():
    text = "  " + "".join(chr(0x1000 + index) for index in range(1600)) + "  "
    chunks = list(token_chunks(text, CharacterTokenizer()))
    covered = set()
    for offset, chunk in chunks:
        assert len(chunk) + 2 <= 512
        assert text[offset : offset + len(chunk)] == chunk
        covered.update(range(offset, offset + len(chunk)))
    assert covered == set(range(len(text)))


@pytest.mark.parametrize(
    "label,entity,language",
    [
        ("PRIVATE", "PRIVATE", "en"),
        ("FIRSTNAME", "PERSON_NAME", "de"),
    ],
)
def test_kserve_alignment_preserves_private_and_specific_german_entities(label, entity, language):
    model = kserve(languages=[language], label_mapping={label: entity})
    labels = {"0": "O", "1": "B-" + label, "2": "I-" + label}
    text = "😀 Anna"
    probabilities = [
        {"0": 1.0, "1": 0.0, "2": 0.0},
        {"0": 1.0, "1": 0.0, "2": 0.0},
        {"0": 0.0, "1": 0.9, "2": 0.1},
        {"0": 0.0, "1": 0.1, "2": 0.9},
        {"0": 1.0, "1": 0.0, "2": 0.0},
    ]
    matches = decode_predictions(
        {"predictions": [probabilities]},
        text,
        [(0, 0), (0, 1), (2, 4), (4, 6), (0, 0)],
        labels,
        model,
    )
    assert len(matches) == 1
    assert matches[0].entity_type == entity
    assert text[matches[0].start : matches[0].end] == "Anna"


@pytest.mark.parametrize(
    "prediction",
    [
        [],
        [{"0": 1}],
        [{"0": 0.5, "1": 0.6, "2": 0.0}],
        [{"0": True, "1": 0, "2": 0}],
        [{"0": float("nan"), "1": 0, "2": 0}],
    ],
)
def test_kserve_rejects_incomplete_or_invalid_probabilities(prediction):
    with pytest.raises(RemoteAnalysisError):
        decode_predictions(
            {"predictions": [prediction]},
            "A",
            [(0, 1)],
            {"0": "O", "1": "B-PRIVATE", "2": "I-PRIVATE"},
            kserve(),
        )


def test_kserve_requests_never_rely_on_remote_truncation():
    tokenizer = CharacterTokenizer()
    seen = []
    text = "\u03b1" * 1200

    def handler(request):
        instances = json.loads(request.content)["instances"]
        assert len(instances) == 1 and len(instances[0]) + 2 <= 512
        seen.append(instances[0])
        return httpx.Response(
            200,
            json={
                "predictions": [
                    [{"0": 1, "1": 0, "2": 0} for _ in tokenizer(instances[0])["input_ids"]]
                ]
            },
        )

    remote = transport(handler)
    with analysis_budget(5, 10):
        assert (
            kserve_matches(
                text, kserve(), remote, tokenizer, {"0": "O", "1": "B-PRIVATE", "2": "I-PRIVATE"}
            )
            == []
        )
    assert len(seen) > 1
    remote.close()


def test_remote_mode_does_not_select_a_local_transformer_bundle(tmp_path):
    settings = Settings(
        allow_test_analyzer=False,
        analyzer_backend="remote-gliner",
        remote_config=tmp_path / "r",
        policy_config=tmp_path / "policy",
        hash_key="h" * 32,
        encryption_key="e" * 32,
    )
    assert resolve_analyzer_mode(settings) == "remote-gliner"


def test_policy_cannot_request_below_gliner_detection_floor():
    analyzer = object.__new__(RemoteAnalyzer)
    analyzer.language_models = None
    analyzer.models = [gliner(inference_threshold=0.5)]
    with pytest.raises(ValueError, match="floor"):
        analyzer._validate_languages(make_test_policy())


@pytest.mark.parametrize(
    "url",
    [
        "http://ner.example.test/extract",
        "https://user:key@ner.example.test/extract",
        "https://ner.example.test/extract?key=value",
        "https://ner.example.test/other",
    ],
)
def test_remote_configuration_rejects_unsafe_endpoints(url):
    with pytest.raises(ValueError):
        gliner(url=url)


def test_tokenizer_files_cannot_be_unverified_or_symlinked(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="digest"):
        _verify_files(tmp_path, {"config.json": "0" * 64})
    path.unlink()
    path.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="symlink"):
        _verify_files(tmp_path, {"config.json": "0" * 64})


def test_kserve_rejects_a_locally_pinned_but_wrong_upstream_tokenizer():
    with pytest.raises(ValueError, match=r"configuration|immutable revision"):
        load_tokenizer(
            kserve(
                tokenizer_sha256={
                    "config.json": "0" * 64,
                    "tokenizer_config.json": "0" * 64,
                }
            )
        )


async def test_remote_failure_propagates_through_policy_without_a_pass():
    class FailingAnalyzer:
        def analyze(self, text, policy=None):
            raise RemoteAnalysisError("remote inference failed")

    from pii_engine.models.contracts import SegmentRequest

    runtime = EngineRuntime(Settings())
    runtime._analyzer = FailingAnalyzer()
    runtime.policy = runtime._policy_service(runtime.policy_settings)
    request = SegmentRequest.model_validate(
        {
            "api_version": "v2",
            "request_kind": "chat",
            "scope": "request",
            "segments": [{"id": "s0", "text": "Anna"}],
            "text_pii_enabled": True,
            "attachments_present": False,
        }
    )
    with pytest.raises(RemoteAnalysisError):
        await runtime.analyze_segments("adapter", request)
    await runtime.close()


@pytest.mark.parametrize("canonical", [False, True])
def test_presidio_coordinator_calls_multilingual_service_once(monkeypatch, tmp_path, canonical):
    class FakeRecognizer:
        def __init__(self, supported_entities, **_kwargs):
            self.supported_entities = supported_entities

    class FakeResult:
        def __init__(self, entity_type, start, end, score):
            self.entity_type, self.start, self.end, self.score = entity_type, start, end, score

    monkeypatch.setitem(
        sys.modules,
        "presidio_analyzer",
        SimpleNamespace(
            RemoteRecognizer=FakeRecognizer,
            RecognizerResult=FakeResult,
        ),
    )
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"model": "ner-multilingual", "entities": []})

    remote = transport(handler)
    monkeypatch.setattr(remote_module, "RemoteTransport", lambda _settings: remote)
    path = tmp_path / "remote.json"
    if canonical:
        path.write_text(
            json.dumps(
                {
                    "mode": "remote",
                    "languageModels": {"en": "multilingual", "de": "multilingual"},
                    "models": {
                        "multilingual": {
                            "profile": "gliner-multilingual-pii-v1",
                            "endpoint": "https://ner.example.com/extract",
                            "inferenceThreshold": 0.4,
                        }
                    },
                }
            )
        )
        settings = Settings(ner_config=path)
    else:
        path.write_text(json.dumps({"models": [gliner().model_dump(mode="json")]}))
        settings = Settings(remote_config=path, analyzer_backend="remote-gliner")
    policy = make_test_policy()
    policy.pii.analyzer_languages = ["en", "de"]
    analyzer = RemoteAnalyzer(settings, policy)
    with analysis_budget(5, 1):
        matches = analyzer.analyze("Mixed text 😀")
    assert len(calls) == 1
    assert matches == []  # Rules are composed separately, not repeated inside NER.
    analyzer.close()


def test_gpu_admission_and_nested_call_budget_are_bounded():
    remote = transport(
        lambda _request: httpx.Response(
            200,
            json={"model": "ner-multilingual", "entities": []},
        )
    )
    assert remote._capacity.acquire(blocking=False)
    try:
        with analysis_budget(0.001, 1), pytest.raises(RemoteAnalysisError, match="deadline"):
            gliner_matches("Anna", gliner(), remote)
    finally:
        remote._capacity.release()
    with analysis_budget(5, 1):
        with analysis_budget(30, 100):
            gliner_matches("Anna", gliner(), remote)
        with pytest.raises(RemoteAnalysisError, match="budget"):
            gliner_matches("Alex", gliner(), remote)
    remote.close()

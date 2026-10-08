"""Protect exact window reuse and bounded, non-text retention."""

from types import SimpleNamespace
from typing import cast

import pytest

from pii_engine.config.remote import RemoteModel
from pii_engine.services.analyzer import EntityMatch
from pii_engine.services.ner_cache import NerWindowCache, accounted_size
from pii_engine.services.remote_analyzer import gliner_matches
from pii_engine.services.remote_http import RemoteAnalysisError, RemoteTransport


def test_cache_bounds_fixed_expiry_config_separation_and_success_only():
    now = [0.0]
    findings = (EntityMatch("PERSON_NAME", 0, 4, 0.9, "spacy"),)
    cache = NerWindowCache(2 * accounted_size(findings), 86400, timer=lambda: now[0])
    calls = []

    def detect():
        calls.append(1)
        return findings

    cache.run("Anna", b"en:0.5:model-a", detect)
    now[0] = 86399
    assert cache.run("Anna", b"en:0.5:model-a", detect) == findings
    assert len(calls) == 1
    cache.run("Anna", b"de:0.7:model-b", detect)
    cache.run("Alex", b"en:0.5:model-a", detect)
    assert cache._cache.currsize <= cache._cache.maxsize
    cache.run("Anna", b"en:0.5:model-a", detect)  # Oldest key was evicted.
    assert len(calls) == 4
    now[0] += 86400
    cache.expire()
    assert cache._cache.currsize == 0
    cache.run("Anna", b"en:0.5:model-a", detect)
    now[0] += 86399
    cache.run("Anna", b"en:0.5:model-a", detect)
    now[0] += 1
    cache.expire()  # A hit did not extend lifetime.
    assert cache._cache.currsize == 0
    assert cache.run("nothing", b"config", lambda: ()) == ()
    assert cache.run("nothing", b"config", detect) == ()

    def fail():
        raise RemoteAnalysisError("failed")

    with pytest.raises(RemoteAnalysisError):
        cache.run("failure", b"config", fail)
    assert cache.run("failure", b"config", detect) == findings
    small = NerWindowCache(1, 86400)
    assert small.run("Anna", b"config", detect) == findings
    assert small._cache.currsize == 0
    assert all(isinstance(key, bytes) and len(key) == 32 for key in cache._cache)
    cache.close()


def test_gliner_history_reuses_exact_windows_and_edits_miss_without_offset_changes():
    model = RemoteModel(
        name="multilingual",
        kind="gliner",
        model_name="ner-multilingual",
        url="https://ner.example.test/extract",
        languages=["en", "de"],
        label_mapping={"person": "PERSON_NAME"},
        inference_threshold=0.3,
    )
    calls = []

    def request(_model, body):
        text = body["text"]
        calls.append(text)
        start = text.find("Anna")
        return {
            "model": model.model_name,
            "entities": []
            if start < 0
            else [{"start": start, "end": start + 4, "label": "person", "score": 0.9}],
        }

    transport = SimpleNamespace(
        request=request, window_cache=NerWindowCache(100000, 86400), cache_namespace=b"loaded-model"
    )
    history = "x" * 950 + "Anna" + "y" * 1100
    remote = cast("RemoteTransport", transport)
    first = gliner_matches(history, model, remote)
    count = len(calls)
    assert gliner_matches(history, model, remote) == first
    assert len(calls) == count
    edited = history[:1500] + "z" + history[1501:]
    cached = gliner_matches(edited, model, remote)
    assert count < len(calls) < 2 * count
    uncached = cast("RemoteTransport", SimpleNamespace(request=request))
    assert cached == gliner_matches(edited, model, uncached)
    assert all(edited[item.start : item.end] == "Anna" for item in cached)
    transport.window_cache.close()

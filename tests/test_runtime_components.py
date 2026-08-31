"""Exercise runtime dependencies without loading the multi-gigabyte checkpoints."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

import pii_engine.runtime as runtime_module
from pii_engine.config.policy import CustomRecognizer, load_policy
from pii_engine.config.policy import test_policy as make_test_policy
from pii_engine.config.settings import Settings
from pii_engine.lib.catalog import ENTITY_CATALOG
from pii_engine.main import _tls_kwargs
from pii_engine.models.contracts import OpenAIChatRequest
from pii_engine.runtime import EngineRuntime
from pii_engine.services.analyzer import (
    DeterministicAnalyzer,
    EntityMatch,
    PresidioAnalyzer,
    PresidioSpacyAnalyzer,
    _baseline_chunks,
    _chunks,
    configure_inference_device,
)
from pii_engine.services.anonymizer import PresidioAnonymizer
from pii_engine.services.limiter import AnalysisCapacityError
from pii_engine.services.policy import PolicyResult
from pii_engine.services.recognizers import (
    custom_recognizers,
    normalized_recognizers,
    normalized_transformers_recognizer,
)
from pii_engine.services.session import SessionDecision, SessionStore


def _bundle(tmp_path: Path) -> tuple[Path, str]:
    checksum_data = f"{hashlib.sha256(b'model').hexdigest()}  english-pii/model.bin\n".encode()
    model = tmp_path / "digest" / "english-pii"
    model.mkdir(parents=True)
    manifest = {
        "schemaVersion": 2,
        "bundleVersion": "1",
        "models": {
            "english-pii": {
                "catalogId": "english",
                "variantId": "transformers",
                "upstream": "owner/model",
                "revision": "revision",
                "path": "english-pii",
                "license": "MIT",
                "licenseUrl": "https://example.test",
                "supportedLanguages": ["en"],
            }
        },
        "runtime": {
            "labelsToIgnore": ["O"],
            "aggregationStrategy": "simple",
            "stride": 64,
            "modelToPresidioEntityMapping": {"B-EMAIL": "EMAIL_ADDRESS"},
        },
        "checksumFile": "checksums.sha256",
        "checksumSha256": hashlib.sha256(checksum_data).hexdigest(),
        "checksumSize": len(checksum_data),
        "fileCount": 1,
        "totalModelBytes": 5,
    }
    data = yaml.safe_dump(manifest, sort_keys=False).encode()
    digest = hashlib.sha256(data).hexdigest()
    root = tmp_path / digest
    root.mkdir()
    (root / "english-pii").mkdir()
    (root / "manifest.yaml").write_bytes(data)
    (root / "checksums.sha256").write_bytes(checksum_data)
    (root / "english-pii/model.bin").write_bytes(b"model")
    return root, digest


class _FakeEngine:
    def analyze(self, **_kwargs: object):
        from presidio_analyzer import RecognizerResult

        result = RecognizerResult("IBAN_CODE", 0, 4, 0.9)
        result.recognition_metadata = {"recognizer_name": "PatternRecognizer"}
        return [result]


class _FakeTokenizer:
    model_max_length = 512

    def __call__(self, text: str, **_kwargs: object) -> dict[str, list[tuple[int, int]]]:
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}


def test_presidio_analyzer_maps_bundle_aliases_without_model_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manifest paths become local transformer paths and results are normalized."""
    root, digest = _bundle(tmp_path)
    policy = make_test_policy()
    policy.pii.ner.per_language = {"en": "english-pii"}
    policy.pii.analyzer_entities = ["IBAN"]
    settings = Settings(
        allow_test_analyzer=True,
        model_cache_path=tmp_path,
        model_bundle_version="1",
        model_manifest_sha256=digest,
    )
    monkeypatch.setattr(PresidioAnalyzer, "_create_engine", lambda _self: _FakeEngine())
    monkeypatch.setattr(
        PresidioAnalyzer, "_load_tokenizers", lambda _self: {"en": _FakeTokenizer()}
    )
    analyzer = PresidioAnalyzer(settings, policy)
    assert analyzer.bundle == root
    assert analyzer.analyze("test") == [EntityMatch("IBAN", 0, 4, 0.9, "deterministic")]


def test_custom_recognizer_entities_survive_default_catalog_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty analyzerEntities list includes validated customer-defined entities."""

    class FakeCustomEngine:
        def analyze(self, **_kwargs: object):
            from presidio_analyzer import RecognizerResult

            result = RecognizerResult("CUSTOMER_ID", 0, 4, 0.9)
            result.recognition_metadata = {"recognizer_name": "Customer ID"}
            return [result]

    root, digest = _bundle(tmp_path)
    policy = make_test_policy()
    policy.pii.ner.per_language = {"en": "english-pii"}
    policy.pii.analyzer_entities = []
    policy.pii.custom_recognizers = [
        CustomRecognizer.model_validate(
            {
                "name": "Customer ID",
                "entity": "CUSTOMER_ID",
                "regex": r"CUST-\d+",
                "score": 0.9,
                "supportedLanguages": ["en"],
            }
        )
    ]
    settings = Settings(
        allow_test_analyzer=True,
        model_cache_path=tmp_path,
        model_bundle_version="1",
        model_manifest_sha256=digest,
    )
    monkeypatch.setattr(PresidioAnalyzer, "_create_engine", lambda _self: FakeCustomEngine())
    monkeypatch.setattr(
        PresidioAnalyzer, "_load_tokenizers", lambda _self: {"en": _FakeTokenizer()}
    )
    analyzer = PresidioAnalyzer(settings, policy)
    assert analyzer.bundle == root
    assert analyzer.analyze("test") == [EntityMatch("CUSTOMER_ID", 0, 4, 0.9, "deterministic")]


def test_retained_tokenizer_chunking_overlaps_long_documents() -> None:
    """Long input is fully covered until native stride differential tests pass."""
    chunks = _chunks("x" * 700, _FakeTokenizer())
    assert chunks[0][0] == 0
    assert chunks[-1][0] + len(chunks[-1][1]) == 700
    assert chunks[1][0] < chunks[0][0] + len(chunks[0][1])


def test_baseline_chunking_preserves_boundary_match_offsets_and_deduplicates() -> None:
    """Overlapping sub-1M spaCy chunks return one correctly offset boundary match."""
    from presidio_analyzer import RecognizerResult

    marker = "boundary@example.com"
    marker_start = 899_995
    text = "x" * marker_start + marker + "x" * 200_000
    calls: list[int] = []

    class BoundaryEngine:
        def analyze(self, *, text: str, **_kwargs: object) -> list[RecognizerResult]:
            calls.append(len(text))
            start = text.find(marker)
            if start < 0:
                return []
            result = RecognizerResult("EMAIL_ADDRESS", start, start + len(marker), 0.9)
            result.recognition_metadata = {"recognizer_name": "SpacyRecognizer"}
            return [result]

    analyzer = PresidioSpacyAnalyzer.__new__(PresidioSpacyAnalyzer)
    analyzer.policy = make_test_policy()
    analyzer._engine = BoundaryEngine()

    assert analyzer.analyze(text) == [
        EntityMatch(
            "EMAIL_ADDRESS",
            marker_start,
            marker_start + len(marker),
            0.9,
            "spacy",
        )
    ]
    assert len(calls) == 2
    assert max(calls) < 1_000_000
    chunks = _baseline_chunks("x" * 4_000_000)
    assert chunks[0][0] == 0
    assert chunks[-1][0] + len(chunks[-1][1]) == 4_000_000
    assert max(len(chunk) for _offset, chunk, _start, _end in chunks) < 1_000_000


def test_presidio_anonymizer_executes_upstream_mask_and_encrypt() -> None:
    """Ordinary and encrypt actions use the in-process upstream engine."""
    anonymizer = PresidioAnonymizer("k" * 32)
    match = EntityMatch("EMAIL_ADDRESS", 0, 4, 0.9, "test")
    assert (
        anonymizer.apply(
            "test", match, "mask", {"masking_char": "*", "chars_to_mask": 4, "from_end": True}
        )
        == "****"
    )
    encrypted = anonymizer.apply("test", match, "encrypt", {})
    assert encrypted != "test"


def test_recognizer_factories_validate_and_build() -> None:
    """Normalized and customer recognizers are real Presidio registry entries."""
    policy = make_test_policy()
    definition = {
        "name": "Customer ID",
        "entity": "CUSTOMER_ID",
        "regex": r"CUST-\d+",
        "score": 0.9,
        "supportedLanguages": ["en"],
    }
    policy.pii.custom_recognizers = [CustomRecognizer.model_validate(definition)]
    transformer = normalized_transformers_recognizer(["PERSON_NAME", "IBAN"], "de")
    assert transformer.supported_language == "de"
    assert transformer.supported_entities == ["PERSON_NAME", "IBAN"]
    normalized = normalized_recognizers(("en",))
    assert "STEUERNUMMER" in {
        entity for recognizer in normalized for entity in recognizer.supported_entities
    }
    assert custom_recognizers(policy.pii.custom_recognizers)[0].supported_entities == [
        "CUSTOMER_ID"
    ]


@pytest.mark.parametrize(
    "value",
    [
        "12345/67890",
        "123/456/78901",
        "12/345/67890",
        "012/345/67890",
        "12 345 67890",
        "012 345 67890",
        "123/4567/8901",
        "2812034567890",
        "9123045678901",
        "1112034567890",
        "3012034567890",
        "2412034567890",
        "2212034567890",
        "2612034567890",
        "4012034567890",
        "2312034567890",
        "5123045678901",
        "2712034567890",
        "1012034567890",
        "3212034567890",
        "3112034567890",
        "2112034567890",
        "4112034567890",
    ],
)
def test_steuernummer_recognizer_covers_regional_and_country_formats(value: str) -> None:
    """The engine-owned entity recognizes every format family from the German table."""
    text = f"Meine Steuernummer ist {value}."
    matches = DeterministicAnalyzer().analyze(text)
    assert [match for match in matches if match.entity_type == "STEUERNUMMER"] == [
        EntityMatch("STEUERNUMMER", 23, 23 + len(value), 0.95, "deterministic")
    ]


def test_steuernummer_preserves_cross_entity_overlap_evidence() -> None:
    matches = DeterministicAnalyzer().analyze("2812034567890")
    assert {match.entity_type for match in matches} >= {
        "STEUERNUMMER",
        "PHONE_NUMBER",
        "CREDIT_CARD_NUMBER",
    }


@pytest.mark.parametrize(
    "value",
    [
        "12345678901",
        "7123045678901",
        "x12345/67890",
        "12345/67890x",
        "12/3456/7890",
    ],
)
def test_steuernummer_recognizer_rejects_other_numeric_identifiers(value: str) -> None:
    """A Steuer-ID, unsupported country prefix, and embedded candidates do not match."""
    assert not any(
        match.entity_type == "STEUERNUMMER" for match in DeterministicAnalyzer().analyze(value)
    )


def test_steuernummer_is_an_engine_owned_entity() -> None:
    assert "STEUERNUMMER" in ENTITY_CATALOG


def test_inference_device_is_explicit_and_cuda_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU is forced and a requested unavailable GPU never silently falls back."""
    import torch

    assert configure_inference_device("cpu") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA device is unavailable"):
        configure_inference_device("cuda:0")


class _FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def ping(self) -> bool:
        return True

    async def getex(self, key: str, **_kwargs: object) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str, **_kwargs: object) -> None:
        self.values[key] = value

    async def aclose(self) -> None:
        return None


async def test_session_store_persists_only_safe_taint_metadata() -> None:
    """Valkey values contain decisions and counts but no request or reversal data."""
    store = SessionStore("redis://unused", 60, "test")
    fake = _FakeRedis()
    store._client = fake
    decision = SessionDecision(
        api_version="v1",
        request_kind="model",
        decision="block",
        entities=["EMAIL_ADDRESS"],
        entity_counts={"EMAIL_ADDRESS": 1},
        report_rows=[],
        overlap_count=0,
        route_class=None,
        remote_allowed=False,
        request_notices=[],
        response_notices=[],
    )
    await store.set("a" * 64, decision)
    assert await store.get("a" * 64) == decision
    serialized = next(iter(fake.values.values()))
    stored = json.loads(serialized)
    assert "request" not in stored
    assert "reversal" not in stored
    assert stored["decision"] == "block"


async def test_session_store_rejects_non_tainted_injected_state() -> None:
    """A modified Valkey record can never bypass policy analysis."""
    store = SessionStore("redis://unused", 60, "test")
    fake = _FakeRedis()
    fake.values["pii-engine:v2:test:session:" + "a" * 64] = json.dumps(
        {
            "api_version": "v1",
            "request_kind": "model",
            "decision": "pass",
            "entities": [],
            "entity_counts": {},
            "report_rows": [],
            "overlap_count": 0,
            "route_class": None,
            "remote_allowed": True,
            "request_notices": [],
            "response_notices": [],
        }
    )
    store._client = fake
    with pytest.raises(RuntimeError, match="invalid policy state"):
        await store.get("a" * 64)


@pytest.mark.parametrize(
    ("timestamps", "expected_ms"),
    [([10.0, 13.2], 3_200), ([10.0, 700.0], 600_000)],
)
async def test_runtime_attaches_bounded_monotonic_scan_duration(
    monkeypatch: pytest.MonkeyPatch,
    timestamps: list[float],
    expected_ms: int,
) -> None:
    runtime = EngineRuntime(Settings(allow_test_analyzer=True))
    values = iter(timestamps)

    class FakeTime:
        @staticmethod
        def monotonic() -> float:
            return next(values)

    monkeypatch.setattr(runtime_module, "time", FakeTime)
    request = OpenAIChatRequest(model="test", messages=[{"role": "user", "content": "hello"}])

    result = await runtime._run_analysis("adapter", runtime.policy, request, asyncio.Event())

    assert result.scan_performed is True
    assert result.duration_ms == expected_ms


async def test_masked_reroute_remains_sticky_after_current_request_reprocessing() -> None:
    """Cached reroutes retain safe routing while each new request is transformed."""
    runtime = EngineRuntime(Settings(allow_test_analyzer=True))
    store = SessionStore("redis://unused", 60, "test")
    store._client = _FakeRedis()
    runtime.session = store
    key = "b" * 64
    tainted = OpenAIChatRequest(
        model="test",
        messages=[{"role": "user", "content": "IBAN DE89370400440532013000"}],
    )
    first = await runtime.analyze("adapter", tainted, key)
    assert first.decision == "reroute"
    assert isinstance(first.request, OpenAIChatRequest)
    first_content = first.request.messages[0].content
    assert isinstance(first_content, str)
    assert "DE89370400440532013000" not in first_content

    clean = OpenAIChatRequest(
        model="test", messages=[{"role": "user", "content": "No identifiers here"}]
    )
    second = await runtime.analyze("adapter", clean, key)
    assert second.decision == "reroute"
    assert second.remote_allowed is False
    assert second.route_class == "local"
    assert second.request == clean


def test_policy_and_process_settings_reject_incomplete_runtime(tmp_path: Path) -> None:
    """Invalid policy and partial production settings fail before serving traffic."""
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text("pii: {}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_policy(policy_file)
    with pytest.raises(ValueError, match="model cache"):
        Settings(model_cache_path=tmp_path)
    with pytest.raises(ValueError, match="rolling hash key"):
        Settings(
            allow_test_analyzer=False,
            policy_config=policy_file,
            hash_key="short",
            encryption_key="e" * 32,
            model_cache_path=tmp_path,
            model_bundle_reference=tmp_path / "desired-bundle.json",
            model_bundle_version="1",
            model_manifest_sha256="0" * 64,
        )


def test_tls_kwargs_require_complete_server_identity(tmp_path: Path) -> None:
    """The application listener cannot start without all mTLS files configured."""
    with pytest.raises(RuntimeError, match="application TLS"):
        _tls_kwargs(Settings(allow_test_analyzer=True))
    settings = Settings(
        allow_test_analyzer=True,
        tls_cert=tmp_path / "tls.crt",
        tls_key=tmp_path / "tls.key",
        tls_ca=tmp_path / "ca.crt",
    )
    assert _tls_kwargs(settings)["ssl_cert_reqs"] != 0


async def test_timed_out_worker_retains_shared_capacity() -> None:
    """A timed-out thread cannot release its limiter slot while still running."""
    started = threading.Event()
    release = threading.Event()

    class BlockingPolicy:
        def analyze(self, request: object) -> PolicyResult:
            started.set()
            release.wait(timeout=1)
            return PolicyResult(request=cast(Any, request), decision="pass", remote_allowed=True)

    runtime = EngineRuntime(
        Settings(
            allow_test_analyzer=True,
            analysis_timeout=1,
            max_concurrent_analyses=1,
            max_queued_analyses=0,
        )
    )
    runtime.policy_settings.pii.timeout = 0.01
    runtime.policy = cast(Any, BlockingPolicy())
    request = cast(Any, {"model": "test", "messages": [{"role": "user", "content": "x"}]})
    try:
        with pytest.raises(TimeoutError):
            await runtime.analyze("adapter", request)
        assert started.is_set()
        with pytest.raises(AnalysisCapacityError, match="full"):
            await runtime.analyze("adapter", request)
    finally:
        release.set()
    for _attempt in range(20):
        if runtime.limiter.stats.in_flight == 0:
            break
        await asyncio.sleep(0.01)
    assert runtime.limiter.stats.in_flight == 0

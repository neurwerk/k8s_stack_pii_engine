"""Test fail-closed production analyzer selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from pii_engine.config.policy import test_policy as make_test_policy
from pii_engine.config.settings import Settings
from pii_engine.runtime import EngineRuntime
from pii_engine.services.analyzer import (
    SPACY_ENTITY_MAPPING,
    SPACY_IGNORED_ENTITY_LABELS,
    DeterministicAnalyzer,
    PresidioSpacyAnalyzer,
    resolve_analyzer_mode,
)


def _verified_bundle(tmp_path: Path) -> tuple[Path, str]:
    checksum_data = f"{hashlib.sha256(b'model').hexdigest()}  english-pii/model.bin\n".encode()
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
    model = root / "english-pii/model.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    (root / "manifest.yaml").write_bytes(data)
    (root / "checksums.sha256").write_bytes(checksum_data)
    (root / ".complete").write_text(digest, encoding="ascii")
    return root, digest


def _policy_file(tmp_path: Path) -> Path:
    path = tmp_path / "policy.yaml"
    path.write_text(
        yaml.safe_dump(make_test_policy().model_dump(mode="json", by_alias=True)),
        encoding="utf-8",
    )
    return path


def _production_settings(policy_file: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "allow_test_analyzer": False,
        "policy_config": policy_file,
        "hash_key": "h" * 32,
        "encryption_key": "e" * 32,
    }
    values.update(overrides)
    return Settings.model_validate(values)


def _write_reference(path: Path, digest: str, version: str = "1") -> None:
    path.write_text(
        json.dumps({"schemaVersion": 1, "bundleVersion": version, "manifestSha256": digest}),
        encoding="ascii",
    )


def test_auto_mode_uses_baseline_when_desired_reference_is_absent(tmp_path: Path) -> None:
    policy = _policy_file(tmp_path)
    assert resolve_analyzer_mode(_production_settings(policy)) == "baseline"
    cache = tmp_path / "cache"
    settings = _production_settings(
        policy,
        model_cache_path=cache,
        model_bundle_reference=cache / "desired-bundle.json",
        model_bundle_version="1",
        model_manifest_sha256="0" * 64,
    )
    assert resolve_analyzer_mode(settings) == "baseline"


def test_production_runtime_uses_spacy_baseline_without_test_analyzer(tmp_path: Path) -> None:
    runtime = EngineRuntime(_production_settings(_policy_file(tmp_path)))
    assert runtime.settings.allow_test_analyzer is False
    assert runtime.analyzer_mode == "baseline"
    assert isinstance(runtime._analyzer, PresidioSpacyAnalyzer)


def test_spacy_baseline_normalizes_financial_entities() -> None:
    policy = make_test_policy()
    policy.pii.supported_languages = ["en"]
    policy.pii.analyzer_languages = ["en"]
    policy.pii.analyzer_entities = ["IBAN", "CREDIT_CARD_NUMBER"]
    text = "IBAN GB82WEST12345698765432 and card 4111 1111 1111 1111"
    entities = {match.entity_type for match in PresidioSpacyAnalyzer(policy).analyze(text)}
    assert entities == {"IBAN", "CREDIT_CARD_NUMBER"}


def test_spacy_mapped_and_ignored_labels_cover_bundled_models() -> None:
    """Every EN, DE, and NL NER label has an explicit non-overlapping disposition."""
    expected = {
        "CARDINAL",
        "DATE",
        "EVENT",
        "FAC",
        "GPE",
        "LANGUAGE",
        "LAW",
        "LOC",
        "MISC",
        "MONEY",
        "NORP",
        "ORDINAL",
        "ORG",
        "PERCENT",
        "PER",
        "PERSON",
        "PRODUCT",
        "QUANTITY",
        "TIME",
        "WORK_OF_ART",
    }
    mapped = set(SPACY_ENTITY_MAPPING)
    ignored = set(SPACY_IGNORED_ENTITY_LABELS)
    assert mapped.isdisjoint(ignored)
    assert mapped | ignored == expected


async def test_baseline_runtime_requests_restart_when_reference_appears(tmp_path: Path) -> None:
    policy = _policy_file(tmp_path)
    cache = tmp_path / "cache"
    reference = cache / "desired-bundle.json"
    settings = _production_settings(
        policy,
        model_cache_path=cache,
        model_bundle_reference=reference,
        model_bundle_version="1",
        model_manifest_sha256="0" * 64,
    )
    runtime = EngineRuntime(settings)
    assert runtime.restart_required() is False
    assert await runtime.ready() is True
    cache.mkdir()
    _root, digest = _verified_bundle(cache)
    _write_reference(reference, digest)
    settings.model_manifest_sha256 = digest
    assert runtime.restart_required() is True
    assert await runtime.ready() is False


def test_changed_pin_uses_baseline_until_new_bundle_is_selected(tmp_path: Path) -> None:
    old_root, old_digest = _verified_bundle(tmp_path)
    reference = tmp_path / "desired-bundle.json"
    _write_reference(reference, old_digest)
    settings = _production_settings(
        _policy_file(tmp_path),
        model_cache_path=tmp_path,
        model_bundle_reference=reference,
        model_bundle_version="2",
        model_manifest_sha256="1" * 64,
    )
    assert old_root.is_dir()
    assert resolve_analyzer_mode(settings) == "baseline"


def test_auto_mode_uses_fully_verified_desired_transformer_bundle(tmp_path: Path) -> None:
    _root, digest = _verified_bundle(tmp_path)
    reference = tmp_path / "desired-bundle.json"
    _write_reference(reference, digest)
    settings = _production_settings(
        _policy_file(tmp_path),
        model_cache_path=tmp_path,
        model_bundle_reference=reference,
        model_bundle_version="1",
        model_manifest_sha256=digest,
    )
    assert resolve_analyzer_mode(settings) == "transformer"


@pytest.mark.parametrize("reference_data", [b"not-json", b"{}"])
def test_auto_mode_fails_closed_for_present_invalid_reference(
    tmp_path: Path, reference_data: bytes
) -> None:
    _root, digest = _verified_bundle(tmp_path)
    reference = tmp_path / "desired-bundle.json"
    reference.write_bytes(reference_data)
    settings = _production_settings(
        _policy_file(tmp_path),
        model_cache_path=tmp_path,
        model_bundle_reference=reference,
        model_bundle_version="1",
        model_manifest_sha256=digest,
    )
    with pytest.raises(ValueError, match="desired-bundle"):
        resolve_analyzer_mode(settings)


def test_auto_mode_fails_closed_for_mismatched_reference(tmp_path: Path) -> None:
    _root, digest = _verified_bundle(tmp_path)
    reference = tmp_path / "desired-bundle.json"
    _write_reference(reference, digest, "wrong")
    settings = _production_settings(
        _policy_file(tmp_path),
        model_cache_path=tmp_path,
        model_bundle_reference=reference,
        model_bundle_version="1",
        model_manifest_sha256=digest,
    )
    with pytest.raises(ValueError, match="desired-bundle"):
        resolve_analyzer_mode(settings)


def test_auto_mode_fails_closed_for_corrupt_bundle(tmp_path: Path) -> None:
    root, digest = _verified_bundle(tmp_path)
    reference = tmp_path / "desired-bundle.json"
    _write_reference(reference, digest)
    settings = _production_settings(
        _policy_file(tmp_path),
        model_cache_path=tmp_path,
        model_bundle_reference=reference,
        model_bundle_version="1",
        model_manifest_sha256=digest,
    )
    (root / "english-pii/model.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="desired-bundle"):
        resolve_analyzer_mode(settings)


async def test_transformer_runtime_loses_readiness_when_reference_is_corrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _root, digest = _verified_bundle(tmp_path)
    reference = tmp_path / "desired-bundle.json"
    _write_reference(reference, digest)
    settings = _production_settings(
        _policy_file(tmp_path),
        model_cache_path=tmp_path,
        model_bundle_reference=reference,
        model_bundle_version="1",
        model_manifest_sha256=digest,
    )
    monkeypatch.setattr(
        "pii_engine.runtime.create_analyzer",
        lambda _settings, _policy, _mode: DeterministicAnalyzer(),
    )
    runtime = EngineRuntime(settings)
    assert runtime.analyzer_mode == "transformer"
    assert await runtime.ready() is True
    reference.write_bytes(b"corrupt")
    assert await runtime.ready() is False

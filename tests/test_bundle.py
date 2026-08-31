"""Test the publisher-compatible immutable bundle consumer."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from pii_engine.jobs.model_sync import sync_bundle
from pii_engine.lib.bundle import (
    BundleError,
    atomic_publish,
    desired_cache,
    parse_checksums,
    parse_manifest,
    verify_bundle,
    verify_checksum_index,
)


def _checksum_data(model: bytes = b"model") -> bytes:
    return f"{hashlib.sha256(model).hexdigest()}  english-pii/model.bin\n".encode()


def _manifest(version: str = "0.1.2", checksum_data: bytes | None = None) -> bytes:
    checksum_data = checksum_data or _checksum_data()
    return yaml.safe_dump(
        {
            "schemaVersion": 2,
            "bundleVersion": version,
            "models": {
                "english-pii": {
                    "catalogId": "ai4privacy-english-pii",
                    "variantId": "transformers",
                    "upstream": "ai4privacy/model",
                    "revision": "abc123",
                    "path": "english-pii",
                    "license": "MIT",
                    "licenseUrl": "https://example.test/license",
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
        },
        sort_keys=False,
    ).encode()


def test_manifest_checksum_and_files_are_verified(tmp_path: Path) -> None:
    """The real publisher shape and portable checksum file pass both gates."""
    content = b"model"
    file_digest = hashlib.sha256(content).hexdigest()
    checksum_data = f"{file_digest}  english-pii/model.bin\n".encode()
    manifest_data = _manifest(checksum_data=checksum_data)
    manifest = parse_manifest(manifest_data, hashlib.sha256(manifest_data).hexdigest(), "0.1.2")
    verify_checksum_index(checksum_data, manifest)
    checksums = parse_checksums(checksum_data)
    (tmp_path / "english-pii").mkdir()
    (tmp_path / "english-pii/model.bin").write_bytes(content)
    (tmp_path / "manifest.yaml").write_bytes(manifest_data)
    (tmp_path / "checksums.sha256").write_bytes(checksum_data)
    verify_bundle(tmp_path, manifest, checksums)


def test_checksum_file_rejects_path_traversal() -> None:
    """Object keys cannot escape the temporary or cache roots."""
    with pytest.raises(BundleError, match="unsafe"):
        parse_checksums(("0" * 64 + "  ../model.bin\n").encode())


def test_manifest_rejects_replaced_checksum_index() -> None:
    """The Git-pinned manifest commits to the exact checksum-index bytes."""
    manifest_data = _manifest()
    manifest = parse_manifest(manifest_data, hashlib.sha256(manifest_data).hexdigest(), "0.1.2")
    altered = _checksum_data(b"malicious")
    with pytest.raises(BundleError, match="checksum file SHA-256"):
        verify_checksum_index(altered, manifest)


def test_bundle_rejects_unexpected_files(tmp_path: Path) -> None:
    """Unlisted mixed-version files invalidate a synchronized bundle."""
    manifest_data = _manifest()
    manifest = parse_manifest(manifest_data, hashlib.sha256(manifest_data).hexdigest(), "0.1.2")
    content = b"model"
    digest = hashlib.sha256(content).hexdigest()
    checksums = parse_checksums(f"{digest}  english-pii/model.bin\n".encode())
    (tmp_path / "english-pii").mkdir()
    (tmp_path / "english-pii/model.bin").write_bytes(content)
    (tmp_path / "english-pii/unlisted.bin").write_bytes(b"unexpected")
    (tmp_path / "manifest.yaml").write_bytes(manifest_data)
    (tmp_path / "checksums.sha256").write_text("checksums", encoding="utf-8")
    with pytest.raises(BundleError, match="unexpected"):
        verify_bundle(tmp_path, manifest, checksums)


def test_atomic_publish_is_digest_addressed_and_marker_is_last(tmp_path: Path) -> None:
    """A completed cache is immutable and selected by manifest digest."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.yaml").write_bytes(_manifest())
    digest = hashlib.sha256((source / "manifest.yaml").read_bytes()).hexdigest()
    destination = atomic_publish(source, tmp_path / "cache", digest)
    assert destination.name == digest
    assert (destination / ".complete").read_text() == digest


class _MemoryDownloader:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.requested: list[str] = []

    def download(self, key: str, destination: Path) -> None:
        self.requested.append(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.objects[key])


def test_model_sync_downloads_only_selected_checksum_entries(tmp_path: Path) -> None:
    """The Job consumes one selected immutable prefix and no unrelated objects."""
    model = b"model"
    checksum = _checksum_data(model)
    manifest = _manifest("1", checksum)
    downloader = _MemoryDownloader(
        {
            "1/manifest.yaml": manifest,
            "1/checksums.sha256": checksum,
            "1/english-pii/model.bin": model,
        }
    )
    result = sync_bundle(
        "1",
        hashlib.sha256(manifest).hexdigest(),
        "http://rgw",
        "pii-models",
        tmp_path / "cache",
        downloader,
    )
    assert result is not None
    assert (result / ".complete").is_file()
    assert (
        desired_cache(
            tmp_path / "cache",
            tmp_path / "cache/desired-bundle.json",
            hashlib.sha256(manifest).hexdigest(),
            "1",
        )
        == result
    )
    assert downloader.requested == [
        "1/manifest.yaml",
        "1/checksums.sha256",
        "1/english-pii/model.bin",
    ]


def test_model_sync_rejects_altered_manifest_and_object(tmp_path: Path) -> None:
    """Neither a changed manifest nor corrupted checksum-listed object can be published."""
    model = b"model"
    checksum = _checksum_data(model)
    manifest = _manifest("1", checksum)
    expected_digest = hashlib.sha256(manifest).hexdigest()
    altered_manifest = manifest.replace(b"revision: abc123", b"revision: altered")
    with pytest.raises(BundleError, match="manifest SHA-256"):
        sync_bundle(
            "1",
            expected_digest,
            "http://rgw",
            "pii-models",
            tmp_path / "altered-cache",
            _MemoryDownloader({"1/manifest.yaml": altered_manifest}),
        )

    downloader = _MemoryDownloader(
        {
            "1/manifest.yaml": manifest,
            "1/checksums.sha256": checksum,
            "1/english-pii/model.bin": b"corrupt",
        }
    )
    with pytest.raises(BundleError, match="checksum mismatch"):
        sync_bundle(
            "1",
            expected_digest,
            "http://rgw",
            "pii-models",
            tmp_path / "corrupt-cache",
            downloader,
        )
    assert not (tmp_path / "corrupt-cache" / expected_digest).exists()


def test_model_sync_rejects_replaced_checksum_index_and_matching_model(tmp_path: Path) -> None:
    """An RGW writer cannot replace both model bytes and their checksum index."""
    trusted_checksum = _checksum_data(b"model")
    manifest = _manifest("1", trusted_checksum)
    malicious = b"malicious"
    downloader = _MemoryDownloader(
        {
            "1/manifest.yaml": manifest,
            "1/checksums.sha256": _checksum_data(malicious),
            "1/english-pii/model.bin": malicious,
        }
    )
    with pytest.raises(BundleError, match="checksum file SHA-256"):
        sync_bundle(
            "1",
            hashlib.sha256(manifest).hexdigest(),
            "http://rgw",
            "pii-models",
            tmp_path / "cache",
            downloader,
        )
    assert downloader.requested == ["1/manifest.yaml", "1/checksums.sha256"]


def test_model_sync_replaces_incomplete_digest_directory(tmp_path: Path) -> None:
    """An incomplete prior attempt is replaced without mutating a completed cache."""
    checksum = _checksum_data()
    manifest = _manifest("1", checksum)
    digest = hashlib.sha256(manifest).hexdigest()
    model = b"model"
    checksum = _checksum_data(model)
    incomplete = tmp_path / "cache" / digest
    incomplete.mkdir(parents=True)
    (incomplete / "partial").write_text("invalid", encoding="utf-8")
    result = sync_bundle(
        "1",
        digest,
        "http://rgw",
        "pii-models",
        tmp_path / "cache",
        _MemoryDownloader(
            {
                "1/manifest.yaml": manifest,
                "1/checksums.sha256": checksum,
                "1/english-pii/model.bin": model,
            }
        ),
    )
    assert result is not None
    assert not (result / "partial").exists()
    assert (result / ".complete").read_text(encoding="ascii") == digest


def test_failed_upgrade_preserves_previous_desired_bundle(tmp_path: Path) -> None:
    """A rejected replacement cannot move the reference away from the serving bundle."""
    cache = tmp_path / "cache"
    reference = cache / "desired-bundle.json"
    checksum = _checksum_data()
    first_manifest = _manifest("1", checksum)
    first_digest = hashlib.sha256(first_manifest).hexdigest()
    model = b"model"
    checksum = _checksum_data(model)
    first = sync_bundle(
        "1",
        first_digest,
        "http://rgw",
        "pii-models",
        cache,
        _MemoryDownloader(
            {
                "1/manifest.yaml": first_manifest,
                "1/checksums.sha256": checksum,
                "1/english-pii/model.bin": model,
            }
        ),
        reference,
    )
    previous_reference = reference.read_bytes()

    replacement_manifest = _manifest("2", checksum)
    replacement_digest = hashlib.sha256(replacement_manifest).hexdigest()
    with pytest.raises(BundleError, match="checksum mismatch"):
        sync_bundle(
            "2",
            replacement_digest,
            "http://rgw",
            "pii-models",
            cache,
            _MemoryDownloader(
                {
                    "2/manifest.yaml": replacement_manifest,
                    "2/checksums.sha256": checksum,
                    "2/english-pii/model.bin": b"corrupt",
                }
            ),
            reference,
        )

    assert reference.read_bytes() == previous_reference
    assert desired_cache(cache, reference, first_digest, "1") == first
    assert not (cache / replacement_digest).exists()


def test_rollback_revalidates_and_reselects_previous_digest(tmp_path: Path) -> None:
    """Rollback retains immutable caches and atomically selects the previous digest."""
    cache = tmp_path / "cache"
    reference = cache / "desired-bundle.json"
    model = b"model"
    checksum = f"{hashlib.sha256(model).hexdigest()}  english-pii/model.bin\n".encode()
    destinations: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for version in ("1", "2"):
        manifest = _manifest(version, checksum)
        digests[version] = hashlib.sha256(manifest).hexdigest()
        destination = sync_bundle(
            version,
            digests[version],
            "http://rgw",
            "pii-models",
            cache,
            _MemoryDownloader(
                {
                    f"{version}/manifest.yaml": manifest,
                    f"{version}/checksums.sha256": checksum,
                    f"{version}/english-pii/model.bin": model,
                }
            ),
            reference,
        )
        assert destination is not None
        destinations[version] = destination

    (destinations["1"] / "english-pii/model.bin").write_bytes(b"corrupt")
    first_manifest = _manifest("1", checksum)
    downloader = _MemoryDownloader(
        {
            "1/manifest.yaml": first_manifest,
            "1/checksums.sha256": checksum,
            "1/english-pii/model.bin": model,
        }
    )
    rolled_back = sync_bundle(
        "1",
        digests["1"],
        "http://rgw",
        "pii-models",
        cache,
        downloader,
        reference,
    )
    assert rolled_back == destinations["1"]
    assert desired_cache(cache, reference, digests["1"], "1") == rolled_back
    assert desired_cache(cache, reference, digests["2"], "2") is None
    assert destinations["2"].is_dir()
    assert downloader.requested == [
        "1/manifest.yaml",
        "1/checksums.sha256",
        "1/english-pii/model.bin",
    ]

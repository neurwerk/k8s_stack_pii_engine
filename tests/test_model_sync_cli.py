"""Test the fail-closed model-sync CLI contract."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import yaml
from botocore.exceptions import ClientError

from pii_engine.jobs import model_sync


class _FailingDownloader:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.requested: list[str] = []

    def download(self, key: str, destination: Path) -> None:
        self.requested.append(key)
        del destination
        raise self.error


class _ObjectDownloader:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    def download(self, key: str, destination: Path) -> None:
        if key not in self.objects:
            raise model_sync.ObjectNotFoundError(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.objects[key])


def _checksum() -> bytes:
    return f"{hashlib.sha256(b'model').hexdigest()}  english-pii/model.bin\n".encode()


def _manifest() -> bytes:
    checksum = _checksum()
    return yaml.safe_dump(
        {
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
            "checksumSha256": hashlib.sha256(checksum).hexdigest(),
            "checksumSize": len(checksum),
            "fileCount": 1,
            "totalModelBytes": 5,
        },
        sort_keys=False,
    ).encode()


def _sync(
    tmp_path: Path,
    downloader: model_sync.ObjectDownloader,
    *,
    manifest_sha256: str | None = None,
    missing_manifest_ok: bool = False,
) -> Path | None:
    return model_sync.sync_bundle(
        "1",
        manifest_sha256 or hashlib.sha256(_manifest()).hexdigest(),
        "http://rgw",
        "pii-models",
        tmp_path / "cache",
        downloader,
        tmp_path / "cache/desired-bundle.json",
        missing_manifest_ok,
    )


def _client_error(code: str, operation: str = "HeadObject") -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": code}, "ResponseMetadata": {}}, operation
    )


def test_missing_manifest_can_succeed_without_changing_reference(tmp_path: Path) -> None:
    reference = tmp_path / "cache/desired-bundle.json"
    reference.parent.mkdir()
    reference.write_bytes(b"previous-reference")
    downloader = _FailingDownloader(model_sync.ObjectNotFoundError("1/manifest.yaml"))

    assert _sync(tmp_path, downloader, missing_manifest_ok=True) is None
    assert downloader.requested == ["1/manifest.yaml"]
    assert reference.read_bytes() == b"previous-reference"


def test_missing_manifest_fails_without_override(tmp_path: Path) -> None:
    downloader = _FailingDownloader(model_sync.ObjectNotFoundError("1/manifest.yaml"))
    with pytest.raises(model_sync.ObjectNotFoundError):
        _sync(tmp_path, downloader)


@pytest.mark.parametrize(
    "error",
    [RuntimeError("offline"), _client_error("AccessDenied"), _client_error("NoSuchBucket")],
)
def test_sync_does_not_suppress_connectivity_auth_or_bucket_errors(
    tmp_path: Path, error: BaseException
) -> None:
    with pytest.raises(type(error)):
        _sync(tmp_path, _FailingDownloader(error))


def test_sync_does_not_suppress_missing_checksum_object(tmp_path: Path) -> None:
    downloader = _ObjectDownloader({"1/manifest.yaml": _manifest()})
    with pytest.raises(model_sync.ObjectNotFoundError, match="checksums"):
        _sync(tmp_path, downloader)


def test_sync_does_not_suppress_missing_model_object(tmp_path: Path) -> None:
    manifest = _manifest()
    downloader = _ObjectDownloader({"1/manifest.yaml": manifest, "1/checksums.sha256": _checksum()})
    with pytest.raises(model_sync.ObjectNotFoundError, match=r"model\.bin"):
        _sync(tmp_path, downloader)


def test_sync_does_not_suppress_model_integrity_failure(tmp_path: Path) -> None:
    downloader = _ObjectDownloader(
        {
            "1/manifest.yaml": _manifest(),
            "1/checksums.sha256": _checksum(),
            "1/english-pii/model.bin": b"corrupt",
        }
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        _sync(tmp_path, downloader)


def test_sync_does_not_suppress_malformed_manifest(tmp_path: Path) -> None:
    manifest = b"invalid: ["
    downloader = _ObjectDownloader({"1/manifest.yaml": manifest})
    with pytest.raises(ValueError, match="manifest is invalid"):
        _sync(tmp_path, downloader, manifest_sha256=hashlib.sha256(manifest).hexdigest())


def test_sync_does_not_suppress_manifest_digest_mismatch(tmp_path: Path) -> None:
    downloader = _ObjectDownloader({"1/manifest.yaml": _manifest() + b"changed"})
    with pytest.raises(ValueError, match="manifest SHA-256"):
        _sync(tmp_path, downloader)


@pytest.mark.parametrize("code", ["NoSuchKey", "404", "NotFound"])
def test_s3_downloader_translates_missing_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: str
) -> None:
    class Client:
        def download_file(self, _bucket: str, _key: str, _destination: str) -> None:
            raise _client_error(code)

    monkeypatch.setattr(model_sync.boto3, "client", lambda *_args, **_kwargs: Client())
    downloader = model_sync.S3Downloader("http://rgw", "pii-models")
    with pytest.raises(model_sync.ObjectNotFoundError):
        downloader.download("1/manifest.yaml", tmp_path / "manifest.yaml")


def test_cli_forwards_missing_manifest_override(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_sync(*_args: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(model_sync, "sync_bundle", fake_sync)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pii-engine-model-sync",
            "--version",
            "1",
            "--manifest-sha256",
            "0" * 64,
            "--rgw-endpoint",
            "http://rgw",
            "--bucket",
            "pii-models",
            "--cache-path",
            "/cache",
            "--reference-path",
            "/cache/desired-bundle.json",
            "--missing-manifest-ok",
        ],
    )
    model_sync.main()
    assert captured["missing_manifest_ok"] is True

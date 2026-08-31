"""Download, verify, and atomically publish one immutable RGW model bundle."""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path
from typing import Protocol

import boto3
from botocore.exceptions import ClientError

from pii_engine.lib.bundle import (
    ManifestFile,
    atomic_publish,
    completed_cache,
    parse_checksums,
    parse_manifest,
    select_desired_bundle,
    verify_bundle,
    verify_checksum_index,
)

logger = logging.getLogger(__name__)
_MISSING_OBJECT_CODES = {"404", "NoSuchKey", "NotFound"}


class ObjectNotFoundError(FileNotFoundError):
    """Raise when the requested S3 object, rather than its bucket, is absent."""


class ObjectDownloader(Protocol):
    """Download an object into a local path."""

    def download(self, key: str, destination: Path) -> None:
        """Download one object."""


class S3Downloader:
    """Download objects through one boto3-compatible endpoint and bucket."""

    def __init__(self, endpoint: str, bucket: str) -> None:
        """Create a path-style RGW client from environment-provided credentials."""
        from botocore.config import Config

        self.bucket = bucket
        self.client = boto3.client(
            "s3", endpoint_url=endpoint, config=Config(s3={"addressing_style": "path"})
        )

    def download(self, key: str, destination: Path) -> None:
        """Download one object without logging its content or credentials."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client.download_file(self.bucket, key, str(destination))
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code not in _MISSING_OBJECT_CODES:
                raise
            raise ObjectNotFoundError(key) from exc


def sync_bundle(
    version: str,
    manifest_sha256: str,
    endpoint: str,
    bucket: str,
    cache_path: Path,
    downloader: ObjectDownloader | None = None,
    reference_path: Path | None = None,
    missing_manifest_ok: bool = False,
) -> Path | None:
    """Download only the selected bundle and publish it after full verification."""
    desired_reference = reference_path or cache_path / "desired-bundle.json"
    if (
        not missing_manifest_ok
        and completed_cache(cache_path, manifest_sha256, version) is not None
    ):
        return select_desired_bundle(cache_path, desired_reference, manifest_sha256, version)
    source = downloader or S3Downloader(endpoint, bucket)
    with tempfile.TemporaryDirectory(prefix="pii-model-sync-") as temporary:
        root = Path(temporary)
        try:
            manifest_data = _download_bytes(
                source, f"{version}/manifest.yaml", root / "manifest.yaml"
            )
        except ObjectNotFoundError:
            if missing_manifest_ok:
                return None
            raise
        manifest = parse_manifest(manifest_data, manifest_sha256, version)
        if completed_cache(cache_path, manifest_sha256, version) is not None:
            return select_desired_bundle(cache_path, desired_reference, manifest_sha256, version)
        checksums_data = _download_bytes(
            source, f"{version}/{manifest.checksum_file}", root / manifest.checksum_file
        )
        verify_checksum_index(checksums_data, manifest)
        files = parse_checksums(checksums_data)
        _download_files(source, version, files, root)
        verify_bundle(root, manifest, files)
        atomic_publish(root, cache_path, manifest_sha256)
        return select_desired_bundle(cache_path, desired_reference, manifest_sha256, version)


def _download_bytes(downloader: ObjectDownloader, key: str, destination: Path) -> bytes:
    """Download one metadata object and return its exact bytes."""
    downloader.download(key, destination)
    return destination.read_bytes()


def _download_files(
    downloader: ObjectDownloader, version: str, files: tuple[ManifestFile, ...], root: Path
) -> None:
    """Download exactly the checksum-listed objects below the selected prefix."""
    for entry in files:
        downloader.download(f"{version}/{entry.path}", root / entry.path)


def _parser() -> argparse.ArgumentParser:
    """Build the model-sync command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--rgw-endpoint", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument(
        "--missing-manifest-ok",
        action="store_true",
        help="exit successfully only when the exact version manifest object is absent",
    )
    return parser


def main() -> None:
    """Synchronize the selected bundle and exit non-zero on any validation failure."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parser().parse_args()
    destination = sync_bundle(
        args.version,
        args.manifest_sha256,
        args.rgw_endpoint,
        args.bucket,
        args.cache_path,
        reference_path=args.reference_path,
        missing_manifest_ok=args.missing_manifest_ok,
    )
    if destination is None:
        logger.info("model manifest absent version=%s", args.version)
        return
    logger.info("model bundle verified version=%s digest=%s", args.version, destination.name)


if __name__ == "__main__":
    main()

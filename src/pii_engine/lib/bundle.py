"""Validate and atomically publish immutable PII model bundles."""

from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_CHECKSUM_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


class BundleError(ValueError):
    """Raise when a model bundle fails validation."""


class ModelEntry(BaseModel):
    """Describe one immutable model in the publisher manifest."""

    model_config = ConfigDict(extra="forbid")

    catalog_id: str = Field(alias="catalogId", min_length=1, max_length=128)
    variant_id: str = Field(alias="variantId", min_length=1, max_length=128)
    upstream: str = Field(min_length=1, max_length=512)
    revision: str = Field(min_length=1, max_length=128)
    path: str = Field(min_length=1, max_length=128)
    license: str = Field(min_length=1, max_length=128)
    license_url: str = Field(alias="licenseUrl", min_length=1, max_length=1024)
    supported_languages: list[str] = Field(alias="supportedLanguages", min_length=1)


class RuntimeEntry(BaseModel):
    """Describe the Presidio transformer settings pinned by the bundle."""

    model_config = ConfigDict(extra="forbid")

    labels_to_ignore: list[str] = Field(alias="labelsToIgnore")
    aggregation_strategy: str = Field(alias="aggregationStrategy", min_length=1)
    stride: int = Field(ge=0, le=512)
    model_to_presidio_entity_mapping: dict[str, str] = Field(
        alias="modelToPresidioEntityMapping", min_length=1
    )


class BundleManifest(BaseModel):
    """Represent the exact manifest emitted by the model publisher."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(alias="schemaVersion")
    bundle_version: str = Field(alias="bundleVersion", min_length=1, max_length=128)
    models: dict[str, ModelEntry] = Field(min_length=1, max_length=64)
    runtime: RuntimeEntry
    checksum_file: str = Field(alias="checksumFile", pattern=r"^[A-Za-z0-9._-]+$")
    checksum_sha256: str = Field(alias="checksumSha256", pattern=r"^[0-9a-f]{64}$")
    checksum_size: int = Field(alias="checksumSize", gt=0, le=1_048_576)
    file_count: int = Field(alias="fileCount", gt=0, le=100_000)
    total_model_bytes: int = Field(alias="totalModelBytes", gt=0, le=1_099_511_627_776)

    @model_validator(mode="after")
    def validate_paths(self) -> BundleManifest:
        """Require unique safe model roots and the portable checksum filename."""
        if self.schema_version != 2:
            raise ValueError("unsupported bundle schema version")
        if self.checksum_file != "checksums.sha256":
            raise ValueError("unsupported checksum filename")
        roots = [_safe_relative_path(model.path).as_posix() for model in self.models.values()]
        if len(roots) != len(set(roots)):
            raise ValueError("model paths must be unique")
        return self


class ManifestFile(BaseModel):
    """Describe one checksum-protected bundle file."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=1024)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DesiredBundleReference(BaseModel):
    """Identify the verified bundle selected for the next runtime rollout."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: int = Field(alias="schemaVersion", ge=1, le=1)
    bundle_version: str = Field(alias="bundleVersion", min_length=1, max_length=128)
    manifest_sha256: str = Field(alias="manifestSha256", pattern=r"^[0-9a-f]{64}$")


def parse_manifest(data: bytes, expected_sha256: str, expected_version: str) -> BundleManifest:
    """Verify and parse the exact publisher manifest."""
    if hashlib.sha256(data).hexdigest() != expected_sha256.lower():
        raise BundleError("manifest SHA-256 does not match the expected digest")
    try:
        manifest = BundleManifest.model_validate(yaml.safe_load(data))
    except (TypeError, ValueError, yaml.YAMLError) as exc:
        raise BundleError("manifest is invalid") from exc
    if manifest.bundle_version != expected_version:
        raise BundleError("manifest bundle version does not match the requested version")
    return manifest


def parse_checksums(data: bytes) -> tuple[ManifestFile, ...]:
    """Parse the publisher's portable checksum file."""
    entries: list[ManifestFile] = []
    seen: set[str] = set()
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise BundleError("checksum file is not UTF-8") from exc
    for line in lines:
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise BundleError("checksum file contains an invalid record")
        path = _safe_relative_path(match.group(2)).as_posix()
        if path in seen or path in {"manifest.yaml", "checksums.sha256", ".complete"}:
            raise BundleError("checksum file contains a duplicate or reserved path")
        seen.add(path)
        entries.append(ManifestFile(path=path, sha256=match.group(1)))
    if not entries:
        raise BundleError("checksum file contains no model files")
    return tuple(entries)


def verify_checksum_index(data: bytes, manifest: BundleManifest) -> None:
    """Verify the manifest-bound exact checksum-index bytes before parsing them."""
    if len(data) != manifest.checksum_size:
        raise BundleError("checksum file size does not match the manifest")
    if hashlib.sha256(data).hexdigest() != manifest.checksum_sha256:
        raise BundleError("checksum file SHA-256 does not match the manifest")


def verify_bundle(root: Path, manifest: BundleManifest, files: tuple[ManifestFile, ...]) -> None:
    """Verify model roots, every listed digest, and the absence of extra files."""
    listed = {entry.path for entry in files}
    if len(files) != manifest.file_count:
        raise BundleError("checksum file count does not match the manifest")
    for model in manifest.models.values():
        prefix = _safe_relative_path(model.path).as_posix().rstrip("/") + "/"
        if not any(path.startswith(prefix) for path in listed):
            raise BundleError(f"model has no checksum-protected files: {model.path}")
    for entry in files:
        file_path = root / _safe_relative_path(entry.path)
        if file_path.is_symlink() or not file_path.is_file():
            raise BundleError(f"bundle file is missing or unsafe: {entry.path}")
        if _file_sha256(file_path) != entry.sha256:
            raise BundleError(f"bundle checksum mismatch: {entry.path}")
    if sum((root / _safe_relative_path(entry.path)).stat().st_size for entry in files) != (
        manifest.total_model_bytes
    ):
        raise BundleError("bundle byte size does not match the manifest")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    expected = listed | {"manifest.yaml", manifest.checksum_file}
    if (root / ".complete").is_file():
        expected.add(".complete")
    if actual != expected:
        raise BundleError("bundle contains missing or unexpected files")


def cache_directory(cache_path: Path, manifest_sha256: str) -> Path:
    """Return the immutable digest-addressed cache directory."""
    return cache_path / manifest_sha256.lower()


def completed_cache(
    cache_path: Path, manifest_sha256: str, expected_version: str | None = None
) -> Path | None:
    """Return a complete cache only after revalidating every model file."""
    destination = cache_directory(cache_path, manifest_sha256)
    try:
        marker = (destination / ".complete").read_text(encoding="ascii").strip()
        manifest_data = (destination / "manifest.yaml").read_bytes()
        raw = yaml.safe_load(manifest_data)
        version = raw.get("bundleVersion") if isinstance(raw, dict) else None
        if not isinstance(version, str):
            return None
        manifest = parse_manifest(manifest_data, manifest_sha256, expected_version or version)
        checksum_data = (destination / manifest.checksum_file).read_bytes()
        verify_checksum_index(checksum_data, manifest)
        files = parse_checksums(checksum_data)
        verify_bundle(destination, manifest, files)
    except (OSError, UnicodeError, BundleError, TypeError, ValueError, yaml.YAMLError):
        return None
    if marker != manifest_sha256.lower():
        return None
    if expected_version is not None and version != expected_version:
        return None
    return destination


def desired_cache(
    cache_path: Path,
    reference_path: Path,
    manifest_sha256: str,
    expected_version: str,
) -> Path | None:
    """Return the complete cache selected by the atomic desired-bundle reference."""
    if not desired_reference_matches(cache_path, reference_path, manifest_sha256, expected_version):
        return None
    return completed_cache(cache_path, manifest_sha256, expected_version)


def desired_reference_matches(
    cache_path: Path,
    reference_path: Path,
    manifest_sha256: str,
    expected_version: str,
) -> bool:
    """Validate the bounded desired reference without rehashing the selected bundle."""
    try:
        _validate_reference_path(cache_path, reference_path)
        if reference_path.is_symlink() or not reference_path.is_file():
            return False
        data = reference_path.read_bytes()
        if len(data) > 4_096:
            return False
        reference = DesiredBundleReference.model_validate_json(data)
    except (OSError, UnicodeError, ValueError):
        return False
    return (
        reference.bundle_version == expected_version
        and reference.manifest_sha256 == manifest_sha256.lower()
    )


def desired_reference_selects_other_complete_cache(
    cache_path: Path,
    reference_path: Path,
    manifest_sha256: str,
    expected_version: str,
) -> bool:
    """Return true only for a valid previous selection during a pin transition."""
    try:
        _validate_reference_path(cache_path, reference_path)
        if reference_path.is_symlink() or not reference_path.is_file():
            return False
        data = reference_path.read_bytes()
        if len(data) > 4_096:
            return False
        reference = DesiredBundleReference.model_validate_json(data)
    except (OSError, UnicodeError, ValueError):
        return False
    if (
        reference.bundle_version == expected_version
        and reference.manifest_sha256 == manifest_sha256.lower()
    ):
        return False
    return (
        completed_cache(cache_path, reference.manifest_sha256, reference.bundle_version) is not None
    )


def desired_reference_absent(reference_path: Path) -> bool:
    """Return true only when the desired-reference path genuinely does not exist."""
    try:
        reference_path.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def select_desired_bundle(
    cache_path: Path,
    reference_path: Path,
    manifest_sha256: str,
    expected_version: str,
) -> Path:
    """Atomically select one fully revalidated immutable cache directory."""
    _validate_reference_path(cache_path, reference_path)
    destination = completed_cache(cache_path, manifest_sha256, expected_version)
    if destination is None:
        raise BundleError("cannot select an incomplete or invalid model bundle")
    reference = DesiredBundleReference(
        schema_version=1,
        bundle_version=expected_version,
        manifest_sha256=manifest_sha256.lower(),
    )
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="ascii", prefix=".desired-bundle-", dir=cache_path, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(reference.model_dump_json(by_alias=True) + "\n")
    try:
        temporary.replace(reference_path)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_publish(source: Path, cache_path: Path, manifest_sha256: str) -> Path:
    """Publish a verified bundle without mutating any completed directory."""
    cache_path.mkdir(parents=True, exist_ok=True)
    destination = cache_directory(cache_path, manifest_sha256)
    existing = completed_cache(cache_path, manifest_sha256)
    if existing is not None:
        return existing
    if destination.exists():
        shutil.rmtree(destination)
    temporary = Path(tempfile.mkdtemp(prefix=f".{manifest_sha256[:12]}-", dir=cache_path))
    try:
        staged = temporary / "bundle"
        shutil.copytree(source, staged)
        (staged / ".complete").write_text(manifest_sha256.lower(), encoding="ascii")
        staged.replace(destination)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _safe_relative_path(value: str) -> Path:
    """Return a normalized safe relative path."""
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or path.as_posix() != value:
        raise BundleError(f"unsafe bundle path: {value}")
    return path


def _validate_reference_path(cache_path: Path, reference_path: Path) -> None:
    """Keep the mutable reference beside, never inside, immutable bundle directories."""
    if reference_path.parent.resolve() != cache_path.resolve() or reference_path.name in {
        "",
        ".",
        "..",
    }:
        raise BundleError("desired-bundle reference must be a file in the cache root")


def _file_sha256(path: Path) -> str:
    """Compute a file SHA-256 digest in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

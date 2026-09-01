"""Test deterministic CPU and CUDA image packaging configuration."""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

from pii_engine.config.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


def test_accelerator_extras_are_mutually_exclusive_and_pinned() -> None:
    """CPU and CUDA builds must not resolve into one oversized environment."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = config["project"]["optional-dependencies"]
    assert extras["cpu"] == ["torch==2.6.0"]
    assert any(item.startswith("cupy-cuda12x==13.6.0") for item in extras["cu124"])
    assert config["tool"]["uv"]["conflicts"] == [[{"extra": "cpu"}, {"extra": "cu124"}]]


def test_torch_sources_use_official_accelerator_indexes() -> None:
    """Each Linux image must resolve Torch from its explicit official index."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    indexes = {item["name"]: item["url"] for item in config["tool"]["uv"]["index"]}
    assert indexes == {
        "pytorch-cpu": "https://download.pytorch.org/whl/cpu",
        "pytorch-cu124": "https://download.pytorch.org/whl/cu124",
    }


def test_images_are_variant_selected_and_exclude_external_model_bundles() -> None:
    """Docker builds install one extra and never copy external model bundles."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert '--extra "${ACCELERATOR}"' in dockerfile
    assert "cu124:amd64" in dockerfile
    assert "COPY models" not in dockerfile
    assert "offline spaCy baseline models" in dockerfile
    assert "org.opencontainers.image.licenses" not in dockerfile


def test_example_secrets_satisfy_runtime_length_validation() -> None:
    """Redacted example keys must remain valid replacements for local parsing."""
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    values = dict(line.split("=", 1) for line in lines if line and not line.startswith("#"))
    settings = Settings(
        policy_config=ROOT / "example-policy.yaml",
        hash_key=values["PII_ENGINE_HASH_KEY"],
        encryption_key=values["PII_ENGINE_ENCRYPTION_KEY"],
        allow_test_analyzer=False,
    )

    assert settings.hash_key is not None
    assert settings.encryption_key is not None
    assert len(settings.hash_key.get_secret_value().encode("ascii")) >= 32
    assert len(settings.encryption_key.get_secret_value().encode("ascii")) == 32


def test_ci_publishes_cpu_and_cuda_images_only_for_version_tags() -> None:
    """Release tags must publish both supported image variants."""
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/build-push.yaml").read_text(encoding="utf-8")
    )
    build = workflow["jobs"]["build"]
    assert build["if"] == "startsWith(github.ref, 'refs/tags/v')"
    assert build["strategy"]["matrix"]["include"] == [
        {"variant": "cpu", "platforms": "linux/amd64"},
        {"variant": "cu124", "platforms": "linux/amd64"},
    ]

    metadata = next(step for step in build["steps"] if step.get("id") == "meta")
    assert metadata["with"]["flavor"] == "latest=false"
    assert metadata["with"]["tags"].splitlines() == [
        "type=semver,pattern={{version}}-${{ matrix.variant }}",
    ]


def test_ci_uses_least_privilege_and_verifies_release_version() -> None:
    """Image publishing keeps read-only defaults and rejects mismatched tags."""
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/build-push.yaml").read_text(encoding="utf-8")
    )
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["build"]["permissions"] == {
        "contents": "read",
        "packages": "write",
    }
    version_step = next(
        step
        for step in workflow["jobs"]["quality"]["steps"]
        if step["name"] == "Verify release tag matches package version"
    )
    assert version_step["if"] == "startsWith(github.ref, 'refs/tags/v')"
    assert version_step["run"] == 'python scripts/verify_release_tag.py "${GITHUB_REF_NAME}"'

    release = workflow["jobs"]["release"]
    assert release["needs"] == "build"
    assert release["permissions"] == {"contents": "write"}

    required_ci = workflow["jobs"]["required_ci"]
    assert required_ci["name"] == "Required CI"
    assert required_ci["needs"] == ["quality", "build", "release"]
    assert "always()" in required_ci["if"]

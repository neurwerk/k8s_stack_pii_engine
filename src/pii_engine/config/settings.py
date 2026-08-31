"""Environment-backed process settings for the PII Engine."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configure listeners, bounds, identities, storage, and runtime secrets."""

    model_config = SettingsConfigDict(env_prefix="PII_ENGINE_")

    analysis_host: str = "0.0.0.0"
    analysis_port: int = Field(default=8443, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    management_host: str = "0.0.0.0"
    management_port: int = Field(default=8001, ge=1, le=65535)
    max_request_bytes: int = Field(default=5_242_880, ge=1_024, le=5_242_880)
    max_adapter_response_bytes: int = Field(default=10_485_760, ge=1_024, le=10_485_760)
    max_studio_evaluation_response_bytes: int = Field(default=10_485_760, ge=1_024, le=10_485_760)
    max_text_leaves: int = Field(default=256, ge=1, le=256)
    max_text_characters: int = Field(default=4_000_000, ge=1, le=4_000_000)
    max_nesting_depth: int = Field(default=32, ge=1, le=128)
    max_concurrent_analyses: int = Field(default=4, ge=1, le=128)
    max_queued_analyses: int = Field(default=32, ge=0, le=1_024)
    studio_max_concurrent_analyses: int = Field(default=1, ge=1, le=32)
    queue_wait_timeout: float = Field(default=2.0, gt=0, le=30)
    analysis_timeout: float = Field(default=600.0, gt=0, le=600)
    studio_analysis_timeout: float = Field(default=30.0, gt=0, le=30)
    device: str = Field(default="cpu", pattern=r"^(?:cpu|cuda(?::\d+)?)$")
    policy_version: str = Field(default="v1", min_length=1, max_length=64)
    policy_config: Path | None = None
    tls_cert: Path | None = None
    tls_key: Path | None = None
    tls_ca: Path | None = None
    adapter_common_name: str = "monitor-agentgateway-extproc"
    studio_common_name: str = "frontend-studio-api"
    enforce_client_identity: bool = True
    model_cache_path: Path | None = None
    model_bundle_reference: Path | None = None
    model_bundle_version: str | None = None
    model_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    valkey_url: SecretStr | None = None
    hash_key: SecretStr | None = None
    encryption_key: SecretStr | None = None
    allow_test_analyzer: bool = False

    @model_validator(mode="after")
    def validate_runtime_contract(self) -> Settings:
        """Require complete production groups while allowing explicit isolated tests."""
        tls = (self.tls_cert, self.tls_key, self.tls_ca)
        if any(tls) and not all(tls):
            raise ValueError("TLS cert, key, and CA must be configured together")
        model = (self.model_cache_path, self.model_bundle_version, self.model_manifest_sha256)
        if any(model) and not all(model):
            raise ValueError(
                "model cache, version, and manifest digest must be configured together"
            )
        if self.model_bundle_reference is not None and self.model_cache_path is None:
            raise ValueError("desired-bundle reference requires the model cache")
        if not self.allow_test_analyzer:
            if self.policy_config is None:
                raise ValueError("production runtime requires policy configuration")
            if self.hash_key is None or len(self.hash_key.get_secret_value()) < 32:
                raise ValueError("production runtime requires a 32-byte rolling hash key")
            if self.encryption_key is None or len(self.encryption_key.get_secret_value()) != 32:
                raise ValueError("production runtime requires an exact 32-byte encryption key")
        return self

    @property
    def model_bundle_path(self) -> Path | None:
        """Return the immutable digest-addressed model directory."""
        if self.model_cache_path is None or self.model_manifest_sha256 is None:
            return None
        return self.model_cache_path / self.model_manifest_sha256.lower()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process settings singleton."""
    return Settings()

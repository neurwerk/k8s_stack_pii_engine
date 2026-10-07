"""Trusted deployment configuration for external NER services."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pii_engine.lib.catalog import ENTITY_CATALOG
from pii_engine.lib.remote_models import (
    GERMAN_LABEL_MAPPING,
    GLINER_LABEL_MAPPING,
    KSERVE_MODEL_PINS,
    KSERVE_TOKENIZER_SHA256,
)


class RemoteModel(BaseModel):
    """Describe one endpoint and its explicit normalized label contract."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")
    kind: Literal["gliner", "kserve"]
    url: str = Field(max_length=2048)
    model_name: str = Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")
    languages: list[Literal["en", "de", "nl"]] = Field(min_length=1)
    label_mapping: dict[str, str] = Field(default_factory=dict, max_length=100)
    api_key_file: Path | None = None
    allow_private_http: bool = False
    tokenizer_path: Path | None = None
    tokenizer_sha256: dict[str, str] = Field(default_factory=dict)
    upstream: str | None = None
    revision: str | None = None
    inference_threshold: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_contract(self) -> RemoteModel:
        """Reject ambiguous endpoints, entities and incomplete tokenizer pins."""
        self._apply_profile()
        self._validate_url()
        if len(self.languages) != len(set(self.languages)):
            raise ValueError("remote model languages are duplicated")
        if self.kind == "gliner" and self.inference_threshold is None:
            raise ValueError("GLiNER requires its configured server-side detection threshold")
        if any(value not in ENTITY_CATALOG for value in self.label_mapping.values()):
            raise ValueError("remote label mapping selects an unknown policy entity")
        if self.kind == "kserve" and (self.tokenizer_path is None or not self.tokenizer_sha256):
            raise ValueError("KServe requires an exact offline tokenizer and file digest pins")
        return self

    def _apply_profile(self) -> None:
        if self.kind == "kserve":
            if len(self.languages) != 1 or self.languages[0] not in KSERVE_MODEL_PINS:
                raise ValueError("KServe requires one supported English or German model")
            upstream, revision = KSERVE_MODEL_PINS[self.languages[0]]
            self.upstream = self.upstream or upstream
            self.revision = self.revision or revision
            if (self.upstream, self.revision) != (upstream, revision):
                raise ValueError("KServe model must match its supported immutable pin")
            if not self.tokenizer_sha256:
                self.tokenizer_sha256 = KSERVE_TOKENIZER_SHA256[self.languages[0]].copy()
        if not self.label_mapping:
            if self.kind == "gliner":
                self.label_mapping = GLINER_LABEL_MAPPING.copy()
            else:
                self.label_mapping = (
                    {"PRIVATE": "PRIVATE"}
                    if self.languages == ["en"]
                    else GERMAN_LABEL_MAPPING.copy()
                )
        if "PRIVATE" in self.label_mapping and self.label_mapping["PRIVATE"] != "PRIVATE":
            raise ValueError("PRIVATE must remain an independent policy entity")

    def _validate_url(self) -> None:
        url = urlsplit(self.url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or (url.scheme == "http" and not self.allow_private_http)
        ):
            raise ValueError("remote endpoint requires HTTPS or explicit private HTTP approval")
        expected = "/extract" if self.kind == "gliner" else f"/v1/models/{self.model_name}:predict"
        if url.path != expected:
            raise ValueError("remote endpoint path does not match its protocol and model")


class RemoteConfig(BaseModel):
    """Select one multilingual service or independently configured language services."""

    model_config = ConfigDict(extra="forbid")
    models: list[RemoteModel] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def unique_names(self) -> RemoteConfig:
        """Require stable unique endpoint identities."""
        if len({model.name for model in self.models}) != len(self.models):
            raise ValueError("remote model names are duplicated")
        return self

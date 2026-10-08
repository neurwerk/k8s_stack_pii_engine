"""Deployment-owned NER selection and versioned adapter profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pii_engine.config.remote import RemoteModel
from pii_engine.lib.remote_models import (
    GERMAN_LABEL_MAPPING,
    GLINER_LABEL_MAPPING,
    KSERVE_MODEL_PINS,
    KSERVE_TOKENIZER_SHA256,
)


class NerModel(BaseModel):
    """Describe deployment identity without copying adapter recipes."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    profile: str
    endpoint: str | None = None
    model_name: str = Field(default="ner-multilingual", alias="modelName")
    upstream: str | None = Field(default=None, min_length=1, max_length=256)
    revision: str | None = Field(default=None, min_length=1, max_length=128)
    inference_threshold: float | None = Field(default=None, alias="inferenceThreshold", ge=0, le=1)
    allow_private_http: bool = Field(default=False, alias="allowPrivateHttp")
    api_key_file: Path | None = Field(default=None, alias="apiKeyFile")
    tokenizer_path: Path | None = Field(default=None, alias="tokenizerPath")

    @property
    def revision_state(self) -> Literal["declared", "unattested"]:
        """Never describe a deployment declaration as inference attestation."""
        profile = PROFILES.get(self.profile)
        return "declared" if self.revision or (profile and profile.revision) else "unattested"


class NerCapacity(BaseModel):
    """Bound the existing single-instance remote protocols."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    call_timeout: float = Field(default=10, alias="callTimeout", gt=0, le=60)
    max_calls: int = Field(default=2048, alias="maxCalls", ge=1, le=10000)
    max_response_bytes: int = Field(default=2097152, alias="maxResponseBytes", ge=1024, le=8388608)
    max_concurrent_calls: int = Field(default=1, alias="maxConcurrentCalls", ge=1, le=16)


class ModelProfile(BaseModel):
    """Keep model-specific metadata independent from adapter implementation."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["spacy", "gliner", "kserve"]
    languages: list[Literal["en", "de", "nl"]]
    upstream: str
    revision: str | None = None
    label_mapping: dict[str, str] = Field(default_factory=dict)
    ignored_labels: list[str] = Field(default_factory=list)
    tokenizer: Literal["spacy", "offline-fast", "server-managed"] = "spacy"
    tokenizer_sha256: dict[str, str] = Field(default_factory=dict)
    max_characters: int = Field(default=1024, gt=0)
    overlap_characters: int = Field(default=128, ge=0)
    max_tokens: int | None = Field(default=None, gt=0)
    window_tokens: int = Field(default=448, gt=0)
    overlap_tokens: int = Field(default=64, ge=0)
    max_words: int | None = Field(default=None, gt=0)
    prompt_overhead: Literal["none", "tokenizer-special-tokens", "server-managed"] = "none"
    batch_size: Literal[1] = 1
    default_capacity: NerCapacity = Field(default_factory=NerCapacity)

    @model_validator(mode="after")
    def validate_limits(self) -> ModelProfile:
        """Require progress and room for tokenizer overhead in every window."""
        if self.overlap_characters >= self.max_characters:
            raise ValueError("profile character overlap must be smaller than its window")
        if self.overlap_tokens >= self.window_tokens:
            raise ValueError("profile token overlap must be smaller than its window")
        if self.kind == "kserve" and (
            self.max_tokens is None
            or self.window_tokens >= self.max_tokens
            or not self.revision
            or not self.tokenizer_sha256
            or not self.label_mapping
        ):
            raise ValueError("KServe profile requires immutable tokenizer and bounded windows")
        expected_tokenizer = {
            "spacy": "spacy",
            "gliner": "server-managed",
            "kserve": "offline-fast",
        }
        if self.tokenizer != expected_tokenizer[self.kind]:
            raise ValueError("profile tokenizer does not match its adapter")
        return self


_SPACY_MODELS: dict[Literal["en", "de", "nl"], str] = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "nl": "nl_core_news_sm",
}
PROFILES = {
    f"spacy-{language}-sm-v1": ModelProfile(
        kind="spacy",
        languages=[language],
        upstream=model,
        revision="3.8.0",
        label_mapping={"PER": "PERSON_NAME", "PERSON": "PERSON_NAME", "LOC": "CITY", "GPE": "CITY"},
        ignored_labels=[
            "CARDINAL",
            "DATE",
            "EVENT",
            "FAC",
            "LANGUAGE",
            "LAW",
            "MISC",
            "MONEY",
            "NORP",
            "ORDINAL",
            "ORG",
            "PERCENT",
            "PRODUCT",
            "QUANTITY",
            "TIME",
            "WORK_OF_ART",
        ],
        max_characters=900000,
        overlap_characters=10000,
    )
    for language, model in _SPACY_MODELS.items()
}
PROFILES["gliner-multilingual-pii-v1"] = ModelProfile(
    kind="gliner",
    languages=["en", "de"],
    upstream="urchade/gliner_multi_pii-v1",
    label_mapping=GLINER_LABEL_MAPPING,
    tokenizer="server-managed",
    max_words=384,
    max_tokens=512,
    prompt_overhead="server-managed",
)
_KSERVE_NAMES: dict[Literal["en", "de", "nl"], str] = {"en": "openpii", "de": "superclinical"}
for _language, _name in _KSERVE_NAMES.items():
    _upstream, _revision = KSERVE_MODEL_PINS[_language]
    PROFILES[f"kserve-{_language}-{_name}-v1"] = ModelProfile(
        kind="kserve",
        languages=[_language],
        upstream=_upstream,
        revision=_revision,
        label_mapping={"PRIVATE": "PRIVATE"} if _language == "en" else GERMAN_LABEL_MAPPING,
        tokenizer_sha256=KSERVE_TOKENIZER_SHA256[_language],
        tokenizer="offline-fast",
        max_tokens=512,
        prompt_overhead="tokenizer-special-tokens",
    )


class NerConfig(BaseModel):
    """Select exactly one deployment-level NER mode."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    mode: Literal["disabled", "local", "remote"]
    language_models: dict[str, str] = Field(default_factory=dict, alias="languageModels")
    models: dict[str, NerModel] = Field(default_factory=dict, max_length=3)
    capacity: NerCapacity = Field(default_factory=NerCapacity)

    @model_validator(mode="after")
    def validate_selection(self) -> NerConfig:
        """Reject missing references, unsupported languages and mixed local modes."""
        if self.mode == "disabled":
            if self.models or self.language_models or "capacity" in self.model_fields_set:
                raise ValueError("disabled NER cannot select models or capacity")
            return self
        if not self.models or set(self.language_models.values()) != set(self.models):
            raise ValueError("NER model references must be complete and all models referenced")
        for language, name in self.language_models.items():
            if (
                not name
                or len(name) > 64
                or not name[0].islower()
                or any(
                    character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in name
                )
            ):
                raise ValueError("NER model ID is invalid")
            model = self.models[name]
            profile = PROFILES.get(model.profile)
            if profile is None or language not in profile.languages:
                raise ValueError("NER profile does not support its assigned language")
            if (profile.kind == "spacy") != (self.mode == "local"):
                raise ValueError("NER profile does not match its mode")
            if profile.kind == "spacy":
                self._validate_local(model, profile)
            else:
                self.remote_model(name)
        if self.mode == "remote" and "capacity" not in self.model_fields_set:
            defaults = [PROFILES[model.profile].default_capacity for model in self.models.values()]
            self.capacity = NerCapacity(
                call_timeout=min(item.call_timeout for item in defaults),
                max_calls=min(item.max_calls for item in defaults),
                max_response_bytes=min(item.max_response_bytes for item in defaults),
                max_concurrent_calls=min(item.max_concurrent_calls for item in defaults),
            )
        return self

    def _validate_local(self, model: NerModel, profile: ModelProfile) -> None:
        """Keep local resources and immutable identities unambiguous."""
        if "capacity" in self.model_fields_set:
            raise ValueError("local NER cannot configure remote capacity")
        remote_fields = {
            "endpoint",
            "model_name",
            "inference_threshold",
            "allow_private_http",
            "api_key_file",
            "tokenizer_path",
        }
        if remote_fields.intersection(model.model_fields_set):
            raise ValueError("local NER cannot configure remote resources")
        if (model.upstream or profile.upstream) != profile.upstream or (
            model.revision or profile.revision
        ) != profile.revision:
            raise ValueError("local NER identity conflicts with its profile")

    def remote_model(self, name: str) -> RemoteModel:
        """Resolve reviewed profile data into the existing transport contract."""
        model = self.models[name]
        profile = PROFILES[model.profile]
        if profile.kind == "spacy" or model.endpoint is None:
            raise ValueError("remote NER requires an endpoint")
        if profile.kind == "kserve" and "model_name" not in model.model_fields_set:
            raise ValueError("KServe requires an explicit modelName")
        if profile.kind == "gliner" and model.tokenizer_path is not None:
            raise ValueError("GLiNER uses server-managed tokenization")
        if profile.revision and (model.revision or profile.revision) != profile.revision:
            raise ValueError("NER revision conflicts with its profile")
        if (model.upstream or profile.upstream) != profile.upstream:
            raise ValueError("NER upstream conflicts with its profile")
        return RemoteModel(
            name=name,
            kind=profile.kind,
            url=model.endpoint,
            model_name=model.model_name,
            languages=profile.languages,
            label_mapping=profile.label_mapping,
            upstream=profile.upstream,
            revision=model.revision or profile.revision,
            api_key_file=model.api_key_file,
            allow_private_http=model.allow_private_http,
            tokenizer_path=model.tokenizer_path,
            tokenizer_sha256=profile.tokenizer_sha256,
            inference_threshold=model.inference_threshold,
            profile=model.profile,
        )


def load_ner(path: Path) -> NerConfig:
    """Read one strict YAML or JSON NER document without remote downloads."""
    try:
        return NerConfig.model_validate(yaml.safe_load(path.read_bytes()))
    except (OSError, ValueError, yaml.YAMLError):
        raise ValueError("NER configuration is invalid") from None

"""Strict, engine-owned policy configuration models."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pii_engine.lib.actions import ACTION_BY_NAME


class PolicyConfigError(ValueError):
    """Raise when policy configuration cannot be trusted."""


class PolicyModel(BaseModel):
    """Reject unrecognized policy settings."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class OperatorParams(PolicyModel):
    """Configure Presidio's default masking operator."""

    type: Literal["mask", "replace", "redact"] = "mask"
    masking_char: str = Field(default="*", min_length=1, max_length=1)
    chars_to_mask: int = Field(default=100, ge=0, le=100_000)
    from_end: bool = True
    new_value: str | None = Field(default=None, max_length=1_000)


class EntityPolicy(PolicyModel):
    """Select one action and its validated parameters for an entity."""

    entity_type: str = Field(alias="entityType", pattern=r"^[A-Z][A-Z0-9_]{0,63}$")
    action: str
    patterns: list[str] = Field(default_factory=list, max_length=64)
    route_class: str | None = Field(
        default=None, alias="routeClass", max_length=128, pattern=r"^[A-Za-z0-9_.:/-]+$"
    )
    masking_char: str = Field(default="*", min_length=1, max_length=1)
    chars_to_mask: int = Field(default=100, ge=0, le=100_000)
    from_end: bool = True
    new_value: str | None = Field(default=None, max_length=1_000)
    hash_type: Literal["sha256"] = "sha256"

    @model_validator(mode="after")
    def validate_action(self) -> EntityPolicy:
        """Validate action-specific fields and every configured regex."""
        if self.action not in ACTION_BY_NAME:
            raise ValueError(f"unsupported entity action: {self.action}")
        if self.route_class is not None and self.action != "reroute":
            raise ValueError("routeClass is valid only for reroute")
        for pattern in self.patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid entity regex for {self.entity_type}") from exc
        return self


class NerSettings(PolicyModel):
    """Select immutable model aliases from the synchronized bundle."""

    strategy: Literal["multilingual", "perLanguage"] = "perLanguage"
    general_model: str = Field(default="multilingual-pii", alias="generalModel")
    per_language: dict[str, str] = Field(
        default_factory=lambda: {
            "en": "english-pii",
            "de": "german-pii",
            "nl": "dutch-pii",
        },
        alias="perLanguage",
    )


class CustomRecognizer(PolicyModel):
    """Configure one customer recognizer for selected languages."""

    name: str = Field(min_length=1, max_length=128)
    entity: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")
    regex: str = Field(min_length=1, max_length=4_000)
    score: float = Field(default=0.85, gt=0, le=1)
    supported_languages: list[str] = Field(alias="supportedLanguages", min_length=1)

    @model_validator(mode="after")
    def compile_regex(self) -> CustomRecognizer:
        """Fail startup instead of silently skipping an invalid recognizer."""
        try:
            re.compile(self.regex)
        except re.error as exc:
            raise ValueError(f"invalid custom recognizer regex: {self.name}") from exc
        return self


class PiiSettings(PolicyModel):
    """Configure languages, entities, actions, and model selection."""

    analyzer_languages: list[str] = Field(alias="analyzerLanguages", min_length=1)
    supported_languages: list[str] = Field(
        default_factory=lambda: ["en", "de", "nl"], alias="supportedLanguages"
    )
    score_threshold: float = Field(default=0.45, alias="scoreThreshold", ge=0, le=1)
    timeout: float = Field(default=600.0, gt=0, le=600)
    default_action: str = Field(default="block", alias="defaultAction")
    default_operator: OperatorParams = Field(
        default_factory=OperatorParams, alias="defaultOperator"
    )
    analyzer_entities: list[str] = Field(default_factory=list, alias="analyzerEntities")
    mask_on_reroute: bool = Field(default=True, alias="maskOnReroute")
    hash_window_hours: int = Field(default=24, alias="hashWindowHours", ge=1, le=720)
    entity_policies: list[EntityPolicy] = Field(default_factory=list, alias="entityPolicies")
    ner: NerSettings = Field(default_factory=NerSettings)
    custom_recognizers: list[CustomRecognizer] = Field(
        default_factory=list, alias="customRecognizers"
    )

    @model_validator(mode="after")
    def validate_catalog_selection(self) -> PiiSettings:
        """Reject duplicate policies, unsupported languages, and invalid defaults."""
        if self.default_action not in ACTION_BY_NAME:
            raise ValueError("unsupported default action")
        if not set(self.analyzer_languages).issubset(self.supported_languages):
            raise ValueError("analyzer language is not supported")
        if len(self.supported_languages) != len(set(self.supported_languages)):
            raise ValueError("supported language is duplicated")
        if len(self.analyzer_languages) != len(set(self.analyzer_languages)):
            raise ValueError("analyzer language is duplicated")
        names = [entry.entity_type for entry in self.entity_policies]
        if len(names) != len(set(names)):
            raise ValueError("entity policy is duplicated")
        custom_names = [entry.entity for entry in self.custom_recognizers]
        if len(custom_names) != len(set(custom_names)):
            raise ValueError("custom recognizer entity is duplicated")
        if any(
            not set(recognizer.supported_languages).issubset(self.supported_languages)
            for recognizer in self.custom_recognizers
        ):
            raise ValueError("custom recognizer language is not supported")
        return self


class AttachmentsSettings(PolicyModel):
    """Configure unsupported attachment handling."""

    policy: Literal["block"] = "block"


class SafetyRuleEntry(PolicyModel):
    """Configure one additional blocking safety expression."""

    name: str = Field(min_length=1, max_length=128)
    pattern: str = Field(min_length=1, max_length=4_000)
    action: Literal["block"] = "block"
    message: str = Field(default="Content blocked by safety rule", max_length=1_000)

    @model_validator(mode="after")
    def compile_pattern(self) -> SafetyRuleEntry:
        """Reject invalid custom safety rules at startup."""
        try:
            re.compile(self.pattern)
        except re.error as exc:
            raise ValueError(f"invalid safety regex: {self.name}") from exc
        return self


class SafetySettings(PolicyModel):
    """Select built-in and custom safety rules."""

    enabled: list[str] = Field(default_factory=list)
    custom: list[SafetyRuleEntry] = Field(default_factory=list)


class ClassifierClass(PolicyModel):
    """Configure one ordered transformed-content route class."""

    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:/-]+$")
    patterns: list[str] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def compile_patterns(self) -> ClassifierClass:
        """Reject invalid classifier patterns at startup."""
        try:
            for pattern in self.patterns:
                re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid classifier regex: {self.name}") from exc
        return self


class ClassifierSettings(PolicyModel):
    """Configure ordered classification after transformation."""

    default_class: str = Field(default="general", alias="defaultClass")
    classes: list[ClassifierClass] = Field(default_factory=list)


class SessionSettings(PolicyModel):
    """Configure Valkey-backed block/reroute session tainting."""

    enabled: bool = True
    ttl_hours: float = Field(default=24, alias="ttlHours", gt=0, le=720)


class NoticeSettings(PolicyModel):
    """Configure safe request and response notice messages."""

    rerouted: str = Field(max_length=4_000)
    masked: str = Field(max_length=4_000)
    show_when_no_pii_detected: bool = Field(default=True, alias="showWhenNoPiiDetected")


class RoutingTarget(PolicyModel):
    """Map one exact route class or class prefix to an AgentGateway model."""

    name: str = Field(pattern=r"^[A-Za-z0-9_.:/-]+$")
    class_prefix: str | None = Field(
        default=None,
        alias="classPrefix",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.:/-]+$",
    )


class RoutingSettings(PolicyModel):
    """Configure trusted AgentGateway routing decisions."""

    default_target: str = Field(alias="defaultTarget", pattern=r"^[A-Za-z0-9_.:/-]+$")
    targets: list[RoutingTarget] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def validate_targets(self) -> RoutingSettings:
        """Reject ambiguous selectors and unmapped configured defaults."""
        selectors = [target.class_prefix or target.name for target in self.targets]
        if len(selectors) != len(set(selectors)):
            raise ValueError("routing selector is duplicated")
        if self.targets and self.default_target not in {target.name for target in self.targets}:
            raise ValueError("routing defaultTarget is not present in targets")
        return self


class PolicySettings(PolicyModel):
    """Represent the complete engine-owned policy core."""

    pii: PiiSettings
    attachments: AttachmentsSettings = Field(default_factory=AttachmentsSettings)
    safety: SafetySettings
    classifier: ClassifierSettings
    session: SessionSettings
    notice: NoticeSettings
    routing: RoutingSettings
    debug: bool = False
    log_format: Literal["text", "json"] = Field(default="text", alias="logFormat")


class PiiOverride(PolicyModel):
    """Allow Studio to preview policy behavior without changing model selection."""

    analyzer_languages: (
        list[Annotated[str, Field(min_length=2, max_length=8, pattern=r"^[a-z]{2,3}$")]] | None
    ) = Field(default=None, alias="analyzerLanguages", min_length=1, max_length=16)
    score_threshold: float | None = Field(default=None, alias="scoreThreshold", ge=0, le=1)
    timeout: float | None = Field(default=None, gt=0, le=600)
    default_action: str | None = Field(default=None, alias="defaultAction")
    default_operator: OperatorParams | None = Field(default=None, alias="defaultOperator")
    analyzer_entities: list[Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")]] | None = (
        Field(default=None, alias="analyzerEntities", max_length=64)
    )
    mask_on_reroute: bool | None = Field(default=None, alias="maskOnReroute")
    hash_window_hours: int | None = Field(default=None, alias="hashWindowHours", ge=1, le=720)
    entity_policies: list[EntityPolicy] | None = Field(
        default=None, alias="entityPolicies", max_length=64
    )


class SafetyOverride(PolicyModel):
    """Allow a request-local selection of built-in and custom safety rules."""

    enabled: list[Annotated[str, Field(min_length=1, max_length=128)]] | None = Field(
        default=None, max_length=16
    )
    custom: list[SafetyRuleEntry] | None = Field(default=None, max_length=16)


class ClassifierOverride(PolicyModel):
    """Allow request-local transformed-content classification settings."""

    default_class: str | None = Field(default=None, alias="defaultClass")
    classes: list[ClassifierClass] | None = Field(default=None, max_length=64)


class PolicyOverride(PolicyModel):
    """Represent the bounded policy fields Studio may override for one request."""

    pii: PiiOverride | None = None
    attachments: AttachmentsSettings | None = None
    safety: SafetyOverride | None = None
    classifier: ClassifierOverride | None = None
    session: SessionSettings | None = None
    notice: NoticeSettings | None = None
    routing: RoutingSettings | None = None
    debug: bool | None = None
    log_format: Literal["text", "json"] | None = Field(default=None, alias="logFormat")


def apply_policy_override(base: PolicySettings, override: PolicyOverride) -> PolicySettings:
    """Deep-merge and fully revalidate a Studio request-local policy preview."""
    merged = base.model_dump(mode="json", by_alias=True)
    updates = override.model_dump(mode="json", by_alias=True, exclude_none=True)
    _deep_update(merged, updates)
    return PolicySettings.model_validate(merged)


def _deep_update(target: dict[str, object], updates: dict[str, object]) -> None:
    for key, value in updates.items():
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            _deep_update(current, value)
        else:
            target[key] = value


def load_policy(path: Path) -> PolicySettings:
    """Load and strictly validate a Helm-rendered policy file."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return PolicySettings.model_validate(raw)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise PolicyConfigError("policy configuration is invalid") from exc


def test_policy() -> PolicySettings:
    """Return the explicit dependency-free policy used by isolated unit tests."""
    return PolicySettings.model_validate(
        {
            "pii": {
                "analyzerLanguages": ["en"],
                "supportedLanguages": ["en"],
                "defaultAction": "pass",
                "entityPolicies": [
                    {"entityType": "EMAIL_ADDRESS", "action": "mask"},
                    {"entityType": "PHONE_NUMBER", "action": "mask"},
                    {"entityType": "IBAN", "action": "reroute", "routeClass": "local"},
                    {"entityType": "PASSWORD_OR_SECRET", "action": "block"},
                ],
            },
            "safety": {
                "enabled": [
                    "promptInjection",
                    "jailbreak",
                    "systemPromptExtraction",
                    "harmfulContent",
                    "encodingEvasion",
                    "selfHarm",
                ]
            },
            "classifier": {"defaultClass": "general", "classes": []},
            "session": {"enabled": False, "ttlHours": 24},
            "notice": {
                "rerouted": "Request rerouted because sensitive data was detected.",
                "masked": "Sensitive data was anonymized.",
                "showWhenNoPiiDetected": True,
            },
            "routing": {"defaultTarget": "local", "targets": []},
        }
    )

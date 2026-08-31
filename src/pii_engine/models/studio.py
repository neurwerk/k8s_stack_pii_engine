"""Studio-only request and response contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from pii_engine.config.policy import PolicyOverride
from pii_engine.models.contracts import (
    AnalysisResponseBase,
    JsonValue,
    PIIAction,
    PIIReport,
    StrictModel,
    SupportedRequest,
)

type EvaluationIssueStage = Literal["schema", "merge", "compile"]
type EvaluationPathPart = (
    Annotated[str, Field(min_length=1, max_length=128)] | Annotated[int, Field(ge=0, le=10_000_000)]
)
type DetectionSource = Literal["deterministic", "spacy", "transformer", "policy_regex"]


class StudioAnalyzeRequest(StrictModel):
    """Wrap a supported request with an optional request-local policy preview."""

    request: SupportedRequest
    policy: PolicyOverride | None = None


class StudioPolicyEvaluationRequest(StrictModel):
    """Accept a request sample and an unvalidated bounded policy candidate."""

    request: SupportedRequest
    policy: dict[str, JsonValue] | None = Field(default=None, max_length=16)
    simulation: Literal["deterministic_echo"] = "deterministic_echo"


class PolicyEvaluationIssue(StrictModel):
    """Describe one sanitized candidate failure without rejected values."""

    stage: EvaluationIssueStage
    path: list[EvaluationPathPart] = Field(max_length=16)
    code: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    message: str = Field(min_length=1, max_length=256)


class StudioPolicyEvaluationInvalidResponse(StrictModel):
    """Return bounded policy diagnostics instead of a request validation failure."""

    api_version: Literal["v1"] = "v1"
    valid: Literal[False] = False
    issues: list[PolicyEvaluationIssue] = Field(min_length=1, max_length=128)
    issues_truncated: bool


class LogicalDetection(StrictModel):
    """Describe one leaf-local logical detection without matched content."""

    path: list[EvaluationPathPart] = Field(max_length=64)
    start: int = Field(ge=0, le=4_000_000)
    end: int = Field(gt=0, le=4_000_000)
    entity_type: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    source: DetectionSource
    configured_action: PIIAction
    resolved_action: PIIAction

    @model_validator(mode="after")
    def validate_span(self) -> LogicalDetection:
        """Require a non-empty code-point span."""
        if self.end <= self.start:
            raise ValueError("detection end must follow start")
        return self


class EffectiveRegion(StrictModel):
    """Describe one non-overlapping region selected for policy execution."""

    path: list[EvaluationPathPart] = Field(max_length=64)
    start: int = Field(ge=0, le=4_000_000)
    end: int = Field(gt=0, le=4_000_000)
    entity_type: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")
    action: PIIAction
    source: DetectionSource
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    member_entity_types: list[str] = Field(min_length=1, max_length=64)
    overlap: bool

    @model_validator(mode="after")
    def validate_region(self) -> EffectiveRegion:
        """Require a non-empty span and deterministic unique member names."""
        if self.end <= self.start:
            raise ValueError("region end must follow start")
        if self.member_entity_types != sorted(set(self.member_entity_types)):
            raise ValueError("region members must be sorted and unique")
        if self.entity_type not in self.member_entity_types:
            raise ValueError("winning entity must be a region member")
        return self


class EvaluationDiagnostics(StrictModel):
    """Carry bounded logical and effective policy evidence."""

    logical_detections: list[LogicalDetection] = Field(max_length=2_048)
    effective_regions: list[EffectiveRegion] = Field(max_length=2_048)
    truncated: bool


class EvaluationSimulation(StrictModel):
    """Describe a local deterministic echo without model transport behavior."""

    type: Literal["deterministic_echo"] = "deterministic_echo"
    status: Literal["completed", "skipped"]
    reason: Literal["request_blocked"] | None = None
    model_called: Literal[False] = False
    model_response: str | None = Field(default=None, max_length=10_485_760)
    user_response: str | None = Field(default=None, max_length=10_485_760)
    restored_entity_counts: dict[str, int] = Field(default_factory=dict, max_length=64)

    @model_validator(mode="after")
    def validate_status(self) -> EvaluationSimulation:
        """Keep skipped and completed simulation fields unambiguous."""
        if self.status == "skipped":
            if self.reason != "request_blocked" or self.model_response or self.user_response:
                raise ValueError("skipped simulations require only a block reason")
        elif self.reason is not None or self.model_response is None or self.user_response is None:
            raise ValueError("completed simulations require both response texts")
        if any(count <= 0 or count > 10_000_000 for count in self.restored_entity_counts.values()):
            raise ValueError("restored entity counts are invalid")
        return self


class StudioPolicyEvaluationValidResponse(AnalysisResponseBase):
    """Return safe aggregate and detailed evidence for one valid candidate."""

    valid: Literal[True] = True
    issues: list[PolicyEvaluationIssue] = Field(default_factory=list, max_length=0)
    issues_truncated: Literal[False] = False
    report: PIIReport
    diagnostics: EvaluationDiagnostics
    simulation: EvaluationSimulation


type StudioPolicyEvaluationResponse = Annotated[
    StudioPolicyEvaluationValidResponse | StudioPolicyEvaluationInvalidResponse,
    Field(discriminator="valid"),
]

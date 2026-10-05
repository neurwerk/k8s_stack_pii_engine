"""Strict versioned request and analysis contracts for all supported callers."""

from __future__ import annotations

from typing import Annotated, Literal

from neurwerk_request_segments import models as provider
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_serializer, model_validator

# Reuse canonical provider definitions; only the legacy Responses tool envelope differs.
AttachmentPart = provider.EngineAttachmentPart
OpenAIChatRequest = provider.EngineChatRequest
ChatStreamOptions = provider.EngineChatStreamOptions
FunctionCall = provider.EngineFunction
McpParams = provider.EngineMcpParams
McpRequest = provider.EngineMcpRequest
ChatMessage = provider.EngineMessage
MessageContent = provider.EngineMessageContent
ResponseFunctionCall = provider.EngineResponseFunctionCall
ResponseFunctionOutput = provider.EngineResponseFunctionOutput
ResponseInput = provider.EngineResponseInput
ResponseInputItem = provider.EngineResponseInputItem
ResponseMessage = provider.EngineResponseMessage
ResponseTextConfig = provider.EngineResponseTextConfig
ResponseTextFormat = provider.EngineResponseTextFormat
ResponseFormatJsonObject = provider.EngineResponseTextFormatObject
ResponseFormatJsonSchema = provider.EngineResponseTextFormatSchema
ResponseFormatText = provider.EngineResponseTextFormatText
ResponseTextPart = provider.EngineResponseTextPart
TextPart = provider.EngineTextPart
ToolCall = provider.EngineToolCall
ToolDefinition = provider.EngineToolDefinition
ToolFunction = provider.EngineToolFunction
type JsonValue = provider.JsonValue
type McpJsonValue = provider.McpJsonValue
type McpRequestId = provider.McpRequestId


class OpenAIResponsesRequest(provider.EngineResponsesRequest):
    """Retain nested function tools accepted by the legacy v1 boundary."""

    tools: list[provider.EngineResponseToolDefinition | ToolDefinition] | None = Field(
        default_factory=list, max_length=128
    )


type SupportedRequest = OpenAIChatRequest | OpenAIResponsesRequest | McpRequest

type AnalysisErrorCode = Literal[
    "invalid_request",
    "request_too_large",
    "capacity_unavailable",
    "runtime_unavailable",
    "analysis_timeout",
    "internal_error",
]
type AnalysisErrorMessage = Literal[
    "The analysis request is invalid.",
    "The analysis request exceeds the configured size limit.",
    "Analysis capacity is temporarily unavailable.",
    "The analysis runtime is unavailable.",
    "Analysis timed out.",
    "Analysis failed.",
]


class StrictModel(BaseModel):
    """Reject undocumented fields at every policy boundary."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class LimitDetail(BaseModel):
    """Carry content-free measurements from the boundary that rejected work."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    component: Literal["pii_engine", "extproc", "request_segments"] = "pii_engine"
    stage: Literal[
        "admission",
        "json",
        "inspection",
        "engine_request",
        "engine_response",
        "provider_response",
        "output",
    ]
    reason: Literal[
        "bytes",
        "declared_bytes",
        "encoded_bytes",
        "decoded_bytes",
        "transformed_bytes",
        "depth",
        "tokens",
        "nodes",
        "text_characters",
        "segments",
        "text_leaves",
        "empty_chunks",
    ]
    measured: int = Field(ge=0)
    maximum: int = Field(ge=0)
    unit: Literal["bytes", "characters", "items", "levels"]
    exact: bool

    @model_validator(mode="after")
    def validate_measurement(self) -> LimitDetail:
        """Reject contradictory units or a measurement that did not exceed its limit."""
        expected_unit = {
            "bytes": "bytes",
            "declared_bytes": "bytes",
            "encoded_bytes": "bytes",
            "decoded_bytes": "bytes",
            "transformed_bytes": "bytes",
            "depth": "levels",
            "text_characters": "characters",
        }.get(self.reason, "items")
        if self.measured <= self.maximum or self.unit != expected_unit:
            raise ValueError("invalid limit measurement")
        return self


class AnalysisErrorDetail(StrictModel):
    """Describe one stable analysis failure without exception or request data."""

    code: AnalysisErrorCode
    message: AnalysisErrorMessage
    retryable: bool


class AnalysisErrorResponse(StrictModel):
    """Return the strict versioned failure envelope for analysis requests."""

    api_version: Literal["v1"] = "v1"
    error: AnalysisErrorDetail


class AnalysisLimitErrorDetail(AnalysisErrorDetail):
    """Extend a size rejection with an observed limit measurement."""

    code: Literal["request_too_large"] = "request_too_large"
    limit: LimitDetail


class AnalysisLimitErrorResponse(StrictModel):
    """Version only the extended error envelope; success contracts are separate."""

    api_version: Literal["v2"] = "v2"
    error: AnalysisLimitErrorDetail


SUPPORTED_REQUEST_ADAPTER = TypeAdapter(SupportedRequest)


def has_attachments(request: SupportedRequest) -> bool:
    """Inspect only schema-designated content blocks for raw attachments."""
    if isinstance(request, OpenAIChatRequest):
        return any(
            isinstance(part, AttachmentPart)
            for message in request.messages
            if isinstance(message.content, list)
            for part in message.content
        )
    if isinstance(request, OpenAIResponsesRequest) and isinstance(request.input, list):
        return any(
            isinstance(part, AttachmentPart)
            for item in request.input
            if isinstance(item, ResponseMessage)
            for part in item.content
        )
    return False


class FaceFindings(StrictModel):
    """Describe current-request aggregate face detections without pixels or identities."""

    scan_status: Literal["complete", "not_scanned", "failed"]
    count: Annotated[int, Field(strict=True, ge=0, le=10_000_000)] | None

    @model_validator(mode="after")
    def validate_scan(self) -> FaceFindings:
        """Require a count exactly when the required inspection completed."""
        if (self.scan_status == "complete") != (self.count is not None):
            raise ValueError("face count must exist exactly for a complete scan")
        return self


class VisualFindings(StrictModel):
    """Carry only the visual evidence the trusted document adapter can supply."""

    faces: FaceFindings


class DocumentAnalyzeRequest(StrictModel):
    """Bind trusted visual findings to one converted, text-only model request."""

    api_version: Literal["v1"]
    request: OpenAIChatRequest | OpenAIResponsesRequest
    text_pii_enabled: Annotated[bool, Field(strict=True)]
    visual_findings: VisualFindings

    @model_validator(mode="after")
    def validate_text_only(self) -> DocumentAnalyzeRequest:
        """Reject pixels and other raw attachments in a visual envelope."""
        if has_attachments(self.request):
            raise ValueError("document envelopes require a converted text-only request")
        return self


class TextSegment(StrictModel):
    """Carry one independently analyzed string under a caller-owned opaque ID."""

    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    text: str


class SegmentRequest(StrictModel):
    """Accept only extracted text and trusted analysis controls, never provider JSON."""

    api_version: Literal["v2"] = "v2"
    request_kind: Literal["chat", "responses", "mcp"]
    scope: Literal["session", "request"]
    segments: list[TextSegment]
    text_pii_enabled: Annotated[bool, Field(strict=True)] = True
    attachments_present: Annotated[bool, Field(strict=True)] = False
    visual_findings: VisualFindings | None = None

    @model_validator(mode="after")
    def validate_segments(self) -> SegmentRequest:
        """Keep IDs unique and visual controls restricted to converted model requests."""
        ids = [segment.id for segment in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("segment IDs must be unique")
        if self.visual_findings is not None and (
            self.request_kind == "mcp" or self.scope != "request" or self.attachments_present
        ):
            raise ValueError("visual findings require converted request-scoped model segments")
        if not self.text_pii_enabled and self.visual_findings is None:
            raise ValueError("disabling text analysis requires visual findings")
        return self


class AnalysisMetadata(StrictModel):
    """Describe bounded analysis facts without prompt values."""

    source: Literal["current_request", "cached_decision"]
    scan_performed: bool
    duration_ms: int | None = Field(ge=0, le=600_000)
    overlap_count: int = Field(ge=0, le=10_000_000)
    overlap_resolution: Literal["strictest_action"]
    policy_version: str = Field(min_length=1, max_length=64)
    text_leaf_count: int = Field(ge=0, le=256)
    cached_decision_applied: bool

    @model_validator(mode="after")
    def validate_provenance(self) -> AnalysisMetadata:
        """Require scan timing and cache provenance to agree."""
        if self.scan_performed != (self.duration_ms is not None):
            raise ValueError("scan duration must exist exactly when a scan was performed")
        if self.scan_performed and self.source != "current_request":
            raise ValueError("performed scans must describe the current request")
        if self.source == "cached_decision" and not self.cached_decision_applied:
            raise ValueError("cached analysis metadata must apply a cached decision")
        if not self.scan_performed and self.source == "current_request" and self.overlap_count:
            raise ValueError("unscanned current requests cannot report overlaps")
        return self


class Notices(StrictModel):
    """Carry policy-owned messages without operational prose."""

    request: list[Annotated[str, Field(max_length=4_000)]] = Field(max_length=16)
    response: list[Annotated[str, Field(max_length=4_000)]] = Field(max_length=16)


type Decision = Literal["pass", "block", "apply_actions", "reroute"]
type PIIAction = Literal[
    "pass",
    "block",
    "reroute",
    "mask",
    "replace",
    "redact",
    "hash",
    "encrypt",
    "reversible_replace",
]


class PIIReportRow(StrictModel):
    """Summarize one entity action without retaining detected values."""

    entity_type: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")
    action: PIIAction | Literal["text-only"]
    detected_count: int = Field(ge=1, le=10_000_000)
    transformed_count: int = Field(ge=0, le=10_000_000)
    unique_transformed_count: int = Field(ge=0, le=10_000_000)

    @model_validator(mode="after")
    def validate_counts(self) -> PIIReportRow:
        """Require transformation totals to describe possible executions."""
        if self.entity_type == "FACE":
            if self.action not in {"block", "text-only", "reroute"} or self.transformed_count:
                raise ValueError("FACE rows require a visual action without transformations")
        elif self.action == "text-only":
            raise ValueError("text-only is reserved for FACE rows")
        if self.transformed_count > self.detected_count:
            raise ValueError("transformed_count cannot exceed detected_count")
        if self.unique_transformed_count > self.transformed_count:
            raise ValueError("unique_transformed_count cannot exceed transformed_count")
        if self.action in {"pass", "block"} and self.transformed_count:
            raise ValueError("pass and block rows cannot claim transformations")
        return self


class PIIReport(StrictModel):
    """Return bounded aggregate PII details safe for adapter transport."""

    rows: list[PIIReportRow] = Field(max_length=64)

    @model_validator(mode="after")
    def validate_rows(self) -> PIIReport:
        """Require one row per entity in deterministic normalized order."""
        entity_types = [row.entity_type for row in self.rows]
        if len(entity_types) != len(set(entity_types)):
            raise ValueError("report rows must contain unique entity types")
        if entity_types != sorted(entity_types):
            raise ValueError("report rows must be sorted by entity_type")
        return self


class AnalysisResponseBase(StrictModel):
    """Common safe analysis fields shared by adapter and Studio."""

    api_version: Literal["v1"]
    decision: Decision
    entities: list[str] = Field(default_factory=list, max_length=64)
    entity_counts: dict[str, int] = Field(default_factory=dict, max_length=64)
    applied_actions: list[str] = Field(default_factory=list, max_length=16)
    remote_allowed: bool
    route_class: str | None = Field(default=None, max_length=128)
    request: SupportedRequest | None = None
    analysis: AnalysisMetadata
    notices: Notices
    safety_rule: str | None = Field(default=None, max_length=128)

    @field_serializer("request")
    def serialize_request(self, request: SupportedRequest | None) -> dict[str, object] | None:
        """Keep omitted provider fields out of the legacy wire response."""
        if request is None:
            return None
        return request.model_dump(mode="json", by_alias=True, exclude_unset=True)

    @model_validator(mode="after")
    def validate_unscanned_success(self) -> AnalysisResponseBase:
        """Validate the caller-specific exception to current-request text scanning."""
        self._validate_unscanned_success()
        return self

    def _validate_unscanned_success(self) -> None:
        """Allow unscanned current success only for an MCP call without string arguments."""
        unscanned_current_success = (
            self.analysis.source == "current_request"
            and not self.analysis.scan_performed
            and self.decision != "block"
        )
        if unscanned_current_success:
            if not _is_no_text_mcp_request(self.request):
                raise ValueError("unscanned current success requires a no-text MCP request")
            if (
                self.decision != "pass"
                or not self.remote_allowed
                or self.entities
                or self.entity_counts
                or self.applied_actions
                or self.route_class is not None
                or self.analysis.text_leaf_count
                or self.analysis.cached_decision_applied
                or self.notices.request
                or self.notices.response
                or self.safety_rule is not None
            ):
                raise ValueError("no-text MCP success must be an unchanged unscanned pass")
        if isinstance(self.request, McpRequest) and (
            self.decision == "reroute"
            or self.route_class is not None
            or self.notices.request
            or self.notices.response
        ):
            raise ValueError("MCP analysis cannot expose model routing or notices")


class AdapterAnalyzeResponse(AnalysisResponseBase):
    """Return trusted request-scoped reversal entries to the adapter only."""

    report: PIIReport
    visual_findings: VisualFindings | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    reversal: dict[
        Annotated[
            str,
            Field(
                min_length=3,
                max_length=256,
                pattern=r"^<(?:REV|ENCRYPTED)_[A-Z][A-Z0-9_]*_[0-9a-f]{16}_[0-9a-f]{16}>$",
            ),
        ],
        Annotated[str, Field(min_length=1, max_length=4_000_000)],
    ] = Field(default_factory=dict)

    def _validate_unscanned_success(self) -> None:
        """Allow visual-only success without weakening the legacy scan contract."""
        if self.visual_findings is None:
            super()._validate_unscanned_success()
            if "FACE" in self.entity_counts:
                raise ValueError("FACE requires current visual findings")
            return
        if self.analysis.source != "current_request" or self.analysis.cached_decision_applied:
            raise ValueError("visual findings cannot use cached decisions")
        if not self.analysis.scan_performed:
            if set(self.entity_counts) - {"FACE"} or self.reversal:
                raise ValueError(
                    "unscanned visual results cannot claim text detections or reversal"
                )
            expected_actions = (
                ["block"]
                if self.decision == "block"
                else sorted({row.action for row in self.report.rows})
            )
            if self.applied_actions != expected_actions:
                raise ValueError("unscanned visual results can apply only their visual action")
        self._validate_visual_decision(self.visual_findings.faces)

    def _validate_visual_decision(self, faces: FaceFindings) -> None:
        """Keep visual evidence, effective actions, and forwarding state consistent."""
        if self.entity_counts.get("FACE", 0) != (faces.count or 0):
            raise ValueError("FACE counts must match visual findings")
        if faces.scan_status == "failed" and self.decision != "block":
            raise ValueError("failed face inspection requires a block")
        face_row = next((row for row in self.report.rows if row.entity_type == "FACE"), None)
        if face_row is not None and (
            (self.decision == "block" and face_row.action != "block")
            or face_row.action not in self.applied_actions
        ):
            raise ValueError("FACE must report its effective action")
        if face_row is None and "text-only" in self.applied_actions:
            raise ValueError("text-only requires a FACE row")
        if self.decision == "block":
            if self.request is not None or self.route_class is not None or self.reversal:
                raise ValueError("visual blocks cannot carry forwarding or reversal state")
        elif not isinstance(self.request, (OpenAIChatRequest, OpenAIResponsesRequest)) or (
            has_attachments(self.request)
        ):
            raise ValueError("visual success requires a text-only model request")
        if self.remote_allowed != (self.decision not in {"block", "reroute"}):
            raise ValueError("visual decision and remote permission disagree")
        if self.decision == "reroute" and not self.route_class:
            raise ValueError("visual reroute requires a route")

    @model_validator(mode="after")
    def validate_report(self) -> AdapterAnalyzeResponse:
        """Require report aggregates to agree with the adapter decision."""
        self._validate_report_counts()
        self._validate_decision_rows()
        if (
            self.analysis.source == "current_request"
            and not self.analysis.scan_performed
            and self.decision == "pass"
            and (self.report.rows or self.reversal)
        ):
            raise ValueError("unscanned MCP passes cannot contain report or reversal material")
        return self

    def _validate_report_counts(self) -> None:
        """Require report counts to match their current or cached provenance."""
        if set(self.entity_counts) != set(self.entities) or any(
            count <= 0 or count > 10_000_000 for count in self.entity_counts.values()
        ):
            raise ValueError("adapter entity counts are inconsistent")
        report_counts = {row.entity_type: row.detected_count for row in self.report.rows}
        if self.analysis.cached_decision_applied:
            if self.decision not in {"block", "reroute"}:
                raise ValueError("cached reports require a cached terminal decision")
            if any(
                entity_type not in self.entity_counts or count > self.entity_counts[entity_type]
                for entity_type, count in report_counts.items()
            ):
                raise ValueError("cached report rows exceed adapter entity counts")
        elif report_counts != self.entity_counts:
            raise ValueError("current report rows must match adapter entity counts")

    def _validate_decision_rows(self) -> None:
        """Reject report rows that contradict the effective decision."""
        actions = {row.action for row in self.report.rows}
        if self.decision == "pass":
            if actions - {"pass"}:
                raise ValueError("pass decisions require pass report rows")
        elif self.decision == "apply_actions":
            if actions & {"block", "reroute"} or not any(
                row.transformed_count or (row.entity_type == "FACE" and row.action == "text-only")
                for row in self.report.rows
            ):
                raise ValueError("action decisions require transformed non-terminal report rows")
        elif self.decision == "reroute":
            if "block" in actions:
                raise ValueError("reroute decisions cannot contain block report rows")
            current_cached_reroute = (
                self.analysis.source == "current_request"
                and self.analysis.scan_performed
                and self.analysis.cached_decision_applied
            )
            if "reroute" not in actions and not current_cached_reroute:
                raise ValueError("reroute decisions require a reroute report row")
        else:
            if any(row.transformed_count for row in self.report.rows):
                raise ValueError("block decisions cannot claim transformations")
            if self.report.rows and "block" not in actions:
                raise ValueError("PII block decisions require a block report row")


class StudioAnalyzeResponse(AnalysisResponseBase):
    """Return policy results without any reversal material."""


class SegmentAnalysisResponse(StrictModel):
    """Return policy facts and transformed segments without provider protocol data."""

    api_version: Literal["v2"] = "v2"
    decision: Decision
    entities: list[str] = Field(default_factory=list, max_length=64)
    entity_counts: dict[str, int] = Field(default_factory=dict, max_length=64)
    applied_actions: list[str] = Field(default_factory=list, max_length=16)
    remote_allowed: bool
    route_class: str | None = Field(default=None, max_length=128)
    segments: list[TextSegment] | None
    analysis: AnalysisMetadata
    notices: Notices
    safety_rule: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def validate_forwarding(self) -> SegmentAnalysisResponse:
        """Require a complete segment result exactly when forwarding is allowed."""
        if (self.decision == "block") != (self.segments is None):
            raise ValueError("blocked results must not contain segments")
        if self.remote_allowed != (self.decision not in {"block", "reroute"}):
            raise ValueError("decision and remote permission disagree")
        if self.decision == "block" and self.route_class is not None:
            raise ValueError("blocked results cannot select a route")
        if self.decision == "reroute" and not self.route_class:
            raise ValueError("reroute requires a route")
        if self.segments is not None:
            ids = [segment.id for segment in self.segments]
            if len(ids) != len(set(ids)) or len(ids) > 256:
                raise ValueError("invalid result segment IDs")
        return self


class AdapterSegmentAnalyzeResponse(SegmentAnalysisResponse):
    """Carry bounded reports and request-local reversal only to the trusted adapter."""

    report: PIIReport
    visual_findings: VisualFindings | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    reversal: dict[
        Annotated[
            str,
            Field(
                min_length=3,
                max_length=256,
                pattern=r"^<(?:REV|ENCRYPTED)_[A-Z][A-Z0-9_]*_[0-9a-f]{16}_[0-9a-f]{16}>$",
            ),
        ],
        Annotated[str, Field(min_length=1, max_length=4_000_000)],
    ] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_report(self) -> AdapterSegmentAnalyzeResponse:
        """Retain report, visual, and reversal provenance without provider models."""
        self._validate_counts()
        self._validate_actions()
        self._validate_visual()
        return self

    def _validate_counts(self) -> None:
        """Check aggregate counts against their current or cached provenance."""
        if set(self.entity_counts) != set(self.entities) or any(
            count <= 0 or count > 10_000_000 for count in self.entity_counts.values()
        ):
            raise ValueError("adapter entity counts are inconsistent")
        counts = {row.entity_type: row.detected_count for row in self.report.rows}
        if self.analysis.cached_decision_applied:
            if self.decision not in {"block", "reroute"} or any(
                entity not in self.entity_counts or count > self.entity_counts[entity]
                for entity, count in counts.items()
            ):
                raise ValueError("cached reports require consistent terminal decisions")
        elif counts != self.entity_counts:
            raise ValueError("current report rows must match adapter entity counts")

    def _validate_actions(self) -> None:
        """Require the report and reversal state to agree with the effective decision."""
        actions = {row.action for row in self.report.rows}
        if self.decision == "pass" and actions - {"pass"}:
            raise ValueError("pass decisions require pass report rows")
        if self.decision == "apply_actions" and (
            actions & {"block", "reroute"}
            or not any(
                row.transformed_count or row.action == "text-only" for row in self.report.rows
            )
        ):
            raise ValueError("action decisions require transformed non-terminal report rows")
        if self.decision == "reroute" and (
            "block" in actions
            or (
                "reroute" not in actions
                and not (
                    self.analysis.source == "current_request"
                    and self.analysis.scan_performed
                    and self.analysis.cached_decision_applied
                )
            )
        ):
            raise ValueError("reroute decisions require reroute evidence")
        if self.decision == "block" and (
            self.reversal
            or any(row.transformed_count for row in self.report.rows)
            or (self.report.rows and "block" not in actions)
        ):
            raise ValueError("blocked results cannot contain transformations")
        if not self.analysis.scan_performed and self.reversal:
            raise ValueError("reversal requires a current text scan")

    def _validate_visual(self) -> None:
        """Validate current visual evidence without inferring text detections."""
        if self.visual_findings is None:
            if "FACE" in self.entity_counts:
                raise ValueError("FACE requires current visual findings")
        else:
            faces = self.visual_findings.faces
            if self.analysis.source != "current_request" or self.analysis.cached_decision_applied:
                raise ValueError("visual findings cannot use cached decisions")
            if self.entity_counts.get("FACE", 0) != (faces.count or 0):
                raise ValueError("FACE counts must match visual findings")
            if faces.scan_status == "failed" and self.decision != "block":
                raise ValueError("failed face inspection requires a block")
            if not self.analysis.scan_performed and set(self.entity_counts) - {"FACE"}:
                raise ValueError("unscanned visual results cannot claim text detections")
            face_row = next((row for row in self.report.rows if row.entity_type == "FACE"), None)
            if face_row is not None and (
                face_row.action not in self.applied_actions
                or (self.decision == "block" and face_row.action != "block")
            ):
                raise ValueError("FACE must report its effective action")
        if "text-only" in self.applied_actions and not any(
            row.entity_type == "FACE" for row in self.report.rows
        ):
            raise ValueError("text-only requires a FACE row")


class ActionParam(StrictModel):
    """Describe one Studio-visible action parameter."""

    name: str
    type: str
    default: str
    description: str
    options: list[str] = Field(default_factory=list)


class ActionDescription(StrictModel):
    """Describe one action from the shared registry."""

    name: str
    decision: str
    reversible: bool
    severity: Literal["pass", "info", "warn", "fail"]
    strictness: int = Field(ge=1, le=9)
    params: list[ActionParam] = Field(default_factory=list)
    notes: str


class PolicyResponse(StrictModel):
    """Expose safe policy metadata and normalized entity names."""

    api_version: Literal["v1"]
    version: str
    default_action: str
    entities: list[str]
    safety_rules: list[str]


def _is_no_text_mcp_request(request: SupportedRequest | None) -> bool:
    """Return whether a response carries an MCP request with no string arguments."""
    return isinstance(request, McpRequest) and not _contains_string(request.params.arguments)


def _contains_string(value: McpJsonValue | None) -> bool:
    if isinstance(value, str):
        return True
    if isinstance(value, list):
        return any(_contains_string(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_string(item) for item in value.values())
    return False

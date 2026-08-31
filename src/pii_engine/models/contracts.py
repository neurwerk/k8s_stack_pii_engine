"""Strict versioned request and analysis contracts for all supported callers."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from pydantic.json_schema import JsonDict, SkipJsonSchema

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type McpJsonValue = (
    Annotated[str, Field(strict=True)]
    | Annotated[int, Field(strict=True)]
    | Annotated[float, Field(strict=True, allow_inf_nan=False)]
    | Annotated[bool, Field(strict=True)]
    | Annotated[list[McpJsonValue], Field(max_length=256)]
    | Annotated[dict[str, McpJsonValue], Field(max_length=256)]
    | None
)
type McpRequestId = (
    Annotated[str, Field(strict=True, min_length=1, max_length=256)]
    | Annotated[
        int,
        Field(strict=True, ge=-9_007_199_254_740_991, le=9_007_199_254_740_991),
    ]
)
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


def _omit_none_default(schema: JsonDict) -> None:
    """Keep omitted optional MCP objects from advertising null as their default."""
    schema.pop("default", None)


class StrictModel(BaseModel):
    """Reject undocumented fields at every policy boundary."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class AnalysisErrorDetail(StrictModel):
    """Describe one stable analysis failure without exception or request data."""

    code: AnalysisErrorCode
    message: AnalysisErrorMessage
    retryable: bool


class AnalysisErrorResponse(StrictModel):
    """Return the strict versioned failure envelope for analysis requests."""

    api_version: Literal["v1"] = "v1"
    error: AnalysisErrorDetail


class TextPart(StrictModel):
    """Represent one OpenAI chat text part."""

    type: Literal["text"]
    text: str = Field(min_length=1)


class AttachmentPart(BaseModel):
    """Accept known attachment blocks only so the block-only policy can reject them."""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=False)

    type: Literal[
        "image_url",
        "input_audio",
        "file",
        "input_image",
        "input_file",
        "image",
        "audio",
        "resource",
        "resource_link",
    ]


type MessageContent = str | Annotated[list[TextPart | AttachmentPart], Field(max_length=64)]


class FunctionCall(StrictModel):
    """Represent a function call and its supported JSON arguments."""

    name: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:-]+$")
    arguments: JsonValue = ""


class ToolCall(StrictModel):
    """Represent an assistant function call."""

    id: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:-]+$")
    type: Literal["function"]
    function: FunctionCall


class ToolFunction(StrictModel):
    """Describe one callable tool."""

    name: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:-]+$")
    description: str | None = Field(default=None, max_length=4_000)
    parameters: dict[str, JsonValue] | None = None


class ToolDefinition(StrictModel):
    """Describe one supported function tool."""

    type: Literal["function"]
    function: ToolFunction


class ChatMessage(StrictModel):
    """Represent system, user, assistant, or tool-result content."""

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: MessageContent | None = None
    name: str | None = Field(default=None, max_length=256)
    tool_calls: list[ToolCall] = Field(default_factory=list, max_length=32)
    tool_call_id: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def validate_role_fields(self) -> ChatMessage:
        """Require the fields that distinguish assistant calls and tool results."""
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("tool_calls are only supported on assistant messages")
        if self.role == "assistant" and self.content is None and not self.tool_calls:
            raise ValueError("assistant messages require content or tool_calls")
        return self


class OpenAIChatRequest(StrictModel):
    """Bound an OpenAI Chat Completions request."""

    model: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_./:-]+$")
    messages: list[ChatMessage] = Field(min_length=1, max_length=256)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1, le=1_000_000)
    stream: bool = False
    n: int | None = Field(default=None, ge=1, le=16)
    stop: str | list[str] | None = None
    tools: list[ToolDefinition] = Field(default_factory=list, max_length=128)
    tool_choice: Literal["none", "auto", "required"] | dict[str, JsonValue] | None = None
    response_format: dict[str, JsonValue] | None = None
    user: str | None = Field(default=None, max_length=256)


class ResponseTextPart(StrictModel):
    """Represent text accepted by the OpenAI Responses API."""

    type: Literal["input_text", "output_text"]
    text: str = Field(min_length=1)


class ResponseMessage(StrictModel):
    """Represent one Responses-style message item."""

    type: Literal["message"] = "message"
    role: Literal["system", "developer", "user", "assistant"]
    content: list[ResponseTextPart | AttachmentPart] = Field(min_length=1, max_length=64)


class ResponseFunctionCall(StrictModel):
    """Represent a Responses-style function call."""

    type: Literal["function_call"]
    call_id: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:-]+$")
    arguments: JsonValue


class ResponseFunctionOutput(StrictModel):
    """Represent nested textual tool output returned to a model."""

    type: Literal["function_call_output"]
    call_id: str = Field(min_length=1, max_length=256)
    output: JsonValue


type ResponseInputItem = ResponseMessage | ResponseFunctionCall | ResponseFunctionOutput
type ResponseInput = str | Annotated[list[ResponseInputItem], Field(min_length=1, max_length=256)]


class ResponseFormatText(StrictModel):
    """Select ordinary text output from the Responses API."""

    type: Literal["text"]


class ResponseFormatJsonObject(StrictModel):
    """Select the legacy JSON object output mode."""

    type: Literal["json_object"]


class ResponseFormatJsonSchema(StrictModel):
    """Configure Responses API structured output with a JSON Schema."""

    type: Literal["json_schema"]
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    description: str | None = Field(default=None, max_length=4_000)
    schema_: dict[str, JsonValue] = Field(alias="schema", max_length=256)
    strict: bool | None = None


type ResponseTextFormat = Annotated[
    ResponseFormatText | ResponseFormatJsonObject | ResponseFormatJsonSchema,
    Field(discriminator="type"),
]


class ResponseTextConfig(StrictModel):
    """Control plain or structured text generated by the Responses API."""

    format: ResponseTextFormat | None = None
    verbosity: Literal["low", "medium", "high"] | None = None


class OpenAIResponsesRequest(StrictModel):
    """Bound the supported OpenAI Responses request shape."""

    model: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_./:-]+$")
    input: ResponseInput
    instructions: str | None = None
    tools: list[ToolDefinition] = Field(default_factory=list, max_length=128)
    tool_choice: Literal["none", "auto", "required"] | dict[str, JsonValue] | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_output_tokens: int | None = Field(default=None, ge=1, le=1_000_000)
    stream: bool = False
    previous_response_id: str | None = Field(default=None, max_length=256)
    text: ResponseTextConfig | None = None
    user: str | None = Field(default=None, max_length=256)


class McpParams(StrictModel):
    """Represent bounded MCP tool-call input and immutable protocol metadata."""

    name: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    arguments: Annotated[dict[str, McpJsonValue], Field(max_length=256)] | SkipJsonSchema[None] = (
        Field(
            default=None,
            exclude_if=lambda value: value is None,
            json_schema_extra=_omit_none_default,
        )
    )
    meta: (
        Annotated[
            dict[Annotated[str, Field(max_length=256)], McpJsonValue],
            Field(max_length=64),
        ]
        | SkipJsonSchema[None]
    ) = Field(
        default=None,
        alias="_meta",
        exclude_if=lambda value: value is None,
        json_schema_extra=_omit_none_default,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_optional_objects(cls, value: object) -> object:
        """Reject explicit null where MCP permits only an omitted or object field."""
        if isinstance(value, dict) and any(
            key in value and value[key] is None for key in ("arguments", "_meta")
        ):
            raise ValueError("optional MCP params must be objects when present")
        return value


class McpRequest(StrictModel):
    """Bound one direct MCP tools/call request containing tool arguments."""

    jsonrpc: Literal["2.0"]
    id: McpRequestId
    method: Literal["tools/call"]
    params: McpParams


type SupportedRequest = OpenAIChatRequest | OpenAIResponsesRequest | McpRequest
SUPPORTED_REQUEST_ADAPTER = TypeAdapter(SupportedRequest)


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
    action: PIIAction
    detected_count: int = Field(ge=1, le=10_000_000)
    transformed_count: int = Field(ge=0, le=10_000_000)
    unique_transformed_count: int = Field(ge=0, le=10_000_000)

    @model_validator(mode="after")
    def validate_counts(self) -> PIIReportRow:
        """Require transformation totals to describe possible executions."""
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

    @model_validator(mode="after")
    def validate_unscanned_success(self) -> AnalysisResponseBase:
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
        return self


class AdapterAnalyzeResponse(AnalysisResponseBase):
    """Return trusted request-scoped reversal entries to the adapter only."""

    report: PIIReport
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
                row.transformed_count for row in self.report.rows
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

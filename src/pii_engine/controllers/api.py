"""Versioned workload-authorized policy routes."""

from __future__ import annotations

import logging
import re
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Response
from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticSerializationError

from pii_engine.config.policy import PolicyOverride
from pii_engine.config.settings import Settings, get_settings
from pii_engine.lib.actions import action_descriptions
from pii_engine.lib.catalog import ENTITY_CATALOG
from pii_engine.lib.identity import Caller, adapter_identity, studio_identity
from pii_engine.lib.safety import SAFETY_BY_NAME
from pii_engine.models.contracts import (
    ActionDescription,
    AdapterAnalyzeResponse,
    AdapterSegmentAnalyzeResponse,
    AnalysisErrorCode,
    AnalysisErrorDetail,
    AnalysisErrorMessage,
    AnalysisErrorResponse,
    AnalysisLimitErrorResponse,
    AnalysisMetadata,
    DocumentAnalyzeRequest,
    LimitDetail,
    Notices,
    OpenAIChatRequest,
    OpenAIResponsesRequest,
    PIIReport,
    PolicyResponse,
    SegmentAnalysisResponse,
    SegmentRequest,
    StudioAnalyzeResponse,
    SupportedRequest,
)
from pii_engine.models.studio import (
    EffectiveRegion,
    EvaluationDiagnostics,
    EvaluationSimulation,
    LogicalDetection,
    SegmentEffectiveRegion,
    SegmentEvaluationDiagnostics,
    SegmentLogicalDetection,
    StudioAnalyzeRequest,
    StudioPolicyEvaluationInvalidResponse,
    StudioPolicyEvaluationRequest,
    StudioPolicyEvaluationResponse,
    StudioPolicyEvaluationValidResponse,
    StudioSegmentAnalyzeRequest,
    StudioSegmentPolicyEvaluationInvalidResponse,
    StudioSegmentPolicyEvaluationRequest,
    StudioSegmentPolicyEvaluationResponse,
    StudioSegmentPolicyEvaluationValidResponse,
)
from pii_engine.runtime import RuntimeNotReadyError, get_runtime
from pii_engine.services.errors import AnalysisRequestTooLargeError, InvalidAnalysisRequestError
from pii_engine.services.limiter import AnalysisCapacityError
from pii_engine.services.policy import PolicyResult

router = APIRouter()
logger = logging.getLogger(__name__)

_ERROR_SPECS: dict[AnalysisErrorCode, tuple[int, AnalysisErrorMessage, bool]] = {
    "invalid_request": (400, "The analysis request is invalid.", False),
    "request_too_large": (
        413,
        "The analysis request exceeds the configured size limit.",
        False,
    ),
    "capacity_unavailable": (503, "Analysis capacity is temporarily unavailable.", True),
    "runtime_unavailable": (503, "The analysis runtime is unavailable.", True),
    "analysis_timeout": (504, "Analysis timed out.", True),
    "internal_error": (500, "Analysis failed.", False),
}
_ANALYSIS_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status_code: {"model": AnalysisErrorResponse} for status_code in (400, 500, 503, 504)
}
_ANALYSIS_ERROR_RESPONSES[413] = {"model": AnalysisErrorResponse | AnalysisLimitErrorResponse}
_SIMULATION_PREFIX = "[SIMULATED - NO MODEL CALLED]"
_REVERSIBLE_PLACEHOLDER = re.compile(
    r"<(?P<prefix>REV|ENCRYPTED)_(?P<entity>[A-Z][A-Z0-9_]*)_"
    r"[0-9a-f]{16}_[0-9a-f]{16}>"
)


class AnalysisAPIError(Exception):
    """Carry a safe typed failure from an analysis route to its app handler."""

    def __init__(self, code: AnalysisErrorCode, *, limit: LimitDetail | None = None) -> None:
        """Build the fixed status and response body for one stable reason code."""
        self.status_code, message, retryable = _ERROR_SPECS[code]
        self.limit = limit
        self.response = AnalysisErrorResponse(
            error=AnalysisErrorDetail(code=code, message=message, retryable=retryable)
        )
        super().__init__(code)


def analysis_api_error(
    code: AnalysisErrorCode, *, limit: LimitDetail | None = None
) -> AnalysisAPIError:
    """Return a typed API failure with its fixed status, message, and retryability."""
    return AnalysisAPIError(code, limit=limit)


def log_analysis_failure(
    caller: str,
    code: AnalysisErrorCode,
    exc: BaseException,
    *,
    debug_details: bool = True,
) -> None:
    """Log only bounded failure metadata, never exception text or request content."""
    exception_class = "".join(
        character if character.isascii() and (character.isalnum() or character == "_") else "_"
        for character in type(exc).__name__
    )[:64]
    logger.error(
        "analysis failed caller=%s reason=%s exception=%s",
        caller,
        code,
        exception_class or "Exception",
    )


def _failure_code(exc: Exception) -> AnalysisErrorCode:
    """Map runtime and domain failures to the public stable error taxonomy."""
    if isinstance(exc, RuntimeNotReadyError):
        return "runtime_unavailable"
    if isinstance(exc, AnalysisCapacityError):
        return "capacity_unavailable"
    if isinstance(exc, TimeoutError):
        return "analysis_timeout"
    if isinstance(exc, AnalysisRequestTooLargeError):
        return "request_too_large"
    if isinstance(exc, InvalidAnalysisRequestError):
        return "invalid_request"
    return "internal_error"


def _measured_api_error(code: AnalysisErrorCode, exc: Exception) -> AnalysisAPIError:
    """Preserve only measurements supplied by the rejecting boundary."""
    return analysis_api_error(
        code, limit=exc.limit if isinstance(exc, AnalysisRequestTooLargeError) else None
    )


@router.get("/v1/adapter/ready", include_in_schema=False)
async def adapter_ready(_caller: Caller = Depends(adapter_identity)) -> dict[str, str]:
    """Verify runtime readiness and the exact adapter mTLS identity."""
    try:
        runtime = get_runtime()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="policy runtime is not ready") from exc
    if not await runtime.ready():
        raise HTTPException(status_code=503, detail="policy runtime is not ready")
    return {"status": "ok"}


@router.get("/v2/adapter/ready", include_in_schema=False)
async def adapter_segments_ready(caller: Caller = Depends(adapter_identity)) -> dict[str, str]:
    """Verify segment API capability using the adapter identity and actual runtime readiness."""
    await adapter_ready(caller)
    return {"api_version": "v2", "status": "ok"}


async def _analyze(
    request: SupportedRequest,
    caller: Caller,
    session_key: str | None = None,
    policy_override: PolicyOverride | None = None,
) -> PolicyResult:
    """Run shared policy work and expose only a stable fail-closed error."""
    try:
        return await get_runtime().analyze(caller, request, session_key, policy_override)
    except Exception as exc:  # noqa: BLE001 - the API boundary must fail closed.
        code = _failure_code(exc)
        log_analysis_failure(caller, code, exc)
        raise _measured_api_error(code, exc) from None


def _analysis(result: PolicyResult, settings: Settings) -> AnalysisMetadata:
    """Build bounded analysis metadata without any request values."""
    return AnalysisMetadata(
        source=result.analysis_source,
        scan_performed=result.scan_performed,
        duration_ms=result.duration_ms,
        overlap_count=result.overlap_count,
        overlap_resolution="strictest_action",
        policy_version=settings.policy_version,
        text_leaf_count=result.text_leaf_count,
        cached_decision_applied=result.cached_decision_applied,
    )


def _notices(result: PolicyResult) -> Notices:
    """Build policy-owned request and response messages."""
    return Notices(request=result.request_notices, response=result.response_notices)


@router.post(
    "/v1/adapter/analyze-request",
    response_model=AdapterAnalyzeResponse,
    responses=_ANALYSIS_ERROR_RESPONSES,
)
async def analyze_adapter(
    request: Annotated[SupportedRequest, Body()],
    caller: Caller = Depends(adapter_identity),
    session_key: str | None = Header(default=None, alias="x-pii-session-key"),
) -> Response:
    """Analyze for extproc and return request-scoped reversal material."""
    result = await _analyze(request, caller, session_key)
    return _adapter_response(result, caller)


@router.post(
    "/v1/adapter/analyze-document-request",
    response_model=AdapterAnalyzeResponse,
    responses=_ANALYSIS_ERROR_RESPONSES,
)
async def analyze_document_adapter(
    request: OpenAIChatRequest | OpenAIResponsesRequest | DocumentAnalyzeRequest,
    caller: Caller = Depends(adapter_identity),
) -> Response:
    """Analyze the whole extracted-text request once, without session reuse or persistence."""
    # PII split across separate lines or table cells may be missed; improve this later.
    try:
        result = await get_runtime().analyze(caller, request, request_scoped=True)
        return _adapter_response(
            result,
            caller,
            debug_details=False,
            document=request if isinstance(request, DocumentAnalyzeRequest) else None,
        )
    except AnalysisAPIError:
        raise
    except Exception as exc:  # noqa: BLE001 - document failures must never log content.
        code = _failure_code(exc)
        log_analysis_failure(caller, code, exc, debug_details=False)
        raise _measured_api_error(code, exc) from None


def _adapter_response(
    result: PolicyResult,
    caller: Caller,
    *,
    debug_details: bool = True,
    document: DocumentAnalyzeRequest | None = None,
) -> Response:
    """Validate and bound the complete adapter response before returning any content."""
    settings = get_runtime().settings
    if document is not None and (
        (result.scan_performed and not document.text_pii_enabled)
        or (not result.scan_performed and document.text_pii_enabled and result.decision != "block")
    ):
        raise analysis_api_error("internal_error")
    try:
        response = AdapterAnalyzeResponse(
            api_version="v1",
            decision=result.decision,
            remote_allowed=result.remote_allowed,
            entities=result.entities,
            entity_counts=result.entity_counts,
            applied_actions=result.applied_actions,
            route_class=result.route_class,
            request=result.request,
            analysis=_analysis(result, settings),
            notices=_notices(result),
            safety_rule=result.safety_rule,
            report=PIIReport(rows=result.report_rows),
            visual_findings=document.visual_findings if document is not None else None,
            reversal=result.reversal,
        )
        content = response.model_dump_json(by_alias=True).encode("utf-8")
    except (ValidationError, PydanticSerializationError) as exc:
        code: AnalysisErrorCode = "internal_error"
        log_analysis_failure(caller, code, exc, debug_details=debug_details)
        raise analysis_api_error(code) from None
    if len(content) > settings.max_adapter_response_bytes:
        limit = LimitDetail(
            stage="engine_response",
            reason="bytes",
            measured=len(content),
            maximum=settings.max_adapter_response_bytes,
            unit="bytes",
            exact=True,
        )
        exc = AnalysisRequestTooLargeError("adapter response body too large", limit=limit)
        code = "request_too_large"
        log_analysis_failure(caller, code, exc, debug_details=False)
        raise analysis_api_error(code, limit=limit) from None
    return Response(content=content, media_type="application/json")


@router.post(
    "/v1/studio/analyze-request",
    response_model=StudioAnalyzeResponse,
    responses=_ANALYSIS_ERROR_RESPONSES,
)
async def analyze_studio(
    body: StudioAnalyzeRequest, caller: Caller = Depends(studio_identity)
) -> StudioAnalyzeResponse:
    """Analyze for Studio using the same core without reversal plaintext."""
    result = await _analyze(body.request, caller, policy_override=body.policy)
    return StudioAnalyzeResponse(
        api_version="v1",
        decision=result.decision,
        remote_allowed=result.remote_allowed,
        entities=result.entities,
        entity_counts=result.entity_counts,
        applied_actions=result.applied_actions,
        route_class=result.route_class,
        request=result.request,
        analysis=_analysis(result, get_runtime().settings),
        notices=_notices(result),
        safety_rule=result.safety_rule,
    )


@router.post(
    "/v1/studio/evaluate-policy",
    response_model=StudioPolicyEvaluationResponse,
    responses=_ANALYSIS_ERROR_RESPONSES,
)
async def evaluate_studio_policy(
    body: StudioPolicyEvaluationRequest,
    caller: Caller = Depends(studio_identity),
) -> Response:
    """Evaluate a raw request-local candidate and run a model-free simulation."""
    try:
        evaluation = await get_runtime().evaluate_policy(caller, body.request, body.policy)
    except Exception as exc:  # noqa: BLE001 - the API boundary must fail closed.
        code = _failure_code(exc)
        log_analysis_failure(caller, code, exc)
        raise _measured_api_error(code, exc) from None
    if evaluation.result is None:
        response: StudioPolicyEvaluationValidResponse | StudioPolicyEvaluationInvalidResponse = (
            StudioPolicyEvaluationInvalidResponse(
                issues=evaluation.issues or [],
                issues_truncated=evaluation.issues_truncated,
            )
        )
    else:
        result = evaluation.result
        response = StudioPolicyEvaluationValidResponse(
            api_version="v1",
            decision=result.decision,
            remote_allowed=result.remote_allowed,
            entities=result.entities,
            entity_counts=result.entity_counts,
            applied_actions=result.applied_actions,
            route_class=result.route_class,
            request=result.request,
            analysis=_analysis(result, get_runtime().settings),
            notices=_notices(result),
            safety_rule=result.safety_rule,
            report=PIIReport(rows=result.report_rows),
            diagnostics=_evaluation_diagnostics(result),
            simulation=_simulation(result),
        )
    try:
        content = response.model_dump_json(by_alias=True).encode("utf-8")
    except (ValidationError, PydanticSerializationError) as exc:
        code: AnalysisErrorCode = "internal_error"
        log_analysis_failure(caller, code, exc)
        raise analysis_api_error(code) from None
    if len(content) > get_runtime().settings.max_studio_evaluation_response_bytes:
        limit = LimitDetail(
            stage="output",
            reason="bytes",
            measured=len(content),
            maximum=get_runtime().settings.max_studio_evaluation_response_bytes,
            unit="bytes",
            exact=True,
        )
        exc = AnalysisRequestTooLargeError("Studio evaluation response body too large", limit=limit)
        code = "request_too_large"
        log_analysis_failure(caller, code, exc, debug_details=False)
        raise analysis_api_error(code, limit=limit) from None
    return Response(content=content, media_type="application/json")


def _evaluation_diagnostics(result: PolicyResult) -> EvaluationDiagnostics:
    """Convert bounded domain diagnostics to the strict Studio contract."""
    return EvaluationDiagnostics(
        logical_detections=[
            LogicalDetection(
                path=list(item.path),
                start=item.start,
                end=item.end,
                entity_type=item.entity_type,
                score=item.score,
                source=item.source,
                configured_action=item.configured_action,
                resolved_action=item.resolved_action,
            )
            for item in result.logical_detections
        ],
        effective_regions=[
            EffectiveRegion(
                path=list(item.path),
                start=item.start,
                end=item.end,
                entity_type=item.entity_type,
                action=item.action,
                source=item.source,
                score=item.score,
                member_entity_types=list(item.member_entity_types),
                overlap=item.overlap,
            )
            for item in result.effective_regions
        ],
        truncated=result.diagnostics_truncated,
    )


def _simulation(result: PolicyResult) -> EvaluationSimulation:
    """Echo model-visible transformed text and reverse only authoritative placeholders."""
    if result.decision == "block":
        return EvaluationSimulation(status="skipped", reason="request_blocked")
    if result.segments is None:
        raise ValueError("non-blocking evaluation is missing its transformed segments")
    text = "\n".join(segment.text for segment in result.segments)
    model_response = f"{_SIMULATION_PREFIX}\n{text}"
    restored_counts: dict[str, int] = {}

    def restore(match: re.Match[str]) -> str:
        placeholder = match.group(0)
        plaintext = result.reversal.get(placeholder)
        if plaintext is None:
            return placeholder
        entity_type = match.group("entity")
        restored_counts[entity_type] = restored_counts.get(entity_type, 0) + 1
        return plaintext

    user_response = _REVERSIBLE_PLACEHOLDER.sub(restore, model_response)
    return EvaluationSimulation(
        status="completed",
        model_response=model_response,
        user_response=user_response,
        restored_entity_counts=dict(sorted(restored_counts.items())),
    )


def _segment_fields(result: PolicyResult, request: SegmentRequest) -> dict[str, Any]:
    """Validate request correspondence before serializing any successful segment result."""
    if result.decision != "block":
        if result.segments is None or [item.id for item in result.segments] != [
            item.id for item in request.segments
        ]:
            raise ValueError("analysis changed segment identity or order")
        if (
            not result.scan_performed
            and result.analysis_source == "current_request"
            and request.visual_findings is None
            and not (
                request.request_kind == "mcp"
                and not request.segments
                and result.decision == "pass"
                and not result.entities
                and not result.applied_actions
                and result.route_class is None
                and not result.request_notices
                and not result.response_notices
                and not result.safety_rule
                and not result.reversal
                and not result.cached_decision_applied
                and not result.text_leaf_count
            )
        ):
            raise ValueError("unscanned success requires visual controls or no-text MCP")
    if request.visual_findings is not None:
        if result.scan_performed and not request.text_pii_enabled:
            raise ValueError("visual-only request performed text analysis")
        if request.text_pii_enabled and not result.scan_performed and result.decision != "block":
            raise ValueError("visual request skipped required text analysis")
        if (
            not request.text_pii_enabled
            and result.segments is not None
            and (result.segments != request.segments)
        ):
            raise ValueError("visual-only request changed text")
    if request.request_kind == "mcp" and (
        result.decision == "reroute"
        or result.route_class is not None
        or result.request_notices
        or result.response_notices
    ):
        raise ValueError("MCP cannot carry model routing or notices")
    return {
        "api_version": "v2",
        "decision": result.decision,
        "remote_allowed": result.remote_allowed,
        "entities": result.entities,
        "entity_counts": result.entity_counts,
        "applied_actions": result.applied_actions,
        "route_class": result.route_class,
        "segments": result.segments,
        "analysis": _analysis(result, get_runtime().settings),
        "notices": _notices(result),
        "safety_rule": result.safety_rule,
    }


def _bounded_segment_response(response: BaseModel, limit: int) -> Response:
    content = response.model_dump_json(by_alias=True).encode("utf-8")
    if len(content) > limit:
        raise AnalysisRequestTooLargeError(
            "segment response body too large",
            limit=LimitDetail(
                stage="engine_response",
                reason="bytes",
                measured=len(content),
                maximum=limit,
                unit="bytes",
                exact=True,
            ),
        )
    return Response(content=content, media_type="application/json")


@router.post(
    "/v2/adapter/analyze-segments",
    response_model=AdapterSegmentAnalyzeResponse,
    responses=_ANALYSIS_ERROR_RESPONSES,
)
async def analyze_adapter_segments(
    request: SegmentRequest,
    caller: Caller = Depends(adapter_identity),
    session_key: str | None = Header(default=None, alias="x-pii-session-key"),
) -> Response:
    """Analyze extracted text with trusted scope and request-local reversal."""
    try:
        result = await get_runtime().analyze_segments(caller, request, session_key)
        response = AdapterSegmentAnalyzeResponse(
            **_segment_fields(result, request),
            report=PIIReport(rows=result.report_rows),
            visual_findings=request.visual_findings,
            reversal=result.reversal,
        )
        return _bounded_segment_response(
            response, get_runtime().settings.max_adapter_response_bytes
        )
    except Exception as exc:  # noqa: BLE001 - safe fail-closed transport boundary.
        code = _failure_code(exc)
        log_analysis_failure(caller, code, exc, debug_details=False)
        raise _measured_api_error(code, exc) from None


@router.post(
    "/v2/studio/analyze-segments",
    response_model=SegmentAnalysisResponse,
    responses=_ANALYSIS_ERROR_RESPONSES,
)
async def analyze_studio_segments(
    body: StudioSegmentAnalyzeRequest,
    caller: Caller = Depends(studio_identity),
) -> Response:
    """Apply a request-local Studio preview without exposing reversal mappings."""
    try:
        result = await get_runtime().analyze_segments(
            caller, body.request, policy_override=body.policy
        )
        response = SegmentAnalysisResponse(**_segment_fields(result, body.request))
        return _bounded_segment_response(
            response,
            get_runtime().settings.max_studio_evaluation_response_bytes,
        )
    except Exception as exc:  # noqa: BLE001 - safe fail-closed transport boundary.
        code = _failure_code(exc)
        log_analysis_failure(caller, code, exc, debug_details=False)
        raise _measured_api_error(code, exc) from None


@router.post(
    "/v2/studio/evaluate-policy",
    response_model=StudioSegmentPolicyEvaluationResponse,
    responses=_ANALYSIS_ERROR_RESPONSES,
)
async def evaluate_studio_segments(
    body: StudioSegmentPolicyEvaluationRequest,
    caller: Caller = Depends(studio_identity),
) -> Response:
    """Evaluate candidate policy against original segment-local offsets."""
    try:
        evaluation = await get_runtime().evaluate_policy(caller, body.request, body.policy)
        response: (
            StudioSegmentPolicyEvaluationInvalidResponse
            | StudioSegmentPolicyEvaluationValidResponse
        )
        if evaluation.result is None:
            response = StudioSegmentPolicyEvaluationInvalidResponse(
                issues=evaluation.issues or [],
                issues_truncated=evaluation.issues_truncated,
            )
        else:
            result = evaluation.result
            response = StudioSegmentPolicyEvaluationValidResponse(
                **_segment_fields(result, body.request),
                report=PIIReport(rows=result.report_rows),
                diagnostics=SegmentEvaluationDiagnostics(
                    logical_detections=[
                        SegmentLogicalDetection(
                            segment_id=item.segment_id,
                            start=item.start,
                            end=item.end,
                            entity_type=item.entity_type,
                            score=item.score,
                            source=item.source,
                            configured_action=item.configured_action,
                            resolved_action=item.resolved_action,
                        )
                        for item in result.logical_detections
                    ],
                    effective_regions=[
                        SegmentEffectiveRegion(
                            segment_id=item.segment_id,
                            start=item.start,
                            end=item.end,
                            entity_type=item.entity_type,
                            action=item.action,
                            source=item.source,
                            score=item.score,
                            member_entity_types=list(item.member_entity_types),
                            overlap=item.overlap,
                        )
                        for item in result.effective_regions
                    ],
                    truncated=result.diagnostics_truncated,
                ),
                simulation=_simulation(result),
            )
        return _bounded_segment_response(
            response,
            get_runtime().settings.max_studio_evaluation_response_bytes,
        )
    except Exception as exc:  # noqa: BLE001 - safe fail-closed transport boundary.
        code = _failure_code(exc)
        log_analysis_failure(caller, code, exc, debug_details=False)
        raise _measured_api_error(code, exc) from None


@router.get("/v1/actions", response_model=list[ActionDescription])
def actions(_caller: Caller = Depends(studio_identity)) -> list[ActionDescription]:
    """Return the shared action registry to authenticated Studio only."""
    return action_descriptions()


@router.get("/v1/policy", response_model=PolicyResponse)
def policy(
    _caller: Caller = Depends(studio_identity), settings: Settings = Depends(get_settings)
) -> PolicyResponse:
    """Return safe engine-owned policy metadata to Studio only."""
    policy_settings = get_runtime().policy_settings
    return PolicyResponse(
        api_version="v1",
        version=settings.policy_version,
        default_action=policy_settings.pii.default_action,
        entities=list(ENTITY_CATALOG),
        safety_rules=list(SAFETY_BY_NAME),
    )

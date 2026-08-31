"""Versioned workload-authorized policy routes."""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from pydantic import ValidationError
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
    AnalysisErrorCode,
    AnalysisErrorDetail,
    AnalysisErrorMessage,
    AnalysisErrorResponse,
    AnalysisMetadata,
    Notices,
    PIIReport,
    PolicyResponse,
    StudioAnalyzeResponse,
    SupportedRequest,
)
from pii_engine.models.studio import (
    EffectiveRegion,
    EvaluationDiagnostics,
    EvaluationSimulation,
    LogicalDetection,
    StudioAnalyzeRequest,
    StudioPolicyEvaluationInvalidResponse,
    StudioPolicyEvaluationRequest,
    StudioPolicyEvaluationResponse,
    StudioPolicyEvaluationValidResponse,
)
from pii_engine.runtime import RuntimeNotReadyError, get_runtime
from pii_engine.services.errors import AnalysisRequestTooLargeError, InvalidAnalysisRequestError
from pii_engine.services.limiter import AnalysisCapacityError
from pii_engine.services.policy import PolicyResult
from pii_engine.services.traversal import iter_text_leaves

router = APIRouter(prefix="/v1")
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
    status_code: {"model": AnalysisErrorResponse} for status_code in (400, 413, 500, 503, 504)
}
_SIMULATION_PREFIX = "[SIMULATED - NO MODEL CALLED]"
_REVERSIBLE_PLACEHOLDER = re.compile(
    r"<(?P<prefix>REV|ENCRYPTED)_(?P<entity>[A-Z][A-Z0-9_]*)_"
    r"[0-9a-f]{16}_[0-9a-f]{16}>"
)


class AnalysisAPIError(Exception):
    """Carry a safe typed failure from an analysis route to its app handler."""

    def __init__(self, code: AnalysisErrorCode) -> None:
        """Build the fixed status and response body for one stable reason code."""
        self.status_code, message, retryable = _ERROR_SPECS[code]
        self.response = AnalysisErrorResponse(
            error=AnalysisErrorDetail(code=code, message=message, retryable=retryable)
        )
        super().__init__(code)


def analysis_api_error(code: AnalysisErrorCode) -> AnalysisAPIError:
    """Return a typed API failure with its fixed status, message, and retryability."""
    return AnalysisAPIError(code)


def log_analysis_failure(
    caller: str,
    code: AnalysisErrorCode,
    exc: BaseException,
    *,
    debug_details: bool = True,
) -> None:
    """Log bounded failure metadata and optional DEBUG-only exception details."""
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
    if debug_details:
        logger.debug(
            "analysis failure details caller=%s reason=%s",
            caller,
            code,
            exc_info=(type(exc), exc, exc.__traceback__),
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


@router.get("/adapter/ready", include_in_schema=False)
async def adapter_ready(_caller: Caller = Depends(adapter_identity)) -> dict[str, str]:
    """Verify runtime readiness and the exact adapter mTLS identity."""
    try:
        runtime = get_runtime()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="policy runtime is not ready") from exc
    if not await runtime.ready():
        raise HTTPException(status_code=503, detail="policy runtime is not ready")
    return {"status": "ok"}


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
        raise analysis_api_error(code) from None


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
    "/adapter/analyze-request",
    response_model=AdapterAnalyzeResponse,
    responses=_ANALYSIS_ERROR_RESPONSES,
)
async def analyze_adapter(
    request: SupportedRequest,
    caller: Caller = Depends(adapter_identity),
    session_key: str | None = Header(default=None, alias="x-pii-session-key"),
) -> Response:
    """Analyze for extproc and return request-scoped reversal material."""
    result = await _analyze(request, caller, session_key)
    settings = get_runtime().settings
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
            reversal=result.reversal,
        )
        content = response.model_dump_json(by_alias=True).encode("utf-8")
    except (ValidationError, PydanticSerializationError) as exc:
        code: AnalysisErrorCode = "internal_error"
        log_analysis_failure(caller, code, exc)
        raise analysis_api_error(code) from None
    if len(content) > settings.max_adapter_response_bytes:
        exc = AnalysisRequestTooLargeError("adapter response body too large")
        code = "request_too_large"
        log_analysis_failure(caller, code, exc, debug_details=False)
        raise analysis_api_error(code) from None
    return Response(content=content, media_type="application/json")


@router.post(
    "/studio/analyze-request",
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
    "/studio/evaluate-policy",
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
        raise analysis_api_error(code) from None
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
        exc = AnalysisRequestTooLargeError("Studio evaluation response body too large")
        code = "request_too_large"
        log_analysis_failure(caller, code, exc, debug_details=False)
        raise analysis_api_error(code) from None
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
    if result.request is None:
        raise ValueError("non-blocking evaluation is missing its transformed request")
    text = "\n".join(
        leaf.text
        for leaf in iter_text_leaves(result.request, get_runtime().settings.max_nesting_depth)
    )
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


@router.get("/actions", response_model=list[ActionDescription])
def actions(_caller: Caller = Depends(studio_identity)) -> list[ActionDescription]:
    """Return the shared action registry to authenticated Studio only."""
    return action_descriptions()


@router.get("/policy", response_model=PolicyResponse)
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

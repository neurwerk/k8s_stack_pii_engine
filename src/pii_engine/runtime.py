"""Validated runtime state shared by policy and management applications."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
from dataclasses import dataclass

from pydantic import ValidationError

from pii_engine.config.policy import (
    PolicyOverride,
    PolicySettings,
    apply_policy_override,
    load_policy,
    test_policy,
)
from pii_engine.config.settings import Settings
from pii_engine.lib.bundle import desired_cache, desired_reference_matches
from pii_engine.metrics import (
    analysis_duration_seconds,
    overlap_regions_total,
    policy_failures_total,
    policy_requests_total,
    queue_rejections_total,
    runtime_analyzer_mode,
    runtime_device,
)
from pii_engine.models.contracts import JsonValue, McpRequest, OpenAIChatRequest, SupportedRequest
from pii_engine.models.studio import EvaluationIssueStage, PolicyEvaluationIssue
from pii_engine.services.analyzer import (
    configure_inference_device,
    create_analyzer,
    resolve_analyzer_mode,
    validate_policy_selection,
)
from pii_engine.services.anonymizer import PresidioAnonymizer, TestAnonymizer
from pii_engine.services.errors import InvalidAnalysisRequestError
from pii_engine.services.limiter import AnalysisCapacityError, AnalysisLimiter
from pii_engine.services.planner import ActionPlanner
from pii_engine.services.policy import PolicyResult, PolicyService
from pii_engine.services.session import RequestKind, SessionDecision, SessionStore
from pii_engine.services.traversal import validate_request_structure

logger = logging.getLogger(__name__)
_MAX_EVALUATION_ISSUES = 128
_POLICY_PATH_COMPONENTS = {
    "pii",
    "attachments",
    "safety",
    "classifier",
    "session",
    "notice",
    "routing",
    "debug",
    "logFormat",
    "analyzerLanguages",
    "scoreThreshold",
    "timeout",
    "defaultAction",
    "defaultOperator",
    "analyzerEntities",
    "maskOnReroute",
    "hashWindowHours",
    "entityPolicies",
    "enabled",
    "custom",
    "defaultClass",
    "classes",
    "policy",
    "type",
    "masking_char",
    "chars_to_mask",
    "from_end",
    "new_value",
    "entityType",
    "action",
    "patterns",
    "routeClass",
    "hash_type",
    "name",
    "pattern",
    "message",
    "ttlHours",
    "rerouted",
    "masked",
    "showWhenNoPiiDetected",
    "defaultTarget",
    "targets",
    "classPrefix",
}


@dataclass
class PolicyEvaluationResult:
    """Return either a completed analysis or sanitized candidate issues."""

    result: PolicyResult | None = None
    issues: list[PolicyEvaluationIssue] | None = None
    issues_truncated: bool = False


class RuntimeNotReadyError(RuntimeError):
    """Raise when the complete policy runtime cannot safely accept work."""


class EngineRuntime:
    """Own initialized models, policy, keys, session state, and shared capacity."""

    def __init__(self, settings: Settings) -> None:
        """Eagerly validate all static dependencies and load configured models."""
        self.settings = settings
        self.policy_settings = self._load_policy()
        try:
            self.analyzer_mode = resolve_analyzer_mode(settings)
        except ValueError as exc:
            raise RuntimeNotReadyError("analyzer selection failed") from exc
        self._static_ready = True
        self.device = (
            "test-cpu"
            if settings.allow_test_analyzer
            else configure_inference_device(settings.device)
        )
        runtime_device.labels(device=self.device).set(1)
        for mode in ("baseline", "transformer", "test"):
            runtime_analyzer_mode.labels(mode=mode).set(mode == self.analyzer_mode)
        analyzer = create_analyzer(settings, self.policy_settings, self.analyzer_mode)
        encryption_key = (
            settings.encryption_key.get_secret_value()
            if settings.encryption_key is not None
            else "t" * 32
        )
        anonymizer = (
            TestAnonymizer() if settings.allow_test_analyzer else PresidioAnonymizer(encryption_key)
        )
        hash_key = (
            settings.hash_key.get_secret_value().encode()
            if settings.hash_key is not None
            else b"test-only-hash-key-that-is-never-used-in-production"
        )
        self._analyzer = analyzer
        self._anonymizer = anonymizer
        self._hash_key = hash_key
        self.policy = self._policy_service(self.policy_settings)
        self.limiter = AnalysisLimiter(
            settings.max_concurrent_analyses,
            settings.max_queued_analyses,
            settings.queue_wait_timeout,
        )
        self._studio_limiter = asyncio.Semaphore(settings.studio_max_concurrent_analyses)
        self._background_analyses: set[asyncio.Task[PolicyResult]] = set()
        self._background_evaluations: set[asyncio.Task[PolicyEvaluationResult]] = set()
        self.session = self._create_session_store()

    async def start(self) -> None:
        """Connect required dynamic dependencies before reporting readiness."""
        if self.session is not None:
            await self.session.start()

    async def close(self) -> None:
        """Release dynamic dependencies."""
        if self.session is not None:
            await self.session.close()

    async def ready(self) -> bool:
        """Return current model, policy, key, and Valkey readiness."""
        if not self._static_ready or not self._cache_still_available():
            return False
        return self.session is None or await self.session.healthy()

    def restart_required(self) -> bool:
        """Return whether a baseline process should restart into transformer mode."""
        if self.analyzer_mode != "baseline":
            return False
        reference = self.settings.model_bundle_reference
        cache = self.settings.model_cache_path
        digest = self.settings.model_manifest_sha256
        version = self.settings.model_bundle_version
        bundle = self.settings.model_bundle_path
        if (
            reference is None
            or cache is None
            or digest is None
            or version is None
            or bundle is None
        ):
            return False
        return desired_cache(cache, reference, digest, version) == bundle

    async def analyze(
        self,
        caller: str,
        request: SupportedRequest,
        session_key: str | None = None,
        policy_override: PolicyOverride | None = None,
    ) -> PolicyResult:
        """Apply session state and one shared bounded CPU analysis queue."""
        validate_request_structure(request, self.settings.max_nesting_depth)
        if not await self.ready():
            raise RuntimeNotReadyError("policy runtime is not ready")
        if policy_override is not None and caller != "studio":
            raise ValueError("policy overrides are accepted only from Studio")
        active_policy = (
            apply_policy_override(self.policy_settings, policy_override)
            if policy_override is not None
            else self.policy_settings
        )
        policy = self._policy_service(active_policy) if policy_override is not None else self.policy
        request_kind = _request_kind(request)
        validated_session_key = _validated_session_key(session_key)
        placeholder_namespace = (
            _conversation_placeholder_namespace(
                validated_session_key,
                self._hash_key,
                self.settings.policy_version,
            )
            if caller == "adapter"
            and isinstance(request, OpenAIChatRequest)
            and validated_session_key is not None
            else None
        )
        cached = await self._cached_decision(caller, session_key, request_kind)
        cached_result = self._final_cached_result(request, cached, active_policy)
        if cached_result is not None:
            self._validate_result_kind(cached_result, request_kind)
            self._record_result(caller, cached_result)
            return cached_result
        analysis_started = asyncio.Event()
        task = asyncio.create_task(
            self._run_analysis(
                caller,
                policy,
                request,
                analysis_started,
                placeholder_namespace=placeholder_namespace,
            )
        )
        started_waiter = asyncio.create_task(analysis_started.wait())
        try:
            done, _pending = await asyncio.wait(
                {task, started_waiter}, return_when=asyncio.FIRST_COMPLETED
            )
            if task in done:
                result = task.result()
            else:
                result = await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=self._analysis_timeout(caller, active_policy),
                )
        except AnalysisCapacityError as exc:
            reason = "full" if "full" in str(exc) else "timeout"
            queue_rejections_total.labels(caller=caller, reason=reason).inc()
            policy_failures_total.labels(caller=caller, error="queue").inc()
            raise
        except TimeoutError:
            self._retain_background_analysis(task)
            policy_failures_total.labels(caller=caller, error="timeout").inc()
            raise
        except asyncio.CancelledError:
            if analysis_started.is_set():
                self._retain_background_analysis(task)
            else:
                task.cancel()
                self._retain_background_analysis(task)
            raise
        except Exception:
            policy_failures_total.labels(caller=caller, error="analysis").inc()
            raise
        finally:
            started_waiter.cancel()
        self._apply_cached_decision(result, cached)
        self._validate_result_kind(result, request_kind)
        await self._taint_session(caller, session_key, result, cached, request_kind)
        self._record_result(caller, result)
        return result

    async def evaluate_policy(
        self,
        caller: str,
        request: SupportedRequest,
        raw_policy: dict[str, JsonValue] | None,
    ) -> PolicyEvaluationResult:
        """Evaluate one raw Studio candidate without sessions or live mutation."""
        validate_request_structure(request, self.settings.max_nesting_depth)
        if caller != "studio":
            raise ValueError("policy evaluation is accepted only from Studio")
        if not await self.ready():
            raise RuntimeNotReadyError("policy runtime is not ready")
        analysis_started = asyncio.Event()
        task = asyncio.create_task(
            self._run_policy_evaluation(request, raw_policy, analysis_started)
        )
        started_waiter = asyncio.create_task(analysis_started.wait())
        try:
            done, _pending = await asyncio.wait(
                {task, started_waiter}, return_when=asyncio.FIRST_COMPLETED
            )
            evaluation = (
                task.result()
                if task in done
                else await asyncio.wait_for(
                    asyncio.shield(task), timeout=self.settings.studio_analysis_timeout
                )
            )
        except AnalysisCapacityError as exc:
            reason = "full" if "full" in str(exc) else "timeout"
            queue_rejections_total.labels(caller=caller, reason=reason).inc()
            policy_failures_total.labels(caller=caller, error="queue").inc()
            raise
        except TimeoutError:
            self._retain_background_evaluation(task)
            policy_failures_total.labels(caller=caller, error="timeout").inc()
            raise
        except asyncio.CancelledError:
            if not analysis_started.is_set():
                task.cancel()
            self._retain_background_evaluation(task)
            raise
        except Exception:
            policy_failures_total.labels(caller=caller, error="analysis").inc()
            raise
        finally:
            started_waiter.cancel()
        if evaluation.result is not None:
            self._record_result(caller, evaluation.result)
        return evaluation

    async def _run_policy_evaluation(
        self,
        request: SupportedRequest,
        raw_policy: dict[str, JsonValue] | None,
        started: asyncio.Event,
    ) -> PolicyEvaluationResult:
        """Run candidate validation, compilation, and analysis under Studio limits."""
        studio_acquired = False
        try:
            try:
                await asyncio.wait_for(
                    self._studio_limiter.acquire(), self.settings.queue_wait_timeout
                )
                studio_acquired = True
            except TimeoutError as exc:
                raise AnalysisCapacityError("Studio analysis wait timed out") from exc
            async with self.limiter.slot("studio"):
                started.set()
                analysis_started = time.monotonic()
                try:
                    evaluation = await asyncio.to_thread(
                        self._evaluate_policy_sync, request, raw_policy
                    )
                finally:
                    elapsed = max(0.0, time.monotonic() - analysis_started)
                    analysis_duration_seconds.labels(caller="studio").observe(elapsed)
                if evaluation.result is not None and evaluation.result.scan_performed:
                    evaluation.result.duration_ms = min(600_000, round(elapsed * 1_000))
                return evaluation
        finally:
            if studio_acquired:
                self._studio_limiter.release()

    def _evaluate_policy_sync(
        self, request: SupportedRequest, raw_policy: dict[str, JsonValue] | None
    ) -> PolicyEvaluationResult:
        """Validate and compile candidate policy before using the live analyzer and planner."""
        policy = self.policy
        if raw_policy is not None:
            try:
                override = PolicyOverride.model_validate(raw_policy)
            except ValidationError as exc:
                return _invalid_evaluation("schema", exc)
            try:
                active_policy = apply_policy_override(self.policy_settings, override)
            except ValidationError as exc:
                return _invalid_evaluation("merge", exc)
            except (TypeError, ValueError):
                return _single_issue("merge", "policy_merge_failed", "Policy merge failed.")
            try:
                validate_policy_selection(active_policy)
                policy = self._policy_service(active_policy)
            except Exception:  # noqa: BLE001 - candidate compilation is a safe result.
                return _single_issue(
                    "compile", "policy_compile_failed", "Policy compilation failed."
                )
        return PolicyEvaluationResult(
            result=policy.analyze(request, include_diagnostics=True), issues=[]
        )

    async def _run_analysis(
        self,
        caller: str,
        policy: PolicyService,
        request: SupportedRequest,
        started: asyncio.Event,
        *,
        placeholder_namespace: str | None = None,
    ) -> PolicyResult:
        """Run CPU work while retaining every acquired limit until the thread exits."""
        studio_acquired = False
        if caller == "studio":
            try:
                await asyncio.wait_for(
                    self._studio_limiter.acquire(), self.settings.queue_wait_timeout
                )
                studio_acquired = True
            except TimeoutError as exc:
                raise AnalysisCapacityError("Studio analysis wait timed out") from exc
        try:
            async with self.limiter.slot(caller):
                started.set()
                analysis_started = time.monotonic()
                try:
                    if placeholder_namespace is None:
                        result = await asyncio.to_thread(policy.analyze, request)
                    else:
                        result = await asyncio.to_thread(
                            policy.analyze,
                            request,
                            placeholder_namespace=placeholder_namespace,
                        )
                finally:
                    elapsed = max(0.0, time.monotonic() - analysis_started)
                    analysis_duration_seconds.labels(caller=caller).observe(elapsed)
                if result.scan_performed:
                    result.duration_ms = min(600_000, round(elapsed * 1_000))
                return result
        finally:
            if studio_acquired:
                self._studio_limiter.release()

    def _caller_timeout(self, caller: str) -> float:
        """Return the process timeout for one authenticated caller class."""
        if caller == "studio":
            return self.settings.studio_analysis_timeout
        return self.settings.analysis_timeout

    def _analysis_timeout(self, caller: str, policy: PolicySettings) -> float:
        """Apply the active policy ceiling to the caller-specific process timeout."""
        return min(self._caller_timeout(caller), policy.pii.timeout)

    def _retain_background_analysis(self, task: asyncio.Task[PolicyResult]) -> None:
        """Keep timed-out CPU work alive and consume its eventual exception safely."""
        if task in self._background_analyses:
            return
        self._background_analyses.add(task)
        task.add_done_callback(self._background_analysis_done)

    def _background_analysis_done(self, task: asyncio.Task[PolicyResult]) -> None:
        self._background_analyses.discard(task)
        if not task.cancelled():
            task.exception()

    def _retain_background_evaluation(self, task: asyncio.Task[PolicyEvaluationResult]) -> None:
        """Retain timed-out Studio evaluation work until its thread exits."""
        if task in self._background_evaluations:
            return
        self._background_evaluations.add(task)
        task.add_done_callback(self._background_evaluation_done)

    def _background_evaluation_done(self, task: asyncio.Task[PolicyEvaluationResult]) -> None:
        self._background_evaluations.discard(task)
        if not task.cancelled():
            task.exception()

    def _load_policy(self) -> PolicySettings:
        if self.settings.allow_test_analyzer and self.settings.policy_config is None:
            return test_policy()
        if self.settings.policy_config is None:
            raise RuntimeNotReadyError("policy configuration is missing")
        return load_policy(self.settings.policy_config)

    def _create_session_store(self) -> SessionStore | None:
        config = self.policy_settings.session
        if not config.enabled:
            return None
        if self.settings.valkey_url is None:
            raise RuntimeNotReadyError("enabled session state requires Valkey")
        return SessionStore(
            self.settings.valkey_url.get_secret_value(),
            int(config.ttl_hours * 3600),
            self.settings.policy_version,
        )

    def _policy_service(self, policy: PolicySettings) -> PolicyService:
        planner = ActionPlanner(
            policy,
            self._anonymizer,
            self._hash_key,
            self.settings.policy_version,
        )
        return PolicyService(self.settings, policy, self._analyzer, planner)

    async def _cached_decision(
        self, caller: str, session_key: str | None, request_kind: RequestKind
    ) -> SessionDecision | None:
        if caller != "adapter" or self.session is None:
            return None
        key = _validated_session_key(session_key)
        if key is None:
            raise InvalidAnalysisRequestError("adapter requests require a trusted session key")
        cached = await self.session.get(key)
        if cached is not None and cached.request_kind != request_kind:
            raise RuntimeError("session cache request kind does not match the current request")
        return cached

    @staticmethod
    def _result_from_cache(request: SupportedRequest, cached: SessionDecision) -> PolicyResult:
        """Return a safe cached block or unmasked trusted-local reroute."""
        return PolicyResult(
            request=None if cached.decision == "block" else request,
            decision=cached.decision,
            remote_allowed=cached.remote_allowed,
            entities=cached.entities,
            entity_counts=cached.entity_counts,
            applied_actions=[cached.decision],
            report_rows=cached.report_rows,
            analysis_source="cached_decision",
            overlap_count=cached.overlap_count,
            cached_decision_applied=True,
            route_class=cached.route_class,
            request_notices=cached.request_notices,
            response_notices=cached.response_notices,
        )

    @classmethod
    def _final_cached_result(
        cls,
        request: SupportedRequest,
        cached: SessionDecision | None,
        policy: PolicySettings,
    ) -> PolicyResult | None:
        """Return cache results that need no current-request transformation."""
        if cached is None:
            return None
        if cached.decision == "block" or not policy.pii.mask_on_reroute:
            return cls._result_from_cache(request, cached)
        return None

    @classmethod
    def _apply_cached_decision(cls, result: PolicyResult, cached: SessionDecision | None) -> None:
        """Apply sticky routing after current-request masking has completed."""
        if cached is not None and cached.decision == "reroute" and result.decision != "block":
            cls._enforce_cached_reroute(result, cached)

    @staticmethod
    def _enforce_cached_reroute(result: PolicyResult, cached: SessionDecision) -> None:
        """Keep a masked reroute sticky after reprocessing the current request."""
        result.decision = "reroute"
        result.remote_allowed = False
        result.route_class = cached.route_class
        result.entities = sorted(set(result.entities) | set(cached.entities))
        for entity, count in cached.entity_counts.items():
            result.entity_counts[entity] = max(result.entity_counts.get(entity, 0), count)
        result.applied_actions = sorted(set(result.applied_actions) | {"reroute"})
        result.request_notices = cached.request_notices
        result.response_notices = cached.response_notices
        result.cached_decision_applied = True

    async def _taint_session(
        self,
        caller: str,
        session_key: str | None,
        result: PolicyResult,
        cached: SessionDecision | None,
        request_kind: RequestKind,
    ) -> None:
        if (
            caller != "adapter"
            or self.session is None
            or result.decision not in {"block", "reroute"}
        ):
            return
        key = _validated_session_key(session_key)
        if key is None:
            raise InvalidAnalysisRequestError(
                "tainted adapter decisions require a trusted session key"
            )
        report_rows = result.report_rows
        overlap_count = result.overlap_count
        if result.cached_decision_applied:
            if cached is None:
                raise ValueError("cache-applied result is missing its session decision")
            report_rows = cached.report_rows
            overlap_count = cached.overlap_count
        await self.session.set(
            key,
            SessionDecision(
                api_version="v1",
                request_kind=request_kind,
                decision=result.decision,
                entities=result.entities,
                entity_counts=result.entity_counts,
                report_rows=report_rows,
                overlap_count=overlap_count,
                route_class=result.route_class,
                remote_allowed=result.remote_allowed,
                request_notices=result.request_notices,
                response_notices=result.response_notices,
            ),
        )

    @staticmethod
    def _validate_result_kind(result: PolicyResult, request_kind: RequestKind) -> None:
        """Fail closed if MCP ever escapes its terminal routing and notice contract."""
        if request_kind == "mcp" and (
            result.decision == "reroute"
            or result.route_class is not None
            or result.request_notices
            or result.response_notices
            or any(row.action == "reroute" for row in result.report_rows)
        ):
            raise ValueError("MCP analysis produced model-only routing state")

    def _cache_still_available(self) -> bool:
        """Cheaply detect a lost immutable cache without rehashing models per probe."""
        if self.analyzer_mode == "test":
            return True
        if self.analyzer_mode == "baseline":
            return not self.restart_required()
        bundle = self.settings.model_bundle_path
        digest = self.settings.model_manifest_sha256
        version = self.settings.model_bundle_version
        cache = self.settings.model_cache_path
        reference = self.settings.model_bundle_reference
        if (
            bundle is None
            or digest is None
            or version is None
            or cache is None
            or reference is None
        ):
            return False
        try:
            marker = (bundle / ".complete").read_text(encoding="ascii").strip()
        except (OSError, UnicodeError):
            return False
        return (
            marker == digest.lower()
            and (bundle / "manifest.yaml").is_file()
            and desired_reference_matches(cache, reference, digest, version)
        )

    @staticmethod
    def _record_result(caller: str, result: PolicyResult) -> None:
        policy_requests_total.labels(caller=caller, decision=result.decision).inc()
        if result.scan_performed:
            overlap_regions_total.labels(caller=caller).inc(result.overlap_count)
        logger.info(
            "analysis completed caller=%s decision=%s entities=%s text_leaves=%d overlaps=%d",
            caller,
            result.decision,
            ",".join(result.entities),
            result.text_leaf_count,
            result.overlap_count,
        )


def _validated_session_key(value: str | None) -> str | None:
    if value is None:
        return None
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise InvalidAnalysisRequestError("invalid trusted session key")
    return value


def _conversation_placeholder_namespace(
    session_key: str, hash_key: bytes, policy_version: str
) -> str:
    """Derive an unlinkable stable namespace for one trusted model conversation."""
    message = (f"pii-engine:v1:conversation-placeholder:{policy_version}:{session_key}").encode()
    return hmac.new(hash_key, message, hashlib.sha256).hexdigest()[:16]


def _request_kind(request: SupportedRequest) -> RequestKind:
    return "mcp" if isinstance(request, McpRequest) else "model"


def _invalid_evaluation(
    stage: EvaluationIssueStage, exc: ValidationError
) -> PolicyEvaluationResult:
    """Convert Pydantic errors to fixed messages without input or context."""
    errors = exc.errors(include_url=False, include_context=False, include_input=False)
    issues = [
        PolicyEvaluationIssue(
            stage=stage,
            path=_safe_issue_path(error["loc"]),
            code=_safe_issue_code(str(error["type"])),
            message=_issue_message(str(error["type"])),
        )
        for error in errors[:_MAX_EVALUATION_ISSUES]
    ]
    return PolicyEvaluationResult(
        issues=issues,
        issues_truncated=len(errors) > _MAX_EVALUATION_ISSUES,
    )


def _single_issue(stage: EvaluationIssueStage, code: str, message: str) -> PolicyEvaluationResult:
    return PolicyEvaluationResult(
        issues=[PolicyEvaluationIssue(stage=stage, path=[], code=code, message=message)]
    )


def _safe_issue_path(location: tuple[int | str, ...]) -> list[int | str]:
    """Expose only recognized field names and bounded list indexes."""
    path: list[int | str] = []
    for part in location[:16]:
        if isinstance(part, int):
            path.append(min(max(part, 0), 10_000_000))
        elif part in _POLICY_PATH_COMPONENTS:
            path.append(part)
        else:
            path.append("field")
    return path


def _safe_issue_code(error_type: str) -> str:
    """Normalize a Pydantic error type into the bounded stable code grammar."""
    normalized = "".join(
        character if character.isascii() and (character.islower() or character.isdigit()) else "_"
        for character in error_type.lower()
    ).strip("_")
    return (normalized or "invalid_policy")[:64]


def _issue_message(error_type: str) -> str:
    """Return a fixed safe explanation for common candidate schema failures."""
    if error_type == "extra_forbidden":
        return "Field is not allowed."
    if error_type == "missing":
        return "Required field is missing."
    if "too_long" in error_type:
        return "Value exceeds the allowed size."
    return "Policy field is invalid."


_runtime: EngineRuntime | None = None


def initialize_runtime(settings: Settings) -> EngineRuntime:
    """Build and register the process runtime."""
    global _runtime
    _runtime = EngineRuntime(settings)
    return _runtime


def set_runtime(runtime: EngineRuntime) -> None:
    """Register an already constructed runtime for both application listeners."""
    global _runtime
    _runtime = runtime


def get_runtime() -> EngineRuntime:
    """Return the registered runtime or reject pre-startup access."""
    if _runtime is None:
        raise RuntimeNotReadyError("policy runtime is not initialized")
    return _runtime

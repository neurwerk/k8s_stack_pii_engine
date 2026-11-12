"""Contractual shared policy pipeline for every supported LLM request."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

from pii_engine.config.policy import PolicySettings
from pii_engine.config.settings import Settings
from pii_engine.lib.safety import SAFETY_BY_NAME, SafetyRule
from pii_engine.metrics import actions_total, entities_total
from pii_engine.models.contracts import (
    DocumentAnalyzeRequest,
    LimitDetail,
    PIIAction,
    PIIReportRow,
    SegmentRequest,
    SupportedRequest,
    TextSegment,
)
from pii_engine.services.analyzer import Analyzer, EntityMatch
from pii_engine.services.errors import AnalysisRequestTooLargeError, InvalidAnalysisRequestError
from pii_engine.services.planner import ActionPlanner, LeafPlan, request_nonce
from pii_engine.services.remote_http import analysis_budget
from pii_engine.services.traversal import (
    TextLeaf,
    legacy_segments,
    restore_legacy_result,
)

_MAX_DIAGNOSTICS_PER_KIND = 2_048


@dataclass(frozen=True)
class LogicalDetectionData:
    """Retain safe logical detection evidence for the Studio response."""

    segment_id: str
    start: int
    end: int
    entity_type: str
    score: float
    source: Literal["deterministic", "spacy", "transformer", "policy_regex"]
    configured_action: PIIAction
    resolved_action: PIIAction
    path: tuple[str | int, ...] = ()


@dataclass(frozen=True)
class EffectiveRegionData:
    """Retain safe effective overlap evidence for the Studio response."""

    segment_id: str
    start: int
    end: int
    entity_type: str
    action: PIIAction
    source: Literal["deterministic", "spacy", "transformer", "policy_regex"]
    score: float
    member_entity_types: tuple[str, ...]
    overlap: bool
    path: tuple[str | int, ...] = ()


@dataclass
class PolicyResult:
    """Hold a fully evaluated request without retaining cross-request plaintext."""

    decision: Literal["pass", "block", "apply_actions", "reroute"]
    remote_allowed: bool
    segments: list[TextSegment] | None = None
    request: SupportedRequest | None = None
    entities: list[str] = field(default_factory=list)
    entity_counts: dict[str, int] = field(default_factory=dict)
    applied_actions: list[str] = field(default_factory=list)
    report_rows: list[PIIReportRow] = field(default_factory=list)
    analysis_source: Literal["current_request", "cached_decision"] = "current_request"
    scan_performed: bool = False
    duration_ms: int | None = None
    overlap_count: int = 0
    cached_decision_applied: bool = False
    route_class: str | None = None
    reversal: dict[str, str] = field(default_factory=dict)
    safety_rule: str | None = None
    request_notices: list[str] = field(default_factory=list)
    response_notices: list[str] = field(default_factory=list)
    text_leaf_count: int = 0
    logical_detections: list[LogicalDetectionData] = field(default_factory=list)
    effective_regions: list[EffectiveRegionData] = field(default_factory=list)
    diagnostics_truncated: bool = False


class PolicyService:
    """Apply the fixed parse-to-notice pipeline using in-process dependencies."""

    def __init__(
        self,
        settings: Settings,
        policy: PolicySettings,
        analyzer: Analyzer,
        planner: ActionPlanner,
    ) -> None:
        """Store validated immutable runtime dependencies."""
        self.settings = settings
        self.policy = policy
        self.analyzer = analyzer
        self.planner = planner
        self._safety_rules = self._compile_safety_rules()
        self._entity_patterns = self._compile_entity_patterns()

    def analyze(
        self,
        request: SegmentRequest | SupportedRequest | DocumentAnalyzeRequest,
        *,
        include_diagnostics: bool = False,
        placeholder_namespace: str | None = None,
    ) -> PolicyResult:
        """Adapt legacy callers once, then run the provider-independent segment core."""
        if isinstance(request, SegmentRequest):
            return self.analyze_segments(
                request,
                include_diagnostics=include_diagnostics,
                placeholder_namespace=placeholder_namespace,
            )
        segments, extracted = legacy_segments(request, self.settings.max_nesting_depth)
        result = self.analyze_segments(
            segments,
            include_diagnostics=include_diagnostics,
            placeholder_namespace=placeholder_namespace,
        )
        restore_legacy_result(result, extracted)
        return result

    def analyze_segments(
        self,
        request: SegmentRequest,
        *,
        include_diagnostics: bool = False,
        placeholder_namespace: str | None = None,
    ) -> PolicyResult:
        """Run safety and global policy decisions over independent caller segments."""
        leaves = [TextLeaf((segment.id,), segment.text) for segment in request.segments]
        result = self._preflight(request, leaves)
        source_rows, source_routes = self._source_restrictions(request)
        face_route = None
        if request.visual_findings is not None:
            faces = request.visual_findings.faces
            face_policy = self.policy.attachments.faces
            if faces.count and face_policy.action == "reroute":
                face_route = face_policy.route_class or self.policy.routing.default_target
            if result is None and (
                faces.scan_status == "failed" or (faces.count and face_policy.action == "block")
            ):
                actions_total.labels(action="block").inc()
                result = PolicyResult(
                    request=None,
                    decision="block",
                    remote_allowed=False,
                    applied_actions=["block"],
                    text_leaf_count=len(leaves),
                )
        if result is None and (
            any(row.action == "block" for row in source_rows) or len(source_routes) > 1
        ):
            result = PolicyResult(
                decision="block",
                remote_allowed=False,
                applied_actions=["block"],
                text_leaf_count=len(leaves),
            )
        if result is None and not request.text_pii_enabled:
            result = PolicyResult(
                segments=request.segments,
                decision="pass",
                remote_allowed=True,
                text_leaf_count=len(leaves),
            )
            required_routes = source_routes | ({face_route} if face_route else set())
            if len(required_routes) > 1:
                block_result(result)
            elif required_routes:
                result.decision = "reroute"
                result.remote_allowed = False
                result.route_class = next(iter(required_routes))
        if result is None:
            result = self._analyze_text(
                request,
                leaves,
                include_diagnostics=include_diagnostics,
                placeholder_namespace=placeholder_namespace,
                face_route=face_route,
                source_routes=source_routes,
            )
        if request.visual_findings is not None:
            self._apply_visual_result(result, request, face_route)
        self._apply_source_result(result, source_rows)
        validate_entity_bound(result.entity_counts)
        return result

    def _source_restrictions(self, request: SegmentRequest) -> tuple[list[PIIReportRow], set[str]]:
        """Resolve trusted whole-request classifications independently of text detection."""
        policies = {entry.entity_type: entry for entry in self.policy.pii.entity_policies}
        rows: list[PIIReportRow] = []
        routes: set[str] = set()
        for source in request.mcp_sources:
            entry = policies.get(source, policies.get("MCP_TOOL_DEFAULT"))
            action = cast(PIIAction, entry.action if entry is not None else "block")
            if action == "reroute" and request.destination_local:
                action = "pass"
            if action == "reroute":
                routes.add(
                    entry.route_class
                    if entry is not None and entry.route_class
                    else self.policy.routing.default_target
                )
            rows.append(
                PIIReportRow(
                    entity_type=source,
                    action=action,
                    detected_count=1,
                    transformed_count=0,
                    unique_transformed_count=0,
                )
            )
        return rows, routes

    def _apply_source_result(self, result: PolicyResult, rows: list[PIIReportRow]) -> None:
        """Report source evidence without inventing spans or transformed content."""
        for row in rows:
            result.entity_counts[row.entity_type] = 1
            result.report_rows.append(row)
        result.entities = sorted(result.entity_counts)
        result.report_rows.sort(key=lambda row: row.entity_type)
        if result.decision == "block":
            block_result(result)
        else:
            result.applied_actions = sorted(
                set(result.applied_actions) | {row.action for row in rows}
            )
            if (
                result.decision == "reroute"
                and self.policy.notice.rerouted not in result.response_notices
            ):
                result.response_notices.append(self.policy.notice.rerouted)

    def _analyze_text(
        self,
        request: SegmentRequest,
        leaves: list[TextLeaf],
        *,
        include_diagnostics: bool,
        placeholder_namespace: str | None,
        face_route: str | None,
        source_routes: set[str],
    ) -> PolicyResult:
        """Resolve text and face routing conflicts before transforming any text."""
        nonce = placeholder_namespace or request_nonce()
        prepared, entities, counts, _route_classes, reroute_entities, overlap_count = (
            self._prepare_plans(leaves, reroute_as_block=request.request_kind == "mcp")
        )
        logical_detections, effective_regions, diagnostics_truncated = (
            self._diagnostics(prepared) if include_diagnostics else ([], [], False)
        )
        original_text = "\n".join(leaf.text for leaf in leaves)
        required_routes = source_routes | self._required_text_routes(reroute_entities)
        if face_route is not None:
            required_routes.add(face_route)
        if any(plan.blocked for _leaf, plan in prepared) or len(required_routes) > 1:
            result = PolicyResult(
                request=None,
                decision="block",
                remote_allowed=False,
                entities=sorted(entities),
                entity_counts=counts,
                applied_actions=["block"],
                report_rows=_report_rows(prepared),
                scan_performed=True,
                overlap_count=overlap_count,
                text_leaf_count=len(leaves),
                logical_detections=logical_detections,
                effective_regions=effective_regions,
                diagnostics_truncated=diagnostics_truncated,
            )
            block_result(result)
            return result

        route_class = next(iter(required_routes), None)
        transformed, actions, reversal = self._transform_plans(request, prepared, nonce)
        if any(placeholder in original_text for placeholder in reversal):
            raise ValueError("generated reversal placeholder already existed in request")
        decision: Literal["pass", "apply_actions", "reroute"]
        if route_class is not None:
            decision = "reroute"
        elif actions - {"pass"}:
            decision = "apply_actions"
        else:
            decision = "pass"
        if route_class is None and request.request_kind != "mcp":
            route_class = self._classify(transformed)
        response_notices = (
            []
            if request.request_kind == "mcp"
            else self._response_notices(decision, bool(entities), actions)
        )
        return PolicyResult(
            segments=transformed,
            decision=decision,
            remote_allowed=decision != "reroute",
            entities=sorted(entities),
            entity_counts=counts,
            applied_actions=sorted(actions),
            report_rows=_report_rows(prepared),
            scan_performed=True,
            overlap_count=overlap_count,
            route_class=route_class,
            reversal=reversal,
            response_notices=response_notices,
            text_leaf_count=len(leaves),
            logical_detections=logical_detections,
            effective_regions=effective_regions,
            diagnostics_truncated=diagnostics_truncated,
        )

    def _required_text_routes(self, reroute_entities: set[str]) -> set[str]:
        """Compare canonical dispatch IDs, not nonterminal classifier strings."""
        policies = {entry.entity_type: entry for entry in self.policy.pii.entity_policies}
        return {
            entry.route_class
            if (entry := policies.get(entity)) is not None and entry.route_class
            else self.policy.routing.default_target
            for entity in reroute_entities
        }

    def _apply_visual_result(
        self, result: PolicyResult, document: SegmentRequest, face_route: str | None
    ) -> None:
        """Add aggregate face evidence without inventing text spans or transformations."""
        if document.visual_findings is None:
            return
        count = document.visual_findings.faces.count
        if not count:
            return
        action = "block" if result.decision == "block" else self.policy.attachments.faces.action
        if result.decision == "pass" and not result.entities:
            result.response_notices = []
        result.entity_counts["FACE"] = count
        result.entities = sorted(result.entity_counts)
        result.report_rows.append(
            PIIReportRow(
                entity_type="FACE",
                action=action,
                detected_count=count,
                transformed_count=0,
                unique_transformed_count=0,
            )
        )
        result.report_rows.sort(key=lambda row: row.entity_type)
        result.applied_actions = sorted(set(result.applied_actions) | {action})
        entities_total.labels(entity_type="FACE").inc(count)
        if action == "reroute":
            result.decision = "reroute"
            result.remote_allowed = False
            result.route_class = face_route
            result.response_notices = [self.policy.notice.rerouted]
        elif action == "text-only":
            if result.decision == "pass":
                result.decision = "apply_actions"
            result.response_notices.append("Faces were detected; images must be withheld.")
        if action != "block":
            actions_total.labels(action=action).inc()

    def _diagnostics(
        self, prepared: list[tuple[TextLeaf, LeafPlan]]
    ) -> tuple[list[LogicalDetectionData], list[EffectiveRegionData], bool]:
        """Build deterministic bounded diagnostics from original leaf-local plans."""
        logical_detections: list[LogicalDetectionData] = []
        effective_regions: list[EffectiveRegionData] = []
        truncated = False
        for leaf, plan in prepared:
            segment_id = str(leaf.path[0])
            for match in plan.matches:
                region = next(
                    item
                    for item in plan.effective_matches
                    if item.start < match.end and item.end > match.start
                )
                if len(logical_detections) < _MAX_DIAGNOSTICS_PER_KIND:
                    logical_detections.append(
                        LogicalDetectionData(
                            segment_id=segment_id,
                            start=match.start,
                            end=match.end,
                            entity_type=match.entity_type,
                            score=match.score,
                            source=_normalized_source(match.source),
                            configured_action=self._entity_action(match.entity_type),
                            resolved_action=plan.entity_actions[region.entity_type],
                        )
                    )
                else:
                    truncated = True
            for match in plan.effective_matches:
                members = sorted(
                    {
                        item.entity_type
                        for item in plan.matches
                        if item.start < match.end and item.end > match.start
                    }
                )
                if len(effective_regions) < _MAX_DIAGNOSTICS_PER_KIND:
                    effective_regions.append(
                        EffectiveRegionData(
                            segment_id=segment_id,
                            start=match.start,
                            end=match.end,
                            entity_type=match.entity_type,
                            action=plan.entity_actions[match.entity_type],
                            source=_normalized_source(match.source),
                            score=match.score,
                            member_entity_types=tuple(members),
                            overlap=len(members) > 1,
                        )
                    )
                else:
                    truncated = True
        return logical_detections, effective_regions, truncated

    def _entity_action(self, entity_type: str) -> PIIAction:
        """Return the configured entity action or the policy default."""
        action = next(
            (
                entry.action
                for entry in self.policy.pii.entity_policies
                if entry.entity_type == entity_type
            ),
            self.policy.pii.default_action,
        )
        return cast(PIIAction, action)

    def _prepare_plans(
        self, leaves: list[TextLeaf], *, reroute_as_block: bool
    ) -> tuple[
        list[tuple[TextLeaf, LeafPlan]],
        set[str],
        dict[str, int],
        list[str],
        set[str],
        int,
    ]:
        """Share a bounded remote-call budget across every independent leaf."""
        with analysis_budget(
            min(self.settings.analysis_timeout, self.policy.pii.timeout),
            self.settings.remote_max_calls,
        ):
            return self._prepare_leaf_plans(leaves, reroute_as_block=reroute_as_block)

    def _prepare_leaf_plans(
        self, leaves: list[TextLeaf], *, reroute_as_block: bool
    ) -> tuple[
        list[tuple[TextLeaf, LeafPlan]],
        set[str],
        dict[str, int],
        list[str],
        set[str],
        int,
    ]:
        """Analyze every leaf and resolve decisions before any transformation."""
        prepared: list[tuple[TextLeaf, LeafPlan]] = []
        entities: set[str] = set()
        counts: dict[str, int] = {}
        route_classes: list[str] = []
        reroute_entities: set[str] = set()
        overlap_count = 0
        for leaf in leaves:
            matches = self.analyzer.analyze(leaf.text, self.policy)
            matches.extend(self._custom_matches(leaf.text))
            plan = self.planner.prepare(leaf.text, matches, reroute_as_block=reroute_as_block)
            prepared.append((leaf, plan))
            entities.update(plan.entities)
            _merge_counts(counts, plan.entity_counts)
            reroute_entities.update(plan.reroute_entities)
            overlap_count += plan.overlap_count
            if plan.route_class:
                route_classes.append(plan.route_class)
        return prepared, entities, counts, route_classes, reroute_entities, overlap_count

    def _transform_plans(
        self,
        request: SegmentRequest,
        prepared: list[tuple[TextLeaf, LeafPlan]],
        nonce: str,
    ) -> tuple[list[TextSegment], set[str], dict[str, str]]:
        """Transform prepared plans only after global block resolution."""
        replacements: dict[tuple[str | int, ...], str] = {}
        actions: set[str] = set()
        reversal: dict[str, str] = {}
        for leaf, plan in prepared:
            self.planner.transform(plan, nonce)
            actions.update(plan.applied_actions)
            _merge_reversal(reversal, plan.reversal)
            if plan.text != leaf.text:
                replacements[leaf.path] = plan.text
        transformed = [
            TextSegment(id=segment.id, text=replacements.get((segment.id,), segment.text))
            for segment in request.segments
        ]
        return transformed, actions, reversal

    def _preflight(self, request: SegmentRequest, leaves: list[TextLeaf]) -> PolicyResult | None:
        """Apply bounds, attachment policy, and original-text safety before PII work."""
        if request.request_kind == "mcp" and not leaves and not request.attachments_present:
            return PolicyResult(segments=[], decision="pass", remote_allowed=True)
        attachments = request.attachments_present
        if (request.text_pii_enabled or request.visual_findings is not None) and (
            leaves or not attachments
        ):
            self._validate_bounds(leaves)
        else:
            self.validate_segment_limits(leaves)
        if attachments:
            actions_total.labels(action="block").inc()
            return PolicyResult(
                request=None,
                decision="block",
                remote_allowed=False,
                applied_actions=["block"],
                response_notices=[]
                if request.request_kind == "mcp"
                else ["Attachments are blocked by the configured policy."],
                text_leaf_count=len(leaves),
            )
        if safety := self._safety_match(leaves):
            actions_total.labels(action="block").inc()
            return PolicyResult(
                request=None,
                decision="block",
                remote_allowed=False,
                applied_actions=["block"],
                safety_rule=safety.name,
                response_notices=[] if request.request_kind == "mcp" else [safety.message],
                text_leaf_count=len(leaves),
            )
        return None

    def _validate_bounds(self, leaves: list[TextLeaf]) -> None:
        if not leaves:
            raise InvalidAnalysisRequestError("request contains no model-visible text")
        self.validate_segment_limits(leaves)

    def validate_segment_limits(self, leaves: Sequence[TextSegment | TextLeaf]) -> None:
        """Enforce measured semantic bounds before cache access or analysis."""
        if len(leaves) > self.settings.max_text_leaves:
            raise AnalysisRequestTooLargeError(
                "request contains too many text segments",
                limit=LimitDetail(
                    stage="inspection",
                    reason="segments",
                    measured=len(leaves),
                    maximum=self.settings.max_text_leaves,
                    unit="items",
                    exact=True,
                ),
            )
        characters = sum(len(leaf.text) for leaf in leaves)
        if characters > self.settings.max_text_characters:
            raise AnalysisRequestTooLargeError(
                "request contains too many text characters",
                limit=LimitDetail(
                    stage="inspection",
                    reason="text_characters",
                    measured=characters,
                    maximum=self.settings.max_text_characters,
                    unit="characters",
                    exact=True,
                ),
            )

    def _compile_safety_rules(self) -> tuple[SafetyRule, ...]:
        rules: list[SafetyRule] = []
        for name in self.policy.safety.enabled:
            try:
                rules.append(SAFETY_BY_NAME[name])
            except KeyError as exc:
                raise ValueError(f"unknown safety rule: {name}") from exc
        rules.extend(
            SafetyRule(entry.name, entry.pattern, entry.message)
            for entry in self.policy.safety.custom
        )
        return tuple(rules)

    def _compile_entity_patterns(self) -> tuple[tuple[str, re.Pattern[str]], ...]:
        return tuple(
            (entry.entity_type, re.compile(pattern))
            for entry in self.policy.pii.entity_policies
            for pattern in entry.patterns
        )

    def _safety_match(self, leaves: list[TextLeaf]) -> SafetyRule | None:
        for rule in self._safety_rules:
            if any(rule.matches(leaf.text) for leaf in leaves):
                return rule
        return None

    def _custom_matches(self, text: str) -> list[EntityMatch]:
        return [
            EntityMatch(entity, match.start(), match.end(), 0.95, "policy-regex")
            for entity, pattern in self._entity_patterns
            for match in pattern.finditer(text)
        ]

    def _classify(self, segments: list[TextSegment]) -> str:
        text = "\n".join(segment.text for segment in segments)
        for item in self.policy.classifier.classes:
            if any(re.search(pattern, text) for pattern in item.patterns):
                return item.name
        return self.policy.classifier.default_class

    def _response_notices(self, decision: str, has_entities: bool, actions: set[str]) -> list[str]:
        if decision == "reroute":
            return [self.policy.notice.rerouted]
        if has_entities and actions == {"pass"}:
            return ["Sensitive data was detected and passed through by policy."]
        if has_entities:
            return [self.policy.notice.masked]
        if self.policy.notice.show_when_no_pii_detected:
            return ["No sensitive data was detected by the configured policy."]
        return []


def validate_entity_bound(counts: dict[str, int]) -> None:
    """Reject aggregate overflow rather than discard policy evidence."""
    if len(counts) > 64:
        raise InvalidAnalysisRequestError("combined entity evidence exceeds the supported limit")


def block_result(result: PolicyResult) -> None:
    """Normalize effective terminal evidence and remove all forwarding material."""
    result.decision = "block"
    result.remote_allowed = False
    result.segments = None
    result.request = None
    result.route_class = None
    result.reversal = {}
    result.applied_actions = ["block"]
    has_block = any(row.action == "block" for row in result.report_rows)
    result.report_rows = [
        row.model_copy(
            update={
                "action": row.action if has_block else "block",
                "transformed_count": 0,
                "unique_transformed_count": 0,
            }
        )
        for row in result.report_rows
    ]


def _merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def _merge_reversal(target: dict[str, str], source: dict[str, str]) -> None:
    for placeholder, plaintext in source.items():
        existing = target.get(placeholder)
        if existing is not None and existing != plaintext:
            raise ValueError("one placeholder maps to multiple plaintext values")
        target[placeholder] = plaintext


def _report_rows(prepared: list[tuple[TextLeaf, LeafPlan]]) -> list[PIIReportRow]:
    """Collapse transient leaf details into safe deterministic aggregate rows."""
    actions: dict[str, PIIAction] = {}
    detected: dict[str, int] = {}
    transformed: dict[str, int] = {}
    transformed_values: dict[str, set[str]] = {}
    for _leaf, plan in prepared:
        for entity_type, count in plan.entity_counts.items():
            action = plan.entity_actions.get(entity_type)
            if action is None:
                raise ValueError("detected entity has no resolved action")
            existing = actions.get(entity_type)
            if existing is not None and existing != action:
                raise ValueError("one entity type resolved to inconsistent actions")
            actions[entity_type] = action
            detected[entity_type] = detected.get(entity_type, 0) + count
            transformed[entity_type] = transformed.get(
                entity_type, 0
            ) + plan.transformed_counts.get(entity_type, 0)
            transformed_values.setdefault(entity_type, set()).update(
                plan.transformed_values.get(entity_type, set())
            )
    return [
        PIIReportRow(
            entity_type=entity_type,
            action=actions[entity_type],
            detected_count=detected[entity_type],
            transformed_count=transformed[entity_type],
            unique_transformed_count=len(transformed_values[entity_type]),
        )
        for entity_type in sorted(detected)
    ]


def _normalized_source(
    source: str,
) -> Literal["deterministic", "spacy", "transformer", "policy_regex"]:
    """Collapse internal recognizer names into the fixed Studio taxonomy."""
    if source == "policy-regex":
        return "policy_regex"
    if source in {"spacy", "transformer"}:
        return source
    return "deterministic"

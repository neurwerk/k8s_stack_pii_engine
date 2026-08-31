"""Contractual shared policy pipeline for every supported LLM request."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, cast

from pii_engine.config.policy import PolicySettings
from pii_engine.config.settings import Settings
from pii_engine.lib.safety import SAFETY_BY_NAME, SafetyRule
from pii_engine.metrics import actions_total
from pii_engine.models.contracts import (
    AttachmentPart,
    McpRequest,
    OpenAIChatRequest,
    OpenAIResponsesRequest,
    PIIAction,
    PIIReportRow,
    ResponseMessage,
    SupportedRequest,
)
from pii_engine.services.analyzer import Analyzer, EntityMatch
from pii_engine.services.errors import AnalysisRequestTooLargeError, InvalidAnalysisRequestError
from pii_engine.services.planner import ActionPlanner, LeafPlan, request_nonce
from pii_engine.services.traversal import TextLeaf, iter_text_leaves, replace_text_leaves

_MAX_DIAGNOSTICS_PER_KIND = 2_048


@dataclass(frozen=True)
class LogicalDetectionData:
    """Retain safe logical detection evidence for the Studio response."""

    path: tuple[str | int, ...]
    start: int
    end: int
    entity_type: str
    score: float
    source: Literal["deterministic", "spacy", "transformer", "policy_regex"]
    configured_action: PIIAction
    resolved_action: PIIAction


@dataclass(frozen=True)
class EffectiveRegionData:
    """Retain safe effective overlap evidence for the Studio response."""

    path: tuple[str | int, ...]
    start: int
    end: int
    entity_type: str
    action: PIIAction
    source: Literal["deterministic", "spacy", "transformer", "policy_regex"]
    score: float
    member_entity_types: tuple[str, ...]
    overlap: bool


@dataclass
class PolicyResult:
    """Hold a fully evaluated request without retaining cross-request plaintext."""

    request: SupportedRequest | None
    decision: Literal["pass", "block", "apply_actions", "reroute"]
    remote_allowed: bool
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
        request: SupportedRequest,
        *,
        include_diagnostics: bool = False,
        placeholder_namespace: str | None = None,
    ) -> PolicyResult:
        """Run bounds, traversal, safety, PII, actions, and classification in order."""
        leaves = iter_text_leaves(request, self.settings.max_nesting_depth)
        if preflight := self._preflight(request, leaves):
            return preflight

        nonce = placeholder_namespace or request_nonce()
        prepared, entities, counts, route_classes, reroute_entities, overlap_count = (
            self._prepare_plans(leaves, reroute_as_block=isinstance(request, McpRequest))
        )
        logical_detections, effective_regions, diagnostics_truncated = (
            self._diagnostics(prepared) if include_diagnostics else ([], [], False)
        )
        original_text = "\n".join(leaf.text for leaf in leaves)
        if any(plan.blocked for _leaf, plan in prepared):
            return PolicyResult(
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

        route_class = self._resolve_route_class(reroute_entities, route_classes)
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
        if route_class is None and not isinstance(request, McpRequest):
            route_class = self._classify(transformed)
        response_notices = (
            []
            if isinstance(request, McpRequest)
            else self._response_notices(decision, bool(entities), actions)
        )
        return PolicyResult(
            request=transformed,
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

    def _diagnostics(
        self, prepared: list[tuple[TextLeaf, LeafPlan]]
    ) -> tuple[list[LogicalDetectionData], list[EffectiveRegionData], bool]:
        """Build deterministic bounded diagnostics from original leaf-local plans."""
        logical_detections: list[LogicalDetectionData] = []
        effective_regions: list[EffectiveRegionData] = []
        truncated = False
        for leaf, plan in prepared:
            public_path, path_truncated = _bounded_public_path(leaf.path)
            truncated = truncated or path_truncated
            for match in plan.matches:
                region = next(
                    item
                    for item in plan.effective_matches
                    if item.start < match.end and item.end > match.start
                )
                if len(logical_detections) < _MAX_DIAGNOSTICS_PER_KIND:
                    logical_detections.append(
                        LogicalDetectionData(
                            path=public_path,
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
                            path=public_path,
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
        request: SupportedRequest,
        prepared: list[tuple[TextLeaf, LeafPlan]],
        nonce: str,
    ) -> tuple[SupportedRequest, set[str], dict[str, str]]:
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
        transformed = replace_text_leaves(request, replacements) if replacements else request
        return transformed, actions, reversal

    def _preflight(self, request: SupportedRequest, leaves: list[TextLeaf]) -> PolicyResult | None:
        """Apply bounds, attachment policy, and original-text safety before PII work."""
        if isinstance(request, McpRequest) and not leaves:
            return PolicyResult(request=request, decision="pass", remote_allowed=True)
        has_attachments = _has_attachments(request)
        if leaves or not has_attachments:
            self._validate_bounds(leaves)
        if has_attachments:
            actions_total.labels(action="block").inc()
            return PolicyResult(
                request=None,
                decision="block",
                remote_allowed=False,
                applied_actions=["block"],
                response_notices=["Attachments are blocked by the configured policy."],
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
                response_notices=[] if isinstance(request, McpRequest) else [safety.message],
                text_leaf_count=len(leaves),
            )
        return None

    def _validate_bounds(self, leaves: list[TextLeaf]) -> None:
        if not leaves:
            raise InvalidAnalysisRequestError("request contains no model-visible text")
        if len(leaves) > self.settings.max_text_leaves:
            raise AnalysisRequestTooLargeError("request contains too many text leaves")
        if sum(len(leaf.text) for leaf in leaves) > self.settings.max_text_characters:
            raise AnalysisRequestTooLargeError("request contains too many text characters")

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

    def _resolve_route_class(self, reroute_entities: set[str], detected: list[str]) -> str | None:
        if not detected:
            return None
        for entry in self.policy.pii.entity_policies:
            if entry.entity_type in reroute_entities and entry.action == "reroute":
                return entry.route_class or self.policy.routing.default_target
        return detected[0]

    def _classify(self, request: SupportedRequest) -> str:
        text = "\n".join(
            leaf.text for leaf in iter_text_leaves(request, self.settings.max_nesting_depth)
        )
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


def _has_attachments(request: SupportedRequest) -> bool:
    """Inspect only schema-designated content blocks for blocked attachments."""
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


def _normalized_source(
    source: str,
) -> Literal["deterministic", "spacy", "transformer", "policy_regex"]:
    """Collapse internal recognizer names into the fixed Studio taxonomy."""
    if source == "policy-regex":
        return "policy_regex"
    if source in {"spacy", "transformer"}:
        return source
    return "deterministic"


def _bounded_public_path(path: tuple[str | int, ...]) -> tuple[tuple[str | int, ...], bool]:
    """Bound public path components without exposing the private storage marker."""
    bounded = tuple(
        min(part, 10_000_000) if isinstance(part, int) else part[:128] for part in path[:64]
    )
    truncated = len(path) > 64 or any(isinstance(part, str) and len(part) > 128 for part in path)
    return bounded, truncated

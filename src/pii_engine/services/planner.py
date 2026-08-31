"""Plan and compose all entity actions without corrupting text offsets."""

from __future__ import annotations

import hashlib
import hmac
import math
import secrets
import time
from dataclasses import dataclass, field
from typing import cast

from pii_engine.config.policy import EntityPolicy, PolicySettings
from pii_engine.lib.actions import ACTION_BY_NAME
from pii_engine.metrics import actions_total, entities_total
from pii_engine.models.contracts import PIIAction
from pii_engine.services.analyzer import EntityMatch
from pii_engine.services.anonymizer import Anonymizer


@dataclass(frozen=True)
class _LogicalMatch:
    """Represent one coalesced same-entity detection and its best owner."""

    entity_type: str
    start: int
    end: int
    score: float
    source: str
    owner_width: int

    def as_entity_match(self) -> EntityMatch:
        """Return the public span representation used for detection reporting."""
        return EntityMatch(self.entity_type, self.start, self.end, self.score, self.source)


@dataclass
class LeafPlan:
    """Hold one leaf's decision, transformation, and request-scoped reversals."""

    text: str
    entities: set[str] = field(default_factory=set)
    entity_counts: dict[str, int] = field(default_factory=dict)
    entity_actions: dict[str, PIIAction] = field(default_factory=dict)
    transformed_counts: dict[str, int] = field(default_factory=dict)
    applied_actions: set[str] = field(default_factory=set)
    reversal: dict[str, str] = field(default_factory=dict)
    blocked: bool = False
    route_class: str | None = None
    overlap_count: int = 0
    reroute_entities: set[str] = field(default_factory=set)
    matches: list[EntityMatch] = field(default_factory=list, repr=False)
    effective_matches: list[EntityMatch] = field(default_factory=list, repr=False)
    transformed_values: dict[str, set[str]] = field(default_factory=dict, repr=False)


class ActionPlanner:
    """Resolve and apply configured actions using Presidio where required."""

    def __init__(
        self,
        policy: PolicySettings,
        anonymizer: Anonymizer,
        hash_key: bytes,
        policy_scope: str,
    ) -> None:
        """Store validated policy and runtime-only key material."""
        self.policy = policy
        self.anonymizer = anonymizer
        self.hash_key = hash_key
        self.policy_scope = policy_scope
        self._by_entity = {entry.entity_type: entry for entry in policy.pii.entity_policies}

    def apply(
        self,
        text: str,
        matches: list[EntityMatch],
        nonce: str,
        *,
        reroute_as_block: bool = False,
    ) -> LeafPlan:
        """Prepare and transform one leaf for isolated callers and tests."""
        plan = self.prepare(text, matches, reroute_as_block=reroute_as_block)
        if not plan.blocked:
            self.transform(plan, nonce)
        return plan

    def prepare(
        self, text: str, matches: list[EntityMatch], *, reroute_as_block: bool = False
    ) -> LeafPlan:
        """Validate spans and resolve actions without mutating text."""
        logical, effective, overlap_count = self._resolve_matches(
            matches, len(text), reroute_as_block
        )
        plan = LeafPlan(
            text=text,
            matches=logical,
            effective_matches=effective,
            overlap_count=overlap_count,
        )
        self._summarize_detections(plan, logical, reroute_as_block)
        self._summarize_effective_actions(plan, effective, reroute_as_block)
        return plan

    def transform(self, plan: LeafPlan, nonce: str) -> LeafPlan:
        """Apply a prepared non-blocking plan from right to left."""
        if plan.blocked:
            raise ValueError("blocked plans cannot be transformed")
        plan.text = self._transform(plan.text, plan.effective_matches, nonce, plan)
        return plan

    def _summarize_detections(
        self, plan: LeafPlan, matches: list[EntityMatch], reroute_as_block: bool
    ) -> None:
        """Collect every logical detection independently from effective actions."""
        for match in matches:
            action, _entry = self._effective_action(match.entity_type, reroute_as_block)
            existing = plan.entity_actions.get(match.entity_type)
            if existing is not None and existing != action:
                raise ValueError("one entity type resolved to inconsistent actions")
            plan.entity_actions[match.entity_type] = action
            plan.entities.add(match.entity_type)
            plan.entity_counts[match.entity_type] = plan.entity_counts.get(match.entity_type, 0) + 1
            entities_total.labels(entity_type=match.entity_type).inc()

    def _summarize_effective_actions(
        self, plan: LeafPlan, matches: list[EntityMatch], reroute_as_block: bool
    ) -> None:
        """Record only actions selected to execute after overlap resolution."""
        for match in matches:
            action, entry = self._effective_action(match.entity_type, reroute_as_block)
            if action == "pass":
                plan.applied_actions.add("pass")
                actions_total.labels(action="pass").inc()
            elif action == "block":
                plan.blocked = True
                plan.applied_actions.add("block")
                actions_total.labels(action="block").inc()
            elif action == "reroute":
                plan.applied_actions.add("reroute")
                actions_total.labels(action="reroute").inc()
                plan.reroute_entities.add(match.entity_type)
                if plan.route_class is None:
                    plan.route_class = (
                        entry.route_class
                        if entry and entry.route_class
                        else self.policy.routing.default_target
                    )

    def _transform(self, text: str, matches: list[EntityMatch], nonce: str, plan: LeafPlan) -> str:
        """Apply non-blocking transformations from right to left."""
        placeholders: dict[tuple[str, str, str], str] = {}
        result = text
        for match in sorted(matches, key=lambda item: (item.start, item.end), reverse=True):
            action, entry = self._resolve(match.entity_type)
            if action in {"pass", "block"}:
                continue
            if action == "reroute" and not self.policy.pii.mask_on_reroute:
                continue
            effective = "mask" if action == "reroute" else action
            adjusted = EntityMatch(
                match.entity_type, match.start, match.end, match.score, match.source
            )
            original = result[adjusted.start : adjusted.end]
            if effective == "reversible_replace":
                replacement = self._placeholder(
                    adjusted.entity_type, original, nonce, "REV", placeholders
                )
                result = result[: adjusted.start] + replacement + result[adjusted.end :]
                self._add_reversal(plan.reversal, replacement, original)
            elif effective == "hash":
                replacement = self._rolling_hash(adjusted.entity_type, original)
                result = result[: adjusted.start] + replacement + result[adjusted.end :]
            elif effective == "encrypt":
                encrypted = self.anonymizer.apply(result, adjusted, "encrypt", {})
                cipher = encrypted[adjusted.start : len(encrypted) - (len(result) - adjusted.end)]
                replacement = self._placeholder(
                    adjusted.entity_type, cipher, nonce, "ENCRYPTED", placeholders
                )
                result = result[: adjusted.start] + replacement + result[adjusted.end :]
                self._add_reversal(plan.reversal, replacement, original)
            else:
                params = self._operator_params(effective, entry, adjusted.entity_type)
                result = self.anonymizer.apply(result, adjusted, effective, params)
            if action != "reroute":
                plan.applied_actions.add(action)
                actions_total.labels(action=action).inc()
            plan.transformed_counts[match.entity_type] = (
                plan.transformed_counts.get(match.entity_type, 0) + 1
            )
            plan.transformed_values.setdefault(match.entity_type, set()).add(original)
        return result

    def _resolve(self, entity_type: str) -> tuple[PIIAction, EntityPolicy | None]:
        entry = self._by_entity.get(entity_type)
        action, resolved_entry = (
            (entry.action, entry) if entry is not None else (self.policy.pii.default_action, None)
        )
        return cast(PIIAction, action), resolved_entry

    def _effective_action(
        self, entity_type: str, reroute_as_block: bool
    ) -> tuple[PIIAction, EntityPolicy | None]:
        """Normalize reroute where the destination protocol has no safe route."""
        action, entry = self._resolve(entity_type)
        return ("block", entry) if reroute_as_block and action == "reroute" else (action, entry)

    def _operator_params(
        self, action: str, entry: EntityPolicy | None, entity_type: str
    ) -> dict[str, object]:
        if entry is None:
            defaults = self.policy.pii.default_operator
            if action == "replace":
                return {"new_value": defaults.new_value or f"<{entity_type}>"}
            if action == "redact":
                return {}
            return {
                "masking_char": defaults.masking_char,
                "chars_to_mask": defaults.chars_to_mask,
                "from_end": defaults.from_end,
            }
        if action == "replace":
            value = entry.new_value or "<ENTITY>"
            return {"new_value": value.replace("<ENTITY>", f"<{entity_type}>")}
        if action == "redact":
            return {}
        return {
            "masking_char": entry.masking_char,
            "chars_to_mask": entry.chars_to_mask,
            "from_end": entry.from_end,
        }

    def _rolling_hash(self, entity_type: str, value: str) -> str:
        bucket = int(time.time()) // (self.policy.pii.hash_window_hours * 3600)
        message = (
            f"pii-engine:v1:rolling-hash:{self.policy_scope}:{bucket}:{entity_type}:{value}"
        ).encode()
        return hmac.new(self.hash_key, message, hashlib.sha256).hexdigest()

    def _placeholder(
        self,
        entity_type: str,
        value: str,
        nonce: str,
        prefix: str,
        cache: dict[tuple[str, str, str], str],
    ) -> str:
        key = (prefix, entity_type, value)
        if key not in cache:
            digest = hmac.new(
                self.hash_key,
                f"pii-engine:v1:placeholder:{nonce}:{entity_type}:{value}".encode(),
                hashlib.sha256,
            ).hexdigest()[:16]
            cache[key] = f"<{prefix}_{entity_type}_{nonce}_{digest}>"
        return cache[key]

    @staticmethod
    def _add_reversal(reversal: dict[str, str], placeholder: str, value: str) -> None:
        existing = reversal.get(placeholder)
        if existing is not None and existing != value:
            raise ValueError("one placeholder maps to multiple plaintext values")
        reversal[placeholder] = value

    def _resolve_matches(
        self, matches: list[EntityMatch], text_length: int, reroute_as_block: bool
    ) -> tuple[list[EntityMatch], list[EntityMatch], int]:
        """Return logical detections and safe effective union-span actions."""
        self._validate_spans(matches, text_length)
        logical = self._coalesce_same_entity(matches)
        effective: list[EntityMatch] = []
        overlap_count = 0
        group: list[_LogicalMatch] = []
        group_end = -1
        for match in sorted(
            logical, key=lambda item: (item.start, item.end, item.entity_type, item.source)
        ):
            if group and match.start >= group_end:
                selected, overlapped = self._resolve_group(group, reroute_as_block)
                effective.append(selected)
                overlap_count += overlapped
                group = []
            group.append(match)
            group_end = max(group_end if len(group) > 1 else match.end, match.end)
        if group:
            selected, overlapped = self._resolve_group(group, reroute_as_block)
            effective.append(selected)
            overlap_count += overlapped
        return [match.as_entity_match() for match in logical], effective, overlap_count

    @staticmethod
    def _validate_spans(matches: list[EntityMatch], text_length: int) -> None:
        """Reject invalid analyzer output before deduplication or grouping."""
        for match in matches:
            if match.start < 0 or match.end > text_length or match.end <= match.start:
                raise ValueError("analyzer returned an invalid entity span")
            if not math.isfinite(match.score) or not 0 <= match.score <= 1:
                raise ValueError("analyzer returned an invalid entity score")

    @classmethod
    def _coalesce_same_entity(cls, matches: list[EntityMatch]) -> list[_LogicalMatch]:
        """Coalesce connected same-entity duplicates into logical detections."""
        by_entity: dict[str, list[EntityMatch]] = {}
        for match in set(matches):
            by_entity.setdefault(match.entity_type, []).append(match)
        logical: list[_LogicalMatch] = []
        for entity_type in sorted(by_entity):
            group: list[EntityMatch] = []
            group_end = -1
            for match in sorted(
                by_entity[entity_type],
                key=lambda item: (item.start, item.end, -item.score, item.source),
            ):
                if group and match.start >= group_end:
                    logical.append(cls._coalesced(group))
                    group = []
                group.append(match)
                group_end = max(group_end if len(group) > 1 else match.end, match.end)
            if group:
                logical.append(cls._coalesced(group))
        return sorted(logical, key=lambda item: (item.start, item.end, item.entity_type))

    @staticmethod
    def _coalesced(matches: list[EntityMatch]) -> _LogicalMatch:
        """Build one logical occurrence and retain deterministic owner evidence."""
        owner = min(
            matches,
            key=lambda item: (
                -(item.end - item.start),
                -item.score,
                item.entity_type,
                item.source,
                item.start,
                item.end,
            ),
        )
        return _LogicalMatch(
            entity_type=owner.entity_type,
            start=min(match.start for match in matches),
            end=max(match.end for match in matches),
            score=owner.score,
            source=owner.source,
            owner_width=owner.end - owner.start,
        )

    def _resolve_group(
        self, matches: list[_LogicalMatch], reroute_as_block: bool
    ) -> tuple[EntityMatch, int]:
        """Choose the strictest owner and execute it over the complete union."""
        winner = min(
            matches,
            key=lambda item: (
                -ACTION_BY_NAME[
                    self._effective_action(item.entity_type, reroute_as_block)[0]
                ].strictness,
                -item.owner_width,
                -item.score,
                item.entity_type,
                item.source,
                item.start,
                item.end,
            ),
        )
        selected = EntityMatch(
            winner.entity_type,
            min(match.start for match in matches),
            max(match.end for match in matches),
            winner.score,
            winner.source,
        )
        overlapped = int(len({match.entity_type for match in matches}) > 1)
        return selected, overlapped


def request_nonce() -> str:
    """Return an unpredictable request-local placeholder namespace."""
    return secrets.token_hex(8)

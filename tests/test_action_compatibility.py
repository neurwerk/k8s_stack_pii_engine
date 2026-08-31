"""Exercise the complete action, routing, reversal, and notice contract."""

from __future__ import annotations

import re
from typing import Protocol

import pytest

import pii_engine.services.planner as planner_module
from pii_engine.config.policy import PolicySettings
from pii_engine.config.settings import Settings
from pii_engine.models.contracts import McpRequest, OpenAIChatRequest
from pii_engine.services.analyzer import DeterministicAnalyzer, EntityMatch
from pii_engine.services.anonymizer import TestAnonymizer
from pii_engine.services.planner import ActionPlanner, LeafPlan
from pii_engine.services.policy import PolicyService, _report_rows
from pii_engine.services.traversal import TextLeaf


class _Anonymizer(Protocol):
    def apply(self, text: str, match: EntityMatch, action: str, params: dict[str, object]) -> str:
        """Apply one action to a match."""


class _PatternAnalyzer:
    def analyze(self, text: str, policy: PolicySettings | None = None) -> list[EntityMatch]:
        """Return deterministic spans for compatibility fixtures."""
        matches: list[EntityMatch] = []
        for entity, value in (
            ("EMAIL_ADDRESS", "a@example.com"),
            ("EMAIL_ADDRESS", "b@example.com"),
            ("PHONE_NUMBER", "+49123456789"),
            ("PASSWORD_OR_SECRET", "hunter2"),
        ):
            start = text.find(value)
            if start >= 0:
                matches.append(EntityMatch(entity, start, start + len(value), 0.99, "test"))
        return matches


def _policy(action: str, **params: object) -> PolicySettings:
    return _policy_for([{"entityType": "EMAIL_ADDRESS", "action": action, **params}])


def _policy_for(entries: list[dict[str, object]]) -> PolicySettings:
    return PolicySettings.model_validate(
        {
            "pii": {
                "analyzerLanguages": ["en"],
                "supportedLanguages": ["en"],
                "defaultAction": "pass",
                "entityPolicies": entries,
            },
            "safety": {"enabled": []},
            "classifier": {"defaultClass": "general", "classes": []},
            "session": {"enabled": False},
            "notice": {
                "rerouted": "rerouted",
                "masked": "masked",
                "showWhenNoPiiDetected": False,
            },
            "routing": {"defaultTarget": "local", "targets": []},
        }
    )


def _service(
    policy: PolicySettings, anonymizer: _Anonymizer | None = None, scope: str = "test"
) -> PolicyService:
    active_anonymizer = anonymizer or TestAnonymizer()
    planner = ActionPlanner(policy, active_anonymizer, b"k" * 32, scope)
    return PolicyService(Settings(allow_test_analyzer=True), policy, _PatternAnalyzer(), planner)


def _request(text: str = "email a@example.com") -> OpenAIChatRequest:
    return OpenAIChatRequest(model="test", messages=[{"role": "user", "content": text}])


def _mcp_request(text: str = "email a@example.com") -> McpRequest:
    return McpRequest(
        jsonrpc="2.0",
        id=1,
        method="tools/call",
        params={"name": "lookup", "arguments": {"query": text}},
    )


def _content(request: OpenAIChatRequest) -> str:
    content = request.messages[0].content
    assert isinstance(content, str)
    return content


def _analyzed_content(service: PolicyService) -> str:
    result = service.analyze(_request())
    assert isinstance(result.request, OpenAIChatRequest)
    return _content(result.request)


@pytest.mark.parametrize(
    ("action", "params", "decision", "expected", "remote_allowed", "route_class"),
    [
        ("pass", {}, "pass", "email a@example.com", True, "general"),
        ("block", {}, "block", None, False, None),
        (
            "mask",
            {"masking_char": "#", "chars_to_mask": 4, "from_end": True},
            "apply_actions",
            "email a@example####",
            True,
            "general",
        ),
        (
            "replace",
            {"new_value": "<ENTITY>"},
            "apply_actions",
            "email <EMAIL_ADDRESS>",
            True,
            "general",
        ),
        ("redact", {}, "apply_actions", "email ", True, "general"),
        ("hash", {}, "apply_actions", "hash", True, "general"),
        ("encrypt", {}, "apply_actions", "encrypted", True, "general"),
        ("reversible_replace", {}, "apply_actions", "reversible", True, "general"),
        (
            "reroute",
            {"routeClass": "local-sensitive"},
            "reroute",
            "email " + "*" * 13,
            False,
            "local-sensitive",
        ),
    ],
)
def test_complete_action_contract(
    action: str,
    params: dict[str, object],
    decision: str,
    expected: str | None,
    remote_allowed: bool,
    route_class: str | None,
) -> None:
    """Every action has explicit externally visible transport semantics."""
    result = _service(_policy(action, **params)).analyze(_request())
    assert result.decision == decision
    assert result.remote_allowed is remote_allowed
    assert result.route_class == route_class
    assert result.entities == ["EMAIL_ADDRESS"]
    assert result.entity_counts == {"EMAIL_ADDRESS": 1}
    assert result.applied_actions == [action]
    assert result.text_leaf_count == 1
    transformed_count = 0 if action in {"pass", "block"} else 1
    assert [row.model_dump() for row in result.report_rows] == [
        {
            "entity_type": "EMAIL_ADDRESS",
            "action": action,
            "detected_count": 1,
            "transformed_count": transformed_count,
            "unique_transformed_count": transformed_count,
        }
    ]
    if expected is None:
        assert result.request is None
    else:
        assert isinstance(result.request, OpenAIChatRequest)
        transformed = _content(result.request)
        if expected == "hash":
            assert re.fullmatch(r"email [0-9a-f]{64}", transformed)
        elif expected in {"encrypted", "reversible"}:
            prefix = "ENCRYPTED" if expected == "encrypted" else "REV"
            pattern = rf"email <{prefix}_EMAIL_ADDRESS_[0-9a-f]{{16}}_[0-9a-f]{{16}}>"
            assert re.fullmatch(pattern, transformed)
        else:
            assert transformed == expected
    if action in {"encrypt", "reversible_replace"}:
        assert list(result.reversal.values()) == ["a@example.com"]
    else:
        assert result.reversal == {}
    if action == "reroute":
        assert result.request_notices == []
        assert result.response_notices == ["rerouted"]
    elif action == "pass":
        assert result.request_notices == []
        assert result.response_notices == [
            "Sensitive data was detected and passed through by policy."
        ]
    elif action == "block":
        assert result.request_notices == result.response_notices == []
    else:
        assert result.request_notices == []
        assert result.response_notices == ["masked"]


@pytest.mark.parametrize(
    ("action", "decision", "effective_action"),
    [
        ("pass", "pass", "pass"),
        ("block", "block", "block"),
        ("mask", "apply_actions", "mask"),
        ("replace", "apply_actions", "replace"),
        ("redact", "apply_actions", "redact"),
        ("hash", "apply_actions", "hash"),
        ("encrypt", "apply_actions", "encrypt"),
        ("reversible_replace", "apply_actions", "reversible_replace"),
        ("reroute", "block", "block"),
    ],
)
def test_mcp_action_contract_has_no_model_routing_or_notices(
    action: str, decision: str, effective_action: str
) -> None:
    params = {"routeClass": "local-sensitive"} if action == "reroute" else {}
    result = _service(_policy(action, **params)).analyze(_mcp_request())

    assert result.decision == decision
    assert result.remote_allowed is (decision != "block")
    assert result.route_class is None
    assert result.applied_actions == [effective_action]
    assert result.request_notices == []
    assert result.response_notices == []
    assert result.report_rows[0].action == effective_action
    if action == "reroute":
        assert result.request is None
        assert result.report_rows[0].transformed_count == 0
        assert result.reversal == {}
    elif action == "block":
        assert result.request is None
    else:
        assert isinstance(result.request, McpRequest)


def test_mcp_reroute_records_only_the_effective_block_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[str] = []

    class Counter:
        def labels(self, *, action: str) -> Counter:
            recorded.append(action)
            return self

        def inc(self) -> None:
            return None

    monkeypatch.setattr(planner_module, "actions_total", Counter())

    result = _service(_policy("reroute", routeClass="local-sensitive")).analyze(_mcp_request())

    assert result.decision == "block"
    assert recorded == ["block"]


def test_mixed_actions_preserve_offsets_and_report_each_action() -> None:
    """Length-changing and pass-through actions compose without offset corruption."""
    policy = _policy_for(
        [
            {"entityType": "EMAIL_ADDRESS", "action": "replace", "new_value": "<ENTITY>"},
            {"entityType": "PHONE_NUMBER", "action": "pass"},
        ]
    )
    result = _service(policy).analyze(_request("a@example.com +49123456789"))
    assert isinstance(result.request, OpenAIChatRequest)
    assert _content(result.request) == "<EMAIL_ADDRESS> +49123456789"
    assert result.decision == "apply_actions"
    assert result.applied_actions == ["pass", "replace"]
    assert result.entity_counts == {"EMAIL_ADDRESS": 1, "PHONE_NUMBER": 1}
    assert [row.model_dump() for row in result.report_rows] == [
        {
            "entity_type": "EMAIL_ADDRESS",
            "action": "replace",
            "detected_count": 1,
            "transformed_count": 1,
            "unique_transformed_count": 1,
        },
        {
            "entity_type": "PHONE_NUMBER",
            "action": "pass",
            "detected_count": 1,
            "transformed_count": 0,
            "unique_transformed_count": 0,
        },
    ]


def test_unique_transformed_values_are_counted_across_leaves() -> None:
    policy = _policy("mask")
    request = OpenAIChatRequest(
        model="test",
        messages=[
            {"role": "user", "content": "a@example.com"},
            {"role": "user", "content": "a@example.com"},
            {"role": "user", "content": "b@example.com"},
        ],
    )

    result = _service(policy).analyze(request)

    assert [row.model_dump() for row in result.report_rows] == [
        {
            "entity_type": "EMAIL_ADDRESS",
            "action": "mask",
            "detected_count": 3,
            "transformed_count": 3,
            "unique_transformed_count": 2,
        }
    ]


def test_unmasked_reroute_reports_no_transformations() -> None:
    policy = _policy("reroute", routeClass="local-sensitive")
    policy.pii.mask_on_reroute = False

    result = _service(policy).analyze(_request())

    assert isinstance(result.request, OpenAIChatRequest)
    assert _content(result.request) == "email a@example.com"
    assert result.report_rows[0].model_dump() == {
        "entity_type": "EMAIL_ADDRESS",
        "action": "reroute",
        "detected_count": 1,
        "transformed_count": 0,
        "unique_transformed_count": 0,
    }


def test_engine_owned_steuernummer_reroutes_to_the_local_target() -> None:
    policy = _policy_for([{"entityType": "STEUERNUMMER", "action": "reroute"}])
    policy.pii.mask_on_reroute = False
    planner = ActionPlanner(policy, TestAnonymizer(), b"k" * 32, "test")
    service = PolicyService(
        Settings(allow_test_analyzer=True), policy, DeterministicAnalyzer(), planner
    )

    result = service.analyze(_request("Meine Steuernummer ist 123/456/78901."))

    assert result.decision == "reroute"
    assert result.remote_allowed is False
    assert result.route_class == "local"
    assert result.entity_counts == {"PHONE_NUMBER": 1, "STEUERNUMMER": 1}
    assert result.overlap_count == 1
    assert [(row.entity_type, row.action, row.transformed_count) for row in result.report_rows] == [
        ("PHONE_NUMBER", "pass", 0),
        ("STEUERNUMMER", "reroute", 0),
    ]
    assert isinstance(result.request, OpenAIChatRequest)
    assert _content(result.request) == "Meine Steuernummer ist 123/456/78901."


@pytest.mark.parametrize(
    ("less_strict", "more_strict"),
    [
        ("pass", "mask"),
        ("mask", "hash"),
        ("hash", "encrypt"),
        ("encrypt", "reversible_replace"),
        ("reversible_replace", "replace"),
        ("replace", "redact"),
        ("redact", "reroute"),
        ("reroute", "block"),
    ],
)
def test_overlap_resolver_selects_each_stricter_adjacent_action(
    less_strict: str, more_strict: str
) -> None:
    policy = _policy_for(
        [
            {"entityType": "EMAIL_ADDRESS", "action": less_strict},
            {"entityType": "PHONE_NUMBER", "action": more_strict},
        ]
    )
    planner = ActionPlanner(policy, TestAnonymizer(), b"k" * 32, "test")

    plan = planner.prepare(
        "abcdefgh",
        [
            EntityMatch("EMAIL_ADDRESS", 0, 8, 0.99, "a"),
            EntityMatch("PHONE_NUMBER", 0, 8, 0.50, "b"),
        ],
    )

    assert plan.overlap_count == 1
    assert [match.entity_type for match in plan.effective_matches] == ["PHONE_NUMBER"]


def test_equal_action_overlap_owner_prefers_width_then_score() -> None:
    policy = _policy_for(
        [
            {"entityType": "EMAIL_ADDRESS", "action": "mask"},
            {"entityType": "PHONE_NUMBER", "action": "mask"},
        ]
    )
    planner = ActionPlanner(policy, TestAnonymizer(), b"k" * 32, "test")

    widest = planner.prepare(
        "abcdefgh",
        [
            EntityMatch("EMAIL_ADDRESS", 0, 8, 0.50, "a"),
            EntityMatch("PHONE_NUMBER", 0, 5, 0.99, "b"),
        ],
    )
    highest_score = planner.prepare(
        "abcdefgh",
        [
            EntityMatch("EMAIL_ADDRESS", 0, 8, 0.50, "a"),
            EntityMatch("PHONE_NUMBER", 0, 8, 0.99, "b"),
        ],
    )

    assert widest.effective_matches[0].entity_type == "EMAIL_ADDRESS"
    assert highest_score.effective_matches[0].entity_type == "PHONE_NUMBER"


def test_global_block_prevents_all_transformations() -> None:
    """A block in any leaf resolves before another leaf can create reversal data."""

    class RejectTransform:
        def apply(
            self, text: str, match: EntityMatch, action: str, params: dict[str, object]
        ) -> str:
            raise AssertionError("a blocked request reached transformation")

    policy = _policy_for(
        [
            {"entityType": "EMAIL_ADDRESS", "action": "mask"},
            {"entityType": "PASSWORD_OR_SECRET", "action": "block"},
        ]
    )
    request = OpenAIChatRequest(
        model="test",
        messages=[
            {"role": "user", "content": "a@example.com"},
            {"role": "user", "content": "password: hunter2"},
        ],
    )
    result = _service(policy, RejectTransform()).analyze(request)
    assert result.decision == "block"
    assert result.request is None
    assert result.reversal == {}
    assert result.entities == ["EMAIL_ADDRESS", "PASSWORD_OR_SECRET"]
    assert [row.model_dump() for row in result.report_rows] == [
        {
            "entity_type": "EMAIL_ADDRESS",
            "action": "mask",
            "detected_count": 1,
            "transformed_count": 0,
            "unique_transformed_count": 0,
        },
        {
            "entity_type": "PASSWORD_OR_SECRET",
            "action": "block",
            "detected_count": 1,
            "transformed_count": 0,
            "unique_transformed_count": 0,
        },
    ]


def test_report_rejects_inconsistent_actions_for_one_entity_type() -> None:
    first = LeafPlan(
        text="first",
        entity_counts={"EMAIL_ADDRESS": 1},
        entity_actions={"EMAIL_ADDRESS": "mask"},
    )
    second = LeafPlan(
        text="second",
        entity_counts={"EMAIL_ADDRESS": 1},
        entity_actions={"EMAIL_ADDRESS": "redact"},
    )

    with pytest.raises(ValueError, match="inconsistent actions"):
        _report_rows(
            [
                (TextLeaf(("first",), "first"), first),
                (TextLeaf(("second",), "second"), second),
            ]
        )


def test_hash_rotates_by_window_and_policy_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hashes are stable only within one configured window and policy scope."""
    monkeypatch.setattr(planner_module.time, "time", lambda: 1_000.0)
    first = _analyzed_content(_service(_policy("hash"), scope="tenant-a"))
    repeated = _analyzed_content(_service(_policy("hash"), scope="tenant-a"))
    other_scope = _analyzed_content(_service(_policy("hash"), scope="tenant-b"))
    monkeypatch.setattr(planner_module.time, "time", lambda: 1_000.0 + 24 * 3600)
    next_window = _analyzed_content(_service(_policy("hash"), scope="tenant-a"))
    assert first == repeated
    assert first != other_scope
    assert first != next_window

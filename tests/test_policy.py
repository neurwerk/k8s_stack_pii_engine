"""Test every action and fail-closed planner invariant."""

from __future__ import annotations

import re

import pytest

from pii_engine.config.policy import PolicySettings, RoutingSettings
from pii_engine.lib.actions import ACTION_BY_NAME
from pii_engine.services.analyzer import EntityMatch
from pii_engine.services.anonymizer import TestAnonymizer
from pii_engine.services.planner import ActionPlanner


def _policy(action: str, **params: object) -> PolicySettings:
    entry = {"entityType": "EMAIL_ADDRESS", "action": action, **params}
    return PolicySettings.model_validate(
        {
            "pii": {
                "analyzerLanguages": ["en"],
                "supportedLanguages": ["en"],
                "defaultAction": "pass",
                "entityPolicies": [entry],
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


def _apply(action: str, **params: object):
    text = "email a@example.com"
    match = EntityMatch("EMAIL_ADDRESS", 6, len(text), 0.99, "test")
    planner = ActionPlanner(_policy(action, **params), TestAnonymizer(), b"k" * 32, "test")
    return planner.apply(text, [match], "0123456789abcdef")


@pytest.mark.parametrize(
    ("action", "params", "expected"),
    [
        ("pass", {}, "email a@example.com"),
        (
            "mask",
            {"masking_char": "#", "chars_to_mask": 4, "from_end": True},
            "email a@example####",
        ),
        ("replace", {"new_value": "<ENTITY>"}, "email <EMAIL_ADDRESS>"),
        ("redact", {}, "email "),
    ],
)
def test_ordinary_actions(action: str, params: dict[str, object], expected: str) -> None:
    """Pass and Presidio-backed ordinary operators retain their semantics."""
    assert _apply(action, **params).text == expected


def test_block_and_reroute_decisions() -> None:
    """Block prevents transformation while reroute masks and selects a class."""
    blocked = _apply("block")
    rerouted = _apply("reroute", routeClass="local-sensitive")
    assert blocked.blocked is True
    assert rerouted.route_class == "local-sensitive"
    assert rerouted.text == "email " + "*" * 13


def test_reversible_replace_is_request_local_and_reuses_equal_values() -> None:
    """Equal request-local values share one nonce-bound unambiguous placeholder."""
    text = "a@example.com and a@example.com"
    matches = [
        EntityMatch("EMAIL_ADDRESS", 0, 13, 0.99, "test"),
        EntityMatch("EMAIL_ADDRESS", 18, 31, 0.99, "test"),
    ]
    planner = ActionPlanner(_policy("reversible_replace"), TestAnonymizer(), b"k" * 32, "test")
    result = planner.apply(text, matches, "0123456789abcdef")
    placeholders = re.findall(r"<REV_EMAIL_ADDRESS_[^>]+>", result.text)
    assert len(placeholders) == 2
    assert placeholders[0] == placeholders[1]
    assert result.reversal == {placeholders[0]: "a@example.com"}


def test_hash_is_keyed_and_encrypt_is_request_reversible() -> None:
    """Hash never exposes a raw digest input and encrypt returns only stream material."""
    hashed = _apply("hash")
    encrypted = _apply("encrypt")
    assert re.fullmatch(r"email [0-9a-f]{64}", hashed.text)
    assert "a@example.com" not in hashed.text
    placeholder = next(iter(encrypted.reversal))
    assert placeholder in encrypted.text
    assert encrypted.reversal[placeholder] == "a@example.com"


def test_overlapping_actions_use_strictest_action_over_union() -> None:
    """Cross-entity overlap is counted once and transformed over its full union."""
    planner = ActionPlanner(_policy("mask"), TestAnonymizer(), b"k" * 32, "test")
    matches = [
        EntityMatch("EMAIL_ADDRESS", 0, 5, 0.9, "a"),
        EntityMatch("PHONE_NUMBER", 4, 8, 0.9, "b"),
    ]
    result = planner.apply("abcdefgh", matches, "nonce")

    assert result.text == "********"
    assert result.overlap_count == 1
    assert result.entity_counts == {"EMAIL_ADDRESS": 1, "PHONE_NUMBER": 1}
    assert result.transformed_counts == {"EMAIL_ADDRESS": 1}


def test_same_entity_overlaps_coalesce_without_overlap_count() -> None:
    """Duplicate recognizer evidence becomes one logical and effective detection."""
    planner = ActionPlanner(_policy("mask"), TestAnonymizer(), b"k" * 32, "test")
    matches = [
        EntityMatch("EMAIL_ADDRESS", 0, 5, 0.8, "a"),
        EntityMatch("EMAIL_ADDRESS", 0, 8, 0.9, "b"),
    ]

    result = planner.apply("abcdefgh", matches, "nonce")

    assert result.text == "********"
    assert result.overlap_count == 0
    assert result.entity_counts == {"EMAIL_ADDRESS": 1}
    assert result.transformed_counts == {"EMAIL_ADDRESS": 1}


@pytest.mark.parametrize(
    "score",
    [float("nan"), float("inf"), float("-inf"), -0.01, 1.01],
    ids=["nan", "positive-infinity", "negative-infinity", "negative", "above-one"],
)
def test_planner_rejects_invalid_analyzer_scores(score: float) -> None:
    planner = ActionPlanner(_policy("mask"), TestAnonymizer(), b"k" * 32, "test")

    with pytest.raises(ValueError, match="invalid entity score"):
        planner.prepare("abcdefgh", [EntityMatch("EMAIL_ADDRESS", 0, 8, score, "test")])


def test_chained_cross_entity_overlaps_form_one_region() -> None:
    """Transitively intersecting spans resolve as one safe union region."""
    planner = ActionPlanner(_policy("mask"), TestAnonymizer(), b"k" * 32, "test")
    matches = [
        EntityMatch("EMAIL_ADDRESS", 0, 3, 0.9, "a"),
        EntityMatch("PHONE_NUMBER", 2, 6, 0.9, "b"),
        EntityMatch("VAT_NUMBER", 5, 8, 0.9, "c"),
    ]

    result = planner.apply("abcdefgh", matches, "nonce")

    assert result.text == "********"
    assert result.overlap_count == 1
    assert result.entity_counts == {
        "EMAIL_ADDRESS": 1,
        "PHONE_NUMBER": 1,
        "VAT_NUMBER": 1,
    }


def test_independent_overlap_regions_are_counted_and_transformed_separately() -> None:
    planner = ActionPlanner(_policy("mask"), TestAnonymizer(), b"k" * 32, "test")
    matches = [
        EntityMatch("EMAIL_ADDRESS", 0, 3, 0.9, "a"),
        EntityMatch("PHONE_NUMBER", 2, 4, 0.9, "b"),
        EntityMatch("EMAIL_ADDRESS", 8, 11, 0.9, "a"),
        EntityMatch("PHONE_NUMBER", 10, 12, 0.9, "b"),
    ]

    result = planner.apply("abcdefghijkl", matches, "nonce")

    assert result.text == "****efgh****"
    assert result.overlap_count == 2
    assert result.entity_counts == {"EMAIL_ADDRESS": 2, "PHONE_NUMBER": 2}
    assert result.transformed_counts == {"EMAIL_ADDRESS": 2}


def test_action_registry_defines_complete_strictness_order() -> None:
    """The overlap resolver's action order is explicit and complete."""
    expected = [
        "pass",
        "mask",
        "hash",
        "encrypt",
        "reversible_replace",
        "replace",
        "redact",
        "reroute",
        "block",
    ]
    assert [ACTION_BY_NAME[name].strictness for name in expected] == list(range(1, 10))


def test_routing_targets_are_bounded_and_unambiguous() -> None:
    """Route mappings must be safe to embed in AgentGateway CEL expressions."""
    routing = RoutingSettings.model_validate(
        {
            "defaultTarget": "local/safe",
            "targets": [
                {"name": "local/safe"},
                {"name": "remote/code", "classPrefix": "code/"},
            ],
        }
    )
    assert routing.targets[1].class_prefix == "code/"
    with pytest.raises(ValueError, match="selector is duplicated"):
        RoutingSettings.model_validate(
            {
                "defaultTarget": "local/safe",
                "targets": [
                    {"name": "local/safe"},
                    {"name": "remote/code", "classPrefix": "local/safe"},
                ],
            }
        )
    with pytest.raises(ValueError, match="defaultTarget"):
        RoutingSettings.model_validate(
            {"defaultTarget": "missing", "targets": [{"name": "local/safe"}]}
        )

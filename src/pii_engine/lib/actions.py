"""Shared PII action registry derived from the legacy policy engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from pii_engine.models.contracts import ActionDescription, ActionParam


@dataclass(frozen=True)
class ActionSpec:
    """Define one policy action and its Studio metadata."""

    name: str
    decision: str
    reversible: bool
    severity: str
    strictness: int
    notes: str
    params: tuple[ActionParam, ...] = ()


ACTION_REGISTRY: tuple[ActionSpec, ...] = (
    ActionSpec("pass", "pass", False, "pass", 1, "Entity passes through unmodified."),
    ActionSpec("block", "block", False, "fail", 9, "Reject the request before an LLM is called."),
    ActionSpec(
        "reroute", "reroute", False, "warn", 8, "Route the request to a trusted local target."
    ),
    ActionSpec(
        "mask",
        "apply_actions",
        False,
        "info",
        2,
        "Replace characters while preserving length.",
        (
            ActionParam(
                name="masking_char", type="string", default="*", description="Mask character"
            ),
            ActionParam(
                name="chars_to_mask",
                type="int",
                default="100",
                description="Maximum characters to mask",
            ),
            ActionParam(
                name="from_end",
                type="bool",
                default="true",
                description="Mask from the end of the value",
            ),
        ),
    ),
    ActionSpec(
        "replace",
        "apply_actions",
        False,
        "info",
        6,
        "Replace with a configured value; <ENTITY> expands to the normalized type.",
        (
            ActionParam(
                name="new_value",
                type="string",
                default="<ENTITY>",
                description="Replacement text",
            ),
        ),
    ),
    ActionSpec("redact", "apply_actions", False, "info", 7, "Remove the detected value."),
    ActionSpec(
        "hash",
        "apply_actions",
        False,
        "info",
        3,
        "Use a keyed policy-scoped rotating-window HMAC-SHA-256 value.",
        (
            ActionParam(
                name="hash_type",
                type="enum",
                default="sha256",
                description="Keyed hash algorithm",
                options=["sha256"],
            ),
        ),
    ),
    ActionSpec(
        "encrypt", "apply_actions", True, "info", 4, "Encrypt and reverse through a trusted map."
    ),
    ActionSpec(
        "reversible_replace",
        "apply_actions",
        True,
        "info",
        5,
        "Use a reversible entity token.",
    ),
)

ACTION_BY_NAME: dict[str, ActionSpec] = {item.name: item for item in ACTION_REGISTRY}


def action_descriptions() -> list[ActionDescription]:
    """Return the registry as response models."""
    return [
        ActionDescription(
            name=item.name,
            decision=item.decision,
            reversible=item.reversible,
            severity=cast(Literal["pass", "info", "warn", "fail"], item.severity),
            strictness=item.strictness,
            params=list(item.params),
            notes=item.notes,
        )
        for item in ACTION_REGISTRY
    ]

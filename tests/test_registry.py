"""Test shared action and policy registry behavior."""

from __future__ import annotations

import httpx

from pii_engine.lib.actions import ACTION_BY_NAME


def test_action_registry_marks_reversible_actions() -> None:
    """Only actions intended for response reversal are marked reversible."""
    assert ACTION_BY_NAME["reversible_replace"].reversible is True
    assert ACTION_BY_NAME["replace"].reversible is False


async def test_actions_endpoint_exposes_registry(client: httpx.AsyncClient) -> None:
    """The HTTP registry is populated from the same source table."""
    response = await client.get("/v1/actions")
    names = {item["name"] for item in response.json()}
    assert names == set(ACTION_BY_NAME)
    assert {item["strictness"] for item in response.json()} == set(range(1, 10))


async def test_policy_endpoint_exposes_catalog_and_safety(client: httpx.AsyncClient) -> None:
    """The policy endpoint exposes normalized entities and safety names."""
    response = await client.get("/v1/policy")
    assert response.status_code == 200
    assert "EMAIL_ADDRESS" in response.json()["entities"]
    assert "promptInjection" in response.json()["safety_rules"]

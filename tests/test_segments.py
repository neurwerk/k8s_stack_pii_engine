"""Exercise provider-independent analysis and its security-sensitive scope boundaries."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from pii_engine.config.policy import PolicySettings
from pii_engine.runtime import get_runtime


def _request(*texts: str, kind: str = "chat", scope: str = "request") -> dict[str, object]:
    return {
        "api_version": "v2",
        "request_kind": kind,
        "scope": scope,
        "segments": [{"id": f"s{index}", "text": text} for index, text in enumerate(texts)],
    }


def _policy(action: str) -> dict[str, object]:
    return {"pii": {"entityPolicies": [{"entityType": "EMAIL_ADDRESS", "action": action}]}}


async def test_segment_readiness_checks_the_runtime(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = await client.get("/v2/adapter/ready")
    assert response.status_code == 200
    assert response.json() == {"api_version": "v2", "status": "ok"}
    monkeypatch.setattr(get_runtime(), "ready", AsyncMock(return_value=False))
    response = await client.get("/v2/adapter/ready")
    assert response.status_code == 503


async def test_segment_evaluation_preserves_ids_offsets_and_local_reversal(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post(
        "/v2/studio/evaluate-policy",
        json={
            "request": _request("email a@example.com", "a@example.com"),
            "policy": _policy("reversible_replace"),
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["api_version"] == "v2"
    assert [segment["id"] for segment in body["segments"]] == ["s0", "s1"]
    assert "request" not in body and "reversal" not in body
    findings = body["diagnostics"]["logical_detections"]
    assert [(item["segment_id"], item["start"], item["end"]) for item in findings] == [
        ("s0", 6, 19),
        ("s1", 0, 13),
    ]
    assert all("path" not in item for item in findings)
    assert "a@example.com" not in body["simulation"]["model_response"]
    assert body["simulation"]["user_response"].count("a@example.com") == 2
    assert body["simulation"]["restored_entity_counts"] == {"EMAIL_ADDRESS": 2}


@pytest.mark.parametrize("action", ["block", "reroute"])
async def test_mcp_terminal_policy_never_produces_model_routing(
    client: httpx.AsyncClient,
    action: str,
) -> None:
    response = await client.post(
        "/v2/studio/evaluate-policy",
        json={
            "request": _request("a@example.com", kind="mcp"),
            "policy": _policy(action),
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["decision"] == "block"
    assert body["segments"] is None
    assert body["route_class"] is None
    assert body["notices"] == {"request": [], "response": []}
    assert body["simulation"]["status"] == "skipped"


async def test_no_text_mcp_is_an_unscanned_pass(client: httpx.AsyncClient) -> None:
    response = await client.post("/v2/adapter/analyze-segments", json=_request(kind="mcp"))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["segments"] == []
    assert body["decision"] == "pass"
    assert not body["analysis"]["scan_performed"]
    assert body["reversal"] == {}


async def test_segment_schema_rejects_duplicate_ids_and_provider_payloads(
    client: httpx.AsyncClient,
) -> None:
    duplicate = _request("first", "second")
    duplicate["segments"] = [{"id": "same", "text": text} for text in ("first", "second")]
    provider = {**_request("text"), "messages": [{"role": "user", "content": "hidden"}]}
    for request in (duplicate, provider):
        response = await client.post("/v2/adapter/analyze-segments", json=request)
        assert response.status_code == 400, response.text


@pytest.mark.parametrize("path", ["/v2/studio/analyze-segments", "/v2/studio/evaluate-policy"])
async def test_studio_cannot_use_adapter_session_or_visual_controls(
    client: httpx.AsyncClient,
    path: str,
) -> None:
    for request in (
        _request("text", scope="session"),
        {**_request("text"), "text_pii_enabled": False},
        {**_request("text"), "visual_findings": {"faces": {"scan_status": "complete", "count": 0}}},
    ):
        response = await client.post(path, json={"request": request})
        assert response.status_code == 400, response.text


async def test_document_scope_ignores_session_and_keeps_aliases_request_local(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = get_runtime()
    raw = runtime.policy_settings.model_dump(by_alias=True)
    raw["pii"]["entityPolicies"][0]["action"] = "reversible_replace"
    runtime.policy_settings = PolicySettings.model_validate(raw)
    runtime.policy = runtime._policy_service(runtime.policy_settings)
    session = AsyncMock()
    session.healthy.return_value = True
    session.get.return_value = None
    monkeypatch.setattr(runtime, "session", session)
    key = {"x-pii-session-key": "a" * 64}
    endpoint = "/v2/adapter/analyze-segments"
    document = _request("a@example.com")
    first = await client.post(endpoint, json=document, headers=key)
    second = await client.post(endpoint, json=document, headers=key)
    assert first.status_code == second.status_code == 200
    assert first.json()["reversal"] != second.json()["reversal"]
    session.get.assert_not_awaited()
    session.set.assert_not_awaited()
    conversation = _request("a@example.com", scope="session")
    first = await client.post(endpoint, json=conversation, headers=key)
    second = await client.post(endpoint, json=conversation, headers=key)
    assert first.status_code == second.status_code == 200
    assert first.json()["reversal"] == second.json()["reversal"]
    assert session.get.await_count == 2


async def test_visual_only_analysis_preserves_text_and_blocks_failed_inspection(
    client: httpx.AsyncClient,
) -> None:
    request = {
        **_request("a@example.com"),
        "text_pii_enabled": False,
        "visual_findings": {"faces": {"scan_status": "complete", "count": 0}},
    }
    response = await client.post("/v2/adapter/analyze-segments", json=request)
    assert response.status_code == 200, response.text
    assert response.json()["segments"] == request["segments"]
    assert not response.json()["analysis"]["scan_performed"]
    request["visual_findings"] = {"faces": {"scan_status": "failed", "count": None}}
    response = await client.post("/v2/adapter/analyze-segments", json=request)
    assert response.status_code == 200, response.text
    assert response.json()["decision"] == "block"
    assert response.json()["segments"] is None
    assert response.json()["reversal"] == {}


async def test_segment_count_limit_reports_the_actual_count(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v2/adapter/analyze-segments",
        json=_request(*(["text"] * 257)),
    )
    assert response.status_code == 413, response.text
    assert response.json()["error"]["limit"] == {
        "component": "pii_engine",
        "stage": "inspection",
        "reason": "segments",
        "measured": 257,
        "maximum": 256,
        "unit": "items",
        "exact": True,
    }


async def test_segment_character_limit_counts_all_segments(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(get_runtime().settings, "max_text_characters", 5)
    response = await client.post(
        "/v2/adapter/analyze-segments",
        json=_request("abc", "def"),
    )
    assert response.status_code == 413, response.text
    assert response.json()["error"]["limit"] == {
        "component": "pii_engine",
        "stage": "inspection",
        "reason": "text_characters",
        "measured": 6,
        "maximum": 5,
        "unit": "characters",
        "exact": True,
    }


async def test_segment_output_limit_reports_serialized_bytes_without_content(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(get_runtime().settings, "max_adapter_response_bytes", 1_024)
    marker = "private-marker-" * 100
    response = await client.post("/v2/adapter/analyze-segments", json=_request(marker))
    assert response.status_code == 413, response.text
    limit = response.json()["error"]["limit"]
    assert limit["stage"] == "engine_response"
    assert limit["measured"] > limit["maximum"] == 1_024
    assert limit["unit"] == "bytes" and limit["exact"] is True
    assert marker not in response.text

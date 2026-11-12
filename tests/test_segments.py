"""Exercise provider-independent analysis and its security-sensitive scope boundaries."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

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
        {**_request("text"), "visual_findings": {"faces": {"scan_status": "complete", "count": 0}}},
    ):
        response = await client.post(path, json={"request": request})
        assert response.status_code == 400, response.text


def _source_rule(source: str, action: str, route: str | None = None) -> dict[str, object]:
    rule: dict[str, object] = {"entityType": source, "action": action}
    if route is not None:
        rule["routeClass"] = route
    return rule


@pytest.mark.parametrize(
    ("model", "model_rules", "global_rules", "sources", "decision"),
    [
        ("approved/a", [("A", "pass"), ("DEFAULT", "block")], [("A", "block")], ["A"], "pass"),
        ("approved/a", [("DEFAULT", "pass")], [("A", "reroute")], ["A"], "pass"),
        ("approved/a", [], [("A", "pass"), ("DEFAULT", "reroute")], ["A"], "pass"),
        ("approved/a", [], [("DEFAULT", "reroute")], ["A"], "reroute"),
        ("approved/a", [], [], ["A"], "block"),
        (None, [("DEFAULT", "pass")], [("DEFAULT", "reroute")], ["A"], "reroute"),
        ("local/unknown", [("DEFAULT", "pass")], [("DEFAULT", "block")], ["A"], "block"),
        ("upstream-alias", [("DEFAULT", "pass")], [], ["A"], "block"),
        ("approved/a", [("A", "pass")], [("DEFAULT", "block")], ["A", "B"], "block"),
    ],
)
async def test_explicit_model_source_precedence(
    client, model, model_rules, global_rules, sources, decision
):
    response = await client.post(
        "/v2/studio/evaluate-policy",
        json={
            "request": {
                **_request("unchanged"),
                "text_pii_enabled": False,
                "destination_model": model,
                "mcp_sources": [f"MCP_TOOL_{source}" for source in sources],
            },
            "policy": {
                "pii": {
                    "entityPolicies": [_source_rule(f"MCP_TOOL_{s}", a) for s, a in global_rules],
                    "modelSourcePolicies": {
                        "approved/a": [_source_rule(f"MCP_TOOL_{s}", a) for s, a in model_rules]
                    },
                }
            },
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["decision"] == decision


@pytest.mark.parametrize("face", [False, True])
async def test_model_default_pass_keeps_text_and_face_blocks(client, face):
    runtime = get_runtime()
    raw = runtime.policy_settings.model_dump(by_alias=True)
    raw["pii"]["modelSourcePolicies"] = {"approved/a": [_source_rule("MCP_TOOL_DEFAULT", "pass")]}
    raw["pii"]["entityPolicies"] = [{"entityType": "EMAIL_ADDRESS", "action": "block"}]
    runtime.policy_settings = PolicySettings.model_validate(raw)
    runtime.policy = runtime._policy_service(runtime.policy_settings)
    request = {
        **_request("a@example.com"),
        "destination_model": "approved/a",
        "mcp_sources": ["MCP_TOOL_FUTURE"],
        "text_pii_enabled": not face,
    }
    if face:
        request["visual_findings"] = {"faces": {"scan_status": "complete", "count": 1}}
    response = await client.post("/v2/adapter/analyze-segments", json=request)
    assert response.status_code == 200, response.text
    assert response.json()["decision"] == "block"


@pytest.mark.parametrize("model", ["", "a" * 129, "local/*", "a b", 42])
async def test_destination_model_validation(client, model):
    response = await client.post(
        "/v2/adapter/analyze-segments", json={**_request("text"), "destination_model": model}
    )
    assert response.status_code == 400


async def test_mcp_operation_rejects_destination_model(client):
    response = await client.post(
        "/v2/adapter/analyze-segments",
        json={**_request("text", kind="mcp"), "destination_model": "approved/a"},
    )
    assert response.status_code == 400


@pytest.mark.parametrize(
    ("rules", "sources", "decision", "route"),
    [
        ([], ["MCP_TOOL_UNKNOWN"], "block", None),
        ([], ["MCP_TOOL_DEFAULT"], "block", None),
        ([], [], "pass", None),
        ([_source_rule("MCP_TOOL_DEFAULT", "pass")], ["MCP_TOOL_UNKNOWN"], "pass", None),
        ([_source_rule("MCP_TOOL_DEFAULT", "reroute")], ["MCP_TOOL_UNKNOWN"], "reroute", "local"),
        (
            [_source_rule("MCP_TOOL_DEFAULT", "block"), _source_rule("MCP_TOOL_A", "pass")],
            ["MCP_TOOL_A"],
            "pass",
            None,
        ),
        (
            [_source_rule("MCP_TOOL_DEFAULT", "pass"), _source_rule("MCP_TOOL_A", "block")],
            ["MCP_TOOL_A"],
            "block",
            None,
        ),
    ],
)
async def test_source_only_http_uses_shared_fallback_without_text_analysis(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    rules: list[dict[str, object]],
    sources: list[str],
    decision: str,
    route: str | None,
) -> None:
    runtime = get_runtime()
    analyzer = Mock(side_effect=AssertionError("text analyzer must not run"))
    monkeypatch.setattr(runtime._analyzer, "analyze", analyzer)
    for kind in ("chat", "responses"):
        request = {
            **_request("a@example.com", kind=kind),
            "text_pii_enabled": False,
            "mcp_sources": sources,
        }
        response = await client.post(
            "/v2/studio/evaluate-policy",
            json={"request": request, "policy": {"pii": {"entityPolicies": rules}}},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["decision"] == decision
        assert body["route_class"] == route
        assert body["analysis"]["scan_performed"] is False
        assert body["analysis"]["duration_ms"] is None
        assert body["analysis"]["source"] == "current_request"
        assert body["segments"] == (None if decision == "block" else request["segments"])
        assert body["entity_counts"] == dict.fromkeys(sources, 1)
        assert body["diagnostics"]["logical_detections"] == []
        assert body["diagnostics"]["effective_regions"] == []
        assert "reversal" not in body
        for row in body["report"]["rows"]:
            assert row["transformed_count"] == row["unique_transformed_count"] == 0
    analyzer.assert_not_called()


@pytest.mark.parametrize("action", ["pass", "reroute"])
@pytest.mark.parametrize("caller", ["adapter", "studio"])
async def test_source_only_success_uses_existing_analysis_endpoints(
    client: httpx.AsyncClient, action: str, caller: str
) -> None:
    runtime = get_runtime()
    raw = runtime.policy_settings.model_dump(by_alias=True)
    raw["pii"]["entityPolicies"] = [_source_rule("MCP_TOOL_A", action)]
    runtime.policy_settings = PolicySettings.model_validate(raw)
    runtime.policy = runtime._policy_service(runtime.policy_settings)
    request = {
        **_request("a@example.com"),
        "text_pii_enabled": False,
        "mcp_sources": ["MCP_TOOL_A"],
    }
    response = await client.post(
        f"/v2/{caller}/analyze-segments",
        json=request if caller == "adapter" else {"request": request},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["decision"] == action
    assert body["segments"] == request["segments"]
    assert body["entity_counts"] == {"MCP_TOOL_A": 1}
    assert body["analysis"]["scan_performed"] is False
    assert body["analysis"]["duration_ms"] is None
    assert body["route_class"] == ("local" if action == "reroute" else None)
    if caller == "adapter":
        assert body["reversal"] == {}
        assert body["report"]["rows"][0]["action"] == action
    else:
        assert "reversal" not in body


@pytest.mark.parametrize("action", ["mask", "block"])
async def test_source_pass_does_not_disable_ordinary_pii(
    client: httpx.AsyncClient, action: str
) -> None:
    response = await client.post(
        "/v2/studio/evaluate-policy",
        json={
            "request": {**_request("a@example.com"), "mcp_sources": ["MCP_TOOL_A"]},
            "policy": {
                "pii": {
                    "entityPolicies": [
                        _source_rule("MCP_TOOL_A", "pass"),
                        _source_rule("MCP_TOOL_DEFAULT", "block"),
                        {"entityType": "EMAIL_ADDRESS", "action": action},
                    ]
                }
            },
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["decision"] == ("block" if action == "block" else "apply_actions")
    assert body["analysis"]["scan_performed"]
    assert body["entity_counts"] == {"EMAIL_ADDRESS": 1, "MCP_TOOL_A": 1}
    rows = {row["entity_type"]: row for row in body["report"]["rows"]}
    assert rows["EMAIL_ADDRESS"]["action"] == action
    assert rows["MCP_TOOL_A"]["action"] == "pass"
    assert rows["MCP_TOOL_A"]["detected_count"] == 1
    assert rows["MCP_TOOL_A"]["transformed_count"] == 0
    assert rows["MCP_TOOL_A"]["unique_transformed_count"] == 0
    if action == "mask":
        assert "a@example.com" not in body["segments"][0]["text"]
    else:
        assert body["remote_allowed"] is False
        assert body["segments"] is None
        assert all(
            row["transformed_count"] == row["unique_transformed_count"] == 0
            for row in rows.values()
        )


@pytest.mark.parametrize("other", ["source", "text", "face"])
@pytest.mark.parametrize("target", ["local-a", "local-b"])
async def test_sources_combine_exact_required_routes_before_transformations(
    client: httpx.AsyncClient,
    other: str,
    target: str,
) -> None:
    runtime = get_runtime()
    raw = runtime.policy_settings.model_dump(by_alias=True)
    rules = [_source_rule("MCP_TOOL_A", "reroute", "local-a")]
    request = {**_request("a@example.com"), "mcp_sources": ["MCP_TOOL_A"]}
    if other == "source":
        rules.append(_source_rule("MCP_TOOL_B", "reroute", target))
        request["mcp_sources"] = ["MCP_TOOL_B", "MCP_TOOL_A", "MCP_TOOL_A"]
    elif other == "text":
        rules.append({"entityType": "EMAIL_ADDRESS", "action": "reroute", "routeClass": target})
    else:
        request["visual_findings"] = {"faces": {"scan_status": "complete", "count": 1}}
        raw["attachments"]["faces"] = {"action": "reroute", "routeClass": target}
        request["text_pii_enabled"] = False
    raw["pii"]["entityPolicies"] = rules
    runtime.policy_settings = PolicySettings.model_validate(raw)
    runtime.policy = runtime._policy_service(runtime.policy_settings)
    response = await client.post("/v2/adapter/analyze-segments", json=request)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["decision"] == ("reroute" if target == "local-a" else "block")
    assert body["route_class"] == ("local-a" if target == "local-a" else None)
    assert body["entity_counts"]["MCP_TOOL_A"] == 1
    if target == "local-b":
        assert body["segments"] is None
        assert body["reversal"] == {}
        assert all(row["transformed_count"] == 0 for row in body["report"]["rows"])
    elif other == "face":
        assert body["segments"] == request["segments"]


@pytest.mark.parametrize("visual", [False, True])
async def test_document_scope_source_block_is_independent_of_text_scanning(
    client: httpx.AsyncClient, visual: bool
) -> None:
    request = {
        **_request("converted document text", kind="responses"),
        "text_pii_enabled": False,
        "mcp_sources": ["MCP_TOOL_REMOVED"],
    }
    if visual:
        request["visual_findings"] = {"faces": {"scan_status": "complete", "count": 0}}
    response = await client.post("/v2/adapter/analyze-segments", json=request)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["decision"] == "block"
    assert body["entity_counts"] == {"MCP_TOOL_REMOVED": 1}
    assert body["analysis"]["cached_decision_applied"] is False
    assert body["analysis"]["scan_performed"] is False


@pytest.mark.parametrize(
    "sources",
    [
        ["MCP_TOOL_"],
        ["MCP_TOOL_a"],
        ["EMAIL_ADDRESS"],
        ["MCP_TOOL_" + "A" * 56],
        ["MCP_TOOL_A"] * 33,
        [1],
    ],
)
async def test_source_boundary_rejects_invalid_or_excessive_inputs(
    client: httpx.AsyncClient, sources: list[object]
) -> None:
    response = await client.post(
        "/v2/adapter/analyze-segments",
        json={**_request("text"), "mcp_sources": sources},
    )
    assert response.status_code == 400, response.text


async def test_source_boundary_normalizes_union_and_rejects_mcp_invocations(
    client: httpx.AsyncClient,
) -> None:
    request = {**_request("text"), "mcp_sources": ["MCP_TOOL_B", "MCP_TOOL_A", "MCP_TOOL_B"]}
    response = await client.post("/v2/adapter/analyze-segments", json=request)
    assert response.status_code == 200, response.text
    assert response.json()["entities"] == ["MCP_TOOL_A", "MCP_TOOL_B"]
    assert response.json()["entity_counts"] == {"MCP_TOOL_A": 1, "MCP_TOOL_B": 1}
    request["request_kind"] = "mcp"
    response = await client.post("/v2/adapter/analyze-segments", json=request)
    assert response.status_code == 400, response.text


async def test_combined_source_and_custom_entity_overflow_rejects_without_truncation(
    client: httpx.AsyncClient,
) -> None:
    sources = [f"MCP_TOOL_{index}" for index in range(32)]
    response = await client.post(
        "/v2/studio/evaluate-policy",
        json={
            "request": {**_request("trigger"), "mcp_sources": sources},
            "policy": {
                "pii": {
                    "entityPolicies": [
                        _source_rule("MCP_TOOL_DEFAULT", "pass"),
                        *[
                            {
                                "entityType": f"CUSTOM_{index}",
                                "action": "pass",
                                "patterns": ["trigger"],
                            }
                            for index in range(33)
                        ],
                    ]
                }
            },
        },
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "invalid_request"


async def test_source_only_request_accepts_exact_source_bounds_without_text(
    client: httpx.AsyncClient,
) -> None:
    sources = ["MCP_TOOL_" + "A" * 55, *[f"MCP_TOOL_{index}" for index in range(31)]]
    response = await client.post(
        "/v2/studio/evaluate-policy",
        json={
            "request": {**_request(), "text_pii_enabled": False, "mcp_sources": sources},
            "policy": {"pii": {"entityPolicies": [_source_rule("MCP_TOOL_DEFAULT", "pass")]}},
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["decision"] == "pass"
    assert body["segments"] == []
    assert body["entity_counts"] == dict.fromkeys(sources, 1)
    assert body["analysis"]["scan_performed"] is False


@pytest.mark.parametrize("other_target", ["local-a", "local-b"])
async def test_overlapping_text_reroutes_require_one_canonical_target(
    client: httpx.AsyncClient, other_target: str
) -> None:
    response = await client.post(
        "/v2/studio/evaluate-policy",
        json={
            "request": _request("a@example.com"),
            "policy": {
                "pii": {
                    "entityPolicies": [
                        {
                            "entityType": "EMAIL_ADDRESS",
                            "action": "reroute",
                            "routeClass": "local-a",
                        },
                        {
                            "entityType": "SENSITIVE_TEXT",
                            "action": "reroute",
                            "routeClass": other_target,
                            "patterns": [r"a@example\.com"],
                        },
                    ]
                }
            },
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["decision"] == ("reroute" if other_target == "local-a" else "block")
    assert body["route_class"] == ("local-a" if other_target == "local-a" else None)
    if other_target == "local-b":
        assert body["segments"] is None
        assert all(row["transformed_count"] == 0 for row in body["report"]["rows"])


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


@pytest.mark.parametrize("maximum", [2_000, 2_100])
async def test_configured_segment_limit_accepts_and_scans_every_leaf(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch, maximum: int
) -> None:
    monkeypatch.setattr(get_runtime().settings, "max_text_leaves", maximum)
    request = _request(*(["text"] * (maximum - 1)), "a@example.com")
    response = await client.post("/v2/adapter/analyze-segments", json=request)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["analysis"]["text_leaf_count"] == maximum
    assert len(body["segments"]) == maximum
    assert body["segments"][-1]["id"] == f"s{maximum - 1}"
    assert body["segments"][-1]["text"] != "a@example.com"
    assert body["entity_counts"]["EMAIL_ADDRESS"] == 1


@pytest.mark.parametrize("maximum", [2_000, 2_100])
async def test_segment_count_limit_reports_the_actual_count(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch, maximum: int
) -> None:
    monkeypatch.setattr(get_runtime().settings, "max_text_leaves", maximum)
    response = await client.post(
        "/v2/adapter/analyze-segments",
        json=_request(*(["text"] * (maximum + 1))),
    )
    assert response.status_code == 413, response.text
    assert response.json()["error"]["limit"] == {
        "component": "pii_engine",
        "stage": "inspection",
        "reason": "segments",
        "measured": maximum + 1,
        "maximum": maximum,
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

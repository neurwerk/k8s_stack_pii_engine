"""Protect legacy request shapes while the runtime uses canonical segments."""

import httpx
import pytest


@pytest.mark.parametrize(
    "endpoint",
    ["adapter/analyze-request", "studio/analyze-request", "studio/evaluate-policy"],
)
@pytest.mark.parametrize(
    "sample",
    [
        {"model": "test", "messages": [{"role": "user", "content": "hello"}]},
        {"model": "test", "input": "hello", "stream": False, "previous_response_id": None},
    ],
)
async def test_legacy_response_preserves_provider_omissions(
    client: httpx.AsyncClient, endpoint, sample
):
    body = sample if endpoint.startswith("adapter/") else {"request": sample}
    response = await client.post(f"/v1/{endpoint}", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["request"] == sample
    assert "safety_rule" in response.json()


async def test_legacy_responses_tools_keep_envelopes_and_diagnostic_paths(
    client: httpx.AsyncClient,
):
    sample = {
        "model": "test",
        "input": "hello",
        "tools": [
            {"type": "function", "function": {"name": "legacy", "description": "a@example.com"}},
            {"type": "function", "name": "current", "description": "b@example.com"},
        ],
    }
    response = await client.post("/v1/studio/evaluate-policy", json={"request": sample})
    assert response.status_code == 200, response.text
    result = response.json()
    tools = result["request"]["tools"]
    assert tools == [
        {"type": "function", "function": {"name": "legacy", "description": "*************"}},
        {"type": "function", "name": "current", "description": "*************"},
    ]
    assert [item["path"] for item in result["diagnostics"]["logical_detections"]] == [
        ["tools", 0, "function", "description"],
        ["tools", 1, "description"],
    ]

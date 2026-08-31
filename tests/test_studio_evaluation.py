"""Test the bounded Studio-only policy evaluation contract."""

from __future__ import annotations

import json
import socket

import httpx
import pytest

from pii_engine.config.policy import PolicySettings
from pii_engine.models.contracts import OpenAIChatRequest
from pii_engine.runtime import PolicyEvaluationResult, get_runtime
from pii_engine.services.analyzer import EntityMatch


def _request(content: str = "email a@example.com") -> dict[str, object]:
    return {"model": "test", "messages": [{"role": "user", "content": content}]}


async def test_evaluation_uses_strict_studio_contract_and_deployed_policy(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post("/v1/studio/evaluate-policy", json={"request": _request()})

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["issues"] == []
    assert body["issues_truncated"] is False
    assert body["report"]["rows"][0]["entity_type"] == "EMAIL_ADDRESS"
    assert body["analysis"]["source"] == "current_request"
    assert body["analysis"]["duration_ms"] is not None
    assert body["simulation"]["type"] == "deterministic_echo"
    assert body["simulation"]["model_called"] is False
    assert body["simulation"]["model_response"].startswith("[SIMULATED - NO MODEL CALLED]\n")
    assert "reversal" not in body


@pytest.mark.parametrize(
    "policy",
    [
        {"pii": {"entityPolicies": [{"entityType": "EMAIL_ADDRESS", "action": "invalid"}]}},
        {"pii": {"analyzerLanguages": ["fr"]}},
        {"safety": {"enabled": ["unknownRule"]}},
    ],
)
async def test_candidate_failures_are_sanitized_evaluation_results(
    client: httpx.AsyncClient,
    policy: dict[str, object],
) -> None:
    marker = "rejected-private-marker"
    policy[marker] = marker
    response = await client.post(
        "/v1/studio/evaluate-policy", json={"request": _request(), "policy": policy}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert body["issues"]
    assert all(issue["stage"] == "schema" for issue in body["issues"])
    assert marker not in response.text
    assert set(body) == {"api_version", "valid", "issues", "issues_truncated"}


async def test_merge_and_compile_failures_have_their_own_stages(
    client: httpx.AsyncClient,
) -> None:
    merge = await client.post(
        "/v1/studio/evaluate-policy",
        json={"request": _request(), "policy": {"pii": {"analyzerLanguages": ["fr"]}}},
    )
    compile_response = await client.post(
        "/v1/studio/evaluate-policy",
        json={"request": _request(), "policy": {"safety": {"enabled": ["unknownRule"]}}},
    )

    assert merge.json()["issues"][0]["stage"] == "merge"
    assert compile_response.json()["issues"][0] == {
        "stage": "compile",
        "path": [],
        "code": "policy_compile_failed",
        "message": "Policy compilation failed.",
    }


async def test_policy_candidate_rejects_model_and_process_configuration(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post(
        "/v1/studio/evaluate-policy",
        json={
            "request": _request(),
            "policy": {
                "pii": {"ner": {"strategy": "multilingual"}, "customRecognizers": []},
                "tls_key": "private-marker",
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["valid"] is False
    assert "private-marker" not in response.text
    assert all(issue["message"] == "Field is not allowed." for issue in response.json()["issues"])


async def test_request_protocol_errors_remain_typed_400(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/studio/evaluate-policy",
        json={"request": _request(), "simulation": "call_a_model"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


async def test_overlap_diagnostics_preserve_logical_unicode_offsets_and_effective_winner(
    client: httpx.AsyncClient,
) -> None:
    class OverlapAnalyzer:
        def analyze(self, text: str, policy: PolicySettings | None = None) -> list[EntityMatch]:
            assert text == "A😀BCDEF"
            return [
                EntityMatch("EMAIL_ADDRESS", 1, 4, 0.8, "private-recognizer-name"),
                EntityMatch("PHONE_NUMBER", 3, 6, 0.9, "spacy"),
            ]

    get_runtime()._analyzer = OverlapAnalyzer()
    response = await client.post(
        "/v1/studio/evaluate-policy",
        json={
            "request": _request("A😀BCDEF"),
            "policy": {
                "pii": {
                    "entityPolicies": [
                        {"entityType": "EMAIL_ADDRESS", "action": "mask"},
                        {"entityType": "PHONE_NUMBER", "action": "redact"},
                    ]
                }
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["diagnostics"]["logical_detections"] == [
        {
            "path": ["messages", 0, "content"],
            "start": 1,
            "end": 4,
            "entity_type": "EMAIL_ADDRESS",
            "score": 0.8,
            "source": "deterministic",
            "configured_action": "mask",
            "resolved_action": "redact",
        },
        {
            "path": ["messages", 0, "content"],
            "start": 3,
            "end": 6,
            "entity_type": "PHONE_NUMBER",
            "score": 0.9,
            "source": "spacy",
            "configured_action": "redact",
            "resolved_action": "redact",
        },
    ]
    assert body["diagnostics"]["effective_regions"] == [
        {
            "path": ["messages", 0, "content"],
            "start": 1,
            "end": 6,
            "entity_type": "PHONE_NUMBER",
            "action": "redact",
            "source": "spacy",
            "score": 0.9,
            "member_entity_types": ["EMAIL_ADDRESS", "PHONE_NUMBER"],
            "overlap": True,
        }
    ]
    assert body["request"]["messages"][0]["content"] == "AF"


async def test_stringified_json_diagnostic_path_is_public(client: httpx.AsyncClient) -> None:
    request = {
        "model": "test",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"query":"a@example.com"}',
                        },
                    }
                ],
            }
        ],
    }
    response = await client.post("/v1/studio/evaluate-policy", json={"request": request})

    path = response.json()["diagnostics"]["logical_detections"][0]["path"]
    assert path == ["messages", 0, "tool_calls", 0, "function", "arguments", "query"]
    assert "PII_ENGINE_JSON_STRING" not in response.text


async def test_pii_block_retains_findings_and_skips_simulation(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/studio/evaluate-policy",
        json={
            "request": _request(),
            "policy": {
                "pii": {"entityPolicies": [{"entityType": "EMAIL_ADDRESS", "action": "block"}]}
            },
        },
    )

    body = response.json()
    assert body["decision"] == "block"
    assert body["request"] is None
    assert body["report"]["rows"][0]["action"] == "block"
    assert len(body["diagnostics"]["logical_detections"]) == 1
    assert body["simulation"] == {
        "type": "deterministic_echo",
        "status": "skipped",
        "reason": "request_blocked",
        "model_called": False,
        "model_response": None,
        "user_response": None,
        "restored_entity_counts": {},
    }


async def test_preflight_block_may_have_no_pii_findings(client: httpx.AsyncClient) -> None:
    request = {
        "model": "test",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a@example.com"},
                    {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
                ],
            }
        ],
    }
    response = await client.post("/v1/studio/evaluate-policy", json={"request": request})

    assert response.json()["decision"] == "block"
    assert response.json()["diagnostics"] == {
        "logical_detections": [],
        "effective_regions": [],
        "truncated": False,
    }


async def test_simulation_restores_only_authoritative_reversible_values_without_network(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", no_network)
    response = await client.post(
        "/v1/studio/evaluate-policy",
        json={
            "request": _request(),
            "policy": {
                "pii": {
                    "entityPolicies": [
                        {"entityType": "EMAIL_ADDRESS", "action": "reversible_replace"}
                    ]
                }
            },
        },
    )

    simulation = response.json()["simulation"]
    assert "<REV_EMAIL_ADDRESS_" in simulation["model_response"]
    assert "a@example.com" not in simulation["model_response"]
    assert "a@example.com" in simulation["user_response"]
    assert simulation["restored_entity_counts"] == {"EMAIL_ADDRESS": 1}
    assert "reversal" not in json.dumps(response.json()).lower()


async def test_nonreversible_simulation_does_not_restore_detected_values(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post(
        "/v1/studio/evaluate-policy",
        json={
            "request": _request(),
            "policy": {
                "pii": {"entityPolicies": [{"entityType": "EMAIL_ADDRESS", "action": "hash"}]}
            },
        },
    )

    simulation = response.json()["simulation"]
    assert simulation["model_response"] == simulation["user_response"]
    assert "a@example.com" not in simulation["user_response"]
    assert simulation["restored_entity_counts"] == {}


async def test_diagnostics_and_issues_truncate_deterministically(
    client: httpx.AsyncClient,
) -> None:
    class DenseAnalyzer:
        def analyze(self, text: str, policy: PolicySettings | None = None) -> list[EntityMatch]:
            return [
                EntityMatch("EMAIL_ADDRESS", index, index + 1, 0.9, "deterministic")
                for index in range(0, len(text), 2)
            ]

    runtime = get_runtime()
    runtime.policy.analyzer = DenseAnalyzer()
    dense = await client.post(
        "/v1/studio/evaluate-policy", json={"request": _request("x " * 2_050)}
    )
    invalid = await client.post(
        "/v1/studio/evaluate-policy",
        json={
            "request": _request(),
            "policy": {"pii": {"entityPolicies": [{"private": "value"}] * 64}},
        },
    )

    diagnostics = dense.json()["diagnostics"]
    assert diagnostics["truncated"] is True
    assert len(diagnostics["logical_detections"]) == 2_048
    assert len(diagnostics["effective_regions"]) == 2_048
    assert diagnostics["logical_detections"][-1]["start"] == 4_094
    assert invalid.json()["issues_truncated"] is True
    assert len(invalid.json()["issues"]) == 128
    assert "private" not in invalid.text
    assert "value" not in invalid.text


async def test_serialized_evaluation_response_has_exact_byte_budget(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = get_runtime()
    request = OpenAIChatRequest.model_validate(_request("hello"))
    result = runtime.policy.analyze(request)
    result.duration_ms = 0
    evaluation = PolicyEvaluationResult(result=result, issues=[])

    async def evaluate(*_args: object, **_kwargs: object) -> PolicyEvaluationResult:
        return evaluation

    monkeypatch.setattr(runtime, "evaluate_policy", evaluate)
    first = await client.post("/v1/studio/evaluate-policy", json={"request": _request("hello")})
    exact_size = len(first.content)
    runtime.settings.max_studio_evaluation_response_bytes = exact_size
    exact = await client.post("/v1/studio/evaluate-policy", json={"request": _request("hello")})
    runtime.settings.max_studio_evaluation_response_bytes = exact_size - 1
    over = await client.post("/v1/studio/evaluate-policy", json={"request": _request("hello")})

    assert exact.status_code == 200
    assert len(exact.content) == exact_size
    assert over.status_code == 413
    assert over.json()["error"]["code"] == "request_too_large"

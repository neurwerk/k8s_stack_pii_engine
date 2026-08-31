"""Exercise coordinated request, semantic text, and adapter response limits."""

from __future__ import annotations

import json
import logging
from collections import deque
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from starlette.types import Message, Scope

from pii_engine.config.policy import test_policy as make_test_policy
from pii_engine.config.settings import Settings
from pii_engine.main import RequestSizeLimitMiddleware
from pii_engine.models.contracts import (
    AdapterAnalyzeResponse,
    AnalysisMetadata,
    McpRequest,
    OpenAIChatRequest,
    OpenAIResponsesRequest,
    PIIReportRow,
    ResponseTextPart,
    TextPart,
)
from pii_engine.runtime import EngineRuntime, get_runtime
from pii_engine.services.errors import AnalysisRequestTooLargeError, InvalidAnalysisRequestError
from pii_engine.services.policy import PolicyResult
from pii_engine.services.traversal import (
    TextLeaf,
    replace_text_leaves,
    validate_request_structure,
)

MAX_REQUEST_BYTES = 5_242_880
MAX_TEXT_CHARACTERS = 4_000_000
MAX_ADAPTER_RESPONSE_BYTES = 10_485_760
MAX_STUDIO_EVALUATION_RESPONSE_BYTES = 10_485_760
PLACEHOLDER = "<REV_EMAIL_ADDRESS_0123456789abcdef_fedcba9876543210>"


def test_limit_settings_have_exact_defaults_and_hard_maximums() -> None:
    settings = Settings(allow_test_analyzer=True)
    assert settings.max_request_bytes == MAX_REQUEST_BYTES
    assert settings.max_adapter_response_bytes == MAX_ADAPTER_RESPONSE_BYTES
    assert settings.max_studio_evaluation_response_bytes == MAX_STUDIO_EVALUATION_RESPONSE_BYTES
    assert settings.max_text_characters == MAX_TEXT_CHARACTERS
    assert settings.max_text_leaves == 256
    assert settings.analysis_timeout == 600
    assert settings.studio_analysis_timeout == 30
    assert make_test_policy().pii.timeout == 600

    assert Settings(allow_test_analyzer=True, max_request_bytes=1_024).max_request_bytes == 1_024
    assert (
        Settings(
            allow_test_analyzer=True, max_adapter_response_bytes=1_024
        ).max_adapter_response_bytes
        == 1_024
    )
    assert (
        Settings(
            allow_test_analyzer=True, max_studio_evaluation_response_bytes=1_024
        ).max_studio_evaluation_response_bytes
        == 1_024
    )
    assert Settings(allow_test_analyzer=True, max_text_characters=1).max_text_characters == 1
    for values in (
        {"max_request_bytes": MAX_REQUEST_BYTES + 1},
        {"max_adapter_response_bytes": MAX_ADAPTER_RESPONSE_BYTES + 1},
        {"max_studio_evaluation_response_bytes": MAX_STUDIO_EVALUATION_RESPONSE_BYTES + 1},
        {"max_text_characters": MAX_TEXT_CHARACTERS + 1},
        {"max_text_leaves": 257},
        {"analysis_timeout": 601},
        {"studio_analysis_timeout": 31},
    ):
        with pytest.raises(ValidationError):
            Settings.model_validate({"allow_test_analyzer": True, **values})

    policy_data = make_test_policy().model_dump(mode="json", by_alias=True)
    policy_data["pii"]["timeout"] = 601
    with pytest.raises(ValidationError):
        type(make_test_policy()).model_validate(policy_data)


async def _run_size_middleware(chunks: list[bytes]) -> tuple[list[Message], bytes | None]:
    received = deque(
        {
            "type": "http.request",
            "body": chunk,
            "more_body": index < len(chunks) - 1,
        }
        for index, chunk in enumerate(chunks)
    )
    forwarded: bytes | None = None

    async def receive() -> Message:
        return received.popleft()

    async def app(_scope: Scope, app_receive: Any, send: Any) -> None:
        nonlocal forwarded
        forwarded = (await app_receive()).get("body", b"")
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/adapter/analyze-request",
        "raw_path": b"/v1/adapter/analyze-request",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("test", 1),
        "server": ("test", 80),
        "state": {},
    }
    await RequestSizeLimitMiddleware(app, MAX_REQUEST_BYTES)(scope, receive, send)
    return sent, forwarded


async def test_request_body_accepts_exact_5_mib_and_rejects_one_more_byte() -> None:
    exact_messages, forwarded = await _run_size_middleware([b"x" * MAX_REQUEST_BYTES])
    assert exact_messages[0]["status"] == 204
    assert forwarded is not None and len(forwarded) == MAX_REQUEST_BYTES

    over_messages, forwarded = await _run_size_middleware([b"x" * (MAX_REQUEST_BYTES + 1)])
    assert over_messages[0]["status"] == 413
    assert forwarded is None
    body = json.loads(over_messages[1]["body"])
    assert body["error"]["code"] == "request_too_large"


async def test_request_chunk_count_does_not_reject_reasonable_nonempty_chunks() -> None:
    messages, forwarded = await _run_size_middleware([b"x"] * 5_000)
    assert messages[0]["status"] == 204
    assert forwarded == b"x" * 5_000


async def test_request_chunk_limit_still_rejects_endless_empty_chunks() -> None:
    messages, forwarded = await _run_size_middleware([b""] * 4_098)
    assert messages[0]["status"] == 413
    assert forwarded is None


def test_semantic_aggregate_accepts_4m_characters_and_rejects_one_more() -> None:
    runtime = EngineRuntime(Settings(allow_test_analyzer=True))
    runtime.policy._validate_bounds([TextLeaf(("input",), "x" * MAX_TEXT_CHARACTERS)])
    with pytest.raises(AnalysisRequestTooLargeError):
        runtime.policy._validate_bounds([TextLeaf(("input",), "x" * (MAX_TEXT_CHARACTERS + 1))])


@pytest.mark.parametrize(
    "factory",
    [
        lambda text: TextPart(type="text", text=text),
        lambda text: OpenAIChatRequest(model="test", messages=[{"role": "user", "content": text}]),
        lambda text: ResponseTextPart(type="input_text", text=text),
        lambda text: OpenAIResponsesRequest(model="test", input=text),
        lambda text: OpenAIResponsesRequest(model="test", input="x", instructions=text),
        lambda text: McpRequest(
            jsonrpc="2.0",
            id=1,
            method="tools/call",
            params={"name": "lookup", "arguments": {"query": text}},
        ),
    ],
)
def test_model_visible_leaves_no_longer_have_an_obsolete_100k_cap(factory: Any) -> None:
    factory("x" * 100_001)


def test_mcp_structural_collections_and_metadata_are_bounded() -> None:
    base = {"jsonrpc": "2.0", "id": 1, "method": "tools/call"}
    McpRequest.model_validate(
        {
            **base,
            "params": {
                "name": "lookup",
                "arguments": {str(index): index for index in range(256)},
            },
        }
    )
    McpRequest.model_validate(
        {**base, "params": {"name": "lookup", "_meta": {str(index): index for index in range(64)}}}
    )

    with pytest.raises(ValidationError):
        McpRequest.model_validate(
            {
                **base,
                "params": {
                    "name": "lookup",
                    "arguments": {str(index): index for index in range(257)},
                },
            }
        )
    with pytest.raises(ValidationError):
        McpRequest.model_validate(
            {
                **base,
                "params": {
                    "name": "lookup",
                    "_meta": {str(index): index for index in range(65)},
                },
            }
        )


def _nested_mcp_meta(depth: int) -> dict[str, object]:
    value: object = 0
    for _ in range(depth):
        value = {"value": value}
    assert isinstance(value, dict)
    return value


def _mcp_meta_with_nodes(total: int) -> dict[str, object]:
    """Build one bounded-width metadata tree with exactly the requested value nodes."""
    if total < 2:
        raise ValueError("metadata node fixtures require at least two nodes")
    remaining = total - 2  # Root dictionary and its outer list.
    groups: list[list[int]] = []
    while remaining:
        values = min(256, remaining - 1)
        groups.append([0] * values)
        remaining -= values + 1  # Each inner list is also one node.
    return {"items": groups}


def _mcp_with_meta(meta: dict[str, object]) -> McpRequest:
    return McpRequest.model_validate(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "lookup", "_meta": meta},
        }
    )


def test_mcp_metadata_accepts_exact_depth_and_rejects_one_deeper() -> None:
    validate_request_structure(_mcp_with_meta(_nested_mcp_meta(32)), max_depth=32)

    with pytest.raises(InvalidAnalysisRequestError, match="nesting"):
        validate_request_structure(_mcp_with_meta(_nested_mcp_meta(33)), max_depth=32)


def test_mcp_metadata_accepts_exact_node_budget_and_rejects_one_more() -> None:
    validate_request_structure(_mcp_with_meta(_mcp_meta_with_nodes(4_096)), max_depth=32)

    with pytest.raises(AnalysisRequestTooLargeError, match="too many JSON nodes"):
        validate_request_structure(_mcp_with_meta(_mcp_meta_with_nodes(4_097)), max_depth=32)


@pytest.mark.parametrize(
    "meta",
    [
        {
            "deep": _nested_mcp_meta(33),
            "wide": _mcp_meta_with_nodes(4_097),
        },
        {
            "wide": _mcp_meta_with_nodes(4_097),
            "deep": _nested_mcp_meta(33),
        },
    ],
    ids=["depth-first-key", "nodes-first-key"],
)
def test_mcp_metadata_depth_error_precedes_node_error_independent_of_key_order(
    meta: dict[str, object],
) -> None:
    with pytest.raises(InvalidAnalysisRequestError, match="nesting"):
        validate_request_structure(_mcp_with_meta(meta), max_depth=32)


@pytest.mark.parametrize(
    "path",
    [
        "/v1/adapter/analyze-request",
        "/v1/studio/analyze-request",
        "/v1/studio/evaluate-policy",
    ],
)
@pytest.mark.parametrize(
    ("meta", "status_code", "error_code"),
    [
        (_nested_mcp_meta(33), 400, "invalid_request"),
        (_mcp_meta_with_nodes(4_097), 413, "request_too_large"),
    ],
    ids=["depth", "nodes"],
)
async def test_mcp_metadata_limit_failures_are_typed_before_all_analysis_endpoints(
    client: httpx.AsyncClient,
    path: str,
    meta: dict[str, object],
    status_code: int,
    error_code: str,
) -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "lookup",
            "arguments": {"query": "analyze only if metadata is valid"},
            "_meta": meta,
        },
    }
    request_body = payload if path == "/v1/adapter/analyze-request" else {"request": payload}

    response = await client.post(path, json=request_body)

    assert response.status_code == status_code, response.text
    assert response.json()["error"]["code"] == error_code


@pytest.mark.parametrize(
    "meta",
    [_nested_mcp_meta(32), _mcp_meta_with_nodes(4_096)],
    ids=["depth", "nodes"],
)
async def test_mcp_metadata_exact_boundaries_are_accepted_and_immutable(
    client: httpx.AsyncClient,
    meta: dict[str, object],
) -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "lookup", "_meta": meta},
    }

    response = await client.post("/v1/adapter/analyze-request", json=payload)

    assert response.status_code == 200, response.text
    assert response.json()["request"] == payload


def test_transformed_placeholder_expansion_beyond_100k_revalidates() -> None:
    request = OpenAIChatRequest(model="test", messages=[{"role": "user", "content": "x"}])
    replacement = PLACEHOLDER * 2_000
    assert len(replacement) > 100_000
    transformed = replace_text_leaves(request, {("messages", 0, "content"): replacement})
    assert isinstance(transformed, OpenAIChatRequest)
    assert transformed.messages[0].content == replacement


def _adapter_payload(reversal: dict[str, str]) -> dict[str, object]:
    count = len(reversal)
    return {
        "api_version": "v1",
        "decision": "apply_actions",
        "entities": ["EMAIL_ADDRESS"],
        "entity_counts": {"EMAIL_ADDRESS": count},
        "applied_actions": ["reversible_replace"],
        "remote_allowed": True,
        "route_class": "general",
        "request": {
            "model": "test",
            "messages": [{"role": "user", "content": "transformed"}],
        },
        "analysis": {
            "source": "current_request",
            "scan_performed": True,
            "duration_ms": 0,
            "overlap_count": 0,
            "overlap_resolution": "strictest_action",
            "policy_version": "v1",
            "text_leaf_count": 1,
            "cached_decision_applied": False,
        },
        "notices": {"request": [], "response": []},
        "safety_rule": None,
        "report": {
            "rows": [
                {
                    "entity_type": "EMAIL_ADDRESS",
                    "action": "reversible_replace",
                    "detected_count": count,
                    "transformed_count": count,
                    "unique_transformed_count": count,
                }
            ]
        },
        "reversal": reversal,
    }


def test_adapter_reversal_accepts_observed_617_entries() -> None:
    reversal = {
        f"<REV_EMAIL_ADDRESS_0123456789abcdef_{index:016x}>": f"value-{index}"
        for index in range(617)
    }
    assert len(AdapterAnalyzeResponse.model_validate(_adapter_payload(reversal)).reversal) == 617


@pytest.mark.parametrize(
    "reversal",
    [
        {"<REV_EMAIL_ADDRESS_BAD>": "plaintext"},
        {PLACEHOLDER: ""},
        {PLACEHOLDER: "x" * (MAX_TEXT_CHARACTERS + 1)},
    ],
)
def test_adapter_reversal_rejects_invalid_keys_and_plaintext(reversal: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        AdapterAnalyzeResponse.model_validate(_adapter_payload(reversal))


def test_adapter_reversal_accepts_plaintext_at_semantic_maximum() -> None:
    response = AdapterAnalyzeResponse.model_validate(
        _adapter_payload({PLACEHOLDER: "x" * MAX_TEXT_CHARACTERS})
    )
    assert len(response.reversal[PLACEHOLDER]) == MAX_TEXT_CHARACTERS


def _response_result(plaintext: str) -> PolicyResult:
    return PolicyResult(
        request=OpenAIChatRequest(
            model="test", messages=[{"role": "user", "content": PLACEHOLDER}]
        ),
        decision="apply_actions",
        remote_allowed=True,
        entities=["EMAIL_ADDRESS"],
        entity_counts={"EMAIL_ADDRESS": 1},
        applied_actions=["reversible_replace"],
        report_rows=[
            PIIReportRow(
                entity_type="EMAIL_ADDRESS",
                action="reversible_replace",
                detected_count=1,
                transformed_count=1,
                unique_transformed_count=1,
            )
        ],
        scan_performed=True,
        duration_ms=0,
        route_class="general",
        reversal={PLACEHOLDER: plaintext},
        text_leaf_count=1,
    )


async def test_adapter_response_budget_accepts_exact_bytes_and_returns_typed_413_over(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = "response-budget-private-marker"
    result = _response_result(marker * 100)

    async def analyze(*_args: object, **_kwargs: object) -> PolicyResult:
        return result

    runtime = get_runtime()
    monkeypatch.setattr(runtime, "analyze", analyze)
    first = await client.post(
        "/v1/adapter/analyze-request",
        json={"model": "test", "messages": [{"role": "user", "content": "x"}]},
    )
    assert first.status_code == 200
    exact_size = len(first.content)

    runtime.settings.max_adapter_response_bytes = exact_size
    exact = await client.post(
        "/v1/adapter/analyze-request",
        json={"model": "test", "messages": [{"role": "user", "content": "x"}]},
    )
    assert exact.status_code == 200
    assert len(exact.content) == exact_size

    caplog.clear()
    caplog.set_level(logging.INFO, logger="pii_engine.controllers.api")
    runtime.settings.max_adapter_response_bytes = exact_size - 1
    over = await client.post(
        "/v1/adapter/analyze-request",
        json={"model": "test", "messages": [{"role": "user", "content": "x"}]},
    )
    assert over.status_code == 413
    assert over.json()["error"]["code"] == "request_too_large"
    assert marker not in caplog.text
    assert "Traceback" not in caplog.text


async def test_invalid_adapter_response_is_contained_without_payload_logging(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = "invalid-response-private-marker"
    result = _response_result(marker)
    result.reversal = {"invalid-key": marker}

    async def analyze(*_args: object, **_kwargs: object) -> PolicyResult:
        return result

    monkeypatch.setattr(get_runtime(), "analyze", analyze)
    caplog.set_level(logging.INFO, logger="pii_engine.controllers.api")
    response = await client.post(
        "/v1/adapter/analyze-request",
        json={"model": "test", "messages": [{"role": "user", "content": "x"}]},
    )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert marker not in caplog.text
    assert "Traceback" not in caplog.text


def test_analysis_metadata_accepts_600_seconds_and_rejects_more() -> None:
    data = {
        "source": "current_request",
        "scan_performed": True,
        "duration_ms": 600_000,
        "overlap_count": 0,
        "overlap_resolution": "strictest_action",
        "policy_version": "v1",
        "text_leaf_count": 1,
        "cached_decision_applied": False,
    }
    assert AnalysisMetadata.model_validate(data).duration_ms == 600_000
    with pytest.raises(ValidationError):
        AnalysisMetadata.model_validate({**data, "duration_ms": 600_001})


def test_runtime_selects_caller_timeout_then_applies_policy_ceiling() -> None:
    runtime = EngineRuntime(Settings(allow_test_analyzer=True))
    policy = make_test_policy()
    assert runtime._analysis_timeout("adapter", policy) == 600
    assert runtime._analysis_timeout("studio", policy) == 30
    policy.pii.timeout = 12
    assert runtime._analysis_timeout("adapter", policy) == 12
    assert runtime._analysis_timeout("studio", policy) == 12

"""Exercise measured admission failures and safe error propagation."""

from __future__ import annotations

import json
from collections import deque

import httpx
import pytest

from pii_engine.main import RequestSizeLimitMiddleware
from pii_engine.models.contracts import LimitDetail
from pii_engine.runtime import get_runtime
from pii_engine.services.errors import AnalysisRequestTooLargeError


@pytest.mark.parametrize("more_body", [False, True])
@pytest.mark.parametrize("path", ["/v1/adapter/analyze-request", "/v2/adapter/analyze-segments"])
async def test_admission_counts_received_bytes_and_stops_reading(
    more_body: bool, path: str
) -> None:
    messages = deque(
        [
            {"type": "http.request", "body": b"123", "more_body": True},
            {"type": "http.request", "body": b"456", "more_body": more_body},
            {"type": "http.request", "body": b"must-not-read", "more_body": False},
        ]
    )
    sent = []

    async def receive():
        return messages.popleft()

    async def send(message):
        sent.append(message)

    async def app(*_args):
        pytest.fail("oversized input reached application")

    await RequestSizeLimitMiddleware(app, max_bytes=5)(
        {
            "type": "http",
            "path": path,
            "headers": [(b"x-correlation-id", b"test-correlation")],
        },
        receive,
        send,
    )
    assert len(messages) == 1
    assert sent[0]["status"] == 413
    assert (b"x-correlation-id", b"test-correlation") in sent[0]["headers"]
    assert json.loads(sent[1]["body"]) == {
        "api_version": "v2",
        "error": {
            "code": "request_too_large",
            "message": "The analysis request exceeds the configured size limit.",
            "retryable": False,
            "limit": {
                "component": "pii_engine",
                "stage": "admission",
                "reason": "encoded_bytes",
                "measured": 6,
                "maximum": 5,
                "unit": "bytes",
                "exact": not more_body,
            },
        },
    }


async def test_service_limit_survives_existing_controller_translation(client, monkeypatch):
    detail = LimitDetail(
        stage="inspection",
        reason="text_characters",
        measured=12,
        maximum=10,
        unit="characters",
        exact=True,
    )

    async def fail(*_args, **_kwargs):
        raise AnalysisRequestTooLargeError("private input marker", limit=detail)

    monkeypatch.setattr(get_runtime(), "analyze", fail)
    response = await client.post(
        "/v1/adapter/analyze-request",
        json={"model": "test", "messages": [{"role": "user", "content": "sample"}]},
        headers={"x-correlation-id": "test-correlation"},
    )
    assert response.status_code == 413
    assert response.json()["api_version"] == "v2"
    assert response.json()["error"]["limit"] == detail.model_dump()
    assert response.headers["x-correlation-id"] == "test-correlation"
    assert "private input marker" not in response.text


async def test_generic_error_has_no_invented_measurement(client: httpx.AsyncClient):
    response = await client.post("/v1/adapter/analyze-request", json={})
    assert response.json()["api_version"] == "v1"
    assert "limit" not in response.json()["error"]
    assert len(response.headers["x-correlation-id"]) == 32

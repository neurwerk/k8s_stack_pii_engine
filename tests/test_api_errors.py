"""Test typed fail-closed analysis errors and safe logging."""

from __future__ import annotations

import logging

import httpx
import pytest
from pydantic import ValidationError

from pii_engine.config.policy import PolicySettings
from pii_engine.config.settings import Settings
from pii_engine.runtime import RuntimeNotReadyError, get_runtime
from pii_engine.services.analyzer import EntityMatch
from pii_engine.services.errors import AnalysisRequestTooLargeError, InvalidAnalysisRequestError
from pii_engine.services.limiter import AnalysisCapacityError


def _request() -> dict[str, object]:
    return {"model": "test", "messages": [{"role": "user", "content": "sample text"}]}


def _error(code: str, message: str, retryable: bool) -> dict[str, object]:
    return {
        "api_version": "v1",
        "error": {"code": code, "message": message, "retryable": retryable},
    }


@pytest.mark.parametrize(
    ("exception", "status", "code", "message", "retryable"),
    [
        (
            RuntimeNotReadyError("diagnostic marker"),
            503,
            "runtime_unavailable",
            "The analysis runtime is unavailable.",
            True,
        ),
        (
            AnalysisCapacityError("diagnostic marker"),
            503,
            "capacity_unavailable",
            "Analysis capacity is temporarily unavailable.",
            True,
        ),
        (
            TimeoutError("diagnostic marker"),
            504,
            "analysis_timeout",
            "Analysis timed out.",
            True,
        ),
        (
            InvalidAnalysisRequestError("diagnostic marker"),
            400,
            "invalid_request",
            "The analysis request is invalid.",
            False,
        ),
        (
            AnalysisRequestTooLargeError("diagnostic marker"),
            413,
            "request_too_large",
            "The analysis request exceeds the configured size limit.",
            False,
        ),
        (
            ValueError("diagnostic marker"),
            500,
            "internal_error",
            "Analysis failed.",
            False,
        ),
        (
            RuntimeError("diagnostic marker"),
            500,
            "internal_error",
            "Analysis failed.",
            False,
        ),
    ],
)
async def test_analysis_failures_use_stable_typed_responses(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    exception: Exception,
    status: int,
    code: str,
    message: str,
    retryable: bool,
) -> None:
    async def fail(*_args: object, **_kwargs: object) -> None:
        raise exception

    monkeypatch.setattr(get_runtime(), "analyze", fail)
    response = await client.post("/v1/adapter/analyze-request", json=_request())
    assert response.status_code == status
    assert response.json() == _error(code, message, retryable)
    assert "diagnostic marker" not in response.text


async def test_planner_overlap_resolves_without_an_analysis_error(
    client: httpx.AsyncClient,
) -> None:
    class OverlappingAnalyzer:
        def analyze(self, text: str, policy: PolicySettings | None = None) -> list[EntityMatch]:
            del text, policy
            return [
                EntityMatch("EMAIL_ADDRESS", 0, 5, 0.9, "test"),
                EntityMatch("PHONE_NUMBER", 4, 8, 0.9, "test"),
            ]

    get_runtime().policy.analyzer = OverlappingAnalyzer()
    response = await client.post("/v1/adapter/analyze-request", json=_request())
    assert response.status_code == 200
    assert response.json()["analysis"]["overlap_count"] == 1


async def test_info_logging_excludes_raw_exception_and_traceback(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = "info-diagnostic-marker"

    async def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(marker)

    monkeypatch.setattr(get_runtime(), "analyze", fail)
    caplog.set_level(logging.INFO, logger="pii_engine.controllers.api")
    response = await client.post("/v1/adapter/analyze-request", json=_request())
    assert response.status_code == 500
    assert "caller=adapter reason=internal_error exception=RuntimeError" in caplog.text
    assert marker not in caplog.text
    assert "Traceback" not in caplog.text


async def test_validation_logging_contains_only_bounded_safe_diagnostics(
    client: httpx.AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    unknown_name = "synthetic_unknown_request_key"
    rejected_value = "synthetic_rejected_request_value"
    caplog.set_level(logging.ERROR, logger="pii_engine.main")

    cases = [
        ({**_request(), unknown_name: rejected_value}, "top_level"),
        (
            {
                **_request(),
                "stream_options": {
                    "include_usage": True,
                    unknown_name: rejected_value,
                },
            },
            "stream_options",
        ),
    ]
    for payload, scope in cases:
        caplog.clear()
        response = await client.post("/v1/adapter/analyze-request", json=payload)

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"
        assert (
            "request validation failed family=chat reason=extra_forbidden "
            f"scope={scope} count=1" in caplog.text
        )
        assert unknown_name not in caplog.text
        assert rejected_value not in caplog.text


async def test_debug_logging_includes_exception_and_traceback(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = "debug-diagnostic-marker"

    async def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(marker)

    monkeypatch.setattr(get_runtime(), "analyze", fail)
    caplog.set_level(logging.DEBUG, logger="pii_engine.controllers.api")
    response = await client.post("/v1/adapter/analyze-request", json=_request())
    assert response.status_code == 500
    assert marker in caplog.text
    assert "Traceback" in caplog.text


def test_log_level_is_validated() -> None:
    assert Settings(allow_test_analyzer=True).log_level == "INFO"
    assert Settings(allow_test_analyzer=True, log_level="DEBUG").log_level == "DEBUG"
    with pytest.raises(ValidationError):
        Settings.model_validate({"allow_test_analyzer": True, "log_level": "TRACE"})

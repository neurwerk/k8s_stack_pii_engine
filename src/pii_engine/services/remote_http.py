"""Bound remote analysis across all leaves; never retain transport error payloads."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from pii_engine.metrics import analysis_stage_duration_seconds

if TYPE_CHECKING:
    from pii_engine.config.remote import RemoteModel
    from pii_engine.config.settings import Settings


class RemoteAnalysisError(ValueError):
    """Stop analysis rather than interpret missing coverage as a clean scan."""


class InputTooLargeError(RemoteAnalysisError):
    """Permit only bounded, coverage-preserving subdivision of GLiNER input."""


@dataclass
class Budget:
    """Track a request's deadline and total remote calls, including subdivisions."""

    deadline: float
    remaining: int


_budget: ContextVar[Budget | None] = ContextVar("remote_ner_budget", default=None)


@contextmanager
def analysis_budget(timeout: float, max_calls: int) -> Iterator[None]:
    """Share one budget across every text leaf and endpoint in a policy request."""
    existing = _budget.get()
    if existing is not None:
        existing.deadline = min(existing.deadline, time.monotonic() + timeout)
        yield
        return
    token = _budget.set(Budget(time.monotonic() + timeout, max_calls))
    try:
        yield
    finally:
        _budget.reset(token)


class RemoteTransport:
    """Own a reusable verified HTTP client and bounded process-wide GPU admission."""

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        """Disable redirects and environment proxies for confidential requests."""
        self.settings = settings
        self.client = client or httpx.Client(trust_env=False, follow_redirects=False)
        self._capacity = threading.BoundedSemaphore(settings.remote_max_concurrent_calls)

    def close(self) -> None:
        """Close the shared HTTP connection pool."""
        self.client.close()

    def healthy(self, model: RemoteModel) -> bool:
        """Check only bounded model readiness metadata, without sending text."""
        path = "/health" if model.kind == "gliner" else f"/v1/models/{model.model_name}"
        url = str(httpx.URL(model.url).copy_with(path=path))
        try:
            with self.client.stream("GET", url, timeout=2, headers=_headers(model)) as response:
                if response.status_code != 200:
                    return False
                value = self._read_response(response, Budget(time.monotonic() + 2, 0))
                if not isinstance(value, dict):
                    return False
                if model.kind == "gliner":
                    return value.get("model") == model.model_name
                return value.get("name") == model.model_name and value.get("ready") is True
        except (httpx.HTTPError, ValueError, OSError, UnicodeError):
            return False

    def request(self, model: RemoteModel, payload: dict[str, Any]) -> object:
        """Read a bounded JSON reply within the shared request deadline."""
        budget = _budget.get()
        if budget is None or budget.remaining <= 0:
            raise RemoteAnalysisError("remote analysis call budget exceeded")
        budget.remaining -= 1
        remaining = budget.deadline - time.monotonic()
        started = time.monotonic()
        try:
            acquired = remaining > 0 and self._capacity.acquire(timeout=max(0, remaining))
        finally:
            analysis_stage_duration_seconds.labels(stage="wait").observe(time.monotonic() - started)
        if not acquired:
            raise RemoteAnalysisError("remote analysis deadline exceeded")
        try:
            return self._request(model, payload, budget)
        finally:
            self._capacity.release()

    def _request(self, model: RemoteModel, payload: dict[str, Any], budget: Budget) -> object:
        remaining = budget.deadline - time.monotonic()
        if remaining <= 0:
            raise RemoteAnalysisError("remote analysis deadline exceeded")
        headers = _headers(model)
        call_deadline = min(budget.deadline, time.monotonic() + self.settings.remote_call_timeout)
        try:
            with self.client.stream(
                "POST",
                model.url,
                json=payload,
                headers=headers,
                timeout=min(remaining, self.settings.remote_call_timeout),
            ) as response:
                if response.status_code == 413 and model.kind == "gliner":
                    raise InputTooLargeError("remote input requires subdivision")
                if response.status_code != 200:
                    raise RemoteAnalysisError("remote inference failed")
                return self._read_response(response, Budget(call_deadline, 0))
        except (httpx.HTTPError, ValueError, UnicodeError) as exc:
            if isinstance(exc, RemoteAnalysisError):
                raise
            raise RemoteAnalysisError("remote inference returned no valid response") from None

    def _read_response(self, response: httpx.Response, budget: Budget) -> object:
        data = bytearray()
        for part in response.iter_bytes():
            if time.monotonic() >= budget.deadline:
                raise RemoteAnalysisError("remote analysis deadline exceeded")
            if len(data) + len(part) > self.settings.remote_max_response_bytes:
                raise RemoteAnalysisError("remote response exceeds size limit")
            data.extend(part)
        return json.loads(data)


def _headers(model: RemoteModel) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if model.api_key_file is not None:
        try:
            key = model.api_key_file.read_text().strip()
        except (OSError, UnicodeError):
            raise RemoteAnalysisError("remote credential is unavailable") from None
        if not key or "\n" in key or "\r" in key:
            raise RemoteAnalysisError("remote credential is invalid")
        headers["Authorization"] = "Bearer " + key
    return headers

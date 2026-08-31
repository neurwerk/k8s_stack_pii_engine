"""Shared isolated API fixtures."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
from httpx import ASGITransport

from pii_engine.config.settings import get_settings
from pii_engine.main import create_app, create_management_app


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Enable only the explicit dependency-free analyzer in unit tests."""
    monkeypatch.setenv("PII_ENGINE_ALLOW_TEST_ANALYZER", "true")
    monkeypatch.setenv("PII_ENGINE_ENFORCE_CLIENT_IDENTITY", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    """Return an in-process analysis API client."""
    async with httpx.AsyncClient(
        transport=ASGITransport(app=create_app()), base_url="http://test"
    ) as value:
        yield value


@pytest.fixture
async def management_client() -> AsyncIterator[httpx.AsyncClient]:
    """Return an in-process management API client."""
    async with httpx.AsyncClient(
        transport=ASGITransport(app=create_management_app()), base_url="http://test"
    ) as value:
        yield value

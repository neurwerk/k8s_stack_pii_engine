"""Test management endpoint contracts."""

from __future__ import annotations

import httpx
import pytest

from pii_engine.runtime import get_runtime


async def test_management_endpoints_are_available(
    management_client: httpx.AsyncClient,
) -> None:
    """Liveness, readiness, and metrics are exposed on the management app."""
    assert (await management_client.get("/live")).json() == {"status": "ok"}
    assert (await management_client.get("/ready")).json() == {
        "status": "ok",
        "analyzer_mode": "test",
    }
    metrics = await management_client.get("/metrics")
    assert metrics.status_code == 200
    assert "python_info" in metrics.text
    assert 'pii_engine_runtime_analyzer_mode{mode="test"} 1.0' in metrics.text


async def test_analysis_exposes_only_adapter_readiness(
    client: httpx.AsyncClient,
) -> None:
    """Keep liveness and metrics on management while serving the adapter handshake."""
    assert (await client.get("/v1/adapter/ready")).json() == {"status": "ok"}
    for path in ("/ready", "/live", "/metrics"):
        assert (await client.get(path)).status_code == 404


async def test_adapter_readiness_tracks_runtime(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Withdraw extProc readiness when the selected runtime becomes unavailable."""

    async def unavailable() -> bool:
        return False

    monkeypatch.setattr(get_runtime(), "ready", unavailable)
    response = await client.get("/v1/adapter/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "policy runtime is not ready"}

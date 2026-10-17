"""Workload-only, expiring continuation evidence across Engine replicas."""

from unittest.mock import AsyncMock

import httpx
import pytest

from pii_engine.config.settings import Settings, get_settings
from pii_engine.runtime import EngineRuntime, set_runtime
from pii_engine.services.session import SessionStore

PATH = "/v2/adapter/reasoning-receipts"
BODY = {"scope": "a" * 64, "digests": ["b" * 64, "c" * 64]}


class ReceiptCache:
    """A deterministic clock/cache fake; the HTTP API and SessionStore remain real."""

    def __init__(self):
        self.now = 0
        self.values = {}
        self.pending = []

    def pipeline(self, *, transaction):
        assert transaction
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        self.pending.clear()

    def set(self, key, value, *, ex):
        self.pending.append((key, value, self.now + ex))

    async def execute(self):
        self.values.update({key: (value, expiry) for key, value, expiry in self.pending})

    async def mget(self, keys):
        return [
            value if expiry > self.now else None
            for value, expiry in (self.values.get(key, (None, 0)) for key in keys)
        ]


def replica(cache, policy="policy-1", **settings):
    runtime = EngineRuntime(Settings(allow_test_analyzer=True, policy_version=policy, **settings))
    runtime.session = SessionStore("redis://unused", 60, policy)
    runtime.session._client = cache
    set_runtime(runtime)
    return runtime


async def test_receipts_are_shared_scoped_and_expire_without_read_refresh(
    client: httpx.AsyncClient,
):
    cache = ReceiptCache()
    replica(cache)
    assert (await client.post(PATH, json=BODY)).status_code == 403
    assert (await client.put(PATH, json={**BODY, "policy_version": "policy-1"})).status_code == 204
    assert len(cache.values) == 2
    assert all(value == "1" and expiry == 60 for value, expiry in cache.values.values())
    replica(cache)  # A different process can authorize the next turn.
    cache.now = 59
    assert (await client.post(PATH, json=BODY)).status_code == 204
    assert (await client.post(PATH, json={**BODY, "policy_version": "policy-2"})).status_code == 503
    assert (await client.post(PATH, json={**BODY, "scope": "d" * 64})).status_code == 403
    assert (await client.post(PATH, json={**BODY, "digests": ["e" * 64]})).status_code == 403
    replica(cache, policy="policy-2")
    assert (await client.post(PATH, json=BODY)).status_code == 403
    before = dict(cache.values)
    assert (await client.put(PATH, json={**BODY, "policy_version": "policy-1"})).status_code == 503
    assert cache.values == before
    replica(cache)
    cache.now = 60
    assert (await client.post(PATH, json=BODY)).status_code == 403


@pytest.mark.parametrize("method", ["PUT", "POST"])
async def test_receipt_dependency_failure_is_closed(client, method):
    cache = ReceiptCache()
    runtime = replica(cache)
    cache.mget = AsyncMock(side_effect=OSError())
    cache.execute = AsyncMock(side_effect=OSError())
    assert (await client.request(method, PATH, json=BODY)).status_code == 503
    runtime.session = None
    assert (await client.request(method, PATH, json=BODY)).status_code == 503


@pytest.mark.parametrize("method", ["PUT", "POST"])
async def test_receipt_endpoint_requires_adapter_identity_and_only_accepts_digests(
    client, monkeypatch, caplog, method
):
    cache = ReceiptCache()
    replica(cache)
    sentinel = "synthetic-private-reasoning"
    for invalid in (
        {**BODY, "digests": [sentinel]},
        {**BODY, "text": sentinel},
    ):
        response = await client.request(method, PATH, json=invalid)
        assert response.status_code == 400
        assert sentinel not in response.text + caplog.text
    monkeypatch.setenv("PII_ENGINE_ENFORCE_CLIENT_IDENTITY", "true")
    get_settings.cache_clear()
    response = await client.request(
        method, PATH, json=BODY, headers={"x-workload-identity": "adapter"}
    )
    assert response.status_code == 403
    assert not cache.values


@pytest.mark.parametrize("configured_limit", [None, 2, 8192])
async def test_long_history_receipts_boundary(client, configured_limit):
    cache = ReceiptCache()
    runtime = replica(
        cache,
        **({} if configured_limit is None else {"max_reasoning_fingerprints": configured_limit}),
    )
    maximum = runtime.settings.max_reasoning_fingerprints
    body = {**BODY, "digests": [f"{index:064x}" for index in range(maximum)]}
    assert (await client.put(PATH, json=body)).status_code == 204
    assert (await client.post(PATH, json=body)).status_code == 204
    before = dict(cache.values)
    runtime.session.record_reasoning = AsyncMock()
    runtime.session.verify_reasoning = AsyncMock()
    oversized = {**body, "digests": [*body["digests"], "f" * 64]}
    for method in ("PUT", "POST"):
        for invalid in (oversized, {**body, "digests": ["a" * 64] * (maximum + 1)}):
            response = await client.request(method, PATH, json=invalid)
            assert response.status_code == 413
            limit = response.json()["error"]["limit"]
            assert limit["reason"] == "reasoning_fingerprints"
            assert limit["maximum"] == maximum
            assert limit["measured"] == maximum + 1
        assert cache.values == before
    runtime.session.record_reasoning.assert_not_called()
    runtime.session.verify_reasoning.assert_not_called()


def test_reasoning_limit_environment(monkeypatch):
    assert Settings(allow_test_analyzer=True).max_reasoning_fingerprints == 4096
    monkeypatch.setenv("PII_ENGINE_MAX_REASONING_FINGERPRINTS", "8192")
    assert Settings(allow_test_analyzer=True).max_reasoning_fingerprints == 8192
    for invalid in ("0", "-1", "1.5", "invalid"):
        monkeypatch.setenv("PII_ENGINE_MAX_REASONING_FINGERPRINTS", invalid)
        with pytest.raises(ValueError):
            Settings(allow_test_analyzer=True)

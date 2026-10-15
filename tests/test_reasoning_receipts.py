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


def replica(cache, policy="policy-1"):
    runtime = EngineRuntime(Settings(allow_test_analyzer=True, policy_version=policy))
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
        {**BODY, "digests": ["b" * 64] * 257},
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

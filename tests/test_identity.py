"""Test route-level workload authorization from verified client certificates."""

from __future__ import annotations

import pytest
from fastapi import HTTPException, Request

from pii_engine.config.settings import get_settings
from pii_engine.lib.identity import adapter_identity, studio_identity


def _request(common_name: str) -> Request:
    return Request(
        {
            "type": "http",
            "state": {
                "peer_certificate": {
                    "subject": ((("commonName", common_name),),),
                }
            },
        }
    )


def test_adapter_and_studio_identities_are_not_interchangeable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Studio cannot call the adapter route that returns reversal plaintext."""
    monkeypatch.setenv("PII_ENGINE_ALLOW_TEST_ANALYZER", "true")
    monkeypatch.setenv("PII_ENGINE_ENFORCE_CLIENT_IDENTITY", "true")
    get_settings.cache_clear()
    assert adapter_identity(_request("monitor-agentgateway-extproc")) == "adapter"
    assert studio_identity(_request("frontend-studio-api")) == "studio"
    with pytest.raises(HTTPException) as error:
        adapter_identity(_request("frontend-studio-api"))
    assert error.value.status_code == 403

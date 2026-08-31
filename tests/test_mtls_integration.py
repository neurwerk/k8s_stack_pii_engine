"""Exercise the application listener through real mutual TLS connections."""

from __future__ import annotations

import asyncio
import socket
import ssl
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
import trustme
import uvicorn
import yaml

from pii_engine.config.policy import test_policy as make_test_policy
from pii_engine.config.settings import Settings, get_settings
from pii_engine.main import ClientCertificateH11Protocol, _tls_kwargs, create_app
from pii_engine.runtime import EngineRuntime

type CertificatePaths = tuple[Path, Path]


def _write_certificate(
    certificate: trustme.LeafCert, directory: Path, name: str
) -> CertificatePaths:
    cert_path = directory / f"{name}.crt"
    key_path = directory / f"{name}.key"
    certificate.cert_chain_pems[0].write_to_path(cert_path)
    certificate.private_key_pem.write_to_path(key_path)
    return cert_path, key_path


def _client_context(ca_path: Path, certificate: CertificatePaths | None) -> ssl.SSLContext:
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=str(ca_path))
    if certificate is not None:
        context.load_cert_chain(str(certificate[0]), str(certificate[1]))
    return context


@asynccontextmanager
async def _running_server(settings: Settings, runtime: EngineRuntime) -> AsyncIterator[str]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.setblocking(False)
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(runtime, manage_runtime=False),
            host="127.0.0.1",
            port=port,
            http=ClientCertificateH11Protocol,
            lifespan="off",
            access_log=False,
            log_level="error",
            **_tls_kwargs(settings),
        )
    )
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        for _attempt in range(100):
            if server.started:
                break
            if task.done():
                await task
            await asyncio.sleep(0.01)
        if not server.started:
            raise RuntimeError("test TLS server did not start")
        yield f"https://localhost:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5)
        listener.close()


async def test_real_mtls_separates_adapter_and_studio_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TLS and route authorization keep reversal material adapter-only."""
    ca = trustme.CA()
    ca_path = tmp_path / "ca.crt"
    ca.cert_pem.write_to_path(ca_path)
    server_cert, server_key = _write_certificate(
        ca.issue_cert("localhost", common_name="monitor-pii-engine-service"),
        tmp_path,
        "server",
    )
    adapter = _write_certificate(
        ca.issue_cert("monitor-agentgateway-extproc", common_name="monitor-agentgateway-extproc"),
        tmp_path,
        "adapter",
    )
    studio = _write_certificate(
        ca.issue_cert("frontend-studio-api", common_name="frontend-studio-api"),
        tmp_path,
        "studio",
    )
    unauthorized = _write_certificate(
        ca.issue_cert("other-workload", common_name="other-workload"),
        tmp_path,
        "unauthorized",
    )

    policy = make_test_policy()
    policy.pii.entity_policies[0].action = "reversible_replace"
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        yaml.safe_dump(policy.model_dump(mode="json", by_alias=True)), encoding="utf-8"
    )
    settings = Settings(
        allow_test_analyzer=True,
        enforce_client_identity=True,
        policy_config=policy_path,
        tls_cert=server_cert,
        tls_key=server_key,
        tls_ca=ca_path,
    )
    runtime = EngineRuntime(settings)
    monkeypatch.setenv("PII_ENGINE_ENFORCE_CLIENT_IDENTITY", "true")
    monkeypatch.setenv("PII_ENGINE_ADAPTER_COMMON_NAME", "monitor-agentgateway-extproc")
    monkeypatch.setenv("PII_ENGINE_STUDIO_COMMON_NAME", "frontend-studio-api")
    get_settings.cache_clear()

    request = {
        "model": "test",
        "messages": [{"role": "user", "content": "email a@example.com"}],
    }
    async with _running_server(settings, runtime) as base_url:
        async with httpx.AsyncClient(
            base_url=base_url,
            verify=_client_context(ca_path, adapter),
            trust_env=False,
        ) as adapter_client:
            adapter_ready = await adapter_client.get("/v1/adapter/ready")
            adapter_result = await adapter_client.post("/v1/adapter/analyze-request", json=request)
            adapter_cross = await adapter_client.post(
                "/v1/studio/analyze-request", json={"request": request}
            )
            adapter_evaluation = await adapter_client.post(
                "/v1/studio/evaluate-policy", json={"request": request}
            )
            adapter_admin = await adapter_client.get("/v1/actions")
        async with httpx.AsyncClient(
            base_url=base_url,
            verify=_client_context(ca_path, studio),
            trust_env=False,
        ) as studio_client:
            studio_ready = await studio_client.get("/v1/adapter/ready")
            studio_result = await studio_client.post(
                "/v1/studio/analyze-request", json={"request": request}
            )
            studio_evaluation = await studio_client.post(
                "/v1/studio/evaluate-policy", json={"request": request}
            )
            studio_cross = await studio_client.post("/v1/adapter/analyze-request", json=request)
            studio_admin = await studio_client.get("/v1/actions")
        async with httpx.AsyncClient(
            base_url=base_url,
            verify=_client_context(ca_path, unauthorized),
            trust_env=False,
        ) as unauthorized_client:
            unauthorized_ready = await unauthorized_client.get("/v1/adapter/ready")
            unauthorized_result = await unauthorized_client.post(
                "/v1/adapter/analyze-request", json=request
            )
        async with httpx.AsyncClient(
            base_url=base_url,
            verify=_client_context(ca_path, None),
            trust_env=False,
        ) as anonymous_client:
            with pytest.raises(httpx.TransportError):
                await anonymous_client.get("/v1/adapter/ready")

    assert adapter_ready.status_code == 200
    assert adapter_ready.json() == {"status": "ok"}
    assert adapter_result.status_code == 200
    assert studio_result.status_code == 200
    adapter_body = adapter_result.json()
    studio_body = studio_result.json()
    assert adapter_body["decision"] == studio_body["decision"] == "apply_actions"
    assert (
        adapter_body["applied_actions"] == studio_body["applied_actions"] == ["reversible_replace"]
    )
    assert list(adapter_body["reversal"].values()) == ["a@example.com"]
    assert "reversal" not in studio_body
    assert adapter_cross.status_code == 403
    assert adapter_evaluation.status_code == 403
    assert studio_ready.status_code == 403
    assert studio_cross.status_code == 403
    assert adapter_admin.status_code == 403
    assert studio_admin.status_code == 200
    assert studio_evaluation.status_code == 200
    assert studio_evaluation.json()["valid"] is True
    assert unauthorized_ready.status_code == 403
    assert unauthorized_result.status_code == 403

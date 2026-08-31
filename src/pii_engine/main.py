"""Dual-port PII Engine with mandatory application mTLS identity propagation."""

from __future__ import annotations

import asyncio
import logging
import ssl
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from importlib.metadata import version as package_version
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from uvicorn.protocols.http.h11_impl import H11Protocol

from pii_engine.config.settings import Settings, get_settings
from pii_engine.controllers.api import (
    AnalysisAPIError,
    analysis_api_error,
    log_analysis_failure,
)
from pii_engine.controllers.api import (
    router as api_router,
)
from pii_engine.controllers.health import router as health_router
from pii_engine.lib.identity import peer_certificate_from_transport
from pii_engine.models.contracts import AnalysisErrorCode
from pii_engine.runtime import EngineRuntime, get_runtime, initialize_runtime, set_runtime
from pii_engine.services.errors import AnalysisRequestTooLargeError, InvalidAnalysisRequestError

_VERSION = package_version("neurwerk-pii-engine")


class ClientCertificateH11Protocol(H11Protocol):
    """Inject the TLS-verified peer certificate into per-connection ASGI state."""

    def connection_made(self, transport: asyncio.Transport) -> None:
        """Copy shared state before adding connection-specific certificate data."""
        self.app_state = self.app_state.copy()
        self.app_state["peer_certificate"] = peer_certificate_from_transport(transport)
        super().connection_made(transport)


class RequestSizeLimitMiddleware:
    """Enforce declared and streamed body limits without pre-buffering the request."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        """Store the wrapped application and byte ceiling."""
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Forward one request through a receive function that counts chunks."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if declared_error := self._declared_error(scope):
            await self._error(scope, receive, send, *declared_error)
            return
        buffered, stream_error = await self._buffer_body(receive)
        if stream_error is not None:
            await self._error(scope, receive, send, "request_too_large", stream_error)
            return

        async def limited_receive() -> Message:
            if buffered:
                return buffered.popleft()
            return {"type": "http.disconnect"}

        await self.app(scope, limited_receive, send)

    def _declared_error(self, scope: Scope) -> tuple[AnalysisErrorCode, str] | None:
        """Validate Content-Length before consuming any request bytes."""
        content_length = Headers(scope=scope).get("content-length")
        if content_length is None:
            return None
        try:
            declared = int(content_length)
        except ValueError:
            return "invalid_request", "invalid content length"
        if declared < 0 or declared > self.max_bytes:
            return "request_too_large", "request body too large"
        return None

    async def _buffer_body(self, receive: Receive) -> tuple[deque[Message], str | None]:
        """Buffer at most the configured bytes while bounding empty chunk churn."""
        body = bytearray()
        consumed = 0
        empty_chunks = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return deque([message]), None
            chunk = message.get("body", b"")
            if not chunk and message.get("more_body", False):
                empty_chunks += 1
                if empty_chunks > 4_096:
                    return deque(), "too many empty request body chunks"
            consumed += len(chunk)
            if consumed > self.max_bytes:
                return deque(), "request body too large"
            body.extend(chunk)
            if not message.get("more_body", False):
                return deque(
                    [{"type": "http.request", "body": bytes(body), "more_body": False}]
                ), None

    @staticmethod
    async def _error(
        scope: Scope, receive: Receive, send: Send, code: AnalysisErrorCode, detail: str
    ) -> None:
        exc = (
            AnalysisRequestTooLargeError(detail)
            if code == "request_too_large"
            else InvalidAnalysisRequestError(detail)
        )
        failure = analysis_api_error(code)
        log_analysis_failure(
            _analysis_caller(scope.get("path", "")), code, exc, debug_details=False
        )
        response = JSONResponse(
            status_code=failure.status_code,
            content=failure.response.model_dump(mode="json"),
        )
        await response(scope, receive, send)


def create_app(runtime: EngineRuntime | None = None, manage_runtime: bool = True) -> FastAPI:
    """Create the workload-authenticated analysis application."""
    selected = runtime or initialize_runtime(get_settings())
    configure_logging(selected.settings)
    set_runtime(selected)
    app = FastAPI(
        title="PII Engine",
        version=_VERSION,
        lifespan=_runtime_lifespan(selected) if manage_runtime else None,
    )
    app.include_router(api_router)
    _install_analysis_error_handlers(app)
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=selected.settings.max_request_bytes)
    return app


def create_management_app(
    runtime: EngineRuntime | None = None, manage_runtime: bool = True
) -> FastAPI:
    """Create the separate unauthenticated management application."""
    selected = runtime or _existing_or_new_runtime()
    configure_logging(selected.settings)
    set_runtime(selected)
    app = FastAPI(
        title="PII Engine Management",
        version=_VERSION,
        lifespan=_runtime_lifespan(selected) if manage_runtime else None,
    )
    app.include_router(health_router)
    return app


def configure_logging(settings: Settings) -> None:
    """Configure process logging from validated service settings."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        level=settings.log_level,
    )
    logging.getLogger().setLevel(settings.log_level)


def _install_analysis_error_handlers(app: FastAPI) -> None:
    """Install strict fail-closed handlers on the workload application only."""

    @app.exception_handler(AnalysisAPIError)
    async def analysis_error_handler(_request: Request, exc: AnalysisAPIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.response.model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        code: AnalysisErrorCode = "invalid_request"
        log_analysis_failure(_analysis_caller(request.url.path), code, exc, debug_details=False)
        failure = analysis_api_error(code)
        return JSONResponse(
            status_code=failure.status_code,
            content=failure.response.model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        code: AnalysisErrorCode = "internal_error"
        log_analysis_failure(_analysis_caller(request.url.path), code, exc)
        failure = analysis_api_error(code)
        return JSONResponse(
            status_code=failure.status_code,
            content=failure.response.model_dump(mode="json"),
        )


def _analysis_caller(path: str) -> str:
    """Derive only the bounded caller class from a workload route path."""
    if path.startswith("/v1/adapter/"):
        return "adapter"
    if path.startswith("/v1/studio/"):
        return "studio"
    return "unknown"


def _existing_or_new_runtime() -> EngineRuntime:
    try:
        return get_runtime()
    except RuntimeError:
        return initialize_runtime(get_settings())


def _runtime_lifespan(
    runtime: EngineRuntime,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await runtime.start()
        try:
            yield
        finally:
            await runtime.close()

    return lifespan


async def _serve() -> None:
    """Initialize once and serve application plus management listeners."""
    settings = get_settings()
    configure_logging(settings)
    runtime = initialize_runtime(settings)
    await runtime.start()
    analysis = uvicorn.Server(
        uvicorn.Config(
            create_app(runtime, manage_runtime=False),
            host=settings.analysis_host,
            port=settings.analysis_port,
            http=ClientCertificateH11Protocol,
            log_level=settings.log_level.lower(),
            **_tls_kwargs(settings),
        )
    )
    management = uvicorn.Server(
        uvicorn.Config(
            create_management_app(runtime, manage_runtime=False),
            host=settings.management_host,
            port=settings.management_port,
            log_level=settings.log_level.lower(),
        )
    )
    try:
        await asyncio.gather(analysis.serve(), management.serve())
    finally:
        await runtime.close()


def _tls_kwargs(settings: Settings) -> dict[str, Any]:
    """Return mandatory server TLS and client-certificate validation settings."""
    if settings.tls_cert is None or settings.tls_key is None or settings.tls_ca is None:
        raise RuntimeError("application TLS requires cert, key, and client CA")
    return {
        "ssl_certfile": str(settings.tls_cert),
        "ssl_keyfile": str(settings.tls_key),
        "ssl_ca_certs": str(settings.tls_ca),
        "ssl_cert_reqs": ssl.CERT_REQUIRED,
    }


def main() -> None:
    """Start the analysis and management servers."""
    asyncio.run(_serve())


if __name__ == "__main__":
    main()

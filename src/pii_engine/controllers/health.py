"""Management endpoints for liveness, readiness, and Prometheus metrics."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from pii_engine.runtime import get_runtime

router = APIRouter()


@router.get("/live")
def live() -> dict[str, str]:
    """Restart a baseline process after a verified transformer bundle activates."""
    try:
        runtime = get_runtime()
    except RuntimeError:
        return {"status": "ok"}
    if runtime.restart_required():
        raise HTTPException(status_code=503, detail="model activation requires restart")
    return {"status": "ok"}


@router.get("/ready")
async def ready() -> dict[str, str]:
    """Report readiness only after startup dependency validation succeeds."""
    try:
        runtime = get_runtime()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="policy runtime is not ready") from exc
    if not await runtime.ready():
        raise HTTPException(status_code=503, detail="policy runtime is not ready")
    return {"status": "ok", "analyzer_mode": runtime.analyzer_mode}


@router.get("/metrics", response_class=PlainTextResponse)
def metrics() -> PlainTextResponse:
    """Expose Prometheus metrics."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

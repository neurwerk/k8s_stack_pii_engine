"""Authorize workload identities from the mTLS peer certificate."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import HTTPException, Request

from pii_engine.config.settings import get_settings

type Caller = Literal["adapter", "studio"]


def adapter_identity(request: Request) -> Caller:
    """Require the dedicated extproc client identity."""
    _require_common_name(request, get_settings().adapter_common_name)
    return "adapter"


def studio_identity(request: Request) -> Caller:
    """Require the dedicated Studio API client identity."""
    _require_common_name(request, get_settings().studio_common_name)
    return "studio"


def _require_common_name(request: Request, expected: str) -> None:
    settings = get_settings()
    if not settings.enforce_client_identity:
        return
    certificate = request.scope.get("state", {}).get("peer_certificate")
    common_names = _common_names(certificate)
    if common_names != [expected]:
        raise HTTPException(status_code=403, detail="workload identity is not authorized")


def _common_names(certificate: object) -> list[str]:
    """Extract exact common names from Python's decoded peer certificate."""
    if not isinstance(certificate, dict):
        return []
    names: list[str] = []
    subject = certificate.get("subject", ())
    if not isinstance(subject, tuple):
        return []
    for relative_name in subject:
        if not isinstance(relative_name, tuple):
            continue
        for attribute in relative_name:
            if (
                isinstance(attribute, tuple)
                and len(attribute) == 2
                and attribute[0] == "commonName"
                and isinstance(attribute[1], str)
            ):
                names.append(attribute[1])
    return names


def peer_certificate_from_transport(transport: Any) -> dict[str, object] | None:  # noqa: ANN401
    """Read the verified certificate decoded by the TLS transport."""
    ssl_object = transport.get_extra_info("ssl_object")
    if ssl_object is None:
        return None
    certificate = ssl_object.getpeercert()
    return certificate if isinstance(certificate, dict) else None

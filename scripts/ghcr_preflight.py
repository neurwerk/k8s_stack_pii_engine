"""Check absence of immutable tags in the fixed public PII Engine repository."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Sequence
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPOSITORY = "neurwerk/k8s-stack-pii-engine"
TOKEN_URL = f"https://ghcr.io/token?service=ghcr.io&scope=repository:{REPOSITORY}:pull"
TAG_PATTERN = re.compile(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)-(?:cpu|cu124)\Z")
ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)


class PreflightError(ValueError):
    """A safe, locally authored error suitable for operator output."""


def read_json(response: BinaryIO | HTTPError) -> dict:
    """Bound response size and require a JSON object without exposing its body."""
    data = response.read(65537)
    if len(data) > 65536:
        raise PreflightError("Registry response exceeded the preflight size limit")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise PreflightError("Registry response was not a JSON object")
    return value


def check_tags(tags: Sequence[str]) -> None:
    """Allow only explicit structured absence; do not test push permissions."""
    if not tags or any(TAG_PATTERN.fullmatch(tag) is None for tag in tags):
        raise PreflightError("Expected immutable X.Y.Z-cpu or X.Y.Z-cu124 tags")
    with urlopen(TOKEN_URL, timeout=30) as response:
        token = read_json(response).get("token")
    if not isinstance(token, str) or not token:
        raise PreflightError("Anonymous GHCR pull token was unavailable")
    for tag in tags:
        request = Request(
            f"https://ghcr.io/v2/{REPOSITORY}/manifests/{tag}",
            headers={"Authorization": f"Bearer {token}", "Accept": ACCEPT},
        )
        try:
            with urlopen(request, timeout=30) as response:  # noqa: S310
                if response.status == 200:
                    raise PreflightError(f"{tag} already exists; deselect that variant to resume")
                raise PreflightError("Unexpected registry response status")
        except HTTPError as error:
            with error:
                if error.code != 404:
                    raise PreflightError(f"Registry preflight failed (HTTP {error.code})") from None
                errors = read_json(error).get("errors")
            if (
                not isinstance(errors, list)
                or not errors
                or any(
                    not isinstance(item, dict)
                    or item.get("code") not in {"MANIFEST_UNKNOWN", "NAME_UNKNOWN"}
                    for item in errors
                )
            ):
                raise PreflightError(
                    "Registry 404 did not explicitly establish tag absence"
                ) from None
        sys.stdout.write(f"Absent: ghcr.io/{REPOSITORY}:{tag}\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run only the read-only, anonymous publication preflight."""
    try:
        check_tags(list(sys.argv[1:] if argv is None else argv))
    except PreflightError as error:
        sys.stderr.write(f"ERROR: {error}. No builds started.\n")
        return 1
    except HTTPError as error:
        sys.stderr.write(f"ERROR: Anonymous GHCR token request failed (HTTP {error.code}).\n")
        return 1
    except (URLError, OSError):
        sys.stderr.write("ERROR: GHCR preflight transport failed. No builds started.\n")
        return 1
    except (ValueError, TypeError):
        # JSON bodies and exception details may contain sensitive server content.
        sys.stderr.write(
            "ERROR: GHCR response was invalid; absence could not be verified. No builds started.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

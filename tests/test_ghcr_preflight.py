import io
import json
from email.message import Message
from urllib.error import HTTPError, URLError

import pytest

from scripts import ghcr_preflight


class Response(io.BytesIO):
    status = 200


@pytest.fixture
def registry(monkeypatch):
    calls = []

    def install(status=404, body=None, token_body=b'{"token": "anonymous-test-token"}'):
        if body is None:
            body = {"errors": [{"code": "MANIFEST_UNKNOWN"}]}
        data = json.dumps(body).encode() if not isinstance(body, bytes) else body

        def open_response(request, timeout):
            assert timeout == 30
            calls.append(request)
            if isinstance(request, str):
                assert request == ghcr_preflight.TOKEN_URL
                assert "scope=repository:neurwerk/k8s-stack-pii-engine:pull" in request
                return Response(token_body)
            assert request.get_header("Authorization") == "Bearer anonymous-test-token"
            assert "application/vnd.oci.image.index.v1+json" in request.get_header("Accept")
            if status == 200:
                return Response(b"{}")
            raise HTTPError(request.full_url, status, "not found", Message(), io.BytesIO(data))

        monkeypatch.setattr(ghcr_preflight, "urlopen", open_response)
        return calls

    return install


@pytest.mark.parametrize("code", ["MANIFEST_UNKNOWN", "NAME_UNKNOWN"])
def test_structured_missing_tags_allow_both_variants(registry, capsys, code):
    calls = registry(body={"errors": [{"code": code}]})
    assert ghcr_preflight.main(["0.13.0-cpu", "0.13.0-cu124"]) == 0
    assert len(calls) == 3  # one scoped anonymous token, then both manifests
    assert calls[1].full_url.endswith("/manifests/0.13.0-cpu")
    assert calls[2].full_url.endswith("/manifests/0.13.0-cu124")
    output = capsys.readouterr()
    assert "Absent:" in output.out
    assert "anonymous-test-token" not in output.out + output.err


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (200, {}),
        (401, {"errors": [{"code": "MANIFEST_UNKNOWN"}]}),
        (403, {"errors": [{"code": "DENIED"}]}),
        (500, {"errors": [{"code": "MANIFEST_UNKNOWN"}]}),
        (404, b"ERROR: ghcr.io/neurwerk/k8s-stack-pii-engine:0.13.0-cpu: not found"),
        (404, {"errors": [{"code": "UNAUTHORIZED"}]}),
        (404, {"errors": []}),
        (404, {}),
        (404, {"errors": [{"code": "MANIFEST_UNKNOWN"}, {"code": "DENIED"}]}),
        (404, b"sensitive-server-detail" * 4096),
    ],
    ids=[
        "exists",
        "unauthorized",
        "denied",
        "server-error",
        "docker-not-found",
        "ambiguous-404",
        "empty-errors",
        "missing-errors",
        "mixed-errors",
        "oversized-body",
    ],
)
def test_existing_or_ambiguous_replies_never_allow_build(registry, capsys, status, body):
    registry(status=status, body=body)
    assert ghcr_preflight.main(["0.13.0-cpu"]) == 1
    output = capsys.readouterr()
    assert "Absent:" not in output.out
    assert "anonymous-test-token" not in output.err
    assert "sensitive-server-detail" not in output.err


@pytest.mark.parametrize("token_body", [b"{}", b'{"token": null}', b"not-json"])
def test_bad_token_response_stops_before_manifest_request(registry, token_body):
    calls = registry(token_body=token_body)
    assert ghcr_preflight.main(["0.13.0-cpu"]) == 1
    assert len(calls) == 1


def test_transport_error_is_bounded_and_redacted(monkeypatch, capsys):
    def unavailable(*args, **kwargs):
        raise URLError("sensitive-server-detail")

    monkeypatch.setattr(ghcr_preflight, "urlopen", unavailable)
    assert ghcr_preflight.main(["0.13.0-cpu"]) == 1
    output = capsys.readouterr()
    assert "transport failed" in output.err
    assert "sensitive-server-detail" not in output.err


@pytest.mark.parametrize("tags", [[], ["latest"], ["0.13.0-arm64"], ["../token"]])
def test_invalid_tags_never_make_http_calls(monkeypatch, tags):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid tags must not contact GHCR")

    monkeypatch.setattr(ghcr_preflight, "urlopen", forbidden)
    assert ghcr_preflight.main(tags) == 1

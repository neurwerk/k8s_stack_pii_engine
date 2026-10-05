# Neurwerk PII Engine

Bounded, deterministic PII and safety policy evaluation for supported LLM and
MCP requests. The service analyzes extracted text segments with Microsoft Presidio,
applies policy-selected transformations or terminal decisions, and returns a
strict typed result to its trusted adapters.

Canonical source: <https://github.com/neurwerk/k8s_stack_pii_engine>

## Architecture

- The analysis listener serves the versioned adapter and Studio APIs over mTLS.
- The separate management listener exposes liveness, readiness, and Prometheus
  metrics.
- Presidio and bundled English, German, and Dutch spaCy pipelines provide the
  offline baseline runtime.
- An optional model-sync process downloads a pinned transformer bundle from an
  S3-compatible store, verifies its manifest and files, and atomically selects
  it. Invalid or unavailable bundles do not replace the baseline.
- Optional Valkey state retains bounded terminal policy decisions. Request text
  and reversal mappings are not persisted there.

This repository owns the engine service and model-sync CLI. Gateway adaptation,
human authorization, Kubernetes charts, and deployment values are separate
components.

## Segment API

`POST /v2/adapter/analyze-segments` accepts extracted text rather than provider JSON:

```json
{
  "api_version": "v2",
  "request_kind": "chat",
  "scope": "session",
  "segments": [{"id": "s0", "text": "Text to inspect"}],
  "text_pii_enabled": true,
  "attachments_present": false
}
```

`request_kind` is `chat`, `responses`, or `mcp`. Segment IDs are unique and opaque;
spans never cross segments. Successful results return the same IDs in the same
order under `segments`; blocked results return `segments: null`. Policy decisions,
reports, notices, and adapter-only reversal mappings retain their existing meaning.
Callers retain provider controls and reconstruct only the original text locations.

Session-scoped adapter requests use the trusted `x-pii-session-key`. Stable aliases
are limited to session-scoped Chat requests. Converted documents and images use
`scope: request`, fresh aliases, and no session reads or writes. Optional
`visual_findings` retains the documented face contract and requires request-scoped
model analysis. Disabling text scanning requires these trusted visual controls.
Unconverted attachments (`attachments_present: true`) block even on cached reroutes.

Studio uses `POST /v2/studio/analyze-segments` with `{request, policy?}` and
`POST /v2/studio/evaluate-policy` with `{request, policy?, simulation?}`. The only
simulation mode is `deterministic_echo`. Studio requires request scope and cannot
submit visual findings or disable text scanning. Evaluation diagnostics identify
`segment_id` and original segment-local offsets; no reversal mapping is exposed.

`GET /v2/adapter/ready` requires the adapter mTLS identity and returns
`{"api_version":"v2","status":"ok"}` only when the analysis runtime is ready.
Measured limit failures use the v2 error envelope with content-free measurements;
generic errors retain the v1 error envelope. Limits are unchanged, and all logging
remains content-free even at DEBUG.

The v1 routes are compatibility wrappers over the same segment core. Their provider
schemas and text extraction come from `neurwerk-request-segments`, maintained in
the extProc repository. `vendor/request_segments/` is an unpublished source snapshot
so ordinary uv and Docker builds need no sibling checkout or package publication.

## Legacy Document Request API

`POST /v1/adapter/analyze-document-request` requires the adapter mTLS identity.
Its legacy body is an existing `OpenAIChatRequest` or `OpenAIResponsesRequest`: the whole
conversation with extracted document text and any retained metadata already in
model-visible text parts, not raw Docling JSON. Typed attachments produce the
existing policy `block` with no request content or reversal; MCP is invalid here.

The endpoint uses the same policy, planner, `AdapterAnalyzeResponse`, and typed
errors as `/v1/adapter/analyze-request`. It analyzes the complete request once,
uses fresh request-local aliases, and never reads or writes session decisions;
`x-pii-session-key` is ignored. Readiness, queue, timeouts, body/text limits, and
the adapter response byte limit remain shared. Failures return no partial result,
and exception details are suppressed even at `DEBUG`. No cross-line or table-cell
reconstruction is performed, so split PII may be missed. Existing adapter, MCP,
and Studio routes are unchanged.

The document endpoint also accepts this strict, adapter-owned envelope:

```json
{
  "api_version": "v1",
  "request": {"model": "example", "input": "Converted attachment text"},
  "text_pii_enabled": true,
  "visual_findings": {"faces": {"scan_status": "complete", "count": 2}}
}
```

All envelope fields are required; unknown fields are rejected. `request` must
be a normal text-only Chat or Responses payload. Only trusted extProc constructs
the findings, never a public caller, Studio or MCP. Engine receives no pixels,
face identities or fabricated `PERSON` entities. Images without OCR text use an
adapter-generated fixed text marker. A complete face scan requires a strict
integer count from 0 through 10,000,000; `failed` and `not_scanned` require null.
`failed` blocks the request. `not_scanned` means face protection was deliberately
disabled and does not evaluate face policy. Findings are never cached.

`text_pii_enabled` is a strict boolean. When false, text PII scanning,
transformations and classification are skipped, but request bounds, safety
checks and face policy still apply. Scan metadata remains truthful:
`scan_performed: false` and `duration_ms: null` describe skipped text scanning.
Envelope replies echo `visual_findings`; legacy replies omit that field entirely.

Face policy is separate from text entity policies and recognizers, where `FACE`
is reserved:

```yaml
attachments:
  policy: block
  faces:
    action: block # block (default), text-only, or reroute
    # routeClass: local/safe # permitted only for reroute
```

An omitted reroute class uses `routing.defaultTarget`. Face blocks and failed
inspection stop processing before text transformations. Text blocks always win;
conflicting text and face reroute classes block instead of choosing one. Positive
counts produce one aggregate `FACE` report row and matching entity counts, with
zero transformations. Its effective action is `block` on any overall block,
otherwise `text-only` or `reroute`; zero or unknown counts produce no face row.
Face-only `text-only` returns `apply_actions`, the text request and
`applied_actions: ["text-only"]`, not a new decision. extProc owns image removal
and safe route enforcement; this policy does not grant image forwarding.

## Configuration

Runtime environment variables use the `PII_ENGINE_` prefix. Supported setting
categories are:

- analysis and management listener addresses;
- mTLS files and allowed adapter/Studio certificate common names;
- request, response, nesting, concurrency, queue, and timeout bounds;
- policy path/version and CPU or CUDA device selection;
- verified model-cache, desired-bundle, version, and manifest-digest selection;
- Valkey session URL and cryptographic hash/encryption keys;
- logging and the isolated test-analyzer switch.

The strict policy YAML covers PII languages, recognizers, entities and actions;
attachment handling; safety rules; transformed-content classification; session
behavior; notices; trusted route targets; and logging. Unknown policy fields are
rejected. See `.env.example` for redacted variable shapes and
`src/pii_engine/config/` for the authoritative schemas and defaults.

## Secrets And Models

Policy configuration and model bundle pins are non-secret. TLS private keys,
hash and encryption keys, credential-bearing Valkey URLs, and object-store
credentials are secrets and must be injected at runtime rather than committed.
The model-sync CLI uses boto3's standard credential provider chain; model-store
credentials are not `PII_ENGINE_` settings.

Release images contain the three small baseline spaCy model wheels. Optional
transformer bundles and Hugging Face caches are external artifacts, not source
files or image build inputs. Operators are responsible for the licenses and
redistribution terms of any separately supplied model bundle. See
`THIRD_PARTY_NOTICES.md` for bundled dependency and image notices.

## Local Validation

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. The complete
offline quality gate installs the CPU and development extras:

```bash
make check
```

This runs the frozen-lock check, Ruff lint/format checks, `ty` type checking,
and pytest with an informational coverage report. `make benchmark` runs the
synthetic benchmark separately. `make build` creates a local CPU image and is
not part of validation.

The Dockerfile keeps version tags for readability and pins their OCI image
indexes by digest. When updating the Dockerfile frontend, uv, or Python image,
inspect the authoritative registry manifest and confirm that the selected index
contains a `linux/amd64` manifest before replacing both the version and digest:

```bash
docker buildx imagetools inspect docker/dockerfile:<version>
docker buildx imagetools inspect ghcr.io/astral-sh/uv:<version>
docker buildx imagetools inspect python:<version>-slim
docker build --check .
docker build --platform linux/amd64 --build-arg ACCELERATOR=cpu \
  -t pii-engine:validation-cpu .
```

## Release Images

Version tags publish Linux AMD64 images to GitHub Container Registry:

- `ghcr.io/neurwerk/k8s-stack-pii-engine:<version>-cpu`
- `ghcr.io/neurwerk/k8s-stack-pii-engine:<version>-cu124`

Only the full version-specific tags are release contracts; no `latest` or
moving major/minor tags are published. CUDA images include PyTorch and NVIDIA
runtime libraries and require a compatible host driver. Releases and source are
available from the canonical repository linked above.

## License And Security

The project is licensed under the MIT License. See `LICENSE`,
`THIRD_PARTY_NOTICES.md`, and `SECURITY.md`.

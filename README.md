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

## External NER (Next Compatible Release)

`PII_ENGINE_ANALYZER_BACKEND` selects `local` (unchanged default), `remote-gliner`,
or `remote-kserve`. Remote mode keeps Presidio's deterministic and custom
recognizers, policy actions, overlap handling and masking. It disables local NER;
an unavailable external model never becomes a clean local fallback scan. The CPU
image still supplies linguistic support, but remote inference needs no local GPU
or transformer weights. `PRIVATE` is an independent entity, using its explicit
client action or the policy's default action.

Set `PII_ENGINE_REMOTE_CONFIG` to a trusted, read-only JSON file. For GLiNER:

```json
{
  "models": [{
    "name": "multilingual",
    "kind": "gliner",
    "model_name": "ner-multilingual",
    "url": "https://ner.example.test/extract",
    "languages": ["en", "de"],
    "inference_threshold": 0.5
  }]
}
```

The server's actual threshold must be declared; policy `scoreThreshold` must be
at least that value, including in Studio overrides. Default mappings match the
inference manager's ten GLiNER labels. `medical condition` and `medication` map
explicitly to `SENSITIVE_TEXT`; the others retain their normalized category.
For different server labels, supply a complete `label_mapping` object. Unknown
returned labels, invalid offsets/scores and wrong service identities fail closed.
GLiNER runs once per short chunk even with both analysis languages selected.
Overlapping character chunks are subdivided only after an explicit HTTP 413;
unscannable small chunks fail rather than dropping content.

For KServe, configure one or both models, each with one language:

```json
{
  "models": [{
    "name": "english",
    "kind": "kserve",
    "model_name": "ner-english",
    "url": "https://ner.example.test/v1/models/ner-english:predict",
    "languages": ["en"],
    "tokenizer_path": "/remote-tokenizers/english"
  }]
}
```

The exact runtime filenames and SHA-256 pins are supplied by the Engine's model
profile. Copy those files from the same immutable revision used on the server,
including special-token files and SentencePiece assets; retain upstream license
notices with the staged assets. Do not copy weights or add unverified files or
symlinks. If notices are inside the tokenizer directory, add their digests through
`tokenizer_sha256` alongside all the profile's pins. Required runtime digests
cannot be changed. Stage the directories on the optional read-only tokenizer PVC
before adoption; runtime never downloads them. Supported pins are:

| Language | Upstream | Revision |
| --- | --- | --- |
| English | `ai4privacy/llama-ai4privacy-english-anonymiser-openpii` | `1efb619f6d9f5a84b5d6ccf65f1f45df961a2167` |
| German | `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1` | `fa8d9c0186635e1ad74f667d320b9d90955523ad` |

Pins and default label mappings are in `src/pii_engine/lib/remote_models.py`.
English retains `PRIVATE`. German labels with a known normalized meaning keep
it; the explicitly listed categories without an exact equivalent use
`SENSITIVE_TEXT`, never guessed categories or discarded evidence. Review that
entity's action before adoption. A custom mapping replaces the entire default;
every non-`O` class must have a mapping before startup. `PRIVATE` cannot be renamed.
BIO spans retain the strongest constituent confidence, so a weaker continuation
cannot remove stronger sensitive evidence. All tokens, special tokens, class
IDs, finite probabilities and normalization are validated before policy filtering.
Only complete, rechecked windows of at most 512 tokens are sent.

Readiness checks model service metadata, cached for five seconds; inference
still validates each reply. The APIs expose service aliases, not weight revision
attestations, so operators must verify the server assignment matches these pins.
GLiNER's actual server threshold and configured labels must also be verified.

Transport settings are `PII_ENGINE_REMOTE_CALL_TIMEOUT` (10 seconds),
`PII_ENGINE_REMOTE_MAX_CALLS` (2048 across all leaves and selected models),
`PII_ENGINE_REMOTE_MAX_RESPONSE_BYTES` (2 MiB per reply), and
`PII_ENGINE_REMOTE_MAX_CONCURRENT_CALLS` (1 per process). Caller/policy deadlines
also bound queued calls and subdivisions; Studio retains its 30-second ceiling.
HTTP 429 and other errors fail immediately without retries or partial decisions.
Keep one Engine replica or coordinate capacity externally when sharing a single
GLiNER instance: the process limit is not a distributed lock.

HTTPS verifies system trust. Private HTTP requires `allow_private_http: true`
and operator-approved private-network/firewall isolation. Optional `api_key_file`
reads a mounted bearer credential; chart consumers use `apiKeySecretRef` instead
of putting keys in values. KServe's current `API_KEY` environment variable does
not enforce authentication. No request text, response body, detected values or
credentials are logged; proxies from the process environment and redirects are
disabled.

Base exposes these controls under `monitorPiiEngine.analyzerBackend` and
`monitorPiiEngine.remote`. Remote mode needs explicit destination CIDRs/ports;
KServe also needs `tokenizerClaimName`. Local cache mounts and bundle selection
remain unchanged for local clients, and are absent from remote Engine Pods.
Existing model-sync jobs may remain installed independently; remote clients do
not depend on their readiness. Do not remove existing cache PVCs as part of a
mode change. Publish and pin a compatible image before selecting remote mode;
client adoption and deployment are separate approvals.

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

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. The quality gate
installs only the base dependencies and development extra:

```bash
make check
```

This runs the frozen-lock check, Ruff lint/format checks, `ty` type checking,
and pytest with an informational coverage report. `make benchmark` runs the
synthetic benchmark separately. `make build` creates a local CPU image and is
not part of validation.

Validation uses the isolated test analyzer and does not install PyTorch,
Transformers, spaCy, or language-model wheels. Tests cover API and policy
behavior, security boundaries, limits, and model-file verification, not model
inference or recognition quality. Running the production service requires
`uv sync --frozen --extra cpu` or the supported `cu124` extra; release images
still include the complete inference stack and three baseline language models.

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

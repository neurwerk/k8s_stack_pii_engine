# Architecture

## Entry points and ownership

[`main.py`](../../src/pii_engine/main.py) starts two Uvicorn listeners sharing one `EngineRuntime`: the analysis API uses mutual TLS, while the management API exposes health and metrics without application authentication. Analysis routes authorize the exact peer-certificate common name for either the adapter or Studio through [`lib/identity.py`](../../src/pii_engine/lib/identity.py). End-user permission checks belong to the gateway and Studio.

[`controllers/api.py`](../../src/pii_engine/controllers/api.py) defines these interfaces:

| Interface | Input and result |
| --- | --- |
| `/v2/adapter/analyze-segments` | Caller-extracted text segments and scope; returns a decision, transformed segments, entity counts, applied actions, routing/notices, analysis metadata, report rows, and request-local reversal mappings. |
| `/v1/adapter/analyze-request` | Supported OpenAI Chat, Responses, or MCP input; adapts it to segments and restores the transformed request shape. |
| `/v1/adapter/analyze-document-request` | Extracted document text or supported model input; always request-scoped. Supplied visual findings are evaluated by policy, not detected by an image model in the Engine. |
| `/v1/studio/analyze-request`, `/v2/studio/analyze-segments` | Request-local policy previews without the adapter's reversal mapping. |
| `/v1/studio/evaluate-policy`, `/v2/studio/evaluate-policy` | Candidate-policy validation, detection diagnostics, and an echo/restoration simulation that makes no generative model call. |
| `/v1/actions`, `/v1/policy` | Studio-only action and policy metadata. |
| `/v2/adapter/reasoning-receipts` | PUT records scoped digests; POST verifies them against session storage. No reasoning text is accepted by this endpoint. |

The adapter applies the Engine's result to gateway traffic and restores eligible response text using the returned mapping. The Engine does not call the destination language model or enforce gateway routing itself. Segment identity, order, and extraction coverage are therefore important caller responsibilities.

## Processing

1. `main.py` bounds the HTTP body and attaches a correlation ID. API models validate the submitted shape.
2. [`runtime.py`](../../src/pii_engine/runtime.py) validates segment limits, checks dependencies, applies eligible session decisions, and admits work through bounded concurrency and queue limits. Studio has an additional concurrency limit.
3. [`services/policy.py`](../../src/pii_engine/services/policy.py) applies preflight safety and attachment rules. For text analysis, it combines detections and policy-specific patterns into plans over the original segment offsets.
4. [`services/planner.py`](../../src/pii_engine/services/planner.py) resolves overlaps using the strictest action and prepares transformations. Blocking prevents transformed content from being returned; model-routing decisions select a route class. MCP rerouting resolves to blocking instead.
5. The API serializes `pass`, `block`, `apply_actions`, or `reroute`, with the relevant content and metadata. Adapter and Studio evaluation response sizes are bounded separately.

Segments are analyzed independently for text detections. Splitting one personal value across segments can change what is detected. A successful analysis describes the configured detectors' findings; it is not evidence that all personal information has been found.

## Recognition and policy

[`services/layers.py`](../../src/pii_engine/services/layers.py) combines CPU rules with one deployment-selected named-entity recognition (NER) layer. Rules and NER examine original text; the policy planner transforms it afterward. Disabling NER leaves rules and policy processing enabled.

[`config/ner.py`](../../src/pii_engine/config/ner.py) supports explicit `disabled`, `local`, and `remote` modes. Local profiles use bundled spaCy models. Remote profiles use GLiNER or KServe endpoints; KServe also requires profile-matched offline tokenizer files. Remote NER sends text windows to the configured inference service. Profile metadata and readiness responses do not attest the server's actual weights or detection quality.

Without explicit NER configuration, [`services/analyzer.py`](../../src/pii_engine/services/analyzer.py) retains legacy backend selection: local baseline models, a verified transformer bundle when activated, or a configured remote backend. A missing desired-bundle reference permits the local baseline; an invalid selected bundle is rejected. Production inference dependencies are supplied by the image's `inference` extra. The test analyzer is only an isolated test substitute.

Policy YAML supplies entity actions, thresholds, safety patterns, notices, classification, routing, and session settings. [`config/policy.py`](../../src/pii_engine/config/policy.py) validates the file and merges Studio overrides for a single request. Overrides do not mutate the live policy or select different deployment models.

## State and failures

[`services/session.py`](../../src/pii_engine/services/session.py) uses Valkey when policy sessions are enabled. It stores block/reroute metadata and reasoning digests under policy-version namespaces, not prompts or reversal mappings. Session-scoped adapter requests require a trusted session key; request-scoped analysis and Studio previews bypass decision reuse and persistence.

[`services/ner_cache.py`](../../src/pii_engine/services/ner_cache.py) is a separate, optional, process-local optimization. It retains findings under keyed text-window digests with a byte-accounted LRU and expiry budget. Rules and current policy still execute. Cache failures fall back to detection, while inference failures propagate.

Missing required configuration or unavailable startup dependencies prevent normal startup. Runtime readiness includes selected analyzer health and enabled Valkey state. Analysis errors produce an error response rather than a successful clean-scan result: invalid requests use 400, size limits 413, unavailable runtime or capacity 503, runtime analysis timeouts 504, and other failures 500. Remote inference failures can reach the generic 500 path; they are not all classified as retryable timeouts.

Timed-out CPU work retains its admission slot until its worker finishes. The HTTP timeout does not terminate that worker. These Engine behaviors do not by themselves establish how a gateway handles errors; that behavior belongs to the processor and its deployment configuration.

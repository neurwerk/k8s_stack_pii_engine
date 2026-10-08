# neurwerk.base - PII Engine

PII Engine uses Presidio to evaluate personally identifiable information and apply safety policies in neurwerk.base. See the [neurwerk.base website](https://base.neurwerk.com/) for more information.

| Repository | Description |
| --- | --- |
| [Base chart](https://github.com/neurwerk/k8s_stack_base) | Shared platform charts and release packages that form the foundation of the stack. |
| [Studio](https://github.com/neurwerk/k8s_stack_studio) | Web dashboard and API for operating AI platform services. |
| [Tooling](https://github.com/neurwerk/k8s_stack_tooling) | One container image plus separate CLI tools for setup and operations. |
| [PII Engine](https://github.com/neurwerk/k8s_stack_pii_engine) | Service that uses Presidio to evaluate PII and apply safety policies (**this repo**). |
| [AgentGateway External Processor](https://github.com/neurwerk/k8s_stack_agentgateway_extproc) | Adapter that processes gateway requests and responses with the PII Engine. |
| [Keycloak API Key Bridge](https://github.com/neurwerk/k8s_stack_keycloak_api_key_bridge) | Separate service that issues and validates API keys using Keycloak permissions. |
| [Keycloak Theme](https://github.com/neurwerk/k8s_stack_keycloak_theme) | Customized Keycloak login pages and emails. |
|  |  |
| [Example client chart](https://github.com/neurwerk/k8s_stack_client_example_com) | Reference client configuration and Flux deployment setup to adapt for a new client. |
|  |  |
| [Dify Add-on](https://github.com/neurwerk/k8s_stack_addon_dify) | Optional Dify package with API and web customizations, including single-workspace enforcement. |

## Contributing and support

- **Contributions:** Read [CONTRIBUTING.md](.github/CONTRIBUTING.md) before proposing a change.
- **Bug reports and feature requests:** Use [GitHub Issues](https://github.com/neurwerk/k8s_stack_pii_engine/issues) for reproducible bugs and clearly scoped feature requests.

## Security

Report vulnerabilities privately by following the instructions in [SECURITY.md](SECURITY.md).

## Licensing

Project-owned content is licensed under the [MIT License](LICENSE). Third-party content retains its upstream license.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance and licensing information for dependencies and model assets.

## Release image and validation

PII Engine publishes one `linux/amd64` image:
`ghcr.io/neurwerk/k8s-stack-pii-engine:<version>`. The Engine runs on CPU,
including local NER and transformer bundles. Remote inference services may use
GPUs independently; the Engine does not select their hardware.

After a reviewed release change is merged, authorized publication uses GitHub
Actions from the exact `v<version>` Git tag. The tag must match `pyproject.toml`
and `uv.lock`. CI publishes no `latest` tag and records the image digest in the
GitHub Release. A retry reuses an existing tag only after verifying its platform,
source revision and version; a mismatch stops publication without overwriting it.
Image publication, verified Base pin adoption and deployment are separate actions.

`make check` installs base and development dependencies only, without model
inference dependencies. Production startup requires `uv sync --frozen --extra
inference`; the release image includes the locked inference dependencies and
offline English, German and Dutch spaCy models. `make build` builds a local
image without publishing it.

## Independent rules and NER configuration

Production analysis runs CPU rules/checksums/custom recognizers over each original
segment, combines their evidence with exactly one NER mode, then applies policy
once. `entityPolicies[].patterns` remain at the shared policy seam and run once;
they are not duplicated in either detector. Remote failure never becomes a
rules-only success or local fallback.

Set `PII_ENGINE_NER_CONFIG` to a JSON or YAML file containing:

```yaml
mode: local
languageModels: {en: english-spacy, de: german-spacy, nl: dutch-spacy}
models:
  english-spacy: {profile: spacy-en-sm-v1}
  german-spacy: {profile: spacy-de-sm-v1}
  dutch-spacy: {profile: spacy-nl-sm-v1}
```

Use `mode: disabled` without model fields to retain rules only. Local pipelines
are loaded only for deployment-selected languages. Policy language overrides
cannot load additional models, even for an empty segment.

```yaml
mode: remote
languageModels: {en: multilingual-pii, de: multilingual-pii}
models:
  multilingual-pii:
    profile: gliner-multilingual-pii-v1
    endpoint: https://ner.example.com/extract
    inferenceThreshold: 0.45
capacity:
  callTimeout: 10
  maxCalls: 2048
  maxResponseBytes: 2097152
  maxConcurrentCalls: 1
```

The GLiNER profile defaults to `urchade/gliner_multi_pii-v1`, server-enforced
384-word/512-token ceilings including server-managed prompts, and overlapping
1024-character client windows. A declared optional `revision` is not remotely
attested; an omitted revision is explicitly unattested. No download or readiness
probe can prove the server's weights. `modelName` defaults to `ner-multilingual`;
the configured server confidence floor must not exceed the policy threshold.

Remote models may mix GLiNER and KServe profiles. The reviewed KServe profiles
are `kserve-en-openpii-v1` and `kserve-de-superclinical-v1`; each requires explicit
`modelName` and a verified offline `tokenizerPath`. Profile data in `config/ner.py`
owns revisions, labels, file digests, input windows, overhead and batch size (1).
New profiles for an existing adapter are data-only additions; new protocols need
code review. Shared multilingual model IDs are called once per chunk.

Optional remote fields are `upstream`, `revision`, `apiKeyFile`,
`allowPrivateHttp` (false by default), and `tokenizerPath` for KServe. Pinned
profile identities cannot be overridden. HTTPS verifies system trust; private
HTTP requires explicit isolation approval. Credentials remain files from Secrets.
Charts remove `apiKeySecretRef`, `egress`, and `tokenizerClaimName` before writing
the Engine document. All serialized configuration fields use camelCase.

Without `PII_ENGINE_NER_CONFIG`, legacy backend/bundle configuration retains its
baseline-to-verified-transformer transition. Canonical and explicitly supplied
legacy runtime selectors are rejected together. Canonical policy must omit legacy
`pii.ner` (or use its neutral defaults). This source contract needs an Engine
release supporting it; existing 0.12.0 images do not support this configuration.

Operational metrics include detector durations (`rules`, `ner`, `wait`) and
chunk counts, with no text, endpoints or model identities as metric labels.

### Process-only NER window cache

Local spaCy/transformer and remote GLiNER/KServe NER reuse successful findings for
exact, unchanged model windows, including their overlap/context. Independent text
leaves and existing window recipes are unchanged. CPU rules, request/model
validation, current policy filtering, and anonymization still run normally.
Empty successful findings are reusable; errors and incomplete windows are not.
Adaptive GLiNER size rejection still subdivides with complete coverage.

Defaults are `PII_ENGINE_NER_CACHE_ENABLED=true`,
`PII_ENGINE_NER_CACHE_MAX_BYTES=134217728` (128 MiB), and
`PII_ENGINE_NER_CACHE_TTL_SECONDS=86400` (24 hours). This is one budget shared by
all selected languages/models in the runtime, not one budget per backend.
LRU eviction may remove findings earlier; hits never extend the creation-based
lifetime. Runtime startup/shutdown manages approximately 60-second idle expiry
cleanup. Disabling the cache restores ordinary detection.

Only immutable normalized relative spans (entity, score, provenance) and
process-random HMAC digests are retained, never input text, NLP artifacts,
mutable Presidio results, reversal mappings, or exceptions. A fresh namespace
identifies each immutable loaded inference configuration; language and local
inference threshold distinguish local keys, and remote model configuration
distinguishes remote keys. Remote findings are cached before policy filtering,
so Studio threshold/entity/action changes use current policy. Current offsets,
ownership and overlap merging are reapplied without mutating cached findings.
Accesses are thread-safe without locking inference; concurrent misses may repeat
inference. Cache failures fall back to detection and oversized entries are skipped.

The accounted-memory bound sums Python sizes of the findings tuple, each span,
its attribute dictionary and values, plus 1 KiB per entry for the digest/key and
cache bookkeeping. It is **not an exact process RSS limit**: allow additional
headroom for Python allocator/container slack, model memory, temporary inputs,
in-flight results and concurrent inference. Content-free cache event counters
(`hit`, `miss`, `eviction`, `expiry`, `error`) and accounted-byte gauge are exposed
as `pii_engine_ner_cache_events_total` and `pii_engine_ner_cache_bytes`.

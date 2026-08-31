# Neurwerk PII Engine

Bounded, deterministic PII and safety policy evaluation for supported LLM and
MCP requests. The service analyzes structured requests with Microsoft Presidio,
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
and pytest with the configured coverage threshold. `make benchmark` runs the
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

# Operations

## Configuration ownership

The [Base PII Engine chart](https://github.com/neurwerk/k8s_stack_base/tree/main/charts/pii-engine) owns Kubernetes resources and deployment configuration. Its [values](https://github.com/neurwerk/k8s_stack_base/blob/main/charts/pii-engine/values.yaml) and [deployment template](https://github.com/neurwerk/k8s_stack_base/blob/main/charts/pii-engine/templates/deployment.yaml) are the source for image selection, environment mappings, mounts, and probes. Use the selected platform revision when assessing a deployment.

[`config/settings.py`](../../src/pii_engine/config/settings.py) reads process settings from `PII_ENGINE_` environment variables. Settings and policy are loaded at process initialization; deployment changes require a rollout rather than an API policy update.

| Configuration | Purpose and prerequisites |
| --- | --- |
| `PII_ENGINE_POLICY_CONFIG` | Path to validated policy YAML, rendered by the chart from policy values. Required outside test mode. |
| `PII_ENGINE_NER_CONFIG` | Optional explicit NER YAML. Cannot be combined with explicit legacy backend/model settings. See the [chart's NER guide](https://github.com/neurwerk/k8s_stack_base/blob/main/charts/pii-engine/README.md) for mode, endpoint, egress, and tokenizer configuration. |
| `PII_ENGINE_HASH_KEY`, `PII_ENGINE_ENCRYPTION_KEY` | Required production values supplied through the runtime Secret. Validation requires at least 32 characters for the hash key and exactly 32 for the encryption key. |
| `PII_ENGINE_TLS_CERT`, `PII_ENGINE_TLS_KEY`, `PII_ENGINE_TLS_CA` | Server certificate, key, and client CA files; all are required by the service CLI. |
| `PII_ENGINE_ADAPTER_COMMON_NAME`, `PII_ENGINE_STUDIO_COMMON_NAME` | Exact authorized client-certificate names. Keep `PII_ENGINE_ENFORCE_CLIENT_IDENTITY` enabled for workload authorization. |
| `PII_ENGINE_VALKEY_URL` | Required when policy `session.enabled` is true; startup connects and pings Valkey. |
| `PII_ENGINE_POLICY_VERSION` | Labels responses and scopes session state and placeholder derivation. Coordinate changes with callers. |
| `PII_ENGINE_MAX_*`, concurrency and timeout settings | Independent request, response, text, receipt, admission, and execution limits. Exact fields and ranges are defined in `Settings`. |
| `PII_ENGINE_NER_CACHE_*` | Enable and bound the optional process-memory NER findings cache. Its accounted bytes are not total process memory. |

The chart mounts policy and NER configuration under `/etc/pii-engine`, TLS material under `/var/run/pii-engine/tls`, and selected model/tokenizer resources read-only. Runtime keys and dependency credentials come from Secrets. Credential setup belongs to the shared [OpenBao procedure](https://github.com/neurwerk/docs/blob/main/dev/operations/openbao.md).

## Verification and diagnosis

The management listener defaults to port 8001 and the analysis listener to 8443. The chart uses management HTTP probes; management routes do not perform workload authentication.

| Check | Meaning and next step |
| --- | --- |
| `GET /live` on management | Process liveness. In legacy baseline mode, activation of the desired verified transformer bundle returns 503 to request a restart. It is not a dependency-health check. |
| `GET /ready` on management | Returns the active analyzer mode when runtime dependencies are ready. On 503, examine selected analyzer health, model-cache availability, and enabled Valkey connectivity. |
| `GET /v2/adapter/ready` on analysis | Requires the adapter certificate, checks runtime readiness, and confirms v2 capability. A 403 indicates workload identity rejection. |
| `GET /metrics` on management | Exposes Prometheus counters and timings for decisions, failures, queue rejection, inference stages, and cache activity; names are defined in [`metrics.py`](../../src/pii_engine/metrics.py). |

For a 413, inspect the returned limit stage, reason, measured value, and maximum when present; request bytes, segment counts, characters, reasoning digests, and response bytes have separate bounds. For a capacity 503, inspect queue rejection and analysis duration together. A 504 does not imply its CPU worker has stopped. Remote inference errors may instead return 500; use the correlation ID and bounded failure metadata to identify the failing boundary.

For startup failure, first compare supplied settings and mounted-file paths with their validators. For remote readiness failure, verify the configured service's health metadata and model name with its owner. Restore the selected dependency or correct its configuration; the runtime does not automatically substitute local NER after a selected remote service fails.

## Release and recovery

The [publication workflow](../../.github/workflows/build-push.yaml) runs quality checks for pull requests and `main`. Version tags additionally verify the package version, publish a `linux/amd64` image to `ghcr.io/neurwerk/k8s-stack-pii-engine`, and record the digest in a GitHub Release. [`pyproject.toml`](../../pyproject.toml) owns the package version; Base owns the deployed image pin.

Follow shared [image release procedures](https://github.com/neurwerk/docs/blob/main/dev/operations/image-releases.md) for publication and pin changes, and [platform upgrades and recovery](https://github.com/neurwerk/docs/blob/main/dev/operations/upgrades.md) for adoption. Recovery must account for the selected NER resources and Valkey session state. Restarting clears only the process-local NER cache; it is not a reset of Valkey decisions or reasoning receipts.

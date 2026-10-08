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

## Manual image publication

After the release change is merged, run from the clean canonical checkout:

```bash
./deploy.sh --help
./deploy.sh 0.13.0
```

Select CPU, CUDA 12.4 (`cu124`), or both; choose GHCR push or local load, then
confirm `Proceed`. Builds use the existing Dockerfile and locked dependencies,
target only `linux/amd64`, and use Docker context `desktop-linux` unless
`DOCKER_CONTEXT` is set. Python 3.11+, Git and Docker Buildx are required.
The version must match both `pyproject.toml` and `uv.lock`.

Push requires clean HEAD equal to freshly fetched canonical `origin/main`.
Use existing Docker authentication or choose login with `GHCR_USERNAME` and
`GHCR_TOKEN` (missing credentials are prompted; token input is hidden).
All selected tags are checked before any build. Existing tags stop publication;
deselect already-published variants to resume. A small anonymous GHCR manifest
preflight accepts only HTTP 404 with explicit `MANIFEST_UNKNOWN` / `NAME_UNKNOWN`
JSON errors, not Docker's ambiguous `not found` text. Authentication, transport
and other registry errors stop publication. This read-only check does not prove
push permission; publication uses your Docker credentials.
Local load skips login and registry checks, but still requires a clean checkout.
Both modes build a committed-file snapshot, excluding ignored workstation files.

Results include the source commit and digest-pinned `0.13.0-cpu` / `0.13.0-cu124`
references; a locally loaded result is not proof of registry publication.
For post-publication status, use `uv run package-checker --json` from
`tooling/cli_tools/package_checker` in the workspace. Independently verify images
before adopting platform pins. The script does not create Git tags or GitHub
Releases, update Base, or deploy to a cluster. Tag-triggered CI publication remains
separate; it must not race manual publication of the same immutable tags.

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

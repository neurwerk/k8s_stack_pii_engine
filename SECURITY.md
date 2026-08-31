# Security Policy

## Supported Versions

Security fixes are provided for the latest released version. Older releases may
be unsupported; upgrade to the newest release before requesting a backport.

## Reporting A Vulnerability

Report vulnerabilities privately through the repository's
[GitHub security advisory form](https://github.com/neurwerk/k8s_stack_pii_engine/security/advisories/new).
Do not open a public issue and do not include real personal data, credentials,
private keys, model-store tokens, or production request payloads.

Include the affected version, impact, reproduction steps using synthetic data,
and any suggested mitigation. Maintainers will acknowledge the report, assess
severity and affected versions, and coordinate remediation and disclosure.
Please allow a reasonable remediation window before public disclosure.

## Operational Guidance

Run the analysis API behind mutual TLS, keep the management listener private,
inject secrets at runtime, and leave `PII_ENGINE_LOG_LEVEL=INFO` in normal use.
Third-party exceptions can contain sensitive request text; use `DEBUG` only for
short, controlled diagnostics with synthetic data.

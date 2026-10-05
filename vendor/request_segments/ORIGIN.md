# Shared request extraction

This source snapshot comes from `packages/request_segments/` in the public
`neurwerk/k8s_stack_agentgateway_extproc` repository. That directory owns this
package. Keep consumer snapshots synchronized; do not maintain a separate fork.

Version: `0.1.0` (unpublished implementation snapshot).

Snapshot includes only `pyproject.toml`, `README.md`, `LICENSE`, and `src/**/*.py`.
The final canonical source includes configurable extraction depth, strict direct
models, MCP omission-preserving schemas, and provider-owned token-limit controls.

Source SHA-256: `2a7006a9c37f73b4f0f96d7b973dee8ea132c880da1d0fe81d85680b61377a89`.
For each relative POSIX path in sorted order, hash its UTF-8 bytes, a NUL byte,
the file bytes, and another NUL byte. Exclude this consumer provenance file.

The Engine uses this lightweight package only at its legacy request boundaries.
The v2 policy core accepts text segments and never accepts provider request JSON.

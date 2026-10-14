# Shared request extraction

This source snapshot comes from `packages/request_segments/` in the public
`neurwerk/k8s_stack_agentgateway_extproc` repository. That directory owns this
package. Keep consumer snapshots synchronized; do not maintain a separate fork.

Package version: `0.1.0` (vendored source snapshot).
Source commit: `64b72b4796696e9c4a9a436610f1e86788437b1c` (extProc PR #82).
Source tree (`packages/request_segments`): `9a6bd46360cd1ba872f27435d50e71190ec21a9f`.

Snapshot includes only `pyproject.toml`, `README.md`, `LICENSE`, and `src/**/*.py`.
It includes canonical native Anthropic/Google extraction and reconstruction.

Source SHA-256: `739d08f23db03d357d48730d0db03a4a48a0c9328be8a4769c3740750d8487c6`.
For each relative POSIX path in sorted order, hash its UTF-8 bytes, a NUL byte,
the file bytes, and another NUL byte. Exclude this consumer provenance file.

Per-file SHA-256:

```text
527229e5c39adcb292bf718cef64a2e87526b37f6e97c42741812bf9e73828e1  LICENSE
67fa957d373521c0f17cb3895a608b7710850ba48e178a230ef77a839c606d46  README.md
e639bbada63dfaccf62464353cebec68bf20bc7f12a1427e8707d45ed7a58315  pyproject.toml
23faf437b5fbf340f5758c51c8d8f317027bc6a8e7c10a12c77828b50bd398b2  src/neurwerk_request_segments/__init__.py
fb0a2f22b8aac0b09b5e0b4e5c637f1dd0ca67135bcf8f1d45af232b9fd700d0  src/neurwerk_request_segments/compatibility.py
ceca08ca19d4342bf7d7a22a09abfca15e7c456bcef6ab1e8a3bda917a365c2b  src/neurwerk_request_segments/extraction.py
211bdb6159f7f8dbeed9162123fb7f6371be683d948d32964941975b6a0453c9  src/neurwerk_request_segments/models.py
7068dc27b1bfc0c079763451ccfbaa3a2d287f6f6947624521ac4f8f02cffea6  src/neurwerk_request_segments/native.py
```

The Engine uses this lightweight package only at its legacy request boundaries.
Its explicit legacy request union remains Chat, Responses and MCP only. Native
requests use the existing v2 `chat` analysis family with `{id, text}` segments;
the v2 policy core never accepts provider request JSON. Reversal mappings remain
request-scoped.

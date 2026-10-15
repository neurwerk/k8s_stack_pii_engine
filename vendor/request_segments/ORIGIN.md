# Shared request extraction

This source snapshot comes from `packages/request_segments/` in the public
`neurwerk/k8s_stack_agentgateway_extproc` repository. That directory owns this
package. Keep consumer snapshots synchronized; do not maintain a separate fork.

Package version: `0.1.1` (vendored source snapshot).
Source commit: `3e4b1478d585d06ba1b1370c2683df3be2276508` (extProc PR #89).
Source tree (`packages/request_segments`): `84e72a1a4fc0ae3b4e7d7eae37bd0884da268f81`.

Snapshot includes only `pyproject.toml`, `README.md`, `LICENSE`, and `src/**/*.py`.
It includes canonical native Anthropic/Google extraction and reconstruction.

Source SHA-256: `c6af208a268b4c5d932d1ec1225a52b8356170c12a3567e2bd90be3b2c64e597`.
For each relative POSIX path in sorted order, hash its UTF-8 bytes, a NUL byte,
the file bytes, and another NUL byte. Exclude this consumer provenance file.

Per-file SHA-256:

```text
527229e5c39adcb292bf718cef64a2e87526b37f6e97c42741812bf9e73828e1  LICENSE
84510c3a2e0c8220abde7780250149b859fe1f6dd3ed3087e267080a18b9bb56  README.md
70c81ebe43957f5a9ed0a04d7da66b232c177dc41431943af34ae9bda7403b85  pyproject.toml
23faf437b5fbf340f5758c51c8d8f317027bc6a8e7c10a12c77828b50bd398b2  src/neurwerk_request_segments/__init__.py
fb0a2f22b8aac0b09b5e0b4e5c637f1dd0ca67135bcf8f1d45af232b9fd700d0  src/neurwerk_request_segments/compatibility.py
ceca08ca19d4342bf7d7a22a09abfca15e7c456bcef6ab1e8a3bda917a365c2b  src/neurwerk_request_segments/extraction.py
2113b8ceee07a9fb6a664ebd912c2b732b2712d4d8a2709efaea2a8cdac917cb  src/neurwerk_request_segments/models.py
74b95bfbad42df372bf1d261a992c80ab3fa0841672944b910794b53ac4c6359  src/neurwerk_request_segments/native.py
```

The Engine uses this lightweight package only at its legacy request boundaries.
Its explicit legacy request union remains Chat, Responses and MCP only. Native
requests use the existing v2 `chat` analysis family with `{id, text}` segments;
the v2 policy core never accepts provider request JSON. Reversal mappings remain
request-scoped.

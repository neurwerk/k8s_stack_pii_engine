# Development

## Prerequisites and checks

Run commands from the repository or worktree root containing `pyproject.toml`. Use Python 3.12, `uv`, and GNU Make; exact CI tool pins are in the [workflow](../../.github/workflows/build-push.yaml). The local `vendor/request_segments` package is resolved through [`pyproject.toml`](../../pyproject.toml), so keep it in the checkout.

```bash
make check
```

[`Makefile`](../../Makefile) checks the lockfile, runs Ruff lint and format validation, runs `ty`, then runs pytest with coverage. `uv run --frozen --extra dev` installs the development dependencies as needed. Production inference packages are deliberately separate from this check path.

[`tests/conftest.py`](../../tests/conftest.py) selects the test analyzer and in-process HTTP clients. The [mutual-TLS tests](../../tests/test_mtls_integration.py) additionally use temporary certificates and loopback sockets. Remote inference and storage tests use substitutes; passing these checks does not validate a deployed model service.

Useful focused checks, from the same directory:

```bash
uv run --frozen --extra dev pytest tests/test_api_errors.py tests/test_segments.py
make benchmark
```

The benchmark uses [`benchmarks/run_synthetic.py`](../../benchmarks/run_synthetic.py); its synthetic results are not production inference measurements. Relevant behavior tests include [policy evaluation](../../tests/test_studio_evaluation.py), [NER layers](../../tests/test_ner_layers.py), and [session compatibility](../../tests/test_session_compatibility.py).

## Running and building

The installed `pii-engine` entry point starts both listeners. It requires the configured policy, runtime keys, TLS files, and selected inference dependencies described in [Operations](../operations/README.md). The test-analyzer flag alone does not make this CLI an unsecured local server: the CLI still requires TLS configuration.

For an image build, Docker must have a working `desktop-linux` context and support `linux/amd64`. From the repository root:

```bash
make build
```

The [Dockerfile](../../Dockerfile) installs the `inference` extra, bundles local spaCy models, sets offline model-loading flags, and runs as a non-root user. The build target produces `pii-engine:local`; it does not publish it. Publication and deployment follow the [operations guide](../operations/README.md).

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REVISION = "a" * 40
IMAGE = "ghcr.io/neurwerk/k8s-stack-pii-engine:0.13.0"


@pytest.fixture
def builder(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    shutil.copyfile(ROOT / "deploy.sh", checkout / "deploy.sh")
    (checkout / "pyproject.toml").write_text('[project]\nname = "example"\nversion = "0.13.0"\n')
    (checkout / "uv.lock").write_text(
        '[[package]]\nname = "example"\nversion = "0.13.0"\nsource = { editable = "." }\n'
    )
    (checkout / "scripts").mkdir()
    shutil.copyfile(
        ROOT / "scripts/verify_release_tag.py", checkout / "scripts/verify_release_tag.py"
    )
    archive = tmp_path / "context.tar"
    with tarfile.open(archive, "w") as context:
        data = b"FROM scratch\n"
        entry = tarfile.TarInfo("Dockerfile")
        entry.size = len(data)
        context.addfile(entry, io.BytesIO(data))
    commands = tmp_path / "bin"
    commands.mkdir()
    fake = commands / "fake"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "name = pathlib.Path(sys.argv[0]).name\n"
        "args = sys.argv[1:]\n"
        "if name == 'python3' and args[0] != 'scripts/ghcr_preflight.py':\n"
        "    os.execv(sys.executable, [sys.executable, *args])\n"
        "with open(os.environ['COMMAND_LOG'], 'a') as log:\n"
        "    log.write(json.dumps([name, *args]) + '\\n')\n"
        "if name == 'git':\n"
        "    if args[0] == 'rev-parse':\n"
        "        print(os.environ.get('MAIN_REVISION', 'a' * 40)\n"
        "              if args[-1] != 'HEAD' else 'a' * 40)\n"
        "    elif args[0] == 'status':\n"
        "        print(os.environ.get('DIRTY', ''), end='')\n"
        "    elif args[0] == 'remote':\n"
        "        print(os.environ.get('ORIGIN', 'https://github.com/neurwerk/k8s_stack_pii_engine.git'))\n"
        "    elif args[0] == 'archive':\n"
        "        sys.stdout.buffer.write(pathlib.Path(os.environ['ARCHIVE']).read_bytes())\n"
        "elif name == 'python3':\n"
        "    for tag in args[1:]:\n"
        "        outcome = os.environ.get('REGISTRY_' + tag.split('-')[-1], 'missing')\n"
        "        if outcome != 'missing':\n"
        "            sys.exit(1)\n"
        "elif name == 'docker':\n"
        "    args = args[2:]\n"
        "    if args[0] == 'login':\n"
        "        assert sys.stdin.read() == 'secret-test-token'\n"
        "    elif args[:2] == ['buildx', 'build']:\n"
        "        if os.environ.get('BUILD_FAIL'):\n"
        "            sys.exit(1)\n"
        "        context = pathlib.Path(args[-1])\n"
        "        assert [p.name for p in context.iterdir()] == ['Dockerfile']\n"
        "        metadata = pathlib.Path(args[args.index('--metadata-file') + 1])\n"
        "        digest = 'invalid' if os.environ.get('BAD_DIGEST') else 'sha256:' + 'b' * 64\n"
        "        metadata.write_text(json.dumps({'containerimage.digest': digest}))\n"
    )
    fake.chmod(0o755)
    for name in ("git", "docker", "python3"):
        (commands / name).symlink_to(fake)
    log = tmp_path / "commands.jsonl"
    env: dict[str, str] = {
        **os.environ,
        "PATH": f"{commands}:{os.environ['PATH']}",
        "COMMAND_LOG": str(log),
        "ARCHIVE": str(archive),
        "TMPDIR": str(tmp_path),
    }
    # Never inherit workstation credentials or Docker context in offline tests.
    for name in ("GHCR_USERNAME", "GHCR_TOKEN", "DOCKER_CONTEXT"):
        env.pop(name, None)

    def run(answers: str = "", version: str | None = "0.13.0", **overrides: str):
        command = ["/bin/bash", str(checkout / "deploy.sh")]
        if version is not None:
            command.append(version)
        run_env: dict[str, str] = {**env, **overrides}
        result = subprocess.run(
            command,
            input=answers,
            text=True,
            capture_output=True,
            env=run_env,
            check=False,
        )
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls

    return run


def test_cancel_and_help_have_no_docker_or_network_calls(builder):
    result, calls = builder("\n\n\n\n")
    assert result.returncode == 0
    assert "Cancelled" in result.stdout
    assert calls == [["git", "rev-parse", "HEAD"]]
    result, calls = builder(version="--help")
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert calls == [["git", "rev-parse", "HEAD"]]
    result, calls = builder(version=None)
    assert result.returncode == 2
    assert "Usage:" in result.stderr
    assert calls == [["git", "rev-parse", "HEAD"]]


def test_local_load_uses_snapshot_without_registry_or_login(builder):
    result, calls = builder("\nn\nn\ny\n", DOCKER_CONTEXT="test-local")
    assert result.returncode == 0, result.stderr
    docker = [call for call in calls if call[0] == "docker"]
    assert all(call[1:3] == ["--context", "test-local"] for call in docker)
    assert len(docker) == 2  # info and one local build only
    build = docker[-1]
    assert "--load" in build and "--push" not in build
    assert f"{IMAGE}-cpu" in build
    assert not any(call[1] in ("fetch", "remote") for call in calls)
    assert f"{IMAGE}-cpu@sha256:" in result.stdout


def test_push_preflights_both_variants_before_build_with_exact_labels(builder):
    result, calls = builder("\n\n\ny\nn\n")
    assert result.returncode == 0, result.stderr
    docker = [call for call in calls if call[0] == "docker"]
    preflight = ["python3", "scripts/ghcr_preflight.py", "0.13.0-cpu", "0.13.0-cu124"]
    assert preflight in calls
    assert calls.index(preflight) < calls.index(docker[1])
    for build, variant in zip(docker[1:], ("cpu", "cu124"), strict=True):
        assert build[3:5] == ["buildx", "build"]
        assert "--push" in build and "--load" not in build
        assert build[build.index("--platform") + 1] == "linux/amd64"
        assert f"ACCELERATOR={variant}" in build
        assert f"{IMAGE}-{variant}" in build
        assert f"org.opencontainers.image.revision={REVISION}" in build
        assert f"org.opencontainers.image.version=0.13.0-{variant}" in build
        source_label = (
            "org.opencontainers.image.source=https://github.com/neurwerk/k8s_stack_pii_engine"
        )
        assert source_label in build
    assert [
        "git",
        "fetch",
        "--quiet",
        "--no-tags",
        "origin",
        "refs/heads/main:refs/remotes/origin/main",
    ] in calls


@pytest.mark.parametrize("outcome", ["exists", "unauthorized", "connection refused", "not found"])
def test_second_target_failure_prevents_every_build(builder, outcome):
    result, calls = builder("\n\n\ny\nn\n", REGISTRY_cu124=outcome)
    assert result.returncode != 0
    assert not any(call[3:5] == ["buildx", "build"] for call in calls)
    assert not any(call[1] == "archive" for call in calls)


@pytest.mark.parametrize(
    "overrides",
    [
        {"DIRTY": " M README.md"},
        {"MAIN_REVISION": "c" * 40},
        {"ORIGIN": "https://github.com/example/fork.git"},
    ],
)
def test_push_rejects_unreviewed_source_before_docker(builder, overrides):
    result, calls = builder("\n\n\ny\nn\n", **overrides)
    assert result.returncode != 0
    assert not any(call[0] == "docker" for call in calls)


def test_version_mismatch_stops_before_git_or_docker(builder):
    result, calls = builder(version="0.12.0")
    assert result.returncode != 0
    assert "does not match project version" in result.stderr
    assert calls == []


def test_no_variants_selected_is_safe_with_bash_32_empty_arrays(builder):
    result, calls = builder("n\nn\n")
    assert result.returncode != 0
    assert "No image selected" in result.stderr
    assert "unbound variable" not in result.stderr
    assert not any(call[0] in ("docker", "python3") for call in calls)


def test_optional_login_keeps_token_out_of_arguments_and_output(builder):
    result, calls = builder(
        "n\n\n\ny\ny\n", GHCR_USERNAME="example", GHCR_TOKEN="secret-test-token"
    )
    assert result.returncode == 0, result.stderr
    assert [
        "docker",
        "--context",
        "desktop-linux",
        "login",
        "ghcr.io",
        "--username",
        "example",
        "--password-stdin",
    ] in calls
    assert "secret-test-token" not in result.stdout + result.stderr + json.dumps(calls)
    assert not any(f"{IMAGE}-cpu" in call for call in calls)


@pytest.mark.parametrize("failure", ["BUILD_FAIL", "BAD_DIGEST"])
def test_failed_build_or_invalid_digest_never_reports_success(builder, failure):
    result, _ = builder("\nn\nn\ny\n", **{failure: "1"})
    assert result.returncode != 0
    assert "Result (" not in result.stdout
    assert "Completed." not in result.stdout


def test_script_is_executable_and_valid_bash():
    assert os.access(ROOT / "deploy.sh", os.X_OK)
    subprocess.run(["/bin/bash", "-n", str(ROOT / "deploy.sh")], check=True)

"""Run bounded no-model performance and fail-closed regression probes."""

from __future__ import annotations

import argparse
import asyncio
import gc
import logging
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import yaml
from redis.exceptions import RedisError

from benchmarks.contracts import (
    BenchmarkReport,
    ColdStartResult,
    FailureChecks,
    LatencyResult,
    MemoryProbe,
    Thresholds,
    evaluate_thresholds,
)
from pii_engine.config.policy import test_policy
from pii_engine.config.settings import Settings
from pii_engine.models.contracts import OpenAIChatRequest, SupportedRequest
from pii_engine.runtime import EngineRuntime, RuntimeNotReadyError
from pii_engine.services.limiter import AnalysisCapacityError
from pii_engine.services.policy import PolicyResult, PolicyService
from pii_engine.services.session import SessionStore

ROOT = Path(__file__).resolve().parent


class _BlockingPolicy:
    """Hold one worker so the bounded queue can be measured."""

    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        self.started = started
        self.release = release

    def analyze(self, request: SupportedRequest) -> PolicyResult:
        """Block until the benchmark releases the worker."""
        self.started.set()
        self.release.wait(timeout=5)
        return PolicyResult(request=request, decision="pass", remote_allowed=True)


class _FailingPolicy:
    """Raise one deterministic processing failure."""

    def analyze(self, request: SupportedRequest) -> PolicyResult:
        """Simulate an internal engine failure before forwarding."""
        raise RuntimeError("synthetic analysis failure")


def _settings(**overrides: object) -> Settings:
    """Create an environment-isolated dependency-free runtime configuration."""
    values: dict[str, object] = {
        "allow_test_analyzer": True,
        "policy_config": None,
        "model_cache_path": None,
        "model_bundle_reference": None,
        "model_bundle_version": None,
        "model_manifest_sha256": None,
        "valkey_url": None,
        "max_concurrent_analyses": 4,
        "max_queued_analyses": 32,
        "studio_max_concurrent_analyses": 1,
    }
    values.update(overrides)
    return Settings.model_validate(values)


def _request(characters: int) -> OpenAIChatRequest:
    """Build one safe request with an exact text size."""
    content = ("safe context " * (characters // 13 + 1))[:characters]
    return OpenAIChatRequest(model="benchmark", messages=[{"role": "user", "content": content}])


def _percentile(values: list[float], quantile: float) -> float:
    """Return a nearest-rank percentile from a non-empty sample."""
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * quantile) - 1)
    return ordered[index]


def _rss_mib() -> float:
    """Return current resident memory on Linux or macOS."""
    statm = Path("/proc/self/statm")
    if statm.is_file():
        pages = int(statm.read_text(encoding="ascii").split()[1])
        return pages * int(os.sysconf("SC_PAGE_SIZE")) / 1_048_576
    executable = shutil.which("ps")
    if executable is None:
        raise RuntimeError("ps is required to read process RSS")
    result = subprocess.run(  # noqa: S603 - executable is the resolved system ps binary.
        [executable, "-o", "rss=", "-p", str(os.getpid())],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("failed to read process RSS")
    return int(result.stdout.strip()) / 1024


def _peak_rss_mib() -> float:
    """Return peak resident memory with platform-specific unit conversion."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1_048_576 if sys.platform == "darwin" else value / 1024


def _cold_start_probe() -> None:
    """Initialize one child runtime and print its memory measurements."""
    runtime = EngineRuntime(_settings())
    asyncio.run(runtime.start())
    gc.collect()
    probe = MemoryProbe(peak_rss_mib=_peak_rss_mib(), steady_rss_mib=_rss_mib())
    asyncio.run(runtime.close())
    sys.stdout.write(probe.model_dump_json() + "\n")


def _measure_cold_start() -> ColdStartResult:
    """Measure process launch through complete dependency-free initialization."""
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("PII_ENGINE_")
    }
    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.run_synthetic", "--cold-start-probe"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError("cold-start benchmark subprocess failed")
    probe = MemoryProbe.model_validate_json(result.stdout)
    return ColdStartResult(seconds=elapsed, **probe.model_dump())


async def _measure_latency(
    runtime: EngineRuntime, characters: int, iterations: int
) -> LatencyResult:
    """Measure sequential latency for one bounded request size."""
    request = _request(characters)
    samples: list[float] = []
    for _iteration in range(iterations):
        started = time.perf_counter()
        await runtime.analyze("adapter", request)
        samples.append((time.perf_counter() - started) * 1000)
    return LatencyResult(
        p50=_percentile(samples, 0.50),
        p95=_percentile(samples, 0.95),
        p99=_percentile(samples, 0.99),
    )


async def _measure_throughput(runtime: EngineRuntime, concurrency: int, operations: int) -> float:
    """Measure fixed-work throughput at one concurrency level."""
    request = _request(128)

    async def worker(count: int) -> None:
        for _operation in range(count):
            await runtime.analyze("adapter", request)

    counts = [operations // concurrency] * concurrency
    for index in range(operations % concurrency):
        counts[index] += 1
    started = time.perf_counter()
    await asyncio.gather(*(worker(count) for count in counts))
    return operations / (time.perf_counter() - started)


async def _measure_queue_rejection() -> float:
    """Measure fail-closed rejection while the only worker is occupied."""
    runtime = EngineRuntime(
        _settings(max_concurrent_analyses=1, max_queued_analyses=0, analysis_timeout=5.0)
    )
    started_event = threading.Event()
    release_event = threading.Event()
    runtime.policy = cast(PolicyService, _BlockingPolicy(started_event, release_event))
    request = _request(128)
    first = asyncio.create_task(runtime.analyze("adapter", request))
    await asyncio.wait_for(asyncio.to_thread(started_event.wait), timeout=1)
    started = time.perf_counter()
    try:
        await runtime.analyze("adapter", request)
    except AnalysisCapacityError:
        elapsed = (time.perf_counter() - started) * 1000
    else:
        raise RuntimeError("saturated queue accepted an analysis")
    finally:
        release_event.set()
        await first
    return elapsed


async def _failure_checks() -> FailureChecks:
    """Exercise local analysis, model-cache, and Valkey failure boundaries."""
    runtime = EngineRuntime(_settings())
    runtime.policy = cast(PolicyService, _FailingPolicy())
    try:
        await runtime.analyze("adapter", _request(128))
    except RuntimeError:
        analysis_exception = True
    else:
        analysis_exception = False

    with tempfile.TemporaryDirectory(prefix="pii-benchmark-") as directory:
        model_cache = _missing_model_cache_fails_closed(Path(directory))

    store = SessionStore(
        "redis://127.0.0.1:1?socket_connect_timeout=0.05&socket_timeout=0.05", 60, "benchmark"
    )
    try:
        await asyncio.wait_for(store.start(), timeout=0.5)
    except (OSError, RedisError, TimeoutError):
        valkey = True
    else:
        valkey = False
    finally:
        await store.close()
    return FailureChecks(
        analysis_exception=analysis_exception,
        missing_model_cache=model_cache,
        unavailable_valkey=valkey,
    )


def _missing_model_cache_fails_closed(root: Path) -> bool:
    """Return whether production initialization rejects an absent selected bundle."""
    policy = root / "policy.yaml"
    policy.write_text(
        yaml.safe_dump(test_policy().model_dump(mode="json", by_alias=True)), encoding="utf-8"
    )
    settings = Settings(
        allow_test_analyzer=False,
        policy_config=policy,
        model_cache_path=root / "cache",
        model_bundle_reference=root / "cache/desired-bundle.json",
        model_bundle_version="missing",
        model_manifest_sha256="0" * 64,
        hash_key="h" * 32,
        encryption_key="e" * 32,
    )
    try:
        EngineRuntime(settings)
    except RuntimeNotReadyError:
        return True
    return False


async def _run(thresholds: Thresholds) -> BenchmarkReport:
    """Run every synthetic measurement and return a thresholded report."""
    cold_start = _measure_cold_start()
    runtime = EngineRuntime(_settings())
    await runtime.start()
    for _warmup in range(10):
        await runtime.analyze("adapter", _request(128))
    latency = {
        size: await _measure_latency(runtime, int(size), thresholds.iterations_per_size)
        for size in thresholds.latency_ms
    }
    throughput = {
        concurrency: await _measure_throughput(
            runtime, int(concurrency), thresholds.throughput_operations
        )
        for concurrency in thresholds.throughput_min_rps
    }
    await runtime.close()
    queue_rejection_ms = await _measure_queue_rejection()
    failures = await _failure_checks()
    violations = evaluate_thresholds(
        thresholds, cold_start, latency, throughput, queue_rejection_ms, failures
    )
    return BenchmarkReport(
        profile=thresholds.profile,
        generated_at=datetime.now(UTC).isoformat(),
        system=platform.system(),
        machine=platform.machine(),
        logical_cpu_count=os.cpu_count() or 1,
        python_version=platform.python_version(),
        cold_start=cold_start,
        latency_ms=latency,
        throughput_rps=throughput,
        queue_rejection_ms=queue_rejection_ms,
        failure_checks=failures,
        thresholds=thresholds,
        violations=violations,
        passed=not violations,
    )


def main() -> None:
    """Run the selected benchmark mode and exit non-zero on regressions."""
    logging.getLogger("pii_engine.runtime").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold-start-probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cold_start_probe:
        _cold_start_probe()
        return
    thresholds = Thresholds.model_validate_json((ROOT / "thresholds.json").read_bytes())
    report = asyncio.run(_run(thresholds))
    sys.stdout.write(report.model_dump_json(indent=2) + "\n")
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

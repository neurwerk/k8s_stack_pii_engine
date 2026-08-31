"""Validated thresholds and reports for local PII Engine benchmarks."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ColdStartBudget(BaseModel):
    """Bound synthetic process startup and memory use."""

    model_config = ConfigDict(extra="forbid")

    max_seconds: float = Field(gt=0)
    max_peak_rss_mib: float = Field(gt=0)
    max_steady_rss_mib: float = Field(gt=0)


class LatencyBudget(BaseModel):
    """Bound high-percentile latency for one request size."""

    model_config = ConfigDict(extra="forbid")

    max_p95: float = Field(gt=0)
    max_p99: float = Field(gt=0)


class Thresholds(BaseModel):
    """Validate the versioned synthetic benchmark budget."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1, le=1)
    profile: str = Field(pattern=r"^[a-z0-9-]+$")
    iterations_per_size: int = Field(ge=100, le=10_000)
    throughput_operations: int = Field(ge=100, le=100_000)
    cold_start: ColdStartBudget
    latency_ms: dict[str, LatencyBudget] = Field(min_length=1)
    throughput_min_rps: dict[str, float] = Field(min_length=1)
    max_queue_rejection_ms: float = Field(gt=0)


class MemoryProbe(BaseModel):
    """Record memory observed inside the cold-start child process."""

    peak_rss_mib: float
    steady_rss_mib: float


class ColdStartResult(MemoryProbe):
    """Record end-to-end child process startup duration."""

    seconds: float


class LatencyResult(BaseModel):
    """Record request latency percentiles in milliseconds."""

    p50: float
    p95: float
    p99: float


class FailureChecks(BaseModel):
    """Record whether local dependency and processing failures fail closed."""

    analysis_exception: bool
    missing_model_cache: bool
    unavailable_valkey: bool


class BenchmarkReport(BaseModel):
    """Emit one self-describing benchmark result."""

    schema_version: int = 1
    profile: str
    generated_at: str
    system: str
    machine: str
    logical_cpu_count: int
    python_version: str
    cold_start: ColdStartResult
    latency_ms: dict[str, LatencyResult]
    throughput_rps: dict[str, float]
    queue_rejection_ms: float
    failure_checks: FailureChecks
    thresholds: Thresholds
    violations: list[str]
    passed: bool


def evaluate_thresholds(
    thresholds: Thresholds,
    cold_start: ColdStartResult,
    latency: dict[str, LatencyResult],
    throughput: dict[str, float],
    queue_rejection_ms: float,
    failures: FailureChecks,
) -> list[str]:
    """Compare all measurements and fail-closed probes to their budgets."""
    violations = _cold_start_violations(thresholds, cold_start)
    violations.extend(_latency_violations(thresholds, latency))
    violations.extend(_throughput_violations(thresholds, throughput))
    if queue_rejection_ms > thresholds.max_queue_rejection_ms:
        violations.append("queue rejection latency exceeded")
    if not all(failures.model_dump().values()):
        violations.append("one or more dependency failures did not fail closed")
    return violations


def _cold_start_violations(thresholds: Thresholds, result: ColdStartResult) -> list[str]:
    """Return startup and memory budget violations."""
    budget = thresholds.cold_start
    violations: list[str] = []
    if result.seconds > budget.max_seconds:
        violations.append("cold-start duration exceeded")
    if result.peak_rss_mib > budget.max_peak_rss_mib:
        violations.append("cold-start peak RSS exceeded")
    if result.steady_rss_mib > budget.max_steady_rss_mib:
        violations.append("steady RSS exceeded")
    return violations


def _latency_violations(thresholds: Thresholds, results: dict[str, LatencyResult]) -> list[str]:
    """Return request percentile budget violations."""
    violations: list[str] = []
    for size, budget in thresholds.latency_ms.items():
        if results[size].p95 > budget.max_p95:
            violations.append(f"{size}-character p95 latency exceeded")
        if results[size].p99 > budget.max_p99:
            violations.append(f"{size}-character p99 latency exceeded")
    return violations


def _throughput_violations(thresholds: Thresholds, results: dict[str, float]) -> list[str]:
    """Return concurrency throughput budget violations."""
    return [
        f"concurrency-{concurrency} throughput was too low"
        for concurrency, minimum in thresholds.throughput_min_rps.items()
        if results[concurrency] < minimum
    ]

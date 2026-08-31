"""Bounded shared analysis capacity for all policy callers."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from pii_engine.metrics import analyses_in_flight, queue_depth, queue_wait_seconds


class AnalysisCapacityError(Exception):
    """Raise when analysis cannot start within the queue budget."""


@dataclass(frozen=True)
class QueueStats:
    """Expose safe queue counters."""

    queued: int
    in_flight: int


class AnalysisLimiter:
    """Limit concurrent work and reject an explicitly bounded waiting queue."""

    def __init__(self, max_concurrent: int, max_queued: int, wait_timeout: float) -> None:
        """Create a limiter with independent concurrency and queue bounds."""
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_queued = max_queued
        self._wait_timeout = wait_timeout
        self._queued = 0
        self._in_flight = 0
        self._lock = asyncio.Lock()

    @property
    def stats(self) -> QueueStats:
        """Return current queue counters."""
        return QueueStats(self._queued, self._in_flight)

    @asynccontextmanager
    async def slot(self, caller: str = "unknown") -> AsyncIterator[None]:
        """Acquire analysis capacity or raise without waiting indefinitely."""
        async with self._lock:
            if self._semaphore.locked() and self._queued >= self._max_queued:
                raise AnalysisCapacityError("analysis queue is full")
            self._queued += 1
            queue_depth.set(self._queued)
        acquired = False
        started = time.monotonic()
        try:
            try:
                await asyncio.wait_for(self._semaphore.acquire(), self._wait_timeout)
                acquired = True
            except TimeoutError as exc:
                raise AnalysisCapacityError("analysis queue wait timed out") from exc
            async with self._lock:
                self._queued -= 1
                self._in_flight += 1
                queue_depth.set(self._queued)
                analyses_in_flight.set(self._in_flight)
            queue_wait_seconds.labels(caller=caller).observe(time.monotonic() - started)
            try:
                yield
            finally:
                async with self._lock:
                    self._in_flight -= 1
                    analyses_in_flight.set(self._in_flight)
                self._semaphore.release()
        finally:
            if not acquired:
                async with self._lock:
                    self._queued -= 1
                    queue_depth.set(self._queued)

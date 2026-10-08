"""Process-only exact-window NER findings, never input text or mutable artifacts."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sys
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from cachetools import TTLCache

from pii_engine.metrics import ner_cache_bytes, ner_cache_events_total

if TYPE_CHECKING:
    from pii_engine.services.analyzer import EntityMatch


def accounted_size(findings: tuple[EntityMatch, ...]) -> int:
    """Count retained objects plus 1 KiB for digest/key and cache bookkeeping."""
    return (
        1024
        + sys.getsizeof(findings)
        + sum(
            sys.getsizeof(item)
            + sys.getsizeof(vars(item))
            + sum(sys.getsizeof(value) for value in vars(item).values())
            for item in findings
        )
    )


class NerWindowCache:
    """Share one byte-accounted LRU/creation-TTL budget across model windows."""

    def __init__(
        self, max_bytes: int, ttl_seconds: int, *, timer: Callable[[], float] = time.monotonic
    ) -> None:
        """Create a fresh unlinkable key space without starting background work."""
        self._secret = secrets.token_bytes(32)
        self._cache = TTLCache(max_bytes, ttl_seconds, timer=timer, getsizeof=accounted_size)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def run(
        self, text: str, context: bytes, detect: Callable[[], tuple[EntityMatch, ...]]
    ) -> tuple[EntityMatch, ...]:
        """Reuse complete findings; inference and its errors stay outside the lock."""
        if self._stop.is_set():
            return detect()
        try:
            digest = hmac.new(self._secret, digestmod=hashlib.sha256)
            digest.update(len(context).to_bytes(8, "big"))
            digest.update(context)
            digest.update(text.encode())
            key = digest.digest()
            with self._lock:
                self._expire()
                cached = self._cache.get(key)
                if cached is not None:
                    ner_cache_events_total.labels(event="hit").inc()
                    return cached
        except Exception:  # noqa: BLE001 - cache failure must never bypass detection.
            ner_cache_events_total.labels(event="error").inc()
            return detect()
        ner_cache_events_total.labels(event="miss").inc()
        findings = detect()
        try:
            with self._lock:
                if self._stop.is_set():
                    return findings
                self._expire()
                if key not in self._cache and accounted_size(findings) <= self._cache.maxsize:
                    before = len(self._cache)
                    self._cache[key] = findings
                    ner_cache_events_total.labels(event="eviction").inc(
                        before + 1 - len(self._cache)
                    )
                ner_cache_bytes.set(self._cache.currsize)
        except Exception:  # noqa: BLE001 - successful detection remains successful.
            ner_cache_events_total.labels(event="error").inc()
        return findings

    def _expire(self) -> None:
        ner_cache_events_total.labels(event="expiry").inc(len(self._cache.expire()))
        ner_cache_bytes.set(self._cache.currsize)

    def expire(self) -> None:
        """Free idle expired entries without changing the inference path."""
        try:
            with self._lock:
                self._expire()
        except Exception:  # noqa: BLE001 - optional optimization, no content logging.
            ner_cache_events_total.labels(event="error").inc()

    def start(self) -> None:
        """Start runtime-owned approximately minute-spaced cleanup."""
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._cleanup, daemon=True)
            self._thread.start()

    def _cleanup(self) -> None:
        while not self._stop.wait(60):
            self.expire()

    def close(self) -> None:
        """Stop cleanup and release retained findings on runtime shutdown."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        with self._lock:
            self._cache.clear()
            ner_cache_bytes.set(0)

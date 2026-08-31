"""Required Valkey-backed session taint state for block and reroute decisions."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from redis.exceptions import RedisError

from pii_engine.metrics import session_cache_total
from pii_engine.models.contracts import PIIReportRow

type RequestKind = Literal["model", "mcp"]


class SessionDecision(BaseModel):
    """Persist only safe decision metadata, never prompts or reversal entries."""

    model_config = ConfigDict(extra="forbid", strict=True)

    api_version: Literal["v1"]
    request_kind: RequestKind
    decision: Literal["block", "reroute"]
    entities: list[Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")]] = Field(max_length=64)
    entity_counts: dict[str, Annotated[int, Field(ge=1, le=10_000_000)]] = Field(max_length=64)
    report_rows: list[PIIReportRow] = Field(max_length=64)
    overlap_count: int = Field(ge=0, le=10_000_000)
    route_class: str | None = Field(default=None, max_length=128, pattern=r"^[A-Za-z0-9_.:/-]+$")
    remote_allowed: bool
    request_notices: list[Annotated[str, Field(max_length=4_000)]] = Field(max_length=16)
    response_notices: list[Annotated[str, Field(max_length=4_000)]] = Field(max_length=16)

    @model_validator(mode="after")
    def validate_taint(self) -> SessionDecision:
        """Require safe routing fields to agree with the cached decision."""
        self._validate_routing()
        self._validate_report_counts()
        self._validate_terminal_rows()
        return self

    def _validate_routing(self) -> None:
        """Require routing fields to describe one fail-closed terminal decision."""
        if self.request_notices:
            raise ValueError("session request notices must remain empty")
        if self.remote_allowed:
            raise ValueError("tainted session decisions cannot allow remote routing")
        if self.decision == "block" and self.route_class is not None:
            raise ValueError("blocked session decisions cannot select a route")
        if self.decision == "reroute" and self.route_class is None:
            raise ValueError("rerouted session decisions require a route")
        if self.request_kind == "mcp" and (
            self.decision != "block"
            or self.route_class is not None
            or self.request_notices
            or self.response_notices
            or any(row.action == "reroute" for row in self.report_rows)
        ):
            raise ValueError("MCP session decisions must contain only effective terminal blocks")

    def _validate_report_counts(self) -> None:
        """Require bounded cached aggregates to remain internally consistent."""
        if len(self.entities) != len(set(self.entities)) or set(self.entity_counts) != set(
            self.entities
        ):
            raise ValueError("session entity counts are inconsistent")
        report_entities = [row.entity_type for row in self.report_rows]
        if len(report_entities) != len(set(report_entities)):
            raise ValueError("session report rows must contain unique entity types")
        if report_entities != sorted(report_entities):
            raise ValueError("session report rows must be sorted by entity_type")
        if any(
            row.entity_type not in self.entity_counts
            or row.detected_count > self.entity_counts[row.entity_type]
            for row in self.report_rows
        ):
            raise ValueError("session report rows exceed entity counts")

    def _validate_terminal_rows(self) -> None:
        """Require cached report actions to agree with their terminal decision."""
        actions = {row.action for row in self.report_rows}
        if self.decision == "block":
            if any(row.transformed_count for row in self.report_rows):
                raise ValueError("blocked session decisions cannot claim transformations")
            if self.report_rows and "block" not in actions:
                raise ValueError("PII block session decisions require a block report row")
        elif "block" in actions or "reroute" not in actions:
            raise ValueError("rerouted session decisions require reroute rows without block rows")


class SessionStore:
    """Use Valkey as a required policy dependency when session state is enabled."""

    def __init__(self, url: str, ttl_seconds: int, policy_scope: str) -> None:
        """Configure namespaced keys and the required connection."""
        self._url = url
        self._ttl_seconds = ttl_seconds
        self._prefix = f"pii-engine:v2:{policy_scope}:session:"
        self._client: Any = None

    async def start(self) -> None:
        """Connect and fail startup if required session state is unavailable."""
        import redis.asyncio as redis

        self._client = redis.from_url(self._url, decode_responses=True)
        await self._client.ping()

    async def close(self) -> None:
        """Close the Valkey connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def healthy(self) -> bool:
        """Return current dependency health without affecting liveness."""
        if self._client is None:
            return False
        try:
            return bool(await self._client.ping())
        except (OSError, RedisError, TimeoutError):
            return False

    async def get(self, key: str) -> SessionDecision | None:
        """Return one safe taint decision and refresh its sliding TTL."""
        self._require_client()
        value = await self._client.getex(self._prefix + key, ex=self._ttl_seconds)
        if value is None:
            session_cache_total.labels(operation="get", outcome="miss").inc()
            return None
        try:
            decision = SessionDecision.model_validate_json(value)
        except ValueError as exc:
            session_cache_total.labels(operation="get", outcome="invalid").inc()
            raise RuntimeError("session cache contains invalid policy state") from exc
        session_cache_total.labels(operation="get", outcome="hit").inc()
        return decision

    async def set(self, key: str, decision: SessionDecision) -> None:
        """Persist a block or trusted-local reroute with bounded expiry."""
        self._require_client()
        if decision.decision not in {"block", "reroute"}:
            raise ValueError("only tainted decisions may enter the session cache")
        await self._client.set(
            self._prefix + key,
            decision.model_dump_json(),
            ex=self._ttl_seconds,
        )
        session_cache_total.labels(operation="set", outcome="success").inc()

    def _require_client(self) -> None:
        if self._client is None:
            raise RuntimeError("required session cache is unavailable")

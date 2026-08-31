"""Low-cardinality Prometheus metrics for policy and model operations."""

from prometheus_client import Counter, Gauge, Histogram

policy_requests_total = Counter(
    "pii_engine_policy_requests_total", "Policy request outcomes", ["caller", "decision"]
)
policy_failures_total = Counter(
    "pii_engine_policy_failures_total", "Policy processing failures", ["caller", "error"]
)
analysis_duration_seconds = Histogram(
    "pii_engine_analysis_duration_seconds",
    "Policy analysis duration",
    ["caller"],
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600),
)
overlap_regions_total = Counter(
    "pii_engine_overlap_regions_total", "Resolved cross-entity overlap regions", ["caller"]
)
queue_wait_seconds = Histogram(
    "pii_engine_queue_wait_seconds", "Time spent waiting for policy capacity", ["caller"]
)
queue_rejections_total = Counter(
    "pii_engine_queue_rejections_total", "Rejected policy queue attempts", ["caller", "reason"]
)
queue_depth = Gauge("pii_engine_queue_depth", "Queued policy analyses")
analyses_in_flight = Gauge("pii_engine_analyses_in_flight", "Policy analyses in flight")
runtime_device = Gauge("pii_engine_runtime_device", "Selected inference device", ["device"])
runtime_analyzer_mode = Gauge(
    "pii_engine_runtime_analyzer_mode", "Selected analyzer mode", ["mode"]
)
entities_total = Counter("pii_engine_entities_total", "Detected entities", ["entity_type"])
actions_total = Counter("pii_engine_actions_total", "Applied entity actions", ["action"])
session_cache_total = Counter(
    "pii_engine_session_cache_total", "Session cache operations", ["operation", "outcome"]
)

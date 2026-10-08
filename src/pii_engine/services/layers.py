"""Compose always-on CPU rules with exactly one deployment-selected NER layer."""

from __future__ import annotations

import secrets
import time
from typing import TYPE_CHECKING, Any, cast

from pii_engine.config.ner import NerConfig, load_ner
from pii_engine.metrics import analysis_chunks_total, analysis_stage_duration_seconds
from pii_engine.services.analyzer import EntityMatch, _deduplicate_matches, create_ner_analyzer

if TYPE_CHECKING:
    from pii_engine.config.policy import PolicySettings
    from pii_engine.config.settings import Settings
    from pii_engine.services.analyzer import AnalyzerMode


class LayeredAnalyzer:
    """Run independent detectors against unchanged original segments."""

    def __init__(self, settings: Settings, policy: PolicySettings, mode: AnalyzerMode) -> None:
        """Load rules and only the deployment-selected local pipelines."""
        from pii_engine.services.rules import RulesAnalyzer

        self.policy = policy
        self.config: NerConfig | None = (
            load_ner(settings.ner_config) if settings.ner_config else None
        )
        if self.config is not None:
            from pii_engine.config.policy import NerSettings

            if policy.pii.ner != NerSettings():
                raise ValueError("canonical NER conflicts with legacy policy model selection")
        self.rules = RulesAnalyzer(policy)
        ner_policy = policy
        if self.config is not None and mode == "local":
            pii = policy.pii.model_copy(
                update={"supported_languages": list(self.config.language_models)}
            )
            ner_policy = policy.model_copy(update={"pii": pii})
        self.ner = None if mode == "disabled" else create_ner_analyzer(settings, ner_policy, mode)
        from pii_engine.services.ner_cache import NerWindowCache

        self.window_cache = (
            NerWindowCache(settings.ner_cache_max_bytes, settings.ner_cache_ttl_seconds)
            if settings.ner_cache_enabled and self.ner is not None
            else None
        )
        if self.window_cache is not None:
            # A fresh namespace identifies this immutable loaded inference configuration.
            target = cast("Any", getattr(self.ner, "transport", self.ner))
            target.window_cache = self.window_cache
            target.cache_namespace = secrets.token_bytes(32)
        self.validate_policy(policy)

    def validate_policy(self, policy: PolicySettings) -> None:
        """Reject request-local language changes before scanning even empty text."""
        loaded = (
            self.config.language_models if self.config and self.config.mode != "disabled" else None
        )
        supported = set(loaded or self.policy.pii.supported_languages)
        if not set(policy.pii.analyzer_languages).issubset(supported):
            raise ValueError("request policy selects a NER language that is not loaded")
        if policy.pii.ner != self.policy.pii.ner:
            raise ValueError("request policy changes deployment model selection")
        selected_aliases = getattr(self.ner, "_selected_aliases", None)
        if selected_aliases is not None:
            aliases = selected_aliases(policy, tuple(policy.pii.analyzer_languages))
            loaded_aliases = getattr(self.ner, "_loaded_aliases", {})
            if any(loaded_aliases.get(language) != alias for language, alias in aliases.items()):
                raise ValueError("request policy selects a model that is not loaded")
        validator = getattr(self.ner, "_validate_languages", None)
        if validator is not None:
            validator(policy)

    def analyze(self, text: str, policy: PolicySettings | None = None) -> list[EntityMatch]:
        """Combine full-text evidence before the existing policy planner runs once."""
        active = policy or self.policy
        self.validate_policy(active)
        started = time.monotonic()
        try:
            matches = self.rules.analyze(text, active)
        finally:
            analysis_stage_duration_seconds.labels(stage="rules").observe(
                time.monotonic() - started
            )
            analysis_chunks_total.labels(stage="rules").inc(bool(text))
        if self.ner is not None:
            started = time.monotonic()
            try:
                matches.extend(self.ner.analyze(text, active))
            finally:
                analysis_stage_duration_seconds.labels(stage="ner").observe(
                    time.monotonic() - started
                )
        return _deduplicate_matches(matches)

    def healthy(self) -> bool:
        """Check selected remote services without inference or downloading models."""
        check = getattr(self.ner, "healthy", None)
        return True if check is None else check()

    def close(self) -> None:
        """Close a selected remote transport."""
        if self.window_cache is not None:
            self.window_cache.close()
        close = getattr(self.ner, "close", None)
        if close is not None:
            close()

    def start(self) -> None:
        """Start idle cache expiry with the process runtime."""
        if self.window_cache is not None:
            self.window_cache.start()

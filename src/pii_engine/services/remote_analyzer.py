"""Coordinate Presidio remote recognizers without repeated multilingual calls."""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from pii_engine.config.remote import RemoteConfig
from pii_engine.services.analyzer import EntityMatch, PresidioSpacyAnalyzer, _selected_entities
from pii_engine.services.remote_http import InputTooLargeError, RemoteAnalysisError, RemoteTransport
from pii_engine.services.remote_tokens import decode_predictions, load_tokenizer, token_chunks

if TYPE_CHECKING:
    from pii_engine.config.policy import PolicySettings
    from pii_engine.config.remote import RemoteModel
    from pii_engine.config.settings import Settings


class RemoteAnalyzer:
    """Keep local pattern recognition and use remote NER once per selected model."""

    def __init__(self, settings: Settings, policy: PolicySettings) -> None:
        """Load only linguistic support and, for KServe, verified tokenizer files."""
        if settings.remote_config is None:
            raise ValueError("remote model configuration is absent")
        try:
            config = RemoteConfig.model_validate(json.loads(settings.remote_config.read_bytes()))
        except (OSError, ValueError, UnicodeError):
            raise ValueError("remote model configuration is invalid") from None
        expected = "gliner" if settings.analyzer_backend == "remote-gliner" else "kserve"
        if any(model.kind != expected for model in config.models):
            raise ValueError("remote models do not match the selected analyzer backend")
        if expected == "gliner" and len(config.models) != 1:
            raise ValueError("multilingual GLiNER requires exactly one endpoint")
        if expected == "kserve":
            languages = [language for model in config.models for language in model.languages]
            if len(languages) != len(set(languages)):
                raise ValueError("KServe language assignments overlap")
        self.models = config.models
        self.policy = policy
        self.transport = RemoteTransport(settings)
        # Remote mode retains Presidio patterns but never falls back to local NER.
        self.baseline = PresidioSpacyAnalyzer(policy, include_ner=False)
        self.recognizers = [make_recognizer(model, self.transport) for model in self.models]
        self._health_lock = threading.Lock()
        self._health_checked = 0.0
        self._healthy = False
        self._validate_languages(policy)

    def close(self) -> None:
        """Release the remote connection pool."""
        self.transport.close()

    def healthy(self) -> bool:
        """Cache bounded readiness checks for five seconds, including outage results."""
        with self._health_lock:
            if time.monotonic() - self._health_checked >= 5:
                self._healthy = all(self.transport.healthy(model) for model in self.models)
                self._health_checked = time.monotonic()
            return self._healthy

    def _validate_languages(self, policy: PolicySettings) -> None:
        supported = {language for model in self.models for language in model.languages}
        if not set(policy.pii.analyzer_languages).issubset(supported):
            raise ValueError("remote model selection does not cover all analysis languages")
        if any(
            model.inference_threshold is not None
            and policy.pii.score_threshold < model.inference_threshold
            for model in self.models
        ):
            raise ValueError("policy threshold is below the remote detector's configured floor")

    def analyze(self, text: str, policy: PolicySettings | None = None) -> list[EntityMatch]:
        """Validate full remote coverage before returning policy-filtered matches."""
        active = policy or self.policy
        self._validate_languages(active)
        matches = self.baseline.analyze(text, active)
        if not text:
            return matches
        selected = _selected_entities(active)
        for model, recognizer in zip(self.models, self.recognizers, strict=True):
            if not set(model.languages).intersection(active.pii.analyzer_languages):
                continue
            for result in recognizer.analyze(text, list(selected), None):
                if result.score >= active.pii.score_threshold:
                    matches.append(
                        EntityMatch(
                            result.entity_type,
                            result.start,
                            result.end,
                            result.score,
                            "transformer",
                        )
                    )
        return _unique(matches)


def make_recognizer(model: RemoteModel, transport: RemoteTransport) -> Any:  # noqa: ANN401
    """Construct Presidio's production-only remote extension lazily."""
    from presidio_analyzer import RecognizerResult, RemoteRecognizer

    class ModelRecognizer(RemoteRecognizer):
        """Expose one pinned external model using Presidio's result contract."""

        def __init__(self) -> None:
            super().__init__(
                supported_entities=sorted(set(model.label_mapping.values())),
                name=model.name,
                supported_language=model.languages[0],
                version="1",
            )
            self.tokenizer, self.labels = (
                load_tokenizer(model)
                if model.kind == "kserve"
                else (
                    None,
                    {},
                )
            )

        def get_supported_entities(self) -> list[str]:
            """Return the explicitly reviewed normalized entity contract."""
            return self.supported_entities

        def analyze(
            self,
            text: str,
            entities: list[str],
            nlp_artifacts: Any = None,  # noqa: ANN401
        ) -> list[Any]:
            """Validate all findings, then apply the caller's entity selection."""
            if model.kind == "gliner":
                matches = gliner_matches(text, model, transport)
            else:
                matches = kserve_matches(text, model, transport, self.tokenizer, self.labels)
            return [
                RecognizerResult(item.entity_type, item.start, item.end, item.score)
                for item in _unique(matches)
                if item.entity_type in entities
            ]

    return ModelRecognizer()


def gliner_matches(text: str, model: RemoteModel, transport: RemoteTransport) -> list[EntityMatch]:
    """Subdivide only explicit size rejections; overlap without losing any characters."""
    pending = []
    for offset in range(0, len(text), 896):
        pending.append((offset, text[offset : offset + 1024]))
        if offset + 1024 >= len(text):
            break
    matches: list[EntityMatch] = []
    while pending:
        offset, chunk = pending.pop()
        try:
            value = transport.request(model, {"text": chunk})
        except InputTooLargeError:
            if len(chunk) <= 64:
                raise RemoteAnalysisError("GLiNER cannot scan bounded input") from None
            middle = len(chunk) // 2
            overlap = min(32, middle // 2)
            pending.extend(
                [
                    (offset, chunk[: middle + overlap]),
                    (offset + middle - overlap, chunk[middle - overlap :]),
                ]
            )
            continue
        matches.extend(
            replace(item, start=item.start + offset, end=item.end + offset)
            for item in _gliner_response(value, chunk, model)
        )
    return matches


def _gliner_response(value: object, text: str, model: RemoteModel) -> list[EntityMatch]:
    if (
        not isinstance(value, dict)
        or set(value) != {"model", "entities"}
        or value.get("model") != model.model_name
    ):
        raise RemoteAnalysisError("GLiNER model identity is invalid")
    entities = value.get("entities")
    if not isinstance(entities, list):
        raise RemoteAnalysisError("GLiNER entity list is absent")
    matches = []
    for item in entities:
        if not isinstance(item, dict) or set(item) != {"start", "end", "label", "score"}:
            raise RemoteAnalysisError("GLiNER entity is invalid")
        start, end, label, score = (item.get(key) for key in ("start", "end", "label", "score"))
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(text)
            or not isinstance(label, str)
            or label not in model.label_mapping
            or isinstance(score, bool)
            or not isinstance(score, int | float)
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise RemoteAnalysisError("GLiNER entity offsets, label or score are invalid")
        matches.append(
            EntityMatch(model.label_mapping[label], start, end, float(score), model.name)
        )
    return matches


def kserve_matches(
    text: str,
    model: RemoteModel,
    transport: RemoteTransport,
    tokenizer: Any,  # noqa: ANN401
    labels: dict[str, str],
) -> list[EntityMatch]:
    """Send one unpadded instance and verify the complete special-token layout."""
    matches: list[EntityMatch] = []
    for offset, chunk in token_chunks(text, tokenizer):
        encoded = tokenizer(
            chunk,
            add_special_tokens=True,
            truncation=False,
            return_offsets_mapping=True,
        )
        if len(encoded["input_ids"]) > 512:
            raise RemoteAnalysisError("KServe input exceeds tokenizer limit")
        value = transport.request(model, {"instances": [chunk]})
        matches.extend(
            replace(item, start=item.start + offset, end=item.end + offset)
            for item in decode_predictions(value, chunk, encoded["offset_mapping"], labels, model)
        )
    return matches


def _unique(matches: list[EntityMatch]) -> list[EntityMatch]:
    """Deduplicate identical overlap evidence, preserving its strongest confidence."""
    unique: dict[tuple[str, int, int, str], EntityMatch] = {}
    for item in matches:
        key = item.entity_type, item.start, item.end, item.source
        if key not in unique or item.score > unique[key].score:
            unique[key] = item
    return sorted(unique.values(), key=lambda item: (item.start, item.end, item.entity_type))

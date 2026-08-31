"""Offline in-process Presidio baseline and verified transformer analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pii_engine.lib.bundle import (
    desired_cache,
    desired_reference_selects_other_complete_cache,
    parse_manifest,
)
from pii_engine.lib.catalog import ENTITY_CATALOG, compiled_recognizers
from pii_engine.services.recognizers import (
    custom_recognizers,
    normalized_recognizers,
    normalized_transformers_recognizer,
)

if TYPE_CHECKING:
    from pii_engine.config.policy import PolicySettings
    from pii_engine.config.settings import Settings

_PRESIDIO_TO_NORMALIZED = {
    "CREDIT_CARD": "CREDIT_CARD_NUMBER",
    "IBAN_CODE": "IBAN",
}
SPACY_ENTITY_MAPPING = {
    "PER": "PERSON_NAME",
    "PERSON": "PERSON_NAME",
    "LOC": "CITY",
    "GPE": "CITY",
}
SPACY_IGNORED_ENTITY_LABELS = (
    "CARDINAL",
    "DATE",
    "EVENT",
    "FAC",
    "LANGUAGE",
    "LAW",
    "MISC",
    "MONEY",
    "NORP",
    "ORDINAL",
    "ORG",
    "PERCENT",
    "PRODUCT",
    "QUANTITY",
    "TIME",
    "WORK_OF_ART",
)
AnalyzerMode = Literal["baseline", "transformer", "test"]
_BASELINE_CHUNK_CHARACTERS = 900_000
_BASELINE_CHUNK_OVERLAP = 10_000


@dataclass(frozen=True)
class EntityMatch:
    """Represent one entity span relative to exactly one text leaf."""

    entity_type: str
    start: int
    end: int
    score: float
    source: str


class Analyzer(Protocol):
    """Analyze one independent text leaf."""

    def analyze(self, text: str, policy: PolicySettings | None = None) -> list[EntityMatch]:
        """Return normalized spans without retaining input text."""


class DeterministicAnalyzer:
    """Small explicit test analyzer; never enabled by production settings."""

    def analyze(self, text: str, policy: PolicySettings | None = None) -> list[EntityMatch]:
        """Return deterministic recognizer matches for isolated unit tests."""
        matches: list[EntityMatch] = []
        for recognizer, pattern in compiled_recognizers():
            matches.extend(
                EntityMatch(
                    recognizer.entity,
                    match.start(),
                    match.end(),
                    0.95,
                    recognizer.source,
                )
                for match in pattern.finditer(text)
            )
        return _deduplicate_matches(matches)


def resolve_analyzer_mode(settings: Settings) -> AnalyzerMode:
    """Select test, bundled baseline, or a verified transformer bundle."""
    if settings.allow_test_analyzer:
        return "test"
    reference = settings.model_bundle_reference
    if reference is None:
        return "baseline"
    try:
        reference.lstat()
    except FileNotFoundError:
        return "baseline"
    except OSError as exc:
        raise ValueError("desired-bundle reference cannot be inspected") from exc
    bundle = settings.model_bundle_path
    cache = settings.model_cache_path
    digest = settings.model_manifest_sha256
    version = settings.model_bundle_version
    if bundle is None or cache is None or digest is None or version is None:
        raise ValueError("desired-bundle reference has incomplete model configuration")
    if desired_cache(cache, reference, digest, version) != bundle:
        if desired_reference_selects_other_complete_cache(cache, reference, digest, version):
            return "baseline"
        raise ValueError("desired-bundle reference or selected model bundle is invalid")
    return "transformer"


def create_analyzer(settings: Settings, policy: PolicySettings, mode: AnalyzerMode) -> Analyzer:
    """Create the analyzer selected by the validated runtime mode."""
    if mode == "test":
        return DeterministicAnalyzer()
    if mode == "baseline":
        return PresidioSpacyAnalyzer(policy)
    return PresidioAnalyzer(settings, policy)


def configure_inference_device(device: str) -> str:
    """Select CPU or require one explicit CUDA device before loading models."""
    import spacy
    import torch
    from thinc.api import get_torch_default_device

    if device == "cpu":
        spacy.require_cpu()
        return "cpu"
    index = int(device.split(":", maxsplit=1)[1]) if ":" in device else 0
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise ValueError("configured CUDA device is unavailable")
    try:
        spacy.require_gpu(index)
    except (ImportError, RuntimeError, ValueError) as exc:
        raise ValueError("configured CUDA device cannot be activated") from exc
    selected = get_torch_default_device()
    if selected.type != "cuda" or selected.index != index:
        raise ValueError("transformer pipeline did not select the configured CUDA device")
    return f"cuda:{index}"


class PresidioAnalyzer:
    """Load configured Presidio transformer engines entirely from local paths."""

    def __init__(self, settings: Settings, policy: PolicySettings) -> None:
        """Validate bundle metadata and eagerly load every configured model."""
        bundle = settings.model_bundle_path
        digest = settings.model_manifest_sha256
        version = settings.model_bundle_version
        if bundle is None or digest is None or version is None:
            raise ValueError("model bundle configuration is incomplete")
        manifest_data = (bundle / "manifest.yaml").read_bytes()
        self.manifest = parse_manifest(manifest_data, digest, version)
        self.policy = policy
        self.bundle = bundle
        self._loaded_aliases = self._selected_aliases(policy, tuple(policy.pii.supported_languages))
        self._validate_selection()
        self._engine = self._create_engine()
        self._tokenizers = self._load_tokenizers()

    def analyze(self, text: str, policy: PolicySettings | None = None) -> list[EntityMatch]:
        """Analyze all configured languages and use Presidio duplicate handling."""
        from presidio_analyzer import EntityRecognizer, RecognizerResult

        active = policy or self.policy
        aliases = self._selected_aliases(active, tuple(active.pii.analyzer_languages))
        if any(self._loaded_aliases.get(language) != alias for language, alias in aliases.items()):
            raise ValueError("request policy selects a model that is not loaded")
        results: list[Any] = []
        for language in active.pii.analyzer_languages:
            tokenizer = self._tokenizers[language]
            for offset, chunk in _chunks(text, tokenizer):
                for result in self._engine.analyze(
                    text=chunk,
                    language=language,
                    score_threshold=active.pii.score_threshold,
                ):
                    result.entity_type = _PRESIDIO_TO_NORMALIZED.get(
                        result.entity_type, result.entity_type
                    )
                    result.start += offset
                    result.end += offset
                    results.append(result)
        deduplicated: list[RecognizerResult] = EntityRecognizer.remove_duplicates(results)
        selected = _selected_entities(active)
        matches = [
            EntityMatch(
                result.entity_type,
                result.start,
                result.end,
                float(result.score),
                _recognizer_source(result),
            )
            for result in sorted(
                deduplicated, key=lambda item: (item.start, item.end, item.entity_type)
            )
            if result.entity_type in selected
        ]
        return _deduplicate_matches(matches)

    def _validate_selection(self) -> None:
        """Reject unknown entities, aliases, paths, and language mismatches."""
        validate_policy_selection(self.policy)
        for language, alias in self._loaded_aliases.items():
            model = self.manifest.models.get(alias)
            if model is None:
                raise ValueError("policy selects an unknown model alias")
            path = (self.bundle / model.path).resolve()
            path.relative_to(self.bundle.resolve())
            if not path.is_dir() or language not in model.supported_languages:
                raise ValueError("selected model is missing or does not support its language")

    def _selected_aliases(
        self, policy: PolicySettings, languages: tuple[str, ...]
    ) -> dict[str, str]:
        """Return one immutable model alias for each configured language."""
        ner = policy.pii.ner
        if ner.strategy == "multilingual":
            return dict.fromkeys(languages, ner.general_model)
        aliases = {
            language: ner.per_language[language]
            for language in languages
            if language in ner.per_language
        }
        if set(aliases) != set(languages):
            raise ValueError("per-language model selection is incomplete")
        return aliases

    def _create_engine(self) -> Any:  # noqa: ANN401
        """Construct and eagerly initialize one normalized TransformersNlpEngine."""
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NerModelConfiguration, TransformersNlpEngine

        models = [
            {
                "lang_code": language,
                "model_name": {
                    "spacy": _spacy_model(language),
                    "transformers": str(self.bundle / self.manifest.models[alias].path),
                },
            }
            for language, alias in self._loaded_aliases.items()
        ]
        runtime = self.manifest.runtime
        nlp_engine = TransformersNlpEngine(
            models=models,
            ner_model_configuration=NerModelConfiguration(
                model_to_presidio_entity_mapping=runtime.model_to_presidio_entity_mapping,
                labels_to_ignore=runtime.labels_to_ignore,
                aggregation_strategy=runtime.aggregation_strategy,
                stride=runtime.stride,
            ),
        )
        nlp_engine.load()
        engine = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=list(self.policy.pii.supported_languages),
        )
        engine.registry.remove_recognizer("TransformersRecognizer")
        for language in self.policy.pii.supported_languages:
            engine.registry.add_recognizer(
                normalized_transformers_recognizer(list(ENTITY_CATALOG), language)
            )
        recognizers = normalized_recognizers(tuple(self.policy.pii.supported_languages))
        recognizers.extend(custom_recognizers(self.policy.pii.custom_recognizers))
        for recognizer in recognizers:
            engine.registry.add_recognizer(recognizer)
        return engine

    def _load_tokenizers(self) -> dict[str, Any]:
        """Load local tokenizers used by the retained complete-document path."""
        from transformers import AutoTokenizer

        return {
            language: AutoTokenizer.from_pretrained(
                self.bundle / self.manifest.models[alias].path,
                local_files_only=True,
            )
            for language, alias in self._loaded_aliases.items()
        }


class PresidioSpacyAnalyzer:
    """Run the bundled EN, DE, and NL spaCy models through Presidio."""

    def __init__(self, policy: PolicySettings) -> None:
        """Eagerly load every policy-supported bundled spaCy model."""
        self.policy = policy
        validate_policy_selection(policy)
        self._engine = self._create_engine()

    def analyze(self, text: str, policy: PolicySettings | None = None) -> list[EntityMatch]:
        """Analyze configured languages with baseline NER and Presidio recognizers."""
        from presidio_analyzer import EntityRecognizer, RecognizerResult

        active = policy or self.policy
        results: list[Any] = []
        for language in active.pii.analyzer_languages:
            for offset, chunk, owned_start, owned_end in _baseline_chunks(text):
                for result in self._engine.analyze(
                    text=chunk,
                    language=language,
                    score_threshold=active.pii.score_threshold,
                ):
                    result.entity_type = _PRESIDIO_TO_NORMALIZED.get(
                        result.entity_type, result.entity_type
                    )
                    result.start += offset
                    result.end += offset
                    midpoint = (result.start + result.end) // 2
                    if owned_start <= midpoint < owned_end:
                        results.append(result)
        deduplicated: list[RecognizerResult] = EntityRecognizer.remove_duplicates(results)
        selected = _selected_entities(active)
        matches = [
            EntityMatch(
                result.entity_type,
                result.start,
                result.end,
                float(result.score),
                _recognizer_source(result),
            )
            for result in sorted(
                deduplicated, key=lambda item: (item.start, item.end, item.entity_type)
            )
            if result.entity_type in selected
        ]
        return _deduplicate_matches(matches)

    def _create_engine(self) -> Any:  # noqa: ANN401
        """Construct and eagerly initialize Presidio's spaCy NLP engine."""
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NerModelConfiguration, SpacyNlpEngine

        nlp_engine = SpacyNlpEngine(
            models=[
                {"lang_code": language, "model_name": _spacy_model(language)}
                for language in self.policy.pii.supported_languages
            ],
            ner_model_configuration=NerModelConfiguration(
                model_to_presidio_entity_mapping=SPACY_ENTITY_MAPPING,
                labels_to_ignore=list(SPACY_IGNORED_ENTITY_LABELS),
            ),
        )
        nlp_engine.load()
        engine = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=list(self.policy.pii.supported_languages),
        )
        recognizers = normalized_recognizers(tuple(self.policy.pii.supported_languages))
        recognizers.extend(custom_recognizers(self.policy.pii.custom_recognizers))
        for recognizer in recognizers:
            engine.registry.add_recognizer(recognizer)
        return engine


def validate_policy_selection(policy: PolicySettings) -> None:
    """Reject entities which are outside the normalized and custom catalogs."""
    custom_entities = {item.entity for item in policy.pii.custom_recognizers}
    unknown = set(policy.pii.analyzer_entities) - set(ENTITY_CATALOG) - custom_entities
    if unknown:
        raise ValueError("policy selects an unknown entity")


def _selected_entities(policy: PolicySettings) -> set[str]:
    """Return the policy's explicit or complete normalized entity selection."""
    custom_entities = {item.entity for item in policy.pii.custom_recognizers}
    return set(policy.pii.analyzer_entities or (*ENTITY_CATALOG, *custom_entities))


def _deduplicate_matches(matches: list[EntityMatch]) -> list[EntityMatch]:
    """Return exact unique matches while preserving cross-entity evidence."""
    return sorted(
        set(matches),
        key=lambda item: (item.start, item.end, item.entity_type, -item.score, item.source),
    )


def _spacy_model(language: str) -> str:
    """Return the pinned small linguistic support model bundled in the image."""
    try:
        return {"en": "en_core_web_sm", "de": "de_core_news_sm", "nl": "nl_core_news_sm"}[language]
    except KeyError as exc:
        raise ValueError("configured language has no spaCy support model") from exc


def _chunks(text: str, tokenizer: Any) -> list[tuple[int, str]]:  # noqa: ANN401
    """Retain tokenizer-aware overlap until native stride passes differential tests."""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [offset for offset in encoded["offset_mapping"] if offset[0] != offset[1]]
    max_tokens = min(int(tokenizer.model_max_length) - 32, 480)
    if max_tokens <= 0:
        raise ValueError("model tokenizer context limit is invalid")
    if len(offsets) <= max_tokens:
        return [(0, text)]
    overlap = min(64, max_tokens // 4)
    chunks: list[tuple[int, str]] = []
    for start in range(0, len(offsets), max_tokens - overlap):
        window = offsets[start : start + max_tokens]
        if not window:
            break
        chunk_start, chunk_end = window[0][0], window[-1][1]
        chunks.append((chunk_start, text[chunk_start:chunk_end]))
        if start + max_tokens >= len(offsets):
            break
    return chunks


def _baseline_chunks(text: str) -> list[tuple[int, str, int, int]]:
    """Split baseline input below spaCy's limit with overlap and unique ownership."""
    if len(text) <= _BASELINE_CHUNK_CHARACTERS:
        return [(0, text, 0, len(text))]
    chunks: list[tuple[int, str, int, int]] = []
    for owned_start in range(0, len(text), _BASELINE_CHUNK_CHARACTERS):
        owned_end = min(len(text), owned_start + _BASELINE_CHUNK_CHARACTERS)
        chunk_start = max(0, owned_start - _BASELINE_CHUNK_OVERLAP)
        chunk_end = min(len(text), owned_end + _BASELINE_CHUNK_OVERLAP)
        chunks.append((chunk_start, text[chunk_start:chunk_end], owned_start, owned_end))
    return chunks


def _recognizer_source(result: Any) -> str:  # noqa: ANN401
    """Return bounded recognizer provenance without exposing analysis text."""
    metadata = getattr(result, "recognition_metadata", None) or {}
    name = str(metadata.get("recognizer_name", "presidio"))
    if "Transformer" in name:
        return "transformer"
    if "Spacy" in name:
        return "spacy"
    return "deterministic"

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
from pii_engine.metrics import analysis_chunks_total
from pii_engine.services.recognizers import (
    custom_recognizers,
    normalized_recognizers,
    normalized_transformers_recognizer,
)

if TYPE_CHECKING:
    from pii_engine.config.ner import ModelProfile
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
AnalyzerMode = Literal[
    "baseline",
    "transformer",
    "test",
    "remote-gliner",
    "remote-kserve",
    "local",
    "remote",
    "disabled",
]
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
    if settings.ner_config is not None:
        from pii_engine.config.ner import load_ner

        return load_ner(settings.ner_config).mode
    if settings.analyzer_backend != "local":
        return settings.analyzer_backend
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
    from pii_engine.services.layers import LayeredAnalyzer

    return LayeredAnalyzer(settings, policy, mode)


def create_ner_analyzer(settings: Settings, policy: PolicySettings, mode: AnalyzerMode) -> Analyzer:
    """Construct only NER; the independent rules layer owns other recognizers."""
    if mode in {"baseline", "local"}:
        profiles = None
        if settings.ner_config is not None:
            from pii_engine.config.ner import PROFILES, load_ner

            config = load_ner(settings.ner_config)
            profiles = {
                language: PROFILES[config.models[name].profile]
                for language, name in config.language_models.items()
            }
        return PresidioSpacyAnalyzer(policy, ner_only=True, profiles=profiles)
    if mode in {"remote-gliner", "remote-kserve", "remote"}:
        from pii_engine.services.remote_analyzer import RemoteAnalyzer

        return RemoteAnalyzer(settings, policy)
    return PresidioAnalyzer(settings, policy, ner_only=True)


def configure_cpu_inference() -> None:
    """Explicitly select CPU before loading local models."""
    import spacy

    spacy.require_cpu()


class PresidioAnalyzer:
    """Load configured Presidio transformer engines entirely from local paths."""

    def __init__(
        self, settings: Settings, policy: PolicySettings, *, ner_only: bool = False
    ) -> None:
        """Validate bundle metadata and eagerly load every configured model."""
        bundle = settings.model_bundle_path
        digest = settings.model_manifest_sha256
        version = settings.model_bundle_version
        if bundle is None or digest is None or version is None:
            raise ValueError("model bundle configuration is incomplete")
        manifest_data = (bundle / "manifest.yaml").read_bytes()
        self.manifest = parse_manifest(manifest_data, digest, version)
        self.policy = policy
        self.ner_only = ner_only
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
        seen: set[str] = set()
        for language in active.pii.analyzer_languages:
            if aliases[language] in seen:
                continue
            seen.add(aliases[language])
            tokenizer = self._tokenizers[language]
            for offset, chunk in _chunks(text, tokenizer):
                analysis_chunks_total.labels(stage="ner").inc(bool(chunk))
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
        engine.registry.remove_recognizer("EmailRecognizer")
        for language in self.policy.pii.supported_languages:
            engine.registry.add_recognizer(
                normalized_transformers_recognizer(list(ENTITY_CATALOG), language)
            )
        recognizers = normalized_recognizers(tuple(self.policy.pii.supported_languages))
        recognizers.extend(custom_recognizers(self.policy.pii.custom_recognizers))
        for recognizer in recognizers:
            engine.registry.add_recognizer(recognizer)
        if self.ner_only:
            engine.registry.recognizers = [
                item for item in engine.registry.recognizers if "Transformer" in item.name
            ]
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

    def __init__(
        self,
        policy: PolicySettings,
        *,
        include_ner: bool = True,
        ner_only: bool = False,
        profiles: dict[str, ModelProfile] | None = None,
    ) -> None:
        """Eagerly load every policy-supported bundled spaCy model."""
        self.policy = policy
        self.profiles = profiles or {}
        validate_policy_selection(policy)
        self._engine = self._create_engine()
        if ner_only:
            self._engine.registry.recognizers = [
                item for item in self._engine.registry.recognizers if item.name == "SpacyRecognizer"
            ]
        if not include_ner:
            self._engine.registry.remove_recognizer("SpacyRecognizer")
            for pipeline in self._engine.nlp_engine.nlp.values():
                if "ner" in pipeline.pipe_names:
                    pipeline.disable_pipe("ner")

    def analyze(self, text: str, policy: PolicySettings | None = None) -> list[EntityMatch]:
        """Analyze configured languages with baseline NER and Presidio recognizers."""
        from presidio_analyzer import EntityRecognizer, RecognizerResult

        active = policy or self.policy
        results: list[Any] = []
        for language in active.pii.analyzer_languages:
            profile = self.profiles.get(language)
            chunks = _baseline_chunks(
                text,
                profile.max_characters if profile else _BASELINE_CHUNK_CHARACTERS,
                profile.overlap_characters if profile else _BASELINE_CHUNK_OVERLAP,
            )
            for offset, chunk, owned_start, owned_end in chunks:
                analysis_chunks_total.labels(stage="ner").inc(bool(chunk))
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

        mapping = SPACY_ENTITY_MAPPING.copy()
        ignored = list(SPACY_IGNORED_ENTITY_LABELS)
        if self.profiles:
            mapping = {}
            ignored = []
            for profile in self.profiles.values():
                for label, entity in profile.label_mapping.items():
                    if label in mapping and mapping[label] != entity:
                        raise ValueError("local profiles have conflicting label mappings")
                    mapping[label] = entity
                ignored.extend(profile.ignored_labels)
        nlp_engine = SpacyNlpEngine(
            models=[
                {
                    "lang_code": language,
                    "model_name": self.profiles[language].upstream
                    if language in self.profiles
                    else _spacy_model(language),
                }
                for language in self.policy.pii.supported_languages
            ],
            ner_model_configuration=NerModelConfiguration(
                model_to_presidio_entity_mapping=mapping,
                labels_to_ignore=sorted(set(ignored) - set(mapping)),
            ),
        )
        nlp_engine.load()
        engine = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=list(self.policy.pii.supported_languages),
        )
        engine.registry.remove_recognizer("EmailRecognizer")
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
        chunk_start = 0 if start == 0 else window[0][0]
        chunk_end = (
            len(text) if start + max_tokens >= len(offsets) else offsets[start + max_tokens][0]
        )
        chunks.append((chunk_start, text[chunk_start:chunk_end]))
        if start + max_tokens >= len(offsets):
            break
    return chunks


def _baseline_chunks(
    text: str,
    size: int = _BASELINE_CHUNK_CHARACTERS,
    overlap: int = _BASELINE_CHUNK_OVERLAP,
) -> list[tuple[int, str, int, int]]:
    """Split baseline input below spaCy's limit with overlap and unique ownership."""
    if len(text) <= size:
        return [(0, text, 0, len(text))]
    chunks: list[tuple[int, str, int, int]] = []
    for owned_start in range(0, len(text), size):
        owned_end = min(len(text), owned_start + size)
        chunk_start = max(0, owned_start - overlap)
        chunk_end = min(len(text), owned_end + overlap)
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

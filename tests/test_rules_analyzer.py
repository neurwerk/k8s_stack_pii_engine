"""Exercise real production recognizers when the production extra is present."""

import pytest

from pii_engine.config.policy import PolicySettings
from pii_engine.config.policy import test_policy as make_policy

pytest.importorskip("presidio_analyzer", reason="production Presidio dependency not installed")

from pii_engine.services.rules import RulesAnalyzer


def policy(*, languages=("en",), entities=(), threshold=0.45, custom=()):
    data = make_policy().model_dump(by_alias=True)
    data["pii"].update(
        supportedLanguages=["en", "de", "nl"],
        analyzerLanguages=list(languages),
        analyzerEntities=list(entities),
        scoreThreshold=threshold,
        customRecognizers=list(custom),
    )
    return PolicySettings.model_validate(data)


def values(analyzer, text, active=None):
    return {
        (match.entity_type, text[match.start : match.end])
        for match in analyzer.analyze(text, active)
    }


def test_no_trained_pipeline_or_network_and_full_original_offsets(monkeypatch):
    import socket

    spacy = pytest.importorskip("spacy")
    presidio = pytest.importorskip("presidio_analyzer")
    nlp = pytest.importorskip("presidio_analyzer.nlp_engine")

    def forbidden(*args, **kwargs):
        pytest.fail("rules attempted NLP/model loading or network access")

    monkeypatch.setattr(spacy, "load", forbidden)
    monkeypatch.setattr(nlp.SpacyNlpEngine, "load", forbidden)
    monkeypatch.setattr(presidio.AnalyzerEngine, "__init__", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    analyzer = RulesAnalyzer(policy(entities=("EMAIL_ADDRESS", "VAT_NUMBER", "STEUERNUMMER")))
    text = "😀 " + "ordinary " * 120_000 + "contact: alex@example.com DE123456789 12/345/67890"
    matches = analyzer.analyze(text)
    assert values(analyzer, text) == {
        ("EMAIL_ADDRESS", "alex@example.com"),
        ("VAT_NUMBER", "DE123456789"),
        ("STEUERNUMMER", "12/345/67890"),
    }
    assert all(match.start == text.index(text[match.start : match.end]) for match in matches)
    assert values(analyzer, "alex@example.invalidsuffix") == set()


@pytest.mark.parametrize(
    ("entity", "valid", "invalid"),
    [
        ("CREDIT_CARD_NUMBER", "4111 1111 1111 1111", "4111 1111 1111 1112"),
        ("IBAN", "DE89370400440532013000", "DE88370400440532013000"),
        ("BSN", "123456782", "123456783"),
        ("BSN", "123456782", "000000000"),
        ("IP_ADDRESS", "192.0.2.1", "999.999.999.999"),
    ],
)
def test_validation_rejects_invalid_even_with_context_and_threshold_zero(entity, valid, invalid):
    analyzer = RulesAnalyzer(policy(entities=(entity,), threshold=0))
    assert (entity, valid) in values(analyzer, f"bank credit card bsn ip {valid}")
    assert values(analyzer, f"bank credit card bsn ip {invalid}") == set()


@pytest.mark.parametrize(
    ("language", "label"),
    [
        ("en", "PHONE"),
        ("en", "telephone"),
        ("en", "phones"),
        ("en", "telephoned"),
        ("de", "Telefonnummern"),
        ("nl", "telefoonnummers"),
    ],
)
def test_phone_context_preserves_boost_and_threshold(language, label):
    active = policy(languages=(language,), entities=("PHONE_NUMBER",), threshold=0.7)
    analyzer = RulesAnalyzer(active)
    text = f"{label}: +31 6 12345678"
    matches = analyzer.analyze(text)
    assert len(matches) == 1
    assert text[matches[0].start : matches[0].end] == "+31 6 12345678"
    assert matches[0].score == 0.75
    assert analyzer.analyze("value: +31 6 12345678") == []
    assert analyzer.analyze("megatelephone: +31 6 12345678") == []
    assert (
        analyzer.analyze(
            text, policy(languages=(language,), entities=("PHONE_NUMBER",), threshold=0.76)
        )
        == []
    )


@pytest.mark.parametrize(
    ("language", "label"),
    [
        ("en", "id number"),
        ("de", "Personalausweises"),
        ("de", "Personalausweisnummer"),
        ("de", "ID Nummer"),
        ("nl", "identiteitskaarten"),
    ],
)
def test_national_id_phrases_and_inflections(language, label):
    analyzer = RulesAnalyzer(
        policy(languages=(language,), entities=("NATIONAL_ID_NUMBER",), threshold=0.65)
    )
    assert values(analyzer, f"{label}: L01X00T47") == {("NATIONAL_ID_NUMBER", "L01X00T47")}
    assert analyzer.analyze("unrelated: L01X00T47") == []
    assert analyzer.analyze("L01X00T47 " + label) == []  # No suffix context boost.
    assert analyzer.analyze(label + " alpha beta gamma delta epsilon zeta L01X00T47") == []


def test_context_uses_words_not_substrings_and_contiguous_phrases():
    analyzer = RulesAnalyzer(policy(entities=("NATIONAL_ID_NUMBER",), threshold=0.65))
    assert analyzer.analyze("avoidnummer: L01X00T47") == []
    assert analyzer.analyze("id unrelated number: L01X00T47") == []
    assert values(analyzer, "the number of my identity cards is L01X00T47") == {
        ("NATIONAL_ID_NUMBER", "L01X00T47")
    }


def test_country_patterns_survive_english_only_and_multilingual_runs_are_unique(monkeypatch):
    entities = ("BSN", "VAT_NUMBER", "STEUERNUMMER", "POSTAL_CODE", "PASSWORD_OR_SECRET")
    analyzer = RulesAnalyzer(policy(entities=entities))
    text = "123456782 NL123456789B01 12/345/67890 1234 AB password=example-only"
    expected = values(analyzer, text)
    assert {entity for entity, _ in expected} == set(entities)
    calls = []
    rule = next(rule for rule in analyzer._rules if rule.supported_entities == ["BSN"])
    analyze = rule.analyze

    def counted(*args, **kwargs):
        calls.append(1)
        return analyze(*args, **kwargs)

    monkeypatch.setattr(rule, "analyze", counted)
    active = policy(languages=("en", "de", "nl"), entities=entities)
    assert values(analyzer, text, active) == expected
    assert len(calls) == 1
    matches = analyzer.analyze(text, active)
    assert len(matches) == len({(match.entity_type, match.start, match.end) for match in matches})


def test_custom_language_selection_threshold_and_request_policy_fail_closed():
    custom = [
        {
            "name": "Reference",
            "entity": "REFERENCE",
            "regex": r"REF-\d{4}",
            "score": 0.65,
            "supportedLanguages": ["de", "nl"],
        }
    ]
    active = policy(languages=("de",), entities=("REFERENCE",), custom=custom, threshold=0.65)
    analyzer = RulesAnalyzer(active)
    text = "😀 REF-1234"
    (match,) = analyzer.analyze(text)
    assert (match.start, match.end, match.score) == (2, 10, 0.65)
    assert values(analyzer, text, policy(languages=("nl",), custom=custom)) == {
        ("REFERENCE", "REF-1234")
    }
    assert analyzer.analyze(text, policy(languages=("en",), custom=custom)) == []
    assert analyzer.analyze(text, policy(languages=("de",), custom=custom, threshold=0.66)) == []
    with pytest.raises(ValueError, match="loaded rule configuration"):
        analyzer.analyze(text, policy(languages=("de",)))
    with pytest.raises(ValueError, match="unknown entity"):
        RulesAnalyzer(policy(entities=("UNKNOWN",)))


def test_useful_predefined_country_rules_are_normalized_in_all_languages():
    analyzer = RulesAnalyzer(
        policy(
            languages=("nl",), entities=("TAX_ID", "PASSPORT_NUMBER", "BANK_ACCOUNT"), threshold=0.4
        )
    )
    assert ("TAX_ID", "900-70-0000") in values(analyzer, "taxpayer itin: 900-70-0000")
    assert ("PASSPORT_NUMBER", "123456789") in values(analyzer, "passport: 123456789")
    assert ("BANK_ACCOUNT", "123456789012") in values(analyzer, "bank account: 123456789012")


@pytest.mark.parametrize("language", ["en", "de", "nl"])
def test_custom_pattern_in_each_language_and_presidio_alias_is_not_rewritten(language):
    custom = [
        {
            "name": "Custom card",
            "entity": "CREDIT_CARD",
            "regex": r"TEST-\d{4}",
            "score": 0.8,
            "supportedLanguages": [language],
        }
    ]
    analyzer = RulesAnalyzer(
        policy(languages=(language,), entities=("CREDIT_CARD",), custom=custom)
    )
    assert values(analyzer, "TEST-1234") == {("CREDIT_CARD", "TEST-1234")}


@pytest.mark.parametrize(
    ("start", "end", "score"),
    [
        (0, 9, float("nan")),
        (0, 9, float("inf")),
        (0.5, 9, 0.9),
        (0, 8.5, 0.9),
        (False, 9, 0.9),
        (0, True, 0.9),
        (0, 9, True),
        (0, 9, False),
        (0, 9, "0.9"),
        (0, 9, -0.1),
        (0, 9, 1.1),
        (0, 9, 10**400),
    ],
)
def test_invalid_recognizer_output_fails_closed(monkeypatch, start, end, score):
    from types import SimpleNamespace

    analyzer = RulesAnalyzer(policy(entities=("BSN",)))
    rule = next(rule for rule in analyzer._rules if rule.supported_entities == ["BSN"])
    monkeypatch.setattr(
        rule,
        "analyze",
        lambda **kwargs: [SimpleNamespace(entity_type="BSN", start=start, end=end, score=score)],
    )
    with pytest.raises(ValueError, match="invalid result"):
        analyzer.analyze("123456782")

"""Normalized entity catalog and deterministic recognizer definitions."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Recognizer:
    """Describe a bounded regular-expression recognizer."""

    entity: str
    pattern: str
    source: str = "deterministic"


STEUERNUMMER_PATTERNS: tuple[tuple[str, str], ...] = (
    ("Baden-Wuerttemberg regional", r"(?<!\w)\d{5}/\d{5}(?!\w)"),
    ("two-digit regional", r"(?<!\w)\d{2}/\d{3}/\d{5}(?!\w)"),
    ("three-digit regional", r"(?<!\w)\d{3}/\d{3}/\d{5}(?!\w)"),
    ("Nordrhein-Westfalen regional", r"(?<!\w)\d{3}/\d{4}/\d{4}(?!\w)"),
    ("space-separated regional", r"(?<!\w)\d{2,3} \d{3} \d{5}(?!\w)"),
    (
        "country-wide",
        r"(?<!\w)(?:(?:10|11|21|22|23|24|26|27|28|30|31|32|40|41)\d{2}|[59]\d{3})0\d{8}(?!\w)",
    ),
)


ENTITY_CATALOG: tuple[str, ...] = (
    "SENSITIVE_TEXT",
    "PERSON_NAME",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "STREET_ADDRESS",
    "POSTAL_CODE",
    "CITY",
    "DATE_OF_BIRTH",
    "PLACE_OF_BIRTH",
    "IP_ADDRESS",
    "BANK_ACCOUNT",
    "IBAN",
    "CREDIT_CARD_NUMBER",
    "PASSPORT_NUMBER",
    "DRIVERS_LICENSE_NUMBER",
    "NATIONAL_ID_NUMBER",
    "BSN",
    "TAX_ID",
    "STEUERNUMMER",
    "VAT_NUMBER",
    "HEALTH_INSURANCE_ID",
    "MEDICAL_RECORD_ID",
    "USERNAME",
    "PASSWORD_OR_SECRET",
    "VEHICLE_REGISTRATION",
    "SIGNATURE",
)

RECOGNIZERS: tuple[Recognizer, ...] = (
    Recognizer(
        "EMAIL_ADDRESS", r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),
    Recognizer("PHONE_NUMBER", r"(?<!\w)(?:\+?[0-9][0-9 .()/-]{7,}[0-9])(?!\w)"),
    Recognizer("IBAN", r"\b[A-Z]{2}[0-9]{2}(?:[ ]?[A-Z0-9]){11,30}\b"),
    Recognizer("CREDIT_CARD_NUMBER", r"\b(?:\d[ -]*?){13,19}\b"),
    Recognizer("IP_ADDRESS", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    Recognizer("BSN", r"\b\d{9}\b"),
    *(Recognizer("STEUERNUMMER", pattern) for _name, pattern in STEUERNUMMER_PATTERNS),
    Recognizer("VAT_NUMBER", r"\b(?:NL\d{9}B\d{2}|DE\d{9})\b"),
    Recognizer(
        "PASSWORD_OR_SECRET", r"(?i)\b(?:password|passwort|api[ _-]?key|secret)\s*[:=]\s*\S+"
    ),
)


def compiled_recognizers() -> tuple[tuple[Recognizer, re.Pattern[str]], ...]:
    """Compile the normalized recognizers once for the in-process service."""
    return tuple((recognizer, re.compile(recognizer.pattern)) for recognizer in RECOGNIZERS)

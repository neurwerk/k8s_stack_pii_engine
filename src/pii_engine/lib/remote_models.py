"""Reviewed model pins and explicit policy mappings for the supported NER recipes."""

KSERVE_MODEL_PINS = {
    "en": (
        "ai4privacy/llama-ai4privacy-english-anonymiser-openpii",
        "1efb619f6d9f5a84b5d6ccf65f1f45df961a2167",
    ),
    "de": (
        "OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1",
        "fa8d9c0186635e1ad74f667d320b9d90955523ad",
    ),
}

# SHA-256 of the exact runtime files fetched from the immutable revisions above.
# This prevents a deployment from accidentally pairing server probabilities with
# a different tokenizer or class-ID ordering, even if its local file hashes match.
KSERVE_TOKENIZER_SHA256 = {
    "en": {
        "config.json": "c0dc3681d2633dcca592c449a75f188f15a9c5c834951202ce9e46c3a27126f7",
        "special_tokens_map.json": (
            "ea97ecdbcc73713039d8d64dbb05e3689495c96657fbd9a18f5bed381be81049"
        ),
        "tokenizer.json": "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
        "tokenizer_config.json": "292edffb55c1afeaad5bd7cf21bae69563c7b7d0cf7f72dfce24132b4267f003",
    },
    "de": {
        "added_tokens.json": "dc046d04c9b0ada7ae6f1dc89c465801799acdf0c9a6aab8c15a1b2d5ca4e91f",
        "config.json": "3e36050544bff921a90874052b19bea430f21e7f27f55dc6459de808d19a100c",
        "special_tokens_map.json": (
            "9463f61e1b109a8eb4688b829260d7c6b1e6dff04c98ff7269bb89e2b92369b9"
        ),
        "spm.model": "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
        "tokenizer.json": "b10b7a38aab2e62572ac50a805095f1fb9d7096d9a9384f5ca2d9b4457c84b33",
        "tokenizer_config.json": "e6c4c771911c211618a2d46488dc7e9e499f051773cd4ff4faeca6fe55fdf569",
    },
}

GLINER_LABEL_MAPPING = {
    "person": "PERSON_NAME",
    "email": "EMAIL_ADDRESS",
    "phone number": "PHONE_NUMBER",
    "address": "STREET_ADDRESS",
    "date of birth": "DATE_OF_BIRTH",
    "iban": "IBAN",
    "credit card number": "CREDIT_CARD_NUMBER",
    "passport number": "PASSPORT_NUMBER",
    "medical condition": "SENSITIVE_TEXT",
    "medication": "SENSITIVE_TEXT",
}

# These labels have no exact normalized category. Preserve their evidence as
# SENSITIVE_TEXT instead of guessing an entity type or discarding the finding.
GERMAN_LABEL_MAPPING = dict.fromkeys(
    (
        "ACCOUNTNAME",
        "AGE",
        "AMOUNT",
        "BIC",
        "BITCOINADDRESS",
        "COUNTY",
        "CREDITCARDISSUER",
        "CURRENCY",
        "CURRENCYCODE",
        "CURRENCYNAME",
        "CURRENCYSYMBOL",
        "DATE",
        "ETHEREUMADDRESS",
        "EYECOLOR",
        "GENDER",
        "GPSCOORDINATES",
        "HEIGHT",
        "IMEI",
        "JOBDEPARTMENT",
        "JOBTITLE",
        "LITECOINADDRESS",
        "MACADDRESS",
        "MASKEDNUMBER",
        "OCCUPATION",
        "ORDINALDIRECTION",
        "ORGANIZATION",
        "SEX",
        "STATE",
        "TIME",
        "URL",
        "USERAGENT",
        "VIN",
    ),
    "SENSITIVE_TEXT",
)
GERMAN_LABEL_MAPPING.update(
    {
        "CITY": "CITY",
        "BANKACCOUNT": "BANK_ACCOUNT",
        "BUILDINGNUMBER": "STREET_ADDRESS",
        "CREDITCARD": "CREDIT_CARD_NUMBER",
        "CVV": "PASSWORD_OR_SECRET",
        "DATEOFBIRTH": "DATE_OF_BIRTH",
        "EMAIL": "EMAIL_ADDRESS",
        "FIRSTNAME": "PERSON_NAME",
        "IBAN": "IBAN",
        "IPADDRESS": "IP_ADDRESS",
        "LASTNAME": "PERSON_NAME",
        "MIDDLENAME": "PERSON_NAME",
        "PASSWORD": "PASSWORD_OR_SECRET",
        "PHONE": "PHONE_NUMBER",
        "PIN": "PASSWORD_OR_SECRET",
        "PREFIX": "PERSON_NAME",
        "SECONDARYADDRESS": "STREET_ADDRESS",
        "SSN": "NATIONAL_ID_NUMBER",
        "STREET": "STREET_ADDRESS",
        "USERNAME": "USERNAME",
        "VRM": "VEHICLE_REGISTRATION",
        "ZIPCODE": "POSTAL_CODE",
    }
)

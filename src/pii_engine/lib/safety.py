"""Built-in safety rules retained from the reviewed policy implementation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SafetyRule:
    """Define one fail-closed blocking expression."""

    name: str
    pattern: str
    message: str
    compiled: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Compile once so invalid built-ins fail process initialization."""
        object.__setattr__(self, "compiled", re.compile(self.pattern))

    def matches(self, text: str) -> bool:
        """Return whether this rule matches one original text leaf."""
        return bool(self.compiled.search(text))


SAFETY_RULES: tuple[SafetyRule, ...] = (
    SafetyRule(
        "promptInjection",
        r"(?i)(ignore|disregard|forget|override|bypass|skip|dismiss|drop|abandon)\s+"
        r"(all\s+|any\s+|your\s+)?(previous|prior|earlier|above|existing|current|"
        r"original|initial|given|preset)\s+(instructions|rules|guidelines|directives|"
        r"constraints|restrictions|prompts|programming|configuration)",
        "Prompt injection detected",
    ),
    SafetyRule(
        "jailbreak",
        r"(?i)(you are now|you're now|from now on you are|henceforth you are|you have "
        r"become|you are no longer)\s+(a |an |the )?(unrestricted|unfiltered|uncensored|"
        r"unlimited|jailbroken|evil|malicious|dangerous|DAN|unethical|amoral|rogue|hacker)",
        "Jailbreak attempt detected",
    ),
    SafetyRule(
        "systemPromptExtraction",
        r"(?i)(show|print|display|reveal|output|tell|give|share|repeat|recite|echo|dump|"
        r"expose|leak|disclose)\s+(me\s+)?(your|the)?\s*(system|initial|original|first|"
        r"hidden|secret|internal|underlying|prepended)\s*(prompt|instructions?|message|"
        r"context|configuration|rules?|directives?|preamble)",
        "System prompt extraction attempt detected",
    ),
    SafetyRule(
        "harmfulContent",
        r"(?i)(how to|instructions for|guide to|steps to|recipe for|teach me to)\s+"
        r"(make|build|create|construct|assemble|synthesiz(e|ing)|manufactur(e|ing))\s+"
        r"(a |an )?(bomb|explosive|detonator|IED|pipe bomb|chemical weapon|biological "
        r"weapon|dirty bomb|napalm|nerve agent|ricin|sarin)",
        "Harmful content detected",
    ),
    SafetyRule(
        "encodingEvasion",
        r"(?i)(decode|decrypt|decipher|translate|interpret|execute|run|follow|obey|"
        r"respond|answer|reply|output)(?:\s+\w+){0,3}\s+(base64|base-64|b64|rot13|"
        r"rot-13|hex|hexadecimal|binary|encoded|encrypted|obfuscated)\s*(encoding|"
        r"string|text|message|instruction|command|payload)?",
        "Encoding evasion attempt detected",
    ),
    SafetyRule(
        "selfHarm",
        r"(?i)(how to|ways to|methods?\s+(for|of|to)|I\s+(want|need|wish|would like|"
        r"am going)\s+to)\s+(kill(ing)? yourself|commit(ting)? suicide|end(ing)? "
        r"(my|your|one's) life|self-?harm(ing)?|cut(ting)? yourself|hurt(ing)? "
        r"yourself|overdos(e|ing) on|harm(ing)? (myself|yourself|themself|oneself))",
        "Self-harm content detected",
    ),
)

SAFETY_BY_NAME = {rule.name: rule for rule in SAFETY_RULES}

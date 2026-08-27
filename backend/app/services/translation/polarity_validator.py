import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PolarityValidationResult:
    passed: bool
    explicit_negation_lost: bool
    ambiguous_negation_risk: bool


class PolarityValidator:
    EXPLICIT_SOURCE_PATTERN = re.compile(
        r"\b(?:no\s+longer|must\s+not|should\s+not|may\s+not|did\s+not|does\s+not|"
        r"is\s+not|are\s+not|was\s+not|were\s+not|cannot|can['’]?t|never|none|neither|nor|not)\b|"
        r"\b[A-Za-z]+n['’]t\b",
        re.IGNORECASE,
    )
    AMBIGUOUS_SOURCE_PATTERN = re.compile(r"\bwithout\b", re.IGNORECASE)
    TARGET_NEGATION_PATTERN = re.compile(
        r"(?<!\w)(?:không\s+còn|không\s+được|không\s+thể|không\s+nên|"
        r"không\s+phải|không\s+có|mà\s+không|không|chưa|chẳng|chả)(?!\w)",
        re.IGNORECASE,
    )

    @classmethod
    def validate(cls, source_text: str, translated_text: str) -> PolarityValidationResult:
        source = source_text or ""
        target = translated_text or ""
        target_has_negation = bool(cls.TARGET_NEGATION_PATTERN.search(target))
        explicit_source = bool(cls.EXPLICIT_SOURCE_PATTERN.search(source))
        ambiguous_source = bool(cls.AMBIGUOUS_SOURCE_PATTERN.search(source))
        explicit_loss = explicit_source and not target_has_negation
        ambiguous_risk = ambiguous_source and not target_has_negation
        return PolarityValidationResult(
            passed=not explicit_loss,
            explicit_negation_lost=explicit_loss,
            ambiguous_negation_risk=ambiguous_risk,
        )

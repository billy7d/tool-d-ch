import re
from collections import Counter
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ReferenceToken:
    reference_type: str
    identifier: str
    raw: str

    @property
    def semantic_key(self) -> str:
        return f"{self.reference_type}:{self.identifier.upper()}"


@dataclass(frozen=True)
class ReferenceValidationResult:
    passed: bool
    missing: List[ReferenceToken]


class ReferenceValidator:
    ALIASES = {
        "figure": "FIGURE",
        "fig": "FIGURE",
        "hình": "FIGURE",
        "table": "TABLE",
        "bảng": "TABLE",
        "chapter": "CHAPTER",
        "ch": "CHAPTER",
        "chương": "CHAPTER",
        "section": "SECTION",
        "sec": "SECTION",
        "mục": "SECTION",
        "phần": "SECTION",
        "tiết": "SECTION",
    }
    LABEL_PATTERN = "|".join(
        sorted((re.escape(alias) for alias in ALIASES), key=len, reverse=True)
    )
    IDENTIFIER_PATTERN = r"(?:\d+(?:\.\d+)*|[A-Za-z](?:\.\d+|-[A-Za-z0-9]+)*)"
    REFERENCE_PATTERN = re.compile(
        rf"(?<!\w)(?P<label>{LABEL_PATTERN})\.?(?!\w)\s+(?P<identifier>{IDENTIFIER_PATTERN})(?![\w-]|\.\d)",
        re.IGNORECASE,
    )

    @classmethod
    def extract(cls, text: str) -> List[ReferenceToken]:
        tokens: List[ReferenceToken] = []
        for match in cls.REFERENCE_PATTERN.finditer(text or ""):
            label = match.group("label").casefold().rstrip(".")
            tokens.append(ReferenceToken(
                reference_type=cls.ALIASES[label],
                identifier=match.group("identifier"),
                raw=match.group(0),
            ))
        return tokens

    @classmethod
    def validate(cls, source_text: str, translated_text: str) -> ReferenceValidationResult:
        source_tokens = cls.extract(source_text)
        target_tokens = cls.extract(translated_text)
        missing_counts = Counter(token.semantic_key for token in source_tokens) - Counter(
            token.semantic_key for token in target_tokens
        )
        remaining = Counter(missing_counts)
        missing: List[ReferenceToken] = []
        for token in source_tokens:
            if remaining[token.semantic_key] > 0:
                missing.append(token)
                remaining[token.semantic_key] -= 1
        return ReferenceValidationResult(not missing, missing)

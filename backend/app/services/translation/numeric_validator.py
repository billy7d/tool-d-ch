import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional

from app.services.translation.reference_validator import ReferenceValidator


UNIT_PATTERN = r"°C|°F|kHz|MHz|GHz|USD|EUR|kg|km|cm|mm|MB|GB|ms|Hz|s"
NUMERIC_PATTERN = re.compile(
    rf"(?P<currency>[$€£])?\s*(?P<number>[+-]?\d+(?:[.,]\d+)*)\s*(?P<percent>%)?\s*(?P<unit>{UNIT_PATTERN})?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NumericToken:
    raw: str
    normalized_value: str
    sign: int
    percent: bool
    currency: Optional[str]
    unit: Optional[str]

    @property
    def comparison_key(self) -> str:
        return "|".join((
            self.normalized_value,
            "%" if self.percent else "",
            (self.currency or "").upper(),
            (self.unit or "").upper(),
        ))


@dataclass(frozen=True)
class NumericValidationResult:
    passed: bool
    missing: List[NumericToken]
    unexpected: List[NumericToken]


def _normalize_decimal(raw: str) -> str:
    sign = ""
    value = raw.strip()
    if value[:1] in {"+", "-"}:
        sign, value = value[0], value[1:]
    if "." in value and "," in value:
        decimal_separator = "." if value.rfind(".") > value.rfind(",") else ","
        grouping_separator = "," if decimal_separator == "." else "."
        value = value.replace(grouping_separator, "").replace(decimal_separator, ".")
    elif "." in value or "," in value:
        separator = "." if "." in value else ","
        chunks = value.split(separator)
        if len(chunks) > 2 or (len(chunks) == 2 and len(chunks[1]) == 3 and 1 <= len(chunks[0]) <= 3):
            value = "".join(chunks)
        else:
            value = ".".join(chunks)
    try:
        decimal_value = Decimal((sign or "") + value)
    except InvalidOperation:
        return raw
    normalized = format(decimal_value.normalize(), "f")
    return "0" if normalized in {"-0", "+0"} else normalized


class NumericValidator:
    @staticmethod
    def extract(text: str) -> List[NumericToken]:
        tokens: List[NumericToken] = []
        # Mã số của Figure/Section thuộc validator reference, không phải numeric fact.
        reference_spans = [
            (match.start(), match.end())
            for match in ReferenceValidator.REFERENCE_PATTERN.finditer(text or "")
        ]
        for match in NUMERIC_PATTERN.finditer(text or ""):
            if any(start <= match.start() < end for start, end in reference_spans):
                continue
            raw_number = match.group("number")
            currency_symbol = match.group("currency")
            unit = match.group("unit")
            currency = currency_symbol
            if unit and unit.upper() in {"USD", "EUR"}:
                currency, unit = unit.upper(), None
            tokens.append(NumericToken(
                raw=match.group(0).strip(),
                normalized_value=_normalize_decimal(raw_number),
                sign=-1 if raw_number.startswith("-") else 1,
                percent=bool(match.group("percent")),
                currency=currency,
                unit=unit,
            ))
        return tokens

    @classmethod
    def validate(cls, source_text: str, translated_text: str) -> NumericValidationResult:
        source_tokens = cls.extract(source_text)
        target_tokens = cls.extract(translated_text)
        source_counts = Counter(token.comparison_key for token in source_tokens)
        target_counts = Counter(token.comparison_key for token in target_tokens)
        missing_counts = source_counts - target_counts
        unexpected_counts = target_counts - source_counts

        def expand(tokens: List[NumericToken], counts: Dict[str, int]) -> List[NumericToken]:
            remaining = Counter(counts)
            result: List[NumericToken] = []
            for token in tokens:
                if remaining[token.comparison_key] > 0:
                    result.append(token)
                    remaining[token.comparison_key] -= 1
            return result

        missing = expand(source_tokens, missing_counts)
        unexpected = expand(target_tokens, unexpected_counts)
        # Candidate chỉ đạt khi bảo toàn đủ số nguồn và không tự thêm số mới.
        return NumericValidationResult(not missing and not unexpected, missing, unexpected)

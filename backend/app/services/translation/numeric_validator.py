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


_EN_SMALL = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_EN_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_EN_SCALES = {"hundred": 100, "thousand": 1000, "million": 1000000}
_EN_NUMBER_WORD = re.compile(
    r"\b(?:(?P<tens>" + "|".join(_EN_TENS) + r")(?:[- ](?P<tens_unit>" + "|".join(list(_EN_SMALL)[:9]) + r"))?"
    r"|(?P<small>" + "|".join(sorted(_EN_SMALL, key=len, reverse=True)) + r")|\ba)"
    r"(?:[- ](?P<scale>" + "|".join(_EN_SCALES) + r"))?\b",
    re.IGNORECASE,
)
_EN_PERCENT = re.compile(r"\s+per\s?cent\b", re.IGNORECASE)
_PERCENT_WORD = re.compile(r"\bper\s?cent\b|%", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?;](?:\s|$)")
# Tên tháng viết hoa: "the 27th of July, 1694" dịch là "ngày 27 tháng 7 năm 1694" không phải tự thêm số 7.
_EN_MONTHS = {
    name: index for index, name in enumerate(
        ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
         "November", "December"), start=1)
}
_EN_MONTH = re.compile(r"\b(" + "|".join(_EN_MONTHS) + r")\b")
_EN_ORDINALS = {
    "first": 1, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
    "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30,
}
_EN_ORDINAL = re.compile(r"\b(" + "|".join(_EN_ORDINALS) + r")\b", re.IGNORECASE)


def spelled_english_numbers(text: str) -> Counter:
    """Số viết bằng chữ trong nguồn: "when I was nine" dịch thành "khi tôi 9 tuổi" không phải số tự thêm."""
    values: Counter = Counter()
    for match in _EN_NUMBER_WORD.finditer(text or ""):
        if match.group("tens"):
            value = _EN_TENS[match.group("tens").lower()] + _EN_SMALL.get((match.group("tens_unit") or "").lower(), 0)
        elif match.group("small"):
            value = _EN_SMALL[match.group("small").lower()]
        elif match.group("scale"):
            value = 1  # "a hundred"
        else:
            continue  # mạo từ "a" đứng riêng
        if match.group("scale"):
            value *= _EN_SCALES[match.group("scale").lower()]
        percent = _EN_PERCENT.match(text, match.end())
        values[f"{value}%" if percent else str(value)] += 1
        if not percent and _percent_later_in_sentence(text, match.end()):
            # "at forty, and fifty, and sixty percent discount": "percent" áp cho cả dãy, bản dịch ghi "40%, 50%, 60%".
            values[f"{value}%"] += 1
    for match in _EN_MONTH.finditer(text or ""):
        values[str(_EN_MONTHS[match.group(1)])] += 1
    # "In November of his eighth year" -> "tháng 11 năm 8 tuổi".
    for match in _EN_ORDINAL.finditer(text or ""):
        values[str(_EN_ORDINALS[match.group(1).lower()])] += 1
    return values


def _percent_later_in_sentence(text: str, position: int) -> bool:
    rest = (text or "")[position:]
    end = _SENTENCE_END.search(rest)
    return bool(_PERCENT_WORD.search(rest[:end.start() if end else len(rest)]))


_VI_DIGITS = {1: "một", 2: "hai", 3: "ba", 4: "bốn", 5: "năm", 6: "sáu", 7: "bảy", 8: "tám", 9: "chín"}


def _vietnamese_number_pattern(value: int) -> Optional[str]:
    if value in _VI_DIGITS:
        return _VI_DIGITS[value]
    if value == 10:
        return "mười"
    if 11 <= value <= 19:
        return "mười " + ("lăm" if value == 15 else _VI_DIGITS[value - 10])
    if 20 <= value <= 99:
        tens = f"{_VI_DIGITS[value // 10]} mươi"
        unit = value % 10
        if not unit:
            return tens
        unit_word = {1: "mốt", 4: "(?:tư|bốn)", 5: "lăm"}.get(unit, _VI_DIGITS[unit])
        return f"{tens} {unit_word}"
    return None


def _spelled_in_vietnamese(key: str, text: str) -> int:
    """Số lần con số đã được viết bằng chữ tiếng Việt, tức là đã dùng hết số viết bằng chữ tương ứng của nguồn."""
    if not key.rstrip("%").isdigit():
        return 0
    pattern = _vietnamese_number_pattern(int(key.rstrip("%")))
    return len(re.findall(rf"(?<!\w){pattern}(?!\w)", text or "", re.IGNORECASE)) if pattern else 0


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
        spelled = spelled_english_numbers(source_text)
        # "ten years ago" đã dịch là "mười năm" thì chữ "lớp 10" là số tự thêm: chạy sách thật thấy "junior year" -> "lớp 10".
        for key in list(spelled):
            spelled[key] -= _spelled_in_vietnamese(key, translated_text)
        kept = []
        for token in unexpected:
            key = f"{token.normalized_value}%" if token.percent else token.normalized_value
            if not (token.currency or token.unit) and spelled[key] > 0:
                spelled[key] -= 1
            else:
                kept.append(token)
        unexpected = kept
        # Số nhỏ trong nguồn được bản dịch viết bằng chữ ("the 7th Anne" -> "năm thứ bảy của Anne") không phải là thiếu.
        # Chỉ dùng phần chữ số tiếng Việt chưa ứng với số viết bằng chữ của nguồn.
        english_spelled = spelled_english_numbers(source_text)
        spelled_out: Counter = Counter()
        kept = []
        for token in missing:
            key = token.normalized_value
            if token.percent or token.currency or token.unit or not key.isdigit():
                kept.append(token)
                continue
            if key not in spelled_out:
                spelled_out[key] = _spelled_in_vietnamese(key, translated_text) - english_spelled[key]
            if spelled_out[key] > 0:
                spelled_out[key] -= 1
            else:
                kept.append(token)
        missing = kept
        # Candidate chỉ đạt khi bảo toàn đủ số nguồn và không tự thêm số mới.
        return NumericValidationResult(not missing and not unexpected, missing, unexpected)

import re


class TermMatcher:
    @staticmethod
    def pattern(term: str) -> re.Pattern[str]:
        normalized = (term or "").strip()
        if not normalized:
            return re.compile(r"(?!x)x")
        inflection = r"(?:s|es|ed|ing)?" if re.fullmatch(r"[A-Za-z][A-Za-z\s'-]*", normalized) else ""
        return re.compile(
            rf"(?<!\w){re.escape(normalized)}{inflection}(?!\w)",
            flags=re.IGNORECASE | re.UNICODE,
        )

    @classmethod
    def contains(cls, text: str, term: str) -> bool:
        return bool(text and term and cls.pattern(term).search(text))

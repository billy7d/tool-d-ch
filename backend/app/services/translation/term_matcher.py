import re

# Chữ viết hoa đứng liền trước hoặc liền sau, trong cùng câu: "Grace Warrack", "Sister Grace".
_CAPITALIZED_AFTER = re.compile(r"\s+[A-Z][a-z]")
_CAPITALIZED_BEFORE = re.compile(r"[A-Z][\w’'-]*\s+$")


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

    @classmethod
    def contains_source_term(cls, text: str, term: str) -> bool:
        """Thuật ngữ glossary có mặt trong văn bản NGUỒN, không tính chỗ chỉ trùng chữ.

        Chạy sách thật: thuật ngữ "God → Chúa" khớp cả "their gods" và cho ra "những Chúa của họ"; thuật ngữ
        "grace" khớp tên người "Grace Warrack" và tên bị dịch thành "Ân Sủng". Vì vậy:
        - chữ viết hoa trong thuật ngữ phải viết hoa trong nguồn ("God" không khớp "gods", "AI" không khớp "ai");
        - thuật ngữ viết thường mà nguồn viết hoa và đứng sát một từ viết hoa khác thì là tên riêng.
        Chỉ dùng cho nguồn tiếng Anh; phía bản dịch vẫn dùng contains.
        """
        normalized = (term or "").strip()
        if not text or not normalized:
            return False
        for match in cls.pattern(normalized).finditer(text):
            found = match.group(0)
            if any(expected.isupper() and not actual.isupper() for expected, actual in zip(normalized, found)):
                continue
            if normalized.islower() and found[0].isupper() and not found.isupper():
                before = text[:match.start()]
                if _CAPITALIZED_AFTER.match(text, match.end()) or _CAPITALIZED_BEFORE.search(before):
                    continue
            return True
        return False

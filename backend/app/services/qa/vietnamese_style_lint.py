"""Lint văn phong tiếng Việt dạng luật, chạy deterministic và không tốn token.

Lint chỉ phát hiện dấu hiệu dịch sát chữ phổ biến; nó không tự sửa câu và không có
quyền từ chối bản dịch. Kết quả được dùng làm gợi ý cho naturalness critic và cho
bước editorial rewrite. WARNING là tín hiệu mạnh, INFO chỉ là gợi ý cân nhắc.
"""

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Pattern


STYLE_LINT_VERSION = "vietnamese-style-lint-v2"
LONG_SENTENCE_SYLLABLES = 60
PASSIVE_DENSITY_LIMIT = 3

# Domain mà câu dài, bị động trang trọng là hợp lệ; chỉ hạ mức cảnh báo, không bỏ luật.
_FORMAL_DOMAINS = {"LEGAL", "ACADEMIC", "TECHNICAL"}

# Câu dặn critic dùng chung cho mọi provider: hint chỉ là ứng viên, critic phải tự xác nhận.
CRITIC_STYLE_HINT_INSTRUCTION = (
    "deterministic_style_hints are rule-based candidates from a Vietnamese style linter and may be false "
    "positives. Verify each one in context: report it as an issue only if it truly reads as unnatural "
    "Vietnamese, and ignore it otherwise."
)
_FORMAL_DOWNGRADE_TYPES = {"PASSIVE_VOICE", "SENTENCE_FLOW", "NOMINALIZATION"}

_CLAUSE = r"[^.,;:!?\n]"
_FUNCTION_WORDS = r"của|các|những|và|là|được|bị|cho|với|trong|một|đã|sẽ|đang|thì|mà|này|đó"
# Danh từ đã từ vựng hóa với "sự", dùng tự nhiên: không phải danh hóa kiểu dịch.
_LEXICAL_SU = r"(?:thật|kiện|việc|cố|nghiệp|vụ|tích|vật|sống|thể|khác biệt)\b"


@dataclass(frozen=True)
class StyleLintIssue:
    rule: str
    type: str
    severity: str
    target_span: str
    message: str
    suggestion: str = ""

    def to_hint(self) -> Dict[str, str]:
        return asdict(self)

    def to_naturalness_issue(self) -> Dict[str, str]:
        """Cùng shape với issue của critic để editorial rewrite dùng chung."""
        return {
            "type": self.type,
            "severity": self.severity,
            "source_span": "",
            "target_span": self.target_span,
            "message": f"[lint:{self.rule}] {self.message}" + (f" Gợi ý: {self.suggestion}" if self.suggestion else ""),
        }


@dataclass(frozen=True)
class _PatternRule:
    rule: str
    type: str
    severity: str
    pattern: Pattern[str]
    message: str
    suggestion: str = ""
    # Luật chỉ áp dụng khi điều kiện trên câu nguồn tiếng Anh thỏa mãn.
    source_condition: Optional[Callable[[str], bool]] = None


def _compile(pattern: str) -> Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


_PERSON_PRONOUN = re.compile(r"\b(?:he|she|him|her|his|hers)\b", re.IGNORECASE)
_IT_PRONOUN = re.compile(r"\b(?:it|its|they|them)\b", re.IGNORECASE)

_PATTERN_RULES = (
    _PatternRule(
        "PASSIVE_BY_AGENT", "PASSIVE_VOICE", "WARNING",
        _compile(rf"\b(?:được|bị)\b{_CLAUSE}{{1,40}}?\bbởi\b"),
        "Cấu trúc bị động 'được/bị … bởi' mô phỏng 'by' của tiếng Anh.",
        "Đưa chủ thể hành động lên làm chủ ngữ.",
    ),
    _PatternRule(
        "MOT_CACH_ADVERB", "REDUNDANCY", "WARNING",
        # Loại trừ danh từ "một cách (tiếp cận/khác/để…)" vốn tự nhiên.
        _compile(
            r"\bmột cách\s+(?!(?:tiếp cận|khác|nào|mới|hay|làm|duy nhất|giải quyết)\b)"
            rf"(?!{_CLAUSE}{{0,25}}\bđể\b)[^\s.,;:!?]+"
        ),
        "'một cách + tính từ' thường là dịch sát trạng từ '-ly'.",
        "Dùng thẳng tính từ/trạng từ sau động từ (vd: 'giải thích rõ' thay vì 'giải thích một cách rõ ràng').",
    ),
    _PatternRule(
        "LIGHT_VERB_VIEC", "NOMINALIZATION", "WARNING",
        _compile(r"\b(?:thực hiện|tiến hành)\s+việc\b"),
        "'thực hiện/tiến hành việc …' là cụm đệm danh hóa.",
        "Dùng trực tiếp động từ chính.",
    ),
    _PatternRule(
        "IT_IS_ADJ", "LITERAL_CALQUE", "WARNING",
        _compile(
            r"\b(?:nó|điều này|điều đó)\s+(?:là|thì)\s+(?:rất\s+|thật\s+)?"
            r"(?:quan trọng|cần thiết|khó|dễ|rõ ràng|có thể|hợp lý|cần)\b"
        ),
        "Dịch sát mẫu 'It is + adj' của tiếng Anh.",
        "Viết thẳng vị ngữ: 'Cần …', 'Rõ ràng là …', 'Điều quan trọng là …'.",
    ),
    _PatternRule(
        "LA_QUAN_TRONG_DE", "LITERAL_CALQUE", "WARNING",
        _compile(r"\blà\s+(?:rất\s+)?quan trọng\s+(?:để|khi|rằng)\b"),
        "Dịch sát 'is important to/that'.",
        "Dùng 'cần …' hoặc 'điều quan trọng là …'.",
    ),
    _PatternRule(
        "IT_IS_ADJ_TO", "LITERAL_CALQUE", "WARNING",
        _compile(
            r"\b(?:là\s+(?:rất\s+)?(?:dễ dàng|dễ|khó khăn|khó|bình thường|cần thiết)\s+để"
            rf"|nó là\b{_CLAUSE}{{1,30}}?\bđể|làm cho nó)\b"
        ),
        "Dịch sát mẫu 'is easy to / it is … to / makes it …'.",
        "Đưa hành động lên làm chủ ngữ: 'Hướng dẫn rất dễ làm theo', 'Cảm thấy … là điều bình thường'.",
    ),
    _PatternRule(
        "LIGHT_VERB_ACTION", "NOMINALIZATION", "WARNING",
        _compile(r"\b(?:thực hiện|tiến hành)\s+(?:một\s+)?(?:hành động|sự)\b|\b(?:làm|thực hiện|tạo ra|đưa ra|cung cấp)\s+một sự\b"),
        "Động từ nhẹ + danh từ ('thực hiện hành động', 'làm một sự …') là dịch sát 'take action/make a …'.",
        "Dùng thẳng động từ: 'hành động', 'nghỉ', 'tạo khác biệt'.",
    ),
    _PatternRule(
        "DUA_RA_QUYET_DINH", "NOMINALIZATION", "INFO",
        _compile(r"\bđưa ra\s+(?:một\s+)?quyết định\b"),
        "'đưa ra (một) quyết định' thường có thể rút gọn thành động từ 'quyết định'.",
        "Dùng 'quyết định' làm động từ khi không cần nhấn mạnh.",
    ),
    _PatternRule(
        "CALQUE_PHRASE", "LITERAL_CALQUE", "WARNING",
        _compile(
            r"\b(?:dẫn đến trong|chỉ nếu|đưa vào (?:trong )?tài khoản|với sự tôn trọng (?:đối với|tới)"
            r"|khác(?: biệt)? từ những gì|được chịu|là bị)\b"
        ),
        "Cụm dịch sát nguyên văn tiếng Anh ('result in', 'only if', 'take into account', 'with respect to'…).",
        "Dùng cụm tiếng Việt tương đương: 'giúp/dẫn đến', 'chỉ khi', 'tính đến', 'về', 'khác với'.",
    ),
    _PatternRule(
        "DE_CO_THE", "REDUNDANCY", "INFO",
        _compile(r"\bđể có thể\b"),
        "'để có thể' thường thừa 'có thể'.",
        "Thường chỉ cần 'để'.",
    ),
    _PatternRule(
        "PLURAL_SU_NOUN", "NOMINALIZATION", "WARNING",
        _compile(rf"\b(?:các|những)\s+sự\s+(?!{_LEXICAL_SU})[^\s.,;:!?]+"),
        "'các/những sự …' là danh hóa số nhiều kiểu tiếng Anh (vd 'các sự chậm trễ').",
        "Bỏ 'sự': 'các chậm trễ' -> 'việc chậm trễ' hoặc viết lại bằng động từ.",
    ),
    _PatternRule(
        "MOT_SU_NOUN", "NOMINALIZATION", "INFO",
        _compile(rf"\bmột\s+sự\s+(?!{_LEXICAL_SU})[^\s.,;:!?]+"),
        "'một sự …' thường dịch sát mạo từ 'a/an' trước danh từ trừu tượng.",
        "Cân nhắc bỏ 'một sự' hoặc dùng động từ.",
    ),
    _PatternRule(
        "PAIRED_CONCESSIVE", "REDUNDANCY", "WARNING",
        _compile(r"\b(?:mặc dù|dù|tuy rằng|tuy)\b[^.!?;]{1,80}?,\s*nhưng\b"),
        "'Dù/Mặc dù …, nhưng …' thừa quan hệ từ.",
        "Giữ một vế: 'Dù …, …' hoặc '…, nhưng …'.",
    ),
    _PatternRule(
        "FOREIGN_WORD", "LITERAL_CALQUE", "WARNING",
        _compile(r"(?<![.\w])(?:brutto|brut|netto|net|gross)(?![.\w])"),
        "Từ nước ngoài lẫn vào văn xuôi tiếng Việt.",
        "Dùng thuật ngữ tiếng Việt: 'gộp', 'ròng'.",
    ),
    _PatternRule(
        "LA_ADJECTIVE", "LITERAL_CALQUE", "WARNING",
        _compile(r"\blà\s+(?:rất\s+)?(?:đáng kể|quan trọng|cần thiết|hiệu quả)(?=\s*[,.;!?]|\s*$)"),
        "'là + tính từ' cuối vế là dịch sát 'is significant/important'.",
        "Bỏ 'là': 'mối tương quan đáng kể', 'việc này rất quan trọng'.",
    ),
    _PatternRule(
        "AI_CUNG_KHONG", "LITERAL_CALQUE", "WARNING",
        _compile(r"\bai cũng không\b"),
        "'Ai cũng không …' là dịch sát 'Nobody …'.",
        "Dùng 'Không ai …'.",
    ),
    _PatternRule(
        "DUPLICATED_FUNCTION_WORD", "REDUNDANCY", "WARNING",
        _compile(rf"\b({_FUNCTION_WORDS})\s+\1\b"),
        "Hư từ bị lặp liền nhau.",
        "Bỏ một lần lặp.",
    ),
    _PatternRule(
        "CUA_CHAIN", "WORD_ORDER", "WARNING",
        _compile(rf"\bcủa\b{_CLAUSE}{{1,25}}?\bcủa\b{_CLAUSE}{{1,25}}?\bcủa\b"),
        "Chuỗi ba 'của' trong một vế, dấu hiệu giữ trật tự cụm danh từ tiếng Anh.",
        "Tách câu hoặc lược bớt 'của' khi quan hệ sở hữu đã rõ.",
    ),
    _PatternRule(
        "SU_OPENING_NOMINAL", "NOMINALIZATION", "INFO",
        _compile(r"(?:^|(?<=[.!?…]\s))(?:Sự|Việc)\s+[^\s.,;:!?]+\s+(?:[^\s.,;:!?]+\s+)?của\b"),
        "Mở câu bằng 'Sự/Việc … của' là danh hóa nặng.",
        "Cân nhắc biến danh từ thành động từ với chủ ngữ rõ ràng.",
    ),
    _PatternRule(
        "DIEU_MA_RELATIVE", "LITERAL_CALQUE", "INFO",
        _compile(r"\b(?:điều|cái|những gì)\s+mà\b"),
        "'điều/cái mà' hay là dịch sát 'what/which'.",
        "Thường có thể bỏ 'mà' hoặc viết lại mệnh đề.",
    ),
    _PatternRule(
        "NO_FOR_PERSON", "PRONOUN_REFERENCE", "INFO",
        _compile(r"\bnó\b"),
        "Câu nguồn nói về người (he/she) nhưng bản dịch dùng 'nó'.",
        "Dùng đại từ hoặc danh xưng phù hợp với nhân vật.",
        source_condition=lambda source: bool(_PERSON_PRONOUN.search(source)) and not _IT_PRONOUN.search(source),
    ),
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
_PASSIVE_MARKER = re.compile(r"\b(?:đã|sẽ|đang|vừa|được)\s+(?:được|bị)\b|\bbị\b", re.IGNORECASE)


def _snippet(text: str, start: int, end: int, pad: int = 12) -> str:
    return text[max(0, start - pad):min(len(text), end + pad)].strip()


class VietnameseStyleLint:
    """Tập luật nhỏ, ưu tiên ít báo nhầm hơn là bắt hết mọi lỗi."""

    @classmethod
    def lint(cls, text: str, source_text: str = "", document_type: str = "GENERAL") -> List[StyleLintIssue]:
        if not text or not text.strip():
            return []
        normalized = unicodedata.normalize("NFC", text)
        source = source_text or ""
        issues: List[StyleLintIssue] = []

        for rule in _PATTERN_RULES:
            if rule.source_condition and not rule.source_condition(source):
                continue
            for match in rule.pattern.finditer(normalized):
                issues.append(StyleLintIssue(
                    rule.rule, rule.type, rule.severity,
                    _snippet(normalized, match.start(), match.end()),
                    rule.message, rule.suggestion,
                ))

        for sentence in _SENTENCE_SPLIT.split(normalized):
            syllables = len(sentence.split())
            if syllables > LONG_SENTENCE_SYLLABLES:
                issues.append(StyleLintIssue(
                    "LONG_SENTENCE", "SENTENCE_FLOW", "INFO",
                    sentence[:80].strip() + "…",
                    f"Câu dài {syllables} âm tiết, dễ khó đọc.",
                    "Cân nhắc tách câu nếu không làm mất quan hệ logic.",
                ))

        passive_count = len(_PASSIVE_MARKER.findall(normalized))
        if passive_count >= PASSIVE_DENSITY_LIMIT:
            issues.append(StyleLintIssue(
                "PASSIVE_DENSITY", "PASSIVE_VOICE", "INFO", "",
                f"Đoạn có {passive_count} cấu trúc bị động.",
                "Chuyển một số câu sang chủ động khi chủ thể đã rõ.",
            ))

        if str(document_type or "").upper() in _FORMAL_DOMAINS:
            issues = [
                StyleLintIssue(item.rule, item.type, "INFO", item.target_span, item.message, item.suggestion)
                if item.type in _FORMAL_DOWNGRADE_TYPES else item
                for item in issues
            ]
        return issues

    @classmethod
    def warnings(cls, text: str, source_text: str = "", document_type: str = "GENERAL") -> List[StyleLintIssue]:
        return [item for item in cls.lint(text, source_text, document_type) if item.severity == "WARNING"]

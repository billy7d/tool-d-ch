"""Sổ xưng hô (address terms) theo project.

Tiếng Anh chỉ có "I/you/he/she", còn tiếng Việt phải chọn anh/em/chị/cô/tôi/bạn... theo
quan hệ và sắc thái. Sổ này khóa các lựa chọn đó để mọi chương dùng nhất quán. Dữ liệu
nằm trong `style_guide["address_terms"]` nên không cần migration:

    {
      "reader": "bạn",                 # tác giả gọi người đọc
      "narrator": "tôi",               # người kể ngôi thứ nhất tự xưng
      "characters": {"Alice": "cô"},   # cách gọi nhân vật ở ngôi thứ ba
      "pairs": [{"speaker": "Alice", "listener": "Bob",
                 "self": "em", "other": "anh", "note": "người yêu"}]
    }
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.services.translation.term_matcher import TermMatcher


ADDRESS_TERMS_VERSION = "address-terms-v1"

# Mặc định cách gọi người đọc khi project chưa cấu hình; None nghĩa là để model tự chọn.
DEFAULT_READER_TERMS: Dict[str, Optional[str]] = {
    "GENERAL": "bạn",
    "SELF_HELP": "bạn",
    "BUSINESS": "bạn",
    "TECHNICAL": "bạn",
    "FINANCE": "bạn",
    "ACADEMIC": None,
    "LEGAL": None,
    "LITERATURE": None,
}
ADDRESS_TERMS_CRITIC_INSTRUCTION = (
    "address_terms are the project's mandatory Vietnamese address-term decisions (xưng hô). Any deviation in "
    "narration, or between the listed speaker and listener, is a pronoun_reference issue."
)
ADDRESS_TERMS_REWRITE_INSTRUCTION = (
    "Use ADDRESS_TERMS exactly for the reader, narrator and listed characters, and keep pronouns consistent "
    "with PREVIOUS_BILINGUAL_CONTEXT."
)
READER_TERM_CANDIDATES = ("bạn", "các bạn", "quý vị", "quý khách", "anh chị", "cậu", "mày")

_SOURCE_YOU = re.compile(r"\byou(?:r|rs|rself|rselves)?\b", re.IGNORECASE)
_SOURCE_I = re.compile(r"\b(?:I|me|my|mine|myself)\b")
_QUOTED = re.compile(r"“[^”]*”|\"[^\"]*\"|«[^»]*»")
_SPEECH_VERBS = "said|says|told|tells|asked|asks|replied|whispered|shouted|answered|smiled|looked|called"
_ENGLISH_PERSON = re.compile(r"\b(?:I|me|my|mine|myself|you|your|yours|yourself)\b", re.IGNORECASE)
_FIRST_PERSON = re.compile(r"I|me|my|mine|myself", re.IGNORECASE)
_VI_WORD = re.compile(r"\w+", re.UNICODE)
# Đại từ "trung tính" model hay quay về khi bỏ qua xưng hô đã khóa.
_GENERIC_PRONOUNS = {"tôi", "bạn", "tao", "mày"}
_THIRD_PERSON_SUFFIX = re.compile(r"\s+(?:ấy|ta|kia)\b", re.IGNORECASE)
_DIALOGUE_LINE = re.compile(r"^\s*[–—-]\s.*$", re.MULTILINE)


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _contains_word(text: str, term: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE))


def _narration_only(text: str) -> str:
    """Bỏ lời thoại để kiểm cách gọi người đọc; trong thoại nhân vật xưng hô tự do."""
    return _DIALOGUE_LINE.sub(" ", _QUOTED.sub(" ", text or ""))


@dataclass(frozen=True)
class AddressPair:
    speaker: str
    listener: str
    speaker_self: str
    listener_term: str
    note: str = ""

    def line(self) -> str:
        suffix = f" ({self.note})" if self.note else ""
        # Nói rõ thay cho I/you: đo thật thấy qwen2.5:7b hay đảo vai hoặc quay về "tôi/bạn".
        return (
            f'- Lời thoại {self.speaker} nói với {self.listener}{suffix}: "I/me" của {self.speaker} dịch là '
            f'"{self.speaker_self}", "you" (chỉ {self.listener}) dịch là "{self.listener_term}". '
            f'Không dùng "tôi/bạn" cho cặp này.'
        )


@dataclass(frozen=True)
class AddressTermLedger:
    reader: str = ""
    narrator: str = ""
    characters: Dict[str, str] = field(default_factory=dict)
    pairs: Tuple[AddressPair, ...] = ()

    @classmethod
    def from_style_guide(cls, style_guide: Optional[Dict[str, Any]], document_type: str = "GENERAL") -> "AddressTermLedger":
        raw = (style_guide or {}).get("address_terms")
        raw = raw if isinstance(raw, dict) else {}
        if "reader" in raw:
            reader = _clean(raw.get("reader"))
        else:
            reader = DEFAULT_READER_TERMS.get(str(document_type or "GENERAL").upper()) or ""
        characters = {
            _clean(name): _clean(term)
            for name, term in (raw.get("characters") or {}).items()
            if _clean(name) and _clean(term)
        } if isinstance(raw.get("characters"), dict) else {}
        pairs = []
        for item in raw.get("pairs") or []:
            if not isinstance(item, dict):
                continue
            pair = AddressPair(
                _clean(item.get("speaker")), _clean(item.get("listener")),
                _clean(item.get("self")), _clean(item.get("other")), _clean(item.get("note")),
            )
            if pair.speaker and pair.listener and pair.speaker_self and pair.listener_term:
                pairs.append(pair)
        return cls(reader, _clean(raw.get("narrator")), characters, tuple(pairs))

    @property
    def is_empty(self) -> bool:
        return not (self.reader or self.narrator or self.characters or self.pairs)

    def relevant(self, source_text: str) -> "AddressTermLedger":
        """Chỉ giữ mục liên quan tới đoạn nguồn để prompt không phình theo độ dài sách."""
        source = source_text or ""
        return AddressTermLedger(
            reader=self.reader if _SOURCE_YOU.search(source) else "",
            narrator=self.narrator if _SOURCE_I.search(source) else "",
            characters={name: term for name, term in self.characters.items() if TermMatcher.contains(source, name)},
            pairs=tuple(
                pair for pair in self.pairs
                if TermMatcher.contains(source, pair.speaker) or TermMatcher.contains(source, pair.listener)
            ),
        )

    def prompt_lines(self) -> List[str]:
        lines = []
        if self.reader:
            lines.append(f'- Người đọc ("you" của tác giả): gọi là "{self.reader}".')
        if self.narrator:
            lines.append(f'- Người kể chuyện ("I"): tự xưng "{self.narrator}".')
        for name, term in self.characters.items():
            # Không nói "dùng X" chung chung: model hiểu thành danh xưng gắn trước tên ("cô Alice").
            lines.append(f'- "he/she/him/her" chỉ {name} dịch là "{term}"; giữ tên {name} trơn, không thêm "{term}" trước tên.')
        lines.extend(pair.line() for pair in self.pairs)
        return lines

    def to_payload(self) -> Dict[str, Any]:
        return {
            "reader": self.reader,
            "narrator": self.narrator,
            "characters": dict(self.characters),
            "pairs": [
                {"speaker": p.speaker, "listener": p.listener, "self": p.speaker_self, "other": p.listener_term, "note": p.note}
                for p in self.pairs
            ],
        }

    def _attribution(self, source_text: str) -> Optional[AddressPair]:
        """Tìm cặp người nói -> người nghe từ câu dẫn tiếng Anh ("X said to Y", "X looked at Y and said").

        Chỉ trả về khi cả đoạn có đúng một cặp được nêu, để không đoán khi có nhiều người nói.
        """
        narration = _QUOTED.sub(" | ", source_text or "")
        found = set()
        for pair in self.pairs:
            pattern = (
                rf"\b{re.escape(pair.speaker)}\b[^|.]*?\b(?:{_SPEECH_VERBS})\b[^|.]*?\b{re.escape(pair.listener)}\b"
                rf"|\b{re.escape(pair.speaker)}\b[^|.]*?\b{re.escape(pair.listener)}\b[^|.]*?\b(?:{_SPEECH_VERBS})\b"
            )
            if re.search(pattern, narration, re.IGNORECASE):
                found.add(pair)
        return next(iter(found)) if len(found) == 1 else None

    def check_dialogue(self, source_text: str, translated_text: str) -> List[Dict[str, str]]:
        """Kiểm xưng hô trong lời thoại theo đúng vai người nói/người nghe.

        Chạy thật thấy qwen2.5:7b hay đảo vai (Bob nói "Em sẽ ở lại") hoặc quay về "tôi/bạn".
        So thứ tự I/you trong lời thoại nguồn với thứ tự đại từ của cặp trong bản dịch.
        """
        pair = self._attribution(source_text)
        if not pair:
            return []
        source_quotes = " ".join(_QUOTED.findall(source_text or ""))
        target_quotes = " ".join(_QUOTED.findall(translated_text or ""))
        if not source_quotes or not target_quotes:
            return []
        expected = [
            pair.speaker_self if _FIRST_PERSON.fullmatch(word) else pair.listener_term
            for word in _ENGLISH_PERSON.findall(source_quotes)
        ]
        pair_terms = {pair.speaker_self.casefold(), pair.listener_term.casefold()}
        actual = [word for word in _VI_WORD.findall(target_quotes.casefold()) if word in pair_terms]
        generic = sorted({
            word for word in _VI_WORD.findall(target_quotes.casefold())
            if word in _GENERIC_PRONOUNS and word not in pair_terms
        })
        expectation = (
            f'{pair.speaker} nói với {pair.listener}: "I" phải là "{pair.speaker_self}", '
            f'"you" phải là "{pair.listener_term}".'
        )
        if generic:
            return [{
                "code": "ADDRESS_TERM_MISMATCH", "severity": "ERROR",
                "message": f"Lời thoại dùng {', '.join(generic)} thay vì xưng hô đã khóa. {expectation}",
            }]
        if expected and len(actual) == len(expected) and actual != [term.casefold() for term in expected]:
            return [{
                "code": "ADDRESS_TERM_MISMATCH", "severity": "ERROR",
                "message": f"Xưng hô trong lời thoại bị đảo vai ({' '.join(actual)}). {expectation}",
            }]
        return []

    def fix_dialogue(self, source_text: str, translated_text: str) -> Optional[str]:
        """Sửa xưng hô lời thoại theo vị trí khi số đại từ khớp đúng chuỗi I/you của nguồn.

        Chạy thật: qua hai lần repair model vẫn ra "Em sẽ ở lại … Anh là lý do anh trở về" cho lời
        của Bob. Khi số đại từ (cặp + tôi/bạn) bằng số I/you trong nguồn thì thay từng vị trí theo
        đúng vai; đếm lệch (vd có "anh ấy" ngôi thứ ba) thì không đoán.
        """
        pair = self._attribution(source_text)
        if not pair:
            return None
        expected = [
            pair.speaker_self if _FIRST_PERSON.fullmatch(word) else pair.listener_term
            for word in _ENGLISH_PERSON.findall(" ".join(_QUOTED.findall(source_text or "")))
        ]
        replaceable = {pair.speaker_self.casefold(), pair.listener_term.casefold()} | _GENERIC_PRONOUNS
        positions = []
        for quote in _QUOTED.finditer(translated_text or ""):
            text = quote.group(0)
            for word in _VI_WORD.finditer(text):
                if word.group(0).casefold() not in replaceable:
                    continue
                if _THIRD_PERSON_SUFFIX.match(text, word.end()):
                    # "anh ấy/anh ta" là ngôi thứ ba, không phải I/you: giữ nguyên và để đếm lệch.
                    continue
                positions.append((quote.start() + word.start(), quote.start() + word.end(), word.group(0)))
        if not expected or len(positions) != len(expected):
            return None
        fixed = translated_text
        for (start, end, original), target in sorted(zip(positions, expected), key=lambda item: -item[0][0]):
            replacement = target[:1].upper() + target[1:] if original[:1].isupper() else target
            fixed = fixed[:start] + replacement + fixed[end:]
        return fixed if fixed != translated_text else None

    def check(self, source_text: str, translated_text: str) -> List[Dict[str, str]]:
        """Kiểm deterministic phần chắc chắn được: cách gọi người đọc trong lời dẫn.

        Cặp xưng hô giữa nhân vật cần biết ai đang nói, nên để critic đánh giá.
        """
        if not self.reader or not _SOURCE_YOU.search(_narration_only(source_text)):
            return []
        narration = _narration_only(translated_text)
        configured = self.reader.casefold()
        issues = []
        for candidate in READER_TERM_CANDIDATES:
            folded = candidate.casefold()
            # "bạn" và "các bạn" là số ít/số nhiều của cùng một lựa chọn.
            if folded in configured or configured in folded:
                continue
            if _contains_word(narration, candidate):
                issues.append({
                    "rule": "ADDRESS_READER_MISMATCH",
                    "type": "PRONOUN_REFERENCE",
                    "severity": "WARNING",
                    "target_span": candidate,
                    "message": f'Người đọc đã được quy ước gọi là "{self.reader}" nhưng bản dịch dùng "{candidate}".',
                    "suggestion": f'Đổi thành "{self.reader}".',
                })
        return issues

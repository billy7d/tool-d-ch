"""Tra nghĩa tham khảo cho từ tiếng Anh mà model để sót, từ một từ điển StarDict cục bộ.

qwen2.5:7b hay chép nguyên từ nguồn ("habit", "rhizome") và tự hỏi nghĩa thì trả lời sai ("microcosm" -> "nhỏ giọt"),
nhưng có nghĩa đúng trong prompt thì thay rất chuẩn. Từ điển nằm ngoài repo (data/dictionaries/en-vi); thiếu thì
không có gợi ý và pipeline vẫn chạy như cũ.
"""
import gzip
import re
import struct
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from app.config import settings

MAX_SENSES_PER_PART = 3
MAX_SENSES = 5
MAX_SENSE_CHARS = 60
MIN_FRAGMENT_CHARS = 4
# Nghĩa cổ, tiếng lóng hiếm khi là nghĩa cần trong sách; dòng còn sót phiên âm hoặc ghi chú dạng số nhiều thì không phải nghĩa.
_UNWANTED_SENSE = re.compile(
    r"^\((?:từ cổ|từ hiếm|nghĩa cổ|từ lóng|thơ ca)[^)]*\)|^số nhiều|^'|/$|[ɔəʃɑɪŋɳθðæʒʌ]", re.IGNORECASE,
)
# Nhãn lĩnh vực tiếng Anh do từ điển chuyên ngành gộp vào ("and" -> "(Tech) VÀ (mạch luận lý)"): xếp sau nghĩa chung.
_FIELD_TAG = re.compile(r"^\([A-Z][A-Za-z]*\)")
_NEGATION_PREFIX = re.compile(r"(?:không|vô|bất|phi)", re.IGNORECASE)
_ASCII_TAIL = re.compile(r"[A-Za-z]+$")
_CAPITAL_SPLIT = re.compile(r"[A-Z][a-z]+$")
_SOURCE_WORD = re.compile(r"[A-Za-z]+")
# Thử từ dạng dài tới ngắn: "chanting" -> "chant", "tripped" -> "trip", "chastised" -> "chastise".
_SUFFIX_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("ingly", ("ing", "e", "")), ("edly", ("ed", "e", "")), ("ively", ("ive",)), ("ally", ("al", "")), ("ily", ("y",)), ("ly", ("",)),
    ("iness", ("y",)), ("ness", ("",)), ("ies", ("y",)), ("ied", ("y",)), ("ing", ("e", "")), ("ed", ("e", "")), ("es", ("e", "")),
    ("er", ("e", "")), ("est", ("e", "")), ("s", ("",)),
)


class StarDict:
    """Bảng tra chữ thường -> (vị trí, độ dài); nội dung giải nén một lần khi cần tra lần đầu."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self._index: Optional[Dict[str, Tuple[int, int]]] = None
        self._data = b""
        self._lock = threading.Lock()

    def _files(self) -> Optional[Tuple[Path, Path]]:
        for idx in sorted(self.directory.glob("*.idx")):
            data = idx.with_name(idx.stem + ".dict.dz")
            if data.exists():
                return idx, data
        return None

    def _load(self) -> None:
        with self._lock:
            if self._index is not None:
                return
            index: Dict[str, Tuple[int, int]] = {}
            files = self._files()
            if files:
                raw = files[0].read_bytes()
                position = 0
                while position < len(raw):
                    end = raw.index(b"\0", position)
                    offset, size = struct.unpack(">II", raw[end + 1:end + 9])
                    index.setdefault(raw[position:end].decode("utf-8", "replace").lower(), (offset, size))
                    position = end + 9
                with gzip.open(files[1], "rb") as handle:
                    self._data = handle.read()
            self._index = index

    def entry(self, word: str) -> Optional[str]:
        self._load()
        location = (self._index or {}).get(word.lower())
        if not location:
            return None
        return self._data[location[0]:location[0] + location[1]].decode("utf-8", "replace")


def senses_from_entry(entry: str) -> List[str]:
    """Lấy vài nghĩa đầu của mỗi từ loại; bỏ ví dụ, thành ngữ, nghĩa cổ. Phần chuyên ngành chỉ dùng khi không có gì khác."""
    general: List[str] = []
    specialized: List[str] = []
    in_general = True
    per_part = 0
    in_idiom = False
    for line in (entry or "").splitlines():
        line = line.strip()
        if line.startswith("@"):
            if general or line.startswith("@Chuyên ngành") or line.startswith("@Lĩnh vực"):
                in_general = False
            continue
        if line.startswith("*"):
            per_part = 0
            in_idiom = False
            continue
        if line.startswith("!"):
            in_idiom = True
            continue
        if not line.startswith("-") or in_idiom:
            continue
        sense = line.lstrip("- ").strip()
        if not sense or _UNWANTED_SENSE.search(sense) or len(sense) > MAX_SENSE_CHARS:
            continue
        if not in_general or _FIELD_TAG.match(sense):
            specialized.append(sense)
        elif per_part < MAX_SENSES_PER_PART:
            general.append(sense)
            per_part += 1
    chosen = general or specialized
    return list(dict.fromkeys(chosen))[:MAX_SENSES]


def _base_forms(word: str) -> Iterable[str]:
    lower = word.lower()
    yield lower
    for suffix, replacements in _SUFFIX_RULES:
        if lower.endswith(suffix) and len(lower) - len(suffix) >= 2:
            stem = lower[: -len(suffix)]
            for replacement in replacements:
                yield stem + replacement
            # "tripped" -> "trip", "whipped" -> "whip"
            if len(stem) >= 3 and stem[-1] == stem[-2] and "" in replacements:
                yield stem[:-1]


def _source_word_for(residue: str, source: str) -> str:
    """Từ model viết ra có thể méo hoặc dính ("passively" cho "passive", "khôngscious" cho "unconscious"): lấy từ gốc trong nguồn."""
    match = _ASCII_TAIL.search(residue)
    tail = (match.group(0) if match else residue).lower()
    words = {word.lower() for word in _SOURCE_WORD.findall(source or "")}
    if tail in words:
        return tail
    # Chữ dính: phần cuối ASCII còn mang cả phụ âm của âm tiết Việt đứng trước ("ng"+"scious"), nên thử bỏ dần chữ đầu.
    negated = bool(_NEGATION_PREFIX.match(residue))
    for start in range(0, max(0, len(tail) - MIN_FRAGMENT_CHARS) + 1):
        fragment = tail[start:]
        containing = sorted((word for word in words if fragment in word), key=lambda word: (len(word), word))
        if not containing:
            continue
        ending = [word for word in containing if word.endswith(fragment)]
        # "khôngscious" khi nguồn có cả "conscious" lẫn "unconscious": chữ "không" đứng trước là phần dịch của "un-".
        if negated and ending:
            return ending[-1]
        return (ending or containing)[0]
    shared = [word for word in words if len(word) >= MIN_FRAGMENT_CHARS and word[:max(4, len(word) - 3)] == tail[:max(4, len(word) - 3)]]
    if shared:
        return sorted(shared, key=lambda word: (abs(len(word) - len(tail)), word))[0]
    capital = _CAPITAL_SPLIT.search(residue)
    return capital.group(0).lower() if capital else tail


class VocabularyHints:
    _instances: Dict[str, StarDict] = {}

    def __init__(self, dictionary: Optional[StarDict] = None):
        self.dictionary = dictionary if dictionary is not None else self._shared(settings.DICTIONARY_DIR)

    @classmethod
    def _shared(cls, directory: Path) -> StarDict:
        key = str(directory)
        if key not in cls._instances:
            cls._instances[key] = StarDict(directory)
        return cls._instances[key]

    def senses(self, word: str) -> List[str]:
        """Nghĩa của đúng từ đó và của từ gốc: "minutes" tra riêng chỉ ra "biên bản", phải có cả "phút" của "minute"."""
        found: List[str] = []
        entries_used = 0
        for form in dict.fromkeys(_base_forms(word)):
            entry = self.dictionary.entry(form)
            if not entry:
                continue
            senses = senses_from_entry(entry)
            if senses:
                found.extend(senses)
                entries_used += 1
            if entries_used == 2:
                break
        return list(dict.fromkeys(found))[:MAX_SENSES]

    def for_words(self, words: Iterable[str], source: str) -> Dict[str, List[str]]:
        hints: Dict[str, List[str]] = {}
        for word in words:
            base = _source_word_for(word, source)
            if not word.isascii() and base not in {item.lower() for item in _SOURCE_WORD.findall(source or "")}:
                # Chữ dính không ứng với từ nào trong nguồn là rác của model: "giườngPush" tra "push" thành "sự xô của giường".
                continue
            senses = self.senses(base)
            if senses:
                hints[word] = senses
        return hints


def format_hints(hints: Dict[str, List[str]]) -> str:
    return "\n".join(f"- {word}: {'; '.join(senses)}" for word, senses in hints.items())

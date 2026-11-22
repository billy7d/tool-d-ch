"""Dùng trang "Mục lục" in sẵn trong sách scan để dọn chính nó và sửa tiêu đề chương/mục.

Trang mục lục in bằng phông thường nên OCR đọc tốt, còn tiêu đề chương in bằng phông trang trí cỡ lớn hay bị
đọc vỡ và cắt thành nhiều mảnh ("Cuộc đời là một ván poker, không phủ" + "¡ mộồt ván cờ vud"). Mô-đun này:

- bỏ số trang cuối mỗi mục (vô nghĩa trong sách điện tử, OCR còn đọc thành "l4", "ó8", "b»)"), tách hai mục bị
  dính vào một dòng, nối mục bị xuống dòng;
- ghép các mảnh tiêu đề ở trang mở đầu chương rồi thay bằng tên chương trong mục lục;
- với tiêu đề mục, chọn từng chữ giữa bản trong mục lục và bản ở thân sách ("LỒI CẢM ƠN" / "LỜI CẢM 0N" ->
  "LỜI CẢM ƠN").

Hàm chỉ làm việc trên chữ (danh sách đơn vị của bộ phân tích bố cục truyền vào qua các hàm get/set).
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Sequence

from app.services.ebook.ocr_repair import core_of, is_garbled
from app.services.ebook.vi_spell import VietSyllableModel

# "CHƯƠNG 2", "(HƯƠNG 5", "| (HƯƠNG 4", "tHƯƠNG 3": OCR hay đọc chữ C đầu tiên thành "(" hoặc "t".
_CHAPTER_RE = re.compile(r"^[\W_]*(?:[C(t]\s*)?H\s*[ƯU]\s*Ơ\s*N\s*G\s+([0-9IlO]{1,2})\b\s*[|:.\-–—]?\s*", re.I)
_REF_MARKS = set("»«§£$¢")


def is_page_ref(token: str) -> bool:
    """Số trang cuối dòng mục lục, kể cả bị đọc sai: "45", "l4", "ó8", "b»)", "55)". Năm ("2008") thì không."""
    token = token.strip()
    if not token or len(token) > 4:
        return False
    if token.isdigit():
        return len(token) <= 3
    letters = sum(ch.isalpha() for ch in token)
    marked = any(ch.isdigit() or ch in _REF_MARKS for ch in token)
    return marked and letters <= 1


def _skeleton(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    return " ".join("".join(ch for ch in text if "a" <= ch <= "z" or ch.isdigit() or ch == " ").split())


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _skeleton(a), _skeleton(b), autojunk=False).ratio()


def chapter_number(text: str) -> Optional[int]:
    match = _CHAPTER_RE.match(text.strip())
    if not match:
        return None
    digits = match.group(1).replace("I", "1").replace("l", "1").replace("O", "0")
    return int(digits) if digits.isdigit() else None


def strip_page_ref(text: str) -> str:
    """Bỏ số trang cuối dòng mục lục."""
    tokens = text.split()
    if len(tokens) >= 2 and is_page_ref(tokens[-1]):
        tokens = tokens[:-1]
    return " ".join(tokens)


def has_page_ref(text: str) -> bool:
    tokens = text.split()
    return len(tokens) >= 2 and is_page_ref(tokens[-1])


def split_merged(text: str) -> List[str]:
    """Hai mục mục lục dính vào một dòng, ngăn cách bằng số trang ("... chính mình b») Sự đặt cược ...")."""
    tokens = text.split()
    parts: List[List[str]] = [[]]
    for index, token in enumerate(tokens):
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        if (
            parts[-1] and following and is_page_ref(token)
            and (following[:1].isupper() or following[:1] in "“\"(")
            and chapter_number(" ".join(parts[-1][-1:] + [token])) is None  # "CHƯƠNG 1 | ..." không phải số trang
        ):
            parts.append([])
            continue
        parts[-1].append(token)
    return [" ".join(p) for p in parts if p]


def clean_entry(text: str) -> str:
    text = " ".join(text.strip().lstrip("|").split())
    number = chapter_number(text)
    if number is not None:
        title = strip_page_ref(_CHAPTER_RE.sub("", text, count=1).strip(" |"))
        if title and is_page_ref(title):
            title = ""
        return f"CHƯƠNG {number} | {title}" if title else f"CHƯƠNG {number}"
    return strip_page_ref(text)


@dataclass
class TocEntry:
    label: str  # "CHƯƠNG 1", "DẪN NHẬP" hoặc "" (mục con)
    title: str  # tên chương/mục (không có số trang)
    chapter: Optional[int] = None

    @property
    def full(self) -> str:
        return f"{self.label} | {self.title}" if self.label and self.title else (self.label or self.title)


def parse_entry(text: str) -> TocEntry:
    text = clean_entry(text)
    number = chapter_number(text)
    if " | " in text:
        label, title = text.split(" | ", 1)
        return TocEntry(label.strip(), title.strip(), number)
    if number is not None:
        return TocEntry(f"CHƯƠNG {number}", "", number)
    return TocEntry("", text)


def _known(word: str, model: VietSyllableModel) -> bool:
    core = core_of(word).lower()
    if not core:
        return True
    if is_garbled(word):
        return False
    parts = [p for p in re.split(r"[-–/]", core) if p]
    return all(model.unigrams.get(p, 0) >= 3 or not p.isalpha() for p in parts)


def merge_readings(primary: str, other: str, model: VietSyllableModel) -> str:
    """Hai cách đọc của cùng một tiêu đề (mục lục và thân sách): chọn từng chữ.

    Chữ có thật thắng chữ vỡ/lạ; cả hai đều có thật thì chọn chữ hợp với chữ đứng trước hơn (theo từ điển).
    Khác số chữ thì giữ bản chính (mục lục).
    """
    a, b = primary.split(), other.split()
    if len(a) != len(b):
        return primary
    out: List[str] = []
    for index, (x, y) in enumerate(zip(a, b)):
        if x == y:
            out.append(x)
            continue
        kx, ky = _known(x, model), _known(y, model)
        if kx != ky:
            out.append(x if kx else y)
            continue
        prev = core_of(out[-1]).lower() if out else None
        nxt = core_of(a[index + 1]).lower() if index + 1 < len(a) else None

        def support(word: str) -> int:
            low = core_of(word).lower()
            return (model.bigrams.get((prev, low), 0) if prev else 0) + (model.bigrams.get((low, nxt), 0) if nxt else 0)

        sx, sy = support(x), support(y)
        if sx == sy:  # không cặp nào có trong từ điển: chọn chữ phổ biến hơn ("Lối" hơn "Lõi")
            sx = model.unigrams.get(core_of(x).lower(), 0)
            sy = model.unigrams.get(core_of(y).lower(), 0)
        out.append(y if sy > sx else x)
    return " ".join(out)


def best_entry(text: str, entries: Sequence[TocEntry], threshold: float = 0.75) -> Optional[TocEntry]:
    best, best_score = None, threshold
    for entry in entries:
        for candidate in {entry.full, entry.title}:
            if not candidate:
                continue
            score = similarity(text, candidate)
            if score > best_score:
                best, best_score = entry, score
    return best

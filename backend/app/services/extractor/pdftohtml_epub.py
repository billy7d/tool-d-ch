"""Dựng lại cấu trúc sách cho EPUB do Calibre chuyển từ PDF (pdftohtml).

Loại EPUB này rất hay gặp ở sách tải về: không có thẻ tiêu đề, mỗi đoạn in ra thành một hoặc nhiều <p>. Đoạn văn
bị cắt ở chỗ sang trang và ở mọi dòng mở bằng ngoặc kép hoặc gạch dài ("...nut in my yard" | "“knows” about the
future..."), số chú thích dính vào chữ ("it does.2"), phần Notes là các dòng rời. Nhập theo cách đọc EPUB thường thì
"Stumbling on Happiness" ra 2 chương, 1 tiêu đề và 1.695 mảnh đoạn, model phải dịch từng nửa câu.

Ở đây đọc cả sách thành chuỗi dòng (kèm số trang từ mốc <a id="pN">, cờ nghiêng/đậm/liên kết), rồi:
- bỏ trang "Contents" in sẵn (các dòng là liên kết nội bộ) và ảnh bìa trước dòng chữ đầu tiên khi EPUB có bìa riêng;
- nhận tiêu đề: nhãn "PART I"/"CHAPTER 3" kèm dòng tên ngay sau, dòng viết hoa trùng một mục của mục lục
  ("ACKNOWLEDGMENTS", "NOTES"), dòng nghiêng/đậm ngắn đứng riêng giữa hai đoạn (tiêu đề mục);
- nối các mảnh của cùng một đoạn, kể cả khi ảnh hoặc chú thích hình chen giữa (ảnh được dời ra sau đoạn);
- phần Notes: mỗi mục "12. ..." là một chú thích, dòng lẻ nối vào mục đang mở, "Chapter 1: ..." là tiêu đề mục;
- số chú thích trong thân bài thành " [n]" khi đúng thứ tự 1, 2, 3... của chương (năm, số thập phân không bị đụng);
- chữ in hoa mở chương ("PRIESTS VOW TO REMAIN CELIBATE, physicians...") về chữ thường, giữ tên riêng;
- đề từ ngắn đứng sau tiêu đề chương được đánh dấu là trích dẫn.

Thà để sót một chỗ ngắt còn hơn dính hai đoạn vốn tách rời: hai mảnh chỉ nối khi mảnh trước chưa kết câu.
"""
from __future__ import annotations

import posixpath
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from bs4 import BeautifulSoup, Tag

from app.services.reconstruction.page_break_joiner import join_text

_INVISIBLE = re.compile("[" + "".join(map(chr, (0xFEFF, 0x2060, 0x200B, 0x00AD))) + "]")
_PAGE_ANCHOR = re.compile(r"^p(\d+)$")
_NUMBER_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    "seventeen|eighteen|nineteen|twenty"
)
_LABEL = re.compile(rf"^(part|chapter|book|section|appendix)\s+([ivxlcdm]+|\d{{1,3}}|{_NUMBER_WORDS})\.?$", re.I)
_CAPTION = re.compile(r"^(?:fig(?:ure)?\.?|table|chart|plate|map)\s*\d+", re.I)
# Kết câu: dấu câu cuối, có thể theo sau bởi ngoặc đóng và số chú thích ("does.2", "———”).3"). Dấu ";" chưa kết câu.
_TERMINAL = re.compile(r"[.!?…:][\"'’”)\]*†]*(\d{0,3})$")
_TRAILING_INITIAL = re.compile(r"(?:^|[\s(])[A-Z]\.$")
_SURNAME_LEAD = re.compile(r"[A-Z][a-z’'-]+,\s")
_SPACED_ELLIPSIS_TAIL = re.compile(r"\.\s\.(?:\s\.)?(?![\w.])")
_TITLE_ABBREVIATION = re.compile(r"(?:^|[\s(“\"‘])(?:Mr|Mrs|Ms|Dr|Prof|vs|cf|e\.g|i\.e)\.$")
_NOTE_ENTRY = re.compile(r"^(\d{1,3})\.\s+\S")
_NOTES_TITLES = {"notes", "endnotes", "end notes", "notes and sources", "source notes"}
_NOTES_SUBHEADING = re.compile(r"^(?:chapter|part|section)\s+(?:\d{1,3}|[ivxlcdm]+|" + _NUMBER_WORDS + r")\b", re.I)
_OUTLINE_TITLE = "document outline"
# Số chú thích dính sau chữ hoặc dấu câu, đứng trước khoảng trắng/dấu đóng hoặc cuối đoạn.
_GLUED_REF = re.compile(r"(?<=[A-Za-z.,;:!?”’\"')\]—–])(\d{1,3})(?=$|[\s”’\"')\],;:—–])")
# Số chú thích cách một khoảng sau dấu kết câu ("pleasure. 1 The first part"): chỉ nhận khi đúng số đang chờ.
_SPACED_REF = re.compile(r"(?<=[.!?”’])\s(\d{1,3})(?=\s+[A-Z“\"‘])")
_WORD = re.compile(r"[A-Za-z][A-Za-z’'\-]*")
_DECIMAL_REF_TAIL = re.compile(r"\s+[A-Z“\"‘]")


def ends_sentence(text: str) -> bool:
    """Mảnh kết thúc bằng dấu kết câu. Số sau dấu chấm chỉ là số chú thích khi dấu chấm dính vào chữ: "the .03" | "percent"
    là một câu đang dở."""
    text = (text or "").rstrip()
    if _TITLE_ABBREVIATION.search(text):
        return False  # danh xưng viết tắt cuối trang: "(Ms." | "Dash’s)"
    match = _TERMINAL.search(text)
    if not match:
        return False
    if match.group(1):
        before = text[match.start() - 1] if match.start() else " "
        return not (before.isspace() or before.isdigit())
    return True


def _clean(text: str) -> str:
    return " ".join(_INVISIBLE.sub("", text or "").split())


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").casefold()).split())


@dataclass
class Line:
    kind: str  # text | image
    text: str = ""
    page: int = 0
    italic: bool = False
    bold: bool = False
    linked: bool = False
    src: str = ""


@dataclass
class Block:
    kind: str  # heading | text | caption | note | image
    text: str = ""
    level: int = 0
    page: int = 0
    src: str = ""
    italic: bool = False
    bold: bool = False
    quote: bool = False
    note_number: str = ""
    extra: dict = field(default_factory=dict)


def is_pdftohtml_document(raw: bytes) -> bool:
    head = (raw or b"")[:4000].lower()
    return b"pdftohtml" in head and b"generator" in head


# ---- Đọc dòng ---------------------------------------------------------------------------------------------------


def _flags(paragraph: Tag) -> Tuple[bool, bool, bool]:
    italic = bold = linked = True
    seen = False
    for piece in paragraph.find_all(string=True):
        if not piece.strip():
            continue
        seen = True
        names, has_link = set(), False
        for parent in piece.parents:
            if parent is paragraph or parent is None:
                break
            names.add(parent.name)
            if parent.name == "a" and parent.get("href"):
                has_link = True
        italic &= bool(names & {"i", "em"})
        bold &= bool(names & {"b", "strong"})
        linked &= has_link
    return (italic and seen, bold and seen, linked and seen)


def _resolve(doc_name: str, src: str) -> str:
    src = (src or "").split("#")[0]
    return posixpath.normpath(posixpath.join(posixpath.dirname(doc_name or ""), src)) if src else ""


def read_lines(documents: Sequence[Tuple[str, BeautifulSoup]]) -> List[Line]:
    lines: List[Line] = []
    page = 0
    for doc_name, soup in documents:
        body = soup.body or soup
        heading = body.find(["h1", "h2"])
        if heading is not None and _norm(heading.get_text()) == _OUTLINE_TITLE:
            continue  # trang "Document Outline" Calibre thêm vào cuối: trùng với mục lục điều hướng
        for paragraph in body.find_all("p"):
            if paragraph.find_parent("p") is not None:
                continue
            for anchor in paragraph.find_all("a", id=True):
                match = _PAGE_ANCHOR.match(str(anchor.get("id")))
                if match:
                    page = int(match.group(1))
                    break
            for image in paragraph.find_all("img"):
                lines.append(Line("image", page=page, src=_resolve(doc_name, image.get("src"))))
            text = _clean(paragraph.get_text(""))
            if text:
                italic, bold, linked = _flags(paragraph)
                lines.append(Line("text", text, page, italic, bold, linked))
    return lines


def _drop_cover_images(lines: List[Line]) -> List[Line]:
    first_text = next((index for index, line in enumerate(lines) if line.kind == "text"), len(lines))
    return [line for index, line in enumerate(lines) if index >= first_text or line.kind != "image"]


def _drop_printed_contents(lines: List[Line]) -> Tuple[List[Line], List[str]]:
    """Bỏ trang mục lục in sẵn (chuỗi dòng toàn liên kết nội bộ) và dòng "CONTENTS" ngay trước nó."""
    keep = [True] * len(lines)
    titles: List[str] = []
    index = 0
    while index < len(lines):
        if not (lines[index].kind == "text" and lines[index].linked):
            index += 1
            continue
        end = index
        while end < len(lines) and lines[end].kind == "text" and lines[end].linked:
            end += 1
        if end - index >= 3:
            for position in range(index, end):
                keep[position] = False
                titles.append(lines[position].text)
            before = index - 1
            if before >= 0 and lines[before].kind == "text" and _norm(lines[before].text) in {"contents", "table of contents"}:
                keep[before] = False
        index = end
    return [line for line, kept in zip(lines, keep) if kept], titles


# ---- Phân loại ----------------------------------------------------------------------------------------------------


def _uppercase(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    return len(letters) >= 3 and all(char.isupper() for char in letters)


def _lead_in_count(text: str) -> int:
    """Số từ in hoa liền nhau ở đầu dòng ("PRIESTS VOW TO REMAIN CELIBATE, physicians" -> 5)."""
    count = 0
    for word in text.split():
        letters = [char for char in word if char.isalpha()]
        if not letters or not all(char.isupper() for char in letters):
            break
        if len(letters) >= 2 or word in ("A", "I"):
            count += 1
    return count


def _is_title_line(line: Line) -> bool:
    text = line.text
    return (
        line.kind == "text" and len(text) <= 90 and len(text.split()) <= 12
        and (line.italic or line.bold or _uppercase(text)) and not text.endswith((".", ",", ";", ":"))
        and not _LABEL.match(text)
    )


def _closed(block: Optional[Block]) -> bool:
    if block is None or block.kind in ("heading", "image", "caption", "note"):
        return True
    return ends_sentence(block.text)


_MINOR_WORDS = {
    "a", "an", "the", "and", "or", "nor", "but", "of", "in", "on", "at", "to", "for", "with", "from", "by", "as",
    "into", "onto", "over", "than", "via", "vs", "per", "up", "out", "off",
}


def _title_case(text: str) -> bool:
    words = re.findall(r"[A-Za-z][\w’'\-]*", text)
    return bool(words) and words[0][0].isupper() and all(
        word[0].isupper() or word.lower() in _MINOR_WORDS for word in words
    )


def _is_section_heading(line: Line, previous: Optional[Block], following: Optional[Line]) -> bool:
    text = line.text
    if len(text) > 80 or len(text.split()) > 10:
        return False
    if not (line.italic or line.bold):
        # pdftohtml có khi mất kiểu chữ của tiêu đề mục ("Prospection and Emotion" in thường): chỉ nhận dòng viết hoa
        # đầu mỗi từ, không có dấu phẩy (dòng ký tên "Shakespeare, Hamlet"), đứng trước một đoạn dài.
        if _uppercase(text) or "," in text or not _title_case(text):
            return False
        if following is None or following.kind != "text" or len(following.text) < 60 or _lead_in_count(following.text) >= 2:
            return False
    if not (text[0].isupper() or text[0].isdigit()) or text.endswith((".", ",", ";", ":", "—", "-")):
        return False
    if previous is None or previous.kind == "heading" or not _closed(previous):
        return False
    if following is None or following.kind != "text":
        return False
    first = following.text[0]
    return first.isupper() or first in "“\"‘"


def classify(lines: Sequence[Line], titles: Iterable[str]) -> Tuple[List[Block], int]:
    """Trả về các khối theo thứ tự đọc và cấp của tiêu đề chương (1 khi sách không chia phần, 2 khi có PART)."""
    title_set = {_norm(title) for title in titles if title.strip()}
    title_set |= {_norm(re.sub(r"^\s*(?:\d{1,3}\.|part\s+[ivxlcdm\d]+)\s*", "", title, flags=re.I)) for title in titles}
    has_parts = any(line.kind == "text" and _LABEL.match(line.text) and line.text.split()[0].lower() == "part" for line in lines)
    chapter_level = 2 if has_parts else 1
    blocks: List[Block] = []
    heading_names: Set[str] = set()
    in_notes = False
    index = 0
    while index < len(lines):
        line = lines[index]
        following = lines[index + 1] if index + 1 < len(lines) else None
        index += 1
        if line.kind == "image":
            blocks.append(Block("image", page=line.page, src=line.src))
            continue
        text = line.text
        label = _LABEL.match(text)
        if label and len(text.split()) <= 3:
            level = 1 if label.group(1).lower() == "part" else chapter_level
            in_notes = False
            blocks.append(Block("heading", text, level, line.page))
            if following is not None and _is_title_line(following):
                title = following.text
                index += 1
                # Tên chương xuống dòng ("In the Blind Spot" | "of the Mind’s Eye"): dòng cùng kiểu chữ, mở bằng chữ thường.
                while (
                    index < len(lines) and _is_title_line(lines[index]) and lines[index].text[0].islower()
                    and (lines[index].italic, lines[index].bold) == (following.italic, following.bold)
                ):
                    title = f"{title} {lines[index].text}"
                    index += 1
                blocks.append(Block("heading", title, level, following.page))
                heading_names.add(_norm(title))
            continue
        if (
            len(text.split()) <= 8 and _norm(text) in title_set
            and (_uppercase(text) or line.bold or line.italic) and not (in_notes and _NOTE_ENTRY.match(text))
            and not (in_notes and line.italic)
        ):
            in_notes = _norm(text) in _NOTES_TITLES
            blocks.append(Block("heading", text, 1, line.page))
            heading_names.add(_norm(text))
            continue
        if in_notes:
            if (line.italic or line.bold) and len(text.split()) <= 14 and (
                _NOTES_SUBHEADING.match(text) or _norm(text) in heading_names or _norm(text) in title_set
            ):
                # Dưới cấp chương để phần Notes là một chương, các mục "Chapter 1: ..." là tiêu đề con của nó.
                blocks.append(Block("heading", text, chapter_level + 1, line.page))
                continue
            entry = _NOTE_ENTRY.match(text)
            if entry:
                blocks.append(Block("note", text, page=line.page, note_number=entry.group(1)))
            elif blocks and blocks[-1].kind == "note":
                blocks[-1].text = join_text(blocks[-1].text, text)
            else:
                blocks.append(Block("text", text, page=line.page))
            continue
        previous = blocks[-1] if blocks else None
        if _is_section_heading(line, previous, following):
            blocks.append(Block("heading", text, chapter_level + 1, line.page))
            heading_names.add(_norm(text))
            continue
        kind = "caption" if _CAPTION.match(text) else "text"
        blocks.append(Block(kind, text, page=line.page, italic=line.italic, bold=line.bold))
    return _split_merged_notes(blocks), chapter_level


def _split_merged_notes(blocks: List[Block]) -> List[Block]:
    """Hai mục chú thích in liền trên một dòng ("...(1990.) 22. G. W. McConkie..."): tách khi gặp đúng số kế tiếp."""
    out: List[Block] = []
    for block in blocks:
        while block.kind == "note" and block.note_number.isdigit():
            following = str(int(block.note_number) + 1)
            match = re.search(rf"(?<=[.)”])\s+({following})\.\s+(?=[A-Z])", block.text)
            if not match:
                break
            out.append(Block("note", block.text[:match.start()].rstrip(), page=block.page, note_number=block.note_number))
            block = Block("note", block.text[match.start(1):], page=block.page, note_number=following)
        out.append(block)
    return out


# ---- Nối mảnh ----------------------------------------------------------------------------------------------------


def continues(previous: str, following: str) -> bool:
    """Mảnh sau có phải phần tiếp của đoạn đang dở không (mảnh trước chưa kết câu)."""
    previous, following = previous.rstrip(), following.lstrip()
    if not previous or not following:
        return False
    if following[0].islower():
        return True  # đoạn văn không mở bằng chữ thường: "...“Yuck!”" | "if they were full) and..."
    if len(following) <= 3 and not any(char.isalnum() for char in following):
        return True  # dấu chấm lửng bị ngắt dòng: "...found out.” . ." | "."
    if _TITLE_ABBREVIATION.search(previous):
        return True  # tên riêng sau danh xưng sang dòng/trang mới: "as Dr." | "Gilbert noted"
    if _TRAILING_INITIAL.search(previous) and _SURNAME_LEAD.match(following):
        return True  # tên người bị ngắt sau chữ viết tắt: "in R. Chester and J." | "Peel, eds., Equalities"
    if _SPACED_ELLIPSIS_TAIL.match(following):
        return True  # dấu chấm lửng cách quãng bị ngắt trang: "behind him ." | ". .”) without also"
    if ends_sentence(previous):
        return False
    if _lead_in_count(following) >= 2:
        return False  # đoạn mở chương ("WHAT WOULD YOU DO right now") hoặc dòng viết hoa riêng
    first = following[0]
    if first.islower() or first.isdigit() or first in "—–-(…“\"‘'’”":
        return True
    # Chữ hoa đầu dòng: dòng trước đủ dài mới là thân bài (dòng thơ, đề từ, trang tên sách thì ngắn).
    return len(previous) >= 40 or previous[-1] in ",;—–-"


def _caption_continues(caption: str, following: str) -> bool:
    """Dòng sau chú thích hình chỉ là phần tiếp khi chắc chắn câu đang dở (thân bài ngay sau hình thì không gộp).

    Trang nguồn hình minh họa: "Figure 14 reprinted from C. Walker, ... in R. Chester and J." | "Peel, eds., ...".
    """
    caption, following = caption.rstrip(), following.lstrip()
    if not caption or not following or not continues(caption, following):
        return False
    return (
        following[0].islower() or following[0] in "(—–-"
        or bool(_TRAILING_INITIAL.search(caption) and _SURNAME_LEAD.match(following))
        or bool(_TITLE_ABBREVIATION.search(caption))
    )


def join_fragments(blocks: Sequence[Block]) -> List[Block]:
    out: List[Block] = []
    open_index: Optional[int] = None
    caption_index: Optional[int] = None
    floating: List[Block] = []
    for block in blocks:
        if block.kind == "text":
            if open_index is not None and continues(out[open_index].text, block.text):
                out[open_index].text = join_text(out[open_index].text, block.text)
                continue
            if caption_index is not None and _caption_continues(out[caption_index].text, block.text):
                out[caption_index].text = join_text(out[caption_index].text, block.text)
                continue
            out.extend(floating)
            floating = []
            out.append(block)
            open_index = len(out) - 1
            caption_index = None
        elif block.kind in ("image", "caption"):
            if open_index is not None and not ends_sentence(out[open_index].text):
                floating.append(block)  # ảnh/chú thích hình chen giữa đoạn: dời ra sau đoạn
                caption_index = None
            else:
                out.extend(floating)
                floating = []
                out.append(block)
                open_index = None
                caption_index = len(out) - 1 if block.kind == "caption" else None
        else:
            out.extend(floating)
            floating = []
            out.append(block)
            open_index = None
            caption_index = None
    out.extend(floating)
    return out


# ---- Hậu xử lý -----------------------------------------------------------------------------------------------------


def mark_epigraphs(blocks: List[Block]) -> None:
    """Đề từ ngắn ngay sau tiêu đề, trước đoạn mở chương viết hoa hoặc tiêu đề kế tiếp: đánh dấu là trích dẫn."""
    index = 0
    while index < len(blocks):
        if blocks[index].kind != "heading":
            index += 1
            continue
        while index < len(blocks) and blocks[index].kind == "heading":
            index += 1
        run = []
        position = index
        while position < len(blocks) and len(run) < 6 and blocks[position].kind == "text":
            block = blocks[position]
            if _lead_in_count(block.text) >= 2 or len(block.text) > 300:
                break
            run.append(block)
            position += 1
        ends_section = position < len(blocks) and (
            blocks[position].kind == "heading"
            or (blocks[position].kind == "text" and _lead_in_count(blocks[position].text) >= 2)
        )
        if run and ends_section:
            for block in run:
                block.quote = True
        index = max(position, index + 1)


def _ref_guard(text: str, start: int) -> bool:
    """Không nhận số sau dấu chấm/phẩy của một số khác (3.5, 1,000) hoặc dính vào mã chữ-số (CO2, B12)."""
    before = text[start - 1] if start else ""
    if before in ".,":
        return not (start >= 2 and text[start - 2].isdigit())
    if before.isalpha():
        word = re.search(r"[A-Za-z]+$", text[:start]).group(0)
        return len(word) >= 3 and not word.isupper()
    return True


def convert_note_refs(blocks: List[Block], chapter_level: int) -> int:
    """Số chú thích dính vào chữ thành " [n]", chỉ khi đúng thứ tự trong chương. Trả về số chỗ đã đổi."""
    if not any(block.kind == "note" for block in blocks):
        return 0
    converted = 0
    expected = 1
    in_notes = False
    for block in blocks:
        if block.kind == "heading":
            if block.level <= chapter_level:
                expected = 1
                in_notes = _norm(block.text) in _NOTES_TITLES or (in_notes and block.level > 1)
            continue
        if in_notes or block.kind not in ("text", "caption"):
            continue
        text = block.text
        pieces: List[str] = []
        cursor = 0
        candidates = sorted(
            [(match.start(1), match.end(1), int(match.group(1)), False) for match in _GLUED_REF.finditer(text)]
            + [(match.start(1), match.end(1), int(match.group(1)), True) for match in _SPACED_REF.finditer(text)]
        )
        for start, end, number, spaced in candidates:
            if start < cursor:
                continue
            allowed = number == expected if spaced else expected <= number <= expected + 1
            if not allowed:
                continue
            if not spaced and not _ref_guard(text, start):
                # Số chú thích dính sau một con số trông như số thập phân ("figure 9.20 Some", "$300.23 Because",
                # "September 10.4 The"): chỉ nhận khi đúng số đang chờ và câu sau mở bằng chữ hoa.
                if not (number == expected and _DECIMAL_REF_TAIL.match(text, end)):
                    continue
            head = text[cursor:start].rstrip() if spaced else text[cursor:start]
            pieces.append(head + f" [{number}]")
            cursor = end
            expected = number + 1
            converted += 1
        if pieces:
            block.text = "".join(pieces) + text[cursor:]
    return converted


def proper_nouns(blocks: Sequence[Block]) -> Dict[str, str]:
    """Từ viết hoa giữa câu nhiều hơn viết thường: tên riêng, dùng khi trả chữ in hoa đầu chương về chữ thường."""
    capital, lower = Counter(), Counter()
    for block in blocks:
        if block.kind not in ("text", "note", "caption"):
            continue
        words = _WORD.findall(block.text)
        for previous, word in zip(words, words[1:]):
            if word[0].isupper() and word[1:].islower() and previous[-1:].islower():
                capital[word.lower()] += 1
            elif word.islower():
                lower[word] += 1
    return {word: word.capitalize() for word, count in capital.items() if count > lower[word]}


def normalize_lead_ins(blocks: Sequence[Block], names: Dict[str, str]) -> int:
    changed = 0
    for block in blocks:
        if block.kind != "text":
            continue
        count = _lead_in_count(block.text)
        words = block.text.split(" ")
        if count < 2 or count >= len(words) or not any(char.islower() for char in words[count]):
            continue
        recased = []
        for position, word in enumerate(words[:count]):
            core = re.sub(r"[^A-Za-z’'\-]", "", word)
            lowered = word.lower()
            if core.lower() in names:
                lowered = lowered.replace(core.lower(), names[core.lower()])
            elif core in ("I",) or "." in word:
                lowered = word
            if position == 0:
                lowered = re.sub(r"[^\W\d_]", lambda match: match.group(0).upper(), lowered, count=1)
            recased.append(lowered)
        block.text = " ".join(recased + words[count:])
        changed += 1
    return changed


# ---- Xuất khối cho StructureBuilder -----------------------------------------------------------------------------


def to_pages(blocks: Sequence[Block], chapter_level: int, image_ids: Dict[str, str]) -> List[dict]:
    """Mỗi chương/phần là một "trang" để StructureBuilder mở chương mới ở tiêu đề đầu tiên của trang."""
    pages: List[dict] = []
    current: List[dict] = []

    def flush() -> None:
        nonlocal current
        if current:
            pages.append({
                "page_number": len(pages) + 1, "page_type": "TEXT_PAGE", "title": "", "blocks": current,
                "word_count": sum(len(block.get("text", "").split()) for block in current),
            })
        current = []

    previous: Optional[Block] = None
    for block in blocks:
        if block.kind == "heading" and block.level <= chapter_level and (previous is None or previous.kind != "heading"):
            flush()
        entry = {"block_id": len(current), "confidence": 1.0, "structured": True, "source_page": block.page}
        if block.kind == "heading":
            level = block.level
            entry.update(type="heading", text=block.text, heading_level=level, font_size=18.0 - level * 2, is_bold=True)
        elif block.kind == "image":
            asset_id = image_ids.get(block.src)
            if not asset_id:
                previous = block
                continue
            entry.update(type="image", asset_id=asset_id)
        else:
            entry.update(type="text", text=block.text, font_size=11.0, is_bold=False)
            if block.kind == "note":
                entry.update(footnote=True, footnote_number=block.note_number)
            elif block.quote:
                entry["layout"] = {"quote": True}
            elif block.kind == "caption":
                entry["layout"] = {"align": "center"}
        current.append(entry)
        previous = block
    flush()
    return pages


def reconstruct(
    documents: Sequence[Tuple[str, BeautifulSoup]],
    image_ids: Dict[str, str],
    outline_titles: Sequence[str] = (),
    has_cover: bool = False,
) -> Tuple[List[dict], dict]:
    """Trang (khối) cho StructureBuilder và số liệu để kiểm tra."""
    lines = read_lines(documents)
    if has_cover:
        lines = _drop_cover_images(lines)
    lines, printed_titles = _drop_printed_contents(lines)
    blocks, chapter_level = classify(lines, list(printed_titles) + list(outline_titles))
    fragments = sum(block.kind == "text" for block in blocks)
    blocks = join_fragments(blocks)
    mark_epigraphs(blocks)
    refs = convert_note_refs(blocks, chapter_level)
    lead_ins = normalize_lead_ins(blocks, proper_nouns(blocks))
    stats = {
        "lines": len(lines), "fragments": fragments,
        "paragraphs": sum(block.kind == "text" for block in blocks),
        "headings": sum(block.kind == "heading" for block in blocks),
        "notes": sum(block.kind == "note" for block in blocks),
        "note_refs": refs, "lead_ins": lead_ins,
        "images": sum(block.kind == "image" for block in blocks),
    }
    return to_pages(blocks, chapter_level, image_ids), stats


__all__ = ["is_pdftohtml_document", "reconstruct", "read_lines", "classify", "join_fragments", "continues",
           "convert_note_refs", "normalize_lead_ins", "proper_nouns", "mark_epigraphs", "Line", "Block"]

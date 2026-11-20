"""Làm sạch chữ lấy từ PDF trước khi đưa vào sách điện tử, và đánh dấu kiểu chữ trong đoạn.

Kiểu chữ một phần đoạn (nghiêng, đậm, chỉ số trên) đi qua bộ dựng cấu trúc dưới dạng ký tự đánh dấu
MARK_OPEN/MARK_CLOSE + mã kiểu, vì nội dung node là chữ thuần. Bộ ghi EPUB/MOBI đổi lại thành thẻ.
"""
import html
import re
import unicodedata
from typing import Iterable

MARK_OPEN = "\x01"
MARK_CLOSE = "\x02"
STYLE_TAGS = {"b": "b", "i": "i", "s": "sup"}
STYLE_ORDER = ("b", "i", "s")

_MARKER_RE = re.compile("[\x01\x02][bis]")
_TOKEN_RE = re.compile("([\x01\x02][bis]|\n)")

_LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
    "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st",
}
_LIGATURE_RE = re.compile("[\ufb00-\ufb06]")
# Ký tự điều khiển, ký tự vô hình và vùng ký tự riêng (PUA): máy đọc hiện thành ô vuông hoặc chép lung tung.
# Giữ \x01/\x02 (đánh dấu kiểu chữ) và \n.
_INVISIBLE_RE = re.compile("[\x00\x03-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200e\u200f\u2060\ufeff\ue000-\uf8ff\ufffe\uffff]")
# Chữ thô từ PDF: bỏ cả \x01/\x02 có sẵn trong file để không bị hiểu nhầm là dấu kiểu chữ.
_RAW_CONTROL_RE = re.compile("[\x00-\x08]")
_SPACES_RE = re.compile("[ \t\u2000-\u200a\u202f\u205f\u3000]+")
# Ký tự XML 1.0 không cho phép (sau khi đã đổi dấu kiểu chữ thành thẻ).
_XML_INVALID_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]")

# Chữ cái chỉ tiếng Việt mới dùng: dấu hỏi (móc trên), dấu nặng, dấu móc (ơ ư) và chữ đ.
_VI_MARKS = {"̉", "̣", "̛"}


def clean_span_text(text: str) -> str:
    """Sửa chữ của một mảnh (span) PDF. Giữ gạch nối mềm để còn nhận ra chỗ ngắt từ cuối dòng."""
    if not text:
        return ""
    text = _LIGATURE_RE.sub(lambda m: _LIGATURES[m.group()], text)
    text = _RAW_CONTROL_RE.sub("", _INVISIBLE_RE.sub("", text))
    return _SPACES_RE.sub(" ", text)


def finalize_text(text: str) -> str:
    """Chuẩn hóa cuối cùng cho một đoạn: bỏ gạch nối mềm, dựng sẵn dấu tiếng Việt (NFC), gom khoảng trắng."""
    text = text.replace("\u00ad", "")
    text = unicodedata.normalize("NFC", text)
    text = _SPACES_RE.sub(" ", text)
    return text.strip()


def strip_markers(text: str) -> str:
    return _MARKER_RE.sub("", text or "")


def xml_safe(text: str) -> str:
    return _XML_INVALID_RE.sub("", text or "")


def inline_markup(text: str) -> str:
    """Chữ (có thể chứa dấu kiểu chữ) -> XHTML nội dòng hợp lệ: chỉ <b>, <i>, <sup>, <br/>, luôn đóng đủ thẻ."""
    out = []
    stack = []
    for token in _TOKEN_RE.split(text or ""):
        if not token:
            continue
        if token == "\n":
            out.append("<br/>")
        elif token[0] == MARK_OPEN:
            tag = STYLE_TAGS[token[1]]
            if tag not in stack:
                stack.append(tag)
                out.append(f"<{tag}>")
        elif token[0] == MARK_CLOSE:
            tag = STYLE_TAGS[token[1]]
            if tag not in stack:
                continue
            reopen = []
            while stack:
                top = stack.pop()
                out.append(f"</{top}>")
                if top == tag:
                    break
                reopen.append(top)
            for top in reversed(reopen):
                stack.append(top)
                out.append(f"<{top}>")
        else:
            out.append(html.escape(xml_safe(token), quote=False))
    while stack:
        out.append(f"</{stack.pop()}>")
    result = "".join(out)
    # Thẻ rỗng sinh ra khi kiểu chữ chỉ phủ khoảng trắng.
    return re.sub(r"<(b|i|sup)>(\s*)</\1>", r"\2", result)


def detect_language(samples: Iterable[str], default: str = "en") -> str:
    """Đoán ngôn ngữ đủ để KOReader chọn đúng bộ ngắt dòng: 'vi' nếu có dấu chỉ tiếng Việt mới có."""
    letters = 0
    vietnamese = 0
    for sample in samples:
        for ch in unicodedata.normalize("NFD", sample or ""):
            if ch.isalpha():
                letters += 1
                if ch in "đĐ":
                    vietnamese += 1
            elif ch in _VI_MARKS:
                vietnamese += 1
        if letters > 200_000:
            break
    if letters and vietnamese / letters >= 0.01:
        return "vi"
    return default

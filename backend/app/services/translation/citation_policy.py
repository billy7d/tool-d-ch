"""Mục chú thích chỉ gồm trích dẫn tài liệu thì giữ nguyên bản gốc.

Phần Notes của sách khoa học phổ thông phần lớn là trích dẫn ("W. A. Roberts, “Are Animals Stuck in Time?”
Psychological Bulletin 128: 473–89 (2002)."). Sách dịch giữ tên tác giả, tên bài, tên tạp chí theo nguyên văn để
người đọc tra được; cho model dịch thì tên bài bị dịch, số trang/năm dễ lệch, và tốn ~1/3 thời gian dịch cả sách
(Stumbling on Happiness: 387 mục chú thích). Chỉ đổi chữ dẫn "See also"/"See" thành "Xem thêm"/"Xem".

Mục có lời bình của tác giả ("Phineas apparently took the rod with him...", "Ibid. Since this paper was published,
...") vẫn được dịch như đoạn văn thường.
"""
import re

# Từ thường gặp trong trích dẫn (không phải lời văn cần dịch).
_CITATION_WORDS = {
    "see", "also", "ibid", "ed", "eds", "trans", "vol", "vols", "pp", "p", "no", "nos", "cf", "et", "al", "in", "and",
    "of", "the", "a", "an", "for", "on", "to", "from", "by", "with", "as", "at", "or", "press", "university",
    "journal", "review", "originally", "published", "reprinted", "translated", "edition", "rev", "quoted", "chap",
    "chapter", "part", "pt", "n", "op", "cit", "idem", "accessed", "available", "retrieved", "unpublished",
    "manuscript", "dissertation", "thesis", "paper", "presented", "meeting", "annual", "conference", "proceedings",
    "personal", "communication", "forthcoming", "submitted", "series", "supplement", "issue", "special", "ser",
    "abstract", "abstracts", "doctoral", "master's", "dept", "department", "report", "technical", "working",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
    "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "et al", "via", "ff", "esp", "e.g", "i.e", "orig", "repr", "comp", "introduction", "foreword", "preface",
}
_NOTE_NUMBER = re.compile(r"^\s*(\d{1,3})\.\s+")
_QUOTED = re.compile(r"“[^”]*”|\"[^\"]*\"")
_BRACKETED = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’]*")
_YEAR = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")
_PAGES = re.compile(r"\d+\s*[:–-]\s*\d+")
_SIGNAL = re.compile(r"\bIbid\b|\bet al\.|“[^”]{6,}”")
_SEE = re.compile(r"(^|(?<=[.;]\s))See also\b|(^|(?<=[.;]\s))See\b")


def is_bibliographic_citation(text: str) -> bool:
    """Mục chú thích chỉ gồm tên tác giả, tên tài liệu, nơi xuất bản, năm, số trang."""
    body = _NOTE_NUMBER.sub("", text or "").strip()
    if not body or len(body) > 1500:
        return False
    if not (_YEAR.search(body) or _PAGES.search(body) or _SIGNAL.search(body)):
        return False
    stripped = _BRACKETED.sub(" ", _QUOTED.sub(" ", body))
    stripped = _BRACKETED.sub(" ", stripped)
    prose = [
        word for word in _WORD.findall(stripped)
        if word[0].islower() and word.lower().strip("'’") not in _CITATION_WORDS
    ]
    return len(prose) <= 2


def citation_passthrough(text: str) -> str:
    """Bản "dịch" của mục trích dẫn: giữ nguyên, chỉ đổi chữ dẫn See/See also."""
    def replace(match: re.Match) -> str:
        return "Xem thêm" if match.group(0).endswith("also") else "Xem"

    return _SEE.sub(replace, text or "")


__all__ = ["is_bibliographic_citation", "citation_passthrough"]

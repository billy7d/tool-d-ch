"""Sửa bằng quy tắc các lỗi hình thức mà chắc chắn sửa được, không gọi model.

Các lỗi này lặp lại trong bản dịch thật của qwen2.5:7b và không cần hiểu nghĩa để sửa:
- chữ dính: "từGod", "một ngày kiaRachel", "giải quyết.Paradox này", "sự.relief";
- ngoặc kép sai kiểu hoặc đảo chiều: `”Kathy“, tôi nói tiếp: ”Tôi có thể…"`;
- thiếu ngoặc đóng ở cuối lời thoại.

Mỗi quy tắc chỉ chạy khi nguồn cho căn cứ rõ (từ bị dính có trong nguồn, số ngoặc bằng số ngoặc của nguồn…).
Không đủ căn cứ thì để nguyên cho gate và critic xử lý như trước.
"""
import re

from app.services.translation.quality_gate import _VI_SYLLABLE

_VI_LOWER = "a-zà-ỹđ"
_STUCK_CAPITAL = re.compile(rf"(?<![A-Za-zÀ-ỹĐđ])([{_VI_LOWER}]{{2,}})([A-Z][a-z]{{2,}})")
_STUCK_SENTENCE = re.compile(rf"([{_VI_LOWER}]{{2,}})\.([A-ZÀ-ỸĐ][{_VI_LOWER}]+)")
_STUCK_DOT_WORD = re.compile(rf"([{_VI_LOWER}]{{2,}})\.([a-z]{{3,}})\b")
_DOUBLE_QUOTES = re.compile(r"[“”\"]")
_SOURCE_WORD = re.compile(r"[A-Za-z]+")
# Ngoặc thẳng trong code (`print("x")`, key="v") phải giữ nguyên.
_CODE_CONTEXT = re.compile(r"`|[=(\[{]\s*\"|\"\s*[)\]}]")


def _has_diacritic(word: str) -> bool:
    return not word.isascii()


def _looks_vietnamese(word: str) -> bool:
    return _has_diacritic(word) or bool(_VI_SYLLABLE.match(word))


def split_stuck_words(source: str, text: str) -> str:
    """Tách chữ tiếng Việt dính liền một từ lấy từ nguồn: "từGod" -> "từ God"."""
    source = source or ""
    source_words = {word.lower() for word in _SOURCE_WORD.findall(source)}

    def capital(match: re.Match) -> str:
        left, right = match.group(1), match.group(2)
        if match.group(0) in source or right.lower() not in source_words or not _looks_vietnamese(left):
            return match.group(0)  # "iPhone", "eBay" hay tên có trong nguồn thì giữ nguyên
        return f"{left} {right}"

    def sentence(match: re.Match) -> str:
        if match.group(0) in source:
            return match.group(0)
        return f"{match.group(1)}. {match.group(2)}"

    def dot_word(match: re.Match) -> str:
        left, right = match.group(1), match.group(2)
        if match.group(0) in source or right not in source_words or not _has_diacritic(left):
            return match.group(0)  # "example.com", "e.g" giữ nguyên
        return f"{left} {right}"

    text = _STUCK_CAPITAL.sub(capital, text)
    text = _STUCK_SENTENCE.sub(sentence, text)
    return _STUCK_DOT_WORD.sub(dot_word, text)


def repair_quotes(source: str, text: str) -> str:
    """Đưa ngoặc kép về đúng kiểu và chiều của nguồn khi hai bên có cùng số ngoặc; thêm ngoặc đóng bị thiếu ở cuối."""
    source = source or ""
    source_marks = _DOUBLE_QUOTES.findall(source)
    # Chỉ làm khi nguồn dùng ngoặc cong và cân bằng: khi đó thứ tự mở/đóng của nguồn là căn cứ chắc chắn.
    if not source_marks or '"' in source_marks or source.count("“") != source.count("”"):
        return text
    marks = _DOUBLE_QUOTES.findall(text)
    if '"' in marks and _CODE_CONTEXT.search(text):
        return text
    if len(marks) == len(source_marks) and marks != source_marks:
        wanted = iter(source_marks)
        text = _DOUBLE_QUOTES.sub(lambda _: next(wanted), text)
        text = re.sub(r"“[ \t]+", "“", text)
        text = re.sub(r"[ \t]+”(?=[\s,.;:!?]|$)", "”", text)
        # Ngoặc mở dính vào chữ đứng trước sau khi đổi chiều: `tiếp:“Tôi` vẫn đọc được, không đụng thêm.
        return text
    if (
        len(marks) == len(source_marks) - 1 and '"' not in marks and marks and marks[-1] == "“"
        and text.count("“") == text.count("”") + 1
        and source.rstrip().endswith("”") and not text.rstrip().endswith("”")
    ):
        return text.rstrip() + "”"
    return text


_ENGLISH_TITLES = {"Mr": "ông", "Mrs": "bà", "Ms": "bà", "Miss": "cô", "Dr": "bác sĩ", "Sir": "ngài"}
# Danh xưng tiếng Việt đứng ngay trước danh xưng tiếng Anh của cùng người: "ông Mr. Cocksure", "Bà Mrs. Oakshott".
_DOUBLE_TITLE = re.compile(
    r"\b(ông|bà|cô|ngài|quý ông|quý bà|bác sĩ|tiến sĩ)\s+(Mr|Mrs|Ms|Miss|Dr|Sir)\.?\s+(?=[A-Z])", re.IGNORECASE,
)


def drop_double_titles(source: str, text: str) -> str:
    """Bỏ danh xưng tiếng Anh khi bản dịch đã có danh xưng tiếng Việt cho cùng tên (dịch thử Sherlock Holmes)."""
    def replace(match: re.Match) -> str:
        if not re.search(rf"\b{match.group(2)}\b\.?\s+[A-Z]", source or ""):
            return match.group(0)
        return match.group(1) + " "
    return _DOUBLE_TITLE.sub(replace, text)


def wrap_lost_quotes(source: str, text: str) -> str:
    """Nguồn là trọn một lời thoại trong ngoặc mà bản dịch mất cả hai ngoặc: thêm lại.

    Dịch thử Sherlock Holmes: “That’s the list of the folk from whom I buy…” thành đoạn không có ngoặc nào, đọc như lời dẫn.
    """
    source = (source or "").strip()
    stripped = text.strip()
    if (
        source.startswith("“") and source.endswith("”") and source.count("“") == 1 and source.count("”") == 1
        and stripped and not _DOUBLE_QUOTES.search(stripped)
    ):
        return f"“{stripped}”"
    return text


def repair_form(source: str, text: str) -> str:
    if not text:
        return text
    text = repair_quotes(source, split_stuck_words(source, text))
    return drop_double_titles(source, wrap_lost_quotes(source, text))


__all__ = ["repair_form", "repair_quotes", "split_stuck_words"]

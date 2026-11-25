"""Kiểm tra bắt buộc từng chữ sau OCR, rồi tự quét lại vùng chữ lỗi cho tới khi không còn lỗi.

Sau khi OCR, sửa dấu và bỏ dòng đầu trang, mọi chữ của trang scan phải thỏa một điều kiện bắt buộc
(check_token). Chữ hợp lệ khi nó là:
- âm tiết tiếng Việt có trong từ điển, hoặc từ tiếng Anh có trong từ điển Anh-Việt;
- số, ngày tháng, khoảng số, phần trăm, mã ("COVID-19", "3D", "60th");
- đường dẫn, email, hay một mảnh đường dẫn bị xuống dòng;
- tên riêng hay chữ viết tắt có hình dạng bình thường;
- dấu câu hay ký hiệu thường gặp.

Chữ lỗi chia theo loại:
- "garbled": ký tự lạ hoặc chữ lẫn số ("øƒ", "7e", "J2z72l");
- "digits": mã chữ+số đáng ngờ ("Z7", "17s");
- "symbol": ký hiệu lạ ("`", "J]im");
- "stray": một chữ cái lẻ loi ("z", "g");
- "joined": hai chữ dính nhau ("ofFame", "máigiữa", "cuộcphỏngvấn");
- "case": chữ hoa lạc giữa từ ("JournAl", "MilI");
- "unknown": chữ không có trong từ điển ("tổi", "jWz").

Chữ chính cuốn sách dùng nhiều lần với độ tin cậy cao thì được coi là chữ có thật ("premortem", "xèng").

Chữ lỗi được xử lý theo ba bước:

1. Sửa ngay khi chắc chắn mà không cần đọc lại (quick_fix, join_split_words):
   - tách chữ dính, thêm khoảng trắng thiếu sau dấu câu ("Collins,2009." thành "Collins, 2009.");
   - nối từ bị tách đôi ("Jour nal" thành "Journal");
   - đổi "~" trong khoảng số ("109~26" thành "109–26"), "ó" lẫn trong số ("1ó" thành "16");
   - đổi ngoặc kép trong chữ ("don”t");
   - sửa chữ hoa lạc ("JournAl");
   - chữ C/G của phông tiêu đề đọc thành "t"/"ô" ("tƯỢt" thành "CƯỢC");
   - lỗi quen thuộc trong đường dẫn ("htrp://", "⁄").
2. Cắt riêng các dòng có chữ lỗi và quét lại theo từng lượt (PASSES). Mỗi lượt có ba cách đọc khác nhau về xử lý
   ảnh (phóng to 2 lần, nắn chữ nghiêng nhiều góc, nhị phân hóa, làm đậm nét) và mô hình (vie, eng, vie+eng).
   Sau mỗi lượt, mỗi vùng lỗi được bỏ phiếu theo mặt chữ (bỏ dấu) của các cách đọc. Bản đọc gốc tính một phiếu,
   trừ khi nó có ký tự lạ. Cách đọc còn ký tự lạ, hay không cho ra chữ thay hợp lệ, là phiếu trắng.
   Một vùng được giải quyết khi một trong các điều kiện sau đúng:
   - "fixed": một cách đọc hợp lệ thắng rõ (ít nhất 2 phiếu, hơn phương án nhì 2 phiếu). Chữ tiếng Việt lấy dấu
     từ bản đọc "vie", vì "eng" không biết dấu. Tên lạ phải có mặt ở chỗ khác trong sách, hoặc được từ 4 cách
     đọc đồng ý. Câu tiếng Việt không nhận từ chỉ có trong tiếng Anh, và ngược lại (adapt_candidate).
   - "confirmed": các cách đọc lại khẳng định đúng chữ gốc (chữ hiếm mà đúng). Chữ có ký tự lạ không bao giờ
     được khẳng định.
   - "dropped": chữ vỡ mà mọi cách đọc lại đều không thấy chữ nào ở đó (vết bẩn, hoa văn, mã vạch).
   Vùng chưa giải quyết đi tiếp sang lượt sau với cách đọc khác. Lượt dừng khi hết vùng lỗi hoặc hết cách đọc.
3. Hết lượt mà vẫn chưa giải quyết được (bảo đảm cuối cùng: không để lại chuỗi ký tự vô nghĩa):
   - "fixed": cách đọc hợp lệ được nhiều phiếu nhất, hơn hẳn phương án khác;
   - "guessed": cách đọc hợp lệ mà cả mô hình Việt lẫn Anh cùng đọc ra (từ 2 phiếu);
   - "removed": chuỗi vỡ hay chữ cái lẻ không còn mảnh chữ có thật nào, không phải số dài hay đường dẫn;
   - "kept": chữ hình dạng bình thường mà không có trong từ điển thì giữ nguyên;
   - "unresolved": còn lại (rất ít) thì giữ nguyên.
   Chữ "guessed" và "unresolved" được bên gọi liệt kê theo số trang để soát lại.

Các dòng của nhiều trang được ghép thành một ảnh cho mỗi cách đọc: Tesseract chỉ phải khởi động một lần cho cả lô.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageFilter

from app.services.ebook.ocr_repair import core_of, is_english_word, is_garbled, shear
from app.services.ebook.vi_spell import VietSyllableModel, is_vietnamese_syllable_shape

# ---- Điều kiện bắt buộc cho từng chữ -----------------------------------------------------------------

# Đường dẫn: có giao thức/www/email/tên miền, hoặc đường dẫn có ít nhất hai dấu "/", hoặc "đoạn/đoạn" mỗi đoạn từ
# 3 ký tự, hoặc "đoạn/" cuối dòng. "s/", "/Zz", "J2722/", "S/g24l" (chữ nghiêng đọc vỡ) thì không.
_LINK = re.compile(r"(?:https?://|www\.|\w@\w)|\w\.(?:com|org|net|edu|gov|uk|vn|io|html?|pdf|php|aspx?)\b"
                   r"|[\w#.-]/[\w#.-]*/|^[\w#.-]*[a-z][\w#.-]{2,}/[\w#.-]{3,}|^[a-z.-]{4,}/$", re.I)
# Mảnh đường dẫn bị xuống dòng: "why_pete_carroll_s", "0cbf01c9cf902", "mobile-app-uses-3d-technology".
_LINK_PIECE = re.compile(r"[A-Za-z0-9]_|_[A-Za-z0-9]|^(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{8,}$"
                         r"|^(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{6,}-$"
                         r"|^[a-z0-9]+(?:-[a-z0-9]+){3,}$")
_URL_CHARS = re.compile(r"^[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%\-"
                        "ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"
                        "ĂÂĐÊÔƠƯÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴ]+$")
_NUMBER = re.compile(
    r"^[+\-−]?\d[\d.,]*(?:%|°|đ|\$|cm|km|kg|m)?$"  # 45, 6,25%, 91,44cm
    r"|^\$\d[\d.,]*$"
    r"|^\d{1,2}/\d{1,2}(?:/\d{2,4})?$|^\d+/\d+$"  # 2/2/2015, 9/2016
    r"|^[A-Z]?\d+(?:[-–—][A-Z]?\d+)+$"  # 109–26, 10-10-10, S23–S37
    r"|^\d+:\d+$"
    r"|^[\d-]{10,}$"  # ISBN
    r"|^[\d+\-=×*/÷^]+[\d—–]*$"  # 2+2=4
    r"|^#\d+$"
)
_CODE = re.compile(r"^(?:\d{1,4}(?:st|nd|rd|th|AM|PM|D|K|M|B)|[A-Z]{1,4}-?\d{1,4}[A-Za-z]?|\d{4}s)$")
_WORD_NUMBER = re.compile(r"^[^\W\d_]+(?:-[^\W\d_]+)*-\d+$")  # "election-231036", "COVID-19"
_LIST_MARK = re.compile(r"^\(?[a-zA-Z]\)[.,;:]?$|^[a-zA-Z]\.$")  # "(b)", "c)", "B."
# "#MAGA", "S&P", "P.7"/"Q.3" (phường, quận), "F.3d" (trích dẫn án lệ).
_SPECIAL = re.compile(r"^#[A-Za-z]\w*$|^[A-Z]{1,4}&[A-Z]{1,4}$|^[A-Z]\.\d{1,3}$|^[A-Z]\.\d[a-z]{1,2}$")
# Chữ hoa lạc vào giữa một từ có thật: "JournAl", "RoyAl" (chữ a nghiêng đọc thành A).
_STRAY_CAP = re.compile(r"^[A-Z]?[a-z]+[A-Z][a-z]*$")
_PUNCT_OK = set(".,;:!?…'\"“”‘’()[]-–—/%&*+=#|•→←↑↓>_<©®°§$€£")
_GARBLED_REASONS = {"garbled", "symbol"}
_BROKEN_REASONS = _GARBLED_REASONS | {"digits", "stray"}

# Hàm kiểm tra sẽ được gọi cho hàng vạn chữ: nhớ kết quả từ điển.
_known_cache: Dict[Tuple[int, str], bool] = {}


def is_known_word(word: str, model: VietSyllableModel) -> bool:
    """Âm tiết tiếng Việt có trong từ điển hoặc từ tiếng Anh (chữ thường)."""
    if not word:
        return True
    key = (id(model), word)
    hit = _known_cache.get(key)
    if hit is None:
        hit = (is_vietnamese_syllable_shape(word) and model.unigrams.get(word, 0) >= 3
               and any(ch in _VOWELS for ch in word)) or (word.isascii() and word.isalpha() and is_english_word(word))
        if len(_known_cache) > 200_000:
            _known_cache.clear()
        _known_cache[key] = hit
    return hit


def _looks_like_name(core: str) -> bool:
    """Tên riêng/viết tắt chưa có trong từ điển: chữ Latin mở đầu bằng chữ hoa, không lẫn hoa-thường lộn xộn."""
    if not core[:1].isupper() or not core.replace("'", "").replace("’", "").isalpha():
        return False
    if core.isupper():
        return len(core) <= 6
    if not core.isascii():
        return False
    return not re.search(r"[a-z][A-Z]", core[1:])


def _has_word(token: str, model: VietSyllableModel, vocab: Counter) -> bool:
    """Trong chuỗi (kể cả chuỗi vỡ) còn một mảnh chữ có thật dài từ 3 chữ cái: không phải rác để bỏ đi."""
    for piece in re.findall(r"[^\W\d_]{3,}", token):
        low = piece.lower()
        if is_known_word(low, model) or vocab.get(low, 0) >= 2:
            return True
    return False  # "jWz", "ZZzzø" thì không; "McFly", "MacCoun" xét riêng


def check_token(token: str, model: VietSyllableModel, conf: Optional[float] = None,
                accepted: Iterable[str] = ()) -> Optional[str]:
    """None nếu chữ thỏa điều kiện bắt buộc, ngược lại là loại lỗi (xem đầu mô-đun).

    accepted: chữ (thường) mà cả sách dùng nhiều lần với độ tin cậy cao: coi là có thật dù từ điển không có.
    """
    if not any(ch.isalnum() for ch in token):
        return None if all(ch in _PUNCT_OK for ch in token) else "symbol"  # "—>", "|", "…" thì được; "`", "‹««" thì không
    core = core_of(token)
    if not core:
        return None
    if (_LINK.search(core) or _LINK_PIECE.search(core)) and _URL_CHARS.match(core) and (
            core.isascii() or re.search(r"https?://|www\.|\w\.[a-z]{2,4}(?:/|$)", core)):
        return None  # đường dẫn (có chữ tiếng Việt thì phải có tên miền: "vi.wikipedia.org/wiki/Xì_tố")
    if _NUMBER.match(core) or _LIST_MARK.match(token) or _WORD_NUMBER.match(core) or _SPECIAL.match(core):
        return None
    if _CODE.match(core) and (conf is None or conf >= 0.8):
        return None
    if is_garbled(token):
        return "garbled"
    low = unicodedata.normalize("NFC", core.lower())
    if any(ch.isdigit() for ch in core):
        return "digits"
    if "".join(ch for ch in low if ch.isalpha()) in accepted:
        return None
    # "don’t", "Harvard's": chỉ xét phần trước dấu nháy khi phần sau là đuôi tiếng Anh.
    low = re.sub(r"['’](?:s|t|d|ll|re|ve|m)$", "", low)
    parts = [p for p in re.split(r"[-–—'’/.]", low) if p]
    if not parts or any(not p.isalpha() for p in parts):
        return "symbol"
    if len(core) == 1:
        if model.unigrams.get(low, 0) >= 3 or core.isupper() or low in "x&":
            return None
        return "stray"
    if split_joined(core, model):
        return "joined"
    if _STRAY_CAP.match(core) and fix_stray_cap(core, model):
        return "case"
    for part in parts:
        if is_known_word(part, model):
            continue
        if _looks_like_name(core):
            continue
        return "unknown"
    return None


def is_corrupt(reason: Optional[str]) -> bool:
    """Lỗi có ký tự lạ: không chữ nào như vậy là đúng, không bao giờ được "khẳng định"."""
    return reason in _GARBLED_REASONS


# ---- Sửa ngay, không cần đọc lại ---------------------------------------------------------------------

_FUNCTION_EN = ("of", "the", "and", "in", "to", "on", "for", "with", "from", "by", "at", "a", "an", "or")
_VI_GLUE = {"và", "của", "là", "có", "trong", "với", "một", "các", "cho", "không", "được", "những", "này", "đã", "để"}


def split_joined(core: str, model: VietSyllableModel) -> Optional[str]:
    """Hai chữ dính nhau do OCR mất khoảng trắng: "ofFame" -> "of Fame", "máigiữa" -> "mái giữa",
    "Hastorfvà" -> "Hastorf và", "tríHãy" -> "trí Hãy". None nếu không chắc."""
    if len(core) < 4 or not core.isalpha():
        return None
    low = core.lower()
    if is_known_word(low, model):
        return None
    # Chữ thường liền chữ hoa: "ofFame", "tríHãy". Tên có chữ hoa ở giữa ("McFly", "MacCoun") mở đầu bằng chữ hoa.
    for i in range(1, len(core)):
        if core[i].isupper() and core[i - 1].islower():
            left, right = core[:i], core[i:]
            if left[:1].islower() and is_known_word(left.lower(), model) and len(right) >= 2 and (
                    is_known_word(right.lower(), model) or _looks_like_name(right)):
                return f"{left} {right}"
            return None
    vietnamese = not low.isascii()
    if vietnamese and len(low) > 7 and not core[:1].isupper():
        pieces = _split_syllables(low, model)  # "cuộcphỏngvấn" -> "cuộc phỏng vấn"
        if pieces and sum(not p.isascii() for p in pieces) >= 2:
            out, position = [], 0
            for piece in pieces:
                out.append(core[position:position + len(piece)])
                position += len(piece)
            return " ".join(out)
    for i in range(1, len(low)):
        left, right = low[:i], low[i:]
        if len(left) < 2 or len(right) < 2:
            continue
        if vietnamese:
            # Hai âm tiết tiếng Việt có dấu dính nhau ("máigiữa"), một âm tiết có dấu + chữ nối ("dícho"),
            # hoặc tên Latin + "và"/"của"... ("Hastorfvà"). Tên Latin khác ("Riích") thì không tách.
            syllables = all(
                is_vietnamese_syllable_shape(p) and model.unigrams.get(p, 0) >= 50 and any(v in _VOWELS for v in p)
                for p in (left, right))
            marked = (not left.isascii(), not right.isascii())
            vi_pair = syllables and (all(marked) or (any(marked) and (left in _VI_GLUE or right in _VI_GLUE)))
            name_glue = core[:1].isupper() and left.isascii() and len(left) >= 3 and right in _VI_GLUE
            if (vi_pair and not (core[:1].isupper() and left.isascii())) or name_glue:
                return f"{core[:i]} {core[i:]}"
        elif core[:1].islower() and left in _FUNCTION_EN and len(right) >= 3 and is_english_word(right):
            return f"{core[:i]} {core[i:]}"  # "ofthe", "inthe"; tên ("Orson", "Andrew") không tách
    return None


def _split_syllables(low: str, model: VietSyllableModel, limit: int = 4) -> Optional[List[str]]:
    """Chia chuỗi âm tiết tiếng Việt dính liền thành ít mảnh nhất, mỗi mảnh là âm tiết thông dụng (có nguyên âm)."""
    best: Dict[int, List[str]] = {0: []}
    for end in range(1, len(low) + 1):
        for start in range(max(0, end - 7), end - 1):
            if start not in best or len(best[start]) >= limit:
                continue
            piece = low[start:end]
            if not (is_vietnamese_syllable_shape(piece) and model.unigrams.get(piece, 0) >= 50
                    and any(ch in _VOWELS for ch in piece)):
                continue
            candidate = best[start] + [piece]
            if end not in best or len(candidate) < len(best[end]):
                best[end] = candidate
    result = best.get(len(low))
    return result if result and len(result) >= 2 else None


_VI_ONLY = set("ăâđêôơư" "àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ")
_VOWELS = set("aeiouyăâêôơư" "àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ")


# Phông tiêu đề chữ hoa đứng (condensed): "C" đọc thành "t"/"(", "G" thành "ô", "O" thành "0".
_CAPS_CONFUSIONS = {"t": "C", "(": "C", "ô": "G", "0": "O", "Ô": "G", "$": "C", "1": "I", "l": "I", "|": "I"}


def fix_caps_word(core: str, model: VietSyllableModel) -> Optional[str]:
    """Chữ in hoa của tiêu đề bị đọc lẫn ký tự: "tƯỢt" -> "CƯỢC", "NHỮNô" -> "NHỮNG", "HỌt" -> "HỌC"."""
    if len(core) < 2 or sum(ch.isupper() for ch in core) < max(1, len(core) // 2):
        return None
    positions = [i for i, ch in enumerate(core) if ch in _CAPS_CONFUSIONS]
    if not positions or len(positions) > 3:
        return None
    rest = [ch for i, ch in enumerate(core) if i not in positions]
    if any(ch.isalpha() and not ch.isupper() for ch in rest):
        return None
    fixed = list(core)
    for i in positions:
        fixed[i] = _CAPS_CONFUSIONS[core[i]]
    word = "".join(fixed)
    low = word.lower()
    if is_vietnamese_syllable_shape(low) and model.unigrams.get(low, 0) >= 3 and word != core:
        return word
    return None


def fix_stray_cap(core: str, model: VietSyllableModel) -> Optional[str]:
    """Chữ hoa lạc giữa từ: "JournAl" -> "Journal", "RoyAl" -> "Royal"; "I" lạc thường là "l" ("MilI" -> "Mill")."""
    head, rest = core[:1], core[1:]
    for variant in (head + rest.replace("I", "l").lower(), head + rest.lower()):
        low = variant.lower()
        if variant != core and (is_english_word(low) if low.isascii() else is_known_word(low, model)):
            return variant
    return None


def _missing_space(core: str, model: VietSyllableModel) -> Optional[str]:
    """Thiếu khoảng trắng sau dấu câu: "cả,một" -> "cả, một", "Collins,2009." -> "Collins, 2009.",
    "Kaplan):You" -> "Kaplan): You", "thống“Bookies" -> "thống “Bookies". Hai nửa đều phải là chữ hợp lệ."""
    for match in re.finditer(r"[,;:)!?.]+(?=[“\"(]?[^\W_])|(?:(?<=[^\W_])|(?<=”))(?=[“(])", core):
        left, right = core[:match.end()], core[match.end():]
        if not left.strip(_SPLIT_EDGE) or not right.strip(_SPLIT_EDGE):
            continue
        if left.rstrip(_SPLIT_EDGE)[-1:].isdigit() and right.lstrip(_SPLIT_EDGE)[:1].isdigit():
            continue  # "6,25", "1.000"
        if match.group() == "." and not right.lstrip(_SPLIT_EDGE)[:1].isupper():
            continue  # "e.g", "U.S"
        if check_token(left, model) is None and check_token(right, model) is None:
            return f"{left} {right}"
    return None


_SPLIT_EDGE = ".,;:!?…'\"“”‘’()[]"


def quick_fix(token: str, model: VietSyllableModel) -> Optional[str]:
    """Bản sửa chắc chắn cho một chữ lỗi mà không cần đọc lại ảnh (có thể thành hai chữ), hoặc None."""
    core = core_of(token)
    if not core:
        return None
    lead = token[: token.find(core)]
    trail = token[token.find(core) + len(core):]
    fixed: Optional[str] = None
    if re.match(r"^(?:h[rt]{2}ps?|ht+ps?):/{1,2}|^www\.", core, re.I) or ("/" in core and "." in core and "⁄" in core):
        # Đường dẫn: "htrp://", "hrtps://" -> "http(s)://"; "⁄" (gạch phân số) -> "/"; "o£" -> "of".
        fixed = re.sub(r"^h[rt]{2}p(s?):/{1,2}", lambda m: f"http{m.group(1)}://", core, flags=re.I)
        fixed = fixed.replace("⁄", "/").replace("£", "f")
    elif re.fullmatch(r"\d+~\d+", core):
        fixed = core.replace("~", "–")
    elif re.fullmatch(r"[\dóÓ]+%?", core) and any(ch.isdigit() for ch in core):
        fixed = core.replace("ó", "6").replace("Ó", "6")  # chữ số 6 của phông sách đọc thành "ó": "1ó" -> "16"
    elif re.search(r"[^\W\d_][”\"](?:s|t|d|ll|re|ve|m)$", core, re.I):
        fixed = re.sub(r"(?<=[^\W\d_])[”\"](?=(?:s|t|d|ll|re|ve|m)$)", "’", core, flags=re.I)  # "don”t", "ISN”T"
    elif _STRAY_CAP.match(core) and fix_stray_cap(core, model):
        fixed = fix_stray_cap(core, model)
    elif re.fullmatch(r"[^\W\d_]+:[^\W\d_]+", core):
        left, right = core.split(":", 1)
        if is_known_word(left.lower(), model) and is_known_word(right.lower(), model):
            fixed = f"{left} {right}"  # "tư:vấn" -> "tư vấn" (vết bẩn giữa hai chữ)
    elif _missing_space(core, model):
        fixed = _missing_space(core, model)
    elif re.fullmatch(r"[^\W\d_]{2,}\d{2,}|\d{1,4}[A-Z][a-z]{2,}", core):
        fixed = re.sub(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])", " ", core, count=1)  # "Ba2017" "4Annual"
    elif "ñ" in core:
        # Chữ ghép "fi" đọc thành "ñ": "ñnal" -> "final" (chỉ khi ra từ tiếng Anh có thật).
        fixed = re.sub(r"[A-Za-z]*ñ[A-Za-z]*",
                       lambda m: m.group().replace("ñ", "fi") if is_english_word(m.group().replace("ñ", "fi").lower())
                       else m.group(), core)
    else:
        fixed = fix_caps_word(core, model) or split_joined(core, model)
    if not fixed or fixed == core:
        return None
    if any(check_token(part, model) for part in fixed.split()):
        return None
    return lead + fixed + trail


def join_split_words(tokens: Sequence[str], reasons: Dict[int, str], model: VietSyllableModel
                     ) -> Dict[int, List[str]]:
    """Một từ bị OCR tách đôi ("Jour nal" -> "Journal", "Neuro science" -> "Neuroscience"): hai chữ liền nhau, ít
    nhất một chữ lỗi, ghép lại thành từ tiếng Anh có thật dài từ 5 chữ cái. Trả về bảng thay {i: [chữ ghép], i+1: []}."""
    out: Dict[int, List[str]] = {}
    for i in range(len(tokens) - 1):
        if i in out or (i not in reasons and i + 1 not in reasons):
            continue
        a, b = tokens[i], tokens[i + 1]
        left = a.lstrip(_SPLIT_EDGE).replace("]", "")
        right = b.rstrip(_SPLIT_EDGE)
        if not left.isalpha() or not right.isalpha() or not right.islower() or not left.isascii() or not right.isascii():
            continue
        joined = (left + right).lower()
        if len(joined) >= 5 and is_english_word(joined) and not (is_english_word(left.lower()) and is_english_word(right.lower())):
            lead = a[: len(a) - len(a.lstrip(_SPLIT_EDGE))]
            trail = b[len(b.rstrip(_SPLIT_EDGE)):]
            out[i] = [lead + left + right + trail]
            out[i + 1] = []
    return out


# ---- Quét lại ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Strategy:
    lang: str
    scale: float = 1.0
    shear: float = 0.0
    filter: str = ""  # "" | "otsu" (nhị phân hóa) | "thick" (làm đậm nét)

    @property
    def name(self) -> str:
        return f"{self.lang}-x{self.scale:g}-k{self.shear:g}{('-' + self.filter) if self.filter else ''}"


# Thứ tự đo trên sách thật (Tư Duy Đặt Cược, 1.500 chữ lỗi): lượt đầu là ba cách đọc cho nhiều chữ hợp lệ nhất,
# các lượt sau thử góc nghiêng, nhị phân hóa, làm đậm nét khác.
PASSES: Tuple[Tuple[Strategy, ...], ...] = (
    (Strategy("vie", 2.0), Strategy("eng", 2.0, 0.25), Strategy("vie", 2.0, 0.25)),
    (Strategy("vie+eng", 2.0, 0.25), Strategy("eng", 1.0, 0.18), Strategy("vie", 1.0, 0.35)),
    (Strategy("eng", 2.0), Strategy("vie", 2.0, 0.25, "otsu"), Strategy("eng", 2.0, 0.25, "thick")),
    (Strategy("vie", 1.0, 0.15), Strategy("vie", 2.0, 0.0, "otsu"), Strategy("vie", 2.0, 0.0, "thick")),
)

_PAD = 8
_GAP = 30
BATCH_LINES = 40


@dataclass
class Region:
    """Một cụm chữ lỗi liền nhau trên một dòng."""
    first: int  # chỉ số chữ trong dòng
    last: int
    tokens: List[str]
    reasons: List[str]
    x0: float  # toạ độ trên ảnh cắt của dòng
    x1: float
    readings: List[Tuple[str, str]] = field(default_factory=list)  # (ngôn ngữ, chữ đọc được)
    before: str = ""  # chữ đứng ngay trước/sau vùng trên dòng (ngữ cảnh)
    after: str = ""
    status: str = ""  # "" | "fixed" | "confirmed" | "dropped" | "unresolved" | "kept"
    result: Optional[str] = None

    @property
    def original(self) -> str:
        return " ".join(self.tokens)

    @property
    def corrupt(self) -> bool:
        """Có ký tự lạ: không bao giờ được "khẳng định" là đúng."""
        return any(is_corrupt(r) for r in self.reasons)

    @property
    def broken(self) -> bool:
        """Chữ hỏng (ký tự lạ, chữ lẫn số, chữ cái lẻ): được thay bằng cách đọc lại dù mặt chữ khác hẳn."""
        return any(r in _BROKEN_REASONS for r in self.reasons)


@dataclass
class LineJob:
    """Ảnh cắt của một dòng có chữ lỗi và các vùng lỗi trên đó."""
    image: Image.Image
    dpi: int
    regions: List[Region]
    key: object = None  # để bên gọi tìm lại dòng


def word_key(text: str) -> str:
    """Mặt chữ để bỏ phiếu: chữ thường, bỏ dấu và dấu câu ("Journal," và "Jourñal" khác nhau, "và"/"va" như nhau)."""
    out = []
    for word in text.split():
        decomposed = unicodedata.normalize("NFD", word.lower().replace("đ", "d"))
        letters = "".join(ch for ch in decomposed if ch.isascii() and ch.isalnum())
        if letters:
            out.append(letters)
    return " ".join(out)


def _otsu(image: Image.Image) -> Image.Image:
    hist = image.histogram()[:256]
    total = sum(hist)
    sum_all = sum(i * h for i, h in enumerate(hist))
    weight, acc, best, threshold = 0, 0.0, -1.0, 128
    for t in range(256):
        weight += hist[t]
        if weight == 0 or weight == total:
            continue
        acc += t * hist[t]
        m0 = acc / weight
        m1 = (sum_all - acc) / (total - weight)
        between = weight * (total - weight) * (m0 - m1) ** 2
        if between > best:
            best, threshold = between, t
    return image.point(lambda v: 255 if v > threshold else 0)


def prepare(image: Image.Image, strategy: Strategy) -> Image.Image:
    if strategy.scale != 1.0:
        image = image.resize((int(image.width * strategy.scale), int(image.height * strategy.scale)), Image.LANCZOS)
    if strategy.filter == "otsu":
        image = _otsu(image)
    elif strategy.filter == "thick":
        image = image.filter(ImageFilter.MinFilter(3))
    if strategy.shear:
        image = shear(image, strategy.shear)
    return image


def compose_lines(images: Sequence[Image.Image]) -> Tuple[Image.Image, List[int]]:
    """Xếp các ảnh dòng thành một cột, cách nhau đủ xa để Tesseract không gộp dòng. Trả về (ảnh, đỉnh mỗi dòng)."""
    width = max(im.width for im in images) + 2 * _GAP
    height = sum(im.height + _GAP for im in images) + _GAP
    canvas = Image.new("L", (width, height), 255)
    tops = []
    top = _GAP
    for im in images:
        canvas.paste(im, (_GAP, top))
        tops.append(top)
        top += im.height + _GAP
    return canvas, tops


Recognizer = Callable[[Image.Image, str, int], object]


def read_batch(jobs: Sequence[LineJob], strategy: Strategy, recognize: Recognizer) -> List[List[Tuple[str, float, float]]]:
    """Đọc lại một lô dòng bằng một cách đọc. Trả về, theo từng dòng, các chữ (chữ, x0, x1) trên ảnh cắt của dòng."""
    canvas, tops = compose_lines([job.image for job in jobs])
    image = prepare(canvas, strategy)
    result = recognize(image, strategy.lang, int(jobs[0].dpi * strategy.scale))
    out: List[List[Tuple[str, float, float]]] = [[] for _ in jobs]
    s, k = strategy.scale, strategy.shear
    for line in getattr(result, "lines", []):
        for text, box in zip(line.text.split(), line.word_boxes):
            x0, y0, x1, y1 = box
            yc = (y0 + y1) / 2
            shift = -k * yc if k else 0.0  # ảnh nắn: x trên ảnh gốc = x - k*y (xem shear)
            cy = yc / s
            index = next((i for i, (top, job) in enumerate(zip(tops, jobs))
                          if top <= cy <= top + job.image.height), None)
            if index is None:
                continue
            out[index].append((text, (x0 + shift) / s - _GAP, (x1 + shift) / s - _GAP))
    for words in out:
        words.sort(key=lambda w: w[1])
    return out


def region_text(words: Sequence[Tuple[str, float, float]], region: Region, line_height: float) -> str:
    """Chữ của một cách đọc nằm trong vùng lỗi (tâm chữ trong vùng, phủ đủ bề rộng vùng)."""
    tolerance = 0.25 * line_height
    width = max(1.0, region.x1 - region.x0)
    inside = [w for w in words if region.x0 - tolerance <= (w[1] + w[2]) / 2 <= region.x1 + tolerance]
    covered = sum(max(0.0, min(w[2], region.x1) - max(w[1], region.x0)) for w in inside)
    if not inside or covered < 0.5 * width:
        # Có chữ phủ lên vùng nhưng tâm nằm ngoài (cách đọc này gộp vùng với chữ bên cạnh): không phải "trống".
        if any(min(w[2], region.x1) - max(w[1], region.x0) >= 0.3 * width for w in words):
            return MERGED
        return ""
    return " ".join(w[0] for w in inside)


MERGED = "\x02"  # cách đọc gộp vùng lỗi với chữ bên cạnh (tâm chữ nằm ngoài vùng)


# ---- Bỏ phiếu ----------------------------------------------------------------------------------------

def _similar(a: str, b: str) -> float:
    x, y = word_key(a).replace(" ", ""), word_key(b).replace(" ", "")
    if not x or not y:
        return 0.0
    return difflib.SequenceMatcher(None, x, y, autojunk=False).ratio()


def candidate_ok(text: str, model: VietSyllableModel, vocab: Counter, accepted: Iterable[str] = (),
                 strong: bool = False) -> bool:
    """Cách đọc lại được dùng để thay: mọi chữ thỏa điều kiện bắt buộc; tên lạ phải có mặt ở chỗ khác trong sách
    (một cách đọc sai của chữ nghiêng rất dễ trông như một cái tên: "dzách" -> "Azach")."""
    words = text.split()
    if not words:
        return False
    for word in words:
        if check_token(word, model, accepted=accepted):
            return False
        core = core_of(word)
        letters = core.replace("'", "").replace("’", "")
        if letters.isalpha() and not any(is_known_word(p, model) for p in re.split(r"[-–'’/.]", core.lower()) if p):
            if vocab.get(core.lower(), 0) < 2 and not (strong and _looks_like_name(core)):
                return False  # tên/viết tắt lạ: phải có mặt ở chỗ khác trong sách, hoặc rất nhiều cách đọc đồng ý
    return True


def _keep_edges(original: str, new: str) -> str:
    """Giữ dấu câu hai đầu của chữ gốc ("Times," -> "Times,"), bỏ dấu câu lạ do cách đọc lại thêm vào."""
    edge = ".,;:!?…'\"“”‘’()[]-–—*"
    lead = original[: len(original) - len(original.lstrip(edge))]
    trail = original[len(original.rstrip(edge)):]
    body = new.strip(edge)
    if not body:
        return original
    return lead + body + trail


_ITALIC_PAIRS = {frozenset(p) for p in ("bh", "uv", "nu", "il", "lt", "ce", "rn", "ao")}


def _italic_variant(original: str, candidate: str) -> bool:
    """Hai chữ chỉ khác nhau ở các cặp chữ cái hay lẫn khi in nghiêng (và dấu), khác ít nhất một chữ cái."""
    a, b = word_key(original), word_key(candidate)
    if len(a) != len(b) or a == b or " " in a or " " in b:
        return False
    return all(x == y or frozenset((x, y)) in _ITALIC_PAIRS for x, y in zip(a, b))


def _vi_marked(word: str) -> bool:
    return any(ch in _VI_ONLY for ch in core_of(word).lower())


def _english_only(word: str, model: VietSyllableModel) -> bool:
    """Từ tiếng Anh không phải âm tiết tiếng Việt ("gid", "dit"; "and" thì có thể là cả hai: không tính)."""
    core = core_of(word)
    low = core.lower()
    return (bool(low) and core[:1].islower() and low.isascii() and low.isalpha() and is_english_word(low)
            and model.unigrams.get(low, 0) < 3)  # tên riêng viết hoa ("Ulysses") không tính


def _context(region: Region, model: VietSyllableModel) -> str:
    """"vi"/"en" khi chữ đứng hai bên (hoặc một bên, bên kia là đầu/cuối dòng) cùng là tiếng Việt/tiếng Anh."""
    sides = [w for w in (region.before, region.after) if core_of(w)]
    if not sides or (len(sides) == 1 and region.before and region.after):
        return ""
    def vietnamese(word: str) -> bool:
        low = core_of(word).lower()
        return _vi_marked(word) or (model.unigrams.get(low, 0) >= 50 and not is_english_word(low))

    def english(word: str) -> bool:
        low = core_of(word).lower()
        return (not _vi_marked(word) and low.isascii() and low.isalpha() and is_english_word(low)
                and (model.unigrams.get(low, 0) < 50 or len(low) >= 4))

    if all(vietnamese(w) for w in sides):
        return "vi"
    if all(english(w) for w in sides):
        return "en"
    return ""


def adapt_candidate(region: Region, text: str, model: VietSyllableModel, vocab: Counter,
                    accepted: Iterable[str] = (), strong: bool = False) -> Optional[str]:
    """Cách đọc lại -> chữ thay cho vùng lỗi, hoặc None nếu không đạt điều kiện:
    - bỏ chữ thừa trùng với chữ đứng cạnh vùng ("¿bể" đọc thành "thể sẽ" khi chữ sau đã là "sẽ");
    - mọi chữ thỏa điều kiện bắt buộc, tên/viết tắt lạ phải có mặt ở chỗ khác trong sách;
    - chữ gốc gần như toàn số thì bản thay cũng phải có số ("16g966" không thành "gases");
    - ngữ cảnh tiếng Việt không nhận từ chỉ có trong tiếng Anh ("giá" không thành "gid"), ngữ cảnh tiếng Anh
      không nhận chữ có dấu tiếng Việt;
    - nhiều âm tiết tiếng Việt thay cho một chữ thì cặp âm tiết phải có trong từ điển ("yêu cầu" được,
      "bổ sướng" không);
    - sửa chữ hoa lạc ("WAll" -> "Wall").
    """
    words = text.split()
    if len(words) > len(region.tokens) and region.after and _similar(words[-1], region.after) >= 0.6:
        words = words[:-1]
    if len(words) > len(region.tokens) and region.before and _similar(words[0], region.before) >= 0.6:
        words = words[1:]
    if not words:
        return None
    fixed_words = []
    for word in words:
        core = core_of(word)
        if re.fullmatch(r"[A-Z]{2}[a-z]+", core) and is_known_word(core.lower(), model):
            word = word.replace(core, core[:1] + core[1:].lower(), 1)
        fixed_words.append(word)
    if not strong and not any(ch.isalnum() for ch in region.original) and any(ch.isalnum() for ch in " ".join(fixed_words)):
        return None  # chữ gốc chỉ là ký hiệu lạ ("@&", "`"): chỉ thành chữ khi rất nhiều cách đọc đồng ý
    # Chữ đọc lại còn lỗi nhỏ sửa được ngay ("Gleitman,bai" thiếu khoảng trắng).
    repaired = []
    for word in fixed_words:
        if check_token(word, model, accepted=accepted):
            word = quick_fix(word, model) or word
        repaired.extend(word.split())
    fixed_words = repaired
    candidate = _keep_edges(region.original, " ".join(fixed_words))
    if not candidate_ok(candidate, model, vocab, accepted, strong):
        return None
    original = core_of(region.original)
    if sum(ch.isdigit() for ch in original) >= 4 and sum(ch.isdigit() for ch in original) > 0.5 * len(original):
        if not any(ch.isdigit() for ch in candidate):
            return None
    context = _context(region, model)
    if context == "vi" and any(_english_only(w, model) for w in fixed_words):
        return None
    if context == "en" and any(_vi_marked(w) for w in fixed_words):
        return None
    if len(fixed_words) > len(region.tokens):
        lows = [core_of(w).lower() for w in fixed_words]
        for a, b in zip(lows, lows[1:]):
            if (_vi_marked(a) or _vi_marked(b)) and not model.bigrams.get((a, b), 0):
                return None
    return candidate


def _best_text(region: Region, key: str, model: VietSyllableModel, vocab: Counter, accepted,
               votes: int = 0) -> Optional[str]:
    """Trong các cách đọc có cùng mặt chữ: ưu tiên bản "vie" (có dấu) hợp lệ, phổ biến nhất. Bản "eng" không biết
    dấu tiếng Việt (đọc "về" thành "vé"): bỏ dấu của nó, rồi ghép mặt chữ với dấu của bản "vie" cùng độ dài
    ("0ể" + "ve" -> "về"); không ghép được thì để chữ không dấu cho bộ sửa dấu xét theo ngữ cảnh."""
    from app.services.ebook.ocr_repair import fuse

    same = [(lang, text) for lang, text in region.readings if text != MERGED and word_key(text) == key]
    strong = votes >= STRONG_VOTES
    for text, _ in Counter(text for lang, text in same if lang == "vie").most_common():
        fixed = adapt_candidate(region, text, model, vocab, accepted, strong=strong)
        if fixed:
            return fixed
    vie_texts = [text for lang, text in region.readings if lang == "vie" and text != MERGED]
    for text, _ in Counter(_strip_marks(text) for lang, text in same if lang != "vie").most_common():
        if len(text.split()) == 1:
            fused = Counter(f for f in (fuse(v, text, model) for v in vie_texts if len(v.split()) == 1) if f)
            # Nhiều cách ghép: chọn theo số lần, rồi theo độ hợp với chữ đứng trước/sau ("Trở về" hơn "Trở vê").
            before, after = core_of(region.before).lower(), core_of(region.after).lower()
            ranked = sorted(fused.items(), key=lambda item: (
                -item[1], -(model.bigrams.get((before, item[0]), 0) + model.bigrams.get((item[0], after), 0)),
                -model.unigrams.get(item[0], 0)))
            if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                ranked.sort(key=lambda item: (
                    -(model.bigrams.get((before, item[0]), 0) + model.bigrams.get((item[0], after), 0)),
                    -item[1], -model.unigrams.get(item[0], 0)))
            for word, _ in ranked:
                core = core_of(text)
                fixed = adapt_candidate(region, text.replace(core, _match_case(core, word), 1), model, vocab,
                                        accepted, strong=strong)
                if fixed:
                    return fixed
        fixed = adapt_candidate(region, text, model, vocab, accepted, strong=strong)
        if fixed:
            return fixed
    return None


def _strip_marks(text: str) -> str:
    """Bỏ dấu Latin mà mô hình "eng" tự thêm ("vé" -> "ve"); giữ nguyên chữ không phải Latin có dấu."""
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(ch for ch in decomposed if not unicodedata.combining(ch)))


def _match_case(original: str, word: str) -> str:
    if original.isupper() and len(original) > 1:
        return word.upper()
    if original[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


STRONG_VOTES = 4  # số cách đọc đồng ý đủ để nhận một tên/viết tắt chưa gặp ở đâu khác trong sách ("NEJM")
NOTHING = "\x00"  # cách đọc không thấy chữ nào ở vùng đó
JUNK = "\x01"  # cách đọc chỉ ra ký tự lạ ("øƒ"): coi như bỏ phiếu trắng


def reading_key(text: str, model: Optional[VietSyllableModel] = None) -> str:
    """Khoá bỏ phiếu của một cách đọc. Cách đọc còn ký tự lạ ("0ƒ", "oƒ") hay gộp với chữ bên cạnh là phiếu trắng."""
    if text == MERGED:
        return JUNK
    if not text.strip():
        return NOTHING
    if model is not None and any(is_corrupt(check_token(word, model)) for word in text.split()):
        return JUNK
    return word_key(text) or JUNK


def decide(region: Region, model: VietSyllableModel, vocab: Counter, final: bool = False,
           accepted: Iterable[str] = ()) -> None:
    """Quyết định cho một vùng lỗi sau mỗi lượt (xem đầu mô-đun). Đặt region.status/result."""
    original_key = word_key(region.original)
    keep_original = bool(original_key) and not region.corrupt  # chữ vỡ không được tính là một phiếu
    votes = Counter(reading_key(text, model) for _, text in region.readings)
    votes.pop(JUNK, None)
    if keep_original:
        votes[original_key] += 1
    all_votes = Counter(votes)
    # Mặt chữ không cho ra chữ thay hợp lệ nào ("gid" trong câu tiếng Việt, tên lạ chỉ một phiếu) là phiếu trắng.
    texts: Dict[str, str] = {}
    for key in list(votes):
        if key == NOTHING or (keep_original and key == original_key):
            continue
        text = _best_text(region, key, model, vocab, accepted, votes=votes[key])
        if text:
            texts[key] = text
        else:
            del votes[key]
    ranked = votes.most_common()
    top, v1 = ranked[0] if ranked else (NOTHING, 0)
    v2 = ranked[1][1] if len(ranked) > 1 else 0
    clear = v1 >= 2 and v1 >= v2 + 2
    if ranked and keep_original and top == original_key:
        # Cùng mặt chữ với bản gốc: dấu lấy theo các bản đọc "vie" ("tổi" -> "tồi", "mế" -> "mẽ"; "eng" không biết dấu).
        exact = Counter(core_of(text) for lang, text in region.readings
                        if lang == "vie" and word_key(text) == top)
        best = exact.most_common(1)[0] if exact else ("", 0)
        own = exact.get(core_of(region.original), 0)
        if best[0] and best[0] != core_of(region.original) and best[1] >= 2 and best[1] >= own + 2:
            text = adapt_candidate(region, best[0], model, vocab, accepted, strong=v1 >= STRONG_VOTES)
            if text:
                region.status, region.result = "fixed", text
                return
        # Chữ in nghiêng: "eng" thấy mặt chữ khác ở cặp chữ cái hay lẫn ("bạnh"/"hanh" -> "hạnh") mà ra chữ có thật,
        # còn chữ gốc thì không có trong từ điển.
        for key, count in ranked:
            if key in texts and count >= 2 and _italic_variant(region.original, texts[key]) \
                    and not is_known_word(core_of(region.original).lower(), model):
                region.status, region.result = "fixed", texts[key]
                return
        # Chữ gốc không có trong từ điển, các bản "vie" cùng mặt chữ nhưng khác dấu ("tiển" -> "tiền"/"tiến"): chọn
        # bản có thật hợp với chữ đứng trước/sau nhất theo từ điển, khi có bằng chứng ngữ cảnh.
        if not is_known_word(core_of(region.original).lower(), model):
            before, after = core_of(region.before).lower(), core_of(region.after).lower()
            scored = []
            for variant, count in exact.items():
                low = variant.lower()
                if variant != core_of(region.original) and is_known_word(low, model):
                    support = model.bigrams.get((before, low), 0) + model.bigrams.get((low, after), 0)
                    scored.append((support, count, model.unigrams.get(low, 0), variant))
            # Chỉ khi có bằng chứng ngữ cảnh: theo độ phổ biến thì hỏng chữ hiếm mà đúng ("nhõm" -> "nhóm").
            if scored and max(scored)[0] > 0:
                text = adapt_candidate(region, max(scored)[3], model, vocab, accepted, strong=v1 >= STRONG_VOTES)
                if text:
                    region.status, region.result = "fixed", text
                    return
        # Khẳng định chữ hiếm: ít nhất 2 cách đọc lại ra đúng chữ gốc, kể cả dấu ("tiền", "tiến" không khẳng định
        # được "tiển"); khác dấu thì để bộ sửa dấu theo ngữ cảnh quyết định (bên gọi).
        same_exact = sum(1 for lang, text in region.readings
                         if core_of(text) == core_of(region.original)
                         and (lang == "vie" or core_of(region.original).isascii()))
        if v1 >= 3 and clear and same_exact >= 2:
            region.status, region.result = "confirmed", region.original
            return
    elif ranked and top == NOTHING:
        if region.corrupt and v1 >= 3 and clear:
            region.status, region.result = "dropped", ""  # vết bẩn, hoa văn, mã vạch
            return
    elif ranked:
        text = texts[top]
        if clear and (region.broken or (v1 >= 3 and _similar(region.original, text) >= 0.5)):
            region.status, region.result = "fixed", text
            return
    if not final:
        return
    if not region.broken:
        region.status, region.result = "kept", region.original  # chữ hình dạng bình thường: giữ nguyên
        return
    keys = [reading_key(text, model) for _, text in region.readings]
    if region.corrupt and keys and all(key == NOTHING for key in keys):
        region.status, region.result = "dropped", ""
        return
    # Hết cách đọc: chữ hỏng nhận cách đọc hợp lệ được nhiều phiếu nhất, khi cách đọc đó có từ 2 phiếu, hơn hẳn
    # phương án kế tiếp và cả mặt chữ gốc, chiếm ít nhất 30% số cách đọc thấy chữ, và phần lớn cách đọc không bị
    # gộp với chữ bên cạnh.
    seen = [text for _, text in region.readings if text != MERGED]
    merged = len(region.readings) - len(seen)
    others = [(key, count) for key, count in ranked if key in texts]
    if others and merged < len(seen):
        key, count = others[0]
        runner = max(others[1][1] if len(others) > 1 else 0, votes.get(original_key, 0) if keep_original else 0)
        if count >= 2 and count > runner and count >= 0.3 * len(seen):
            region.status, region.result = "fixed", texts[key]
            return
    if keep_original and votes.get(original_key, 0) >= 3 and sum(
            1 for _, text in region.readings if core_of(text) == core_of(region.original)) >= 2:
        region.status, region.result = "confirmed", region.original  # mã chữ+số đọc giống nhau ("G7")
        return
    # Bảo đảm cuối cùng: không để lại chuỗi ký tự vô nghĩa. Cách đọc hợp lệ có từ 2 phiếu (tên lạ từ 3 phiếu) thì
    # lấy cách được nhiều phiếu nhất ("guessed", báo lại để soát); không có thì bỏ chuỗi vỡ/chữ cái lẻ không còn
    # mảnh chữ có thật nào ("removed"); còn lại giữ nguyên và báo ("unresolved").
    if not others:
        for key, count in all_votes.most_common():
            if key != NOTHING and count >= 3 and not (keep_original and key == original_key):
                text = _best_text(region, key, model, vocab, accepted, votes=STRONG_VOTES)
                if text:
                    others = [(key, count)]
                    texts[key] = text
                    break
    if others:
        key, count = others[0]
        langs = {"vie" if lang == "vie" else "eng" for lang, text in region.readings
                 if text != MERGED and word_key(text) == key}
        # Đoán chỉ khi cả mô hình tiếng Việt lẫn tiếng Anh cùng đọc ra mặt chữ đó (hai nguồn độc lập).
        if count >= 2 and langs == {"vie", "eng"} and (count >= 3 or len(core_of(texts[key])) > 1):
            region.status, region.result = "guessed", texts[key]
            return
    if (region.corrupt or all(r == "stray" for r in region.reasons)) and not _has_word(region.original, model, vocab) \
            and not re.search(r"\d{3,}|[/_]", region.original):  # số dài, mảnh đường dẫn: giữ lại
        region.status, region.result = "removed", ""
        return
    region.status, region.result = "unresolved", None


def rescan(jobs: Iterable[LineJob], recognize: Recognizer, model: VietSyllableModel, vocab: Counter,
           accepted: Iterable[str] = (), passes: Sequence[Sequence[Strategy]] = PASSES,
           executor=None, progress: Optional[Callable[[int, int, int], None]] = None
           ) -> Tuple[List[LineJob], Dict[str, int]]:
    """Quét lại theo từng lượt cho tới khi mọi vùng lỗi được giải quyết hoặc hết cách đọc.

    jobs có thể là một bộ sinh: lượt đầu gửi từng lô đi đọc ngay khi đủ dòng, nên việc cắt ảnh trang (bên gọi)
    chạy song song với việc đọc. executor: ThreadPoolExecutor để đọc song song (None: đọc tuần tự).
    progress(lượt, số lượt, số vùng còn lại). Trả về (các dòng, số vùng theo kết quả).
    """
    accepted = set(accepted)

    def read(batch: List[LineJob], strategy: Strategy):
        try:
            return read_batch(batch, strategy, recognize)
        except Exception as exc:  # một cách đọc hỏng thì bỏ qua cách đó
            print(f"[OCR] quét lại ({strategy.name}) lỗi: {exc}")
            return None

    def submit(batch: List[LineJob], strategies: Sequence[Strategy], tasks: list) -> None:
        for strategy in strategies:
            if executor is not None:
                tasks.append((batch, strategy, executor.submit(read, batch, strategy)))
            else:
                tasks.append((batch, strategy, read(batch, strategy)))

    def collect(tasks: list) -> None:
        for batch, strategy, outcome in tasks:
            words_by_line = outcome.result() if executor is not None else outcome
            if words_by_line is None:
                continue
            for job, words in zip(batch, words_by_line):
                for region in job.regions:
                    if not region.status:
                        region.readings.append((strategy.lang, region_text(words, region, job.image.height - 2 * _PAD)))

    def batches_of(pending: Iterable[LineJob]):
        """Lô dòng cùng độ phân giải, mỗi lô tối đa BATCH_LINES dòng; trả về ngay khi một lô đầy."""
        open_batches: Dict[int, List[LineJob]] = {}
        for job in pending:
            batch = open_batches.setdefault(job.dpi, [])
            batch.append(job)
            if len(batch) >= BATCH_LINES:
                yield open_batches.pop(job.dpi)
        yield from (batch for batch in open_batches.values() if batch)

    all_jobs: List[LineJob] = []

    def remember(source: Iterable[LineJob]):
        for job in source:
            all_jobs.append(job)
            yield job

    for number, strategies in enumerate(passes, start=1):
        if number == 1:
            pending: Iterable[LineJob] = remember(jobs)
        else:
            pending = [job for job in all_jobs if any(not r.status for r in job.regions)]
            if progress:
                progress(number, len(passes), sum(1 for job in pending for r in job.regions if not r.status))
            if not pending:
                break
        tasks: list = []
        for batch in batches_of(pending):
            submit(batch, strategies, tasks)
        if number == 1 and progress:
            progress(1, len(passes), sum(len(job.regions) for job in all_jobs))
        collect(tasks)
        final = number == len(passes)
        for job in all_jobs:
            for region in job.regions:
                if not region.status:
                    decide(region, model, vocab, final=final, accepted=accepted)
    for job in all_jobs:
        for region in job.regions:
            if not region.status:
                decide(region, model, vocab, final=True, accepted=accepted)
    return all_jobs, dict(Counter(region.status for job in all_jobs for region in job.regions))


def line_job(page_image: Image.Image, line_box: Tuple[float, float, float, float], dpi: int,
             tokens: Sequence[str], boxes: Sequence[Tuple[float, float, float, float]], reasons: Dict[int, str],
             key: object = None) -> LineJob:
    """Cắt dòng (pixel trên ảnh trang) kèm lề, đổi toạ độ chữ lỗi sang toạ độ ảnh cắt."""
    x0, y0, x1, y1 = line_box
    x0 = min([x0] + [b[0] for b in boxes])
    x1 = max([x1] + [b[2] for b in boxes])
    crop_box = (max(0, int(x0) - _PAD), max(0, int(y0) - _PAD),
                min(page_image.width, int(x1) + _PAD), min(page_image.height, int(y1) + _PAD))
    image = page_image.crop(crop_box)
    shifted = [(b[0] - crop_box[0], b[1] - crop_box[1], b[2] - crop_box[0], b[3] - crop_box[1]) for b in boxes]
    return LineJob(image, dpi, find_regions(tokens, shifted, reasons), key)


def find_regions(tokens: Sequence[str], boxes: Sequence[Tuple[float, float, float, float]],
                 reasons: Dict[int, str]) -> List[Region]:
    """Gộp các chữ lỗi liền nhau trên dòng thành vùng (toạ độ x của chữ, cùng hệ với boxes)."""
    regions: List[Region] = []
    for index in sorted(reasons):
        x0, _, x1, _ = boxes[index]
        if regions and regions[-1].last == index - 1:
            region = regions[-1]
            region.last = index
            region.tokens.append(tokens[index])
            region.reasons.append(reasons[index])
            region.x1 = max(region.x1, x1)
        else:
            regions.append(Region(index, index, [tokens[index]], [reasons[index]], x0, x1,
                                  before=tokens[index - 1] if index > 0 else ""))
    for region in regions:
        region.after = tokens[region.last + 1] if region.last + 1 < len(tokens) else ""
    return regions

"""Đọc lại những dòng OCR bị vỡ chữ (chữ in nghiêng, tên tiếng Anh) rồi chọn cách đọc tốt nhất cho từng chữ.

Mô hình "vie" của Tesseract đọc chữ in nghiêng gần như vô nghĩa ("USA Today" -> "US⁄1 Toảzy", "Washington
Post" -> "J2sb7ze#øn Pos") và đọc tiếng Anh theo thói quen tiếng Việt ("the" -> "tbe", "Harvard" -> "Haruard").
Đo trên sách thật: cắt riêng dòng đó ra và đọc lại thì
- mô hình "eng" đọc đúng chữ nghiêng tiếng Anh ("Washington Post", "Seattle Times");
- mô hình "vie" trên ảnh đã nắn thẳng chữ nghiêng đọc đúng phần lớn chữ nghiêng tiếng Việt;
- mô hình "vie" trên riêng một dòng đọc đúng hơn cả trang ở nhiều chỗ ("lôi" -> "Tôi").
Mỗi chữ của dòng gốc được so với các cách đọc lại nằm cùng vị trí; chỉ thay khi cách đọc mới là chữ có thật
(âm tiết tiếng Việt có trong từ điển, từ tiếng Anh có trong từ điển Anh-Việt) còn chữ gốc thì vỡ.

Các dòng cần đọc lại của một trang được ghép thành một ảnh để mỗi cách đọc chỉ gọi Tesseract một lần.
"""
from __future__ import annotations

import difflib
import re
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from app.config import settings
from app.services.ebook.vi_spell import VietSyllableModel, is_vietnamese_syllable_shape

SHEAR = 0.25  # chữ nghiêng của sách in lệch khoảng 12-15 độ
_PAD = 8
_GAP = 30

_LETTERS_VI = set("abcdefghijklmnopqrstuvwxyzăâđêôơư" "àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ")
_ALLOWED_PUNCT = set(".,;:!?…'\"“”‘’()[]-–—/%&*+=#@$|_<>")
_EDGE_PUNCT = ".,;:!?…'\"“”‘’()[]-–—*"
_ORDINAL = re.compile(r"^\d+(?:st|nd|rd|th|s|D|B|%|k|K|m|M)?$|^[A-Z]{1,4}\d{1,4}$|^\d{1,4}[A-Z]{1,3}$")
_NUMBERISH = re.compile(r"^[\d.,%/:–-]+$")
_CODE = re.compile(r"^[A-Za-z]+-\d+$|^\d+-[A-Za-z]+$")  # "COVID-19", "3-D"

_english: Optional[set] = None
_english_lock = threading.Lock()


def english_words() -> set:
    """Từ tiếng Anh một chữ (chữ thường) lấy từ mục từ của từ điển StarDict Anh-Việt; rỗng nếu không có."""
    global _english
    with _english_lock:
        if _english is not None:
            return _english
        words: set = set()
        directory = Path(settings.DICTIONARY_DIR)
        for idx in sorted(directory.glob("*.idx")) if directory.exists() else []:
            try:
                raw = idx.read_bytes()
            except OSError:
                continue
            position = 0
            while position < len(raw):
                end = raw.find(b"\0", position)
                if end < 0:
                    break
                word = raw[position:end].decode("utf-8", "replace").strip().lower()
                if word.isalpha() and word.isascii():
                    words.add(word)
                position = end + 9
        _english = words
        return words


def _english_forms(word: str) -> List[str]:
    out = [word]
    for suffix, repl in (("ies", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"), ("ing", ""), ("ing", "e"),
                         ("ly", ""), ("er", ""), ("ers", ""), ("est", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            out.append(word[: -len(suffix)] + repl)
    if word.endswith("'s") or word.endswith("’s"):
        out.append(word[:-2])
    return out


def is_english_word(word: str) -> bool:
    words = english_words()
    low = word.lower()
    return bool(words) and any(form in words for form in _english_forms(low))


def core_of(token: str) -> str:
    return token.strip(_EDGE_PUNCT)


def is_garbled(token: str) -> bool:
    """Chữ OCR vỡ: có ký tự lạ ("Toảzy" thì không, "US⁄1", "J2sb7ze#øn", "7Zø" thì có) hoặc chữ lẫn số."""
    core = core_of(token)
    if not core:
        return False
    if _NUMBERISH.match(core) or _ORDINAL.match(core) or _CODE.match(core):
        return False
    if re.match(r"^(?:https?://|www\.)", core, re.I) or re.search(r"\w\.(?:com|org|net|edu|gov)\b", core, re.I):
        return False
    for ch in core:
        low = ch.lower()
        if low in _LETTERS_VI or ch.isdigit() or ch in _ALLOWED_PUNCT:
            continue
        return True  # ø ¿ ⁄ ƒ Œ Š Ä ï ¡ » £ ...
    letters = sum(1 for ch in core if ch.isalpha())
    digits = sum(1 for ch in core if ch.isdigit())
    if letters and digits:
        return True  # "J2sb7ze", "toá72", "D4z7zø"
    if letters and re.search(r"\w[,;:!?]+\w", core):
        return True  # "Neumansk,cgacy"
    if "/" in core and letters and not re.search(r"\w/\w+/|^\w+/\w+$", core):
        return True  # "cø/", "/Z/2ø"
    return False


@dataclass
class Word:
    text: str
    box: Tuple[int, int, int, int]
    conf: float
    alternatives: tuple = ()


def _skeleton(token: str) -> str:
    text = unicodedata.normalize("NFD", core_of(token).lower().replace("đ", "d"))
    return "".join(ch for ch in text if "a" <= ch <= "z")


def _similar(a: str, b: str) -> float:
    x, y = _skeleton(a), _skeleton(b)
    if not x or not y:
        return 0.0
    return difflib.SequenceMatcher(None, x, y, autojunk=False).ratio()


def word_quality(token: str, model: VietSyllableModel, lang: str) -> float:
    """Điểm "có thật" của một chữ: âm tiết tiếng Việt quen thuộc, từ tiếng Anh, hay chuỗi vỡ."""
    core = core_of(token)
    if not core:
        return 0.0
    if is_garbled(token):
        return -3.0
    low = unicodedata.normalize("NFC", core.lower())
    if low.isdigit():
        return 1.0
    parts = [p for p in re.split(r"[-–/']", low) if p]
    score = 0.0
    for part in parts:
        if is_vietnamese_syllable_shape(part) and model.unigrams.get(part, 0) >= 3:
            score += 2.0 if lang == "vie" or not part.isascii() else 1.2
        elif part.isascii() and is_english_word(part):
            score += 2.0 if lang == "eng" or len(part) >= 4 else 1.0
        elif part.isascii() and core[:1].isupper() and len(part) >= 3:
            score += 0.5  # tên riêng chưa biết
        else:
            score -= 1.0
    return score / max(1, len(parts))


def suspicious_line(words: Sequence[str], confidences: Sequence[float], model: VietSyllableModel) -> bool:
    """Dòng cần đọc lại: có chữ vỡ, hoặc có chữ tiếng Anh mà mô hình tiếng Việt đọc sai ("tbe", "Haruard")."""
    for token, conf in zip(words, confidences):
        if is_garbled(token):
            return True
        core = core_of(token)
        low = core.lower()
        if not low or not low.isalpha():
            continue
        if low.isascii() and len(low) >= 3 and not is_vietnamese_syllable_shape(low):
            if not is_english_word(low) and not core[:1].isupper() and conf < 0.9:
                return True  # "tbe", "diuided", "pouer"
        if is_vietnamese_syllable_shape(low) and model.unigrams.get(low, 0) <= 1:
            if conf < 0.6:
                return True
            if low.isascii() and len(low) >= 3 and not is_english_word(low):
                return True  # "Reuieu", "Healtp": tiếng Anh đọc lệch mà vẫn trông như âm tiết
    return False


def shear(image: Image.Image, k: float) -> Image.Image:
    """Nắn thẳng chữ nghiêng sang phải: dịch mỗi hàng điểm ảnh sang trái tỷ lệ với độ cao."""
    width, height = image.size
    extra = int(abs(k) * height) + 2
    return image.transform((width + extra, height), Image.AFFINE, (1, -k, k * height, 0, 1, 0),
                           resample=Image.BICUBIC, fillcolor=255)


@dataclass
class _Placed:
    box: Tuple[int, int, int, int]  # dòng trên trang (pixel)
    top: int  # vị trí trên ảnh ghép


def compose(page: Image.Image, boxes: Sequence[Tuple[int, int, int, int]]) -> Tuple[Image.Image, List[_Placed]]:
    """Ghép các dòng cần đọc lại thành một cột ảnh, mỗi dòng cách nhau đủ xa để Tesseract không gộp dòng."""
    crops = []
    for x0, y0, x1, y1 in boxes:
        box = (max(0, x0 - _PAD), max(0, y0 - _PAD), min(page.width, x1 + _PAD), min(page.height, y1 + _PAD))
        crops.append((box, page.crop(box)))
    width = max(c.width for _, c in crops) + 2 * _GAP
    height = sum(c.height + _GAP for _, c in crops) + _GAP
    canvas = Image.new("L", (width, height), 255)
    placed: List[_Placed] = []
    top = _GAP
    for box, crop in crops:
        canvas.paste(crop, (_GAP, top))
        placed.append(_Placed(box, top))
        top += crop.height + _GAP
    return canvas, placed


def _line_of(placed: Sequence[_Placed], y_center: float) -> Optional[int]:
    for index, item in enumerate(placed):
        height = item.box[3] - item.box[1]
        if item.top <= y_center <= item.top + height:
            return index
    return None


def read_back(result_lines, placed: Sequence[_Placed], sheared: float = 0.0, canvas_height: int = 0
              ) -> Dict[int, List[Word]]:
    """Chữ của một lần đọc ảnh ghép, chia về từng dòng gốc với toạ độ trên trang."""
    out: Dict[int, List[Word]] = {}
    for line in result_lines:
        alternatives = line.word_alternatives or [()] * len(line.word_boxes)
        for text, box, conf, alts in zip(line.text.split(), line.word_boxes, line.word_confidences, alternatives):
            x0, y0, x1, y1 = box
            yc = (y0 + y1) / 2
            index = _line_of(placed, yc)
            if index is None:
                continue
            item = placed[index]
            # Ảnh đã nắn: điểm (x, y) trên ảnh nắn ứng với x + k*(H - y) trên ảnh gốc.
            shift = sheared * (canvas_height - yc) if sheared else 0.0
            px0 = x0 + shift - _GAP + item.box[0]
            px1 = x1 + shift - _GAP + item.box[0]
            out.setdefault(index, []).append(
                Word(text, (int(px0), item.box[1], int(px1), item.box[3]), conf, alts if not sheared else ()))
    for words in out.values():
        words.sort(key=lambda w: w.box[0])
    return out


def _overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    inter = min(a[2], b[2]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    return inter / max(1, min(a[2] - a[0], b[2] - b[0]))


_VOWELS = set("aeiouy")
# Thanh (huyền, sắc, ngã, hỏi, nặng) và dấu mũ/trăng/móc dạng tổ hợp (NFD).
_MARKS = {chr(c) for c in (0x300, 0x301, 0x303, 0x309, 0x323, 0x302, 0x306, 0x31B)}


def fuse(vie_text: str, eng_text: str, model: VietSyllableModel) -> Optional[str]:
    """Mặt chữ của bản "eng" (đọc chữ nghiêng đúng: "hoc") + dấu của bản "vie" ("bọc") -> "học".

    Chỉ trả về chữ có thật trong từ điển. Không ghép được thì chọn, trong các chữ cùng mặt chữ với bản "eng",
    chữ gần dấu bản "vie" nhất (hòa thì chữ phổ biến hơn): "uể" + "vé" -> "về".
    """
    from app.services.ebook.vi_spell import mark_distance

    vie, eng = core_of(vie_text), core_of(eng_text)
    # Bản "eng" đôi khi tự thêm dấu Latin ("vé"): chỉ lấy mặt chữ.
    eng = "".join(ch for ch in unicodedata.normalize("NFD", eng) if not unicodedata.combining(ch))
    if not vie or not eng or len(vie) != len(eng) or not eng.isascii() or not eng.isalpha():
        return None
    pieces = []
    for v, e in zip(vie.lower(), eng.lower()):
        decomposed = unicodedata.normalize("NFD", v)
        marks = "".join(ch for ch in decomposed if ch in _MARKS)
        if e == "d" and v == "đ":
            pieces.append("đ")
        elif e in _VOWELS:
            pieces.append(unicodedata.normalize("NFC", e + marks))
        else:
            pieces.append(e)
    fused = "".join(pieces)
    if is_vietnamese_syllable_shape(fused) and model.unigrams.get(fused, 0) >= 3:
        return fused
    best, best_key = None, None
    base = eng.lower()
    variants = _variants_by_skeleton(model).get(base, ())
    for word in variants:
        if len(word) != len(fused):
            continue
        key = (mark_distance(fused, word), -model.unigrams.get(word, 0))
        if best_key is None or key < best_key:
            best, best_key = word, key
    if best is not None and best_key[0] <= 1:
        return best
    return None


_variant_index: Dict[int, Dict[str, List[str]]] = {}


def _variants_by_skeleton(model: VietSyllableModel) -> Dict[str, List[str]]:
    from app.services.ebook.vi_spell import skeleton

    key = id(model)
    if key not in _variant_index:
        index: Dict[str, List[str]] = {}
        for word, count in model.unigrams.items():
            if count >= 3:
                index.setdefault(skeleton(word), []).append(word)
        _variant_index.clear()
        _variant_index[key] = index
    return _variant_index[key]


# Chữ nghiêng hay bị đọc lẫn giữa các cặp chữ cái này ("học" -> "bọc", "về" -> "uể", "nhân" -> "nbÂn").
_ITALIC_CONFUSIONS = {frozenset(p) for p in ("bh", "uv", "nu", "il", "lt", "ce", "rn", "ao")}


def _italic_variant(original: str, candidate: str) -> bool:
    """Bản đọc lại chỉ khác chữ gốc ở các chữ cái hay lẫn khi in nghiêng (và dấu)."""
    a, b = _skeleton(original), _skeleton(candidate)
    if len(a) != len(b):
        return False
    return all(x == y or frozenset((x, y)) in _ITALIC_CONFUSIONS for x, y in zip(a, b))


def merge_line(original: Sequence[Word], readings: Sequence[Tuple[str, List[Word]]], model: VietSyllableModel
               ) -> Tuple[List[Word], List[bool]]:
    """Chọn cách đọc cho từng chữ của dòng gốc. Trả về (chữ, đã thay?).

    Chỉ thay chữ gốc hỏng (vỡ ký tự, hoặc không phải chữ có thật) bằng chữ đọc lại không vỡ, giống chữ gốc hoặc
    được ít nhất hai cách đọc đồng thuận. Chữ gốc có thật chỉ thay trong dòng chữ nghiêng (nhiều chữ vỡ), khi hai
    cách đọc đồng thuận ("bọc" -> "học").
    """
    garbled = [is_garbled(w.text) for w in original]
    out: List[Word] = []
    changed: List[bool] = []
    for position, word in enumerate(original):
        if not any(ch.isalnum() for ch in word.text):
            out.append(word)  # dấu câu, vạch ngăn "|" của đầu trang: giữ nguyên
            changed.append(False)
            continue
        # Chữ nằm cạnh chữ vỡ thường cùng một cụm in nghiêng ("toá72 bọc" là "toán học").
        italic_context = any(garbled[max(0, position - 1): position + 2])
        own_quality = word_quality(word.text, model, "vie")
        own_bad = is_garbled(word.text) or own_quality <= 0
        own_marked = any(ch in _LETTERS_VI and not ch.isascii() for ch in word.text.lower())
        candidates: List[Tuple[str, Word]] = []
        for lang, words in readings:
            covering = [w for w in words if _overlap(w.box, word.box) >= 0.6 and _overlap(word.box, w.box) >= 0.5]
            if len(covering) == 1:
                candidates.append((lang, covering[0]))
        if garbled[position] or italic_context:
            # Bản "vie" đọc trên ảnh đã nắn (cách đọc thứ hai) ghép với bản "eng".
            vie_reads = [c for lang, c in candidates if lang == "vie"]
            eng_reads = [c for lang, c in candidates if lang == "eng"]
            letters_differ = _skeleton(eng_reads[0].text) != _skeleton(word.text) if eng_reads else False
            if vie_reads and eng_reads and (garbled[position] or letters_differ):
                # Chỉ ghép khi bản "eng" thấy mặt chữ khác ("bọc"/"hoc"): cùng mặt chữ thì dấu của dòng gốc (chữ
                # đứng) đáng tin hơn dấu đọc trên ảnh đã nắn ("va" không thành "và").
                fused = fuse(vie_reads[-1].text, eng_reads[0].text, model)
                if fused:
                    core = core_of(eng_reads[0].text)
                    text = eng_reads[0].text.replace(core, _match_case(core, fused), 1)
                    fused_word = Word(text, eng_reads[0].box, min(vie_reads[-1].conf, eng_reads[0].conf), ())
                    candidates += [("fuse", fused_word), ("fuse", fused_word)]  # ghép từ hai cách đọc: hai phiếu
        best: Optional[Word] = None
        best_score = own_quality + 0.5 * word.conf
        for lang, cand in candidates:
            if core_of(cand.text) == core_of(word.text) or is_garbled(cand.text):
                continue
            own_core = core_of(word.text)
            if sum(ch.isdigit() for ch in own_core) * 2 >= len(own_core) > 0 and not any(ch.isdigit() for ch in cand.text):
                continue  # "ó0%" là "60%": không đổi số thành chữ
            if lang == "eng" and own_marked and not is_garbled(word.text):
                continue  # chữ gốc có dấu tiếng Việt: bản "eng" (không dấu) không thay được
            if lang == "eng" and _skeleton(cand.text) == _skeleton(word.text):
                continue  # chỉ khác dấu ("Clément" -> "Clement"): bản "eng" không biết dấu
            agree = sum(1 for _, other in candidates if _similar(other.text, cand.text) >= 0.7)
            score = word_quality(cand.text, model, "vie" if lang == "fuse" else lang) + 0.5 * cand.conf
            if lang == "fuse":
                score += 1.5  # hai mô hình khớp nhau: mặt chữ của "eng", dấu của "vie"
            english_fix = (
                lang == "eng" and own_quality < 1 and core_of(word.text).isascii()
                and is_english_word(core_of(cand.text)) and _similar(word.text, cand.text) >= 0.5
            )  # "Reuieu" -> "Review": tiếng Anh đọc lệch (tên riêng chưa biết cũng chỉ được 0,5 điểm)
            if english_fix:
                pass
            elif own_bad:
                # Âm tiết tiếng Việt lạ nhưng không vỡ ("tổi") để bộ sửa dấu lo; đọc lại chỉ thay khi hai cách đồng thuận.
                syllable = is_vietnamese_syllable_shape(core_of(word.text).lower()) and not is_garbled(word.text)
                short = len(core_of(word.text)) <= 3  # "NJ" -> "NT": chữ quá ngắn, giống nhau chẳng nói lên gì
                if agree < 2 and (syllable or short or _similar(word.text, cand.text) < 0.34):
                    continue
                if agree < 2 and len(core_of(cand.text)) < 0.6 * len(core_of(word.text)):
                    continue  # bản đọc lại chỉ khớp một mảnh ("øƒ.Persouality" -> "aality")
                if not garbled[position] and word_quality(cand.text, model, lang) < 1.0:
                    continue  # chữ gốc chỉ là chữ lạ ("blogger"): không đổi sang một tên riêng đoán mò ("Vogger")
            elif (
                word.text.isupper() or not italic_context or agree < 2 or score < best_score + 1.0
                or not _italic_variant(word.text, cand.text)
            ):
                continue
            if score > best_score:
                best, best_score = cand, score
        if best is None:
            out.append(word)
            changed.append(False)
        else:
            out.append(Word(_keep_edges(word.text, best.text), word.box, max(best.conf, 0.5), best.alternatives))
            changed.append(True)
    return out, changed


def _match_case(original: str, word: str) -> str:
    if original.isupper() and len(original) > 1:
        return word.upper()
    if original[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def _keep_edges(original: str, new: str) -> str:
    """Giữ dấu câu hai đầu của chữ gốc khi bản đọc lại làm mất ("Bowl”" -> "Bowl")."""
    if new == original:
        return new
    lead = original[: len(original) - len(original.lstrip(_EDGE_PUNCT))]
    trail = original[len(original.rstrip(_EDGE_PUNCT)):]
    if lead and not new.startswith(lead):
        new = lead + new.lstrip(_EDGE_PUNCT)
    if trail and not new.endswith(trail):
        new = new.rstrip(_EDGE_PUNCT) + trail
    return new


Recognizer = Callable[[Image.Image, str, int], object]
# Các cách đọc lại: "vie" trên ảnh đã nắn thẳng chữ nghiêng và "eng".
READINGS = (("vie", SHEAR), ("eng", 0.0))


def repair_page_lines(page_image: Image.Image, lines: Sequence, recognize: Recognizer, dpi: int,
                      model: Optional[VietSyllableModel] = None) -> int:
    """Đọc lại các dòng vỡ của một trang OCR (OcrLine) và sửa tại chỗ. Trả về số chữ đã thay."""
    model = model or VietSyllableModel.shared()
    todo = [line for line in lines
            if line.word_boxes and len(line.word_boxes) == len(line.text.split())
            and suspicious_line(line.text.split(), line.word_confidences or [1.0] * len(line.word_boxes), model)]
    if not todo:
        return 0
    canvas, placed = compose(page_image, [line.bbox for line in todo])
    readings_by_line: Dict[int, List[Tuple[str, List[Word]]]] = {i: [] for i in range(len(todo))}
    for lang, k in READINGS:
        image = shear(canvas, k) if k else canvas
        try:
            result = recognize(image, lang, dpi)
        except Exception:
            continue
        for index, words in read_back(result.lines, placed, sheared=k, canvas_height=canvas.height).items():
            readings_by_line[index].append((lang, words))
    replaced = 0
    for index, line in enumerate(todo):
        alternatives = line.word_alternatives or [()] * len(line.word_boxes)
        original = [Word(t, b, c, a) for t, b, c, a in
                    zip(line.text.split(), line.word_boxes, line.word_confidences, alternatives)]
        words, changed = merge_line(original, readings_by_line[index], model)
        if any(changed):
            replaced += sum(changed)
            line.text = " ".join(w.text for w in words)
            line.word_confidences = [w.conf for w in words]
            if line.word_alternatives:
                # Chữ thay mới mang phương án thay thế của lần đọc lại (độ dài phải khớp chữ, nếu không thì bỏ).
                line.word_alternatives = [w.alternatives if len(w.alternatives) == len(w.text) else ()
                                          for w in words]
    return replaced


# ---- Tên riêng / chữ nước ngoài thống nhất theo cả cuốn sách -------------------------------------------

# Cặp ký tự OCR hay nhầm trong chữ Latin. Chữ hiếm chỉ được coi là đọc sai của chữ hay gặp khi hai chữ khác nhau
# đúng ở những cặp này (tên thật khác nhau một chữ cái như "Joel"/"Joe", "Davis"/"David" thì không đổi).
_LATIN_CONFUSIONS = {frozenset(p) for p in ("ce", "mn", "rt", "il", "it", "uv", "hb", "]l", "|l", "|i")}
_LATIN_TOKEN = re.compile(r"^[A-Za-z][A-Za-z'’\]|]*[A-Za-z\]|]$")


def _confusable(x: str, y: str) -> bool:
    return frozenset((x.lower(), y.lower())) in _LATIN_CONFUSIONS


def _latin_distance(a: str, b: str, limit: float) -> float:
    """Khoảng cách chỉ gồm các nhầm lẫn OCR: thay cặp hay nhầm 0,5; thừa/thiếu một ký tự hay nhầm với ký tự
    đứng cạnh ("Anmnie"/"Annie", "Washingtron"/"Washington") 0,75. Khác kiểu khác thì coi như không giống."""
    inf = limit + 1
    if abs(len(a) - len(b)) > 1:
        return inf
    rows = [[inf] * (len(b) + 1) for _ in range(len(a) + 1)]
    rows[0][0] = 0.0
    for i in range(len(a) + 1):
        for j in range(len(b) + 1):
            here = rows[i][j]
            if here > limit:
                continue
            if i < len(a) and j < len(b):
                if a[i] == b[j]:
                    rows[i + 1][j + 1] = min(rows[i + 1][j + 1], here)
                elif _confusable(a[i], b[j]):
                    rows[i + 1][j + 1] = min(rows[i + 1][j + 1], here + 0.5)
            if i < len(a):  # a thừa ký tự a[i]
                near = [b[k] for k in (j - 1, j) if 0 <= k < len(b)]
                if any(_confusable(a[i], c) for c in near):  # chữ cái đôi ("Garry"/"Gary") là tên khác, không tính
                    rows[i + 1][j] = min(rows[i + 1][j], here + 0.75)
            if j < len(b):  # a thiếu ký tự b[j]
                near = [a[k] for k in (i - 1, i) if 0 <= k < len(a)]
                if any(_confusable(b[j], c) for c in near):
                    rows[i][j + 1] = min(rows[i][j + 1], here + 0.75)
    return rows[len(a)][len(b)]


def unify_rare_names(texts: Sequence[str], model: Optional[VietSyllableModel] = None) -> Dict[str, str]:
    """Bảng thay thế cho chữ Latin hiếm (1-2 lần) gần giống một chữ Latin hay gặp trong sách.

    "Scahawks" (2 lần) -> "Seahawks" (30 lần), "Amnie" -> "Annie", "Ietlock" -> "Tetlock", "Bow]" -> "Bowl".
    Bỏ qua chữ có thật trong từ điển (Anh hoặc Việt) và chữ quá ngắn.
    """
    model = model or VietSyllableModel.shared()
    counts: Dict[str, int] = {}
    brackets: Dict[str, str] = {}
    for text in texts:
        for token in text.split():
            match = re.match(r"^\W*([A-Za-z]{2,})[\]|](?=\W*$)", token)
            if match:
                brackets[match.group(1) + token[match.end(1)]] = match.group(1) + "l"  # "Bow]" là "Bowl"
            core = core_of(token).rstrip(".,;:")
            if len(core) >= 3 and _LATIN_TOKEN.match(core) and "'" not in core and "’" not in core:
                counts[core] = counts.get(core, 0) + 1
    frequent: Dict[int, List[str]] = {}
    for word, count in counts.items():
        if count >= 3 and "]" not in word and "|" not in word:
            frequent.setdefault(len(word), []).append(word)
    mapping: Dict[str, str] = {}
    for raw, fixed in brackets.items():
        if counts.get(fixed, 0) >= 3:  # "Hai]" trong "(tháng Mười Hai]" không phải "Hail"
            mapping[raw] = fixed
    for word, count in counts.items():
        if count > 3 or len(word) < 4:
            continue
        low = word.lower()
        clean = low.replace("]", "l").replace("|", "l")
        if clean.isalpha() and (is_english_word(clean) or model.unigrams.get(clean, 0) >= 3):
            if "]" not in word and "|" not in word:
                continue  # chữ có thật ("Hall", "using")
        limit = 1.0
        best, best_key = None, None
        for length in (len(word) - 1, len(word), len(word) + 1):
            for target in frequent.get(length, ()):
                if counts[target] < 3 * count:
                    continue
                if target[:1].isupper() != word[:1].isupper() and word[:1] not in "Il":
                    continue
                distance = _latin_distance(word, target if word[:1].islower() or target[:1].isupper()
                                           else target[:1].upper() + target[1:], limit)
                if distance <= limit:
                    key = (distance, -counts[target])
                    if best_key is None or key < best_key:
                        best, best_key = target, key
        if best and best != word:
            if word[:1].isupper() and best[:1].islower():
                best = best[:1].upper() + best[1:]  # "Iheo" đầu câu -> "Theo"
            mapping[word] = best
    return mapping


def apply_name_mapping(text: str, mapping: Dict[str, str]) -> str:
    if not mapping:
        return text
    out = []
    for token in re.split(r"(\s+)", text):
        core = core_of(token).rstrip(".,;:")
        bracket = re.match(r"^\W*([A-Za-z]{2,}[\]|])", token)
        if core in mapping:
            token = token.replace(core, mapping[core], 1)
        elif bracket and bracket.group(1) in mapping:
            token = token.replace(bracket.group(1), mapping[bracket.group(1)], 1)
        out.append(token)
    return "".join(out)


# ---- Chữ hoa lớn đầu chương (drop cap) -------------------------------------------------------------------

def restore_drop_caps(page_image: Image.Image, lines: Sequence, recognize: Recognizer, dpi: int,
                      model: Optional[VietSyllableModel] = None) -> int:
    """Chữ hoa lớn chiếm 2-3 dòng đầu chương ("N" + "ăm 26 tuổi") thường bị Tesseract bỏ qua.

    Nhận ra qua bố cục: dòng mở đầu bằng chữ thường, dòng ngay dưới thụt vào bằng đúng chỗ đó, rồi dòng sau nữa
    trở về lề. Cắt riêng khoảng trống bên trái hai dòng đầu, đọc một ký tự (psm 10) và chỉ ghép lại khi được
    một chữ có thật ("Năm"). Trả về số chữ đã khôi phục.
    """
    model = model or VietSyllableModel.shared()
    ordered = sorted((l for l in lines if l.text.strip()), key=lambda l: (l.bbox[1], l.bbox[0]))
    restored = 0
    for index in range(len(ordered) - 2):
        first, second, third = ordered[index], ordered[index + 1], ordered[index + 2]
        height = max(1.0, first.x_size)
        token = first.text.split()[0]
        if not token[:1].isalpha() or not token[:1].islower():
            continue
        if abs(second.bbox[0] - first.bbox[0]) > 0.4 * height or first.bbox[0] - third.bbox[0] < 1.2 * height:
            continue
        if second.bbox[1] - first.bbox[3] > 1.5 * height or third.bbox[1] - second.bbox[3] > 1.5 * height:
            continue
        if any(o is not first and o.bbox[2] <= first.bbox[0] and o.bbox[3] > first.bbox[1] and o.bbox[1] < first.bbox[3]
               for o in ordered):
            continue  # có chữ khác bên trái dòng đầu: không phải chỗ trống của chữ hoa lớn
        box = (int(third.bbox[0]) - 4, int(first.bbox[1]) - int(0.6 * height),
               int(first.bbox[0]) - 2, int(second.bbox[3]) + 4)
        if box[2] - box[0] < 0.6 * height:
            continue
        crop = page_image.crop(box)
        pad = Image.new("L", (crop.width + 40, crop.height + 40), 255)
        pad.paste(crop, (20, 20))
        try:
            result = recognize(pad, "vie", dpi)
        except Exception:
            continue
        letter = "".join(result.text.split())
        if len(letter) != 1 or not letter.isalpha() or not letter.isupper():
            continue
        word = core_of(letter + token).lower()
        if model.unigrams.get(word, 0) < 3 and not is_english_word(word):
            continue
        first.text = letter + first.text
        if first.word_alternatives:
            first.word_alternatives[0] = ()  # chữ đầu đã đổi: phương án thay thế cũ không còn khớp
        restored += 1
    return restored


# ---- Số bị đọc lẫn chữ ----------------------------------------------------------------------------------

_NUMBER_TOKEN = re.compile(r"^(\W*?)([\dóÓ]+)(%?[^\w]*)$")


def fix_number_token(token: str) -> str:
    """Chữ số 6 phông sách hay bị đọc thành "ó": "201ó" -> "2016", "9ó%" -> "96%", "ó0%" -> "60%"."""
    match = _NUMBER_TOKEN.match(token)
    if not match:
        return token
    body = match.group(2)
    digits = sum(ch.isdigit() for ch in body)
    if digits == len(body) or not (digits >= 2 or (digits >= 1 and "%" in match.group(3))):
        return token
    return match.group(1) + body.replace("ó", "6").replace("Ó", "6") + match.group(3)

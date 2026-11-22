"""Sửa lỗi dấu tiếng Việt sau OCR bằng mô hình ngôn ngữ âm tiết.

Tesseract đọc đúng mặt chữ nhưng hay nhầm dấu: "kẻ"->"kể", "rằng"->"răng", "trước đây"->"trước dây",
"cái bẫy"->"cái bãy" (đo trên sách scan thật: khoảng 1 từ trên 100-150 từ). Từ đúng gần như luôn là một biến thể
cùng mặt chữ, khác dấu. Bộ sửa xét các biến thể đó và chọn theo ngữ cảnh (mô hình "kênh nhiễu"):

    điểm(c) = log P(c | từ trước) + log P(từ sau | c) - phạt(số dấu khác với chữ OCR)

Tần suất lấy từ hai nguồn có sẵn trên máy: phần nghĩa tiếng Việt của từ điển Anh-Việt StarDict
(data/dictionaries/en-vi, ~5 triệu âm tiết) và chính cuốn sách đang OCR (đa số chỗ trong sách đã đọc đúng).
Chỉ sửa khi chênh lệch điểm đủ lớn, để không "sửa" nhầm chữ hiếm mà đúng. Không có từ điển thì chỉ dùng
thống kê của chính cuốn sách.
"""
from __future__ import annotations

import gzip
import itertools
import math
import pickle
import re
import threading
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from app.config import settings

_WORD_RE = re.compile(r"[^\W\d_]+")
# Chữ cái tiếng Việt (không có f, j, w, z): chỉ những âm tiết như vậy mới đem ra sửa dấu.
_VI_LETTERS = set("abcdeghiklmnopqrstuvxyăâđêôơư" "àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ")
_VI_MARKED = set("ăâđêôơư" "àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ")
_INNER_HYPHEN = re.compile(r"(?<=\w)-(?=\w)")
_SENTENCE_BREAK = re.compile(r"[.!?…:;]|[“\"(\[]\s*$")
_STRONG_BREAK = re.compile(r"[.!?…]")
_LINKISH = re.compile(r"/|@|www|\w\.\w|^https?|^htt?p", re.I)  # đường dẫn, email, tên miền: không phải chữ thường  # hết câu thật sự (sau ":" ";" câu vẫn có thể viết thường)
_CACHE_VERSION = 4

# Phạt (đơn vị log tự nhiên) cho mỗi dấu khác giữa chữ OCR và chữ sửa; ngưỡng chênh lệch điểm để được sửa.
MARK_COST = 4.0
MARGIN_KNOWN = 2.5  # chữ OCR là âm tiết có thật: cần ngữ cảnh rất rõ mới sửa
MARGIN_UNKNOWN = 0.0  # chữ OCR không có trong kho chữ: gần như chắc là sai
MARGIN_KNOWN_IMAGE = 2.0  # chữ OCR có thật nhưng chính ảnh cũng gợi ý bản sửa (phương án thay thế của Tesseract)
SENTENCE_START_EXTRA = 1.5
DOCUMENT_WEIGHT = 3.0
LOWERCASE_T_COST = 1.5  # câu không mở đầu bằng chữ thường: "lôi" đầu câu gần như chắc là "Tôi"
# Phương án thay thế từng chữ cái mà LSTM của Tesseract đã cân nhắc (lstm_choice_mode=2): "tiển" có "ề" 64 điểm
# bên cạnh "ể" 94 điểm nghĩa là chính ảnh cũng gợi ý "tiền". Chi phí đổi sang phương án đó tính theo khoảng
# cách điểm, thay cho chi phí cố định MARK_COST theo số dấu.
IMAGE_COST_BASE = 0.5
IMAGE_COST_SCALE = 15.0
IMAGE_MIN_CHOICE = 15.0
IMAGE_CONTRADICTION_COST = 1.5


def confidence_adjustment(confidence: Optional[float]) -> Tuple[float, float]:
    """(hệ số phạt, cộng thêm vào ngưỡng) theo độ tin cậy Tesseract của từ.

    Đo trên sách thật: từ sai được chấm 66-92 điểm, trung vị cả trang 96, chỉ ~5% số từ dưới 90.
    """
    if confidence is None:
        return 1.0, 0.0
    if confidence >= 0.95:
        return 1.6, 1.0
    if confidence >= 0.90:
        return 1.0, 0.0
    if confidence >= 0.80:
        return 0.7, -0.7
    return 0.45, -1.2  # cuốn sách ~100-200 nghìn âm tiết, từ điển ~5 triệu: cân lại cho chữ trong sách có tiếng nói
BIGRAM_WEIGHT = 0.7


def skeleton(word: str) -> str:
    """Mặt chữ không dấu: "Đường" -> "duong"."""
    word = unicodedata.normalize("NFD", word.lower().replace("đ", "d").replace("Đ", "d"))
    return "".join(ch for ch in word if not unicodedata.combining(ch))


_TONE_MARKS = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}  # huyền, sắc, ngã, hỏi, nặng
_SHAPE_MARKS = {"\u0302", "\u0306", "\u031b"}  # mũ (â ê ô), trăng (ă), móc (ơ ư)


def _analyse(word: str) -> Tuple[str, List[str]]:
    """(thanh của âm tiết, dấu mũ/móc/đ theo từng chữ cái)."""
    tone = ""
    shapes = []
    for ch in word.lower():
        if ch == "đ":
            shapes.append("đ")
            continue
        decomposed = unicodedata.normalize("NFD", ch)
        tone += "".join(c for c in decomposed if c in _TONE_MARKS)
        shapes.append("".join(sorted(c for c in decomposed if c in _SHAPE_MARKS)))
    return tone, shapes


def mark_distance(a: str, b: str) -> int:
    """Số lần sửa dấu: đổi thanh tính 1 ("ổ"->"ồ"), thêm/bớt/đổi một dấu mũ/móc tính 1, d<->đ tính 1."""
    tone_a, shapes_a = _analyse(a)
    tone_b, shapes_b = _analyse(b)
    if len(shapes_a) != len(shapes_b):
        return max(len(shapes_a), len(shapes_b))
    distance = 0 if tone_a == tone_b else 1
    distance += sum(1 for x, y in zip(shapes_a, shapes_b) if x != y)
    return distance


# Mô hình lỗi OCR (đo trên trang scan có đáp án): Tesseract hay bỏ sót thanh, nhầm thanh, thêm/bớt dấu mũ và
# gạch ngang chữ đ; nhưng ít khi "thấy" một dấu thanh không có thật. Bỏ dấu thanh đã thấy vì vậy đắt hơn.
_TONE_REMOVE_WEIGHT = 1.3


def edit_cost(observed: str, candidate: str) -> float:
    """Chi phí (theo số dấu, có trọng số) để chữ đúng `candidate` bị OCR đọc thành `observed`."""
    tone_o, shapes_o = _analyse(observed)
    tone_c, shapes_c = _analyse(candidate)
    if len(shapes_o) != len(shapes_c):
        return float(max(len(shapes_o), len(shapes_c)))
    cost = 0.0
    if tone_o != tone_c:
        cost += _TONE_REMOVE_WEIGHT if tone_o and not tone_c else 1.0
    cost += sum(1.0 for x, y in zip(shapes_o, shapes_c) if x != y)
    return cost


def is_vietnamese_syllable_shape(word: str) -> bool:
    w = word.lower()
    return 1 <= len(w) <= 7 and all(ch in _VI_LETTERS for ch in w)


def _match_case(original: str, candidate: str) -> str:
    if original.isupper() and len(original) > 1:
        return candidate.upper()
    if original[:1].isupper():
        return candidate[:1].upper() + candidate[1:]
    return candidate


class VietSyllableModel:
    """Tần suất âm tiết và cặp âm tiết tiếng Việt (chữ thường)."""

    def __init__(self, unigrams: Counter, bigrams: Counter) -> None:
        self.unigrams = unigrams
        self.bigrams = bigrams
        self.total = max(1, sum(unigrams.values()))

    _shared: Optional["VietSyllableModel"] = None
    _lock = threading.Lock()

    @classmethod
    def shared(cls) -> "VietSyllableModel":
        with cls._lock:
            if cls._shared is None:
                cls._shared = cls._load_or_build(Path(settings.DICTIONARY_DIR))
            return cls._shared

    @classmethod
    def _load_or_build(cls, directory: Path) -> "VietSyllableModel":
        data_files = sorted(directory.glob("*.dict.dz")) if directory.exists() else []
        if not data_files:
            return cls(Counter(), Counter())
        source = data_files[0]
        stamp = f"{_CACHE_VERSION}-{source.stat().st_size}-{int(source.stat().st_mtime)}"
        cache = Path(settings.DATA_DIR) / "cache" / "vi_syllable_model.pkl"
        try:
            if cache.exists():
                saved = pickle.loads(cache.read_bytes())
                if saved.get("stamp") == stamp:
                    return cls(saved["unigrams"], saved["bigrams"])
        except Exception:
            pass
        unigrams, bigrams = cls._count_dictionary(source)
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(pickle.dumps({"stamp": stamp, "unigrams": unigrams, "bigrams": bigrams}, protocol=4))
        except OSError:
            pass
        return cls(unigrams, bigrams)

    @staticmethod
    def _count_dictionary(source: Path) -> Tuple[Counter, Counter]:
        """Đếm âm tiết trong các dòng nghĩa/ví dụ tiếng Việt của từ điển StarDict."""
        text = gzip.open(source, "rb").read().decode("utf-8", "replace")
        unigrams: Counter = Counter()
        bigrams: Counter = Counter()
        for line in text.splitlines():
            line = line.strip()
            if not line or line[0] not in "-=+!":
                continue
            low = unicodedata.normalize("NFC", line.lower())
            if not any(ch in _VI_MARKED for ch in low):
                continue
            low = _INNER_HYPHEN.sub(" ", low)
            for segment in re.split(r"[^\w\s]+", low):
                tokens = [t for t in _WORD_RE.findall(segment) if is_vietnamese_syllable_shape(t)]
                unigrams.update(tokens)
                bigrams.update(zip(tokens, tokens[1:]))
        # Bỏ cặp chỉ gặp một lần cho gọn bộ nhớ, trừ cặp có chữ hiếm: đó là bằng chứng duy nhất cho chữ hiếm
        # ("tép riu", "vỏn vẹn").
        bigrams = Counter({
            k: v for k, v in bigrams.items()
            if v >= 2 or min(unigrams[k[0]], unigrams[k[1]]) <= 50
        })
        return unigrams, bigrams


class OcrSpellCorrector:
    def __init__(self, model: VietSyllableModel, document_texts: Iterable[str],
                 confident_words: Optional[Iterable[str]] = None) -> None:
        """confident_words: chữ (thường) mà cả cuốn sách dùng >= 3 lần với độ tin cậy OCR trung bình rất cao."""
        self.model = model
        self.confident_words = set(confident_words or ())
        self.doc_unigrams: Counter = Counter()
        self.doc_bigrams: Counter = Counter()
        for text in document_texts:
            for segment in re.split(r"[^\w\s]+", unicodedata.normalize("NFC", text.lower())):
                tokens = [t for t in _WORD_RE.findall(segment) if is_vietnamese_syllable_shape(t)]
                self.doc_unigrams.update(tokens)
                self.doc_bigrams.update(zip(tokens, tokens[1:]))
        self.doc_total = max(1, sum(self.doc_unigrams.values()))
        self.vocab_size = max(1, len(set(model.unigrams) | set(self.doc_unigrams)))
        # Chỉ mục mặt chữ -> các biến thể có thật (từ điển >= 3 lần, hoặc sách >= 3 lần).
        self.variants: Dict[str, List[str]] = {}
        for word, count in model.unigrams.items():
            if count >= 3:
                self.variants.setdefault(skeleton(word), []).append(word)
        for word, count in self.doc_unigrams.items():
            if count >= 3 and model.unigrams.get(word, 0) < 3:
                self.variants.setdefault(skeleton(word), []).append(word)
        self.known_words = {word for words in self.variants.values() for word in words}
        self.changes: Counter = Counter()
        self._untrusted: set = set()

    # ---- Xác suất ---------------------------------------------------------------------------

    @staticmethod
    def _discount(count: int) -> float:
        # Chữ gặp một lần trong sách không được tính: lỗi OCR thường lẻ tẻ, chữ đúng thì lặp lại.
        return max(0.0, count - 1.5)

    def _count(self, word: str) -> float:
        if word in self._untrusted:
            return float(self.model.unigrams.get(word, 0))
        return self.model.unigrams.get(word, 0) + DOCUMENT_WEIGHT * self._discount(self.doc_unigrams.get(word, 0))

    def _p_uni(self, word: str) -> float:
        total = self.model.total + DOCUMENT_WEIGHT * self.doc_total
        return (self._count(word) + 0.5) / (total + 0.5 * self.vocab_size)

    def _p_bi(self, prev: str, word: str) -> float:
        prev_count = self._count(prev)
        doc_pair = 0.0 if (prev in self._untrusted or word in self._untrusted) else self._discount(self.doc_bigrams.get((prev, word), 0))
        pair = self.model.bigrams.get((prev, word), 0) + DOCUMENT_WEIGHT * doc_pair
        ml = pair / prev_count if prev_count > 0 else 0.0
        return BIGRAM_WEIGHT * ml + (1 - BIGRAM_WEIGHT) * self._p_uni(word)

    def _support(self, prev: Optional[str], word: str, nxt: Optional[str]) -> float:
        """Bằng chứng cặp từ ở hai bên của một chữ, chỉ lấy từ từ điển: lỗi OCR lặp lại có hệ thống trong sách
        ("Tôi" đầu câu thường thành "lôi") không được tự làm bằng chứng cho chính nó."""
        total = 0.0
        for pair in ((prev, word), (word, nxt)):
            if pair[0] and pair[1]:
                total += self.model.bigrams.get(pair, 0)
        return total

    def _support_with_neighbour_variants(self, prev: Optional[str], word: str, nxt: Optional[str]) -> float:
        """Như _support, nhưng chữ bên cạnh được phép mang dấu khác (chính nó có thể đang sai: "vỏn vẻn")."""
        best = 0.0
        for left in ([prev] + self.variants.get(skeleton(prev), []) if prev else [None]):
            for right in ([nxt] + self.variants.get(skeleton(nxt), []) if nxt else [None]):
                best = max(best, self._support(left, word, right))
        return best

    def _dictionary_prefers_variant(self, prev: Optional[str], low: str, nxt: Optional[str],
                                    candidates: Sequence[Tuple[str, float]]) -> bool:
        """Từ điển nghiêng hẳn về một bản sửa ở ít nhất một phía và không phía nào nghiêng về chữ OCR.

        Xét từng phía riêng: "chỉ tiết" (vài lần, lỗi gõ) so với "chi tiết" (hàng trăm lần) thì phía phải chọn bản
        sửa; còn "các ván bài": phía trái "các vấn" phổ biến hơn nhưng phía phải chỉ có "ván bài", nên không tính.
        """
        bigrams = self.model.bigrams
        for candidate, _ in candidates:
            prefers_candidate = prefers_own = False
            for own_pair, cand_pair in (((prev, low), (prev, candidate)), ((low, nxt), (candidate, nxt))):
                if not own_pair[0] or not own_pair[1]:
                    continue
                own, alt = bigrams.get(own_pair, 0), bigrams.get(cand_pair, 0)
                if own <= 2 and alt >= max(3, 20 * own):
                    prefers_candidate = True  # cặp của chữ OCR gần như không có (vài lần là lỗi gõ của từ điển)
                if own >= 2 and own >= 3 * alt:
                    prefers_own = True
                if alt == 0 and self.doc_bigrams.get(own_pair, 0) >= 3 and not self.doc_bigrams.get(cand_pair, 0):
                    prefers_own = True  # cặp riêng của sách ("ván poker"): từ điển không biết cả hai
            if prefers_candidate and not prefers_own:
                return True
        return False

    def _book_spelling(self, word: str) -> str:
        """Cùng chữ nhưng theo kiểu đặt dấu sách đang dùng ("hoá" -> "hóa" nếu sách viết "hóa")."""
        best, best_count = word, self.doc_unigrams.get(word, 0)
        for variant in self.variants.get(skeleton(word), ()):
            if len(variant) == len(word) and mark_distance(word, variant) == 0:
                count = self.doc_unigrams.get(variant, 0)
                if count > best_count:
                    best, best_count = variant, count
        return best

    def _score(self, prev: Optional[str], word: str, nxt: Optional[str]) -> float:
        score = math.log(self._p_bi(prev, word)) if prev else math.log(self._p_uni(word))
        if nxt:
            score += math.log(self._p_bi(word, nxt))
        return score

    # ---- Sửa --------------------------------------------------------------------------------

    def _candidates(self, low: str, sentence_start: bool, heading: bool, lowercase: bool = True) -> List[Tuple[str, float]]:
        out: Dict[str, float] = {}
        for variant in self.variants.get(skeleton(low), ()):
            if variant != low and len(variant) == len(low):
                distance = mark_distance(low, variant)
                if distance:  # khác chỗ đặt dấu thanh ("hóa"/"hoá") là chính tả tương đương: không đổi
                    out[variant] = MARK_COST * edit_cost(low, variant)
        if heading and lowercase and "u" in skeleton(low):
            # Phông tiêu đề chữ "a" thường một tầng: Tesseract đọc thành "u" ("lù một vún" -> "là một ván").
            # Chữ hoa "A" không bị nhầm nên chữ viết hoa ("MỤC LỤC") không xét.
            base = skeleton(low)
            for index, ch in enumerate(base):
                if ch == "u":
                    for variant in self.variants.get(base[:index] + "a" + base[index + 1:], ()):
                        if len(variant) == len(low):
                            out.setdefault(variant, MARK_COST * (1 + mark_distance(low[:index] + low[index + 1:], variant[:index] + variant[index + 1:])))
        if sentence_start and low.startswith("l"):
            # Chữ "T" hoa đầu câu bị đọc thành "l" ("“lôi không chắc" -> "“Tôi không chắc").
            for variant in self.variants.get("t" + skeleton(low)[1:], ()):
                if len(variant) == len(low):
                    out.setdefault(variant, LOWERCASE_T_COST + MARK_COST * mark_distance(low[1:], variant[1:]))
        return list(out.items())

    @staticmethod
    def _contradiction(low: str, candidate: str, alternatives: Optional[Sequence[tuple]]) -> float:
        """Chi phí thêm khi bản sửa đổi một chữ cái sang chữ mà Tesseract đã cân nhắc các phương án khác nhưng
        không có nó ("bãy": Tesseract phân vân "ẫ"/"ấ", tức là chắc chắn có dấu, nên "bay" không dấu bị phạt)."""
        if not alternatives or len(alternatives) != len(low) or len(candidate) != len(low):
            return 0.0
        extra = 0.0
        for index, (old, new) in enumerate(zip(low, candidate)):
            if old == new:
                continue
            choices = {unicodedata.normalize("NFC", ch.lower()) for ch, _ in alternatives[index][1]}
            if choices and new not in choices:
                extra += IMAGE_CONTRADICTION_COST
        return extra

    def _image_candidates(self, low: str, alternatives: Optional[Sequence[tuple]]) -> Dict[str, float]:
        """Chữ có thật ghép được từ phương án thay thế của Tesseract (đổi tối đa 2 chữ cái), kèm chi phí."""
        if not alternatives or len(alternatives) != len(low):
            return {}
        options: List[List[Tuple[str, float]]] = []
        for index, (own, choices) in enumerate(alternatives):
            opts = []
            for char, conf in choices:
                char = unicodedata.normalize("NFC", char.lower())
                if len(char) != 1 or char == low[index] or not char.isalpha() or conf < IMAGE_MIN_CHOICE:
                    continue
                opts.append((char, IMAGE_COST_BASE + max(0.0, own - conf) / IMAGE_COST_SCALE))
            options.append(opts)
        positions = [i for i, opts in enumerate(options) if opts]
        out: Dict[str, float] = {}
        for i in positions:
            for char, cost in options[i]:
                cand = low[:i] + char + low[i + 1:]
                out[cand] = min(out.get(cand, cost), cost)
        for i, j in itertools.combinations(positions, 2):
            for (ci, cost_i), (cj, cost_j) in itertools.product(options[i], options[j]):
                cand = low[:i] + ci + low[i + 1:j] + cj + low[j + 1:]
                out[cand] = min(out.get(cand, cost_i + cost_j), cost_i + cost_j)
        return {cand: cost for cand, cost in out.items() if cand in self.known_words and cand != low}

    def correct_tokens(self, text: str, heading: bool = False, confidences: Optional[Sequence[float]] = None,
                       alternatives: Optional[Sequence[tuple]] = None) -> str:
        """Sửa một đoạn chữ OCR (giữ nguyên khoảng trắng, dấu câu, kiểu chữ hoa/thường).

        confidences: độ tin cậy Tesseract theo từng từ cách nhau bởi khoảng trắng trong text (có thể bỏ trống).
        alternatives: phương án thay thế từng chữ cái của các từ đó (xem OcrLine.word_alternatives).
        """
        matches = list(_WORD_RE.finditer(text))
        if not matches:
            return text
        word_conf: List[Optional[float]] = [None] * len(matches)
        word_alts: List[Optional[tuple]] = [None] * len(matches)
        in_link = [False] * len(matches)
        chunks = list(re.finditer(r"\S+", text))
        if confidences is not None and len(confidences) != len(chunks):
            confidences = None
        if alternatives is not None and len(alternatives) != len(chunks):
            alternatives = None
        chunk_index = 0
        for index, match in enumerate(matches):
            while chunk_index < len(chunks) and chunks[chunk_index].end() <= match.start():
                chunk_index += 1
            if chunk_index >= len(chunks):
                break
            chunk = chunks[chunk_index]
            if _LINKISH.search(chunk.group()):
                in_link[index] = True  # chữ trong đường dẫn/tên miền: không phải âm tiết
            if confidences is not None:
                word_conf[index] = confidences[chunk_index]
            chunk_alts = alternatives[chunk_index] if alternatives is not None else None
            if chunk_alts and len(chunk_alts) == len(chunk.group()):
                offset = match.start() - chunk.start()
                word_alts[index] = tuple(chunk_alts[offset: offset + len(match.group())])
        words = [m.group() for m in matches]
        lows = [unicodedata.normalize("NFC", w.lower()) for w in words]
        gaps = [text[: matches[0].start()]] + [
            text[matches[i - 1].end(): matches[i].start()] for i in range(1, len(matches))
        ]
        result = list(words)

        def suspicion(index: int) -> Tuple[int, float]:
            unknown = self.model.unigrams.get(lows[index], 0) <= 1
            conf = word_conf[index] if word_conf[index] is not None else 0.92
            return (0 if unknown else 1, conf)

        # Chữ đáng ngờ nhất (không có trong từ điển, độ tin cậy thấp) được sửa trước, để khi xét chữ bên cạnh
        # thì ngữ cảnh đã đúng ("vỏn vẻn": sửa "vẻn"->"vẹn" trước, chứ không sửa nhầm "vỏn"->"vốn").
        for i in sorted(range(len(words)), key=suspicion):
            word, low = words[i], lows[i]
            if not is_vietnamese_syllable_shape(low) or in_link[i]:
                continue
            acronym = word.isupper() and len(word) <= 4 and not any(ch in _VI_MARKED for ch in low)
            image = {} if acronym else self._image_candidates(low, word_alts[i])
            if low in self.confident_words and not image:
                continue
            start, end = matches[i].start(), matches[i].end()
            if (start and text[start - 1].isdigit()) or (end < len(text) and text[end].isdigit()):
                continue  # mảnh chữ dính số ("L0ẠI", "KHÔN6"): OCR đọc O/G thành số, không phải âm tiết
            # Viết hoa giữa câu ("Pete", "Sài Gòn") là tên riêng: không sửa dấu, trừ chữ "l" đầu câu.
            sentence_start = i == 0 or bool(_SENTENCE_BREAK.search(gaps[i]))
            if word[:1].isupper() and not sentence_start and not word.isupper():
                continue
            prev = lows[i - 1] if i > 0 and gaps[i].strip() == "" else None
            nxt = lows[i + 1] if i + 1 < len(lows) and gaps[i + 1].strip() == "" else None
            if prev is not None and not is_vietnamese_syllable_shape(prev):
                prev = None
            if nxt is not None and not is_vietnamese_syllable_shape(nxt):
                nxt = None
            candidates = self._candidates(low, sentence_start, heading, lowercase=not word.isupper())
            listed = {cand for cand, _ in candidates}
            candidates += [(cand, math.inf) for cand in image if cand not in listed]
            if not candidates:
                continue
            cost_factor, margin_extra = confidence_adjustment(word_conf[i])
            # Chữ gần như không có trong từ điển (<= 1 lần / 5 triệu âm tiết) là lỗi OCR dù lặp lại trong sách:
            # thống kê của sách không được bênh nó ("bãy" 8 lần, "bẫy" 2 lần do phông chữ).
            if self.model.unigrams.get(low, 0) <= 1 and self.model.total > 100_000:
                self._untrusted = {low}
            elif self.model.total > 100_000 and self._dictionary_prefers_variant(prev, low, nxt, candidates):
                # "khả di" lặp lại trong sách (do phông chữ) nhưng từ điển chỉ có "khả dĩ".
                self._untrusted = {low}
            base_score = self._score(prev, low, nxt)
            self._untrusted = set()
            best, best_score = None, base_score
            # Câu không mở đầu bằng chữ thường: ở đầu câu, chữ OCR viết thường ("lồi") gần như chắc là chữ hoa đọc
            # sai ("Tôi"), nên bản sửa vẫn để chữ thường đầu câu ("lời") bị phạt thêm.
            lowercase_at_start = sentence_start and word[:1].islower() and low.startswith("l")
            # Ngay sau dấu chấm câu, chữ "l" thường chỉ có thể là chữ hoa đọc sai ("T"): chữ hoa "L" thì OCR đọc
            # đúng, nên bản sửa vẫn bắt đầu bằng "l" ("lồi" -> "lời") gần như chắc chắn sai.
            strong_start = lowercase_at_start and i > 0 and bool(_STRONG_BREAK.search(gaps[i]))
            best_from_image = False
            for candidate, cost in candidates:
                # Bản sửa ít gặp trong từ điển mà sách lại dùng nhiều thường chính là một lỗi phông lặp lại ("bẫy"
                # đọc thành "bấy" ở nhiều chỗ): thống kê của sách không được ủng hộ nó.
                doc_trusted = self.model.unigrams.get(candidate, 0) >= 50 or candidate in self.confident_words
                self._untrusted = set() if doc_trusted else {candidate}
                if lowercase_at_start and not candidate.startswith("t"):
                    cost += 4.0 if strong_start else 0.5
                effective = cost * cost_factor + self._contradiction(low, candidate, word_alts[i])
                from_image = candidate in image and image[candidate] < effective
                if from_image:
                    effective = image[candidate]  # chính ảnh đã gợi ý chữ này: không nhân hệ số độ tin cậy
                score = self._score(prev, candidate, nxt) - effective
                if score > best_score:
                    best, best_score, best_from_image = candidate, score, from_image
            self._untrusted = set()
            if best is None:
                continue
            # "Có thật" theo từ điển, không theo sách (lỗi OCR lặp lại trong sách không làm chữ sai thành có thật).
            known = self.model.unigrams.get(low, 0) >= 3
            if strong_start and best.startswith("t"):
                known = False  # "lồi" viết thường ngay sau dấu chấm câu thì chắc chắn đã đọc sai, dù "lồi" có thật
            margin = (MARGIN_KNOWN_IMAGE if best_from_image else MARGIN_KNOWN) if known else MARGIN_UNKNOWN
            lowercase_start = sentence_start and low.startswith("l") and word[:1].islower() and best.startswith("t")
            if prev is None and not lowercase_start and not (best_from_image and sentence_start):
                margin += SENTENCE_START_EXTRA  # thiếu ngữ cảnh bên trái (chữ thường đầu câu thì tự nó đã bất thường)
            margin = max(0.0, margin + margin_extra)
            if best_from_image:
                # Tesseract chấm phương án thay thế cao hơn chữ nó đã chọn ("đ" 82,8 / "d" 79,9) và chữ OCR hiếm gặp:
                # chỉ cần ngữ cảnh không phản đối. Còn lại thì cần từ điển ủng hộ bản sửa hơn chữ OCR.
                strong_image = image.get(best, math.inf) <= IMAGE_COST_BASE + 0.05 and self.model.unigrams.get(low, 0) < 50
                if known and not strong_image and self._support(prev, best, nxt) <= self._support(prev, low, nxt):
                    continue  # ảnh gợi ý chữ khác nhưng từ điển không ủng hộ hơn chữ OCR
            elif known and self._support(prev, best, nxt) < 2 * self._support(prev, low, nxt) + 2:
                continue  # chữ OCR có thật: chỉ sửa khi bản sửa có bằng chứng cặp từ mạnh hơn hẳn
            if (
                known and prev and self.model.bigrams.get((prev, best), 0) == 0
                and self.doc_bigrams.get((prev, low), 0) >= 2
            ):
                continue  # sách lặp lại "quyết định tồi" mà từ điển không có "định tôi": giữ chữ của sách
            sides = [
                (self.model.bigrams.get(own, 0), self.model.bigrams.get(alt, 0), self.doc_bigrams.get(own, 0))
                for own, alt in (((prev, low), (prev, best)), ((low, nxt), (best, nxt))) if own[0] and own[1]
            ]
            if known and any(
                book >= 2 and own >= 1 and 10 * own >= alt for own, alt, book in sides
            ) and not any(alt >= max(3, 20 * own) for own, alt, _ in sides):
                # Sách dùng lặp lại một cặp từ có trong từ điển ("một kẻ", "các ván"): chữ của sách đúng. Trừ khi
                # từ điển nghiêng hẳn về bản sửa ("chỉ tiết" 3 lần, "chi tiết" hàng nghìn lần: lỗi phông lặp lại).
                continue
            if not known and not strong_start and self._support_with_neighbour_variants(prev, low, nxt) >= 1:
                continue  # từ điển có sẵn cặp từ với chữ OCR ("tép riu", "vỏn vẹn"): chữ hiếm nhưng đúng
            if not known and (prev or nxt) and self._support_with_neighbour_variants(prev, best, nxt) < 1:
                continue  # chữ lạ, nhưng bản sửa cũng không có cặp từ nào làm chứng ("tép riu" -> "tép rìu"): giữ nguyên
            if best_score - base_score > margin:
                best = self._book_spelling(best)
                fixed = _match_case(word, best)
                if sentence_start and low.startswith("l") and best.startswith("t"):
                    fixed = fixed[:1].upper() + fixed[1:]
                result[i] = fixed
                lows[i] = best
                self.changes[(low, best)] += 1
        out = []
        for gap, word in zip(gaps, result):
            out.append(gap)
            out.append(word)
        out.append(text[matches[-1].end():])
        return "".join(out)


def lowercase_stray_capitals(text: str, model: VietSyllableModel) -> str:
    """OCR hay trả chữ HOA lẻ giữa câu thường ("có VỊ đắng", "VIỆC"): đưa về chữ thường.

    Chỉ áp cho 1-3 từ viết hoa toàn bộ, nằm giữa hai từ chữ thường, và là âm tiết tiếng Việt quen
    thuộc (có dấu, hoặc rất phổ biến), để không đụng tới từ viết tắt như "NFL", "TV".
    """
    matches = list(_WORD_RE.finditer(text))
    if len(matches) < 3:
        return text
    words = [m.group() for m in matches]
    caps = [len(w) >= 2 and w.isupper() for w in words]
    if sum(caps) > 0.5 * len(words):
        return text  # dòng tiêu đề viết hoa
    out = list(words)
    i = 0
    while i < len(words):
        if not caps[i]:
            i += 1
            continue
        j = i
        while j < len(words) and caps[j]:
            j += 1
        before = words[i - 1] if i > 0 else ""
        after = words[j] if j < len(words) else ""
        if j - i <= 3 and before[:1].islower() and after[:1].islower():
            for k in range(i, j):
                low = words[k].lower()
                common = model.unigrams.get(low, 0) >= 200
                if is_vietnamese_syllable_shape(low) and (any(ch in _VI_MARKED for ch in low) or common):
                    out[k] = low
        i = j
    pieces = []
    last = 0
    for m, word in zip(matches, out):
        pieces.append(text[last:m.start()])
        pieces.append(word)
        last = m.end()
    pieces.append(text[last:])
    return "".join(pieces)

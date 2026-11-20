"""Đọc bố cục trang PDF và dựng lại đoạn văn cho sách reflow (EPUB/MOBI).

PyMuPDF trả về "khối" chữ theo khoảng cách trên trang, không theo đoạn văn: sách dàn trang thụt lề dòng đầu,
không cách đoạn (đa số sách in) cho cả trang vào một khối; đoạn vắt qua trang bị cắt đôi; chữ cuối dòng bị
ngắt bằng gạch nối; đầu trang/chân trang và số trang nằm chen giữa văn bản. Mô-đun này đọc tới từng dòng để:

- tách đoạn theo thụt lề dòng đầu, dòng cuối ngắn và khoảng cách dòng;
- nối từ bị ngắt cuối dòng theo từ điển lấy từ chính cuốn sách ("invest-|ment" -> "investment",
  "self-|discipline" giữ gạch nối);
- bỏ đầu trang/chân trang lặp lại (kể cả loại có số trang thay đổi) và số trang;
- nối đoạn bị cắt khi sang trang hoặc sang cột;
- giữ căn giữa/căn phải, thơ, dấu ngắt cảnh, chữ nghiêng/đậm một phần đoạn và chỉ số chú thích;
- lấy ảnh minh họa (bỏ logo lặp lại, nền cả trang), bìa sách và mục lục (outline) của PDF.

Kết quả là danh sách trang/khối cùng định dạng StructureBuilder đang dùng, để chia chương như luồng dịch.
"""
from __future__ import annotations

import hashlib
import io
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image

try:
    import pymupdf
except ImportError:  # pragma: no cover
    pymupdf = None

from app.services.ebook.devices import DeviceProfile, get_device
from app.services.ebook.text_cleanup import (
    MARK_CLOSE,
    MARK_OPEN,
    STYLE_ORDER,
    clean_span_text,
    detect_language,
    finalize_text,
    strip_markers,
)
from app.services.extractor.ocr_engine import OCREngine
from app.services.extractor.scan_detector import PageType, ScanDetector
from app.services.reconstruction.page_break_joiner import continues_sentence

ProgressFn = Callable[[float, str], None]

# Cờ mặc định của get_text("dict") trừ giữ chữ ghép: để MuPDF tự tách chữ ghép (U+FB01) thành "fi".
_TEXT_FLAGS = (pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_LIGATURES) if pymupdf else 0
_BOLD_FONT = re.compile(r"bold|black|heavy|semibold|demibold|extrabold|ultrabold", re.I)
_ITALIC_FONT = re.compile(r"italic|oblique", re.I)
_WORD_RE = re.compile(r"[^\W\d_]+")
_HYPHEN_COMPOUND_RE = re.compile(r"([^\W\d_]+)-([^\W\d_]+)")
_LINE_HYPHEN_RE = re.compile(r"([^\W\d_]+)-$")
_SENTENCE_END_RE = re.compile(r"[.!?…:;][\"'’”»)\]*†‡]*\d{0,3}$")
_PAGE_NUMBER_RE = re.compile(r"^(?:(?:page|trang)\s*)?(?:#|[ivxlcdm]{1,7})$|^[-–—\s]*#[-–—\s]*$|^#\s*(?:of|/|trên)\s*#$", re.I)
_SCENE_BREAK_RE = re.compile(r"^[\s*•·~#⁂◆◇✦✧❖○●□■=_—–-]{1,24}$")
_SUP_TEXT_RE = re.compile(r"^[\d*†‡§,\-–\s]{1,6}$")
_FOOTNOTE_START_RE = re.compile(r"^(?:[\x01\x02][bis])*\s*(?:\d{1,3}|[*†‡§])")
_JUNK_TITLE_RE = re.compile(r"^(?:untitled|document\d*|microsoft word\s*-?.*|\s*)$", re.I)


# ---------------------------------------------------------------------------------------------
# Cấu trúc dữ liệu
# ---------------------------------------------------------------------------------------------

@dataclass
class Span:
    text: str
    size: float
    bold: bool = False
    italic: bool = False
    sup: bool = False


@dataclass
class Line:
    spans: List[Span]
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans)

    @property
    def size(self) -> float:
        weights: Counter = Counter()
        for s in self.spans:
            weights[round(s.size, 1)] += len(s.text.strip()) or 0
        if not weights or not sum(weights.values()):
            return self.spans[0].size if self.spans else 0.0
        return weights.most_common(1)[0][0]

    @property
    def all_sup(self) -> bool:
        visible = [s for s in self.spans if s.text.strip()]
        return bool(visible) and all(s.sup for s in visible)


@dataclass
class TextBlock:
    lines: List[Line]
    ocr: bool = False

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (
            min(l.x0 for l in self.lines), min(l.y0 for l in self.lines),
            max(l.x1 for l in self.lines), max(l.y1 for l in self.lines),
        )

    @property
    def text(self) -> str:
        return " ".join(l.text.strip() for l in self.lines)

    @property
    def size(self) -> float:
        weights: Counter = Counter()
        for l in self.lines:
            for s in l.spans:
                weights[round(s.size, 1)] += len(s.text.strip())
        return weights.most_common(1)[0][0] if weights else 0.0


@dataclass
class ImageItem:
    path: Path
    bbox: Tuple[float, float, float, float]
    width: int
    height: int
    digest: str
    scan: bool = False  # ảnh chụp cả trang scan (không có OCR)


@dataclass
class PageLayout:
    number: int
    width: float
    height: float
    items: list = field(default_factory=list)
    char_count: int = 0
    scanned: bool = False
    ocr_used: bool = False


@dataclass
class _Metrics:
    left: float
    right: float


@dataclass
class _Style:
    indent: bool = False
    indent_em: float = 1.2
    justify: bool = True


class _Para:
    """Chữ của một đoạn dưới dạng các mảnh [chữ, tập kiểu chữ]."""

    def __init__(self) -> None:
        self.segs: List[list] = []

    def visible(self) -> str:
        return "".join(t for t, _ in self.segs)

    def _push(self, text: str, styles: frozenset) -> None:
        if not text:
            return
        if self.segs and (self.segs[-1][1] == styles or not text.strip()):
            self.segs[-1][0] += text
        elif self.segs and not self.segs[-1][0].strip():
            self.segs[-1][0] += text
            self.segs[-1][1] = styles
        else:
            self.segs.append([text, styles])

    def append_spans(self, spans: Sequence[Span], lstrip: bool = False) -> None:
        for index, span in enumerate(spans):
            text = span.text.lstrip() if (lstrip and index == 0) else span.text
            styles = set()
            if span.bold:
                styles.add("b")
            if span.italic:
                styles.add("i")
            if span.sup and _SUP_TEXT_RE.match(span.text.strip() or "x"):
                styles.add("s")
            self._push(text, frozenset(styles))

    def _boundary(self, next_text: str, vocab, hyphen_vocab, tight: bool = False) -> None:
        """Nối phần cuối đoạn với dòng/đoạn tiếp theo: thêm khoảng trắng hoặc xử lý gạch nối cuối dòng."""
        if not self.segs:
            return
        last = self.segs[-1]
        tail = last[0].rstrip()
        if tight:
            last[0] = tail
            return
        if tail.endswith("\u00ad"):
            last[0] = tail[:-1]
            return
        match = _LINE_HYPHEN_RE.search(tail)
        if match and next_text[:1].islower():
            head = _WORD_RE.match(next_text)
            if _keep_hyphen(match.group(1), head.group() if head else "", vocab, hyphen_vocab):
                last[0] = tail
            else:
                last[0] = tail[:-1]
            return
        if re.search(r"\d[-–]$", tail) and next_text[:1].isdigit():
            last[0] = tail
            return
        if tail.endswith(("—", "–", "/")) and not tail.endswith((" —", " –")):
            last[0] = tail
            return
        last[0] = tail + " "

    def add_line(self, line: Line, vocab, hyphen_vocab, hard_break: bool = False, tight: bool = False) -> None:
        if not self.segs:
            self.append_spans(line.spans, lstrip=True)
            return
        if hard_break:
            self.segs[-1][0] = self.segs[-1][0].rstrip() + "\n"
        else:
            self._boundary(line.text.lstrip(), vocab, hyphen_vocab, tight=tight)
        self.append_spans(line.spans, lstrip=True)

    def extend(self, other: "_Para", vocab, hyphen_vocab) -> None:
        if not other.segs:
            return
        self._boundary(other.visible().lstrip(), vocab, hyphen_vocab)
        first = True
        for text, styles in other.segs:
            self._push(text.lstrip() if first else text, styles)
            first = False

    def render(self) -> Tuple[str, float, float]:
        """Chữ có dấu kiểu chữ + tỷ lệ chữ đậm/nghiêng. Kiểu phủ cả đoạn không đánh dấu (đã nằm ở cấp khối)."""
        counts: Counter = Counter()
        total = 0
        for text, styles in self.segs:
            n = len(re.sub(r"\s", "", text))
            total += n
            for s in styles:
                counts[s] += n
        if not total:
            return "", 0.0, 0.0
        drop = {s for s in ("b", "i") if counts[s] >= 0.9 * total}
        out: List[str] = []
        opened: List[str] = []
        for text, styles in self.segs:
            want = [s for s in STYLE_ORDER if s in styles and s not in drop]
            if not text.strip():
                want = opened
            if want != opened:
                out.extend(MARK_CLOSE + s for s in reversed(opened))
                out.extend(MARK_OPEN + s for s in want)
                opened = list(want)
            out.append(text)
        out.extend(MARK_CLOSE + s for s in reversed(opened))
        rendered = "".join(out)
        # Khoảng trắng đứng ngoài thẻ: "<i> chữ</i>" -> " <i>chữ</i>".
        rendered = re.sub(r"((?:\x01[bis])+)([ \n]+)", r"\2\1", rendered)
        rendered = re.sub(r"([ \n]+)((?:\x02[bis])+)", r"\2\1", rendered)
        return finalize_text(rendered), counts["b"] / total, counts["i"] / total


@dataclass
class _Unit:
    kind: str  # "text" | "image"
    page: int
    bbox: List[float]
    segment: Tuple[int, int] = (0, 0)  # (trang, cột)
    para: Optional[_Para] = None
    size: float = 0.0
    body: bool = False
    align: str = ""
    verse: bool = False
    scene: bool = False
    indented: bool = False
    last_full: bool = False
    first_y0: float = 0.0
    last_y0: float = 0.0
    last_y1: float = 0.0
    line_count: int = 1
    quote: bool = False
    heading_level: int = 0
    heading_like: bool = False
    footnote: bool = False
    ocr: bool = False
    first_in_segment: bool = False
    last_in_segment: bool = False
    image: Optional[ImageItem] = None

    @property
    def joinable(self) -> bool:
        return (
            self.kind == "text" and self.body and not self.align and not self.verse and not self.scene
            and not self.heading_level and not self.heading_like and not self.footnote
        )


@dataclass
class PdfAnalysis:
    pages: List[dict]
    assets: List[dict]
    title: str
    author: str
    language: str
    cover: Optional[bytes]
    outline_used: bool
    paragraph_style: str
    indent_em: float
    justify: bool
    body_size: float
    page_count: int
    scanned_pages: int
    ocr_pages: int
    image_count: int
    warnings: List[str] = field(default_factory=list)


# Phần sau chỗ ngắt là hậu tố ("adult-|hood", "kind-|ness"): gần như luôn là một từ bị ngắt dòng.
_JOIN_SUFFIXES = {
    "hood", "hoods", "ship", "ships", "less", "ness", "ful", "fully", "ward", "wards", "wise", "dom", "able",
    "ably", "ible", "ment", "ments", "tion", "tions", "sion", "sions", "ing", "ings", "ly", "ity", "ities",
    "ies", "ed", "er", "ers", "est", "ism", "ist", "ists", "ize", "ized", "ise", "ised", "ous", "ive",
}


def _keep_hyphen(a: str, b: str, vocab, hyphen_vocab) -> bool:
    """Gạch nối cuối dòng là của từ ghép thật (giữ) hay chỉ là chỗ ngắt từ (bỏ)."""
    if not a or not b:
        return False
    joined = (a + b).lower()
    compound = f"{a}-{b}".lower()
    if compound in hyphen_vocab and joined not in vocab:
        return True
    if joined in vocab:
        return False
    if compound in hyphen_vocab:
        return True
    if b.lower() in _JOIN_SUFFIXES:
        return False
    return len(a) >= 2 and len(b) >= 3 and a.lower() in vocab and b.lower() in vocab


def _ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END_RE.search(strip_markers(text).rstrip()))


def _norm_title(text: str) -> str:
    text = unicodedata.normalize("NFC", strip_markers(text or "")).lower()
    return " ".join(re.sub(r"[^\w]+", " ", text).split())


def render_page_image(page, max_width: int, max_height: int, gray: bool = False) -> bytes:
    zoom = min(max_width / max(page.rect.width, 1), max_height / max(page.rect.height, 1))
    zoom = max(0.2, min(zoom, 6.0))
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(zoom, zoom), alpha=False,
        colorspace=pymupdf.csGRAY if gray else pymupdf.csRGB,
    )
    return pix.tobytes("png")


def render_pdf_cover(pdf_path: Path, device: Optional[DeviceProfile] = None, require_image: bool = False) -> Optional[bytes]:
    """Ảnh bìa từ trang 1. require_image=True: chỉ lấy khi trang 1 là ảnh bìa (không phải trang chữ)."""
    if pymupdf is None:
        return None
    device = device or get_device(None)
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception:
        return None
    try:
        if len(doc) == 0:
            return None
        page = doc[0]
        if require_image:
            area = page.rect.width * page.rect.height or 1.0
            covered = 0.0
            for info in page.get_image_info():
                r = pymupdf.Rect(info["bbox"]) & page.rect
                covered += max(0.0, r.width) * max(0.0, r.height)
            if covered / area < 0.5 or len(page.get_text().strip()) > 200:
                return None
        return render_page_image(page, device.screen_width, device.screen_height)
    except Exception:
        return None
    finally:
        doc.close()


# ---------------------------------------------------------------------------------------------
# Bộ phân tích
# ---------------------------------------------------------------------------------------------

class PdfLayoutAnalyzer:
    def __init__(
        self,
        work_dir: Path,
        device: Optional[DeviceProfile] = None,
        include_images: bool = True,
        cover: bool = True,
        language: Optional[str] = None,
        title: Optional[str] = None,
        author: Optional[str] = None,
        progress: Optional[ProgressFn] = None,
        source_name: Optional[str] = None,
    ) -> None:
        if pymupdf is None:
            raise ImportError("Cần thư viện PyMuPDF để đọc PDF.")
        self.work_dir = Path(work_dir)
        self.image_dir = self.work_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or get_device(None)
        self.include_images = include_images
        self.want_cover = cover
        self.language_override = (language or "").strip() or None
        self.title_override = (title or "").strip() or None
        self.author_override = (author or "").strip() or None
        self.source_name = (source_name or "").strip() or None
        self.progress = progress or (lambda fraction, message: None)
        self.ocr = OCREngine(lang=self._ocr_lang())
        self.body_size = 11.0
        self.pitch = 13.0
        self.vocab: set = set()
        self.hyphen_vocab: set = set()

    @staticmethod
    def _ocr_lang() -> str:
        try:
            import pytesseract

            langs = set(pytesseract.get_languages(config=""))
            return "eng+vie" if "vie" in langs else "eng"
        except Exception:
            return "eng"

    # ---- Trích xuất ---------------------------------------------------------------------------

    def _extract_image(self, page, block) -> Optional[ImageItem]:
        x0, y0, x1, y1 = block.get("bbox", (0, 0, 0, 0))
        if (x1 - x0) < 12 or (y1 - y0) < 12:
            return None
        width, height = int(block.get("width") or 0), int(block.get("height") or 0)
        if width and height and (width < 16 or height < 16):
            return None
        data = block.get("image") or b""
        ext = (block.get("ext") or "").lower()
        readable = False
        if data and ext not in ("jpx", "jb2", "jbig2"):
            try:
                with Image.open(io.BytesIO(data)) as im:
                    im.verify()
                readable = True
            except Exception:
                readable = False
        if not readable:
            # JPEG2000, JBIG2 hoặc ảnh có mặt nạ: vẽ lại đúng vùng ảnh trên trang.
            try:
                clip = pymupdf.Rect(x0, y0, x1, y1) & page.rect
                if clip.is_empty:
                    return None
                pix = page.get_pixmap(clip=clip, dpi=150, alpha=False)
                data, ext, width, height = pix.tobytes("png"), "png", pix.width, pix.height
            except Exception:
                return None
        digest = hashlib.md5(data).hexdigest()
        path = self.image_dir / f"{digest}.{ext or 'img'}"
        if not path.exists():
            path.write_bytes(data)
        return ImageItem(path, (x0, y0, x1, y1), width, height, digest)

    def _extract_page(self, page, number: int) -> PageLayout:
        rect = page.rect
        layout = PageLayout(number=number, width=rect.width, height=rect.height)
        data = page.get_text("dict", flags=_TEXT_FLAGS)
        words = 0
        image_area = 0.0
        image_count = 0
        for block in data.get("blocks", []):
            if block.get("type") == 0:
                lines = []
                for raw_line in block.get("lines", []):
                    dx, dy = raw_line.get("dir", (1.0, 0.0))
                    if abs(dy) > 0.3 or dx < 0:
                        continue  # chữ xoay dọc (ghi chú lề, chữ in chìm)
                    spans = []
                    for raw in raw_line.get("spans", []):
                        text = clean_span_text(raw.get("text", ""))
                        if not text:
                            continue
                        font = raw.get("font", "") or ""
                        flags = int(raw.get("flags", 0) or 0)
                        spans.append(Span(
                            text=text,
                            size=float(raw.get("size", 0.0) or 0.0),
                            bold=bool(flags & 16) or bool(_BOLD_FONT.search(font)),
                            italic=bool(flags & 2) or bool(_ITALIC_FONT.search(font)),
                            sup=bool(flags & 1),
                        ))
                    line_text = "".join(s.text for s in spans).strip()
                    if not line_text:
                        continue
                    x0, y0, x1, y1 = raw_line.get("bbox", (0, 0, 0, 0))
                    lines.append(Line(spans, x0, y0, x1, y1))
                    layout.char_count += len(line_text)
                    words += len(line_text.split())
                if lines:
                    layout.items.append(TextBlock(lines))
            elif block.get("type") == 1:
                image_count += 1
                r = pymupdf.Rect(block.get("bbox", (0, 0, 0, 0))) & rect
                image_area += max(0.0, r.width) * max(0.0, r.height)
                if self.include_images:
                    item = self._extract_image(page, block)
                    if item:
                        layout.items.append(item)

        page_area = rect.width * rect.height or 1.0
        page_type = ScanDetector.classify_page(layout.char_count, words, image_count, image_area / page_area, page_area)
        if page_type == PageType.SCANNED_PAGE and layout.char_count < 60:
            self._handle_scanned_page(page, layout)
        return layout

    def _handle_scanned_page(self, page, layout: PageLayout) -> None:
        if self.ocr.is_available:
            try:
                pix = page.get_pixmap(dpi=300)
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_blocks = self.ocr.ocr_image(image)
            except Exception:
                ocr_blocks = []
            if ocr_blocks:
                scale = 72.0 / 300.0
                lines = []
                for ob in ocr_blocks:
                    text = clean_span_text(ob.get("text", "")).strip()
                    bbox = ob.get("bbox") or [0, 0, 0, 0]
                    if not text:
                        continue
                    x0, y0, x1, y1 = (v * scale for v in bbox)
                    lines.append(Line([Span(text, size=max(6.0, (y1 - y0)))], x0, y0, x1, y1))
                if lines:
                    layout.items = [TextBlock(lines, ocr=True)]
                    layout.char_count = sum(len(l.text) for l in lines)
                    layout.ocr_used = True
                    return
        # Không OCR được: giữ nguyên trang dưới dạng ảnh xám vừa màn hình để không mất nội dung.
        try:
            side = self.device.image_max_side
            data = render_page_image(page, side, side, gray=True)
        except Exception:
            return
        digest = hashlib.md5(data).hexdigest()
        path = self.image_dir / f"scan_{layout.number:05d}.png"
        path.write_bytes(data)
        layout.items = [ImageItem(path, (0, 0, layout.width, layout.height), 0, 0, digest, scan=True)]
        layout.scanned = True

    # ---- Thống kê toàn sách -----------------------------------------------------------------

    def _compute_body_size(self, pages: Sequence[PageLayout]) -> float:
        weights: Counter = Counter()
        for page in pages:
            for item in page.items:
                if isinstance(item, TextBlock):
                    for line in item.lines:
                        for s in line.spans:
                            weights[round(s.size * 2) / 2] += len(s.text.strip())
        return float(weights.most_common(1)[0][0]) if weights else 11.0

    def _is_body_size(self, size: float) -> bool:
        return abs(size - self.body_size) <= 0.12 * self.body_size

    def _body_lines(self, blocks: Sequence[TextBlock]) -> List[Line]:
        return [
            line for block in blocks for line in block.lines
            if self._is_body_size(line.size) and len(line.text.strip()) >= 20
        ]

    @staticmethod
    def _metrics_from_lines(lines: Sequence[Line]) -> Optional[_Metrics]:
        if len(lines) < 4:
            return None
        lefts = Counter(round(l.x0) for l in lines)
        repeated = [x for x, n in lefts.items() if n >= 2]
        left = float(min(repeated) if repeated else min(lefts))
        rights = sorted(l.x1 for l in lines)
        right = rights[min(len(rights) - 1, int(len(rights) * 0.9))]
        if right - left < 50:
            return None
        return _Metrics(left, right)

    def _page_metrics(self, pages: Sequence[PageLayout]) -> Dict[int, _Metrics]:
        own: Dict[int, Optional[_Metrics]] = {}
        for page in pages:
            blocks = [i for i in page.items if isinstance(i, TextBlock)]
            own[page.number] = self._metrics_from_lines(self._body_lines(blocks))
        by_parity: Dict[int, List[_Metrics]] = defaultdict(list)
        for number, m in own.items():
            if m:
                by_parity[number % 2].append(m)
        all_metrics = [m for ms in by_parity.values() for m in ms]

        def median_of(ms: List[_Metrics]) -> Optional[_Metrics]:
            if not ms:
                return None
            return _Metrics(statistics.median(m.left for m in ms), statistics.median(m.right for m in ms))

        fallback = {parity: median_of(ms) for parity, ms in by_parity.items()}
        overall = median_of(all_metrics)
        em = self.body_size
        result: Dict[int, _Metrics] = {}
        for page in pages:
            m = own.get(page.number)
            book = fallback.get(page.number % 2) or overall
            ocr_page = any(isinstance(i, TextBlock) and i.ocr for i in page.items)
            if m and book and not ocr_page:
                # Lề trang sách in giống nhau theo trang chẵn/lẻ. Trang toàn trích dẫn (cả trang thụt vào) vẫn
                # dùng lề chung của sách, để trích dẫn vắt sang trang sau được nhận ra và nối đúng.
                left = m.left if m.left < book.left + em else book.left
                right = m.right if m.right > book.right - 2 * em else book.right
                m = _Metrics(left, right)
            m = m or book
            if not m:
                blocks = [i for i in page.items if isinstance(i, TextBlock)]
                if blocks:
                    m = _Metrics(min(b.bbox[0] for b in blocks), max(b.bbox[2] for b in blocks))
                else:
                    m = _Metrics(page.width * 0.1, page.width * 0.9)
            result[page.number] = m
        return result

    def _compute_pitch(self, pages: Sequence[PageLayout]) -> float:
        gaps = []
        for page in pages:
            for item in page.items:
                if not isinstance(item, TextBlock):
                    continue
                for prev, line in zip(item.lines, item.lines[1:]):
                    diff = line.y0 - prev.y0
                    if self._is_body_size(line.size) and 0.5 * self.body_size < diff < 3 * self.body_size:
                        gaps.append(diff)
        return statistics.median(gaps) if gaps else self.body_size * 1.2

    # ---- Dọn trang ----------------------------------------------------------------------------

    def _remove_running_heads(self, pages: Sequence[PageLayout]) -> int:
        """Bỏ đầu trang/chân trang lặp lại và số trang. Trả về số khối đã bỏ."""

        def margin_key(text: str) -> str:
            key = re.sub(r"\d+", "#", text.lower())
            key = re.sub(r"\s+", " ", key)
            return key.strip(" .·•|-–—")

        candidates = []  # (page, block, key, text)
        pages_by_key: Dict[str, List[int]] = defaultdict(list)
        exact_count: Counter = Counter()
        for page in pages:
            for item in page.items:
                if not isinstance(item, TextBlock) or item.ocr:
                    continue
                x0, y0, x1, y1 = item.bbox
                in_margin = y1 <= page.height * 0.12 or y0 >= page.height * 0.88
                if not in_margin or len(item.lines) > 2:
                    continue
                text = item.text.strip()
                if not text or len(text) > 150 or item.size > self.body_size * 1.5:
                    continue
                key = margin_key(text)
                zone = "top" if y1 <= page.height * 0.12 else "bottom"
                candidates.append((page, item, f"{zone}:{key}", text))
                pages_by_key[f"{zone}:{key}"].append(page.number)
                exact_count[(zone, text)] += 1

        removed = 0
        total_pages = max(1, len(pages))
        for page, item, key, text in candidates:
            bare = key.split(":", 1)[1]
            numbers = sorted(set(pages_by_key[key]))
            adjacent = sum(1 for a, b in zip(numbers, numbers[1:]) if b - a <= 2)
            is_page_number = bool(_PAGE_NUMBER_RE.match(bare)) and item.size <= self.body_size * 1.5
            is_running = (
                len(numbers) >= 3 and item.size <= self.body_size * 1.15
                and (adjacent >= 2 or len(numbers) >= 0.25 * total_pages)
            )
            if is_page_number or is_running:
                if item in page.items:
                    page.items.remove(item)
                    removed += 1
        return removed

    def _fix_drop_caps(self, page: PageLayout) -> None:
        """Chữ hoa lớn đầu chương ("T" + "he morning...") ghép lại vào dòng đầu đoạn."""

        def is_cap(line: Line) -> bool:
            text = line.text.strip()
            return len(text) == 1 and text.isalpha() and line.size >= 1.8 * self.body_size

        def merge_into(cap: Line, lines: List[Line]) -> None:
            first = lines[0]
            first.spans[0].text = first.spans[0].text.lstrip()
            first.spans.insert(0, Span(cap.text.strip(), first.size, bold=False, italic=False))
            for line in lines:
                if line.y0 < cap.y1:
                    line.x0 = min(line.x0, cap.x0)

        items = page.items
        index = 0
        while index < len(items):
            item = items[index]
            if isinstance(item, TextBlock):
                if len(item.lines) >= 2 and is_cap(item.lines[0]):
                    cap = item.lines.pop(0)
                    merge_into(cap, item.lines)
                elif len(item.lines) == 1 and is_cap(item.lines[0]):
                    following = next((i for i in items[index + 1:] if isinstance(i, TextBlock)), None)
                    if following is not None:
                        merge_into(item.lines[0], following.lines)
                        items.pop(index)
                        continue
            index += 1

    def _order_items(self, page: PageLayout) -> List[Tuple[object, int]]:
        """Thứ tự đọc: giữ thứ tự của PDF, trừ trang hai cột rõ ràng thì đọc hết cột trái rồi tới cột phải."""
        items = page.items
        mid = page.width / 2
        margin = page.width * 0.03
        chars = {"left": 0, "right": 0, "all": 0}
        counts = {"left": 0, "right": 0}
        for item in items:
            if not isinstance(item, TextBlock):
                continue
            x0, _, x1, _ = item.bbox
            n = len(item.text)
            chars["all"] += n
            if x1 <= mid + margin:
                chars["left"] += n
                counts["left"] += 1
            elif x0 >= mid - margin:
                chars["right"] += n
                counts["right"] += 1
        two_columns = (
            chars["all"] >= 400 and counts["left"] >= 2 and counts["right"] >= 2
            and chars["left"] >= 0.25 * chars["all"] and chars["right"] >= 0.25 * chars["all"]
        )
        if not two_columns:
            return [(item, 0) for item in items]

        def column_of(item) -> int:
            x0, _, x1, _ = item.bbox if isinstance(item, TextBlock) else item.bbox
            if x1 <= mid + margin:
                return 1
            if x0 >= mid - margin:
                return 2
            return 0

        columns = [(item, column_of(item)) for item in items]
        top_of_columns = min((it.bbox[1] for it, c in columns if c), default=0)
        head = sorted([(it, c) for it, c in columns if c == 0 and it.bbox[1] < top_of_columns], key=lambda p: p[0].bbox[1])
        tail = sorted([(it, c) for it, c in columns if c == 0 and it.bbox[1] >= top_of_columns], key=lambda p: p[0].bbox[1])
        left = sorted([(it, c) for it, c in columns if c == 1], key=lambda p: p[0].bbox[1])
        right = sorted([(it, c) for it, c in columns if c == 2], key=lambda p: p[0].bbox[1])
        return head + left + right + tail

    # ---- Phong cách đoạn văn và từ điển -------------------------------------------------------

    def _alignment(self, lines: Sequence[Line], m: _Metrics) -> str:
        em = self.body_size
        width = m.right - m.left
        center = (m.left + m.right) / 2
        tolerance = max(0.04 * width, 0.8 * em)
        if len(lines) > 1:
            # Các dòng cùng tâm nhưng khác độ dài: căn giữa, kể cả khi khối đặt lệch tâm trang.
            centers = [(l.x0 + l.x1) / 2 for l in lines]
            spread = max(centers) - min(centers)
            x0_range = max(l.x0 for l in lines) - min(l.x0 for l in lines)
            x1_range = max(l.x1 for l in lines) - min(l.x1 for l in lines)
            short = all((l.x1 - l.x0) < 0.75 * width for l in lines)
            if spread <= 0.8 * em and x0_range > 1.5 * em and x1_range > 1.5 * em:
                return "center"
            if short and spread <= 0.3 * em and x0_range > 0.6 * em and x1_range > 0.6 * em:
                return "center"
        if all(abs((l.x0 + l.x1) / 2 - center) <= tolerance for l in lines):
            not_full = any((l.x1 - l.x0) < width - 3 * em for l in lines)
            if len(lines) > 1:
                varies = max(l.x0 for l in lines) - min(l.x0 for l in lines) > 1.5 * em
            else:
                varies = lines[0].x0 - m.left > 3 * em
            if not_full and varies:
                return "center"
        if all(abs(l.x1 - m.right) <= em for l in lines) and all(l.x0 - m.left > 0.35 * width for l in lines):
            return "right"
        return ""

    def _detect_style(self, pages: Sequence[PageLayout], metrics: Dict[int, _Metrics]) -> _Style:
        em = self.body_size
        indent_votes, flush_votes = 0, 0
        indents: List[float] = []
        full_lines, measured = 0, 0
        for page in pages:
            m = metrics[page.number]
            for item in page.items:
                if not isinstance(item, TextBlock) or len(item.lines) < 2 or not self._is_body_size(item.size):
                    continue
                if self._alignment(item.lines, m):
                    continue
                rest = statistics.median(l.x0 for l in item.lines[1:]) - m.left
                first = item.lines[0].x0 - m.left
                if abs(rest) <= 0.5 * em:
                    if 0.5 * em <= first <= 6 * em:
                        indent_votes += 1
                        indents.append(first)
                    elif abs(first) < 0.3 * em:
                        flush_votes += 1
                for line in item.lines[:-1]:
                    measured += 1
                    if line.x1 >= m.right - 0.6 * em:
                        full_lines += 1
        style = _Style()
        style.indent = indent_votes >= 3 and indent_votes >= 0.3 * (indent_votes + flush_votes)
        if indents:
            style.indent_em = round(min(3.0, max(0.8, statistics.median(indents) / em)), 2)
        style.justify = measured == 0 or full_lines / measured >= 0.6
        return style

    def _build_vocabulary(self, pages: Sequence[PageLayout]) -> None:
        vocab: Counter = Counter()
        hyphen_vocab: set = set()
        for page in pages:
            for item in page.items:
                if not isinstance(item, TextBlock):
                    continue
                previous_hyphen = False
                for line in item.lines:
                    text = line.text.strip()
                    tokens = _WORD_RE.findall(text)
                    ends_hyphen = bool(_LINE_HYPHEN_RE.search(text)) or text.endswith("\u00ad")
                    if previous_hyphen and tokens:
                        tokens = tokens[1:]
                    if ends_hyphen and tokens:
                        tokens = tokens[:-1]
                    vocab.update(t.lower() for t in tokens)
                    for a, b in _HYPHEN_COMPOUND_RE.findall(text):
                        hyphen_vocab.add(f"{a}-{b}".lower())
                    previous_hyphen = ends_hyphen
        self.vocab = set(vocab)
        self.hyphen_vocab = hyphen_vocab

    # ---- Tách đoạn ----------------------------------------------------------------------------

    def _split_paragraphs(self, lines: List[Line], m: _Metrics, style: _Style) -> List[List[Line]]:
        em = self.body_size
        short_limit = 2.0 * em if style.justify else 0.25 * (m.right - m.left)
        groups: List[List[Line]] = [[lines[0]]]
        for prev, line in zip(lines, lines[1:]):
            same_row = abs(line.y0 - prev.y0) < 0.4 * em
            indent = line.x0 - m.left
            prev_indent = prev.x0 - m.left
            prev_short = prev.x1 < m.right - short_limit
            new_paragraph = False
            if same_row:
                new_paragraph = False
            elif style.indent and 0.5 * em <= indent <= 6 * em and (abs(indent - prev_indent) > 0.3 * em or prev_short):
                new_paragraph = True
            elif prev_short and _ends_sentence(prev.text) and not line.all_sup:
                new_paragraph = True
            elif self.pitch and (line.y0 - prev.y0) > self.pitch * 1.6:
                new_paragraph = True
            if new_paragraph:
                groups.append([line])
            else:
                groups[-1].append(line)
        return groups

    def _is_verse(self, lines: Sequence[Line], m: _Metrics) -> bool:
        if len(lines) < 3:
            return False
        em = self.body_size
        short = [l for l in lines if l.x1 < m.right - 4 * em]
        if len(short) < 0.7 * len(lines):
            return False
        sentence_ends = sum(1 for l in short if _ends_sentence(l.text))
        return sentence_ends < 0.5 * len(short)

    def _make_unit(self, page: PageLayout, segment: Tuple[int, int], lines: List[Line], m: _Metrics,
                   style: _Style, align: str = "", verse: bool = False, ocr: bool = False,
                   quote: bool = False) -> _Unit:
        em = self.body_size
        para = _Para()
        for index, line in enumerate(lines):
            if index == 0:
                para.add_line(line, self.vocab, self.hyphen_vocab)
                continue
            prev = lines[index - 1]
            tight = line.all_sup and abs(line.y0 - prev.y0) < em
            para.add_line(line, self.vocab, self.hyphen_vocab, hard_break=verse, tight=tight)
        weights: Counter = Counter()
        for line in lines:
            weights[line.size] += len(line.text.strip())
        size = weights.most_common(1)[0][0] if weights else self.body_size
        first_indent = lines[0].x0 - m.left
        unit = _Unit(
            kind="text",
            page=page.number,
            bbox=[min(l.x0 for l in lines), min(l.y0 for l in lines), max(l.x1 for l in lines), max(l.y1 for l in lines)],
            segment=segment,
            para=para,
            size=size,
            body=self._is_body_size(size),
            align=align,
            verse=verse,
            indented=0.5 * em <= first_indent <= 6 * em,
            last_full=lines[-1].x1 >= m.right - 1.5 * em,
            first_y0=lines[0].y0,
            last_y0=lines[-1].y0,
            last_y1=lines[-1].y1,
            line_count=len(lines),
            ocr=ocr,
            quote=quote,
        )
        visible = para.visible().strip()
        if _SCENE_BREAK_RE.match(visible) and len(lines) == 1:
            unit.scene = True
            unit.align = "center"
        _, bold, _ = para.render()
        if unit.body and bold >= 0.9 and len(lines) == 1 and len(visible.split()) <= 15:
            # Dòng đậm ngắn cỡ chữ thường thường là tiêu đề mục: không nối với đoạn trước/sau.
            unit.heading_like = True
        return unit

    def _block_units(self, page: PageLayout, segment: Tuple[int, int], block: TextBlock, m: _Metrics, style: _Style) -> List[_Unit]:
        lines = block.lines
        if not lines:
            return []
        align = self._alignment(lines, m)
        body = self._is_body_size(block.size) or block.ocr
        if not body:
            return [self._make_unit(page, segment, lines, m, style, align=align, ocr=block.ocr)]
        if align == "center" and len(lines) > 1:
            return [self._make_unit(page, segment, lines, m, style, align="center", verse=True, ocr=block.ocr)]
        if align and len(lines) == 1:
            return [self._make_unit(page, segment, lines, m, style, align=align, ocr=block.ocr)]
        units = []
        for region_lines, region_m, quote in self._margin_regions(lines, m, style):
            if self._is_verse(region_lines, region_m):
                units.append(self._make_unit(page, segment, region_lines, region_m, style, verse=True,
                                             ocr=block.ocr, quote=quote))
                continue
            for group in self._split_paragraphs(region_lines, region_m, style):
                group_align = self._alignment(group, m) if len(group) == 1 and not quote else ""
                units.append(self._make_unit(page, segment, group, region_m, style, align=group_align,
                                             ocr=block.ocr, quote=quote))
        return units

    def _margin_regions(self, lines: List[Line], m: _Metrics, style: _Style) -> List[Tuple[List[Line], _Metrics, bool]]:
        """Chia khối thành các vùng cùng lề trái; vùng thụt sâu hơn thụt đầu dòng là trích dẫn.

        Lề trái của một chỗ lấy từ dòng nối tiếp (dòng ngay sau một dòng đầy), vì dòng đầu đoạn thụt vào
        còn dòng cuối đoạn ngắn. Nhờ vậy trích dẫn nằm chung khối PyMuPDF với văn bản thường vẫn tách được.
        """
        em = self.body_size
        n = len(lines)
        if n < 2:
            return [(lines, m, False)]
        right = max(l.x1 for l in lines)
        short_limit = 2.0 * em if style.justify else 0.25 * (m.right - m.left)
        known: List[Optional[float]] = [None] * n
        for i in range(1, n):
            prev, line = lines[i - 1], lines[i]
            if abs(line.y0 - prev.y0) >= 0.4 * em and prev.x1 >= right - short_limit:
                known[i] = line.x0
        margins: List[Optional[float]] = [None] * n
        for i in range(n):
            if known[i] is not None:
                margins[i] = known[i]
            elif i + 1 < n and known[i + 1] is not None and lines[i].x0 >= known[i + 1] - 0.5 * em:
                margins[i] = known[i + 1]  # dòng đầu đoạn lấy lề của dòng nối tiếp ngay sau
        previous: Optional[float] = None
        for i in range(n):
            if margins[i] is None:
                margins[i] = previous
            previous = margins[i]
        first_known = next((x for x in margins if x is not None), m.left)
        filled = [first_known if x is None else x for x in margins]

        threshold = max(1.5 * em, (style.indent_em + 0.5) * em)
        regions: List[Tuple[List[Line], _Metrics, bool]] = []
        start = 0
        for i in range(1, n + 1):
            if i == n or abs(filled[i] - filled[i - 1]) >= 0.5 * em:
                region = lines[start:i]
                left = filled[start]
                if left - m.left >= threshold:
                    regions.append((region, _Metrics(left, max(l.x1 for l in region)), True))
                else:
                    regions.append((region, m, False))
                start = i
        return regions

    # ---- Nối đoạn sang trang/cột --------------------------------------------------------------

    def _continues(self, prev: _Unit, unit: _Unit, style: _Style) -> bool:
        prev_text = prev.para.visible().rstrip()
        next_text = unit.para.visible().lstrip()
        if not prev_text or not next_text or not prev.last_full or prev.quote != unit.quote:
            return False
        if prev.segment == unit.segment:
            if unit.first_y0 - prev.last_y0 > self.pitch * 1.35:
                return False
            if style.indent:
                return not unit.indented
            return continues_sentence(prev_text, next_text) or not _ends_sentence(prev_text)
        if not (prev.last_in_segment and unit.first_in_segment):
            return False
        if style.indent:
            return not unit.indented
        return not _ends_sentence(prev_text) or next_text[:1].islower()

    def _merge_centered(self, units: List[_Unit]) -> List[_Unit]:
        """Khối căn giữa liền nhau, cùng tâm (lời đề tặng, khổ thơ bị PyMuPDF chia đôi) gộp thành một khổ."""
        em = self.body_size
        out: List[_Unit] = []
        for unit in units:
            prev = out[-1] if out else None
            if prev is None or unit.kind != "text" or prev.kind != "text":
                out.append(unit)
                continue
            center_gap = abs((prev.bbox[0] + prev.bbox[2]) - (unit.bbox[0] + unit.bbox[2])) / 2
            # Dòng lẻ ngay dưới khối căn giữa, cùng tâm: cũng là dòng căn giữa của khổ đó.
            centered = unit.align == "center" or (
                not unit.align and unit.line_count == 1 and center_gap <= 0.5 * em and not unit.heading_like
            )
            if (
                prev.align == "center" and centered
                and not prev.scene and not unit.scene and prev.body and unit.body
                and prev.segment == unit.segment
                and center_gap <= 0.8 * em
                and unit.first_y0 - prev.last_y0 <= self.pitch * 1.6
            ):
                self._absorb_line_block(prev, unit)
                continue
            out.append(unit)
        return out

    @staticmethod
    def _absorb_line_block(prev: _Unit, unit: _Unit) -> None:
        """Gộp khối sau vào khối trước, giữ xuống dòng giữa hai khối (thơ, lời đề tặng)."""
        prev.para.segs[-1][0] = prev.para.segs[-1][0].rstrip() + "\n"
        first = True
        for text, styles in unit.para.segs:
            prev.para._push(text.lstrip() if first else text, styles)
            first = False
        prev.verse = True
        prev.align = "center"
        prev.bbox = [min(prev.bbox[0], unit.bbox[0]), prev.bbox[1], max(prev.bbox[2], unit.bbox[2]), prev.bbox[3]]
        prev.last_y0, prev.last_y1 = unit.last_y0, unit.last_y1
        prev.segment = unit.segment
        prev.last_in_segment = unit.last_in_segment
        prev.line_count += unit.line_count

    def _continues_centered(self, prev: _Unit, unit: _Unit) -> bool:
        """Khổ thơ căn giữa bị cắt sang trang/cột sau: cùng tâm, nằm cuối trang trước và đầu trang sau."""
        return (
            prev.kind == "text" and unit.kind == "text"
            and prev.align == "center" and unit.align == "center"
            and prev.body and unit.body and not prev.scene and not unit.scene
            and prev.segment != unit.segment and prev.last_in_segment and unit.first_in_segment
            and abs((prev.bbox[0] + prev.bbox[2]) - (unit.bbox[0] + unit.bbox[2])) / 2 <= 0.8 * self.body_size
        )

    def _join_units(self, units: List[_Unit], style: _Style) -> List[_Unit]:
        out: List[_Unit] = []
        last_body: Optional[int] = None
        last_flow: Optional[int] = None
        for unit in units:
            if unit.kind == "text" and unit.footnote:
                out.append(unit)
                continue
            if last_flow is not None and self._continues_centered(out[last_flow], unit):
                self._absorb_line_block(out[last_flow], unit)
                continue
            if unit.joinable and last_body is not None and self._continues(out[last_body], unit, style):
                prev = out[last_body]
                prev.para.extend(unit.para, self.vocab, self.hyphen_vocab)
                prev.last_full = unit.last_full
                prev.last_y0, prev.last_y1 = unit.last_y0, unit.last_y1
                prev.segment = unit.segment
                prev.last_in_segment = unit.last_in_segment
                prev.line_count += unit.line_count
                continue
            out.append(unit)
            last_body = len(out) - 1 if unit.joinable else None
            last_flow = len(out) - 1
        return out

    # ---- Outline (mục lục PDF) ----------------------------------------------------------------

    def _apply_outline(self, units: List[_Unit], outline: List[list]) -> bool:
        entries = [(int(e[0]), str(e[1]), int(e[2])) for e in outline if len(e) >= 3 and str(e[1]).strip() and int(e[2]) >= 1]
        if not entries:
            return False
        min_level = min(level for level, _, _ in entries)
        by_page: Dict[int, List[int]] = defaultdict(list)
        for index, unit in enumerate(units):
            if unit.kind == "text" and not unit.footnote:
                by_page[unit.page].append(index)
        used: set = set()
        matched = 0
        for level, title, page_number in entries:
            target = _norm_title(title)
            if not target:
                continue
            found = None
            for page in (page_number, page_number + 1, page_number - 1):
                for index in by_page.get(page, []):
                    if index in used:
                        continue
                    text = _norm_title(units[index].para.visible())
                    if not text or len(text) > 250:
                        continue
                    if (
                        text == target
                        or (target.startswith(text) and len(text) >= max(4, 0.4 * len(target)))
                        or (text.startswith(target) and len(text) <= len(target) + 40)
                    ):
                        found = index
                        break
                if found is not None:
                    break
            if found is not None:
                used.add(found)
                units[found].heading_level = min(4, level - min_level + 1)
                matched += 1
        return matched >= max(1, 0.5 * len(entries))

    # ---- Siêu dữ liệu -------------------------------------------------------------------------

    @staticmethod
    def _clean_title(raw: str) -> str:
        title = (raw or "").strip()
        title = re.sub(r"^microsoft word\s*-\s*", "", title, flags=re.I)
        title = re.sub(r"\.(docx?|pdf|odt|rtf)$", "", title, flags=re.I).strip()
        return "" if _JUNK_TITLE_RE.match(title) else title

    def _title_from_filename(self, pdf_path: Path) -> str:
        stem = re.sub(r"[_]+", " ", Path(pdf_path).stem).strip()
        return stem or "Sách"

    # ---- Chạy ---------------------------------------------------------------------------------

    def analyze(self, pdf_path: Path) -> PdfAnalysis:
        try:
            doc = pymupdf.open(str(pdf_path))
        except Exception as exc:
            raise ValueError(f"Không mở được file PDF: {exc}") from exc
        if doc.needs_pass:
            doc.close()
            raise ValueError("PDF được bảo vệ bằng mật khẩu, không thể chuyển đổi.")

        warnings: List[str] = []
        try:
            meta = doc.metadata or {}
            try:
                outline = doc.get_toc(simple=True) or []
            except Exception:
                outline = []
            catalog_lang = ""
            try:
                kind, value = doc.xref_get_key(doc.pdf_catalog(), "Lang")
                if kind == "string":
                    catalog_lang = value.strip().split("-")[0].lower()
            except Exception:
                pass

            page_count = len(doc)
            pages: List[PageLayout] = []
            for index in range(page_count):
                pages.append(self._extract_page(doc[index], index + 1))
                if index % 5 == 0 or index == page_count - 1:
                    self.progress(0.05 + 0.65 * (index + 1) / max(1, page_count), f"Đọc trang {index + 1}/{page_count}")

            cover_bytes = None
            skip_first_page = False
            if self.want_cover and pages:
                first = pages[0]
                images = [i for i in first.items if isinstance(i, ImageItem)]
                page_area = first.width * first.height or 1.0
                big = [
                    i for i in images
                    if (i.bbox[2] - i.bbox[0]) * (i.bbox[3] - i.bbox[1]) >= 0.6 * page_area
                ]
                try:
                    cover_bytes = render_page_image(doc[0], self.device.screen_width, self.device.screen_height)
                except Exception:
                    cover_bytes = None
                if cover_bytes and big and first.char_count < 80:
                    skip_first_page = True  # trang 1 chính là ảnh bìa: không lặp lại trong nội dung
        finally:
            doc.close()

        self.progress(0.72, "Phân tích bố cục trang")
        if skip_first_page:
            pages[0].items = []
            pages[0].scanned = False

        self.body_size = self._compute_body_size(pages)
        for page in pages:
            self._fix_drop_caps(page)
        self._remove_running_heads(pages)
        metrics = self._page_metrics(pages)
        self.pitch = self._compute_pitch(pages)
        style = self._detect_style(pages, metrics)
        self._build_vocabulary(pages)

        # Ảnh lặp lại trên nhiều trang (logo, hoa văn) và ảnh nền cả trang dưới lớp chữ.
        image_pages: Dict[str, set] = defaultdict(set)
        for page in pages:
            for item in page.items:
                if isinstance(item, ImageItem) and not item.scan:
                    image_pages[item.digest].add(page.number)

        units: List[_Unit] = []
        for page in pages:
            m = metrics[page.number]
            ordered = self._order_items(page)
            columns_metrics: Dict[int, _Metrics] = {}
            for column in {c for _, c in ordered if c}:
                lines = self._body_lines([it for it, c in ordered if c == column and isinstance(it, TextBlock)])
                columns_metrics[column] = self._metrics_from_lines(lines) or m
            page_units: List[_Unit] = []
            for item, column in ordered:
                segment = (page.number, column)
                cm = columns_metrics.get(column, m)
                if isinstance(item, ImageItem):
                    area = (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
                    if not item.scan:
                        if len(image_pages[item.digest]) >= 3:
                            continue
                        if area >= 0.85 * page.width * page.height and page.char_count >= 200:
                            continue
                    page_units.append(_Unit(kind="image", page=page.number, bbox=list(item.bbox), segment=segment, image=item))
                else:
                    page_units.extend(self._block_units(page, segment, item, cm, style))
            page_units = self._merge_centered(page_units)
            # Chú thích cuối trang: chữ nhỏ ở phần dưới trang, mở đầu bằng số/dấu sao.
            for unit in page_units:
                if (
                    unit.kind == "text" and not unit.body and unit.size <= 0.92 * self.body_size
                    and unit.bbox[1] >= page.height * 0.6
                    and _FOOTNOTE_START_RE.match(unit.para.visible().strip() or "")
                ):
                    unit.footnote = True
            by_segment: Dict[Tuple[int, int], List[_Unit]] = defaultdict(list)
            for unit in page_units:
                if not unit.footnote:
                    by_segment[unit.segment].append(unit)
            for seg_units in by_segment.values():
                seg_units[0].first_in_segment = True
                seg_units[-1].last_in_segment = True
            units.extend(page_units)

        outline_used = self._apply_outline(units, outline)
        units = self._join_units(units, style)
        self.progress(0.8, "Dựng đoạn văn")

        # Siêu dữ liệu.
        title = (
            self.title_override or self._clean_title(meta.get("title", ""))
            or self._title_from_filename(Path(self.source_name) if self.source_name else pdf_path)
        )
        author = self.author_override or (meta.get("author") or "").strip()
        samples = [u.para.visible() for u in units[:4000] if u.kind == "text"]
        language = self.language_override or catalog_lang or detect_language(samples, default="en")

        # Xuất định dạng trang/khối cho StructureBuilder.
        out_pages: Dict[int, dict] = {
            page.number: {"page_number": page.number, "width": page.width, "height": page.height, "blocks": []}
            for page in pages
        }
        assets: List[dict] = []
        asset_ids: Dict[str, str] = {}
        image_count = 0
        for block_id, unit in enumerate(units):
            target = out_pages[unit.page]["blocks"]
            if unit.kind == "image":
                item = unit.image
                asset_id = asset_ids.get(item.digest)
                if not asset_id:
                    asset_id = f"pdfimg_{len(asset_ids) + 1:05d}"
                    asset_ids[item.digest] = asset_id
                    assets.append({
                        "id": asset_id,
                        "original_path": str(item.path),
                        "source_page": unit.page,
                        "width": item.width or None,
                        "height": item.height or None,
                        "mime_type": "image/png" if item.path.suffix == ".png" else "image/jpeg",
                    })
                image_count += 1
                target.append({"block_id": block_id, "type": "image", "asset_id": asset_id, "bbox": unit.bbox, "confidence": 1.0})
                continue
            text, bold, italic = unit.para.render()
            if not strip_markers(text).strip():
                continue
            block = {
                "block_id": block_id,
                "type": "text",
                "text": text,
                "bbox": unit.bbox,
                "font_size": round(unit.size, 1),
                "font_name": "",
                "is_bold": bold >= 0.6,
                "is_italic": italic >= 0.6,
                "confidence": 0.85 if unit.ocr else 1.0,
                "footnote": unit.footnote,
                "layout": {
                    "align": unit.align,
                    "verse": unit.verse,
                    "scene": unit.scene,
                    "indent": unit.indented,
                    "quote": unit.quote,
                    "body": unit.body,
                },
            }
            if unit.heading_level:
                block.update({"type": "heading", "structured": True, "heading_level": unit.heading_level})
            target.append(block)

        scanned = sum(1 for p in pages if p.scanned)
        ocr_pages = sum(1 for p in pages if p.ocr_used)
        if scanned:
            warnings.append(
                f"{scanned} trang là ảnh scan không có lớp chữ"
                + (" (máy chưa cài Tesseract OCR)" if not self.ocr.is_available else "")
                + ": giữ nguyên dạng ảnh cả trang, không co giãn chữ được."
            )
        if ocr_pages:
            warnings.append(f"{ocr_pages} trang scan đã được nhận dạng chữ (OCR); nên đọc soát lỗi chính tả.")
        if not any(b["type"] != "image" for p in out_pages.values() for b in p["blocks"]):
            warnings.append("Không tìm thấy chữ trong PDF: sách chỉ gồm ảnh trang.")

        return PdfAnalysis(
            pages=[out_pages[p.number] for p in pages],
            assets=assets,
            title=title,
            author=author,
            language=language,
            cover=cover_bytes,
            outline_used=outline_used,
            paragraph_style="indent" if style.indent else "block",
            indent_em=style.indent_em,
            justify=style.justify,
            body_size=self.body_size,
            page_count=len(pages),
            scanned_pages=scanned,
            ocr_pages=ocr_pages,
            image_count=image_count,
            warnings=warnings,
        )

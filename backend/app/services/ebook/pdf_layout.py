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

import copy
import difflib
import hashlib
import io
import os
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
from app.services.extractor.ocr_engine import OCREngine, OcrPage
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
# Đầu trang kiểu "46 | TÊN SÁCH" / "TÊN CHƯƠNG | 47"; OCR hay đọc số trang thành I, l, J, ?, §.
_FOLIO_HEAD_RE = re.compile(
    r"^\W{0,3}[\dIlJ?§$]{1,4}\s*[|•·]\s*\S|\S\s*[|•·]\s*[\dIlJ?§$]{1,4}\W{0,3}$"
    r"|^\W{0,3}\S{0,4}\s*\|\s*\S|\S\s*\|\s*\S{0,4}\W{0,3}$"  # vạch ngăn cạnh số trang OCR đọc sai ("š0 |", "| 8ï")
)
_LINE_BREAK = chr(10)
_PAGE_REF_END_RE = re.compile(r"\s[\dIl]{1,4}\s*$")
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
    word_confidences: Optional[List[float]] = None  # chữ OCR: độ tin cậy từng từ (0..1)
    word_alternatives: Optional[list] = None  # chữ OCR: phương án thay thế từng chữ cái (xem OcrLine)


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
    needs_ocr: bool = False


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
    toc_entry: bool = False  # dòng của trang "Mục lục" in sẵn trong sách
    footnote: bool = False
    ocr: bool = False
    first_in_segment: bool = False
    last_in_segment: bool = False
    image: Optional[ImageItem] = None

    @property
    def joinable(self) -> bool:
        return (
            self.kind == "text" and self.body and not self.align and not self.verse and not self.scene
            and not self.heading_level and not self.heading_like and not self.footnote and not self.toc_entry
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


_OCR_BULLET_RE = re.compile(r"^(?:e|®|©|«|°|¢|o)\s+(\S)")


def _fix_ocr_bullet(text: str) -> str:
    """Dấu đầu dòng "•" mà OCR đọc thành "e", "®", "«"... ("e FoxSports.com: ..." -> "• FoxSports.com: ...")."""
    match = _OCR_BULLET_RE.match(text)
    if match and (match.group(1)[0].isupper() or match.group(1)[0] in "“\"'"):
        return "• " + text[match.start(1):]
    return text


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
        self.ocr = OCREngine(lang="eng")
        self.ocr_lang = ""
        self.spell_fixes = 0
        self.ocr_repairs = 0
        self.name_fixes = 0
        self.heading_fixes = 0
        self.body_size = 11.0
        self.pitch = 13.0
        self.vocab: set = set()
        self.hyphen_vocab: set = set()

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

    def _read_blocks(self, page, layout: PageLayout, with_images: bool) -> int:
        """Đọc khối chữ (và ảnh nếu cần) của trang vào layout.items. Trả về số từ."""
        flags = _TEXT_FLAGS if with_images else (_TEXT_FLAGS & ~pymupdf.TEXT_PRESERVE_IMAGES)
        data = page.get_text("dict", flags=flags)
        layout.items = []
        layout.char_count = 0
        words = 0
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
                        flags_ = int(raw.get("flags", 0) or 0)
                        spans.append(Span(
                            text=text,
                            size=float(raw.get("size", 0.0) or 0.0),
                            bold=bool(flags_ & 16) or bool(_BOLD_FONT.search(font)),
                            italic=bool(flags_ & 2) or bool(_ITALIC_FONT.search(font)),
                            sup=bool(flags_ & 1),
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
            elif block.get("type") == 1 and with_images:
                item = self._extract_image(page, block)
                if item:
                    layout.items.append(item)
        return words

    def _extract_page(self, page, number: int) -> PageLayout:
        rect = page.rect
        layout = PageLayout(number=number, width=rect.width, height=rect.height)
        try:
            image_infos = page.get_image_info()  # chỉ toạ độ, không giải mã ảnh
        except Exception:
            image_infos = []
        image_area = 0.0
        for info in image_infos:
            r = pymupdf.Rect(info.get("bbox", (0, 0, 0, 0))) & rect
            image_area += max(0.0, r.width) * max(0.0, r.height)
        words = self._read_blocks(page, layout, with_images=False)

        page_area = rect.width * rect.height or 1.0
        page_type = ScanDetector.classify_page(layout.char_count, words, len(image_infos), image_area / page_area, page_area)
        if page_type == PageType.SCANNED_PAGE and layout.char_count < 60:
            # Trang scan: không trích ảnh scan (rất nặng với ảnh 600 dpi); OCR song song sau khi đọc hết trang.
            layout.needs_ocr = True
            return layout
        if image_infos and self.include_images:
            # Trang có cả chữ và ảnh: đọc lại kèm ảnh để giữ đúng thứ tự ảnh giữa các khối chữ.
            self._read_blocks(page, layout, with_images=True)
        return layout

    def _keep_scan_image(self, page, layout: PageLayout) -> None:
        """Không OCR được: giữ nguyên trang dưới dạng ảnh xám vừa màn hình để không mất nội dung."""
        layout.needs_ocr = False
        try:
            side = self.device.image_max_side
            data = render_page_image(page, side, side, gray=True)
        except Exception:
            layout.items = []
            return
        digest = hashlib.md5(data).hexdigest()
        path = self.image_dir / f"scan_{layout.number:05d}.png"
        path.write_bytes(data)
        layout.items = [ImageItem(path, (0, 0, layout.width, layout.height), 0, 0, digest, scan=True)]
        layout.scanned = True

    # ---- OCR trang scan -----------------------------------------------------------------------

    @staticmethod
    def _ocr_dpi(page) -> int:
        """300 dpi (chuẩn của Tesseract); trang khổ lớn thì hạ xuống để ảnh không quá ~35 megapixel."""
        area_in2 = (page.rect.width / 72.0) * (page.rect.height / 72.0)
        limit = (35_000_000 / max(area_in2, 1e-6)) ** 0.5
        return int(max(150, min(300, limit)))

    @staticmethod
    def _render_for_ocr(page, dpi: int) -> Image.Image:
        """Ảnh xám của trang (render ở luồng chính; mã hóa PNG để ở luồng OCR cho song song)."""
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
        return Image.frombytes("L", (pix.width, pix.height), pix.samples)

    def _choose_ocr_lang(self, doc, pages: Sequence[PageLayout], pending: Sequence[PageLayout], catalog_lang: str) -> str:
        """Một ngôn ngữ cho cả sách: ghép "vie+eng" chậm hơn mà lại nhận sai nhiều hơn chỉ dùng "vie"."""
        installed = self.ocr.available_languages()
        to_tess = {"vi": "vie", "en": "eng"}

        def usable(code: str) -> Optional[str]:
            code = to_tess.get(code, code)
            return code if code in installed else None

        forced = (self.language_override or catalog_lang or "").split("-")[0].lower()
        if forced and usable(forced):
            return usable(forced)
        text_pages = [p for p in pages if not p.needs_ocr and p.char_count > 200]
        if text_pages:
            samples = [it.text for p in text_pages[:40] for it in p.items if isinstance(it, TextBlock)]
            return usable(detect_language(samples, default="en")) or "eng"
        if "vie" not in installed:
            return "eng"
        # Sách toàn ảnh scan: OCR thử một trang giữa sách bằng "eng+vie" rồi chọn theo dấu tiếng Việt.
        probe = pending[len(pending) // 2]
        try:
            page = doc[probe.number - 1]
            dpi = self._ocr_dpi(page)
            result = self.ocr.recognize_image(self._render_for_ocr(page, dpi), "eng+vie", dpi)
            return "vie" if detect_language([result.text], default="en") == "vi" else "eng"
        except Exception:
            return "eng"

    def _ocr_figures(self, layout: PageLayout, blocks: Sequence[TextBlock], im: Image.Image, dpi: int) -> List[ImageItem]:
        """Hình minh họa trên trang scan: khoảng trống lớn giữa các đoạn chữ mà vẫn có nhiều điểm tối."""
        width, height = im.size
        scale = dpi / 72.0
        x_lo = int(min(b.bbox[0] for b in blocks) * scale)
        x_hi = int(max(b.bbox[2] for b in blocks) * scale)
        if x_hi - x_lo < width * 0.3:
            x_lo, x_hi = int(width * 0.08), int(width * 0.92)
        intervals = sorted((b.bbox[1] * scale, b.bbox[3] * scale) for b in blocks)
        merged: List[List[float]] = []
        for y0, y1 in intervals:
            if merged and y0 <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], y1)
            else:
                merged.append([y0, y1])
        top, bottom = height * 0.06, height * 0.94
        gaps, cursor = [], top
        for y0, y1 in merged:
            if y0 - cursor >= 0.10 * height:
                gaps.append((cursor, min(y0, bottom)))
            cursor = max(cursor, y1)
        if bottom - cursor >= 0.10 * height:
            gaps.append((cursor, bottom))

        figures: List[ImageItem] = []
        for index, (g0, g1) in enumerate(gaps):
            region = im.crop((x_lo, int(g0), x_hi, int(g1)))
            mask = region.point(lambda v: 255 if v < 140 else 0)
            dark = mask.histogram()[255] / max(1, region.width * region.height)
            box = mask.getbbox()
            if dark < 0.03 or dark > 0.9 or not box:
                continue
            if box[2] - box[0] < 0.15 * width or box[3] - box[1] < 0.05 * height:
                continue
            pad = 8
            crop_box = (max(0, box[0] - pad), max(0, box[1] - pad),
                        min(region.width, box[2] + pad), min(region.height, box[3] + pad))
            crop = region.crop(crop_box)
            buf = io.BytesIO()
            crop.save(buf, format="PNG", optimize=True)
            data = buf.getvalue()
            path = self.image_dir / f"ocrfig_{layout.number:05d}_{index}.png"
            path.write_bytes(data)
            bbox = (
                (x_lo + crop_box[0]) / scale, (g0 + crop_box[1]) / scale,
                (x_lo + crop_box[2]) / scale, (g0 + crop_box[3]) / scale,
            )
            figures.append(ImageItem(path, bbox, crop.width, crop.height, hashlib.md5(data).hexdigest()))
        return figures

    def _ocr_diagrams(self, layout: PageLayout, result: OcrPage, image: Image.Image, dpi: int
                      ) -> Tuple[List[ImageItem], set]:
        """Sơ đồ có chữ trên trang scan (vòng lặp, mũi tên): giữ thành hình, bỏ chữ OCR rời rạc bên trong.

        Xoá chữ đã nhận ra khỏi ảnh trang; mực còn lại là nét vẽ. Dải nét vẽ đủ lớn (cao >= ~1,2% trang, rộng
        >= 20% trang) có chữ OCR nằm cùng chỗ là sơ đồ. Gạch chân, vạch kẻ mảnh, chữ hoa đầu chương (hẹp) không đủ.
        Dòng chữ to (tiêu đề sơ đồ) để nguyên là chữ. Trả về (hình, id các dòng OCR đã đưa vào hình).
        """
        from PIL import ImageDraw

        lines = [l for l in result.lines if l.text.strip()]
        if not lines:
            return [], set()
        factor = 4
        small = image.reduce(factor)
        width, height = small.size
        mask = small.point(lambda v: 255 if v < 128 else 0)
        draw = ImageDraw.Draw(mask)
        for line in lines:
            for box in line.word_boxes or [line.bbox]:
                draw.rectangle((box[0] // factor - 1, box[1] // factor - 1, box[2] // factor + 1, box[3] // factor + 1),
                               fill=0)
        top, bottom = int(height * 0.08), int(height * 0.92)
        draw.rectangle((0, 0, width, top), fill=0)
        draw.rectangle((0, bottom, width, height), fill=0)
        rows = [mask.crop((0, y, width, y + 1)).histogram()[255] for y in range(height)]
        bands: List[List[int]] = []
        for y, count in enumerate(rows):
            if count >= 2:
                if bands and y - bands[-1][1] <= 3:
                    bands[-1][1] = y
                else:
                    bands.append([y, y])
        kept = []
        for y0, y1 in bands:
            box = mask.crop((0, y0, width, y1 + 1)).getbbox()
            if not box:
                continue
            band_h, band_w = y1 - y0 + 1, box[2] - box[0]
            ink = sum(rows[y0:y1 + 1])
            wide_enough = band_w >= 0.2 * width or (band_h >= 0.03 * height and band_w >= 0.1 * width)
            if band_h >= 0.012 * height and wide_enough and ink <= 0.35 * band_h * band_w:
                kept.append([y0, y1, box[0], box[2]])
        if not kept:
            return [], set()
        merged = [kept[0]]
        for band in kept[1:]:
            if band[0] - merged[-1][1] <= 0.03 * height:
                last = merged[-1]
                merged[-1] = [last[0], band[1], min(last[2], band[2]), max(last[3], band[3])]
            else:
                merged.append(band)
        sizes = sorted(l.x_size for l in lines)
        median = sizes[len(sizes) // 2]
        figures: List[ImageItem] = []
        taken: set = set()
        for index, (y0, y1, x0, x1) in enumerate(merged):
            py0, py1, px0, px1 = y0 * factor, (y1 + 1) * factor, x0 * factor, x1 * factor
            inside = []
            for line in lines:
                center = (line.bbox[1] + line.bbox[3]) / 2
                slack = 0.6 * (line.bbox[3] - line.bbox[1])
                if py0 - slack <= center <= py1 + slack and line.x_size < 1.3 * median:
                    inside.append(line)
            if not inside:
                continue  # hình không có chữ: _ocr_figures lo
            text_width = max(l.bbox[2] for l in lines) - min(l.bbox[0] for l in lines)
            if sum(1 for l in inside if l.bbox[2] - l.bbox[0] >= 0.85 * text_width) > 2:
                continue  # dải mực cạnh đoạn văn dài (vạch kẻ lề...), không phải sơ đồ
            fy0 = min([py0] + [l.bbox[1] for l in inside]) - 12
            fy1 = max([py1] + [l.bbox[3] for l in inside]) + 12
            fx0 = min([px0] + [l.bbox[0] for l in inside]) - 12
            fx1 = max([px1] + [l.bbox[2] for l in inside]) + 12
            box = (max(0, fx0), max(0, fy0), min(image.width, fx1), min(image.height, fy1))
            crop = image.crop(box)
            buf = io.BytesIO()
            crop.save(buf, format="PNG", optimize=True)
            data = buf.getvalue()
            path = self.image_dir / f"ocrdiagram_{layout.number:05d}_{index}.png"
            path.write_bytes(data)
            scale = 72.0 / dpi
            figures.append(ImageItem(path, tuple(v * scale for v in box), crop.width, crop.height,
                                     hashlib.md5(data).hexdigest()))
            taken.update(id(l) for l in inside)
        return figures, taken

    def _apply_ocr(self, doc, layout: PageLayout, result: Optional[OcrPage], image: Image.Image, dpi: int) -> None:
        if result is not None:
            scale = 72.0 / dpi
            blocks: List[TextBlock] = []
            diagrams, in_diagram = (
                self._ocr_diagrams(layout, result, image, dpi) if self.include_images else ([], set())
            )
            for paragraph in result.paragraphs:
                lines = []
                for ocr_line in paragraph.lines:
                    if id(ocr_line) in in_diagram:
                        continue  # chữ nằm trong sơ đồ: đã giữ trong hình
                    text = _fix_ocr_bullet(clean_span_text(ocr_line.text).strip())
                    if not text or ocr_line.confidence < 0.4:
                        continue  # chữ "nhận" nhầm trong ảnh, vết bẩn
                    alnum = sum(1 for ch in text if ch.isalnum())
                    if alnum < 3 and not text.strip(" .").isdigit() or alnum < 0.4 * len(text.replace(" ", "")):
                        continue  # vạch trang trí đọc thành ký hiệu ("\" ˆ LẠ")
                    x0, y0, x1, y1 = (v * scale for v in ocr_line.bbox)
                    same_words = text.split() == ocr_line.text.split()
                    confidences = ocr_line.word_confidences if len(ocr_line.word_confidences) == len(text.split()) else None
                    alternatives = ocr_line.word_alternatives if same_words and ocr_line.word_alternatives else None
                    lines.append(Line([Span(text, size=max(5.0, ocr_line.x_size * scale), word_confidences=confidences,
                                            word_alternatives=alternatives)],
                                      x0, y0, x1, y1))
                if lines:
                    blocks.append(TextBlock(lines, ocr=True))
            chars = sum(len(l.text) for b in blocks for l in b.lines)
            confidence = result.mean_confidence
            # Trang ít chữ (trang tên phần "PHẦN I") vẫn nhận chữ nếu OCR rất chắc chắn.
            if (chars >= 20 and confidence >= 0.5) or (chars >= 2 and confidence >= 0.8):
                items: list = list(blocks)
                if self.include_images:
                    for figure in self._ocr_figures(layout, blocks, image, dpi) + diagrams:
                        position = next((i for i, it in enumerate(items) if it.bbox[1] > figure.bbox[1]), len(items))
                        items.insert(position, figure)
                layout.items = items
                layout.char_count = chars
                layout.ocr_used = True
                layout.needs_ocr = False
                return
        # Trang gần như không có chữ (ảnh, hình vẽ) hoặc OCR quá kém: giữ ảnh cả trang.
        self._keep_scan_image(doc[layout.number - 1], layout)

    @staticmethod
    def _confident_ocr_words(blocks: Sequence[TextBlock]) -> set:
        """Chữ sách dùng >= 3 lần mà lần nào Tesseract cũng chấm cao ("xèng" trong sách về poker): chữ có thật của
        sách dù từ điển ít gặp. Lỗi lặp lại do phông chữ ("điểu", "tổn") luôn có lần bị chấm thấp hơn."""
        confidences: Dict[str, List[float]] = defaultdict(list)
        for block in blocks:
            for line in block.lines:
                for span in line.spans:
                    words = span.text.split()
                    if not span.word_confidences or len(words) != len(span.word_confidences):
                        continue
                    for word, conf in zip(words, span.word_confidences):
                        token = "".join(ch for ch in word.lower() if ch.isalpha())
                        if token:
                            confidences[token].append(conf)
        return {
            token for token, values in confidences.items()
            if len(values) >= 3 and sum(values) / len(values) >= 0.92 and min(values) >= 0.88
        }

    def _spell_check_ocr(self, pages: Sequence[PageLayout]) -> None:
        """Sửa lỗi dấu tiếng Việt sau OCR theo ngữ cảnh (xem vi_spell). Chạy trên dòng OCR, trước khi dựng đoạn."""
        if not self.ocr_lang.startswith("vie"):
            return
        ocr_blocks = [it for p in pages for it in p.items if isinstance(it, TextBlock) and it.ocr]
        if not ocr_blocks:
            return
        from app.services.ebook.vi_spell import OcrSpellCorrector, VietSyllableModel, lowercase_stray_capitals

        self.progress(0.705, "Sửa lỗi dấu sau OCR")
        model = VietSyllableModel.shared()
        if not model.unigrams:
            return  # không có từ điển Anh-Việt (data/dictionaries/en-vi): không đủ căn cứ để sửa dấu
        all_blocks = [it for p in pages for it in p.items if isinstance(it, TextBlock)]
        # Tên riêng/chữ nước ngoài đọc sai lẻ tẻ thống nhất theo cách đọc phổ biến trong sách ("Scahawks" -> "Seahawks").
        from app.services.ebook.ocr_repair import apply_name_mapping, fix_number_token, unify_rare_names

        names = unify_rare_names([l.text for b in ocr_blocks for l in b.lines], model)
        self.name_fixes = len(names)
        for block in ocr_blocks:
            for line in block.lines:
                for span in line.spans:
                    text = apply_name_mapping(span.text, names)
                    span.text = re.sub(r"\S+", lambda m: fix_number_token(m.group()), text)
        corrector = OcrSpellCorrector(
            model, (_LINE_BREAK.join(l.text for l in b.lines) for b in all_blocks),
            confident_words=self._confident_ocr_words(ocr_blocks),
        )
        sizes = sorted(l.size for b in ocr_blocks for l in b.lines)
        median = sizes[len(sizes) // 2] if sizes else 0.0
        def text_span(line: Line) -> Optional[Span]:
            if len(line.spans) == 1:
                return line.spans[0]
            if len(line.spans) == 2 and len(line.spans[0].text.strip()) == 1:
                return line.spans[1]  # dòng đầu chương có chữ hoa lớn (drop cap) đã ghép vào
            return None

        corrected: List[Tuple[Span, str]] = []
        for block in ocr_blocks:
            lines = [l for l in block.lines if text_span(l) is not None]
            if not lines:
                continue
            spans = [text_span(l) for l in lines]
            heading = all(l.size >= 1.12 * median for l in lines)
            texts = [lowercase_stray_capitals(span.text, model) for span in spans]
            # Dòng thiếu độ tin cậy/phương án thay thế (dòng bị sửa dấu đầu dòng) vẫn sửa được: chỗ trống là None/().
            confidences: List[Optional[float]] = []
            alternatives: list = []
            for span, text in zip(spans, texts):
                count = len(text.split())
                word_confs = span.word_confidences
                confidences.extend(word_confs if word_confs and len(word_confs) == count else [None] * count)
                word_alts = span.word_alternatives
                alternatives.extend(word_alts if word_alts and len(word_alts) == count else [()] * count)
            fixed = corrector.correct_tokens(_LINE_BREAK.join(texts), heading=heading, confidences=confidences,
                                             alternatives=alternatives).split(_LINE_BREAK)
            if len(fixed) != len(lines):
                continue
            for span, text, before in zip(spans, fixed, texts):
                span.text = text
                corrected.append((span, before))
        self.spell_fixes = sum(corrector.changes.values())
        self.spell_fixes += self._consistent_collocations(corrected)

    @staticmethod
    def _consistent_collocations(corrected: Sequence[Tuple[Span, str]]) -> int:
        """Một cặp chữ OCR ("cái bãy") được sửa thống nhất ở hầu hết các chỗ ("cái bẫy" 6/7 lần) thì chỗ còn lại
        theo đa số: ngữ cảnh ở chỗ đó có thể kéo sai ("cái bay nằm")."""
        def tokens(text: str) -> List[str]:
            return text.split()

        votes: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        rows = []
        for span, before in corrected:
            old, new = tokens(before), tokens(span.text)
            if len(old) != len(new):
                continue
            rows.append((span, old, new))
            for index in range(1, len(old)):
                key = (new[index - 1].lower(), old[index].lower())
                if old[index].isalpha() and new[index - 1].isalpha():
                    votes[key][new[index].lower()] += 1
        fixes = 0
        for span, old, new in rows:
            changed = False
            for index in range(1, len(old)):
                key = (new[index - 1].lower(), old[index].lower())
                counter = votes.get(key)
                if not counter or old[index].lower() == new[index].lower() and len(counter) == 1:
                    continue
                winner, count = counter.most_common(1)[0]
                total = sum(counter.values())
                if (
                    winner != new[index].lower() and winner != old[index].lower()
                    and count >= 3 and count >= 0.75 * total
                ):
                    word = new[index]
                    new[index] = winner[:1].upper() + winner[1:] if word[:1].isupper() else winner
                    changed = True
                    fixes += 1
            if changed:
                # Giữ nguyên khoảng trắng/xuống dòng: thay từng chữ theo thứ tự.
                pieces = re.split(r"(\s+)", span.text)
                position = 0
                for k, piece in enumerate(pieces):
                    if piece and not piece.isspace():
                        pieces[k] = new[position]
                        position += 1
                span.text = "".join(pieces)
        return fixes

    def _ocr_page(self, image: Image.Image, dpi: int, single_thread: bool) -> OcrPage:
        """OCR một trang (chạy ở luồng phụ). Sách tiếng Việt: lấy thêm phương án thay thế từng chữ cái cho bộ sửa
        dấu, rồi đọc lại các dòng vỡ chữ (chữ in nghiêng, tên tiếng Anh) bằng "eng" và "vie" trên ảnh nắn thẳng."""
        vietnamese = self.ocr_lang.startswith("vie")
        result = self.ocr.recognize_image(image, self.ocr_lang, dpi, single_thread, alternatives=vietnamese)
        if vietnamese and "eng" in self.ocr.available_languages():
            from app.services.ebook.ocr_repair import repair_page_lines

            def recognize(im: Image.Image, lang: str, resolution: int) -> OcrPage:
                return self.ocr.recognize_image(im, lang, resolution, single_thread, psm=6,
                                                alternatives=lang == "vie")

            try:
                result.repaired = repair_page_lines(image, result.lines, recognize, dpi)
            except Exception as exc:  # đọc lại chỉ là bước làm tốt thêm: lỗi thì giữ kết quả OCR gốc
                print(f"[OCR] đọc lại dòng vỡ chữ lỗi: {exc}")
        if vietnamese:
            from app.services.ebook.ocr_repair import restore_drop_caps

            def recognize_char(im: Image.Image, lang: str, resolution: int) -> OcrPage:
                return self.ocr.recognize_image(im, lang, resolution, single_thread, psm=10)

            try:
                restore_drop_caps(image, result.lines, recognize_char, dpi)
            except Exception as exc:
                print(f"[OCR] khôi phục chữ hoa đầu chương lỗi: {exc}")
        return result

    def _run_ocr(self, doc, pages: Sequence[PageLayout], catalog_lang: str) -> None:
        """OCR các trang scan song song: luồng chính render (PyMuPDF không an toàn đa luồng), mỗi luồng phụ
        chạy một tiến trình tesseract 1 luồng. Nhanh gấp ~5 lần để Tesseract tự chia luồng trên từng trang."""
        pending = [p for p in pages if p.needs_ocr]
        if not pending:
            return
        if not self.ocr.is_available:
            for layout in pending:
                self._keep_scan_image(doc[layout.number - 1], layout)
            return
        self.ocr_lang = self._choose_ocr_lang(doc, pages, pending, catalog_lang)
        workers = max(1, min(len(pending), os.cpu_count() or 2))
        single_thread = len(pending) > 1
        total = len(pending)
        completed = 0
        inflight: Dict[object, Tuple[PageLayout, Image.Image, int]] = {}

        def collect() -> None:
            nonlocal completed
            finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for future in finished:
                layout, image, dpi = inflight.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"[OCR] trang {layout.number}: {exc}")
                    result = None
                self.ocr_repairs += getattr(result, "repaired", 0) if result is not None else 0
                self._apply_ocr(doc, layout, result, image, dpi)
                completed += 1
                self.progress(0.3 + 0.4 * completed / total, f"Nhận dạng chữ (OCR) trang {completed}/{total}")

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ocr") as pool:
            for layout in pending:
                page = doc[layout.number - 1]
                dpi = self._ocr_dpi(page)
                try:
                    image = self._render_for_ocr(page, dpi)
                except Exception:
                    self._keep_scan_image(page, layout)
                    completed += 1
                    continue
                future = pool.submit(self._ocr_page, image, dpi, single_thread)
                inflight[future] = (layout, image, dpi)
                while len(inflight) >= workers * 2:
                    collect()
            while inflight:
                collect()

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

    def _is_body_size(self, size: float, ocr: bool = False) -> bool:
        # Cỡ chữ OCR (x_size) dao động theo chữ trong dòng: "màu xám." chỉ có chữ thấp nên bị ước lượng 9,9pt
        # khi thân bài 12pt. Trang có lớp chữ thì cỡ chữ chính xác, giữ ngưỡng chặt.
        return abs(size - self.body_size) <= (0.22 if ocr else 0.12) * self.body_size

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

    @staticmethod
    def _split_margin_lines(pages: Sequence[PageLayout]) -> None:
        """Tách dòng đầu/cuối của khối nếu nó nằm ở lề trên/dưới và cách phần còn lại một khoảng trống.

        Tesseract hay gộp đầu trang ("46 | TƯ DUY ĐẶT CƯỢC") vào cùng đoạn với các dòng thân bài ngay dưới,
        khiến bộ lọc đầu trang (chỉ xét khối ngắn) bỏ sót và dòng đầu trang bị nối vào giữa câu.
        """
        for page in pages:
            top, bottom = page.height * 0.12, page.height * 0.88
            new_items: list = []
            for item in page.items:
                if not isinstance(item, TextBlock) or len(item.lines) < 2:
                    new_items.append(item)
                    continue
                lines = list(item.lines)
                head = tail = None
                first, second = lines[0], lines[1]
                height = max(1.0, first.y1 - first.y0)
                if first.y1 <= top and second.y0 - first.y1 >= 0.5 * height:
                    head = lines.pop(0)
                if len(lines) >= 2:
                    last, before = lines[-1], lines[-2]
                    height = max(1.0, last.y1 - last.y0)
                    if last.y0 >= bottom and last.y0 - before.y1 >= 0.5 * height:
                        tail = lines.pop()
                if head:
                    new_items.append(TextBlock([head], ocr=item.ocr))
                new_items.append(TextBlock(lines, ocr=item.ocr))
                if tail:
                    new_items.append(TextBlock([tail], ocr=item.ocr))
            page.items = new_items

    def _remove_running_heads(self, pages: Sequence[PageLayout]) -> int:
        """Bỏ đầu trang/chân trang lặp lại và số trang. Trả về số khối đã bỏ.

        So khớp gần đúng theo "khung chữ" (bỏ dấu, bỏ số, bỏ ký hiệu): trên trang scan, OCR đọc cùng một đầu
        trang thành "TƯ DUY ĐẶT CƯỢC", "TƯ DUY ĐẶT tƯỢt", "TƯ DUY BẬT CUýt"... nên so khớp chính xác bỏ sót.
        """

        def margin_key(text: str) -> str:
            key = re.sub(r"\d+", "#", text.lower())
            key = re.sub(r"\s+", " ", key)
            return key.strip(" .·•|-–—")

        def skeleton(text: str) -> str:
            text = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
            return "".join(ch for ch in text.lower() if "a" <= ch <= "z")

        self._split_margin_lines(pages)
        candidates = []  # (page, block, zone, text, skeleton)
        for page in pages:
            for item in page.items:
                if not isinstance(item, TextBlock):
                    continue
                x0, y0, x1, y1 = item.bbox
                in_margin = y1 <= page.height * 0.12 or y0 >= page.height * 0.88
                if not in_margin or len(item.lines) > 2:
                    continue
                text = item.text.strip()
                if not text or len(text) > 150 or item.size > self.body_size * 1.5:
                    continue
                zone = "top" if y1 <= page.height * 0.12 else "bottom"
                candidates.append((page, item, zone, text, skeleton(text)))

        # Gom cụm theo khung chữ giống nhau >= 75%.
        clusters: List[dict] = []
        membership: Dict[int, dict] = {}
        for index, (page, item, zone, text, skel) in enumerate(candidates):
            if len(skel) < 3:
                continue
            found = None
            for cluster in clusters:
                if cluster["zone"] == zone and abs(len(cluster["skel"]) - len(skel)) <= max(3, len(skel) // 3):
                    if difflib.SequenceMatcher(None, cluster["skel"], skel, autojunk=False).ratio() >= 0.75:
                        found = cluster
                        break
            if found is None:
                found = {"zone": zone, "skel": skel, "pages": set()}
                clusters.append(found)
            found["pages"].add(page.number)
            membership[index] = found

        removed = 0
        total_pages = max(1, len(pages))
        for index, (page, item, zone, text, skel) in enumerate(candidates):
            size_limit = 1.5 if item.ocr else 1.15
            bare = margin_key(text)
            is_page_number = bool(_PAGE_NUMBER_RE.match(bare)) and item.size <= self.body_size * 1.5
            # "46 | TƯ DUY ĐẶT CƯỢC", "MUỐN CƯỢC KHÔNG? | 47": số trang (OCR có thể đọc thành "I0", "?8") cạnh vạch ngăn.
            is_folio_head = bool(_FOLIO_HEAD_RE.search(text)) and len(text.split()) <= 20
            cluster = membership.get(index)
            numbers = sorted(cluster["pages"]) if cluster else []
            adjacent = sum(1 for a, b in zip(numbers, numbers[1:]) if b - a <= 2)
            is_running = (
                len(numbers) >= 3 and item.size <= self.body_size * size_limit
                and (adjacent >= 2 or len(numbers) >= 0.25 * total_pages)
            )
            if is_page_number or is_folio_head or is_running:
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
        # Khối giống trang "Mục lục" in sẵn: nhiều dòng kết thúc bằng số trang căn phải (dòng nào cũng "đầy").
        page_refs = sum(1 for l in lines if _PAGE_REF_END_RE.search(l.text))
        toc_like = page_refs >= 3 and page_refs >= 0.4 * len(lines)
        for prev, line in zip(lines, lines[1:]):
            same_row = abs(line.y0 - prev.y0) < 0.4 * em
            indent = line.x0 - m.left
            prev_indent = prev.x0 - m.left
            prev_short = prev.x1 < m.right - short_limit
            new_paragraph = False
            hanging = (
                prev.text.lstrip().startswith(("•", "-", "–", "*")) and not prev_short
                and 0 < line.x0 - prev.x0 <= 3 * em
            )
            if same_row or hanging:
                new_paragraph = False
            elif line.text.lstrip().startswith("•") or (toc_like and _PAGE_REF_END_RE.search(prev.text)):
                new_paragraph = True
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

    @staticmethod
    def _continuation_ratio(lines: Sequence[Line]) -> float:
        """Tỷ lệ dòng (trừ dòng đầu) mở đầu bằng chữ thường: câu chạy tiếp từ dòng trên, kiểu văn xuôi."""
        if len(lines) < 2:
            return 0.0
        lower = 0
        for line in lines[1:]:
            first = next((ch for ch in line.text if ch.isalpha()), "")
            lower += first.islower()
        return lower / (len(lines) - 1)

    def _ragged_prose(self, lines: Sequence[Line]) -> bool:
        """Văn xuôi canh trái, lề phải không đều (bìa sau, lời giới thiệu): dòng nào cũng gần kín bề rộng khối và
        câu chạy tiếp sang dòng sau. Không phải thơ, không giữ xuống dòng."""
        if len(lines) < 3:
            return False
        left, right = min(l.x0 for l in lines), max(l.x1 for l in lines)
        width = max(1.0, right - left)
        body = lines[:-1]
        wide = sum(1 for l in body if l.x1 >= right - 0.2 * width)
        return wide >= 0.7 * len(body) and self._continuation_ratio(lines) >= 0.5

    def _is_verse(self, lines: Sequence[Line], m: _Metrics) -> bool:
        if len(lines) < 3 or self._ragged_prose(lines):
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
            body=self._is_body_size(size, ocr=ocr),
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
        if (
            ocr and len(lines) == 1 and size >= 1.12 * self.body_size and len(visible.split()) <= 12
            and not _ends_sentence(visible)
        ):
            unit.heading_like = True  # OCR không có chữ đậm: dòng to hơn thân bài, đứng riêng
        return unit

    def _block_units(self, page: PageLayout, segment: Tuple[int, int], block: TextBlock, m: _Metrics, style: _Style) -> List[_Unit]:
        lines = block.lines
        if not lines:
            return []
        align = self._alignment(lines, m)
        body = self._is_body_size(block.size, ocr=block.ocr)
        if not body:
            return [self._make_unit(page, segment, lines, m, style, align=align, ocr=block.ocr)]
        if align == "center" and len(lines) > 1:
            # Đoạn văn căn giữa (câu chạy qua nhiều dòng) thì không giữ xuống dòng như thơ.
            prose = len(lines) >= 3 and self._continuation_ratio(lines) >= 0.6
            return [self._make_unit(page, segment, lines, m, style, align="center", verse=not prose, ocr=block.ocr)]
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

    def _mark_footnote_continuations(self, page: PageLayout, page_units: List[_Unit]) -> None:
        """Chú thích không mở đầu bằng số/dấu sao nhưng vẫn là chú thích (trang scan):

        - dòng tiếp theo của một chú thích bị tách thành khối riêng ("* Pinball là ..." + "mặt bàn nghiêng (ND).");
        - phần còn lại của chú thích trang trước, nằm cuối trang này, chữ nhỏ hẳn so với thân bài ngay trên nó
          ("chúng từ cuốn sách ... Richard Lederer.").
        Không nhận ra thì khối chữ nhỏ đó chen vào giữa đoạn văn bị ngắt trang.
        """
        texts = [u for u in page_units if u.kind == "text"]
        for index, unit in enumerate(texts):
            if unit.footnote or not unit.ocr:
                continue
            previous = texts[index - 1] if index else None
            small = unit.size <= 0.88 * self.body_size
            below = unit.bbox[1] >= page.height * 0.55
            if not (small and below):
                continue
            if previous is not None and previous.footnote and unit.bbox[1] >= previous.bbox[1]:
                unit.footnote = True
                continue
            rest = texts[index + 1:]
            if (
                previous is not None and previous.size >= 0.92 * self.body_size and not previous.footnote
                and all(u.size <= 0.92 * self.body_size or u.footnote for u in rest)
                and unit.para.visible().strip()[:1].islower()
            ):
                unit.footnote = True

    def _continues(self, prev: _Unit, unit: _Unit, style: _Style) -> bool:
        prev_text = prev.para.visible().rstrip()
        next_text = unit.para.visible().lstrip()
        if not prev_text or not next_text or not prev.last_full:
            return False
        if prev.quote != unit.quote and not (
            prev.segment != unit.segment and prev.last_in_segment and unit.first_in_segment
            and continues_sentence(prev_text, next_text)
        ):
            return False  # (mục danh sách thụt lề bị ngắt sang trang sau thì vẫn nối)
        if next_text.startswith("•"):
            return False  # mục danh sách mới
        if prev.segment == unit.segment:
            # Dòng có chỉ số chú thích cao hơn dòng thường: đo khoảng trống từ đáy dòng trước.
            if unit.first_y0 - prev.last_y0 > self.pitch * 1.35 and unit.first_y0 - prev.last_y1 > self.pitch * 0.5:
                return False
            if continues_sentence(prev_text, next_text):
                return True  # dòng thụt lề treo của mục danh sách, dòng cuối đoạn bị tách riêng
            if style.indent:
                return not unit.indented
            return continues_sentence(prev_text, next_text) or not _ends_sentence(prev_text)
        if not (prev.last_in_segment and unit.first_in_segment):
            return False
        if continues_sentence(prev_text, next_text):
            return True
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
        last_note: Optional[int] = None
        last_small: Optional[int] = None

        def small_flow(u: _Unit) -> bool:
            # Chữ nhỏ hơn thân bài nhưng vẫn là văn xuôi (bìa sau, lời khen): nối các dòng cùng cỡ trên một trang.
            return (
                u.kind == "text" and u.ocr and not u.body and not u.align and not u.verse and not u.scene
                and not u.heading_level and not u.heading_like and not u.footnote and not u.toc_entry
            )

        for unit in units:
            # Chữ nhỏ: dòng sau có thể được ước lượng cỡ khác một chút (8,9 / 9,4pt) nên lúc là "thân bài", lúc không.
            flow_prev = last_small if last_small is not None else (last_body if unit.ocr else None)
            if (small_flow(unit) or (unit.joinable and unit.ocr)) and flow_prev is not None and (
                small_flow(unit) or small_flow(out[flow_prev])
            ):
                prev = out[flow_prev]
                if (
                    prev.segment == unit.segment and abs(prev.size - unit.size) <= 0.12 * max(prev.size, unit.size)
                    and prev.last_full and unit.first_y0 - prev.last_y1 <= self.pitch
                    and continues_sentence(prev.para.visible().rstrip(), unit.para.visible().lstrip())
                ):
                    prev.para.extend(unit.para, self.vocab, self.hyphen_vocab)
                    prev.last_full = unit.last_full
                    prev.last_y0, prev.last_y1 = unit.last_y0, unit.last_y1
                    prev.line_count += unit.line_count
                    continue
            if unit.kind == "text" and unit.footnote:
                text = unit.para.visible().strip()
                if last_note is not None and text[:1].islower() and not _FOOTNOTE_START_RE.match(text):
                    # Phần tiếp của chú thích (dòng bị tách, hoặc vắt sang trang sau): nối vào chú thích trước.
                    out[last_note].para.extend(unit.para, self.vocab, self.hyphen_vocab)
                    continue
                out.append(unit)
                last_note = len(out) - 1
                continue
            if (
                unit.kind == "text" and unit.ocr and not unit.joinable and unit.line_count == 1 and last_body is not None
                and not unit.align and not unit.heading_level and len(unit.para.visible().split()) <= 3
                and continues_sentence(out[last_body].para.visible().rstrip(), unit.para.visible().lstrip())
                and out[last_body].last_full
            ):
                # Mảnh cuối câu chỉ có chữ thấp ("sau:"): OCR đoán cỡ chữ quá nhỏ nên không coi là thân bài.
                unit.body = True
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
            last_small = len(out) - 1 if small_flow(unit) or (unit.joinable and unit.ocr and unit.size < 0.85 * self.body_size) else None
        return out

    # ---- Mục lục in sẵn và tiêu đề (sách scan) -----------------------------------------------------

    @staticmethod
    def _set_text(unit: _Unit, text: str) -> None:
        unit.para = _Para()
        unit.para._push(text, frozenset())

    @staticmethod
    def _junk_text(text: str) -> bool:
        """Mảnh OCR vô nghĩa (vạch trang trí, "\" ˆ LẠ", "IIIII"): không có chữ nào dài hơn 2 ký tự có nghĩa."""
        from app.services.ebook.ocr_repair import core_of, is_garbled

        tokens = text.split()
        if not tokens:
            return True
        for token in tokens:
            core = core_of(token)
            if len(core) <= 2 or is_garbled(token) or (len(set(core.lower())) == 1 and len(core) >= 3):
                continue
            return False
        return True

    def _repair_toc_and_headings(self, units: List[_Unit]) -> List[_Unit]:
        """Dọn trang "Mục lục" in sẵn và sửa tiêu đề chương/mục theo nó (xem toc_repair)."""
        from app.services.ebook.ocr_repair import core_of
        from app.services.ebook.toc_repair import (
            _CHAPTER_RE, best_entry, chapter_number, has_page_ref, merge_readings, parse_entry, similarity,
            split_merged,
        )
        from app.services.ebook.vi_spell import VietSyllableModel

        body = self.body_size
        # Mảnh OCR vô nghĩa đứng riêng: bỏ.
        units = [u for u in units if not (
            u.kind == "text" and u.ocr and not u.footnote and u.line_count <= 2
            and len(u.para.visible().strip()) <= 16 and self._junk_text(u.para.visible())
        )]
        toc_pages = {u.page for u in units if u.toc_entry}
        if not toc_pages:
            return units
        model = VietSyllableModel.shared()

        # 1. Trang mục lục: nối mục bị xuống dòng, tách mục dính nhau, bỏ số trang; mục lục không phải tiêu đề.
        out: List[_Unit] = []
        toc_units: List[Tuple[_Unit, object]] = []
        open_entry: Optional[_Unit] = None  # mục chưa có số trang: dòng sau có thể là phần còn lại của nó
        last_toc = -1
        for unit in units:
            if unit.kind != "text" or unit.page not in toc_pages or unit.footnote:
                out.append(unit)
                continue
            text = " ".join(unit.para.visible().split())
            if not unit.toc_entry and len(text) <= 20 and similarity(text.strip("| "), "MỤC LỤC") >= 0.6:
                self._set_text(unit, "MỤC LỤC")
                out.append(unit)
                continue
            previous = out[-1] if out else None
            if (
                open_entry is not None and previous is open_entry and previous.page == unit.page
                and (text[:1].islower() or text[:1] == "(") and chapter_number(text) is None
            ):
                # "“Một trăm cái bánh ... sữa lắc" + "sô cô la lớn”: ..." là một mục bị xuống dòng.
                text = previous.para.visible().strip() + " " + text
                out.pop()
                toc_units.pop()
            open_entry = None
            pieces = split_merged(text)
            for index, piece in enumerate(pieces):
                target = unit if index == 0 else copy.copy(unit)
                entry = parse_entry(piece)
                self._set_text(target, entry.full)
                target.toc_entry = True
                target.heading_like = False
                target.size = min(target.size, body)  # dòng mục lục không được nhận nhầm thành tiêu đề chương
                target.align = ""
                out.append(target)
                toc_units.append((target, entry))
                last_toc = len(out) - 1
            if not has_page_ref(pieces[-1]) and chapter_number(pieces[-1]) is None:
                open_entry = out[-1]
        if not toc_units:
            return out

        entries = [entry for _, entry in toc_units]
        chapters = {e.chapter: e for e in entries if e.chapter is not None}
        tops = [e for e in entries if e.chapter is None and (e.label or e.title.isupper())]
        sections = [e for e in entries if e.chapter is None and not e.label and not e.title.isupper()]
        toc_unit_of = {id(entry): unit for unit, entry in toc_units}

        def heading_candidate(unit: _Unit) -> bool:
            return (
                unit.kind == "text" and not unit.footnote and not unit.toc_entry and unit.line_count <= 4
                and (unit.heading_like or unit.size >= 1.25 * body)
            )

        # 2. Thân sách: tiêu đề chương, phần đầu/cuối sách, tiêu đề mục.
        result = out[: last_toc + 1]
        rest = out[last_toc + 1:]
        index = 0
        while index < len(rest):
            unit = rest[index]
            if not heading_candidate(unit):
                result.append(unit)
                index += 1
                continue
            text = " ".join(unit.para.visible().split())
            number = chapter_number(text)
            if number in chapters:
                entry = chapters[number]
                remainder = _CHAPTER_RE.sub("", text, count=1).strip(" |")
                gathered = remainder
                fragments: List[_Unit] = []
                j = index + 1
                while j < len(rest) and len(fragments) < 5:
                    other = rest[j]
                    if other.kind != "text" or other.page != unit.page or other.footnote:
                        break
                    other_text = " ".join(other.para.visible().split())
                    if self._junk_text(other_text):
                        fragments.append(other)
                        j += 1
                        continue
                    if not heading_candidate(other) or best_entry(other_text, sections) is not None:
                        break
                    joined = (gathered + " " + other_text).strip()
                    if entry.title and similarity(joined, entry.title) <= similarity(gathered, entry.title):
                        break
                    gathered = joined
                    fragments.append(other)
                    j += 1
                if entry.title and similarity(gathered, entry.title) >= 0.45:
                    title = merge_readings(entry.title, gathered, model)
                    self._set_text(unit, f"CHƯƠNG {number}")
                    result.append(unit)
                    holder = next((f for f in fragments if not self._junk_text(f.para.visible())), None)
                    if holder is None:
                        holder = copy.copy(unit)
                    self._set_text(holder, title)
                    holder.toc_entry = False
                    result.append(holder)
                    index = j
                    self.heading_fixes += 1
                    continue
                result.append(unit)
                index += 1
                continue
            top = best_entry(text.strip("| "), tops, threshold=0.6)
            if top is not None:
                if top.label and top.title:
                    # "DẪN NHẬP Vì sao đâu không phải..." -> "DẪN NHẬP" + tên phần.
                    self._set_text(unit, top.label)
                    result.append(unit)
                    holder = copy.copy(unit)
                    body_part = text.strip("| ")
                    if body_part.upper().startswith(top.label.upper()):
                        body_part = body_part[len(top.label):].strip(" |")
                    self._set_text(holder, merge_readings(top.title, body_part, model))
                    result.append(holder)
                else:
                    merged = merge_readings(top.full, text.strip("| "), model)
                    self._set_text(unit, merged)
                    result.append(unit)
                    toc_unit = toc_unit_of.get(id(top))
                    if toc_unit is not None:
                        self._set_text(toc_unit, merged)  # sửa luôn dòng mục lục ("LỒI CẢM ƠN")
                index += 1
                self.heading_fixes += 1
                continue
            section = best_entry(text, sections)
            if section is not None and len(section.title.split()) == len(text.split()):
                merged = merge_readings(section.title, text, model)
                if merged != text:
                    self._set_text(unit, merged)
                    self.heading_fixes += 1
                toc_unit = toc_unit_of.get(id(section))
                if toc_unit is not None and merged != section.title:
                    self._set_text(toc_unit, merged)
            result.append(unit)
            index += 1
        return self._split_toc_by_headings(result, last_toc, heading_candidate)

    def _split_toc_by_headings(self, units: List[_Unit], last_toc: int, is_heading) -> List[_Unit]:
        """Dòng mục lục gộp hai mục mà không có số trang ở giữa ("... thầy bói nói dựa Sự nguy hiểm của resulting"):
        tách theo tiêu đề mục có thật trong thân sách."""
        from app.services.ebook.toc_repair import similarity

        headings = [" ".join(u.para.visible().split()) for u in units[last_toc + 1:] if is_heading(u)]
        headings = [h for h in headings if len(h.split()) >= 2]
        out: List[_Unit] = []
        for position, unit in enumerate(units):
            text = " ".join(unit.para.visible().split()) if unit.kind == "text" else ""
            if position > last_toc or not unit.toc_entry or "|" in text:
                out.append(unit)  # "CHƯƠNG 1 | ..." / "DẪN NHẬP | ..." là một mục có nhãn, không tách
                continue
            words = text.split()
            split_at = None
            for cut in range(2, len(words) - 1):
                head, tail = " ".join(words[:cut]), " ".join(words[cut:])
                if not tail[:1].isupper():
                    continue
                if any(similarity(head, h) >= 0.9 for h in headings) and any(similarity(tail, h) >= 0.85 for h in headings):
                    split_at = cut
                    break
            if split_at is None:
                out.append(unit)
                continue
            self._set_text(unit, " ".join(words[:split_at]))
            second = copy.copy(unit)
            self._set_text(second, " ".join(words[split_at:]))
            out += [unit, second]
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
        # Mục lục 1-2 mục (calibre hay ghi "Bìa", "Mục lục") không đủ làm khung chương.
        return matched >= max(3, 0.5 * len(entries))

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
                    self.progress(0.05 + 0.25 * (index + 1) / max(1, page_count), f"Đọc trang {index + 1}/{page_count}")

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
                if cover_bytes and (big or first.needs_ocr) and first.char_count < 80:
                    skip_first_page = True  # trang 1 chính là ảnh bìa: không lặp lại trong nội dung
            self._run_ocr(doc, pages, catalog_lang)
            if skip_first_page and pages[0].ocr_used and pages[0].char_count >= 300:
                # Trang 1 là trang scan đầy chữ (không phải bìa): giữ nội dung, ảnh bìa vẫn lấy từ trang này.
                skip_first_page = False
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
        # Sửa dấu sau khi bỏ đầu trang: đầu trang lặp lại trên mọi trang ("MỘT VÁN POKER" đọc thành "VẤN POKER")
        # sẽ làm sai thống kê của chính cuốn sách mà bộ sửa dựa vào.
        self._spell_check_ocr(pages)
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
            # Trang "Mục lục" in sẵn: số trang căn phải làm dòng nào cũng "đầy" nên các mục bị nối thành đoạn.
            refs = [u for u in page_units if u.kind == "text" and _PAGE_REF_END_RE.search(u.para.visible())]
            if len(refs) >= 3:
                for unit in refs:
                    unit.toc_entry = True
            # Chú thích cuối trang: chữ nhỏ ở phần dưới trang, mở đầu bằng số/dấu sao.
            for unit in page_units:
                if (
                    unit.kind == "text" and unit.size <= 0.92 * self.body_size
                    and unit.bbox[1] >= page.height * 0.6
                    and _FOOTNOTE_START_RE.match(unit.para.visible().strip() or "")
                ):
                    unit.footnote = True
            self._mark_footnote_continuations(page, page_units)
            by_segment: Dict[Tuple[int, int], List[_Unit]] = defaultdict(list)
            for unit in page_units:
                if not unit.footnote:
                    by_segment[unit.segment].append(unit)
            for seg_units in by_segment.values():
                seg_units[0].first_in_segment = True
                seg_units[-1].last_in_segment = True
            units.extend(page_units)

        if any(u.ocr for u in units if u.kind == "text"):
            units = self._repair_toc_and_headings(units)
        outline_used = self._apply_outline(units, outline)
        units = self._join_units(units, style)
        self.progress(0.8, "Dựng đoạn văn")

        # Siêu dữ liệu.
        title = (
            self.title_override or self._clean_title(meta.get("title", ""))
            or self._title_from_filename(Path(self.source_name) if self.source_name else pdf_path)
        )
        author = self.author_override or (meta.get("author") or "").strip()
        # Tên file chép từ macOS/điện thoại thường ở dạng tách dấu (NFD): dựng sẵn dấu để máy đọc hiện đúng.
        title = unicodedata.normalize("NFC", title)
        author = unicodedata.normalize("NFC", author)
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
            visible = strip_markers(text).strip()
            if not visible:
                continue
            font_size = unit.size
            ratio = unit.size / max(self.body_size, 1.0)
            if (
                unit.ocr and unit.line_count == 1 and not unit.footnote and 1.12 <= ratio < 1.31
                and len(visible.split()) <= 12 and not _ends_sentence(visible)
            ):
                # OCR không cho biết chữ đậm và ước lượng cỡ chữ tiêu đề nghiêng thấp hơn thật (1,23 lần thay vì
                # 1,41): dòng đứng riêng, ngắn, to hơn thân bài thì báo đủ cỡ để StructureBuilder nhận là tiêu đề.
                font_size = self.body_size * 1.31
            block = {
                "block_id": block_id,
                "type": "text",
                "text": text,
                "bbox": unit.bbox,
                "font_size": round(font_size, 1),
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
            lang_label = {"vie": "tiếng Việt", "eng": "tiếng Anh"}.get(self.ocr_lang, self.ocr_lang)
            fixed_note = f", tự sửa {self.spell_fixes} lỗi dấu theo ngữ cảnh" if self.spell_fixes else ""
            if self.ocr_repairs:
                fixed_note += f", đọc lại {self.ocr_repairs} chữ in nghiêng/tiếng Anh"
            warnings.append(
                f"{ocr_pages} trang scan đã được nhận dạng chữ bằng Tesseract ({lang_label}{fixed_note}); "
                "chữ in nghiêng và tiêu đề phông trang trí có thể còn sai, nên đọc soát."
            )
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

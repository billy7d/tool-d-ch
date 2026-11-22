"""OCR bằng Tesseract, gọi thẳng tesseract.exe qua stdin/stdout.

- Không qua pytesseract: pytesseract ghi ảnh và kết quả ra file tạm (mặc định thư mục Temp trên ổ C:);
  ở đây ảnh đi vào stdin, hOCR đi ra stdout, không tạo file nào.
- Đọc hOCR thay vì TSV để có x_size (chiều cao chữ của từng dòng): bộ dựng ebook dùng nó phân biệt tiêu đề
  với thân bài như cỡ chữ của PDF có lớp chữ.
- Tesseract bản Windows không mở được tessdata nằm dưới đường dẫn có chữ có dấu ("E:\\tool-dịch\\..."), nên
  chạy với thư mục làm việc là thư mục của tesseract.exe và đường dẫn tessdata tương đối.
- Mỗi tiến trình chỉ dùng 1 luồng (OMP_THREAD_LIMIT=1) khi chạy song song nhiều trang: nhanh hơn hẳn để
  Tesseract tự chia luồng trên một trang.
"""
from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from app.config import settings

_BBOX_RE = re.compile(r"bbox (\d+) (\d+) (\d+) (\d+)")
_XSIZE_RE = re.compile(r"x_size ([\d.]+)")
_WCONF_RE = re.compile(r"x_wconf (\d+)")
_LINE_CLASSES = {"ocr_line", "ocr_header", "ocr_caption", "ocr_textfloat"}
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

_lang_cache: Dict[str, set] = {}
_lang_lock = threading.Lock()


@dataclass
class OcrLine:
    text: str
    bbox: Tuple[int, int, int, int]  # pixel
    x_size: float  # chiều cao chữ ước lượng (pixel)
    confidence: float  # 0..1
    word_confidences: List[float] = field(default_factory=list)  # theo từng từ (cách nhau bởi khoảng trắng)
    word_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)  # pixel, theo từng từ
    # Theo từng từ, từng chữ cái: (điểm của chữ đã chọn, ((chữ thay thế, điểm), ...)) do LSTM của Tesseract
    # cân nhắc (lstm_choice_mode=2). Rỗng khi không bật chế độ đó.
    word_alternatives: List[tuple] = field(default_factory=list)


@dataclass
class OcrParagraph:
    lines: List[OcrLine] = field(default_factory=list)


@dataclass
class OcrPage:
    paragraphs: List[OcrParagraph]
    dpi: int

    @property
    def lines(self) -> List[OcrLine]:
        return [line for para in self.paragraphs for line in para.lines]

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def mean_confidence(self) -> float:
        lines = self.lines
        if not lines:
            return 0.0
        weights = [max(1, len(l.text)) for l in lines]
        return sum(l.confidence * w for l, w in zip(lines, weights)) / sum(weights)


def find_tesseract() -> Optional[Path]:
    """tools/tesseract (hoặc TOOL_DICH_TESSERACT) -> PATH -> thư mục cài mặc định trên Windows."""
    candidates: List[Path] = []
    configured = Path(settings.TESSERACT_PATH)
    candidates.append(configured / ("tesseract.exe" if sys.platform == "win32" else "tesseract") if configured.is_dir() else configured)
    on_path = shutil.which("tesseract")
    if on_path:
        candidates.append(Path(on_path))
    for root in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"), os.environ.get("LOCALAPPDATA")):
        if root:
            candidates.append(Path(root) / "Tesseract-OCR" / "tesseract.exe")
            candidates.append(Path(root) / "Programs" / "Tesseract-OCR" / "tesseract.exe")
    for path in candidates:
        if path and path.is_file():
            return path
    return None


def _parse_title(title: str) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[float], Optional[int]]:
    bbox = _BBOX_RE.search(title or "")
    xsize = _XSIZE_RE.search(title or "")
    wconf = _WCONF_RE.search(title or "")
    return (
        tuple(int(v) for v in bbox.groups()) if bbox else None,
        float(xsize.group(1)) if xsize else None,
        int(wconf.group(1)) if wconf else None,
    )


_CHOICE_CONF_RE = re.compile(r"x_confs ([\d.]+)")


def _word_chars(word) -> Tuple[str, tuple]:
    """Chữ của một từ hOCR và phương án thay thế từng chữ cái (khi có lstm_choice_mode=2).

    Ở chế độ đó mỗi chữ cái là một span "x_bboxes" theo sau là nhóm span "lstm_choices"; text_content() của cả
    từ khi đó lẫn cả chữ thay thế nên phải ghép lại từ các span chữ cái.
    """
    chars: List[str] = []
    alternatives: List[list] = []
    for child in word:
        title = child.get("title") or ""
        ident = child.get("id") or ""
        if title.startswith("x_bboxes"):
            chars.append(child.text or "")
            alternatives.append([])
        elif ident.startswith("lstm_choices") and alternatives:
            for choice in child:
                match = _CHOICE_CONF_RE.search(choice.get("title") or "")
                if match and choice.text:
                    alternatives[-1].append((choice.text, float(match.group(1))))
    if not chars:
        return (word.text_content() or "").strip(), ()
    text = "".join(chars)
    packed = []
    for char, choices in zip(chars, alternatives):
        own = max((conf for alt, conf in choices if alt == char), default=None)
        top = max((conf for _, conf in choices), default=0.0)
        others = {}
        for alt, conf in choices:
            if alt != char and conf >= 5.0 and conf > others.get(alt, 0.0):
                others[alt] = conf
        packed.append((round(own if own is not None else top, 1),
                       tuple(sorted(((a, round(c, 1)) for a, c in others.items()), key=lambda x: -x[1]))))
    if text.strip() != text:
        return text.strip(), ()
    return text, tuple(packed)


def parse_hocr(data: bytes, dpi: int = 300) -> OcrPage:
    from lxml import html as lxml_html

    if not data.strip():
        return OcrPage([], dpi)
    root = lxml_html.fromstring(data)
    paragraphs: List[OcrParagraph] = []
    for par in root.iter():
        if par.get("class") != "ocr_par":
            continue
        paragraph = OcrParagraph()
        for line in par.iter():
            if line.get("class") not in _LINE_CLASSES:
                continue
            bbox, x_size, _ = _parse_title(line.get("title", ""))
            words, confs, boxes, alternatives = [], [], [], []
            for word in line.iter():
                if word.get("class") != "ocrx_word":
                    continue
                text, alts = _word_chars(word)
                if not text or any(ch.isspace() for ch in text):
                    continue
                words.append(text)
                word_box, _, conf = _parse_title(word.get("title", ""))
                confs.append(conf if conf is not None else 50)
                boxes.append(word_box or bbox or (0, 0, 0, 0))
                alternatives.append(alts)
            if not words or not bbox:
                continue
            height = bbox[3] - bbox[1]
            paragraph.lines.append(OcrLine(
                text=" ".join(words),
                bbox=bbox,
                x_size=x_size or float(height),
                confidence=sum(confs) / len(confs) / 100.0,
                word_confidences=[c / 100.0 for c in confs],
                word_boxes=boxes,
                word_alternatives=alternatives if any(alternatives) else [],
            ))
        if paragraph.lines:
            paragraphs.append(paragraph)
    return OcrPage(paragraphs, dpi)


def encode_png(image: Image.Image) -> bytes:
    """PNG nén nhẹ cho Tesseract. Không dùng PNM/JPEG: trên Windows Leptonica đọc hai loại đó từ bộ nhớ bằng
    cách ghi ra file tạm (và lỗi khi thư mục tạm có chữ có dấu); PNG được giải mã thẳng trong bộ nhớ."""
    if image.mode not in ("L", "RGB"):
        image = image.convert("L" if image.mode in ("1", "LA", "I", "I;16", "F") else "RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


class OCREngine:
    def __init__(self, lang: str = "eng", executable: Optional[Path] = None):
        self.executable = Path(executable) if executable else find_tesseract()
        self.is_available = self.executable is not None
        self.lang = lang

    # ---- Cấu hình ---------------------------------------------------------------------------

    def _tessdata_args(self) -> Tuple[List[str], Optional[str]]:
        """(tham số tessdata, thư mục làm việc). Dùng đường dẫn tương đối để né lỗi đường dẫn có dấu."""
        exe_dir = self.executable.parent
        if (exe_dir / "tessdata").is_dir():
            return ["--tessdata-dir", "tessdata"], str(exe_dir)
        return [], None

    def _env(self, single_thread: bool) -> Dict[str, str]:
        env = dict(os.environ)
        if single_thread:
            env["OMP_THREAD_LIMIT"] = "1"
        temp_dir = settings.DATA_DIR / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        env["TMP"] = env["TEMP"] = env["TMPDIR"] = str(temp_dir)
        return env

    def available_languages(self) -> set:
        if not self.is_available:
            return set()
        key = str(self.executable)
        with _lang_lock:
            if key in _lang_cache:
                return _lang_cache[key]
        args, cwd = self._tessdata_args()
        try:
            completed = subprocess.run(
                [str(self.executable), "--list-langs", *args], capture_output=True, timeout=30, cwd=cwd,
                env=self._env(False), creationflags=_CREATE_NO_WINDOW,
            )
            out = (completed.stdout or b"").decode("utf-8", "replace").splitlines()
            langs = {line.strip() for line in out[1:] if line.strip()}
        except Exception:
            langs = set()
        with _lang_lock:
            _lang_cache[key] = langs
        return langs

    def resolve_lang(self, lang: Optional[str]) -> str:
        """Bỏ các ngôn ngữ chưa có dữ liệu ("eng+vie" khi thiếu vie -> "eng")."""
        wanted = [part for part in (lang or self.lang or "eng").split("+") if part]
        installed = self.available_languages()
        usable = [part for part in wanted if part in installed] if installed else wanted
        return "+".join(usable) or ("eng" if "eng" in installed or not installed else sorted(installed)[0])

    # ---- Nhận dạng --------------------------------------------------------------------------

    def recognize(self, image_bytes: bytes, lang: Optional[str] = None, dpi: int = 300,
                  single_thread: bool = False, timeout: int = 300, psm: int = 3,
                  alternatives: bool = False) -> OcrPage:
        """OCR một ảnh (nên là PNG, xem encode_png). Trả về các đoạn/dòng kèm toạ độ pixel.

        psm: 3 cho cả trang, 6 cho một khối dòng đã cắt sẵn, 7 cho một dòng.
        alternatives: lấy thêm phương án thay thế của từng chữ cái (chậm hơn ~10%, hOCR lớn hơn nhiều): bộ sửa lỗi
        dấu dùng để biết Tesseract đã phân vân giữa "ể" và "ề" hay chắc chắn.
        """
        if not self.is_available:
            return OcrPage([], dpi)
        args, cwd = self._tessdata_args()
        cmd = [str(self.executable), "stdin", "stdout", *args, "-l", self.resolve_lang(lang),
               "--psm", str(psm), "--dpi", str(dpi), "-c", "tessedit_create_hocr=1"]
        if alternatives:
            cmd += ["-c", "lstm_choice_mode=2", "-c", "hocr_char_boxes=1"]
        completed = subprocess.run(
            cmd, input=image_bytes, capture_output=True, timeout=timeout, cwd=cwd,
            env=self._env(single_thread), creationflags=_CREATE_NO_WINDOW,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or b"").decode("utf-8", "replace").strip()[-300:]
            raise RuntimeError(f"Tesseract lỗi (mã {completed.returncode}): {detail}")
        return parse_hocr(completed.stdout, dpi)

    def recognize_image(self, image: Image.Image, lang: Optional[str] = None, dpi: int = 300,
                        single_thread: bool = False, psm: int = 3, alternatives: bool = False) -> OcrPage:
        """Mã hóa PNG rồi OCR; gọi được từ luồng phụ (Pillow nhả GIL khi nén)."""
        return self.recognize(encode_png(image), lang=lang, dpi=dpi, single_thread=single_thread, psm=psm,
                              alternatives=alternatives)

    def ocr_image(self, image: Image.Image, dpi: int = 300) -> List[Dict[str, Any]]:
        """API cũ cho PDFExtractor: mỗi dòng một khối {text, confidence, bbox (pixel)}."""
        if not self.is_available:
            return []
        try:
            page = self.recognize_image(image, dpi=dpi)
        except Exception as e:
            print(f"OCR execution warning: {e}")
            return []
        return [
            {"text": line.text, "confidence": line.confidence, "bbox": list(line.bbox)}
            for line in page.lines
        ]

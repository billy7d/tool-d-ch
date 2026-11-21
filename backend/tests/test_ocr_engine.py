"""OCR trang scan bằng Tesseract cho bộ chuyển PDF -> ebook và luồng dịch.

Test đọc hOCR chạy mọi nơi; test OCR thật chỉ chạy khi có Tesseract (tools/tesseract hoặc trên PATH),
máy CI không cài nên tự bỏ qua.
"""
import io
import re
import zipfile
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from app.services.ebook.pdf_to_ebook import ConversionOptions, convert_pdf_to_ebook
from app.services.extractor.ocr_engine import OCREngine, parse_hocr
from app.services.extractor.pdf_extractor import PDFExtractor

HOCR = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en"><head><title></title></head><body>
<div class='ocr_page' id='page_1' title='image "stdin"; bbox 0 0 2480 3508; ppageno 0'>
 <div class='ocr_carea' id='block_1_1' title="bbox 300 300 1200 380">
  <p class='ocr_par' id='par_1_1' lang='vie' title="bbox 300 300 1200 380">
   <span class='ocr_header' id='line_1_1' title="bbox 300 300 1200 380; baseline 0 -10; x_size 80; x_descenders 15; x_ascenders 20">
    <span class='ocrx_word' id='word_1_1' title='bbox 300 300 700 380; x_wconf 96'>Chương</span>
    <span class='ocrx_word' id='word_1_2' title='bbox 740 300 1200 380; x_wconf 90'>Một</span>
   </span>
  </p>
 </div>
 <div class='ocr_carea' id='block_1_2' title="bbox 300 500 2100 620">
  <p class='ocr_par' id='par_1_2' lang='vie' title="bbox 300 500 2100 620">
   <span class='ocr_line' id='line_1_2' title="bbox 360 500 2100 555; baseline 0 -12; x_size 52; x_descenders 12; x_ascenders 14">
    <span class='ocrx_word' id='word_1_3' title='bbox 360 500 600 555; x_wconf 95'>Con</span>
    <span class='ocrx_word' id='word_1_4' title='bbox 620 500 900 555; x_wconf 85'>&#273;&#432;&#7901;ng</span>
   </span>
   <span class='ocr_line' id='line_1_3' title="bbox 300 565 1400 620; baseline 0 -12; x_size 51; x_descenders 12; x_ascenders 14">
    <span class='ocrx_word' id='word_1_5' title='bbox 300 565 1400 620; x_wconf 70'>d&#224;i.</span>
    <span class='ocrx_word' id='word_1_6' title='bbox 1400 565 1400 620; x_wconf 0'> </span>
   </span>
  </p>
 </div>
</div></body></html>""".encode("utf-8")


def test_parse_hocr_reads_paragraphs_lines_and_sizes():
    page = parse_hocr(HOCR, dpi=300)
    assert [len(p.lines) for p in page.paragraphs] == [1, 2]
    header, first, second = page.lines
    assert header.text == "Chương Một" and header.x_size == 80 and header.bbox == (300, 300, 1200, 380)
    assert first.text == "Con đường" and abs(first.confidence - 0.90) < 1e-9
    assert second.text == "dài."  # từ rỗng bị bỏ
    assert page.text.splitlines()[0] == "Chương Một"
    assert parse_hocr(b"").paragraphs == []


needs_tesseract = pytest.mark.skipif(not OCREngine().is_available, reason="máy chưa có Tesseract")

PARAGRAPHS = [
    "The road out of the valley was long and narrow, and the travellers who used it in winter spoke of it "
    "with a mixture of fear and affection. Every family in the village had made some investment in the bridge.",
    "Nobody remembered who had first called the crossing a test of patience, but the phrase stuck. Children "
    "learned it before they learned to read, and the ferryman repeated it to every stranger.",
]


def _scanned_pdf(path: Path) -> Path:
    """Trang chữ dàn như sách rồi chụp thành ảnh: PDF chỉ có ảnh, không lớp chữ."""
    source = pymupdf.open()
    for number in range(2):
        page = source.new_page(width=420, height=595)
        y = 70
        if number == 0:
            page.insert_text((50, y), "Chapter One", fontname="tibo", fontsize=20)
            y += 40
        for text in PARAGRAPHS:
            rect = pymupdf.Rect(50, y, 370, y + 150)
            page.insert_textbox(rect, "     " + text, fontname="tiro", fontsize=11, align=pymupdf.TEXT_ALIGN_JUSTIFY)
            y += 110
        page.insert_text((205, 570), str(number + 1), fontname="tiro", fontsize=9)
    scanned = pymupdf.open()
    for page in source:
        pix = page.get_pixmap(dpi=300, colorspace=pymupdf.csGRAY)
        buf = io.BytesIO()
        Image.frombytes("L", (pix.width, pix.height), pix.samples).save(buf, format="PNG")
        new = scanned.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=buf.getvalue())
    scanned.save(str(path))
    return path


@needs_tesseract
def test_engine_finds_languages_and_drops_missing_ones():
    engine = OCREngine()
    langs = engine.available_languages()
    assert "eng" in langs
    assert engine.resolve_lang("eng+xyz") == "eng"


@needs_tesseract
def test_scanned_pdf_becomes_reflowable_text(tmp_path):
    pdf = _scanned_pdf(tmp_path / "scan.pdf")
    assert sum(len(p.get_text()) for p in pymupdf.open(str(pdf))) == 0
    out = tmp_path / "scan.epub"
    result = convert_pdf_to_ebook(pdf, out, tmp_path / "work",
                                  ConversionOptions(output_format="epub", device="kindle", cover=False))
    assert result.ocr_pages == 2 and result.scanned_pages == 0
    assert result.language == "en"
    with zipfile.ZipFile(out) as zf:
        body = "\n".join(zf.read(n).decode("utf-8") for n in sorted(zf.namelist()) if n.startswith("OEBPS/text/part"))
        assert not any(n.startswith("OEBPS/images/") for n in zf.namelist())  # chữ thật, không phải ảnh trang
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body))
    assert re.search(r"<h[12][^>]*>Chapter One</h[12]>", body)
    assert "every family in the village had made some investment" in text.lower()
    assert "the ferryman repeated it to every stranger" in text.lower()
    # Số trang ở chân trang bị bỏ như với PDF có lớp chữ.
    assert not re.search(r"<p[^>]*>\s*[12]\s*</p>", body)


@needs_tesseract
def test_pdf_extractor_ocr_boxes_use_page_points(tmp_path):
    pdf = _scanned_pdf(tmp_path / "scan.pdf")
    data = PDFExtractor(tmp_path / "project").extract_document(pdf)
    blocks = [b for p in data["pages"] for b in p["blocks"] if b.get("is_ocr")]
    assert blocks and any("Chapter One" in b["text"] for b in blocks)
    assert all(0 <= b["bbox"][3] <= 595 + 1 and b["bbox"][2] <= 420 + 1 for b in blocks)

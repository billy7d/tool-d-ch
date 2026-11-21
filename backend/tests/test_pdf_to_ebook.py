"""Chuyển PDF -> EPUB/MOBI cho KOReader trên Kindle cũ.

Sách thử được dàn trang như sách in thật: căn đều hai bên, thụt đầu dòng, không cách đoạn, đầu trang/chân trang
có số trang, từ bị ngắt gạch nối cuối dòng, đoạn vắt qua trang, chú thích chữ nhỏ cuối trang, thơ căn giữa,
dấu ngắt cảnh, chữ nghiêng giữa câu, ảnh minh họa, trang bìa là ảnh và mục lục (outline) của PDF.
"""
import io
import re
import time
import unicodedata
import zipfile
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient
from lxml import etree
from PIL import Image, ImageDraw

from app.models.canonical import (
    Asset, CanonicalDocument, Chapter, DocumentMetadata, DocumentNode, NodeMetadata, NodeType,
)
from app.services.ebook.book_plan import build_book_plan
from app.services.ebook.devices import get_device
from app.services.ebook.mobi_writer import palmdoc_compress, palmdoc_decompress, write_mobi
from app.services.ebook.pdf_to_ebook import ConversionOptions, convert_pdf_to_ebook
from app.services.ebook.text_cleanup import detect_language, finalize_text, inline_markup
from app.services.exporter.epub_exporter import EPUBExporter
from app.services.exporter.mobi_exporter import MOBIExporter
from mobi_reader import filepos_targets, read_mobi

PAGE_W, PAGE_H = 420, 420
LEFT, RIGHT = 50, 370
TOP, BOTTOM = 70, 345
SIZE, LEADING, INDENT = 10, 13, 15

CHAPTER_ONE = [
    "The road out of the valley was long and narrow, and the travellers who used it in winter spoke of it with a "
    "mixture of fear and affection. Every family in the village had made some invest-\nment in the old bridge, and "
    "each spring the elders counted the planks again before the carts were allowed to cross the river.",
    "Nobody remembered who had first called the crossing a test of self-\ndiscipline, but the phrase stuck. Children "
    "learned it before they learned to read, and the ferryman repeated it to every stranger who asked why the "
    "toll was paid in labour rather than in coin.",
    "Self-discipline was not admired for its own sake. It was simply the price of living beside a river that "
    "changed its mind every season, and the people who forgot it rarely stayed long enough to be remembered by "
    "anyone in the valley.",
    "In the third year of the drought the river fell so low that the old stones of an earlier bridge appeared "
    "beneath the water, black and smooth and covered with marks that nobody could read. The schoolmaster drew "
    "them carefully in his notebook, the priest declared them a warning, and the miller, who had no patience "
    "for either of them, used the dry season to repair his wheel. By the time the rains returned every house "
    "had its own story about the stones, and the stories grew longer each time they were told at the market "
    "on the first morning of the month, when the whole valley came down to trade and to argue.",
    "It was during one of those arguments that the stranger arrived. She carried a leather case and a folded "
    "map, and she asked the way to the bridge in a voice so quiet that the children had to lean close to hear "
    "her. Nobody knew then that she had come to measure the river.",
]
CHAPTER_TWO = [
    "The measurements took the whole summer. Every morning she walked the bank with a long pole marked in red "
    "and white, and every evening she wrote the numbers into a book bound in green cloth.",
    "When the work was done she showed the drawings to the elders, who looked at them for a long time without "
    "speaking. Then the oldest of them nodded and said that the bridge would stand for another hundred years.",
]
VERSE = ["Over the water and under the stone,", "the road goes on", "and we are never alone."]
FOOTNOTE = "1 The bridge was rebuilt twice in the following century."


def _font_for(word: str) -> str:
    return "tiit" if word.startswith("_") else "tiro"


def _clean(word: str) -> str:
    return word.strip("_")


def _wrap(text: str, width: float, first_indent: float):
    """Ngắt dòng tham lam; '\n' là chỗ ngắt cưỡng bức (từ bị ngắt gạch nối). Trả về [(từ, căn đều?)]."""
    lines = []
    chunks = text.split("\n")
    for chunk_index, chunk in enumerate(chunks):
        words = chunk.split()
        current = []
        for word in words:
            avail = width - first_indent if not lines else width
            trial = " ".join(_clean(w) for w in current + [word])
            if current and pymupdf.get_text_length(trial, fontname="tiro", fontsize=SIZE) > avail:
                lines.append((current, True))
                current = [word]
            else:
                current.append(word)
        lines.append((current, chunk_index < len(chunks) - 1))
    return lines


class _Typesetter:
    def __init__(self, doc):
        self.doc = doc
        self.page = None
        self.y = 0.0
        self.page_number = 1  # trang bìa là trang 1

    def new_page(self):
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self.page_number += 1
        self.page.insert_text((165, 38), "THE RIVER BOOK", fontname="tiro", fontsize=8)
        self.page.insert_text((206, PAGE_H - 20), str(self.page_number), fontname="tiro", fontsize=8)
        self.y = TOP

    def ensure(self, height):
        if self.page is None or self.y + height > BOTTOM:
            self.new_page()

    def heading(self, text):
        self.new_page()
        self.y += 10
        self.page.insert_text((LEFT, self.y), text, fontname="tibo", fontsize=18)
        self.y += 34

    def paragraph(self, text):
        for index, (words, justify) in enumerate(_wrap(text, RIGHT - LEFT, INDENT)):
            self.ensure(LEADING)
            x = LEFT + (INDENT if index == 0 else 0)
            widths = [pymupdf.get_text_length(_clean(w), fontname=_font_for(w), fontsize=SIZE) for w in words]
            space = pymupdf.get_text_length(" ", fontname="tiro", fontsize=SIZE)
            if justify and len(words) > 1:
                space = (RIGHT - x - sum(widths)) / (len(words) - 1)
            for word, width in zip(words, widths):
                self.page.insert_text((x, self.y), _clean(word), fontname=_font_for(word), fontsize=SIZE)
                x += width + space
            self.y += LEADING

    def centered(self, lines):
        self.y += 6
        for line in lines:
            self.ensure(LEADING)
            width = pymupdf.get_text_length(line, fontname="tiro", fontsize=SIZE)
            self.page.insert_text(((LEFT + RIGHT) / 2 - width / 2, self.y), line, fontname="tiro", fontsize=SIZE)
            self.y += LEADING
        self.y += 6

    def footnote(self, text):
        self.page.insert_text((LEFT, BOTTOM + 25), text, fontname="tiro", fontsize=7)

    def image(self, png: bytes, height=110):
        self.ensure(height + 10)
        self.page.insert_image(pymupdf.Rect(110, self.y, 310, self.y + height), stream=png)
        self.y += height + 12


def _png(size=(400, 220), seed=0) -> bytes:
    image = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for i in range(0, size[0], 4):
        draw.line([(i, 0), (i, size[1])], fill=((i * 3 + seed) % 255, (i * 7) % 255, 120))
    draw.ellipse([60, 40, 200, 180], fill=(30, 30, 30))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def make_book_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    cover = doc.new_page(width=PAGE_W, height=PAGE_H)
    cover.insert_image(cover.rect, stream=_png((420, 420), seed=40))
    ts = _Typesetter(doc)

    chapter_one = [p.replace("in the village", "in the _village") for p in CHAPTER_ONE]
    ts.heading("Chapter One")
    for index, text in enumerate(chapter_one):
        ts.paragraph(text)
        if index == 1:
            ts.footnote(FOOTNOTE)
    ts.centered(["* * *"])
    ts.paragraph("The stranger stayed through the winter.")

    ts.heading("Chapter Two")
    ts.paragraph(CHAPTER_TWO[0])
    ts.image(_png(seed=7))
    ts.paragraph(CHAPTER_TWO[1])
    ts.centered(VERSE)

    doc.set_toc([[1, "Chapter One", 2], [1, "Chapter Two", ts.page_number]])
    doc.set_metadata({"title": "The River Book", "author": "Ada Writer"})
    doc.save(str(path))
    doc.close()
    return path


def _plain(fragment: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", fragment)).strip()


def _epub_body(zf: zipfile.ZipFile) -> str:
    return "\n".join(zf.read(n).decode("utf-8") for n in sorted(zf.namelist()) if n.startswith("OEBPS/text/part"))


@pytest.fixture(scope="module")
def book_pdf(tmp_path_factory) -> Path:
    return make_book_pdf(tmp_path_factory.mktemp("pdfbook") / "river.pdf")


@pytest.fixture(scope="module")
def epub_file(book_pdf, tmp_path_factory) -> tuple:
    out_dir = tmp_path_factory.mktemp("epub")
    result = convert_pdf_to_ebook(book_pdf, out_dir / "river.epub", out_dir / "work",
                                  ConversionOptions(output_format="epub", device="kindle"))
    return out_dir / "river.epub", result


def test_epub_container_is_valid_for_old_readers(epub_file):
    path, result = epub_file
    with zipfile.ZipFile(path) as zf:
        first = zf.infolist()[0]
        assert first.filename == "mimetype" and first.compress_type == zipfile.ZIP_STORED
        assert zf.read("mimetype") == b"application/epub+zip"
        for name in zf.namelist():
            if name.endswith((".xhtml", ".opf", ".ncx", ".xml")):
                etree.fromstring(zf.read(name))  # XHTML phải là XML hợp lệ, crengine không tự sửa
        opf = zf.read("OEBPS/content.opf").decode("utf-8")
        assert "<dc:title>The River Book</dc:title>" in opf
        assert "<dc:creator id=\"creator\">Ada Writer</dc:creator>" in opf
        assert "<dc:language>en</dc:language>" in opf
        assert 'properties="cover-image"' in opf and '<meta name="cover" content="cover-image"/>' in opf
        assert 'href="toc.ncx"' in opf and 'properties="nav"' in opf
        css = zf.read("OEBPS/styles/book.css").decode("utf-8")
        # Không ép font/cỡ chữ tuyệt đối để KOReader tùy chỉnh được.
        assert "font-family" not in css.replace("font-family: monospace", "")
        assert not re.search(r"\d(pt|px)\b", css.replace("1px solid", ""))
    # Mục lục PDF chỉ 2 mục không đủ làm khung chương (calibre hay ghi "Bìa", "Mục lục"): chương lấy theo tiêu đề,
    # mục lục điều hướng vẫn đúng hai chương (xem test_epub_keeps_layout_details).
    assert result.outline_used is False
    assert result.paragraph_style == "indent"


def test_epub_rebuilds_paragraphs_like_the_printed_book(epub_file):
    path, _ = epub_file
    with zipfile.ZipFile(path) as zf:
        body = _epub_body(zf)
    paragraphs = [_plain(p) for p in re.findall(r"<p(?: class=\"[^\"]*\")?>(.*?)</p>", body, re.S)]
    text = " ".join(paragraphs)

    # Đầu trang và số trang bị bỏ.
    assert "THE RIVER BOOK" not in text
    assert not any(re.fullmatch(r"\d{1,3}", p) for p in paragraphs)
    # Ngắt gạch nối cuối dòng: bỏ ở từ thường, giữ ở từ ghép có thật trong sách.
    assert "investment in the old bridge" in text and "invest-" not in text
    assert "a test of self-discipline" in text
    # Mỗi đoạn gốc là đúng một <p>, kể cả đoạn vắt qua trang.
    for source in CHAPTER_ONE + CHAPTER_TWO:
        expected = _plain(source.replace("invest-\nment", "investment").replace("self-\ndiscipline", "self-discipline"))
        assert expected in paragraphs, expected[:60]
    assert not any(p[:1].islower() for p in paragraphs)
    # Chữ nghiêng giữa câu được giữ.
    assert "<i>village</i>" in body


def test_epub_keeps_layout_details(epub_file):
    path, _ = epub_file
    with zipfile.ZipFile(path) as zf:
        body = _epub_body(zf)
        names = zf.namelist()
        nav = zf.read("OEBPS/nav.xhtml").decode("utf-8")
        ncx = zf.read("OEBPS/toc.ncx").decode("utf-8")
    assert '<p class="scene">* * *</p>' in body
    assert '<p class="verse center">' + "<br/>".join(VERSE) + "</p>" in body
    # Chú thích chữ nhỏ cuối trang dời về cuối chương, không chen giữa hai nửa đoạn văn.
    assert '<div class="footnotes"><p class="footnote">' + FOOTNOTE + "</p></div>" in body
    # Ảnh minh họa nhúng thật trong EPUB, ảnh bìa không lặp lại trong nội dung.
    images = [n for n in names if n.startswith("OEBPS/images/img")]
    assert len(images) == 1
    assert f'src="../images/{Path(images[0]).name}"' in body
    assert "OEBPS/images/cover.jpg" in names
    # Mục lục lấy từ outline của PDF, mỗi chương một file (KOReader mở trang mới).
    assert re.findall(r'<a href="[^"]+">([^<]+)</a>', nav)[:2] == ["Chapter One", "Chapter Two"]
    assert ncx.count("<navPoint") == 2
    assert len([n for n in names if n.startswith("OEBPS/text/part")]) == 2


def test_mobi_conversion_is_readable_mobi6(book_pdf, tmp_path):
    out = tmp_path / "river.mobi"
    result = convert_pdf_to_ebook(book_pdf, out, tmp_path / "work",
                                  ConversionOptions(output_format="mobi", device="kindle", mobi_engine="builtin"))
    assert result.engine == "builtin"
    book = read_mobi(out.read_bytes())
    assert book.compression == 2 and book.file_version == 6 and book.encoding == 65001
    assert book.text_length == len(book.text)
    html = book.text.decode("utf-8")
    assert "investment in the old bridge" in html
    assert "<mbp:pagebreak />" in html
    assert book.full_name == "The River Book"
    assert book.exth[100] == [b"Ada Writer"] and book.exth[524] == [b"en"]
    targets = filepos_targets(book.text)
    assert targets and all(book.text[t:t + 1] == b"<" for t in targets)
    images = book.records[book.first_image_index:book.first_image_index + 3]
    assert all(r[:3] == b"\xff\xd8\xff" and len(r) <= 127 * 1024 for r in images)
    assert 201 in book.exth  # bìa


def test_palmdoc_roundtrip_and_utf8_record_boundaries(tmp_path):
    samples = [
        b"",
        b"a" * 5000,
        bytes(range(256)) * 20,
        ("Người đọc sách tiếng Việt trên Kindle cũ. " * 300).encode("utf-8"),
    ]
    for data in samples:
        for start in range(0, max(1, len(data)), 4096):
            chunk = data[start:start + 4096]
            assert palmdoc_decompress(palmdoc_compress(chunk)) == chunk

    # Chữ Việt dài: ranh giới 4096 byte cắt ngang ký tự UTF-8 nhiều byte.
    paragraphs = [f"Đoạn {i}: Mùa xuân năm ấy, dòng sông lặng lẽ chảy qua những cánh đồng ướt sương." for i in range(400)]
    nodes = [DocumentNode(id=f"p{i}", type=NodeType.PARAGRAPH, content=t) for i, t in enumerate(paragraphs)]
    doc = CanonicalDocument(id="vi", metadata=DocumentMetadata(title="Sông", language="vi"),
                            chapters=[Chapter(id="c1", title="Chương 1", nodes=nodes)])
    out = tmp_path / "vi.mobi"
    write_mobi(build_book_plan(doc), out, get_device("kindle"))
    book = read_mobi(out.read_bytes())
    text = book.text.decode("utf-8")  # không được vỡ ký tự giữa hai bản ghi
    assert book.text_record_count > 1
    assert all(p in text for p in paragraphs[::50])
    assert book.locale == 42  # tiếng Việt


def test_text_cleanup_helpers():
    decomposed = unicodedata.normalize("NFD", "Tiếng Việt")
    assert finalize_text(decomposed + "\u00ad ") == "Tiếng Việt"
    assert detect_language(["Mùa xuân năm ấy, dòng sông lặng lẽ chảy qua những cánh đồng."]) == "vi"
    assert detect_language(["The quick brown fox jumps over the lazy dog."]) == "en"
    # Dấu kiểu chữ luôn thành XHTML đóng đủ thẻ, ký tự đặc biệt được thoát.
    assert inline_markup("a \x01bbold \x01iboth\x02b it\x02i & <x>") == "a <b>bold <i>both</i></b><i> it</i> &amp; &lt;x&gt;"


def test_project_export_embeds_images_and_uses_translation(tmp_path):
    image_path = tmp_path / "figure.png"
    image_path.write_bytes(_png())
    nodes = [
        DocumentNode(id="heading_1", type=NodeType.HEADING, content="Chapter 1: The River",
                     translated_content="Chương 1: Dòng sông", metadata=NodeMetadata(heading_level=1)),
        DocumentNode(id="paragraph_1", type=NodeType.PARAGRAPH, content="The river was low.",
                     translated_content="Dòng sông cạn nước."),
        DocumentNode(id="image_1", type=NodeType.IMAGE, content="[Image]",
                     metadata=NodeMetadata(image_asset_id="asset-1")),
    ]
    doc = CanonicalDocument(
        id="export-doc",
        metadata=DocumentMetadata(title="Dòng sông", author="Ada Writer"),
        chapters=[Chapter(id="chapter_0001", number="1", title="The River", translated_title="Dòng sông", nodes=nodes)],
        assets=[Asset(id="asset-1", original_path=str(image_path))],
    )
    epub_path = EPUBExporter.export_epub(doc, tmp_path / "book.epub", device="kindle_pw")
    with zipfile.ZipFile(epub_path) as zf:
        body = _epub_body(zf)
        opf = zf.read("OEBPS/content.opf").decode("utf-8")
        assert any(n.startswith("OEBPS/images/img0001.") for n in zf.namelist())
    assert "<dc:language>vi</dc:language>" in opf
    assert 'xml:lang="vi"' in body
    assert "Chương 1: Dòng sông" in body and "Dòng sông cạn nước." in body
    assert "/api/assets/" not in body
    # Tiêu đề chương không bị in hai lần (tiêu đề chương + node tiêu đề).
    assert body.count("Chương 1: Dòng sông</h1>") == 1

    mobi_path, engine = MOBIExporter.export_mobi_with_engine(doc, tmp_path / "book.mobi", device="kindle",
                                                             engine="builtin")
    assert engine == "builtin"
    book = read_mobi(mobi_path.read_bytes())
    assert "Dòng sông cạn nước." in book.text.decode("utf-8")
    assert book.first_image_index != 0xFFFFFFFF


def test_convert_api_runs_job_and_serves_file(book_pdf):
    from app.main import app

    client = TestClient(app)
    options = client.get("/api/convert/options").json()
    assert {d["key"] for d in options["devices"]} == {"kindle", "kindle_pw", "kindle_hd"}

    with open(book_pdf, "rb") as fh:
        res = client.post(
            "/api/convert",
            files={"file": ("river.pdf", fh, "application/pdf")},
            data={"output_format": "epub", "device": "kindle_hd", "language": "auto"},
        )
    assert res.status_code == 200, res.text
    job_id = res.json()["id"]
    deadline = time.time() + 120
    job = res.json()
    while job["status"] not in ("done", "error") and time.time() < deadline:
        time.sleep(0.2)
        job = client.get(f"/api/convert/jobs/{job_id}").json()
    assert job["status"] == "done", job
    assert job["result"]["title"] == "The River Book"
    assert job["output_name"].endswith(".epub")

    download = client.get(job["download_url"])
    assert download.status_code == 200
    assert download.content[:2] == b"PK"

    assert client.post("/api/convert", files={"file": ("notes.txt", b"hello", "text/plain")}).status_code == 400
    assert client.delete(f"/api/convert/jobs/{job_id}").status_code == 200
    assert client.get(f"/api/convert/jobs/{job_id}").status_code == 404


def test_decomposed_vietnamese_names_are_composed(tmp_path):
    """Tên file chép từ macOS/điện thoại ở dạng tách dấu (NFD) từng thành "Tu_ Duy Đa__t Cu_o__c.epub"."""
    from app.services.ebook.jobs import ConversionJobManager

    nfd = unicodedata.normalize("NFD", "Tư Duy Đặt Cược")
    assert ConversionJobManager()._output_name(nfd, "epub") == "Tư Duy Đặt Cược.epub"

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 100), "A short page of text for the title test.", fontsize=12)
    pdf = tmp_path / "upload.pdf"
    doc.save(str(pdf))
    result = convert_pdf_to_ebook(pdf, tmp_path / "out.epub", tmp_path / "work",
                                  ConversionOptions(source_name=nfd + ".pdf", cover=False))
    assert result.title == "Tư Duy Đặt Cược"

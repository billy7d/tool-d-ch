"""PDF span-flag regression: italic text must not be read as bold.

PyMuPDF span flags are bit 0 superscript, bit 1 italic, bit 4 bold. The extractor used to read
bit 1 as bold, so any short body paragraph with one italic word became an H4 heading and large
italic text could start a new chapter.
"""
import pymupdf

from app.models.canonical import NodeType
from app.services.extractor.pdf_extractor import PDFExtractor, span_style
from app.services.reconstruction.structure_builder import StructureBuilder

CHAPTER_TITLE = "The Wealth of Cities"
SUBHEADING = "Methods of Study"
MIXED_PARAGRAPH = "The author calls this laissez-faire in the text."
ITALIC_PARAGRAPH = "A wholly italic sentence stays body text."
LARGE_ITALIC = "An italic aside"


def _build_pdf(path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    regular, italic = pymupdf.Font("tiro"), pymupdf.Font("tiit")

    page.insert_text((72, 90), CHAPTER_TITLE, fontname="tibo", fontsize=20)
    page.insert_text((72, 140), SUBHEADING, fontname="tibo", fontsize=14)
    page.insert_text((72, 190), "Cities grew where trade routes met and markets stayed open.",
                     fontname="tiro", fontsize=11)

    # One italic word inside an otherwise regular, short body paragraph.
    tw = pymupdf.TextWriter(page.rect)
    tw.append((72, 230), "The author calls this ", font=regular, fontsize=11)
    tw.append(tw.last_point, "laissez-faire", font=italic, fontsize=11)
    tw.append(tw.last_point, " in the text.", font=regular, fontsize=11)
    tw.write_text(page)

    page.insert_text((72, 270), ITALIC_PARAGRAPH, fontname="tiit", fontsize=11)
    page.insert_text((72, 310), "Most of the town lived by the harbour and the river.",
                     fontname="tiro", fontsize=11)
    page.insert_text((72, 350), LARGE_ITALIC, fontname="tiit", fontsize=18)
    page.insert_text((72, 400), "Prices followed the seasons more than the law.",
                     fontname="tiro", fontsize=11)
    doc.save(str(path))
    doc.close()


def _extract(tmp_path):
    pdf_path = tmp_path / "styles.pdf"
    _build_pdf(pdf_path)
    return PDFExtractor(tmp_path / "project").extract_document(pdf_path)


def _blocks_by_text(extracted):
    return {b["text"]: b for p in extracted["pages"] for b in p["blocks"] if b["type"] == "text"}


def test_span_style_reads_pymupdf_flag_bits():
    # Italic (bit 1) + serif (bit 2): italic, not bold.
    assert span_style({"flags": 6, "font": "NimbusRoman-Italic"}) == (False, True)
    # Superscript (bit 0) alone is neither bold nor italic.
    assert span_style({"flags": 1 | 4, "font": "Times-Roman"}) == (False, False)
    # Bold (bit 4) + serif.
    assert span_style({"flags": 20, "font": "Times-Bold"}) == (True, False)
    # Font-name fallback when the flags carry no style.
    assert span_style({"flags": 0, "font": "ABCDEF+Garamond-Semibold"}) == (True, False)
    assert span_style({"flags": 0, "font": "Helvetica-Oblique"}) == (False, True)
    assert span_style({"flags": 0, "font": "MinionPro-BoldIt"}) == (True, True)
    assert span_style({"flags": 0, "font": "Lit-Regular"}) == (False, False)


def test_block_style_needs_majority_of_characters(tmp_path):
    blocks = _blocks_by_text(_extract(tmp_path))

    mixed = blocks[MIXED_PARAGRAPH]
    assert mixed["is_bold"] is False
    assert mixed["is_italic"] is False  # one italic word is a minority of the block

    assert blocks[ITALIC_PARAGRAPH]["is_italic"] is True
    assert blocks[ITALIC_PARAGRAPH]["is_bold"] is False
    assert blocks[LARGE_ITALIC]["is_bold"] is False

    assert blocks[CHAPTER_TITLE]["is_bold"] is True
    assert blocks[SUBHEADING]["is_bold"] is True


def test_italic_text_stays_paragraph_and_bold_headings_still_detected(tmp_path):
    extracted = _extract(tmp_path)
    doc = StructureBuilder.build_canonical_document(
        project_id="span_styles",
        filename="styles.pdf",
        pages_data=extracted["pages"],
        assets_data=extracted["assets"],
    )

    # Large italic text no longer opens a second chapter.
    assert [ch.title for ch in doc.chapters] == [CHAPTER_TITLE]
    nodes = {n.content: n for n in doc.chapters[0].nodes}

    assert nodes[CHAPTER_TITLE].type == NodeType.HEADING
    assert nodes[CHAPTER_TITLE].metadata.heading_level == 1

    # Bold, larger-than-body line is still a subheading.
    assert nodes[SUBHEADING].type == NodeType.HEADING
    assert nodes[SUBHEADING].metadata.heading_level == 3

    # Body-size paragraphs with italic text stay paragraphs (they used to become H4).
    assert nodes[MIXED_PARAGRAPH].type == NodeType.PARAGRAPH
    assert nodes[ITALIC_PARAGRAPH].type == NodeType.PARAGRAPH
    assert nodes[ITALIC_PARAGRAPH].metadata.is_italic is True
    assert nodes[ITALIC_PARAGRAPH].metadata.is_bold is False

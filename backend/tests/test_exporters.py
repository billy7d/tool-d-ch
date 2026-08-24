import pytest
from pathlib import Path
from app.models.canonical import CanonicalDocument, Chapter, DocumentNode, NodeType, DocumentMetadata
from app.db.models import LayoutProfileModel
from app.services.exporter.pdf_exporter import PDFExporter
from app.services.exporter.epub_exporter import EPUBExporter
from app.services.exporter.validator import ExportValidator


def test_export_pdf_and_epub_pipeline(tmp_path: Path):
    nodes = [
        DocumentNode(
            id="heading_0001_0001",
            type=NodeType.HEADING,
            content="Chapter 1: The Wealth of Nations",
            translated_content="Chương 1: Sự giàu có của các quốc gia"
        ),
        DocumentNode(
            id="paragraph_0001_0002",
            type=NodeType.PARAGRAPH,
            content="The greatest improvement in the productive powers of labour...",
            translated_content="Sự cải tiến lớn nhất trong năng lực sản xuất của lao động..."
        )
    ]

    chapter = Chapter(
        id="chapter_0001",
        number="1",
        title="The Wealth of Nations",
        translated_title="Sự giàu có của các quốc gia",
        nodes=nodes
    )

    doc = CanonicalDocument(
        id="export_test_doc",
        metadata=DocumentMetadata(
            title="Sự giàu có của các quốc gia",
            author="Adam Smith"
        ),
        chapters=[chapter]
    )

    profile = LayoutProfileModel(
        id="classic_profile",
        project_id="export_test_doc",
        name="Classic Book"
    )

    # 1. Test EPUB Export
    epub_out = tmp_path / "book.epub"
    EPUBExporter.export_epub(doc, epub_out)
    valid, msg = ExportValidator.validate_export_file(epub_out, "epub")
    assert valid is True
    assert epub_out.stat().st_size > 0

    # 2. Test PDF Export
    pdf_out = tmp_path / "book.pdf"
    PDFExporter.export_pdf(doc, profile, pdf_out)
    valid_pdf, msg_pdf = ExportValidator.validate_export_file(pdf_out, "pdf")
    assert valid_pdf is True
    assert pdf_out.stat().st_size > 0

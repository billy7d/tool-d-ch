import pytest
from app.services.reconstruction.structure_builder import StructureBuilder
from app.models.canonical import NodeType


def test_structure_builder_chapters_and_headings():
    pages_data = [
        {
            "page_number": 1,
            "width": 595.0,
            "height": 842.0,
            "blocks": [
                {
                    "type": "text",
                    "text": "CHAPTER 1: Introduction to Capital",
                    "font_size": 22.0,
                    "is_bold": True,
                    "bbox": [50, 100, 400, 140],
                    "confidence": 1.0,
                },
                {
                    "type": "text",
                    "text": "1.1 The Nature of Markets",
                    "font_size": 16.0,
                    "is_bold": True,
                    "bbox": [50, 160, 300, 190],
                    "confidence": 1.0,
                },
                {
                    "type": "text",
                    "text": "The market operates through supply and demand.",
                    "font_size": 11.0,
                    "is_bold": False,
                    "bbox": [50, 200, 500, 240],
                    "confidence": 1.0,
                },
                {
                    "type": "text",
                    "text": "[1] Note on historical market data.",
                    "font_size": 9.0,
                    "is_bold": False,
                    "bbox": [50, 780, 400, 800],
                    "confidence": 1.0,
                }
            ]
        }
    ]

    canonical_doc = StructureBuilder.build_canonical_document(
        project_id="test_proj",
        filename="economics.pdf",
        pages_data=pages_data,
        assets_data=[]
    )

    assert len(canonical_doc.chapters) >= 1
    ch = canonical_doc.chapters[0]
    assert "Introduction to Capital" in ch.title

    node_types = [n.type for n in ch.nodes]
    assert NodeType.HEADING in node_types
    assert NodeType.PARAGRAPH in node_types
    assert NodeType.FOOTNOTE in node_types

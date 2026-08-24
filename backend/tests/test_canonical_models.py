import pytest
from app.models.canonical import (
    CanonicalDocument,
    Chapter,
    DocumentNode,
    NodeType,
    NodeStatus,
    ApprovalStatus,
    SourceMapping,
    NodeMetadata,
)


def test_canonical_document_serialization():
    node1 = DocumentNode(
        id="paragraph_0001_0001",
        type=NodeType.PARAGRAPH,
        content="The intelligent investor is a realist who sells to optimists and buys from pessimists.",
        translated_content="Nhà đầu tư thông minh là một người thực tế...",
        source_mapping=SourceMapping(
            source_document="book.pdf",
            source_page_start=12,
            source_page_end=12,
            source_block_ids=[1]
        ),
        metadata=NodeMetadata(
            confidence=0.98,
            font_size=11.0,
            font_name="Noto Serif"
        )
    )

    chapter = Chapter(
        id="chapter_0001",
        number="1",
        title="Investment Versus Speculation",
        level=1,
        source_pages=[12, 13, 14],
        nodes=[node1]
    )

    doc = CanonicalDocument(
        id="test_doc_1",
        chapters=[chapter]
    )

    json_data = doc.model_dump()
    assert json_data["id"] == "test_doc_1"
    assert len(json_data["chapters"]) == 1
    assert json_data["chapters"][0]["nodes"][0]["id"] == "paragraph_0001_0001"
    assert json_data["chapters"][0]["nodes"][0]["type"] == "paragraph"

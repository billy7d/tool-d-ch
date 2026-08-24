import pytest
from app.services.translation.mock_provider import MockProvider
from app.services.translation.chunker import SemanticChunker
from app.services.translation.prompt_builder import PromptBuilder
from app.models.canonical import DocumentNode, NodeType, TranslationMode


def test_translation_chunker_and_prompt():
    nodes = [
        DocumentNode(
            id=f"paragraph_0001_{i:04d}",
            type=NodeType.PARAGRAPH,
            content=f"Sentence {i}: Understanding cash flow and compound interest is fundamental."
        )
        for i in range(1, 10)
    ]

    chunks = SemanticChunker.chunk_nodes(nodes, target_tokens=200)
    assert len(chunks) >= 1

    sys_prompt = PromptBuilder.build_system_prompt(
        document_type="FINANCE",
        translation_mode=TranslationMode.NATURAL
    )
    assert "Natural, fluent Vietnamese" in sys_prompt or "Natural Vietnamese" in sys_prompt

    user_prompt = PromptBuilder.build_user_prompt(
        nodes=chunks[0].nodes,
        chapter_title="Chapter 1",
        glossary_terms={"cash flow": "dòng tiền"}
    )
    assert "dòng tiền" in user_prompt

    # Test Mock Provider Translation
    provider = MockProvider()
    results = provider.translate(
        blocks=[{"id": n.id, "text": n.content} for n in chunks[0].nodes],
        system_prompt=sys_prompt,
        user_prompt=user_prompt
    )

    assert len(results) == len(chunks[0].nodes)
    assert "dòng tiền" in results[0]["text"]

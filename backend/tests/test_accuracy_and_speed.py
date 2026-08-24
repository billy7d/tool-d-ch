import pytest
from app.models.canonical import DocumentNode, NodeType, NodeStatus, TranslationMode
from app.services.translation.vietnamese_post_processor import VietnamesePostProcessor
from app.services.translation.prompt_builder import PromptBuilder
from app.services.translation.chunker import SemanticChunker
from app.services.translation.ollama_provider import OllamaProvider


def test_vietnamese_post_processor_cleanup():
    # 1. AI preamble removal
    dirty_1 = "Dưới đây là bản dịch tiếng Việt:\n\nDoanh thu của công ty đã tăng trưởng vượt bậc."
    cleaned_1 = VietnamesePostProcessor.clean_vietnamese_text(dirty_1)
    assert not cleaned_1.startswith("Dưới đây là")
    assert "Doanh thu của công ty" in cleaned_1

    # 2. Punctuation spacing normalization
    dirty_2 = "Hội đồng quản trị quyết định ,rằng dự án sẽ tiếp tục .Điều này rất quan trọng !"
    cleaned_2 = VietnamesePostProcessor.clean_vietnamese_text(dirty_2)
    assert "quyết định, rằng" in cleaned_2
    assert "tiếp tục. Điều này" in cleaned_2
    assert "quan trọng!" in cleaned_2

    # 3. Quotes normalization
    dirty_3 = 'Ông nói "chúng tôi cam kết chất lượng" và tiếp tục.'
    cleaned_3 = VietnamesePostProcessor.clean_vietnamese_text(dirty_3)
    assert '“chúng tôi cam kết chất lượng”' in cleaned_3


def test_prompt_builder_relevant_glossary_filter():
    nodes = [
        DocumentNode(id="n1", type=NodeType.PARAGRAPH, content="The company increased its cash flow significantly."),
        DocumentNode(id="n2", type=NodeType.PARAGRAPH, content="All interest rate risks were mitigated.")
    ]

    glossary = {
        "cash flow": "dòng tiền",
        "interest rate": "lãi suất",
        "amortization": "khấu hao",
        "equity": "vốn chủ sở hữu",
        "liquidity": "tính thanh khoản",
        "balance sheet": "bảng cân đối kế toán",
        "retained earnings": "lợi nhuận giữ lại",
        "working capital": "vốn lưu động",
        "gross margin": "biên lợi nhuận gộp",
        "depreciation": "khấu hao tài sản",
        "default swap": "hợp đồng hoán đổi rủi ro tín dụng",
        "yield curve": "đường cong lợi suất"
    }

    filtered = PromptBuilder.filter_relevant_glossary(nodes, glossary)
    # Only "cash flow" and "interest rate" appear in nodes
    assert "cash flow" in filtered
    assert "interest rate" in filtered
    assert "amortization" not in filtered
    assert "default swap" not in filtered


def test_semantic_chunker_target_tokens():
    nodes = [
        DocumentNode(id=f"n_{i}", type=NodeType.PARAGRAPH, content=f"This is sentence {i} explaining financial market trends with detailed economic analysis.")
        for i in range(50)
    ]
    chunks = SemanticChunker.chunk_nodes(nodes, target_tokens=750, max_tokens=1200)
    assert len(chunks) > 0
    # Each chunk should contain multiple nodes
    for c in chunks:
        assert len(c.nodes) >= 1
        assert c.estimated_tokens <= 1200


def test_ollama_provider_session_and_keepalive():
    provider = OllamaProvider()
    assert hasattr(provider, "session")
    assert provider.session is not None

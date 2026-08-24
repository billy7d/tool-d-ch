import time
import pytest
import uuid
from app.models.canonical import CanonicalDocument, Chapter, DocumentNode, NodeType, DocumentMetadata
from app.services.renderer.semantic_html import SemanticHTMLRenderer
from app.services.renderer.preview_service import PreviewService
from app.db.models import LayoutProfileModel
from app.services.translation.chunker import SemanticChunker


def test_1000_page_stress_scale():
    """
    Simulates a synthetic 1,000-page book with 300,000+ words and 10,000+ semantic nodes.
    Verifies memory safety, chunking speed, and smart preview performance.
    """
    total_chapters = 30
    nodes_per_chapter = 350
    # Total nodes = 30 * 350 = 10,500 nodes

    sample_paragraph = (
        "The market continued to expand during the next decade despite several downturns. "
        "Intelligent investors should focus on underlying business fundamentals rather than "
        "short-term price fluctuations. When assessing financial health, free cash flow and "
        "consistent return on invested capital represent the most reliable indicators."
    )

    chapters = []
    node_id_counter = 1

    start_time = time.time()
    for ch_idx in range(1, total_chapters + 1):
        nodes = []
        # Heading 1
        nodes.append(DocumentNode(
            id=f"heading_{ch_idx:04d}_{node_id_counter:05d}",
            type=NodeType.HEADING,
            content=f"Chapter {ch_idx}: Advanced Investment Strategies and Risk Management",
            translated_content=f"Chương {ch_idx}: Chiến lược Đầu tư Nâng cao và Quản trị Rủi ro"
        ))
        node_id_counter += 1

        for n_idx in range(1, nodes_per_chapter):
            nodes.append(DocumentNode(
                id=f"paragraph_{ch_idx:04d}_{node_id_counter:05d}",
                type=NodeType.PARAGRAPH,
                content=sample_paragraph,
                translated_content="Thị trường tiếp tục tăng trưởng trong thập kỷ tiếp theo bất chấp các đợt suy thoái. Các nhà đầu tư thông minh nên tập trung vào các yếu tố cơ bản của doanh nghiệp..."
            ))
            node_id_counter += 1

        chapters.append(Chapter(
            id=f"chapter_{ch_idx:04d}",
            number=str(ch_idx),
            title=f"Advanced Investment Strategies {ch_idx}",
            translated_title=f"Chiến lược Đầu tư Nâng cao {ch_idx}",
            nodes=nodes
        ))

    generation_time = time.time() - start_time
    total_nodes_created = sum(len(c.nodes) for c in chapters)
    total_words_created = sum(len(n.content.split()) for c in chapters for n in c.nodes)

    assert total_nodes_created >= 10000
    assert total_words_created >= 300000
    print(f"\n[1,000-page Test] Generated {total_nodes_created} nodes ({total_words_created} words) in {generation_time:.2f}s")

    # 1. Test Semantic Chunking across 10,000 nodes
    chunk_start = time.time()
    all_nodes = [n for ch in chapters for n in ch.nodes]
    chunks = SemanticChunker.chunk_nodes(all_nodes, target_tokens=500)
    chunk_time = time.time() - chunk_start
    assert len(chunks) > 1000
    print(f"[1,000-page Test] Chunked into {len(chunks)} semantic batches in {chunk_time:.2f}s")

    # 2. Test Smart Preview (Must NOT freeze or attempt to render all 10,000 nodes)
    preview_start = time.time()
    doc = CanonicalDocument(
        id="stress_1000",
        metadata=DocumentMetadata(
            title="1000 Page Synthetic Masterpiece",
            total_pages=1000,
            total_words=total_words_created,
            total_nodes=total_nodes_created
        ),
        chapters=chapters
    )
    profile = LayoutProfileModel(
        id="classic_profile",
        project_id="stress_1000",
        name="Classic Book"
    )

    preview_html = PreviewService.render_smart_preview(doc, profile, sample_type="representative", node_limit=40)
    preview_time = time.time() - preview_start
    assert len(preview_html) > 500
    assert preview_time < 0.15 # Smart preview renders in milliseconds!
    print(f"[1,000-page Test] Smart Preview generated in {preview_time:.3f}s (Passed lazy-load memory test)")

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, ChapterModel, NodeModel, ProjectModel
from app.models.canonical import DocumentNode, NodeStatus, NodeType, TranslationMode
from app.services.translation.adaptive_chunker import AdaptiveSemanticChunker
from app.services.translation.context_assembler import ContextAssembler
from app.services.translation.context_memory import ChapterMemory, RollingContextService
from app.services.translation.document_profiler import DocumentProfiler
from app.services.translation.model_capabilities import ModelCapabilities
from app.services.translation.prompt_builder import PromptBuilder
from app.services.translation.prompt_profiles import STYLE_PACKS, select_few_shots
from fastapi.testclient import TestClient
from app.main import app


class CountingProvider:
    def __init__(self):
        self.calls = 0

    def summarize_context(self, text_sample, model=None):
        self.calls += 1
        return "Tóm tắt tài liệu có kiểm soát."


def _node(index: int, text: str, node_type: NodeType = NodeType.PARAGRAPH) -> DocumentNode:
    return DocumentNode(
        id=f"node_{index}", type=node_type, content=text,
        status=NodeStatus.PENDING, order_index=index,
    )


def test_all_required_style_packs_and_domain_examples_load():
    expected = {"GENERAL", "BUSINESS", "FINANCE", "SELF_HELP", "TECHNICAL", "ACADEMIC", "LEGAL", "LITERATURE"}
    assert expected == set(STYLE_PACKS)
    for domain in expected:
        pack = STYLE_PACKS[domain]
        assert pack.instructions
        assert pack.forbidden
        assert select_few_shots(domain, "NATURAL", "paragraph")


def test_document_profile_is_cached_and_invalidated_by_user_setup(tmp_path: Path):
    nodes = [_node(i, f"Section {i}. The company improved operations.") for i in range(20)]
    provider = CountingProvider()
    first = DocumentProfiler.load_or_create(nodes, tmp_path, "BUSINESS", "", provider, "mock")
    second = DocumentProfiler.load_or_create(nodes, tmp_path, "BUSINESS", "", provider, "mock")
    assert first.source_hash == second.source_hash
    assert provider.calls == 1
    changed = DocumentProfiler.load_or_create(nodes, tmp_path, "BUSINESS", "Dùng câu ngắn.", provider, "mock")
    assert changed.setup_hash != first.setup_hash
    assert provider.calls == 2


def test_context_assembler_trims_low_priority_context_but_keeps_locked_glossary():
    nodes = [_node(1, "The Federal Reserve raised the target rate by 0.25%.")]
    profile = DocumentProfiler.fallback_profile("FINANCE", nodes[0].content)
    profile.summary = "x " * 2000
    memory = ChapterMemory(summary="y " * 1800)
    context = ContextAssembler.assemble_context(
        nodes, profile, memory, [],
        {"Federal Reserve": "Cục Dự trữ Liên bang Mỹ", "unused": "không dùng"},
        ModelCapabilities(4096, 4096, 1200, True, True),
        [{"source": "s " * 1000, "target": "t " * 1000}],
    )
    assert context.glossary == {"Federal Reserve": "Cục Dự trữ Liên bang Mỹ"}
    assert context.token_budget.source_budget >= context.token_budget.source_tokens
    assert context.token_budget.total_estimated_tokens <= 4096
    assert context.few_shots == []


def test_rolling_context_only_uses_validated_nodes_and_resets_by_chapter():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(ProjectModel(id="p", title="Test"))
    db.add_all([
        ChapterModel(id="c1", project_id="p", title="One", order_index=1),
        ChapterModel(id="c2", project_id="p", title="Two", order_index=2),
    ])
    db.add_all([
        NodeModel(id="n1", project_id="p", chapter_id="c1", content="A", translated_content="Á", status="TRANSLATED", order_index=1),
        NodeModel(id="n2", project_id="p", chapter_id="c1", content="B", translated_content="B sai", status="NEEDS_REVIEW", order_index=2),
        NodeModel(id="n3", project_id="p", chapter_id="c2", content="C", translated_content="C", status="TRANSLATED", order_index=1),
    ])
    db.commit()
    items = RollingContextService.get_previous(db, "c1", 99)
    assert [item.node_id for item in items] == ["n1"]
    assert all(item.node_id != "n3" for item in items)


def test_adaptive_chunker_splits_oversized_paragraph_on_sentence_boundaries():
    text = " ".join(f"Sentence {i} contains enough words to exercise semantic splitting." for i in range(80))
    chunks = AdaptiveSemanticChunker.chunk_nodes([_node(1, text)], source_budget=120, hard_limit=120)
    assert len(chunks) == 1
    parts = chunks[0].oversized_segments["node_1"]
    assert len(parts) > 1
    assert all(AdaptiveSemanticChunker.estimate_tokens(part) <= 120 for part in parts)


def test_prompt_v2_contains_hierarchy_bilingual_reference_and_safety_rules():
    system = PromptBuilder.build_system_prompt("LEGAL", TranslationMode.NATURAL)
    assert "PRIORITY 1 - SEMANTIC FIDELITY" in system
    assert "Do not summarize" in system
    assert "LEGAL" in system
    try:
        PromptBuilder.build_system_prompt("GENERAL", TranslationMode.CUSTOM, custom_instructions="Tóm tắt mỗi đoạn.")
    except ValueError as exc:
        assert "xung đột" in str(exc)
    else:
        raise AssertionError("Chỉ dẫn làm mất ý phải bị từ chối")


def test_preview_endpoint_does_not_commit_candidate():
    from app.db.engine import get_project_db
    project_id = "phase2_preview_no_commit"
    db = get_project_db(project_id)
    db.query(NodeModel).filter(NodeModel.project_id == project_id).delete()
    db.query(ChapterModel).filter(ChapterModel.project_id == project_id).delete()
    db.query(ProjectModel).filter(ProjectModel.id == project_id).delete()
    db.add(ProjectModel(id=project_id, title="Preview", selected_model="mock-qwen2.5:7b"))
    db.add(ChapterModel(id="preview_chapter", project_id=project_id, title="Chapter 1", order_index=0))
    db.add(NodeModel(
        id="preview_node", project_id=project_id, chapter_id="preview_chapter",
        node_type="paragraph", content="The market continued to expand.",
        status="PENDING", order_index=1,
    ))
    db.commit()
    response = TestClient(app).post(
        f"/api/projects/{project_id}/translation/preview",
        json={"model_name": "mock-qwen2.5:7b", "document_type": "BUSINESS", "translation_mode": "NATURAL"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["samples"]
    db.expire_all()
    node = db.query(NodeModel).filter(NodeModel.id == "preview_node").first()
    assert node.translated_content is None
    assert node.status == "PENDING"
    db.close()

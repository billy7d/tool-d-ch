import pytest
from app.services.translation.json_parser import clean_and_parse_llm_json
from app.services.translation.mock_provider import MockProvider
from app.services.translation.worker import TranslationWorker
from app.db.engine import get_project_db
from app.db.models import ProjectModel, ChapterModel, NodeModel
from app.db.repository import ProjectRepository


def test_robust_json_parser_cases():
    # Case 1: Markdown code fences
    md_fenced = '```json\n{\n  "translations": [\n    {"node_id": "n1", "text": "Đoạn 1"}\n  ]\n}\n```'
    parsed = clean_and_parse_llm_json(md_fenced, expected_node_ids=["n1"])
    assert len(parsed) == 1
    assert parsed[0]["node_id"] == "n1"
    assert parsed[0]["text"] == "Đoạn 1"

    # Case 2: Preamble and postamble with trailing commas
    preamble = 'Here is your translated JSON output:\n{\n  "translations": [\n    {"node_id": "n1", "text": "Xin chào",},\n    {"node_id": "n2", "text": "Thế giới",},\n  ]\n}\nHope this helps!'
    parsed2 = clean_and_parse_llm_json(preamble, expected_node_ids=["n1", "n2"])
    assert len(parsed2) == 2
    assert parsed2[0]["text"] == "Xin chào"
    assert parsed2[1]["text"] == "Thế giới"

    # Case 3: List without node_id (model forgot IDs) mapped by position
    list_no_ids = '[\n  {"text": "Bản dịch 1"},\n  {"text": "Bản dịch 2"}\n]'
    parsed3 = clean_and_parse_llm_json(list_no_ids, expected_node_ids=["node_alpha", "node_beta"])
    assert len(parsed3) == 2
    assert parsed3[0]["node_id"] == "node_alpha"
    assert parsed3[0]["text"] == "Bản dịch 1"
    assert parsed3[1]["node_id"] == "node_beta"
    assert parsed3[1]["text"] == "Bản dịch 2"

    # Case 4: Direct dict key-value mapping
    dict_map = '{"node_1": "Nội dung dịch 1", "node_2": "Nội dung dịch 2"}'
    parsed4 = clean_and_parse_llm_json(dict_map, expected_node_ids=["node_1", "node_2"])
    assert len(parsed4) == 2
    assert parsed4[0]["node_id"] == "node_1"
    assert parsed4[0]["text"] == "Nội dung dịch 1"


class FlakyBatchProvider(MockProvider):
    """Simulates an LLM provider that omits some nodes in batch, forcing single-node fallback and auto-healing."""
    def translate(self, blocks, system_prompt, user_prompt, model=None, temperature=0.3):
        # Only return the first block, omitting other blocks in the batch
        if blocks:
            first = blocks[0]
            nid = first.get("id") or first.get("node_id", "")
            return [{"node_id": nid, "text": f"[Batch Tr] {first.get('text', '')}"}]
        return []


def test_single_fallback_and_auto_healing_flow():
    project_id = "test_self_healing_proj"
    db = get_project_db(project_id)
    try:
        # Clear existing
        db.query(NodeModel).filter(NodeModel.project_id == project_id).delete()
        db.query(ChapterModel).filter(ChapterModel.project_id == project_id).delete()
        db.query(ProjectModel).filter(ProjectModel.id == project_id).delete()
        db.commit()

        # Create project with 4 nodes in a chapter
        proj = ProjectModel(id=project_id, title="Self Healing Test Book")
        db.add(proj)
        ch = ChapterModel(id="ch_heal_1", project_id=project_id, number=1, title="Chapter 1", order_index=0)
        db.add(ch)
        
        for i in range(4):
            nm = NodeModel(
                id=f"node_heal_{i}",
                project_id=project_id,
                chapter_id="ch_heal_1",
                node_type="paragraph",
                content=f"This is paragraph {i} discussing compound interest and investment cash flow.",
                order_index=i,
                status="PENDING",
                approval_status="UNAPPROVED",
                version=1
            )
            db.add(nm)
        db.commit()

        # Run translation with FlakyBatchProvider (which drops batch nodes)
        worker = TranslationWorker()
        flaky_provider = FlakyBatchProvider()
        worker.translate_project_sync(
            project_id=project_id,
            model_name="mock",
            provider=flaky_provider
        )

        # Verify all 4 nodes were successfully translated via Single Fallback or Auto-Healing
        nodes = db.query(NodeModel).filter(NodeModel.project_id == project_id).all()
        assert len(nodes) == 4
        for n in nodes:
            assert n.status == "TRANSLATED", f"Node {n.id} status is {n.status}, expected TRANSLATED"
            assert n.translated_content is not None
            assert len(n.translated_content) > 0

        # Verify project status is TRANSLATED
        proj_refreshed = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
        assert proj_refreshed.current_stage == "TRANSLATED"

    finally:
        db.close()

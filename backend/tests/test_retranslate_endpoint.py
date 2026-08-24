import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.db.engine import get_project_db
from app.db.models import ProjectModel, ChapterModel, NodeModel
from app.services.translation.vietnamese_post_processor import VietnamesePostProcessor

client = TestClient(app)


def test_retranslate_node_endpoint_success():
    project_id = "test_retranslate_api_proj"
    node_id = "paragraph_0001_00208"
    db = get_project_db(project_id)
    try:
        # Clear existing
        db.query(NodeModel).filter(NodeModel.project_id == project_id).delete()
        db.query(ChapterModel).filter(ChapterModel.project_id == project_id).delete()
        db.query(ProjectModel).filter(ProjectModel.id == project_id).delete()
        db.commit()

        # Create project, chapter, node
        proj = ProjectModel(id=project_id, title="Test Retranslate Book", selected_model="mock-qwen2.5:7b")
        db.add(proj)
        ch = ChapterModel(id="ch_1", project_id=project_id, title="Chapter 1", order_index=0)
        db.add(ch)
        node = NodeModel(
            id=node_id,
            project_id=project_id,
            chapter_id="ch_1",
            node_type="paragraph",
            content="Compound interest is the eighth wonder of the world.",
            translated_content="Lãi kép là kỳ quan thứ tám của thế giới.",
            order_index=0,
            status="TRANSLATED",
            version=None  # Test version=None edge case!
        )
        db.add(node)
        db.commit()

        # Call retranslate endpoint
        response = client.post(
            f"/api/projects/{project_id}/translation/nodes/{node_id}/retranslate",
            json={
                "instruction": "Dịch văn phong triết lý và sâu sắc hơn",
                "custom_model": "mock-qwen2.5:7b"
            }
        )

        assert response.status_code == 200, f"Expected 200 but got {response.status_code}: {response.text}"
        data = response.json()
        assert data["node_id"] == node_id
        assert data["translated_content"] is not None
        assert len(data["translated_content"]) > 0

        # Verify DB node updated
        db.refresh(node)
        assert node.translated_content == data["translated_content"]
        assert node.version >= 1

        # Test empty payload (defaults)
        response2 = client.post(
            f"/api/projects/{project_id}/translation/nodes/{node_id}/retranslate",
            json={}
        )
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["node_id"] == node_id

    finally:
        db.close()


def test_deduplicate_repetition_loops():
    repeated_para = (
        "Tuy nhiên, chính trong toàn bộ quá trình đối mặt và giải quyết vấn đề, cuộc sống mới có ý nghĩa.\n\n"
        "Tuy nhiên, chính trong toàn bộ quá trình đối mặt và giải quyết vấn đề, cuộc sống mới có ý nghĩa.\n\n"
        "Tuy nhiên, chính trong toàn bộ quá trình đối mặt và giải quyết vấn đề, cuộc sống mới có ý nghĩa.\n\n"
        "Tuy nhiên, chính trong toàn bộ quá trình đối mặt và giải quyết vấn đề, cuộc sống mới có ý nghĩa."
    )
    cleaned = VietnamesePostProcessor.clean_vietnamese_text(repeated_para)
    # Must collapse 4 identical paragraphs down to 1 single paragraph
    assert cleaned.count("Tuy nhiên") == 1
    assert "quá trình đối mặt" in cleaned


def test_strip_chinese_characters():
    mixed = "Vấn đề là 刃口 là điểm mấu chốt phân biệt giữa thành công và thất bại."
    stripped = VietnamesePostProcessor.strip_chinese_characters(mixed)
    assert not VietnamesePostProcessor.contains_chinese(stripped)
    assert "điểm mấu chốt" in stripped


def test_clean_json_wrapper_and_ai_notes():
    raw_problematic_output = (
        '{\n\n'
        '“translation”: “Tuy nhiên, chính trong suốt quá trình gặp gỡ và giải quyết vấn đề mới có ý nghĩa của cuộc sống. Vấn đề là，。；，。”\n\n'
        '}\n\n'
        'Lưu Ý:\n\n'
        '- Bản dịch đã được viết hoàn toàn bằng tiếng Việt. - Đã loại bỏ tất cả các từ Hán hoặc bất kỳ ký tự Trung Quốc nào khác.'
    )
    cleaned = VietnamesePostProcessor.clean_vietnamese_text(raw_problematic_output)

    # 1. Must NOT contain JSON markers or keys
    assert "{" not in cleaned
    assert "}" not in cleaned
    assert "translation" not in cleaned

    # 2. Must NOT contain AI notes
    assert "Lưu Ý" not in cleaned
    assert "Bản dịch đã được" not in cleaned

    # 3. Must NOT contain Chinese punctuation
    assert not VietnamesePostProcessor.contains_chinese(cleaned)

    # 4. Must contain the core translation
    assert "Tuy nhiên, chính trong suốt quá trình" in cleaned


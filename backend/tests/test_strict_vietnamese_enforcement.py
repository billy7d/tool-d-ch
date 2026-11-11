import pytest
from app.services.translation.vietnamese_post_processor import VietnamesePostProcessor
from app.services.translation.worker import TranslationWorker
from app.services.translation.mock_provider import MockProvider
from app.db.engine import get_project_db
from app.db.models import ProjectModel, ChapterModel, NodeModel


def test_chinese_character_detection():
    # 1. Pure Vietnamese
    pure_vi_1 = "Công ty đã ghi nhận mức tăng trưởng doanh thu ấn tượng trong năm 2026."
    pure_vi_2 = "Hội đồng quản trị cam kết tối ưu hóa dòng tiền và quản trị rủi ro."
    assert not VietnamesePostProcessor.contains_chinese(pure_vi_1)
    assert not VietnamesePostProcessor.contains_chinese(pure_vi_2)

    # 2. Text containing Chinese characters (Simplified / Traditional / Mixed)
    mixed_1 = "Dự án này rất 好 (tốt) và có tiềm năng lớn."
    mixed_2 = "Công ty 投资 (đầu tư) mạnh mẽ vào thị trường Đông Nam Á."
    mixed_3 = "这是一个测试 (Đây là bản thử nghiệm)."
    mixed_4 = "Tiến độ thực hiện: 完成 100%."
    assert VietnamesePostProcessor.contains_chinese(mixed_1)
    assert VietnamesePostProcessor.contains_chinese(mixed_2)
    assert VietnamesePostProcessor.contains_chinese(mixed_3)
    assert VietnamesePostProcessor.contains_chinese(mixed_4)

    # 3. Validation method
    valid, reason = VietnamesePostProcessor.validate_pure_vietnamese(pure_vi_1)
    assert valid is True
    assert reason == "OK"

    valid_cn, reason_cn = VietnamesePostProcessor.validate_pure_vietnamese(mixed_1)
    assert valid_cn is False
    assert reason_cn == "CONTAINS_CHINESE_CHARACTERS"


class FlakyChineseProvider(MockProvider):
    """Đoạn thứ hai đầy chữ Hán ở lần dịch đầu nên phải dịch lại; lần repair trả bản sạch."""

    def __init__(self):
        self.second_paragraph_calls = 0

    def translate_single(self, text, system_prompt, glossary_terms=None, model=None, temperature=0.2):
        if not text.startswith("Second"):
            return "Đoạn đầu tiên đã được dịch sạch sẽ sang tiếng Việt."
        self.second_paragraph_calls += 1
        if self.second_paragraph_calls == 1:
            return "这是一个严重错误需要重新翻译 và cần dịch lại."
        return "Bản dịch đã được tự động sửa lại thành tiếng Việt chuẩn không có chữ Hán."


def test_auto_retry_on_chinese_detection():
    project_id = "test_chinese_rejection_proj"
    db = get_project_db(project_id)
    try:
        # Clear existing test data
        db.query(NodeModel).filter(NodeModel.project_id == project_id).delete()
        db.query(ChapterModel).filter(ChapterModel.project_id == project_id).delete()
        db.query(ProjectModel).filter(ProjectModel.id == project_id).delete()
        db.commit()

        proj = ProjectModel(id=project_id, title="Chinese Rejection Test Project")
        db.add(proj)
        ch = ChapterModel(id="ch_cn_1", project_id=project_id, number="1", title="Chương 1", order_index=0)
        db.add(ch)

        n1 = NodeModel(
            id="ch1_node_1",
            project_id=project_id,
            chapter_id="ch_cn_1",
            node_type="paragraph",
            content="First paragraph in English.",
            order_index=0,
            status="PENDING",
            approval_status="UNAPPROVED",
            version=1
        )
        n2 = NodeModel(
            id="ch1_node_2",
            project_id=project_id,
            chapter_id="ch_cn_1",
            node_type="paragraph",
            content="Second paragraph in English.",
            order_index=1,
            status="PENDING",
            approval_status="UNAPPROVED",
            version=1
        )
        db.add(n1)
        db.add(n2)
        db.commit()

        worker = TranslationWorker()
        provider = FlakyChineseProvider()

        # Run translation loop
        worker.translate_project_sync(
            project_id=project_id,
            model_name="mock",
            provider=provider
        )

        # Verify results in SQLite
        db_n1 = db.query(NodeModel).filter(NodeModel.id == "ch1_node_1").first()
        db_n2 = db.query(NodeModel).filter(NodeModel.id == "ch1_node_2").first()

        assert db_n1.status == "TRANSLATED"
        assert not VietnamesePostProcessor.contains_chinese(db_n1.translated_content)

        assert db_n2.status == "TRANSLATED"
        # Node 2 must have been auto-retried and repaired into pure Vietnamese
        assert not VietnamesePostProcessor.contains_chinese(db_n2.translated_content)
        assert "không có chữ Hán" in db_n2.translated_content

    finally:
        db.close()

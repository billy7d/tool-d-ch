import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.db.models import (
    Base,
    ChapterModel,
    GlossaryModel,
    LayoutProfileModel,
    NodeModel,
    ProjectModel,
    QAIssueModel,
    TranslationMemoryModel,
    TranslationModel,
    TranslationVersionModel,
)
from app.db.engine import get_project_db
from app.main import app
from app.models.canonical import DocumentNode, NodeStatus, NodeType
from app.services.qa.deterministic_qa import DeterministicQA
from app.services.qa.result_validator import qa_error, validate_qa_result
from app.services.translation.context_memory import ChapterMemoryBuilder
from app.services.translation.mock_provider import MockProvider
from app.services.translation.numeric_validator import NumericValidator
from app.services.translation.quality_gate import TranslationQualityGate
from app.services.translation.reference_validator import ReferenceValidator
from app.services.translation.translation_memory import TranslationMemoryService
from app.services.translation.worker import translation_worker


def _codes(result):
    return {issue["code"] for issue in result.issues}


def test_added_number_is_hard_failure_and_missing_number_is_preserved():
    added = TranslationQualityGate().validate(
        "Revenue increased by 5%.",
        "Doanh thu tăng 5% và mục tiêu là 20%.",
        {},
    )
    missing = TranslationQualityGate().validate(
        "Revenue increased by 5%.",
        "Doanh thu tăng.",
        {},
    )

    assert NumericValidator.validate(
        "Revenue increased by 5%.",
        "Doanh thu tăng 5% và mục tiêu là 20%.",
    ).passed is False
    assert "NUMBER_ADDITION" in _codes(added)
    assert next(issue for issue in added.issues if issue["code"] == "NUMBER_ADDITION")["severity"] == "ERROR"
    assert added.hard_fail is True
    assert "NUMBER_MISMATCH" in _codes(missing)
    assert missing.hard_fail is True


def test_numeric_locale_normalization_still_passes():
    result = TranslationQualityGate().validate(
        "The amount is 1,000.50 USD.",
        "Số tiền là 1.000,50 USD.",
        {},
    )
    assert result.passed is True
    assert NumericValidator.validate("-10°C", "-10°C").passed is True


def test_reference_addition_fails_but_localized_reference_passes():
    localized = ReferenceValidator.validate("See Figure 3.", "Xem Hình 3.")
    added = ReferenceValidator.validate("See Figure 3.", "Xem Hình 3 và Hình 7.")
    quality = TranslationQualityGate().validate(
        "See Figure 3.",
        "Xem Hình 3 và Hình 7.",
        {},
    )

    assert localized.passed is True
    assert localized.unexpected == []
    assert localized.missing == []
    assert localized.passed is True
    assert added.passed is False
    assert [token.semantic_key for token in added.unexpected] == ["FIGURE:7"]
    assert "REFERENCE_ADDITION" in _codes(quality)
    assert "NUMBER_ADDITION" not in _codes(quality)
    assert next(issue for issue in quality.issues if issue["code"] == "REFERENCE_ADDITION")["severity"] == "ERROR"


def test_url_validation_is_bidirectional_and_order_independent():
    same_urls_different_order = TranslationQualityGate().validate(
        "Visit https://a.example and https://b.example.",
        "Truy cập https://b.example và https://a.example.",
        {},
    )
    added = TranslationQualityGate().validate(
        "Visit https://example.com.",
        "Truy cập https://example.com hoặc https://fake.example.",
        {},
    )
    missing = TranslationQualityGate().validate(
        "Visit https://example.com.",
        "Truy cập trang web.",
        {},
    )

    assert same_urls_different_order.passed is True
    assert "URL_ADDITION" in _codes(added)
    assert next(issue for issue in added.issues if issue["code"] == "URL_ADDITION")["severity"] == "ERROR"
    assert "URL_MISMATCH" in _codes(missing)
    assert missing.hard_fail is True


def test_deterministic_qa_uses_quality_gate_for_added_facts():
    node = NodeModel(
        id="phase2_1_2_qa_node",
        content="Revenue increased by 5%.",
        translated_content="Doanh thu tăng 5% và mục tiêu là 20%.",
    )
    issues = DeterministicQA.audit_node(node)
    issue_types = {issue["issue_type"] for issue in issues}
    assert "NUMBER_ADDITION" in issue_types
    assert next(issue for issue in issues if issue["issue_type"] == "NUMBER_ADDITION")["severity"] == "ERROR"


def test_qa_error_keeps_provider_detail_when_normalizing_response():
    normalized = validate_qa_result(qa_error("provider timeout"))
    assert normalized["status"] == "ERROR"
    assert normalized["error"] == "provider timeout"


class BulkReviewProvider(MockProvider):
    def __init__(self, review_result=None, review_error=None):
        self.review_result = review_result
        self.review_error = review_error

    def translate_single(
        self,
        text,
        system_prompt,
        glossary_terms=None,
        model=None,
        temperature=0.3,
        user_prompt=None,
    ):
        return "Doanh thu tăng 5%."

    def review_translation(self, source_text, translated_text, glossary_terms, model=None):
        if self.review_error:
            raise self.review_error
        return self.review_result


def _clear_project_data(db, project_id):
    db.query(TranslationMemoryModel).delete(synchronize_session=False)
    db.query(QAIssueModel).filter(QAIssueModel.project_id == project_id).delete(synchronize_session=False)
    db.query(TranslationModel).filter(TranslationModel.project_id == project_id).delete(synchronize_session=False)
    db.query(TranslationVersionModel).filter(
        TranslationVersionModel.node_id.in_(
            db.query(NodeModel.id).filter(NodeModel.project_id == project_id)
        )
    ).delete(synchronize_session=False)
    db.query(NodeModel).filter(NodeModel.project_id == project_id).delete(synchronize_session=False)
    db.query(ChapterModel).filter(ChapterModel.project_id == project_id).delete(synchronize_session=False)
    db.query(GlossaryModel).filter(GlossaryModel.project_id == project_id).delete(synchronize_session=False)
    db.query(LayoutProfileModel).filter(LayoutProfileModel.project_id == project_id).delete(synchronize_session=False)
    db.query(ProjectModel).filter(ProjectModel.id == project_id).delete(synchronize_session=False)
    db.commit()


def _seed_bulk_project(project_id, issue_type="SEMANTIC_MISMATCH"):
    db = get_project_db(project_id)
    _clear_project_data(db, project_id)
    db.add(ProjectModel(id=project_id, title="Phase 2.1.2 QA", selected_model="mock"))
    db.add(ChapterModel(id=f"{project_id}_chapter", project_id=project_id, title="Revenue", order_index=0))
    db.add(NodeModel(
        id=f"{project_id}_node",
        project_id=project_id,
        chapter_id=f"{project_id}_chapter",
        node_type="paragraph",
        content="Revenue increased by 5%.",
        translated_content="Bản dịch cũ.",
        status="TRANSLATED",
        order_index=0,
    ))
    db.commit()
    db.add(QAIssueModel(
        id=f"{project_id}_issue",
        project_id=project_id,
        node_id=f"{project_id}_node",
        issue_type=issue_type,
        severity="ERROR",
        message="Cần kiểm tra lại ý nghĩa.",
        status="OPEN",
    ))
    db.commit()
    return db


def test_bulk_semantic_timeout_persists_qa_error_and_stays_fail_closed(monkeypatch):
    project_id = "phase2_1_2_bulk_timeout"
    provider = BulkReviewProvider(review_error=TimeoutError("provider timeout"))
    monkeypatch.setattr(translation_worker, "get_provider", lambda model_name: provider)
    db = _seed_bulk_project(project_id)
    try:
        response = TestClient(app).post(
            f"/api/projects/{project_id}/qa/retranslate_all_issues",
            json={"instruction": "Sửa lại bản dịch."},
        )
        assert response.status_code == 200

        db.expire_all()
        node = db.query(NodeModel).filter(NodeModel.id == f"{project_id}_node").one()
        original = db.query(QAIssueModel).filter(QAIssueModel.id == f"{project_id}_issue").one()
        qa_errors = db.query(QAIssueModel).filter(
            QAIssueModel.node_id == node.id,
            QAIssueModel.issue_type == "QA_ERROR",
        ).all()
        assert original.status == "OPEN"
        assert node.status == "NEEDS_REVIEW"
        assert node.translated_content == "Bản dịch cũ."
        assert len(qa_errors) == 1
        assert qa_errors[0].status == "OPEN"
        assert "provider timeout" in qa_errors[0].message
        assert db.query(TranslationMemoryModel).count() == 0
    finally:
        db.close()


def test_bulk_malformed_semantic_response_persists_qa_error(monkeypatch):
    project_id = "phase2_1_2_bulk_malformed"
    provider = BulkReviewProvider(review_result={"foo": "bar"})
    monkeypatch.setattr(translation_worker, "get_provider", lambda model_name: provider)
    db = _seed_bulk_project(project_id)
    try:
        response = TestClient(app).post(f"/api/projects/{project_id}/qa/retranslate_all_issues", json={})
        assert response.status_code == 200
        db.expire_all()
        issue = db.query(QAIssueModel).filter(
            QAIssueModel.project_id == project_id,
            QAIssueModel.issue_type == "QA_ERROR",
        ).one()
        original = db.query(QAIssueModel).filter(QAIssueModel.issue_type == "SEMANTIC_MISMATCH").one()
        assert original.status == "OPEN"
        assert issue.status == "OPEN"
        assert "is_passed" in issue.message
    finally:
        db.close()


def test_bulk_semantic_pass_resolves_issue_saves_node_and_tm(monkeypatch):
    project_id = "phase2_1_2_bulk_pass"
    provider = BulkReviewProvider(review_result={
        "is_passed": True,
        "score": 1.0,
        "issues": [],
        "suggested_revision": "",
    })
    monkeypatch.setattr(translation_worker, "get_provider", lambda model_name: provider)
    db = _seed_bulk_project(project_id)
    try:
        response = TestClient(app).post(f"/api/projects/{project_id}/qa/retranslate_all_issues", json={})
        assert response.status_code == 200
        db.expire_all()
        node = db.query(NodeModel).filter(NodeModel.id == f"{project_id}_node").one()
        issue = db.query(QAIssueModel).filter(QAIssueModel.id == f"{project_id}_issue").one()
        assert issue.status == "RESOLVED"
        assert node.status == "TRANSLATED"
        assert node.translated_content == "Doanh thu tăng 5%."
        assert db.query(QAIssueModel).filter(QAIssueModel.issue_type == "QA_ERROR").count() == 0
        assert db.query(TranslationMemoryModel).count() == 1
    finally:
        db.close()


@pytest.fixture()
def memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _tm_store(db, source, target, glossary=None):
    TranslationMemoryService.store(
        db,
        source,
        target,
        style_hash="style-hash",
        glossary_hash="glossary-hash",
        model_name="mock",
        prompt_version="phase2-1-2-test",
        locked_glossary=glossary,
    )


def test_tm_store_rejects_locked_glossary_mismatch_without_row(memory_db):
    with pytest.raises(ValueError, match="GLOSSARY_MISMATCH"):
        _tm_store(
            memory_db,
            "Cash flow improved.",
            "Luồng tiền đã cải thiện.",
            {"cash flow": "dòng tiền"},
        )
    assert memory_db.query(TranslationMemoryModel).count() == 0


def test_tm_store_rejects_added_number_and_negation_loss(memory_db):
    with pytest.raises(ValueError, match="NUMBER_ADDITION"):
        _tm_store(
            memory_db,
            "Revenue increased by 5%.",
            "Doanh thu tăng 5% và mục tiêu là 20%.",
        )
    with pytest.raises(ValueError, match="NEGATION_LOSS"):
        _tm_store(
            memory_db,
            "The company did not approve the transaction.",
            "Công ty đã phê duyệt giao dịch.",
        )
    assert memory_db.query(TranslationMemoryModel).count() == 0


def test_tm_store_validates_and_lookup_rechecks_current_glossary(memory_db):
    glossary = {"cash flow": "dòng tiền"}
    source = "Cash flow improved by 5%."
    target = "Dòng tiền cải thiện 5%."
    _tm_store(memory_db, source, target, glossary)
    assert memory_db.query(TranslationMemoryModel).count() == 1
    assert TranslationMemoryService.lookup(
        memory_db,
        source,
        "style-hash",
        "glossary-hash",
        "phase2-1-2-test",
        locked_glossary=glossary,
    ) == target
    assert TranslationMemoryService.lookup(
        memory_db,
        source,
        "style-hash",
        "glossary-hash",
        "phase2-1-2-test",
        locked_glossary={"cash flow": "luồng tiền"},
    ) is None


class FailedChapterMemoryProvider:
    def build_chapter_memory(self, text_sample, chapter_title, document_type, model=None):
        raise RuntimeError("provider unavailable")


def test_chapter_memory_provider_failure_does_not_copy_raw_source(tmp_path):
    node = DocumentNode(
        id="chapter-memory-fallback",
        type=NodeType.PARAGRAPH,
        content="Acme improved cash flow in a unique source paragraph.",
        status=NodeStatus.PENDING,
        order_index=0,
    )
    memory = ChapterMemoryBuilder.load_or_create(
        "fallback-chapter",
        "Growth",
        [node],
        tmp_path,
        {"cash flow": "dòng tiền"},
        FailedChapterMemoryProvider(),
        "mock",
        "BUSINESS",
    )
    assert "Acme improved cash flow in a unique source paragraph." not in memory.summary
    assert {item["source"] for item in memory.entities} >= {"cash flow"}


def test_benchmark_reports_added_fact_and_qa_metrics():
    from tests.translation_eval import _metric_bucket, _record_issues

    bucket = _metric_bucket()
    _record_issues(bucket, [
        {"code": "NUMBER_ADDITION"},
        {"code": "REFERENCE_ADDITION"},
        {"code": "URL_MISMATCH"},
        {"code": "URL_ADDITION"},
        {"code": "QA_ERROR"},
    ])
    assert bucket["number_addition"] == 1
    assert bucket["reference_addition"] == 1
    assert bucket["url_mismatch"] == 1
    assert bucket["url_addition"] == 1
    assert bucket["qa_error"] == 1

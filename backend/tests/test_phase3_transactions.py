from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import (
    Base,
    NodeModel,
    ProjectModel,
    QAIssueModel,
    SemanticReviewModel,
    TranslationMemoryModel,
    TranslationModel,
    TranslationVersionModel,
)
from app.db.repository import TranslationRepository
from app.services.translation.translation_commit import TranslationCommitService


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(ProjectModel(id="p3", title="Phase 3"))
    session.add(NodeModel(id="n1", project_id="p3", content="Revenue increased by 5%.", status="PENDING"))
    session.add(QAIssueModel(id="q1", project_id="p3", node_id="n1", issue_type="MEANING_DRIFT", severity="ERROR", message="Sai nghĩa", status="OPEN"))
    session.commit()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _signature():
    return SimpleNamespace(style_hash="style", glossary_hash="glossary", prompt_version="translation-v3.2-semantic")


def test_atomic_commit_persists_node_version_tm_issue_and_semantic_review(db):
    node = db.query(NodeModel).filter_by(id="n1").one()
    TranslationCommitService.commit_validated_translation(
        db, "p3", node, "Doanh thu tăng 5%.", _signature(), {}, "mock", "phase3",
        resolved_issue_ids=["q1"],
        semantic_result={"signature": "semantic-signature", "risk_score": 0.7, "risk_level": "HIGH", "status": "PASS", "score": 0.98},
    )
    assert db.query(NodeModel).filter_by(id="n1").one().translated_content == "Doanh thu tăng 5%."
    assert db.query(TranslationModel).count() == 1
    assert db.query(TranslationVersionModel).count() == 1
    assert db.query(TranslationMemoryModel).count() == 1
    assert db.query(QAIssueModel).filter_by(id="q1").one().status == "RESOLVED"
    assert db.query(SemanticReviewModel).count() == 1


def test_atomic_commit_rolls_back_tm_issue_and_node_when_node_save_fails(db, monkeypatch):
    node = db.query(NodeModel).filter_by(id="n1").one()
    original = TranslationRepository.save_node_translation

    def fail_after_staging(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError("node save failure")

    monkeypatch.setattr(TranslationRepository, "save_node_translation", fail_after_staging)
    with pytest.raises(RuntimeError, match="node save failure"):
        TranslationCommitService.commit_validated_translation(
            db, "p3", node, "Doanh thu tăng 5%.", _signature(), {}, "mock", "phase3",
            resolved_issue_ids=["q1"],
            semantic_result={"signature": "semantic-signature", "risk_level": "HIGH", "status": "PASS"},
        )

    db.expire_all()
    unchanged = db.query(NodeModel).filter_by(id="n1").one()
    assert unchanged.translated_content is None
    assert unchanged.status == "PENDING"
    assert unchanged.version == 1
    assert db.query(TranslationModel).count() == 0
    assert db.query(TranslationVersionModel).count() == 0
    assert db.query(TranslationMemoryModel).count() == 0
    assert db.query(SemanticReviewModel).count() == 0
    assert db.query(QAIssueModel).filter_by(id="q1").one().status == "OPEN"

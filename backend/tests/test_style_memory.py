from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, StyleMemoryModel, TranslationMemoryModel
from app.services.translation.style_memory import StyleMemoryService


def _db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _ingest(db, project_id="project-a", **kwargs):
    return StyleMemoryService.ingest_approved_example(
        db,
        project_id,
        kwargs.pop("source_text", "The company improved operations."),
        kwargs.pop("approved_vi", "Công ty cải thiện hoạt động."),
        domain=kwargs.pop("domain", "BUSINESS"),
        document_type=kwargs.pop("document_type", "BUSINESS"),
        translation_mode=kwargs.pop("translation_mode", "NATURAL"),
        node_type=kwargs.pop("node_type", "paragraph"),
        approval_source=kwargs.pop("approval_source", "QA_EDITOR"),
        explicit_approval=kwargs.pop("explicit_approval", True),
        semantic_approved=kwargs.pop("semantic_approved", True),
        **kwargs,
    )


def test_style_memory_requires_explicit_user_approval():
    db = _db()
    try:
        result = _ingest(db, explicit_approval=False)
        assert not result.ingested
        assert result.reason == "EXPLICIT_APPROVAL_REQUIRED"
        assert db.query(StyleMemoryModel).count() == 0
    finally:
        db.close()


def test_user_edited_qa_text_is_ingested_after_approval():
    db = _db()
    try:
        result = _ingest(db, approval_source="QA_EDITOR")
        assert result.ingested
        row = db.query(StyleMemoryModel).one()
        assert row.quality_status == "USER_APPROVED"
        assert row.approval_source == "QA_EDITOR"
        assert row.approved_vi == "Công ty cải thiện hoạt động."
    finally:
        db.close()


def test_retranslation_approval_is_a_valid_style_source():
    db = _db()
    try:
        result = _ingest(db, approval_source="RETRANSLATION_APPROVED")
        assert result.ingested
        assert db.query(StyleMemoryModel).one().approval_source == "RETRANSLATION_APPROVED"
    finally:
        db.close()


def test_style_memory_deduplicates_source_target_and_metadata():
    db = _db()
    try:
        first = _ingest(db)
        second = _ingest(db)
        assert first.ingested
        assert not second.ingested
        assert second.deduplicated
        assert second.reason == "DUPLICATE"
        assert db.query(StyleMemoryModel).count() == 1
    finally:
        db.close()


def test_style_memory_does_not_change_translation_memory():
    db = _db()
    try:
        before = db.query(TranslationMemoryModel).count()
        result = _ingest(db)
        assert result.ingested
        assert db.query(TranslationMemoryModel).count() == before
    finally:
        db.close()


def test_style_memory_rejects_raw_ai_or_unknown_provenance():
    db = _db()
    try:
        for source in ("AI", "UNAPPROVED", "MODEL_OUTPUT"):
            result = _ingest(db, approval_source=source)
            assert not result.ingested
            assert result.reason == "UNAPPROVED_SOURCE"
        assert db.query(StyleMemoryModel).count() == 0
    finally:
        db.close()

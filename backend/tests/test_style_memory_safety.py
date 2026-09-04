from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, StyleMemoryModel
from app.services.translation.quality_gate import TranslationQualityGate
from app.services.translation.style_memory import StyleMemoryService


def _db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_deterministic_error_never_enters_style_memory():
    db = _db()
    try:
        result = StyleMemoryService.ingest_approved_example(
            db, "p", "Revenue rose 10%.", "Doanh thu tăng 20%.",
            explicit_approval=True, semantic_approved=True,
        )
        assert not result.ingested
        assert result.reason == "DETERMINISTIC_GATE_FAILED"
        assert db.query(StyleMemoryModel).count() == 0
    finally:
        db.close()


def test_semantic_failure_never_enters_style_memory():
    db = _db()
    try:
        semantic = SimpleNamespace(approved=False, status="FAIL")
        result = StyleMemoryService.ingest_approved_example(
            db, "p", "The company improved operations.", "Công ty cải thiện hoạt động.",
            explicit_approval=True, semantic_result=semantic,
        )
        assert not result.ingested
        assert result.reason == "SEMANTIC_APPROVAL_REQUIRED"
        assert db.query(StyleMemoryModel).count() == 0
    finally:
        db.close()


def test_old_example_cannot_inject_its_facts_into_current_retrieval():
    db = _db()
    try:
        result = StyleMemoryService.ingest_approved_example(
            db,
            "p",
            "The old company had 10 billion in revenue.",
            "Công ty cũ có doanh thu 10 tỷ.",
            domain="BUSINESS",
            explicit_approval=True,
            semantic_approved=True,
        )
        assert result.ingested
        examples = StyleMemoryService.retrieve_examples(
            db, "p", "The current company has no revenue figure.", domain="BUSINESS",
        )
        assert examples
        # Service chỉ trả ví dụ tham khảo; nó không tạo candidate hay ghép facts vào source hiện tại.
        assert examples[0]["approved_vi"] == "Công ty cũ có doanh thu 10 tỷ."
        assert "current" not in examples[0]["approved_vi"].casefold()
    finally:
        db.close()


def test_style_memory_validation_accepts_warning_but_never_hides_hard_error():
    soft = TranslationQualityGate().validate(
        "The company improved operations.",
        "Công ty nâng cao vận hành.",
        [{"source_term": "operations", "preferred_target": "hoạt động", "lock_level": "SOFT"}],
    )
    hard = TranslationQualityGate().validate(
        "The company improved operations.",
        "Công ty nâng cao vận hành.",
        [{"source_term": "operations", "preferred_target": "hoạt động", "lock_level": "HARD"}],
    )
    assert soft.passed
    assert any(issue["code"] == "GLOSSARY_PREFERENCE" for issue in soft.issues)
    assert not hard.passed
    assert any(issue["code"] == "GLOSSARY_MISMATCH" for issue in hard.issues)


def test_retrieved_style_example_cannot_add_old_facts_to_current_translation():
    result = TranslationQualityGate().validate(
        "The current company improved revenue.",
        "Công ty hiện tại, Apple, có doanh thu 10 tỷ đô la.",
        {},
    )

    assert not result.passed
    assert any(issue["code"] == "NUMBER_ADDITION" for issue in result.issues)

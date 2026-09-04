from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, StyleMemoryModel
from app.services.translation.style_memory import StyleMemoryService


def _db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _add(db, project_id, source, target, domain, patterns=None, node_type="paragraph"):
    return StyleMemoryService.ingest_approved_example(
        db,
        project_id,
        source,
        target,
        domain=domain,
        document_type=domain,
        node_type=node_type,
        source_patterns=patterns,
        explicit_approval=True,
        semantic_approved=True,
    )


def test_retrieval_is_project_scoped_and_never_global():
    db = _db()
    try:
        _add(db, "p-a", "The board approved the strategy.", "Hội đồng quản trị phê duyệt chiến lược.", "BUSINESS")
        _add(db, "p-b", "The board approved the strategy.", "Hội đồng đã thông qua chiến lược.", "BUSINESS")
        rows = StyleMemoryService.retrieve_examples(
            db, "p-a", "The board reviewed the strategy.", domain="BUSINESS", document_type="BUSINESS",
        )
        assert len(rows) == 1
        assert rows[0]["approved_vi"] == "Hội đồng quản trị phê duyệt chiến lược."
    finally:
        db.close()


def test_retrieval_prefers_domain_pattern_and_returns_at_most_three():
    db = _db()
    try:
        for index in range(4):
            _add(
                db,
                "p",
                f"If the process is delayed, the team will review step {index}.",
                f"Nếu quy trình chậm trễ, nhóm sẽ rà soát bước {index}.",
                "TECHNICAL",
                ["conditional", "passive"],
            )
        _add(db, "p", "The market grew steadily.", "Thị trường tăng trưởng ổn định.", "FINANCE")
        rows = StyleMemoryService.retrieve_examples(
            db, "p", "If the service is delayed, the team will review the process.",
            domain="TECHNICAL", document_type="TECHNICAL", limit=10,
        )
        assert len(rows) == 3
        assert all(row["domain"] == "TECHNICAL" for row in rows)
        assert all("approved_vi" in row and "source_text" in row for row in rows)
    finally:
        db.close()


def test_retrieval_is_deterministic_and_uses_only_approved_rows():
    db = _db()
    try:
        _add(db, "p", "The report was revised.", "Báo cáo đã được chỉnh sửa.", "GENERAL")
        db.add(StyleMemoryModel(
            id="raw", project_id="p", source_text="The report was revised.",
            approved_vi="Báo cáo thô.", source_hash="raw-source", approved_hash="raw-target",
            domain="GENERAL", document_type="GENERAL", quality_status="UNREVIEWED",
        ))
        db.commit()
        kwargs = {"domain": "GENERAL", "document_type": "GENERAL", "translation_mode": "NATURAL"}
        first = StyleMemoryService.retrieve_examples(db, "p", "The report was revised.", **kwargs)
        second = StyleMemoryService.retrieve_examples(db, "p", "The report was revised.", **kwargs)
        assert first == second
        assert all(row["quality_status"] == "USER_APPROVED" for row in first)
        assert all(row["approved_vi"] != "Báo cáo thô." for row in first)
    finally:
        db.close()

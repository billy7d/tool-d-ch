from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, EntityDecisionModel, GlossaryModel, NodeModel, ProjectModel
from app.services.qa.global_consistency import GlobalConsistencyScanner
from app.services.translation.entity_ledger import EntityLedgerService


def _db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_locked_entity_validation_and_relevant_context_only():
    db = _db()
    db.add(ProjectModel(id="p", title="P"))
    db.add_all([
        EntityDecisionModel(id="e1", project_id="p", source_key="Federal Reserve", preferred_translation="Cục Dự trữ Liên bang Mỹ", locked=True),
        EntityDecisionModel(id="e2", project_id="p", source_key="OpenAI", preferred_translation="OpenAI", locked=False),
    ])
    db.commit()
    relevant = EntityLedgerService.relevant_decisions(db, "p", "The Federal Reserve met today.")
    assert relevant == {"Federal Reserve": "Cục Dự trữ Liên bang Mỹ"}
    assert EntityLedgerService.validate_locked(db, "p", "Federal Reserve met.", "Fed đã họp.")
    assert not EntityLedgerService.validate_locked(db, "p", "Federal Reserve met.", "Cục Dự trữ Liên bang Mỹ đã họp.")


def test_entity_extraction_avoids_heading_and_sentence_start_noise():
    assert EntityLedgerService.extract_source_entities("CHAPTER 1: Risk and Return") == []
    assert EntityLedgerService.extract_source_entities("High returns accompany uncertainty.") == []
    assert ("Federal Reserve", "OTHER") in EntityLedgerService.extract_source_entities("The Federal Reserve met today.")


def test_locked_glossary_has_priority_over_entity_ledger():
    db = _db()
    db.add(ProjectModel(id="p", title="P"))
    db.add(EntityDecisionModel(id="e1", project_id="p", source_key="Federal Reserve", preferred_translation="Fed", locked=True))
    db.commit()
    issues = EntityLedgerService.validate_locked(
        db, "p", "Federal Reserve met.", "Cục Dự trữ Liên bang đã họp.",
        {"Federal Reserve": "Cục Dự trữ Liên bang"},
    )
    assert issues == []


def test_global_consistency_flags_unlocked_variation_as_warning():
    db = _db()
    db.add(ProjectModel(id="p", title="P"))
    db.add(EntityDecisionModel(
        id="e1", project_id="p", source_key="Federal Reserve",
        preferred_translation="Cục Dự trữ Liên bang", aliases_json=["Fed"], locked=False,
    ))
    for index, target in enumerate(["Cục Dự trữ Liên bang", "Fed", "Ngân hàng trung ương"]):
        db.add(NodeModel(id=f"n{index}", project_id="p", content="Federal Reserve policy.", translated_content=f"{target} ban hành chính sách.", status="TRANSLATED", order_index=index))
    db.commit()
    findings = GlobalConsistencyScanner.scan_project(db, "p", persist=False)
    entity_finding = next(item for item in findings if item["issue_type"] == "ENTITY_VARIANT")
    assert entity_finding["severity"] == "WARNING"


def test_validated_observation_creates_unlocked_decision_without_overwrite():
    db = _db()
    db.add(ProjectModel(id="p", title="P"))
    db.commit()
    EntityLedgerService.observe_validated(db, "p", "n1", "OpenAI released a model.", "OpenAI đã phát hành một mô hình.")
    db.commit()
    decision = db.query(EntityDecisionModel).one()
    assert decision.source_key == "OpenAI"
    assert decision.preferred_translation == "OpenAI"
    assert decision.locked is False


def test_global_consistency_reports_term_and_repeated_phrase_warnings():
    db = _db()
    db.add(ProjectModel(id="p", title="P"))
    db.add(GlossaryModel(id="g", project_id="p", source_term="operating margin", target_term="biên lợi nhuận hoạt động", locked=False))
    db.add_all([
        NodeModel(id="n1", project_id="p", content="Operating margin improved.", translated_content="Biên lợi nhuận hoạt động đã cải thiện.", status="TRANSLATED"),
        NodeModel(id="n2", project_id="p", content="Operating margin improved.", translated_content="Biên hoạt động đã tăng.", status="TRANSLATED"),
    ])
    db.commit()
    findings = GlobalConsistencyScanner.scan_project(db, "p", persist=False)
    assert any(item["issue_type"] == "TERM_VARIANT" and item["severity"] == "WARNING" for item in findings)
    assert any(item["issue_type"] == "REPEATED_PHRASE_INCONSISTENCY" for item in findings)

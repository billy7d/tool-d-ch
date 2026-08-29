from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api import entities_router
from app.main import app
from app.db.models import (
    Base,
    EntityDecisionModel,
    GlossaryModel,
    NodeModel,
    ProjectModel,
    SemanticReviewModel,
)
from app.models.canonical import DocumentNode, NodeStatus, NodeType
from app.models.schemas import EntityDecisionCreate
from app.services.translation.context_memory import ChapterMemory
from app.services.translation.contextual_engine import ContextualTranslationEngine
from app.services.translation.document_profiler import DocumentTranslationProfile
from app.services.translation.entity_ledger import EntityLedgerService
from app.services.translation.mock_provider import MockProvider
from app.services.translation.model_capabilities import get_model_capabilities
from app.services.translation.quality_gate import TranslationQualityGate
from app.services.translation.semantic_assurance import SemanticAssuranceService
from app.services.translation.semantic_risk import SemanticRiskResult, SemanticRiskScorer


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(ProjectModel(id="p31", title="Phase 3.1"))
    session.commit()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _engine(db, **overrides):
    config = SimpleNamespace(
        document_type="GENERAL",
        translation_mode="NATURAL",
        model_name="mock",
        semantic_critic_model="mock",
        semantic_risk_medium=0.35,
        semantic_risk_high=0.65,
        semantic_max_repairs=2,
        custom_instructions="",
        register="ACCESSIBLE",
        sentence_style="MODERATE",
        style_guide={},
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    provider = MockProvider()
    engine = object.__new__(ContextualTranslationEngine)
    engine.db = db
    engine.project = SimpleNamespace(id="p31")
    engine.project_id = "p31"
    engine.provider = provider
    engine.config = config
    engine.locked_glossary = {}
    engine.capabilities = provider.get_model_capabilities("mock")
    engine.effective_context_window = min(
        engine.capabilities.context_window, engine.capabilities.recommended_context_window,
    )
    engine.system_prompt = ""
    engine.compact_system_prompt = ""
    engine.document_profile = DocumentTranslationProfile()
    engine._chapter_memories = {}
    engine.chapter_memory = lambda chapter: ChapterMemory()
    engine._rolling_context = lambda chapter, nodes, neighbors: []
    return engine


def _document_node(node_id="n31", content="The door is open."):
    return DocumentNode(
        id=node_id, type=NodeType.PARAGRAPH, content=content,
        status=NodeStatus.PENDING, order_index=0,
    )


def _pass_payload():
    return {
        "status": "PASS", "score": 0.98, "errors": [],
        "checks": {key: "PASS" for key in (
            "completeness", "meaning", "polarity", "modality",
            "causality", "scope", "entity_reference",
        )},
    }


class _RecordingProvider(MockProvider):
    def __init__(self):
        self.semantic_calls = 0

    def review_semantic_fidelity(self, *args, **kwargs):
        self.semantic_calls += 1
        return _pass_payload()


def test_negation_warning_reaches_risk_and_invokes_critic(db):
    engine = _engine(db)
    provider = _RecordingProvider()
    engine.provider = provider
    node = _document_node(content="The company expanded without compromising quality.")
    candidate = "Công ty mở rộng và vẫn duy trì chất lượng."
    quality = TranslationQualityGate().validate(node.content, candidate, {})
    warning_codes = [item["code"] for item in quality.issues if item["severity"] == "WARNING"]
    assert warning_codes == ["NEGATION_RISK"]
    result = SemanticAssuranceService.evaluate_candidate(engine, SimpleNamespace(), node, candidate, {}, max_repairs=0)
    assert result.risk_level == "MEDIUM"
    assert result.risk_score >= engine.config.semantic_risk_medium
    assert result.approved is True
    assert provider.semantic_calls == 1


def test_length_warning_increases_risk_without_new_detector():
    source = "This is a deliberately long source sentence used to exercise the deterministic length anomaly warning. " * 2
    base = SemanticRiskScorer.score(source, "Bản dịch đủ dài.")
    warned = SemanticRiskScorer.score(source, "Bản dịch đủ dài.", qa_warnings=["LENGTH_ANOMALY"])
    assert warned.score > base.score
    assert "QA_WARNING_LENGTH_ANOMALY" in warned.reasons


def test_deterministic_fail_cannot_be_semantic_approved(db):
    engine = _engine(db)
    provider = _RecordingProvider()
    engine.provider = provider
    node = _document_node(content="The company is not profitable.")
    result = SemanticAssuranceService.evaluate_candidate(engine, SimpleNamespace(), node, "Công ty có lợi nhuận.", {})
    assert result.approved is False
    assert result.status == "FAIL"
    assert any(item["type"] == "NEGATION_LOSS" for item in result.errors)
    assert provider.semantic_calls == 0


def test_clean_low_risk_node_still_skips_critic(db):
    engine = _engine(db)
    provider = _RecordingProvider()
    engine.provider = provider
    result = SemanticAssuranceService.evaluate_candidate(
        engine, SimpleNamespace(), _document_node(), "Cánh cửa đang mở.", {}, max_repairs=0,
    )
    assert result.risk_level == "LOW"
    assert result.status == "NOT_REQUIRED"
    assert result.approved is True
    assert provider.semantic_calls == 0


def test_same_engine_sees_new_entity_decision(db):
    engine = _engine(db)
    node_a = "OpenAI released a model."
    EntityLedgerService.observe_validated(db, "p31", "n-a", node_a, "OpenAI đã phát hành một mô hình.")
    db.commit()
    context = engine.assemble_context(SimpleNamespace(id="chapter"), [_document_node("n-b", "OpenAI released a safer model.")], "single")
    assert context.entity_decisions == {"OpenAI": "OpenAI"}


def test_same_engine_sees_edited_entity_decision(db):
    db.add(EntityDecisionModel(
        id="e31", project_id="p31", source_key="OpenAI",
        preferred_translation="OpenAI", entity_type="ORGANIZATION",
    ))
    db.commit()
    engine = _engine(db)
    row = db.query(EntityDecisionModel).filter_by(id="e31").one()
    row.preferred_translation = "Tổ chức OpenAI"
    row.revision = 2
    db.commit()
    context = engine.assemble_context(SimpleNamespace(id="chapter"), [_document_node("n-b", "OpenAI released a safer model.")], "single")
    assert context.entity_decisions == {"OpenAI": "Tổ chức OpenAI"}


def test_locked_glossary_overrides_entity_and_only_relevant_entities_enter_context(db):
    db.add_all([
        EntityDecisionModel(id="e1", project_id="p31", source_key="Federal Reserve", preferred_translation="Fed", entity_type="ORGANIZATION"),
        EntityDecisionModel(id="e2", project_id="p31", source_key="OpenAI", preferred_translation="OpenAI", entity_type="ORGANIZATION"),
    ])
    db.commit()
    engine = _engine(db)
    engine.locked_glossary = {"Federal Reserve": "Cục Dự trữ Liên bang"}
    context = engine.assemble_context(
        SimpleNamespace(id="chapter"),
        [_document_node("n-b", "Federal Reserve works with OpenAI.")],
        "single",
    )
    assert context.glossary == {"Federal Reserve": "Cục Dự trữ Liên bang"}
    assert context.entity_decisions == {"OpenAI": "OpenAI"}


def test_entity_context_filters_large_ledger_to_current_source(db):
    db.add(EntityDecisionModel(
        id="relevant", project_id="p31", source_key="OpenAI",
        preferred_translation="Tổ chức OpenAI", entity_type="ORGANIZATION",
    ))
    db.add_all([
        EntityDecisionModel(
            id=f"unrelated-{index}", project_id="p31", source_key=f"Unrelated Entity {index}",
            preferred_translation=f"Entity {index}", entity_type="OTHER",
        ) for index in range(100)
    ])
    db.commit()
    engine = _engine(db)
    context = engine.assemble_context(SimpleNamespace(id="chapter"), [_document_node("n-b", "OpenAI released a model.")], "single")
    assert context.entity_decisions == {"OpenAI": "Tổ chức OpenAI"}


def test_threshold_change_changes_signature_and_forces_critic(db, monkeypatch):
    def fixed_score(*args, **kwargs):
        value = 0.40
        medium = kwargs.get("medium_threshold", 0.35)
        high = kwargs.get("high_threshold", 0.65)
        level = "HIGH" if value >= high else "MEDIUM" if value >= medium else "LOW"
        return SemanticRiskResult(value, level, ["TEST_FIXED_SCORE"], level != "LOW")

    monkeypatch.setattr(SemanticRiskScorer, "score", staticmethod(fixed_score))
    engine = _engine(db, semantic_risk_medium=0.50)
    provider = _RecordingProvider()
    engine.provider = provider
    db_node = NodeModel(
        id="threshold-node", project_id="p31", content="The company grew.",
        translated_content="Công ty đã tăng trưởng.", status="TRANSLATED", order_index=0,
    )
    db.add(db_node)
    db.commit()
    first = SemanticAssuranceService.evaluate_candidate(engine, SimpleNamespace(), engine.canonical_node(db_node), db_node.translated_content, {}, max_repairs=0)
    SemanticAssuranceService.persist_existing_review(db, "p31", db_node, first, "mock")
    assert first.status == "NOT_REQUIRED"
    engine.config.semantic_risk_medium = 0.35
    second = SemanticAssuranceService.evaluate_candidate(engine, SimpleNamespace(), engine.canonical_node(db_node), db_node.translated_content, {}, max_repairs=0)
    assert second.signature != first.signature
    assert second.from_cache is False
    assert second.status == "PASS"
    assert provider.semantic_calls == 1


def _add_review(db, node_id: str, signature: str):
    db.add(SemanticReviewModel(
        id=f"review-{node_id}", project_id="p31", node_id=node_id,
        translation_version=1, signature=signature, risk_score=0.1,
        risk_level="LOW", critic_status="NOT_REQUIRED", is_stale=False,
    ))


def test_relevant_entity_invalidation_does_not_touch_unrelated_review(db):
    db.add_all([
        NodeModel(id="openai-node", project_id="p31", content="OpenAI released a model.", order_index=0),
        NodeModel(id="fed-node", project_id="p31", content="Federal Reserve met today.", order_index=1),
    ])
    db.commit()
    _add_review(db, "openai-node", "sig-openai")
    _add_review(db, "fed-node", "sig-fed")
    db.commit()
    changed = SemanticAssuranceService.invalidate_for_source_term(db, "p31", "OpenAI")
    db.commit()
    assert changed == 1
    assert db.query(SemanticReviewModel).filter_by(signature="sig-openai").one().is_stale is True
    assert db.query(SemanticReviewModel).filter_by(signature="sig-fed").one().is_stale is False


def test_relevant_glossary_invalidation_marks_affected_review(db):
    db.add(NodeModel(id="term-node", project_id="p31", content="Operating margin improved.", order_index=0))
    db.commit()
    _add_review(db, "term-node", "sig-term")
    db.commit()
    assert SemanticAssuranceService.invalidate_for_source_term(db, "p31", "operating margin") == 1
    db.commit()
    assert db.query(SemanticReviewModel).filter_by(signature="sig-term").one().is_stale is True


def _create(monkeypatch, db, payload):
    monkeypatch.setattr(entities_router, "get_project_db", lambda project_id: db)
    return entities_router.create_entity("p31", payload)


def test_manual_entity_creation_creates_user_decision_and_invalidates_reviews(db, monkeypatch):
    db.add(NodeModel(id="manual-node", project_id="p31", content="Federal Reserve met today.", order_index=0))
    db.commit()
    _add_review(db, "manual-node", "sig-manual")
    db.commit()
    result = _create(monkeypatch, db, EntityDecisionCreate(
        source_key=" Federal Reserve ", preferred_translation=" Cục Dự trữ Liên bang Mỹ ",
        entity_type="organization", aliases=["Fed"], locked=True,
    ))
    assert result["source_key"] == "Federal Reserve"
    assert result["source"] == "USER"
    assert result["confidence"] == 1.0
    assert result["locked"] is True
    assert db.query(EntityDecisionModel).one().preferred_translation == "Cục Dự trữ Liên bang Mỹ"
    assert db.query(SemanticReviewModel).one().is_stale is True


def test_manual_entity_duplicate_returns_controlled_conflict(db, monkeypatch):
    _create(monkeypatch, db, EntityDecisionCreate(source_key="OpenAI", preferred_translation="OpenAI"))
    with pytest.raises(HTTPException) as exc:
        _create(monkeypatch, db, EntityDecisionCreate(source_key=" OpenAI ", preferred_translation="Tổ chức OpenAI"))
    assert exc.value.status_code == 409
    assert exc.value.detail["code"] == "ENTITY_ALREADY_EXISTS"


def test_manual_entity_invalid_type_and_empty_values_are_rejected(db, monkeypatch):
    with pytest.raises(HTTPException) as invalid_type:
        _create(monkeypatch, db, EntityDecisionCreate(source_key="OpenAI", preferred_translation="OpenAI", entity_type="ALIEN"))
    assert invalid_type.value.status_code == 422
    with pytest.raises(HTTPException) as empty_value:
        _create(monkeypatch, db, EntityDecisionCreate(source_key=" ", preferred_translation=" "))
    assert empty_value.value.status_code == 422


def test_manual_entity_glossary_conflict_is_rejected(db, monkeypatch):
    db.add(GlossaryModel(
        id="g31", project_id="p31", source_term="Federal Reserve",
        target_term="Cục Dự trữ Liên bang", locked=True,
    ))
    db.commit()
    with pytest.raises(HTTPException) as exc:
        _create(monkeypatch, db, EntityDecisionCreate(
            source_key="Federal Reserve", preferred_translation="Fed", locked=True,
        ))
    assert exc.value.status_code == 409
    assert exc.value.detail["code"] == "ENTITY_GLOSSARY_CONFLICT"


def test_manual_entity_is_visible_to_existing_engine_without_restart(db, monkeypatch):
    engine = _engine(db)
    _create(monkeypatch, db, EntityDecisionCreate(
        source_key="OpenAI", preferred_translation="Tổ chức OpenAI", locked=False,
    ))
    context = engine.assemble_context(SimpleNamespace(id="chapter"), [_document_node("n-next", "OpenAI released a model.")], "single")
    assert context.entity_decisions == {"OpenAI": "Tổ chức OpenAI"}


def test_entity_patch_is_visible_to_existing_engine(db, monkeypatch):
    engine = _engine(db)
    created = _create(monkeypatch, db, EntityDecisionCreate(
        source_key="OpenAI", preferred_translation="OpenAI", locked=False,
    ))
    updated = entities_router.update_entity("p31", created["id"], {"preferred_translation": "Tổ chức OpenAI"})
    assert updated["preferred_translation"] == "Tổ chức OpenAI"
    context = engine.assemble_context(SimpleNamespace(id="chapter"), [_document_node("n-next", "OpenAI released a model.")], "single")
    assert context.entity_decisions == {"OpenAI": "Tổ chức OpenAI"}


def test_manual_entity_requires_existing_project(db, monkeypatch):
    monkeypatch.setattr(entities_router, "get_project_db", lambda project_id: db)
    with pytest.raises(HTTPException) as exc:
        entities_router.create_entity("missing-project", EntityDecisionCreate(source_key="OpenAI", preferred_translation="OpenAI"))
    assert exc.value.status_code == 404
    assert exc.value.detail["code"] == "PROJECT_NOT_FOUND"


def test_post_entity_http_contract_returns_created_and_conflict(db, monkeypatch):
    monkeypatch.setattr(entities_router, "get_project_db", lambda project_id: db)
    client = TestClient(app)
    payload = {
        "source_key": "OpenAI",
        "preferred_translation": "Tổ chức OpenAI",
        "entity_type": "ORGANIZATION",
        "aliases": ["OA"],
        "locked": True,
    }
    created = client.post("/api/projects/p31/entities", json=payload)
    assert created.status_code == 201
    assert created.json()["source"] == "USER"
    duplicate = client.post("/api/projects/p31/entities", json=payload)
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "ENTITY_ALREADY_EXISTS"

from app.services.qa.semantic_critic import SemanticCritic, validate_semantic_result
from app.services.translation.mock_provider import MockProvider
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, NodeModel, ProjectModel, TranslationMemoryModel
from app.models.canonical import DocumentNode, NodeStatus, NodeType
from app.services.translation.contextual_engine import EngineNodeResult
from app.services.translation.quality_gate import TranslationQualityGate
from app.services.translation.semantic_assurance import SemanticAssuranceService
from app.services.translation.semantic_risk import SemanticRiskScorer


def test_semantic_risk_is_selective_and_deterministic():
    simple = SemanticRiskScorer.score("The door is open.", "Cánh cửa đang mở.")
    conditional = SemanticRiskScorer.score(
        "Unless the company obtains approval, it may not distribute the funds.",
        "Trừ khi công ty được phê duyệt, công ty có thể không phân phối tiền.",
        document_type="LEGAL",
    )
    repaired = SemanticRiskScorer.score("The borrower shall comply.", previous_repairs=1, document_type="LEGAL")
    assert simple.level == "LOW" and simple.requires_critic is False
    assert conditional.level == "HIGH" and conditional.requires_critic is True
    assert repaired.level == "HIGH"


def test_semantic_critic_accepts_natural_nonliteral_vietnamese():
    result = SemanticCritic.review(
        MockProvider(), "It is raining heavily.", "Mưa như trút nước.", {}, {}, "LITERATURE", "mock",
    )
    assert result.status == "PASS"
    assert result.errors == []


def test_semantic_critic_detects_seeded_error_categories():
    provider = MockProvider()
    for marker, issue_type in [
        ("[[OMISSION]]", "SEMANTIC_OMISSION"),
        ("[[ADDITION]]", "SEMANTIC_ADDITION"),
        ("[[MODALITY]]", "MODALITY_ERROR"),
        ("[[CAUSALITY]]", "CAUSALITY_ERROR"),
        ("[[SCOPE]]", "SCOPE_ERROR"),
    ]:
        result = SemanticCritic.review(provider, "Source", f"Candidate {marker}", {}, {}, "GENERAL", "mock")
        assert result.status == "FAIL"
        assert result.errors[0]["type"] == issue_type


def test_malformed_semantic_provider_fails_closed():
    class BadProvider(MockProvider):
        def review_semantic_fidelity(self, *args, **kwargs):
            return {"status": "PASS"}

    result = SemanticCritic.review(BadProvider(), "Source", "Bản dịch", {}, {}, "GENERAL", "mock")
    assert result.status == "ERROR"
    assert result.errors[0]["type"] == "SEMANTIC_QA_ERROR"


def test_semantic_validator_rejects_contradictory_pass():
    try:
        validate_semantic_result({
            "status": "PASS", "score": 1.0,
            "errors": [{"type": "MEANING_DRIFT", "message": "Sai"}],
            "checks": {key: "PASS" for key in ("completeness", "meaning", "polarity", "modality", "causality", "scope", "entity_reference")},
        })
        assert False, "Kết quả mâu thuẫn phải bị từ chối"
    except ValueError:
        pass


def _semantic_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(ProjectModel(id="p", title="P"))
    db.commit()
    return db


class _RepairEngine:
    def __init__(self, db, provider):
        self.db = db
        self.provider = provider
        self.project_id = "p"
        self.locked_glossary = {}
        self.config = SimpleNamespace(
            document_type="LEGAL", model_name="mock", semantic_critic_model="mock",
            semantic_risk_medium=0.35, semantic_risk_high=0.65, semantic_max_repairs=2,
        )
        self.repair_sources = []

    def repair_node(self, chapter, node, issues, max_attempts=1):
        self.repair_sources.append(node.content)
        candidate = "Trừ khi có phê duyệt, công ty không được phân phối tiền."
        quality = TranslationQualityGate().validate(node.content, candidate, {})
        return EngineNodeResult(node.id, candidate, quality, quality.passed)

    @staticmethod
    def canonical_node(node):
        return DocumentNode(
            id=node.id, type=NodeType.PARAGRAPH, content=node.content,
            status=NodeStatus.PENDING, order_index=0,
        )


class _SequenceProvider(MockProvider):
    def __init__(self, always_fail=False):
        self.calls = 0
        self.always_fail = always_fail

    def review_semantic_fidelity(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1 or self.always_fail:
            return {
                "status": "FAIL", "score": 0.3,
                "errors": [{"type": "CONDITION_ERROR", "message": "Mất điều kiện."}],
                "checks": {
                    "completeness": "PASS", "meaning": "FAIL", "polarity": "PASS",
                    "modality": "PASS", "causality": "PASS", "scope": "PASS", "entity_reference": "PASS",
                },
            }
        return {
            "status": "PASS", "score": 0.98, "errors": [],
            "checks": {key: "PASS" for key in ("completeness", "meaning", "polarity", "modality", "causality", "scope", "entity_reference")},
        }


def test_semantic_repair_uses_original_source_then_revalidates_and_passes():
    db = _semantic_db()
    provider = _SequenceProvider()
    engine = _RepairEngine(db, provider)
    node = DocumentNode(
        id="n", type=NodeType.PARAGRAPH,
        content="Unless approval is obtained, the company may not distribute the funds.",
        status=NodeStatus.PENDING, order_index=0,
    )
    result = SemanticAssuranceService.evaluate_candidate(
        engine, SimpleNamespace(), node, "Công ty có thể phân phối tiền.", {}, max_repairs=2,
    )
    assert result.approved is True
    assert result.status == "PASS"
    assert result.repair_attempts == 1
    assert engine.repair_sources == [node.content]
    assert provider.calls == 2


def test_semantic_repair_is_bounded_and_rejected_candidate_never_enters_tm():
    db = _semantic_db()
    provider = _SequenceProvider(always_fail=True)
    engine = _RepairEngine(db, provider)
    db_node = NodeModel(
        id="n", project_id="p", content="Unless approval is obtained, the company may not distribute the funds.",
        translated_content="Bản dịch cũ.", status="PENDING", order_index=0,
    )
    db.add(db_node)
    db.commit()
    result = SemanticAssuranceService.evaluate_candidate(
        engine, SimpleNamespace(), engine.canonical_node(db_node),
        "Công ty có thể phân phối tiền.", {}, max_repairs=2,
    )
    SemanticAssuranceService.persist_rejected(db, "p", db_node, result, "mock")
    assert result.approved is False
    assert result.repair_attempts == 2
    assert provider.calls == 3
    assert db.query(TranslationMemoryModel).count() == 0
    assert db.query(NodeModel).filter_by(id="n").one().translated_content == "Bản dịch cũ."
    assert db.query(NodeModel).filter_by(id="n").one().status == "NEEDS_REVIEW"


def test_semantic_review_cache_survives_new_engine_and_avoids_duplicate_call():
    db = _semantic_db()
    node = NodeModel(
        id="n", project_id="p", content="Unless approval is obtained, the company may not distribute the funds.",
        translated_content="Trừ khi có phê duyệt, công ty không được phân phối tiền.", status="TRANSLATED", order_index=0,
    )
    db.add(node)
    db.commit()
    first_provider = _SequenceProvider()
    first_provider.calls = 1  # Lần kế tiếp trả PASS để tạo cache hợp lệ.
    first_engine = _RepairEngine(db, first_provider)
    first = SemanticAssuranceService.evaluate_candidate(
        first_engine, SimpleNamespace(), first_engine.canonical_node(node), node.translated_content, {}, max_repairs=0,
    )
    SemanticAssuranceService.persist_existing_review(db, "p", node, first, "mock")

    second_provider = _SequenceProvider(always_fail=True)
    second_engine = _RepairEngine(db, second_provider)
    cached = SemanticAssuranceService.evaluate_candidate(
        second_engine, SimpleNamespace(), second_engine.canonical_node(node), node.translated_content, {}, max_repairs=0,
    )
    assert cached.approved is True
    assert cached.from_cache is True
    assert second_provider.calls == 0

    SemanticAssuranceService.invalidate_for_nodes(db, "p", ["n"])
    db.commit()
    after_change = SemanticAssuranceService.evaluate_candidate(
        second_engine, SimpleNamespace(), second_engine.canonical_node(node), node.translated_content, {}, max_repairs=0,
    )
    assert after_change.approved is False
    assert after_change.from_cache is False
    assert second_provider.calls == 1

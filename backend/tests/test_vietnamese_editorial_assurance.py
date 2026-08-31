from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, NodeModel, ProjectModel, QAIssueModel, TranslationMemoryModel
from app.models.canonical import DocumentNode, NodeStatus, NodeType
from app.services.qa.vietnamese_naturalness_critic import NATURALNESS_CHECKS
from app.services.translation.mock_provider import MockProvider
from app.services.translation.semantic_assurance import SemanticAssuranceService


def _checks(value="PASS"):
    return {key: value for key in NATURALNESS_CHECKS}


def _naturalness(status="PASS", score=0.96, issue_type="LITERAL_CALQUE"):
    if status == "PASS":
        return {"status": "PASS", "score": score, "issues": [], "checks": _checks()}
    checks = _checks()
    checks["literal_calque"] = "FAIL"
    return {
        "status": "FAIL",
        "score": score,
        "issues": [{
            "type": issue_type,
            "severity": "WARNING",
            "target_span": "candidate",
            "message": "Câu còn mang dấu vết dịch sát.",
        }],
        "checks": checks,
    }


def _semantic_pass():
    return {
        "status": "PASS",
        "score": 0.98,
        "errors": [],
        "checks": {key: "PASS" for key in (
            "completeness", "meaning", "polarity", "modality", "causality", "scope", "entity_reference",
        )},
    }


def _semantic_failure(issue_type, check):
    checks = {key: "PASS" for key in (
        "completeness", "meaning", "polarity", "modality", "causality", "scope", "entity_reference",
    )}
    checks[check] = "FAIL"
    return {
        "status": "FAIL",
        "score": 0.32,
        "errors": [{"type": issue_type, "message": "Editorial rewrite làm thay đổi ý nguồn."}],
        "checks": checks,
    }


class ScriptedProvider(MockProvider):
    def __init__(self, naturalness_results, rewritten, semantic_results=None):
        self.naturalness_results = list(naturalness_results)
        self.rewritten = rewritten
        self.semantic_results = list(semantic_results or [_semantic_pass()])
        self.naturalness_calls = 0
        self.editorial_calls = 0
        self.semantic_calls = 0
        self.editorial_inputs = []

    def review_naturalness(self, *args, **kwargs):
        self.naturalness_calls += 1
        if self.naturalness_results:
            return self.naturalness_results.pop(0)
        return _naturalness("PASS")

    def editorial_rewrite(self, *args, **kwargs):
        self.editorial_calls += 1
        self.editorial_inputs.append(kwargs)
        return self.rewritten

    def review_semantic_fidelity(self, *args, **kwargs):
        self.semantic_calls += 1
        if self.semantic_results:
            return self.semantic_results.pop(0)
        return _semantic_pass()


class MarkerSemanticProvider(ScriptedProvider):
    def __init__(self, marker, issue_type, check, rewritten):
        super().__init__([_naturalness("FAIL", 0.58)], rewritten)
        self.marker = marker
        self.semantic_failure = (issue_type, check)

    def review_semantic_fidelity(self, source_text, translated_text, *args, **kwargs):
        if self.marker in translated_text:
            return _semantic_failure(*self.semantic_failure)
        return _semantic_pass()


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(ProjectModel(id="editorial-p0", title="Editorial P0"))
    session.commit()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _engine(db, provider, source, glossary=None, **config_overrides):
    node = NodeModel(
        id="editorial-node",
        project_id="editorial-p0",
        content=source,
        node_type="paragraph",
        status="PENDING",
        order_index=0,
        version=1,
    )
    db.add(node)
    db.commit()
    config = SimpleNamespace(
        document_type="GENERAL",
        translation_mode="NATURAL",
        model_name="mock",
        semantic_critic_model="mock",
        naturalness_critic_model="mock",
        semantic_risk_medium=0.35,
        semantic_risk_high=0.65,
        semantic_max_repairs=2,
        naturalness_pass_threshold=0.80,
        naturalness_rewrite_threshold=0.55,
        editorial_max_rewrites=1,
        quality_profile="HIGH_QUALITY",
        custom_instructions="",
        register="NEUTRAL",
        sentence_style="MODERATE",
        style_guide={},
    )
    for key, value in config_overrides.items():
        setattr(config, key, value)
    engine_stub = SimpleNamespace(
        db=db,
        project_id="editorial-p0",
        provider=provider,
        config=config,
        locked_glossary=dict(glossary or {}),
        canonical_node=lambda item: DocumentNode(
            id=item.id,
            type=NodeType.PARAGRAPH,
            content=item.content,
            status=NodeStatus.PENDING,
            order_index=item.order_index,
        ),
    )
    engine_stub._rolling_context = lambda chapter, nodes, neighbors: []
    return engine_stub, node


def _signature():
    return SimpleNamespace(style_hash="style", glossary_hash="glossary", prompt_version="translation-v3.3-naturalness-p0")


def _chapter():
    return SimpleNamespace(id="chapter-editorial", title="Chapter")


def test_naturalness_pass_commits_without_editorial_rewrite(db):
    provider = ScriptedProvider([_naturalness("PASS")], "Không được dùng")
    engine, node = _engine(db, provider, "The door is open.")

    result = SemanticAssuranceService.assure_and_commit(
        engine, _chapter(), node, "Cánh cửa đang mở.", _signature(), "mock", "test",
    )

    assert result.approved is True
    assert result.naturalness_status == "PASS"
    assert result.editorial_rewrite_count == 0
    assert provider.editorial_calls == 0
    assert db.query(NodeModel).filter_by(id=node.id).one().translated_content == "Cánh cửa đang mở."
    assert db.query(TranslationMemoryModel).count() == 1


def test_medium_naturalness_rewrites_once_and_commits_final_candidate(db):
    source = "The company decided to take action immediately."
    literal = "Công ty đã đưa ra quyết định để thực hiện hành động ngay lập tức."
    natural = "Công ty quyết định hành động ngay."
    provider = ScriptedProvider([_naturalness("FAIL", 0.58), _naturalness("PASS")], natural)
    engine, node = _engine(db, provider, source)

    result = SemanticAssuranceService.assure_and_commit(
        engine, _chapter(), node, literal, _signature(), "mock", "test",
    )

    saved = db.query(NodeModel).filter_by(id=node.id).one()
    assert result.approved is True
    assert result.editorial_rewrite_count == 1
    assert result.editorial_rewrite_success is True
    assert provider.editorial_calls == 1
    assert provider.naturalness_calls == 2
    assert provider.editorial_inputs[0]["source_text"] == source
    assert provider.editorial_inputs[0]["current_translation"] == literal
    assert saved.translated_content == natural
    assert db.query(TranslationMemoryModel).one().translated_text == natural


def test_low_naturalness_needs_review_without_rewrite_or_persistence(db):
    provider = ScriptedProvider([_naturalness("FAIL", 0.42, "SENTENCE_FLOW")], "Không được dùng")
    engine, node = _engine(db, provider, "The explanation is difficult to follow.")

    result = SemanticAssuranceService.assure_and_commit(
        engine, _chapter(), node, "Lời giải thích khó để theo dõi.", _signature(), "mock", "test",
    )

    saved = db.query(NodeModel).filter_by(id=node.id).one()
    assert result.approved is False
    assert result.publication_status == "NEEDS_REVIEW"
    assert provider.editorial_calls == 0
    assert saved.translated_content is None
    assert saved.status == "NEEDS_REVIEW"
    assert db.query(TranslationMemoryModel).count() == 0
    assert db.query(QAIssueModel).filter_by(issue_type="VIETNAMESE_SENTENCE_FLOW").count() == 1


def test_editorial_rewrite_losing_number_is_rejected(db):
    source = "Revenue rose 12%."
    provider = ScriptedProvider([_naturalness("FAIL", 0.58)], "Doanh thu tăng.")
    engine, node = _engine(db, provider, source)

    result = SemanticAssuranceService.assure_and_commit(
        engine, _chapter(), node, "Doanh thu đã thực hiện việc tăng 12%.", _signature(), "mock", "test",
    )

    saved = db.query(NodeModel).filter_by(id=node.id).one()
    assert result.approved is False
    assert any(issue["type"] == "NUMBER_MISMATCH" for issue in result.errors)
    assert saved.translated_content is None
    assert db.query(TranslationMemoryModel).count() == 0


def test_editorial_rewrite_losing_negation_is_rejected(db):
    source = "There is not enough evidence to reject the hypothesis."
    provider = ScriptedProvider([_naturalness("FAIL", 0.58)], "Có đủ bằng chứng để bác bỏ giả thuyết.")
    engine, node = _engine(db, provider, source)

    result = SemanticAssuranceService.assure_and_commit(
        engine, _chapter(), node, "Không có đủ bằng chứng để bác bỏ giả thuyết.", _signature(), "mock", "test",
    )

    assert result.approved is False
    assert any(issue["type"] == "NEGATION_LOSS" for issue in result.errors)
    assert db.query(NodeModel).filter_by(id=node.id).one().translated_content is None
    assert db.query(TranslationMemoryModel).count() == 0


def test_editorial_rewrite_changing_locked_glossary_is_rejected(db):
    source = "Cash flow improved."
    provider = ScriptedProvider([_naturalness("FAIL", 0.58)], "Luồng tiền cải thiện.")
    engine, node = _engine(db, provider, source, glossary={"Cash flow": "dòng tiền"})

    result = SemanticAssuranceService.assure_and_commit(
        engine, _chapter(), node, "Dòng tiền đã thực hiện việc cải thiện.", _signature(), "mock", "test",
    )

    assert result.approved is False
    assert any(issue["type"] == "GLOSSARY_MISMATCH" for issue in result.errors)
    assert db.query(TranslationMemoryModel).count() == 0


def test_editorial_rewrite_semantic_drift_is_rejected(db):
    source = "The company may reduce production."
    provider = ScriptedProvider(
        [_naturalness("FAIL", 0.58), _naturalness("PASS")],
        "Công ty sẽ giảm sản lượng.",
        semantic_results=[{
            "status": "FAIL", "score": 0.32,
            "errors": [{"type": "MODALITY_ERROR", "message": "May bị đổi thành sẽ."}],
            "checks": {
                "completeness": "PASS", "meaning": "FAIL", "polarity": "PASS", "modality": "FAIL",
                "causality": "PASS", "scope": "PASS", "entity_reference": "PASS",
            },
        }],
    )
    engine, node = _engine(db, provider, source)

    result = SemanticAssuranceService.assure_and_commit(
        engine, _chapter(), node, "Công ty có thể đưa ra quyết định để giảm sản lượng.", _signature(), "mock", "test",
    )

    assert result.approved is False
    assert any(issue["type"] == "MODALITY_ERROR" for issue in result.errors)
    assert db.query(NodeModel).filter_by(id=node.id).one().translated_content is None
    assert db.query(TranslationMemoryModel).count() == 0


@pytest.mark.parametrize(
    "source,initial,revised,marker,issue_type,check",
    [
        (
            "The license may be revoked only if the holder breaches the terms.",
            "Giấy phép có thể bị thu hồi chỉ khi người sở hữu vi phạm các điều khoản.",
            "Giấy phép có thể bị thu hồi nếu người sở hữu vi phạm các điều khoản. [[CONDITION]]",
            "[[CONDITION]]", "CONDITION_ERROR", "scope",
        ),
        (
            "Incident A caused incident B.",
            "Sự cố A đã gây ra sự cố B.",
            "Sự cố B chỉ xảy ra sau sự cố A. [[CAUSALITY]]",
            "[[CAUSALITY]]", "CAUSALITY_ERROR", "causality",
        ),
        (
            "Acme launched Nova.",
            "Acme đã ra mắt Nova.",
            "Công ty khác đã ra mắt Nova. [[ENTITY]]",
            "[[ENTITY]]", "ENTITY_REFERENCE_ERROR", "entity_reference",
        ),
    ],
)
def test_editorial_rewrite_preserves_condition_causality_and_entities(
    db, source, initial, revised, marker, issue_type, check,
):
    provider = MarkerSemanticProvider(marker, issue_type, check, revised)
    engine, node = _engine(db, provider, source)

    result = SemanticAssuranceService.assure_and_commit(
        engine, _chapter(), node, initial, _signature(), "mock", "test",
    )

    assert result.approved is False
    assert any(issue["type"] == issue_type for issue in result.errors)
    assert db.query(NodeModel).filter_by(id=node.id).one().translated_content is None
    assert db.query(TranslationMemoryModel).count() == 0


def test_rewrite_that_remains_unnatural_needs_review_and_does_not_loop(db):
    literal = "Công ty đã đưa ra quyết định để thực hiện hành động ngay lập tức."
    provider = ScriptedProvider(
        [_naturalness("FAIL", 0.58), _naturalness("FAIL", 0.58)], literal,
    )
    engine, node = _engine(db, provider, "The company decided to take action immediately.")

    result = SemanticAssuranceService.assure_and_commit(
        engine, _chapter(), node, literal, _signature(), "mock", "test",
    )

    assert result.approved is False
    assert result.editorial_rewrite_count == 1
    assert provider.editorial_calls == 1
    assert provider.naturalness_calls == 2
    assert db.query(NodeModel).filter_by(id=node.id).one().translated_content is None
    assert db.query(TranslationMemoryModel).count() == 0


def test_naturalness_error_is_fail_closed_without_editorial_rewrite(db):
    class ErrorProvider(ScriptedProvider):
        def review_naturalness(self, *args, **kwargs):
            self.naturalness_calls += 1
            raise RuntimeError("critic timeout")

    provider = ErrorProvider([], "Không được dùng")
    engine, node = _engine(db, provider, "The door is open.")

    result = SemanticAssuranceService.assure_and_commit(
        engine, _chapter(), node, "Cánh cửa đang mở.", _signature(), "mock", "test",
    )

    assert result.approved is False
    assert result.naturalness_status == "ERROR"
    assert result.publication_status == "NEEDS_REVIEW"
    assert provider.editorial_calls == 0
    assert db.query(NodeModel).filter_by(id=node.id).one().translated_content is None
    assert db.query(TranslationMemoryModel).count() == 0

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, NodeModel, ProjectModel, StyleRuleModel, TranslationModel
from app.models.canonical import DocumentNode, NodeStatus, NodeType, TranslationMode
from app.services.qa.vietnamese_naturalness_critic import (
    NATURALNESS_CHECKS,
    compare_original_and_rewrite,
    validate_naturalness_result,
)
from app.services.translation.address_terms import AddressTermLedger
from app.services.translation.context_assembler import TranslationContext
from app.services.translation.mock_provider import MockProvider
from app.services.translation.prompt_builder import PromptBuilder
from app.services.translation.semantic_assurance import SemanticAssuranceService
from app.services.translation.style_rules import StyleRuleService
from app.services.translation.translation_eval_metrics import compare_regression, summarize_evaluations


LITERATURE_GUIDE = {
    "address_terms": {
        "narrator": "tôi",
        "characters": {"Alice": "cô"},
        "pairs": [{"speaker": "Alice", "listener": "Bob", "self": "em", "other": "anh", "note": "người yêu"}],
    }
}


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(ProjectModel(id="p1", title="P1 upgrades"))
    session.commit()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


# ---------- Sổ xưng hô ----------

def test_address_ledger_uses_domain_default_reader_unless_configured():
    assert AddressTermLedger.from_style_guide({}, "SELF_HELP").reader == "bạn"
    assert AddressTermLedger.from_style_guide({}, "LITERATURE").reader == ""
    # Người dùng xóa trống có chủ đích thì không áp mặc định.
    assert AddressTermLedger.from_style_guide({"address_terms": {"reader": ""}}, "SELF_HELP").reader == ""


def test_address_ledger_keeps_only_entries_relevant_to_source():
    ledger = AddressTermLedger.from_style_guide(LITERATURE_GUIDE, "LITERATURE")
    relevant = ledger.relevant("Alice looked at Bob and smiled.")
    assert relevant.narrator == ""
    assert relevant.characters == {"Alice": "cô"}
    assert len(relevant.pairs) == 1
    lines = relevant.prompt_lines()
    assert any('dịch là "em"' in line and 'dịch là "anh"' in line for line in lines)
    assert ledger.relevant("The rain kept falling.").is_empty


def test_address_check_flags_reader_mismatch_but_ignores_dialogue_and_plural():
    ledger = AddressTermLedger.from_style_guide({"address_terms": {"reader": "bạn"}}, "SELF_HELP")
    source = "You can start today."
    assert [item["target_span"] for item in ledger.check(source, "Quý vị có thể bắt đầu hôm nay.")] == ["quý vị"]
    assert ledger.check(source, "Các bạn có thể bắt đầu hôm nay.") == []
    assert ledger.check(source, "“Cậu bắt đầu đi”, anh nói. Bạn có thể bắt đầu hôm nay.") == []


# ---------- Prompt ----------

def test_natural_prompt_contains_vietnamese_style_guide_and_conventions():
    prompt = PromptBuilder.build_system_prompt(document_type="LITERATURE", translation_mode=TranslationMode.NATURAL)
    assert "VIETNAMESE STYLE GUIDE" in prompt
    assert "Hội đồng đã phê duyệt đề xuất." in prompt
    assert "Vietnamese conventions:" in prompt and "Lời thoại" in prompt


def test_faithful_prompt_skips_naturalizing_style_guide():
    prompt = PromptBuilder.build_system_prompt(document_type="LEGAL", translation_mode=TranslationMode.FAITHFUL)
    assert "VIETNAMESE STYLE GUIDE" not in prompt
    assert "Viết hoa 'Điều'" in prompt


def test_address_terms_are_not_dumped_into_optional_style_notes():
    prompt = PromptBuilder.build_system_prompt(
        document_type="LITERATURE", translation_mode=TranslationMode.NATURAL, style_guide=LITERATURE_GUIDE,
    )
    assert "address_terms" not in prompt


def test_context_parts_render_address_terms_and_style_rules():
    context = TranslationContext(
        address_terms=['- Người đọc ("you" của tác giả): gọi là "bạn".'],
        style_rules=['- Tránh "một cách rõ ràng", dùng "rõ" (2 lần sửa).'],
    )
    rendered = "\n\n".join(PromptBuilder._context_parts(context))
    assert "ADDRESS TERMS (XƯNG HÔ)" in rendered
    assert "PROJECT STYLE RULES" in rendered
    assert 'gọi là "bạn"' in rendered


# ---------- Quy tắc học từ chỉnh sửa ----------

def test_extract_edit_rules_keeps_short_style_edits_only():
    rules = StyleRuleService.extract_edit_rules(
        "Anh ấy giải thích một cách rõ ràng về kế hoạch năm 2025.",
        "Anh ấy giải thích rõ về kế hoạch năm 2025.",
    )
    assert rules == [("một cách rõ ràng", "rõ")]
    # Đổi số liệu không phải là quy tắc văn phong.
    assert StyleRuleService.extract_edit_rules("Doanh thu tăng 12%.", "Doanh thu tăng 15%.") == []
    # Viết lại gần như toàn bộ câu thì bỏ qua.
    assert StyleRuleService.extract_edit_rules("Một câu hoàn toàn khác.", "Hôm nay trời rất đẹp và mát.") == []


def test_rule_activates_after_two_distinct_nodes_and_is_not_double_counted(db):
    ai, edited = "Cô giải thích một cách rõ ràng.", "Cô giải thích rõ."
    StyleRuleService.observe_edit(db, "p1", "n1", ai, edited)
    StyleRuleService.observe_edit(db, "p1", "n1", ai, edited)
    rule = db.query(StyleRuleModel).one()
    assert rule.occurrences == 1 and rule.status == "CANDIDATE"
    StyleRuleService.observe_edit(db, "p1", "n2", "Họ trình bày một cách rõ ràng.", "Họ trình bày rõ.")
    db.refresh(rule)
    assert rule.occurrences == 2 and rule.status == "ACTIVE"
    active = StyleRuleService.active_rules(db, "p1")
    assert StyleRuleService.prompt_lines(active) == ['- Tránh "một cách rõ ràng", dùng "rõ" (2 lần sửa).']
    hints = StyleRuleService.lint(active, "Bà ấy nói một cách rõ ràng.")
    assert [hint["rule"] for hint in hints] == ["PROJECT_STYLE_RULE"]


def test_observe_approved_node_compares_against_latest_ai_translation(db):
    node = NodeModel(
        id="n1", project_id="p1", content="He explained it clearly.", node_type="paragraph",
        status="APPROVED", order_index=0, version=1, translated_content="Anh ấy giải thích rõ.",
    )
    db.add_all([node, TranslationModel(id="t1", node_id="n1", project_id="p1", translated_text="Anh ấy giải thích một cách rõ ràng.")])
    db.commit()
    learned = StyleRuleService.observe_approved_node(db, "p1", node)
    assert [(rule.avoid, rule.prefer) for rule in learned] == [("một cách rõ ràng", "rõ")]


# ---------- Critic ----------

def _checks(value="PASS"):
    return {key: value for key in NATURALNESS_CHECKS}


def test_critic_pass_may_carry_info_notes_but_not_warnings():
    info = {"type": "REGISTER", "severity": "INFO", "target_span": "x", "message": "Có thể trang trọng hơn."}
    result = validate_naturalness_result({"status": "PASS", "score": 0.9, "issues": [info], "checks": _checks()})
    assert result.passed()
    with pytest.raises(ValueError):
        validate_naturalness_result({
            "status": "PASS", "score": 0.9, "checks": _checks(),
            "issues": [dict(info, severity="WARNING")],
        })


class _PairwiseProvider:
    def __init__(self, answers):
        self.answers = list(answers)

    def compare_naturalness(self, **kwargs):
        return {"preferred": self.answers.pop(0), "reason": "test"}


def test_pairwise_vetoes_only_when_original_wins_both_orders():
    # Lần 1 bản gốc ở vị trí A, lần 2 ở vị trí B.
    assert compare_original_and_rewrite(_PairwiseProvider(["A", "B"]), "s", "gốc", "mới").original_preferred is True
    assert compare_original_and_rewrite(_PairwiseProvider(["A", "A"]), "s", "gốc", "mới").original_preferred is False
    unsupported = compare_original_and_rewrite(object(), "s", "gốc", "mới")
    assert unsupported.original_preferred is None


# ---------- Editorial rewrite orchestration ----------

def _naturalness(status, score):
    if status == "PASS":
        return {"status": "PASS", "score": score, "issues": [], "checks": _checks()}
    checks = _checks()
    checks["pronoun_reference"] = "FAIL"
    return {
        "status": "FAIL", "score": score, "checks": checks,
        "issues": [{"type": "PRONOUN_REFERENCE", "severity": "WARNING", "target_span": "nó", "message": "Đại từ sai."}],
    }


class _RewriteProvider(MockProvider):
    def __init__(self, rewritten, pairwise=("B", "A")):
        self.rewritten = rewritten
        self.naturalness = [_naturalness("FAIL", 0.6), _naturalness("PASS", 0.95)]
        self.pairwise = list(pairwise)
        self.editorial_inputs = []
        self.critic_inputs = []

    def review_naturalness(self, **kwargs):
        self.critic_inputs.append(kwargs)
        return self.naturalness.pop(0) if self.naturalness else _naturalness("PASS", 0.95)

    def editorial_rewrite(self, **kwargs):
        self.editorial_inputs.append(kwargs)
        return self.rewritten

    def review_semantic_fidelity(self, *args, **kwargs):
        return {
            "status": "PASS", "score": 0.98, "errors": [],
            "checks": {key: "PASS" for key in (
                "completeness", "meaning", "polarity", "modality", "causality", "scope", "entity_reference",
            )},
        }

    def compare_naturalness(self, **kwargs):
        return {"preferred": self.pairwise.pop(0), "reason": "test"}


def _engine(db, provider, source, style_guide=None):
    node = NodeModel(
        id="rewrite-node", project_id="p1", content=source, node_type="paragraph",
        status="PENDING", order_index=1, version=1,
    )
    db.add(node)
    db.commit()
    config = SimpleNamespace(
        document_type="LITERATURE", translation_mode="NATURAL", model_name="mock",
        semantic_critic_model="mock", naturalness_critic_model="mock",
        semantic_risk_medium=0.35, semantic_risk_high=0.65, semantic_max_repairs=2,
        naturalness_pass_threshold=0.80, naturalness_rewrite_threshold=0.55, editorial_max_rewrites=1,
        quality_profile="HIGH_QUALITY", custom_instructions="", register="NEUTRAL",
        sentence_style="MODERATE", style_guide=style_guide or {},
    )
    engine = SimpleNamespace(
        db=db, project_id="p1", provider=provider, config=config, locked_glossary={},
        canonical_node=lambda item: DocumentNode(
            id=item.id, type=NodeType.PARAGRAPH, content=item.content,
            status=NodeStatus.PENDING, order_index=item.order_index,
        ),
    )
    previous = [SimpleNamespace(source="Alice met Bob.", translation="Alice gặp Bob.")]
    engine._rolling_context = lambda chapter, nodes, neighbors: previous
    return engine, node


def _assure(engine, node, text):
    return SemanticAssuranceService.assure_and_commit(
        engine, SimpleNamespace(id="ch", title="Ch"), node, text,
        SimpleNamespace(style_hash="s", glossary_hash="g", prompt_version="p"), "mock", "test",
    )


def test_rewrite_receives_previous_context_and_address_terms(db):
    provider = _RewriteProvider("Alice nhìn Bob, rồi em mỉm cười.")
    engine, node = _engine(db, provider, "Alice looked at Bob, then she smiled.", LITERATURE_GUIDE)

    result = _assure(engine, node, "Alice nhìn Bob, rồi nó mỉm cười.")

    assert result.approved is True
    sent = provider.editorial_inputs[0]
    assert sent["previous_context"] == [{"source": "Alice met Bob.", "translation": "Alice gặp Bob."}]
    assert sent["address_terms"]["pairs"][0]["self"] == "em"
    assert provider.critic_inputs[0]["address_terms"]["characters"] == {"Alice": "cô"}


def test_pairwise_guard_rejects_rewrite_that_is_worse_than_original(db):
    provider = _RewriteProvider("Alice nhìn Bob, rồi cô mỉm cười.", pairwise=("A", "B"))
    engine, node = _engine(db, provider, "Alice looked at Bob, then she smiled.", LITERATURE_GUIDE)

    result = _assure(engine, node, "Alice nhìn Bob, rồi nó mỉm cười.")

    assert result.approved is False
    assert result.publication_status == "NEEDS_REVIEW"
    assert any("So sánh cặp" in issue["message"] for issue in result.naturalness_issues)
    assert db.query(NodeModel).filter_by(id=node.id).one().translated_content is None


def test_pairwise_guard_can_be_disabled(db):
    provider = _RewriteProvider("Alice nhìn Bob, rồi cô mỉm cười.", pairwise=("A", "B"))
    guide = dict(LITERATURE_GUIDE, naturalness_pairwise_guard=False)
    engine, node = _engine(db, provider, "Alice looked at Bob, then she smiled.", guide)

    assert _assure(engine, node, "Alice nhìn Bob, rồi nó mỉm cười.").approved is True


# ---------- Eval metrics ----------

def test_lint_metrics_and_regression_check():
    baseline = summarize_evaluations([
        {"lint_warnings": 2, "word_count": 100, "naturalness_status": "FAIL"},
        {"lint_warnings": 0, "word_count": 100, "naturalness_status": "PASS"},
    ])
    assert baseline["lint_warnings_per_1000_words"] == 10.0
    assert baseline["lint_clean_rate"] == 0.5
    assert baseline["naturalness_first_pass_rate"] == 0.5
    worse = summarize_evaluations([{"lint_warnings": 4, "word_count": 200}])
    assert compare_regression(baseline, worse)["checks"]["style_lint_no_regression"]["passed"] is False
    assert compare_regression(baseline, summarize_evaluations([]))["checks"]["style_lint_no_regression"]["passed"] is True


# ---------- API ----------

def test_approving_edited_nodes_learns_rule_and_api_can_disable_it():
    from fastapi.testclient import TestClient

    from app.db.engine import get_project_db
    from app.main import app

    client = TestClient(app)
    project_id = "test_style_rules_api_proj"
    db = get_project_db(project_id)
    try:
        for model in (StyleRuleModel, TranslationModel, NodeModel):
            db.query(model).filter(model.project_id == project_id).delete()
        db.query(ProjectModel).filter(ProjectModel.id == project_id).delete()
        db.commit()
        db.add(ProjectModel(id=project_id, title="Style rules", selected_model="mock-qwen2.5:7b"))
        db.commit()
        for index, (ai, edited) in enumerate((
            ("Cô giải thích một cách rõ ràng.", "Cô giải thích rõ."),
            ("Họ trình bày một cách rõ ràng.", "Họ trình bày rõ."),
        )):
            node_id = f"style_rule_node_{index}"
            db.add(NodeModel(
                id=node_id, project_id=project_id, node_type="paragraph", content="She explained it clearly.",
                translated_content=ai, order_index=index, status="TRANSLATED", version=1,
            ))
            db.add(TranslationModel(id=f"style_rule_tr_{index}", node_id=node_id, project_id=project_id, translated_text=ai))
            db.commit()
            response = client.patch(
                f"/api/projects/{project_id}/structure/nodes/{node_id}",
                json={"translated_content": edited, "approval_status": "APPROVED"},
            )
            assert response.status_code == 200, response.text
            assert response.json()["style_rules"]["observed"] == 1

        rules = client.get(f"/api/projects/{project_id}/style/rules").json()
        assert [(rule["avoid"], rule["prefer"], rule["status"]) for rule in rules] == [("một cách rõ ràng", "rõ", "ACTIVE")]

        disabled = client.patch(f"/api/projects/{project_id}/style/rules/{rules[0]['id']}", json={"status": "DISABLED"})
        assert disabled.json()["status"] == "DISABLED"
        assert client.patch(f"/api/projects/{project_id}/style/rules/{rules[0]['id']}", json={"status": "BOGUS"}).status_code == 422
    finally:
        db.rollback()
        for model in (StyleRuleModel, TranslationModel, NodeModel):
            db.query(model).filter(model.project_id == project_id).delete()
        db.query(ProjectModel).filter(ProjectModel.id == project_id).delete()
        db.commit()
        db.close()

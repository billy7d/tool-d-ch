"""P1/P2 sau đợt kiểm tra: preset thuật ngữ, luật lint mới, dịch từng câu khi lọt chữ Hán."""
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.db.models import Base, ChapterModel, GlossaryModel, NodeModel, ProjectModel
from app.services.qa.vietnamese_style_lint import VietnameseStyleLint
from app.services.translation.contextual_engine import ContextualTranslationEngine
from app.services.translation.glossary_presets import preset_entries, presets_for
from app.services.translation.mock_provider import MockProvider
from app.services.translation.prompt_builder import PromptBuilder
from app.services.translation.translation_config import TranslationConfig


def _engine(tmp_path, monkeypatch, provider=None, document_type="LEGAL", style_guide=None, glossary=None, locked=None, content="The licensee may terminate."):
    monkeypatch.setattr(settings, "PROJECTS_DIR", tmp_path)
    db_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(db_engine)
    db = sessionmaker(bind=db_engine)()
    project = ProjectModel(id="g", title="G", selected_model="mock", document_type=document_type, style_guide=style_guide or {})
    chapter = ChapterModel(id="g_ch", project_id="g", title="C", order_index=0)
    node = NodeModel(id="g_n", project_id="g", chapter_id="g_ch", node_type="paragraph", content=content, status="PENDING", order_index=0)
    db.add_all([project, chapter, node] + list(glossary or []))
    db.commit()
    engine = ContextualTranslationEngine(db, project, provider or MockProvider(), TranslationConfig.from_project(project, model_override="mock"), locked)
    return engine, chapter, engine.canonical_node(node)


# ---------- Preset thuật ngữ ----------

def test_presets_default_to_document_domain_and_can_be_disabled():
    assert presets_for({}, "LEGAL") == ["LEGAL"]
    assert presets_for({}, "LITERATURE") == []
    assert presets_for({"glossary_presets": []}, "LEGAL") == []
    assert presets_for({"glossary_presets": ["technical", "bogus"]}, "LEGAL") == ["TECHNICAL"]


def test_presets_are_soft_and_reach_the_prompt_only_when_term_is_present(tmp_path, monkeypatch):
    engine, chapter, node = _engine(tmp_path, monkeypatch)
    assert engine.soft_glossary["licensee"] == "bên được cấp phép"
    assert "licensee" not in engine.locked_glossary
    context = engine.assemble_context(chapter, [node], "single")
    rendered = "\n\n".join(PromptBuilder._context_parts(context))
    assert '"licensee" -> "bên được cấp phép"' in rendered
    assert "indemnify" not in rendered  # không có trong đoạn nguồn thì không đưa vào prompt


def test_presets_never_drop_legacy_hard_terms_or_override_project_terms(tmp_path, monkeypatch):
    project_term = GlossaryModel(id="t1", project_id="g", source_term="licensee", target_term="người được cấp phép", locked=True)
    engine, _, _ = _engine(tmp_path, monkeypatch, glossary=[project_term])
    assert engine.locked_glossary["licensee"] == "người được cấp phép"
    assert "licensee" not in engine.soft_glossary

    engine, _, _ = _engine(tmp_path / "b", monkeypatch, locked={"Agreement": "Thỏa thuận"})
    # Hard term truyền riêng (legacy) vẫn còn dù preset đã thêm entry mềm.
    assert engine.locked_glossary == {"Agreement": "Thỏa thuận"}
    assert engine.glossary_validation_terms == {"Agreement": "Thỏa thuận"}


def test_preset_entries_skip_terms_the_project_already_has():
    entries = preset_entries(["LEGAL"], existing_sources=["Licensee"])
    assert all(entry["source_term"] != "licensee" for entry in entries)
    assert all(entry["lock_level"] == "SOFT" for entry in entries)


# ---------- Luật lint mới ----------

@pytest.mark.parametrize("text,rule", [
    ("Công ty đã đưa ra một quyết định để đi vào thị trường.", "DUA_RA_MOT_QUYET_DINH"),
    ("Khoản đầu tư có thể tạo ra một lợi nhuận ổn định.", "LIGHT_VERB_ARTICLE"),
    ("Quyết định được đưa ra sau một sự xem xét cẩn thận.", "CONNECTOR_SU_NOUN"),
    ("Tỷ lệ vẫn còn ở tại 0,8.", "CALQUE_PHRASE_2"),
    ("Hàm nên trả về lỗi nếu đầu vào là không hợp lệ.", "LA_KHONG_ADJECTIVE"),
    ("Cô ấy đã nói rằng cô ấy sẽ trở lại.", "REPEATED_SUBJECT_RANG"),
])
def test_new_lint_rules(text, rule):
    assert rule in {item.rule for item in VietnameseStyleLint.warnings(text)}


@pytest.mark.parametrize("text", [
    "Bạn có thể tạo ra một số thay đổi nhỏ.",
    "Khi sự cố xảy ra, hãy liên hệ bộ phận hỗ trợ.",
    "Cô ấy nói rằng anh ấy sẽ đến.",
    "Hệ thống cung cấp một phần dữ liệu.",
])
def test_new_lint_rules_leave_natural_text_alone(text):
    assert VietnameseStyleLint.warnings(text) == []


# ---------- Dịch từng câu khi cả đoạn vẫn lọt chữ Hán ----------

class _SentenceProvider(MockProvider):
    def translate(self, blocks, system_prompt, user_prompt, model=None, temperature=0.2):
        return [{"node_id": block["id"], "text": "Ngôi nhà nhỏ hơn钢琴键似乎无法正常显示"} for block in blocks]

    def translate_single(self, text, system_prompt, glossary_terms=None, model=None, temperature=0.15, user_prompt=None):
        if system_prompt == ContextualTranslationEngine.FOREIGN_SCRIPT_SYSTEM_PROMPT:
            if text.startswith("The house"):
                return "Ngôi nhà nhỏ hơn anh nhớ, nhưng ấm áp hơn."
            if text.startswith("Nobody"):
                return "Không ai chạm vào cây đàn piano kể từ khi mẹ anh mất năm 2019."
        return "Ngôi nhà nhỏ hơn钢琴键似乎无法正常显示"


def test_sentence_by_sentence_recovery_when_whole_paragraph_keeps_leaking(tmp_path, monkeypatch):
    source = "The house was smaller than he remembered, but warmer. Nobody had touched the piano since his mother died in 2019."
    engine, chapter, node = _engine(tmp_path, monkeypatch, provider=_SentenceProvider(), document_type="LITERATURE", content=source)
    result = engine.translate_batch(chapter, [node]).results[0]
    assert result.passed
    assert result.translated_text == "Ngôi nhà nhỏ hơn anh nhớ, nhưng ấm áp hơn. Không ai chạm vào cây đàn piano kể từ khi mẹ anh mất năm 2019."


# ---------- Sau lần chạy thật thứ hai ----------

def test_first_dialogue_pronoun_must_match_role_even_when_counts_differ():
    from app.services.translation.address_terms import AddressTermLedger

    ledger = AddressTermLedger.from_style_guide({"address_terms": {"pairs": [
        {"speaker": "Bob", "listener": "Alice", "self": "anh", "other": "em"}]}}, "LITERATURE")
    source = '"I will stay," Bob told Alice. "You are the reason I came home."'
    assert ledger.check_dialogue(source, "“Em sẽ ở lại,” Bob nói với Alice. “Anh về vì em là lý do anh trở về nhà.”")
    assert not ledger.check_dialogue(source, "“Anh sẽ ở lại,” Bob nói với Alice. “Em là lý do anh trở về, em biết không.”")


def test_headings_are_not_forced_to_semantic_critic(tmp_path, monkeypatch):
    from app.services.translation.semantic_risk import SemanticRiskScorer

    # Critic 7B chấm sai heading 1–3 từ, nên heading không bị ép qua critic.
    assert not SemanticRiskScorer.score("The House", "Nhà", node_type="heading").requires_critic


def test_heading_reusing_another_headings_translation_is_rejected(tmp_path, monkeypatch):
    from app.models.canonical import NodeType

    class _ReusedHeading(MockProvider):
        def translate(self, blocks, system_prompt, user_prompt, model=None, temperature=0.2):
            return [{"node_id": block["id"], "text": "Trở Lại"} for block in blocks]

        def translate_single(self, text, system_prompt, glossary_terms=None, model=None, temperature=0.15, user_prompt=None):
            return "Trở Lại"

    engine, chapter, node = _engine(tmp_path, monkeypatch, provider=_ReusedHeading(), document_type="LITERATURE", content="The House")
    engine.db.add(NodeModel(id="g_title", project_id="g", chapter_id="g_ch", node_type="heading", content="Chapter 1: The Return",
                            translated_content="Chương 1: Trở Lại", status="TRANSLATED", order_index=-1))
    engine.db.commit()
    result = engine.translate_batch(chapter, [node.model_copy(update={"type": NodeType.HEADING})]).results[0]
    assert not result.passed
    assert "HEADING_DUPLICATE" in {issue["code"] for issue in result.quality.issues}


def test_missed_soft_term_becomes_lint_warning_but_not_on_headings(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from app.services.translation.vietnamese_editorial_assurance import TranslationPublicationAssuranceService as S

    engine, _, _ = _engine(tmp_path, monkeypatch, document_type="TECHNICAL",
                           content="Run npm install before executing the build command.")
    source = "Run npm install before executing the build command."
    hints = S._soft_term_hints(engine, source, "Chạy npm install trước khi thực hiện lệnh xây dựng.")
    assert [hint["target_span"] for hint in hints] == ["build command"]
    assert S._soft_term_hints(engine, source, "Chạy npm install trước khi chạy lệnh build.") == []
    ledger = engine.address_terms_for(source)
    heading = SimpleNamespace(node_type="heading", content=source)
    assert all(hint["rule"] != "SOFT_TERM_MISSED" for hint in S._hints_for_node(engine, ledger, heading, "Lệnh xây dựng"))


def test_single_line_heading_keeps_only_first_line_of_batch_output(tmp_path, monkeypatch):
    class _HeadingProvider(MockProvider):
        def translate(self, blocks, system_prompt, user_prompt, model=None, temperature=0.2):
            return [{"node_id": block["id"], "text": "Nhà\n\n“Anh về rồi,” Alice nói với Bob."} for block in blocks]

    engine, chapter, node = _engine(tmp_path, monkeypatch, provider=_HeadingProvider(), document_type="LITERATURE", content="The House")
    from app.models.canonical import NodeType

    node = node.model_copy(update={"type": NodeType.HEADING})
    result = engine.translate_batch(chapter, [node]).results[0]
    assert result.passed and result.translated_text == "Nhà"


def test_rejection_reasons_include_gate_codes_and_skip_internal_repair_policy():
    from types import SimpleNamespace

    from app.services.translation.semantic_assurance import SemanticAssuranceResult, SemanticAssuranceService

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(ProjectModel(id="r", title="R"))
    db.add(NodeModel(id="r_n", project_id="r", content="The House", node_type="heading", status="PENDING", order_index=0, version=1))
    db.commit()
    node = db.query(NodeModel).one()
    rejected = SemanticAssuranceResult(
        "Nhà Bob", False, "FAIL", 0.5, "MEDIUM",
        [{"code": "SEMANTIC_REPAIR_POLICY", "message": "chỉ dẫn"}, {"code": "CONTEXT_LEAK", "severity": "ERROR", "message": "chép"}],
        "sig",
    )
    SemanticAssuranceService.persist_rejected(db, "r", node, rejected, "mock")
    db.refresh(node)
    assert node.node_metadata["draft_translation"]["reasons"] == ["CONTEXT_LEAK"]
    from app.db.models import QAIssueModel
    assert {issue.issue_type for issue in db.query(QAIssueModel).all()} == {"CONTEXT_LEAK"}
    db.close()

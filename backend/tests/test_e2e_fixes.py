"""Các lỗi tìm ra khi chạy end-to-end thật (API -> worker -> QA -> export) với qwen2.5:7b."""
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.db.engine import get_project_db, init_global_db
from app.db.models import Base, ChapterModel, NodeModel, ProjectModel
from app.db.repository import ProjectRepository
from app.main import app
from app.services.qa.vietnamese_naturalness_critic import NATURALNESS_CHECKS, naturalness_cache_key
from app.services.translation.address_terms import AddressTermLedger
from app.services.translation.contextual_engine import ContextualTranslationEngine
from app.services.translation.mock_provider import MockProvider
from app.services.translation.quality_gate import TranslationQualityGate
from app.services.translation.translation_commit import TranslationCommitService
from app.services.translation.translation_config import TranslationConfig
from app.services.translation.vietnamese_post_processor import VietnamesePostProcessor

client = TestClient(app)
LOVERS = {"address_terms": {"pairs": [
    {"speaker": "Alice", "listener": "Bob", "self": "em", "other": "anh"},
    {"speaker": "Bob", "listener": "Alice", "self": "anh", "other": "em"},
]}}


# ---------- Stage của project phải hiện đúng ở UI ----------

def test_stage_written_in_project_db_is_visible_in_project_api():
    init_global_db()
    project_id = client.post("/api/projects", json={"title": "Stage sync", "selected_model": "mock"}).json()["id"]
    try:
        db = get_project_db(project_id)
        ProjectRepository(db).update_project(project_id, current_stage="STRUCTURE_CONFIRMED", structure_confirmed=True)
        db.close()
        project = client.get(f"/api/projects/{project_id}").json()
        assert project["current_stage"] == "STRUCTURE_CONFIRMED"
        assert project["structure_confirmed"] is True
    finally:
        client.delete(f"/api/projects/{project_id}")


# ---------- Commit chấp nhận ghi chú INFO và lưu kết luận để QA dùng lại ----------

@pytest.fixture()
def memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(ProjectModel(id="p", title="P"))
    db.add(NodeModel(id="n", project_id="p", content="The door is open.", node_type="paragraph",
                     status="PENDING", order_index=0, version=1))
    db.commit()
    yield db
    db.close()
    engine.dispose()


def test_commit_accepts_info_notes_and_stores_naturalness_cache(memory_db):
    node = memory_db.query(NodeModel).one()
    TranslationCommitService.commit_validated_translation(
        memory_db, "p", node, "Cánh cửa đang mở.",
        SimpleNamespace(style_hash="s", glossary_hash="g", prompt_version="v"), {}, "mock", "test",
        naturalness_result={
            "status": "PASS", "score": 0.9, "pass_threshold": 0.8, "critic_model": "mock",
            "checks": {key: "PASS" for key in NATURALNESS_CHECKS},
            "issues": [{"type": "REGISTER", "severity": "INFO", "message": "Có thể trang trọng hơn."}],
        },
    )
    memory_db.refresh(node)
    cached = node.node_metadata["naturalness"]
    assert node.translated_content == "Cánh cửa đang mở."
    assert cached["status"] == "PASS"
    assert cached["cache_key"] == naturalness_cache_key("The door is open.", "Cánh cửa đang mở.", "mock")


def test_qa_reuses_cached_naturalness_and_rechecks_edited_text():
    init_global_db()
    project_id = client.post("/api/projects", json={"title": "QA cache", "selected_model": "mock"}).json()["id"]
    try:
        db = get_project_db(project_id)
        db.add(ChapterModel(id=f"{project_id}_ch", project_id=project_id, title="C", order_index=0))
        for index, (source, target, cached_target) in enumerate((
            ("The door is open.", "Cánh cửa đang mở.", "Cánh cửa đang mở."),
            ("The room is quiet.", "Căn phòng yên tĩnh.", "Căn phòng im lặng."),  # người dùng đã sửa sau khi duyệt
        )):
            db.add(NodeModel(
                id=f"{project_id}_n{index}", project_id=project_id, chapter_id=f"{project_id}_ch",
                content=source, translated_content=target, node_type="paragraph", status="TRANSLATED",
                order_index=index, version=1,
                node_metadata={"naturalness": {"status": "PASS", "cache_key": naturalness_cache_key(source, cached_target, "mock")}},
            ))
        db.commit()
        db.close()
        result = client.post(f"/api/projects/{project_id}/qa/run").json()
        assert result["naturalness_cached"] == 1
        assert result["naturalness_reviewed"] == 1
        refreshed = client.post(f"/api/projects/{project_id}/qa/run?refresh_naturalness=true").json()
        assert refreshed["naturalness_cached"] == 0 and refreshed["naturalness_reviewed"] == 2
    finally:
        client.delete(f"/api/projects/{project_id}")


# ---------- Gate: token kỹ thuật ----------

def _codes(source, target):
    return {issue["code"] for issue in TranslationQualityGate().validate(source, target, {}).issues}


def test_gate_requires_paths_and_calls_to_survive_unchanged():
    source = r"Run npm install before executing the build command in C:\workspace\app."
    assert "TECHNICAL_TOKEN_MISMATCH" in _codes(source, r"Trước khi build trong \workspace\app, hãy chạy npm install.")
    assert "TECHNICAL_TOKEN_MISMATCH" not in _codes(source, r"Chạy npm install trước khi build trong C:\workspace\app.")
    source = "Do not rename UserService.getById(); see /etc/nginx/nginx.conf."
    assert "TECHNICAL_TOKEN_MISMATCH" in _codes(source, "Không đổi tên UserService.getByID(); xem /etc/nginx/nginx.conf.")


# ---------- Ngoặc kép lời thoại ----------

def test_dialogue_quotes_become_curly_without_inner_spaces():
    text = '"Anh về rồi, " Alice nói với Bob. "Em đợi anh suốt mùa đông."'
    assert VietnamesePostProcessor.normalize_safely(text) == "“Anh về rồi,” Alice nói với Bob. “Em đợi anh suốt mùa đông.”"
    assert VietnamesePostProcessor.normalize_safely('Gọi print("x") và key="v".') == 'Gọi print("x") và key="v".'
    assert VietnamesePostProcessor.normalize_safely('"Một ngoặc lẻ') == '"Một ngoặc lẻ'


# ---------- Xưng hô theo vai người nói ----------

def test_dialogue_check_flags_reversed_roles_and_generic_pronouns():
    ledger = AddressTermLedger.from_style_guide(LOVERS, "LITERATURE")
    source = '"I will stay," Bob told Alice. "You are the reason I came home."'
    assert ledger.check_dialogue(source, "“Em sẽ ở lại,” Bob nói với Alice. “Anh là lý do em về nhà.”")
    assert not ledger.check_dialogue(source, "“Anh sẽ ở lại,” Bob nói với Alice. “Em là lý do anh về nhà.”")
    promise = 'Bob looked at Alice and said, "I promised you I would return, didn\'t I?"'
    assert ledger.check_dialogue(promise, "Bob nhìn Alice và nói, “Tôi đã hứa sẽ trở lại, phải không?”")
    # Không có câu dẫn rõ người nói thì không đoán.
    assert not ledger.check_dialogue('"I will stay."', "“Em sẽ ở lại.”")


# ---------- Nhánh batch của worker ----------

class _BatchChineseProvider(MockProvider):
    def translate(self, blocks, system_prompt, user_prompt, model=None, temperature=0.2):
        return [{"node_id": block["id"], "text": "Nhà cung cấp不得披露机密信息。"} for block in blocks]

    def translate_single(self, text, system_prompt, glossary_terms=None, model=None, temperature=0.15, user_prompt=None):
        if system_prompt == ContextualTranslationEngine.FOREIGN_SCRIPT_SYSTEM_PROMPT:
            return "Nhà cung cấp không được tiết lộ thông tin mật."
        return "Nhà cung cấp不得披露机密信息。"


def test_batch_path_also_recovers_chinese_output():
    original = settings.PROJECTS_DIR
    with tempfile.TemporaryDirectory(prefix="batch-cjk-") as temp_dir:
        settings.PROJECTS_DIR = Path(temp_dir)
        engine_db = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine_db)
        db = sessionmaker(bind=engine_db)()
        try:
            project = ProjectModel(id="b", title="B", selected_model="mock", document_type="LEGAL")
            chapter = ChapterModel(id="b_ch", project_id="b", title="Điều khoản", order_index=0)
            node = NodeModel(id="b_n", project_id="b", chapter_id="b_ch", node_type="paragraph",
                             content="The supplier must not disclose confidential information.", status="PENDING", order_index=0)
            db.add_all([project, chapter, node])
            db.commit()
            engine = ContextualTranslationEngine(db, project, _BatchChineseProvider(), TranslationConfig.from_project(project, model_override="mock"), {})
            result = engine.translate_batch(chapter, [engine.canonical_node(node)]).results[0]
            assert result.passed and result.translated_text == "Nhà cung cấp không được tiết lộ thông tin mật."
        finally:
            db.close()
            engine_db.dispose()
            settings.PROJECTS_DIR = original


# ---------- Lint-first naturalness và heading ----------

def test_lint_review_needs_no_model_call():
    from app.services.qa.vietnamese_naturalness_critic import VietnameseNaturalnessCritic

    literal = VietnameseNaturalnessCritic.lint_review("The proposal was approved by the board.", "Đề xuất đã được phê duyệt bởi hội đồng.")
    natural = VietnameseNaturalnessCritic.lint_review("The proposal was approved by the board.", "Hội đồng đã phê duyệt đề xuất.")
    assert literal.verdict_source == natural.verdict_source == "LINT_ONLY"
    assert literal.critic_calls == natural.critic_calls == 0
    assert literal.rewrite_allowed() and natural.passed()


def test_llm_critic_only_runs_in_publishing_tier_or_always_mode():
    from app.services.translation.vietnamese_editorial_assurance import TranslationPublicationAssuranceService as S

    paragraph = SimpleNamespace(node_type="paragraph")
    heading = SimpleNamespace(node_type="heading")

    def engine(tier, mode=None):
        guide = {"naturalness_critic_mode": mode} if mode else {}
        return SimpleNamespace(config=SimpleNamespace(quality_tier=tier, style_guide=guide))

    assert S._use_llm_critic(engine("HIGH_QUALITY"), paragraph) is False
    assert S._use_llm_critic(engine("PUBLISHING"), paragraph) is True
    assert S._use_llm_critic(engine("HIGH_QUALITY", "ALWAYS"), paragraph) is True
    assert S._use_llm_critic(engine("PUBLISHING", "LINT_ONLY"), paragraph) is False
    # Heading không qua critic LLM và không bị rewrite (từng thành "Nhà Bob").
    assert S._use_llm_critic(engine("PUBLISHING", "ALWAYS"), heading) is False
    assert S._rewrite_permitted(engine("HIGH_QUALITY"), SimpleNamespace(issues=[]), heading) is False


# ---------- Gate: từ tiếng Anh sót và nhãn chưa dịch ----------

def test_gate_rejects_leftover_english_words_and_labels():
    source = "Nobody had touched the piano since his mother died in 2019."
    assert "ENGLISH_RESIDUE" in _codes(source, "Từ khi mẹ anh qua đời năm 2019, nobody chạm vào piano nữa.")
    assert "ENGLISH_RESIDUE" not in _codes(source, "Từ khi mẹ anh qua đời năm 2019, không ai chạm vào cây đàn piano.")
    assert "UNTRANSLATED_LABEL" in _codes("Chapter 1: Operations Review", "Chapter 1: Kiểm tra hoạt động")
    assert "UNTRANSLATED_LABEL" not in _codes("Chapter 1: Operations Review", "Chương 1: Rà soát hoạt động")
    # Code inline và tên riêng viết hoa được giữ nguyên hợp lệ.
    assert "ENGLISH_RESIDUE" not in _codes("Use `if x is None`.", "Dùng `if x is None`.")


# ---------- Sửa deterministic trước khi chạy gate ----------

def test_positional_dialogue_fix_and_third_person_guard():
    ledger = AddressTermLedger.from_style_guide(LOVERS, "LITERATURE")
    source = '"I will stay," Bob told Alice. "You are the reason I came home."'
    fixed = ledger.fix_dialogue(source, "“Em sẽ ở lại,” Bob nói với Alice. “Anh là lý do anh trở về nhà.”")
    assert fixed == "“Anh sẽ ở lại,” Bob nói với Alice. “Em là lý do anh trở về nhà.”"
    assert ledger.fix_dialogue(source, "“Em sẽ ở lại,” Bob nói với Alice. “Anh ấy là lý do em về.”") is None


class _ScriptedSingle(MockProvider):
    def __init__(self, answers):
        self.answers = list(answers)

    def translate(self, blocks, system_prompt, user_prompt, model=None, temperature=0.2):
        return [{"node_id": block["id"], "text": self.answers.pop(0)} for block in blocks]

    def translate_single(self, text, system_prompt, glossary_terms=None, model=None, temperature=0.15, user_prompt=None):
        return self.answers.pop(0) if self.answers else "Nhà nhỏ hơn anh nhớ."


def _engine_for(provider, content, node_type="paragraph", document_type="LITERATURE", style_guide=None):
    db_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(db_engine)
    db = sessionmaker(bind=db_engine)()
    project = ProjectModel(id="f", title="F", selected_model="mock", document_type=document_type, style_guide=style_guide or {})
    chapter = ChapterModel(id="f_ch", project_id="f", title="Chapter 1: Operations Review", order_index=0)
    node = NodeModel(id="f_n", project_id="f", chapter_id="f_ch", node_type=node_type, content=content, status="PENDING", order_index=0)
    db.add_all([project, chapter, node])
    db.commit()
    engine = ContextualTranslationEngine(db, project, provider, TranslationConfig.from_project(project, model_override="mock"), {})
    return engine, chapter, engine.canonical_node(node)


def test_heading_does_not_inherit_chapter_label_from_context(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PROJECTS_DIR", tmp_path)
    engine, chapter, node = _engine_for(_ScriptedSingle(["Chương 1: Điều khoản pháp lý"]), "Legal Terms", "heading", "BUSINESS")
    result = engine.translate_batch(chapter, [node]).results[0]
    assert result.passed and result.translated_text == "Điều khoản pháp lý"


def test_reversed_roles_are_fixed_without_extra_model_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PROJECTS_DIR", tmp_path)
    provider = _ScriptedSingle(['"Em sẽ ở lại," Bob nói với Alice. "Anh là lý do anh trở về nhà."'])
    engine, chapter, node = _engine_for(provider, '"I will stay," Bob told Alice. "You are the reason I came home."', style_guide=LOVERS)
    result = engine.translate_batch(chapter, [node]).results[0]
    assert result.passed
    assert result.translated_text == "“Anh sẽ ở lại,” Bob nói với Alice. “Em là lý do anh trở về nhà.”"


def test_english_residue_uses_short_vietnamese_recovery_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PROJECTS_DIR", tmp_path)
    leftover = "Từ khi mẹ anh qua đời năm 2019, nobody chạm vào piano nữa."
    provider = _ScriptedSingle([leftover, leftover, leftover, "Từ khi mẹ anh qua đời năm 2019, không ai chạm vào cây đàn piano nữa."])
    engine, chapter, node = _engine_for(provider, "Nobody had touched the piano since his mother died in 2019.")
    result = engine.translate_batch(chapter, [node]).results[0]
    assert result.passed and "không ai" in result.translated_text


# ---------- Lượng từ "either" và calque "Ai cũng không" ----------

def test_either_clause_is_risky_and_both_shift_is_rejected():
    from app.services.translation.semantic_risk import SemanticRiskScorer
    from app.services.qa.vietnamese_style_lint import VietnameseStyleLint

    source = "Nothing in this clause shall be construed as a waiver of either party's rights."
    assert SemanticRiskScorer.score(source, "x", document_type="BUSINESS").requires_critic
    assert "QUANTIFIER_SHIFT" in _codes(source, "Không có gì trong điều khoản này được hiểu là từ bỏ quyền của cả hai bên.")
    assert "QUANTIFIER_SHIFT" not in _codes(source, "Không nội dung nào trong điều khoản này được hiểu là từ bỏ quyền của bất kỳ bên nào.")
    assert "QUANTIFIER_SHIFT" not in _codes("Both parties and either signatory agree.", "Cả hai bên và bất kỳ bên ký nào đều đồng ý.")
    assert [item.rule for item in VietnameseStyleLint.lint("Ai cũng không đụng đến cây đàn.")] == ["AI_CUNG_KHONG"]

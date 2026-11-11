"""Những gì giữ đoạn lại chờ người duyệt: chỉ lỗi thật mới chặn, gợi ý văn phong thì ghi lại làm cảnh báo."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.db.engine as db_engine_module
import app.services.translation.contextual_engine as contextual_engine_module
import app.services.translation.worker as worker_module
from app.db.models import Base, ChapterModel, NodeModel, ProjectModel, QAIssueModel
from app.services.qa.vietnamese_naturalness_critic import NATURALNESS_CHECKS, VietnameseNaturalnessResult
from app.services.translation.contextual_engine import ContextualTranslationEngine
from app.services.translation.mock_provider import MockProvider
from app.services.translation.numeric_validator import NumericValidator, spelled_english_numbers
from app.services.translation.translation_config import TranslationConfig
from app.services.translation.vietnamese_editorial_assurance import TranslationPublicationAssuranceService as Assurance
from app.services.translation.worker import TranslationWorker

SOURCE = "The proposal was approved by the board."
LITERAL = "Đề xuất đã được phê duyệt bởi hội đồng."


class Provider(MockProvider):
    def __init__(self, critic_status="PASS"):
        self.critic_status = critic_status
        self.critic_calls = 0
        self.editorial_calls = 0

    def translate_single(self, text, system_prompt, glossary_terms=None, model=None, temperature=0.15, user_prompt=None):
        return LITERAL

    def review_semantic_fidelity(self, source_text, translated_text, *args, **kwargs):
        self.critic_calls += 1
        marked = translated_text + (" [[OMISSION]]" if self.critic_status == "FAIL" else "")
        return super().review_semantic_fidelity(source_text, marked, *args, **kwargs)

    def editorial_rewrite(self, **kwargs):
        self.editorial_calls += 1
        return "Hội đồng đã phê duyệt đề xuất."


def _translate(monkeypatch, tmp_path, provider, style_guide=None):
    sqlite_engine = create_engine(f"sqlite:///{(tmp_path / 'project.db').as_posix()}")
    Base.metadata.create_all(sqlite_engine)
    new_session = sessionmaker(bind=sqlite_engine)
    db = new_session()
    db.add(ProjectModel(id="p", title="P", selected_model="mock", document_type="SELF_HELP", style_guide=style_guide or {}))
    db.add(ChapterModel(id="c", project_id="p", title="Chương", order_index=0))
    db.add(NodeModel(id="n", project_id="p", chapter_id="c", node_type="paragraph", content=SOURCE, status="PENDING", order_index=0))
    db.commit()
    db.close()
    monkeypatch.setattr(worker_module, "get_project_db", lambda requested_id: new_session())
    monkeypatch.setattr(db_engine_module, "get_project_db", lambda requested_id: new_session())
    monkeypatch.setattr(contextual_engine_module.settings, "PROJECTS_DIR", tmp_path / "projects")
    TranslationWorker().translate_project_sync("p", model_name="mock", provider=provider)
    return new_session()


def test_style_warning_publishes_the_text_unchanged_and_keeps_an_open_warning(monkeypatch, tmp_path):
    provider = Provider()
    db = _translate(monkeypatch, tmp_path, provider)
    node = db.query(NodeModel).one()
    assert node.status == "TRANSLATED" and node.translated_content == LITERAL
    assert provider.editorial_calls == 0
    issue = db.query(QAIssueModel).filter_by(node_id="n", status="OPEN").one()
    assert issue.severity == "WARNING" and "lint:PASSIVE_BY_AGENT" in issue.message
    assert node.node_metadata["naturalness"]["verdict_source"] == "LINT_ADVISORY"


def test_style_flagged_paragraph_is_still_read_by_the_semantic_critic(monkeypatch, tmp_path):
    provider = Provider(critic_status="FAIL")
    db = _translate(monkeypatch, tmp_path, provider)
    node = db.query(NodeModel).one()
    assert provider.critic_calls >= 1
    assert node.status == "NEEDS_REVIEW" and not node.translated_content
    assert node.node_metadata["draft_translation"]["stage"] == "SEMANTIC"


def _lint_result(rule, source="LINT_ONLY", status="FAIL"):
    hint = {"rule": rule, "type": "REDUNDANCY", "severity": "WARNING", "target_span": "x", "message": "m", "suggestion": ""}
    return VietnameseNaturalnessResult(status, 0.67, [], {key: "PASS" for key in NATURALNESS_CHECKS}, None, 0, 0.0, 0, "", [hint], source)


def test_project_specific_hints_still_go_through_the_rewrite_path():
    for rule in ("ADDRESS_READER_MISMATCH", "PROJECT_STYLE_RULE", "SOFT_TERM_MISSED"):
        assert not Assurance._style_notes_only(_lint_result(rule))
    assert Assurance._style_notes_only(_lint_result("MOT_CACH_ADVERB"))


def test_llm_critic_verdict_is_never_downgraded_to_a_note():
    for source in ("CRITIC", "LINT_OVERRIDE", "LINT_FALLBACK"):
        assert not Assurance._style_notes_only(_lint_result("MOT_CACH_ADVERB", source=source))
    assert not Assurance._style_notes_only(_lint_result("MOT_CACH_ADVERB", status="PASS"))


def test_spelled_out_source_number_may_be_written_in_digits():
    for source, target in (
        ("When I was nine years old.", "Khi tôi 9 tuổi."),
        ("Ninety minutes later she cried.", "90 phút sau, cô ấy khóc."),
        ("A fifty-two-year-old man.", "Một người đàn ông 52 tuổi."),
        ("Two hundred people came.", "200 người đã đến."),
        ("About fifty percent of us.", "Khoảng 50% chúng ta."),
    ):
        assert NumericValidator.validate(source, target).passed, source


def test_digits_that_do_not_match_the_spelled_number_are_still_additions():
    for source, target in (
        ("When I was nine years old.", "Khi tôi 8 tuổi."),
        ("Ten people came.", "10 người đến vào năm 1990."),
        ("About fifty people.", "Khoảng 50% mọi người."),
        ("She came.", "Cô ấy đến lúc 5 giờ."),
        ("It was ten years ago, in my junior year.", "Đó là cách đây mười năm, khi còn học lớp 10."),
        ("Thirty-two people came.", "Ba mươi hai người đến, trong đó có 32 người lớn."),
    ):
        assert NumericValidator.validate(source, target).unexpected, source


def test_article_a_is_not_a_number():
    assert not spelled_english_numbers("A man in a car.")
    assert spelled_english_numbers("a hundred times")["100"] == 1


class ChineseFragments(MockProvider):
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def translate_single(self, text, system_prompt, glossary_terms=None, model=None, temperature=0.15, user_prompt=None):
        self.calls.append(text)
        return self.reply


def _engine(tmp_path, monkeypatch, provider):
    sqlite_engine = create_engine(f"sqlite:///{(tmp_path / 'fragments.db').as_posix()}")
    Base.metadata.create_all(sqlite_engine)
    db = sessionmaker(bind=sqlite_engine)()
    project = ProjectModel(id="p", title="P", selected_model="mock", document_type="SELF_HELP")
    db.add(project)
    db.add(ChapterModel(id="c", project_id="p", title="Chương", order_index=0))
    node = NodeModel(id="n", project_id="p", chapter_id="c", node_type="paragraph",
                     content="I rode my bike and I rode it again today, happily.", status="PENDING", order_index=0)
    db.add(node)
    db.commit()
    monkeypatch.setattr(contextual_engine_module.settings, "PROJECTS_DIR", tmp_path / "projects")
    engine = ContextualTranslationEngine(db, project, provider, TranslationConfig.from_project(project, model_override="mock"), {})
    return engine, engine.canonical_node(node)


def test_repeated_chinese_fragment_is_translated_once(tmp_path, monkeypatch):
    provider = ChineseFragments("đạp xe")
    engine, node = _engine(tmp_path, monkeypatch, provider)
    text, gate = engine._finalize_candidate(node, "Tôi 骑车 và tôi lại 骑车 hôm nay, thật vui.")
    assert provider.calls == ["骑车"]
    assert text == "Tôi đạp xe và tôi lại đạp xe hôm nay, thật vui."
    assert gate.passed


def test_chinese_reply_to_a_fragment_stops_without_trying_the_other_fragments(tmp_path, monkeypatch):
    provider = ChineseFragments("骑自行车")
    engine, node = _engine(tmp_path, monkeypatch, provider)
    text, gate = engine._finalize_candidate(node, "Tôi 骑车 và tôi lại 开心 hôm nay, thật vui.")
    assert len(provider.calls) == 1
    assert "骑车" in text and not gate.passed

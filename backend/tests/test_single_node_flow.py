"""Worker dịch từng đoạn bằng văn bản thuần; từ tiếng Anh model không biết thì không thử lại vô ích."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.db.engine as db_engine_module
import app.services.translation.contextual_engine as contextual_engine_module
import app.services.translation.worker as worker_module
from app.db.models import Base, ChapterModel, NodeModel, ProjectModel
from app.services.translation.contextual_engine import ContextualTranslationEngine
from app.services.translation.mock_provider import MockProvider
from app.services.translation.worker import TranslationWorker


SECOND_SOURCE = "Even as the strings of a lute are alone. They quiver with the same music."


class RecordingProvider(MockProvider):
    def __init__(self, second_translation="Như những dây đàn tách biệt nhau. Chúng cùng rung lên một khúc nhạc."):
        self.batch_calls = 0
        self.single_calls = []
        self.second_translation = second_translation

    def translate(self, blocks, system_prompt, user_prompt, model=None, temperature=0.2):
        self.batch_calls += 1
        return super().translate(blocks, system_prompt, user_prompt, model, temperature)

    def translate_single(self, text, system_prompt, glossary_terms=None, model=None, temperature=0.15, user_prompt=None):
        self.single_calls.append({"text": text, "system": system_prompt, "user": user_prompt or ""})
        if text == SECOND_SOURCE:
            return self.second_translation
        return "Tôi mỉm cười vì thấy mình được thấu hiểu."


def _run(monkeypatch, tmp_path, provider=None):
    # File thay vì :memory: vì worker mở và đóng nhiều session giữa các đoạn như khi chạy thật.
    sqlite_engine = create_engine(f"sqlite:///{(tmp_path / 'project.db').as_posix()}")
    Base.metadata.create_all(sqlite_engine)
    new_session = sessionmaker(bind=sqlite_engine)
    db = new_session()
    db.add(ProjectModel(id="p", title="P", selected_model="mock", document_type="LITERATURE"))
    db.add(ChapterModel(id="c", project_id="p", title="Chương", order_index=0))
    db.add(NodeModel(id="n1", project_id="p", chapter_id="c", node_type="paragraph",
                     content="I smiled, feeling understood.", status="PENDING", order_index=0))
    db.add(NodeModel(id="n2", project_id="p", chapter_id="c", node_type="paragraph",
                     content=SECOND_SOURCE, status="PENDING", order_index=1))
    db.commit()
    db.close()
    monkeypatch.setattr(worker_module, "get_project_db", lambda requested_id: new_session())
    monkeypatch.setattr(db_engine_module, "get_project_db", lambda requested_id: new_session())
    monkeypatch.setattr(contextual_engine_module.settings, "PROJECTS_DIR", tmp_path / "projects")
    provider = provider or RecordingProvider()
    TranslationWorker().translate_project_sync("p", model_name="mock", provider=provider)
    return new_session(), provider


def test_worker_translates_one_node_at_a_time_without_json_batches(monkeypatch, tmp_path):
    db, provider = _run(monkeypatch, tmp_path)
    assert provider.batch_calls == 0
    assert db.query(NodeModel).filter_by(status="TRANSLATED").count() == 2


def test_each_paragraph_sees_the_approved_translation_right_before_it(monkeypatch, tmp_path):
    _, provider = _run(monkeypatch, tmp_path)
    second = next(call for call in provider.single_calls if call["text"] == SECOND_SOURCE)
    assert "Tôi mỉm cười vì thấy mình được thấu hiểu." in second["user"]


def test_source_comes_after_context_right_before_the_request(monkeypatch, tmp_path):
    _, provider = _run(monkeypatch, tmp_path)
    prompt = next(call for call in provider.single_calls if call["text"] == SECOND_SOURCE)["user"]
    assert prompt.index("Tôi mỉm cười") < prompt.index("CURRENT SOURCE\n" + SECOND_SOURCE) < prompt.index("Translate CURRENT SOURCE")


def test_chapter_title_is_not_right_above_the_source(monkeypatch, tmp_path):
    _, provider = _run(monkeypatch, tmp_path)
    prompt = next(call for call in provider.single_calls if call["text"] == SECOND_SOURCE)["user"]
    assert prompt.index("CURRENT CHAPTER") < prompt.index("Tôi mỉm cười") < prompt.index("CURRENT SOURCE")


def test_translation_calls_share_one_system_prompt(monkeypatch, tmp_path):
    _, provider = _run(monkeypatch, tmp_path)
    assert len({call["system"] for call in provider.single_calls}) == 1


class PatchingProvider(RecordingProvider):
    """Bản dịch sót "lute"; câu sửa do `patch` quyết định."""

    def __init__(self, patch):
        super().__init__("Như những dây đàn lute tách biệt nhau. Chúng cùng rung lên một khúc nhạc.")
        self.patch = patch

    def translate_single(self, text, system_prompt, glossary_terms=None, model=None, temperature=0.15, user_prompt=None):
        if "Câu cần sửa" in (user_prompt or ""):
            self.single_calls.append({"text": text, "system": system_prompt, "user": user_prompt})
            return self.patch
        return super().translate_single(text, system_prompt, glossary_terms, model, temperature, user_prompt)


def test_leftover_word_is_fixed_in_its_sentence_without_retranslating(monkeypatch, tmp_path):
    provider = PatchingProvider("Như những dây đàn tách biệt nhau.")
    db, _ = _run(monkeypatch, tmp_path, provider)
    node = db.query(NodeModel).filter_by(id="n2").one()
    assert node.status == "TRANSLATED"
    assert node.translated_content == "Như những dây đàn tách biệt nhau. Chúng cùng rung lên một khúc nhạc."
    patch_calls = [call for call in provider.single_calls if "Câu cần sửa" in call["user"]]
    assert len(patch_calls) == 1 and "cùng rung lên" not in patch_calls[0]["text"]
    assert not any("VALIDATION OR QA ISSUES" in call["user"] for call in provider.single_calls)


def test_word_the_model_keeps_leaving_in_goes_to_review_without_more_attempts(monkeypatch, tmp_path):
    provider = PatchingProvider("Như những dây đàn lute tách biệt nhau.")
    db, _ = _run(monkeypatch, tmp_path, provider)
    node = db.query(NodeModel).filter_by(id="n2").one()
    assert node.status == "NEEDS_REVIEW"
    assert "lute" in node.node_metadata["draft_translation"]["text"]
    assert not any("VALIDATION OR QA ISSUES" in call["user"] for call in provider.single_calls)
    assert not any(call["system"] == ContextualTranslationEngine.FOREIGN_SCRIPT_SYSTEM_PROMPT for call in provider.single_calls)


def test_patch_that_adds_a_sentence_is_rejected(monkeypatch, tmp_path):
    provider = PatchingProvider("Như những dây đàn tách biệt nhau. Chúng cùng rung lên một khúc nhạc.")
    db, _ = _run(monkeypatch, tmp_path, provider)
    assert db.query(NodeModel).filter_by(id="n2").one().status == "NEEDS_REVIEW"


def test_heading_prompt_has_no_rolling_context_and_asks_for_a_title(monkeypatch, tmp_path):
    from app.services.translation.contextual_engine import ContextualTranslationEngine as Engine

    class HeadingProvider(RecordingProvider):
        def translate_single(self, text, system_prompt, glossary_terms=None, model=None, temperature=0.15, user_prompt=None):
            self.single_calls.append({"text": text, "system": system_prompt, "user": user_prompt or ""})
            return "Em bé và nước tắm" if text == "The Baby and the Bathwater" else "Tôi mỉm cười vì thấy mình được thấu hiểu."

    sqlite_engine = create_engine(f"sqlite:///{(tmp_path / 'heading.db').as_posix()}")
    Base.metadata.create_all(sqlite_engine)
    db = sessionmaker(bind=sqlite_engine)()
    db.add(ProjectModel(id="p", title="P", selected_model="mock", document_type="LITERATURE"))
    db.add(ChapterModel(id="c", project_id="p", title="Introduction", order_index=0))
    db.add(NodeModel(id="n1", project_id="p", chapter_id="c", node_type="paragraph", content="I smiled, feeling understood.",
                     translated_content="Tôi mỉm cười vì thấy mình được thấu hiểu.", status="TRANSLATED", order_index=0))
    heading = NodeModel(id="n2", project_id="p", chapter_id="c", node_type="heading",
                        content="The Baby and the Bathwater", status="PENDING", order_index=1)
    db.add(heading)
    db.commit()
    monkeypatch.setattr(contextual_engine_module.settings, "PROJECTS_DIR", tmp_path / "projects")
    provider = HeadingProvider()
    from app.services.translation.translation_config import TranslationConfig

    project = db.query(ProjectModel).one()
    engine = Engine(db, project, provider, TranslationConfig.from_project(project, model_override="mock"), {})
    result = engine.translate_node(db.query(ChapterModel).one(), engine.canonical_node(heading))
    assert result.passed and result.translated_text == "Em bé và nước tắm"
    prompt = next(call for call in provider.single_calls if call["text"] == "The Baby and the Bathwater")["user"]
    assert "Tôi mỉm cười" not in prompt and "Introduction" not in prompt and "CURATED DOMAIN" not in prompt
    assert prompt.endswith("Dịch tiêu đề sau sang tiếng Việt. Chỉ trả về tiêu đề đã dịch, không giải thích:\nThe Baby and the Bathwater")


def test_citation_labelled_as_heading_is_translated_like_a_paragraph():
    from app.models.canonical import DocumentNode, NodeStatus, NodeType
    from app.services.translation.context_assembler import TranslationContext
    from app.services.translation.prompt_builder import PromptBuilder

    citation = "* The Portable Jung, Joseph Campbell, ed. (New York: Viking Press, 1971), pp. 511-12."
    node = DocumentNode(id="n", type=NodeType.HEADING, content=citation, status=NodeStatus.PENDING, order_index=1)
    prompt = PromptBuilder.build_single_prompt(node, TranslationContext())
    assert "CURRENT SOURCE" in prompt and "Dịch tiêu đề" not in prompt


def test_stray_chinese_is_replaced_in_place_without_retranslating_the_paragraph(monkeypatch, tmp_path):
    class StrayChinese(RecordingProvider):
        def translate_single(self, text, system_prompt, glossary_terms=None, model=None, temperature=0.15, user_prompt=None):
            if text == "成千上万的":
                self.single_calls.append({"text": text, "system": system_prompt, "user": user_prompt or ""})
                return "hàng ngàn"
            return super().translate_single(text, system_prompt, glossary_terms, model, temperature, user_prompt)

    provider = StrayChinese("Như 成千上万的 dây đàn tách biệt nhau. Chúng cùng rung lên một khúc nhạc.")
    db, _ = _run(monkeypatch, tmp_path, provider)
    node = db.query(NodeModel).filter_by(id="n2").one()
    assert node.status == "TRANSLATED"
    assert node.translated_content == "Như hàng ngàn dây đàn tách biệt nhau. Chúng cùng rung lên một khúc nhạc."
    assert not any("VALIDATION OR QA ISSUES" in call["user"] for call in provider.single_calls)

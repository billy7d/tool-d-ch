"""Nghĩa tham khảo từ từ điển cục bộ cho từ model để sót, và lời bình của model không được lọt qua gate."""
import gzip
import struct

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.db.engine as db_engine_module
import app.services.translation.contextual_engine as contextual_engine_module
import app.services.translation.worker as worker_module
from app.db.models import Base, ChapterModel, NodeModel, ProjectModel
from app.services.translation.mock_provider import MockProvider
from app.services.translation.quality_gate import TranslationQualityGate
from app.services.translation.vocabulary_hints import StarDict, VocabularyHints, _source_word_for, format_hints, senses_from_entry
from app.services.translation.worker import TranslationWorker

ENTRIES = {
    "habit": "@habit /'hæbit/\n*  danh từ\n- thói quen, tập quán\n=to be in the habit of+ có thói quen\n"
             "- (từ cổ,nghĩa cổ) áo của thầy tu\n*  ngoại động từ\n- mặc quần áo cho\n!to kick the habit\n- bỏ thói quen\n"
             "@Chuyên ngành kinh tế\n-tập tính",
    "rhizome": "@rhizome /'raizəm/\n*  danh từ\n- (thực vật học) thân rễ",
    "kaizen": "@kaizen\n@Chuyên ngành kinh tế\n-cải tiến liên tục",
    "chant": "@chant /tʃɑ:nt/\n*  danh từ\n- thánh ca\n*  động từ\n- tụng kinh",
    "minutes": "@minutes\n*  danh từ số nhiều\n- biên bản",
    "minute": "@minute /'minit/\n*  danh từ\n- phút",
    "trip": "@trip\n*  động từ\n- vấp, ngã",
    "passive": "@passive /'pæsiv/\n*  tính từ\n- thụ động",
    "unconscious": "@unconscious\n*  tính từ\n- vô ý thức",
    "push": "@push\n*  danh từ\n- sự xô",
    "lute": "@lute /lu:t/\n*  danh từ\n- (âm nhạc) đàn luýt",
}


def _write_dictionary(directory, entries=ENTRIES):
    directory.mkdir(parents=True, exist_ok=True)
    data = b""
    index = b""
    for word in sorted(entries):
        body = entries[word].encode("utf-8")
        index += word.encode("utf-8") + b"\0" + struct.pack(">II", len(data), len(body))
        data += body
    (directory / "test.idx").write_bytes(index)
    with gzip.open(directory / "test.dict.dz", "wb") as handle:
        handle.write(data)
    return directory


@pytest.fixture
def vocabulary(tmp_path):
    return VocabularyHints(StarDict(_write_dictionary(tmp_path / "dict")))


def test_only_general_senses_are_taken_not_examples_idioms_or_archaic_ones():
    assert senses_from_entry(ENTRIES["habit"]) == ["thói quen, tập quán", "mặc quần áo cho"]


def test_specialized_senses_are_used_when_nothing_else_exists():
    assert senses_from_entry(ENTRIES["kaizen"]) == ["cải tiến liên tục"]


def test_english_field_tag_senses_come_after_general_ones():
    entry = "@ceiling /'si:liŋ/\n*  danh từ\n- (Econ) Mức trần\n- trần nhà"
    assert senses_from_entry(entry) == ["trần nhà"]


def test_pronunciation_line_is_not_a_sense():
    entry = "@consecrate /'kɔnsikrit - 'kɔnsikreit/\n- 'kɔnsikreit/\n*  ngoại động từ\n- hiến dâng"
    assert senses_from_entry(entry) == ["hiến dâng"]


@pytest.mark.parametrize("word,expected", [("chanting", "tụng kinh"), ("tripped", "vấp, ngã"), ("passively", "thụ động")])
def test_inflected_forms_reach_their_base_word(vocabulary, word, expected):
    assert expected in vocabulary.senses(word)


def test_plural_with_its_own_entry_keeps_the_senses_of_its_base_word(vocabulary):
    assert vocabulary.senses("minutes") == ["biên bản", "phút"]


@pytest.mark.parametrize("residue,source,expected", [
    ("khôngscious", "to know with our unconscious mind", "unconscious"),
    ("Củasciousness", "The Evolution of Consciousness", "consciousness"),
    ("passively", "characteristic of passive dependent people", "passive"),
    ("brutale", "often brutally, out of anger", "brutally"),
    ("habit", "a pattern of clinging", "habit"),
    ("khôngscious", "the conscious mind knows what the unconscious already knows", "unconscious"),
    ("Củasciousness", "the conscious mind and the evolution of consciousness", "consciousness"),
])
def test_distorted_output_is_mapped_back_to_the_source_word(residue, source, expected):
    assert _source_word_for(residue, source) == expected


def test_fused_garbage_unrelated_to_the_source_gets_no_hint(vocabulary):
    assert vocabulary.for_words(["giườngPush"], "neither the crib nor the ceiling move") == {}


def test_fused_word_from_the_source_gets_its_meaning(vocabulary):
    assert vocabulary.for_words(["khôngscious"], "the unconscious part") == {"khôngscious": ["vô ý thức"]}


def test_missing_dictionary_gives_no_hints(tmp_path):
    assert VocabularyHints(StarDict(tmp_path / "none")).for_words(["habit"], "a habit") == {}


def test_format_lists_one_word_per_line():
    assert format_hints({"lute": ["đàn luýt"], "habit": ["thói quen", "tập quán"]}) == "- lute: đàn luýt\n- habit: thói quen; tập quán"


SOURCE = "Even as the strings of a lute are alone. They quiver with the same music."


class LeftoverProvider(MockProvider):
    """Từ hiếm: tự sửa thì model chép lại nguyên câu, có nghĩa từ điển mới thay được."""

    knows_word = False

    def __init__(self):
        self.patch_prompts = []

    def translate_single(self, text, system_prompt, glossary_terms=None, model=None, temperature=0.15, user_prompt=None):
        if "Câu cần sửa" in (user_prompt or ""):
            self.patch_prompts.append(user_prompt)
            if self.knows_word or "Nghĩa tham khảo" in user_prompt:
                return "Như những dây đàn luýt tách biệt nhau."
            return user_prompt.split("Câu cần sửa:\n", 1)[1].split("\n\n", 1)[0]
        return "Như những dây đàn lute tách biệt nhau. Chúng cùng rung lên một khúc nhạc."


class KnowingProvider(LeftoverProvider):
    knows_word = True


class ManglingProvider(LeftoverProvider):
    """Tự sửa thì làm méo từ ("passively" -> "passivamente"); có nghĩa thì sửa đúng nhưng in đậm chỗ thay."""

    def translate_single(self, text, system_prompt, glossary_terms=None, model=None, temperature=0.15, user_prompt=None):
        if "Câu cần sửa" in (user_prompt or ""):
            self.patch_prompts.append(user_prompt)
            if "Nghĩa tham khảo" in user_prompt:
                return "Như những dây **đàn luýt** tách biệt nhau."
            return "Như những dây đàn lutelòng tách biệt nhau."
        return super().translate_single(text, system_prompt, glossary_terms, model, temperature, user_prompt)


def _translate(monkeypatch, tmp_path, dictionary_dir, provider_class=LeftoverProvider):
    sqlite_engine = create_engine(f"sqlite:///{(tmp_path / 'project.db').as_posix()}")
    Base.metadata.create_all(sqlite_engine)
    new_session = sessionmaker(bind=sqlite_engine)
    db = new_session()
    db.add(ProjectModel(id="p", title="P", selected_model="mock", document_type="SELF_HELP"))
    db.add(ChapterModel(id="c", project_id="p", title="Chương", order_index=0))
    db.add(NodeModel(id="n", project_id="p", chapter_id="c", node_type="paragraph", content=SOURCE, status="PENDING", order_index=0))
    db.commit()
    db.close()
    monkeypatch.setattr(worker_module, "get_project_db", lambda requested_id: new_session())
    monkeypatch.setattr(db_engine_module, "get_project_db", lambda requested_id: new_session())
    monkeypatch.setattr(contextual_engine_module.settings, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(contextual_engine_module.settings, "DICTIONARY_DIR", dictionary_dir)
    provider = provider_class()
    TranslationWorker().translate_project_sync("p", model_name="mock", provider=provider)
    return new_session().query(NodeModel).one(), provider


def test_dictionary_meaning_is_used_only_after_the_model_could_not_fix_the_word(monkeypatch, tmp_path):
    node, provider = _translate(monkeypatch, tmp_path, _write_dictionary(tmp_path / "dict"))
    first, retry = provider.patch_prompts
    assert "Nghĩa tham khảo" not in first
    assert "Nghĩa tham khảo từ từ điển (chọn một nghĩa hợp với câu gốc):\n- lute: (âm nhạc) đàn luýt" in retry
    assert node.status == "TRANSLATED" and "đàn luýt" in node.translated_content


def test_word_the_model_fixes_by_itself_gets_no_dictionary_meaning(monkeypatch, tmp_path):
    node, provider = _translate(monkeypatch, tmp_path, _write_dictionary(tmp_path / "dict"), KnowingProvider)
    assert len(provider.patch_prompts) == 1 and "Nghĩa tham khảo" not in provider.patch_prompts[0]
    assert node.status == "TRANSLATED"


class FusedProvider(LeftoverProvider):
    def translate_single(self, text, system_prompt, glossary_terms=None, model=None, temperature=0.15, user_prompt=None):
        if "Câu cần sửa" in (user_prompt or ""):
            return super().translate_single(text, system_prompt, glossary_terms, model, temperature, user_prompt)
        return "Như những dâylute tách biệt nhau. Chúng cùng rung lên một khúc nhạc."


def test_fused_fragment_gets_the_dictionary_meaning_on_the_first_try(monkeypatch, tmp_path):
    node, provider = _translate(monkeypatch, tmp_path, _write_dictionary(tmp_path / "dict"), FusedProvider)
    assert len(provider.patch_prompts) == 1 and "- dâylute: (âm nhạc) đàn luýt" in provider.patch_prompts[0]
    assert node.status == "TRANSLATED" and "đàn luýt" in node.translated_content


def test_mangled_first_fix_is_retried_from_the_original_sentence(monkeypatch, tmp_path):
    node, provider = _translate(monkeypatch, tmp_path, _write_dictionary(tmp_path / "dict"), ManglingProvider)
    retry = provider.patch_prompts[1]
    assert "Câu cần sửa:\nNhư những dây đàn lute tách biệt nhau." in retry and "- lute: (âm nhạc) đàn luýt" in retry
    assert node.status == "TRANSLATED"
    assert "Như những dây đàn luýt tách biệt nhau." in node.translated_content and "**" not in node.translated_content


@pytest.mark.parametrize("target", ["số phận coldlòng đặt lên tôi", "cả giườngPush và trần nhà"])
def test_english_fused_onto_a_vietnamese_word_is_residue(target):
    issues = TranslationQualityGate().validate("fate is coldhearted; the crib and ceiling", target, {}).issues
    assert "ENGLISH_RESIDUE" in [issue["code"] for issue in issues]


def test_long_vietnamese_syllables_are_not_fused_words():
    target = "Chuyện khuyên nhủ nghiêng về lý thuyết truyền thống của người Nguyễn."
    issues = TranslationQualityGate().validate("Advice leans toward the traditional theory of the Nguyen.", target, {}).issues
    assert "ENGLISH_RESIDUE" not in [issue["code"] for issue in issues]


def test_patch_still_works_without_a_dictionary(monkeypatch, tmp_path):
    node, provider = _translate(monkeypatch, tmp_path, tmp_path / "no-dictionary")
    assert len(provider.patch_prompts) == 1 and "Nghĩa tham khảo" not in provider.patch_prompts[0]
    assert node.status == "NEEDS_REVIEW"


@pytest.mark.parametrize("target", [
    "Một doanh nhân thành đạt không phù hợp với ngữ cảnh, nên cần điều chỉnh thành cách diễn đạt khác.",
    "Tôi không biết. (Note: từ này được giữ nguyên.)",
    "Dưới đây là bản dịch được sửa đổi: Tôi không biết.",
    "Tôi không biết. Dịch lại thành tiếng Việt: Tôi không biết.",
])
def test_model_commentary_is_rejected(target):
    issues = TranslationQualityGate().validate("I do not know.", target, {}).issues
    assert "MODEL_COMMENTARY" in [issue["code"] for issue in issues]


def test_book_about_translation_may_mention_it():
    issues = TranslationQualityGate().validate("The translation of the original poem.", "Bản dịch của bài thơ nguyên văn.", {}).issues
    assert "MODEL_COMMENTARY" not in [issue["code"] for issue in issues]


def test_quoted_foreign_phrase_explained_by_the_source_stays_as_is():
    source = "Saint Augustine wrote, ‘‘Dilige et quod vis fac,” meaning “If you are loving, you may do whatever you want.”"
    target = "Thánh Augustin viết, “Dilige et quod vis fac,” nghĩa là “Nếu bạn yêu thương, bạn có thể làm điều bạn muốn.”"
    assert TranslationQualityGate().validate(source, target, {}).passed
